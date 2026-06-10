#!/usr/bin/env python3
"""End-to-end BGM explainer runner: resolve inputs from the DB, then run the pipeline.

Resolution — for each ITM id in --item-ids:
  * orchestrator.products      -> vertical (slug) + super_category
  * qc_review.items            -> the two auto_qc_approved explainer rows
                                  (tags {"role": "explainer"}), newest first
  * orchestrator.product_data  -> script1/script2 texts via script_product_data_id

  Writes <work-dir>/manifest.jsonl and <work-dir>/scripts/scripts.jsonl.
  An existing manifest is reused unless --re-resolve is passed.

Run — invokes run_full_batch_pipeline.py with --manifest-jsonl. The pipeline
  downloads each item's two videos from the artifacts service on demand inside
  prepare_item (no upfront bulk download); already-downloaded videos in
  <work-dir>/batch are reused. Images come from --image-tsv exactly as before.
  Any unrecognised CLI args are forwarded to the pipeline verbatim.

Example:
  python bgm_explainer_pipeline.py \
    --item-ids applicable_item_ids_both_auto_qc_approved.txt \
    --work-dir /data/experiments/expaliner_videos/bgm_explainers \
    --limit 200 --parallel 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pg8000.native

REPO_ROOT = Path(__file__).resolve().parent
PIPELINE_SCRIPT = REPO_ROOT / "run_full_batch_pipeline.py"

DB_HOST = os.environ.get("MINIVET_DB_HOST", "10.12.46.7")
DB_PORT = int(os.environ.get("MINIVET_DB_PORT", "30432"))
DB_USER = os.environ.get("MINIVET_DB_USER", "minivet_ro_user")
DB_PASSWORD = os.environ.get("MINIVET_DB_PASSWORD", "")

EXPLAINER_TAG = '{"role": "explainer"}'
APPROVED_STATUS = "auto_qc_approved"
CHUNK = 2000


def connect(database: str) -> pg8000.native.Connection:
    if not DB_PASSWORD:
        sys.exit("MINIVET_DB_PASSWORD is not set; export it before running.")
    return pg8000.native.Connection(
        user=DB_USER, host=DB_HOST, port=DB_PORT,
        database=database, password=DB_PASSWORD,
    )


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", value or "").strip()
    if not cleaned:
        return "Product"
    return "".join(part[:1].upper() + part[1:] for part in cleaned.split(" "))


def chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def run_chunked(database: str, ids: list, run_chunk) -> None:
    """Run run_chunk(conn, chunk) over ids with a fresh connection per call.

    The DB endpoint sits behind a pooler that can reset idle session state
    (pg8000 then raises "unnamed prepared statement does not exist"), so each
    phase opens its own connection and each chunk retries once on a reconnect.
    """
    conn = connect(database)
    try:
        for chunk in chunked(ids, CHUNK):
            try:
                run_chunk(conn, chunk)
            except pg8000.exceptions.DatabaseError:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = connect(database)
                run_chunk(conn, chunk)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# --- DB resolution ------------------------------------------------------------
def fetch_products(itm_ids: list[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}

    def run_chunk(conn, chunk):
        rows = conn.run(
            """
            SELECT external_id,
                   product_metadata->'category'->>'analytic_vertical',
                   product_metadata->'category'->>'analytic_super_category'
            FROM public.products
            WHERE external_id = ANY(:ids)
            """,
            ids=chunk,
        )
        for itm, vertical, sc in rows:
            out[itm] = {"vertical": vertical or "", "super_category": sc or ""}

    run_chunked("orchestrator", itm_ids, run_chunk)
    print(f"  products: {len(out)}/{len(itm_ids)} resolved", flush=True)
    return out


def fetch_approved_rows(itm_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """ITM -> approved explainer rows (newest first) with video + script ids."""
    out: dict[str, list[dict[str, Any]]] = {}

    def run_chunk(conn, chunk):
        rows = conn.run(
            """
            SELECT product_id, item_id, item_payload
            FROM public.items
            WHERE product_id = ANY(:ids)
              AND review_status = :status
              AND tags = CAST(:tag AS jsonb)
              AND review_type = 'EXPLAINER_VIDEO'
            ORDER BY product_id, item_id DESC
            """,
            ids=chunk,
            status=APPROVED_STATUS,
            tag=EXPLAINER_TAG,
        )
        for itm, _row_id, payload in rows:
            if not isinstance(payload, dict):
                continue
            meta = payload.get("metadata") or {}
            vid = payload.get("video_artifact_id")
            pdi = meta.get("script_product_data_id")
            if not vid or not pdi:
                continue
            out.setdefault(itm, []).append(
                {"video_artifact_id": vid, "script_product_data_id": str(pdi)}
            )

    run_chunked("qc_review", itm_ids, run_chunk)
    print(f"  qc rows: {len(out)} items with approved explainer rows", flush=True)
    return out


def fetch_scripts(pdis: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    def run_chunk(conn, chunk):
        rows = conn.run(
            """
            SELECT product_data_id, metadata
            FROM public.product_data
            WHERE type = 'SCRIPT'
              AND product_data_id = ANY(CAST(:ids AS uuid[]))
            """,
            ids=chunk,
        )
        for pdi, meta in rows:
            meta = meta or {}
            out[str(pdi)] = {
                "script_text": meta.get("script_text", ""),
                "script_index": meta.get("script_index"),
            }

    run_chunked("orchestrator", pdis, run_chunk)
    print(f"  scripts: {len(out)} resolved", flush=True)
    return out


def resolve(itm_ids: list[str], manifest_path: Path) -> list[dict[str, Any]]:
    print(f"Resolving {len(itm_ids)} items from DB ...", flush=True)
    products = fetch_products(itm_ids)
    approved = fetch_approved_rows(itm_ids)
    all_pdis = sorted({r["script_product_data_id"]
                       for rows in approved.values() for r in rows})
    scripts = fetch_scripts(all_pdis)

    records: list[dict[str, Any]] = []
    skipped = {"no_product": 0, "lt2_rows": 0, "no_script_pair": 0}
    for itm in itm_ids:
        product = products.get(itm)
        if not product:
            skipped["no_product"] += 1
            continue
        rows = approved.get(itm) or []
        if len(rows) < 2:
            skipped["lt2_rows"] += 1
            continue
        by_index: dict[int, dict[str, Any]] = {}
        for row in rows:  # newest first; keep first valid binding per index
            info = scripts.get(row["script_product_data_id"])
            if not info or not info.get("script_text"):
                continue
            try:
                idx = int(info.get("script_index"))
            except (TypeError, ValueError):
                continue
            if idx not in (1, 2) or idx in by_index:
                continue
            by_index[idx] = {
                "video_artifact_id": row["video_artifact_id"],
                "script_text": info["script_text"],
            }
        if 1 not in by_index or 2 not in by_index:
            skipped["no_script_pair"] += 1
            continue
        records.append({
            "item_id": itm,
            "slug": slugify(product["vertical"]),
            "super_category": product["super_category"],
            "script1": by_index[1]["script_text"],
            "script2": by_index[2]["script_text"],
            "video_artifact_script1": by_index[1]["video_artifact_id"],
            "video_artifact_script2": by_index[2]["video_artifact_id"],
        })

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Resolved {len(records)} items -> {manifest_path} (skipped: {skipped})",
          flush=True)
    return records


# --- Main -------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--item-ids", type=Path,
                        default=Path("applicable_item_ids_both_auto_qc_approved.txt"),
                        help="File with one ITM id per line.")
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="Where batch/, scripts/ and manifest.jsonl live.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Pipeline output root (default <work-dir>/outputs).")
    parser.add_argument("--num-items", type=int, default=None,
                        help="Only resolve the first N item ids.")
    parser.add_argument("--re-resolve", action="store_true",
                        help="Ignore an existing manifest.jsonl and hit the DB again.")
    parser.add_argument("--resolve-only", action="store_true",
                        help="Write manifest + scripts.jsonl and stop.")
    # pipeline inputs kept exactly as before
    parser.add_argument("--image-tsv", type=Path, default=Path("bgmh_enriched.tsv"))
    parser.add_argument("--style", type=Path, default=Path("scripts/global_style.json"))
    parser.add_argument("--timeline-config", type=Path,
                        default=Path("scripts/timeline_generation_config.json"))
    args, pipeline_extra = parser.parse_known_args()

    itm_ids = [line.strip() for line in args.item_ids.read_text().splitlines()
               if line.strip()]
    if args.num_items:
        itm_ids = itm_ids[:args.num_items]

    batch_dir = args.work_dir / "batch"
    scripts_dir = args.work_dir / "scripts"
    batch_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.work_dir / "manifest.jsonl"

    if manifest_path.exists() and not args.re_resolve:
        records = [json.loads(line) for line in
                   manifest_path.read_text(encoding="utf-8").splitlines() if line]
        print(f"Reusing manifest: {len(records)} records from {manifest_path}",
              flush=True)
        if args.num_items:
            records = records[:args.num_items]
    else:
        records = resolve(itm_ids, manifest_path)

    scripts_path = scripts_dir / "scripts.jsonl"
    with scripts_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps({
                "item_id": rec["item_id"],
                "super_category": rec["super_category"],
                "script1": rec["script1"],
                "script2": rec["script2"],
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} items -> {scripts_path}", flush=True)

    if args.resolve_only:
        return 0

    # Run the pipeline; it downloads each item's videos on demand via the manifest.
    out_dir = args.out_dir or (args.work_dir / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(PIPELINE_SCRIPT),
        "--batch-dir", str(batch_dir),
        "--json-dir", str(scripts_dir),
        "--manifest-jsonl", str(manifest_path),
        "--image-tsv", str(args.image_tsv),
        "--style", str(args.style),
        "--timeline-config", str(args.timeline_config),
        "--ass-dir", str(out_dir / "ass"),
        "--trimmed-video-dir", str(out_dir / "trimmed"),
        "--out-timeline-dir", str(out_dir / "timelines"),
        "--out-video-dir", str(out_dir / "videos"),
        "--report-csv", str(out_dir / "report.csv"),
        "--report-json", str(out_dir / "report.json"),
        *pipeline_extra,
    ]
    print("\nRunning pipeline:\n  " + " \\\n  ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
