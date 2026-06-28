#!/usr/bin/env python3
"""End-to-end BGM explainer runner: resolve inputs from the DB, then run the pipeline.

Resolution — for each ITM id in --item-ids:
  * orchestrator.products      -> vertical (slug) + super_category
  * orchestrator.product_data  -> the two explainer videos and their scripts:
                                  VIDEO rows (role='explainer', download_url) bound
                                  to SCRIPT rows (script_index 1/2) by job_id.

  Writes <work-dir>/manifest.jsonl and <work-dir>/scripts/scripts.jsonl.
  An existing manifest is reused unless --re-resolve is passed.

Run — invokes run_full_batch_pipeline.py with --manifest-jsonl. The pipeline
  downloads each item's two videos from the product_data download_url on demand
  inside prepare_item (no upfront bulk download); already-downloaded videos in
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

ARTIFACT_URL = "http://10.12.46.7:30080/artifacts/{artifact_id}/file?tenant_id=Flipkart"

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


def fix_mojibake(value: str) -> str:
    """Repair UTF-8 text that was stored as latin-1 (e.g. 'â€“' -> '–')."""
    try:
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def attributes_from_description(description: str) -> dict[str, str]:
    """products.description for explainer products is 'key: value | key: value'."""
    out: dict[str, str] = {}
    for part in (description or "").split(" | "):
        if ": " in part:
            key, value = part.split(": ", 1)
            key, value = key.strip(), fix_mojibake(value.strip())
            if key and value:
                out[key] = value
    return out


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
def fetch_products(itm_ids: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    def run_chunk(conn, chunk):
        rows = conn.run(
            """
            SELECT external_id,
                   product_id,
                   title,
                   description,
                   product_metadata->'category'->>'analytic_vertical',
                   product_metadata->'category'->>'analytic_super_category',
                   product_metadata->'attributes'
            FROM public.products
            WHERE external_id = ANY(:ids)
            """,
            ids=chunk,
        )
        for itm, uuid, title, description, vertical, sc, pm_attrs in rows:
            if pm_attrs and isinstance(pm_attrs, dict):
                attributes = {k: str(v) for k, v in pm_attrs.items() if v}
            else:
                attributes = attributes_from_description(description or "")
            out[itm] = {
                "product_uuid": str(uuid),
                "title": fix_mojibake((title or "").strip()),
                "attributes": attributes,
                "vertical": vertical or "",
                "super_category": sc or "",
            }

    run_chunked("orchestrator", itm_ids, run_chunk)
    print(f"  products: {len(out)}/{len(itm_ids)} resolved", flush=True)
    return out


def fetch_product_images(products: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """ITM -> ordered artifact download URLs from orchestrator.product_images."""
    uuid_to_itm = {p["product_uuid"]: itm for itm, p in products.items()}
    collected: dict[str, list[tuple[int, str]]] = {}

    def run_chunk(conn, chunk):
        rows = conn.run(
            """
            SELECT product_id,
                   artifact_id,
                   COALESCE((image_metadata->>'image_order')::int, 0)
            FROM public.product_images
            WHERE product_id = ANY(CAST(:ids AS uuid[]))
            """,
            ids=chunk,
        )
        for uuid, artifact_id, order in rows:
            itm = uuid_to_itm.get(str(uuid))
            if itm and artifact_id:
                collected.setdefault(itm, []).append((order, artifact_id))

    run_chunked("orchestrator", sorted(uuid_to_itm), run_chunk)
    out = {
        itm: [ARTIFACT_URL.format(artifact_id=aid) for _order, aid in sorted(pairs)]
        for itm, pairs in collected.items()
    }
    print(f"  images: {len(out)} items with product_images", flush=True)
    return out


def fetch_video_pairs(
    products: dict[str, dict[str, Any]]
) -> dict[str, dict[int, dict[str, Any]]]:
    """ITM -> {script_index: {script_text, video_url}} from orchestrator.product_data.

    For each product UUID we read its VIDEO and SCRIPT rows. A SCRIPT row carries
    script_index (1/2) + script_text + job_id; an explainer VIDEO row carries a
    download_url + job_id. They bind by matching job_id, so each script index gets
    the download_url of the video produced from that script.
    """
    uuid_to_itm = {p["product_uuid"]: itm for itm, p in products.items()}
    # itm -> {job_id: download_url} (explainer videos) and pending script rows
    video_by_job: dict[str, dict[str, str]] = {}
    out: dict[str, dict[int, dict[str, Any]]] = {}
    pending_scripts: dict[str, dict[int, dict[str, Any]]] = {}

    def run_chunk(conn, chunk):
        rows = conn.run(
            """
            SELECT product_id, type, metadata
            FROM public.product_data
            WHERE product_id = ANY(CAST(:ids AS uuid[]))
              AND type IN ('VIDEO', 'SCRIPT')
            """,
            ids=chunk,
        )
        for puid, typ, meta in rows:
            itm = uuid_to_itm.get(str(puid))
            if not itm or not isinstance(meta, dict):
                continue
            if typ == "VIDEO":
                if (meta.get("role") == "explainer" and meta.get("download_url")
                        and meta.get("job_id")):
                    video_by_job.setdefault(itm, {})[meta["job_id"]] = meta["download_url"]
            elif typ == "SCRIPT":
                jid, text = meta.get("job_id"), meta.get("script_text")
                try:
                    idx = int(meta.get("script_index"))
                except (TypeError, ValueError):
                    continue
                if idx in (1, 2) and jid and text:
                    # keep first seen per index
                    pending_scripts.setdefault(itm, {}).setdefault(
                        idx, {"job_id": jid, "script_text": text})

    run_chunked("orchestrator", sorted(uuid_to_itm), run_chunk)

    for itm, by_index in pending_scripts.items():
        jobs = video_by_job.get(itm, {})
        bound: dict[int, dict[str, Any]] = {}
        for idx, info in by_index.items():
            url = jobs.get(info["job_id"])
            if url:
                bound[idx] = {"script_text": info["script_text"], "video_url": url}
        if 1 in bound and 2 in bound:
            out[itm] = bound

    print(f"  video pairs: {len(out)} items with both script videos", flush=True)
    return out


def resolve(itm_ids: list[str], manifest_path: Path) -> list[dict[str, Any]]:
    print(f"Resolving {len(itm_ids)} items from DB ...", flush=True)
    products = fetch_products(itm_ids)
    images = fetch_product_images(products)
    video_pairs = fetch_video_pairs(products)

    records: list[dict[str, Any]] = []
    skipped = {"no_product": 0, "no_images": 0, "no_video_pair": 0}
    for itm in itm_ids:
        product = products.get(itm)
        if not product:
            skipped["no_product"] += 1
            continue
        image_urls = images.get(itm) or []
        if not image_urls:
            skipped["no_images"] += 1
            continue
        by_index = video_pairs.get(itm)
        if not by_index or 1 not in by_index or 2 not in by_index:
            skipped["no_video_pair"] += 1
            continue
        records.append({
            "item_id": itm,
            "slug": slugify(product["vertical"]),
            "super_category": product["super_category"],
            "title": product["title"],
            "attributes": product["attributes"],
            "image_urls": image_urls,
            "script1": by_index[1]["script_text"],
            "script2": by_index[2]["script_text"],
            "video_url_script1": by_index[1]["video_url"],
            "video_url_script2": by_index[2]["video_url"],
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
    parser.add_argument("--image-tsv", type=Path, default=None,
                        help="Optional fallback image TSV; by default images come "
                             "from product_images via the manifest.")
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
        *(["--image-tsv", str(args.image_tsv)] if args.image_tsv else []),
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
