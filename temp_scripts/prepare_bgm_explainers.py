#!/usr/bin/env python3
"""Prepare batch/ + scripts/ pipeline inputs for an explicit list of ITM ids.

Set-based variant of prepare_pipeline_inputs.py for large runs (~65k items):
instead of per-item queries it resolves everything in chunked ANY() queries,
then downloads the two explainer videos per item with a thread pool.

Phases (resumable — each phase skips work already on disk):
  1. resolve: products -> approved explainer rows -> script texts
     writes <out-dir>/manifest.jsonl  (one fully-resolved record per item)
  2. write   <out-dir>/scripts/scripts.jsonl
  3. download videos -> <out-dir>/batch/   (skips existing non-empty files)

Usage:
  python temp_scripts/prepare_bgm_explainers.py \
    --item-ids applicable_item_ids_both_auto_qc_approved.txt \
    --out-dir /data/experiments/expaliner_videos/bgm_explainers
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pg8000.native

DB_HOST = os.environ.get("MINIVET_DB_HOST", "10.12.46.7")
DB_PORT = int(os.environ.get("MINIVET_DB_PORT", "30432"))
DB_USER = os.environ.get("MINIVET_DB_USER", "minivet_ro_user")
DB_PASSWORD = os.environ.get("MINIVET_DB_PASSWORD", "")

ARTIFACT_URL = "http://10.12.46.7:30080/artifacts/{artifact_id}/file?tenant_id=Flipkart"

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


# --- Phase 1: resolve ---------------------------------------------------------
def fetch_products(orc, itm_ids: list[str]) -> dict[str, dict[str, str]]:
    """ITM -> {product_uuid, vertical, super_category}."""
    out: dict[str, dict[str, str]] = {}
    for n, chunk in enumerate(chunked(itm_ids, CHUNK), 1):
        rows = orc.run(
            """
            SELECT external_id,
                   product_id,
                   product_metadata->'category'->>'analytic_vertical',
                   product_metadata->'category'->>'analytic_super_category'
            FROM public.products
            WHERE external_id = ANY(:ids)
            """,
            ids=chunk,
        )
        for itm, uuid, vertical, sc in rows:
            out[itm] = {
                "product_uuid": str(uuid),
                "vertical": vertical or "",
                "super_category": sc or "",
            }
        print(f"  products: chunk {n}, {len(out)} resolved", flush=True)
    return out


def fetch_approved_rows(qc, itm_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """ITM -> approved explainer rows (newest first), with artifact + script pdi."""
    out: dict[str, list[dict[str, Any]]] = {}
    for n, chunk in enumerate(chunked(itm_ids, CHUNK), 1):
        rows = qc.run(
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
        print(f"  qc rows: chunk {n}, {len(out)} items with rows", flush=True)
    return out


def fetch_scripts(orc, pdis: list[str]) -> dict[str, dict[str, Any]]:
    """script_product_data_id -> {script_text, script_index}."""
    out: dict[str, dict[str, Any]] = {}
    for n, chunk in enumerate(chunked(pdis, CHUNK), 1):
        rows = orc.run(
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
        print(f"  scripts: chunk {n}, {len(out)} resolved", flush=True)
    return out


def resolve(itm_ids: list[str], manifest_path: Path) -> list[dict[str, Any]]:
    orc = connect("orchestrator")
    qc = connect("qc_review")
    try:
        print(f"Resolving {len(itm_ids)} items ...", flush=True)
        products = fetch_products(orc, itm_ids)
        approved = fetch_approved_rows(qc, itm_ids)
        all_pdis = sorted({r["script_product_data_id"]
                           for rows in approved.values() for r in rows})
        scripts = fetch_scripts(orc, all_pdis)
    finally:
        orc.close()
        qc.close()

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


# --- Phase 3: download ----------------------------------------------------------
def download_artifact(artifact_id: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    url = ARTIFACT_URL.format(artifact_id=artifact_id)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} for {url}")
        with tmp.open("wb") as handle:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                handle.write(chunk)
    tmp.replace(dest)


def video_filename(item: dict[str, Any], script_index: int) -> str:
    # Must satisfy generate_batch_timelines.VIDEO_RE:
    #   ^(ITM[A-Za-z0-9]+).*_script([12])_([^./]+)\.(?:mp4|mov)$
    return f"{item['item_id']}_{item['slug']}_script{script_index}_{item['slug']}.mp4"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item-ids", type=Path, required=True,
                        help="File with one ITM id per line.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N item ids (for smoke tests).")
    args = parser.parse_args()

    itm_ids = [line.strip() for line in args.item_ids.read_text().splitlines()
               if line.strip()]
    if args.limit:
        itm_ids = itm_ids[:args.limit]

    batch_dir = args.out_dir / "batch"
    scripts_dir = args.out_dir / "scripts"
    batch_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"

    if manifest_path.exists() and not args.limit:
        records = [json.loads(line) for line in
                   manifest_path.read_text(encoding="utf-8").splitlines() if line]
        print(f"Reusing manifest: {len(records)} records from {manifest_path}",
              flush=True)
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

    if args.no_download:
        return 0

    jobs = [(rec[f"video_artifact_script{idx}"], batch_dir / video_filename(rec, idx))
            for rec in records for idx in (1, 2)]
    pending = [(aid, dest) for aid, dest in jobs
               if not (dest.exists() and dest.stat().st_size > 0)]
    print(f"Downloading {len(pending)}/{len(jobs)} videos -> {batch_dir}", flush=True)

    failures = 0
    done = 0
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        futures = {pool.submit(download_artifact, aid, dest): dest
                   for aid, dest in pending}
        for fut in as_completed(futures):
            dest = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                failures += 1
                print(f"  FAIL {dest.name}: {exc}", flush=True)
            done += 1
            if done % 1000 == 0:
                print(f"  progress: {done}/{len(pending)} ({failures} failures)",
                      flush=True)
    print(f"Downloaded {len(pending) - failures}/{len(pending)} videos "
          f"({failures} failures).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
