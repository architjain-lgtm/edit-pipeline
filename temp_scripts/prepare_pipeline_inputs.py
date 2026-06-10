#!/usr/bin/env python3
"""Prepare run_full_batch_pipeline.py inputs for N items per BGM super_category.

For each BGM ``analytic_super_category`` this script:
  1. finds products in that super_category (orchestrator.products),
  2. keeps only items whose two EXPLAINER rows in qc_review.items are BOTH
     ``auto_qc_approved`` and tagged ``{"role": "explainer"}``,
  3. pulls the two script texts (script1/script2) from orchestrator.product_data
     via each row's ``script_product_data_id``,
  4. pulls the product's image artifact ids from orchestrator.product_images
     (joined on products.product_id) — items with no images are skipped,
  5. downloads the two videos and all images from the artifacts service.

Outputs (under --out-dir, default temp_scripts/pipeline_inputs):
  batch/<ITM>_<slug>_script{1,2}_<Slug>.mp4   -> feeds --batch-dir
  scripts/scripts.jsonl  ({item_id, script1, script2, ...})  -> feeds --json-dir
  images/<ITM>/img_<order>.jpg                 -> downloaded product images
  images/images.tsv  (item_id, image_paths)    -> feeds --image-tsv

Example downstream call (printed at the end of a run):
  python code/run_full_batch_pipeline.py \
    --batch-dir temp_scripts/pipeline_inputs/batch \
    --json-dir  temp_scripts/pipeline_inputs/scripts \
    --image-tsv temp_scripts/pipeline_inputs/images/images.tsv \
    --style code/global_style.json \
    --timeline-config code/timeline_generation_config.json \
    --ass-dir outputs/ass --trimmed-video-dir outputs/trimmed \
    --out-timeline-dir outputs/timelines --out-video-dir outputs/videos \
    --report-csv outputs/report.csv --report-json outputs/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pg8000.native

# --- Connection -------------------------------------------------------------
DB_HOST = os.environ.get("MINIVET_DB_HOST", "10.12.46.7")
DB_PORT = int(os.environ.get("MINIVET_DB_PORT", "30432"))
DB_USER = os.environ.get("MINIVET_DB_USER", "minivet_ro_user")
DB_PASSWORD = os.environ.get("MINIVET_DB_PASSWORD", "")

ARTIFACT_URL = "http://10.12.46.7:30080/artifacts/{artifact_id}/file?tenant_id=Flipkart"

BUSINESS_UNIT = "BGM"
EXPLAINER_TAG = {"role": "explainer"}
APPROVED_STATUS = "auto_qc_approved"


def connect(database: str) -> pg8000.native.Connection:
    if not DB_PASSWORD:
        sys.exit("MINIVET_DB_PASSWORD is not set; export it before running.")
    return pg8000.native.Connection(
        user=DB_USER,
        host=DB_HOST,
        port=DB_PORT,
        database=database,
        password=DB_PASSWORD,
    )


def slugify(value: str) -> str:
    """Turn a vertical/category into a CamelCase-ish slug usable in filenames."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", value or "").strip()
    if not cleaned:
        return "Product"
    return "".join(part[:1].upper() + part[1:] for part in cleaned.split(" "))


# --- Step 1: super categories ----------------------------------------------
def fetch_super_categories(orc: pg8000.native.Connection) -> list[str]:
    rows = orc.run(
        """
        SELECT DISTINCT product_metadata->'category'->>'analytic_super_category' AS super_category
        FROM public.products
        WHERE product_metadata->'category'->>'analytic_business_unit' = :bu
          AND product_metadata->'category'->>'analytic_super_category' IS NOT NULL
        ORDER BY 1
        """,
        bu=BUSINESS_UNIT,
    )
    return [r[0] for r in rows]


def fetch_candidate_products(
    orc: pg8000.native.Connection, super_category: str, scan_limit: int
) -> list[dict[str, str]]:
    """Return candidate products (external_id=ITM, product_id=UUID, vertical) for a super_category."""
    rows = orc.run(
        """
        SELECT external_id,
               product_id,
               product_metadata->'category'->>'analytic_vertical' AS vertical
        FROM public.products
        WHERE product_metadata->'category'->>'analytic_business_unit' = :bu
          AND product_metadata->'category'->>'analytic_super_category' = :sc
          AND external_id LIKE 'ITM%'
        ORDER BY product_id
        LIMIT :lim
        """,
        bu=BUSINESS_UNIT,
        sc=super_category,
        lim=scan_limit,
    )
    return [{"item_id": r[0], "product_uuid": r[1], "vertical": r[2] or ""} for r in rows]


# --- Step 2: qc_review explainer rows --------------------------------------
def fetch_approved_explainer_rows(
    qc: pg8000.native.Connection, itm_id: str
) -> list[dict[str, Any]]:
    """Approved explainer rows for an ITM, newest first.

    tags / item_payload are jsonb so pg8000 hands them back as dicts.
    """
    rows = qc.run(
        """
        SELECT item_id, review_status, tags, item_payload
        FROM public.items
        WHERE product_id = :p
          AND review_status = :status
          AND tags = :tag
          AND review_type = 'EXPLAINER_VIDEO'
        ORDER BY item_id DESC
        """,
        p=itm_id,
        status=APPROVED_STATUS,
        tag=json.dumps(EXPLAINER_TAG),
    )
    out: list[dict[str, Any]] = []
    for _row_id, status, tags, payload in rows:
        if not isinstance(payload, dict):
            continue
        meta = payload.get("metadata") or {}
        video_artifact_id = payload.get("video_artifact_id")
        script_pdi = meta.get("script_product_data_id")
        if not video_artifact_id or not script_pdi:
            continue
        out.append(
            {
                "video_artifact_id": video_artifact_id,
                "script_product_data_id": script_pdi,
                "duration_in_sec": meta.get("duration_in_sec"),
            }
        )
    return out


# --- Step 3: product images ---------------------------------------------------
def fetch_product_images(
    orc: pg8000.native.Connection, product_uuid: str
) -> list[dict[str, Any]]:
    """Image artifact ids for a product, ordered by image_order."""
    rows = orc.run(
        """
        SELECT pi.artifact_id,
               COALESCE((pi.image_metadata->>'image_order')::int, 0) AS image_order,
               pi.image_metadata->>'source_image_url' AS source_image_url
        FROM public.product_images pi
        WHERE pi.product_id = :p
        ORDER BY 2, 1
        """,
        p=product_uuid,
    )
    return [
        {"artifact_id": r[0], "image_order": r[1], "source_image_url": r[2] or ""}
        for r in rows
        if r[0]
    ]


# --- Step 4: script texts ---------------------------------------------------
def fetch_scripts(
    orc: pg8000.native.Connection, product_data_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Map script_product_data_id -> {script_text, script_index}."""
    if not product_data_ids:
        return {}
    placeholders = ", ".join(f":id{i}" for i in range(len(product_data_ids)))
    params = {f"id{i}": pdi for i, pdi in enumerate(product_data_ids)}
    rows = orc.run(
        f"""
        SELECT product_data_id, metadata
        FROM public.product_data
        WHERE type = 'SCRIPT'
          AND product_data_id IN ({placeholders})
        """,
        **params,
    )
    result: dict[str, dict[str, Any]] = {}
    for pdi, meta in rows:
        meta = meta or {}
        result[str(pdi)] = {
            "script_text": meta.get("script_text", ""),
            "script_index": meta.get("script_index"),
        }
    return result


# --- Selection per item -----------------------------------------------------
def build_item_record(
    qc: pg8000.native.Connection,
    orc: pg8000.native.Connection,
    product: dict[str, str],
) -> dict[str, Any] | None:
    """Return a fully-resolved item (2 scripts + 2 videos + images) or None if incomplete."""
    itm = product["item_id"]
    rows = fetch_approved_explainer_rows(qc, itm)
    if len(rows) < 2:
        return None

    images = fetch_product_images(orc, product["product_uuid"])
    if not images:
        return None  # would fail downstream with "zero images"

    scripts = fetch_scripts(orc, [r["script_product_data_id"] for r in rows])

    # Bind each approved row to its script_index (1 or 2) using product_data.
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        info = scripts.get(row["script_product_data_id"])
        if not info or not info.get("script_text"):
            continue
        idx = info.get("script_index")
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            continue
        if idx not in (1, 2) or idx in by_index:
            continue
        by_index[idx] = {
            "video_artifact_id": row["video_artifact_id"],
            "script_text": info["script_text"],
        }

    if 1 not in by_index or 2 not in by_index:
        return None

    return {
        "item_id": itm,
        "product_uuid": product["product_uuid"],
        "slug": slugify(product["vertical"]),
        "script1": by_index[1]["script_text"],
        "script2": by_index[2]["script_text"],
        "video_artifact_script1": by_index[1]["video_artifact_id"],
        "video_artifact_script2": by_index[2]["video_artifact_id"],
        "images": images,
    }


# --- Step 5: download artifacts (videos + images) ----------------------------
def download_artifact(artifact_id: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    url = ARTIFACT_URL.format(artifact_id=artifact_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
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


def image_filename(image: dict[str, Any], position: int) -> str:
    suffix = Path(urllib.parse.urlparse(image["source_image_url"]).path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    order = image["image_order"] or position
    return f"img_{order:02d}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=10,
                        help="Number of valid items to collect per super_category.")
    parser.add_argument("--scan-limit", type=int, default=400,
                        help="Max products to scan per super_category while looking for valid items.")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("temp_scripts/pipeline_inputs"))
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--super-category", action="append", default=None,
                        help="Restrict to specific super_category(ies); repeatable.")
    parser.add_argument("--no-download", action="store_true",
                        help="Resolve items and write scripts.jsonl/images.tsv but skip downloads.")
    args = parser.parse_args()

    batch_dir = args.out_dir / "batch"
    scripts_dir = args.out_dir / "scripts"
    images_dir = args.out_dir / "images"
    batch_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    orc = connect("orchestrator")
    qc = connect("qc_review")

    try:
        super_categories = args.super_category or fetch_super_categories(orc)
        print(f"Super categories ({BUSINESS_UNIT}): {super_categories}", flush=True)

        selected: list[dict[str, Any]] = []
        per_category_counts: dict[str, int] = {}

        for sc in super_categories:
            candidates = fetch_candidate_products(orc, sc, args.scan_limit)
            found = 0
            for product in candidates:
                if found >= args.per_category:
                    break
                try:
                    record = build_item_record(qc, orc, product)
                except Exception as exc:  # keep going; one bad product shouldn't abort
                    print(f"  [{sc}] {product['item_id']} error: {exc}", flush=True)
                    continue
                if record is None:
                    continue
                record["super_category"] = sc
                selected.append(record)
                found += 1
                print(f"  [{sc}] {found}/{args.per_category} {record['item_id']} ({record['slug']})",
                      flush=True)
            per_category_counts[sc] = found
            if found < args.per_category:
                print(f"  [{sc}] WARNING: only {found}/{args.per_category} valid items "
                      f"in first {len(candidates)} products (raise --scan-limit).", flush=True)
    finally:
        orc.close()
        qc.close()

    # Write scripts.jsonl
    scripts_path = scripts_dir / "scripts.jsonl"
    with scripts_path.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps({
                "item_id": item["item_id"],
                "super_category": item["super_category"],
                "script1": item["script1"],
                "script2": item["script2"],
            }, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(selected)} items -> {scripts_path}", flush=True)

    # Write images.tsv (item_id -> pipe-delimited local paths, fed to --image-tsv)
    image_paths: dict[str, list[Path]] = {}
    for item in selected:
        item_dir = images_dir / item["item_id"]
        image_paths[item["item_id"]] = [
            item_dir / image_filename(img, pos)
            for pos, img in enumerate(item["images"], start=1)
        ]
    images_tsv = images_dir / "images.tsv"
    with images_tsv.open("w", encoding="utf-8") as handle:
        handle.write("item_id\timage_paths\n")
        for item in selected:
            paths = "|".join(str(p.resolve()) for p in image_paths[item["item_id"]])
            handle.write(f"{item['item_id']}\t{paths}\n")
    print(f"Wrote image index -> {images_tsv}", flush=True)

    # Download videos (2 per item) + images
    if not args.no_download:
        jobs: list[tuple[str, Path]] = []
        n_videos = 0
        for item in selected:
            for idx in (1, 2):
                dest = batch_dir / video_filename(item, idx)
                jobs.append((item[f"video_artifact_script{idx}"], dest))
                n_videos += 1
            for img, dest in zip(item["images"], image_paths[item["item_id"]]):
                jobs.append((img["artifact_id"], dest))

        print(f"Downloading {n_videos} videos -> {batch_dir} and "
              f"{len(jobs) - n_videos} images -> {images_dir}", flush=True)
        failures = 0
        with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
            futures = {pool.submit(download_artifact, aid, dest): dest for aid, dest in jobs}
            for fut in as_completed(futures):
                dest = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    failures += 1
                    print(f"  FAIL {dest.name}: {exc}", flush=True)
        print(f"Downloaded {len(jobs) - failures}/{len(jobs)} artifacts "
              f"({failures} failures).", flush=True)

    # Summary
    print("\nPer-category valid items:")
    for sc, n in per_category_counts.items():
        print(f"  {sc}: {n}")

    print("\nDownstream command:")
    print(f"""  python code/run_full_batch_pipeline.py \\
    --batch-dir {batch_dir} \\
    --json-dir  {scripts_dir} \\
    --image-tsv {images_tsv} \\
    --style code/global_style.json \\
    --timeline-config code/timeline_generation_config.json \\
    --ass-dir outputs/ass --trimmed-video-dir outputs/trimmed \\
    --out-timeline-dir outputs/timelines --out-video-dir outputs/videos \\
    --report-csv outputs/report.csv --report-json outputs/report.json""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
