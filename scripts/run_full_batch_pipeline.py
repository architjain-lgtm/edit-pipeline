#!/usr/bin/env python3
"""Run the full paired-video batch pipeline for a bounded ITM set.

The runner intentionally keeps the large image TSV on a streaming path: it first
selects the batch ITMs to process, then reads only matching TSV rows and keeps
only matching local image files or cached HTTP(S) images for those ITMs.
"""

from __future__ import annotations

import argparse
import hashlib
import mimetypes
import csv
import json
import os
import pickle
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR / "code"
if CODE_DIR.exists():
    sys.path.insert(0, str(CODE_DIR))

from generate_batch_timelines import (  # noqa: E402
    BatchVideo,
    VIDEO_RE,
    build_scenes_from_config,
    load_timeline_config,
    product_payload,
    repo_relative,
    timeline_relative,
)
from render_with_clean_ass import (  # noqa: E402
    CLEAN_ASS_SUBTITLE_FONT,
    CLEAN_ASS_SUBTITLE_FONT_SIZE,
    compare_against_record,
    repair_clean_timeline_media_paths,
    timeline_relative as clean_timeline_relative,
    write_clean_ass,
)
from verify_ass_against_script import (  # noqa: E402
    ID_KEYS,
    VerificationError,
    extract_reference_text,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_PICKER_URL = "http://10.12.46.8:8084/image_picker_triplet"


def raise_csv_field_limit() -> None:
    """Allow very large TSV cells, such as long pipe-delimited image URL lists."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


REPORT_FIELDS = [
    "ITM",
    "status",
    "failure_reason",
    "video_path_script1",
    "video_path_script2",
    "trimmed_video_path_script1",
    "trimmed_video_path_script2",
    "ass_path_script1",
    "ass_path_script2",
    "clean_ass_path_script1",
    "clean_ass_path_script2",
    "images_found_count",
    "images_found_paths",
    "timeline_json_path",
    "clean_timeline_json_path",
    "output_video_path",
]


@dataclass(frozen=True)
class DiscoveredItem:
    item_id: str
    pair: dict[int, BatchVideo]
    first_seen: int


@dataclass
class ScriptRecord:
    record: dict[str, Any] | None = None
    json_file: Path | None = None
    json_path: str = ""
    duplicate_count: int = 0
    error: str = ""


@dataclass
class TsvItemData:
    images: list[Path]
    record: dict[str, Any]
    raw_images: list[str] = field(default_factory=list)


@dataclass
class PipelineContext:
    args: argparse.Namespace
    code_dir: Path
    timeline_config: dict[str, Any]
    timeline_defaults: dict[str, Any]
    selected_ids: list[str]
    script_records: dict[str, ScriptRecord]
    tsv_data: dict[str, TsvItemData]
    print_lock: threading.Lock


def repo_path(path: Path | str | None) -> str:
    if path is None:
        return ""
    return repo_relative(Path(path))


def resolve_existing_path(value: str | Path, *, search_code_dir: bool = False) -> Path:
    path = Path(value)
    candidates = [path if path.is_absolute() else Path.cwd() / path]
    if not path.is_absolute():
        candidates.append(SCRIPT_DIR / path)
        if search_code_dir:
            candidates.append(CODE_DIR / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_output_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (Path.cwd() / path)


def parse_itm_list(value: str | None) -> set[str] | None:
    if not value:
        return None
    # A comma-separated list of ITM codes cannot be a file path — skip the
    # path resolution to avoid OSError: [Errno 36] File name too long.
    is_inline_list = "," in value or (value.startswith("ITM") and " " not in value.strip())
    path = None if is_inline_list else resolve_existing_path(value)
    if path is not None and path.exists() and path.is_file():
        raw_items: list[str] = []
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "item_id" not in reader.fieldnames:
                    raise ValueError(
                        f"CSV must have an 'item_id' column. Got header: {reader.fieldnames} in {path}"
                    )
                for row in reader:
                    item_id = str(row.get("item_id", "")).strip()
                    if item_id.startswith("ITM"):
                        raw_items.append(item_id)
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                raw_items.extend(
                    part.strip()
                    for part in re.split(r"[, \t]+", line)
                    if part.strip().startswith("ITM")
                )
    else:
        raw_items = [
            part.strip()
            for part in value.split(",")
            if part.strip().startswith("ITM")
        ]
    return set(dict.fromkeys(raw_items))


def discover_batch_items(batch_dir: Path) -> tuple[list[DiscoveredItem], dict[str, dict[int, BatchVideo]]]:
    pairs: dict[str, dict[int, BatchVideo]] = {}
    first_seen: dict[str, int] = {}
    for index, path in enumerate(sorted(batch_dir.iterdir())):
        if not path.is_file():
            continue
        match = VIDEO_RE.match(path.name)
        if match is None:
            continue
        item_id = match.group(1)
        script_index = int(match.group(2))
        product_slug = match.group(3)
        first_seen.setdefault(item_id, index)
        pairs.setdefault(item_id, {})[script_index] = BatchVideo(item_id, script_index, product_slug, path)

    complete = [
        DiscoveredItem(item_id=item_id, pair=pair, first_seen=first_seen[item_id])
        for item_id, pair in pairs.items()
        if 1 in pair and 2 in pair
    ]
    complete.sort(key=lambda item: item.first_seen)
    return complete, pairs


def select_items(
    complete: list[DiscoveredItem],
    all_pairs: dict[str, dict[int, BatchVideo]],
    allowlist: set[str] | None,
    limit: int,
) -> tuple[list[DiscoveredItem], list[dict[str, Any]]]:
    selected: list[DiscoveredItem] = []
    missing_rows: list[dict[str, Any]] = []
    allowed = allowlist

    for item in complete:
        if allowed is not None and item.item_id not in allowed:
            continue
        selected.append(item)
        if limit and len(selected) >= limit:
            break

    if allowed is not None:
        selected_ids = {item.item_id for item in selected}
        for item_id in sorted(allowed):
            if item_id in selected_ids:
                continue
            pair = all_pairs.get(item_id)
            if pair is None or 1 not in pair or 2 not in pair:
                missing_rows.append(base_report_row(item_id, status="FAIL", failure_reason="missing pair"))
    return selected, missing_rows


def auto_detect_itm_column(fieldnames: list[str]) -> str:
    exact = ["item_id", "itm_id", "ITM_ID", "ITM", "itm", "id"]
    lowered = {name.lower(): name for name in fieldnames}
    for key in exact:
        if key.lower() in lowered:
            return lowered[key.lower()]
    for name in fieldnames:
        normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        if normalized in {"item_id", "itm_id", "itm"}:
            return name
    raise ValueError(f"Could not auto-detect ITM column from TSV header: {fieldnames}")


def auto_detect_image_columns(fieldnames: list[str]) -> list[str]:
    preferred: list[str] = []
    fallback: list[str] = []
    for name in fieldnames:
        normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        if normalized in {"image_urls", "image_url", "image_paths", "image_path", "images"}:
            preferred.append(name)
        elif "image" in normalized or normalized in {"img", "img_url", "img_path"}:
            fallback.append(name)
    columns = [*preferred, *fallback]
    if not columns:
        raise ValueError(f"Could not auto-detect image path columns from TSV header: {fieldnames}")
    return list(dict.fromkeys(columns))


def split_image_cell(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    if value.startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    if "|" in value:
        return [part.strip() for part in value.split("|") if part.strip()]
    return [value]


def resolve_image_path(raw_value: str, tsv_parent: Path, repo_root: Path) -> Path | None:
    raw_value = raw_value.strip().strip("\"'")
    if not raw_value:
        return None
    if raw_value.startswith("file://"):
        raw_value = raw_value[7:]
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw_value):
        return None

    raw_path = Path(raw_value)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([tsv_parent / raw_path, repo_root / raw_path])
    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS:
            return candidate.resolve()
    return None


def image_extension_from_url(raw_url: str, content_type: str = "") -> str:
    suffix = Path(urllib.parse.urlparse(raw_url).path).suffix.lower()
    if suffix in IMAGE_EXTS:
        return suffix
    guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip().lower()) if content_type else None
    if guessed == ".jpe":
        guessed = ".jpg"
    return guessed if guessed in IMAGE_EXTS else ".jpg"


def download_image_url(raw_url: str, item_id: str, image_cache_dir: Path) -> Path | None:
    item_dir = image_cache_dir / item_id
    item_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(raw_url.encode("utf-8")).hexdigest()[:20]
    for existing in item_dir.glob(f"{digest}.*"):
        if existing.is_file() and existing.suffix.lower() in IMAGE_EXTS and existing.stat().st_size > 0:
            return existing.resolve()

    request = urllib.request.Request(raw_url, headers={"User-Agent": "Mozilla/5.0"})
    tmp_path: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content_type = response.headers.get("Content-Type", "")
            media_type = content_type.lower().split(";", 1)[0].strip()
            if media_type and not media_type.startswith("image/"):
                return None
            ext = image_extension_from_url(raw_url, content_type)
            target = item_dir / f"{digest}{ext}"
            if target.exists() and target.stat().st_size > 0:
                return target.resolve()
            tmp_path = target.with_suffix(target.suffix + ".tmp")
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            if tmp_path.stat().st_size == 0:
                tmp_path.unlink(missing_ok=True)
                return None
            os.replace(tmp_path, target)
            return target.resolve()
    except (OSError, TimeoutError, urllib.error.URLError):
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return None


def resolve_image_reference(
    raw_value: str,
    tsv_parent: Path,
    repo_root: Path,
    item_id: str,
    image_cache_dir: Path,
) -> Path | None:
    local_path = resolve_image_path(raw_value, tsv_parent, repo_root)
    if local_path is not None:
        return local_path
    raw_value = raw_value.strip().strip("\"'")
    parsed = urllib.parse.urlparse(raw_value)
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    return download_image_url(raw_value, item_id, image_cache_dir)


def tsv_record_from_row(row: dict[str, str]) -> dict[str, Any]:
    return dict(row)


def stream_selected_tsv(
    tsv_path: Path,
    selected_ids: list[str],
    expected_images: int,
    *,
    scan_full_tsv: bool,
    image_cache_dir: Path,
) -> dict[str, TsvItemData]:
    raise_csv_field_limit()
    selected = set(selected_ids)
    found: dict[str, TsvItemData] = {
        item_id: TsvItemData(images=[], record={"item_id": item_id}) for item_id in selected_ids
    }
    seen_paths: dict[str, set[Path]] = {item_id: set() for item_id in selected_ids}
    seen_raw: dict[str, set[str]] = {item_id: set() for item_id in selected_ids}
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"TSV has no header: {tsv_path}")
        itm_column = auto_detect_itm_column(reader.fieldnames)
        image_columns = auto_detect_image_columns(reader.fieldnames)
        tsv_parent = tsv_path.resolve().parent

        for row in reader:
            item_id = str(row.get(itm_column, "")).strip()
            if item_id not in selected:
                continue
            item_data = found[item_id]
            if item_data.record == {"item_id": item_id}:
                item_data.record = tsv_record_from_row(row)
            for column in image_columns:
                for raw_image in split_image_cell(row.get(column, "")):
                    cleaned_raw = raw_image.strip().strip("\"'")
                    if cleaned_raw and cleaned_raw not in seen_raw[item_id]:
                        seen_raw[item_id].add(cleaned_raw)
                        item_data.raw_images.append(cleaned_raw)
                    resolved = resolve_image_reference(raw_image, tsv_parent, SCRIPT_DIR, item_id, image_cache_dir)
                    if resolved is None or resolved in seen_paths[item_id]:
                        continue
                    item_data.images.append(resolved)
                    seen_paths[item_id].add(resolved)
                    if not scan_full_tsv and len(item_data.images) >= expected_images:
                        break
                if not scan_full_tsv and len(item_data.images) >= expected_images:
                    break
            if not scan_full_tsv and all(len(found[item_id].images) >= expected_images for item_id in selected):
                break
    return found


def build_tsv_index(tsv_path: Path, index_path: Path) -> dict[str, dict[str, Any]]:
    """Full scan of the TSV; saves {item_id: {record, raw_images}} as a pickle index."""
    raise_csv_field_limit()
    index: dict[str, dict[str, Any]] = {}
    print(f"Building TSV index from {tsv_path} — this runs once and is reused on future calls.", flush=True)
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"TSV has no header: {tsv_path}")
        itm_column = auto_detect_itm_column(reader.fieldnames)
        image_columns = auto_detect_image_columns(reader.fieldnames)
        for row in reader:
            item_id = str(row.get(itm_column, "")).strip()
            if not item_id:
                continue
            if item_id not in index:
                raw_images: list[str] = []
                seen_raw: set[str] = set()
                for column in image_columns:
                    for raw in split_image_cell(row.get(column, "")):
                        if raw and raw not in seen_raw:
                            raw_images.append(raw)
                            seen_raw.add(raw)
                index[item_id] = {"record": tsv_record_from_row(row), "raw_images": raw_images}
            else:
                seen_raw = set(index[item_id]["raw_images"])
                for column in image_columns:
                    for raw in split_image_cell(row.get(column, "")):
                        if raw and raw not in seen_raw:
                            index[item_id]["raw_images"].append(raw)
                            seen_raw.add(raw)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved TSV index ({len(index)} items) to {index_path}", flush=True)
    return index


def load_or_build_tsv_index(tsv_path: Path, index_path: Path, *, force_rebuild: bool) -> dict[str, dict[str, Any]]:
    if not force_rebuild and index_path.exists():
        if index_path.stat().st_mtime >= tsv_path.stat().st_mtime:
            print(f"Loading TSV index from {index_path}", flush=True)
            with index_path.open("rb") as fh:
                return pickle.load(fh)
            print("TSV index loaded.", flush=True)
    return build_tsv_index(tsv_path, index_path)


def resolve_from_index(
    index: dict[str, dict[str, Any]],
    selected_ids: list[str],
    expected_images: int,
    *,
    tsv_path: Path,
    image_cache_dir: Path,
    scan_full_tsv: bool,
) -> dict[str, TsvItemData]:
    """O(1) per-item lookup from pre-built index; image resolution logic is unchanged."""
    tsv_parent = tsv_path.resolve().parent
    found: dict[str, TsvItemData] = {}
    for item_id in selected_ids:
        entry = index.get(item_id)
        if entry is None:
            found[item_id] = TsvItemData(images=[], record={"item_id": item_id})
            continue
        images: list[Path] = []
        seen: set[Path] = set()
        for raw in entry["raw_images"]:
            if not scan_full_tsv and len(images) >= expected_images:
                break
            resolved = resolve_image_reference(raw, tsv_parent, SCRIPT_DIR, item_id, image_cache_dir)
            if resolved is None or resolved in seen:
                continue
            images.append(resolved)
            seen.add(resolved)
        found[item_id] = TsvItemData(images=images, record=entry["record"], raw_images=list(entry["raw_images"]))
    return found


def call_image_picker_api(
    item_id: str,
    script1: str,
    script2: str,
    image_urls: list[str],
    api_url: str,
) -> dict[str, Any] | None:
    payload = json.dumps({
        "id": item_id,
        "script1": script1,
        "script2": script2,
        "images": image_urls[:10],
    }).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


def iter_records(data: Any, path: str = "$") -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(data, dict):
        if any(key in data for key in ID_KEYS):
            yield path, data
        for key, value in data.items():
            yield from iter_records(value, f"{path}.{key}")
    elif isinstance(data, list):
        for index, value in enumerate(data):
            yield from iter_records(value, f"{path}[{index}]")


def record_item_id(record: dict[str, Any]) -> str | None:
    for key in ID_KEYS:
        if key in record and record[key] not in {None, ""}:
            return str(record[key])
    return None


def add_script_record(
    out: dict[str, ScriptRecord],
    item_id: str,
    record: dict[str, Any],
    json_file: Path,
    json_path: str,
) -> None:
    state = out.setdefault(item_id, ScriptRecord())
    try:
        extract_reference_text(record, ["script1"])
        extract_reference_text(record, ["script2"])
    except VerificationError:
        return
    if state.record is not None:
        state.duplicate_count += 1
        state.error = "multiple script records"
        return
    state.record = record
    state.json_file = json_file
    state.json_path = json_path


def load_selected_script_records(json_dir: Path, selected_ids: list[str]) -> dict[str, ScriptRecord]:
    selected = set(selected_ids)
    records: dict[str, ScriptRecord] = {item_id: ScriptRecord() for item_id in selected_ids}
    paths = sorted([*json_dir.rglob("*.json"), *json_dir.rglob("*.jsonl")])
    if not paths:
        raise VerificationError(f"No .json or .jsonl files found under {json_dir}")

    remaining = set(selected)
    for path in paths:
        if not remaining:
            break
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise VerificationError(f"Invalid JSONL in {path}:{line_no}: {exc}") from exc
                    if isinstance(row, dict):
                        item_id = record_item_id(row)
                        if item_id in selected:
                            add_script_record(records, item_id, row, path, f"$[{line_no}]")
                            if records[item_id].record is not None and not records[item_id].error:
                                remaining.discard(item_id)
                    else:
                        for json_path, record in iter_records(row, f"$[{line_no}]"):
                            item_id = record_item_id(record)
                            if item_id in selected:
                                add_script_record(records, item_id, record, path, json_path)
                                if records[item_id].record is not None and not records[item_id].error:
                                    remaining.discard(item_id)
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            for json_path, record in iter_records(data):
                item_id = record_item_id(record)
                if item_id in selected:
                    add_script_record(records, item_id, record, path, json_path)
                    if records[item_id].record is not None and not records[item_id].error:
                        remaining.discard(item_id)

    for item_id, state in records.items():
        if state.record is None and not state.error:
            state.error = "missing script record"
    return records


def media_duration(path: Path) -> float:
    if shutil_which("ffprobe"):
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
        ).strip()
        if out:
            return float(out)

    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    for line in probe.stdout.splitlines():
        if "Duration:" not in line:
            continue
        stamp = line.split("Duration:", 1)[1].split(",", 1)[0].strip()
        hours, minutes, seconds = stamp.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Could not determine media duration: {path}")


def shutil_which(cmd: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        path = Path(directory) / cmd
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


def run_command(cmd: list[str], *, cwd: Path | None = None) -> tuple[int, str]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout.strip()


def trim_video(src: Path, dst: Path, *, force: bool) -> None:
    if dst.exists() and not force:
        return
    duration = media_duration(src)
    trimmed_duration = max(duration - 1.0, 0.1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-t",
        f"{trimmed_duration:.3f}",
        "-map",
        "0",
        "-c",
        "copy",
        str(dst),
    ]
    code, output = run_command(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg trim failed for {src.name}: {output}")


def stable_ts_command(args: argparse.Namespace, video_path: Path, ass_path: Path) -> list[str]:
    if args.stable_ts_template:
        rendered = args.stable_ts_template.format(input=str(video_path), output=str(ass_path))
        return shlex.split(rendered)
    cmd = [args.stable_ts_cmd, str(video_path), "-o", str(ass_path)]
    for extra_arg in args.stable_ts_arg:
        cmd.extend(shlex.split(extra_arg))
    return cmd


def generate_ass(args: argparse.Namespace, video_path: Path, ass_path: Path) -> None:
    if ass_path.exists() and ass_path.stat().st_size > 0 and not args.force_ass:
        return
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    code, output = run_command(stable_ts_command(args, video_path, ass_path))
    if code != 0:
        raise RuntimeError(f"stable-ts failed for {video_path.name}: {output}")
    if not ass_path.exists() or ass_path.stat().st_size == 0:
        raise RuntimeError(f"stable-ts did not create ASS output: {ass_path}")


def build_timeline_for_item(
    item_id: str,
    pair: dict[int, BatchVideo],
    *,
    catalog: dict[str, dict[str, Any]],
    images: list[Path],
    output_dir: Path,
    ass_dir: Path,
    rendered_dir: Path,
    timeline_config: dict[str, Any],
    bridge_duration: float,
    end_card_duration: float,
    expected_images: int,
) -> dict[str, Any]:
    first = pair[1]
    second = pair[2]
    image_paths = [timeline_relative(path, output_dir) for path in images[:expected_images]]
    product = product_payload(item_id, first.product_slug, image_paths, catalog, output_dir)
    durations = [media_duration(first.path), media_duration(second.path)]
    videos = [timeline_relative(first.path, output_dir), timeline_relative(second.path, output_dir)]
    ass_files = [
        timeline_relative(ass_dir / f"{first.path.stem}.ass", output_dir),
        timeline_relative(ass_dir / f"{second.path.stem}.ass", output_dir),
    ]
    scenes = build_scenes_from_config(
        timeline_config,
        durations=durations,
        item_id=item_id,
        image_count=len(image_paths),
        bridge_duration=bridge_duration,
        end_card_duration=end_card_duration,
    )
    return {
        "talking_head_video": videos[0],
        "talking_head_videos": videos,
        "base_subtitles_ass": ass_files[0],
        "base_subtitles_ass_files": ass_files,
        "output": repo_relative(rendered_dir / f"{item_id}_stitched.mp4"),
        "auto_bridge": {"enabled": False},
        "products": [product],
        "scenes": scenes,
    }


def clean_ass_and_timeline(
    timeline_path: Path,
    record: dict[str, Any],
    *,
    reference_fields: list[str],
    style_path: Path | None = None,
) -> tuple[list[Path], Path, int]:
    subtitle_font = CLEAN_ASS_SUBTITLE_FONT
    subtitle_font_size = CLEAN_ASS_SUBTITLE_FONT_SIZE
    if style_path is not None:
        subtitle_cfg = json.loads(style_path.read_text(encoding="utf-8")).get("subtitle_style", {})
        subtitle_font = str(subtitle_cfg.get("font", CLEAN_ASS_SUBTITLE_FONT)).strip() or CLEAN_ASS_SUBTITLE_FONT
        subtitle_font_size = int(subtitle_cfg.get("font_size", CLEAN_ASS_SUBTITLE_FONT_SIZE))

    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    timeline_dir = timeline_path.resolve().parent
    values = timeline.get("base_subtitles_ass_files")
    if not values:
        single = timeline.get("base_subtitles_ass")
        values = [single] if single else []
    ass_paths = []
    for value in values:
        path = Path(value)
        ass_paths.append(path if path.is_absolute() else timeline_dir / path)
    if len(reference_fields) != len(ass_paths):
        raise RuntimeError(f"Need {len(ass_paths)} reference field(s), got {reference_fields}")

    clean_paths: list[Path] = []
    total_remaining_issues = 0
    for ass_path, reference_field in zip(ass_paths, reference_fields):
        issues, _field_used = compare_against_record(record, ass_path, reference_field)
        clean_path = ass_path.with_name(f"{ass_path.stem}.clean.ass")
        write_clean_ass(ass_path, clean_path, issues, subtitle_font, subtitle_font_size)
        clean_issues, _clean_field = compare_against_record(record, clean_path, reference_field)
        total_remaining_issues += len(clean_issues)
        clean_paths.append(clean_path)

    clean_timeline = dict(timeline)
    clean_timeline["base_subtitles_ass_files"] = [
        clean_timeline_relative(path, timeline_dir) for path in clean_paths
    ]
    clean_timeline["base_subtitles_ass"] = clean_timeline["base_subtitles_ass_files"][0]
    repair_clean_timeline_media_paths(clean_timeline, timeline_dir)
    clean_timeline_path = timeline_path.with_name(f"{timeline_path.stem}.clean.json")
    clean_timeline_path.write_text(json.dumps(clean_timeline, indent=2) + "\n", encoding="utf-8")
    return clean_paths, clean_timeline_path, total_remaining_issues


def render_item(args: argparse.Namespace, code_dir: Path, timeline_path: Path, output_path: Path) -> None:
    cmd = [
        sys.executable,
        str(code_dir / "render_video.py"),
        "--style",
        str(args.style),
        "--timeline",
        str(timeline_path),
        "--out",
        str(output_path),
    ]
    if args.subtitle_primary_color:
        cmd.extend(["--subtitle-primary-color", args.subtitle_primary_color])
    if args.subtitle_secondary_color:
        cmd.extend(["--subtitle-secondary-color", args.subtitle_secondary_color])
    code, output = run_command(cmd, cwd=SCRIPT_DIR)
    if code != 0:
        raise RuntimeError(f"render failed: {output}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"render did not create output: {output_path}")


def base_report_row(item_id: str, *, status: str = "", failure_reason: str = "") -> dict[str, Any]:
    return {field: "" for field in REPORT_FIELDS} | {
        "ITM": item_id,
        "status": status,
        "failure_reason": failure_reason,
        "images_found_count": 0,
        "images_found_paths": [],
    }


def process_item(item: DiscoveredItem, ctx: PipelineContext) -> dict[str, Any]:
    args = ctx.args
    item_id = item.item_id
    pair = item.pair
    row = base_report_row(item_id)
    review_reasons: list[str] = []
    failure_reasons: list[str] = []

    row["video_path_script1"] = repo_path(pair[1].path)
    row["video_path_script2"] = repo_path(pair[2].path)
    output_path = args.out_video_dir / f"{item_id}_stitched.mp4"
    row["output_video_path"] = repo_path(output_path)

    try:
        if 1 not in pair or 2 not in pair:
            raise RuntimeError("missing pair")

        script_record = ctx.script_records.get(item_id)
        if script_record is None or script_record.record is None:
            raise RuntimeError(script_record.error if script_record else "missing script record")
        if script_record.error == "multiple script records":
            raise RuntimeError("multiple script records")

        tsv_item = ctx.tsv_data.get(item_id, TsvItemData(images=[], record={"item_id": item_id}))
        images = tsv_item.images
        if not images:
            raise RuntimeError("zero images")
        expected_images = args.expected_images

        _t = time.monotonic()
        if tsv_item.raw_images:
            http_urls = [
                u for u in tsv_item.raw_images
                if urllib.parse.urlparse(u).scheme in {"http", "https"}
            ]
            if http_urls:
                try:
                    script1_text = str(extract_reference_text(script_record.record, ["script1"]))
                    script2_text = str(extract_reference_text(script_record.record, ["script2"]))
                    api_result = call_image_picker_api(
                        item_id, script1_text, script2_text, http_urls, IMAGE_PICKER_URL
                    )
                except VerificationError:
                    api_result = None
                if api_result is not None:
                    hero_url = api_result.get("hero_image_url", "")
                    s1_urls = api_result.get("script1_image_urls", [])
                    s2_urls = api_result.get("script2_image_urls", [])
                    # Slot order matches image_index convention:
                    # [0] hero, [1] script-1 image, [2] script-2 image.
                    # Only the first picked image per script is used; s1[1]/s2[1] are intentionally dropped.
                    url_slots = [
                        hero_url,
                        s1_urls[0] if s1_urls else "",
                        s2_urls[0] if s2_urls else "",
                    ]
                    picked: list[Path] = []
                    for slot_url in url_slots:
                        if not slot_url:
                            continue
                        resolved = download_image_url(slot_url, item_id, args.image_cache_dir)
                        if resolved:
                            picked.append(resolved)
                    if len(picked) >= 2:
                        images = picked
        with ctx.print_lock:
            print(f"  [{item_id}] image_picker: {time.monotonic()-_t:.1f}s", flush=True)

        row["images_found_count"] = len(images)
        row["images_found_paths"] = [repo_path(path) for path in images]
        if len(images) < expected_images:
            review_reasons.append(f"fewer than {expected_images} images")

        _t = time.monotonic()
        args.trimmed_video_dir.mkdir(parents=True, exist_ok=True)
        trimmed_pair: dict[int, BatchVideo] = {}
        for script_index in [1, 2]:
            src = pair[script_index].path
            dst = args.trimmed_video_dir / src.name
            trim_video(src, dst, force=args.force_trim)
            row[f"trimmed_video_path_script{script_index}"] = repo_path(dst)
            trimmed_pair[script_index] = BatchVideo(
                item_id=item_id,
                script_index=script_index,
                product_slug=pair[script_index].product_slug,
                path=dst,
            )
        with ctx.print_lock:
            print(f"  [{item_id}] trim: {time.monotonic()-_t:.1f}s", flush=True)

        _t = time.monotonic()
        args.ass_dir.mkdir(parents=True, exist_ok=True)
        for script_index in [1, 2]:
            video_path = trimmed_pair[script_index].path
            ass_path = args.ass_dir / f"{video_path.stem}.ass"
            generate_ass(args, video_path, ass_path)
            row[f"ass_path_script{script_index}"] = repo_path(ass_path)
        with ctx.print_lock:
            print(f"  [{item_id}] ass: {time.monotonic()-_t:.1f}s", flush=True)

        _t = time.monotonic()
        args.out_timeline_dir.mkdir(parents=True, exist_ok=True)
        catalog = {item_id: tsv_item.record}
        timeline = build_timeline_for_item(
            item_id,
            trimmed_pair,
            catalog=catalog,
            images=images,
            output_dir=args.out_timeline_dir,
            ass_dir=args.ass_dir,
            rendered_dir=args.out_video_dir,
            timeline_config=ctx.timeline_config,
            bridge_duration=args.bridge_duration,
            end_card_duration=args.end_card_duration,
            expected_images=expected_images,
        )
        timeline_path = args.out_timeline_dir / f"{item_id}.json"
        timeline_path.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")
        row["timeline_json_path"] = repo_path(timeline_path)

        clean_paths, clean_timeline_path, remaining_issues = clean_ass_and_timeline(
            timeline_path,
            script_record.record,
            reference_fields=["script1", "script2"],
            style_path=args.style,
        )
        with ctx.print_lock:
            print(f"  [{item_id}] timeline+clean_ass: {time.monotonic()-_t:.1f}s", flush=True)

        row["clean_ass_path_script1"] = repo_path(clean_paths[0]) if clean_paths else ""
        row["clean_ass_path_script2"] = repo_path(clean_paths[1]) if len(clean_paths) > 1 else ""
        row["clean_timeline_json_path"] = repo_path(clean_timeline_path)
        if remaining_issues:
            review_reasons.append(f"clean ASS has {remaining_issues} remaining review issue(s)")

        if not args.skip_render:
            _t = time.monotonic()
            args.out_video_dir.mkdir(parents=True, exist_ok=True)
            with ctx.print_lock:
                print(f"  [{item_id}] rendering...", flush=True)
            render_item(args, ctx.code_dir, clean_timeline_path, output_path)
            with ctx.print_lock:
                print(f"  [{item_id}] render: {time.monotonic()-_t:.1f}s", flush=True)

        row["status"] = "REVIEW" if review_reasons else "PASS"
        row["failure_reason"] = "; ".join(review_reasons)
    except Exception as exc:  # noqa: BLE001 - the report should capture all item failures.
        failure_reasons.append(str(exc))
        row["status"] = "FAIL"
        row["failure_reason"] = "; ".join(reason for reason in failure_reasons if reason)

    with ctx.print_lock:
        print(f"{row['status']:6s} {item_id} {row['failure_reason']}", flush=True)
    return row


def write_reports(rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["images_found_paths"] = json.dumps(row.get("images_found_paths", []), ensure_ascii=False)
            writer.writerow(csv_row)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full paired-video batch pipeline for selected ITMs.")
    parser.add_argument("--batch-dir", required=True, type=Path)
    parser.add_argument("--json-dir", required=True, type=Path)
    parser.add_argument("--image-tsv", required=True, type=Path)
    parser.add_argument("--image-cache-dir", default="outputs/image_cache", type=Path)
    parser.add_argument("--style", required=True, type=Path)
    parser.add_argument("--timeline-config", required=True, type=Path)
    parser.add_argument("--ass-dir", required=True, type=Path)
    parser.add_argument("--trimmed-video-dir", required=True, type=Path)
    parser.add_argument("--out-timeline-dir", required=True, type=Path)
    parser.add_argument("--out-video-dir", required=True, type=Path)
    parser.add_argument("--report-csv", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--limit", default=200, type=positive_int)
    parser.add_argument("--parallel", default=5, type=positive_int)
    parser.add_argument("--itm-list", help="Allowlist as comma-separated ITMs or a text file path.")
    parser.add_argument("--expected-images", default=None, type=positive_int)
    parser.add_argument("--scan-full-tsv", action="store_true")
    parser.add_argument(
        "--tsv-index",
        default=None,
        type=Path,
        help="Path to a pickle index of the TSV (e.g. outputs/bgmh.pkl). Built on first run, loaded on subsequent runs.",
    )
    parser.add_argument("--rebuild-tsv-index", action="store_true", help="Force rebuild of the TSV index even if it is up to date.")
    parser.add_argument("--force-ass", action="store_true")
    parser.add_argument("--force-trim", action="store_true")
    parser.add_argument("--force-timeline", action="store_true",
                        help="Rebuild timelines and clean ASS files only; skip rendering. "
                             "Trim and ASS generation are skipped when their output files already exist.")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--subtitle-primary-color")
    parser.add_argument("--subtitle-secondary-color")
    parser.add_argument("--stable-ts-cmd", default="stable-ts")
    parser.add_argument(
        "--stable-ts-arg",
        action="append",
        default=["--max_words 4"],
        help="Extra stable-ts argument(s). May be passed multiple times; shell-style splitting is applied.",
    )
    parser.add_argument(
        "--stable-ts-template",
        help="Full command template using {input} and {output}, overriding --stable-ts-cmd/--stable-ts-arg.",
    )
    parser.add_argument("--bridge-duration", default=None, type=float)
    parser.add_argument("--end-card-duration", default=None, type=float)
    args = parser.parse_args()

    args.batch_dir = resolve_existing_path(args.batch_dir)
    args.json_dir = resolve_existing_path(args.json_dir)
    args.image_tsv = resolve_existing_path(args.image_tsv)
    args.image_cache_dir = resolve_output_path(args.image_cache_dir)
    args.style = resolve_existing_path(args.style, search_code_dir=True)
    args.timeline_config = resolve_existing_path(args.timeline_config, search_code_dir=True)
    args.ass_dir = resolve_output_path(args.ass_dir)
    args.trimmed_video_dir = resolve_output_path(args.trimmed_video_dir)
    args.out_timeline_dir = resolve_output_path(args.out_timeline_dir)
    args.out_video_dir = resolve_output_path(args.out_video_dir)
    args.report_csv = resolve_output_path(args.report_csv)
    args.report_json = resolve_output_path(args.report_json)
    args.image_cache_dir = resolve_output_path(args.image_cache_dir) if args.image_cache_dir else None
    return args


def main() -> int:
    started = time.monotonic()
    args = parse_args()
    if args.force_timeline:
        args.skip_render = True
    for label in ["batch_dir", "json_dir", "image_tsv", "style", "timeline_config"]:
        path = getattr(args, label)
        if not path.exists():
            print(f"ERROR: --{label.replace('_', '-')} does not exist: {path}", file=sys.stderr)
            return 2

    timeline_config = load_timeline_config(args.timeline_config)
    timeline_defaults = timeline_config.get("defaults", {})
    if args.expected_images is None:
        args.expected_images = int(timeline_defaults.get("expected_images", 4))
    if args.bridge_duration is None:
        args.bridge_duration = float(timeline_defaults.get("bridge_duration", 3.0))
    if args.end_card_duration is None:
        args.end_card_duration = float(timeline_defaults.get("end_card_duration", 1.0))

    allowlist = parse_itm_list(args.itm_list)
    complete, all_pairs = discover_batch_items(args.batch_dir)
    selected, preflight_rows = select_items(complete, all_pairs, allowlist, args.limit)
    selected_ids = [item.item_id for item in selected]
    if not selected:
        write_reports(preflight_rows, args.report_csv, args.report_json)
        print(f"No complete ITM pairs selected from {args.batch_dir}")
        return 1

    print(
        f"Selected {len(selected_ids)} complete ITM pair(s); loading up to "
        f"{args.expected_images} image(s) each.",
        flush=True,
    )
    if args.tsv_index:
        tsv_index = load_or_build_tsv_index(args.image_tsv, args.tsv_index, force_rebuild=args.rebuild_tsv_index)
        tsv_data = resolve_from_index(
            tsv_index,
            selected_ids,
            args.expected_images,
            tsv_path=args.image_tsv,
            image_cache_dir=args.image_cache_dir,
            scan_full_tsv=args.scan_full_tsv,
        )
    else:
        tsv_data = stream_selected_tsv(
            args.image_tsv,
            selected_ids,
            args.expected_images,
            scan_full_tsv=args.scan_full_tsv,
            image_cache_dir=args.image_cache_dir,
        )
    print("Loading selected script records.", flush=True)
    script_records = load_selected_script_records(args.json_dir, selected_ids)
    print("Script records loaded.", flush=True)

    ctx = PipelineContext(
        args=args,
        code_dir=CODE_DIR,
        timeline_config=timeline_config,
        timeline_defaults=timeline_defaults,
        selected_ids=selected_ids,
        script_records=script_records,
        tsv_data=tsv_data,
        print_lock=threading.Lock(),
    )

    rows_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(process_item, item, ctx): item.item_id for item in selected}
        for future in as_completed(futures):
            item_id = futures[future]
            try:
                rows_by_id[item_id] = future.result()
            except Exception as exc:  # noqa: BLE001 - protects report writing from worker crashes.
                rows_by_id[item_id] = base_report_row(item_id, status="FAIL", failure_reason=str(exc))

    rows = [*preflight_rows, *[rows_by_id[item_id] for item_id in selected_ids]]
    write_reports(rows, args.report_csv, args.report_json)
    counts = {status: sum(1 for row in rows if row["status"] == status) for status in ["PASS", "REVIEW", "FAIL"]}
    elapsed = time.monotonic() - started
    print(
        f"Wrote reports: {args.report_csv} and {args.report_json}. "
        f"PASS={counts['PASS']} REVIEW={counts['REVIEW']} FAIL={counts['FAIL']} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return 0 if counts["FAIL"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
