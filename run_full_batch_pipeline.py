#!/usr/bin/env python3
"""Run the full paired-video batch pipeline for a bounded ITM set.

The runner intentionally keeps the large image TSV on a streaming path: it first
selects the batch ITMs to process, then reads only matching TSV rows and keeps
only matching local image files or cached HTTP(S) images for those ITMs.
"""

from __future__ import annotations

import argparse
import contextlib
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
import queue
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
CODE_DIR = SCRIPT_DIR / "scripts"
if CODE_DIR.exists():
    sys.path.insert(0, str(CODE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

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
TSV_PROGRESS_EVERY_ROWS = 100_000


class ApiCallError(Exception):
    """Raised when an HTTP service call fails. Carries enough info to debug.

    str(err) renders as a JSON list: [api_url, payload, error]
    so it round-trips cleanly into report.csv / report.json failure_reason fields.
    """

    def __init__(self, api_url: str, payload: str, error: str) -> None:
        self.api_url = api_url
        self.payload = payload
        self.error = error
        super().__init__(self.as_failure_reason())

    def as_failure_reason(self) -> str:
        # Truncate payload to keep report rows readable; full payload still logged elsewhere.
        truncated_payload = self.payload if len(self.payload) <= 500 else self.payload[:500] + "...[truncated]"
        return json.dumps([self.api_url, truncated_payload, self.error], ensure_ascii=False)


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
    "time_tsv_index_load_s",
    "time_image_picker_s",
    "time_trim_s",
    "time_ass_s",
    "time_ass_script1_s",
    "time_ass_script2_s",
    "time_ass_worker_wait_s",
    "time_ass_model_load_s",
    "time_ass_audio_extract_s",
    "time_ass_transcribe_s",
    "time_ass_write_s",
    "time_timeline_clean_ass_s",
    "time_tag_windows_s",
    "time_render_s",
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
class PreparedItem:
    """Output of stage 1 (prepare). Carries everything stage 2 (render) needs."""
    item_id: str
    row: dict[str, Any]
    clean_timeline_path: Path | None = None
    output_path: Path | None = None
    review_reasons: list[str] = field(default_factory=list)
    failed: bool = False


class WhisperXWorker:
    """Loads the wav2vec2 alignment model once and reuses it for all ASS generation calls."""

    def __init__(self, device: str | None = None, language: str = "en") -> None:
        self._device = device
        self._language = language
        self._align_model: Any = None
        self._metadata: Any = None
        self._load_lock = threading.Lock()

    def _resolve_device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch  # noqa: PLC0415
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_loaded(self) -> float:
        if self._align_model is not None:
            return 0.0
        with self._load_lock:
            if self._align_model is not None:
                return 0.0
            import torch  # noqa: PLC0415
            import whisperx  # noqa: PLC0415
            torch.set_num_threads(2)
            started = time.monotonic()
            self._align_model, self._metadata = whisperx.load_align_model(
                language_code=self._language,
                device=self._resolve_device(),
            )
            return time.monotonic() - started

    def generate(self, video_path: Path, ass_path: Path, script_text: str) -> dict[str, float]:
        import re as _re  # noqa: PLC0415
        import tempfile  # noqa: PLC0415
        import whisperx  # noqa: PLC0415
        from whisperX_scripts.align import extract_audio, SAMPLE_RATE  # noqa: PLC0415
        from whisperX_scripts.json_to_ass import group_words, build_dialogue, ASS_HEADER  # noqa: PLC0415

        metrics = {"model_load_s": 0.0, "audio_extract_s": 0.0, "transcribe_s": 0.0, "write_s": 0.0}
        metrics["model_load_s"] = self._ensure_loaded()
        device = self._resolve_device()

        wav_path = Path(tempfile.mktemp(suffix=".wav"))
        try:
            started = time.monotonic()
            extract_audio(str(video_path), str(wav_path))
            metrics["audio_extract_s"] = time.monotonic() - started

            started = time.monotonic()
            audio = whisperx.load_audio(str(wav_path))
            duration = len(audio) / SAMPLE_RATE
            norm_text = _re.sub(r"[^\w\s']", " ", script_text)
            norm_text = _re.sub(r"\s+", " ", norm_text).strip()
            segments = [{"start": 0.0, "end": duration, "text": norm_text}]
            result = whisperx.align(
                segments, self._align_model, self._metadata, audio, device,
                return_char_alignments=False,
            )
            words = [
                {
                    "word": w.get("word", "").strip(),
                    "start": round(w.get("start", 0.0), 4),
                    "end": round(w.get("end", 0.0), 4),
                }
                for seg in result.get("segments", [])
                for w in seg.get("words", [])
                if w.get("word", "").strip()
            ]
            metrics["transcribe_s"] = time.monotonic() - started

            started = time.monotonic()
            ass_path.parent.mkdir(parents=True, exist_ok=True)
            groups = group_words(words)
            dialogues = [build_dialogue(i, g) for i, g in enumerate(groups)]
            content = ASS_HEADER + "\n".join(dialogues) + "\n"
            ass_path.write_text(content, encoding="utf-8")
            metrics["write_s"] = time.monotonic() - started
        finally:
            wav_path.unlink(missing_ok=True)

        if not ass_path.exists() or ass_path.stat().st_size == 0:
            raise RuntimeError(f"WhisperX did not create ASS output: {ass_path}")
        return metrics


class WhisperXWorkerPool:
    """Round-robin pool of WhisperXWorker instances across one or more CUDA devices.

    Each worker holds its own alignment model, so N workers → N parallel GPU streams.
    A thread blocks on acquire() only when all workers are busy.
    """

    def __init__(self, workers: list[WhisperXWorker]) -> None:
        self._q: queue.Queue[WhisperXWorker] = queue.Queue()
        for w in workers:
            self._q.put(w)

    @contextlib.contextmanager
    def acquire(self):
        worker = self._q.get()
        try:
            yield worker
        finally:
            self._q.put(worker)


class StableTsWorker:
    """Loads the Whisper model once and reuses it for all ASS generation calls."""

    def __init__(self, model_name: str, device: str | None, max_words: int) -> None:
        self._model_name = model_name
        self._device = device
        self._max_words = max_words
        self._model: Any = None
        self._load_lock = threading.Lock()

    @contextlib.contextmanager
    def _device_context(self):
        if not self._device or not str(self._device).startswith("cuda"):
            yield
            return
        import torch  # noqa: PLC0415
        with torch.cuda.device(self._device):
            yield

    def _ensure_loaded(self) -> float:
        if self._model is not None:
            return 0.0
        with self._load_lock:
            if self._model is not None:
                return 0.0
            import stable_whisper  # noqa: PLC0415
            started = time.monotonic()
            with self._device_context():
                self._model = stable_whisper.load_model(
                    self._model_name,
                    device=self._device,
                    cpu_preload=False,
                )
            return time.monotonic() - started

    def generate(self, video_path: Path, ass_path: Path, script_text: str = "") -> dict[str, float]:
        metrics = {"model_load_s": 0.0, "audio_extract_s": 0.0, "transcribe_s": 0.0, "write_s": 0.0}
        with self._device_context():
            metrics["model_load_s"] = self._ensure_loaded()
            ass_path.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            result = self._model.transcribe(str(video_path), word_timestamps=True)
            metrics["transcribe_s"] = time.monotonic() - started
        if self._max_words:
            started = time.monotonic()
            result.split_by_length(max_words=self._max_words)
            metrics["split_s"] = time.monotonic() - started
        started = time.monotonic()
        result.to_ass(filepath=str(ass_path))
        metrics["write_s"] = time.monotonic() - started
        if not ass_path.exists() or ass_path.stat().st_size == 0:
            raise RuntimeError(f"stable-ts did not create ASS output: {ass_path}")
        return metrics


class StableTsWorkerPool:
    """Round-robin pool of StableTsWorker instances across one or more CUDA devices."""

    def __init__(self, workers: list[StableTsWorker]) -> None:
        self._q: queue.Queue[StableTsWorker] = queue.Queue()
        for w in workers:
            self._q.put(w)

    @contextlib.contextmanager
    def acquire(self):
        worker = self._q.get()
        try:
            yield worker
        finally:
            self._q.put(worker)


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
    ass_semaphore: threading.Semaphore
    ass_pool: Any = None  # WhisperXWorkerPool | StableTsWorkerPool | None
    tsv_index_load_s: float = 0.0


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
    # A comma-separated list of ITM codes cannot be a file path — skip path
    # resolution to avoid OSError: [Errno 36] File name too long.
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
    err_msg = ""
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content_type = response.headers.get("Content-Type", "")
            media_type = content_type.lower().split(";", 1)[0].strip()
            if media_type and not media_type.startswith("image/"):
                err_msg = f"non-image content-type: {content_type!r}"
            else:
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
                    err_msg = "downloaded zero bytes"
                else:
                    os.replace(tmp_path, target)
                    return target.resolve()
    except urllib.error.HTTPError as exc:
        err_msg = f"HTTP {exc.code} {exc.reason}"
    except urllib.error.URLError as exc:
        err_msg = f"URLError: {exc.reason}"
    except TimeoutError:
        err_msg = "timeout after 30s"
    except OSError as exc:
        err_msg = f"OSError: {exc}"

    if tmp_path is not None:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    if err_msg:
        print(
            f"  [{item_id}] image download failed: "
            f"{json.dumps([raw_url, 'GET', err_msg], ensure_ascii=False)}",
            flush=True,
        )
    return None


def resolve_image_reference(
    raw_value: str,
    tsv_parent: Path,
    repo_root: Path,
    item_id: str,
    image_cache_dir: Path,
    *,
    download_remote: bool = True,
) -> Path | None:
    local_path = resolve_image_path(raw_value, tsv_parent, repo_root)
    if local_path is not None:
        return local_path
    raw_value = raw_value.strip().strip("\"'")
    parsed = urllib.parse.urlparse(raw_value)
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    if not download_remote:
        return None
    return download_image_url(raw_value, item_id, image_cache_dir)


def tsv_record_from_row(row: dict[str, str]) -> dict[str, Any]:
    return dict(row)


def format_file_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def has_enough_image_candidates(item_data: TsvItemData, expected_images: int) -> bool:
    return len(item_data.raw_images) >= expected_images or len(item_data.images) >= expected_images


def http_image_urls(raw_values: Iterable[str]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        cleaned = raw_value.strip().strip("\"'")
        if not cleaned or cleaned in seen:
            continue
        parsed = urllib.parse.urlparse(cleaned)
        if parsed.scheme.lower() in {"http", "https"}:
            urls.append(cleaned)
            seen.add(cleaned)
    return urls


def download_selected_http_images(
    raw_urls: Iterable[str],
    item_id: str,
    image_cache_dir: Path,
    *,
    limit: int,
    existing: Iterable[Path] = (),
) -> list[Path]:
    if limit <= 0:
        return []

    downloaded: list[Path] = []
    seen_paths = {path.resolve() for path in existing}
    for raw_url in raw_urls:
        resolved = download_image_url(raw_url, item_id, image_cache_dir)
        if resolved is None or resolved in seen_paths:
            continue
        downloaded.append(resolved)
        seen_paths.add(resolved)
        if len(downloaded) >= limit:
            break
    return downloaded


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
    selected_count = len(selected)
    found: dict[str, TsvItemData] = {
        item_id: TsvItemData(images=[], record={"item_id": item_id}) for item_id in selected_ids
    }
    seen_paths: dict[str, set[Path]] = {item_id: set() for item_id in selected_ids}
    seen_raw: dict[str, set[str]] = {item_id: set() for item_id in selected_ids}
    ready_ids: set[str] = set()
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"TSV has no header: {tsv_path}")
        itm_column = auto_detect_itm_column(reader.fieldnames)
        image_columns = auto_detect_image_columns(reader.fieldnames)
        tsv_parent = tsv_path.resolve().parent

        for row_count, row in enumerate(reader, start=1):
            if row_count % TSV_PROGRESS_EVERY_ROWS == 0:
                print(
                    f"Scanned {row_count:,} TSV rows; prepared image references for "
                    f"{len(ready_ids)}/{selected_count} selected ITMs.",
                    flush=True,
                )
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
                    resolved = resolve_image_reference(
                        raw_image,
                        tsv_parent,
                        SCRIPT_DIR,
                        item_id,
                        image_cache_dir,
                        download_remote=False,
                    )
                    if resolved is None or resolved in seen_paths[item_id]:
                        continue
                    item_data.images.append(resolved)
                    seen_paths[item_id].add(resolved)
                    if not scan_full_tsv and has_enough_image_candidates(item_data, expected_images):
                        break
                if not scan_full_tsv and has_enough_image_candidates(item_data, expected_images):
                    break
            if has_enough_image_candidates(item_data, expected_images):
                ready_ids.add(item_id)
            if not scan_full_tsv and len(ready_ids) == selected_count:
                print(
                    f"Prepared image references for all {selected_count} selected ITMs after "
                    f"{row_count:,} TSV rows.",
                    flush=True,
                )
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
        for row_count, row in enumerate(reader, start=1):
            if row_count % TSV_PROGRESS_EVERY_ROWS == 0:
                print(
                    f"Indexed {row_count:,} TSV rows; captured {len(index):,} unique ITMs.",
                    flush=True,
                )
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
            size_text = format_file_size(index_path.stat().st_size)
            print(f"Loading TSV index from {index_path} ({size_text})", flush=True)
            with index_path.open("rb") as fh:
                index = pickle.load(fh)
            print(f"TSV index loaded ({len(index):,} items).", flush=True)
            return index
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
    """O(1) per-item lookup from pre-built index with local-path resolution only."""
    tsv_parent = tsv_path.resolve().parent
    found: dict[str, TsvItemData] = {}
    total_selected = len(selected_ids)
    for index_pos, item_id in enumerate(selected_ids, start=1):
        entry = index.get(item_id)
        if entry is None:
            found[item_id] = TsvItemData(images=[], record={"item_id": item_id})
            continue
        images: list[Path] = []
        seen: set[Path] = set()
        for raw in entry["raw_images"]:
            if not scan_full_tsv and len(images) >= expected_images:
                break
            resolved = resolve_image_reference(
                raw,
                tsv_parent,
                SCRIPT_DIR,
                item_id,
                image_cache_dir,
                download_remote=False,
            )
            if resolved is None or resolved in seen:
                continue
            images.append(resolved)
            seen.add(resolved)
        found[item_id] = TsvItemData(images=images, record=entry["record"], raw_images=list(entry["raw_images"]))
        if index_pos % 250 == 0 or index_pos == total_selected:
            print(
                f"Resolved local image paths for {index_pos:,}/{total_selected:,} selected ITMs.",
                flush=True,
            )
    return found


def call_image_picker_api(
    item_id: str,
    script1: str,
    script2: str,
    image_urls: list[str],
    api_url: str,
) -> dict[str, Any]:
    """Raises ApiCallError on any failure (HTTP error, timeout, URLError, etc.)."""
    payload_str = json.dumps({
        "id": item_id,
        "script1": script1,
        "script2": script2,
        "images": image_urls[:10],
    })
    request = urllib.request.Request(
        api_url,
        data=payload_str.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ApiCallError(api_url, payload_str, f"HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise ApiCallError(api_url, payload_str, f"URLError: {exc.reason}") from exc
    except TimeoutError:
        raise ApiCallError(api_url, payload_str, "timeout after 60s") from None
    except Exception as exc:  # noqa: BLE001
        raise ApiCallError(api_url, payload_str, f"{type(exc).__name__}: {exc}") from exc


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


TAG_MATCHER_API_URL = "http://10.12.46.8:8084/tag_matcher"


def _ass_time_to_sec(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _parse_ass_dialogues(path: Path) -> list[dict[str, Any]]:
    import re
    dialogues: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue
        text = re.sub(r"\{[^}]*\}", "", parts[9]).strip()
        if text:
            dialogues.append({"start": parts[1].strip(), "end": parts[2].strip(), "text": text})
    return dialogues


def attributes_from_tsv_record(record: dict[str, Any]) -> list[dict[str, str]]:
    raw = record.get("attributes", "")
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return [{"key": k.replace("_", " "), "value": str(v)} for k, v in parsed.items() if v]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _call_tag_matcher_batch(
    items: list[dict[str, Any]],
    api_url: str,
) -> dict[int, list[dict[str, Any]]]:
    """Send a batch of script items; return a mapping of script_id -> tags.

    Raises ApiCallError on any failure.
    """
    payload_str = json.dumps({"items": items})
    req = urllib.request.Request(
        api_url,
        data=payload_str.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ApiCallError(api_url, payload_str, f"HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise ApiCallError(api_url, payload_str, f"URLError: {exc.reason}") from exc
    except TimeoutError:
        raise ApiCallError(api_url, payload_str, "timeout after 60s") from None
    except Exception as exc:  # noqa: BLE001
        raise ApiCallError(api_url, payload_str, f"{type(exc).__name__}: {exc}") from exc
    # Response: {"results": [{"script_id": 1, "tags": [...]}, ...]}
    return {entry["script_id"]: entry.get("tags", []) for entry in result.get("results", [])}


def _attributes_as_tags(attributes: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [{"name": a["key"], "value": a["value"]} for a in attributes if a.get("key") and a.get("value")]


def fetch_tags(
    item_id: str,
    clean_ass_paths: list[Path],
    api_url: str,
    attributes: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Single batch call to tag_matcher; returns (tag_windows, bridge_points).

    Builds one item per dialogue chunk ([:4] and [4:] per video) plus one item
    covering all dialogues for the bridge overlay. script_id is used to match
    results back to their chunk.
    """
    # script_id -> (video_index, start_sec, end_sec)  for tag_window chunks
    chunk_meta: dict[int, tuple[int, float, float]] = {}
    items: list[dict[str, Any]] = []
    script_id = 0

    for video_index, ass_path in enumerate(clean_ass_paths):
        dialogues = _parse_ass_dialogues(ass_path)
        if not dialogues:
            continue
        for chunk in [dialogues[:4], dialogues[4:]]:
            if not chunk:
                continue
            items.append({
                "id": item_id,
                "script_id": script_id,
                "script": " ".join(d["text"] for d in chunk),
                "attributes": attributes,
            })
            chunk_meta[script_id] = (
                video_index,
                _ass_time_to_sec(chunk[0]["start"]),
                _ass_time_to_sec(chunk[-1]["end"]),
            )
            script_id += 1

    # One extra item with the full script across all videos for bridge points.
    bridge_script_id = script_id
    all_dialogues = [d for path in clean_ass_paths for d in _parse_ass_dialogues(path)]
    if all_dialogues:
        items.append({
            "id": item_id,
            "script_id": bridge_script_id,
            "script": " ".join(d["text"] for d in all_dialogues),
            "attributes": attributes,
        })

    if not items:
        return [], []

    tags_by_id = _call_tag_matcher_batch(items, api_url)

    windows: list[dict[str, Any]] = []
    for sid, (video_index, start_sec, end_sec) in chunk_meta.items():
        windows.append({
            "video_index": video_index,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "tags": tags_by_id.get(sid, []),
        })

    bridge_tags = tags_by_id.get(bridge_script_id, [])
    bridge_points = [f"{t['name']}: {t['value']}" for t in bridge_tags if t.get("name") and t.get("value")][:4]

    return windows, bridge_points


def shutil_which(cmd: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        path = Path(directory) / cmd
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


def run_command(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> tuple[int, str]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
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


def generate_ass(
    args: argparse.Namespace,
    video_path: Path,
    ass_path: Path,
    script_text: str = "",
    *,
    worker: Any = None,
) -> dict[str, float]:
    metrics = {"model_load_s": 0.0, "audio_extract_s": 0.0, "transcribe_s": 0.0, "write_s": 0.0}
    if ass_path.exists() and ass_path.stat().st_size > 0 and not args.force_ass:
        return metrics
    if worker is not None:
        return worker.generate(video_path, ass_path, script_text)
    # Subprocess fallback (stable-ts CLI only)
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    env: dict[str, str] | None = None
    if args.stable_ts_cuda_device is not None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.stable_ts_cuda_device)
    started = time.monotonic()
    code, output = run_command(stable_ts_command(args, video_path, ass_path), env=env)
    metrics["transcribe_s"] = time.monotonic() - started
    if code != 0:
        raise RuntimeError(f"stable-ts failed for {video_path.name}: {output}")
    if not ass_path.exists() or ass_path.stat().st_size == 0:
        raise RuntimeError(f"stable-ts did not create ASS output: {ass_path}")
    return metrics


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
        issues, _ = compare_against_record(record, ass_path, reference_field)
        clean_path = ass_path.with_name(f"{ass_path.stem}.clean.ass")
        write_clean_ass(ass_path, clean_path, issues, subtitle_font, subtitle_font_size)
        clean_issues, _ = compare_against_record(record, clean_path, reference_field)
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


def render_item(
    args: argparse.Namespace,
    code_dir: Path,
    timeline_path: Path,
    output_path: Path,
) -> tuple[float, float]:
    """Run the renderer and return (wall_s, cpu_s) for the subprocess tree."""
    import resource as _resource

    renderer = getattr(args, "renderer", "render_scenes")
    if renderer == "render_scenes":
        cmd = [
            sys.executable,
            str(code_dir / "render_scenes.py"),
            "--style", str(args.style),
            "--timeline", str(timeline_path),
            "--out", str(output_path),
        ]
    else:
        cmd = [
            sys.executable,
            str(code_dir / "render_video.py"),
            "--style", str(args.style),
            "--timeline", str(timeline_path),
            "--out", str(output_path),
        ]
        if args.subtitle_primary_color:
            cmd.extend(["--subtitle-primary-color", args.subtitle_primary_color])
        if args.subtitle_secondary_color:
            cmd.extend(["--subtitle-secondary-color", args.subtitle_secondary_color])

    # Snapshot children CPU before so concurrent siblings don't inflate our reading
    before = _resource.getrusage(_resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    code, output = run_command(cmd, cwd=SCRIPT_DIR)
    wall_s = time.monotonic() - t0
    after = _resource.getrusage(_resource.RUSAGE_CHILDREN)
    cpu_s = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)

    if code != 0:
        raise RuntimeError(f"render failed: {output}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"render did not create output: {output_path}")
    return wall_s, cpu_s


def base_report_row(item_id: str, *, status: str = "", failure_reason: str = "") -> dict[str, Any]:
    return {field: "" for field in REPORT_FIELDS} | {
        "ITM": item_id,
        "status": status,
        "failure_reason": failure_reason,
        "images_found_count": 0,
        "images_found_paths": [],
    }


def prepare_item(item: DiscoveredItem, ctx: PipelineContext) -> PreparedItem:
    """Stage 1: image lookup, trim, ASS, timeline, clean ASS, tag windows.

    Does NOT render. Returns a PreparedItem the render stage can consume.
    """
    args = ctx.args
    item_id = item.item_id
    pair = item.pair
    row = base_report_row(item_id)
    review_reasons: list[str] = []
    failure_reasons: list[str] = []

    row["video_path_script1"] = repo_path(pair[1].path)
    row["video_path_script2"] = repo_path(pair[2].path)
    row["time_tsv_index_load_s"] = ctx.tsv_index_load_s
    output_path = args.out_video_dir / f"{item_id}_stitched.mp4"
    row["output_video_path"] = repo_path(output_path)

    if not args.force_render and output_path.exists() and output_path.stat().st_size > 0:
        row["status"] = "SKIPPED"
        row["failure_reason"] = "stitched output already exists"
        with ctx.print_lock:
            print(f"SKIP   {item_id} (existing output {output_path.name})", flush=True)
        return PreparedItem(
            item_id=item_id,
            row=row,
            clean_timeline_path=None,
            output_path=output_path,
            review_reasons=[],
            failed=False,
        )

    clean_timeline_path = args.out_timeline_dir / f"{item_id}.clean.json"
    if (
        not args.force_rebuild_timeline
        and clean_timeline_path.exists()
        and clean_timeline_path.stat().st_size > 0
    ):
        row["timeline_json_path"] = repo_path(args.out_timeline_dir / f"{item_id}.json")
        row["clean_timeline_json_path"] = repo_path(clean_timeline_path)
        row["status"] = "PASS"
        with ctx.print_lock:
            print(f"  [{item_id}] reusing existing clean timeline → render", flush=True)
        return PreparedItem(
            item_id=item_id,
            row=row,
            clean_timeline_path=clean_timeline_path,
            output_path=output_path,
            review_reasons=[],
            failed=False,
        )

    try:
        if 1 not in pair or 2 not in pair:
            raise RuntimeError("missing pair")

        script_record = ctx.script_records.get(item_id)
        if script_record is None or script_record.record is None:
            raise RuntimeError(script_record.error if script_record else "missing script record")
        if script_record.error == "multiple script records":
            raise RuntimeError("multiple script records")

        tsv_item = ctx.tsv_data.get(item_id, TsvItemData(images=[], record={"item_id": item_id}))
        images = list(tsv_item.images)
        expected_images = args.expected_images

        http_urls = http_image_urls(tsv_item.raw_images)

        # --- Phase 1: trim + ASS generation in parallel ---
        args.trimmed_video_dir.mkdir(parents=True, exist_ok=True)
        args.ass_dir.mkdir(parents=True, exist_ok=True)
        trimmed_pair: dict[int, BatchVideo] = {}
        ass_model_load_s = 0.0
        ass_audio_extract_s = 0.0
        ass_transcribe_s = 0.0
        ass_write_s = 0.0
        ass_worker_wait_s = 0.0

        def run_trim_and_ass(script_index: int) -> tuple[int, Path, Path, float, float, float, dict[str, float]]:
            trim_started = time.monotonic()
            src = pair[script_index].path
            dst = args.trimmed_video_dir / src.name
            trim_video(src, dst, force=args.force_trim)
            time_trim = time.monotonic() - trim_started

            ass_started = time.monotonic()
            ass_path = args.ass_dir / f"{dst.stem}.ass"
            metrics = {"model_load_s": 0.0, "transcribe_s": 0.0, "split_s": 0.0, "write_s": 0.0}
            wait_s = 0.0
            try:
                script_text = extract_reference_text(script_record.record, [f"script{script_index}"]).text
            except VerificationError:
                script_text = ""
            if ctx.ass_pool is not None:
                wait_started = time.monotonic()
                with ctx.ass_pool.acquire() as worker:
                    wait_s = time.monotonic() - wait_started
                    metrics = generate_ass(args, dst, ass_path, script_text, worker=worker)
            else:
                wait_started = time.monotonic()
                with ctx.ass_semaphore:
                    wait_s = time.monotonic() - wait_started
                    metrics = generate_ass(args, dst, ass_path, script_text)
            time_ass = time.monotonic() - ass_started
            return script_index, dst, ass_path, time_trim, time_ass, wait_s, metrics

        _t_phase1 = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as phase1_executor:
            trim_ass_futures = [phase1_executor.submit(run_trim_and_ass, si) for si in [1, 2]]
            for future in as_completed(trim_ass_futures):
                script_index, trimmed_path, ass_path, time_trim, time_ass, worker_wait_s, metrics = future.result()
                trimmed_pair[script_index] = BatchVideo(
                    item_id=item_id,
                    script_index=script_index,
                    product_slug=pair[script_index].product_slug,
                    path=trimmed_path,
                )
                row[f"trimmed_video_path_script{script_index}"] = repo_path(trimmed_path)
                row[f"ass_path_script{script_index}"] = repo_path(ass_path)
                row[f"time_ass_script{script_index}_s"] = round(time_trim + time_ass, 2)
                ass_model_load_s += metrics["model_load_s"]
                ass_audio_extract_s += metrics.get("audio_extract_s", 0.0)
                ass_transcribe_s += metrics["transcribe_s"]
                ass_write_s += metrics["write_s"]
                ass_worker_wait_s += worker_wait_s

        _elapsed_phase1 = time.monotonic() - _t_phase1
        row["time_trim_s"] = round(_elapsed_phase1, 2)
        row["time_ass_s"] = round(_elapsed_phase1, 2)
        row["time_ass_worker_wait_s"] = round(ass_worker_wait_s, 2)
        row["time_ass_model_load_s"] = round(ass_model_load_s, 2)
        row["time_ass_audio_extract_s"] = round(ass_audio_extract_s, 2)
        row["time_ass_transcribe_s"] = round(ass_transcribe_s, 2)
        row["time_ass_write_s"] = round(ass_write_s, 2)
        with ctx.print_lock:
            print(
                f"  [{item_id}] phase1 (trim+ass): {_elapsed_phase1:.1f}s "
                f"(s1={row['time_ass_script1_s']}, s2={row['time_ass_script2_s']}, "
                f"worker_wait={row['time_ass_worker_wait_s']}, "
                f"load={row['time_ass_model_load_s']}, tx={row['time_ass_transcribe_s']})",
                flush=True,
            )

        # --- Phase 2: image_picker + fetch_tags in parallel (both need clean ASS) ---
        _t = time.monotonic()
        args.out_timeline_dir.mkdir(parents=True, exist_ok=True)
        catalog = {item_id: tsv_item.record}

        # Build a placeholder timeline with existing images to get clean_paths for tag matching.
        # The final timeline is rebuilt after images are resolved.
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
        _elapsed = time.monotonic() - _t
        row["time_timeline_clean_ass_s"] = round(_elapsed, 2)
        with ctx.print_lock:
            print(f"  [{item_id}] timeline+clean_ass: {_elapsed:.1f}s", flush=True)

        row["clean_ass_path_script1"] = repo_path(clean_paths[0]) if clean_paths else ""
        row["clean_ass_path_script2"] = repo_path(clean_paths[1]) if len(clean_paths) > 1 else ""
        row["clean_timeline_json_path"] = repo_path(clean_timeline_path)
        if remaining_issues:
            review_reasons.append(f"clean ASS has {remaining_issues} remaining review issue(s)")

        attributes = attributes_from_tsv_record(tsv_item.record)

        def run_image_picker() -> dict[str, Any] | None:
            if not http_urls:
                return None
            if args.fallback_image_picker:
                with ctx.print_lock:
                    print(f"  [{item_id}] image picker skipped (--fallback-image-picker)", flush=True)
                return None
            script1_text = str(extract_reference_text(script_record.record, ["script1"]))
            script2_text = str(extract_reference_text(script_record.record, ["script2"]))
            return call_image_picker_api(item_id, script1_text, script2_text, http_urls, IMAGE_PICKER_URL)

        def run_fetch_tags() -> tuple[list[dict[str, Any]], list[str]]:
            if not clean_paths or args.fallback_tag_picker:
                with ctx.print_lock:
                    print(f"  [{item_id}] tag picker skipped (--fallback-tag-picker), leaving tag_windows and bridge_points empty", flush=True)
                return [], []
            return fetch_tags(item_id, clean_paths, TAG_MATCHER_API_URL, attributes)

        _t_phase2 = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as phase2_executor:
            image_picker_future = phase2_executor.submit(run_image_picker)
            fetch_tags_future = phase2_executor.submit(run_fetch_tags)
            image_picker_api_result = image_picker_future.result()
            tag_windows, bridge_points = fetch_tags_future.result()

        _elapsed_image_picker = time.monotonic() - _t_phase2
        row["time_image_picker_s"] = round(_elapsed_image_picker, 2)
        _elapsed_tag_windows = time.monotonic() - _t_phase2
        row["time_tag_windows_s"] = round(_elapsed_tag_windows, 2)
        with ctx.print_lock:
            print(
                f"  [{item_id}] phase2 (image_picker+tag_matcher): {_elapsed_image_picker:.1f}s "
                f"tag_windows={len(tag_windows)}, bridge_points={len(bridge_points)}",
                flush=True,
            )

        # Resolve images from picker result
        _t = time.monotonic()
        if http_urls:
            if image_picker_api_result is not None:
                hero_url = image_picker_api_result.get("hero_image_url", "")
                s1_urls = image_picker_api_result.get("script1_image_urls", [])
                s2_urls = image_picker_api_result.get("script2_image_urls", [])
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
            if len(images) < expected_images:
                images.extend(
                    download_selected_http_images(
                        http_urls,
                        item_id,
                        args.image_cache_dir,
                        limit=expected_images - len(images),
                        existing=images,
                    )
                )
        with ctx.print_lock:
            print(f"  [{item_id}] images: {len(images)} ready in {time.monotonic() - _t:.1f}s", flush=True)

        if not images:
            diagnostic = json.dumps(
                [
                    IMAGE_PICKER_URL,
                    f"local={len(tsv_item.images)}, http_urls={len(http_urls)}",
                    "all downloads failed (see earlier 'image download failed' lines)",
                ],
                ensure_ascii=False,
            )
            raise RuntimeError(f"zero images: {diagnostic}")

        row["images_found_count"] = len(images)
        row["images_found_paths"] = [repo_path(path) for path in images]
        if len(images) < expected_images:
            review_reasons.append(f"fewer than {expected_images} images")

        # Rebuild timeline with final resolved images and write tag_windows into clean.json
        _t = time.monotonic()
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
        timeline_path.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")

        clean_paths, clean_timeline_path, remaining_issues = clean_ass_and_timeline(
            timeline_path,
            script_record.record,
            reference_fields=["script1", "script2"],
            style_path=args.style,
        )
        _elapsed = time.monotonic() - _t
        row["time_timeline_clean_ass_s"] = round(row["time_timeline_clean_ass_s"] + _elapsed, 2)
        with ctx.print_lock:
            print(f"  [{item_id}] timeline+clean_ass (final): {_elapsed:.1f}s", flush=True)

        row["clean_ass_path_script1"] = repo_path(clean_paths[0]) if clean_paths else ""
        row["clean_ass_path_script2"] = repo_path(clean_paths[1]) if len(clean_paths) > 1 else ""
        row["clean_timeline_json_path"] = repo_path(clean_timeline_path)
        if remaining_issues:
            review_reasons.append(f"clean ASS has {remaining_issues} remaining review issue(s)")

        if clean_paths:
            clean_tl = json.loads(clean_timeline_path.read_text(encoding="utf-8"))
            if clean_tl.get("products"):
                clean_tl["products"][0]["tag_windows"] = tag_windows
                clean_tl["products"][0]["bridge_overlay_points"] = bridge_points
            clean_timeline_path.write_text(json.dumps(clean_tl, indent=2) + "\n", encoding="utf-8")

        # Tentative status; render_step will overwrite to FAIL if render breaks.
        row["status"] = "REVIEW" if review_reasons else "PASS"
        row["failure_reason"] = "; ".join(review_reasons)
        return PreparedItem(
            item_id=item_id,
            row=row,
            clean_timeline_path=clean_timeline_path,
            output_path=output_path,
            review_reasons=list(review_reasons),
            failed=False,
        )
    except Exception as exc:  # noqa: BLE001 - the report should capture all item failures.
        failure_reasons.append(str(exc))
        row["status"] = "FAIL"
        row["failure_reason"] = "; ".join(reason for reason in failure_reasons if reason)
        with ctx.print_lock:
            print(f"{row['status']:6s} {item_id} {row['failure_reason']}", flush=True)
        return PreparedItem(
            item_id=item_id,
            row=row,
            clean_timeline_path=None,
            output_path=None,
            review_reasons=list(review_reasons),
            failed=True,
        )


def render_step(prepared: PreparedItem, ctx: PipelineContext) -> dict[str, Any]:
    """Stage 2: render the clean timeline produced by prepare_item."""
    args = ctx.args
    item_id = prepared.item_id
    row = prepared.row
    review_reasons = prepared.review_reasons

    if prepared.clean_timeline_path is None or prepared.output_path is None:
        return row

    try:
        _t = time.monotonic()
        args.out_video_dir.mkdir(parents=True, exist_ok=True)
        with ctx.print_lock:
            print(f"  [{item_id}] rendering...", flush=True)
        _render_wall, _render_cpu = render_item(args, ctx.code_dir, prepared.clean_timeline_path, prepared.output_path)
        _elapsed = time.monotonic() - _t
        row["time_render_s"] = round(_elapsed, 2)
        row["time_render_cpu_s"] = round(_render_cpu, 2)
        row["time_render_cpu_pct"] = round(100.0 * _render_cpu / _render_wall, 1) if _render_wall > 0 else 0.0
        with ctx.print_lock:
            print(f"  [{item_id}] render: {_elapsed:.1f}s  cpu={_render_cpu:.1f}s ({row['time_render_cpu_pct']:.0f}%)", flush=True)
        row["status"] = "REVIEW" if review_reasons else "PASS"
        row["failure_reason"] = "; ".join(review_reasons)
    except Exception as exc:  # noqa: BLE001
        row["status"] = "FAIL"
        row["failure_reason"] = str(exc)

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
    parser.add_argument(
        "--parallel",
        default=5,
        type=positive_int,
        help="Max items concurrently in the prepare stage (image picker, trim, ASS, timeline, clean ASS, tag windows).",
    )
    parser.add_argument(
        "--render-parallel",
        default=16,
        type=positive_int,
        help="Max items concurrently in the render stage. Independent of --parallel because render is CPU/ffmpeg-bound, not GPU-bound.",
    )
    parser.add_argument(
        "--ass-backend",
        default="whisperx",
        choices=["whisperx", "stable-ts"],
        help="ASS generation backend: 'whisperx' (forced alignment, needs script text) or 'stable-ts' (transcription). Default: whisperx.",
    )
    # --- WhisperX backend args ---
    parser.add_argument(
        "--whisperx-workers",
        default=1,
        type=positive_int,
        help="Max concurrent WhisperX alignment workers. Set to the number of GPUs available (default: 1).",
    )
    parser.add_argument(
        "--whisperx-cuda-device",
        default=None,
        help="Single CUDA device index for WhisperX (used when --whisperx-devices is not set). E.g. --whisperx-cuda-device 1",
    )
    parser.add_argument(
        "--whisperx-devices",
        default=None,
        help="Comma-separated CUDA device indices for WhisperX workers. E.g. --whisperx-devices 0,1,2,3",
    )
    parser.add_argument(
        "--whisperx-language",
        default="en",
        help="Language code for WhisperX forced alignment (default: en).",
    )
    # --- stable-ts backend args ---
    parser.add_argument(
        "--stable-ts-workers",
        default=1,
        type=positive_int,
        help="Max concurrent stable-ts (GPU) workers (default: 1).",
    )
    parser.add_argument("--stable-ts-cmd", default="stable-ts")
    parser.add_argument(
        "--stable-ts-model",
        default="base",
        help="Whisper model name for the in-process stable-ts worker (e.g. base, medium, large-v2).",
    )
    parser.add_argument(
        "--stable-ts-max-words",
        default=4,
        type=int,
        help="Max words per subtitle line for the in-process stable-ts worker (default: 4).",
    )
    parser.add_argument(
        "--stable-ts-cuda-device",
        default=None,
        help="Single CUDA device index for stable-ts. E.g. --stable-ts-cuda-device 1",
    )
    parser.add_argument(
        "--stable-ts-devices",
        default=None,
        help="Comma-separated CUDA device indices for stable-ts workers. E.g. --stable-ts-devices 0,1,2,3",
    )
    parser.add_argument(
        "--stable-ts-arg",
        action="append",
        default=["--max_words 4"],
        help="Extra stable-ts CLI argument(s). May be passed multiple times.",
    )
    parser.add_argument(
        "--stable-ts-template",
        default=None,
        help="Full stable-ts command template using {input} and {output}, overriding --stable-ts-cmd/--stable-ts-arg.",
    )
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
    parser.add_argument("--force-render", action="store_true", help="Re-render items even if the stitched output video already exists.")
    parser.add_argument("--force-rebuild-timeline", action="store_true", help="Rebuild timeline + clean ASS + tag windows even if <ITM>.clean.json already exists.")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument(
        "--renderer",
        default="render_scenes",
        choices=["render_video", "render_scenes"],
        help=(
            "Rendering backend. 'render_scenes' (default) renders each scene to a lossless intermediate then concatenates. "
            "'render_video' runs a single monolithic FFmpeg pass."
        ),
    )
    parser.add_argument("--subtitle-primary-color")
    parser.add_argument("--subtitle-secondary-color")
    parser.add_argument("--bridge-duration", default=None, type=float)
    parser.add_argument("--end-card-duration", default=None, type=float)
    parser.add_argument("--fallback-image-picker", action="store_true", help="Skip image picker API; use images in original order.")
    parser.add_argument("--fallback-tag-picker", action="store_true", help="Skip tag matcher API; use item attributes as tags in original order.")
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

    if not args.force_render:
        kept: list[DiscoveredItem] = []
        already_rendered = 0
        for item in complete:
            out_file = args.out_video_dir / f"{item.item_id}_stitched.mp4"
            if out_file.exists() and out_file.stat().st_size > 0:
                already_rendered += 1
                continue
            kept.append(item)
        if already_rendered:
            print(
                f"Pre-filtered {already_rendered} item(s) with existing stitched output "
                f"in {args.out_video_dir} (use --force-render to re-render).",
                flush=True,
            )
        complete = kept

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
    _tsv_t = time.monotonic()
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
    tsv_index_load_s = round(time.monotonic() - _tsv_t, 2)
    print(f"TSV data loaded in {tsv_index_load_s:.2f}s.", flush=True)
    print("Loading selected script records.", flush=True)
    script_records = load_selected_script_records(args.json_dir, selected_ids)
    print("Script records loaded.", flush=True)

    ass_pool: Any = None
    if args.ass_backend == "whisperx":
        if args.whisperx_devices:
            devices = [f"cuda:{d.strip()}" for d in args.whisperx_devices.split(",") if d.strip()]
        elif args.whisperx_cuda_device is not None:
            devices = [f"cuda:{args.whisperx_cuda_device}"]
        else:
            devices = ["cpu"]
        wx_workers = [
            WhisperXWorker(device=devices[i % len(devices)], language=args.whisperx_language)
            for i in range(args.whisperx_workers)
        ]
        ass_pool = WhisperXWorkerPool(wx_workers)
        device_summary = ", ".join(f"worker{i}→{devices[i % len(devices)]}" for i in range(len(wx_workers)))
        print(
            f"ASS backend: whisperx — {len(wx_workers)} worker(s), "
            f"language={args.whisperx_language} [{device_summary}]",
            flush=True,
        )
    elif args.ass_backend == "stable-ts" and not args.stable_ts_template:
        if args.stable_ts_devices:
            devices = [f"cuda:{d.strip()}" for d in args.stable_ts_devices.split(",") if d.strip()]
        elif args.stable_ts_cuda_device is not None:
            devices = [f"cuda:{args.stable_ts_cuda_device}"]
        else:
            devices = ["cuda"]
        st_workers = [
            StableTsWorker(
                model_name=args.stable_ts_model,
                device=devices[i % len(devices)],
                max_words=args.stable_ts_max_words,
            )
            for i in range(args.stable_ts_workers)
        ]
        ass_pool = StableTsWorkerPool(st_workers)
        device_summary = ", ".join(f"worker{i}→{devices[i % len(devices)]}" for i in range(len(st_workers)))
        print(
            f"ASS backend: stable-ts — {len(st_workers)} worker(s), "
            f"model={args.stable_ts_model} [{device_summary}]",
            flush=True,
        )
    else:
        print(f"ASS backend: stable-ts (subprocess mode)", flush=True)

    n_workers = args.whisperx_workers if args.ass_backend == "whisperx" else args.stable_ts_workers
    ctx = PipelineContext(
        args=args,
        code_dir=CODE_DIR,
        timeline_config=timeline_config,
        timeline_defaults=timeline_defaults,
        selected_ids=selected_ids,
        script_records=script_records,
        tsv_data=tsv_data,
        print_lock=threading.Lock(),
        ass_semaphore=threading.Semaphore(n_workers),
        ass_pool=ass_pool,
        tsv_index_load_s=tsv_index_load_s,
    )

    rows_by_id: dict[str, dict[str, Any]] = {}
    total_items = len(selected)
    stage_lock = threading.Lock()
    counters = {"prepare": 0, "render": 0}

    def bump(stage: str) -> int:
        with stage_lock:
            counters[stage] += 1
            return counters[stage]

    print(
        f"Starting two-stage pipeline: prepare_parallel={args.parallel}, "
        f"render_parallel={args.render_parallel}, items={total_items}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.parallel) as prepare_pool, \
         ThreadPoolExecutor(max_workers=args.render_parallel) as render_pool:
        prepare_futures = {
            prepare_pool.submit(prepare_item, item, ctx): item.item_id
            for item in selected
        }
        render_futures: dict[Any, str] = {}

        for future in as_completed(prepare_futures):
            item_id = prepare_futures[future]
            try:
                prepared = future.result()
            except Exception as exc:  # noqa: BLE001
                prepared = PreparedItem(
                    item_id=item_id,
                    row=base_report_row(item_id, status="FAIL", failure_reason=str(exc)),
                    failed=True,
                )
            done = bump("prepare")
            print(f"[stage] prepare: {done}/{total_items}", flush=True)

            if args.skip_render or prepared.failed or prepared.clean_timeline_path is None:
                rows_by_id[item_id] = prepared.row
                continue

            rf = render_pool.submit(render_step, prepared, ctx)
            render_futures[rf] = item_id

        for future in as_completed(render_futures):
            item_id = render_futures[future]
            try:
                rows_by_id[item_id] = future.result()
            except Exception as exc:  # noqa: BLE001
                rows_by_id[item_id] = base_report_row(item_id, status="FAIL", failure_reason=str(exc))
            done = bump("render")
            print(f"[stage] render: {done}/{len(render_futures)}", flush=True)

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
