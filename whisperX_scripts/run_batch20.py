"""
Process all videos in batch20/:
  - look up the matching script from 300k/scripts_batch_0020.jsonl
  - extract audio with ffmpeg
  - run WhisperX forced alignment
  - write per-video JSON to alignments/batch20/
  - convert each JSON to an ASS subtitle file with per-word karaoke tags

Output JSON shape:
  {
    "item_id":    "ITM...",
    "script_key": "script1" | "script2",
    "video":      "filename.mp4",
    "script":     "<original script text>",
    "words": [
      {"word": "Tired", "start": 0.12, "end": 0.45, "score": 0.99},
      ...
    ]
  }
"""

import json
import logging
import os
import re
import tempfile
from pathlib import Path

from align import align_audio, extract_audio
from json_to_ass import json_to_ass
from burn_subs import burn_subtitles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent
BATCH_DIR   = BASE_DIR / "batch20"
SCRIPTS_FILE = BASE_DIR / "300k" / "scripts_batch_0020.jsonl"
OUTPUT_DIR  = BASE_DIR / "alignments" / "batch20"
BURNED_DIR  = BASE_DIR / "outputs" / "batch20"


def load_scripts(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            records[d["item_id"]] = d
    logger.info("Loaded %d script records", len(records))
    return records


def parse_video_name(fname: str) -> tuple[str | None, str | None]:
    """Return (item_id, 'script1'|'script2') from a batch20 filename."""
    m = re.match(r"(ITM[A-F0-9]+)_.*?(script[12])_", fname)
    return (m.group(1), m.group(2)) if m else (None, None)


def process_video(video_path: Path, script_text: str, out_file: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        extract_audio(str(video_path), tmp_wav)
        words = align_audio(tmp_wav, script_text)
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)

    item_id, script_key = parse_video_name(video_path.name)
    payload = {
        "item_id": item_id,
        "script_key": script_key,
        "video": video_path.name,
        "script": script_text,
        "words": words,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved %s  (%d words)", out_file.name, len(words))

    ass_file = json_to_ass(out_file)
    logger.info("Saved %s", ass_file.name)

    burned = BURNED_DIR / video_path.name
    burn_subtitles(video_path, ass_file, burned)


def main() -> None:
    scripts = load_scripts(SCRIPTS_FILE)
    videos  = sorted(BATCH_DIR.glob("*.mp4"))
    logger.info("Found %d videos in %s", len(videos), BATCH_DIR)

    skipped = errors = done = 0

    for video_path in videos:
        item_id, script_key = parse_video_name(video_path.name)

        if not item_id or item_id not in scripts:
            logger.warning("No script found for %s — skipping", video_path.name)
            skipped += 1
            continue

        out_file = OUTPUT_DIR / f"{video_path.stem}.json"
        if out_file.exists():
            logger.info("Already done: %s", video_path.name)
            skipped += 1
            continue

        script_text = scripts[item_id][script_key]
        logger.info("▶  %s", video_path.name)

        try:
            process_video(video_path, script_text, out_file)
            done += 1
        except Exception as exc:
            logger.error("FAILED %s: %s", video_path.name, exc)
            errors += 1

    logger.info("Done: %d  Skipped: %d  Errors: %d", done, skipped, errors)


if __name__ == "__main__":
    main()
