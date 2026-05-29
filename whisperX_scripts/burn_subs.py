"""Burn ASS subtitles into a video using ffmpeg's ass filter."""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def burn_subtitles(
    video_path: str | Path,
    ass_path: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Burn ASS subtitles into video.  Audio is stream-copied (no re-encode).
    Returns the output path.
    """
    video_path  = Path(video_path)
    ass_path    = Path(ass_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy ASS to a temp path guaranteed to have no spaces / special chars so
    # the ffmpeg filter string doesn't need escaping.
    tmp_ass = Path(tempfile.mktemp(suffix=".ass"))
    shutil.copy(ass_path, tmp_ass)

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"ass={tmp_ass}",
            "-c:a", "copy",
            str(output_path),
        ]
        logger.info("Burning %s → %s", video_path.name, output_path.name)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2000:]}")
    finally:
        tmp_ass.unlink(missing_ok=True)

    logger.info("Saved: %s", output_path)
    return output_path
