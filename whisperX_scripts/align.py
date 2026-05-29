import os
import re
import subprocess
import logging
import torch
import whisperx

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def extract_audio(video_path: str, output_wav: str) -> None:
    """Extract mono 16 kHz PCM WAV from a video file using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le", output_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}:\n{result.stderr}")
    logger.info("Extracted audio → %s", output_wav)


def _normalize_for_alignment(text: str) -> str:
    """Remove punctuation the aligner vocabulary won't recognise, keep apostrophes."""
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def align_audio(
    audio_path: str,
    script_text: str,
    language: str = "en",
    device: str | None = None,
) -> list[dict]:
    """
    Forced-align audio_path against script_text with WhisperX / wav2vec2.

    Returns a list of word dicts:
        {"word": str, "start": float, "end": float, "score": float}
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Alignment device: %s", device)

    audio = whisperx.load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE

    logger.info("Loading alignment model (lang=%s)", language)
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)

    norm_text = _normalize_for_alignment(script_text)
    segments = [{"start": 0.0, "end": duration, "text": norm_text}]

    logger.info("Running forced alignment on %.1fs of audio", duration)
    result = whisperx.align(
        segments, model_a, metadata, audio, device,
        return_char_alignments=False,
    )

    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append({
                "word": w.get("word", "").strip(),
                "start": round(w.get("start", 0.0), 4),
                "end": round(w.get("end", 0.0), 4),
                "score": round(w.get("score", 0.0), 4),
            })

    logger.info("Aligned %d words", len(words))
    return words
