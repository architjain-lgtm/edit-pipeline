#!/usr/bin/env python3
"""
JSON-driven FFmpeg renderer for influencer-style product videos.

The renderer composes video/image/PiP/transitions in FFmpeg and burns merged ASS
captions at the very end of the filter graph. Existing stable-ts ASS generation
stays outside this script; ass_overlay_generator.py only preserves and appends.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from ass_overlay_generator import GradientWindow, collect_gradient_windows, merge_ass

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None
    _CV2_AVAILABLE = False

try:
    import numpy as _np
except ImportError:
    _np = None


RENDER_START = time.monotonic()


def detect_face_for_pip(
    video_path: Path,
    out_w: int,
    out_h: int,
    pip_w: int,
    pip_h: int,
    face_fill_ratio: float = 0.8,
) -> tuple[int, int, int, int] | None:
    """Return (crop_x, crop_y, crop_w, crop_h) on the out_w×out_h head stream so
    the detected face fills face_fill_ratio of the pip_w×pip_h box when scaled.
    Returns None when opencv is unavailable or no face is found."""
    if not _CV2_AVAILABLE:
        return None
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        duration = float(probe.stdout.strip() or "10")
        seek = duration * 0.3
        subprocess.run(
            [
                "ffmpeg", "-ss", f"{seek:.3f}", "-i", str(video_path),
                "-vf",
                f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                f"crop={out_w}:{out_h}",
                "-frames:v", "1", "-q:v", "2", "-y", tmp_path,
            ],
            capture_output=True, timeout=30,
        )
        img = _cv2.imread(tmp_path)
        if img is None:
            return None
        gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
        cascade_path = _cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = _cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        if not len(faces):
            return None
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_cx = fx + fw // 2
        face_cy = fy + fh // 2
        render_scale = min(pip_w * face_fill_ratio / fw, pip_h * face_fill_ratio / fh)
        crop_w = int(round(pip_w / render_scale))
        crop_h = int(round(pip_h / render_scale))
        crop_x = max(0, min(face_cx - crop_w // 2, out_w - crop_w))
        crop_y = max(0, min(face_cy - crop_h // 2, out_h - crop_h))
        return crop_x, crop_y, crop_w, crop_h
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def log_step(message: str) -> None:
    elapsed = time.monotonic() - RENDER_START
    print(f"[render {elapsed:7.2f}s] {message}", flush=True)


PRODUCT_SCENE_TYPES = {
    "product_scene",
    "product_pip_intro_scene",
    "pip_talking_head_no_effect",
    "product_bridge_gradient_overlay",
    "product_highlight_pip_scene",
    "product_highlight_pip_scene_with_gradient_overlay",
    "product_pip_head_exit_scene",
    "talking_head_product_strip_scene",
    "product_bridge_scene",
    "pip_scene",
    "talk_to_pip_scene",
    "pip_to_talk_scene",
    "product_overlay_float_scene",
    "flipkart_end_scene",
}

TALKING_HEAD_AUDIO_SCENE_TYPES = {
    "talking_head_scene",
    "product_pip_intro_scene",
    "pip_talking_head_no_effect",
    "product_highlight_pip_scene",
    "product_highlight_pip_scene_with_gradient_overlay",
    "product_pip_head_exit_scene",
    "talking_head_with_gradient_overlay",
    "talking_head_product_strip_scene",
    "pip_scene",
    "talk_to_pip_scene",
    "pip_to_talk_scene",
    "product_overlay_float_scene",
}

# Fullscreen talking-head scenes that should show a product image PiP
PRODUCT_IMAGE_PIP_SCENE_TYPES = {
    "talking_head_scene",
    "talking_head_with_gradient_overlay",
}

# Talking-head positional groups — used to detect full-screen ↔ PiP transitions.
_HEAD_FULL_TYPES = frozenset({
    "talking_head_scene",
    "talking_head_with_gradient_overlay",
    "talking_head_product_strip_scene",
})
_HEAD_PIP_TYPES = frozenset({
    "product_pip_intro_scene",
    "pip_talking_head_no_effect",
    "product_highlight_pip_scene",
    "product_highlight_pip_scene_with_gradient_overlay",
    "product_pip_head_exit_scene",
    "pip_scene",
    "talk_to_pip_scene",
    "pip_to_talk_scene",
})
# These scene types contain their own animated transition; dip-to-white is suppressed
# at their boundaries so the internal animation serves as the visual cut.
_HEAD_TRANSITION_TYPES = frozenset({"talk_to_pip_scene", "pip_to_talk_scene"})

# Product-only bridge scenes should participate in the same boundary fade on both
# sides so their entry/exit transitions stay visually consistent.
_BRIDGE_SCENE_TYPES = frozenset({
    "product_bridge_scene",
    "product_bridge_gradient_overlay",
})

# Scenes where the product image occupies the full-screen hero area.
# When transitioning FROM one of these TO a PRODUCT_IMAGE_PIP_SCENE_TYPES scene,
# the product image animates from full-screen to the PiP badge corner.
_PRODUCT_HERO_TYPES = frozenset({
    "product_pip_intro_scene",
    "pip_talking_head_no_effect",
    "product_highlight_pip_scene",
    "product_highlight_pip_scene_with_gradient_overlay",
    "product_pip_head_exit_scene",
    "product_bridge_gradient_overlay",
})


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")


def products_by_id(timeline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    products: dict[str, dict[str, Any]] = {}
    for product in timeline.get("products", []):
        pid = product.get("id")
        if pid:
            products[str(pid)] = product
    return products


def scene_product(scene: dict[str, Any], products: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    pid = scene.get("product_id")
    if pid is None:
        return None
    return products.get(str(pid))


SCRIPT_IMAGE_FALLBACK: dict[int, list[int]] = {
    0: [1, 0],
    1: [2, 1, 0],
}


def scene_image_path(scene: dict[str, Any], product: dict[str, Any], base: Path) -> Path:
    images = product.get("images") or []
    if not images:
        raise ValueError(f"Product {product.get('id')} has no images")
    requested = int(scene.get("image_index", 0))
    video_index = int(scene.get("talking_head_video_index", scene.get("video_index", 0)))
    candidates = [requested] + SCRIPT_IMAGE_FALLBACK.get(video_index, [0])
    index = 0
    for candidate in candidates:
        if 0 <= candidate < len(images):
            index = candidate
            break
    path = resolve(base, images[index])
    assert path is not None
    return path


def talking_head_paths(timeline: dict[str, Any], base: Path) -> list[Path]:
    values = timeline.get("talking_head_videos")
    if values is None:
        single = timeline.get("talking_head_video")
        values = [single] if single else []
    paths = [resolve(base, str(value)) for value in values if value]
    return [path for path in paths if path is not None]


def subtitle_ass_paths(timeline: dict[str, Any], base: Path) -> list[Path]:
    values = timeline.get("base_subtitles_ass_files")
    if values is None:
        single = timeline.get("base_subtitles_ass")
        values = [single] if single else []
    paths = [resolve(base, str(value)) for value in values if value]
    return [path for path in paths if path is not None]


def scene_video_index(scene: dict[str, Any]) -> int:
    return int(scene.get("talking_head_video_index", scene.get("video_index", 0)))


def scene_source_start(scene: dict[str, Any]) -> float:
    if "source_start" in scene:
        return float(scene["source_start"])
    if scene_video_index(scene) == 0:
        return float(scene["start"])
    return 0.0


def normalize_scene_times(
    style: dict[str, Any],
    timeline: dict[str, Any],
    timeline_path: Path | None = None,
) -> list[dict[str, Any]]:
    presets = style.get("scene_type_presets", {})
    normalized: list[dict[str, Any]] = []
    cursor = 0.0
    base = timeline_path.resolve().parent if timeline_path is not None else Path.cwd()
    head_paths = talking_head_paths(timeline, base)

    for raw_scene in timeline.get("scenes", []):
        scene = dict(raw_scene)
        scene_type = scene.get("type")
        has_start = "start" in scene
        has_end = "end" in scene

        start_value = scene.get("start")
        if not has_start or str(start_value).lower() in {"auto", "after_previous"}:
            start = cursor
        else:
            start = float(start_value)
        scene["start"] = start

        end_value = scene.get("end")
        duration_value = scene.get("duration")
        if has_end and str(end_value).lower() in {"video_end", "source_end"}:
            video_index = scene_video_index(scene)
            if video_index < 0 or video_index >= len(head_paths):
                raise ValueError(f"{scene_type} has end={end_value!r} but no valid talking-head video")
            source_start = scene_source_start(scene)
            end = start + max(media_duration(head_paths[video_index]) - source_start, 0.0)
        elif has_end:
            end = float(end_value)
        elif duration_value is not None:
            end = start + float(duration_value)
        elif scene_type in {"product_bridge_scene", "product_bridge_gradient_overlay"}:
            duration = float(presets.get("product_bridge_scene", {}).get("duration", 3.0))
            end = start + duration
        else:
            raise ValueError(f"{scene_type} requires end, duration, or end=\"video_end\"")

        scene["start"] = start
        scene["end"] = end
        normalized.append(scene)
        cursor = end

    return normalized


def validate(style: dict[str, Any], timeline: dict[str, Any], style_path: Path, timeline_path: Path) -> list[dict[str, Any]]:
    base = timeline_path.resolve().parent
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg was not found on PATH")
    require_file(style_path.resolve(), "Style JSON")
    head_paths = talking_head_paths(timeline, base)
    if not head_paths:
        raise FileNotFoundError("No talking-head video configured")
    for i, path in enumerate(head_paths):
        require_file(path, f"Talking-head video {i}")
    sub_paths = subtitle_ass_paths(timeline, base)
    if not sub_paths:
        raise FileNotFoundError("No stable-ts ASS configured")
    for i, path in enumerate(sub_paths):
        require_file(path, f"Stable-ts ASS {i}")

    products = products_by_id(timeline)
    scenes = normalize_scene_times(style, timeline, timeline_path)
    if not scenes:
        raise ValueError("Timeline has no scenes")

    prev_end = 0.0
    for i, scene in enumerate(scenes):
        if "type" not in scene:
            raise ValueError(f"Scene {i} is missing type")
        start = float(scene.get("start", -1))
        end = float(scene.get("end", -1))
        if start < 0 or end < 0:
            raise ValueError(f"Scene {i} has negative start/end")
        if end <= start:
            raise ValueError(f"Scene {i} has non-positive duration: {scene}")
        if start < prev_end - 0.001:
            raise ValueError(f"Scene {i} overlaps previous scene: start={start}, previous_end={prev_end}")
        prev_end = end

        if scene["type"] in PRODUCT_SCENE_TYPES:
            product = scene_product(scene, products)
            if not product:
                raise ValueError(f"{scene['type']} requires a valid product_id")
            require_file(scene_image_path(scene, product, base), "Product image")
            logo = resolve(base, product.get("flipkart_logo"))
            if logo is not None:
                require_file(logo, "Flipkart logo")
        if scene["type"] in TALKING_HEAD_AUDIO_SCENE_TYPES:
            video_index = scene_video_index(scene)
            if video_index < 0 or video_index >= len(head_paths):
                raise ValueError(f"Scene {i} has invalid talking-head video index {video_index}")

    allowed_types = set(style.get("scene_type_presets", {}).keys())
    for scene in scenes:
        if allowed_types and scene["type"] not in allowed_types:
            raise ValueError(f"Unknown scene type: {scene['type']}")

    return scenes


def check_encoder(name: str) -> bool:
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "nullsrc=s=64x64:d=1",
                "-vframes", "1", "-c:v", name, "-f", "null", "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def media_duration(path: Path) -> float:
    if shutil.which("ffprobe") is not None:
        try:
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
        except (subprocess.CalledProcessError, ValueError):
            pass

    probe_result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    probe = probe_result.stdout
    for line in probe.splitlines():
        if "Duration:" not in line:
            continue
        stamp = line.split("Duration:", 1)[1].split(",", 1)[0].strip()
        hours, minutes, seconds = stamp.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Could not determine media duration: {path}")


def apply_video_args(
    style: dict[str, Any],
    timeline: dict[str, Any],
    timeline_path: Path,
    vid1: str | None,
    vid2: str | None,
    bridge_duration: float | None,
    bridge_product_id: str | None,
    bridge_image_index: int,
    bridge_transition: str | None,
    use_timeline_scenes: bool,
) -> dict[str, Any]:
    if not vid1 and not vid2:
        return timeline

    updated = dict(timeline)
    existing_videos = list(updated.get("talking_head_videos") or [])
    if vid1:
        if existing_videos:
            existing_videos[0] = vid1
        else:
            existing_videos.append(vid1)
        updated["talking_head_video"] = vid1
    if vid2:
        while len(existing_videos) < 2:
            existing_videos.append(existing_videos[0] if existing_videos else vid1)
        existing_videos[1] = vid2
    updated["talking_head_videos"] = existing_videos

    if not (vid1 and vid2) or use_timeline_scenes:
        return updated

    base = timeline_path.resolve().parent
    vid1_path = resolve(base, vid1)
    vid2_path = resolve(base, vid2)
    assert vid1_path is not None and vid2_path is not None
    dur1 = media_duration(vid1_path)
    dur2 = media_duration(vid2_path)

    products = updated.get("products") or []
    product_id = bridge_product_id or (products[0].get("id") if products else None)
    if not product_id:
        raise ValueError("--vid1/--vid2 auto bridge requires --bridge-product-id or at least one product in the timeline")

    bridge_defaults = style.get("scene_type_presets", {}).get("product_bridge_scene", {})
    bridge_dur = float(bridge_duration or bridge_defaults.get("duration", 4.0))
    transition_name = bridge_transition or bridge_defaults.get(
        "transition",
        style.get("transitions", {}).get("bridge_default", {}).get("name", "dip_to_white"),
    )

    updated["scenes"] = [
        {
            "type": "talking_head_scene",
            "start": 0.0,
            "end": dur1,
            "talking_head_video_index": 0,
            "source_start": 0.0,
        },
        {
            "type": "product_bridge_scene",
            "duration": bridge_dur,
            "transition": transition_name,
            "product_id": product_id,
            "image_index": bridge_image_index,
        },
        {
            "type": "talking_head_scene",
            "start": dur1 + bridge_dur,
            "end": dur1 + bridge_dur + dur2,
            "talking_head_video_index": 1,
            "source_start": 0.0,
        },
    ]
    return updated


def apply_timeline_auto_bridge(style: dict[str, Any], timeline: dict[str, Any], timeline_path: Path) -> dict[str, Any]:
    bridge = timeline.get("auto_bridge")
    if not bridge or bridge.get("enabled") is False:
        return timeline

    videos = list(timeline.get("talking_head_videos") or [])
    if len(videos) < 2:
        raise ValueError("auto_bridge requires talking_head_videos with at least two videos")

    products = timeline.get("products") or []
    product_id = bridge.get("product_id") or (products[0].get("id") if products else None)
    if not product_id:
        raise ValueError("auto_bridge requires product_id or at least one product")

    base = timeline_path.resolve().parent
    vid1 = resolve(base, videos[0])
    vid2 = resolve(base, videos[1])
    assert vid1 is not None and vid2 is not None

    dur1 = float(bridge.get("vid1_duration", media_duration(vid1)))
    dur2 = float(bridge.get("vid2_duration", media_duration(vid2)))
    bridge_defaults = style.get("scene_type_presets", {}).get("product_bridge_scene", {})
    bridge_duration = float(bridge.get("duration", bridge_defaults.get("duration", 4.0)))
    transition = bridge.get(
        "transition",
        bridge_defaults.get("transition", style.get("transitions", {}).get("bridge_default", {}).get("name", "dip_to_white")),
    )
    image_index = int(bridge.get("image_index", 0))
    vid1_source_start = float(bridge.get("vid1_source_start", 0.0))
    vid2_source_start = float(bridge.get("vid2_source_start", 0.0))

    updated = dict(timeline)
    updated["talking_head_video"] = videos[0]
    updated["scenes"] = [
        {
            "type": "talking_head_scene",
            "start": 0.0,
            "end": dur1,
            "talking_head_video_index": 0,
            "source_start": vid1_source_start,
        },
        {
            "type": "product_bridge_scene",
            "duration": bridge_duration,
            "transition": transition,
            "product_id": product_id,
            "image_index": image_index,
        },
        {
            "type": "talking_head_scene",
            "start": dur1 + bridge_duration,
            "end": dur1 + bridge_duration + dur2,
            "talking_head_video_index": 1,
            "source_start": vid2_source_start,
        },
    ]
    return updated


def optional_sfx_events(
    style: dict[str, Any],
    timeline: dict[str, Any],
    scenes: list[dict[str, Any]],
    timeline_path: Path,
) -> list[dict[str, Any]]:
    base = timeline_path.resolve().parent
    transitions = style.get("transitions", {})
    events: list[dict[str, Any]] = []

    for scene in scenes:
        sfx_name: str | None = None
        if scene.get("type") == "product_overlay_float_scene":
            sfx_name = transitions.get("product_float_to_lower_30", {}).get("sfx")
        elif scene.get("transition") == "product_enter":
            sfx_name = transitions.get("product_enter", {}).get("sfx")
        elif scene.get("type") == "flipkart_end_scene":
            sfx_name = style.get("flipkart_end_scene_style", {}).get("optional_sfx")

        path = resolve(base, sfx_name)
        if path is not None and path.exists():
            events.append(
                {
                    "path": path.resolve(),
                    "start": float(scene["start"]),
                    "duration": min(float(scene["end"]) - float(scene["start"]), 1.5),
                }
            )
    return events


def scene_has_talking_audio(scene: dict[str, Any]) -> bool:
    if scene.get("audio") is False:
        return False
    if scene.get("silent") is True:
        return False
    return scene.get("type") in TALKING_HEAD_AUDIO_SCENE_TYPES


def build_scene_audio_filters(
    style: dict[str, Any],
    scenes: list[dict[str, Any]],
    head_count: int,
) -> tuple[list[str], str]:
    sample_rate = int(style.get("output", {}).get("sample_rate", 48000))
    filters: list[str] = []
    labels: list[str] = []

    for idx, scene in enumerate(scenes):
        duration = float(scene["end"]) - float(scene["start"])
        label = f"aud{idx}"
        if scene_has_talking_audio(scene):
            input_idx = scene_video_index(scene)
            if input_idx < 0 or input_idx >= head_count:
                raise ValueError(f"Invalid talking-head video index for audio: {input_idx}")
            source_start = scene_source_start(scene)
            source_end = source_start + duration
            filters.append(
                f"[{input_idx}:a]atrim=start={source_start:.3f}:end={source_end:.3f},"
                f"asetpts=PTS-STARTPTS,aresample={sample_rate},"
                f"apad,atrim=duration={duration:.3f},asetpts=PTS-STARTPTS[{label}]"
            )
        else:
            filters.append(
                f"anullsrc=channel_layout=stereo:sample_rate={sample_rate},"
                f"atrim=duration={duration:.3f},asetpts=PTS-STARTPTS[{label}]"
            )
        labels.append(f"[{label}]")

    if len(labels) == 1:
        return filters, "aud0"

    filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[ascene]")
    return filters, "ascene"


def output_size(style: dict[str, Any]) -> tuple[int, int, int]:
    output = style.get("output", {})
    return int(output.get("width", 720)), int(output.get("height", 960)), int(output.get("fps", 30))


def product_data_region(style: dict[str, Any], out_w: int, out_h: int) -> tuple[int, int, int, int]:
    region = (
        style.get("layout_presets", {})
        .get("product_top_2_3_data_bottom_1_3", {})
        .get("data_region", {})
    )
    if not region:
        region = (
            style.get("layout_presets", {})
            .get("talking_head_with_product_data_strip", {})
            .get("strip", {})
        )
    x = int(region.get("x", 0))
    y = int(region.get("y", int(out_h * 2 / 3)))
    w = int(region.get("w", out_w))
    h = int(region.get("h", out_h - y))
    return x, y, w, h


def pip_geometry(style: dict[str, Any], out_w: int, out_h: int) -> tuple[int, int, int, int, int]:
    pip = style.get("pip_style", {})
    w = int(pip.get("width", 240))
    h = int(pip.get("height", 180))
    aspect = str(pip.get("aspect_mode", pip.get("lock_aspect_ratio", "config_dimensions"))).lower()
    if aspect in {"config", "config_dimensions", "explicit"}:
        pass
    elif aspect in {"4:3", "landscape_4:3", "4:3_landscape"}:
        if w < int(pip.get("height", 180)):
            w = int(pip.get("height", 180))
        h = int(round(w * 3 / 4))
    elif aspect in {"3:4", "portrait_3:4", "3:4_portrait", "source", "source_aspect"}:
        w = int(round(h * 3 / 4))
    w = max(2, (w // 2) * 2)
    h = max(2, (h // 2) * 2)
    border = int(pip.get("border", 4))
    component = style.get("product_data_layout", {}).get("component", {})
    mx = int(component.get("margin_x", pip.get("margin_x", 32)))
    my = int(component.get("margin_y", pip.get("margin_y", 32)))
    region_x, region_y, region_w, _region_h = product_data_region(style, out_w, out_h)
    x = region_x + region_w - w - border * 2 - mx
    y = region_y + my
    return w, h, border, x, y


def ffmpeg_filter_quote(path: Path) -> str:
    text = str(path.resolve()).replace("\\", "/")
    text = text.replace("'", r"\'").replace(":", r"\:")
    return f"'{text}'"


def generate_gradient_png(
    grad_right: int,
    height: int,
    out_path: Path,
) -> bool:
    """Generate a black left→transparent right RGBA gradient PNG.

    Matches the ASS _grad_bg_strips alpha ramp exactly:
      left edge  alpha = 0x77 in ASS  → PNG alpha = 255 - 0x77 = 136  (~53% opaque)
      right edge alpha = 0xFF in ASS  → PNG alpha = 255 - 0xFF = 0    (fully transparent)

    The PNG is grad_right × height pixels. The overlay in the filter graph
    crops it to the active window's y1→y2 height via scale/crop or by
    placing it at the right y offset with overlay.
    """
    # ASS alpha is inverted: 0x00=opaque, 0xFF=transparent
    # PNG alpha is normal:   255=opaque, 0=transparent
    # ASS left  = 0x77  → PNG = 255 - 0x77 = 136
    # ASS right = 0xFF  → PNG = 255 - 0xFF = 0
    alpha_left = 255 - 0x77   # 136
    alpha_right = 0            # 0

    # Build with ffmpeg geq so we have no PIL dependency
    # geq alpha expression: linear ramp from alpha_left at x=0 to alpha_right at x=W-1
    alpha_expr = f"{alpha_left}+({alpha_right}-{alpha_left})*X/(W-1)"
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"color=c=black:s={grad_right}x{height}:r=1:d=1",
            "-vf", f"format=rgba,geq=r=0:g=0:b=0:a='{alpha_expr}'",
            "-frames:v", "1",
            str(out_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        log_step(f"gradient PNG generation failed: {(result.stderr or result.stdout or '').strip()[:200]}")
        return False
    return True


def build_gradient_overlay_filters(
    windows: list[GradientWindow],
    grad_png_input_idx: int,
    in_label: str,
    out_label: str,
    fps: int,
    absolute_ts: bool = False,
) -> list[str]:
    """Build FFmpeg overlay filters for all gradient windows.

    absolute_ts=False (default, per-scene render_video.py):
      PNG stream is trimmed to window duration and its clock reset to t=0.
      Fade st= values are relative to the trimmed stream's own t=0 (= window start).
      overlay enable= uses absolute video time to gate activation.

    absolute_ts=True (full-video subtitle pass in render_scenes.py):
      PNG stream is looped infinitely — never exhausted before the window activates.
      Fade st= values are absolute video timestamps so they fire at the right moment.
      overlay enable= still uses absolute video time.

    Returns a list of filter strings. The last filter maps [in_label] → [out_label].
    """
    if not windows:
        return []

    filters: list[str] = []
    n = len(windows)

    # Split the shared PNG input so each window can consume its own reference
    if n == 1:
        split_labels = ["grad_png0"]
        filters.append(f"[{grad_png_input_idx}:v]loop=loop=-1:size=1:start=0[grad_png0]")
    else:
        split_labels = [f"grad_png{i}" for i in range(n)]
        loop_labels = "".join(f"[grad_loop{i}]" for i in range(n))
        filters.append(f"[{grad_png_input_idx}:v]split={n}{loop_labels}")
        for i in range(n):
            filters.append(f"[grad_loop{i}]loop=loop=-1:size=1:start=0[{split_labels[i]}]")

    current = in_label

    for i, w in enumerate(windows):
        panel_h = max(1, w.y2 - w.y1)
        fade_sec = w.fade_ms / 1000.0
        duration = max(0.0, w.end - w.start)

        crop_label = f"grad_crop{i}"
        if absolute_ts:
            # Looped PNG — crop only, no trim. Fade uses absolute timestamps.
            filters.append(
                f"[{split_labels[i]}]crop={w.grad_right}:{panel_h}:0:0[{crop_label}]"
            )
            faded_label = f"grad_faded{i}"
            fade_in_st = w.start
            fade_out_st = w.end - fade_sec
            if fade_sec > 0 and duration > fade_sec * 2:
                filters.append(
                    f"[{crop_label}]"
                    f"fade=t=in:st={fade_in_st:.3f}:d={fade_sec:.3f}:alpha=1,"
                    f"fade=t=out:st={fade_out_st:.3f}:d={fade_sec:.3f}:alpha=1"
                    f"[{faded_label}]"
                )
            else:
                filters.append(f"[{crop_label}]copy[{faded_label}]")
        else:
            # Trimmed PNG — reset clock to t=0, fade relative to trimmed stream.
            filters.append(
                f"[{split_labels[i]}]"
                f"crop={w.grad_right}:{panel_h}:0:0,"
                f"trim=duration={duration:.3f},setpts=PTS-STARTPTS"
                f"[{crop_label}]"
            )
            faded_label = f"grad_faded{i}"
            if fade_sec > 0 and duration > fade_sec * 2:
                filters.append(
                    f"[{crop_label}]"
                    f"fade=t=in:st=0:d={fade_sec:.3f}:alpha=1,"
                    f"fade=t=out:st={duration - fade_sec:.3f}:d={fade_sec:.3f}:alpha=1"
                    f"[{faded_label}]"
                )
            else:
                filters.append(f"[{crop_label}]copy[{faded_label}]")

        next_label = out_label if i == n - 1 else f"grad_v{i}"
        filters.append(
            f"[{current}][{faded_label}]"
            f"overlay=x=0:y={w.y1}:format=auto:eof_action=pass:"
            f"enable='between(t,{w.start:.3f},{w.end:.3f})'"
            f"[{next_label}]"
        )
        current = next_label

    return filters


def normalized_corner_mode(mode: str) -> str:
    value = str(mode).lower()
    if value in {"top", "top_only"}:
        return "top"
    if value in {"bottom", "bottom_only"}:
        return "bottom"
    return "all"


def rounded_mask_filename(width: int, height: int, radius: int, mode: str = "all") -> str:
    mode_value = normalized_corner_mode(mode)
    suffix = "" if mode_value == "all" else f"_{mode_value}"
    return f"mask_{int(width)}x{int(height)}_r{int(radius)}{suffix}.png"


def rounded_mask_fill_expr(radius: int, mode: str = "all") -> str:
    radius = max(0, int(radius))
    if radius <= 0:
        return "255"

    corners = {"tl", "tr", "bl", "br"}
    mode_value = normalized_corner_mode(mode)
    if mode_value == "top":
        corners = {"tl", "tr"}
    elif mode_value == "bottom":
        corners = {"bl", "br"}

    value = "255"
    checks = [
        ("tl", f"lt(X,{radius})*lt(Y,{radius})", f"hypot(X-{radius},Y-{radius})"),
        ("tr", f"gt(X,W-{radius})*lt(Y,{radius})", f"hypot(X-(W-{radius}),Y-{radius})"),
        ("bl", f"lt(X,{radius})*gt(Y,H-{radius})", f"hypot(X-{radius},Y-(H-{radius}))"),
        ("br", f"gt(X,W-{radius})*gt(Y,H-{radius})", f"hypot(X-(W-{radius}),Y-(H-{radius}))"),
    ]
    for key, condition, distance in reversed(checks):
        if key in corners:
            value = f"if({condition},if(lte({distance},{radius}),255,0),{value})"
    return value


def generate_rounded_rect_mask_ffmpeg(
    width: int,
    height: int,
    radius: int,
    out_path: Path,
    mode: str = "all",
) -> bool:
    expr = rounded_mask_fill_expr(radius, mode)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:r=1:d=1",
            "-vf",
            f"format=gray,geq=lum='{expr}'",
            "-frames:v",
            "1",
            str(out_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        log_step(
            f"Rounded-mask fallback failed for {out_path.name}: {detail[:200]}"
        )
        return False
    return True


def generate_rounded_rect_mask(
    width: int,
    height: int,
    radius: int,
    out_path: Path,
    mode: str = "all",
) -> bool:
    width = max(2, int(width))
    height = max(2, int(height))
    radius = max(0, int(radius))
    radius = min(radius, width // 2, height // 2)
    mode_value = normalized_corner_mode(mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if _cv2 is not None and _np is not None:
        mask = _np.zeros((height, width), dtype=_np.uint8)
        if radius <= 0:
            mask[:] = 255
        else:
            right = width - radius
            bottom = height - radius

            def draw_rect(x0: int, y0: int, x1: int, y1: int) -> None:
                if x1 >= x0 and y1 >= y0:
                    _cv2.rectangle(
                        mask,
                        (int(x0), int(y0)),
                        (int(x1), int(y1)),
                        255,
                        thickness=-1,
                        lineType=_cv2.LINE_AA,
                    )

            def draw_circle(cx: int, cy: int) -> None:
                _cv2.circle(
                    mask,
                    (int(cx), int(cy)),
                    int(radius),
                    255,
                    thickness=-1,
                    lineType=_cv2.LINE_AA,
                )

            if mode_value == "all":
                draw_rect(radius, 0, right, height - 1)
                draw_rect(0, radius, width - 1, bottom)
                draw_circle(radius, radius)
                draw_circle(right, radius)
                draw_circle(radius, bottom)
                draw_circle(right, bottom)
            elif mode_value == "top":
                draw_rect(0, radius, width - 1, height - 1)
                draw_rect(radius, 0, right, radius)
                draw_circle(radius, radius)
                draw_circle(right, radius)
            elif mode_value == "bottom":
                draw_rect(0, 0, width - 1, bottom)
                draw_rect(radius, bottom, right, height - 1)
                draw_circle(radius, bottom)
                draw_circle(right, bottom)

        if _cv2.imwrite(str(out_path), mask):
            return True

    return generate_rounded_rect_mask_ffmpeg(width, height, radius, out_path, mode_value)


def ease_expr(duration: float) -> str:
    return f"(1-pow(1-clip(t/{duration:.6f},0,1),3))"



def bridge_transition_config(style: dict[str, Any], scene: dict[str, Any] | None = None) -> tuple[str, float, str]:
    default_name = style.get("transitions", {}).get("bridge_default", {}).get("name", "dip_to_white")
    name = default_name
    if scene is not None:
        name = scene.get("transition", name)
    if str(name).lower() in {"none", "cut", "hard_cut"}:
        return str(name), 0.0, "white"
    cfg = style.get("transition_library", {}).get(name)
    if cfg is None:
        raise ValueError(f"Unknown bridge transition: {name}")
    transition_type = str(cfg.get("type", "dip"))
    if transition_type == "none":
        return str(name), 0.0, str(cfg.get("color", "white"))
    if transition_type != "dip":
        raise ValueError(f"Unsupported bridge transition type for {name}: {transition_type}")
    return str(name), float(cfg.get("duration", 0.25)), str(cfg.get("color", "white"))


def scene_dip_transition_config(style: dict[str, Any], scene: dict[str, Any] | None) -> tuple[str, float, str] | None:
    """Return dip transition config for scene-level transitions, or None."""
    if scene is None:
        return None
    name = scene.get("transition")
    if not name or str(name) in {"talk_to_pip", "pip_to_talk", "product_float_to_lower_30"}:
        return None
    cfg = style.get("transition_library", {}).get(name)
    if cfg is None:
        return None
    transition_type = str(cfg.get("type", "dip"))
    if transition_type == "none":
        return str(name), 0.0, str(cfg.get("color", "white"))
    if transition_type != "dip":
        raise ValueError(f"Unsupported scene transition type for {name}: {transition_type}")
    return str(name), float(cfg.get("duration", 0.25)), str(cfg.get("color", "white"))


class FilterBuilder:
    def __init__(self, style: dict[str, Any], timeline: dict[str, Any], timeline_path: Path) -> None:
        self.style = style
        self.timeline = timeline
        self.timeline_base = timeline_path.resolve().parent
        self.out_w, self.out_h, self.fps = output_size(style)
        self.products = products_by_id(timeline)
        self.head_paths = talking_head_paths(timeline, self.timeline_base)
        self.filters: list[str] = []
        self.image_inputs: dict[Path, int] = {}
        self.logo_inputs: dict[Path, int] = {}
        # Per video-input index: queue of pre-split stream labels ready for head_full.
        # Populated by presplit_head_inputs() before scene_filter calls.
        self._head_split_labels: dict[int, list[str]] = {}
        self._image_split_labels: dict[int, list[str]] = {}
        # When enabled, head_full reads from a dedicated input that is seeked with
        # input-level -ss to source_start (fast keyframe seek) instead of decoding the
        # shared head video from t=0 and discarding frames with an in-graph trim. This
        # removes the cold-start "boundary stall" where ffmpeg decodes through to the
        # talking-head segment's source_start before it can emit a frame.
        encoder_cfg = style.get("encoder", {})
        self.head_input_seek: bool = bool(
            encoder_cfg.get("head_input_seek", False)
        ) and not bool(encoder_cfg.get("disable_head_input_seek", False))
        self.disable_subtitles_in_main_graph: bool = bool(
            encoder_cfg.get("disable_subtitles_in_main_graph", False)
        )
        self.disable_watermark_in_main_graph: bool = bool(
            encoder_cfg.get("disable_watermark_in_main_graph", False)
        )
        self.disable_final_yuv420p_in_graph: bool = bool(
            encoder_cfg.get("disable_final_yuv420p_in_graph", False)
        )
        self.disable_pair_concat_optimization: bool = bool(
            encoder_cfg.get("disable_pair_concat_optimization", False)
        )
        self.disable_fps_normalization: bool = bool(
            encoder_cfg.get("disable_fps_normalization", False)
        )
        self.scene_prerender_mode: str = str(encoder_cfg.get("scene_prerender_mode", "off"))
        # key (video_index, rounded source_start) -> slot; build_command turns each slot
        # into an appended "-ss <src> -t <dur> -i <path>" input and rewrites the
        # __HSEEK<slot>__ token in the filter graph with the real input index.
        self.head_seek_index: dict[tuple[int, float], int] = {}
        # Per slot: [video_index, source_start, path, max_duration].
        self.head_seek_specs: list[list[Any]] = []
        # When the flipkart end scene is a prebuilt video asset, scene_filter emits a
        # __FLIPKARTVID__ token and build_command appends "-i <path>" + rewrites it,
        # replacing the per-product composed end card (product hero + logo + data strip).
        # _end_video() resolves the asset once (None => fall back to composed scene), and
        # the end scene is then excluded from image/logo input registration.
        self.flipkart_end_video_path: Path | None = None
        self._flipkart_end_video_resolved: bool = False
        # Cache face-crop params per video index; None means detection failed/skipped.
        self._face_pip_crops: dict[int, tuple[int, int, int, int] | None] = {}
        # Set by build_command() before build() if a watermark image is configured.
        self.watermark_input_idx: int = -1
        self.prerender_scene_inputs: dict[Path, int] = {}
        self.prerender_scene_paths: dict[Path, Path] = {}
        # Pre-blurred background cache. Each key is
        # (image_input_idx, target_w, target_h, blur_radius, blur_power) and the
        # value is the ffmpeg input index assigned by build_command() once the
        # PNG has been written to disk. Missing key => fall back to inline blur.
        self.bg_blur_inputs: dict[tuple[int, int, int, int, int], int] = {}
        self.bg_blur_paths: dict[tuple[int, int, int, int, int], Path] = {}
        self.alpha_mask_inputs: dict[tuple[int, int, int, str], int] = {}
        self.alpha_mask_paths: dict[tuple[int, int, int, str], Path] = {}
        self.alpha_mask_use_counts: dict[tuple[int, int, int, str], int] = {}
        self._mask_split_labels: dict[tuple[int, int, int, str], list[str]] = {}

    def presplit_head_inputs(self, scenes: list[dict[str, Any]]) -> None:
        """Pre-split each talking-head video input so head_full can reference it N times.

        [N:v] can only be consumed once in filter_complex. When the same video is used
        by multiple scenes (e.g. product_highlight_pip_scene PiP + talking_head_scene),
        we emit a single split=K at the top and hand out the resulting labels.
        """
        # With input-level seeking each head_full usage gets its own dedicated, pre-seeked
        # input, so the shared-input split (and the in-graph decode-through it forces) is
        # neither needed nor wanted — emitting it would leave unused split outputs.
        if self.head_input_seek:
            return
        from collections import Counter
        counts: Counter[int] = Counter()
        for scene in scenes:
            if scene.get("type") in TALKING_HEAD_AUDIO_SCENE_TYPES or scene.get("type") == "talking_head_scene":
                counts[scene_video_index(scene)] += 1
        for idx, count in counts.items():
            if count == 1:
                self._head_split_labels[idx] = [f"{idx}:v"]
            else:
                labels = [f"hd{idx}_{i}" for i in range(count)]
                split_outs = "".join(f"[{l}]" for l in labels)
                self.filters.append(f"[{idx}:v]split={count}{split_outs}")
                self._head_split_labels[idx] = labels

    def presplit_image_inputs(self, scenes: list[dict[str, Any]]) -> None:
        """Pre-split shared still image inputs so each scene consumes its own branch."""
        from collections import Counter

        counts: Counter[int] = Counter()
        for scene in scenes:
            product = scene_product(scene, self.products)
            scene_type = scene.get("type")
            if scene_type == "flipkart_end_scene" and self._end_video() is not None:
                continue  # end scene reads the prebuilt video, not the product image
            if product and scene_type in PRODUCT_SCENE_TYPES:
                image = scene_image_path(scene, product, self.timeline_base).resolve()
                counts[self.image_inputs[image]] += 1
            elif product and scene_type in PRODUCT_IMAGE_PIP_SCENE_TYPES:
                try:
                    image = scene_image_path(scene, product, self.timeline_base).resolve()
                    if image in self.image_inputs:
                        counts[self.image_inputs[image]] += 1
                except (ValueError, AssertionError):
                    pass
        for idx, count in counts.items():
            if count == 1:
                self._image_split_labels[idx] = [f"{idx}:v"]
            else:
                labels = [f"img{idx}_{i}" for i in range(count)]
                split_outs = "".join(f"[{label}]" for label in labels)
                self.filters.append(f"[{idx}:v]split={count}{split_outs}")
                self._image_split_labels[idx] = labels

    def image_stream_ref(self, image_idx: int) -> str:
        queue = self._image_split_labels.get(image_idx)
        return queue.pop(0) if queue else f"{image_idx}:v"

    def add_image_inputs(self, scenes: list[dict[str, Any]]) -> None:
        end_video_active = self._end_video() is not None
        next_idx = len(self.head_paths)
        for scene in scenes:
            product = scene_product(scene, self.products)
            scene_type = scene.get("type", "")
            if scene_type == "flipkart_end_scene" and end_video_active:
                continue  # end scene reads the prebuilt video, not the product image
            if product and scene_type in (PRODUCT_SCENE_TYPES | PRODUCT_IMAGE_PIP_SCENE_TYPES):
                try:
                    image = scene_image_path(scene, product, self.timeline_base).resolve()
                except (ValueError, AssertionError):
                    continue
                if image not in self.image_inputs:
                    self.image_inputs[image] = next_idx
                    next_idx += 1

        for scene in scenes:
            product = scene_product(scene, self.products)
            if scene.get("type") == "flipkart_end_scene" and end_video_active:
                continue  # no flipkart logo PNG input when using the end video
            if product and scene["type"] in PRODUCT_SCENE_TYPES:
                logo = resolve(self.timeline_base, product.get("flipkart_logo"))
                if logo is not None:
                    logo = logo.resolve()
                    if logo not in self.logo_inputs:
                        self.logo_inputs[logo] = next_idx
                        next_idx += 1

    def register_bg_blur(
        self,
        image_input_idx: int,
        target_w: int,
        target_h: int,
        blur_radius: int,
        blur_power: int,
    ) -> tuple[int, int, int, int, int]:
        """Mark a (image, size, blur-params) tuple as wanting a pre-blurred PNG.
        Input index is assigned later by build_command(); we just record the
        key here so a single pass can dedupe identical requests across scenes."""
        key = (
            int(image_input_idx),
            int(target_w),
            int(target_h),
            int(blur_radius),
            int(blur_power),
        )
        self.bg_blur_inputs.setdefault(key, -1)
        return key

    def bg_blur_input_idx(
        self,
        image_input_idx: int,
        target_w: int,
        target_h: int,
        blur_radius: int,
        blur_power: int,
    ) -> int | None:
        """Return the ffmpeg input index for a pre-blurred background, or None
        when the PNG wasn't produced (caller should fall back to inline blur)."""
        key = (
            int(image_input_idx),
            int(target_w),
            int(target_h),
            int(blur_radius),
            int(blur_power),
        )
        idx = self.bg_blur_inputs.get(key)
        if idx is None or idx < 0:
            return None
        if key not in self.bg_blur_paths:
            return None
        return idx

    def collect_bg_blur_requirements(self, scenes: list[dict[str, Any]]) -> None:
        """Walk scenes and register every static blurred-background variant
        that product_hero_canvas / product_base would otherwise compute inline.
        Called from build_command() after add_image_inputs(), before scene
        filters reference any blurred inputs."""
        hero_layout = self.style.get("layout_presets", {}).get(
            "product_fullscreen_hero_image", {}
        )
        hero_blur = hero_layout.get("background", {}).get("blur", {})
        hero_radius = int(hero_blur.get("radius", 28))
        hero_power = int(hero_blur.get("power", 1))

        base_layout = self.style.get("layout_presets", {}).get(
            "product_top_2_3_data_bottom_1_3", {}
        )
        base_blur = (
            base_layout.get("product_image", {}).get("background", {}).get("blur", {})
        )
        base_radius = int(base_blur.get("radius", 28))
        base_power = int(base_blur.get("power", 1))
        region = base_layout.get("product_region", {})
        rw = int(region.get("w", self.out_w))
        rh = int(region.get("h", self.out_h))

        hero_types = {
            "product_pip_intro_scene",
            "pip_talking_head_no_effect",
            "product_highlight_pip_scene",
            "product_highlight_pip_scene_with_gradient_overlay",
            "product_pip_head_exit_scene",
            "product_bridge_gradient_overlay",
        }
        base_types = {
            "product_scene",
            "product_bridge_scene",
            "pip_scene",
            "talk_to_pip_scene",
            "pip_to_talk_scene",
            "flipkart_end_scene",
        }

        for scene in scenes:
            scene_type = scene.get("type")
            product = scene_product(scene, self.products)
            if not product:
                continue
            try:
                image = scene_image_path(scene, product, self.timeline_base).resolve()
            except (ValueError, AssertionError):
                continue
            image_idx = self.image_inputs.get(image)
            if image_idx is None:
                continue
            if scene_type in hero_types:
                self.register_bg_blur(image_idx, self.out_w, self.out_h, hero_radius, hero_power)
            elif scene_type in base_types:
                self.register_bg_blur(image_idx, self.out_w, self.out_h, base_radius, base_power)
                self.register_bg_blur(image_idx, rw, rh, base_radius, base_power)

    def register_alpha_mask(
        self,
        width: int,
        height: int,
        radius: int,
        mode: str = "all",
    ) -> tuple[int, int, int, str] | None:
        width = max(2, int(width))
        height = max(2, int(height))
        radius = max(0, int(radius))
        radius = min(radius, width // 2, height // 2)
        if radius <= 0:
            return None
        key = (width, height, radius, normalized_corner_mode(mode))
        self.alpha_mask_inputs.setdefault(key, -1)
        self.alpha_mask_use_counts.setdefault(key, 0)
        return key

    def note_alpha_mask_use(
        self,
        width: int,
        height: int,
        radius: int,
        mode: str = "all",
    ) -> None:
        key = self.register_alpha_mask(width, height, radius, mode)
        if key is not None:
            self.alpha_mask_use_counts[key] = self.alpha_mask_use_counts.get(key, 0) + 1

    def presplit_alpha_masks(self) -> None:
        for key, count in sorted(
            self.alpha_mask_use_counts.items(),
            key=lambda item: self.alpha_mask_inputs.get(item[0], -1),
        ):
            idx = self.alpha_mask_inputs.get(key, -1)
            if idx < 0:
                continue
            if count <= 1:
                self._mask_split_labels[key] = [f"{idx}:v"]
                continue
            labels = [f"mask{idx}_{i}" for i in range(count)]
            split_outs = "".join(f"[{label}]" for label in labels)
            self.filters.append(f"[{idx}:v]split={count}{split_outs}")
            self._mask_split_labels[key] = labels

    def mask_stream_ref(self, width: int, height: int, radius: int, mode: str = "all") -> str:
        key = self.register_alpha_mask(width, height, radius, mode)
        if key is None:
            raise RuntimeError("mask_stream_ref called with a non-positive radius")
        queue = self._mask_split_labels.get(key)
        if queue:
            return queue.pop(0)
        idx = self.alpha_mask_inputs.get(key, -1)
        if idx < 0:
            raise RuntimeError(f"Rounded mask input was not assigned for {key}")
        return f"{idx}:v"

    def apply_rounded_alpha_mask(
        self,
        source: str,
        target: str,
        width: int,
        height: int,
        radius: int,
        mode: str = "all",
    ) -> str:
        width = max(2, int(width))
        height = max(2, int(height))
        radius = max(0, int(radius))
        radius = min(radius, width // 2, height // 2)
        if radius <= 0:
            self.filters.append(f"[{source}]format=rgba[{target}]")
            return target

        mask_ref = self.mask_stream_ref(width, height, radius, mode)
        rgba_label = f"{target}_rgba"
        rgb_label = f"{target}_rgb"
        alpha_src_label = f"{target}_alpha_src"
        src_alpha_label = f"{target}_src_alpha"
        mask_label = f"{target}_mask"
        alpha_label = f"{target}_alpha"
        self.filters.extend(
            [
                f"[{source}]format=rgba,split=2[{rgba_label}][{alpha_src_label}]",
                f"[{rgba_label}]format=rgb24[{rgb_label}]",
                f"[{alpha_src_label}]alphaextract[{src_alpha_label}]",
                f"[{mask_ref}]format=gray[{mask_label}]",
                f"[{src_alpha_label}][{mask_label}]blend=all_mode=multiply[{alpha_label}]",
                f"[{rgb_label}][{alpha_label}]alphamerge[{target}]",
            ]
        )
        return target

    def collect_alpha_mask_requirements(self, scenes: list[dict[str, Any]]) -> None:
        pip = self.style.get("pip_style", {})
        pip_w, pip_h, border, _px, _py = pip_geometry(self.style, self.out_w, self.out_h)
        pip_frame_w = pip_w + border * 2
        pip_frame_h = pip_h + border * 2
        pip_radius = int(pip.get("corner_radius", 18))
        pip_scale = pip_w / self.out_w if self.out_w > 0 else 1.0
        animated_radius = (
            0
            if pip_radius <= 0
            else max(1, round(pip_radius / pip_scale)) if pip_scale > 0 else pip_radius
        )

        base_layout = self.style.get("layout_presets", {}).get(
            "product_top_2_3_data_bottom_1_3", {}
        )
        data_region = base_layout.get("data_region", {})
        data_w = int(data_region.get("w", self.out_w))
        data_h = int(data_region.get("h", self.out_h - int(data_region.get("y", 0))))
        data_radius = int(data_region.get("corner_radius", 0))
        data_corner_mode = str(data_region.get("corner_mode", "all"))

        strip_layout = self.style.get("layout_presets", {}).get(
            "talking_head_with_product_data_strip", {}
        )
        strip = strip_layout.get("strip", {})
        strip_y = int(strip.get("y", 640))
        strip_w = int(strip.get("w", self.out_w))
        strip_h = int(strip.get("h", self.out_h - strip_y))
        strip_radius = int(strip.get("corner_radius", 0))
        strip_corner_mode = str(strip.get("corner_mode", "all"))

        static_pip_scene_types = {
            "product_pip_intro_scene",
            "pip_talking_head_no_effect",
            "product_highlight_pip_scene",
            "product_highlight_pip_scene_with_gradient_overlay",
            "product_pip_head_exit_scene",
            "pip_scene",
        }
        base_panel_scene_types = {
            "product_scene",
            "product_bridge_scene",
            "pip_scene",
            "talk_to_pip_scene",
            "pip_to_talk_scene",
            "flipkart_end_scene",
        }

        segments = self.build_segments(scenes)
        for seq, scene in enumerate(segments):
            scene_type = scene.get("type")
            prev_type = segments[seq - 1].get("type") if seq > 0 else None

            if scene_type in static_pip_scene_types:
                self.note_alpha_mask_use(pip_frame_w, pip_frame_h, pip_radius)
                continue

            if scene_type in PRODUCT_IMAGE_PIP_SCENE_TYPES:
                _product = scene_product(scene, self.products)
                if _product:
                    try:
                        _image = scene_image_path(scene, _product, self.timeline_base).resolve()
                        if self.image_inputs.get(_image) is not None:
                            if prev_type in _PRODUCT_HERO_TYPES:
                                self.note_alpha_mask_use(self.out_w, self.out_h, animated_radius)
                            else:
                                self.note_alpha_mask_use(pip_frame_w, pip_frame_h, pip_radius)
                    except (ValueError, AssertionError):
                        pass
                continue

            if scene_type == "talking_head_product_strip_scene":
                self.note_alpha_mask_use(strip_w, strip_h, strip_radius, strip_corner_mode)
                self.note_alpha_mask_use(pip_frame_w, pip_frame_h, pip_radius)
                continue

            if scene_type in base_panel_scene_types:
                self.note_alpha_mask_use(data_w, data_h, data_radius, data_corner_mode)

    def head_full(self, label: str, scene: dict[str, Any]) -> str:
        duration = float(scene["end"]) - float(scene["start"])
        source_start = scene_source_start(scene)
        input_idx = scene_video_index(scene)
        # Consume next pre-split label for this input index (see presplit_head_inputs).
        # Falls back to [N:v] directly when presplit wasn't called (e.g. single use).
        queue = self._head_split_labels.get(input_idx)
        stream_ref = f"[{queue.pop(0)}]" if queue else f"[{input_idx}:v]"
        self.filters.append(
            f"{stream_ref}trim=start={source_start:.3f}:duration={duration:.3f},setpts=PTS-STARTPTS,"
            f"scale={self.out_w}:{self.out_h}:force_original_aspect_ratio=increase,"
            f"crop={self.out_w}:{self.out_h},setsar=1,format=rgba,"
            f"tpad=stop_mode=clone:stop_duration={duration:.3f},"
            f"trim=duration={duration:.3f},setpts=PTS-STARTPTS[{label}]"
        )
        return label

    def blank_scene(self, label: str, duration: float) -> str:
        self.filters.append(
            f"color=c=#101216:s={self.out_w}x{self.out_h}:r={self.fps}:d={duration:.3f},"
            f"format=rgba[{label}]"
        )
        return label

    def product_image_motion(
        self,
        image_idx: int,
        layout: dict[str, Any],
        scene_image_index: int | None = None,
    ) -> dict[str, Any]:
        motion_map = self.style.get("product_image_motion", {})
        if isinstance(motion_map, dict) and scene_image_index is not None:
            motion = motion_map.get(str(scene_image_index)) or motion_map.get(scene_image_index)
            if isinstance(motion, dict):
                return motion
        if isinstance(motion_map, dict):
            motion = motion_map.get(str(image_idx)) or motion_map.get(image_idx)
            if isinstance(motion, dict):
                return motion
        foreground = None
        if "product_image" in layout and isinstance(layout["product_image"], dict):
            foreground = layout["product_image"].get("foreground", {})
        elif "foreground" in layout:
            foreground = layout.get("foreground", {})
        if not isinstance(foreground, dict):
            return {}
        fallback = foreground.get("motion", {}) or {}
        if isinstance(fallback, dict):
            by_index = fallback.get("by_index", {})
            if isinstance(by_index, dict):
                indexed = by_index.get(str(image_idx)) or by_index.get(image_idx)
                if isinstance(indexed, dict):
                    return indexed
            return fallback.get("default", fallback)
        return {}

    def product_motion_expr(self, duration: float, motion: dict[str, Any]) -> tuple[str, str, str]:
        if duration <= 0:
            duration = 1.0
        if any(name in motion for name in ("x_translate", "y_translate", "scale")):
            x_start = float(motion.get("x_translate", {}).get("start", 0.0))
            x_end = float(motion.get("x_translate", {}).get("end", x_start))
            y_start = float(motion.get("y_translate", {}).get("start", 0.0))
            y_end = float(motion.get("y_translate", {}).get("end", y_start))
            scale_start = float(motion.get("scale", {}).get("start", 1.0))
            scale_end = float(motion.get("scale", {}).get("end", scale_start))
            x_expr = f"round((W-w)/2 + {x_start:.3f} + ({x_end - x_start:.3f})*t/{duration:.6f})"
            y_expr = f"round((H-h)/2 - {y_start:.3f} - ({y_end - y_start:.3f})*t/{duration:.6f})"
            scale_expr = f"({scale_start:.6f} + ({scale_end - scale_start:.6f})*t/{duration:.6f})"
            return x_expr, y_expr, scale_expr
        if any(name in motion for name in ("x_amplitude", "y_amplitude", "period_seconds")):
            x_expr = f"(W-w)/2+{float(motion.get('x_amplitude', 0)):.3f}*sin(2*3.14159*t/{float(motion.get('period_seconds', duration)):.3f})"
            y_expr = f"(H-h)/2+{float(motion.get('y_amplitude', 0)):.3f}*cos(2*3.14159*t/{float(motion.get('period_seconds', duration)):.3f})"
            return x_expr, y_expr, "1.0"
        return "(W-w)/2", "(H-h)/2", "1.0"

    def product_base(self, label: str, image_idx: int, duration: float, *, animate_in: bool = False, scene_image_index: int = 0, motion_index: int | None = None) -> str:
        image_ref = self.image_stream_ref(image_idx)
        layout = self.style["layout_presets"]["product_top_2_3_data_bottom_1_3"]
        region = layout["product_region"]
        data_region = layout.get("data_region", {})
        rx, ry, rw, rh = int(region["x"]), int(region["y"]), int(region["w"]), int(region["h"])
        dx = int(data_region.get("x", 0))
        dy = int(data_region.get("y", rh))
        dw = int(data_region.get("w", self.out_w))
        dh = int(data_region.get("h", self.out_h - dy))
        data_color = str(data_region.get("color", "#101216"))
        data_alpha = float(data_region.get("alpha", 1.0))
        data_radius = int(data_region.get("corner_radius", 0))
        data_corner_mode = str(data_region.get("corner_mode", "all"))
        bg = layout["product_image"]["background"]
        fg = layout["product_image"]["foreground"]
        blur = bg.get("blur", {})
        blur_radius = int(blur.get("radius", 28))
        blur_power = int(blur.get("power", 1))
        max_w = int(fg.get("max_w", rw - 60))
        max_h = int(fg.get("max_h", rh - 40))

        motion = self.product_image_motion(image_idx, layout, motion_index if motion_index is not None else scene_image_index)
        motion_x, motion_y, motion_scale = self.product_motion_expr(duration, motion)

        canvas_blur_idx = self.bg_blur_input_idx(
            image_idx, self.out_w, self.out_h, blur_radius, blur_power
        )
        inner_blur_idx = self.bg_blur_input_idx(
            image_idx, rw, rh, blur_radius, blur_power
        )

        if canvas_blur_idx is not None and inner_blur_idx is not None:
            self.filters.extend(
                [
                    f"[{canvas_blur_idx}:v]trim=duration={duration:.3f},setpts=PTS-STARTPTS,format=rgba[{label}_canvas]",
                    f"[{inner_blur_idx}:v]trim=duration={duration:.3f},setpts=PTS-STARTPTS,format=rgba[{label}_bg]",
                    f"[{image_ref}]trim=duration={duration:.3f},setpts=PTS-STARTPTS,"
                    f"scale='if(gt(iw/ih,{max_w}/{max_h}),trunc({max_w}*{motion_scale}/2)*2,-2)':"
                    f"'if(gt(iw/ih,{max_w}/{max_h}),-2,trunc({max_h}*{motion_scale}/2)*2)':"
                    f"eval=frame,setsar=1,format=rgba[{label}_fg]",
                    f"[{label}_bg][{label}_fg]overlay=x='{motion_x}':y='{motion_y}':eval=frame:format=auto[{label}_top]",
                    f"[{label}_canvas][{label}_top]overlay={rx}:{ry}:format=auto[{label}_base0]",
                    f"color=c={data_color}@{data_alpha}:s={dw}x{dh}:r={self.fps}:d={duration:.3f},format=rgba[{label}_data_raw]",
                ]
            )
            self.apply_rounded_alpha_mask(
                f"{label}_data_raw",
                f"{label}_data_rounded",
                dw,
                dh,
                data_radius,
                data_corner_mode,
            )
            self.filters.append(
                f"[{label}_base0][{label}_data_rounded]overlay={dx}:{dy}:format=auto,format=rgb24,format=rgba[{label}]"
            )
            return label

        self.filters.extend(
            [
                f"[{image_ref}]trim=duration={duration:.3f},setpts=PTS-STARTPTS,"
                f"split=3[{label}_raw_canvas][{label}_raw_bg][{label}_raw_fg]",
                f"[{label}_raw_canvas]"
                f"scale={self.out_w}:{self.out_h}:force_original_aspect_ratio=increase,"
                f"crop={self.out_w}:{self.out_h},setsar=1,boxblur={blur_radius}:{blur_power},format=rgba[{label}_canvas]",
                f"[{label}_raw_bg]"
                f"scale={rw}:{rh}:force_original_aspect_ratio=increase,crop={rw}:{rh},"
                f"setsar=1,boxblur={blur_radius}:{blur_power},format=rgba[{label}_bg]",
                f"[{label}_raw_fg]"
                f"scale='if(gt(iw/ih,{max_w}/{max_h}),trunc({max_w}*{motion_scale}/2)*2,-2)':"
                f"'if(gt(iw/ih,{max_w}/{max_h}),-2,trunc({max_h}*{motion_scale}/2)*2)':"
                f"eval=frame,setsar=1,format=rgba[{label}_fg]",
                f"[{label}_bg][{label}_fg]overlay=x='{motion_x}':y='{motion_y}':eval=frame:format=auto[{label}_top]",
                f"[{label}_canvas][{label}_top]overlay={rx}:{ry}:format=auto[{label}_base0]",
                f"color=c={data_color}@{data_alpha}:s={dw}x{dh}:r={self.fps}:d={duration:.3f},format=rgba[{label}_data_raw]",
            ]
        )
        self.apply_rounded_alpha_mask(
            f"{label}_data_raw",
            f"{label}_data_rounded",
            dw,
            dh,
            data_radius,
            data_corner_mode,
        )
        self.filters.append(
            f"[{label}_base0][{label}_data_rounded]overlay={dx}:{dy}:format=auto,format=rgb24,format=rgba[{label}]"
        )
        return label

    def product_hero_canvas(self, label: str, image_idx: int, duration: float, scene_image_index: int = 0, motion_index: int | None = None) -> str:
        image_ref = self.image_stream_ref(image_idx)
        layout = self.style["layout_presets"]["product_fullscreen_hero_image"]
        bg_cfg = layout["background"]
        fg_cfg = layout["foreground"]
        blur = bg_cfg.get("blur", {})
        blur_radius = int(blur.get("radius", 28))
        blur_power = int(blur.get("power", 1))
        max_w = int(fg_cfg.get("max_w", self.out_w - 60))
        max_h = int(fg_cfg.get("max_h", self.out_h - 90))

        motion = self.product_image_motion(image_idx, layout, motion_index if motion_index is not None else scene_image_index)
        motion_x, motion_y, motion_scale = self.product_motion_expr(duration, motion)

        bg_idx = self.bg_blur_input_idx(
            image_idx, self.out_w, self.out_h, blur_radius, blur_power
        )
        if bg_idx is not None:
            self.filters.extend([
                f"[{bg_idx}:v]trim=duration={duration:.3f},setpts=PTS-STARTPTS,format=rgba[{label}_bg]",
                f"[{image_ref}]trim=duration={duration:.3f},setpts=PTS-STARTPTS,"
                f"scale='if(gt(iw/ih,{max_w}/{max_h}),trunc({max_w}*{motion_scale}/2)*2,-2)':"
                f"'if(gt(iw/ih,{max_w}/{max_h}),-2,trunc({max_h}*{motion_scale}/2)*2)':"
                f"eval=frame,setsar=1,format=rgba[{label}_fg]",
                f"[{label}_bg][{label}_fg]overlay=x='{motion_x}':y='{motion_y}':eval=frame:format=auto,format=rgba[{label}]",
            ])
            return label

        self.filters.extend([
            f"[{image_ref}]trim=duration={duration:.3f},setpts=PTS-STARTPTS,split=2[{label}_raw_a][{label}_raw_b]",
            f"[{label}_raw_a]scale={self.out_w}:{self.out_h}:force_original_aspect_ratio=increase,"
            f"crop={self.out_w}:{self.out_h},setsar=1,"
            f"boxblur={blur_radius}:{blur_power},format=rgba[{label}_bg]",
            f"[{label}_raw_b]scale='if(gt(iw/ih,{max_w}/{max_h}),trunc({max_w}*{motion_scale}/2)*2,-2)':"
            f"'if(gt(iw/ih,{max_w}/{max_h}),-2,trunc({max_h}*{motion_scale}/2)*2)':"
            f"eval=frame,setsar=1,format=rgba[{label}_fg]",
            f"[{label}_bg][{label}_fg]overlay=x='{motion_x}':y='{motion_y}':eval=frame:format=auto,format=rgba[{label}]",
        ])
        return label

    def _get_face_pip_crop(
        self, scene: dict[str, Any] | None, pip_w: int, pip_h: int
    ) -> tuple[int, int, int, int] | None:
        return None

    def _pip_raw_filter(
        self,
        src_label: str,
        out_label: str,
        pip_w: int,
        pip_h: int,
        face_crop: tuple[int, int, int, int] | None,
    ) -> str:
        if face_crop is not None:
            cx, cy, cw, ch = face_crop
            return (
                f"[{src_label}]crop={cw}:{ch}:{cx}:{cy},"
                f"scale={pip_w}:{pip_h},setsar=1[{out_label}]"
            )
        return (
            f"[{src_label}]scale={pip_w}:{pip_h}:force_original_aspect_ratio=increase,"
            f"crop={pip_w}:{pip_h},setsar=1[{out_label}]"
        )

    def static_pip(
        self,
        base: str,
        head: str,
        out_label: str,
        duration: float,
        scene: dict[str, Any] | None = None,
    ) -> str:
        pip = self.style.get("pip_style", {})
        pip_w, pip_h, border, px, py = pip_geometry(self.style, self.out_w, self.out_h)
        corner_radius = int(pip.get("corner_radius", 18))
        frame_w = pip_w + border * 2
        frame_h = pip_h + border * 2
        x_expr = str(px)
        y_expr = str(py)
        if scene and scene.get("pip_exit_transition") in {"slide_out", "slide_out_right"}:
            trans = self.style.get("transitions", {}).get("pip_slide_out", {})
            exit_dur = min(float(scene.get("pip_exit_duration", trans.get("duration", 0.6))), duration)
            exit_dur = max(exit_dur, 0.001)
            exit_start = max(duration - exit_dur, 0.0)
            ease = f"(1-pow(1-clip((t-{exit_start:.6f})/{exit_dur:.6f},0,1),3))"
            offscreen_x = self.out_w + border * 2
            x_expr = f"({px}+({offscreen_x - px})*{ease})"
        face_crop = self._get_face_pip_crop(scene, pip_w, pip_h)
        self.filters.extend(
            [
                self._pip_raw_filter(head, f"{out_label}_pipraw", pip_w, pip_h, face_crop),
                f"[{out_label}_pipraw]pad={frame_w}:{frame_h}:{border}:{border}:color=white[{out_label}_pipframe]",
            ]
        )
        self.apply_rounded_alpha_mask(
            f"{out_label}_pipframe",
            f"{out_label}_frame",
            frame_w,
            frame_h,
            corner_radius,
        )
        self.filters.extend(
            [
                f"[{out_label}_frame]format=rgba[{out_label}_front]",
                f"[{base}][{out_label}_front]overlay=x='{x_expr}':y='{y_expr}':eval=frame:format=auto[{out_label}]",
            ]
        )
        return out_label

    def product_image_pip(self, base: str, image_idx: int, out_label: str, duration: float) -> str:
        """Overlay a static product image in the PiP corner on top of a fullscreen base."""
        pip = self.style.get("pip_style", {})
        pip_w, pip_h, border, px, py = pip_geometry(self.style, self.out_w, self.out_h)
        corner_radius = int(pip.get("corner_radius", 18))
        frame_w = pip_w + border * 2
        frame_h = pip_h + border * 2
        image_ref = self.image_stream_ref(image_idx)
        self.filters.extend([
            f"[{image_ref}]trim=duration={duration:.3f},setpts=PTS-STARTPTS,"
            f"scale={pip_w}:{pip_h}:force_original_aspect_ratio=decrease,"
            f"pad={pip_w}:{pip_h}:(ow-iw)/2:(oh-ih)/2:color=white,"
            f"setsar=1[{out_label}_pipraw]",
            f"[{out_label}_pipraw]pad={frame_w}:{frame_h}:{border}:{border}:color=white[{out_label}_pipframe]",
        ])
        self.apply_rounded_alpha_mask(
            f"{out_label}_pipframe",
            f"{out_label}_frame",
            frame_w,
            frame_h,
            corner_radius,
        )
        self.filters.extend([
            f"[{out_label}_frame]format=rgba[{out_label}_front]",
            f"[{base}][{out_label}_front]overlay=x='{px}':y='{py}':eval=frame:format=auto[{out_label}]",
        ])
        return out_label

    def animated_head(
        self,
        base: str,
        head: str,
        out_label: str,
        mode: str,
        duration: float,
        scene: dict[str, Any] | None = None,
    ) -> str:
        pip = self.style.get("pip_style", {})
        pip_w, pip_h, border, px, py = pip_geometry(self.style, self.out_w, self.out_h)
        trans = self.style.get("transitions", {})
        trans_name = "talk_to_pip" if mode == "to_pip" else "pip_to_talk"
        anim_dur = min(float(trans.get(trans_name, {}).get("duration", 0.8)), duration)

        p = f"clip(t/{anim_dur:.6f},0,1)"
        ease = f"(1-pow(1-{p},3))"

        if mode == "to_pip":
            # Head video scales from full canvas down to pip size and slides to the corner.
            w_expr = f"max(2,2*trunc(({self.out_w}*(1-{ease})+{pip_w}*{ease})/2))"
            h_expr = f"max(2,2*trunc(({self.out_h}*(1-{ease})+{pip_h}*{ease})/2))"
            x_expr = f"round({px}*{ease})"
            y_expr = f"round({py}*{ease})"
            self.filters.extend([
                f"[{head}]scale=w='{w_expr}':h='{h_expr}':eval=frame,"
                f"setsar=1,format=rgba[{out_label}_scaled]",
                f"[{base}][{out_label}_scaled]overlay=x='{x_expr}':y='{y_expr}'"
                f":eval=frame:format=auto[{out_label}]",
            ])
        else:
            # Zoom out from the centre of the head frame — simulates the pip box expanding
            # to fill the screen. crop+scale keeps the output at a fixed out_w×out_h.
            inv = f"pow(1-{p},3)"  # 1 at t=0, 0 at t≥anim_dur
            cx = (self.out_w - pip_w) // 2
            cy = (self.out_h - pip_h) // 2
            crop_x = f"round({cx}*({inv}))"
            crop_y = f"round({cy}*({inv}))"
            crop_w = f"max(4,2*trunc(({pip_w}*({inv})+{self.out_w}*(1-({inv})))/2))"
            crop_h = f"max(4,2*trunc(({pip_h}*({inv})+{self.out_h}*(1-({inv})))/2))"
            self.filters.extend([
                f"[{head}]crop=x='{crop_x}':y='{crop_y}':w='{crop_w}':h='{crop_h}':eval=frame,"
                f"scale={self.out_w}:{self.out_h}:flags=bilinear,setsar=1[{out_label}_expanded]",
                f"[{base}][{out_label}_expanded]overlay=0:0:format=auto[{out_label}]",
            ])
        return out_label

    def animated_product_to_pip(
        self,
        base: str,
        image_idx: int,
        out_label: str,
        duration: float,
    ) -> str:
        """Product image shrinks from full-screen to the PiP badge corner over anim_dur."""
        pip = self.style.get("pip_style", {})
        pip_w, pip_h, border, px, py = pip_geometry(self.style, self.out_w, self.out_h)
        corner_radius = int(pip.get("corner_radius", 18))
        trans = self.style.get("transitions", {})
        anim_dur = min(float(trans.get("talk_to_pip", {}).get("duration", 0.8)), duration)
        image_ref = self.image_stream_ref(image_idx)

        p = f"clip(t/{anim_dur:.6f},0,1)"
        ease = f"(1-pow(1-{p},3))"
        w_expr = f"max(2,2*trunc(({self.out_w}*(1-{ease})+{pip_w}*{ease})/2))"
        h_expr = f"max(2,2*trunc(({self.out_h}*(1-{ease})+{pip_h}*{ease})/2))"
        x_expr = f"round({px}*{ease})"
        y_expr = f"round({py}*{ease})"
        pip_scale = pip_w / self.out_w if self.out_w > 0 else 1.0
        adjusted_radius = (
            0
            if corner_radius <= 0
            else max(1, round(corner_radius / pip_scale)) if pip_scale > 0 else corner_radius
        )
        self.filters.extend([
            f"[{image_ref}]trim=duration={duration:.3f},setpts=PTS-STARTPTS,"
            f"scale={self.out_w}:{self.out_h}:force_original_aspect_ratio=increase,"
            f"crop={self.out_w}:{self.out_h},setsar=1[{out_label}_prodraw]",
        ])
        self.apply_rounded_alpha_mask(
            f"{out_label}_prodraw",
            f"{out_label}_prodcorner",
            self.out_w,
            self.out_h,
            adjusted_radius,
        )
        self.filters.extend([
            f"[{out_label}_prodcorner]scale=w='{w_expr}':h='{h_expr}':eval=frame,"
            f"setsar=1,format=rgba[{out_label}_prodshrink]",
            f"[{base}][{out_label}_prodshrink]overlay=x='{x_expr}':y='{y_expr}'"
            f":eval=frame:format=auto[{out_label}]",
        ])
        return out_label

    def product_data_strip(self, base: str, out_label: str, duration: float) -> str:
        layout = self.style.get("layout_presets", {}).get("talking_head_with_product_data_strip", {})
        strip = layout.get("strip", {})
        x = int(strip.get("x", 0))
        y = int(strip.get("y", 640))
        w = int(strip.get("w", self.out_w))
        h = int(strip.get("h", self.out_h - y))
        color = str(strip.get("color", "black"))
        alpha = float(strip.get("alpha", 0.86))
        radius = int(strip.get("corner_radius", 0))
        corner_mode = str(strip.get("corner_mode", "all"))
        transition = str(strip.get("transition", "none"))
        transition_duration = max(0.0, min(float(strip.get("transition_duration", 0.0)), duration))
        box_raw = f"{out_label}_box_raw"
        box_rounded = f"{out_label}_box_rounded"
        box_label = box_rounded
        self.filters.append(
            f"color=c={color}@{alpha}:s={w}x{h}:r={self.fps}:d={duration:.3f},format=rgba[{box_raw}]"
        )
        self.apply_rounded_alpha_mask(box_raw, box_rounded, w, h, radius, corner_mode)
        if transition == "cross_dissolve" and transition_duration > 0:
            box_label = f"{out_label}_box"
            self.filters.append(
                f"[{box_rounded}]fade=t=in:st=0:d={transition_duration:.3f}:alpha=1[{box_label}]"
            )
        y_expr = str(y)
        if transition == "slide_up" and transition_duration > 0:
            ease = ease_expr(transition_duration)
            y_expr = f"round({y}+({h})*(1-{ease}))"
        self.filters.append(f"[{base}][{box_label}]overlay=x={x}:y='{y_expr}':eval=frame:format=auto[{out_label}]")
        return out_label

    def product_badge(self, base: str, image_idx: int, out_label: str, duration: float) -> str:
        image_ref = self.image_stream_ref(image_idx)
        pip = self.style.get("pip_style", {})
        strip_cfg = (
            self.style.get("layout_presets", {})
            .get("talking_head_with_product_data_strip", {})
            .get("strip", {})
        )
        badge_cfg = (
            self.style.get("layout_presets", {})
            .get("talking_head_with_product_data_strip", {})
            .get("product_badge", {})
        )
        pip_w, pip_h, border, px, py = pip_geometry(self.style, self.out_w, self.out_h)
        frame_w = pip_w + border * 2
        frame_h = pip_h + border * 2
        alpha = float(pip.get("shadow_alpha", 0.28))
        blur = int(pip.get("shadow_blur", 12))
        corner_radius = int(pip.get("corner_radius", 18))
        background = str(badge_cfg.get("background_color", "white"))
        transition_duration = 0.0
        transition = str(strip_cfg.get("transition", "none"))
        if transition in {"cross_dissolve", "slide_up"}:
            transition_duration = max(0.0, min(float(strip_cfg.get("transition_duration", 0.0)), duration))
        fade = f",fade=t=in:st=0:d={transition_duration:.3f}:alpha=1" if transition == "cross_dissolve" and transition_duration > 0 else ""
        front_label = f"{out_label}_front"

        filters = [
            f"[{image_ref}]trim=duration={duration:.3f},setpts=PTS-STARTPTS,"
            f"scale={pip_w}:{pip_h}:force_original_aspect_ratio=decrease,"
            f"pad={pip_w}:{pip_h}:(ow-iw)/2:(oh-ih)/2:color={background},setsar=1,format=rgba[{out_label}_raw]",
            f"[{out_label}_raw]pad={frame_w}:{frame_h}:{border}:{border}:color=white[{out_label}_unrounded]",
        ]
        self.filters.extend(filters)
        self.apply_rounded_alpha_mask(
            f"{out_label}_unrounded",
            f"{out_label}_frame",
            frame_w,
            frame_h,
            corner_radius,
        )
        self.filters.extend(
            [
                f"[{out_label}_frame]format=rgba,split=2[{out_label}_shsrc][{out_label}_front]",
                f"[{out_label}_shsrc]colorchannelmixer=rr=0:gg=0:bb=0:aa={alpha},boxblur={blur}:1{fade}[{out_label}_shadow]",
            ]
        )
        if fade:
            front_label = f"{out_label}_front_faded"
            self.filters.append(f"[{out_label}_front]{fade.lstrip(',')}[{front_label}]")
        y_expr = str(py)
        if transition == "slide_up" and transition_duration > 0:
            strip_h = int(strip_cfg.get("h", self.out_h - int(strip_cfg.get("y", 640))))
            ease = ease_expr(transition_duration)
            y_expr = f"round({py}+({strip_h})*(1-{ease}))"
        self.filters.extend(
            [
                f"[{base}][{out_label}_shadow]overlay=x={px + 10}:y='{y_expr}+12':eval=frame:format=auto[{out_label}_s0]",
                f"[{out_label}_s0][{front_label}]overlay=x={px}:y='{y_expr}':eval=frame:format=auto[{out_label}]",
            ]
        )
        return out_label

    def floating_product(self, base: str, image_idx: int, out_label: str, duration: float) -> str:
        image_ref = self.image_stream_ref(image_idx)
        trans = self.style.get("transitions", {}).get("product_float_to_lower_30", {})
        card_w = int(trans.get("card_width", 320))
        card_h = int(trans.get("card_height", 320))
        anim_dur = min(float(trans.get("duration", 0.8)), duration)
        final_center_ratio = float(trans.get("final_center_y_ratio", 0.7))
        final_top = self.out_h * final_center_ratio - card_h / 2
        ease = ease_expr(anim_dur)
        y_expr = f"({self.out_h}-({self.out_h - final_top:.3f})*{ease})"
        x_expr = f"({(self.out_w - card_w) / 2:.3f})"
        blur = int(trans.get("shadow_blur", 18))
        alpha = float(trans.get("shadow_alpha", 0.3))
        self.filters.extend(
            [
                f"[{image_ref}]trim=duration={duration:.3f},setpts=PTS-STARTPTS,"
                f"scale={card_w - 24}:{card_h - 24}:force_original_aspect_ratio=decrease,"
                f"pad={card_w}:{card_h}:(ow-iw)/2:(oh-ih)/2:color=white,setsar=1,format=rgba[{out_label}_card]",
                f"[{out_label}_card]split=2[{out_label}_shsrc][{out_label}_front]",
                f"[{out_label}_shsrc]colorchannelmixer=rr=0:gg=0:bb=0:aa={alpha},boxblur={blur}:1[{out_label}_shadow]",
                f"[{base}][{out_label}_shadow]overlay=x='{x_expr}+10':y='{y_expr}+14':eval=frame:format=auto[{out_label}_s0]",
                f"[{out_label}_s0][{out_label}_front]overlay=x='{x_expr}':y='{y_expr}':eval=frame:format=auto[{out_label}]",
            ]
        )
        return out_label

    def _end_video(self) -> Path | None:
        """Resolve the flipkart end-card video asset once (config + existence).

        Returns the resolved Path, or None when disabled/missing (fall back to the
        composed flipkart_scene). Cached so input-registration and scene_filter agree.
        """
        if not self._flipkart_end_video_resolved:
            self._flipkart_end_video_resolved = True
            end_video = self.style.get("flipkart_end_scene_style", {}).get(
                "end_video", "assets/flipkart_logo.mp4"
            )
            if end_video:
                for search_base in (self.timeline_base, Path.cwd()):
                    candidate = resolve(search_base, str(end_video))
                    if candidate is not None and candidate.exists():
                        self.flipkart_end_video_path = candidate
                        break
        return self.flipkart_end_video_path

    def flipkart_end_video_scene(self, out_label: str, duration: float) -> str | None:
        """Use a prebuilt end-card video (default assets/flipkart_logo.mp4) as the final
        scene instead of compositing product hero + logo + data strip per render.

        Returns the scene label, or None to fall back to the composed flipkart_scene
        (e.g. when the asset is disabled or missing).
        """
        if self._end_video() is None:
            return None
        # Scale-to-cover guard in case the asset isn't exactly the output size; the
        # seg wrapper adds the dip-to-white fade-in, so none is baked here.
        self.filters.append(
            f"[__FLIPKARTVID__:v]trim=duration={duration:.3f},setpts=PTS-STARTPTS,"
            f"scale={self.out_w}:{self.out_h}:force_original_aspect_ratio=increase,"
            f"crop={self.out_w}:{self.out_h},setsar=1,format=rgba[{out_label}]"
        )
        return out_label

    def scene_prerender_enabled(self, scene: dict[str, Any], seq: int) -> bool:
        if self.scene_prerender_mode == "all_scenes":
            return True
        if self.scene_prerender_mode == "scene0":
            return bool(scene.get("prerendered_video_path"))
        return False

    def collect_prerender_scene_inputs(self, scenes: list[dict[str, Any]]) -> None:
        for scene in scenes:
            prerendered = scene.get("prerendered_video_path")
            if not prerendered:
                continue
            path = resolve(self.timeline_base, str(prerendered))
            if path is None or not path.exists():
                continue
            resolved = path.resolve()
            if resolved not in self.prerender_scene_inputs:
                self.prerender_scene_inputs[resolved] = -1
                self.prerender_scene_paths[resolved] = resolved

    def prerendered_scene_video_scene(self, scene: dict[str, Any], seq: int, duration: float) -> str | None:
        if not self.scene_prerender_enabled(scene, seq):
            return None
        prerendered = scene.get("prerendered_video_path")
        if not prerendered:
            return None
        path = resolve(self.timeline_base, str(prerendered))
        if path is None or not path.exists():
            return None
        resolved = path.resolve()
        input_idx = self.prerender_scene_inputs.get(resolved)
        if input_idx is None or input_idx < 0:
            return None
        out_label = f"s{seq}_prerendered"
        self.filters.append(
            f"[{input_idx}:v]trim=duration={duration:.3f},setpts=PTS-STARTPTS,"
            f"scale={self.out_w}:{self.out_h}:force_original_aspect_ratio=increase,"
            f"crop={self.out_w}:{self.out_h},setsar=1,format=rgba[{out_label}]"
        )
        return out_label

    def flipkart_scene(self, base: str, scene: dict[str, Any], product: dict[str, Any], out_label: str, duration: float) -> str:
        logo = resolve(self.timeline_base, product.get("flipkart_logo"))
        if logo is None:
            self.filters.append(f"[{base}]null[{out_label}]")
            return out_label
        logo_idx = self.logo_inputs[logo.resolve()]
        fit = self.style.get("flipkart_end_scene_style", {}).get("logo_fit", {})
        max_w = int(fit.get("max_w", self.out_w))
        max_h = int(fit.get("max_h", self.out_h))
        x = "(W-w)/2" if fit.get("x", "center") == "center" else str(int(fit.get("x", 0)))
        y = "(H-h)/2" if fit.get("y", "center") == "center" else str(int(fit.get("y", 0)))
        self.filters.extend(
            [
                f"[{logo_idx}:v]trim=duration={duration:.3f},setpts=PTS-STARTPTS,"
                f"scale={max_w}:{max_h}:force_original_aspect_ratio=decrease,setsar=1,format=rgba[{out_label}_logo]",
                f"[{base}][{out_label}_logo]overlay={x}:{y}:format=auto[{out_label}]",
            ]
        )
        return out_label

    def scene_filter(
        self,
        scene: dict[str, Any],
        seq: int,
        prev_scene: dict[str, Any] | None = None,
    ) -> str:
        start = float(scene["start"])
        end = float(scene["end"])
        duration = end - start
        scene_type = scene["type"]

        if scene_type == "blank_scene":
            return self.blank_scene(f"s{seq}_blank", duration)

        prerendered_label = self.prerendered_scene_video_scene(scene, seq, duration)
        if prerendered_label is not None:
            return prerendered_label

        if scene_type in PRODUCT_IMAGE_PIP_SCENE_TYPES:
            head = self.head_full(f"s{seq}_head", scene)
            _product = scene_product(scene, self.products)
            if _product:
                try:
                    _image = scene_image_path(scene, _product, self.timeline_base).resolve()
                    _image_idx = self.image_inputs.get(_image)
                    if _image_idx is not None:
                        prev_type = prev_scene.get("type") if prev_scene else None
                        if prev_type in _PRODUCT_HERO_TYPES:
                            return self.animated_product_to_pip(
                                head, _image_idx, f"s{seq}_prod_pip", duration
                            )
                        return self.product_image_pip(head, _image_idx, f"s{seq}_prod_pip", duration)
                except (ValueError, AssertionError):
                    pass
            return head

        if scene_type == "flipkart_end_scene":
            end_label = self.flipkart_end_video_scene(f"s{seq}_flipkart", duration)
            if end_label is not None:
                return end_label

        product = scene_product(scene, self.products)
        if not product:
            raise ValueError(f"{scene_type} requires product_id")
        image = scene_image_path(scene, product, self.timeline_base).resolve()
        image_idx = self.image_inputs[image]

        if scene_type == "product_overlay_float_scene":
            head = self.head_full(f"s{seq}_head", scene)
            return self.floating_product(head, image_idx, f"s{seq}_float", duration)

        if scene_type == "talking_head_product_strip_scene":
            head = self.head_full(f"s{seq}_head", scene)
            strip = self.product_data_strip(head, f"s{seq}_strip", duration)
            return self.product_badge(strip, image_idx, f"s{seq}_badge", duration)

        scene_image_index = int(scene.get("image_index", 0))
        motion_index = int(scene["motion_index"]) if "motion_index" in scene else None

        if scene_type in {"product_pip_intro_scene", "pip_talking_head_no_effect"}:
            product_base = self.product_hero_canvas(f"s{seq}_hero", image_idx, duration, scene_image_index, motion_index=motion_index)
            head = self.head_full(f"s{seq}_head", scene)
            return self.static_pip(product_base, head, f"s{seq}_intro_pip", duration, scene)

        if scene_type in {"product_highlight_pip_scene", "product_highlight_pip_scene_with_gradient_overlay"}:
            product_bg = self.product_hero_canvas(f"s{seq}_hero", image_idx, duration, scene_image_index, motion_index=motion_index)
            head = self.head_full(f"s{seq}_head", scene)
            return self.static_pip(product_bg, head, f"s{seq}_highlight_pip", duration, scene)

        if scene_type == "product_pip_head_exit_scene":
            product_bg = self.product_hero_canvas(f"s{seq}_hero", image_idx, duration, scene_image_index, motion_index=motion_index)
            head = self.head_full(f"s{seq}_head", scene)
            return self.static_pip(product_bg, head, f"s{seq}_pip_exit", duration, scene)

        if scene_type == "product_bridge_gradient_overlay":
            product_bg = self.product_hero_canvas(f"s{seq}_bridge_hero", image_idx, duration, scene_image_index, motion_index=motion_index)
            return product_bg

        product_base = self.product_base(f"s{seq}_prod", image_idx, duration, scene_image_index=scene_image_index, motion_index=motion_index)

        if scene_type in {"product_scene", "product_bridge_scene"}:
            return product_base
        if scene_type == "pip_scene":
            head = self.head_full(f"s{seq}_head", scene)
            return self.static_pip(product_base, head, f"s{seq}_pip", duration, scene)
        if scene_type == "talk_to_pip_scene":
            head = self.head_full(f"s{seq}_head", scene)
            return self.animated_head(product_base, head, f"s{seq}_to_pip", "to_pip", duration, scene)
        if scene_type == "pip_to_talk_scene":
            head = self.head_full(f"s{seq}_head", scene)
            return self.animated_head(product_base, head, f"s{seq}_to_talk", "to_talk", duration, scene)
        if scene_type == "flipkart_end_scene":
            return self.flipkart_scene(product_base, scene, product, f"s{seq}_flipkart", duration)

        raise ValueError(f"Unsupported scene type: {scene_type}")

    def build_segments(self, scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        cursor = 0.0
        for scene in scenes:
            start = float(scene["start"])
            if start > cursor + 0.001:
                segments.append({"type": "blank_scene", "start": cursor, "end": start, "_generated_gap": True})
            segments.append(scene)
            cursor = float(scene["end"])
        return segments

    def _pip_exit_grad_windows(self, segments: list[dict[str, Any]]) -> list[tuple[float, float]]:
        """Compute absolute (output) start/end pairs for tag-window gradient across all product_pip_head_exit_scene."""
        windows: list[tuple[float, float]] = []
        for scene in segments:
            if scene.get("type") != "product_pip_head_exit_scene":
                continue
            product = scene_product(scene, self.products)
            if not product:
                continue
            scene_start = float(scene["start"])
            scene_dur = float(scene["end"]) - scene_start
            source_start = scene_source_start(scene)
            video_idx = scene_video_index(scene)
            raw = sorted(
                [w for w in product.get("tag_windows", []) if w.get("video_index") == video_idx],
                key=lambda w: w["start_sec"],
            )
            for i, w in enumerate(raw):
                abs_start = scene_start if i == 0 else scene_start + (float(w["start_sec"]) - source_start)
                abs_end = scene_start + min(float(w["end_sec"]) - source_start, scene_dur)
                if abs_end > abs_start:
                    windows.append((abs_start, abs_end))
        return windows

    def build(self, scenes: list[dict[str, Any]], captions_path: Path, gradient_windows: list[GradientWindow] | None = None, gradient_png_input_idx: int = -1) -> tuple[str, float]:
        segments = self.build_segments(scenes)
        self.presplit_head_inputs(segments)
        self.presplit_image_inputs(segments)
        self.presplit_alpha_masks()
        labels: list[str] = []
        skip_next = False
        for seq, scene in enumerate(segments):
            if skip_next:
                skip_next = False
                continue

            next_scene = segments[seq + 1] if seq + 1 < len(segments) else None
            if (
                not self.disable_pair_concat_optimization
                and scene.get("type") == "product_highlight_pip_scene_with_gradient_overlay"
                and next_scene is not None
                and next_scene.get("type") == "talking_head_with_gradient_overlay"
            ):
                prev_scene_here = segments[seq - 1] if seq > 0 else None
                first_label = self.scene_filter(scene, seq, prev_scene_here)
                second_label = self.scene_filter(next_scene, seq + 1, scene)
                first_duration = float(scene["end"]) - float(scene["start"])
                second_duration = float(next_scene["end"]) - float(next_scene["start"])
                first_seg = f"seg{seq}_pair_a"
                second_seg = f"seg{seq}_pair_b"
                self.filters.extend(
                    [
                        f"[{first_label}]trim=duration={first_duration:.3f},setpts=PTS-STARTPTS,"
                        f"{'fps=' + str(self.fps) + ',' if not self.disable_fps_normalization else ''}format=rgba,format=rgb24,setpts=PTS-STARTPTS[{first_seg}]",
                        f"[{second_label}]trim=duration={second_duration:.3f},setpts=PTS-STARTPTS,"
                        f"{'fps=' + str(self.fps) + ',' if not self.disable_fps_normalization else ''}format=rgba,format=rgb24,setpts=PTS-STARTPTS[{second_seg}]",
                        f"[{first_seg}][{second_seg}]concat=n=2:v=1:a=0[seg{seq}]",
                    ]
                )
                labels.append(f"[seg{seq}]")
                skip_next = True
                continue

            prev_scene_for_filter = segments[seq - 1] if seq > 0 else None
            label = self.scene_filter(scene, seq, prev_scene_for_filter)
            duration = float(scene["end"]) - float(scene["start"])
            bridge_scene = None
            if scene.get("type") in _BRIDGE_SCENE_TYPES:
                bridge_scene = scene
            elif seq > 0 and segments[seq - 1].get("type") in _BRIDGE_SCENE_TYPES:
                bridge_scene = segments[seq - 1]
            elif seq + 1 < len(segments) and segments[seq + 1].get("type") in _BRIDGE_SCENE_TYPES:
                bridge_scene = segments[seq + 1]

            scene_transition = scene_dip_transition_config(self.style, scene)
            previous_scene_transition = scene_dip_transition_config(
                self.style,
                segments[seq - 1] if seq > 0 else None,
            )
            next_scene_transition = scene_dip_transition_config(
                self.style,
                segments[seq + 1] if seq + 1 < len(segments) else None,
            )
            if scene_transition is not None:
                _fade_name, fade_duration, fade_color = scene_transition
            elif previous_scene_transition is not None:
                _fade_name, fade_duration, fade_color = previous_scene_transition
            elif next_scene_transition is not None:
                _fade_name, fade_duration, fade_color = next_scene_transition
            else:
                _fade_name, fade_duration, fade_color = bridge_transition_config(self.style, bridge_scene)
            fade_duration = max(0.0, min(fade_duration, duration / 2))
            fade_filters: list[str] = []
            previous_is_bridge = seq > 0 and segments[seq - 1].get("type") in _BRIDGE_SCENE_TYPES
            current_is_bridge = scene.get("type") in _BRIDGE_SCENE_TYPES
            next_is_bridge = seq + 1 < len(segments) and segments[seq + 1].get("type") in _BRIDGE_SCENE_TYPES
            # If an unrelated scene-level transition resolves to duration=0, don't let
            # it suppress bridge-boundary dips. Re-derive from bridge config.
            if fade_duration <= 0 and (current_is_bridge or previous_is_bridge or next_is_bridge):
                _bridge_name, bridge_fade_duration, bridge_fade_color = bridge_transition_config(self.style, bridge_scene)
                fade_duration = max(0.0, min(bridge_fade_duration, duration / 2))
                fade_color = bridge_fade_color
            previous_has_scene_transition = previous_scene_transition is not None
            current_has_scene_transition = scene_transition is not None
            next_has_scene_transition = next_scene_transition is not None

            # Keep highlight->talking_head_with_gradient handoff clean; suppress only
            # the fades AT that boundary so other boundaries (e.g. talking_head->bridge)
            # can still apply their own dip transitions.
            previous_type = segments[seq - 1].get("type") if seq > 0 else None
            current_type = scene.get("type")
            next_type = next_scene.get("type") if next_scene else None
            suppress_fade_in_at_start = (
                previous_type == "product_highlight_pip_scene_with_gradient_overlay"
                and current_type == "talking_head_with_gradient_overlay"
            )
            suppress_fade_out_at_end = (
                current_type == "product_highlight_pip_scene_with_gradient_overlay"
                and next_type == "talking_head_with_gradient_overlay"
            )

            # Dip-to-white when transitioning between full-screen head and PiP-head scenes.
            # Suppressed when a talk_to_pip/pip_to_talk scene is involved — those scenes
            # carry their own internal animation that serves as the visual transition.
            head_pip_in = bool(previous_type) and (
                (current_type in _HEAD_PIP_TYPES and previous_type in _HEAD_FULL_TYPES)
                or (current_type in _HEAD_FULL_TYPES and previous_type in _HEAD_PIP_TYPES)
            ) and current_type not in _HEAD_TRANSITION_TYPES and previous_type not in _HEAD_TRANSITION_TYPES
            head_pip_out = bool(next_type) and (
                (current_type in _HEAD_PIP_TYPES and next_type in _HEAD_FULL_TYPES)
                or (current_type in _HEAD_FULL_TYPES and next_type in _HEAD_PIP_TYPES)
            ) and current_type not in _HEAD_TRANSITION_TYPES and next_type not in _HEAD_TRANSITION_TYPES

            if (
                fade_duration > 0
                and (current_is_bridge or previous_is_bridge or current_has_scene_transition or head_pip_in)
                and not suppress_fade_in_at_start
            ):
                fade_filters.append(f"fade=t=in:st=0:d={fade_duration:.3f}:color={fade_color}")
            if (
                fade_duration > 0
                and (current_is_bridge or next_is_bridge or next_has_scene_transition or head_pip_out)
                and not suppress_fade_out_at_end
            ):
                fade_start = max(duration - fade_duration, 0.0)
                fade_filters.append(f"fade=t=out:st={fade_start:.3f}:d={fade_duration:.3f}:color={fade_color}")

            fade_chain = ""
            if fade_filters:
                fade_chain = "," + ",".join(fade_filters)
            fps_prefix = f"fps={self.fps}," if not self.disable_fps_normalization else ""
            self.filters.append(
                f"[{label}]trim=duration={duration:.3f},setpts=PTS-STARTPTS,"
                f"{fps_prefix}format=rgba{fade_chain},format=rgb24,setpts=PTS-STARTPTS[seg{seq}]"
            )
            labels.append(f"[seg{seq}]")

        if len(labels) == 1:
            pre_label = "seg0"
        else:
            self.filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[preass]")
            pre_label = "preass"

        total_duration = max(float(scene["end"]) for scene in scenes)

        # Watermark overlay — sits above video content, below subtitles
        if self.watermark_input_idx >= 0 and not self.disable_watermark_in_main_graph:
            wm_cfg = self.style.get("watermark", {})
            wm_w = int(wm_cfg.get("width", 140))
            mx = int(wm_cfg.get("margin_x", 20))
            my = int(wm_cfg.get("margin_y", 20))
            self.filters.append(
                f"[{self.watermark_input_idx}:v]scale={wm_w}:-2:flags=lanczos,format=rgba[wm_scaled]"
            )
            self.filters.append(
                f"[{pre_label}][wm_scaled]overlay=x={mx}:y=H-h-{my}:format=auto[pre_wm]"
            )
            pre_label = "pre_wm"

        # Gradient overlay — native FFmpeg panels replacing ASS _grad_bg_strips
        if gradient_windows and gradient_png_input_idx >= 0:
            grad_filters = build_gradient_overlay_filters(
                gradient_windows, gradient_png_input_idx, pre_label, "pre_grad", self.fps
            )
            self.filters.extend(grad_filters)
            pre_label = "pre_grad"

        if not self.disable_subtitles_in_main_graph:
            captions = ffmpeg_filter_quote(captions_path)
            user_fonts = Path.home() / ".local/share/fonts"
            fontsdir_part = f":fontsdir={ffmpeg_filter_quote(user_fonts)}" if user_fonts.is_dir() else ""
            if self.disable_final_yuv420p_in_graph:
                self.filters.append(f"[{pre_label}]subtitles=filename={captions}{fontsdir_part}[v]")
            else:
                self.filters.append(f"[{pre_label}]subtitles=filename={captions}{fontsdir_part},format=yuv420p[v]")
        elif self.disable_final_yuv420p_in_graph:
            self.filters.append(f"[{pre_label}]null[v]")
        else:
            self.filters.append(f"[{pre_label}]format=yuv420p[v]")
        return ";\n".join(self.filters), total_duration


def ffprobe_json(args: list[str]) -> dict[str, Any]:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-of", "json", *args],
        text=True,
    )
    return json.loads(out)


def validate_output(path: Path, width: int, height: int, fps: int, expected_duration: float) -> None:
    if shutil.which("ffprobe") is None:
        print("Warning: ffprobe was not found on PATH; skipped output validation.")
        return
    info = ffprobe_json(["-show_streams", "-show_format", str(path)])
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video:
        raise RuntimeError("ffprobe did not find a video stream in output")
    got_w = int(video.get("width", 0))
    got_h = int(video.get("height", 0))
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    num, den = [float(part) for part in rate.split("/")]
    got_fps = num / den if den else 0
    duration = float(info.get("format", {}).get("duration") or video.get("duration") or 0)

    errors: list[str] = []
    if (got_w, got_h) != (width, height):
        errors.append(f"resolution {got_w}x{got_h}, expected {width}x{height}")
    if not math.isclose(got_fps, fps, abs_tol=0.05):
        errors.append(f"fps {got_fps:.3f}, expected {fps}")
    if duration and abs(duration - expected_duration) > 0.6:
        errors.append(f"duration {duration:.3f}s, expected about {expected_duration:.3f}s")
    if errors:
        raise RuntimeError("Output validation failed: " + "; ".join(errors))


def sanitize_still_image(path: Path, temp_dir: Path, index: int) -> Path:
    clean_path = temp_dir / f"still_{index:02d}.png"
    log_step(f"Sanitizing image {index + 1}: {path.name}")
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map_metadata",
            "-1",
            "-frames:v",
            "1",
            str(clean_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        message = f"Failed to sanitize still image: {path}"
        if detail:
            message += f"\n{detail}"
        raise RuntimeError(message)
    log_step(f"Sanitized image {index + 1}: {clean_path.name}")
    return clean_path


def generate_blurred_background(
    src_path: Path,
    target_w: int,
    target_h: int,
    blur_radius: int,
    blur_power: int,
    out_path: Path,
) -> bool:
    """Produce a single static PNG matching the scale+crop+boxblur chain that
    product_hero_canvas / product_base would otherwise compute every frame.
    Returns True on success, False if ffmpeg failed (caller falls back to
    inline blur)."""
    vf = (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
        f"crop={target_w}:{target_h},setsar=1,"
        f"boxblur={blur_radius}:{blur_power}"
    )
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "error",
            "-i",
            str(src_path),
            "-vf",
            vf,
            "-frames:v",
            "1",
            str(out_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        log_step(
            f"Pre-blur failed for {src_path.name} -> {out_path.name}: {detail[:200]}"
        )
        return False
    return True


def build_command(
    style: dict[str, Any],
    timeline: dict[str, Any],
    timeline_path: Path,
    scenes: list[dict[str, Any]],
    out_path: Path,
    captions_path: Path,
    sanitized_image_dir: Path | None = None,
    style_path_for_gradient: Path | None = None,
    timeline_path_for_gradient: Path | None = None,
) -> tuple[list[str], str, float]:
    base = timeline_path.resolve().parent
    builder = FilterBuilder(style, timeline, timeline_path)
    log_step("Collecting media inputs")
    builder.add_image_inputs(scenes)
    log_step(
        f"Inputs: {len(builder.head_paths)} video(s), "
        f"{len(builder.image_inputs)} product image(s), {len(builder.logo_inputs)} logo(s)"
    )

    # Sanitize stills first so we can build pre-blurred backgrounds from the
    # cleaned PNGs (sanitize strips metadata but preserves pixels).
    image_paths: dict[Path, Path] = {}
    if sanitized_image_dir is not None:
        for image, _idx in sorted(builder.image_inputs.items(), key=lambda item: item[1]):
            image_paths[image] = sanitize_still_image(image, sanitized_image_dir, len(image_paths))

    # Pre-blur each (image, target-size, blur-params) combination needed by
    # product_hero_canvas / product_base, so the main render avoids the
    # per-frame boxblur. Falls back to inline blur if no temp dir was provided
    # or any single ffmpeg pass fails.
    builder.collect_bg_blur_requirements(scenes)
    builder.collect_alpha_mask_requirements(scenes)
    mask_cache_dir = timeline_path.resolve().parent / "_mask_cache"
    if sanitized_image_dir is not None and builder.bg_blur_inputs:
        log_step(
            f"Pre-blurring {len(builder.bg_blur_inputs)} background variant(s)"
        )
        idx_to_path: dict[int, Path] = {}
        for image, idx in builder.image_inputs.items():
            idx_to_path[idx] = image_paths.get(image, image)
        for key in list(builder.bg_blur_inputs.keys()):
            image_idx, target_w, target_h, blur_radius, blur_power = key
            src = idx_to_path.get(image_idx)
            if src is None:
                continue
            blur_out_path = sanitized_image_dir / (
                f"blur_{image_idx:02d}_{target_w}x{target_h}_"
                f"r{blur_radius}_p{blur_power}.png"
            )
            if not blur_out_path.exists() and not generate_blurred_background(
                src, target_w, target_h, blur_radius, blur_power, blur_out_path
            ):
                continue
            builder.bg_blur_paths[key] = blur_out_path
        log_step(
            f"Pre-blur complete: {len(builder.bg_blur_paths)}/{len(builder.bg_blur_inputs)} succeeded"
        )

    if builder.alpha_mask_inputs:
        generated_masks = 0
        reused_masks = 0
        log_step(f"Preparing {len(builder.alpha_mask_inputs)} rounded-mask variant(s)")
        for key in sorted(builder.alpha_mask_inputs.keys()):
            width, height, radius, mode = key
            mask_out_path = mask_cache_dir / rounded_mask_filename(width, height, radius, mode)
            if mask_out_path.exists():
                reused_masks += 1
            else:
                if not generate_rounded_rect_mask(width, height, radius, mask_out_path, mode):
                    raise RuntimeError(f"Failed to generate rounded mask: {mask_out_path}")
                generated_masks += 1
            builder.alpha_mask_paths[key] = mask_out_path
        log_step(
            f"Rounded masks ready: {len(builder.alpha_mask_paths)} total "
            f"({generated_masks} generated, {reused_masks} reused)"
        )

    # Assign ffmpeg input indices for the produced PNGs. They sit AFTER the
    # logo inputs so existing image/logo indices remain stable, and BEFORE the
    # watermark — watermark / sfx indices shift by len(bg_blur_paths).
    next_idx = (
        len(builder.head_paths)
        + len(builder.image_inputs)
        + len(builder.logo_inputs)
    )
    for key in sorted(builder.bg_blur_paths.keys()):
        builder.bg_blur_inputs[key] = next_idx
        next_idx += 1
    for key in sorted(builder.alpha_mask_paths.keys()):
        builder.alpha_mask_inputs[key] = next_idx
        next_idx += 1
    # Drop any keys that did not produce a PNG so the helpers skip them.
    for key in list(builder.bg_blur_inputs.keys()):
        if key not in builder.bg_blur_paths:
            builder.bg_blur_inputs[key] = -1
    for key in list(builder.alpha_mask_inputs.keys()):
        if key not in builder.alpha_mask_paths:
            builder.alpha_mask_inputs[key] = -1

    builder.collect_prerender_scene_inputs(scenes)
    for scene_path in sorted(builder.prerender_scene_inputs.keys()):
        builder.prerender_scene_inputs[scene_path] = next_idx
        next_idx += 1

    # Resolve watermark before build() so the input index is baked into the filter graph.
    # Try timeline dir first, then cwd (project root when called from run_full_batch_pipeline.py).
    wm_path: Path | None = None
    wm_path_str = style.get("watermark", {}).get("path")
    if wm_path_str:
        for _search_base in [base, Path.cwd()]:
            _wm = resolve(_search_base, wm_path_str)
            if _wm is not None and _wm.exists():
                wm_path = _wm
                break
    if wm_path is not None and not builder.disable_watermark_in_main_graph:
        builder.watermark_input_idx = next_idx
        next_idx += 1

    # Collect gradient windows and generate the gradient PNG
    gradient_windows: list[GradientWindow] = []
    gradient_png_path: Path | None = None
    gradient_png_input_idx: int = -1
    _grad_style = style_path_for_gradient
    _grad_timeline = timeline_path_for_gradient
    if _grad_style is not None and _grad_timeline is not None and not builder.disable_subtitles_in_main_graph:
        try:
            gradient_windows = collect_gradient_windows(_grad_style, _grad_timeline)
        except Exception as e:
            log_step(f"Warning: gradient window collection failed ({e}); falling back to ASS strips")
            gradient_windows = []
    if gradient_windows:
        layout = style.get("layout_presets", {}).get("product_highlight_pip", {})
        grad_right = int(builder.out_w * float(layout.get("gradient", {}).get("width_ratio", 0.5)))
        text_cfg = layout.get("text", {})
        first_label_y = int(text_cfg.get("first_label_y", 180))
        label_value_gap = int(text_cfg.get("label_value_gap", 52))
        row_gap = int(text_cfg.get("row_gap", 120))
        # max panel height = 2 rows: first_label_y - 15 to first_label_y + row_gap + label_value_gap + 30 + 15
        max_panel_h = first_label_y + row_gap + label_value_gap + 45 - (first_label_y - 15)
        grad_dir = sanitized_image_dir if sanitized_image_dir is not None else timeline_path.resolve().parent
        gradient_png_path = grad_dir / f"gradient_{grad_right}x{max_panel_h}.png"
        if not gradient_png_path.exists():
            if generate_gradient_png(grad_right, max_panel_h, gradient_png_path):
                log_step(f"Generated gradient PNG: {gradient_png_path.name}")
            else:
                gradient_png_path = None
        if gradient_png_path is not None and gradient_png_path.exists():
            gradient_png_input_idx = next_idx
            next_idx += 1
        else:
            gradient_windows = []

    log_step("Building video filter graph")
    filter_complex, total_duration = builder.build(scenes, captions_path, gradient_windows=gradient_windows, gradient_png_input_idx=gradient_png_input_idx)
    sfx_events = optional_sfx_events(style, timeline, scenes, timeline_path)
    log_step(f"Timeline duration: {total_duration:.2f}s across {len(scenes)} scene(s)")

    encoder = style.get("encoder", {})
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        str(encoder.get("ffmpeg_log_level", "fatal")),
    ]
    filter_threads = int(encoder.get("filter_threads", 0))
    filter_complex_threads = int(encoder.get("filter_complex_threads", filter_threads))
    if filter_threads > 0:
        cmd.extend(["-filter_threads", str(filter_threads)])
    if filter_complex_threads > 0:
        cmd.extend(["-filter_complex_threads", str(filter_complex_threads)])

    for path in builder.head_paths:
        cmd.extend(["-i", str(path)])

    for image, _idx in sorted(builder.image_inputs.items(), key=lambda item: item[1]):
        cmd.extend(
            [
                "-loop",
                "1",
                "-framerate",
                str(builder.fps),
                "-t",
                f"{total_duration:.3f}",
                "-i",
                str(image_paths.get(image, image)),
            ]
        )
    for logo, _idx in sorted(builder.logo_inputs.items(), key=lambda item: item[1]):
        cmd.extend(
            [
                "-loop",
                "1",
                "-framerate",
                str(builder.fps),
                "-t",
                f"{total_duration:.3f}",
                "-i",
                str(logo),
            ]
        )

    # Pre-blurred backgrounds, in the same order as their assigned input idx.
    blur_input_count = 0
    for key, idx in sorted(
        builder.bg_blur_inputs.items(), key=lambda item: item[1]
    ):
        if idx < 0:
            continue
        blur_path = builder.bg_blur_paths.get(key)
        if blur_path is None:
            continue
        cmd.extend(
            [
                "-loop",
                "1",
                "-framerate",
                str(builder.fps),
                "-t",
                f"{total_duration:.3f}",
                "-i",
                str(blur_path),
            ]
        )
        blur_input_count += 1

    mask_input_count = 0
    for key, idx in sorted(
        builder.alpha_mask_inputs.items(), key=lambda item: item[1]
    ):
        if idx < 0:
            continue
        mask_path = builder.alpha_mask_paths.get(key)
        if mask_path is None:
            continue
        cmd.extend(
            [
                "-loop",
                "1",
                "-framerate",
                str(builder.fps),
                "-t",
                f"{total_duration:.3f}",
                "-i",
                str(mask_path),
            ]
        )
        mask_input_count += 1

    prerender_input_count = 0
    for scene_path, _idx in sorted(builder.prerender_scene_inputs.items(), key=lambda item: item[1]):
        cmd.extend(["-i", str(scene_path)])
        prerender_input_count += 1

    if wm_path is not None and not builder.disable_watermark_in_main_graph:
        cmd.extend(["-loop", "1", "-framerate", str(builder.fps), "-t", f"{total_duration:.3f}", "-i", str(wm_path)])

    if gradient_png_path is not None and gradient_png_path.exists():
        cmd.extend(["-loop", "1", "-framerate", str(builder.fps), "-t", f"{total_duration:.3f}", "-i", str(gradient_png_path)])

    next_input_idx = (
        len(builder.head_paths)
        + len(builder.image_inputs)
        + len(builder.logo_inputs)
        + blur_input_count
        + mask_input_count
        + prerender_input_count
    )
    if wm_path is not None and not builder.disable_watermark_in_main_graph:
        next_input_idx += 1
    if gradient_png_path is not None and gradient_png_path.exists():
        next_input_idx += 1

    # Dedicated input-seeked head inputs (see FilterBuilder.head_seek_token), appended
    # after all other inputs so the index assignments above are untouched. Each
    # __HSEEK<slot>__ token in the filter graph is rewritten to its real input index.
    for slot, (_video_index, source_start, path, seek_dur) in enumerate(builder.head_seek_specs):
        cmd.extend(["-ss", f"{source_start:.3f}", "-t", f"{seek_dur:.3f}", "-i", str(path)])
        filter_complex = filter_complex.replace(f"__HSEEK{slot}__", str(next_input_idx))
        next_input_idx += 1

    # Prebuilt flipkart end-card video (replaces the composed end card), if used.
    if builder.flipkart_end_video_path is not None:
        cmd.extend(["-i", str(builder.flipkart_end_video_path)])
        filter_complex = filter_complex.replace("__FLIPKARTVID__", str(next_input_idx))
        next_input_idx += 1

    for event in sfx_events:
        event["input_idx"] = next_input_idx
        next_input_idx += 1
        cmd.extend(["-i", str(event["path"])])

    audio_segments = builder.build_segments(scenes)
    audio_filters, base_audio_label = build_scene_audio_filters(style, audio_segments, len(builder.head_paths))
    audio_map = ["-map", "[aout]"]
    if sfx_events:
        audio_inputs = [f"[{base_audio_label}]"]
        for idx, event in enumerate(sfx_events):
            delay_ms = int(round(float(event["start"]) * 1000))
            audio_filters.append(
                f"[{event['input_idx']}:a]atrim=duration={float(event['duration']):.3f},"
                f"asetpts=PTS-STARTPTS,adelay={delay_ms}|{delay_ms},volume=0.65[sfx{idx}]"
            )
            audio_inputs.append(f"[sfx{idx}]")
        audio_filters.append(
            f"{''.join(audio_inputs)}amix=inputs={len(audio_inputs)}:duration=first:normalize=0[aout]"
        )
    else:
        audio_filters.append(f"[{base_audio_label}]anull[aout]")

    filter_complex = filter_complex + ";\n" + ";\n".join(audio_filters)

    prefer_nvenc = encoder.get("prefer", "h264_nvenc") == "h264_nvenc"
    use_nvenc = prefer_nvenc and check_encoder("h264_nvenc")
    log_step(f"Encoder: {'h264_nvenc' if use_nvenc else 'libx264'}")
    if use_nvenc:
        enc = encoder.get("nvenc", {})
        v_flags = ["-c:v", "h264_nvenc", "-preset", str(enc.get("preset", "p4")), "-rc", str(enc.get("rc", "vbr")), "-cq", str(enc.get("cq", 24))]
        if enc.get("profile"):
            v_flags.extend(["-profile:v", str(enc["profile"])])
        if enc.get("level"):
            v_flags.extend(["-level:v", str(enc["level"])])
    else:
        enc = encoder.get("libx264", {})
        v_flags = ["-c:v", "libx264", "-preset", str(enc.get("preset", "medium")), "-crf", str(enc.get("crf", 23))]
        if enc.get("profile"):
            v_flags.extend(["-profile:v", str(enc["profile"])])
        if enc.get("level"):
            v_flags.extend(["-level:v", str(enc["level"])])

    encoder_threads = int(encoder.get("threads", 1))
    if encoder_threads > 0:
        # libx264 forwards -threads to x264, which by default uses frame-based
        # threading (N frame threads + lookahead + ~N frames buffered), so the OS
        # thread count runs above N and isn't deterministic. With
        # encoder.libx264.sliced_threads enabled we instead pin x264 to sliced
        # threading, where exactly N threads collaborate on one frame with no
        # frame buffering — making per-ffmpeg thread count predictable
        # (1 main + filter_complex_threads + N) for parallel-process packing.
        # nvenc ignores -threads entirely, so this only applies to libx264.
        if not use_nvenc and enc.get("sliced_threads"):
            v_flags.extend([
                "-x264-params",
                f"threads={encoder_threads}:sliced-threads=1:lookahead-threads=1",
            ])
        else:
            v_flags.extend(["-threads", str(encoder_threads)])

    container_flags: list[str] = []
    if encoder.get("video_tag"):
        container_flags.extend(["-tag:v", str(encoder["video_tag"])])
    if encoder.get("brand"):
        container_flags.extend(["-brand", str(encoder["brand"])])

    out_w, out_h, fps = output_size(style)
    scene_boundaries = sorted(
        {
            round(float(scene["start"]), 3)
            for scene in scenes
            if 0.0 < float(scene["start"]) < total_duration
        }
    )
    keyframe_flags: list[str] = []
    if scene_boundaries:
        keyframe_flags = ["-force_key_frames", ",".join(f"{boundary:.3f}" for boundary in scene_boundaries)]
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            *audio_map,
            "-t",
            f"{total_duration:.3f}",
            "-r",
            str(fps),
            *keyframe_flags,
            *v_flags,
            "-pix_fmt",
            style.get("output", {}).get("pix_fmt", "yuv420p"),
            "-c:a",
            encoder.get("audio_codec", "aac"),
            "-b:a",
            encoder.get("audio_bitrate", "160k"),
            "-movflags",
            encoder.get("movflags", "+faststart"),
            *container_flags,
            str(out_path),
        ]
    )
    return cmd, filter_complex, total_duration


def main() -> int:
    parser = argparse.ArgumentParser(description="Render influencer-style product video from JSON timeline.")
    parser.add_argument("--style", default="global_style.json")
    parser.add_argument("--timeline", default="example_timeline.json")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--subtitle-primary-color",
        default=None,
        help="Override ASS PrimaryColour for subtitles, e.g. &H00E5FF& for Flipkart yellow.",
    )
    parser.add_argument(
        "--subtitle-secondary-color",
        default=None,
        help="Override ASS SecondaryColour for karaoke subtitles, e.g. &HFFFFFF&.",
    )
    parser.add_argument("--vid1", default=None, help="Override the first talking-head video input.")
    parser.add_argument("--vid2", default=None, help="Override the second talking-head video input.")
    parser.add_argument(
        "--use-timeline-scenes",
        action="store_true",
        help="Use scenes from the timeline even when --vid1 and --vid2 are provided.",
    )
    parser.add_argument("--bridge-duration", type=float, default=None)
    parser.add_argument("--bridge-product-id", default=None)
    parser.add_argument("--bridge-image-index", type=int, default=0)
    parser.add_argument("--bridge-transition", default=None)
    args = parser.parse_args()

    style_path = Path(args.style)
    timeline_path = Path(args.timeline)
    log_step(f"Loading style: {style_path}")
    style = load_json(style_path)
    if args.subtitle_primary_color or args.subtitle_secondary_color:
        log_step("Applying runtime subtitle color override")
        style = dict(style)
        subtitle_style = dict(style.get("subtitle_style", {}))
        if args.subtitle_primary_color:
            subtitle_style["primary_color"] = args.subtitle_primary_color
        if args.subtitle_secondary_color:
            subtitle_style["secondary_color"] = args.subtitle_secondary_color
        style["subtitle_style"] = subtitle_style
    log_step(f"Loading timeline: {timeline_path}")
    timeline = load_json(timeline_path)
    log_step("Applying timeline/video options")
    timeline = apply_timeline_auto_bridge(style, timeline, timeline_path)
    timeline = apply_video_args(
        style,
        timeline,
        timeline_path,
        args.vid1,
        args.vid2,
        args.bridge_duration,
        args.bridge_product_id,
        args.bridge_image_index,
        args.bridge_transition,
        args.use_timeline_scenes,
    )
    out_path = Path(args.out or timeline.get("output", "output.mp4"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    captions_path = out_path.with_suffix(".ass")
    log_step(f"Output: {out_path}")

    log_step("Validating inputs and scene timings")
    scenes = validate(style, timeline, style_path, timeline_path)
    log_step(f"Validated {len(scenes)} scene(s)")
    runtime_timeline_path = timeline_path.with_name(f"_{timeline_path.stem}.runtime.json")
    log_step(f"Writing runtime timeline: {runtime_timeline_path}")
    runtime_timeline_path.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")
    runtime_style_path = timeline_path.with_name(f"_{timeline_path.stem}.runtime_style.json")
    log_step(f"Writing runtime style: {runtime_style_path}")
    runtime_style_path.write_text(json.dumps(style, indent=2) + "\n", encoding="utf-8")
    log_step(f"Merging captions/product overlays: {captions_path}")
    merge_ass(runtime_style_path, runtime_timeline_path, captions_path)
    log_step("Caption merge complete")

    log_step("Creating temporary output file")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        prefix=f"{out_path.stem}.",
        suffix=out_path.suffix or ".mp4",
        dir=str(out_path.parent),
        delete=False,
    )
    temp_out_path = Path(temp_file.name)
    temp_file.close()
    log_step(f"Temporary output: {temp_out_path}")

    with tempfile.TemporaryDirectory(prefix="render_video_images.") as image_tmp:
        log_step(f"Preparing sanitized image workspace: {image_tmp}")
        cmd, filter_complex, total_duration = build_command(
            style,
            timeline,
            timeline_path,
            scenes,
            temp_out_path,
            captions_path,
            Path(image_tmp),
            style_path_for_gradient=runtime_style_path,
            timeline_path_for_gradient=runtime_timeline_path,
        )
        log_step("Writing debug filter graph: debug_filter.txt")
        Path("debug_filter.txt").write_text(filter_complex + "\n", encoding="utf-8")

        log_step("Starting FFmpeg render")
        print("\nGenerated FFmpeg command:\n", flush=True)
        print(shlex.join(cmd))
        print(flush=True)

        result = subprocess.run(cmd)
        if result.returncode != 0:
            log_step(f"FFmpeg failed with exit code {result.returncode}")
            try:
                temp_out_path.unlink(missing_ok=True)
            except OSError:
                pass
            return result.returncode
        log_step("FFmpeg render complete")

        out_w, out_h, fps = output_size(style)
        log_step("Validating rendered output")
        validate_output(temp_out_path, out_w, out_h, fps, total_duration)
        log_step("Moving temporary output into final path")
        os.replace(temp_out_path, out_path)
    out_path.chmod(0o644)
    log_step(f"Rendered {out_path} ({out_w}x{out_h}, {fps}fps)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
