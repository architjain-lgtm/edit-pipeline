#!/usr/bin/env python3
"""
Scene-by-scene renderer for product_pip_intro timelines.

Each scene is rendered to a lossless FFV1 MKV with its own FFmpeg command.
Transitions (dip-to-white fades) are applied in the final concat pass —
not baked into individual scene renders.

Supported scene types (timeline_generation_config_product_pip_intro.json):
  product_highlight_pip_scene_with_gradient_overlay
  talking_head_with_gradient_overlay
  product_bridge_gradient_overlay
  flipkart_end_scene
  blank_scene  (auto-inserted gap fills)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from render_video import (
    FilterBuilder,
    GradientWindow,
    PRODUCT_SCENE_TYPES,
    PRODUCT_IMAGE_PIP_SCENE_TYPES,
    TALKING_HEAD_AUDIO_SCENE_TYPES,
    _BRIDGE_SCENE_TYPES,
    _HEAD_FULL_TYPES,
    _HEAD_PIP_TYPES,
    _HEAD_TRANSITION_TYPES,
    _PRODUCT_HERO_TYPES,
    apply_timeline_auto_bridge,
    apply_video_args,
    bridge_transition_config,
    build_gradient_overlay_filters,
    check_encoder,
    collect_gradient_windows,
    ffmpeg_filter_quote,
    generate_blurred_background,
    generate_gradient_png,
    generate_rounded_rect_mask,
    load_json,
    log_step,
    merge_ass,
    normalized_corner_mode,
    output_size,
    pip_geometry,
    products_by_id,
    resolve,
    rounded_mask_filename,
    scene_dip_transition_config,
    scene_has_talking_audio,
    scene_image_path,
    scene_product,
    scene_source_start,
    scene_video_index,
    talking_head_paths,
    validate,
)


# ---------------------------------------------------------------------------
# Shared resource container
# ---------------------------------------------------------------------------

@dataclass
class SceneResources:
    """Pre-computed paths and lookup tables shared across all per-scene builds."""
    image_inputs: dict[Path, int]          # image_path -> global_image_idx
    blur_paths: dict[tuple, Path]          # (global_idx, w, h, r, p) -> blurred PNG
    mask_paths: dict[tuple, Path]          # (w, h, r, mode) -> mask PNG
    head_paths: list[Path]                 # indexed by video_idx
    logo_inputs: dict[Path, int]           # logo_path -> global_idx (composed end scene)
    hero_canvas_paths: dict[int, Path] = field(default_factory=dict)  # seg_idx -> pre-rendered canvas


# ---------------------------------------------------------------------------
# Per-scene filter builder
# ---------------------------------------------------------------------------

class SceneFilterBuilder(FilterBuilder):
    """
    FFmpeg filter builder for a single scene with 0-based local inputs.

    All filter methods inherited from FilterBuilder work unchanged because they
    read shared state dicts (_head_split_labels, _image_split_labels,
    bg_blur_inputs, _mask_split_labels, logo_inputs) that this class populates
    via add_*_input() with local indices.
    """

    def __init__(
        self,
        style: dict[str, Any],
        timeline: dict[str, Any],
        timeline_path: Path,
    ) -> None:
        # Skip FilterBuilder.__init__ — we manage our own local state.
        self.style = style
        self.timeline = timeline
        self.timeline_base = timeline_path.resolve().parent
        self.out_w, self.out_h, self.fps = output_size(style)
        self.products = products_by_id(timeline)
        encoder_cfg = style.get("encoder", {})

        self.filters: list[str] = []
        self._input_args: list[list[str]] = []   # per-input ffmpeg arg groups

        # Dicts populated with LOCAL indices; consumed by FilterBuilder filter methods.
        self._head_split_labels: dict[int, list[str]] = {}
        self._image_split_labels: dict[int, list[str]] = {}
        self.image_inputs: dict[Path, int] = {}
        self.logo_inputs: dict[Path, int] = {}
        self.bg_blur_inputs: dict[tuple, int] = {}
        self.bg_blur_paths: dict[tuple, Path] = {}
        self.alpha_mask_inputs: dict[tuple, int] = {}
        self.alpha_mask_paths: dict[tuple, Path] = {}
        self.alpha_mask_use_counts: dict[tuple, int] = {}
        self._mask_split_labels: dict[tuple, list[str]] = {}
        self.prerender_scene_inputs: dict[Path, int] = {}
        self.prerender_scene_paths: dict[Path, Path] = {}
        self.head_paths: list[Path] = []

        self.flipkart_end_video_path: Path | None = None
        self._flipkart_end_video_resolved: bool = False
        self._flipkart_local_idx: int = -1
        self._hero_canvas_local_idx: int = -1   # set by add_hero_canvas_input()
        self._face_pip_crops: dict[int, tuple[int, int, int, int] | None] = {}
        self.watermark_input_idx: int = -1

        # input-level seek offset per video_idx; subtracted in head_full() trim
        self._head_seek_offset: dict[int, float] = {}

        # Feature flags: watermark, subtitles, yuv420p all handled by concat pass
        self.head_input_seek: bool = False
        self.disable_subtitles_in_main_graph: bool = True
        self.disable_watermark_in_main_graph: bool = True
        self.disable_final_yuv420p_in_graph: bool = True
        self.disable_fps_normalization: bool = bool(encoder_cfg.get("disable_fps_normalization", False))
        self.disable_pair_concat_optimization: bool = True
        self.scene_prerender_mode: str = "off"
        self.head_seek_index: dict[tuple, int] = {}
        self.head_seek_specs: list[list[Any]] = []

    # ── helpers ──────────────────────────────────────────────────────────────

    def _next_idx(self) -> int:
        return len(self._input_args)

    # ── input registration ────────────────────────────────────────────────────

    def add_video_input(
        self, path: Path, video_idx: int, source_start: float, duration: float
    ) -> int:
        """Head video with input-level fast seek to source_start."""
        local_idx = self._next_idx()
        self._input_args.append([
            "-ss", f"{source_start:.3f}",
            "-t", f"{duration + 2.0:.3f}",
            "-i", str(path),
        ])
        self._head_split_labels[video_idx] = [f"{local_idx}:v"]
        self._head_seek_offset[video_idx] = source_start
        return local_idx

    def add_image_input(self, path: Path, global_image_idx: int, duration: float) -> int:
        """Looped still image."""
        local_idx = self._next_idx()
        self._input_args.append([
            "-loop", "1", "-framerate", str(self.fps),
            "-t", f"{duration:.3f}", "-i", str(path),
        ])
        self._image_split_labels[global_image_idx] = [f"{local_idx}:v"]
        self.image_inputs[path] = global_image_idx
        return local_idx

    def add_blur_input(
        self,
        path: Path,
        global_image_idx: int,
        target_w: int,
        target_h: int,
        blur_radius: int,
        blur_power: int,
        duration: float,
    ) -> int:
        """Pre-blurred background PNG."""
        local_idx = self._next_idx()
        self._input_args.append([
            "-loop", "1", "-framerate", str(self.fps),
            "-t", f"{duration:.3f}", "-i", str(path),
        ])
        key = (int(global_image_idx), int(target_w), int(target_h), int(blur_radius), int(blur_power))
        self.bg_blur_inputs[key] = local_idx
        self.bg_blur_paths[key] = path
        return local_idx

    def add_mask_input(self, path: Path, key: tuple, duration: float) -> int:
        """Rounded-corner mask PNG (used at most once per scene)."""
        local_idx = self._next_idx()
        self._input_args.append([
            "-loop", "1", "-framerate", str(self.fps),
            "-t", f"{duration:.3f}", "-i", str(path),
        ])
        self.alpha_mask_inputs[key] = local_idx
        self.alpha_mask_paths[key] = path
        self._mask_split_labels[key] = [f"{local_idx}:v"]
        return local_idx

    def add_logo_input(self, path: Path, duration: float) -> int:
        """Logo PNG (composed flipkart end scene)."""
        local_idx = self._next_idx()
        self._input_args.append([
            "-loop", "1", "-framerate", str(self.fps),
            "-t", f"{duration:.3f}", "-i", str(path),
        ])
        self.logo_inputs[path.resolve()] = local_idx
        return local_idx

    def add_flipkart_end_video(self, path: Path) -> int:
        """Prebuilt Flipkart end-card video."""
        local_idx = self._next_idx()
        self._input_args.append(["-i", str(path)])
        self.flipkart_end_video_path = path
        self._flipkart_end_video_resolved = True
        self._flipkart_local_idx = local_idx
        return local_idx

    def add_hero_canvas_input(self, path: Path) -> int:
        """Pre-rendered Ken Burns hero canvas video (CPU numpy)."""
        local_idx = self._next_idx()
        self._input_args.append(["-i", str(path)])
        self._hero_canvas_local_idx = local_idx
        return local_idx

    # ── filter overrides ──────────────────────────────────────────────────────

    def product_hero_canvas(
        self,
        label: str,
        image_idx: int,
        duration: float,
        scene_image_index: int = 0,
        motion_index: int | None = None,
    ) -> str:
        """Use pre-rendered canvas video if available; otherwise fall back to parent chain."""
        if self._hero_canvas_local_idx >= 0:
            self.filters.append(
                f"[{self._hero_canvas_local_idx}:v]"
                f"trim=duration={duration:.3f},setpts=PTS-STARTPTS,format=rgba[{label}]"
            )
            return label
        return super().product_hero_canvas(label, image_idx, duration, scene_image_index, motion_index)

    # ── head_full override ────────────────────────────────────────────────────

    def head_full(self, label: str, scene: dict[str, Any]) -> str:
        """Same as FilterBuilder.head_full but adjusts trim for input-level seek offset."""
        duration = float(scene["end"]) - float(scene["start"])
        source_start = scene_source_start(scene)
        input_idx = scene_video_index(scene)
        queue = self._head_split_labels.get(input_idx)
        stream_ref = f"[{queue.pop(0)}]" if queue else f"[{input_idx}:v]"
        trim_start = max(0.0, round(source_start - self._head_seek_offset.get(input_idx, 0.0), 6))
        self.filters.append(
            f"{stream_ref}trim=start={trim_start:.3f}:duration={duration:.3f},setpts=PTS-STARTPTS,"
            f"scale={self.out_w}:{self.out_h}:force_original_aspect_ratio=increase,"
            f"crop={self.out_w}:{self.out_h},setsar=1,format=rgba,"
            f"tpad=stop_mode=clone:stop_duration={duration:.3f},"
            f"trim=duration={duration:.3f},setpts=PTS-STARTPTS[{label}]"
        )
        return label

    # ── command builder ───────────────────────────────────────────────────────

    def build_scene_cmd(
        self,
        scene: dict[str, Any],
        prev_scene: dict[str, Any] | None,
        out_path: Path,
    ) -> list[str]:
        """Build FFmpeg command to render this scene to a lossless FFV1 MKV."""
        duration = float(scene["end"]) - float(scene["start"])
        label = self.scene_filter(scene, 0, prev_scene)

        fps_prefix = f"fps={self.fps}," if not self.disable_fps_normalization else ""
        self.filters.append(
            f"[{label}]trim=duration={duration:.3f},setpts=PTS-STARTPTS,"
            f"{fps_prefix}format=yuv420p,setpts=PTS-STARTPTS[vscene]"
        )

        filter_str = ";\n".join(self.filters)
        if self._flipkart_local_idx >= 0:
            filter_str = filter_str.replace("__FLIPKARTVID__", str(self._flipkart_local_idx))

        encoder_cfg = self.style.get("encoder", {})
        filter_threads = int(encoder_cfg.get("filter_threads", 0))
        filter_complex_threads = int(encoder_cfg.get("filter_complex_threads", filter_threads))

        cmd: list[str] = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-nostats",
            "-loglevel", str(encoder_cfg.get("ffmpeg_log_level", "fatal")),
        ]
        if filter_threads > 0:
            cmd.extend(["-filter_threads", str(filter_threads)])
        if filter_complex_threads > 0:
            cmd.extend(["-filter_complex_threads", str(filter_complex_threads)])
        for args in self._input_args:
            cmd.extend(args)
        cmd.extend([
            "-filter_complex", filter_str,
            "-map", "[vscene]",
            "-an",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
            "-t", f"{duration:.3f}",
            "-r", str(self.fps),
            str(out_path),
        ])
        return cmd


# ---------------------------------------------------------------------------
# Resource preparation
# ---------------------------------------------------------------------------

def prepare_resources(
    style: dict[str, Any],
    timeline: dict[str, Any],
    timeline_path: Path,
    scenes: list[dict[str, Any]],
    work_dir: Path,
) -> SceneResources:
    """Collect blur/mask requirements via FilterBuilder, then generate the PNGs."""
    builder = FilterBuilder(style, timeline, timeline_path)
    builder.add_image_inputs(scenes)
    builder.collect_bg_blur_requirements(scenes)
    builder.collect_alpha_mask_requirements(scenes)

    idx_to_path: dict[int, Path] = {idx: p for p, idx in builder.image_inputs.items()}

    work_dir.mkdir(parents=True, exist_ok=True)
    log_step(f"Pre-blurring {len(builder.bg_blur_inputs)} background variant(s)")
    for key in list(builder.bg_blur_inputs.keys()):
        img_idx, target_w, target_h, blur_radius, blur_power = key
        src = idx_to_path.get(img_idx)
        if src is None:
            continue
        blur_out = work_dir / f"blur_{img_idx:02d}_{target_w}x{target_h}_r{blur_radius}_p{blur_power}.png"
        if blur_out.exists() or generate_blurred_background(
            src, target_w, target_h, blur_radius, blur_power, blur_out
        ):
            builder.bg_blur_paths[key] = blur_out

    mask_cache_dir = timeline_path.resolve().parent / "_mask_cache"
    mask_cache_dir.mkdir(parents=True, exist_ok=True)
    log_step(f"Preparing {len(builder.alpha_mask_inputs)} rounded-mask variant(s)")
    for key in list(builder.alpha_mask_inputs.keys()):
        w, h, r, mode = key
        mask_out = mask_cache_dir / rounded_mask_filename(w, h, r, mode)
        if not mask_out.exists():
            generate_rounded_rect_mask(w, h, r, mask_out, mode)
        if mask_out.exists():
            builder.alpha_mask_paths[key] = mask_out

    return SceneResources(
        image_inputs=dict(builder.image_inputs),
        blur_paths=dict(builder.bg_blur_paths),
        mask_paths=dict(builder.alpha_mask_paths),
        head_paths=list(builder.head_paths),
        logo_inputs=dict(builder.logo_inputs),
    )


# ---------------------------------------------------------------------------
#  frame utilities
# ---------------------------------------------------------------------------

def _extract_frames(
    path: Path, ss: float, duration: float, w: int, h: int
) -> list[np.ndarray]:
    """Decode a segment of a video into raw RGB24 numpy frames."""
    frame_size = w * h * 3
    r = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{ss:.3f}", "-t", f"{duration:.3f}",
         "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    raw = r.stdout
    return [
        np.frombuffer(raw[i:i + frame_size], dtype=np.uint8).reshape((h, w, 3)).copy()
        for i in range(0, len(raw) - frame_size + 1, frame_size)
    ]


def _encode_frames(
    frames: list[np.ndarray], dst: Path, w: int, h: int, fps: int
) -> None:
    """Encode a list of RGB24 numpy frames to a lossless H.264 file."""
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
         "-loglevel", "error", str(dst)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()


def _blend_frames(
    a: np.ndarray, b: np.ndarray, mask: float | np.ndarray
) -> np.ndarray:
    return np.clip(
        a.astype(np.float32) * (1.0 - mask) + b.astype(np.float32) * mask,
        0, 255,
    ).astype(np.uint8)


def _fade_white(a: np.ndarray, b: np.ndarray, p: float) -> np.ndarray:
    """Dip-to-white: a → white → b over p ∈ [0, 1]."""
    white = np.full_like(a, 255, dtype=np.float32)
    if p < 0.5:
        return _blend_frames(a.astype(np.float32), white, p * 2).astype(np.uint8)
    return _blend_frames(white, b.astype(np.float32), (p - 0.5) * 2).astype(np.uint8)


# ---------------------------------------------------------------------------
# Ken Burns hero canvas pre-renderer (CPU numpy)
# ---------------------------------------------------------------------------

def _render_hero_canvas(
    product_img_path: Path,
    bg_path: Path,
    motion: dict[str, Any],
    max_w: int,
    max_h: int,
    out_w: int,
    out_h: int,
    duration: float,
    fps: int,
    out_path: Path,
) -> None:
    """
    Pre-render the Ken Burns pan/zoom hero canvas to a video file using PIL/numpy.
    Replicates the eval=frame scale+overlay chain from product_hero_canvas().
    """
    bg = np.array(Image.open(bg_path).convert("RGB").resize((out_w, out_h), Image.LANCZOS))
    fg_orig = Image.open(product_img_path).convert("RGB")
    iw, ih = fg_orig.size
    n_frames = max(1, round(duration * fps))

    x_cfg = motion.get("x_translate", {})
    y_cfg = motion.get("y_translate", {})
    s_cfg = motion.get("scale", {})
    x_start = float(x_cfg.get("start", 0.0))
    x_end   = float(x_cfg.get("end",   x_start))
    y_start = float(y_cfg.get("start", 0.0))
    y_end   = float(y_cfg.get("end",   y_start))
    s_start = float(s_cfg.get("start", 1.0))
    s_end   = float(s_cfg.get("end",   s_start))
    dur = max(duration, 1e-6)

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{out_w}x{out_h}", "-r", str(fps), "-i", "pipe:0",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
         "-loglevel", "error", str(out_path)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    def trunc_even(x: float) -> int:
        return max(2, int(x) // 2 * 2)

    def scaled_dims(scale: float) -> tuple[int, int]:
        if iw / ih > max_w / max_h:
            w = trunc_even(max_w * scale)
            h = trunc_even(w * ih / iw)
        else:
            h = trunc_even(max_h * scale)
            w = trunc_even(h * iw / ih)
        return w, h

    constant_scale = abs(s_start - s_end) < 1e-6

    if constant_scale:
        # Resize once; per-frame work is only a numpy copy + slice — very fast.
        new_w, new_h = scaled_dims(s_start)
        fg_scaled = np.array(fg_orig.resize((new_w, new_h), Image.BILINEAR))

        for idx in range(n_frames):
            t = idx / fps
            x = round((out_w - new_w) / 2 + x_start + (x_end - x_start) * t / dur)
            y = round((out_h - new_h) / 2 - y_start - (y_end - y_start) * t / dur)
            frame = bg.copy()
            x0, y0 = max(x, 0), max(y, 0)
            x1, y1 = min(x + new_w, out_w), min(y + new_h, out_h)
            if x1 > x0 and y1 > y0:
                frame[y0:y1, x0:x1] = fg_scaled[y0 - y : y1 - y, x0 - x : x1 - x]
            proc.stdin.write(frame.tobytes())
    else:
        # Variable scale (zoom): resize per frame with BILINEAR (~4× faster than LANCZOS).
        for idx in range(n_frames):
            t = idx / fps
            scale = s_start + (s_end - s_start) * t / dur
            new_w, new_h = scaled_dims(scale)
            x = round((out_w - new_w) / 2 + x_start + (x_end - x_start) * t / dur)
            y = round((out_h - new_h) / 2 - y_start - (y_end - y_start) * t / dur)
            fg = np.array(fg_orig.resize((new_w, new_h), Image.BILINEAR))
            frame = bg.copy()
            x0, y0 = max(x, 0), max(y, 0)
            x1, y1 = min(x + new_w, out_w), min(y + new_h, out_h)
            if x1 > x0 and y1 > y0:
                frame[y0:y1, x0:x1] = fg[y0 - y : y1 - y, x0 - x : x1 - x]
            proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()


def render_hero_canvases(
    segments: list[dict[str, Any]],
    resources: SceneResources,
    style: dict[str, Any],
    timeline: dict[str, Any],
    timeline_path: Path,
    work_dir: Path,
) -> dict[int, Path]:
    """Pre-render Ken Burns hero canvases for all applicable segments."""
    out_w, out_h, fps = output_size(style)
    layout = style.get("layout_presets", {}).get("product_fullscreen_hero_image", {})
    fg_cfg = layout.get("foreground", {})
    max_w = int(fg_cfg.get("max_w", out_w - 60))
    max_h = int(fg_cfg.get("max_h", out_h - 90))

    idx_to_path: dict[int, Path] = {idx: p for p, idx in resources.image_inputs.items()}
    products = products_by_id(timeline)
    timeline_base = timeline_path.resolve().parent
    # In per-scene renders scene_filter() is always called with seq=0,
    # so scene_image_index=0 → motion preset "0".
    motion = style.get("product_image_motion", {}).get("0") or {}
    if not isinstance(motion, dict):
        motion = {}

    _HERO_TYPES = {
        "product_highlight_pip_scene_with_gradient_overlay",
        "product_bridge_gradient_overlay",
    }

    result: dict[int, Path] = {}
    for seg_idx, scene in enumerate(segments):
        if scene.get("type") not in _HERO_TYPES:
            continue
        duration = float(scene["end"]) - float(scene["start"])
        product = scene_product(scene, products)
        if product is None:
            continue
        try:
            image_path = scene_image_path(scene, product, timeline_base).resolve()
        except (ValueError, AssertionError):
            continue
        global_idx = resources.image_inputs.get(image_path)
        if global_idx is None:
            continue
        blur_key = _hero_blur_key(style, global_idx, out_w, out_h)
        bg_path = resources.blur_paths.get(blur_key)
        if bg_path is None:
            continue  # no pre-blurred PNG; FFmpeg fallback will handle it

        canvas_path = work_dir / f"hero_canvas_{seg_idx:04d}.mp4"
        log_step(f"  hero canvas [{seg_idx}] {duration:.2f}s → {canvas_path.name}")
        _render_hero_canvas(
            product_img_path=image_path,
            bg_path=bg_path,
            motion=motion,
            max_w=max_w,
            max_h=max_h,
            out_w=out_w,
            out_h=out_h,
            duration=duration,
            fps=fps,
            out_path=canvas_path,
        )
        result[seg_idx] = canvas_path

    return result


# ---------------------------------------------------------------------------
# Blur key helpers
# ---------------------------------------------------------------------------

def _hero_blur_key(
    style: dict[str, Any],
    global_image_idx: int,
    out_w: int,
    out_h: int,
) -> tuple:
    layout = style.get("layout_presets", {}).get("product_fullscreen_hero_image", {})
    blur = layout.get("background", {}).get("blur", {})
    return (global_image_idx, out_w, out_h, int(blur.get("radius", 28)), int(blur.get("power", 1)))


def _base_blur_keys(
    style: dict[str, Any],
    global_image_idx: int,
    out_w: int,
    out_h: int,
) -> tuple[tuple, tuple]:
    """Returns (canvas_key, inner_key) — the two blur PNGs product_base() needs."""
    layout = style.get("layout_presets", {}).get("product_top_2_3_data_bottom_1_3", {})
    blur = layout.get("product_image", {}).get("background", {}).get("blur", {})
    r = int(blur.get("radius", 28))
    p = int(blur.get("power", 1))
    region = layout.get("product_region", {})
    rw = int(region.get("w", out_w))
    rh = int(region.get("h", out_h))
    return (global_image_idx, out_w, out_h, r, p), (global_image_idx, rw, rh, r, p)


# ---------------------------------------------------------------------------
# CPU transition renderer (dip-to-white between scene MKVs)
# ---------------------------------------------------------------------------

def _parse_fade_dur(fade_str: str) -> float:
    m = re.search(r":d=([0-9.]+)", fade_str)
    return float(m.group(1)) if m else 0.0


def render_cpu_transitions(
    scene_paths: list[Path],
    segments: list[dict[str, Any]],
    style: dict[str, Any],
    work_dir: Path,
) -> list[Path]:
    """
    Replace per-clip fade filters with CPU numpy dip-to-white blends.

    For each boundary where _compute_scene_fades() would emit a fade-out,
    extract the tail frames of scene[i] and head frames of scene[i+1],
    blend them with _fade_white, and write a short transition clip.
    Content clips are trimmed to remove the overlapping regions.

    Returns an ordered sequence:
        [trimmed_s0, trans_01, trimmed_s1, trans_12, ..., trimmed_sN]
    suitable for direct ffconcat / filter_complex concat (no fade filters needed).
    """
    out_w, out_h, fps = output_size(style)

    fade_in_dur  = [0.0] * len(segments)
    fade_out_dur = [0.0] * len(segments)
    for i in range(len(segments)):
        for f in _compute_scene_fades(segments, i, style):
            if "t=in"  in f:
                fade_in_dur[i]  = _parse_fade_dur(f)
            elif "t=out" in f:
                fade_out_dur[i] = _parse_fade_dur(f)

    result: list[Path] = []

    for i, (scene_path, scene) in enumerate(zip(scene_paths, segments)):
        duration  = float(scene["end"]) - float(scene["start"])
        td_in     = fade_in_dur[i]
        td_out    = fade_out_dur[i]
        trim_dur  = duration - td_in - td_out

        # Trimmed content clip (everything outside the transition overlap)
        if trim_dur > 0.05:
            trimmed = work_dir / f"trimmed_{i:04d}.mp4"
            subprocess.run(
                ["ffmpeg", "-y",
                 "-ss", f"{td_in:.3f}", "-t", f"{trim_dur:.3f}",
                 "-i", str(scene_path),
                 "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
                 "-loglevel", "error", str(trimmed)],
                check=True,
            )
            result.append(trimmed)

        # Transition clip at the tail of this segment
        if td_out > 0.0 and i + 1 < len(scene_paths):
            frames_a = _extract_frames(scene_path, max(0.0, duration - td_out), td_out, out_w, out_h)
            frames_b = _extract_frames(scene_paths[i + 1], 0.0, td_out, out_w, out_h)
            n = min(len(frames_a), len(frames_b))
            if n > 0:
                blended: list[np.ndarray] = []
                for j in range(n):
                    p = j / max(1, n - 1)
                    p = p * p * (3.0 - 2.0 * p)   # smoothstep ease-in-out
                    blended.append(_fade_white(frames_a[j], frames_b[j], p))
                trans_path = work_dir / f"trans_{i:04d}_{i + 1:04d}.mp4"
                _encode_frames(blended, trans_path, out_w, out_h, fps)
                result.append(trans_path)

    return result


# ---------------------------------------------------------------------------
# Scene builder setup
# ---------------------------------------------------------------------------

def setup_scene_builder(
    scene: dict[str, Any],
    prev_scene: dict[str, Any] | None,
    resources: SceneResources,
    style: dict[str, Any],
    timeline: dict[str, Any],
    timeline_path: Path,
    seg_idx: int = -1,
) -> SceneFilterBuilder:
    """Create a SceneFilterBuilder and register inputs needed by this scene."""
    builder = SceneFilterBuilder(style, timeline, timeline_path)
    scene_type = scene.get("type")
    duration = float(scene["end"]) - float(scene["start"])
    out_w, out_h, _ = output_size(style)

    # blank gap — no inputs needed
    if scene_type == "blank_scene":
        return builder

    # ── product_bridge_gradient_overlay ──────────────────────────────────────
    if scene_type == "product_bridge_gradient_overlay":
        product = scene_product(scene, builder.products)
        if product:
            try:
                image = scene_image_path(scene, product, builder.timeline_base).resolve()
                global_idx = resources.image_inputs.get(image)
                if global_idx is not None:
                    builder.add_image_input(image, global_idx, duration)
                    blur_key = _hero_blur_key(style, global_idx, out_w, out_h)
                    blur_path = resources.blur_paths.get(blur_key)
                    if blur_path:
                        _, w, h, r, p = blur_key
                        builder.add_blur_input(blur_path, global_idx, w, h, r, p, duration)
            except (ValueError, AssertionError):
                pass
        canvas_path = resources.hero_canvas_paths.get(seg_idx)
        if canvas_path is not None:
            builder.add_hero_canvas_input(canvas_path)
        return builder

    # ── product_highlight_pip_scene_with_gradient_overlay ────────────────────
    if scene_type == "product_highlight_pip_scene_with_gradient_overlay":
        video_idx = scene_video_index(scene)
        if video_idx < len(resources.head_paths):
            builder.add_video_input(
                resources.head_paths[video_idx], video_idx, scene_source_start(scene), duration
            )
        product = scene_product(scene, builder.products)
        if product:
            try:
                image = scene_image_path(scene, product, builder.timeline_base).resolve()
                global_idx = resources.image_inputs.get(image)
                if global_idx is not None:
                    builder.add_image_input(image, global_idx, duration)
                    blur_key = _hero_blur_key(style, global_idx, out_w, out_h)
                    blur_path = resources.blur_paths.get(blur_key)
                    if blur_path:
                        _, w, h, r, p = blur_key
                        builder.add_blur_input(blur_path, global_idx, w, h, r, p, duration)
            except (ValueError, AssertionError):
                pass
        pip = style.get("pip_style", {})
        pip_w, pip_h, border, _, _ = pip_geometry(style, out_w, out_h)
        frame_w, frame_h = pip_w + border * 2, pip_h + border * 2
        radius = int(pip.get("corner_radius", 18))
        if radius > 0:
            mask_key = (frame_w, frame_h, radius, normalized_corner_mode("all"))
            mask_path = resources.mask_paths.get(mask_key)
            if mask_path:
                builder.add_mask_input(mask_path, mask_key, duration)
        canvas_path = resources.hero_canvas_paths.get(seg_idx)
        if canvas_path is not None:
            builder.add_hero_canvas_input(canvas_path)
        return builder

    # ── talking_head_scene ───────────────────────────────────────────────────
    if scene_type == "talking_head_scene":
        video_idx = scene_video_index(scene)
        if video_idx < len(resources.head_paths):
            builder.add_video_input(
                resources.head_paths[video_idx], video_idx, scene_source_start(scene), duration
            )
        product = scene_product(scene, builder.products)
        if product:
            try:
                image = scene_image_path(scene, product, builder.timeline_base).resolve()
                global_idx = resources.image_inputs.get(image)
                if global_idx is not None:
                    builder.add_image_input(image, global_idx, duration)
                    pip = style.get("pip_style", {})
                    pip_w, pip_h, border, _, _ = pip_geometry(style, out_w, out_h)
                    frame_w, frame_h = pip_w + border * 2, pip_h + border * 2
                    pip_radius = int(pip.get("corner_radius", 18))
                    prev_type = prev_scene.get("type") if prev_scene else None
                    if prev_type in _PRODUCT_HERO_TYPES:
                        pip_scale = pip_w / out_w if out_w > 0 else 1.0
                        anim_radius = (
                            0 if pip_radius <= 0
                            else max(1, round(pip_radius / pip_scale)) if pip_scale > 0 else pip_radius
                        )
                        mask_key = (out_w, out_h, anim_radius, normalized_corner_mode("all"))
                    else:
                        mask_key = (frame_w, frame_h, pip_radius, normalized_corner_mode("all"))
                    if mask_key[2] > 0:
                        mask_path = resources.mask_paths.get(mask_key)
                        if mask_path:
                            builder.add_mask_input(mask_path, mask_key, duration)
            except (ValueError, AssertionError):
                pass
        return builder

    # ── talking_head_product_strip_scene ─────────────────────────────────────
    if scene_type == "talking_head_product_strip_scene":
        video_idx = scene_video_index(scene)
        if video_idx < len(resources.head_paths):
            builder.add_video_input(
                resources.head_paths[video_idx], video_idx, scene_source_start(scene), duration
            )
        product = scene_product(scene, builder.products)
        if product:
            try:
                image = scene_image_path(scene, product, builder.timeline_base).resolve()
                global_idx = resources.image_inputs.get(image)
                if global_idx is not None:
                    builder.add_image_input(image, global_idx, duration)
            except (ValueError, AssertionError):
                pass
        # strip mask
        strip_layout = style.get("layout_presets", {}).get("talking_head_with_product_data_strip", {})
        strip = strip_layout.get("strip", {})
        strip_y = int(strip.get("y", 640))
        strip_w = int(strip.get("w", out_w))
        strip_h = int(strip.get("h", out_h - strip_y))
        strip_radius = int(strip.get("corner_radius", 0))
        strip_mode = normalized_corner_mode(str(strip.get("corner_mode", "all")))
        if strip_radius > 0:
            mask_key = (strip_w, strip_h, strip_radius, strip_mode)
            mask_path = resources.mask_paths.get(mask_key)
            if mask_path:
                builder.add_mask_input(mask_path, mask_key, duration)
        # pip badge mask
        pip = style.get("pip_style", {})
        pip_w, pip_h, border, _, _ = pip_geometry(style, out_w, out_h)
        frame_w, frame_h = pip_w + border * 2, pip_h + border * 2
        pip_radius = int(pip.get("corner_radius", 18))
        if pip_radius > 0:
            pip_mask_key = (frame_w, frame_h, pip_radius, normalized_corner_mode("all"))
            pip_mask_path = resources.mask_paths.get(pip_mask_key)
            if pip_mask_path:
                builder.add_mask_input(pip_mask_path, pip_mask_key, duration)
        return builder

    # ── talking_head_with_gradient_overlay ───────────────────────────────────
    if scene_type == "talking_head_with_gradient_overlay":
        video_idx = scene_video_index(scene)
        if video_idx < len(resources.head_paths):
            builder.add_video_input(
                resources.head_paths[video_idx], video_idx, scene_source_start(scene), duration
            )
        prev_type = prev_scene.get("type") if prev_scene else None
        product = scene_product(scene, builder.products)
        if product:
            try:
                image = scene_image_path(scene, product, builder.timeline_base).resolve()
                global_idx = resources.image_inputs.get(image)
                if global_idx is not None:
                    builder.add_image_input(image, global_idx, duration)
                    pip = style.get("pip_style", {})
                    pip_w, pip_h, border, _, _ = pip_geometry(style, out_w, out_h)
                    frame_w, frame_h = pip_w + border * 2, pip_h + border * 2
                    pip_radius = int(pip.get("corner_radius", 18))
                    if prev_type in _PRODUCT_HERO_TYPES:
                        pip_scale = pip_w / out_w if out_w > 0 else 1.0
                        anim_radius = (
                            0 if pip_radius <= 0
                            else max(1, round(pip_radius / pip_scale)) if pip_scale > 0 else pip_radius
                        )
                        mask_key = (out_w, out_h, anim_radius, normalized_corner_mode("all"))
                    else:
                        mask_key = (frame_w, frame_h, pip_radius, normalized_corner_mode("all"))
                    if mask_key[2] > 0:
                        mask_path = resources.mask_paths.get(mask_key)
                        if mask_path:
                            builder.add_mask_input(mask_path, mask_key, duration)
            except (ValueError, AssertionError):
                pass
        return builder

    # ── flipkart_end_scene ────────────────────────────────────────────────────
    if scene_type == "flipkart_end_scene":
        end_video = builder._end_video()
        if end_video is not None:
            builder.add_flipkart_end_video(end_video)
        else:
            # Composed fallback: product_base + flipkart logo overlay
            product = scene_product(scene, builder.products)
            if product:
                try:
                    image = scene_image_path(scene, product, builder.timeline_base).resolve()
                    global_idx = resources.image_inputs.get(image)
                    if global_idx is not None:
                        builder.add_image_input(image, global_idx, duration)
                        canvas_key, inner_key = _base_blur_keys(style, global_idx, out_w, out_h)
                        for bkey in (canvas_key, inner_key):
                            bpath = resources.blur_paths.get(bkey)
                            if bpath:
                                _, w, h, r, p = bkey
                                builder.add_blur_input(bpath, global_idx, w, h, r, p, duration)
                except (ValueError, AssertionError):
                    pass

                # data panel rounded-corner mask
                base_layout = style.get("layout_presets", {}).get(
                    "product_top_2_3_data_bottom_1_3", {}
                )
                data_region = base_layout.get("data_region", {})
                dw = int(data_region.get("w", out_w))
                dy = int(data_region.get("y", 0))
                dh = int(data_region.get("h", out_h - dy))
                d_radius = int(data_region.get("corner_radius", 0))
                d_mode = normalized_corner_mode(str(data_region.get("corner_mode", "all")))
                if d_radius > 0:
                    mask_key = (dw, dh, d_radius, d_mode)
                    mask_path = resources.mask_paths.get(mask_key)
                    if mask_path:
                        builder.add_mask_input(mask_path, mask_key, duration)

                # logo
                logo = resolve(builder.timeline_base, product.get("flipkart_logo"))
                if logo is not None:
                    logo = logo.resolve()
                    if logo in resources.logo_inputs:
                        builder.add_logo_input(logo, duration)

        return builder

    return builder


# ---------------------------------------------------------------------------
# Fade computation (mirrors FilterBuilder.build() logic)
# ---------------------------------------------------------------------------

def _compute_scene_fades(
    segments: list[dict[str, Any]],
    idx: int,
    style: dict[str, Any],
) -> list[str]:
    """Return fade filter strings for segment[idx] to apply in the concat pass."""
    scene = segments[idx]
    duration = float(scene["end"]) - float(scene["start"])
    next_scene = segments[idx + 1] if idx + 1 < len(segments) else None

    bridge_scene = None
    if scene.get("type") in _BRIDGE_SCENE_TYPES:
        bridge_scene = scene
    elif idx > 0 and segments[idx - 1].get("type") in _BRIDGE_SCENE_TYPES:
        bridge_scene = segments[idx - 1]
    elif next_scene and next_scene.get("type") in _BRIDGE_SCENE_TYPES:
        bridge_scene = next_scene

    scene_transition = scene_dip_transition_config(style, scene)
    prev_transition = scene_dip_transition_config(style, segments[idx - 1] if idx > 0 else None)
    next_transition = scene_dip_transition_config(style, next_scene)

    if scene_transition is not None:
        _, fade_duration, fade_color = scene_transition
    elif prev_transition is not None:
        _, fade_duration, fade_color = prev_transition
    elif next_transition is not None:
        _, fade_duration, fade_color = next_transition
    else:
        _, fade_duration, fade_color = bridge_transition_config(style, bridge_scene)

    fade_duration = max(0.0, min(fade_duration, duration / 2))

    current_is_bridge = scene.get("type") in _BRIDGE_SCENE_TYPES
    previous_is_bridge = idx > 0 and segments[idx - 1].get("type") in _BRIDGE_SCENE_TYPES
    next_is_bridge = bool(next_scene) and next_scene.get("type") in _BRIDGE_SCENE_TYPES

    if fade_duration <= 0 and (current_is_bridge or previous_is_bridge or next_is_bridge):
        _, fade_duration, fade_color = bridge_transition_config(style, bridge_scene)
        fade_duration = max(0.0, min(fade_duration, duration / 2))

    prev_type = segments[idx - 1].get("type") if idx > 0 else None
    current_type = scene.get("type")
    next_type = next_scene.get("type") if next_scene else None

    suppress_in = (
        prev_type == "product_highlight_pip_scene_with_gradient_overlay"
        and current_type == "talking_head_with_gradient_overlay"
    )
    suppress_out = (
        current_type == "product_highlight_pip_scene_with_gradient_overlay"
        and next_type == "talking_head_with_gradient_overlay"
    )

    head_pip_in = bool(prev_type) and (
        (current_type in _HEAD_PIP_TYPES and prev_type in _HEAD_FULL_TYPES)
        or (current_type in _HEAD_FULL_TYPES and prev_type in _HEAD_PIP_TYPES)
    ) and current_type not in _HEAD_TRANSITION_TYPES and prev_type not in _HEAD_TRANSITION_TYPES

    head_pip_out = bool(next_type) and (
        (current_type in _HEAD_PIP_TYPES and next_type in _HEAD_FULL_TYPES)
        or (current_type in _HEAD_FULL_TYPES and next_type in _HEAD_PIP_TYPES)
    ) and current_type not in _HEAD_TRANSITION_TYPES and next_type not in _HEAD_TRANSITION_TYPES

    fade_filters: list[str] = []
    if (
        fade_duration > 0
        and (current_is_bridge or previous_is_bridge or scene_transition is not None or head_pip_in)
        and not suppress_in
    ):
        fade_filters.append(f"fade=t=in:st=0:d={fade_duration:.3f}:color={fade_color}")

    if (
        fade_duration > 0
        and (current_is_bridge or next_is_bridge or next_transition is not None or head_pip_out)
        and not suppress_out
    ):
        fade_start = max(0.0, duration - fade_duration)
        fade_filters.append(f"fade=t=out:st={fade_start:.3f}:d={fade_duration:.3f}:color={fade_color}")

    return fade_filters


# ---------------------------------------------------------------------------
# Audio filters for concat pass (head inputs offset by N scene files)
# ---------------------------------------------------------------------------

def _build_audio_filters(
    style: dict[str, Any],
    segments: list[dict[str, Any]],
    head_offset: int,
    head_seek: dict[int, float] | None = None,
) -> tuple[list[str], str]:
    sample_rate = int(style.get("output", {}).get("sample_rate", 48000))
    filters: list[str] = []
    labels: list[str] = []
    for i, scene in enumerate(segments):
        duration = float(scene["end"]) - float(scene["start"])
        label = f"aud{i}"
        if scene_has_talking_audio(scene):
            video_idx = scene_video_index(scene)
            source_start = scene_source_start(scene)
            seek_offset = (head_seek or {}).get(video_idx, 0.0)
            trim_start = max(0.0, source_start - seek_offset)
            filters.append(
                f"[{head_offset + video_idx}:a]"
                f"atrim=start={trim_start:.3f}:end={trim_start + duration:.3f},"
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


# ---------------------------------------------------------------------------
# Concat + finalize command
# ---------------------------------------------------------------------------

def _has_any_fades(segments: list[dict[str, Any]], style: dict[str, Any]) -> bool:
    return any(_compute_scene_fades(segments, i, style) for i in range(len(segments)))



def build_concat_cmd(
    scene_paths: list[Path],
    segments: list[dict[str, Any]],
    style: dict[str, Any],
    timeline: dict[str, Any],
    timeline_path: Path,
    out_path: Path,
    clip_sequence: list[Path] | None = None,
) -> list[str]:
    """Concat + fades + watermark + audio → lossless intermediate (no subtitles).

    Subtitles are burned in a separate pass by build_subtitle_cmd() so that
    libass rendering does not block the multi-threaded filter graph here.
    Uses a stream-copy fast path when there are no fades and no watermark.
    """
    out_w, out_h, fps = output_size(style)
    encoder = style.get("encoder", {})
    video_clips = clip_sequence if clip_sequence is not None else scene_paths
    N_clips = len(video_clips)

    head_paths = talking_head_paths(timeline, timeline_path.resolve().parent)
    total_duration = max(float(s["end"]) for s in segments)

    cmd: list[str] = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-nostats",
        "-loglevel", str(encoder.get("ffmpeg_log_level", "fatal")),
    ]
    filter_threads = int(encoder.get("filter_threads", 0))
    filter_complex_threads = int(encoder.get("filter_complex_threads", filter_threads))
    if filter_threads > 0:
        cmd.extend(["-filter_threads", str(filter_threads)])
    if filter_complex_threads > 0:
        cmd.extend(["-filter_complex_threads", str(filter_complex_threads)])

    for p in video_clips:        # [0..N_clips-1]
        cmd.extend(["-i", str(p)])

    # Input-level seek for each head video: jump to the earliest segment that
    # uses it so atrim doesn't have to decode from t=0.
    head_seek: dict[int, float] = {}
    for scene in segments:
        if scene_has_talking_audio(scene):
            vid_idx = scene_video_index(scene)
            ss = scene_source_start(scene)
            if vid_idx not in head_seek or ss < head_seek[vid_idx]:
                head_seek[vid_idx] = ss
    for vid_idx, p in enumerate(head_paths):
        ss = head_seek.get(vid_idx, 0.0)
        if ss > 0:
            cmd.extend(["-ss", f"{ss:.3f}"])
        cmd.extend(["-i", str(p)])

    # video filter — no subtitles here
    vf_parts: list[str] = []
    seg_labels: list[str] = []
    if clip_sequence is not None:
        for i in range(N_clips):
            vf_parts.append(f"[{i}:v]setpts=PTS-STARTPTS[sv{i}]")
            seg_labels.append(f"[sv{i}]")
    else:
        for i, scene in enumerate(segments):
            duration = float(scene["end"]) - float(scene["start"])
            fades = _compute_scene_fades(segments, i, style)
            if fades:
                fade_chain = "," + ",".join(fades)
                vf_parts.append(
                    f"[{i}:v]trim=duration={duration:.3f},setpts=PTS-STARTPTS"
                    f"{fade_chain},setpts=PTS-STARTPTS[sv{i}]"
                )
            else:
                vf_parts.append(f"[{i}:v]trim=duration={duration:.3f},setpts=PTS-STARTPTS[sv{i}]")
            seg_labels.append(f"[sv{i}]")

    if len(seg_labels) == 1:
        vf_parts.append(f"[sv0]null[vcat]")
    else:
        vf_parts.append(f"{''.join(seg_labels)}concat=n={len(seg_labels)}:v=1:a=0[vcat]")

    vf_parts.append(f"[vcat]format=yuv420p[v]")

    # audio filter
    audio_filters, base_audio_label = _build_audio_filters(style, segments, N_clips, head_seek)
    audio_filters.append(f"[{base_audio_label}]anull[aout]")

    filter_complex = ";\n".join(vf_parts) + ";\n" + ";\n".join(audio_filters)

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "[aout]",
        "-t", f"{total_duration:.3f}",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
        "-pix_fmt", style.get("output", {}).get("pix_fmt", "yuv420p"),
        "-c:a", encoder.get("audio_codec", "aac"),
        "-b:a", encoder.get("audio_bitrate", "160k"),
        str(out_path),
    ])
    return cmd


def build_subtitle_cmd(
    intermediate_path: Path,
    captions_path: Path,
    out_path: Path,
    style: dict[str, Any],
    gradient_windows: list[GradientWindow] | None = None,
    gradient_png_path: Path | None = None,
) -> list[str]:
    """Burn subtitles into the concat intermediate and produce the final output."""
    encoder = style.get("encoder", {})
    _, _, fps = output_size(style)

    user_fonts = Path.home() / ".local/share/fonts"
    fontsdir = f":fontsdir={ffmpeg_filter_quote(user_fonts)}" if user_fonts.is_dir() else ""
    captions_q = ffmpeg_filter_quote(captions_path)

    use_gradient = (
        gradient_windows
        and gradient_png_path is not None
        and gradient_png_path.exists()
    )

    cmd: list[str] = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-nostats",
        "-loglevel", str(encoder.get("ffmpeg_log_level", "fatal")),
        "-i", str(intermediate_path),
    ]

    if use_gradient:
        # PNG input: loop it for the full video duration so the overlay filter
        # always has a frame to sample from (trim+setpts inside the overlay chain
        # resets each window's clock independently).
        cmd.extend([
            "-framerate", str(fps),
            "-i", str(gradient_png_path),
        ])
        grad_input_idx = 1

        grad_filters = build_gradient_overlay_filters(
            gradient_windows, grad_input_idx, "0:v", "grad_out", fps,
            absolute_ts=True,
        )
        sub_filter = f"[grad_out]subtitles=filename={captions_q}{fontsdir},format=yuv420p[vout]"
        vf = ";".join(grad_filters) + ";" + sub_filter
    else:
        vf = f"subtitles=filename={captions_q}{fontsdir},format=yuv420p"

    prefer_nvenc = encoder.get("prefer", "libx264") == "h264_nvenc"
    use_nvenc = prefer_nvenc and check_encoder("h264_nvenc")
    if use_nvenc:
        enc = encoder.get("nvenc", {})
        v_flags = [
            "-c:v", "h264_nvenc",
            "-preset", str(enc.get("preset", "p4")),
            "-rc", str(enc.get("rc", "vbr")),
            "-cq", str(enc.get("cq", 24)),
        ]
    else:
        enc = encoder.get("libx264", {})
        v_flags = [
            "-c:v", "libx264",
            "-preset", str(enc.get("preset", "medium")),
            "-crf", str(enc.get("crf", 23)),
        ]
    if enc.get("profile"):
        v_flags.extend(["-profile:v", str(enc["profile"])])
    if enc.get("level"):
        v_flags.extend(["-level:v", str(enc["level"])])

    encoder_threads = int(encoder.get("threads", 1))
    if encoder_threads > 0:
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

    if use_gradient:
        # filter_complex needed when we have multiple inputs (video + gradient PNG)
        cmd.extend([
            "-filter_complex", vf,
            "-map", "[vout]",
            "-map", "0:a",
        ])
    else:
        cmd.extend([
            "-vf", vf,
            "-map", "0:v",
            "-map", "0:a",
        ])
    cmd.extend([
        "-r", str(fps),
        *v_flags,
        "-pix_fmt", style.get("output", {}).get("pix_fmt", "yuv420p"),
        "-c:a", "copy",
        "-movflags", encoder.get("movflags", "+faststart"),
        *container_flags,
        str(out_path),
    ])
    return cmd


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def render_scenes_pipeline(
    style: dict[str, Any],
    timeline: dict[str, Any],
    style_path: Path,
    timeline_path: Path,
    out_path: Path,
    captions_path: Path,
    work_dir: Path,
) -> None:
    t0 = time.monotonic()

    log_step("Validating scenes")
    scenes = validate(style, timeline, style_path, timeline_path)
    log_step(f"Validated {len(scenes)} scene(s)")

    runtime_timeline_path = timeline_path.with_name(f"_{timeline_path.stem}.runtime.json")
    runtime_style_path = timeline_path.with_name(f"_{timeline_path.stem}.runtime_style.json")
    runtime_timeline_path.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")
    runtime_style_path.write_text(json.dumps(style, indent=2) + "\n", encoding="utf-8")

    log_step(f"Merging captions: {captions_path}")
    # collect_gradient_windows mirrors generated_events but only records geometry;
    # merge_ass always suppresses ASS strips internally (_gradient_windows=[])
    gradient_windows: list[GradientWindow] = collect_gradient_windows(
        runtime_style_path, runtime_timeline_path
    )
    merge_ass(runtime_style_path, runtime_timeline_path, captions_path)

    gradient_png_path: Path | None = None
    if gradient_windows:
        max_grad_right = max(w.grad_right for w in gradient_windows)
        max_panel_h = max(w.y2 - w.y1 for w in gradient_windows)
        gradient_png_path = work_dir / "gradient_overlay.png"
        ok = generate_gradient_png(max_grad_right, max_panel_h, gradient_png_path)
        if not ok:
            gradient_png_path = None
            gradient_windows = []
        else:
            log_step(f"Gradient PNG: {gradient_png_path.name}  ({max_grad_right}x{max_panel_h}, {len(gradient_windows)} window(s))")

    # build segments (same logic as FilterBuilder.build_segments)
    segments: list[dict[str, Any]] = []
    cursor = 0.0
    for scene in scenes:
        start = float(scene["start"])
        if start > cursor + 0.001:
            segments.append({"type": "blank_scene", "start": cursor, "end": start, "_generated_gap": True})
        segments.append(scene)
        cursor = float(scene["end"])

    log_step(f"Preparing shared resources ({len(segments)} segment(s))")
    resources = prepare_resources(style, timeline, timeline_path, scenes, work_dir)

    log_step("Pre-rendering hero canvases (CPU numpy)")
    hero_canvas_paths = render_hero_canvases(
        segments, resources, style, timeline, timeline_path, work_dir
    )
    resources.hero_canvas_paths = hero_canvas_paths
    log_step(f"  {len(hero_canvas_paths)} hero canvas(es) pre-rendered")

    scene_paths: list[Path] = []
    for i, scene in enumerate(segments):
        prev_scene = segments[i - 1] if i > 0 else None
        scene_type = scene.get("type", "?")
        duration = float(scene["end"]) - float(scene["start"])
        scene_path = work_dir / f"scene_{i:04d}.mp4"
        log_step(f"[{i+1}/{len(segments)}] {scene_type} {duration:.2f}s")
        builder = setup_scene_builder(
            scene, prev_scene, resources, style, timeline, timeline_path, seg_idx=i
        )
        cmd = builder.build_scene_cmd(scene, prev_scene, scene_path)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed for segment {i} ({scene_type}):\n"
                + (result.stderr or result.stdout or "")[:600]
            )
        size_kb = scene_path.stat().st_size // 1024
        log_step(f"    → {scene_path.name}  {size_kb} KB")
        scene_paths.append(scene_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat_intermediate = work_dir / "concat_pre.mp4"

    # Pass 1: concat + fades + watermark + audio → lossless intermediate (no subtitles)
    log_step("Running concat pass")
    concat_cmd = build_concat_cmd(
        scene_paths, segments, style, timeline, timeline_path, concat_intermediate,
    )
    result = subprocess.run(concat_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Concat pass failed:\n" + (result.stderr or result.stdout or "")[:1000]
        )

    # Pass 2: burn subtitles (+ gradient PNG overlay if any) → final output
    log_step("Burning subtitles")
    subtitle_cmd = build_subtitle_cmd(
        concat_intermediate, captions_path, out_path, style,
        gradient_windows=gradient_windows,
        gradient_png_path=gradient_png_path,
    )
    result = subprocess.run(subtitle_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Subtitle pass failed:\n" + (result.stderr or result.stdout or "")[:2000]
        )

    elapsed = time.monotonic() - t0
    log_step(f"Done → {out_path}  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scene-by-scene renderer for product_pip_intro timelines."
    )
    parser.add_argument("--style", default="code/global_style.json")
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--work-dir", default=None,
        help="Directory for scene MKV intermediates (default: auto tempdir, deleted on exit)",
    )
    parser.add_argument("--vid1", default=None)
    parser.add_argument("--vid2", default=None)
    parser.add_argument("--bridge-duration", type=float, default=None)
    parser.add_argument("--bridge-product-id", default=None)
    parser.add_argument("--bridge-image-index", type=int, default=0)
    parser.add_argument("--bridge-transition", default=None)
    args = parser.parse_args()

    style_path = Path(args.style)
    timeline_path = Path(args.timeline)

    style = load_json(style_path)
    timeline = load_json(timeline_path)
    timeline = apply_timeline_auto_bridge(style, timeline, timeline_path)
    timeline = apply_video_args(
        style, timeline, timeline_path,
        args.vid1, args.vid2,
        args.bridge_duration, args.bridge_product_id,
        args.bridge_image_index, args.bridge_transition,
        False,
    )

    out_path = Path(args.out or timeline.get("output", "output.mp4"))
    captions_path = out_path.with_suffix(".ass")

    def _run(work_dir: Path) -> None:
        render_scenes_pipeline(
            style, timeline, style_path, timeline_path, out_path, captions_path, work_dir
        )

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        _run(work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="render_scenes_") as tmp:
            _run(Path(tmp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
