#!/usr/bin/env python3
"""
Merge existing stable-ts ASS subtitles with product overlay ASS events.

This module intentionally does not regenerate stable-ts subtitles. It reads the
existing ASS file named by the timeline and appends deterministic product/title,
price/rating, bullet, CTA, and Flipkart end-card events.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import textwrap
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GradientWindow:
    """Geometry + timing for one highlight gradient panel, for FFmpeg overlay."""
    start: float       # absolute video time (seconds)
    end: float
    y1: int            # top of gradient rect in ASS/video pixel coords
    y2: int            # bottom
    grad_right: int    # rightmost pixel of gradient (left-edge is always 0)
    fade_ms: int       # fade-in AND fade-out duration in ms (mirrors slide_ms)


STYLE_FORMAT = (
    "Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, "
    "Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)

EVENT_FORMAT = "Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"


PRODUCT_SCENE_TYPES = {
    "product_scene",
    "talking_head_product_strip_scene",
    "product_bridge_scene",
    "pip_scene",
    "talk_to_pip_scene",
    "pip_to_talk_scene",
    "product_highlight_pip_scene_with_gradient_overlay",
}

# Full-screen talking head scenes that should show keyword-synced product details
AUDIO_OVERLAY_SCENE_TYPES = {
    "talking_head_scene",
    "talking_head_with_gradient_overlay",
}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def ass_time(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis == 100:
        secs += 1
        centis = 0
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def ass_seconds(value: str) -> float:
    hours, minutes, seconds = value.strip().split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def escape_ass_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{").replace("}", "\\}")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", r"\N")
    return text


def normalize_for_compare(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


_APPROVED_BRAND_LIST_PATH = Path(__file__).with_name("approved_brand_list.txt")
_APPROVED_BRANDS: set[str] | None = None
_INVALID_HIGHLIGHT_VALUES = frozenset({"na", "n a", "n/a", "not applicable", "none", "null"})


def approved_brand_names() -> set[str]:
    global _APPROVED_BRANDS
    if _APPROVED_BRANDS is None:
        try:
            _APPROVED_BRANDS = {
                normalize_for_compare(line)
                for line in _APPROVED_BRAND_LIST_PATH.read_text(encoding="utf-8").splitlines()
                if normalize_for_compare(line)
            }
        except OSError:
            _APPROVED_BRANDS = set()
    return _APPROVED_BRANDS


def _is_invalid_highlight_value(value: str) -> bool:
    normalized = normalize_for_compare(value)
    return not normalized or normalized in _INVALID_HIGHLIGHT_VALUES


def _filter_overlay_gradient_items(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    filtered: list[tuple[str, str]] = []
    approved_brands = approved_brand_names()
    for raw_label, raw_value in items:
        label = re.sub(r"\s+", " ", str(raw_label or "")).strip()
        value = re.sub(r"\s+", " ", str(raw_value or "")).strip()
        if _is_invalid_highlight_value(value):
            continue
        label_key = normalize_for_compare(label)
        if label_key == "type":
            continue
        if label_key == "brand" and approved_brands and normalize_for_compare(value) not in approved_brands:
            continue
        filtered.append((label, value))
    return filtered


_PRODUCT_NAME_TAIL_NOISE = frozenset({
    # pack / quantity
    "set", "pack", "combo", "piece", "pieces", "units", "nos", "pcs", "pc",
    # measurement units
    "litre", "liter", "litres", "liters", "ml", "kg", "gm", "gram", "grams",
    "cm", "mm", "inch", "inches", "ft", "lbs",
    # materials/colours that appear as trailing qualifiers
    "plastic", "metal", "steel", "stainless", "iron", "aluminium", "aluminum",
    "wooden", "wood", "cotton", "polyester", "nylon", "rubber", "leather",
    "silicone", "ceramic", "fiber", "fibre",
    "black", "white", "red", "blue", "green", "yellow", "brown",
    "grey", "gray", "pink", "orange", "purple", "golden", "silver",
    # size qualifiers
    "large", "medium", "small", "xl", "xxl", "xs", "xxs", "size",
    # functional
    "for", "with", "and", "the", "of", "in", "by", "a", "an",
})

# Words to remove from anywhere (not just tail) — only pure function words
_PRODUCT_NAME_FUNCTION_WORDS = frozenset({
    "for", "with", "and", "the", "of", "in", "by", "a", "an",
})


def shorten_product_name(name: str, max_words: int = 3) -> str:
    """Return at most max_words words.

    Strategy:
    1. Strip tail noise (quantities, units, colours, trailing materials) until
       ≤ max_words remain or the tail is no longer noise — preserving order so
       leading descriptors like 'Aluminium' stay intact.
    2. Remove pure function words (for/of/with/…) from anywhere.
    3. Take the first max_words of whatever is left.
    """
    words = re.sub(r"\s+", " ", name).strip().split()
    if len(words) <= max_words:
        return name

    # Step 1 — strip from the tail
    result = words[:]
    while len(result) > max_words:
        tail = result[-1]
        if tail.lower() in _PRODUCT_NAME_TAIL_NOISE or re.fullmatch(r"[\d.]+", tail):
            result.pop()
        else:
            break

    # Step 2 — remove function words from anywhere
    cleaned = [w for w in result if w.lower() not in _PRODUCT_NAME_FUNCTION_WORDS]
    if not cleaned:
        cleaned = result  # guard: don't produce an empty name

    return " ".join(cleaned[:max_words])


def shorten(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    placeholder = "..." if max_chars > 3 else ""
    shortened = textwrap.shorten(text, width=max_chars, placeholder=placeholder)
    if shortened:
        return shortened.strip()
    return text[: max(max_chars - len(placeholder), 1)].rstrip() + placeholder


def product_by_id(timeline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    products: dict[str, dict[str, Any]] = {}
    for product in timeline.get("products", []):
        pid = product.get("id")
        if pid:
            products[str(pid)] = product
    return products


def resolve(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def talking_head_paths(timeline: dict[str, Any], base: Path) -> list[Path]:
    values = timeline.get("talking_head_videos")
    if values is None:
        single = timeline.get("talking_head_video")
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


def media_duration(path: Path) -> float:
    probe_result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    for line in probe_result.stdout.splitlines():
        if "Duration:" not in line:
            continue
        stamp = line.split("Duration:", 1)[1].split(",", 1)[0].strip()
        hours, minutes, seconds = stamp.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Could not determine media duration: {path}")


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


def selected_bullets(product: dict[str, Any], style: dict[str, Any]) -> list[Any]:
    rules = style.get("salient_point_rules", {})
    max_bullets = int(rules.get("max_bullets", 3))
    max_chars = int(rules.get("max_chars_per_bullet", 38))
    source = product.get("selected_bullets") or product.get("raw_features") or []

    bullets: list[Any] = []
    seen: set[str] = set()
    avoid_words = {
        "best",
        "amazing",
        "unbelievable",
        "guaranteed",
        "world class",
        "revolutionary",
    }

    for raw in source:
        if isinstance(raw, dict):
            label = re.sub(r"\s+", " ", str(raw.get("label") or raw.get("name") or "")).strip(" -•\t:")
            value = re.sub(r"\s+", " ", str(raw.get("value") or "")).strip(" -•\t:")
            if not label or not value:
                continue
            short = {"label": shorten(label, max_chars), "value": shorten(value, max_chars)}
            key = normalize_for_compare(f"{short['label']} {short['value']}")
            if not key or key in seen:
                continue
            seen.add(key)
            bullets.append(short)
            if len(bullets) >= max_bullets:
                break
            continue

        candidate = re.sub(r"\s+", " ", str(raw)).strip(" -•\t")
        if not candidate:
            continue
        lowered = candidate.lower()
        if any(word in lowered for word in avoid_words):
            continue
        short = shorten(candidate, max_chars)
        key = normalize_for_compare(short)
        if not key or key in seen:
            continue
        seen.add(key)
        bullets.append(short)
        if len(bullets) >= max_bullets:
            break

    return bullets


def style_line(name: str, cfg: dict[str, Any], *, secondary: str | None = None) -> str:
    font = cfg.get("font", "Montserrat SemiBold")
    size = cfg.get("font_size", 36)
    primary = cfg.get("color") or cfg.get("normal_color") or "&H00FFFFFF"
    secondary_colour = secondary or cfg.get("highlight_color") or "&H00FFFFFF"
    outline_colour = cfg.get("outline_color", "&H00000000")
    back_colour = cfg.get("back_color", "&H80000000")
    outline = cfg.get("outline", 3)
    shadow = cfg.get("shadow", 1)
    alignment = cfg.get("alignment", 7)
    margin_l = cfg.get("margin_l", 20)
    margin_r = cfg.get("margin_r", 20)
    margin_v = cfg.get("margin_v", 20)
    bold_flag = -1 if cfg.get("bold") else 0
    return (
        f"Style: {name},{font},{size},{primary},{secondary_colour},{outline_colour},"
        f"{back_colour},{bold_flag},0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},"
        f"{margin_l},{margin_r},{margin_v},1"
    )


def required_style_lines(style: dict[str, Any]) -> dict[str, str]:
    ass_styles = style.get("ass_styles", {})
    subtitle_cfg = style.get("subtitle_style", {})
    out: dict[str, str] = {}
    for key in [
        "karaoke",
        "product_title",
        "product_bullet",
        "product_cta",
        "flipkart_brand",
        "highlight_title",
        "highlight_label",
        "highlight_value",
        "product_name_intro",
    ]:
        cfg = deepcopy(ass_styles.get(key, {}))
        if key == "karaoke":
            # Keep karaoke fallback styles aligned with the main subtitle style so
            # any inherited/raw karaoke rows render with the configured subtitle font.
            cfg["font"] = subtitle_cfg.get("font", cfg.get("font", "Montserrat SemiBold"))
            cfg["font_size"] = subtitle_cfg.get("font_size", cfg.get("font_size", 40))
            cfg["normal_color"] = subtitle_cfg.get("primary_color", cfg.get("normal_color", "&H00FFFFFF"))
            cfg["highlight_color"] = subtitle_cfg.get("secondary_color", cfg.get("highlight_color", "&H00FFFFFF"))
            cfg["back_color"] = subtitle_cfg.get("back_color", cfg.get("back_color", "&H80000000"))
            cfg["outline"] = subtitle_cfg.get("outline", cfg.get("outline", 3))
            cfg["shadow"] = subtitle_cfg.get("shadow", cfg.get("shadow", 1))
        name = cfg.get("name") or "".join(part.capitalize() for part in key.split("_"))
        out[name] = style_line(name, cfg)
    return out


def ensure_style_section(lines: list[str]) -> tuple[list[str], int]:
    for i, line in enumerate(lines):
        if line.strip().lower() == "[v4+ styles]":
            return lines, i

    insert_at = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "[script info]":
            insert_at = i + 1
            break

    block = [
        "",
        "[V4+ Styles]",
        f"Format: {STYLE_FORMAT}",
    ]
    lines[insert_at:insert_at] = block
    return lines, insert_at + 1


def ensure_script_info_resolution(lines: list[str], style: dict[str, Any]) -> list[str]:
    output = style.get("output", {})
    width = str(int(output.get("width", 720)))
    height = str(int(output.get("height", 960)))
    seen_x = False
    seen_y = False
    in_script_info = False
    insert_at = 0
    out: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "[script info]":
            in_script_info = True
            insert_at = len(out) + 1
            out.append(line)
            continue
        if in_script_info and stripped.startswith("[") and stripped.endswith("]"):
            if not seen_x:
                out.insert(insert_at, f"PlayResX: {width}")
                insert_at += 1
            if not seen_y:
                out.insert(insert_at, f"PlayResY: {height}")
                insert_at += 1
            in_script_info = False
        if in_script_info and stripped.lower().startswith("playresx:"):
            out.append(f"PlayResX: {width}")
            seen_x = True
            continue
        if in_script_info and stripped.lower().startswith("playresy:"):
            out.append(f"PlayResY: {height}")
            seen_y = True
            continue
        out.append(line)

    if in_script_info:
        if not seen_x:
            out.append(f"PlayResX: {width}")
        if not seen_y:
            out.append(f"PlayResY: {height}")
    return out


def _subtitle_style_values(style: dict[str, Any]) -> tuple[str, int, str, str, int, int, str]:
    cfg = style.get("subtitle_style")
    if not isinstance(cfg, dict):
        raise ValueError("Missing required style.subtitle_style object")

    font = str(cfg.get("font", "")).strip()
    if not font:
        raise ValueError("Missing required style.subtitle_style.font")

    if "font_size" not in cfg:
        raise ValueError("Missing required style.subtitle_style.font_size")
    font_size = int(cfg["font_size"])

    primary_color = str(cfg.get("primary_color", "")).strip()
    if not primary_color:
        raise ValueError("Missing required style.subtitle_style.primary_color")

    secondary_color = str(cfg.get("secondary_color", "")).strip()
    if not secondary_color:
        raise ValueError("Missing required style.subtitle_style.secondary_color")

    outline = int(cfg.get("outline", 0))
    shadow = int(cfg.get("shadow", 2))
    back_color = str(cfg.get("back_color", "&H80000000")).strip() or "&H80000000"
    return font, font_size, primary_color, secondary_color, outline, shadow, back_color


def normalize_default_subtitle_style(
    lines: list[str],
    font: str = "Montserrat SemiBold",
    font_size: int = 55,
    primary_color: str = "&H00FF0000",
    secondary_color: str = "&H00FFFFFF",
    outline: int = 0,
    shadow: int = 2,
    back_color: str = "&H80000000",
) -> list[str]:
    out: list[str] = []
    for line in lines:
        if not line.lstrip().startswith("Style:"):
            out.append(line)
            continue

        prefix, payload = line.split(":", 1)
        fields = [field.strip() for field in payload.split(",")]
        if len(fields) < 23:
            out.append(line)
            continue

        style_key = fields[0].strip().lower()
        if style_key not in {"default", "karaoke"}:
            out.append(line)
            continue

        fields[1] = font
        fields[2] = str(font_size)
        if style_key == "default":
            fields[3] = primary_color
            fields[4] = secondary_color  # unspoken words colour; switches to primary as each word is spoken
            fields[6] = back_color       # shadow colour
            fields[7] = "0"              # not bold (weight comes from font face name)
            fields[15] = "1"             # BorderStyle: outline+shadow
            fields[16] = str(outline)    # outline width
            fields[17] = str(shadow)     # shadow depth
            fields[18] = "2"             # alignment: bottom center
            fields[21] = "55"            # MarginV

        out.append(prefix + ": " + ",".join(fields))
    return out


def enforce_subtitle_style_overrides(lines: list[str], style: dict[str, Any]) -> list[str]:
    """Force Default/Karaoke ASS styles to match subtitle_style config."""
    font, font_size, primary_color, secondary_color, outline, shadow, back_color = _subtitle_style_values(style)
    return normalize_default_subtitle_style(
        lines,
        font,
        font_size,
        primary_color,
        secondary_color,
        outline=outline,
        shadow=shadow,
        back_color=back_color,
    )


def existing_style_names(lines: list[str]) -> set[str]:
    names: set[str] = set()
    for line in lines:
        if line.startswith("Style:"):
            payload = line.split(":", 1)[1].strip()
            name = payload.split(",", 1)[0].strip()
            if name:
                names.add(name)
    return names


def ensure_styles(lines: list[str], style: dict[str, Any]) -> list[str]:
    lines, section_idx = ensure_style_section(lines)
    names = existing_style_names(lines)
    needed = required_style_lines(style)

    insert_at = section_idx + 1
    for i in range(section_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            break
        insert_at = i + 1

    if not any(line.startswith("Format:") for line in lines[section_idx + 1 : insert_at]):
        lines.insert(section_idx + 1, f"Format: {STYLE_FORMAT}")
        insert_at += 1

    for name, line in needed.items():
        if name not in names:
            lines.insert(insert_at, line)
            insert_at += 1
            names.add(name)
    return lines


def ensure_events_section(lines: list[str]) -> list[str]:
    for i, line in enumerate(lines):
        if line.strip().lower() == "[events]":
            section_end = len(lines)
            for j in range(i + 1, len(lines)):
                if lines[j].strip().startswith("[") and lines[j].strip().endswith("]"):
                    section_end = j
                    break
            if not any(line.startswith("Format:") for line in lines[i + 1 : section_end]):
                lines.insert(i + 1, f"Format: {EVENT_FORMAT}")
            return lines
    lines.extend(["", "[Events]", f"Format: {EVENT_FORMAT}"])
    return lines


def remove_event_rows(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    in_events = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "[events]":
            in_events = True
            cleaned.append(line)
            continue
        if in_events and stripped.startswith("[") and stripped.endswith("]"):
            in_events = False
        if in_events and (line.startswith("Dialogue:") or line.startswith("Comment:")):
            continue
        cleaned.append(line)
    return cleaned


def ass_dialogue_fields(line: str) -> list[str] | None:
    if not line.startswith("Dialogue:"):
        return None
    payload = line.split(":", 1)[1].strip()
    fields = payload.split(",", 9)
    return fields if len(fields) == 10 else None


def uppercase_ass_visible_text(text: str) -> str:
    parts = re.split(r"(\{[^}]*\})", text)
    return "".join(part if part.startswith("{") and part.endswith("}") else part.upper() for part in parts)


def subtitle_ass_paths(timeline: dict[str, Any], timeline_path: Path) -> list[Path]:
    base = timeline_path.resolve().parent
    configured = timeline.get("base_subtitles_ass_files")
    if configured is None:
        single = timeline.get("base_subtitles_ass")
        configured = [single] if single else []
    paths: list[Path] = []
    for value in configured:
        path = resolve(base, str(value))
        if path is None:
            continue
        paths.append(path)
    return paths


def shifted_subtitle_events(
    style: dict[str, Any],
    timeline: dict[str, Any],
    timeline_path: Path,
) -> list[str]:
    paths = subtitle_ass_paths(timeline, timeline_path)
    if not paths:
        return []
    scenes = normalize_scene_times(style, timeline, timeline_path)
    subtitle_cfg = style.get("subtitle_style", {})
    primary_color = str(subtitle_cfg.get("primary_color", "&H00FF0000"))
    secondary_color = str(subtitle_cfg.get("secondary_color", "&H00FFFFFF"))
    events: list[str] = []

    for scene in scenes:
        if scene.get("show_subtitles") is False:
            continue
        if "talking_head_video_index" not in scene and "video_index" not in scene:
            continue
        video_index = scene_video_index(scene)
        if video_index < 0 or video_index >= len(paths):
            continue
        sub_path = paths[video_index]
        if not sub_path.exists():
            raise FileNotFoundError(f"Stable-ts ASS not found for video {video_index}: {sub_path}")

        scene_start = float(scene["start"])
        scene_end = float(scene["end"])
        source_start = scene_source_start(scene)
        source_end = source_start + (scene_end - scene_start)

        for line in sub_path.read_text(encoding="utf-8").splitlines():
            fields = ass_dialogue_fields(line)
            if fields is None:
                continue
            try:
                sub_start = ass_seconds(fields[1])
                sub_end = ass_seconds(fields[2])
            except ValueError:
                continue
            clipped_start = max(sub_start, source_start)
            clipped_end = min(sub_end, source_end)
            if clipped_end <= clipped_start:
                continue

            fields[9] = uppercase_ass_visible_text(fields[9])
            fields[9] = re.sub(r"(\w)(\{\\k)", r"\1 \2", fields[9])

            if r"\k" in fields[9] and fields[3].strip() == "Default":
                expanded_fields = _expand_karaoke_fields(
                    fields,
                    primary_color,
                    secondary_color,
                    clip_start_cs=int(round(clipped_start * 100)),
                    clip_end_cs=int(round(clipped_end * 100)),
                )
                for expanded in expanded_fields:
                    mapped = list(expanded)
                    mapped_start = ass_seconds(mapped[1])
                    mapped_end = ass_seconds(mapped[2])
                    mapped[1] = ass_time(scene_start + mapped_start - source_start)
                    mapped[2] = ass_time(scene_start + mapped_end - source_start)
                    events.append("Dialogue: " + ",".join(mapped))
                continue

            fields[1] = ass_time(scene_start + clipped_start - source_start)
            fields[2] = ass_time(scene_start + clipped_end - source_start)
            events.append("Dialogue: " + ",".join(fields))
    return events


def dialogue(start: float, end: float, style_name: str, text: str, *, layer: int = 20) -> str:
    return (
        f"Dialogue: {layer},{ass_time(start)},{ass_time(end)},{style_name},,"
        f"0,0,0,,{text}"
    )


def intervals_for_subtitle_safe_area(
    style: dict[str, Any],
    timeline: dict[str, Any],
    timeline_path: Path | None = None,
) -> list[tuple[float, float]]:
    safe = style.get("subtitle_safe_area", {})
    if safe.get("enabled") is False:
        return []
    active_types = set(
        safe.get(
            "active_scene_types",
            [
                "product_scene",
                "talking_head_product_strip_scene",
                "product_bridge_scene",
                "pip_scene",
                "talk_to_pip_scene",
                "pip_to_talk_scene",
                "flipkart_end_scene",
            ],
        )
    )
    intervals: list[tuple[float, float]] = []
    for scene in normalize_scene_times(style, timeline, timeline_path):
        if scene.get("type") in active_types:
            intervals.append((float(scene["start"]), float(scene["end"])))
    return intervals


def overlaps_any(start: float, end: float, intervals: list[tuple[float, float]]) -> bool:
    return any(start < interval_end and end > interval_start for interval_start, interval_end in intervals)


def move_existing_subtitles_above_safe_area(
    lines: list[str],
    style: dict[str, Any],
    timeline: dict[str, Any],
    timeline_path: Path | None = None,
) -> list[str]:
    intervals = intervals_for_subtitle_safe_area(style, timeline, timeline_path)
    if not intervals:
        return lines

    output_h = int(style.get("output", {}).get("height", 960))
    safe_y = int(style.get("subtitle_safe_area", {}).get("y", 600))
    margin_v = max(output_h - safe_y, 0)
    product_style_names = {
        cfg.get("name")
        for cfg in style.get("ass_styles", {}).values()
        if isinstance(cfg, dict) and cfg.get("name")
    }

    moved: list[str] = []
    for line in lines:
        if not line.startswith("Dialogue:"):
            moved.append(line)
            continue
        payload = line.split(":", 1)[1].strip()
        fields = payload.split(",", 9)
        if len(fields) != 10:
            moved.append(line)
            continue
        if fields[3] in product_style_names:
            moved.append(line)
            continue
        try:
            start = ass_seconds(fields[1])
            end = ass_seconds(fields[2])
        except (ValueError, IndexError):
            moved.append(line)
            continue
        if overlaps_any(start, end, intervals):
            fields[7] = str(margin_v)
            moved.append("Dialogue: " + ",".join(fields))
        else:
            moved.append(line)
    return moved


def positioned_text(x: int, y: int, text: Any, extra_tags: str = "") -> str:
    return r"{" + f"\\pos({x},{y}){extra_tags}" + r"}" + escape_ass_text(text)


def moving_text(x: int, start_y: int, end_y: int, duration_ms: int, text: Any, extra_tags: str = "") -> str:
    return (
        r"{"
        + f"\\move({x},{start_y},{x},{end_y},0,{duration_ms}){extra_tags}"
        + r"}"
        + escape_ass_text(text)
    )


def _slide_left_events(
    start: float,
    end: float,
    style_name: str,
    text: str,
    x: int,
    y: int,
    start_x: int,
    slide_in_ms: int,
    slide_out_ms: int,
    layer: int = 24,
) -> list[str]:
    """Slide-in from left entry event + slide-out to left exit event."""
    exit_dur = slide_out_ms / 1000.0
    entry_end = max(start, end - exit_dur)
    move_in = r"{" + f"\\move({start_x},{y},{x},{y},0,{slide_in_ms})" + r"}" + text
    result = [dialogue(start, entry_end, style_name, move_in, layer=layer)]
    if slide_out_ms > 0 and entry_end < end:
        move_out = r"{" + f"\\move({x},{y},{start_x},{y},0,{slide_out_ms})" + r"}" + text
        result.append(dialogue(entry_end, end, style_name, move_out, layer=layer))
    return result


def split_bullet_label_value(text: Any) -> tuple[str, str] | None:
    if isinstance(text, dict):
        label = str(text.get("label") or text.get("name") or "").strip(" -•\t:")
        value = str(text.get("value") or "").strip(" -•\t:")
        if label and value:
            return label, value
        return None
    text = str(text)
    if ":" not in text:
        return None
    label, value = text.split(":", 1)
    label = label.strip(" -•\t:")
    value = value.strip(" -•\t:")
    if not label or not value:
        return None
    return label, value


def apply_text_case(text: str, text_case: str | None) -> str:
    if not text_case:
        return text
    if str(text_case).lower() in {"upper", "uppercase", "caps", "all_caps"}:
        return text.upper()
    if str(text_case).lower() in {"lower", "lowercase"}:
        return text.lower()
    return text


def product_data_region(style: dict[str, Any], scene: dict[str, Any]) -> tuple[int, int, int, int]:
    if scene.get("type") == "talking_head_product_strip_scene":
        region = (
            style.get("layout_presets", {})
            .get("talking_head_with_product_data_strip", {})
            .get("strip", {})
        )
    else:
        region = (
            style.get("layout_presets", {})
            .get("product_top_2_3_data_bottom_1_3", {})
            .get("data_region", {})
        )
    output = style.get("output", {})
    out_w = int(output.get("width", 720))
    out_h = int(output.get("height", 960))
    x = int(region.get("x", 0))
    y = int(region.get("y", int(out_h * 2 / 3)))
    w = int(region.get("w", out_w))
    h = int(region.get("h", out_h - y))
    return x, y, w, h


def product_component_positions(
    style: dict[str, Any],
    scene: dict[str, Any],
    title_cfg: dict[str, Any],
    bullets_cfg: dict[str, Any],
) -> tuple[int, int, int, int, int]:
    component = style.get("product_data_layout", {}).get("component", {})
    region_x, region_y, _region_w, _region_h = product_data_region(style, scene)
    if not component:
        bullet_x = int(bullets_cfg.get("x", 60))
        return (
            int(title_cfg.get("x", 60)),
            int(title_cfg.get("y", 675)),
            bullet_x,
            int(bullets_cfg.get("value_x", bullet_x + 160)),
            int(bullets_cfg.get("start_y", 785)),
        )

    margin_x = int(component.get("margin_x", 40))
    margin_y = int(component.get("margin_y", 40))
    title_x = region_x + margin_x
    title_y = region_y + int(component.get("title_offset_y", margin_y))
    bullet_name_x = region_x + margin_x
    if "value_x" in component:
        bullet_value_x = region_x + int(component["value_x"])
    else:
        bullet_value_x = bullet_name_x + int(component.get("value_offset_x", 160))
    bullet_y = region_y + int(component.get("bullet_offset_y", title_y - region_y + 84))
    return title_x, title_y, bullet_name_x, bullet_value_x, bullet_y


def product_text_payload(
    style: dict[str, Any],
    scene: dict[str, Any],
    x: int,
    y: int,
    text: Any,
    extra_tags: str = "",
) -> str:
    if scene.get("type") == "talking_head_product_strip_scene":
        strip = (
            style.get("layout_presets", {})
            .get("talking_head_with_product_data_strip", {})
            .get("strip", {})
        )
        if str(strip.get("transition", "none")) == "slide_up":
            duration_ms = int(max(0.0, float(strip.get("transition_duration", 0.0))) * 1000)
            if duration_ms > 0:
                _region_x, _region_y, _region_w, region_h = product_data_region(style, scene)
                return moving_text(x, y + region_h, y, duration_ms, text, extra_tags)
    return positioned_text(x, y, text, extra_tags)


def product_scene_events(
    scene: dict[str, Any],
    product: dict[str, Any],
    style: dict[str, Any],
) -> list[str]:
    layout = style.get("product_data_layout", {})
    title_cfg = layout.get("title", {})
    bullets_cfg = layout.get("bullets", {})
    cta_cfg = layout.get("cta", {})
    ass_styles = style.get("ass_styles", {})

    title_style = ass_styles.get("product_title", {}).get("name", "ProductTitle")
    bullet_style = ass_styles.get("product_bullet", {}).get("name", "ProductBullet")
    cta_style = ass_styles.get("product_cta", {}).get("name", "ProductCTA")

    start = float(scene["start"])
    end = float(scene["end"])
    title = shorten(product.get("name", ""), int(title_cfg.get("max_chars", 32)))
    title = apply_text_case(title, title_cfg.get("text_case"))
    text_tags = ""
    if scene.get("type") == "talking_head_product_strip_scene":
        strip = (
            style.get("layout_presets", {})
            .get("talking_head_with_product_data_strip", {})
            .get("strip", {})
        )
        if str(strip.get("transition", "none")) == "cross_dissolve":
            fade_ms = int(max(0.0, float(strip.get("transition_duration", 0.0))) * 1000)
            if fade_ms > 0:
                text_tags = f"\\fad({fade_ms},0)"
    elif (
        scene.get("type") == "talk_to_pip_scene"
        and scene.get("pip_exit_transition") in {"slide_out", "slide_out_right"}
    ):
        scene_dur = end - start
        slide_cfg = style.get("transitions", {}).get("pip_slide_out", {})
        exit_dur = float(scene.get("pip_exit_duration", slide_cfg.get("duration", 0.6)))
        exit_dur = max(min(exit_dur, scene_dur), 0.001)
        anim_dur = float(scene.get("pip_entry_duration") if scene.get("pip_entry_duration") is not None else style.get("transitions", {}).get("talk_to_pip", {}).get("duration", 0.8))
        exit_start_in_scene = max(scene_dur - exit_dur, min(anim_dur, scene_dur))
        fade_ms = int(exit_dur * 1000)
        end = start + exit_start_in_scene + exit_dur
        text_tags = f"\\fad(0,{fade_ms})"

    title_x, title_y, bullet_name_x, bullet_value_x, bullet_y = product_component_positions(
        style,
        scene,
        title_cfg,
        bullets_cfg,
    )

    events: list[str] = []

    bullet_x = int(bullets_cfg.get("x", 60))
    if not style.get("product_data_layout", {}).get("component"):
        bullet_name_x = int(bullets_cfg.get("name_x", bullet_x))
        bullet_value_x = int(bullets_cfg.get("value_x", bullet_x + 160))
    bullet_font_size = int(ass_styles.get("product_bullet", {}).get("font_size", 32))
    gap = bullet_font_size + int(bullets_cfg.get("gap", 10))
    for idx, bullet in enumerate(selected_bullets(product, style)):
        y = bullet_y + idx * gap
        label_value = split_bullet_label_value(bullet)
        if label_value is None:
            events.append(
                dialogue(
                    start,
                    end,
                    bullet_style,
                    product_text_payload(style, scene, bullet_x, y, bullet, text_tags),
                    layer=24,
                )
            )
            continue

        label, value = label_value
        events.extend(
            [
                dialogue(
                    start,
                    end,
                    bullet_style,
                    product_text_payload(style, scene, bullet_name_x, y, label, text_tags),
                    layer=24,
                ),
                dialogue(
                    start,
                    end,
                    bullet_style,
                    product_text_payload(style, scene, bullet_value_x, y, value, text_tags),
                    layer=24,
                ),
            ]
        )

    if product.get("cta") and scene.get("show_cta") is not False:
        events.append(
            dialogue(
                start,
                end,
                cta_style,
                product_text_payload(
                    style,
                    scene,
                    int(cta_cfg.get("x", 360)),
                    int(cta_cfg.get("y", 925)),
                    product["cta"],
                    text_tags,
                ),
                layer=25,
            )
        )
    return events


def flipkart_events(scene: dict[str, Any], product: dict[str, Any], style: dict[str, Any]) -> list[str]:
    start = float(scene["start"])
    end = float(scene["end"])
    brand_style = style.get("ass_styles", {}).get("flipkart_brand", {}).get("name", "FlipkartBrand")
    cta_style = style.get("ass_styles", {}).get("product_cta", {}).get("name", "ProductCTA")
    fallback = style.get("flipkart_end_scene_style", {}).get("logo_missing_fallback_text", "Flipkart")
    brand_text = "" if product.get("flipkart_logo") else fallback
    cta = product.get("cta")
    price = product.get("price", "")

    events: list[str] = []
    if brand_text:
        events.append(
            dialogue(start, end, brand_style, positioned_text(360, 675, brand_text), layer=28)
        )
    if scene.get("show_cta") is not False and cta:
        cta_line = f"{cta}  {price}".strip()
        events.append(dialogue(start, end, cta_style, positioned_text(360, 895, cta_line), layer=29))
    return events


_SUBTITLE_STOPWORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "of", "to", "and", "or", "for", "with",
    "this", "that", "so", "also", "get", "you", "your", "its", "are", "has",
    "be", "on", "at", "by", "as", "if", "but", "not", "we", "they", "have",
})


def _parse_karaoke_words(text: str) -> list[tuple[int, str]]:
    """Parse \\k/\\kf/\\ko karaoke tags, return (centisecs, display_word) pairs."""
    result = []
    for m in re.finditer(r'\{\\k[fo]?(\d+)\}([^{]*)', text):
        dur_cs = int(m.group(1))
        word = m.group(2).rstrip()
        if word.strip():
            result.append((dur_cs, word))
    return result


def _ass_time_to_cs(t: str) -> int:
    """Parse ASS time 'H:MM:SS.cc' to centiseconds."""
    t = t.strip()
    h, m, rest = t.split(":")
    s, cs = rest.split(".")
    return int(h) * 360000 + int(m) * 6000 + int(s) * 100 + int(cs)


def _cs_to_ass_time(cs: int) -> str:
    """Convert centiseconds to ASS time format 'H:MM:SS.cc'."""
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


_K_TAG_RE = re.compile(r'\{\\k[fo]?(\d+)\}([^{]*)')


def _normalize_ass_color(color: str, fallback: str) -> str:
    color_text = str(color or "").strip()
    if not color_text:
        color_text = fallback
    if not color_text.upper().startswith("&H"):
        color_text = "&H" + color_text.lstrip("&")
    body = color_text[2:]
    if body.endswith("&"):
        body = body[:-1]
    return f"&H{body.upper()}&"


_NO_SPACE_BEFORE_CHARS = frozenset("-',.?!;:/")


def _parse_karaoke_chunks(text: str) -> list[tuple[int, str]]:
    chunks: list[tuple[int, str]] = []
    for match in _K_TAG_RE.finditer(text):
        dur_cs = int(match.group(1))
        raw_chunk = match.group(2)
        if raw_chunk.strip():
            chunks.append((dur_cs, raw_chunk))
    # Strip trailing space from a chunk whose next chunk starts with punctuation
    # that should not be preceded by whitespace (e.g. "multi " + "-colour" → "multi-colour").
    result: list[tuple[int, str]] = []
    for i, (dur_cs, chunk) in enumerate(chunks):
        if i + 1 < len(chunks) and chunk.endswith(" ") and chunks[i + 1][1][:1] in _NO_SPACE_BEFORE_CHARS:
            chunk = chunk[:-1]
        result.append((dur_cs, chunk))
    return result


def _expand_karaoke_fields(
    fields: list[str],
    primary_color: str,
    secondary_color: str,
    *,
    clip_start_cs: int | None = None,
    clip_end_cs: int | None = None,
) -> list[list[str]]:
    if len(fields) < 10 or r"\k" not in fields[9]:
        return [fields]
    if fields[3].strip() != "Default":
        return [fields]

    chunks = _parse_karaoke_chunks(fields[9])
    if not chunks:
        return [fields]

    primary = _normalize_ass_color(primary_color, "&H00FF0000&")
    secondary = _normalize_ass_color(secondary_color, "&H00FFFFFF&")
    line_start_cs = _ass_time_to_cs(fields[1])
    line_end_cs = _ass_time_to_cs(fields[2])
    clip_start_cs = line_start_cs if clip_start_cs is None else max(line_start_cs, clip_start_cs)
    clip_end_cs = line_end_cs if clip_end_cs is None else min(line_end_cs, clip_end_cs)
    if clip_end_cs <= clip_start_cs:
        return []

    secondary_text = "".join(f"{{\\1c{secondary}}}{chunk}" for _, chunk in chunks)
    out: list[list[str]] = []
    cumulative_cs = 0

    for active_idx, (dur_cs, _) in enumerate(chunks):
        word_start_cs = line_start_cs + cumulative_cs
        word_end_cs = word_start_cs + dur_cs
        cumulative_cs += dur_cs
        event_start_cs = max(word_start_cs, clip_start_cs)
        event_end_cs = min(word_end_cs, clip_end_cs)
        if event_end_cs <= event_start_cs:
            continue

        event_text = "".join(
            f"{{\\1c{primary if chunk_idx == active_idx else secondary}}}{chunk}"
            for chunk_idx, (_, chunk) in enumerate(chunks)
        )
        event_fields = list(fields)
        event_fields[1] = _cs_to_ass_time(event_start_cs)
        event_fields[2] = _cs_to_ass_time(event_end_cs)
        event_fields[9] = event_text
        out.append(event_fields)

    tail_start_cs = max(line_start_cs + cumulative_cs, clip_start_cs)
    if clip_end_cs > tail_start_cs:
        tail_fields = list(fields)
        tail_fields[1] = _cs_to_ass_time(tail_start_cs)
        tail_fields[2] = _cs_to_ass_time(clip_end_cs)
        tail_fields[9] = secondary_text
        out.append(tail_fields)

    return out


def _expand_karaoke_to_word_events(
    lines: list[str],
    primary_color: str,
    secondary_color: str,
) -> list[str]:
    """Expand \\k-tagged Default dialogue lines into per-word events.

    Each output event covers exactly one word's timing window (from the
    Whisper/stable-ts \\k timestamps) and shows the full line text with
    only the active word in primary_color; all others in secondary_color.
    Events are non-overlapping and require no character-count estimation.
    """
    out: list[str] = []
    for line in lines:
        fields = ass_dialogue_fields(line)
        if fields is None:
            out.append(line)
            continue
        expanded_fields = _expand_karaoke_fields(fields, primary_color, secondary_color)
        out.extend("Dialogue: " + ",".join(expanded) for expanded in expanded_fields)

    return out


_SUBTITLE_MARGIN_V = 55  # matches normalize_default_subtitle_style hardcoded value


def _word_box_events(
    fields: list[str],
    font_size: int,
    play_res_x: int,
    play_res_y: int,
    pad_x: int | None = None,
    pad_y: int | None = None,
    font_name: str = "Montserrat SemiBold",
    layer: int = 0,
) -> list[str]:
    """Emit one yellow rectangle drawing event per spoken word (karaoke highlight box).

    Handles multi-line subtitles by estimating which words wrap to a second line
    and offsetting those boxes upward by one line-height.
    """
    try:
        line_start = ass_seconds(fields[1].strip())
    except ValueError:
        return []

    # Skip if there is no visible text (e.g. empty line or tag-only content)
    visible_text = re.sub(r'\{[^}]*\}', '', fields[9]).strip()
    if not visible_text:
        return []

    words = _parse_karaoke_words(fields[9])
    if not words:
        return []

    timed: list[tuple[float, float, str]] = []
    cursor = line_start
    for dur_cs, word_text in words:
        w_start = cursor
        cursor += dur_cs / 100.0
        timed.append((w_start, cursor, word_text))

    metrics = _subtitle_box_metrics(font_name, font_size)
    pad_x = metrics["pad_x"] if pad_x is None else pad_x
    pad_y = metrics["pad_y"] if pad_y is None else pad_y

    space_w = _estimate_text_width(" ", font_size, font_name=font_name)
    # Approximate wrap width: canvas minus ~40 px margins each side
    max_line_w = play_res_x - 80
    text_h = metrics["text_h"]
    line_height = metrics["line_height"]
    cx = play_res_x // 2
    text_bottom = play_res_y - _SUBTITLE_MARGIN_V  # alignment=2, bottom-center

    # Split into visual lines matching ASS word-wrap behaviour
    visual_lines: list[list[tuple[float, float, str]]] = []
    cur_line: list[tuple[float, float, str]] = []
    cur_w = 0
    for item in timed:
        ww = _estimate_text_width(item[2], font_size, font_name=font_name)
        extra = ww + (space_w if cur_line else 0)
        if cur_line and cur_w + extra > max_line_w:
            visual_lines.append(cur_line)
            cur_line = [item]
            cur_w = ww
        else:
            cur_line.append(item)
            cur_w += extra
    if cur_line:
        visual_lines.append(cur_line)

    n_lines = len(visual_lines)
    events: list[str] = []
    for line_idx, line_words in enumerate(visual_lines):
        lines_from_bottom = n_lines - 1 - line_idx
        y2 = text_bottom + pad_y - lines_from_bottom * line_height
        y1 = y2 - text_h - 2 * pad_y

        # Each line is centered independently
        line_text = " ".join(w for _, _, w in line_words)
        line_w = _estimate_text_width(line_text, font_size, font_name=font_name)
        x = cx - line_w // 2

        for w_start, w_end, word_text in line_words:
            ww = _estimate_text_width(word_text.strip(), font_size, font_name=font_name)
            if word_text.strip():
                bx1, bx2 = x - pad_x, x + ww + pad_x
                path = f"m {bx1} {y1} l {bx2} {y1} {bx2} {y2} {bx1} {y2}"
                body = r"{\p1\an7\pos(0,0)\1c&H0000FFFF&\bord0\shad0}" + path
                events.append(
                    f"Dialogue: {layer},{ass_time(w_start)},{ass_time(w_end)},Default,,0,0,0,,{body}"
                )
            x += ww + space_w
    return events


def _subtitle_words_in_range(
    ass_path: Path,
    source_start: float,
    source_end: float,
) -> list[tuple[float, float, str]]:
    """Parse word-level timing from karaoke \\k tags in an ASS file.

    Returns (word_start_sec, word_end_sec, lowercased_word) tuples for words
    that fall within [source_start, source_end] in the file's own timeline.
    """
    words: list[tuple[float, float, str]] = []
    try:
        text = ass_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return words
    for line in text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue
        try:
            t_start = ass_seconds(parts[1].strip())
            t_end = ass_seconds(parts[2].strip())
        except Exception:
            continue
        if t_end <= source_start or t_start >= source_end:
            continue
        raw_text = parts[9]
        cursor = t_start
        for m in re.finditer(r'\{\\k(\d+)\}([^{]*)', raw_text):
            k_s = int(m.group(1)) / 100.0
            raw_word = re.sub(r"[^a-zA-Z0-9]", "", m.group(2)).lower()
            w_start = cursor
            cursor += k_s
            if raw_word and len(raw_word) > 1:
                words.append((w_start, cursor, raw_word))
    return words


def _match_bullets_to_subtitle_windows(
    chunks: list[list[tuple[str, str]]],
    ass_path: Path | None,
    source_start: float,
    scene_dur: float,
    delay_sec: float,
) -> list[tuple[float, float]] | None:
    """Return scene-local (start, end) windows by matching bullet keywords to speech.

    Each chunk gets a window starting when the matching word is spoken.
    Returns None if the ASS file is missing, has no matching words, or all
    chunks are unmatched (caller should fall back to even distribution).
    """
    if ass_path is None:
        return None
    word_times = _subtitle_words_in_range(ass_path, source_start, source_start + scene_dur)
    if not word_times:
        return None

    chunk_starts: list[float | None] = []
    for chunk in chunks:
        keywords: set[str] = set()
        for lbl, val in chunk:
            for w in (str(lbl) + " " + str(val)).lower().split():
                w_clean = re.sub(r"[^a-z]", "", w)
                if len(w_clean) >= 3 and w_clean not in _SUBTITLE_STOPWORDS:
                    keywords.add(w_clean)

        match_t: float | None = None
        for w_start, _w_end, word in word_times:
            local_t = w_start - source_start
            if local_t < 0 or local_t >= scene_dur:
                continue
            if word in keywords:
                match_t = local_t
                break
            # Substring match for longer keywords (e.g. "multicolour" ↔ "multicolor")
            for kw in keywords:
                if len(kw) >= 4 and (kw in word or word in kw):
                    match_t = local_t
                    break
            if match_t is not None:
                break
        chunk_starts.append(match_t)

    if all(t is None for t in chunk_starts):
        return None

    windows: list[tuple[float, float]] = []
    for i, start in enumerate(chunk_starts):
        if start is None:
            continue
        win_start = max(start, delay_sec)
        # End = start of next matched chunk, or scene end
        win_end = scene_dur
        for j in range(i + 1, len(chunk_starts)):
            if chunk_starts[j] is not None:
                win_end = chunk_starts[j]
                break
        if win_end > win_start:
            windows.append((win_start, win_end))

    return windows or None


def _truncate_highlight_text(text: str, max_chars: int = 16) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _dedup_window_tags(
    product: dict[str, Any], max_per_window: int = 2
) -> dict[tuple[int, float], list[tuple[str, str]]]:
    """Assign each tag window up to max_per_window display tags, never
    repeating an attribute name within the same video's windows (repeats
    across videos are fine — each script covers its own ground). Windows are
    processed in playback order; a window whose valid attributes were all
    shown by an earlier window of the same video simply gets none.

    Computed from product data alone so every per-scene call sees the same
    assignment.
    """
    windows = sorted(
        product.get("tag_windows", []) or [],
        key=lambda w: (int(w.get("video_index", 0)), float(w.get("start_sec", 0.0))),
    )
    used_names_by_video: dict[int, set[str]] = {}
    assigned: dict[tuple[int, float], list[tuple[str, str]]] = {}
    for w in windows:
        video_index = int(w.get("video_index", 0))
        key = (video_index, float(w.get("start_sec", 0.0)))
        used_names = used_names_by_video.setdefault(video_index, set())
        picked: list[tuple[str, str]] = []
        for tag in w.get("tags", []):
            if len(picked) >= max_per_window:
                break
            lbl = str(tag.get("short_attribute") or tag.get("name", "")).replace("_", " ").strip()
            val = str(tag.get("value", "")).strip()
            if not (lbl or val):
                continue
            filtered = _filter_overlay_gradient_items([(lbl, val)])
            if not filtered:
                continue
            name_key = normalize_for_compare(filtered[0][0])
            if name_key in used_names:
                continue
            used_names.add(name_key)
            picked.append(filtered[0])
        assigned[key] = picked
    return assigned


def _emit_highlight_items(
    items: list[tuple[str, str]],
    evt_start: float,
    evt_end: float,
    label_style: str,
    value_style: str,
    x: int,
    first_label_y: int,
    label_value_gap: int,
    row_gap: int,
    start_x: int,
    slide_ms: int,
    slide_out_ms: int,
    grad_right: int = 0,
    _gradient_windows: list[GradientWindow] | None = None,
) -> list[str]:
    events = []
    capped = items[:2]

    if grad_right > 0 and capped:
        n_rows = len(capped)
        bg_y1 = first_label_y - 15
        bg_y2 = first_label_y + (n_rows - 1) * row_gap + label_value_gap + 30 + 15
        if _gradient_windows is not None:
            # FFmpeg gradient mode: record the window, skip ASS strips
            _gradient_windows.append(GradientWindow(
                start=evt_start, end=evt_end,
                y1=bg_y1, y2=bg_y2,
                grad_right=grad_right, fade_ms=slide_ms,
            ))
        else:
            events.extend(_grad_bg_strips(evt_start, evt_end, label_style, bg_y1, bg_y2, grad_right, slide_ms, slide_out_ms))

    for idx, (lbl, val) in enumerate(capped):
        label_y = first_label_y + idx * row_gap
        value_y = label_y + label_value_gap
        lbl = _truncate_highlight_text(lbl.replace("_", " ").upper()) if lbl else ""
        val = _truncate_highlight_text(val.replace("_", " ").upper()) if val else ""
        fad = f"\\fad({slide_ms},{slide_out_ms})"
        if lbl and val:
            events.append(dialogue(evt_start, evt_end, label_style, positioned_text(x, label_y, lbl, fad), layer=24))
            events.append(dialogue(evt_start, evt_end, value_style, positioned_text(x, value_y, val, fad), layer=24))
        elif lbl or val:
            events.append(dialogue(evt_start, evt_end, value_style, positioned_text(x, label_y, lbl or val, fad), layer=24))
    return events


def bridge_gradient_events(scene: dict[str, Any], product: dict[str, Any], style: dict[str, Any], delay_override: float | None = None, _gradient_windows: list[GradientWindow] | None = None) -> list[str]:
    layout = style.get("layout_presets", {}).get("product_highlight_pip", {})
    text_cfg = layout.get("text", {})
    x = int(text_cfg.get("x", 40))
    first_label_y = int(text_cfg.get("first_label_y", 180))
    label_value_gap = int(text_cfg.get("label_value_gap", 35))
    row_gap = int(text_cfg.get("row_gap", 120))
    slide_ms = int(layout.get("slide_in_duration_ms", 500))
    slide_out_ms = int(layout.get("slide_out_duration_ms", slide_ms))
    start_x = int(layout.get("slide_in_start_x", -300))
    delay_sec = delay_override if delay_override is not None else float(layout.get("highlight_delay_sec", 1.0))
    out_w = int(style.get("output", {}).get("width", 720))
    grad_right = int(out_w * float(layout.get("gradient", {}).get("width_ratio", 0.5)))
    ass_styles = style.get("ass_styles", {})
    label_style = ass_styles.get("highlight_label", {}).get("name", "HighlightLabel")
    value_style = ass_styles.get("highlight_value", {}).get("name", "HighlightValue")
    start = float(scene["start"])
    end = float(scene["end"])
    delayed_start = min(start + delay_sec, end)

    raw_points = product.get("bridge_overlay_points") or []
    str_points = [str(p).strip() for p in raw_points if str(p).strip()]

    items: list[tuple[str, str]] = []
    if str_points:
        for point in str_points:
            label_val = split_bullet_label_value(point)
            items.append(label_val if label_val else ("", point))
        items = _filter_overlay_gradient_items(items)[:2]

    if not items:
        # Fallback 1: tag_windows for this scene's video index
        video_idx = int(scene.get("talking_head_video_index", scene.get("video_index", 0)))
        tag_windows = sorted(
            [w for w in product.get("tag_windows", []) if w.get("video_index") == video_idx],
            key=lambda w: w["start_sec"],
        )
        for w in tag_windows:
            for tag in w.get("tags", []):
                lbl = str(tag.get("short_attribute") or tag.get("name", "")).replace("_", " ").strip()
                val = str(tag.get("value", "")).strip()
                if not (lbl or val):
                    continue
                filtered = _filter_overlay_gradient_items([(lbl, val)])
                if filtered:
                    items.append(filtered[0])
            if len(items) >= 2:
                break
        items = items[:2]

    if not items:
        # Fallback 2: generic selected bullets
        for b in selected_bullets(product, style):
            label_val = split_bullet_label_value(b)
            if label_val:
                items.append(label_val)
        items = _filter_overlay_gradient_items(items)[:2]

    if not items:
        return []

    return _emit_highlight_items(items, delayed_start, end, label_style, value_style, x, first_label_y, label_value_gap, row_gap, start_x, slide_ms, slide_out_ms, grad_right=grad_right, _gradient_windows=_gradient_windows)


def highlight_pip_events(
    scene: dict[str, Any],
    product: dict[str, Any],
    style: dict[str, Any],
    subtitle_asses: list[Path | None] | None = None,
    show_heading: bool = True,
    chunk_size: int = 2,
    delay_override: float | None = None,
    _gradient_windows: list[GradientWindow] | None = None,
    sibling_scenes: list[dict[str, Any]] | None = None,
) -> list[str]:
    layout = style.get("layout_presets", {}).get("product_highlight_pip", {})
    text_cfg = layout.get("text", {})
    x = int(text_cfg.get("x", 40))
    first_label_y = int(text_cfg.get("first_label_y", 180))
    label_value_gap = int(text_cfg.get("label_value_gap", 35))
    row_gap = int(text_cfg.get("row_gap", 120))
    slide_ms = int(layout.get("slide_in_duration_ms", 500))
    slide_out_ms = int(layout.get("slide_out_duration_ms", slide_ms))
    start_x = int(layout.get("slide_in_start_x", -300))
    delay_sec = delay_override if delay_override is not None else float(layout.get("highlight_delay_sec", 1.0))
    heading_sec = float(layout.get("highlight_heading_sec", 3.0))
    out_w = int(style.get("output", {}).get("width", 720))
    grad_right = int(out_w * float(layout.get("gradient", {}).get("width_ratio", 0.5)))

    ass_styles = style.get("ass_styles", {})
    label_style = ass_styles.get("highlight_label", {}).get("name", "HighlightLabel")
    value_style = ass_styles.get("highlight_value", {}).get("name", "HighlightValue")

    scene_start = float(scene["start"])
    scene_end = float(scene["end"])
    scene_dur = scene_end - scene_start
    video_idx = int(scene.get("talking_head_video_index", scene.get("video_index", 0)))
    source_start = scene_source_start(scene)

    fad = f"\\fad({slide_ms},{slide_ms})"
    if show_heading:
        title_style = ass_styles.get("highlight_title", {}).get("name", "HighlightTitle")
        heading_y = int(text_cfg.get("heading_y", 80))
        heading_end = min(scene_start + heading_sec, scene_end)
        title_font_size = int(ass_styles.get("highlight_title", {}).get("font_size", 38))
        hbg_y1 = heading_y - 15
        hbg_y2 = heading_y + title_font_size + 15
        hbg_x2 = x + _estimate_text_width("KEY HIGHLIGHTS", title_font_size) + 55
        events: list[str] = [
            _bg_rect_event(scene_start, heading_end, title_style, 0, hbg_y1, hbg_x2, hbg_y2, opacity_pct=50, layer=22, fade_ms=slide_ms),
            dialogue(scene_start, heading_end, title_style, positioned_text(x, heading_y, "KEY HIGHLIGHTS", fad), layer=25),
        ]
    else:
        events: list[str] = []

    item_dur = float(layout.get("highlight_item_duration_sec", 3.0))
    max_slots = int(layout.get("highlight_max_slots", 2))

    def _bullets_to_items(bullets: list) -> list[tuple[str, str]]:
        result = []
        for b in bullets:
            lv = split_bullet_label_value(b)
            result.append(lv if lv else ("", str(b)))
        return _filter_overlay_gradient_items(result)

    def _place_items(items: list[tuple[str, str]], start_times: list[float]) -> None:
        """Emit each item at its keyword-matched start time with fixed duration.

        Items stack into slots so an arriving item appears below any still-visible
        item.  When a slot frees up it is reused from the top down.
        """
        slot_busy_until = [0.0] * max_slots
        for (lbl, val), t_start in zip(items, start_times):
            if t_start >= scene_dur:
                break
            t_end = t_start + item_dur
            if t_end > scene_dur:
                break  # not enough room for a full display; skip this and all later items
            # Pick the first slot that has already freed up; fall back to the
            # slot that frees soonest (stack below the still-visible item).
            free = next((s for s in range(max_slots) if slot_busy_until[s] <= t_start), None)
            slot = free if free is not None else min(range(max_slots), key=lambda s: slot_busy_until[s])
            slot_busy_until[slot] = t_end
            slot_y = first_label_y + slot * row_gap
            events.extend(_emit_highlight_items(
                [(lbl, val)],
                scene_start + t_start, scene_start + t_end,
                label_style, value_style, x, slot_y, label_value_gap, row_gap,
                start_x, slide_ms, slide_out_ms, grad_right=grad_right,
                _gradient_windows=_gradient_windows,
            ))

    # --- tag_windows path: pre-computed audio-synced windows ---
    _BURN_SCENE_TYPES = {"product_highlight_pip_scene", "product_highlight_pip_scene_with_gradient_overlay"}
    tag_windows = sorted(
        [w for w in product.get("tag_windows", []) if w.get("video_index") == video_idx],
        key=lambda w: w["start_sec"],
    )
    had_tag_windows = bool(tag_windows)
    if tag_windows and sibling_scenes and scene.get("type") in _BURN_SCENE_TYPES:
        # Assign each window to exactly one eligible scene so a window that
        # spans a scene boundary doesn't burn into every scene it overlaps
        # (duplicate, overlapping rows). A window belongs to the last eligible
        # scene starting at or before it; windows that start before all
        # eligible scenes belong to the first one.
        eligible = sorted(
            [
                s for s in sibling_scenes
                if s.get("type") in _BURN_SCENE_TYPES
                and int(s.get("talking_head_video_index", s.get("video_index", 0))) == video_idx
            ],
            key=scene_source_start,
        )
        if eligible:
            def _owner(w: dict[str, Any]) -> dict[str, Any]:
                owner = eligible[0]
                for s in eligible:
                    if scene_source_start(s) <= float(w["start_sec"]):
                        owner = s
                return owner

            tag_windows = [w for w in tag_windows if _owner(w) is scene]
    if had_tag_windows and scene.get("type") in _BURN_SCENE_TYPES:
        if not tag_windows:
            # This video has tag windows, but they all belong to sibling
            # scenes — no highlight overlay here.
            return []
        # Per-window display tags: filtered (type, unapproved brand, ...),
        # capped at 2 and deduped so an attribute name shown in one window
        # never repeats in a later window of either script (a window may end
        # up with no tags at all).
        window_tags = _dedup_window_tags(product)
        tw_items: list[tuple[str, str]] = []
        tw_starts: list[float] = []
        for w in tag_windows:
            raw_t = max(0.0, float(w["start_sec"]) - source_start)
            t = max(raw_t, delay_sec)
            if t >= scene_dur:
                continue
            valid_tags = window_tags.get(
                (int(w.get("video_index", 0)), float(w.get("start_sec", 0.0))), []
            )
            # Pull a late item back so it still gets its full display before
            # the scene ends (slots stack, so overlapping items render in
            # separate rows).
            for j, (lbl, val) in enumerate(valid_tags[:2]):
                item_t = t + j * item_dur
                if item_t + item_dur > scene_dur:
                    item_t = max(t, scene_dur - item_dur)
                if item_t + item_dur > scene_dur:
                    break
                tw_items.append((lbl, val))
                tw_starts.append(item_t)
        if not tw_items:
            # tag_windows present for this scene but all attributes were filtered
            # (type-only, unapproved brand, etc.) — suppress the entire overlay.
            return []
        # Pin the first item to delay_sec when heading is shown, regardless of
        # which window it came from (earlier windows may have had empty tags).
        if show_heading and tw_starts:
            tw_starts[0] = delay_sec
        _place_items(tw_items, tw_starts)
        return events

    # --- subtitle keyword matching: each bullet individually ---
    bullets = selected_bullets(product, style)
    all_items = _bullets_to_items(bullets)
    if not all_items:
        return events

    individual_chunks = [[item] for item in all_items]
    sub_ass = subtitle_asses[video_idx] if subtitle_asses and video_idx < len(subtitle_asses) else None
    sub_windows = _match_bullets_to_subtitle_windows(individual_chunks, sub_ass, source_start, scene_dur, delay_sec)
    if sub_windows:
        starts = [ws for ws, _ in sub_windows]
        if show_heading and starts:
            starts[0] = delay_sec
        _place_items(all_items[:len(sub_windows)], starts)
        return events

    # --- even distribution fallback ---
    available = scene_dur - delay_sec
    if available <= 0:
        return events
    spacing = available / len(all_items)
    starts = [delay_sec + i * spacing for i in range(len(all_items))]
    _place_items(all_items, starts)
    return events


def _bg_rect_event(
    start: float,
    end: float,
    style_name: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    opacity_pct: int = 80,
    layer: int = 29,
    fade_ms: int = 0,
) -> str:
    # ASS alpha: 0x00=fully opaque, 0xFF=fully transparent
    alpha_hex = format(round((1 - opacity_pct / 100) * 255), "02X")
    fade_tag = f"\\fad({fade_ms},{fade_ms})" if fade_ms > 0 else ""
    path = f"m {x1} {y1} l {x2} {y1} {x2} {y2} {x1} {y2}"
    text = "{" + f"{fade_tag}\\p1\\an7\\pos(0,0)\\1c&H000000\\1a&H{alpha_hex}\\bord0\\shad0" + "}" + path
    return dialogue(start, end, style_name, text, layer=layer)


def _grad_bg_strips(
    evt_start: float,
    evt_end: float,
    style_name: str,
    y1: int,
    y2: int,
    grad_right: int,
    fade_in_ms: int,
    fade_out_ms: int,
    n_strips: int = 32,
    layer: int = 22,
) -> list[str]:
    """Left-to-right black gradient background behind key highlight text.

    Fades in/out with the highlight window. Each strip has its own opacity so
    the left edge is ~80% opaque and the right edge is fully transparent.
    Using 32 strips (≈11px each at 360px width) gives a smooth, band-free gradient.
    """
    events = []
    strip_w = grad_right / n_strips
    for i in range(n_strips):
        sx1 = round(i * strip_w)
        sx2 = round((i + 1) * strip_w)
        # ASS alpha: 0x77 (~53% opaque) at left → 0xFF (transparent) at right
        alpha = round(0x77 + (0xFF - 0x77) * i / max(n_strips - 1, 1))
        alpha_hex = format(alpha, "02X")
        path = f"m {sx1} {y1} l {sx2} {y1} {sx2} {y2} {sx1} {y2}"
        tags = "{" + f"\\fad({fade_in_ms},{fade_out_ms})\\p1\\an7\\pos(0,0)\\1c&H000000\\1a&H{alpha_hex}\\bord0\\shad0" + "}"
        events.append(dialogue(evt_start, evt_end, style_name, tags + path, layer=layer))
    return events


def _subtitle_box_metrics(font_name: str, font_size: int) -> dict[str, int]:
    name = font_name.lower()
    if "montserrat" in name and "medium" in name:
        return {
            "pad_x": max(4, int(round(font_size * 0.08))),
            "pad_y": max(5, int(round(font_size * 0.12))),
            "text_h": int(round(font_size * 1.00)),
            "line_height": int(round(font_size * 1.26)),
        }
    if "montserrat" in name:
        return {
            "pad_x": max(4, int(round(font_size * 0.09))),
            "pad_y": max(4, int(round(font_size * 0.10))),
            "text_h": int(round(font_size * 0.96)),
            "line_height": int(round(font_size * 1.22)),
        }
    return {
        "pad_x": max(4, int(round(font_size * 0.10))),
        "pad_y": max(3, int(round(font_size * 0.08))),
        "text_h": int(round(font_size * 0.82)),
        "line_height": int(round(font_size * 1.18)),
    }


def _estimate_text_width(text: str, font_size: int, *, font_name: str = "Montserrat SemiBold") -> int:
    """Rough estimate of rendered text width in ASS script-space units.

    Uses font-aware width factors so highlight boxes track the active subtitle face.
    """
    name = font_name.lower()
    if "montserrat" in name and "medium" in name:
        space = 0.22
        narrow = 0.25
        wide = 0.60
        digit = 0.46
        upper = 0.43
        lower = 0.40
        other = 0.42
    elif "montserrat" in name:
        space = 0.23
        narrow = 0.27
        wide = 0.63
        digit = 0.48
        upper = 0.46
        lower = 0.42
        other = 0.44
    else:
        space = 0.24
        narrow = 0.28
        wide = 0.66
        digit = 0.50
        upper = 0.49
        lower = 0.45
        other = 0.47

    SPACE = set(" ")
    NARROW = set("iIl1!|;:.,'\"[]()")
    WIDE = set("mwWMQDGO0%@#&")
    total = 0.0
    for ch in text:
        if ch in SPACE:
            total += space
        elif ch in NARROW:
            total += narrow
        elif ch in WIDE:
            total += wide
        elif ch.isdigit():
            total += digit
        elif ch.isupper():
            total += upper
        elif ch.islower():
            total += lower
        else:
            total += other
    return int(total * font_size)


def product_name_intro_events(
    style: dict[str, Any],
    timeline: dict[str, Any],
    duration: float = 3.0,
) -> list[str]:
    products = list(product_by_id(timeline).values())
    if not products:
        return []
    name = shorten_product_name(str(products[0].get("name", "")).strip())
    if not name:
        return []
    name_cfg = style.get("ass_styles", {}).get("product_name_intro", {})
    style_name = name_cfg.get("name", "ProductNameIntro")
    output = style.get("output", {})
    out_w = int(output.get("width", 720))
    cx = out_w // 2
    top_y = int(name_cfg.get("top_y", 70))
    font_size = int(name_cfg.get("font_size", 44))
    pad_x = int(name_cfg.get("bg_pad_x", 24))
    pad_y = int(name_cfg.get("bg_pad_y", 15))
    name = name.upper()
    fade_ms = int(name_cfg.get("fade_ms", 300))
    text_w = _estimate_text_width(name, font_size)
    bar_x1 = max(0, cx - text_w // 2 - pad_x)
    bar_x2 = min(out_w, cx + text_w // 2 + pad_x)
    bar_y1 = max(0, top_y - pad_y)
    bar_y2 = top_y + font_size + pad_y
    fade_tag = f"{{\\fad({fade_ms},{fade_ms})}}"
    return [
        _bg_rect_event(0.0, duration, style_name, bar_x1, bar_y1, bar_x2, bar_y2, opacity_pct=int(name_cfg.get("bg_opacity", 40)), layer=29, fade_ms=fade_ms),
        dialogue(0.0, duration, style_name, fade_tag + positioned_text(cx, top_y, name), layer=30),
    ]


def generated_events(
    style: dict[str, Any],
    timeline: dict[str, Any],
    timeline_path: Path | None = None,
    _gradient_windows: list[GradientWindow] | None = None,
) -> list[str]:
    products = product_by_id(timeline)
    scenes = normalize_scene_times(style, timeline, timeline_path)
    events = ["Comment: 0,0:00:00.00,0:00:00.00,ProductTitle,,0,0,0,,Generated product overlay events"]
    seen_product_scene_keys: set[tuple[str, str, float, float]] = set()
    heading_shown = False

    # Resolve subtitle ASS paths (indexed by video_index) for keyword-based timing
    timeline_dir = Path(timeline_path).parent if timeline_path else Path(".")
    raw_ass = timeline.get("base_subtitles_ass_files") or []
    if isinstance(raw_ass, str):
        raw_ass = [raw_ass]
    subtitle_asses: list[Path | None] = []
    for p in raw_ass:
        resolved = (timeline_dir / p).resolve()
        subtitle_asses.append(resolved if resolved.exists() else None)

    for scene in scenes:
        scene_type = scene.get("type")
        pid = scene.get("product_id")
        product = products.get(str(pid)) if pid is not None else None
        if scene_type in {"product_highlight_pip_scene", "product_highlight_pip_scene_with_gradient_overlay", "product_pip_head_exit_scene"} and product:
            key = (scene_type, str(pid), float(scene["start"]), float(scene["end"]))
            if key not in seen_product_scene_keys:
                events.extend(highlight_pip_events(scene, product, style, subtitle_asses=subtitle_asses, show_heading=not heading_shown, _gradient_windows=_gradient_windows, sibling_scenes=scenes))
                heading_shown = True
                seen_product_scene_keys.add(key)
        elif scene_type == "product_bridge_gradient_overlay" and product:
            key = (scene_type, str(pid), float(scene["start"]), float(scene["end"]))
            if key not in seen_product_scene_keys:
                events.extend(bridge_gradient_events(scene, product, style, delay_override=0.0, _gradient_windows=_gradient_windows))
                seen_product_scene_keys.add(key)
        elif scene_type in AUDIO_OVERLAY_SCENE_TYPES and product:
            key = (scene_type, str(pid), float(scene["start"]), float(scene["end"]))
            if key not in seen_product_scene_keys:
                events.extend(highlight_pip_events(
                    scene, product, style,
                    subtitle_asses=subtitle_asses,
                    show_heading=False,
                    chunk_size=1,
                    delay_override=0.0,
                    _gradient_windows=_gradient_windows,
                    sibling_scenes=scenes,
                ))
                seen_product_scene_keys.add(key)
        elif scene_type in PRODUCT_SCENE_TYPES and product:
            key = (scene_type, str(pid), float(scene["start"]), float(scene["end"]))
            if key not in seen_product_scene_keys:
                events.extend(product_scene_events(scene, product, style))
                seen_product_scene_keys.add(key)
        elif scene_type == "flipkart_end_scene" and product:
            if scene.get("show_product_info") is not False:
                events.extend(product_scene_events(scene, product, style))
            events.extend(flipkart_events(scene, product, style))
        elif scene_type == "product_overlay_float_scene" and product:
            cta = product.get("cta")
            if cta:
                events.append(
                    dialogue(
                        float(scene["start"]),
                        float(scene["end"]),
                        style.get("ass_styles", {}).get("product_cta", {}).get("name", "ProductCTA"),
                        positioned_text(360, 925, cta),
                        layer=25,
                    )
                )
    return events


_STRINGIFIED_DICT_PATTERNS = (
    r"{'LABEL'",
    r"'VALUE':",
    r"label':",
    '"label":',
    r"{'label'",
    r'"label"',
)


def _assert_no_stringified_dicts(ass_text: str, out_path: Path) -> None:
    """Raise if any Dialogue line contains a stringified Python/JSON dict artifact."""
    for line in ass_text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        # Only check the text payload after the last leading comma (the override block onward)
        parts = line.split(",,", 1)
        if len(parts) < 2:
            continue
        payload = parts[1]
        for pat in _STRINGIFIED_DICT_PATTERNS:
            if pat.upper() in payload.upper():
                raise ValueError(
                    f"Stringified dict detected in ASS output ({out_path}): {line!r}\n"
                    f"Matched pattern: {pat!r}"
                )


def collect_gradient_windows(
    style_path: str | Path,
    timeline_path: str | Path,
) -> list[GradientWindow]:
    """Return gradient panel specs for every highlight window in the timeline.

    Mirrors the scene-dispatch logic in generated_events() but passes a shared
    _gradient_windows list so _emit_highlight_items records geometry instead of
    emitting ASS strip events.  Call this BEFORE merge_ass() so the same timing
    logic is used for both the FFmpeg overlay and the ASS text events.
    """
    style = load_json(style_path)
    timeline = load_json(timeline_path)
    timeline_path_obj = Path(timeline_path)

    products = product_by_id(timeline)
    scenes = normalize_scene_times(style, timeline, timeline_path_obj)
    seen_product_scene_keys: set[tuple[str, str, float, float]] = set()
    heading_shown = False

    timeline_dir = timeline_path_obj.parent
    raw_ass = timeline.get("base_subtitles_ass_files") or []
    if isinstance(raw_ass, str):
        raw_ass = [raw_ass]
    subtitle_asses: list[Path | None] = []
    for p in raw_ass:
        resolved = (timeline_dir / p).resolve()
        subtitle_asses.append(resolved if resolved.exists() else None)

    windows: list[GradientWindow] = []

    for scene in scenes:
        scene_type = scene.get("type")
        pid = scene.get("product_id")
        product = products.get(str(pid)) if pid is not None else None

        if scene_type in {"product_highlight_pip_scene", "product_highlight_pip_scene_with_gradient_overlay", "product_pip_head_exit_scene"} and product:
            key = (scene_type, str(pid), float(scene["start"]), float(scene["end"]))
            if key not in seen_product_scene_keys:
                highlight_pip_events(scene, product, style, subtitle_asses=subtitle_asses, show_heading=not heading_shown, _gradient_windows=windows, sibling_scenes=scenes)
                heading_shown = True
                seen_product_scene_keys.add(key)
        elif scene_type == "product_bridge_gradient_overlay" and product:
            key = (scene_type, str(pid), float(scene["start"]), float(scene["end"]))
            if key not in seen_product_scene_keys:
                bridge_gradient_events(scene, product, style, delay_override=0.0, _gradient_windows=windows)
                seen_product_scene_keys.add(key)
        elif scene_type in AUDIO_OVERLAY_SCENE_TYPES and product:
            key = (scene_type, str(pid), float(scene["start"]), float(scene["end"]))
            if key not in seen_product_scene_keys:
                highlight_pip_events(scene, product, style, subtitle_asses=subtitle_asses, show_heading=False, chunk_size=1, delay_override=0.0, _gradient_windows=windows, sibling_scenes=scenes)
                seen_product_scene_keys.add(key)

    return windows


def merge_ass(style_path: str | Path, timeline_path: str | Path, out_path: str | Path) -> Path:
    style = load_json(style_path)
    timeline = load_json(timeline_path)
    timeline_path_obj = Path(timeline_path)
    subtitle_paths = subtitle_ass_paths(timeline, timeline_path_obj)
    if not subtitle_paths:
        raise FileNotFoundError("No stable-ts ASS configured. Set base_subtitles_ass or base_subtitles_ass_files.")
    base_path = subtitle_paths[0]
    if not base_path.exists():
        raise FileNotFoundError(f"Base stable-ts ASS not found: {base_path}")
    for i, sub_path in enumerate(subtitle_paths):
        if not sub_path.exists():
            raise FileNotFoundError(f"Stable-ts ASS {i} not found: {sub_path}")

    lines = remove_event_rows(base_path.read_text(encoding="utf-8").splitlines())
    lines = ensure_script_info_resolution(lines, style)
    font, font_size, primary_color, secondary_color, outline, shadow, back_color = _subtitle_style_values(style)
    lines = enforce_subtitle_style_overrides(lines, style)
    lines = _expand_karaoke_to_word_events(lines, primary_color, secondary_color)
    lines = ensure_styles(lines, style)
    lines = ensure_events_section(lines)
    sub_events = shifted_subtitle_events(style, timeline, timeline_path_obj)
    if sub_events:
        lines.extend(["", *sub_events])
        lines = move_existing_subtitles_above_safe_area(lines, style, timeline, timeline_path_obj)
    lines.extend(["", *generated_events(style, timeline, timeline_path_obj, _gradient_windows=[])])
    # Final guard: ensure style overrides still hold in the final ASS output.
    lines = enforce_subtitle_style_overrides(lines, style)

    out = Path(out_path)
    final_text = "\n".join(lines) + "\n"
    _assert_no_stringified_dicts(final_text, out)
    out.write_text(final_text, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge stable-ts ASS with product overlay ASS events.")
    parser.add_argument("--style", default="global_style.json")
    parser.add_argument("--timeline", default="example_timeline.json")
    parser.add_argument("--out", default="final_captions.ass")
    args = parser.parse_args()

    out = merge_ass(args.style, args.timeline, args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
