#!/usr/bin/env python3
"""
Generate timeline JSON files for paired batch videos.

The script scans a folder for files named like:

  ITM..._script1_Product.mp4
  ITM..._script2_Product.mp4

It pairs videos by the leading ITM item id, points each timeline at the ASS files
created by generate_ass.sh, loads product information from home_306210_items.json
by item_id, and creates scene timestamps from the real source durations so
subtitles shift correctly during render.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

FLIPKART_LOGO = Path(__file__).resolve().parent.parent / "assets" / "flipkart_logo.png"
from typing import Any


VIDEO_RE = re.compile(r"^(ITM[A-Za-z0-9]+).*_script([12])(?:_([^./]+))?\.(?:mp4|mov)$", re.IGNORECASE)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_NUMBER_RE = re.compile(r"_(\d+)(?:\.[^.]+)$")
DEFAULT_TIMELINE_CONFIG_PATH = Path("timeline_generation_config.json")
DEFAULT_TIMELINE_CONFIG: dict[str, Any] = {
    "defaults": {
        "bridge_duration": 3.0,
        "end_card_duration": 1.0,
        "expected_images": 5,
    },
    "per_video": [
        {
            "type": "talking_head_scene",
            "usage": "vid{video_number} talking head from 0s to 4s.",
            "source_start": "source_cursor",
            "source_duration": 4.0,
        },
        {
            "type": "talking_head_product_strip_scene",
            "usage": "vid{video_number} talking head with black product info strip and bottom-right product image from 4s to 8s.",
            "source_start": "source_cursor",
            "source_duration": 4.0,
            "absorb_remaining_under": 1.0,
            "product_id": "$item_id",
            "image_index": {"by_video": [1, 2], "fallback": 0},
            "show_cta": False,
            "show_meta": False,
        },
        {
            "type": "product_highlight_pip_scene_with_gradient_overlay",
            "usage": "vid{video_number} product image fullscreen with baked-in gradient highlights and PiP, 0s to 6s.",
            "source_start": "source_cursor",
            "source_duration": 6.0,
            "product_id": "$item_id",
            "image_index": 0,
            "show_cta": False,
            "show_meta": False,
            "show_product_info": False,
        },
        {
            "type": "talking_head_with_gradient_overlay",
            "usage": "vid{video_number} talking head for remaining source duration with baked-in gradient overlay.",
            "source_start": "source_cursor",
            "source_end": "video_end",
            "product_id": "$item_id",
            "absorb_remaining_under": 0.0,
        },
    ],
    "between_videos": [
        {
            "type": "product_bridge_gradient_overlay",
            "usage": "Gradient bridge before vid{next_video_number} with product info overlay.",
            "duration": "$bridge_duration",
            "transition": "dip_to_black",
            "product_id": "$item_id",
            "image_index": 0,
            "show_meta": False,
        }
    ],
    "after_all_videos": [
        {
            "type": "flipkart_end_scene",
            "usage": "Flipkart end scene.",
            "duration": "$end_card_duration",
            "transition": "dip_to_white",
            "product_id": "$item_id",
            "image_index": 0,
            "show_cta": False,
            "show_meta": False,
            "show_product_info": False,
            "show_subtitles": False,
        }
    ],
}


@dataclass(frozen=True)
class BatchVideo:
    item_id: str
    script_index: int
    product_slug: str
    path: Path


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def timeline_relative(path: Path, timeline_dir: Path) -> str:
    base = timeline_dir.resolve()
    target = path if path.is_absolute() else (Path.cwd() / path)
    return os.path.relpath(target.resolve(), base).replace(os.sep, "/")


def load_catalog(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Catalog must be a JSON array: {path}")
    return {str(row.get("item_id")): row for row in data if row.get("item_id")}


def load_timeline_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return DEFAULT_TIMELINE_CONFIG
    if not path.exists():
        if path == DEFAULT_TIMELINE_CONFIG_PATH:
            return DEFAULT_TIMELINE_CONFIG
        raise FileNotFoundError(f"Timeline generation config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Timeline generation config must be a JSON object: {path}")
    return data


def media_duration(path: Path) -> float:
    if shutil.which("ffprobe") is not None:
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

    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("Neither ffprobe nor ffmpeg was found on PATH")
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


def discover_videos(batch_dir: Path) -> dict[str, dict[int, BatchVideo]]:
    pairs: dict[str, dict[int, BatchVideo]] = {}
    for path in sorted(batch_dir.iterdir()):
        if not path.is_file():
            continue
        match = VIDEO_RE.match(path.name)
        if match is None:
            continue
        item_id = match.group(1)
        script_index = int(match.group(2))
        product_slug = match.group(3) or ""
        pairs.setdefault(item_id, {})[script_index] = BatchVideo(item_id, script_index, product_slug, path)
    return pairs


def discover_images(
    image_dir: Path,
    item_id: str,
    expected_count: int,
    placeholder: Path | None,
    timeline_dir: Path,
) -> list[str]:
    candidates: list[Path] = []
    if image_dir.exists():
        item_subdir = image_dir / item_id
        nested_product_dir = image_dir / "product_images"
        search_roots = [item_subdir] if item_subdir.exists() else [image_dir]
        if nested_product_dir.exists():
            search_roots.append(nested_product_dir)
        for root in search_roots:
            candidates.extend(path for path in root.rglob("*") if is_item_image(path, item_id))
            if root == item_subdir:
                candidates.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    deduped = sorted(dict.fromkeys(candidates), key=image_sort_key)
    images = [timeline_relative(path, timeline_dir) for path in deduped[:expected_count]]
    if len(images) < expected_count and placeholder is not None:
        images.extend([timeline_relative(placeholder, timeline_dir)] * (expected_count - len(images)))
    return images


def is_item_image(path: Path, item_id: str) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS and path.name.startswith(item_id)


def image_sort_key(path: Path) -> tuple[str, int, str]:
    match = IMAGE_NUMBER_RE.search(path.name)
    number = int(match.group(1)) if match else 0
    return (path.name.split("_", 1)[0], number, path.name)


def attributes_from_record(record: dict[str, Any]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    raw_attrs = record.get("attributes")
    if isinstance(raw_attrs, str):
        try:
            parsed = json.loads(raw_attrs)
            if isinstance(parsed, dict):
                attrs = parsed
        except json.JSONDecodeError:
            attrs = {}
    if record.get("brand") and not attrs.get("brand"):
        attrs["brand"] = record["brand"]
    return attrs


def strip_leading_brand(title: str, brand: str | None) -> str:
    if not brand:
        return title
    title = title.strip()
    brand = brand.strip()
    if not title.lower().startswith(brand.lower()):
        return title
    stripped = title[len(brand) :].lstrip(" -:|,")
    return stripped or title


def bullets_from_record(record: dict[str, Any]) -> list[Any]:
    attrs = attributes_from_record(record)
    bullets: list[Any] = []
    for key in ["brand", "material", "color", "type", "pack_of", "capacity", "thread_count", "pattern"]:
        value = attrs.get(key)
        if value:
            label = key.replace("_", " ").title()
            bullets.append({"label": label, "value": str(value)})
        if len(bullets) >= 4:
            break

    if not bullets and record.get("semantic_description"):
        sentences = re.split(r"(?<=[.!?])\s+", str(record["semantic_description"]).strip())
        bullets = [sentence.rstrip(".") for sentence in sentences[:4] if sentence]
    return bullets[:4]


def strip_ass_tags(text: str) -> str:
    text = re.sub(r"\{[^}]*\}", "", text)
    text = text.replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
    return re.sub(r"\s+", " ", text).strip()


def combined_script_from_ass_paths(paths: list[Path]) -> str:
    parts: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("Dialogue:"):
                continue
            fields = line.split(",", 9)
            if len(fields) != 10:
                continue
            visible = strip_ass_tags(fields[9])
            if visible:
                parts.append(visible)
    return " ".join(parts)


def heuristic_bridge_points(script_text: str, limit: int = 5) -> list[str]:
    candidates = [piece.strip(" .") for piece in re.split(r"(?<=[.!?])\s+", script_text) if piece.strip()]
    points: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        points.append(candidate)
        if len(points) >= limit:
            break
    return points


def bridge_points_from_script(script_text: str, limit: int = 4) -> list[str]:
    script_text = re.sub(r"\s+", " ", script_text).strip()
    if not script_text:
        return []

    api_url = os.environ.get("BRIDGE_POINTS_API_URL", "").strip()
    api_key = os.environ.get("BRIDGE_POINTS_API_KEY", "").strip()
    if api_url:
        payload = json.dumps({"combined_script": script_text}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(api_url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            raw_points = (
                data.get("bridge_overlay_points")
                or data.get("points")
                or data.get("items")
                or []
            )
            points = [str(point).strip() for point in raw_points if str(point).strip()]
            if points:
                return points[:limit]
        except (OSError, urllib.error.URLError, json.JSONDecodeError, AttributeError):
            pass

    return heuristic_bridge_points(script_text, limit=limit)


def product_payload(
    item_id: str,
    product_slug: str,
    images: list[str],
    catalog: dict[str, dict[str, Any]],
    timeline_dir: Path,
    bridge_overlay_points: list[str] | None = None,
) -> dict[str, Any]:
    record = catalog.get(item_id, {})
    attrs = attributes_from_record(record)
    brand = str(attrs.get("brand", "")).strip()
    title = strip_leading_brand(str(record.get("title") or product_slug), brand)
    return {
        "id": item_id,
        "images": images,
        "name": title,
        "brand": brand,
        "raw_features": bullets_from_record(record),
        "selected_bullets": bullets_from_record(record),
        "bridge_overlay_points": (bridge_overlay_points or [])[:4],
        "flipkart_url": f"https://www.flipkart.com/search?q={item_id}",
        "flipkart_logo": timeline_relative(FLIPKART_LOGO, timeline_dir),
    }


def config_number(value: Any, context: dict[str, Any]) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.startswith("$"):
        key = value[1:]
        if key not in context:
            raise ValueError(f"Unknown timeline config reference: {value}")
        return float(context[key])
    return float(value)


SCRIPT_FALLBACK_CHAIN: dict[int, list[int]] = {
    0: [1, 0],
    1: [2, 1, 0],
}


def resolve_image_index(value: Any, image_count: int, video_index: int) -> int:
    if isinstance(value, dict):
        by_video = value.get("by_video")
        if isinstance(by_video, list) and video_index < len(by_video):
            value = by_video[video_index]
        else:
            value = value.get("fallback", 0)
    candidates = [int(value)] + SCRIPT_FALLBACK_CHAIN.get(video_index, [0])
    for idx in candidates:
        if 0 <= idx < image_count:
            return idx
    return 0


def clamp_image_index(value: Any, image_count: int, video_index: int) -> int:
    return resolve_image_index(value, image_count, video_index)


def resolve_config_value(value: Any, context: dict[str, Any], image_count: int, video_index: int) -> Any:
    if isinstance(value, str):
        if value.startswith("$"):
            key = value[1:]
            if key not in context:
                raise ValueError(f"Unknown timeline config reference: {value}")
            return context[key]
        return value.format(**context) if "{" in value and "}" in value else value
    if isinstance(value, list):
        return [resolve_config_value(item, context, image_count, video_index) for item in value]
    if isinstance(value, dict):
        if "by_video" in value or "fallback" in value:
            return clamp_image_index(value, image_count, video_index)
        return {key: resolve_config_value(item, context, image_count, video_index) for key, item in value.items()}
    return value


def scene_from_template(
    template: dict[str, Any],
    context: dict[str, Any],
    image_count: int,
    video_index: int,
) -> dict[str, Any]:
    generation_keys = {
        "duration",
        "source_start",
        "source_duration",
        "source_end",
        "absorb_remaining_under",
    }
    scene = {
        key: resolve_config_value(value, context, image_count, video_index)
        for key, value in template.items()
        if key not in generation_keys
    }
    if "image_index" in scene:
        scene["image_index"] = resolve_image_index(scene["image_index"], image_count, video_index)
    return scene


def append_video_scenes(
    scenes: list[dict[str, Any]],
    *,
    templates: list[dict[str, Any]],
    video_index: int,
    video_duration: float,
    cursor: float,
    item_id: str,
    image_count: int,
    bridge_duration: float,
    end_card_duration: float,
) -> float:
    source_cursor = 0.0
    context = {
        "item_id": item_id,
        "video_index": video_index,
        "video_number": video_index + 1,
        "video_label": f"vid{video_index + 1}",
        "video_duration": video_duration,
        "video_half_duration": video_duration / 2.0,
        "bridge_duration": bridge_duration,
        "end_card_duration": end_card_duration,
    }
    for template in templates:
        source_start_value = template.get("source_start", "source_cursor")
        if source_start_value in {"source_cursor", "after_previous_source"}:
            source_start = source_cursor
        elif source_start_value == "video_end":
            source_start = video_duration
        else:
            source_start = config_number(source_start_value, context)
        if template.get("source_end") == "video_end":
            source_end = video_duration
        elif "source_end" in template:
            source_end = config_number(template["source_end"], context)
        elif "source_duration" in template:
            source_end = source_start + config_number(template["source_duration"], context)
        else:
            raise ValueError(f"Video scene template requires source_duration or source_end: {template}")

        source_end = min(source_end, video_duration)
        absorb_remaining_under = template.get("absorb_remaining_under")
        if absorb_remaining_under is not None:
            threshold = config_number(absorb_remaining_under, context)
            remaining = video_duration - source_end
            if 0.0 < remaining < threshold:
                source_end = video_duration

        if source_end <= source_start:
            source_cursor = max(source_cursor, source_end)
            continue

        scene = scene_from_template(template, context, image_count, video_index)
        scene.update(
            {
                "start": round(cursor, 3),
                "end": round(cursor + source_end - source_start, 3),
                "talking_head_video_index": video_index,
                "source_start": round(source_start, 3),
            }
        )
        scenes.append(scene)
        cursor = float(scene["end"])
        source_cursor = source_end
    return cursor


def append_duration_scenes(
    scenes: list[dict[str, Any]],
    *,
    templates: list[dict[str, Any]],
    cursor: float,
    item_id: str,
    image_count: int,
    video_index: int,
    bridge_duration: float,
    end_card_duration: float,
    extra_context: dict[str, Any] | None = None,
) -> float:
    context = {
        "item_id": item_id,
        "video_index": video_index,
        "video_number": video_index + 1,
        "bridge_duration": bridge_duration,
        "end_card_duration": end_card_duration,
        **(extra_context or {}),
    }
    for template in templates:
        duration = config_number(template.get("duration", 0), context)
        if duration <= 0:
            continue
        scene = scene_from_template(template, context, image_count, video_index)
        scene["start"] = round(cursor, 3)
        scene["end"] = round(cursor + duration, 3)
        scenes.append(scene)
        cursor = float(scene["end"])
    return cursor


def build_scenes_from_config(
    config: dict[str, Any],
    *,
    durations: list[float],
    item_id: str,
    image_count: int,
    bridge_duration: float,
    end_card_duration: float,
) -> list[dict[str, Any]]:
    per_video_templates = config.get("per_video")
    if not isinstance(per_video_templates, list) or not per_video_templates:
        raise ValueError("Timeline generation config requires a non-empty per_video list")
    between_templates = config.get("between_videos", [])
    after_templates = config.get("after_all_videos", [])
    if not isinstance(between_templates, list):
        raise ValueError("Timeline generation config between_videos must be a list")
    if not isinstance(after_templates, list):
        raise ValueError("Timeline generation config after_all_videos must be a list")

    scenes: list[dict[str, Any]] = []
    cursor = 0.0
    for video_index, duration in enumerate(durations):
        cursor = append_video_scenes(
            scenes,
            templates=per_video_templates,
            video_index=video_index,
            video_duration=duration,
            cursor=cursor,
            item_id=item_id,
            image_count=image_count,
            bridge_duration=bridge_duration,
            end_card_duration=end_card_duration,
        )
        if video_index + 1 < len(durations):
            cursor = append_duration_scenes(
                scenes,
                templates=between_templates,
                cursor=cursor,
                item_id=item_id,
                image_count=image_count,
                video_index=video_index,
                bridge_duration=bridge_duration,
                end_card_duration=end_card_duration,
                extra_context={
                    "previous_video_index": video_index,
                    "previous_video_number": video_index + 1,
                    "next_video_index": video_index + 1,
                    "next_video_number": video_index + 2,
                },
            )
    append_duration_scenes(
        scenes,
        templates=after_templates,
        cursor=cursor,
        item_id=item_id,
        image_count=image_count,
        video_index=max(len(durations) - 1, 0),
        bridge_duration=bridge_duration,
        end_card_duration=end_card_duration,
    )
    return scenes


def build_timeline(
    pair: dict[int, BatchVideo],
    *,
    catalog: dict[str, dict[str, Any]],
    image_dir: Path,
    output_dir: Path,
    ass_dir: Path,
    rendered_dir: Path,
    bridge_duration: float,
    end_card_duration: float,
    expected_images: int,
    placeholder: Path | None,
    timeline_config: dict[str, Any],
) -> dict[str, Any]:
    first = pair[1]
    second = pair[2]
    item_id = first.item_id
    images = discover_images(image_dir, item_id, expected_images, placeholder, output_dir)
    if not images:
        raise ValueError(f"No product images found for {item_id} in {image_dir}")

    durations = [media_duration(first.path), media_duration(second.path)]
    videos = [timeline_relative(first.path, output_dir), timeline_relative(second.path, output_dir)]
    ass_files = [
        timeline_relative(ass_dir / f"{first.path.stem}.ass", output_dir),
        timeline_relative(ass_dir / f"{second.path.stem}.ass", output_dir),
    ]
    ass_paths = [ass_dir / f"{first.path.stem}.ass", ass_dir / f"{second.path.stem}.ass"]
    bridge_overlay_points = bridge_points_from_script(combined_script_from_ass_paths(ass_paths), limit=4)
    product = product_payload(
        item_id,
        first.product_slug,
        images,
        catalog,
        output_dir,
        bridge_overlay_points,
    )

    scenes = build_scenes_from_config(
        timeline_config,
        durations=durations,
        item_id=item_id,
        image_count=len(images),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paired batch timeline JSON files.")
    parser.add_argument("--batch-dir", default="batch1", type=Path)
    parser.add_argument("--out-dir", default="batch_timelines", type=Path)
    parser.add_argument("--ass-dir", default="generated_ass", type=Path)
    parser.add_argument("--rendered-dir", default="batch_outputs", type=Path)
    parser.add_argument(
        "--image-dir",
        default="product_images/product_images",
        type=Path,
        help="Product image folder. Images are matched by filenames starting with item_id.",
    )
    parser.add_argument(
        "--catalog",
        default="home_306210_items.json",
        type=Path,
        help="Product information JSON. Rows are matched by item_id.",
    )
    parser.add_argument("--bridge-duration", default=None, type=float)
    parser.add_argument("--end-card-duration", default=None, type=float)
    parser.add_argument("--expected-images", default=None, type=int)
    parser.add_argument(
        "--timeline-config",
        default=DEFAULT_TIMELINE_CONFIG_PATH,
        type=Path,
        help="JSON scene recipe for timeline generation. Defaults to timeline_generation_config.json.",
    )
    parser.add_argument(
        "--placeholder-image",
        default=None,
        type=Path,
        help="Use this existing image to fill missing product image slots while final pictures are pending.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = discover_videos(args.batch_dir)
    complete_pairs = {item_id: pair for item_id, pair in pairs.items() if 1 in pair and 2 in pair}
    incomplete = sorted(item_id for item_id, pair in pairs.items() if 1 not in pair or 2 not in pair)
    if not complete_pairs:
        raise SystemExit(f"No complete script1/script2 pairs found in {args.batch_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.rendered_dir.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(args.catalog)
    timeline_config = load_timeline_config(args.timeline_config)
    timeline_defaults = timeline_config.get("defaults", {})
    bridge_duration = args.bridge_duration
    if bridge_duration is None:
        bridge_duration = float(timeline_defaults.get("bridge_duration", 5.0))
    end_card_duration = args.end_card_duration
    if end_card_duration is None:
        end_card_duration = float(timeline_defaults.get("end_card_duration", 1.0))
    expected_images = args.expected_images
    if expected_images is None:
        expected_images = int(timeline_defaults.get("expected_images", 4))
    placeholder = args.placeholder_image if args.placeholder_image and args.placeholder_image.exists() else None

    manifest: list[dict[str, Any]] = []
    skipped: list[str] = []
    for item_id, pair in sorted(complete_pairs.items()):
        try:
            timeline = build_timeline(
                pair,
                catalog=catalog,
                image_dir=args.image_dir,
                output_dir=args.out_dir,
                ass_dir=args.ass_dir,
                rendered_dir=args.rendered_dir,
                bridge_duration=bridge_duration,
                end_card_duration=end_card_duration,
                expected_images=expected_images,
                placeholder=placeholder,
                timeline_config=timeline_config,
            )
        except ValueError as exc:
            skipped.append(f"{item_id} ({exc})")
            continue
        path = args.out_dir / f"{item_id}.json"
        path.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")
        manifest.append(
            {
                "item_id": item_id,
                "timeline": repo_relative(path),
                "output": timeline["output"],
                "videos": timeline["talking_head_videos"],
                "ass_files": timeline["base_subtitles_ass_files"],
                "images": timeline["products"][0]["images"],
            }
        )

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} timelines to {args.out_dir}")
    print(f"Wrote manifest to {manifest_path}")
    print(f"Loaded product information for {len(catalog)} items from {args.catalog}")
    print(f"Matched product images from {args.image_dir}")
    print(f"Used timeline generation config: {args.timeline_config}")
    if incomplete:
        print(f"Skipped incomplete item ids: {', '.join(incomplete)}")
    if skipped:
        print("Skipped item ids:")
        for item in skipped:
            print(f"  - {item}")
    if placeholder is not None:
        print(f"Missing image slots were filled with placeholder: {placeholder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
