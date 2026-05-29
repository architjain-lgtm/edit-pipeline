#!/usr/bin/env python3
"""Create clean ASS subtitles from source scripts, point a timeline at them, and render."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from verify_ass_against_script import (
    AssEvent,
    Issue,
    VerificationError,
    compare_protected_tokens,
    compare_token_sequences,
    extract_protected_tokens,
    extract_reference_text,
    find_record_by_itm_id,
    load_json_files,
    map_diff_to_ass_events,
    normalize_tokens,
    parse_ass_events,
    tokenize_ass_events,
    tokenize_product_aware,
)

CLEAN_ASS_SUBTITLE_FONT_SIZE = 40
CLEAN_ASS_SUBTITLE_FONT = "Montserrat SemiBold"


def resolve(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def timeline_relative(path: Path, timeline_dir: Path) -> str:
    return os.path.relpath(path.resolve(), timeline_dir.resolve()).replace(os.sep, "/")


def repair_media_path(value: str, timeline_dir: Path) -> str:
    """Keep a timeline path if valid, otherwise repair common moved-output paths."""
    path = resolve(timeline_dir, value)
    if path is not None and path.exists():
        return value

    original = Path(value)
    candidates = [
        Path.cwd() / value,
        Path.cwd() / original.name,
        Path.cwd() / "generated_ass" / original.name,
        Path.cwd() / "product_images" / "product_images" / original.name,
        Path.cwd() / "product_images" / "fetched_images_aj" / original.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return timeline_relative(candidate, timeline_dir)
    return value


def subtitle_ass_paths(timeline: dict[str, Any], timeline_path: Path) -> list[Path]:
    base = timeline_path.resolve().parent
    values = timeline.get("base_subtitles_ass_files")
    if not values:
        single = timeline.get("base_subtitles_ass")
        values = [single] if single else []
    paths: list[Path] = []
    for value in values:
        path = resolve(base, str(value))
        if path is not None and not path.exists():
            fallback = Path.cwd() / "generated_ass" / path.name
            if fallback.exists():
                path = fallback
        if path is not None:
            paths.append(path)
    return paths


def repair_clean_timeline_media_paths(timeline: dict[str, Any], timeline_dir: Path) -> None:
    """Repair product image/logo paths in a generated clean timeline if they moved."""
    for product in timeline.get("products", []):
        if not isinstance(product, dict):
            continue
        images = product.get("images")
        if isinstance(images, list):
            product["images"] = [repair_media_path(str(image), timeline_dir) for image in images]
        logo = product.get("flipkart_logo")
        if logo:
            product["flipkart_logo"] = repair_media_path(str(logo), timeline_dir)


def find_script_record(json_dir: Path, itm_id: str, required_fields: list[str]) -> tuple[Path, str, dict[str, Any]]:
    """Find the matching script record, ignoring product catalog rows without script fields."""
    candidates: list[tuple[Path, str, dict[str, Any]]] = []
    for path, data in load_json_files(json_dir):
        for match in find_record_by_itm_id(data, itm_id):
            try:
                extract_reference_text(match.record, required_fields)
            except VerificationError:
                continue
            candidates.append((path, match.json_path, match.record))

    if not candidates:
        raise VerificationError(f"No script record for {itm_id!r} with fields {required_fields} under {json_dir}")
    if len(candidates) > 1:
        details = "\n".join(f"  - {path}:{json_path}" for path, json_path, _record in candidates)
        raise VerificationError(f"Multiple script records found for {itm_id!r}:\n{details}")
    return candidates[0]


def compare_against_record(record: dict[str, Any], ass_path: Path, reference_field: str) -> tuple[list[Issue], str]:
    reference = extract_reference_text(record, [reference_field])
    events = parse_ass_events(ass_path)
    if not events:
        raise VerificationError(f"No Dialogue events found in ASS file: {ass_path}")

    reference_raw_tokens = tokenize_product_aware(reference.text)
    ass_raw_tokens = tokenize_ass_events(events)
    reference_tokens = normalize_tokens(reference_raw_tokens)
    ass_tokens = normalize_tokens(ass_raw_tokens)
    comparison = compare_token_sequences(reference_tokens, ass_tokens)
    issues = compare_protected_tokens(
        extract_protected_tokens(reference_raw_tokens),
        extract_protected_tokens(ass_raw_tokens),
    )
    if not comparison.exact:
        issues.extend(map_diff_to_ass_events(comparison, ass_raw_tokens))
    return issues, reference.field_used


def split_dialogue_line(line: str, event_format: list[str]) -> tuple[list[str], str] | None:
    if not line.startswith("Dialogue:"):
        return None
    payload = line.split(":", 1)[1].strip()
    values = payload.split(",", max(len(event_format) - 1, 1))
    if len(values) < len(event_format):
        values.extend([""] * (len(event_format) - len(values)))
    return values, payload


def replace_visible_phrase(text: str, ass_phrase: str, reference_phrase: str) -> tuple[str, bool]:
    """Replace a visible phrase in ASS text while leaving tags around it intact."""
    if not ass_phrase or not reference_phrase:
        return text, False
    pattern = re.compile(r"\b" + re.escape(ass_phrase) + r"\b", flags=re.IGNORECASE)
    updated, count = pattern.subn(reference_phrase, text, count=1)
    if count:
        return updated, True

    # Allow ASS override blocks {…} and/or whitespace between tokens — handles karaoke-tagged
    # multi-word mismatches like "{\k10}every {\k10}eddy" matching phrase "every eddy".
    _ASS_GAP = r"(?:\s*\{[^}]*\}\s*|\s)+"
    parts = ass_phrase.split()
    spaced = _ASS_GAP.join(re.escape(part) for part in parts)
    pattern = re.compile(r"\b" + spaced + r"\b", flags=re.IGNORECASE)
    updated, count = pattern.subn(reference_phrase, text, count=1)
    return updated, bool(count)


def _fix_karaoke_spaces(text: str) -> str:
    """Ensure one space between karaoke \\k-tagged words; strip leading/trailing visible spaces."""
    # Add space before {\k...} when preceded by a word character (no space already there)
    text = re.sub(r"(\w)(\{\\k)", r"\1 \2", text)
    # Strip leading/trailing spaces from visible segments only (outside tags)
    parts = re.split(r"(\{[^}]*\})", text)
    for i, part in enumerate(parts):
        if not part.startswith("{"):
            parts[i] = re.sub(r" +", " ", part).strip()
    return "".join(parts)


def _enforce_subtitle_style(line: str, font_name: str, font_size: int) -> str:
    if not line.lstrip().startswith("Style:"):
        return line
    prefix, payload = line.split(":", 1)
    fields = [field.strip() for field in payload.split(",")]
    if len(fields) < 23:
        return line
    style_key = fields[0].strip().lower()
    if style_key not in {"default", "karaoke", "subtitle_style", "subtitles_style", "subtitles_styles"}:
        return line
    fields[1] = str(font_name)
    fields[2] = str(font_size)
    return prefix + ": " + ",".join(fields)


def write_clean_ass(
    ass_path: Path,
    clean_path: Path,
    issues: list[Issue],
    subtitle_font: str = "Montserrat SemiBold",
    subtitle_font_size: int = 40,
) -> tuple[int, list[Issue]]:
    """Apply deterministic event-local replacements and write a clean ASS file."""
    replacements: dict[int, list[Issue]] = {}
    for issue in issues:
        if issue.event_index is None:
            continue
        if issue.issue_type == "text_replace" or issue.issue_type == "product_decimal_mismatch":
            replacements.setdefault(issue.event_index, []).append(issue)

    lines = ass_path.read_text(encoding="utf-8").splitlines()
    event_format = ["Layer", "Start", "End", "Style", "Name", "MarginL", "MarginR", "MarginV", "Effect", "Text"]
    in_events = False
    dialogue_index = 0
    applied = 0
    unapplied: list[Issue] = []
    output: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_events = stripped.lower() == "[events]"
            output.append(line)
            continue
        if in_events and stripped.lower().startswith("format:"):
            event_format = [field.strip() for field in stripped.split(":", 1)[1].split(",")]
            output.append(line)
            continue
        if not in_events and line.lstrip().startswith("Style:"):
            output.append(_enforce_subtitle_style(line, subtitle_font, subtitle_font_size))
            continue
        if in_events and line.startswith("Dialogue:"):
            parsed = split_dialogue_line(line, event_format)
            if parsed is None:
                output.append(line)
                continue
            values, _payload = parsed
            for issue in replacements.get(dialogue_index, []):
                text_index = event_format.index("Text") if "Text" in event_format else len(values) - 1
                new_text, changed = replace_visible_phrase(values[text_index], issue.ass_token, issue.reference_token)
                if changed:
                    values[text_index] = new_text
                    applied += 1
                else:
                    unapplied.append(issue)
            text_idx = event_format.index("Text") if "Text" in event_format else len(values) - 1
            values[text_idx] = _fix_karaoke_spaces(values[text_idx])
            output.append("Dialogue: " + ",".join(values))
            dialogue_index += 1
            continue
        output.append(line)

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.write_text("\n".join(output) + "\n", encoding="utf-8")
    return applied, unapplied


def render_with_clean_ass(args: argparse.Namespace) -> int:
    timeline_path = args.timeline
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    style_cfg = json.loads(args.style.read_text(encoding="utf-8"))
    subtitle_cfg = style_cfg.get("subtitle_style", {})
    subtitle_font = str(subtitle_cfg.get("font", CLEAN_ASS_SUBTITLE_FONT)).strip() or CLEAN_ASS_SUBTITLE_FONT
    subtitle_font_size = int(subtitle_cfg.get("font_size", CLEAN_ASS_SUBTITLE_FONT_SIZE))
    timeline_dir = timeline_path.resolve().parent
    itm_id = args.itm_id or (timeline.get("products") or [{}])[0].get("id")
    if not itm_id:
        raise VerificationError("Could not infer --itm-id from timeline products; pass --itm-id explicitly")

    ass_paths = subtitle_ass_paths(timeline, timeline_path)
    if not ass_paths:
        raise VerificationError("Timeline has no base_subtitles_ass/base_subtitles_ass_files")
    reference_fields = args.reference_field or [f"script{i + 1}" for i in range(len(ass_paths))]
    if len(reference_fields) != len(ass_paths):
        raise VerificationError(
            f"Need {len(ass_paths)} reference field(s), got {len(reference_fields)}: {reference_fields}"
        )

    _json_file, _json_path, record = find_script_record(args.json_dir, str(itm_id), reference_fields)
    clean_paths: list[Path] = []
    total_issues = 0

    for ass_path, reference_field in zip(ass_paths, reference_fields):
        issues, field_used = compare_against_record(record, ass_path, reference_field)
        clean_path = ass_path.with_name(f"{ass_path.stem}.clean.ass")
        applied, unapplied = write_clean_ass(
            ass_path,
            clean_path,
            issues,
            subtitle_font,
            subtitle_font_size,
        )
        clean_issues, _clean_field = compare_against_record(record, clean_path, reference_field)
        total_issues += len(clean_issues)
        print(
            f"clean-ass {ass_path.name} field={field_used} issues_before={len(issues)} "
            f"auto_fixes={applied} unapplied={len(unapplied)} issues_after={len(clean_issues)} -> {clean_path}"
        )
        if clean_issues:
            for issue in clean_issues[:10]:
                print(f"  {issue.severity} {issue.issue_type}: {issue.message}")
            if not args.allow_review:
                raise VerificationError(f"Clean ASS still has {len(clean_issues)} issue(s): {clean_path}")
        clean_paths.append(clean_path)

    clean_timeline = dict(timeline)
    clean_timeline["base_subtitles_ass_files"] = [timeline_relative(path, timeline_dir) for path in clean_paths]
    clean_timeline["base_subtitles_ass"] = clean_timeline["base_subtitles_ass_files"][0]
    repair_clean_timeline_media_paths(clean_timeline, timeline_dir)
    clean_timeline_path = timeline_path.with_name(f"{timeline_path.stem}.clean.json")
    clean_timeline_path.write_text(json.dumps(clean_timeline, indent=2) + "\n", encoding="utf-8")
    print(f"clean-timeline -> {clean_timeline_path}")

    cmd = [
        sys.executable,
        "render_video.py",
        "--style",
        str(args.style),
        "--timeline",
        str(clean_timeline_path),
    ]
    if args.subtitle_primary_color:
        cmd.extend(["--subtitle-primary-color", args.subtitle_primary_color])
    if args.subtitle_secondary_color:
        cmd.extend(["--subtitle-secondary-color", args.subtitle_secondary_color])
    if args.out:
        cmd.extend(["--out", str(args.out)])

    print("render-command:", " ".join(shlex_quote(part) for part in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return result.returncode
    return 1 if total_issues and args.fail_on_review else 0


def shlex_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=+-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean subtitle ASS files, write a clean timeline, and render it.")
    parser.add_argument("--style", required=True, type=Path)
    parser.add_argument("--timeline", required=True, type=Path)
    parser.add_argument("--json-dir", required=True, type=Path)
    parser.add_argument("--itm-id")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--subtitle-primary-color")
    parser.add_argument("--subtitle-secondary-color")
    parser.add_argument("--reference-field", action="append", default=[])
    parser.add_argument("--allow-review", action="store_true")
    parser.add_argument("--fail-on-review", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return render_with_clean_ass(args)
    except VerificationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
