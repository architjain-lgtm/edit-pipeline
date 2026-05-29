#!/usr/bin/env python3
"""Proof-check generated ASS subtitles against source script JSON/JSONL records."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import difflib
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ID_KEYS = {"ITM_ID", "itm_id", "item_id", "id"}
AUTO_REFERENCE_FIELDS = [
    "script",
    "script_text",
    "script1",
    "script2",
    "voiceover",
    "voice_over",
    "narration",
    "narration_text",
    "caption_script",
    "final_script",
    "text",
    "content",
]
LIKELY_TEXT_FIELDS = AUTO_REFERENCE_FIELDS + ["line", "caption", "sentence", "value"]
UNIT_NORMALIZATION = {
    "litre": "l",
    "litres": "l",
    "liter": "l",
    "liters": "l",
    "ltr": "l",
    "ltrs": "l",
    "kilogram": "kg",
    "kilograms": "kg",
    "kgs": "kg",
    "tonne": "ton",
    "tons": "ton",
}
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
PROTECTED_UNITS = {"kg", "l", "ton", "star", "watt", "kwh", "units"}
ACRONYMS = {"iseer", "bee", "inr", "rs"}


@dataclasses.dataclass(frozen=True)
class JsonMatch:
    json_file: Path
    json_path: str
    record: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class ReferenceText:
    text: str
    field_used: str


@dataclasses.dataclass(frozen=True)
class AssEvent:
    event_index: int
    start: str
    end: str
    style: str
    raw_text: str
    clean_text_for_matching: str


@dataclasses.dataclass(frozen=True)
class Token:
    raw: str
    kind: str
    norm: str
    event_index: int | None = None
    start: str = ""
    end: str = ""


@dataclasses.dataclass(frozen=True)
class ProtectedToken:
    raw: str
    norm: str
    kind: str
    token_index: int
    event_index: int | None = None
    start: str = ""
    end: str = ""


@dataclasses.dataclass
class Issue:
    status: str
    severity: str
    issue_type: str
    message: str
    event_index: int | None = None
    start: str = ""
    end: str = ""
    reference_text: str = ""
    ass_text: str = ""
    reference_token: str = ""
    ass_token: str = ""


@dataclasses.dataclass
class ComparisonResult:
    exact: bool
    opcodes: list[tuple[str, int, int, int, int]]
    reference_tokens: list[str]
    ass_tokens: list[str]


class VerificationError(RuntimeError):
    """Raised for runtime/config errors that should exit with code 2."""


TOKEN_RE = re.compile(
    r"""
    (?P<currency>₹\s*\d[\d,]*(?:\.\d+)?|(?:rs\.?|inr)\s*\d[\d,]*(?:\.\d+)?)
    |(?P<model>[A-Za-z]{1,}[A-Za-z0-9]*[-/][A-Za-z0-9][A-Za-z0-9-/]*)
    |(?P<number>\d[\d,]*(?:\.\d+)?%?)
    |(?P<word>[A-Za-z]+(?:'[A-Za-z]+)?)
    """,
    re.VERBOSE | re.IGNORECASE,
)
ASS_TAG_RE = re.compile(r"\{[^{}]*\}")


def load_json_files(json_dir: Path) -> list[tuple[Path, Any]]:
    """Load JSON and JSONL files recursively from a directory."""
    if not json_dir.exists():
        raise VerificationError(f"JSON directory does not exist: {json_dir}")
    if not json_dir.is_dir():
        raise VerificationError(f"--json-dir is not a directory: {json_dir}")

    loaded: list[tuple[Path, Any]] = []
    paths = sorted([*json_dir.rglob("*.json"), *json_dir.rglob("*.jsonl")])
    for path in paths:
        try:
            if path.suffix.lower() == ".jsonl":
                rows = []
                for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise VerificationError(f"Invalid JSONL in {path}:{line_no}: {exc}") from exc
                loaded.append((path, rows))
            else:
                loaded.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except UnicodeDecodeError as exc:
            raise VerificationError(f"Could not read UTF-8 JSON file {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise VerificationError(f"Invalid JSON in {path}: {exc}") from exc
    if not loaded:
        raise VerificationError(f"No .json or .jsonl files found under {json_dir}")
    return loaded


def find_record_by_itm_id(data: Any, itm_id: str) -> list[JsonMatch]:
    """Recursively find dict records whose ID key matches the requested ITM ID."""
    wanted = str(itm_id)
    matches: list[JsonMatch] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if any(str(node.get(key)) == wanted for key in ID_KEYS if key in node):
                matches.append(JsonMatch(Path(), path, node))
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(data, "$")
    return matches


def find_unique_record(json_dir: Path, itm_id: str) -> JsonMatch:
    all_matches: list[JsonMatch] = []
    for path, data in load_json_files(json_dir):
        for match in find_record_by_itm_id(data, itm_id):
            all_matches.append(JsonMatch(path, match.json_path, match.record))
    if not all_matches:
        raise VerificationError(f"No JSON record found for ITM ID {itm_id!r} under {json_dir}")
    if len(all_matches) > 1:
        details = "\n".join(f"  - {m.json_file}:{m.json_path}" for m in all_matches)
        raise VerificationError(f"Multiple JSON records found for ITM ID {itm_id!r}:\n{details}")
    return all_matches[0]


def extract_reference_text(record: dict[str, Any], preferred_fields: list[str]) -> ReferenceText:
    """Extract source-of-truth script text from a matched JSON record."""
    fields = preferred_fields or AUTO_REFERENCE_FIELDS
    for field in fields:
        if field in record:
            text = value_to_text(record[field])
            if text:
                return ReferenceText(text=text, field_used=field)

    script_number_fields = sorted(
        (key for key in record if re.fullmatch(r"script\d+", str(key), flags=re.IGNORECASE)),
        key=lambda value: int(re.search(r"\d+", value).group(0)),  # type: ignore[union-attr]
    )
    if script_number_fields:
        parts = [value_to_text(record[key]) for key in script_number_fields]
        text = " ".join(part for part in parts if part)
        if text:
            return ReferenceText(text=text, field_used="+".join(script_number_fields))

    keys = ", ".join(sorted(map(str, record.keys())))
    raise VerificationError(f"No reference script text found. Matched JSON object keys: {keys}")


def value_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for field in LIKELY_TEXT_FIELDS:
                    if field in item:
                        item_text = value_to_text(item[field])
                        if item_text:
                            parts.append(item_text)
                            break
        return " ".join(parts).strip()
    return ""


def parse_ass_events(ass_path: Path) -> list[AssEvent]:
    """Parse Dialogue lines inside the [Events] section of an ASS file."""
    if not ass_path.exists():
        raise VerificationError(f"ASS file does not exist: {ass_path}")
    common_format = ["Layer", "Start", "End", "Style", "Name", "MarginL", "MarginR", "MarginV", "Effect", "Text"]
    in_events = False
    fields = common_format
    events: list[AssEvent] = []

    for raw_line in ass_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_events = line.lower() == "[events]"
            continue
        if not in_events:
            continue
        if line.lower().startswith("format:"):
            fields = [field.strip() for field in line.split(":", 1)[1].split(",")]
            continue
        if not line.startswith("Dialogue:"):
            continue
        payload = line.split(":", 1)[1].strip()
        values = payload.split(",", max(len(fields) - 1, 1))
        if len(values) < len(fields):
            values.extend([""] * (len(fields) - len(values)))
        row = dict(zip(fields, values))
        raw_text = row.get("Text", values[-1] if values else "")
        events.append(
            AssEvent(
                event_index=len(events),
                start=row.get("Start", ""),
                end=row.get("End", ""),
                style=row.get("Style", ""),
                raw_text=raw_text,
                clean_text_for_matching=clean_ass_text(raw_text),
            )
        )
    return events


def clean_ass_text(text: str) -> str:
    """Remove ASS visual markup and normalize line-break escapes for matching."""
    text = ASS_TAG_RE.sub(" ", text)
    text = text.replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_text_for_tokenization(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\bi\s*[- ]\s*seer\b", "ISEER", text, flags=re.IGNORECASE)
    text = re.sub(r"\bb\s+e\s+e\b", "BEE", text, flags=re.IGNORECASE)
    text = re.sub(r"\bone\s+point\s+five\b", "1.5", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthree\s+point\s+nine\s+five\b", "3.95", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfive\s+star\b", "5 Star", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthree\s+star\b", "3 Star", text, flags=re.IGNORECASE)
    return text


def tokenize_product_aware(text: str) -> list[Token]:
    """Tokenize text while preserving product-critical numbers, prices, and models."""
    normalized_input = normalize_text_for_tokenization(text)
    tokens: list[Token] = []
    for match in TOKEN_RE.finditer(normalized_input):
        raw = match.group(0).strip()
        kind = match.lastgroup or "word"
        tokens.append(Token(raw=raw, kind=kind, norm=normalize_token(raw, kind)))
    return tokens


def tokenize_ass_events(events: list[AssEvent]) -> list[Token]:
    tokens: list[Token] = []
    for event in events:
        for token in tokenize_product_aware(event.clean_text_for_matching):
            tokens.append(
                Token(
                    raw=token.raw,
                    kind=token.kind,
                    norm=token.norm,
                    event_index=event.event_index,
                    start=event.start,
                    end=event.end,
                )
            )
    return tokens


def normalize_token(raw: str, kind: str) -> str:
    value = raw.strip().lower()
    value = value.replace("₹", "rs ")
    value = re.sub(r"[,_]", "", value)
    value = re.sub(r"\brs\.\s*", "rs ", value)
    value = re.sub(r"[^a-z0-9.%/-]+", "", value) if kind == "model" else value
    value = UNIT_NORMALIZATION.get(value, value)
    if kind == "currency":
        value = re.sub(r"\s+", "", value)
    if kind == "word":
        value = re.sub(r"^[^\w]+|[^\w]+$", "", value)
        value = UNIT_NORMALIZATION.get(value, value)
    return value


def normalize_tokens(tokens: list[Token]) -> list[str]:
    """Normalize tokens and combine safe multi-token ASR/product variants."""
    values = [token.norm for token in tokens if token.norm]
    result: list[str] = []
    i = 0
    while i < len(values):
        current = values[i]
        nxt = values[i + 1] if i + 1 < len(values) else ""
        third = values[i + 2] if i + 2 < len(values) else ""

        if current == "i" and nxt == "seer":
            result.append("iseer")
            i += 2
        elif current == "b" and nxt == "e" and third == "e":
            result.append("bee")
            i += 3
        elif current in NUMBER_WORDS and nxt == "star":
            result.extend([NUMBER_WORDS[current], "star"])
            i += 2
        elif current == "one" and nxt == "point" and third == "five":
            result.append("1.5")
            i += 3
        elif current == "three" and nxt == "point" and third == "nine" and i + 3 < len(values) and values[i + 3] == "five":
            result.append("3.95")
            i += 4
        else:
            result.append(current)
            i += 1
    return result


def extract_protected_tokens(tokens: list[Token]) -> list[ProtectedToken]:
    """Extract product-critical tokens and short phrases from token stream."""
    normalized = normalize_tokens(tokens)
    protected: list[ProtectedToken] = []
    norm_to_first_token: list[Token] = []

    # Re-run simple alignment for normalized output. This is sufficient for event mapping.
    raw_i = 0
    for norm in normalized:
        while raw_i < len(tokens) and not tokens[raw_i].norm:
            raw_i += 1
        norm_to_first_token.append(tokens[min(raw_i, len(tokens) - 1)] if tokens else Token("", "", ""))
        raw_i = min(raw_i + 1, len(tokens))

    for i, norm in enumerate(normalized):
        token = norm_to_first_token[i]
        raw = token.raw
        kind = token.kind
        protected_kind: str | None = None
        protected_norm = norm

        if re.fullmatch(r"\d+\.\d+%?", norm):
            protected_kind = "decimal"
        elif kind == "currency" or norm.startswith(("rs", "inr")):
            protected_kind = "price"
        elif kind == "model" or is_model_like(raw):
            protected_kind = "model"
            protected_norm = compact_model(norm)
        elif norm in ACRONYMS or (raw.isupper() and raw.isalpha() and len(raw) >= 2):
            protected_kind = "acronym"
        elif norm.isdigit() and i + 1 < len(normalized) and normalized[i + 1] == "star":
            protected_kind = "star_rating"
            protected_norm = f"{norm} star"
            raw = f"{raw} {norm_to_first_token[i + 1].raw}"
        elif is_number_like(norm) and i + 1 < len(normalized) and normalized[i + 1] in PROTECTED_UNITS:
            protected_kind = "capacity"
            protected_norm = f"{norm} {normalized[i + 1]}"
            raw = f"{raw} {norm_to_first_token[i + 1].raw}"
        elif norm in {"annual", "energy", "consumption", "units", "year"}:
            protected_kind = "energy_term"

        if protected_kind:
            protected.append(
                ProtectedToken(
                    raw=raw,
                    norm=protected_norm,
                    kind=protected_kind,
                    token_index=i,
                    event_index=token.event_index,
                    start=token.start,
                    end=token.end,
                )
            )
    return protected


def is_number_like(value: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", value))


def is_model_like(raw: str) -> bool:
    return bool(re.search(r"[A-Za-z]", raw) and re.search(r"\d", raw) and (re.search(r"[-/]", raw) or raw.isupper()))


def compact_model(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def compare_token_sequences(reference_tokens: list[str], ass_tokens: list[str]) -> ComparisonResult:
    """Fast exact compare first; run difflib only for mismatches."""
    if reference_tokens == ass_tokens:
        return ComparisonResult(True, [], reference_tokens, ass_tokens)
    matcher = difflib.SequenceMatcher(a=reference_tokens, b=ass_tokens, autojunk=False)
    return ComparisonResult(False, matcher.get_opcodes(), reference_tokens, ass_tokens)


def compare_protected_tokens(reference_protected: list[ProtectedToken], ass_protected: list[ProtectedToken]) -> list[Issue]:
    """Compare product-critical tokens in order, producing ERROR/REVIEW issues."""
    issues: list[Issue] = []
    ass_by_norm = Counter(token.norm for token in ass_protected)
    ass_norms = [token.norm for token in ass_protected]
    ass_raw_by_norm: dict[str, ProtectedToken] = {token.norm: token for token in ass_protected}

    for ref in reference_protected:
        if ass_by_norm[ref.norm] > 0:
            ass_by_norm[ref.norm] -= 1
            ass = ass_raw_by_norm.get(ref.norm)
            if ass and ref.raw.lower() != ass.raw.lower() and ref.kind in {"acronym", "star_rating"}:
                issues.append(
                    Issue(
                        status="PASS",
                        severity="INFO",
                        issue_type="normalized_fix",
                        message=f"Accepted normalized product token {ass.raw!r} as {ref.raw!r}.",
                        event_index=ass.event_index,
                        start=ass.start,
                        end=ass.end,
                        reference_token=ref.raw,
                        ass_token=ass.raw,
                    )
                )
            continue

        if ref.kind == "decimal":
            compact = ref.norm.replace(".", "").replace("%", "")
            collapsed = next((token for token in ass_protected if token.norm.replace("%", "") == compact), None)
            if collapsed:
                issues.append(
                    Issue(
                        status="FAIL",
                        severity="ERROR",
                        issue_type="product_decimal_mismatch",
                        message=f"Reference has {ref.raw} but ASS appears to have {collapsed.raw}. Decimal point may be lost.",
                        event_index=collapsed.event_index,
                        start=collapsed.start,
                        end=collapsed.end,
                        reference_token=ref.raw,
                        ass_token=collapsed.raw,
                    )
                )
                continue
        if ref.kind == "model":
            equivalent = next((token for token in ass_protected if token.kind == "model" and compact_model(token.norm) == ref.norm), None)
            if equivalent:
                issues.append(
                    Issue(
                        status="REVIEW",
                        severity="REVIEW",
                        issue_type="model_format_changed",
                        message=f"Model number formatting changed from {ref.raw!r} to {equivalent.raw!r}; review if acceptable.",
                        event_index=equivalent.event_index,
                        start=equivalent.start,
                        end=equivalent.end,
                        reference_token=ref.raw,
                        ass_token=equivalent.raw,
                    )
                )
                continue

        nearby = ", ".join(ass_norms[max(0, ref.token_index - 3) : ref.token_index + 4])
        issues.append(
            Issue(
                status="FAIL",
                severity="ERROR",
                issue_type=f"missing_product_{ref.kind}",
                message=f"Protected product token {ref.raw!r} from reference is missing in ASS. Nearby ASS protected tokens: {nearby}",
                reference_token=ref.raw,
            )
        )
    return issues


def map_diff_to_ass_events(diff: ComparisonResult, ass_tokens: list[Token]) -> list[Issue]:
    """Map token diff ranges to nearby ASS events for concise review."""
    issues: list[Issue] = []
    for tag, i1, i2, j1, j2 in diff.opcodes:
        if tag == "equal":
            continue
        ass_slice = ass_tokens[j1:j2] if j1 < len(ass_tokens) else ass_tokens[max(0, j1 - 1) : j1]
        event = ass_slice[0] if ass_slice else None
        ref_text = " ".join(diff.reference_tokens[i1:i2])
        ass_text = " ".join(diff.ass_tokens[j1:j2])
        issues.append(
            Issue(
                status="REVIEW",
                severity="REVIEW",
                issue_type=f"text_{tag}",
                message=f"Token {tag}: reference={ref_text!r}, ass={ass_text!r}",
                event_index=event.event_index if event else None,
                start=event.start if event else "",
                end=event.end if event else "",
                reference_text=ref_text,
                ass_text=ass_text,
                reference_token=ref_text,
                ass_token=ass_text,
            )
        )
    return issues


def write_csv_report(path: Path, issues: list[Issue], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "status",
        "severity",
        "itm_id",
        "json_file",
        "json_path",
        "event_index",
        "start",
        "end",
        "issue_type",
        "reference_text",
        "ass_text",
        "reference_token",
        "ass_token",
        "message",
    ]
    rows = issues or [Issue(status="PASS", severity="INFO", issue_type="pass", message="No issues found.")]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for issue in rows:
            writer.writerow(
                {
                    "status": issue.status,
                    "severity": issue.severity,
                    "itm_id": metadata.get("itm_id", ""),
                    "json_file": metadata.get("json_file", ""),
                    "json_path": metadata.get("json_path", ""),
                    "event_index": "" if issue.event_index is None else issue.event_index,
                    "start": issue.start,
                    "end": issue.end,
                    "issue_type": issue.issue_type,
                    "reference_text": issue.reference_text,
                    "ass_text": issue.ass_text,
                    "reference_token": issue.reference_token,
                    "ass_token": issue.ass_token,
                    "message": issue.message,
                }
            )


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_verification(
    ass_path: Path,
    json_dir: Path,
    itm_id: str,
    preferred_fields: list[str],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    match = find_unique_record(json_dir, itm_id)
    reference = extract_reference_text(match.record, preferred_fields)
    events = parse_ass_events(ass_path)
    if not events:
        raise VerificationError(f"No Dialogue events found in ASS file: {ass_path}")

    reference_raw_tokens = tokenize_product_aware(reference.text)
    ass_raw_tokens = tokenize_ass_events(events)
    reference_tokens = normalize_tokens(reference_raw_tokens)
    ass_tokens = normalize_tokens(ass_raw_tokens)
    comparison = compare_token_sequences(reference_tokens, ass_tokens)

    reference_protected = extract_protected_tokens(reference_raw_tokens)
    ass_protected = extract_protected_tokens(ass_raw_tokens)
    product_issues = compare_protected_tokens(reference_protected, ass_protected)
    general_issues = [] if comparison.exact else map_diff_to_ass_events(comparison, ass_raw_tokens)

    issues = [*product_issues]
    if strict or any(issue.severity == "ERROR" for issue in product_issues) or not comparison.exact:
        issues.extend(general_issues)

    overall_status = "PASS"
    if any(issue.severity == "ERROR" for issue in issues):
        overall_status = "FAIL"
    elif any(issue.severity == "REVIEW" for issue in issues) or not comparison.exact:
        overall_status = "REVIEW"

    return {
        "overall_status": overall_status,
        "itm_id": itm_id,
        "matched_json_file": str(match.json_file),
        "matched_json_path": match.json_path,
        "reference_field_used": reference.field_used,
        "ass_file": str(ass_path),
        "token_counts": {
            "reference": len(reference_tokens),
            "ass": len(ass_tokens),
            "reference_protected": len(reference_protected),
            "ass_protected": len(ass_protected),
        },
        "product_check": {
            "status": "FAIL" if any(issue.severity == "ERROR" for issue in product_issues) else "PASS",
            "reference_protected": [dataclasses.asdict(token) for token in reference_protected],
            "ass_protected": [dataclasses.asdict(token) for token in ass_protected],
        },
        "general_text_check": {
            "status": "PASS" if comparison.exact else "REVIEW",
            "exact_token_match": comparison.exact,
        },
        "issues": [dataclasses.asdict(issue) for issue in issues],
        "_issue_objects": issues,
    }


def print_diff(issues: list[Issue]) -> None:
    if not issues:
        print("Diff: no issues.")
        return
    print("Diff summary:")
    for issue in issues[:30]:
        loc = f" event={issue.event_index} {issue.start}->{issue.end}" if issue.event_index is not None else ""
        print(f"- {issue.severity} {issue.issue_type}{loc}: {issue.message}")
    if len(issues) > 30:
        print(f"... {len(issues) - 30} more issue(s)")


def run_self_test() -> int:
    def write_ass(path: Path, text: str) -> None:
        path.write_text(
            "[Script Info]\nScriptType: v4.00+\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            f"Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{text}\n",
            encoding="utf-8",
        )

    cases: list[tuple[str, str, str, str, str]] = [
        ("exact match", "ITEM1", "This AC cools fast", "This AC cools fast", "PASS"),
        ("ISEER 3.95 preserved", "ITEM2", "This AC has ISEER 3.95 rating", "This AC has ISEER 3.95 rating", "PASS"),
        ("3.95 vs 395 fails", "ITEM3", "This AC has ISEER 3.95 and 5 Star rating", "This AC has ISEER 395 and five star rating", "FAIL"),
        ("5 Star vs five star passes", "ITEM4", "This AC has 5 Star rating", "This AC has five star rating", "PASS"),
        ("ASS commas parse", "ITEM5", "Dust, noise, and insects stay out", "Dust, noise, and insects stay out", "PASS"),
        ("ASS tags stripped", "ITEM6", "This AC has ISEER 3.95", r"{\\pos(1,2)}This AC has {\\i1}ISEER 3.95", "PASS"),
    ]
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, itm_id, reference, ass_text, expected in cases:
            json_dir = root / name.replace(" ", "_")
            json_dir.mkdir()
            (json_dir / "sample.json").write_text(json.dumps({"item_id": itm_id, "script": reference}), encoding="utf-8")
            ass_path = json_dir / "sample.ass"
            write_ass(ass_path, ass_text)
            payload = run_verification(ass_path, json_dir, itm_id, [])
            got = payload["overall_status"]
            ok = got == expected
            failures += 0 if ok else 1
            print(f"{'PASS' if ok else 'FAIL'} self-test: {name} -> {got}")

        wrong_dir = root / "wrong_id"
        wrong_dir.mkdir()
        (wrong_dir / "sample.json").write_text(json.dumps({"item_id": "RIGHT", "script": "Hello"}), encoding="utf-8")
        ass_path = wrong_dir / "sample.ass"
        write_ass(ass_path, "Hello")
        try:
            run_verification(ass_path, wrong_dir, "WRONG", [])
        except VerificationError:
            print("PASS self-test: wrong ITM_ID fails -> runtime_error")
        else:
            failures += 1
            print("FAIL self-test: wrong ITM_ID fails -> no error")

    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify generated ASS subtitles against source script JSON/JSONL.")
    parser.add_argument("--ass", type=Path)
    parser.add_argument("--json-dir", type=Path)
    parser.add_argument("--itm-id")
    parser.add_argument("--report-csv", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--show-diff", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--reference-field", action="append", default=[])
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    missing = [name for name in ["ass", "json_dir", "itm_id"] if getattr(args, name) is None]
    if missing:
        print(f"Missing required argument(s): {', '.join('--' + name.replace('_', '-') for name in missing)}", file=sys.stderr)
        return 2
    try:
        payload = run_verification(args.ass, args.json_dir, args.itm_id, args.reference_field, strict=args.strict)
        issues: list[Issue] = payload.pop("_issue_objects")
        metadata = {
            "itm_id": args.itm_id,
            "json_file": payload["matched_json_file"],
            "json_path": payload["matched_json_path"],
        }
        if args.report_csv:
            write_csv_report(args.report_csv, issues, metadata)
        if args.report_json:
            write_json_report(args.report_json, payload)
        if args.show_diff:
            print_diff(issues)
        print(
            f"{payload['overall_status']} itm_id={args.itm_id} "
            f"tokens ref={payload['token_counts']['reference']} ass={payload['token_counts']['ass']} "
            f"issues={len(issues)}"
        )
        if args.fail_on_error and payload["overall_status"] != "PASS":
            return 1
        return 0
    except VerificationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
