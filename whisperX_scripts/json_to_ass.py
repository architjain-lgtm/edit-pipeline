"""
Convert a word-level alignment JSON (produced by align.py / run_batch20.py)
to an ASS subtitle file with per-word karaoke {\\k} tags.

Format matches the reference:
    Dialogue: 0,0:00:0.44,0:00:1.22,Default,,0,0,0,,{\\k14}If {\\k14}your {\\k22}room {\\k28}feels

{\\k} value is in centiseconds and equals the time from this word's start
to the next word's start (so it covers the inter-word gap).  The last word
in a group uses (group_end - word_start).
"""

import json
import sys
from pathlib import Path

ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 384
PlayResY: 288
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00ff00,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

MAX_WORDS = 4
GAP_BREAK = 0.30  # seconds — start a new line at pauses this long or longer


def ass_time(seconds: float) -> str:
    """H:MM:S.cc  (seconds not zero-padded, matching reference format)."""
    total_cs = round(seconds * 100)
    h  = total_cs // 360000; total_cs %= 360000
    m  = total_cs //   6000; total_cs %=   6000
    s  = total_cs //    100
    cs = total_cs %     100
    return f"{h}:{m:02d}:{s}.{cs:02d}"


def group_words(words: list[dict]) -> list[list[dict]]:
    """
    Split into groups of up to MAX_WORDS, also breaking at pauses >= GAP_BREAK.
    This produces natural phrase-level subtitle lines.
    """
    if not words:
        return []
    groups, current = [], [words[0]]
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if len(current) >= MAX_WORDS or gap >= GAP_BREAK:
            groups.append(current)
            current = []
        current.append(words[i])
    if current:
        groups.append(current)
    return groups


def build_dialogue(layer: int, group: list[dict]) -> str:
    line_start = group[0]["start"]
    line_end   = group[-1]["end"]

    parts = []
    for i, w in enumerate(group):
        if i < len(group) - 1:
            k_cs = round((group[i + 1]["start"] - w["start"]) * 100)
        else:
            k_cs = round((line_end - w["start"]) * 100)
        k_cs = max(1, k_cs)
        parts.append(f"{{\\k{k_cs}}}{w['word']}")

    text = " ".join(parts)
    return (
        f"Dialogue: {layer},"
        f"{ass_time(line_start)},"
        f"{ass_time(line_end)},"
        f"Default,,0,0,0,,{text}"
    )


def json_to_ass(json_path: str | Path, ass_path: str | Path | None = None) -> Path:
    json_path = Path(json_path)
    with open(json_path) as f:
        data = json.load(f)

    words = data.get("words", [])
    if not words:
        raise ValueError(f"No words found in {json_path}")

    groups   = group_words(words)
    dialogues = [build_dialogue(i, g) for i, g in enumerate(groups)]
    content  = ASS_HEADER + "\n".join(dialogues) + "\n"

    if ass_path is None:
        ass_path = json_path.with_suffix(".ass")
    ass_path = Path(ass_path)
    ass_path.write_text(content, encoding="utf-8")
    return ass_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <alignment.json> [output.ass]")
        sys.exit(1)
    out = json_to_ass(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"Written: {out}")
