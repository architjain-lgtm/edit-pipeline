#!/usr/bin/env python3
"""Profile Python methods inside run_full_batch_pipeline.py, enriched with
per-item stage timings from the pipeline's own report JSON.

Usage (from project root):
    python code/profile_run_full_batch_pipeline.py -- \\
        --batch-dir ... --report-json outputs/batch1/report.json ...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "run_full_batch_pipeline.py"


def _read_system_cpu_ticks() -> tuple[int, int] | None:
    """Returns (total_ticks, idle_ticks) from /proc/stat line 1."""
    try:
        with open("/proc/stat") as fh:
            line = fh.readline()
        parts = line.split()
        if not parts or parts[0] != "cpu":
            return None
        vals = [int(x) for x in parts[1:]]
        total = sum(vals)
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
        return total, idle
    except (OSError, ValueError, IndexError):
        return None


def _read_self_cpu_ticks() -> int | None:
    """Returns (utime + stime) ticks for the current Python process (not children)."""
    try:
        with open("/proc/self/stat") as fh:
            text = fh.read()
        rparen = text.rfind(")")
        if rparen < 0:
            return None
        fields = text[rparen + 2:].split()
        return int(fields[11]) + int(fields[12])
    except (OSError, ValueError, IndexError):
        return None

TARGET_FUNCTIONS = [
    "discover_batch_items",
    "select_items",
    "stream_selected_tsv",
    "build_tsv_index",
    "load_or_build_tsv_index",
    "resolve_from_index",
    "load_selected_script_records",
    "call_image_picker_api",
    "media_duration",
    "run_command",
    "trim_video",
    "generate_ass",
    "build_timeline_for_item",
    "clean_ass_and_timeline",
    "render_item",
    "prepare_item",
    "fetch_tag_windows",
    "fetch_bridge_overlay_points",
]


@dataclass
class FunctionStats:
    call_count: int = 0
    wall_inclusive_s: float = 0.0
    wall_exclusive_s: float = 0.0
    cpu_inclusive_s: float = 0.0
    cpu_exclusive_s: float = 0.0
    max_concurrent_calls: int = 0


@dataclass
class Frame:
    name: str
    start_wall: float
    start_cpu: float
    child_wall: float = 0.0
    child_cpu: float = 0.0


class ProfilerState:
    def __init__(self) -> None:
        self.stats: dict[str, FunctionStats] = defaultdict(FunctionStats)
        self.active_counts: dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
        self.local = threading.local()
        self.thread_samples: list[dict[str, float | int]] = []
        self._stop_event = threading.Event()
        self._sampler: threading.Thread | None = None

    def stack(self) -> list[Frame]:
        stack = getattr(self.local, "stack", None)
        if stack is None:
            stack = []
            self.local.stack = stack
        return stack

    def start_thread_sampler(self, interval: float) -> None:
        clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])

        def loop() -> None:
            started = time.monotonic()
            prev_wall = started
            prev_sys = _read_system_cpu_ticks()
            prev_proc = _read_self_cpu_ticks()
            while not self._stop_event.is_set():
                self._stop_event.wait(interval)
                now = time.monotonic()
                d_wall = max(now - prev_wall, 1e-6)
                try:
                    threads = len(os.listdir("/proc/self/task"))
                except OSError:
                    threads = threading.active_count()
                sys_ticks = _read_system_cpu_ticks()
                proc_ticks = _read_self_cpu_ticks()

                system_cpu_pct: float | None = None
                process_cpu_pct: float | None = None
                if prev_sys and sys_ticks:
                    d_total = sys_ticks[0] - prev_sys[0]
                    d_idle = sys_ticks[1] - prev_sys[1]
                    if d_total > 0:
                        system_cpu_pct = round(100.0 * (1.0 - d_idle / d_total), 2)
                if prev_proc is not None and proc_ticks is not None:
                    d_ticks = max(0, proc_ticks - prev_proc)
                    process_cpu_pct = round(100.0 * (d_ticks / clk_tck) / d_wall, 2)

                self.thread_samples.append({
                    "elapsed_s": round(now - started, 4),
                    "thread_count": threads,
                    "system_cpu_pct": system_cpu_pct,
                    "process_cpu_pct": process_cpu_pct,
                })
                prev_wall = now
                prev_sys = sys_ticks
                prev_proc = proc_ticks

        self._sampler = threading.Thread(target=loop, name="thread-count-sampler", daemon=True)
        self._sampler.start()

    def stop_thread_sampler(self) -> None:
        self._stop_event.set()
        if self._sampler is not None:
            self._sampler.join(timeout=2.0)


def wrap_callable(state: ProfilerState, name: str, func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        stack = state.stack()
        frame = Frame(name=name, start_wall=time.perf_counter(), start_cpu=time.thread_time())
        with state.lock:
            stat = state.stats[name]
            stat.call_count += 1
            state.active_counts[name] += 1
            stat.max_concurrent_calls = max(stat.max_concurrent_calls, state.active_counts[name])
        stack.append(frame)
        try:
            return func(*args, **kwargs)
        finally:
            end_wall = time.perf_counter()
            end_cpu = time.thread_time()
            stack.pop()

            inclusive_wall = end_wall - frame.start_wall
            inclusive_cpu = end_cpu - frame.start_cpu
            exclusive_wall = inclusive_wall - frame.child_wall
            exclusive_cpu = inclusive_cpu - frame.child_cpu

            with state.lock:
                stat = state.stats[name]
                stat.wall_inclusive_s += inclusive_wall
                stat.wall_exclusive_s += exclusive_wall
                stat.cpu_inclusive_s += inclusive_cpu
                stat.cpu_exclusive_s += exclusive_cpu
                state.active_counts[name] -= 1

            if stack:
                parent = stack[-1]
                parent.child_wall += inclusive_wall
                parent.child_cpu += inclusive_cpu

    wrapper.__name__ = getattr(func, "__name__", name)
    wrapper.__doc__ = getattr(func, "__doc__", None)
    return wrapper


def load_module() -> ModuleType:
    """Import run_full_batch_pipeline.py as a real module so attribute patching
    is visible to its own global lookups — same technique as profile_render_video."""
    spec = importlib.util.spec_from_file_location(
        "profiled_run_full_batch_pipeline", str(SCRIPT_PATH)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to build import spec for {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def instrument_module(module: ModuleType, state: ProfilerState) -> None:
    for name in TARGET_FUNCTIONS:
        func = getattr(module, name, None)
        if callable(func):
            setattr(module, name, wrap_callable(state, name, func))


def find_report_json_path(pipeline_args: list[str]) -> Path | None:
    for i, arg in enumerate(pipeline_args):
        if arg == "--report-json" and i + 1 < len(pipeline_args):
            return Path(pipeline_args[i + 1])
    return None


def _stage_stats(items: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    vals = sorted(float(r[key]) for r in items if r.get(key) not in (None, ""))
    if not vals:
        return None
    n = len(vals)
    return {
        "n": n,
        "sum": round(sum(vals), 3),
        "mean": round(statistics.mean(vals), 3),
        "p50": round(vals[n // 2], 3),
        "p90": round(vals[min(int(n * 0.9), n - 1)], 3),
        "p95": round(vals[min(int(n * 0.95), n - 1)], 3),
        "max": round(max(vals), 3),
    }


def load_report_insights(report_json: Path) -> dict[str, Any] | None:
    if not report_json.exists():
        return None
    try:
        data = json.loads(report_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    items = data if isinstance(data, list) else []
    if not items:
        return None

    statuses: dict[str, int] = {}
    for it in items:
        s = str(it.get("status", "?"))
        statuses[s] = statuses.get(s, 0) + 1

    stage_keys = [
        "time_ass_s",
        "time_render_s",
        "time_render_cpu_s",
        "time_timeline_clean_ass_s",
        "time_image_picker_s",
        "time_tag_windows_s",
        "time_trim_s",
        "time_tsv_index_load_s",
    ]
    stages = {k: st for k in stage_keys if (st := _stage_stats(items, k))}

    ass_sub_keys = [
        "time_ass_script1_s",
        "time_ass_script2_s",
        "time_ass_worker_wait_s",
        "time_ass_model_load_s",
        "time_ass_audio_extract_s",
        "time_ass_transcribe_s",
        "time_ass_write_s",
    ]
    ass_detail = {k: st for k in ass_sub_keys if (st := _stage_stats(items, k))}

    failures = [
        {
            "item_id": r.get("ITM", r.get("item_id", "?")),
            "reason": str(r.get("failure_reason", "")),
        }
        for r in items if r.get("status") == "FAIL"
    ]
    reviews = [
        {
            "item_id": r.get("ITM", r.get("item_id", "?")),
            "reason": str(r.get("failure_reason", "")),
        }
        for r in items if r.get("status") == "REVIEW"
    ]

    per_item_keys = [
        "time_trim_s",
        "time_ass_s",
        "time_ass_audio_extract_s",
        "time_ass_transcribe_s",
        "time_ass_write_s",
        "time_ass_worker_wait_s",
        "time_timeline_clean_ass_s",
        "time_image_picker_s",
        "time_tag_windows_s",
        "time_render_s",
        "time_render_cpu_s",
        "time_render_cpu_pct",
    ]
    per_item = [
        {
            "item_id": r.get("ITM", r.get("item_id", "?")),
            "status": r.get("status", "?"),
            **{k: float(r[k]) if r.get(k) not in (None, "") else None for k in per_item_keys},
        }
        for r in items
    ]

    return {
        "total": len(items),
        "statuses": statuses,
        "stages": stages,
        "ass_detail": ass_detail,
        "per_item": per_item,
        "failures": failures,
        "reviews": reviews,
    }


def to_report_dict(
    state: ProfilerState,
    argv: list[str],
    started: float,
    finished: float,
    report_insights: dict[str, Any] | None,
) -> dict[str, Any]:
    stats_rows = []
    for name, stat in sorted(
        state.stats.items(),
        key=lambda item: (item[1].wall_inclusive_s, item[1].cpu_inclusive_s),
        reverse=True,
    ):
        row = asdict(stat)
        row["name"] = name
        stats_rows.append(row)

    peak_thread_count = max((int(s["thread_count"]) for s in state.thread_samples), default=1)

    def _pct_stats(vals: list[float]) -> dict[str, float] | None:
        if not vals:
            return None
        s = sorted(vals)
        n = len(s)
        return {
            "p50": round(s[n // 2], 2),
            "p90": round(s[min(int(n * 0.9), n - 1)], 2),
            "peak": round(s[-1], 2),
        }

    sys_vals = [s["system_cpu_pct"] for s in state.thread_samples if s.get("system_cpu_pct") is not None]
    proc_vals = [s["process_cpu_pct"] for s in state.thread_samples if s.get("process_cpu_pct") is not None]

    return {
        "argv": argv,
        "wall_total_s": finished - started,
        "peak_thread_count": peak_thread_count,
        "cpu_system": _pct_stats(sys_vals),
        "cpu_process": _pct_stats(proc_vals),
        "thread_samples": state.thread_samples,
        "functions": stats_rows,
        "report_insights": report_insights,
    }


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total > 0 else "n/a"


def write_summary(report: dict[str, Any], summary_path: Path) -> None:
    wall = float(report["wall_total_s"])
    peak = int(report["peak_thread_count"])
    sys_cpu = report.get("cpu_system") or {}
    proc_cpu = report.get("cpu_process") or {}
    cpu_sys_line = (
        f"  CPU system   p50={sys_cpu.get('p50', 0.0):5.1f}%  p90={sys_cpu.get('p90', 0.0):5.1f}%  peak={sys_cpu.get('peak', 0.0):5.1f}%"
        if sys_cpu else ""
    )
    cpu_proc_line = (
        f"  CPU process  p50={proc_cpu.get('p50', 0.0):5.1f}%  p90={proc_cpu.get('p90', 0.0):5.1f}%  peak={proc_cpu.get('peak', 0.0):5.1f}%  (Python pid only, not children)"
        if proc_cpu else ""
    )
    lines = [
        "Profile summary for run_full_batch_pipeline.py",
        f"Wall total: {wall:.2f}s   Peak threads: {peak}",
    ]
    if cpu_sys_line:
        lines.append(cpu_sys_line)
    if cpu_proc_line:
        lines.append(cpu_proc_line)
    lines.append("")

    insights = report.get("report_insights")
    if insights:
        total = int(insights.get("total", 0))
        statuses = insights.get("statuses", {})
        lines.append(f"Outcomes ({total} items)")
        for status in ("PASS", "REVIEW", "FAIL", "SKIPPED"):
            n = statuses.get(status, 0)
            if n:
                lines.append(f"  {status:8s} {n:4d}  ({_pct(n, total)})")
        lines.append("")

        stages = insights.get("stages", {})
        if stages:
            lines.append("Stage timing (wall seconds, per item, sorted by sum):")
            lines.append(
                f"  {'stage':30s} {'n':>4s} {'sum':>9s} {'mean':>8s} "
                f"{'p50':>8s} {'p90':>8s} {'max':>8s}"
            )
            for key, st in sorted(stages.items(), key=lambda kv: kv[1]["sum"], reverse=True):
                label = key.removeprefix("time_").removesuffix("_s")
                lines.append(
                    f"  {label:30s} {st['n']:>4d} {st['sum']:>9.1f} "
                    f"{st['mean']:>8.2f} {st['p50']:>8.2f} {st['p90']:>8.2f} {st['max']:>8.2f}"
                )
            lines.append("")

        ass_detail = insights.get("ass_detail", {})
        if ass_detail:
            lines.append("ASS sub-timings (mean seconds per item, sorted by mean):")
            for key, st in sorted(ass_detail.items(), key=lambda kv: kv[1]["mean"], reverse=True):
                label = key.removeprefix("time_").removesuffix("_s")
                lines.append(
                    f"  {label:30s}  mean={st['mean']:7.2f}s  "
                    f"p50={st['p50']:7.2f}s  p90={st['p90']:7.2f}s  max={st['max']:7.2f}s"
                )
            lines.append("")

        per_item = insights.get("per_item", [])
        if per_item:
            col_keys = [
                ("trim",    "time_trim_s"),
                ("ass",     "time_ass_s"),
                ("extract", "time_ass_audio_extract_s"),
                ("align",   "time_ass_transcribe_s"),
                ("ass_wrt", "time_ass_write_s"),
                ("wkr_wt",  "time_ass_worker_wait_s"),
                ("tl+cln",  "time_timeline_clean_ass_s"),
                ("img_api", "time_image_picker_s"),
                ("tags",    "time_tag_windows_s"),
                ("render",  "time_render_s"),
                ("rnd_cpu", "time_render_cpu_s"),
                ("cpu%",    "time_render_cpu_pct"),
            ]
            hdr = f"  {'ITM':24s} {'st':4s}"
            for label, _ in col_keys:
                hdr += f" {label:>7s}"
            lines.append(f"Per-item step timings (seconds)  [n={len(per_item)}]:")
            lines.append(hdr)
            for row in per_item:
                st = str(row.get("status", "?"))[:4]
                line = f"  {str(row['item_id']):24s} {st:4s}"
                for _, k in col_keys:
                    v = row.get(k)
                    line += f" {v:7.1f}" if v is not None else f" {'—':>7s}"
                lines.append(line)
            lines.append("")

        failures = insights.get("failures", [])
        if failures:
            lines.append(f"Failures ({len(failures)}):")
            for f in failures:
                lines.append(f"  {f['item_id']:24s}  {f['reason'][:120]}")
            lines.append("")

        reviews = insights.get("reviews", [])
        if reviews:
            reasons: dict[str, int] = {}
            for r in reviews:
                for part in r["reason"].split(";"):
                    part = part.strip()
                    if part:
                        reasons[part] = reasons.get(part, 0) + 1
            lines.append(f"Review reasons ({len(reviews)} items, distinct reasons):")
            for reason, cnt in sorted(reasons.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {cnt:3d}×  {reason[:120]}")
            lines.append("")

    funcs = report.get("functions", [])
    if funcs:
        lines.append(f"Top Python functions by inclusive wall time ({len(funcs)} captured):")
        lines.append(
            f"  {'name':36s} {'wall_incl':>10s} {'wall_excl':>10s} "
            f"{'cpu_incl':>10s} {'calls':>6s} {'max_conc':>8s}"
        )
        for row in funcs[:20]:
            lines.append(
                f"  {row['name']:36s} "
                f"{row['wall_inclusive_s']:10.2f} {row['wall_exclusive_s']:10.2f} "
                f"{row['cpu_inclusive_s']:10.2f} {row['call_count']:6d} "
                f"{row['max_concurrent_calls']:8d}"
            )
    else:
        lines.append(
            "(No Python function timing captured — module instrumentation may have failed)"
        )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("outputs/run_full_batch_pipeline_profile.json"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("outputs/run_full_batch_pipeline_profile.txt"),
    )
    parser.add_argument(
        "--thread-sample-interval",
        type=float,
        default=0.1,
    )
    parser.add_argument("pipeline_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.pipeline_args and args.pipeline_args[0] == "--":
        args.pipeline_args = args.pipeline_args[1:]
    return args


def main() -> int:
    args = parse_args()
    if not args.pipeline_args:
        print("Pass pipeline arguments after --", file=sys.stderr)
        return 2

    state = ProfilerState()
    module = load_module()
    instrument_module(module, state)

    report_json_path = find_report_json_path(args.pipeline_args)

    original_argv = sys.argv[:]
    sys.argv = [str(SCRIPT_PATH), *args.pipeline_args]
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    state.start_thread_sampler(args.thread_sample_interval)
    exit_code = 0
    try:
        exit_code = int(module.main())
    finally:
        state.stop_thread_sampler()
        finished = time.perf_counter()

        report_insights = load_report_insights(report_json_path) if report_json_path else None
        report = to_report_dict(state, sys.argv[:], started, finished, report_insights)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        write_summary(report, args.summary_out)
        sys.argv = original_argv

    print(f"Wrote JSON profile to {args.json_out}", flush=True)
    print(f"Wrote text summary to {args.summary_out}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
