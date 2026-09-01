#!/usr/bin/env python3
"""Display live progress for scripts/run_qwen35_all_data.sh."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence


PHASES = (
    ("private", "私有数据"),
    ("tusz", "TUSZ"),
    ("chbmit", "CHB-MIT"),
    ("tuev_scan", "TUEV 初筛"),
    ("tuev", "TUEV Qwen"),
    ("tuep_scan", "TUEP 初筛"),
    ("tuep", "TUEP Qwen"),
)
FINAL_SUCCESS = {
    "llm_candidate_ready",
    "llm_abstained",
    "llm_context_exhausted",
}
COMPLETION_RE = re.compile(
    r"^\[(?P<index>\d+)/(?P<total>\d+)\]\s+"
    r"(?P<event>\S+)\s+(?P<status>\S+)",
    re.MULTILINE,
)
STAGE_RE = re.compile(r"^\s*(Qwen stage .+)$", re.MULTILINE)
SCANNER_RE = re.compile(
    r"^\[(?P<index>\d+)/(?P<total>\d+)\]\s+(?P<event>\S+)(?:\s+(?P<status>\S+))?",
    re.MULTILINE,
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _tail_text(path: Path, limit: int = 512 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _record_counts(records_dir: Path) -> tuple[int, int, int]:
    successes = 0
    failures = 0
    other = 0
    if not records_dir.is_dir():
        return successes, failures, other
    for path in records_dir.glob("*.json"):
        if path.name.endswith(".core_checkpoint.json"):
            continue
        payload = _read_json(path)
        status = str(payload.get("record_status") or "")
        if status in FINAL_SUCCESS:
            successes += 1
        elif status == "processing_failed_closed":
            failures += 1
        elif status:
            other += 1
    return successes, failures, other


def _active_evidence_event(run_root: Path, phase: str) -> str:
    evidence_root = run_root / phase / "evidence"
    records_root = run_root / phase / "records"
    if not evidence_root.is_dir():
        return ""
    unfinished: list[Path] = []
    for directory in evidence_root.iterdir():
        if not directory.is_dir():
            continue
        if not (records_root / f"{directory.name}.json").is_file():
            unfinished.append(directory)
    if not unfinished:
        return ""
    newest = max(unfinished, key=lambda path: path.stat().st_mtime_ns)
    return newest.name


def _summary_total(run_root: Path, phase: str) -> tuple[int | None, bool]:
    summary_path = run_root / phase / "summary.json"
    payload = _read_json(summary_path)
    total = payload.get("selected_events")
    return (
        int(total) if isinstance(total, (int, float)) else None,
        bool(payload),
    )


def phase_snapshot(
    run_root: Path,
    phase: str,
    label: str,
    *,
    stale_after_s: float = 1800.0,
) -> dict[str, Any]:
    log_path = run_root / "logs" / f"{phase}.log"
    text = _tail_text(log_path)
    summary_total, complete = _summary_total(run_root, phase)
    records_phase = phase.removesuffix("_scan")
    records_dir = run_root / records_phase / "records"
    if phase.endswith("_scan"):
        records_dir = run_root / phase / "records"
    successes, failures, other = _record_counts(records_dir)
    progress_path = run_root / records_phase / "progress.json"
    progress = (
        _read_json(progress_path) if not phase.endswith("_scan") else {}
    )
    has_live_progress = (
        progress.get("schema_version") == "qwen35_soz_live_progress_v1"
    )
    if has_live_progress:
        successes = int(progress.get("successful_events") or 0)
        failures = int(progress.get("failed_events") or 0)
        other = int(progress.get("other_finalized_events") or 0)

    matches = list(COMPLETION_RE.finditer(text))
    if not matches:
        matches = list(SCANNER_RE.finditer(text))
    log_index = int(matches[-1].group("index")) if matches else 0
    log_total = int(matches[-1].group("total")) if matches else None
    total = (
        int(progress.get("selected_events") or 0)
        if has_live_progress
        else (summary_total or log_total)
    )
    total = total or None
    finalized = (
        int(progress.get("finalized_events") or 0)
        if has_live_progress
        else max(successes + failures + other, log_index)
    )

    stages = list(STAGE_RE.finditer(text))
    stage = stages[-1].group(1).strip() if stages else ""
    last_event = (
        str(progress.get("last_event_id") or "")
        if has_live_progress
        else (matches[-1].group("event") if matches else "")
    )
    active_event = (
        str(progress.get("active_event_id") or "")
        if has_live_progress
        else _active_evidence_event(run_root, records_phase)
    )
    freshness_paths = [
        path for path in (log_path, progress_path) if path.is_file()
    ]
    log_age_s = (
        max(
            0.0,
            time.time()
            - max(path.stat().st_mtime for path in freshness_paths),
        )
        if freshness_paths
        else None
    )
    progress_state = str(progress.get("state") or "")
    if has_live_progress and progress_state in {
        "completed",
        "completed_with_failures",
    }:
        state = "已完成"
    elif has_live_progress and progress_state == "aborted":
        state = "已中止"
    elif not has_live_progress and complete:
        state = "已完成"
    elif log_path.is_file() and log_age_s is not None and log_age_s > stale_after_s:
        state = "可能停滞"
    elif log_path.is_file() or has_live_progress:
        state = "运行中"
    else:
        state = "等待"
    if state in {"运行中", "可能停滞"} and stage:
        active_index = (
            int(progress.get("active_position") or 0)
            if has_live_progress
            else 0
        )
        if active_index <= 0:
            active_index = finalized + 1 if total and finalized < total else finalized
        event_text = f" {active_event}" if active_event else ""
        active = f"{active_index}/{total or '?'}{event_text} | {stage}"
    elif state in {"运行中", "可能停滞"} and active_event:
        active_index = int(progress.get("active_position") or 0)
        active = f"{active_index or finalized + 1}/{total or '?'} {active_event}"
    elif last_event:
        active = f"最近：{last_event}"
    else:
        active = ""
    return {
        "phase": phase,
        "label": label,
        "state": state,
        "successes": successes,
        "failures": failures,
        "other": other,
        "finalized": finalized,
        "total": total,
        "active": active,
        "log_path": str(log_path),
        "log_age_s": log_age_s,
    }


def _bar(done: int, total: int | None, width: int = 28) -> str:
    if not total or total <= 0:
        return "[" + "·" * width + "]   ?.?%"
    ratio = min(1.0, max(0.0, done / total))
    filled = int(round(width * ratio))
    return (
        "["
        + "█" * filled
        + "░" * (width - filled)
        + f"] {ratio * 100:5.1f}%"
    )


def render(run_root: Path, *, stale_after_s: float = 1800.0) -> str:
    snapshots = [
        phase_snapshot(
            run_root, phase, label, stale_after_s=stale_after_s
        )
        for phase, label in PHASES
    ]
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        f"Qwen3.5 SOZ 全量进度  {now}",
        f"运行目录：{run_root}",
        "",
    ]
    for item in snapshots:
        total_text = str(item["total"]) if item["total"] else "?"
        lines.append(
            f"{item['label']:<11} {_bar(item['finalized'], item['total'])} "
            f"{item['finalized']}/{total_text}  "
            f"成功={item['successes']} 失败={item['failures']} "
            f"状态={item['state']}"
        )
        if item["active"]:
            lines.append(f"  └─ {item['active']}")
        if item["state"] == "可能停滞" and item["log_age_s"] is not None:
            lines.append(
                f"  └─ 日志已 {item['log_age_s'] / 60:.0f} 分钟没有更新；"
                "请确认原命令是否仍占用终端/GPU"
            )
    successes = sum(item["successes"] for item in snapshots)
    failures = sum(item["failures"] for item in snapshots)
    lines.extend(
        [
            "",
            f"当前已落盘：成功 {successes}，失败 {failures}",
            "说明：完成数按最终事件 JSON 统计；core checkpoint 不计为完成事件。",
        ]
    )
    return "\n".join(lines)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--watch-pid", type=int)
    parser.add_argument("--stale-after-s", type=float, default=1800.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = Path(args.run_root).expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"run root not found: {run_root}")
    interactive = sys.stdout.isatty()
    interval = max(0.5, float(args.interval))
    if not interactive:
        interval = max(60.0, interval)
    try:
        while True:
            snapshot = render(run_root, stale_after_s=float(args.stale_after_s))
            if interactive:
                sys.stdout.write("\033[2J\033[H" + snapshot + "\n")
            else:
                sys.stdout.write(snapshot + "\n\n")
            sys.stdout.flush()
            if args.once:
                break
            if args.watch_pid is not None and not _pid_alive(args.watch_pid):
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
