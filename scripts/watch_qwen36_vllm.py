#!/usr/bin/env python3
"""Show Qwen3.6 vLLM queue concurrency, KV usage and prefix-cache hit rate."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Sequence


SAMPLE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[-+0-9.eE]+)$"
)


def _metrics_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        cleaned = cleaned[:-3]
    return cleaned.rstrip("/") + "/metrics"


def _fetch_metrics(base_url: str, timeout_s: float) -> dict[str, float]:
    with urllib.request.urlopen(
        _metrics_url(base_url), timeout=timeout_s
    ) as response:
        text = response.read().decode("utf-8", errors="replace")
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        match = SAMPLE_RE.match(line.strip())
        if not match:
            continue
        name = match.group("name")
        metrics[name] = metrics.get(name, 0.0) + float(match.group("value"))
    return metrics


def _metric(metrics: dict[str, float], *suffixes: str) -> float | None:
    for suffix in suffixes:
        matches = [
            value
            for name, value in metrics.items()
            if name == suffix or name.endswith(":" + suffix)
        ]
        if matches:
            return sum(matches)
    return None


def _read_progress(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("*/progress.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") == "qwen36_vllm_queue_progress_v1":
            rows.append({"dataset": path.parent.name, **payload})
    return rows


def render(run_root: Path, base_url: str, timeout_s: float) -> str:
    progress = _read_progress(run_root)
    metrics = _fetch_metrics(base_url, timeout_s)
    running = _metric(metrics, "num_requests_running") or 0.0
    waiting = _metric(metrics, "num_requests_waiting") or 0.0
    kv_usage = _metric(
        metrics, "gpu_cache_usage_perc", "kv_cache_usage_perc"
    )
    hits = _metric(
        metrics, "prefix_cache_hits_total", "prefix_cache_hits"
    )
    queries = _metric(
        metrics, "prefix_cache_queries_total", "prefix_cache_queries"
    )
    hit_rate = (
        100.0 * hits / queries
        if hits is not None and queries is not None and queries > 0
        else None
    )
    lines = [
        f"Qwen3.6 vLLM  running={running:.0f} waiting={waiting:.0f}",
        (
            f"KV cache usage={kv_usage * 100:.1f}%"
            if kv_usage is not None and kv_usage <= 1.0
            else (
                f"KV cache usage={kv_usage:.1f}%"
                if kv_usage is not None
                else "KV cache usage=metric unavailable"
            )
        ),
        (
            f"prefix cache hits={hits:.0f}/{queries:.0f} ({hit_rate:.1f}%)"
            if hit_rate is not None
            else "prefix cache hit rate=metric unavailable until requests run"
        ),
        "",
    ]
    for row in progress:
        lines.append(
            f"{row['dataset']}: {row.get('completed_events', 0)}/"
            f"{row.get('selected_events', '?')} completed, "
            f"failed={row.get('failed_events', 0)}, "
            f"active={row.get('active_event_ids') or []}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000/v1"
    )
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = Path(args.run_root).expanduser().resolve()
    try:
        while True:
            print(render(run_root, args.base_url, args.timeout_s), flush=True)
            if args.once:
                return 0
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
