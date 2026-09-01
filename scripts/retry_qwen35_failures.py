#!/usr/bin/env python3
"""Retry unresolved Qwen3.5 SOZ failures listed by index_qwen35_results.py."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.index_qwen35_results import build_index  # noqa: E402


DATASET_ROOTS = {
    "chbmit": Path("/mnt/hd1/dyf/dataset/CHB-MIT"),
    "tuev": Path("/mnt/hd1/dyf/dataset/tuh_eeg_events"),
    "tuep": Path("/mnt/hd1/dyf/dataset/tuh_eeg_epilepsy"),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(payload)
    return rows


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def _rtk_path() -> str:
    value = shutil.which("rtk")
    if not value:
        raise FileNotFoundError("rtk is required but was not found on PATH")
    return value


def _generic_root(dataset: str, row: Mapping[str, Any]) -> Path:
    configured = DATASET_ROOTS.get(dataset.lower())
    if configured and configured.is_dir():
        return configured
    source_file = Path(str(row.get("source_file") or "")).expanduser()
    if source_file.is_file():
        return source_file.parent.resolve()
    raise FileNotFoundError(
        f"cannot resolve EEG root for generic dataset={dataset!r}"
    )


def build_retry_command(
    row: Mapping[str, Any],
    *,
    retry_output_dir: Path,
    local_validation_retries: int,
) -> list[str]:
    record_path = Path(str(row.get("record_path") or "")).expanduser().resolve()
    if not record_path.is_file():
        raise FileNotFoundError(f"failure record not found: {record_path}")
    event_id = str(row.get("event_id") or "").strip()
    dataset = str(row.get("dataset") or "").strip()
    runner_dataset = str(row.get("runner_dataset") or "").strip()
    if not event_id or not dataset:
        raise ValueError("failure row requires event_id and dataset")
    if runner_dataset not in {"private", "tusz", "generic"}:
        runner_dataset = dataset if dataset in {"private", "tusz"} else "generic"

    command = [
        _rtk_path(),
        str(ROOT / ".venv-qwen35" / "bin" / "python"),
        "-u",
        str((ROOT / "scripts" / "run_qwen35_soz_annotation.py").resolve()),
        "--dataset",
        runner_dataset,
    ]
    if runner_dataset == "generic":
        manifest = Path(str(row.get("manifest_path") or "")).expanduser().resolve()
        if not manifest.is_file():
            raise FileNotFoundError(
                f"original generic manifest is unavailable: {manifest}"
            )
        anchor_fields = [
            str(item).strip()
            for item in row.get("generic_anchor_fields") or []
            if str(item).strip()
        ]
        command.extend(
            [
                "--dataset-name",
                dataset,
                "--manifest",
                str(manifest),
                "--eeg-root",
                str(_generic_root(dataset, row)),
                "--generic-anchor-fields",
                ",".join(anchor_fields or ["coarse_candidate_s"]),
            ]
        )

    command.extend(
        [
            "--event-id",
            event_id,
            "--max-events",
            "1",
            "--max-review-rounds",
            "1",
            "--local-validation-retries",
            str(local_validation_retries),
            "--local-narrative-max-tokens",
            "2400",
            "--output-dir",
            str(retry_output_dir.resolve()),
            "--resume",
            "--fail-fast",
        ]
    )
    for option, field in (
        ("--context-pre-s", "context_pre_s"),
        ("--context-post-s", "context_post_s"),
        ("--search-pre-s", "search_pre_s"),
        ("--search-post-s", "search_post_s"),
    ):
        value = row.get(field)
        if value is not None:
            command.extend([option, str(value)])

    checkpoint_text = str(row.get("core_checkpoint_path") or "").strip()
    checkpoint = Path(checkpoint_text).expanduser().resolve() if checkpoint_text else None
    if checkpoint is not None and checkpoint.is_file():
        command.extend(
            [
                "--reuse-core-checkpoint",
                str(checkpoint),
                "--retry-narrative-from-record",
                str(record_path),
            ]
        )
    else:
        command.extend(["--retry-core-from-failure-record", str(record_path)])
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-events", type=int, default=-1)
    parser.add_argument("--local-validation-retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    failure_path = Path(args.failures).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    rows = _read_jsonl(failure_path)
    if args.max_events >= 0:
        rows = rows[: args.max_events]
    if not rows:
        print("No unresolved failures to retry.")
        return 0

    plans: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows, start=1):
        dataset = str(row.get("dataset") or "unknown")
        event_id = str(row.get("event_id") or f"event-{ordinal}")
        safe_event = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in event_id
        )
        destination = output_root / f"{ordinal:04d}_{dataset}_{safe_event}"
        log_path = output_root / "logs" / f"{ordinal:04d}_{dataset}_{safe_event}.log"
        planning_error = ""
        try:
            command = build_retry_command(
                row,
                retry_output_dir=destination,
                local_validation_retries=args.local_validation_retries,
            )
        except Exception as exc:
            command = []
            planning_error = f"{type(exc).__name__}: {exc}"
        plans.append(
            {
                "dataset": dataset,
                "event_id": event_id,
                "source_failure_record": row.get("record_path"),
                "retry_strategy": row.get("retry_strategy"),
                "output_dir": str(destination.resolve()),
                "log_path": str(log_path.resolve()),
                "command": command,
                "returncode": None,
                "status": (
                    "planning_failed" if planning_error else "planned"
                ),
                "planning_error": planning_error,
                "execution_error": "",
            }
        )

    _atomic_jsonl(output_root / "retry_plan.jsonl", plans)
    if args.dry_run:
        print(json.dumps(plans, ensure_ascii=False, indent=2))
        return 1 if any(plan["planning_error"] for plan in plans) else 0

    output_root.mkdir(parents=True, exist_ok=True)
    for plan in plans:
        if plan["planning_error"]:
            _atomic_jsonl(output_root / "retry_attempts.jsonl", plans)
            continue
        log_path = Path(plan["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        plan["status"] = "running"
        _atomic_jsonl(output_root / "retry_attempts.jsonl", plans)
        try:
            with log_path.open("w", encoding="utf-8") as log_handle:
                completed = subprocess.run(
                    plan["command"],
                    cwd=ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            plan["returncode"] = int(completed.returncode)
            plan["status"] = (
                "completed"
                if completed.returncode == 0
                else "retry_failed"
            )
        except Exception as exc:
            plan["returncode"] = 127
            plan["status"] = "execution_failed"
            plan["execution_error"] = f"{type(exc).__name__}: {exc}"
        _atomic_jsonl(output_root / "retry_attempts.jsonl", plans)

    original_roots = sorted(
        {
            str(Path(str(row["run_root"])).expanduser().resolve())
            for row in rows
            if row.get("run_root")
        }
    )
    index_summary = build_index(
        [*original_roots, output_root],
        output_root / "indexes",
    )
    print(json.dumps(index_summary, ensure_ascii=False, indent=2))
    return (
        1
        if any(
            plan["status"]
            in {"planning_failed", "retry_failed", "execution_failed"}
            for plan in plans
        )
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
