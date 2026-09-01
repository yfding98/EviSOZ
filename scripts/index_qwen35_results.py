#!/usr/bin/env python3
"""Build auditable success and unresolved-failure indexes for Qwen3.5 SOZ runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "qwen35_soz_result_index_v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []
    return rows


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    fieldnames = [
        "dataset",
        "runner_dataset",
        "event_id",
        "patient_id",
        "record_status",
        "candidate_status",
        "source_file",
        "navigation_anchor_s",
        "t0_s",
        "t_spread_s",
        "t_end_s",
        "soz_channels",
        "soz_regions",
        "spread_channels",
        "spread_regions",
        "processing_error",
        "failed_stages",
        "retry_strategy",
        "record_path",
        "core_checkpoint_path",
        "summary_path",
        "run_root",
        "training_export_allowed",
    ]
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            exported = {name: row.get(name) for name in fieldnames}
            for name in (
                "soz_channels",
                "soz_regions",
                "spread_channels",
                "spread_regions",
                "failed_stages",
            ):
                exported[name] = json.dumps(
                    exported.get(name) or [], ensure_ascii=False, separators=(",", ":")
                )
            writer.writerow(exported)
    temporary.replace(path)


def _summary_for_record(record_path: Path) -> tuple[Path | None, dict[str, Any]]:
    candidate = record_path.parent.parent / "summary.json"
    if not candidate.is_file():
        return None, {}
    try:
        return candidate, _read_json(candidate)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return candidate, {}


def _model_release(record: Mapping[str, Any]) -> str:
    metadata = record.get("llm_response_metadata")
    if isinstance(metadata, Mapping):
        visual = metadata.get("visual_stage")
        if isinstance(visual, Mapping) and visual.get("model_release"):
            return str(visual["model_release"])
    for attempt in reversed(record.get("failed_generation_attempts") or []):
        if not isinstance(attempt, Mapping):
            continue
        response_metadata = attempt.get("response_metadata")
        if isinstance(response_metadata, Mapping) and response_metadata.get(
            "model_release"
        ):
            return str(response_metadata["model_release"])
    return ""


def _record_row(record_path: Path, run_root: Path) -> dict[str, Any] | None:
    try:
        record = _read_json(record_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    record_status = str(record.get("record_status") or "")
    event_id = str(record.get("event_id") or "")
    if not record_status or not event_id:
        return None
    summary_path, summary = _summary_for_record(record_path)
    result = record.get("llm_result")
    if not isinstance(result, Mapping):
        result = {}
    dataset = str(record.get("dataset") or summary.get("dataset_name") or "unknown")
    runner_dataset = str(summary.get("dataset") or "")
    if runner_dataset not in {"private", "tusz", "generic"}:
        runner_dataset = dataset if dataset in {"private", "tusz"} else "generic"
    checkpoint = record_path.with_name(
        record_path.stem + ".core_checkpoint.json"
    )
    failed_stages = [
        str(item.get("stage"))
        for item in record.get("failed_generation_attempts") or []
        if isinstance(item, Mapping) and item.get("stage")
    ]
    retry_strategy = ""
    if record_status == "processing_failed_closed":
        retry_strategy = "narrative" if checkpoint.is_file() else "core"
    audit = summary.get("manifest_audit")
    if not isinstance(audit, Mapping):
        audit = {}
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "runner_dataset": runner_dataset,
        "event_id": event_id,
        "patient_id": str(record.get("patient_id") or ""),
        "record_status": record_status,
        "candidate_status": str(result.get("status") or ""),
        "source_file": str(record.get("source_file") or ""),
        "navigation_anchor_s": record.get("navigation_anchor_s"),
        "t0_s": result.get("t0_s"),
        "t_spread_s": result.get("t_spread_s"),
        "t_end_s": result.get("t_end_s"),
        "soz_channels": list(result.get("soz_channels") or []),
        "soz_regions": list(result.get("soz_regions") or []),
        "spread_channels": list(result.get("spread_channels") or []),
        "spread_regions": list(result.get("spread_regions") or []),
        "processing_error": str(record.get("processing_error") or ""),
        "failed_stages": failed_stages,
        "retry_strategy": retry_strategy,
        "record_path": str(record_path.resolve()),
        "core_checkpoint_path": (
            str(checkpoint.resolve()) if checkpoint.is_file() else ""
        ),
        "summary_path": str(summary_path.resolve()) if summary_path else "",
        "run_root": str(run_root.resolve()),
        "manifest_path": str(audit.get("manifest") or ""),
        "generic_anchor_fields": list(audit.get("anchor_field_precedence") or []),
        "context_pre_s": summary.get("context_pre_s"),
        "context_post_s": summary.get("context_post_s"),
        "search_pre_s": summary.get("search_pre_s"),
        "search_post_s": summary.get("search_post_s"),
        "model_release": _model_release(record),
        "source_spatial_labels_used": record.get("source_spatial_labels_used"),
        "autolabel_predictions_used": record.get("autolabel_predictions_used"),
        "training_export_allowed": record.get("training_export_allowed"),
        "record_mtime_ns": record_path.stat().st_mtime_ns,
    }


def discover_rows(run_roots: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for raw_root in run_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            continue
        for record_path in sorted(root.rglob("records/*.json")):
            if record_path.name.endswith(".core_checkpoint.json"):
                continue
            key = str(record_path.resolve())
            if key in seen_paths:
                continue
            seen_paths.add(key)
            row = _record_row(record_path, root)
            if row is not None:
                rows.append(row)
    return rows


def discover_failure_history(
    run_roots: Sequence[str | Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for raw_root in run_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            continue
        for history_path in sorted(root.rglob("failure_history.jsonl")):
            key = str(history_path.resolve())
            if key in seen_paths:
                continue
            seen_paths.add(key)
            rows.extend(_read_jsonl(history_path))
    return rows


def _preferred_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row.get("dataset")), str(row.get("event_id"))), []
        ).append(row)
    successes: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for key in sorted(grouped):
        candidates = grouped[key]
        valid = [
            row
            for row in candidates
            if row.get("record_status") in {
                "llm_candidate_ready",
                "llm_abstained",
                "llm_context_exhausted",
            }
        ]
        if valid:
            selected = max(valid, key=lambda row: int(row.get("record_mtime_ns") or 0))
            successes.append(dict(selected))
        else:
            failures = [
                row
                for row in candidates
                if row.get("record_status") == "processing_failed_closed"
            ]
            if failures:
                selected = max(
                    failures, key=lambda row: int(row.get("record_mtime_ns") or 0)
                )
                unresolved.append(dict(selected))
    return successes, unresolved


def build_index(
    run_roots: Sequence[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    rows = discover_rows(run_roots)
    successes, failures = _preferred_rows(rows)
    current_failure_history = [
        dict(row)
        for row in rows
        if row.get("record_status") == "processing_failed_closed"
    ]
    merged_failure_history = [
        *_read_jsonl(output / "failure_history.jsonl"),
        *discover_failure_history(run_roots),
        *current_failure_history,
    ]
    history_by_identity: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    for row in merged_failure_history:
        identity = (
            str(row.get("dataset") or ""),
            str(row.get("event_id") or ""),
            str(row.get("record_path") or ""),
            str(row.get("processing_error") or ""),
            int(row.get("record_mtime_ns") or 0),
        )
        history_by_identity[identity] = dict(row)
    failure_history = list(history_by_identity.values())
    for collection in (rows, successes, failures, failure_history):
        collection.sort(
            key=lambda row: (
                str(row.get("dataset")),
                str(row.get("event_id")),
                str(row.get("record_path")),
            )
        )
    _atomic_jsonl(output / "all_records.jsonl", rows)
    _atomic_jsonl(output / "successes.jsonl", successes)
    _atomic_jsonl(output / "failures.jsonl", failures)
    _atomic_jsonl(output / "failure_history.jsonl", failure_history)
    _atomic_csv(output / "successes.csv", successes)
    _atomic_csv(output / "failures.csv", failures)
    per_dataset: dict[str, dict[str, int]] = {}
    for row in successes:
        per_dataset.setdefault(str(row["dataset"]), {"successes": 0, "failures": 0})[
            "successes"
        ] += 1
    for row in failures:
        per_dataset.setdefault(str(row["dataset"]), {"successes": 0, "failures": 0})[
            "failures"
        ] += 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_roots": [str(Path(root).expanduser().resolve()) for root in run_roots],
        "record_versions": len(rows),
        "successful_events": len(successes),
        "unresolved_failed_events": len(failures),
        "failure_history_records": len(failure_history),
        "per_dataset": per_dataset,
        "success_index_jsonl": str((output / "successes.jsonl").resolve()),
        "success_index_csv": str((output / "successes.csv").resolve()),
        "failure_index_jsonl": str((output / "failures.jsonl").resolve()),
        "failure_index_csv": str((output / "failures.csv").resolve()),
        "primary_results": "Each index row points to the immutable record_path JSON.",
        "training_export_allowed": False,
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        action="append",
        required=True,
        help="Run directory to search recursively; repeat for multiple runs.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_index(args.run_root, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
