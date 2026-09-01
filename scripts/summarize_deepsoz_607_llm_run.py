#!/usr/bin/env python3
"""Summarize live/final LLM coverage for the 607-record DeepSOZ mapping."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COMPLETED_LLM_STATUSES = {
    "llm_candidate_ready",
    "llm_abstained",
    "llm_context_exhausted",
}


def _failure_category(error: object) -> str:
    text = str(error or "unknown_failure").strip()
    return text.split(":", 1)[0] or "unknown_failure"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "outputs/deepsoz_607_llm_batch_20260801/tusz_deepsoz_607_events_manifest.csv",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "outputs/deepsoz_607_llm_qwen36_full_v3_20260801",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig")
    records: dict[str, tuple[Path, dict]] = {}
    for path in args.run_root.glob("shard_*/records/*.json"):
        if path.name.endswith(".core_checkpoint.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        event_id = str(payload.get("event_id") or "")
        previous = records.get(event_id)
        if event_id and (previous is None or path.stat().st_mtime > previous[0].stat().st_mtime):
            records[event_id] = (path, payload)

    rows: list[dict] = []
    failure_categories: Counter[str] = Counter()
    for _, source in manifest.iterrows():
        event_id = str(source.event_id)
        saved = records.get(event_id)
        if saved is None:
            rows.append(
                {
                    "event_id": event_id,
                    "patient_id": source.patient_id,
                    "edf_path": source.edf_path,
                    "event_index": source.event_index,
                    "sz_start": source.sz_start,
                    "sz_end": source.sz_end,
                    "execution_state": "queued_or_active",
                    "record_status": "",
                    "llm_soz_channels": "",
                    "processing_error": "",
                    "record_path": "",
                }
            )
            continue

        path, payload = saved
        status = str(payload.get("record_status") or "unknown")
        error = str(payload.get("processing_error") or "")
        if status in COMPLETED_LLM_STATUSES:
            execution_state = "llm_completed"
        elif status == "processing_failed_closed":
            execution_state = "llm_failed_closed"
            failure_categories[_failure_category(error)] += 1
        elif status == "packet_only_llm_not_called":
            execution_state = "packet_only_llm_not_called"
        else:
            execution_state = "other_record_status"
        llm_result = payload.get("llm_result") or {}
        rows.append(
            {
                "event_id": event_id,
                "patient_id": source.patient_id,
                "edf_path": source.edf_path,
                "event_index": source.event_index,
                "sz_start": source.sz_start,
                "sz_end": source.sz_end,
                "execution_state": execution_state,
                "record_status": status,
                "llm_soz_channels": "|".join(llm_result.get("soz_channels") or []),
                "processing_error": error,
                "record_path": str(path.resolve()),
            }
        )

    audit = pd.DataFrame(rows)
    execution_counts = audit.execution_state.value_counts().to_dict()
    status_counts = (
        audit.loc[audit.record_status.astype(bool), "record_status"]
        .value_counts()
        .to_dict()
    )
    progress = []
    for path in sorted(args.run_root.glob("shard_*/progress.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        progress.append({"shard": path.parent.name, **payload})

    summary = {
        "mapped_recordings": int(manifest.edf_path.nunique()),
        "target_events": len(manifest),
        "saved_event_records": len(records),
        "execution_counts": execution_counts,
        "record_status_counts": status_counts,
        "failure_category_counts": dict(failure_categories),
        "all_events_have_completed_llm_call": execution_counts.get("llm_completed", 0)
        == len(manifest),
        "shard_progress": progress,
        "semantics": {
            "llm_abstained": "LLM ran successfully but declined to localize an SOZ candidate",
            "processing_failed_closed": "LLM/pipeline ran but strict validation or processing failed",
            "queued_or_active": "the launched batch has not yet written a final event record",
        },
    }
    args.run_root.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.run_root / "live_coverage.csv", index=False, encoding="utf-8-sig")
    (args.run_root / "live_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
