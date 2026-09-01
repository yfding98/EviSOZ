#!/usr/bin/env python3
"""Recover a validated Qwen visual-core checkpoint from a fail-closed record.

This utility performs no model inference and invents no clinical fields. It
revalidates saved, image-grounded core JSON with the current strict validator
and writes a checkpoint only when that response now passes. Training export
remains disabled; the checkpoint is only an auditable seed for regenerating
the professional narrative with the same local Qwen release.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.auto_annotate.adaptive_review import AdaptiveReviewWindow  # noqa: E402
from code.auto_annotate.llm_soz_annotator import (  # noqa: E402
    SUPPORTED_LOCAL_QWEN_RELEASES,
    parse_json_object,
    validate_local_core_result,
)
from code.auto_annotate.nearby_onset_autolabel import sha256_file  # noqa: E402
from code.auto_annotate.recover_local_qwen_two_stage import (  # noqa: E402
    _task_from_record,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


def recover_core_checkpoint(record_path: Path, output_path: Path) -> dict[str, Any]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("training_export_allowed") is not False:
        raise ValueError("source record must preserve training_export_allowed=false")
    if record.get("record_status") != "processing_failed_closed":
        raise ValueError("source record is not a fail-closed processing record")

    task = _task_from_record(record)
    packet = record.get("eeg_packet") or {}
    available_channels = list(packet.get("available_channels") or [])
    review_windows = [
        AdaptiveReviewWindow(**item)
        for item in packet.get("initial_review_windows") or []
    ]
    source_ids = list(
        (record.get("knowledge_context") or {}).get("retrieved_source_ids") or []
    )
    if not available_channels or not review_windows or not source_ids:
        raise ValueError("source record lacks rendered channels, windows, or knowledge IDs")

    failures = [
        dict(item)
        for item in record.get("failed_generation_attempts") or []
        if isinstance(item, Mapping) and item.get("stage") == "core_localization"
    ]
    if not failures:
        raise ValueError("source record has no saved core localization response")

    selected: dict[str, Any] | None = None
    core_result: dict[str, Any] | None = None
    warnings: list[str] = []
    last_error: Exception | None = None
    for failure in reversed(failures):
        metadata = failure.get("response_metadata") or {}
        release = str(metadata.get("model_release") or "")
        if release not in SUPPORTED_LOCAL_QWEN_RELEASES:
            continue
        try:
            payload = parse_json_object(str(failure.get("raw_response") or ""))
            core_result, warnings = validate_local_core_result(
                payload,
                task,
                available_channels,
                review_windows=review_windows,
                allowed_knowledge_source_ids=source_ids,
            )
            selected = failure
            break
        except ValueError as exc:
            last_error = exc
    if selected is None or core_result is None:
        raise ValueError(f"no saved core response passes current validation: {last_error}")

    metadata = dict(selected.get("response_metadata") or {})
    metadata.update(
        {
            "recovered_from_fail_closed_core_response": True,
            "recovery_source_sha256": sha256_file(record_path),
            "local_conservative_normalizations": [
                item for item in warnings if item.startswith("local_")
            ],
        }
    )
    raw_response = str(selected.get("raw_response") or "")
    checkpoint = {
        "schema_version": "local_qwen_core_checkpoint_v1",
        "identity": "recovered-fail-closed:" + sha256_file(record_path),
        "core_result": core_result,
        "response_metadata": metadata,
        "raw_response": raw_response,
        "failed_validation_attempts": failures,
        "recovery_audit": {
            "source_record": str(record_path),
            "source_event_id": record.get("event_id"),
            "selected_attempt": selected.get("attempt"),
            "validation_warnings": warnings,
            "training_export_allowed": False,
            "model_reexecuted": False,
        },
    }
    _atomic_json(output_path, checkpoint)
    return checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record_path = args.record.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {output_path}")
    checkpoint = recover_core_checkpoint(record_path, output_path)
    print(
        json.dumps(
            {
                "checkpoint": str(output_path),
                "model_release": checkpoint["response_metadata"].get("model_release"),
                "event_id": checkpoint["recovery_audit"]["source_event_id"],
                "selected_attempt": checkpoint["recovery_audit"]["selected_attempt"],
                "normalizations": checkpoint["response_metadata"].get(
                    "local_conservative_normalizations", []
                ),
                "training_export_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
