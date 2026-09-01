#!/usr/bin/env python3
"""Run and freeze the source-dev common-17 adaptive/fixed support ablation.

Extraction opens only the frozen navigation-onset manifest and raw EDF signal.
The TERM onset/offset reference file is opened only after every event receipt
and the target-blind cohort receipt have been written and content-addressed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.common17_support_policy_comparator_v1 import (  # noqa: E402
    evaluate_common17_support_policy_postfreeze_references_v1,
)
from src.clinical_eeg_long_recording.tusz_real_edf_support_comparison_v1 import (  # noqa: E402
    load_tusz_real_edf_support_comparison_manifest_v1,
    materialize_tusz_real_edf_support_comparison_entry_v1,
    summarize_tusz_real_edf_support_comparison_cohort_v1,
    validate_tusz_real_edf_support_comparison_event_v1,
)


DEFAULT_MANIFEST = (
    ROOT
    / "outputs/tusz_common17_support_comparison_source_dev259_manifest_v1_20260825"
    / "extraction_manifest.json"
)
DEFAULT_POSTFREEZE_REFERENCES = DEFAULT_MANIFEST.parent / "postfreeze_references.json"
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT = (
    ROOT / "outputs/tusz_common17_support_comparison_source_dev259_v1_20260825"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _materialize_worker(
    entry: dict[str, object], tusz_root: str, manifest_sha256: str
) -> dict[str, Any]:
    return materialize_tusz_real_edf_support_comparison_entry_v1(
        entry=entry,
        tusz_root=tusz_root,
        manifest_sha256=manifest_sha256,
    )


def _validated_resume_receipt(
    *, path: Path, rollout_id: str, manifest_sha256: str
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = validate_tusz_real_edf_support_comparison_event_v1(_load_json(path))
    except (OSError, TypeError, ValueError, KeyError):
        return None
    if value["rollout_id"] != rollout_id or value["manifest_sha256"] != manifest_sha256:
        return None
    return value


def _validate_postfreeze_bundle(
    *,
    value: Mapping[str, object],
    cohort_id: str,
    manifest_sha256: str,
    selected_event_ids: set[str],
) -> dict[str, dict[str, float]]:
    if value.get("schema_version") != (
        "clinical_eeg_tusz_support_comparison_postfreeze_references_v1"
    ):
        raise ValueError("post-freeze reference schema drifted")
    if value.get("cohort_id") != cohort_id:
        raise ValueError("post-freeze reference cohort differs from extraction")
    if value.get("extraction_manifest_sha256") != manifest_sha256:
        raise ValueError("post-freeze reference manifest hash differs from extraction")
    firewall = value.get("firewall")
    if firewall != {
        "file_is_input_to_extraction": False,
        "file_may_be_opened_only_after_event_receipts_are_frozen": True,
        "contains_channel_or_SOZ_targets": False,
    }:
        raise ValueError("post-freeze reference firewall drifted")
    supplied_hash = value.get("receipt_sha256")
    unhashed = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if supplied_hash != _canonical_sha256(unhashed):
        raise ValueError("post-freeze reference bundle hash mismatch")
    references = value.get("reference_intervals_by_event_id")
    if not isinstance(references, dict):
        raise TypeError("post-freeze reference map is absent")
    missing = selected_event_ids - set(references)
    if missing:
        raise ValueError(f"post-freeze references miss events: {sorted(missing)[:5]}")
    return {
        event_id: {
            "onset_seconds": float(references[event_id]["onset_seconds"]),
            "offset_seconds": float(references[event_id]["offset_seconds"]),
        }
        for event_id in selected_event_ids
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    manifest_path = arguments.manifest.resolve(strict=True)
    manifest = load_tusz_real_edf_support_comparison_manifest_v1(manifest_path)
    manifest_sha256 = _file_sha256(manifest_path)
    tusz_root = arguments.tusz_root.resolve(strict=True)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    requested = set(arguments.rollout_id or [])
    available = {str(row["rollout_id"]) for row in manifest["entries"]}
    unknown = requested - available
    if unknown:
        raise ValueError(f"unknown rollout IDs: {sorted(unknown)}")
    entries = [
        dict(row)
        for row in manifest["entries"]
        if not requested or str(row["rollout_id"]) in requested
    ]
    if not entries:
        raise ValueError("no support-comparison entries selected")

    by_rollout: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, object]] = []
    for entry in entries:
        rollout_id = str(entry["rollout_id"])
        receipt_path = output / "events" / rollout_id / "receipt.json"
        resumed = (
            _validated_resume_receipt(
                path=receipt_path,
                rollout_id=rollout_id,
                manifest_sha256=manifest_sha256,
            )
            if arguments.resume
            else None
        )
        if resumed is None:
            pending.append(entry)
        else:
            by_rollout[rollout_id] = resumed

    completed = len(by_rollout)
    print(
        json.dumps(
            {
                "stage": "target_blind_extraction_start",
                "selected": len(entries),
                "resumed": completed,
                "pending": len(pending),
                "workers": arguments.workers,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if pending:
        with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
            futures = {
                executor.submit(
                    _materialize_worker, entry, str(tusz_root), manifest_sha256
                ): entry
                for entry in pending
            }
            for future in as_completed(futures):
                entry = futures[future]
                rollout_id = str(entry["rollout_id"])
                receipt = future.result()
                target = output / "events" / rollout_id / "receipt.json"
                _atomic_json(target, receipt)
                by_rollout[rollout_id] = receipt
                completed += 1
                if completed == len(entries) or completed % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "target_blind_event_frozen",
                                "completed": completed,
                                "total": len(entries),
                                "elapsed_seconds": round(time.monotonic() - started, 3),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    event_receipts = [by_rollout[str(entry["rollout_id"])] for entry in entries]
    cohort = summarize_tusz_real_edf_support_comparison_cohort_v1(
        manifest_sha256=manifest_sha256,
        event_receipts=event_receipts,
    )
    cohort_path = output / "target_blind_cohort_receipt.json"
    _atomic_json(cohort_path, cohort)
    cohort_sha256 = _file_sha256(cohort_path)
    # This print is the explicit phase boundary: only now may TERM references open.
    print(
        json.dumps(
            {
                "stage": "target_blind_cohort_frozen",
                "receipt_sha256": cohort["receipt_sha256"],
                "file_sha256": cohort_sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    postfreeze_path = arguments.postfreeze_references.resolve(strict=True)
    raw_references = _load_json(postfreeze_path)
    event_ids = {str(entry["event_id"]) for entry in entries}
    references = _validate_postfreeze_bundle(
        value=raw_references,
        cohort_id=str(manifest["cohort_id"]),
        manifest_sha256=manifest_sha256,
        selected_event_ids=event_ids,
    )
    postfreeze = evaluate_common17_support_policy_postfreeze_references_v1(
        frozen_receipts=[row["event_comparison_receipt"] for row in event_receipts],
        reference_intervals_by_event_id=references,
    )
    postfreeze_output = output / "postfreeze_reference_audit.json"
    _atomic_json(postfreeze_output, postfreeze)

    receipt: dict[str, Any] = {
        "schema_version": "clinical_eeg_tusz_common17_support_comparison_run_receipt_v1",
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "status": "completed_target_blind_then_postfreeze_reference_audit",
        "cohort": {
            "event_count": len(entries),
            "recording_count": cohort["recording_count"],
            "patient_group_count": cohort["patient_group_count"],
            "full_frozen_source_dev259": not requested and len(entries) == 259,
        },
        "source_bindings": {
            "extraction_manifest_path": str(manifest_path),
            "extraction_manifest_sha256": manifest_sha256,
            "tusz_root": str(tusz_root),
            "postfreeze_reference_path": str(postfreeze_path),
            "postfreeze_reference_file_sha256": _file_sha256(postfreeze_path),
        },
        "phase_order_receipt": {
            "all_event_receipts_frozen_before_TERM_file_open": True,
            "target_blind_cohort_frozen_before_TERM_file_open": True,
            "TERM_used_for_query_window_or_stopping": False,
            "TERM_used_for_feature_measurement": False,
        },
        "target_blind_cohort": {
            "path": str(cohort_path),
            "file_sha256": cohort_sha256,
            "receipt_sha256": cohort["receipt_sha256"],
            "strategy_summary": cohort["target_blind_comparison_summary"][
                "strategy_summary"
            ],
            "same_budget_adaptive_vs_fixed_summary": cohort[
                "target_blind_comparison_summary"
            ]["same_budget_adaptive_vs_fixed_summary"],
            "high_budget_shadow_comparison_summary": cohort[
                "target_blind_comparison_summary"
            ]["high_budget_shadow_comparison_summary"],
        },
        "postfreeze_reference_audit": {
            "path": str(postfreeze_output),
            "file_sha256": _file_sha256(postfreeze_output),
            "receipt_sha256": postfreeze["receipt_sha256"],
            "per_strategy": postfreeze["per_strategy"],
        },
        "runtime": {
            "workers": arguments.workers,
            "resumed_event_count": len(entries) - len(pending),
            "materialized_event_count": len(pending),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
        "claim_limits": {
            "oracle_navigation_anchor_is_detector_performance": False,
            "fixed_120s_shadow_is_ground_truth": False,
            "SOZ_accuracy_measured": False,
            "source_dev_is_independent_test": False,
        },
    }
    receipt["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    _atomic_json(output / "receipt.json", receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    value.add_argument(
        "--postfreeze-references", type=Path, default=DEFAULT_POSTFREEZE_REFERENCES
    )
    value.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--workers", type=int, default=4)
    value.add_argument("--no-resume", action="store_false", dest="resume")
    value.set_defaults(resume=True)
    value.add_argument(
        "--rollout-id",
        action="append",
        help="Run only this frozen rollout ID (repeatable); default runs all 259.",
    )
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.workers < 1:
        raise ValueError("--workers must be >= 1")
    result = run(arguments)
    print(
        json.dumps(
            {
                "status": result["status"],
                "cohort": result["cohort"],
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
                "output": str(arguments.output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
