#!/usr/bin/env python3
"""Materialize real-EDF S04/S06 Findings receipts for one frozen A0 event.

This is a software/replay smoke, not an end-to-end detector or clinical
qualification experiment.  The supplied A0 navigation window is explicitly a
conditional-on-public-interval upper-bound support.  The interval payload is
not passed to either native Findings producer; only its already frozen
analysis support and event identity are used.  All S04/S06 measurements are
recomputed from EEG samples and acquisition metadata.  EDF annotations,
spreadsheets, channel targets, clinical text, reports, video, behaviour,
sleep/activation labels, ECG and LLMs are never opened.

The first invocation writes append-only canonical JSON.  ``--verify-existing``
reopens the EDF in a separate process invocation, recomputes every artifact,
requires byte-equivalent payloads, and writes a replay verification receipt.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.canonical_edf_materialization import (  # noqa: E402
    CanonicalEDFConfig,
    load_canonical_edf_views,
)
from src.clinical_eeg_long_recording.deterministic_event_findings import (  # noqa: E402
    DeterministicViewInput,
)
from src.clinical_eeg_long_recording.deterministic_event_morphology_primitives_v1 import (  # noqa: E402
    EventMorphologyPrimitiveViewInput,
)
from src.clinical_eeg_long_recording.deterministic_periodicity_candidate import (  # noqa: E402
    produce_deterministic_periodicity_candidate,
)
from src.clinical_eeg_long_recording.event_component_cycle_element_ledger_v1 import (  # noqa: E402
    materialize_event_component_cycle_element_ledger_from_signal_v1,
    validate_event_component_cycle_element_ledger_v1,
)
from src.clinical_eeg_long_recording.event_component_cycle_element_query_adapter_v1 import (  # noqa: E402
    materialize_event_component_cycle_element_query_adapter_v1,
    validate_event_component_cycle_element_query_adapter_v1,
)
from src.clinical_eeg_long_recording.event_physical_amplitude_findings_v1 import (  # noqa: E402
    EventPhysicalAmplitudeQuery,
    EventPhysicalAmplitudeViewInput,
    materialize_event_physical_amplitude_findings_v1,
    validate_event_physical_amplitude_findings_v1,
)
from src.clinical_eeg_long_recording.event_physical_amplitude_query_adapter_v1 import (  # noqa: E402
    materialize_event_physical_amplitude_query_adapter_v1,
    validate_event_physical_amplitude_query_adapter_v1,
)


_ARTIFACT_FILES = {
    "s04_findings": "s04_event_physical_amplitude_findings.json",
    "s04_query_adapter": "s04_query_adapter.json",
    "s06_ledger": "s06_component_cycle_element_ledger.json",
    "s06_query_adapter": "s06_query_adapter.json",
    "smoke_receipt": "smoke_receipt.json",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _sha(value: object) -> str:
    payload = deepcopy(value)
    return hashlib.sha256(_canonical_bytes(payload)[:-1]).hexdigest()


def _self_hash(value: Mapping[str, object], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _sha(body)


def _file_sha(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8", errors="strict"))
    canonical = _canonical_bytes(value)
    if type(value) is not dict or raw not in {canonical, canonical[:-1]}:
        raise ValueError(
            f"{path.name} is not canonical JSON with an optional final newline"
        )
    return value


def _write_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(value))


def _strict_source_identity(
    *,
    edf: Path,
    canonical: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    expected = identity["canonical_signal"]
    comparisons = {
        "recording_id": canonical["recording_id"],
        "canonical_signal_id": canonical["canonical_signal_id"],
        "canonical_receipt_sha256": canonical["receipt_sha256"],
        "source_signal_sha256": canonical["source_signal_sha256"],
        "recording_duration_seconds": canonical["recording_duration_seconds"],
    }
    for name, observed in comparisons.items():
        if observed != expected[name]:
            raise ValueError(f"real A0 canonical identity drifted at {name}")
    size, digest = _file_sha(edf)
    if size != identity["source_container"]["size_bytes"] or digest != identity[
        "source_container"
    ]["sha256"]:
        raise ValueError("real A0 EDF container bytes drifted")


def _course_intervals(
    analysis: tuple[float, float],
    context: tuple[float, float],
    *,
    width_seconds: float = 5.0,
) -> list[tuple[float, float]]:
    start = max(analysis[0], context[1])
    result: list[tuple[float, float]] = []
    while start < analysis[1] - 1e-9:
        stop = min(analysis[1], start + width_seconds)
        if stop - start >= 1.0:
            result.append((start, stop))
        start = stop
    if len(result) < 2:
        raise ValueError("real A0 support cannot provide two S04 course intervals")
    return result


def _materialize(
    *,
    edf: Path,
    navigation: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    bundle = load_canonical_edf_views(edf, config=CanonicalEDFConfig())
    canonical = bundle.canonical_record.canonical_receipt
    _strict_source_identity(edf=edf, canonical=canonical, identity=identity)
    event_id = str(navigation["event_identity"]["event_id"])
    if navigation["event_identity"]["recording_id"] != canonical["recording_id"]:
        raise ValueError("A0 navigation/canonical recording identity drifted")
    timing = navigation["timing"]
    analysis = tuple(float(item) for item in timing[
        "analysis_interval_recording_seconds"
    ])
    context = tuple(float(item) for item in timing[
        "baseline_context_recording_seconds"
    ])
    if not (
        analysis[0] <= context[0] < context[1] <= analysis[1]
        and analysis[1] <= float(canonical["recording_duration_seconds"])
    ):
        raise ValueError("A0 analysis/context support is invalid")

    references = ("referential", "tcp_bipolar")
    trusted_parent_views = {
        str(view.receipt["view_id"]): view.receipt
        for role_views in bundle.task_reference_views.values()
        for view in role_views.values()
    }
    native_views = [
        bundle.task_reference_views["findings_native_morphology"][reference]
        for reference in references
    ]
    offline_views = [
        bundle.task_reference_views["context_offline"][reference]
        for reference in references
    ]
    periodicity = []
    for view in offline_views:
        for unit in view.receipt["output_units"]:
            if not unit["observed"] or unit["imputed"]:
                continue
            periodicity.append(
                produce_deterministic_periodicity_candidate(
                    event_id=event_id,
                    analysis_unit_id=str(unit["unit_id"]),
                    analysis_interval_recording_seconds=analysis,
                    canonical_receipt=canonical,
                    views=[
                        DeterministicViewInput(
                            view_receipt=view.receipt,
                            tensor=view.tensor,
                        )
                    ],
                    trusted_parent_views=trusted_parent_views,
                )
            )
    s06 = materialize_event_component_cycle_element_ledger_from_signal_v1(
        event_id=event_id,
        canonical_receipt=canonical,
        morphology_views=[
            EventMorphologyPrimitiveViewInput(
                view_receipt=view.receipt,
                tensor=view.tensor,
            )
            for view in native_views
        ],
        periodicity_candidates=periodicity,
        analysis_interval_seconds=analysis,
        trusted_parent_views=trusted_parent_views,
    )
    s06_adapter = materialize_event_component_cycle_element_query_adapter_v1(s06)

    selection_policy = {
        "policy_id": "A0-REAL-S04-UNIFORM-SUPPORT-SCHEDULE-SMOKE-V1",
        "event_id": event_id,
        "analysis_interval_seconds": list(analysis),
        "comparison_context_interval_seconds": list(context),
        "course_intervals_seconds": [
            list(item) for item in _course_intervals(analysis, context)
        ],
        "selection_semantics": (
            "a0_navigation_bound_uniform_measurement_schedule_not_clinical_context"
        ),
        "a0_navigation_receipt_sha256": navigation["receipt_sha256"],
        "clinical_context_qualification_authorized": False,
        "onset_or_soz_support_authorized": False,
    }
    selection_sha = _sha(selection_policy)
    s04_queries = []
    course = _course_intervals(analysis, context)
    for view in native_views:
        for unit in view.receipt["output_units"]:
            if not unit["observed"] or unit["imputed"]:
                continue
            unit_id = str(unit["unit_id"])
            comparison_set_id = "CMP-A0-S04-" + _sha(
                [event_id, str(view.receipt["view_id"]), unit_id]
            )[:20]
            s04_queries.append(
                EventPhysicalAmplitudeQuery(
                    view_id=str(view.receipt["view_id"]),
                    unit_id=unit_id,
                    recording_interval_seconds=context,
                    measurement_role="signal_selected_comparison_context",
                    comparison_set_id=comparison_set_id,
                    ordinal=1,
                    query_authority="deterministic_signal_proposal",
                    selection_receipt_sha256=selection_sha,
                )
            )
            for ordinal, interval in enumerate(course, start=1):
                s04_queries.append(
                    EventPhysicalAmplitudeQuery(
                        view_id=str(view.receipt["view_id"]),
                        unit_id=unit_id,
                        recording_interval_seconds=interval,
                        measurement_role="event_course",
                        comparison_set_id=comparison_set_id,
                        ordinal=ordinal,
                        query_authority="deterministic_signal_proposal",
                        selection_receipt_sha256=selection_sha,
                    )
                )
    s04 = materialize_event_physical_amplitude_findings_v1(
        event_id=event_id,
        canonical_receipt=canonical,
        views=[
            EventPhysicalAmplitudeViewInput(
                view_receipt=view.receipt,
                tensor=view.tensor,
            )
            for view in native_views
        ],
        analysis_interval_seconds=analysis,
        queries=s04_queries,
        trusted_parent_views=trusted_parent_views,
    )
    s04_adapter = materialize_event_physical_amplitude_query_adapter_v1(s04)
    source_size, source_sha = _file_sha(edf)
    smoke = {
        "schema_version": "clinical_eeg_a0_real_findings_s04_s06_smoke_v1",
        "method_id": "A0-REAL-EDF-FINDINGS-S04-S06-SMOKE-V1",
        "event_id": event_id,
        "recording_id": canonical["recording_id"],
        "canonical_signal_id": canonical["canonical_signal_id"],
        "canonical_receipt_sha256": canonical["receipt_sha256"],
        "source_signal_sha256": canonical["source_signal_sha256"],
        "source_container": {
            "logical_source_recording_id": identity["source_recording_id"],
            "size_bytes": source_size,
            "sha256": source_sha,
        },
        "analysis_interval_seconds": list(analysis),
        "support_authority": {
            "navigation_arm": navigation["navigation_arm"],
            "evaluation_semantics": navigation["evaluation_semantics"],
            "a0_navigation_receipt_sha256": navigation["receipt_sha256"],
            "public_tusz_interval_used_for_outer_navigation": True,
            "public_reference_interval_payload_copied_into_findings": False,
            "detector_frozen_claim_authorized": False,
            "final_rule_adaptive_support_materialized": False,
            "software_smoke_only": True,
        },
        "s04": {
            "selection_policy": selection_policy,
            "selection_policy_sha256": selection_sha,
            "findings_receipt_sha256": s04["receipt_sha256"],
            "query_adapter_receipt_sha256": s04_adapter["receipt_sha256"],
            "measurement_count": len(s04["measurements"]),
            "trajectory_count": len(s04["amplitude_trajectories"]),
            "report_promotion_authorized": False,
            "onset_or_soz_support_authorized": False,
        },
        "s06": {
            "ledger_receipt_sha256": s06["receipt_sha256"],
            "query_adapter_receipt_sha256": s06_adapter["receipt_sha256"],
            "periodicity_source_candidate_count": len(periodicity),
            "periodicity_candidate_only_count": sum(
                row["qualification_status"] == "candidate_only"
                for row in periodicity
            ),
            "component_instance_count": s06["component_instance_ledger"]["count"],
            "cycle_instance_count": s06["cycle_instance_ledger"]["count"],
            "element_instance_count": s06["element_instance_ledger"]["count"],
            "report_promotion_authorized": False,
            "clinical_term_qualification_authorized": False,
            "onset_or_soz_support_authorized": False,
        },
        "input_firewall": {
            "real_eeg_samples_used": True,
            "allowlisted_acquisition_metadata_used": True,
            "edf_annotations_opened": False,
            "spreadsheets_opened": False,
            "doctor_labels_or_reports_opened": False,
            "clinical_text_opened": False,
            "video_or_behavior_opened": False,
            "sleep_or_activation_labels_opened": False,
            "ecg_emg_eog_opened": False,
            "qwen_or_other_llm_used": False,
        },
        "authorization": {
            "software_and_replay_smoke_only": True,
            "clinical_or_production_use_authorized": False,
            "performance_claim_authorized": False,
            "report_eligible_term_allowlist": [],
        },
        "receipt_sha256": "",
    }
    smoke["receipt_sha256"] = _self_hash(smoke, "receipt_sha256")
    return {
        "s04_findings": validate_event_physical_amplitude_findings_v1(s04),
        "s04_query_adapter": validate_event_physical_amplitude_query_adapter_v1(
            s04_adapter
        ),
        "s06_ledger": validate_event_component_cycle_element_ledger_v1(s06),
        "s06_query_adapter": (
            validate_event_component_cycle_element_query_adapter_v1(s06_adapter)
        ),
        "smoke_receipt": smoke,
    }


def _verify_existing(output: Path, expected: Mapping[str, Mapping[str, Any]]) -> None:
    observed = {
        key: _read_json(output / filename)
        for key, filename in _ARTIFACT_FILES.items()
    }
    if observed != expected:
        raise ValueError("real A0 S04/S06 artifacts do not replay exactly from EDF")
    verification = {
        "schema_version": "clinical_eeg_a0_real_findings_s04_s06_replay_v1",
        "method_id": "SEPARATE-PROCESS-REAL-EDF-EXACT-REPLAY-V1",
        "source_smoke_receipt_sha256": expected["smoke_receipt"][
            "receipt_sha256"
        ],
        "artifact_sha256s": {
            key: (
                value.get("receipt_sha256")
                if key != "smoke_receipt"
                else value["receipt_sha256"]
            )
            for key, value in expected.items()
        },
        "separate_process_recomputation": True,
        "edf_reopened": True,
        "all_payloads_exact_match": True,
        "performance_or_clinical_claim_authorized": False,
        "receipt_sha256": "",
    }
    verification["receipt_sha256"] = _self_hash(
        verification, "receipt_sha256"
    )
    _write_json_no_clobber(output / "replay_verification.json", verification)
    print(json.dumps(verification, sort_keys=True, ensure_ascii=False))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--navigation-window", type=Path, required=True)
    parser.add_argument("--canonical-identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    edf = args.edf.resolve(strict=True)
    if not edf.is_file() or edf.is_symlink():
        raise ValueError("--edf must be a non-symlink regular file")
    navigation = _read_json(args.navigation_window)
    identity = _read_json(args.canonical_identity)
    artifacts = _materialize(
        edf=edf,
        navigation=navigation,
        identity=identity,
    )
    output = args.output.resolve()
    if args.verify_existing:
        if not output.is_dir() or output.is_symlink():
            raise ValueError("--verify-existing requires an existing regular output")
        _verify_existing(output, artifacts)
        return
    output.mkdir(parents=True, exist_ok=False)
    for key, filename in _ARTIFACT_FILES.items():
        _write_json_no_clobber(output / filename, artifacts[key])
    summary = {
        "output": str(output),
        "receipt_sha256": artifacts["smoke_receipt"]["receipt_sha256"],
        "s04": artifacts["smoke_receipt"]["s04"],
        "s06": artifacts["smoke_receipt"]["s06"],
        "separate_process_replay_pending": True,
    }
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
