#!/usr/bin/env python3
"""Materialize and independently replay real-EDF S03 frequency Findings.

This is a native-measurement and disk-replay smoke over one frozen A0 event.
The A0 support is an oracle-conditioned upper-bound support and therefore is
not admissible as patient-OOF detector evidence, model performance, or a
clinical frequency/evolution/SOZ claim.  The script never opens EDF
annotations, spreadsheets, clinical text, reports, video/behaviour, sleep or
activation labels, ECG, or an LLM.

The first invocation writes canonical append-only JSON.  ``--verify-existing``
reopens the EDF in another process invocation, rebuilds the canonical views,
remeasures every spectral window from native EEG samples, and requires exact
payload equality before writing a replay receipt.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.ba_ieg_dense_measurement_sidecar import (  # noqa: E402
    BAIEGDenseMeasurementPolicy,
    BAIEGDenseMeasurementViewInput,
    materialize_ba_ieg_dense_measurement_sidecar,
)
from src.clinical_eeg_long_recording.canonical_edf_materialization import (  # noqa: E402
    CanonicalEDFConfig,
    load_canonical_edf_views,
)
from src.clinical_eeg_long_recording.event_frequency_findings_v1 import (  # noqa: E402
    materialize_event_frequency_findings_from_native_signal_v1,
    validate_event_frequency_findings_v1,
)
from src.clinical_eeg_long_recording.event_frequency_matched_context_selector_v1 import (  # noqa: E402
    materialize_s03_matched_context_selector_v1,
    validate_s03_matched_context_selector_v1,
)


_ARTIFACT_FILES = {
    "context_selector": "s03_matched_context_selector.json",
    "context_opportunity_sidecar": "s03_context_opportunity_sidecar.json",
    "s03_findings": "s03_event_frequency_findings.json",
    "dense_sidecar": "s03_dense_measurement_sidecar.json",
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
    return hashlib.sha256(_canonical_bytes(value)[:-1]).hexdigest()


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
    if (
        size != identity["source_container"]["size_bytes"]
        or digest != identity["source_container"]["sha256"]
    ):
        raise ValueError("real A0 EDF container bytes drifted")


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
    analysis = tuple(
        float(item) for item in timing["analysis_interval_recording_seconds"]
    )
    # The selector receives only the frozen A0 navigation anchor and physical
    # support.  It never reads the public reference interval or the legacy
    # baseline-context payload.  This remains an A0 conditional smoke, not an
    # A1 or clinical-onset claim.
    navigation_anchor = float(timing["navigation_anchor_seconds"])
    event_course = (navigation_anchor, analysis[1])
    if not (
        analysis[0] < event_course[0] < event_course[1] <= analysis[1]
        and analysis[1] <= float(canonical["recording_duration_seconds"])
    ):
        raise ValueError("A0 S03 analysis/navigation support is invalid")

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
    view_inputs: list[BAIEGDenseMeasurementViewInput] = []
    next_unit_index = 0
    for view_index, view in enumerate(native_views):
        unit_count = len(view.receipt["output_units"])
        view_inputs.append(
            BAIEGDenseMeasurementViewInput(
                view_index=view_index,
                unit_indices=tuple(
                    range(next_unit_index, next_unit_index + unit_count)
                ),
                view_receipt=view.receipt,
                tensor=view.tensor,
            )
        )
        next_unit_index += unit_count

    measurement_policy = BAIEGDenseMeasurementPolicy(
        window_seconds=1.0,
        step_seconds=0.5,
        minimum_baseline_windows=2,
    )
    context_opportunity = materialize_ba_ieg_dense_measurement_sidecar(
        canonical_receipt=canonical,
        views=view_inputs,
        analysis_interval_seconds=analysis,
        background_intervals_seconds=(),
        policy=measurement_policy,
        trusted_parent_views=trusted_parent_views,
    )
    context_selector = validate_s03_matched_context_selector_v1(
        materialize_s03_matched_context_selector_v1(
            event_id=event_id,
            dense_measurement_opportunity=context_opportunity,
            event_interval_seconds=event_course,
        )
    )
    comparison_contexts = tuple(
        tuple(float(value) for value in interval)
        for interval in context_selector[
            "s03_comparison_context_intervals_seconds"
        ]
    )
    s03, sidecar = materialize_event_frequency_findings_from_native_signal_v1(
        event_id=event_id,
        canonical_receipt=canonical,
        views=view_inputs,
        analysis_interval_seconds=analysis,
        event_course_interval_seconds=event_course,
        comparison_context_intervals_seconds=comparison_contexts,
        comparison_set_id=(
            "CMP-A0-S03-SELECTOR-" + context_selector["receipt_sha256"][:20]
        ),
        selection_receipt_sha256=context_selector["receipt_sha256"],
        query_authority="deterministic_signal_proposal",
        measurement_policy=measurement_policy,
        trusted_parent_views=trusted_parent_views,
    )
    s03 = validate_event_frequency_findings_v1(s03)
    dense_sidecar = sidecar.to_dict()
    context_opportunity_sidecar = context_opportunity.to_dict()

    units = s03["units"]
    event_windows = [
        row
        for unit in units
        for row in unit["window_measurements"]
        if row["measurement_role"] == "event_course"
    ]
    context_windows = [
        row
        for unit in units
        for row in unit["window_measurements"]
        if row["measurement_role"] == "signal_selected_comparison_context"
    ]
    source_size, source_sha = _file_sha(edf)
    smoke: dict[str, Any] = {
        "schema_version": "clinical_eeg_a0_real_findings_s03_selector_smoke_v1",
        "method_id": "A0-REAL-EDF-NATIVE-S03-EEG-ONLY-CONTEXT-SELECTOR-SMOKE-V1",
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
            "public_reference_interval_payload_copied_into_context_selector": False,
            "navigation_anchor_used_as_a0_candidate_support_boundary": True,
            "patient_oof_a1_claim_authorized": False,
            "final_rule_adaptive_support_materialized": False,
            "software_and_replay_smoke_only": True,
        },
        "s03": {
            "selection_policy": context_selector["policy"],
            "selection_policy_sha256": context_selector["policy"][
                "policy_sha256"
            ],
            "context_selector_receipt_sha256": context_selector[
                "receipt_sha256"
            ],
            "context_selector_decision_receipt_sha256": context_selector[
                "decision_receipt_sha256"
            ],
            "context_opportunity_sidecar_receipt_sha256": (
                context_opportunity.receipt_sha256
            ),
            "comparison_context_intervals_seconds": [
                list(item) for item in comparison_contexts
            ],
            "local_pre_event_context_status": context_selector["selections"][
                "local_pre_event"
            ]["status"],
            "distant_pre_event_context_status": context_selector["selections"][
                "distant_pre_event"
            ]["status"],
            "course_only_future_context_status": context_selector["selections"][
                "course_only_future_context"
            ]["status"],
            "context_selector_only_defines_delta": True,
            "context_selector_direct_soz_score_authorized": False,
            "context_selector_onset_support_permission": "forbidden",
            "measurement_policy_sha256": measurement_policy.sha256,
            "findings_receipt_sha256": s03["receipt_sha256"],
            "dense_sidecar_receipt_sha256": sidecar.receipt_sha256,
            "typed_unit_count": len(units),
            "electrode_unit_count": sum(
                unit["unit_type"] == "electrode" for unit in units
            ),
            "whole_lead_unit_count": sum(
                unit["unit_type"] == "lead" for unit in units
            ),
            "event_window_count": len(event_windows),
            "evaluable_event_window_count": sum(
                row["status"] == "measured" for row in event_windows
            ),
            "context_window_count": len(context_windows),
            "evaluable_context_window_count": sum(
                row["status"] == "measured" for row in context_windows
            ),
            "pathological_frequency_term_authorized": False,
            "clinical_evolution_term_authorized": False,
            "report_promotion_authorized": False,
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
            "clinical_term_qualification_authorized": False,
            "report_eligible_term_allowlist": [],
        },
        "receipt_sha256": "",
    }
    smoke["receipt_sha256"] = _self_hash(smoke, "receipt_sha256")
    return {
        "context_selector": context_selector,
        "context_opportunity_sidecar": context_opportunity_sidecar,
        "s03_findings": s03,
        "dense_sidecar": dense_sidecar,
        "smoke_receipt": smoke,
    }


def _verify_existing(output: Path, expected: Mapping[str, Mapping[str, Any]]) -> None:
    observed = {
        key: _read_json(output / filename)
        for key, filename in _ARTIFACT_FILES.items()
    }
    # ``BAIEGDenseMeasurementSidecar.to_dict`` intentionally retains tuples
    # in a few in-memory provenance fields.  Canonical JSON represents those
    # sequences as arrays, so compare the exact disk projection rather than
    # Python's tuple-vs-list container identity.
    expected_disk = {
        key: json.loads(_canonical_bytes(value).decode("utf-8"))
        for key, value in expected.items()
    }
    if observed != expected_disk:
        raise ValueError("real A0 S03 artifacts do not replay exactly from EDF")
    verification: dict[str, Any] = {
        "schema_version": "clinical_eeg_a0_real_findings_s03_selector_replay_v1",
        "method_id": (
            "SEPARATE-PROCESS-REAL-EDF-S03-EEG-ONLY-CONTEXT-SELECTOR-"
            "AND-NATIVE-FINDINGS-EXACT-REPLAY-V1"
        ),
        "source_smoke_receipt_sha256": expected["smoke_receipt"][
            "receipt_sha256"
        ],
        "artifact_sha256s": {
            "context_selector": expected["context_selector"]["receipt_sha256"],
            "context_selector_decision": expected["context_selector"][
                "decision_receipt_sha256"
            ],
            "context_opportunity_sidecar": expected[
                "context_opportunity_sidecar"
            ]["receipt_sha256"],
            "s03_findings": expected["s03_findings"]["receipt_sha256"],
            "dense_sidecar": expected["dense_sidecar"]["receipt_sha256"],
            "smoke_receipt": expected["smoke_receipt"]["receipt_sha256"],
        },
        "separate_process_recomputation": True,
        "edf_reopened": True,
        "eeg_only_context_selector_recomputed": True,
        "measurement_target_values_used_for_context_selection": False,
        "native_spectral_measurements_recomputed": True,
        "all_payloads_exact_match": True,
        "patient_oof_a1_claim_authorized": False,
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
        "s03": artifacts["smoke_receipt"]["s03"],
        "separate_process_replay_pending": True,
    }
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
