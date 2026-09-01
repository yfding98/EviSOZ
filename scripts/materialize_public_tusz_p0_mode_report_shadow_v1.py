#!/usr/bin/env python3
"""Materialize one real TUSZ P0 EEG-only report shadow end to end.

The CLI opens exactly four input classes: a public EDF through the canonical
signal-only reader, one target-free adaptive-search artifact, its target-free
P0 candidate roster and the corresponding detector manifest.  It has no
argument for a TUSZ annotation sidecar, target, label, spreadsheet, report or
clinical narrative.

The resulting report is an audit preview.  It is not a clinical diagnosis and
does not promote the untrained mode-aware adapter or unqualified deterministic
Findings into a localized onset claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.adaptive_event_window import (
    derive_adaptive_event_analysis_window,
)
from src.clinical_eeg_long_recording.canonical_edf_materialization import (
    CanonicalEDFConfig,
    load_canonical_edf_views,
)
from src.clinical_eeg_long_recording.deterministic_event_findings import (
    DeterministicEventFindingsPolicy,
    DeterministicViewInput,
)
from src.clinical_eeg_long_recording.deterministic_event_findings_v2 import (
    DEFAULT_EVENT_FINDINGS_V2_REGISTRY_BINDINGS,
)
from src.clinical_eeg_long_recording.deterministic_event_findings_v3 import (
    produce_deterministic_event_eeg_findings_v3_candidate,
)
from src.clinical_eeg_long_recording.event_processing_ledger_v2 import (
    COMPLETED_FINDINGS,
    COMPLETED_FINDINGS_ONSET_NONLOCALIZABLE,
    build_event_processing_ledger_v2,
)
from src.clinical_eeg_long_recording.mode_aware_claim_locked_report_shadow_v1 import (
    materialize_mode_aware_claim_locked_report_shadow_v1,
)
from src.clinical_eeg_long_recording.mode_aware_mil_report_graph_v2_bridge_shadow_v1 import (
    materialize_mode_aware_mil_report_graph_v2_bridge_shadow_v1,
)
from src.clinical_eeg_long_recording.multievent_soz_report_graph_v2 import (
    materialize_multievent_soz_report_graph_v2,
)
from src.clinical_eeg_long_recording.public_findings_v3_mode_aware_shadow_adapter_v1 import (
    build_public_findings_v3_mode_aware_shadow_inputs_v1,
)


SCHEMA_VERSION = "public_tusz_p0_mode_aware_report_shadow_smoke_v1"
METHOD_ID = "real_public_signal_only_p0_event_to_claim_locked_report_shadow_v1"


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain one JSON object")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _event_row(roster: Mapping[str, Any], event_id: str) -> dict[str, Any]:
    rows = [item for item in roster.get("rows", []) if item.get("event_id") == event_id]
    if len(rows) != 1:
        raise ValueError(
            "target-free P0 roster must contain the requested event exactly once"
        )
    row = deepcopy(dict(rows[0]))
    if (
        roster.get("method_id")
        != "post_p0_projection_pre_reference_global_roster_freeze_v1"
    ):
        raise ValueError("candidate roster is not the frozen target-free P0 roster")
    return row


def _adaptive_event(artifact: Mapping[str, Any], event_id: str) -> dict[str, Any]:
    rows = [
        item
        for item in artifact.get("events", [])
        if item.get("eeg_event_id") == event_id
    ]
    if len(rows) != 1:
        raise ValueError(
            "adaptive-search artifact must contain the requested event once"
        )
    row = deepcopy(dict(rows[0]))
    search = row.get("adaptive_search_receipt")
    if not isinstance(search, dict):
        raise ValueError("requested event lacks an adaptive-search receipt")
    scope = search.get("scope_receipt")
    if (
        not isinstance(scope, dict)
        or scope.get("scope") != "eeg_signal_only_no_annotation_excel_or_ground_truth"
        or scope.get("eeg_signal_used") is not True
        or scope.get("edf_annotations_used") is not False
        or scope.get("excel_used") is not False
        or scope.get("labels_or_ground_truth_used") is not False
        or scope.get("detector_anchor_used_for_navigation_only") is not True
        or scope.get("candidate_is_confirmed_seizure") is not False
    ):
        raise ValueError("adaptive-search receipt violates the EEG-only firewall")
    return row


def _detector_candidate(
    manifest: Mapping[str, Any], candidate_id: str
) -> dict[str, Any]:
    receipt = manifest.get("detector_receipt")
    if (
        not isinstance(receipt, dict)
        or receipt.get("annotations_used") is not False
        or receipt.get("labels_used") is not False
    ):
        raise ValueError("detector manifest is not target-free")
    rows = [
        item
        for item in manifest.get("merge_candidates", [])
        if item.get("candidate_id") == candidate_id
    ]
    if len(rows) != 1:
        raise ValueError("detector manifest must contain the selected candidate once")
    return deepcopy(dict(rows[0]))


def _producer_receipts(findings: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["receipt_id"]): deepcopy(dict(item))
        for item in findings["producer_receipts"]
    }


def _stage_hashes(
    *,
    search: Mapping[str, Any],
    window: Mapping[str, Any],
    findings: Mapping[str, Any],
    findings_sha256: str,
) -> dict[str, str | None]:
    quality_findings = [
        item for item in findings["findings"] if item["family"] == "quality"
    ]
    return {
        "adaptive_search_sha256": _canonical_sha256(search),
        "adaptive_window_sha256": _canonical_sha256(window),
        "quality_assessment_sha256": _canonical_sha256(quality_findings),
        "event_findings_sha256": findings_sha256,
        "waveform_manifest_sha256": None,
        "record_claim_sha256": None,
        "rendered_claim_sha256": None,
        "technical_failure_receipt_sha256": None,
    }


def materialize(arguments: argparse.Namespace) -> dict[str, Any]:
    edf_path = arguments.edf.resolve(strict=True)
    if edf_path.suffix.lower() != ".edf" or not edf_path.is_file():
        raise ValueError("--edf must be a regular EDF file")
    adaptive_path = arguments.adaptive_search.resolve(strict=True)
    roster_path = arguments.candidate_roster.resolve(strict=True)
    manifest_path = arguments.detection_manifest.resolve(strict=True)
    if adaptive_path.name != "adaptive_search.json":
        raise ValueError("--adaptive-search must be a target-free adaptive_search.json")
    if roster_path.name != "target_free_candidate_roster.json":
        raise ValueError("--candidate-roster must be target_free_candidate_roster.json")
    if manifest_path.name != "detection_manifest.json":
        raise ValueError("--detection-manifest must be detection_manifest.json")

    roster = _json(roster_path)
    row = _event_row(roster, arguments.event_id)
    adaptive_artifact = _json(adaptive_path)
    adaptive_event = _adaptive_event(adaptive_artifact, arguments.event_id)
    search = adaptive_event["adaptive_search_receipt"]
    manifest = _json(manifest_path)
    manifest_sha = _canonical_sha256(manifest)
    if manifest_sha != row["detection_manifest_sha256"]:
        raise ValueError("P0 roster and detector manifest content hashes disagree")
    candidate = _detector_candidate(manifest, str(row["detector_candidate_id"]))
    if adaptive_event["candidate_id"] != candidate["candidate_id"]:
        raise ValueError("adaptive event and detector candidate IDs disagree")
    if [
        float(candidate["start_offset_seconds"]),
        float(candidate["stop_offset_seconds"]),
    ] != [
        float(item)
        for item in row["detector_candidate_support_interval_recording_seconds"]
    ]:
        raise ValueError("P0 roster and detector candidate interval disagree")
    if _canonical_sha256(search) != row["adaptive_search_receipt_sha256"]:
        raise ValueError("P0 roster and adaptive-search receipt hashes disagree")
    window = derive_adaptive_event_analysis_window(search)
    if _canonical_sha256(window) != row["adaptive_window_receipt_sha256"]:
        raise ValueError("derived variable window does not replay the frozen P0 hash")

    bundle = load_canonical_edf_views(
        edf_path,
        config=CanonicalEDFConfig(onset_fir_numtaps=arguments.onset_fir_numtaps),
    )
    canonical = bundle.canonical_record.canonical_receipt
    binding = search["canonical_signal_binding"]
    if (
        canonical["receipt_sha256"] != binding["canonical_receipt_sha256"]
        or canonical["source_signal_sha256"]
        != binding["canonical_source_signal_sha256"]
    ):
        raise ValueError("public EDF does not replay the P0 canonical signal root")
    qualification = bundle.materialization_receipt["onset_fir_response_qualification"]
    clinical_admission = bundle.materialization_receipt[
        "onset_fir_clinical_admission_qualification"
    ]
    if qualification["target_band_claim_authorized"] is not True:
        raise ValueError(
            "configured causal FIR is not response-qualified; this MIL smoke "
            "cannot create future-free onset candidate inputs"
        )
    if clinical_admission["clinical_onset_support_authorized"] is not True:
        raise ValueError(
            "configured causal FIR lacks complete clinical onset admission; "
            "response qualification alone cannot create onset/SOZ evidence"
        )
    causal_parent = bundle.onset_causal
    offline_parent = bundle.context_offline
    causal = bundle.task_reference_views["onset_causal"]["tcp_bipolar"]
    offline = bundle.task_reference_views["context_offline"]["tcp_bipolar"]
    findings = produce_deterministic_event_eeg_findings_v3_candidate(
        event_id=arguments.event_id,
        adaptive_search_receipt=search,
        adaptive_window_receipt=window,
        canonical_receipt=canonical,
        views=[
            DeterministicViewInput(
                view_receipt=causal.receipt,
                tensor=causal.tensor,
                onset_fir_response_qualification=qualification,
                onset_fir_clinical_admission_qualification=clinical_admission,
            ),
            DeterministicViewInput(
                view_receipt=offline.receipt,
                tensor=offline.tensor,
            ),
        ],
        trusted_parent_views={
            causal_parent.receipt["view_id"]: causal_parent.receipt,
            offline_parent.receipt["view_id"]: offline_parent.receipt,
        },
        policy=DeterministicEventFindingsPolicy(
            change_score_threshold=arguments.change_score_threshold
        ),
    )
    producers = _producer_receipts(findings)
    registry = deepcopy(DEFAULT_EVENT_FINDINGS_V2_REGISTRY_BINDINGS)
    graph_kwargs = {
        "trusted_producer_receipts": producers,
        "trusted_registry_bindings": registry,
    }
    graph = materialize_multievent_soz_report_graph_v2(
        [findings], route_scope="public", **graph_kwargs
    )
    (
        bag,
        policy,
        forward,
        hard_bindings,
        adapter_receipt,
    ) = build_public_findings_v3_mode_aware_shadow_inputs_v1(
        graph,
        trusted_source_event_findings_v3=[findings],
        record_pseudonym=("PUBREC-" + str(canonical["source_signal_sha256"])[:20]),
        **graph_kwargs,
    )
    event_adapter = adapter_receipt["event_receipts"][0]
    status = str(event_adapter["recommended_event_ledger_status"])
    if status not in {
        COMPLETED_FINDINGS,
        COMPLETED_FINDINGS_ONSET_NONLOCALIZABLE,
    }:
        raise ValueError("adapter recommended an unsupported event ledger status")
    wrapper = graph["source_event_graphs"][0]
    outcome_reasons = (
        []
        if status == COMPLETED_FINDINGS
        else ["untrained_shadow_event_qualification_not_promoted"]
    )
    ledger = build_event_processing_ledger_v2(
        recording_id=str(graph["record"]["record_id"]),
        recording_duration_seconds=float(graph["record"]["recording_duration_seconds"]),
        canonical_signal_sha256=str(graph["record"]["canonical_signal_sha256"]),
        canonical_materialization_receipt_sha256=str(
            bundle.materialization_receipt["receipt_sha256"]
        ),
        detection_manifest_sha256=manifest_sha,
        detector_selected_roster=[
            {
                "event_id": arguments.event_id,
                "detector_candidate_id": str(candidate["candidate_id"]),
                "detector_event_ordinal": 1,
                "candidate_start_seconds": float(candidate["start_offset_seconds"]),
                "candidate_stop_seconds": float(candidate["stop_offset_seconds"]),
                "candidate_anchor_seconds": float(candidate["anchor_offset_seconds"]),
                "source_detector_candidate_sha256": str(
                    row["detector_candidate_receipt_sha256"]
                ),
            }
        ],
        event_outcomes=[
            {
                "event_id": arguments.event_id,
                "detector_candidate_id": str(candidate["candidate_id"]),
                "outcome_status": status,
                "reason_codes": outcome_reasons,
                "stage_hashes": _stage_hashes(
                    search=search,
                    window=window,
                    findings=findings,
                    findings_sha256=str(wrapper["source_event_findings_v3_sha256"]),
                ),
            }
        ],
    )
    bridge = materialize_mode_aware_mil_report_graph_v2_bridge_shadow_v1(
        graph,
        trusted_source_event_findings_v3=[findings],
        event_processing_ledger_v2=ledger,
        mil_bag=bag,
        mil_policy=policy,
        mil_forward=forward,
        hard_input_bindings=hard_bindings,
        **graph_kwargs,
    )
    report = materialize_mode_aware_claim_locked_report_shadow_v1(
        bridge,
        report_graph_v2=graph,
        trusted_source_event_findings_v3=[findings],
        event_processing_ledger_v2=ledger,
        mil_bag=bag,
        mil_policy=policy,
        mil_forward=forward,
        hard_input_bindings=hard_bindings,
        **graph_kwargs,
    )

    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "event_findings_v3.json": findings,
        "report_graph_v2.json": graph,
        "event_processing_ledger_v2.json": ledger,
        "mode_aware_adapter_receipt.json": adapter_receipt,
        "mode_aware_bridge_shadow.json": bridge,
        "claim_locked_report_shadow.json": report,
    }
    for name, value in artifacts.items():
        _atomic_json(output / name, value)
    _atomic_text(output / "report.txt", str(report["canonical_report_text_zh"]))

    family_counts = Counter(str(item["family"]) for item in findings["findings"])
    status_counts = Counter(str(item["status"]) for item in findings["findings"])
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "scope": {
            "real_public_eeg_signal_used": True,
            "p0_selected_event_roster_only": True,
            "full_detector_record_diagnostic_claimed": False,
            "target_or_label_artifact_read": False,
            "edf_annotation_api_called": False,
            "edf_annotations_used": False,
            "excel_used": False,
            "doctor_labels_used": False,
            "clinical_text_used": False,
            "private_source_used": False,
            "clinical_use_authorized": False,
            "candidate_shadow_only": True,
        },
        "input_bindings": {
            "event_id": arguments.event_id,
            "p0_candidate_roster_receipt_sha256": roster["receipt_sha256"],
            "p0_candidate_row_roster_sha256": roster["row_roster_sha256"],
            "detector_manifest_sha256": manifest_sha,
            "detector_candidate_receipt_sha256": row[
                "detector_candidate_receipt_sha256"
            ],
            "adaptive_search_receipt_sha256": _canonical_sha256(search),
            "adaptive_window_receipt_sha256": _canonical_sha256(window),
            "canonical_signal_sha256": canonical["source_signal_sha256"],
            "canonical_receipt_sha256": canonical["receipt_sha256"],
            "canonical_materialization_receipt_sha256": (
                bundle.materialization_receipt["receipt_sha256"]
            ),
        },
        "signal_only_loader_scope_receipt": deepcopy(
            bundle.canonical_record.source_header_receipt["scope_receipt"]
        ),
        "causal_fir_qualification": deepcopy(qualification),
        "variable_window": {
            "status": window["status"],
            "analysis_interval_recording_seconds": window[
                "analysis_interval_recording_seconds"
            ],
            "right_censored": window["censoring"]["right"],
            "termination_observed": window["censoring"]["termination_observed"],
        },
        "findings": {
            "event_qualification_status": findings["event_qualification"]["status"],
            "event_outcome": deepcopy(findings["event_outcome"]),
            "finding_count": len(findings["findings"]),
            "family_counts": dict(sorted(family_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "future_free_onset_evidence_ids": event_adapter[
                "future_free_onset_evidence_ids"
            ],
        },
        "event_to_record": {
            "ledger_outcome_status": status,
            "record_phenotype_candidate": forward.record_phenotype,
            "record_resolution_ceiling": forward.record_resolution_ceiling,
            "channel_logits_uniform": event_adapter["channel_logits_uniform"],
            "formal_report_authorized": bridge["promotion_gate"][
                "formal_report_authorized"
            ],
        },
        "serialization": deepcopy(report["faithful_rendering_evaluation"]),
        "artifact_hashes": {
            name: _canonical_sha256(value) for name, value in artifacts.items()
        },
        "report_text_sha256": hashlib.sha256(
            str(report["canonical_report_text_zh"]).encode("utf-8")
        ).hexdigest(),
        "remaining_blockers": [
            "real_event_head_is_unqualified_so_localized_channel_claim_is_withheld",
            "mode_aware_adapter_is_untrained_and_uncalibrated",
            "spread_channel_logits_are_not_value_bound_to_course_findings_in_bridge_v1",
            "morphology_clinical_terms_remain_not_evaluable_without_qualified_heads",
            "smoke_covers_one_frozen_p0_selected_event_not_the_complete_record_roster",
            "waveform_panel_metadata_is_bound_but_png_pixels_are_not_materialized",
        ],
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = _canonical_sha256(
        {"binding_domain": SCHEMA_VERSION, "value": receipt}
    )
    _atomic_json(output / "smoke_receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--adaptive-search", type=Path, required=True)
    parser.add_argument("--candidate-roster", type=Path, required=True)
    parser.add_argument("--detection-manifest", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onset-fir-numtaps", type=int, default=1201)
    parser.add_argument("--change-score-threshold", type=float, default=3.5)
    return parser


def main() -> None:
    receipt = materialize(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema_version": receipt["schema_version"],
                "event_id": receipt["input_bindings"]["event_id"],
                "receipt_sha256": receipt["receipt_sha256"],
                "ledger_outcome_status": receipt["event_to_record"][
                    "ledger_outcome_status"
                ],
                "record_phenotype_candidate": receipt["event_to_record"][
                    "record_phenotype_candidate"
                ],
                "formal_report_authorized": receipt["event_to_record"][
                    "formal_report_authorized"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
