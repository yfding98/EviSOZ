"""Canonical, onset-safe Findings-v3 to common17 record-evidence adapter.

The adapter is deliberately a serialization and permission boundary.  It does
not measure EEG, infer a channel rank, calibrate a score, or lexicalize a
diagnosis.  A separately content-bound K3 spatial producer supplies exact
common17 and independent pattern-state support.  This module verifies that the
producer is trusted, binds it to one fully validated ``event_eeg_findings_v3``
graph, and requires every positive spatial rank to be authorized by the
selected future-free cerebral-ictal hypothesis.

Only uncalibrated support is accepted in v5.  The output therefore enters the
record aggregator as ``uncalibrated_nonnegative_score`` and cannot be called a
probability.  FZ/PZ are absent, nonlocalized state mass is never mapped to CZ,
and late-course samples cannot increase positive onset rank.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .common17_record_soz_evidence_aggregation_v1 import (
    COMMON17_CHANNEL_IDS,
    INDEPENDENT_PATTERN_STATE_IDS,
    NOT_APPLICABLE_NONLOCALIZED,
    UNCALIBRATED_NONNEGATIVE_SCORE,
    Common17EventQCProfileV1,
    Common17EventSOZEvidenceV1,
)
from .event_findings_v3_downstream_projection import (
    project_event_eeg_findings_v3_downstream,
)


COMMON17_FINDINGS_V3_SPATIAL_SIDECAR_SCHEMA_VERSION = (
    "clinical_eeg_common17_findings_v3_spatial_support_sidecar_v5"
)
COMMON17_FINDINGS_V3_RECORD_ADAPTER_SCHEMA_VERSION = (
    "clinical_eeg_common17_findings_v3_record_adapter_receipt_v5"
)
COMMON17_FINDINGS_V3_RECORD_ADAPTER_METHOD_ID = (
    "findings_v3_k3_permission_to_common17_uncalibrated_record_evidence_v5"
)

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_TOL = 1e-12


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _self_hash(value: Mapping[str, object], field: str, domain: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _sha256({"binding_domain": domain, "value": body})


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an opaque identifier")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def _nonnegative_mapping(
    value: object,
    identifiers: Sequence[str],
    *,
    name: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or tuple(value.keys()) != tuple(identifiers):
        raise ValueError(f"{name} must exactly cover its frozen ontology in order")
    result: dict[str, float] = {}
    for identifier in identifiers:
        raw = value[identifier]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"{name}.{identifier} must be numeric")
        score = float(raw)
        if not math.isfinite(score) or score < 0.0:
            raise ValueError(f"{name} must contain finite nonnegative scores")
        result[identifier] = score
    if math.fsum(result.values()) <= 0.0:
        raise ValueError(f"{name} must contain positive support")
    return result


def _projection_and_selected_support(
    source_event_findings_v3: object,
    *,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ),
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    projection = project_event_eeg_findings_v3_downstream(
        source_event_findings_v3,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    source = deepcopy(source_event_findings_v3)
    selected_id = source["competing_hypotheses"]["selected_hypothesis_id"]
    selected = next(
        (
            row
            for row in source["competing_hypotheses"]["hypotheses"]
            if row["hypothesis_id"] == selected_id
        ),
        None,
    )
    if selected is None:
        return projection, source, (), ()
    supporting = tuple(str(item) for item in selected["supporting_evidence_ids"])
    permitted = [
        row
        for row in projection["concept_claims"]
        if row["concept"] == "competing_hypothesis"
        and row["value"].get("selected") is True
        and row["positive_onset_support_permitted"] is True
    ]
    if len(permitted) != 1:
        return projection, source, supporting, ()
    dependency_hashes: set[str] = set()
    finding_map = {str(row["evidence_id"]): row for row in source["findings"]}
    waveform_map = {
        str(row["waveform_evidence_id"]): row for row in source["waveform_evidence"]
    }
    for evidence_id in supporting:
        finding = finding_map[evidence_id]
        dependencies = []
        for measurement in finding["measurements"]:
            dependency = measurement["source_binding"]["raw_sample_dependency"]
            if dependency is not None:
                dependencies.append(dependency)
        for waveform_id in finding["waveform_evidence_ids"]:
            dependency = waveform_map[str(waveform_id)]["raw_sample_dependency"]
            if dependency is not None:
                dependencies.append(dependency)
        if not dependencies:
            raise ValueError("selected onset evidence lacks a raw dependency")
        for dependency in dependencies:
            if (
                dependency["view_role"] != "onset_causal"
                or dependency["dependency_policy"] != "past_and_present_only"
                or dependency["future_sample_access"] is not False
                or dependency["onset_evidence_authorized"] is not True
                or dependency["onset_support_eligible"] is not True
            ):
                raise ValueError("selected onset evidence is not future-free and causal")
            dependency_hashes.add(
                _digest(dependency["dependency_sha256"], "raw dependency")
            )
    return projection, source, supporting, tuple(sorted(dependency_hashes))


def build_common17_findings_v3_spatial_sidecar_v5(
    *,
    source_event_findings_v3: object,
    producer_id: str,
    producer_artifact_sha256: str,
    mode_id: str,
    channel_scores: Mapping[str, float] | None,
    independent_pattern_state_scores: Mapping[str, float],
    model_reliability: float,
    qc: Mapping[str, float],
    trusted_spatial_producer_registry: Mapping[str, str],
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Build a content-bound sidecar from caller-supplied K3 support scores."""

    projection, source, supporting, dependency_hashes = (
        _projection_and_selected_support(
            source_event_findings_v3,
            trusted_producer_receipts=trusted_producer_receipts,
            trusted_calibration_receipts=trusted_calibration_receipts,
            trusted_capability_qualification_receipts=(
                trusted_capability_qualification_receipts
            ),
            trusted_sensitivity_receipts=trusted_sensitivity_receipts,
            trusted_term_decision_receipts=trusted_term_decision_receipts,
            trusted_registry_bindings=trusted_registry_bindings,
        )
    )
    producer_id = _identifier(producer_id, "producer_id")
    producer_artifact_sha256 = _digest(
        producer_artifact_sha256, "producer_artifact_sha256"
    )
    if trusted_spatial_producer_registry.get(producer_id) != producer_artifact_sha256:
        raise ValueError("spatial producer is not host-trusted")
    if not supporting or not dependency_hashes:
        raise ValueError("a v5 K3 sidecar requires one selected causal hypothesis")
    state = _nonnegative_mapping(
        independent_pattern_state_scores,
        INDEPENDENT_PATTERN_STATE_IDS,
        name="independent_pattern_state_scores",
    )
    if channel_scores is None:
        channel_axis = {
            "value_semantics": NOT_APPLICABLE_NONLOCALIZED,
            "scores": None,
        }
        if state["localized_scalp_onset"] > _TOL:
            raise ValueError("nonlocalized sidecar cannot carry localized state mass")
    else:
        channel_axis = {
            "value_semantics": UNCALIBRATED_NONNEGATIVE_SCORE,
            "scores": _nonnegative_mapping(
                channel_scores, COMMON17_CHANNEL_IDS, name="channel_scores"
            ),
        }
        if state["localized_scalp_onset"] <= _TOL:
            raise ValueError("channel support requires localized state mass")
    required_qc = (
        "signal_valid_fraction",
        "common17_channel_coverage_fraction",
        "artifact_free_fraction",
        "reference_stability",
        "onset_boundary_support",
        "adaptive_support_coverage",
    )
    if not isinstance(qc, Mapping) or tuple(qc.keys()) != required_qc:
        raise ValueError("qc must exactly cover the common17 record QC profile")
    qc_values = {key: _unit_interval(qc[key], f"qc.{key}") for key in required_qc}
    result: dict[str, Any] = {
        "schema_version": COMMON17_FINDINGS_V3_SPATIAL_SIDECAR_SCHEMA_VERSION,
        "event_id": str(source["event_id"]),
        "source_event_findings_v3_sha256": projection["source_binding"][
            "source_event_findings_v3_sha256"
        ],
        "source_downstream_projection_sha256": projection["projection_sha256"],
        "producer_id": producer_id,
        "producer_artifact_sha256": producer_artifact_sha256,
        "mode_id": _identifier(mode_id, "mode_id"),
        "channel_axis": channel_axis,
        "independent_pattern_state_axis": {
            "value_semantics": UNCALIBRATED_NONNEGATIVE_SCORE,
            "scores": state,
        },
        "K3_permission": {
            "onset_evidence_ids": list(supporting),
            "raw_dependency_sha256s": list(dependency_hashes),
            "future_samples_used_for_positive_rank": False,
            "late_course_used_to_increase_positive_rank": False,
            "late_suffix_invariance_passed": True,
        },
        "model_reliability": _unit_interval(model_reliability, "model_reliability"),
        "qc": qc_values,
        "labels_or_external_context_present": False,
        "sidecar_sha256": "",
    }
    result["sidecar_sha256"] = _self_hash(
        result,
        "sidecar_sha256",
        "clinical-eeg-common17-findings-v3-spatial-sidecar-v5",
    )
    return result


def validate_common17_findings_v3_spatial_sidecar_v5(
    value: object,
    *,
    source_event_findings_v3: object,
    trusted_spatial_producer_registry: Mapping[str, str],
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Validate the exact sidecar and replay all Findings/K3 bindings."""

    if type(value) is not dict:
        raise TypeError("v5 spatial sidecar must be an object")
    candidate = deepcopy(value)
    expected_top = {
        "schema_version", "event_id", "source_event_findings_v3_sha256",
        "source_downstream_projection_sha256", "producer_id",
        "producer_artifact_sha256", "mode_id", "channel_axis",
        "independent_pattern_state_axis", "K3_permission", "model_reliability",
        "qc", "labels_or_external_context_present", "sidecar_sha256",
    }
    if set(candidate) != expected_top:
        raise ValueError("v5 spatial sidecar fields drifted")
    if candidate["schema_version"] != COMMON17_FINDINGS_V3_SPATIAL_SIDECAR_SCHEMA_VERSION:
        raise ValueError("unexpected v5 spatial sidecar schema")
    observed_hash = _digest(candidate["sidecar_sha256"], "sidecar_sha256")
    expected_hash = _self_hash(
        candidate,
        "sidecar_sha256",
        "clinical-eeg-common17-findings-v3-spatial-sidecar-v5",
    )
    if observed_hash != expected_hash:
        raise ValueError("v5 spatial sidecar self hash mismatch")
    projection, source, supporting, dependency_hashes = (
        _projection_and_selected_support(
            source_event_findings_v3,
            trusted_producer_receipts=trusted_producer_receipts,
            trusted_calibration_receipts=trusted_calibration_receipts,
            trusted_capability_qualification_receipts=(
                trusted_capability_qualification_receipts
            ),
            trusted_sensitivity_receipts=trusted_sensitivity_receipts,
            trusted_term_decision_receipts=trusted_term_decision_receipts,
            trusted_registry_bindings=trusted_registry_bindings,
        )
    )
    if candidate["event_id"] != source["event_id"]:
        raise ValueError("v5 sidecar event/source mismatch")
    if candidate["source_event_findings_v3_sha256"] != projection["source_binding"]["source_event_findings_v3_sha256"]:
        raise ValueError("v5 sidecar Findings hash mismatch")
    if candidate["source_downstream_projection_sha256"] != projection["projection_sha256"]:
        raise ValueError("v5 sidecar projection hash mismatch")
    producer_id = _identifier(candidate["producer_id"], "producer_id")
    producer_hash = _digest(candidate["producer_artifact_sha256"], "producer hash")
    if trusted_spatial_producer_registry.get(producer_id) != producer_hash:
        raise ValueError("spatial producer is not host-trusted")
    _identifier(candidate["mode_id"], "mode_id")
    if candidate["labels_or_external_context_present"] is not False:
        raise ValueError("labels and external context are forbidden in the v5 adapter")

    permission = candidate["K3_permission"]
    if set(permission) != {
        "onset_evidence_ids", "raw_dependency_sha256s",
        "future_samples_used_for_positive_rank",
        "late_course_used_to_increase_positive_rank",
        "late_suffix_invariance_passed",
    }:
        raise ValueError("v5 K3 permission fields drifted")
    if tuple(permission["onset_evidence_ids"]) != supporting:
        raise ValueError("v5 K3 evidence does not match the selected causal hypothesis")
    if tuple(permission["raw_dependency_sha256s"]) != dependency_hashes:
        raise ValueError("v5 K3 raw dependency hashes drifted")
    if permission["future_samples_used_for_positive_rank"] is not False:
        raise ValueError("future samples entered positive onset rank")
    if permission["late_course_used_to_increase_positive_rank"] is not False:
        raise ValueError("late course increased positive onset rank")
    if permission["late_suffix_invariance_passed"] is not True:
        raise ValueError("late-suffix invariance did not pass")

    state_axis = candidate["independent_pattern_state_axis"]
    if set(state_axis) != {"value_semantics", "scores"} or state_axis["value_semantics"] != UNCALIBRATED_NONNEGATIVE_SCORE:
        raise ValueError("v5 state axis must be uncalibrated support")
    state = _nonnegative_mapping(
        state_axis["scores"],
        INDEPENDENT_PATTERN_STATE_IDS,
        name="independent_pattern_state_scores",
    )
    channel_axis = candidate["channel_axis"]
    if set(channel_axis) != {"value_semantics", "scores"}:
        raise ValueError("v5 channel axis fields drifted")
    if channel_axis["value_semantics"] == NOT_APPLICABLE_NONLOCALIZED:
        if channel_axis["scores"] is not None or state["localized_scalp_onset"] > _TOL:
            raise ValueError("nonlocalized v5 sidecar carries channel/localized mass")
    elif channel_axis["value_semantics"] == UNCALIBRATED_NONNEGATIVE_SCORE:
        _nonnegative_mapping(
            channel_axis["scores"], COMMON17_CHANNEL_IDS, name="channel_scores"
        )
        if not supporting or not dependency_hashes:
            raise ValueError("localized channel rank lacks a selected causal hypothesis")
        if state["localized_scalp_onset"] <= _TOL:
            raise ValueError("channel rank lacks localized pattern-state mass")
    else:
        raise ValueError("v5 channel axis has unsupported semantics")

    required_qc = (
        "signal_valid_fraction", "common17_channel_coverage_fraction",
        "artifact_free_fraction", "reference_stability",
        "onset_boundary_support", "adaptive_support_coverage",
    )
    if not isinstance(candidate["qc"], Mapping) or tuple(candidate["qc"].keys()) != required_qc:
        raise ValueError("v5 sidecar QC profile drifted")
    for key in required_qc:
        _unit_interval(candidate["qc"][key], f"qc.{key}")
    _unit_interval(candidate["model_reliability"], "model_reliability")
    return candidate


def adapt_common17_findings_v3_to_record_event_v5(
    *,
    source_event_findings_v3: object,
    spatial_sidecar: object,
    trusted_spatial_producer_registry: Mapping[str, str],
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[Common17EventSOZEvidenceV1, dict[str, Any]]:
    """Create one typed record-aggregation event and a replayable adapter receipt."""

    sidecar = validate_common17_findings_v3_spatial_sidecar_v5(
        spatial_sidecar,
        source_event_findings_v3=source_event_findings_v3,
        trusted_spatial_producer_registry=trusted_spatial_producer_registry,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    channel_axis = sidecar["channel_axis"]
    channel_scores = channel_axis["scores"]
    state_scores = sidecar["independent_pattern_state_axis"]["scores"]
    qc = Common17EventQCProfileV1(**sidecar["qc"])
    event = Common17EventSOZEvidenceV1(
        event_id=sidecar["event_id"],
        source_event_evidence_sha256=sidecar["sidecar_sha256"],
        mode_id=sidecar["mode_id"],
        channel_values=(
            None
            if channel_scores is None
            else tuple(float(channel_scores[item]) for item in COMMON17_CHANNEL_IDS)
        ),
        channel_value_semantics=channel_axis["value_semantics"],
        state_values=tuple(
            float(state_scores[item]) for item in INDEPENDENT_PATTERN_STATE_IDS
        ),
        state_value_semantics=UNCALIBRATED_NONNEGATIVE_SCORE,
        model_reliability=float(sidecar["model_reliability"]),
        qc=qc,
        onset_evidence_ids=tuple(sidecar["K3_permission"]["onset_evidence_ids"]),
        labels_or_external_context_present=False,
    )
    receipt: dict[str, Any] = {
        "schema_version": COMMON17_FINDINGS_V3_RECORD_ADAPTER_SCHEMA_VERSION,
        "method_id": COMMON17_FINDINGS_V3_RECORD_ADAPTER_METHOD_ID,
        "status": "adapted_uncalibrated_onset_safe_event_support",
        "event_id": event.event_id,
        "source_event_findings_v3_sha256": sidecar[
            "source_event_findings_v3_sha256"
        ],
        "source_downstream_projection_sha256": sidecar[
            "source_downstream_projection_sha256"
        ],
        "source_spatial_sidecar_sha256": sidecar["sidecar_sha256"],
        "record_event_content_sha256": event.content_sha256,
        "value_semantics": {
            "channel": event.channel_value_semantics,
            "independent_pattern_state": event.state_value_semantics,
            "probability_language_authorized": False,
        },
        "permissions": {
            "future_samples_used_for_positive_rank": False,
            "late_course_used_to_increase_positive_rank": False,
            "late_suffix_invariance_passed": True,
            "FZ_or_PZ_present": False,
            "nonlocalized_state_mapped_to_CZ": False,
            "labels_or_external_context_used": False,
        },
        "claim_limits": {
            "clinical_SOZ_or_diagnosis_authorized": False,
            "calibrated_probability_available": False,
            "real_EEG_or_patient_level_efficacy_estimated": False,
        },
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = _self_hash(
        receipt,
        "receipt_sha256",
        "clinical-eeg-common17-findings-v3-record-adapter-receipt-v5",
    )
    return event, receipt


def validate_common17_findings_v3_record_adapter_receipt_v5(
    value: object,
    *,
    source_event_findings_v3: object,
    spatial_sidecar: object,
    trusted_spatial_producer_registry: Mapping[str, str],
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Replay the adapter and reject any receipt drift."""

    if type(value) is not dict:
        raise TypeError("v5 adapter receipt must be an object")
    _, expected = adapt_common17_findings_v3_to_record_event_v5(
        source_event_findings_v3=source_event_findings_v3,
        spatial_sidecar=spatial_sidecar,
        trusted_spatial_producer_registry=trusted_spatial_producer_registry,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    if value != expected:
        raise ValueError("v5 adapter receipt does not replay from Findings and sidecar")
    return deepcopy(expected)


__all__ = [
    "COMMON17_FINDINGS_V3_RECORD_ADAPTER_METHOD_ID",
    "COMMON17_FINDINGS_V3_RECORD_ADAPTER_SCHEMA_VERSION",
    "COMMON17_FINDINGS_V3_SPATIAL_SIDECAR_SCHEMA_VERSION",
    "adapt_common17_findings_v3_to_record_event_v5",
    "build_common17_findings_v3_spatial_sidecar_v5",
    "validate_common17_findings_v3_record_adapter_receipt_v5",
    "validate_common17_findings_v3_spatial_sidecar_v5",
]
