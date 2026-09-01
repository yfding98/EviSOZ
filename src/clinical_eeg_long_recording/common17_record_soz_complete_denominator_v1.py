"""Additive complete-denominator wrapper for common17 record SOZ evidence.

The existing :mod:`common17_record_soz_evidence_aggregation_v1` core is kept
unchanged and remains the numerical owner when at least one detector event has
typed event evidence.  This module supplies the missing record-level terminal
states around that core:

* a completed detector run with zero candidates is a valid, typed unresolved
  output rather than an exception;
* an all-background-censored event roster remains in the denominator and
  yields a typed unresolved output;
* detector or downstream technical failure remains a distinct typed failure,
  never relabelled as zero-candidate or EEG-negative;
* an EEG-measurable record may retain a separately labelled, uncalibrated
  research channel ranking, without promoting it into the primary diagnosis.

The independent bilateral/generalized/unresolved state axis is never projected
onto CZ.  CZ may occur in the optional research ranking only through its
directly observed common17 signal component; FZ/PZ never enter the prediction
axis and no prediction score is remapped to CZ.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .common17_record_soz_evidence_aggregation_v1 import (
    COMMON17_CHANNEL_IDS,
    INDEPENDENT_PATTERN_STATE_IDS,
    NORMALIZED_SUPPORT_SCORE,
    UNCALIBRATED_NONNEGATIVE_SCORE,
    validate_common17_record_soz_evidence_aggregation_v1,
)


SCHEMA_VERSION = "clinical_eeg_common17_record_soz_complete_denominator_v1"
METHOD_ID = "common17_record_soz_complete_denominator_wrapper_v1"

DETECTOR_COMPLETED_WITH_CANDIDATES = "completed_with_candidates"
DETECTOR_COMPLETED_ZERO_CANDIDATE = "completed_zero_candidate"
DETECTOR_TECHNICAL_FAILURE = "technical_failure"
DETECTOR_TERMINAL_OUTCOMES = (
    DETECTOR_COMPLETED_WITH_CANDIDATES,
    DETECTOR_COMPLETED_ZERO_CANDIDATE,
    DETECTOR_TECHNICAL_FAILURE,
)

EVENT_EVIDENCE_AVAILABLE = "evidence_available"
EVENT_BACKGROUND_CENSORED = "background_censored"
EVENT_NO_QUALIFIED_EEG_CHANGE = "no_qualified_eeg_change"
EVENT_TECHNICAL_FAILURE = "technical_failure"
EVENT_ANALYSIS_OUTCOMES = (
    EVENT_EVIDENCE_AVAILABLE,
    EVENT_BACKGROUND_CENSORED,
    EVENT_NO_QUALIFIED_EEG_CHANGE,
    EVENT_TECHNICAL_FAILURE,
)

DISPOSITION_EVIDENCE_AVAILABLE = "eeg_evidence_available"
DISPOSITION_TYPED_UNRESOLVED = "typed_unresolved"
DISPOSITION_TYPED_TECHNICAL_FAILURE = "typed_technical_failure"

SIGNAL_MEASURABLE = "measurable"
SIGNAL_NOT_MEASURABLE = "not_measurable"
SIGNAL_MEASUREMENT_STATUSES = (SIGNAL_MEASURABLE, SIGNAL_NOT_MEASURABLE)

_CENSOR_OUTCOMES = {
    EVENT_BACKGROUND_CENSORED,
    EVENT_NO_QUALIFIED_EEG_CHANGE,
}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_REASON_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,191}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_TOL = 2e-9

_SOURCE_SCOPES = {"public_source", "synthetic", "deployment_eeg_only"}
_INFERENCE_EXCLUSIONS = {
    "EDF_annotations": False,
    "Excel_onset_fields": False,
    "doctor_labels_or_text": False,
    "clinical_history": False,
    "video_or_behavior": False,
    "sleep_staging": False,
    "provocation": False,
    "ECG_EMG_EOG": False,
    "Qwen_or_other_LLM": False,
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _self_hash(value: Mapping[str, object], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _canonical_sha256(body)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{name} is not a bounded identifier")
    return value


def _reason(value: object, name: str) -> str:
    if not isinstance(value, str) or _REASON_RE.fullmatch(value) is None:
        raise ValueError(f"{name} is not a bounded reason code")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} is not a SHA-256")
    return value


def _reason_codes(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of reason codes")
    result = tuple(_reason(value, f"{name}[{index}]") for index, value in enumerate(values))
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate reason codes")
    return result


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _normalise(values: Sequence[float]) -> tuple[float, ...]:
    converted = tuple(
        _finite_nonnegative(value, f"research channel value[{index}]")
        for index, value in enumerate(values)
    )
    if len(converted) != len(COMMON17_CHANNEL_IDS):
        raise ValueError("research ranking must use the exact common17 axis")
    total = math.fsum(converted)
    if total <= 0.0:
        raise ValueError("research ranking requires at least one positive value")
    return tuple(value / total for value in converted)


@dataclass(frozen=True)
class Common17EventAnalysisTerminalV1:
    """One detector candidate's terminal evidence-analysis state."""

    event_id: str
    terminal_receipt_sha256: str
    outcome: str
    reason_codes: tuple[str, ...] = ()
    signal_measurable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _identifier(self.event_id, "event_id"))
        object.__setattr__(
            self,
            "terminal_receipt_sha256",
            _sha256(self.terminal_receipt_sha256, "terminal_receipt_sha256"),
        )
        if self.outcome not in EVENT_ANALYSIS_OUTCOMES:
            raise ValueError("unsupported event analysis outcome")
        reasons = _reason_codes(self.reason_codes, "event reason_codes")
        if self.outcome == EVENT_EVIDENCE_AVAILABLE and reasons:
            raise ValueError("evidence-available event may not carry failure reasons")
        if self.outcome != EVENT_EVIDENCE_AVAILABLE and not reasons:
            raise ValueError("non-evaluable event requires a typed reason code")
        if not isinstance(self.signal_measurable, bool):
            raise TypeError("signal_measurable must be boolean")
        if self.outcome in _CENSOR_OUTCOMES and self.signal_measurable is not True:
            raise ValueError("EEG censor outcomes require a measurable signal")
        object.__setattr__(self, "reason_codes", reasons)

    def as_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "terminal_receipt_sha256": self.terminal_receipt_sha256,
            "outcome": self.outcome,
            "reason_codes": list(self.reason_codes),
            "signal_measurable": self.signal_measurable,
            "roster_included": True,
        }


@dataclass(frozen=True)
class Common17TechnicalFailureV1:
    """Typed record-level failure, kept separate from zero-candidate."""

    stage: str
    failure_type: str
    failure_receipt_sha256: str
    reason_codes: tuple[str, ...]
    recoverable: bool = False

    def __post_init__(self) -> None:
        if self.stage not in {
            "signal_io",
            "detector",
            "event_analysis",
            "record_aggregation",
        }:
            raise ValueError("unsupported technical failure stage")
        object.__setattr__(
            self, "failure_type", _reason(self.failure_type, "failure_type")
        )
        object.__setattr__(
            self,
            "failure_receipt_sha256",
            _sha256(self.failure_receipt_sha256, "failure_receipt_sha256"),
        )
        reasons = _reason_codes(self.reason_codes, "technical failure reason_codes")
        if not reasons:
            raise ValueError("technical failure requires at least one reason code")
        if not isinstance(self.recoverable, bool):
            raise TypeError("recoverable must be boolean")
        object.__setattr__(self, "reason_codes", reasons)

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "failure_type": self.failure_type,
            "failure_receipt_sha256": self.failure_receipt_sha256,
            "reason_codes": list(self.reason_codes),
            "recoverable": self.recoverable,
            "is_zero_candidate": False,
            "is_eeg_negative": False,
        }


@dataclass(frozen=True)
class Common17UncalibratedResearchRankingV1:
    """Optional observed-signal ranking that can never become a primary claim."""

    source_evidence_sha256: str
    channel_values: Sequence[float]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_evidence_sha256",
            _sha256(self.source_evidence_sha256, "source_evidence_sha256"),
        )
        normalised = _normalise(self.channel_values)
        object.__setattr__(self, "channel_values", normalised)
        if isinstance(self.evidence_ids, (str, bytes)) or not self.evidence_ids:
            raise ValueError("research ranking requires evidence identifiers")
        evidence_ids = tuple(
            _identifier(value, f"evidence_ids[{index}]")
            for index, value in enumerate(self.evidence_ids)
        )
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("research ranking repeats an evidence identifier")
        object.__setattr__(self, "evidence_ids", evidence_ids)

    def as_dict(self) -> dict[str, object]:
        values = {
            channel: float(self.channel_values[index])
            for index, channel in enumerate(COMMON17_CHANNEL_IDS)
        }
        ranked = sorted(
            COMMON17_CHANNEL_IDS,
            key=lambda channel: (-values[channel], COMMON17_CHANNEL_IDS.index(channel)),
        )
        return {
            "status": "available_research_only",
            "source_evidence_sha256": self.source_evidence_sha256,
            "evidence_ids": list(self.evidence_ids),
            "channel_ids": list(COMMON17_CHANNEL_IDS),
            "channel_values": values,
            "channel_ranking": [
                {
                    "rank": index + 1,
                    "candidate_id": channel,
                    "value": values[channel],
                }
                for index, channel in enumerate(ranked)
            ],
            "input_value_semantics": UNCALIBRATED_NONNEGATIVE_SCORE,
            "display_value_semantics": NORMALIZED_SUPPORT_SCORE,
            "probability_language_authorized": False,
            "clinical_or_primary_SOZ_claim_authorized": False,
            "observed_signal_axis": "common17_direct_observation_only",
            "excluded_signal_and_prediction_channels": ["FZ", "PZ"],
            "prediction_side_fz_pz_to_cz_mapping_used": False,
            "nonlocalized_or_unresolved_state_projected_to_CZ": False,
            "cz_score_origin": "observed_CZ_signal_component_only",
        }


def _unresolved_primary(subtype: str) -> dict[str, Any]:
    state = {
        "status": subtype,
        "value_semantics": "deterministic_method_disposition_not_probability",
        "probability_language_authorized": False,
        "mass_values": {
            state_id: 1.0 if state_id == "unresolved" else 0.0
            for state_id in INDEPENDENT_PATTERN_STATE_IDS
        },
        "ranking": [
            {"rank": 1, "candidate_id": "unresolved", "value": 1.0}
        ],
        "states_are_independent_from_common17_channel_axis": True,
    }
    return {
        "source": "complete_denominator_typed_terminal",
        "independent_pattern_state": state,
        "spatial_localization": _not_estimable_spatial(subtype),
    }


def _technical_primary() -> dict[str, Any]:
    return {
        "source": "complete_denominator_typed_terminal",
        "independent_pattern_state": {
            "status": "not_evaluable_technical_failure",
            "value_semantics": "not_available",
            "probability_language_authorized": False,
            "mass_values": None,
            "ranking": [],
            "states_are_independent_from_common17_channel_axis": True,
        },
        "spatial_localization": _not_estimable_spatial(
            "not_evaluable_technical_failure"
        ),
    }


def _not_estimable_spatial(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "conditioning": "localized_scalp_onset_pattern_only",
        "value_semantics": "not_available",
        "probability_language_authorized": False,
        "channel_values": None,
        "channel_ranking": [],
        "region_values": None,
        "region_ranking": [],
        "laterality_values": None,
        "laterality_ranking": [],
        "nonlocalized_states_projected_to_channels": False,
        "unresolved_state_projected_to_CZ": False,
        "cz_is_observed_midline_electrode_not_nonlocalized_bucket": True,
    }


def _aggregation_primary(aggregation: Mapping[str, Any]) -> dict[str, Any]:
    spatial = deepcopy(aggregation["spatial_localization"])
    spatial["unresolved_state_projected_to_CZ"] = False
    return {
        "source": "validated_common17_record_soz_evidence_aggregation_v1",
        "independent_pattern_state": deepcopy(
            aggregation["independent_pattern_state"]
        ),
        "spatial_localization": spatial,
    }


def materialize_common17_record_soz_complete_denominator_v1(
    *,
    record_id: str,
    canonical_signal_sha256: str,
    upstream_model_artifact_sha256: str,
    detector_prediction_receipt_sha256: str,
    detector_terminal_outcome: str,
    signal_measurement_status: str,
    detector_event_roster: Sequence[str] = (),
    event_analysis_outcomes: Sequence[Common17EventAnalysisTerminalV1] = (),
    record_aggregation: object | None = None,
    technical_failure: Common17TechnicalFailureV1 | None = None,
    research_ranking: Common17UncalibratedResearchRankingV1 | None = None,
    source_scope: str = "deployment_eeg_only",
) -> dict[str, Any]:
    """Materialize one terminal receipt for every record in the denominator."""

    record = _identifier(record_id, "record_id")
    canonical = _sha256(canonical_signal_sha256, "canonical_signal_sha256")
    upstream = _sha256(
        upstream_model_artifact_sha256, "upstream_model_artifact_sha256"
    )
    prediction = _sha256(
        detector_prediction_receipt_sha256,
        "detector_prediction_receipt_sha256",
    )
    if detector_terminal_outcome not in DETECTOR_TERMINAL_OUTCOMES:
        raise ValueError("unsupported detector terminal outcome")
    if signal_measurement_status not in SIGNAL_MEASUREMENT_STATUSES:
        raise ValueError("unsupported signal measurement status")
    if source_scope not in _SOURCE_SCOPES:
        raise ValueError("unsupported source scope")
    if isinstance(detector_event_roster, (str, bytes)):
        raise TypeError("detector_event_roster must be a sequence")
    roster = tuple(
        _identifier(value, f"detector_event_roster[{index}]")
        for index, value in enumerate(detector_event_roster)
    )
    if len(roster) != len(set(roster)):
        raise ValueError("detector event roster contains duplicate identifiers")
    if isinstance(event_analysis_outcomes, (str, bytes)) or not all(
        isinstance(row, Common17EventAnalysisTerminalV1)
        for row in event_analysis_outcomes
    ):
        raise TypeError("event_analysis_outcomes must contain typed rows")
    event_rows = tuple(event_analysis_outcomes)

    if detector_terminal_outcome == DETECTOR_COMPLETED_WITH_CANDIDATES:
        if not roster:
            raise ValueError("completed-with-candidates requires a non-empty roster")
        if tuple(row.event_id for row in event_rows) != roster:
            raise ValueError("event analysis rows must exactly close the detector roster")
        if signal_measurement_status != SIGNAL_MEASURABLE:
            raise ValueError("completed candidate inference requires measurable signal")
    else:
        if roster or event_rows:
            raise ValueError("zero-candidate/technical detector outcome must have no roster")
        if detector_terminal_outcome == DETECTOR_COMPLETED_ZERO_CANDIDATE:
            if signal_measurement_status != SIGNAL_MEASURABLE:
                raise ValueError("completed zero-candidate still requires measurable signal")
            if record_aggregation is not None:
                raise ValueError("zero-candidate record cannot carry event aggregation")

    aggregation: dict[str, Any] | None = None
    if record_aggregation is not None:
        if detector_terminal_outcome != DETECTOR_COMPLETED_WITH_CANDIDATES:
            raise ValueError("aggregation requires completed detector candidates")
        aggregation = validate_common17_record_soz_evidence_aggregation_v1(
            record_aggregation
        )
        aggregation_record = aggregation["record"]
        if (
            aggregation_record["record_id"] != record
            or aggregation_record["canonical_signal_sha256"] != canonical
            or aggregation_record["upstream_model_artifact_sha256"] != upstream
            or aggregation_record["source_scope"] != source_scope
        ):
            raise ValueError("record aggregation identity binding mismatch")
        ledger = aggregation["evidence_ledger"]
        if ledger["detector_event_roster"] != list(roster):
            raise ValueError("record aggregation does not close the detector roster")
        source_hashes = [row["source_event_evidence_sha256"] for row in ledger["event_rows"]]
        terminal_hashes = [row.terminal_receipt_sha256 for row in event_rows]
        if source_hashes != terminal_hashes:
            raise ValueError("record aggregation lost event terminal receipt binding")

    outcomes = tuple(row.outcome for row in event_rows)
    all_censored = bool(outcomes) and all(value in _CENSOR_OUTCOMES for value in outcomes)
    all_technical = bool(outcomes) and all(
        value == EVENT_TECHNICAL_FAILURE for value in outcomes
    )

    if detector_terminal_outcome == DETECTOR_TECHNICAL_FAILURE or all_technical:
        disposition = DISPOSITION_TYPED_TECHNICAL_FAILURE
        subtype = (
            "detector_technical_failure"
            if detector_terminal_outcome == DETECTOR_TECHNICAL_FAILURE
            else "all_event_analysis_technical_failure"
        )
        if technical_failure is None:
            raise ValueError("typed technical disposition requires failure details")
        if aggregation is not None:
            raise ValueError("all-technical record cannot expose a primary aggregation")
        primary = _technical_primary()
        disposition_reasons = list(technical_failure.reason_codes)
    elif detector_terminal_outcome == DETECTOR_COMPLETED_ZERO_CANDIDATE:
        disposition = DISPOSITION_TYPED_UNRESOLVED
        subtype = "zero_detector_candidate"
        if technical_failure is not None:
            raise ValueError("zero-candidate is not a technical failure")
        primary = _unresolved_primary("unresolved_zero_detector_candidate")
        disposition_reasons = ["zero_detector_candidate"]
    elif all_censored:
        disposition = DISPOSITION_TYPED_UNRESOLVED
        subtype = "all_event_evidence_censored"
        if technical_failure is not None:
            raise ValueError("background/no-change censor is not a technical failure")
        if aggregation is not None:
            spatial = aggregation["spatial_localization"]
            states = aggregation["independent_pattern_state"]["mass_values"]
            if spatial["channel_values"] is not None or not math.isclose(
                float(states["unresolved"]), 1.0, abs_tol=_TOL
            ):
                raise ValueError("all-censored aggregation fabricated localized evidence")
        primary = _unresolved_primary("unresolved_all_event_evidence_censored")
        disposition_reasons = sorted(
            {reason for row in event_rows for reason in row.reason_codes}
        )
    else:
        disposition = DISPOSITION_EVIDENCE_AVAILABLE
        subtype = "validated_multievent_aggregation"
        if aggregation is None:
            raise ValueError("evidence-available roster requires record aggregation")
        if technical_failure is not None:
            raise ValueError("partial event failures belong in the event ledger")
        primary = _aggregation_primary(aggregation)
        disposition_reasons = []
        if any(value == EVENT_TECHNICAL_FAILURE for value in outcomes):
            disposition_reasons.append("partial_event_analysis_technical_failure")
        if any(value in _CENSOR_OUTCOMES for value in outcomes):
            disposition_reasons.append("partial_event_evidence_censoring")

    if research_ranking is not None:
        if not isinstance(research_ranking, Common17UncalibratedResearchRankingV1):
            raise TypeError("research_ranking must be typed")
        if signal_measurement_status != SIGNAL_MEASURABLE:
            raise ValueError("research ranking requires a measurable EEG signal")
        research_output = research_ranking.as_dict()
    else:
        research_output = None

    detector_roster_sha256 = _canonical_sha256(
        {
            "record_id": record,
            "detector_prediction_receipt_sha256": prediction,
            "detector_terminal_outcome": detector_terminal_outcome,
            "detector_event_roster": roster,
        }
    )
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "record": {
            "record_id": record,
            "canonical_signal_sha256": canonical,
            "upstream_model_artifact_sha256": upstream,
            "source_scope": source_scope,
            "signal_measurement_status": signal_measurement_status,
            "source_inference_exclusions": deepcopy(_INFERENCE_EXCLUSIONS),
        },
        "common17_ontology": {
            "channel_ids": list(COMMON17_CHANNEL_IDS),
            "excluded_signal_and_prediction_channels": ["FZ", "PZ"],
            "missing_channel_imputation_used": False,
            "prediction_side_fz_pz_to_cz_mapping_used": False,
            "broad_generalized_or_unresolved_state_mapped_to_CZ": False,
            "cz_is_observed_midline_electrode_only": True,
        },
        "detector_inventory": {
            "terminal_outcome": detector_terminal_outcome,
            "prediction_receipt_sha256": prediction,
            "detector_event_roster": list(roster),
            "detector_event_roster_sha256": detector_roster_sha256,
            "candidate_count": len(roster),
            "zero_candidate_record": detector_terminal_outcome
            == DETECTOR_COMPLETED_ZERO_CANDIDATE,
            "technical_failure_record": detector_terminal_outcome
            == DETECTOR_TECHNICAL_FAILURE,
            "zero_candidate_is_valid_completion": True,
            "technical_failure_is_zero_candidate": False,
            "complete_denominator_record_retained": True,
            "silent_record_drop_allowed": False,
        },
        "event_analysis_ledger": {
            "rows": [row.as_dict() for row in event_rows],
            "input_detector_candidate_count": len(roster),
            "ledger_event_count": len(event_rows),
            "all_detector_candidates_accounted": True,
            "excluded_event_ids": [],
        },
        "terminal_disposition": {
            "kind": disposition,
            "subtype": subtype,
            "reason_codes": disposition_reasons,
            "record_remains_in_primary_denominator": True,
            "eeg_negative_claimed": False,
            "clinical_SOZ_claim_authorized": False,
        },
        "primary_output": primary,
        "record_aggregation": aggregation,
        "technical_failure": (
            None if technical_failure is None else technical_failure.as_dict()
        ),
        "uncalibrated_research_ranking": research_output,
        "authorization": {
            "eeg_signal_only": True,
            "uncalibrated_research_ranking_may_be_primary_diagnosis": False,
            "research_ranking_probability_language_authorized": False,
            "late_spread_may_create_positive_onset_support": False,
            "cortical_SOZ_EZ_or_surgical_target_claim_authorized": False,
            "report_lexicalization_authorized_by_this_module": False,
        },
    }
    body["receipt_sha256"] = _self_hash(body, "receipt_sha256")
    return validate_common17_record_soz_complete_denominator_v1(body)


def _require_normalized_mapping(
    value: object, expected_ids: Sequence[str], name: str
) -> None:
    if not isinstance(value, Mapping) or list(value) != list(expected_ids):
        raise ValueError(f"{name} has the wrong closed axis or order")
    numbers = [_finite_nonnegative(value[item], f"{name}.{item}") for item in expected_ids]
    if not math.isclose(math.fsum(numbers), 1.0, abs_tol=_TOL):
        raise ValueError(f"{name} must sum to one")


def validate_common17_record_soz_complete_denominator_v1(
    payload: object,
) -> dict[str, Any]:
    """Fail-closed validation of denominator retention and axis separation."""

    if type(payload) is not dict:
        raise TypeError("complete-denominator receipt must be an object")
    data = deepcopy(payload)
    required = {
        "schema_version",
        "method_id",
        "receipt_sha256",
        "record",
        "common17_ontology",
        "detector_inventory",
        "event_analysis_ledger",
        "terminal_disposition",
        "primary_output",
        "record_aggregation",
        "technical_failure",
        "uncalibrated_research_ranking",
        "authorization",
    }
    if set(data) != required:
        raise ValueError("complete-denominator receipt fields drifted")
    if data["schema_version"] != SCHEMA_VERSION or data["method_id"] != METHOD_ID:
        raise ValueError("complete-denominator schema or method drifted")

    record = data["record"]
    if not isinstance(record, Mapping):
        raise TypeError("record binding is missing")
    record_id = _identifier(record.get("record_id"), "record_id")
    canonical = _sha256(record.get("canonical_signal_sha256"), "canonical signal")
    upstream = _sha256(record.get("upstream_model_artifact_sha256"), "upstream model")
    if record.get("source_scope") not in _SOURCE_SCOPES:
        raise ValueError("record source scope drifted")
    if record.get("signal_measurement_status") not in SIGNAL_MEASUREMENT_STATUSES:
        raise ValueError("signal measurement status drifted")
    if record.get("source_inference_exclusions") != _INFERENCE_EXCLUSIONS:
        raise ValueError("EEG-only source firewall drifted")

    ontology = data["common17_ontology"]
    if not isinstance(ontology, Mapping) or ontology.get("channel_ids") != list(
        COMMON17_CHANNEL_IDS
    ):
        raise ValueError("common17 ontology drifted")
    if ontology.get("excluded_signal_and_prediction_channels") != ["FZ", "PZ"]:
        raise ValueError("FZ/PZ entered the prediction ontology")
    for field in (
        "missing_channel_imputation_used",
        "prediction_side_fz_pz_to_cz_mapping_used",
        "broad_generalized_or_unresolved_state_mapped_to_CZ",
    ):
        if ontology.get(field) is not False:
            raise ValueError("complete-denominator ontology introduced a CZ fallback")
    if ontology.get("cz_is_observed_midline_electrode_only") is not True:
        raise ValueError("CZ lost its observed-electrode semantics")

    detector = data["detector_inventory"]
    if not isinstance(detector, Mapping):
        raise TypeError("detector inventory is missing")
    outcome = detector.get("terminal_outcome")
    if outcome not in DETECTOR_TERMINAL_OUTCOMES:
        raise ValueError("detector terminal outcome drifted")
    prediction = _sha256(detector.get("prediction_receipt_sha256"), "prediction receipt")
    roster = detector.get("detector_event_roster")
    if not isinstance(roster, list) or len(roster) != len(set(roster)):
        raise ValueError("detector roster is invalid")
    for index, event_id in enumerate(roster):
        _identifier(event_id, f"detector roster[{index}]")
    expected_roster_sha = _canonical_sha256(
        {
            "record_id": record_id,
            "detector_prediction_receipt_sha256": prediction,
            "detector_terminal_outcome": outcome,
            "detector_event_roster": tuple(roster),
        }
    )
    if detector.get("detector_event_roster_sha256") != expected_roster_sha:
        raise ValueError("detector roster hash mismatch")
    if detector.get("candidate_count") != len(roster):
        raise ValueError("detector candidate count mismatch")
    expected_zero = outcome == DETECTOR_COMPLETED_ZERO_CANDIDATE
    expected_failure = outcome == DETECTOR_TECHNICAL_FAILURE
    if detector.get("zero_candidate_record") is not expected_zero or detector.get(
        "technical_failure_record"
    ) is not expected_failure:
        raise ValueError("detector terminal flags drifted")
    if any(
        detector.get(field) is not expected
        for field, expected in (
            ("zero_candidate_is_valid_completion", True),
            ("technical_failure_is_zero_candidate", False),
            ("complete_denominator_record_retained", True),
            ("silent_record_drop_allowed", False),
        )
    ):
        raise ValueError("complete denominator detector guarantees drifted")
    if outcome == DETECTOR_COMPLETED_WITH_CANDIDATES and not roster:
        raise ValueError("completed-with-candidates receipt has an empty roster")
    if outcome != DETECTOR_COMPLETED_WITH_CANDIDATES and roster:
        raise ValueError("zero/technical detector receipt carries candidates")
    if outcome in {
        DETECTOR_COMPLETED_WITH_CANDIDATES,
        DETECTOR_COMPLETED_ZERO_CANDIDATE,
    } and record.get("signal_measurement_status") != SIGNAL_MEASURABLE:
        raise ValueError("completed detector outcome lacks a measurable signal")

    ledger = data["event_analysis_ledger"]
    if not isinstance(ledger, Mapping) or not isinstance(ledger.get("rows"), list):
        raise TypeError("event analysis ledger is missing")
    rows = ledger["rows"]
    if [row.get("event_id") for row in rows if isinstance(row, Mapping)] != roster:
        raise ValueError("event analysis ledger does not close the detector roster")
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("event analysis row is invalid")
        expected_fields = {
            "event_id",
            "terminal_receipt_sha256",
            "outcome",
            "reason_codes",
            "signal_measurable",
            "roster_included",
        }
        if set(row) != expected_fields:
            raise ValueError("event analysis row fields drifted")
        _identifier(row["event_id"], "event analysis event_id")
        _sha256(row["terminal_receipt_sha256"], "event terminal receipt")
        if row["outcome"] not in EVENT_ANALYSIS_OUTCOMES:
            raise ValueError("event analysis outcome drifted")
        reasons = _reason_codes(row["reason_codes"], "event analysis reasons")
        if (row["outcome"] == EVENT_EVIDENCE_AVAILABLE) is bool(reasons):
            raise ValueError("event analysis reason/outcome semantics drifted")
        if row["roster_included"] is not True or not isinstance(
            row["signal_measurable"], bool
        ):
            raise ValueError("event analysis roster/measurability drifted")
        if row["outcome"] in _CENSOR_OUTCOMES and row["signal_measurable"] is not True:
            raise ValueError("EEG censor row lacks a measurable signal")
    if any(
        ledger.get(field) != expected
        for field, expected in (
            ("input_detector_candidate_count", len(roster)),
            ("ledger_event_count", len(rows)),
            ("all_detector_candidates_accounted", True),
            ("excluded_event_ids", []),
        )
    ):
        raise ValueError("event analysis ledger counts drifted")

    aggregation = data["record_aggregation"]
    if aggregation is not None:
        aggregation = validate_common17_record_soz_evidence_aggregation_v1(
            aggregation
        )
        aggregation_record = aggregation["record"]
        if (
            aggregation_record["record_id"] != record_id
            or aggregation_record["canonical_signal_sha256"] != canonical
            or aggregation_record["upstream_model_artifact_sha256"] != upstream
            or aggregation_record["source_scope"] != record["source_scope"]
            or aggregation["evidence_ledger"]["detector_event_roster"] != roster
        ):
            raise ValueError("embedded aggregation binding drifted")
        if [
            row["source_event_evidence_sha256"]
            for row in aggregation["evidence_ledger"]["event_rows"]
        ] != [row["terminal_receipt_sha256"] for row in rows]:
            raise ValueError("embedded aggregation event hashes drifted")

    disposition = data["terminal_disposition"]
    if not isinstance(disposition, Mapping) or disposition.get("kind") not in {
        DISPOSITION_EVIDENCE_AVAILABLE,
        DISPOSITION_TYPED_UNRESOLVED,
        DISPOSITION_TYPED_TECHNICAL_FAILURE,
    }:
        raise ValueError("terminal disposition drifted")
    if disposition.get("record_remains_in_primary_denominator") is not True:
        raise ValueError("terminal record left the primary denominator")
    if disposition.get("eeg_negative_claimed") is not False or disposition.get(
        "clinical_SOZ_claim_authorized"
    ) is not False:
        raise ValueError("terminal disposition made an unauthorized claim")
    disposition_reasons = _reason_codes(
        disposition.get("reason_codes", []), "disposition reasons"
    )
    row_outcomes = tuple(row["outcome"] for row in rows)
    all_censored = bool(row_outcomes) and all(
        value in _CENSOR_OUTCOMES for value in row_outcomes
    )
    all_technical = bool(row_outcomes) and all(
        value == EVENT_TECHNICAL_FAILURE for value in row_outcomes
    )
    if outcome == DETECTOR_TECHNICAL_FAILURE:
        expected_kind = DISPOSITION_TYPED_TECHNICAL_FAILURE
        expected_subtype = "detector_technical_failure"
    elif outcome == DETECTOR_COMPLETED_ZERO_CANDIDATE:
        expected_kind = DISPOSITION_TYPED_UNRESOLVED
        expected_subtype = "zero_detector_candidate"
        if disposition_reasons != ("zero_detector_candidate",):
            raise ValueError("zero-candidate disposition reasons drifted")
    elif all_technical:
        expected_kind = DISPOSITION_TYPED_TECHNICAL_FAILURE
        expected_subtype = "all_event_analysis_technical_failure"
    elif all_censored:
        expected_kind = DISPOSITION_TYPED_UNRESOLVED
        expected_subtype = "all_event_evidence_censored"
    else:
        expected_kind = DISPOSITION_EVIDENCE_AVAILABLE
        expected_subtype = "validated_multievent_aggregation"
    if disposition["kind"] != expected_kind or disposition.get("subtype") != (
        expected_subtype
    ):
        raise ValueError("terminal disposition is inconsistent with terminal outcomes")

    failure = data["technical_failure"]
    if disposition["kind"] == DISPOSITION_TYPED_TECHNICAL_FAILURE:
        if not isinstance(failure, Mapping) or failure.get("is_zero_candidate") is not False:
            raise ValueError("technical disposition lacks typed failure details")
        if failure.get("is_eeg_negative") is not False:
            raise ValueError("technical failure was relabelled EEG-negative")
        if failure.get("stage") not in {
            "signal_io",
            "detector",
            "event_analysis",
            "record_aggregation",
        }:
            raise ValueError("technical failure stage drifted")
        _reason(failure.get("failure_type"), "failure_type")
        _sha256(failure.get("failure_receipt_sha256"), "failure receipt")
        failure_reasons = _reason_codes(
            failure.get("reason_codes", []), "technical failure reasons"
        )
        if not failure_reasons:
            raise ValueError("technical failure lacks reason codes")
        if not isinstance(failure.get("recoverable"), bool):
            raise ValueError("technical failure recoverability drifted")
        if disposition_reasons != failure_reasons:
            raise ValueError("technical disposition lost failure reason binding")
    elif failure is not None:
        raise ValueError("nontechnical disposition carries technical failure details")

    primary = data["primary_output"]
    if not isinstance(primary, Mapping):
        raise TypeError("primary output is missing")
    state = primary.get("independent_pattern_state")
    spatial = primary.get("spatial_localization")
    if not isinstance(state, Mapping) or not isinstance(spatial, Mapping):
        raise TypeError("primary state/spatial output is missing")
    if state.get("states_are_independent_from_common17_channel_axis") is not True:
        raise ValueError("primary pattern state leaked into the channel axis")
    if spatial.get("nonlocalized_states_projected_to_channels") is not False or spatial.get(
        "unresolved_state_projected_to_CZ"
    ) is not False or spatial.get(
        "cz_is_observed_midline_electrode_not_nonlocalized_bucket"
    ) is not True:
        raise ValueError("primary nonlocalized state was mapped to CZ")

    if disposition["kind"] == DISPOSITION_EVIDENCE_AVAILABLE:
        if aggregation is None or primary.get("source") != (
            "validated_common17_record_soz_evidence_aggregation_v1"
        ):
            raise ValueError("evidence disposition lacks its validated aggregation")
        expected_primary = _aggregation_primary(aggregation)
        if primary != expected_primary:
            raise ValueError("primary output drifted from validated aggregation")
    elif disposition["kind"] == DISPOSITION_TYPED_UNRESOLVED:
        mass = state.get("mass_values")
        _require_normalized_mapping(
            mass, INDEPENDENT_PATTERN_STATE_IDS, "typed unresolved state"
        )
        if mass.get("unresolved") != 1.0 or any(
            spatial.get(field) is not None
            for field in ("channel_values", "region_values", "laterality_values")
        ) or any(
            spatial.get(field) != []
            for field in ("channel_ranking", "region_ranking", "laterality_ranking")
        ):
            raise ValueError("typed unresolved output fabricated spatial evidence")
        if all_censored and aggregation is not None:
            aggregation_spatial = aggregation["spatial_localization"]
            aggregation_states = aggregation["independent_pattern_state"][
                "mass_values"
            ]
            if aggregation_spatial["channel_values"] is not None or not math.isclose(
                float(aggregation_states["unresolved"]), 1.0, abs_tol=_TOL
            ):
                raise ValueError("all-censored aggregation fabricated localized evidence")
    else:
        if state.get("mass_values") is not None or any(
            spatial.get(field) is not None
            for field in ("channel_values", "region_values", "laterality_values")
        ) or any(
            spatial.get(field) != []
            for field in ("channel_ranking", "region_ranking", "laterality_ranking")
        ):
            raise ValueError("technical failure fabricated EEG state/spatial evidence")

    research = data["uncalibrated_research_ranking"]
    if research is not None:
        if record["signal_measurement_status"] != SIGNAL_MEASURABLE:
            raise ValueError("research ranking survived an unmeasurable signal")
        expected_research_fields = {
            "status",
            "source_evidence_sha256",
            "evidence_ids",
            "channel_ids",
            "channel_values",
            "channel_ranking",
            "input_value_semantics",
            "display_value_semantics",
            "probability_language_authorized",
            "clinical_or_primary_SOZ_claim_authorized",
            "observed_signal_axis",
            "excluded_signal_and_prediction_channels",
            "prediction_side_fz_pz_to_cz_mapping_used",
            "nonlocalized_or_unresolved_state_projected_to_CZ",
            "cz_score_origin",
        }
        if not isinstance(research, Mapping) or set(research) != expected_research_fields:
            raise ValueError("research ranking fields drifted")
        if research["channel_ids"] != list(COMMON17_CHANNEL_IDS):
            raise ValueError("research ranking left common17")
        _sha256(research.get("source_evidence_sha256"), "research source evidence")
        evidence_ids = research.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise ValueError("research ranking lacks evidence identifiers")
        for index, evidence_id in enumerate(evidence_ids):
            _identifier(evidence_id, f"research evidence_ids[{index}]")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("research ranking repeats evidence identifiers")
        _require_normalized_mapping(
            research["channel_values"],
            COMMON17_CHANNEL_IDS,
            "research channel values",
        )
        expected_ranking = sorted(
            COMMON17_CHANNEL_IDS,
            key=lambda channel: (
                -float(research["channel_values"][channel]),
                COMMON17_CHANNEL_IDS.index(channel),
            ),
        )
        if [row.get("candidate_id") for row in research["channel_ranking"]] != expected_ranking:
            raise ValueError("research channel ranking is inconsistent with values")
        if [row.get("rank") for row in research["channel_ranking"]] != list(
            range(1, len(COMMON17_CHANNEL_IDS) + 1)
        ) or any(
            not math.isclose(
                float(row.get("value")),
                float(research["channel_values"][row["candidate_id"]]),
                abs_tol=_TOL,
            )
            for row in research["channel_ranking"]
        ):
            raise ValueError("research ranking rows drifted from values")
        if any(
            research.get(field) is not expected
            for field, expected in (
                ("probability_language_authorized", False),
                ("clinical_or_primary_SOZ_claim_authorized", False),
                ("prediction_side_fz_pz_to_cz_mapping_used", False),
                ("nonlocalized_or_unresolved_state_projected_to_CZ", False),
            )
        ):
            raise ValueError("research ranking was promoted or mapped to CZ")
        if research.get("input_value_semantics") != UNCALIBRATED_NONNEGATIVE_SCORE or research.get(
            "display_value_semantics"
        ) != NORMALIZED_SUPPORT_SCORE:
            raise ValueError("research ranking calibration semantics drifted")
        if (
            research.get("status") != "available_research_only"
            or research.get("observed_signal_axis")
            != "common17_direct_observation_only"
            or research.get("excluded_signal_and_prediction_channels")
            != ["FZ", "PZ"]
            or research.get("cz_score_origin")
            != "observed_CZ_signal_component_only"
        ):
            raise ValueError("research ranking signal provenance drifted")

    authorization = data["authorization"]
    expected_authorization = {
        "eeg_signal_only": True,
        "uncalibrated_research_ranking_may_be_primary_diagnosis": False,
        "research_ranking_probability_language_authorized": False,
        "late_spread_may_create_positive_onset_support": False,
        "cortical_SOZ_EZ_or_surgical_target_claim_authorized": False,
        "report_lexicalization_authorized_by_this_module": False,
    }
    if authorization != expected_authorization:
        raise ValueError("complete-denominator authorization drifted")
    if data["receipt_sha256"] != _self_hash(data, "receipt_sha256"):
        raise ValueError("complete-denominator receipt hash mismatch")
    return data


__all__ = [
    "DETECTOR_COMPLETED_WITH_CANDIDATES",
    "DETECTOR_COMPLETED_ZERO_CANDIDATE",
    "DETECTOR_TECHNICAL_FAILURE",
    "DISPOSITION_EVIDENCE_AVAILABLE",
    "DISPOSITION_TYPED_TECHNICAL_FAILURE",
    "DISPOSITION_TYPED_UNRESOLVED",
    "EVENT_BACKGROUND_CENSORED",
    "EVENT_EVIDENCE_AVAILABLE",
    "EVENT_NO_QUALIFIED_EEG_CHANGE",
    "EVENT_TECHNICAL_FAILURE",
    "METHOD_ID",
    "SCHEMA_VERSION",
    "SIGNAL_MEASURABLE",
    "SIGNAL_NOT_MEASURABLE",
    "Common17EventAnalysisTerminalV1",
    "Common17TechnicalFailureV1",
    "Common17UncalibratedResearchRankingV1",
    "materialize_common17_record_soz_complete_denominator_v1",
    "validate_common17_record_soz_complete_denominator_v1",
]
