"""Target-blind association of coarse proposals with one detector occurrence.

This additive engineering component connects three already-frozen artifacts:

* a validated continuous coarse sentinel receipt;
* one target-blind detector occurrence group containing only a frozen envelope,
  uncalibrated posterior summaries and anchor uncertainty; and
* deterministic measurements made after the sentinel's native-query proposals.

Every sentinel proposal remains in the output candidate ledger.  A small,
deterministic, censor-aware segmental rule baseline scores monotone single-bout
``S0 -> S1 -> S2 -> S3`` paths, then combines that score with detector-envelope
compatibility.  The result is an association *proposal* with top-K and explicit
ambiguity; it is not a seizure, onset, Finding, channel rank, SOZ or report
fact.  The physically earliest proposal receives no special preference.

The point anchor is retained for provenance but is not a ranking feature; the
frozen uncertainty interval is used instead.  Once two deterministic return-
compatible cells close the association prefix, later cells are excluded from
the association core.  Consequently point-anchor jitter within the same
uncertainty interval and mutations confined to the closed late suffix leave the
association-core fingerprint unchanged while the outer provenance hash changes
truthfully.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence

from .continuous_coarse_sentinel_cache_v1 import (
    METHOD_ID as SENTINEL_METHOD_ID,
    SCHEMA_VERSION as SENTINEL_SCHEMA_VERSION,
    validate_common17_continuous_coarse_sentinel_cache_v1,
)


SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_continuous_coarse_detector_occurrence_association_ledger_v1"
)
METHOD_ID: Final[str] = (
    "COMMON17-CONTINUOUS-COARSE-DETECTOR-OCCURRENCE-ASSOCIATION-RULE-V1"
)
DETECTOR_OCCURRENCE_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_target_blind_frozen_detector_occurrence_group_v1"
)
NATIVE_MEASUREMENT_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_native_query_deterministic_measurement_sidecar_v1"
)

_STATES: Final[tuple[str, ...]] = ("S0", "S1", "S2", "S3")
_CENSOR_KINDS: Final[tuple[str, ...]] = (
    "record_edge",
    "qc_gap",
    "neighbor_guard",
    "native_query_budget",
)
_CENSOR_SIDES: Final[tuple[str, ...]] = ("left", "right", "internal")
_HASH_FIELDS: Final[tuple[str, ...]] = (
    "checkpoint_sha256",
    "transform_sha256",
    "decoder_sha256",
    "prediction_sha256",
)

_DETECTOR_SCOPE: Final[dict[str, object]] = {
    "prediction_frozen_before_association": True,
    "EEG_signal_only_inference": True,
    "reference_event_or_TERM_used": False,
    "SOZ_or_channel_target_used": False,
    "EDF_annotation_or_spreadsheet_used": False,
    "doctor_clinical_text_history_video_or_behaviour_used": False,
    "sleep_activation_provocation_or_auxiliary_physiology_used": False,
    "LLM_output_used": False,
}
_MEASUREMENT_SCOPE: Final[dict[str, object]] = {
    "native_EEG_deterministic_measurements_used": True,
    "measurement_values_are_clinical_assertions": False,
    "detector_anchor_or_score_used_to_compute_measurements": False,
    "reference_event_or_TERM_used": False,
    "SOZ_or_channel_target_used": False,
    "EDF_annotation_or_spreadsheet_used": False,
    "doctor_clinical_text_history_video_or_behaviour_used": False,
    "sleep_activation_provocation_or_auxiliary_physiology_used": False,
    "LLM_output_used": False,
}
_OUTPUT_SCOPE: Final[dict[str, object]] = {
    "continuous_sentinel_proposals_used": True,
    "frozen_target_blind_detector_summary_used": True,
    "deterministic_native_measurement_sidecar_used": True,
    "reference_event_or_TERM_used": False,
    "SOZ_or_channel_target_used": False,
    "EDF_annotation_or_spreadsheet_used": False,
    "doctor_clinical_text_history_video_or_behaviour_used": False,
    "sleep_activation_provocation_or_auxiliary_physiology_used": False,
    "LLM_output_used": False,
}
_AUTHORIZATION: Final[dict[str, object]] = {
    "output_namespace": "target_blind_occurrence_association_proposal_only",
    "may_retain_and_rank_competing_association_proposals": True,
    "may_trigger_additional_native_query": True,
    "may_assert_eeg_finding": False,
    "may_assert_clinical_term": False,
    "may_assert_seizure": False,
    "may_assert_onset_or_offset": False,
    "may_rank_channels_regions_or_laterality": False,
    "may_assert_SOZ_EZ_or_diagnosis": False,
    "may_enter_clinical_report_as_fact": False,
}
_ROBUSTNESS_CONTRACT: Final[dict[str, object]] = {
    "physical_earliest_candidate_wins_unconditionally": False,
    "chronological_order_used_as_score_or_tie_breaker": False,
    "all_sentinel_candidates_retained": True,
    "point_anchor_used_for_ranking": False,
    "anchor_uncertainty_interval_used_for_ranking": True,
    "closed_late_suffix_values_used_for_association_score": False,
    "association_core_fingerprint_excludes_source_hashes_point_anchor_and_closed_suffix_values": True,
    "single_bout_reentry_allowed": False,
    "top_k_is_association_proposal_not_probability_or_fact": True,
}


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


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be a non-empty identifier")
    return value


def _hash(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _finite(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} is below its allowed range")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} exceeds its allowed range")
    return result


def _interval(
    value: object,
    field: str,
    *,
    lower: int,
    upper: int,
) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
        or not isinstance(value[0], int)
        or not isinstance(value[1], int)
        or not lower <= value[0] < value[1] <= upper
    ):
        raise ValueError(f"{field} is not a legal sample interval")
    return int(value[0]), int(value[1])


def _round(value: float, digits: int = 8) -> float:
    result = round(float(value), digits)
    return 0.0 if result == 0.0 else result


def _verify_content_hash(data: Mapping[str, Any], *, field: str = "receipt_sha256") -> None:
    supplied = _hash(data.get(field), field)
    expected = _canonical_sha256({key: value for key, value in data.items() if key != field})
    if supplied != expected:
        raise ValueError(f"{field} content hash mismatch")


@dataclass(frozen=True)
class OccurrenceAssociationPolicyV1:
    """Frozen, engineering-only weights for the deterministic rule baseline."""

    top_k: int = 3
    ambiguity_score_margin: float = 0.08
    minimum_preferred_score: float = 0.50
    top_k_weight_temperature: float = 0.20
    anchor_uncertainty_decay_seconds: float = 8.0
    return_closure_threshold: float = 0.72
    return_background_threshold: float = 0.55
    return_closure_consecutive_cells: int = 2
    course_departure_threshold: float = 0.45
    course_persistence_threshold: float = 0.50
    maximum_measurement_cells_per_candidate: int = 1024
    semi_markov_weight: float = 0.55
    detector_posterior_weight: float = 0.20
    detector_envelope_weight: float = 0.15
    anchor_uncertainty_weight: float = 0.10
    typed_censor_penalty: float = 0.05
    unevaluable_path_penalty: float = 0.25
    unmeasured_candidate_score_ceiling: float = 0.20

    def __post_init__(self) -> None:
        if not 1 <= self.top_k <= 16:
            raise ValueError("association top_k is outside its engineering range")
        if self.return_closure_consecutive_cells < 1:
            raise ValueError("return closure requires at least one cell")
        if self.maximum_measurement_cells_per_candidate < 4:
            raise ValueError("measurement cell cap is unsafe")
        for name, value in asdict(self).items():
            if isinstance(value, int):
                continue
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"association policy {name} must be positive")
        if abs(
            self.semi_markov_weight
            + self.detector_posterior_weight
            + self.detector_envelope_weight
            + self.anchor_uncertainty_weight
            - 1.0
        ) > 1.0e-12:
            raise ValueError("association evidence weights must sum to one")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "weight_source": "frozen_engineering_rule_baseline_not_target_or_clinical_tuned",
            "score_semantics": "uncalibrated_within_occurrence_association_score",
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


DEFAULT_POLICY = OccurrenceAssociationPolicyV1()


def validate_target_blind_detector_occurrence_group_v1(payload: object) -> dict[str, Any]:
    """Validate the small prediction-frozen detector handoff."""

    if type(payload) is not dict:
        raise TypeError("detector occurrence group must be an object")
    data = deepcopy(payload)
    required = {
        "schema_version",
        "receipt_sha256",
        "recording_id",
        "candidate_group_id",
        "occurrence_group_id",
        "sampling_rate_hz",
        "recording_sample_count",
        "frozen_artifacts",
        "occurrence_envelope",
        "posterior_summary",
        "neighbor_occurrence_intervals_samples",
        "scope_receipt",
    }
    if set(data) != required:
        raise ValueError("detector occurrence fields drifted")
    if data["schema_version"] != DETECTOR_OCCURRENCE_SCHEMA_VERSION:
        raise ValueError("detector occurrence schema drifted")
    _identifier(data["recording_id"], "recording_id")
    _identifier(data["candidate_group_id"], "candidate_group_id")
    _identifier(data["occurrence_group_id"], "occurrence_group_id")
    rate = _finite(data["sampling_rate_hz"], "sampling_rate_hz", minimum=1.0)
    count = _integer(data["recording_sample_count"], "recording_sample_count", minimum=1)
    artifacts = data["frozen_artifacts"]
    if type(artifacts) is not dict or set(artifacts) != set(_HASH_FIELDS):
        raise ValueError("detector frozen-artifact roster drifted")
    for field in _HASH_FIELDS:
        _hash(artifacts[field], field)
    envelope = data["occurrence_envelope"]
    if type(envelope) is not dict or set(envelope) != {
        "interval_samples",
        "anchor_sample",
        "anchor_uncertainty_interval_samples",
        "anchor_point_used_for_ranking",
        "semantics",
    }:
        raise ValueError("detector occurrence envelope fields drifted")
    envelope_interval = _interval(
        envelope["interval_samples"], "detector envelope", lower=0, upper=count
    )
    uncertainty = _interval(
        envelope["anchor_uncertainty_interval_samples"],
        "anchor uncertainty interval",
        lower=0,
        upper=count,
    )
    anchor = _integer(envelope["anchor_sample"], "anchor_sample")
    if (
        not uncertainty[0] <= anchor < uncertainty[1]
        or not envelope_interval[0] <= uncertainty[0]
        or uncertainty[1] > envelope_interval[1]
        or envelope["anchor_point_used_for_ranking"] is not False
        or envelope["semantics"]
        != "frozen_detector_occurrence_hypothesis_not_reference_or_clinical_fact"
    ):
        raise ValueError("detector anchor/envelope semantics drifted")
    summary = data["posterior_summary"]
    if type(summary) is not dict or set(summary) != {
        "bins",
        "score_semantics",
        "dense_posterior_included",
    }:
        raise ValueError("detector posterior summary fields drifted")
    if (
        summary["score_semantics"]
        != "frozen_uncalibrated_target_blind_detector_score"
        or summary["dense_posterior_included"] is not False
        or not isinstance(summary["bins"], list)
        or not summary["bins"]
    ):
        raise ValueError("detector posterior summary semantics drifted")
    previous_stop: int | None = None
    for index, row in enumerate(summary["bins"]):
        if type(row) is not dict or set(row) != {
            "bin_id",
            "interval_samples",
            "mean_score",
            "maximum_score",
        }:
            raise ValueError("detector posterior-bin fields drifted")
        start, stop = _interval(
            row["interval_samples"], "posterior bin", lower=0, upper=count
        )
        mean = _finite(row["mean_score"], "mean detector score", minimum=0.0, maximum=1.0)
        maximum = _finite(
            row["maximum_score"], "maximum detector score", minimum=0.0, maximum=1.0
        )
        if (
            row["bin_id"] != f"DP{index:06d}"
            or maximum + 1.0e-12 < mean
            or index == 0
            and start != 0
            or previous_stop is not None
            and start != previous_stop
        ):
            raise ValueError(
                "detector posterior bins must gap-free partition the recording clock"
            )
        previous_stop = stop
    if previous_stop != count:
        raise ValueError(
            "detector posterior bins must gap-free partition the recording clock"
        )
    neighbors = data["neighbor_occurrence_intervals_samples"]
    if not isinstance(neighbors, list):
        raise ValueError("neighbor occurrence intervals must be a list")
    previous_stop = None
    for row in neighbors:
        start, stop = _interval(row, "neighbor occurrence interval", lower=0, upper=count)
        if previous_stop is not None and start < previous_stop:
            raise ValueError("neighbor occurrence intervals overlap or are unsorted")
        previous_stop = stop
    if data["scope_receipt"] != _DETECTOR_SCOPE:
        raise ValueError("detector target-blind scope escalated")
    _verify_content_hash(data)
    _ = rate
    return data


def _validate_typed_censors(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("typed censors must be a list")
    result: list[dict[str, str]] = []
    observed: set[tuple[str, str]] = set()
    for row in value:
        if type(row) is not dict or set(row) != {"kind", "side"}:
            raise ValueError("typed censor fields drifted")
        kind = row["kind"]
        side = row["side"]
        if kind not in _CENSOR_KINDS or side not in _CENSOR_SIDES:
            raise ValueError("typed censor kind or side is invalid")
        if side == "internal" and kind != "qc_gap":
            raise ValueError("only a QC gap may be an internal typed censor")
        key = (kind, side)
        if key in observed:
            raise ValueError("duplicate typed censor")
        observed.add(key)
        result.append({"kind": str(kind), "side": str(side)})
    return result


def validate_native_query_deterministic_measurement_sidecar_v1(
    payload: object,
) -> dict[str, Any]:
    """Validate deterministic, non-clinical measurements after native query."""

    if type(payload) is not dict:
        raise TypeError("native measurement sidecar must be an object")
    data = deepcopy(payload)
    required = {
        "schema_version",
        "receipt_sha256",
        "recording_id",
        "candidate_group_id",
        "occurrence_group_id",
        "source_sentinel_receipt_sha256",
        "source_detector_occurrence_receipt_sha256",
        "recording_sample_count",
        "sampling_rate_hz",
        "measurement_artifacts",
        "query_measurements",
        "scope_receipt",
    }
    if set(data) != required:
        raise ValueError("native measurement sidecar fields drifted")
    if data["schema_version"] != NATIVE_MEASUREMENT_SCHEMA_VERSION:
        raise ValueError("native measurement sidecar schema drifted")
    _identifier(data["recording_id"], "recording_id")
    _identifier(data["candidate_group_id"], "candidate_group_id")
    _identifier(data["occurrence_group_id"], "occurrence_group_id")
    _hash(data["source_sentinel_receipt_sha256"], "source sentinel receipt hash")
    _hash(
        data["source_detector_occurrence_receipt_sha256"],
        "source detector occurrence receipt hash",
    )
    count = _integer(data["recording_sample_count"], "recording_sample_count", minimum=1)
    _finite(data["sampling_rate_hz"], "sampling_rate_hz", minimum=1.0)
    artifacts = data["measurement_artifacts"]
    if type(artifacts) is not dict or set(artifacts) != {
        "measurement_method_sha256",
        "native_signal_dependency_sha256",
        "qc_dependency_sha256",
    }:
        raise ValueError("native measurement artifact roster drifted")
    for field, value in artifacts.items():
        _hash(value, field)
    rows = data["query_measurements"]
    if not isinstance(rows, list):
        raise ValueError("native query measurements must be a list")
    observed_ids: set[str] = set()
    row_required = {
        "proposal_id",
        "proposal_interval_samples",
        "observed_interval_samples",
        "measurement_status",
        "typed_censors",
        "measurement_cells",
    }
    cell_required = {
        "cell_id",
        "ordinal",
        "interval_samples",
        "qc_valid_fraction",
        "background_compatibility",
        "multifeature_departure_score",
        "persistence_score",
        "return_compatibility",
        "permission",
    }
    for row in rows:
        if type(row) is not dict or set(row) != row_required:
            raise ValueError("native query measurement-row fields drifted")
        proposal_id = _identifier(row["proposal_id"], "proposal_id")
        if proposal_id in observed_ids:
            raise ValueError("duplicate native measurement proposal")
        observed_ids.add(proposal_id)
        proposal_interval = _interval(
            row["proposal_interval_samples"],
            "measurement proposal interval",
            lower=0,
            upper=count,
        )
        status = row["measurement_status"]
        if status not in {
            "measured",
            "censored_partial_measurement",
            "censored_without_measurement",
        }:
            raise ValueError("native measurement status is invalid")
        censors = _validate_typed_censors(row["typed_censors"])
        cells = row["measurement_cells"]
        if not isinstance(cells, list):
            raise ValueError("measurement cells must be a list")
        if status == "measured" and censors:
            raise ValueError("uncensored measurement cannot carry typed censors")
        if status != "measured" and not censors:
            raise ValueError("censored measurement requires a typed censor")
        if status == "censored_without_measurement":
            if row["observed_interval_samples"] is not None or cells:
                raise ValueError("unmeasured censor cannot fabricate measurement cells")
            continue
        observed = _interval(
            row["observed_interval_samples"],
            "observed native interval",
            lower=0,
            upper=count,
        )
        if not (observed[0] < proposal_interval[1] and proposal_interval[0] < observed[1]):
            raise ValueError("observed native interval does not overlap its proposal")
        if not cells:
            raise ValueError("measured native interval requires cells")
        if len(cells) > DEFAULT_POLICY.maximum_measurement_cells_per_candidate:
            raise ValueError("native measurement cell cap exceeded")
        previous_stop = observed[0]
        for index, cell in enumerate(cells):
            if type(cell) is not dict or set(cell) != cell_required:
                raise ValueError("native deterministic measurement-cell fields drifted")
            start, stop = _interval(
                cell["interval_samples"], "measurement cell", lower=0, upper=count
            )
            if (
                cell["cell_id"] != f"{proposal_id}-M{index:06d}"
                or cell["ordinal"] != index
                or start != previous_stop
                or cell["permission"] != "deterministic_association_measurement_only"
            ):
                raise ValueError("native measurement cell ordering/permission drifted")
            previous_stop = stop
            for field in (
                "qc_valid_fraction",
                "background_compatibility",
                "multifeature_departure_score",
                "persistence_score",
                "return_compatibility",
            ):
                _finite(cell[field], field, minimum=0.0, maximum=1.0)
        if previous_stop != observed[1]:
            raise ValueError("measurement cells do not partition observed interval")
    if data["scope_receipt"] != _MEASUREMENT_SCOPE:
        raise ValueError("native measurement scope escalated")
    _verify_content_hash(data)
    return data


def _sentinel_projection(receipt: Mapping[str, Any]) -> dict[str, Any]:
    proposal_fields = (
        "proposal_id",
        "interval_samples",
        "source_transition_ids",
        "source_scales_seconds",
        "maximum_screening_score",
        "trigger_native_query",
        "permission",
        "clinical_assertion_authorized",
    )
    projection: dict[str, Any] = {
        "schema_version": receipt["schema_version"],
        "method_id": receipt["method_id"],
        "source_receipt_sha256": receipt["receipt_sha256"],
        "recording_id": receipt["recording_id"],
        "candidate_group_id": receipt["candidate_group_id"],
        "sampling_rate_hz": receipt["acquisition"]["sampling_rate_hz"],
        "recording_sample_count": receipt["acquisition"]["recording_sample_count"],
        "legal_horizon_interval_samples": receipt["legal_horizon"]["interval_samples"],
        "native_query_proposals": [
            {field: deepcopy(row[field]) for field in proposal_fields}
            for row in receipt["native_query_proposals"]
        ],
    }
    projection["projection_sha256"] = _canonical_sha256(projection)
    return projection


def _validate_sentinel_projection(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("sentinel projection must be an object")
    data = deepcopy(value)
    required = {
        "schema_version",
        "method_id",
        "source_receipt_sha256",
        "recording_id",
        "candidate_group_id",
        "sampling_rate_hz",
        "recording_sample_count",
        "legal_horizon_interval_samples",
        "native_query_proposals",
        "projection_sha256",
    }
    if set(data) != required:
        raise ValueError("sentinel association projection fields drifted")
    if data["schema_version"] != SENTINEL_SCHEMA_VERSION or data["method_id"] != SENTINEL_METHOD_ID:
        raise ValueError("sentinel association source binding drifted")
    _hash(data["source_receipt_sha256"], "sentinel source receipt hash")
    _identifier(data["recording_id"], "recording_id")
    _identifier(data["candidate_group_id"], "candidate_group_id")
    _finite(data["sampling_rate_hz"], "sampling_rate_hz", minimum=1.0)
    count = _integer(data["recording_sample_count"], "recording_sample_count", minimum=1)
    horizon = _interval(
        data["legal_horizon_interval_samples"], "sentinel legal horizon", lower=0, upper=count
    )
    proposals = data["native_query_proposals"]
    if not isinstance(proposals, list):
        raise ValueError("sentinel native-query proposals must be a list")
    required_proposal = {
        "proposal_id",
        "interval_samples",
        "source_transition_ids",
        "source_scales_seconds",
        "maximum_screening_score",
        "trigger_native_query",
        "permission",
        "clinical_assertion_authorized",
    }
    previous_stop: int | None = None
    for index, row in enumerate(proposals):
        if type(row) is not dict or set(row) != required_proposal:
            raise ValueError("sentinel proposal projection fields drifted")
        start, stop = _interval(
            row["interval_samples"], "sentinel proposal", lower=horizon[0], upper=horizon[1]
        )
        if (
            row["proposal_id"] != f"NQ{index:06d}"
            or row["trigger_native_query"] is not True
            or row["permission"] != "native_query_only"
            or row["clinical_assertion_authorized"] is not False
            or previous_stop is not None
            and start <= previous_stop
            or not isinstance(row["source_transition_ids"], list)
            or not row["source_transition_ids"]
            or not isinstance(row["source_scales_seconds"], list)
            or not set(row["source_scales_seconds"]).issubset({1, 4, 16})
        ):
            raise ValueError("sentinel proposal association semantics drifted")
        _finite(row["maximum_screening_score"], "screening score", minimum=0.0)
        previous_stop = stop
    expected = _canonical_sha256({key: item for key, item in data.items() if key != "projection_sha256"})
    if data["projection_sha256"] != expected:
        raise ValueError("sentinel association projection hash mismatch")
    return data


def _overlap(left: Sequence[int], right: Sequence[int]) -> int:
    return max(0, min(int(left[1]), int(right[1])) - max(int(left[0]), int(right[0])))


def _interval_iou(left: Sequence[int], right: Sequence[int]) -> float:
    intersection = _overlap(left, right)
    union = max(int(left[1]), int(right[1])) - min(int(left[0]), int(right[0]))
    return 0.0 if union <= 0 else intersection / union


def _posterior_overlap(
    bins: Sequence[Mapping[str, Any]], interval: Sequence[int]
) -> tuple[float, float]:
    total = 0.0
    covered = 0
    for row in bins:
        width = _overlap(row["interval_samples"], interval)
        if width:
            total += width * float(row["mean_score"])
            covered += width
    interval_width = int(interval[1]) - int(interval[0])
    return (
        (total / covered if covered else 0.0),
        (covered / interval_width if interval_width else 0.0),
    )


def _association_prefix_cell_count(
    cells: Sequence[Mapping[str, Any]], policy: OccurrenceAssociationPolicyV1
) -> int:
    course_seen = False
    return_run = 0
    for index, cell in enumerate(cells):
        if (
            float(cell["multifeature_departure_score"])
            >= policy.course_departure_threshold
            and float(cell["persistence_score"])
            >= policy.course_persistence_threshold
        ):
            course_seen = True
        if (
            course_seen
            and float(cell["return_compatibility"])
            >= policy.return_closure_threshold
            and float(cell["background_compatibility"])
            >= policy.return_background_threshold
        ):
            return_run += 1
            if return_run >= policy.return_closure_consecutive_cells:
                return index + 1
        else:
            return_run = 0
    return len(cells)


def _state_emission(cell: Mapping[str, Any], state: str) -> float:
    quality = float(cell["qc_valid_fraction"])
    background = float(cell["background_compatibility"])
    departure = float(cell["multifeature_departure_score"])
    persistence = float(cell["persistence_score"])
    returned = float(cell["return_compatibility"])
    if state == "S0":
        value = 0.75 * background + 0.25 * (1.0 - departure)
    elif state == "S1":
        value = 0.70 * departure + 0.30 * (1.0 - background)
    elif state == "S2":
        value = 0.55 * persistence + 0.30 * departure + 0.15 * (1.0 - background)
    elif state == "S3":
        value = 0.60 * returned + 0.40 * background
    else:  # pragma: no cover - internal roster is frozen
        raise ValueError("unknown association state")
    return quality * value


def _segment_duration_penalty(state: str, duration_seconds: float) -> float:
    if state == "S1":
        return 0.04 * max(0.0, duration_seconds - 4.0)
    if state == "S2":
        return 0.08 * max(0.0, 1.0 - duration_seconds)
    return 0.02 * max(0.0, 0.5 - duration_seconds)


def _segmental_rule_path(
    cells: Sequence[Mapping[str, Any]],
    *,
    rate: float,
    left_censored: bool,
    right_censored: bool,
    internal_censored: bool,
) -> tuple[dict[str, Any], int]:
    if internal_censored or not cells:
        return {
            "model": "censor_aware_single_bout_S0_S1_S2_S3_segmental_rule_baseline_v1",
            "topology": list(_STATES),
            "single_bout": True,
            "reentry_allowed": False,
            "evaluable": False,
            "observed_start_state": None,
            "observed_end_state": None,
            "state_segments": [],
            "state_transition_proposals_samples": [],
            "active_course_interval_samples": None,
            "uncalibrated_path_score": None,
            "left_censored": left_censored,
            "right_censored": right_censored,
            "internal_censored": internal_censored,
            "permission": "state_path_association_proposal_only",
            "clinical_assertion_authorized": False,
        }, 0
    first = 1 if left_censored else 0
    last = 2 if right_censored else 3
    states = list(_STATES[first : last + 1])
    count = len(cells)
    if count < len(states):
        return {
            "model": "censor_aware_single_bout_S0_S1_S2_S3_segmental_rule_baseline_v1",
            "topology": list(_STATES),
            "single_bout": True,
            "reentry_allowed": False,
            "evaluable": False,
            "observed_start_state": states[0],
            "observed_end_state": states[-1],
            "state_segments": [],
            "state_transition_proposals_samples": [],
            "active_course_interval_samples": None,
            "uncalibrated_path_score": None,
            "left_censored": left_censored,
            "right_censored": right_censored,
            "internal_censored": False,
            "permission": "state_path_association_proposal_only",
            "clinical_assertion_authorized": False,
        }, 0
    durations = [
        (int(cell["interval_samples"][1]) - int(cell["interval_samples"][0])) / rate
        for cell in cells
    ]
    duration_prefix = [0.0]
    for duration in durations:
        duration_prefix.append(duration_prefix[-1] + duration)
    emission_prefix: dict[str, list[float]] = {}
    for state in states:
        prefix = [0.0]
        for cell, duration in zip(cells, durations):
            prefix.append(prefix[-1] + _state_emission(cell, state) * duration)
        emission_prefix[state] = prefix

    segment_count = 0
    backpointers: list[list[int | None]] = []
    previous = [-math.inf] * (count + 1)
    first_state = states[0]
    first_back = [None] * (count + 1)
    for end in range(1, count + 1):
        duration = duration_prefix[end]
        previous[end] = (
            emission_prefix[first_state][end]
            - _segment_duration_penalty(first_state, duration)
        )
        first_back[end] = 0
        segment_count += 1
    backpointers.append(first_back)
    for state_index, state in enumerate(states[1:], start=1):
        current = [-math.inf] * (count + 1)
        current_back: list[int | None] = [None] * (count + 1)
        minimum_end = state_index + 1
        for end in range(minimum_end, count + 1):
            best_score = -math.inf
            best_start: int | None = None
            for start in range(state_index, end):
                if not math.isfinite(previous[start]):
                    continue
                duration = duration_prefix[end] - duration_prefix[start]
                score = (
                    previous[start]
                    + emission_prefix[state][end]
                    - emission_prefix[state][start]
                    - _segment_duration_penalty(state, duration)
                )
                segment_count += 1
                if score > best_score + 1.0e-12 or (
                    abs(score - best_score) <= 1.0e-12
                    and (best_start is None or start < best_start)
                ):
                    best_score = score
                    best_start = start
            current[end] = best_score
            current_back[end] = best_start
        previous = current
        backpointers.append(current_back)
    if not math.isfinite(previous[count]):  # pragma: no cover - guarded above
        raise RuntimeError("segmental rule path unexpectedly has no legal partition")
    bounds: list[tuple[int, int]] = []
    end = count
    for state_index in reversed(range(len(states))):
        start = backpointers[state_index][end]
        if start is None:
            raise RuntimeError("segmental rule backpointer is incomplete")
        bounds.append((start, end))
        end = start
    bounds.reverse()
    segments = []
    transitions = []
    for index, (state, (start, stop)) in enumerate(zip(states, bounds)):
        interval = [
            int(cells[start]["interval_samples"][0]),
            int(cells[stop - 1]["interval_samples"][1]),
        ]
        segments.append({"state": state, "interval_samples": interval})
        if index:
            transitions.append(
                {
                    "from_state": states[index - 1],
                    "to_state": state,
                    "sample": interval[0],
                }
            )
    active_segments = [row for row in segments if row["state"] in {"S1", "S2"}]
    active_interval = [
        active_segments[0]["interval_samples"][0],
        active_segments[-1]["interval_samples"][1],
    ]
    total_duration = duration_prefix[-1]
    score = max(0.0, min(1.0, previous[count] / max(total_duration, 1.0e-12)))
    return {
        "model": "censor_aware_single_bout_S0_S1_S2_S3_segmental_rule_baseline_v1",
        "topology": list(_STATES),
        "single_bout": True,
        "reentry_allowed": False,
        "evaluable": True,
        "observed_start_state": states[0],
        "observed_end_state": states[-1],
        "state_segments": segments,
        "state_transition_proposals_samples": transitions,
        "active_course_interval_samples": active_interval,
        "uncalibrated_path_score": _round(score),
        "left_censored": left_censored,
        "right_censored": right_censored,
        "internal_censored": False,
        "permission": "state_path_association_proposal_only",
        "clinical_assertion_authorized": False,
    }, segment_count


def _anchor_compatibility(
    sample: int,
    uncertainty: Sequence[int],
    *,
    rate: float,
    policy: OccurrenceAssociationPolicyV1,
) -> tuple[float, float]:
    if int(uncertainty[0]) <= sample < int(uncertainty[1]):
        distance_samples = 0
    elif sample < int(uncertainty[0]):
        distance_samples = int(uncertainty[0]) - sample
    else:
        distance_samples = sample - int(uncertainty[1]) + 1
    distance_seconds = distance_samples / rate
    compatibility = math.exp(-distance_seconds / policy.anchor_uncertainty_decay_seconds)
    return distance_seconds, compatibility


def _candidate_row(
    proposal: Mapping[str, Any],
    measurement: Mapping[str, Any],
    detector: Mapping[str, Any],
    *,
    chronological_ordinal: int,
    policy: OccurrenceAssociationPolicyV1,
) -> tuple[dict[str, Any], int]:
    cells = measurement["measurement_cells"]
    prefix_count = _association_prefix_cell_count(cells, policy)
    prefix = cells[:prefix_count]
    typed_censors = deepcopy(measurement["typed_censors"])
    left_censored = any(row["side"] == "left" for row in typed_censors)
    right_censored = any(row["side"] == "right" for row in typed_censors)
    internal_censored = any(row["side"] == "internal" for row in typed_censors)
    path, segment_count = _segmental_rule_path(
        prefix,
        rate=float(detector["sampling_rate_hz"]),
        left_censored=left_censored,
        right_censored=right_censored,
        internal_censored=internal_censored,
    )
    association_interval = (
        path["active_course_interval_samples"]
        if path["evaluable"]
        else proposal["interval_samples"]
    )
    if path["evaluable"]:
        s01 = [
            row
            for row in path["state_transition_proposals_samples"]
            if row["from_state"] == "S0" and row["to_state"] == "S1"
        ]
        association_sample = (
            int(s01[0]["sample"])
            if s01
            else int(association_interval[0])
        )
    else:
        association_sample = int(proposal["interval_samples"][0])
    envelope = detector["occurrence_envelope"]
    posterior_mean, posterior_coverage = _posterior_overlap(
        detector["posterior_summary"]["bins"], association_interval
    )
    envelope_iou = _interval_iou(association_interval, envelope["interval_samples"])
    anchor_distance_seconds, anchor_compatibility = _anchor_compatibility(
        association_sample,
        envelope["anchor_uncertainty_interval_samples"],
        rate=float(detector["sampling_rate_hz"]),
        policy=policy,
    )
    path_score = float(path["uncalibrated_path_score"] or 0.0)
    score = (
        policy.semi_markov_weight * path_score
        + policy.detector_posterior_weight * posterior_mean
        + policy.detector_envelope_weight * envelope_iou
        + policy.anchor_uncertainty_weight * anchor_compatibility
        - policy.typed_censor_penalty * len(typed_censors)
        - (0.0 if path["evaluable"] else policy.unevaluable_path_penalty)
    )
    unmeasured_ceiling_applied = (
        measurement["measurement_status"] == "censored_without_measurement"
    )
    if unmeasured_ceiling_applied:
        score = min(score, policy.unmeasured_candidate_score_ceiling)
    proposal_id = str(proposal["proposal_id"])
    row = {
        "proposal_id": proposal_id,
        "chronological_ordinal": chronological_ordinal,
        "proposal_interval_samples": deepcopy(proposal["interval_samples"]),
        "measurement_status": measurement["measurement_status"],
        "observed_interval_samples": deepcopy(measurement["observed_interval_samples"]),
        "measurement_cell_count": len(cells),
        "association_prefix_cell_count": prefix_count,
        "closed_late_suffix_cell_count": len(cells) - prefix_count,
        "closed_late_suffix_values_used_for_score": False,
        "typed_censors": typed_censors,
        "segmental_rule_path": path,
        "detector_compatibility": {
            "association_interval_samples": list(association_interval),
            "posterior_overlap_mean_score": _round(posterior_mean),
            "posterior_interval_coverage_fraction": _round(posterior_coverage),
            "detector_envelope_IoU": _round(envelope_iou),
            "anchor_uncertainty_distance_seconds": _round(anchor_distance_seconds),
            "anchor_uncertainty_compatibility": _round(anchor_compatibility),
            "point_anchor_used_for_ranking": False,
        },
        "score_components": {
            "segmental_rule": _round(path_score),
            "detector_posterior": _round(posterior_mean),
            "detector_envelope": _round(envelope_iou),
            "anchor_uncertainty": _round(anchor_compatibility),
            "typed_censor_count": len(typed_censors),
            "unevaluable_path_penalty_applied": not bool(path["evaluable"]),
            "unmeasured_candidate_score_ceiling_applied": unmeasured_ceiling_applied,
        },
        "uncalibrated_association_score": _round(score),
        "nonchronological_tie_break_sha256": _canonical_sha256(
            {"method_id": METHOD_ID, "proposal_id": proposal_id}
        ),
        "association_rank": None,
        "retained_in_all_candidate_ledger": True,
        "in_top_k": False,
        "top_k_conditional_weight": None,
        "permission": "occurrence_association_proposal_only",
        "clinical_assertion_authorized": False,
    }
    return row, segment_count


def _compute_association_outputs(
    sentinel: Mapping[str, Any],
    detector: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    *,
    policy: OccurrenceAssociationPolicyV1,
) -> dict[str, Any]:
    measurement_by_id = {
        str(row["proposal_id"]): row for row in sidecar["query_measurements"]
    }
    rows: list[dict[str, Any]] = []
    segment_candidates = 0
    for index, proposal in enumerate(sentinel["native_query_proposals"]):
        row, count = _candidate_row(
            proposal,
            measurement_by_id[str(proposal["proposal_id"])],
            detector,
            chronological_ordinal=index,
            policy=policy,
        )
        rows.append(row)
        segment_candidates += count
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["uncalibrated_association_score"]),
            str(row["nonchronological_tie_break_sha256"]),
        ),
    )
    for rank, row in enumerate(ordered, start=1):
        row["association_rank"] = rank
    top = ordered[: policy.top_k]
    if top:
        maximum = max(float(row["uncalibrated_association_score"]) for row in top)
        exponentials = [
            math.exp(
                (float(row["uncalibrated_association_score"]) - maximum)
                / policy.top_k_weight_temperature
            )
            for row in top
        ]
        denominator = sum(exponentials)
        for row, value in zip(top, exponentials):
            row["in_top_k"] = True
            row["top_k_conditional_weight"] = _round(value / denominator)
    rows.sort(key=lambda row: int(row["chronological_ordinal"]))
    top_k = [
        {
            "association_rank": row["association_rank"],
            "proposal_id": row["proposal_id"],
            "uncalibrated_association_score": row["uncalibrated_association_score"],
            "conditional_weight_within_retained_top_k": row["top_k_conditional_weight"],
            "typed_censored": bool(row["typed_censors"]),
            "permission": "occurrence_association_proposal_only",
            "clinical_assertion_authorized": False,
        }
        for row in top
    ]
    ambiguity_reasons: list[str] = []
    if not ordered:
        ambiguity_reasons.append("no_sentinel_native_query_candidate")
    else:
        if not any(row["segmental_rule_path"]["evaluable"] for row in ordered):
            ambiguity_reasons.append("all_candidate_paths_not_evaluable")
        if ordered[0]["typed_censors"]:
            ambiguity_reasons.append("rank1_candidate_is_typed_censored")
        if ordered[0]["measurement_status"] == "censored_without_measurement":
            ambiguity_reasons.append("rank1_candidate_has_no_native_measurement")
        if float(ordered[0]["uncalibrated_association_score"]) < policy.minimum_preferred_score:
            ambiguity_reasons.append("rank1_score_below_engineering_preference_floor")
        if len(ordered) > 1:
            gap = float(ordered[0]["uncalibrated_association_score"]) - float(
                ordered[1]["uncalibrated_association_score"]
            )
            if gap <= policy.ambiguity_score_margin + 1.0e-12:
                ambiguity_reasons.append("rank1_rank2_score_margin_is_small")
    ambiguous = bool(ambiguity_reasons)
    if not ordered:
        status = "no_candidate_association_proposal"
    elif ambiguous:
        status = "ambiguous_competing_association_proposals"
    else:
        status = "preferred_association_proposal_with_all_competitors_retained"
    summary = {
        "status": status,
        "ambiguous": ambiguous,
        "ambiguity_reasons": ambiguity_reasons,
        "candidate_count": len(rows),
        "evaluable_path_count": sum(bool(row["segmental_rule_path"]["evaluable"]) for row in rows),
        "typed_censored_candidate_count": sum(bool(row["typed_censors"]) for row in rows),
        "top_k_count": len(top_k),
        "preferred_proposal_id": (
            str(ordered[0]["proposal_id"]) if ordered and not ambiguous else None
        ),
        "ranked_candidate_ids": [str(row["proposal_id"]) for row in ordered],
        "all_candidates_retained": len(rows) == len(sentinel["native_query_proposals"]),
        "physical_earliest_candidate_wins_unconditionally": False,
        "permission": "occurrence_association_summary_proposal_only",
        "clinical_assertion_authorized": False,
    }
    censor_rows = [
        {
            "proposal_id": row["proposal_id"],
            "association_rank": row["association_rank"],
            "typed_censors": deepcopy(row["typed_censors"]),
            "normal_closure_inferred_from_censor": False,
        }
        for row in rows
        if row["typed_censors"]
    ]
    counts = {kind: 0 for kind in _CENSOR_KINDS}
    for row in censor_rows:
        for censor in row["typed_censors"]:
            counts[str(censor["kind"])] += 1
    censor_ledger = {
        "rows": censor_rows,
        "count_by_kind": counts,
        "record_QC_neighbor_budget_are_typed_censors_not_normal_stop": True,
    }
    core = {
        "candidate_ledger": rows,
        "top_k_associations": top_k,
        "association_summary": summary,
        "typed_censor_ledger": censor_ledger,
    }
    return {
        **core,
        "association_core_fingerprint_sha256": _canonical_sha256(core),
        "compute_ledger": {
            "sentinel_candidate_count": len(sentinel["native_query_proposals"]),
            "native_measurement_row_count": len(sidecar["query_measurements"]),
            "candidate_paths_evaluated": len(rows),
            "segment_candidates_evaluated": segment_candidates,
            "detector_dense_posterior_loaded": False,
            "raw_EEG_read_by_association_module": False,
            "additional_native_query_executed_by_association_module": False,
        },
    }


def _cross_validate_inputs(
    sentinel: Mapping[str, Any],
    detector: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> None:
    for value, name in ((detector, "detector"), (sidecar, "measurement sidecar")):
        if value["recording_id"] != sentinel["recording_id"]:
            raise ValueError(f"{name} recording identity crosses sentinel")
        if value["candidate_group_id"] != sentinel["candidate_group_id"]:
            raise ValueError(f"{name} candidate-group identity crosses sentinel")
        if value["recording_sample_count"] != sentinel["recording_sample_count"]:
            raise ValueError(f"{name} recording length crosses sentinel")
        if abs(float(value["sampling_rate_hz"]) - float(sentinel["sampling_rate_hz"])) > 1.0e-12:
            raise ValueError(f"{name} sampling rate crosses sentinel")
    if sidecar["occurrence_group_id"] != detector["occurrence_group_id"]:
        raise ValueError("measurement occurrence group crosses detector")
    if sidecar["source_sentinel_receipt_sha256"] != sentinel["source_receipt_sha256"]:
        raise ValueError("measurement sidecar is not bound to the sentinel receipt")
    if sidecar["source_detector_occurrence_receipt_sha256"] != detector["receipt_sha256"]:
        raise ValueError("measurement sidecar is not bound to the detector receipt")
    horizon = sentinel["legal_horizon_interval_samples"]
    envelope = detector["occurrence_envelope"]["interval_samples"]
    if not (horizon[0] <= envelope[0] < envelope[1] <= horizon[1]):
        raise ValueError("detector occurrence envelope lies outside sentinel horizon")
    proposals = sentinel["native_query_proposals"]
    measurements = sidecar["query_measurements"]
    if [row["proposal_id"] for row in measurements] != [row["proposal_id"] for row in proposals]:
        raise ValueError("native measurement roster does not preserve every sentinel candidate")
    for proposal, measurement in zip(proposals, measurements):
        if measurement["proposal_interval_samples"] != proposal["interval_samples"]:
            raise ValueError("native measurement proposal interval crosses sentinel")
        observed = measurement["observed_interval_samples"]
        if observed is not None and not (
            horizon[0] <= observed[0] < observed[1] <= horizon[1]
        ):
            raise ValueError("native observed interval lies outside sentinel horizon")
        if any(
            censor["kind"] == "neighbor_guard"
            for censor in measurement["typed_censors"]
        ) and not detector["neighbor_occurrence_intervals_samples"]:
            raise ValueError("neighbor typed censor lacks a frozen neighboring occurrence")


def _source_bindings(
    sentinel: Mapping[str, Any],
    detector: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "sentinel_schema_version": sentinel["schema_version"],
        "sentinel_method_id": sentinel["method_id"],
        "sentinel_receipt_sha256": sentinel["source_receipt_sha256"],
        "sentinel_projection_sha256": sentinel["projection_sha256"],
        "detector_occurrence_schema_version": detector["schema_version"],
        "detector_occurrence_receipt_sha256": detector["receipt_sha256"],
        "native_measurement_schema_version": sidecar["schema_version"],
        "native_measurement_receipt_sha256": sidecar["receipt_sha256"],
        "prediction_frozen_before_association": True,
        "native_measurements_completed_before_association": True,
    }


def materialize_continuous_coarse_detector_occurrence_association_ledger_v1(
    *,
    sentinel_receipt: object,
    detector_occurrence_group: object,
    native_measurement_sidecar: object,
) -> dict[str, Any]:
    """Build a content-addressed, proposal-only occurrence association ledger."""

    validated_sentinel = validate_common17_continuous_coarse_sentinel_cache_v1(
        sentinel_receipt
    )
    sentinel = _sentinel_projection(validated_sentinel)
    detector = validate_target_blind_detector_occurrence_group_v1(
        detector_occurrence_group
    )
    sidecar = validate_native_query_deterministic_measurement_sidecar_v1(
        native_measurement_sidecar
    )
    _cross_validate_inputs(sentinel, detector, sidecar)
    outputs = _compute_association_outputs(
        sentinel, detector, sidecar, policy=DEFAULT_POLICY
    )
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "method_id": METHOD_ID,
        "policy": DEFAULT_POLICY.to_dict(),
        "policy_sha256": DEFAULT_POLICY.sha256,
        "recording_id": sentinel["recording_id"],
        "candidate_group_id": sentinel["candidate_group_id"],
        "occurrence_group_id": detector["occurrence_group_id"],
        "source_bindings": _source_bindings(sentinel, detector, sidecar),
        "input_projection": {
            "sentinel": sentinel,
            "detector_occurrence_group": detector,
            "native_measurement_sidecar": sidecar,
        },
        **outputs,
        "scope_receipt": deepcopy(_OUTPUT_SCOPE),
        "authorization": deepcopy(_AUTHORIZATION),
        "robustness_contract": deepcopy(_ROBUSTNESS_CONTRACT),
    }
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    return validate_continuous_coarse_detector_occurrence_association_ledger_v1(body)


def validate_continuous_coarse_detector_occurrence_association_ledger_v1(
    payload: object,
) -> dict[str, Any]:
    """Fail closed on lineage, permissions, candidate loss or score tampering."""

    if type(payload) is not dict:
        raise TypeError("occurrence association ledger must be an object")
    data = deepcopy(payload)
    required = {
        "schema_version",
        "receipt_sha256",
        "method_id",
        "policy",
        "policy_sha256",
        "recording_id",
        "candidate_group_id",
        "occurrence_group_id",
        "source_bindings",
        "input_projection",
        "candidate_ledger",
        "top_k_associations",
        "association_summary",
        "typed_censor_ledger",
        "association_core_fingerprint_sha256",
        "compute_ledger",
        "scope_receipt",
        "authorization",
        "robustness_contract",
    }
    if set(data) != required:
        raise ValueError("occurrence association top-level fields drifted")
    if data["schema_version"] != SCHEMA_VERSION or data["method_id"] != METHOD_ID:
        raise ValueError("occurrence association method binding drifted")
    if data["policy"] != DEFAULT_POLICY.to_dict() or data["policy_sha256"] != DEFAULT_POLICY.sha256:
        raise ValueError("occurrence association frozen policy drifted")
    _identifier(data["recording_id"], "recording_id")
    _identifier(data["candidate_group_id"], "candidate_group_id")
    _identifier(data["occurrence_group_id"], "occurrence_group_id")
    projection = data["input_projection"]
    if type(projection) is not dict or set(projection) != {
        "sentinel",
        "detector_occurrence_group",
        "native_measurement_sidecar",
    }:
        raise ValueError("occurrence association input projection drifted")
    sentinel = _validate_sentinel_projection(projection["sentinel"])
    detector = validate_target_blind_detector_occurrence_group_v1(
        projection["detector_occurrence_group"]
    )
    sidecar = validate_native_query_deterministic_measurement_sidecar_v1(
        projection["native_measurement_sidecar"]
    )
    _cross_validate_inputs(sentinel, detector, sidecar)
    if (
        data["recording_id"] != sentinel["recording_id"]
        or data["candidate_group_id"] != sentinel["candidate_group_id"]
        or data["occurrence_group_id"] != detector["occurrence_group_id"]
    ):
        raise ValueError("occurrence association output identity crosses its inputs")
    expected_bindings = _source_bindings(sentinel, detector, sidecar)
    if data["source_bindings"] != expected_bindings:
        raise ValueError("occurrence association source bindings drifted")
    expected = _compute_association_outputs(
        sentinel, detector, sidecar, policy=DEFAULT_POLICY
    )
    for field in (
        "candidate_ledger",
        "top_k_associations",
        "association_summary",
        "typed_censor_ledger",
        "association_core_fingerprint_sha256",
        "compute_ledger",
    ):
        if data[field] != expected[field]:
            raise ValueError(f"occurrence association {field} was not deterministically replayed")
    if data["scope_receipt"] != _OUTPUT_SCOPE or data["authorization"] != _AUTHORIZATION:
        raise ValueError("occurrence association scope or permission escalated")
    if data["robustness_contract"] != _ROBUSTNESS_CONTRACT:
        raise ValueError("occurrence association robustness contract drifted")
    expected_hash = _canonical_sha256(
        {key: value for key, value in data.items() if key != "receipt_sha256"}
    )
    if data["receipt_sha256"] != expected_hash:
        raise ValueError("occurrence association content hash mismatch")
    return data


__all__ = [
    "DEFAULT_POLICY",
    "DETECTOR_OCCURRENCE_SCHEMA_VERSION",
    "METHOD_ID",
    "NATIVE_MEASUREMENT_SCHEMA_VERSION",
    "OccurrenceAssociationPolicyV1",
    "SCHEMA_VERSION",
    "materialize_continuous_coarse_detector_occurrence_association_ledger_v1",
    "validate_continuous_coarse_detector_occurrence_association_ledger_v1",
    "validate_native_query_deterministic_measurement_sidecar_v1",
    "validate_target_blind_detector_occurrence_group_v1",
]
