"""Kind-separated spatial projection for complete-ITA BA-IEG outputs.

The complete-ITA aggregator intentionally retains every typed onset unit, but
its internal event softmax may contain both constructive physical electrodes
and whole bipolar leads.  Those are different spatial estimands and must not
share the final reporting simplex.  This module is therefore a mandatory
post-aggregation projection for record-level research rankings:

* a record with any physical-electrode opportunity uses the physical track;
* otherwise a record with whole-lead opportunity uses the whole-lead track;
* an occurrence lacking the selected record resolution contributes one unit
  of explicit unresolved mass instead of falling back to another resolution;
* event distributions are renormalized within the selected kind and then
  combined with exactly equal occurrence weight;
* a whole bipolar lead remains one indivisible unit; only same-side leads may
  support laterality, and only same-side/same-region leads may support region;
* failure/not-evaluable/zero-candidate denominators remain visible and are
  never converted to negative evidence.

All emitted values are uncalibrated normalized evidence masses.  They are not
clinical probabilities, cortical SOZ/EZ estimates, or report-authorized facts.
The formal input must pair the typed, content-replaying complete-ITA output
with an opaque candidate-roster authority issued after replay of the bound
native-inventory artifact bytes.  Raw mappings and caller-supplied probability
tables are not accepted.  A separately gated inventory-free path exists only
for synthetic component tests and is never formally admitted.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence

from src.soz.geometry import STANDARD_19

from .ba_ieg_complete_ita_multievent_aggregation_v1 import (
    BA_IEG_COMPLETE_ITA_MULTIEVENT_AGGREGATION_ID_V1,
    BA_IEG_UNRESOLVED_TYPED_UNIT_KEY_V1,
    BAIEGAggregationTrackV1,
    BAIEGCompleteITAMultiEventAggregationOutputV1,
    BAIEGCompleteITARecordOutputV1,
    BAIEGPerOccurrenceDistributionV1,
    BAIEGTypedUnitProbabilityV1,
)
from .ba_ieg_complete_ita_candidate_roster_materializer_v1 import (
    ValidatedBAIEGCompleteITACandidateRosterAuthorityV1,
    require_validated_ba_ieg_complete_ita_candidate_roster_authority_v1,
)


BA_IEG_COMPLETE_ITA_KIND_SEPARATED_SPATIAL_PROJECTION_ID_V1: Final[str] = (
    "ba_ieg_complete_ita_kind_separated_spatial_projection_v1"
)
BA_IEG_COMPLETE_ITA_KIND_SEPARATED_SPATIAL_SCHEMA_V1: Final[str] = (
    "ba_ieg_complete_ita_kind_separated_spatial_output_v1"
)
BA_IEG_SPATIAL_UNRESOLVED_KEY_V1: Final[str] = (
    "unresolved:not_identifiable_at_requested_spatial_axis"
)

_TOL: Final[float] = 1e-9
_PHYSICAL_KIND: Final[str] = "physical_electrode"
_LEAD_KIND: Final[str] = "whole_bipolar_lead"
_UNRESOLVED_KIND: Final[str] = "unresolved"
_RESOLUTION_KINDS: Final[frozenset[str]] = frozenset(
    {_PHYSICAL_KIND, _LEAD_KIND, _UNRESOLVED_KIND}
)

_LEFT: Final[frozenset[str]] = frozenset(
    {"FP1", "F7", "F3", "T7", "C3", "P7", "P3", "O1"}
)
_RIGHT: Final[frozenset[str]] = frozenset(
    {"FP2", "F8", "F4", "T8", "C4", "P8", "P4", "O2"}
)
_MIDLINE: Final[frozenset[str]] = frozenset({"FZ", "CZ", "PZ"})
BA_IEG_ITA_LATERALITY_IDS_V1: Final[tuple[str, ...]] = (
    "left",
    "right",
    "midline",
)
BA_IEG_ITA_COARSE_REGION_IDS_V1: Final[tuple[str, ...]] = (
    "frontal",
    "temporal",
    "central",
    "parietal",
    "occipital",
)


def _laterality(electrode: str) -> str:
    if electrode in _LEFT:
        return "left"
    if electrode in _RIGHT:
        return "right"
    if electrode in _MIDLINE:
        return "midline"
    raise ValueError(f"unknown standard-19 electrode: {electrode}")


def _region(electrode: str) -> str:
    if electrode in {"F7", "T7", "P7", "F8", "T8", "P8"}:
        return "temporal"
    if electrode.startswith("FP") or electrode.startswith("F"):
        return "frontal"
    if electrode.startswith("C"):
        return "central"
    if electrode.startswith("P"):
        return "parietal"
    if electrode.startswith("O"):
        return "occipital"
    raise ValueError(f"standard-19 electrode has no coarse region: {electrode}")


BA_IEG_ITA_JOINT_REGION_IDS_V1: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(f"{_laterality(item)}_{_region(item)}" for item in STANDARD_19)
)


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


def _finite_mass(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0 + _TOL:
        raise ValueError(f"{context} must lie in [0,1]")
    return min(1.0, result)


def _fraction_pair(numerator: int, denominator: int) -> list[int] | None:
    if denominator == 0:
        return None
    value = Fraction(numerator, denominator)
    return [value.numerator, value.denominator]


def _top_key(mapping: Mapping[str, float]) -> str | None:
    if not mapping:
        return None
    return min(mapping, key=lambda key: (-float(mapping[key]), key))


def _mix(maps: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not maps:
        return {}
    keys = sorted(set().union(*(row.keys() for row in maps)))
    denominator = float(len(maps))
    result = {
        key: math.fsum(row.get(key, 0.0) for row in maps) / denominator
        for key in keys
    }
    if abs(math.fsum(result.values()) - 1.0) > 1e-7:
        raise AssertionError("equal-occurrence projection did not preserve unit mass")
    return result


def _total_variation(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = sorted(set(left).union(right))
    return 0.5 * math.fsum(
        abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0)))
        for key in keys
    )


@dataclass(frozen=True)
class BAIEGNormalizedEvidenceMassV1:
    candidate_id: str
    mass: float

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "mass": self.mass}


@dataclass(frozen=True)
class BAIEGKindSeparatedOccurrenceProjectionV1:
    recording_id: str
    occurrence_id: str
    canonical_event_id: str
    source_distribution_sha256: str
    resolution_kind: str
    unit_masses: tuple[BAIEGTypedUnitProbabilityV1, ...]
    unresolved_mass: float
    suppressed_alternate_kind_mass: float
    suppressed_alternate_kind_count: int
    top_unit_key: str
    occurrence_projection_sha256: str

    def probability_map(self) -> dict[str, float]:
        result = {row.unit_key: float(row.probability) for row in self.unit_masses}
        result[BA_IEG_UNRESOLVED_TYPED_UNIT_KEY_V1] = float(self.unresolved_mass)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "occurrence_id": self.occurrence_id,
            "canonical_event_id": self.canonical_event_id,
            "source_distribution_sha256": self.source_distribution_sha256,
            "resolution_kind": self.resolution_kind,
            "unit_masses": [row.to_dict() for row in self.unit_masses],
            "unresolved_mass": self.unresolved_mass,
            "suppressed_alternate_kind_mass": self.suppressed_alternate_kind_mass,
            "suppressed_alternate_kind_count": self.suppressed_alternate_kind_count,
            "top_unit_key": self.top_unit_key,
            "mass_semantics": "within_occurrence_kind_normalized_uncalibrated_evidence_mass",
            "occurrence_projection_sha256": self.occurrence_projection_sha256,
        }


@dataclass(frozen=True)
class BAIEGSpatialAxisMassV1:
    axis_id: str
    candidate_roster: tuple[str, ...]
    opportunity_ids: tuple[str, ...]
    ranked_candidate_masses: tuple[BAIEGNormalizedEvidenceMassV1, ...]
    unresolved_mass: float | None
    top_key: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_id": self.axis_id,
            "candidate_roster": list(self.candidate_roster),
            "opportunity_ids": list(self.opportunity_ids),
            "ranked_candidate_masses": [
                row.to_dict() for row in self.ranked_candidate_masses
            ],
            "unresolved_mass": self.unresolved_mass,
            "top_key": self.top_key,
            "absent_opportunity_is_negative": False,
            "mass_semantics": "equal_occurrence_uncalibrated_spatial_evidence_mass",
        }


@dataclass(frozen=True)
class BAIEGKindSeparatedAggregationTrackV1:
    track_id: str
    role: str
    aggregation_status: str
    resolution_kind: str
    included_occurrence_ids: tuple[str, ...]
    excluded_occurrence_ids: tuple[str, ...]
    per_occurrence: tuple[BAIEGKindSeparatedOccurrenceProjectionV1, ...]
    equal_occurrence_unit_mixture: tuple[BAIEGTypedUnitProbabilityV1, ...]
    unresolved_mass: float | None
    top_unit_key: str | None
    laterality: BAIEGSpatialAxisMassV1
    coarse_region: BAIEGSpatialAxisMassV1
    joint_region: BAIEGSpatialAxisMassV1
    pairwise_max_total_variation: float | None
    discordant_occurrence_pair_count: int
    maximum_leave_one_occurrence_out_total_variation: float | None
    leave_one_occurrence_out_top1_change_count: int
    single_occurrence_dependence: float | None
    leave_one_occurrence_out_status: str
    report_claim_authorized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "role": self.role,
            "aggregation_status": self.aggregation_status,
            "resolution_kind": self.resolution_kind,
            "included_occurrence_ids": list(self.included_occurrence_ids),
            "excluded_occurrence_ids": list(self.excluded_occurrence_ids),
            "per_occurrence": [row.to_dict() for row in self.per_occurrence],
            "equal_occurrence_unit_mixture": [
                row.to_dict() for row in self.equal_occurrence_unit_mixture
            ],
            "unresolved_mass": self.unresolved_mass,
            "top_unit_key": self.top_unit_key,
            "laterality": self.laterality.to_dict(),
            "coarse_region": self.coarse_region.to_dict(),
            "joint_region": self.joint_region.to_dict(),
            "pairwise_max_total_variation": self.pairwise_max_total_variation,
            "discordant_occurrence_pair_count": self.discordant_occurrence_pair_count,
            "maximum_leave_one_occurrence_out_total_variation": (
                self.maximum_leave_one_occurrence_out_total_variation
            ),
            "leave_one_occurrence_out_top1_change_count": (
                self.leave_one_occurrence_out_top1_change_count
            ),
            "single_occurrence_dependence": self.single_occurrence_dependence,
            "leave_one_occurrence_out_status": self.leave_one_occurrence_out_status,
            "report_claim_authorized": self.report_claim_authorized,
            "electrode_and_whole_lead_share_simplex": False,
            "whole_lead_endpoint_votes_created": False,
        }


@dataclass(frozen=True)
class BAIEGKindSeparatedRecordProjectionV1:
    recording_id: str
    resolution_kind: str
    source_record_output_sha256: str
    source_denominator: Mapping[str, Any]
    eeg_evaluable_occurrence_coverage_fraction: list[int] | None
    resolved_selected_kind_coverage_fraction: list[int] | None
    ita_primary: BAIEGKindSeparatedAggregationTrackV1
    qualified_only_secondary: BAIEGKindSeparatedAggregationTrackV1
    record_projection_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "resolution_kind": self.resolution_kind,
            "source_record_output_sha256": self.source_record_output_sha256,
            "source_denominator": deepcopy(dict(self.source_denominator)),
            "eeg_evaluable_occurrence_coverage_fraction": (
                self.eeg_evaluable_occurrence_coverage_fraction
            ),
            "resolved_selected_kind_coverage_fraction": (
                self.resolved_selected_kind_coverage_fraction
            ),
            "ita_primary": self.ita_primary.to_dict(),
            "qualified_only_secondary": self.qualified_only_secondary.to_dict(),
            "record_projection_sha256": self.record_projection_sha256,
        }


@dataclass(frozen=True)
class BAIEGCompleteITAKindSeparatedSpatialOutputV1:
    schema_version: str
    implementation_id: str
    source_complete_ita_output_sha256: str
    complete_candidate_roster_receipt_sha256: str | None
    candidate_roster_authority_receipt_sha256: str | None
    formal_complete_roster_materializer_admitted: bool
    records: tuple[BAIEGKindSeparatedRecordProjectionV1, ...]
    output_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "implementation_id": self.implementation_id,
            "source_complete_ita_output_sha256": (
                self.source_complete_ita_output_sha256
            ),
            "complete_candidate_roster_receipt_sha256": (
                self.complete_candidate_roster_receipt_sha256
            ),
            "candidate_roster_authority_receipt_sha256": (
                self.candidate_roster_authority_receipt_sha256
            ),
            "records": [row.to_dict() for row in self.records],
            "output_semantics": (
                "kind_separated_equal_occurrence_uncalibrated_scalp_visible_"
                "onset_evidence_not_clinical_probability_cortical_soz_or_ez"
            ),
            "report_claim_authorized": False,
            "formal_complete_roster_materializer_admitted": (
                self.formal_complete_roster_materializer_admitted
            ),
            "output_sha256": self.output_sha256,
        }


def _validate_source_distribution(
    value: BAIEGPerOccurrenceDistributionV1,
) -> tuple[dict[str, float], dict[str, BAIEGTypedUnitProbabilityV1]]:
    if not isinstance(value, BAIEGPerOccurrenceDistributionV1):
        raise TypeError("spatial projection accepts typed occurrence distributions only")
    serialized = value.to_dict()
    observed = serialized.pop("distribution_sha256")
    serialized.pop("probability_semantics")
    if observed != _canonical_sha256(serialized):
        raise ValueError("source occurrence distribution does not replay")
    available_kinds = value.available_resolution_kinds
    if (
        not isinstance(available_kinds, tuple)
        or available_kinds != tuple(sorted(set(available_kinds)))
        or set(available_kinds).difference({_PHYSICAL_KIND, _LEAD_KIND})
    ):
        raise ValueError("source occurrence available resolution kinds drifted")
    mapping: dict[str, float] = {}
    rows_by_key: dict[str, BAIEGTypedUnitProbabilityV1] = {}
    standard_index = {item: index for index, item in enumerate(STANDARD_19)}
    for row in value.typed_unit_probabilities:
        probability = _finite_mass(row.probability, "typed-unit probability")
        if row.unit_key in mapping:
            raise ValueError("source occurrence repeats a typed-unit key")
        if row.unit_kind == _PHYSICAL_KIND:
            if (
                row.electrode_id not in STANDARD_19
                or row.whole_bipolar_lead is not None
                or row.unit_key != f"physical_electrode:{row.electrode_id}"
            ):
                raise ValueError("physical typed-unit ontology drifted")
        elif row.unit_kind == _LEAD_KIND:
            endpoints = row.whole_bipolar_lead
            if (
                row.electrode_id is not None
                or endpoints is None
                or len(endpoints) != 2
                or endpoints[0] not in STANDARD_19
                or endpoints[1] not in STANDARD_19
                or standard_index[endpoints[0]] >= standard_index[endpoints[1]]
                or row.unit_key
                != f"whole_bipolar_lead:{endpoints[0]}--{endpoints[1]}"
            ):
                raise ValueError("whole-lead typed-unit ontology drifted")
        else:
            raise ValueError("source occurrence contains an unsupported unit kind")
        if row.unit_kind not in available_kinds:
            raise ValueError("ranked typed-unit kind is absent from its inventory")
        mapping[row.unit_key] = probability
        rows_by_key[row.unit_key] = row
    unresolved = _finite_mass(value.unresolved_probability, "source unresolved mass")
    mapping[BA_IEG_UNRESOLVED_TYPED_UNIT_KEY_V1] = unresolved
    if abs(math.fsum(mapping.values()) - 1.0) > 1e-7:
        raise ValueError("source occurrence distribution does not sum to one")
    if value.top_distribution_key != _top_key(mapping):
        raise ValueError("source occurrence top key does not replay")
    expected_order = tuple(
        sorted(value.typed_unit_probabilities, key=lambda row: (-row.probability, row.unit_key))
    )
    if value.typed_unit_probabilities != expected_order:
        raise ValueError("source occurrence typed-unit ordering drifted")
    return mapping, rows_by_key


def _validate_source_output(
    value: BAIEGCompleteITAMultiEventAggregationOutputV1,
) -> None:
    if not isinstance(value, BAIEGCompleteITAMultiEventAggregationOutputV1):
        raise TypeError("kind-separated projection requires typed complete-ITA output")
    if value.implementation_id != BA_IEG_COMPLETE_ITA_MULTIEVENT_AGGREGATION_ID_V1:
        raise ValueError("complete-ITA source implementation is unsupported")
    if (
        len(value.source_manifest_sha256) != 64
        or set(value.source_manifest_sha256).difference("0123456789abcdef")
    ):
        raise ValueError("complete-ITA source manifest binding is invalid")
    if tuple(record.recording_id for record in value.records) != tuple(
        sorted({record.recording_id for record in value.records})
    ):
        raise ValueError("complete-ITA source record roster is not unique and sorted")
    serialized = value.to_dict()
    observed = serialized.pop("output_sha256")
    if observed != _canonical_sha256(serialized):
        raise ValueError("complete-ITA source output does not replay")
    for record in value.records:
        record_value = record.to_dict()
        record_observed = record_value.pop("record_output_sha256")
        if record_observed != _canonical_sha256(record_value):
            raise ValueError("complete-ITA source record does not replay")
        denominator = record.denominator
        if (
            denominator.recording_id != record.recording_id
            or denominator.eeg_evaluable_occurrence_count
            != len(record.ita_primary.per_occurrence_distributions)
            or record.ita_primary.complete_record_denominator_authorized is not True
            or record.qualified_only_secondary.complete_record_denominator_authorized
            is not False
            or record.ita_primary.report_claim_authorized is not False
            or record.qualified_only_secondary.report_claim_authorized is not False
            or not set(record.qualified_only_secondary.included_occurrence_ids).issubset(
                record.ita_primary.included_occurrence_ids
            )
        ):
            raise ValueError("complete-ITA source denominator/track semantics drifted")
        for track in (record.ita_primary, record.qualified_only_secondary):
            if tuple(row.occurrence_id for row in track.per_occurrence_distributions) != (
                track.included_occurrence_ids
            ):
                raise ValueError("source track membership/distribution roster drifted")
            for distribution in track.per_occurrence_distributions:
                _validate_source_distribution(distribution)


def _choose_record_resolution(record: BAIEGCompleteITARecordOutputV1) -> str:
    kinds = {
        kind
        for distribution in record.ita_primary.per_occurrence_distributions
        for kind in distribution.available_resolution_kinds
    }
    if _PHYSICAL_KIND in kinds:
        return _PHYSICAL_KIND
    if _LEAD_KIND in kinds:
        return _LEAD_KIND
    return _UNRESOLVED_KIND


def _validate_source_units_against_authority(
    record: BAIEGCompleteITARecordOutputV1,
    authority_record: Mapping[str, Any],
) -> None:
    """Fail closed if model evidence introduces a non-native typed unit."""

    allowed_by_kind = {
        _PHYSICAL_KIND: set(authority_record["physical_electrode_unit_keys"]),
        _LEAD_KIND: set(authority_record["whole_bipolar_lead_unit_keys"]),
    }
    for track in (record.ita_primary, record.qualified_only_secondary):
        for distribution in track.per_occurrence_distributions:
            for kind in distribution.available_resolution_kinds:
                if not allowed_by_kind[kind]:
                    raise PermissionError(
                        "complete-ITA source advertises a resolution absent from "
                        "the observed native candidate inventory"
                    )
            for row in distribution.typed_unit_probabilities:
                if row.unit_key not in allowed_by_kind[row.unit_kind]:
                    raise PermissionError(
                        "complete-ITA source contains a typed unit outside the "
                        "observed native candidate inventory"
                    )


def _project_occurrence(
    source: BAIEGPerOccurrenceDistributionV1,
    *,
    resolution_kind: str,
) -> BAIEGKindSeparatedOccurrenceProjectionV1:
    if resolution_kind not in _RESOLUTION_KINDS:
        raise ValueError("record resolution kind is unsupported")
    _mapping, _rows_by_key = _validate_source_distribution(source)
    selected = [
        row for row in source.typed_unit_probabilities if row.unit_kind == resolution_kind
    ]
    alternate = [
        row for row in source.typed_unit_probabilities if row.unit_kind != resolution_kind
    ]
    selected_mass = math.fsum(float(row.probability) for row in selected)
    if selected_mass > 0.0:
        normalized = tuple(
            sorted(
                (
                    BAIEGTypedUnitProbabilityV1(
                        unit_key=row.unit_key,
                        unit_kind=row.unit_kind,
                        probability=float(row.probability) / selected_mass,
                        electrode_id=row.electrode_id,
                        whole_bipolar_lead=row.whole_bipolar_lead,
                    )
                    for row in selected
                ),
                key=lambda row: (-row.probability, row.unit_key),
            )
        )
        unresolved = 0.0
    else:
        normalized = ()
        unresolved = 1.0
    projected_map = {row.unit_key: row.probability for row in normalized}
    projected_map[BA_IEG_UNRESOLVED_TYPED_UNIT_KEY_V1] = unresolved
    body = {
        "recording_id": source.recording_id,
        "occurrence_id": source.occurrence_id,
        "canonical_event_id": source.canonical_event_id,
        "source_distribution_sha256": source.distribution_sha256,
        "resolution_kind": resolution_kind,
        "unit_masses": [row.to_dict() for row in normalized],
        "unresolved_mass": unresolved,
        "suppressed_alternate_kind_mass": math.fsum(
            float(row.probability) for row in alternate
        ),
        "suppressed_alternate_kind_count": len(alternate),
        "top_unit_key": _top_key(projected_map),
        "mass_semantics": "within_occurrence_kind_normalized_uncalibrated_evidence_mass",
    }
    return BAIEGKindSeparatedOccurrenceProjectionV1(
        recording_id=source.recording_id,
        occurrence_id=source.occurrence_id,
        canonical_event_id=source.canonical_event_id,
        source_distribution_sha256=source.distribution_sha256,
        resolution_kind=resolution_kind,
        unit_masses=normalized,
        unresolved_mass=unresolved,
        suppressed_alternate_kind_mass=body["suppressed_alternate_kind_mass"],
        suppressed_alternate_kind_count=len(alternate),
        top_unit_key=body["top_unit_key"],
        occurrence_projection_sha256=_canonical_sha256(body),
    )


def _unit_spatial_candidates(
    row: BAIEGTypedUnitProbabilityV1,
) -> tuple[str | None, str | None, str | None]:
    if row.unit_kind == _PHYSICAL_KIND:
        if row.electrode_id is None:
            raise AssertionError("validated physical unit lost its electrode")
        laterality = _laterality(row.electrode_id)
        region = _region(row.electrode_id)
        return laterality, region, f"{laterality}_{region}"
    endpoints = row.whole_bipolar_lead
    if endpoints is None:
        raise AssertionError("validated whole lead lost its endpoints")
    left_side, right_side = (_laterality(item) for item in endpoints)
    if left_side != right_side:
        return None, None, None
    left_region, right_region = (_region(item) for item in endpoints)
    if left_region != right_region:
        return left_side, None, None
    return left_side, left_region, f"{left_side}_{left_region}"


def _event_axis_map(
    event: BAIEGKindSeparatedOccurrenceProjectionV1,
    axis: str,
) -> tuple[dict[str, float], set[str]]:
    result: dict[str, float] = {
        BA_IEG_SPATIAL_UNRESOLVED_KEY_V1: event.unresolved_mass
    }
    opportunities: set[str] = set()
    axis_index = {"laterality": 0, "coarse_region": 1, "joint_region": 2}[axis]
    for row in event.unit_masses:
        candidate = _unit_spatial_candidates(row)[axis_index]
        if candidate is None:
            result[BA_IEG_SPATIAL_UNRESOLVED_KEY_V1] += float(row.probability)
        else:
            opportunities.add(candidate)
            result[candidate] = result.get(candidate, 0.0) + float(row.probability)
    if abs(math.fsum(result.values()) - 1.0) > 1e-7:
        raise AssertionError("event spatial projection did not preserve unit mass")
    return result, opportunities


def _axis_output(
    events: Sequence[BAIEGKindSeparatedOccurrenceProjectionV1],
    *,
    axis_id: str,
    roster: tuple[str, ...],
) -> BAIEGSpatialAxisMassV1:
    if not events:
        return BAIEGSpatialAxisMassV1(
            axis_id=axis_id,
            candidate_roster=roster,
            opportunity_ids=(),
            ranked_candidate_masses=(),
            unresolved_mass=None,
            top_key=None,
        )
    maps: list[dict[str, float]] = []
    opportunities: set[str] = set()
    for event in events:
        mapping, event_opportunities = _event_axis_map(event, axis_id)
        maps.append(mapping)
        opportunities.update(event_opportunities)
    mixture = _mix(maps)
    rows = tuple(
        sorted(
            (
                BAIEGNormalizedEvidenceMassV1(candidate_id=key, mass=mixture.get(key, 0.0))
                for key in opportunities
            ),
            key=lambda row: (-row.mass, row.candidate_id),
        )
    )
    return BAIEGSpatialAxisMassV1(
        axis_id=axis_id,
        candidate_roster=roster,
        opportunity_ids=tuple(item for item in roster if item in opportunities),
        ranked_candidate_masses=rows,
        unresolved_mass=mixture.get(BA_IEG_SPATIAL_UNRESOLVED_KEY_V1, 0.0),
        top_key=_top_key(mixture),
    )


def _unit_rows_from_mixture(
    mixture: Mapping[str, float],
    descriptors: Mapping[str, BAIEGTypedUnitProbabilityV1],
) -> tuple[BAIEGTypedUnitProbabilityV1, ...]:
    return tuple(
        sorted(
            (
                BAIEGTypedUnitProbabilityV1(
                    unit_key=key,
                    unit_kind=descriptors[key].unit_kind,
                    probability=float(value),
                    electrode_id=descriptors[key].electrode_id,
                    whole_bipolar_lead=descriptors[key].whole_bipolar_lead,
                )
                for key, value in mixture.items()
                if key != BA_IEG_UNRESOLVED_TYPED_UNIT_KEY_V1
            ),
            key=lambda row: (-row.probability, row.unit_key),
        )
    )


def _project_track(
    source: BAIEGAggregationTrackV1,
    *,
    resolution_kind: str,
) -> BAIEGKindSeparatedAggregationTrackV1:
    events = tuple(
        _project_occurrence(row, resolution_kind=resolution_kind)
        for row in source.per_occurrence_distributions
    )
    if not events:
        mixture: dict[str, float] = {}
        descriptors: dict[str, BAIEGTypedUnitProbabilityV1] = {}
        unresolved: float | None = None
        top: str | None = None
        pairwise_max = None
        discordant_pairs = 0
        loeo_max = None
        loeo_top_changes = 0
        single_dependence = None
        loeo_status = "not_applicable_no_included_occurrence"
    else:
        event_maps = [row.probability_map() for row in events]
        mixture = _mix(event_maps)
        descriptors = {
            row.unit_key: row for event in events for row in event.unit_masses
        }
        unresolved = mixture.get(BA_IEG_UNRESOLVED_TYPED_UNIT_KEY_V1, 0.0)
        top = _top_key(mixture)
        pairwise_distances: list[float] = []
        discordant_pairs = 0
        for left_index, left in enumerate(event_maps):
            for right in event_maps[left_index + 1 :]:
                pairwise_distances.append(_total_variation(left, right))
                discordant_pairs += _top_key(left) != _top_key(right)
        pairwise_max = max(pairwise_distances) if pairwise_distances else None
        if len(event_maps) == 1:
            loeo_max = None
            loeo_top_changes = 0
            single_dependence = 1.0
            loeo_status = "not_evaluable_single_included_occurrence"
        else:
            loeo_distances: list[float] = []
            loeo_top_changes = 0
            for index in range(len(event_maps)):
                retained = _mix(event_maps[:index] + event_maps[index + 1 :])
                loeo_distances.append(_total_variation(mixture, retained))
                loeo_top_changes += _top_key(retained) != top
            loeo_max = max(loeo_distances)
            single_dependence = loeo_max
            loeo_status = "evaluated_maximum_full_vs_leave_one_occurrence_out_tv"
    unit_rows = _unit_rows_from_mixture(mixture, descriptors)
    status = source.aggregation_status
    if events and unresolved is not None and unresolved >= 1.0 - _TOL:
        status = "evaluable_without_selected_resolution_distribution"
    return BAIEGKindSeparatedAggregationTrackV1(
        track_id=source.track_id,
        role=source.role,
        aggregation_status=status,
        resolution_kind=resolution_kind,
        included_occurrence_ids=source.included_occurrence_ids,
        excluded_occurrence_ids=source.excluded_occurrence_ids,
        per_occurrence=events,
        equal_occurrence_unit_mixture=unit_rows,
        unresolved_mass=unresolved,
        top_unit_key=top,
        laterality=_axis_output(
            events,
            axis_id="laterality",
            roster=BA_IEG_ITA_LATERALITY_IDS_V1,
        ),
        coarse_region=_axis_output(
            events,
            axis_id="coarse_region",
            roster=BA_IEG_ITA_COARSE_REGION_IDS_V1,
        ),
        joint_region=_axis_output(
            events,
            axis_id="joint_region",
            roster=BA_IEG_ITA_JOINT_REGION_IDS_V1,
        ),
        pairwise_max_total_variation=pairwise_max,
        discordant_occurrence_pair_count=discordant_pairs,
        maximum_leave_one_occurrence_out_total_variation=loeo_max,
        leave_one_occurrence_out_top1_change_count=loeo_top_changes,
        single_occurrence_dependence=single_dependence,
        leave_one_occurrence_out_status=loeo_status,
        report_claim_authorized=False,
    )


class BAIEGCompleteITAKindSeparatedSpatialProjectionV1:
    """Project complete ITA onto one record-level spatial resolution."""

    implementation_id: Final[str] = (
        BA_IEG_COMPLETE_ITA_KIND_SEPARATED_SPATIAL_PROJECTION_ID_V1
    )

    def __init__(self, *, allow_component_test_inventory: bool = False) -> None:
        if not isinstance(allow_component_test_inventory, bool):
            raise TypeError("component-test inventory gate must be boolean")
        self.allow_component_test_inventory = allow_component_test_inventory

    def __call__(
        self,
        source: BAIEGCompleteITAMultiEventAggregationOutputV1,
        candidate_roster_authority: (
            ValidatedBAIEGCompleteITACandidateRosterAuthorityV1 | None
        ) = None,
    ) -> BAIEGCompleteITAKindSeparatedSpatialOutputV1:
        _validate_source_output(source)
        authority_receipt: dict[str, Any] | None = None
        authority_by_record: dict[str, Mapping[str, Any]] = {}
        if candidate_roster_authority is not None:
            authority = (
                require_validated_ba_ieg_complete_ita_candidate_roster_authority_v1(
                    candidate_roster_authority
                )
            )
            authority_receipt = authority.receipt
            manifest = authority.manifest
            if source.source_manifest_sha256 != manifest.manifest_sha256 or (
                authority_receipt["manifest_sha256"] != manifest.manifest_sha256
            ):
                raise PermissionError(
                    "complete-ITA source manifest is stale relative to candidate roster authority"
                )
            authority_by_record = {
                row["recording_id"]: row for row in authority_receipt["records"]
            }
            if set(authority_by_record) != {
                record.recording_id for record in source.records
            }:
                raise PermissionError(
                    "complete-ITA source record roster differs from formal candidate authority"
                )
        elif not self.allow_component_test_inventory:
            raise TypeError(
                "formal kind-separated projection requires an opaque complete-candidate-roster authority"
            )
        records: list[BAIEGKindSeparatedRecordProjectionV1] = []
        for record in source.records:
            if authority_receipt is None:
                resolution = _choose_record_resolution(record)
            else:
                authority_record = authority_by_record[record.recording_id]
                resolution = authority_record["resolution_kind"]
                if list(record.ita_primary.included_occurrence_ids) != list(
                    authority_record["eeg_evaluable_occurrence_ids"]
                ):
                    raise PermissionError(
                        "complete-ITA source dropped or added an evaluable occurrence"
                    )
                if record.denominator.unique_occurrence_count != len(
                    authority_record["occurrence_ids"]
                ):
                    raise PermissionError(
                        "complete-ITA source occurrence denominator differs from formal roster"
                    )
                _validate_source_units_against_authority(record, authority_record)
            ita = _project_track(record.ita_primary, resolution_kind=resolution)
            secondary = _project_track(
                record.qualified_only_secondary,
                resolution_kind=resolution,
            )
            denominator = record.denominator.to_dict()
            unique_count = int(denominator["unique_occurrence_count"])
            evaluable_count = int(denominator["eeg_evaluable_occurrence_count"])
            resolved_count = sum(
                row.unresolved_mass < 1.0 - _TOL for row in ita.per_occurrence
            )
            body = {
                "recording_id": record.recording_id,
                "resolution_kind": resolution,
                "source_record_output_sha256": record.record_output_sha256,
                "source_denominator": denominator,
                "eeg_evaluable_occurrence_coverage_fraction": _fraction_pair(
                    evaluable_count, unique_count
                ),
                "resolved_selected_kind_coverage_fraction": _fraction_pair(
                    resolved_count, unique_count
                ),
                "ita_primary": ita.to_dict(),
                "qualified_only_secondary": secondary.to_dict(),
            }
            records.append(
                BAIEGKindSeparatedRecordProjectionV1(
                    recording_id=record.recording_id,
                    resolution_kind=resolution,
                    source_record_output_sha256=record.record_output_sha256,
                    source_denominator=denominator,
                    eeg_evaluable_occurrence_coverage_fraction=body[
                        "eeg_evaluable_occurrence_coverage_fraction"
                    ],
                    resolved_selected_kind_coverage_fraction=body[
                        "resolved_selected_kind_coverage_fraction"
                    ],
                    ita_primary=ita,
                    qualified_only_secondary=secondary,
                    record_projection_sha256=_canonical_sha256(body),
                )
            )
        output_body = {
            "schema_version": BA_IEG_COMPLETE_ITA_KIND_SEPARATED_SPATIAL_SCHEMA_V1,
            "implementation_id": self.implementation_id,
            "source_complete_ita_output_sha256": source.output_sha256,
            "complete_candidate_roster_receipt_sha256": (
                None
                if authority_receipt is None
                else authority_receipt["candidate_roster_receipt_sha256"]
            ),
            "candidate_roster_authority_receipt_sha256": (
                None
                if authority_receipt is None
                else authority_receipt["receipt_sha256"]
            ),
            "records": [row.to_dict() for row in records],
            "output_semantics": (
                "kind_separated_equal_occurrence_uncalibrated_scalp_visible_"
                "onset_evidence_not_clinical_probability_cortical_soz_or_ez"
            ),
            "report_claim_authorized": False,
            "formal_complete_roster_materializer_admitted": (
                authority_receipt is not None
            ),
        }
        return BAIEGCompleteITAKindSeparatedSpatialOutputV1(
            schema_version=BA_IEG_COMPLETE_ITA_KIND_SEPARATED_SPATIAL_SCHEMA_V1,
            implementation_id=self.implementation_id,
            source_complete_ita_output_sha256=source.output_sha256,
            complete_candidate_roster_receipt_sha256=output_body[
                "complete_candidate_roster_receipt_sha256"
            ],
            candidate_roster_authority_receipt_sha256=output_body[
                "candidate_roster_authority_receipt_sha256"
            ],
            formal_complete_roster_materializer_admitted=(
                authority_receipt is not None
            ),
            records=tuple(records),
            output_sha256=_canonical_sha256(output_body),
        )


__all__ = [
    "BA_IEG_COMPLETE_ITA_KIND_SEPARATED_SPATIAL_PROJECTION_ID_V1",
    "BA_IEG_COMPLETE_ITA_KIND_SEPARATED_SPATIAL_SCHEMA_V1",
    "BA_IEG_SPATIAL_UNRESOLVED_KEY_V1",
    "BA_IEG_ITA_LATERALITY_IDS_V1",
    "BA_IEG_ITA_COARSE_REGION_IDS_V1",
    "BA_IEG_ITA_JOINT_REGION_IDS_V1",
    "BAIEGNormalizedEvidenceMassV1",
    "BAIEGKindSeparatedOccurrenceProjectionV1",
    "BAIEGSpatialAxisMassV1",
    "BAIEGKindSeparatedAggregationTrackV1",
    "BAIEGKindSeparatedRecordProjectionV1",
    "BAIEGCompleteITAKindSeparatedSpatialOutputV1",
    "BAIEGCompleteITAKindSeparatedSpatialProjectionV1",
]
