"""Deterministic aggregation of event EEG Findings into a record SOZ graph.

This module is deliberately not a prose generator.  It consumes only validated
``event_eeg_findings_v1`` payloads, preserves event ownership and heuristic
event-similarity groups,
and emits a closed ``clinical_eeg_multievent_soz_report_v1`` claim graph.  The
record validator is called before returning, so a partially closed graph is
never exposed to downstream lexicalization.

The v1 ranking policy is intentionally transparent:

* duplicated event/waveform material is counted once;
* every retained event contributes one reciprocal-rank vote per spatial axis;
* only present ``onset_support`` Findings can positively support SOZ;
* spread/context Findings remain in the catalog but never enter SOZ support;
* bipolar lead evidence may back off to a declared region, but is never split
  into its endpoint electrodes;
* spatial resolution is emitted only when a trusted patient-disjoint risk
  controller receipt is supplied.  Otherwise the graph backs off to phenotype.

No EDF annotation, spreadsheet field, doctor label, clinical text, patient
metadata, video, auxiliary physiology, sleep stage, or provocation result is
accepted by this API.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from .event_findings_validation import (
    event_term_decision_source_binding_sha256,
    validate_event_eeg_findings_payload,
)
from .multievent_soz_claim_validation import (
    MULTIEVENT_SOZ_REPORT_SCHEMA_VERSION,
    validate_multievent_soz_report_payload,
)


DETERMINISTIC_MULTIEVENT_AGGREGATION_POLICY = (
    "event_equal_reciprocal_rank_hierarchical_mode_preserving_v2"
)
LATENT_MODE_GROUP_REASON_CODE = (
    "latent_heuristic_event_group_without_event_to_mode_and_event_onset_field_gold"
)
DISCORDANT_EVENT_BACKOFF_REASON_CODE = (
    "discordant_cross_event_onset_evidence_without_mode_identifiability"
)

_FORBIDDEN_SOURCES = [
    "edf_annotations",
    "excel_onset_fields",
    "doctor_labels",
    "clinical_text",
    "patient_metadata",
    "video",
    "ecg_emg_eog",
    "sleep_staging",
    "provocation",
]
_INPUT_SOURCES = [
    "scalp_eeg_signal",
    "eeg_derived_findings",
    "detector_outputs",
    "signal_quality",
    "montage_metadata",
]
_PHENOTYPES = {
    "focal",
    "focal_with_rapid_bilateralization",
    "bilateral_synchronous_or_rapid_bilateralization_ambiguous",
    "generalized_synchronous",
    "multiple_scalp_onset_modes",
    "scalp_onset_nonlocalizable",
}
_LATERALITIES = {"left", "right", "bilateral", "midline", "indeterminate"}
_SPATIAL_RESOLUTIONS = {"electrode", "region", "laterality"}
_RESOLUTION_SPECIFICITY = {
    "phenotype_only": 0,
    "laterality": 1,
    "region": 2,
    "electrode": 3,
}
_TOL = 1e-6


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


def _bounded_id(prefix: str, *parts: object) -> str:
    raw = ":".join([prefix, *(str(part) for part in parts)])
    if len(raw) <= 240:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _as_receipt_registry(
    value: Mapping[str, Mapping[str, object]] | Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if isinstance(value, Mapping):
        rows: Iterable[tuple[str, Mapping[str, object]]] = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows = ((str(row.get("receipt_id", "")), row) for row in value)
    else:
        raise TypeError("producer_receipts must be a mapping or a sequence of receipts")

    result: dict[str, dict[str, object]] = {}
    for key, raw in rows:
        if not isinstance(key, str) or not key or not isinstance(raw, Mapping):
            raise TypeError("producer_receipts contains an invalid entry")
        row = deepcopy(dict(raw))
        if row.get("receipt_id") != key:
            raise ValueError("producer receipt key/receipt_id mismatch")
        if key in result:
            raise ValueError(f"duplicate producer receipt_id {key!r}")
        result[key] = row
    return result


def _receipt_for_type(
    registry: Mapping[str, Mapping[str, object]],
    producer_type: str,
    *,
    required: bool = True,
) -> str | None:
    identifiers = sorted(
        receipt_id
        for receipt_id, row in registry.items()
        if row.get("producer_type") == producer_type
    )
    if not identifiers:
        if required:
            raise ValueError(f"producer_receipts lacks {producer_type!r}")
        return None
    return identifiers[0]


def _interval_resolution(bundle: Mapping[str, Any]) -> float:
    interval = bundle["window"]["onset_interval"].get("interval")
    if isinstance(interval, Mapping):
        value = float(interval.get("resolution_seconds", 0.001))
        if math.isfinite(value) and value > 0:
            return value
    return 0.001


def _onset_bounds(bundle: Mapping[str, Any]) -> tuple[float, float] | None:
    interval = bundle["window"]["onset_interval"].get("interval")
    status = str(bundle["window"]["onset_interval"].get("status"))
    if not isinstance(interval, Mapping) or status not in {
        "observed",
        "interval_estimate",
        "legacy_interval_estimate",
    }:
        return None
    return float(interval["lower"]), float(interval["upper"])


def _physical_identity(bundle: Mapping[str, Any]) -> tuple[float, ...]:
    final = bundle["window"]["final_interval"]
    return (
        round(float(final[0]), 6),
        round(float(final[1]), 6),
    )


def _waveform_identity(bundle: Mapping[str, Any]) -> tuple[object, ...]:
    rows = []
    for waveform in bundle["waveform_evidence"]:
        rows.append(
            (
                tuple(round(float(item), 6) for item in waveform["interval"]),
                tuple(sorted(str(item) for item in waveform["unit_ids"])),
                str(waveform["signal_sha256"]),
            )
        )
    return tuple(sorted(rows))


def _deduplicate_events(
    bundles: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate exact IDs, physical intervals, or waveform identities.

    When aliases collide, the lexicographically smallest canonical payload hash
    is retained.  Conflicting payloads with the same event ID are rejected,
    because silently choosing between two meanings of one identifier would make
    downstream event ownership ambiguous.
    """

    by_event_id: dict[str, str] = {}
    candidates: list[tuple[str, dict[str, Any]]] = []
    for bundle in bundles:
        row = deepcopy(dict(bundle))
        digest = _sha256(row)
        event_id = str(row["event_id"])
        previous = by_event_id.get(event_id)
        if previous is not None and previous != digest:
            raise ValueError(f"event_id {event_id!r} has conflicting payloads")
        by_event_id[event_id] = digest
        candidates.append((digest, row))

    # Union aliases so transitive duplicates are also counted once.
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    identities: dict[tuple[str, object], int] = {}
    for index, (_, bundle) in enumerate(candidates):
        keys: list[tuple[str, object]] = [
            ("event", str(bundle["event_id"])),
            ("physical", _physical_identity(bundle)),
        ]
        waveform_identity = _waveform_identity(bundle)
        if waveform_identity:
            keys.append(("waveform", waveform_identity))
        for key in keys:
            if key in identities:
                union(index, identities[key])
            else:
                identities[key] = index

    groups: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        groups[find(index)].append(candidate)
    retained = [
        min(rows, key=lambda item: (item[0], str(item[1]["event_id"])))[1]
        for rows in groups.values()
    ]
    return sorted(
        retained,
        key=lambda row: (
            float(row["window"]["final_interval"][0]),
            str(row["event_id"]),
        ),
    )


def _score_order(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed: set[str] | None = None,
    chosen: str | None = None,
) -> list[str]:
    ordered = [
        str(row["name"] if "name" in row else row["candidate_id"])
        for row in sorted(
            rows,
            key=lambda row: (
                -float(row.get("score", 0.0)),
                str(row.get("name", row.get("candidate_id", ""))),
            ),
        )
    ]
    if allowed is not None:
        ordered = [item for item in ordered if item in allowed]
    result: list[str] = []
    if chosen is not None and (allowed is None or chosen in allowed):
        result.append(chosen)
    for item in ordered:
        if item not in result:
            result.append(item)
    return result


def _axis_payload(entries: Sequence[tuple[str, float]]) -> dict[str, Any]:
    return {
        "score_semantics": "uncalibrated_ranking_score",
        "calibration_receipt_id": None,
        "entries": [
            {"rank": rank, "candidate_id": candidate, "score": float(score)}
            for rank, (candidate, score) in enumerate(entries, start=1)
        ],
        "prediction_set": [],
    }


def _event_axis(order: Sequence[str]) -> list[tuple[str, float]]:
    return [(candidate, 1.0 / rank) for rank, candidate in enumerate(order, start=1)]


def _aggregate_axis(
    states: Sequence[Mapping[str, Any]], axis_name: str
) -> list[tuple[str, float]]:
    totals: dict[str, float] = defaultdict(float)
    denominator = float(len(states))
    for state in states:
        for rank, candidate in enumerate(
            state["axis_orders"].get(axis_name, []), start=1
        ):
            totals[str(candidate)] += 1.0 / rank
    return sorted(
        ((candidate, score / denominator) for candidate, score in totals.items()),
        key=lambda item: (-item[1], item[0]),
    )


def _ontology_maps(
    ontology: Mapping[str, Any]
) -> tuple[set[str], dict[str, str], dict[str, str]]:
    electrodes = {str(item) for item in ontology.get("electrode_ids", [])}
    electrode_to_region: dict[str, str] = {}
    region_to_laterality: dict[str, str] = {}
    for region in ontology.get("regions", []):
        region_id = str(region["region_id"])
        region_to_laterality[region_id] = str(region["laterality"])
        for electrode in region["electrode_ids"]:
            electrode_id = str(electrode)
            if electrode_id in electrode_to_region:
                raise ValueError(
                    f"ontology electrode {electrode_id!r} belongs to multiple regions"
                )
            electrode_to_region[electrode_id] = region_id
    if set(electrode_to_region) != electrodes:
        raise ValueError("ontology regions must cover every electrode exactly once")
    return electrodes, electrode_to_region, region_to_laterality


def _event_spatial_state(
    bundle: Mapping[str, Any],
    *,
    electrodes: set[str],
    electrode_to_region: Mapping[str, str],
    region_to_laterality: Mapping[str, str],
) -> dict[str, Any]:
    spatial = bundle["spatial_onset"]
    phenotype_order = _score_order(spatial["phenotype_scores"], allowed=_PHENOTYPES)
    if not phenotype_order:
        phenotype_order = ["scalp_onset_nonlocalizable"]
    phenotype = phenotype_order[0]

    raw_laterality = _score_order(spatial["laterality_scores"], allowed=_LATERALITIES)
    raw_region = _score_order(
        spatial["region_scores"], allowed=set(region_to_laterality)
    )
    top_k = spatial["top_k"]
    electrode_candidates = [
        str(row["candidate_id"])
        for row in top_k
        if row["candidate_type"] == "electrode"
        and str(row["candidate_id"]) in electrodes
    ]

    preferred_resolution = "phenotype_only"
    selected_channel: str | None = None
    selected_region: str | None = None
    selected_laterality: str | None = None
    allowed_resolution = str(spatial["allowed_resolution"])

    if phenotype == "generalized_synchronous":
        if "bilateral" in raw_laterality:
            preferred_resolution = "laterality"
            selected_laterality = "bilateral"
    elif phenotype not in {
        "scalp_onset_nonlocalizable",
        "multiple_scalp_onset_modes",
    }:
        if allowed_resolution == "electrode" and electrode_candidates:
            selected_channel = electrode_candidates[0]
            selected_region = electrode_to_region[selected_channel]
            selected_laterality = region_to_laterality[selected_region]
            preferred_resolution = "electrode"
        elif allowed_resolution in {"electrode", "region", "lead"} and raw_region:
            # A lead may vote for its declared region, but never for either endpoint.
            selected_region = raw_region[0]
            selected_laterality = region_to_laterality[selected_region]
            preferred_resolution = "region"
        elif raw_laterality and raw_laterality[0] != "indeterminate":
            selected_laterality = raw_laterality[0]
            preferred_resolution = "laterality"

    laterality_order = _score_order(
        spatial["laterality_scores"],
        allowed=_LATERALITIES,
        chosen=selected_laterality,
    )
    region_order = _score_order(
        spatial["region_scores"],
        allowed=set(region_to_laterality),
        chosen=selected_region,
    )
    channel_order = []
    if selected_channel is not None:
        channel_order = [selected_channel] + [
            item for item in electrode_candidates if item != selected_channel
        ]

    return {
        "phenotype": phenotype,
        "preferred_resolution": preferred_resolution,
        "mode_top_laterality": raw_laterality[0] if raw_laterality else None,
        "mode_top_region": raw_region[0] if raw_region else None,
        "selected_laterality": selected_laterality,
        "selected_region": selected_region,
        "selected_channel": selected_channel,
        "axis_orders": {
            "phenotype": _score_order(
                spatial["phenotype_scores"], allowed=_PHENOTYPES, chosen=phenotype
            ),
            "laterality": laterality_order,
            "region": region_order,
            "channel": channel_order,
        },
    }


def _mode_signature(state: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(state["phenotype"]),
        str(state.get("mode_top_laterality") or "none"),
        str(state.get("mode_top_region") or "none"),
        str(state["preferred_resolution"]),
        str(
            state.get("selected_channel")
            or state.get("selected_region")
            or state.get("selected_laterality")
            or "none"
        ),
    )


def _mode_states_hierarchically_compatible(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    """Return whether two events can belong to one onset mode.

    Missing finer-resolution axes behave as unknowns, not disagreements.  Two
    observed axes must agree at every shared level, so a region-only left
    temporal event can join a T7/left-temporal event, while observed T7 and F7
    channel hypotheses remain distinct even though they share a region.
    """

    if str(left["phenotype"]) != str(right["phenotype"]):
        return False
    for key in ("selected_laterality", "selected_region", "selected_channel"):
        left_value = left.get(key)
        right_value = right.get(key)
        if (
            left_value is not None
            and right_value is not None
            and left_value != right_value
        ):
            return False
    return True


def _common_mode_resolution(states: Sequence[Mapping[str, Any]]) -> str:
    if not states:
        raise ValueError("a mode requires at least one event state")
    resolutions = [str(row["preferred_resolution"]) for row in states]
    if any(item not in _RESOLUTION_SPECIFICITY for item in resolutions):
        raise ValueError("mode contains an unsupported preferred resolution")
    return min(resolutions, key=lambda item: (_RESOLUTION_SPECIFICITY[item], item))


def _group_hierarchical_mode_states(
    states: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Build deterministic complete-link modes without resolution fragmentation.

    More specific events are placed first.  A coarser event joins a mode only
    when it is compatible with every member and the destination is unique.
    If it is compatible with multiple distinct fine modes, it remains in a
    separate coarse mode rather than arbitrarily bridging them.  Repeated
    events with the same exact signature preferentially join each other.
    """

    ordered = sorted(
        states,
        key=lambda row: (
            -_RESOLUTION_SPECIFICITY[str(row["preferred_resolution"])],
            _mode_signature(row),
            str(row["event_id"]),
        ),
    )
    groups: list[list[dict[str, Any]]] = []
    for state in ordered:
        compatible = [
            index
            for index, members in enumerate(groups)
            if all(
                _mode_states_hierarchically_compatible(state, member)
                for member in members
            )
        ]
        exact = [
            index
            for index in compatible
            if all(
                _mode_signature(state) == _mode_signature(member)
                for member in groups[index]
            )
        ]
        destinations = exact if len(exact) == 1 else compatible
        if len(destinations) == 1:
            groups[destinations[0]].append(state)
        else:
            groups.append([state])
    for members in groups:
        members.sort(key=lambda row: str(row["event_id"]))
    return sorted(
        groups,
        key=lambda members: (
            _mode_signature(members[0]),
            tuple(str(row["event_id"]) for row in members),
        ),
    )


def _risk_control(
    resolution: str,
    *,
    risk_receipt_id: str | None,
) -> dict[str, Any]:
    if resolution in _SPATIAL_RESOLUTIONS and risk_receipt_id is not None:
        rejected = {
            "electrode": [],
            "region": ["electrode_resolution_not_supported"],
            "laterality": [
                "electrode_resolution_not_supported",
                "region_resolution_not_supported",
            ],
        }[resolution]
        return {
            "status": "passed",
            "selected_resolution": resolution,
            "policy_receipt_id": risk_receipt_id,
            # A deterministic v1 baseline uses the only universally valid upper
            # bound.  It does not mislabel ranking scores as probabilities.
            "estimated_conditional_risk": 1.0,
            "risk_limit": 1.0,
            "risk_semantics": "patient_disjoint_conditional_error_estimate",
            "finer_resolution_rejected_reason_codes": rejected,
        }
    if resolution == "technical_limited":
        return {
            "status": "not_applicable_technical",
            "selected_resolution": resolution,
            "policy_receipt_id": None,
            "estimated_conditional_risk": None,
            "risk_limit": None,
            "risk_semantics": "not_available",
            "finer_resolution_rejected_reason_codes": [],
        }
    return {
        "status": "backoff_no_risk_calibration",
        "selected_resolution": resolution,
        "policy_receipt_id": None,
        "estimated_conditional_risk": None,
        "risk_limit": None,
        "risk_semantics": "not_available",
        "finer_resolution_rejected_reason_codes": [
            "finer_spatial_resolution_not_risk_calibrated"
        ],
    }


def _effective_resolution(preferred: str, risk_receipt_id: str | None) -> str:
    if preferred in _SPATIAL_RESOLUTIONS and risk_receipt_id is None:
        return "phenotype_only"
    return preferred


def _hypothesis_axes(
    state_rows: Sequence[Mapping[str, Any]],
    *,
    phenotype: str,
    resolution: str,
    multiple_modes: bool = False,
) -> dict[str, dict[str, Any] | None]:
    if resolution == "technical_limited":
        return {
            "phenotype_scores": None,
            "laterality_scores": None,
            "region_scores": None,
            "channel_scores": None,
        }
    if multiple_modes:
        underlying = _aggregate_axis(state_rows, "phenotype")
        underlying = [
            (candidate, score)
            for candidate, score in underlying
            if candidate != phenotype
        ]
        phenotype_entries = [(phenotype, 2.0), *underlying]
    else:
        phenotype_entries = _aggregate_axis(state_rows, "phenotype")
        if not phenotype_entries or phenotype_entries[0][0] != phenotype:
            phenotype_entries = [(phenotype, 2.0)] + [
                row for row in phenotype_entries if row[0] != phenotype
            ]
    result: dict[str, dict[str, Any] | None] = {
        "phenotype_scores": _axis_payload(phenotype_entries),
        "laterality_scores": None,
        "region_scores": None,
        "channel_scores": None,
    }
    required = {
        "phenotype_only": (),
        "multiple_modes": (),
        "laterality": ("laterality",),
        "region": ("laterality", "region"),
        "electrode": ("laterality", "region", "channel"),
    }[resolution]
    for axis in required:
        entries = _aggregate_axis(state_rows, axis)
        if not entries:
            raise ValueError(
                f"cannot emit {resolution} resolution without {axis} ranking"
            )
        result[f"{axis}_scores"] = _axis_payload(entries)
    return result


def _reason_codes(phenotype: str | None, resolution: str) -> list[str]:
    codes: list[str] = [DETERMINISTIC_MULTIEVENT_AGGREGATION_POLICY]
    if phenotype == "scalp_onset_nonlocalizable":
        codes.append("no_stable_focal_onset_field")
    if phenotype == "multiple_scalp_onset_modes":
        codes.append("distinct_event_onset_modes_preserved")
    if resolution == "phenotype_only":
        codes.append("spatial_granularity_backoff")
    if resolution == "technical_limited":
        codes.append("no_usable_onset_support")
    return codes


def _hypothesis(
    *,
    hypothesis_id: str,
    scope: str,
    role: str,
    event_id: str | None,
    mode_id: str | None,
    phenotype: str | None,
    preferred_resolution: str,
    state_rows: Sequence[Mapping[str, Any]],
    supporting_evidence_ids: Sequence[str],
    contradictory_evidence_ids: Sequence[str],
    evidence_owner: Mapping[str, str],
    model_receipt_id: str,
    risk_receipt_id: str | None,
    multiple_modes: bool = False,
) -> dict[str, Any]:
    resolution = _effective_resolution(preferred_resolution, risk_receipt_id)
    if resolution == "multiple_modes":
        # Multiple modes is an epistemic backoff, not a calibrated spatial axis.
        risk_for_hypothesis = None
    else:
        risk_for_hypothesis = risk_receipt_id
    axes = _hypothesis_axes(
        state_rows,
        phenotype=str(phenotype) if phenotype is not None else "focal",
        resolution=resolution,
        multiple_modes=multiple_modes,
    )
    support = sorted(set(str(item) for item in supporting_evidence_ids))
    contradiction = sorted(set(str(item) for item in contradictory_evidence_ids))
    return {
        "hypothesis_id": hypothesis_id,
        "layer": "research_ai_hypothesis",
        "scope": scope,
        "role": role,
        "event_id": event_id,
        "mode_id": mode_id,
        "core_claim_id": _bounded_id("CORE", hypothesis_id),
        "phenotype": phenotype,
        "selected_resolution": resolution,
        **axes,
        "supporting_event_ids": sorted({evidence_owner[item] for item in support}),
        "contradictory_event_ids": sorted(
            {evidence_owner[item] for item in contradiction}
        ),
        "supporting_evidence_ids": support,
        "contradictory_evidence_ids": contradiction,
        "reason_codes": _reason_codes(phenotype, resolution),
        "model_receipt_id": model_receipt_id,
        "risk_control": _risk_control(resolution, risk_receipt_id=risk_for_hypothesis),
    }


def _none_time() -> dict[str, Any]:
    return {
        "kind": "none",
        "timebase": "not_applicable",
        "lower": None,
        "upper": None,
        "left_censored": False,
        "right_censored": False,
    }


def _hypothesis_entities(hypothesis: Mapping[str, Any]) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    if hypothesis["phenotype"] is not None:
        entities.append({"type": "phenotype", "id": str(hypothesis["phenotype"])})
    for axis_name, entity_type in (
        ("laterality_scores", "laterality"),
        ("region_scores", "region"),
        ("channel_scores", "electrode"),
    ):
        axis = hypothesis[axis_name]
        if axis is not None:
            entities.append(
                {
                    "type": entity_type,
                    "id": str(axis["entries"][0]["candidate_id"]),
                }
            )
    return entities


def _core_predicate_frame(hypothesis: Mapping[str, Any]) -> tuple[str, str]:
    scope = str(hypothesis["scope"])
    phenotype = hypothesis["phenotype"]
    has_counter = bool(hypothesis["contradictory_evidence_ids"])
    if scope == "event":
        return "event_supports_soz_candidate", "event_hypothesis_v1"
    if scope == "mode":
        return (
            "mode_supports_soz_candidate",
            "mode_recurrence_with_counterevidence_v1"
            if has_counter
            else "mode_recurrence_v1",
        )
    if hypothesis["selected_resolution"] == "technical_limited":
        return "record_technical_limited", "record_technical_limited_v1"
    if phenotype == "multiple_scalp_onset_modes":
        return "record_has_multiple_onset_modes", "record_multiple_modes_v1"
    if phenotype == "generalized_synchronous":
        return (
            "record_has_generalized_synchronous_onset",
            "record_generalized_synchronous_v1",
        )
    if phenotype == "scalp_onset_nonlocalizable":
        return "record_onset_nonlocalizable", "record_nonlocalizable_v1"
    if hypothesis["role"] == "alternative":
        return "record_alternative_soz_hypothesis", "record_alternative_hypothesis_v1"
    return (
        "record_primary_soz_hypothesis",
        "record_primary_hypothesis_with_counterevidence_v1"
        if has_counter
        else "record_primary_focal_hypothesis_v1",
    )


def _observation_predicate_frame(phenotype: str) -> tuple[str, str]:
    if phenotype == "generalized_synchronous":
        return (
            "bilateral_synchronous_evolution_observed",
            "record_generalized_synchronous_v1",
        )
    if phenotype == "scalp_onset_nonlocalizable":
        return "no_stable_focal_lead_observed", "record_nonlocalizable_v1"
    return "earliest_sustained_change_maximal_at", "event_onset_maximal_at_v1"


def _finding_entities(
    finding: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    phenotype: str,
    fallback_hypothesis: Mapping[str, Any],
    electrodes: set[str],
    electrode_to_region: Mapping[str, str],
    region_to_laterality: Mapping[str, str],
) -> list[dict[str, str]]:
    if phenotype in {"generalized_synchronous", "scalp_onset_nonlocalizable"}:
        return [{"type": "phenotype", "id": phenotype}]

    lateralities: set[str] = set()
    regions: set[str] = set()
    channels: set[str] = set()
    input_units = bundle["montage"]["input_units"]
    for support in finding["spatial_support"]:
        unit_type = str(support["unit_type"])
        identifier = str(support["id"])
        if unit_type == "laterality" and identifier in _LATERALITIES:
            lateralities.add(identifier)
        elif unit_type == "region" and identifier in region_to_laterality:
            regions.add(identifier)
            lateralities.add(region_to_laterality[identifier])
        elif unit_type == "electrode" and identifier in electrodes:
            channels.add(identifier)
            region_id = electrode_to_region[identifier]
            regions.add(region_id)
            lateralities.add(region_to_laterality[region_id])
        elif unit_type in {"lead", "input_unit"}:
            for input_unit in input_units:
                if identifier not in {
                    str(input_unit["unit_id"]),
                    str(input_unit["canonical_name"]),
                }:
                    continue
                region_id = str(input_unit["region"])
                laterality = str(input_unit["laterality"])
                if region_id in region_to_laterality:
                    regions.add(region_id)
                if laterality in _LATERALITIES:
                    lateralities.add(laterality)

    entities = [
        {"type": "laterality", "id": identifier} for identifier in sorted(lateralities)
    ]
    entities.extend(
        {"type": "region", "id": identifier} for identifier in sorted(regions)
    )
    entities.extend(
        {"type": "electrode", "id": identifier} for identifier in sorted(channels)
    )
    return entities or _hypothesis_entities(fallback_hypothesis)


def _base_claim(
    *,
    claim_id: str,
    layer: str,
    claim_kind: str,
    subject: Mapping[str, str],
    predicate: str,
    entities: Sequence[Mapping[str, str]],
    event_id: str | None,
    mode_id: str | None,
    hypothesis_id: str | None,
    time: Mapping[str, Any],
    epistemic_status: str,
    evidence_ids: Sequence[str],
    producer_receipt_id: str,
    qualification_receipt_id: str | None,
    term_decision_receipt_id: str | None,
    frame: str,
    mandatory: bool,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "layer": layer,
        "claim_kind": claim_kind,
        "subject": dict(subject),
        "predicate": predicate,
        "object_or_value": {
            "entities": [dict(item) for item in entities],
            "measurements": [],
            "code": None,
        },
        "event_id": event_id,
        "mode_id": mode_id,
        "hypothesis_id": hypothesis_id,
        "time": dict(time),
        "polarity": "affirmed",
        "negation_scope": "none",
        "epistemic_status": epistemic_status,
        "evidence_ids": list(evidence_ids),
        "producer_receipt_id": producer_receipt_id,
        "qualification_receipt_id": qualification_receipt_id,
        "term_decision_receipt_id": term_decision_receipt_id,
        "allowed_surface_frames": [frame],
        "mandatory_for_report": mandatory,
        "supporting_relation_claim_ids": [],
        "contradictory_relation_claim_ids": [],
    }


def _build_claim_graph(
    *,
    record_id: str,
    hypotheses: Sequence[Mapping[str, Any]],
    event_states: Sequence[Mapping[str, Any]],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    finding_by_evidence: Mapping[str, Mapping[str, Any]],
    claim_receipt_id: str,
    electrodes: set[str],
    electrode_to_region: Mapping[str, str],
    region_to_laterality: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    claims: list[dict[str, Any]] = []
    event_by_id = {str(row["event_id"]): row for row in event_states}
    observation_by_evidence: dict[str, str] = {}

    used_evidence = sorted(
        {
            str(item)
            for hypothesis in hypotheses
            for field in ("supporting_evidence_ids", "contradictory_evidence_ids")
            for item in hypothesis[field]
        }
    )
    for evidence_id in used_evidence:
        evidence = evidence_by_id[evidence_id]
        event_id = str(evidence["event_id"])
        event = event_by_id[event_id]
        phenotype = str(event["phenotype"])
        if evidence["evidence_role"] == "contradiction":
            predicate, frame = (
                "earliest_sustained_change_maximal_at",
                "event_onset_maximal_at_v1",
            )
        else:
            predicate, frame = _observation_predicate_frame(phenotype)
        onset = event["event_row"]["onset_interval"]
        finding = finding_by_evidence[evidence_id]
        claim_id = _bounded_id("OBS", evidence_id)
        observation_by_evidence[evidence_id] = claim_id
        entities = _finding_entities(
            finding,
            bundle=event["bundle"],
            phenotype=phenotype,
            fallback_hypothesis=event["event_hypothesis"],
            electrodes=electrodes,
            electrode_to_region=electrode_to_region,
            region_to_laterality=region_to_laterality,
        )
        claims.append(
            _base_claim(
                claim_id=claim_id,
                layer="eeg_findings_observation",
                claim_kind="observation",
                subject={"type": "finding", "id": str(evidence["finding_id"])},
                predicate=predicate,
                entities=entities,
                event_id=event_id,
                mode_id=str(event["mode_id"]),
                hypothesis_id=None,
                time={
                    "kind": "recording_interval",
                    "timebase": "recording_relative_seconds",
                    "lower": float(onset["lower"]),
                    "upper": float(onset["upper"]),
                    "left_censored": bool(event["bundle"]["window"]["left_censored"]),
                    "right_censored": bool(event["bundle"]["window"]["right_censored"]),
                },
                epistemic_status=str(evidence["assertion_level"]),
                evidence_ids=[evidence_id],
                producer_receipt_id=claim_receipt_id,
                qualification_receipt_id=finding.get("qualification_receipt_id"),
                term_decision_receipt_id=finding.get("term_decision_receipt_id"),
                frame=frame,
                mandatory=False,
            )
        )

    core_by_hypothesis: dict[str, dict[str, Any]] = {}
    frames_by_hypothesis: dict[str, str] = {}
    for hypothesis in hypotheses:
        predicate, frame = _core_predicate_frame(hypothesis)
        hypothesis_id = str(hypothesis["hypothesis_id"])
        frames_by_hypothesis[hypothesis_id] = frame
        subject_type = {
            "event": "eeg_event",
            "mode": "mode",
            "record": "eeg_record",
        }[str(hypothesis["scope"])]
        subject_id = {
            "event": str(hypothesis["event_id"]),
            "mode": str(hypothesis["mode_id"]),
            "record": record_id,
        }[str(hypothesis["scope"])]
        epistemic = {
            "passed": "risk_controlled_hypothesis",
            "backoff_no_risk_calibration": "research_ai_hypothesis",
            "not_applicable_technical": "technical_limited",
        }[str(hypothesis["risk_control"]["status"])]
        core = _base_claim(
            claim_id=str(hypothesis["core_claim_id"]),
            layer="research_ai_hypothesis",
            claim_kind={
                "event": "event_inference",
                "mode": "mode_inference",
                "record": "record_hypothesis",
            }[str(hypothesis["scope"])],
            subject={"type": subject_type, "id": subject_id},
            predicate=predicate,
            entities=_hypothesis_entities(hypothesis),
            event_id=hypothesis["event_id"],
            mode_id=hypothesis["mode_id"],
            hypothesis_id=hypothesis_id,
            time=_none_time(),
            epistemic_status=epistemic,
            evidence_ids=hypothesis["supporting_evidence_ids"],
            producer_receipt_id=claim_receipt_id,
            qualification_receipt_id=None,
            term_decision_receipt_id=None,
            frame=frame,
            mandatory=True,
        )
        core_by_hypothesis[hypothesis_id] = core
        claims.append(core)

    for hypothesis in hypotheses:
        hypothesis_id = str(hypothesis["hypothesis_id"])
        core = core_by_hypothesis[hypothesis_id]
        target_frame = frames_by_hypothesis[hypothesis_id]
        for relation_kind, field, incoming_field in (
            (
                "supports_claim",
                "supporting_evidence_ids",
                "supporting_relation_claim_ids",
            ),
            (
                "contradicts_claim",
                "contradictory_evidence_ids",
                "contradictory_relation_claim_ids",
            ),
        ):
            for evidence_id in hypothesis[field]:
                source_claim_id = observation_by_evidence[str(evidence_id)]
                source_claim = next(
                    row for row in claims if row["claim_id"] == source_claim_id
                )
                relation_id = _bounded_id(
                    "REL",
                    "S" if relation_kind == "supports_claim" else "C",
                    hypothesis_id,
                    evidence_id,
                )
                relation = _base_claim(
                    claim_id=relation_id,
                    layer="research_ai_hypothesis",
                    claim_kind="evidence_relation",
                    subject={"type": "claim", "id": source_claim_id},
                    predicate=relation_kind,
                    entities=[{"type": "claim", "id": str(core["claim_id"])}],
                    event_id=source_claim["event_id"],
                    mode_id=source_claim["mode_id"],
                    hypothesis_id=None,
                    time=_none_time(),
                    epistemic_status="research_ai_hypothesis",
                    evidence_ids=[str(evidence_id)],
                    producer_receipt_id=claim_receipt_id,
                    qualification_receipt_id=None,
                    term_decision_receipt_id=None,
                    frame=target_frame,
                    mandatory=False,
                )
                claims.append(relation)
                core[incoming_field].append(relation_id)

    sentence_specs = [
        (
            str(hypothesis["core_claim_id"]),
            frames_by_hypothesis[str(hypothesis["hypothesis_id"])],
            {
                "event": "ictal_findings",
                "mode": "cross_event_summary",
                "record": "technical_quality"
                if hypothesis["selected_resolution"] == "technical_limited"
                else "impression",
            }[str(hypothesis["scope"])],
        )
        for hypothesis in hypotheses
    ]
    return claims, sentence_specs


def build_deterministic_multievent_soz_report(
    event_bundles: Sequence[Mapping[str, Any]],
    *,
    ontology: Mapping[str, Any],
    producer_receipts: (
        Mapping[str, Mapping[str, object]] | Sequence[Mapping[str, object]]
    ),
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_term_decision_receipts: (Mapping[str, Mapping[str, object]] | None) = None,
    trusted_qualification_receipts: Mapping[str, Mapping[str, object]] | None = None,
    policy_sha256: str,
) -> dict[str, Any]:
    """Build and validate an EEG-only record-level SOZ hypothesis graph.

    ``producer_receipts`` is also used as the host-trusted producer registry for
    final validation.  It must therefore originate from the host application,
    not from an event payload or an LLM response.
    """

    if not isinstance(event_bundles, Sequence) or isinstance(
        event_bundles, (str, bytes, bytearray)
    ):
        raise TypeError("event_bundles must be a sequence")
    if not event_bundles:
        raise ValueError("event_bundles must contain at least one event")
    if not isinstance(ontology, Mapping):
        raise TypeError("ontology must be an object")
    if (
        trusted_capability_qualification_receipts is not None
        and trusted_qualification_receipts is not None
    ):
        raise ValueError(
            "supply only trusted_capability_qualification_receipts or the "
            "legacy trusted_qualification_receipts alias"
        )
    capability_registry = (
        trusted_capability_qualification_receipts
        if trusted_capability_qualification_receipts is not None
        else trusted_qualification_receipts
    )

    validated_events = [
        validate_event_eeg_findings_payload(
            bundle,
            trusted_capability_qualification_receipts=capability_registry,
            trusted_term_decision_receipts=trusted_term_decision_receipts,
        )
        for bundle in event_bundles
    ]
    record_ids = {str(row["provenance"]["record_id"]) for row in validated_events}
    signal_hashes = {
        str(row["provenance"]["signal_sha256"]) for row in validated_events
    }
    durations = {
        round(float(row["coordinates"]["recording_duration_seconds"]), 6)
        for row in validated_events
    }
    policies = {str(row["provenance"]["policy_sha256"]) for row in validated_events}
    if len(record_ids) != 1 or len(signal_hashes) != 1 or len(durations) != 1:
        raise ValueError("all event bundles must belong to one signal record")
    if policies != {str(policy_sha256)}:
        raise ValueError(
            "event bundle policy_sha256 does not match the requested policy"
        )

    registry = _as_receipt_registry(producer_receipts)
    event_receipt_id = str(_receipt_for_type(registry, "event_findings_provider"))
    model_receipt_id = str(
        _receipt_for_type(registry, "hierarchical_mil_hypothesis_model")
    )
    claim_receipt_id = str(_receipt_for_type(registry, "deterministic_claim_builder"))
    planner_receipt_id = str(
        _receipt_for_type(registry, "deterministic_sentence_planner")
    )
    risk_receipt_id = _receipt_for_type(registry, "risk_controller", required=False)

    retained = _deduplicate_events(validated_events)
    capability_receipts_by_id: dict[str, dict[str, Any]] = {}
    term_decision_receipts_by_id: dict[str, dict[str, Any]] = {}
    for bundle in retained:
        for field, destination in (
            ("qualification_receipts", capability_receipts_by_id),
            ("term_decision_receipts", term_decision_receipts_by_id),
        ):
            for receipt in bundle[field]:
                receipt_id = str(receipt["receipt_id"])
                existing = destination.get(receipt_id)
                if existing is not None and _canonical_json(
                    existing
                ) != _canonical_json(receipt):
                    raise ValueError(
                        f"conflicting {field} content for receipt {receipt_id!r}"
                    )
                destination[receipt_id] = deepcopy(receipt)
    ontology_payload = deepcopy(dict(ontology))
    electrodes, electrode_to_region, region_to_laterality = _ontology_maps(
        ontology_payload
    )

    evidence_catalog: list[dict[str, Any]] = []
    evidence_owner: dict[str, str] = {}
    evidence_by_id: dict[str, dict[str, Any]] = {}
    finding_by_evidence: dict[str, dict[str, Any]] = {}
    event_states: list[dict[str, Any]] = []

    for bundle in retained:
        event_id = str(bundle["event_id"])
        local_support = set(
            str(item) for item in bundle["spatial_onset"]["supporting_evidence_ids"]
        )
        local_contradiction = set(
            str(item) for item in bundle["spatial_onset"]["contradictory_evidence_ids"]
        )
        finding_map = {str(row["evidence_id"]): row for row in bundle["findings"]}
        namespaced: dict[str, str] = {}
        for finding in bundle["findings"]:
            local_id = str(finding["evidence_id"])
            evidence_id = _bounded_id("EVID", event_id, local_id)
            finding_id = _bounded_id("FIND", event_id, local_id)
            namespaced[local_id] = evidence_id
            waveform_ids = [
                _bounded_id("WAVE", event_id, item)
                for item in finding["waveform_evidence_ids"]
            ]
            evidence = {
                "evidence_id": evidence_id,
                "event_id": event_id,
                "finding_id": finding_id,
                "family": str(finding["family"]),
                "term": str(finding["term"]),
                "evidence_role": str(finding["evidence_role"]),
                "status": str(finding["status"]),
                "assertion_level": str(finding["assertion_level"]),
                "waveform_evidence_ids": waveform_ids,
                "producer_receipt_id": event_receipt_id,
                "qualification_receipt_id": finding.get("qualification_receipt_id"),
                "term_decision_receipt_id": finding.get("term_decision_receipt_id"),
            }
            evidence_catalog.append(evidence)
            evidence_owner[evidence_id] = event_id
            evidence_by_id[evidence_id] = evidence
            finding_by_evidence[evidence_id] = finding

        support = sorted(
            namespaced[item]
            for item in local_support
            if finding_map[item]["status"] == "present"
            and finding_map[item]["evidence_role"] == "onset_support"
        )
        contradiction = sorted(
            namespaced[item]
            for item in local_contradiction
            if finding_map[item]["status"] != "not_evaluable"
            and finding_map[item]["evidence_role"] == "contradiction"
        )
        spatial_state = _event_spatial_state(
            bundle,
            electrodes=electrodes,
            electrode_to_region=electrode_to_region,
            region_to_laterality=region_to_laterality,
        )
        final_start, final_stop = (
            float(item) for item in bundle["window"]["final_interval"]
        )
        onset = _onset_bounds(bundle) or (final_start, final_stop)
        resolution = _interval_resolution(bundle)
        event_row = {
            "event_id": event_id,
            "event_bundle_sha256": _sha256(bundle),
            "term_decision_source_binding_sha256": (
                event_term_decision_source_binding_sha256(bundle)
            ),
            "analysis_interval": {
                "lower": final_start,
                "upper": final_stop,
                "resolution_seconds": resolution,
            },
            "onset_interval": {
                "lower": onset[0],
                "upper": onset[1],
                "resolution_seconds": resolution,
            },
            "mode_id": None,
            "usable_for_hypothesis": bool(support),
            "finding_evidence_ids": sorted(namespaced.values()),
            "limitation_codes": sorted(
                str(row["code"]) for row in bundle["limitations"]
            ),
        }
        event_states.append(
            {
                **spatial_state,
                "event_id": event_id,
                "bundle": bundle,
                "event_row": event_row,
                "supporting_evidence_ids": support,
                "contradictory_evidence_ids": contradiction,
                "mode_id": None,
            }
        )

    usable_states = [
        row for row in event_states if row["event_row"]["usable_for_hypothesis"]
    ]
    mode_states: list[dict[str, Any]] = []
    hierarchical_groups = _group_hierarchical_mode_states(usable_states)
    for index, members in enumerate(hierarchical_groups, start=1):
        mode_id = f"MODE-{index:03d}"
        for member in members:
            member["mode_id"] = mode_id
            member["event_row"]["mode_id"] = mode_id
        mode_states.append(
            {
                "mode_id": mode_id,
                "signature": tuple(_mode_signature(row) for row in members),
                "members": sorted(members, key=lambda row: str(row["event_id"])),
                "phenotype": str(members[0]["phenotype"]),
                "preferred_resolution": _common_mode_resolution(members),
            }
        )

    hypotheses: list[dict[str, Any]] = []
    for state in usable_states:
        hypothesis = _hypothesis(
            hypothesis_id=_bounded_id("H-EVENT", state["event_id"]),
            scope="event",
            role="event_specific",
            event_id=str(state["event_id"]),
            mode_id=str(state["mode_id"]),
            phenotype=str(state["phenotype"]),
            preferred_resolution=str(state["preferred_resolution"]),
            state_rows=[state],
            supporting_evidence_ids=state["supporting_evidence_ids"],
            contradictory_evidence_ids=state["contradictory_evidence_ids"],
            evidence_owner=evidence_owner,
            model_receipt_id=model_receipt_id,
            risk_receipt_id=risk_receipt_id,
        )
        state["event_hypothesis"] = hypothesis
        hypotheses.append(hypothesis)

    modes: list[dict[str, Any]] = []
    for mode in mode_states:
        members = mode["members"]
        support = sorted(
            {item for row in members for item in row["supporting_evidence_ids"]}
        )
        contradiction = sorted(
            {item for row in members for item in row["contradictory_evidence_ids"]}
        )
        hypothesis_id = _bounded_id("H-MODE", mode["mode_id"])
        hypothesis = _hypothesis(
            hypothesis_id=hypothesis_id,
            scope="mode",
            role="mode_specific",
            event_id=None,
            mode_id=str(mode["mode_id"]),
            phenotype=str(mode["phenotype"]),
            preferred_resolution=str(mode["preferred_resolution"]),
            state_rows=members,
            supporting_evidence_ids=support,
            contradictory_evidence_ids=contradiction,
            evidence_owner=evidence_owner,
            model_receipt_id=model_receipt_id,
            risk_receipt_id=risk_receipt_id,
        )
        hypotheses.append(hypothesis)
        hypothesis["reason_codes"].append(LATENT_MODE_GROUP_REASON_CODE)
        hypothesis["reason_codes"] = sorted(set(hypothesis["reason_codes"]))
        mode["hypothesis"] = hypothesis
        modes.append(
            {
                "mode_id": str(mode["mode_id"]),
                "event_ids": sorted(str(row["event_id"]) for row in members),
                "event_count": len(members),
                "total_usable_event_count": len(usable_states),
                "primary_hypothesis_id": hypothesis_id,
                "hypothesis_ids": [hypothesis_id],
                "supporting_event_ids": hypothesis["supporting_event_ids"],
                "contradictory_event_ids": hypothesis["contradictory_event_ids"],
                "supporting_evidence_ids": hypothesis["supporting_evidence_ids"],
                "contradictory_evidence_ids": hypothesis["contradictory_evidence_ids"],
            }
        )

    record_id = next(iter(record_ids))
    if usable_states:
        if len(mode_states) == 1:
            record_phenotype = str(mode_states[0]["phenotype"])
            preferred_resolution = str(mode_states[0]["preferred_resolution"])
            multiple_modes = False
        else:
            # These groups are deterministic similarity clusters, not
            # identifiable clinical onset modes.  Without a host-trusted,
            # patient-disjoint and exhaustive event->mode plus event-level
            # onset-field reference, the record conclusion must back off and
            # retain only the fact that event-level onset evidence conflicts.
            record_phenotype = "scalp_onset_nonlocalizable"
            preferred_resolution = "phenotype_only"
            multiple_modes = False
        record_support = sorted(
            {item for row in usable_states for item in row["supporting_evidence_ids"]}
        )
        record_contradiction = sorted(
            {
                item
                for row in usable_states
                for item in row["contradictory_evidence_ids"]
            }
        )
        record_hypothesis = _hypothesis(
            hypothesis_id="H-RECORD-PRIMARY",
            scope="record",
            role="primary",
            event_id=None,
            mode_id=None,
            phenotype=record_phenotype,
            preferred_resolution=preferred_resolution,
            state_rows=usable_states,
            supporting_evidence_ids=record_support,
            contradictory_evidence_ids=record_contradiction,
            evidence_owner=evidence_owner,
            model_receipt_id=model_receipt_id,
            risk_receipt_id=risk_receipt_id,
            multiple_modes=multiple_modes,
        )
        if len(mode_states) > 1:
            record_hypothesis["reason_codes"].append(
                DISCORDANT_EVENT_BACKOFF_REASON_CODE
            )
            record_hypothesis["reason_codes"] = sorted(
                set(record_hypothesis["reason_codes"])
            )
        analysis_status = "analyzable"
    else:
        record_hypothesis = _hypothesis(
            hypothesis_id="H-RECORD-PRIMARY",
            scope="record",
            role="primary",
            event_id=None,
            mode_id=None,
            phenotype=None,
            preferred_resolution="technical_limited",
            state_rows=[],
            supporting_evidence_ids=[],
            contradictory_evidence_ids=[],
            evidence_owner=evidence_owner,
            model_receipt_id=model_receipt_id,
            risk_receipt_id=None,
        )
        analysis_status = "technical_limited"
    hypotheses.append(record_hypothesis)

    claims, sentence_specs = _build_claim_graph(
        record_id=record_id,
        hypotheses=hypotheses,
        event_states=usable_states,
        evidence_by_id=evidence_by_id,
        finding_by_evidence=finding_by_evidence,
        claim_receipt_id=claim_receipt_id,
        electrodes=electrodes,
        electrode_to_region=electrode_to_region,
        region_to_laterality=region_to_laterality,
    )
    required_claim_ids = [
        str(row["claim_id"]) for row in claims if row["mandatory_for_report"]
    ]
    sentences = [
        {
            "sentence_id": f"SENT-{index:03d}",
            "section_id": section_id,
            "template_id": frame,
            "claim_ids": [claim_id],
            "claim_order": [claim_id],
            "connector_ids": [],
            "optional_style_choices": {"compact": False},
        }
        for index, (claim_id, frame, section_id) in enumerate(sentence_specs, start=1)
    ]

    payload = {
        "schema_version": MULTIEVENT_SOZ_REPORT_SCHEMA_VERSION,
        "record_id": record_id,
        "analysis_status": analysis_status,
        "provenance": {
            "signal_sha256": next(iter(signal_hashes)),
            "recording_duration_seconds": next(iter(durations)),
            "policy_sha256": str(policy_sha256),
            "input_sources": _INPUT_SOURCES,
            "inference_exclusions": {
                "edf_annotations_used": False,
                "excel_used": False,
                "doctor_labels_used": False,
                "clinical_text_used": False,
                "patient_metadata_used": False,
                "video_used": False,
                "ecg_emg_eog_used": False,
                "sleep_staging_used": False,
                "provocation_used": False,
            },
        },
        "ontology": ontology_payload,
        "producer_receipts": [registry[key] for key in sorted(registry)],
        "calibration_receipts": [],
        "capability_qualification_receipts": [
            capability_receipts_by_id[key] for key in sorted(capability_receipts_by_id)
        ],
        "term_decision_receipts": [
            term_decision_receipts_by_id[key]
            for key in sorted(term_decision_receipts_by_id)
        ],
        "events": [row["event_row"] for row in event_states],
        "evidence_catalog": sorted(
            evidence_catalog, key=lambda row: str(row["evidence_id"])
        ),
        "modes": modes,
        "hypotheses": hypotheses,
        "claims": claims,
        "sentence_plan": {
            "planner_mode": "deterministic_claim_plan",
            "planner_receipt_id": planner_receipt_id,
            "free_text_included": False,
            "required_claim_ids": required_claim_ids,
            "sentences": sentences,
        },
        "report_policy": {
            "report_generation_required": True,
            "free_text_generation_allowed": False,
            "lexicalizer": "deterministic_claim_lexicalizer_v1",
            "qwen_role": "not_used",
            "forbidden_sources": _FORBIDDEN_SOURCES,
        },
    }
    return validate_multievent_soz_report_payload(
        payload,
        trusted_producer_receipts=registry,
        trusted_calibration_receipts={},
        trusted_capability_qualification_receipts=capability_registry,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
    )


__all__ = [
    "DETERMINISTIC_MULTIEVENT_AGGREGATION_POLICY",
    "build_deterministic_multievent_soz_report",
]
