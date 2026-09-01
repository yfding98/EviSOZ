"""Record-level synthesis of signal-qualified long-EEG event findings.

The summary accounts for every detector-selected event, but only promotes
features that passed the same signal gates used by the diagnostic classifier.
Neutral quantitative changes may contribute frequency, rhythmicity,
amplitude, trajectory and bipolar-derivation recurrence.  Morphology (for
example spike/sharp-wave wording), ictal onset, laterality and region are
included only when an independently qualified ``ictal_onset_pattern`` exists.

This module never reads EDF annotations, spreadsheets, doctor labels, research
rankings or free-form LLM text.  Its output is structured evidence for the
deterministic renderer, not a new source of patient facts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Mapping

from .report_outcome import (
    qualified_ictal_onset_value,
    qualified_sustained_change_value,
)


RECORD_FINDINGS_SUMMARY_SCHEMA_VERSION = (
    "long_term_eeg_record_findings_summary_v1"
)
RESEARCH_CHANNEL_CONSISTENCY_SCHEMA_VERSION = (
    "long_term_research_multievent_channel_consistency_v1"
)


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _events(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = bundle.get("events")
    if not isinstance(raw, list):
        raise TypeError("bundle.events must be a list")
    if bundle.get("event_count") != len(raw):
        raise ValueError("bundle.event_count does not match events")
    events = [_mapping(item, f"bundle.events[{index}]") for index, item in enumerate(raw)]
    expected_numbers = list(range(1, len(events) + 1))
    if [item.get("event_number") for item in events] != expected_numbers:
        raise ValueError("bundle events are not in canonical recording order")
    return events


def _strings(value: object, context: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise TypeError(f"{context} must be a string list")
    if len(value) != len(set(value)):
        raise ValueError(f"{context} contains duplicates")
    return list(value)


def _optional_strings(value: object, context: str) -> list[str]:
    if value is None:
        return []
    return _strings(value, context)


def build_recording_eeg_findings_summary(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Build a deterministic all-event feature and recurrence summary."""

    events = _events(bundle)
    event_rows: list[dict[str, Any]] = []
    band_counts: Counter[str] = Counter()
    rhythm_counts: Counter[str] = Counter()
    morphology_counts: Counter[str] = Counter()
    derivation_events: dict[str, list[int]] = defaultdict(list)
    qualified_signal_count = 0
    qualified_onset_count = 0

    for event in events:
        event_number = int(event["event_number"])
        sustained = qualified_sustained_change_value(event)
        onset = qualified_ictal_onset_value(event)
        row: dict[str, Any] = {
            "event_number": event_number,
            "signal_finding_status": "qualified" if sustained is not None else "abstained",
            "qualified_feature_families": [],
            "derivations": [],
            "frequency_band": None,
            "frequency_hz": None,
            "amplitude_uv": None,
            "rhythmicity": None,
            "trajectory_dimensions": [],
            "later_derivations": [],
            "return_to_baseline_candidate": False,
            "qualified_onset": None,
        }
        if sustained is not None:
            qualified_signal_count += 1
            derivations = _strings(
                sustained.get("derivations"),
                f"event {event_number} sustained derivations",
            )
            row["derivations"] = derivations
            row["qualified_feature_families"].append("sustained_change")
            for derivation in derivations:
                derivation_events[derivation].append(event_number)

            band = sustained.get("frequency_band")
            if isinstance(band, str) and band:
                row["frequency_band"] = band
                band_counts[band] += 1
                row["qualified_feature_families"].append("frequency")
            frequency = sustained.get("frequency_hz")
            if isinstance(frequency, Mapping):
                row["frequency_hz"] = deepcopy(dict(frequency))
            amplitude = sustained.get("amplitude_uv")
            if isinstance(amplitude, Mapping):
                row["amplitude_uv"] = deepcopy(dict(amplitude))
                row["qualified_feature_families"].append("amplitude")
            rhythm = sustained.get("rhythmicity")
            if isinstance(rhythm, str) and rhythm:
                row["rhythmicity"] = rhythm
                rhythm_counts[rhythm] += 1
                row["qualified_feature_families"].append("rhythmicity")

            trajectory = sustained.get("quantitative_trajectory")
            if isinstance(trajectory, Mapping):
                if trajectory.get("amplitude_change_alone_is_not_ictal_evolution") is not True:
                    raise ValueError("quantitative trajectory lost its non-ictal boundary")
                dimensions = _strings(
                    trajectory.get("change_dimensions"),
                    f"event {event_number} trajectory dimensions",
                )
                row["trajectory_dimensions"] = dimensions
                row["qualified_feature_families"].append("quantitative_trajectory")

            later = sustained.get("later_derivation_changes")
            if later is not None:
                if not isinstance(later, list):
                    raise TypeError("later_derivation_changes must be a list")
                later_derivations: list[str] = []
                for index, raw in enumerate(later):
                    observation = _mapping(
                        raw,
                        f"event {event_number} later derivation {index}",
                    )
                    derivation = observation.get("derivation")
                    if not isinstance(derivation, str) or not derivation:
                        raise TypeError("later derivation observation is malformed")
                    later_derivations.append(derivation)
                if len(later_derivations) != len(set(later_derivations)):
                    raise ValueError("later derivation observations contain duplicates")
                row["later_derivations"] = later_derivations
                row["qualified_feature_families"].append("later_derivation_timing")

            if sustained.get("candidate_return_to_baseline_offset_seconds") is not None:
                row["return_to_baseline_candidate"] = True
                row["qualified_feature_families"].append("return_to_baseline_candidate")

        if onset is not None:
            qualified_onset_count += 1
            morphology = onset.get("morphology")
            if not isinstance(morphology, str) or not morphology:
                raise TypeError("qualified onset is missing morphology")
            morphology_counts[morphology] += 1
            row["qualified_feature_families"].append("independently_qualified_onset")
            row["qualified_onset"] = {
                "onset_type": onset.get("onset_type"),
                "morphology": morphology,
                "rhythmicity": onset.get("rhythmicity"),
                "laterality": onset.get("laterality"),
                "distribution": onset.get("distribution"),
                "regions": _optional_strings(
                    onset.get("regions"),
                    f"event {event_number} onset regions",
                ),
                "derivations": _optional_strings(
                    onset.get("derivations"),
                    f"event {event_number} onset derivations",
                ),
                "electrodes": _optional_strings(
                    onset.get("electrodes"),
                    f"event {event_number} onset electrodes",
                ),
                "claim_authority": "independently_qualified_ictal_onset",
            }
        event_rows.append(row)

    recurring = [
        {
            "derivation": derivation,
            "event_count": len(event_numbers),
            "event_numbers": event_numbers,
        }
        for derivation, event_numbers in derivation_events.items()
        if len(event_numbers) >= 2
    ]
    recurring.sort(key=lambda item: (-int(item["event_count"]), str(item["derivation"])))

    return {
        "schema_version": RECORD_FINDINGS_SUMMARY_SCHEMA_VERSION,
        "event_count": len(events),
        "all_detected_candidates_accounted_for": len(event_rows) == len(events),
        "qualified_signal_event_count": qualified_signal_count,
        "events_without_qualified_signal_findings": len(events) - qualified_signal_count,
        "qualified_onset_event_count": qualified_onset_count,
        "event_findings": event_rows,
        "cross_event_support": {
            "frequency_band_counts": dict(sorted(band_counts.items())),
            "rhythmicity_counts": dict(sorted(rhythm_counts.items())),
            "morphology_counts_from_independently_qualified_onsets_only": dict(
                sorted(morphology_counts.items())
            ),
            "recurring_bipolar_derivations": recurring,
        },
        "claim_boundary": {
            "neutral_change_is_not_electrographic_seizure": True,
            "bipolar_derivation_recurrence_is_not_laterality_region_or_soz": True,
            "spike_ied_or_other_morphology_requires_independent_qualification": True,
            "edf_annotations_excel_doctor_labels_or_research_rankings_used": False,
        },
    }


def build_research_multievent_channel_consistency(
    bundle: Mapping[str, Any],
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    """Aggregate v29 scalp-channel rankings in a segregated research layer.

    The result is deliberately separate from ``build_recording_eeg_findings_summary``.
    It may be displayed only in a research appendix and must never enter the
    clinical Findings, EEG impression or SOZ/EZ conclusion.
    """

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    events = _events(bundle)
    by_electrode: dict[str, dict[str, Any]] = {}
    ranked_event_count = 0
    for event in events:
        receipt = event.get("research_soz_ranking_receipt")
        if receipt is None:
            continue
        ranking = _mapping(receipt, "research electrode ranking receipt")
        if (
            ranking.get("interpretation_status")
            != "research_scalp_electrode_ranking_not_clinical_soz"
        ):
            raise ValueError("research ranking lost its non-clinical boundary")
        raw = ranking.get("ranked_electrodes")
        if not isinstance(raw, list):
            raise TypeError("ranked_electrodes must be a list")
        ranked_event_count += 1
        seen: set[str] = set()
        for expected_rank, raw_item in enumerate(raw, start=1):
            item = _mapping(raw_item, "ranked scalp electrode")
            if item.get("rank") != expected_rank:
                raise ValueError("research electrode ranking order drifted")
            if expected_rank > top_k:
                break
            electrode = item.get("electrode")
            if not isinstance(electrode, str) or not electrode or electrode in seen:
                raise ValueError("research ranking has an invalid or repeated electrode")
            seen.add(electrode)
            score = item.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError("research electrode score must be numeric")
            aggregate = by_electrode.setdefault(
                electrode,
                {
                    "electrode": electrode,
                    "event_numbers": [],
                    "ranks": [],
                    "scores": [],
                    "top1_event_count": 0,
                },
            )
            aggregate["event_numbers"].append(int(event["event_number"]))
            aggregate["ranks"].append(expected_rank)
            aggregate["scores"].append(float(score))
            if expected_rank == 1:
                aggregate["top1_event_count"] += 1

    rows: list[dict[str, Any]] = []
    for aggregate in by_electrode.values():
        event_support_count = len(aggregate["event_numbers"])
        rows.append(
            {
                "electrode": aggregate["electrode"],
                "event_support_count": event_support_count,
                "ranked_event_count": ranked_event_count,
                "event_support_rate": (
                    event_support_count / ranked_event_count
                    if ranked_event_count
                    else 0.0
                ),
                "top1_event_count": aggregate["top1_event_count"],
                "mean_rank": sum(aggregate["ranks"]) / event_support_count,
                "mean_score": sum(aggregate["scores"]) / event_support_count,
                "event_numbers": aggregate["event_numbers"],
            }
        )
    rows.sort(
        key=lambda item: (
            -int(item["event_support_count"]),
            -int(item["top1_event_count"]),
            float(item["mean_rank"]),
            str(item["electrode"]),
        )
    )
    return {
        "schema_version": RESEARCH_CHANNEL_CONSISTENCY_SCHEMA_VERSION,
        "top_k": top_k,
        "record_event_count": len(events),
        "ranked_event_count": ranked_event_count,
        "channel_support": rows,
        "clinical_use_prohibited": True,
        "entered_clinical_findings_or_impression": False,
        "interpretation": (
            "research_scalp_channel_recurrence_not_clinical_soz_or_ez"
        ),
    }


__all__ = [
    "RECORD_FINDINGS_SUMMARY_SCHEMA_VERSION",
    "RESEARCH_CHANNEL_CONSISTENCY_SCHEMA_VERSION",
    "build_recording_eeg_findings_summary",
    "build_research_multievent_channel_consistency",
]
