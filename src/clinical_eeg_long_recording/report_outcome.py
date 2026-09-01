"""Deterministic EEG-only recording outcome classification.

The report must remain publishable when no event is localizable and when the
optional language model fails.  This module therefore separates *report
completion* from *SOZ evidence strength*.  It consumes only validated event
fact ledgers; detector scores, research electrode rankings, EDF annotations,
spreadsheets and free-form language never participate.

``completed_localizable`` requires a separately qualified electrographic
seizure/onset fact with a spatial field.  A merely sustained quantitative
change is useful as a Finding but is never promoted to a seizure or SOZ.  The
two abstention outcomes are completed reports, not processing failures.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


REPORT_OUTCOME_SCHEMA_VERSION = "long_term_eeg_diagnostic_outcome_v1"

COMPLETED_LOCALIZABLE = "completed_localizable"
COMPLETED_NONLOCALIZABLE = "completed_nonlocalizable"
COMPLETED_INSUFFICIENT_EVIDENCE = "completed_insufficient_evidence"

_REGION_ZH = {
    "frontal": "额区",
    "temporal": "颞区",
    "central": "中央区",
    "parietal": "顶区",
    "occipital": "枕区",
    "frontotemporal": "额颞区",
    "centrotemporal": "中央颞区",
    "temporoparietal": "颞顶区",
    "posterior": "后头部",
    "midline": "中线区",
}
_LATERALITY_ZH = {"left": "左侧", "right": "右侧", "midline": "中线"}
_NONFOCAL_DISTRIBUTIONS = {
    "bilateral_synchronous",
    "generalized",
    "diffuse",
}


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
    return [_mapping(item, f"bundle.events[{index}]") for index, item in enumerate(raw)]


def qualified_sustained_change_value(
    event: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return the one signal-qualified neutral sustained-change fact.

    This helper is shared by the recording summary and the diagnostic outcome
    so a feature cannot be displayed under a weaker gate than the one used for
    inference.
    """
    event_id = event.get("eeg_event_id")
    report = _mapping(event.get("event_report_payload"), "event report")
    facts = report.get("facts")
    if not isinstance(facts, list):
        raise TypeError("event report facts must be a list")
    matches: list[Mapping[str, Any]] = []
    for raw in facts:
        fact = _mapping(raw, "event report fact")
        if (
            fact.get("fact_type") != "algorithmic_sustained_eeg_change"
            or fact.get("eeg_event_id") != event_id
        ):
            continue
        value = _mapping(fact.get("value"), "sustained-change fact value")
        qualification = value.get("qualification")
        if qualification is None:
            continue
        gate = _mapping(qualification, "sustained-change qualification")
        if any(
            gate.get(key) is not True
            for key in (
                "artifact_gate_passed",
                "sustained_change_gate_passed",
                "reproducibility_gate_passed",
                "source_signal_only",
            )
        ):
            continue
        if any(
            gate.get(key) is not False
            for key in ("external_context_used", "research_ranking_used")
        ):
            continue
        matches.append(value)
    if len(matches) > 1:
        raise ValueError("event has repeated qualified sustained-change facts")
    return matches[0] if matches else None


def qualified_ictal_onset_value(
    event: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return an independently qualified onset fact, if one exists.

    Current v1 ledgers have no automatic electrographic-seizure promotion
    receipt.  Consequently an algorithm-candidate ``ictal_onset_pattern`` is
    not sufficient.  A future externally validated producer may attach the
    explicit qualification object checked below; existing physician-verified
    signal facts remain admissible.  Neither route is populated by the
    quantitative sustained-change producer.
    """

    event_id = event.get("eeg_event_id")
    report = _mapping(event.get("event_report_payload"), "event report")
    facts = report.get("facts")
    if not isinstance(facts, list):
        raise TypeError("event report facts must be a list")
    matches: list[Mapping[str, Any]] = []
    for raw in facts:
        fact = _mapping(raw, "event report fact")
        if fact.get("fact_type") != "ictal_onset_pattern" or fact.get(
            "eeg_event_id"
        ) != event_id:
            continue
        verification = _mapping(fact.get("verification"), "onset verification")
        value = _mapping(fact.get("value"), "ictal onset fact value")
        physician_verified = verification.get("status") == "physician_verified"
        qualification = value.get("qualification")
        qualified_algorithm = False
        if isinstance(qualification, Mapping):
            qualified_algorithm = (
                qualification.get("electrographic_seizure_gate_passed") is True
                and qualification.get("spatial_field_gate_passed") is True
                and qualification.get("morphology_gate_passed") is True
                and qualification.get("artifact_gate_passed") is True
                and qualification.get("source_signal_only") is True
                and qualification.get("external_context_used") is False
                and qualification.get("research_ranking_used") is False
                and qualification.get("promotion_status")
                == "passed_external_validation_gate"
            )
        if physician_verified or qualified_algorithm:
            matches.append(value)
    if len(matches) > 1:
        raise ValueError("event has repeated qualified ictal-onset facts")
    return matches[0] if matches else None


# Private aliases keep older internal imports and frozen notebooks working.
_qualified_sustained_value = qualified_sustained_change_value
_qualified_onset_value = qualified_ictal_onset_value


def _spatial_signature(value: Mapping[str, Any]) -> tuple[str, frozenset[str]] | None:
    laterality = value.get("laterality")
    regions = value.get("regions")
    derivations = value.get("derivations")
    distribution = value.get("distribution")
    if distribution in _NONFOCAL_DISTRIBUTIONS:
        return None
    if laterality not in _LATERALITY_ZH:
        return None
    if not isinstance(regions, list) or not regions:
        return None
    electrodes = value.get("electrodes")
    has_spatial_carrier = (
        isinstance(derivations, list) and bool(derivations)
    ) or (isinstance(electrodes, list) and bool(electrodes))
    if not has_spatial_carrier:
        return None
    normalized_regions = frozenset(
        str(region) for region in regions if str(region) in _REGION_ZH
    )
    if not normalized_regions:
        return None
    return str(laterality), normalized_regions


def _localized_text(laterality: str, regions: frozenset[str]) -> str:
    region_text = "、".join(_REGION_ZH[item] for item in sorted(regions))
    return _LATERALITY_ZH[laterality] + region_text


def _shared_spatial_carriers(
    spatial: list[
        tuple[Mapping[str, Any], Mapping[str, Any], tuple[str, frozenset[str]]]
    ],
) -> tuple[list[str], list[str]]:
    """Return only channels repeated by every spatially supporting event."""

    if not spatial:
        return [], []
    derivation_sets = [
        {str(item) for item in value.get("derivations", [])}
        if isinstance(value.get("derivations"), list)
        else set()
        for _, value, _ in spatial
    ]
    electrode_sets = []
    for _, value, _ in spatial:
        carrier = value.get("maximal_electrodes")
        if not isinstance(carrier, list) or not carrier:
            carrier = value.get("electrodes")
        electrode_sets.append(
            {str(item) for item in carrier} if isinstance(carrier, list) else set()
        )
    shared_derivations = set.intersection(*derivation_sets) if derivation_sets else set()
    shared_electrodes = set.intersection(*electrode_sets) if electrode_sets else set()
    return sorted(shared_derivations), sorted(shared_electrodes)


def classify_recording_eeg_outcome(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a completed EEG report without inventing a diagnosis.

    A single-event record may carry a lower-strength scalp localization
    tendency.  For a multi-event record, localizability requires at least two
    mutually consistent qualified spatial events; one isolated spatial event
    among several candidates is reported as nonlocalizable.
    """

    source = deepcopy(dict(_mapping(bundle, "long-term EEG bundle")))
    events = _events(source)
    qualified_changes: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    qualified: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    spatial: list[
        tuple[Mapping[str, Any], Mapping[str, Any], tuple[str, frozenset[str]]]
    ] = []
    nonfocal_event_ids: list[str] = []
    for event in events:
        change = qualified_sustained_change_value(event)
        if change is not None:
            qualified_changes.append((event, change))
        value = qualified_ictal_onset_value(event)
        if value is None:
            continue
        qualified.append((event, value))
        if value.get("distribution") in _NONFOCAL_DISTRIBUTIONS:
            nonfocal_event_ids.append(str(event.get("eeg_event_id")))
        signature = _spatial_signature(value)
        if signature is not None:
            spatial.append((event, value, signature))

    event_ids = [str(event.get("eeg_event_id")) for event, _ in qualified]
    spatial_event_ids = [str(event.get("eeg_event_id")) for event, _, _ in spatial]
    evidence_reasons: list[str] = []
    shared_derivations: list[str] = []
    shared_electrodes: list[str] = []

    if not qualified:
        report_status = COMPLETED_INSUFFICIENT_EVIDENCE
        conclusion_code = "soz_not_assessable_insufficient_evidence"
        evidence_reasons.append(
            "no_event_has_independently_qualified_electrographic_seizure_onset"
        )
        if qualified_changes:
            conclusion = (
                f"当前形成 {len(qualified_changes)} 个通过量化信号门槛的持续变化候选，"
                "但尚无通过独立电图发作与空间电场资格门槛的起始事实；"
                "SOZ 定位证据不足，无法判断。"
            )
            evidence_reasons.append(
                "quantitative_change_candidates_not_promoted_to_electrographic_seizure"
            )
        else:
            conclusion = (
                "当前头皮 EEG 未形成通过伪迹、持续性及复现性门槛的最早持续变化事实；"
                "SOZ 定位证据不足，无法判断。"
            )
        evidence_tier = "abstention"
        consensus = None
    else:
        consensus: tuple[str, frozenset[str]] | None = None
        if spatial:
            lateralities = {signature[0] for _, _, signature in spatial}
            common_regions = set(spatial[0][2][1])
            for _, _, signature in spatial[1:]:
                common_regions.intersection_update(signature[1])
            if len(lateralities) == 1 and common_regions:
                consensus = (next(iter(lateralities)), frozenset(common_regions))

        single_event_localizable = len(qualified) == 1 and len(spatial) == 1
        repeated_localizable = len(spatial) >= 2 and consensus is not None
        no_conflicting_qualified_spatial = (
            consensus is not None
            and all(item[2][0] == consensus[0] for item in spatial)
        )
        if single_event_localizable or (
            repeated_localizable and no_conflicting_qualified_spatial
        ):
            assert consensus is not None
            consensus_spatial = [
                item
                for item in spatial
                if item[2][0] == consensus[0]
                and bool(set(item[2][1]).intersection(consensus[1]))
            ]
            shared_derivations, shared_electrodes = _shared_spatial_carriers(
                consensus_spatial
            )
            report_status = COMPLETED_LOCALIZABLE
            conclusion_code = "scalp_eeg_onset_tendency_localizable"
            localized = _localized_text(*consensus)
            if single_event_localizable:
                evidence_tier = "single_event_scalp_tendency"
                evidence_reasons.append("one_qualified_spatially_restricted_event")
                prefix = "单个合格事件的"
            else:
                evidence_tier = "repeated_consistent_scalp_tendency"
                evidence_reasons.append(
                    "at_least_two_qualified_events_with_consistent_laterality_and_region"
                )
                prefix = f"{len(spatial)} 个合格事件的"
            carrier_text = ""
            if shared_derivations:
                carrier_text += "共同支持双极导联为" + "、".join(shared_derivations) + "。"
            if shared_electrodes:
                carrier_text += "共同支持头皮电极为" + "、".join(shared_electrodes) + "。"
            conclusion = (
                f"{prefix}最早持续头皮 EEG 变化定位倾向指向{localized}。"
                + carrier_text
                + "该结论仅为头皮 EEG 定位倾向，不等同于皮层 SOZ、致痫区或治疗靶点。"
            )
        else:
            report_status = COMPLETED_NONLOCALIZABLE
            conclusion_code = "soz_not_localizable_from_current_scalp_eeg"
            evidence_tier = "abstention"
            if nonfocal_event_ids and len(nonfocal_event_ids) == len(qualified):
                evidence_reasons.append(
                    "qualified_events_have_bilateral_synchronous_or_diffuse_scalp_distribution"
                )
                conclusion = (
                    "合格事件呈双侧同步或弥漫性头皮分布，未形成稳定领先的局灶区域；"
                    "无法据此定位局灶 SOZ，且该表现本身不能用于诊断广泛性癫痫。"
                )
            elif not spatial:
                evidence_reasons.append("qualified_change_without_localizing_spatial_fact")
                conclusion = (
                    "已形成合格的持续头皮 EEG 变化事实，但缺少通过资格门槛的稳定"
                    "侧别/区域证据；SOZ 无法定位。"
                )
            elif len(events) > 1 and len(spatial) == 1:
                evidence_reasons.append(
                    "only_one_spatially_qualified_event_among_multiple_candidates"
                )
                conclusion = (
                    "多事件记录中仅一个事件形成合格空间倾向，缺少跨事件复现；"
                    "当前 SOZ 无法可靠定位。"
                )
            else:
                evidence_reasons.append("qualified_spatial_events_are_inconsistent")
                conclusion = (
                    "各合格事件的最早持续头皮 EEG 变化侧别或区域不一致，"
                    "不能合并为统一定位结论；当前 SOZ 无法定位。"
                )

    return {
        "schema_version": REPORT_OUTCOME_SCHEMA_VERSION,
        "report_status": report_status,
        "soz_conclusion_code": conclusion_code,
        "evidence_tier": evidence_tier,
        "event_count": len(events),
        "qualified_event_count": len(qualified),
        "qualified_change_candidate_count": len(qualified_changes),
        "spatially_qualified_event_count": len(spatial),
        "supporting_event_ids": event_ids,
        "spatial_supporting_event_ids": spatial_event_ids,
        "nonfocal_supporting_event_ids": nonfocal_event_ids,
        "spatial_consensus": (
            {
                "laterality": consensus[0],
                "regions": sorted(consensus[1]),
                "supporting_event_count": len(spatial),
                "shared_derivations": shared_derivations,
                "shared_electrodes": shared_electrodes,
            }
            if consensus is not None
            else None
        ),
        "evidence_reasons": evidence_reasons,
        "conclusion_zh": conclusion,
        "boundary_zh": (
            "结论仅依据本流程实际生成并通过门控的头皮 EEG 信号事实；"
            "未使用 EDF annotation、Excel、病史或其他临床资料。"
        ),
    }


__all__ = [
    "COMPLETED_INSUFFICIENT_EVIDENCE",
    "COMPLETED_LOCALIZABLE",
    "COMPLETED_NONLOCALIZABLE",
    "REPORT_OUTCOME_SCHEMA_VERSION",
    "classify_recording_eeg_outcome",
    "qualified_ictal_onset_value",
    "qualified_sustained_change_value",
]
