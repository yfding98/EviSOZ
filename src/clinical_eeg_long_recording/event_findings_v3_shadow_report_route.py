"""Public/synthetic-only v3 EvidenceGraph shadow report route.

This module connects validated ``event_eeg_findings_v3`` event graphs to the
typed v3 sidecar, a record claim plan, and a deterministic Chinese report.  It
is deliberately separate from the private long-recording production batch:
``ADAPTIVE_REPORT_ROUTE_CONNECTED`` remains ``False`` and ``route_scope`` is
restricted to ``synthetic`` or ``public``.

The existing :mod:`multievent_report_render` renderer is retained for the
frozen v1 claim graph.  Its predicate set cannot faithfully encode v3
periodicity, denominator-bound burden, competing signal hypotheses, or the
six explicit event outcomes.  This route therefore consumes the separate v3
sidecar rather than overloading that legacy renderer.

Every record-plan claim owns one clause.  A complete ordered detector roster
is mandatory, while a missing or invalid event Findings payload becomes a
typed, deterministic not-evaluable event row so a valid shadow record always
has report output.  Record burden is emitted only after a separate complete
mode roster, deduplication ownership roster, canonical interval union, and
evaluable record-time denominator pass their gates.

An optional injected Qwen selector may choose only host-authored clause IDs
from a closed set.  It never receives source IDs or raw facts, cannot submit
free text, and any exception or invalid response falls back to the exact
deterministic report.  No network or model service is called by this module.

This is an engineering shadow path, not a clinical validation claim.  It
performs no file I/O and must never receive EDF annotations, spreadsheets,
physician labels, clinical text, patient metadata, or private production
records.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping, Sequence

from .event_findings_v3_downstream_projection import (
    _evidence_time_permissions,
    project_event_eeg_findings_v3_downstream,
)


ADAPTIVE_REPORT_ROUTE_CONNECTED = False
EVENT_FINDINGS_V3_SHADOW_ROUTE_ID = (
    "public_synthetic_event_findings_v3_shadow_report_route_v1"
)
EVENT_FINDINGS_V3_SHADOW_INPUT_SCHEMA_VERSION = (
    "clinical_eeg_event_findings_v3_shadow_input_v1"
)
EVENT_FINDINGS_V3_SHADOW_ROSTER_SCHEMA_VERSION = (
    "clinical_eeg_event_findings_v3_complete_event_roster_v1"
)
EVENT_FINDINGS_V3_RECORD_CLAIM_PLAN_SCHEMA_VERSION = (
    "clinical_eeg_event_findings_v3_record_claim_plan_v1"
)
EVENT_FINDINGS_V3_CLOSED_LEXICALIZATION_SCHEMA_VERSION = (
    "clinical_eeg_event_findings_v3_closed_lexicalization_v1"
)
EVENT_FINDINGS_V3_REPORT_RENDER_SCHEMA_VERSION = (
    "clinical_eeg_event_findings_v3_report_render_v1"
)
EVENT_FINDINGS_V3_SHADOW_BUNDLE_SCHEMA_VERSION = (
    "clinical_eeg_event_findings_v3_shadow_report_bundle_v1"
)
LEGACY_RECORD_CANDIDATE_BASELINE_METHOD_ID = (
    "legacy_event_rank1_count_baseline_v1"
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_SAFE_SCALP_ID_RE = re.compile(
    r"^[A-Z]{1,3}[0-9Z]{0,2}(?:-[A-Z]{1,3}[0-9Z]{0,2})?$"
)
_ROUTE_SCOPES = {"synthetic", "public"}
_SECTION_ORDER = (
    "technical_scope",
    "eeg_findings",
    "event_findings",
    "record_summary",
    "impression",
    "limitations",
)
_SECTION_LABELS = {
    "technical_scope": "技术与范围",
    "eeg_findings": "脑电所见",
    "event_findings": "事件所见",
    "record_summary": "多事件汇总",
    "impression": "脑电图印象",
    "limitations": "局限性",
}
_CONCEPT_SECTION = {
    "acquisition_capability": "technical_scope",
    "occurrence": "eeg_findings",
    "burden": "eeg_findings",
    "variability": "eeg_findings",
    "rhythmicity": "eeg_findings",
    "periodicity": "eeg_findings",
    "competing_hypothesis": "event_findings",
    "event_outcome": "event_findings",
}
_FIREWALL: Mapping[str, bool] = {
    "eeg_signal_claims_only": True,
    "edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_used": False,
    "ecg_emg_eog_used": False,
    "sleep_staging_used": False,
    "provocation_used": False,
}
_EXTERNAL_INPUT_KEY_FRAGMENTS = (
    "annotation",
    "excel",
    "doctor",
    "physician",
    "clinical_history",
    "clinical_text",
    "patient_metadata",
    "behavior",
    "performance",
)
_FORBIDDEN_REPORT_FRAGMENTS = (
    "Excel",
    "EDF annotation",
    "医生标注",
    "医师标注",
    "病史",
    "临床表现",
    "行为表现",
    "睡眠脑电",
    "诱发试验",
    "过度换气",
    "闪光刺激",
    "心电",
    "肌电",
    "眼电",
    "治疗建议",
    "手术建议",
)
_PHENOTYPE_LABELS = {
    "focal": "局灶性",
    "focal_with_rapid_bilateralization": "局灶起始伴快速双侧化",
    "bilateral_synchronous_or_rapid_bilateralization_ambiguous": (
        "双侧近同步或快速双侧化不易区分"
    ),
    "generalized_synchronous": "头皮广泛近同步起始",
    "scalp_onset_nonlocalizable": "头皮起始不可定位",
    "not_evaluable": "头皮起始无法评价",
}
_LATERALITY_LABELS = {
    "left": "左侧",
    "right": "右侧",
    "bilateral": "双侧",
    "midline": "中线",
    "indeterminate": "侧别不定",
}
_REGION_LABELS = {
    "temporal": "颞区",
    "frontal": "额区",
    "central": "中央区",
    "parietal": "顶区",
    "occipital": "枕区",
    "frontotemporal": "额颞区",
    "temporoparietal": "颞顶区",
    "hemispheric": "半球",
    "diffuse": "弥散区域",
    "left_temporal": "左侧颞区",
    "right_temporal": "右侧颞区",
    "left_frontal": "左侧额区",
    "right_frontal": "右侧额区",
}


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


def _seal(value: dict[str, Any], field: str, domain: str) -> None:
    value[field] = "CONTENT-ADDRESS-PENDING"
    value[field] = _sha256({"binding_domain": domain, "value": value})


def _reject_nonfinite(value: object, context: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} must be finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{context}[{index}]")


def _strict_object(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    missing = keys.difference(value)
    extra = set(value).difference(keys)
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{context} has unknown keys: {sorted(extra)}")
    return {str(key): deepcopy(item) for key, item in value.items()}


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{context} must be an opaque identifier")
    return value


def _sha(value: object, context: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _finite_nonnegative(
    value: object, context: str, *, positive: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{context} must be finite and {qualifier}")
    return result


def _unique_ids(value: object, context: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    result = [_identifier(item, f"{context}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise ValueError(f"{context} contains duplicate IDs")
    return result


def _validate_firewall(value: object) -> dict[str, bool]:
    data = _strict_object(value, set(_FIREWALL), "source_firewall")
    for key, expected in _FIREWALL.items():
        if data[key] is not expected:
            raise ValueError(f"source_firewall.{key} must be {expected}")
    return dict(_FIREWALL)


def _reject_external_event_keys(value: object, path: str = "event_findings_v3") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            # The validated v2/v3 provenance contract contains a closed
            # ``inference_exclusions`` receipt whose field names explicitly
            # state that external sources were not used (or, for a lossy v1
            # migration, remain unknown and therefore cannot support a
            # positive claim).  Those receipt keys are not external content.
            in_closed_exclusion_receipt = path.endswith(
                ".provenance.inference_exclusions"
            ) and normalized in {
                "edf_annotations_used",
                "excel_used",
                "doctor_labels_used",
                "clinical_text_used",
                "patient_metadata_used",
                "video_used",
                "ecg_emg_eog_used",
                "sleep_staging_used",
                "provocation_used",
            }
            if any(fragment in normalized for fragment in _EXTERNAL_INPUT_KEY_FRAGMENTS):
                if not in_closed_exclusion_receipt:
                    raise ValueError(
                        f"{path}.{key} is an external/private context field and is forbidden"
                    )
            _reject_external_event_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_external_event_keys(item, f"{path}[{index}]")


def _surface_guard(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("rendered report text must be non-empty")
    forbidden = [item for item in _FORBIDDEN_REPORT_FRAGMENTS if item in text]
    if forbidden:
        raise ValueError(f"report contains forbidden external content: {forbidden}")
    return text


def shadow_event_roster_sha256(
    *, record_id: str, signal_sha256: str, event_ids: Sequence[str]
) -> str:
    """Bind the ordered, complete detector event roster to one record."""

    return _sha256(
        {
            "binding_domain": "clinical-eeg-v3-shadow-complete-event-roster-v1",
            "record_id": str(record_id),
            "signal_sha256": str(signal_sha256),
            "event_ids": [str(item) for item in event_ids],
        }
    )


def _interval(value: object, context: str) -> dict[str, float]:
    data = _strict_object(
        value, {"start", "stop", "resolution_seconds"}, context
    )
    start = _finite_nonnegative(data["start"], f"{context}.start")
    stop = _finite_nonnegative(data["stop"], f"{context}.stop")
    resolution = _finite_nonnegative(
        data["resolution_seconds"],
        f"{context}.resolution_seconds",
        positive=True,
    )
    if stop <= start:
        raise ValueError(f"{context} must have positive duration")
    return {
        "start": start,
        "stop": stop,
        "resolution_seconds": resolution,
    }


def _canonical_union(rows: Sequence[Mapping[str, float]]) -> list[dict[str, float]]:
    ordered = sorted(
        (deepcopy(dict(row)) for row in rows),
        key=lambda row: (row["start"], row["stop"]),
    )
    result: list[dict[str, float]] = []
    for row in ordered:
        if not result or row["start"] > result[-1]["stop"] + 1e-6:
            result.append(row)
        else:
            result[-1]["stop"] = max(result[-1]["stop"], row["stop"])
            result[-1]["resolution_seconds"] = max(
                result[-1]["resolution_seconds"], row["resolution_seconds"]
            )
    return result


def _validate_shadow_input(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        {
            "schema_version",
            "route_scope",
            "record_id",
            "signal_sha256",
            "recording_duration_seconds",
            "source_firewall",
            "complete_event_roster",
            "event_sources",
            "record_aggregate_inputs",
        },
        "v3 shadow input",
    )
    if data["schema_version"] != EVENT_FINDINGS_V3_SHADOW_INPUT_SCHEMA_VERSION:
        raise ValueError("v3 shadow input schema_version mismatch")
    if data["route_scope"] not in _ROUTE_SCOPES:
        raise ValueError("v3 shadow route accepts only public or synthetic sources")
    data["record_id"] = _identifier(data["record_id"], "record_id")
    data["signal_sha256"] = _sha(data["signal_sha256"], "signal_sha256")
    data["recording_duration_seconds"] = _finite_nonnegative(
        data["recording_duration_seconds"],
        "recording_duration_seconds",
        positive=True,
    )
    data["source_firewall"] = _validate_firewall(data["source_firewall"])

    roster = _strict_object(
        data["complete_event_roster"],
        {"schema_version", "event_ids", "roster_sha256"},
        "complete_event_roster",
    )
    if roster["schema_version"] != EVENT_FINDINGS_V3_SHADOW_ROSTER_SCHEMA_VERSION:
        raise ValueError("complete_event_roster schema_version mismatch")
    roster["event_ids"] = _unique_ids(
        roster["event_ids"], "complete_event_roster.event_ids"
    )
    roster["roster_sha256"] = _sha(
        roster["roster_sha256"], "complete_event_roster.roster_sha256"
    )
    expected_roster_sha = shadow_event_roster_sha256(
        record_id=data["record_id"],
        signal_sha256=data["signal_sha256"],
        event_ids=roster["event_ids"],
    )
    if roster["roster_sha256"] != expected_roster_sha:
        raise ValueError("complete event roster content binding mismatch")
    data["complete_event_roster"] = roster

    if not isinstance(data["event_sources"], list):
        raise TypeError("event_sources must be a list")
    sources: list[dict[str, Any]] = []
    for index, row in enumerate(data["event_sources"]):
        source = _strict_object(
            row,
            {"event_id", "event_findings_v3"},
            f"event_sources[{index}]",
        )
        source["event_id"] = _identifier(
            source["event_id"], f"event_sources[{index}].event_id"
        )
        payload = source["event_findings_v3"]
        if payload is not None and not isinstance(payload, Mapping):
            raise TypeError(
                f"event_sources[{index}].event_findings_v3 must be an object or null"
            )
        if payload is not None:
            _reject_external_event_keys(payload)
            source["event_findings_v3"] = deepcopy(dict(payload))
        sources.append(source)
    if [row["event_id"] for row in sources] != roster["event_ids"]:
        raise ValueError(
            "event_sources must exactly follow the complete ordered event roster"
        )
    data["event_sources"] = sources

    aggregate = _strict_object(
        data["record_aggregate_inputs"],
        {
            "status",
            "event_to_mode_roster",
            "deduplicated_record_occurrences",
            "evaluable_record_seconds",
            "interval_union",
            "deduplication_policy_id",
            "interval_union_policy_id",
            "reason_codes",
        },
        "record_aggregate_inputs",
    )
    if aggregate["status"] not in {"complete", "not_available"}:
        raise ValueError("record_aggregate_inputs.status is invalid")
    if not isinstance(aggregate["event_to_mode_roster"], list):
        raise TypeError("event_to_mode_roster must be a list")
    mode_rows: list[dict[str, str]] = []
    for index, row in enumerate(aggregate["event_to_mode_roster"]):
        item = _strict_object(
            row, {"event_id", "mode_id"}, f"event_to_mode_roster[{index}]"
        )
        mode_rows.append(
            {
                "event_id": _identifier(
                    item["event_id"], f"event_to_mode_roster[{index}].event_id"
                ),
                "mode_id": _identifier(
                    item["mode_id"], f"event_to_mode_roster[{index}].mode_id"
                ),
            }
        )
    aggregate["event_to_mode_roster"] = mode_rows

    if not isinstance(aggregate["deduplicated_record_occurrences"], list):
        raise TypeError("deduplicated_record_occurrences must be a list")
    occurrence_rows: list[dict[str, Any]] = []
    for index, row in enumerate(aggregate["deduplicated_record_occurrences"]):
        item = _strict_object(
            row,
            {"record_occurrence_id", "mode_id", "interval", "source_refs"},
            f"deduplicated_record_occurrences[{index}]",
        )
        if not isinstance(item["source_refs"], list) or not item["source_refs"]:
            raise ValueError(
                f"deduplicated_record_occurrences[{index}].source_refs must be non-empty"
            )
        refs: list[dict[str, str]] = []
        for ref_index, ref in enumerate(item["source_refs"]):
            source_ref = _strict_object(
                ref,
                {"event_id", "occurrence_id"},
                (
                    f"deduplicated_record_occurrences[{index}]."
                    f"source_refs[{ref_index}]"
                ),
            )
            refs.append(
                {
                    "event_id": _identifier(
                        source_ref["event_id"], "source_ref.event_id"
                    ),
                    "occurrence_id": _identifier(
                        source_ref["occurrence_id"], "source_ref.occurrence_id"
                    ),
                }
            )
        occurrence_rows.append(
            {
                "record_occurrence_id": _identifier(
                    item["record_occurrence_id"], "record_occurrence_id"
                ),
                "mode_id": _identifier(item["mode_id"], "occurrence.mode_id"),
                "interval": _interval(
                    item["interval"],
                    f"deduplicated_record_occurrences[{index}].interval",
                ),
                "source_refs": refs,
            }
        )
    aggregate["deduplicated_record_occurrences"] = occurrence_rows
    if not isinstance(aggregate["interval_union"], list):
        raise TypeError("record_aggregate_inputs.interval_union must be a list")
    aggregate["interval_union"] = [
        _interval(row, f"record_aggregate_inputs.interval_union[{index}]")
        for index, row in enumerate(aggregate["interval_union"])
    ]
    if aggregate["evaluable_record_seconds"] is not None:
        aggregate["evaluable_record_seconds"] = _finite_nonnegative(
            aggregate["evaluable_record_seconds"],
            "record_aggregate_inputs.evaluable_record_seconds",
            positive=True,
        )
    for key in ("deduplication_policy_id", "interval_union_policy_id"):
        if aggregate[key] is not None:
            aggregate[key] = _identifier(aggregate[key], f"record_aggregate_inputs.{key}")
    aggregate["reason_codes"] = _unique_ids(
        aggregate["reason_codes"], "record_aggregate_inputs.reason_codes"
    )
    if aggregate["status"] == "not_available":
        if (
            mode_rows
            or occurrence_rows
            or aggregate["evaluable_record_seconds"] is not None
            or aggregate["interval_union"]
            or aggregate["deduplication_policy_id"] is not None
            or aggregate["interval_union_policy_id"] is not None
            or not aggregate["reason_codes"]
        ):
            raise ValueError(
                "not-available record aggregate inputs cannot imply partial facts"
            )
    else:
        if (
            aggregate["evaluable_record_seconds"] is None
            or aggregate["deduplication_policy_id"] is None
            or aggregate["interval_union_policy_id"] is None
            or aggregate["reason_codes"]
        ):
            raise ValueError("complete record aggregate inputs are incomplete")
    data["record_aggregate_inputs"] = aggregate
    return data


def _clause_choices(
    claim_id: str, text: str
) -> tuple[str, list[dict[str, Any]]]:
    """Return host-authored, fact-equivalent surface choices.

    The deterministic text is always first.  Event-prefixed clauses receive
    one mechanical Chinese word-order variant; quantities, entities, status,
    negation and epistemic strength remain byte-identical.  Qwen may select a
    choice ID but can never author text.
    """

    deterministic = _surface_guard(text)
    variants = [deterministic]
    alternate = re.sub(r"^第 ([0-9]+) 个事件：", r"事件 \1：", deterministic)
    if alternate == deterministic:
        alternate = re.sub(r"^第 ([0-9]+) 个事件的", r"事件 \1 的", deterministic)
    if alternate != deterministic:
        variants.append(_surface_guard(alternate))
    choices = [
        {
            "choice_id": (
                f"CHOICE-{_sha256({'claim_id': claim_id, 'text': variant})[:24]}"
            ),
            "text_zh": variant,
        }
        for variant in variants
    ]
    return str(choices[0]["choice_id"]), choices


def _number(value: object) -> str:
    number = float(value)
    if abs(number) < 0.5e-9:
        number = 0.0
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _candidate_text(candidate_type: str, candidate_id: str) -> str:
    if candidate_type == "laterality":
        return _LATERALITY_LABELS.get(candidate_id, "一个侧别候选")
    if candidate_type == "region":
        return _REGION_LABELS.get(candidate_id, "一个区域候选")
    if candidate_type in {"lead", "electrode"}:
        if _SAFE_SCALP_ID_RE.fullmatch(candidate_id):
            suffix = "导联" if candidate_type == "lead" else "通道"
            return f"{candidate_id} {suffix}"
        return "一个头皮导联候选" if candidate_type == "lead" else "一个头皮通道候选"
    return _PHENOTYPE_LABELS.get(candidate_id, "一种头皮起始表型候选")


def _spatial_candidate_rows(
    source: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    *,
    event_ordinal: int,
) -> list[dict[str, Any]]:
    positive_differential = any(
        claim["concept"] == "competing_hypothesis"
        and claim["positive_onset_support_permitted"]
        for claim in sidecar["concept_claims"]
    )
    if not positive_differential:
        return []
    hypothesis = source["scalp_onset_hypothesis"]
    if hypothesis["localization_status"] not in {
        "ranked_candidates",
        "phenotype_only",
    }:
        return []
    relation_map = {
        str(row["relation_id"]): row
        for row in source["hypothesis_evidence_relations"]
    }
    result: list[dict[str, Any]] = []
    candidates = hypothesis["candidate_scores"]
    if not candidates and hypothesis["localization_status"] == "phenotype_only":
        phenotype = str(hypothesis["phenotype"])
        relations = [
            row
            for row in relation_map.values()
            if row["axis"] == "phenotype"
            and row["candidate_id"] == phenotype
            and row["relation"] == "supports"
        ]
        evidence_ids = sorted(
            {
                str(evidence_id)
                for row in relations
                for evidence_id in row["evidence_ids"]
            }
        )
        permissions = _evidence_time_permissions(source, evidence_ids)
        if evidence_ids and all(
            row["positive_onset_support_permitted"] for row in permissions
        ):
            result.append(
                {
                    "candidate_type": "phenotype",
                    "candidate_id": phenotype,
                    "rank": 1,
                    "score": None,
                    "score_semantics": "not_applicable",
                    "relation_ids": sorted(str(row["relation_id"]) for row in relations),
                    "evidence_ids": evidence_ids,
                    "evidence_time_permissions": permissions,
                    "text_zh": (
                        f"第 {event_ordinal} 个事件形成"
                        f"{_candidate_text('phenotype', phenotype)}。"
                    ),
                }
            )
        return result

    for candidate in candidates:
        relation_ids = [str(item) for item in candidate["supporting_relation_ids"]]
        relations = [relation_map[item] for item in relation_ids]
        evidence_ids = sorted(
            {
                str(evidence_id)
                for row in relations
                for evidence_id in row["evidence_ids"]
            }
        )
        permissions = _evidence_time_permissions(source, evidence_ids)
        if not evidence_ids or not all(
            row["positive_onset_support_permitted"] for row in permissions
        ):
            continue
        candidate_type = str(candidate["candidate_type"])
        candidate_id = str(candidate["candidate_id"])
        rank = int(candidate["rank"])
        score = float(candidate["score"])
        if candidate["score_semantics"] == "patient_disjoint_calibrated_probability":
            score_text = f"校准概率 {_number(score * 100.0)}%"
        else:
            score_text = f"研究排序分值 {_number(score)}"
        result.append(
            {
                "candidate_type": candidate_type,
                "candidate_id": candidate_id,
                "rank": rank,
                "score": score,
                "score_semantics": str(candidate["score_semantics"]),
                "relation_ids": relation_ids,
                "evidence_ids": evidence_ids,
                "evidence_time_permissions": permissions,
                "text_zh": (
                    f"第 {event_ordinal} 个事件的头皮起始候选第 {rank} 位为"
                    f"{_candidate_text(candidate_type, candidate_id)}"
                    f"（{score_text}）。"
                ),
            }
        )
    return result


def _source_occurrences(
    sidecars: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, float]]:
    result: dict[tuple[str, str], dict[str, float]] = {}
    for sidecar in sidecars:
        event_id = str(sidecar["event_id"])
        for claim in sidecar["concept_claims"]:
            if claim["concept"] != "occurrence":
                continue
            occurrence = claim["value"].get("occurrence")
            if not isinstance(occurrence, Mapping) or occurrence["status"] != "measured":
                continue
            for row in occurrence["deduplicated_occurrences"]:
                key = (event_id, str(row["occurrence_id"]))
                if key in result:
                    raise ValueError("source sidecars repeat an event occurrence ID")
                result[key] = {
                    "start": float(row["interval"]["start"]),
                    "stop": float(row["interval"]["stop"]),
                    "resolution_seconds": float(
                        row["interval"]["resolution_seconds"]
                    ),
                }
    return result


def _validate_complete_aggregate_gate(
    aggregate: Mapping[str, Any],
    *,
    event_ids: Sequence[str],
    sidecars: Sequence[Mapping[str, Any]],
    recording_duration_seconds: float,
) -> dict[str, Any]:
    if aggregate["status"] == "not_available":
        return {
            "status": "not_evaluable",
            "complete_event_roster": True,
            "complete_event_to_mode_roster": False,
            "deduplication_ownership_complete": False,
            "evaluable_record_time_denominator_available": False,
            "canonical_interval_union_available": False,
            "event_count": len(event_ids),
            "mode_count": None,
            "record_occurrence_count": None,
            "observed_seconds": None,
            "evaluable_seconds": None,
            "proportion": None,
            "reason_codes": deepcopy(aggregate["reason_codes"]),
        }
    if len(sidecars) != len(event_ids):
        raise ValueError(
            "complete record aggregate inputs require one valid sidecar per roster event"
        )
    mode_rows = aggregate["event_to_mode_roster"]
    if [row["event_id"] for row in mode_rows] != list(event_ids):
        raise ValueError("event-to-mode roster must exactly follow the event roster")
    mode_by_event = {str(row["event_id"]): str(row["mode_id"]) for row in mode_rows}
    if len(mode_by_event) != len(event_ids):
        raise ValueError("event-to-mode roster repeats an event")

    source_occurrences = _source_occurrences(sidecars)
    occurrence_claims = [
        claim
        for sidecar in sidecars
        for claim in sidecar["concept_claims"]
        if claim["concept"] == "occurrence"
    ]
    burden_claims = [
        claim
        for sidecar in sidecars
        for claim in sidecar["concept_claims"]
        if claim["concept"] == "burden"
    ]
    if not occurrence_claims or not burden_claims or any(
        claim["epistemic_status"] != "measured"
        for claim in occurrence_claims + burden_claims
    ):
        raise ValueError(
            "record aggregate gate requires measured event occurrence and burden inputs"
        )

    seen_source_refs: list[tuple[str, str]] = []
    record_ids: set[str] = set()
    previous_key: tuple[float, float, str] | None = None
    previous_stop: float | None = None
    record_intervals: list[dict[str, float]] = []
    for row in aggregate["deduplicated_record_occurrences"]:
        record_id = str(row["record_occurrence_id"])
        if record_id in record_ids:
            raise ValueError("record occurrence roster contains duplicate IDs")
        record_ids.add(record_id)
        interval = row["interval"]
        key = (float(interval["start"]), float(interval["stop"]), record_id)
        if previous_key is not None and key <= previous_key:
            raise ValueError("record occurrence roster is not canonical")
        if previous_stop is not None and key[0] < previous_stop - 1e-6:
            raise ValueError("record occurrence roster contains overlapping duplicates")
        previous_key = key
        previous_stop = key[1]
        refs = [
            (str(ref["event_id"]), str(ref["occurrence_id"]))
            for ref in row["source_refs"]
        ]
        if len(refs) != len(set(refs)):
            raise ValueError("one record occurrence repeats a source occurrence")
        missing = sorted(set(refs).difference(source_occurrences))
        if missing:
            raise ValueError(f"record occurrence references unknown sources: {missing}")
        if any(mode_by_event[event_id] != row["mode_id"] for event_id, _ in refs):
            raise ValueError("record occurrence ownership conflicts with event mode")
        source_union = _canonical_union([source_occurrences[ref] for ref in refs])
        if len(source_union) != 1 or any(
            abs(float(source_union[0][field]) - float(interval[field])) > 1e-6
            for field in ("start", "stop", "resolution_seconds")
        ):
            raise ValueError("record occurrence interval does not bind its source union")
        seen_source_refs.extend(refs)
        record_intervals.append(deepcopy(interval))
    if len(seen_source_refs) != len(set(seen_source_refs)):
        raise ValueError("deduplication ownership assigns one source occurrence twice")
    if set(seen_source_refs) != set(source_occurrences):
        raise ValueError("deduplication ownership does not cover every source occurrence")

    expected_union = _canonical_union(record_intervals)
    if expected_union != aggregate["interval_union"]:
        raise ValueError("record interval union is not canonical")
    if any(
        row["stop"] > recording_duration_seconds + 1e-6
        for row in expected_union
    ):
        raise ValueError("record interval union lies outside the recording")
    evaluable = float(aggregate["evaluable_record_seconds"])
    if evaluable > recording_duration_seconds + 1e-6:
        raise ValueError("record evaluable denominator exceeds recording duration")
    observed = sum(row["stop"] - row["start"] for row in expected_union)
    if observed > evaluable + 1e-6:
        raise ValueError("record observed burden exceeds its evaluable denominator")
    return {
        "status": "passed",
        "complete_event_roster": True,
        "complete_event_to_mode_roster": True,
        "deduplication_ownership_complete": True,
        "evaluable_record_time_denominator_available": True,
        "canonical_interval_union_available": True,
        "event_count": len(event_ids),
        "mode_count": len(set(mode_by_event.values())),
        "record_occurrence_count": len(record_ids),
        "observed_seconds": observed,
        "evaluable_seconds": evaluable,
        "proportion": observed / evaluable,
        "reason_codes": [],
    }


def build_event_findings_v3_record_claim_plan(
    shadow_input: object,
    *,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Build a source-bound v3 record plan; invalid events become fallbacks."""

    source = _validate_shadow_input(shadow_input)
    _reject_nonfinite(source)
    event_ids = source["complete_event_roster"]["event_ids"]
    claims: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    event_ledger: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    used_claim_ids: set[str] = set()
    used_owner_ids: set[str] = set()
    used_slot_ids: set[str] = set()

    def emit(
        *,
        concept: str,
        section_id: str,
        discriminator: object,
        text_zh: str,
        event_id: str | None,
        source_claim_ids: Sequence[str],
        source_evidence_ids: Sequence[str],
        positive_onset_support_permitted: bool,
        value: object,
    ) -> None:
        seed = {
            "record_id": source["record_id"],
            "concept": concept,
            "event_id": event_id,
            "discriminator": discriminator,
        }
        claim_id = f"V3RCLAIM-{_sha256(seed)[:24]}"
        owner_id = f"V3ROWNER-{_sha256({'claim_id': claim_id})[:24]}"
        slot_id = f"V3SLOT-{_sha256({'claim_id': claim_id})[:24]}"
        if claim_id in used_claim_ids or owner_id in used_owner_ids or slot_id in used_slot_ids:
            raise ValueError("record claim/clause ownership collision")
        used_claim_ids.add(claim_id)
        used_owner_ids.add(owner_id)
        used_slot_ids.add(slot_id)
        choice_id, choices = _clause_choices(claim_id, text_zh)
        claims.append(
            {
                "claim_id": claim_id,
                "claim_owner_id": owner_id,
                "atomic_clause_slot_id": slot_id,
                "concept": concept,
                "section_id": section_id,
                "event_id": event_id,
                "source_claim_ids": sorted(set(str(item) for item in source_claim_ids)),
                "source_evidence_ids": sorted(
                    set(str(item) for item in source_evidence_ids)
                ),
                "positive_onset_support_permitted": bool(
                    positive_onset_support_permitted
                ),
                "value": deepcopy(value),
                "allowed_clause_choices": choices,
                "deterministic_choice_id": choice_id,
            }
        )

    for ordinal, wrapper in enumerate(source["event_sources"], start=1):
        event_id = str(wrapper["event_id"])
        payload = wrapper["event_findings_v3"]
        sidecar: dict[str, Any] | None = None
        failure_reason: str | None = None
        if payload is None:
            failure_reason = "missing_event_findings_v3"
        else:
            try:
                if str(payload.get("event_id", "")) != event_id:
                    raise ValueError("event wrapper/source event ID mismatch")
                if (
                    payload.get("provenance", {}).get("record_id")
                    != source["record_id"]
                ):
                    raise ValueError("event source record binding mismatch")
                if (
                    payload.get("provenance", {}).get("canonical_signal_sha256")
                    != source["signal_sha256"]
                ):
                    raise ValueError("event source canonical signal mismatch")
                if abs(
                    float(payload.get("coordinates", {}).get(
                        "recording_duration_seconds", -1.0
                    ))
                    - float(source["recording_duration_seconds"])
                ) > 1e-6:
                    raise ValueError("event source recording duration mismatch")
                sidecar = project_event_eeg_findings_v3_downstream(
                    payload,
                    trusted_producer_receipts=trusted_producer_receipts,
                    trusted_calibration_receipts=trusted_calibration_receipts,
                    trusted_capability_qualification_receipts=(
                        trusted_capability_qualification_receipts
                    ),
                    trusted_sensitivity_receipts=trusted_sensitivity_receipts,
                    trusted_term_decision_receipts=trusted_term_decision_receipts,
                    trusted_registry_bindings=trusted_registry_bindings,
                )
            except (TypeError, ValueError):
                failure_reason = "invalid_or_untrusted_event_findings_v3"
        if sidecar is None:
            emit(
                concept="event_projection_outcome",
                section_id="event_findings",
                discriminator={"event_id": event_id, "failure": failure_reason},
                text_zh=(
                    f"第 {ordinal} 个候选事件未形成可验证的 v3 信号证据；"
                    "该事件无法评价。"
                ),
                event_id=event_id,
                source_claim_ids=[],
                source_evidence_ids=[],
                positive_onset_support_permitted=False,
                value={"status": "not_evaluable", "reason_code": failure_reason},
            )
            event_ledger.append(
                {
                    "event_id": event_id,
                    "ordinal": ordinal,
                    "projection_status": "not_evaluable",
                    "reason_code": failure_reason,
                    "source_event_findings_v3_sha256": (
                        None if payload is None else _sha256(payload)
                    ),
                    "sidecar_sha256": None,
                }
            )
            continue

        sidecars.append(sidecar)
        clause_by_claim = {
            str(row["claim_id"]): str(row["text_zh"])
            for row in sidecar["atomic_clauses"]
        }
        for claim in sidecar["concept_claims"]:
            concept = str(claim["concept"])
            emit(
                concept=concept,
                section_id=_CONCEPT_SECTION[concept],
                discriminator={"sidecar_claim_id": claim["claim_id"]},
                text_zh=f"第 {ordinal} 个事件：{clause_by_claim[claim['claim_id']]}",
                event_id=event_id,
                source_claim_ids=[str(claim["claim_id"])],
                source_evidence_ids=claim["source_evidence_ids"],
                positive_onset_support_permitted=bool(
                    claim["positive_onset_support_permitted"]
                ),
                value={
                    "source_assertion_status": claim["assertion_status"],
                    "source_epistemic_status": claim["epistemic_status"],
                    "source_value": deepcopy(claim["value"]),
                },
            )
        event_spatial = _spatial_candidate_rows(
            payload, sidecar, event_ordinal=ordinal
        )
        for row in event_spatial:
            emit(
                concept="scalp_onset_candidate",
                section_id="event_findings",
                discriminator={
                    "event_id": event_id,
                    "candidate_type": row["candidate_type"],
                    "candidate_id": row["candidate_id"],
                    "rank": row["rank"],
                },
                text_zh=row["text_zh"],
                event_id=event_id,
                source_claim_ids=row["relation_ids"],
                source_evidence_ids=row["evidence_ids"],
                positive_onset_support_permitted=True,
                value={key: deepcopy(value) for key, value in row.items() if key != "text_zh"},
            )
            spatial_rows.append({"event_id": event_id, **deepcopy(row)})
        event_ledger.append(
            {
                "event_id": event_id,
                "ordinal": ordinal,
                "projection_status": "projected",
                "reason_code": None,
                "source_event_findings_v3_sha256": sidecar["source_binding"][
                    "source_event_findings_v3_sha256"
                ],
                "sidecar_sha256": sidecar["projection_sha256"],
            }
        )

    aggregate_gate = _validate_complete_aggregate_gate(
        source["record_aggregate_inputs"],
        event_ids=event_ids,
        sidecars=sidecars,
        recording_duration_seconds=float(source["recording_duration_seconds"]),
    )
    emit(
        concept="complete_event_roster_summary",
        section_id="record_summary",
        discriminator="complete-event-roster",
        text_zh=(
            f"完整事件清单含 {len(event_ids)} 个候选事件，其中 "
            f"{len(sidecars)} 个形成可验证的 v3 事件证据。"
        ),
        event_id=None,
        source_claim_ids=[],
        source_evidence_ids=[],
        positive_onset_support_permitted=False,
        value={
            "event_count": len(event_ids),
            "projected_event_count": len(sidecars),
            "roster_sha256": source["complete_event_roster"]["roster_sha256"],
        },
    )
    if aggregate_gate["status"] == "passed":
        emit(
            concept="record_aggregate",
            section_id="record_summary",
            discriminator="record-aggregate-passed",
            text_zh=(
                f"记录级聚合输入门通过：{aggregate_gate['event_count']} 个事件归入 "
                f"{aggregate_gate['mode_count']} 个模式；"
                f"{aggregate_gate['record_occurrence_count']} 个去重模式候选在 "
                f"{_number(aggregate_gate['evaluable_seconds'])} 秒可评价记录中累计占用 "
                f"{_number(aggregate_gate['observed_seconds'])} 秒"
                f"（{_number(float(aggregate_gate['proportion']) * 100.0)}%）。"
            ),
            event_id=None,
            source_claim_ids=[],
            source_evidence_ids=[],
            positive_onset_support_permitted=False,
            value=aggregate_gate,
        )
    else:
        emit(
            concept="record_aggregate_gate",
            section_id="record_summary",
            discriminator="record-aggregate-not-evaluable",
            text_zh=(
                "整段记录级模式、负担与变异性未通过完整聚合输入门，"
                "不由单事件窗结果直接推导。"
            ),
            event_id=None,
            source_claim_ids=[],
            source_evidence_ids=[],
            positive_onset_support_permitted=False,
            value=aggregate_gate,
        )

    qualified_outcome_count = sum(
        1
        for sidecar in sidecars
        for claim in sidecar["concept_claims"]
        if claim["concept"] == "event_outcome"
        and claim["value"]["outcome"]
        in {
            "qualified_electrographic_seizure",
            "qualified_electrographic_event",
        }
    )
    # Retain the old rank-one count only as a sealed ablation diagnostic.  The
    # complete aggregate gate above closes occurrence ownership, mode IDs and
    # the record-time denominator; it is not a trusted SOZ-model receipt and
    # therefore cannot authorize a record-level spatial conclusion.
    top_rows = [row for row in spatial_rows if row["rank"] == 1]
    rank_one_counts: dict[tuple[str, str], int] = {}
    for row in top_rows:
        key = (str(row["candidate_type"]), str(row["candidate_id"]))
        rank_one_counts[key] = rank_one_counts.get(key, 0) + 1
    if rank_one_counts:
        best_count = max(rank_one_counts.values())
        best = sorted(
            key for key, count in rank_one_counts.items() if count == best_count
        )
        if len(best) == 1:
            candidate_type, candidate_id = best[0]
            legacy_result: dict[str, Any] = {
                "status": "record_candidate_summary",
                "candidate_type": candidate_type,
                "candidate_id": candidate_id,
                "supporting_event_count": best_count,
                "projected_event_count": len(sidecars),
            }
        else:
            legacy_result = {
                "status": "tied_record_candidates",
                "candidate_keys": [list(item) for item in best],
                "projected_event_count": len(sidecars),
            }
    else:
        legacy_result = {
            "status": "no_rank_one_candidate",
            "projected_event_count": len(sidecars),
        }
    legacy_record_candidate_baseline = {
        "method_id": LEGACY_RECORD_CANDIDATE_BASELINE_METHOD_ID,
        "method_role": "shadow_ablation_only",
        "aggregation_semantics": "event_rank_one_count_without_mode_aware_mil",
        "trusted_hierarchical_mil_receipt_id": None,
        "formal_hierarchical_mil_authorized": False,
        "used_for_record_impression": False,
        "result": legacy_result,
    }

    if spatial_rows:
        impression_text = (
            "事件级头皮起始候选已列于事件所见；"
            "当前 shadow 输入未提供可信的模式感知层级聚合回执，"
            "因此不形成整段记录级空间候选。"
        )
        impression_value: object = {
            "status": "event_candidates_not_record_aggregated",
            "reason_codes": ["trusted_mode_aware_hierarchical_mil_receipt_not_available"],
            "legacy_baseline_method_id": LEGACY_RECORD_CANDIDATE_BASELINE_METHOD_ID,
            "legacy_baseline_used_for_record_impression": False,
        }
    elif qualified_outcome_count:
        impression_text = (
            f"记录到 {qualified_outcome_count} 个通过自动资格门的电图事件，"
            "但未形成具备时间权限的头皮起始候选。"
        )
        impression_value = {
            "status": "qualified_events_without_permitted_spatial_candidate",
            "qualified_event_count": qualified_outcome_count,
        }
    else:
        impression_text = "本记录未形成具备时间权限的头皮起始候选。"
        impression_value = {"status": "no_permitted_scalp_onset_candidate"}
    emit(
        concept="record_impression",
        section_id="impression",
        discriminator=impression_value,
        text_zh=impression_text,
        event_id=None,
        source_claim_ids=[
            claim["claim_id"]
            for claim in claims
            if claim["concept"] in {"event_outcome", "scalp_onset_candidate"}
        ],
        source_evidence_ids=[
            evidence_id
            for claim in claims
            if claim["concept"] == "scalp_onset_candidate"
            for evidence_id in claim["source_evidence_ids"]
        ],
        positive_onset_support_permitted=False,
        value=impression_value,
    )
    emit(
        concept="eeg_only_limitation",
        section_id="limitations",
        discriminator="eeg-only-firewall",
        text_zh=(
            "本报告仅依据本 shadow 路线中通过验证的头皮 EEG 信号证据，"
            "未使用外部资料。"
        ),
        event_id=None,
        source_claim_ids=[],
        source_evidence_ids=[],
        positive_onset_support_permitted=False,
        value={"source_firewall": dict(_FIREWALL)},
    )

    slots = [
        {
            "slot_id": claim["atomic_clause_slot_id"],
            "claim_id": claim["claim_id"],
            "concept": claim["concept"],
            "section_id": claim["section_id"],
            "allowed_choices": deepcopy(claim["allowed_clause_choices"]),
            "deterministic_choice_id": claim["deterministic_choice_id"],
        }
        for claim in claims
    ]
    plan: dict[str, Any] = {
        "schema_version": EVENT_FINDINGS_V3_RECORD_CLAIM_PLAN_SCHEMA_VERSION,
        "route_id": EVENT_FINDINGS_V3_SHADOW_ROUTE_ID,
        "route_scope": source["route_scope"],
        "record_id": source["record_id"],
        "signal_sha256": source["signal_sha256"],
        "source_shadow_input_sha256": _sha256(source),
        "complete_event_roster_sha256": source["complete_event_roster"][
            "roster_sha256"
        ],
        "event_ledger": event_ledger,
        "sidecar_projection_sha256s": [
            sidecar["projection_sha256"] for sidecar in sidecars
        ],
        "record_aggregate_gate": aggregate_gate,
        "legacy_record_candidate_baseline": legacy_record_candidate_baseline,
        "claims": claims,
        "lexicalization_slots": slots,
        "ownership_receipt": {
            "policy_id": "one_record_claim_one_clause_owner_v1",
            "claim_count": len(claims),
            "slot_count": len(slots),
            "all_claim_owners_unique": True,
            "all_slots_unique": True,
        },
        "source_firewall": dict(_FIREWALL),
        "production_route_connected": ADAPTIVE_REPORT_ROUTE_CONNECTED,
        "plan_sha256": "",
    }
    _seal(
        plan,
        "plan_sha256",
        "clinical-eeg-event-findings-v3-record-claim-plan-v1",
    )
    return plan


def _closed_selection_request(plan: Mapping[str, Any]) -> dict[str, Any]:
    request = {
        "schema_version": EVENT_FINDINGS_V3_CLOSED_LEXICALIZATION_SCHEMA_VERSION,
        "role": "closed_choice_clause_selection_only",
        "slots": [
            {
                "slot_id": slot["slot_id"],
                "concept": slot["concept"],
                "section_id": slot["section_id"],
                "allowed_choices": deepcopy(slot["allowed_choices"]),
            }
            for slot in plan["lexicalization_slots"]
        ],
        "prohibitions": {
            "free_text": True,
            "new_claims": True,
            "new_entities": True,
            "new_times": True,
            "source_identifiers": True,
            "external_context": True,
        },
    }
    return request


def _validate_closed_selection(
    value: object, slots: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    data = _strict_object(value, {"schema_version", "selections"}, "Qwen selection")
    if data["schema_version"] != EVENT_FINDINGS_V3_CLOSED_LEXICALIZATION_SCHEMA_VERSION:
        raise ValueError("Qwen selection schema_version mismatch")
    if not isinstance(data["selections"], list):
        raise TypeError("Qwen selections must be a list")
    expected_slot_ids = [str(slot["slot_id"]) for slot in slots]
    selections: list[dict[str, str]] = []
    for index, row in enumerate(data["selections"]):
        item = _strict_object(row, {"slot_id", "choice_id"}, f"selections[{index}]")
        slot_id = _identifier(item["slot_id"], f"selections[{index}].slot_id")
        choice_id = _identifier(item["choice_id"], f"selections[{index}].choice_id")
        selections.append({"slot_id": slot_id, "choice_id": choice_id})
    if [row["slot_id"] for row in selections] != expected_slot_ids:
        raise ValueError("Qwen selections must exactly cover slots in frozen order")
    for row, slot in zip(selections, slots):
        allowed = {str(choice["choice_id"]) for choice in slot["allowed_choices"]}
        if row["choice_id"] not in allowed:
            raise ValueError("Qwen selected a clause outside the closed choice set")
    return {
        "schema_version": EVENT_FINDINGS_V3_CLOSED_LEXICALIZATION_SCHEMA_VERSION,
        "selections": selections,
    }


def _deterministic_selection(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": EVENT_FINDINGS_V3_CLOSED_LEXICALIZATION_SCHEMA_VERSION,
        "selections": [
            {
                "slot_id": slot["slot_id"],
                "choice_id": slot["deterministic_choice_id"],
            }
            for slot in plan["lexicalization_slots"]
        ],
    }


def _render_from_selection(
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    language_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    choice_by_slot: dict[str, dict[str, str]] = {}
    for slot in plan["lexicalization_slots"]:
        choice_by_slot[str(slot["slot_id"])] = {
            str(row["choice_id"]): str(row["text_zh"])
            for row in slot["allowed_choices"]
        }
    selected_by_slot = {
        str(row["slot_id"]): str(row["choice_id"])
        for row in selection["selections"]
    }
    claim_by_id = {str(row["claim_id"]): row for row in plan["claims"]}
    sections: list[dict[str, Any]] = []
    report_parts = ["长程头皮 EEG 信号报告（shadow）"]
    for section_id in _SECTION_ORDER:
        section_claims = [
            claim for claim in plan["claims"] if claim["section_id"] == section_id
        ]
        lines: list[str] = []
        claim_ids: list[str] = []
        for claim in section_claims:
            slot_id = str(claim["atomic_clause_slot_id"])
            choice_id = selected_by_slot[slot_id]
            lines.append(choice_by_slot[slot_id][choice_id])
            claim_ids.append(str(claim["claim_id"]))
        if not lines:
            lines = ["本节无可输出的信号事实。"]
        sections.append(
            {
                "section_id": section_id,
                "label_zh": _SECTION_LABELS[section_id],
                "claim_ids": claim_ids,
                "text_lines_zh": lines,
            }
        )
        report_parts.append(f"【{_SECTION_LABELS[section_id]}】")
        report_parts.extend(lines)
    if set(claim_by_id) != {
        claim_id for section in sections for claim_id in section["claim_ids"]
    }:
        raise ValueError("render did not retain every planned claim exactly once")
    report_text = _surface_guard("\n".join(report_parts))
    result: dict[str, Any] = {
        "schema_version": EVENT_FINDINGS_V3_REPORT_RENDER_SCHEMA_VERSION,
        "route_id": EVENT_FINDINGS_V3_SHADOW_ROUTE_ID,
        "record_id": plan["record_id"],
        "plan_sha256": plan["plan_sha256"],
        "sections": sections,
        "report_text_zh": report_text,
        "language_selection_receipt": deepcopy(dict(language_receipt)),
        "render_sha256": "",
    }
    _seal(
        result,
        "render_sha256",
        "clinical-eeg-event-findings-v3-report-render-v1",
    )
    return result


def render_event_findings_v3_report_zh(
    record_claim_plan: object,
    *,
    qwen_clause_selector: Callable[[dict[str, Any]], object] | None = None,
) -> dict[str, Any]:
    """Render every valid plan; Qwen is closed-choice and failure-safe."""

    if type(record_claim_plan) is not dict:
        raise TypeError("record claim plan must be an object")
    plan = deepcopy(record_claim_plan)
    deterministic = _deterministic_selection(plan)
    request = _closed_selection_request(plan)
    if qwen_clause_selector is None:
        selection = deterministic
        receipt = {
            "qwen_requested": False,
            "qwen_role": "not_used",
            "selection_status": "deterministic_fallback",
            "fallback_reason": "qwen_not_requested",
            "request_sha256": _sha256(request),
            "normalized_selection_sha256": _sha256(selection),
            "normalized_selections": deepcopy(selection["selections"]),
        }
    else:
        try:
            candidate = qwen_clause_selector(deepcopy(request))
            selection = _validate_closed_selection(
                candidate, plan["lexicalization_slots"]
            )
            receipt = {
                "qwen_requested": True,
                "qwen_role": "closed_choice_clause_selection_only",
                "selection_status": "validated_closed_choice",
                "fallback_reason": None,
                "request_sha256": _sha256(request),
                "normalized_selection_sha256": _sha256(selection),
                "normalized_selections": deepcopy(selection["selections"]),
            }
        except Exception:  # deterministic fact-preserving fallback boundary
            selection = deterministic
            receipt = {
                "qwen_requested": True,
                "qwen_role": "closed_choice_clause_selection_only",
                "selection_status": "deterministic_fallback",
                "fallback_reason": "qwen_exception_or_invalid_closed_choice",
                "request_sha256": _sha256(request),
                "normalized_selection_sha256": _sha256(selection),
                "normalized_selections": deepcopy(selection["selections"]),
            }
    return _render_from_selection(
        plan, selection, language_receipt=receipt
    )


def _validate_render_against_plan(
    value: object, plan: Mapping[str, Any]
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("v3 report render must be an object")
    candidate = deepcopy(value)
    receipt = _strict_object(
        candidate.get("language_selection_receipt"),
        {
            "qwen_requested",
            "qwen_role",
            "selection_status",
            "fallback_reason",
            "request_sha256",
            "normalized_selection_sha256",
            "normalized_selections",
        },
        "language_selection_receipt",
    )
    if not isinstance(receipt["qwen_requested"], bool):
        raise TypeError("language receipt qwen_requested must be boolean")
    if receipt["selection_status"] not in {
        "validated_closed_choice",
        "deterministic_fallback",
    }:
        raise ValueError("language receipt selection_status is invalid")
    expected_request_sha = _sha256(_closed_selection_request(plan))
    if receipt["request_sha256"] != expected_request_sha:
        raise ValueError("language receipt request hash mismatch")
    normalized = _validate_closed_selection(
        {
            "schema_version": EVENT_FINDINGS_V3_CLOSED_LEXICALIZATION_SCHEMA_VERSION,
            "selections": receipt["normalized_selections"],
        },
        plan["lexicalization_slots"],
    )
    if receipt["normalized_selection_sha256"] != _sha256(normalized):
        raise ValueError("language receipt selection hash mismatch")
    deterministic = _deterministic_selection(plan)
    if receipt["selection_status"] == "deterministic_fallback":
        if normalized != deterministic or receipt["fallback_reason"] not in {
            "qwen_not_requested",
            "qwen_exception_or_invalid_closed_choice",
        }:
            raise ValueError("deterministic language fallback receipt is inconsistent")
        if receipt["fallback_reason"] == "qwen_not_requested" and (
            receipt["qwen_requested"]
            or receipt["qwen_role"] != "not_used"
        ):
            raise ValueError("not-requested Qwen receipt is inconsistent")
        if receipt["fallback_reason"] == "qwen_exception_or_invalid_closed_choice" and (
            not receipt["qwen_requested"]
            or receipt["qwen_role"] != "closed_choice_clause_selection_only"
        ):
            raise ValueError("failed Qwen closed-choice receipt is inconsistent")
    else:
        if (
            not receipt["qwen_requested"]
            or receipt["qwen_role"] != "closed_choice_clause_selection_only"
            or receipt["fallback_reason"] is not None
        ):
            raise ValueError("validated Qwen closed-choice receipt is inconsistent")
    expected = _render_from_selection(
        plan, normalized, language_receipt=receipt
    )
    if candidate != expected:
        raise ValueError("v3 report render does not replay from its claim plan")
    return expected


def materialize_event_findings_v3_shadow_report(
    shadow_input: object,
    *,
    qwen_clause_selector: Callable[[dict[str, Any]], object] | None = None,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Run the public/synthetic v3 shadow route end to end."""

    source = _validate_shadow_input(shadow_input)
    plan = build_event_findings_v3_record_claim_plan(
        source,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    render = render_event_findings_v3_report_zh(
        plan, qwen_clause_selector=qwen_clause_selector
    )
    bundle: dict[str, Any] = {
        "schema_version": EVENT_FINDINGS_V3_SHADOW_BUNDLE_SCHEMA_VERSION,
        "route_id": EVENT_FINDINGS_V3_SHADOW_ROUTE_ID,
        "route_scope": source["route_scope"],
        "production_route_connected": ADAPTIVE_REPORT_ROUTE_CONNECTED,
        "source_shadow_input_sha256": _sha256(source),
        "record_claim_plan": plan,
        "report_render": render,
        "route_receipt": {
            "public_or_synthetic_only": True,
            "private_production_inputs_authorized": False,
            "complete_event_roster_bound": True,
            "invalid_event_has_deterministic_output": True,
            "qwen_free_text_authorized": False,
            "qwen_failure_has_deterministic_fallback": True,
            "source_replay_required": True,
        },
        "bundle_sha256": "",
    }
    _seal(
        bundle,
        "bundle_sha256",
        "clinical-eeg-event-findings-v3-shadow-report-bundle-v1",
    )
    return bundle


def validate_event_findings_v3_shadow_report(
    value: object,
    *,
    shadow_input: object,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Rebuild the plan from source and replay the deterministic renderer."""

    if type(value) is not dict:
        raise TypeError("v3 shadow report bundle must be an object")
    candidate = deepcopy(value)
    _reject_nonfinite(candidate)
    source = _validate_shadow_input(shadow_input)
    plan = build_event_findings_v3_record_claim_plan(
        source,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    if candidate.get("record_claim_plan") != plan:
        raise ValueError("shadow record claim plan does not replay from source")
    render = _validate_render_against_plan(candidate.get("report_render"), plan)
    expected = deepcopy(candidate)
    expected["record_claim_plan"] = plan
    expected["report_render"] = render
    fixed = {
        "schema_version": EVENT_FINDINGS_V3_SHADOW_BUNDLE_SCHEMA_VERSION,
        "route_id": EVENT_FINDINGS_V3_SHADOW_ROUTE_ID,
        "route_scope": source["route_scope"],
        "production_route_connected": False,
        "source_shadow_input_sha256": _sha256(source),
        "record_claim_plan": plan,
        "report_render": render,
        "route_receipt": {
            "public_or_synthetic_only": True,
            "private_production_inputs_authorized": False,
            "complete_event_roster_bound": True,
            "invalid_event_has_deterministic_output": True,
            "qwen_free_text_authorized": False,
            "qwen_failure_has_deterministic_fallback": True,
            "source_replay_required": True,
        },
        "bundle_sha256": candidate.get("bundle_sha256"),
    }
    if set(candidate) != set(fixed):
        raise ValueError("shadow report bundle keys are not closed")
    recomputed = deepcopy(fixed)
    _seal(
        recomputed,
        "bundle_sha256",
        "clinical-eeg-event-findings-v3-shadow-report-bundle-v1",
    )
    if candidate != recomputed:
        raise ValueError("shadow report bundle content binding mismatch")
    return recomputed


__all__ = [
    "ADAPTIVE_REPORT_ROUTE_CONNECTED",
    "EVENT_FINDINGS_V3_SHADOW_ROUTE_ID",
    "EVENT_FINDINGS_V3_SHADOW_INPUT_SCHEMA_VERSION",
    "EVENT_FINDINGS_V3_SHADOW_ROSTER_SCHEMA_VERSION",
    "EVENT_FINDINGS_V3_RECORD_CLAIM_PLAN_SCHEMA_VERSION",
    "EVENT_FINDINGS_V3_CLOSED_LEXICALIZATION_SCHEMA_VERSION",
    "EVENT_FINDINGS_V3_REPORT_RENDER_SCHEMA_VERSION",
    "EVENT_FINDINGS_V3_SHADOW_BUNDLE_SCHEMA_VERSION",
    "LEGACY_RECORD_CANDIDATE_BASELINE_METHOD_ID",
    "shadow_event_roster_sha256",
    "build_event_findings_v3_record_claim_plan",
    "render_event_findings_v3_report_zh",
    "materialize_event_findings_v3_shadow_report",
    "validate_event_findings_v3_shadow_report",
]
