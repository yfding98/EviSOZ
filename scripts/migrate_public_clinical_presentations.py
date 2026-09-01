#!/usr/bin/env python3
"""Create fact-derived public clinical views from sealed v3 reports.

This is a presentation-only migration. It reads neither raw EEG nor target
values and it never changes a localization score. Unlike the historical v1
migration, every public clause is rebuilt from ``typed_facts`` and the sealed
source phrases are checked for drift before a record can be presented.

The renderer is deliberately conservative:

* an algorithmic first-visible scalp change is not called an SOZ onset;
* later-visible evidence is not called propagation;
* ``evolving_rhythmic`` from the v1 phenotype producer is rendered only as a
  spectral rhythm feature because that producer did not isolate temporal
  frequency variation from between-edge variation;
* uncalibrated localization scores are never rendered as probabilities; and
* absent artifact facts are stated as unavailable, not silently invented.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
from typing import Mapping


SOURCE_SCHEMA = "soz_target_free_oof_report_assembler_v3"
SOURCE_STATUS = "completed_target_free_oof_reports_all_rankings_abstain"
OUTPUT_SCHEMA = "soz_public_clinical_presentations_v2"
OUTPUT_STATUS = "completed_fact_derived_presentations_all_rankings_abstain"

_EVENT_COUNT_RE = re.compile(r"共(?P<count>\d+)个发作事件")
_FORBIDDEN = (
    "首先出现",
    "最早可见",
    "最早物理电极",
    "传播至",
    "传播到",
    "后扩展至",
    "后扩展到",
    "皮层SOZ可疑位于",
    "皮层SOZ位于",
    "轻度肌电",
    "中度肌电",
    "重度肌电",
    "定位概率",
    "SOZ概率",
)
_TECHNICAL = (
    re.compile(r"sha256", re.IGNORECASE),
    re.compile(r"checkpoint", re.IGNORECASE),
    re.compile(r"__ev\d+"),
    re.compile(r"原因代码"),
)
_RHYTHM_SOURCE_ZH = {
    "rhythmic": "节律性活动",
    "evolving_rhythmic": "持续演变的节律性活动",
}
_ARTIFACT_ZH = {
    "muscle": "肌电",
    "ocular": "眼动",
    "movement": "运动",
    "electrode_transient": "电极瞬态",
    "line_noise": "工频",
}
_REGION_ZH = {
    "frontal": "额区",
    "temporal": "颞区",
    "central": "中央区",
    "parietal": "顶区",
    "occipital": "枕区",
    "multiregional": "多区域",
}
_LATERALITY_ZH = {
    "left": "左",
    "right": "右",
    "bilateral": "双侧",
    "midline": "中线",
    "indeterminate": "侧别不确定的",
}
_SOURCE_LIMITATION = (
    "患者级排序仅为头皮物理电极SOZ-reference候选排序，不等同于侵入式"
    "皮层SOZ、致痫区或治疗靶点，不得单独作为手术决策依据，需由医生结合"
    "完整临床资料确认"
)
_PUBLIC_LIMITATION = (
    "本结果仅为头皮物理电极SOZ-reference研究候选，不等同于侵入式皮层SOZ、"
    "致痫区或手术靶点，不能独立用于手术决策，需由医生结合症状学、影像及"
    "侵入式检查复核"
)


def _object(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")
    return value


def _finite(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _join(parts: list[str]) -> str:
    return "。".join(part.rstrip("。") for part in parts if part) + "。"


def _source_frequency_phrase(value: object) -> str:
    if value is None:
        return ""
    pair = _list(value, name="frequency_range_hz")
    if len(pair) != 2:
        raise ValueError("frequency_range_hz must contain two values")
    lower = _finite(pair[0], name="frequency_range_hz[0]")
    upper = _finite(pair[1], name="frequency_range_hz[1]")
    if lower <= 0.0 or upper < lower or upper > 45.0:
        raise ValueError("frequency_range_hz is outside the report contract")
    bands = (
        (0.5, 4.0, "δ"),
        (4.0, 8.0, "θ"),
        (8.0, 13.0, "α"),
        (13.0, 30.0, "β"),
        (30.0, 45.000001, "低γ"),
    )
    for band_lower, band_upper, label in bands:
        if lower >= band_lower and upper <= band_upper:
            return f"主频位于{label}频段（{lower:.1f}–{upper:.1f} Hz）"
    return f"主频范围为{lower:.1f}–{upper:.1f} Hz"


def _public_frequency_phrase(value: object) -> str:
    source = _source_frequency_phrase(value)
    if not source:
        return ""
    if source.startswith("主频位于"):
        return "局部" + source
    return "局部主频估计" + source.removeprefix("主频")


def _event_clauses(
    *,
    row: Mapping[str, object],
    report: Mapping[str, object],
    assembly: Mapping[str, object],
    facts: Mapping[str, object],
) -> tuple[str, str, str, str]:
    event = _object(facts.get("event_phenotype"), name="typed event phenotype")
    receipt = _object(event.get("receipt"), name="event evidence receipt")
    patient_id = _text(row.get("patient_id"), name="patient_id")
    event_id = _text(row.get("event_id"), name="event_id")
    if receipt.get("patient_pseudonym") != patient_id:
        raise ValueError("event receipt patient identity drifted")
    if receipt.get("event_pseudonym") != event_id:
        raise ValueError("event receipt event identity drifted")
    if receipt.get("time_coordinate_semantics") != "recording_start_seconds":
        raise ValueError("public relative time requires recording-start coordinates")
    if receipt.get("soz_labels_used_for_event_evidence") is not False:
        raise ValueError("event evidence is not SOZ-target-free")
    if receipt.get("private_labels_used_for_event_evidence") is not False:
        raise ValueError("event evidence used private labels")

    source_onset = _text(
        report.get("event_phenotype_phrase"), name="event_phenotype_phrase"
    )
    if event.get("onset_start_sec") is None:
        reasons = [
            _text(value, name="event abstention reason")
            for value in _list(event.get("reason_codes"), name="event reason_codes")
        ]
        if not reasons:
            raise ValueError("event abstention lacks a reason")
        expected = (
            "事件级头皮持续变化候选：固定算法未形成可报告候选；原因代码："
            + "、".join(reasons)
        )
        if source_onset != expected:
            raise ValueError("sealed event-abstention phrase disagrees with typed facts")
        return (
            "事件级头皮证据未形成可报告的持续变化候选",
            "",
            "当前未形成经验证的伪迹类型及严重度结论",
            "event_phenotype_abstained",
        )

    start = _finite(event.get("onset_start_sec"), name="onset_start_sec")
    end = _finite(event.get("onset_end_sec"), name="onset_end_sec")
    if end < start:
        raise ValueError("event evidence interval is unordered")
    anchor = _finite(assembly.get("global_t0_sec"), name="global_t0_sec")
    relative_start = start - anchor
    relative_end = end - anchor
    if relative_start < -1e-9:
        raise ValueError("event evidence precedes its declared clinical anchor")
    edges = [
        _text(value, name="first_visible_derivation")
        for value in _list(
            event.get("first_visible_derivations"),
            name="first_visible_derivations",
        )
    ]
    if not edges:
        raise ValueError("reportable event has no first-visible derivation")
    display_edges = "及".join(value.replace("-", "–") for value in edges)
    expected_parts = [
        "事件级头皮持续变化候选："
        f"自记录起{start:.1f}–{end:.1f}秒，"
        f"固定算法最先检出的持续变化候选位于{display_edges}"
    ]
    rhythm = event.get("rhythm_state")
    if rhythm is not None:
        if rhythm not in _RHYTHM_SOURCE_ZH:
            raise ValueError("unknown rhythm_state")
        expected_parts.append(f"表现为{_RHYTHM_SOURCE_ZH[str(rhythm)]}")
    frequency = _source_frequency_phrase(event.get("frequency_range_hz"))
    if frequency:
        expected_parts.append(frequency)
    expected_source = "，".join(expected_parts)
    if source_onset != expected_source:
        raise ValueError("sealed event-evidence phrase disagrees with typed facts")

    public_parts = [
        "事件级头皮证据：相对TUSZ发作事件标注起点"
        f"{relative_start:.1f}–{relative_end:.1f}秒的算法证据支持窗内，"
        f"按固定检测规则标记的首批持续变化候选双极导联为{display_edges}"
    ]
    if rhythm is not None:
        # Producer v1's frequency span mixes time and between-edge variation.
        # Therefore both states support only a conservative rhythm-feature claim.
        public_parts.append("局部频谱门控支持节律性特征")
    public_frequency = _public_frequency_phrase(event.get("frequency_range_hz"))
    if public_frequency:
        public_parts.append(public_frequency)
    event_clause = "，".join(public_parts)

    later_edges = [
        _text(value, name="later_visible_derivation")
        for value in _list(
            event.get("later_visible_derivations", []),
            name="later_visible_derivations",
        )
    ]
    later_region = event.get("later_visible_region_zh")
    if later_region is not None:
        later_region = _text(later_region, name="later_visible_region_zh")
    source_later = str(report.get("later_visible_phrase", ""))
    if not later_edges and later_region is None:
        if source_later:
            raise ValueError("sealed later-visible phrase lacks typed support")
        later_clause = ""
    else:
        destination = later_region or "、".join(
            value.replace("-", "–") for value in later_edges
        )
        delay_value = event.get("later_visible_delay_sec")
        if delay_value is None:
            source_timing = "随后"
            public_timing = "随后"
        else:
            delay = _finite(delay_value, name="later_visible_delay_sec")
            if delay < 0.0:
                raise ValueError("later-visible delay is negative")
            source_timing = f"约{delay:.1f}秒后"
            public_timing = source_timing
        expected_later = (
            f"事件内{source_timing}在{destination}可见后续受累/范围扩展"
            "（仅描述头皮可见范围变化，不作为传播真值）"
        )
        if source_later != expected_later:
            raise ValueError("sealed later-visible phrase disagrees with typed facts")
        later_clause = (
            f"{public_timing}在{destination}检测到后续持续变化候选；"
            "该先后次序仅描述头皮可见信号，不是传播路径、SOZ起始顺序或"
            "皮层传播速度真值"
        )

    assessed = event.get("artifact_assessed")
    artifact_types = [
        _text(value, name="artifact type")
        for value in _list(event.get("artifact_types", []), name="artifact_types")
    ]
    burden_value = event.get("artifact_burden")
    if assessed is not True:
        if artifact_types or burden_value is not None:
            raise ValueError("artifact details lack an assessed artifact fact")
        artifact_clause = "当前未形成经验证的伪迹类型及严重度结论"
    elif not artifact_types:
        artifact_clause = "预定义伪迹评估未提示主导伪迹"
    else:
        try:
            names = "、".join(_ARTIFACT_ZH[value] for value in artifact_types)
        except KeyError as exc:
            raise ValueError("unknown artifact type") from exc
        artifact_clause = f"预定义伪迹评估记录到{names}伪迹"
        if burden_value is not None:
            burden = _finite(burden_value, name="artifact_burden")
            if not 0.0 <= burden <= 1.0:
                raise ValueError("artifact_burden lies outside [0,1]")
            artifact_clause += f"，连续负担指标为{burden:.2f}（未经临床阈值校准）"
    return event_clause, later_clause, artifact_clause, "event_phenotype_reportable"


def _spatial_location(spatial: Mapping[str, object]) -> str:
    regions = [
        _text(value, name="top scalp region")
        for value in _list(
            spatial.get("top_scalp_regions"), name="top_scalp_regions"
        )
    ]
    region = regions[0] if len(regions) == 1 else "multiregional"
    laterality = _text(spatial.get("laterality"), name="laterality")
    try:
        return _LATERALITY_ZH[laterality] + _REGION_ZH[region]
    except KeyError as exc:
        raise ValueError("spatial report uses an unknown clinical mapping") from exc


def _ranking_clause(
    *,
    report: Mapping[str, object],
    facts: Mapping[str, object],
) -> tuple[str, int, tuple[str, ...]]:
    ranking = _object(facts.get("patient_ranking"), name="patient_ranking")
    spatial = _object(ranking.get("spatial_report"), name="spatial_report")
    if spatial.get("score_semantics") != "uncalibrated_localization_score":
        raise ValueError("unexpected localization-score semantics")
    channels = tuple(
        _text(value, name="top channel")
        for value in _list(spatial.get("top_channels"), name="top_channels")
    )
    if not channels:
        raise ValueError("patient ranking has no top channel")
    uncertainty = _object(ranking.get("uncertainty"), name="uncertainty")
    if uncertainty.get("abstain") is not True:
        raise ValueError("public migration requires an abstained patient ranking")
    reasons = tuple(
        _text(value, name="ranking abstention reason")
        for value in _list(
            uncertainty.get("abstention_reason_codes"),
            name="abstention_reason_codes",
        )
    )
    if "selective_threshold_undefined" not in reasons:
        raise ValueError("ranking lacks the undefined-threshold abstention")
    count_value = ranking.get("aggregation_event_count")
    if (
        isinstance(count_value, bool)
        or not isinstance(count_value, int)
        or count_value < 1
    ):
        raise ValueError("aggregation_event_count must be a positive integer")
    event_ids = tuple(
        _text(value, name="aggregation event id")
        for value in _list(
            ranking.get("aggregation_event_ids"), name="aggregation_event_ids"
        )
    )
    if len(event_ids) != count_value or len(set(event_ids)) != len(event_ids):
        raise ValueError("patient aggregation roster is inconsistent")
    aggregation = _text(
        report.get("event_aggregation_phrase"), name="event_aggregation_phrase"
    )
    match = _EVENT_COUNT_RE.search(aggregation)
    if match is None or int(match.group("count")) != count_value:
        raise ValueError("sealed aggregation phrase disagrees with typed facts")

    display_channels = "/".join(channels)
    tie = "并列首位候选" if len(channels) > 1 else "首位候选"
    expected_source = (
        "患者级SOZ-reference排序已拒绝形成稳定结论；"
        f"当前SOZ-reference{tie}为{display_channels}，仅保留供人工复核"
    )
    if report.get("patient_ranking_phrase") != expected_source:
        raise ValueError("sealed patient-ranking phrase disagrees with typed facts")
    location = _spatial_location(spatial)
    public = (
        f"患者级结果综合{count_value}次发作；C18 SOZ-reference排序尚无独立"
        "校准的自动采信阈值，未形成自动定位结论；未校准研究排序的"
        f"当前{tie}为{display_channels}，其确定性头皮区域映射为{location}，"
        "仅供医生复核"
    )
    return public, count_value, event_ids


def _reference_clause(
    *, facts: Mapping[str, object], patient_id: str, event_ids: tuple[str, ...]
) -> tuple[str, bool]:
    reference = _object(
        facts.get("final_score_reference_disagreement_receipt"),
        name="final-score reference receipt",
    )
    if reference.get("patient_pseudonym") != patient_id:
        raise ValueError("final-score reference receipt patient drifted")
    receipt_events = tuple(
        _text(value, name="reference aggregation event")
        for value in _list(
            reference.get("aggregation_event_ids"),
            name="reference aggregation_event_ids",
        )
    )
    if receipt_events != event_ids:
        raise ValueError("reference receipt and patient aggregation roster disagree")
    if reference.get("primary_arm_id") != "C-CAR19":
        raise ValueError("unexpected primary reference arm")
    if reference.get("sensitivity_arm_id") != "C-REF19":
        raise ValueError("unexpected sensitivity reference arm")
    if reference.get("same_frozen_model") is not True:
        raise ValueError("reference comparison did not use one frozen model")
    if reference.get("training_performed") is not False:
        raise ValueError("reference comparison unexpectedly trained a model")
    agreement = reference.get("top1_reference_agreement")
    if type(agreement) is not bool:
        raise ValueError("top1_reference_agreement must be bool")
    primary_top1 = _text(reference.get("primary_top1_channel"), name="primary_top1")
    sensitivity_top1 = _text(
        reference.get("sensitivity_top1_channel"), name="sensitivity_top1"
    )
    if agreement != (primary_top1 == sensitivity_top1):
        raise ValueError("reference-agreement flag disagrees with top channels")
    state = "一致" if agreement else "不一致"
    return (
        "同一冻结定位器在C-CAR19与C-REF19下的患者级首位候选"
        f"{state}；该结果仅描述参考方式敏感性，不代表定位正确性",
        agreement,
    )


def _validate_public_text(clinical_text: str) -> None:
    if any(value in clinical_text for value in _FORBIDDEN):
        raise ValueError("clinical text contains an unsupported clinical claim")
    if any(pattern.search(clinical_text) for pattern in _TECHNICAL):
        raise ValueError("clinical text leaks machine-audit details")


def migrate(source: dict[str, object]) -> dict[str, object]:
    if source.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("source schema drifted")
    if source.get("status") != SOURCE_STATUS:
        raise ValueError("source status drifted")
    boundary = _object(source.get("scientific_boundary"), name="scientific_boundary")
    if boundary.get("all_patient_rankings_abstain") is not True:
        raise ValueError("source rankings are not uniformly abstained")
    if boundary.get("clinical_deployment_eligible") is not False:
        raise ValueError("source unexpectedly claims deployment eligibility")
    if boundary.get("cortical_soz_claim_allowed") is not False:
        raise ValueError("source unexpectedly allows a cortical SOZ claim")
    records = _list(source.get("records"), name="records")

    patient_event_rosters: dict[str, set[str]] = defaultdict(set)
    for raw in records:
        row = _object(raw, name="source record")
        if row.get("report") is None:
            continue
        patient_id = _text(row.get("patient_id"), name="patient_id")
        event_id = _text(row.get("event_id"), name="event_id")
        if event_id in patient_event_rosters[patient_id]:
            raise ValueError("duplicate patient/event source record")
        patient_event_rosters[patient_id].add(event_id)

    output_rows: list[dict[str, object]] = []
    blocked = 0
    event_reportable = 0
    event_abstained = 0
    later_visible_reported = 0
    artifact_assessed = 0
    reference_top1_agreement = 0
    for raw in records:
        row = _object(raw, name="source record")
        report_value = row.get("report")
        if report_value is None:
            blocked += 1
            reasons = [
                _text(value, name="blocked reason")
                for value in _list(row.get("reason_codes"), name="blocked reason_codes")
            ]
            output_rows.append(
                {
                    "patient_id": row.get("patient_id"),
                    "event_id": row.get("event_id"),
                    "status": "blocked_source_report",
                    "clinical_text": None,
                    "blocked_reason_codes": reasons,
                }
            )
            continue
        report = _object(report_value, name="report")
        assembly = _object(row.get("assembly_receipt"), name="assembly_receipt")
        facts = _object(row.get("typed_facts"), name="typed_facts")
        patient_id = _text(row.get("patient_id"), name="patient_id")
        event_id = _text(row.get("event_id"), name="event_id")
        if assembly.get("patient_id") != patient_id:
            raise ValueError("assembly receipt patient identity drifted")
        if assembly.get("event_id") != event_id:
            raise ValueError("assembly receipt event identity drifted")
        if assembly.get("private_data_loaded") is not False:
            raise ValueError("assembly receipt used private data")
        if assembly.get("deepsoz_target_values_loaded") is not False:
            raise ValueError("report assembly loaded DeepSOZ target values")

        event, later, artifact, event_status = _event_clauses(
            row=row,
            report=report,
            assembly=assembly,
            facts=facts,
        )
        ranking, aggregation_count, event_ids = _ranking_clause(
            report=report, facts=facts
        )
        if set(event_ids) != patient_event_rosters[patient_id]:
            raise ValueError("patient ranking does not aggregate the full source roster")
        if aggregation_count != len(patient_event_rosters[patient_id]):
            raise ValueError("aggregation count disagrees with source patient roster")
        reference, agrees = _reference_clause(
            facts=facts, patient_id=patient_id, event_ids=event_ids
        )
        if report.get("limitation_phrase") != _SOURCE_LIMITATION:
            raise ValueError("sealed limitation phrase drifted")
        clinical_text = _join(
            [event, later, artifact, ranking, reference, _PUBLIC_LIMITATION]
        )
        _validate_public_text(clinical_text)

        event_fact = _object(facts.get("event_phenotype"), name="event phenotype")
        if event_status == "event_phenotype_reportable":
            event_reportable += 1
        else:
            event_abstained += 1
        if later:
            later_visible_reported += 1
        if event_fact.get("artifact_assessed") is True:
            artifact_assessed += 1
        if agrees:
            reference_top1_agreement += 1
        output_rows.append(
            {
                "patient_id": patient_id,
                "event_id": event_id,
                "status": f"clinical_view_{event_status}_ranking_abstained",
                "clinical_text": clinical_text,
                "claim_support": {
                    "event_time_and_edges": "typed_facts.event_phenotype",
                    "clinical_anchor": "assembly_receipt.global_t0_sec",
                    "later_visible_order": "typed_facts.event_phenotype.later_visible_*",
                    "artifact_statement": "typed_facts.event_phenotype.artifact_*",
                    "patient_candidate": "typed_facts.patient_ranking",
                    "reference_sensitivity": (
                        "typed_facts.final_score_reference_disagreement_receipt"
                    ),
                },
            }
        )

    assembled = sum(row["clinical_text"] is not None for row in output_rows)
    source_counts = _object(source.get("counts"), name="source counts")
    if source_counts.get("output_records") != len(records):
        raise ValueError("source output-record count drifted")
    if source_counts.get("assembled") != assembled:
        raise ValueError("source assembled count drifted")
    if source_counts.get("blocked") != blocked:
        raise ValueError("source blocked count drifted")
    return {
        "schema_version": OUTPUT_SCHEMA,
        "status": OUTPUT_STATUS,
        "source_schema": SOURCE_SCHEMA,
        "counts": {
            "source_records": len(records),
            "assembled": assembled,
            "blocked": blocked,
            "patients_with_presentations": len(patient_event_rosters),
            "event_phenotype_reportable": event_reportable,
            "event_phenotype_abstained": event_abstained,
            "later_visible_reported": later_visible_reported,
            "artifact_assessed": artifact_assessed,
            "reference_top1_agreement": reference_top1_agreement,
            "reference_top1_disagreement": assembled - reference_top1_agreement,
        },
        "access_receipt": {
            "raw_eeg_loaded": False,
            "deepsoz_target_values_loaded": False,
            "private_data_loaded": False,
            "localization_scores_changed": False,
            "training_performed": False,
            "model_selection_performed": False,
            "llm_used": False,
        },
        "claim_grounding": {
            "time_window": (
                "typed event onset interval minus the bound TUSZ event anchor; "
                "algorithmic evidence-support window, not cortical onset"
            ),
            "first_visible_edges": (
                "fixed producer ordering among detected bipolar scalp changes; "
                "not an onset-electrode or SOZ label"
            ),
            "rhythm_and_frequency": (
                "v1 local spectral gate; evolving_rhythmic is deliberately not "
                "rendered as clinical temporal evolution"
            ),
            "later_visible": (
                "subsequent scalp-visible change only; no propagation, direction, "
                "speed, or origin claim"
            ),
            "artifact": (
                "rendered unavailable unless artifact_assessed is explicitly true"
            ),
            "patient_candidate": (
                "full patient event-bag C18 uncalibrated ranking; every ranking abstains"
            ),
            "region": "deterministic scalp-ontology mapping of the candidate electrode",
            "reference_sensitivity": (
                "same frozen locator under C-CAR19 and C-REF19; not correctness"
            ),
        },
        "scientific_boundary": {
            "presentation_only": True,
            "all_rankings_abstain": True,
            "evaluation_eligible": False,
            "clinical_deployment_eligible": False,
            "cortical_soz_claim_allowed": False,
            "machine_audit_remains_source_of_truth": True,
            "artifact_severity_claim_allowed": artifact_assessed > 0,
            "propagation_claim_allowed": False,
            "evolving_rhythmic_rendered_as_clinical_evolution": False,
            "patient_candidate_uses_full_available_event_roster": True,
        },
        "records": output_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output = migrate(source)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
