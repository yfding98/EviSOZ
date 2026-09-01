#!/usr/bin/env python3
"""Materialize v22 qualified SOZ candidate/abstention reports.

The command is presentation-only.  It combines the already frozen, target-
blind v21.1 candidate/abstention decisions with sealed public typed facts and
private target-blind signal metadata.  It does not load raw EEG, SOZ targets,
private evaluation rows, or model weights and never changes a localization
score or threshold.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.migrate_public_clinical_presentations import (  # noqa: E402
    SOURCE_SCHEMA,
    SOURCE_STATUS,
    _event_clauses,
    _object,
    _ranking_clause,
    _reference_clause,
    _text,
)
from src.soz.clinical_reporting import (  # noqa: E402
    CLINICAL_SCALP_REGIONS,
    LATERALITY_GROUPS,
)
from src.soz.geometry import STANDARD_19  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs/trustworthy_soz_qualified_reporting_v22.json"
DEFAULT_CANDIDATE_DIRECTORY = (
    ROOT / "outputs/trustworthy_soz_selective_reports_v21_1_20260815"
)
DEFAULT_PUBLIC_SOURCE = (
    ROOT / "outputs/target_free_oof_reports_v3_recovered_20260813.json"
)
DEFAULT_PRIVATE_MANIFEST = (
    ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814/manifest.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_qualified_reports_v22_20260815"

OUTPUT_SCHEMA = "trustworthy_soz_qualified_report_v22"
MANIFEST_SCHEMA = "trustworthy_soz_qualified_reporting_manifest_v22"

_REGION_ZH = {
    "frontal": "额区",
    "temporal": "颞区",
    "central": "中央区",
    "parietal": "顶区",
    "occipital": "枕区",
}
_LATERALITY_ZH = {"left": "左", "right": "右", "midline": "中线"}

_LIMITATION = (
    "本结果仅为头皮物理电极SOZ-reference研究候选，不等同于侵入式皮层SOZ、"
    "致痫区或治疗靶点，不能独立用于手术决策，需由医生结合症状学、影像及"
    "必要时的侵入式电生理检查复核"
)
_QUALIFICATION = (
    "当前形态学和发作受累concept未通过独立资格门，时间变化仅作头皮可见描述；"
    "因此不生成形态类别、皮层起源、传播路径或伪迹严重度结论"
)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.resolve(strict=True).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _unique_group(
    channel: str, groups: Mapping[str, Sequence[str]], *, name: str
) -> str:
    matches = [group for group, channels in groups.items() if channel in channels]
    if len(matches) != 1:
        raise ValueError(f"{channel} has {len(matches)} memberships in {name}")
    return matches[0]


def _top1_location_zh(channel: str) -> str:
    if channel not in STANDARD_19:
        raise ValueError(f"unknown standard-19 channel: {channel}")
    region = _unique_group(channel, CLINICAL_SCALP_REGIONS, name="clinical regions")
    laterality = _unique_group(channel, LATERALITY_GROUPS, name="laterality groups")
    return _LATERALITY_ZH[laterality] + _REGION_ZH[region]


def _candidate_clause(
    record: Mapping[str, object],
) -> tuple[str, dict[str, object], list[str]]:
    if record.get("schema_version") != "trustworthy_soz_selective_report_v21_1":
        raise ValueError("candidate decision schema drifted")
    decision = _object(record.get("decision"), name="candidate decision")
    uncertainty = _object(record.get("uncertainty"), name="candidate uncertainty")
    margin = uncertainty.get("top1_top2_margin")
    threshold = uncertainty.get("frozen_threshold")
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(float(margin))
    ):
        raise ValueError("candidate margin is invalid")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        raise ValueError("candidate threshold is invalid")
    displayed = decision.get("displayed_candidates")
    if not isinstance(displayed, list):
        raise TypeError("displayed_candidates must be a list")
    action = decision.get("action")
    if action == "display_candidate":
        if float(margin) < float(threshold) or not displayed:
            raise ValueError("display decision disagrees with its frozen margin gate")
        candidates: list[dict[str, object]] = []
        channels: list[str] = []
        for raw in displayed:
            item = _object(raw, name="displayed candidate")
            channel = _text(item.get("channel"), name="candidate channel")
            score = item.get("normalized_candidate_score")
            if channel not in STANDARD_19 or channel in channels:
                raise ValueError("candidate channels are invalid or duplicated")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise ValueError("candidate score is invalid")
            channels.append(channel)
            candidates.append(
                {"channel": channel, "normalized_candidate_score": float(score)}
            )
        top1 = channels[0]
        location = _top1_location_zh(top1)
        text = (
            "冻结H-only定位器的C18头皮电极SOZ-reference候选依次为"
            + "、".join(channels)
            + f"；首位候选{top1}的确定性头皮空间投影为{location}。"
            "候选分数及显示门未经独立临床校准，仅供医生复核"
        )
        structured = {
            "action": action,
            "displayed_candidates": candidates,
            "top1_region_projection_zh": location,
            "top1_top2_margin": float(margin),
            "frozen_threshold": float(threshold),
            "calibrated_error_probability": None,
        }
        paths = [
            "candidate_report.decision.displayed_candidates",
            "candidate_report.uncertainty.top1_top2_margin",
            "candidate_report.uncertainty.frozen_threshold",
            "protocol.candidate_display",
        ]
        return text, structured, paths
    if action == "localization_abstain":
        if displayed or float(margin) >= float(threshold):
            raise ValueError("abstention disagrees with its frozen margin gate")
        text = (
            "冻结H-only定位器因候选分离度不足而对C18 SOZ-reference定位弃权，"
            "不显示隐藏排序；弃权不表示不存在SOZ"
        )
        structured = {
            "action": action,
            "displayed_candidates": [],
            "top1_region_projection_zh": None,
            "top1_top2_margin": float(margin),
            "frozen_threshold": float(threshold),
            "calibrated_error_probability": None,
        }
        paths = [
            "candidate_report.decision.action",
            "candidate_report.decision.reason_codes",
            "candidate_report.uncertainty.top1_top2_margin",
            "candidate_report.uncertainty.frozen_threshold",
            "protocol.candidate_display.abstention_hides_candidate_ranking",
        ]
        return text, structured, paths
    raise ValueError(f"unsupported candidate action: {action!r}")


def _clauses_to_text(clauses: Sequence[Mapping[str, object]]) -> str:
    values = [_text(clause.get("text"), name="report clause") for clause in clauses]
    return "。".join(value.rstrip("。") for value in values) + "。"


def _validate_clinical_text(text: str, forbidden: Sequence[str]) -> None:
    hits = [phrase for phrase in forbidden if phrase in text]
    if hits:
        raise ValueError(f"qualified clinical text contains forbidden phrases: {hits}")
    if "sha256" in text.lower() or "checkpoint" in text.lower():
        raise ValueError("qualified clinical text leaks machine-audit details")


def _base_record(
    *,
    cohort: str,
    unit_id: str,
    patient_id: str,
    candidate_report: Mapping[str, object] | None,
    clauses: list[dict[str, object]],
    forbidden: Sequence[str],
    evidence_status: str,
) -> dict[str, object]:
    if candidate_report is None:
        candidate_text = (
            "该事件所属患者不在当前冻结102人定位候选roster中，C18定位结果不可用；"
            "系统不显示其他版本或隐藏排名"
        )
        candidate = {
            "action": "localization_unavailable",
            "displayed_candidates": [],
            "top1_region_projection_zh": None,
            "top1_top2_margin": None,
            "frozen_threshold": None,
            "calibrated_error_probability": None,
        }
        candidate_paths = [
            "public_source.patient_id",
            "candidate_manifest.public_patient_roster",
            "protocol.candidate_display.abstention_hides_candidate_ranking",
        ]
    else:
        candidate_text, candidate, candidate_paths = _candidate_clause(candidate_report)
    clauses.extend(
        [
            {
                "type": "concept_qualification",
                "text": _QUALIFICATION,
                "fact_paths": ["protocol.concept_qualification"],
            },
            {
                "type": "soz_candidate_or_abstention",
                "text": candidate_text,
                "fact_paths": candidate_paths,
            },
            {
                "type": "clinical_limitation",
                "text": _LIMITATION,
                "fact_paths": ["protocol.clinical_boundary"],
            },
        ]
    )
    clinical_text = _clauses_to_text(clauses)
    _validate_clinical_text(clinical_text, forbidden)
    return {
        "schema_version": OUTPUT_SCHEMA,
        "cohort": cohort,
        "unit_id": unit_id,
        "patient_id": patient_id,
        "report_status": (
            f"{evidence_status}_{candidate['action']}_facts_locked"
        ),
        "concept_qualification": {
            "morphology": "failed_native_gate_structurally_absent",
            "ictal_involvement": "failed_native_gate_structurally_absent",
            "temporal_evolution": (
                "direct_observable_description_only_not_origin_or_propagation"
            ),
            "artifact_type_and_severity": (
                "unavailable_without_independent_reader_qualification"
            ),
        },
        "localization": candidate,
        "clinical_text_zh": clinical_text,
        "clauses": clauses,
        "sentence_fact_map": [
            {
                "sentence": index,
                "clause_type": clause["type"],
                "fact_paths": clause["fact_paths"],
            }
            for index, clause in enumerate(clauses, start=1)
        ],
        "claim_boundary": {
            "output": "scalp_electrode_clinical_reference_candidate",
            "candidate_score_is_calibrated_probability": False,
            "margin_is_error_probability": False,
            "clinical_risk_control_guarantee": False,
            "concepts_causally_explain_h_only_score": False,
            "cortical_soz_ez_or_treatment_target": False,
            "clinician_review_required": True,
        },
        "facts_locked": True,
        "llm_used": False,
    }


def _candidate_maps(
    directory: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], dict[str, object]]:
    manifest = _read_json(directory / "manifest.json")
    if manifest.get("schema_version") != (
        "trustworthy_soz_selective_reporting_manifest_v21_1"
    ):
        raise ValueError("candidate reporting manifest schema drifted")
    access = _object(manifest.get("access_receipt"), name="candidate access receipt")
    for field in (
        "soz_target_tensor_loaded",
        "private_target_ledger_loaded",
        "private_evaluation_rows_loaded",
    ):
        if access.get(field) is not False:
            raise ValueError(f"candidate report input violates target-blind rule: {field}")

    def keyed(filename: str, key: str) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for row in _read_jsonl(directory / filename):
            value = _text(row.get(key), name=key)
            if value in result:
                raise ValueError(f"duplicate candidate report {key}: {value}")
            result[value] = row
        return result

    return (
        keyed("public_reports.jsonl", "patient_id"),
        keyed("private_reports.jsonl", "unit_id"),
        manifest,
    )


def materialize(
    *,
    protocol: Mapping[str, object],
    public_source: Mapping[str, object],
    private_manifest: Mapping[str, object],
    public_candidates: Mapping[str, Mapping[str, object]],
    private_candidates: Mapping[str, Mapping[str, object]],
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    if protocol.get("schema_version") != (
        "trustworthy_soz_qualified_reporting_protocol_v22"
    ):
        raise ValueError("qualified reporting protocol schema drifted")
    forbidden_value = protocol.get("forbidden_clinical_phrases")
    if not isinstance(forbidden_value, list) or not all(
        isinstance(value, str) and value for value in forbidden_value
    ):
        raise ValueError("forbidden clinical phrase list is invalid")
    forbidden = tuple(forbidden_value)

    if public_source.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("public typed-fact source schema drifted")
    if public_source.get("status") != SOURCE_STATUS:
        raise ValueError("public typed-fact source status drifted")
    public_rows_value = public_source.get("records")
    if not isinstance(public_rows_value, list):
        raise TypeError("public source records must be a list")

    patient_event_rosters: dict[str, set[str]] = defaultdict(set)
    for raw in public_rows_value:
        row = _object(raw, name="public source row")
        if row.get("report") is None:
            continue
        patient_id = _text(row.get("patient_id"), name="public patient_id")
        event_id = _text(row.get("event_id"), name="public event_id")
        if event_id in patient_event_rosters[patient_id]:
            raise ValueError("duplicate public patient/event source row")
        patient_event_rosters[patient_id].add(event_id)

    public_event_reports: list[dict[str, object]] = []
    public_blocked_event_evidence = 0
    for raw in public_rows_value:
        row = _object(raw, name="public source row")
        patient_id = _text(row.get("patient_id"), name="public patient_id")
        event_id = _text(row.get("event_id"), name="public event_id")
        candidate_report = public_candidates.get(patient_id)
        report_value = row.get("report")
        if candidate_report is None and report_value is not None:
            raise ValueError(
                "a reportable public event lacks its frozen candidate decision: "
                f"{patient_id}"
            )
        clauses: list[dict[str, object]] = []
        if report_value is None:
            public_blocked_event_evidence += 1
            clauses.append(
                {
                    "type": "event_evidence_unavailable",
                    "text": (
                        "本事件的可追溯事件级证据未通过身份与收据完整性门，"
                        "因此不生成时间、双极边、节律或伪迹陈述"
                    ),
                    "fact_paths": ["public_source.reason_codes"],
                }
            )
            evidence_status = "event_evidence_unavailable"
        else:
            report = _object(report_value, name="public grounded report")
            assembly = _object(row.get("assembly_receipt"), name="assembly_receipt")
            facts = _object(row.get("typed_facts"), name="typed_facts")
            event, later, artifact, event_status = _event_clauses(
                row=row, report=report, assembly=assembly, facts=facts
            )
            _, aggregation_count, event_ids = _ranking_clause(
                report=report, facts=facts
            )
            if set(event_ids) != patient_event_rosters[patient_id]:
                raise ValueError("public ranking does not aggregate its full event roster")
            if aggregation_count != len(patient_event_rosters[patient_id]):
                raise ValueError("public aggregation count drifted")
            reference, _ = _reference_clause(
                facts=facts, patient_id=patient_id, event_ids=event_ids
            )
            clauses.append(
                {
                    "type": "event_scalp_evidence",
                    "text": event,
                    "fact_paths": [
                        "public_source.typed_facts.event_phenotype",
                        "public_source.assembly_receipt.global_t0_sec",
                    ],
                }
            )
            if later:
                clauses.append(
                    {
                        "type": "later_visible_order",
                        "text": later,
                        "fact_paths": [
                            "public_source.typed_facts.event_phenotype.later_visible_*"
                        ],
                    }
                )
            clauses.extend(
                [
                    {
                        "type": "artifact_qualification",
                        "text": artifact,
                        "fact_paths": [
                            "public_source.typed_facts.event_phenotype.artifact_*"
                        ],
                    },
                    {
                        "type": "reference_sensitivity",
                        "text": reference,
                        "fact_paths": [
                            "public_source.typed_facts.final_score_reference_disagreement_receipt"
                        ],
                    },
                ]
            )
            evidence_status = event_status
        public_event_reports.append(
            _base_record(
                cohort="public_deepsoz_development_event_report",
                unit_id=event_id,
                patient_id=patient_id,
                candidate_report=candidate_report,
                clauses=clauses,
                forbidden=forbidden,
                evidence_status=evidence_status,
            )
        )

    public_patient_reports = [
        _base_record(
            cohort="public_deepsoz_development_patient_report",
            unit_id=patient_id,
            patient_id=patient_id,
            candidate_report=candidate_report,
            clauses=[
                {
                    "type": "analysis_scope",
                    "text": (
                        "该结果来自已确认发作事件的患者级冻结H-only表征聚合；"
                        "本展示层未读取SOZ标签或改变定位分数"
                    ),
                    "fact_paths": [
                        "candidate_manifest.access_receipt",
                        "candidate_report.patient_id",
                    ],
                }
            ],
            forbidden=forbidden,
            evidence_status="patient_level_candidate_scope",
        )
        for patient_id, candidate_report in sorted(public_candidates.items())
    ]

    if private_manifest.get("schema_version") != (
        "soz_private_target_blind_labram_evidence_v18"
    ):
        raise ValueError("private target-blind manifest schema drifted")
    private_access = _object(
        private_manifest.get("access_receipt"), name="private access receipt"
    )
    for field in (
        "target_ledger_opened",
        "private_target_values_loaded",
        "deepsoz_target_values_loaded",
    ):
        if private_access.get(field) is not False:
            raise ValueError(
                f"private target-blind manifest violated target access: {field}"
            )
    preprocessing = _object(
        private_manifest.get("preprocessing"), name="private preprocessing"
    )
    if preprocessing.get("output_sfreq_hz") != 200:
        raise ValueError("private output sampling rate drifted")
    if preprocessing.get("reference_policy") != "unlabeled_common_car19":
        raise ValueError("private reference policy drifted")
    window = preprocessing.get("window_sec")
    if window != [-12, 48]:
        raise ValueError("private event window drifted")
    private_events = private_manifest.get("events")
    if not isinstance(private_events, list):
        raise TypeError("private events must be a list")
    private_event_reports: list[dict[str, object]] = []
    seen_private: set[str] = set()
    for raw in private_events:
        event = _object(raw, name="private event")
        event_id = _text(event.get("event_id"), name="private event_id")
        patient_id = _text(event.get("patient_id"), name="private patient_id")
        if event_id in seen_private:
            raise ValueError("duplicate private event")
        seen_private.add(event_id)
        candidate_report = private_candidates.get(event_id)
        if candidate_report is None:
            raise ValueError(f"private event lacks frozen candidate decision: {event_id}")
        source_sfreq = event.get("source_sfreq_hz")
        if isinstance(source_sfreq, bool) or not isinstance(source_sfreq, (int, float)):
            raise ValueError("private source sampling rate is invalid")
        clauses = [
            {
                "type": "analysis_scope",
                "text": (
                    "本次分析使用已确认发作事件的标准19导头皮EEG；"
                    f"原始采样率{float(source_sfreq):g} Hz，按冻结流程输出为"
                    "200 Hz、CAR19及相对事件锚点[-12,+48)秒窗口"
                ),
                "fact_paths": [
                    "private_target_blind_manifest.events.source_sfreq_hz",
                    "private_target_blind_manifest.preprocessing",
                ],
            },
            {
                "type": "event_evidence_unavailable",
                "text": (
                    "private事件当前没有经独立资格化的时间、双极边、形态、"
                    "传播或伪迹类型证据，相关报告槽保持缺席"
                ),
                "fact_paths": ["protocol.private_event_evidence"],
            },
        ]
        private_event_reports.append(
            _base_record(
                cohort="private_post_open_target_blind_event_report",
                unit_id=event_id,
                patient_id=patient_id,
                candidate_report=candidate_report,
                clauses=clauses,
                forbidden=forbidden,
                evidence_status="private_event_evidence_unavailable",
            )
        )
    if seen_private != set(private_candidates):
        raise ValueError("private target-blind event and candidate rosters disagree")

    all_reports = public_event_reports + public_patient_reports + private_event_reports
    for report in all_reports:
        if report.get("facts_locked") is not True or report.get("llm_used") is not False:
            raise RuntimeError("qualified report lost its facts-locked boundary")
        _validate_clinical_text(_text(report.get("clinical_text_zh"), name="text"), forbidden)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "completed_target_blind_qualified_reports_v22",
        "counts": {
            "public_patient_reports": len(public_patient_reports),
            "public_event_reports": len(public_event_reports),
            "public_event_evidence_blocked": public_blocked_event_evidence,
            "public_event_localization_unavailable": sum(
                row["localization"]["action"] == "localization_unavailable"
                for row in public_event_reports
            ),
            "private_event_reports": len(private_event_reports),
            "public_patient_display_candidate": sum(
                row["localization"]["action"] == "display_candidate"
                for row in public_patient_reports
            ),
            "public_patient_localization_abstain": sum(
                row["localization"]["action"] == "localization_abstain"
                for row in public_patient_reports
            ),
            "private_event_display_candidate": sum(
                row["localization"]["action"] == "display_candidate"
                for row in private_event_reports
            ),
            "private_event_localization_abstain": sum(
                row["localization"]["action"] == "localization_abstain"
                for row in private_event_reports
            ),
            "forbidden_phrase_hits": 0,
        },
        "access_receipt": {
            "raw_eeg_loaded": False,
            "deepsoz_target_values_loaded": False,
            "private_target_ledger_loaded": False,
            "private_evaluation_rows_loaded": False,
            "private_target_blind_signal_metadata_loaded": True,
            "localization_scores_or_threshold_changed": False,
            "training_performed": False,
            "model_selection_performed": False,
            "llm_used": False,
        },
        "files": {
            "public_patient_reports": "public_patient_reports.jsonl",
            "public_event_reports": "public_event_reports.jsonl",
            "private_event_reports": "private_event_reports.jsonl",
        },
        "scientific_boundary": {
            "public_102_is_developmental": True,
            "private_is_post_open_descriptive": True,
            "reports_are_diagnostic_candidates_not_confirmed_diagnoses": True,
            "morphology_and_ictal_concepts_are_structurally_absent": True,
            "propagation_or_artifact_severity_claim_allowed": False,
            "cortical_soz_ez_or_surgical_target_claim_allowed": False,
        },
    }
    return manifest, public_patient_reports, public_event_reports, private_event_reports


def run(args: argparse.Namespace) -> dict[str, object]:
    protocol = _read_json(args.protocol)
    public_source = _read_json(args.public_source)
    private_manifest = _read_json(args.private_manifest)
    public_candidates, private_candidates, _ = _candidate_maps(
        args.candidate_directory
    )
    manifest, public_patient, public_event, private_event = materialize(
        protocol=protocol,
        public_source=public_source,
        private_manifest=private_manifest,
        public_candidates=public_candidates,
        private_candidates=private_candidates,
    )

    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        _write_jsonl(bundle / "public_patient_reports.jsonl", public_patient)
        _write_jsonl(bundle / "public_event_reports.jsonl", public_event)
        _write_jsonl(bundle / "private_event_reports.jsonl", private_event)
        (bundle / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        bundle.replace(destination)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--candidate-directory", type=Path, default=DEFAULT_CANDIDATE_DIRECTORY
    )
    parser.add_argument("--public-source", type=Path, default=DEFAULT_PUBLIC_SOURCE)
    parser.add_argument(
        "--private-manifest", type=Path, default=DEFAULT_PRIVATE_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = run(args)
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
