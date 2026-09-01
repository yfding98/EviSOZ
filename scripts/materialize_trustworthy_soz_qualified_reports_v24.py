#!/usr/bin/env python3
"""Enrich frozen v22 reports with target-blind private signal descriptors.

Candidate rankings, abstentions, scores, thresholds, and all public clauses are
preserved. Private reports receive only algorithmic sustained-change and later
scalp-visible descriptors. Rhythm, propagation, artifact, montage-consistency,
physical-electrode onset, and cortical SOZ claims remain withheld.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs/trustworthy_soz_qualified_reports_v22_20260815"
DEFAULT_DESCRIPTORS = ROOT / "outputs/private_event_descriptors_target_blind_v24_20260815"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_qualified_reports_v24_1_20260815"
INPUT_REPORT_SCHEMA = "trustworthy_soz_qualified_report_v22"
OUTPUT_REPORT_SCHEMA = "trustworthy_soz_qualified_report_v24"
INPUT_MANIFEST_SCHEMA = "trustworthy_soz_qualified_reporting_manifest_v22"
DESCRIPTOR_MANIFEST_SCHEMA = "soz_private_event_descriptors_target_blind_v24"
DESCRIPTOR_ROW_SCHEMA = "soz_private_event_descriptor_target_blind_v24"
OUTPUT_MANIFEST_SCHEMA = "trustworthy_soz_qualified_reporting_manifest_v24"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.resolve(strict=True).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _clause(kind: str, text: str, paths: Sequence[str]) -> dict[str, object]:
    return {"type": kind, "text": text, "fact_paths": list(paths)}


def _descriptor_clauses(descriptor: Mapping[str, object]) -> list[dict[str, object]]:
    change = descriptor.get("algorithmic_sustained_change")
    later = descriptor.get("later_scalp_visible_change_candidates")
    if not isinstance(change, Mapping) or not isinstance(later, list):
        raise TypeError("private descriptor is incomplete")
    clauses: list[dict[str, object]] = []
    if change.get("status") == "detected":
        interval = change.get("support_interval_sec_relative_to_clinical_event_anchor")
        edges = change.get("bipolar_derivation_candidates")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not isinstance(edges, list)
            or not edges
        ):
            raise ValueError("detected private descriptor lacks interval/edges")
        edge_text = "及".join(str(value).replace("-", "–") for value in edges)
        clauses.append(
            _clause(
                "event_algorithmic_sustained_change",
                f"相对临床事件锚点{float(interval[0]):.2f}–{float(interval[1]):.2f}秒，"
                f"冻结信号检测器在{edge_text}标记持续双极变化候选；"
                "该时段不是临床确认的最早物理电极、SOZ起点或皮层起始真值",
                [
                    "private_descriptor.algorithmic_sustained_change.support_interval_sec_relative_to_clinical_event_anchor",
                    "private_descriptor.algorithmic_sustained_change.bipolar_derivation_candidates",
                ],
            )
        )
    else:
        clauses.append(
            _clause(
                "event_algorithmic_change_abstention",
                "冻结信号检测器未标记满足固定条件的持续双极变化候选；这不表示发作不可见或不存在SOZ",
                ["private_descriptor.algorithmic_sustained_change.status"],
            )
        )
    if later:
        first = later[0]
        if not isinstance(first, Mapping):
            raise TypeError("later-visible descriptor is not an object")
        clauses.append(
            _clause(
                "later_visible_order",
                f"约{float(first['delay_sec']):.2f}秒后在{first['channel']}标记后续头皮可见变化候选；"
                "该先后次序不解释为传播路径、传播速度或SOZ起始顺序",
                ["private_descriptor.later_scalp_visible_change_candidates"],
            )
        )
    clauses.extend(
        [
            _clause(
                "rhythm_qualification",
                "当前节律/频段producer未通过独立原生precision资格门，因此不生成δ、θ、α、β或持续演变节律结论",
                ["private_descriptor.qualification.rhythm_or_frequency_phrase"],
            ),
            _clause(
                "montage_artifact_qualification",
                "private记录只有未证实原始参考下的CAR19结果，不能评价跨蒙太奇一致性；"
                "伪迹类型和严重度也未经独立医生资格化，相关结论保持缺席",
                [
                    "private_descriptor.qualification.montage_consistency",
                    "private_descriptor.qualification.artifact_type_or_severity",
                ],
            ),
        ]
    )
    return clauses


def _upgrade_private(
    report: Mapping[str, object], descriptor: Mapping[str, object]
) -> dict[str, object]:
    if report.get("schema_version") != INPUT_REPORT_SCHEMA:
        raise ValueError("private v22 report schema drifted")
    if report.get("unit_id") != descriptor.get("event_id") or report.get(
        "patient_id"
    ) != descriptor.get("patient_id"):
        raise ValueError("private report/descriptor identity mismatch")
    clauses = report.get("clauses")
    if not isinstance(clauses, list):
        raise TypeError("private v22 report has no clauses")
    replacement = _descriptor_clauses(descriptor)
    output_clauses: list[dict[str, object]] = []
    inserted = False
    for value in clauses:
        if not isinstance(value, Mapping):
            raise TypeError("private report clause is not an object")
        if value.get("type") == "event_evidence_unavailable":
            if inserted:
                raise ValueError("private report repeats event evidence placeholder")
            output_clauses.extend(replacement)
            inserted = True
        else:
            output_clauses.append(dict(value))
    if not inserted:
        raise ValueError("private report lacks replaceable event evidence placeholder")
    output = dict(report)
    output["schema_version"] = OUTPUT_REPORT_SCHEMA
    localization = report.get("localization")
    if not isinstance(localization, Mapping):
        raise TypeError("private v22 report lacks localization")
    action = str(localization.get("action", ""))
    if action not in {"display_candidate", "localization_abstain"}:
        raise ValueError(f"unsupported private localization action: {action!r}")
    output["report_status"] = (
        f"private_event_qualified_target_blind_descriptors_{action}_facts_locked"
    )
    output["clauses"] = output_clauses
    output["clinical_text_zh"] = "。".join(
        str(clause["text"]).rstrip("。") for clause in output_clauses
    ) + "。"
    output["sentence_fact_map"] = [
        {
            "sentence_index": index,
            "clause_type": clause["type"],
            "fact_paths": clause["fact_paths"],
        }
        for index, clause in enumerate(output_clauses)
    ]
    output["private_event_descriptor"] = dict(descriptor)
    return output


def _upgrade_public(report: Mapping[str, object]) -> dict[str, object]:
    if report.get("schema_version") != INPUT_REPORT_SCHEMA:
        raise ValueError("public v22 report schema drifted")
    output = dict(report)
    output["schema_version"] = OUTPUT_REPORT_SCHEMA
    return output


def materialize(
    source_directory: Path,
    descriptor_directory: Path,
    output_directory: Path,
) -> dict[str, object]:
    source = source_directory.resolve(strict=True)
    descriptors = descriptor_directory.resolve(strict=True)
    source_manifest = _json(source / "manifest.json")
    descriptor_manifest = _json(descriptors / "manifest.json")
    if source_manifest.get("schema_version") != INPUT_MANIFEST_SCHEMA:
        raise ValueError("v22 report manifest schema drifted")
    if descriptor_manifest.get("schema_version") != DESCRIPTOR_MANIFEST_SCHEMA:
        raise ValueError("private descriptor manifest schema drifted")
    access = descriptor_manifest.get("access_receipt")
    if not isinstance(access, Mapping) or any(
        access.get(field) is not False
        for field in (
            "private_soz_targets_loaded",
            "deepsoz_targets_loaded",
            "model_predictions_loaded",
            "training_or_threshold_selection_performed",
        )
    ):
        raise ValueError("private descriptors do not prove target-blind lineage")
    descriptor_rows = _jsonl(descriptors / str(descriptor_manifest["descriptor_file"]))
    descriptor_by_id: dict[str, dict[str, object]] = {}
    for row in descriptor_rows:
        if row.get("schema_version") != DESCRIPTOR_ROW_SCHEMA:
            raise ValueError("private descriptor row schema drifted")
        event_id = str(row.get("event_id", ""))
        if not event_id or event_id in descriptor_by_id:
            raise ValueError("private descriptor ID is empty or repeated")
        descriptor_by_id[event_id] = row
    public_patient = [
        _upgrade_public(row) for row in _jsonl(source / "public_patient_reports.jsonl")
    ]
    public_event = [
        _upgrade_public(row) for row in _jsonl(source / "public_event_reports.jsonl")
    ]
    private_source = _jsonl(source / "private_event_reports.jsonl")
    if {str(row["unit_id"]) for row in private_source} != set(descriptor_by_id):
        raise ValueError("private v22 reports and descriptors have different rosters")
    private_event = [
        _upgrade_private(row, descriptor_by_id[str(row["unit_id"])])
        for row in private_source
    ]
    counts: Counter[str] = Counter()
    for row in private_event:
        localization = row.get("localization")
        if not isinstance(localization, Mapping):
            raise TypeError("private v24 report lacks localization")
        counts[str(localization["action"])] += 1
        clause_types = {str(clause["type"]) for clause in row["clauses"]}
        counts["private_with_algorithmic_descriptor"] += int(
            bool(
                clause_types
                & {"event_algorithmic_sustained_change", "event_algorithmic_change_abstention"}
            )
        )
        counts["private_with_later_visible_descriptor"] += int(
            "later_visible_order" in clause_types
        )
    target = output_directory.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        _write_jsonl(staging / "public_patient_reports.jsonl", public_patient)
        _write_jsonl(staging / "public_event_reports.jsonl", public_event)
        _write_jsonl(staging / "private_event_reports.jsonl", private_event)
        output_manifest: dict[str, object] = {
            "schema_version": OUTPUT_MANIFEST_SCHEMA,
            "status": "completed_target_blind_private_descriptor_enrichment_v24",
            "counts": {
                "public_patient_reports": len(public_patient),
                "public_event_reports": len(public_event),
                "private_event_reports": len(private_event),
                **dict(sorted(counts.items())),
            },
            "files": {
                "public_patient_reports": "public_patient_reports.jsonl",
                "public_event_reports": "public_event_reports.jsonl",
                "private_event_reports": "private_event_reports.jsonl",
            },
            "access_receipt": {
                "raw_eeg_loaded": False,
                "private_soz_targets_loaded": False,
                "deepsoz_targets_loaded": False,
                "evaluation_rows_loaded": False,
                "scores_thresholds_or_candidate_decisions_changed": False,
                "llm_used": False,
            },
            "claim_boundary": {
                "algorithmic_change_is_clinical_onset": False,
                "later_visible_is_propagation": False,
                "rhythm_artifact_or_montage_claim_emitted": False,
                "report_is_confirmed_diagnosis": False,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(output_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return output_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--source-directory", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--descriptor-directory", type=Path, default=DEFAULT_DESCRIPTORS)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = materialize(args.source_directory, args.descriptor_directory, args.output_directory)
    print(json.dumps({"output": str(args.output_directory), **result["counts"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
