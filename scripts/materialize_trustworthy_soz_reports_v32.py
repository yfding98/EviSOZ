#!/usr/bin/env python3
"""Create clinician-facing v32 reports with concise reference opinions.

The command preserves every localization score, candidate rank, abstention,
and fact path from v24.1.  It replaces engineering-heavy prose with a concise
clinical summary and attaches target-blind processed-waveform metadata.  Model
implementation names remain available in the technical audit layer, not in
the clinician-facing paragraph.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCHEMA = "trustworthy_soz_qualified_report_v24"
OUTPUT_SCHEMA = "trustworthy_soz_clinical_reference_report_v32"
MANIFEST_SCHEMA = "trustworthy_soz_clinical_reference_reporting_manifest_v32"
WAVEFORM_SCHEMA = "trustworthy_soz_processed_waveform_figures_v32"
DEFAULT_SOURCE = ROOT / "outputs/trustworthy_soz_qualified_reports_v24_1_20260815"
DEFAULT_WAVEFORMS = ROOT / "outputs/trustworthy_soz_processed_waveforms_v32_20260816"
DEFAULT_PUBLIC_SOURCE = ROOT / "outputs/target_free_oof_reports_v3_recovered_20260813.json"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_clinical_reports_v32_20260816"

FORBIDDEN_CLINICAL_WORDING = (
    "H-only",
    "冻结",
    "concept",
    "producer",
    "precision",
    "private",
    "C18",
    "SOZ-reference",
    "资格门",
    "相关结论保持缺席",
)


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


def _join_zh(values: Sequence[str], *, limit: int | None = None) -> str:
    items = [str(value).replace("-", "–") for value in values]
    total = len(items)
    if limit is not None and total > limit:
        items = items[:limit]
    if not items:
        return ""
    if len(items) == 1:
        text = items[0]
    else:
        text = "、".join(items[:-1]) + "及" + items[-1]
    return f"{text}等{total}组导联组合" if limit is not None and total > limit else text


def _time_interval_phrase(interval: Sequence[object]) -> str:
    if len(interval) != 2:
        raise ValueError("time interval must have two endpoints")
    start, stop = float(interval[0]), float(interval[1])
    if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
        raise ValueError("time interval is invalid")
    if start >= 0:
        return f"事件标记后{start:.2f}–{stop:.2f}秒"
    if stop <= 0:
        return f"事件标记前{abs(start):.2f}–{abs(stop):.2f}秒"
    return f"相对事件标记{start:.2f}至{stop:.2f}秒"


def _clause(kind: str, text: str, paths: Sequence[str]) -> dict[str, object]:
    return {"type": kind, "text": text.rstrip("。"), "fact_paths": list(paths)}


def _candidate_language(localization: Mapping[str, object]) -> tuple[str, str]:
    action = localization.get("action")
    displayed = localization.get("displayed_candidates")
    if not isinstance(displayed, list):
        raise TypeError("displayed candidates must be a list")
    if action == "display_candidate":
        channels = [str(item["channel"]) for item in displayed if isinstance(item, Mapping)]
        if not channels:
            raise ValueError("candidate report has no displayed channels")
        region = str(localization.get("top1_region_projection_zh") or "对应头皮区域")
        result = f"模型候选排序为{_join_zh(channels)}，首位{channels[0]}对应{region}"
        opinion = (
            f"参考意见：建议优先复核{region}及{channels[0]}相关导联的发作期变化，"
            "并结合完整发作期脑电、症状学、影像学及必要时颅内脑电综合定位"
        )
        return result, opinion
    if action == "localization_abstain":
        return (
            "模型候选之间的区分度不足，本次不提供单一电极优先级",
            "参考意见：本次定位依据有限，不建议依据模型结果缩小SOZ范围；应结合完整发作期脑电、症状学和影像学重新评估",
        )
    if action == "localization_unavailable":
        return (
            "当前记录未形成可用的电极候选排序",
            "参考意见：当前信息不足以提供SOZ电极优先级，建议依据完整临床资料重新评估",
        )
    raise ValueError(f"unsupported localization action: {action!r}")


def _private_observation(report: Mapping[str, object]) -> str:
    descriptor = report.get("private_event_descriptor")
    if not isinstance(descriptor, Mapping):
        return "本次事件未形成满足预设条件的持续波形变化区间"
    change = descriptor.get("algorithmic_sustained_change")
    later = descriptor.get("later_scalp_visible_change_candidates")
    if not isinstance(change, Mapping) or not isinstance(later, list):
        raise TypeError("private descriptor is incomplete")
    interval = change.get("support_interval_sec_relative_to_clinical_event_anchor")
    edges = change.get("bipolar_derivation_candidates")
    if change.get("status") != "detected" or not isinstance(interval, list) or not isinstance(edges, list) or not edges:
        return "本次事件未形成满足预设条件的持续波形变化区间"
    text = f"{_time_interval_phrase(interval)}，{_join_zh([str(edge) for edge in edges], limit=6)}出现算法标记的持续波形变化"
    if later and isinstance(later[0], Mapping):
        text += (
            f"；自上述首段变化开始约{float(later[0]['delay_sec']):.2f}秒后，"
            f"{later[0]['channel']}出现后续可见变化"
        )
    return text


def _public_observation(
    source_record: Mapping[str, object] | None,
    waveform: Mapping[str, object] | None,
    *,
    patient_level: bool,
) -> str:
    if patient_level and waveform is None:
        return (
            "当前未找到可与本报告可靠绑定的代表性事件波形；"
            "患者级定位结果仍由全部符合质量要求的发作综合获得"
        )
    prefix = "图示代表性事件" if patient_level else "本次事件"
    if source_record is None:
        return f"{prefix}未形成可用于正文描述的持续波形变化区间"
    facts = source_record.get("typed_facts")
    phenotype = facts.get("event_phenotype") if isinstance(facts, Mapping) else None
    if not isinstance(phenotype, Mapping):
        return f"{prefix}未形成可用于正文描述的持续波形变化区间"
    derivations = phenotype.get("first_visible_derivations")
    interval = waveform.get("evidence_interval_sec") if isinstance(waveform, Mapping) else None
    if not isinstance(derivations, list) or not derivations or not isinstance(interval, list):
        return f"{prefix}未形成可用于正文描述的持续波形变化区间"
    text = f"{prefix}在{_time_interval_phrase(interval)}，{_join_zh([str(value) for value in derivations], limit=6)}出现算法标记的持续波形变化"
    delay = phenotype.get("later_visible_delay_sec")
    region = phenotype.get("later_visible_region_zh")
    if isinstance(delay, (int, float)) and isinstance(region, str) and region:
        text += (
            f"；自上述首段变化开始约{float(delay):.2f}秒后，"
            f"{region}出现后续可见变化"
        )
    if patient_level:
        text += "。患者级候选由全部合格发作综合获得，图示事件仅用于波形复核"
    return text


def _scope_text(*, patient_level: bool) -> str:
    if patient_level:
        return "本次分析综合患者全部符合质量要求的发作期标准19导头皮脑电；信号经0.5–45 Hz滤波、重采样至200 Hz并采用共平均参考"
    return "本次分析采用标准19导头皮脑电；信号经0.5–45 Hz滤波、重采样至200 Hz并采用共平均参考，分析范围为事件标记前12秒至后48秒"


def _upgrade(
    report: Mapping[str, object],
    *,
    waveform: Mapping[str, object] | None,
    public_source_record: Mapping[str, object] | None,
) -> dict[str, object]:
    if report.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("source report schema drifted")
    localization = report.get("localization")
    if not isinstance(localization, Mapping):
        raise TypeError("source report lacks localization")
    cohort = str(report.get("cohort", ""))
    patient_level = cohort == "public_deepsoz_development_patient_report"
    is_private = cohort == "private_post_open_target_blind_event_report"
    scope = _scope_text(patient_level=patient_level)
    if is_private:
        observation = _private_observation(report)
    else:
        observation = _public_observation(public_source_record, waveform, patient_level=patient_level)
    localization_text, opinion = _candidate_language(localization)
    applicability = (
        "波形形态、节律特征、伪迹影响和空间扩展顺序已纳入证据可用性评估；"
        "因当前缺少独立临床验证，具体分类仅供复核，不作为确定性定位依据"
    )
    boundary = (
        "现有结果反映头皮电极层面的SOZ候选，不等同于皮层SOZ、致痫区或手术靶点"
    )
    clauses = [
        _clause("analysis_scope", scope, ["waveform.preprocessing", "source_report.unit_id"]),
        _clause(
            "waveform_observation",
            observation,
            [
                "waveform.evidence_interval_sec",
                "source_signal_descriptor.sustained_change",
                "source_signal_descriptor.later_visible_change",
            ],
        ),
        _clause(
            "localization_result",
            localization_text,
            ["source_report.localization.displayed_candidates", "source_report.localization.top1_region_projection_zh"],
        ),
        _clause("reference_opinion", opinion, ["clinical_language_policy.reference_opinion"]),
        _clause("evidence_applicability", applicability, ["source_report.concept_qualification", "clinical_language_policy.evidence_applicability"]),
        _clause("clinical_boundary", boundary, ["source_report.claim_boundary"]),
    ]
    summary = "。".join((scope, observation, localization_text)) + "。"
    clinical_surface = "。".join((summary, opinion, applicability, boundary))
    hits = [word for word in FORBIDDEN_CLINICAL_WORDING if word in clinical_surface]
    if hits:
        raise ValueError(f"clinical summary contains engineering wording: {hits}")
    output = dict(report)
    output.update({
        "schema_version": OUTPUT_SCHEMA,
        "report_status": f"clinical_reference_language_v32_{localization.get('action')}",
        "clinical_text_zh": summary,
        "clinical_summary_zh": summary,
        "reference_opinion_zh": opinion + "。",
        "evidence_applicability_zh": applicability + "。",
        "clinical_boundary_zh": boundary + "。",
        "clauses": clauses,
        "sentence_fact_map": [
            {"sentence_index": index, "clause_type": clause["type"], "fact_paths": clause["fact_paths"]}
            for index, clause in enumerate(clauses)
        ],
        "waveform_figure": dict(waveform) if waveform is not None else None,
        "technical_audit": {
            "source_report_schema": SOURCE_SCHEMA,
            "localization_profile": "public_development_frozen_h_only_v21",
            "candidate_scores_or_abstention_changed": False,
            "clinical_language_changed_only": True,
            "legacy_clauses": report.get("clauses"),
        },
        "facts_locked": True,
        "llm_used": False,
    })
    return output


def materialize(args: argparse.Namespace) -> dict[str, object]:
    source = args.source.resolve(strict=True)
    waveform_manifest = _json(args.waveforms / "manifest.json")
    if waveform_manifest.get("schema_version") != WAVEFORM_SCHEMA:
        raise ValueError("waveform manifest schema drifted")
    waveform_entries = waveform_manifest.get("entries")
    if not isinstance(waveform_entries, list):
        raise TypeError("waveform manifest has no entries")
    waveforms = {
        (str(entry["scope"]), str(entry["unit_id"])): entry
        for entry in waveform_entries
        if isinstance(entry, dict)
    }
    public_source = _json(args.public_source)
    public_records = public_source.get("records")
    if not isinstance(public_records, list):
        raise TypeError("public source has no records")
    public_by_event = {
        str(record["event_id"]): record for record in public_records if isinstance(record, dict)
    }

    output_rows: dict[str, list[dict[str, object]]] = {}
    specifications = (
        ("public_patient_reports.jsonl", "public_patient"),
        ("public_event_reports.jsonl", "public_event"),
        ("private_event_reports.jsonl", "private_event"),
    )
    counts: Counter[str] = Counter()
    for filename, scope in specifications:
        upgraded: list[dict[str, object]] = []
        for report in _jsonl(source / filename):
            unit_id = str(report["unit_id"])
            waveform = waveforms.get((scope, unit_id))
            public_record = None
            if scope == "public_event":
                public_record = public_by_event.get(unit_id)
            elif scope == "public_patient" and waveform is not None:
                public_record = public_by_event.get(str(waveform["event_id"]))
            row = _upgrade(report, waveform=waveform, public_source_record=public_record)
            upgraded.append(row)
            counts[f"{scope}_reports"] += 1
            counts[str(row["localization"]["action"])] += 1
            counts[f"{scope}_with_waveform"] += int(waveform is not None)
        output_rows[filename] = upgraded

    target = args.output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        for filename, rows in output_rows.items():
            _write_jsonl(staging / filename, rows)
        manifest: dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "completed_clinical_reference_language_and_waveform_binding_v32",
            "counts": dict(sorted(counts.items())),
            "files": {filename.removesuffix(".jsonl"): filename for filename in output_rows},
            "waveform_source": str(args.waveforms),
            "clinical_language_policy": {
                "summary_structure": ["analysis_scope", "waveform_observation", "localization_result"],
                "reference_opinion_is_a_separate_clinical_section": True,
                "engineering_terms_kept_out_of_clinical_summary": list(FORBIDDEN_CLINICAL_WORDING),
                "insufficiently_validated_features_are_reference_only": True,
            },
            "access_receipt": {
                "raw_eeg_loaded": False,
                "private_soz_targets_loaded": False,
                "deepsoz_targets_loaded": False,
                "evaluation_rows_loaded": False,
                "candidate_scores_thresholds_or_decisions_changed": False,
                "clinical_language_changed": True,
                "llm_used": False,
            },
            "claim_boundary": {
                "waveform_is_clinician_interpretation": False,
                "reference_opinion_is_confirmed_diagnosis": False,
                "report_is_cortical_soz_or_surgical_target": False,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--waveforms", type=Path, default=DEFAULT_WAVEFORMS)
    parser.add_argument("--public-source", type=Path, default=DEFAULT_PUBLIC_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = materialize(args)
    print(json.dumps({"output": str(args.output), **result["counts"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
