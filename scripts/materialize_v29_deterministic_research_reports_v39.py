#!/usr/bin/env python3
"""Materialize v29-bound deterministic private research reports.

All candidate facts come from the frozen target-blind v29 tensor.  Signal
observations come only from the existing target-blind private descriptors.
The generator has no target-ledger input.  The wording was authored after the
private cohort had historically been opened, so these reports are research
communication artifacts rather than target-blind clinical validation.
"""

from __future__ import annotations

import argparse
import csv
from html import escape
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.clinical_reporting import (  # noqa: E402
    CLINICAL_SCALP_REGIONS,
    LATERALITY_GROUPS,
)
from src.soz.geometry import STANDARD_19  # noqa: E402


SCHEMA = "trustworthy_soz_v29_deterministic_research_report_v39"
DEFAULT_PREDICTIONS = (
    ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
)
DEFAULT_DESCRIPTORS = (
    ROOT / "outputs/trustworthy_soz_qualified_reports_v24_20260815/private_event_reports.jsonl"
)
DEFAULT_WAVEFORMS = (
    ROOT
    / "outputs/trustworthy_soz_clinician_html_v34_20260816/waveforms/private_event"
)
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_v29_research_reports_v39_20260816"


REGION_ZH = {
    "frontal": "额区",
    "temporal": "颞区",
    "central": "中央区",
    "parietal": "顶区",
    "occipital": "枕区",
}
LATERALITY_ZH = {"left": "左", "right": "右", "midline": "中线"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.resolve(strict=True).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _membership(channel: str, groups: Mapping[str, Sequence[str]]) -> str:
    matches = [name for name, channels in groups.items() if channel in channels]
    if len(matches) != 1:
        raise ValueError(f"{channel} has invalid group membership")
    return matches[0]


def _location(channel: str) -> tuple[str, str, str]:
    region = _membership(channel, CLINICAL_SCALP_REGIONS)
    laterality = _membership(channel, LATERALITY_GROUPS)
    return region, laterality, LATERALITY_ZH[laterality] + REGION_ZH[region]


def _descriptor_text(descriptor: Mapping[str, object]) -> tuple[list[str], list[str]]:
    observations: list[str] = []
    fact_ids: list[str] = []
    sustained = descriptor.get("algorithmic_sustained_change")
    if isinstance(sustained, Mapping) and sustained.get("status") == "detected":
        interval = sustained.get("support_interval_sec_relative_to_clinical_event_anchor")
        derivations = sustained.get("bipolar_derivation_candidates")
        if (
            isinstance(interval, list)
            and len(interval) == 2
            and all(isinstance(value, (int, float)) for value in interval)
            and isinstance(derivations, list)
            and derivations
        ):
            start, end = (float(interval[0]), float(interval[1]))
            names = "、".join(str(value) for value in derivations[:5])
            observations.append(
                f"相对事件标记约{start:.2f}–{end:.2f}秒，自动波形检索在{names}观察到持续变化候选。"
            )
            fact_ids.extend(
                [
                    "descriptor.sustained_change.interval",
                    "descriptor.sustained_change.derivations",
                ]
            )
    later = descriptor.get("later_scalp_visible_change_candidates")
    if isinstance(later, list) and later:
        phrases = []
        for item in later[:3]:
            if not isinstance(item, Mapping):
                continue
            channel = item.get("channel")
            delay = item.get("delay_sec")
            if channel in STANDARD_19 and isinstance(delay, (int, float)):
                phrases.append(f"{channel}（约{float(delay):.2f}秒后）")
        if phrases:
            observations.append("随后可见变化候选包括" + "、".join(phrases) + "。")
            fact_ids.append("descriptor.later_visible_candidates")
    if not observations:
        observations.append("当前自动波形检索未形成可稳定展示的时间变化摘要，建议直接复核原始波形。")
        fact_ids.append("descriptor.no_stable_observation")
    observations.append(
        "上述时间标记用于定位复核片段，不代表皮层起始时刻、SOZ先后顺序或传播方向。"
    )
    fact_ids.append("protocol.temporal_observation_boundary")
    return observations, fact_ids


def _format_score(value: float) -> str:
    return f"{value:.3f}"


def _build_report(
    event: Mapping[str, object],
    probability: torch.Tensor,
    fold_probability: torch.Tensor,
    candidate_mask: torch.Tensor,
    descriptor: Mapping[str, object],
    waveform_name: str,
) -> tuple[dict[str, object], dict[str, object]]:
    event_id = str(event["event_id"])
    patient_id = str(event["patient_id"])
    if tuple(probability.shape) != (19,) or tuple(fold_probability.shape) != (5, 19):
        raise ValueError(f"invalid prediction shape for {event_id}")
    if not torch.isfinite(probability).all() or not torch.isfinite(fold_probability).all():
        raise ValueError(f"non-finite prediction for {event_id}")
    if not math.isclose(float(probability[candidate_mask].sum()), 1.0, abs_tol=1e-5):
        raise ValueError(f"candidate probability does not sum to one for {event_id}")
    ranking = torch.topk(
        probability.masked_fill(~candidate_mask, -torch.inf), k=5
    ).indices.tolist()
    candidates = [
        {
            "rank": rank,
            "channel": STANDARD_19[index],
            "relative_support": float(probability[index]),
        }
        for rank, index in enumerate(ranking, start=1)
    ]
    top1 = candidates[0]["channel"]
    region, laterality, location_zh = _location(top1)
    ordered_values = torch.topk(
        probability.masked_fill(~candidate_mask, -torch.inf), k=2
    ).values
    margin = float(ordered_values[0] - ordered_values[1])
    fold_top1 = torch.argmax(
        fold_probability.masked_fill(~candidate_mask.unsqueeze(0), -torch.inf), dim=1
    )
    top1_index = STANDARD_19.index(top1)
    fold_agreement_count = int((fold_top1 == top1_index).sum())
    observations, observation_fact_ids = _descriptor_text(descriptor)
    candidate_names = "、".join(str(item["channel"]) for item in candidates)
    source_sfreq = float(event["source_sfreq_hz"])

    summary = (
        f"本次标准19导头皮脑电的SOZ-reference候选依次为{candidate_names}。"
        f"首位候选{top1}位于{location_zh}。建议优先复核{top1}及相邻导联的发作期波形，"
        "并结合症状学、影像和既往检查综合判断。"
    )
    interpretation = (
        f"模型在5个患者外训练的子模型中有{fold_agreement_count}/5个将{top1}列为首位；"
        f"首位与第二位的相对支持度差为{margin:.3f}。这些数值反映模型内部排序及折间一致性，"
        "不是候选正确率或临床置信概率。"
    )
    evidence_limit = (
        "当前形态学、发作受累和学习型时间演变分支未达到预设的独立资格要求，未参与候选排序。"
        "现有时间变化信息仅作为波形复核线索；节律类型、起始模式及传播方向证据不足时不作确定判断。"
    )
    clinical_boundary = (
        "本报告输出的是标准头皮电极上的SOZ-reference研究候选，不等同于侵入式皮层SOZ、"
        "致痫区或治疗靶点，不能单独用于手术决策。"
    )

    report = {
        "schema_version": SCHEMA,
        "cohort": "private_target_blind_prediction_post_open_research_communication",
        "event_id": event_id,
        "patient_id": patient_id,
        "title_zh": "头皮脑电SOZ-reference候选研究报告",
        "analysis_scope": {
            "source_sampling_rate_hz": source_sfreq,
            "processed_sampling_rate_hz": 200,
            "channel_system": "standard_19",
            "analysis_reference": "CAR19",
            "event_window_sec": [-12.0, 48.0],
        },
        "localization": {
            "action": "display_uncalibrated_research_candidate_set",
            "candidates": candidates,
            "top1_region": region,
            "top1_laterality": laterality,
            "top1_location_zh": location_zh,
            "top1_top2_margin": margin,
            "fold_top1_agreement_count": fold_agreement_count,
            "fold_count": 5,
            "score_semantics": "within_report_relative_support_not_correctness_probability",
            "clinical_risk_guarantee": False,
        },
        "waveform": {
            "image": waveform_name,
            "observations_zh": observations,
            "model_aligned_causal_evidence": False,
        },
        "clinical_summary_zh": summary,
        "model_uncertainty_zh": interpretation,
        "evidence_qualification_zh": evidence_limit,
        "clinical_boundary_zh": clinical_boundary,
        "system_profile": {
            "ranker": "v29_equal_H_D_probability_ensemble",
            "foundation": "official_pretrained_LaBraM_block9_frozen",
            "foundation_trainable_parameters": 0,
            "concept_states": {
                "morphology": "FAIL_NATIVE_STRUCTURALLY_ABSENT",
                "ictal_involvement": "FAIL_NATIVE_STRUCTURALLY_ABSENT",
                "direct_temporal_observation": "DESCRIPTION_ONLY",
                "learned_temporal_future": "NO_GO_STRUCTURALLY_ABSENT",
            },
        },
        "release_boundary": {
            "private_target_read_by_generator": False,
            "target_or_correctness_used_for_candidate_or_text": False,
            "report_wording_authored_after_historical_private_open": True,
            "target_blind_clinical_validation": False,
            "llm_used": False,
            "facts_locked": True,
        },
    }
    audit = {
        "event_id": event_id,
        "patient_id": patient_id,
        "candidate_fact_ids": [
            "v29.private_portable_equal_probability",
            "v29.private_portable_equal_fold_probability",
            "v29.candidate_mask",
            "v29.manifest.events",
        ],
        "observation_fact_ids": observation_fact_ids,
        "visible_in_clinician_html": False,
        "gold_or_outcome_fact_ids": [],
        "candidate_sha256": hashlib.sha256(
            json.dumps(candidates, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    return report, audit


def _render_html(report: Mapping[str, object]) -> str:
    localization = report["localization"]
    waveform = report["waveform"]
    scope = report["analysis_scope"]
    assert isinstance(localization, Mapping) and isinstance(waveform, Mapping)
    assert isinstance(scope, Mapping)
    candidate_rows = "".join(
        "<tr>"
        f"<td>{int(item['rank'])}</td>"
        f"<td><strong>{escape(str(item['channel']))}</strong></td>"
        f"<td>{_format_score(float(item['relative_support']))}</td>"
        "</tr>"
        for item in localization["candidates"]
    )
    observation_items = "".join(
        f"<li>{escape(str(value))}</li>" for value in waveform["observations_zh"]
    )
    event_id = escape(str(report["event_id"]))
    patient_id = escape(str(report["patient_id"]))
    image = escape(str(waveform["image"]))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(str(report['title_zh']))} · {event_id}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans CJK SC",sans-serif;background:#f4f7f8;color:#203038;margin:0;line-height:1.65}}
main{{max-width:1060px;margin:24px auto;padding:0 18px 40px}} .card{{background:white;border:1px solid #dce5e8;border-radius:12px;padding:20px 24px;margin:14px 0;box-shadow:0 2px 10px #2030380a}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:17px;color:#245b66;margin:0 0 10px}} .meta{{color:#60747b;font-size:13px}} .notice{{border-left:4px solid #c8872b;background:#fff8e9}}
.grid{{display:grid;grid-template-columns:minmax(280px,.75fr) minmax(0,1.6fr);gap:18px}} table{{border-collapse:collapse;width:100%}} th,td{{padding:8px 10px;border-bottom:1px solid #e5ecee;text-align:left}} th{{color:#536970;font-size:13px}}
img{{width:100%;height:auto;border:1px solid #d8e2e5;border-radius:8px;background:white}} ul{{margin:6px 0;padding-left:22px}} .small{{font-size:13px;color:#60747b}} .candidate{{font-size:18px;color:#124d5a}}
@media(max-width:780px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="card"><h1>{escape(str(report['title_zh']))}</h1><div class="meta">研究用途 · 事件 {event_id} · 患者编码 {patient_id}</div></div>
<div class="card notice"><h2>定位参考意见</h2><p class="candidate">{escape(str(report['clinical_summary_zh']))}</p></div>
<div class="grid"><section class="card"><h2>候选电极排序</h2><table><thead><tr><th>顺位</th><th>电极</th><th>相对支持度</th></tr></thead><tbody>{candidate_rows}</tbody></table><p class="small">相对支持度仅用于本次报告内排序，不是候选正确概率。</p></section>
<section class="card"><h2>处理后波形</h2><img src="{image}" alt="事件{event_id}处理后标准19导波形"></section></div>
<section class="card"><h2>波形复核线索</h2><ul>{observation_items}</ul></section>
<section class="card"><h2>模型一致性与证据范围</h2><p>{escape(str(report['model_uncertainty_zh']))}</p><p>{escape(str(report['evidence_qualification_zh']))}</p><p class="small">输入：标准19导，{float(scope['source_sampling_rate_hz']):g} Hz原始采样；处理：200 Hz、CAR19、相对事件标记[-12,+48)秒。</p></section>
<section class="card notice"><h2>使用边界</h2><p>{escape(str(report['clinical_boundary_zh']))}</p></section>
</main></body></html>"""


def _render_index(reports: Sequence[Mapping[str, object]]) -> str:
    rows = "".join(
        "<tr>"
        f"<td><a href=\"private_event/{escape(str(row['event_id']))}.html\">{escape(str(row['event_id']))}</a></td>"
        f"<td>{escape(str(row['patient_id']))}</td>"
        f"<td>{escape(str(row['localization']['candidates'][0]['channel']))}</td>"
        f"<td>{escape(str(row['localization']['top1_location_zh']))}</td>"
        "</tr>"
        for row in reports
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>v29 private研究报告索引</title>
<style>body{{font-family:sans-serif;max-width:960px;margin:28px auto;padding:0 18px;color:#203038}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #dde6e8;text-align:left}}a{{color:#176779}}.note{{background:#fff8e9;border-left:4px solid #c8872b;padding:12px}}</style></head><body>
<h1>v29 private研究报告索引</h1><p class="note">全部88个target-blind可预测事件一次性生成；该索引不含reference命中或病例结果分层。</p>
<table><thead><tr><th>事件</th><th>患者</th><th>首位候选</th><th>头皮区域</th></tr></thead><tbody>{rows}</tbody></table></body></html>"""


def run(
    prediction_directory: Path,
    descriptor_path: Path,
    waveform_directory: Path,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], dict[str, Path]]:
    prediction_manifest_path = (prediction_directory / "manifest.json").resolve(strict=True)
    prediction_tensor_path = (prediction_directory / "predictions.safetensors").resolve(strict=True)
    descriptor_path = descriptor_path.resolve(strict=True)
    prediction_manifest = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
    predictions = load_file(str(prediction_tensor_path), device="cpu")
    events = prediction_manifest["events"]
    probability = predictions["private_portable_equal_probability"].float()
    fold_probability = predictions["private_portable_equal_fold_probability"].float()
    candidate_mask = predictions["candidate_mask"].bool()
    if len(events) != 88 or tuple(probability.shape) != (88, 19) or tuple(
        fold_probability.shape
    ) != (88, 5, 19):
        raise ValueError("v39 requires the frozen 88-event v29 carrier")
    if tuple(prediction_manifest["channels"]) != tuple(STANDARD_19):
        raise ValueError("v29 channel order drifted")

    descriptor_rows = _read_jsonl(descriptor_path)
    by_event: dict[str, Mapping[str, object]] = {}
    for row in descriptor_rows:
        event_id = str(row["unit_id"])
        descriptor = row.get("private_event_descriptor")
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"missing target-blind descriptor for {event_id}")
        lineage = descriptor.get("lineage")
        if not isinstance(lineage, Mapping) or lineage.get("private_soz_target_used") is not False:
            raise ValueError(f"descriptor target boundary is invalid for {event_id}")
        by_event[event_id] = descriptor
    if set(by_event) != {str(event["event_id"]) for event in events}:
        raise ValueError("descriptor and v29 event rosters differ")

    reports: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    waveforms: dict[str, Path] = {}
    for index, event in enumerate(events):
        event_id = str(event["event_id"])
        waveform = (waveform_directory / f"{event_id}.png").resolve(strict=True)
        report, audit = _build_report(
            event,
            probability[index],
            fold_probability[index],
            candidate_mask,
            by_event[event_id],
            f"../../waveforms/private_event/{event_id}.png",
        )
        reports.append(report)
        audits.append(audit)
        waveforms[event_id] = waveform

    manifest = {
        "schema_version": "trustworthy_soz_v29_deterministic_research_report_manifest_v39",
        "status": "completed_post_open_v29_bound_research_communication",
        "report_count": len(reports),
        "patient_count": len({str(row["patient_id"]) for row in reports}),
        "candidate_profile": "v29_equal_H_D_probability_ensemble",
        "all_target_blind_prediction_events_materialized": True,
        "case_selection_performed": False,
        "visible_fact_paths": False,
        "machine_audit_file_separate": True,
        "source_files": {
            "prediction_manifest": str(prediction_manifest_path.relative_to(ROOT)),
            "prediction_manifest_sha256": _sha256(prediction_manifest_path),
            "prediction_tensor": str(prediction_tensor_path.relative_to(ROOT)),
            "prediction_tensor_sha256": _sha256(prediction_tensor_path),
            "descriptor_jsonl": str(descriptor_path.relative_to(ROOT)),
            "descriptor_jsonl_sha256": _sha256(descriptor_path),
            "waveform_directory": str(waveform_directory.resolve(strict=True).relative_to(ROOT)),
        },
        "access_receipt": {
            "private_target_or_error_audit_loaded": False,
            "target_path_argument_available": False,
            "v29_target_blind_prediction_loaded": True,
            "target_blind_descriptors_loaded": True,
            "model_training_or_threshold_selection_performed": False,
            "llm_used": False,
        },
        "claim_boundary": {
            "wording_authored_after_historical_private_open": True,
            "target_blind_clinical_validation": False,
            "candidate_scores_calibrated_correctness_probabilities": False,
            "clinical_abstention_or_risk_guarantee": False,
            "waveform_observations_causally_explain_v29": False,
            "output_is_cortical_soz_ez_or_treatment_target": False,
        },
    }
    return manifest, reports, audits, waveforms


def publish(
    output: Path,
    manifest: Mapping[str, object],
    reports: Sequence[Mapping[str, object]],
    audits: Sequence[Mapping[str, object]],
    waveforms: Mapping[str, Path],
) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        html_directory = staging / "html/private_event"
        waveform_directory = staging / "waveforms/private_event"
        html_directory.mkdir(parents=True)
        waveform_directory.mkdir(parents=True)
        _write_jsonl(staging / "private_event_reports.jsonl", reports)
        _write_jsonl(staging / "machine_audit_fact_ids.jsonl", audits)
        with (staging / "candidate_table.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "event_id",
                    "patient_id",
                    "top1",
                    "top1_location_zh",
                    "top1_top2_margin",
                    "fold_top1_agreement_count",
                    "top5",
                ),
            )
            writer.writeheader()
            for report in reports:
                localization = report["localization"]
                writer.writerow(
                    {
                        "event_id": report["event_id"],
                        "patient_id": report["patient_id"],
                        "top1": localization["candidates"][0]["channel"],
                        "top1_location_zh": localization["top1_location_zh"],
                        "top1_top2_margin": localization["top1_top2_margin"],
                        "fold_top1_agreement_count": localization[
                            "fold_top1_agreement_count"
                        ],
                        "top5": "|".join(
                            item["channel"] for item in localization["candidates"]
                        ),
                    }
                )
        for report in reports:
            event_id = str(report["event_id"])
            (html_directory / f"{event_id}.html").write_text(
                _render_html(report), encoding="utf-8"
            )
            shutil.copy2(waveforms[event_id], waveform_directory / f"{event_id}.png")
        (staging / "html/index.html").write_text(
            _render_index(reports), encoding="utf-8"
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--descriptors", type=Path, default=DEFAULT_DESCRIPTORS)
    parser.add_argument("--waveforms", type=Path, default=DEFAULT_WAVEFORMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, reports, audits, waveforms = run(
        args.predictions, args.descriptors, args.waveforms
    )
    output = publish(args.output, manifest, reports, audits, waveforms)
    print(
        json.dumps(
            {
                "output": str(output),
                "reports": len(reports),
                "patients": manifest["patient_count"],
                "target_loaded": False,
                "llm_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
