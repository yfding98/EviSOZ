#!/usr/bin/env python3
"""Build a deterministic four-stratum private case audit for the paper.

This step intentionally reads the already opened private reference/error
audit.  It never changes a prediction or report.  Within each prespecified
outcome stratum, the event closest to the stratum-median v29 margin is chosen,
with event ID as the deterministic tie breaker.  The resulting package is
reviewer-facing failure analysis, not a target-blind case sample.
"""

from __future__ import annotations

import argparse
import ast
import csv
from html import escape
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ERRORS = (
    ROOT
    / "outputs/trustworthy_soz_private_frozen_publication_v36_20260816/private_event_error_audit.csv"
)
DEFAULT_REPORTS = ROOT / "outputs/trustworthy_soz_v29_research_reports_v39_20260816"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_private_case_audit_v40_20260816"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _list_field(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("invalid serialized channel list")
    return parsed


def _selected_cases(
    errors: Sequence[Mapping[str, str]],
    candidates: Mapping[str, Mapping[str, str]],
) -> list[dict[str, object]]:
    definitions = (
        ("exact", lambda row: float(row["strict"]) == 1.0),
        ("neighbor_only", lambda row: float(row["neighbor_only"]) == 1.0),
        ("contralateral_far", lambda row: row["far_subtype"] == "contralateral_far"),
        ("known_spread_top1", lambda row: float(row["known_spread_top1"]) == 1.0),
    )
    selected = []
    for stratum, predicate in definitions:
        rows = [row for row in errors if predicate(row)]
        if not rows:
            raise ValueError(f"empty required case stratum: {stratum}")
        margins = [float(candidates[row["unit_id"]]["top1_top2_margin"]) for row in rows]
        median_margin = float(statistics.median(margins))
        chosen = min(
            rows,
            key=lambda row: (
                abs(float(candidates[row["unit_id"]]["top1_top2_margin"]) - median_margin),
                row["unit_id"],
            ),
        )
        candidate = candidates[chosen["unit_id"]]
        selected.append(
            {
                "stratum": stratum,
                "stratum_event_count": len(rows),
                "stratum_median_margin": median_margin,
                "selection_rule": "closest_to_stratum_median_margin_then_event_id",
                "event_id": chosen["unit_id"],
                "patient_id": chosen["patient_id"],
                "predicted_top1": chosen["top1"],
                "predicted_top5": candidate["top5"].split("|"),
                "top1_location_zh": candidate["top1_location_zh"],
                "top1_top2_margin": float(candidate["top1_top2_margin"]),
                "fold_top1_agreement_count": int(candidate["fold_top1_agreement_count"]),
                "reference_positive_channels": _list_field(chosen["positive_channels"]),
                "known_spread_channels": _list_field(chosen["known_spread_channels"]),
                "first_positive_rank": int(chosen["first_positive_rank"]),
                "far_subtype": chosen["far_subtype"] or None,
                "reference_laterality_stratum": chosen["reference_laterality_stratum"],
                "source_sfreq_hz": float(chosen["source_sfreq_hz"]),
            }
        )
    if len({row["event_id"] for row in selected}) != 4:
        raise ValueError("case strata selected duplicate events")
    return selected


def _interpretation(case: Mapping[str, object]) -> str:
    stratum = case["stratum"]
    if stratum == "exact":
        return "首位候选属于事件级医生显著电极集合；该病例用于展示严格一致，但不能代表队列整体。"
    if stratum == "neighbor_only":
        return (
            "首位候选不在医生显著电极集合内，仅因DeepSOZ邻域规则计为命中；"
            "该病例说明neighborhood-4不能替代严格电极评价。"
        )
    if stratum == "contralateral_far":
        return (
            "首位候选位于reference对侧且属于far错误，即使首个reference阳性仍出现在候选列表中，"
            "Top-1跨侧错误仍具有临床风险。"
        )
    if stratum == "known_spread_top1":
        return (
            "首位候选落入医生显式记录的spread集合，说明模型可能将后续头皮受累误排为起始区参考候选。"
        )
    raise ValueError(stratum)


def _render_html(cases: Sequence[Mapping[str, object]], report_directory_name: str) -> str:
    sections = []
    for case in cases:
        event_id = str(case["event_id"])
        positive = "、".join(case["reference_positive_channels"]) or "无"
        spread = "、".join(case["known_spread_channels"]) or "未记录"
        top5 = "、".join(case["predicted_top5"])
        sections.append(
            f"""<section class="card"><h2>{escape(str(case['stratum']))} · {escape(event_id)}</h2>
<div class="grid"><div><p><b>患者：</b>{escape(str(case['patient_id']))}</p><p><b>v29 Top-5：</b>{escape(top5)}</p>
<p><b>医生显著电极：</b>{escape(positive)}</p><p><b>已知spread：</b>{escape(spread)}</p>
<p><b>首个阳性顺位：</b>{int(case['first_positive_rank'])}</p><p><b>折间首位一致：</b>{int(case['fold_top1_agreement_count'])}/5</p>
<p>{escape(_interpretation(case))}</p><p><a href="../{escape(report_directory_name)}/html/private_event/{escape(event_id)}.html">打开不含gold的v29研究报告</a></p></div>
<img src="../{escape(report_directory_name)}/waveforms/private_event/{escape(event_id)}.png" alt="{escape(event_id)}波形"></div></section>"""
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Private outcome-stratified case audit</title>
<style>body{{font-family:sans-serif;background:#f4f7f8;color:#203038;max-width:1100px;margin:24px auto;padding:0 18px;line-height:1.55}}.card{{background:#fff;border:1px solid #dce5e8;border-radius:10px;padding:18px 22px;margin:14px 0}}.warning{{border-left:4px solid #c8872b;background:#fff8e9}}.grid{{display:grid;grid-template-columns:1fr 1.5fr;gap:20px}}img{{width:100%;border:1px solid #dce5e8}}a{{color:#176779}}@media(max-width:780px){{.grid{{grid-template-columns:1fr}}}}</style></head><body>
<h1>Private outcome-stratified case audit</h1><div class="card warning">这是开标后的审稿/失败分析材料。四个病例按预设结果分层后，以最接近各层中位margin的事件确定性选取；不是盲法临床病例抽样，且没有修改任何v29预测或报告。</div>{''.join(sections)}</body></html>"""


def _markdown(cases: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Private outcome-stratified病例审计 v40",
        "",
        "本材料在private reference已打开后生成。病例先按exact、neighbor-only、contralateral-far和known-spread Top-1分层，再选择最接近该层v29 margin中位数的事件；并列时按event ID。它是审稿/失败分析，不是target-blind病例抽样。",
        "",
        "| Stratum | Event / patient | v29 Top-1 | Reference positives | Known spread | First positive rank | Margin |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for case in cases:
        lines.append(
            "| {stratum} | {event_id} / {patient_id} | {predicted_top1} | {positive} | {spread} | {rank} | {margin:.4f} |".format(
                stratum=case["stratum"],
                event_id=case["event_id"],
                patient_id=case["patient_id"],
                predicted_top1=case["predicted_top1"],
                positive=", ".join(case["reference_positive_channels"]),
                spread=", ".join(case["known_spread_channels"]) or "not recorded",
                rank=case["first_positive_rank"],
                margin=case["top1_top2_margin"],
            )
        )
    lines.extend(["", "## 逐例解释", ""])
    for case in cases:
        lines.extend(
            [
                f"### {case['stratum']}：{case['event_id']}",
                "",
                _interpretation(case),
                "",
                f"- v29 Top-5：{', '.join(case['predicted_top5'])}",
                f"- 分层大小：{case['stratum_event_count']}个事件；选择规则：最接近该层中位margin。",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def run(
    error_path: Path,
    report_directory: Path,
) -> tuple[dict[str, object], list[dict[str, object]], str, str]:
    error_path = error_path.resolve(strict=True)
    report_directory = report_directory.resolve(strict=True)
    candidate_path = (report_directory / "candidate_table.csv").resolve(strict=True)
    report_manifest_path = (report_directory / "manifest.json").resolve(strict=True)
    errors = _csv_rows(error_path)
    candidate_rows = _csv_rows(candidate_path)
    candidates = {row["event_id"]: row for row in candidate_rows}
    if len(errors) != 51 or len(candidates) != 88:
        raise ValueError("unexpected private case-audit cohort size")
    if not {row["unit_id"] for row in errors}.issubset(candidates):
        raise ValueError("error audit contains an event without a v29 report")
    cases = _selected_cases(errors, candidates)
    manifest = {
        "schema_version": "trustworthy_soz_private_outcome_stratified_case_audit_v40",
        "status": "completed_post_open_outcome_stratified_failure_analysis",
        "case_count": len(cases),
        "strata": [row["stratum"] for row in cases],
        "selection_rule": "closest_to_within_stratum_median_v29_margin_then_event_id",
        "source_files": {
            "private_error_audit": str(error_path.relative_to(ROOT)),
            "private_error_audit_sha256": _sha256(error_path),
            "v29_candidate_table": str(candidate_path.relative_to(ROOT)),
            "v29_candidate_table_sha256": _sha256(candidate_path),
            "v29_report_manifest": str(report_manifest_path.relative_to(ROOT)),
            "v29_report_manifest_sha256": _sha256(report_manifest_path),
        },
        "access_receipt": {
            "private_reference_and_outcome_loaded": True,
            "prediction_or_report_modified": False,
            "model_threshold_or_wording_selected": False,
            "case_selection_used_outcome_strata": True,
        },
        "claim_boundary": {
            "target_blind_case_sample": False,
            "representative_prevalence_sample": False,
            "success_only_case_selection": False,
            "reviewer_facing_failure_analysis": True,
        },
        "cases": cases,
    }
    html = _render_html(cases, report_directory.name)
    markdown = _markdown(cases)
    return manifest, cases, html, markdown


def publish(
    output: Path,
    manifest: Mapping[str, object],
    cases: Sequence[Mapping[str, object]],
    html: str,
    markdown: str,
) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with (staging / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
            writer.writeheader()
            writer.writerows(cases)
        (staging / "index.html").write_text(html, encoding="utf-8")
        (staging / "paper_case_summary.md").write_text(markdown, encoding="utf-8")
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, cases, html, markdown = run(args.errors, args.reports)
    output = publish(args.output, manifest, cases, html, markdown)
    print(
        json.dumps(
            {
                "output": str(output),
                "cases": [row["event_id"] for row in cases],
                "outcome_stratified": True,
                "prediction_modified": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
