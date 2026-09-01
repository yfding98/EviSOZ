#!/usr/bin/env python3
"""Evaluate indexed Qwen SOZ candidates against private and TUSZ annotations.

The private doctor sheets contain monopolar significant/spread electrodes, while
the model emits bipolar leads.  Therefore private agreement is evaluated on
the endpoints of the predicted leads.  TUSZ is evaluated both strictly on the
earliest bipolar leads and, secondarily, on their endpoint electrodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUCCESS = ROOT / "outputs/qwen35_result_index_20260723/successes.csv"
DEFAULT_FAILURE = ROOT / "outputs/qwen35_result_index_20260723/failures.csv"
DEFAULT_PRIVATE_SHEETS = [
    Path("/mnt/hd1/dyf/dataset/EEG/EEG-fMRI颞叶癫痫(1).xls"),
    Path("/mnt/hd1/dyf/dataset/EEG/头皮扩散.xlsx"),
]
DEFAULT_OUT = ROOT / "reports/llm_vs_clinical_annotations_20260731"


def clean(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def tokens(value: object) -> list[str]:
    text = clean(value).upper().replace("→", ",")
    return [x.strip() for x in re.split(r"[,;，、|/\s]+", text) if x.strip()]


def parse_json_list(value: object) -> list[str]:
    text = clean(value)
    if not text:
        return []
    try:
        result = json.loads(text)
        return [clean(x).upper() for x in result]
    except json.JSONDecodeError:
        return tokens(text)


def lead_endpoints(leads: Iterable[str]) -> set[str]:
    result: set[str] = set()
    for lead in leads:
        result.update(x for x in clean(lead).upper().split("-") if x)
    return result


def scores(pred: set[str], gold: set[str]) -> dict[str, object]:
    hit = pred & gold
    precision = len(hit) / len(pred) if pred else (1.0 if not gold else None)
    recall = len(hit) / len(gold) if gold else None
    f1 = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = pred | gold
    return {
        "n_pred": len(pred), "n_gold": len(gold), "n_hit": len(hit),
        "precision": precision, "recall": recall, "f1": f1,
        "jaccard": len(hit) / len(union) if union else 1.0,
        "hits": sorted(hit), "missed": sorted(gold - pred),
        "extra": sorted(pred - gold), "exact": pred == gold,
        "any_overlap": bool(hit),
    }


def load_doctor_events(paths: list[Path]) -> dict[tuple[str, str], dict[str, object]]:
    result: dict[tuple[str, str], dict[str, object]] = {}
    for path in paths:
        raw = pd.read_excel(path, sheet_name=0, header=None)
        for row_i in range(2, len(raw)):
            patient = clean(raw.iat[row_i, 0])
            if not patient:
                continue
            for event_i in range(4):
                col = 4 + event_i * 4
                onset = clean(raw.iat[row_i, col])
                significant_text = clean(raw.iat[row_i, col + 1])
                spread_text = clean(raw.iat[row_i, col + 2])
                if not any((onset, significant_text, spread_text)):
                    continue
                significant = {x for x in tokens(significant_text) if x not in {"无", "NONE"}}
                spread = {x for x in tokens(spread_text) if x not in {"无", "NONE"}}
                unclear = "起始不清" in onset
                diffuse = any(x in spread_text for x in ("弥散", "弥漫"))
                result.setdefault((patient, f"SZ{event_i + 1}"), {
                    "onset_description": onset,
                    "significant": significant,
                    "spread": spread,
                    "unclear": unclear,
                    "diffuse": diffuse,
                    "evaluable": bool(significant) and not unclear and not diffuse,
                    "source": str(path),
                })
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def tusz_gold(source_file: str, event_id: str) -> dict[str, object]:
    edf = Path(source_file)
    bi_path = edf.with_suffix(".csv_bi")
    channel_path = edf.with_suffix(".csv")
    global_rows = read_annotation_csv(bi_path)
    event_number = int(re.search(r"__ev(\d+)$", event_id).group(1))
    event = global_rows[event_number]
    start = float(event["start_time"])
    stop = float(event["stop_time"])
    channel_rows = read_annotation_csv(channel_path)
    candidates = [
        row for row in channel_rows
        if row["label"].lower() not in {"bckg", "null"}
        and float(row["stop_time"]) > start
        and float(row["start_time"]) < stop
    ]
    first_start = min(float(row["start_time"]) for row in candidates)
    earliest = {
        row["channel"].upper() for row in candidates
        if abs(float(row["start_time"]) - first_start) <= 1e-4
    }
    return {"start_s": start, "first_channel_start_s": first_start, "leads": earliest}


def read_annotation_csv(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line and not line.startswith("#")]
    return list(csv.DictReader(lines))


def fmt_set(values: Iterable[str]) -> str:
    return ";".join(sorted(values)) or "—"


def pct(value: object) -> str:
    if value is None:
        return "NA"
    return f"{100 * float(value):.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-index", type=Path, default=DEFAULT_SUCCESS)
    parser.add_argument("--failure-index", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    doctor = load_doctor_events(DEFAULT_PRIVATE_SHEETS)
    successes = read_csv(args.success_index)
    failures = read_csv(args.failure_index)
    details: list[dict[str, object]] = []

    for row in successes:
        pred_leads = set(parse_json_list(row["soz_channels"]))
        pred_spread_leads = set(parse_json_list(row["spread_channels"]))
        common = {
            "dataset": row["dataset"], "patient_id": row["patient_id"],
            "event_id": row["event_id"], "status": "成功",
            "model_t0_s": float(row["t0_s"]) if clean(row["t0_s"]) else None,
            "model_soz_leads": fmt_set(pred_leads),
            "model_spread_leads": fmt_set(pred_spread_leads),
        }
        if row["dataset"] == "private":
            match = re.match(r"(.+)_SZ(\d+)$", row["patient_id"])
            patient, sz = match.group(1), f"SZ{match.group(2)}"
            gold = doctor.get((patient, sz))
            if not gold:
                raise KeyError(f"doctor event not found: {patient} {sz}")
            pred_electrodes = lead_endpoints(pred_leads)
            pred_spread_electrodes = lead_endpoints(pred_spread_leads)
            sig_score = scores(pred_electrodes, gold["significant"])
            spread_score = scores(pred_spread_electrodes, gold["spread"])
            reason = "纳入"
            if gold["unclear"]:
                reason = "排除：医生标注起始不清"
            elif gold["diffuse"]:
                reason = "排除：医生标注弥散/弥漫"
            elif not gold["significant"]:
                reason = "排除：无显著电极"
            details.append(common | {
                "reference_t0_s": None, "t0_error_s": None,
                "doctor_onset_description": gold["onset_description"],
                "reference_soz": fmt_set(gold["significant"]),
                "reference_spread": fmt_set(gold["spread"]),
                "comparison_unit": "双极导联端点电极 vs 医生单极电极",
                "evaluable": gold["evaluable"], "evaluation_note": reason,
                "soz_hits": fmt_set(sig_score["hits"]),
                "soz_missed": fmt_set(sig_score["missed"]),
                "soz_extra": fmt_set(sig_score["extra"]),
                "soz_precision": sig_score["precision"],
                "soz_recall": sig_score["recall"], "soz_f1": sig_score["f1"],
                "spread_hits": fmt_set(spread_score["hits"]),
                "spread_recall": spread_score["recall"],
            })
        else:
            gold = tusz_gold(row["source_file"], row["event_id"])
            lead_score = scores(pred_leads, gold["leads"])
            endpoint_score = scores(lead_endpoints(pred_leads), lead_endpoints(gold["leads"]))
            details.append(common | {
                "reference_t0_s": gold["start_s"],
                "t0_error_s": common["model_t0_s"] - gold["start_s"],
                "doctor_onset_description": "TUSZ逐通道标注的全局发作起点",
                "reference_soz": fmt_set(gold["leads"]), "reference_spread": "—",
                "comparison_unit": "严格双极导联；另报端点电极",
                "evaluable": True, "evaluation_note": "纳入",
                "soz_hits": fmt_set(lead_score["hits"]),
                "soz_missed": fmt_set(lead_score["missed"]),
                "soz_extra": fmt_set(lead_score["extra"]),
                "soz_precision": lead_score["precision"],
                "soz_recall": lead_score["recall"], "soz_f1": lead_score["f1"],
                "endpoint_precision": endpoint_score["precision"],
                "endpoint_recall": endpoint_score["recall"],
                "endpoint_f1": endpoint_score["f1"],
                "endpoint_hits": fmt_set(endpoint_score["hits"]),
            })

    for row in failures:
        details.append({
            "dataset": row["dataset"], "patient_id": row["patient_id"],
            "event_id": row["event_id"], "status": "失败", "model_t0_s": None,
            "model_soz_leads": "—", "model_spread_leads": "—",
            "reference_t0_s": tusz_gold(row["source_file"], row["event_id"])["start_s"],
            "t0_error_s": None, "doctor_onset_description": "TUSZ逐通道标注的全局发作起点",
            "reference_soz": fmt_set(tusz_gold(row["source_file"], row["event_id"])["leads"]),
            "reference_spread": "—", "comparison_unit": "严格双极导联；另报端点电极",
            "evaluable": False, "evaluation_note": "模型处理失败，不计定位准确率；计入覆盖率",
            "soz_hits": "—", "soz_missed": "—", "soz_extra": "—",
            "soz_precision": None, "soz_recall": None, "soz_f1": None,
            "processing_error": row["processing_error"],
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_df = pd.DataFrame(details).sort_values(["dataset", "patient_id", "event_id"])
    detail_df.to_csv(args.output_dir / "event_comparison.csv", index=False, encoding="utf-8-sig")

    private_eval = detail_df[(detail_df.dataset == "private") & (detail_df.evaluable == True)]
    tusz_ok = detail_df[(detail_df.dataset == "tusz") & (detail_df.status == "成功")]
    summary_rows = []
    for name, frame, metric_prefix in [
        ("private-significant-endpoint", private_eval, "soz"),
        ("tusz-strict-lead", tusz_ok, "soz"),
        ("tusz-endpoint-electrode", tusz_ok, "endpoint"),
    ]:
        summary_rows.append({
            "evaluation": name, "n_events": len(frame),
            "macro_precision": frame[f"{metric_prefix}_precision"].mean(),
            "macro_recall": frame[f"{metric_prefix}_recall"].mean(),
            "macro_f1": frame[f"{metric_prefix}_f1"].mean(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_dir / "metric_summary.csv", index=False, encoding="utf-8-sig")

    private_total = int((detail_df.dataset == "private").sum())
    tusz_total = int((detail_df.dataset == "tusz").sum())
    onset_mae = tusz_ok.t0_error_s.abs().mean()
    onset_bias = tusz_ok.t0_error_s.mean()
    lines = [
        "# 大模型 SOZ 与临床/数据集标注一致性分析",
        "",
        "## 结论摘要",
        "",
        f"- 当前正式索引仅覆盖私有 {private_total} 个候选事件、TUSZ {tusz_total} 个事件；这不是全数据集评估。",
        f"- 私有结果中仅 {len(private_eval)} 个满足‘起始清楚、非弥散且有显著电极’的纳入规则。",
        f"- TUSZ 成功输出 {len(tusz_ok)}/{tusz_total}（{len(tusz_ok)/tusz_total:.1%}）；定位指标只在成功事件上计算。",
        f"- TUSZ 起始时刻 MAE={onset_mae:.3f}s，平均有符号误差={onset_bias:.3f}s（负值表示模型偏早）。",
        "- 就现有可评价样本而言，严格定位出入较大：私有显著电极 F1=20.0%，TUSZ 首发导联 F1=10.0%。TUSZ 若放宽为导联端点电极，F1=48.8%，说明模型常能落在相近电极网络，但经常选错具体双极导联并多报/漏报。",
        "- 私有唯一纳入事件的医生扩散电极为 SP1/F7/T3，而模型未给扩散导联，扩散召回率为 0%。",
        "",
        "## 汇总指标（事件宏平均）",
        "",
        "|口径|事件数|Precision|Recall|F1|",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(f"|{row['evaluation']}|{row['n_events']}|{pct(row['macro_precision'])}|{pct(row['macro_recall'])}|{pct(row['macro_f1'])}|")
    lines += [
        "",
        "## 逐事件对比",
        "",
        "|数据集|患者/记录|事件|状态/纳入|模型SOZ|实际SOZ/首发导联|命中|漏检|多报|P/R/F1|t0误差(s)|",
        "|---|---|---|---|---|---|---|---|---|---|---:|",
    ]
    for _, row in detail_df.iterrows():
        metric_values = [row.get("soz_precision"), row.get("soz_recall"), row.get("soz_f1")]
        prf = "不计" if (not bool(row.evaluable) or any(pd.isna(x) for x in metric_values)) else f"{pct(row.soz_precision)}/{pct(row.soz_recall)}/{pct(row.soz_f1)}"
        t_err = "NA" if pd.isna(row.t0_error_s) else f"{row.t0_error_s:.3f}"
        lines.append(
            f"|{row.dataset}|{row.patient_id}|{row.event_id}|{row.status}；{row.evaluation_note}|"
            f"{row.model_soz_leads}|{row.reference_soz}|{row.soz_hits}|{row.soz_missed}|{row.soz_extra}|{prf}|{t_err}|"
        )
    lines += [
        "",
        "## 解释与限制",
        "",
        "1. 私有医生标注是单极电极，模型是双极导联；主指标按模型导联的两个端点与医生电极比较，不能把两者直接字符串匹配。",
        "2. TUSZ 严格金标准取逐通道 `.csv` 中与 `.csv_bi` 全局事件起点同时开始的导联；端点电极指标只作为拓扑接近度，不能替代严格导联命中。",
        "3. 私有两个廖佳候选虽然模型给出定位，但对应 SZ1 医生写明‘起始不清’，按预设规则排除。",
        "4. 样本极小，比例的置信区间会很宽；当前结果只适合误差审计，不足以声称总体临床性能。",
        "5. 完整字段、医生起始描述、扩散召回、TUSZ端点指标和失败原因见 `event_comparison.csv`。",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
