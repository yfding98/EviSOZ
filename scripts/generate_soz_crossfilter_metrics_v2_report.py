#!/usr/bin/env python3
"""Collect SOZCrossFilter metrics-v2 full and ablation results into a report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


VARIANT_ORDER = [
    "baseline",
    "no_motif",
    "no_topology_prior",
    "no_graph_filter",
    "no_ranking_loss",
    "weight_channel1_region05",
    "weight_channel1_region2",
    "weight_channel05_region1",
    "weight_channel2_region1",
]

VARIANT_LABELS = {
    "baseline": "完整 SOZCrossFilter",
    "no_motif": "去掉 motif",
    "no_topology_prior": "去掉 topology prior",
    "no_graph_filter": "去掉 graph filter",
    "no_ranking_loss": "去掉 ranking loss",
    "weight_channel1_region05": "channel=1.0, region=0.5",
    "weight_channel1_region2": "channel=1.0, region=2.0",
    "weight_channel05_region1": "channel=0.5, region=1.0",
    "weight_channel2_region1": "channel=2.0, region=1.0",
}

KEY_ABLATION_METRICS = [
    ("region_compact_f1", "region compact F1"),
    ("channel_compact_f1", "channel compact F1"),
    ("channel_region_constrained_compact_f1", "region-constrained channel F1"),
    ("region_top1", "region Top1"),
    ("region_top3", "region Top3"),
    ("region_mrr", "region MRR"),
    ("channel_top1", "channel Top1"),
    ("channel_top5", "channel Top5"),
    ("channel_mrr", "channel MRR"),
    ("region_auroc", "region AUROC"),
    ("region_auprc_lift", "region AUPRC lift"),
    ("channel_auroc", "channel AUROC"),
    ("channel_auprc_lift", "channel AUPRC lift"),
    ("deepsoz_seizure_accuracy", "DeepSOZ seizure acc"),
    ("deepsoz_patient_accuracy", "DeepSOZ patient acc"),
]

FULL_METRIC_COLUMNS = [
    ("region_compact_f1", "region F1"),
    ("region_compact_precision", "region P"),
    ("region_compact_recall", "region R"),
    ("region_compact_specificity", "region spec"),
    ("region_compact_false_positive_rate", "region FPR"),
    ("region_compact_avg_predicted", "region avg pred"),
    ("region_compact_avg_true", "region avg true"),
    ("region_maxf1_f1", "region maxF1"),
    ("region_lowfpr_f1", "region lowFPR F1"),
    ("region_top1", "region Top1"),
    ("region_top2", "region Top2"),
    ("region_top3", "region Top3"),
    ("region_mrr", "region MRR"),
    ("region_auroc", "region AUROC"),
    ("region_auprc", "region AUPRC"),
    ("region_chance_auprc", "region chance AUPRC"),
    ("region_auprc_lift", "region lift"),
    ("channel_compact_f1", "channel F1"),
    ("channel_compact_avg_predicted", "channel avg pred"),
    ("channel_region_constrained_compact_f1", "region-constr ch F1"),
    ("channel_top1", "channel Top1"),
    ("channel_top3", "channel Top3"),
    ("channel_top5", "channel Top5"),
    ("channel_mrr", "channel MRR"),
    ("channel_auroc", "channel AUROC"),
    ("channel_auprc", "channel AUPRC"),
    ("channel_chance_auprc", "channel chance AUPRC"),
    ("channel_auprc_lift", "channel lift"),
    ("deepsoz_seizure_accuracy", "DeepSOZ seizure acc"),
    ("deepsoz_patient_accuracy", "DeepSOZ patient acc"),
    ("deepsoz_n_channels", "DeepSOZ n_ch"),
    ("deepsoz_mean_neighbours", "DeepSOZ mean neighbours"),
]

SET_METRIC_SUFFIXES = [
    ("precision", "P"),
    ("recall", "R"),
    ("specificity", "Spec"),
    ("false_positive_rate", "FPR"),
    ("false_negative_rate", "FNR"),
    ("f1", "F1"),
    ("dice", "Dice"),
    ("jaccard", "Jaccard"),
    ("exact_match", "Exact"),
    ("any_hit", "Any hit"),
    ("all_coverage", "All covered"),
    ("avg_predicted", "Avg pred"),
    ("avg_true", "Avg true"),
    ("tp", "TP"),
    ("fp", "FP"),
    ("tn", "TN"),
    ("fn", "FN"),
    ("policy_threshold", "tau"),
    ("policy_gap", "gap"),
    ("policy_topk", "topk"),
]

IMBALANCE_METRICS = [
    ("region_positive_prevalence", "region prev"),
    ("region_negative_to_positive_ratio", "region neg:pos"),
    ("region_support_positive", "region pos"),
    ("region_support_negative", "region neg"),
    ("region_support_total", "region total"),
    ("region_auroc", "region AUROC"),
    ("region_auprc", "region AUPRC"),
    ("region_chance_auprc", "region chance"),
    ("region_auprc_lift", "region lift"),
    ("channel_positive_prevalence", "channel prev"),
    ("channel_negative_to_positive_ratio", "channel neg:pos"),
    ("channel_support_positive", "channel pos"),
    ("channel_support_negative", "channel neg"),
    ("channel_support_total", "channel total"),
    ("channel_auroc", "channel AUROC"),
    ("channel_auprc", "channel AUPRC"),
    ("channel_chance_auprc", "channel chance"),
    ("channel_auprc_lift", "channel lift"),
]

DEEPSOZ_METRICS = [
    ("deepsoz_seizure_accuracy", "seizure acc"),
    ("deepsoz_seizure_uncertainty", "seizure uncertainty"),
    ("deepsoz_patient_accuracy", "patient acc"),
    ("deepsoz_patient_uncertainty", "patient uncertainty"),
    ("deepsoz_n_seizures", "n seizures"),
    ("deepsoz_n_patients", "n patients"),
    ("deepsoz_n_channels", "n channels"),
    ("deepsoz_mean_neighbours", "mean neighbours"),
    ("region_deepsoz_n_channels", "region n labels"),
    ("region_deepsoz_mean_neighbours", "region mean neighbours"),
    ("channel_deepsoz_n_channels", "channel n channels"),
    ("channel_deepsoz_mean_neighbours", "channel mean neighbours"),
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_private_summary(path: Path, *, expected_folds: int = 43) -> Dict[str, Any]:
    data = load_json(path)
    aggregate = data.get("aggregate", {})
    metrics = {
        key.removeprefix("macro_test_"): value
        for key, value in aggregate.items()
        if key.startswith("macro_test_")
    }
    folds_total = aggregate.get("folds_total")
    folds_ok = aggregate.get("folds_ok")
    status = "ok"
    if expected_folds > 0:
        try:
            if int(folds_total) != expected_folds or int(folds_ok) != expected_folds:
                status = "incomplete"
        except (TypeError, ValueError):
            status = "incomplete"
    return {
        "status": status,
        "folds_total": folds_total,
        "folds_ok": folds_ok,
        "selection_metric": aggregate.get("selection_metric"),
        "best_val_score": aggregate.get("best_val_score"),
        "metrics": metrics,
        "raw_aggregate": aggregate,
    }


def flatten_tusz_metrics(path: Path, *, expected_samples: int = 205) -> Dict[str, Any]:
    data = load_json(path)
    metrics = data.get("test_metrics", {})
    status = "ok"
    if expected_samples > 0:
        try:
            if int(metrics.get("samples")) != expected_samples:
                status = "incomplete"
        except (TypeError, ValueError):
            status = "incomplete"
    return {
        "status": status,
        "folds_total": None,
        "folds_ok": None,
        "selection_metric": data.get("selection_metric"),
        "best_val_score": data.get("best_val_score"),
        "metrics": metrics,
        "raw_aggregate": data,
    }


def missing_row(dataset: str, scope: str, variant: str, source: Path) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "scope": scope,
        "variant": variant,
        "variant_label": VARIANT_LABELS.get(variant, variant),
        "status": "missing",
        "source": str(source),
        "folds_total": None,
        "folds_ok": None,
        "selection_metric": None,
        "best_val_score": None,
        "metrics": {},
        "raw_aggregate": {},
    }


def build_row(
    dataset: str,
    scope: str,
    variant: str,
    source: Path,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    row = {
        "dataset": dataset,
        "scope": scope,
        "variant": variant,
        "variant_label": VARIANT_LABELS.get(variant, variant),
        "source": str(source),
    }
    row.update(payload)
    return row


def collect_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    private_full = Path(args.private_full_summary)
    if private_full.is_file():
        rows.append(
            build_row(
                "private",
                "full",
                "baseline",
                private_full,
                flatten_private_summary(private_full, expected_folds=int(args.expected_private_folds)),
            )
        )
    else:
        rows.append(missing_row("private", "full", "baseline", private_full))

    tusz_full = Path(args.tusz_full_metrics)
    if tusz_full.is_file():
        rows.append(
            build_row(
                "tusz",
                "full",
                "baseline",
                tusz_full,
                flatten_tusz_metrics(tusz_full, expected_samples=int(args.expected_tusz_samples)),
            )
        )
    else:
        rows.append(missing_row("tusz", "full", "baseline", tusz_full))

    private_root = Path(args.private_ablation_root)
    for variant in VARIANT_ORDER:
        path = private_root / variant / "private_lopo" / "lopo_summary.json"
        if path.is_file():
            rows.append(
                build_row(
                    "private",
                    "ablation",
                    variant,
                    path,
                    flatten_private_summary(path, expected_folds=int(args.expected_private_folds)),
                )
            )
        else:
            rows.append(missing_row("private", "ablation", variant, path))

    tusz_root = Path(args.tusz_ablation_root)
    for variant in VARIANT_ORDER:
        path = tusz_root / variant / "tusz" / "metrics.json"
        if path.is_file():
            rows.append(
                build_row(
                    "tusz",
                    "ablation",
                    variant,
                    path,
                    flatten_tusz_metrics(path, expected_samples=int(args.expected_tusz_samples)),
                )
            )
        else:
            rows.append(missing_row("tusz", "ablation", variant, path))

    return rows


def metric(row: Mapping[str, Any], name: str) -> Any:
    metrics = row.get("metrics", {})
    if isinstance(metrics, Mapping):
        return metrics.get(name)
    return None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if is_number(value):
        number = float(value)
        if abs(number - round(number)) < 1e-10 and abs(number) >= 10:
            return f"{number:.0f}"
        return f"{number:.{digits}f}"
    return str(value)


def fmt_delta(value: Any, baseline: Any) -> str:
    if not is_number(value) or not is_number(baseline):
        return ""
    diff = float(value) - float(baseline)
    return f"{diff:+.4f}"


def escape_md(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|")


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(escape_md(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_md(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    metric_keys = sorted({key for row in rows for key in row.get("metrics", {})})
    fixed_keys = [
        "dataset",
        "scope",
        "variant",
        "variant_label",
        "status",
        "folds_ok",
        "folds_total",
        "selection_metric",
        "best_val_score",
        "source",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fixed_keys + metric_keys)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in fixed_keys}
            metrics = row.get("metrics", {})
            if isinstance(metrics, Mapping):
                out.update(metrics)
            writer.writerow(out)


def row_index(rows: Iterable[Mapping[str, Any]]) -> Dict[tuple, Mapping[str, Any]]:
    return {(row.get("dataset"), row.get("scope"), row.get("variant")): row for row in rows}


def folds_or_samples(row: Mapping[str, Any]) -> str:
    if row.get("dataset") == "private":
        ok = row.get("folds_ok")
        total = row.get("folds_total")
        return f"{fmt(ok, 0)}/{fmt(total, 0)} folds"
    samples = metric(row, "samples")
    return f"{fmt(samples, 0)} seizures"


def build_full_table(rows: Sequence[Mapping[str, Any]]) -> str:
    selected = [row for row in rows if row.get("scope") == "full"]
    table_rows: List[List[str]] = []
    for row in selected:
        table_rows.append(
            [
                str(row.get("dataset")),
                folds_or_samples(row),
                str(row.get("status")),
                str(row.get("selection_metric") or ""),
                fmt(row.get("best_val_score")),
                *[fmt(metric(row, key)) for key, _label in FULL_METRIC_COLUMNS],
            ]
        )
    return md_table(
        ["dataset", "n", "status", "selection", "best val", *[label for _key, label in FULL_METRIC_COLUMNS]],
        table_rows,
    )


def build_ablation_key_table(rows: Sequence[Mapping[str, Any]], dataset: str) -> str:
    selected = [row for row in rows if row.get("scope") == "ablation" and row.get("dataset") == dataset]
    by_variant = {str(row.get("variant")): row for row in selected}
    baseline = by_variant.get("baseline")
    table_rows: List[List[str]] = []
    for variant in VARIANT_ORDER:
        row = by_variant.get(variant)
        if not row:
            continue
        values = [
            variant,
            str(row.get("variant_label")),
            str(row.get("status")),
            folds_or_samples(row),
        ]
        for key, _label in KEY_ABLATION_METRICS:
            value = metric(row, key)
            delta = fmt_delta(value, metric(baseline, key) if baseline else None)
            values.append(fmt(value))
            values.append(delta if variant != "baseline" else "")
        table_rows.append(values)
    headers = ["variant", "说明", "status", "n"]
    for _key, label in KEY_ABLATION_METRICS:
        headers.extend([label, "delta"])
    return md_table(headers, table_rows)


def build_set_table(rows: Sequence[Mapping[str, Any]], dataset: str, prefix: str, title_label: str) -> str:
    selected = [
        row
        for row in rows
        if row.get("scope") == "ablation" and row.get("dataset") == dataset and row.get("variant") in VARIANT_ORDER
    ]
    by_variant = {str(row.get("variant")): row for row in selected}
    table_rows: List[List[str]] = []
    for variant in VARIANT_ORDER:
        row = by_variant.get(variant)
        if not row:
            continue
        table_rows.append(
            [
                variant,
                str(row.get("status")),
                *[fmt(metric(row, f"{prefix}_{suffix}")) for suffix, _label in SET_METRIC_SUFFIXES],
            ]
        )
    headers = ["variant", "status", *[label for _suffix, label in SET_METRIC_SUFFIXES]]
    return f"### {dataset} - {title_label}\n\n" + md_table(headers, table_rows)


def build_imbalance_table(rows: Sequence[Mapping[str, Any]], dataset: str) -> str:
    selected = [row for row in rows if row.get("scope") == "ablation" and row.get("dataset") == dataset]
    by_variant = {str(row.get("variant")): row for row in selected}
    table_rows: List[List[str]] = []
    for variant in VARIANT_ORDER:
        row = by_variant.get(variant)
        if not row:
            continue
        table_rows.append([variant, *[fmt(metric(row, key)) for key, _label in IMBALANCE_METRICS]])
    return md_table(["variant", *[label for _key, label in IMBALANCE_METRICS]], table_rows)


def build_deepsoz_table(rows: Sequence[Mapping[str, Any]], dataset: str) -> str:
    selected = [row for row in rows if row.get("scope") == "ablation" and row.get("dataset") == dataset]
    by_variant = {str(row.get("variant")): row for row in selected}
    table_rows: List[List[str]] = []
    for variant in VARIANT_ORDER:
        row = by_variant.get(variant)
        if not row:
            continue
        table_rows.append([variant, *[fmt(metric(row, key)) for key, _label in DEEPSOZ_METRICS]])
    return md_table(["variant", *[label for _key, label in DEEPSOZ_METRICS]], table_rows)


def build_source_table(rows: Sequence[Mapping[str, Any]]) -> str:
    table_rows = [
        [
            str(row.get("dataset")),
            str(row.get("scope")),
            str(row.get("variant")),
            str(row.get("status")),
            str(row.get("source")),
        ]
        for row in rows
    ]
    return md_table(["dataset", "scope", "variant", "status", "source"], table_rows)


def best_variant_summary(rows: Sequence[Mapping[str, Any]], dataset: str, key: str) -> str:
    candidates = [
        row
        for row in rows
        if row.get("scope") == "ablation" and row.get("dataset") == dataset and is_number(metric(row, key))
    ]
    if not candidates:
        return ""
    best = max(candidates, key=lambda row: float(metric(row, key)))
    baseline = next((row for row in candidates if row.get("variant") == "baseline"), None)
    delta = fmt_delta(metric(best, key), metric(baseline, key) if baseline else None)
    return f"{dataset}: `{best.get('variant')}` = {fmt(metric(best, key))} ({delta} vs baseline)"


def build_report(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    missing = [row for row in rows if row.get("status") != "ok"]
    lines: List[str] = []
    lines.append("# SOZCrossFilter metrics-v2 完整实验与消融实验报告")
    lines.append("")
    lines.append(f"- 生成时间：{generated}")
    lines.append("- 评估口径：`code/tfm_soz/SOZ_MULTIREGION_TASK_DESIGN_ZH.md` 的 multiregion SOZ fixed-window 指标。")
    lines.append("- private：43 个 held-out patient 的 LOPO macro average；TUSZ：train/dev/eval split，表中为 eval/test 指标。")
    lines.append("- 主 operating point：`compact_f1`，即验证集约束平均候选数量后最大化 F1。")
    lines.append("- DeepSOZ-style 指标使用 rows119 32-channel bipolar adjacency；报告中保留 `deepsoz_n_channels` 和 `deepsoz_mean_neighbours` 元数据。")
    lines.append(f"- 完整宽表 CSV：`{args.output_csv}`")
    lines.append(f"- 完整 JSON：`{args.output_json}`")
    if missing:
        lines.append(f"- 注意：有 {len(missing)} 个结果缺失或未完成，详见最后的 source table。")
    lines.append("")
    lines.append("## 1. Answer-first 结果概览")
    lines.append("")
    lines.append("完整模型在本轮更新指标下的主结果如下。private 与 TUSZ 都以 region/channel compact F1 为主，同时保留 Top-k、AUROC/AUPRC lift 和 DeepSOZ-style accuracy，避免只看 raw AUPRC 或单一 Top1。")
    lines.append("")
    lines.append(build_full_table(rows))
    lines.append("")
    lines.append("本轮消融中，各数据集按主 F1 的最佳变体为：")
    for dataset in ["private", "tusz"]:
        for key in ["region_compact_f1", "channel_compact_f1", "channel_region_constrained_compact_f1"]:
            summary = best_variant_summary(rows, dataset, key)
            if summary:
                lines.append(f"- {key}: {summary}")
    lines.append("")
    lines.append("## 2. TUSZ 消融主指标")
    lines.append("")
    lines.append(build_ablation_key_table(rows, "tusz"))
    lines.append("")
    lines.append("## 3. Private 消融主指标")
    lines.append("")
    lines.append(build_ablation_key_table(rows, "private"))
    lines.append("")
    lines.append("## 4. Region compact candidate-set 详细指标")
    lines.append("")
    lines.append(build_set_table(rows, "tusz", "region_compact", "Region compact"))
    lines.append("")
    lines.append(build_set_table(rows, "private", "region_compact", "Region compact"))
    lines.append("")
    lines.append("## 5. Channel compact candidate-set 详细指标")
    lines.append("")
    lines.append(build_set_table(rows, "tusz", "channel_compact", "Channel compact"))
    lines.append("")
    lines.append(build_set_table(rows, "private", "channel_compact", "Channel compact"))
    lines.append("")
    lines.append("## 6. Region-constrained channel compact 详细指标")
    lines.append("")
    lines.append(build_set_table(rows, "tusz", "channel_region_constrained_compact", "Region-constrained channel compact"))
    lines.append("")
    lines.append(build_set_table(rows, "private", "channel_region_constrained_compact", "Region-constrained channel compact"))
    lines.append("")
    lines.append("## 7. AUROC / AUPRC / 类别不平衡")
    lines.append("")
    lines.append("AUPRC 同时列出 chance AUPRC 和 lift，便于在不同 positive prevalence 下比较。")
    lines.append("")
    lines.append("### TUSZ")
    lines.append("")
    lines.append(build_imbalance_table(rows, "tusz"))
    lines.append("")
    lines.append("### Private")
    lines.append("")
    lines.append(build_imbalance_table(rows, "private"))
    lines.append("")
    lines.append("## 8. DeepSOZ-style accuracy 与邻接元数据")
    lines.append("")
    lines.append("### TUSZ")
    lines.append("")
    lines.append(build_deepsoz_table(rows, "tusz"))
    lines.append("")
    lines.append("### Private")
    lines.append("")
    lines.append(build_deepsoz_table(rows, "private"))
    lines.append("")
    lines.append("## 9. 复现命令")
    lines.append("")
    lines.append("### Private full LOPO")
    lines.append("")
    lines.append("```bash")
    lines.append("rtk python3 -m code.soz_crossfilter.run_lopo_private_rows119 \\")
    lines.append("  --preprocessed-dir outputs/soz_crossfilter/private_rows119_segments_15s \\")
    lines.append("  --output-root outputs/soz_crossfilter/metrics_v2_crossfilter_deepregion_private_lopo_full_30epoch \\")
    lines.append("  --epochs 30 --batch-size 8 --device cuda \\")
    lines.append("  --region-channel-pool max --detach-region-channel-evidence \\")
    lines.append("  --region-channel-blend-init -1.5 --final-region-channel-blend-init -1.25 \\")
    lines.append("  --tracker-region-blend-init -1.0 --deepsoz-region-blend-init 1.0")
    lines.append("```")
    lines.append("")
    lines.append("### TUSZ full")
    lines.append("")
    lines.append("```bash")
    lines.append("rtk python3 -m code.soz_crossfilter.train_private_rows119 \\")
    lines.append("  --preprocessed-dir outputs/soz_crossfilter/tusz_fnsz_onset_crossfilter \\")
    lines.append("  --output-dir outputs/soz_crossfilter/metrics_v2_crossfilter_restored_tusz_3epoch_lr5e5_ch8 \\")
    lines.append("  --split-mode index --train-splits train --val-splits dev --test-splits eval \\")
    lines.append("  --sources tusz --epochs 3 --batch-size 8 --device cuda --model crossfilter \\")
    lines.append("  --epoch-log-interval 1 --deepsoz-style-mc-samples 20")
    lines.append("```")
    lines.append("")
    lines.append("### Ablation")
    lines.append("")
    lines.append("```bash")
    lines.append("rtk python3 -m code.soz_crossfilter.run_crossfilter_ablation \\")
    lines.append("  --datasets private,tusz \\")
    lines.append("  --output-root <ablation-output-root> \\")
    lines.append("  --private-epochs 30 --tusz-epochs 3 --batch-size 8 --lr 5e-5 --device cuda \\")
    lines.append("  --region-channel-pool max --detach-region-channel-evidence \\")
    lines.append("  --region-channel-blend-init -1.5 --final-region-channel-blend-init -1.25 \\")
    lines.append("  --tracker-region-blend-init -1.0 --deepsoz-region-blend-init 1.0")
    lines.append("```")
    lines.append("")
    lines.append("## 10. Source table")
    lines.append("")
    lines.append(build_source_table(rows))
    lines.append("")
    lines.append("说明：正文表格保留论文主指标、候选集合指标、排序指标、类别不平衡指标和 DeepSOZ-style 指标；完整宽表 CSV/JSON 保留每个 JSON 中的全部 metric 字段，包括 per-region/per-channel AUROC、AUPRC、support 和 policy 字段。")
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--private-full-summary",
        default="outputs/soz_crossfilter/metrics_v2_crossfilter_deepregion_private_lopo_full_30epoch/lopo_summary.json",
    )
    parser.add_argument(
        "--tusz-full-metrics",
        default="outputs/soz_crossfilter/metrics_v2_crossfilter_restored_tusz_3epoch_lr5e5_ch8/metrics.json",
    )
    parser.add_argument(
        "--private-ablation-root",
        default="outputs/soz_crossfilter/metrics_v2_ablation_optimized_private_deepregion_30epoch",
    )
    parser.add_argument(
        "--tusz-ablation-root",
        default="outputs/soz_crossfilter/metrics_v2_ablation_optimized_tusz_restored_3epoch",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/soz_crossfilter/metrics_v2_optimized_full_ablation_report_zh.md",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/soz_crossfilter/metrics_v2_optimized_full_ablation_all_metrics_wide.csv",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/soz_crossfilter/metrics_v2_optimized_full_ablation_all_metrics.json",
    )
    parser.add_argument("--expected-private-folds", type=int, default=43)
    parser.add_argument("--expected-tusz-samples", type=int, default=205)
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = collect_rows(args)
    missing = [row for row in rows if row.get("status") != "ok"]
    if args.require_complete and missing:
        details = ", ".join(f"{row.get('dataset')}/{row.get('scope')}/{row.get('variant')}" for row in missing)
        raise SystemExit(f"Missing or incomplete rows: {details}")

    for output in [Path(args.output_md), Path(args.output_csv), Path(args.output_json)]:
        output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(Path(args.output_csv), rows)
    write_json(Path(args.output_json), rows)
    Path(args.output_md).write_text(build_report(rows, args), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "missing": len(missing), "output_md": args.output_md}, ensure_ascii=False))


if __name__ == "__main__":
    main()
