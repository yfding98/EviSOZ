#!/usr/bin/env python3
"""Build the self-contained SOZ-CrossFilter technical experiment report."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "experiment_report_20260710"
SHELL_TEMPLATE = Path(
    "/home/hci-p920-5/.codex/plugins/cache/openai-curated-remote/"
    "data-analytics/0.2.6-d37358633e00/assets/html-report-shell.html"
)
V41 = ROOT / "outputs/soz_crossfilter/pure_v41_private_lopo43_baseline_full"
V43_PRIVATE = ROOT / "outputs/soz_crossfilter/pure_v43_fusiononly_private_lopo43_core_ablation"
V43_TUSZ = ROOT / "outputs/soz_crossfilter/pure_v43_fusiononly_tusz_3epoch_core_ablation/ablation_summary.json"
PRIVATE_INDEX = ROOT / "outputs/soz_crossfilter/private_rows119_segments_15s/index.csv"
BASELINES = ROOT / "outputs/soz_crossfilter/baseline_comparison_multiregion_full/baseline_comparison_summary.json"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_lopo(root: Path, variant: str):
    return load_json(root / variant / "private_lopo/lopo_summary.json")


def fold_map(summary, metric: str):
    return {
        str(row["patient"]): float(row[metric])
        for row in summary["folds"]
        if row.get("status") == "ok" and metric in row
    }


def paired_ci(baseline, ablation, metric: str, seed: int):
    left = fold_map(baseline, metric)
    right = fold_map(ablation, metric)
    patients = sorted(set(left) & set(right))
    values = np.asarray([left[p] - right[p] for p in patients], dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
    return {
        "delta": float(values.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "wins": int((values > 1e-12).sum()),
        "ties": int((np.abs(values) <= 1e-12).sum()),
        "losses": int((values < -1e-12).sum()),
        "n": len(values),
    }


_tip_id = 0


def tip(value: str, source: str, *, cls: str = "") -> str:
    global _tip_id
    _tip_id += 1
    ident = f"source-tip-{_tip_id}"
    class_attr = f" {cls}" if cls else ""
    return (
        f'<span class="source-tooltip{class_attr}" tabindex="0" aria-describedby="{ident}">{html.escape(value)}'
        f'<span class="source-tooltip-content" id="{ident}" role="tooltip">{source}</span></span>'
    )


def chart_source(ident: str, source: str) -> str:
    return (
        f'<button type="button" class="source-tooltip" aria-describedby="{ident}">Source'
        f'<span class="source-tooltip-content" id="{ident}" role="tooltip">{source}</span></button>'
    )


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def ablation_svg(rows):
    width, height = 960, 430
    left, right, top, bottom = 260, 70, 48, 48
    x_min, x_max = -0.055, 0.135
    plot_w = width - left - right
    row_h = (height - top - bottom) / len(rows)

    def sx(v):
        return left + (v - x_min) / (x_max - x_min) * plot_w

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Paired private ablation effects">']
    for tick in (-0.04, 0.00, 0.04, 0.08, 0.12):
        x = sx(tick)
        stroke = "var(--border-strong)" if tick == 0 else "var(--grid)"
        parts.append(f'<line x1="{x:.1f}" y1="{top-10}" x2="{x:.1f}" y2="{height-bottom+4}" stroke="{stroke}"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-15}" text-anchor="middle" fill="currentColor" font-size="12">{tick:+.2f}</text>')
    for i, row in enumerate(rows):
        cy = top + i * row_h + row_h / 2
        zero, end = sx(0), sx(row["delta"])
        x = min(zero, end)
        bar_w = max(abs(end - zero), 2)
        fill = "var(--blue)" if row["delta"] >= 0 else "var(--warning)"
        parts.append(f'<text x="{left-14}" y="{cy+5:.1f}" text-anchor="end" fill="currentColor" font-size="13">{html.escape(row["label"])}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{cy-11:.1f}" width="{bar_w:.1f}" height="22" rx="4" fill="{fill}"/>')
        parts.append(f'<line x1="{sx(row["low"]):.1f}" y1="{cy:.1f}" x2="{sx(row["high"]):.1f}" y2="{cy:.1f}" stroke="currentColor" stroke-width="2"/>')
        parts.append(f'<line x1="{sx(row["low"]):.1f}" y1="{cy-5:.1f}" x2="{sx(row["low"]):.1f}" y2="{cy+5:.1f}" stroke="currentColor"/>')
        parts.append(f'<line x1="{sx(row["high"]):.1f}" y1="{cy-5:.1f}" x2="{sx(row["high"]):.1f}" y2="{cy+5:.1f}" stroke="currentColor"/>')
        anchor = "start" if row["delta"] >= 0 else "end"
        label_x = end + (7 if row["delta"] >= 0 else -7)
        parts.append(f'<text x="{label_x:.1f}" y="{cy+5:.1f}" text-anchor="{anchor}" fill="currentColor" font-size="12">{row["delta"]:+.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def chance_svg(rows):
    width, height = 960, 400
    left, right, top, bottom = 75, 35, 35, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    group_w = plot_w / len(rows)
    bar_w = 44
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Observed Top-k any-hit against random expectation">']
    for tick in np.linspace(0, 1, 6):
        y = top + (1 - tick) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="var(--grid)"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" fill="currentColor" font-size="12">{tick:.1f}</text>')
    for i, row in enumerate(rows):
        cx = left + (i + 0.5) * group_w
        for j, (series, value, fill) in enumerate((("Observed", row["observed"], "var(--blue)"), ("Random", row["random"], "var(--surface-tertiary)"))):
            x = cx + (j - 0.5) * (bar_w + 10) - bar_w / 2
            y = top + (1 - value) * plot_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{top+plot_h-y:.1f}" rx="4" fill="{fill}" stroke="var(--border-strong)"/>')
            parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle" fill="currentColor" font-size="12">{value:.3f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{height-bottom+28}" text-anchor="middle" fill="currentColor" font-size="12">{html.escape(row["metric"])}</text>')
    parts.append(f'<rect x="{width-190}" y="12" width="12" height="12" rx="2" fill="var(--blue)"/><text x="{width-171}" y="23" fill="currentColor" font-size="12">Observed</text>')
    parts.append(f'<rect x="{width-100}" y="12" width="12" height="12" rx="2" fill="var(--surface-tertiary)" stroke="var(--border-strong)"/><text x="{width-81}" y="23" fill="currentColor" font-size="12">Random</text>')
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v41 = {name: load_lopo(V41, name) for name in ("baseline", "no_motif", "no_graph_filter", "no_set_refiner")}
    v43_private = load_lopo(V43_PRIVATE, "baseline")
    v43_agg = v43_private["aggregate"]
    v43_tusz = load_json(V43_TUSZ)["rows"]
    tusz_by_variant = {row["variant"]: row for row in v43_tusz}
    baseline_rows = load_json(BASELINES)["rows"]
    private_baselines = {row["model"]: row for row in baseline_rows if row["dataset"] == "private"}

    metric_specs = [
        ("no_graph_filter", "test_channel_compact_f1", "– graph · Channel F1"),
        ("no_set_refiner", "test_channel_compact_f1", "– set refiner · Channel F1"),
        ("no_motif", "test_channel_compact_f1", "– motif · Channel F1"),
        ("no_graph_filter", "test_channel_auroc", "– graph · Channel AUROC"),
        ("no_set_refiner", "test_channel_auroc", "– set refiner · Channel AUROC"),
        ("no_motif", "test_channel_auroc", "– motif · Channel AUROC"),
    ]
    ablation_rows = []
    for i, (variant, metric, label) in enumerate(metric_specs):
        result = paired_ci(v41["baseline"], v41[variant], metric, 20260710 + i)
        ablation_rows.append({"variant": variant, "metric": metric.removeprefix("test_"), "label": label, **result})

    import csv
    with PRIVATE_INDEX.open("r", encoding="utf-8-sig", newline="") as handle:
        private_rows = list(csv.DictReader(handle))

    def chance(n, m, k):
        n, m, k = int(n), int(m), int(k)
        return 1.0 if n - m < k else 1.0 - math.comb(n - m, k) / math.comb(n, k)

    chance_rows = [
        {
            "metric": "Region Top-1",
            "observed": float(v43_agg["macro_test_region_top1"]),
            "random": float(np.mean([chance(5, r["n_positive_regions"], 1) for r in private_rows])),
        },
        {
            "metric": "Region Top-3",
            "observed": float(v43_agg["macro_test_region_top3"]),
            "random": float(np.mean([chance(5, r["n_positive_regions"], 3) for r in private_rows])),
        },
        {
            "metric": "Channel Top-1",
            "observed": float(v43_agg["macro_test_channel_top1"]),
            "random": float(np.mean([chance(r["n_available_channels"], r["n_positive"], 1) for r in private_rows])),
        },
        {
            "metric": "Channel Top-5",
            "observed": float(v43_agg["macro_test_channel_top5"]),
            "random": float(np.mean([chance(r["n_available_channels"], r["n_positive"], 5) for r in private_rows])),
        },
    ]

    source_v41 = "Source: local experiment artifact<br>File: v41 private 43-fold core ablation lopo_summary.json files"
    source_v43 = "Source: local experiment artifact<br>File: v43 private baseline lopo_summary.json"
    source_index = "Source: local preprocessed-data index<br>File: private rows119 index.csv"
    source_tusz = "Source: local experiment artifact<br>File: v43 TUSZ core ablation_summary.json"
    source_baselines = "Source: local experiment artifact<br>File: baseline comparison summary.json"
    source_model = "Source: local model code and run configuration<br>File: model.py and v43 run_config.json"

    graph = next(r for r in ablation_rows if r["variant"] == "no_graph_filter" and r["metric"] == "channel_auroc")
    setref = next(r for r in ablation_rows if r["variant"] == "no_set_refiner" and r["metric"] == "channel_auroc")
    motif = next(r for r in ablation_rows if r["variant"] == "no_motif" and r["metric"] == "channel_auroc")

    def result_row(label, row, prefix, source):
        cells = [f"<td>{html.escape(label)}</td>"]
        keys = ["region_compact_f1", "channel_compact_f1", "channel_auroc", "channel_auprc_lift", "channel_compact_avg_predicted"]
        for key in keys:
            value = float(row[f"{prefix}{key}"])
            cells.append(f"<td>{tip(num(value), source)}</td>")
        return "<tr>" + "".join(cells) + "</tr>"

    tusz_table_rows = "".join(
        result_row(label, tusz_by_variant[variant], "test_", source_tusz)
        for label, variant in [
            ("Full v43", "baseline"),
            ("– Graph filter", "no_graph_filter"),
            ("– Set refiner", "no_set_refiner"),
            ("– Motif", "no_motif"),
        ]
    )

    comparison_rows = [
        (
            "SOZCrossFilter v43",
            v43_agg,
            "macro_test_",
            source_v43,
            "10 ep · balanced_szcore · max 8",
        ),
        (
            "DeepSOZ",
            private_baselines["deepsoz"],
            "macro_test_",
            source_baselines,
            "30 ep · region F1 · max 5",
        ),
        (
            "SZTrack",
            private_baselines["sztrack"],
            "macro_test_",
            source_baselines,
            "30 ep · region F1 · max 5",
        ),
    ]
    comparison_html = ""
    for label, row, prefix, source, protocol in comparison_rows:
        comparison_html += (
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{tip(num(float(row[prefix+'region_compact_f1'])), source)}</td>"
            f"<td>{tip(num(float(row[prefix+'channel_compact_f1'])), source)}</td>"
            f"<td>{tip(num(float(row[prefix+'channel_auroc'])), source)}</td>"
            f"<td>{tip(protocol, source)}</td>"
            "</tr>"
        )

    title = "SOZ-CrossFilter 顶会投稿实验审计"
    technical_summary = f"""
      <p><strong>结论：当前还不适合按“已达到顶会投稿标准”直接收口。</strong>纯模型已经有可信的模块级证据，但尚未形成同一锁定协议下的 SOTA 证据链；目前最稳的论文主线应收缩到 graph/TimeFilter 局部对比与 set refiner 候选边界，而不是三类先验都同等有效。</p>
      <p><strong>Graph 和 set refiner 是明确的主贡献。</strong>完整 {tip('43', source_v41)} 患者配对消融中，去掉 graph 后 channel AUROC 平均下降 {tip(num(graph['delta']), source_v41)}（{tip(num(graph['low']), source_v41)}–{tip(num(graph['high']), source_v41)}），去掉 set refiner 后下降 {tip(num(setref['delta']), source_v41)}（{tip(num(setref['low']), source_v41)}–{tip(num(setref['high']), source_v41)}）。</p>
      <p><strong>Motif 不是稳定贡献。</strong>私有 v41 上 full 相对 no-motif 的 channel AUROC 差为 {tip(num(motif['delta']), source_v41)}，置信区间完全低于零；v43 TUSZ 仅有小幅正信号且 Top-k/F1 混合，适合降级为数据集敏感的辅助先验。</p>
      <p><strong>最高风险在实验设计而不是继续堆模块。</strong>跨模型比较使用了不同的 checkpoint 选择、候选策略和最大预测集合；私有数据被反复用于模型迭代；Beta uncertainty head 没有监督损失。这三项会直接削弱审稿人对主结论的信任。</p>
    """

    extra_css = """
    .verdict { display:inline-flex; margin:18px 0 0; padding:6px 10px; border-radius:999px; background:var(--warning-bg); color:var(--warning); font-weight:650; }
    .section-rule { margin:48px 0 0; padding-top:36px; border-top:1px solid var(--border-strong); }
    .flow { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:18px 0 28px; }
    .flow-step { position:relative; min-height:116px; padding:14px; border:1px solid var(--border); border-radius:16px; background:var(--surface); }
    .flow-step:not(:last-child)::after { content:'→'; position:absolute; right:-10px; top:42%; color:var(--muted); z-index:2; }
    .flow-step b { display:block; margin-bottom:6px; font-size:12px; }
    .flow-step span { color:var(--secondary); font-size:11px; line-height:17px; }
    .risk-list { display:grid; gap:10px; margin:16px 0 0; }
    .risk { padding:15px 17px; border-left:3px solid var(--warning); border-radius:0 12px 12px 0; background:var(--surface-tertiary); }
    .risk b { display:block; margin-bottom:4px; }
    .risk p { margin:0; color:var(--secondary); }
    .next { counter-reset:step; display:grid; gap:12px; padding:0; list-style:none; }
    .next li { position:relative; padding:15px 17px 15px 54px; border:1px solid var(--border); border-radius:14px; }
    .next li::before { counter-increment:step; content:counter(step); position:absolute; left:17px; top:15px; width:24px; height:24px; display:grid; place-items:center; border-radius:50%; background:var(--blue); color:white; font-weight:700; font-size:12px; }
    .muted { color:var(--muted); }
    .callout { margin:18px 0; padding:16px 18px; border-radius:14px; background:var(--warning-bg); color:var(--secondary); }
    .callout strong { color:var(--text); }
    @media (max-width:800px) { .flow { grid-template-columns:1fr; } .flow-step:not(:last-child)::after { content:'↓'; right:18px; top:auto; bottom:-13px; } }
    """

    body = f"""
<body>
  <div class="shell">
    <header class="topbar"><div class="brand"><span class="mark" aria-hidden="true"></span>EEG Research Audit</div><div class="meta">截至 2026-07-10 · technical report</div></header>
    <main data-report-audience="technical">
      <article class="reading">
        <div class="kicker">Model architecture · experiments · publication readiness</div>
        <header data-contract-section="title"><h1>{title}</h1></header>
        <div class="verdict">总体判断：Needs revision</div>
        <p class="deck">当前结果足以支持一篇扎实的“方法诊断 + 临床 SOZ 定位”论文雏形，但还不足以支撑顶会级 SOTA 与可靠不确定性主张。</p>
        <section class="summary" data-contract-section="technical-summary"><div class="summary-label">Technical Summary</div><div class="summary-body">{technical_summary}</div></section>
        <section class="metrics">
          <div class="metric"><div class="metric-label">Private cohort</div><div class="metric-value">{tip('43 patients', source_index)}</div><div class="metric-note">{tip('119 seizures', source_index)} · patient-disjoint LOPO</div></div>
          <div class="metric"><div class="metric-label">Graph effect · Ch AUROC</div><div class="metric-value">{tip('+'+num(graph['delta']), source_v41)}</div><div class="metric-note">95% CI {tip(num(graph['low']), source_v41)} to {tip(num(graph['high']), source_v41)}</div></div>
          <div class="metric"><div class="metric-label">Set refiner effect · Ch AUROC</div><div class="metric-value">{tip('+'+num(setref['delta']), source_v41)}</div><div class="metric-note">95% CI {tip(num(setref['low']), source_v41)} to {tip(num(setref['high']), source_v41)}</div></div>
          <div class="metric"><div class="metric-label">Model complexity</div><div class="metric-value">{tip('3.71M params', source_model)}</div><div class="metric-note">{tip('303 modules', source_model)} · many interacting gates</div></div>
        </section>
      </article>

      <article class="reading" data-contract-section="key-findings">
        <section class="narrative"><h2>Graph 与 set refiner 是目前唯一足够强的主创新证据</h2><p>下图以完整私有 {tip('43-fold', source_v41)} 配对消融为准，条形表示 full minus ablation，线段是患者 bootstrap {tip('95% CI', source_v41)}。Graph 和 set refiner 在 Channel F1 与 AUROC 上均为稳定正效应；motif 的 F1 很小且不确定，AUROC 反而为负。</p></section>
      </article>
      <div class="wide"><figure class="card source-figure"><div class="card-head"><h3>Private paired ablation effects</h3><p>Full minus ablation; 43 held-out patients, macro patient effects with bootstrap intervals</p></div><div class="chart-wrap"><div data-recharts-chart="paired-ablation"><div class="chart-fallback" data-recharts-fallback>{ablation_svg(ablation_rows)}</div><div data-recharts-live aria-hidden="true"></div></div></div><figcaption class="chart-note">正值表示完整模型更好。置信区间跨零的 motif/F1 不应写成确定提升。</figcaption>{chart_source('chart-source-ablation', source_v41)}</figure></div>

      <article class="reading">
        <section class="narrative"><h2>Top-k 很高，但多阳性标签使随机基线本身就高</h2><p>私有数据平均每次发作标注 {tip('6.49/32 positive channels', source_index)} 和 {tip('3.10/5 positive regions', source_index)}。因此 Channel Top-5 随机任一命中期望已经达到 {tip(num(chance_rows[3]['random']), source_index)}；模型的 {tip(num(chance_rows[3]['observed']), source_v43)} 仍明显更好，但不能把绝对值直接解读为接近临床可用。</p></section>
      </article>
      <div class="wide"><figure class="card source-figure"><div class="card-head"><h3>Observed Top-k any-hit versus random expectation</h3><p>Private v43 baseline; random expectation computed from each seizure's label cardinality</p></div><div class="chart-wrap"><div data-recharts-chart="topk-chance"><div class="chart-fallback" data-recharts-fallback>{chance_svg(chance_rows)}</div><div data-recharts-live aria-hidden="true"></div></div></div><figcaption class="chart-note">Top-k any-hit 只要求预测集合中至少一个标签命中；应与 AUROC、AUPRC lift、compact F1、预测集合大小一起报告。</figcaption>{chart_source('chart-source-chance', source_v43 + '<br>' + source_index)}</figure></div>

      <article class="reading">
        <section class="narrative"><h2>v43 TUSZ 复现了核心模块排序，但仍只是单次 split 证据</h2><p>Graph filter 的降幅最大，set refiner 次之；motif 只在部分 ranking 指标上小幅正向。由于全表来自同一 seed、固定 eval split、仅 {tip('3 epochs', source_tusz)}，不能据此声称跨种子稳定或统计显著。</p></section>
        <section class="card table-card"><div class="card-head"><h3>v43 TUSZ core ablation</h3><p>Fixed eval split; exact test metrics, channel compact policy predicts eight labels on average</p></div><div class="table-scroll"><table><thead><tr><th>Variant</th><th>Region F1</th><th>Channel F1</th><th>Ch AUROC</th><th>Ch AUPRC lift</th><th>Avg predicted</th></tr></thead><tbody>{tusz_table_rows}</tbody></table></div></section>
      </article>

      <article class="reading section-rule" data-contract-section="scope-data-and-metric-definitions">
        <h2>数据完整，但规模与指标口径限制了外推</h2>
        <p>私有预处理写入 {tip('119/119', source_index)} 个有效发作窗口、无缺失样本文件、无重复 sample ID/患者-文件-窗口组合，全部具有 {tip('32 channels', source_index)}。LOPO 按 base patient 分组，当前代码没有看到患者跨 train/val/test 的直接泄漏。</p>
        <div class="callout"><strong>粒度定义。</strong> 私有主数是每位患者测试指标的 macro average；平均每折仅 {tip('2.77 seizures', source_v43)}，中位数 {tip('3', source_index)}。这会使单折 Top-k/F1 呈离散跳变，患者 bootstrap 比只报四位小数更合适。</div>
        <p><strong>指标。</strong> Compact F1 使用验证集选择的候选策略；AUROC 衡量全阈值排序；AUPRC lift 以阳性率为随机基线；Top-k 是任一真标签命中，不等价于集合召回或精确定位。主文应优先展示 patient-macro compact F1、AUROC/AUPRC lift、预测集合大小与 bootstrap CI。</p>
      </article>

      <article class="reading section-rule" data-contract-section="methodology">
        <h2>架构的有效主干清楚，但系统复杂度超过当前证据</h2>
        <div class="flow" aria-label="SOZ-CrossFilter architecture flow">
          <div class="flow-step"><b>EEG input window</b><span>{tip('15 s · 32 × 3000 samples · 5 s per segment · 1 s patch', source_model)}。</span></div>
          <div class="flow-step"><b>Motif tokenizer</b><span>raw waveform + FFT + five bands；codebook residual 初始强度仅 {tip('0.00034', source_model)}。</span></div>
          <div class="flow-step"><b>Criss-cross + graph</b><span>{tip('3 blocks', source_model)}；空间/时间注意力与 patch-specific high-pass graph filter。</span></div>
          <div class="flow-step"><b>Expert fusion</b><span>motif、graph、topology、multi-scale TimeFilter；跨 pre/onset/post temporal gate。</span></div>
          <div class="flow-step"><b>Channel / region output</b><span>region relation、set refiner、compact policy；Beta evidence side head。</span></div>
        </div>
        <p>当前 v43 有 {tip('3.71M trainable parameters', source_model)}、{tip('303 module objects', source_model)}，并在配置中启用约 {tip('15 non-zero top-level loss terms', source_model)}。对只有 {tip('119 private seizures', source_index)} 的任务，这会增加模块互相补偿、消融不可辨识和复现难度。建议将 paper method 收缩为“局部图对比 + 跨层候选集合精化”，motif 与 topology 作为可选正则或附录。</p>
      </article>

      <article class="reading section-rule" data-contract-section="limitations-uncertainty-and-robustness-checks">
        <h2>四个问题会阻止当前版本成为可信的顶会主结果</h2>
        <div class="risk-list">
          <div class="risk"><b>High · 跨模型协议不一致</b><p>v43 proposed 使用 {tip('10 epochs / balanced_szcore / max 8 labels', source_v43)}，baseline runner 使用 {tip('30 epochs / region_compact_f1 / max 5 labels', source_baselines)}。因此下面数值只用于暴露差距，不能当公平 leaderboard。</p></div>
          <div class="risk"><b>High · Adaptive test overfitting</b><p>输出目录中已有 {tip('180 top-level experiment directories', source_model)}，其中 {tip('96 smoke/probe/tune routes', source_model)}。私有 {tip('43 patients', source_index)} 和 TUSZ eval 结果被反复查看后，常规 test CI 不再覆盖模型选择带来的乐观偏差。</p></div>
          <div class="risk"><b>High · Uncertainty head 未训练</b><p><code>evidence_head</code> 产生 Beta alpha/beta，但 <code>crossfilter_loss</code> 没有任何项监督 <code>channel_soz_prob</code> 或 <code>channel_uncertainty</code>。当前 uncertainty 数字不可解释为校准置信度。</p></div>
          <div class="risk"><b>Medium · 稳定性证据不足</b><p>TUSZ v43 仅单 seed/split；私有 LOPO CI 只反映患者抽样，不包含随机初始化、预处理或超参数选择的不确定性。</p></div>
        </div>
        <section class="card table-card"><div class="card-head"><h3>Private cross-model numbers are not yet a leaderboard</h3><p>Different selection and candidate-set protocols; shown for audit, not for a superiority claim</p></div><div class="table-scroll"><table><thead><tr><th>Model</th><th>Region F1</th><th>Channel F1</th><th>Ch AUROC</th><th>Protocol</th></tr></thead><tbody>{comparison_html}</tbody></table></div></section>
      </article>

      <article class="reading section-rule" data-contract-section="recommended-next-steps">
        <h2>按这个顺序补实验，最接近可投顶会</h2>
        <ol class="next">
          <li><strong>锁定一个统一协议并重跑所有模型。</strong>统一数据、inner validation、selection metric、candidate policy、max labels、训练预算与 seed；至少 {tip('3–5 seeds', source_model)}，报告患者配对 bootstrap 与 seed 方差。此项完成前不要写 SOTA。</li>
          <li><strong>做一次真正“未看过”的最终评估。</strong>最优方案是独立医院/独立标注者外部测试；次优是先冻结架构和分析脚本，再对保留患者集一次性解封。当前 {tip('43 patients', source_index)} 已用于大量迭代，不能再充当严格 untouched test。</li>
          <li><strong>把主模型简化为 Graph + Set Refiner。</strong>完整重跑 full、–graph、–set、–both、轻量 backbone；motif 改为 auxiliary-only/gated optional，topology 放附录。检查交互消融，避免只做单模块 leave-one-out。</li>
          <li><strong>修复或删除 uncertainty claim。</strong>若保留，训练 evidential/ensemble uncertainty，并报告 NLL、Brier、ECE、risk–coverage、错误检出 AUROC与跨患者校准；否则从摘要和结构图删掉该贡献。</li>
          <li><strong>加入临床与效率证据。</strong>报告 hemisphere/lobe 级别、Top-k precision/recall、预测集合大小、patient-level failure cases；同时补参数、FLOPs、推理时延和显存，与 DeepSOZ/SZTrack 同硬件比较。</li>
          <li><strong>控制标签与数据偏差。</strong>补 inter-rater agreement、标签来源说明、事件/患者分布、不同阳性基数分层；在 TUSZ 与 private 上使用相同的 patient-level leakage audit。</li>
        </ol>
      </article>

      <article class="reading section-rule" data-contract-section="further-questions">
        <h2>在写论文前还需要回答的问题</h2>
        <ul>
          <li>目标任务究竟是“任一 SOZ 区域命中”、完整多标签集合恢复，还是术前临床决策支持？主指标必须对应临床决策。</li>
          <li>私有 <code>soz_bipolar</code> 标签由几位医生产生，是否有独立复核和一致性统计？</li>
          <li>TUSZ 的当前 eval 路径是否在患者/医院层面完全隔离，且从未用于历次模型版本的选择？</li>
          <li>Graph 与 set refiner 的收益是否在低/高标签基数、不同发作类型、不同患者样本数中一致？</li>
        </ul>
        <section class="caveat"><strong>审计结论。</strong>现阶段可对内分享并指导实验，但对外投稿应标为 “Needs revision”。Graph/TimeFilter 与 set refiner 的配对证据可信；SOTA、motif 普适性和 uncertainty 仍不可对外作强主张。</section>
      </article>
    </main>
  </div>
  <!-- DATA_ANALYTICS_HTML_REPORT_RUNTIME -->
</body>
</html>
"""

    template = SHELL_TEMPLATE.read_text(encoding="utf-8")
    head = template.split("<body>", 1)[0]
    head = head.replace('<html lang="en">', '<html lang="zh-CN">').replace("{{TITLE}}", title)
    head = head.replace("</style>", extra_css + "\n  </style>")
    shell = head + body
    shell_path = OUT / "report-shell.html"
    payload_path = OUT / "report-payload.json"
    shell_path.write_text(shell, encoding="utf-8")

    ablation_payload = [
        {"label": row["label"], "delta": row["delta"], "ci_low": row["low"], "ci_high": row["high"], "metric": row["metric"], "variant": row["variant"], "patients": row["n"]}
        for row in ablation_rows
    ]
    chance_payload = [
        {"metric": row["metric"], "series": series, "value": row[key], "cohort": "private v43", "label_count_adjusted": True}
        for row in chance_rows
        for series, key in (("Observed", "observed"), ("Random expectation", "random"))
    ]
    payload = {
        "charts": [
            {
                "id": "paired-ablation",
                "height": 360,
                "type": "bar",
                "settings": {"orientation": "horizontal", "groupMode": "grouped"},
                "dataset": {
                    "id": "paired-ablation",
                    "title": "Private paired ablation effects",
                    "data": ablation_payload,
                    "chart_spec": {
                        "id": "paired-ablation", "dataset": "paired-ablation", "title": "Private paired ablation effects", "type": "bar",
                        "encodings": {"x": {"field": "label", "type": "nominal"}, "y": {"field": "delta", "label": "Full minus ablation", "type": "quantitative"}, "tooltip": [{"field": "ci_low", "type": "quantitative"}, {"field": "ci_high", "type": "quantitative"}, {"field": "patients", "type": "quantitative"}]},
                        "xAxisTitle": "", "yAxisTitle": "Effect", "valueFormat": "number"
                    },
                },
            },
            {
                "id": "topk-chance",
                "height": 340,
                "type": "bar",
                "settings": {"orientation": "vertical", "groupMode": "grouped"},
                "dataset": {
                    "id": "topk-chance",
                    "title": "Observed Top-k any-hit versus random expectation",
                    "data": chance_payload,
                    "chart_spec": {
                        "id": "topk-chance", "dataset": "topk-chance", "title": "Observed Top-k any-hit versus random expectation", "type": "bar",
                        "encodings": {"x": {"field": "metric", "type": "nominal"}, "y": {"field": "value", "label": "Any-hit rate", "type": "quantitative"}, "color": {"field": "series", "type": "nominal"}},
                        "xAxisTitle": "", "yAxisTitle": "Rate", "valueFormat": "percent"
                    },
                },
            },
        ]
    }
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    source_notes = {
        "audience": "technical",
        "delivery_mode": "html",
        "report_structure": ["title", "technical-summary", "key-findings", "scope-data-and-metric-definitions", "methodology", "limitations-uncertainty-and-robustness-checks", "recommended-next-steps", "further-questions"],
        "as_of": "2026-07-10 Asia/Shanghai",
        "sources": [str(p.relative_to(ROOT)) for p in [PRIVATE_INDEX, V43_TUSZ, BASELINES]],
        "source_roots": [str(V41.relative_to(ROOT)), str(V43_PRIVATE.relative_to(ROOT))],
        "chart_map": [
            {"id": "paired-ablation", "question": "Which modules contribute reliably on private LOPO?", "family": "Uncertainty & Benchmark", "type": "signed horizontal bar with patient-bootstrap intervals", "fields": ["variant", "metric", "delta", "ci_low", "ci_high", "patients"], "takeaway": "Graph and set refiner are strong; motif is unstable.", "palette": "hard two-root cap"},
            {"id": "topk-chance", "question": "How much do multi-positive labels inflate Top-k any-hit?", "family": "Comparison & Benchmark", "type": "grouped bar", "fields": ["metric", "series", "value", "cohort"], "takeaway": "Observed performance beats chance, but absolute Top-k is inflated.", "palette": "hard two-root cap"},
        ],
        "validation_notes": [
            "All private v41 paired ablations are 43/43 folds.",
            "v43 private no_motif is still running and excluded from final claims.",
            "TUSZ v43 is single-seed descriptive evidence.",
            "Cross-model results are displayed only as a protocol-comparability warning.",
            "Uncertainty head lacks an explicit training objective in crossfilter_loss.",
        ],
        "omissions": ["No causal claim.", "No visible sources appendix by report contract.", "No trend chart because experiments are discrete variants, not time series."],
    }
    (OUT / "source_notes.json").write_text(json.dumps(source_notes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(shell_path)
    print(payload_path)


if __name__ == "__main__":
    main()
