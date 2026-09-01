#!/usr/bin/env python3
"""Build an EEG viewer for DeepSOZ-vs-LLM SOZ labels on mapped TUSZ data.

DeepSOZ's public manifest uses legacy numeric TUH identifiers, whereas TUSZ
v2.0.3 uses anonymized alphabetic identifiers.  This script maps recordings
using the complete seizure start/end sequence from DeepSOZ and the local
``*.csv_bi`` annotation.  A mapping is accepted only when exactly one local
recording is within the requested time tolerance.

By default only strict recording mappings are displayed.  ``--match-level
patient`` additionally permits an LLM recording absent from the DeepSOZ
manifest when its local patient has a mutual, unique patient mapping supported
by at least ``--min-patient-records`` strictly mapped sibling recordings.  Such
rows are visibly marked as patient-level cross-record comparisons.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import math
import re
import shutil
import sys
import urllib.request
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEEPSOZ_COMMIT = "913c921f8a08fa4df76ca0708126f565860f1068"
DEEPSOZ_URL = (
    "https://raw.githubusercontent.com/deeksha-ms/DeepSOZ/"
    f"{DEEPSOZ_COMMIT}/data/TUH_manifest_final.csv"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_LLM_SUCCESS = ROOT / "outputs/qwen35_result_index_20260723/successes.csv"
DEFAULT_OUTPUT = ROOT / "outputs/deepsoz_llm_tusz_viewer"

TCP_LEADS = (
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "A1-T3", "T3-C3", "C3-CZ", "CZ-C4", "C4-T4", "T4-A2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
)
ELECTRODE_COLUMNS = (
    "fp1", "f7", "t3", "t5", "o1", "f3", "c3", "p3", "fz", "cz",
    "pz", "fp2", "f8", "t4", "t6", "o2", "f4", "c4", "p4", "oz",
    "a1", "a2",
)


def split_tokens(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [
            str(item).strip().upper().replace("_", "-")
            for item in value
            if str(item).strip()
        ]
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip().upper().replace("_", "-") for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass
    return [x.strip().upper().replace("_", "-") for x in re.split(r"[,;|\s]+", text) if x.strip()]


def parse_number_list(value: Any) -> list[float]:
    try:
        values = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    if not isinstance(values, (list, tuple)):
        return []
    return [float(x) for x in values]


def annotation_rows(path: Path) -> list[dict[str, str]]:
    lines = [
        line for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return list(csv.DictReader(lines))


def seizure_intervals(path: Path) -> list[tuple[float, float]]:
    result = []
    for row in annotation_rows(path):
        if row.get("channel", "").upper() == "TERM" and row.get("label", "").lower() == "seiz":
            result.append((float(row["start_time"]), float(row["stop_time"])))
    return result


def max_interval_error(
    deep_starts: list[float], deep_ends: list[float], local: list[tuple[float, float]]
) -> float:
    if len(deep_starts) != len(deep_ends) or len(deep_starts) != len(local) or not local:
        return math.inf
    errors = [abs(a - b) for a, (b, _) in zip(deep_starts, local)]
    errors.extend(abs(a - b) for a, (_, b) in zip(deep_ends, local))
    return max(errors)


def local_patient(path: Path, tusz_root: Path) -> str:
    rel = path.relative_to(tusz_root)
    return rel.parts[1] if len(rel.parts) > 1 else ""


def scan_local(tusz_root: Path) -> dict[int, list[tuple[Path, list[tuple[float, float]]]]]:
    by_count: dict[int, list[tuple[Path, list[tuple[float, float]]]]] = defaultdict(list)
    for path in sorted(tusz_root.rglob("*.csv_bi")):
        try:
            intervals = seizure_intervals(path)
        except (OSError, KeyError, ValueError):
            continue
        if intervals:
            by_count[len(intervals)].append((path, intervals))
    return by_count


def load_deepsoz(source: str, cache_path: Path) -> tuple[pd.DataFrame, str]:
    candidate = Path(source).expanduser()
    if candidate.is_file():
        data = candidate.read_bytes()
    elif source.startswith(("http://", "https://")):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.is_file():
            with urllib.request.urlopen(source, timeout=60) as response, cache_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        data = cache_path.read_bytes()
    else:
        raise FileNotFoundError(f"DeepSOZ manifest not found: {source}")
    sha = hashlib.sha256(data).hexdigest()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.is_file() or cache_path.read_bytes() != data:
        cache_path.write_bytes(data)
    return pd.read_csv(cache_path), sha


def deepsoz_electrodes(row: pd.Series) -> set[str]:
    result = set()
    for column in ELECTRODE_COLUMNS:
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.notna(value) and float(value) > 0:
            result.add(column.upper())
    return result


def map_recordings(
    deep: pd.DataFrame,
    local_by_count: dict[int, list[tuple[Path, list[tuple[float, float]]]]],
    tusz_root: Path,
    tolerance: float,
) -> pd.DataFrame:
    rows = []
    for index, row in deep.iterrows():
        starts = parse_number_list(row.get("sz_starts"))
        ends = parse_number_list(row.get("sz_ends"))
        candidates = []
        for path, intervals in local_by_count.get(len(starts), []):
            error = max_interval_error(starts, ends, intervals)
            if error <= tolerance:
                candidates.append((path, error))
        status = "unmapped"
        selected: Path | None = None
        error: float | None = None
        if len(candidates) == 1:
            selected, error = candidates[0]
            status = "unique"
        elif len(candidates) > 1:
            status = "ambiguous"
        rows.append({
            "deepsoz_row": int(index),
            "deepsoz_patient": str(row.get("pt_id", "")),
            "deepsoz_record": str(row.get("fn", "")),
            "local_patient": local_patient(selected, tusz_root) if selected else "",
            "local_csv_bi": str(selected) if selected else "",
            "local_edf": str(selected.with_suffix(".edf")) if selected else "",
            "max_time_error_s": error,
            "candidate_count": len(candidates),
            "mapping_status": status,
        })
    return pd.DataFrame(rows)


def patient_mapping(record_map: pd.DataFrame, minimum_records: int) -> dict[str, str]:
    unique = record_map[record_map.mapping_status.eq("unique")]
    counts = Counter(zip(unique.deepsoz_patient.astype(str), unique.local_patient.astype(str)))
    deep_options: dict[str, set[str]] = defaultdict(set)
    local_options: dict[str, set[str]] = defaultdict(set)
    for (deep_patient, local_id), count in counts.items():
        if count >= minimum_records:
            deep_options[deep_patient].add(local_id)
            local_options[local_id].add(deep_patient)
    result = {}
    for local_id, deep_ids in local_options.items():
        if len(deep_ids) != 1:
            continue
        deep_id = next(iter(deep_ids))
        if len(deep_options[deep_id]) == 1:
            result[local_id] = deep_id
    return result


def normalize_electrode(name: str) -> str:
    value = name.upper().strip()
    value = re.sub(r"^EEG\s+", "", value)
    value = re.sub(r"-(REF|LE|AR|AVG)$", "", value)
    return value.strip()


def bipolar_data(
    raw: mne.io.BaseRaw,
    start: float,
    stop: float,
    extra_leads: Iterable[str] = (),
) -> tuple[np.ndarray, list[str], np.ndarray]:
    lookup = {normalize_electrode(name): index for index, name in enumerate(raw.ch_names)}
    first = max(0, int(math.floor(start * raw.info["sfreq"])))
    last = min(raw.n_times, int(math.ceil(stop * raw.info["sfreq"])))
    source = raw.get_data(start=first, stop=last)
    times = np.arange(first, last) / raw.info["sfreq"]
    data, names = [], []
    requested = list(TCP_LEADS)
    requested.extend(lead for lead in sorted(set(extra_leads)) if lead not in set(TCP_LEADS))
    for lead in requested:
        if lead.count("-") != 1:
            continue
        left, right = lead.split("-")
        if left in lookup and right in lookup:
            data.append(source[lookup[left]] - source[lookup[right]])
            names.append(lead)
    if not data:
        raise ValueError("no standard TCP bipolar leads could be derived from EDF")
    return np.asarray(data), names, times


def render_eeg(
    edf: Path,
    output: Path,
    center: float,
    pre: float,
    post: float,
    llm_leads: set[str],
    deep_electrodes: set[str],
    title: str,
) -> dict[str, Any]:
    raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
    start = max(0.0, center - pre)
    stop = min(float(raw.times[-1]), center + post)
    data, names, times = bipolar_data(raw, start, stop, llm_leads)
    scale = float(np.nanpercentile(np.abs(data), 95))
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    normalized = np.clip(data / scale, -4, 4)
    offsets = np.arange(len(names))[::-1] * 3.0
    fig_height = max(8.0, 0.38 * len(names) + 2.5)
    fig, ax = plt.subplots(figsize=(17, fig_height))
    for index, lead in enumerate(names):
        endpoints = set(lead.split("-"))
        in_llm = lead in llm_leads
        in_deep = bool(endpoints & deep_electrodes)
        color = "#7b2cbf" if in_llm and in_deep else "#d62728" if in_llm else "#1f77b4" if in_deep else "#333333"
        width = 1.25 if in_llm or in_deep else 0.65
        ax.plot(times, normalized[index] + offsets[index], color=color, linewidth=width)
    ax.axvline(center, color="#ff8c00", linestyle="--", linewidth=1.5, label=f"LLM t0={center:.3f}s")
    ax.set_yticks(offsets)
    labels = []
    for lead in names:
        tags = []
        if lead in llm_leads:
            tags.append("LLM")
        if set(lead.split("-")) & deep_electrodes:
            tags.append("DeepSOZ endpoint")
        labels.append(f"{lead}  [{'|'.join(tags)}]" if tags else lead)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.15)
    ax.legend(loc="upper right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return {"window_start_s": start, "window_stop_s": stop, "n_tcp_leads": len(names)}


def set_scores(predicted_leads: set[str], gold_electrodes: set[str]) -> dict[str, Any]:
    predicted_electrodes = {x for lead in predicted_leads for x in lead.split("-")}
    hits = predicted_electrodes & gold_electrodes
    precision = len(hits) / len(predicted_electrodes) if predicted_electrodes else None
    recall = len(hits) / len(gold_electrodes) if gold_electrodes else None
    f1 = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "llm_endpoint_electrodes": ";".join(sorted(predicted_electrodes)),
        "hits": ";".join(sorted(hits)),
        "missed": ";".join(sorted(gold_electrodes - predicted_electrodes)),
        "extra": ";".join(sorted(predicted_electrodes - gold_electrodes)),
        "precision": precision, "recall": recall, "f1": f1,
    }


def pct(value: Any) -> str:
    return "NA" if value is None or pd.isna(value) else f"{100 * float(value):.1f}%"


def build_html(rows: list[dict[str, Any]], summary: dict[str, Any], output: Path) -> None:
    cards = []
    for row in rows:
        warning = (
            "<div class='warning'>患者级跨记录标签：该 EDF 本身不在 DeepSOZ manifest 中。</div>"
            if row["comparison_level"] == "patient" else ""
        )
        cards.append(f"""
<section class="card">
  <h2>{html.escape(row['event_id'])}</h2>{warning}
  <div class="grid">
    <div><b>本地 EDF</b><br>{html.escape(row['local_edf'])}</div>
    <div><b>映射</b><br>{html.escape(row['mapping_description'])}</div>
    <div><b>大模型 SOZ 导联</b><br>{html.escape(row['llm_soz_leads'] or '—')}</div>
    <div><b>DeepSOZ SOZ 电极</b><br>{html.escape(row['deepsoz_soz_electrodes'] or '—')}</div>
    <div><b>命中 / 漏检 / 多报</b><br>{html.escape(row['hits'] or '—')} / {html.escape(row['missed'] or '—')} / {html.escape(row['extra'] or '—')}</div>
    <div><b>P / R / F1</b><br>{pct(row['precision'])} / {pct(row['recall'])} / {pct(row['f1'])}</div>
  </div>
  <img src="{html.escape(row['image_relpath'])}" alt="EEG {html.escape(row['event_id'])}">
</section>""")
    if not cards:
        cards.append("<section class='card'><h2>没有严格匹配事件</h2><p>当前大模型 EDF 没有对应的 DeepSOZ manifest 记录。可检查 mapping.csv，或显式使用 <code>--match-level patient</code> 查看高置信患者级跨记录参照。</p></section>")
    payload = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>DeepSOZ vs LLM TUSZ Viewer</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f5f7fa;color:#17202a}}main{{max-width:1800px;margin:auto;padding:24px}}
.card{{background:white;border-radius:10px;padding:20px;margin:18px 0;box-shadow:0 2px 10px #0001}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:12px 0}}
img{{width:100%;border:1px solid #ddd}}.warning{{background:#fff3cd;color:#664d03;padding:10px;border-radius:6px}}pre{{white-space:pre-wrap;background:#eef2f5;padding:12px}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>DeepSOZ 原始 manifest 与大模型 SOZ 对比</h1><pre>{payload}</pre>{''.join(cards)}</main></body></html>"""
    output.write_text(page, encoding="utf-8")


def local_event_onsets(csv_bi_path: Path) -> list[dict[str, Any]]:
    """Return local v2.0.3 TERM intervals and simultaneous earliest leads."""
    channel_path = csv_bi_path.with_suffix(".csv")
    term = seizure_intervals(csv_bi_path)
    channel_rows = annotation_rows(channel_path) if channel_path.is_file() else []
    result = []
    for event_index, (start, stop) in enumerate(term):
        candidates = []
        for row in channel_rows:
            if row.get("label", "").lower() in {"bckg", "null", ""}:
                continue
            row_start = float(row["start_time"])
            row_stop = float(row["stop_time"])
            if start - 1e-6 <= row_start < stop and row_stop > start:
                candidates.append(row)
        earliest = min((float(row["start_time"]) for row in candidates), default=None)
        leads = sorted({
            row["channel"].upper() for row in candidates
            if earliest is not None and abs(float(row["start_time"]) - earliest) <= 1e-4
        })
        result.append({
            "event_index": event_index,
            "start_s": start,
            "stop_s": stop,
            "earliest_channel_start_s": earliest,
            "earliest_leads": leads,
        })
    return result


def build_mapped_catalog(
    deep: pd.DataFrame,
    mapping: pd.DataFrame,
    llm: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Read all uniquely mapped EDF headers and build the 607-record catalog."""
    llm_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, row in llm.iterrows():
        source = str(Path(str(row.get("source_file", ""))).resolve())
        llm_by_source[source].append(row.to_dict())
    rows = []
    accepted = mapping[mapping.mapping_status.eq("unique")].sort_values(
        ["local_patient", "local_edf", "deepsoz_record"]
    )
    for display_index, (_, mapped) in enumerate(accepted.iterrows(), start=1):
        deep_row = deep.iloc[int(mapped.deepsoz_row)]
        edf = Path(str(mapped.local_edf)).resolve()
        csv_bi = Path(str(mapped.local_csv_bi)).resolve()
        header_error = ""
        sfreq = duration = n_channels = None
        signal_read_status = "not_attempted"
        signal_window_start_s = signal_window_stop_s = signal_rms_uv = None
        try:
            events = local_event_onsets(csv_bi)
        except Exception as exc:
            events = []
            header_error = f"annotation: {type(exc).__name__}: {exc}"
        try:
            raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
            sfreq = float(raw.info["sfreq"])
            duration = float(raw.n_times / sfreq)
            n_channels = len(raw.ch_names)
            center = float(events[0]["start_s"]) if events else min(1.0, duration / 2)
            signal_window_start_s = max(0.0, center - 1.0)
            signal_window_stop_s = min(duration, center + 2.0)
            first = int(math.floor(signal_window_start_s * sfreq))
            last = int(math.ceil(signal_window_stop_s * sfreq))
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Loading an EDF with mixed sampling frequencies.*",
                    category=RuntimeWarning,
                )
                signal = raw.get_data(start=first, stop=last)
            if signal.size and np.isfinite(signal).all():
                signal_rms_uv = float(np.sqrt(np.mean(np.square(signal))) * 1e6)
                signal_read_status = "success"
            else:
                signal_read_status = "empty_or_nonfinite"
            del raw
        except Exception as exc:  # preserve the catalog even for a bad EDF header
            header_error = f"{type(exc).__name__}: {exc}"
            signal_read_status = "failed"
        llm_rows = llm_by_source.get(str(edf), [])
        llm_events = []
        for item in llm_rows:
            llm_events.append({
                "event_id": str(item.get("event_id", "")),
                "t0_s": None if pd.isna(item.get("t0_s")) else float(item["t0_s"]),
                "soz_leads": split_tokens(item.get("soz_channels")),
            })
        rows.append({
            "display_index": display_index,
            "mapping_status": "unique",
            "max_time_error_s": mapped.max_time_error_s,
            "deepsoz_patient": str(deep_row.get("pt_id", "")),
            "deepsoz_record": str(deep_row.get("fn", "")),
            "deepsoz_soz_electrodes": ";".join(sorted(deepsoz_electrodes(deep_row))),
            "deepsoz_hemi": str(deep_row.get("hemi", "")).strip(),
            "deepsoz_region": str(deep_row.get("region", "")).strip(),
            "deepsoz_comment": str(deep_row.get("Comments", "")).strip(),
            "local_patient": str(mapped.local_patient),
            "local_edf": str(edf),
            "local_csv_bi": str(csv_bi),
            "edf_sfreq_hz": sfreq,
            "edf_duration_s": duration,
            "edf_n_channels": n_channels,
            "edf_header_error": header_error,
            "eeg_signal_read_status": signal_read_status,
            "eeg_signal_window_start_s": signal_window_start_s,
            "eeg_signal_window_stop_s": signal_window_stop_s,
            "eeg_signal_rms_uv": signal_rms_uv,
            "local_event_count": len(events),
            "local_event_starts_s": ";".join(f"{event['start_s']:.4f}" for event in events),
            "local_event_stops_s": ";".join(f"{event['stop_s']:.4f}" for event in events),
            "local_earliest_leads_by_event": " | ".join(
                f"ev{event['event_index']:04d}:{';'.join(event['earliest_leads']) or '—'}"
                for event in events
            ),
            "llm_result_status": "available" if llm_events else "missing_not_negative",
            "llm_event_count": len(llm_events),
            "llm_events_json": json.dumps(llm_events, ensure_ascii=False),
        })
    return rows


def build_catalog_html(rows: list[dict[str, Any]], summary: dict[str, Any], output: Path) -> None:
    table_rows = []
    for row in rows:
        llm_text = "无大模型结果（缺失，不是阴性）"
        llm_class = "missing"
        if row["llm_event_count"]:
            llm_values = json.loads(row["llm_events_json"])
            llm_text = " | ".join(
                f"{item['event_id']}: {','.join(item['soz_leads']) or '—'}"
                for item in llm_values
            )
            llm_class = "available"
        search = " ".join(str(row.get(key, "")) for key in (
            "deepsoz_patient", "deepsoz_record", "deepsoz_soz_electrodes",
            "deepsoz_comment", "local_patient", "local_edf",
            "local_earliest_leads_by_event", "llm_events_json",
        )).lower()
        edf_info = "ERR"
        if row["edf_sfreq_hz"] is not None and row["edf_duration_s"] is not None:
            edf_info = (
                f"{row['edf_sfreq_hz']:g} Hz<br>{row['edf_duration_s']:.1f}s / {row['edf_n_channels']}ch"
                f"<br>signal: {html.escape(row['eeg_signal_read_status'])}"
            )
        table_rows.append(f"""
<tr data-search="{html.escape(search, quote=True)}">
  <td>{row['display_index']}</td>
  <td><b>{html.escape(row['deepsoz_record'])}</b><br>pt={html.escape(row['deepsoz_patient'])}</td>
  <td><b>{html.escape(row['local_patient'])}</b><br><span class="path">{html.escape(row['local_edf'])}</span></td>
  <td><b>{html.escape(row['deepsoz_soz_electrodes'] or '—')}</b><br>{html.escape(row['deepsoz_hemi'])} / {html.escape(row['deepsoz_comment'])}</td>
  <td>{row['local_event_count']}<br><span class="small">{html.escape(row['local_event_starts_s'])}</span></td>
  <td class="small">{html.escape(row['local_earliest_leads_by_event'] or '—')}</td>
  <td class="{llm_class}">{html.escape(llm_text)}</td>
  <td>{edf_info}</td>
</tr>""")
    payload = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>DeepSOZ 607-record TUSZ Mapping Catalog</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f4f6f8;color:#17202a}}main{{padding:20px}}.summary{{background:white;padding:15px;border-radius:8px}}
.toolbar{{position:sticky;top:0;background:#f4f6f8;padding:12px 0;z-index:2}}input{{width:min(850px,90vw);padding:10px;font-size:16px}}
table{{width:100%;border-collapse:collapse;background:white;font-size:13px}}th{{position:sticky;top:67px;background:#26384a;color:white;z-index:1}}th,td{{border:1px solid #d8dee4;padding:8px;vertical-align:top;text-align:left}}
tr:nth-child(even){{background:#f8fafb}}.path{{word-break:break-all;color:#455a64}}.small{{font-size:11px;word-break:break-word}}.missing{{color:#7a5b00;background:#fff8dd}}.available{{color:#0b6b2f;background:#e8f7ed;font-weight:600}}.count{{font-weight:700;margin-left:12px}}
</style></head><body><main><h1>DeepSOZ → 本地 TUSZ v2.0.3：全部唯一映射记录</h1>
<div class="summary"><pre>{payload}</pre></div><div class="toolbar"><input id="q" placeholder="搜索旧/新患者、文件、电极、首发导联或大模型结果…"><span class="count" id="count"></span></div>
<table><thead><tr><th>#</th><th>DeepSOZ旧记录</th><th>本地v2.0.3 EDF</th><th>DeepSOZ SOZ</th><th>本地事件数/起点</th><th>本地逐事件首发导联</th><th>大模型SOZ</th><th>EDF信息</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<script>const q=document.getElementById('q'), rs=[...document.querySelectorAll('tbody tr')], c=document.getElementById('count');function f(){{const s=q.value.trim().toLowerCase();let n=0;for(const r of rs){{const ok=!s||r.dataset.search.includes(s);r.style.display=ok?'':'none';if(ok)n++}}c.textContent=`显示 ${{n}} / ${{rs.length}}`;}}q.addEventListener('input',f);f();</script>
</main></body></html>"""
    output.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepsoz-manifest", default=DEEPSOZ_URL)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--llm-success-index", type=Path, default=DEFAULT_LLM_SUCCESS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scope", choices=("llm", "all-mapped"), default="llm",
        help="llm: EEG comparison cards; all-mapped: searchable catalog of every unique DeepSOZ-to-local mapping.",
    )
    parser.add_argument("--match-level", choices=("record", "patient"), default="record")
    parser.add_argument("--mapping-tolerance-s", type=float, default=0.25)
    parser.add_argument("--min-patient-records", type=int, default=2)
    parser.add_argument("--pre-s", type=float, default=10.0)
    parser.add_argument("--post-s", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "source" / "TUH_manifest_final.csv"
    deep, deep_sha = load_deepsoz(args.deepsoz_manifest, cache)
    local_by_count = scan_local(args.tusz_root)
    mapping = map_recordings(deep, local_by_count, args.tusz_root, args.mapping_tolerance_s)
    mapping.to_csv(args.output_dir / "mapping.csv", index=False, encoding="utf-8-sig")
    patient_map = patient_mapping(mapping, args.min_patient_records)

    deep_by_row = {int(index): row for index, row in deep.iterrows()}
    record_lookup = {
        str(Path(row.local_edf).resolve()): deep_by_row[int(row.deepsoz_row)]
        for _, row in mapping[mapping.mapping_status.eq("unique")].iterrows()
    }
    deep_patient_rows = {
        str(patient): frame for patient, frame in deep.groupby(deep.pt_id.astype(str))
    }
    llm = pd.read_csv(args.llm_success_index, encoding="utf-8-sig")
    llm = llm[llm.dataset.astype(str).str.lower().eq("tusz")]
    if args.scope == "all-mapped":
        catalog_rows = build_mapped_catalog(deep, mapping, llm)
        catalog = pd.DataFrame(catalog_rows)
        catalog.to_csv(
            args.output_dir / "mapped_records.csv", index=False, encoding="utf-8-sig"
        )
        summary = {
            "deepsoz_source": args.deepsoz_manifest,
            "deepsoz_sha256": deep_sha,
            "deepsoz_rows": len(deep),
            "local_tusz_root": str(args.tusz_root.resolve()),
            "mapping_tolerance_s": args.mapping_tolerance_s,
            "record_mapping_counts": mapping.mapping_status.value_counts().to_dict(),
            "displayed_unique_mapped_records": len(catalog_rows),
            "edf_headers_read_successfully": int(catalog.edf_header_error.eq("").sum()),
            "edf_header_failures": int(catalog.edf_header_error.ne("").sum()),
            "eeg_signal_windows_read_successfully": int(catalog.eeg_signal_read_status.eq("success").sum()),
            "eeg_signal_window_failures": int(catalog.eeg_signal_read_status.ne("success").sum()),
            "records_with_llm_results": int(catalog.llm_event_count.gt(0).sum()),
            "records_without_llm_results": int(catalog.llm_event_count.eq(0).sum()),
            "missing_llm_semantics": "missing_not_negative",
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        build_catalog_html(catalog_rows, summary, args.output_dir / "index.html")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(args.output_dir / "index.html")
        return
    display_rows: list[dict[str, Any]] = []
    skipped = Counter()

    for _, llm_row in llm.iterrows():
        source = Path(str(llm_row.get("source_file", ""))).resolve()
        if not source.is_file():
            skipped["local_edf_missing"] += 1
            continue
        deep_row = record_lookup.get(str(source))
        level = "record"
        mapping_description = "strict unique recording mapping"
        if deep_row is None and args.match_level == "patient":
            try:
                local_id = source.relative_to(args.tusz_root.resolve()).parts[1]
            except (ValueError, IndexError):
                skipped["source_outside_tusz_root"] += 1
                continue
            deep_id = patient_map.get(local_id)
            frame = deep_patient_rows.get(str(deep_id)) if deep_id is not None else None
            if frame is not None:
                label_sets = {tuple(sorted(deepsoz_electrodes(row))) for _, row in frame.iterrows()}
                if len(label_sets) == 1:
                    deep_row = frame.iloc[0]
                    level = "patient"
                    evidence = int(((mapping.deepsoz_patient.astype(str) == str(deep_id)) & (mapping.local_patient == local_id) & mapping.mapping_status.eq("unique")).sum())
                    mapping_description = f"patient {local_id} ↔ legacy {deep_id}, supported by {evidence} exact sibling records"
                else:
                    skipped["deepsoz_patient_labels_inconsistent"] += 1
        if deep_row is None:
            skipped["no_accepted_deepsoz_mapping"] += 1
            continue

        llm_leads = set(split_tokens(llm_row.get("soz_channels")))
        if not llm_leads:
            skipped["empty_llm_soz"] += 1
            continue
        deep_electrodes = deepsoz_electrodes(deep_row)
        if not deep_electrodes:
            skipped["empty_deepsoz_soz"] += 1
            continue
        t0 = float(llm_row["t0_s"])
        event_id = str(llm_row["event_id"])
        image_path = args.output_dir / "eeg" / f"{event_id}.png"
        render_meta = render_eeg(
            source, image_path, t0, args.pre_s, args.post_s, llm_leads,
            deep_electrodes, f"{event_id} | DeepSOZ vs LLM",
        )
        score = set_scores(llm_leads, deep_electrodes)
        display_rows.append({
            "event_id": event_id,
            "local_patient": str(llm_row.get("patient_id", "")),
            "local_edf": str(source),
            "comparison_level": level,
            "mapping_description": mapping_description,
            "deepsoz_patient": str(deep_row.get("pt_id", "")),
            "deepsoz_record": str(deep_row.get("fn", "")) if level == "record" else "record absent; patient label only",
            "deepsoz_hemi": str(deep_row.get("hemi", "")),
            "deepsoz_comment": str(deep_row.get("Comments", "")),
            "deepsoz_soz_electrodes": ";".join(sorted(deep_electrodes)),
            "llm_soz_leads": ";".join(sorted(llm_leads)),
            "llm_t0_s": t0,
            "image_relpath": image_path.relative_to(args.output_dir).as_posix(),
            **score, **render_meta,
        })
        if args.limit and len(display_rows) >= args.limit:
            break

    pd.DataFrame(display_rows).to_csv(
        args.output_dir / "matched_events.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "deepsoz_source": args.deepsoz_manifest,
        "deepsoz_sha256": deep_sha,
        "deepsoz_rows": len(deep),
        "local_tusz_root": str(args.tusz_root.resolve()),
        "mapping_tolerance_s": args.mapping_tolerance_s,
        "record_mapping_counts": mapping.mapping_status.value_counts().to_dict(),
        "accepted_patient_mappings": len(patient_map),
        "match_level": args.match_level,
        "llm_tusz_success_rows": len(llm),
        "displayed_events": len(display_rows),
        "skipped": dict(skipped),
        "scientific_boundary": "patient fallback is cross-record patient-level reference, not event-level ground truth",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_html(display_rows, summary, args.output_dir / "index.html")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
