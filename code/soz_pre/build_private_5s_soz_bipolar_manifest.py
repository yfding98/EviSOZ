#!/usr/bin/env python3
"""Build private SOZ 5s context segments from the latest annotation manifest.

The private annotation marks the seizure-start interval [sz_start, sz_start+5).
Only ``soz_bipolar`` is treated as the SOZ label source. For each eligible
event this script emits:

* pre_soz_negative:   [sz_start-5, sz_start), all SOZ labels 0
* onset_soz_positive: [sz_start, sz_start+5), labels from soz_bipolar
* post_soz_negative:  [sz_start+5, sz_start+10), all SOZ labels 0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from soz_pre.constants import (  # noqa: E402
    CANONICAL_MANIFEST_FIELDS,
    CHANNEL_LABEL_MASK_COLUMNS,
    CHANNEL_PROP_COLUMNS,
    CHANNEL_TO_REGIONS,
    HEMISPHERE_INDEX,
    REGION_LABEL_COLUMNS,
    REGION_MASK_COLUMNS,
    REGION_NAMES,
    REGION_PROP_COLUMNS,
    TCP_CHANNELS,
    TCP_COLUMNS,
)
from soz_pre.utils import (  # noqa: E402
    base_patient_id as infer_base_patient_id,
    canonical_bipolar_token,
    clean_cell,
    normalize_hemisphere,
    parse_float,
    read_csv_rows,
    split_label_tokens,
    write_csv_rows,
)


DEFAULT_INPUT = "private_sz_union_relabel_manifest_0622_fix.csv"
DEFAULT_OUTPUT = "outputs/soz_pre/private_sz_union_relabel_manifest_0622_fix_5s_soz_bipolar.csv"


def _parse_soz_bipolar(value: object) -> Tuple[List[str], List[str]]:
    channels: List[str] = []
    dropped: List[str] = []
    for token in split_label_tokens(value):
        channel = canonical_bipolar_token(token)
        if channel is None:
            dropped.append(clean_cell(token))
            continue
        if channel not in channels:
            channels.append(channel)
    return channels, dropped


def _empty_label_fields(*, label_mask: float, region_mask: float) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for col in TCP_COLUMNS:
        out[col] = 0
    for col in CHANNEL_LABEL_MASK_COLUMNS:
        out[col] = float(label_mask)
    for col in CHANNEL_PROP_COLUMNS:
        out[col] = 0
    for col in REGION_LABEL_COLUMNS:
        out[col] = 0
    for col in REGION_MASK_COLUMNS:
        out[col] = float(region_mask)
    for col in REGION_PROP_COLUMNS:
        out[col] = 0
    return out


def _positive_label_fields(channels: Sequence[str]) -> Dict[str, object]:
    out = _empty_label_fields(label_mask=1.0, region_mask=1.0)
    regions: List[str] = []
    channel_set = set(channels)
    for channel, col in zip(TCP_CHANNELS, TCP_COLUMNS):
        out[col] = int(channel in channel_set)
        if channel not in channel_set:
            continue
        for region in CHANNEL_TO_REGIONS.get(channel, ()):
            if region not in regions:
                regions.append(region)
    for region, col in zip(REGION_NAMES, REGION_LABEL_COLUMNS):
        out[col] = int(region in regions)
    return out


def _row_duration(row: Dict[str, str]) -> float:
    for key in ("duration_sec", "duration", "record_duration", "file_duration"):
        duration = parse_float(row.get(key))
        if np.isfinite(duration) and duration > 0:
            return float(duration)
    return float("nan")


def _event_id(row: Dict[str, str], idx: int) -> str:
    return clean_cell(row.get("event_id")) or f"{clean_cell(row.get('patient_id')) or 'private'}_row{idx:04d}"


def _base_row(
    row: Dict[str, str],
    *,
    idx: int,
    role: str,
    segment_start: float,
    segment_sec: float,
    original_start: float,
    original_end: float,
    duration: float,
    original_soz: str,
    dropped: Sequence[str],
    negative_weight: float,
) -> Dict[str, object]:
    patient_id = clean_cell(row.get("patient_id"))
    base_pid = clean_cell(row.get("base_patient_id")) or infer_base_patient_id(patient_id)
    is_positive = role == "onset_soz_positive"
    hemi = normalize_hemisphere(row.get("hemisphere")) if is_positive else ""
    item: Dict[str, object] = {field: "" for field in CANONICAL_MANIFEST_FIELDS}
    for field in CANONICAL_MANIFEST_FIELDS:
        if field in row:
            item[field] = row.get(field, "")

    item.update({
        "source": "private",
        "split": clean_cell(row.get("split")) or "private",
        "patient_id": patient_id,
        "base_patient_id": base_pid,
        "edf_path": clean_cell(row.get("edf_path")),
        "event_id": f"{_event_id(row, idx)}_{role}",
        "event_index": idx,
        "duration_sec": duration if np.isfinite(duration) else "",
        "t_event_marker": segment_start,
        "t_eeg_onset": segment_start,
        "t_end": segment_start + segment_sec,
        "sz_start": segment_start,
        "sz_end": segment_start + segment_sec,
        "sz_duration": segment_sec,
        "seizure_type": clean_cell(row.get("seizure_type")),
        "hemisphere": hemi,
        "hemisphere_label": HEMISPHERE_INDEX.get(hemi, "") if hemi else "",
        "label_source": "private_soz_bipolar_only",
        "label_type": "private_onset_5s_soz" if is_positive else "private_context_5s_non_soz",
        "label_confidence": 1.0,
        "spatial_loss_weight": 1.0 if is_positive else float(negative_weight),
        "raw_label_text": original_soz,
        "doctor_significant_electrodes": "",
        "doctor_spread_electrodes": "",
        "onset_channels": "",
        "soz_bipolar": "",
        "regions": "",
        "propagation_regions": "",
        "review_status": clean_cell(row.get("review_status")) or "auto_5s_context",
        "quality_flags": clean_cell(row.get("quality_flags")),
        "source_file": clean_cell(row.get("source_file")) or DEFAULT_INPUT,
        "sample_role": role,
        "orig_sz_start": original_start,
        "orig_sz_end": original_end,
        "orig_soz_bipolar": original_soz,
        "dropped_soz_bipolar": ",".join(dropped),
    })
    return item


def build_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    source_rows = read_csv_rows(Path(args.input))
    out: List[Dict[str, object]] = []
    stats = Counter()
    positive_channels = Counter()
    positive_regions = Counter()
    dropped_tokens = Counter()
    segment_sec = float(args.segment_sec)
    negative_weight = float(args.negative_weight)

    for idx, row in enumerate(source_rows):
        if clean_cell(row.get("source")).lower() not in {"", "private"}:
            stats["skipped_non_private"] += 1
            continue
        start = parse_float(row.get("sz_start"))
        if not np.isfinite(start):
            stats["skipped_missing_sz_start"] += 1
            continue
        original_end = start + segment_sec
        duration = _row_duration(row)
        channels, dropped = _parse_soz_bipolar(row.get("soz_bipolar"))
        for token in dropped:
            dropped_tokens[token] += 1
        if not channels:
            stats["skipped_no_modeled_soz_bipolar"] += 1
            continue

        roles: List[Tuple[str, float]] = []
        pre_start = start - segment_sec
        if pre_start >= 0.0:
            roles.append(("pre_soz_negative", pre_start))
        else:
            stats["skipped_pre_out_of_bounds"] += 1
        roles.append(("onset_soz_positive", start))
        post_start = start + segment_sec
        if not np.isfinite(duration) or post_start + segment_sec <= duration + 1e-6:
            roles.append(("post_soz_negative", post_start))
        else:
            stats["skipped_post_out_of_bounds"] += 1

        for role, segment_start in roles:
            if np.isfinite(duration) and segment_start + segment_sec > duration + 1e-6:
                stats[f"skipped_{role}_past_duration"] += 1
                continue
            item = _base_row(
                row,
                idx=idx,
                role=role,
                segment_start=segment_start,
                segment_sec=segment_sec,
                original_start=start,
                original_end=original_end,
                duration=duration,
                original_soz=clean_cell(row.get("soz_bipolar")),
                dropped=dropped,
                negative_weight=negative_weight,
            )
            if role == "onset_soz_positive":
                item.update(_positive_label_fields(channels))
                item["soz_bipolar"] = ",".join(channels)
                region_names = [
                    region
                    for region, col in zip(REGION_NAMES, REGION_LABEL_COLUMNS)
                    if int(item[col]) > 0
                ]
                item["regions"] = ";".join(region_names)
                for channel in channels:
                    positive_channels[channel] += 1
                for region in region_names:
                    positive_regions[region] += 1
            else:
                item.update(_empty_label_fields(label_mask=1.0, region_mask=1.0))
            out.append(item)
            stats[f"role_{role}"] += 1

    patient_ids = {
        clean_cell(row.get("base_patient_id")) or clean_cell(row.get("patient_id"))
        for row in out
    }
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "input_rows": len(source_rows),
        "output_rows": len(out),
        "patients": len({pid for pid in patient_ids if pid}),
        "segment_sec": segment_sec,
        "negative_weight": negative_weight,
        "stats": dict(stats),
        "positive_channel_counts": dict(positive_channels),
        "positive_region_counts": dict(positive_regions),
        "dropped_soz_bipolar_tokens": dict(dropped_tokens),
        "label_rule": "SOZ positives use soz_bipolar only; onset_channels is blanked and never used.",
    }
    return out, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build private 5s SOZ-bipolar segment manifest")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--segment_sec", type=float, default=5.0)
    parser.add_argument(
        "--negative_weight",
        type=float,
        default=0.5,
        help="Spatial sample weight for pre/post context negatives in NPZ preprocessing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = build_rows(args)
    output = Path(args.output)
    write_csv_rows(output, rows, CANONICAL_MANIFEST_FIELDS)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
