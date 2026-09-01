#!/usr/bin/env python3
"""Merge TUSZ weak labels and private strong labels into a canonical manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from soz_pre.constants import (  # noqa: E402
    CANONICAL_MANIFEST_FIELDS,
    FOCAL_SEIZURE_TYPES,
    REGION_LABEL_COLUMNS,
    REGION_MASK_COLUMNS,
    SPATIAL_SEIZURE_TYPE_WEIGHT,
    TCP_COLUMNS,
)
from soz_pre.label_mapping import map_tusz_row, vectors_to_manifest_fields  # noqa: E402
from soz_pre.utils import (  # noqa: E402
    clean_cell,
    normalize_hemisphere,
    parse_float,
    read_csv_rows,
    semicolon,
    write_csv_rows,
)


DEFAULT_TUSZ = "outputs/deepsoz/tusz_v203_manifest_vote_v1.csv"
DEFAULT_PRIVATE = "outputs/soz_pre/private_edf_soz_manifest.csv"
DEFAULT_OUTPUT = "outputs/soz_pre/unified_region_soz_manifest.csv"


def _tusz_spatial_weight(seizure_type: str, include_generalized: bool) -> float:
    label = clean_cell(seizure_type).lower()
    if include_generalized and label in {"gnsz", "absz", "tcsz", "tnsz", "mysz"}:
        return 0.2
    return float(SPATIAL_SEIZURE_TYPE_WEIGHT.get(label, 0.0))


def _infer_region_hemisphere(region_values: Dict[str, object]) -> str:
    left = bool(float(region_values.get("region_left_temporal", 0) or 0) > 0.5 or float(region_values.get("region_left_frontal", 0) or 0) > 0.5)
    right = bool(float(region_values.get("region_right_temporal", 0) or 0) > 0.5 or float(region_values.get("region_right_frontal", 0) or 0) > 0.5)
    if left and right:
        return "B"
    if left:
        return "L"
    if right:
        return "R"
    return ""


def canonicalize_tusz_rows(rows: Sequence[Dict[str, str]], *, include_generalized: bool) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for idx, row in enumerate(rows):
        seizure_type = clean_cell(row.get("seizure_type")).lower()
        spatial_weight = _tusz_spatial_weight(seizure_type, include_generalized=include_generalized)
        mapped = map_tusz_row(row, spatial_weight=spatial_weight)
        fields = vectors_to_manifest_fields(mapped)
        hemisphere = _infer_region_hemisphere(fields)
        quality_flags: List[str] = []
        if spatial_weight <= 0:
            quality_flags.append("tusz_temporal_only_spatial_masked")
        if seizure_type not in FOCAL_SEIZURE_TYPES:
            quality_flags.append(f"tusz_non_focal_type_{seizure_type or 'unknown'}")
        if not mapped.get("regions"):
            quality_flags.append("no_region_label")
        duration = parse_float(row.get("sz_duration"))
        start = parse_float(row.get("sz_start"))
        end = parse_float(row.get("sz_end"))
        vote_counts = dict(mapped.get("soz_region_votes", {}))
        sorted_votes = sorted((int(value) for value in vote_counts.values()), reverse=True)
        earliest_onset = parse_float(row.get("earliest_channel_onset_sec"), start)
        item: Dict[str, object] = {
            "source": "tusz",
            "split": clean_cell(row.get("split")),
            "patient_id": clean_cell(row.get("patient_id")),
            "base_patient_id": clean_cell(row.get("patient_id")),
            "edf_path": clean_cell(row.get("edf_path")),
            "event_id": clean_cell(row.get("event_id")) or f"tusz_ev{idx:06d}",
            "event_index": clean_cell(row.get("event_index")) or idx,
            "duration_sec": "",
            "t_event_marker": start,
            "t_eeg_onset": earliest_onset,
            "t_end": end,
            "sz_start": start,
            "sz_end": end,
            "sz_duration": duration,
            "seizure_type": seizure_type,
            "hemisphere": hemisphere,
            "hemisphere_label": "",
            "label_source": "tusz_soz_bipolar_plus_1s_endpoint_vote",
            "label_type": "weak_scalp_onset_region_ranking",
            "label_confidence": spatial_weight,
            "spatial_loss_weight": spatial_weight,
            "raw_label_text": clean_cell(row.get("soz_bipolar")),
            "doctor_significant_electrodes": "",
            "doctor_spread_electrodes": "",
            "onset_channels": clean_cell(row.get("onset_channels")),
            "soz_bipolar": ",".join(mapped.get("soz_bipolar", [])),
            "candidate_seizure_types": clean_cell(row.get("candidate_seizure_types")),
            "mixed_channel_seizure_types": clean_cell(row.get("mixed_channel_seizure_types")) or 0,
            "earliest_onset_channels": clean_cell(row.get("earliest_onset_channels")),
            "plus1_added_channels": clean_cell(row.get("plus1_added_channels")),
            "n_earliest_onset_channels": clean_cell(row.get("n_earliest_onset_channels")),
            "n_plus1_added_channels": clean_cell(row.get("n_plus1_added_channels")),
            "earliest_channel_onset_sec": earliest_onset,
            "onset_candidate_limit_sec": clean_cell(row.get("onset_candidate_limit_sec")),
            "onset_tolerance_sec": clean_cell(row.get("onset_tolerance_sec")) or "1",
            "soz_region": mapped.get("soz_region", ""),
            "soz_region_source": mapped.get("soz_region_source", ""),
            "soz_region_ranking": ">".join(mapped.get("soz_region_ranking", [])),
            "soz_region_top1_tied_regions": semicolon(mapped.get("soz_region_top1_tied_regions", [])),
            "soz_region_top1_tie_size": len(mapped.get("soz_region_top1_tied_regions", [])),
            "soz_region_top1_margin": (sorted_votes[0] - sorted_votes[1]) if len(sorted_votes) > 1 else "",
            "soz_region_votes_left_frontal": int(vote_counts.get("left_frontal", 0)),
            "soz_region_votes_right_frontal": int(vote_counts.get("right_frontal", 0)),
            "soz_region_votes_left_temporal": int(vote_counts.get("left_temporal", 0)),
            "soz_region_votes_right_temporal": int(vote_counts.get("right_temporal", 0)),
            "soz_region_votes_central_parietal": int(vote_counts.get("central_parietal", 0)),
            "regions": semicolon(mapped.get("regions", [])),
            "propagation_regions": "",
            "review_status": "auto_accepted" if spatial_weight > 0 else "temporal_only",
            "quality_flags": semicolon(quality_flags),
            "source_file": clean_cell(row.get("csv_path")),
            "tusz_csv_bi_path": clean_cell(row.get("csv_bi_path")),
            "tusz_montage": clean_cell(row.get("montage")),
            "tusz_session": clean_cell(row.get("session")),
            "n_onset_channels": clean_cell(row.get("n_onset_channels")),
            "n_event_channels": clean_cell(row.get("n_event_channels")),
        }
        item.update(fields)
        out.append(item)
    return out


def canonicalize_private_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        item: Dict[str, object] = {key: row.get(key, "") for key in CANONICAL_MANIFEST_FIELDS}
        for key, value in row.items():
            if key not in item:
                item[key] = value
        if not item.get("source"):
            item["source"] = "private"
        if not item.get("base_patient_id"):
            item["base_patient_id"] = clean_cell(item.get("patient_id")).rsplit("_", 1)[0]
        if not item.get("split"):
            item["split"] = "private"
        if not item.get("hemisphere"):
            item["hemisphere"] = normalize_hemisphere(row.get("致痫灶侧别", ""))
        out.append(item)
    return out


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    source_counts = Counter(clean_cell(row.get("source")) for row in rows)
    split_counts = Counter(clean_cell(row.get("split")) for row in rows)
    seizure_types = Counter(clean_cell(row.get("seizure_type")) for row in rows)
    region_counts = Counter()
    region_mask_counts = Counter()
    for row in rows:
        for col in REGION_LABEL_COLUMNS:
            if parse_float(row.get(col), 0.0) > 0.5:
                region_counts[col.replace("region_", "")] += 1
        for col in REGION_MASK_COLUMNS:
            if parse_float(row.get(col), 0.0) > 0.5:
                region_mask_counts[col.replace("region_mask_", "")] += 1
    return {
        "n_rows": len(rows),
        "source_counts": dict(source_counts),
        "split_counts": dict(split_counts),
        "seizure_type_counts": dict(seizure_types),
        "region_positive_counts": dict(region_counts),
        "region_mask_counts": dict(region_mask_counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical TUSZ+private region SOZ manifest")
    parser.add_argument("--tusz_manifest", default=DEFAULT_TUSZ)
    parser.add_argument("--private_manifest", default=DEFAULT_PRIVATE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--include_generalized_spatial", action="store_true", help="Give generalized TUSZ events low-weight spatial labels")
    parser.add_argument("--order", choices=["tusz-first", "private-first"], default="tusz-first")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows: List[Dict[str, object]] = []
    tusz_rows: List[Dict[str, object]] = []
    private_rows: List[Dict[str, object]] = []
    if args.tusz_manifest and Path(args.tusz_manifest).is_file():
        tusz_rows = canonicalize_tusz_rows(
            read_csv_rows(Path(args.tusz_manifest)),
            include_generalized=bool(args.include_generalized_spatial),
        )
    if args.private_manifest and Path(args.private_manifest).is_file():
        private_rows = canonicalize_private_rows(read_csv_rows(Path(args.private_manifest)))
    all_rows = private_rows + tusz_rows if args.order == "private-first" else tusz_rows + private_rows
    output = Path(args.output)
    write_csv_rows(output, all_rows, CANONICAL_MANIFEST_FIELDS)
    summary = summarize(all_rows)
    summary.update({
        "tusz_manifest": args.tusz_manifest,
        "private_manifest": args.private_manifest,
        "include_generalized_spatial": bool(args.include_generalized_spatial),
    })
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(all_rows)} rows to {output}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
