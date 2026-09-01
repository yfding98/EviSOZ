#!/usr/bin/env python3
"""Audit sample-level SOZ channel labels and their training-time region mapping."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.tfm_soz.constants import (
    FULL_TCP_CHANNELS,
    REGION_NAMES,
    REGION_TO_CHANNELS,
    label_vector_from_soz_bipolar,
    region_vector_from_channel_labels,
    unknown_soz_tokens,
)


REGION_CN = {
    "left_frontal": "左额区",
    "right_frontal": "右额区",
    "left_temporal": "左颞区",
    "right_temporal": "右颞区",
    "central_parietal": "中央-顶叶区",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("private_sz_union_relabel_manifest_0622_fix.csv"),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("outputs/tfm_soz/private_0622_fix_rows119_segments_15s/index.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tfm_soz/soz_label_distribution_0622_fix"),
    )
    return parser.parse_args()


def describe_counts(values: np.ndarray) -> dict[str, object]:
    counter = Counter(int(value) for value in values.tolist())
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "min": int(values.min()),
        "max": int(values.max()),
        "total_positive": int(values.sum()),
        "distribution": {str(key): value for key, value in sorted(counter.items())},
    }


def percent(count: int, total: int) -> str:
    return f"{100.0 * count / total:.2f}%"


def main() -> None:
    args = parse_args()
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    with args.index.open("r", encoding="utf-8-sig", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    if len(manifest_rows) != len(index_rows):
        raise ValueError(
            f"Manifest/index row mismatch: {len(manifest_rows)} != {len(index_rows)}"
        )

    channel_labels: list[np.ndarray] = []
    region_labels: list[np.ndarray] = []
    sample_rows: list[dict[str, object]] = []
    unknown_counter: Counter[str] = Counter()
    index_mismatches: list[int] = []
    npz_mismatches: list[int] = []

    for manifest_row, (row, index_row) in enumerate(zip(manifest_rows, index_rows)):
        channel_y = np.asarray(
            label_vector_from_soz_bipolar(row.get("soz_bipolar")), dtype=np.int64
        )
        region_y = np.asarray(
            region_vector_from_channel_labels(channel_y), dtype=np.int64
        )
        channel_labels.append(channel_y)
        region_labels.append(region_y)
        unknown_counter.update(unknown_soz_tokens(row.get("soz_bipolar")))

        if (
            int(index_row["manifest_row"]) != manifest_row
            or int(index_row["n_positive"]) != int(channel_y.sum())
            or int(index_row["n_positive_regions"]) != int(region_y.sum())
        ):
            index_mismatches.append(manifest_row)

        npz_path = args.index.parent / index_row["npz_path"]
        with np.load(npz_path, allow_pickle=True) as npz:
            stored_channel_y = np.asarray(npz["y_segments"])[1]
            stored_region_y = np.asarray(npz["region_y_segments"])[1]
        if not (
            np.array_equal(channel_y, stored_channel_y)
            and np.array_equal(region_y, stored_region_y)
        ):
            npz_mismatches.append(manifest_row)

        active_channels = [
            channel for channel, active in zip(FULL_TCP_CHANNELS, channel_y) if active
        ]
        active_regions = [
            region for region, active in zip(REGION_NAMES, region_y) if active
        ]
        grouped_channels = []
        for region in active_regions:
            active_in_region = [
                channel
                for channel in REGION_TO_CHANNELS[region]
                if channel in active_channels
            ]
            grouped_channels.append(
                f"{REGION_CN[region]}({region}):{','.join(active_in_region)}"
            )
        sample_rows.append(
            {
                "manifest_row": manifest_row,
                "sample_id": index_row["sample_id"],
                "patient_id": row.get("patient_id", ""),
                "base_patient_id": index_row.get("base_patient_id", ""),
                "edf_path": row.get("edf_path", ""),
                "sz_start": row.get("sz_start", ""),
                "sz_end": row.get("sz_end", ""),
                "soz_bipolar": row.get("soz_bipolar", ""),
                "n_soz_channels": int(channel_y.sum()),
                "soz_regions": ";".join(active_regions),
                "soz_regions_cn": ";".join(REGION_CN[r] for r in active_regions),
                "n_soz_regions": int(region_y.sum()),
                "active_channels_by_region": " | ".join(grouped_channels),
            }
        )

    channel_matrix = np.stack(channel_labels)
    region_matrix = np.stack(region_labels)
    channel_counts = channel_matrix.sum(axis=1)
    region_counts = region_matrix.sum(axis=1)
    region_combinations = Counter(
        "+".join(np.asarray(REGION_NAMES)[row.astype(bool)]) for row in region_matrix
    )
    patients = {row["base_patient_id"] for row in sample_rows}

    summary = {
        "manifest": str(args.manifest),
        "training_index": str(args.index),
        "label_source": "soz_bipolar",
        "region_rule": "OR/max pooling over each fixed REGION_TO_CHANNELS group",
        "n_samples": len(sample_rows),
        "n_patients": len(patients),
        "channel_count_per_sample": describe_counts(channel_counts),
        "region_count_per_sample": describe_counts(region_counts),
        "channel_positive_slot_rate": float(channel_matrix.mean()),
        "region_positive_slot_rate": float(region_matrix.mean()),
        "channel_frequency": {
            channel: int(channel_matrix[:, idx].sum())
            for idx, channel in enumerate(FULL_TCP_CHANNELS)
        },
        "region_frequency": {
            region: int(region_matrix[:, idx].sum())
            for idx, region in enumerate(REGION_NAMES)
        },
        "region_combinations": dict(region_combinations.most_common()),
        "unknown_soz_tokens": dict(unknown_counter),
        "index_mismatch_rows": index_mismatches,
        "npz_mismatch_rows": npz_mismatches,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = args.output_dir / "sample_region_mapping.csv"
    with mapping_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    report_lines = [
        "# SOZ 标签分布与训练脑区映射",
        "",
        f"- 样本数：{len(sample_rows)}；患者数：{len(patients)}。",
        (
            f"- 每样本 SOZ 通道数：均值 {channel_counts.mean():.3f}，中位数 "
            f"{np.median(channel_counts):.0f}，范围 {channel_counts.min()}–{channel_counts.max()}。"
        ),
        (
            f"- 每样本 SOZ 脑区数：均值 {region_counts.mean():.3f}，中位数 "
            f"{np.median(region_counts):.0f}，范围 {region_counts.min()}–{region_counts.max()}。"
        ),
        (
            f"- 通道标签阳性槽位率：{percent(int(channel_matrix.sum()), channel_matrix.size)}；"
            f"脑区标签阳性槽位率：{percent(int(region_matrix.sum()), region_matrix.size)}。"
        ),
        "- 脑区标签仅由 `soz_bipolar` 按固定通道组 OR/max pooling 得到；未使用标注表脑区字段。",
        "",
        "## 固定映射",
        "",
        "| 脑区 | 纳入的双极通道 |",
        "|---|---|",
    ]
    report_lines.extend(
        f"| {REGION_CN[region]} (`{region}`) | {', '.join(REGION_TO_CHANNELS[region])} |"
        for region in REGION_NAMES
    )
    report_lines.extend(
        [
            "",
            "## 每样本映射结果",
            "",
            "| row | sample_id | SOZ通道数 | 映射后SOZ脑区数 | 映射后SOZ脑区 |",
            "|---:|---|---:|---:|---|",
        ]
    )
    report_lines.extend(
        f"| {row['manifest_row']} | {row['sample_id']} | {row['n_soz_channels']} | "
        f"{row['n_soz_regions']} | {row['soz_regions_cn']} |"
        for row in sample_rows
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
