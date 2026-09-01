#!/usr/bin/env python3
"""PyTorch dataset for preprocessed heterogeneous SOZ NPZ sequences."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from soz_pre.constants import HEMISPHERE_CLASSES, REGION_NAMES, TCP_CHANNELS
    from soz_pre.utils import clean_cell
except ImportError:  # pragma: no cover - package import fallback
    from code.soz_pre.constants import HEMISPHERE_CLASSES, REGION_NAMES, TCP_CHANNELS
    from code.soz_pre.utils import clean_cell


SOURCE_ID = {"tusz": 0, "private": 1}
SOURCE_NAME = {0: "tusz", 1: "private", 2: "other"}


def read_index(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _matches(value: str, allowed: Optional[set[str]]) -> bool:
    return allowed is None or clean_cell(value).lower() in allowed


class UnifiedSOZDataset(Dataset):
    def __init__(
        self,
        preprocessed_dir: str | Path,
        *,
        splits: Optional[Sequence[str]] = None,
        sources: Optional[Sequence[str]] = None,
        include_patients: Optional[Sequence[str]] = None,
        exclude_patients: Optional[Sequence[str]] = None,
        index_name: str = "index.csv",
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        index_path = self.preprocessed_dir / index_name
        if not index_path.is_file():
            raise FileNotFoundError(f"Index not found: {index_path}")
        rows = read_index(index_path)
        split_filter = {item.lower() for item in splits} if splits else None
        source_filter = {item.lower() for item in sources} if sources else None
        include = set(include_patients or [])
        exclude = set(exclude_patients or [])
        filtered: List[Dict[str, str]] = []
        for row in rows:
            split = clean_cell(row.get("split")).lower()
            source = clean_cell(row.get("source")).lower()
            patient = clean_cell(row.get("base_patient_id")) or clean_cell(row.get("patient_id"))
            if not _matches(split, split_filter):
                continue
            if not _matches(source, source_filter):
                continue
            if include and patient not in include:
                continue
            if exclude and patient in exclude:
                continue
            filtered.append(row)
        if not filtered:
            raise ValueError(
                f"No samples found in {index_path} for splits={splits}, "
                f"sources={sources}, include={include_patients}, exclude={exclude_patients}"
            )
        self.rows = filtered
        self.segment_meta: List[Dict[str, object]] = []
        self.sample_shape = None
        channel_labels = []
        channel_masks = []
        region_labels = []
        region_masks = []
        propagation_labels = []
        propagation_masks = []
        hemisphere_labels = []
        hemisphere_masks = []
        seizure_labels = []
        sample_weights = []
        source_ids = []
        input_masks = []
        artifact_scores = []
        artifact_masks = []
        for row in self.rows:
            npz_path = self.preprocessed_dir / row["npz_path"]
            with np.load(npz_path, allow_pickle=True) as npz:
                if self.sample_shape is None:
                    x_shape = tuple(npz["x"].shape)
                    self.sample_shape = x_shape
                else:
                    x_shape = self.sample_shape
                channel_labels.append(npz["channel_labels"].astype(np.float32))
                channel_masks.append(npz["channel_label_mask"].astype(np.float32))
                region_labels.append(npz["region_labels"].astype(np.float32))
                region_masks.append(npz["region_label_mask"].astype(np.float32))
                propagation_labels.append(npz["propagation_region_labels"].astype(np.float32))
                propagation_masks.append(npz["propagation_region_mask"].astype(np.float32))
                hemisphere_labels.append(int(npz["hemisphere_label"]))
                hemisphere_masks.append(float(npz["hemisphere_mask"]))
                seizure_labels.append(npz["seizure_y"].astype(np.float32))
                sample_weights.append(float(npz["sample_weight"]))
                input_masks.append(npz["input_channel_mask"].astype(np.float32))
                source = clean_cell(row.get("source")).lower()
                source_ids.append(SOURCE_ID.get(source, 2))
                if "artifact_score" in npz:
                    artifact_score = npz["artifact_score"].astype(np.float32)
                else:
                    artifact_score = np.zeros(x_shape[:2], dtype=np.float32)
                if "artifact_mask" in npz:
                    artifact_mask = npz["artifact_mask"].astype(np.float32)
                else:
                    artifact_mask = np.zeros(x_shape[:2], dtype=np.float32)
                if artifact_score.shape != x_shape[:2]:
                    raise ValueError(f"{npz_path} artifact_score shape {artifact_score.shape} != x[:2] {x_shape[:2]}")
                if artifact_mask.shape != x_shape[:2]:
                    raise ValueError(f"{npz_path} artifact_mask shape {artifact_mask.shape} != x[:2] {x_shape[:2]}")
                artifact_scores.append(artifact_score)
                artifact_masks.append(artifact_mask)
            self.segment_meta.append({
                "patient_id": row.get("patient_id", ""),
                "base_patient_id": row.get("base_patient_id", ""),
                "source": row.get("source", ""),
                "split": row.get("split", ""),
                "event_id": row.get("event_id", ""),
                "sample_id": row.get("sample_id", ""),
                "sample_role": row.get("sample_role", ""),
                "edf_path": row.get("edf_path", ""),
                "sz_start": row.get("sz_start", ""),
                "sz_end": row.get("sz_end", ""),
                "hemisphere": row.get("hemisphere", ""),
                "artifact_score_mean": row.get("artifact_score_mean", ""),
                "artifact_score_max": row.get("artifact_score_max", ""),
            })
        self.channel_labels_np = np.stack(channel_labels).astype(np.float32)
        self.channel_masks_np = np.stack(channel_masks).astype(np.float32)
        self.region_labels_np = np.stack(region_labels).astype(np.float32)
        self.region_masks_np = np.stack(region_masks).astype(np.float32)
        self.propagation_labels_np = np.stack(propagation_labels).astype(np.float32)
        self.propagation_masks_np = np.stack(propagation_masks).astype(np.float32)
        self.hemisphere_labels_np = np.asarray(hemisphere_labels, dtype=np.int64)
        self.hemisphere_masks_np = np.asarray(hemisphere_masks, dtype=np.float32)
        self.seizure_y_np = np.stack(seizure_labels).astype(np.float32)
        self.seizure_mask_np = np.ones_like(self.seizure_y_np, dtype=np.float32)
        self.sample_weights_np = np.asarray(sample_weights, dtype=np.float32)
        self.source_ids_np = np.asarray(source_ids, dtype=np.int64)
        self.input_masks_np = np.stack(input_masks).astype(np.float32)
        self.artifact_scores_np = np.stack(artifact_scores).astype(np.float32)
        self.artifact_masks_np = np.stack(artifact_masks).astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        with np.load(self.preprocessed_dir / row["npz_path"], allow_pickle=True) as npz:
            x = npz["x"].astype(np.float32)
            if "artifact_score" in npz:
                artifact_score = npz["artifact_score"].astype(np.float32)
            else:
                artifact_score = np.zeros(x.shape[:2], dtype=np.float32)
            if "artifact_mask" in npz:
                artifact_mask = npz["artifact_mask"].astype(np.float32)
            else:
                artifact_mask = np.zeros(x.shape[:2], dtype=np.float32)
            return {
                "x": torch.from_numpy(x),
                "input_mask": torch.from_numpy(npz["input_channel_mask"].astype(np.float32)),
                "artifact_score": torch.from_numpy(artifact_score),
                "artifact_mask": torch.from_numpy(artifact_mask),
                "channel_y": torch.from_numpy(npz["channel_labels"].astype(np.float32)),
                "channel_mask": torch.from_numpy(npz["channel_label_mask"].astype(np.float32)),
                "region_y": torch.from_numpy(npz["region_labels"].astype(np.float32)),
                "region_mask": torch.from_numpy(npz["region_label_mask"].astype(np.float32)),
                "propagation_y": torch.from_numpy(npz["propagation_region_labels"].astype(np.float32)),
                "propagation_mask": torch.from_numpy(npz["propagation_region_mask"].astype(np.float32)),
                "seizure_y": torch.from_numpy(npz["seizure_y"].astype(np.float32)),
                "seizure_mask": torch.from_numpy(npz["seizure_mask"].astype(np.float32)),
                "hemisphere_y": torch.tensor(int(npz["hemisphere_label"]), dtype=torch.long),
                "hemisphere_mask": torch.tensor(float(npz["hemisphere_mask"]), dtype=torch.float32),
                "sample_weight": torch.tensor(float(npz["sample_weight"]), dtype=torch.float32),
                "source_id": torch.tensor(int(self.source_ids_np[idx]), dtype=torch.long),
                "index": torch.tensor(idx, dtype=torch.long),
            }

    @property
    def n_windows(self) -> int:
        return int(self.sample_shape[0])

    @property
    def n_input_channels(self) -> int:
        return int(self.sample_shape[1])

    @property
    def window_samples(self) -> int:
        return int(self.sample_shape[2])

    @property
    def n_label_channels(self) -> int:
        return len(TCP_CHANNELS)

    @property
    def n_regions(self) -> int:
        return len(REGION_NAMES)

    @property
    def n_hemisphere_classes(self) -> int:
        return len(HEMISPHERE_CLASSES)

    @property
    def patients(self) -> List[str]:
        return sorted({str(meta.get("base_patient_id", "")) for meta in self.segment_meta})


def list_private_patients(preprocessed_dir: str | Path) -> List[str]:
    rows = read_index(Path(preprocessed_dir) / "index.csv")
    return sorted({
        clean_cell(row.get("base_patient_id")) or clean_cell(row.get("patient_id"))
        for row in rows
        if clean_cell(row.get("source")).lower() == "private"
    })
