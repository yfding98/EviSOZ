#!/usr/bin/env python3
"""VEPiSet adapters for SOZ-like IED localization experiments.

The VEPiSet release stores each 4 s EEG window as a 29-channel ``.npy`` file
under class folders. These helpers expose that data through the same sample
interfaces used by the local SOZ trainers without copying the 17 GB dataset.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from soz_pre.constants import TCP_CHANNELS, TCP_PAIRS
except ImportError:  # pragma: no cover
    from code.soz_pre.constants import TCP_CHANNELS, TCP_PAIRS


VEP_CHANNELS_29: Tuple[str, ...] = (
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ",
    "PG1", "PG2", "A1", "A2", "ECG1", "ECG2", "EMG1", "EMG2", "EMG3", "EMG4",
)
VEP_MONOPOLAR_19: Tuple[str, ...] = VEP_CHANNELS_29[:19]
VEP_MONOPOLAR_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(VEP_CHANNELS_29)}

VEP_CLASS_DIRS: Tuple[Tuple[str, int], ...] = (
    ("Non-IED", 0),
    ("Generalized-IED", 1),
    ("Frontal-IED", 2),
    ("Temporal-IED", 3),
    ("Centro-Parietal-IED", 4),
    ("Occipital-IED", 5),
)
VEP_CLASS_NAME_BY_ID: Dict[int, str] = {idx: name for name, idx in VEP_CLASS_DIRS}
VEP_CLASS_ID_BY_DIR: Dict[str, int] = {name: idx for name, idx in VEP_CLASS_DIRS}
VEP_STATE_NAME_BY_ID: Dict[int, str] = {
    0: "unknown",
    1: "waking",
    2: "sleeping",
}
VEP_STATE_ID_BY_NAME: Dict[str, int] = {name: idx for idx, name in VEP_STATE_NAME_BY_ID.items()}

VEP_SOZ_REGION_NAMES: Tuple[str, ...] = (
    "frontal",
    "temporal",
    "centro_parietal",
    "occipital",
)
VEP_BRAIN_REGION_NAMES: Tuple[str, ...] = ("FP", "F", "C", "T", "P", "O")

REGION_ELECTRODES: Dict[str, Tuple[str, ...]] = {
    "frontal": ("FP1", "FP2", "F3", "F4", "F7", "F8", "FZ"),
    "temporal": ("F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2"),
    "centro_parietal": ("C3", "C4", "CZ", "P3", "P4", "PZ"),
    "occipital": ("O1", "O2"),
}
CLASS_TO_REGIONS: Dict[int, Tuple[str, ...]] = {
    0: (),
    1: VEP_SOZ_REGION_NAMES,
    2: ("frontal",),
    3: ("temporal",),
    4: ("centro_parietal",),
    5: ("occipital",),
}
CLASS_TO_BRAIN_REGIONS: Dict[int, Tuple[str, ...]] = {
    0: (),
    1: VEP_BRAIN_REGION_NAMES,
    2: ("FP", "F"),
    3: ("T",),
    4: ("C", "P"),
    5: ("O",),
}
CLASS_TO_ELECTRODES: Dict[int, Tuple[str, ...]] = {
    class_id: tuple(
        dict.fromkeys(
            electrode
            for region in regions
            for electrode in REGION_ELECTRODES[region]
        )
    )
    for class_id, regions in CLASS_TO_REGIONS.items()
}

# Order used by code/models/manifest_dataset.py, brain_network_extractor.py and
# labram_timefilter_soz.py.
BRAIN_TCP_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("A1", "T3"), ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"),
    ("C4", "T4"), ("T4", "A2"),
)
BRAIN_TCP_CHANNELS: Tuple[str, ...] = tuple(f"{a}-{b}" for a, b in BRAIN_TCP_PAIRS)


def _parse_vepiset_name(path: Path) -> Dict[str, object]:
    left, raw_label = path.stem.rsplit("__", 1)
    patient_id, raw_start, raw_end, raw_fs = left.rsplit("_", 3)
    return {
        "patient_id": patient_id,
        "start_sample": int(raw_start),
        "end_sample": int(raw_end),
        "fs": float(raw_fs),
        "class_id": int(raw_label),
    }


@lru_cache(maxsize=8)
def load_vepiset_state_events(root_value: str) -> Dict[str, Tuple[Tuple[float, int], ...]]:
    """Load wake/sleep state events from MAT files without reading EEG arrays."""
    root = Path(root_value)
    mat_dir = root / "MAT_Files"
    if not mat_dir.is_dir():
        return {}
    try:
        import scipy.io as sio
    except Exception:
        return {}

    state_events: Dict[str, Tuple[Tuple[float, int], ...]] = {}
    for mat_path in sorted(mat_dir.glob("*.mat")):
        try:
            payload = sio.loadmat(
                mat_path,
                simplify_cells=True,
                variable_names=["events"],
            )
        except Exception:
            continue
        events = payload.get("events")
        if events is None:
            continue
        events_arr = np.asarray(events, dtype=object)
        if events_arr.ndim == 1:
            events_arr = events_arr.reshape(1, -1)
        items: List[Tuple[float, int]] = []
        for row in events_arr:
            if len(row) < 3:
                continue
            label = str(row[2]).strip().lower()
            if label not in {"waking", "sleeping"}:
                continue
            try:
                timestamp = float(str(row[0]).strip())
            except ValueError:
                continue
            items.append((timestamp, VEP_STATE_ID_BY_NAME[label]))
        if items:
            state_events[mat_path.stem] = tuple(sorted(items, key=lambda item: item[0]))
    return state_events


def infer_vepiset_state_label(
    row: Dict[str, object],
    state_events: Dict[str, Tuple[Tuple[float, int], ...]],
) -> int:
    patient_id = str(row["patient_id"])
    events = state_events.get(patient_id, ())
    if not events:
        return VEP_STATE_ID_BY_NAME["unknown"]
    fs = float(row.get("fs", 500.0))
    midpoint_sec = (float(row["start_sample"]) + float(row["end_sample"])) / max(2.0 * fs, 1e-6)
    state_label = int(events[0][1])
    for timestamp, candidate in events:
        if timestamp <= midpoint_sec:
            state_label = int(candidate)
        else:
            break
    return state_label


def annotate_vepiset_states(
    root: str | Path,
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    state_events = load_vepiset_state_events(str(Path(root)))
    annotated: List[Dict[str, object]] = []
    for row in rows:
        state_label = infer_vepiset_state_label(row, state_events)
        annotated.append(
            {
                **row,
                "state_label": int(state_label),
                "state_name": VEP_STATE_NAME_BY_ID[int(state_label)],
            }
        )
    return annotated


def scan_vepiset(root: str | Path) -> List[Dict[str, object]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"VEPiSet root not found: {root}")
    rows: List[Dict[str, object]] = []
    for dirname, expected_class_id in VEP_CLASS_DIRS:
        class_dir = root / dirname
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.glob("*.npy")):
            meta = _parse_vepiset_name(path)
            class_id = int(meta["class_id"])
            if class_id != expected_class_id:
                raise ValueError(
                    f"Class id mismatch for {path}: folder={expected_class_id}, file={class_id}"
                )
            rows.append(
                {
                    **meta,
                    "path": str(path),
                    "class_dir": dirname,
                    "class_name": VEP_CLASS_NAME_BY_ID[class_id],
                }
            )
    if not rows:
        raise ValueError(f"No VEPiSet .npy files found under {root}")
    rows.sort(key=lambda r: (str(r["patient_id"]), int(r["start_sample"]), int(r["class_id"]), str(r["path"])))
    return rows


def _patient_class_count_vectors(
    rows: Sequence[Dict[str, object]],
    n_classes: int,
) -> Dict[str, np.ndarray]:
    counts: Dict[str, np.ndarray] = {}
    for row in rows:
        patient = str(row["patient_id"])
        if patient not in counts:
            counts[patient] = np.zeros(n_classes, dtype=np.float64)
        counts[patient][int(row["class_id"])] += 1.0
    return counts


def _patient_class_presence_vectors(
    rows: Sequence[Dict[str, object]],
    n_classes: int,
) -> Dict[str, np.ndarray]:
    counts = _patient_class_count_vectors(rows, n_classes=n_classes)
    return {patient: (values > 0).astype(np.float64) for patient, values in counts.items()}


def _split_score(
    patient_counts: Dict[str, np.ndarray],
    split_ids: Dict[str, set[str]],
    val_ratio: float,
    test_ratio: float,
    patient_presence: Optional[Dict[str, np.ndarray]] = None,
) -> float:
    total = np.sum(np.stack(list(patient_counts.values()), axis=0), axis=0)
    desired = {
        "train": total * max(1.0 - float(val_ratio) - float(test_ratio), 0.0),
        "val": total * max(float(val_ratio), 0.0),
        "test": total * max(float(test_ratio), 0.0),
    }
    class_weight = 1.0 / np.sqrt(np.maximum(total, 1.0))
    class_weight = class_weight / max(float(class_weight.mean()), 1e-8)

    score = 0.0
    for split_name, patients in split_ids.items():
        observed = (
            np.sum([patient_counts[p] for p in patients], axis=0)
            if patients
            else np.zeros_like(total)
        )
        target = desired[split_name]
        score += float(np.sum(np.abs(np.log1p(observed) - np.log1p(target)) * class_weight))

        # Keep every class represented in every split whenever the dataset can support it.
        for class_id in range(len(total)):
            if total[class_id] <= 0:
                continue
            if observed[class_id] <= 0:
                score += 1000.0
                continue
            if split_name in {"val", "test"} and class_id > 0:
                min_support = max(1.0, min(20.0, target[class_id] * 0.20))
                if observed[class_id] < min_support:
                    score += float((min_support - observed[class_id]) * 10.0)

    if patient_presence:
        presence_total = np.sum(np.stack(list(patient_presence.values()), axis=0), axis=0)
        desired_presence = {
            "train": presence_total * max(1.0 - float(val_ratio) - float(test_ratio), 0.0),
            "val": presence_total * max(float(val_ratio), 0.0),
            "test": presence_total * max(float(test_ratio), 0.0),
        }
        presence_weight = 1.0 / np.sqrt(np.maximum(presence_total, 1.0))
        presence_weight = presence_weight / max(float(presence_weight.mean()), 1e-8)
        for split_name, patients in split_ids.items():
            observed_presence = (
                np.sum([patient_presence[p] for p in patients], axis=0)
                if patients
                else np.zeros_like(presence_total)
            )
            score += float(
                np.sum(
                    np.abs(np.log1p(observed_presence) - np.log1p(desired_presence[split_name]))
                    * presence_weight
                    * 25.0
                )
            )
            for class_id in range(1, len(presence_total)):
                total_patients = int(presence_total[class_id])
                if total_patients <= 0:
                    continue
                if split_name == "train":
                    min_train = max(1, int(np.floor(total_patients * max(1.0 - val_ratio - test_ratio, 0.0))))
                    if total_patients >= 4:
                        min_train = max(min_train, 2)
                    if observed_presence[class_id] < min_train:
                        score += float((min_train - observed_presence[class_id]) * 250.0)
                elif split_name in {"val", "test"} and total_patients >= 3:
                    if observed_presence[class_id] <= 0:
                        score += 250.0
    return score


def _split_patient_ids(
    rows: Sequence[Dict[str, object]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    strategy: str = "balanced",
    search_trials: int = 4096,
) -> Dict[str, set[str]]:
    patients = sorted({str(row["patient_id"]) for row in rows})
    rng = np.random.default_rng(int(seed))
    n_patients = len(patients)
    n_val = int(round(n_patients * max(float(val_ratio), 0.0)))
    n_test = int(round(n_patients * max(float(test_ratio), 0.0)))
    if val_ratio > 0:
        n_val = max(1, n_val)
    if test_ratio > 0:
        n_test = max(1, n_test)
    while n_val + n_test >= n_patients and n_patients > 1:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break
    def split_from_order(order: Sequence[str]) -> Dict[str, set[str]]:
        val_ids = set(order[:n_val])
        test_ids = set(order[n_val:n_val + n_test])
        train_ids = set(order[n_val + n_test:])
        return {"train": train_ids, "val": val_ids, "test": test_ids}

    shuffled = patients[:]
    rng.shuffle(shuffled)
    best = split_from_order(shuffled)
    strategy_name = str(strategy).lower()
    if strategy_name in {"balanced", "stratified", "patient_class_balanced"} and (n_val > 0 or n_test > 0):
        n_classes = max(int(row["class_id"]) for row in rows) + 1
        patient_counts = _patient_class_count_vectors(rows, n_classes=n_classes)
        patient_presence = (
            _patient_class_presence_vectors(rows, n_classes=n_classes)
            if strategy_name == "patient_class_balanced"
            else None
        )
        best_score = _split_score(
            patient_counts,
            best,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            patient_presence=patient_presence,
        )
        for _ in range(max(int(search_trials), 1) - 1):
            candidate_order = patients[:]
            rng.shuffle(candidate_order)
            candidate = split_from_order(candidate_order)
            score = _split_score(
                patient_counts,
                candidate,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                patient_presence=patient_presence,
            )
            if score < best_score:
                best = candidate
                best_score = score
    elif strategy_name not in {"random", "balanced", "stratified", "patient_class_balanced"}:
        raise ValueError(f"Unsupported VEPiSet split strategy: {strategy}")

    train = best["train"]
    if not train:
        raise ValueError("VEPiSet patient split produced an empty train set")
    return best


def _limit_rows(
    rows: Sequence[Dict[str, object]],
    max_samples_per_class: int = 0,
    max_non_ied_samples: int = 0,
    seed: int = 42,
) -> List[Dict[str, object]]:
    if max_samples_per_class <= 0 and max_non_ied_samples <= 0:
        return list(rows)
    rng = np.random.default_rng(int(seed))
    grouped: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["class_id"])].append(row)
    selected: List[Dict[str, object]] = []
    for class_id, items in grouped.items():
        cap = int(max_samples_per_class)
        if class_id == 0 and max_non_ied_samples > 0:
            cap = int(max_non_ied_samples)
        if cap <= 0 or len(items) <= cap:
            selected.extend(items)
            continue
        order = np.arange(len(items))
        rng.shuffle(order)
        selected.extend(items[int(i)] for i in order[:cap])
    selected.sort(key=lambda r: (str(r["patient_id"]), int(r["start_sample"]), int(r["class_id"]), str(r["path"])))
    return selected


def build_vepiset_rows(
    root: str | Path,
    split: str,
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    split_strategy: str = "balanced",
    split_search_trials: int = 4096,
    max_samples_per_class: int = 0,
    max_non_ied_samples: int = 0,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows = scan_vepiset(root)
    splits = _split_patient_ids(
        rows,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        strategy=split_strategy,
        search_trials=split_search_trials,
    )
    split = str(split).lower()
    if split not in splits:
        raise ValueError(f"Unsupported VEPiSet split: {split}; choices={tuple(splits)}")
    split_rows = [row for row in rows if str(row["patient_id"]) in splits[split]]
    limited = _limit_rows(
        split_rows,
        max_samples_per_class=max_samples_per_class,
        max_non_ied_samples=max_non_ied_samples,
        seed=seed + {"train": 0, "val": 101, "test": 202}[split],
    )
    limited = annotate_vepiset_states(root, limited)
    meta = {
        "split": split,
        "split_strategy": split_strategy,
        "patients": {key: sorted(value) for key, value in splits.items()},
        "all_counts": dict(Counter(int(row["class_id"]) for row in rows)),
        "split_counts_before_limit": dict(Counter(int(row["class_id"]) for row in split_rows)),
        "split_counts": dict(Counter(int(row["class_id"]) for row in limited)),
        "state_counts": dict(Counter(str(row.get("state_name", "unknown")) for row in limited)),
    }
    return limited, meta


def normalize_eeg(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def fix_window_length(x: np.ndarray, target_samples: Optional[int]) -> np.ndarray:
    if target_samples is None or int(target_samples) <= 0:
        return x
    target = int(target_samples)
    current = int(x.shape[-1])
    if current == target:
        return x
    if current > target:
        start = (current - target) // 2
        return x[..., start:start + target]
    left = (target - current) // 2
    right = target - current - left
    return np.pad(x, ((0, 0), (left, right)), mode="constant")


def load_vepiset_window(
    path: str | Path,
    *,
    normalize: bool = True,
    target_samples: Optional[int] = 2000,
) -> np.ndarray:
    x = np.load(path).astype(np.float32, copy=False)
    if x.shape[0] < len(VEP_CHANNELS_29):
        raise ValueError(f"Expected at least {len(VEP_CHANNELS_29)} channels in {path}, got {x.shape}")
    x = x[: len(VEP_CHANNELS_29)]
    if normalize:
        x = normalize_eeg(x)
    x = fix_window_length(x, target_samples)
    return np.asarray(x, dtype=np.float32)


def _tcp_labels_for_pairs(class_id: int, pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
    labels = np.zeros(len(pairs), dtype=np.float32)
    if class_id == 0:
        return labels
    if class_id == 1:
        labels[:] = 1.0
        return labels
    electrodes = set(CLASS_TO_ELECTRODES[class_id])
    for idx, (left, right) in enumerate(pairs):
        if left in electrodes or right in electrodes:
            labels[idx] = 1.0
    return labels


def _monopolar_labels(class_id: int) -> np.ndarray:
    labels = np.zeros(len(VEP_MONOPOLAR_19), dtype=np.float32)
    if class_id == 0:
        return labels
    if class_id == 1:
        labels[:] = 1.0
        return labels
    electrodes = set(CLASS_TO_ELECTRODES[class_id])
    for idx, name in enumerate(VEP_MONOPOLAR_19):
        if name in electrodes:
            labels[idx] = 1.0
    return labels


def _soz_region_labels(class_id: int) -> np.ndarray:
    labels = np.zeros(len(VEP_SOZ_REGION_NAMES), dtype=np.float32)
    for region in CLASS_TO_REGIONS[class_id]:
        labels[VEP_SOZ_REGION_NAMES.index(region)] = 1.0
    return labels


def _brain_region_labels(class_id: int) -> np.ndarray:
    labels = np.zeros(len(VEP_BRAIN_REGION_NAMES), dtype=np.float32)
    for region in CLASS_TO_BRAIN_REGIONS[class_id]:
        labels[VEP_BRAIN_REGION_NAMES.index(region)] = 1.0
    return labels


def _class_weights(rows: Sequence[Dict[str, object]], cap: float = 10.0) -> Dict[int, float]:
    counts = Counter(int(row["class_id"]) for row in rows)
    total = max(sum(counts.values()), 1)
    n_classes = max(len(counts), 1)
    weights = {
        class_id: total / max(n_classes * count, 1)
        for class_id, count in counts.items()
    }
    return {class_id: float(np.clip(weight, 1.0 / cap, cap)) for class_id, weight in weights.items()}


def _region_channel_indices(
    region_names: Sequence[str],
    pairs: Sequence[Tuple[str, str]],
) -> Tuple[Tuple[int, ...], ...]:
    result: List[Tuple[int, ...]] = []
    for region in region_names:
        electrodes = set(REGION_ELECTRODES[region])
        indices = tuple(
            idx
            for idx, (left, right) in enumerate(pairs)
            if left in electrodes or right in electrodes
        )
        result.append(indices)
    return tuple(result)


class VEPiSetSOZPreDataset(Dataset):
    """VEPiSet dataset with the UnifiedSOZDataset sample contract."""

    region_names = VEP_SOZ_REGION_NAMES
    channel_names = TCP_CHANNELS
    region_channel_indices = _region_channel_indices(VEP_SOZ_REGION_NAMES, TCP_PAIRS)

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        split_strategy: str = "balanced",
        split_search_trials: int = 4096,
        max_samples_per_class: int = 0,
        max_non_ied_samples: int = 0,
        normalize: bool = True,
        target_samples: int = 2000,
        montage: str = "tcp22",
        augment: bool = False,
        augment_time_shift: int = 0,
        augment_noise_std: float = 0.0,
        augment_scale_min: float = 1.0,
        augment_scale_max: float = 1.0,
        augment_channel_dropout: float = 0.0,
        augment_polarity_prob: float = 0.0,
    ):
        self.root = Path(root)
        self.split = str(split).lower()
        self.normalize = bool(normalize)
        self.target_samples = int(target_samples)
        self.montage = str(montage).lower()
        self.augment = bool(augment)
        self.augment_time_shift = max(int(augment_time_shift), 0)
        self.augment_noise_std = max(float(augment_noise_std), 0.0)
        self.augment_scale_min = float(augment_scale_min)
        self.augment_scale_max = float(augment_scale_max)
        self.augment_channel_dropout = float(np.clip(float(augment_channel_dropout), 0.0, 1.0))
        self.augment_polarity_prob = float(np.clip(float(augment_polarity_prob), 0.0, 1.0))
        if self.montage not in {"tcp22", "monopolar19"}:
            raise ValueError(f"Unsupported VEPiSet montage: {montage}")
        if self.augment_scale_min <= 0 or self.augment_scale_max <= 0:
            raise ValueError("VEPiSet augmentation scale bounds must be positive")
        if self.augment_scale_min > self.augment_scale_max:
            raise ValueError("VEPiSet augmentation scale min cannot exceed max")
        self.rows, self.split_meta = build_vepiset_rows(
            self.root,
            self.split,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            split_strategy=split_strategy,
            split_search_trials=split_search_trials,
            max_samples_per_class=max_samples_per_class,
            max_non_ied_samples=max_non_ied_samples,
        )
        if not self.rows:
            raise ValueError(f"No VEPiSet rows for split={split}")
        self.sample_shape = (1, len(VEP_CHANNELS_29), self.target_samples)
        weights = _class_weights(self.rows)
        self.channel_labels_np = np.stack([
            _tcp_labels_for_pairs(int(row["class_id"]), TCP_PAIRS) for row in self.rows
        ]).astype(np.float32)
        self.channel_masks_np = np.ones_like(self.channel_labels_np, dtype=np.float32)
        self.region_labels_np = np.stack([
            _soz_region_labels(int(row["class_id"])) for row in self.rows
        ]).astype(np.float32)
        self.region_masks_np = np.ones_like(self.region_labels_np, dtype=np.float32)
        self.propagation_labels_np = np.zeros_like(self.region_labels_np, dtype=np.float32)
        self.propagation_masks_np = np.zeros_like(self.region_labels_np, dtype=np.float32)
        self.hemisphere_labels_np = np.full(len(self.rows), -100, dtype=np.int64)
        self.hemisphere_masks_np = np.zeros(len(self.rows), dtype=np.float32)
        self.seizure_y_np = np.asarray(
            [[1.0 if int(row["class_id"]) > 0 else 0.0] for row in self.rows],
            dtype=np.float32,
        )
        self.seizure_mask_np = np.ones_like(self.seizure_y_np, dtype=np.float32)
        self.sample_weights_np = np.asarray(
            [weights[int(row["class_id"])] for row in self.rows],
            dtype=np.float32,
        )
        self.source_ids_np = np.full(len(self.rows), 2, dtype=np.int64)
        self.input_masks_np = np.ones((len(self.rows), len(VEP_CHANNELS_29)), dtype=np.float32)
        self.segment_meta = [
            {
                "patient_id": str(row["patient_id"]),
                "base_patient_id": str(row["patient_id"]),
                "source": "vepiset",
                "split": self.split,
                "event_id": Path(str(row["path"])).stem,
                "sample_id": Path(str(row["path"])).stem,
                "sample_role": str(row["class_name"]),
                "edf_path": str(row["path"]),
                "hemisphere": "",
                "vepiset_class_id": int(row["class_id"]),
                "vepiset_class_name": str(row["class_name"]),
                "vepiset_state_label": int(row.get("state_label", 0)),
                "vepiset_state_name": str(row.get("state_name", "unknown")),
            }
            for row in self.rows
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        x = load_vepiset_window(
            row["path"],
            normalize=self.normalize,
            target_samples=self.target_samples,
        )[None, :, :]
        return {
            "x": torch.from_numpy(x),
            "input_mask": torch.ones(len(VEP_CHANNELS_29), dtype=torch.float32),
            "artifact_score": torch.zeros((1, len(VEP_CHANNELS_29)), dtype=torch.float32),
            "artifact_mask": torch.zeros((1, len(VEP_CHANNELS_29)), dtype=torch.float32),
            "channel_y": torch.from_numpy(self.channel_labels_np[idx]),
            "channel_mask": torch.from_numpy(self.channel_masks_np[idx]),
            "region_y": torch.from_numpy(self.region_labels_np[idx]),
            "region_mask": torch.from_numpy(self.region_masks_np[idx]),
            "propagation_y": torch.from_numpy(self.propagation_labels_np[idx]),
            "propagation_mask": torch.from_numpy(self.propagation_masks_np[idx]),
            "seizure_y": torch.from_numpy(self.seizure_y_np[idx]),
            "seizure_mask": torch.from_numpy(self.seizure_mask_np[idx]),
            "hemisphere_y": torch.tensor(-100, dtype=torch.long),
            "hemisphere_mask": torch.tensor(0.0, dtype=torch.float32),
            "sample_weight": torch.tensor(float(self.sample_weights_np[idx]), dtype=torch.float32),
            "source_id": torch.tensor(2, dtype=torch.long),
            "state_label": torch.tensor(int(row.get("state_label", 0)), dtype=torch.long),
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
        return len(self.region_names)

    @property
    def n_hemisphere_classes(self) -> int:
        return 4

    @property
    def patients(self) -> List[str]:
        return sorted({str(row["patient_id"]) for row in self.rows})


class VEPiSetBrainNetworkDataset(Dataset):
    """VEPiSet dataset with the brain-network trainer sample contract."""

    region_names = VEP_BRAIN_REGION_NAMES
    channel_names = VEP_MONOPOLAR_19
    bipolar_channel_names = BRAIN_TCP_CHANNELS

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        split_strategy: str = "balanced",
        split_search_trials: int = 4096,
        max_samples_per_class: int = 0,
        max_non_ied_samples: int = 0,
        normalize: bool = True,
        target_samples: int = 2000,
        montage: str = "tcp22",
        augment: bool = False,
        augment_time_shift: int = 0,
        augment_noise_std: float = 0.0,
        augment_scale_min: float = 1.0,
        augment_scale_max: float = 1.0,
        augment_channel_dropout: float = 0.0,
        augment_polarity_prob: float = 0.0,
    ):
        self.root = Path(root)
        self.split = str(split).lower()
        self.normalize = bool(normalize)
        self.target_samples = int(target_samples)
        self.montage = str(montage).lower()
        self.augment = bool(augment)
        self.augment_time_shift = max(int(augment_time_shift), 0)
        self.augment_noise_std = max(float(augment_noise_std), 0.0)
        self.augment_scale_min = float(augment_scale_min)
        self.augment_scale_max = float(augment_scale_max)
        self.augment_channel_dropout = float(np.clip(float(augment_channel_dropout), 0.0, 1.0))
        self.augment_polarity_prob = float(np.clip(float(augment_polarity_prob), 0.0, 1.0))
        if self.montage not in {"tcp22", "monopolar19"}:
            raise ValueError(f"Unsupported VEPiSet montage: {montage}")
        if self.augment_scale_min <= 0 or self.augment_scale_max <= 0:
            raise ValueError("VEPiSet augmentation scale bounds must be positive")
        if self.augment_scale_min > self.augment_scale_max:
            raise ValueError("VEPiSet augmentation scale min cannot exceed max")
        self.rows, self.split_meta = build_vepiset_rows(
            self.root,
            self.split,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            split_strategy=split_strategy,
            split_search_trials=split_search_trials,
            max_samples_per_class=max_samples_per_class,
            max_non_ied_samples=max_non_ied_samples,
        )
        if not self.rows:
            raise ValueError(f"No VEPiSet rows for split={split}")
        self.window_samples = self.target_samples
        self.sample_weights = _class_weights(self.rows)
        if self.montage == "monopolar19":
            self.input_channel_names = VEP_MONOPOLAR_19
        else:
            self.input_channel_names = BRAIN_TCP_CHANNELS

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def monopolar_to_bipolar(x29: np.ndarray) -> np.ndarray:
        pairs = []
        for left, right in BRAIN_TCP_PAIRS:
            left_idx = VEP_MONOPOLAR_INDEX[left]
            right_idx = VEP_MONOPOLAR_INDEX[right]
            pairs.append(x29[left_idx] - x29[right_idx])
        return np.stack(pairs, axis=0).astype(np.float32)

    def _augment_window(self, x: np.ndarray) -> np.ndarray:
        if not self.augment:
            return x
        x_aug = np.asarray(x, dtype=np.float32).copy()
        if self.augment_scale_min != 1.0 or self.augment_scale_max != 1.0:
            scale = np.random.uniform(self.augment_scale_min, self.augment_scale_max)
            x_aug *= np.float32(scale)
        if self.augment_polarity_prob > 0.0 and np.random.random() < self.augment_polarity_prob:
            x_aug *= np.float32(-1.0)
        if self.augment_time_shift > 0:
            shift = int(np.random.randint(-self.augment_time_shift, self.augment_time_shift + 1))
            if shift > 0:
                shifted = np.zeros_like(x_aug)
                shifted[:, shift:] = x_aug[:, :-shift]
                x_aug = shifted
            elif shift < 0:
                shifted = np.zeros_like(x_aug)
                shifted[:, :shift] = x_aug[:, -shift:]
                x_aug = shifted
        if self.augment_channel_dropout > 0.0:
            keep = np.random.random(x_aug.shape[0]) >= self.augment_channel_dropout
            if not bool(keep.any()):
                keep[np.random.randint(0, x_aug.shape[0])] = True
            x_aug *= keep.astype(np.float32)[:, None]
        if self.augment_noise_std > 0.0:
            channel_std = np.std(x_aug, axis=1, keepdims=True).astype(np.float32)
            noise_scale = np.maximum(channel_std, np.float32(1e-6)) * np.float32(self.augment_noise_std)
            x_aug += np.random.normal(0.0, 1.0, size=x_aug.shape).astype(np.float32) * noise_scale
        return np.asarray(x_aug, dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        class_id = int(row["class_id"])
        x29 = load_vepiset_window(
            row["path"],
            normalize=self.normalize,
            target_samples=self.target_samples,
        )
        if self.montage == "monopolar19":
            x_model = x29[: len(VEP_MONOPOLAR_19)]
            model_channel_label = _monopolar_labels(class_id)
        else:
            x_model = self.monopolar_to_bipolar(x29)
            model_channel_label = _tcp_labels_for_pairs(class_id, BRAIN_TCP_PAIRS)
        x_model = self._augment_window(x_model)
        fs = float(row.get("fs", 500.0))
        onset_sec = x_model.shape[-1] / fs / 2.0
        return {
            "idx": idx,
            "x": torch.from_numpy(np.asarray(x_model, dtype=np.float32)),
            "label": torch.from_numpy(_monopolar_labels(class_id)),
            "bipolar_label": torch.from_numpy(np.asarray(model_channel_label, dtype=np.float32)),
            "monopolar_label": torch.from_numpy(_monopolar_labels(class_id)),
            "region_label": torch.from_numpy(_brain_region_labels(class_id)),
            "class_label": torch.tensor(class_id, dtype=torch.long),
            "ied_binary_label": torch.tensor(1 if class_id > 0 else 0, dtype=torch.long),
            "hemisphere_label": torch.tensor(-100, dtype=torch.long),
            "state_label": torch.tensor(int(row.get("state_label", 0)), dtype=torch.long),
            "onset_sec": torch.tensor(onset_sec, dtype=torch.float32),
            "start_sec": torch.tensor(0.0, dtype=torch.float32),
            "source": "vepiset",
            "patient_id": str(row["patient_id"]),
            "edf_path": str(row["path"]),
            "sample_weight": torch.tensor(self.sample_weights[class_id], dtype=torch.float32),
        }


def summarize_vepiset_dataset(dataset: Dataset) -> Dict[str, object]:
    rows = getattr(dataset, "rows", [])
    counts = Counter(int(row["class_id"]) for row in rows)
    return {
        "rows": int(len(rows)),
        "patients": int(len({str(row["patient_id"]) for row in rows})),
        "classes": {VEP_CLASS_NAME_BY_ID[int(k)]: int(v) for k, v in counts.items()},
        "states": dict(Counter(str(row.get("state_name", "unknown")) for row in rows)),
        "split": getattr(dataset, "split", ""),
    }
