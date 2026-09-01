#!/usr/bin/env python3
"""Evaluate published DeepSOZ fold weights on locally mapped TUSZ signals.

This is a clean-room evaluation adapter, not a copy of the upstream source.
It follows the architecture and preprocessing contract documented by the
DeepSOZ paper/repository while making held-out lineage and endpoint semantics
explicit.  Private data are intentionally out of scope.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path, PurePosixPath
import time
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import pyedflib
from scipy import signal
import torch
from torch import nn
import torch.nn.functional as F


CHANNELS = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ",
    "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
)
CHANNEL_INDEX = {channel: index for index, channel in enumerate(CHANNELS)}
LEGACY_TO_MODERN = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
PZ_INDEX = CHANNEL_INDEX["PZ"]

# The graph is transcribed as an endpoint definition, not learned or tuned.
OFFICIAL_NEIGHBORS = {
    0: (1, 2, 3, 4),
    1: (0, 4, 5, 6),
    2: (0, 3, 4, 7, 8),
    3: (0, 2, 4, 8, 9),
    4: (0, 1, 3, 5, 9),
    5: (1, 4, 6, 9, 10),
    6: (1, 4, 5, 10, 11),
    7: (2, 8, 12, 13, 17),
    8: (2, 3, 4, 7, 9, 12, 13, 14),
    9: (3, 4, 5, 8, 10, 13, 14, 15),
    10: (4, 5, 6, 9, 11, 14, 15, 16),
    11: (6, 10, 15, 16, 18),
    12: (7, 8, 13, 17),
    13: (7, 8, 9, 12, 14, 17),
    14: (8, 9, 10, 13, 15, 17, 18),
    15: (9, 10, 11, 14, 16, 18),
    16: (10, 11, 15, 18),
    17: (7, 12, 13, 14, 18),
    18: (11, 14, 15, 16, 17),
}


def _normalize_label(value: str) -> str:
    label = str(value).strip().upper().replace("EEG ", "")
    label = label.split("-")[0].strip().replace(" ", "")
    return LEGACY_TO_MODERN.get(label, label)


class PublishedDeepSOZ(nn.Module):
    """Architecture matching the published 19-channel DeepSOZ weight schema."""

    def __init__(self, dropout: float = 0.15) -> None:
        super().__init__()
        self.detector = nn.Module()
        self.detector.pos_encoder = nn.Embedding(20, 200)
        self.detector.tx_encoder = nn.TransformerEncoderLayer(
            d_model=200,
            nhead=8,
            dim_feedforward=256,
            dropout=float(dropout),
            batch_first=True,
        )
        self.detector.multi_lstm = nn.LSTM(
            input_size=200,
            hidden_size=100,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
        )
        self.detector.multi_linear = nn.Linear(200, 2)
        self.hc_linear = nn.Linear(200, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5 or x.shape[-2:] != (19, 200):
            raise ValueError(f"Expected [B,N,T,19,200], got {tuple(x.shape)}")
        batch, n_seizures, n_seconds, n_channels, width = x.shape
        position = self.detector.pos_encoder(
            torch.arange(19, device=x.device)
        ).view(1, 1, 19, 200)
        channel = x + position
        global_token = self.detector.pos_encoder(
            torch.full(
                (batch, n_seizures, n_seconds, 1),
                19,
                device=x.device,
                dtype=torch.long,
            )
        )
        encoded = self.detector.tx_encoder(
            torch.cat(
                [
                    channel.reshape(batch * n_seizures * n_seconds, 19, 200),
                    global_token.reshape(batch * n_seizures * n_seconds, 1, 200),
                ],
                dim=1,
            )
        )
        channel_encoded = encoded[:, :19].reshape(
            batch * n_seizures, n_seconds, n_channels, width
        )
        global_encoded = encoded[:, 19].reshape(
            batch * n_seizures, n_seconds, width
        )
        temporal, _ = self.detector.multi_lstm(global_encoded)
        detection_logits = self.detector.multi_linear(
            temporal.reshape(batch * n_seizures * n_seconds, 200)
        ).reshape(batch, n_seizures, n_seconds, 2)
        channel_logits = self.hc_linear(
            channel_encoded.reshape(batch * n_seizures * n_seconds * 19, 200)
        ).reshape(batch * n_seizures, n_seconds, 19)
        attention = F.softmax(
            detection_logits.reshape(batch * n_seizures, n_seconds, 2)[:, :, 1],
            dim=1,
        )
        seizure_channel_probability = torch.sigmoid(
            (attention.unsqueeze(-1) * channel_logits).sum(dim=1)
        ).reshape(batch, n_seizures, 19)
        return detection_logits, seizure_channel_probability


def _safe_edf(root: Path, relative: str) -> Path:
    value = PurePosixPath(str(relative))
    if value.is_absolute() or ".." in value.parts or value.suffix.lower() != ".edf":
        raise ValueError(f"Unsafe EDF path: {relative!r}")
    path = root.joinpath(*value.parts).resolve(strict=True)
    path.relative_to(root)
    return path


def _read_standard19(path: Path) -> tuple[np.ndarray, float, tuple[str, ...]]:
    reader = pyedflib.EdfReader(str(path))
    try:
        labels = reader.getSignalLabels()
        bound: dict[str, int] = {}
        for index, label in enumerate(labels):
            channel = _normalize_label(label)
            if channel in CHANNEL_INDEX and channel not in bound:
                bound[channel] = index
        if "FP1" not in bound:
            raise ValueError(f"FP1 is absent from {path}")
        reference_index = bound["FP1"]
        sample_count = int(reader.getNSamples()[reference_index])
        reference_rate = float(reader.getSampleFrequency(reference_index))
        output: list[np.ndarray] = []
        missing: list[str] = []
        for channel in CHANNELS:
            index = bound.get(channel)
            if index is None:
                output.append(np.zeros(sample_count, dtype=np.float64))
                missing.append(channel)
                continue
            rate = float(reader.getSampleFrequency(index))
            count = int(reader.getNSamples()[index])
            if abs(rate - reference_rate) > 1e-9 or count != sample_count:
                raise ValueError(f"Mixed standard-19 sampling in {path}")
            output.append(np.asarray(reader.readSignal(index), dtype=np.float64))
        return np.stack(output), reference_rate, tuple(missing)
    finally:
        reader.close()


def _preprocess_record(raw: np.ndarray, sfreq: float) -> np.ndarray:
    if raw.ndim != 2 or raw.shape[0] != 19 or sfreq <= 0:
        raise ValueError("Invalid raw standard-19 record")
    target_samples = int(raw.shape[1] * 200.0 / float(sfreq))
    if target_samples < 200:
        raise ValueError("Record is shorter than one output second")
    low_b, low_a = signal.butter(4, 30.0 / 100.0)
    high_b, high_a = signal.butter(4, 1.6 / 100.0, btype="highpass")
    def preprocess_channel(channel: np.ndarray) -> np.ndarray:
        # Each channel follows the exact published scalar operation order.
        # ThreadPoolExecutor changes scheduling only; map preserves channel
        # order and is bit-identical to the former sequential loop.
        resampled = signal.resample(channel, target_samples)
        filtered = signal.filtfilt(low_b, low_a, resampled, method="gust")
        filtered = signal.filtfilt(high_b, high_a, filtered, method="gust")
        center = float(filtered.mean())
        spread = float(filtered.std())
        if not math.isfinite(spread) or spread <= 0:
            return np.zeros(target_samples, dtype=np.float64)
        return np.clip(
            filtered, center - 2.0 * spread, center + 2.0 * spread
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        processed = np.stack(list(executor.map(preprocess_channel, raw)))
    seconds = processed.shape[1] // 200
    return processed[:, : seconds * 200].reshape(19, seconds, 200).transpose(1, 0, 2)


def _fit_600(value: np.ndarray) -> np.ndarray:
    output = np.zeros((600, 19, 200), dtype=np.float64)
    keep = min(600, int(value.shape[0]))
    if keep:
        output[:keep] = value[:keep]
    return output


def _crop_600(
    windows: np.ndarray,
    starts_sec: Iterable[float],
    stops_sec: Iterable[float],
) -> list[np.ndarray]:
    """Reproduce the published 600-second seizure-context selection policy."""
    starts = np.ceil(np.asarray(list(starts_sec), dtype=np.float64)).astype(int)
    stops = np.ceil(np.asarray(list(stops_sec), dtype=np.float64)).astype(int)
    if starts.shape != stops.shape or starts.ndim != 1 or starts.size == 0:
        raise ValueError("Seizure boundaries are invalid")
    order = np.argsort(starts, kind="stable")
    starts = starts[order][:10]
    stops = stops[order][:10]
    n_seizures = int(starts.size)
    total_seconds = int(windows.shape[0])
    starts_with_end = np.concatenate([starts, np.asarray([total_seconds])])
    result: list[np.ndarray] = []
    for index in range(n_seizures):
        if index == 0:
            begin = 0
            end = min(
                int(starts_with_end[index + 1] - 1),
                int((stops[index] + starts_with_end[index + 1]) / 2),
            )
        elif index == n_seizures - 1:
            end = int(starts_with_end[index + 1] - 1)
            begin = int((stops[index - 1] + starts[index]) / 2)
        else:
            begin = int((stops[index - 1] + starts[index]) / 2)
            end = min(
                int(starts_with_end[index + 1] - 1),
                int((stops[index] + starts_with_end[index + 1]) / 2),
            )
        begin = max(0, min(begin, total_seconds))
        end = max(begin - 1, min(end, total_seconds - 1))
        total_length = end - begin + 1
        seizure_length = int(stops[index] - starts[index] + 1)
        if total_length <= 600:
            crop = windows[begin : end + 1]
        elif seizure_length >= 600:
            begin = max(int(starts[index] - 300), 0)
            crop = windows[begin : begin + 600]
        else:
            previous_end = int(stops[index - 1]) if index > 0 else 0
            new_begin = max(previous_end, int(starts[index] - 120))
            new_end = min(
                int(stops[index] + 30), int(starts_with_end[index + 1] - 1)
            )
            new_total = new_end - new_begin + 1
            if new_total <= 600:
                new_begin = max(previous_end, int(new_begin - 600 + new_total))
                new_total = new_end - new_begin + 1
                if new_total < 600:
                    new_end = min(
                        int(starts_with_end[index + 1]),
                        int(new_end + 600 - new_total),
                    )
                crop = windows[new_begin : new_end + 1]
            else:
                crop = windows[new_begin : new_begin + 600]
        result.append(_fit_600(crop))
    return result


def _load_targets(path: Path, allowed_ids: set[str]) -> dict[str, dict[str, np.ndarray]]:
    frame = pd.read_csv(path, dtype={"deepsoz_patient_id": str})
    output: dict[str, dict[str, np.ndarray]] = {}
    for row in frame.to_dict("records"):
        patient = str(row["deepsoz_patient_id"]).strip().lstrip("0") or "0"
        if patient not in allowed_ids or int(row["eligible_for_localization"]) != 1:
            continue
        target = np.asarray(
            [float(row[f"benchmark_value_{channel}"]) for channel in CHANNELS],
            dtype=np.float64,
        )
        mask = np.asarray(
            [float(row[f"benchmark_mask_{channel}"]) for channel in CHANNELS],
            dtype=np.float64,
        ).astype(bool)
        if not np.any((target > 0.5) & mask):
            raise ValueError(f"Patient {patient} has no evaluable positive")
        output[patient] = {"target": target, "mask": mask}
    return output


def _load_events(path: Path, patient_ids: set[str]) -> dict[str, dict[str, list[dict[str, object]]]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    receipt = artifact["receipt"]
    records: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen: set[str] = set()
    for row in [*receipt.get("events", []), *receipt.get("exclusions", [])]:
        patient = str(row.get("patient_id", "")).strip().lstrip("0") or "0"
        event_id = str(row.get("event_id", "")).strip()
        relative = str(row.get("relative_edf_path", "")).strip()
        if patient not in patient_ids or not event_id or not relative or event_id in seen:
            continue
        if row.get("global_t0_sec") is None or row.get("global_stop_sec") is None:
            continue
        seen.add(event_id)
        records[patient][relative].append(dict(row))
    for patient_records in records.values():
        for rows in patient_records.values():
            rows.sort(key=lambda row: (float(row["global_t0_sec"]), str(row["event_id"])))
    return records


def _folds_by_patient(fold_directory: Path) -> dict[str, tuple[int, ...]]:
    output: dict[str, list[int]] = defaultdict(list)
    for fold in range(15):
        path = fold_directory / f"deepsoz_official_pts_test_fold{fold}.npy"
        values = np.load(path).astype(int).tolist()
        for value in values:
            output[str(int(value))].append(fold)
    return {patient: tuple(folds) for patient, folds in output.items()}


def _load_models(weights: Path, folds: set[int], device: torch.device) -> dict[int, PublishedDeepSOZ]:
    models: dict[int, PublishedDeepSOZ] = {}
    for fold in sorted(folds):
        state = torch.load(
            weights / f"fold{fold}.pth.tar",
            map_location="cpu",
            weights_only=True,
        )
        model = PublishedDeepSOZ(dropout=0.15).double()
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        models[fold] = model
    return models


def _predict_crops(
    model: PublishedDeepSOZ,
    crops: list[np.ndarray],
    device: torch.device,
) -> list[np.ndarray]:
    if not crops:
        return []
    values = np.stack(crops)
    center = float(values.mean())
    scale = float(values.std())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Record-level normalization has zero/nonfinite scale")
    with torch.inference_mode():
        # The published DataLoader sends all seizures from one manifest record
        # as the Nsz axis of one batch.  There is no cross-seizure operation in
        # the model, but preserving that execution contract is both faithful
        # and substantially faster than one forward per crop.
        x = torch.from_numpy((values - center) / scale).unsqueeze(0)
        _, probability = model(x.to(device=device, dtype=torch.float64))
    return [
        row.copy()
        for row in probability[0].detach().cpu().numpy()
    ]


def _score_one(scores: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    if scores.ndim != 2 or scores.shape[1] != 19:
        raise ValueError("Patient scores must be [events,19]")
    per_event = scores / np.maximum(scores.max(axis=1, keepdims=True), 1e-12)
    patient_score = per_event.mean(axis=0)
    eligible_score = patient_score.copy()
    eligible_score[~mask] = -np.inf
    top1 = int(np.argmax(eligible_score))
    positives = np.flatnonzero((target > 0.5) & mask)
    exact = bool(top1 in set(positives.tolist()))
    neighbor = any(top1 in OFFICIAL_NEIGHBORS[int(index)] for index in positives)
    return {
        "score": patient_score.tolist(),
        "top1_index": top1,
        "top1_channel": CHANNELS[top1],
        "positive_channels": [CHANNELS[int(index)] for index in positives],
        "positive_count": int(positives.size),
        "exact": exact,
        "neighborhood2": bool(exact or (positives.size <= 2 and neighbor)),
        "neighborhood4": bool(exact or (positives.size <= 4 and neighbor)),
    }


def _metrics(rows: list[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "exact_n": sum(bool(row["exact"]) for row in rows),
        "exact": float(np.mean([bool(row["exact"]) for row in rows])),
        "neighborhood2_n": sum(bool(row["neighborhood2"]) for row in rows),
        "neighborhood2": float(np.mean([bool(row["neighborhood2"]) for row in rows])),
        "neighborhood4_n": sum(bool(row["neighborhood4"]) for row in rows),
        "neighborhood4": float(np.mean([bool(row["neighborhood4"]) for row in rows])),
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    v16 = json.loads(Path(args.local_manifest).read_text(encoding="utf-8"))
    local_ids = [str(value).strip().lstrip("0") or "0" for value in v16["patient_ids"]]
    if args.patients:
        requested = {value.strip().lstrip("0") or "0" for value in args.patients.split(",") if value.strip()}
        unknown = requested - set(local_ids)
        if unknown:
            raise ValueError(f"Unknown local patient IDs: {sorted(unknown)}")
        local_ids = [patient for patient in local_ids if patient in requested]
    if args.max_patients > 0:
        local_ids = local_ids[: args.max_patients]
    patient_set = set(local_ids)
    targets = _load_targets(Path(args.target_csv), patient_set)
    if set(targets) != patient_set:
        raise ValueError(f"Target roster mismatch: {sorted(patient_set - set(targets))}")
    events = _load_events(Path(args.signal_artifact), patient_set)
    if set(events) != patient_set:
        raise ValueError(f"Signal roster mismatch: {sorted(patient_set - set(events))}")
    held_out = _folds_by_patient(Path(args.fold_directory))
    missing_folds = sorted(patient for patient in patient_set if patient not in held_out)
    if missing_folds:
        raise ValueError(f"Patients absent from all official test folds: {missing_folds}")
    needed_folds = {fold for patient in patient_set for fold in held_out[patient]}
    device = torch.device(args.device)
    models = _load_models(Path(args.weights), needed_folds, device)
    root = Path(args.tusz_root).resolve(strict=True)
    fold_scores: dict[int, dict[str, list[np.ndarray]]] = {
        fold: defaultdict(list) for fold in needed_folds
    }
    patient_receipts: list[dict[str, object]] = []
    started = time.monotonic()
    for ordinal, patient in enumerate(local_ids, start=1):
        record_count = 0
        event_count = 0
        missing_channels: set[str] = set()
        patient_folds = held_out[patient]
        for relative, boundaries in events[patient].items():
            path = _safe_edf(root, relative)
            raw, sfreq, missing = _read_standard19(path)
            windows = _preprocess_record(raw, sfreq)
            crops = _crop_600(
                windows,
                [float(row["global_t0_sec"]) for row in boundaries],
                [float(row["global_stop_sec"]) for row in boundaries],
            )
            for fold in patient_folds:
                fold_scores[fold][patient].extend(
                    _predict_crops(models[fold], crops, device)
                )
            record_count += 1
            event_count += len(crops)
            missing_channels.update(missing)
        patient_receipts.append(
            {
                "patient_id": patient,
                "held_out_folds": list(patient_folds),
                "held_out_repeat_count": len(patient_folds),
                "record_count": record_count,
                "event_count": event_count,
                "zero_filled_channels": sorted(missing_channels),
            }
        )
        print(
            json.dumps(
                {
                    "patient": ordinal,
                    "total": len(local_ids),
                    "patient_id": patient,
                    "held_out_folds": list(patient_folds),
                    "records": record_count,
                    "events": event_count,
                    "elapsed_sec": round(time.monotonic() - started, 2),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    prediction_rows: list[dict[str, object]] = []
    ensemble_rows: list[dict[str, object]] = []
    for patient in local_ids:
        target = targets[patient]["target"]
        mask = targets[patient]["mask"]
        patient_fold_rows: list[dict[str, object]] = []
        for fold in held_out[patient]:
            scores = np.stack(fold_scores[fold][patient])
            scored = _score_one(scores, target, mask)
            row = {
                "patient_id": patient,
                "fold": fold,
                "repeat": fold // 5,
                "event_count": int(scores.shape[0]),
                **scored,
            }
            prediction_rows.append(row)
            patient_fold_rows.append(row)
        ensemble_score = np.mean(
            [np.asarray(row["score"], dtype=np.float64) for row in patient_fold_rows],
            axis=0,
        )
        ensemble_scored = _score_one(
            ensemble_score.reshape(1, 19), target, mask
        )
        ensemble_rows.append(
            {
                "patient_id": patient,
                "held_out_repeat_count": len(patient_fold_rows),
                **ensemble_scored,
            }
        )
    repeat_metrics = {
        str(repeat): _metrics(
            [row for row in prediction_rows if int(row["repeat"]) == repeat]
        )
        for repeat in range(3)
    }
    return {
        "schema_version": "official_deepsoz_local_oof_evaluation_v1",
        "status": "smoke" if args.max_patients > 0 or args.patients else "full",
        "claim_boundary": (
            "published_weight_signal_version_transfer; official held-out folds; "
            "not an exact original-data reproduction"
        ),
        "private_data_used": False,
        "unlabeled_36_used_as_soz_target": False,
        "preprocessing": {
            "channels": list(CHANNELS),
            "reference": "physical REF with published zero-fill compatibility",
            "sampling_hz": 200,
            "band_hz": [1.6, 30.0],
            "butterworth_order": 4,
            "filter_phase": "forward_backward_gustafsson",
            "clip_sd": 2.0,
            "seconds_per_context": 600,
            "record_level_normalization": True,
        },
        "endpoint": {
            "primary": "C18 exact positive-set membership Top-1",
            "masked_channel": "PZ",
            "secondary": ["official-neighborhood-2", "official-neighborhood-4"],
        },
        "patient_count": len(local_ids),
        "per_fold_prediction_count": len(prediction_rows),
        "repeat_metrics": repeat_metrics,
        "pooled_repeat_metrics": _metrics(prediction_rows),
        "held_out_ensemble_metrics": _metrics(ensemble_rows),
        "patient_receipts": patient_receipts,
        "fold_predictions": prediction_rows,
        "held_out_ensemble_predictions": ensemble_rows,
        "elapsed_sec": time.monotonic() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signal-artifact",
        default="outputs/deepsoz_signal_preflight_identity_v3_20260812/deepsoz_signal_preflight_identity_v3.json",
    )
    parser.add_argument(
        "--target-csv",
        default="outputs/deepsoz_target_v2_identity_recovery_20260812/patient_targets_v2.csv",
    )
    parser.add_argument(
        "--local-manifest",
        default="outputs/labram_identity_recovery_closed_replay_v16_20260812/manifest.json",
    )
    parser.add_argument("--tusz-root", default="/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
    parser.add_argument("--fold-directory", default="/tmp")
    parser.add_argument("--weights", default="models/deepsoz_official_weights")
    parser.add_argument(
        "--output", default="outputs/deepsoz_official_local_oof_full.json"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--patients", default="")
    parser.add_argument("--max-patients", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = evaluate(arguments)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), **result["held_out_ensemble_metrics"]}, sort_keys=True))
