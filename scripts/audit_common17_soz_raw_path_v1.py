#!/usr/bin/env python3
"""Audit a strict signal-level common-17 path for the DeepSOZ 102 cohort.

The direct path reads only ``STANDARD_19 - {FZ, PZ}`` from each EDF, applies
the frozen causal filter/resample contract independently per retained
channel, and optionally applies CAR17.  A second audit path replays the
historical CAR19 loader and proves that retained CAR19 waveforms can be
converted to CAR17 by subtracting the retained-channel mean.

This script is deliberately target-free.  It reads event identities/onsets
but never opens SOZ labels, channel annotations, EDF annotations, Excel data,
or private clinical references.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pyedflib
from safetensors.torch import save_file
from scipy.signal import firwin, sosfilt, upfirdn
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.edf import (  # noqa: E402
    CausalEDFConfig,
    _causal_bandpass_sos,
    _max_extreme_run_samples,
    _max_flatline_run_samples,
    _rational_resampling,
    load_standard19_edf_event,
)
from src.soz.geometry import STANDARD_19, normalize_electrode_name  # noqa: E402
from src.soz.models.labram import (  # noqa: E402
    LABRAM_POSITION_ID_BY_NAME,
    LABRAM_RAW_HEADER_POSITION_BINDING_POLICY,
    _raw_electrode_position_name,
)


SCHEMA = "clinical_eeg_common17_soz_raw_path_audit_v1"
EXCLUDED = ("FZ", "PZ")
COMMON17 = tuple(channel for channel in STANDARD_19 if channel not in EXCLUDED)
COMMON17_INDICES = tuple(STANDARD_19.index(channel) for channel in COMMON17)
DEFAULT_UNION = ROOT / "outputs/public_development_union_identity_v12_20260812/manifest.json"
DEFAULT_ROSTER = (
    ROOT
    / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815/manifest.json"
)
DEFAULT_SIGNAL = (
    ROOT
    / "outputs/deepsoz_signal_preflight_identity_v3_20260812/"
    "deepsoz_signal_preflight_identity_v3.json"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_HISTORICAL_RAW_CACHE = ROOT / "outputs/trustworthy_soz_raw200_events_v60_20260816"
DEFAULT_OUTPUT = ROOT / "outputs/common17_soz_raw_path_smoke_v1_20260824"
_UNIT_TO_VOLTS = {"v": 1.0, "mv": 1e-3, "uv": 1e-6}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _half_up_samples(seconds: float, sfreq_hz: float) -> int:
    if not math.isfinite(float(seconds)) or float(seconds) < 0:
        raise ValueError("Time values must be finite and non-negative")
    exact = Decimal(str(float(seconds))) * Decimal(str(float(sfreq_hz)))
    return int(exact.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _unit_scale(unit: object) -> float:
    normalized = str(unit).strip().lower().replace("µ", "u").replace("μ", "u")
    try:
        return _UNIT_TO_VOLTS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported physical unit: {unit!r}") from exc


def _safe_edf(root: Path, relative: object) -> Path:
    value = PurePosixPath(str(relative))
    if value.is_absolute() or ".." in value.parts or value.suffix.lower() != ".edf":
        raise ValueError(f"Unsafe relative EDF path: {relative!r}")
    path = root.joinpath(*value.parts).resolve(strict=True)
    path.relative_to(root)
    return path


def _channel_dim(value: torch.Tensor) -> int:
    if value.ndim < 2:
        raise ValueError("Waveform must have at least channel and time dimensions")
    return value.ndim - 2


def reference_to_car17(reference17: torch.Tensor) -> torch.Tensor:
    """Apply CAR17 to ``[...,17,T]`` without any omitted-channel input."""

    value = torch.as_tensor(reference17)
    channel_dim = _channel_dim(value)
    if value.shape[channel_dim] != len(COMMON17):
        raise ValueError("reference_to_car17 expects [...,17,T]")
    return (value - value.mean(dim=channel_dim, keepdim=True)).contiguous()


def recover_car17_from_retained_car19(car19: torch.Tensor) -> torch.Tensor:
    """Recover CAR17 from ``[...,19,T]`` CAR19 or retained ``[...,17,T]``.

    If ``y_i = x_i - mean_19(x)`` for every retained electrode, then
    ``y_i - mean_R(y) = x_i - mean_R(x)``.  FZ/PZ therefore cancel as a
    channel-common offset and cannot affect the recovered CAR17 waveform.
    """

    value = torch.as_tensor(car19)
    channel_dim = _channel_dim(value)
    n_channels = int(value.shape[channel_dim])
    if n_channels == len(STANDARD_19):
        indices = torch.tensor(COMMON17_INDICES, dtype=torch.long, device=value.device)
        retained = value.index_select(channel_dim, indices)
    elif n_channels == len(COMMON17):
        retained = value
    else:
        raise ValueError("CAR19 replay expects [...,19,T] or retained [...,17,T]")
    return reference_to_car17(retained)


def _filter_resample_common17(
    raw: np.ndarray,
    *,
    source_sfreq_hz: float,
    config: CausalEDFConfig,
) -> tuple[np.ndarray, int, int, int, float]:
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(COMMON17) or values.shape[1] < 2:
        raise ValueError("common17 preprocessing expects [17,T]")
    if not np.isfinite(values).all():
        raise ValueError("EDF payload contains non-finite samples")
    filtered = sosfilt(
        _causal_bandpass_sos(source_sfreq_hz, config), values, axis=-1
    )
    up, down = _rational_resampling(source_sfreq_hz, config.output_sfreq_hz)
    max_rate = max(up, down)
    half_length = int(config.fir_half_length_per_rate) * max_rate
    taps = firwin(2 * half_length + 1, 1.0 / max_rate, window=("kaiser", 5.0))
    taps *= up
    resampled = upfirdn(taps, filtered, up=up, down=down, axis=-1)
    latency_sec = half_length / (up * float(source_sfreq_hz))
    return resampled, up, down, len(taps), latency_sec


def load_common17_edf_event(
    edf_path: Path,
    onset_sec: float,
    *,
    config: CausalEDFConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Load one event while never reading FZ/PZ samples or EDF annotations."""

    if config.reference_policy != "primary_ref" or config.sensitivity_reference is not None:
        raise ValueError("Strict public common17 audit requires primary_ref")
    reader = pyedflib.EdfReader(str(edf_path))
    try:
        labels = tuple(str(value).strip() for value in reader.getSignalLabels())
        candidates: dict[str, list[int]] = {channel: [] for channel in COMMON17}
        for index, label in enumerate(labels):
            canonical = normalize_electrode_name(label)
            if canonical in candidates:
                candidates[canonical].append(index)
        missing = [channel for channel, indices in candidates.items() if not indices]
        duplicates = {
            channel: [labels[index] for index in indices]
            for channel, indices in candidates.items()
            if len(indices) > 1
        }
        if missing or duplicates:
            raise ValueError(
                f"EDF lacks unambiguous common17 channels; missing={missing}, "
                f"duplicates={duplicates}"
            )
        indices = tuple(candidates[channel][0] for channel in COMMON17)
        selected_names = tuple(labels[index] for index in indices)
        labram_position_names = tuple(
            _raw_electrode_position_name(name) for name in selected_names
        )
        if any(
            normalize_electrode_name(position) != semantic
            for position, semantic in zip(labram_position_names, COMMON17)
        ):
            raise ValueError("Record-specific LaBraM positions disagree with common17")
        labram_position_ids = tuple(
            LABRAM_POSITION_ID_BY_NAME[name] for name in labram_position_names
        )
        if len(set(labram_position_ids)) != len(COMMON17):
            raise ValueError("Record-specific common17 LaBraM positions are duplicated")
        references = tuple(
            "REF" if name.upper().replace("_", "-").endswith("-REF") else ""
            for name in selected_names
        )
        if any(reference != "REF" for reference in references):
            raise ValueError("Strict common17 source channels must uniformly encode -REF")
        sampling_rates = tuple(float(reader.getSampleFrequency(index)) for index in indices)
        if any(not math.isfinite(value) or value <= 0 for value in sampling_rates):
            raise ValueError("Invalid common17 source sampling frequency")
        source_sfreq = sampling_rates[0]
        if any(abs(value - source_sfreq) > 1e-9 for value in sampling_rates):
            raise ValueError("Mixed common17 source sampling frequencies")
        sample_counts_raw = reader.getNSamples()
        sample_counts = tuple(int(sample_counts_raw[index]) for index in indices)
        if any(value <= 0 for value in sample_counts) or len(set(sample_counts)) != 1:
            raise ValueError("Mixed or invalid common17 source sample counts")
        units = tuple(str(reader.getPhysicalDimension(index)).strip() for index in indices)
        scales = np.asarray([_unit_scale(unit) for unit in units], dtype=np.float64)

        up, down = _rational_resampling(source_sfreq, config.output_sfreq_hz)
        half_length = int(config.fir_half_length_per_rate) * max(up, down)
        latency_sec = half_length / (up * source_sfreq)
        onset_sample = _half_up_samples(float(onset_sec), source_sfreq)
        read_start = onset_sample - _half_up_samples(
            config.warmup_sec + config.pre_onset_sec, source_sfreq
        )
        read_stop = onset_sample + _half_up_samples(
            config.post_onset_sec + latency_sec + 1.0 / source_sfreq,
            source_sfreq,
        )
        if read_start < 0 or read_stop > sample_counts[0]:
            raise ValueError("Event lacks strict common17 warmup/pre/post support")
        n_read = read_stop - read_start
        # This is the central firewall: only the 17 retained EDF indices are read.
        raw = np.stack(
            [
                np.asarray(reader.readSignal(index, read_start, n_read), dtype=np.float64)
                for index in indices
            ]
        )
        if tuple(raw.shape) != (len(COMMON17), n_read):
            raise RuntimeError("EDF returned an unexpected common17 payload shape")
        raw_volts = raw * scales[:, None]
        flatline_limit = _half_up_samples(config.flatline_run_sec, source_sfreq)
        clipping_limit = _half_up_samples(config.clipping_run_sec, source_sfreq)
        bad_flatline = [
            COMMON17[index]
            for index, channel in enumerate(raw_volts)
            if _max_flatline_run_samples(channel, config.qc_tolerance_volts)
            >= flatline_limit
        ]
        bad_clipping = [
            COMMON17[index]
            for index, channel in enumerate(raw_volts)
            if _max_extreme_run_samples(channel, config.qc_tolerance_volts)
            >= clipping_limit
        ]
        if bad_flatline or bad_clipping:
            raise ValueError(
                f"Retained-channel raw QC failed; flatline={bad_flatline}, "
                f"clipping={bad_clipping}"
            )

        processed, actual_up, actual_down, taps, actual_latency = (
            _filter_resample_common17(
                raw, source_sfreq_hz=source_sfreq, config=config
            )
        )
        if (actual_up, actual_down) != (up, down) or abs(actual_latency - latency_sec) > 1e-12:
            raise RuntimeError("common17 resampling receipt drifted")
        # Match the frozen standard-19 numerical order: float32 first, then unit scale.
        reference = torch.from_numpy(processed).to(dtype=torch.float32)
        reference = reference * reference.new_tensor(scales).unsqueeze(1)
        onset_in_processed = (onset_sample - read_start) / source_sfreq + latency_sec
        output_onset = _half_up_samples(onset_in_processed, config.output_sfreq_hz)
        pre_samples = _half_up_samples(config.pre_onset_sec, config.output_sfreq_hz)
        post_samples = _half_up_samples(config.post_onset_sec, config.output_sfreq_hz)
        start = output_onset - pre_samples
        stop = output_onset + post_samples
        reference = reference[:, start:stop].contiguous()
        if tuple(reference.shape) != (len(COMMON17), pre_samples + post_samples):
            raise RuntimeError("common17 event crop has an unexpected shape")
        car17 = reference_to_car17(reference)
        receipt: dict[str, object] = {
            "semantic_channels": list(COMMON17),
            "excluded_channels": list(EXCLUDED),
            "selected_raw_names": list(selected_names),
            "selected_edf_indices": list(indices),
            "labram_position_binding_policy": (
                LABRAM_RAW_HEADER_POSITION_BINDING_POLICY
            ),
            "labram_position_names": list(labram_position_names),
            "labram_position_ids": list(labram_position_ids),
            "excluded_samples_read": False,
            "edf_annotations_read": False,
            "source_reference": "REF",
            "output_references": ["source_uniform_REF", "common_average_standard17"],
            "source_sfreq_hz": source_sfreq,
            "output_sfreq_hz": float(config.output_sfreq_hz),
            "read_start_sample": read_start,
            "read_stop_sample": read_stop,
            "resample_up": up,
            "resample_down": down,
            "resample_fir_taps": taps,
            "resample_latency_sec": latency_sec,
            "window_shape": list(reference.shape),
        }
        return reference, car17, receipt
    finally:
        reader.close()


def _primary_events(
    union: Mapping[str, object], roster: Mapping[str, object]
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    patient_ids = tuple(str(value) for value in roster.get("patient_ids", ()))
    if len(patient_ids) != 102 or len(set(patient_ids)) != 102:
        raise ValueError("Primary patient roster must contain exactly 102 unique IDs")
    patient_set = set(patient_ids)
    source_events = union.get("events")
    if not isinstance(source_events, list) or len(source_events) != 1_149:
        raise ValueError("Identity-v12 union must contain exactly 1,149 events")
    events = [dict(row) for row in source_events if str(row.get("patient_id")) in patient_set]
    identities = [(str(row.get("event_id")), str(row.get("patient_id"))) for row in events]
    if len(events) != 1_145 or len(set(identities)) != len(events):
        raise ValueError("Primary common17 event selection must yield 1,145 unique events")
    if {patient for _, patient in identities} != patient_set:
        raise ValueError("Primary common17 event selection lost a patient")
    return events, patient_ids


def _evenly_spaced_rows(events: Sequence[dict[str, object]], count: int) -> list[int]:
    if count < 1 or count > len(events):
        raise ValueError("smoke event count is outside the primary event roster")
    return np.linspace(0, len(events) - 1, count).round().astype(int).tolist()


def audit(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    started = time.monotonic()
    union = _read_json(args.union)
    roster = _read_json(args.roster)
    signal = _read_json(args.signal)
    events, patient_ids = _primary_events(union, roster)
    config_payload = signal.get("receipt", {}).get("preprocess_config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("Signal artifact lacks the frozen preprocessing config")
    historical_config = CausalEDFConfig(**dict(config_payload))
    if not historical_config.apply_car19:
        raise ValueError("Historical signal artifact is not CAR19")
    direct_config = replace(historical_config, apply_car19=False)
    root = args.tusz_root.resolve(strict=True)
    missing_paths = [
        str(row["relative_edf_path"])
        for row in events
        if not root.joinpath(*PurePosixPath(str(row["relative_edf_path"])).parts).is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(f"Primary event roster has missing EDF paths: {missing_paths[:3]}")

    selected_rows = _evenly_spaced_rows(events, args.smoke_events)
    reference_rows: list[torch.Tensor] = []
    car17_rows: list[torch.Tensor] = []
    recovered_rows: list[torch.Tensor] = []
    smoke_receipts: list[dict[str, object]] = []
    per_event_seconds: list[float] = []
    for ordinal, row_index in enumerate(selected_rows, start=1):
        event_started = time.monotonic()
        event = events[row_index]
        path = _safe_edf(root, event["relative_edf_path"])
        reference17, direct_car17, direct_receipt = load_common17_edf_event(
            path, float(event["global_t0_sec"]), config=direct_config
        )
        historical = load_standard19_edf_event(
            path,
            float(event["global_t0_sec"]),
            config=historical_config,
            use_edf_gap_annotations_for_signal_qc=False,
        )
        recovered = recover_car17_from_retained_car19(historical.window.data.float())
        difference = direct_car17 - recovered
        direct_scale = max(float(direct_car17.abs().max()), torch.finfo(torch.float32).tiny)
        elapsed = time.monotonic() - event_started
        per_event_seconds.append(elapsed)
        reference_rows.append(reference17.mul(1_000_000.0).contiguous())
        car17_rows.append(direct_car17.mul(1_000_000.0).contiguous())
        recovered_rows.append(recovered.mul(1_000_000.0).contiguous())
        smoke_receipts.append(
            {
                "smoke_ordinal": ordinal,
                "primary_event_row": row_index,
                "event_id": str(event["event_id"]),
                "patient_id": str(event["patient_id"]),
                "official_split": str(event.get("official_split", "")),
                "relative_edf_path": str(event["relative_edf_path"]),
                "global_t0_sec": float(event["global_t0_sec"]),
                "direct_loader": direct_receipt,
                "historical_car19_output_reference": historical.signal_receipt.output_reference,
                "max_abs_difference_volts": float(difference.abs().max()),
                "mean_abs_difference_volts": float(difference.abs().mean()),
                "max_relative_to_direct_peak": float(difference.abs().max()) / direct_scale,
                "direct_car17_channel_sum_max_abs_volts": float(
                    direct_car17.sum(dim=0).abs().max()
                ),
                "recovered_car17_channel_sum_max_abs_volts": float(
                    recovered.sum(dim=0).abs().max()
                ),
                "direct_car17_sha256": _tensor_sha256(direct_car17),
                "recovered_car17_sha256": _tensor_sha256(recovered),
                "elapsed_sec": elapsed,
            }
        )
        print(
            json.dumps(
                {
                    "smoke": ordinal,
                    "total": len(selected_rows),
                    "event_id": event["event_id"],
                    "max_abs_difference_volts": smoke_receipts[-1][
                        "max_abs_difference_volts"
                    ],
                    "elapsed_sec": round(elapsed, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    reference_tensor = torch.stack(reference_rows).float().contiguous()
    car17_tensor = torch.stack(car17_rows).float().contiguous()
    recovered_tensor = torch.stack(recovered_rows).float().contiguous()
    mean_event_sec = float(np.mean(per_event_seconds))
    unique_records = len({str(row["relative_edf_path"]) for row in events})
    one_representation_bytes = len(events) * len(COMMON17) * 12_000 * 4
    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_signal_level_smoke",
        "scope": {
            "primary_patient_count": len(patient_ids),
            "primary_event_count": len(events),
            "unique_edf_record_count": unique_records,
            "union_event_count_before_primary_roster_filter": len(union["events"]),
            "all_primary_edf_paths_available": True,
            "primary_event_roster_sha256": _canonical_sha256(
                [(row["event_id"], row["patient_id"]) for row in events]
            ),
        },
        "channel_contract": {
            "channels": list(COMMON17),
            "excluded": list(EXCLUDED),
            "FZ_or_PZ_read_by_direct_loader": False,
            "FZ_or_PZ_used_in_direct_reference": False,
            "FZ_or_PZ_used_in_direct_features": False,
            "direct_references": ["source_uniform_REF", "common_average_standard17"],
            "zero_fill_interpolation_or_synthesis": False,
        },
        "algebraic_reconstruction": {
            "possible_from_retained_car19_waveforms": True,
            "formula": "CAR17(y_R) = y_R - mean_R(y_R), where y_R = x_R - mean_19(x)",
            "proof": (
                "y_R - mean_R(y_R) = x_R - mean_19(x) - "
                "[mean_R(x_R) - mean_19(x)] = x_R - mean_R(x_R)"
            ),
            "omitted_channel_contribution": "cancels_as_a_common_offset",
            "valid_stage": "waveform_before_channelwise_nonlinear_encoding",
            "invalid_stage": (
                "post-LaBraM/post-CNN/nonlinear channel features cannot in general be "
                "converted from CAR19 to strict common17"
            ),
        },
        "source_cache_audit": {
            "historical_raw200_cache_path": str(args.historical_raw_cache),
            "historical_raw200_cache_currently_available": args.historical_raw_cache.is_dir(),
            "current_feature_caches_are_waveform_reconstructible": False,
            "raw_edf_reconstruction_available": True,
        },
        "preprocessing": {
            **asdict(direct_config),
            "apply_car19": False,
            "final_optional_reference": "CAR17",
            "edf_annotations_used": False,
        },
        "smoke": {
            "event_count": len(selected_rows),
            "selection": "deterministic_even_spacing_over_frozen_1145_event_order",
            "events": smoke_receipts,
            "max_abs_difference_volts": max(
                float(row["max_abs_difference_volts"]) for row in smoke_receipts
            ),
            "mean_abs_difference_volts": float(
                np.mean([float(row["mean_abs_difference_volts"]) for row in smoke_receipts])
            ),
            "mean_elapsed_sec_per_event_for_dual_direct_plus_historical_replay": mean_event_sec,
        },
        "full_materialization_estimate": {
            "events": len(events),
            "shape_per_event": [len(COMMON17), 12_000],
            "one_float32_representation_bytes": one_representation_bytes,
            "one_float32_representation_mib": one_representation_bytes / (1024**2),
            "two_ref17_plus_car17_representations_mib": 2
            * one_representation_bytes
            / (1024**2),
            "dual_replay_linear_wall_time_sec": mean_event_sec * len(events),
            "direct_only_rough_upper_bound_sec": mean_event_sec * len(events),
            "full_cache_started": False,
        },
        "tensor_file": "common17_smoke_waveforms.safetensors",
        "tensor_specs": {
            "ref17_waveform_microvolts": list(reference_tensor.shape),
            "car17_direct_waveform_microvolts": list(car17_tensor.shape),
            "car17_recovered_from_car19_microvolts": list(recovered_tensor.shape),
        },
        "access_receipt": {
            "raw_public_eeg_loaded": True,
            "SOZ_target_values_loaded": False,
            "channel_annotations_loaded": False,
            "edf_annotations_loaded": False,
            "excel_or_clinical_text_loaded": False,
            "private_data_loaded": False,
            "model_training_performed": False,
        },
        "elapsed_sec": time.monotonic() - started,
    }
    tensors = {
        "ref17_waveform_microvolts": reference_tensor,
        "car17_direct_waveform_microvolts": car17_tensor,
        "car17_recovered_from_car19_microvolts": recovered_tensor,
    }
    return result, tensors


def publish(
    output: Path, result: Mapping[str, object], tensors: Mapping[str, torch.Tensor]
) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        save_file(dict(tensors), str(staging / str(result["tensor_file"])))
        (staging / "receipt.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--union", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--signal", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument(
        "--historical-raw-cache", type=Path, default=DEFAULT_HISTORICAL_RAW_CACHE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-events", type=int, default=6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, tensors = audit(args)
    output = publish(args.output, result, tensors)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output),
                "patients": result["scope"]["primary_patient_count"],
                "events": result["scope"]["primary_event_count"],
                "smoke_events": result["smoke"]["event_count"],
                "max_abs_difference_volts": result["smoke"][
                    "max_abs_difference_volts"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
