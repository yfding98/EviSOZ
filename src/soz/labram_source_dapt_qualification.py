"""Paired representation qualification for the locked source-only LaBraM DAPT.

This module is deliberately downstream-target blind.  It replays the frozen
pretext-dev windows and masks for the zero-LoRA and selected DAPT arms, and it
implements only the four representation gates frozen in protocol sections
8.4--8.5.  It never accepts SOZ targets, event times, or private data.

The original DAPT data/model/runner modules are hash-bound into the completed
training receipt.  Consequently the C-REF19 sensitivity reader lives here,
rather than changing any of those completed-run implementation files.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .data.labram_source_dapt import (
    CAUSAL_WARMUP_SECONDS,
    OUTPUT_SFREQ_HZ,
    SAMPLES_PER_TOKEN,
    STRIDE_SECONDS,
    TOKENS_PER_CHANNEL,
    WINDOW_SECONDS,
    SourceDAPTWindowDataset,
)
from .geometry import STANDARD_19
from .preprocessing_arm_runtime import (
    PhysicalEDFRecord,
    _apply_car,
    _causal_pre_reference_interval,
)


QUALIFICATION_SCHEMA_VERSION = "soz_labram_source_only_dapt_paired_qualification_v1"
QUALIFICATION_PROTOCOL_VERSION = "labram-source-only-dapt-v1"
QUALIFICATION_SEED = 20260811
QUALIFICATION_PATIENTS = 12
QUALIFICATION_WINDOWS_PER_PATIENT = 32
QUALIFICATION_BOOTSTRAP_REPLICATES = 10_000
QUALIFICATION_CI = (0.025, 0.975)
QUALIFICATION_CODEBOOK_SIZE = 8192
QUALIFICATION_TOKENS_PER_WINDOW = 152

MARGIN_ACCURACY = 0.0
MARGIN_LOG_PERPLEXITY = math.log(0.90)
MARGIN_TOP_FRACTION = 0.05
MARGIN_REFERENCE_JSD = 0.0


def canonical_json_bytes(value: object) -> bytes:
    """Return the one allowed canonical JSON representation for hashes/output."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path_value: str | Path) -> str:
    path = Path(path_value).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Qualification lineage input is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def ordered_window_identity(
    patient_id: str, record_uid: str, grid_index: int
) -> dict[str, object]:
    """Canonical object for one sampler draw; duplicate draws remain duplicated."""

    return {
        "patient_id": str(patient_id),
        "record_uid": str(record_uid),
        "grid_index": int(grid_index),
    }


def fixed_mask_sha256(masks: torch.Tensor) -> str:
    if masks.dtype != torch.bool or masks.ndim != 2 or masks.shape[1] != 152:
        raise ValueError("Qualification fixed masks must be bool [N,152]")
    array = masks.detach().to(device="cpu").numpy().astype(np.uint8, copy=False)
    header = np.asarray(array.shape, dtype="<i8").tobytes(order="C")
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def patient_bootstrap_draws(
    *,
    patient_count: int = QUALIFICATION_PATIENTS,
    replicates: int = QUALIFICATION_BOOTSTRAP_REPLICATES,
    seed: int = QUALIFICATION_SEED,
) -> np.ndarray:
    """Frozen PCG64 same-index patient draws shared by every paired metric."""

    if patient_count != 12 or replicates != 10_000 or seed != 20260811:
        raise ValueError("Formal paired bootstrap is frozen to 12 x 10,000, seed 20260811")
    generator = np.random.Generator(np.random.PCG64(seed))
    draws = generator.integers(
        0,
        patient_count,
        size=(replicates, patient_count),
        dtype=np.int64,
    )
    if draws.shape != (10_000, 12) or not (0 <= draws).all() or not (draws < 12).all():
        raise RuntimeError("Frozen patient bootstrap produced invalid index draws")
    return draws


def patient_index_draws_sha256(draws: np.ndarray) -> str:
    values = np.asarray(draws)
    if values.shape != (10_000, 12) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Bootstrap draw digest requires int [10000,12]")
    canonical = np.asarray(values, dtype="<i8", order="C")
    # Frozen encoding: raw little-endian int64 C-order bytes, with no shape
    # header.  Shape is independently fixed to [10000,12] above.
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def paired_percentile_interval(
    patient_values: Sequence[float], draws: np.ndarray
) -> tuple[float, float]:
    values = np.asarray(patient_values, dtype=np.float64)
    indices = np.asarray(draws)
    if values.shape != (12,) or indices.shape != (10_000, 12):
        raise ValueError("Paired qualification requires 12 patient values and fixed draws")
    if not np.isfinite(values).all():
        raise ValueError("Paired qualification patient values must be finite")
    replicate_means = values[indices].mean(axis=1, dtype=np.float64)
    lower, upper = np.quantile(
        replicate_means,
        np.asarray(QUALIFICATION_CI, dtype=np.float64),
        method="linear",
    )
    return float(lower), float(upper)


def jensen_shannon_from_logits(
    primary_logits: torch.Tensor, sensitivity_logits: torch.Tensor
) -> torch.Tensor:
    """Per-token 8192-way JSD, evaluated stably in log-probability space."""

    if primary_logits.shape != sensitivity_logits.shape or primary_logits.ndim != 2:
        raise ValueError("JSD logits must have the same [tokens,codes] shape")
    if primary_logits.shape[1] != QUALIFICATION_CODEBOOK_SIZE:
        raise ValueError("Qualification JSD requires the official 8192-code head")
    if not torch.isfinite(primary_logits).all() or not torch.isfinite(
        sensitivity_logits
    ).all():
        raise ValueError("Qualification JSD logits must be finite")
    # Q4 has a zero non-inferiority margin.  Evaluate the probability algebra
    # in float64 with natural logs so float32 softmax round-off cannot decide it.
    log_primary = F.log_softmax(primary_logits.to(dtype=torch.float64), dim=-1)
    log_sensitivity = F.log_softmax(
        sensitivity_logits.to(dtype=torch.float64), dim=-1
    )
    log_midpoint = torch.logaddexp(log_primary, log_sensitivity) - math.log(2.0)
    jsd = 0.5 * (
        torch.sum(
            torch.exp(log_primary) * (log_primary - log_midpoint), dim=-1
        )
        + torch.sum(
            torch.exp(log_sensitivity) * (log_sensitivity - log_midpoint),
            dim=-1,
        )
    )
    # The mathematical quantity is non-negative.  Clamp only tiny floating
    # round-off below zero; a materially negative value is an implementation error.
    if torch.any(jsd < -1e-6) or not torch.isfinite(jsd).all():
        raise RuntimeError("Softmax JSD became materially negative or non-finite")
    return torch.clamp_min(jsd, 0.0)


def _unit_scale(unit: object) -> float:
    key = str(unit).strip().lower().replace("µ", "u").replace("μ", "u")
    scales = {"v": 1.0, "mv": 1e-3, "uv": 1e-6}
    if key not in scales:
        raise ValueError(f"Unsupported physical EDF unit in qualification: {unit!r}")
    return scales[key]


def _file_identity(descriptor: int) -> tuple[int, int, int, int, int]:
    value = os.fstat(descriptor)
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("Qualification EDF descriptor is not a regular file")
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def read_causal_reference_pair(
    source: Path,
    row: Mapping[str, object],
    *,
    grid_index: int,
    expected_file_identity: tuple[int, int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Read one signal-only finite segment and derive paired REF19/CAR19 views.

    Both returned views come from one direct physical payload and one call to
    the already-audited causal filter/resample/crop implementation.  No EDF
    annotation method or sidecar is opened.
    """

    try:
        import pyedflib
    except ImportError as exc:  # pragma: no cover - deployment dependency gate
        raise RuntimeError("pyedflib is required for DAPT qualification") from exc

    if grid_index < int(row["first_grid_index"]):
        raise ValueError("Qualification grid window lacks real causal warmup")
    start_sec = int(grid_index) * STRIDE_SECONDS
    stop_sec = start_sec + WINDOW_SECONDS
    sfreq = float(row["source_sfreq_hz"])
    state_reset = int(math.floor((start_sec - CAUSAL_WARMUP_SECONDS) * sfreq + 1e-12))
    read_stop = min(
        int(row["source_sample_count"]),
        int(math.ceil(stop_sec * sfreq - 1e-12)) + 1,
    )
    if state_reset < 0 or read_stop <= state_reset:
        raise ValueError("Qualification causal source support is invalid")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        if _file_identity(descriptor) != expected_file_identity:
            raise RuntimeError("Qualification EDF identity differs from primary loader")
        reader = pyedflib.EdfReader(f"/proc/self/fd/{descriptor}")
        try:
            labels = tuple(str(value).strip() for value in reader.getSignalLabels())
            expected_names = tuple(str(value) for value in row["raw_channel_names"])
            indices: list[int] = []
            for expected in expected_names:
                matches = [index for index, label in enumerate(labels) if label == expected]
                if len(matches) != 1:
                    raise ValueError("Qualification raw EDF channel binding changed")
                indices.append(matches[0])
            frequencies = tuple(float(reader.getSampleFrequency(index)) for index in indices)
            units = tuple(str(reader.getPhysicalDimension(index)).strip() for index in indices)
            if frequencies != (sfreq,) * 19 or units != tuple(row["raw_units"]):
                raise ValueError("Qualification EDF frequency/unit contract changed")
            n_read = read_stop - state_reset
            raw = np.stack(
                [
                    np.asarray(reader.readSignal(index, state_reset, n_read), dtype=np.float64)
                    for index in indices
                ]
            )
        finally:
            reader.close()
        if _file_identity(descriptor) != expected_file_identity:
            raise RuntimeError("Qualification EDF changed while reading its signal payload")
    finally:
        os.close(descriptor)

    if raw.shape != (19, read_stop - state_reset) or not np.isfinite(raw).all():
        raise ValueError("Qualification direct physical payload is invalid")
    scales = np.asarray([_unit_scale(unit) for unit in row["raw_units"]], dtype=np.float64)
    raw_volts = np.ascontiguousarray(raw * scales[:, None], dtype=np.float64)
    segment_start_sec = state_reset / sfreq
    relative_start_sec = start_sec - segment_start_sec
    relative_stop_sec = stop_sec - segment_start_sec
    record = PhysicalEDFRecord(
        path=source.resolve(strict=True),
        channel_names=tuple(str(value) for value in row["raw_channel_names"]),
        channel_keys=STANDARD_19,
        units=tuple(str(value) for value in row["raw_units"]),
        source_sfreq_hz=sfreq,
        values_volts=raw_volts,
    )
    # Use the pre-dataclass helper directly.  It produces both float32 REF and
    # float64-derived CAR from one ``processed`` array and one logical crop.
    # ``PreparedCausalReferencePair`` intentionally checks whether CAR can be
    # reconstructed from an already quantized float32 REF; that is a different
    # numerical statement and is reported below rather than imposed here.
    ref_values, car_values, shared = _causal_pre_reference_interval(
        record,
        start_sec=relative_start_sec,
        stop_sec=relative_stop_sec,
    )
    expected_shared_keys = {
        "arm_spec_receipt_sha256",
        "implementation",
        "state_scope",
        "state_reset_source_sample",
        "read_stop_source_sample",
        "real_warmup_sec",
        "source_sfreq_hz",
        "output_sfreq_hz",
        "resample_up",
        "resample_down",
        "resample_fir_taps",
        "resampling_delay_sec",
        "delay_compensated_crop_offset",
    }
    if set(shared) != expected_shared_keys:
        raise RuntimeError("Causal pre-reference shared receipt fields changed")
    if (
        shared["implementation"]
        != "scipy_sosfilt_upfirdn_finite_segment_v1"
        or shared["state_reset_source_sample"] != 0
        or shared["read_stop_source_sample"] != raw_volts.shape[1]
        or float(shared["real_warmup_sec"]) < 30.0
        or float(shared["source_sfreq_hz"]) != sfreq
        or float(shared["output_sfreq_hz"]) != OUTPUT_SFREQ_HZ
    ):
        raise RuntimeError("Causal pre-reference shared processing contract changed")
    ref = np.ascontiguousarray(
        ref_values.reshape(19, TOKENS_PER_CHANNEL, SAMPLES_PER_TOKEN),
        dtype=np.float32,
    )
    car = np.ascontiguousarray(
        car_values.reshape(19, TOKENS_PER_CHANNEL, SAMPLES_PER_TOKEN),
        dtype=np.float32,
    )
    expected_shape = (19, TOKENS_PER_CHANNEL, SAMPLES_PER_TOKEN)
    if ref.shape != expected_shape or car.shape != expected_shape:
        raise RuntimeError("Qualification reference views have invalid geometry")
    if not np.isfinite(ref).all() or not np.isfinite(car).all():
        raise RuntimeError("Qualification reference views are non-finite")
    car_from_float32_ref = np.ascontiguousarray(_apply_car(ref), dtype=np.float32)
    quantization_error = float(
        np.max(
            np.abs(
                np.asarray(car, dtype=np.float64)
                - np.asarray(car_from_float32_ref, dtype=np.float64)
            )
        )
    )
    if not math.isfinite(quantization_error) or quantization_error < 0:
        raise RuntimeError("Float32 REF-to-CAR numerical discrepancy is invalid")
    return ref, car, quantization_error


class PairedReferenceQualificationDataset(Dataset[dict[str, object]]):
    """Ordered replay wrapper around the completed-run primary C-CAR loader."""

    def __init__(
        self,
        primary_dataset: SourceDAPTWindowDataset,
        ordered_indices: Sequence[int],
        *,
        replay_atol_volts: float = 0.0,
    ) -> None:
        if not ordered_indices:
            raise ValueError("Qualification ordered window index list is empty")
        if not math.isfinite(replay_atol_volts) or replay_atol_volts < 0:
            raise ValueError("CAR replay tolerance must be finite and non-negative")
        self.primary_dataset = primary_dataset
        self.ordered_indices = tuple(int(value) for value in ordered_indices)
        self.replay_atol_volts = float(replay_atol_volts)

    def __len__(self) -> int:
        return len(self.ordered_indices)

    def __getitem__(self, ordinal: int) -> dict[str, object]:
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise IndexError(ordinal)
        dataset_index = self.ordered_indices[ordinal]
        row, expected_grid_index = self.primary_dataset.locate(dataset_index)
        primary = self.primary_dataset[dataset_index]
        if int(primary["grid_index"]) != expected_grid_index:
            raise RuntimeError("Primary DAPT locate/getitem grid identity changed")
        source = (
            self.primary_dataset.manifest.tusz_root / str(row["relative_edf_path"])
        ).resolve(strict=True)
        record_uid = str(row["record_uid"])
        identity = self.primary_dataset._verified_file_identities.get(record_uid)
        if identity is None:
            raise RuntimeError("Primary DAPT loader did not bind the EDF content identity")
        ref, replay_car, float32_ref_car_error = read_causal_reference_pair(
            source,
            row,
            grid_index=expected_grid_index,
            expected_file_identity=identity,
        )
        primary_car = primary["eeg"].detach().cpu().numpy()
        replay_error = float(
            np.max(
                np.abs(
                    np.asarray(primary_car, dtype=np.float64)
                    - np.asarray(replay_car, dtype=np.float64)
                )
            )
        )
        if replay_error > self.replay_atol_volts:
            raise RuntimeError(
                "Paired reference CAR does not replay the completed-run C-CAR19 "
                f"loader: max_abs_error_volts={replay_error:.17g}"
            )
        return {
            "primary_car": primary["eeg"],
            "sensitivity_ref": torch.from_numpy(ref),
            "position_ids": primary["position_ids"],
            "patient_id": str(primary["patient_id"]),
            "record_uid": str(primary["record_uid"]),
            "grid_index": int(primary["grid_index"]),
            "car_replay_max_abs_error_volts": replay_error,
            "car_from_float32_ref_max_abs_error_volts": float32_ref_car_error,
        }


@dataclass(frozen=True)
class QualificationArmStatistics:
    patient_ids: tuple[str, ...]
    patient_ce: np.ndarray
    patient_accuracy: np.ndarray
    patient_reference_jsd: np.ndarray
    prediction_counts: np.ndarray
    aggregate_prediction_counts: np.ndarray
    target_ids_sha256: str

    def __post_init__(self) -> None:
        if len(self.patient_ids) != 12 or tuple(sorted(self.patient_ids)) != self.patient_ids:
            raise ValueError("Qualification arm patient IDs must be 12 sorted identities")
        vectors = (self.patient_ce, self.patient_accuracy, self.patient_reference_jsd)
        if any(np.asarray(value).shape != (12,) for value in vectors):
            raise ValueError("Qualification arm metrics must contain 12 patient values")
        if any(not np.isfinite(np.asarray(value)).all() for value in vectors):
            raise ValueError("Qualification arm metrics must be finite")
        if np.asarray(self.prediction_counts).shape != (12, 8192):
            raise ValueError("Qualification patient code counts must be [12,8192]")
        if np.asarray(self.aggregate_prediction_counts).shape != (8192,):
            raise ValueError("Qualification aggregate code counts must be [8192]")
        if not isinstance(self.target_ids_sha256, str) or len(self.target_ids_sha256) != 64:
            raise ValueError("Qualification target code digest is invalid")


def _entropy_and_top_fraction(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(counts, dtype=np.int64)
    if values.shape != (12, 8192) or (values < 0).any():
        raise ValueError("Prediction counts must be non-negative int [12,8192]")
    totals = values.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("Every qualification patient must have predicted codes")
    entropy = np.empty(12, dtype=np.float64)
    top = np.empty(12, dtype=np.float64)
    for index, (row, total) in enumerate(zip(values, totals)):
        positive = row[row > 0].astype(np.float64) / float(total)
        entropy[index] = -float(np.sum(positive * np.log(positive), dtype=np.float64))
        top[index] = float(np.max(row)) / float(total)
    return entropy, top


def _metric_payload(
    values: np.ndarray,
    *,
    margin: float,
    draws: np.ndarray,
    direction: str,
    strict_mean: bool,
    strict_interval: bool,
) -> dict[str, object]:
    patient_values = np.asarray(values, dtype=np.float64)
    lower, upper = paired_percentile_interval(patient_values, draws)
    mean = float(np.mean(patient_values, dtype=np.float64))
    if direction == "greater":
        point_pass = mean > margin if strict_mean else mean >= margin
        interval_pass = lower > margin if strict_interval else lower >= margin
    elif direction == "less":
        point_pass = mean < margin if strict_mean else mean <= margin
        interval_pass = upper < margin if strict_interval else upper <= margin
    else:
        raise ValueError("Paired metric direction must be greater or less")
    return {
        "patient_values": [float(value) for value in patient_values],
        "patient_macro_mean": mean,
        "margin": float(margin),
        "ci_lower": lower,
        "ci_upper": upper,
        "passed": bool(point_pass and interval_pass),
    }


def build_paired_metrics(
    zero: QualificationArmStatistics,
    dapt: QualificationArmStatistics,
    *,
    draws: np.ndarray,
) -> dict[str, object]:
    """Compute all four frozen gates from patient-paired arm summaries."""

    if zero.patient_ids != dapt.patient_ids:
        raise ValueError("Zero/DAPT qualification patients are not aligned")
    ce = _metric_payload(
        zero.patient_ce - dapt.patient_ce,
        margin=0.0,
        draws=draws,
        direction="greater",
        strict_mean=True,
        strict_interval=True,
    )
    accuracy = _metric_payload(
        dapt.patient_accuracy - zero.patient_accuracy,
        margin=MARGIN_ACCURACY,
        draws=draws,
        direction="greater",
        strict_mean=False,
        strict_interval=False,
    )

    zero_entropy, zero_top = _entropy_and_top_fraction(zero.prediction_counts)
    dapt_entropy, dapt_top = _entropy_and_top_fraction(dapt.prediction_counts)
    log_perplexity = _metric_payload(
        dapt_entropy - zero_entropy,
        margin=MARGIN_LOG_PERPLEXITY,
        draws=draws,
        direction="greater",
        strict_mean=True,
        strict_interval=False,
    )
    top_fraction = _metric_payload(
        dapt_top - zero_top,
        margin=MARGIN_TOP_FRACTION,
        draws=draws,
        direction="less",
        strict_mean=True,
        strict_interval=False,
    )
    # Only the parent occupancy gate has a protocol field named ``passed``.
    log_perplexity_passed = bool(log_perplexity.pop("passed"))
    top_fraction_passed = bool(top_fraction.pop("passed"))
    zero_unique = np.count_nonzero(zero.prediction_counts, axis=1)
    dapt_unique = np.count_nonzero(dapt.prediction_counts, axis=1)
    zero_aggregate_unique = int(np.count_nonzero(zero.aggregate_prediction_counts))
    dapt_aggregate_unique = int(np.count_nonzero(dapt.aggregate_prediction_counts))
    target_equal = zero.target_ids_sha256 == dapt.target_ids_sha256
    unique_passed = bool(
        np.all(zero_unique >= 2)
        and np.all(dapt_unique >= 2)
        and zero_aggregate_unique >= 2
        and dapt_aggregate_unique >= 2
    )
    occupancy = {
        "target_ids_equal": bool(target_equal),
        "zero_patient_unique_counts": [int(value) for value in zero_unique],
        "dapt_patient_unique_counts": [int(value) for value in dapt_unique],
        "zero_aggregate_unique_count": zero_aggregate_unique,
        "dapt_aggregate_unique_count": dapt_aggregate_unique,
        "log_perplexity_delta": log_perplexity,
        "top_fraction_delta": top_fraction,
        "passed": bool(
            target_equal
            and unique_passed
            and log_perplexity_passed
            and top_fraction_passed
        ),
    }
    reference_jsd = _metric_payload(
        dapt.patient_reference_jsd - zero.patient_reference_jsd,
        margin=MARGIN_REFERENCE_JSD,
        draws=draws,
        direction="less",
        strict_mean=False,
        strict_interval=False,
    )
    return {
        "ce_zero_minus_dapt": ce,
        "accuracy_dapt_minus_zero": accuracy,
        "prediction_occupancy": occupancy,
        "reference_jsd_dapt_minus_zero": reference_jsd,
    }


_ROOT_KEYS = {
    "schema_version",
    "protocol_version",
    "source_run_receipt_path",
    "source_run_receipt_sha256",
    "selected_adapter_path",
    "selected_adapter_sha256",
    "manifest_path",
    "manifest_sha256",
    "qualification_cohort",
    "reference_view_contract",
    "bootstrap",
    "paired_metrics",
    "all_representation_gates_pass",
    "representation_qualified",
    "eligible_for_locked_downstream_comparison",
    "soz_promotion",
    "candidate_promotable",
    "target_values_loaded",
    "private_data_loaded",
    "annotation_times_used",
}
_COHORT_KEYS = {
    "patient_ids",
    "patient_ids_sha256",
    "patient_count",
    "windows_per_patient",
    "window_uid_sha256",
    "fixed_mask_sha256",
}
_REFERENCE_KEYS = {
    "source",
    "shared_filter_resample_crop",
    "primary",
    "sensitivity",
    "car_replay_max_abs_error_volts",
    "car_from_float32_ref_max_abs_error_volts",
}
_BOOTSTRAP_KEYS = {
    "unit",
    "replicates",
    "seed",
    "ci",
    "patient_index_draws_encoding",
    "patient_index_draws_sha256",
}
_PAIRED_KEYS = {
    "ce_zero_minus_dapt",
    "accuracy_dapt_minus_zero",
    "prediction_occupancy",
    "reference_jsd_dapt_minus_zero",
}
_METRIC_KEYS = {
    "patient_values",
    "patient_macro_mean",
    "margin",
    "ci_lower",
    "ci_upper",
    "passed",
}
_OCCUPANCY_KEYS = {
    "target_ids_equal",
    "zero_patient_unique_counts",
    "dapt_patient_unique_counts",
    "zero_aggregate_unique_count",
    "dapt_aggregate_unique_count",
    "log_perplexity_delta",
    "top_fraction_delta",
    "passed",
}
_OCCUPANCY_METRIC_KEYS = _METRIC_KEYS - {"passed"}


def _require_exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} fields changed; missing={sorted(expected-observed)}, "
            f"unknown={sorted(observed-expected)}"
        )


def _require_finite_json(value: object, *, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite_json(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_json(child, location=f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Qualification JSON contains NaN/Inf at {location}")


def _recompute_metric_contract(
    metric: Mapping[str, object],
    *,
    draws: np.ndarray,
    expected_margin: float,
    direction: str,
    strict_mean: bool,
    strict_interval: bool,
    has_passed_field: bool,
    label: str,
) -> bool:
    values = np.asarray(metric["patient_values"], dtype=np.float64)
    if values.shape != (12,) or not np.isfinite(values).all():
        raise ValueError(f"{label} must contain 12 finite patient values")
    if metric["margin"] != expected_margin:
        raise ValueError(f"{label} margin changed")
    expected_mean = float(np.mean(values, dtype=np.float64))
    expected_lower, expected_upper = paired_percentile_interval(values, draws)
    if (
        metric["patient_macro_mean"] != expected_mean
        or metric["ci_lower"] != expected_lower
        or metric["ci_upper"] != expected_upper
    ):
        raise ValueError(f"{label} mean/CI was not recomputed from patient values")
    if direction == "greater":
        point_pass = (
            expected_mean > expected_margin
            if strict_mean
            else expected_mean >= expected_margin
        )
        interval_pass = (
            expected_lower > expected_margin
            if strict_interval
            else expected_lower >= expected_margin
        )
    elif direction == "less":
        point_pass = (
            expected_mean < expected_margin
            if strict_mean
            else expected_mean <= expected_margin
        )
        interval_pass = (
            expected_upper < expected_margin
            if strict_interval
            else expected_upper <= expected_margin
        )
    else:  # pragma: no cover - caller is frozen below
        raise ValueError("Unknown qualification metric direction")
    expected_pass = bool(point_pass and interval_pass)
    if has_passed_field and metric["passed"] is not expected_pass:
        raise ValueError(f"{label} passed flag contradicts its patient values/CI")
    return expected_pass


def validate_qualification_artifact(payload: Mapping[str, object]) -> None:
    """Fail closed on missing/unknown fields, non-finite values, or contradictions."""

    if not isinstance(payload, Mapping):
        raise TypeError("Qualification artifact must be a mapping")
    _require_exact_keys(payload, _ROOT_KEYS, "qualification artifact")
    _require_finite_json(payload)
    if payload["schema_version"] != QUALIFICATION_SCHEMA_VERSION:
        raise ValueError("Qualification schema version changed")
    if payload["protocol_version"] != QUALIFICATION_PROTOCOL_VERSION:
        raise ValueError("Qualification protocol version changed")
    for field in (
        "source_run_receipt_sha256",
        "selected_adapter_sha256",
        "manifest_sha256",
    ):
        value = payload[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"Qualification lineage digest is invalid: {field}")
    for path_field, digest_field in (
        ("source_run_receipt_path", "source_run_receipt_sha256"),
        ("selected_adapter_path", "selected_adapter_sha256"),
        ("manifest_path", "manifest_sha256"),
    ):
        raw_path = payload[path_field]
        if not isinstance(raw_path, str):
            raise TypeError(f"Qualification lineage path is not text: {path_field}")
        path = Path(raw_path)
        resolved = path.resolve(strict=True)
        if not path.is_absolute() or path.is_symlink() or str(resolved) != raw_path:
            raise ValueError(f"Qualification lineage path is not canonical: {path_field}")
        if sha256_file(resolved) != payload[digest_field]:
            raise ValueError(f"Qualification lineage file/hash mismatch: {path_field}")

    cohort = payload["qualification_cohort"]
    if not isinstance(cohort, Mapping):
        raise TypeError("qualification_cohort must be a mapping")
    _require_exact_keys(cohort, _COHORT_KEYS, "qualification cohort")
    patients = cohort["patient_ids"]
    if (
        not isinstance(patients, list)
        or len(patients) != 12
        or patients != sorted(set(patients))
        or cohort["patient_count"] != 12
        or cohort["windows_per_patient"] != 32
        or cohort["patient_ids_sha256"] != sha256_json(patients)
    ):
        raise ValueError("Qualification cohort is not the canonical 12 x 32 patient cohort")
    for field in ("patient_ids_sha256", "window_uid_sha256", "fixed_mask_sha256"):
        value = cohort[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"Qualification cohort digest is invalid: {field}")

    reference = payload["reference_view_contract"]
    if not isinstance(reference, Mapping):
        raise TypeError("reference_view_contract must be a mapping")
    _require_exact_keys(reference, _REFERENCE_KEYS, "reference view contract")
    if (
        reference["source"] != "same_direct_physical_REF_payload"
        or reference["shared_filter_resample_crop"] is not True
        or reference["primary"] != "C-CAR19"
        or reference["sensitivity"] != "C-REF19"
        or float(reference["car_replay_max_abs_error_volts"]) < 0
        or float(reference["car_replay_max_abs_error_volts"]) != 0.0
        or float(reference["car_from_float32_ref_max_abs_error_volts"]) < 0
    ):
        raise ValueError("Qualification paired reference contract changed")

    bootstrap = payload["bootstrap"]
    if not isinstance(bootstrap, Mapping):
        raise TypeError("bootstrap must be a mapping")
    _require_exact_keys(bootstrap, _BOOTSTRAP_KEYS, "qualification bootstrap")
    expected_draw_digest = patient_index_draws_sha256(patient_bootstrap_draws())
    if bootstrap != {
        "unit": "patient",
        "replicates": 10_000,
        "seed": 20260811,
        "ci": [0.025, 0.975],
        "patient_index_draws_encoding": "numpy_dtype_<i8_C_order_raw_bytes_no_header",
        "patient_index_draws_sha256": expected_draw_digest,
    }:
        raise ValueError("Qualification patient bootstrap contract changed")

    paired = payload["paired_metrics"]
    if not isinstance(paired, Mapping):
        raise TypeError("paired_metrics must be a mapping")
    _require_exact_keys(paired, _PAIRED_KEYS, "paired metrics")
    for name in (
        "ce_zero_minus_dapt",
        "accuracy_dapt_minus_zero",
        "reference_jsd_dapt_minus_zero",
    ):
        metric = paired[name]
        if not isinstance(metric, Mapping):
            raise TypeError(f"{name} must be a mapping")
        _require_exact_keys(metric, _METRIC_KEYS, name)
        if not isinstance(metric["patient_values"], list) or len(metric["patient_values"]) != 12:
            raise ValueError(f"{name} must contain 12 patient values")
    draws = patient_bootstrap_draws()
    ce_pass = _recompute_metric_contract(
        paired["ce_zero_minus_dapt"],
        draws=draws,
        expected_margin=0.0,
        direction="greater",
        strict_mean=True,
        strict_interval=True,
        has_passed_field=True,
        label="Q1 CE",
    )
    accuracy_pass = _recompute_metric_contract(
        paired["accuracy_dapt_minus_zero"],
        draws=draws,
        expected_margin=0.0,
        direction="greater",
        strict_mean=False,
        strict_interval=False,
        has_passed_field=True,
        label="Q2 accuracy",
    )
    reference_pass = _recompute_metric_contract(
        paired["reference_jsd_dapt_minus_zero"],
        draws=draws,
        expected_margin=0.0,
        direction="less",
        strict_mean=False,
        strict_interval=False,
        has_passed_field=True,
        label="Q4 reference JSD",
    )

    occupancy = paired["prediction_occupancy"]
    if not isinstance(occupancy, Mapping):
        raise TypeError("prediction_occupancy must be a mapping")
    _require_exact_keys(occupancy, _OCCUPANCY_KEYS, "prediction occupancy")
    for field in ("zero_patient_unique_counts", "dapt_patient_unique_counts"):
        if (
            not isinstance(occupancy[field], list)
            or len(occupancy[field]) != 12
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 8192
                for value in occupancy[field]
            )
        ):
            raise ValueError("Prediction occupancy must contain 12 patient unique counts")
    for field, margin in (
        ("log_perplexity_delta", MARGIN_LOG_PERPLEXITY),
        ("top_fraction_delta", MARGIN_TOP_FRACTION),
    ):
        metric = occupancy[field]
        if not isinstance(metric, Mapping):
            raise TypeError(f"{field} must be a mapping")
        _require_exact_keys(metric, _OCCUPANCY_METRIC_KEYS, field)
        if (
            not isinstance(metric["patient_values"], list)
            or len(metric["patient_values"]) != 12
            or metric["margin"] != margin
        ):
            raise ValueError(f"Prediction occupancy metric changed: {field}")

    log_perplexity_pass = _recompute_metric_contract(
        occupancy["log_perplexity_delta"],
        draws=draws,
        expected_margin=MARGIN_LOG_PERPLEXITY,
        direction="greater",
        strict_mean=True,
        strict_interval=False,
        has_passed_field=False,
        label="Q3 log perplexity",
    )
    top_fraction_pass = _recompute_metric_contract(
        occupancy["top_fraction_delta"],
        draws=draws,
        expected_margin=MARGIN_TOP_FRACTION,
        direction="less",
        strict_mean=True,
        strict_interval=False,
        has_passed_field=False,
        label="Q3 top fraction",
    )
    aggregate_counts = (
        occupancy["zero_aggregate_unique_count"],
        occupancy["dapt_aggregate_unique_count"],
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 8192
        for value in aggregate_counts
    ):
        raise ValueError("Prediction aggregate unique counts are invalid")
    unique_pass = bool(
        all(value >= 2 for value in occupancy["zero_patient_unique_counts"])
        and all(value >= 2 for value in occupancy["dapt_patient_unique_counts"])
        and all(value >= 2 for value in aggregate_counts)
    )
    if not isinstance(occupancy["target_ids_equal"], bool):
        raise TypeError("Q3 target_ids_equal must be boolean")
    expected_occupancy_pass = bool(
        occupancy["target_ids_equal"]
        and unique_pass
        and log_perplexity_pass
        and top_fraction_pass
    )
    if occupancy["passed"] is not expected_occupancy_pass:
        raise ValueError("Q3 occupancy passed flag contradicts its frozen sub-gates")

    gate_values = [ce_pass, accuracy_pass, expected_occupancy_pass, reference_pass]
    if any(not isinstance(value, bool) for value in gate_values):
        raise TypeError("Qualification gate values must be booleans")
    all_pass = all(gate_values)
    if any(
        payload[field] is not all_pass
        for field in (
            "all_representation_gates_pass",
            "representation_qualified",
            "eligible_for_locked_downstream_comparison",
        )
    ):
        raise ValueError("Qualification aggregate gate flags are contradictory")
    for field in (
        "soz_promotion",
        "candidate_promotable",
        "target_values_loaded",
        "private_data_loaded",
        "annotation_times_used",
    ):
        if payload[field] is not False:
            raise ValueError(f"Qualification safety flag must remain false: {field}")


def build_qualification_artifact(
    *,
    source_run_receipt_path: Path,
    source_run_receipt_sha256: str,
    selected_adapter_path: Path,
    selected_adapter_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    patient_ids: Sequence[str],
    ordered_window_identities: Sequence[Mapping[str, object]],
    masks_sha256: str,
    car_replay_max_abs_error_volts: float,
    car_from_float32_ref_max_abs_error_volts: float,
    draws: np.ndarray,
    paired_metrics: Mapping[str, object],
) -> dict[str, object]:
    patients = list(patient_ids)
    all_pass = bool(
        paired_metrics["ce_zero_minus_dapt"]["passed"]
        and paired_metrics["accuracy_dapt_minus_zero"]["passed"]
        and paired_metrics["prediction_occupancy"]["passed"]
        and paired_metrics["reference_jsd_dapt_minus_zero"]["passed"]
    )
    payload: dict[str, object] = {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "protocol_version": QUALIFICATION_PROTOCOL_VERSION,
        "source_run_receipt_path": str(source_run_receipt_path.resolve(strict=True)),
        "source_run_receipt_sha256": source_run_receipt_sha256,
        "selected_adapter_path": str(selected_adapter_path.resolve(strict=True)),
        "selected_adapter_sha256": selected_adapter_sha256,
        "manifest_path": str(manifest_path.resolve(strict=True)),
        "manifest_sha256": manifest_sha256,
        "qualification_cohort": {
            "patient_ids": patients,
            "patient_ids_sha256": sha256_json(patients),
            "patient_count": 12,
            "windows_per_patient": 32,
            "window_uid_sha256": sha256_json(list(ordered_window_identities)),
            "fixed_mask_sha256": masks_sha256,
        },
        "reference_view_contract": {
            "source": "same_direct_physical_REF_payload",
            "shared_filter_resample_crop": True,
            "primary": "C-CAR19",
            "sensitivity": "C-REF19",
            "car_replay_max_abs_error_volts": float(
                car_replay_max_abs_error_volts
            ),
            "car_from_float32_ref_max_abs_error_volts": float(
                car_from_float32_ref_max_abs_error_volts
            ),
        },
        "bootstrap": {
            "unit": "patient",
            "replicates": 10_000,
            "seed": 20260811,
            "ci": [0.025, 0.975],
            "patient_index_draws_encoding": "numpy_dtype_<i8_C_order_raw_bytes_no_header",
            "patient_index_draws_sha256": patient_index_draws_sha256(draws),
        },
        "paired_metrics": dict(paired_metrics),
        "all_representation_gates_pass": all_pass,
        "representation_qualified": all_pass,
        "eligible_for_locked_downstream_comparison": all_pass,
        "soz_promotion": False,
        "candidate_promotable": False,
        "target_values_loaded": False,
        "private_data_loaded": False,
        "annotation_times_used": False,
    }
    validate_qualification_artifact(payload)
    return payload


__all__ = [
    "PairedReferenceQualificationDataset",
    "QualificationArmStatistics",
    "build_paired_metrics",
    "build_qualification_artifact",
    "canonical_json_bytes",
    "fixed_mask_sha256",
    "jensen_shannon_from_logits",
    "ordered_window_identity",
    "paired_percentile_interval",
    "patient_bootstrap_draws",
    "patient_index_draws_sha256",
    "read_causal_reference_pair",
    "sha256_file",
    "sha256_json",
    "validate_qualification_artifact",
]
