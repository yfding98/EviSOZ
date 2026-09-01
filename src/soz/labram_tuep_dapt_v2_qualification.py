"""Locked TUH-internal representation qualification for LaBraM DAPT-v2.

This module contains only target-free, patient-paired representation
statistics.  It deliberately does not import SOZ labels, seizure times, or
private data.  Signal replay is implemented by the separate runner, after it
has proved that the formal DAPT-v2 receipt selected a dev-eligible non-zero
adapter.

The cohort is internal to the TUH ecosystem and LaBraM may have seen its
patients during foundation pretraining.  Passing these gates therefore only
authorizes a locked downstream comparison; it is not external validation and
is not SOZ promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .labram_source_dapt_qualification import (
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)


QUALIFICATION_SCHEMA_VERSION = (
    "soz_labram_tuep_dapt_v2_internal_paired_qualification_v1"
)
QUALIFICATION_PROTOCOL_VERSION = "labram-tuep-diversity-dapt-v2"
QUALIFICATION_SCOPE = (
    "TUH-internal, target-excluded, likely pretraining-exposed"
)
QUALIFICATION_SEED = 20260812
QUALIFICATION_PATIENTS = 36
QUALIFICATION_WINDOWS_PER_PATIENT = 32
QUALIFICATION_WINDOW_DRAWS = 1_152
QUALIFICATION_BOOTSTRAP_REPLICATES = 10_000
QUALIFICATION_CI = (0.025, 0.975)
QUALIFICATION_CODEBOOK_SIZE = 8_192
QUALIFICATION_TOKENS_PER_WINDOW = 152

MARGIN_CE = 0.0
MARGIN_ACCURACY = 0.0
MARGIN_HARD_LOG_PERPLEXITY = math.log(0.90)
MARGIN_REFERENCE_JSD = 0.0


def patient_bootstrap_draws(
    *,
    patient_count: int = QUALIFICATION_PATIENTS,
    replicates: int = QUALIFICATION_BOOTSTRAP_REPLICATES,
    seed: int = QUALIFICATION_SEED,
) -> np.ndarray:
    """Return the one frozen PCG64 patient-index bootstrap matrix."""

    if (
        patient_count != QUALIFICATION_PATIENTS
        or replicates != QUALIFICATION_BOOTSTRAP_REPLICATES
        or seed != QUALIFICATION_SEED
    ):
        raise ValueError(
            "Formal DAPT-v2 qualification bootstrap is frozen to "
            "36 x 10,000 with seed 20260812"
        )
    generator = np.random.Generator(np.random.PCG64(seed))
    draws = generator.integers(
        0,
        patient_count,
        size=(replicates, patient_count),
        dtype=np.int64,
    )
    if draws.shape != (10_000, 36) or not (0 <= draws).all() or not (
        draws < 36
    ).all():
        raise RuntimeError("DAPT-v2 qualification bootstrap produced invalid draws")
    return draws


def patient_index_draws_sha256(draws: np.ndarray) -> str:
    values = np.asarray(draws)
    if values.shape != (10_000, 36) or not np.issubdtype(
        values.dtype, np.integer
    ):
        raise ValueError("DAPT-v2 bootstrap digest requires integer [10000,36]")
    canonical = np.asarray(values, dtype="<i8", order="C")
    import hashlib

    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def paired_percentile_interval(
    patient_values: Sequence[float], draws: np.ndarray
) -> tuple[float, float]:
    values = np.asarray(patient_values, dtype=np.float64)
    indices = np.asarray(draws)
    if values.shape != (36,) or indices.shape != (10_000, 36):
        raise ValueError(
            "DAPT-v2 qualification requires 36 patient values and fixed draws"
        )
    if not np.isfinite(values).all():
        raise ValueError("DAPT-v2 paired patient values must be finite")
    replicate_means = values[indices].mean(axis=1, dtype=np.float64)
    lower, upper = np.quantile(
        replicate_means,
        np.asarray(QUALIFICATION_CI, dtype=np.float64),
        method="linear",
    )
    return float(lower), float(upper)


@dataclass(frozen=True)
class QualificationArmStatistics:
    """Patient-level summaries for one arm on the same 36 x 32 replay."""

    patient_ids: tuple[str, ...]
    patient_ce: np.ndarray
    patient_accuracy: np.ndarray
    patient_reference_jsd: np.ndarray
    prediction_counts: np.ndarray
    aggregate_prediction_counts: np.ndarray
    target_ids_sha256: str

    def __post_init__(self) -> None:
        if (
            len(self.patient_ids) != QUALIFICATION_PATIENTS
            or tuple(sorted(self.patient_ids)) != self.patient_ids
            or len(set(self.patient_ids)) != QUALIFICATION_PATIENTS
        ):
            raise ValueError(
                "DAPT-v2 qualification arm requires 36 sorted patient identities"
            )
        vectors = (
            self.patient_ce,
            self.patient_accuracy,
            self.patient_reference_jsd,
        )
        if any(
            np.asarray(value).shape != (QUALIFICATION_PATIENTS,)
            for value in vectors
        ):
            raise ValueError("DAPT-v2 arm metrics must contain 36 patient values")
        if any(not np.isfinite(np.asarray(value)).all() for value in vectors):
            raise ValueError("DAPT-v2 arm metrics must be finite")
        counts = np.asarray(self.prediction_counts)
        aggregate = np.asarray(self.aggregate_prediction_counts)
        if counts.shape != (36, 8192) or (counts < 0).any():
            raise ValueError("DAPT-v2 patient code counts must be non-negative [36,8192]")
        if aggregate.shape != (8192,) or (aggregate < 0).any():
            raise ValueError("DAPT-v2 aggregate code counts must be [8192]")
        if not np.array_equal(aggregate, counts.sum(axis=0, dtype=np.int64)):
            raise ValueError("DAPT-v2 aggregate code counts do not sum patient counts")
        if not np.all(
            counts.sum(axis=1)
            == QUALIFICATION_WINDOWS_PER_PATIENT * QUALIFICATION_TOKENS_PER_WINDOW
        ):
            raise ValueError("Every DAPT-v2 patient must contribute exactly 32 x 152 codes")
        if (
            not isinstance(self.target_ids_sha256, str)
            or len(self.target_ids_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.target_ids_sha256)
        ):
            raise ValueError("DAPT-v2 target-code digest is invalid")


def _hard_code_statistics(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(counts, dtype=np.int64)
    if values.shape != (36, 8192) or (values < 0).any():
        raise ValueError("Hard prediction counts must be non-negative [36,8192]")
    totals = values.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("Every DAPT-v2 qualification patient needs predictions")
    entropy = np.empty(36, dtype=np.float64)
    top_fraction = np.empty(36, dtype=np.float64)
    unique = np.count_nonzero(values, axis=1).astype(np.int64, copy=False)
    for index, (row, total) in enumerate(zip(values, totals)):
        probabilities = row[row > 0].astype(np.float64) / float(total)
        entropy[index] = -float(
            np.sum(probabilities * np.log(probabilities), dtype=np.float64)
        )
        top_fraction[index] = float(np.max(row)) / float(total)
    return entropy, top_fraction, unique


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
    v2: QualificationArmStatistics,
    *,
    draws: np.ndarray,
) -> dict[str, object]:
    """Compute exactly Q1--Q4 from patient-paired arm summaries."""

    if zero.patient_ids != v2.patient_ids:
        raise ValueError("Zero-LoRA and DAPT-v2 patient identities are not aligned")
    if zero.target_ids_sha256 != v2.target_ids_sha256:
        raise ValueError("Zero-LoRA and DAPT-v2 did not score identical target codes")
    zero_log_ppl, zero_top, zero_unique = _hard_code_statistics(
        zero.prediction_counts
    )
    v2_log_ppl, v2_top, v2_unique = _hard_code_statistics(v2.prediction_counts)
    return {
        "q1_ce_zero_minus_v2": _metric_payload(
            zero.patient_ce - v2.patient_ce,
            margin=MARGIN_CE,
            draws=draws,
            direction="greater",
            strict_mean=True,
            strict_interval=True,
        ),
        "q2_accuracy_v2_minus_zero": _metric_payload(
            v2.patient_accuracy - zero.patient_accuracy,
            margin=MARGIN_ACCURACY,
            draws=draws,
            direction="greater",
            strict_mean=False,
            strict_interval=False,
        ),
        "q3_hard_log_perplexity_v2_minus_zero": _metric_payload(
            v2_log_ppl - zero_log_ppl,
            margin=MARGIN_HARD_LOG_PERPLEXITY,
            draws=draws,
            direction="greater",
            strict_mean=True,
            strict_interval=False,
        ),
        "q4_source_reference_car_jsd_v2_minus_zero": _metric_payload(
            v2.patient_reference_jsd - zero.patient_reference_jsd,
            margin=MARGIN_REFERENCE_JSD,
            draws=draws,
            direction="less",
            strict_mean=False,
            strict_interval=False,
        ),
        "descriptive_prediction_support": {
            "target_ids_equal": True,
            "target_ids_sha256": zero.target_ids_sha256,
            "zero_patient_unique_counts": [int(value) for value in zero_unique],
            "v2_patient_unique_counts": [int(value) for value in v2_unique],
            "zero_patient_top_fractions": [float(value) for value in zero_top],
            "v2_patient_top_fractions": [float(value) for value in v2_top],
            "zero_aggregate_unique_count": int(
                np.count_nonzero(zero.aggregate_prediction_counts)
            ),
            "v2_aggregate_unique_count": int(
                np.count_nonzero(v2.aggregate_prediction_counts)
            ),
        },
    }


def summarize_arm(statistics: QualificationArmStatistics) -> dict[str, object]:
    log_ppl, top, unique = _hard_code_statistics(statistics.prediction_counts)
    return {
        "patient_macro_official_ce": float(
            np.mean(statistics.patient_ce, dtype=np.float64)
        ),
        "patient_macro_accuracy": float(
            np.mean(statistics.patient_accuracy, dtype=np.float64)
        ),
        "patient_macro_source_reference_car_jsd": float(
            np.mean(statistics.patient_reference_jsd, dtype=np.float64)
        ),
        "patient_macro_hard_prediction_log_perplexity": float(
            np.mean(log_ppl, dtype=np.float64)
        ),
        "patient_macro_hard_prediction_effective_perplexity": math.exp(
            float(np.mean(log_ppl, dtype=np.float64))
        ),
        "patient_macro_hard_prediction_top_fraction": float(
            np.mean(top, dtype=np.float64)
        ),
        "patient_unique_count_minimum": int(np.min(unique)),
        "patient_unique_count_maximum": int(np.max(unique)),
        "aggregate_unique_count": int(
            np.count_nonzero(statistics.aggregate_prediction_counts)
        ),
        "target_ids_sha256": statistics.target_ids_sha256,
    }


_ROOT_KEYS = {
    "schema_version",
    "protocol_version",
    "qualification_scope",
    "external_validation",
    "foundation_pretraining_patient_exposure_excluded",
    "source_run_lineage",
    "qualification_cohort",
    "reference_stratification",
    "reference_view_contract",
    "bootstrap",
    "arm_summaries",
    "paired_metrics",
    "all_representation_gates_pass",
    "representation_qualified",
    "eligible_for_locked_downstream_comparison",
    "qualification_signal_split_loaded",
    "qualification_patient_signals_seen",
    "soz_promotion",
    "candidate_promotable",
    "target_values_loaded",
    "diagnostic_directory_labels_used",
    "private_data_loaded",
    "annotation_sidecars_opened",
    "annotation_times_used",
}
_LINEAGE_KEYS = {
    "source_run_receipt_path",
    "source_run_receipt_sha256",
    "selected_adapter_path",
    "selected_adapter_sha256",
    "selected_epoch",
    "selected_epoch_dev_eligible",
    "manifest_path",
    "manifest_sha256",
    "qualification_runner_sha256",
    "qualification_statistics_sha256",
}
_COHORT_KEYS = {
    "patient_ids",
    "patient_ids_sha256",
    "patient_count",
    "windows_per_patient",
    "ordered_window_draw_count",
    "unique_window_count",
    "window_sampler_seed",
    "window_sampler_epoch",
    "ordered_window_uid_sha256",
    "fixed_mask_seed",
    "fixed_mask_sha256",
}
_REFERENCE_VIEW_KEYS = {
    "source",
    "shared_filter_resample_crop",
    "primary",
    "sensitivity",
    "q4_interpretation",
    "car_replay_max_abs_error_volts",
    "car_from_float32_source_reference_max_abs_error_volts",
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
    "q1_ce_zero_minus_v2",
    "q2_accuracy_v2_minus_zero",
    "q3_hard_log_perplexity_v2_minus_zero",
    "q4_source_reference_car_jsd_v2_minus_zero",
    "descriptive_prediction_support",
}
_METRIC_KEYS = {
    "patient_values",
    "patient_macro_mean",
    "margin",
    "ci_lower",
    "ci_upper",
    "passed",
}
_ARM_SUMMARY_KEYS = {
    "patient_macro_official_ce",
    "patient_macro_accuracy",
    "patient_macro_source_reference_car_jsd",
    "patient_macro_hard_prediction_log_perplexity",
    "patient_macro_hard_prediction_effective_perplexity",
    "patient_macro_hard_prediction_top_fraction",
    "patient_unique_count_minimum",
    "patient_unique_count_maximum",
    "aggregate_unique_count",
    "target_ids_sha256",
}


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
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
        raise ValueError(f"DAPT-v2 qualification JSON contains NaN/Inf at {location}")


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _recompute_metric(
    metric: Mapping[str, object],
    *,
    draws: np.ndarray,
    margin: float,
    direction: str,
    strict_mean: bool,
    strict_interval: bool,
    label: str,
) -> bool:
    _require_exact_keys(metric, _METRIC_KEYS, label)
    values = np.asarray(metric["patient_values"], dtype=np.float64)
    if values.shape != (36,) or not np.isfinite(values).all():
        raise ValueError(f"{label} must contain 36 finite patient values")
    if metric["margin"] != margin:
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
        point = expected_mean > margin if strict_mean else expected_mean >= margin
        interval = (
            expected_lower > margin
            if strict_interval
            else expected_lower >= margin
        )
    elif direction == "less":
        point = expected_mean < margin if strict_mean else expected_mean <= margin
        interval = (
            expected_upper < margin
            if strict_interval
            else expected_upper <= margin
        )
    else:  # pragma: no cover - frozen callers below
        raise ValueError("Unknown DAPT-v2 qualification direction")
    expected_pass = bool(point and interval)
    if metric["passed"] is not expected_pass:
        raise ValueError(f"{label} passed flag contradicts patient values/CI")
    return expected_pass


def _validate_reference_stratification(value: Mapping[str, object]) -> None:
    _require_exact_keys(
        value,
        {
            "qualification_eligible_inventory",
            "fixed_window_replay",
        },
        "reference stratification",
    )
    for section_name, expected_draws in (
        ("qualification_eligible_inventory", None),
        ("fixed_window_replay", QUALIFICATION_WINDOW_DRAWS),
    ):
        section = value[section_name]
        if not isinstance(section, Mapping):
            raise TypeError(f"{section_name} must be a mapping")
        _require_exact_keys(
            section,
            {"REF", "LE", "patient_composition"},
            section_name,
        )
        total_draws = 0
        for reference in ("REF", "LE"):
            row = section[reference]
            if not isinstance(row, Mapping):
                raise TypeError(f"{section_name}.{reference} must be a mapping")
            expected_keys = {
                "patient_count",
                "patient_ids_sha256",
                "unique_record_count",
                "record_uids_sha256",
            }
            if expected_draws is not None:
                expected_keys.add("window_draw_count")
            _require_exact_keys(row, expected_keys, f"{section_name}.{reference}")
            for count_key in ("patient_count", "unique_record_count"):
                count = row[count_key]
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 0
                    or (count_key == "patient_count" and count > 36)
                ):
                    raise ValueError(f"Invalid reference stratum count: {count_key}")
            for digest_key in ("patient_ids_sha256", "record_uids_sha256"):
                if not _is_sha256(row[digest_key]):
                    raise ValueError(f"Invalid reference stratum digest: {digest_key}")
            if expected_draws is not None:
                draw_count = row["window_draw_count"]
                if (
                    isinstance(draw_count, bool)
                    or not isinstance(draw_count, int)
                    or draw_count < 0
                ):
                    raise ValueError("Invalid fixed-window reference draw count")
                if row["unique_record_count"] > draw_count:
                    raise ValueError("Fixed-window unique records exceed its draws")
                total_draws += draw_count
        if expected_draws is not None and total_draws != expected_draws:
            raise ValueError("REF/LE fixed-window draws do not sum to 1,152")
        composition = section["patient_composition"]
        if not isinstance(composition, Mapping):
            raise TypeError("Reference patient composition must be a mapping")
        _require_exact_keys(
            composition,
            {"REF_only", "LE_only", "mixed_REF_LE"},
            f"{section_name}.patient_composition",
        )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in composition.values()
        ) or sum(composition.values()) != QUALIFICATION_PATIENTS:
            raise ValueError("Reference patient composition must partition 36 patients")
        if (
            section["REF"]["patient_count"]
            != composition["REF_only"] + composition["mixed_REF_LE"]
            or section["LE"]["patient_count"]
            != composition["LE_only"] + composition["mixed_REF_LE"]
        ):
            raise ValueError("REF/LE patient counts contradict patient composition")


def validate_qualification_artifact(payload: Mapping[str, object]) -> None:
    """Recompute all gates and fail closed on schema or lineage drift."""

    if not isinstance(payload, Mapping):
        raise TypeError("DAPT-v2 qualification artifact must be a mapping")
    _require_exact_keys(payload, _ROOT_KEYS, "DAPT-v2 qualification artifact")
    _require_finite_json(payload)
    if payload["schema_version"] != QUALIFICATION_SCHEMA_VERSION:
        raise ValueError("DAPT-v2 qualification schema changed")
    if payload["protocol_version"] != QUALIFICATION_PROTOCOL_VERSION:
        raise ValueError("DAPT-v2 qualification protocol changed")
    if (
        payload["qualification_scope"] != QUALIFICATION_SCOPE
        or payload["external_validation"] is not False
        or payload["foundation_pretraining_patient_exposure_excluded"] is not False
    ):
        raise ValueError("DAPT-v2 qualification scope was overstated")

    lineage = payload["source_run_lineage"]
    if not isinstance(lineage, Mapping):
        raise TypeError("source_run_lineage must be a mapping")
    _require_exact_keys(lineage, _LINEAGE_KEYS, "source run lineage")
    for digest_field in (
        "source_run_receipt_sha256",
        "selected_adapter_sha256",
        "manifest_sha256",
        "qualification_runner_sha256",
        "qualification_statistics_sha256",
    ):
        if not _is_sha256(lineage[digest_field]):
            raise ValueError(f"Invalid DAPT-v2 lineage digest: {digest_field}")
    for path_field, digest_field in (
        ("source_run_receipt_path", "source_run_receipt_sha256"),
        ("selected_adapter_path", "selected_adapter_sha256"),
        ("manifest_path", "manifest_sha256"),
    ):
        raw_path = lineage[path_field]
        if not isinstance(raw_path, str):
            raise TypeError(f"DAPT-v2 lineage path is not text: {path_field}")
        path = Path(raw_path)
        resolved = path.resolve(strict=True)
        if not path.is_absolute() or path.is_symlink() or str(resolved) != raw_path:
            raise ValueError(f"DAPT-v2 lineage path is not canonical: {path_field}")
        if sha256_file(resolved) != lineage[digest_field]:
            raise ValueError(f"DAPT-v2 lineage file/hash mismatch: {path_field}")
    if (
        isinstance(lineage["selected_epoch"], bool)
        or not isinstance(lineage["selected_epoch"], int)
        or not 0 <= lineage["selected_epoch"] < 10
        or lineage["selected_epoch_dev_eligible"] is not True
    ):
        raise ValueError("Qualification lineage lacks a dev-eligible non-zero epoch")

    cohort = payload["qualification_cohort"]
    if not isinstance(cohort, Mapping):
        raise TypeError("qualification_cohort must be a mapping")
    _require_exact_keys(cohort, _COHORT_KEYS, "qualification cohort")
    patients = cohort["patient_ids"]
    if (
        not isinstance(patients, list)
        or len(patients) != 36
        or patients != sorted(set(patients))
        or cohort["patient_ids_sha256"] != sha256_json(patients)
        or cohort["patient_count"] != 36
        or cohort["windows_per_patient"] != 32
        or cohort["ordered_window_draw_count"] != 1_152
        or not 1 <= cohort["unique_window_count"] <= 1_152
        or cohort["window_sampler_seed"] != 20260829
        or cohort["window_sampler_epoch"] != 0
        or cohort["fixed_mask_seed"] != 20260812
    ):
        raise ValueError("Qualification cohort is not the frozen 36 x 32 replay")
    for digest_field in (
        "patient_ids_sha256",
        "ordered_window_uid_sha256",
        "fixed_mask_sha256",
    ):
        if not _is_sha256(cohort[digest_field]):
            raise ValueError(f"Invalid qualification cohort digest: {digest_field}")

    stratification = payload["reference_stratification"]
    if not isinstance(stratification, Mapping):
        raise TypeError("reference_stratification must be a mapping")
    _validate_reference_stratification(stratification)

    reference = payload["reference_view_contract"]
    if not isinstance(reference, Mapping):
        raise TypeError("reference_view_contract must be a mapping")
    _require_exact_keys(reference, _REFERENCE_VIEW_KEYS, "reference view contract")
    if (
        reference["source"] != "same_direct_physical_source_reference_payload"
        or reference["shared_filter_resample_crop"] is not True
        or reference["primary"] != "C-CAR19"
        or reference["sensitivity"] != "C-SOURCE19_stratified_as_REF_or_LE"
        or reference["q4_interpretation"]
        != "within-arm source-reference-vs-CAR softmax JSD; paired delta v2-minus-zero"
        or reference["car_replay_max_abs_error_volts"] != 0.0
        or reference[
            "car_from_float32_source_reference_max_abs_error_volts"
        ]
        < 0.0
    ):
        raise ValueError("DAPT-v2 paired reference contract changed")

    bootstrap = payload["bootstrap"]
    if not isinstance(bootstrap, Mapping):
        raise TypeError("bootstrap must be a mapping")
    _require_exact_keys(bootstrap, _BOOTSTRAP_KEYS, "qualification bootstrap")
    expected_draw_digest = patient_index_draws_sha256(patient_bootstrap_draws())
    if bootstrap != {
        "unit": "patient",
        "replicates": 10_000,
        "seed": 20260812,
        "ci": [0.025, 0.975],
        "patient_index_draws_encoding": (
            "numpy_dtype_<i8_C_order_raw_bytes_no_header"
        ),
        "patient_index_draws_sha256": expected_draw_digest,
    }:
        raise ValueError("DAPT-v2 patient bootstrap contract changed")

    arm_summaries = payload["arm_summaries"]
    if not isinstance(arm_summaries, Mapping):
        raise TypeError("arm_summaries must be a mapping")
    _require_exact_keys(arm_summaries, {"exact_zero_lora", "selected_dapt_v2"}, "arm summaries")
    for name, summary in arm_summaries.items():
        if not isinstance(summary, Mapping):
            raise TypeError(f"Arm summary must be a mapping: {name}")
        _require_exact_keys(summary, _ARM_SUMMARY_KEYS, f"arm summary {name}")
        if not _is_sha256(summary["target_ids_sha256"]):
            raise ValueError("Arm target digest is invalid")
        for count_key in (
            "patient_unique_count_minimum",
            "patient_unique_count_maximum",
            "aggregate_unique_count",
        ):
            count = summary[count_key]
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or not 1 <= count <= 8192
            ):
                raise ValueError("Arm predicted-code support count is invalid")
        if (
            summary["patient_unique_count_minimum"]
            > summary["patient_unique_count_maximum"]
            or not 0.0 <= summary["patient_macro_hard_prediction_top_fraction"] <= 1.0
        ):
            raise ValueError("Arm hard-prediction summary is invalid")
    if (
        arm_summaries["exact_zero_lora"]["target_ids_sha256"]
        != arm_summaries["selected_dapt_v2"]["target_ids_sha256"]
    ):
        raise ValueError("Qualification arms scored different neural-code targets")

    paired = payload["paired_metrics"]
    if not isinstance(paired, Mapping):
        raise TypeError("paired_metrics must be a mapping")
    _require_exact_keys(paired, _PAIRED_KEYS, "paired metrics")
    draws = patient_bootstrap_draws()
    gate_specs = (
        (
            "q1_ce_zero_minus_v2",
            MARGIN_CE,
            "greater",
            True,
            True,
        ),
        (
            "q2_accuracy_v2_minus_zero",
            MARGIN_ACCURACY,
            "greater",
            False,
            False,
        ),
        (
            "q3_hard_log_perplexity_v2_minus_zero",
            MARGIN_HARD_LOG_PERPLEXITY,
            "greater",
            True,
            False,
        ),
        (
            "q4_source_reference_car_jsd_v2_minus_zero",
            MARGIN_REFERENCE_JSD,
            "less",
            False,
            False,
        ),
    )
    gate_passes: list[bool] = []
    for name, margin, direction, strict_mean, strict_interval in gate_specs:
        metric = paired[name]
        if not isinstance(metric, Mapping):
            raise TypeError(f"Paired metric must be a mapping: {name}")
        gate_passes.append(
            _recompute_metric(
                metric,
                draws=draws,
                margin=margin,
                direction=direction,
                strict_mean=strict_mean,
                strict_interval=strict_interval,
                label=name,
            )
        )
    support = paired["descriptive_prediction_support"]
    if not isinstance(support, Mapping):
        raise TypeError("descriptive_prediction_support must be a mapping")
    _require_exact_keys(
        support,
        {
            "target_ids_equal",
            "target_ids_sha256",
            "zero_patient_unique_counts",
            "v2_patient_unique_counts",
            "zero_patient_top_fractions",
            "v2_patient_top_fractions",
            "zero_aggregate_unique_count",
            "v2_aggregate_unique_count",
        },
        "descriptive prediction support",
    )
    if support["target_ids_equal"] is not True or not _is_sha256(
        support["target_ids_sha256"]
    ):
        raise ValueError("Paired target identity contract failed")
    if support["target_ids_sha256"] != arm_summaries["exact_zero_lora"][
        "target_ids_sha256"
    ]:
        raise ValueError("Paired target digest differs from arm summaries")
    for list_key in ("zero_patient_unique_counts", "v2_patient_unique_counts"):
        values = support[list_key]
        if (
            not isinstance(values, list)
            or len(values) != 36
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 8192
                for value in values
            )
        ):
            raise ValueError("Descriptive support must contain 36 patient values")
    for list_key in ("zero_patient_top_fractions", "v2_patient_top_fractions"):
        values = support[list_key]
        if (
            not isinstance(values, list)
            or len(values) != 36
            or any(not 0.0 <= float(value) <= 1.0 for value in values)
        ):
            raise ValueError("Descriptive top fractions must contain 36 valid values")
    for count_key in (
        "zero_aggregate_unique_count",
        "v2_aggregate_unique_count",
    ):
        count = support[count_key]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= 8192
        ):
            raise ValueError("Descriptive aggregate code support is invalid")

    all_pass = all(gate_passes)
    for field in (
        "all_representation_gates_pass",
        "representation_qualified",
        "eligible_for_locked_downstream_comparison",
    ):
        if payload[field] is not all_pass:
            raise ValueError("DAPT-v2 aggregate qualification flags contradict Q1--Q4")
    if (
        payload["qualification_signal_split_loaded"] is not True
        or payload["qualification_patient_signals_seen"] != 36
    ):
        raise ValueError("Qualification signal access disclosure is inconsistent")
    for field in (
        "soz_promotion",
        "candidate_promotable",
        "target_values_loaded",
        "diagnostic_directory_labels_used",
        "private_data_loaded",
        "annotation_sidecars_opened",
        "annotation_times_used",
    ):
        if payload[field] is not False:
            raise ValueError(f"DAPT-v2 qualification safety flag must be false: {field}")


def build_qualification_artifact(
    *,
    source_run_receipt_path: Path,
    source_run_receipt_sha256: str,
    selected_adapter_path: Path,
    selected_adapter_sha256: str,
    selected_epoch: int,
    manifest_path: Path,
    manifest_sha256: str,
    qualification_runner_sha256: str,
    qualification_statistics_sha256: str,
    patient_ids: Sequence[str],
    ordered_window_identities: Sequence[Mapping[str, object]],
    unique_window_count: int,
    fixed_mask_sha256: str,
    reference_stratification: Mapping[str, object],
    car_replay_max_abs_error_volts: float,
    car_from_float32_source_reference_max_abs_error_volts: float,
    draws: np.ndarray,
    zero_statistics: QualificationArmStatistics,
    v2_statistics: QualificationArmStatistics,
    paired_metrics: Mapping[str, object],
) -> dict[str, object]:
    patients = list(patient_ids)
    if (
        tuple(patients) != zero_statistics.patient_ids
        or tuple(patients) != v2_statistics.patient_ids
        or len(ordered_window_identities) != QUALIFICATION_WINDOW_DRAWS
        or not 1 <= int(unique_window_count) <= QUALIFICATION_WINDOW_DRAWS
    ):
        raise ValueError("Qualification artifact inputs are not the paired 36 x 32 cohort")
    identity_patients = [
        str(identity.get("patient_id")) for identity in ordered_window_identities
    ]
    if (
        set(identity_patients) != set(patients)
        or any(identity_patients.count(patient) != 32 for patient in patients)
    ):
        raise ValueError("Ordered qualification windows are not 32 per patient")
    all_pass = all(
        bool(paired_metrics[name]["passed"])
        for name in (
            "q1_ce_zero_minus_v2",
            "q2_accuracy_v2_minus_zero",
            "q3_hard_log_perplexity_v2_minus_zero",
            "q4_source_reference_car_jsd_v2_minus_zero",
        )
    )
    payload: dict[str, object] = {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "protocol_version": QUALIFICATION_PROTOCOL_VERSION,
        "qualification_scope": QUALIFICATION_SCOPE,
        "external_validation": False,
        "foundation_pretraining_patient_exposure_excluded": False,
        "source_run_lineage": {
            "source_run_receipt_path": str(
                source_run_receipt_path.resolve(strict=True)
            ),
            "source_run_receipt_sha256": source_run_receipt_sha256,
            "selected_adapter_path": str(selected_adapter_path.resolve(strict=True)),
            "selected_adapter_sha256": selected_adapter_sha256,
            "selected_epoch": int(selected_epoch),
            "selected_epoch_dev_eligible": True,
            "manifest_path": str(manifest_path.resolve(strict=True)),
            "manifest_sha256": manifest_sha256,
            "qualification_runner_sha256": qualification_runner_sha256,
            "qualification_statistics_sha256": qualification_statistics_sha256,
        },
        "qualification_cohort": {
            "patient_ids": patients,
            "patient_ids_sha256": sha256_json(patients),
            "patient_count": 36,
            "windows_per_patient": 32,
            "ordered_window_draw_count": 1_152,
            "unique_window_count": int(unique_window_count),
            "window_sampler_seed": 20260829,
            "window_sampler_epoch": 0,
            "ordered_window_uid_sha256": sha256_json(
                list(ordered_window_identities)
            ),
            "fixed_mask_seed": 20260812,
            "fixed_mask_sha256": fixed_mask_sha256,
        },
        "reference_stratification": dict(reference_stratification),
        "reference_view_contract": {
            "source": "same_direct_physical_source_reference_payload",
            "shared_filter_resample_crop": True,
            "primary": "C-CAR19",
            "sensitivity": "C-SOURCE19_stratified_as_REF_or_LE",
            "q4_interpretation": (
                "within-arm source-reference-vs-CAR softmax JSD; "
                "paired delta v2-minus-zero"
            ),
            "car_replay_max_abs_error_volts": float(
                car_replay_max_abs_error_volts
            ),
            "car_from_float32_source_reference_max_abs_error_volts": float(
                car_from_float32_source_reference_max_abs_error_volts
            ),
        },
        "bootstrap": {
            "unit": "patient",
            "replicates": 10_000,
            "seed": 20260812,
            "ci": [0.025, 0.975],
            "patient_index_draws_encoding": (
                "numpy_dtype_<i8_C_order_raw_bytes_no_header"
            ),
            "patient_index_draws_sha256": patient_index_draws_sha256(draws),
        },
        "arm_summaries": {
            "exact_zero_lora": summarize_arm(zero_statistics),
            "selected_dapt_v2": summarize_arm(v2_statistics),
        },
        "paired_metrics": dict(paired_metrics),
        "all_representation_gates_pass": all_pass,
        "representation_qualified": all_pass,
        "eligible_for_locked_downstream_comparison": all_pass,
        "qualification_signal_split_loaded": True,
        "qualification_patient_signals_seen": 36,
        "soz_promotion": False,
        "candidate_promotable": False,
        "target_values_loaded": False,
        "diagnostic_directory_labels_used": False,
        "private_data_loaded": False,
        "annotation_sidecars_opened": False,
        "annotation_times_used": False,
    }
    validate_qualification_artifact(payload)
    return payload


__all__ = [
    "QUALIFICATION_SCOPE",
    "QUALIFICATION_SEED",
    "QualificationArmStatistics",
    "build_paired_metrics",
    "build_qualification_artifact",
    "canonical_json_bytes",
    "paired_percentile_interval",
    "patient_bootstrap_draws",
    "patient_index_draws_sha256",
    "summarize_arm",
    "validate_qualification_artifact",
]
