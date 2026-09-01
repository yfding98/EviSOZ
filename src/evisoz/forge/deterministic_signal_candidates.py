"""Deterministic, fail-closed Stage-0 signal candidate extraction.

The artifact built here is deliberately weaker than a clinical label.  It
contains reproducible feature summaries and uncalibrated proposals derived
only from the validated CAR19 and signed TCP22 caches.  Every row is marked
``signal_derived`` / ``derived_candidate`` and is prohibited from directly
supervising node localization.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from code.soz_pre.constants import TCP_CHANNELS, TCP_PAIRS
from src.evisoz.data.artifact_ref import (
    build_json_artifact_ref,
    canonical_json_bytes,
    canonical_json_sha256,
    validate_artifact_ref,
    verify_artifact_content,
)
from src.evisoz.data.private_stage0_cohort_materializer import (
    PRIVATE_STAGE0_COHORT_SCHEMA_VERSION,
    validate_private_stage0_cohort_artifact,
)
from src.evisoz.data.stage0_dual_montage_cache import (
    MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
    OpenedStage0DualMontageCache,
    open_stage0_dual_montage_cache_from_disk,
)
from src.soz.geometry import STANDARD_19


CANDIDATE_CACHE_SCHEMA_VERSION = "evisoz_deterministic_signal_candidate_cache_v1"
CANDIDATE_MATERIALIZATION_SCHEMA_VERSION = (
    "evisoz_deterministic_signal_candidate_materialization_v1"
)
FEATURE_SPEC_VERSION = "evisoz_deterministic_signal_features_v1"
_HASH_PLACEHOLDER = "0" * 64
_PENDING_ID = "CONTENT-ADDRESS-PENDING"

_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)
_CONCEPTS = (
    "possible_attenuation",
    "possible_LVFA",
    "possible_rhythmic_theta",
    "possible_rhythmic_delta",
    "frequency_evolution_present",
    "near_synchronous_bilateral_change",
    "possible_phase_reversal",
)
_POLICY: dict[str, object] = {
    "authority": "signal_derived",
    "status": "derived_candidate",
    "calibration_state": "uncalibrated",
    "soft_auxiliary_only": True,
    "may_create_clinical_label": False,
    "may_be_treated_as_measured_fact": False,
    "may_supervise_node_localization": False,
    "fold_local_calibration_receipt_present": False,
}
_HOMOLOGOUS_PAIRS: tuple[tuple[str, str], ...] = (
    ("FP1", "FP2"),
    ("F7", "F8"),
    ("F3", "F4"),
    ("T7", "T8"),
    ("C3", "C4"),
    ("P7", "P8"),
    ("P3", "P4"),
    ("O1", "O2"),
)


def build_deterministic_signal_feature_spec() -> dict[str, Any]:
    """Return the frozen v1 windows, estimators, and proposal thresholds."""

    return {
        "spec_version": FEATURE_SPEC_VERSION,
        "sampling_rate_hz": 200,
        "context_interval_seconds": [-12.0, 48.0],
        "onset_interval_seconds": [-2.0, 8.0],
        "baseline_interval_seconds": [-10.0, -2.0],
        "early_interval_seconds": [0.0, 8.0],
        "evolution_patch_seconds": 2.0,
        "evolution_hop_seconds": 1.0,
        "spectral_range_hz": [0.5, 45.0],
        "bands": [
            {"name": name, "low_hz": low, "high_hz": high}
            for name, low, high in _BANDS
        ],
        "ratio_epsilon_policy": "max_denominator_times_1e-6_or_1e-18",
        "stable_change_rule": (
            "two_consecutive_one_second_patches_with_max_abs_log2_ratio_at_least_1"
        ),
        "candidate_rules": {
            "attenuation_max_rms_ratio": 0.60,
            "attenuation_max_amplitude_ratio": 0.70,
            "lvfa_min_beta_gamma_relative_power": 0.45,
            "lvfa_min_line_length_ratio": 1.50,
            "lvfa_min_dominant_frequency_hz": 13.0,
            "rhythmic_min_relative_band_power": 0.35,
            "rhythmic_min_band_power_ratio": 1.50,
            "rhythmic_min_peak_concentration": 0.18,
            "frequency_evolution_min_abs_slope_hz_per_second": 0.25,
            "frequency_evolution_min_range_hz": 2.0,
            "bilateral_max_median_change_lag_seconds": 1.0,
            "bilateral_min_homologous_pair_count": 2.0,
            "phase_reversal_min_oriented_correlation": 0.65,
            "phase_reversal_min_geometric_line_length_ratio": 1.25,
        },
        "quantization_significant_digits": 10,
    }


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _q(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("deterministic signal feature is not finite")
    if abs(result) < 1e-300:
        return 0.0
    return float(f"{result:.10g}")


def _ratio(numerator: float, denominator: float) -> float:
    epsilon = max(abs(float(denominator)) * 1e-6, 1e-18)
    return _q(float(numerator) / max(abs(float(denominator)), epsilon))


def _linear_slope(times: Sequence[float], values: Sequence[float]) -> float:
    x = np.asarray(times, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0.0:
        return 0.0
    return _q(float(np.dot(centered, y - y.mean()) / denominator))


def _spectral_features(signal: np.ndarray, sample_rate: int) -> dict[str, object]:
    x = np.asarray(signal, dtype=np.float64)
    x = x - float(x.mean())
    window = np.hanning(x.size)
    transformed = np.fft.rfft(x * window)
    normalization = max(float(sample_rate) * float(np.sum(window * window)), 1e-18)
    power = (np.abs(transformed) ** 2) / normalization
    frequencies = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)
    selected = (frequencies >= 0.5) & (frequencies <= 45.0)
    selected_power = power[selected]
    selected_frequencies = frequencies[selected]
    total = float(selected_power.sum())
    if total <= 1e-30 or selected_power.size == 0:
        dominant = 0.0
        entropy = 0.0
        rhythmicity = 0.0
    else:
        dominant = float(selected_frequencies[int(np.argmax(selected_power))])
        probabilities = selected_power / total
        entropy = float(
            -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))
            / math.log(probabilities.size)
        )
        peak_mask = np.abs(selected_frequencies - dominant) <= 1.0
        rhythmicity = float(selected_power[peak_mask].sum() / total)
    absolute: dict[str, float] = {}
    relative: dict[str, float] = {}
    for name, low, high in _BANDS:
        include = (frequencies >= low) & (frequencies < high)
        value = float(power[include].sum())
        absolute[name] = _q(value)
        relative[name] = _q(value / max(total, 1e-30))
    return {
        "dominant_frequency_hz": _q(dominant),
        "spectral_entropy": _q(min(max(entropy, 0.0), 1.0)),
        "rhythmicity": _q(min(max(rhythmicity, 0.0), 1.0)),
        "absolute_band_power": absolute,
        "relative_band_power": relative,
    }


def _window_features(signal: np.ndarray, sample_rate: int) -> dict[str, object]:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("signal window must be a finite one-dimensional vector")
    spectral = _spectral_features(x, sample_rate)
    return {
        "rms": _q(float(np.sqrt(np.mean(x * x)))),
        "median_absolute_amplitude": _q(float(np.median(np.abs(x)))),
        "line_length_per_sample": _q(float(np.mean(np.abs(np.diff(x))))),
        **spectral,
    }


def _patch_windows(early: np.ndarray, sample_rate: int) -> tuple[list[np.ndarray], list[float]]:
    patch = 2 * sample_rate
    hop = sample_rate
    windows = [early[start : start + patch] for start in range(0, early.size - patch + 1, hop)]
    centers = [float(start + patch / 2) / sample_rate for start in range(0, early.size - patch + 1, hop)]
    return windows, centers


def _first_stable_change(
    early: np.ndarray,
    baseline: Mapping[str, object],
    sample_rate: int,
) -> float | None:
    patch_size = sample_rate
    scores: list[float] = []
    baseline_total = sum(float(value) for value in baseline["absolute_band_power"].values())
    for start in range(0, early.size - patch_size + 1, patch_size):
        current = _window_features(early[start : start + patch_size], sample_rate)
        current_total = sum(float(value) for value in current["absolute_band_power"].values())
        ratios = (
            _ratio(float(current["rms"]), float(baseline["rms"])),
            _ratio(
                float(current["line_length_per_sample"]),
                float(baseline["line_length_per_sample"]),
            ),
            _ratio(current_total, baseline_total),
        )
        scores.append(max(abs(math.log2(max(value, 1e-12))) for value in ratios))
    for index in range(len(scores) - 1):
        if scores[index] >= 1.0 and scores[index + 1] >= 1.0:
            return _q(float(index))
    return None


def _change_features(
    baseline_signal: np.ndarray,
    early_signal: np.ndarray,
    baseline: Mapping[str, object],
    early: Mapping[str, object],
    sample_rate: int,
) -> dict[str, object]:
    windows, centers = _patch_windows(early_signal, sample_rate)
    patch_features = [_window_features(window, sample_rate) for window in windows]
    dominant = [float(row["dominant_frequency_hz"]) for row in patch_features]
    envelopes = [float(row["rms"]) for row in patch_features]
    band_ratios = {
        name: _ratio(
            float(early["absolute_band_power"][name]),
            float(baseline["absolute_band_power"][name]),
        )
        for name, _, _ in _BANDS
    }
    return {
        "rms_ratio": _ratio(float(early["rms"]), float(baseline["rms"])),
        "median_absolute_amplitude_ratio": _ratio(
            float(early["median_absolute_amplitude"]),
            float(baseline["median_absolute_amplitude"]),
        ),
        "line_length_ratio": _ratio(
            float(early["line_length_per_sample"]),
            float(baseline["line_length_per_sample"]),
        ),
        "band_power_ratios": band_ratios,
        "dominant_frequency_delta_hz": _q(
            float(early["dominant_frequency_hz"])
            - float(baseline["dominant_frequency_hz"])
        ),
        "frequency_slope_hz_per_second": _linear_slope(centers, dominant),
        "frequency_range_hz": _q(max(dominant) - min(dominant)),
        "envelope_slope_per_second": _linear_slope(centers, envelopes),
        "first_stable_change_seconds": _first_stable_change(
            early_signal, baseline, sample_rate
        ),
    }


def _unit_rows(
    data: np.ndarray,
    observed: Sequence[object],
    labels: Sequence[str],
    *,
    view: str,
    unit_type: str,
) -> list[dict[str, object]]:
    if data.shape != (len(labels), 12000):
        raise ValueError(f"{view} tensor geometry drifted")
    if len(observed) != len(labels):
        raise ValueError(f"{view} observed mask geometry drifted")
    rows: list[dict[str, object]] = []
    for index, (label, have) in enumerate(zip(labels, observed)):
        is_observed = have is True
        baseline: dict[str, object] | None = None
        early: dict[str, object] | None = None
        change: dict[str, object] | None = None
        if is_observed:
            baseline_signal = np.asarray(data[index, 400:2000], dtype=np.float64)
            early_signal = np.asarray(data[index, 2400:4000], dtype=np.float64)
            baseline = _window_features(baseline_signal, 200)
            early = _window_features(early_signal, 200)
            change = _change_features(
                baseline_signal, early_signal, baseline, early, 200
            )
        rows.append(
            {
                "view": view,
                "unit_index": index,
                "unit_id": label,
                "unit_type": unit_type,
                "observed": is_observed,
                "baseline": baseline,
                "early": early,
                "change": change,
                "authority": "signal_derived",
                "status": "derived_candidate",
            }
        )
    return rows


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    x = x - float(x.mean())
    y = y - float(y.mean())
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denominator <= 1e-30:
        return 0.0
    return _q(float(np.dot(x, y) / denominator))


def _global_rows(
    car: np.ndarray,
    car_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    index = {label: position for position, label in enumerate(STANDARD_19)}
    correlations: list[float] = []
    support: list[str] = []
    for left, right in _HOMOLOGOUS_PAIRS:
        left_row = car_rows[index[left]]
        right_row = car_rows[index[right]]
        if left_row["observed"] is True and right_row["observed"] is True:
            correlations.append(
                abs(_safe_corr(car[index[left], 2400:4000], car[index[right], 2400:4000]))
            )
            support.extend((left, right))
    synchrony = float(np.mean(correlations)) if correlations else 0.0
    observed_rows = [row for row in car_rows if row["observed"] is True]
    energies = np.asarray(
        [float(row["early"]["rms"]) ** 2 for row in observed_rows],
        dtype=np.float64,
    )
    total = float(energies.sum())
    if energies.size <= 1 or total <= 1e-30:
        spatial_entropy = 0.0
    else:
        probabilities = energies / total
        spatial_entropy = float(
            -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))
            / math.log(probabilities.size)
        )
    return [
        {
            "feature_id": "left_right_synchrony",
            "value": _q(min(max(synchrony, 0.0), 1.0)),
            "support_units": sorted(set(support)),
            "authority": "signal_derived",
            "status": "derived_candidate",
        },
        {
            "feature_id": "spatial_entropy",
            "value": _q(min(max(spatial_entropy, 0.0), 1.0)),
            "support_units": sorted(row["unit_id"] for row in observed_rows),
            "authority": "signal_derived",
            "status": "derived_candidate",
        },
    ]


def _candidate(
    *,
    concept: str,
    support_view: str,
    support_units: Sequence[str],
    heuristic_score: float,
    rule_id: str,
    shared_electrode: str | None = None,
) -> dict[str, object]:
    if concept not in _CONCEPTS:
        raise ValueError("unsupported deterministic candidate concept")
    body: dict[str, object] = {
        "candidate_id": _PENDING_ID,
        "concept": concept,
        "support_view": support_view,
        "support_units": sorted(set(support_units)),
        "support_interval_seconds": [0.0, 8.0],
        "heuristic_score": _q(min(max(float(heuristic_score), 0.0), 1.0)),
        "rule_id": rule_id,
        "shared_electrode": shared_electrode,
        "authority": "signal_derived",
        "status": "derived_candidate",
        "calibration_state": "uncalibrated",
        "permitted_uses": ["soft_auxiliary"],
        "prohibited_uses": [
            "clinical_label",
            "measured_fact",
            "node_localization_supervision",
        ],
    }
    identity_source = deepcopy(body)
    identity_source["candidate_id"] = _PENDING_ID
    body["candidate_id"] = "EVISOZ-CAND-" + canonical_json_sha256(identity_source)[:24]
    return body


def _motif_candidates(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in rows:
        if row["observed"] is not True:
            continue
        early = row["early"]
        change = row["change"]
        view = str(row["view"])
        unit = str(row["unit_id"])
        rms_ratio = float(change["rms_ratio"])
        amplitude_ratio = float(change["median_absolute_amplitude_ratio"])
        line_ratio = float(change["line_length_ratio"])
        relative = early["relative_band_power"]
        band_ratios = change["band_power_ratios"]
        rhythmicity = float(early["rhythmicity"])
        if rms_ratio <= 0.60 and amplitude_ratio <= 0.70:
            score = 0.5 * (1.0 - rms_ratio / 0.60) + 0.5 * (1.0 - amplitude_ratio / 0.70)
            result.append(
                _candidate(
                    concept="possible_attenuation",
                    support_view=view,
                    support_units=[unit],
                    heuristic_score=score,
                    rule_id="attenuation_ratio_rule_v1",
                )
            )
        fast_fraction = float(relative["beta"]) + float(relative["gamma"])
        if (
            fast_fraction >= 0.45
            and line_ratio >= 1.50
            and float(early["dominant_frequency_hz"]) >= 13.0
        ):
            score = min(
                fast_fraction / 0.90,
                line_ratio / 3.0,
                float(early["dominant_frequency_hz"]) / 26.0,
            )
            result.append(
                _candidate(
                    concept="possible_LVFA",
                    support_view=view,
                    support_units=[unit],
                    heuristic_score=score,
                    rule_id="lvfa_fast_line_length_rule_v1",
                )
            )
        for band, concept in (
            ("theta", "possible_rhythmic_theta"),
            ("delta", "possible_rhythmic_delta"),
        ):
            if (
                float(relative[band]) >= 0.35
                and float(band_ratios[band]) >= 1.50
                and rhythmicity >= 0.18
            ):
                score = min(
                    float(relative[band]) / 0.70,
                    float(band_ratios[band]) / 3.0,
                    rhythmicity / 0.36,
                )
                result.append(
                    _candidate(
                        concept=concept,
                        support_view=view,
                        support_units=[unit],
                        heuristic_score=score,
                        rule_id=f"{band}_rhythmic_power_rule_v1",
                    )
                )
        slope = abs(float(change["frequency_slope_hz_per_second"]))
        frequency_range = float(change["frequency_range_hz"])
        if slope >= 0.25 and frequency_range >= 2.0:
            result.append(
                _candidate(
                    concept="frequency_evolution_present",
                    support_view=view,
                    support_units=[unit],
                    heuristic_score=min(slope / 1.0, frequency_range / 8.0),
                    rule_id="dominant_frequency_patch_slope_rule_v1",
                )
            )
    return result


def _bilateral_candidate(
    car_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_id = {str(row["unit_id"]): row for row in car_rows}
    lags: list[float] = []
    support: list[str] = []
    for left, right in _HOMOLOGOUS_PAIRS:
        left_time = by_id[left]["change"]["first_stable_change_seconds"]
        right_time = by_id[right]["change"]["first_stable_change_seconds"]
        if left_time is not None and right_time is not None:
            lag = abs(float(left_time) - float(right_time))
            if lag <= 1.0:
                lags.append(lag)
                support.extend((left, right))
    if len(lags) < 2:
        return []
    score = (1.0 - float(np.median(lags))) * min(len(lags) / 4.0, 1.0)
    return [
        _candidate(
            concept="near_synchronous_bilateral_change",
            support_view="car19_context",
            support_units=support,
            heuristic_score=score,
            rule_id="homologous_stable_change_lag_rule_v1",
        )
    ]


def _phase_reversal_candidates(
    tcp: np.ndarray,
    tcp_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for left_index in range(len(TCP_PAIRS)):
        if tcp_rows[left_index]["observed"] is not True:
            continue
        for right_index in range(left_index + 1, len(TCP_PAIRS)):
            if tcp_rows[right_index]["observed"] is not True:
                continue
            shared = set(TCP_PAIRS[left_index]).intersection(TCP_PAIRS[right_index])
            if len(shared) != 1:
                continue
            electrode = next(iter(shared))
            left_sign = 1.0 if TCP_PAIRS[left_index][0] == electrode else -1.0
            right_sign = 1.0 if TCP_PAIRS[right_index][0] == electrode else -1.0
            correlation = _safe_corr(
                left_sign * tcp[left_index, 2400:4000],
                right_sign * tcp[right_index, 2400:4000],
            )
            line_activation = math.sqrt(
                max(float(tcp_rows[left_index]["change"]["line_length_ratio"]), 0.0)
                * max(float(tcp_rows[right_index]["change"]["line_length_ratio"]), 0.0)
            )
            if correlation >= 0.65 and line_activation >= 1.25:
                score = min(
                    max((correlation - 0.65) / 0.35, 0.0),
                    max((line_activation - 1.25) / 1.25, 0.0),
                )
                result.append(
                    _candidate(
                        concept="possible_phase_reversal",
                        support_view="tcp22_edge_context",
                        support_units=(TCP_CHANNELS[left_index], TCP_CHANNELS[right_index]),
                        heuristic_score=score,
                        rule_id="shared_electrode_oriented_correlation_rule_v1",
                        shared_electrode=electrode,
                    )
                )
    return result


def _cache_hash_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _cache_id_source(value: Mapping[str, object]) -> dict[str, object]:
    result = _cache_hash_source(value)
    result["cache_id"] = _PENDING_ID
    return result


def build_deterministic_signal_candidate_cache(
    opened: OpenedStage0DualMontageCache,
) -> dict[str, Any]:
    """Build one content-addressed candidate cache from a replayed event."""

    if not isinstance(opened, OpenedStage0DualMontageCache):
        raise TypeError("opened must be an OpenedStage0DualMontageCache")
    car_tensor = opened.checkout_v29_reference()
    if car_tensor is None:
        raise ValueError("deterministic v1 candidates require the exact CAR19 reference view")
    tcp_tensor = opened.checkout_tcp22_context()
    car = np.asarray(car_tensor.numpy(), dtype=np.float64)
    tcp = np.asarray(tcp_tensor.numpy(), dtype=np.float64)
    montage = opened.montage_receipt
    car_mask = montage["views"]["v29_reference"]["unit_observed_mask"]
    tcp_mask = montage["views"]["tcp22_context"]["unit_observed_mask"]
    car_rows = _unit_rows(
        car,
        car_mask,
        STANDARD_19,
        view="car19_context",
        unit_type="electrode",
    )
    tcp_rows = _unit_rows(
        tcp,
        tcp_mask,
        TCP_CHANNELS,
        view="tcp22_edge_context",
        unit_type="bipolar_derivation",
    )
    unit_rows = car_rows + tcp_rows
    global_rows = _global_rows(car, car_rows)
    candidates = (
        _motif_candidates(unit_rows)
        + _bilateral_candidate(car_rows)
        + _phase_reversal_candidates(tcp, tcp_rows)
    )
    candidates.sort(
        key=lambda row: (
            str(row["concept"]),
            str(row["support_view"]),
            tuple(row["support_units"]),
            str(row["candidate_id"]),
        )
    )
    concept_counts = Counter(str(row["concept"]) for row in candidates)
    spec = build_deterministic_signal_feature_spec()
    body: dict[str, Any] = {
        "schema_version": CANDIDATE_CACHE_SCHEMA_VERSION,
        "cache_id": _PENDING_ID,
        "status": "complete_uncalibrated_soft_auxiliary_only",
        "event_identity_ref": _plain(opened.materialization_receipt["event_identity_ref"]),
        "dual_montage_cache_ref": build_json_artifact_ref(
            _plain(opened.materialization_receipt),
            artifact_kind="dual_montage_cache_materialization_receipt",
            payload_schema_version=MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
        ),
        "montage_receipt_ref": _plain(opened.materialization_receipt["montage_receipt_ref"]),
        "feature_spec": spec,
        "feature_spec_ref": build_json_artifact_ref(
            spec,
            artifact_kind="deterministic_signal_feature_spec",
            payload_schema_version=FEATURE_SPEC_VERSION,
        ),
        "unit_feature_rows": unit_rows,
        "global_feature_rows": global_rows,
        "candidate_rows": candidates,
        "authority_policy": deepcopy(_POLICY),
        "counts": {
            "unit_feature_row_count": len(unit_rows),
            "observed_unit_count": sum(row["observed"] is True for row in unit_rows),
            "global_feature_row_count": len(global_rows),
            "candidate_count": len(candidates),
            "candidate_concept_counts": dict(sorted(concept_counts.items())),
        },
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["cache_id"] = "EVISOZ-SIGCAND-" + canonical_json_sha256(
        _cache_id_source(body)
    )[:24]
    body["receipt_sha256"] = canonical_json_sha256(_cache_hash_source(body))
    return validate_deterministic_signal_candidate_cache(
        body,
        trusted_dual_montage_cache_receipt=opened.materialization_receipt,
        trusted_montage_receipt=opened.montage_receipt,
    )


def _finite_tree(value: object, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")


def validate_deterministic_signal_candidate_cache(
    value: object,
    *,
    trusted_dual_montage_cache_receipt: Mapping[str, object] | None = None,
    trusted_montage_receipt: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Validate closed semantics, authority, ordering, and content seals."""

    required = {
        "schema_version", "cache_id", "status", "event_identity_ref",
        "dual_montage_cache_ref", "montage_receipt_ref", "feature_spec",
        "feature_spec_ref", "unit_feature_rows", "global_feature_rows",
        "candidate_rows", "authority_policy", "counts", "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("deterministic signal candidate cache fields drifted")
    data = deepcopy(value)
    _finite_tree(data)
    if data["schema_version"] != CANDIDATE_CACHE_SCHEMA_VERSION:
        raise ValueError("deterministic signal candidate cache schema drifted")
    if data["status"] != "complete_uncalibrated_soft_auxiliary_only":
        raise ValueError("deterministic signal candidate cache status drifted")
    event_ref = validate_artifact_ref(data["event_identity_ref"])
    dual_ref = validate_artifact_ref(data["dual_montage_cache_ref"])
    montage_ref = validate_artifact_ref(data["montage_receipt_ref"])
    spec_ref = validate_artifact_ref(data["feature_spec_ref"])
    if event_ref["artifact_kind"] != "event_identity":
        raise ValueError("candidate cache event reference kind drifted")
    if dual_ref["artifact_kind"] != "dual_montage_cache_materialization_receipt":
        raise ValueError("candidate cache dual-montage reference kind drifted")
    if montage_ref["artifact_kind"] != "montage_derivation_receipt":
        raise ValueError("candidate cache montage reference kind drifted")
    spec = build_deterministic_signal_feature_spec()
    if data["feature_spec"] != spec:
        raise ValueError("deterministic signal feature specification drifted")
    verify_artifact_content(spec_ref, spec)
    if spec_ref["artifact_kind"] != "deterministic_signal_feature_spec":
        raise ValueError("deterministic signal feature reference kind drifted")
    if trusted_dual_montage_cache_receipt is not None:
        trusted_dual = _plain(trusted_dual_montage_cache_receipt)
        verify_artifact_content(dual_ref, trusted_dual)
        expected_event = trusted_dual["event_identity_ref"]
        expected_montage = trusted_dual["montage_receipt_ref"]
        if event_ref != expected_event or montage_ref != expected_montage:
            raise ValueError("candidate cache source bindings drifted")
    if trusted_montage_receipt is not None:
        verify_artifact_content(montage_ref, _plain(trusted_montage_receipt))

    rows = data["unit_feature_rows"]
    if not isinstance(rows, list) or len(rows) != len(STANDARD_19) + len(TCP_CHANNELS):
        raise ValueError("deterministic signal unit feature denominator drifted")
    expected_units = [
        *(('car19_context', index, label, 'electrode') for index, label in enumerate(STANDARD_19)),
        *(('tcp22_edge_context', index, label, 'bipolar_derivation') for index, label in enumerate(TCP_CHANNELS)),
    ]
    observed_count = 0
    for row, expected in zip(rows, expected_units):
        if type(row) is not dict or set(row) != {
            "view", "unit_index", "unit_id", "unit_type", "observed",
            "baseline", "early", "change", "authority", "status",
        }:
            raise ValueError("deterministic signal unit feature row fields drifted")
        if tuple(row[key] for key in ("view", "unit_index", "unit_id", "unit_type")) != expected:
            raise ValueError("deterministic signal unit feature order drifted")
        if row["authority"] != "signal_derived" or row["status"] != "derived_candidate":
            raise ValueError("deterministic signal unit feature authority drifted")
        if type(row["observed"]) is not bool:
            raise TypeError("deterministic signal observed flag must be boolean")
        feature_values = (row["baseline"], row["early"], row["change"])
        if row["observed"] is True:
            observed_count += 1
            if any(item is None for item in feature_values):
                raise ValueError("observed signal unit lacks deterministic features")
        elif any(item is not None for item in feature_values):
            raise ValueError("unobserved signal unit carries deterministic features")

    global_rows = data["global_feature_rows"]
    if not isinstance(global_rows, list) or [row.get("feature_id") for row in global_rows] != [
        "left_right_synchrony", "spatial_entropy"
    ]:
        raise ValueError("deterministic global signal features drifted")
    for row in global_rows:
        if type(row) is not dict or set(row) != {
            "feature_id", "value", "support_units", "authority", "status"
        }:
            raise ValueError("deterministic global feature row fields drifted")
        if (
            row["authority"] != "signal_derived"
            or row["status"] != "derived_candidate"
            or not isinstance(row["value"], (int, float))
            or isinstance(row["value"], bool)
            or not 0.0 <= float(row["value"]) <= 1.0
        ):
            raise ValueError("deterministic global feature semantics drifted")

    candidates = data["candidate_rows"]
    if not isinstance(candidates, list):
        raise TypeError("deterministic candidate rows must be an array")
    expected_order = sorted(
        candidates,
        key=lambda row: (
            str(row["concept"]), str(row["support_view"]),
            tuple(row["support_units"]), str(row["candidate_id"]),
        ),
    )
    if candidates != expected_order:
        raise ValueError("deterministic candidate rows are not canonically sorted")
    valid_units = {
        "car19_context": set(STANDARD_19),
        "tcp22_edge_context": set(TCP_CHANNELS),
    }
    candidate_ids: set[str] = set()
    concept_counts: Counter[str] = Counter()
    for row in candidates:
        if type(row) is not dict or set(row) != {
            "candidate_id", "concept", "support_view", "support_units",
            "support_interval_seconds", "heuristic_score", "rule_id",
            "shared_electrode", "authority", "status", "calibration_state",
            "permitted_uses", "prohibited_uses",
        }:
            raise ValueError("deterministic candidate row fields drifted")
        identity_source = deepcopy(row)
        identity_source["candidate_id"] = _PENDING_ID
        expected_id = "EVISOZ-CAND-" + canonical_json_sha256(identity_source)[:24]
        if row["candidate_id"] != expected_id or expected_id in candidate_ids:
            raise ValueError("deterministic candidate identity drifted or duplicated")
        candidate_ids.add(expected_id)
        view = row["support_view"]
        units = row["support_units"]
        if (
            row["concept"] not in _CONCEPTS
            or view not in valid_units
            or not isinstance(units, list)
            or not units
            or units != sorted(set(units))
            or not set(units).issubset(valid_units[view])
        ):
            raise ValueError("deterministic candidate support drifted")
        if row["concept"] == "possible_phase_reversal":
            if view != "tcp22_edge_context" or not isinstance(row["shared_electrode"], str):
                raise ValueError("phase-reversal candidate support drifted")
        elif row["shared_electrode"] is not None:
            raise ValueError("non-phase candidate carries a shared electrode")
        if (
            row["authority"] != "signal_derived"
            or row["status"] != "derived_candidate"
            or row["calibration_state"] != "uncalibrated"
            or row["permitted_uses"] != ["soft_auxiliary"]
            or row["prohibited_uses"] != [
                "clinical_label", "measured_fact", "node_localization_supervision"
            ]
            or isinstance(row["heuristic_score"], bool)
            or not isinstance(row["heuristic_score"], (int, float))
            or not 0.0 <= float(row["heuristic_score"]) <= 1.0
        ):
            raise ValueError("deterministic candidate authority or score drifted")
        concept_counts[str(row["concept"])] += 1

    if data["authority_policy"] != _POLICY:
        raise ValueError("deterministic signal candidate authority policy drifted")
    expected_counts = {
        "unit_feature_row_count": len(rows),
        "observed_unit_count": observed_count,
        "global_feature_row_count": len(global_rows),
        "candidate_count": len(candidates),
        "candidate_concept_counts": dict(sorted(concept_counts.items())),
    }
    if data["counts"] != expected_counts:
        raise ValueError("deterministic signal candidate counts drifted")
    expected_id = "EVISOZ-SIGCAND-" + canonical_json_sha256(_cache_id_source(data))[:24]
    if data["cache_id"] != expected_id:
        raise ValueError("deterministic signal candidate cache ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_cache_hash_source(data)):
        raise ValueError("deterministic signal candidate cache hash drifted")
    return data


def _materialization_hash_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _materialization_id_source(value: Mapping[str, object]) -> dict[str, object]:
    result = _materialization_hash_source(value)
    result["materialization_id"] = _PENDING_ID
    return result


def _json_file(path: Path) -> tuple[bytes, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("deterministic signal candidate JSON must be a regular file")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if type(value) is not dict or raw != canonical_json_bytes(value):
        raise ValueError("deterministic signal candidate JSON is not canonical")
    return raw, value


def _safe_relative_file(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise TypeError("deterministic signal candidate relative path must be a string")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise ValueError("deterministic signal candidate relative path is unsafe")
    path = root.joinpath(*rel.parts)
    resolved = path.resolve(strict=True)
    resolved.relative_to(root.resolve(strict=True))
    if path.is_symlink() or not path.is_file():
        raise ValueError("deterministic signal candidate payload must be a regular file")
    return resolved


def _read_validated_cohort(root: Path) -> dict[str, Any]:
    validate_private_stage0_cohort_artifact(root)
    _, manifest = _json_file(root.resolve(strict=True) / "manifest.json")
    if manifest["schema_version"] != PRIVATE_STAGE0_COHORT_SCHEMA_VERSION:
        raise ValueError("deterministic signal source cohort schema drifted")
    return manifest


def materialize_deterministic_signal_candidates(
    *,
    real_cohort_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Atomically materialize and replay candidates for every real event."""

    cohort_root = Path(real_cohort_root).resolve(strict=True)
    cohort = _read_validated_cohort(cohort_root)
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    committed = False
    try:
        event_rows: list[dict[str, object]] = []
        role_counts: Counter[str] = Counter()
        concept_counts: Counter[str] = Counter()
        unit_count = 0
        global_count = 0
        for source_row in cohort["events"]:
            event_id = str(source_row["event_id"])
            cache_root = cohort_root / str(source_row["relative_cache_path"])
            opened = open_stage0_dual_montage_cache_from_disk(cache_root)
            cache = build_deterministic_signal_candidate_cache(opened)
            relative = f"events/{event_id}/candidate_cache.json"
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=False)
            target.write_bytes(canonical_json_bytes(cache))
            cache_ref = build_json_artifact_ref(
                cache,
                artifact_kind="deterministic_signal_candidate_cache",
                payload_schema_version=CANDIDATE_CACHE_SCHEMA_VERSION,
            )
            source_ref = build_json_artifact_ref(
                _plain(opened.materialization_receipt),
                artifact_kind="dual_montage_cache_materialization_receipt",
                payload_schema_version=MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
            )
            row = {
                "event_id": event_id,
                "linkage_group_id": source_row["linkage_group_id"],
                "evisoz_role": source_row["evisoz_role"],
                "outer_holdout_fold": source_row["outer_holdout_fold"],
                "source_dual_montage_cache_ref": source_ref,
                "candidate_cache_ref": cache_ref,
                "relative_candidate_cache_path": relative,
                "candidate_count": cache["counts"]["candidate_count"],
                "candidate_concept_counts": cache["counts"]["candidate_concept_counts"],
            }
            event_rows.append(row)
            role_counts[str(row["evisoz_role"])] += 1
            concept_counts.update(cache["counts"]["candidate_concept_counts"])
            unit_count += int(cache["counts"]["unit_feature_row_count"])
            global_count += int(cache["counts"]["global_feature_row_count"])
        event_rows.sort(key=lambda row: str(row["event_id"]))
        spec = build_deterministic_signal_feature_spec()
        manifest: dict[str, Any] = {
            "schema_version": CANDIDATE_MATERIALIZATION_SCHEMA_VERSION,
            "materialization_id": _PENDING_ID,
            "status": "complete_real_signal_deterministic_candidates_uncalibrated",
            "source_cohort_ref": build_json_artifact_ref(
                cohort,
                artifact_kind="private_real_stage0_cohort_manifest",
                payload_schema_version=PRIVATE_STAGE0_COHORT_SCHEMA_VERSION,
            ),
            "feature_spec_ref": build_json_artifact_ref(
                spec,
                artifact_kind="deterministic_signal_feature_spec",
                payload_schema_version=FEATURE_SPEC_VERSION,
            ),
            "events": event_rows,
            "counts": {
                "event_count": len(event_rows),
                "role_event_counts": dict(sorted(role_counts.items())),
                "unit_feature_row_count": unit_count,
                "global_feature_row_count": global_count,
                "candidate_count": sum(concept_counts.values()),
                "candidate_concept_counts": dict(sorted(concept_counts.items())),
                "fold_local_calibration_receipt_count": 0,
                "node_localization_supervision_candidate_count": 0,
            },
            "authority_policy": deepcopy(_POLICY),
            "receipt_sha256": _HASH_PLACEHOLDER,
        }
        manifest["materialization_id"] = "EVISOZ-SIGCANDS-" + canonical_json_sha256(
            _materialization_id_source(manifest)
        )[:24]
        manifest["receipt_sha256"] = canonical_json_sha256(
            _materialization_hash_source(manifest)
        )
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        validate_deterministic_signal_candidate_materialization(
            manifest,
            output_root=staging,
            real_cohort_root=cohort_root,
            replay_features=True,
        )
        os.rename(staging, destination)
        committed = True
        return manifest
    finally:
        if not committed:
            shutil.rmtree(staging, ignore_errors=True)


def validate_deterministic_signal_candidate_materialization(
    value: object,
    *,
    output_root: str | Path | None = None,
    real_cohort_root: str | Path | None = None,
    replay_features: bool = False,
) -> dict[str, Any]:
    """Validate a cohort receipt and optionally replay all source features."""

    required = {
        "schema_version", "materialization_id", "status", "source_cohort_ref",
        "feature_spec_ref", "events", "counts", "authority_policy",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("deterministic signal materialization fields drifted")
    data = deepcopy(value)
    _finite_tree(data)
    if data["schema_version"] != CANDIDATE_MATERIALIZATION_SCHEMA_VERSION:
        raise ValueError("deterministic signal materialization schema drifted")
    if data["status"] != "complete_real_signal_deterministic_candidates_uncalibrated":
        raise ValueError("deterministic signal materialization status drifted")
    source_ref = validate_artifact_ref(data["source_cohort_ref"])
    spec_ref = validate_artifact_ref(data["feature_spec_ref"])
    if source_ref["artifact_kind"] != "private_real_stage0_cohort_manifest":
        raise ValueError("deterministic signal materialization source kind drifted")
    spec = build_deterministic_signal_feature_spec()
    verify_artifact_content(spec_ref, spec)
    if spec_ref["artifact_kind"] != "deterministic_signal_feature_spec":
        raise ValueError("deterministic signal materialization spec kind drifted")
    if data["authority_policy"] != _POLICY:
        raise ValueError("deterministic signal materialization authority drifted")
    events = data["events"]
    if (
        not isinstance(events, list)
        or not events
        or events != sorted(events, key=lambda row: str(row["event_id"]))
        or len({row["event_id"] for row in events}) != len(events)
    ):
        raise ValueError("deterministic signal materialization event roster drifted")

    cohort: dict[str, Any] | None = None
    cohort_root: Path | None = None
    source_by_event: dict[str, Mapping[str, object]] = {}
    if real_cohort_root is not None:
        cohort_root = Path(real_cohort_root).resolve(strict=True)
        cohort = _read_validated_cohort(cohort_root)
        verify_artifact_content(source_ref, cohort)
        source_by_event = {str(row["event_id"]): row for row in cohort["events"]}
        if set(source_by_event) != {str(row["event_id"]) for row in events}:
            raise ValueError("deterministic signal materialization is not cohort-complete")

    root = Path(output_root).resolve(strict=True) if output_root is not None else None
    expected_files = {"manifest.json"}
    role_counts: Counter[str] = Counter()
    concept_counts: Counter[str] = Counter()
    unit_count = 0
    global_count = 0
    for row in events:
        if type(row) is not dict or set(row) != {
            "event_id", "linkage_group_id", "evisoz_role", "outer_holdout_fold",
            "source_dual_montage_cache_ref", "candidate_cache_ref",
            "relative_candidate_cache_path", "candidate_count",
            "candidate_concept_counts",
        }:
            raise ValueError("deterministic signal materialization event fields drifted")
        source_cache_ref = validate_artifact_ref(row["source_dual_montage_cache_ref"])
        candidate_ref = validate_artifact_ref(row["candidate_cache_ref"])
        if (
            source_cache_ref["artifact_kind"] != "dual_montage_cache_materialization_receipt"
            or candidate_ref["artifact_kind"] != "deterministic_signal_candidate_cache"
        ):
            raise ValueError("deterministic signal materialization event reference kind drifted")
        cache: dict[str, Any] | None = None
        if root is not None:
            path = _safe_relative_file(root, row["relative_candidate_cache_path"])
            expected_files.add(str(PurePosixPath(row["relative_candidate_cache_path"])))
            _, cache = _json_file(path)
            verify_artifact_content(candidate_ref, cache)
        if cohort_root is not None:
            source_row = source_by_event[str(row["event_id"])]
            if (
                row["linkage_group_id"] != source_row["linkage_group_id"]
                or row["evisoz_role"] != source_row["evisoz_role"]
                or row["outer_holdout_fold"] != source_row["outer_holdout_fold"]
            ):
                raise ValueError("deterministic signal split binding drifted")
            opened = open_stage0_dual_montage_cache_from_disk(
                cohort_root / str(source_row["relative_cache_path"])
            )
            verify_artifact_content(source_cache_ref, _plain(opened.materialization_receipt))
            if cache is not None:
                cache = validate_deterministic_signal_candidate_cache(
                    cache,
                    trusted_dual_montage_cache_receipt=opened.materialization_receipt,
                    trusted_montage_receipt=opened.montage_receipt,
                )
                if replay_features:
                    rebuilt = build_deterministic_signal_candidate_cache(opened)
                    if rebuilt != cache:
                        raise ValueError("deterministic signal features do not replay")
        if cache is None:
            if not isinstance(row["candidate_count"], int):
                raise ValueError("deterministic signal event count is invalid")
            event_concepts = row["candidate_concept_counts"]
            unit_rows = 41
            global_rows = 2
        else:
            if (
                row["candidate_count"] != cache["counts"]["candidate_count"]
                or row["candidate_concept_counts"]
                != cache["counts"]["candidate_concept_counts"]
            ):
                raise ValueError("deterministic signal event summary drifted")
            event_concepts = cache["counts"]["candidate_concept_counts"]
            unit_rows = int(cache["counts"]["unit_feature_row_count"])
            global_rows = int(cache["counts"]["global_feature_row_count"])
        role_counts[str(row["evisoz_role"])] += 1
        concept_counts.update(event_concepts)
        unit_count += unit_rows
        global_count += global_rows
    if root is not None:
        actual_files: set[str] = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("deterministic signal materialization contains a symlink")
            if path.is_file():
                actual_files.add(path.relative_to(root).as_posix())
        if actual_files != expected_files:
            raise ValueError("deterministic signal materialization file inventory drifted")
    expected_counts = {
        "event_count": len(events),
        "role_event_counts": dict(sorted(role_counts.items())),
        "unit_feature_row_count": unit_count,
        "global_feature_row_count": global_count,
        "candidate_count": sum(concept_counts.values()),
        "candidate_concept_counts": dict(sorted(concept_counts.items())),
        "fold_local_calibration_receipt_count": 0,
        "node_localization_supervision_candidate_count": 0,
    }
    if data["counts"] != expected_counts:
        raise ValueError("deterministic signal materialization counts drifted")
    expected_id = "EVISOZ-SIGCANDS-" + canonical_json_sha256(
        _materialization_id_source(data)
    )[:24]
    if data["materialization_id"] != expected_id:
        raise ValueError("deterministic signal materialization ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(
        _materialization_hash_source(data)
    ):
        raise ValueError("deterministic signal materialization hash drifted")
    return data


__all__ = [
    "CANDIDATE_CACHE_SCHEMA_VERSION",
    "CANDIDATE_MATERIALIZATION_SCHEMA_VERSION",
    "FEATURE_SPEC_VERSION",
    "build_deterministic_signal_feature_spec",
    "build_deterministic_signal_candidate_cache",
    "validate_deterministic_signal_candidate_cache",
    "materialize_deterministic_signal_candidates",
    "validate_deterministic_signal_candidate_materialization",
]
