#!/usr/bin/env python3
"""Replay ST16 short-record context choices on two fixed real train EDFs.

This is an engineering sensitivity audit, not a detector benchmark, a model
selection run, transform-parity evidence, or a clinical claim.  It executes
only the already-trained exploratory epoch-0 checkpoint on CPU.  The two
source-train identities and the sensitivity thresholds are frozen below.

The audit compares the current formal short-record ordering

    bipolar -> right context -> resample -> filter -> whole-tile MAD

against (a) an audit-only ``wrap`` context and (b) an audit-only stage-order
counterfactual

    observed bipolar -> resample -> filter -> observed-only MAD
    -> reflect normalized carrier.

Only original observed support contributes to comparisons, target, loss,
metric, or the Finding-candidate eligibility proof.  EDF+ annotations and all
non-EEG clinical inputs remain unopened by the canonical EDF loader.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Final, Mapping, Sequence

import numpy as np
from scipy.signal import resample_poly, sosfiltfilt
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.canonical_edf_materialization import (  # noqa: E402
    load_canonical_edf_record,
)
from src.clinical_eeg_long_recording.detector_signal_lineage_authority_v1 import (  # noqa: E402
    authorize_detector_signal_lineage_from_canonical_record,
    require_validated_detector_signal_lineage_authority,
    verify_provider_referential_payload,
)
from src.clinical_eeg_long_recording.eventnet_common17_streaming_v1 import (  # noqa: E402
    load_common17_manifest,
)
from src.clinical_eeg_long_recording import (  # noqa: E402
    seizuretransformer_cleanroom_registry_v1 as st,
)
from src.clinical_eeg_long_recording.st16_common17_exploratory_runner_v1 import (  # noqa: E402
    _load_exploratory_checkpoint,
    _model_predictor,
    _safe_edf,
    event_sample_spans,
)


SCHEMA_VERSION: Final[str] = "st16_short_context_real_edf_sensitivity_audit_v1"
PENDING: Final[str] = "CONTENT-ADDRESS-PENDING"
DEFAULT_MANIFEST: Final[Path] = (
    ROOT / "outputs/eventnet_common17_streaming_v1_20260824/manifest.json"
)
DEFAULT_CHECKPOINT: Final[Path] = (
    ROOT
    / "outputs/st16_common17_exploratory_source_train_epoch1_v1_20260825"
    / "epoch_0000.pt"
)
DEFAULT_OUTPUT: Final[Path] = (
    ROOT
    / "outputs/st16_short_context_real_edf_sensitivity_audit_v1_20260825"
    / "receipt.json"
)

POSITIVE_ID: Final[str] = (
    "TUSZANALYSIS-8f653216b8ab26677d6a1c948d12e9c00b4ceca4144ede7cab8bb02eceff2e72"
)
NEGATIVE_ID: Final[str] = (
    "TUSZANALYSIS-d1143c90969dca21de227849accb9f9a2fa585da4fec649a8521b09d69c65f5c"
)
FROZEN_IDENTITIES: Final[tuple[str, str]] = (POSITIVE_ID, NEGATIVE_ID)

# Frozen before the real replay.  These are engineering-sensitivity limits,
# not performance operating points and not clinically qualified thresholds.
FROZEN_PROBABILITY_SENSITIVITY_LIMITS: Final[dict[str, float]] = {
    "mean_absolute_difference_maximum": 0.02,
    "p95_absolute_difference_maximum": 0.05,
    "maximum_absolute_difference_maximum": 0.20,
    "threshold_0_5_discordance_rate_maximum": 0.02,
}
AUDIT_THRESHOLD: Final[float] = 0.5
PADLEN_SAMPLES: Final[int] = 768


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = PENDING
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_receipt(value: object, *, semantic: str) -> dict[str, Any]:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("audit payload must be numeric")
    canonical = np.ascontiguousarray(array)
    if np.issubdtype(canonical.dtype, np.floating) and not np.isfinite(
        canonical
    ).all():
        raise ValueError("audit payload contains nonfinite values")
    return {
        "semantic": semantic,
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
        "payload_sha256": hashlib.sha256(
            canonical.tobytes(order="C")
        ).hexdigest(),
    }


def _write_json_atomic(
    path: Path, value: Mapping[str, Any], *, replace: bool
) -> None:
    target = path.resolve(strict=False)
    if not replace and (target.exists() or target.is_symlink()):
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _select_fixed_train_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_identity: dict[str, dict[str, Any]] = {}
    for raw in manifest["records"]:
        identity = raw.get("analysis_identity_id")
        if identity in FROZEN_IDENTITIES:
            if identity in by_identity:
                raise ValueError("fixed short-record identity is duplicated")
            by_identity[str(identity)] = dict(raw)
    if set(by_identity) != set(FROZEN_IDENTITIES):
        raise ValueError("fixed short-record identity is absent from manifest")
    rows = [by_identity[identity] for identity in FROZEN_IDENTITIES]
    for row in rows:
        if row.get("model_split") != "source_train":
            raise PermissionError("fixed short audit may open source_train only")
        count = row.get("target_sample_count_256hz")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not PADLEN_SAMPLES < count < st.TILE_SAMPLES
        ):
            raise ValueError("fixed record is not an admitted >768-sample short EDF")
    if len(rows[0].get("seizure_events", [])) != 1:
        raise ValueError("frozen positive identity lost its one event")
    if rows[1].get("seizure_events") != []:
        raise ValueError("frozen negative identity unexpectedly became positive")
    return rows


def _observed_first_then_normalized_reflect(
    referential_volts: np.ndarray,
    *,
    signal_lineage_authority: object,
    registry: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Audit-only stage-order counterfactual using the frozen ST16 math."""

    validated_registry = st.validate_registry(dict(registry))
    canonical, electrode_order, rate_pair = verify_provider_referential_payload(
        signal_lineage_authority, referential_volts
    )
    order = tuple(electrode_order)
    index = {electrode: position for position, electrode in enumerate(order)}
    pairs = tuple(tuple(unit.split("-", 1)) for unit in st.ST16_TYPED_UNITS)
    if any(left not in index or right not in index for left, right in pairs):
        raise ValueError("stage-order counterfactual lacks an ST16 electrode")
    bipolar = np.ascontiguousarray(
        np.stack(
            [canonical[index[left]] - canonical[index[right]] for left, right in pairs],
            axis=0,
        ),
        dtype="<f8",
    )
    rate = Fraction(int(rate_pair[0]), int(rate_pair[1]))
    ratio = Fraction(st.TARGET_FS_HZ, 1) / rate
    up, down = ratio.numerator, ratio.denominator
    observed_target_count = (bipolar.shape[1] * up) // down
    if not PADLEN_SAMPLES < observed_target_count < st.TILE_SAMPLES:
        raise ValueError("stage-order audit requires >768 observed target samples")
    if up == down == 1:
        taps = None
        resampled = bipolar.copy()
    else:
        # Private names are intentional here: the counterfactual must bind the
        # exact frozen registry taps/SOS rather than silently redesign them.
        taps = st._polyphase_taps(up, down)
        resampled = resample_poly(
            bipolar,
            up,
            down,
            axis=1,
            window=taps,
            padtype="line",
        )
    if resampled.shape[1] < observed_target_count:
        raise RuntimeError("observed-only resampler returned insufficient support")
    resampled = np.ascontiguousarray(
        resampled[:, :observed_target_count], dtype="<f8"
    )
    filtered = sosfiltfilt(
        st._BANDPASS_SOS.copy(),
        resampled,
        axis=1,
        padtype="odd",
        padlen=PADLEN_SAMPLES,
    )
    filtered = np.ascontiguousarray(filtered, dtype="<f8")
    center = np.median(filtered, axis=1).astype("<f8", copy=False)
    mad = np.median(np.abs(filtered - center[:, None]), axis=1).astype(
        "<f8", copy=False
    )
    scale = np.ascontiguousarray(1.4826 * mad, dtype="<f8")
    if not np.isfinite(scale).all() or np.any(scale <= 1e-12):
        raise ValueError("observed-only robust scale is numerically degenerate")
    normalized_observed = (filtered - center[:, None]) / scale[:, None]
    np.clip(normalized_observed, -20.0, 20.0, out=normalized_observed)
    extended = st._right_context_extend(
        normalized_observed,
        target_sample_count=st.TILE_SAMPLES,
        mode="reflect",
    )
    output = np.ascontiguousarray(extended, dtype="<f4")
    if output.shape != (len(st.ST16_TYPED_UNITS), st.TILE_SAMPLES):
        raise RuntimeError("stage-order counterfactual emitted the wrong shape")
    output.setflags(write=False)
    receipt = _content_address(
        {
            "schema_version": "st16_observed_first_normalized_reflect_audit_transform_v1",
            "claim_status": "audit_only_stage_order_counterfactual",
            "registry_id": validated_registry["registry_id"],
            "registry_sha256": validated_registry["registry_sha256"],
            "registry_implementation_code_sha256": validated_registry[
                "implementation"
            ]["code_sha256"],
            "variant_id": st.ST16_VARIANT_ID,
            "stage_order": [
                "derive_observed_ST16_bipolar",
                "resample_observed_support_only",
                "SOS_filtfilt_observed_support_only",
                "median_MAD_normalize_observed_support_only",
                "reflect_normalized_carrier_to_60_seconds",
            ],
            "formal_transform_stage_order_used": False,
            "sampling_rate_fraction_hz": [rate.numerator, rate.denominator],
            "resample_ratio": [up, down],
            "observed_target_sample_count": observed_target_count,
            "model_context_sample_count": st.TILE_SAMPLES,
            "valid_support_sample_range": [0, observed_target_count],
            "right_context_sample_range": [observed_target_count, st.TILE_SAMPLES],
            "filter_padlen_samples": PADLEN_SAMPLES,
            "filter_sos_sha256": hashlib.sha256(
                np.ascontiguousarray(st._BANDPASS_SOS).tobytes(order="C")
            ).hexdigest(),
            "polyphase_taps_payload": (
                None
                if taps is None
                else _array_receipt(taps, semantic="frozen_polyphase_taps")
            ),
            "observed_bipolar_payload": _array_receipt(
                bipolar, semantic="observed_ST16_bipolar_volts"
            ),
            "observed_resampled_payload": _array_receipt(
                resampled, semantic="observed_resampled_ST16_bipolar_volts"
            ),
            "observed_filtered_payload": _array_receipt(
                filtered, semantic="observed_filtered_ST16_bipolar_volts"
            ),
            "observed_center_payload": _array_receipt(
                center, semantic="observed_only_channel_median_volts"
            ),
            "observed_scale_payload": _array_receipt(
                scale, semantic="observed_only_channel_1_4826_MAD_volts"
            ),
            "observed_normalized_payload": _array_receipt(
                output[:, :observed_target_count],
                semantic="observed_normalized_ST16_carrier",
            ),
            "extended_output_payload": _array_receipt(
                output, semantic="observed_first_normalized_reflect_ST16_carrier"
            ),
            "context_is_observed_EEG": False,
            "context_may_receive_target_loss_metric_or_Finding_weight": False,
            "receipt_sha256": PENDING,
        }
    )
    return output, receipt


def _region_bounds(observed_sample_count: int) -> dict[str, tuple[int, int]]:
    if observed_sample_count <= 5 * st.TARGET_FS_HZ:
        raise ValueError("fixed audit records must exceed five seconds")
    return {
        "valid_support_full": (0, observed_sample_count),
        "valid_support_last_1_second": (
            observed_sample_count - st.TARGET_FS_HZ,
            observed_sample_count,
        ),
        "valid_support_last_2_seconds": (
            observed_sample_count - 2 * st.TARGET_FS_HZ,
            observed_sample_count,
        ),
        "valid_support_last_5_seconds": (
            observed_sample_count - 5 * st.TARGET_FS_HZ,
            observed_sample_count,
        ),
        "valid_support_earlier_than_last_5_seconds": (
            0,
            observed_sample_count - 5 * st.TARGET_FS_HZ,
        ),
    }


def _absolute_difference_summary(
    left: np.ndarray,
    right: np.ndarray,
    *,
    observed_sample_count: int,
    include_threshold_discordance: bool,
) -> dict[str, Any]:
    first = np.asarray(left)
    second = np.asarray(right)
    if first.shape != second.shape or first.ndim not in {1, 2}:
        raise ValueError("sensitivity arrays must share [time] or [axis,time]")
    if first.shape[-1] < observed_sample_count:
        raise ValueError("sensitivity arrays do not cover valid support")
    regions: dict[str, Any] = {}
    for name, (start, stop) in _region_bounds(observed_sample_count).items():
        flat = np.abs(first[..., start:stop] - second[..., start:stop]).astype(
            np.float64, copy=False
        ).reshape(-1)
        if not flat.size or not np.isfinite(flat).all():
            raise ValueError("sensitivity region is empty or nonfinite")
        row: dict[str, Any] = {
            "sample_range": [start, stop],
            "time_axis_sample_count": stop - start,
            "compared_scalar_count": int(flat.size),
            "mean_absolute_difference": float(np.mean(flat, dtype=np.float64)),
            "p95_absolute_difference": float(
                np.percentile(flat, 95.0, method="linear")
            ),
            "maximum_absolute_difference": float(np.max(flat)),
        }
        if include_threshold_discordance:
            a = first[..., start:stop] >= AUDIT_THRESHOLD
            b = second[..., start:stop] >= AUDIT_THRESHOLD
            discordant = int(np.count_nonzero(a != b))
            row.update(
                {
                    "threshold": AUDIT_THRESHOLD,
                    "threshold_discordant_scalar_count": discordant,
                    "threshold_discordance_rate": discordant / int(a.size),
                }
            )
        regions[name] = row
    return {
        "comparison_restricted_to_valid_support": True,
        "valid_support_sample_range": [0, observed_sample_count],
        "regions": regions,
    }


def _probability_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    full = summary["regions"]["valid_support_full"]
    checks = {
        "mean_absolute_difference_within_limit": (
            full["mean_absolute_difference"]
            <= FROZEN_PROBABILITY_SENSITIVITY_LIMITS[
                "mean_absolute_difference_maximum"
            ]
        ),
        "p95_absolute_difference_within_limit": (
            full["p95_absolute_difference"]
            <= FROZEN_PROBABILITY_SENSITIVITY_LIMITS[
                "p95_absolute_difference_maximum"
            ]
        ),
        "maximum_absolute_difference_within_limit": (
            full["maximum_absolute_difference"]
            <= FROZEN_PROBABILITY_SENSITIVITY_LIMITS[
                "maximum_absolute_difference_maximum"
            ]
        ),
        "threshold_0_5_discordance_rate_within_limit": (
            full["threshold_discordance_rate"]
            <= FROZEN_PROBABILITY_SENSITIVITY_LIMITS[
                "threshold_0_5_discordance_rate_maximum"
            ]
        ),
    }
    return {
        "gate_basis": "valid_support_full_probability_only",
        "limits": deepcopy(FROZEN_PROBABILITY_SENSITIVITY_LIMITS),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _infer_probability_and_native_logit(
    model: torch.nn.Module,
    predictor: Any,
    carrier: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    captured: list[np.ndarray] = []

    def capture(_module: object, _inputs: object, output: torch.Tensor) -> None:
        captured.append(output.detach().float().cpu().numpy())

    hook = model.conv_d.register_forward_hook(capture)
    try:
        writable_input = np.array(
            carrier[None, ...], dtype=np.float32, order="C", copy=True
        )
        probability = np.asarray(predictor(writable_input))
    finally:
        hook.remove()
    if probability.shape != (1, st.TILE_SAMPLES) or len(captured) != 1:
        raise RuntimeError("ST16 inference returned an unexpected shape")
    native = np.asarray(captured[0])
    if native.shape != (1, 1, st.TILE_SAMPLES):
        raise RuntimeError("ST16 native detection logit hook returned wrong shape")
    probability = np.ascontiguousarray(probability[0], dtype="<f4")
    logit = np.ascontiguousarray(native[0, 0], dtype="<f4")
    replay = 1.0 / (1.0 + np.exp(-logit.astype(np.float64)))
    replay_error = float(
        np.max(np.abs(replay - probability.astype(np.float64)))
    )
    return probability, logit, {
        "probability_payload": _array_receipt(
            probability, semantic="ST16_prethreshold_probability"
        ),
        "native_pre_sigmoid_logit_payload": _array_receipt(
            logit, semantic="ST16_conv_d_native_pre_sigmoid_logit"
        ),
        "sigmoid_native_logit_probability_maximum_replay_error": replay_error,
        "threshold_morphology_hysteresis_or_NMS_applied": False,
    }


def _mask_exclusion_proof(
    row: Mapping[str, Any],
    probability: np.ndarray,
    valid_support_mask: np.ndarray,
    *,
    observed_sample_count: int,
) -> dict[str, Any]:
    spans = event_sample_spans(row)
    target, target_mask, target_receipt = (
        st.build_seizuretransformer_dense_target_pure_primitive(
            spans,
            target_start_sample=0,
            valid_support_sample_count=observed_sample_count,
        )
    )
    if not np.array_equal(target_mask, valid_support_mask):
        raise RuntimeError("transform mask and dense-target mask disagree")
    context = slice(observed_sample_count, st.TILE_SAMPLES)
    if np.any(target[context]) or np.any(target_mask[context]):
        raise RuntimeError("context entered target or observed mask")
    counterfactual = probability.copy()
    counterfactual[context] = 1.0 - counterfactual[context]
    loss_checks: list[dict[str, Any]] = []
    for positive_weight in (1.0, 50.0):
        original_loss = st.masked_patient_macro_bce_pure_primitive(
            probability[None, :],
            target[None, :],
            target_mask[None, :],
            [str(row["patient_id"])],
            positive_weight=positive_weight,
        )
        perturbed_loss = st.masked_patient_macro_bce_pure_primitive(
            counterfactual[None, :],
            target[None, :],
            target_mask[None, :],
            [str(row["patient_id"])],
            positive_weight=positive_weight,
        )
        loss_checks.append(
            {
                "positive_weight": positive_weight,
                "original_masked_loss": original_loss,
                "context_inverted_masked_loss": perturbed_loss,
                "bitwise_equal_float64": (
                    np.float64(original_loss).tobytes()
                    == np.float64(perturbed_loss).tobytes()
                ),
            }
        )
    original_metric = st.masked_dense_binary_metric_counts_pure_primitive(
        probability[None, :],
        target[None, :],
        target_mask[None, :],
        threshold=AUDIT_THRESHOLD,
    )
    perturbed_metric = st.masked_dense_binary_metric_counts_pure_primitive(
        counterfactual[None, :],
        target[None, :],
        target_mask[None, :],
        threshold=AUDIT_THRESHOLD,
    )
    original_finding_indices = np.flatnonzero(
        (probability >= AUDIT_THRESHOLD) & target_mask.astype(bool)
    ).astype("<i8")
    perturbed_finding_indices = np.flatnonzero(
        (counterfactual >= AUDIT_THRESHOLD) & target_mask.astype(bool)
    ).astype("<i8")
    finding_equal = np.array_equal(
        original_finding_indices, perturbed_finding_indices
    )
    loss_equal = all(item["bitwise_equal_float64"] for item in loss_checks)
    metric_equal = original_metric == perturbed_metric
    if not loss_equal or not metric_equal or not finding_equal:
        raise RuntimeError("context exclusion invariance proof failed")
    return _content_address(
        {
            "schema_version": "st16_short_context_target_loss_metric_finding_exclusion_v1",
            "source_train_target_accessed": True,
            "positive_sample_spans": spans,
            "target_receipt_sha256": target_receipt["receipt_sha256"],
            "target_payload": _array_receipt(
                target, semantic="masked_dense_binary_target"
            ),
            "valid_support_mask_payload": _array_receipt(
                target_mask, semantic="target_loss_metric_Finding_valid_support"
            ),
            "observed_support_sample_count": observed_sample_count,
            "zero_weight_context_sample_count": (
                st.TILE_SAMPLES - observed_sample_count
            ),
            "context_target_nonzero_sample_count": int(
                np.count_nonzero(target[context])
            ),
            "context_loss_or_metric_mask_nonzero_sample_count": int(
                np.count_nonzero(target_mask[context])
            ),
            "loss_context_inversion_checks": loss_checks,
            "metric_context_inversion_receipts_bitwise_equal": metric_equal,
            "metric_receipt_sha256": original_metric["receipt_sha256"],
            "Finding_candidate_contract": {
                "scope": "audit_threshold_candidate_support_not_clinical_Finding_generation",
                "eligible_sample_range": [0, observed_sample_count],
                "context_eligible": False,
                "original_candidate_index_payload": _array_receipt(
                    original_finding_indices,
                    semantic="valid_support_threshold_candidate_indices",
                ),
                "context_inversion_candidate_indices_equal": finding_equal,
                "context_above_threshold_but_rejected_sample_count": int(
                    np.count_nonzero(probability[context] >= AUDIT_THRESHOLD)
                ),
            },
            "context_contributed_to_target": False,
            "context_contributed_to_loss": False,
            "context_contributed_to_metric": False,
            "context_contributed_to_Finding_candidate": False,
            "receipt_sha256": PENDING,
        }
    )


def _edge_attempt(
    referential_volts: np.ndarray,
    *,
    signal_lineage_authority: object,
    registry: Mapping[str, Any],
) -> tuple[dict[str, Any], Any | None]:
    try:
        edge = st.apply_short_record_context_sensitivity_transform(
            referential_volts,
            context_mode="edge",
            signal_lineage_authority=signal_lineage_authority,
            registry=registry,
        )
    except Exception as error:  # typed audit result, not silent equivalence
        message = str(error)
        return (
            {
                "attempted": True,
                "status": "typed_technical_failure",
                "failure_type": type(error).__name__,
                "failure_message": message,
                "robust_scale_numerical_degeneracy": (
                    "scale" in message.lower() and "degenerate" in message.lower()
                ),
                "treated_as_zero_difference_or_parity": False,
                "model_forward_executed": False,
                "observed_support_probability_count": 0,
                "observed_support_native_logit_count": 0,
                "formal_reflect_comparison_available": False,
            },
            None,
        )
    return (
        {
            "attempted": True,
            "status": "numeric_success_audit_only",
            "transform_receipt_sha256": edge.receipt["receipt_sha256"],
            "context_ledger_receipt_sha256": edge.receipt[
                "short_record_context"
            ]["receipt_sha256"],
            "output_payload_sha256": edge.receipt["output"]["payload_receipt"][
                "payload_sha256"
            ],
            "treated_as_zero_difference_or_parity": False,
            "model_forward_executed": False,
        },
        edge,
    )


def _comparison_bundle(
    formal_carrier: np.ndarray,
    alternative_carrier: np.ndarray,
    formal_probability: np.ndarray,
    alternative_probability: np.ndarray,
    formal_logit: np.ndarray,
    alternative_logit: np.ndarray,
    *,
    observed_sample_count: int,
) -> dict[str, Any]:
    probability_summary = _absolute_difference_summary(
        formal_probability,
        alternative_probability,
        observed_sample_count=observed_sample_count,
        include_threshold_discordance=True,
    )
    return {
        "carrier_absolute_difference": _absolute_difference_summary(
            formal_carrier,
            alternative_carrier,
            observed_sample_count=observed_sample_count,
            include_threshold_discordance=False,
        ),
        "native_logit_absolute_difference": _absolute_difference_summary(
            formal_logit,
            alternative_logit,
            observed_sample_count=observed_sample_count,
            include_threshold_discordance=False,
        ),
        "probability_absolute_difference_and_threshold_discordance": (
            probability_summary
        ),
        "frozen_probability_sensitivity_gate": _probability_gate(
            probability_summary
        ),
        "transform_parity_claimed": False,
        "performance_claimed": False,
    }


def run(
    *,
    manifest_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    replace: bool,
    cpu_threads: int,
) -> dict[str, Any]:
    if isinstance(cpu_threads, bool) or not 1 <= cpu_threads <= 32:
        raise ValueError("cpu_threads must be in [1,32]")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only audit refuses a CUDA-initialized process")
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Safe for test harnesses that already froze the global interop pool.
        pass
    torch.use_deterministic_algorithms(True)

    manifest_source = manifest_path.resolve(strict=True)
    checkpoint_source = checkpoint_path.resolve(strict=True)
    manifest = load_common17_manifest(manifest_source, require_complete=True)
    rows = _select_fixed_train_rows(manifest)
    tusz_root = Path(manifest["source_bindings"]["tusz_root"]).resolve(
        strict=True
    )
    registry_path = (ROOT / st.CONFIG_RELATIVE_PATH).resolve(strict=True)
    registry = st.load_registry(registry_path)
    model, checkpoint, checkpoint_hash = _load_exploratory_checkpoint(
        checkpoint_source, device=torch.device("cpu")
    )
    if model.training or any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("audit checkpoint is not in CPU eval mode")
    predictor = _model_predictor(model, torch.device("cpu"))

    result_rows: list[dict[str, Any]] = []
    all_gates: list[bool] = []
    edge_failure_count = 0
    edge_numeric_success_count = 0
    edge_model_forward_count = 0
    for row in rows:
        edf_path = _safe_edf(
            tusz_root, row["edf_relative_path"], expected_split="source_train"
        )
        edf_hash = _file_sha256(edf_path)
        canonical = load_canonical_edf_record(edf_path)
        lineage_authority = authorize_detector_signal_lineage_from_canonical_record(
            canonical
        )
        lineage = require_validated_detector_signal_lineage_authority(
            lineage_authority
        )
        if (
            lineage["canonical_physical_signal"]["source_tensor_sha256"]
            != row["canonical_source_tensor_sha256"]
        ):
            raise PermissionError("real EDF tensor failed manifest byte replay")
        referential_volts = np.asarray(
            canonical.observed_signal_volts.detach().cpu().numpy()
        )
        formal = st.apply_full_record_transform(
            referential_volts,
            variant_id=st.ST16_VARIANT_ID,
            signal_lineage_authority=lineage_authority,
            registry=registry,
        )
        wrap = st.apply_short_record_context_sensitivity_transform(
            referential_volts,
            context_mode="wrap",
            signal_lineage_authority=lineage_authority,
            registry=registry,
        )
        stage_order_carrier, stage_order_receipt = (
            _observed_first_then_normalized_reflect(
                referential_volts,
                signal_lineage_authority=lineage_authority,
                registry=registry,
            )
        )
        formal_mask, formal_mask_receipt = (
            st.seizuretransformer_transform_valid_support_mask(formal)
        )
        wrap_mask, wrap_mask_receipt = (
            st.seizuretransformer_transform_valid_support_mask(wrap)
        )
        observed = int(np.count_nonzero(formal_mask))
        if (
            observed != row["target_sample_count_256hz"]
            or not np.array_equal(formal_mask, wrap_mask)
            or stage_order_receipt["observed_target_sample_count"] != observed
        ):
            raise RuntimeError("short-record valid support drifted")

        formal_probability, formal_logit, formal_prediction_receipt = (
            _infer_probability_and_native_logit(model, predictor, formal.signal)
        )
        wrap_probability, wrap_logit, wrap_prediction_receipt = (
            _infer_probability_and_native_logit(model, predictor, wrap.signal)
        )
        stage_probability, stage_logit, stage_prediction_receipt = (
            _infer_probability_and_native_logit(
                model, predictor, stage_order_carrier
            )
        )
        wrap_comparison = _comparison_bundle(
            formal.signal,
            wrap.signal,
            formal_probability,
            wrap_probability,
            formal_logit,
            wrap_logit,
            observed_sample_count=observed,
        )
        stage_comparison = _comparison_bundle(
            formal.signal,
            stage_order_carrier,
            formal_probability,
            stage_probability,
            formal_logit,
            stage_logit,
            observed_sample_count=observed,
        )
        all_gates.extend(
            [
                wrap_comparison["frozen_probability_sensitivity_gate"]["passed"],
                stage_comparison["frozen_probability_sensitivity_gate"]["passed"],
            ]
        )
        edge_attempt, edge_transform = _edge_attempt(
            referential_volts,
            signal_lineage_authority=lineage_authority,
            registry=registry,
        )
        if edge_attempt["status"] == "typed_technical_failure":
            edge_failure_count += 1
        else:
            if edge_transform is None:
                raise RuntimeError("successful edge attempt lacks a transform")
            edge_mask, edge_mask_receipt = (
                st.seizuretransformer_transform_valid_support_mask(edge_transform)
            )
            if not np.array_equal(formal_mask, edge_mask):
                raise RuntimeError("edge valid support disagrees with formal mask")
            edge_probability, edge_logit, edge_prediction_receipt = (
                _infer_probability_and_native_logit(
                    model, predictor, edge_transform.signal
                )
            )
            edge_comparison = _comparison_bundle(
                formal.signal,
                edge_transform.signal,
                formal_probability,
                edge_probability,
                formal_logit,
                edge_logit,
                observed_sample_count=observed,
            )
            all_gates.append(
                edge_comparison["frozen_probability_sensitivity_gate"]["passed"]
            )
            edge_attempt.update(
                {
                    "model_forward_executed": True,
                    "observed_support_probability_count": observed,
                    "observed_support_native_logit_count": observed,
                    "valid_support_mask_receipt_sha256": edge_mask_receipt[
                        "receipt_sha256"
                    ],
                    "valid_support_carrier_payload": _array_receipt(
                        edge_transform.signal[:, :observed],
                        semantic="audit_edge_valid_support_carrier",
                    ),
                    "valid_support_probability_payload": _array_receipt(
                        edge_probability[:observed],
                        semantic="audit_edge_valid_support_probability",
                    ),
                    "valid_support_native_logit_payload": _array_receipt(
                        edge_logit[:observed],
                        semantic="audit_edge_valid_support_native_logit",
                    ),
                    "prediction": edge_prediction_receipt,
                    "formal_reflect_comparison_available": True,
                    "formal_reflect_comparison": edge_comparison,
                }
            )
            edge_numeric_success_count += 1
            edge_model_forward_count += 1
        exclusion_proof = _mask_exclusion_proof(
            row,
            formal_probability,
            formal_mask,
            observed_sample_count=observed,
        )
        result_rows.append(
            _content_address(
                {
                    "schema_version": "st16_short_context_real_edf_record_audit_v1",
                    "analysis_identity_id": row["analysis_identity_id"],
                    "patient_id": row["patient_id"],
                    "source_train_class": (
                        "positive" if row["seizure_event_count"] else "negative"
                    ),
                    "source_binding": {
                        "manifest_row_sha256": _canonical_sha256(row),
                        "edf_relative_path": row["edf_relative_path"],
                        "edf_file_sha256": edf_hash,
                        "canonical_source_tensor_sha256": row[
                            "canonical_source_tensor_sha256"
                        ],
                        "lineage_receipt_sha256": lineage["receipt_sha256"],
                    },
                    "valid_support": {
                        "observed_sample_count": observed,
                        "model_context_sample_count": st.TILE_SAMPLES,
                        "valid_support_sample_range": [0, observed],
                        "right_context_sample_range": [observed, st.TILE_SAMPLES],
                        "mask_payload": _array_receipt(
                            formal_mask, semantic="ST16_original_valid_support_mask"
                        ),
                        "formal_mask_receipt_sha256": formal_mask_receipt[
                            "receipt_sha256"
                        ],
                        "wrap_mask_receipt_sha256": wrap_mask_receipt[
                            "receipt_sha256"
                        ],
                    },
                    "formal_reflect": {
                        "stage_order": (
                            "extend_bipolar_then_resample_filter_whole_tile_MAD"
                        ),
                        "transform_receipt_sha256": formal.receipt[
                            "receipt_sha256"
                        ],
                        "context_ledger_receipt_sha256": formal.receipt[
                            "short_record_context"
                        ]["receipt_sha256"],
                        "carrier_payload_sha256": formal.receipt["output"][
                            "payload_receipt"
                        ]["payload_sha256"],
                        "valid_support_carrier_payload": _array_receipt(
                            formal.signal[:, :observed],
                            semantic="formal_reflect_valid_support_carrier",
                        ),
                        "valid_support_probability_payload": _array_receipt(
                            formal_probability[:observed],
                            semantic="formal_reflect_valid_support_probability",
                        ),
                        "valid_support_native_logit_payload": _array_receipt(
                            formal_logit[:observed],
                            semantic="formal_reflect_valid_support_native_logit",
                        ),
                        "prediction": formal_prediction_receipt,
                    },
                    "audit_only_wrap": {
                        "stage_order": (
                            "extend_bipolar_then_resample_filter_whole_tile_MAD"
                        ),
                        "transform_receipt_sha256": wrap.receipt["receipt_sha256"],
                        "context_ledger_receipt_sha256": wrap.receipt[
                            "short_record_context"
                        ]["receipt_sha256"],
                        "carrier_payload_sha256": wrap.receipt["output"][
                            "payload_receipt"
                        ]["payload_sha256"],
                        "valid_support_carrier_payload": _array_receipt(
                            wrap.signal[:, :observed],
                            semantic="audit_wrap_valid_support_carrier",
                        ),
                        "valid_support_probability_payload": _array_receipt(
                            wrap_probability[:observed],
                            semantic="audit_wrap_valid_support_probability",
                        ),
                        "valid_support_native_logit_payload": _array_receipt(
                            wrap_logit[:observed],
                            semantic="audit_wrap_valid_support_native_logit",
                        ),
                        "prediction": wrap_prediction_receipt,
                    },
                    "audit_only_observed_first_then_normalized_reflect": {
                        "transform_receipt": stage_order_receipt,
                        "carrier_payload_sha256": stage_order_receipt[
                            "extended_output_payload"
                        ]["payload_sha256"],
                        "valid_support_carrier_payload": _array_receipt(
                            stage_order_carrier[:, :observed],
                            semantic=(
                                "observed_first_normalized_reflect_valid_support_carrier"
                            ),
                        ),
                        "valid_support_probability_payload": _array_receipt(
                            stage_probability[:observed],
                            semantic=(
                                "observed_first_normalized_reflect_valid_support_probability"
                            ),
                        ),
                        "valid_support_native_logit_payload": _array_receipt(
                            stage_logit[:observed],
                            semantic=(
                                "observed_first_normalized_reflect_valid_support_native_logit"
                            ),
                        ),
                        "prediction": stage_prediction_receipt,
                    },
                    "audit_only_edge_attempt": edge_attempt,
                    "formal_reflect_vs_audit_wrap": wrap_comparison,
                    "formal_reflect_vs_observed_first_normalized_reflect": (
                        stage_comparison
                    ),
                    "target_loss_metric_and_Finding_context_exclusion": (
                        exclusion_proof
                    ),
                    "receipt_sha256": PENDING,
                }
            )
        )

    cuda_initialized_after = torch.cuda.is_initialized()
    if cuda_initialized_after:
        raise RuntimeError("CPU-only audit unexpectedly initialized CUDA")
    overall_gate = all(all_gates)
    receipt = _content_address(
        {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "completed_frozen_engineering_sensitivity_gate_passed"
                if overall_gate
                else "completed_frozen_engineering_sensitivity_gate_failed"
            ),
            "claim_status": "engineering_sensitivity_only_nonpromotable",
            "scope": {
                "detector_benchmark_or_accuracy_estimate": False,
                "transform_parity_claim": False,
                "performance_claim": False,
                "clinical_use_authorized": False,
                "formal_training_completion_claim": False,
            },
            "frozen_before_real_replay": {
                "fixed_analysis_identity_ids": list(FROZEN_IDENTITIES),
                "probability_sensitivity_limits": deepcopy(
                    FROZEN_PROBABILITY_SENSITIVITY_LIMITS
                ),
                "limits_sha256": _canonical_sha256(
                    FROZEN_PROBABILITY_SENSITIVITY_LIMITS
                ),
                "gate_basis": "each_record_each_counterfactual_valid_support_full",
                "threshold": AUDIT_THRESHOLD,
                "limits_selected_after_observing_results": False,
            },
            "source_bindings": {
                "script_path": str(Path(__file__).resolve().relative_to(ROOT)),
                "script_file_sha256": _file_sha256(Path(__file__).resolve()),
                "manifest_path": str(manifest_source),
                "manifest_file_sha256": _file_sha256(manifest_source),
                "manifest_receipt_sha256": manifest["receipt_sha256"],
                "registry_path": str(registry_path.relative_to(ROOT)),
                "registry_file_sha256": _file_sha256(registry_path),
                "registry_sha256": registry["registry_sha256"],
                "registry_implementation_code_sha256": registry[
                    "implementation"
                ]["code_sha256"],
                "checkpoint_path": str(checkpoint_source),
                "checkpoint_file_sha256": checkpoint_hash,
                "checkpoint_schema_version": checkpoint["schema_version"],
                "checkpoint_completed_epoch_count": checkpoint[
                    "completed_epoch_count"
                ],
            },
            "execution_control": {
                "device": "cpu",
                "cpu_threads": cpu_threads,
                "CUDA_initialized_before": False,
                "CUDA_initialized_after": cuda_initialized_after,
                "CUDA_used": False,
                "training_started": False,
                "optimizer_created_or_stepped": False,
                "backward_called": False,
                "checkpoint_parameters_mutated": False,
                "vLLM_endpoint_contacted": False,
                "vLLM_process_signalled_stopped_or_mutated": False,
            },
            "information_access": {
                "source_train_EDF_signal_open_count": len(result_rows),
                "source_train_target_row_access_count": len(result_rows),
                "source_dev_EDF_or_target_open_count": 0,
                "source_eval_EDF_or_target_open_count": 0,
                "EDF_plus_annotation_API_called": False,
                "spreadsheet_doctor_text_or_clinical_history_opened": False,
                "EEG_signal_and_source_train_global_TERM_target_only": True,
            },
            "fixed_record_replay": {
                "record_count": len(result_rows),
                "positive_record_count": sum(
                    row["source_train_class"] == "positive" for row in result_rows
                ),
                "negative_record_count": sum(
                    row["source_train_class"] == "negative" for row in result_rows
                ),
                "records": result_rows,
            },
            "edge_counterfactual": {
                "attempted_record_count": len(result_rows),
                "typed_technical_failure_count": edge_failure_count,
                "numeric_success_count": edge_numeric_success_count,
                "model_forward_and_observed_support_logit_comparison_count": (
                    edge_model_forward_count
                ),
                "technical_failure_zero_logit_count": edge_failure_count,
                "technical_failure_not_treated_as_no_difference": True,
            },
            "exclude_short_ablation": {
                "policy": "require_at_least_one_native_fully_observed_60_second_tile",
                "selected_record_count": len(result_rows),
                "excluded_record_count": len(result_rows),
                "excluded_fraction": 1.0,
                "excluded_positive_record_count": 1,
                "excluded_negative_record_count": 1,
                "comparable_carrier_probability_or_logit_count": 0,
                "interpretation": (
                    "excluding short records removes 2/2 fixed records and the "
                    "one positive; no short-record context sensitivity can then "
                    "be measured"
                ),
            },
            "frozen_probability_sensitivity_gate": {
                "individual_gate_count": len(all_gates),
                "individual_pass_count": sum(all_gates),
                "all_records_and_counterfactuals_passed": overall_gate,
            },
            "receipt_sha256": PENDING,
        }
    )
    _write_json_atomic(output_path, receipt, replace=replace)
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    receipt = run(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        replace=args.replace,
        cpu_threads=args.cpu_threads,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve(strict=False)),
                "status": receipt["status"],
                "receipt_sha256": receipt["receipt_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
