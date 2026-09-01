"""Fail-closed DeepSOZ shadow adapter for variable-length EEG events.

The adapter preserves the published DeepSOZ channel-logit plus temporal-
attention equation while deliberately changing the context contract: each
input is one independently frozen, variable-length event window of at most
300 seconds and is never padded.  Preprocessing and normalization are applied
to the complete STANDARD_19 recording before an event is sliced.

This is a research transfer, not a reproduction of the published fixed
600-second localization endpoint.  Returned sigmoid values are retained only
as uncalibrated model scores used to form a complete ordinal C18 ranking.  They
are neither probabilities nor a clinical/cortical SOZ diagnosis.

Only EEG samples, their sampling rate, exact channel order, and frozen event
intervals are accepted.  The interval schema is closed so annotation, Excel,
ground-truth, physician text, or clinical context cannot enter this adapter.
TUSZ execution requires an exact published patient-held-out fold binding.
Private execution emits non-consumable fold members until all 15 published
folds are combined by the fixed research ensemble contract below.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .deepsoz_temporal_adapter import (
    DEEPSOZ_FOLD_ASSIGNMENT_SCHEMA_VERSION,
    PUBLISHED_DEEPSOZ_FOLD_WEIGHT_SHA256,
    PUBLISHED_DEEPSOZ_TEST_FOLD_NPY_SHA256,
    PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256,
    STANDARD_19,
    _canonical_sha256,
    _normalize_patient_id,
    _preprocess_standard19,
    _signal_tensor_sha256,
    _snapshot_weights_only_state,
)
from .research_soz_prediction import C18_ELECTRODES


DEEPSOZ_EVENT_SPATIAL_FOLD_SCHEMA_VERSION = (
    "deepsoz_variable_event_spatial_fold_shadow_v1"
)
DEEPSOZ_EVENT_SPATIAL_ENSEMBLE_SCHEMA_VERSION = (
    "deepsoz_variable_event_spatial_ensemble_shadow_v1"
)
DEEPSOZ_EVENT_SPATIAL_ADAPTER_ID = (
    "deepsoz_fullrecord_preprocess_variable_event_attention_c18_shadow_v1"
)
DEEPSOZ_EVENT_MAX_SECONDS = 300
DEEPSOZ_EVENT_SCORE_SEMANTICS = (
    "uncalibrated_sigmoid_model_score_for_ordinal_ranking_not_probability"
)
DEEPSOZ_EVENT_INTERPRETATION_STATUS = (
    "research_scalp_electrode_ranked_hypothesis_not_clinical_or_cortical_soz"
)

_SYNTHETIC = "synthetic_smoke_test"
_TUSZ_OOF = "tusz_patient_oof"
_PRIVATE_MEMBER = "private_research_ensemble_member"
_INFERENCE_MODES = frozenset({_SYNTHETIC, _TUSZ_OOF, _PRIVATE_MEMBER})
_EVENT_KEYS = frozenset(
    {"event_id", "start_offset_seconds", "stop_offset_seconds"}
)
_C18_INDEX = {electrode: index for index, electrode in enumerate(C18_ELECTRODES)}
_FOLD_ARTIFACT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_id",
        "adapter_id",
        "adapter_code_sha256",
        "provider_id",
        "recording_id",
        "inference_mode",
        "fold_index",
        "deepsoz_patient_id",
        "held_out_fold_indices",
        "fold_assignment_receipt_sha256",
        "private_ensemble_contract_sha256",
        "checkpoint_sha256",
        "weights_manifest_sha256",
        "source_signal_tensor_sha256",
        "recording_duration_seconds",
        "preprocessing_receipt",
        "frozen_event_intervals_sha256",
        "event_count",
        "events",
        "output_contract",
        "scope_receipt",
    }
)
_FOLD_EVENT_KEYS = frozenset(
    {
        "event_id",
        "requested_interval_recording_seconds",
        "modeled_interval_recording_seconds",
        "modeled_duration_seconds",
        "processed_window_sha256",
        "boundary_receipt",
        "ranked_electrodes",
        "candidate_space",
        "excluded_electrode",
        "score_semantics",
        "interpretation_status",
        "temporal_attention_receipt",
    }
)
_ENSEMBLE_ARTIFACT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_id",
        "adapter_id",
        "adapter_code_sha256",
        "provider_id",
        "recording_id",
        "inference_mode",
        "deepsoz_patient_id",
        "held_out_fold_indices",
        "fold_assignment_receipt_sha256",
        "private_ensemble_contract_sha256",
        "fold_member_artifact_ids",
        "fold_indices",
        "weights_manifest_sha256",
        "source_signal_tensor_sha256",
        "recording_duration_seconds",
        "preprocessing_receipt",
        "frozen_event_intervals_sha256",
        "event_count",
        "events",
        "output_contract",
        "scope_receipt",
    }
)
_ENSEMBLE_EVENT_KEYS = frozenset(
    {
        "event_id",
        "requested_interval_recording_seconds",
        "modeled_interval_recording_seconds",
        "modeled_duration_seconds",
        "processed_window_sha256",
        "boundary_receipt",
        "ranked_electrodes",
        "candidate_space",
        "excluded_electrode",
        "score_semantics",
        "interpretation_status",
        "fold_fusion",
    }
)
_RANKING_KEYS = frozenset({"rank", "electrode", "score"})
_BOUNDARY_KEYS = frozenset(
    {
        "start_quantization",
        "stop_quantization",
        "requested_interval_fully_retained",
        "recording_left_boundary_reached",
        "recording_right_full_second_boundary_reached",
        "zero_or_silent_padding_seconds",
        "neighbor_event_midpoint_clipping_used",
        "maximum_modeled_seconds",
    }
)
_TEMPORAL_ATTENTION_KEYS = frozenset(
    {
        "formula",
        "attention_sum",
        "attention_is_clinical_onset_probability",
        "variable_length_transfer_is_published_endpoint",
    }
)
_FOLD_OUTPUT_KEYS = frozenset(
    {
        "complete_C18_ordinal_ranking_per_event",
        "score_semantics",
        "interpretation_status",
        "fold_member_consumable_as_private_prediction",
        "fold_member_is_complete_tusz_oof_ensemble",
        "default_pipeline_enabled",
        "shadow_only",
    }
)
_ENSEMBLE_OUTPUT_KEYS = frozenset(
    {
        "complete_C18_ordinal_ranking_per_event",
        "score_semantics",
        "interpretation_status",
        "default_pipeline_enabled",
        "shadow_only",
    }
)
_FOLD_SCOPE_KEYS = frozenset(
    {
        "complete_standard19_eeg_only",
        "edf_annotations_used",
        "excel_used",
        "ground_truth_or_physician_labels_used_for_inference",
        "clinical_context_used",
        "fixed_60_second_compatibility_crop_used",
        "event_padding_used",
        "full_record_preprocessing_before_event_slice",
        "published_fixed_600_second_spatial_endpoint_reproduced",
        "research_only",
        "clinical_soz_claim_authorized",
        "sota_claim_authorized",
    }
)
_PREPROCESSING_KEYS = frozenset(
    {
        "source_sampling_rate_hz",
        "target_sampling_rate_hz",
        "source_sample_count",
        "resampled_sample_count_before_full_second_trim",
        "modeled_full_second_count",
        "resampling",
        "low_pass_filter",
        "high_pass_filter",
        "per_channel_clipping",
        "record_normalization",
        "constant_channel_count",
        "missing_channel_imputation",
        "silent_time_padding",
    }
)

DEEPSOZ_PRIVATE_RESEARCH_ENSEMBLE_CONTRACT: dict[str, Any] = {
    "contract_version": "deepsoz_private_all15_fold_score_mean_shadow_v1",
    "required_fold_indices": list(range(15)),
    "fold_fusion": "arithmetic_mean_of_uncalibrated_per_electrode_model_scores",
    "event_alignment": "exact_frozen_event_interval_and_processed_window_hash",
    "candidate_space": list(C18_ELECTRODES),
    "pz_policy": "excluded_from_C18_endpoint",
    "fold_member_is_consumable_before_complete_ensemble": False,
    "research_only": True,
    "default_batch_enabled": False,
}
DEEPSOZ_PRIVATE_RESEARCH_ENSEMBLE_CONTRACT_SHA256 = _canonical_sha256(
    DEEPSOZ_PRIVATE_RESEARCH_ENSEMBLE_CONTRACT
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise TypeError(f"{context} must be a non-empty opaque identifier")
    allowed = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
    )
    if any(character not in allowed for character in value):
        raise ValueError(f"{context} must be an opaque identifier, not prose or a path")
    return value


def _finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _strict_mapping(
    value: object, *, keys: frozenset[str], context: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    missing = sorted(keys - set(value))
    unknown = sorted(set(value) - keys)
    if missing:
        raise ValueError(f"{context} is missing required keys: {missing}")
    if unknown:
        raise ValueError(f"{context} contains unknown keys: {unknown}")
    return value


def _require_sha256(value: object, context: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return str(value)


def _normalize_event_intervals(
    intervals: object,
    *,
    recording_duration_seconds: float,
) -> list[dict[str, Any]]:
    if not isinstance(intervals, Sequence) or isinstance(intervals, (str, bytes)):
        raise TypeError("frozen_event_intervals must be a non-empty sequence")
    if not intervals:
        raise ValueError("frozen_event_intervals must not be empty")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(intervals):
        if not isinstance(raw, Mapping):
            raise TypeError(f"frozen_event_intervals[{index}] must be a mapping")
        unknown = sorted(set(raw) - _EVENT_KEYS)
        missing = sorted(_EVENT_KEYS - set(raw))
        if missing:
            raise ValueError(
                f"frozen_event_intervals[{index}] is missing keys: {missing}"
            )
        if unknown:
            raise ValueError(
                f"frozen_event_intervals[{index}] contains forbidden/unknown keys: "
                f"{unknown}"
            )
        event_id = _identifier(raw["event_id"], f"frozen_event_intervals[{index}].event_id")
        if event_id in seen:
            raise ValueError("frozen_event_intervals contains duplicate event_id values")
        seen.add(event_id)
        start = _finite_number(
            raw["start_offset_seconds"],
            f"frozen_event_intervals[{index}].start_offset_seconds",
        )
        stop = _finite_number(
            raw["stop_offset_seconds"],
            f"frozen_event_intervals[{index}].stop_offset_seconds",
        )
        if start < 0 or stop <= start or stop > recording_duration_seconds:
            raise ValueError(
                f"frozen event {event_id} must be inside the recording with stop > start"
            )
        modeled_start = int(math.floor(start))
        modeled_stop = int(math.ceil(stop))
        if modeled_stop - modeled_start > DEEPSOZ_EVENT_MAX_SECONDS:
            raise ValueError(
                f"frozen event {event_id} exceeds the {DEEPSOZ_EVENT_MAX_SECONDS}s "
                "modeled-window maximum after full-second retention"
            )
        normalized.append(
            {
                "event_id": event_id,
                "start_offset_seconds": start,
                "stop_offset_seconds": stop,
            }
        )
    return normalized


def frozen_deepsoz_event_intervals_sha256(
    intervals: object,
    *,
    recording_duration_seconds: float,
) -> str:
    """Hash the exact closed-schema event registry accepted by the adapter."""

    normalized = _normalize_event_intervals(
        intervals, recording_duration_seconds=recording_duration_seconds
    )
    return _canonical_sha256(normalized)


def _validate_assignment_receipt(
    receipt: object,
) -> tuple[dict[str, tuple[int, ...]], str]:
    if not isinstance(receipt, Mapping):
        raise TypeError("TUSZ OOF mode requires a fold-assignment receipt mapping")
    value = deepcopy(dict(receipt))
    if value.get("schema_version") != DEEPSOZ_FOLD_ASSIGNMENT_SCHEMA_VERSION:
        raise ValueError("DeepSOZ fold-assignment receipt schema drifted")
    claimed_id = value.get("receipt_id")
    pending = deepcopy(value)
    pending["receipt_id"] = "DEEPSOZ-FOLD-ASSIGNMENT-PENDING"
    if claimed_id != "DSZFOLD-" + _canonical_sha256(pending)[:24]:
        raise ValueError("DeepSOZ fold-assignment receipt content binding failed")
    rows = value.get("patient_fold_assignments")
    if not isinstance(rows, list) or not rows:
        raise ValueError("DeepSOZ fold-assignment receipt has no patient bindings")
    files = value.get("files")
    if not isinstance(files, list) or len(files) != 15:
        raise ValueError("DeepSOZ fold-assignment receipt lacks all 15 audited arrays")
    for fold, file_receipt in enumerate(files):
        if not isinstance(file_receipt, Mapping) or file_receipt != {
            "fold_index": fold,
            "filename": f"deepsoz_official_pts_test_fold{fold}.npy",
            "sha256": PUBLISHED_DEEPSOZ_TEST_FOLD_NPY_SHA256[fold],
            "patient_count": 24,
        }:
            raise ValueError("DeepSOZ fold-assignment array receipt drifted")
    if (
        value.get("file_count") != 15
        or value.get("unique_patient_count") != 124
        or value.get("total_held_out_memberships") != 360
        or value.get("held_out_repeat_count_distribution")
        != {"1": 1, "2": 10, "3": 113}
        or value.get("all_124_patients_have_at_least_one_held_out_fold") is not True
        or value.get("patient_assignment_sha256") != _canonical_sha256(rows)
    ):
        raise ValueError("DeepSOZ fold-assignment cohort receipt drifted")
    assignments: dict[str, tuple[int, ...]] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != 2 or not isinstance(row[1], list):
            raise ValueError("DeepSOZ patient fold binding is malformed")
        patient = _normalize_patient_id(row[0])
        folds = tuple(row[1])
        if (
            not folds
            or len(set(folds)) != len(folds)
            or any(isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < 15 for fold in folds)
        ):
            raise ValueError("DeepSOZ patient fold binding contains invalid folds")
        if patient in assignments:
            raise ValueError("DeepSOZ patient fold binding is duplicated")
        assignments[patient] = folds
    distribution = {
        str(count): sum(len(folds) == count for folds in assignments.values())
        for count in sorted({len(folds) for folds in assignments.values()})
    }
    if (
        len(assignments) != 124
        or sum(len(folds) for folds in assignments.values()) != 360
        or distribution != {"1": 1, "2": 10, "3": 113}
    ):
        raise ValueError("DeepSOZ fold-assignment patient bindings drifted")
    return assignments, _canonical_sha256(value)


class _PublishedDeepSOZEventSpatial(nn.Module):
    """Published 25-tensor architecture with the spatial output exposed."""

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

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            x.ndim != 5
            or x.shape[0:2] != (1, 1)
            or x.shape[-2:] != (19, 200)
            or not 1 <= x.shape[2] <= DEEPSOZ_EVENT_MAX_SECONDS
        ):
            raise ValueError(
                "DeepSOZ event input must be [1,1,T,19,200] with 1<=T<=300"
            )
        seconds = int(x.shape[2])
        positions = self.detector.pos_encoder(
            torch.arange(19, device=x.device)
        ).view(1, 19, 200)
        channel = x.reshape(seconds, 19, 200) + positions
        global_token = self.detector.pos_encoder(
            torch.full((seconds, 1), 19, device=x.device, dtype=torch.long)
        )
        encoded = self.detector.tx_encoder(torch.cat([channel, global_token], dim=1))
        channel_encoded = encoded[:, :19]
        global_encoded = encoded[:, 19].reshape(1, seconds, 200)
        temporal, _ = self.detector.multi_lstm(global_encoded)
        detection_logits = self.detector.multi_linear(temporal).reshape(seconds, 2)
        channel_logits = self.hc_linear(channel_encoded).reshape(seconds, 19)
        attention = F.softmax(detection_logits[:, 1], dim=0)
        channel_scores = torch.sigmoid(
            (attention.unsqueeze(-1) * channel_logits).sum(dim=0)
        )
        return detection_logits, channel_scores, attention


def _window_sha256(value: np.ndarray, *, event_id: str, start: int, stop: int) -> str:
    metadata = json.dumps(
        {
            "dtype": "little_endian_float64",
            "event_id": event_id,
            "modeled_start_second": start,
            "modeled_stop_second": stop,
            "shape": list(value.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    canonical = np.ascontiguousarray(value, dtype="<f8")
    digest = hashlib.sha256(metadata)
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def _rank_c18(scores_19: np.ndarray) -> list[dict[str, Any]]:
    if scores_19.shape != (19,) or not np.isfinite(scores_19).all():
        raise RuntimeError("DeepSOZ produced invalid event spatial scores")
    by_electrode = {
        electrode: float(scores_19[STANDARD_19.index(electrode)])
        for electrode in C18_ELECTRODES
    }
    ordered = sorted(
        C18_ELECTRODES,
        key=lambda electrode: (-by_electrode[electrode], _C18_INDEX[electrode]),
    )
    return [
        {"rank": rank, "electrode": electrode, "score": by_electrode[electrode]}
        for rank, electrode in enumerate(ordered, start=1)
    ]


class DeepSOZEventSpatialShadowAdapter:
    """One-fold, hash-bound variable-event spatial shadow adapter."""

    provider_id = "deepsoz_event_spatial_shadow_candidate_v1"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        expected_checkpoint_sha256: str,
        weights_manifest_sha256: str,
        fold_index: int,
        inference_mode: str,
        deepsoz_patient_id: str | None = None,
        fold_assignment_receipt: Mapping[str, Any] | None = None,
        expected_fold_assignment_receipt_sha256: str | None = None,
        private_ensemble_contract_sha256: str | None = None,
        device: str = "cpu",
    ) -> None:
        if isinstance(fold_index, bool) or not isinstance(fold_index, int) or not 0 <= fold_index < 15:
            raise ValueError("DeepSOZ fold_index must be between 0 and 14")
        if inference_mode not in _INFERENCE_MODES:
            raise ValueError("DeepSOZ event spatial inference mode is invalid")
        if expected_checkpoint_sha256 != PUBLISHED_DEEPSOZ_FOLD_WEIGHT_SHA256[fold_index]:
            raise ValueError("checkpoint SHA-256 is not the published hash for this fold")
        if weights_manifest_sha256 != PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256:
            raise ValueError("DeepSOZ weights manifest is not the published audited manifest")

        self.deepsoz_patient_id: str | None = None
        self.fold_assignment_receipt_sha256: str | None = None
        self.held_out_fold_indices: tuple[int, ...] = ()
        if inference_mode == _TUSZ_OOF:
            if deepsoz_patient_id is None:
                raise ValueError("TUSZ OOF mode requires deepsoz_patient_id")
            patient = _normalize_patient_id(deepsoz_patient_id)
            assignments, receipt_sha256 = _validate_assignment_receipt(
                fold_assignment_receipt
            )
            if (
                not _is_sha256(expected_fold_assignment_receipt_sha256)
                or receipt_sha256 != expected_fold_assignment_receipt_sha256
            ):
                raise ValueError(
                    "TUSZ OOF fold-assignment receipt does not match its trusted SHA-256"
                )
            held_out = assignments.get(patient)
            if held_out is None or fold_index not in held_out:
                raise ValueError(
                    "selected DeepSOZ fold did not hold out the requested TUSZ patient"
                )
            if private_ensemble_contract_sha256 is not None:
                raise ValueError("TUSZ OOF mode cannot claim the private ensemble contract")
            self.deepsoz_patient_id = patient
            self.fold_assignment_receipt_sha256 = receipt_sha256
            self.held_out_fold_indices = held_out
        elif inference_mode == _PRIVATE_MEMBER:
            if (
                deepsoz_patient_id is not None
                or fold_assignment_receipt is not None
                or expected_fold_assignment_receipt_sha256 is not None
            ):
                raise ValueError("private mode cannot consume TUSZ patient split metadata")
            if (
                private_ensemble_contract_sha256
                != DEEPSOZ_PRIVATE_RESEARCH_ENSEMBLE_CONTRACT_SHA256
            ):
                raise ValueError(
                    "private mode requires the exact all-15-fold research ensemble contract"
                )
        else:
            if any(
                value is not None
                for value in (
                    deepsoz_patient_id,
                    fold_assignment_receipt,
                    expected_fold_assignment_receipt_sha256,
                    private_ensemble_contract_sha256,
                )
            ):
                raise ValueError("synthetic smoke mode cannot claim cohort lineage")

        self.checkpoint_path = Path(checkpoint_path)
        self.expected_checkpoint_sha256 = expected_checkpoint_sha256
        self.weights_manifest_sha256 = weights_manifest_sha256
        self.fold_index = fold_index
        self.inference_mode = inference_mode
        self.private_ensemble_contract_sha256 = private_ensemble_contract_sha256
        self.device = torch.device(device)
        state, audit = _snapshot_weights_only_state(
            self.checkpoint_path, expected_sha256=expected_checkpoint_sha256
        )
        model = _PublishedDeepSOZEventSpatial(dropout=0.15).double()
        model.load_state_dict(state, strict=True)
        model.eval().to(self.device)
        self.model = model
        self.checkpoint_audit = audit

    def materialize_event_rankings(
        self,
        *,
        recording_id: str,
        standardized_eeg: object,
        sampling_rate_hz: float,
        channel_names: Sequence[str],
        frozen_event_intervals: object,
        expected_frozen_event_intervals_sha256: str,
    ) -> dict[str, Any]:
        recording = _identifier(recording_id, "recording_id")
        names = tuple(str(value).strip().upper() for value in channel_names)
        if names != STANDARD_19:
            raise ValueError("DeepSOZ adapter requires the exact STANDARD_19 channel order")
        if not isinstance(standardized_eeg, np.ndarray):
            raise TypeError("standardized_eeg must be a NumPy array in microvolts")
        if not _is_sha256(expected_frozen_event_intervals_sha256):
            raise ValueError("expected frozen event interval SHA-256 is invalid")
        raw = np.asarray(standardized_eeg, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[0] != 19:
            raise ValueError("standardized_eeg must have shape [19,n_samples]")
        if (
            raw.shape[1] < 2
            or not np.isfinite(raw).all()
            or np.any(np.std(raw, axis=1) <= 0)
        ):
            raise ValueError(
                "complete STANDARD_19 is required; constant/empty channels fail closed"
            )
        source_rate = _finite_number(sampling_rate_hz, "sampling_rate_hz")
        if source_rate <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        duration = raw.shape[1] / source_rate
        intervals = _normalize_event_intervals(
            frozen_event_intervals, recording_duration_seconds=duration
        )
        interval_sha256 = _canonical_sha256(intervals)
        if interval_sha256 != expected_frozen_event_intervals_sha256:
            raise ValueError("frozen event interval registry SHA-256 mismatch")

        processed, preprocessing = _preprocess_standard19(
            raw, sampling_rate_hz=source_rate
        )
        if preprocessing["constant_channel_count"] != 0:
            raise ValueError(
                "complete STANDARD_19 is required; constant/empty channels fail closed"
            )
        full_seconds = int(preprocessing["modeled_full_second_count"])
        windows = processed.reshape(19, full_seconds, 200).transpose(1, 0, 2)
        source_sha256 = _signal_tensor_sha256(
            raw, sampling_rate_hz=source_rate, channel_names=names
        )
        events: list[dict[str, Any]] = []
        with torch.inference_mode():
            for event in intervals:
                requested_start = float(event["start_offset_seconds"])
                requested_stop = float(event["stop_offset_seconds"])
                start = int(math.floor(requested_start))
                stop = int(math.ceil(requested_stop))
                if start < 0 or stop > full_seconds or stop <= start:
                    raise ValueError(
                        f"frozen event {event['event_id']} reaches an unmodeled partial-second "
                        "recording boundary; trimming or padding is forbidden"
                    )
                event_window = np.ascontiguousarray(windows[start:stop])
                if not 1 <= event_window.shape[0] <= DEEPSOZ_EVENT_MAX_SECONDS:
                    raise RuntimeError("validated DeepSOZ event window length drifted")
                x = torch.from_numpy(event_window).unsqueeze(0).unsqueeze(0)
                _, scores, attention = self.model(
                    x.to(device=self.device, dtype=torch.float64)
                )
                score_values = scores.detach().cpu().numpy()
                attention_values = attention.detach().cpu().numpy()
                if (
                    attention_values.shape != (stop - start,)
                    or not np.isfinite(attention_values).all()
                    or abs(float(attention_values.sum()) - 1.0) > 1e-8
                ):
                    raise RuntimeError("DeepSOZ temporal attention output is invalid")
                events.append(
                    {
                        "event_id": event["event_id"],
                        "requested_interval_recording_seconds": [
                            requested_start,
                            requested_stop,
                        ],
                        "modeled_interval_recording_seconds": [float(start), float(stop)],
                        "modeled_duration_seconds": stop - start,
                        "processed_window_sha256": _window_sha256(
                            event_window,
                            event_id=event["event_id"],
                            start=start,
                            stop=stop,
                        ),
                        "boundary_receipt": {
                            "start_quantization": "floor_to_retain_requested_signal",
                            "stop_quantization": "ceil_to_retain_requested_signal",
                            "requested_interval_fully_retained": True,
                            "recording_left_boundary_reached": start == 0,
                            "recording_right_full_second_boundary_reached": stop == full_seconds,
                            "zero_or_silent_padding_seconds": 0,
                            "neighbor_event_midpoint_clipping_used": False,
                            "maximum_modeled_seconds": DEEPSOZ_EVENT_MAX_SECONDS,
                        },
                        "ranked_electrodes": _rank_c18(score_values),
                        "candidate_space": list(C18_ELECTRODES),
                        "excluded_electrode": "PZ",
                        "score_semantics": DEEPSOZ_EVENT_SCORE_SEMANTICS,
                        "interpretation_status": DEEPSOZ_EVENT_INTERPRETATION_STATUS,
                        "temporal_attention_receipt": {
                            "formula": (
                                "softmax(detection_class1_logits_over_event_seconds); "
                                "sigmoid(sum(attention*channel_logits))"
                            ),
                            "attention_sum": float(attention_values.sum()),
                            "attention_is_clinical_onset_probability": False,
                            "variable_length_transfer_is_published_endpoint": False,
                        },
                    }
                )

        body: dict[str, Any] = {
            "schema_version": DEEPSOZ_EVENT_SPATIAL_FOLD_SCHEMA_VERSION,
            "artifact_id": "DEEPSOZ-EVENT-SPATIAL-FOLD-PENDING",
            "adapter_id": DEEPSOZ_EVENT_SPATIAL_ADAPTER_ID,
            "adapter_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "provider_id": self.provider_id,
            "recording_id": recording,
            "inference_mode": self.inference_mode,
            "fold_index": self.fold_index,
            "deepsoz_patient_id": self.deepsoz_patient_id,
            "held_out_fold_indices": list(self.held_out_fold_indices),
            "fold_assignment_receipt_sha256": self.fold_assignment_receipt_sha256,
            "private_ensemble_contract_sha256": self.private_ensemble_contract_sha256,
            "checkpoint_sha256": self.expected_checkpoint_sha256,
            "weights_manifest_sha256": self.weights_manifest_sha256,
            "source_signal_tensor_sha256": source_sha256,
            "recording_duration_seconds": float(duration),
            "preprocessing_receipt": preprocessing,
            "frozen_event_intervals_sha256": interval_sha256,
            "event_count": len(events),
            "events": events,
            "output_contract": {
                "complete_C18_ordinal_ranking_per_event": True,
                "score_semantics": DEEPSOZ_EVENT_SCORE_SEMANTICS,
                "interpretation_status": DEEPSOZ_EVENT_INTERPRETATION_STATUS,
                "fold_member_consumable_as_private_prediction": False,
                "fold_member_is_complete_tusz_oof_ensemble": (
                    self.inference_mode == _TUSZ_OOF
                    and len(self.held_out_fold_indices) == 1
                ),
                "default_pipeline_enabled": False,
                "shadow_only": True,
            },
            "scope_receipt": {
                "complete_standard19_eeg_only": True,
                "edf_annotations_used": False,
                "excel_used": False,
                "ground_truth_or_physician_labels_used_for_inference": False,
                "clinical_context_used": False,
                "fixed_60_second_compatibility_crop_used": False,
                "event_padding_used": False,
                "full_record_preprocessing_before_event_slice": True,
                "published_fixed_600_second_spatial_endpoint_reproduced": False,
                "research_only": True,
                "clinical_soz_claim_authorized": False,
                "sota_claim_authorized": False,
            },
        }
        body["artifact_id"] = "DSZESF-" + _canonical_sha256(body)[:24]
        return validate_deepsoz_event_spatial_fold_artifact(body)


def _validate_preprocessing_receipt(
    value: object, *, recording_duration_seconds: float
) -> dict[str, Any]:
    receipt = _strict_mapping(
        value, keys=_PREPROCESSING_KEYS, context="preprocessing_receipt"
    )
    source_rate = _finite_number(
        receipt["source_sampling_rate_hz"], "preprocessing source rate"
    )
    target_rate = _finite_number(
        receipt["target_sampling_rate_hz"], "preprocessing target rate"
    )
    integer_fields = (
        "source_sample_count",
        "resampled_sample_count_before_full_second_trim",
        "modeled_full_second_count",
        "constant_channel_count",
    )
    if any(
        isinstance(receipt[field], bool)
        or not isinstance(receipt[field], int)
        or receipt[field] < 0
        for field in integer_fields
    ):
        raise ValueError("DeepSOZ preprocessing integer receipt is invalid")
    expected_literals = {
        "resampling": "scipy_signal_fft_resample_whole_record",
        "low_pass_filter": "butterworth_order4_30Hz_then_filtfilt_gust",
        "high_pass_filter": "butterworth_order4_1_6Hz_then_filtfilt_gust",
        "per_channel_clipping": "mean_plus_or_minus_2_standard_deviations",
        "record_normalization": (
            "whole_processed_record_global_mean_standard_deviation"
        ),
    }
    if (
        source_rate <= 0
        or target_rate != 200.0
        or receipt["source_sample_count"] < 2
        or receipt["modeled_full_second_count"] < 1
        or receipt["modeled_full_second_count"]
        != receipt["resampled_sample_count_before_full_second_trim"] // 200
        or receipt["constant_channel_count"] != 0
        or receipt["missing_channel_imputation"] is not False
        or receipt["silent_time_padding"] is not False
        or any(receipt[key] != expected for key, expected in expected_literals.items())
        or abs(
            receipt["source_sample_count"] / source_rate
            - recording_duration_seconds
        )
        > 1e-9
    ):
        raise ValueError("DeepSOZ preprocessing receipt semantics drifted")
    return deepcopy(dict(receipt))


def _validate_ranking(value: object, *, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(C18_ELECTRODES):
        raise ValueError(f"{context} must contain complete C18")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous_score = math.inf
    for expected_rank, raw_row in enumerate(value, start=1):
        row = _strict_mapping(
            raw_row, keys=_RANKING_KEYS, context=f"{context}[{expected_rank - 1}]"
        )
        if isinstance(row["rank"], bool) or row["rank"] != expected_rank:
            raise ValueError("DeepSOZ event spatial ranks are invalid")
        electrode = row["electrode"]
        if electrode not in _C18_INDEX or electrode in seen:
            raise ValueError("DeepSOZ event spatial C18 electrode set is invalid")
        seen.add(electrode)
        score = _finite_number(row["score"], "DeepSOZ event spatial score")
        if not 0 <= score <= 1:
            raise ValueError("DeepSOZ event spatial score is outside [0,1]")
        if score > previous_score:
            raise ValueError("DeepSOZ event spatial scores are not rank ordered")
        previous_score = score
        output.append({"rank": expected_rank, "electrode": electrode, "score": score})
    if seen != set(C18_ELECTRODES):
        raise ValueError("DeepSOZ event spatial ranking does not cover C18")
    return output


def _validate_event_row(
    value: object,
    *,
    context: str,
    keys: frozenset[str],
    recording_duration_seconds: float,
    modeled_full_seconds: int,
    require_attention: bool,
) -> dict[str, Any]:
    event = _strict_mapping(value, keys=keys, context=context)
    _identifier(event["event_id"], f"{context}.event_id")
    _require_sha256(event["processed_window_sha256"], f"{context}.window hash")
    requested = event["requested_interval_recording_seconds"]
    modeled = event["modeled_interval_recording_seconds"]
    if (
        not isinstance(requested, list)
        or len(requested) != 2
        or not isinstance(modeled, list)
        or len(modeled) != 2
    ):
        raise ValueError(f"{context} intervals must be two-element lists")
    requested_start = _finite_number(requested[0], f"{context} requested start")
    requested_stop = _finite_number(requested[1], f"{context} requested stop")
    modeled_start = _finite_number(modeled[0], f"{context} modeled start")
    modeled_stop = _finite_number(modeled[1], f"{context} modeled stop")
    duration = event["modeled_duration_seconds"]
    if (
        isinstance(duration, bool)
        or not isinstance(duration, int)
        or not 1 <= duration <= DEEPSOZ_EVENT_MAX_SECONDS
        or requested_start < 0
        or requested_stop <= requested_start
        or requested_stop > recording_duration_seconds
        or modeled_start != float(math.floor(requested_start))
        or modeled_stop != float(math.ceil(requested_stop))
        or modeled_start < 0
        or modeled_stop > modeled_full_seconds
        or modeled_stop - modeled_start != duration
    ):
        raise ValueError(f"{context} interval/duration binding drifted")
    boundary = _strict_mapping(
        event["boundary_receipt"],
        keys=_BOUNDARY_KEYS,
        context=f"{context}.boundary_receipt",
    )
    expected_boundary = {
        "start_quantization": "floor_to_retain_requested_signal",
        "stop_quantization": "ceil_to_retain_requested_signal",
        "requested_interval_fully_retained": True,
        "recording_left_boundary_reached": modeled_start == 0,
        "recording_right_full_second_boundary_reached": (
            modeled_stop == modeled_full_seconds
        ),
        "zero_or_silent_padding_seconds": 0,
        "neighbor_event_midpoint_clipping_used": False,
        "maximum_modeled_seconds": DEEPSOZ_EVENT_MAX_SECONDS,
    }
    if dict(boundary) != expected_boundary:
        raise ValueError(f"{context} boundary receipt drifted")
    if (
        event["candidate_space"] != list(C18_ELECTRODES)
        or event["excluded_electrode"] != "PZ"
        or event["score_semantics"] != DEEPSOZ_EVENT_SCORE_SEMANTICS
        or event["interpretation_status"]
        != DEEPSOZ_EVENT_INTERPRETATION_STATUS
    ):
        raise ValueError(f"{context} candidate/interpretation contract drifted")
    _validate_ranking(event["ranked_electrodes"], context=f"{context}.ranking")
    if require_attention:
        attention = _strict_mapping(
            event["temporal_attention_receipt"],
            keys=_TEMPORAL_ATTENTION_KEYS,
            context=f"{context}.temporal_attention_receipt",
        )
        if (
            attention["formula"]
            != (
                "softmax(detection_class1_logits_over_event_seconds); "
                "sigmoid(sum(attention*channel_logits))"
            )
            or abs(
                _finite_number(attention["attention_sum"], f"{context} attention sum")
                - 1.0
            )
            > 1e-8
            or attention["attention_is_clinical_onset_probability"] is not False
            or attention["variable_length_transfer_is_published_endpoint"] is not False
        ):
            raise ValueError(f"{context} temporal attention receipt drifted")
    else:
        if event["fold_fusion"] != (
            "arithmetic_mean_of_uncalibrated_per_electrode_model_scores"
        ):
            raise ValueError(f"{context} fold fusion contract drifted")
    return deepcopy(dict(event))


def _validate_scope(value: object, *, context: str) -> dict[str, Any]:
    scope = _strict_mapping(value, keys=_FOLD_SCOPE_KEYS, context=context)
    expected = {
        "complete_standard19_eeg_only": True,
        "edf_annotations_used": False,
        "excel_used": False,
        "ground_truth_or_physician_labels_used_for_inference": False,
        "clinical_context_used": False,
        "fixed_60_second_compatibility_crop_used": False,
        "event_padding_used": False,
        "full_record_preprocessing_before_event_slice": True,
        "published_fixed_600_second_spatial_endpoint_reproduced": False,
        "research_only": True,
        "clinical_soz_claim_authorized": False,
        "sota_claim_authorized": False,
    }
    if dict(scope) != expected:
        raise ValueError(f"{context} semantics drifted")
    return deepcopy(dict(scope))


def validate_deepsoz_event_spatial_fold_artifact(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly validate a one-fold event ranking shadow artifact."""

    artifact_map = _strict_mapping(
        value, keys=_FOLD_ARTIFACT_KEYS, context="DeepSOZ spatial fold artifact"
    )
    artifact = deepcopy(dict(artifact_map))
    if artifact["schema_version"] != DEEPSOZ_EVENT_SPATIAL_FOLD_SCHEMA_VERSION:
        raise ValueError("DeepSOZ event spatial fold schema drifted")
    if artifact["adapter_id"] != DEEPSOZ_EVENT_SPATIAL_ADAPTER_ID:
        raise ValueError("DeepSOZ event spatial adapter ID drifted")
    if artifact["provider_id"] != DeepSOZEventSpatialShadowAdapter.provider_id:
        raise ValueError("DeepSOZ event spatial provider ID drifted")
    current_code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if artifact["adapter_code_sha256"] != current_code_sha256:
        raise ValueError("DeepSOZ event spatial adapter code hash drifted")
    claimed = artifact["artifact_id"]
    pending = deepcopy(artifact)
    pending["artifact_id"] = "DEEPSOZ-EVENT-SPATIAL-FOLD-PENDING"
    if claimed != "DSZESF-" + _canonical_sha256(pending)[:24]:
        raise ValueError("DeepSOZ event spatial fold artifact content binding failed")
    _identifier(artifact["recording_id"], "fold artifact recording_id")
    _require_sha256(
        artifact["source_signal_tensor_sha256"], "fold artifact source signal hash"
    )
    _require_sha256(
        artifact["frozen_event_intervals_sha256"], "fold artifact event registry hash"
    )
    duration = _finite_number(
        artifact["recording_duration_seconds"], "fold artifact recording duration"
    )
    if duration <= 0:
        raise ValueError("DeepSOZ event spatial recording duration must be positive")
    preprocessing = _validate_preprocessing_receipt(
        artifact["preprocessing_receipt"], recording_duration_seconds=duration
    )
    fold = artifact["fold_index"]
    if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < 15:
        raise ValueError("DeepSOZ event spatial fold index is invalid")
    if artifact["checkpoint_sha256"] != PUBLISHED_DEEPSOZ_FOLD_WEIGHT_SHA256[fold]:
        raise ValueError("DeepSOZ event spatial checkpoint binding drifted")
    if artifact["weights_manifest_sha256"] != PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256:
        raise ValueError("DeepSOZ event spatial weight manifest drifted")

    mode = artifact["inference_mode"]
    held_out = artifact["held_out_fold_indices"]
    if not isinstance(held_out, list):
        raise TypeError("DeepSOZ held_out_fold_indices must be a list")
    if mode == _SYNTHETIC:
        valid_lineage = (
            artifact["deepsoz_patient_id"] is None
            and held_out == []
            and artifact["fold_assignment_receipt_sha256"] is None
            and artifact["private_ensemble_contract_sha256"] is None
        )
    elif mode == _TUSZ_OOF:
        patient = _normalize_patient_id(artifact["deepsoz_patient_id"])
        valid_folds = (
            held_out
            and len(held_out) == len(set(held_out))
            and all(
                not isinstance(value, bool)
                and isinstance(value, int)
                and 0 <= value < 15
                for value in held_out
            )
            and fold in held_out
        )
        valid_lineage = (
            artifact["deepsoz_patient_id"] == patient
            and bool(valid_folds)
            and _is_sha256(artifact["fold_assignment_receipt_sha256"])
            and artifact["private_ensemble_contract_sha256"] is None
        )
    elif mode == _PRIVATE_MEMBER:
        valid_lineage = (
            artifact["deepsoz_patient_id"] is None
            and held_out == []
            and artifact["fold_assignment_receipt_sha256"] is None
            and artifact["private_ensemble_contract_sha256"]
            == DEEPSOZ_PRIVATE_RESEARCH_ENSEMBLE_CONTRACT_SHA256
        )
    else:
        valid_lineage = False
    if not valid_lineage:
        raise ValueError("DeepSOZ event spatial inference lineage drifted")

    output = _strict_mapping(
        artifact["output_contract"],
        keys=_FOLD_OUTPUT_KEYS,
        context="fold artifact output_contract",
    )
    expected_output = {
        "complete_C18_ordinal_ranking_per_event": True,
        "score_semantics": DEEPSOZ_EVENT_SCORE_SEMANTICS,
        "interpretation_status": DEEPSOZ_EVENT_INTERPRETATION_STATUS,
        "fold_member_consumable_as_private_prediction": False,
        "fold_member_is_complete_tusz_oof_ensemble": (
            mode == _TUSZ_OOF and len(held_out) == 1
        ),
        "default_pipeline_enabled": False,
        "shadow_only": True,
    }
    if dict(output) != expected_output:
        raise ValueError("DeepSOZ event spatial fold output contract drifted")
    _validate_scope(artifact["scope_receipt"], context="fold artifact scope_receipt")

    events = artifact["events"]
    event_count = artifact["event_count"]
    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count < 1
        or not isinstance(events, list)
        or len(events) != event_count
    ):
        raise ValueError("DeepSOZ event spatial artifact has invalid event rows")
    seen_events: set[str] = set()
    for index, event in enumerate(events):
        validated = _validate_event_row(
            event,
            context=f"fold artifact events[{index}]",
            keys=_FOLD_EVENT_KEYS,
            recording_duration_seconds=duration,
            modeled_full_seconds=preprocessing["modeled_full_second_count"],
            require_attention=True,
        )
        if validated["event_id"] in seen_events:
            raise ValueError("DeepSOZ event spatial artifact repeats event_id")
        seen_events.add(validated["event_id"])
    return artifact


def _validate_fold_artifact(value: object) -> dict[str, Any]:
    """Backward-private alias retained for the internal ensemble path."""

    if not isinstance(value, Mapping):
        raise TypeError("DeepSOZ event spatial fold artifact must be a mapping")
    return validate_deepsoz_event_spatial_fold_artifact(value)


def _aggregate_fold_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    expected_folds: Sequence[int],
    output_mode: str,
    patient_id: str | None,
    fold_assignment_receipt_sha256: str | None,
    private_contract_sha256: str | None,
) -> dict[str, Any]:
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise TypeError("fold_artifacts must be a non-empty sequence")
    values = [_validate_fold_artifact(value) for value in artifacts]
    observed_folds = sorted(value["fold_index"] for value in values)
    if observed_folds != sorted(expected_folds) or len(observed_folds) != len(set(observed_folds)):
        raise ValueError("DeepSOZ event spatial ensemble fold roster is incomplete or duplicated")
    values.sort(key=lambda value: value["fold_index"])
    first = values[0]
    comparison_fields = (
        "adapter_code_sha256",
        "recording_id",
        "source_signal_tensor_sha256",
        "recording_duration_seconds",
        "preprocessing_receipt",
        "frozen_event_intervals_sha256",
        "event_count",
    )
    for value in values[1:]:
        if any(value.get(field) != first.get(field) for field in comparison_fields):
            raise ValueError("DeepSOZ spatial fold members do not share identical EEG/event lineage")

    output_events: list[dict[str, Any]] = []
    for event_index in range(first["event_count"]):
        rows = [value["events"][event_index] for value in values]
        event_id = rows[0].get("event_id")
        binding_fields = (
            "event_id",
            "requested_interval_recording_seconds",
            "modeled_interval_recording_seconds",
            "modeled_duration_seconds",
            "processed_window_sha256",
            "boundary_receipt",
        )
        if any(
            any(row.get(field) != rows[0].get(field) for field in binding_fields)
            for row in rows[1:]
        ):
            raise ValueError(f"DeepSOZ spatial fold event lineage differs for {event_id}")
        score_maps = [
            {item["electrode"]: float(item["score"]) for item in row["ranked_electrodes"]}
            for row in rows
        ]
        mean_19 = np.zeros(19, dtype=np.float64)
        for electrode in C18_ELECTRODES:
            mean_19[STANDARD_19.index(electrode)] = float(
                np.mean([mapping[electrode] for mapping in score_maps])
            )
        output_events.append(
            {
                "event_id": event_id,
                "requested_interval_recording_seconds": deepcopy(
                    rows[0]["requested_interval_recording_seconds"]
                ),
                "modeled_interval_recording_seconds": deepcopy(
                    rows[0]["modeled_interval_recording_seconds"]
                ),
                "modeled_duration_seconds": rows[0]["modeled_duration_seconds"],
                "processed_window_sha256": rows[0]["processed_window_sha256"],
                "boundary_receipt": deepcopy(rows[0]["boundary_receipt"]),
                "ranked_electrodes": _rank_c18(mean_19),
                "candidate_space": list(C18_ELECTRODES),
                "excluded_electrode": "PZ",
                "score_semantics": DEEPSOZ_EVENT_SCORE_SEMANTICS,
                "interpretation_status": DEEPSOZ_EVENT_INTERPRETATION_STATUS,
                "fold_fusion": (
                    "arithmetic_mean_of_uncalibrated_per_electrode_model_scores"
                ),
            }
        )

    body: dict[str, Any] = {
        "schema_version": DEEPSOZ_EVENT_SPATIAL_ENSEMBLE_SCHEMA_VERSION,
        "artifact_id": "DEEPSOZ-EVENT-SPATIAL-ENSEMBLE-PENDING",
        "adapter_id": DEEPSOZ_EVENT_SPATIAL_ADAPTER_ID,
        "adapter_code_sha256": first["adapter_code_sha256"],
        "provider_id": DeepSOZEventSpatialShadowAdapter.provider_id,
        "recording_id": first["recording_id"],
        "inference_mode": output_mode,
        "deepsoz_patient_id": patient_id,
        "held_out_fold_indices": list(sorted(expected_folds)) if patient_id is not None else [],
        "fold_assignment_receipt_sha256": fold_assignment_receipt_sha256,
        "private_ensemble_contract_sha256": private_contract_sha256,
        "fold_member_artifact_ids": [value["artifact_id"] for value in values],
        "fold_indices": list(sorted(expected_folds)),
        "weights_manifest_sha256": PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256,
        "source_signal_tensor_sha256": first["source_signal_tensor_sha256"],
        "recording_duration_seconds": first["recording_duration_seconds"],
        "preprocessing_receipt": first["preprocessing_receipt"],
        "frozen_event_intervals_sha256": first["frozen_event_intervals_sha256"],
        "event_count": len(output_events),
        "events": output_events,
        "output_contract": {
            "complete_C18_ordinal_ranking_per_event": True,
            "score_semantics": DEEPSOZ_EVENT_SCORE_SEMANTICS,
            "interpretation_status": DEEPSOZ_EVENT_INTERPRETATION_STATUS,
            "default_pipeline_enabled": False,
            "shadow_only": True,
        },
        "scope_receipt": {
            "complete_standard19_eeg_only": True,
            "edf_annotations_used": False,
            "excel_used": False,
            "ground_truth_or_physician_labels_used_for_inference": False,
            "clinical_context_used": False,
            "fixed_60_second_compatibility_crop_used": False,
            "event_padding_used": False,
            "full_record_preprocessing_before_event_slice": True,
            "published_fixed_600_second_spatial_endpoint_reproduced": False,
            "research_only": True,
            "clinical_soz_claim_authorized": False,
            "sota_claim_authorized": False,
        },
    }
    body["artifact_id"] = "DSZESE-" + _canonical_sha256(body)[:24]
    return validate_deepsoz_event_spatial_ensemble_artifact(body)


def validate_deepsoz_event_spatial_ensemble_artifact(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly validate a released TUSZ/private spatial shadow ensemble."""

    artifact_map = _strict_mapping(
        value,
        keys=_ENSEMBLE_ARTIFACT_KEYS,
        context="DeepSOZ spatial ensemble artifact",
    )
    artifact = deepcopy(dict(artifact_map))
    if artifact["schema_version"] != DEEPSOZ_EVENT_SPATIAL_ENSEMBLE_SCHEMA_VERSION:
        raise ValueError("DeepSOZ event spatial ensemble schema drifted")
    if artifact["adapter_id"] != DEEPSOZ_EVENT_SPATIAL_ADAPTER_ID:
        raise ValueError("DeepSOZ event spatial ensemble adapter ID drifted")
    if artifact["provider_id"] != DeepSOZEventSpatialShadowAdapter.provider_id:
        raise ValueError("DeepSOZ event spatial ensemble provider ID drifted")
    current_code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if artifact["adapter_code_sha256"] != current_code_sha256:
        raise ValueError("DeepSOZ event spatial ensemble code hash drifted")
    claimed = artifact["artifact_id"]
    pending = deepcopy(artifact)
    pending["artifact_id"] = "DEEPSOZ-EVENT-SPATIAL-ENSEMBLE-PENDING"
    if claimed != "DSZESE-" + _canonical_sha256(pending)[:24]:
        raise ValueError("DeepSOZ event spatial ensemble content binding failed")
    _identifier(artifact["recording_id"], "ensemble recording_id")
    _require_sha256(
        artifact["source_signal_tensor_sha256"], "ensemble source signal hash"
    )
    _require_sha256(
        artifact["frozen_event_intervals_sha256"], "ensemble event registry hash"
    )
    if artifact["weights_manifest_sha256"] != PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256:
        raise ValueError("DeepSOZ event spatial ensemble weight manifest drifted")
    duration = _finite_number(
        artifact["recording_duration_seconds"], "ensemble recording duration"
    )
    if duration <= 0:
        raise ValueError("DeepSOZ ensemble recording duration must be positive")
    preprocessing = _validate_preprocessing_receipt(
        artifact["preprocessing_receipt"], recording_duration_seconds=duration
    )

    folds = artifact["fold_indices"]
    held_out = artifact["held_out_fold_indices"]
    member_ids = artifact["fold_member_artifact_ids"]
    if (
        not isinstance(folds, list)
        or not folds
        or folds != sorted(set(folds))
        or any(
            isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < 15
            for fold in folds
        )
        or not isinstance(held_out, list)
        or not isinstance(member_ids, list)
        or len(member_ids) != len(folds)
        or len(set(member_ids)) != len(member_ids)
        or any(
            not isinstance(member_id, str)
            or not member_id.startswith("DSZESF-")
            or len(member_id) != 31
            or any(character not in "0123456789abcdef" for character in member_id[7:])
            for member_id in member_ids
        )
    ):
        raise ValueError("DeepSOZ event spatial ensemble fold/member roster drifted")
    mode = artifact["inference_mode"]
    if mode == "tusz_patient_oof_ensemble":
        patient = _normalize_patient_id(artifact["deepsoz_patient_id"])
        valid_lineage = (
            artifact["deepsoz_patient_id"] == patient
            and held_out == folds
            and 1 <= len(folds) <= 3
            and _is_sha256(artifact["fold_assignment_receipt_sha256"])
            and artifact["private_ensemble_contract_sha256"] is None
        )
    elif mode == "private_research_all15_fold_ensemble":
        valid_lineage = (
            artifact["deepsoz_patient_id"] is None
            and held_out == []
            and folds == list(range(15))
            and artifact["fold_assignment_receipt_sha256"] is None
            and artifact["private_ensemble_contract_sha256"]
            == DEEPSOZ_PRIVATE_RESEARCH_ENSEMBLE_CONTRACT_SHA256
        )
    else:
        valid_lineage = False
    if not valid_lineage:
        raise ValueError("DeepSOZ event spatial ensemble inference lineage drifted")

    output = _strict_mapping(
        artifact["output_contract"],
        keys=_ENSEMBLE_OUTPUT_KEYS,
        context="ensemble output_contract",
    )
    expected_output = {
        "complete_C18_ordinal_ranking_per_event": True,
        "score_semantics": DEEPSOZ_EVENT_SCORE_SEMANTICS,
        "interpretation_status": DEEPSOZ_EVENT_INTERPRETATION_STATUS,
        "default_pipeline_enabled": False,
        "shadow_only": True,
    }
    if dict(output) != expected_output:
        raise ValueError("DeepSOZ event spatial ensemble output contract drifted")
    _validate_scope(
        artifact["scope_receipt"], context="ensemble artifact scope_receipt"
    )

    events = artifact["events"]
    event_count = artifact["event_count"]
    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count < 1
        or not isinstance(events, list)
        or len(events) != event_count
    ):
        raise ValueError("DeepSOZ event spatial ensemble has invalid event rows")
    seen_events: set[str] = set()
    for index, event in enumerate(events):
        validated = _validate_event_row(
            event,
            context=f"ensemble events[{index}]",
            keys=_ENSEMBLE_EVENT_KEYS,
            recording_duration_seconds=duration,
            modeled_full_seconds=preprocessing["modeled_full_second_count"],
            require_attention=False,
        )
        if validated["event_id"] in seen_events:
            raise ValueError("DeepSOZ event spatial ensemble repeats event_id")
        seen_events.add(validated["event_id"])
    return artifact


def aggregate_deepsoz_tusz_oof_event_rankings(
    *,
    patient_id: str,
    fold_assignment_receipt: Mapping[str, Any],
    fold_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fuse every published fold that held out one TUSZ patient."""

    patient = _normalize_patient_id(patient_id)
    assignments, receipt_sha256 = _validate_assignment_receipt(
        fold_assignment_receipt
    )
    expected = assignments.get(patient)
    if expected is None:
        raise ValueError("TUSZ patient is absent from the published held-out assignments")
    for artifact in fold_artifacts:
        if (
            artifact.get("inference_mode") != _TUSZ_OOF
            or artifact.get("deepsoz_patient_id") != patient
            or artifact.get("fold_assignment_receipt_sha256") != receipt_sha256
        ):
            raise ValueError("TUSZ spatial fold artifact lineage does not match the patient")
    return _aggregate_fold_artifacts(
        fold_artifacts,
        expected_folds=expected,
        output_mode="tusz_patient_oof_ensemble",
        patient_id=patient,
        fold_assignment_receipt_sha256=receipt_sha256,
        private_contract_sha256=None,
    )


def aggregate_deepsoz_private_event_rankings(
    *,
    fold_artifacts: Sequence[Mapping[str, Any]],
    private_ensemble_contract_sha256: str,
) -> dict[str, Any]:
    """Release a private shadow ranking only after all 15 folds are present."""

    if (
        private_ensemble_contract_sha256
        != DEEPSOZ_PRIVATE_RESEARCH_ENSEMBLE_CONTRACT_SHA256
    ):
        raise ValueError("private DeepSOZ ensemble contract SHA-256 mismatch")
    for artifact in fold_artifacts:
        if (
            artifact.get("inference_mode") != _PRIVATE_MEMBER
            or artifact.get("private_ensemble_contract_sha256")
            != private_ensemble_contract_sha256
            or artifact.get("deepsoz_patient_id") is not None
            or artifact.get("fold_assignment_receipt_sha256") is not None
        ):
            raise ValueError("private spatial fold artifact has incompatible lineage")
    return _aggregate_fold_artifacts(
        fold_artifacts,
        expected_folds=range(15),
        output_mode="private_research_all15_fold_ensemble",
        patient_id=None,
        fold_assignment_receipt_sha256=None,
        private_contract_sha256=private_ensemble_contract_sha256,
    )


__all__ = [
    "DEEPSOZ_EVENT_INTERPRETATION_STATUS",
    "DEEPSOZ_EVENT_MAX_SECONDS",
    "DEEPSOZ_EVENT_SCORE_SEMANTICS",
    "DEEPSOZ_EVENT_SPATIAL_ADAPTER_ID",
    "DEEPSOZ_EVENT_SPATIAL_ENSEMBLE_SCHEMA_VERSION",
    "DEEPSOZ_EVENT_SPATIAL_FOLD_SCHEMA_VERSION",
    "DEEPSOZ_PRIVATE_RESEARCH_ENSEMBLE_CONTRACT",
    "DEEPSOZ_PRIVATE_RESEARCH_ENSEMBLE_CONTRACT_SHA256",
    "DeepSOZEventSpatialShadowAdapter",
    "aggregate_deepsoz_private_event_rankings",
    "aggregate_deepsoz_tusz_oof_event_rankings",
    "frozen_deepsoz_event_intervals_sha256",
    "validate_deepsoz_event_spatial_ensemble_artifact",
    "validate_deepsoz_event_spatial_fold_artifact",
]
