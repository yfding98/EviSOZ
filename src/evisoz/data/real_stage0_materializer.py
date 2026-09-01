"""Real-EDF source reader for the EviSOZ Stage-0 dual-montage carrier.

Only signal labels, per-signal physical metadata and physical samples are read.
Patient/recording headers, EDF annotations, spreadsheets and clinical text are
outside this module's API.  The selected direct electrode field is processed
once, before rereferencing, so the frozen CAR19 v29 view and signed TCP22 view
share the same causal filter, rational resampling delay and event clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from src.clinical_eeg_long_recording.canonical_edf_materialization import (
    _reader_factory as _signal_only_reader_factory,
)
from src.clinical_eeg_long_recording.montage_reference_observability import (
    build_montage_reference_observability_receipt,
    classify_signal_labels,
)
from src.evisoz.baseline.v29_cache import canonical_tensor_bytes
from src.evisoz.data.artifact_ref import (
    build_json_artifact_ref,
    build_raw_artifact_ref,
    canonical_json_sha256,
    validate_artifact_ref,
)
from src.evisoz.data.event_identity import build_event_identity
from src.evisoz.data.opaque_reference_authority import (
    OPAQUE_REFERENCE_EVENT_AUTHORIZATION_SCHEMA_VERSION,
    build_opaque_reference_event_authorization,
    validate_private_opaque_reference_authority,
)
from src.evisoz.data.stage0_dual_montage_cache import (
    Stage0DualMontageCarrier,
    build_common_reference_event_carrier,
)
from src.soz.data.edf import (
    CausalEDFConfig,
    causal_bandpass_resample_channel_field,
)
from src.soz.geometry import STANDARD_19, normalize_electrode_name


REAL_STAGE0_PREPROCESSING_SCHEMA_VERSION = (
    "evisoz_real_common_reference_event_preprocessing_v1"
)
PARENT_ELECTRODES = (*STANDARD_19, "A1", "A2")
PARENT_SAMPLE_COUNT = 12_000
OUTPUT_SFREQ_HZ = 200.0
ANALYSIS_INTERVAL_SECONDS = (-12.0, 48.0)

_PLACEHOLDER = "0" * 64
_UNIT_TO_VOLTS = {
    "v": 1.0,
    "mv": 1e-3,
    "uv": 1e-6,
}


@dataclass(frozen=True)
class RealStage0EventCarrier:
    """One real event carrier plus its signal-only preprocessing receipt."""

    carrier: Stage0DualMontageCarrier
    preprocessing_receipt: Mapping[str, Any]


def _half_up(value: float) -> int:
    if not math.isfinite(float(value)):
        raise ValueError("sample coordinate must be finite")
    return int(
        Decimal(str(float(value))).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )


def _unit_scale(value: object) -> float:
    normalized = str(value).strip().lower().replace("µ", "u").replace("μ", "u")
    try:
        return _UNIT_TO_VOLTS[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported EDF physical unit: {value!r}") from exc


def _source_edf(path: str | Path) -> Path:
    source = Path(path)
    if source.is_symlink():
        raise ValueError("real Stage-0 EDF source must not be a symbolic link")
    resolved = source.resolve(strict=True)
    if not resolved.is_file() or resolved.suffix.lower() != ".edf":
        raise ValueError("real Stage-0 source must be a regular EDF file")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config() -> CausalEDFConfig:
    return CausalEDFConfig(
        output_sfreq_hz=OUTPUT_SFREQ_HZ,
        highpass_hz=0.5,
        lowpass_hz=45.0,
        butterworth_order=4,
        warmup_sec=30.0,
        pre_onset_sec=12.0,
        post_onset_sec=48.0,
        apply_car19=False,
    )


def _inventory(labels: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    candidates: dict[str, list[int]] = {name: [] for name in PARENT_ELECTRODES}
    for index, label in enumerate(labels):
        normalized = normalize_electrode_name(label)
        if normalized in candidates:
            candidates[normalized].append(index)
    duplicates = {
        name: tuple(labels[index] for index in indices)
        for name, indices in candidates.items()
        if len(indices) > 1
    }
    if duplicates:
        raise ValueError(
            "real Stage-0 EDF has ambiguous direct electrode rows: "
            f"{sorted(duplicates)}"
        )
    observed = tuple(name for name in PARENT_ELECTRODES if candidates[name])
    if not any(name in STANDARD_19 for name in observed):
        raise ValueError("real Stage-0 EDF has no directly observed Standard19 row")
    indices = tuple(candidates[name][0] for name in observed)
    return observed, indices


def _preprocessing_receipt(
    *,
    source_edf_sha256: str,
    source_labels_sha256: str,
    reader_policy: str,
    source_sfreq_hz: float,
    source_sample_count: int,
    read_start_sample: int,
    read_stop_sample: int,
    requested_onset_seconds: float,
    source_aligned_onset_seconds: float,
    observed_mask: torch.Tensor,
    resample_up: int,
    resample_down: int,
    resample_fir_taps: int,
    resample_latency_seconds: float,
    crop_start_output_sample: int,
    parent_signal_ref: Mapping[str, object],
    event_identity: Mapping[str, object],
    reference_route: str,
    reference_authorization_ref: Mapping[str, object] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": REAL_STAGE0_PREPROCESSING_SCHEMA_VERSION,
        "source_edf_sha256": source_edf_sha256,
        "source_signal_labels_sha256": source_labels_sha256,
        "reader_policy": reader_policy,
        "source_sampling_rate_hz": source_sfreq_hz,
        "source_sample_count": source_sample_count,
        "read_sample_interval": [read_start_sample, read_stop_sample],
        "requested_onset_seconds": requested_onset_seconds,
        "source_aligned_onset_seconds": source_aligned_onset_seconds,
        "source_alignment_error_seconds": (
            source_aligned_onset_seconds - requested_onset_seconds
        ),
        "parent_electrode_order": list(PARENT_ELECTRODES),
        "parent_observed_mask": observed_mask.tolist(),
        "output_sampling_rate_hz": OUTPUT_SFREQ_HZ,
        "analysis_interval_seconds": list(ANALYSIS_INTERVAL_SECONDS),
        "output_sample_count": PARENT_SAMPLE_COUNT,
        "filter": {
            "kind": "causal_sos_butter_bandpass_v1",
            "highpass_hz": 0.5,
            "lowpass_hz": 45.0,
            "order": 4,
            "state_reset_at_read_start": True,
            "real_warmup_seconds": 30.0,
        },
        "resampling": {
            "kind": "causal_upfirdn_kaiser5_v1_delay_retained",
            "up": resample_up,
            "down": resample_down,
            "fir_taps": resample_fir_taps,
            "latency_seconds": resample_latency_seconds,
            "crop_start_output_sample": crop_start_output_sample,
        },
        "parent_signal_ref": deepcopy(dict(parent_signal_ref)),
        "reference_route": reference_route,
        "reference_authorization_ref": (
            None
            if reference_authorization_ref is None
            else deepcopy(dict(reference_authorization_ref))
        ),
        "event_identity_ref": build_json_artifact_ref(
            event_identity,
            artifact_kind="event_identity",
            payload_schema_version="evisoz_event_identity_v1",
        ),
        "scope_contract": {
            "eeg_samples_used": True,
            "edf_signal_header_used": True,
            "edf_patient_or_recording_header_api_called": False,
            "edf_annotation_api_called": False,
            "edf_annotations_used": False,
            "clinical_text_used": False,
            "doctor_labels_used": False,
            "knowledge_used": False,
        },
        "receipt_sha256": _PLACEHOLDER,
    }
    body["receipt_sha256"] = canonical_json_sha256(body)
    return body


def load_real_stage0_event_carrier(
    edf_path: str | Path,
    onset_seconds: float,
    *,
    dataset_id: str,
    sample_id: str,
    event_id: str,
    linkage_group_id: str,
    source_patient_sha256: str,
    anchor_source_ref: Mapping[str, object],
    anchor_quality: str = "exact",
    opaque_reference_authority: Mapping[str, object] | None = None,
    reader_factory: Callable[[str], object] | None = None,
) -> RealStage0EventCarrier:
    """Read one real known-onset event and derive same-clock CAR19/TCP22.

    The raw file path and source labels are never persisted.  Their byte and
    ordered-inventory hashes are retained in the preprocessing receipt.
    """

    source = _source_edf(edf_path)
    source_edf_sha256 = _sha256_file(source)
    requested_onset = float(onset_seconds)
    if not math.isfinite(requested_onset) or requested_onset < 0:
        raise ValueError("onset_seconds must be finite and non-negative")
    anchor_ref = validate_artifact_ref(anchor_source_ref)
    config = _config()
    factory = _signal_only_reader_factory if reader_factory is None else reader_factory
    reader = factory(str(source))
    try:
        labels = tuple(str(value).strip() for value in reader.getSignalLabels())
        classification = classify_signal_labels(labels)
        trusted_opaque_authority = None
        if classification["common_reference_compatible"] is True:
            reference_route = "header_observable_common_reference"
        elif opaque_reference_authority is not None:
            trusted_opaque_authority = validate_private_opaque_reference_authority(
                opaque_reference_authority
            )
            if (
                classification["montage_class"] != "unknown"
                or classification["classification_reason_codes"]
                != ["direct_electrode_reference_token_unobservable"]
                or classification["duplicate_direct_electrode_ids"]
            ):
                raise ValueError(
                    "real Stage-0 EDF is outside the authorized opaque reference profile"
                )
            reference_route = "protocol_authorized_opaque_common_reference"
        else:
            raise ValueError(
                "real Stage-0 has a mixed or unknown acquisition reference "
                "without an authorized opaque route"
            )
        observed_names, indices = _inventory(labels)
        rates = tuple(float(reader.getSampleFrequency(index)) for index in indices)
        if any(not math.isfinite(value) or value <= 0 for value in rates):
            raise ValueError("selected real Stage-0 rows have invalid sampling rates")
        source_sfreq = rates[0]
        if any(abs(value - source_sfreq) > 1e-9 for value in rates):
            raise ValueError("selected real Stage-0 rows have mixed sampling rates")
        if config.lowpass_hz >= 0.5 * source_sfreq:
            raise ValueError("source sampling rate is too low for frozen v29 filtering")
        sample_counts_raw = reader.getNSamples()
        sample_counts = tuple(int(sample_counts_raw[index]) for index in indices)
        if any(value <= 0 for value in sample_counts) or len(set(sample_counts)) != 1:
            raise ValueError("selected real Stage-0 rows have mixed sample counts")
        source_sample_count = sample_counts[0]
        units = tuple(str(reader.getPhysicalDimension(index)).strip() for index in indices)
        scales = np.asarray([_unit_scale(unit) for unit in units], dtype=np.float64)

        onset_source_sample = _half_up(requested_onset * source_sfreq)
        aligned_onset = onset_source_sample / source_sfreq
        start_seconds = aligned_onset + ANALYSIS_INTERVAL_SECONDS[0]
        stop_seconds = aligned_onset + ANALYSIS_INTERVAL_SECONDS[1]
        if start_seconds < config.warmup_sec:
            raise ValueError("event lacks the frozen 30-second real causal warmup")
        if stop_seconds > source_sample_count / source_sfreq + 1e-9:
            raise ValueError("event lacks the complete +48-second context")
        read_start = int(math.floor((start_seconds - config.warmup_sec) * source_sfreq + 1e-12))
        read_stop = min(
            source_sample_count,
            int(math.ceil(stop_seconds * source_sfreq - 1e-12)) + 1,
        )
        n_read = read_stop - read_start
        raw_rows = tuple(
            np.asarray(reader.readSignal(index, read_start, n_read), dtype=np.float64)
            for index in indices
        )
        if any(row.shape != (n_read,) or not np.isfinite(row).all() for row in raw_rows):
            raise ValueError("EDF reader returned an invalid Stage-0 signal payload")
        raw_volts = np.stack(raw_rows) * scales[:, None]
        processed, up, down, taps, latency = causal_bandpass_resample_channel_field(
            raw_volts,
            source_sfreq_hz=source_sfreq,
            config=config,
        )
        segment_start_seconds = read_start / source_sfreq
        crop_start = _half_up(
            (start_seconds - segment_start_seconds) * OUTPUT_SFREQ_HZ
            + latency * OUTPUT_SFREQ_HZ
        )
        crop_stop = crop_start + PARENT_SAMPLE_COUNT
        if crop_start < 0 or crop_stop > processed.shape[1]:
            raise ValueError("delayed Stage-0 crop lies outside processed payload")
        observed_window = np.ascontiguousarray(
            processed[:, crop_start:crop_stop], dtype=np.float32
        )
        if observed_window.shape != (len(observed_names), PARENT_SAMPLE_COUNT):
            raise ValueError("real Stage-0 processed window geometry drifted")

        parent = torch.zeros(
            (len(PARENT_ELECTRODES), PARENT_SAMPLE_COUNT), dtype=torch.float32
        )
        observed_mask = torch.zeros(len(PARENT_ELECTRODES), dtype=torch.bool)
        for source_index, electrode in enumerate(observed_names):
            target_index = PARENT_ELECTRODES.index(electrode)
            parent[target_index] = torch.from_numpy(observed_window[source_index])
            observed_mask[target_index] = True
        parent = parent.contiguous()
        parent_ref = build_raw_artifact_ref(
            canonical_tensor_bytes(parent),
            artifact_kind="canonical_signal",
            media_type="application/x-evisoz-canonical-tensor",
            payload_schema_version="evisoz_canonical_tensor_v1",
        )
        observability = build_montage_reference_observability_receipt(
            signal_labels=labels,
            source_signal_sha256=str(parent_ref["content_hash"]["sha256"]),
        )
        reference_authorization = None
        montage_reference_receipt: Mapping[str, object] = observability
        if trusted_opaque_authority is not None:
            reference_authorization = build_opaque_reference_event_authorization(
                trusted_opaque_authority,
                source_edf_sha256=source_edf_sha256,
                parent_signal_sha256=str(parent_ref["content_hash"]["sha256"]),
                montage_reference_observability_receipt=observability,
            )
            montage_reference_receipt = reference_authorization
        identity = build_event_identity(
            dataset_id=dataset_id,
            sample_id=sample_id,
            event_id=event_id,
            linkage_group_id=linkage_group_id,
            source_patient_sha256=source_patient_sha256,
            parent_signal_ref=parent_ref,
            anchor_source_ref=anchor_ref,
            anchor_quality=anchor_quality,
        )
        authoritative_v29 = None
        if bool(torch.all(observed_mask[: len(STANDARD_19)]).item()):
            authoritative_v29 = (
                parent[: len(STANDARD_19)]
                - parent[: len(STANDARD_19)].mean(dim=0, keepdim=True)
            ).contiguous()
        carrier = build_common_reference_event_carrier(
            parent_signal=parent,
            observed_mask=observed_mask,
            authoritative_v29_car19=authoritative_v29,
            event_identity=identity,
            parent_signal_ref=parent_ref,
            montage_reference_observability_receipt=montage_reference_receipt,
        )
        receipt = _preprocessing_receipt(
            source_edf_sha256=source_edf_sha256,
            source_labels_sha256=str(observability["signal_labels_sha256"]),
            reader_policy=str(
                getattr(reader, "canonical_reader_policy", type(reader).__name__)
            ),
            source_sfreq_hz=source_sfreq,
            source_sample_count=source_sample_count,
            read_start_sample=read_start,
            read_stop_sample=read_stop,
            requested_onset_seconds=requested_onset,
            source_aligned_onset_seconds=aligned_onset,
            observed_mask=observed_mask,
            resample_up=up,
            resample_down=down,
            resample_fir_taps=taps,
            resample_latency_seconds=latency,
            crop_start_output_sample=crop_start,
            parent_signal_ref=parent_ref,
            event_identity=identity,
            reference_route=reference_route,
            reference_authorization_ref=(
                None
                if reference_authorization is None
                else build_json_artifact_ref(
                    reference_authorization,
                    artifact_kind="opaque_common_reference_event_authorization",
                    payload_schema_version=(
                        OPAQUE_REFERENCE_EVENT_AUTHORIZATION_SCHEMA_VERSION
                    ),
                )
            ),
        )
        return RealStage0EventCarrier(
            carrier=carrier,
            preprocessing_receipt=receipt,
        )
    finally:
        if hasattr(reader, "close"):
            reader.close()


__all__ = [
    "ANALYSIS_INTERVAL_SECONDS",
    "OUTPUT_SFREQ_HZ",
    "PARENT_ELECTRODES",
    "PARENT_SAMPLE_COUNT",
    "REAL_STAGE0_PREPROCESSING_SCHEMA_VERSION",
    "RealStage0EventCarrier",
    "load_real_stage0_event_carrier",
]
