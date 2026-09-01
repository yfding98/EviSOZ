"""Static, fail-closed audit for an unsigned SeizureTransformer mirror.

The public SeizureTransformer source does not ship a checkpoint.  A third
party Hugging Face repository publishes a safetensors conversion that claims
to originate from ``wu_2025/model.pth`` in the challenge Docker image.  That
claim is useful for locating an artifact, but it is not an author signature or
an immutable link to the upstream container.  This module therefore verifies
only facts that can be replayed locally:

* fixed Hugging Face revision, byte length and file SHA-256;
* strict safetensors header/offset validation and finite F32 payload values;
* exact key/dtype/shape compatibility with the pinned public 19-channel model;
* separate gates for loading, prediction materialization, benchmark evidence,
  and an official-reproduction claim.

It never reads EEG, annotations, spreadsheets, physician labels, clinical
text, or reports.  A successful static audit is not a local accuracy result and
does not activate the detector.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping

import numpy as np


SEIZURETRANSFORMER_HF_REPOSITORY = "eugenehp/seizuretransformer"
SEIZURETRANSFORMER_HF_REVISION = (
    "92c2bffa632d967868a820ba3153f2828d72b496"
)
SEIZURETRANSFORMER_HF_FILENAME = "model.safetensors"
SEIZURETRANSFORMER_HF_SHA256 = (
    "2cdc841001a0fbcdf1dfcbb02b3a26fa7af14002e01ebf9815fa09c82be06f61"
)
SEIZURETRANSFORMER_HF_SIZE_BYTES = 176_373_916
SEIZURETRANSFORMER_HF_TENSOR_COUNT = 215
SEIZURETRANSFORMER_HF_F32_NUMEL = 44_087_617
SEIZURETRANSFORMER_HF_HEADER_BYTES = 23_440
SEIZURETRANSFORMER_HF_HEADER_SHA256 = (
    "e94b1acadcbddbd54754220bd26f0dca7d6b25b7a84b7f54ede03033283cdef7"
)
SEIZURETRANSFORMER_HF_KEY_DTYPE_SHAPE_SHA256 = (
    "2e8906b6aabe5bdf4231c639b3d4385b6dd7a082aaa73854b506f58338ac116e"
)
SEIZURETRANSFORMER_PUBLIC_SOURCE_COMMIT = (
    "cf83f5906a8aea88b60b56e4f962c5d6657c28f7"
)
SEIZURETRANSFORMER_PUBLIC_SOURCE_STATE_TENSOR_COUNT = 229
SEIZURETRANSFORMER_OMITTED_NUM_BATCHES_TRACKED = 14

SEIZURETRANSFORMER_SOURCE_19_CHANNEL_ORDER = (
    "FP1",
    "F3",
    "C3",
    "P3",
    "O1",
    "F7",
    "T3",
    "T5",
    "FZ",
    "CZ",
    "PZ",
    "FP2",
    "F4",
    "C4",
    "P4",
    "O2",
    "F8",
    "T4",
    "T6",
)

SEIZURETRANSFORMER_PAPER_18_BIPOLAR_ORDER = (
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FZ-CZ",
    "CZ-PZ",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
)

SEIZURETRANSFORMER_DEFAULT_ARTIFACT_PATH = Path(
    "models/seizuretransformer_third_party_research_only/"
    "sha256-"
    + SEIZURETRANSFORMER_HF_SHA256
    + "/model.safetensors"
)

_DTYPE_NBYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

_EEG_ONLY_SCOPE = {
    "eeg_samples_used": False,
    "edf_signal_header_used": False,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "report_text_used": False,
}


class SafeTensorAuditError(ValueError):
    """Raised when a safetensors container violates the static contract."""


@dataclass(frozen=True)
class SafeTensorHeaderAudit:
    file_size_bytes: int
    header_bytes: int
    header_sha256: str
    tensor_count: int
    tensor_numel: int
    data_bytes: int
    dtype_counts: Mapping[str, int]
    metadata: Mapping[str, str]
    key_dtype_shape_sha256: str
    offsets_contiguous: bool
    data_section_start_bytes: int


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stream_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SafeTensorAuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_strict_header_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise SafeTensorAuditError("truncated safetensors length prefix")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes < 2 or header_bytes > 64 * 1024 * 1024:
            raise SafeTensorAuditError("implausible safetensors header length")
        if 8 + header_bytes > path.stat().st_size:
            raise SafeTensorAuditError("safetensors header exceeds file length")
        raw_header = stream.read(header_bytes)
        if len(raw_header) != header_bytes:
            raise SafeTensorAuditError("truncated safetensors header")
    try:
        header = json.loads(
            raw_header.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SafeTensorAuditError("invalid safetensors JSON header") from error
    if not isinstance(header, dict):
        raise SafeTensorAuditError("safetensors header must be a JSON object")
    return raw_header, header


def inspect_safetensors_header(path: Path | str) -> SafeTensorHeaderAudit:
    """Parse and validate a safetensors header without deserializing tensors."""

    artifact = Path(path)
    if not artifact.is_file() or artifact.is_symlink():
        raise SafeTensorAuditError("artifact must be a regular non-symlink file")
    file_size = artifact.stat().st_size
    if file_size < 10:
        raise SafeTensorAuditError("artifact is too short for safetensors")

    raw_header, header = _read_strict_header_object(artifact)
    header_bytes = len(raw_header)

    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise SafeTensorAuditError("safetensors metadata must map strings to strings")
    if not header:
        raise SafeTensorAuditError("safetensors artifact contains no tensors")

    descriptors: list[dict[str, Any]] = []
    intervals: list[tuple[int, int, str]] = []
    dtype_counts: dict[str, int] = {}
    total_numel = 0
    for key in sorted(header):
        descriptor = header[key]
        if not isinstance(key, str) or not key or key == "__metadata__":
            raise SafeTensorAuditError("invalid tensor key")
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "dtype",
            "shape",
            "data_offsets",
        }:
            raise SafeTensorAuditError(f"invalid tensor descriptor: {key}")
        dtype = descriptor["dtype"]
        shape = descriptor["shape"]
        offsets = descriptor["data_offsets"]
        if dtype not in _DTYPE_NBYTES:
            raise SafeTensorAuditError(f"unsupported tensor dtype: {dtype}")
        if not isinstance(shape, list) or not all(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0
            for size in shape
        ):
            raise SafeTensorAuditError(f"invalid tensor shape: {key}")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(
            isinstance(offset, int)
            and not isinstance(offset, bool)
            and offset >= 0
            for offset in offsets
        ):
            raise SafeTensorAuditError(f"invalid tensor offsets: {key}")
        start, end = offsets
        if end < start:
            raise SafeTensorAuditError(f"reversed tensor offsets: {key}")
        numel = math.prod(shape)
        if end - start != numel * _DTYPE_NBYTES[dtype]:
            raise SafeTensorAuditError(f"tensor byte length does not match shape: {key}")
        descriptors.append({"key": key, "dtype": dtype, "shape": shape})
        intervals.append((start, end, key))
        total_numel += numel
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

    intervals.sort()
    expected_start = 0
    offsets_contiguous = True
    for start, end, key in intervals:
        if start != expected_start:
            offsets_contiguous = False
            relation = "overlap" if start < expected_start else "gap"
            raise SafeTensorAuditError(f"tensor data offsets contain a {relation}: {key}")
        expected_start = end
    data_start = 8 + header_bytes
    if data_start + expected_start != file_size:
        raise SafeTensorAuditError("tensor data does not consume the complete file")

    return SafeTensorHeaderAudit(
        file_size_bytes=file_size,
        header_bytes=header_bytes,
        header_sha256=hashlib.sha256(raw_header).hexdigest(),
        tensor_count=len(descriptors),
        tensor_numel=total_numel,
        data_bytes=expected_start,
        dtype_counts=dict(sorted(dtype_counts.items())),
        metadata=dict(sorted(metadata.items())),
        key_dtype_shape_sha256=_canonical_sha256(descriptors),
        offsets_contiguous=offsets_contiguous,
        data_section_start_bytes=data_start,
    )


def _audit_fixed_f32_payload_finite(
    path: Path,
    header: SafeTensorHeaderAudit,
    *,
    chunk_numel: int = 2_000_000,
) -> tuple[bool, float | None, float | None]:
    if header.dtype_counts != {"F32": header.tensor_count}:
        return False, None, None
    values = np.memmap(
        path,
        mode="r",
        dtype="<f4",
        offset=header.data_section_start_bytes,
        shape=(header.tensor_numel,),
    )
    global_min: float | None = None
    global_max: float | None = None
    finite = True
    for start in range(0, header.tensor_numel, chunk_numel):
        block = np.asarray(values[start : start + chunk_numel])
        if not np.isfinite(block).all():
            finite = False
            break
        if block.size:
            block_min = float(block.min())
            block_max = float(block.max())
            global_min = block_min if global_min is None else min(global_min, block_min)
            global_max = block_max if global_max is None else max(global_max, block_max)
    del values
    return finite, global_min, global_max


def audit_seizuretransformer_third_party_artifact(
    path: Path | str = SEIZURETRANSFORMER_DEFAULT_ARTIFACT_PATH,
    *,
    verify_finite_payload: bool = True,
) -> dict[str, Any]:
    """Return replayable static facts; malformed/missing artifacts fail closed."""

    artifact = Path(path)
    blockers: list[str] = []
    try:
        header = inspect_safetensors_header(artifact)
        observed_sha256 = _stream_sha256(artifact)
    except (OSError, SafeTensorAuditError) as error:
        receipt: dict[str, Any] = {
            "schema_version": "seizuretransformer_third_party_artifact_audit_v1",
            "artifact_repository": SEIZURETRANSFORMER_HF_REPOSITORY,
            "artifact_revision": SEIZURETRANSFORMER_HF_REVISION,
            "artifact_filename": SEIZURETRANSFORMER_HF_FILENAME,
            "artifact_path": str(artifact),
            "static_artifact_verified": False,
            "public_source_architecture_header_compatible": False,
            "parse_error": f"{type(error).__name__}:{error}",
            "blockers": ["artifact_missing_or_malformed"],
            "eeg_only_scope": dict(_EEG_ONLY_SCOPE),
        }
        receipt["receipt_id"] = "ST3PAUD-" + _canonical_sha256(receipt)[:24]
        return receipt

    checks = {
        "file_sha256_matches": observed_sha256 == SEIZURETRANSFORMER_HF_SHA256,
        "file_size_matches": header.file_size_bytes
        == SEIZURETRANSFORMER_HF_SIZE_BYTES,
        "header_bytes_match": header.header_bytes
        == SEIZURETRANSFORMER_HF_HEADER_BYTES,
        "header_sha256_matches": header.header_sha256
        == SEIZURETRANSFORMER_HF_HEADER_SHA256,
        "tensor_count_matches": header.tensor_count
        == SEIZURETRANSFORMER_HF_TENSOR_COUNT,
        "f32_tensor_count_matches": header.dtype_counts
        == {"F32": SEIZURETRANSFORMER_HF_TENSOR_COUNT},
        "f32_numel_matches": header.tensor_numel
        == SEIZURETRANSFORMER_HF_F32_NUMEL,
        "key_dtype_shape_matches_public_source": header.key_dtype_shape_sha256
        == SEIZURETRANSFORMER_HF_KEY_DTYPE_SHAPE_SHA256,
        "metadata_empty": not header.metadata,
        "offsets_contiguous": header.offsets_contiguous,
    }
    finite: bool | None = None
    global_min: float | None = None
    global_max: float | None = None
    if verify_finite_payload:
        finite, global_min, global_max = _audit_fixed_f32_payload_finite(
            artifact,
            header,
        )
        checks["all_f32_payload_values_finite"] = finite
    else:
        blockers.append("finite_payload_check_not_run")
    for check_name, passed in checks.items():
        if passed is not True:
            blockers.append(f"failed_static_check:{check_name}")

    static_verified = not blockers
    architecture_compatible = (
        checks["key_dtype_shape_matches_public_source"] is True
        and checks["tensor_count_matches"] is True
        and checks["f32_numel_matches"] is True
    )
    receipt = {
        "schema_version": "seizuretransformer_third_party_artifact_audit_v1",
        "artifact_repository": SEIZURETRANSFORMER_HF_REPOSITORY,
        "artifact_revision": SEIZURETRANSFORMER_HF_REVISION,
        "artifact_filename": SEIZURETRANSFORMER_HF_FILENAME,
        "artifact_path": str(artifact),
        "expected_sha256": SEIZURETRANSFORMER_HF_SHA256,
        "observed_sha256": observed_sha256,
        "file_size_bytes": header.file_size_bytes,
        "header": {
            "header_bytes": header.header_bytes,
            "header_sha256": header.header_sha256,
            "tensor_count": header.tensor_count,
            "tensor_numel": header.tensor_numel,
            "data_bytes": header.data_bytes,
            "dtype_counts": dict(header.dtype_counts),
            "metadata": dict(header.metadata),
            "key_dtype_shape_sha256": header.key_dtype_shape_sha256,
            "offsets_contiguous": header.offsets_contiguous,
        },
        "payload_finite_audit": {
            "run": verify_finite_payload,
            "all_finite": finite,
            "global_min": global_min,
            "global_max": global_max,
        },
        "checks": checks,
        "static_artifact_verified": static_verified,
        "public_source_commit": SEIZURETRANSFORMER_PUBLIC_SOURCE_COMMIT,
        "public_source_architecture": {
            "in_channels": 19,
            "in_samples": 15_360,
            "dim_feedforward": 2_048,
            "num_layers": 8,
            "num_heads": 4,
            "state_tensor_count": (
                SEIZURETRANSFORMER_PUBLIC_SOURCE_STATE_TENSOR_COUNT
            ),
            "artifact_tensor_count": SEIZURETRANSFORMER_HF_TENSOR_COUNT,
            "omitted_buffers": {
                "suffix": ".num_batches_tracked",
                "count": SEIZURETRANSFORMER_OMITTED_NUM_BATCHES_TRACKED,
                "effect": "PyTorch_BatchNorm_compatibility_buffers_only",
            },
        },
        "public_source_architecture_header_compatible": architecture_compatible,
        "distribution_terms": {
            "license_identifier": "other",
            "research_or_education_only": True,
            "non_commercial_without_written_permission": True,
            "clinical_use_prohibited": True,
            "upstream_rights_prevail": True,
        },
        "provenance": {
            "uploader_claimed_container": "yujjio/seizure_transformer",
            "uploader_claimed_checkpoint_path": "wu_2025/model.pth",
            "uploader_is_upstream_author_verified": False,
            "immutable_upstream_container_digest_verified": False,
            "original_model_pth_sha256_verified": False,
            "conversion_log_or_signature_verified": False,
            "official_checkpoint_provenance_verified": False,
        },
        "claim_status": (
            "static_unsigned_third_party_artifact_only_not_official_reproduction_"
            "not_local_accuracy"
        ),
        "blockers": blockers,
        "eeg_only_scope": dict(_EEG_ONLY_SCOPE),
    }
    receipt["receipt_id"] = "ST3PAUD-" + _canonical_sha256(receipt)[:24]
    return receipt


def compare_artifact_header_to_pinned_public_source(
    artifact_path: Path | str = SEIZURETRANSFORMER_DEFAULT_ARTIFACT_PATH,
    *,
    public_source_root: Path | str = Path("third_party/SeizureTransformer"),
    verify_cpu_state_dict_load: bool = False,
) -> dict[str, Any]:
    """Compare every artifact key/dtype/shape to the hash-pinned source model."""

    from .seizuretransformer_source_shadow_contract import (
        audit_pinned_seizuretransformer_source,
    )

    artifact = Path(artifact_path)
    source_root = Path(public_source_root)
    source_audit = audit_pinned_seizuretransformer_source(source_root)
    if source_audit.get("source_identity_verified") is not True:
        raise SafeTensorAuditError("pinned public source identity is not verified")
    inspect_safetensors_header(artifact)
    _, header = _read_strict_header_object(artifact)
    header.pop("__metadata__", None)
    artifact_spec = {
        key: (descriptor["dtype"], tuple(descriptor["shape"]))
        for key, descriptor in header.items()
    }

    model_path = source_root / "time_step_level" / "model.py"
    module_spec = importlib.util.spec_from_file_location(
        "_pinned_seizuretransformer_model_for_static_audit",
        model_path,
    )
    if module_spec is None or module_spec.loader is None:
        raise SafeTensorAuditError("unable to import pinned public architecture")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = module.SeizureTransformer(
        in_channels=19,
        in_samples=15_360,
        dim_feedforward=2_048,
        num_layers=8,
        num_heads=4,
        drop_rate=0.1,
    )
    source_state = model.state_dict()
    torch_dtype_to_safe = {
        "torch.float32": "F32",
        "torch.float64": "F64",
        "torch.float16": "F16",
        "torch.bfloat16": "BF16",
        "torch.int64": "I64",
        "torch.int32": "I32",
        "torch.int16": "I16",
        "torch.int8": "I8",
        "torch.uint8": "U8",
        "torch.bool": "BOOL",
    }
    source_spec = {
        key: (torch_dtype_to_safe.get(str(value.dtype), str(value.dtype)), tuple(value.shape))
        for key, value in source_state.items()
    }
    artifact_keys = set(artifact_spec)
    source_keys = set(source_spec)
    missing = sorted(source_keys - artifact_keys)
    unexpected = sorted(artifact_keys - source_keys)
    dtype_mismatches = [
        {
            "key": key,
            "artifact": artifact_spec[key][0],
            "source": source_spec[key][0],
        }
        for key in sorted(artifact_keys & source_keys)
        if artifact_spec[key][0] != source_spec[key][0]
    ]
    shape_mismatches = [
        {
            "key": key,
            "artifact": list(artifact_spec[key][1]),
            "source": list(source_spec[key][1]),
        }
        for key in sorted(artifact_keys & source_keys)
        if artifact_spec[key][1] != source_spec[key][1]
    ]
    missing_only_tracking_counters = (
        len(missing) == SEIZURETRANSFORMER_OMITTED_NUM_BATCHES_TRACKED
        and all(key.endswith(".num_batches_tracked") for key in missing)
    )
    header_load_compatible = all(
        (
            missing_only_tracking_counters,
            not unexpected,
            not dtype_mismatches,
            not shape_mismatches,
        )
    )

    torch_load_missing: list[str] | None = None
    torch_load_unexpected: list[str] | None = None
    if verify_cpu_state_dict_load:
        from safetensors.torch import load_file

        state = load_file(str(artifact), device="cpu")
        load_result = model.load_state_dict(state, strict=False)
        torch_load_missing = list(load_result.missing_keys)
        torch_load_unexpected = list(load_result.unexpected_keys)
        model.eval()

    source_without_tracking = [
        {"key": key, "dtype": dtype, "shape": list(shape)}
        for key, (dtype, shape) in sorted(source_spec.items())
        if not key.endswith(".num_batches_tracked")
    ]
    receipt: dict[str, Any] = {
        "schema_version": "seizuretransformer_artifact_source_compatibility_v1",
        "artifact_sha256": _stream_sha256(artifact),
        "source_audit_receipt_id": source_audit.get("receipt_id"),
        "source_commit": SEIZURETRANSFORMER_PUBLIC_SOURCE_COMMIT,
        "source_state_tensor_count": len(source_spec),
        "artifact_tensor_count": len(artifact_spec),
        "common_tensor_count": len(source_keys & artifact_keys),
        "missing_source_keys": missing,
        "unexpected_artifact_keys": unexpected,
        "dtype_mismatches": dtype_mismatches,
        "shape_mismatches": shape_mismatches,
        "missing_only_batchnorm_tracking_counters": missing_only_tracking_counters,
        "source_without_tracking_key_dtype_shape_sha256": _canonical_sha256(
            source_without_tracking
        ),
        "artifact_key_dtype_shape_sha256": _canonical_sha256(
            [
                {"key": key, "dtype": dtype, "shape": list(shape)}
                for key, (dtype, shape) in sorted(artifact_spec.items())
            ]
        ),
        "header_load_compatible": header_load_compatible,
        "cpu_state_dict_load": {
            "run": verify_cpu_state_dict_load,
            "missing_keys": torch_load_missing,
            "unexpected_keys": torch_load_unexpected,
            "forward_inference_run": False,
        },
        "claim_guard": "load_compatibility_is_not_model_provenance_or_accuracy",
        "eeg_only_scope": dict(_EEG_ONLY_SCOPE),
    }
    receipt["receipt_id"] = "STSRCCOMP-" + _canonical_sha256(receipt)[:24]
    return receipt


def seizuretransformer_native_preprocessing_contract() -> dict[str, Any]:
    """Expose known alternatives without guessing which one produced the weights."""

    contract: dict[str, Any] = {
        "schema_version": "seizuretransformer_native_preprocessing_contract_v1",
        "checkpoint_tensor_binding": {
            "input_channels_from_first_convolution": 19,
            "input_samples_from_public_architecture_and_mirror_config": 15_360,
            "nominal_sampling_rate_hz": 256,
            "nominal_tile_seconds": 60,
            "status": "architecture_shape_verified",
        },
        "candidate_profiles": [
            {
                "profile_id": "public_source_get_data_19_referential",
                "input_channel_count": 19,
                "channel_order": list(SEIZURETRANSFORMER_SOURCE_19_CHANNEL_ORDER),
                "montage": "nineteen_selected_referential_signal_labels",
                "channel_match": "case_insensitive_substring_first_match",
                "missing_channel_policy": "assert_fail",
                "whole_record_transforms_before_tiling": [
                    "per_channel_mean_std_zscore_over_complete_record",
                    "scipy_signal_resample_fourier_to_256_hz_if_needed",
                ],
                "per_tile_transforms_with_state_reset": [
                    "causal_order3_butterworth_bandpass_0_5_to_120_hz",
                    "causal_iir_notch_1_hz_Q30",
                    "causal_iir_notch_60_hz_Q30",
                ],
                "checkpoint_shape_compatible": True,
                "checkpoint_provenance_binding_verified": False,
            },
            {
                "profile_id": "public_source_get_data_18_bipolar",
                "input_channel_count": 18,
                "channel_order": "delegated_to_unpinned_epilepsy2bids_rereference",
                "montage": "longitudinal_bipolar",
                "whole_record_transforms_before_tiling": [
                    "per_channel_mean_std_zscore_over_complete_record",
                    "scipy_signal_resample_fourier_to_256_hz_if_needed",
                ],
                "per_tile_transforms_with_state_reset": [
                    "causal_order3_butterworth_bandpass_0_5_to_120_hz",
                    "causal_iir_notch_1_hz_Q30",
                    "causal_iir_notch_60_hz_Q30",
                ],
                "checkpoint_shape_compatible": False,
                "checkpoint_provenance_binding_verified": False,
            },
            {
                "profile_id": "paper_appendix_18_bipolar",
                "input_channel_count": 18,
                "channel_order": list(SEIZURETRANSFORMER_PAPER_18_BIPOLAR_ORDER),
                "montage": "longitudinal_bipolar",
                "paper_bandpass_hz": [0.5, 100.0],
                "checkpoint_shape_compatible": False,
                "checkpoint_provenance_binding_verified": False,
            },
            {
                "profile_id": "challenge_container_wu_2025",
                "input_channel_count": None,
                "channel_order": None,
                "montage": None,
                "transforms": None,
                "checkpoint_shape_compatible": None,
                "checkpoint_provenance_binding_verified": False,
                "reason": "container_package_and_immutable_manifest_not_audited",
            },
        ],
        "released_inference_geometry": {
            "target_tile_stride_samples": 15_360,
            "target_tiles_overlap": False,
            "final_tile_right_zero_padding": True,
            "posterior_tail_padding_trimmed_to_observed_samples": True,
            "released_decoder": {
                "threshold_strict_greater_than": 0.8,
                "binary_opening_kernel_samples": 5,
                "binary_closing_kernel_samples": 5,
                "minimum_event_seconds": 2.0,
            },
        },
        "activation_profile_id": None,
        "contract_complete": False,
        "blockers": [
            "unsigned_mirror_not_bound_to_original_model_pth_hash",
            "immutable_challenge_container_manifest_not_verified",
            "wu_2025_package_and_native_preprocessing_not_available",
            "paper_and_public_source_disagree_on_18_or_19_channels",
            "paper_and_public_source_disagree_on_100_or_120_hz_high_cut",
            "exact_montage_order_aliases_units_and_missing_channel_policy_not_bound",
            "transform_order_and_filter_state_not_confirmed_against_container",
        ],
        "parity_requirements": [
            "resolve_and_hash_immutable_container_manifest_and_layers",
            "extract_wu_2025_without_executing_pickle_or_private_data",
            "bind_original_model_pth_hash_to_converted_safetensors",
            "export_exact_channel_alias_montage_unit_resample_normalize_filter_profile",
            "run_synthetic_impulse_step_sine_and_tail_padding_tensor_parity",
            "freeze_profile_hash_before_any_TUSZ_prediction_materialization",
        ],
        "findings_noninterference": (
            "checkpoint_native_detector_tensor_is_navigation_only_and_must_not_"
            "replace_physical_unit_findings_views"
        ),
        "eeg_only_scope": dict(_EEG_ONLY_SCOPE),
    }
    contract["contract_id"] = "STPREP-" + _canonical_sha256(contract)[:24]
    return contract


def build_seizuretransformer_third_party_activation_gate(
    *,
    artifact_audit: Mapping[str, Any],
    research_only_terms_acknowledged: bool = False,
    exact_checkpoint_native_preprocessing_verified: bool = False,
    author_provenance_verified: bool = False,
    immutable_upstream_container_digest: str | None = None,
    original_checkpoint_sha256: str | None = None,
    training_exposure_documented: bool = False,
    full_dev_prediction_inventory_frozen: bool = False,
    full_dev_postfreeze_scored: bool = False,
    local_efficiency_profile_complete: bool = False,
    qualified_operating_point_observed: bool = False,
) -> dict[str, Any]:
    """Keep artifact loading, benchmarking and promotion as separate gates."""

    def is_sha256(value: str | None, *, prefix: bool = False) -> bool:
        if not isinstance(value, str):
            return False
        candidate = value.removeprefix("sha256:") if prefix else value
        return (
            len(candidate) == 64
            and all(character in "0123456789abcdef" for character in candidate)
        )

    static_verified = artifact_audit.get("static_artifact_verified") is True
    architecture_compatible = (
        artifact_audit.get("public_source_architecture_header_compatible") is True
    )
    terms_ok = research_only_terms_acknowledged is True
    shadow_load_allowed = static_verified and architecture_compatible and terms_ok
    prediction_materialization_allowed = (
        shadow_load_allowed and exact_checkpoint_native_preprocessing_verified is True
    )
    immutable_digest_verified = is_sha256(
        immutable_upstream_container_digest,
        prefix=True,
    )
    original_checkpoint_verified = is_sha256(original_checkpoint_sha256)
    official_reproduction_claim_allowed = all(
        (
            prediction_materialization_allowed,
            author_provenance_verified is True,
            immutable_digest_verified,
            original_checkpoint_verified,
            training_exposure_documented is True,
        )
    )
    local_benchmark_evidence_available = all(
        (
            prediction_materialization_allowed,
            full_dev_prediction_inventory_frozen is True,
            full_dev_postfreeze_scored is True,
            local_efficiency_profile_complete is True,
        )
    )
    accuracy_primary_promotion_allowed = all(
        (
            local_benchmark_evidence_available,
            qualified_operating_point_observed is True,
            training_exposure_documented is True,
        )
    )

    blockers: list[str] = []
    conditions = (
        (static_verified, "static_artifact_not_verified"),
        (architecture_compatible, "public_source_architecture_not_compatible"),
        (terms_ok, "research_only_terms_not_acknowledged"),
        (
            exact_checkpoint_native_preprocessing_verified is True,
            "checkpoint_native_preprocessing_not_verified",
        ),
        (author_provenance_verified is True, "author_provenance_not_verified"),
        (immutable_digest_verified, "immutable_upstream_container_digest_missing"),
        (original_checkpoint_verified, "original_checkpoint_sha256_missing"),
        (training_exposure_documented is True, "training_exposure_not_documented"),
        (
            full_dev_prediction_inventory_frozen is True,
            "complete_dev_prediction_inventory_not_frozen",
        ),
        (full_dev_postfreeze_scored is True, "complete_dev_postfreeze_score_missing"),
        (
            local_efficiency_profile_complete is True,
            "local_end_to_end_efficiency_profile_missing",
        ),
        (
            qualified_operating_point_observed is True,
            "no_qualified_operating_point_observed",
        ),
    )
    blockers.extend(message for passed, message in conditions if not passed)

    receipt: dict[str, Any] = {
        "schema_version": "seizuretransformer_third_party_activation_gate_v1",
        "artifact_audit_receipt_id": artifact_audit.get("receipt_id"),
        "research_shadow_model_load_allowed": shadow_load_allowed,
        "prediction_first_materialization_allowed": (
            prediction_materialization_allowed
        ),
        "official_reproduction_claim_allowed": (
            official_reproduction_claim_allowed
        ),
        "local_benchmark_evidence_available": local_benchmark_evidence_available,
        "accuracy_primary_promotion_allowed": accuracy_primary_promotion_allowed,
        "blockers": blockers,
        "claim_guard": (
            "architecture_loadability_or_static_compatibility_is_not_accuracy"
        ),
        "eeg_only_scope": dict(_EEG_ONLY_SCOPE),
    }
    receipt["receipt_id"] = "ST3PGATE-" + _canonical_sha256(receipt)[:24]
    return receipt
