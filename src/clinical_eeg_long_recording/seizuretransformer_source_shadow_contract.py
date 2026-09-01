"""Fail-closed source contract for the 2025 SeizureTransformer challenger.

The public SeizureTransformer repository is useful enough to freeze the
architecture, full-record tiling and released post-processing semantics, but it
does *not* contain a checkpoint.  The README instead points to a Docker image
whose immutable manifest and embedded model have not been audited locally.
Moreover, the v3 paper and repository disagree on several preprocessing
details.  This module therefore deliberately does not load a model or transform
EEG.  It closes the parts that can be reproduced without guessing:

* pinned upstream source identity and an explicit activation gate;
* non-overlapping 60 s target tiles with complete final-tail coverage;
* concatenation/trim of one posterior value per target sample; and
* the repository's threshold/open/close/minimum-duration decoder.

The result is a research challenger *contract*, not a runnable detector and not
clinical evidence.  Annotation, spreadsheet, physician-label and clinical-text
inputs are absent by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import binary_closing, binary_opening


SEIZURETRANSFORMER_PROVIDER_ID = "seizuretransformer_timestep_shadow_v1"
SEIZURETRANSFORMER_UPSTREAM_COMMIT = (
    "cf83f5906a8aea88b60b56e4f962c5d6657c28f7"
)
SEIZURETRANSFORMER_ARXIV = "2504.00336v3"
SEIZURETRANSFORMER_CODE_LICENSE = "MIT"
SEIZURETRANSFORMER_CONTAINER_REFERENCE = (
    "docker.io/yujjio/seizure_transformer:latest"
)

SEIZURETRANSFORMER_SAMPLING_RATE_HZ = 256
SEIZURETRANSFORMER_TILE_SECONDS = 60
SEIZURETRANSFORMER_TILE_SAMPLES = (
    SEIZURETRANSFORMER_SAMPLING_RATE_HZ * SEIZURETRANSFORMER_TILE_SECONDS
)
SEIZURETRANSFORMER_RELEASED_THRESHOLD = 0.8
SEIZURETRANSFORMER_MORPHOLOGY_KERNEL_SAMPLES = 5
SEIZURETRANSFORMER_MINIMUM_EVENT_SECONDS = 2.0

# The explicit 18-derivation order in Appendix I.1 of arXiv:2504.00336v3.
# T3/T4/T5/T6 are retained because that is how the paper names the montage;
# a future activated adapter must bind these to canonical T7/T8/P7/P8 names.
SEIZURETRANSFORMER_PAPER_BIPOLAR_ORDER = (
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

_PINNED_SOURCE_SHA256 = {
    "LICENSE": "494e0c113cbaa2ee22bc3ede9d0b85e0b830df42446b2c34503db8fe37ea8041",
    "Dockerfile": "08c7224534e16cca656025d79bd09285718018899374e0fef92ae54f3b18f07c",
    "time_step_level/model.py": "0c3fd38a5350bb293e5337c26bb01c83945624b6eb8000da50e955e54174c7b2",
    "time_step_level/get_dataset.py": "0ab8d19853470250e262187bfba2725731f1b5bd5daa9c907d30678af3c7c4fe",
    "time_step_level/service/handle_data.py": "cbc088d9c5ba9b78b1457c461d1788419ae13c3b82c06442291e34cebcf6f2f0",
    "time_step_level/service/result.py": "33b4b626cf8f23d6f127354431e89b9e08a71e6fa9ddd01d719eac59c86fff98",
    "time_step_level/service/post_process.py": "e7eb3939d13c169efbbbbe8ff7a31f7597b77c3633ebcb6917aa89342ad01ebe",
    "time_step_level/eval_test.py": "accb259ca0f72193a21d1a620a98a15fbf6450442a103071592354d0df1acc73",
    "time_step_level/train_sd.py": "3b655f0b81dc9324f2a041173fe720cbd4556f4930c5e8e1a748ec28781ed101",
}

_PREPROCESSING_CONFLICTS = (
    "paper_appendix_lists_18_bipolar_inputs_but_architecture_table_lists_19_channels",
    "paper_v3_states_0_5_to_100_hz_but_repository_code_uses_0_5_to_120_hz",
    "paper_and_repository_do_not_bind_the_same_resample_normalize_filter_order",
    "repository_applies_state_reset_causal_IIR_filters_independently_per_tile",
    "repository_depends_on_epilepsy2bids_bipolar_mapping_without_a_frozen_version",
    "competition_Dockerfile_installs_an_unpublished_wu_2025_package_not_repository_code",
)

_EEG_ONLY_SCOPE = {
    "eeg_samples_used": True,
    "edf_signal_header_used": True,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "identity_fields_used": False,
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class SeizureTransformerTile:
    """One target-time tile; padding is never part of observed coverage."""

    tile_index: int
    target_start_sample: int
    target_end_sample: int
    model_input_samples: int
    right_padding_samples: int

    @property
    def observed_samples(self) -> int:
        return self.target_end_sample - self.target_start_sample


def audit_pinned_seizuretransformer_source(
    repository_root: Path | str,
) -> dict[str, Any]:
    """Verify the locally vendored public source without executing it."""

    root = Path(repository_root)
    files: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for relative_path, expected_sha256 in _PINNED_SOURCE_SHA256.items():
        path = root / relative_path
        if not path.is_file() or path.is_symlink():
            observed_sha256 = None
            mismatches.append(f"missing_or_nonregular:{relative_path}")
        else:
            observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed_sha256 != expected_sha256:
                mismatches.append(f"sha256_mismatch:{relative_path}")
        files.append(
            {
                "relative_path": relative_path,
                "expected_sha256": expected_sha256,
                "observed_sha256": observed_sha256,
                "matches": observed_sha256 == expected_sha256,
            }
        )

    receipt: dict[str, Any] = {
        "schema_version": "seizuretransformer_source_audit_v1",
        "provider_id": SEIZURETRANSFORMER_PROVIDER_ID,
        "upstream_commit": SEIZURETRANSFORMER_UPSTREAM_COMMIT,
        "source_code_license": SEIZURETRANSFORMER_CODE_LICENSE,
        "files": files,
        "source_identity_verified": not mismatches,
        "mismatches": mismatches,
        "checkpoint_present_in_public_repository": False,
        "container_reference": SEIZURETRANSFORMER_CONTAINER_REFERENCE,
        "container_digest_verified": False,
        "container_layers_audited": False,
        "preprocessing_conflicts": list(_PREPROCESSING_CONFLICTS),
        "eeg_only_scope": dict(_EEG_ONLY_SCOPE),
    }
    receipt["receipt_id"] = (
        "STSRCAUD-" + _canonical_sha256(receipt)[:24]
    )
    return receipt


def build_seizuretransformer_activation_receipt(
    *,
    source_audit: Mapping[str, Any],
    immutable_container_digest: str | None = None,
    container_layers_audited: bool = False,
    tensor_only_checkpoint_sha256: str | None = None,
    checkpoint_architecture_verified: bool = False,
    checkpoint_training_exposure_documented: bool = False,
    exact_preprocessing_profile_verified: bool = False,
) -> dict[str, Any]:
    """Build a fail-closed receipt for later model activation.

    A source hash alone can never activate inference.  The caller must resolve
    an immutable container, inspect its layers, extract a tensor-only checkpoint,
    verify exact architecture keys and bind the checkpoint to one unambiguous
    preprocessing profile.  Training exposure is kept separate: it blocks
    promotion/lockbox claims even when engineering inference becomes runnable.
    """

    digest_verified = (
        isinstance(immutable_container_digest, str)
        and immutable_container_digest.startswith("sha256:")
        and _is_sha256(immutable_container_digest.removeprefix("sha256:"))
    )
    checkpoint_verified = _is_sha256(tensor_only_checkpoint_sha256)
    source_verified = source_audit.get("source_identity_verified") is True

    engineering_activation_allowed = all(
        (
            source_verified,
            digest_verified,
            container_layers_audited is True,
            checkpoint_verified,
            checkpoint_architecture_verified is True,
            exact_preprocessing_profile_verified is True,
        )
    )
    benchmark_promotion_allowed = (
        engineering_activation_allowed
        and checkpoint_training_exposure_documented is True
    )
    blockers: list[str] = []
    if not source_verified:
        blockers.append("pinned_source_identity_not_verified")
    if not digest_verified:
        blockers.append("immutable_container_digest_not_verified")
    if container_layers_audited is not True:
        blockers.append("container_layers_not_audited")
    if not checkpoint_verified:
        blockers.append("tensor_only_checkpoint_not_verified")
    if checkpoint_architecture_verified is not True:
        blockers.append("checkpoint_architecture_not_verified")
    if exact_preprocessing_profile_verified is not True:
        blockers.append("checkpoint_preprocessing_profile_not_verified")
    if checkpoint_training_exposure_documented is not True:
        blockers.append("checkpoint_training_exposure_not_documented")

    receipt: dict[str, Any] = {
        "schema_version": "seizuretransformer_activation_receipt_v1",
        "provider_id": SEIZURETRANSFORMER_PROVIDER_ID,
        "source_audit_receipt_id": source_audit.get("receipt_id"),
        "immutable_container_digest": immutable_container_digest,
        "container_layers_audited": container_layers_audited is True,
        "tensor_only_checkpoint_sha256": tensor_only_checkpoint_sha256,
        "checkpoint_architecture_verified": (
            checkpoint_architecture_verified is True
        ),
        "checkpoint_training_exposure_documented": (
            checkpoint_training_exposure_documented is True
        ),
        "exact_preprocessing_profile_verified": (
            exact_preprocessing_profile_verified is True
        ),
        "engineering_activation_allowed": engineering_activation_allowed,
        "benchmark_promotion_allowed": benchmark_promotion_allowed,
        "blockers": blockers,
        "eeg_only_scope": dict(_EEG_ONLY_SCOPE),
    }
    receipt["receipt_id"] = "STACT-" + _canonical_sha256(receipt)[:24]
    return receipt


def plan_seizuretransformer_full_record_tiles(
    number_of_target_samples: int,
) -> tuple[SeizureTransformerTile, ...]:
    """Cover every observed sample once using released 60 s target tiles."""

    if isinstance(number_of_target_samples, bool) or not isinstance(
        number_of_target_samples, int
    ):
        raise TypeError("number_of_target_samples must be an integer")
    if number_of_target_samples <= 0:
        raise ValueError("number_of_target_samples must be positive")

    tiles: list[SeizureTransformerTile] = []
    for tile_index, start in enumerate(
        range(0, number_of_target_samples, SEIZURETRANSFORMER_TILE_SAMPLES)
    ):
        end = min(start + SEIZURETRANSFORMER_TILE_SAMPLES, number_of_target_samples)
        observed = end - start
        tiles.append(
            SeizureTransformerTile(
                tile_index=tile_index,
                target_start_sample=start,
                target_end_sample=end,
                model_input_samples=SEIZURETRANSFORMER_TILE_SAMPLES,
                right_padding_samples=SEIZURETRANSFORMER_TILE_SAMPLES - observed,
            )
        )

    if tiles[0].target_start_sample != 0:
        raise AssertionError("full-record tile plan does not start at sample zero")
    if tiles[-1].target_end_sample != number_of_target_samples:
        raise AssertionError("full-record tile plan leaves the final tail uncovered")
    for left, right in zip(tiles, tiles[1:]):
        if left.target_end_sample != right.target_start_sample:
            raise AssertionError("full-record tile plan has a gap or overlap")
    return tuple(tiles)


def stitch_seizuretransformer_tile_posteriors(
    tile_posteriors: Sequence[np.ndarray],
    tiles: Sequence[SeizureTransformerTile],
    *,
    number_of_target_samples: int,
) -> np.ndarray:
    """Concatenate model-length tile outputs and trim only non-observed tails."""

    if len(tile_posteriors) != len(tiles) or not tiles:
        raise ValueError("one posterior array is required for every planned tile")
    expected_tiles = plan_seizuretransformer_full_record_tiles(
        number_of_target_samples
    )
    if tuple(tiles) != expected_tiles:
        raise ValueError("tiles do not match the canonical full-record plan")

    pieces: list[np.ndarray] = []
    for tile, posterior in zip(tiles, tile_posteriors):
        array = np.asarray(posterior, dtype=np.float32)
        if array.shape != (tile.model_input_samples,):
            raise ValueError("every tile posterior must match the model input length")
        if not np.isfinite(array).all():
            raise ValueError("tile posterior contains non-finite values")
        if np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError("tile posterior values must lie in [0, 1]")
        pieces.append(array[: tile.observed_samples])

    stitched = np.concatenate(pieces).astype(np.float32, copy=False)
    if stitched.shape != (number_of_target_samples,):
        raise AssertionError("stitched posterior does not cover the full record")
    return stitched


def decode_seizuretransformer_posterior(
    posterior: np.ndarray,
    *,
    threshold: float = SEIZURETRANSFORMER_RELEASED_THRESHOLD,
    minimum_event_seconds: float = SEIZURETRANSFORMER_MINIMUM_EVENT_SECONDS,
) -> dict[str, Any]:
    """Replay the repository's native sample-domain event decoder.

    The default 0.8 threshold is a released development choice, not a locally
    calibrated operating point.  This function is therefore suitable for a
    frozen decoder grid or shadow benchmark only.
    """

    values = np.asarray(posterior, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("posterior must be a non-empty one-dimensional array")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("posterior values must be finite and lie in [0, 1]")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError("threshold must be numeric")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if not isinstance(minimum_event_seconds, (int, float)) or isinstance(
        minimum_event_seconds, bool
    ):
        raise TypeError("minimum_event_seconds must be numeric")
    minimum_event_seconds = float(minimum_event_seconds)
    if not math.isfinite(minimum_event_seconds) or minimum_event_seconds < 0.0:
        raise ValueError("minimum_event_seconds must be finite and non-negative")

    binary = values > threshold
    structure = np.ones(SEIZURETRANSFORMER_MORPHOLOGY_KERNEL_SAMPLES, dtype=bool)
    binary = binary_opening(binary, structure=structure)
    binary = binary_closing(binary, structure=structure)

    minimum_samples = int(
        minimum_event_seconds * SEIZURETRANSFORMER_SAMPLING_RATE_HZ
    )
    events_samples: list[tuple[int, int]] = []
    padded = np.pad(binary.astype(np.int8), (1, 1), constant_values=0)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    for start, end in zip(starts.tolist(), ends.tolist()):
        if end - start >= minimum_samples:
            events_samples.append((start, end))

    retained = np.zeros(values.shape, dtype=np.uint8)
    for start, end in events_samples:
        retained[start:end] = 1
    events_seconds = [
        {
            "onset_seconds": start / SEIZURETRANSFORMER_SAMPLING_RATE_HZ,
            "offset_seconds": end / SEIZURETRANSFORMER_SAMPLING_RATE_HZ,
            "duration_seconds": (
                (end - start) / SEIZURETRANSFORMER_SAMPLING_RATE_HZ
            ),
        }
        for start, end in events_samples
    ]

    receipt: dict[str, Any] = {
        "schema_version": "seizuretransformer_native_decoder_shadow_v1",
        "provider_id": SEIZURETRANSFORMER_PROVIDER_ID,
        "sampling_rate_hz": SEIZURETRANSFORMER_SAMPLING_RATE_HZ,
        "threshold": threshold,
        "threshold_status": "released_not_locally_calibrated",
        "morphology": {
            "order": ["binary_opening", "binary_closing"],
            "kernel_samples": SEIZURETRANSFORMER_MORPHOLOGY_KERNEL_SAMPLES,
            "kernel_seconds": (
                SEIZURETRANSFORMER_MORPHOLOGY_KERNEL_SAMPLES
                / SEIZURETRANSFORMER_SAMPLING_RATE_HZ
            ),
        },
        "minimum_event_seconds": minimum_event_seconds,
        "posterior_samples": int(values.size),
        "binary_mask_sha256": hashlib.sha256(retained.tobytes()).hexdigest(),
        "events": events_seconds,
        "eeg_only_scope": dict(_EEG_ONLY_SCOPE),
        "diagnostic_role": "detector_navigation_only_not_findings_or_soz_evidence",
    }
    receipt["receipt_id"] = "STDEC-" + _canonical_sha256(receipt)[:24]
    return receipt


def seizuretransformer_source_shadow_definition() -> dict[str, Any]:
    """Return the honest registry projection before immutable model audit."""

    return {
        "provider_id": SEIZURETRANSFORMER_PROVIDER_ID,
        "model_family": (
            "Large_EEG_U_Transformer_sequence_to_sequence_time_step_detector"
        ),
        "research_role": "shadow_continuous_comparator",
        "implementation_status": "source_contract_only_artifact_blocked",
        "qualification_status": "not_evaluated",
        "upstream_commit": SEIZURETRANSFORMER_UPSTREAM_COMMIT,
        "paper_status": "arxiv_preprint_v3_under_review",
        "paper_identifier": SEIZURETRANSFORMER_ARXIV,
        "weights_manifest_sha256": None,
        "checkpoint_loader_policy": "artifact_unavailable_no_load",
        "full_record_tiling_contract_available": True,
        "native_decoder_shadow_available": True,
        "exact_preprocessing_contract_available": False,
        "posterior_calibration_status": "not_locally_verified",
        "continuous_operating_point_status": "not_locally_verified",
        "reported_metrics_are_upstream_only": True,
        "eeg_signal_only": True,
        "edf_annotations_allowed": False,
        "excel_or_clinical_labels_allowed": False,
        "claimed_sota": False,
    }


__all__ = [
    "SEIZURETRANSFORMER_ARXIV",
    "SEIZURETRANSFORMER_CONTAINER_REFERENCE",
    "SEIZURETRANSFORMER_MINIMUM_EVENT_SECONDS",
    "SEIZURETRANSFORMER_PAPER_BIPOLAR_ORDER",
    "SEIZURETRANSFORMER_PROVIDER_ID",
    "SEIZURETRANSFORMER_RELEASED_THRESHOLD",
    "SEIZURETRANSFORMER_SAMPLING_RATE_HZ",
    "SEIZURETRANSFORMER_TILE_SAMPLES",
    "SEIZURETRANSFORMER_TILE_SECONDS",
    "SEIZURETRANSFORMER_UPSTREAM_COMMIT",
    "SeizureTransformerTile",
    "audit_pinned_seizuretransformer_source",
    "build_seizuretransformer_activation_receipt",
    "decode_seizuretransformer_posterior",
    "plan_seizuretransformer_full_record_tiles",
    "seizuretransformer_source_shadow_definition",
    "stitch_seizuretransformer_tile_posteriors",
]
