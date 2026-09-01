"""Executable clean-room EventNet registry for independent EN19 and EN17.

This module freezes the project-side implementation needed to retrain the
MIT-licensed EventNet 1-D U-Net from random initialization.  The released
``eventnet_full_record_adapter.py`` remains a separate, immutable public-weight
comparison lane; this module neither imports its checkpoint nor changes that
adapter.

Two models are deliberately distinct:

* EN19 consumes the exact 19-electrode axis order used by the MIT release.
* EN17 consumes the same order with FZ and PZ removed, projected *directly*
  from an externally verified referential-volts carrier.  It is never made by
  zero filling, interpolating, or deleting axes from an EN19 model tensor.

Only EEG samples, acquisition clock/reference provenance, and EEG-only QC may
enter the provider transform.  The low-level target/loss/sampling functions in
this module are deliberately named ``*_pure_primitive``: they are useful for
testing the published mathematics, but they are *not* training authority.
Formal training first requires an opaque fold-phase authority whose reference
bytes were replayed, and then an opaque target bundle bound to the exact record,
transform, unpadded tile, event inventory, and fold-owned patient key.  There is
no annotation, spreadsheet, physician-text, identity-feature, or clinical-
report input to model forward.

The architecture, transform, target/loss mathematics, tiling, deterministic
sampling primitives, shared opaque phase adapter, and provider-neutral
target-blind variant-roster API are CPU executable.  No real complete
pre-reference eligibility Cartesian, epoch executor, or byte-replayed
checkpoint-admission artifact has yet been materialized.
Importing the module does not query a GPU, load a
checkpoint, open an EDF/reference sidecar, or contact a model service.  No
checkpoint, OOF prediction, operating point, accuracy, or clinical claim is
materialized by this registry.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
from scipy.signal import resample_poly
import torch
from torch import Tensor, nn
import torch.nn.functional as torch_functional

from .detector_fold_reference_authority_v1 import (
    ValidatedDetectorFoldReferencePhaseAuthorityV1,
    require_validated_detector_fold_reference_phase_authority_v1,
)
from .detector_channel_support_router_v1 import (
    detector_channel_support_policy_receipt,
    route_detector_channel_support,
)
from .detector_signal_lineage_authority_v1 import (
    ValidatedDetectorSignalLineageAuthority,
    require_validated_detector_signal_lineage_authority,
    verify_provider_referential_payload,
)


SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_eventnet_cleanroom_transform_trainer_registry_v1"
)
REGISTRY_ID: Final[str] = (
    "CLINICAL-EEG-EVENTNET-CLEANROOM-TRANSFORM-TRAINER-V1-20260824"
)
PROVIDER_ID: Final[str] = "eventnet_cleanroom_retrained_v1"
EN19_VARIANT_ID: Final[str] = "eventnet_en19_cleanroom_v1"
EN17_VARIANT_ID: Final[str] = "eventnet_en17_common_support_cleanroom_v1"

UPSTREAM_REPOSITORY: Final[str] = "https://github.com/esl-epfl/eventnet_2024"
UPSTREAM_COMMIT: Final[str] = "d13866820f436b1d767ef7f27a5419a7735efa5b"
UPSTREAM_ARCHITECTURE_GIT_BLOB: Final[str] = (
    "58796bfd238f03cdd07f9909caa76269589d3230"
)
# SHA-256 of eventnet/src/eventnet/architecture.py at the commit above.
UPSTREAM_ARCHITECTURE_SHA256: Final[str] = (
    "f92a176caeac4126ca1230cc10ab0c98d62dabe7e34832f1b3f407d7d7b9f18c"
)

TARGET_FS_HZ: Final[int] = 256
TARGET_TILE_SECONDS: Final[int] = 120
TARGET_TILE_SAMPLES: Final[int] = TARGET_FS_HZ * TARGET_TILE_SECONDS
CONTEXT_SAMPLES_PER_SIDE: Final[int] = 128
MODEL_INPUT_SAMPLES: Final[int] = (
    TARGET_TILE_SAMPLES + 2 * CONTEXT_SAMPLES_PER_SIDE
)
MAXIMUM_DURATION_SECONDS: Final[int] = 300
FOCAL_ALPHA_C: Final[float] = 0.1
FOCAL_ALPHA: Final[float] = 2.0
FOCAL_BETA: Final[float] = 4.0
DURATION_LOSS_WEIGHT: Final[float] = 5.0
CONFIG_RELATIVE_PATH: Final[str] = (
    "configs/clinical_eeg_eventnet_cleanroom_transform_trainer_registry_v1.json"
)

# Exact epilepsy2bids.Eeg.ELECTRODES_10_20 order used by the MIT release.
EN19_CHANNEL_ORDER: Final[tuple[str, ...]] = (
    "FP1",
    "F3",
    "C3",
    "P3",
    "O1",
    "F7",
    "T7",
    "P7",
    "FZ",
    "CZ",
    "PZ",
    "FP2",
    "F4",
    "C4",
    "P4",
    "O2",
    "F8",
    "T8",
    "P8",
)
EN17_CHANNEL_ORDER: Final[tuple[str, ...]] = tuple(
    channel for channel in EN19_CHANNEL_ORDER if channel not in {"FZ", "PZ"}
)

_CONTENT_PENDING: Final[str] = "CONTENT-ADDRESS-PENDING"
_SHA256_CHARS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_TRANSFORM_RESULT_SEAL = object()
_MODEL_TILE_SEAL = object()
_FOLD_PHASE_AUTHORITY_SEAL = object()
_PRE_REFERENCE_ELIGIBILITY_SEAL = object()
_VARIANT_TRAINING_ROSTER_AUTHORITY_SEAL = object()
_TARGET_BUNDLE_SEAL = object()
_RECORD_POOL_AUTHORITY_SEAL = object()
_BOUND_LOGITS_SEAL = object()


@dataclass(frozen=True)
class EventNetTransformResult:
    """One immutable provider-native full-record carrier and receipt."""

    signal_uv: np.ndarray
    receipt: dict[str, Any]
    _validation_seal: object


@dataclass(frozen=True)
class EventNetModelTile:
    """One fixed-shape model input and an output-aligned observed mask."""

    model_input_uv: np.ndarray
    output_observed_mask: np.ndarray
    receipt: dict[str, Any]
    _validation_seal: object


@dataclass(frozen=True)
class AuthorizedEventNetFoldPhase:
    """Opaque, byte-replayed fold/phase reference authority.

    A raw mapping or content hash cannot construct an accepted instance.  The
    private seal is issued only after the shared detector validator has replayed
    every authorized sidecar byte under the exact fold plan.
    """

    _phase_receipt_json: str
    _patient_by_identity_json: str
    _authority_receipt_json: str
    _validation_seal: object

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._authority_receipt_json)


@dataclass(frozen=True)
class EventNetPreReferenceEligibilityOutcome:
    """Opaque target-blind support/technical outcome for one record/variant.

    The issuer has no phase/reference/event argument.  It replays the exact
    provider EEG payload, the frozen support route and the deterministic
    provider transform before returning either an eligible transform or a
    typed terminal exclusion.  A serialized outcome is evidence only and is
    never accepted in place of this process-sealed object.
    """

    transform_result: EventNetTransformResult | None
    _receipt_json: str
    _validation_seal: object

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class AuthorizedEventNetVariantTrainingRoster:
    """Opaque target-blind variant/technical-eligibility training roster.

    The eventual issuer must derive this denominator from the exact authorized
    fold phase, frozen provider support route, and pre-reference technical
    eligibility.  Neither a caller-owned subset nor the complete phase roster
    alone is a valid variant-training denominator.
    """

    _roster_json: str
    _receipt_json: str
    _validation_seal: object

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class AuthorizedEventNetTargetBundle:
    """Opaque formal-training target bound to one exact unpadded tile."""

    center_target: np.ndarray
    duration_target: np.ndarray
    center_loss_mask: np.ndarray
    duration_loss_mask: np.ndarray
    distinct_center_count: int
    _receipt_json: str
    _validation_seal: object

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class AuthorizedEventNetRecordPool:
    """Opaque full-record positive/background pool bound to a fold phase."""

    _pool_json: str
    _receipt_json: str
    _validation_seal: object

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class BoundEventNetTrainingLogits:
    """Differentiable logits bound to the exact ordered target-tile roster."""

    center_logits: Tensor
    duration_logits: Tensor
    _receipt_json: str
    _validation_seal: object

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class EventNetTargets:
    """Center/duration targets and separate loss-opportunity masks."""

    center_target: np.ndarray
    duration_target: np.ndarray
    center_loss_mask: np.ndarray
    duration_loss_mask: np.ndarray
    distinct_center_count: int
    receipt: dict[str, Any]


@dataclass(frozen=True)
class EventNetLossResult:
    """Differentiable patient-macro EventNet loss decomposition."""

    loss: Tensor
    center_loss: Tensor
    duration_loss: Tensor
    per_tile_center_loss: Tensor
    per_tile_duration_loss: Tensor


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not set(value).difference(_SHA256_CHARS)
    )


def _require_sha256(value: object, context: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return str(value)


def _strict_dict(
    value: object, required: Iterable[str], context: str
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    expected = set(required)
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise ValueError(f"{context} fields drifted; missing={missing}, extra={extra}")
    return deepcopy(value)


def _payload_receipt(value: np.ndarray, *, semantic: str) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    dtype_map = {
        np.dtype("float32"): ("<f4", "float32_little_endian"),
        np.dtype("float64"): ("<f8", "float64_little_endian"),
        np.dtype("bool"): ("u1", "boolean_uint8"),
    }
    if array.dtype not in dtype_map:
        raise TypeError("unsupported EventNet payload receipt dtype")
    target_dtype, dtype_name = dtype_map[array.dtype]
    canonical = np.ascontiguousarray(array, dtype=target_dtype)
    result: dict[str, Any] = {
        "semantic": semantic,
        "dtype": dtype_name,
        "shape": [int(item) for item in canonical.shape],
        "payload_sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
        "minimum": None,
        "maximum": None,
    }
    if canonical.size:
        result["minimum"] = float(np.min(canonical))
        result["maximum"] = float(np.max(canonical))
    return result


def _tensor_payload_receipt(value: Tensor, *, semantic: str) -> dict[str, Any]:
    """Hash a finite training tensor without detaching the caller's graph."""

    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError("EventNet training payload must be a floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("EventNet training payload contains nonfinite values")
    detached = value.detach().cpu().contiguous()
    dtype_names = {
        torch.float16: "float16_little_endian",
        torch.bfloat16: "bfloat16_little_endian",
        torch.float32: "float32_little_endian",
        torch.float64: "float64_little_endian",
    }
    if detached.dtype not in dtype_names:
        raise TypeError("unsupported EventNet training tensor dtype")
    # NumPy cannot expose bfloat16 bytes portably; view the canonical CPU
    # storage as uint8 instead.  The dtype and shape remain explicit fields.
    raw = detached.view(torch.uint8).numpy().tobytes(order="C")
    return {
        "semantic": semantic,
        "dtype": dtype_names[detached.dtype],
        "shape": [int(item) for item in detached.shape],
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _CONTENT_PENDING
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def _validate_content_address(
    value: object, *, required: Iterable[str], context: str
) -> dict[str, Any]:
    data = _strict_dict(value, required, context)
    supplied = _require_sha256(data["receipt_sha256"], f"{context} receipt")
    pending = deepcopy(data)
    pending["receipt_sha256"] = _CONTENT_PENDING
    if supplied != _canonical_sha256(pending):
        raise ValueError(f"{context} is not content-addressed")
    return data


def eventnet_cleanroom_registry_code_sha256() -> str:
    """Return the exact implementation hash used by the registry."""

    return _file_sha256(Path(__file__).resolve(strict=True))


def _variant_profile(variant_id: str) -> dict[str, Any]:
    if variant_id == EN19_VARIANT_ID:
        order = EN19_CHANNEL_ORDER
        source_support = "complete19_only"
    elif variant_id == EN17_VARIANT_ID:
        order = EN17_CHANNEL_ORDER
        source_support = "lateral17_including_complete19"
    else:
        raise ValueError("unknown independent EventNet variant")
    return {
        "variant_id": variant_id,
        "architecture_input_channels": len(order),
        "provider_channel_order": list(order),
        "required_source_electrodes": list(order),
        "training_population": source_support,
        "direct_from_externally_verified_referential_volts": True,
        "EN19_tensor_or_model_intermediate_used_for_EN17": False,
        "zero_fill_or_interpolation_allowed": False,
        "share_model_object_parameter_storage_or_checkpoint_with_other_variant_allowed": False,
        "independent_random_initialization_and_five_fold_checkpoint_required": True,
    }


class EventNetCleanroomUNet(nn.Module):
    """Executable logit-capable copy of the MIT release's 1-D U-Net.

    Module names, layer shapes, padding modes, and sigmoid-facing ``forward``
    are state-key compatible with the release for EN19.  ``forward_logits`` is
    added because the paper explicitly requires a numerically stable focal
    loss written in terms of logits.  EN17 changes only ``enc0.0.in_channels``
    and is trained as a separate randomly initialized network.
    """

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        if input_channels not in {17, 19}:
            raise ValueError("EventNet clean-room input width must be 17 or 19")
        channels = 16
        stride = 4
        padding = "same"
        kernel_size = 9
        bias = False

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.up = nn.Upsample(scale_factor=stride)
        self.sigm = nn.Sigmoid()
        self.enc0 = nn.Sequential(
            nn.Conv1d(
                input_channels,
                channels,
                kernel_size,
                stride=1,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.BatchNorm1d(channels),
            nn.ELU(inplace=True),
        )
        self.down = nn.MaxPool1d(kernel_size=stride)
        self.enc1 = nn.Sequential(
            nn.Conv1d(
                channels,
                2 * channels,
                kernel_size,
                bias=bias,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.BatchNorm1d(2 * channels),
            nn.ELU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(
                2 * channels,
                4 * channels,
                kernel_size,
                bias=bias,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.BatchNorm1d(4 * channels),
            nn.ELU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv1d(
                12 * channels,
                4 * channels,
                15,
                bias=bias,
                padding=padding,
            ),
            nn.BatchNorm1d(4 * channels),
            nn.ELU(),
            nn.Conv1d(
                4 * channels,
                4 * channels,
                kernel_size,
                bias=bias,
                padding=padding,
            ),
            nn.BatchNorm1d(4 * channels),
            nn.ELU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(
                4 * channels,
                8 * channels,
                kernel_size,
                bias=bias,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.BatchNorm1d(8 * channels),
            nn.ELU(),
        )
        self.dec3 = nn.Sequential(
            nn.Conv1d(
                24 * channels,
                8 * channels,
                15,
                bias=bias,
                padding=padding,
            ),
            nn.BatchNorm1d(8 * channels),
            nn.ELU(),
            nn.Conv1d(
                8 * channels,
                8 * channels,
                15,
                bias=bias,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.BatchNorm1d(8 * channels),
            nn.ELU(),
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(
                8 * channels,
                16 * channels,
                kernel_size,
                bias=bias,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.BatchNorm1d(16 * channels),
            nn.ELU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv1d(
                6 * channels,
                2 * channels,
                15,
                bias=bias,
                padding=padding,
            ),
            nn.BatchNorm1d(2 * channels),
            nn.ELU(inplace=True),
        )
        self.dec0 = nn.Sequential(
            nn.Conv1d(
                3 * channels,
                channels,
                15,
                bias=bias,
                padding=padding,
            ),
            nn.BatchNorm1d(channels),
            nn.ELU(inplace=True),
        )
        self.center_logit = nn.Sequential(
            nn.Conv1d(channels, 1, 21, stride=1, padding=padding),
            nn.MaxPool1d(kernel_size=21, stride=1),
        )
        self.duration_logit = nn.Sequential(
            nn.Conv1d(channels, 1, 21, stride=1, padding=padding),
            nn.MaxPool1d(kernel_size=21, stride=1),
        )

    @staticmethod
    def expected_output_samples(input_samples: int) -> int:
        """Replay the exact release geometry without running a tensor."""

        if isinstance(input_samples, bool) or not isinstance(input_samples, int):
            raise TypeError("input_samples must be an integer")
        if input_samples % 256 != 0 or input_samples < 1280:
            raise ValueError(
                "EventNet input must be a multiple of 256 with bottleneck length >=5"
            )
        return input_samples - 256

    def _validate_input(self, value: Tensor) -> int:
        if not isinstance(value, Tensor) or value.ndim != 3:
            raise ValueError("EventNet input must have shape [batch,channels,samples]")
        if value.shape[1] != self.input_channels:
            raise ValueError("EventNet tensor channel width disagrees with model")
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError("EventNet input must be finite floating point")
        return self.expected_output_samples(int(value.shape[-1]))

    def forward_logits(self, value: Tensor) -> tuple[Tensor, Tensor]:
        expected_output = self._validate_input(value)
        lvl0 = self.enc0(value)
        lvl1 = self.enc1(self.down(lvl0))
        lvl2 = self.enc2(self.down(lvl1))
        lvl3 = self.enc3(self.down(lvl2))
        lvl4 = self.enc4(self.down(lvl3))

        out3 = self.dec3(torch.cat((self.up(lvl4), lvl3), dim=1))
        out2 = self.dec2(torch.cat((self.up(out3), lvl2), dim=1))
        out1 = self.dec1(torch.cat((self.up(out2), lvl1), dim=1))
        out0 = self.dec0(torch.cat((self.up(out1), lvl0), dim=1))

        # The release first loses 20 samples in MaxPool1d(21,stride=1), then
        # crops 118 samples from each side.  Total input-to-output loss is 256.
        crop_each_side = (256 - 21 + 1) // 2
        center = self.center_logit(out0)[:, :, crop_each_side:-crop_each_side]
        duration = self.duration_logit(out0)[:, :, crop_each_side:-crop_each_side]
        if center.shape != duration.shape or int(center.shape[-1]) != expected_output:
            raise RuntimeError("EventNet architecture output geometry drifted")
        return center, duration

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor]:
        center_logit, duration_logit = self.forward_logits(value)
        return self.sigm(center_logit), self.sigm(duration_logit)


def architecture_shape_ledger(input_samples: int = MODEL_INPUT_SAMPLES) -> dict[str, Any]:
    """Return exact encoder/head geometry for a release-compatible input."""

    output_samples = EventNetCleanroomUNet.expected_output_samples(input_samples)
    levels = [input_samples]
    for _ in range(4):
        levels.append(levels[-1] // 4)
    return {
        "input_samples": input_samples,
        "encoder_level_samples": levels,
        "upsample_reconstruction_samples": list(reversed(levels[:-1])),
        "head_conv_same_samples": input_samples,
        "head_max_pool_samples": input_samples - 20,
        "symmetric_head_crop_samples_each_side": 118,
        "output_samples": output_samples,
        "target_tile_samples": TARGET_TILE_SAMPLES,
        "output_equals_target_tile": output_samples == TARGET_TILE_SAMPLES,
    }


def _state_dict_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def derive_training_seed(
    *, variant_id: str, outer_fold: int, stage: str, base_seed: int = 20260824
) -> int:
    """Derive one fixed, non-selectable, variant-specific CPU seed."""

    _variant_profile(variant_id)
    if isinstance(outer_fold, bool) or not isinstance(outer_fold, int) or not 0 <= outer_fold < 5:
        raise ValueError("outer_fold must be one of 0..4")
    if stage not in {"selection", "final_refit"}:
        raise ValueError("stage must be selection or final_refit")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed <= 0:
        raise ValueError("base_seed must be a positive integer")
    token = f"{base_seed}|{variant_id}|{outer_fold}|{stage}|eventnet-v1"
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big") % (2**31 - 1) + 1


def build_randomly_initialized_model(
    *, variant_id: str, outer_fold: int, stage: str
) -> tuple[EventNetCleanroomUNet, dict[str, Any]]:
    """Build a CPU model from scratch without touching the global RNG or GPU."""

    profile = _variant_profile(variant_id)
    seed = derive_training_seed(
        variant_id=variant_id, outer_fold=outer_fold, stage=stage
    )
    # ``default_generator`` is CPU-only.  Preserve caller state explicitly;
    # torch.manual_seed is intentionally avoided because it also seeds CUDA.
    state = torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(seed)
        model = EventNetCleanroomUNet(profile["architecture_input_channels"])
    finally:
        torch.random.set_rng_state(state)
    receipt: dict[str, Any] = {
        "schema_version": "eventnet_cleanroom_random_initialization_receipt_v1",
        "variant_id": variant_id,
        "outer_fold": outer_fold,
        "stage": stage,
        "derived_seed": seed,
        "device": "cpu",
        "published_or_other_variant_checkpoint_loaded": False,
        "parameter_storage_shared_with_other_variant": False,
        "initial_state_dict_sha256": _state_dict_sha256(model),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "receipt_sha256": _CONTENT_PENDING,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return model, receipt


def _polyphase_taps(up: int, down: int) -> np.ndarray:
    rate = max(up, down)
    half_length = 10 * rate
    index = np.arange(-half_length, half_length + 1, dtype=np.float64)
    cutoff = 1.0 / float(rate)
    taps = cutoff * np.sinc(cutoff * index)
    taps *= np.kaiser(taps.size, 5.0)
    taps /= np.sum(taps, dtype=np.float64)
    result = np.ascontiguousarray(taps, dtype="<f8")
    result.setflags(write=False)
    return result


def _direct_project_verified_axes(
    verified_referential_volts: np.ndarray,
    *,
    verified_electrode_order: Sequence[str],
    variant_id: str,
) -> np.ndarray:
    """Pure direct-axis projection used only after external payload replay."""

    profile = _variant_profile(variant_id)
    source = np.asarray(verified_referential_volts)
    order = tuple(verified_electrode_order)
    if source.dtype != np.dtype("float64") or source.ndim != 2:
        raise ValueError("verified carrier must be a float64 [channels,samples] array")
    if source.shape[0] != len(order) or len(set(order)) != len(order):
        raise ValueError("verified carrier order is invalid")
    index = {channel: position for position, channel in enumerate(order)}
    missing = sorted(set(profile["required_source_electrodes"]).difference(index))
    if missing:
        raise ValueError(f"variant source support is incomplete: {missing}")
    # There is intentionally no 19-axis staging array in this code path.
    projected = np.stack(
        [source[index[channel]] for channel in profile["provider_channel_order"]],
        axis=0,
    )
    return np.ascontiguousarray(projected, dtype="<f8")


def apply_full_record_transform(
    referential_volts: object,
    *,
    variant_id: str,
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
    registry: Mapping[str, Any],
) -> EventNetTransformResult:
    """Create one EN19/EN17 carrier from typed, replayed referential volts."""

    validated_registry = validate_registry(dict(registry))
    lineage = require_validated_detector_signal_lineage_authority(
        signal_lineage_authority
    )
    verified, order, rate_pair = verify_provider_referential_payload(
        signal_lineage_authority, referential_volts
    )
    profile = _variant_profile(variant_id)
    usable = set(
        lineage["EEG_only_channel_QC_authority"]["usable_standard_channel_ids"]
    )
    missing_usable = sorted(set(profile["required_source_electrodes"]).difference(usable))
    if missing_usable:
        raise ValueError(f"variant EEG-QC usable support is incomplete: {missing_usable}")
    projected_volts = _direct_project_verified_axes(
        verified,
        verified_electrode_order=order,
        variant_id=variant_id,
    )

    rate = Fraction(int(rate_pair[0]), int(rate_pair[1]))
    if rate < 1 or rate > 4096:
        raise ValueError("EventNet source sampling rate is outside 1--4096 Hz")
    ratio = Fraction(TARGET_FS_HZ, 1) / rate
    up, down = ratio.numerator, ratio.denominator
    if max(up, down) > 4096:
        raise ValueError("EventNet reduced resampling ratio exceeds frozen support")
    target_count = (projected_volts.shape[1] * up) // down
    if target_count < 1:
        raise ValueError("EventNet full-record transform has no target sample")
    if up == down == 1:
        taps: np.ndarray | None = None
        resampled_volts = projected_volts.copy()
    else:
        if projected_volts.shape[1] < 2:
            raise ValueError("non-identity EventNet resampling needs at least two samples")
        taps = _polyphase_taps(up, down)
        resampled_volts = resample_poly(
            projected_volts,
            up,
            down,
            axis=1,
            window=taps,
            padtype="line",
        )
        if resampled_volts.shape[1] < target_count:
            raise RuntimeError("EventNet polyphase resampler returned insufficient support")
        resampled_volts = resampled_volts[:, :target_count]
    output_uv = np.ascontiguousarray(resampled_volts * 1_000_000.0, dtype="<f4")
    if not np.isfinite(output_uv).all():
        raise ValueError("EventNet provider carrier contains nonfinite values")
    output_uv.setflags(write=False)

    tap_receipt = None if taps is None else _payload_receipt(
        taps, semantic="explicit_polyphase_kaiser_sinc_taps"
    )
    receipt: dict[str, Any] = {
        "schema_version": "eventnet_cleanroom_full_record_transform_receipt_v1",
        "registry_id": validated_registry["registry_id"],
        "registry_sha256": validated_registry["registry_sha256"],
        "implementation_code_sha256": validated_registry["implementation"]["code_sha256"],
        "provider_id": PROVIDER_ID,
        "variant_id": variant_id,
        "detector_signal_lineage_authority_sha256": lineage["receipt_sha256"],
        "canonical_signal_receipt_sha256": lineage["canonical_physical_signal"][
            "canonical_signal_receipt_sha256"
        ],
        "canonical_source_header_receipt_sha256": lineage["canonical_physical_signal"][
            "source_header_receipt_sha256"
        ],
        "canonical_source_tensor_sha256": lineage["canonical_physical_signal"][
            "source_tensor_sha256"
        ],
        "observed_roster_authority_sha256": lineage["observed_roster_authority"][
            "receipt_sha256"
        ],
        "common_sampling_clock_authority_sha256": lineage[
            "common_sampling_clock_authority"
        ]["receipt_sha256"],
        "EEG_electrical_reference_system_authority_sha256": lineage[
            "electrical_reference_system_authority"
        ]["receipt_sha256"],
        "EEG_only_channel_QC_authority_sha256": lineage[
            "EEG_only_channel_QC_authority"
        ]["receipt_sha256"],
        "input_electrode_order": list(order),
        "input_sampling_rate_fraction_hz": [rate.numerator, rate.denominator],
        "input_payload_receipt": _payload_receipt(
            verified, semantic="externally_replayed_canonical_referential_EEG_volts"
        ),
        "direct_variant_axis_projection": {
            "provider_channel_order": profile["provider_channel_order"],
            "EN19_intermediate_used_for_EN17": False,
            "zero_filled_or_interpolated_channel_count": 0,
            "selected_volts_payload_receipt": _payload_receipt(
                projected_volts,
                semantic="direct_selected_variant_referential_EEG_volts",
            ),
        },
        "resample": {
            "whole_record_single_execution": True,
            "up": up,
            "down": down,
            "target_sample_count_floor_policy": target_count,
            "tap_payload_receipt": tap_receipt,
            "padtype": "line",
        },
        "provider_preprocessing": {
            "volts_to_microvolts_multiplier": 1_000_000.0,
            "filtering": "none",
            "normalization": "none",
            "clipping": "none",
            "fold_fitted_statistics": "none",
            "target_or_cross_record_information_used": False,
        },
        "output": {
            "provider_channel_order": profile["provider_channel_order"],
            "sampling_rate_fraction_hz": [TARGET_FS_HZ, 1],
            "sample_count": target_count,
            "physical_unit": "uV",
            "payload_receipt": _payload_receipt(
                output_uv, semantic="EventNet_cleanroom_provider_native_full_record_uV"
            ),
        },
        "scope_receipt": {
            "EEG_samples_used": True,
            "acquisition_clock_used": True,
            "EEG_electrical_reference_provenance_used_as_control_plane": True,
            "EEG_electrical_reference_provenance_used_as_model_feature": False,
            "EEG_only_QC_used_as_admission_control_plane": True,
            "seizure_target_or_reference_label_used": False,
            "EDF_annotation_used": False,
            "spreadsheet_or_doctor_text_used": False,
            "clinical_history_used": False,
            "auxiliary_non_EEG_channel_used": False,
        },
        "receipt_sha256": _CONTENT_PENDING,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return EventNetTransformResult(
        signal_uv=output_uv,
        receipt=receipt,
        _validation_seal=_TRANSFORM_RESULT_SEAL,
    )


def validate_transform_result(
    result: EventNetTransformResult,
    *,
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
    registry: Mapping[str, Any],
) -> EventNetTransformResult:
    """Validate payload identity and typed external lineage without rerunning."""

    if (
        not isinstance(result, EventNetTransformResult)
        or result._validation_seal is not _TRANSFORM_RESULT_SEAL
    ):
        raise TypeError("result must be an opaque materialized EventNet transform")
    validated_registry = validate_registry(dict(registry))
    lineage = require_validated_detector_signal_lineage_authority(
        signal_lineage_authority
    )
    receipt = deepcopy(result.receipt)
    if receipt.get("schema_version") != "eventnet_cleanroom_full_record_transform_receipt_v1":
        raise ValueError("EventNet transform receipt schema drifted")
    if (
        receipt.get("registry_sha256") != validated_registry["registry_sha256"]
        or receipt.get("detector_signal_lineage_authority_sha256")
        != lineage["receipt_sha256"]
    ):
        raise ValueError("EventNet transform registry or lineage binding drifted")
    profile = _variant_profile(str(receipt.get("variant_id")))
    signal = np.asarray(result.signal_uv)
    if (
        signal.dtype != np.dtype("float32")
        or signal.ndim != 2
        or signal.shape[0] != profile["architecture_input_channels"]
        or signal.shape[1] < 1
        or not np.isfinite(signal).all()
    ):
        raise ValueError("EventNet transform output payload is invalid")
    expected = _payload_receipt(
        signal, semantic="EventNet_cleanroom_provider_native_full_record_uV"
    )
    if receipt["output"]["payload_receipt"] != expected:
        raise ValueError("EventNet transform output payload receipt drifted")
    if (
        receipt["direct_variant_axis_projection"]["provider_channel_order"]
        != profile["provider_channel_order"]
        or receipt["direct_variant_axis_projection"][
            "EN19_intermediate_used_for_EN17"
        ]
        is not False
        or receipt["direct_variant_axis_projection"][
            "zero_filled_or_interpolated_channel_count"
        ]
        != 0
    ):
        raise ValueError("EventNet independent direct-axis contract drifted")
    pending = deepcopy(receipt)
    supplied = pending["receipt_sha256"]
    pending["receipt_sha256"] = _CONTENT_PENDING
    if supplied != _canonical_sha256(pending):
        raise ValueError("EventNet transform receipt is not content-addressed")
    validated_signal = np.ascontiguousarray(signal, dtype="<f4")
    validated_signal.setflags(write=False)
    return EventNetTransformResult(
        signal_uv=validated_signal,
        receipt=receipt,
        _validation_seal=_TRANSFORM_RESULT_SEAL,
    )


def enumerate_target_tiles(sample_count: int) -> tuple[tuple[int, int], ...]:
    """Cover every observed 256-Hz sample exactly once in 120-second targets."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    return tuple(
        (start, min(TARGET_TILE_SAMPLES, sample_count - start))
        for start in range(0, sample_count, TARGET_TILE_SAMPLES)
    )


def enumerate_training_target_tiles(sample_count: int) -> tuple[tuple[int, int], ...]:
    """Return only tiles whose target and both context sides are observed.

    Training starts at sample 128 rather than zero.  With a 120-second hop,
    this excludes only the first/last 0.5 seconds instead of feeding zero
    padding through temporal BatchNorm and contaminating every observed logit.
    Short records and incomplete tails remain inference-evaluable through the
    explicit mask path but are fail-closed for gradient/selection loss.
    """

    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    starts: list[tuple[int, int]] = []
    start = CONTEXT_SAMPLES_PER_SIDE
    while start + TARGET_TILE_SAMPLES + CONTEXT_SAMPLES_PER_SIDE <= sample_count:
        starts.append((start, TARGET_TILE_SAMPLES))
        start += TARGET_TILE_SAMPLES
    return tuple(starts)


def materialize_model_tile(
    provider_signal_uv: object, *, target_start_sample: int
) -> EventNetModelTile:
    """Create a fixed input with explicit non-observed tail/short-record mask."""

    source = np.asarray(provider_signal_uv)
    if source.dtype != np.dtype("float32") or source.ndim != 2:
        raise ValueError("provider signal must be float32 [channels,samples]")
    if source.shape[0] not in {17, 19} or source.shape[1] <= 0:
        raise ValueError("provider signal has unsupported shape")
    if not np.isfinite(source).all():
        raise ValueError("provider signal contains nonfinite values")
    if (
        isinstance(target_start_sample, bool)
        or not isinstance(target_start_sample, int)
        or target_start_sample < 0
        or target_start_sample >= source.shape[1]
    ):
        raise ValueError("target tile start lies outside the recording")

    actual = min(TARGET_TILE_SAMPLES, source.shape[1] - target_start_sample)
    wanted_start = target_start_sample - CONTEXT_SAMPLES_PER_SIDE
    wanted_stop = target_start_sample + TARGET_TILE_SAMPLES + CONTEXT_SAMPLES_PER_SIDE
    observed_start = max(0, wanted_start)
    observed_stop = min(source.shape[1], wanted_stop)
    destination_start = observed_start - wanted_start
    destination_stop = destination_start + observed_stop - observed_start
    model_input = np.zeros(
        (source.shape[0], MODEL_INPUT_SAMPLES), dtype=np.float32
    )
    model_input[:, destination_start:destination_stop] = source[
        :, observed_start:observed_stop
    ]
    output_mask = np.zeros(TARGET_TILE_SAMPLES, dtype=np.bool_)
    output_mask[:actual] = True
    model_input.setflags(write=False)
    output_mask.setflags(write=False)
    receipt: dict[str, Any] = {
        "schema_version": "eventnet_cleanroom_model_tile_v1",
        "target_start_sample": target_start_sample,
        "target_stop_sample_exclusive": target_start_sample + actual,
        "actual_observed_target_samples": actual,
        "model_target_capacity_samples": TARGET_TILE_SAMPLES,
        "model_input_samples": MODEL_INPUT_SAMPLES,
        "observed_source_start_sample": observed_start,
        "observed_source_stop_sample_exclusive": observed_stop,
        "left_context_padding_samples": max(0, -wanted_start),
        "right_context_or_tail_padding_samples": max(0, wanted_stop - source.shape[1]),
        "padding_value_uV": 0.0,
        "padding_is_observed_EEG": False,
        "nonobserved_output_samples_enter_loss": False,
        "training_loss_eligible": bool(
            actual == TARGET_TILE_SAMPLES
            and wanted_start >= 0
            and wanted_stop <= source.shape[1]
        ),
        "training_forward_with_any_padding_allowed": False,
        "model_input_payload_receipt": _payload_receipt(
            model_input, semantic="EventNet_cleanroom_fixed_shape_model_input_uV"
        ),
        "output_observed_mask_payload_receipt": _payload_receipt(
            output_mask, semantic="EventNet_output_observed_sample_mask"
        ),
        "receipt_sha256": _CONTENT_PENDING,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return EventNetModelTile(
        model_input_uv=model_input,
        output_observed_mask=output_mask,
        receipt=receipt,
        _validation_seal=_MODEL_TILE_SEAL,
    )


def _require_materialized_model_tile(
    value: object,
    *,
    provider_signal_uv: np.ndarray | None = None,
) -> EventNetModelTile:
    """Replay an opaque tile and optionally its exact parent transform bytes."""

    if (
        not isinstance(value, EventNetModelTile)
        or value._validation_seal is not _MODEL_TILE_SEAL
    ):
        raise TypeError("formal EventNet training requires an opaque model tile")
    model_input = np.asarray(value.model_input_uv)
    observed_mask = np.asarray(value.output_observed_mask)
    receipt = deepcopy(value.receipt)
    if (
        model_input.dtype != np.dtype("float32")
        or model_input.ndim != 2
        or model_input.shape[0] not in {17, 19}
        or model_input.shape[1] != MODEL_INPUT_SAMPLES
        or not np.isfinite(model_input).all()
        or observed_mask.dtype != np.dtype("bool")
        or observed_mask.shape != (TARGET_TILE_SAMPLES,)
    ):
        raise ValueError("EventNet model tile payload drifted")
    if receipt.get("schema_version") != "eventnet_cleanroom_model_tile_v1":
        raise ValueError("EventNet model tile receipt schema drifted")
    if receipt.get("model_input_payload_receipt") != _payload_receipt(
        model_input, semantic="EventNet_cleanroom_fixed_shape_model_input_uV"
    ) or receipt.get("output_observed_mask_payload_receipt") != _payload_receipt(
        observed_mask, semantic="EventNet_output_observed_sample_mask"
    ):
        raise ValueError("EventNet model tile payload receipt drifted")
    pending = deepcopy(receipt)
    supplied = pending.get("receipt_sha256")
    pending["receipt_sha256"] = _CONTENT_PENDING
    if supplied != _canonical_sha256(pending):
        raise ValueError("EventNet model tile receipt is not content-addressed")
    if provider_signal_uv is not None:
        expected = materialize_model_tile(
            provider_signal_uv,
            target_start_sample=int(receipt["target_start_sample"]),
        )
        if (
            expected.receipt != receipt
            or not np.array_equal(expected.model_input_uv, model_input)
            or not np.array_equal(expected.output_observed_mask, observed_mask)
        ):
            raise ValueError("EventNet tile disagrees with exact parent transform")
    return value


def _fraction_seconds(value: object, context: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{context} must be numeric")
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{context} is not a finite rational time") from exc
    if not math.isfinite(float(result)):
        raise ValueError(f"{context} must be finite")
    return result


def _nearest_integer_ties_earlier(value: Fraction) -> int:
    floor_value = value.numerator // value.denominator
    remainder = value - floor_value
    return floor_value + int(remainder > Fraction(1, 2))


def _event_rows(
    events: Sequence[Mapping[str, Any]], *, record_sample_count: int
) -> list[dict[str, Any]]:
    record_duration = Fraction(record_sample_count, TARGET_FS_HZ)
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(events):
        if not isinstance(raw, Mapping):
            raise TypeError("EventNet target event must be an object")
        if not {"start_seconds", "stop_seconds"}.issubset(raw):
            raise ValueError("EventNet target event lacks start/stop seconds")
        allowed = {"start_seconds", "stop_seconds", "left_censored", "right_censored"}
        if set(raw).difference(allowed):
            raise ValueError("EventNet target event contains an unrecognized field")
        start = _fraction_seconds(raw["start_seconds"], "event start")
        stop = _fraction_seconds(raw["stop_seconds"], "event stop")
        if start < 0 or stop <= start or stop > record_duration:
            raise ValueError("EventNet target event lies outside the record")
        left_censored = raw.get("left_censored", start == 0)
        right_censored = raw.get("right_censored", stop == record_duration)
        if type(left_censored) is not bool or type(right_censored) is not bool:
            raise TypeError("EventNet censor flags must be booleans")
        duration_samples = (stop - start) * TARGET_FS_HZ
        center_exact = (start + stop) * TARGET_FS_HZ / 2
        center_sample = _nearest_integer_ties_earlier(center_exact)
        rows.append(
            {
                "event_index": index,
                "start_seconds_fraction": [start.numerator, start.denominator],
                "stop_seconds_fraction": [stop.numerator, stop.denominator],
                "start_sample_fraction": [
                    (start * TARGET_FS_HZ).numerator,
                    (start * TARGET_FS_HZ).denominator,
                ],
                "stop_sample_fraction": [
                    (stop * TARGET_FS_HZ).numerator,
                    (stop * TARGET_FS_HZ).denominator,
                ],
                "duration_samples": float(duration_samples),
                "center_sample_exact": float(center_exact),
                "center_sample_quantized_ties_earlier": center_sample,
                "center_quantization_error_samples": float(center_sample - center_exact),
                "left_censored": left_censored,
                "right_censored": right_censored,
            }
        )
    return rows


def _mask_event_intersection(
    mask: np.ndarray,
    *,
    event: Mapping[str, Any],
    tile_start_sample: int,
    actual_observed_samples: int,
) -> None:
    start_fraction = Fraction(*event["start_sample_fraction"])
    stop_fraction = Fraction(*event["stop_sample_fraction"])
    start_index = math.floor(start_fraction) - tile_start_sample
    stop_index = math.ceil(stop_fraction) - tile_start_sample
    local_start = max(0, start_index)
    local_stop = min(actual_observed_samples, stop_index)
    if local_stop > local_start:
        mask[local_start:local_stop] = False


def build_eventnet_targets_pure_primitive(
    events: Sequence[Mapping[str, Any]],
    *,
    record_sample_count: int,
    target_start_sample: int,
    actual_observed_target_samples: int,
) -> EventNetTargets:
    """Pure numerical target primitive; never a reference/fold authority.

    A complete event contributes only to the tile containing its quantized
    center.  Multiple Gaussians are combined pointwise by ``max``.  If two
    events quantize to the same center, that center is counted once and the
    larger capped duration target wins.  Boundary-censored events and pieces
    of a complete event whose center belongs to another tile are masked from
    center loss, preventing them from being mislabeled as background.
    """

    if (
        isinstance(record_sample_count, bool)
        or not isinstance(record_sample_count, int)
        or record_sample_count <= 0
    ):
        raise ValueError("record_sample_count must be positive")
    if (
        isinstance(target_start_sample, bool)
        or not isinstance(target_start_sample, int)
        or target_start_sample < 0
    ):
        raise ValueError("target_start_sample is invalid")
    expected_actual = min(TARGET_TILE_SAMPLES, record_sample_count - target_start_sample)
    if expected_actual <= 0 or actual_observed_target_samples != expected_actual:
        raise ValueError("actual target support disagrees with record/tile geometry")

    rows = _event_rows(events, record_sample_count=record_sample_count)
    center = np.zeros(TARGET_TILE_SAMPLES, dtype=np.float64)
    duration = np.zeros(TARGET_TILE_SAMPLES, dtype=np.float64)
    center_mask = np.zeros(TARGET_TILE_SAMPLES, dtype=np.bool_)
    center_mask[:actual_observed_target_samples] = True
    duration_mask = np.zeros(TARGET_TILE_SAMPLES, dtype=np.bool_)
    local_centers: dict[int, list[int]] = {}
    censored_count = 0
    outside_center_overlap_count = 0
    duration_capped_count = 0

    sample_axis = np.arange(actual_observed_target_samples, dtype=np.float64)
    for row in rows:
        censored = bool(row["left_censored"] or row["right_censored"])
        local_center = int(row["center_sample_quantized_ties_earlier"]) - target_start_sample
        if censored:
            censored_count += 1
            _mask_event_intersection(
                center_mask,
                event=row,
                tile_start_sample=target_start_sample,
                actual_observed_samples=actual_observed_target_samples,
            )
            continue
        if not 0 <= local_center < actual_observed_target_samples:
            before = int(np.count_nonzero(center_mask))
            _mask_event_intersection(
                center_mask,
                event=row,
                tile_start_sample=target_start_sample,
                actual_observed_samples=actual_observed_target_samples,
            )
            if int(np.count_nonzero(center_mask)) < before:
                outside_center_overlap_count += 1
            continue

        duration_samples = float(row["duration_samples"])
        sigma_samples = 0.5 * duration_samples / 6.0
        if not math.isfinite(sigma_samples) or sigma_samples <= 0:
            raise ValueError("EventNet target sigma is invalid")
        # Preserve the half-open interval's exact fractional midpoint for the
        # Gaussian.  Only the sparse duration anchor is quantized to a sample.
        # This avoids silently moving an onset/offset-derived center by up to
        # half a 256-Hz sample.
        exact_local_center = float(row["center_sample_exact"]) - target_start_sample
        gaussian = np.exp(
            -((sample_axis - exact_local_center) ** 2) / (2.0 * sigma_samples**2)
        )
        center[:actual_observed_target_samples] = np.maximum(
            center[:actual_observed_target_samples], gaussian
        )
        normalized_duration = min(
            duration_samples / (MAXIMUM_DURATION_SECONDS * TARGET_FS_HZ), 1.0
        )
        if normalized_duration >= 1.0 and duration_samples > MAXIMUM_DURATION_SECONDS * TARGET_FS_HZ:
            duration_capped_count += 1
        duration[local_center] = max(duration[local_center], normalized_duration)
        duration_mask[local_center] = True
        local_centers.setdefault(local_center, []).append(int(row["event_index"]))

    center[~center_mask] = 0.0
    duration[~duration_mask] = 0.0
    center32 = np.ascontiguousarray(center, dtype="<f4")
    duration32 = np.ascontiguousarray(duration, dtype="<f4")
    center_mask = np.ascontiguousarray(center_mask)
    duration_mask = np.ascontiguousarray(duration_mask)
    for value in (center32, duration32, center_mask, duration_mask):
        value.setflags(write=False)

    receipt: dict[str, Any] = {
        "schema_version": "eventnet_center_duration_target_v1",
        "paper_doi": "10.1109/TBME.2024.3375759",
        "target_start_sample": target_start_sample,
        "actual_observed_target_samples": actual_observed_target_samples,
        "source_event_count": len(rows),
        "distinct_complete_center_count": len(local_centers),
        "center_collision_count": sum(max(0, len(indices) - 1) for indices in local_centers.values()),
        "center_collision_policy": "one_center_target_and_maximum_capped_duration",
        "censored_event_count": censored_count,
        "outside_center_overlapping_event_count": outside_center_overlap_count,
        "duration_capped_above_300_seconds_count": duration_capped_count,
        "center_target": {
            "formula": "exp(-(t-t_star)^2/(2*sigma^2))",
            "sigma": "0.5*actual_event_duration_samples/6",
            "multiple_event_reduction": "pointwise_max",
            "Gaussian_center_clock": "exact_fractional_midpoint_of_half_open_interval",
            "positive_anchor_clock": "nearest_256Hz_sample_ties_to_earlier",
            "positive_focal_branch_uses_sparse_duration_anchor_not_float_equality_to_one": True,
        },
        "duration_target": {
            "defined_only_at_distinct_complete_target_centers": True,
            "normalization": "min(actual_duration_seconds,300)/300",
        },
        "edge_and_censor_policy": {
            "nonobserved_tail_loss_mask": False,
            "boundary_censored_event_center_and_duration_supervision": False,
            "event_piece_whose_center_is_in_another_tile_treated_as_background": False,
        },
        "event_projection_rows": rows,
        "center_target_payload_receipt": _payload_receipt(center32, semantic="EventNet_center_target"),
        "duration_target_payload_receipt": _payload_receipt(duration32, semantic="EventNet_duration_fraction_target"),
        "center_loss_mask_payload_receipt": _payload_receipt(center_mask, semantic="EventNet_center_loss_opportunity_mask"),
        "duration_loss_mask_payload_receipt": _payload_receipt(duration_mask, semantic="EventNet_duration_loss_center_mask"),
        "receipt_sha256": _CONTENT_PENDING,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return EventNetTargets(
        center_target=center32,
        duration_target=duration32,
        center_loss_mask=center_mask,
        duration_loss_mask=duration_mask,
        distinct_center_count=len(local_centers),
        receipt=receipt,
    )


def _as_batch_time_tensor(value: object, *, context: str, dtype: torch.dtype | None = None) -> Tensor:
    # Target arrays are intentionally immutable.  Copy NumPy inputs so PyTorch
    # can never expose a writable view of a read-only receipt-bound payload.
    tensor = (
        value
        if isinstance(value, Tensor)
        else torch.as_tensor(np.array(value, copy=True))
    )
    if tensor.ndim == 3 and tensor.shape[1] == 1:
        tensor = tensor[:, 0, :]
    if tensor.ndim != 2:
        raise ValueError(f"{context} must have shape [batch,time] or [batch,1,time]")
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def eventnet_multitask_loss_from_logits_pure_primitive(
    center_logits: object,
    duration_logits: object,
    center_targets: object,
    duration_targets: object,
    center_loss_mask: object,
    duration_loss_mask: object,
    distinct_center_count_by_tile: Sequence[int],
    *,
    patient_keys: Sequence[str] | None = None,
) -> EventNetLossResult:
    """Pure numerical loss primitive; raw masks/keys are not training authority."""

    center_logit = _as_batch_time_tensor(center_logits, context="center logits")
    duration_logit = _as_batch_time_tensor(duration_logits, context="duration logits")
    if not center_logit.is_floating_point() or not duration_logit.is_floating_point():
        raise TypeError("EventNet logits must be floating point")
    if center_logit.shape != duration_logit.shape:
        raise ValueError("EventNet center/duration logits must share shape")
    if not bool(torch.isfinite(center_logit).all()) or not bool(torch.isfinite(duration_logit).all()):
        raise ValueError("EventNet logits contain nonfinite values")
    target = _as_batch_time_tensor(
        center_targets, context="center targets", dtype=center_logit.dtype
    ).to(device=center_logit.device)
    duration_target = _as_batch_time_tensor(
        duration_targets, context="duration targets", dtype=duration_logit.dtype
    ).to(device=duration_logit.device)
    center_mask = _as_batch_time_tensor(
        center_loss_mask, context="center loss mask"
    ).to(device=center_logit.device, dtype=torch.bool)
    duration_mask = _as_batch_time_tensor(
        duration_loss_mask, context="duration loss mask"
    ).to(device=duration_logit.device, dtype=torch.bool)
    if not (
        target.shape
        == duration_target.shape
        == center_mask.shape
        == duration_mask.shape
        == center_logit.shape
    ):
        raise ValueError("EventNet logits, targets and masks must share shape")
    if (
        not bool(torch.isfinite(target).all())
        or not bool(torch.isfinite(duration_target).all())
        or bool(torch.any(target < 0))
        or bool(torch.any(target > 1))
        or bool(torch.any(duration_target < 0))
        or bool(torch.any(duration_target > 1))
        or bool(torch.any(duration_mask & ~center_mask))
    ):
        raise ValueError("EventNet targets or masks violate frozen support")
    batch_size = int(center_logit.shape[0])
    if len(distinct_center_count_by_tile) != batch_size:
        raise ValueError("distinct center counts disagree with batch")

    # For a fractional interval midpoint, no discrete sample necessarily has
    # c(t)==1.  The nearest-sample (ties earlier) duration anchor is therefore
    # the frozen positive focal branch; every other observed sample uses the
    # modified negative branch and the exact fractional Gaussian weight.
    positive = duration_mask & center_mask
    negative = center_mask & ~positive
    observed_positive_counts = positive.sum(dim=1)
    expected_counts = torch.as_tensor(
        list(distinct_center_count_by_tile),
        device=observed_positive_counts.device,
        dtype=observed_positive_counts.dtype,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in distinct_center_count_by_tile):
        raise ValueError("distinct center counts must be nonnegative integers")
    if not bool(torch.equal(observed_positive_counts, expected_counts)):
        raise ValueError("target positive centers disagree with supplied event counts")
    if not bool(torch.equal(duration_mask.sum(dim=1), expected_counts)):
        raise ValueError("duration-mask centers disagree with supplied event counts")

    # log(sigmoid(z))=-softplus(-z), log(1-sigmoid(z))=-softplus(z).
    positive_term = (
        (1.0 - FOCAL_ALPHA_C)
        * torch.sigmoid(-center_logit).pow(FOCAL_ALPHA)
        * torch_functional.softplus(-center_logit)
    )
    negative_term = (
        FOCAL_ALPHA_C
        * (1.0 - target).pow(FOCAL_BETA)
        * torch.sigmoid(center_logit).pow(FOCAL_ALPHA)
        * torch_functional.softplus(center_logit)
    )
    focal_sum = (positive_term * positive).sum(dim=1) + (
        negative_term * negative
    ).sum(dim=1)
    # CenterNet convention resolves the paper's N=0 background-tile hole:
    # event tiles divide by N; background tiles retain the summed negative loss.
    denominator = observed_positive_counts.clamp_min(1).to(center_logit.dtype)
    per_tile_center = focal_sum / denominator

    predicted_duration = torch.sigmoid(duration_logit)
    minimum = torch.minimum(predicted_duration, duration_target)
    maximum = torch.maximum(predicted_duration, duration_target)
    iou = minimum / maximum.clamp_min(torch.finfo(duration_logit.dtype).tiny)
    duration_error = (1.0 - iou) * duration_mask
    per_tile_duration = torch.where(
        observed_positive_counts > 0,
        duration_error.sum(dim=1) / denominator,
        duration_logit.sum(dim=1) * 0.0,
    )

    if patient_keys is None:
        patient_keys = tuple(f"tile-{index:08d}" for index in range(batch_size))
    if len(patient_keys) != batch_size or any(
        not isinstance(patient, str) or not patient for patient in patient_keys
    ):
        raise ValueError("patient keys must align with the batch")
    groups: dict[str, list[int]] = {}
    for index, patient in enumerate(patient_keys):
        groups.setdefault(patient, []).append(index)
    patient_center = torch.stack(
        [per_tile_center[indices].mean() for patient, indices in sorted(groups.items())]
    ).mean()
    patient_duration = torch.stack(
        [per_tile_duration[indices].mean() for patient, indices in sorted(groups.items())]
    ).mean()
    total = patient_center + DURATION_LOSS_WEIGHT * patient_duration
    return EventNetLossResult(
        loss=total,
        center_loss=patient_center,
        duration_loss=patient_duration,
        per_tile_center_loss=per_tile_center,
        per_tile_duration_loss=per_tile_duration,
    )


def build_record_tile_pools_pure_primitive(
    events: Sequence[Mapping[str, Any]],
    *,
    record_key: str,
    record_sample_count: int,
) -> dict[str, Any]:
    """Pure numerical pool primitive; raw events are not training authority."""

    if not isinstance(record_key, str) or not record_key:
        raise ValueError("record_key must be a non-empty control-plane key")
    rows = _event_rows(events, record_sample_count=record_sample_count)
    positive: list[str] = []
    background: list[str] = []
    transition: list[str] = []
    tile_rows: list[dict[str, Any]] = []
    for tile_index, (start, actual) in enumerate(
        enumerate_training_target_tiles(record_sample_count)
    ):
        stop = start + actual
        support_start = max(0, start - CONTEXT_SAMPLES_PER_SIDE)
        support_stop = min(
            record_sample_count,
            start + TARGET_TILE_SAMPLES + CONTEXT_SAMPLES_PER_SIDE,
        )
        centers = [
            row
            for row in rows
            if not row["left_censored"]
            and not row["right_censored"]
            and start <= row["center_sample_quantized_ties_earlier"] < stop
        ]
        intersects_support = any(
            Fraction(*row["stop_sample_fraction"]) > support_start
            and Fraction(*row["start_sample_fraction"]) < support_stop
            for row in rows
        )
        tile_id = f"{record_key}:eventnet-tile-{tile_index:06d}"
        if centers:
            pool = "positive"
            positive.append(tile_id)
        elif not intersects_support:
            pool = "background"
            background.append(tile_id)
        else:
            pool = "transition_excluded_from_background"
            transition.append(tile_id)
        tile_rows.append(
            {
                "tile_id": tile_id,
                "target_start_sample": start,
                "actual_observed_target_samples": actual,
                "pool": pool,
            }
        )
    body: dict[str, Any] = {
        "schema_version": "eventnet_cleanroom_record_tile_pool_v1",
        "record_key": record_key,
        "positive": positive,
        "background": background,
        "transition_excluded_from_background": transition,
        "tile_rows": tile_rows,
        "target_reference_used_only_for_training_target_and_sampler_control_plane": True,
        "record_or_patient_key_used_as_model_feature": False,
        "all_pool_tiles_have_fully_observed_target_and_context": True,
        "padded_or_short_or_tail_tile_gradient_loss_allowed": False,
        "receipt_sha256": _CONTENT_PENDING,
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def build_patient_balanced_epoch_plan_pure_primitive(
    tile_pools_by_patient: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    variant_id: str,
    outer_fold: int,
    stage: str,
    epoch_index: int,
) -> dict[str, Any]:
    """Pure deterministic sampler primitive over a caller-supplied roster."""

    if isinstance(epoch_index, bool) or not isinstance(epoch_index, int) or epoch_index < 0:
        raise ValueError("epoch_index must be a nonnegative integer")
    seed = derive_training_seed(
        variant_id=variant_id, outer_fold=outer_fold, stage=stage
    )
    draws_by_patient: dict[str, list[dict[str, str]]] = {}
    pool_rows: list[dict[str, Any]] = []

    def ordered(values: Sequence[str], *, patient: str, pool: str) -> list[str]:
        return sorted(
            values,
            key=lambda tile: hashlib.sha256(
                f"{seed}|{patient}|{pool}|{tile}".encode("utf-8")
            ).digest(),
        )

    for patient in sorted(tile_pools_by_patient):
        if not isinstance(patient, str) or not patient:
            raise ValueError("patient grouping key must be non-empty")
        pools = _strict_dict(
            dict(tile_pools_by_patient[patient]),
            {"positive", "background"},
            "EventNet patient tile pools",
        )
        positive = tuple(pools["positive"])
        background = tuple(pools["background"])
        if not positive and not background:
            raise ValueError("each patient needs an eligible positive or background tile")
        if (
            any(not isinstance(tile, str) or not tile for tile in positive + background)
            or len(set(positive)) != len(positive)
            or len(set(background)) != len(background)
            or set(positive).intersection(background)
        ):
            raise ValueError("patient EventNet tile pools are invalid")
        positive_order = ordered(positive, patient=patient, pool="positive")
        background_order = ordered(background, patient=patient, pool="background")
        quotas = (
            {"positive": 4, "background": 4}
            if positive and background
            else {"positive": 8 if positive else 0, "background": 8 if background else 0}
        )
        draws: list[dict[str, str]] = []
        for pool_name, pool_order in (
            ("positive", positive_order),
            ("background", background_order),
        ):
            for offset in range(quotas[pool_name]):
                tile = pool_order[(epoch_index * quotas[pool_name] + offset) % len(pool_order)]
                draws.append({"tile_id": tile, "pool": pool_name})
        draws_by_patient[patient] = draws
        pool_rows.append(
            {
                "patient_key": patient,
                "positive": list(positive),
                "background": list(background),
            }
        )

    if not draws_by_patient:
        raise ValueError("EventNet epoch plan needs at least one patient")
    batches: list[list[dict[str, str]]] = []
    for draw_index in range(8):
        patient_order = sorted(
            draws_by_patient,
            key=lambda patient: hashlib.sha256(
                f"{seed}|{epoch_index}|{draw_index}|{patient}".encode("utf-8")
            ).digest(),
        )
        for start in range(0, len(patient_order), 16):
            batch: list[dict[str, str]] = []
            for patient in patient_order[start : start + 16]:
                draw = draws_by_patient[patient][draw_index]
                batch.append({"patient_key": patient, **draw})
            batches.append(batch)
    body: dict[str, Any] = {
        "schema_version": "eventnet_cleanroom_patient_balanced_epoch_plan_v1",
        "variant_id": variant_id,
        "outer_fold": outer_fold,
        "stage": stage,
        "epoch_index": epoch_index,
        "derived_seed": seed,
        "patient_count": len(draws_by_patient),
        "draws_per_patient": 8,
        "positive_draw_target_when_both_pools_exist": 4,
        "background_draw_target_when_both_pools_exist": 4,
        "one_tile_per_patient_per_batch": True,
        "batch_patient_maximum": 16,
        "pool_roster_sha256": _canonical_sha256(pool_rows),
        "batches": batches,
        "receipt_sha256": _CONTENT_PENDING,
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def _require_canonical_eventnet_registry(
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the repository-owned registry and replay every bound file byte."""

    validated = validate_registry(dict(registry))
    if (
        validated["implementation"]["code_sha256"]
        != eventnet_cleanroom_registry_code_sha256()
    ):
        raise ValueError("EventNet formal registry implementation binding is stale")
    project_root = Path(__file__).resolve().parents[2]
    registry_path = project_root / CONFIG_RELATIVE_PATH
    if not registry_path.is_file() or registry_path.is_symlink():
        raise ValueError("EventNet canonical registry file is unavailable")
    try:
        on_disk = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("EventNet canonical registry file is unreadable") from exc
    if on_disk != validated:
        raise PermissionError("caller-owned EventNet registry is not formal authority")
    validate_static_execution_bindings(project_root, registry=validated)
    return validated


def authorize_eventnet_fold_phase(
    detector_phase_authority: ValidatedDetectorFoldReferencePhaseAuthorityV1,
    *,
    registry: Mapping[str, Any],
) -> AuthorizedEventNetFoldPhase:
    """Adapt the shared process-sealed detector phase to EventNet.

    The serialized detector receipt is intentionally not accepted.  Patient
    grouping is recovered only from the registry-bound canonical fold-plan
    bytes, while event/reference rows come only from the shared opaque
    authority whose issuer has already replayed the required artifacts and
    reference bytes.
    """

    eventnet_registry = _require_canonical_eventnet_registry(registry)
    shared = require_validated_detector_fold_reference_phase_authority_v1(
        detector_phase_authority
    )
    validated_phase = shared.to_receipt()
    expected_fold_authority = eventnet_registry["trainer"][
        "fold_reference_authority"
    ]["registry_receipt_sha256"]
    if (
        validated_phase.get("registry_receipt_sha256")
        != expected_fold_authority
    ):
        raise ValueError("EventNet binds a different fold reference authority")
    project_root = Path(__file__).resolve().parents[2]
    plan_binding = eventnet_registry["trainer"]["fold_plan"]
    plan_path = project_root / plan_binding["path"]
    if not plan_path.is_file() or plan_path.is_symlink():
        raise ValueError("EventNet canonical fold plan is unavailable")
    if _file_sha256(plan_path) != plan_binding["file_sha256"]:
        raise ValueError("EventNet canonical fold-plan bytes drifted")
    try:
        fold_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("EventNet canonical fold plan is unreadable") from exc
    phase_plan_binding = validated_phase.get("fold_plan_binding")
    if (
        type(phase_plan_binding) is not dict
        or phase_plan_binding.get("file_sha256") != plan_binding["file_sha256"]
        or phase_plan_binding.get("plan_receipt_sha256")
        != fold_plan.get("receipt_sha256")
    ):
        raise ValueError("EventNet phase and canonical fold plan bindings differ")
    plan_rows = fold_plan.get("source_record_duration_rows")
    if type(plan_rows) is not list:
        raise ValueError("fold plan lacks its source-record denominator")
    plan_by_identity: dict[str, Mapping[str, Any]] = {}
    for row in plan_rows:
        if type(row) is not dict or not isinstance(
            row.get("analysis_identity_id"), str
        ):
            raise ValueError("fold-plan source record row is invalid")
        identity = str(row["analysis_identity_id"])
        if identity in plan_by_identity:
            raise ValueError("fold plan repeats an analysis identity")
        plan_by_identity[identity] = row
    patient_by_identity: dict[str, str] = {}
    for phase_row in validated_phase["records"]:
        identity = str(phase_row["analysis_identity_id"])
        plan_row = plan_by_identity.get(identity)
        if plan_row is None:
            raise ValueError("phase identity is absent from the fold plan")
        if (
            plan_row.get("local_edf_path")
            != phase_row["source_edf_relative_path"]
            or plan_row.get("recording_duration_seconds_fraction")
            != phase_row["recording_duration_seconds_fraction"]
        ):
            raise ValueError("phase record disagrees with fold-owned identity")
        patient = plan_row.get("local_patient_id")
        if not isinstance(patient, str) or not patient:
            raise ValueError("fold-owned patient grouping key is invalid")
        patient_by_identity[identity] = patient
    identities = sorted(patient_by_identity)
    if len(identities) != validated_phase["authorized_roster"]["recording_count"]:
        raise ValueError("EventNet phase record denominator drifted")

    authority_receipt = _content_address(
        {
            "schema_version": "eventnet_byte_replayed_fold_phase_authority_v1",
            "registry_sha256": eventnet_registry["registry_sha256"],
            "fold_reference_registry_receipt_sha256": expected_fold_authority,
            "detector_fold_phase_receipt_sha256": validated_phase["receipt_sha256"],
            "outer_fold": validated_phase["outer_fold_id"],
            "phase": validated_phase["phase"],
            "authorized_record_count": len(identities),
            "authorized_patient_count": len(set(patient_by_identity.values())),
            "analysis_identity_roster_sha256": _canonical_sha256(identities),
            "fold_owned_patient_mapping_sha256": _canonical_sha256(
                [
                    {
                        "analysis_identity_id": identity,
                        "patient_key": patient_by_identity[identity],
                    }
                    for identity in identities
                ]
            ),
            "reference_event_inventory_sha256": validated_phase[
                "reference_event_inventory_sha256"
            ],
            "reference_file_sha256_roster_sha256": validated_phase[
                "reference_file_sha256_roster_sha256"
            ],
            "shared_detector_phase_authority_type": (
                "ValidatedDetectorFoldReferencePhaseAuthorityV1"
            ),
            "shared_detector_phase_authority_required": True,
            "actual_reference_bytes_replayed": True,
            "raw_mapping_or_bare_hash_is_authority": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return AuthorizedEventNetFoldPhase(
        _phase_receipt_json=_canonical_json_bytes(validated_phase).decode("utf-8"),
        _patient_by_identity_json=_canonical_json_bytes(
            patient_by_identity
        ).decode("utf-8"),
        _authority_receipt_json=_canonical_json_bytes(authority_receipt).decode(
            "utf-8"
        ),
        _validation_seal=_FOLD_PHASE_AUTHORITY_SEAL,
    )


def _require_authorized_eventnet_fold_phase(
    value: object,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    if (
        not isinstance(value, AuthorizedEventNetFoldPhase)
        or value._validation_seal is not _FOLD_PHASE_AUTHORITY_SEAL
    ):
        raise TypeError("formal EventNet targets require an opaque fold-phase authority")
    try:
        phase = json.loads(value._phase_receipt_json)
        patient_by_identity = json.loads(value._patient_by_identity_json)
    except json.JSONDecodeError as exc:
        raise ValueError("opaque EventNet phase payload is unreadable") from exc
    receipt = _validate_content_address(
        value.receipt,
        required={
            "schema_version",
            "registry_sha256",
            "fold_reference_registry_receipt_sha256",
            "detector_fold_phase_receipt_sha256",
            "outer_fold",
            "phase",
            "authorized_record_count",
            "authorized_patient_count",
            "analysis_identity_roster_sha256",
            "fold_owned_patient_mapping_sha256",
            "reference_event_inventory_sha256",
            "reference_file_sha256_roster_sha256",
            "shared_detector_phase_authority_type",
            "shared_detector_phase_authority_required",
            "actual_reference_bytes_replayed",
            "raw_mapping_or_bare_hash_is_authority",
            "receipt_sha256",
        },
        context="authorized EventNet fold phase",
    )
    if (
        receipt["schema_version"]
        != "eventnet_byte_replayed_fold_phase_authority_v1"
        or receipt["shared_detector_phase_authority_type"]
        != "ValidatedDetectorFoldReferencePhaseAuthorityV1"
        or receipt["shared_detector_phase_authority_required"] is not True
        or receipt["actual_reference_bytes_replayed"] is not True
        or receipt["raw_mapping_or_bare_hash_is_authority"] is not False
        or phase.get("receipt_sha256")
        != receipt["detector_fold_phase_receipt_sha256"]
        or phase.get("outer_fold_id") != receipt["outer_fold"]
        or phase.get("phase") != receipt["phase"]
        or phase.get("reference_event_inventory_sha256")
        != receipt["reference_event_inventory_sha256"]
        or type(patient_by_identity) is not dict
    ):
        raise ValueError("opaque EventNet fold-phase semantics drifted")
    identities = sorted(str(row["analysis_identity_id"]) for row in phase["records"])
    if (
        set(patient_by_identity) != set(identities)
        or any(
            not isinstance(patient, str) or not patient
            for patient in patient_by_identity.values()
        )
        or receipt["authorized_record_count"] != len(identities)
        or receipt["authorized_patient_count"]
        != len(set(patient_by_identity.values()))
        or receipt["analysis_identity_roster_sha256"]
        != _canonical_sha256(identities)
        or receipt["fold_owned_patient_mapping_sha256"]
        != _canonical_sha256(
            [
                {
                    "analysis_identity_id": identity,
                    "patient_key": patient_by_identity[identity],
                }
                for identity in identities
            ]
        )
    ):
        raise ValueError("opaque EventNet fold-owned roster drifted")
    return phase, patient_by_identity, receipt


def _eventnet_pre_reference_technical_policy(variant_id: str) -> dict[str, Any]:
    profile = _variant_profile(variant_id)
    return {
        "schema_version": "eventnet_pre_reference_technical_eligibility_policy_v1",
        "variant_id": variant_id,
        "required_source_electrodes": profile["required_source_electrodes"],
        "accepted_support_profiles": (
            ["complete19"]
            if variant_id == EN19_VARIANT_ID
            else ["complete19", "lateral17"]
        ),
        "source_sampling_rate_hz_closed_interval": [1, 4096],
        "maximum_reduced_polyphase_factor": 4096,
        "minimum_provider_target_samples": MODEL_INPUT_SAMPLES,
        "minimum_fully_observed_training_tile_count": 1,
        "provider_transform_payload_replay_required_for_eligible_status": True,
        "target_reference_annotation_or_clinical_input_allowed": False,
    }


def _bind_provider_and_identity_authorities(
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
    record_identity_authority: ValidatedDetectorSignalLineageAuthority,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    provider = require_validated_detector_signal_lineage_authority(
        signal_lineage_authority
    )
    identity = require_validated_detector_signal_lineage_authority(
        record_identity_authority
    )
    provider_signal = provider["canonical_physical_signal"]
    identity_signal = identity["canonical_physical_signal"]
    analysis_identity_id = identity_signal.get("analysis_identity_id")
    if (
        provider.get("authority_tier")
        != "provider_transform_payload_replayed"
        or identity.get("authority_tier")
        != "canonical_audit_policy_route_only"
        or identity.get("provider_transform_authorized") is not False
        or not isinstance(analysis_identity_id, str)
        or not analysis_identity_id
        or provider_signal.get("source_tensor_sha256")
        != identity_signal.get("source_tensor_sha256")
        or provider_signal.get("source_header_receipt_sha256")
        != identity_signal.get("source_header_receipt_sha256")
        or provider_signal.get("source_signal_sha256")
        != identity_signal.get("source_signal_sha256")
    ):
        raise PermissionError(
            "EventNet pre-reference record identity is not externally bound "
            "to the exact provider EEG payload"
        )
    return provider, identity, analysis_identity_id


def materialize_eventnet_pre_reference_eligibility(
    referential_volts: object,
    *,
    variant_id: str,
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
    record_identity_authority: ValidatedDetectorSignalLineageAuthority,
    registry: Mapping[str, Any],
) -> EventNetPreReferenceEligibilityOutcome:
    """Materialize one target-blind route/technical outcome.

    The API deliberately has no fold-phase, event, annotation, spreadsheet or
    label input.  Integrity failures (forged/mismatched authorities or EEG
    payload) raise; deterministic support/clock/length/transform failures are
    retained as typed exclusion outcomes for the full benchmark denominator.
    """

    eventnet_registry = _require_canonical_eventnet_registry(registry)
    provider, identity, analysis_identity_id = _bind_provider_and_identity_authorities(
        signal_lineage_authority, record_identity_authority
    )
    profile = _variant_profile(variant_id)
    route = route_detector_channel_support(
        signal_lineage_authority=signal_lineage_authority
    )
    technical_policy = _eventnet_pre_reference_technical_policy(variant_id)
    expected_policy_sha256 = detector_channel_support_policy_receipt()[
        "policy_sha256"
    ]
    if route["policy_sha256"] != expected_policy_sha256:
        raise ValueError("EventNet support-route policy binding drifted")

    reason_codes: list[str] = []
    accepted_profiles = set(technical_policy["accepted_support_profiles"])
    if (
        route["support_policy_status"] != "policy_route_available"
        or route["profile_id"] not in accepted_profiles
    ):
        reason_codes.append("variant_support_route_excluded")
    usable = set(route["usable_standard_channel_ids"])
    if not set(profile["required_source_electrodes"]).issubset(usable):
        reason_codes.append("variant_EEG_QC_usable_support_incomplete")
    if provider["provider_transform_authorized"] is not True:
        reason_codes.append("provider_transform_lineage_not_authorized")

    clock = provider["common_sampling_clock_authority"]
    try:
        source_rate = Fraction(*clock["sampling_rate_fraction_hz"])
        source_sample_count = int(clock["sample_count"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("EventNet typed source clock is malformed") from exc
    if source_rate < 1 or source_rate > 4096:
        reason_codes.append("source_sampling_rate_outside_frozen_support")
        up = down = 0
        target_sample_count = 0
    else:
        ratio = Fraction(TARGET_FS_HZ, 1) / source_rate
        up, down = ratio.numerator, ratio.denominator
        if max(up, down) > 4096:
            reason_codes.append("reduced_polyphase_ratio_outside_frozen_support")
        target_sample_count = (source_sample_count * up) // down
    training_tile_count = len(enumerate_training_target_tiles(target_sample_count))
    if training_tile_count < 1:
        reason_codes.append("no_fully_observed_target_plus_context_training_tile")

    transform: EventNetTransformResult | None = None
    if not reason_codes:
        # Replay the caller's array against the sealed provider authority before
        # entering the narrowly caught deterministic transform failure path.
        verify_provider_referential_payload(
            signal_lineage_authority, referential_volts
        )
        try:
            transform = apply_full_record_transform(
                referential_volts,
                variant_id=variant_id,
                signal_lineage_authority=signal_lineage_authority,
                registry=eventnet_registry,
            )
            transform = validate_transform_result(
                transform,
                signal_lineage_authority=signal_lineage_authority,
                registry=eventnet_registry,
            )
        except (FloatingPointError, OverflowError, RuntimeError, ValueError):
            reason_codes.append("provider_transform_deterministic_technical_failure")
            transform = None
    eligible = not reason_codes and transform is not None
    transform_receipt_sha256 = (
        None if transform is None else transform.receipt["receipt_sha256"]
    )
    if transform is not None and (
        transform.receipt["variant_id"] != variant_id
        or transform.receipt["output"]["sample_count"] != target_sample_count
    ):
        raise ValueError("EventNet pre-reference transform/clock binding drifted")
    receipt = _content_address(
        {
            "schema_version": "eventnet_pre_reference_record_eligibility_outcome_v1",
            "registry_sha256": eventnet_registry["registry_sha256"],
            "variant_id": variant_id,
            "analysis_identity_id": analysis_identity_id,
            "provider_signal_lineage_authority_sha256": provider["receipt_sha256"],
            "record_identity_authority_sha256": identity["receipt_sha256"],
            "canonical_source_tensor_sha256": provider["canonical_physical_signal"][
                "source_tensor_sha256"
            ],
            "support_route_policy_sha256": expected_policy_sha256,
            "support_route_receipt_sha256": route["route_sha256"],
            "support_profile_id": route["profile_id"],
            "technical_eligibility_policy_sha256": _canonical_sha256(
                technical_policy
            ),
            "source_sampling_rate_fraction_hz": [
                source_rate.numerator,
                source_rate.denominator,
            ],
            "source_sample_count": source_sample_count,
            "provider_target_sample_count": target_sample_count,
            "fully_observed_training_tile_count": training_tile_count,
            "status": "eligible" if eligible else "typed_exclusion",
            "reason_codes": reason_codes,
            "transform_receipt_sha256": transform_receipt_sha256,
            "phase_reference_event_annotation_or_clinical_input_consumed": False,
            "must_be_frozen_before_corresponding_reference_phase_open": True,
            "raw_caller_status_or_reason_code_accepted": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return EventNetPreReferenceEligibilityOutcome(
        transform_result=transform,
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
        _validation_seal=_PRE_REFERENCE_ELIGIBILITY_SEAL,
    )


def _require_eventnet_pre_reference_eligibility(
    value: object,
) -> tuple[EventNetTransformResult | None, dict[str, Any]]:
    if (
        not isinstance(value, EventNetPreReferenceEligibilityOutcome)
        or value._validation_seal is not _PRE_REFERENCE_ELIGIBILITY_SEAL
    ):
        raise TypeError(
            "EventNet variant roster requires opaque pre-reference eligibility outcomes"
        )
    receipt = _validate_content_address(
        value.receipt,
        required={
            "schema_version",
            "registry_sha256",
            "variant_id",
            "analysis_identity_id",
            "provider_signal_lineage_authority_sha256",
            "record_identity_authority_sha256",
            "canonical_source_tensor_sha256",
            "support_route_policy_sha256",
            "support_route_receipt_sha256",
            "support_profile_id",
            "technical_eligibility_policy_sha256",
            "source_sampling_rate_fraction_hz",
            "source_sample_count",
            "provider_target_sample_count",
            "fully_observed_training_tile_count",
            "status",
            "reason_codes",
            "transform_receipt_sha256",
            "phase_reference_event_annotation_or_clinical_input_consumed",
            "must_be_frozen_before_corresponding_reference_phase_open",
            "raw_caller_status_or_reason_code_accepted",
            "receipt_sha256",
        },
        context="EventNet pre-reference eligibility outcome",
    )
    transform = value.transform_result
    status = receipt["status"]
    if (
        receipt["schema_version"]
        != "eventnet_pre_reference_record_eligibility_outcome_v1"
        or status not in {"eligible", "typed_exclusion"}
        or type(receipt["reason_codes"]) is not list
        or receipt["phase_reference_event_annotation_or_clinical_input_consumed"]
        is not False
        or receipt["must_be_frozen_before_corresponding_reference_phase_open"]
        is not True
        or receipt["raw_caller_status_or_reason_code_accepted"] is not False
        or (status == "eligible") is not (transform is not None)
        or (status == "eligible") is not (receipt["reason_codes"] == [])
        or (status == "typed_exclusion") is not bool(receipt["reason_codes"])
        or receipt["transform_receipt_sha256"]
        != (None if transform is None else transform.receipt.get("receipt_sha256"))
    ):
        raise ValueError("EventNet pre-reference eligibility semantics drifted")
    if transform is not None:
        if transform._validation_seal is not _TRANSFORM_RESULT_SEAL:
            raise TypeError("EventNet eligibility embeds a forged transform")
        output = np.asarray(transform.signal_uv)
        if (
            transform.receipt.get("variant_id") != receipt["variant_id"]
            or transform.receipt.get("registry_sha256") != receipt["registry_sha256"]
            or transform.receipt.get("detector_signal_lineage_authority_sha256")
            != receipt["provider_signal_lineage_authority_sha256"]
            or transform.receipt.get("canonical_source_tensor_sha256")
            != receipt["canonical_source_tensor_sha256"]
            or transform.receipt.get("output", {}).get("sample_count")
            != receipt["provider_target_sample_count"]
            or transform.receipt.get("output", {}).get("payload_receipt")
            != _payload_receipt(
                output,
                semantic="EventNet_cleanroom_provider_native_full_record_uV",
            )
        ):
            raise ValueError("EventNet embedded pre-reference transform drifted")
    return transform, receipt


def authorize_eventnet_variant_training_roster(
    phase_authority: AuthorizedEventNetFoldPhase,
    pre_reference_outcomes: Sequence[EventNetPreReferenceEligibilityOutcome],
    *,
    variant_id: str,
    registry: Mapping[str, Any],
) -> AuthorizedEventNetVariantTrainingRoster:
    """Intersect the exact phase with route and technical eligibility.

    All phase records must have exactly one process-sealed pre-reference
    outcome.  Ineligible records are retained as typed exclusions; they are
    omitted only from the gradient roster, never from the all-record
    prediction-first benchmark denominator.
    """

    eventnet_registry = _require_canonical_eventnet_registry(registry)
    phase, patient_by_identity, phase_receipt = (
        _require_authorized_eventnet_fold_phase(phase_authority)
    )
    _variant_profile(variant_id)
    outcomes_by_identity: dict[str, dict[str, Any]] = {}
    for outcome in pre_reference_outcomes:
        _transform, receipt = _require_eventnet_pre_reference_eligibility(outcome)
        identity = str(receipt["analysis_identity_id"])
        if identity in outcomes_by_identity:
            raise ValueError("EventNet pre-reference roster repeats an identity")
        if (
            receipt["registry_sha256"] != eventnet_registry["registry_sha256"]
            or receipt["variant_id"] != variant_id
        ):
            raise ValueError("EventNet pre-reference outcome method binding drifted")
        outcomes_by_identity[identity] = receipt
    expected_identities = {
        str(row["analysis_identity_id"]) for row in phase["records"]
    }
    if set(outcomes_by_identity) != expected_identities:
        missing = sorted(expected_identities.difference(outcomes_by_identity))
        extra = sorted(set(outcomes_by_identity).difference(expected_identities))
        raise PermissionError(
            "EventNet complete phase-by-variant pre-reference Cartesian set was "
            f"not supplied; missing={missing}, extra={extra}"
        )
    support_policy_sha256 = detector_channel_support_policy_receipt()[
        "policy_sha256"
    ]
    technical_policy_sha256 = _canonical_sha256(
        _eventnet_pre_reference_technical_policy(variant_id)
    )
    eligible_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    for identity in sorted(expected_identities):
        outcome = outcomes_by_identity[identity]
        if (
            outcome["support_route_policy_sha256"] != support_policy_sha256
            or outcome["technical_eligibility_policy_sha256"]
            != technical_policy_sha256
        ):
            raise ValueError("EventNet pre-reference policy binding drifted")
        common = {
            "analysis_identity_id": identity,
            "fold_owned_patient_key": patient_by_identity[identity],
            "pre_reference_eligibility_receipt_sha256": outcome["receipt_sha256"],
            "support_route_receipt_sha256": outcome[
                "support_route_receipt_sha256"
            ],
            "technical_eligibility_receipt_sha256": outcome["receipt_sha256"],
        }
        if outcome["status"] == "eligible":
            eligible_rows.append(
                {
                    **common,
                    "provider_signal_lineage_authority_sha256": outcome[
                        "provider_signal_lineage_authority_sha256"
                    ],
                    "record_identity_authority_sha256": outcome[
                        "record_identity_authority_sha256"
                    ],
                    "transform_receipt_sha256": outcome[
                        "transform_receipt_sha256"
                    ],
                }
            )
        else:
            exclusion_rows.append(
                {
                    **common,
                    "terminal_status": "technical_or_support_exclusion",
                    "reason_codes": outcome["reason_codes"],
                    "retained_in_full_prediction_first_benchmark_denominator": True,
                }
            )
    roster = {
        "eligible_records": eligible_rows,
        "typed_exclusions": exclusion_rows,
    }
    receipt = _content_address(
        {
            "schema_version": "eventnet_target_blind_variant_training_roster_authority_v1",
            "registry_sha256": eventnet_registry["registry_sha256"],
            "variant_id": variant_id,
            "outer_fold": phase_receipt["outer_fold"],
            "phase": phase_receipt["phase"],
            "detector_fold_phase_receipt_sha256": phase_receipt[
                "detector_fold_phase_receipt_sha256"
            ],
            "support_route_policy_sha256": support_policy_sha256,
            "pre_reference_technical_eligibility_policy_sha256": technical_policy_sha256,
            "phase_record_count": len(expected_identities),
            "eligible_record_count": len(eligible_rows),
            "eligible_patient_count": len(
                {row["fold_owned_patient_key"] for row in eligible_rows}
            ),
            "excluded_record_count": len(exclusion_rows),
            "eligible_analysis_identity_roster_sha256": _canonical_sha256(
                sorted(row["analysis_identity_id"] for row in eligible_rows)
            ),
            "typed_exclusion_ledger_sha256": _canonical_sha256(exclusion_rows),
            "all_phase_records_accounted_for": True,
            "prediction_first_denominator_preserved": True,
            "phase_reference_events_used_for_route_or_eligibility": False,
            "caller_owned_subset_accepted": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return AuthorizedEventNetVariantTrainingRoster(
        _roster_json=_canonical_json_bytes(roster).decode("utf-8"),
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
        _validation_seal=_VARIANT_TRAINING_ROSTER_AUTHORITY_SEAL,
    )


def _require_authorized_eventnet_variant_training_roster(
    value: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a future process-sealed roster without accepting plain JSON."""

    if (
        not isinstance(value, AuthorizedEventNetVariantTrainingRoster)
        or value._validation_seal is not _VARIANT_TRAINING_ROSTER_AUTHORITY_SEAL
    ):
        raise TypeError(
            "formal EventNet training requires an opaque variant-training roster"
        )
    try:
        roster = json.loads(value._roster_json)
    except json.JSONDecodeError as exc:
        raise ValueError("opaque EventNet variant roster is unreadable") from exc
    receipt = _validate_content_address(
        value.receipt,
        required={
            "schema_version",
            "registry_sha256",
            "variant_id",
            "outer_fold",
            "phase",
            "detector_fold_phase_receipt_sha256",
            "support_route_policy_sha256",
            "pre_reference_technical_eligibility_policy_sha256",
            "phase_record_count",
            "eligible_record_count",
            "eligible_patient_count",
            "excluded_record_count",
            "eligible_analysis_identity_roster_sha256",
            "typed_exclusion_ledger_sha256",
            "all_phase_records_accounted_for",
            "prediction_first_denominator_preserved",
            "phase_reference_events_used_for_route_or_eligibility",
            "caller_owned_subset_accepted",
            "receipt_sha256",
        },
        context="authorized EventNet variant-training roster",
    )
    if (
        receipt["schema_version"]
        != "eventnet_target_blind_variant_training_roster_authority_v1"
        or receipt["all_phase_records_accounted_for"] is not True
        or receipt["prediction_first_denominator_preserved"] is not True
        or receipt["phase_reference_events_used_for_route_or_eligibility"]
        is not False
        or receipt["caller_owned_subset_accepted"] is not False
        or type(roster) is not dict
        or set(roster) != {"eligible_records", "typed_exclusions"}
        or type(roster["eligible_records"]) is not list
        or type(roster["typed_exclusions"]) is not list
    ):
        raise ValueError("opaque EventNet variant-training roster drifted")
    eligible_fields = {
        "analysis_identity_id",
        "fold_owned_patient_key",
        "pre_reference_eligibility_receipt_sha256",
        "support_route_receipt_sha256",
        "technical_eligibility_receipt_sha256",
        "provider_signal_lineage_authority_sha256",
        "record_identity_authority_sha256",
        "transform_receipt_sha256",
    }
    if any(
        type(row) is not dict or set(row) != eligible_fields
        for row in roster["eligible_records"]
    ):
        raise ValueError("opaque EventNet eligible roster row drifted")
    identities = [row["analysis_identity_id"] for row in roster["eligible_records"]]
    patients = [row["fold_owned_patient_key"] for row in roster["eligible_records"]]
    exclusion_fields = {
        "analysis_identity_id",
        "fold_owned_patient_key",
        "pre_reference_eligibility_receipt_sha256",
        "support_route_receipt_sha256",
        "technical_eligibility_receipt_sha256",
        "terminal_status",
        "reason_codes",
        "retained_in_full_prediction_first_benchmark_denominator",
    }
    excluded_identities = [
        row.get("analysis_identity_id")
        for row in roster["typed_exclusions"]
        if type(row) is dict
    ]
    if (
        any(not isinstance(identity, str) or not identity for identity in identities)
        or any(not isinstance(patient, str) or not patient for patient in patients)
        or len(set(identities)) != len(identities)
        or len(set(excluded_identities)) != len(excluded_identities)
        or any(
            type(row) is not dict
            or set(row) != exclusion_fields
            or row["terminal_status"] != "technical_or_support_exclusion"
            or type(row["reason_codes"]) is not list
            or not row["reason_codes"]
            or row[
                "retained_in_full_prediction_first_benchmark_denominator"
            ]
            is not True
            for row in roster["typed_exclusions"]
        )
        or set(identities).intersection(
            str(identity) for identity in excluded_identities
        )
        or receipt["eligible_record_count"] != len(identities)
        or receipt["eligible_patient_count"] != len(set(patients))
        or receipt["excluded_record_count"] != len(roster["typed_exclusions"])
        or receipt["phase_record_count"]
        != len(identities) + len(roster["typed_exclusions"])
        or receipt["eligible_analysis_identity_roster_sha256"]
        != _canonical_sha256(sorted(identities))
        or receipt["typed_exclusion_ledger_sha256"]
        != _canonical_sha256(roster["typed_exclusions"])
    ):
        raise ValueError("opaque EventNet variant roster payload drifted")
    return roster, receipt


def _phase_record(
    phase: Mapping[str, Any], *, analysis_identity_id: str
) -> dict[str, Any]:
    rows = [
        deepcopy(row)
        for row in phase["records"]
        if row["analysis_identity_id"] == analysis_identity_id
    ]
    if len(rows) != 1:
        raise PermissionError("record is absent or duplicated in the authorized phase")
    return rows[0]


def _authorized_record_context(
    phase_authority: AuthorizedEventNetFoldPhase,
    variant_roster_authority: AuthorizedEventNetVariantTrainingRoster,
    transform_result: EventNetTransformResult,
    *,
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
    record_identity_authority: ValidatedDetectorSignalLineageAuthority,
    registry: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
    dict[str, Any],
    EventNetTransformResult,
    dict[str, Any],
    dict[str, Any],
]:
    eventnet_registry = _require_canonical_eventnet_registry(registry)
    phase, patient_by_identity, phase_authority_receipt = (
        _require_authorized_eventnet_fold_phase(phase_authority)
    )
    variant_roster, variant_roster_receipt = (
        _require_authorized_eventnet_variant_training_roster(
            variant_roster_authority
        )
    )
    if phase_authority_receipt["registry_sha256"] != eventnet_registry["registry_sha256"]:
        raise ValueError("EventNet phase and method registry bindings differ")
    if (
        variant_roster_receipt["registry_sha256"]
        != phase_authority_receipt["registry_sha256"]
        or variant_roster_receipt["outer_fold"]
        != phase_authority_receipt["outer_fold"]
        or variant_roster_receipt["phase"] != phase_authority_receipt["phase"]
        or variant_roster_receipt["detector_fold_phase_receipt_sha256"]
        != phase_authority_receipt["detector_fold_phase_receipt_sha256"]
    ):
        raise ValueError("EventNet variant roster and fold phase bindings differ")
    lineage = require_validated_detector_signal_lineage_authority(
        signal_lineage_authority
    )
    identity_lineage = require_validated_detector_signal_lineage_authority(
        record_identity_authority
    )
    if (
        lineage["authority_tier"] != "provider_transform_payload_replayed"
        or identity_lineage["authority_tier"]
        != "canonical_audit_policy_route_only"
        or identity_lineage["provider_transform_authorized"] is not False
        or lineage["canonical_physical_signal"]["source_tensor_sha256"]
        != identity_lineage["canonical_physical_signal"]["source_tensor_sha256"]
        or lineage["canonical_physical_signal"]["source_header_receipt_sha256"]
        != identity_lineage["canonical_physical_signal"][
            "source_header_receipt_sha256"
        ]
    ):
        raise PermissionError(
            "EventNet training record identity is not externally bound to the "
            "exact provider EEG tensor"
        )
    identity = identity_lineage["canonical_physical_signal"][
        "analysis_identity_id"
    ]
    phase_record = _phase_record(phase, analysis_identity_id=identity)
    patient_key = patient_by_identity[identity]
    validated_transform = validate_transform_result(
        transform_result,
        signal_lineage_authority=signal_lineage_authority,
        registry=eventnet_registry,
    )
    receipt = validated_transform.receipt
    eligible_rows = [
        row
        for row in variant_roster["eligible_records"]
        if row["analysis_identity_id"] == identity
    ]
    if (
        len(eligible_rows) != 1
        or eligible_rows[0]["fold_owned_patient_key"] != patient_key
        or eligible_rows[0]["provider_signal_lineage_authority_sha256"]
        != lineage["receipt_sha256"]
        or eligible_rows[0]["record_identity_authority_sha256"]
        != identity_lineage["receipt_sha256"]
        or eligible_rows[0]["transform_receipt_sha256"]
        != receipt["receipt_sha256"]
        or variant_roster_receipt["variant_id"] != receipt["variant_id"]
        or
        receipt["detector_signal_lineage_authority_sha256"]
        != lineage["receipt_sha256"]
        or receipt["output"]["sample_count"]
        != validated_transform.signal_uv.shape[1]
        or receipt["output"]["sampling_rate_fraction_hz"] != [TARGET_FS_HZ, 1]
    ):
        raise ValueError("EventNet transform/record binding drifted")
    clock = lineage["common_sampling_clock_authority"]
    source_duration = Fraction(clock["sample_count"] * clock["sampling_rate_fraction_hz"][1], clock["sampling_rate_fraction_hz"][0])
    if [source_duration.numerator, source_duration.denominator] != phase_record[
        "recording_duration_seconds_fraction"
    ]:
        raise ValueError("EventNet signal clock disagrees with fold-owned record duration")
    return (
        phase,
        phase_record,
        patient_key,
        phase_authority_receipt,
        validated_transform,
        identity_lineage,
        variant_roster_receipt,
    )


def _project_phase_events_to_provider_clock(
    phase_record: Mapping[str, Any], *, provider_sample_count: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project exact authorized events onto the floor-length provider clock."""

    provider_duration = Fraction(provider_sample_count, TARGET_FS_HZ)
    source_duration = Fraction(*phase_record["recording_duration_seconds_fraction"])
    if provider_duration > source_duration:
        raise ValueError("EventNet provider clock exceeds the authorized source duration")
    projected: list[dict[str, Any]] = []
    unobservable_tail_events: list[int] = []
    right_clamped_events: list[int] = []
    for index, event in enumerate(phase_record["seizure_intervals"]):
        start = Fraction(str(event["start_seconds"]))
        stop = Fraction(str(event["stop_seconds"]))
        if start >= provider_duration:
            unobservable_tail_events.append(index)
            continue
        projected_stop = min(stop, provider_duration)
        if projected_stop <= start:
            unobservable_tail_events.append(index)
            continue
        if projected_stop < stop:
            right_clamped_events.append(index)
        projected.append(
            {
                "start_seconds": str(start),
                "stop_seconds": str(projected_stop),
                "left_censored": start == 0,
                "right_censored": stop >= source_duration or projected_stop < stop,
            }
        )
    ledger = {
        "source_event_count": len(phase_record["seizure_intervals"]),
        "provider_clock_event_count": len(projected),
        "right_boundary_clamped_event_indices": right_clamped_events,
        "entirely_unobservable_floor_tail_event_indices": unobservable_tail_events,
        "no_silent_event_deletion": (
            len(projected) + len(unobservable_tail_events)
            == len(phase_record["seizure_intervals"])
        ),
    }
    if ledger["no_silent_event_deletion"] is not True:
        raise RuntimeError("EventNet event projection denominator drifted")
    return projected, ledger


def authorize_eventnet_target_bundle(
    phase_authority: AuthorizedEventNetFoldPhase,
    variant_roster_authority: AuthorizedEventNetVariantTrainingRoster,
    transform_result: EventNetTransformResult,
    model_tile: EventNetModelTile,
    *,
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
    record_identity_authority: ValidatedDetectorSignalLineageAuthority,
    registry: Mapping[str, Any],
) -> AuthorizedEventNetTargetBundle:
    """Create one formal target without accepting caller-owned event rows."""

    (
        _phase,
        phase_record,
        patient_key,
        phase_authority_receipt,
        validated_transform,
        identity_lineage,
        variant_roster_receipt,
    ) = _authorized_record_context(
        phase_authority,
        variant_roster_authority,
        transform_result,
        signal_lineage_authority=signal_lineage_authority,
        record_identity_authority=record_identity_authority,
        registry=registry,
    )
    tile = _require_materialized_model_tile(
        model_tile, provider_signal_uv=validated_transform.signal_uv
    )
    tile_receipt = tile.receipt
    target_start = int(tile_receipt["target_start_sample"])
    if (
        tile_receipt["training_loss_eligible"] is not True
        or tile_receipt["training_forward_with_any_padding_allowed"] is not False
        or tile_receipt["left_context_padding_samples"] != 0
        or tile_receipt["right_context_or_tail_padding_samples"] != 0
        or (target_start, TARGET_TILE_SAMPLES)
        not in enumerate_training_target_tiles(validated_transform.signal_uv.shape[1])
    ):
        raise PermissionError("padded, short, tail, or off-grid tile cannot train EventNet")
    projected_events, event_projection_ledger = _project_phase_events_to_provider_clock(
        phase_record, provider_sample_count=validated_transform.signal_uv.shape[1]
    )
    targets = build_eventnet_targets_pure_primitive(
        projected_events,
        record_sample_count=validated_transform.signal_uv.shape[1],
        target_start_sample=target_start,
        actual_observed_target_samples=TARGET_TILE_SAMPLES,
    )
    variant_id = str(validated_transform.receipt["variant_id"])
    target_receipt = targets.receipt
    receipt = _content_address(
        {
            "schema_version": "eventnet_authorized_target_bundle_v1",
            "registry_sha256": phase_authority_receipt["registry_sha256"],
            "variant_id": variant_id,
            "outer_fold": phase_authority_receipt["outer_fold"],
            "phase": phase_authority_receipt["phase"],
            "detector_fold_phase_receipt_sha256": phase_authority_receipt[
                "detector_fold_phase_receipt_sha256"
            ],
            "variant_training_roster_receipt_sha256": variant_roster_receipt[
                "receipt_sha256"
            ],
            "analysis_identity_id": phase_record["analysis_identity_id"],
            "record_identity_authority_sha256": identity_lineage[
                "receipt_sha256"
            ],
            "fold_owned_patient_key": patient_key,
            "patient_key_used_as_model_feature": False,
            "record_event_inventory_sha256": phase_record[
                "event_inventory_sha256"
            ],
            "reference_file_sha256": phase_record["reference_file_sha256"],
            "event_projection_ledger": event_projection_ledger,
            "transform_receipt_sha256": validated_transform.receipt[
                "receipt_sha256"
            ],
            "transform_output_payload_sha256": validated_transform.receipt[
                "output"
            ]["payload_receipt"]["payload_sha256"],
            "model_tile_receipt_sha256": tile_receipt["receipt_sha256"],
            "model_input_payload_sha256": tile_receipt[
                "model_input_payload_receipt"
            ]["payload_sha256"],
            "target_start_sample": target_start,
            "target_stop_sample_exclusive": target_start + TARGET_TILE_SAMPLES,
            "training_loss_eligible": True,
            "padding_entered_training_forward": False,
            "target_receipt_sha256": target_receipt["receipt_sha256"],
            "center_target_payload_receipt": target_receipt[
                "center_target_payload_receipt"
            ],
            "duration_target_payload_receipt": target_receipt[
                "duration_target_payload_receipt"
            ],
            "center_loss_mask_payload_receipt": target_receipt[
                "center_loss_mask_payload_receipt"
            ],
            "duration_loss_mask_payload_receipt": target_receipt[
                "duration_loss_mask_payload_receipt"
            ],
            "distinct_center_count": targets.distinct_center_count,
            "raw_caller_events_masks_counts_or_patient_keys_accepted": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return AuthorizedEventNetTargetBundle(
        center_target=targets.center_target,
        duration_target=targets.duration_target,
        center_loss_mask=targets.center_loss_mask,
        duration_loss_mask=targets.duration_loss_mask,
        distinct_center_count=targets.distinct_center_count,
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
        _validation_seal=_TARGET_BUNDLE_SEAL,
    )


def _require_authorized_eventnet_target_bundle(
    value: object,
) -> AuthorizedEventNetTargetBundle:
    if (
        not isinstance(value, AuthorizedEventNetTargetBundle)
        or value._validation_seal is not _TARGET_BUNDLE_SEAL
    ):
        raise TypeError("formal EventNet loss requires an opaque target bundle")
    receipt = _validate_content_address(
        value.receipt,
        required={
            "schema_version",
            "registry_sha256",
            "variant_id",
            "outer_fold",
            "phase",
            "detector_fold_phase_receipt_sha256",
            "variant_training_roster_receipt_sha256",
            "analysis_identity_id",
            "record_identity_authority_sha256",
            "fold_owned_patient_key",
            "patient_key_used_as_model_feature",
            "record_event_inventory_sha256",
            "reference_file_sha256",
            "event_projection_ledger",
            "transform_receipt_sha256",
            "transform_output_payload_sha256",
            "model_tile_receipt_sha256",
            "model_input_payload_sha256",
            "target_start_sample",
            "target_stop_sample_exclusive",
            "training_loss_eligible",
            "padding_entered_training_forward",
            "target_receipt_sha256",
            "center_target_payload_receipt",
            "duration_target_payload_receipt",
            "center_loss_mask_payload_receipt",
            "duration_loss_mask_payload_receipt",
            "distinct_center_count",
            "raw_caller_events_masks_counts_or_patient_keys_accepted",
            "receipt_sha256",
        },
        context="authorized EventNet target bundle",
    )
    center = np.asarray(value.center_target)
    duration = np.asarray(value.duration_target)
    center_mask = np.asarray(value.center_loss_mask)
    duration_mask = np.asarray(value.duration_loss_mask)
    if (
        receipt["schema_version"] != "eventnet_authorized_target_bundle_v1"
        or receipt["training_loss_eligible"] is not True
        or receipt["padding_entered_training_forward"] is not False
        or receipt["patient_key_used_as_model_feature"] is not False
        or receipt["raw_caller_events_masks_counts_or_patient_keys_accepted"]
        is not False
        or center.shape != (TARGET_TILE_SAMPLES,)
        or duration.shape != center.shape
        or center_mask.shape != center.shape
        or duration_mask.shape != center.shape
        or receipt["center_target_payload_receipt"]
        != _payload_receipt(center, semantic="EventNet_center_target")
        or receipt["duration_target_payload_receipt"]
        != _payload_receipt(
            duration, semantic="EventNet_duration_fraction_target"
        )
        or receipt["center_loss_mask_payload_receipt"]
        != _payload_receipt(
            center_mask, semantic="EventNet_center_loss_opportunity_mask"
        )
        or receipt["duration_loss_mask_payload_receipt"]
        != _payload_receipt(
            duration_mask, semantic="EventNet_duration_loss_center_mask"
        )
        or receipt["distinct_center_count"] != value.distinct_center_count
        or int(np.count_nonzero(duration_mask)) != value.distinct_center_count
        or np.any(duration_mask & ~center_mask)
    ):
        raise ValueError("authorized EventNet target bundle payload drifted")
    return value


def bind_eventnet_training_logits(
    center_logits: object,
    duration_logits: object,
    *,
    target_bundles: Sequence[AuthorizedEventNetTargetBundle],
    model_tiles: Sequence[EventNetModelTile],
) -> BoundEventNetTrainingLogits:
    """Bind differentiable logits to an exact ordered opaque tile roster."""

    center = _as_batch_time_tensor(center_logits, context="center logits")
    duration = _as_batch_time_tensor(duration_logits, context="duration logits")
    if (
        not center.is_floating_point()
        or not duration.is_floating_point()
        or center.shape != duration.shape
        or center.shape != (len(target_bundles), TARGET_TILE_SAMPLES)
        or len(model_tiles) != len(target_bundles)
        or not bool(torch.isfinite(center).all())
        or not bool(torch.isfinite(duration).all())
    ):
        raise ValueError("EventNet formal logits/batch geometry drifted")
    bundle_hashes: list[str] = []
    tile_hashes: list[str] = []
    for bundle, tile_value in zip(target_bundles, model_tiles):
        checked_bundle = _require_authorized_eventnet_target_bundle(bundle)
        checked_tile = _require_materialized_model_tile(tile_value)
        if (
            checked_tile.receipt["receipt_sha256"]
            != checked_bundle.receipt["model_tile_receipt_sha256"]
            or checked_tile.receipt["model_input_payload_receipt"][
                "payload_sha256"
            ]
            != checked_bundle.receipt["model_input_payload_sha256"]
        ):
            raise ValueError("EventNet logits were bound to the wrong training tile")
        bundle_hashes.append(checked_bundle.receipt["receipt_sha256"])
        tile_hashes.append(checked_tile.receipt["receipt_sha256"])
    receipt = _content_address(
        {
            "schema_version": "eventnet_bound_training_logits_v1",
            "ordered_target_bundle_receipt_sha256": bundle_hashes,
            "ordered_model_tile_receipt_sha256": tile_hashes,
            "center_logits_payload_receipt": _tensor_payload_receipt(
                center, semantic="EventNet_center_training_logits"
            ),
            "duration_logits_payload_receipt": _tensor_payload_receipt(
                duration, semantic="EventNet_duration_training_logits"
            ),
            "caller_owned_masks_counts_or_patient_keys_accepted": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return BoundEventNetTrainingLogits(
        center_logits=center,
        duration_logits=duration,
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
        _validation_seal=_BOUND_LOGITS_SEAL,
    )


def _require_bound_eventnet_training_logits(
    value: object,
) -> BoundEventNetTrainingLogits:
    if (
        not isinstance(value, BoundEventNetTrainingLogits)
        or value._validation_seal is not _BOUND_LOGITS_SEAL
    ):
        raise TypeError("formal EventNet loss requires opaque tile-bound logits")
    receipt = _validate_content_address(
        value.receipt,
        required={
            "schema_version",
            "ordered_target_bundle_receipt_sha256",
            "ordered_model_tile_receipt_sha256",
            "center_logits_payload_receipt",
            "duration_logits_payload_receipt",
            "caller_owned_masks_counts_or_patient_keys_accepted",
            "receipt_sha256",
        },
        context="bound EventNet training logits",
    )
    if (
        receipt["schema_version"] != "eventnet_bound_training_logits_v1"
        or receipt["caller_owned_masks_counts_or_patient_keys_accepted"]
        is not False
        or receipt["center_logits_payload_receipt"]
        != _tensor_payload_receipt(
            value.center_logits, semantic="EventNet_center_training_logits"
        )
        or receipt["duration_logits_payload_receipt"]
        != _tensor_payload_receipt(
            value.duration_logits, semantic="EventNet_duration_training_logits"
        )
    ):
        raise ValueError("bound EventNet training logits drifted")
    return value


def eventnet_authorized_multitask_loss(
    bound_logits: BoundEventNetTrainingLogits,
    *,
    target_bundles: Sequence[AuthorizedEventNetTargetBundle],
) -> EventNetLossResult:
    """Formal loss path: no raw event, mask, count, or patient-key arguments."""

    checked_logits = _require_bound_eventnet_training_logits(bound_logits)
    bundles = [
        _require_authorized_eventnet_target_bundle(bundle)
        for bundle in target_bundles
    ]
    expected_hashes = [bundle.receipt["receipt_sha256"] for bundle in bundles]
    if (
        checked_logits.receipt["ordered_target_bundle_receipt_sha256"]
        != expected_hashes
        or len(bundles) != int(checked_logits.center_logits.shape[0])
    ):
        raise ValueError("EventNet formal loss target/logit roster drifted")
    return eventnet_multitask_loss_from_logits_pure_primitive(
        checked_logits.center_logits,
        checked_logits.duration_logits,
        np.stack([bundle.center_target for bundle in bundles]),
        np.stack([bundle.duration_target for bundle in bundles]),
        np.stack([bundle.center_loss_mask for bundle in bundles]),
        np.stack([bundle.duration_loss_mask for bundle in bundles]),
        [bundle.distinct_center_count for bundle in bundles],
        patient_keys=[
            str(bundle.receipt["fold_owned_patient_key"]) for bundle in bundles
        ],
    )


def authorize_eventnet_record_tile_pool(
    phase_authority: AuthorizedEventNetFoldPhase,
    variant_roster_authority: AuthorizedEventNetVariantTrainingRoster,
    transform_result: EventNetTransformResult,
    *,
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
    record_identity_authority: ValidatedDetectorSignalLineageAuthority,
    registry: Mapping[str, Any],
) -> AuthorizedEventNetRecordPool:
    """Build a full-record pool from an opaque phase; raw events are forbidden."""

    (
        _phase,
        phase_record,
        patient_key,
        phase_authority_receipt,
        validated_transform,
        identity_lineage,
        variant_roster_receipt,
    ) = _authorized_record_context(
        phase_authority,
        variant_roster_authority,
        transform_result,
        signal_lineage_authority=signal_lineage_authority,
        record_identity_authority=record_identity_authority,
        registry=registry,
    )
    projected_events, event_projection_ledger = _project_phase_events_to_provider_clock(
        phase_record, provider_sample_count=validated_transform.signal_uv.shape[1]
    )
    variant_id = str(validated_transform.receipt["variant_id"])
    pool = build_record_tile_pools_pure_primitive(
        projected_events,
        record_key=phase_record["analysis_identity_id"],
        record_sample_count=validated_transform.signal_uv.shape[1],
    )
    receipt = _content_address(
        {
            "schema_version": "eventnet_authorized_record_tile_pool_v1",
            "registry_sha256": phase_authority_receipt["registry_sha256"],
            "variant_id": variant_id,
            "outer_fold": phase_authority_receipt["outer_fold"],
            "phase": phase_authority_receipt["phase"],
            "detector_fold_phase_receipt_sha256": phase_authority_receipt[
                "detector_fold_phase_receipt_sha256"
            ],
            "variant_training_roster_receipt_sha256": variant_roster_receipt[
                "receipt_sha256"
            ],
            "analysis_identity_id": phase_record["analysis_identity_id"],
            "record_identity_authority_sha256": identity_lineage[
                "receipt_sha256"
            ],
            "fold_owned_patient_key": patient_key,
            "record_event_inventory_sha256": phase_record[
                "event_inventory_sha256"
            ],
            "event_projection_ledger": event_projection_ledger,
            "transform_receipt_sha256": validated_transform.receipt[
                "receipt_sha256"
            ],
            "pool_receipt_sha256": pool["receipt_sha256"],
            "eligible_tile_count": len(pool["positive"]) + len(pool["background"]),
            "transition_excluded_tile_count": len(
                pool["transition_excluded_from_background"]
            ),
            "raw_caller_events_or_patient_roster_accepted": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return AuthorizedEventNetRecordPool(
        _pool_json=_canonical_json_bytes(pool).decode("utf-8"),
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
        _validation_seal=_RECORD_POOL_AUTHORITY_SEAL,
    )


def _require_authorized_eventnet_record_pool(
    value: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        not isinstance(value, AuthorizedEventNetRecordPool)
        or value._validation_seal is not _RECORD_POOL_AUTHORITY_SEAL
    ):
        raise TypeError("formal EventNet epoch plan requires opaque record pools")
    try:
        pool = json.loads(value._pool_json)
    except json.JSONDecodeError as exc:
        raise ValueError("opaque EventNet record pool is unreadable") from exc
    receipt = _validate_content_address(
        value.receipt,
        required={
            "schema_version",
            "registry_sha256",
            "variant_id",
            "outer_fold",
            "phase",
            "detector_fold_phase_receipt_sha256",
            "variant_training_roster_receipt_sha256",
            "analysis_identity_id",
            "record_identity_authority_sha256",
            "fold_owned_patient_key",
            "record_event_inventory_sha256",
            "event_projection_ledger",
            "transform_receipt_sha256",
            "pool_receipt_sha256",
            "eligible_tile_count",
            "transition_excluded_tile_count",
            "raw_caller_events_or_patient_roster_accepted",
            "receipt_sha256",
        },
        context="authorized EventNet record pool",
    )
    pending_pool = deepcopy(pool)
    supplied_pool_hash = pending_pool.get("receipt_sha256")
    pending_pool["receipt_sha256"] = _CONTENT_PENDING
    if (
        receipt["schema_version"] != "eventnet_authorized_record_tile_pool_v1"
        or receipt["raw_caller_events_or_patient_roster_accepted"] is not False
        or receipt["pool_receipt_sha256"] != supplied_pool_hash
        or supplied_pool_hash != _canonical_sha256(pending_pool)
        or pool.get("record_key") != receipt["analysis_identity_id"]
        or receipt["eligible_tile_count"]
        != len(pool.get("positive", [])) + len(pool.get("background", []))
    ):
        raise ValueError("authorized EventNet record pool drifted")
    return pool, receipt


def build_authorized_patient_balanced_epoch_plan(
    phase_authority: AuthorizedEventNetFoldPhase,
    variant_roster_authority: AuthorizedEventNetVariantTrainingRoster,
    record_pools: Sequence[AuthorizedEventNetRecordPool],
    *,
    variant_id: str,
    outer_fold: int,
    stage: str,
    epoch_index: int,
) -> dict[str, Any]:
    """Plan from the complete target-blind variant-eligible denominator."""

    phase, patient_by_identity, phase_receipt = (
        _require_authorized_eventnet_fold_phase(phase_authority)
    )
    variant_roster, variant_roster_receipt = (
        _require_authorized_eventnet_variant_training_roster(
            variant_roster_authority
        )
    )
    expected_phase = {"selection": "selection_fit", "final_refit": "final_refit"}.get(
        stage
    )
    if (
        expected_phase is None
        or phase_receipt["outer_fold"] != outer_fold
        or phase_receipt["phase"] != expected_phase
        or variant_roster_receipt["variant_id"] != variant_id
        or variant_roster_receipt["outer_fold"] != outer_fold
        or variant_roster_receipt["phase"] != expected_phase
        or variant_roster_receipt["registry_sha256"]
        != phase_receipt["registry_sha256"]
        or variant_roster_receipt["detector_fold_phase_receipt_sha256"]
        != phase_receipt["detector_fold_phase_receipt_sha256"]
    ):
        raise PermissionError("EventNet epoch stage lacks the matching fold phase")
    pools_by_identity: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for pool_value in record_pools:
        pool, receipt = _require_authorized_eventnet_record_pool(pool_value)
        identity = str(receipt["analysis_identity_id"])
        if identity in pools_by_identity:
            raise ValueError("EventNet epoch repeats a record pool")
        if (
            receipt["registry_sha256"] != phase_receipt["registry_sha256"]
            or receipt["variant_id"] != variant_id
            or receipt["outer_fold"] != outer_fold
            or receipt["phase"] != expected_phase
            or receipt["detector_fold_phase_receipt_sha256"]
            != phase_receipt["detector_fold_phase_receipt_sha256"]
            or receipt["variant_training_roster_receipt_sha256"]
            != variant_roster_receipt["receipt_sha256"]
            or receipt["fold_owned_patient_key"] != patient_by_identity.get(identity)
        ):
            raise ValueError("EventNet record pool authority disagrees with epoch phase")
        pools_by_identity[identity] = (pool, receipt)
    expected_identities = {
        str(row["analysis_identity_id"])
        for row in variant_roster["eligible_records"]
    }
    if set(pools_by_identity) != expected_identities:
        missing = sorted(expected_identities.difference(pools_by_identity))
        extra = sorted(set(pools_by_identity).difference(expected_identities))
        raise PermissionError(
            "EventNet variant-eligible record denominator was deleted; "
            f"missing={missing}, extra={extra}"
        )

    patient_pools: dict[str, dict[str, list[str]]] = {}
    globally_seen_tiles: set[str] = set()
    for identity in sorted(expected_identities):
        pool, receipt = pools_by_identity[identity]
        patient = str(receipt["fold_owned_patient_key"])
        aggregate = patient_pools.setdefault(
            patient, {"positive": [], "background": []}
        )
        for pool_name in ("positive", "background"):
            for tile in pool[pool_name]:
                if tile in globally_seen_tiles:
                    raise ValueError("EventNet tile identity collides across records")
                globally_seen_tiles.add(tile)
                aggregate[pool_name].append(tile)
    no_eligible = sorted(
        patient
        for patient, pools in patient_pools.items()
        if not pools["positive"] and not pools["background"]
    )
    if no_eligible:
        raise PermissionError(
            "complete fold contains patients with no fully observed EventNet tile; "
            f"explicit cohort authority is required before exclusion: {no_eligible}"
        )
    expected_patients = {
        str(row["fold_owned_patient_key"])
        for row in variant_roster["eligible_records"]
    }
    if set(patient_pools) != expected_patients:
        raise PermissionError("EventNet variant-eligible patient denominator was deleted")
    primitive = build_patient_balanced_epoch_plan_pure_primitive(
        patient_pools,
        variant_id=variant_id,
        outer_fold=outer_fold,
        stage=stage,
        epoch_index=epoch_index,
    )
    return _content_address(
        {
            "schema_version": "eventnet_authorized_patient_balanced_epoch_plan_v1",
            "variant_id": variant_id,
            "outer_fold": outer_fold,
            "stage": stage,
            "epoch_index": epoch_index,
            "detector_fold_phase_receipt_sha256": phase_receipt[
                "detector_fold_phase_receipt_sha256"
            ],
            "variant_training_roster_receipt_sha256": variant_roster_receipt[
                "receipt_sha256"
            ],
            "complete_variant_eligible_record_count": len(expected_identities),
            "complete_variant_eligible_patient_count": len(patient_pools),
            "typed_excluded_phase_record_count": len(
                variant_roster["typed_exclusions"]
            ),
            "prediction_first_denominator_preserved": True,
            "authorized_record_pool_receipt_roster_sha256": _canonical_sha256(
                [
                    pools_by_identity[identity][1]["receipt_sha256"]
                    for identity in sorted(expected_identities)
                ]
            ),
            "eligible_record_or_patient_deletion_allowed": False,
            "primitive_plan": primitive,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )


def validate_fold_phase_authority(*_args: object, **_kwargs: object) -> None:
    """Legacy raw-mapping gate, permanently fail-closed.

    A plain validated mapping is still serializable and can be caller-owned.
    Formal consumers must call :func:`authorize_eventnet_fold_phase`, which
    requires actual reference-byte replay and returns an opaque sealed object.
    """

    raise PermissionError(
        "raw fold-phase mappings are not EventNet training authority; use "
        "authorize_eventnet_fold_phase with reference-byte replay"
    )


def validate_checkpoint_receipt_schema_only(
    value: Mapping[str, Any], *, registry: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate only the JSON shape of a future checkpoint declaration.

    This function intentionally does not admit a checkpoint: it reads no model,
    optimizer, initialization, sampler, roster, or phase-receipt artifact byte.
    Syntactically valid hashes remain untrusted declarations.
    """

    validated_registry = validate_registry(dict(registry))
    required = {
        "schema_version",
        "registry_sha256",
        "variant_id",
        "architecture_input_channels",
        "outer_fold",
        "stage",
        "epoch_completed",
        "derived_seed",
        "random_initialization_receipt_sha256",
        "transform_profile_id",
        "gradient_patient_roster_sha256",
        "validation_patient_roster_sha256",
        "typed_fold_reference_phase_receipt_sha256",
        "epoch_sampler_receipt_sha256",
        "optimizer_state_payload_sha256",
        "model_safetensors_sha256",
        "model_payload_variant_id",
        "published_checkpoint_loaded",
        "other_variant_checkpoint_loaded",
        "parameter_storage_shared_with_other_variant",
        "out_of_scope_reference_open_count",
        "save_boundary",
        "receipt_sha256",
    }
    data = _strict_dict(dict(value), required, "EventNet checkpoint receipt")
    if data["schema_version"] != "eventnet_cleanroom_epoch_checkpoint_receipt_v1":
        raise ValueError("EventNet checkpoint receipt schema drifted")
    if data["registry_sha256"] != validated_registry["registry_sha256"]:
        raise ValueError("EventNet checkpoint registry binding drifted")
    profile = _variant_profile(str(data["variant_id"]))
    if (
        data["architecture_input_channels"]
        != profile["architecture_input_channels"]
        or data["model_payload_variant_id"] != data["variant_id"]
    ):
        raise ValueError("EventNet checkpoint variant or input width drifted")
    expected_seed = derive_training_seed(
        variant_id=data["variant_id"],
        outer_fold=data["outer_fold"],
        stage=data["stage"],
    )
    if data["derived_seed"] != expected_seed:
        raise ValueError("EventNet checkpoint seed drifted")
    if (
        isinstance(data["epoch_completed"], bool)
        or not isinstance(data["epoch_completed"], int)
        or data["epoch_completed"] <= 0
    ):
        raise ValueError("EventNet checkpoint epoch must be positive")
    if data["transform_profile_id"] != validated_registry["transform"]["profile_id"]:
        raise ValueError("EventNet checkpoint transform profile drifted")
    for field in (
        "random_initialization_receipt_sha256",
        "gradient_patient_roster_sha256",
        "typed_fold_reference_phase_receipt_sha256",
        "epoch_sampler_receipt_sha256",
        "optimizer_state_payload_sha256",
        "model_safetensors_sha256",
    ):
        _require_sha256(data[field], field)
    if data["stage"] == "selection":
        _require_sha256(
            data["validation_patient_roster_sha256"],
            "validation_patient_roster_sha256",
        )
    elif data["stage"] == "final_refit":
        if data["validation_patient_roster_sha256"] is not None:
            raise ValueError("EventNet final refit may not bind a validation roster")
    else:
        raise ValueError("EventNet checkpoint stage is invalid")
    if (
        data["published_checkpoint_loaded"] is not False
        or data["other_variant_checkpoint_loaded"] is not False
        or data["parameter_storage_shared_with_other_variant"] is not False
        or data["out_of_scope_reference_open_count"] != 0
        or data["save_boundary"] != "completed_epoch_only"
    ):
        raise PermissionError("EventNet checkpoint independence/firewall drifted")
    pending = deepcopy(data)
    supplied = pending["receipt_sha256"]
    pending["receipt_sha256"] = _CONTENT_PENDING
    if supplied != _canonical_sha256(pending):
        raise ValueError("EventNet checkpoint receipt is not content-addressed")
    return data


def admit_eventnet_checkpoint(
    *_args: object, **_kwargs: object
) -> None:
    """Formal checkpoint admission is fail-closed until byte replay exists."""

    raise PermissionError(
        "EventNet checkpoint admission is unavailable: model/optimizer/init/"
        "sampler/roster/phase artifact byte replay and safetensors state-key "
        "verification are not implemented"
    )


def validate_checkpoint_receipt(*_args: object, **_kwargs: object) -> None:
    """Compatibility name for formal admission; never a schema-only bypass."""

    return admit_eventnet_checkpoint(*_args, **_kwargs)


def _architecture_contract(variant_id: str) -> dict[str, Any]:
    profile = _variant_profile(variant_id)
    return {
        "class": "EventNetCleanroomUNet",
        "MIT_release_repository": UPSTREAM_REPOSITORY,
        "MIT_release_commit": UPSTREAM_COMMIT,
        "MIT_release_architecture_git_blob": UPSTREAM_ARCHITECTURE_GIT_BLOB,
        "MIT_release_architecture_sha256": UPSTREAM_ARCHITECTURE_SHA256,
        "input_channels": profile["architecture_input_channels"],
        "base_channels": 16,
        "downsample_and_upsample_factor_per_level": 4,
        "level_count": 4,
        "encoder_kernel_size": 9,
        "decoder_primary_kernel_size": 15,
        "center_and_duration_head_kernel_size": 21,
        "head_max_pool_kernel_and_stride": [21, 1],
        "probability_activation": "sigmoid",
        "training_loss_input": "pre_sigmoid_logits",
        "model_input_samples": MODEL_INPUT_SAMPLES,
        "model_output_samples": TARGET_TILE_SAMPLES,
        "shape_ledger": architecture_shape_ledger(),
        "random_initialization_only": True,
        "released_checkpoint_warm_start_allowed": False,
        "other_variant_checkpoint_or_parameter_storage_allowed": False,
    }


def build_registry(
    *,
    implementation_code_sha256: str,
    signal_lineage_authority_source_sha256: str,
    channel_router_source_sha256: str,
    fold_plan_file_sha256: str,
    fold_authority_registry_file_sha256: str,
    fold_authority_registry_receipt_sha256: str,
    fold_authority_source_sha256: str,
    phase_gate_source_sha256: str,
    released_adapter_file_sha256: str,
) -> dict[str, Any]:
    """Build the exact content-addressed clean-room EventNet registry."""

    hashes = {
        "implementation_code_sha256": implementation_code_sha256,
        "signal_lineage_authority_source_sha256": signal_lineage_authority_source_sha256,
        "channel_router_source_sha256": channel_router_source_sha256,
        "fold_plan_file_sha256": fold_plan_file_sha256,
        "fold_authority_registry_file_sha256": fold_authority_registry_file_sha256,
        "fold_authority_registry_receipt_sha256": fold_authority_registry_receipt_sha256,
        "fold_authority_source_sha256": fold_authority_source_sha256,
        "phase_gate_source_sha256": phase_gate_source_sha256,
        "released_adapter_file_sha256": released_adapter_file_sha256,
    }
    for context, value in hashes.items():
        _require_sha256(value, context)
    registry: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "registry_id": REGISTRY_ID,
        "status": (
            "CPU_architecture_transform_and_numeric_primitives_executable_"
            "opaque_training_authority_APIs_executable_artifacts_and_models_not_materialized"
        ),
        "provider_id": PROVIDER_ID,
        "extends_without_modifying": [
            "configs/clinical_eeg_detector_cleanroom_execution_freeze_v1.json",
            "configs/clinical_eeg_detector_channel_support_routing_addendum_v1.json",
            "src/clinical_eeg_long_recording/eventnet_full_record_adapter.py",
        ],
        "implementation": {
            "path": "src/clinical_eeg_long_recording/eventnet_cleanroom_registry_v1.py",
            "code_sha256": implementation_code_sha256,
        },
        "variant_profiles": {
            variant: {
                "input_profile": _variant_profile(variant),
                "architecture": _architecture_contract(variant),
            }
            for variant in (EN19_VARIANT_ID, EN17_VARIANT_ID)
        },
        "transform": {
            "profile_id": "eventnet_whole_record_referential_uV_polyphase256_no_filter_no_normalization_v1",
            "input_authority": "typed_replayed_canonical_referential_volts_only",
            "shared_with_SeizureTransformer": [
                "canonical_physical_signal_identity",
                "observed_roster",
                "electrical_reference_system",
                "common_exact_clock",
                "EEG_only_QC_authority",
                "whole_record_explicit_polyphase_resampling_basis",
            ],
            "not_shared_with_SeizureTransformer": [
                "bipolar_montage",
                "0.5_to_100Hz_filter",
                "median_MAD_normalization",
                "clipping",
            ],
            "source_unit": "V",
            "provider_unit": "uV",
            "target_sampling_rate_hz": TARGET_FS_HZ,
            "whole_record_before_tiling": True,
            "resample": "explicit_Kaiser5_polyphase_padtype_line_floor_clock_v1",
            "filtering": None,
            "normalization": None,
            "clipping": None,
            "fold_fitted_statistics": None,
            "target_or_cross_record_information_used": False,
            "EN17_direct_axis_selection_before_resampling": True,
            "missing_channel_fill_or_interpolation_allowed": False,
        },
        "target_and_loss": {
            "API_boundary": {
                "raw_event_target_loss_and_sampler_functions": "pure_numeric_primitives_only",
                "formal_target_input": "opaque_byte_replayed_fold_phase_plus_exact_record_transform_unpadded_tile_bundle",
                "formal_loss_accepts_raw_masks_counts_or_patient_keys": False,
                "formal_checkpoint_admission": "fail_closed_pending_actual_artifact_byte_replay",
            },
            "center_target": {
                "formula": "exp(-(t-t_star)^2/(2*sigma^2))",
                "sigma": "0.5*actual_duration_samples/6",
                "multiple_event_reduction": "pointwise_max",
                "Gaussian_center_clock": "exact_fractional_midpoint_of_half_open_interval",
                "positive_anchor_clock": "nearest_256Hz_sample_ties_earlier",
                "positive_focal_branch": "sparse_duration_anchor_not_target_float_equality",
            },
            "duration_target": {
                "defined_only_at_distinct_complete_centers": True,
                "value": "min(actual_duration_seconds,300)/300",
                "same_quantized_center_conflict": "maximum_duration",
            },
            "center_loss": {
                "name": "logit_stable_modified_focal_loss",
                "alpha_c": FOCAL_ALPHA_C,
                "alpha": FOCAL_ALPHA,
                "beta": FOCAL_BETA,
                "positive_normalization": "divide_by_distinct_center_count",
                "zero_event_background_resolution": "CenterNet_sum_negative_loss_with_denominator_one",
            },
            "duration_loss": {
                "implemented": "mean(1-min(pred,target)/max(pred,target))_at_distinct_complete_centers",
                "paper_printed_equation": "mean(min(pred,target)/max(pred,target))",
                "ambiguity_resolution": "use_one_minus_IoU_because_optimization_minimizes_loss",
                "empty_duration_mask_loss": 0.0,
            },
            "lambda_duration": DURATION_LOSS_WEIGHT,
            "nonobserved_tail_or_short_record_samples_enter_loss": False,
            "boundary_censored_duration_supervision": False,
            "event_piece_with_center_in_other_tile_treated_as_background": False,
        },
        "tiling": {
            "target_tile_seconds": TARGET_TILE_SECONDS,
            "target_tile_samples": TARGET_TILE_SAMPLES,
            "context_samples_each_side": CONTEXT_SAMPLES_PER_SIDE,
            "model_input_samples": MODEL_INPUT_SAMPLES,
            "target_hop_samples": TARGET_TILE_SAMPLES,
            "inference_full_record_coverage": "starts_0_30720_..._tail_zero_padded",
            "training_target_starts": "128_then_hop30720_only_when_both_context_sides_and_full_target_observed",
            "short_record_policy": "inference_one_fixed_shape_tile_with_only_observed_output_mask_true_training_loss_fail_closed",
            "tail_policy": "inference_fixed_shape_zero_padding_and_explicit_output_mask_training_loss_fail_closed",
            "padding_is_observed_EEG": False,
            "padding_may_enter_center_or_duration_loss": False,
            "padding_may_enter_training_forward": False,
            "reason": "release_BatchNorm_spans_time_so_masking_output_alone_cannot_prevent_padding_statistic_contamination",
            "tile_outputs_flattened_only_after_absolute_nonoverlap_alignment": True,
        },
        "trainer": {
            "trainer_id": (
                "eventnet_cleanroom_patient_balanced_direct_event_"
                "trainer_contract_authority_wiring_pending_v1"
            ),
            "fold_plan": {
                "path": "outputs/tusz_canonical_physical_signal_audit_v1_full_20260824r2/detector_cleanroom_fold_plan.json",
                "file_sha256": fold_plan_file_sha256,
                "fold_count": 5,
                "inner_validation_fold": "(outer_fold+1)%5",
            },
            "fold_reference_authority": {
                "path": "configs/clinical_eeg_detector_fold_reference_authority_registry_v1.json",
                "file_sha256": fold_authority_registry_file_sha256,
                "registry_receipt_sha256": fold_authority_registry_receipt_sha256,
                "typed_phase_replay_required": True,
                "bare_SHA_gate_allowed": False,
            },
            "formal_authority_wiring": {
                "fold_phase_type": "AuthorizedEventNetFoldPhase",
                "actual_reference_byte_replay_required": True,
                "shared_controller_artifact_replay_opaque_authority_required": True,
                "shared_controller_opaque_authority_consumed": True,
                "formal_fold_phase_issuer_status": "implemented_requires_process_sealed_shared_authority",
                "variant_training_roster_type": (
                    "AuthorizedEventNetVariantTrainingRoster"
                ),
                "variant_training_roster_definition": (
                    "authorized_phase_intersection_target_blind_provider_route_"
                    "intersection_pre_reference_technical_eligibility"
                ),
                "variant_training_roster_issuer_status": (
                    "implemented_requires_complete_process_sealed_pre_reference_outcome_Cartesian"
                ),
                "pre_reference_eligibility_type": (
                    "EventNetPreReferenceEligibilityOutcome"
                ),
                "target_bundle_type": "AuthorizedEventNetTargetBundle",
                "tile_bound_logits_type": "BoundEventNetTrainingLogits",
                "record_pool_type": "AuthorizedEventNetRecordPool",
                "complete_phase_record_Cartesian_is_variant_training_roster": False,
                "complete_variant_eligible_record_denominator_required": True,
                "typed_exclusions_retained_in_prediction_first_denominator": True,
                "raw_events_masks_counts_patient_keys_or_rosters_allowed": False,
                "complete_variant_route_roster_materialized": False,
                "materialized_pre_reference_outcome_roster_receipt_sha256": None,
                "materialized_fold_phase_adapter_receipt_sha256": None,
                "materialized_variant_training_roster_receipt_sha256": None,
                "epoch_executor_implemented": False,
            },
            "stages": [
                {
                    "stage": "selection",
                    "gradient_roster": "source_train_three_selection_fit_folds",
                    "validation_roster": "source_train_next_inner_fold",
                    "epoch_selection": "minimum_patient_macro_center_plus_5_duration_loss",
                    "source_dev_used": False,
                },
                {
                    "stage": "final_refit",
                    "gradient_roster": "source_train_all_four_outer_train_folds",
                    "validation_roster": None,
                    "epoch_count": "selected_epoch_from_corresponding_inner_selection",
                    "reinitialize_from_scratch": True,
                },
            ],
            "patient_balanced_sampling": {
                "tiles_per_patient_per_epoch": 8,
                "when_both_pools_exist_positive_background_quota": [4, 4],
                "one_tile_per_patient_per_batch": True,
                "batch_patient_maximum": 16,
                "background_tile_requires_no_event_intersection_in_target_plus_context": True,
                "event_transition_without_center_excluded_from_background": True,
                "gradient_tile_requires_fully_observed_target_and_both_context_sides": True,
                "identity_used_as_model_feature": False,
            },
            "optimizer": {
                "class": "torch.optim.AdamW",
                "learning_rate": 0.0001,
                "betas": [0.9, 0.999],
                "epsilon": 1e-8,
                "weight_decay": 0.00002,
                "gradient_clip_global_L2_norm": 1.0,
            },
            "seed": {
                "base_seed": 20260824,
                "variant_fold_stage_SHA256_derivation": True,
                "best_seed_selection_allowed": False,
            },
            "checkpoint": {
                "format": "tensor_only_safetensors_plus_typed_JSON_receipt",
                "schema_only_validator_implemented": True,
                "formal_admission_status": "fail_closed_pending_actual_artifact_byte_replay",
                "formal_admission_reads_model_optimizer_init_sampler_roster_and_phase_bytes": False,
                "formal_admission_artifact_receipt_sha256": None,
                "published_checkpoint_initialization_allowed": False,
                "cross_variant_checkpoint_initialization_or_sharing_allowed": False,
                "final_checkpoint_count_required_per_variant": 5,
                "current_EN19_checkpoint_count": 0,
                "current_EN17_checkpoint_count": 0,
            },
        },
        "selection_authority": {
            "source_train_inner_may_select": ["epoch_only"],
            "source_train_inner_may_not_select": [
                "EN19_versus_EN17_route",
                "channel_order_or_missingness_policy",
                "target_sigma_or_loss_hyperparameters",
                "transform_or_architecture",
                "random_seed",
                "operating_point",
            ],
            "source_dev_role": "post_OOF_provider_policy_and_operating_point_only",
            "source_eval_role": "one_shot_after_complete_freeze_only",
            "private_or_source_eval_labels_available_to_training": False,
        },
        "execution": {
            "static_source_bindings": [
                {
                    "semantic": "typed_detector_signal_lineage_authority",
                    "path": "src/clinical_eeg_long_recording/detector_signal_lineage_authority_v1.py",
                    "file_sha256": signal_lineage_authority_source_sha256,
                },
                {
                    "semantic": "target_blind_channel_support_router",
                    "path": "src/clinical_eeg_long_recording/detector_channel_support_router_v1.py",
                    "file_sha256": channel_router_source_sha256,
                },
                {
                    "semantic": "patient_disjoint_fold_plan",
                    "path": "outputs/tusz_canonical_physical_signal_audit_v1_full_20260824r2/detector_cleanroom_fold_plan.json",
                    "file_sha256": fold_plan_file_sha256,
                },
                {
                    "semantic": "typed_fold_reference_authority_registry",
                    "path": "configs/clinical_eeg_detector_fold_reference_authority_registry_v1.json",
                    "file_sha256": fold_authority_registry_file_sha256,
                },
                {
                    "semantic": "typed_fold_reference_authority_implementation",
                    "path": "src/clinical_eeg_long_recording/detector_fold_reference_authority_v1.py",
                    "file_sha256": fold_authority_source_sha256,
                },
                {
                    "semantic": "typed_reference_phase_gate",
                    "path": "src/clinical_eeg_long_recording/detector_reference_phase_gate_v1.py",
                    "file_sha256": phase_gate_source_sha256,
                },
                {
                    "semantic": "released_public_weight_adapter_unchanged_comparator",
                    "path": "src/clinical_eeg_long_recording/eventnet_full_record_adapter.py",
                    "file_sha256": released_adapter_file_sha256,
                },
            ],
            "forward_allowlist": ["provider_preprocessed_EEG_tensor", "fixed_shape_control_plane"],
            "forward_forbidden": [
                "EDF_annotation",
                "spreadsheet_or_doctor_text",
                "clinical_history",
                "video_or_behavior",
                "sleep_or_activation_labels",
                "ECG_EMG_EOG",
                "patient_identity_feature",
                "reference_interval_or_target_tensor",
                "lineage_hash_feature",
            ],
            "GPU_long_training_started_by_this_registry": False,
            "current_vLLM_service_may_be_stopped_by_this_registry": False,
        },
        "scientific_claim_boundary": {
            "architecture_and_typed_transform_CPU_executable": True,
            "target_loss_tiling_sampler_numeric_primitives_CPU_executable": True,
            "opaque_fold_phase_target_bundle_and_bound_loss_APIs_implemented": True,
            "shared_controller_replayed_opaque_phase_consumed": True,
            "target_blind_variant_training_roster_authority_API_implemented": True,
            "target_blind_variant_training_roster_artifact_materialized": False,
            "formal_training_path_executable": False,
            "complete_variant_fold_owned_epoch_roster_materialized": False,
            "checkpoint_schema_validation_is_checkpoint_authority": False,
            "formal_checkpoint_admission_executable": False,
            "architecture_transform_target_loss_tiling_sampler_CPU_executable": False,
            "PyTorch_epoch_training_executor_implemented": False,
            "EN19_checkpoint_count": 0,
            "EN17_checkpoint_count": 0,
            "five_fold_OOF_materialized": False,
            "source_dev_operating_point_selected": False,
            "source_eval_one_shot_executed": False,
            "accuracy_primary": None,
            "performance_or_SOTA_claim_allowed": False,
            "clinical_or_production_use_allowed": False,
        },
        "registry_sha256": _CONTENT_PENDING,
    }
    registry["registry_sha256"] = _canonical_sha256(registry)
    return registry


def validate_registry(value: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on registry semantic or content-address drift."""

    required = {
        "schema_version",
        "registry_id",
        "status",
        "provider_id",
        "extends_without_modifying",
        "implementation",
        "variant_profiles",
        "transform",
        "target_and_loss",
        "tiling",
        "trainer",
        "selection_authority",
        "execution",
        "scientific_claim_boundary",
        "registry_sha256",
    }
    data = _strict_dict(dict(value), required, "EventNet clean-room registry")
    if data["schema_version"] != SCHEMA_VERSION or data["registry_id"] != REGISTRY_ID:
        raise ValueError("EventNet clean-room registry identity drifted")
    if data["provider_id"] != PROVIDER_ID:
        raise ValueError("EventNet clean-room provider identity drifted")
    bindings = {row["semantic"]: row for row in data["execution"]["static_source_bindings"]}
    expected = build_registry(
        implementation_code_sha256=data["implementation"]["code_sha256"],
        signal_lineage_authority_source_sha256=bindings[
            "typed_detector_signal_lineage_authority"
        ]["file_sha256"],
        channel_router_source_sha256=bindings[
            "target_blind_channel_support_router"
        ]["file_sha256"],
        fold_plan_file_sha256=bindings["patient_disjoint_fold_plan"]["file_sha256"],
        fold_authority_registry_file_sha256=bindings[
            "typed_fold_reference_authority_registry"
        ]["file_sha256"],
        fold_authority_registry_receipt_sha256=data["trainer"][
            "fold_reference_authority"
        ]["registry_receipt_sha256"],
        fold_authority_source_sha256=bindings[
            "typed_fold_reference_authority_implementation"
        ]["file_sha256"],
        phase_gate_source_sha256=bindings["typed_reference_phase_gate"]["file_sha256"],
        released_adapter_file_sha256=bindings[
            "released_public_weight_adapter_unchanged_comparator"
        ]["file_sha256"],
    )
    if data != expected:
        raise ValueError("EventNet clean-room registry semantic content drifted")
    return data


def load_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.is_file() or registry_path.is_symlink():
        raise ValueError("EventNet registry path must be a regular non-symlink file")
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("EventNet registry is not readable JSON") from exc
    validated = validate_registry(payload)
    if validated["implementation"]["code_sha256"] != eventnet_cleanroom_registry_code_sha256():
        raise ValueError("EventNet registry implementation binding is stale")
    return validated


def validate_static_execution_bindings(
    project_root: str | Path, *, registry: Mapping[str, Any]
) -> dict[str, Any]:
    """Hash-check all local authority/adapter/fold bindings before execution."""

    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ValueError("project_root must be a directory")
    validated = validate_registry(dict(registry))
    rows: list[dict[str, Any]] = []
    for binding in validated["execution"]["static_source_bindings"]:
        path = root / binding["path"]
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"EventNet static binding is missing: {binding['path']}")
        observed = _file_sha256(path)
        if observed != binding["file_sha256"]:
            raise ValueError(f"EventNet static binding drifted: {binding['semantic']}")
        rows.append(
            {
                "semantic": binding["semantic"],
                "path": binding["path"],
                "observed_file_sha256": observed,
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": "eventnet_cleanroom_static_binding_receipt_v1",
        "registry_sha256": validated["registry_sha256"],
        "binding_count": len(rows),
        "bindings": rows,
        "all_exact": True,
        "receipt_sha256": _CONTENT_PENDING,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


__all__ = [
    "AuthorizedEventNetFoldPhase",
    "AuthorizedEventNetRecordPool",
    "AuthorizedEventNetTargetBundle",
    "AuthorizedEventNetVariantTrainingRoster",
    "BoundEventNetTrainingLogits",
    "CONTEXT_SAMPLES_PER_SIDE",
    "DURATION_LOSS_WEIGHT",
    "EN17_CHANNEL_ORDER",
    "EN17_VARIANT_ID",
    "EN19_CHANNEL_ORDER",
    "EN19_VARIANT_ID",
    "EventNetCleanroomUNet",
    "EventNetLossResult",
    "EventNetModelTile",
    "EventNetPreReferenceEligibilityOutcome",
    "EventNetTargets",
    "EventNetTransformResult",
    "MAXIMUM_DURATION_SECONDS",
    "MODEL_INPUT_SAMPLES",
    "PROVIDER_ID",
    "REGISTRY_ID",
    "SCHEMA_VERSION",
    "TARGET_FS_HZ",
    "TARGET_TILE_SAMPLES",
    "admit_eventnet_checkpoint",
    "apply_full_record_transform",
    "architecture_shape_ledger",
    "authorize_eventnet_fold_phase",
    "authorize_eventnet_record_tile_pool",
    "authorize_eventnet_target_bundle",
    "authorize_eventnet_variant_training_roster",
    "bind_eventnet_training_logits",
    "build_authorized_patient_balanced_epoch_plan",
    "build_eventnet_targets_pure_primitive",
    "build_patient_balanced_epoch_plan_pure_primitive",
    "build_randomly_initialized_model",
    "build_record_tile_pools_pure_primitive",
    "build_registry",
    "derive_training_seed",
    "enumerate_target_tiles",
    "enumerate_training_target_tiles",
    "eventnet_cleanroom_registry_code_sha256",
    "eventnet_authorized_multitask_loss",
    "eventnet_multitask_loss_from_logits_pure_primitive",
    "load_registry",
    "materialize_model_tile",
    "materialize_eventnet_pre_reference_eligibility",
    "validate_fold_phase_authority",
    "validate_checkpoint_receipt",
    "validate_checkpoint_receipt_schema_only",
    "validate_registry",
    "validate_static_execution_bindings",
    "validate_transform_result",
]
