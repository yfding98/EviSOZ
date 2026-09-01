"""Strict adapter for the audited official LaBraM-Base implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import io
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from ..geometry import N_STANDARD_CHANNELS, STANDARD_19, normalize_electrode_name


AUDITED_LABRAM_BASE_SHA256 = (
    "7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c"
)
AUDITED_LABRAM_MODELING_SHA256 = (
    "7b3514c9a661ebd417a2dd9834a9f2fab0a2e518c8212c73ebebaaa165c29d88"
)
AUDITED_ENCODER_TENSOR_COUNT = 221

# The target uses modern semantic names, but the TUH-family checkpoint was
# trained with legacy T3/T4/T5/T6 positions for these four physical sites.
LABRAM_LEGACY_POSITION_NAMES: tuple[str, ...] = (
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T3",
    "C3",
    "CZ",
    "C4",
    "T4",
    "T5",
    "P3",
    "PZ",
    "P4",
    "T6",
    "O1",
    "O2",
)

# Official ``utils.standard_1020.index(name) + 1`` IDs.  Zero is reserved for
# the CLS token.  These values are frozen in the feature receipt and never
# inferred from a target label.
LABRAM_POSITION_ID_BY_NAME: dict[str, int] = {
    "FP1": 1,
    "FP2": 3,
    "F7": 16,
    "F3": 18,
    "FZ": 20,
    "F4": 22,
    "F8": 24,
    "T7": 38,
    "C3": 40,
    "CZ": 42,
    "C4": 44,
    "T8": 46,
    "P7": 60,
    "P3": 62,
    "PZ": 64,
    "P4": 66,
    "P8": 68,
    "O1": 81,
    "O2": 83,
    "T3": 89,
    "T5": 90,
    "T4": 91,
    "T6": 92,
}

LABRAM_RAW_HEADER_POSITION_BINDING_POLICY = (
    "exact_raw_header_legacy_modern_alias_to_official_1020_id_v1"
)


def _raw_electrode_position_name(raw_name: object) -> str:
    """Return the exact legacy/modern electrode token encoded by one header.

    ``normalize_electrode_name`` deliberately merges T3/T7-style aliases.  That
    is correct for the physical standard-19 carrier, but it is not sufficient
    for LaBraM: the official position table gives legacy and modern aliases
    different IDs.  This parser therefore removes only the recognized EEG and
    reference wrappers while preserving the alias written in the EDF header.
    """

    text = str(raw_name).strip().upper().replace("_", "-")
    for prefix in ("EEG ", "EEG-"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    for suffix in ("-REF", "-LE", "-AR", "-AVG", "-AV", "-CAR"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    text = text.strip("- ")
    if not text or "-" in text or text not in LABRAM_POSITION_ID_BY_NAME:
        raise ValueError(
            f"Raw EEG header {raw_name!r} does not encode one supported "
            "LaBraM electrode position"
        )
    return text


@dataclass(frozen=True)
class LaBraMRecordPositionBinding:
    """Exact per-record bridge from physical EEG headers to LaBraM IDs."""

    policy: str
    semantic_channels: tuple[str, ...]
    raw_channel_names: tuple[str, ...]
    position_names: tuple[str, ...]
    position_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.policy != LABRAM_RAW_HEADER_POSITION_BINDING_POLICY:
            raise ValueError("Unsupported LaBraM raw-header position-binding policy")
        if self.semantic_channels != STANDARD_19:
            raise ValueError("LaBraM record binding must use frozen standard-19 order")
        fields = (
            self.raw_channel_names,
            self.position_names,
            self.position_ids,
        )
        if any(len(values) != N_STANDARD_CHANNELS for values in fields):
            raise ValueError("LaBraM record binding fields must contain 19 values")
        expected_names = tuple(
            _raw_electrode_position_name(raw_name)
            for raw_name in self.raw_channel_names
        )
        if self.position_names != expected_names:
            raise ValueError("LaBraM position names drifted from the raw EEG headers")
        semantic_mismatches = tuple(
            f"{semantic}->{position_name}"
            for semantic, position_name in zip(
                self.semantic_channels, self.position_names
            )
            if normalize_electrode_name(position_name) != semantic
        )
        if semantic_mismatches:
            raise ValueError(
                "Raw EEG aliases are not aligned with the physical carrier: "
                + ",".join(semantic_mismatches)
            )
        expected_ids = tuple(
            LABRAM_POSITION_ID_BY_NAME[name] for name in self.position_names
        )
        if self.position_ids != expected_ids:
            raise ValueError("LaBraM position IDs drifted from the official table")
        if len(set(self.position_ids)) != N_STANDARD_CHANNELS:
            raise ValueError("LaBraM record binding contains duplicate position IDs")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def bind_labram_record_positions(
    raw_channel_names: Sequence[object],
    *,
    semantic_channels: Sequence[str] = STANDARD_19,
) -> LaBraMRecordPositionBinding:
    """Bind one EDF record's selected headers to official LaBraM IDs.

    Legacy aliases (T3/T4/T5/T6) and modern aliases (T7/T8/P7/P8) remain
    distinct position names/IDs even though they share physical semantics.
    No default legacy substitution is made here.
    """

    raw_names = tuple(str(value).strip() for value in raw_channel_names)
    semantics = tuple(str(value).strip().upper() for value in semantic_channels)
    if len(raw_names) != N_STANDARD_CHANNELS or len(semantics) != N_STANDARD_CHANNELS:
        raise ValueError("Raw headers and semantic channels must contain 19 values")
    positions = tuple(_raw_electrode_position_name(value) for value in raw_names)
    return LaBraMRecordPositionBinding(
        policy=LABRAM_RAW_HEADER_POSITION_BINDING_POLICY,
        semantic_channels=semantics,
        raw_channel_names=raw_names,
        position_names=positions,
        position_ids=tuple(LABRAM_POSITION_ID_BY_NAME[name] for name in positions),
    )


def require_feature_receipt_position_binding(
    receipt: "LaBraMFeatureReceipt",
    binding: LaBraMRecordPositionBinding,
) -> None:
    """Fail closed unless one record matches the encoder's actual position IDs."""

    if not isinstance(receipt, LaBraMFeatureReceipt):
        raise TypeError("receipt must be a LaBraMFeatureReceipt")
    if not isinstance(binding, LaBraMRecordPositionBinding):
        raise TypeError("binding must be a LaBraMRecordPositionBinding")
    if receipt.semantic_channels != binding.semantic_channels:
        raise ValueError("Foundation semantic channels differ from the EDF binding")
    if (
        receipt.position_names != binding.position_names
        or receipt.position_ids != binding.position_ids
    ):
        raise ValueError(
            "EDF raw-header LaBraM position binding differs from the encoder "
            "feature receipt; refusing default legacy position IDs"
        )


@dataclass(frozen=True)
class LaBraMFeatureReceipt:
    checkpoint_path: str
    checkpoint_sha256: str
    modeling_path: str
    modeling_sha256: str
    encoder_tensor_count: int
    semantic_channels: tuple[str, ...]
    position_names: tuple[str, ...]
    position_ids: tuple[int, ...]
    tile_seconds: int
    pretraining_window_seconds: int = 8
    samples_per_token: int = 200
    token_dim: int = 200
    input_scale_from_volts: float = 1e4

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _StableFileSnapshot:
    path: Path
    content: bytes
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _stable_file_snapshot(path_value: str | Path, *, label: str) -> _StableFileSnapshot:
    """Read one regular file through a stable FD and bind its path to that inode."""

    path = Path(path_value).resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise RuntimeError(f"{label} changed while its stable snapshot was read")
    content = b"".join(chunks)
    if len(content) != before.st_size:
        raise RuntimeError(f"{label} snapshot size disagrees with its file descriptor")
    path_state = os.stat(path, follow_symlinks=False)
    if (path_state.st_dev, path_state.st_ino) != (before.st_dev, before.st_ino):
        raise RuntimeError(f"{label} path changed inode while it was snapshotted")
    return _StableFileSnapshot(
        path=path,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        device=before.st_dev,
        inode=before.st_ino,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
        ctime_ns=before.st_ctime_ns,
    )


@lru_cache(maxsize=4)
def _load_modeling_module(
    path_text: str, expected_sha256: str, source_snapshot: bytes
) -> ModuleType:
    path = Path(path_text)
    if hashlib.sha256(source_snapshot).hexdigest() != expected_sha256:
        raise RuntimeError("LaBraM modeling snapshot does not match its SHA-256")
    module_name = (
        f"_neurosoz_official_labram_{abs(hash(str(path)))}_{expected_sha256[:16]}"
    )
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        code = compile(source_snapshot, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if not hasattr(module, "labram_base_patch200_200"):
        raise AttributeError("Official module lacks labram_base_patch200_200")
    return module


def _load_checkpoint_snapshot(
    path_value: str | Path, *, expected_sha256: str
) -> tuple[object, _StableFileSnapshot]:
    snapshot = _stable_file_snapshot(path_value, label="Official LaBraM checkpoint")
    if snapshot.sha256 != expected_sha256:
        raise ValueError(
            "Official LaBraM checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, got {snapshot.sha256}"
        )
    payload = torch.load(
        io.BytesIO(snapshot.content), map_location="cpu", weights_only=False
    )
    return payload, snapshot


def _extract_student_encoder_state(payload: object) -> dict[str, torch.Tensor]:
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise TypeError("LaBraM checkpoint must contain a model mapping")
    raw_state = payload["model"]
    if not isinstance(raw_state, Mapping):
        raise TypeError("LaBraM model state must be a mapping")
    state: dict[str, torch.Tensor] = {}
    for raw_name, value in raw_state.items():
        name = str(raw_name)
        if not name.startswith("student."):
            continue
        name = name[len("student.") :]
        if name == "mask_token" or name.startswith("lm_head."):
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Checkpoint entry {raw_name!r} is not a tensor")
        state[name] = value
    if len(state) != AUDITED_ENCODER_TENSOR_COUNT:
        raise ValueError(
            "Unexpected LaBraM encoder tensor count: "
            f"expected {AUDITED_ENCODER_TENSOR_COUNT}, got {len(state)}"
        )
    return state


def _validated_position_contract(
    position_names: Sequence[str],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Validate the one-to-one semantic-channel to LaBraM-position contract."""

    if len(position_names) != N_STANDARD_CHANNELS:
        raise ValueError("position_names must align with all 19 semantic channels")
    names = tuple(str(name).strip().upper() for name in position_names)
    unknown = [name for name in names if name not in LABRAM_POSITION_ID_BY_NAME]
    if unknown:
        raise ValueError(f"Unknown LaBraM position names: {unknown}")

    semantic_mismatches = [
        f"{semantic}->{position_name}"
        for semantic, position_name in zip(STANDARD_19, names)
        if normalize_electrode_name(position_name) != semantic
    ]
    position_ids = tuple(LABRAM_POSITION_ID_BY_NAME[name] for name in names)
    duplicate_ids = sorted(
        {position_id for position_id in position_ids if position_ids.count(position_id) > 1}
    )
    problems: list[str] = []
    if semantic_mismatches:
        problems.append("semantic mismatches=" + ",".join(semantic_mismatches))
    if duplicate_ids:
        problems.append(f"duplicate position IDs={duplicate_ids}")
    if problems:
        raise ValueError("Invalid LaBraM channel-position contract: " + "; ".join(problems))
    return names, position_ids


class OfficialLaBraMEncoder(nn.Module):
    """Frozen official LaBraM patch-token extractor with no random fallback."""

    token_dim = 200
    samples_per_token = 200

    def __init__(
        self,
        *,
        modeling_path: str | Path,
        checkpoint_path: str | Path,
        expected_sha256: str = AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256: str = AUDITED_LABRAM_MODELING_SHA256,
        tile_seconds: int = 4,
        position_names: Sequence[str] = LABRAM_LEGACY_POSITION_NAMES,
    ) -> None:
        super().__init__()
        if not 1 <= int(tile_seconds) <= 16:
            raise ValueError("LaBraM tile_seconds must lie in [1,16]")
        position_names_tuple, position_ids = _validated_position_contract(position_names)
        self.seconds_per_call = int(tile_seconds)

        modeling_snapshot = _stable_file_snapshot(
            modeling_path, label="Official LaBraM modeling source"
        )
        modeling_path = modeling_snapshot.path
        actual_modeling_sha256 = modeling_snapshot.sha256
        if actual_modeling_sha256 != expected_modeling_sha256.strip().lower():
            raise ValueError(
                "Official LaBraM modeling source SHA-256 mismatch: "
                f"expected {expected_modeling_sha256}, got {actual_modeling_sha256}"
            )
        module = _load_modeling_module(
            str(modeling_path),
            actual_modeling_sha256,
            modeling_snapshot.content,
        )
        self.backbone = module.labram_base_patch200_200(
            EEG_size=1600,
            in_chans=1,
            out_chans=8,
            num_classes=0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_values=0.1,
            use_mean_pooling=False,
        )

        # This audited upstream checkpoint predates weights-only serialization
        # support and contains a NumPy scalar in optimizer metadata.  We first
        # verify its exact SHA-256, then use legacy pickle only for this pinned
        # artifact and extract encoder tensors exclusively.
        payload, checkpoint_snapshot = _load_checkpoint_snapshot(
            checkpoint_path,
            expected_sha256=expected_sha256.strip().lower(),
        )
        checkpoint_path = checkpoint_snapshot.path
        actual_sha256 = checkpoint_snapshot.sha256
        state = _extract_student_encoder_state(payload)
        self.backbone.load_state_dict(state, strict=True)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

        self.register_buffer(
            "input_chans",
            torch.tensor((0, *position_ids), dtype=torch.long),
            persistent=True,
        )
        self.receipt = LaBraMFeatureReceipt(
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=actual_sha256,
            modeling_path=str(modeling_path),
            modeling_sha256=actual_modeling_sha256,
            encoder_tensor_count=len(state),
            semantic_channels=STANDARD_19,
            position_names=position_names_tuple,
            position_ids=position_ids,
            tile_seconds=self.seconds_per_call,
        )

    def train(self, mode: bool = True) -> "OfficialLaBraMEncoder":
        super().train(mode)
        self.backbone.eval()
        return self

    def _forward_with_input_chans(
        self,
        patches: torch.Tensor,
        input_chans: torch.Tensor,
    ) -> torch.Tensor:
        expected = (
            N_STANDARD_CHANNELS,
            self.seconds_per_call,
            self.samples_per_token,
        )
        if patches.ndim != 4 or tuple(patches.shape[1:]) != expected:
            raise ValueError(
                f"Official LaBraM input must have shape [B,{expected[0]},"
                f"{expected[1]},{expected[2]}], got {tuple(patches.shape)}"
            )
        if not patches.is_floating_point() or not torch.isfinite(patches).all():
            raise ValueError("Official LaBraM input must be finite floating-point EEG")
        expected_input_chans = (N_STANDARD_CHANNELS + 1,)
        if (
            tuple(input_chans.shape) != expected_input_chans
            or input_chans.dtype != torch.long
        ):
            raise ValueError("LaBraM input_chans must be long [20] including CLS")
        input_chans = input_chans.to(device=patches.device)
        self.backbone.eval()
        with torch.no_grad():
            flat_tokens = self.backbone.forward_features(
                patches * self.receipt.input_scale_from_volts,
                input_chans=input_chans,
                return_patch_tokens=True,
            )
        expected_flat = (
            patches.shape[0],
            N_STANDARD_CHANNELS * self.seconds_per_call,
            self.token_dim,
        )
        if tuple(flat_tokens.shape) != expected_flat:
            raise ValueError(
                f"Official LaBraM returned {tuple(flat_tokens.shape)}, "
                f"expected {expected_flat}"
            )
        return flat_tokens.reshape(
            patches.shape[0],
            N_STANDARD_CHANNELS,
            self.seconds_per_call,
            self.token_dim,
        ).detach()

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self._forward_with_input_chans(patches, self.input_chans)

    def feature_receipt_for_record_binding(
        self,
        binding: LaBraMRecordPositionBinding,
    ) -> LaBraMFeatureReceipt:
        """Return the exact feature receipt for one raw-header position binding."""

        if not isinstance(binding, LaBraMRecordPositionBinding):
            raise TypeError("binding must be LaBraMRecordPositionBinding")
        return replace(
            self.receipt,
            position_names=binding.position_names,
            position_ids=binding.position_ids,
        )

    def forward_with_record_binding(
        self,
        patches: torch.Tensor,
        binding: LaBraMRecordPositionBinding,
    ) -> torch.Tensor:
        """Encode one batch using position IDs derived from its real EDF headers."""

        receipt = self.feature_receipt_for_record_binding(binding)
        input_chans = torch.tensor(
            (0, *receipt.position_ids),
            dtype=torch.long,
            device=patches.device,
        )
        return self._forward_with_input_chans(patches, input_chans)
