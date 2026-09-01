"""Fail-closed interfaces for a frozen pretrained EEG foundation encoder."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Mapping

import torch
import torch.nn as nn

from ..geometry import N_STANDARD_CHANNELS


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a checkpoint without loading it into memory."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Foundation checkpoint not found: {checkpoint_path}")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_strict(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
    state_key: str | None = None,
) -> str:
    """Load an exact checkpoint or raise; random fallback is forbidden."""

    expected = expected_sha256.strip().lower()
    if len(expected) != 64:
        raise ValueError("expected_sha256 must be a 64-character SHA-256 digest")
    actual = sha256_file(checkpoint_path)
    if actual != expected:
        raise ValueError(
            f"Foundation checkpoint SHA-256 mismatch: expected {expected}, got {actual}"
        )
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if state_key is not None:
        if not isinstance(payload, Mapping) or state_key not in payload:
            raise KeyError(f"Checkpoint does not contain state key {state_key!r}")
        payload = payload[state_key]
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint state must be a mapping of parameter names to tensors")
    model.load_state_dict(payload, strict=True)
    return actual


class FrozenFoundationEncoder(nn.Module):
    """Freeze an encoder with ``[B,19,12,S] → [B,19,12,D]`` contract."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        token_dim: int = 200,
        samples_per_token: int = 200,
        seconds_per_call: int = 4,
    ) -> None:
        super().__init__()
        if token_dim < 1 or samples_per_token < 1 or seconds_per_call < 1:
            raise ValueError(
                "token_dim, samples_per_token, and seconds_per_call must be positive"
            )
        self.backbone = backbone
        self.token_dim = int(token_dim)
        self.samples_per_token = int(samples_per_token)
        self.seconds_per_call = int(seconds_per_call)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "FrozenFoundationEncoder":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        expected_input_tail = (
            N_STANDARD_CHANNELS,
            self.seconds_per_call,
            self.samples_per_token,
        )
        if patches.ndim != 4 or tuple(patches.shape[1:]) != expected_input_tail:
            raise ValueError(
                f"Foundation input must have shape [B,{expected_input_tail[0]},"
                f"{expected_input_tail[1]},{expected_input_tail[2]}], "
                f"got {tuple(patches.shape)}"
            )
        if not patches.is_floating_point() or not torch.isfinite(patches).all():
            raise ValueError("Foundation input must be finite floating-point EEG patches")
        self.backbone.eval()
        with torch.no_grad():
            tokens = self.backbone(patches)
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("Foundation backbone must return a token tensor")
        expected_output = (
            patches.shape[0],
            N_STANDARD_CHANNELS,
            self.seconds_per_call,
            self.token_dim,
        )
        if tuple(tokens.shape) != expected_output:
            raise ValueError(
                "Foundation tokens must have shape "
                f"{expected_output}, got {tuple(tokens.shape)}"
            )
        if not torch.isfinite(tokens).all():
            raise ValueError("Foundation encoder returned non-finite tokens")
        return tokens.detach()


class TiledFoundationEncoder(nn.Module):
    """Apply equal independent calls to cover the fixed 60-second window."""

    def __init__(self, encoder: FrozenFoundationEncoder, *, n_calls: int = 15) -> None:
        super().__init__()
        if n_calls < 1:
            raise ValueError("n_calls must be positive")
        self.encoder = encoder
        self.n_calls = int(n_calls)

    @property
    def n_samples(self) -> int:
        return (
            self.n_calls
            * self.encoder.seconds_per_call
            * self.encoder.samples_per_token
        )

    @property
    def n_tokens(self) -> int:
        return self.n_calls * self.encoder.seconds_per_call

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        expected = (N_STANDARD_CHANNELS, self.n_samples)
        if eeg.ndim != 3 or tuple(eeg.shape[1:]) != expected:
            raise ValueError(
                f"Tiled EEG must have shape [B,{expected[0]},{expected[1]}], "
                f"got {tuple(eeg.shape)}"
            )
        batch_size = eeg.shape[0]
        patches = eeg.reshape(
            batch_size,
            N_STANDARD_CHANNELS,
            self.n_calls,
            self.encoder.seconds_per_call,
            self.encoder.samples_per_token,
        )
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.reshape(
            batch_size * self.n_calls,
            N_STANDARD_CHANNELS,
            self.encoder.seconds_per_call,
            self.encoder.samples_per_token,
        )
        tokens = self.encoder(patches)
        tokens = tokens.reshape(
            batch_size,
            self.n_calls,
            N_STANDARD_CHANNELS,
            self.encoder.seconds_per_call,
            self.encoder.token_dim,
        )
        tokens = tokens.permute(0, 2, 1, 3, 4).contiguous()
        return tokens.reshape(
            batch_size,
            N_STANDARD_CHANNELS,
            self.n_tokens,
            self.encoder.token_dim,
        )


class OverlappingContextFoundationEncoder(nn.Module):
    """Restore native local context while preserving one token per second.

    The fixed recovery interface covers a 60-second, 200 Hz event with
    fourteen eight-second calls starting at ``0, 4, ..., 52`` seconds.  Calls
    are correlated views, never additional samples.  Tokens that describe the
    same absolute second are combined by an equal, inverse-coverage mean, so
    the returned carrier remains ``[B,19,60,D]`` and can be compared with the
    existing four-second non-overlap representation using the same head.
    """

    total_seconds = 60
    context_seconds = 8
    stride_seconds = 4

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        required = {
            "seconds_per_call": self.context_seconds,
            "samples_per_token": 200,
            "token_dim": 200,
        }
        for name, expected in required.items():
            actual = getattr(encoder, name, None)
            if actual != expected:
                raise ValueError(
                    f"overlapping context encoder requires {name}={expected}, "
                    f"got {actual!r}"
                )
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            if parameter.requires_grad:
                raise ValueError("overlapping context requires a frozen encoder")
        self.encoder.eval()
        starts = tuple(
            range(
                0,
                self.total_seconds - self.context_seconds + 1,
                self.stride_seconds,
            )
        )
        if starts != tuple(range(0, 53, 4)):
            raise RuntimeError("native-context start contract drifted")
        self.register_buffer(
            "start_seconds",
            torch.tensor(starts, dtype=torch.long),
            persistent=True,
        )
        coverage = torch.zeros(self.total_seconds, dtype=torch.long)
        for start in starts:
            coverage[start : start + self.context_seconds] += 1
        if coverage.tolist() != [1] * 4 + [2] * 52 + [1] * 4:
            raise RuntimeError("native-context coverage contract drifted")
        self.register_buffer("coverage_counts", coverage, persistent=True)

    @property
    def n_calls(self) -> int:
        return int(self.start_seconds.numel())

    @property
    def n_samples(self) -> int:
        return self.total_seconds * int(self.encoder.samples_per_token)

    def train(self, mode: bool = True) -> "OverlappingContextFoundationEncoder":
        super().train(mode)
        self.encoder.eval()
        return self

    def _calls(self, eeg: torch.Tensor) -> torch.Tensor:
        expected = (N_STANDARD_CHANNELS, self.n_samples)
        if eeg.ndim != 3 or tuple(eeg.shape[1:]) != expected:
            raise ValueError(
                f"overlapping EEG must have shape [B,{expected[0]},{expected[1]}], "
                f"got {tuple(eeg.shape)}"
            )
        if not eeg.is_floating_point() or not torch.isfinite(eeg).all():
            raise ValueError("overlapping EEG must be finite floating point")
        samples = int(self.encoder.samples_per_token)
        windows = eeg.unfold(
            dimension=-1,
            size=self.context_seconds * samples,
            step=self.stride_seconds * samples,
        )
        expected_windows = (
            eeg.shape[0],
            N_STANDARD_CHANNELS,
            self.n_calls,
            self.context_seconds * samples,
        )
        if tuple(windows.shape) != expected_windows:
            raise RuntimeError("overlapping EEG unfold shape drifted")
        return (
            windows.reshape(
                eeg.shape[0],
                N_STANDARD_CHANNELS,
                self.n_calls,
                self.context_seconds,
                samples,
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

    def _aggregate(self, call_tokens: torch.Tensor) -> torch.Tensor:
        expected = (
            call_tokens.shape[0],
            self.n_calls,
            N_STANDARD_CHANNELS,
            self.context_seconds,
            int(self.encoder.token_dim),
        )
        if call_tokens.ndim != 5 or tuple(call_tokens.shape) != expected:
            raise ValueError(
                "overlapping call tokens must have shape "
                f"[B,{self.n_calls},19,8,{self.encoder.token_dim}]"
            )
        if not call_tokens.is_floating_point() or not torch.isfinite(
            call_tokens
        ).all():
            raise ValueError("overlapping call tokens must be finite floating point")
        result = call_tokens.new_zeros(
            (
                call_tokens.shape[0],
                N_STANDARD_CHANNELS,
                self.total_seconds,
                int(self.encoder.token_dim),
            )
        )
        for call_index, start in enumerate(self.start_seconds.tolist()):
            result[:, :, start : start + self.context_seconds] += call_tokens[
                :, call_index
            ]
        coverage = self.coverage_counts.to(
            device=result.device, dtype=result.dtype
        ).view(1, 1, self.total_seconds, 1)
        result = result / coverage
        if not torch.isfinite(result).all():
            raise RuntimeError("overlap aggregation returned non-finite tokens")
        return result.detach()

    def _forward_with(
        self,
        eeg: torch.Tensor,
        encode: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        calls = self._calls(eeg)
        batch_size = calls.shape[0]
        flat = calls.reshape(
            batch_size * self.n_calls,
            N_STANDARD_CHANNELS,
            self.context_seconds,
            int(self.encoder.samples_per_token),
        )
        self.encoder.eval()
        with torch.no_grad():
            encoded = encode(flat)
        expected = (
            batch_size * self.n_calls,
            N_STANDARD_CHANNELS,
            self.context_seconds,
            int(self.encoder.token_dim),
        )
        if not isinstance(encoded, torch.Tensor) or tuple(encoded.shape) != expected:
            raise ValueError(
                f"native context encoder returned invalid shape; expected {expected}"
            )
        encoded = encoded.reshape(
            batch_size,
            self.n_calls,
            N_STANDARD_CHANNELS,
            self.context_seconds,
            int(self.encoder.token_dim),
        )
        return self._aggregate(encoded)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self._forward_with(eeg, self.encoder)

    def forward_with_record_binding(
        self,
        eeg: torch.Tensor,
        binding: object,
    ) -> torch.Tensor:
        """Encode one record with position IDs derived from its EDF header."""

        method = getattr(self.encoder, "forward_with_record_binding", None)
        if method is None or not callable(method):
            raise TypeError("foundation encoder lacks record-bound forward support")
        return self._forward_with(eeg, lambda value: method(value, binding))

    def aggregate_existing_second_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pipeline control: duplicate/average 60 tokens and reproduce them."""

        expected = (
            N_STANDARD_CHANNELS,
            self.total_seconds,
            int(self.encoder.token_dim),
        )
        if tokens.ndim != 4 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(
                f"existing tokens must have shape [B,{expected[0]},"
                f"{expected[1]},{expected[2]}]"
            )
        calls = torch.stack(
            tuple(
                tokens[:, :, start : start + self.context_seconds]
                for start in self.start_seconds.tolist()
            ),
            dim=1,
        )
        return self._aggregate(calls)
