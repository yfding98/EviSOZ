"""Independent one-second-stride LaBraM feature path for morphology.

The ictal/evolution branches use fifteen non-overlapping four-second calls.
Morphology does not.  It makes 57 overlapping four-second calls over a real
60-second window and retains output slot zero from each call, yielding
``H_M[B,19,57,200]``.  The last three reported seconds are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from .geometry import N_STANDARD_CHANNELS, N_TCP_EDGES


MORPHOLOGY_WINDOW_SECONDS = 60
MORPHOLOGY_SAMPLES_PER_SECOND = 200
MORPHOLOGY_CONTEXT_SECONDS = 4
MORPHOLOGY_STRIDE_SECONDS = 1
MORPHOLOGY_READ_SLOT = 0
MORPHOLOGY_ANCHOR_COUNT = 57
MORPHOLOGY_TOKEN_DIM = 200
MORPHOLOGY_TILE_SECONDS = 4
MORPHOLOGY_TILE_COUNT = 15

MORPHOLOGY_PRE_ANCHORS = tuple(range(0, 9))
MORPHOLOGY_EARLY_ANCHORS = tuple(range(12, 21))
MORPHOLOGY_LATE_ANCHORS = tuple(range(24, 57))
MORPHOLOGY_PHASE_ANCHORS = (
    MORPHOLOGY_PRE_ANCHORS
    + MORPHOLOGY_EARLY_ANCHORS
    + MORPHOLOGY_LATE_ANCHORS
)
MORPHOLOGY_BOUNDARY_CROSSING_ANCHORS = (9, 10, 11, 21, 22, 23)
MORPHOLOGY_VALID_PHASE_TILES = (0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13)
MORPHOLOGY_FORBIDDEN_PHASE_TILES = (2, 5, 14)


def _bool_tensor(value: torch.Tensor, *, shape: tuple[int, ...], field: str) -> None:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{field} must have shape {shape}")
    if value.dtype != torch.bool:
        raise TypeError(f"{field} must use torch.bool")


@dataclass(frozen=True)
class MorphologyDeploymentMasks:
    """Separate reporting and full-tile phase-reasoning availability."""

    edge_available_mask: torch.Tensor
    second_available_mask: torch.Tensor
    phase_tile_mask: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.edge_available_mask, torch.Tensor):
            raise TypeError("edge_available_mask must be a tensor")
        if self.edge_available_mask.ndim != 2 or self.edge_available_mask.shape[1] != N_TCP_EDGES:
            raise ValueError("edge_available_mask must have shape [B,20]")
        if self.edge_available_mask.dtype != torch.bool:
            raise TypeError("edge_available_mask must use torch.bool")
        batch = int(self.edge_available_mask.shape[0])
        _bool_tensor(
            self.second_available_mask,
            shape=(batch, N_TCP_EDGES, MORPHOLOGY_WINDOW_SECONDS),
            field="second_available_mask",
        )
        _bool_tensor(
            self.phase_tile_mask,
            shape=(batch, N_TCP_EDGES, MORPHOLOGY_TILE_COUNT),
            field="phase_tile_mask",
        )
        if not (
            self.edge_available_mask.device
            == self.second_available_mask.device
            == self.phase_tile_mask.device
        ):
            raise ValueError("Morphology masks must share one device")
        expected_seconds = self.edge_available_mask.unsqueeze(-1).expand(
            -1, -1, MORPHOLOGY_WINDOW_SECONDS
        ).clone()
        expected_seconds[:, :, MORPHOLOGY_ANCHOR_COUNT:] = False
        if not torch.equal(self.second_available_mask, expected_seconds):
            raise ValueError(
                "second_available_mask must expose exactly anchors 0..56; "
                "seconds 57..59 are unavailable"
            )
        expected_tiles = torch.zeros_like(self.phase_tile_mask)
        expected_tiles[:, :, list(MORPHOLOGY_VALID_PHASE_TILES)] = (
            self.edge_available_mask.unsqueeze(-1)
        )
        if not torch.equal(self.phase_tile_mask, expected_tiles):
            raise ValueError(
                "phase_tile_mask must admit only complete phase-contained tiles "
                "{0,1,3,4,6,...,13}"
            )


def morphology_deployment_masks(
    batch_size: int,
    *,
    edge_available_mask: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> MorphologyDeploymentMasks:
    """Build the frozen 57-second reporting and 12-tile reasoning masks."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if edge_available_mask is None:
        edges = torch.ones((batch_size, N_TCP_EDGES), dtype=torch.bool, device=device)
    else:
        if not isinstance(edge_available_mask, torch.Tensor):
            raise TypeError("edge_available_mask must be a tensor")
        if tuple(edge_available_mask.shape) != (batch_size, N_TCP_EDGES):
            raise ValueError("edge_available_mask must have shape [B,20]")
        if edge_available_mask.dtype != torch.bool:
            raise TypeError("edge_available_mask must use torch.bool")
        edges = edge_available_mask.to(device=device) if device is not None else edge_available_mask
    seconds = torch.zeros(
        (batch_size, N_TCP_EDGES, MORPHOLOGY_WINDOW_SECONDS),
        dtype=torch.bool,
        device=edges.device,
    )
    seconds[:, :, :MORPHOLOGY_ANCHOR_COUNT] = edges.unsqueeze(-1)
    tiles = torch.zeros(
        (batch_size, N_TCP_EDGES, MORPHOLOGY_TILE_COUNT),
        dtype=torch.bool,
        device=edges.device,
    )
    tiles[:, :, list(MORPHOLOGY_VALID_PHASE_TILES)] = edges.unsqueeze(-1)
    return MorphologyDeploymentMasks(
        edge_available_mask=edges,
        second_available_mask=seconds,
        phase_tile_mask=tiles,
    )


def phase_reasoning_anchor_mask(
    *, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return a 60-element mask; boundary-crossing and unavailable anchors are false."""

    mask = torch.zeros(MORPHOLOGY_WINDOW_SECONDS, dtype=torch.bool, device=device)
    mask[list(MORPHOLOGY_PHASE_ANCHORS)] = True
    return mask


class MorphologySlidingEncoder(nn.Module):
    """Run a frozen four-second foundation call at one-second stride."""

    def __init__(self, foundation: nn.Module, *, call_microbatch_size: int = 32) -> None:
        super().__init__()
        if not isinstance(foundation, nn.Module):
            raise TypeError("foundation must be a torch module")
        if (
            isinstance(call_microbatch_size, bool)
            or not isinstance(call_microbatch_size, int)
            or call_microbatch_size < 1
        ):
            raise ValueError("call_microbatch_size must be a positive integer")
        seconds_per_call = getattr(foundation, "seconds_per_call", None)
        if seconds_per_call != MORPHOLOGY_CONTEXT_SECONDS:
            raise ValueError("Morphology requires a four-second LaBraM call")
        token_dim = getattr(foundation, "token_dim", None)
        if token_dim != MORPHOLOGY_TOKEN_DIM:
            raise ValueError("Morphology requires the audited 200-dimensional LaBraM token")
        samples_per_token = getattr(foundation, "samples_per_token", None)
        if samples_per_token != MORPHOLOGY_SAMPLES_PER_SECOND:
            raise ValueError("Morphology requires 200 samples per LaBraM token")
        self.foundation = foundation
        self.call_microbatch_size = call_microbatch_size
        for parameter in self.foundation.parameters():
            parameter.requires_grad_(False)
        self.foundation.eval()

    def train(self, mode: bool = True) -> "MorphologySlidingEncoder":
        super().train(mode)
        self.foundation.eval()
        return self

    @staticmethod
    def _validate_window(eeg: torch.Tensor) -> None:
        expected_samples = MORPHOLOGY_WINDOW_SECONDS * MORPHOLOGY_SAMPLES_PER_SECOND
        if eeg.ndim != 3 or tuple(eeg.shape[1:]) != (
            N_STANDARD_CHANNELS,
            expected_samples,
        ):
            raise ValueError(
                f"Morphology deployment EEG must have shape [B,19,{expected_samples}]"
            )
        if not eeg.is_floating_point() or not torch.isfinite(eeg).all():
            raise ValueError("Morphology deployment EEG must be finite floating point")
        if eeg.shape[0] < 1:
            raise ValueError("Morphology deployment batch cannot be empty")

    @staticmethod
    def _validate_source_crops(eeg: torch.Tensor) -> None:
        expected_samples = MORPHOLOGY_CONTEXT_SECONDS * MORPHOLOGY_SAMPLES_PER_SECOND
        if eeg.ndim != 3 or tuple(eeg.shape[1:]) != (
            N_STANDARD_CHANNELS,
            expected_samples,
        ):
            raise ValueError(
                f"Morphology source crops must have shape [N,19,{expected_samples}]"
            )
        if not eeg.is_floating_point() or not torch.isfinite(eeg).all():
            raise ValueError("Morphology source crops must be finite floating point")
        if eeg.shape[0] < 1:
            raise ValueError("Morphology source-crop batch cannot be empty")

    def _run_calls(self, calls: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        self.foundation.eval()
        with torch.no_grad():
            for start in range(0, calls.shape[0], self.call_microbatch_size):
                stop = min(start + self.call_microbatch_size, calls.shape[0])
                output = self.foundation(calls[start:stop])
                expected = (
                    stop - start,
                    N_STANDARD_CHANNELS,
                    MORPHOLOGY_CONTEXT_SECONDS,
                    MORPHOLOGY_TOKEN_DIM,
                )
                if tuple(output.shape) != expected:
                    raise ValueError(
                        f"Foundation returned {tuple(output.shape)}, expected {expected}"
                    )
                if not output.is_floating_point() or not torch.isfinite(output).all():
                    raise ValueError("Foundation returned invalid morphology tokens")
                outputs.append(output.detach())
        return torch.cat(outputs, dim=0).detach()

    def encode_source_crops(self, eeg: torch.Tensor) -> torch.Tensor:
        """Return full ``[N,19,4,200]`` tokens for slot-0 TUEV supervision."""

        self._validate_source_crops(eeg)
        calls = eeg.reshape(
            eeg.shape[0],
            N_STANDARD_CHANNELS,
            MORPHOLOGY_CONTEXT_SECONDS,
            MORPHOLOGY_SAMPLES_PER_SECOND,
        )
        return self._run_calls(calls)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """Return detached ``H_M[B,19,57,200]`` and never a 60-token cache."""

        self._validate_window(eeg)
        batch = int(eeg.shape[0])
        call_samples = MORPHOLOGY_CONTEXT_SECONDS * MORPHOLOGY_SAMPLES_PER_SECOND
        stride_samples = MORPHOLOGY_STRIDE_SECONDS * MORPHOLOGY_SAMPLES_PER_SECOND
        windows = eeg.unfold(-1, call_samples, stride_samples)
        if tuple(windows.shape) != (
            batch,
            N_STANDARD_CHANNELS,
            MORPHOLOGY_ANCHOR_COUNT,
            call_samples,
        ):
            raise RuntimeError("Sliding morphology crop construction drifted")
        calls = windows.permute(0, 2, 1, 3).contiguous().reshape(
            batch * MORPHOLOGY_ANCHOR_COUNT,
            N_STANDARD_CHANNELS,
            MORPHOLOGY_CONTEXT_SECONDS,
            MORPHOLOGY_SAMPLES_PER_SECOND,
        )
        all_slots = self._run_calls(calls)
        slot_zero = all_slots[:, :, MORPHOLOGY_READ_SLOT, :]
        tokens = slot_zero.reshape(
            batch,
            MORPHOLOGY_ANCHOR_COUNT,
            N_STANDARD_CHANNELS,
            MORPHOLOGY_TOKEN_DIM,
        ).permute(0, 2, 1, 3).contiguous()
        return tokens.detach()


@dataclass(frozen=True)
class MorphologyTilePool:
    mean: torch.Tensor
    maximum: torch.Tensor
    mask: torch.Tensor


def pool_full_phase_tiles(
    values: torch.Tensor,
    masks: MorphologyDeploymentMasks,
) -> MorphologyTilePool:
    """Pool only complete four-anchor, phase-contained morphology tiles.

    A caller cannot opt into a partial-tile mean/max: the supplied masks are
    first validated against the frozen availability contract.
    """

    if not isinstance(masks, MorphologyDeploymentMasks):
        raise TypeError("masks must be MorphologyDeploymentMasks")
    batch = int(masks.edge_available_mask.shape[0])
    if values.ndim < 4 or tuple(values.shape[:3]) != (
        batch,
        N_TCP_EDGES,
        MORPHOLOGY_WINDOW_SECONDS,
    ):
        raise ValueError("values must begin with shape [B,20,60,...]")
    if not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError("Morphology values must be finite floating point")
    if values.device != masks.second_available_mask.device:
        raise ValueError("Morphology values and masks must share one device")
    feature_shape = tuple(values.shape[3:])
    tiles = values.reshape(
        batch,
        N_TCP_EDGES,
        MORPHOLOGY_TILE_COUNT,
        MORPHOLOGY_TILE_SECONDS,
        *feature_shape,
    )
    second_tiles = masks.second_available_mask.reshape(
        batch, N_TCP_EDGES, MORPHOLOGY_TILE_COUNT, MORPHOLOGY_TILE_SECONDS
    )
    if torch.any(masks.phase_tile_mask & ~second_tiles.all(dim=-1)):
        raise ValueError("A phase tile was admitted without all four observed anchors")
    mean = tiles.mean(dim=3)
    maximum = tiles.amax(dim=3)
    expand_dims = (1,) * len(feature_shape)
    expanded_mask = masks.phase_tile_mask.reshape(
        *masks.phase_tile_mask.shape, *expand_dims
    )
    mean = torch.where(expanded_mask, mean, torch.zeros_like(mean))
    maximum = torch.where(expanded_mask, maximum, torch.zeros_like(maximum))
    return MorphologyTilePool(
        mean=mean,
        maximum=maximum,
        mask=masks.phase_tile_mask,
    )


__all__ = [
    "MORPHOLOGY_ANCHOR_COUNT",
    "MORPHOLOGY_BOUNDARY_CROSSING_ANCHORS",
    "MORPHOLOGY_CONTEXT_SECONDS",
    "MORPHOLOGY_EARLY_ANCHORS",
    "MORPHOLOGY_FORBIDDEN_PHASE_TILES",
    "MORPHOLOGY_LATE_ANCHORS",
    "MORPHOLOGY_PHASE_ANCHORS",
    "MORPHOLOGY_PRE_ANCHORS",
    "MORPHOLOGY_READ_SLOT",
    "MORPHOLOGY_SAMPLES_PER_SECOND",
    "MORPHOLOGY_STRIDE_SECONDS",
    "MORPHOLOGY_TILE_COUNT",
    "MORPHOLOGY_TOKEN_DIM",
    "MORPHOLOGY_VALID_PHASE_TILES",
    "MORPHOLOGY_WINDOW_SECONDS",
    "MorphologyDeploymentMasks",
    "MorphologySlidingEncoder",
    "MorphologyTilePool",
    "morphology_deployment_masks",
    "phase_reasoning_anchor_mask",
    "pool_full_phase_tiles",
]
