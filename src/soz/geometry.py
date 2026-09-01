"""Canonical electrode and bipolar-edge geometry for NEB-LaBraM-v2.

The SOZ target lives on 19 physical scalp electrodes.  TUEV/TUSZ concept
labels live on bipolar derivations, represented here as an independent set of
20 undirected edges.  Nothing in this module expands an edge label into an
electrode target.
"""

from __future__ import annotations

from typing import Final

import torch


STANDARD_19: Final[tuple[str, ...]] = (
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "P7",
    "P3",
    "PZ",
    "P4",
    "P8",
    "O1",
    "O2",
)

TCP_20_EDGES: Final[tuple[tuple[str, str], ...]] = (
    ("FP1", "F7"),
    ("F7", "T7"),
    ("T7", "P7"),
    ("P7", "O1"),
    ("FP2", "F8"),
    ("F8", "T8"),
    ("T8", "P8"),
    ("P8", "O2"),
    ("T7", "C3"),
    ("C3", "CZ"),
    ("CZ", "C4"),
    ("C4", "T8"),
    ("FP1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("FP2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
)

MORPHOLOGY_CLASSES: Final[tuple[str, ...]] = (
    "SPSW",
    "GPED",
    "PLED",
    "EYEM",
    "ARTF",
    "BCKG",
)

EVOLUTION_FEATURES: Final[tuple[str, ...]] = (
    "log_rms",
    "log_line_length",
    "spectral_centroid",
    "normalized_spectral_entropy",
    "rhythmicity",
    "mean_neighbor_coherence",
)

LEGACY_TO_CANONICAL: Final[dict[str, str]] = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}

CHANNEL_INDEX: Final[dict[str, int]] = {
    channel: index for index, channel in enumerate(STANDARD_19)
}
EDGE_INDEX: Final[dict[tuple[str, str], int]] = {
    edge: index for index, edge in enumerate(TCP_20_EDGES)
}

N_STANDARD_CHANNELS: Final[int] = len(STANDARD_19)
N_TCP_EDGES: Final[int] = len(TCP_20_EDGES)
N_TIME_TILES: Final[int] = 15
N_MORPHOLOGY_FEATURES: Final[int] = 12
N_ICTAL_FEATURES: Final[int] = 2
N_EDGE_FEATURES: Final[int] = N_MORPHOLOGY_FEATURES + N_ICTAL_FEATURES
N_NODE_FEATURES: Final[int] = len(EVOLUTION_FEATURES)


def normalize_electrode_name(name: object) -> str:
    """Normalize identity aliases without making a spatial substitution.

    Outside-head names such as ``A1`` or ``SPHL`` are returned as such and do
    not become a standard-19 channel.  Bipolar strings are not parsed here.
    """

    text = str(name).strip().upper().replace("_", "-")
    for prefix in ("EEG ", "EEG-", "EEG_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    for suffix in ("-REF", "-LE", "-AR", "-AVG", "-AV", "-CAR"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    text = text.strip("- ")
    return LEGACY_TO_CANONICAL.get(text, text)


def require_standard19(name: object) -> str:
    """Return a canonical standard-19 name or raise on an outside electrode."""

    canonical = normalize_electrode_name(name)
    if canonical not in CHANNEL_INDEX:
        raise ValueError(f"Electrode {name!r} is not in the standard-19 output head")
    return canonical


def edge_endpoint_indices(*, device: torch.device | str | None = None) -> torch.Tensor:
    """Return endpoint indices with shape ``[20, 2]`` in fixed TCP order."""

    return torch.tensor(
        [[CHANNEL_INDEX[left], CHANNEL_INDEX[right]] for left, right in TCP_20_EDGES],
        dtype=torch.long,
        device=device,
    )


def unsigned_incidence_matrix(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return an unsigned node-by-edge incidence matrix with shape ``[19,20]``.

    Edge evidence is symmetric with respect to its two endpoints.  Bipolar
    polarity is useful when learning the edge concept itself, but it is not
    evidence that one endpoint rather than the other is the SOZ.
    """

    incidence = torch.zeros(
        (N_STANDARD_CHANNELS, N_TCP_EDGES), dtype=dtype, device=device
    )
    endpoints = edge_endpoint_indices(device=device)
    edge_ids = torch.arange(N_TCP_EDGES, device=device)
    incidence[endpoints[:, 0], edge_ids] = 1
    incidence[endpoints[:, 1], edge_ids] = 1
    return incidence


if len(set(STANDARD_19)) != N_STANDARD_CHANNELS:
    raise RuntimeError("STANDARD_19 contains duplicate electrodes")
if any(endpoint not in CHANNEL_INDEX for edge in TCP_20_EDGES for endpoint in edge):
    raise RuntimeError("TCP_20_EDGES contains an endpoint outside STANDARD_19")

