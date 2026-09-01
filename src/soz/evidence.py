"""Strict finite evidence contract consumed by the SOZ reasoner."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .evidence_schema import (
    EVIDENCE_TENSOR_SEMANTICS_SHA256,
    TypedMorphologyEvidence,
    split_typed_morphology,
)
from .geometry import (
    N_EDGE_FEATURES,
    N_NODE_FEATURES,
    N_STANDARD_CHANNELS,
    N_TCP_EDGES,
)
from .temporal_masks import physical_node_to_edge_mask


@dataclass(frozen=True)
class EvidenceBatch:
    """A batch of concept evidence with explicit availability masks.

    Parameters
    ----------
    node:
        Physical-electrode evolution evidence, ``[B,19,T,6]``.
    edge:
        Typed bipolar carrier, ``[B,20,T,14]``.  Columns are frozen as the CE6
        mean block ``SPSW,GPED,PLED,EYEM,ARTF,BCKG``, the same CE6 maximum
        block, then ictal-involvement mean/maximum.  The semantic digest is
        exposed by :attr:`evidence_semantics_sha256`; this is never an
        unconstrained 14-feature latent.
    node_mask:
        Boolean availability mask, ``[B,19,T]``.
    edge_mask:
        Boolean physical edge availability mask, ``[B,20,T]``, derived from
        the two endpoint entries in ``physical_signal_mask``.
    physical_signal_mask:
        Actual physical node/time signal validity, ``[B,19,T]``.  This is
        separate from producer availability and source annotation coverage.
    ictal_phase_mask:
        Offset-aware primary phase validity, ``[B,T]``. Pre-anchor context is
        all-valid only when the previous-seizure timeline rules out overlap;
        it is not assumed interictal baseline. Transition/postictal tiles are
        excluded, and post-onset validity must be a prefix generated from the
        frozen timing policy.
    morphology_mask:
        Localizing morphology mask ``[B,20,T]``.  It is true only when the
        frozen SPSW/PLED candidate rule authorizes positive localization
        support.  It must be a subset of ``morphology_context_mask``.
    morphology_context_mask:
        CE6 context/quality availability ``[B,20,T]``. GPED, EYEM/ARTF and
        BCKG use this mask even when the localizing candidate rule fails.
        When omitted it inherits ``morphology_mask`` for backward-compatible
        diagnostic construction; formal morphology caches must provide it.
    ictal_mask:
        Ictal-family deployment mask ``[B,20,T]``. Keeping family and
        morphology-port masks separate prevents one unavailable concept port
        from erasing valid evidence from another.

    Missing evidence must be finite-filled and masked.  A numeric zero without
    a false mask is treated as an observed value, never as an implicit missing
    marker.
    """

    node: torch.Tensor
    edge: torch.Tensor
    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    physical_signal_mask: torch.Tensor
    ictal_phase_mask: torch.Tensor
    morphology_mask: torch.Tensor | None = None
    morphology_context_mask: torch.Tensor | None = None
    ictal_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.morphology_mask is None:
            object.__setattr__(self, "morphology_mask", self.edge_mask)
        if self.morphology_context_mask is None:
            object.__setattr__(
                self, "morphology_context_mask", self.morphology_mask
            )
        if self.ictal_mask is None:
            object.__setattr__(self, "ictal_mask", self.edge_mask)
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.node.shape[0])

    @property
    def n_tiles(self) -> int:
        return int(self.node.shape[2])

    @property
    def evidence_semantics_sha256(self) -> str:
        return EVIDENCE_TENSOR_SEMANTICS_SHA256

    def typed_morphology(
        self, *, edge_mask: torch.Tensor | None = None
    ) -> TypedMorphologyEvidence:
        """Return named morphology ports under the authoritative schema."""

        mask = self.morphology_mask if edge_mask is None else edge_mask
        if mask is None:  # pragma: no cover - initialized in __post_init__
            raise RuntimeError("Morphology mask was not initialized")
        return split_typed_morphology(
            self.edge[..., :12],
            mask,
            self.morphology_context_mask,
        )

    def validate(self) -> None:
        if self.node.ndim != 4:
            raise ValueError(
                f"node must have shape [B,19,T,6], got {tuple(self.node.shape)}"
            )
        if self.edge.ndim != 4:
            raise ValueError(
                f"edge must have shape [B,20,T,14], got {tuple(self.edge.shape)}"
            )

        batch_size, n_nodes, n_tiles, n_node_features = self.node.shape
        edge_batch, n_edges, edge_tiles, n_edge_features = self.edge.shape
        expected_node = (N_STANDARD_CHANNELS, N_NODE_FEATURES)
        expected_edge = (N_TCP_EDGES, N_EDGE_FEATURES)
        if (n_nodes, n_node_features) != expected_node:
            raise ValueError(
                f"node expected channel/features {expected_node}, "
                f"got {(n_nodes, n_node_features)}"
            )
        if (n_edges, n_edge_features) != expected_edge:
            raise ValueError(
                f"edge expected edge/features {expected_edge}, "
                f"got {(n_edges, n_edge_features)}"
            )
        if batch_size != edge_batch or n_tiles != edge_tiles:
            raise ValueError("node and edge batches must share B and T")
        if batch_size < 1 or n_tiles < 1:
            raise ValueError("EvidenceBatch requires non-empty batch and time axes")

        if tuple(self.node_mask.shape) != (batch_size, n_nodes, n_tiles):
            raise ValueError(
                "node_mask must have shape [B,19,T], got "
                f"{tuple(self.node_mask.shape)}"
            )
        if tuple(self.edge_mask.shape) != (batch_size, n_edges, n_tiles):
            raise ValueError(
                "edge_mask must have shape [B,20,T], got "
                f"{tuple(self.edge_mask.shape)}"
            )
        if tuple(self.physical_signal_mask.shape) != (
            batch_size,
            n_nodes,
            n_tiles,
        ):
            raise ValueError("physical_signal_mask must have shape [B,19,T]")
        if tuple(self.ictal_phase_mask.shape) != (batch_size, n_tiles):
            raise ValueError("ictal_phase_mask must have shape [B,T]")
        family_masks = (
            self.morphology_mask,
            self.morphology_context_mask,
            self.ictal_mask,
        )
        if any(mask is None for mask in family_masks):
            raise RuntimeError("Family masks were not initialized")
        for name, mask in (
            ("morphology_mask", self.morphology_mask),
            ("morphology_context_mask", self.morphology_context_mask),
            ("ictal_mask", self.ictal_mask),
        ):
            if tuple(mask.shape) != (batch_size, n_edges, n_tiles):
                raise ValueError(f"{name} must have shape [B,20,T]")
        if any(
            mask.dtype != torch.bool
            for mask in (
                self.node_mask,
                self.edge_mask,
                self.physical_signal_mask,
                self.ictal_phase_mask,
                self.morphology_mask,
                self.morphology_context_mask,
                self.ictal_mask,
            )
        ):
            raise TypeError("All evidence masks must be torch.bool")
        if (self.morphology_mask & ~self.edge_mask).any() or (
            self.ictal_mask & ~self.edge_mask
        ).any():
            raise ValueError("Family-specific masks must be subsets of edge_mask")
        if (self.morphology_context_mask & ~self.edge_mask).any():
            raise ValueError(
                "morphology_context_mask must be a subset of edge_mask"
            )
        if (self.morphology_mask & ~self.morphology_context_mask).any():
            raise ValueError(
                "Localizing morphology mask must be a subset of context mask"
            )
        if (self.node_mask & ~self.physical_signal_mask).any():
            raise ValueError("node_mask must be a subset of physical_signal_mask")
        expected_physical_edges = physical_node_to_edge_mask(
            self.physical_signal_mask
        )
        if not torch.equal(self.edge_mask, expected_physical_edges):
            raise ValueError(
                "edge_mask must equal physical availability of both edge endpoints"
            )

        # Pre-anchor context is accepted or rejected as one conservative
        # block: any overlap with a previous seizure invalidates all three
        # tiles. After t0, primary validity can only end once: a crossing/
        # offset tile and every later tile are excluded. This catches
        # arbitrary masks without guessing timing from the crop boundary.
        if n_tiles != 15:
            raise ValueError("Offset-aware evidence requires exactly 15 time tiles")
        pre_context = self.ictal_phase_mask[:, :3]
        if (pre_context.any(dim=1) != pre_context.all(dim=1)).any():
            raise ValueError(
                "ictal_phase_mask must accept or reject all pre-anchor tiles together"
            )
        post_onset = self.ictal_phase_mask[:, 3:]
        if (post_onset[:, 1:] & ~post_onset[:, :-1]).any():
            raise ValueError(
                "ictal_phase_mask must be prefix-valid after seizure onset"
            )

        devices = {
            self.node.device,
            self.edge.device,
            self.node_mask.device,
            self.edge_mask.device,
            self.physical_signal_mask.device,
            self.ictal_phase_mask.device,
            self.morphology_mask.device,
            self.morphology_context_mask.device,
            self.ictal_mask.device,
        }
        if len(devices) != 1:
            raise ValueError("All evidence tensors and masks must share a device")
        if not self.node.is_floating_point() or not self.edge.is_floating_point():
            raise TypeError("node and edge evidence must be floating-point tensors")
        if not torch.isfinite(self.node).all() or not torch.isfinite(self.edge).all():
            raise ValueError("Evidence tensors must be finite; use masks for missing values")

    def to(self, *args: object, **kwargs: object) -> "EvidenceBatch":
        """Move evidence while preserving boolean mask dtypes."""

        node = self.node.to(*args, **kwargs)
        edge = self.edge.to(*args, **kwargs)
        device = node.device
        return EvidenceBatch(
            node=node,
            edge=edge,
            node_mask=self.node_mask.to(device=device),
            edge_mask=self.edge_mask.to(device=device),
            physical_signal_mask=self.physical_signal_mask.to(device=device),
            ictal_phase_mask=self.ictal_phase_mask.to(device=device),
            morphology_mask=self.morphology_mask.to(device=device),
            morphology_context_mask=self.morphology_context_mask.to(
                device=device
            ),
            ictal_mask=self.ictal_mask.to(device=device),
        )

    def detach(self) -> "EvidenceBatch":
        """Create the stop-gradient view required at the evidence bottleneck."""

        return EvidenceBatch(
            node=self.node.detach(),
            edge=self.edge.detach(),
            node_mask=self.node_mask,
            edge_mask=self.edge_mask,
            physical_signal_mask=self.physical_signal_mask,
            ictal_phase_mask=self.ictal_phase_mask,
            morphology_mask=self.morphology_mask,
            morphology_context_mask=self.morphology_context_mask,
            ictal_mask=self.ictal_mask,
        )
