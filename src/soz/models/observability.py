"""Fail-closed monotone scalp-observability gate for future SOZ arms.

The gate is deliberately separate from the frozen v18 reasoner.  It accepts
only an already-computed reliability index and named reasoner contributions;
it has no path to raw EEG, patient identity, clinical text, or SOZ labels.
Reliability may attenuate non-negative localizing support, but it cannot alter
the channel prior or signed residual evidence.  This decomposition prevents a
lower reliability value from increasing a logit by shrinking negative
evidence toward zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class _ReasonerContributionOutput(Protocol):
    """Minimal structural interface required from a reasoner receipt."""

    event_logits: torch.Tensor

    def family_contributions(self) -> dict[str, torch.Tensor]: ...


@dataclass(frozen=True)
class ObservabilityGatedOutput:
    """Auditable result of a monotone observability intervention."""

    event_logits: torch.Tensor
    source_event_logits: torch.Tensor
    channel_prior: torch.Tensor
    nonnegative_support: torch.Tensor
    signed_residual: torch.Tensor
    reliability: torch.Tensor

    def reconstructed_logits(self) -> torch.Tensor:
        return (
            self.channel_prior
            + self.reliability.to(dtype=self.nonnegative_support.dtype)
            * self.nonnegative_support
            + self.signed_residual
        )


def _validate_channel_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} must have non-empty [B,C] shape")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def apply_monotone_observability_gate(
    *,
    channel_prior: torch.Tensor,
    nonnegative_support: torch.Tensor,
    signed_residual: torch.Tensor,
    reliability: torch.Tensor,
    source_event_logits: torch.Tensor | None = None,
) -> ObservabilityGatedOutput:
    """Attenuate positive support without changing prior or signed residual.

    ``reliability`` is a frozen target-free index in ``[0,1]`` with exact
    ``[B,C]`` shape.  It must be detached: fitting it through an SOZ loss would
    turn a reliability port into an unconstrained localization shortcut.
    """

    tensors = {
        "channel_prior": channel_prior,
        "nonnegative_support": nonnegative_support,
        "signed_residual": signed_residual,
        "reliability": reliability,
    }
    for name, value in tensors.items():
        _validate_channel_tensor(name, value)
    expected_shape = tuple(channel_prior.shape)
    if any(tuple(value.shape) != expected_shape for value in tensors.values()):
        raise ValueError("All observability tensors must share exact [B,C] shape")
    if len({value.device for value in tensors.values()}) != 1:
        raise ValueError("All observability tensors must share one device")
    if reliability.requires_grad:
        raise ValueError("Observability reliability must be detached from SOZ loss")
    if torch.any((reliability < 0) | (reliability > 1)):
        raise ValueError("Observability reliability must lie in [0,1]")
    if torch.any(nonnegative_support < 0):
        raise ValueError(
            "Observability may gate only non-negative localizing support"
        )

    event_logits = (
        channel_prior
        + reliability.to(dtype=nonnegative_support.dtype) * nonnegative_support
        + signed_residual
    )
    if source_event_logits is None:
        source_event_logits = channel_prior + nonnegative_support + signed_residual
    else:
        _validate_channel_tensor("source_event_logits", source_event_logits)
        if tuple(source_event_logits.shape) != expected_shape:
            raise ValueError("source_event_logits must share [B,C] shape")
        if source_event_logits.device != channel_prior.device:
            raise ValueError("source_event_logits must share the gate device")
        expected_source = channel_prior + nonnegative_support + signed_residual
        if not torch.allclose(source_event_logits, expected_source, rtol=1e-6, atol=1e-7):
            raise ValueError(
                "Source logits do not reconstruct from prior/support/residual"
            )

    output = ObservabilityGatedOutput(
        event_logits=event_logits,
        source_event_logits=source_event_logits,
        channel_prior=channel_prior,
        nonnegative_support=nonnegative_support,
        signed_residual=signed_residual,
        reliability=reliability,
    )
    if not torch.allclose(
        output.event_logits,
        output.reconstructed_logits(),
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("Observability receipt does not reconstruct logits")
    return output


def gate_reasoner_output(
    output: _ReasonerContributionOutput,
    reliability: torch.Tensor,
) -> ObservabilityGatedOutput:
    """Apply the gate to the typed families exposed by a reasoner receipt.

    Evolution and morphology are required to be non-negative support.  The
    ictal family may be signed; its positive part is attenuated and its
    negative part remains a signed residual.  Reliability one must reproduce
    the ungated event logits within numerical tolerance.
    """

    families = output.family_contributions()
    required = {
        "channel_prior",
        "evolution",
        "morphology",
        "ictal_involvement",
    }
    if set(families) != required:
        raise ValueError(
            "Reasoner family receipt must contain exactly channel_prior, "
            "evolution, morphology, and ictal_involvement"
        )
    evolution = families["evolution"]
    morphology = families["morphology"]
    ictal = families["ictal_involvement"]
    if torch.any(evolution < 0) or torch.any(morphology < 0):
        raise ValueError(
            "Evolution and morphology must remain non-negative support paths"
        )
    support = evolution + morphology + ictal.clamp_min(0.0)
    residual = ictal.clamp_max(0.0)
    return apply_monotone_observability_gate(
        channel_prior=families["channel_prior"],
        nonnegative_support=support,
        signed_residual=residual,
        reliability=reliability,
        source_event_logits=output.event_logits,
    )
