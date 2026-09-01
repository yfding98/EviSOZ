"""Development-only I+V evidence reasoner with a closed authorization boundary.

This module is deliberately separate from :mod:`soz.formal_reasoner_pipeline`.
It does not reinterpret the upstream LaBraM-k31 ``reasoner_authorized=false``
flag as permission.  Instead, :func:`issue_development_iv_evidence_capability`
reviews the exact v1.2 I artifact and the two strict V+A/Q artifacts against a
closed *candidate-only* policy.  The resulting capability can be joined to a
strictly verified DeepSOZ target-v2 artifact, but it can never satisfy the
formal evidence issuer or formal reasoner APIs.

The model input contains only finite, detached evidence:

* I: TUSZ retrospective bipolar involvement probabilities pooled from
  ``[E,20,60]`` one-second scores to auditable four-second mean/max values
  ``[E,20,15,2]``;
* V: six scaled, observable evolution descriptors ``[E,19,15,6]``; and
* A/Q: a deterministic node reliability multiplier ``[E,19,15]`` plus an
  abstention recommendation.  Quality diagnostics and burden components are
  intentionally discarded before the model boundary.

There is no morphology tensor, raw EEG, foundation latent, path, report, LLM,
patient identity, target, or source annotation mask in the model input.  The
reasoner uses fixed unsigned, valid-degree-normalized edge-to-node routing.
A/Q can only attenuate the positive part of an already computed I/V
contribution; negative evidence and the channel prior are unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator, Mapping, Sequence

import torch
import torch.nn as nn

from .aggregation import PatientAggregation, aggregate_patient_logits
from .data.deepsoz import normalize_patient_id
from .data.deepsoz_target_v2 import (
    TARGET_V2_POLICY_SHA256,
    VerifiedDeepSOZTargetV2Artifact,
    _build_target_frame,
    _target_csv_bytes,
)
from .development_vaq import (
    DEVELOPMENT_VAQ_PURPOSE,
    DEVELOPMENT_VAQ_SPLIT,
    load_development_vaq_evidence,
)
from .development_vaq_oof import (
    SOURCE_TRAIN_OOF_VAQ_PURPOSE,
    SOURCE_TRAIN_SPLIT,
    load_source_train_oof_vaq_evidence,
)
from .geometry import (
    N_ICTAL_FEATURES,
    N_NODE_FEATURES,
    N_STANDARD_CHANNELS,
    N_TCP_EDGES,
    unsigned_incidence_matrix,
)
from .ictal_prediction_artifacts import (
    _canonical_json_bytes as _ictal_manifest_canonical_bytes,
    _tensor_sha256 as _ictal_tensor_sha256,
)
from .ictal_recovery_evidence_v1_2 import (
    LABRAM_K31_DEVELOPMENT_SCORE_PURPOSE_V1_2,
    LABRAM_K31_DEVELOPMENT_SCORE_SCHEMA_V1_2,
    VerifiedLaBraMK31DevelopmentScoreArtifactV12,
)
from .losses import PatientLevelSOZObjective, SOZLossOutput
from .models.reasoner import (
    N_REASONER_TILES,
    PHASE_COMPONENT_NAMES,
    _FamilyScorer,
    _PhaseCombiner,
    _PositiveLocalizingScorer,
    _PositivePhaseCombiner,
    _phase_components,
)
from .temporal_masks import physical_node_to_edge_mask


DEVELOPMENT_IV_AUTHORIZATION_SCHEMA = (
    "soz_development_iv_candidate_evidence_authorization_v1"
)
DEVELOPMENT_IV_CAPABILITY_SCHEMA = "soz_development_iv_evidence_capability_v1"
DEVELOPMENT_IV_DATASET_SCHEMA = "soz_development_iv_reasoner_dataset_v1"
DEVELOPMENT_IV_REASONER_SCHEMA = "soz_development_iv_additive_reasoner_v1"
DEVELOPMENT_IV_EXPLANATION_MODE = (
    "numeric_additive_contribution_decomposition_only_no_llm_prediction"
)

ACTIVE_EVIDENCE_FAMILIES = ("ictal_involvement", "temporal_evolution")
ABSENT_EVIDENCE_FAMILIES = ("morphology",)
_ALLOWED_SPLITS = ("source_train", "source_dev")
_PHASE_BOUNDS = ((0, 3), (3, 6), (6, 15))
_CAPABILITY_MARKER = object()
_DATASET_MARKER = object()
_BUNDLE_MARKER = object()
_PATIENT_BATCH_MARKER = object()
DEVELOPMENT_IV_CAPABILITY_MANIFEST_FILENAME = "manifest.json"
DEVELOPMENT_IV_CAPABILITY_EVENTS_FILENAME = "events.json"
DEVELOPMENT_IV_CAPABILITY_TENSORS_FILENAME = "evidence.safetensors"
_CAPABILITY_FILE_SET = {
    DEVELOPMENT_IV_CAPABILITY_MANIFEST_FILENAME,
    DEVELOPMENT_IV_CAPABILITY_EVENTS_FILENAME,
    DEVELOPMENT_IV_CAPABILITY_TENSORS_FILENAME,
}
_MAX_CAPABILITY_JSON_BYTES = 64 * 1024 * 1024
_MAX_CAPABILITY_TENSOR_BYTES = 512 * 1024 * 1024


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Development reasoner receipt is not canonical JSON") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _tensor_sha256(name: str, value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    metadata = f"{name}|{tuple(tensor.shape)}|{tensor.dtype}".encode("ascii")
    digest.update(len(metadata).to_bytes(4, "little"))
    digest.update(metadata)
    raw = tensor.view(torch.uint8).numpy().tobytes()
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)
    return digest.hexdigest()


def _require_sha256(value: object, *, field_name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(raw: bytes, *, field_name: str) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{field_name} contains duplicate field {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        raise ValueError(f"{field_name} contains non-finite constant {value}")

    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} is not strict UTF-8 JSON") from exc


def _safe_new_directory(path: str | Path, *, field_name: str) -> Path:
    target = Path(os.path.abspath(path))
    if target.name in {"", ".", ".."}:
        raise ValueError(f"{field_name} requires a concrete directory")
    for component in (target.parent, *target.parent.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field_name} path cannot contain symlink components")
    if os.path.lexists(target):
        raise FileExistsError(f"{field_name} already exists: {target}")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"{field_name} parent does not exist")
    return target


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


_DEVELOPMENT_IV_AUTHORIZATION_POLICY = {
    "schema_version": DEVELOPMENT_IV_AUTHORIZATION_SCHEMA,
    "status": "development_candidate_only_not_formal",
    "authorization_basis": (
        "explicit_user_requested_development_policy_amendment_20260810;"
        "not_a_reinterpretation_of_upstream_reasoner_authorized_false"
    ),
    "active_evidence_families": list(ACTIVE_EVIDENCE_FAMILIES),
    "absent_evidence_families": list(ABSENT_EVIDENCE_FAMILIES),
    "ictal_input": (
        "strict_v1_2_source_train_patient_oof_and_source_dev_final_scores_only"
    ),
    "ictal_pooling": "four_nonoverlapping_seconds_to_mean_and_max",
    "ictal_routing": "fixed_unsigned_valid_degree_normalized_incidence",
    "evolution_input": "strict_scaled_six_descriptor_vaq_evidence_only",
    "quality_access": (
        "reliability_multiplier_and_abstention_only;"
        "diagnostics_and_burdens_forbidden_from_learned_path"
    ),
    "quality_monotonicity": (
        "multiply_positive_contribution_only;negative_and_prior_unchanged"
    ),
    "target_access": "strict_verified_deepsoz_target_v2_at_reasoner_join_only",
    "event_aggregation": "equal_event_mean_logits_before_patient_loss",
    "objective": "patient_macro_masked_balanced_bce_plus_0.25_pairwise_ranking",
    "source_eval_allowed": False,
    "private_allowed": False,
    "formal_promotion": False,
    "formal_reasoner_authorized": False,
    "llm_prediction_allowed": False,
}
DEVELOPMENT_IV_AUTHORIZATION_POLICY_SHA256 = _canonical_sha256(
    _DEVELOPMENT_IV_AUTHORIZATION_POLICY
)


def pool_ictal_seconds_to_tiles(
    scores: torch.Tensor,
    deployment_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool 60 one-second I scores into 15 complete four-second mean/max tiles.

    A tile is deployable only when all four seconds are available.  Partial
    availability is masked, never imputed or silently treated as negative.
    """

    if scores.ndim != 3 or tuple(scores.shape[1:]) != (N_TCP_EDGES, 60):
        raise ValueError("I scores must have shape [E,20,60]")
    if deployment_mask.dtype != torch.bool or deployment_mask.shape != scores.shape:
        raise TypeError("I deployment mask must be bool with shape [E,20,60]")
    if scores.shape[0] < 1 or not scores.is_floating_point():
        raise ValueError("I scores must be a non-empty floating-point tensor")
    if scores.requires_grad or not torch.isfinite(scores).all():
        raise ValueError("I scores must be finite detached evidence")
    observed = scores[deployment_mask]
    if observed.numel() and torch.any((observed < 0.0) | (observed > 1.0)):
        raise ValueError("Observed I scores must lie in [0,1]")

    seconds = scores.reshape(scores.shape[0], N_TCP_EDGES, N_REASONER_TILES, 4)
    masks = deployment_mask.reshape(
        scores.shape[0], N_TCP_EDGES, N_REASONER_TILES, 4
    )
    tile_mask = masks.all(dim=-1)
    safe = torch.where(masks, seconds, torch.zeros_like(seconds))
    mean = safe.mean(dim=-1)
    maximum = torch.where(
        masks, seconds, torch.full_like(seconds, -torch.inf)
    ).amax(dim=-1)
    mean = torch.where(tile_mask, mean, torch.zeros_like(mean))
    maximum = torch.where(tile_mask, maximum, torch.zeros_like(maximum))
    return torch.stack((mean, maximum), dim=-1).contiguous(), tile_mask.contiguous()


@dataclass(frozen=True)
class DevelopmentIVEvidenceBatch:
    """Finite model input; deliberately has no M/raw/latent/path/text/label port."""

    evolution: torch.Tensor
    ictal: torch.Tensor
    evolution_mask: torch.Tensor
    ictal_mask: torch.Tensor
    phase_mask: torch.Tensor
    reliability: torch.Tensor
    event_abstain: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.evolution.shape[0])

    @property
    def n_tiles(self) -> int:
        return int(self.evolution.shape[2])

    def validate(self) -> None:
        if self.evolution.ndim != 4 or tuple(self.evolution.shape[1:]) != (
            N_STANDARD_CHANNELS,
            N_REASONER_TILES,
            N_NODE_FEATURES,
        ):
            raise ValueError("V evidence must have shape [E,19,15,6]")
        batch = self.evolution.shape[0]
        if batch < 1 or self.ictal.ndim != 4 or tuple(self.ictal.shape) != (
            batch,
            N_TCP_EDGES,
            N_REASONER_TILES,
            N_ICTAL_FEATURES,
        ):
            raise ValueError("I evidence must have shape [E,20,15,2]")
        expected_node = (batch, N_STANDARD_CHANNELS, N_REASONER_TILES)
        expected_edge = (batch, N_TCP_EDGES, N_REASONER_TILES)
        if tuple(self.evolution_mask.shape) != expected_node:
            raise ValueError("V mask must have shape [E,19,15]")
        if tuple(self.ictal_mask.shape) != expected_edge:
            raise ValueError("I mask must have shape [E,20,15]")
        if tuple(self.phase_mask.shape) != (batch, N_REASONER_TILES):
            raise ValueError("Phase mask must have shape [E,15]")
        if tuple(self.reliability.shape) != expected_node:
            raise ValueError("A/Q reliability must have shape [E,19,15]")
        if tuple(self.event_abstain.shape) != (batch,):
            raise ValueError("A/Q event abstention must have shape [E]")
        if any(
            value.dtype != torch.bool
            for value in (
                self.evolution_mask,
                self.ictal_mask,
                self.phase_mask,
                self.event_abstain,
            )
        ):
            raise TypeError("All availability and abstention tensors must be bool")
        if not self.evolution.is_floating_point() or not self.ictal.is_floating_point():
            raise TypeError("I/V evidence must be floating point")
        if not self.reliability.is_floating_point():
            raise TypeError("A/Q reliability must be floating point")
        tensors = (
            self.evolution,
            self.ictal,
            self.evolution_mask,
            self.ictal_mask,
            self.phase_mask,
            self.reliability,
            self.event_abstain,
        )
        if len({value.device for value in tensors}) != 1:
            raise ValueError("All development evidence tensors must share a device")
        if any(value.requires_grad for value in (self.evolution, self.ictal, self.reliability)):
            raise ValueError("Development evidence must be detached")
        if not torch.isfinite(self.evolution).all() or not torch.isfinite(self.ictal).all():
            raise ValueError("I/V evidence must be finite")
        if not torch.isfinite(self.reliability).all() or torch.any(
            (self.reliability < 0.0) | (self.reliability > 1.0)
        ):
            raise ValueError("A/Q reliability must be finite in [0,1]")
        observed_ictal = self.ictal[self.ictal_mask]
        if observed_ictal.numel() and torch.any(
            (observed_ictal < 0.0) | (observed_ictal > 1.0)
        ):
            raise ValueError("Observed I mean/max evidence must lie in [0,1]")
        physical_edge = physical_node_to_edge_mask(self.evolution_mask)
        if (self.ictal_mask & ~physical_edge).any():
            raise ValueError("I evidence cannot exist without both physical endpoints")
        pre = self.phase_mask[:, :3]
        if (pre.any(dim=1) != pre.all(dim=1)).any():
            raise ValueError("Pre-anchor phase must be accepted or rejected as a block")
        post = self.phase_mask[:, 3:]
        if (post[:, 1:] & ~post[:, :-1]).any():
            raise ValueError("Post-onset phase validity must be a prefix")

    def to(self, device: str | torch.device) -> "DevelopmentIVEvidenceBatch":
        return DevelopmentIVEvidenceBatch(
            evolution=self.evolution.to(device=device),
            ictal=self.ictal.to(device=device),
            evolution_mask=self.evolution_mask.to(device=device),
            ictal_mask=self.ictal_mask.to(device=device),
            phase_mask=self.phase_mask.to(device=device),
            reliability=self.reliability.to(device=device),
            event_abstain=self.event_abstain.to(device=device),
        )

    def index_select(self, indices: torch.Tensor) -> "DevelopmentIVEvidenceBatch":
        if indices.dtype != torch.long or indices.ndim != 1:
            raise TypeError("Evidence indices must be a one-dimensional long tensor")
        if indices.device != self.evolution.device:
            indices = indices.to(device=self.evolution.device)
        return DevelopmentIVEvidenceBatch(
            evolution=self.evolution.index_select(0, indices),
            ictal=self.ictal.index_select(0, indices),
            evolution_mask=self.evolution_mask.index_select(0, indices),
            ictal_mask=self.ictal_mask.index_select(0, indices),
            phase_mask=self.phase_mask.index_select(0, indices),
            reliability=self.reliability.index_select(0, indices),
            event_abstain=self.event_abstain.index_select(0, indices),
        )


def _evidence_batch_sha256(value: DevelopmentIVEvidenceBatch) -> str:
    return _canonical_sha256(
        {
            name: _tensor_sha256(name, getattr(value, name))
            for name in (
                "evolution",
                "ictal",
                "evolution_mask",
                "ictal_mask",
                "phase_mask",
                "reliability",
                "event_abstain",
            )
        }
    )


def _route_edge_tiles_to_nodes(
    edge_support: torch.Tensor,
    edge_mask: torch.Tensor,
    incidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if edge_support.ndim != 3 or edge_support.shape != edge_mask.shape:
        raise ValueError("Edge support/mask must share shape [E,20,15]")
    if tuple(incidence.shape) != (N_STANDARD_CHANNELS, N_TCP_EDGES):
        raise ValueError("Unsigned incidence shape drifted")
    matrix = incidence.to(device=edge_support.device, dtype=edge_support.dtype)
    valid = edge_mask.to(dtype=edge_support.dtype)
    degree = torch.einsum("ce,bet->bct", matrix, valid)
    routed = torch.einsum("ce,bet->bct", matrix, edge_support * valid)
    routed = routed / degree.clamp_min(1.0)
    node_mask = degree > 0
    return torch.where(node_mask, routed, 0.0), node_mask, degree


def _phase_minimum_reliability(
    reliability: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if reliability.shape != mask.shape or reliability.ndim != 3:
        raise ValueError("Reliability/mask must share [E,19,15]")
    values: list[torch.Tensor] = []
    valids: list[torch.Tensor] = []
    for start, stop in _PHASE_BOUNDS:
        phase_mask = mask[..., start:stop]
        valid = phase_mask.any(dim=-1)
        conservative = torch.where(
            phase_mask,
            reliability[..., start:stop],
            torch.ones_like(reliability[..., start:stop]),
        ).amin(dim=-1)
        values.append(torch.where(valid, conservative, torch.zeros_like(conservative)))
        valids.append(valid)
    pre, early, late = values
    pre_valid, early_valid, late_valid = valids
    early_pre_valid = early_valid & pre_valid
    late_early_valid = late_valid & early_valid
    component_values = torch.stack(
        (
            pre,
            early,
            late,
            torch.minimum(early, pre),
            torch.minimum(late, early),
        ),
        dim=-1,
    )
    component_mask = torch.stack(
        (pre_valid, early_valid, late_valid, early_pre_valid, late_early_valid),
        dim=-1,
    )
    return torch.where(component_mask, component_values, 0.0), component_mask


def _attenuate_positive_only(
    raw_contribution: torch.Tensor,
    reliability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if raw_contribution.shape != reliability.shape:
        raise ValueError("Contribution and phase reliability shapes differ")
    if not torch.isfinite(raw_contribution).all() or not torch.isfinite(reliability).all():
        raise ValueError("A/Q attenuation requires finite tensors")
    if torch.any((reliability < 0.0) | (reliability > 1.0)):
        raise ValueError("A/Q phase reliability must lie in [0,1]")
    positive = raw_contribution.clamp_min(0.0)
    negative = raw_contribution.clamp_max(0.0)
    gated = negative + positive * reliability
    attenuation = gated - raw_contribution
    if torch.any(attenuation > 1e-7):
        raise RuntimeError("A/Q increased a localizing contribution")
    return gated, attenuation


@dataclass(frozen=True)
class DevelopmentIVReasonerOutput:
    event_logits: torch.Tensor
    channel_prior: torch.Tensor
    evolution_tile_score: torch.Tensor
    evolution_phase_component: torch.Tensor
    evolution_phase_mask: torch.Tensor
    evolution_raw_phase_contribution: torch.Tensor
    evolution_gated_phase_contribution: torch.Tensor
    evolution_quality_attenuation: torch.Tensor
    evolution_phase_reliability: torch.Tensor
    ictal_edge_tile_support: torch.Tensor
    ictal_node_tile_support: torch.Tensor
    ictal_node_valid_degree: torch.Tensor
    ictal_phase_component: torch.Tensor
    ictal_phase_mask: torch.Tensor
    ictal_raw_phase_contribution: torch.Tensor
    ictal_gated_phase_contribution: torch.Tensor
    ictal_quality_attenuation: torch.Tensor
    ictal_phase_reliability: torch.Tensor
    event_abstain_recommended: torch.Tensor
    active_evidence_families: tuple[str, ...] = ACTIVE_EVIDENCE_FAMILIES
    absent_evidence_families: tuple[str, ...] = ABSENT_EVIDENCE_FAMILIES
    explanation_mode: str = DEVELOPMENT_IV_EXPLANATION_MODE
    formal_promotion: bool = False

    def component_contributions(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {"channel_prior": self.channel_prior}
        for index, name in enumerate(PHASE_COMPONENT_NAMES):
            result[f"evolution/raw/{name}"] = self.evolution_raw_phase_contribution[
                ..., index
            ]
            result[f"quality_attenuation/evolution/{name}"] = (
                self.evolution_quality_attenuation[..., index]
            )
            result[f"ictal_involvement/raw/{name}"] = (
                self.ictal_raw_phase_contribution[..., index]
            )
            result[f"quality_attenuation/ictal_involvement/{name}"] = (
                self.ictal_quality_attenuation[..., index]
            )
        return result

    def reconstructed_logits(self) -> torch.Tensor:
        return sum(self.component_contributions().values())

    def family_contributions(self) -> dict[str, torch.Tensor]:
        return {
            "channel_prior": self.channel_prior,
            "temporal_evolution_raw": self.evolution_raw_phase_contribution.sum(
                dim=-1
            ),
            "ictal_involvement_raw": self.ictal_raw_phase_contribution.sum(
                dim=-1
            ),
            "quality_attenuation": (
                self.evolution_quality_attenuation.sum(dim=-1)
                + self.ictal_quality_attenuation.sum(dim=-1)
            ),
        }


class DevelopmentIVAdditiveReasoner(nn.Module):
    """Small signed/additive candidate reasoner; never a formal issuer model."""

    def __init__(self, *, hidden_dim: int = 16) -> None:
        super().__init__()
        if hidden_dim != 16:
            raise ValueError("Development candidate hidden_dim is frozen at 16")
        self.evolution_scorer = _FamilyScorer(N_NODE_FEATURES, hidden_dim)
        self.evolution_phase_combiner = _PhaseCombiner()
        self.ictal_scorer = _PositiveLocalizingScorer(N_ICTAL_FEATURES)
        self.ictal_phase_combiner = _PositivePhaseCombiner()
        self.channel_bias = nn.Parameter(torch.zeros(N_STANDARD_CHANNELS))
        self.register_buffer("incidence", unsigned_incidence_matrix(), persistent=True)
        if self.n_trainable_parameters >= 50_000:
            raise ValueError("Development candidate exceeds the 50k capacity gate")

    @property
    def n_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(self, evidence: DevelopmentIVEvidenceBatch) -> DevelopmentIVReasonerOutput:
        if not isinstance(evidence, DevelopmentIVEvidenceBatch):
            raise TypeError("Development reasoner accepts DevelopmentIVEvidenceBatch only")
        evidence.validate()
        phase_node = evidence.phase_mask.unsqueeze(1)
        evolution_mask = evidence.evolution_mask & phase_node
        ictal_mask = evidence.ictal_mask & evidence.phase_mask.unsqueeze(1)

        safe_v = torch.where(
            evolution_mask.unsqueeze(-1), evidence.evolution, 0.0
        )
        evolution_tile = self.evolution_scorer(safe_v)
        evolution_tile = evolution_tile * evolution_mask.to(evolution_tile.dtype)
        evolution_component, evolution_component_mask = _phase_components(
            evolution_tile, evolution_mask
        )
        evolution_raw = self.evolution_phase_combiner(
            evolution_component, evolution_component_mask
        )
        evolution_reliability, evolution_reliability_mask = (
            _phase_minimum_reliability(evidence.reliability, evolution_mask)
        )
        if not torch.equal(evolution_component_mask, evolution_reliability_mask):
            raise RuntimeError("V phase reliability mask drifted")
        evolution_gated, evolution_attenuation = _attenuate_positive_only(
            evolution_raw, evolution_reliability
        )

        safe_i = torch.where(ictal_mask.unsqueeze(-1), evidence.ictal, 0.0)
        ictal_edge_support = self.ictal_scorer(safe_i)
        ictal_edge_support = ictal_edge_support * ictal_mask.to(
            ictal_edge_support.dtype
        )
        ictal_node_support, ictal_node_mask, ictal_degree = (
            _route_edge_tiles_to_nodes(
                ictal_edge_support, ictal_mask, self.incidence
            )
        )
        ictal_component, ictal_component_mask = _phase_components(
            ictal_node_support, ictal_node_mask
        )
        ictal_raw = self.ictal_phase_combiner(
            ictal_component, ictal_component_mask
        )
        ictal_reliability, ictal_reliability_mask = _phase_minimum_reliability(
            evidence.reliability, ictal_node_mask
        )
        if not torch.equal(ictal_component_mask, ictal_reliability_mask):
            raise RuntimeError("I phase reliability mask drifted")
        ictal_gated, ictal_attenuation = _attenuate_positive_only(
            ictal_raw, ictal_reliability
        )

        prior = self.channel_bias.to(dtype=evidence.evolution.dtype).unsqueeze(0)
        prior = prior.expand(evidence.batch_size, -1)
        logits = prior + evolution_gated.sum(dim=-1) + ictal_gated.sum(dim=-1)
        output = DevelopmentIVReasonerOutput(
            event_logits=logits,
            channel_prior=prior,
            evolution_tile_score=evolution_tile,
            evolution_phase_component=evolution_component,
            evolution_phase_mask=evolution_component_mask,
            evolution_raw_phase_contribution=evolution_raw,
            evolution_gated_phase_contribution=evolution_gated,
            evolution_quality_attenuation=evolution_attenuation,
            evolution_phase_reliability=evolution_reliability,
            ictal_edge_tile_support=ictal_edge_support,
            ictal_node_tile_support=ictal_node_support,
            ictal_node_valid_degree=ictal_degree,
            ictal_phase_component=ictal_component,
            ictal_phase_mask=ictal_component_mask,
            ictal_raw_phase_contribution=ictal_raw,
            ictal_gated_phase_contribution=ictal_gated,
            ictal_quality_attenuation=ictal_attenuation,
            ictal_phase_reliability=ictal_reliability,
            event_abstain_recommended=evidence.event_abstain,
        )
        if not torch.allclose(
            output.reconstructed_logits(), output.event_logits, atol=1e-6, rtol=1e-6
        ):
            raise RuntimeError("Development reasoner explanation does not reconstruct logits")
        return output


@dataclass(frozen=True)
class _DevelopmentSplitEvidence:
    model_split: str
    event_ids: tuple[str, ...]
    patient_ids_by_event: tuple[str, ...]
    oof_folds: tuple[int | None, ...]
    evidence: DevelopmentIVEvidenceBatch

    def __post_init__(self) -> None:
        if self.model_split not in _ALLOWED_SPLITS:
            raise ValueError("Development evidence rejects source_eval/private")
        count = self.evidence.batch_size
        if not (
            len(self.event_ids)
            == len(self.patient_ids_by_event)
            == len(self.oof_folds)
            == count
        ):
            raise ValueError("Development event identity and tensors disagree")
        if len(set(self.event_ids)) != count:
            raise ValueError("Development event IDs must be unique")
        normalized = tuple(normalize_patient_id(value) for value in self.patient_ids_by_event)
        object.__setattr__(self, "patient_ids_by_event", normalized)
        if self.model_split == "source_train":
            if any(fold not in range(5) for fold in self.oof_folds):
                raise ValueError("Every source-train event requires its patient OOF fold")
            folds_by_patient: dict[str, set[int | None]] = {}
            for patient_id, fold in zip(self.patient_ids_by_event, self.oof_folds):
                folds_by_patient.setdefault(patient_id, set()).add(fold)
            if any(len(folds) != 1 for folds in folds_by_patient.values()):
                raise ValueError("One source-train patient cannot cross OOF folds")
        elif any(fold is not None for fold in self.oof_folds):
            raise ValueError("Source-dev must use final producers, never OOF fold labels")

    @property
    def patient_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.patient_ids_by_event)))

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(
            {
                "model_split": self.model_split,
                "event_ids": self.event_ids,
                "patient_ids_by_event": self.patient_ids_by_event,
                "oof_folds": self.oof_folds,
                "evidence_sha256": _evidence_batch_sha256(self.evidence),
            }
        )


@dataclass(frozen=True)
class DevelopmentIVEvidenceAuthorizationReceipt:
    policy_sha256: str
    ictal_artifact_sha256: str
    ictal_receipt_sha256: str
    source_train_vaq_manifest_sha256: str
    source_dev_vaq_manifest_sha256: str
    verified_target_v2_artifact_sha256: str
    verified_target_v2_receipt_sha256: str
    verified_target_v2_policy_sha256: str
    source_train_evidence_receipt_sha256: str
    source_dev_evidence_receipt_sha256: str
    source_train_event_roster_sha256: str
    source_dev_event_roster_sha256: str
    source_train_patient_oof_assignment_sha256: str
    source_train_patient_ids: tuple[str, ...]
    source_dev_patient_ids: tuple[str, ...]
    upstream_ictal_reasoner_authorized: bool = False
    candidate_reasoner_input_authorized: bool = True
    formal_reasoner_authorized: bool = False
    formal_promotion: bool = False
    source_eval_used: bool = False
    private_used: bool = False
    morphology_present: bool = False
    schema_version: str = DEVELOPMENT_IV_AUTHORIZATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "policy_sha256",
            "ictal_artifact_sha256",
            "ictal_receipt_sha256",
            "source_train_vaq_manifest_sha256",
            "source_dev_vaq_manifest_sha256",
            "verified_target_v2_artifact_sha256",
            "verified_target_v2_receipt_sha256",
            "verified_target_v2_policy_sha256",
            "source_train_evidence_receipt_sha256",
            "source_dev_evidence_receipt_sha256",
            "source_train_event_roster_sha256",
            "source_dev_event_roster_sha256",
            "source_train_patient_oof_assignment_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), field_name=name)
            )
        if self.policy_sha256 != DEVELOPMENT_IV_AUTHORIZATION_POLICY_SHA256:
            raise ValueError("Development I+V authorization policy digest drifted")
        if self.verified_target_v2_policy_sha256 != TARGET_V2_POLICY_SHA256:
            raise ValueError("Candidate capability is not bound to target-v2 policy")
        if any(
            (
                self.upstream_ictal_reasoner_authorized,
                self.formal_reasoner_authorized,
                self.formal_promotion,
                self.source_eval_used,
                self.private_used,
                self.morphology_present,
            )
        ) or not self.candidate_reasoner_input_authorized:
            raise ValueError("Development-only authorization boundary changed")
        for name in ("source_train_patient_ids", "source_dev_patient_ids"):
            roster = tuple(sorted(normalize_patient_id(value) for value in getattr(self, name)))
            if not roster or len(roster) != len(set(roster)):
                raise ValueError(f"{name} must be non-empty, sorted, and unique")
            object.__setattr__(self, name, roster)
        if set(self.source_train_patient_ids) & set(self.source_dev_patient_ids):
            raise ValueError("Candidate train/dev patient rosters overlap")
        if self.schema_version != DEVELOPMENT_IV_AUTHORIZATION_SCHEMA:
            raise ValueError("Unsupported development authorization schema")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True, init=False)
class VerifiedDevelopmentIVEvidenceCapability:
    """Opaque target-free candidate capability; never formal authority."""

    source_train: _DevelopmentSplitEvidence = field(repr=False)
    source_dev: _DevelopmentSplitEvidence = field(repr=False)
    receipt: DevelopmentIVEvidenceAuthorizationReceipt

    def __init__(
        self,
        *,
        _verification_marker: object,
        source_train: _DevelopmentSplitEvidence,
        source_dev: _DevelopmentSplitEvidence,
        receipt: DevelopmentIVEvidenceAuthorizationReceipt,
    ) -> None:
        if _verification_marker is not _CAPABILITY_MARKER:
            raise TypeError("Development I+V capability requires the closed issuer")
        if source_train.model_split != "source_train" or source_dev.model_split != "source_dev":
            raise ValueError("Development capability requires source_train and source_dev")
        object.__setattr__(self, "source_train", source_train)
        object.__setattr__(self, "source_dev", source_dev)
        object.__setattr__(self, "receipt", receipt)
        self.assert_unchanged()

    def assert_unchanged(self) -> None:
        if self.source_train.receipt_sha256 != self.receipt.source_train_evidence_receipt_sha256:
            raise ValueError("Source-train development evidence changed after authorization")
        if self.source_dev.receipt_sha256 != self.receipt.source_dev_evidence_receipt_sha256:
            raise ValueError("Source-dev development evidence changed after authorization")
        if self.source_train.patient_ids != self.receipt.source_train_patient_ids:
            raise ValueError("Source-train patient roster changed after authorization")
        if self.source_dev.patient_ids != self.receipt.source_dev_patient_ids:
            raise ValueError("Source-dev patient roster changed after authorization")
        if _canonical_sha256(self.source_train.event_ids) != (
            self.receipt.source_train_event_roster_sha256
        ):
            raise ValueError("Source-train event roster changed after authorization")
        if _canonical_sha256(self.source_dev.event_ids) != (
            self.receipt.source_dev_event_roster_sha256
        ):
            raise ValueError("Source-dev event roster changed after authorization")
        assignment = tuple(
            sorted(
                {
                    (patient_id, fold)
                    for patient_id, fold in zip(
                        self.source_train.patient_ids_by_event,
                        self.source_train.oof_folds,
                    )
                }
            )
        )
        if _canonical_sha256(assignment) != (
            self.receipt.source_train_patient_oof_assignment_sha256
        ):
            raise ValueError("Source-train patient-OOF assignment changed")


def _capability_tensors(
    capability: VerifiedDevelopmentIVEvidenceCapability,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for split_name, split in (
        ("source_train", capability.source_train),
        ("source_dev", capability.source_dev),
    ):
        for field_name in (
            "evolution",
            "ictal",
            "evolution_mask",
            "ictal_mask",
            "phase_mask",
            "reliability",
            "event_abstain",
        ):
            result[f"{split_name}_{field_name}"] = (
                getattr(split.evidence, field_name).detach().cpu().contiguous()
            )
    return result


def _capability_tensor_specs(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "tensor_sha256": _tensor_sha256(name, value),
        }
        for name, value in sorted(tensors.items())
    }


def _capability_event_payload(
    capability: VerifiedDevelopmentIVEvidenceCapability,
) -> dict[str, object]:
    return {
        "schema_version": "soz_development_iv_candidate_event_roster_v1",
        "source_train": [
            {
                "event_id": event_id,
                "patient_id": patient_id,
                "oof_fold": fold,
            }
            for event_id, patient_id, fold in zip(
                capability.source_train.event_ids,
                capability.source_train.patient_ids_by_event,
                capability.source_train.oof_folds,
            )
        ],
        "source_dev": [
            {
                "event_id": event_id,
                "patient_id": patient_id,
                "oof_fold": fold,
            }
            for event_id, patient_id, fold in zip(
                capability.source_dev.event_ids,
                capability.source_dev.patient_ids_by_event,
                capability.source_dev.oof_folds,
            )
        ],
    }


def _capability_manifest_payload(
    capability: VerifiedDevelopmentIVEvidenceCapability,
    *,
    tensor_specs: Mapping[str, object],
    files: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": DEVELOPMENT_IV_CAPABILITY_SCHEMA,
        "purpose": "development_iv_candidate_reasoner_evidence_only",
        "serialization": "canonical_json_plus_safetensors_no_pickle",
        "authorization_policy": dict(_DEVELOPMENT_IV_AUTHORIZATION_POLICY),
        "authorization_policy_sha256": DEVELOPMENT_IV_AUTHORIZATION_POLICY_SHA256,
        "authorization_receipt": asdict(capability.receipt),
        "authorization_receipt_sha256": capability.receipt.receipt_sha256,
        "active_evidence_families": list(ACTIVE_EVIDENCE_FAMILIES),
        "absent_evidence_families": list(ABSENT_EVIDENCE_FAMILIES),
        "development_only": True,
        "candidate_reasoner_input_authorized": True,
        "upstream_ictal_reasoner_authorized": False,
        "formal_reasoner_authorized": False,
        "formal_promotion": False,
        "source_eval_used": False,
        "private_used": False,
        "target_values_loaded": False,
        "raw_eeg_present": False,
        "foundation_latent_present": False,
        "path_or_report_present": False,
        "quality_diagnostics_or_burden_present": False,
        "source_train_event_count": capability.source_train.evidence.batch_size,
        "source_dev_event_count": capability.source_dev.evidence.batch_size,
        "source_train_patient_count": len(capability.source_train.patient_ids),
        "source_dev_patient_count": len(capability.source_dev.patient_ids),
        "tensor_specs": dict(tensor_specs),
        "files": dict(files),
    }


@dataclass(frozen=True)
class PublishedDevelopmentIVEvidenceCapability:
    path: Path
    manifest_sha256: str
    authorization_receipt_sha256: str
    capability: VerifiedDevelopmentIVEvidenceCapability = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "manifest_sha256",
            _require_sha256(self.manifest_sha256, field_name="manifest_sha256"),
        )
        object.__setattr__(
            self,
            "authorization_receipt_sha256",
            _require_sha256(
                self.authorization_receipt_sha256,
                field_name="authorization_receipt_sha256",
            ),
        )
        if self.authorization_receipt_sha256 != self.capability.receipt.receipt_sha256:
            raise ValueError("Published capability receipt does not match capability")


def publish_development_iv_evidence_capability(
    capability: VerifiedDevelopmentIVEvidenceCapability,
    output_directory: str | Path,
) -> PublishedDevelopmentIVEvidenceCapability:
    """Atomically persist a target-free candidate capability."""

    if type(capability) is not VerifiedDevelopmentIVEvidenceCapability:
        raise TypeError("Only the closed candidate issuer may publish a capability")
    capability.assert_unchanged()
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for capability publication") from exc
    target = _safe_new_directory(
        output_directory, field_name="Development I+V capability output"
    )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    published = False
    try:
        tensors = _capability_tensors(capability)
        tensor_path = temporary / DEVELOPMENT_IV_CAPABILITY_TENSORS_FILENAME
        save_file(tensors, str(tensor_path))
        events_path = temporary / DEVELOPMENT_IV_CAPABILITY_EVENTS_FILENAME
        events_raw = _canonical_json_bytes(_capability_event_payload(capability))
        events_path.write_bytes(events_raw)
        tensor_size = tensor_path.stat().st_size
        if not 1 <= tensor_size <= _MAX_CAPABILITY_TENSOR_BYTES:
            raise ValueError("Candidate capability tensor artifact has invalid size")
        if not 1 <= len(events_raw) <= _MAX_CAPABILITY_JSON_BYTES:
            raise ValueError("Candidate capability event artifact has invalid size")
        files = {
            DEVELOPMENT_IV_CAPABILITY_TENSORS_FILENAME: {
                "sha256": _file_sha256(tensor_path),
                "size_bytes": tensor_size,
            },
            DEVELOPMENT_IV_CAPABILITY_EVENTS_FILENAME: {
                "sha256": hashlib.sha256(events_raw).hexdigest(),
                "size_bytes": len(events_raw),
            },
        }
        manifest = _capability_manifest_payload(
            capability,
            tensor_specs=_capability_tensor_specs(tensors),
            files=files,
        )
        manifest_raw = _canonical_json_bytes(manifest)
        if not 1 <= len(manifest_raw) <= _MAX_CAPABILITY_JSON_BYTES:
            raise ValueError("Candidate capability manifest has invalid size")
        manifest_path = temporary / DEVELOPMENT_IV_CAPABILITY_MANIFEST_FILENAME
        manifest_path.write_bytes(manifest_raw)
        for path in (tensor_path, events_path, manifest_path):
            _fsync_file(path)
        _fsync_directory(temporary)
        os.rename(temporary, target)
        published = True
        _fsync_directory(target.parent)
        return load_development_iv_evidence_capability(
            target,
            expected_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _read_capability_file(
    path: Path,
    *,
    maximum_bytes: int,
    field_name: str,
) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field_name} must be a regular non-symlinked file")
    size = path.stat().st_size
    if not 1 <= size <= maximum_bytes:
        raise ValueError(f"{field_name} has an invalid size")
    return path.read_bytes()


def load_development_iv_evidence_capability(
    bundle_directory: str | Path,
    *,
    expected_manifest_sha256: str,
) -> PublishedDevelopmentIVEvidenceCapability:
    """Strict closed loader for a published target-free candidate capability."""

    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for capability loading") from exc
    bundle = Path(os.path.abspath(bundle_directory))
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("Candidate capability must be a regular directory")
    if {path.name for path in bundle.iterdir()} != _CAPABILITY_FILE_SET:
        raise ValueError("Candidate capability violates its closed file schema")
    manifest_raw = _read_capability_file(
        bundle / DEVELOPMENT_IV_CAPABILITY_MANIFEST_FILENAME,
        maximum_bytes=_MAX_CAPABILITY_JSON_BYTES,
        field_name="Candidate capability manifest",
    )
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if manifest_sha != _require_sha256(
        expected_manifest_sha256, field_name="expected_manifest_sha256"
    ):
        raise ValueError("Candidate capability manifest SHA mismatch")
    manifest = _strict_json(manifest_raw, field_name="Candidate capability manifest")
    if not isinstance(manifest, dict) or _canonical_json_bytes(manifest) != manifest_raw:
        raise ValueError("Candidate capability manifest is not canonical JSON")
    fixed = {
        "schema_version": DEVELOPMENT_IV_CAPABILITY_SCHEMA,
        "purpose": "development_iv_candidate_reasoner_evidence_only",
        "serialization": "canonical_json_plus_safetensors_no_pickle",
        "authorization_policy": _DEVELOPMENT_IV_AUTHORIZATION_POLICY,
        "authorization_policy_sha256": DEVELOPMENT_IV_AUTHORIZATION_POLICY_SHA256,
        "active_evidence_families": list(ACTIVE_EVIDENCE_FAMILIES),
        "absent_evidence_families": list(ABSENT_EVIDENCE_FAMILIES),
        "development_only": True,
        "candidate_reasoner_input_authorized": True,
        "upstream_ictal_reasoner_authorized": False,
        "formal_reasoner_authorized": False,
        "formal_promotion": False,
        "source_eval_used": False,
        "private_used": False,
        "target_values_loaded": False,
        "raw_eeg_present": False,
        "foundation_latent_present": False,
        "path_or_report_present": False,
        "quality_diagnostics_or_burden_present": False,
    }
    if any(manifest.get(name) != value for name, value in fixed.items()):
        raise ValueError("Candidate capability scientific boundary changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {
        DEVELOPMENT_IV_CAPABILITY_TENSORS_FILENAME,
        DEVELOPMENT_IV_CAPABILITY_EVENTS_FILENAME,
    }:
        raise ValueError("Candidate capability file receipt changed")
    for name, maximum in (
        (DEVELOPMENT_IV_CAPABILITY_TENSORS_FILENAME, _MAX_CAPABILITY_TENSOR_BYTES),
        (DEVELOPMENT_IV_CAPABILITY_EVENTS_FILENAME, _MAX_CAPABILITY_JSON_BYTES),
    ):
        record = files[name]
        if not isinstance(record, dict) or set(record) != {"sha256", "size_bytes"}:
            raise ValueError("Candidate capability file receipt schema changed")
        raw = _read_capability_file(
            bundle / name, maximum_bytes=maximum, field_name=name
        )
        if len(raw) != record["size_bytes"] or hashlib.sha256(raw).hexdigest() != (
            record["sha256"]
        ):
            raise ValueError(f"Candidate capability payload changed: {name}")
    events_raw = (bundle / DEVELOPMENT_IV_CAPABILITY_EVENTS_FILENAME).read_bytes()
    events = _strict_json(events_raw, field_name="Candidate capability events")
    if not isinstance(events, dict) or _canonical_json_bytes(events) != events_raw:
        raise ValueError("Candidate capability events are not canonical JSON")
    if set(events) != {"schema_version", "source_train", "source_dev"} or events[
        "schema_version"
    ] != "soz_development_iv_candidate_event_roster_v1":
        raise ValueError("Candidate capability event schema changed")
    tensors = load_file(
        str(bundle / DEVELOPMENT_IV_CAPABILITY_TENSORS_FILENAME), device="cpu"
    )
    expected_names = {
        f"{split}_{name}"
        for split in ("source_train", "source_dev")
        for name in (
            "evolution",
            "ictal",
            "evolution_mask",
            "ictal_mask",
            "phase_mask",
            "reliability",
            "event_abstain",
        )
    }
    if set(tensors) != expected_names or _capability_tensor_specs(tensors) != manifest.get(
        "tensor_specs"
    ):
        raise ValueError("Candidate capability tensor schema or content changed")

    def split_from_payload(model_split: str) -> _DevelopmentSplitEvidence:
        rows = events[model_split]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Candidate capability {model_split} roster is invalid")
        expected_fields = {"event_id", "patient_id", "oof_fold"}
        if any(not isinstance(row, dict) or set(row) != expected_fields for row in rows):
            raise ValueError(f"Candidate capability {model_split} row schema changed")
        prefix = model_split
        evidence = DevelopmentIVEvidenceBatch(
            evolution=tensors[f"{prefix}_evolution"],
            ictal=tensors[f"{prefix}_ictal"],
            evolution_mask=tensors[f"{prefix}_evolution_mask"],
            ictal_mask=tensors[f"{prefix}_ictal_mask"],
            phase_mask=tensors[f"{prefix}_phase_mask"],
            reliability=tensors[f"{prefix}_reliability"],
            event_abstain=tensors[f"{prefix}_event_abstain"],
        )
        return _DevelopmentSplitEvidence(
            model_split=model_split,
            event_ids=tuple(str(row["event_id"]) for row in rows),
            patient_ids_by_event=tuple(str(row["patient_id"]) for row in rows),
            oof_folds=tuple(row["oof_fold"] for row in rows),
            evidence=evidence,
        )

    receipt_payload = manifest.get("authorization_receipt")
    if not isinstance(receipt_payload, dict):
        raise ValueError("Candidate capability authorization receipt is missing")
    try:
        receipt = DevelopmentIVEvidenceAuthorizationReceipt(**receipt_payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("Candidate authorization receipt is invalid") from exc
    if receipt.receipt_sha256 != manifest.get("authorization_receipt_sha256"):
        raise ValueError("Candidate authorization receipt SHA mismatch")
    capability = VerifiedDevelopmentIVEvidenceCapability(
        _verification_marker=_CAPABILITY_MARKER,
        source_train=split_from_payload("source_train"),
        source_dev=split_from_payload("source_dev"),
        receipt=receipt,
    )
    expected_manifest = _capability_manifest_payload(
        capability,
        tensor_specs=_capability_tensor_specs(tensors),
        files=files,
    )
    if _canonical_json_bytes(manifest) != _canonical_json_bytes(expected_manifest):
        raise ValueError("Candidate capability manifest does not match reconstructed evidence")
    return PublishedDevelopmentIVEvidenceCapability(
        path=bundle,
        manifest_sha256=manifest_sha,
        authorization_receipt_sha256=receipt.receipt_sha256,
        capability=capability,
    )


def _assert_ictal_v12_unchanged(
    artifact: VerifiedLaBraMK31DevelopmentScoreArtifactV12,
) -> Mapping[str, object]:
    if type(artifact) is not VerifiedLaBraMK31DevelopmentScoreArtifactV12:
        raise TypeError("I input must come from the strict v1.2 score loader")
    manifest = artifact.manifest
    if hashlib.sha256(_ictal_manifest_canonical_bytes(manifest)).hexdigest() != (
        artifact.artifact_sha256
    ):
        raise ValueError("v1.2 I manifest changed after strict replay")
    boundary = {
        "schema_version": LABRAM_K31_DEVELOPMENT_SCORE_SCHEMA_V1_2,
        "purpose": LABRAM_K31_DEVELOPMENT_SCORE_PURPOSE_V1_2,
        "development_only": True,
        "formal_promotion": False,
        "authorized_for_formal_evidence_or_reasoner": False,
        "reasoner_authorized": False,
        "target_vectors_loaded": False,
        "target_values_present": False,
        "source_annotation_targets_present": False,
        "source_annotation_coverage_present": False,
        "private_data_used": False,
        "source_eval_signals_or_events_used": False,
        "deepsoz_target_source_loaded": False,
        "deepsoz_target_values_reachable": False,
    }
    if any(manifest.get(name) != value for name, value in boundary.items()):
        raise ValueError("Upstream v1.2 I boundary changed; candidate issue denied")
    values = {
        "source_train_oof_scores": artifact.source_train_scores,
        "source_train_deployment_mask": artifact.source_train_deployment_mask,
        "source_train_ictal_phase_mask": artifact.source_train_phase_mask,
        "source_dev_final_scores": artifact.source_dev_scores,
        "source_dev_deployment_mask": artifact.source_dev_deployment_mask,
        "source_dev_ictal_phase_mask": artifact.source_dev_phase_mask,
    }
    records = manifest.get("tensor_files")
    if not isinstance(records, Mapping) or set(records) != set(values):
        raise ValueError("v1.2 I tensor record schema changed")
    for name, tensor in values.items():
        record = records[name]
        if not isinstance(record, Mapping) or record.get("tensor_sha256") != _ictal_tensor_sha256(
            name, tensor
        ):
            raise ValueError(f"v1.2 I tensor changed after strict replay: {name}")
    return manifest


def _validated_i_rows(
    manifest: Mapping[str, object], model_split: str
) -> tuple[Mapping[str, object], ...]:
    key = "source_train_event_rows" if model_split == "source_train" else "source_dev_event_rows"
    raw = manifest.get(key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"v1.2 I {key} is missing")
    expected_fields = {
        "event_id",
        "token_event_id",
        "target_patient_id",
        "public_patient_id",
        "oof_fold",
        "producer_selection",
    }
    rows: list[Mapping[str, object]] = []
    for row in raw:
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ValueError(f"v1.2 I {key} row schema changed")
        if model_split == "source_train":
            fold = row["oof_fold"]
            if fold not in range(5) or row["producer_selection"] != f"fold{fold}":
                raise ValueError("Source-train I row is not patient-OOF")
        elif row["oof_fold"] is not None or row["producer_selection"] != "final":
            raise ValueError("Source-dev I row does not use the final producer")
        rows.append(row)
    event_ids = tuple(str(row["event_id"]) for row in rows)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("v1.2 I event roster contains duplicates")
    return tuple(rows)


def _reorder_vaq_to_i(
    i_rows: Sequence[Mapping[str, object]],
    vaq_rows: Sequence[Mapping[str, object]],
) -> torch.Tensor:
    if len({str(row.get("event_id")) for row in vaq_rows}) != len(vaq_rows):
        raise ValueError("V+A/Q event roster contains duplicate event IDs")
    index = {str(row["event_id"]): position for position, row in enumerate(vaq_rows)}
    i_ids = tuple(str(row["event_id"]) for row in i_rows)
    if set(index) != set(i_ids) or len(index) != len(i_ids):
        raise ValueError("I and V+A/Q event rosters are not identical")
    order = torch.tensor([index[event_id] for event_id in i_ids], dtype=torch.long)
    for i_row, position in zip(i_rows, order.tolist()):
        v_row = vaq_rows[position]
        if normalize_patient_id(v_row.get("patient_id")) != normalize_patient_id(
            i_row["target_patient_id"]
        ):
            raise ValueError("I and V+A/Q event patient identities disagree")
    return order


def _build_split_evidence(
    *,
    model_split: str,
    i_rows: tuple[Mapping[str, object], ...],
    i_scores: torch.Tensor,
    i_mask: torch.Tensor,
    i_phase: torch.Tensor,
    vaq_events: Mapping[str, object],
    vaq_tensors: Mapping[str, torch.Tensor],
) -> _DevelopmentSplitEvidence:
    if model_split not in _ALLOWED_SPLITS:
        raise ValueError("Candidate reasoner rejects source_eval/private")
    rows = vaq_events.get("events")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("V+A/Q event payload is invalid")
    expected_split = SOURCE_TRAIN_SPLIT if model_split == "source_train" else DEVELOPMENT_VAQ_SPLIT
    if any(row.get("model_split") != expected_split for row in rows):
        raise ValueError("V+A/Q events escaped the requested development split")
    order = _reorder_vaq_to_i(i_rows, rows)
    if i_scores.shape[0] != len(i_rows) or i_mask.shape != i_scores.shape:
        raise ValueError("I score tensor does not align with its event roster")
    ictal, pooled_mask = pool_ictal_seconds_to_tiles(i_scores, i_mask)
    reordered = {
        name: value.index_select(0, order)
        if value.ndim > 0 and value.shape[0] == len(rows)
        else value
        for name, value in vaq_tensors.items()
    }
    phase = reordered["ictal_phase_mask"]
    if not torch.equal(i_phase.to(torch.bool), phase):
        raise ValueError("I and V+A/Q phase masks differ after event alignment")
    evolution = reordered["evolution_scaled"].to(torch.float32).contiguous()
    evolution_mask = reordered["evolution_mask"].to(torch.bool).contiguous()
    reliability = reordered["reliability"].to(torch.float32).contiguous()
    event_abstain = reordered["event_abstain"].to(torch.bool).contiguous()
    physical_edge = physical_node_to_edge_mask(evolution_mask)
    evidence = DevelopmentIVEvidenceBatch(
        evolution=evolution.detach(),
        ictal=ictal.to(torch.float32).detach(),
        evolution_mask=evolution_mask,
        ictal_mask=(pooled_mask & physical_edge).contiguous(),
        phase_mask=phase.to(torch.bool).contiguous(),
        reliability=reliability.detach(),
        event_abstain=event_abstain,
    )
    folds = tuple(
        None if row["oof_fold"] is None else int(row["oof_fold"])
        for row in i_rows
    )
    return _DevelopmentSplitEvidence(
        model_split=model_split,
        event_ids=tuple(str(row["event_id"]) for row in i_rows),
        patient_ids_by_event=tuple(
            normalize_patient_id(row["target_patient_id"]) for row in i_rows
        ),
        oof_folds=folds,
        evidence=evidence,
    )


def issue_development_iv_evidence_capability(
    *,
    ictal_artifact: VerifiedLaBraMK31DevelopmentScoreArtifactV12,
    source_train_vaq_bundle: str | Path,
    expected_source_train_vaq_manifest_sha256: str,
    source_dev_vaq_bundle: str | Path,
    expected_source_dev_vaq_manifest_sha256: str,
) -> VerifiedDevelopmentIVEvidenceCapability:
    """Issue a target-free candidate capability without altering formal status."""

    i_manifest = _assert_ictal_v12_unchanged(ictal_artifact)
    train_manifest, train_events, train_tensors = load_source_train_oof_vaq_evidence(
        source_train_vaq_bundle,
        expected_manifest_sha256=expected_source_train_vaq_manifest_sha256,
    )
    dev_manifest, dev_events, dev_tensors = load_development_vaq_evidence(
        source_dev_vaq_bundle,
        expected_manifest_sha256=expected_source_dev_vaq_manifest_sha256,
    )
    if train_manifest.get("purpose") != SOURCE_TRAIN_OOF_VAQ_PURPOSE or not train_manifest.get(
        "candidate_reasoner_input_authorized"
    ):
        raise ValueError("Source-train V+A/Q is not candidate-reasoner evidence")
    if train_manifest.get("formal_promotion_authorized") is not False:
        raise ValueError("Source-train V+A/Q formal boundary changed")
    if dev_manifest.get("purpose") != DEVELOPMENT_VAQ_PURPOSE or dev_manifest.get(
        "training_authorized"
    ) is not False:
        raise ValueError("Source-dev V+A/Q may be used only for development evaluation")

    timeline = i_manifest.get("signal_timeline_lineage")
    if not isinstance(timeline, Mapping):
        raise ValueError("v1.2 I signal timeline lineage is missing")
    shared_checks = {
        "train signal artifact": train_manifest.get("signal_preflight_artifact_sha256")
        == timeline.get("signal_preflight_artifact_sha256"),
        "dev signal artifact": dev_manifest.get("signal_preflight_artifact_sha256")
        == timeline.get("signal_preflight_artifact_sha256"),
        "train signal receipt": train_manifest.get("signal_preflight_receipt_sha256")
        == timeline.get("signal_preflight_receipt_sha256"),
        "dev signal receipt": dev_manifest.get("signal_preflight_receipt_sha256")
        == timeline.get("signal_preflight_receipt_sha256"),
        "train split": train_manifest.get("split_manifest_sha256")
        == timeline.get("split_manifest_sha256"),
        "dev split": dev_manifest.get("split_manifest_sha256")
        == timeline.get("split_manifest_sha256"),
    }
    failed = tuple(name for name, passed in shared_checks.items() if not passed)
    if failed:
        raise ValueError(f"I and V+A/Q lineage disagree: {failed}")

    train_rows = _validated_i_rows(i_manifest, "source_train")
    dev_rows = _validated_i_rows(i_manifest, "source_dev")
    source_train = _build_split_evidence(
        model_split="source_train",
        i_rows=train_rows,
        i_scores=ictal_artifact.source_train_scores,
        i_mask=ictal_artifact.source_train_deployment_mask,
        i_phase=ictal_artifact.source_train_phase_mask,
        vaq_events=train_events,
        vaq_tensors=train_tensors,
    )
    source_dev = _build_split_evidence(
        model_split="source_dev",
        i_rows=dev_rows,
        i_scores=ictal_artifact.source_dev_scores,
        i_mask=ictal_artifact.source_dev_deployment_mask,
        i_phase=ictal_artifact.source_dev_phase_mask,
        vaq_events=dev_events,
        vaq_tensors=dev_tensors,
    )
    receipt = DevelopmentIVEvidenceAuthorizationReceipt(
        policy_sha256=DEVELOPMENT_IV_AUTHORIZATION_POLICY_SHA256,
        ictal_artifact_sha256=ictal_artifact.artifact_sha256,
        ictal_receipt_sha256=ictal_artifact.receipt_sha256,
        source_train_vaq_manifest_sha256=_require_sha256(
            expected_source_train_vaq_manifest_sha256,
            field_name="expected_source_train_vaq_manifest_sha256",
        ),
        source_dev_vaq_manifest_sha256=_require_sha256(
            expected_source_dev_vaq_manifest_sha256,
            field_name="expected_source_dev_vaq_manifest_sha256",
        ),
        verified_target_v2_artifact_sha256=_require_sha256(
            timeline.get("verified_target_v2_artifact_sha256"),
            field_name="verified_target_v2_artifact_sha256",
        ),
        verified_target_v2_receipt_sha256=_require_sha256(
            timeline.get("verified_target_v2_receipt_sha256"),
            field_name="verified_target_v2_receipt_sha256",
        ),
        verified_target_v2_policy_sha256=_require_sha256(
            timeline.get("verified_target_v2_policy_sha256"),
            field_name="verified_target_v2_policy_sha256",
        ),
        source_train_evidence_receipt_sha256=source_train.receipt_sha256,
        source_dev_evidence_receipt_sha256=source_dev.receipt_sha256,
        source_train_event_roster_sha256=_canonical_sha256(
            source_train.event_ids
        ),
        source_dev_event_roster_sha256=_canonical_sha256(source_dev.event_ids),
        source_train_patient_oof_assignment_sha256=_canonical_sha256(
            tuple(
                sorted(
                    {
                        (patient_id, fold)
                        for patient_id, fold in zip(
                            source_train.patient_ids_by_event,
                            source_train.oof_folds,
                        )
                    }
                )
            )
        ),
        source_train_patient_ids=source_train.patient_ids,
        source_dev_patient_ids=source_dev.patient_ids,
    )
    return VerifiedDevelopmentIVEvidenceCapability(
        _verification_marker=_CAPABILITY_MARKER,
        source_train=source_train,
        source_dev=source_dev,
        receipt=receipt,
    )


def _assert_verified_target_unchanged(
    target: VerifiedDeepSOZTargetV2Artifact,
) -> None:
    if type(target) is not VerifiedDeepSOZTargetV2Artifact:
        raise TypeError("Reasoner labels must come from the strict target-v2 loader")
    target.__post_init__()
    frame = _build_target_frame(
        target.registry,
        source_sha256=target.receipt.source_input_sha256,
        split_sha256=target.receipt.split_input_sha256,
    )
    digest = hashlib.sha256(_target_csv_bytes(frame)).hexdigest()
    if digest != target.receipt.target_artifact_sha256:
        raise ValueError("In-memory target-v2 values changed after strict verification")


@dataclass(frozen=True, init=False)
class DevelopmentReasonerPatientBatch:
    evidence: DevelopmentIVEvidenceBatch
    event_patient_index: torch.Tensor
    patient_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    expected_event_counts: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor

    def __init__(
        self,
        *,
        _verification_marker: object,
        evidence: DevelopmentIVEvidenceBatch,
        event_patient_index: torch.Tensor,
        patient_ids: Sequence[object],
        event_ids: Sequence[object],
        expected_event_counts: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> None:
        if _verification_marker is not _PATIENT_BATCH_MARKER:
            raise TypeError("Patient targets require the strict target-v2 join")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "event_patient_index", event_patient_index)
        object.__setattr__(
            self,
            "patient_ids",
            tuple(normalize_patient_id(value) for value in patient_ids),
        )
        object.__setattr__(self, "event_ids", tuple(str(value) for value in event_ids))
        object.__setattr__(self, "expected_event_counts", expected_event_counts)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "target_mask", target_mask)
        self._validate()

    def _validate(self) -> None:
        events = self.evidence.batch_size
        patients = len(self.patient_ids)
        if tuple(self.event_patient_index.shape) != (events,) or self.event_patient_index.dtype != torch.long:
            raise TypeError("event_patient_index must be long [E]")
        if len(self.event_ids) != events or len(set(self.event_ids)) != events:
            raise ValueError("Patient batch event IDs must be unique and aligned")
        if tuple(self.expected_event_counts.shape) != (patients,) or self.expected_event_counts.dtype != torch.long:
            raise TypeError("expected_event_counts must be long [P]")
        if tuple(self.targets.shape) != (patients, N_STANDARD_CHANNELS) or tuple(
            self.target_mask.shape
        ) != (patients, N_STANDARD_CHANNELS):
            raise ValueError("Targets and masks must have shape [P,19]")
        if self.target_mask.dtype != torch.bool:
            raise TypeError("Target mask must be bool")
        devices = {
            self.evidence.evolution.device,
            self.event_patient_index.device,
            self.expected_event_counts.device,
            self.targets.device,
            self.target_mask.device,
        }
        if len(devices) != 1:
            raise ValueError("Patient batch tensors must share one device")
        counts = torch.bincount(self.event_patient_index, minlength=patients)
        if not torch.equal(counts, self.expected_event_counts):
            raise ValueError("Patient batch is not a complete event bag")

    def aggregate(self, event_logits: torch.Tensor) -> PatientAggregation:
        aggregation = aggregate_patient_logits(event_logits, self.event_patient_index)
        expected = torch.arange(
            len(self.patient_ids), dtype=torch.long, device=event_logits.device
        )
        if not torch.equal(aggregation.patient_ids, expected) or not torch.equal(
            aggregation.event_counts, self.expected_event_counts
        ):
            raise RuntimeError("Patient mean lost or reweighted an event")
        return aggregation

    def patient_abstain_recommended(self) -> torch.Tensor:
        result = torch.ones(
            len(self.patient_ids), dtype=torch.bool, device=self.event_patient_index.device
        )
        for index in range(len(self.patient_ids)):
            result[index] = self.evidence.event_abstain[
                self.event_patient_index == index
            ].all()
        return result

    def to(self, device: str | torch.device) -> "DevelopmentReasonerPatientBatch":
        return DevelopmentReasonerPatientBatch(
            _verification_marker=_PATIENT_BATCH_MARKER,
            evidence=self.evidence.to(device),
            event_patient_index=self.event_patient_index.to(device=device),
            patient_ids=self.patient_ids,
            event_ids=self.event_ids,
            expected_event_counts=self.expected_event_counts.to(device=device),
            targets=self.targets.to(device=device),
            target_mask=self.target_mask.to(device=device),
        )


@dataclass(frozen=True, init=False)
class DevelopmentReasonerDataset(Sequence[DevelopmentReasonerPatientBatch]):
    """Complete split dataset issued only after the target-v2 join stage."""

    model_split: str
    patient_ids: tuple[str, ...]
    _full_batch: DevelopmentReasonerPatientBatch = field(repr=False)
    evidence_authorization_sha256: str
    verified_target_v2_receipt_sha256: str
    receipt_sha256: str

    @staticmethod
    def _receipt_payload(
        *,
        model_split: str,
        full_batch: DevelopmentReasonerPatientBatch,
        evidence_authorization_sha256: str,
        verified_target_v2_receipt_sha256: str,
    ) -> dict[str, object]:
        return {
            "schema_version": DEVELOPMENT_IV_DATASET_SCHEMA,
            "model_split": model_split,
            "patient_ids": full_batch.patient_ids,
            "event_ids": full_batch.event_ids,
            "evidence_sha256": _evidence_batch_sha256(full_batch.evidence),
            "targets_sha256": _tensor_sha256("targets", full_batch.targets),
            "target_mask_sha256": _tensor_sha256(
                "target_mask", full_batch.target_mask
            ),
            "evidence_authorization_sha256": evidence_authorization_sha256,
            "verified_target_v2_receipt_sha256": verified_target_v2_receipt_sha256,
            "formal_promotion": False,
            "source_eval_used": False,
            "private_used": False,
        }

    def __init__(
        self,
        *,
        _verification_marker: object,
        model_split: str,
        full_batch: DevelopmentReasonerPatientBatch,
        evidence_authorization_sha256: str,
        verified_target_v2_receipt_sha256: str,
    ) -> None:
        if _verification_marker is not _DATASET_MARKER:
            raise TypeError("Development reasoner dataset requires the strict target join")
        if model_split not in _ALLOWED_SPLITS:
            raise ValueError("Development reasoner dataset rejects source_eval/private")
        roster = tuple(full_batch.patient_ids)
        payload = self._receipt_payload(
            model_split=model_split,
            full_batch=full_batch,
            evidence_authorization_sha256=evidence_authorization_sha256,
            verified_target_v2_receipt_sha256=verified_target_v2_receipt_sha256,
        )
        object.__setattr__(self, "model_split", model_split)
        object.__setattr__(self, "patient_ids", roster)
        object.__setattr__(self, "_full_batch", full_batch)
        object.__setattr__(
            self,
            "evidence_authorization_sha256",
            _require_sha256(
                evidence_authorization_sha256,
                field_name="evidence_authorization_sha256",
            ),
        )
        object.__setattr__(
            self,
            "verified_target_v2_receipt_sha256",
            _require_sha256(
                verified_target_v2_receipt_sha256,
                field_name="verified_target_v2_receipt_sha256",
            ),
        )
        object.__setattr__(self, "receipt_sha256", _canonical_sha256(payload))

    def assert_unchanged(self) -> None:
        payload = self._receipt_payload(
            model_split=self.model_split,
            full_batch=self._full_batch,
            evidence_authorization_sha256=self.evidence_authorization_sha256,
            verified_target_v2_receipt_sha256=self.verified_target_v2_receipt_sha256,
        )
        if _canonical_sha256(payload) != self.receipt_sha256:
            raise ValueError("Development reasoner dataset changed after target join")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def full_batch(self) -> DevelopmentReasonerPatientBatch:
        return self._full_batch

    def __getitem__(self, index: int) -> DevelopmentReasonerPatientBatch:
        patient_id = self.patient_ids[index]
        mask = self._full_batch.event_patient_index == index
        indices = mask.nonzero(as_tuple=False).flatten()
        return DevelopmentReasonerPatientBatch(
            _verification_marker=_PATIENT_BATCH_MARKER,
            evidence=self._full_batch.evidence.index_select(indices),
            event_patient_index=torch.zeros(
                len(indices), dtype=torch.long, device=indices.device
            ),
            patient_ids=(patient_id,),
            event_ids=tuple(
                event_id
                for event_id, include in zip(
                    self._full_batch.event_ids, mask.detach().cpu().tolist()
                )
                if include
            ),
            expected_event_counts=torch.tensor(
                [len(indices)], dtype=torch.long, device=indices.device
            ),
            targets=self._full_batch.targets[index : index + 1],
            target_mask=self._full_batch.target_mask[index : index + 1],
        )

    def iter_epoch(
        self, patient_order: Sequence[object] | None = None
    ) -> Iterator[DevelopmentReasonerPatientBatch]:
        order = self.patient_ids if patient_order is None else tuple(
            normalize_patient_id(value) for value in patient_order
        )
        if len(order) != len(self.patient_ids) or set(order) != set(self.patient_ids):
            raise ValueError("Epoch order must contain every patient exactly once")
        index = {patient_id: position for position, patient_id in enumerate(self.patient_ids)}
        for patient_id in order:
            yield self[index[patient_id]]


@dataclass(frozen=True, init=False)
class VerifiedDevelopmentReasonerDataBundle:
    source_train: DevelopmentReasonerDataset
    source_dev: DevelopmentReasonerDataset
    evidence_authorization_sha256: str
    verified_target_v2_receipt_sha256: str
    formal_promotion: bool
    formal_reasoner_authorized: bool
    source_eval_used: bool
    private_used: bool
    receipt_sha256: str

    def __init__(
        self,
        *,
        _verification_marker: object,
        source_train: DevelopmentReasonerDataset,
        source_dev: DevelopmentReasonerDataset,
        evidence_authorization_sha256: str,
        verified_target_v2_receipt_sha256: str,
    ) -> None:
        if _verification_marker is not _BUNDLE_MARKER:
            raise TypeError("Development reasoner bundle requires the strict join")
        payload = {
            "schema_version": "soz_verified_development_iv_reasoner_data_bundle_v1",
            "source_train_dataset_sha256": source_train.receipt_sha256,
            "source_dev_dataset_sha256": source_dev.receipt_sha256,
            "evidence_authorization_sha256": evidence_authorization_sha256,
            "verified_target_v2_receipt_sha256": verified_target_v2_receipt_sha256,
            "formal_promotion": False,
            "formal_reasoner_authorized": False,
            "source_eval_used": False,
            "private_used": False,
        }
        object.__setattr__(self, "source_train", source_train)
        object.__setattr__(self, "source_dev", source_dev)
        object.__setattr__(self, "evidence_authorization_sha256", evidence_authorization_sha256)
        object.__setattr__(self, "verified_target_v2_receipt_sha256", verified_target_v2_receipt_sha256)
        object.__setattr__(self, "formal_promotion", False)
        object.__setattr__(self, "formal_reasoner_authorized", False)
        object.__setattr__(self, "source_eval_used", False)
        object.__setattr__(self, "private_used", False)
        object.__setattr__(self, "receipt_sha256", _canonical_sha256(payload))

    def assert_unchanged(self) -> None:
        self.source_train.assert_unchanged()
        self.source_dev.assert_unchanged()
        payload = {
            "schema_version": "soz_verified_development_iv_reasoner_data_bundle_v1",
            "source_train_dataset_sha256": self.source_train.receipt_sha256,
            "source_dev_dataset_sha256": self.source_dev.receipt_sha256,
            "evidence_authorization_sha256": self.evidence_authorization_sha256,
            "verified_target_v2_receipt_sha256": self.verified_target_v2_receipt_sha256,
            "formal_promotion": False,
            "formal_reasoner_authorized": False,
            "source_eval_used": False,
            "private_used": False,
        }
        if _canonical_sha256(payload) != self.receipt_sha256:
            raise ValueError("Development reasoner data bundle changed")


def _dataset_from_split(
    split: _DevelopmentSplitEvidence,
    *,
    capability_receipt_sha256: str,
    target: VerifiedDeepSOZTargetV2Artifact,
) -> DevelopmentReasonerDataset:
    expected_roster = tuple(
        values
        for name, values in target.receipt.eligible_split_patient_ids
        if name == split.model_split
    )
    if len(expected_roster) != 1:
        raise RuntimeError("Target-v2 split receipt is incomplete")
    if split.patient_ids != expected_roster[0]:
        raise ValueError(
            f"{split.model_split} evidence does not cover the exact eligible target roster"
        )
    target_batch = target.registry.target_batch(split.patient_ids)
    patient_to_index = {
        patient_id: index for index, patient_id in enumerate(split.patient_ids)
    }
    event_patient_index = torch.tensor(
        [patient_to_index[value] for value in split.patient_ids_by_event],
        dtype=torch.long,
    )
    counts = torch.bincount(event_patient_index, minlength=len(split.patient_ids))
    full = DevelopmentReasonerPatientBatch(
        _verification_marker=_PATIENT_BATCH_MARKER,
        evidence=split.evidence,
        event_patient_index=event_patient_index,
        patient_ids=split.patient_ids,
        event_ids=split.event_ids,
        expected_event_counts=counts,
        targets=target_batch.values.to(torch.float32),
        target_mask=target_batch.mask,
    )
    return DevelopmentReasonerDataset(
        _verification_marker=_DATASET_MARKER,
        model_split=split.model_split,
        full_batch=full,
        evidence_authorization_sha256=capability_receipt_sha256,
        verified_target_v2_receipt_sha256=target.receipt.receipt_sha256,
    )


def join_development_iv_targets(
    capability: VerifiedDevelopmentIVEvidenceCapability,
    target: VerifiedDeepSOZTargetV2Artifact,
) -> VerifiedDevelopmentReasonerDataBundle:
    """The sole target-reading stage for the development I+V candidate."""

    if type(capability) is not VerifiedDevelopmentIVEvidenceCapability:
        raise TypeError("Target join requires the closed development evidence capability")
    capability.assert_unchanged()
    _assert_verified_target_unchanged(target)
    receipt = capability.receipt
    bindings = {
        "target artifact": receipt.verified_target_v2_artifact_sha256
        == target.receipt.target_artifact_sha256,
        "target receipt": receipt.verified_target_v2_receipt_sha256
        == target.receipt.receipt_sha256,
        "target policy": receipt.verified_target_v2_policy_sha256
        == target.receipt.policy_sha256,
    }
    failed = tuple(name for name, passed in bindings.items() if not passed)
    if failed:
        raise ValueError(f"Candidate evidence and target-v2 lineage disagree: {failed}")
    authorization_sha = receipt.receipt_sha256
    source_train = _dataset_from_split(
        capability.source_train,
        capability_receipt_sha256=authorization_sha,
        target=target,
    )
    source_dev = _dataset_from_split(
        capability.source_dev,
        capability_receipt_sha256=authorization_sha,
        target=target,
    )
    return VerifiedDevelopmentReasonerDataBundle(
        _verification_marker=_BUNDLE_MARKER,
        source_train=source_train,
        source_dev=source_dev,
        evidence_authorization_sha256=authorization_sha,
        verified_target_v2_receipt_sha256=target.receipt.receipt_sha256,
    )


@dataclass(frozen=True)
class DevelopmentReasonerStepOutput:
    reasoner: DevelopmentIVReasonerOutput
    patient_logits: torch.Tensor
    patient_probabilities: torch.Tensor
    event_counts: torch.Tensor
    patient_abstain_recommended: torch.Tensor
    loss: SOZLossOutput
    ranking_weight: float = 0.25
    event_aggregation: str = "equal_event_mean_logits_before_patient_loss"
    patient_weighting: str = "equal_patient_macro"
    formal_promotion: bool = False


def development_reasoner_step(
    model: DevelopmentIVAdditiveReasoner,
    batch: DevelopmentReasonerPatientBatch,
) -> DevelopmentReasonerStepOutput:
    """Compute the frozen candidate objective; this function does not optimize."""

    if not isinstance(model, DevelopmentIVAdditiveReasoner):
        raise TypeError("Candidate step requires DevelopmentIVAdditiveReasoner")
    if not isinstance(batch, DevelopmentReasonerPatientBatch):
        raise TypeError("Candidate step requires a verified patient batch")
    output = model(batch.evidence)
    aggregation = batch.aggregate(output.event_logits)
    objective = PatientLevelSOZObjective(
        ranking_weight=0.25,
        ranking_margin=0.0,
        require_positive=True,
    )
    loss = objective(aggregation.logits, batch.targets, batch.target_mask)
    return DevelopmentReasonerStepOutput(
        reasoner=output,
        patient_logits=aggregation.logits,
        patient_probabilities=torch.sigmoid(aggregation.logits),
        event_counts=aggregation.event_counts,
        patient_abstain_recommended=batch.patient_abstain_recommended(),
        loss=loss,
    )


@dataclass(frozen=True)
class DevelopmentPatientExplanation:
    patient_logits: torch.Tensor
    component_contributions: Mapping[str, torch.Tensor]
    event_counts: torch.Tensor
    explanation_mode: str = DEVELOPMENT_IV_EXPLANATION_MODE
    llm_used_for_prediction: bool = False
    formal_promotion: bool = False


def aggregate_numeric_explanations(
    output: DevelopmentIVReasonerOutput,
    event_patient_index: torch.Tensor,
) -> DevelopmentPatientExplanation:
    """Mean additive numerical receipts using the same event policy as logits."""

    logit_aggregation = aggregate_patient_logits(
        output.event_logits, event_patient_index
    )
    components = {
        name: aggregate_patient_logits(value, event_patient_index).logits
        for name, value in output.component_contributions().items()
    }
    reconstructed = sum(components.values())
    if not torch.allclose(
        reconstructed, logit_aggregation.logits, atol=1e-6, rtol=1e-6
    ):
        raise RuntimeError("Patient numerical explanation does not reconstruct logits")
    return DevelopmentPatientExplanation(
        patient_logits=logit_aggregation.logits,
        component_contributions=components,
        event_counts=logit_aggregation.event_counts,
    )


__all__ = [
    "ABSENT_EVIDENCE_FAMILIES",
    "ACTIVE_EVIDENCE_FAMILIES",
    "DEVELOPMENT_IV_AUTHORIZATION_POLICY_SHA256",
    "DEVELOPMENT_IV_AUTHORIZATION_SCHEMA",
    "DEVELOPMENT_IV_EXPLANATION_MODE",
    "DevelopmentIVAdditiveReasoner",
    "DevelopmentIVEvidenceAuthorizationReceipt",
    "DevelopmentIVEvidenceBatch",
    "DevelopmentIVReasonerOutput",
    "DevelopmentPatientExplanation",
    "DevelopmentReasonerDataset",
    "DevelopmentReasonerPatientBatch",
    "DevelopmentReasonerStepOutput",
    "VerifiedDevelopmentIVEvidenceCapability",
    "VerifiedDevelopmentReasonerDataBundle",
    "aggregate_numeric_explanations",
    "development_reasoner_step",
    "issue_development_iv_evidence_capability",
    "join_development_iv_targets",
    "pool_ictal_seconds_to_tiles",
]
