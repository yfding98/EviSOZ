"""Typed Clinical Evidence modules for the EviSOZ-LM research route.

The implementation is intentionally independent of any teacher runtime.  It
accepts node/edge token tensors plus explicit masks, produces fixed query
evidence slots, and applies only a non-negative residual to the frozen v29
logits.  Residual use is a hard opt-in that requires an explicit ``GO`` Stage-0
status; the default path is an exact identity bypass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


N_NODES = 19
N_EDGES = 22
N_MOTIFS = 9
MOTIF_NAMES = (
    "attenuation",
    "LVFA",
    "rhythmic_theta",
    "rhythmic_delta",
    "rhythmic_alpha_beta",
    "spike_or_sharp_like",
    "evolving_rhythm",
    "artifact",
    "uncertain",
)
N_QUERIES = 6
QUERY_NAMES = (
    "quality",
    "morphology",
    "onset",
    "spread",
    "evolution",
    "localizability",
)

STRUCTURED_EVIDENCE_PIPELINE_CONFIG_SCHEMA_VERSION = (
    "evisoz_structured_evidence_pipeline_config_v1"
)


def _require_float_tokens(value: Tensor, *, name: str, ndim: int = 4) -> None:
    if not isinstance(value, Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a rank-{ndim} tensor")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite floating point")


def _require_mask(value: Tensor, *, name: str, shape: tuple[int, ...]) -> None:
    if not isinstance(value, Tensor) or value.dtype != torch.bool or tuple(value.shape) != shape:
        raise ValueError(f"{name} must be bool with shape {shape}")


class ClinicalMotifAdapter(nn.Module):
    """Fuse raw, frequency and baseline token views into motif-aware tokens."""

    def __init__(self, token_dim: int, *, motif_count: int = N_MOTIFS) -> None:
        super().__init__()
        if token_dim < 4 or motif_count < 1:
            raise ValueError("token_dim and motif_count must be positive")
        self.token_dim = token_dim
        self.motif_count = motif_count
        self.raw_norm = nn.LayerNorm(token_dim)
        self.frequency_projection = nn.Linear(token_dim, token_dim, bias=False)
        self.baseline_projection = nn.Linear(token_dim, token_dim, bias=False)
        self.frequency_gate = nn.Parameter(torch.tensor(-2.0))
        self.baseline_gate = nn.Parameter(torch.tensor(-2.0))
        self.motif_head = nn.Linear(token_dim, motif_count)

    def forward(
        self,
        raw_tokens: Tensor,
        *,
        frequency_tokens: Tensor | None = None,
        baseline_tokens: Tensor | None = None,
        token_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        _require_float_tokens(raw_tokens, name="raw_tokens")
        b, c, t, d = raw_tokens.shape
        if d != self.token_dim:
            raise ValueError("raw token dimension does not match adapter")
        fused = self.raw_norm(raw_tokens)
        for extra, projection, gate, name in (
            (frequency_tokens, self.frequency_projection, self.frequency_gate, "frequency_tokens"),
            (baseline_tokens, self.baseline_projection, self.baseline_gate, "baseline_tokens"),
        ):
            if extra is None:
                continue
            _require_float_tokens(extra, name=name)
            if tuple(extra.shape) != tuple(raw_tokens.shape):
                raise ValueError(f"{name} must match raw_tokens shape")
            fused = fused + torch.sigmoid(gate) * projection(extra)
        if token_mask is not None:
            _require_mask(token_mask, name="token_mask", shape=(b, c, t))
            fused = torch.where(token_mask.unsqueeze(-1), fused, torch.zeros_like(fused))
        motif_logits = self.motif_head(fused)
        return fused, motif_logits


class SparseTemporalSpatialEvidenceEncoder(nn.Module):
    """Model per-channel time evolution and retain only top-k spatial links."""

    def __init__(self, token_dim: int, *, hidden_dim: int | None = None, top_k: int = 4) -> None:
        super().__init__()
        if token_dim < 4:
            raise ValueError("token_dim must be at least 4")
        hidden = token_dim if hidden_dim is None else hidden_dim
        if hidden < 4 or top_k < 1 or top_k >= N_NODES:
            raise ValueError("invalid hidden_dim or top_k")
        self.token_dim = token_dim
        self.hidden_dim = hidden
        self.top_k = top_k
        self.temporal = nn.Sequential(
            nn.Conv1d(token_dim, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
        )
        self.node_projection = nn.Linear(hidden, token_dim)
        self.spatial_gate = nn.Parameter(torch.tensor(-2.0))
        self.spatial_norm = nn.LayerNorm(token_dim)
        self.edge_projection = nn.Linear(token_dim, token_dim)
        self.relation_query = nn.Linear(token_dim, token_dim, bias=False)
        self.relation_key = nn.Linear(token_dim, token_dim, bias=False)

    def _temporal_encode(self, tokens: Tensor, mask: Tensor, *, name: str) -> Tensor:
        _require_float_tokens(tokens, name=name)
        b, units, t, d = tokens.shape
        if d != self.token_dim:
            raise ValueError(f"{name} token dimension does not match encoder")
        _require_mask(mask, name=f"{name}_mask", shape=(b, units, t))
        x = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        x = x.reshape(b * units, t, d).transpose(1, 2)
        x = self.temporal(x).transpose(1, 2).reshape(b, units, t, self.hidden_dim)
        x = self.node_projection(x)
        return x * mask.unsqueeze(-1).to(x.dtype)

    def forward(
        self,
        node_tokens: Tensor,
        node_mask: Tensor,
        *,
        edge_tokens: Tensor | None = None,
        edge_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        node = self._temporal_encode(node_tokens, node_mask, name="node_tokens")
        b, _, t, _ = node.shape
        if edge_tokens is not None:
            if edge_mask is None:
                raise ValueError("edge_mask is required when edge_tokens are provided")
            if edge_tokens.shape[1] != N_EDGES:
                raise ValueError("edge_tokens must contain exactly 22 TCP edges")
            edge = self._temporal_encode(edge_tokens, edge_mask, name="edge_tokens")
            edge = self.edge_projection(edge)
        else:
            edge = None
            if edge_mask is not None:
                _require_mask(edge_mask, name="edge_mask", shape=(b, N_EDGES, t))

        # Dynamic relation mask is node-local and never expands edge evidence
        # to endpoints.  It is a sparse diagnostic carrier for spatial mixing.
        pooled = node.mean(dim=2)
        q = F.normalize(self.relation_query(pooled), dim=-1)
        k = F.normalize(self.relation_key(pooled), dim=-1)
        scores = torch.einsum("bcd,bed->bce", q, k)
        valid = node_mask.any(dim=2)
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        diagonal = torch.eye(N_NODES, device=node.device, dtype=torch.bool).unsqueeze(0)
        scores = scores.masked_fill(diagonal, float("-inf"))
        neighbours = scores.topk(self.top_k, dim=-1).indices
        # When fewer than ``top_k`` nodes are available, topk necessarily
        # returns ``-inf`` placeholders.  Keep those placeholders out of the
        # relation mask instead of silently treating missing nodes as links.
        finite_neighbours = torch.isfinite(torch.gather(scores, -1, neighbours))
        relation_mask = torch.zeros_like(scores, dtype=torch.bool)
        relation_mask.scatter_(-1, neighbours, finite_neighbours)

        # Apply the sparse relation graph to every time patch.  Rows without a
        # valid neighbour (for example an entirely masked or singleton node)
        # use a zero aggregate; they must not produce NaNs or invent signal.
        relation_scores = scores.masked_fill(~relation_mask, float("-inf"))
        row_has_relation = relation_mask.any(dim=-1)
        relation_scores = torch.where(
            row_has_relation.unsqueeze(-1),
            relation_scores,
            torch.zeros_like(relation_scores),
        )
        relation_weights = torch.softmax(relation_scores, dim=-1)
        relation_weights = relation_weights * relation_mask.to(relation_weights.dtype)
        spatial_context = torch.einsum(
            "bij,bjtd->bitd", relation_weights, node
        )
        node = self.spatial_norm(
            node + torch.sigmoid(self.spatial_gate) * spatial_context
        )
        node = node * node_mask.unsqueeze(-1).to(node.dtype)
        return node, edge, relation_mask


@dataclass(frozen=True)
class ClinicalEvidenceOutput:
    """Fixed-slot clinical evidence and typed prediction heads."""

    evidence_tokens: Tensor
    query_names: tuple[str, ...]
    quality_logits: Tensor
    morphology_logits: Tensor
    onset_logits: Tensor
    spread_logits: Tensor
    evolution_logits: Tensor
    localizability_logits: Tensor
    relation_mask: Tensor
    motif_logits: Tensor

    def validate(self, *, batch_size: int, token_dim: int) -> None:
        if tuple(self.evidence_tokens.shape) != (batch_size, N_QUERIES, token_dim):
            raise ValueError("evidence token shape drifted")
        if self.query_names != QUERY_NAMES:
            raise ValueError("query roster drifted")
        expected = {
            "quality_logits": (batch_size, 3),
            "morphology_logits": (batch_size, N_MOTIFS),
            "onset_logits": (batch_size, N_NODES),
            "spread_logits": (batch_size, N_EDGES),
            "evolution_logits": (batch_size, 4),
            "localizability_logits": (batch_size, 3),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} shape drifted")
        if self.relation_mask.ndim != 3 or self.relation_mask.shape[:2] != (batch_size, N_NODES):
            raise ValueError("relation mask shape drifted")


class ClinicalEvidenceQueryDecoder(nn.Module):
    """Read sparse node/edge tokens through six clinical evidence queries."""

    def __init__(self, token_dim: int, *, heads: int = 4) -> None:
        super().__init__()
        if token_dim % heads or heads < 1:
            raise ValueError("token_dim must be divisible by heads")
        self.token_dim = token_dim
        self.query = nn.Parameter(torch.zeros(N_QUERIES, token_dim))
        nn.init.normal_(self.query, std=0.02)
        self.cross_attention = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)
        self.quality_head = nn.Linear(token_dim, 3)
        self.morphology_head = nn.Linear(token_dim, N_MOTIFS)
        self.onset_head = nn.Linear(token_dim, N_NODES)
        self.spread_head = nn.Linear(token_dim, N_EDGES)
        self.evolution_head = nn.Linear(token_dim, 4)
        self.localizability_head = nn.Linear(token_dim, 3)

    def forward(
        self,
        node_tokens: Tensor,
        node_mask: Tensor,
        *,
        edge_tokens: Tensor | None,
        edge_mask: Tensor | None,
        relation_mask: Tensor,
        motif_logits: Tensor,
    ) -> ClinicalEvidenceOutput:
        _require_float_tokens(node_tokens, name="encoded_node_tokens")
        b, nodes, t, d = node_tokens.shape
        if (nodes, d) != (N_NODES, self.token_dim):
            raise ValueError("encoded node token shape drifted")
        _require_mask(node_mask, name="node_mask", shape=(b, N_NODES, t))
        if edge_tokens is not None:
            _require_float_tokens(edge_tokens, name="encoded_edge_tokens")
            if edge_mask is None or tuple(edge_tokens.shape[:2]) != (b, N_EDGES):
                raise ValueError("encoded edge token shape drifted")
            _require_mask(edge_mask, name="edge_mask", shape=tuple(edge_tokens.shape[:3]))
            source = torch.cat([node_tokens, edge_tokens], dim=1)
            source_mask = torch.cat([node_mask, edge_mask], dim=1)
        else:
            source = node_tokens
            source_mask = node_mask
        source = source.permute(0, 1, 2, 3).reshape(b, -1, d)
        source_mask = source_mask.reshape(b, -1)
        # MultiheadAttention produces NaNs when every key is masked for one
        # sample.  A zero sentinel is not evidence and only keeps the query
        # decoder numerically defined for fully missing/invalid inputs.
        empty_rows = ~source_mask.any(dim=1)
        if empty_rows.any():
            source = source.clone()
            source_mask = source_mask.clone()
            source[empty_rows, 0] = 0
            source_mask[empty_rows, 0] = True
        queries = self.query.unsqueeze(0).expand(b, -1, -1)
        attended, _ = self.cross_attention(
            queries,
            source,
            source,
            key_padding_mask=~source_mask,
            need_weights=False,
        )
        evidence = self.norm(queries + attended)
        output = ClinicalEvidenceOutput(
            evidence_tokens=evidence,
            query_names=QUERY_NAMES,
            quality_logits=self.quality_head(evidence[:, 0]),
            morphology_logits=self.morphology_head(evidence[:, 1]),
            onset_logits=self.onset_head(evidence[:, 2]),
            spread_logits=self.spread_head(evidence[:, 3]),
            evolution_logits=self.evolution_head(evidence[:, 4]),
            localizability_logits=self.localizability_head(evidence[:, 5]),
            relation_mask=relation_mask,
            motif_logits=motif_logits,
        )
        output.validate(batch_size=b, token_dim=d)
        return output


class ResidualSOZHead(nn.Module):
    """Apply a gated, non-negative residual to frozen v29 node logits."""

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.delta_projection = nn.Linear(token_dim, N_NODES)
        self.gate_projection = nn.Linear(token_dim, 1)
        nn.init.zeros_(self.delta_projection.weight)
        nn.init.zeros_(self.delta_projection.bias)
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.constant_(self.gate_projection.bias, -8.0)

    def forward(
        self,
        z0_node: Tensor,
        evidence_tokens: Tensor,
        *,
        candidate_mask: Tensor,
        residual_mode_eligible: Tensor,
        residual_enabled: bool = False,
        alpha: float = 0.0,
        stage0_status: str = "NO_GO",
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(z0_node, Tensor) or z0_node.ndim != 2 or z0_node.shape[1] != N_NODES:
            raise ValueError("z0_node must have shape [B,19]")
        if not torch.isfinite(z0_node).all():
            raise ValueError("z0_node must be finite")
        b = z0_node.shape[0]
        if tuple(evidence_tokens.shape[:2]) != (b, N_QUERIES):
            raise ValueError("evidence_tokens shape drifted")
        if evidence_tokens.shape[-1] != self.token_dim:
            raise ValueError("evidence token dimension drifted")
        _require_mask(candidate_mask, name="candidate_mask", shape=(b, N_NODES))
        _require_mask(residual_mode_eligible, name="residual_mode_eligible", shape=(b,))
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0,1]")
        if not residual_enabled or alpha == 0 or not residual_mode_eligible.any():
            return z0_node, torch.zeros_like(z0_node), torch.zeros((b, 1), device=z0_node.device)
        if stage0_status != "GO":
            raise PermissionError("non-zero residual requires Stage-0 status GO")
        pooled = evidence_tokens.mean(dim=1)
        raw_delta = self.delta_projection(pooled)
        delta = F.softplus(raw_delta) * candidate_mask.to(raw_delta.dtype)
        gate = torch.sigmoid(self.gate_projection(pooled))
        gate = gate * residual_mode_eligible.unsqueeze(-1).to(gate.dtype)
        return z0_node + float(alpha) * gate * delta, float(alpha) * gate * delta, gate


class EviSOZEvidencePipeline(nn.Module):
    """Composable evidence encoder/query decoder/residual path."""

    def __init__(self, token_dim: int, *, top_k: int = 4, query_heads: int = 4) -> None:
        super().__init__()
        self.motif_adapter = ClinicalMotifAdapter(token_dim)
        self.spatial_encoder = SparseTemporalSpatialEvidenceEncoder(token_dim, top_k=top_k)
        self.query_decoder = ClinicalEvidenceQueryDecoder(token_dim, heads=query_heads)
        self.residual_head = ResidualSOZHead(token_dim)

    def forward(
        self,
        node_tokens: Tensor,
        node_mask: Tensor,
        *,
        edge_tokens: Tensor | None = None,
        edge_mask: Tensor | None = None,
        frequency_tokens: Tensor | None = None,
        baseline_tokens: Tensor | None = None,
        z0_node: Tensor | None = None,
        candidate_mask: Tensor | None = None,
        residual_mode_eligible: Tensor | None = None,
        residual_enabled: bool = False,
        alpha: float = 0.0,
        stage0_status: str = "NO_GO",
    ) -> tuple[ClinicalEvidenceOutput, Tensor | None, Tensor | None, Tensor | None]:
        enriched, motif_logits = self.motif_adapter(
            node_tokens,
            frequency_tokens=frequency_tokens,
            baseline_tokens=baseline_tokens,
            token_mask=node_mask,
        )
        encoded_nodes, encoded_edges, relation_mask = self.spatial_encoder(
            enriched,
            node_mask,
            edge_tokens=edge_tokens,
            edge_mask=edge_mask,
        )
        evidence = self.query_decoder(
            encoded_nodes,
            node_mask,
            edge_tokens=encoded_edges,
            edge_mask=edge_mask,
            relation_mask=relation_mask,
            motif_logits=motif_logits,
        )
        if z0_node is None:
            if residual_enabled or alpha != 0:
                raise ValueError("z0_node is required for residual output")
            return evidence, None, None, None
        if candidate_mask is None or residual_mode_eligible is None:
            raise ValueError("candidate_mask and residual_mode_eligible are required with z0_node")
        z1, delta, gate = self.residual_head(
            z0_node,
            evidence.evidence_tokens,
            candidate_mask=candidate_mask,
            residual_mode_eligible=residual_mode_eligible,
            residual_enabled=residual_enabled,
            alpha=alpha,
            stage0_status=stage0_status,
        )
        return evidence, z1, delta, gate


class PatientEvidenceAggregator:
    """Aggregate event probabilities without averaging non-localizing events."""

    @staticmethod
    def aggregate(
        event_probabilities: Tensor,
        localization_quality: Tensor,
        signal_quality: Tensor,
        uncertainty: Tensor,
        *,
        localizing_event_mask: Tensor | None = None,
    ) -> dict[str, Any]:
        if event_probabilities.ndim != 2 or event_probabilities.shape[1] != N_NODES:
            raise ValueError("event_probabilities must have shape [S,19]")
        s = event_probabilities.shape[0]
        if s < 1 or not torch.isfinite(event_probabilities).all():
            raise ValueError("event probabilities must be non-empty and finite")
        for name, value in (
            ("localization_quality", localization_quality),
            ("signal_quality", signal_quality),
            ("uncertainty", uncertainty),
        ):
            if value.ndim != 1 or value.shape[0] != s or not torch.isfinite(value).all():
                raise ValueError(f"{name} must have shape [S] and be finite")
        if localizing_event_mask is None:
            localizing_event_mask = torch.ones(s, dtype=torch.bool, device=event_probabilities.device)
        _require_mask(localizing_event_mask, name="localizing_event_mask", shape=(s,))
        weights = localization_quality.clamp(0, 1) * signal_quality.clamp(0, 1)
        weights = weights * (1 - uncertainty.clamp(0, 1)) * localizing_event_mask.to(weights.dtype)
        denominator = weights.sum()
        if denominator <= 0:
            return {
                "patient_probability": None,
                "weights": weights,
                "abstain": True,
                "nonlocalizing_event_count": int((~localizing_event_mask).sum().item()),
            }
        probability = (weights.unsqueeze(-1) * event_probabilities).sum(dim=0) / denominator
        return {
            "patient_probability": probability,
            "weights": weights,
            "abstain": False,
            "nonlocalizing_event_count": int((~localizing_event_mask).sum().item()),
        }


def validate_structured_evidence_pipeline_config(value: object) -> dict[str, Any]:
    """Validate semantic invariants for the implementation-only pipeline config.

    JSON Schema validates types and enumerations; this validator additionally
    protects the frozen v29 route and the Stage-0 safety boundary.  It is
    intentionally pure and does not inspect checkpoints or private data.
    """

    if type(value) is not dict:
        raise TypeError("structured evidence pipeline config must be an object")
    data = dict(value)
    if data.get("schema_version") != STRUCTURED_EVIDENCE_PIPELINE_CONFIG_SCHEMA_VERSION:
        raise ValueError("structured evidence pipeline config schema_version drifted")
    baseline = data.get("baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("baseline must be an object")
    if baseline.get("model") != "canonical_v29_H_D":
        raise ValueError("pipeline must use the frozen canonical v29 H/D baseline")
    if baseline.get("route") != "standard19_car_exact_reference":
        raise ValueError("pipeline baseline route must be exact Standard19-CAR")
    if baseline.get("frozen") is not True:
        raise ValueError("v29 baseline must remain frozen")
    if baseline.get("residual_default_enabled") is not False:
        raise ValueError("residual path must default to disabled")
    if baseline.get("residual_requires_stage0_status") != "GO":
        raise ValueError("residual path must require explicit Stage-0 GO")

    contract = data.get("input_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("input_contract must be an object")
    node_shape = contract.get("node_token_shape")
    edge_shape = contract.get("edge_token_shape")
    if node_shape != ["B", 19, "T", 128] or edge_shape != ["B", 22, "T", 128]:
        raise ValueError("input contract must be Standard19 nodes and TCP22 edges")
    if contract.get("node_mask_required") is not True:
        raise ValueError("node mask is mandatory")
    if contract.get("edge_mask_required_when_edge_tokens_present") is not True:
        raise ValueError("edge mask is mandatory when TCP22 tokens are present")
    if contract.get("missing_channel_policy") != "explicit_mask_only":
        raise ValueError("missing channels must use explicit masks")
    if contract.get("interpolation") != "independent_ablation_only":
        raise ValueError("interpolation must remain an independent ablation")

    expected_queries = list(QUERY_NAMES)
    if data.get("query_names") != expected_queries:
        raise ValueError("query roster drifted")
    if data.get("motif_names") != list(MOTIF_NAMES):
        raise ValueError("motif roster drifted")
    qwen = data.get("qwen_connector")
    if not isinstance(qwen, Mapping):
        raise ValueError("qwen_connector must be an object")
    if qwen.get("model_family") != "Qwen3.8-27B":
        raise ValueError("qwen connector model family drifted")
    if qwen.get("hidden_size") != 5120 or qwen.get("evidence_token_count") != 32:
        raise ValueError("qwen connector dimensions drifted")
    if qwen.get("base_model_frozen") is not True or qwen.get("visual_tower_frozen") is not True:
        raise ValueError("Qwen base model and visual tower must remain frozen")
    if qwen.get("requires_stage0_go_for_training") is not True:
        raise ValueError("Qwen training must require Stage-0 GO")
    safety = data.get("safety")
    if not isinstance(safety, Mapping):
        raise ValueError("safety must be an object")
    for key in (
        "edge_to_node_endpoint_expansion",
        "text_directly_supervises_localization",
        "knowledge_creates_patient_facts",
        "teacher_runtime_required",
        "negative_residual_allowed",
    ):
        if safety.get(key) is not False:
            raise ValueError(f"safety.{key} must be false")
    if data.get("status") == "implementation_only_stage0_training_blocked":
        if baseline.get("residual_default_enabled") is not False:
            raise ValueError("blocked Stage-0 config cannot enable residuals")
    return data


__all__ = [
    "ClinicalEvidenceOutput",
    "ClinicalEvidenceQueryDecoder",
    "ClinicalMotifAdapter",
    "EviSOZEvidencePipeline",
    "PatientEvidenceAggregator",
    "ResidualSOZHead",
    "SparseTemporalSpatialEvidenceEncoder",
    "MOTIF_NAMES",
    "STRUCTURED_EVIDENCE_PIPELINE_CONFIG_SCHEMA_VERSION",
    "validate_structured_evidence_pipeline_config",
]
