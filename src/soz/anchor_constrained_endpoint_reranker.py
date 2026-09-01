"""Constrained exact-label reranking between adjacent scalp electrodes.

This module is deliberately downstream of the existing temporal-MIL anchor.
It has no data loader and accepts neither raw EEG nor dataset paths.  Its only
trainable input is a detached physical-node feature tensor.  Training pairs
are created from the *exact* DeepSOZ positive set only when both endpoints of
one frozen candidate edge are observed and exactly one endpoint is positive.
The candidate graph is the deduplicated TCP-20 plus official one-hop union
from :func:`anchor_endpoint_features.endpoint_adjacency_edges`.  This is a
metric-informed safety prior, not an anatomical propagation graph.

The reranker's logit is ``u(second) - u(first)`` for one shared endpoint
utility function.  Swapping endpoints therefore negates the logit.  At
deployment, a unique anchor Top-1 may only be replaced by one directly
adjacent candidate endpoint.  All flip thresholds are fixed before OOF
evaluation; OOF exact outcomes do not select or enable a threshold.

The pair probability is conditional on the XOR endpoint target.  It is not a
probability of seizure involvement, propagation, or causal SOZ membership,
and a spatial neighbour is never promoted to a positive label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchor_endpoint_features import (
    ENDPOINT_NODE_FEATURE_DIM,
    endpoint_adjacency_edges,
)
from .geometry import CHANNEL_INDEX, N_STANDARD_CHANNELS, TCP_20_EDGES


ANCHOR_CONSTRAINED_ENDPOINT_RERANKER_SCHEMA = (
    "soz_anchor_constrained_exact_endpoint_pairwise_reranker_v9"
)
ENDPOINT_L2_WEIGHT = 0.05
ENDPOINT_LBFGS_MAX_ITER = 200
ENDPOINT_FLIP_LOGIT_MARGIN = math.log(3.0)
MAX_ANCHOR_GAP_Z = 1.0
CROSSFIT_GATE_SCOPE = "outer_train_inner_oof_only"

_PAIR_BATCH_MARKER = object()
_TCP_INDEX_PAIRS = tuple(
    (CHANNEL_INDEX[left], CHANNEL_INDEX[right]) for left, right in TCP_20_EDGES
)
_TCP_UNORDERED_PAIRS = frozenset(
    (min(left, right), max(left, right)) for left, right in _TCP_INDEX_PAIRS
)
_ADJACENCY_INDEX_PAIRS = endpoint_adjacency_edges()
_ADJACENCY_UNORDERED_PAIRS = frozenset(_ADJACENCY_INDEX_PAIRS)


def _is_tcp_pair(first: int, second: int) -> bool:
    return (min(first, second), max(first, second)) in _TCP_UNORDERED_PAIRS


def _is_adjacent_pair(first: int, second: int) -> bool:
    return (min(first, second), max(first, second)) in _ADJACENCY_UNORDERED_PAIRS


def _validate_patient_ids(patient_ids: Sequence[str], patients: int) -> tuple[str, ...]:
    values = tuple(str(value).strip() for value in patient_ids)
    if len(values) != patients or any(not value for value in values):
        raise ValueError("patient_ids must contain one non-empty ID per patient")
    if len(set(values)) != patients:
        raise ValueError("patient_ids must be unique")
    return values


def _validate_exact_targets(
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    patients: int,
    device: torch.device,
) -> None:
    expected = (patients, N_STANDARD_CHANNELS)
    if tuple(targets.shape) != expected or tuple(target_mask.shape) != expected:
        raise ValueError("exact targets and target_mask must have shape [P,19]")
    if not targets.is_floating_point() or target_mask.dtype != torch.bool:
        raise TypeError("exact targets must be floating point and target_mask bool")
    if targets.device != device or target_mask.device != device:
        raise ValueError("features, exact targets, and target_mask must share a device")
    observed = targets[target_mask]
    if not torch.isfinite(observed).all() or (
        observed.numel() and not torch.all((observed == 0) | (observed == 1))
    ):
        raise ValueError("observed exact targets must be finite binary values")


@dataclass(frozen=True)
class ExactEndpointPairBatch:
    """Detached XOR-labelled pairs derived by the exact-label builder."""

    endpoint_features: torch.Tensor
    endpoint_indices: torch.Tensor
    edge_is_tcp: torch.Tensor
    pair_patient_index: torch.Tensor
    second_endpoint_positive: torch.Tensor
    patient_ids: tuple[str, ...]
    _marker: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._marker is not _PAIR_BATCH_MARKER:
            raise TypeError(
                "ExactEndpointPairBatch must be built from exact DeepSOZ targets"
            )
        pairs = int(self.endpoint_features.shape[0])
        if tuple(self.endpoint_features.shape[1:]) != (
            2,
            ENDPOINT_NODE_FEATURE_DIM,
        ) or self.endpoint_features.ndim != 3:
            raise ValueError("endpoint_features must have frozen shape [Q,2,70]")
        if pairs < 1:
            raise ValueError("at least one informative exact endpoint pair is required")
        if tuple(self.endpoint_indices.shape) != (pairs, 2) or (
            self.endpoint_indices.dtype != torch.long
        ):
            raise TypeError("endpoint_indices must be long [Q,2]")
        if tuple(self.edge_is_tcp.shape) != (pairs,) or self.edge_is_tcp.dtype != (
            torch.bool
        ):
            raise TypeError("edge_is_tcp must be bool [Q]")
        if tuple(self.pair_patient_index.shape) != (pairs,) or (
            self.pair_patient_index.dtype != torch.long
        ):
            raise TypeError("pair_patient_index must be long [Q]")
        if tuple(self.second_endpoint_positive.shape) != (pairs,) or not (
            self.second_endpoint_positive.is_floating_point()
        ):
            raise TypeError("second_endpoint_positive must be floating point [Q]")
        devices = {
            self.endpoint_features.device,
            self.endpoint_indices.device,
            self.edge_is_tcp.device,
            self.pair_patient_index.device,
            self.second_endpoint_positive.device,
        }
        if len(devices) != 1:
            raise ValueError("all exact endpoint pair tensors must share a device")
        if not torch.isfinite(self.endpoint_features).all() or (
            self.endpoint_features.requires_grad
        ):
            raise ValueError("endpoint features must be finite and detached")
        if not torch.isfinite(self.second_endpoint_positive).all() or not torch.all(
            (self.second_endpoint_positive == 0)
            | (self.second_endpoint_positive == 1)
        ):
            raise ValueError("pair targets must be finite binary XOR directions")
        patients = len(self.patient_ids)
        _validate_patient_ids(self.patient_ids, patients)
        if patients < 1 or torch.any(self.pair_patient_index < 0) or torch.any(
            self.pair_patient_index >= patients
        ):
            raise ValueError("pair_patient_index lies outside patient_ids")
        indices = self.endpoint_indices.detach().cpu().tolist()
        if any(
            first == second
            or first < 0
            or second < 0
            or first >= N_STANDARD_CHANNELS
            or second >= N_STANDARD_CHANNELS
            or not _is_adjacent_pair(first, second)
            for first, second in indices
        ):
            raise ValueError("every training pair must use the frozen candidate graph")
        expected_tcp = torch.tensor(
            [_is_tcp_pair(first, second) for first, second in indices],
            dtype=torch.bool,
            device=self.edge_is_tcp.device,
        )
        if not torch.equal(self.edge_is_tcp, expected_tcp):
            raise ValueError("edge_is_tcp provenance disagrees with TCP-20")
        keys = [
            (int(patient), min(first, second), max(first, second))
            for patient, (first, second) in zip(
                self.pair_patient_index.detach().cpu().tolist(), indices
            )
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("a patient cannot contribute the same candidate pair twice")

    @property
    def patient_count(self) -> int:
        return len(self.patient_ids)

    @property
    def pair_count(self) -> int:
        return int(self.endpoint_features.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.endpoint_features.shape[2])

    def swapped(self) -> "ExactEndpointPairBatch":
        """Return the same exact pairs with endpoint order and target reversed."""

        return ExactEndpointPairBatch(
            endpoint_features=self.endpoint_features.flip(1).contiguous(),
            endpoint_indices=self.endpoint_indices.flip(1).contiguous(),
            edge_is_tcp=self.edge_is_tcp,
            pair_patient_index=self.pair_patient_index,
            second_endpoint_positive=1.0 - self.second_endpoint_positive,
            patient_ids=self.patient_ids,
            _marker=_PAIR_BATCH_MARKER,
        )

    def to(self, device: str | torch.device) -> "ExactEndpointPairBatch":
        return ExactEndpointPairBatch(
            endpoint_features=self.endpoint_features.to(device=device),
            endpoint_indices=self.endpoint_indices.to(device=device),
            edge_is_tcp=self.edge_is_tcp.to(device=device),
            pair_patient_index=self.pair_patient_index.to(device=device),
            second_endpoint_positive=self.second_endpoint_positive.to(device=device),
            patient_ids=self.patient_ids,
            _marker=_PAIR_BATCH_MARKER,
        )


def build_deepsoz_exact_endpoint_training_pairs(
    node_features: torch.Tensor,
    exact_targets: torch.Tensor,
    target_mask: torch.Tensor,
    patient_ids: Sequence[str],
) -> ExactEndpointPairBatch:
    """Build adjacent pairs with one and only one exact positive endpoint.

    ``target_mask=False`` means unknown and excludes the pair.  No spatial
    official one-hop adjacency limits which exact endpoint pairs may be
    compared, but never expands the positive set.  No seizure-involvement
    label or missing-label imputation is performed.  Features are detached so
    this recovery head cannot update the upstream LaBraM/evidence producer.
    """

    if node_features.ndim != 3 or tuple(node_features.shape[1:]) != (
        N_STANDARD_CHANNELS,
        ENDPOINT_NODE_FEATURE_DIM,
    ):
        raise ValueError("node_features must have frozen shape [P,19,70]")
    if not node_features.is_floating_point() or not torch.isfinite(node_features).all():
        raise ValueError("node_features must be finite floating point")
    patients = int(node_features.shape[0])
    if patients < 1:
        raise ValueError("at least one patient is required")
    ids = _validate_patient_ids(patient_ids, patients)
    _validate_exact_targets(
        exact_targets,
        target_mask,
        patients=patients,
        device=node_features.device,
    )

    feature_rows: list[torch.Tensor] = []
    endpoint_rows: list[tuple[int, int]] = []
    patient_rows: list[int] = []
    target_rows: list[torch.Tensor] = []
    for patient in range(patients):
        for left, right in _ADJACENCY_INDEX_PAIRS:
            if not (
                bool(target_mask[patient, left])
                and bool(target_mask[patient, right])
            ):
                continue
            left_positive = bool(exact_targets[patient, left] == 1)
            right_positive = bool(exact_targets[patient, right] == 1)
            if left_positive == right_positive:
                continue
            feature_rows.append(node_features[patient, [left, right]].detach())
            endpoint_rows.append((left, right))
            patient_rows.append(patient)
            target_rows.append(
                exact_targets.new_tensor(1.0 if right_positive else 0.0)
            )
    if not feature_rows:
        raise ValueError(
            "no observed candidate edge has exactly one DeepSOZ exact positive endpoint"
        )
    return ExactEndpointPairBatch(
        endpoint_features=torch.stack(feature_rows).contiguous(),
        endpoint_indices=torch.tensor(
            endpoint_rows, dtype=torch.long, device=node_features.device
        ),
        edge_is_tcp=torch.tensor(
            [_is_tcp_pair(left, right) for left, right in endpoint_rows],
            dtype=torch.bool,
            device=node_features.device,
        ),
        pair_patient_index=torch.tensor(
            patient_rows, dtype=torch.long, device=node_features.device
        ),
        second_endpoint_positive=torch.stack(target_rows).contiguous(),
        patient_ids=ids,
        _marker=_PAIR_BATCH_MARKER,
    )


class AnchorConstrainedEndpointReranker(nn.Module):
    """Low-capacity shared endpoint utility with an antisymmetric pair logit."""

    def __init__(self, input_dim: int = ENDPOINT_NODE_FEATURE_DIM) -> None:
        super().__init__()
        if input_dim != ENDPOINT_NODE_FEATURE_DIM:
            raise ValueError("v9 endpoint input_dim is frozen at 70")
        self.input_dim = ENDPOINT_NODE_FEATURE_DIM
        self.endpoint_utility = nn.Linear(self.input_dim, 1, bias=False)

    @property
    def n_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def score_endpoints(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 2 or features.shape[-1] != self.input_dim:
            raise ValueError(f"endpoint features must end in dimension {self.input_dim}")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("endpoint features must be finite floating point")
        return self.endpoint_utility(features).squeeze(-1)

    def forward(self, endpoint_features: torch.Tensor) -> torch.Tensor:
        if endpoint_features.ndim != 3 or endpoint_features.shape[1] != 2:
            raise ValueError("endpoint_features must have shape [Q,2,D]")
        difference = endpoint_features[:, 1] - endpoint_features[:, 0]
        return F.linear(
            difference,
            self.endpoint_utility.weight,
            bias=None,
        ).squeeze(-1)


@dataclass(frozen=True)
class EndpointRerankerObjectiveOutput:
    total: torch.Tensor
    bradley_terry: torch.Tensor
    l2_penalty: torch.Tensor


def anchor_constrained_endpoint_objective(
    model: AnchorConstrainedEndpointReranker,
    batch: ExactEndpointPairBatch,
) -> EndpointRerankerObjectiveOutput:
    """Frozen v9 objective: patient-balanced BT plus ``0.05 * ||w||²``."""

    if type(model) is not AnchorConstrainedEndpointReranker or type(batch) is not (
        ExactEndpointPairBatch
    ):
        raise TypeError("objective requires the frozen reranker and exact pair batch")
    logits = model(batch.endpoint_features)
    bradley_terry = patient_balanced_bradley_terry_loss(
        logits,
        batch.second_endpoint_positive,
        batch.pair_patient_index,
        patient_count=batch.patient_count,
    )
    l2_penalty = model.endpoint_utility.weight.square().sum()
    total = bradley_terry + ENDPOINT_L2_WEIGHT * l2_penalty
    return EndpointRerankerObjectiveOutput(
        total=total,
        bradley_terry=bradley_terry,
        l2_penalty=l2_penalty,
    )


def patient_balanced_bradley_terry_loss(
    pair_logits: torch.Tensor,
    second_endpoint_positive: torch.Tensor,
    pair_patient_index: torch.Tensor,
    *,
    patient_count: int,
) -> torch.Tensor:
    """Average Bradley--Terry loss within patient, then across patients.

    The target is conditional: one means the second endpoint is the exact
    positive and zero means the first endpoint is.  Patients without an
    informative XOR edge are excluded rather than assigned an artificial
    target.
    """

    pairs = int(pair_logits.shape[0])
    if pair_logits.ndim != 1 or pairs < 1:
        raise ValueError("pair_logits must be a non-empty vector")
    if tuple(second_endpoint_positive.shape) != (pairs,) or not (
        second_endpoint_positive.is_floating_point()
    ):
        raise TypeError("second_endpoint_positive must be floating point [Q]")
    if tuple(pair_patient_index.shape) != (pairs,) or (
        pair_patient_index.dtype != torch.long
    ):
        raise TypeError("pair_patient_index must be long [Q]")
    if isinstance(patient_count, bool) or int(patient_count) != patient_count or (
        patient_count < 1
    ):
        raise ValueError("patient_count must be a positive integer")
    if pair_logits.device != second_endpoint_positive.device or (
        pair_logits.device != pair_patient_index.device
    ):
        raise ValueError("Bradley--Terry inputs must share a device")
    if not pair_logits.is_floating_point() or not torch.isfinite(pair_logits).all():
        raise ValueError("pair_logits must be finite floating point")
    if not torch.isfinite(second_endpoint_positive).all() or not torch.all(
        (second_endpoint_positive == 0) | (second_endpoint_positive == 1)
    ):
        raise ValueError("Bradley--Terry targets must be finite binary values")
    if torch.any(pair_patient_index < 0) or torch.any(
        pair_patient_index >= patient_count
    ):
        raise ValueError("pair_patient_index lies outside patient_count")

    direction = 2.0 * second_endpoint_positive - 1.0
    elementwise = F.softplus(-direction * pair_logits)
    sums = pair_logits.new_zeros((patient_count,))
    counts = pair_logits.new_zeros((patient_count,))
    sums.scatter_add_(0, pair_patient_index, elementwise)
    counts.scatter_add_(0, pair_patient_index, torch.ones_like(elementwise))
    eligible = counts > 0
    return (sums[eligible] / counts[eligible]).mean()


def _validate_anchor_inputs(
    anchor_scores: torch.Tensor,
    evaluable_mask: torch.Tensor,
) -> None:
    expected = (anchor_scores.shape[0], N_STANDARD_CHANNELS)
    if anchor_scores.ndim != 2 or tuple(anchor_scores.shape) != expected:
        raise ValueError("anchor_scores must have shape [P,19]")
    if tuple(evaluable_mask.shape) != expected or evaluable_mask.dtype != torch.bool:
        raise TypeError("evaluable_mask must be bool [P,19]")
    if anchor_scores.device != evaluable_mask.device:
        raise ValueError("anchor_scores and evaluable_mask must share a device")
    if not anchor_scores.is_floating_point() or not torch.isfinite(anchor_scores).all():
        raise ValueError("anchor_scores must be finite floating point")
    if anchor_scores.shape[0] < 1 or not evaluable_mask.any(dim=1).all():
        raise ValueError("every patient needs at least one evaluable electrode")


def _masked_top_set(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    maximum = scores.masked_fill(~mask, -torch.inf).max(dim=1, keepdim=True).values
    return mask & (scores == maximum)


@dataclass(frozen=True)
class AnchorEndpointProposal:
    """One best adjacent candidate plus all frozen v9 local safety checks.

    ``h_direction_pass`` and ``v_direction_pass`` mean only that two declared
    feature-group contributions both rank the candidate above the anchor.
    They are not independent evidence sources and do not establish causality.
    """

    anchor_index: torch.Tensor
    candidate_index: torch.Tensor
    candidate_edge_is_tcp: torch.Tensor
    candidate_minus_anchor_logit: torch.Tensor
    candidate_confidence: torch.Tensor
    candidate_available: torch.Tensor
    margin_pass: torch.Tensor
    h_direction_pass: torch.Tensor
    v_direction_pass: torch.Tensor
    anchor_gap_pass: torch.Tensor
    eligible: torch.Tensor

    def __post_init__(self) -> None:
        patients = int(self.anchor_index.shape[0])
        if self.anchor_index.ndim != 1 or self.anchor_index.dtype != torch.long:
            raise TypeError("anchor_index must be long [P]")
        if tuple(self.candidate_index.shape) != (patients,) or (
            self.candidate_index.dtype != torch.long
        ):
            raise TypeError("candidate_index must be long [P]")
        if tuple(self.candidate_edge_is_tcp.shape) != (patients,) or (
            self.candidate_edge_is_tcp.dtype != torch.bool
        ):
            raise TypeError("candidate_edge_is_tcp must be bool [P]")
        if tuple(self.candidate_minus_anchor_logit.shape) != (patients,) or not (
            self.candidate_minus_anchor_logit.is_floating_point()
        ):
            raise TypeError("candidate logit must be floating point [P]")
        if tuple(self.candidate_confidence.shape) != (patients,) or not (
            self.candidate_confidence.is_floating_point()
        ):
            raise TypeError("candidate confidence must be floating point [P]")
        bool_fields = (
            "candidate_available",
            "margin_pass",
            "h_direction_pass",
            "v_direction_pass",
            "anchor_gap_pass",
            "eligible",
        )
        for name in bool_fields:
            value = getattr(self, name)
            if tuple(value.shape) != (patients,) or value.dtype != torch.bool:
                raise TypeError(f"{name} must be bool [P]")
        devices = {
            self.anchor_index.device,
            self.candidate_index.device,
            self.candidate_edge_is_tcp.device,
            self.candidate_minus_anchor_logit.device,
            self.candidate_confidence.device,
            *(getattr(self, name).device for name in bool_fields),
        }
        if len(devices) != 1:
            raise ValueError("proposal tensors must share a device")
        if not torch.isfinite(self.candidate_minus_anchor_logit).all() or not (
            torch.isfinite(self.candidate_confidence).all()
        ):
            raise ValueError("proposal logits and confidences must be finite")
        if torch.any(self.candidate_confidence < 0) or torch.any(
            self.candidate_confidence > 1
        ):
            raise ValueError("proposal confidence must lie in [0,1]")
        expected_confidence = torch.sigmoid(self.candidate_minus_anchor_logit)
        if not torch.allclose(
            self.candidate_confidence, expected_confidence, atol=1e-7, rtol=1e-7
        ):
            raise ValueError("proposal confidence must be sigmoid(pair logit)")
        expected_eligible = (
            self.candidate_available
            & self.margin_pass
            & self.h_direction_pass
            & self.v_direction_pass
            & self.anchor_gap_pass
        )
        if not torch.equal(self.eligible, expected_eligible):
            raise ValueError("eligible does not equal all frozen v9 local checks")
        for name in (
            "h_direction_pass",
            "v_direction_pass",
            "anchor_gap_pass",
        ):
            if bool((getattr(self, name) & ~self.candidate_available).any()):
                raise ValueError(f"{name} cannot pass without an available candidate")
        if not torch.equal(
            self.margin_pass,
            self.candidate_available
            & (self.candidate_minus_anchor_logit >= ENDPOINT_FLIP_LOGIT_MARGIN),
        ):
            raise ValueError("margin_pass disagrees with the frozen log(3) margin")
        for patient in range(patients):
            anchor = int(self.anchor_index[patient].item())
            candidate = int(self.candidate_index[patient].item())
            if bool(self.candidate_available[patient]):
                if not (
                    0 <= anchor < N_STANDARD_CHANNELS
                    and 0 <= candidate < N_STANDARD_CHANNELS
                    and _is_adjacent_pair(anchor, candidate)
                ):
                    raise ValueError("proposal is outside the frozen candidate graph")
                if bool(self.candidate_edge_is_tcp[patient]) != _is_tcp_pair(
                    anchor, candidate
                ):
                    raise ValueError("proposal edge provenance disagrees with TCP-20")
            elif anchor != -1 or candidate != -1:
                raise ValueError("ineligible proposals must use sentinel endpoint -1")
            elif bool(self.candidate_edge_is_tcp[patient]):
                raise ValueError("unavailable proposal cannot have TCP provenance")

    @property
    def proposal_count(self) -> int:
        return int(self.eligible.sum().item())

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_available.sum().item())

    @property
    def official_only_candidate_count(self) -> int:
        return int(
            (self.candidate_available & ~self.candidate_edge_is_tcp).sum().item()
        )


def propose_anchor_adjacent_endpoint(
    model: AnchorConstrainedEndpointReranker,
    node_features: torch.Tensor,
    anchor_scores: torch.Tensor,
    evaluable_mask: torch.Tensor,
    h_node_contribution: torch.Tensor,
    v_node_contribution: torch.Tensor,
    anchor_gap_z: torch.Tensor,
) -> AnchorEndpointProposal:
    """Score frozen-graph neighbours and apply every frozen v9 local check.

    ``anchor_gap_z`` must use the caller-defined, frozen, target-free
    normalization.  This function does not inspect labels or fit a normalizer.
    """

    if type(model) is not AnchorConstrainedEndpointReranker:
        raise TypeError("model must be AnchorConstrainedEndpointReranker")
    _validate_anchor_inputs(anchor_scores, evaluable_mask)
    patients = int(anchor_scores.shape[0])
    if tuple(node_features.shape) != (
        patients,
        N_STANDARD_CHANNELS,
        ENDPOINT_NODE_FEATURE_DIM,
    ):
        raise ValueError("node_features must have frozen shape [P,19,70]")
    for name, contribution in (
        ("h_node_contribution", h_node_contribution),
        ("v_node_contribution", v_node_contribution),
    ):
        if tuple(contribution.shape) != tuple(anchor_scores.shape) or not (
            contribution.is_floating_point()
        ):
            raise TypeError(f"{name} must be floating point [P,19]")
        if contribution.device != anchor_scores.device or not torch.isfinite(
            contribution
        ).all():
            raise ValueError(f"{name} must be finite and share the anchor device")
    if tuple(anchor_gap_z.shape) != (patients,) or not anchor_gap_z.is_floating_point():
        raise TypeError("anchor_gap_z must be floating point [P]")
    if anchor_gap_z.device != anchor_scores.device or not torch.isfinite(
        anchor_gap_z
    ).all():
        raise ValueError("anchor_gap_z must be finite and share the anchor device")
    if node_features.device != anchor_scores.device:
        raise ValueError("node features and anchor inputs must share a device")
    parameter_device = next(model.parameters()).device
    if parameter_device != node_features.device:
        raise ValueError("reranker and node features must share a device")
    if not node_features.is_floating_point() or not torch.isfinite(node_features).all():
        raise ValueError("node_features must be finite floating point")

    with torch.no_grad():
        utilities = model.score_endpoints(node_features.detach())
    anchor_index = torch.full(
        (patients,), -1, dtype=torch.long, device=anchor_scores.device
    )
    candidate_index = torch.full_like(anchor_index, -1)
    candidate_edge_is_tcp = torch.zeros(
        patients, dtype=torch.bool, device=anchor_scores.device
    )
    pair_logit = anchor_scores.new_zeros((patients,))
    candidate_available = torch.zeros(
        patients, dtype=torch.bool, device=anchor_scores.device
    )
    top_set = _masked_top_set(anchor_scores, evaluable_mask)

    neighbours: list[list[int]] = [[] for _ in range(N_STANDARD_CHANNELS)]
    for left, right in _ADJACENCY_INDEX_PAIRS:
        neighbours[left].append(right)
        neighbours[right].append(left)
    for patient in range(patients):
        top = torch.nonzero(top_set[patient], as_tuple=False).flatten()
        if top.numel() != 1:
            continue
        anchor = int(top.item())
        candidates = tuple(
            index for index in neighbours[anchor] if bool(evaluable_mask[patient, index])
        )
        if not candidates:
            continue
        candidate_tensor = torch.tensor(
            candidates, dtype=torch.long, device=anchor_scores.device
        )
        logits = utilities[patient].index_select(0, candidate_tensor) - utilities[
            patient, anchor
        ]
        maximum = logits.max()
        tied = candidate_tensor[logits == maximum]
        if tied.numel() != 1:
            continue
        candidate = int(tied.item())
        anchor_index[patient] = anchor
        candidate_index[patient] = candidate
        candidate_edge_is_tcp[patient] = _is_tcp_pair(anchor, candidate)
        pair_logit[patient] = maximum
        candidate_available[patient] = True
    confidence = torch.sigmoid(pair_logit)
    safe_anchor = anchor_index.clamp_min(0)
    safe_candidate = candidate_index.clamp_min(0)
    rows = torch.arange(patients, device=anchor_scores.device)
    margin_pass = candidate_available & (pair_logit >= ENDPOINT_FLIP_LOGIT_MARGIN)
    h_direction_pass = candidate_available & (
        h_node_contribution[rows, safe_candidate]
        > h_node_contribution[rows, safe_anchor]
    )
    v_direction_pass = candidate_available & (
        v_node_contribution[rows, safe_candidate]
        > v_node_contribution[rows, safe_anchor]
    )
    anchor_gap_pass = candidate_available & (anchor_gap_z <= MAX_ANCHOR_GAP_Z)
    eligible = (
        candidate_available
        & margin_pass
        & h_direction_pass
        & v_direction_pass
        & anchor_gap_pass
    )
    return AnchorEndpointProposal(
        anchor_index=anchor_index,
        candidate_index=candidate_index,
        candidate_edge_is_tcp=candidate_edge_is_tcp,
        candidate_minus_anchor_logit=pair_logit,
        candidate_confidence=confidence,
        candidate_available=candidate_available,
        margin_pass=margin_pass,
        h_direction_pass=h_direction_pass,
        v_direction_pass=v_direction_pass,
        anchor_gap_pass=anchor_gap_pass,
        eligible=eligible,
    )


@dataclass(frozen=True)
class CrossFittedConfidenceGate:
    """Target-free audit receipt for fixed-rule inner-OOF proposals.

    This object does not select a threshold and is not required to apply v9.
    It merely verifies that proposal rows carry at least two declared OOF folds
    while preserving the already frozen selective margin.
    """

    logit_margin: float
    confidence_threshold: float
    candidate_count: int
    eligible_count: int
    fold_count: int
    scope: str

    def __post_init__(self) -> None:
        expected_confidence = 1.0 / (1.0 + math.exp(-ENDPOINT_FLIP_LOGIT_MARGIN))
        if self.logit_margin != ENDPOINT_FLIP_LOGIT_MARGIN or not math.isclose(
            self.confidence_threshold,
            expected_confidence,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("confidence gate must use the frozen log(3) margin")
        counts = (self.candidate_count, self.eligible_count, self.fold_count)
        if any(isinstance(value, bool) or value < 0 for value in counts):
            raise ValueError("confidence-gate counts must be non-negative integers")
        if self.eligible_count > self.candidate_count or self.fold_count < 2:
            raise ValueError("cross-fit audit counts are inconsistent")
        if self.scope != CROSSFIT_GATE_SCOPE:
            raise ValueError("confidence gate has a non-nested fitting scope")


def fit_cross_fitted_confidence_gate(
    proposal: AnchorEndpointProposal,
    inner_oof_fold_index: torch.Tensor,
    *,
    scope: str = CROSSFIT_GATE_SCOPE,
) -> CrossFittedConfidenceGate:
    """Validate cross-fit structure without labels or threshold selection.

    The name is retained as the cross-fitted confidence-gate interface, but it
    performs no fitting from OOF outcomes.  v9 fixes ``log(3)``, H/V direction,
    and gap-z rules before evaluation.  The resulting receipt is optional and
    cannot disable :func:`apply_fixed_selective_endpoint_rerank`.
    """

    if type(proposal) is not AnchorEndpointProposal:
        raise TypeError("proposal must come from propose_anchor_adjacent_endpoint")
    patients = int(proposal.eligible.shape[0])
    if tuple(inner_oof_fold_index.shape) != (patients,) or (
        inner_oof_fold_index.dtype != torch.long
    ):
        raise TypeError("inner_oof_fold_index must be long [P]")
    if inner_oof_fold_index.device != proposal.eligible.device:
        raise ValueError("fold indices and proposals must share a device")
    if scope != CROSSFIT_GATE_SCOPE:
        raise ValueError("confidence gate accepts outer-train inner-OOF rows only")
    if not bool(proposal.candidate_available.any()):
        raise ValueError("no cross-fitted endpoint candidate is available")
    if bool((inner_oof_fold_index[proposal.candidate_available] < 0).any()):
        raise ValueError("every candidate requires an inner-OOF fold")
    fold_count = int(
        torch.unique(inner_oof_fold_index[proposal.candidate_available]).numel()
    )
    if fold_count < 2:
        raise ValueError("cross-fit audit requires at least two inner-OOF folds")
    confidence_threshold = 1.0 / (1.0 + math.exp(-ENDPOINT_FLIP_LOGIT_MARGIN))
    return CrossFittedConfidenceGate(
        logit_margin=ENDPOINT_FLIP_LOGIT_MARGIN,
        confidence_threshold=confidence_threshold,
        candidate_count=proposal.candidate_count,
        eligible_count=proposal.proposal_count,
        fold_count=fold_count,
        scope=CROSSFIT_GATE_SCOPE,
    )


@dataclass(frozen=True)
class AnchorPreservingRerankOutput:
    scores: torch.Tensor
    applied: torch.Tensor

    @property
    def applied_count(self) -> int:
        return int(self.applied.sum().item())


def apply_fixed_selective_endpoint_rerank(
    anchor_scores: torch.Tensor,
    evaluable_mask: torch.Tensor,
    proposal: AnchorEndpointProposal,
) -> AnchorPreservingRerankOutput:
    """Apply the frozen v9 local rule; every rejected row keeps its anchor."""

    _validate_anchor_inputs(anchor_scores, evaluable_mask)
    if type(proposal) is not AnchorEndpointProposal:
        raise TypeError("application requires an AnchorEndpointProposal")
    patients = int(anchor_scores.shape[0])
    if tuple(proposal.eligible.shape) != (patients,) or (
        proposal.eligible.device != anchor_scores.device
    ):
        raise ValueError("proposal does not align with held anchor scores")
    scores = anchor_scores.clone()
    applied = torch.zeros(patients, dtype=torch.bool, device=anchor_scores.device)
    apply_mask = proposal.eligible
    original_top_set = _masked_top_set(anchor_scores, evaluable_mask)
    for patient in torch.nonzero(apply_mask, as_tuple=False).flatten().tolist():
        anchor = int(proposal.anchor_index[patient].item())
        candidate = int(proposal.candidate_index[patient].item())
        if int(original_top_set[patient].sum().item()) != 1 or not bool(
            original_top_set[patient, anchor]
        ):
            raise ValueError("proposal anchor is not the unique current anchor Top-1")
        if not (
            bool(evaluable_mask[patient, candidate])
            and _is_adjacent_pair(anchor, candidate)
        ):
            raise ValueError("held candidate is outside the evaluable frozen graph")
        anchor_value = scores[patient, anchor].clone()
        candidate_value = scores[patient, candidate].clone()
        scores[patient, anchor] = candidate_value
        scores[patient, candidate] = anchor_value
        applied[patient] = True

    if not torch.equal(scores[~applied], anchor_scores[~applied]):
        raise RuntimeError("a confidence-rejected patient did not keep its anchor")
    new_top_set = _masked_top_set(scores, evaluable_mask)
    for patient in torch.nonzero(applied, as_tuple=False).flatten().tolist():
        candidate = int(proposal.candidate_index[patient].item())
        if int(new_top_set[patient].sum().item()) != 1 or not bool(
            new_top_set[patient, candidate]
        ):
            raise RuntimeError("endpoint swap failed to assign the proposed Top-1")
    return AnchorPreservingRerankOutput(scores=scores, applied=applied)


def apply_confidence_gated_endpoint_rerank(
    anchor_scores: torch.Tensor,
    evaluable_mask: torch.Tensor,
    proposal: AnchorEndpointProposal,
    gate: CrossFittedConfidenceGate,
) -> AnchorPreservingRerankOutput:
    """Apply fixed v9 rules after an optional target-free cross-fit audit."""

    if type(gate) is not CrossFittedConfidenceGate:
        raise TypeError("gate must be a target-free CrossFittedConfidenceGate")
    return apply_fixed_selective_endpoint_rerank(
        anchor_scores,
        evaluable_mask,
        proposal,
    )


__all__ = [
    "ANCHOR_CONSTRAINED_ENDPOINT_RERANKER_SCHEMA",
    "CROSSFIT_GATE_SCOPE",
    "ENDPOINT_FLIP_LOGIT_MARGIN",
    "ENDPOINT_L2_WEIGHT",
    "ENDPOINT_LBFGS_MAX_ITER",
    "ENDPOINT_NODE_FEATURE_DIM",
    "MAX_ANCHOR_GAP_Z",
    "AnchorEndpointProposal",
    "AnchorPreservingRerankOutput",
    "AnchorConstrainedEndpointReranker",
    "CrossFittedConfidenceGate",
    "ExactEndpointPairBatch",
    "EndpointRerankerObjectiveOutput",
    "anchor_constrained_endpoint_objective",
    "apply_confidence_gated_endpoint_rerank",
    "apply_fixed_selective_endpoint_rerank",
    "build_deepsoz_exact_endpoint_training_pairs",
    "fit_cross_fitted_confidence_gate",
    "patient_balanced_bradley_terry_loss",
    "propose_anchor_adjacent_endpoint",
]
