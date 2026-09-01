"""Development-only LaBraM morphology hierarchy recovery.

This module deliberately stays in the native TUEV bipolar-edge coordinate.
It consumes frozen contextualized LaBraM tokens and explicit CE6 cells only;
it has no channel/SOZ output and never treats an unannotated cell as negative.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import MORPHOLOGY_CLASSES, N_STANDARD_CHANNELS, N_TCP_EDGES
from .models.concept_heads import NodeToEdgeTokens
from .models.labram import AUDITED_LABRAM_BASE_SHA256


MORPHOLOGY_RECOVERY_PROTOCOL_SCHEMA = (
    "soz_labram_morphology_hierarchical_recovery_protocol_v1"
)
MORPHOLOGY_RECOVERY_PREFLIGHT_SCHEMA = (
    "soz_labram_morphology_hierarchical_recovery_preflight_v1"
)
AUXILIARY_MORPHOLOGY_ROLES = (
    "localizing",
    "artifact",
    "generalized",
)

# Rows follow MORPHOLOGY_CLASSES exactly:
# SPSW, GPED, PLED, EYEM, ARTF, BCKG.
_CE6_TO_AUXILIARY = torch.tensor(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    ),
    dtype=torch.float32,
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def morphology_recovery_protocol_payload() -> dict[str, object]:
    """Return the single frozen candidate; intentionally no threshold grid."""

    return {
        "schema_version": MORPHOLOGY_RECOVERY_PROTOCOL_SCHEMA,
        "candidate_count": 1,
        "candidate_name": "shared_adapter_ce6_plus_three_ce6_derived_roles",
        "status": "development_only",
        "foundation": {
            "name": "LaBraM-Base",
            "checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
            "frozen": True,
            "train_from_scratch": False,
        },
        "input": {
            "shape": ["B", N_STANDARD_CHANNELS, 1, 200],
            "meaning": "contextualized_slot0_from_four_second_encoder_call",
            "preprocessing_arm": "C-CAR19",
        },
        "coordinate": {
            "target": "native_common20_bipolar_edge",
            "edge_count": N_TCP_EDGES,
            "endpoint_expansion": False,
            "channel_or_soz_head": False,
        },
        "architecture": {
            "ordered_edge_features": "[H_a,H_b,H_a-H_b]",
            "edge_feature_dim": 600,
            "shared_hidden_dim": 128,
            "native_head": list(MORPHOLOGY_CLASSES),
            "auxiliary_head": list(AUXILIARY_MORPHOLOGY_ROLES),
        },
        "auxiliary_target_source": "deterministic_coarsening_of_observed_ce6_only",
        "loss": {
            "formula": "CE6+(localizing_BCE+artifact_BCE+generalized_BCE)/3",
            "group_equal": True,
            "overlap_component_weighted": True,
            "ce6_class_weight_cap": 10.0,
            "auxiliary_pos_weight_cap": 10.0,
            "unknown_as_negative": False,
        },
        "schedule": {
            "fold_count": 5,
            "fixed_epochs": 20,
            "seed_base": 20260808,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0,
            "early_stopping": False,
            "hyperparameter_sweep": False,
        },
        "uses_official_tuev_eval": False,
        "uses_tusz_labels": False,
        "uses_deepsoz_soz_labels": False,
        "uses_private": False,
        "formal_promotion": False,
        "dense_deployment_authorized": False,
        "soz_reasoner_authorized": False,
    }


MORPHOLOGY_RECOVERY_PROTOCOL_SHA256 = _canonical_sha256(
    morphology_recovery_protocol_payload()
)


@dataclass(frozen=True)
class HierarchicalMorphologyOutput:
    ce6_logits: torch.Tensor
    auxiliary_logits: torch.Tensor

    def __post_init__(self) -> None:
        if self.ce6_logits.ndim != 4 or self.ce6_logits.shape[-1] != 6:
            raise ValueError("CE6 logits must have shape [B,20,T,6]")
        if self.ce6_logits.shape[1] != N_TCP_EDGES:
            raise ValueError("CE6 logits must stay on the native common20 edges")
        if tuple(self.auxiliary_logits.shape) != (
            *self.ce6_logits.shape[:-1],
            len(AUXILIARY_MORPHOLOGY_ROLES),
        ):
            raise ValueError("Auxiliary logits must have shape [B,20,T,3]")
        for value in (self.ce6_logits, self.auxiliary_logits):
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError("Morphology recovery logits must be finite floating point")


class HierarchicalMorphologyEvidenceHead(nn.Module):
    """One shared adapter with native CE6 and training-only role heads."""

    def __init__(self, *, token_dim: int = 200, hidden_dim: int = 128) -> None:
        super().__init__()
        if token_dim < 1 or hidden_dim < 1:
            raise ValueError("token_dim and hidden_dim must be positive")
        self.edge_tokens = NodeToEdgeTokens(token_dim=token_dim)
        self.adapter = nn.Sequential(
            nn.Linear(self.edge_tokens.output_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.ce6_classifier = nn.Linear(hidden_dim, len(MORPHOLOGY_CLASSES))
        self.auxiliary_classifier = nn.Linear(
            hidden_dim, len(AUXILIARY_MORPHOLOGY_ROLES)
        )

    def forward(self, tokens: torch.Tensor) -> HierarchicalMorphologyOutput:
        hidden = self.adapter(self.edge_tokens(tokens))
        return HierarchicalMorphologyOutput(
            ce6_logits=self.ce6_classifier(hidden),
            auxiliary_logits=self.auxiliary_classifier(hidden),
        )


def _validate_native_targets(
    labels: torch.Tensor,
    source_target_mask: torch.Tensor,
    overlap_component_weights: torch.Tensor | None = None,
) -> None:
    if labels.ndim != 3 or labels.shape[1] != N_TCP_EDGES:
        raise ValueError("Morphology labels must have shape [B,20,T]")
    if tuple(source_target_mask.shape) != tuple(labels.shape):
        raise ValueError("Morphology target mask must match [B,20,T]")
    if labels.dtype != torch.long or source_target_mask.dtype != torch.bool:
        raise TypeError("Morphology labels must be long and mask must be bool")
    observed = labels[source_target_mask]
    if observed.numel() < 1:
        raise ValueError("At least one explicit CE6 target is required")
    if torch.any((observed < 0) | (observed >= len(MORPHOLOGY_CLASSES))):
        raise ValueError("Observed morphology labels must be native CE6")
    if overlap_component_weights is not None:
        if tuple(overlap_component_weights.shape) != tuple(labels.shape):
            raise ValueError("Morphology component weights must match [B,20,T]")
        if (
            not overlap_component_weights.is_floating_point()
            or not torch.isfinite(overlap_component_weights).all()
        ):
            raise ValueError("Morphology component weights must be finite floating point")
        if torch.any(overlap_component_weights[~source_target_mask] != 0):
            raise ValueError("Unknown morphology cells must carry zero component weight")
        if torch.any(overlap_component_weights[source_target_mask] <= 0):
            raise ValueError("Observed morphology cells require positive component weight")


def derive_morphology_role_targets(
    labels: torch.Tensor,
    source_target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coarsen explicit CE6 cells; unknown cells remain all-zero and masked."""

    _validate_native_targets(labels, source_target_mask)
    safe = torch.where(source_target_mask, labels, torch.zeros_like(labels))
    mapping = _CE6_TO_AUXILIARY.to(device=labels.device)
    roles = mapping[safe]
    role_mask = source_target_mask.unsqueeze(-1).expand_as(roles)
    roles = torch.where(role_mask, roles, torch.zeros_like(roles))
    return roles, role_mask


@dataclass(frozen=True)
class HierarchicalMorphologyWeights:
    ce6_class_weights: torch.Tensor
    auxiliary_pos_weights: torch.Tensor

    def __post_init__(self) -> None:
        if tuple(self.ce6_class_weights.shape) != (len(MORPHOLOGY_CLASSES),):
            raise ValueError("CE6 class weights must have shape [6]")
        if tuple(self.auxiliary_pos_weights.shape) != (
            len(AUXILIARY_MORPHOLOGY_ROLES),
        ):
            raise ValueError("Auxiliary positive weights must have shape [3]")
        for value in (self.ce6_class_weights, self.auxiliary_pos_weights):
            if (
                not value.is_floating_point()
                or not torch.isfinite(value).all()
                or torch.any(value <= 0)
            ):
                raise ValueError("Morphology recovery weights must be finite and positive")


def fit_hierarchical_morphology_weights(
    labels: torch.Tensor,
    source_target_mask: torch.Tensor,
    overlap_component_weights: torch.Tensor,
    *,
    cap: float = 10.0,
) -> HierarchicalMorphologyWeights:
    """Fit CE6 and role imbalance weights from one fit fold only."""

    _validate_native_targets(labels, source_target_mask, overlap_component_weights)
    if not math.isfinite(float(cap)) or cap < 1:
        raise ValueError("Weight cap must be finite and >=1")
    observed_labels = labels[source_target_mask]
    observed_weights = overlap_component_weights[source_target_mask].double()
    ce6_mass = torch.zeros(len(MORPHOLOGY_CLASSES), dtype=torch.float64, device=labels.device)
    ce6_mass.scatter_add_(0, observed_labels, observed_weights)
    if torch.any(ce6_mass <= 0):
        missing = [
            MORPHOLOGY_CLASSES[index]
            for index in torch.where(ce6_mass <= 0)[0].tolist()
        ]
        raise ValueError(f"Fit fold lacks CE6 support: {missing}")
    ce6_weights = ce6_mass.sum() / (len(MORPHOLOGY_CLASSES) * ce6_mass)
    ce6_weights /= ce6_weights.mean()
    ce6_weights = ce6_weights.clamp(max=float(cap))
    ce6_weights /= ce6_weights.mean()

    roles, _ = derive_morphology_role_targets(labels, source_target_mask)
    observed_roles = roles[source_target_mask].double()
    positive_mass = (observed_roles * observed_weights[:, None]).sum(dim=0)
    total_mass = observed_weights.sum()
    negative_mass = total_mass - positive_mass
    if torch.any(positive_mass <= 0) or torch.any(negative_mass <= 0):
        raise ValueError("Fit fold lacks positive or negative auxiliary-role support")
    auxiliary = (negative_mass / positive_mass).clamp(max=float(cap))
    return HierarchicalMorphologyWeights(
        ce6_class_weights=ce6_weights.float(),
        auxiliary_pos_weights=auxiliary.float(),
    )


def hierarchical_morphology_group_balanced_loss(
    output: HierarchicalMorphologyOutput,
    labels: torch.Tensor,
    source_target_mask: torch.Tensor,
    overlap_component_weights: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    ce6_class_weights: torch.Tensor,
    auxiliary_pos_weights: torch.Tensor,
) -> torch.Tensor:
    """Native CE6 plus three CE6-derived tasks, equalised over source groups."""

    if not isinstance(output, HierarchicalMorphologyOutput):
        raise TypeError("output must be HierarchicalMorphologyOutput")
    _validate_native_targets(labels, source_target_mask, overlap_component_weights)
    if tuple(output.ce6_logits.shape[:-1]) != tuple(labels.shape):
        raise ValueError("Recovery logits and native targets have different geometry")
    if (
        group_ids.ndim != 1
        or group_ids.shape[0] != labels.shape[0]
        or group_ids.dtype != torch.long
    ):
        raise ValueError("group_ids must be long [B]")
    tensors = (
        labels,
        source_target_mask,
        overlap_component_weights,
        group_ids,
        ce6_class_weights,
        auxiliary_pos_weights,
    )
    if any(value.device != output.ce6_logits.device for value in tensors):
        raise ValueError("Loss tensors must share one device")
    if tuple(ce6_class_weights.shape) != (6,) or tuple(
        auxiliary_pos_weights.shape
    ) != (3,):
        raise ValueError("Recovery loss weights must have shapes [6] and [3]")
    if (
        not torch.isfinite(ce6_class_weights).all()
        or not torch.isfinite(auxiliary_pos_weights).all()
        or torch.any(ce6_class_weights <= 0)
        or torch.any(auxiliary_pos_weights <= 0)
    ):
        raise ValueError("Recovery loss weights must be finite and positive")

    safe_labels = torch.where(source_target_mask, labels, torch.zeros_like(labels))
    ce6_element = F.cross_entropy(
        output.ce6_logits.movedim(-1, 1),
        safe_labels,
        weight=ce6_class_weights,
        reduction="none",
    )
    role_targets, _ = derive_morphology_role_targets(labels, source_target_mask)
    auxiliary_element = F.binary_cross_entropy_with_logits(
        output.auxiliary_logits,
        role_targets.to(output.auxiliary_logits.dtype),
        pos_weight=auxiliary_pos_weights,
        reduction="none",
    ).mean(dim=-1)
    combined = ce6_element + auxiliary_element

    group_losses: list[torch.Tensor] = []
    for group_id in torch.unique(group_ids, sorted=True):
        rows = group_ids == group_id
        observed = source_target_mask[rows]
        weights = overlap_component_weights[rows]
        denominator = weights[observed].sum()
        if observed.any() and denominator > 0:
            group_losses.append((combined[rows] * weights)[observed].sum() / denominator)
    if not group_losses:
        raise ValueError("No observed morphology target remains after grouping")
    return torch.stack(group_losses).mean()


@dataclass(frozen=True)
class MorphologyRecoveryPreflightReceipt:
    item_count: int
    group_count: int
    fold_count: int
    fold_group_counts: tuple[int, ...]
    observed_cell_count: int
    unknown_cell_count: int
    fold_ce6_component_mass: tuple[tuple[float, ...], ...]
    fold_role_positive_mass: tuple[tuple[float, ...], ...]
    group_fold_roster_sha256: str
    source_plan_sha256: str
    protocol_sha256: str = MORPHOLOGY_RECOVERY_PROTOCOL_SHA256
    passed: bool = True
    formal_promotion: bool = False
    dense_deployment_authorized: bool = False
    soz_reasoner_authorized: bool = False
    schema_version: str = MORPHOLOGY_RECOVERY_PREFLIGHT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MORPHOLOGY_RECOVERY_PREFLIGHT_SCHEMA:
            raise ValueError("Unexpected morphology recovery preflight schema")
        if not self.passed:
            raise ValueError("Only passed preflight receipts may be instantiated")
        if self.fold_count != 5 or len(self.fold_group_counts) != self.fold_count:
            raise ValueError("Morphology recovery preflight requires five folds")
        if len(self.fold_ce6_component_mass) != self.fold_count or len(
            self.fold_role_positive_mass
        ) != self.fold_count:
            raise ValueError("Morphology recovery fold support receipt is incomplete")
        if any(len(row) != 6 or any(value <= 0 for value in row) for row in self.fold_ce6_component_mass):
            raise ValueError("Every held fold must support all CE6 classes")
        if any(len(row) != 3 or any(value <= 0 for value in row) for row in self.fold_role_positive_mass):
            raise ValueError("Every held fold must support all positive auxiliary roles")
        if self.formal_promotion or self.dense_deployment_authorized or self.soz_reasoner_authorized:
            raise ValueError("Development morphology preflight cannot authorize deployment")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "item_count": self.item_count,
            "group_count": self.group_count,
            "fold_count": self.fold_count,
            "fold_group_counts": list(self.fold_group_counts),
            "observed_cell_count": self.observed_cell_count,
            "unknown_cell_count": self.unknown_cell_count,
            "fold_ce6_component_mass": [list(row) for row in self.fold_ce6_component_mass],
            "fold_role_positive_mass": [list(row) for row in self.fold_role_positive_mass],
            "group_fold_roster_sha256": self.group_fold_roster_sha256,
            "source_plan_sha256": self.source_plan_sha256,
            "protocol_sha256": self.protocol_sha256,
            "formal_promotion": self.formal_promotion,
            "dense_deployment_authorized": self.dense_deployment_authorized,
            "soz_reasoner_authorized": self.soz_reasoner_authorized,
        }

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_morphology_recovery_preflight(
    value: str | Path,
    *,
    expected_receipt_sha256: str | None = None,
    verify_source_files: bool = True,
) -> tuple[MorphologyRecoveryPreflightReceipt, dict[str, object]]:
    """Strictly reload a passed, non-authorizing recovery preflight bundle."""

    path = Path(value)
    if path.is_dir():
        path = path / "preflight.json"
    path = path.resolve(strict=True)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Morphology recovery preflight exceeds the closed size limit")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Morphology recovery preflight must be a JSON object")
    receipt_fields = {
        "item_count",
        "group_count",
        "fold_count",
        "fold_group_counts",
        "observed_cell_count",
        "unknown_cell_count",
        "fold_ce6_component_mass",
        "fold_role_positive_mass",
        "group_fold_roster_sha256",
        "source_plan_sha256",
        "protocol_sha256",
        "passed",
        "formal_promotion",
        "dense_deployment_authorized",
        "soz_reasoner_authorized",
        "schema_version",
    }
    missing = receipt_fields - set(payload)
    if missing:
        raise ValueError(f"Morphology recovery preflight lacks fields: {sorted(missing)}")
    receipt = MorphologyRecoveryPreflightReceipt(
        item_count=int(payload["item_count"]),
        group_count=int(payload["group_count"]),
        fold_count=int(payload["fold_count"]),
        fold_group_counts=tuple(int(value) for value in payload["fold_group_counts"]),
        observed_cell_count=int(payload["observed_cell_count"]),
        unknown_cell_count=int(payload["unknown_cell_count"]),
        fold_ce6_component_mass=tuple(
            tuple(float(value) for value in row)
            for row in payload["fold_ce6_component_mass"]
        ),
        fold_role_positive_mass=tuple(
            tuple(float(value) for value in row)
            for row in payload["fold_role_positive_mass"]
        ),
        group_fold_roster_sha256=str(payload["group_fold_roster_sha256"]),
        source_plan_sha256=str(payload["source_plan_sha256"]),
        protocol_sha256=str(payload["protocol_sha256"]),
        passed=payload["passed"],
        formal_promotion=payload["formal_promotion"],
        dense_deployment_authorized=payload["dense_deployment_authorized"],
        soz_reasoner_authorized=payload["soz_reasoner_authorized"],
        schema_version=str(payload["schema_version"]),
    )
    declared = str(payload.get("receipt_sha256", ""))
    if declared != receipt.receipt_sha256:
        raise ValueError("Morphology recovery preflight receipt SHA mismatch")
    if expected_receipt_sha256 is not None and declared != expected_receipt_sha256:
        raise ValueError("Morphology recovery preflight is not the expected receipt")
    if payload.get("official_tuev_eval_opened_for_candidate") is not False:
        raise ValueError("Recovery preflight must not open official TUEV evaluation")
    if payload.get("gpu_used") is not False or payload.get("optimization_performed") is not False:
        raise ValueError("A preflight bundle cannot claim optimization or GPU execution")
    protocol_file = path.parent / str(payload.get("protocol_file", ""))
    if not protocol_file.is_file() or _file_sha256(protocol_file) != payload.get(
        "protocol_file_sha256"
    ):
        raise ValueError("Morphology recovery protocol file binding changed")
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    if (
        not isinstance(protocol, dict)
        or _canonical_sha256(protocol) != receipt.protocol_sha256
        or protocol != morphology_recovery_protocol_payload()
    ):
        raise ValueError("Morphology recovery frozen protocol changed")
    sources = payload.get("source_files")
    if not isinstance(sources, dict) or set(sources) != {
        "run_plan",
        "tokens",
        "labels",
        "mask",
        "weights",
    }:
        raise ValueError("Morphology recovery preflight source roster is not closed")
    if verify_source_files:
        for name, row in sources.items():
            if not isinstance(row, dict):
                raise TypeError(f"Recovery source row {name} must be a mapping")
            source = Path(str(row.get("path", ""))).resolve(strict=True)
            if (
                source.stat().st_size != int(row.get("size_bytes", -1))
                or _file_sha256(source) != row.get("sha256")
            ):
                raise ValueError(f"Morphology recovery source changed: {name}")
    return receipt, payload


def _numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def audit_morphology_recovery_source(
    *,
    run_plan: Mapping[str, object],
    tokens: object,
    labels: object,
    source_target_mask: object,
    overlap_component_weights: object,
) -> MorphologyRecoveryPreflightReceipt:
    """Fail closed before any morphology recovery optimization is allowed."""

    if not isinstance(run_plan, Mapping):
        raise TypeError("run_plan must be a mapping")
    if (
        run_plan.get("source_scope") != "public_source_train_only"
        or run_plan.get("private_data_used") is not False
        or run_plan.get("soz_labels_used") is not False
    ):
        raise ValueError("Morphology recovery crossed a forbidden data boundary")
    if run_plan.get("labram_checkpoint_sha256") != AUDITED_LABRAM_BASE_SHA256:
        raise ValueError("Morphology recovery requires the audited LaBraM-Base checkpoint")
    nested = run_plan.get("nested_dev_manifest")
    if not isinstance(nested, Mapping):
        raise ValueError("Morphology recovery lacks its nested source-train manifest")
    if (
        nested.get("fold_count") != 5
        or nested.get("source_dev_used") is not False
        or nested.get("source_eval_used") is not False
        or nested.get("private_data_used") is not False
        or nested.get("soz_labels_used") is not False
    ):
        raise ValueError("Morphology recovery nested manifest crossed a forbidden data boundary")

    token_array = _numpy(tokens)
    label_array = _numpy(labels)
    mask_array = _numpy(source_target_mask)
    weight_array = _numpy(overlap_component_weights)
    if token_array.ndim != 4 or tuple(token_array.shape[1:]) != (19, 1, 200):
        raise ValueError("Recovery tokens must be contextualized [N,19,1,200]")
    expected = (token_array.shape[0], N_TCP_EDGES)
    if label_array.shape != expected or mask_array.shape != expected or weight_array.shape != expected:
        raise ValueError("Recovery CE6 labels/mask/weights must be [N,20]")
    if not np.issubdtype(token_array.dtype, np.floating) or not np.isfinite(token_array).all():
        raise ValueError("Recovery tokens must be finite floating point")
    if not np.issubdtype(label_array.dtype, np.integer):
        raise TypeError("Recovery CE6 labels must be integer")
    if mask_array.dtype != np.bool_:
        raise TypeError("Recovery target mask must be bool")
    if not np.issubdtype(weight_array.dtype, np.floating) or not np.isfinite(weight_array).all():
        raise ValueError("Recovery component weights must be finite floating point")
    if np.any(weight_array[~mask_array] != 0):
        raise ValueError("Unknown morphology cells must carry zero component weight")
    if not mask_array.any() or np.any(weight_array[mask_array] <= 0):
        raise ValueError("Observed morphology cells require positive component weight")
    observed_labels = label_array[mask_array]
    if np.any((observed_labels < 0) | (observed_labels >= 6)):
        raise ValueError("Observed recovery targets must be native CE6")

    items = run_plan.get("tuev_items")
    if not isinstance(items, list) or len(items) != token_array.shape[0]:
        raise ValueError("TUEV item roster and recovery arrays disagree")
    indices = [int(item.get("index", -1)) for item in items if isinstance(item, Mapping)]
    if len(indices) != len(items) or sorted(indices) != list(range(len(items))):
        raise ValueError("TUEV recovery item indices are not a complete unique roster")
    group_fold: dict[str, int] = {}
    item_fold = np.empty(len(items), dtype=np.int64)
    for item in items:
        assert isinstance(item, Mapping)
        index = int(item["index"])
        fold = int(item.get("fold", -1))
        group_id = str(item.get("parent_group_id", ""))
        record_id = str(item.get("record_id", ""))
        if fold not in range(5):
            raise ValueError("TUEV recovery fold must be 0--4")
        if not group_id.startswith("train-subject:") or not record_id.startswith("train/"):
            raise ValueError("Morphology recovery item is not official-train TUEV")
        previous = group_fold.setdefault(group_id, fold)
        if previous != fold:
            raise ValueError("One morphology group crosses OOF folds")
        item_fold[index] = fold

    records = nested.get("records")
    if not isinstance(records, list):
        raise ValueError("Nested source-train manifest lacks records")
    eligible_records: dict[str, Mapping[str, object]] = {}
    patient_fold: dict[str, int] = {}
    component_fold: dict[str, int] = {}
    for record in records:
        if not isinstance(record, Mapping) or record.get("common_raw_qc_eligible") is not True:
            continue
        fold = int(record.get("nested_dev_fold", -1))
        if fold not in range(5):
            raise ValueError("Eligible nested record lacks a valid fold")
        if (
            record.get("official_partition") != "train"
            or record.get("source_train_only") is not True
            or record.get("soz_labels_present") is not False
        ):
            raise ValueError("Nested record is not label-safe official source-train")
        patient = str(record.get("patient_identity_key", ""))
        component = str(record.get("content_component_id", ""))
        if not patient or not component:
            raise ValueError("Nested record lacks patient/content identity")
        previous_patient = patient_fold.setdefault(patient, fold)
        previous_component = component_fold.setdefault(component, fold)
        if previous_patient != fold:
            raise ValueError("One patient crosses morphology OOF folds")
        if previous_component != fold:
            raise ValueError("One exact-content component crosses morphology OOF folds")
        if record.get("dataset_id") == "TUEV":
            record_id = str(record.get("record_id", ""))
            if record_id in eligible_records:
                raise ValueError("Nested TUEV record roster contains a duplicate")
            eligible_records[record_id] = record
    for item in items:
        assert isinstance(item, Mapping)
        record_id = str(item["record_id"])
        record = eligible_records.get(record_id)
        if record is None or int(record["nested_dev_fold"]) != int(item["fold"]):
            raise ValueError("TUEV item fold disagrees with its nested record")

    fold_group_counts = tuple(
        sum(value == fold for value in group_fold.values()) for fold in range(5)
    )
    if any(count < 1 for count in fold_group_counts):
        raise ValueError("Every morphology OOF fold requires at least one group")
    fold_ce6_mass: list[tuple[float, ...]] = []
    fold_role_mass: list[tuple[float, ...]] = []
    role_mapping = _CE6_TO_AUXILIARY.numpy().astype(np.float64)
    for fold in range(5):
        held_rows = item_fold == fold
        held_mask = mask_array[held_rows]
        held_labels = label_array[held_rows]
        held_weights = weight_array[held_rows].astype(np.float64, copy=False)
        masses = tuple(
            float(held_weights[(held_labels == class_index) & held_mask].sum())
            for class_index in range(6)
        )
        if any(value <= 0 for value in masses):
            raise ValueError(f"Held fold {fold} lacks native CE6 support")
        role_masses = tuple(
            sum(masses[class_index] * role_mapping[class_index, role] for class_index in range(6))
            for role in range(3)
        )
        if any(value <= 0 for value in role_masses):
            raise ValueError(f"Held fold {fold} lacks positive auxiliary-role support")
        # Fit-only weight estimation must also be feasible for every OOF model.
        fit_rows = ~held_rows
        fit_mask = mask_array[fit_rows]
        fit_labels = label_array[fit_rows]
        fit_weights = weight_array[fit_rows].astype(np.float64, copy=False)
        fit_masses = tuple(
            float(fit_weights[(fit_labels == class_index) & fit_mask].sum())
            for class_index in range(6)
        )
        if any(value <= 0 for value in fit_masses):
            raise ValueError(f"Fit fold {fold} lacks native CE6 support")
        fit_total = sum(fit_masses)
        for role in range(3):
            positive = sum(
                fit_masses[class_index] * role_mapping[class_index, role]
                for class_index in range(6)
            )
            if positive <= 0 or fit_total - positive <= 0:
                raise ValueError(f"Fit fold {fold} lacks positive/negative role support")
        fold_ce6_mass.append(masses)
        fold_role_mass.append(role_masses)

    group_fold_roster = tuple(sorted(group_fold.items()))
    return MorphologyRecoveryPreflightReceipt(
        item_count=len(items),
        group_count=len(group_fold),
        fold_count=5,
        fold_group_counts=fold_group_counts,
        observed_cell_count=int(mask_array.sum()),
        unknown_cell_count=int(mask_array.size - mask_array.sum()),
        fold_ce6_component_mass=tuple(fold_ce6_mass),
        fold_role_positive_mass=tuple(fold_role_mass),
        group_fold_roster_sha256=_canonical_sha256(group_fold_roster),
        source_plan_sha256=_canonical_sha256(run_plan),
    )


__all__ = [
    "AUXILIARY_MORPHOLOGY_ROLES",
    "MORPHOLOGY_RECOVERY_PREFLIGHT_SCHEMA",
    "MORPHOLOGY_RECOVERY_PROTOCOL_SCHEMA",
    "MORPHOLOGY_RECOVERY_PROTOCOL_SHA256",
    "HierarchicalMorphologyEvidenceHead",
    "HierarchicalMorphologyOutput",
    "HierarchicalMorphologyWeights",
    "MorphologyRecoveryPreflightReceipt",
    "audit_morphology_recovery_source",
    "derive_morphology_role_targets",
    "fit_hierarchical_morphology_weights",
    "hierarchical_morphology_group_balanced_loss",
    "load_morphology_recovery_preflight",
    "morphology_recovery_protocol_payload",
]
