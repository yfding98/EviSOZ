"""Formal native-CE6 morphology training, evaluation, and typed routing.

The optimizer sees only frozen LaBraM interval-group tokens and sparse native
TUEV edge/slot targets.  It never receives SOZ labels, private data, TUSZ
ictal targets, patient laterality, paths, or source identifiers as features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data.tuev_morphology import (
    EVAL_GROUP_KIND,
    FOLD_COUNT_SEMANTICS,
    TRAIN_GROUP_KIND,
    TUEVMorphologyIntervalGroup,
    TUEVMorphologyManifest,
)
from .geometry import (
    MORPHOLOGY_CLASSES,
    N_STANDARD_CHANNELS,
    N_TCP_EDGES,
    unsigned_incidence_matrix,
)
from .models.concept_heads import MorphologyEvidenceHead
from .morphology_token_io import (
    MORPHOLOGY_TRAINING_TOKEN_SHAPE,
    MorphologyTrainingTokenBinding,
    VerifiedMorphologyTrainingTokenCorpus,
    load_morphology_training_group_tokens,
    select_morphology_fold_bindings,
)


MORPHOLOGY_TRAINING_CONFIG_SCHEMA = "soz_morphology_training_config_v1"
MORPHOLOGY_TRAINING_RUN_SCHEMA = "soz_morphology_training_run_v1"
MORPHOLOGY_EVALUATION_SCHEMA = "soz_morphology_native_evaluation_v2"

MORPHOLOGY_TYPED_ROUTING_POLICY = {
    "schema_version": "soz_morphology_typed_routing_v1",
    "localizing_codes": ["SPSW", "PLED"],
    "generalized_conflict_codes": ["GPED"],
    "quality_abstention_codes": ["EYEM", "ARTF"],
    "support_ood_codes": ["BCKG"],
    "gped_electrode_identity": "discarded_before_reasoning",
    "positive_localizing_path": "spsw_pled_only",
    "nonpositive_ports": ["GPED", "EYEM", "ARTF", "BCKG"],
}


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


MORPHOLOGY_TYPED_ROUTING_POLICY_SHA256 = _canonical_sha256(
    MORPHOLOGY_TYPED_ROUTING_POLICY
)


def _sha(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _roster_sha256(values: Sequence[str]) -> str:
    roster = tuple(sorted(values))
    if len(set(roster)) != len(roster):
        raise ValueError("Roster values must be duplicate-free")
    return _canonical_sha256(roster)


@dataclass(frozen=True)
class MorphologyCropExample:
    crop_id: str
    record_id: str
    parent_group_id: str
    official_split: str
    group_kind: str
    binding: MorphologyTrainingTokenBinding
    labels: torch.Tensor
    source_target_mask: torch.Tensor
    overlap_component_weights: torch.Tensor
    cached_tokens: torch.Tensor | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.binding.crop_id != self.crop_id:
            raise ValueError("Token binding was swapped across morphology crops")
        expected = (N_TCP_EDGES, 4)
        if tuple(self.labels.shape) != expected or tuple(self.source_target_mask.shape) != expected:
            raise ValueError("Morphology source labels/mask must have shape [20,4]")
        if tuple(self.overlap_component_weights.shape) != expected:
            raise ValueError("Morphology overlap weights must have shape [20,4]")
        if self.labels.dtype != torch.long or self.source_target_mask.dtype != torch.bool:
            raise TypeError("Morphology labels must be long and mask must be bool")
        if not self.overlap_component_weights.is_floating_point() or not torch.isfinite(
            self.overlap_component_weights
        ).all():
            raise ValueError("Morphology overlap weights must be finite floating point")
        if self.source_target_mask[:, 1:].any():
            raise ValueError("TUEV morphology supervision may read slot 0 only")
        if torch.any(self.overlap_component_weights[~self.source_target_mask] != 0):
            raise ValueError("Unknown morphology cells must have zero component weight")
        observed_weights = self.overlap_component_weights[self.source_target_mask]
        if observed_weights.numel() < 1 or torch.any(observed_weights <= 0):
            raise ValueError("Observed morphology targets require positive component weights")
        observed_labels = self.labels[self.source_target_mask]
        if torch.any((observed_labels < 0) | (observed_labels >= len(MORPHOLOGY_CLASSES))):
            raise ValueError("Observed morphology labels must be native CE6")
        if self.official_split == "train":
            if self.group_kind != TRAIN_GROUP_KIND:
                raise ValueError("TUEV train examples require verified-subject grouping")
        elif self.official_split == "eval":
            if self.group_kind != EVAL_GROUP_KIND:
                raise ValueError("TUEV eval examples require official-session grouping")
        else:
            raise ValueError("official_split must be train or eval")
        if self.cached_tokens is not None:
            if (
                not isinstance(self.cached_tokens, torch.Tensor)
                or tuple(self.cached_tokens.shape) != MORPHOLOGY_TRAINING_TOKEN_SHAPE
                or self.cached_tokens.dtype != torch.float32
                or self.cached_tokens.device.type != "cpu"
                or self.cached_tokens.requires_grad
                or not torch.isfinite(self.cached_tokens).all()
            ):
                raise ValueError(
                    "Cached morphology tokens must be detached finite CPU "
                    "float32 [19,4,200]"
                )

    def load_tokens(self) -> torch.Tensor:
        if self.cached_tokens is not None:
            return self.cached_tokens
        loaded = load_morphology_training_group_tokens(
            self.binding.bundle_path,
            expected_manifest_sha256=self.binding.bundle_manifest_sha256,
        )
        if loaded.crop_id != self.crop_id or loaded.tensor_sha256 != self.binding.tensor_sha256:
            raise ValueError("Morphology token binding changed after dataset construction")
        return loaded.tokens


@dataclass(frozen=True)
class MorphologyGroupBag:
    parent_group_id: str
    official_split: str
    group_kind: str
    crops: tuple[MorphologyCropExample, ...]

    def __post_init__(self) -> None:
        if not self.crops:
            raise ValueError("Morphology group bags cannot be empty")
        crop_ids = tuple(crop.crop_id for crop in self.crops)
        if crop_ids != tuple(sorted(set(crop_ids))):
            raise ValueError("Morphology group-bag crops must be unique and sorted")
        if any(
            crop.parent_group_id != self.parent_group_id
            or crop.official_split != self.official_split
            or crop.group_kind != self.group_kind
            for crop in self.crops
        ):
            raise ValueError("A morphology bag mixes parent/split/group semantics")


@dataclass(frozen=True)
class MorphologyBagDataset(Sequence[MorphologyGroupBag]):
    role: str
    fold_manifest_sha256: str
    master_manifest_sha256: str
    master_token_corpus_index_sha256: str
    foundation_feature_receipt_sha256: str
    groups: tuple[MorphologyGroupBag, ...]

    def __post_init__(self) -> None:
        if self.role not in {"fit", "held"}:
            raise ValueError("Morphology dataset role must be fit or held")
        for field in (
            "fold_manifest_sha256",
            "master_manifest_sha256",
            "master_token_corpus_index_sha256",
            "foundation_feature_receipt_sha256",
        ):
            _sha(getattr(self, field), field=field)
        if not self.groups:
            raise ValueError("Morphology dataset cannot be empty")
        group_ids = tuple(group.parent_group_id for group in self.groups)
        if group_ids != tuple(sorted(set(group_ids))):
            raise ValueError("Morphology dataset group bags must be unique and sorted")
        if self.role == "fit" and any(
            group.official_split != "train" or group.group_kind != TRAIN_GROUP_KIND
            for group in self.groups
        ):
            raise ValueError("Morphology fitting may use verified TUEV train subjects only")

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> MorphologyGroupBag:
        return self.groups[index]

    def __iter__(self) -> Iterator[MorphologyGroupBag]:
        return iter(self.groups)

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(group.parent_group_id for group in self.groups)

    @property
    def crop_count(self) -> int:
        return sum(len(group.crops) for group in self.groups)

    @property
    def target_count(self) -> int:
        return sum(
            int(crop.source_target_mask.sum().item())
            for group in self.groups
            for crop in group.crops
        )


def _target_tensors(
    interval_group: TUEVMorphologyIntervalGroup,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.zeros((N_TCP_EDGES, 4), dtype=torch.long)
    mask = torch.zeros((N_TCP_EDGES, 4), dtype=torch.bool)
    weights = torch.zeros((N_TCP_EDGES, 4), dtype=torch.float32)
    for target in interval_group.targets:
        labels[target.edge_index, 0] = target.label_index
        mask[target.edge_index, 0] = True
        weights[target.edge_index, 0] = target.component_weight
    return labels, mask, weights


def morphology_target_bearing_group_ids(
    fold_manifest: TUEVMorphologyManifest,
    *,
    role: str,
) -> tuple[str, ...]:
    """Return the role roster that actually has native CE6 targets.

    The authorization manifest assigns every signal-eligible parent group to
    fit or held, including records with no surviving annotated interval after
    warm-up and signal-window rules.  Such targetless groups are part of the
    availability denominator but cannot contribute an optimizer step or an
    evaluation observation.  Keeping the two denominators explicit prevents a
    targetless subject/session from being fabricated as background.
    """

    if not isinstance(fold_manifest, TUEVMorphologyManifest):
        raise TypeError("fold_manifest must be TUEVMorphologyManifest")
    if fold_manifest.count_semantics != FOLD_COUNT_SEMANTICS:
        raise ValueError("Target-bearing rosters require a fold-specific manifest")
    if role == "fit":
        authorized = set(fold_manifest.fit_group_ids)
    elif role == "held":
        authorized = set(fold_manifest.held_group_ids)
    else:
        raise ValueError("role must be fit or held")
    return tuple(
        sorted(
            {
                group.parent_group_id
                for group in fold_manifest.interval_groups
                if group.parent_group_id in authorized and group.targets
            }
        )
    )


def build_morphology_bag_dataset(
    fold_manifest: TUEVMorphologyManifest,
    master_corpus: VerifiedMorphologyTrainingTokenCorpus,
    *,
    role: str,
    preload_tokens: bool = False,
) -> MorphologyBagDataset:
    """Join labels to target-free master tokens under strict crop/group receipts."""

    if not isinstance(preload_tokens, bool):
        raise TypeError("preload_tokens must be bool")
    if not isinstance(fold_manifest, TUEVMorphologyManifest):
        raise TypeError("fold_manifest must be TUEVMorphologyManifest")
    if fold_manifest.count_semantics != FOLD_COUNT_SEMANTICS:
        raise ValueError("Morphology datasets require a fold-specific final manifest")
    bindings = select_morphology_fold_bindings(master_corpus, fold_manifest, role=role)
    binding_by_crop = {binding.crop_id: binding for binding in bindings}
    selected_group_ids = set(
        fold_manifest.fit_group_ids if role == "fit" else fold_manifest.held_group_ids
    )
    records = {record.record_id: record for record in fold_manifest.records}
    by_parent: dict[str, list[MorphologyCropExample]] = {}
    group_semantics: dict[str, tuple[str, str]] = {}
    for interval_group in fold_manifest.interval_groups:
        if interval_group.parent_group_id not in selected_group_ids:
            continue
        binding = binding_by_crop.get(interval_group.crop_id)
        if binding is None:
            raise ValueError("Fold interval group lacks its strict master-token binding")
        record = records[interval_group.record_id]
        labels, mask, weights = _target_tensors(interval_group)
        cached_tokens = None
        if preload_tokens:
            loaded = load_morphology_training_group_tokens(
                binding.bundle_path,
                expected_manifest_sha256=binding.bundle_manifest_sha256,
            )
            if (
                loaded.crop_id != interval_group.crop_id
                or loaded.tensor_sha256 != binding.tensor_sha256
            ):
                raise ValueError(
                    "Morphology token binding changed during in-memory preload"
                )
            cached_tokens = loaded.tokens
        example = MorphologyCropExample(
            crop_id=interval_group.crop_id,
            record_id=interval_group.record_id,
            parent_group_id=interval_group.parent_group_id,
            official_split=record.official_split,
            group_kind=record.group_kind,
            binding=binding,
            labels=labels,
            source_target_mask=mask,
            overlap_component_weights=weights,
            cached_tokens=cached_tokens,
        )
        by_parent.setdefault(interval_group.parent_group_id, []).append(example)
        previous = group_semantics.setdefault(
            interval_group.parent_group_id,
            (record.official_split, record.group_kind),
        )
        if previous != (record.official_split, record.group_kind):
            raise ValueError("One TUEV parent group has contradictory split semantics")
    bags = tuple(
        MorphologyGroupBag(
            parent_group_id=group_id,
            official_split=group_semantics[group_id][0],
            group_kind=group_semantics[group_id][1],
            crops=tuple(sorted(crops, key=lambda item: item.crop_id)),
        )
        for group_id, crops in sorted(by_parent.items())
    )
    expected_group_ids = morphology_target_bearing_group_ids(
        fold_manifest, role=role
    )
    if tuple(bag.parent_group_id for bag in bags) != expected_group_ids:
        raise ValueError(
            "Morphology dataset differs from the authorized target-bearing roster"
        )
    return MorphologyBagDataset(
        role=role,
        fold_manifest_sha256=fold_manifest.manifest_sha256,
        master_manifest_sha256=master_corpus.source_morphology_manifest_sha256,
        master_token_corpus_index_sha256=master_corpus.index_sha256,
        foundation_feature_receipt_sha256=(
            master_corpus.foundation_feature_receipt_sha256
        ),
        groups=bags,
    )


def morphology_component_group_balanced_ce6_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    source_target_mask: torch.Tensor,
    overlap_component_weights: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean of group-normalised, overlap-component-weighted native CE6 loss."""

    if logits.ndim != 4 or logits.shape[1] != N_TCP_EDGES or logits.shape[-1] != 6:
        raise ValueError("Morphology logits must have shape [B,20,T,6]")
    expected = tuple(logits.shape[:-1])
    if tuple(labels.shape) != expected or tuple(source_target_mask.shape) != expected:
        raise ValueError("Morphology labels/mask must have shape [B,20,T]")
    if tuple(overlap_component_weights.shape) != expected:
        raise ValueError("Morphology overlap weights must match [B,20,T]")
    if labels.dtype != torch.long or source_target_mask.dtype != torch.bool:
        raise TypeError("Morphology labels must be long and mask must be bool")
    if not overlap_component_weights.is_floating_point() or not torch.isfinite(
        overlap_component_weights
    ).all():
        raise ValueError("Morphology overlap weights must be finite floating point")
    if source_target_mask.shape[-1] == 4 and source_target_mask[:, :, 1:].any():
        raise ValueError("Morphology source loss may read slot 0 only")
    if torch.any(overlap_component_weights[~source_target_mask] != 0):
        raise ValueError("Unknown morphology cells must carry zero component weight")
    observed_weights = overlap_component_weights[source_target_mask]
    if observed_weights.numel() < 1 or torch.any(observed_weights <= 0):
        raise ValueError("Observed morphology targets require positive component weights")
    if group_ids.ndim != 1 or group_ids.shape[0] != logits.shape[0] or group_ids.dtype != torch.long:
        raise ValueError("group_ids must be long [B]")
    if group_ids.device != logits.device:
        raise ValueError("group_ids and logits must share one device")
    safe_labels = torch.where(source_target_mask, labels, 0)
    element = F.cross_entropy(
        logits.movedim(-1, 1),
        safe_labels,
        weight=class_weights,
        reduction="none",
    )
    group_losses: list[torch.Tensor] = []
    for group_id in torch.unique(group_ids, sorted=True):
        example_mask = group_ids == group_id
        observed = source_target_mask[example_mask]
        weights = overlap_component_weights[example_mask]
        denominator = weights[observed].sum()
        if observed.any() and denominator > 0:
            group_losses.append(
                (element[example_mask] * weights)[observed].sum() / denominator
            )
    if not group_losses:
        raise ValueError("No observed morphology targets are available")
    return torch.stack(group_losses).mean()


def fit_morphology_class_weights(
    dataset: MorphologyBagDataset,
    *,
    cap: float = 10.0,
) -> torch.Tensor:
    """Fit capped inverse prevalence on fit subjects with component weighting."""

    if not isinstance(dataset, MorphologyBagDataset) or dataset.role != "fit":
        raise ValueError("Morphology class weights must be fitted on fit subjects only")
    if not math.isfinite(float(cap)) or cap < 1:
        raise ValueError("Class-weight cap must be finite and >=1")
    group_prevalences: list[torch.Tensor] = []
    for bag in dataset:
        counts = torch.zeros(6, dtype=torch.float64)
        for crop in bag.crops:
            labels = crop.labels[crop.source_target_mask]
            weights = crop.overlap_component_weights[crop.source_target_mask].double()
            counts.scatter_add_(0, labels, weights)
        total = counts.sum()
        if total > 0:
            group_prevalences.append(counts / total)
    if not group_prevalences:
        raise ValueError("Fit dataset contains no morphology targets")
    prevalence = torch.stack(group_prevalences).mean(dim=0)
    if torch.any(prevalence <= 0):
        missing = [MORPHOLOGY_CLASSES[index] for index in torch.where(prevalence <= 0)[0].tolist()]
        raise ValueError(f"Fit subjects contain no support for CE6 classes: {missing}")
    weights = prevalence.reciprocal()
    weights /= weights.mean()
    weights = torch.clamp(weights, max=float(cap))
    weights /= weights.mean()
    return weights.float()


@dataclass(frozen=True)
class MorphologyTrainingConfig:
    fixed_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    crop_microbatch_size: int = 32
    gradient_clip_norm: float = 1.0
    class_weight_cap: float = 10.0
    seed: int = 20260808
    deterministic_algorithms: bool = True
    schema_version: str = MORPHOLOGY_TRAINING_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MORPHOLOGY_TRAINING_CONFIG_SCHEMA:
            raise ValueError("Unexpected morphology training config schema")
        for field in ("fixed_epochs", "crop_microbatch_size"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for field in (
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "class_weight_cap",
        ):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if not isinstance(self.deterministic_algorithms, bool):
            raise TypeError("deterministic_algorithms must be bool")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fixed_epochs": self.fixed_epochs,
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "crop_microbatch_size": self.crop_microbatch_size,
            "gradient_clip_norm": float(self.gradient_clip_norm),
            "class_weight_cap": float(self.class_weight_cap),
            "seed": self.seed,
            "deterministic_algorithms": self.deterministic_algorithms,
        }


@dataclass(frozen=True)
class MorphologyTrainingRunReceipt:
    config: MorphologyTrainingConfig
    fold_manifest_sha256: str
    master_manifest_sha256: str
    master_token_corpus_index_sha256: str
    foundation_feature_receipt_sha256: str
    routing_policy_sha256: str
    fit_group_ids: tuple[str, ...]
    held_group_ids: tuple[str, ...]
    fit_group_roster_sha256: str
    held_group_roster_sha256: str
    class_weights: tuple[float, ...]
    epoch_group_mean_losses: tuple[float, ...]
    fit_crop_count: int
    fit_target_count: int
    schema_version: str = MORPHOLOGY_TRAINING_RUN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MORPHOLOGY_TRAINING_RUN_SCHEMA:
            raise ValueError("Unexpected morphology training run schema")
        for field in (
            "fold_manifest_sha256",
            "master_manifest_sha256",
            "master_token_corpus_index_sha256",
            "foundation_feature_receipt_sha256",
            "routing_policy_sha256",
            "fit_group_roster_sha256",
            "held_group_roster_sha256",
        ):
            _sha(getattr(self, field), field=field)
        if self.routing_policy_sha256 != MORPHOLOGY_TYPED_ROUTING_POLICY_SHA256:
            raise ValueError("Morphology typed-routing policy hash drifted")
        if set(self.fit_group_ids) & set(self.held_group_ids):
            raise ValueError("Morphology fit and held group rosters overlap")
        if self.fit_group_ids != tuple(sorted(set(self.fit_group_ids))) or self.held_group_ids != tuple(sorted(set(self.held_group_ids))):
            raise ValueError("Morphology group rosters must be sorted and unique")
        if self.fit_group_roster_sha256 != _roster_sha256(self.fit_group_ids) or self.held_group_roster_sha256 != _roster_sha256(self.held_group_ids):
            raise ValueError("Morphology group-roster SHA mismatch")
        if len(self.class_weights) != 6 or any(not math.isfinite(value) or value <= 0 for value in self.class_weights):
            raise ValueError("Morphology run requires six finite positive class weights")
        if len(self.epoch_group_mean_losses) != self.config.fixed_epochs or any(
            not math.isfinite(value) or value < 0 for value in self.epoch_group_mean_losses
        ):
            raise ValueError("Morphology epoch-loss history is incomplete")
        if self.fit_crop_count < 1 or self.fit_target_count < 1:
            raise ValueError("Morphology run counts must be positive")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.canonical_payload,
            "fold_manifest_sha256": self.fold_manifest_sha256,
            "master_manifest_sha256": self.master_manifest_sha256,
            "master_token_corpus_index_sha256": self.master_token_corpus_index_sha256,
            "foundation_feature_receipt_sha256": self.foundation_feature_receipt_sha256,
            "routing_policy_sha256": self.routing_policy_sha256,
            "fit_group_ids": list(self.fit_group_ids),
            "held_group_ids": list(self.held_group_ids),
            "fit_group_roster_sha256": self.fit_group_roster_sha256,
            "held_group_roster_sha256": self.held_group_roster_sha256,
            "class_weights": list(self.class_weights),
            "epoch_group_mean_losses": list(self.epoch_group_mean_losses),
            "fit_crop_count": self.fit_crop_count,
            "fit_target_count": self.fit_target_count,
        }

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload)


def _stack_crops(
    crops: Sequence[MorphologyCropExample], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.stack([crop.load_tokens() for crop in crops]).to(device)
    labels = torch.stack([crop.labels for crop in crops]).to(device)
    masks = torch.stack([crop.source_target_mask for crop in crops]).to(device)
    weights = torch.stack([crop.overlap_component_weights for crop in crops]).to(device)
    if tuple(tokens.shape[1:]) != MORPHOLOGY_TRAINING_TOKEN_SHAPE:
        raise RuntimeError("Morphology master token corpus changed shape")
    return tokens, labels, masks, weights


def train_fixed_epoch_morphology_head(
    head: MorphologyEvidenceHead,
    fit_dataset: MorphologyBagDataset,
    fold_manifest: TUEVMorphologyManifest,
    *,
    config: MorphologyTrainingConfig = MorphologyTrainingConfig(),
    device: torch.device | str = "cpu",
) -> MorphologyTrainingRunReceipt:
    """Fit only the CE6 head with one equal-contribution step per train subject."""

    if not isinstance(head, MorphologyEvidenceHead):
        raise TypeError("head must be MorphologyEvidenceHead")
    if not isinstance(fit_dataset, MorphologyBagDataset) or fit_dataset.role != "fit":
        raise ValueError("Morphology optimizer requires a fit-role dataset")
    if not isinstance(fold_manifest, TUEVMorphologyManifest) or fold_manifest.count_semantics != FOLD_COUNT_SEMANTICS:
        raise ValueError("Morphology optimizer requires a fold-specific manifest")
    if fit_dataset.fold_manifest_sha256 != fold_manifest.manifest_sha256:
        raise ValueError("Morphology dataset is bound to another fold manifest")
    expected_fit_groups = morphology_target_bearing_group_ids(
        fold_manifest, role="fit"
    )
    expected_held_groups = morphology_target_bearing_group_ids(
        fold_manifest, role="held"
    )
    if tuple(fit_dataset.group_ids) != expected_fit_groups:
        raise ValueError(
            "Morphology fit dataset does not match the signed target-bearing roster"
        )
    if set(fold_manifest.fit_group_ids) & set(fold_manifest.held_group_ids):
        raise ValueError("Morphology fit/held firewall failed")
    execution_device = torch.device(device)
    if execution_device.type not in {"cpu", "cuda"}:
        raise ValueError("Morphology training device must be cpu or cuda")
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    head.to(execution_device)
    class_weights = fit_morphology_class_weights(
        fit_dataset, cap=config.class_weight_cap
    ).to(execution_device)
    parameters = tuple(parameter for parameter in head.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("Morphology head has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_parameters != {id(parameter) for parameter in parameters}:
        raise RuntimeError("Optimizer parameters differ from the morphology head")

    previous_determinism = torch.are_deterministic_algorithms_enabled()
    epoch_losses: list[float] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    try:
        torch.use_deterministic_algorithms(config.deterministic_algorithms)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        for _epoch in range(config.fixed_epochs):
            order = torch.randperm(len(fit_dataset), generator=generator).tolist()
            group_losses: list[float] = []
            head.train()
            for bag_index in order:
                bag = fit_dataset[bag_index]
                optimizer.zero_grad(set_to_none=True)
                denominator = sum(
                    float(
                        crop.overlap_component_weights[
                            crop.source_target_mask
                        ].sum()
                    )
                    for crop in bag.crops
                )
                if denominator <= 0:
                    raise RuntimeError("Morphology group has no effective target weight")
                group_loss_value = 0.0
                for start in range(0, len(bag.crops), config.crop_microbatch_size):
                    crops = bag.crops[start : start + config.crop_microbatch_size]
                    tokens, labels, masks, weights = _stack_crops(crops, execution_device)
                    logits = head(tokens)
                    safe_labels = torch.where(masks, labels, 0)
                    element = F.cross_entropy(
                        logits.movedim(-1, 1),
                        safe_labels,
                        weight=class_weights,
                        reduction="none",
                    )
                    numerator = (element * weights)[masks].sum()
                    scaled = numerator / denominator
                    scaled.backward()
                    group_loss_value += float(scaled.detach().cpu())
                torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
                optimizer.step()
                group_losses.append(group_loss_value)
            epoch_losses.append(sum(group_losses) / len(group_losses))
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
    return MorphologyTrainingRunReceipt(
        config=config,
        fold_manifest_sha256=fold_manifest.manifest_sha256,
        master_manifest_sha256=fit_dataset.master_manifest_sha256,
        master_token_corpus_index_sha256=fit_dataset.master_token_corpus_index_sha256,
        foundation_feature_receipt_sha256=fit_dataset.foundation_feature_receipt_sha256,
        routing_policy_sha256=MORPHOLOGY_TYPED_ROUTING_POLICY_SHA256,
        fit_group_ids=expected_fit_groups,
        held_group_ids=expected_held_groups,
        fit_group_roster_sha256=_roster_sha256(expected_fit_groups),
        held_group_roster_sha256=_roster_sha256(expected_held_groups),
        class_weights=tuple(float(value) for value in class_weights.detach().cpu()),
        epoch_group_mean_losses=tuple(epoch_losses),
        fit_crop_count=fit_dataset.crop_count,
        fit_target_count=fit_dataset.target_count,
    )


def _weighted_average_precision(
    scores: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> float | None:
    positives = weights[targets].sum()
    if positives <= 0:
        return None
    order = torch.argsort(scores, descending=True, stable=True)
    ordered_targets = targets[order]
    ordered_weights = weights[order]
    true_mass = torch.cumsum(ordered_weights * ordered_targets.float(), dim=0)
    total_mass = torch.cumsum(ordered_weights, dim=0)
    precision = true_mass / total_mass.clamp_min(torch.finfo(torch.float64).eps)
    increments = ordered_weights * ordered_targets.float() / positives
    return float((precision * increments).sum())


@dataclass(frozen=True)
class MorphologyEvaluationReceipt:
    dataset_role: str
    fold_manifest_sha256: str
    master_token_corpus_index_sha256: str
    checkpoint_or_run_sha256: str
    group_ids: tuple[str, ...]
    group_roster_sha256: str
    group_kind_counts: tuple[tuple[str, int], ...]
    target_count: int
    weighted_nll: float
    weighted_brier: float
    weighted_ece: float
    group_macro_balanced_accuracy: float
    class_metrics: tuple[
        tuple[str, float, float, float, float, float | None], ...
    ]
    schema_version: str = MORPHOLOGY_EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MORPHOLOGY_EVALUATION_SCHEMA:
            raise ValueError("Unexpected morphology evaluation schema")
        if self.dataset_role != "held":
            raise ValueError("Primary morphology evaluation requires held groups")
        for field in (
            "fold_manifest_sha256",
            "master_token_corpus_index_sha256",
            "checkpoint_or_run_sha256",
            "group_roster_sha256",
        ):
            _sha(getattr(self, field), field=field)
        if self.group_roster_sha256 != _roster_sha256(self.group_ids):
            raise ValueError("Morphology evaluation group-roster SHA mismatch")
        if self.target_count < 1:
            raise ValueError("Morphology evaluation requires observed native targets")
        finite_metrics = (
            self.weighted_nll,
            self.weighted_brier,
            self.weighted_ece,
            self.group_macro_balanced_accuracy,
        )
        if any(not math.isfinite(value) for value in finite_metrics):
            raise ValueError("Morphology primary evaluation metrics must be finite")
        if tuple(row[0] for row in self.class_metrics) != MORPHOLOGY_CLASSES:
            raise ValueError("Morphology class metrics must follow native CE6 order")
        for row in self.class_metrics:
            if len(row) != 6:
                raise ValueError(
                    "Morphology class metrics must be "
                    "[class,support,precision,recall,f1,average_precision]"
                )
            _, support, precision, recall, f1, average_precision = row
            if not math.isfinite(support) or support < 0:
                raise ValueError("Morphology class support must be finite and non-negative")
            if any(
                not math.isfinite(value) or not 0 <= value <= 1
                for value in (precision, recall, f1)
            ):
                raise ValueError("Morphology class precision/recall/F1 must lie in [0,1]")
            if support == 0:
                if average_precision is not None:
                    raise ValueError("Unsupported morphology classes require AP=null")
            elif average_precision is None or not math.isfinite(average_precision) or not 0 <= average_precision <= 1:
                raise ValueError("Supported morphology classes require finite AP in [0,1]")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_role": self.dataset_role,
            "fold_manifest_sha256": self.fold_manifest_sha256,
            "master_token_corpus_index_sha256": self.master_token_corpus_index_sha256,
            "checkpoint_or_run_sha256": self.checkpoint_or_run_sha256,
            "group_ids": list(self.group_ids),
            "group_roster_sha256": self.group_roster_sha256,
            "group_kind_counts": [list(item) for item in self.group_kind_counts],
            "target_count": self.target_count,
            "weighted_nll": self.weighted_nll,
            "weighted_brier": self.weighted_brier,
            "weighted_ece": self.weighted_ece,
            "group_macro_balanced_accuracy": self.group_macro_balanced_accuracy,
            "class_metrics": [list(item) for item in self.class_metrics],
        }

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload)


def evaluate_morphology_groups(
    head: MorphologyEvidenceHead,
    dataset: MorphologyBagDataset,
    *,
    checkpoint_or_run_sha256: str,
    device: torch.device | str = "cpu",
    crop_microbatch_size: int = 32,
    ece_bins: int = 15,
) -> MorphologyEvaluationReceipt:
    """Report verified-subject or official-eval-session macro native metrics."""

    if not isinstance(head, MorphologyEvidenceHead):
        raise TypeError("head must be MorphologyEvidenceHead")
    if not isinstance(dataset, MorphologyBagDataset) or dataset.role != "held":
        raise ValueError("Morphology evaluation requires a held-role dataset")
    _sha(checkpoint_or_run_sha256, field="checkpoint_or_run_sha256")
    if crop_microbatch_size < 1 or ece_bins < 2:
        raise ValueError("Evaluation microbatch/ECE settings are invalid")
    execution_device = torch.device(device)
    head.to(execution_device)
    head.eval()
    all_probabilities: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_weights: list[torch.Tensor] = []
    group_balanced: list[float] = []
    with torch.no_grad():
        for bag in dataset:
            bag_probabilities: list[torch.Tensor] = []
            bag_labels: list[torch.Tensor] = []
            bag_weights: list[torch.Tensor] = []
            for start in range(0, len(bag.crops), crop_microbatch_size):
                crops = bag.crops[start : start + crop_microbatch_size]
                tokens, labels, masks, weights = _stack_crops(crops, execution_device)
                probabilities = head(tokens).softmax(dim=-1)
                bag_probabilities.append(probabilities[masks].double().cpu())
                bag_labels.append(labels[masks].cpu())
                bag_weights.append(weights[masks].double().cpu())
            probabilities = torch.cat(bag_probabilities)
            labels = torch.cat(bag_labels)
            weights = torch.cat(bag_weights)
            predictions = probabilities.argmax(dim=-1)
            recalls: list[float] = []
            for class_index in range(6):
                truth = labels == class_index
                if truth.any():
                    recalls.append(
                        float(weights[truth & (predictions == class_index)].sum() / weights[truth].sum())
                    )
            if recalls:
                group_balanced.append(sum(recalls) / len(recalls))
            all_probabilities.append(probabilities)
            all_labels.append(labels)
            all_weights.append(weights)
    probabilities = torch.cat(all_probabilities)
    labels = torch.cat(all_labels)
    weights = torch.cat(all_weights)
    total_weight = weights.sum()
    true_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
    nll = float((weights * -true_probability.clamp_min(1e-12).log()).sum() / total_weight)
    one_hot = F.one_hot(labels, num_classes=6).double()
    brier = float((weights * ((probabilities - one_hot) ** 2).sum(dim=-1)).sum() / total_weight)
    confidence, predictions = probabilities.max(dim=-1)
    correct = predictions == labels
    ece = 0.0
    boundaries = torch.linspace(0.0, 1.0, ece_bins + 1, dtype=torch.float64)
    for index in range(ece_bins):
        member = (confidence >= boundaries[index]) & (
            confidence <= boundaries[index + 1]
            if index == ece_bins - 1
            else confidence < boundaries[index + 1]
        )
        if member.any():
            mass = weights[member].sum()
            accuracy = (weights[member] * correct[member].double()).sum() / mass
            mean_confidence = (weights[member] * confidence[member]).sum() / mass
            ece += float(mass / total_weight * (accuracy - mean_confidence).abs())
    class_metrics: list[
        tuple[str, float, float, float, float, float | None]
    ] = []
    for class_index, class_name in enumerate(MORPHOLOGY_CLASSES):
        truth = labels == class_index
        predicted = predictions == class_index
        tp = weights[truth & predicted].sum()
        fp = weights[~truth & predicted].sum()
        fn = weights[truth & ~predicted].sum()
        precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
        recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        average_precision = _weighted_average_precision(
            probabilities[:, class_index], truth, weights
        )
        support = float(weights[truth].sum())
        class_metrics.append(
            (class_name, support, precision, recall, f1, average_precision)
        )
    kind_counts: dict[str, int] = {}
    for group in dataset:
        kind_counts[group.group_kind] = kind_counts.get(group.group_kind, 0) + 1
    return MorphologyEvaluationReceipt(
        dataset_role=dataset.role,
        fold_manifest_sha256=dataset.fold_manifest_sha256,
        master_token_corpus_index_sha256=dataset.master_token_corpus_index_sha256,
        checkpoint_or_run_sha256=checkpoint_or_run_sha256,
        group_ids=dataset.group_ids,
        group_roster_sha256=_roster_sha256(dataset.group_ids),
        group_kind_counts=tuple(sorted(kind_counts.items())),
        target_count=dataset.target_count,
        weighted_nll=nll,
        weighted_brier=brier,
        weighted_ece=ece,
        group_macro_balanced_accuracy=sum(group_balanced) / len(group_balanced),
        class_metrics=tuple(class_metrics),
    )


@dataclass(frozen=True)
class TypedMorphologyPorts:
    spsw_localizing: torch.Tensor
    pled_localizing: torch.Tensor
    generalized_conflict: torch.Tensor
    quality_abstention: torch.Tensor
    support_ood: torch.Tensor
    edge_mask: torch.Tensor

    def __post_init__(self) -> None:
        edge_shape = tuple(self.spsw_localizing.shape)
        if len(edge_shape) != 3 or edge_shape[1] != N_TCP_EDGES:
            raise ValueError("Spatial morphology ports must have shape [B,20,T]")
        for tensor in (
            self.pled_localizing,
            self.quality_abstention,
            self.support_ood,
            self.edge_mask,
        ):
            if tuple(tensor.shape) != edge_shape:
                raise ValueError("Typed morphology edge ports have inconsistent shapes")
        if tuple(self.generalized_conflict.shape) != (edge_shape[0], edge_shape[2]):
            raise ValueError(
                "GPED generalized conflict must be [B,T] with no electrode identity"
            )
        if self.edge_mask.dtype != torch.bool:
            raise TypeError("Typed morphology edge mask must be bool")
        for tensor in (
            self.spsw_localizing,
            self.pled_localizing,
            self.generalized_conflict,
            self.quality_abstention,
            self.support_ood,
        ):
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError("Typed morphology ports must be finite floating point")
            if torch.any((tensor < 0) | (tensor > 1)):
                raise ValueError("Typed morphology port values must lie in [0,1]")


def route_ce6_to_typed_ports(
    probabilities: torch.Tensor,
    edge_mask: torch.Tensor,
) -> TypedMorphologyPorts:
    """Discard GPED electrode identity and expose four semantically typed ports."""

    if probabilities.ndim != 4 or probabilities.shape[1] != N_TCP_EDGES or probabilities.shape[-1] != 6:
        raise ValueError("CE6 probabilities must have shape [B,20,T,6]")
    if tuple(edge_mask.shape) != tuple(probabilities.shape[:-1]) or edge_mask.dtype != torch.bool:
        raise ValueError("edge_mask must be bool [B,20,T]")
    if not probabilities.is_floating_point() or not torch.isfinite(probabilities).all():
        raise ValueError("CE6 probabilities must be finite floating point")
    if torch.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("CE6 probabilities must lie in [0,1]")
    if not torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones_like(probabilities[..., 0]),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("CE6 probabilities must sum to one")
    mask_float = edge_mask.to(probabilities.dtype)
    counts = mask_float.sum(dim=1)
    generalized = (
        probabilities[..., 1] * mask_float
    ).sum(dim=1) / counts.clamp_min(1.0)
    generalized = torch.where(counts > 0, generalized, torch.zeros_like(generalized))
    zero = torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)
    return TypedMorphologyPorts(
        spsw_localizing=torch.where(edge_mask, probabilities[..., 0], zero),
        pled_localizing=torch.where(edge_mask, probabilities[..., 2], zero),
        generalized_conflict=generalized,
        quality_abstention=torch.where(
            edge_mask, probabilities[..., 3] + probabilities[..., 4], zero
        ),
        support_ood=torch.where(edge_mask, probabilities[..., 5], zero),
        edge_mask=edge_mask,
    )


@dataclass(frozen=True)
class TypedMorphologyRoutingOutput:
    edge_localizing_score: torch.Tensor
    channel_localizing_score: torch.Tensor
    generalized_conflict: torch.Tensor
    quality_abstention: torch.Tensor
    support_ood: torch.Tensor
    edge_mask: torch.Tensor


@dataclass(frozen=True)
class MorphologyRegionAggregation:
    """Deterministic caller-defined aggregation; no region ontology is inferred."""

    scores: torch.Tensor
    available_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.scores.ndim != 3 or tuple(self.available_mask.shape) != tuple(
            self.scores.shape
        ):
            raise ValueError("Morphology region scores/mask must have shape [B,R,T]")
        if not self.scores.is_floating_point() or not torch.isfinite(self.scores).all():
            raise ValueError("Morphology region scores must be finite floating point")
        if torch.any(self.scores < 0):
            raise ValueError("Morphology region scores cannot be negative")
        if self.available_mask.dtype != torch.bool:
            raise TypeError("Morphology region availability mask must be bool")
        if torch.any(self.scores[~self.available_mask] != 0):
            raise ValueError("Unavailable morphology regions must have zero score")


def aggregate_nonnegative_morphology_regions(
    channel_scores: torch.Tensor,
    region_weights: torch.Tensor,
    *,
    channel_available_mask: torch.Tensor | None = None,
) -> MorphologyRegionAggregation:
    """Aggregate channel support with a fixed nonnegative region map.

    ``region_weights[R,19]`` is supplied by the caller and is never learned or
    interpreted here.  Negative or gradient-bearing weights are rejected, so
    a channel-wise non-increase under GPED/artifact interventions remains a
    region-wise non-increase.  This function does not create a clinical lobe
    ontology or a second region target.
    """

    if (
        not isinstance(channel_scores, torch.Tensor)
        or channel_scores.ndim != 3
        or channel_scores.shape[1] != N_STANDARD_CHANNELS
    ):
        raise ValueError("channel_scores must have shape [B,19,T]")
    if not channel_scores.is_floating_point() or not torch.isfinite(
        channel_scores
    ).all():
        raise ValueError("channel_scores must be finite floating point")
    if torch.any(channel_scores < 0):
        raise ValueError("channel_scores cannot be negative")
    if (
        not isinstance(region_weights, torch.Tensor)
        or region_weights.ndim != 2
        or region_weights.shape[1] != N_STANDARD_CHANNELS
        or region_weights.shape[0] < 1
    ):
        raise ValueError("region_weights must have shape [R,19] with R>=1")
    if region_weights.requires_grad or region_weights.grad_fn is not None:
        raise ValueError("region_weights must be fixed and detached")
    if not region_weights.is_floating_point() or not torch.isfinite(
        region_weights
    ).all():
        raise ValueError("region_weights must be finite floating point")
    if torch.any(region_weights < 0):
        raise ValueError("region_weights must be nonnegative")
    if torch.any(region_weights.sum(dim=1) <= 0):
        raise ValueError("Every caller-defined region must contain positive weight")
    if region_weights.device != channel_scores.device:
        raise ValueError("region_weights and channel_scores must share one device")
    if channel_available_mask is None:
        available = torch.ones_like(channel_scores, dtype=torch.bool)
    else:
        if (
            not isinstance(channel_available_mask, torch.Tensor)
            or tuple(channel_available_mask.shape) != tuple(channel_scores.shape)
            or channel_available_mask.dtype != torch.bool
        ):
            raise ValueError("channel_available_mask must be bool [B,19,T]")
        if channel_available_mask.device != channel_scores.device:
            raise ValueError("channel availability and scores must share one device")
        available = channel_available_mask
    weights = region_weights.to(dtype=channel_scores.dtype)
    observed_weights = (
        weights.unsqueeze(0).unsqueeze(-1)
        * available.unsqueeze(1).to(channel_scores.dtype)
    )
    numerator = (
        observed_weights * channel_scores.unsqueeze(1)
    ).sum(dim=2)
    denominator = observed_weights.sum(dim=2)
    region_available = denominator > 0
    scores = numerator / denominator.clamp_min(1.0)
    scores = torch.where(region_available, scores, torch.zeros_like(scores))
    return MorphologyRegionAggregation(
        scores=scores,
        available_mask=region_available,
    )


class TypedMorphologyRouter(nn.Module):
    """Signed router with no trainable positive path from nonlocalizing CE6 codes."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_spsw_weight = nn.Parameter(torch.tensor(0.0))
        self.raw_pled_weight = nn.Parameter(torch.tensor(0.0))
        self.raw_generalized_penalty = nn.Parameter(torch.tensor(0.0))
        self.raw_quality_penalty = nn.Parameter(torch.tensor(0.0))
        self.register_buffer(
            "incidence", unsigned_incidence_matrix(dtype=torch.float32), persistent=True
        )
        self.routing_policy_sha256 = MORPHOLOGY_TYPED_ROUTING_POLICY_SHA256

    def forward(self, ports: TypedMorphologyPorts) -> TypedMorphologyRoutingOutput:
        if not isinstance(ports, TypedMorphologyPorts):
            raise TypeError("TypedMorphologyRouter accepts typed ports, not raw CE6 concatenation")
        positive = (
            F.softplus(self.raw_spsw_weight) * ports.spsw_localizing
            + F.softplus(self.raw_pled_weight) * ports.pled_localizing
        )
        generalized_penalty = F.softplus(self.raw_generalized_penalty)
        quality_penalty = F.softplus(self.raw_quality_penalty)
        gate = torch.exp(
            -generalized_penalty * ports.generalized_conflict.unsqueeze(1)
            - quality_penalty * ports.quality_abstention
        )
        edge_score = torch.where(
            ports.edge_mask, positive * gate, torch.zeros_like(positive)
        )
        incidence = self.incidence.to(edge_score)
        channel_numerator = torch.einsum("ce,bet->bct", incidence, edge_score)
        edge_availability = ports.edge_mask.to(edge_score.dtype)
        channel_denominator = torch.einsum("ce,bet->bct", incidence, edge_availability)
        channel_score = channel_numerator / channel_denominator.clamp_min(1.0)
        channel_score = torch.where(
            channel_denominator > 0, channel_score, torch.zeros_like(channel_score)
        )
        # support_ood is returned for calibration/explanation only.  It does
        # not occur in edge_score or channel_score, so increasing BCKG cannot
        # increase a localization logit through this family.
        return TypedMorphologyRoutingOutput(
            edge_localizing_score=edge_score,
            channel_localizing_score=channel_score,
            generalized_conflict=ports.generalized_conflict,
            quality_abstention=ports.quality_abstention,
            support_ood=ports.support_ood,
            edge_mask=ports.edge_mask,
        )


__all__ = [
    "MORPHOLOGY_EVALUATION_SCHEMA",
    "MORPHOLOGY_TRAINING_CONFIG_SCHEMA",
    "MORPHOLOGY_TRAINING_RUN_SCHEMA",
    "MORPHOLOGY_TYPED_ROUTING_POLICY",
    "MORPHOLOGY_TYPED_ROUTING_POLICY_SHA256",
    "MorphologyBagDataset",
    "MorphologyCropExample",
    "MorphologyEvaluationReceipt",
    "MorphologyGroupBag",
    "MorphologyRegionAggregation",
    "MorphologyTrainingConfig",
    "MorphologyTrainingRunReceipt",
    "TypedMorphologyPorts",
    "TypedMorphologyRouter",
    "TypedMorphologyRoutingOutput",
    "aggregate_nonnegative_morphology_regions",
    "build_morphology_bag_dataset",
    "morphology_target_bearing_group_ids",
    "evaluate_morphology_groups",
    "fit_morphology_class_weights",
    "morphology_component_group_balanced_ce6_loss",
    "route_ce6_to_typed_ports",
    "train_fixed_epoch_morphology_head",
]
