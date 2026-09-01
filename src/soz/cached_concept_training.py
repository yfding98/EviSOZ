"""Patient-balanced ictal-head training from target-free LaBraM token caches."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable, Iterator, Sequence

import torch

from .concept_losses import ictal_involvement_loss
from .concept_metrics import IctalConceptMetrics, patient_macro_ictal_metrics
from .concept_token_io import LoadedLaBraMConceptTokens
from .concept_training import DEFAULT_EVENT_MICROBATCH_SIZE, IctalEpochOutput
from .geometry import N_TCP_EDGES
from .models.concept_heads import IctalInvolvementHead


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class IctalTokenPatientBag:
    """All cached foundation tokens and native targets for one patient.

    Token bundles contain no targets.  This object performs the explicit,
    receipt-checked join to an independently frozen TUSZ training manifest.
    It has no conversion path to ``EvidenceBatch`` and is never accepted by
    the SOZ reasoner.
    """

    patient_id: str
    event_ids: tuple[str, ...]
    expected_event_ids: tuple[str, ...]
    training_manifest_sha256: str
    expected_event_record_sha256s: tuple[str, ...]
    token_events: tuple[LoadedLaBraMConceptTokens, ...]
    targets: torch.Tensor
    target_mask: torch.Tensor
    training_authorized: bool = True

    def __post_init__(self) -> None:
        patient = str(self.patient_id).strip()
        if not patient:
            raise ValueError("Ictal token-bag patient_id cannot be empty")
        object.__setattr__(self, "patient_id", patient)
        if not _SHA256_RE.fullmatch(str(self.training_manifest_sha256)):
            raise ValueError("training_manifest_sha256 must be a lowercase SHA256")
        if not self.event_ids or len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("Ictal token bag requires unique non-empty event IDs")
        if set(self.event_ids) != set(self.expected_event_ids) or len(
            self.event_ids
        ) != len(self.expected_event_ids):
            raise ValueError("Ictal token bag is incomplete relative to its manifest")
        n_events = len(self.event_ids)
        if len(self.token_events) != n_events:
            raise ValueError("Token bundles must align one-to-one with event IDs")
        if len(self.expected_event_record_sha256s) != n_events:
            raise ValueError("Event-record SHA roster must align with event IDs")
        if any(
            not _SHA256_RE.fullmatch(str(value))
            for value in self.expected_event_record_sha256s
        ):
            raise ValueError("Event-record roster must contain lowercase SHA256 values")
        if tuple(event.event_id for event in self.token_events) != self.event_ids:
            raise ValueError("Cached token event order disagrees with the training manifest")
        actual_records = tuple(
            event.event_record_sha256 for event in self.token_events
        )
        if actual_records != self.expected_event_record_sha256s:
            raise ValueError("Cached token event-record SHA disagrees with the manifest")
        if len(
            {event.source_concept_manifest_sha256 for event in self.token_events}
        ) != 1:
            raise ValueError("One patient token bag cannot mix source manifests")
        if len(
            {event.foundation_feature_receipt_sha256 for event in self.token_events}
        ) != 1:
            raise ValueError("One patient token bag cannot mix foundation extractors")
        expected_target = (n_events, N_TCP_EDGES, 60)
        if tuple(self.targets.shape) != expected_target or tuple(
            self.target_mask.shape
        ) != expected_target:
            raise ValueError("Cached ictal targets/mask must have shape [E,20,60]")
        if self.targets.dtype != torch.float32 or self.target_mask.dtype != torch.bool:
            raise TypeError("Cached ictal targets must be float32 and mask must be bool")
        if not torch.isfinite(self.targets).all():
            raise ValueError("Cached ictal targets must be finite")
        observed = self.targets[self.target_mask]
        if observed.numel() and not torch.all((observed == 0) | (observed == 1)):
            raise ValueError("Observed cached ictal targets must be binary")
        if not self.target_mask.any():
            raise ValueError("Ictal token bag contains no observed labels")
        if not isinstance(self.training_authorized, bool):
            raise TypeError("training_authorized must be bool")

    @property
    def token_source_manifest_sha256(self) -> str:
        return self.token_events[0].source_concept_manifest_sha256

    @property
    def foundation_feature_receipt_sha256(self) -> str:
        return self.token_events[0].foundation_feature_receipt_sha256

    @property
    def source_target_mask(self) -> torch.Tensor:
        """Explicit TUSZ supervision mask; never deployment availability."""

        return self.target_mask


class IctalTokenBagDataset(Sequence[IctalTokenPatientBag]):
    """Lazy one-patient-at-a-time token-bag source.

    The declared lineage is checked again against every loaded bag.  This
    keeps multi-patient token tensors out of host memory while preserving the
    exactly-once patient epoch contract.
    """

    def __init__(
        self,
        patient_ids: Sequence[object],
        bag_loader: Callable[[str], IctalTokenPatientBag],
        *,
        training_manifest_sha256: str,
        token_source_manifest_sha256: str,
        foundation_feature_receipt_sha256: str,
        formal_token_corpus_verified: bool = False,
        formal_token_corpus_index_sha256: str | None = None,
        formal_token_corpus_training_bundle_manifest_sha256: str | None = None,
        formal_token_corpus_event_roster_sha256: str | None = None,
        formal_token_corpus_patient_roster_sha256: str | None = None,
        formal_token_corpus_tensor_roster_sha256: str | None = None,
        training_authorized: bool = True,
    ) -> None:
        patients = tuple(sorted(str(value).strip() for value in patient_ids))
        if not patients or any(not value for value in patients):
            raise ValueError("Lazy ictal token dataset needs non-empty patient IDs")
        if len(set(patients)) != len(patients):
            raise ValueError("Lazy ictal token patient roster must be unique")
        if not callable(bag_loader):
            raise TypeError("bag_loader must be callable")
        for field, value in (
            ("training_manifest_sha256", training_manifest_sha256),
            ("token_source_manifest_sha256", token_source_manifest_sha256),
            (
                "foundation_feature_receipt_sha256",
                foundation_feature_receipt_sha256,
            ),
        ):
            if not _SHA256_RE.fullmatch(str(value)):
                raise ValueError(f"{field} must be a lowercase SHA256")
        self._patient_ids = patients
        self._bag_loader = bag_loader
        self.training_manifest_sha256 = str(training_manifest_sha256)
        self.token_source_manifest_sha256 = str(token_source_manifest_sha256)
        self.foundation_feature_receipt_sha256 = str(
            foundation_feature_receipt_sha256
        )
        if not isinstance(training_authorized, bool):
            raise TypeError("training_authorized must be bool")
        self.training_authorized = training_authorized
        if not isinstance(formal_token_corpus_verified, bool):
            raise TypeError("formal_token_corpus_verified must be bool")
        formal_fields = {
            "formal_token_corpus_index_sha256": formal_token_corpus_index_sha256,
            "formal_token_corpus_training_bundle_manifest_sha256": (
                formal_token_corpus_training_bundle_manifest_sha256
            ),
            "formal_token_corpus_event_roster_sha256": (
                formal_token_corpus_event_roster_sha256
            ),
            "formal_token_corpus_patient_roster_sha256": (
                formal_token_corpus_patient_roster_sha256
            ),
            "formal_token_corpus_tensor_roster_sha256": (
                formal_token_corpus_tensor_roster_sha256
            ),
        }
        if formal_token_corpus_verified:
            for field, value in formal_fields.items():
                if not _SHA256_RE.fullmatch(str(value)):
                    raise ValueError(f"{field} must bind a verified lowercase SHA256")
        elif any(value is not None for value in formal_fields.values()):
            raise ValueError("Nonformal datasets cannot declare formal corpus lineage")
        self.formal_token_corpus_verified = formal_token_corpus_verified
        for field, value in formal_fields.items():
            setattr(self, field, None if value is None else str(value))

    @property
    def patient_ids(self) -> tuple[str, ...]:
        return self._patient_ids

    def __len__(self) -> int:
        return len(self._patient_ids)

    def _load(self, patient_id: str) -> IctalTokenPatientBag:
        bag = self._bag_loader(patient_id)
        if not isinstance(bag, IctalTokenPatientBag):
            raise TypeError("bag_loader must return IctalTokenPatientBag")
        checks = {
            "patient_id": bag.patient_id == patient_id,
            "training_manifest_sha256": (
                bag.training_manifest_sha256 == self.training_manifest_sha256
            ),
            "token_source_manifest_sha256": (
                bag.token_source_manifest_sha256
                == self.token_source_manifest_sha256
            ),
            "foundation_feature_receipt_sha256": (
                bag.foundation_feature_receipt_sha256
                == self.foundation_feature_receipt_sha256
            ),
            "training_authorized": (
                bag.training_authorized == self.training_authorized
            ),
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"Lazy ictal token bag failed declared lineage: {failed}")
        return bag

    def __getitem__(self, index: int) -> IctalTokenPatientBag:
        return self._load(self._patient_ids[index])

    def iter_epoch(
        self,
        patient_order: Sequence[object] | None = None,
    ) -> Iterator[IctalTokenPatientBag]:
        if patient_order is None:
            order = self._patient_ids
        else:
            order = tuple(str(value).strip() for value in patient_order)
            if len(order) != len(self._patient_ids) or set(order) != set(
                self._patient_ids
            ):
                raise ValueError(
                    "Epoch order must contain every cached-token patient exactly once"
                )
        for patient_id in order:
            yield self._load(patient_id)

    def iter_subset(
        self,
        patient_ids: Sequence[object],
    ) -> Iterator[IctalTokenPatientBag]:
        """Stream one explicit non-empty patient subset for native evaluation."""

        requested = tuple(sorted(str(value).strip() for value in patient_ids))
        if not requested or any(not value for value in requested):
            raise ValueError("Evaluation patient subset must be non-empty")
        if len(set(requested)) != len(requested):
            raise ValueError("Evaluation patient subset must be unique")
        missing = tuple(sorted(set(requested) - set(self._patient_ids)))
        if missing:
            raise ValueError(
                f"Evaluation patients are absent from the token corpus: {missing}"
            )
        for patient_id in requested:
            yield self._load(patient_id)

    def subset(self, patient_ids: Sequence[object]) -> "IctalTokenBagDataset":
        """Return a lazy patient subset without loading excluded patients.

        This is used by the formal-v5 auxiliary split.  The parent corpus
        lineage remains unchanged, while the epoch roster is exactly the
        requested subset.  No target, token, or availability tensor from an
        excluded patient is materialized by this operation.
        """

        requested = tuple(sorted(str(value).strip() for value in patient_ids))
        if not requested or any(not value for value in requested):
            raise ValueError("Ictal token subset requires non-empty patient IDs")
        if len(set(requested)) != len(requested):
            raise ValueError("Ictal token subset patient IDs must be unique")
        missing = tuple(sorted(set(requested) - set(self._patient_ids)))
        if missing:
            raise ValueError(
                f"Ictal token subset patients are absent from the corpus: {missing}"
            )
        return IctalTokenBagDataset(
            requested,
            self._bag_loader,
            training_manifest_sha256=self.training_manifest_sha256,
            token_source_manifest_sha256=self.token_source_manifest_sha256,
            foundation_feature_receipt_sha256=(
                self.foundation_feature_receipt_sha256
            ),
            formal_token_corpus_verified=self.formal_token_corpus_verified,
            formal_token_corpus_index_sha256=(
                self.formal_token_corpus_index_sha256
            ),
            formal_token_corpus_training_bundle_manifest_sha256=(
                self.formal_token_corpus_training_bundle_manifest_sha256
            ),
            formal_token_corpus_event_roster_sha256=(
                self.formal_token_corpus_event_roster_sha256
            ),
            formal_token_corpus_patient_roster_sha256=(
                self.formal_token_corpus_patient_roster_sha256
            ),
            formal_token_corpus_tensor_roster_sha256=(
                self.formal_token_corpus_tensor_roster_sha256
            ),
            training_authorized=self.training_authorized,
        )


def _event_slices(n_events: int, size: int | None) -> Iterator[slice]:
    if isinstance(size, bool) or (
        size is not None and (not isinstance(size, int) or size < 1)
    ):
        raise ValueError("event_microbatch_size must be a positive integer or None")
    step = n_events if size is None else size
    for start in range(0, n_events, step):
        yield slice(start, min(start + step, n_events))


def _head_device(head: IctalInvolvementHead) -> torch.device:
    devices = {parameter.device for parameter in head.parameters()}
    if len(devices) != 1:
        raise ValueError("Ictal head must occupy exactly one device")
    return next(iter(devices))


def _validate_optimizer(
    head: IctalInvolvementHead,
    optimizer: torch.optim.Optimizer,
) -> None:
    expected = {id(parameter) for parameter in head.parameters() if parameter.requires_grad}
    actual = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if actual != expected:
        raise ValueError("Optimizer parameters must exactly equal trainable ictal-head parameters")


def _forward_slice(
    head: IctalInvolvementHead,
    bag: IctalTokenPatientBag,
    event_slice: slice,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    device = _head_device(head)
    tokens = torch.stack(
        [event.tokens for event in bag.token_events[event_slice]], dim=0
    ).to(device=device, non_blocking=True)
    targets = bag.targets[event_slice].to(device=device, non_blocking=True)
    mask = bag.target_mask[event_slice].to(device=device, non_blocking=True)
    logits = head(tokens.detach())
    observed = int(mask.sum().item())
    if not observed:
        return logits, logits.sum() * 0.0, 0
    patient_ids = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
    loss = ictal_involvement_loss(logits, targets, mask, patient_ids)
    return logits, loss, observed


def _validate_bag_cohort(
    patient_bags: Sequence[IctalTokenPatientBag] | IctalTokenBagDataset,
) -> None:
    if not patient_bags:
        raise ValueError("Cached ictal epoch requires at least one patient bag")
    if isinstance(patient_bags, IctalTokenBagDataset):
        return
    patients = tuple(bag.patient_id for bag in patient_bags)
    if len(set(patients)) != len(patients):
        raise ValueError("A cached ictal epoch may contain each patient exactly once")
    checks = (
        ("training manifest", {bag.training_manifest_sha256 for bag in patient_bags}),
        ("token source manifest", {bag.token_source_manifest_sha256 for bag in patient_bags}),
        ("foundation extractor", {bag.foundation_feature_receipt_sha256 for bag in patient_bags}),
    )
    for label, values in checks:
        if len(values) != 1:
            raise ValueError(f"One cached ictal epoch must use one {label}")


def _iter_patient_bags(
    patient_bags: Sequence[IctalTokenPatientBag] | IctalTokenBagDataset,
    patient_order: Sequence[object] | None,
) -> Iterator[IctalTokenPatientBag]:
    if isinstance(patient_bags, IctalTokenBagDataset):
        yield from patient_bags.iter_epoch(patient_order)
        return
    if patient_order is None:
        yield from patient_bags
        return
    requested = tuple(str(value).strip() for value in patient_order)
    available = {bag.patient_id: bag for bag in patient_bags}
    if len(requested) != len(available) or set(requested) != set(available):
        raise ValueError("Epoch order must contain every cached-token patient exactly once")
    for patient_id in requested:
        yield available[patient_id]


def train_cached_ictal_epoch(
    head: IctalInvolvementHead,
    patient_bags: Sequence[IctalTokenPatientBag] | IctalTokenBagDataset,
    optimizer: torch.optim.Optimizer,
    *,
    patient_order: Sequence[object] | None = None,
    max_grad_norm: float | None = 1.0,
    event_microbatch_size: int | None = DEFAULT_EVENT_MICROBATCH_SIZE,
) -> IctalEpochOutput:
    """Train one patient-macro epoch with one update per complete token bag."""

    if isinstance(patient_bags, IctalTokenBagDataset):
        if not patient_bags.training_authorized:
            raise ValueError("Evaluation-only ictal token corpora cannot be used for training")
    elif any(not bag.training_authorized for bag in patient_bags):
        raise ValueError("Evaluation-only ictal token bags cannot be used for training")
    _validate_bag_cohort(patient_bags)
    tuple(_event_slices(1, event_microbatch_size))
    if max_grad_norm is not None and (
        not math.isfinite(float(max_grad_norm)) or float(max_grad_norm) <= 0
    ):
        raise ValueError("max_grad_norm must be positive or None")
    _validate_optimizer(head, optimizer)
    head.train()
    losses: list[float] = []
    n_events = 0
    n_observed = 0
    for bag in _iter_patient_bags(patient_bags, patient_order):
        optimizer.zero_grad(set_to_none=True)
        patient_observed = int(bag.target_mask.sum().item())
        patient_loss = 0.0
        for event_slice in _event_slices(len(bag.event_ids), event_microbatch_size):
            _, micro_loss, micro_observed = _forward_slice(head, bag, event_slice)
            if not micro_observed:
                continue
            weight = micro_observed / patient_observed
            (micro_loss * weight).backward()
            patient_loss += float(micro_loss.detach().cpu()) * weight
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(head.parameters(), float(max_grad_norm))
        optimizer.step()
        losses.append(patient_loss)
        n_events += len(bag.event_ids)
        n_observed += patient_observed
    return IctalEpochOutput(
        mean_patient_loss=sum(losses) / len(losses),
        n_patients=len(losses),
        n_events=n_events,
        n_observed_labels=n_observed,
    )


@torch.no_grad()
def evaluate_cached_ictal_epoch(
    head: IctalInvolvementHead,
    patient_bags: Sequence[IctalTokenPatientBag] | IctalTokenBagDataset,
    *,
    patient_order: Sequence[object] | None = None,
    event_microbatch_size: int | None = DEFAULT_EVENT_MICROBATCH_SIZE,
) -> tuple[IctalEpochOutput, IctalConceptMetrics]:
    """Evaluate loss and threshold-free metrics without re-running LaBraM."""

    _validate_bag_cohort(patient_bags)
    tuple(_event_slices(1, event_microbatch_size))
    head.eval()
    losses: list[float] = []
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    patient_indices: list[torch.Tensor] = []
    n_events = 0
    n_observed = 0
    for patient_index, bag in enumerate(
        _iter_patient_bags(patient_bags, patient_order)
    ):
        patient_observed = int(bag.target_mask.sum().item())
        patient_loss = 0.0
        for event_slice in _event_slices(len(bag.event_ids), event_microbatch_size):
            micro_logits, micro_loss, micro_observed = _forward_slice(
                head, bag, event_slice
            )
            count = micro_logits.shape[0]
            logits.append(micro_logits.detach().cpu())
            targets.append(bag.targets[event_slice].detach().cpu())
            masks.append(bag.target_mask[event_slice].detach().cpu())
            patient_indices.append(
                torch.full((count,), patient_index, dtype=torch.long)
            )
            if micro_observed:
                patient_loss += (
                    float(micro_loss.detach().cpu())
                    * micro_observed
                    / patient_observed
                )
        losses.append(patient_loss)
        n_events += len(bag.event_ids)
        n_observed += patient_observed
    epoch = IctalEpochOutput(
        mean_patient_loss=sum(losses) / len(losses),
        n_patients=len(losses),
        n_events=n_events,
        n_observed_labels=n_observed,
    )
    metrics = patient_macro_ictal_metrics(
        torch.cat(logits, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(masks, dim=0),
        torch.cat(patient_indices, dim=0),
    )
    return epoch, metrics


@torch.no_grad()
def evaluate_cached_ictal_patients(
    head: IctalInvolvementHead,
    dataset: IctalTokenBagDataset,
    patient_ids: Sequence[object],
    *,
    event_microbatch_size: int | None = DEFAULT_EVENT_MICROBATCH_SIZE,
) -> tuple[IctalEpochOutput, IctalConceptMetrics]:
    """Evaluate a declared held-out subset using only native TUSZ targets.

    Unlike the epoch API, this function deliberately permits a strict subset
    of a larger evaluation corpus.  It never accepts SOZ targets and streams
    token bundles one patient at a time.
    """

    if not isinstance(dataset, IctalTokenBagDataset):
        raise TypeError("dataset must be IctalTokenBagDataset")
    tuple(_event_slices(1, event_microbatch_size))
    head.eval()
    losses: list[float] = []
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    patient_indices: list[torch.Tensor] = []
    n_events = 0
    n_observed = 0
    for patient_index, bag in enumerate(dataset.iter_subset(patient_ids)):
        patient_observed = int(bag.target_mask.sum().item())
        patient_loss = 0.0
        for event_slice in _event_slices(len(bag.event_ids), event_microbatch_size):
            micro_logits, micro_loss, micro_observed = _forward_slice(
                head, bag, event_slice
            )
            count = micro_logits.shape[0]
            logits.append(micro_logits.detach().cpu())
            targets.append(bag.targets[event_slice].detach().cpu())
            masks.append(bag.target_mask[event_slice].detach().cpu())
            patient_indices.append(
                torch.full((count,), patient_index, dtype=torch.long)
            )
            if micro_observed:
                patient_loss += (
                    float(micro_loss.detach().cpu())
                    * micro_observed
                    / patient_observed
                )
        losses.append(patient_loss)
        n_events += len(bag.event_ids)
        n_observed += patient_observed
    if not losses:
        raise ValueError("Native ictal evaluation requires at least one patient")
    epoch = IctalEpochOutput(
        mean_patient_loss=sum(losses) / len(losses),
        n_patients=len(losses),
        n_events=n_events,
        n_observed_labels=n_observed,
    )
    metrics = patient_macro_ictal_metrics(
        torch.cat(logits, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(masks, dim=0),
        torch.cat(patient_indices, dim=0),
    )
    return epoch, metrics


__all__ = [
    "IctalTokenBagDataset",
    "IctalTokenPatientBag",
    "evaluate_cached_ictal_epoch",
    "evaluate_cached_ictal_patients",
    "train_cached_ictal_epoch",
]
