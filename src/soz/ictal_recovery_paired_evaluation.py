"""Development-only paired evaluation for the LaBraM k31 recovery.

The comparison unit is one held TUSZ patient.  Both heads are replayed on the
same cached LaBraM tokens and the same hash-pinned native TUSZ target/mask
tensors.  Unknown edge-time cells are never converted to negatives.

This module deliberately has no loader for DeepSOZ SOZ targets, private data,
source-dev, source-eval, or a ``final`` producer.  The formal-v4 comparator is
the original independent-second head (temporal context one), not the k5 head
that was evaluated transiently during formal-v5 development.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Mapping, Sequence

import torch

from .cached_concept_training import IctalTokenBagDataset, IctalTokenPatientBag
from .concept_metrics import patient_macro_ictal_metrics
from .ictal_production import LoadedIctalProductionRun
from .ictal_recovery_oof_v1_2 import LoadedLaBraMK31OOFRecoveryRunV12
from .ictal_target_snapshot import VerifiedIctalTargetSnapshot
from .models.concept_heads import (
    IctalInvolvementHead,
    LongContextTemporalResidualIctalInvolvementHead,
)


PAIRED_EVALUATION_SCHEMA = (
    "soz_labram_k31_vs_formal_v4_independent_paired_patient_evaluation_v1"
)
PAIRED_EVALUATION_RECEIPT_SCHEMA = (
    "soz_labram_k31_vs_formal_v4_independent_paired_patient_receipt_v1"
)
TARGET_SEMANTICS = "tusz_bipolar_edge_time_involvement_not_soz"
SELECTIONS = tuple(f"fold{index}" for index in range(5))
METRICS = ("bce", "brier", "auroc", "average_precision")
LOSS_METRICS = frozenset(("bce", "brier"))
DEFAULT_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_BOOTSTRAP_SEED = 20_260_810
DEFAULT_EVENT_MICROBATCH_SIZE = 8
_ARTIFACT_FILENAME = "paired_evaluation.json"
_RECEIPT_FILENAME = "receipt.json"
_EXPECTED_FILES = frozenset((_ARTIFACT_FILENAME, _RECEIPT_FILENAME))
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_JSON_BYTES = 16 * 1024 * 1024


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value).strip().lower()
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{field} must be a lowercase SHA256")
    return text


def _roster_sha256(values: Sequence[object]) -> str:
    roster = tuple(str(value).strip() for value in values)
    if (
        not roster
        or roster != tuple(sorted(roster))
        or len(roster) != len(set(roster))
        or any(not value for value in roster)
    ):
        raise ValueError("Patient roster must be non-empty, sorted, and unique")
    # Existing formal-v4 and k31 manifests use canonical JSON without the
    # artifact-file trailing newline for roster identities.
    encoded = json.dumps(
        list(roster),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _tensor_sha256(name: str, tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    metadata = f"{name}|{tuple(value.shape)}|{value.dtype}".encode("ascii")
    digest.update(len(metadata).to_bytes(4, "little"))
    digest.update(metadata)
    raw = value.view(torch.uint8).numpy().tobytes()
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)
    return digest.hexdigest()


def _head_device(head: torch.nn.Module) -> torch.device:
    devices = {parameter.device for parameter in head.parameters()}
    devices.update(buffer.device for buffer in head.buffers())
    if len(devices) != 1:
        raise ValueError("Each replay head must occupy exactly one device")
    device = next(iter(devices))
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Paired replay supports only CPU or CUDA")
    return device


def _event_slices(n_events: int, size: int) -> tuple[slice, ...]:
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("event_microbatch_size must be a positive integer")
    return tuple(
        slice(start, min(start + size, n_events))
        for start in range(0, n_events, size)
    )


def _model_metric_payload(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, float | None]:
    patient_indices = torch.zeros(logits.shape[0], dtype=torch.long)
    metrics = patient_macro_ictal_metrics(
        logits, targets, target_mask, patient_indices
    )
    if metrics.n_patients != 1:
        raise RuntimeError("One per-patient replay unexpectedly produced multiple patients")
    return {
        "bce": float(metrics.patient_macro_bce),
        "brier": float(metrics.patient_macro_brier),
        "auroc": (
            None
            if metrics.patient_macro_auroc is None
            else float(metrics.patient_macro_auroc)
        ),
        "average_precision": (
            None
            if metrics.patient_macro_average_precision is None
            else float(metrics.patient_macro_average_precision)
        ),
    }


@torch.inference_mode()
def replay_paired_patient(
    *,
    selection: str,
    bag: IctalTokenPatientBag,
    comparator_head: IctalInvolvementHead,
    candidate_head: LongContextTemporalResidualIctalInvolvementHead,
    event_microbatch_size: int = DEFAULT_EVENT_MICROBATCH_SIZE,
) -> dict[str, object]:
    """Replay both heads on one exact token/target/mask patient bag."""

    if selection not in SELECTIONS:
        raise ValueError("Paired evaluation accepts fold0..fold4 only")
    if not isinstance(bag, IctalTokenPatientBag):
        raise TypeError("bag must be an IctalTokenPatientBag")
    # Exact type checks prevent silently labelling a k5 or another subclass as
    # the formal-v4 independent-second comparator.
    if type(comparator_head) is not IctalInvolvementHead:
        raise TypeError(
            "formal-v4 comparator must be the exact independent-second head"
        )
    if type(candidate_head) is not LongContextTemporalResidualIctalInvolvementHead:
        raise TypeError("candidate must be the exact LaBraM k31 recovery head")
    comparator_device = _head_device(comparator_head)
    candidate_device = _head_device(candidate_head)
    if comparator_device != candidate_device:
        raise ValueError("Paired heads must occupy the same execution device")
    comparator_head.eval()
    candidate_head.eval()

    comparator_logits: list[torch.Tensor] = []
    candidate_logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for event_slice in _event_slices(len(bag.event_ids), event_microbatch_size):
        # Both heads receive the same in-memory tensor object before either
        # result is reduced, making the paired denominator explicit.
        tokens = torch.stack(
            [event.tokens for event in bag.token_events[event_slice]], dim=0
        ).detach().to(comparator_device)
        comparator_micro = comparator_head(tokens)
        candidate_micro = candidate_head(tokens)
        if comparator_micro.shape != candidate_micro.shape:
            raise ValueError("Paired heads emitted different logit shapes")
        if not torch.isfinite(comparator_micro).all() or not torch.isfinite(
            candidate_micro
        ).all():
            raise ValueError("Paired replay produced non-finite logits")
        comparator_logits.append(comparator_micro.detach().cpu())
        candidate_logits.append(candidate_micro.detach().cpu())
        targets.append(bag.targets[event_slice].detach().cpu())
        masks.append(bag.target_mask[event_slice].detach().cpu())

    full_comparator = torch.cat(comparator_logits, dim=0)
    full_candidate = torch.cat(candidate_logits, dim=0)
    full_targets = torch.cat(targets, dim=0).to(torch.float32)
    full_mask = torch.cat(masks, dim=0).to(torch.bool)
    observed_targets = full_targets[full_mask]
    if not observed_targets.numel():
        raise ValueError("Held patient contains no observed TUSZ target")
    if not torch.all((observed_targets == 0) | (observed_targets == 1)):
        raise ValueError("Observed TUSZ targets must remain binary")
    positive = int(observed_targets.sum().item())
    negative = int(observed_targets.numel()) - positive
    comparator = _model_metric_payload(
        full_comparator, full_targets, full_mask
    )
    candidate = _model_metric_payload(full_candidate, full_targets, full_mask)
    if (comparator["auroc"] is None) != (candidate["auroc"] is None):
        raise RuntimeError("Paired discrimination definedness changed by model")

    delta: dict[str, float | None] = {}
    improvement: dict[str, float | None] = {}
    for metric in METRICS:
        comparator_value = comparator[metric]
        candidate_value = candidate[metric]
        if comparator_value is None or candidate_value is None:
            delta[metric] = None
            improvement[metric] = None
            continue
        raw_delta = float(candidate_value - comparator_value)
        delta[metric] = raw_delta
        improvement[metric] = (
            -raw_delta if metric in LOSS_METRICS else raw_delta
        )

    return {
        "selection": selection,
        "patient_id": bag.patient_id,
        "event_count": len(bag.event_ids),
        "event_roster_sha256": _sha256_bytes(
            _canonical_json_bytes(list(bag.event_ids))
        ),
        "target_sha256": _tensor_sha256("targets", full_targets),
        "target_mask_sha256": _tensor_sha256("target_mask", full_mask),
        "n_observed_labels": int(observed_targets.numel()),
        "n_positive_labels": positive,
        "n_negative_labels": negative,
        "n_unknown_labels": int(full_mask.numel() - full_mask.sum().item()),
        "discrimination_evaluable": bool(positive and negative),
        "formal_v4_independent_second": comparator,
        "labram_k31_v1_2": candidate,
        "delta_candidate_minus_comparator": delta,
        "candidate_improvement": improvement,
    }


def replay_paired_fold(
    *,
    selection: str,
    dataset: IctalTokenBagDataset,
    patient_ids: Sequence[object],
    comparator_head: IctalInvolvementHead,
    candidate_head: LongContextTemporalResidualIctalInvolvementHead,
    event_microbatch_size: int = DEFAULT_EVENT_MICROBATCH_SIZE,
) -> tuple[dict[str, object], ...]:
    if selection not in SELECTIONS:
        raise ValueError("Paired evaluation accepts fold0..fold4 only")
    if not isinstance(dataset, IctalTokenBagDataset):
        raise TypeError("dataset must be an IctalTokenBagDataset")
    patients = tuple(str(value).strip() for value in patient_ids)
    if (
        not patients
        or patients != tuple(sorted(patients))
        or len(patients) != len(set(patients))
    ):
        raise ValueError("Held patient IDs must be non-empty, sorted, and unique")
    return tuple(
        replay_paired_patient(
            selection=selection,
            bag=bag,
            comparator_head=comparator_head,
            candidate_head=candidate_head,
            event_microbatch_size=event_microbatch_size,
        )
        for bag in dataset.iter_subset(patients)
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_seed(base_seed: int, *, scope: str, metric: str) -> int:
    identity = f"{base_seed}|{scope}|{metric}".encode("ascii")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")


def _metric_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    scope: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    paired: list[tuple[float, float]] = []
    for row in rows:
        comparator_payload = row["formal_v4_independent_second"]
        candidate_payload = row["labram_k31_v1_2"]
        if not isinstance(comparator_payload, Mapping) or not isinstance(
            candidate_payload, Mapping
        ):
            raise TypeError("Patient metric payload is malformed")
        comparator_value = comparator_payload[metric]
        candidate_value = candidate_payload[metric]
        if (comparator_value is None) != (candidate_value is None):
            raise ValueError("Paired metric definedness differs between models")
        if comparator_value is not None:
            paired.append((float(comparator_value), float(candidate_value)))
    if not paired:
        return {
            "n_paired_patients": 0,
            "formal_v4_independent_second_patient_macro": None,
            "labram_k31_v1_2_patient_macro": None,
            "paired_delta_candidate_minus_comparator": None,
            "paired_delta_patient_cluster_bootstrap_95_ci": None,
            "candidate_improvement": None,
            "candidate_improvement_patient_cluster_bootstrap_95_ci": None,
        }
    comparator_values = [value[0] for value in paired]
    candidate_values = [value[1] for value in paired]
    paired_deltas = [candidate - comparator for comparator, candidate in paired]
    delta = sum(paired_deltas) / len(paired_deltas)
    rng = random.Random(
        _bootstrap_seed(bootstrap_seed, scope=scope, metric=metric)
    )
    bootstrap_deltas = []
    for _ in range(bootstrap_replicates):
        sampled = [paired_deltas[rng.randrange(len(paired_deltas))] for _ in paired]
        bootstrap_deltas.append(sum(sampled) / len(sampled))
    delta_ci = [
        _quantile(bootstrap_deltas, 0.025),
        _quantile(bootstrap_deltas, 0.975),
    ]
    sign = -1.0 if metric in LOSS_METRICS else 1.0
    return {
        "n_paired_patients": len(paired),
        "formal_v4_independent_second_patient_macro": (
            sum(comparator_values) / len(comparator_values)
        ),
        "labram_k31_v1_2_patient_macro": (
            sum(candidate_values) / len(candidate_values)
        ),
        "paired_delta_candidate_minus_comparator": delta,
        "paired_delta_patient_cluster_bootstrap_95_ci": delta_ci,
        "candidate_improvement": sign * delta,
        "candidate_improvement_patient_cluster_bootstrap_95_ci": [
            min(sign * delta_ci[0], sign * delta_ci[1]),
            max(sign * delta_ci[0], sign * delta_ci[1]),
        ],
    }


def summarize_paired_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    scope: str,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < 1
    ):
        raise ValueError("bootstrap_replicates must be a positive integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise TypeError("bootstrap_seed must be an integer")
    normalized = tuple(rows)
    if not normalized:
        raise ValueError("Paired summary requires at least one patient")
    patient_ids = tuple(str(row["patient_id"]) for row in normalized)
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("Paired summary may contain each patient exactly once")
    return {
        "scope": scope,
        "n_patients": len(normalized),
        "n_discrimination_patients": sum(
            bool(row["discrimination_evaluable"]) for row in normalized
        ),
        "n_observed_labels": sum(
            int(row["n_observed_labels"]) for row in normalized
        ),
        "n_positive_labels": sum(
            int(row["n_positive_labels"]) for row in normalized
        ),
        "n_negative_labels": sum(
            int(row["n_negative_labels"]) for row in normalized
        ),
        "n_unknown_labels_masked": sum(
            int(row["n_unknown_labels"]) for row in normalized
        ),
        "metrics": {
            metric: _metric_summary(
                normalized,
                metric=metric,
                scope=scope,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            )
            for metric in METRICS
        },
    }


def validate_fold_pair_lineage(
    *,
    selection: str,
    formal_v4: LoadedIctalProductionRun,
    k31_v1_2: LoadedLaBraMK31OOFRecoveryRunV12,
    target_snapshot: VerifiedIctalTargetSnapshot,
) -> dict[str, object]:
    """Cross-check one strict fold pair and return its complete lineage."""

    if selection not in SELECTIONS:
        raise ValueError("Lineage comparison accepts fold0..fold4 only")
    if not isinstance(formal_v4, LoadedIctalProductionRun):
        raise TypeError("formal_v4 must come from the strict production loader")
    if not isinstance(k31_v1_2, LoadedLaBraMK31OOFRecoveryRunV12):
        raise TypeError("k31_v1_2 must come from the strict v1.2 loader")
    if not isinstance(target_snapshot, VerifiedIctalTargetSnapshot):
        raise TypeError("target_snapshot must come from the strict snapshot loader")
    formal = formal_v4.manifest
    candidate = k31_v1_2.manifest
    if formal["selection"] != selection or candidate["selection"] != selection:
        raise ValueError("Fold selection differs across paired manifests")
    if type(formal_v4.checkpoint.head) is not IctalInvolvementHead:
        raise ValueError("formal-v4 checkpoint is not independent-second context=1")
    if type(k31_v1_2.head) is not LongContextTemporalResidualIctalInvolvementHead:
        raise ValueError("Recovery checkpoint is not the exact k31 candidate")
    common_fields = (
        "split_manifest_sha256",
        "oof_protocol_artifact_sha256",
        "oof_protocol_receipt_sha256",
        "oof_plan_receipt_sha256",
        "training_manifest_sha256",
        "training_corpus_index_sha256",
        "native_evaluation_manifest_sha256",
        "native_evaluation_corpus_index_sha256",
        "native_evaluation_public_patient_ids",
    )
    mismatches = tuple(
        field for field in common_fields if formal[field] != candidate[field]
    )
    if mismatches:
        raise ValueError(f"Paired fold lineage differs: {mismatches}")
    if formal["native_evaluation_role"] != "source_train_oof_fold_heldout_native_tusz":
        raise ValueError("formal-v4 comparator is not a source-train OOF fold")
    if candidate["target_snapshot_manifest_sha256"] != target_snapshot.manifest_sha256:
        raise ValueError("k31 fold binds another target snapshot manifest")
    if candidate["target_snapshot_receipt_sha256"] != target_snapshot.receipt_sha256:
        raise ValueError("k31 fold binds another target snapshot receipt")
    forbidden_flags = {
        "formal_v4.deepsoz_soz_labels_used": formal["native_metrics"][
            "deepsoz_soz_labels_used"
        ],
        "formal_v4.missing_tusz_bins_imputed_as_negative": formal[
            "native_metrics"
        ]["missing_tusz_bins_imputed_as_negative"],
        "k31.deepsoz_soz_labels_used": candidate["deepsoz_soz_labels_used"],
        "k31.private_labels_used": candidate["private_labels_used"],
        "k31.missing_tusz_cells_imputed_as_negative": candidate[
            "missing_tusz_cells_imputed_as_negative"
        ],
        "k31.deepsoz_target_source_loaded": candidate[
            "deepsoz_target_source_loaded"
        ],
        "k31.deepsoz_target_values_reachable": candidate[
            "deepsoz_target_values_reachable"
        ],
    }
    if any(bool(value) for value in forbidden_flags.values()):
        raise ValueError("A forbidden target source or missing-label imputation entered replay")
    native_patients = tuple(formal["native_evaluation_public_patient_ids"])
    native_roster_sha = _roster_sha256(native_patients)
    if native_roster_sha != formal["native_evaluation_public_roster_sha256"]:
        raise ValueError("formal-v4 native patient roster receipt mismatch")
    if native_roster_sha != candidate["native_evaluation_public_roster_sha256"]:
        raise ValueError("k31 native patient roster receipt mismatch")
    formal_training = tuple(formal["training_source_public_patient_ids"])
    candidate_training = tuple(candidate["training_public_patient_ids"])
    return {
        "selection": selection,
        "native_evaluation_public_patient_ids": list(native_patients),
        "native_evaluation_public_roster_sha256": native_roster_sha,
        "common_split_and_data_lineage": {
            field: formal[field] for field in common_fields[:-1]
        },
        "formal_v4_independent_second": {
            "temporal_context_seconds": 1,
            "production_run_manifest_sha256": formal_v4.manifest_sha256,
            "checkpoint_manifest_sha256": formal_v4.checkpoint.manifest_sha256,
            "checkpoint_sha256": formal_v4.checkpoint.checkpoint_sha256,
            "foundation_feature_receipt_sha256": formal_v4.checkpoint.metadata[
                "foundation_feature_receipt_sha256"
            ],
            "training_public_patient_ids": list(formal_training),
            "training_public_roster_sha256": _roster_sha256(formal_training),
        },
        "labram_k31_v1_2": {
            "temporal_context_seconds": 31,
            "recovery_run_manifest_sha256": k31_v1_2.manifest_sha256,
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "head_state_sha256": candidate["head_state_sha256"],
            "execution_receipt_sha256": candidate["execution_receipt_sha256"],
            "training_public_patient_ids": list(candidate_training),
            "training_public_roster_sha256": _roster_sha256(candidate_training),
        },
        "training_rosters_identical": formal_training == candidate_training,
        "architecture_only_attribution_allowed": False,
        "target_snapshot": {
            "manifest_sha256": target_snapshot.manifest_sha256,
            "receipt_sha256": target_snapshot.receipt_sha256,
        },
    }


def _replayed_manifest_metrics(
    rows: Sequence[Mapping[str, object]], model_key: str
) -> dict[str, object]:
    summary = summarize_paired_rows(
        rows, scope="manifest_replay_check", bootstrap_replicates=1
    )
    metrics = summary["metrics"]
    field = (
        "formal_v4_independent_second_patient_macro"
        if model_key == "formal_v4_independent_second"
        else "labram_k31_v1_2_patient_macro"
    )
    return {
        "patient_macro_bce": metrics["bce"][field],
        "patient_macro_brier": metrics["brier"][field],
        "patient_macro_auroc": metrics["auroc"][field],
        "patient_macro_average_precision": metrics["average_precision"][field],
        "n_patients": summary["n_patients"],
        "n_discrimination_patients": summary["n_discrimination_patients"],
        "n_observed_labels": summary["n_observed_labels"],
        "n_positive_labels": summary["n_positive_labels"],
        "n_negative_labels": summary["n_negative_labels"],
    }


def verify_replay_against_run_metrics(
    *,
    rows: Sequence[Mapping[str, object]],
    formal_v4: LoadedIctalProductionRun,
    k31_v1_2: LoadedLaBraMK31OOFRecoveryRunV12,
    tolerance: float = 5e-6,
) -> dict[str, object]:
    """Require strict replay to reproduce both stored native metric receipts."""

    if not math.isfinite(float(tolerance)) or tolerance < 0:
        raise ValueError("Replay tolerance must be finite and non-negative")
    replayed = {
        "formal_v4_independent_second": _replayed_manifest_metrics(
            rows, "formal_v4_independent_second"
        ),
        "labram_k31_v1_2": _replayed_manifest_metrics(rows, "labram_k31_v1_2"),
    }
    recorded = {
        "formal_v4_independent_second": {
            field: formal_v4.manifest["native_metrics"][field]
            for field in replayed["formal_v4_independent_second"]
        },
        "labram_k31_v1_2": dict(k31_v1_2.manifest["training_run"]["metrics"]),
    }
    maximum_difference = 0.0
    for model in replayed:
        for field, value in replayed[model].items():
            expected = recorded[model][field]
            if isinstance(value, int):
                if value != expected:
                    raise ValueError(f"Strict replay count differs for {model}.{field}")
            elif value is None or expected is None:
                if value is not None or expected is not None:
                    raise ValueError(
                        f"Strict replay metric definedness differs for {model}.{field}"
                    )
            else:
                difference = abs(float(value) - float(expected))
                maximum_difference = max(maximum_difference, difference)
                if difference > tolerance:
                    raise ValueError(
                        f"Strict replay differs from stored metrics for {model}.{field}"
                    )
    return {
        "absolute_tolerance": float(tolerance),
        "maximum_absolute_difference": maximum_difference,
        "passed": True,
    }


def build_paired_evaluation_payload(
    *,
    patient_rows: Sequence[Mapping[str, object]],
    fold_lineage: Sequence[Mapping[str, object]],
    replay_checks: Sequence[Mapping[str, object]],
    target_snapshot: VerifiedIctalTargetSnapshot,
    master_bundle_manifest_sha256: str,
    master_source_manifest_sha256: str,
    master_corpus_index_sha256: str,
    foundation_feature_receipt_sha256: str,
    preprocessing_selection_artifact_sha256: str,
    preprocessing_protocol_receipt_sha256: str,
    execution_device: str,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    rows = tuple(patient_rows)
    lineages = tuple(fold_lineage)
    checks = tuple(replay_checks)
    if tuple(lineage["selection"] for lineage in lineages) != SELECTIONS:
        raise ValueError("Paired payload requires exactly fold0..fold4 in order")
    if len(checks) != len(SELECTIONS) or not all(check["passed"] for check in checks):
        raise ValueError("Every paired fold must pass strict metric replay")
    row_selections = tuple(str(row["selection"]) for row in rows)
    if set(row_selections) != set(SELECTIONS):
        raise ValueError("Paired patient rows must cover all five folds")
    patient_ids = tuple(sorted(str(row["patient_id"]) for row in rows))
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("OOF paired payload contains a patient in multiple folds")
    if execution_device not in {"cpu", "cuda"}:
        raise ValueError("execution_device must be cpu or cuda")
    fold_summaries = {
        selection: summarize_paired_rows(
            tuple(row for row in rows if row["selection"] == selection),
            scope=selection,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )
        for selection in SELECTIONS
    }
    overall = summarize_paired_rows(
        rows,
        scope="overall_five_fold_oof",
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "schema_version": PAIRED_EVALUATION_SCHEMA,
        "status": "development_only",
        "development_only": True,
        "formal_promotion": False,
        "model_selection_performed": False,
        "checkpoint_authorized_for_formal_evidence_or_reasoner": False,
        "target_semantics": TARGET_SEMANTICS,
        "clinical_interpretation": "retrospective_scalp_visible_ictal_involvement_not_soz",
        "comparison": {
            "candidate": "labram_k31_v1_2_symmetric_retrospective",
            "candidate_temporal_context_seconds": 31,
            "comparator": "formal_v4_independent_second",
            "comparator_temporal_context_seconds": 1,
            "comparator_is_k5": False,
            "same_native_patient_target_and_mask_for_each_pair": True,
            "architecture_only_attribution_allowed": False,
            "architecture_only_attribution_limitation": (
                "formal-v4 and k31-v1.2 training rosters differ because the "
                "recovery excludes unopened I-gate patients"
            ),
        },
        "data_firewall": {
            "scope": "five_source_train_oof_native_held_tusz_folds_only",
            "source_dev_used": False,
            "source_eval_used": False,
            "private_used": False,
            "deepsoz_soz_target_values_used": False,
            "final_producer_used": False,
            "missing_tusz_cells_imputed_as_negative": False,
        },
        "bootstrap": {
            "unit": "patient",
            "paired": True,
            "method": "ordinary_percentile_patient_cluster_bootstrap",
            "confidence_level": 0.95,
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "overall_resampling": "pooled_unique_oof_patients",
        },
        "execution": {
            "device_type": execution_device,
            "inference_only": True,
            "optimizer_used": False,
        },
        "shared_data_lineage": {
            "master_bundle_manifest_sha256": _require_sha256(
                master_bundle_manifest_sha256,
                field="master_bundle_manifest_sha256",
            ),
            "master_source_manifest_sha256": _require_sha256(
                master_source_manifest_sha256,
                field="master_source_manifest_sha256",
            ),
            "master_corpus_index_sha256": _require_sha256(
                master_corpus_index_sha256, field="master_corpus_index_sha256"
            ),
            "foundation_feature_receipt_sha256": _require_sha256(
                foundation_feature_receipt_sha256,
                field="foundation_feature_receipt_sha256",
            ),
            "preprocessing_selection_artifact_sha256": _require_sha256(
                preprocessing_selection_artifact_sha256,
                field="preprocessing_selection_artifact_sha256",
            ),
            "preprocessing_protocol_receipt_sha256": _require_sha256(
                preprocessing_protocol_receipt_sha256,
                field="preprocessing_protocol_receipt_sha256",
            ),
            "target_snapshot_manifest_sha256": target_snapshot.manifest_sha256,
            "target_snapshot_receipt_sha256": target_snapshot.receipt_sha256,
        },
        "fold_lineage": list(lineages),
        "strict_replay_checks": list(checks),
        "patient_roster_sha256": _roster_sha256(patient_ids),
        "patient_rows": [dict(row) for row in rows],
        "fold_results": fold_summaries,
        "overall_result": overall,
    }


def _safe_output_directory(value: str | Path) -> Path:
    target = Path(os.path.abspath(value))
    if target.name in {"", ".", ".."} or not target.parent.is_dir():
        raise ValueError("Paired output requires a concrete path with an existing parent")
    for component in (target.parent, *target.parent.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("Paired output path cannot traverse symlinks")
    if os.path.lexists(target):
        raise FileExistsError(f"Paired output already exists: {target}")
    return target


def save_paired_evaluation(
    output_directory: str | Path, payload: Mapping[str, object]
) -> tuple[Path, str, str]:
    target = _safe_output_directory(output_directory)
    document = dict(payload)
    if document.get("schema_version") != PAIRED_EVALUATION_SCHEMA:
        raise ValueError("Paired payload has the wrong schema")
    raw = _canonical_json_bytes(document)
    artifact_sha = _sha256_bytes(raw)
    receipt = {
        "schema_version": PAIRED_EVALUATION_RECEIPT_SCHEMA,
        "artifact_filename": _ARTIFACT_FILENAME,
        "artifact_sha256": artifact_sha,
        "patient_roster_sha256": document["patient_roster_sha256"],
        "development_only": True,
        "formal_promotion": False,
    }
    receipt_raw = _canonical_json_bytes(receipt)
    receipt_sha = _sha256_bytes(receipt_raw)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / _ARTIFACT_FILENAME).write_bytes(raw)
        (staging / _RECEIPT_FILENAME).write_bytes(receipt_raw)
        os.rename(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return target, artifact_sha, receipt_sha


def load_paired_evaluation(
    path: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> Mapping[str, object]:
    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_dir() or source.resolve() != source:
        raise ValueError("Paired evaluation must be a regular absolute directory")
    if {item.name for item in source.iterdir()} != _EXPECTED_FILES:
        raise ValueError("Paired evaluation has missing or unknown files")
    artifact_raw = (source / _ARTIFACT_FILENAME).read_bytes()
    receipt_raw = (source / _RECEIPT_FILENAME).read_bytes()
    if not 1 <= len(artifact_raw) <= _MAX_JSON_BYTES or not 1 <= len(
        receipt_raw
    ) <= _MAX_JSON_BYTES:
        raise ValueError("Paired evaluation JSON has an invalid size")
    artifact_sha = _sha256_bytes(artifact_raw)
    receipt_sha = _sha256_bytes(receipt_raw)
    if artifact_sha != _require_sha256(
        expected_artifact_sha256, field="expected_artifact_sha256"
    ):
        raise ValueError("Paired evaluation artifact SHA mismatch")
    if receipt_sha != _require_sha256(
        expected_receipt_sha256, field="expected_receipt_sha256"
    ):
        raise ValueError("Paired evaluation receipt SHA mismatch")
    try:
        artifact = json.loads(artifact_raw.decode("utf-8"))
        receipt = json.loads(receipt_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Paired evaluation is not valid UTF-8 JSON") from exc
    if (
        not isinstance(artifact, dict)
        or not isinstance(receipt, dict)
        or _canonical_json_bytes(artifact) != artifact_raw
        or _canonical_json_bytes(receipt) != receipt_raw
    ):
        raise ValueError("Paired evaluation JSON is not canonical")
    expected_receipt = {
        "schema_version": PAIRED_EVALUATION_RECEIPT_SCHEMA,
        "artifact_filename": _ARTIFACT_FILENAME,
        "artifact_sha256": artifact_sha,
        "patient_roster_sha256": artifact.get("patient_roster_sha256"),
        "development_only": True,
        "formal_promotion": False,
    }
    if receipt != expected_receipt:
        raise ValueError("Paired evaluation receipt does not bind the artifact")
    fixed = {
        "schema_version": PAIRED_EVALUATION_SCHEMA,
        "development_only": True,
        "formal_promotion": False,
        "model_selection_performed": False,
        "checkpoint_authorized_for_formal_evidence_or_reasoner": False,
        "target_semantics": TARGET_SEMANTICS,
    }
    if any(artifact.get(field) != value for field, value in fixed.items()):
        raise ValueError("Paired evaluation changed a scientific boundary")
    firewall = artifact.get("data_firewall")
    if not isinstance(firewall, Mapping) or any(
        firewall.get(field) is not False
        for field in (
            "source_dev_used",
            "source_eval_used",
            "private_used",
            "deepsoz_soz_target_values_used",
            "final_producer_used",
            "missing_tusz_cells_imputed_as_negative",
        )
    ):
        raise ValueError("Paired evaluation violates its data firewall")
    return artifact


__all__ = (
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_EVENT_MICROBATCH_SIZE",
    "PAIRED_EVALUATION_RECEIPT_SCHEMA",
    "PAIRED_EVALUATION_SCHEMA",
    "SELECTIONS",
    "build_paired_evaluation_payload",
    "load_paired_evaluation",
    "replay_paired_fold",
    "replay_paired_patient",
    "save_paired_evaluation",
    "summarize_paired_rows",
    "validate_fold_pair_lineage",
    "verify_replay_against_run_metrics",
)
