#!/usr/bin/env python3
"""Independent common17 SOZ OOF retraining, split, metric, and CI audit.

The audit consumes only public EEG-derived frozen features/targets.  It:

* compares an independently rerun five-fold training tensor bit-for-bit;
* replays every held-fold prediction from its saved fold-specific state;
* perturbs held-patient targets and refits each fold as a leakage negative
  control (the resulting state must remain bitwise identical);
* independently reconstructs the GT-only FZ/PZ->CZ projection;
* reconstructs the published DeepSOZ one-hop table induced on common17;
* reports integer patient outcomes, per-fold Wilson intervals, and aggregate
  patient bootstrap percentile intervals.

This remains an oracle-event SOZ localization experiment, not an end-to-end
long-recording detection-plus-localization result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from safetensors import safe_open
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sota_soz.train_common17_oracle_event_oof_v1 import (  # noqa: E402
    Common17EventSetReasoner,
    _fit_fold,
    _jeffreys_prior,
    _patient_event_rows,
    _predict,
)
from src.clinical_eeg_long_recording.common17_experiment_v1 import (  # noqa: E402
    COMMON_17,
    DEEPSOZ_COMMON17_INDUCED_NEIGHBORS,
)
from src.soz.geometry import CHANNEL_INDEX  # noqa: E402
from src.soz.metrics import DEEPSOZ_STANDARD19_NEIGHBORS  # noqa: E402


SCHEMA_VERSION = "clinical_eeg_common17_oracle_oof_training_split_bootstrap_audit_v1"
PRIMARY_ARM = "strict_car17_labram"
DEFAULT_FROZEN = ROOT / "outputs/clinical_eeg_common17_oracle_event_oof_r3r3_20260824"
DEFAULT_RERUN = (
    ROOT
    / "outputs/clinical_eeg_common17_oracle_event_oof_independent_rerun_v1_20260825"
)
DEFAULT_PHASE = ROOT / "outputs/clinical_eeg_common17_car17_labram_phase_v1_20260824"
DEFAULT_TARGET = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/clinical_eeg_common17_oracle_oof_training_split_bootstrap_audit_v1_20260825"
    / "receipt.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load(path: Path, keys: Sequence[str]) -> dict[str, torch.Tensor]:
    with safe_open(
        str(path.resolve(strict=True)), framework="pt", device="cpu"
    ) as source:
        missing = sorted(set(keys).difference(source.keys()))
        if missing:
            raise KeyError(f"Missing tensors in {path}: {missing}")
        return {key: source.get_tensor(key) for key in keys}


def _all_tensor_keys(path: Path) -> tuple[str, ...]:
    with safe_open(
        str(path.resolve(strict=True)), framework="pt", device="cpu"
    ) as source:
        return tuple(source.keys())


def _wilson(successes: int, total: int) -> list[float]:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [center - half, center + half]


def _induced_neighbor_graph() -> tuple[tuple[int, ...], ...]:
    retained19 = tuple(CHANNEL_INDEX[channel] for channel in COMMON_17)
    old_to_new = {old: new for new, old in enumerate(retained19)}
    graph = tuple(
        tuple(
            old_to_new[neighbor]
            for neighbor in DEEPSOZ_STANDARD19_NEIGHBORS[old]
            if neighbor in old_to_new
        )
        for old in retained19
    )
    named = {
        channel: tuple(COMMON_17[index] for index in row)
        for channel, row in zip(COMMON_17, graph)
    }
    if named != dict(DEEPSOZ_COMMON17_INDUCED_NEIGHBORS):
        raise RuntimeError("independent induced graph differs from common17 contract")
    return graph


def metric_outcomes(
    probability: torch.Tensor,
    targets: torch.Tensor,
    pre_mapping_positive_count: torch.Tensor,
    graph: Sequence[Sequence[int]],
) -> dict[str, np.ndarray]:
    """Return one independent metric value per held-out patient."""

    if tuple(probability.shape) != tuple(targets.shape) or tuple(targets.shape[1:]) != (
        17,
    ):
        raise ValueError("metric inputs must be aligned [P,17]")
    if not torch.isfinite(probability).all() or not torch.isfinite(targets).all():
        raise ValueError("metric inputs must be finite")
    order = torch.argsort(probability, dim=1, descending=True, stable=True)
    top_values = probability.gather(1, order[:, :1])
    if bool(((probability == top_values).sum(dim=1) != 1).any()):
        raise ValueError("integer endpoint audit requires unique Top-1")
    ranked = targets.gather(1, order)
    exact = ranked[:, 0].bool()
    hit3 = ranked[:, :3].sum(dim=1) > 0
    hit5 = ranked[:, :5].sum(dim=1) > 0
    first = ranked.argmax(dim=1)
    n2: list[bool] = []
    n4: list[bool] = []
    for row in range(len(probability)):
        positive = targets[row] == 1
        top = int(order[row, 0])
        for gate, destination in ((2, n2), (4, n4)):
            acceptable = positive.clone()
            if int(pre_mapping_positive_count[row]) <= gate:
                for index in torch.nonzero(positive, as_tuple=False).flatten().tolist():
                    acceptable[list(graph[index])] = True
            destination.append(bool(acceptable[top]))
    return {
        "exact_top1": exact.numpy().astype(np.float64),
        "accuracy": exact.numpy().astype(np.float64),
        "deepsoz_N2": np.asarray(n2, dtype=np.float64),
        "deepsoz_N4": np.asarray(n4, dtype=np.float64),
        "hit_at_3": hit3.numpy().astype(np.float64),
        "hit_at_5": hit5.numpy().astype(np.float64),
        "mrr": (1.0 / (first.double() + 1.0)).numpy(),
    }


def patient_bootstrap_intervals(
    outcomes: Mapping[str, np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Patient bootstrap with one shared resample stream for every endpoint."""

    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    names = tuple(outcomes)
    lengths = {len(np.asarray(outcomes[name])) for name in names}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError("bootstrap outcomes must share at least two patients")
    patient_count = next(iter(lengths))
    rng = np.random.default_rng(seed)
    samples: dict[str, list[np.ndarray]] = {name: [] for name in names}
    remaining = replicates
    while remaining:
        count = min(10_000, remaining)
        indices = rng.integers(
            0, patient_count, size=(count, patient_count), dtype=np.int16
        )
        for name in names:
            values = np.asarray(outcomes[name], dtype=np.float64)
            samples[name].append(values[indices].mean(axis=1))
        remaining -= count
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        values = np.concatenate(samples[name])
        result[name] = {
            "point_estimate": float(np.asarray(outcomes[name]).mean()),
            "patient_bootstrap_percentile_95_ci": [
                float(item) for item in np.quantile(values, (0.025, 0.975))
            ],
            "bootstrap_mean": float(values.mean()),
            "replicates": replicates,
            "seed": seed,
            "resampling_unit": "patient",
        }
    return result


def _fold_metrics(
    outcomes: Mapping[str, np.ndarray], folds: torch.Tensor
) -> list[dict[str, Any]]:
    result = []
    for fold in range(5):
        indices = torch.nonzero(folds == fold, as_tuple=False).flatten().numpy()
        metrics: dict[str, Any] = {}
        for name, values in outcomes.items():
            selected = np.asarray(values)[indices]
            if name == "mrr":
                metrics[name] = {"mean": float(selected.mean()), "total": len(selected)}
            else:
                successes = int(selected.sum())
                metrics[name] = {
                    "successes": successes,
                    "total": len(selected),
                    "rate": successes / len(selected),
                    "wilson_95_ci": _wilson(successes, len(selected)),
                }
        result.append({"fold": fold, "held_patients": len(indices), "metrics": metrics})
    return result


def _independent_target_projection(
    targets19: torch.Tensor,
    mask19: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    observed = (targets19 == 1) & mask19
    pre_count = observed.sum(dim=1).long()
    mapped = observed.clone()
    fz = CHANNEL_INDEX["FZ"]
    pz = CHANNEL_INDEX["PZ"]
    cz = CHANNEL_INDEX["CZ"]
    mapped[:, cz] |= observed[:, fz] | observed[:, pz]
    indices = torch.tensor(
        [CHANNEL_INDEX[channel] for channel in COMMON_17], dtype=torch.long
    )
    target17 = mapped.index_select(1, indices).float().contiguous()
    mask17 = mask19.index_select(1, indices).bool().contiguous()
    return (
        target17,
        mask17,
        pre_count,
        {
            "affected_patient_count": int((observed[:, fz] | observed[:, pz]).sum()),
            "observed_fz_positive_rows": int(observed[:, fz].sum()),
            "observed_pz_positive_rows": int(observed[:, pz].sum()),
            "preexisting_cz_positive_rows": int(observed[:, cz].sum()),
            "mapped_cz_positive_rows": int(target17[:, COMMON_17.index("CZ")].sum()),
            "mapping_collision_rows_deduplicated": int(
                (observed[:, cz] & (observed[:, fz] | observed[:, pz])).sum()
            ),
        },
    )


def _state_for_fold(
    payload: Mapping[str, torch.Tensor], fold: int
) -> dict[str, torch.Tensor]:
    prefix = f"{PRIMARY_ARM}.fold{fold}."
    result = {
        key[len(prefix) :]: value
        for key, value in payload.items()
        if key.startswith(prefix)
    }
    if not result:
        raise KeyError(f"missing saved state for fold {fold}")
    return result


def _new_initial_model(
    *,
    seed: int,
    targets: torch.Tensor,
    mask: torch.Tensor,
    train_patients: torch.Tensor,
) -> Common17EventSetReasoner:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Common17EventSetReasoner(
            input_dim=200,
            phase_count=5,
            latent_dim=32,
            prior_logits=_jeffreys_prior(
                targets.index_select(0, train_patients),
                mask.index_select(0, train_patients),
            ),
        )


def _training_and_split_audit(
    *,
    payload: Mapping[str, torch.Tensor],
    features: torch.Tensor,
    event_patient: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    folds: torch.Tensor,
    manifest: Mapping[str, Any],
    run_held_target_negative_control: bool,
) -> dict[str, Any]:
    patient_rows = _patient_event_rows(event_patient, 102)
    oof = payload[f"oof_probability.{PRIMARY_ARM}"].float()
    fold_results = []
    config = {
        **dict(manifest["training"]),
        "latent_dimension": int(manifest["training"]["latent_dimension"]),
    }
    for fold in range(5):
        train = torch.nonzero(folds != fold, as_tuple=False).flatten()
        held = torch.nonzero(folds == fold, as_tuple=False).flatten()
        if set(train.tolist()).intersection(held.tolist()):
            raise RuntimeError("patient split leakage")
        train_event_patients = set(
            event_patient[
                torch.cat([patient_rows[int(patient)] for patient in train.tolist()])
            ].tolist()
        )
        held_event_patients = set(
            event_patient[
                torch.cat([patient_rows[int(patient)] for patient in held.tolist()])
            ].tolist()
        )
        if train_event_patients.intersection(held_event_patients):
            raise RuntimeError("event carrier crosses train/held patients")

        seed = int(manifest["folds"][fold]["seed"])
        saved_state = _state_for_fold(payload, fold)
        initial = _new_initial_model(
            seed=seed,
            targets=targets,
            mask=mask,
            train_patients=train,
        )
        initial_state = initial.state_dict()
        if set(initial_state) != set(saved_state):
            raise RuntimeError("saved fold state keys differ from model state")
        changed = []
        squared_delta = 0.0
        trainable_names = {name for name, _ in initial.named_parameters()}
        for name in sorted(trainable_names):
            delta = saved_state[name].float() - initial_state[name].float()
            if not torch.equal(saved_state[name], initial_state[name]):
                changed.append(name)
            squared_delta += float((delta.double() ** 2).sum())
        trainable_parameters = sum(value.numel() for value in initial.parameters())
        if trainable_parameters != 6_967 or not changed or squared_delta <= 0.0:
            raise RuntimeError(
                "saved fold does not prove a trained 6,967-parameter head"
            )
        expected_prior = _jeffreys_prior(
            targets.index_select(0, train), mask.index_select(0, train)
        )
        if not torch.equal(saved_state["prior_logits"], expected_prior):
            raise RuntimeError("saved prior used patients outside the training fold")

        replay_model = _new_initial_model(
            seed=seed,
            targets=targets,
            mask=mask,
            train_patients=train,
        )
        replay_model.load_state_dict(saved_state, strict=True)
        replay_model.eval().requires_grad_(False)
        replay = _predict(
            model=replay_model,
            features=features,
            patient_rows=patient_rows,
            patients=held,
            device=torch.device("cpu"),
        )
        frozen = oof.index_select(0, held)
        if not torch.equal(replay, frozen):
            raise RuntimeError("fold-specific state does not bitwise replay held OOF")

        negative_control = None
        if run_held_target_negative_control:
            perturbed = targets.clone()
            perturbed[held] = torch.roll(perturbed[held], shifts=1, dims=1)
            refit, _ = _fit_fold(
                features=features,
                event_patient=event_patient,
                patient_rows=patient_rows,
                train_patients=train,
                targets=perturbed,
                mask=mask,
                config=config,
                seed=seed,
                device=torch.device("cpu"),
            )
            refit_state = refit.state_dict()
            bitwise = all(
                torch.equal(refit_state[name], saved_state[name])
                for name in saved_state
            )
            if not bitwise:
                raise RuntimeError(
                    "held-target perturbation changed a trained fold state"
                )
            negative_control = {
                "held_targets_cyclically_permuted": True,
                "refit_state_bitwise_identical_to_frozen": True,
            }
        fold_results.append(
            {
                "fold": fold,
                "seed": seed,
                "train_patients": len(train),
                "held_patients": len(held),
                "train_held_patient_overlap": 0,
                "train_held_event_patient_overlap": 0,
                "trainable_parameters": trainable_parameters,
                "changed_trainable_tensor_count": len(changed),
                "trainable_tensor_count": len(trainable_names),
                "changed_trainable_tensor_names": changed,
                "initial_to_fitted_l2": math.sqrt(squared_delta),
                "saved_training_prior_uses_train_patients_only": True,
                "held_oof_bitwise_replayed_from_fold_state": True,
                "held_target_negative_control": negative_control,
            }
        )
    return {
        "patient_count": 102,
        "event_count": len(event_patient),
        "fold_count": 5,
        "each_patient_has_exactly_one_held_fold": bool(
            torch.all((folds >= 0) & (folds < 5))
        ),
        "all_events_bind_one_patient": bool(
            torch.all((event_patient >= 0) & (event_patient < 102))
        ),
        "folds": fold_results,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    frozen_manifest_path = args.frozen / "manifest.json"
    frozen_tensor_path = args.frozen / "oof_predictions_and_states.safetensors"
    rerun_manifest_path = args.rerun / "manifest.json"
    rerun_tensor_path = args.rerun / "oof_predictions_and_states.safetensors"
    phase_manifest_path = args.phase / "manifest.json"
    phase_tensor_path = args.phase / "common17_car17_labram_phase.safetensors"
    target_tensor_path = args.target / "oof_predictions.safetensors"

    frozen_manifest = _read_json(frozen_manifest_path)
    rerun_manifest = _read_json(rerun_manifest_path)
    phase_manifest = _read_json(phase_manifest_path)
    if (
        frozen_manifest["primary_arm"] != PRIMARY_ARM
        or rerun_manifest["primary_arm"] != PRIMARY_ARM
    ):
        raise ValueError("audit input is not strict CAR17 primary")
    if _file_sha256(frozen_tensor_path) != _file_sha256(rerun_tensor_path):
        raise RuntimeError("independent retraining tensor is not bitwise identical")

    state_keys = tuple(
        key
        for key in _all_tensor_keys(frozen_tensor_path)
        if key.startswith(f"{PRIMARY_ARM}.fold")
    )
    base_keys = (
        "targets",
        "target_mask",
        "pre_mapping_positive_count",
        "patient_folds",
        "event_patient_index",
        "common17_standard19_indices",
        f"oof_probability.{PRIMARY_ARM}",
        *state_keys,
    )
    payload = _load(frozen_tensor_path, base_keys)
    target19 = _load(target_tensor_path, ("targets", "target_mask", "patient_folds"))
    phase = _load(phase_tensor_path, ("phase_features", "event_patient_index"))
    targets, mask, pre_count, mapping_stats = _independent_target_projection(
        target19["targets"].float(), target19["target_mask"].bool()
    )
    for name, expected, actual in (
        ("targets", targets, payload["targets"].float()),
        ("target_mask", mask, payload["target_mask"].bool()),
        (
            "pre_mapping_positive_count",
            pre_count,
            payload["pre_mapping_positive_count"].long(),
        ),
        (
            "patient_folds",
            target19["patient_folds"].long(),
            payload["patient_folds"].long(),
        ),
        (
            "event_patient_index",
            phase["event_patient_index"].long(),
            payload["event_patient_index"].long(),
        ),
    ):
        if not torch.equal(expected, actual):
            raise RuntimeError(f"independent carrier reconstruction failed for {name}")
    expected_indices = torch.tensor(
        [CHANNEL_INDEX[channel] for channel in COMMON_17], dtype=torch.long
    )
    if not torch.equal(payload["common17_standard19_indices"].long(), expected_indices):
        raise RuntimeError("saved common17 indices differ from frozen ontology")

    phase_keys = set(_all_tensor_keys(phase_tensor_path))
    if any("target" in key.lower() or "label" in key.lower() for key in phase_keys):
        raise RuntimeError("target/label tensor leaked into target-blind phase cache")
    access = phase_manifest.get("access_receipt", {})
    if (
        access.get("SOZ_targets_loaded") is not False
        or access.get("FZ_or_PZ_samples_loaded") is not False
    ):
        raise RuntimeError(
            "phase cache access receipt violates target/channel firewall"
        )

    graph = _induced_neighbor_graph()
    probability = payload[f"oof_probability.{PRIMARY_ARM}"].float()
    outcomes = metric_outcomes(probability, targets, pre_count, graph)
    observed19 = (target19["targets"].float() == 1) & target19["target_mask"].bool()
    before_mapping_targets = observed19.index_select(1, expected_indices).float()
    before_mapping_outcomes = metric_outcomes(
        probability, before_mapping_targets, pre_count, graph
    )
    mapping_effect = {
        name: {
            "before": int(before_mapping_outcomes[name].sum()),
            "after": int(outcomes[name].sum()),
            "delta": int(outcomes[name].sum() - before_mapping_outcomes[name].sum()),
            "denominator": len(probability),
        }
        for name in ("exact_top1", "deepsoz_N2", "deepsoz_N4")
    }
    bootstrap = patient_bootstrap_intervals(
        outcomes,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    folds = payload["patient_folds"].long()
    per_fold = _fold_metrics(outcomes, folds)

    manifest_metrics = frozen_manifest["metrics"][PRIMARY_ARM]
    frozen_values = {
        "exact_top1": float(manifest_metrics["exact_top1_accuracy"]),
        "accuracy": float(manifest_metrics["accuracy"]),
        "deepsoz_N2": float(manifest_metrics["deepsoz_N2"]["relaxed_top1"]),
        "deepsoz_N4": float(manifest_metrics["deepsoz_N4"]["relaxed_top1"]),
        "hit_at_3": float(manifest_metrics["hit_at_3"]),
        "hit_at_5": float(manifest_metrics["hit_at_5"]),
        "mrr": float(manifest_metrics["mrr"]),
    }
    if any(
        abs(frozen_values[name] - float(values.mean())) > 1e-7
        for name, values in outcomes.items()
    ):
        raise RuntimeError("independently recomputed metric differs from manifest")

    training = _training_and_split_audit(
        payload=payload,
        features=phase["phase_features"].float().contiguous(),
        event_patient=phase["event_patient_index"].long(),
        targets=targets,
        mask=mask,
        folds=folds,
        manifest=frozen_manifest,
        run_held_target_negative_control=args.run_held_target_negative_control,
    )
    graph_named = {
        channel: [COMMON_17[index] for index in row]
        for channel, row in zip(COMMON_17, graph)
    }
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass_independent_retraining_split_metric_bootstrap_audit",
        "analysis_role": "oracle_event_SOZ_localization_not_end_to_end_detection",
        "channel_contract": {
            "common17_channels": list(COMMON_17),
            "excluded_signal_channels": ["FZ", "PZ"],
            "prediction_side_fz_pz_to_cz_mapping_used": False,
            "phase_tensor_shape": list(phase["phase_features"].shape),
            "phase_cache_has_no_target_or_label_tensor": True,
            "phase_cache_loaded_fz_or_pz_samples": False,
        },
        "target_mapping": {
            "policy": "hard GT FZ/PZ OR into CZ, then delete FZ/PZ and deduplicate",
            "mapping_applies_to_ground_truth_only": True,
            "statistics": mapping_stats,
            "aggregate_success_counts_before_after": mapping_effect,
            "n2_n4_gate_uses_pre_mapping_positive_count": True,
        },
        "deepsoz_neighbor_tolerance": {
            "source": "published DeepSOZ standard19 directed one-hop table",
            "common17_operation": "induced_subgraph_delete_FZ_PZ_without_transitive_shortcuts",
            "named_directed_neighbors": graph_named,
            "N2_definition": "neighbor expansion only when pre-mapping hard-positive count <=2",
            "N4_definition": "neighbor expansion only when pre-mapping hard-positive count <=4",
            "not_top_k": True,
            "not_exact_electrode_accuracy": True,
        },
        "retraining": {
            "independent_rerun_performed": True,
            "frozen_tensor_sha256": _file_sha256(frozen_tensor_path),
            "independent_rerun_tensor_sha256": _file_sha256(rerun_tensor_path),
            "tensor_bitwise_identical": True,
            "foundation_encoder_frozen": True,
            "trained_component": "five_fold_specific_event_set_aggregation_heads",
            "trainable_parameters_per_fold": 6_967,
            "epochs_per_fold": int(frozen_manifest["training"]["epochs"]),
            "not_end_to_end_labram_finetuning": True,
        },
        "split_and_state_replay": training,
        "metric_definition": {
            "evaluation_unit": "patient held-out OOF row",
            "accuracy": "alias of exact_top1: predicted Top-1 belongs to the patient's hard-positive set",
            "accuracy_is_not_an_independent_second_endpoint": True,
            "deepsoz_N2_N4": "directed one-hop neighbor-tolerant Top-1 sensitivity endpoints with pre-mapping positive-count gates",
        },
        "metrics": {
            name: {
                "point_estimate": float(values.mean()),
                "successes": (None if name == "mrr" else int(values.sum())),
                "total": len(values),
                "patient_bootstrap_percentile_95_ci": bootstrap[name][
                    "patient_bootstrap_percentile_95_ci"
                ],
            }
            for name, values in outcomes.items()
        },
        "bootstrap": {
            "resampling_unit": "patient",
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "shared_resample_stream_across_metrics": True,
            "details": bootstrap,
        },
        "per_fold": per_fold,
        "lineage": {
            "frozen_manifest": {
                "path": str(frozen_manifest_path),
                "sha256": _file_sha256(frozen_manifest_path),
            },
            "frozen_tensor": {
                "path": str(frozen_tensor_path),
                "sha256": _file_sha256(frozen_tensor_path),
            },
            "rerun_manifest": {
                "path": str(rerun_manifest_path),
                "sha256": _file_sha256(rerun_manifest_path),
            },
            "rerun_tensor": {
                "path": str(rerun_tensor_path),
                "sha256": _file_sha256(rerun_tensor_path),
            },
            "phase_manifest": {
                "path": str(phase_manifest_path),
                "sha256": _file_sha256(phase_manifest_path),
            },
            "phase_tensor": {
                "path": str(phase_tensor_path),
                "sha256": _file_sha256(phase_tensor_path),
            },
            "target_tensor": {
                "path": str(target_tensor_path),
                "sha256": _file_sha256(target_tensor_path),
            },
            "audit_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": _file_sha256(Path(__file__).resolve()),
            },
        },
        "access_receipt": {
            "public_eeg_derived_features_loaded": True,
            "public_targets_loaded_for_training_split_audit": True,
            "private_eeg_or_labels_loaded": False,
            "edf_annotations_loaded": False,
            "excel_or_doctor_text_loaded": False,
            "raw_eeg_loaded": False,
        },
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--rerun", type=Path, default=DEFAULT_RERUN)
    parser.add_argument("--phase", type=Path, default=DEFAULT_PHASE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    parser.add_argument(
        "--run-held-target-negative-control",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    receipt = run(args)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output),
                "exact_top1": receipt["metrics"]["exact_top1"],
                "deepsoz_N2": receipt["metrics"]["deepsoz_N2"],
                "deepsoz_N4": receipt["metrics"]["deepsoz_N4"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
