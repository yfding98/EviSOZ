#!/usr/bin/env python3
"""Run target-free patient-OOF LaBraM temporal-future qualification v1."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

# Required by torch deterministic algorithms for CUDA >= 10.2.  This must be
# set before the first CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from safetensors.torch import load_file, save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.concept_losses import temporal_evolution_loss  # noqa: E402
from src.soz.concept_token_io import load_labram_concept_tokens  # noqa: E402
from src.soz.models.concept_heads import TemporalEvolutionHead  # noqa: E402
from src.soz.temporal_future_qualification import (  # noqa: E402
    bootstrap_paired_difference,
    encode_patient_ids,
    fit_patient_balanced_linear_ar,
    fit_patient_balanced_time_only,
    future_targets_and_mask,
    patient_macro_feature_smooth_l1,
    patient_macro_smooth_l1,
    predict_linear_ar,
    scaler_from_receipt,
    temporal_predictability_decision,
)


CONFIG_SCHEMA = "soz_labram_temporal_future_qualification_config_v1"
RESULT_SCHEMA = "soz_labram_temporal_future_qualification_result_v1"
MANIFEST_SCHEMA = "soz_labram_temporal_future_qualification_manifest_v1"
EXPECTED_FEATURES = (
    "log_rms",
    "log_line_length",
    "spectral_centroid",
    "normalized_spectral_entropy",
    "rhythmicity",
    "mean_neighbor_coherence",
)


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(_canonical_json(payload) + b"\n")


def _bound_file(record: Mapping[str, object], *, label: str) -> Path:
    path_text = record.get("path")
    expected = record.get("sha256")
    if not isinstance(path_text, str) or not isinstance(expected, str):
        raise TypeError(f"{label} path/sha256 binding is invalid")
    path = (ROOT / path_text).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    actual = _file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected}, actual={actual}")
    return path


def _validate_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = _load_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unexpected temporal-future config schema")
    sources = config.get("source_files")
    code = config.get("code_files")
    if not isinstance(sources, dict) or not isinstance(code, dict):
        raise TypeError("config source_files/code_files must be objects")
    paths = {
        name: _bound_file(record, label=f"source_files.{name}")
        for name, record in sources.items()
        if isinstance(record, Mapping)
    }
    code_paths = {
        f"code:{name}": _bound_file(record, label=f"code_files.{name}")
        for name, record in code.items()
        if isinstance(record, Mapping)
    }
    if len(paths) != len(sources) or len(code_paths) != len(code):
        raise TypeError("every source/code binding must be a JSON object")
    paths.update(code_paths)
    required = {
        "protocol",
        "token_index",
        "ictal_master_receipt",
        "vaq_manifest",
        "vaq_events",
        "vaq_tensor",
        *(f"scaler_fold_{fold}" for fold in range(5)),
    }
    missing = sorted(required - set(paths))
    if missing:
        raise ValueError(f"config is missing bound source files: {missing}")
    return config, paths


def _safety_contract(vaq_manifest: Mapping[str, object]) -> dict[str, bool]:
    checks = {
        "contains_soz_labels_false": vaq_manifest.get("contains_soz_labels") is False,
        "target_vectors_loaded_false": vaq_manifest.get("target_vectors_loaded") is False,
        "private_events_used_false": vaq_manifest.get("private_events_used") is False,
        "source_dev_events_used_false": vaq_manifest.get("source_dev_events_used") is False,
        "source_eval_events_used_false": vaq_manifest.get("source_eval_events_used") is False,
        "formal_promotion_authorized_false": vaq_manifest.get("formal_promotion_authorized")
        is False,
    }
    if not all(checks.values()):
        raise ValueError(f"target-free VAQ safety contract failed: {checks}")
    return checks


def _event_key(relative_path: object, onset: object) -> tuple[str, int]:
    if not isinstance(relative_path, str) or not isinstance(onset, (int, float)):
        raise TypeError("event path/onset has an invalid type")
    return relative_path, int(round(float(onset) * 10_000_000.0))


def _crosswalk_token_bundles(
    vaq_events: Sequence[Mapping[str, object]],
    master_receipt: Mapping[str, object],
    token_index: Mapping[str, object],
    token_root: Path,
) -> list[tuple[Path, str, str]]:
    master_events = master_receipt.get("events")
    token_events = token_index.get("events")
    if not isinstance(master_events, list) or not isinstance(token_events, list):
        raise TypeError("master/token index lacks event arrays")
    by_key: dict[tuple[str, int], Mapping[str, object]] = {}
    for row in master_events:
        if not isinstance(row, Mapping):
            raise TypeError("master event row must be an object")
        key = _event_key(row.get("relative_edf_path"), row.get("event_t0_sec"))
        if key in by_key:
            raise ValueError(f"ambiguous master path/onset key: {key}")
        by_key[key] = row
    indexed: dict[str, Mapping[str, object]] = {}
    for row in token_events:
        if not isinstance(row, Mapping) or not isinstance(row.get("event_id"), str):
            raise TypeError("token-index event row is invalid")
        indexed[str(row["event_id"])] = row
    output: list[tuple[Path, str, str]] = []
    seen_global: set[str] = set()
    for event in vaq_events:
        key = _event_key(event.get("relative_edf_path"), event.get("global_t0_sec"))
        master = by_key.get(key)
        if master is None:
            raise ValueError(f"VAQ event has no exact TUSZ master match: {key}")
        if master.get("edf_sha256") != event.get("edf_sha256"):
            raise ValueError(f"EDF hash mismatch for crosswalk key: {key}")
        global_id = master.get("event_id")
        if not isinstance(global_id, str) or global_id in seen_global:
            raise ValueError("token crosswalk is not one-to-one")
        seen_global.add(global_id)
        token = indexed.get(global_id)
        if token is None:
            raise ValueError(f"TUSZ master event is absent from token index: {global_id}")
        bundle = token_root / str(token.get("bundle_path"))
        manifest_sha = token.get("bundle_manifest_sha256")
        tensor_sha = token.get("tensor_sha256")
        if not isinstance(manifest_sha, str) or not isinstance(tensor_sha, str):
            raise TypeError("token-index hashes are invalid")
        output.append((bundle, manifest_sha, tensor_sha))
    return output


def _smoke_indices(events: Sequence[Mapping[str, object]]) -> list[int]:
    selected: list[int] = []
    for fold in range(5):
        patients = sorted(
            {
                str(row["patient_id"])
                for row in events
                if int(row["oof_fold"]) == fold
            }
        )[:2]
        for patient in patients:
            candidates = [
                index
                for index, row in enumerate(events)
                if str(row["patient_id"]) == patient
            ][:2]
            selected.extend(candidates)
    result = sorted(set(selected))
    if len(result) < 10:
        raise ValueError("smoke roster did not retain enough patients/events")
    return result


def _load_tokens(
    bundles: Sequence[tuple[Path, str, str]], indices: Sequence[int]
) -> tuple[torch.Tensor, list[str]]:
    rows: list[torch.Tensor] = []
    hashes: list[str] = []
    for position, source_index in enumerate(indices, start=1):
        bundle, manifest_sha, tensor_sha = bundles[source_index]
        loaded = load_labram_concept_tokens(
            bundle, expected_manifest_sha256=manifest_sha
        )
        if loaded.tensor_sha256 != tensor_sha:
            raise ValueError("token tensor hash differs from bound corpus index")
        if tuple(loaded.tokens.shape) != (19, 60, 200):
            raise ValueError("LaBraM token shape changed")
        rows.append(loaded.tokens)
        hashes.append(loaded.tensor_sha256)
        if position % 100 == 0:
            print(f"validated and loaded {position}/{len(indices)} token bundles", flush=True)
    return torch.stack(rows), hashes


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_one_fold(
    *,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    patient_ids: torch.Tensor,
    train_indices: torch.Tensor,
    held_indices: torch.Tensor,
    seed: int,
    fold: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    future_weight: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    _set_seed(seed)
    model = TemporalEvolutionHead(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_patients = patient_ids.index_select(0, train_indices)
    counts = {
        int(patient): int((train_patients == patient).sum().item())
        for patient in torch.unique(train_patients, sorted=True)
    }
    weights = torch.tensor(
        [1.0 / counts[int(value)] for value in train_patients.tolist()],
        dtype=torch.float64,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 10_000 * fold)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        sampled_local = torch.multinomial(
            weights,
            num_samples=max(len(train_indices), batch_size),
            replacement=True,
            generator=generator,
        )
        sampled = train_indices.index_select(0, sampled_local)
        totals = []
        descriptors = []
        futures = []
        for start in range(0, len(sampled), batch_size):
            batch = sampled[start : start + batch_size]
            batch_tokens = tokens.index_select(0, batch).to(device, non_blocking=True)
            batch_targets = targets.index_select(0, batch).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            batch_mask = mask.index_select(0, batch).to(device, non_blocking=True)
            batch_patients = patient_ids.index_select(0, batch).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_tokens)
            losses = temporal_evolution_loss(
                output,
                batch_targets,
                batch_mask,
                batch_patients,
                future_weight=future_weight,
            )
            losses.total.backward()
            optimizer.step()
            totals.append(float(losses.total.detach().cpu()))
            descriptors.append(float(losses.descriptor.detach().cpu()))
            futures.append(float(losses.future_change.detach().cpu()))
        history.append(
            {
                "epoch": float(epoch + 1),
                "total": float(np.mean(totals)),
                "descriptor": float(np.mean(descriptors)),
                "future": float(np.mean(futures)),
            }
        )
    model.eval()
    descriptor_rows = []
    future_rows = []
    with torch.no_grad():
        for start in range(0, len(held_indices), batch_size):
            batch = held_indices[start : start + batch_size]
            output = model(tokens.index_select(0, batch).to(device, non_blocking=True))
            descriptor_rows.append(output.descriptors.detach().cpu())
            future_rows.append(output.future_change.detach().cpu())
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    return state, torch.cat(descriptor_rows), torch.cat(future_rows), history


def _patient_metrics(
    prediction: torch.Tensor,
    descriptor_prediction: torch.Tensor,
    target_delta: torch.Tensor,
    descriptor_target: torch.Tensor,
    future_mask: torch.Tensor,
    descriptor_mask: torch.Tensor,
    patient_ids: torch.Tensor,
) -> dict[str, object]:
    future_loss, patient_future = patient_macro_smooth_l1(
        prediction, target_delta, future_mask, patient_ids
    )
    descriptor_loss, patient_descriptor = patient_macro_smooth_l1(
        descriptor_prediction,
        descriptor_target,
        descriptor_mask,
        patient_ids,
    )
    return {
        "future_smooth_l1": future_loss,
        "future_feature_smooth_l1": patient_macro_feature_smooth_l1(
            prediction, target_delta, future_mask, patient_ids
        ),
        "descriptor_smooth_l1": descriptor_loss,
        "descriptor_feature_smooth_l1": patient_macro_feature_smooth_l1(
            descriptor_prediction,
            descriptor_target,
            descriptor_mask,
            patient_ids,
        ),
        "patient_future": {str(key): value for key, value in patient_future.items()},
        "patient_descriptor": {
            str(key): value for key, value in patient_descriptor.items()
        },
    }


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def run(config_path: Path, output: Path, *, smoke: bool, device_name: str) -> dict[str, object]:
    config_path = config_path.resolve(strict=True)
    config, paths = _validate_config(config_path)
    config_sha = _file_sha256(config_path)
    output = output.absolute()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    temporary = output.parent / f".{output.name}.tmp_{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    (temporary / "checkpoints").mkdir()

    token_index = _load_json(paths["token_index"])
    master_receipt = _load_json(paths["ictal_master_receipt"])
    vaq_manifest = _load_json(paths["vaq_manifest"])
    vaq_events_payload = _load_json(paths["vaq_events"])
    safety = _safety_contract(vaq_manifest)
    if token_index.get("purpose") != "ictal_concept_training_only":
        raise ValueError("source token purpose changed")
    foundation = token_index.get("foundation")
    if not isinstance(foundation, Mapping) or foundation.get("frozen") is not True:
        raise ValueError("token corpus must come from a frozen foundation encoder")
    events = vaq_events_payload.get("events")
    if not isinstance(events, list) or not all(isinstance(row, dict) for row in events):
        raise TypeError("VAQ event file is invalid")
    if len(events) != int(vaq_manifest.get("event_count", -1)):
        raise ValueError("VAQ event count mismatch")
    token_root = paths["token_index"].parent
    bundles = _crosswalk_token_bundles(events, master_receipt, token_index, token_root)

    tensors = load_file(str(paths["vaq_tensor"]), device="cpu")
    required_tensors = {
        "evolution_raw",
        "evolution_scaled",
        "evolution_mask",
        "artifact_burden",
    }
    if not required_tensors <= set(tensors):
        raise ValueError("VAQ tensor bundle lacks required target-free tensors")
    raw_all = tensors["evolution_raw"].to(torch.float64)
    scaled_oof_all = tensors["evolution_scaled"].to(torch.float64)
    mask_all = tensors["evolution_mask"].to(torch.bool)
    artifact_all = tensors["artifact_burden"].to(torch.float64)
    if tuple(raw_all.shape) != (len(events), 19, 15, 6):
        raise ValueError("VAQ evolution tensor shape changed")
    if not mask_all.all():
        raise ValueError("v1 requires the frozen complete-19 all-true evolution mask")

    indices = _smoke_indices(events) if smoke else list(range(len(events)))
    selected_events = [events[index] for index in indices]
    selected_tokens, token_hashes = _load_tokens(bundles, indices)
    raw = raw_all.index_select(0, torch.tensor(indices)).contiguous()
    scaled_oof = scaled_oof_all.index_select(0, torch.tensor(indices)).contiguous()
    mask = mask_all.index_select(0, torch.tensor(indices)).contiguous()
    artifact = artifact_all.index_select(0, torch.tensor(indices)).contiguous()
    folds = torch.tensor([int(row["oof_fold"]) for row in selected_events], dtype=torch.long)
    patient_names = [str(row["patient_id"]) for row in selected_events]
    patient_ids, patient_mapping = encode_patient_ids(patient_names)
    if set(folds.tolist()) != set(range(5)):
        raise ValueError("selected roster must contain all five OOF folds")

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("config training section is invalid")
    seeds = [int(value) for value in training["seeds"]]
    epochs = 1 if smoke else int(training["epochs"])
    batch_size = 4 if smoke else int(training["batch_size"])
    hidden_dim = int(training["hidden_dim"])
    learning_rate = float(training["learning_rate"])
    weight_decay = float(training["weight_decay"])
    future_weight = float(training["future_weight"])
    ridge = float(training["linear_ar_ridge"])
    if smoke:
        seeds = seeds[:1]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be unique and nonempty")

    device = _resolve_device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(training.get("cpu_threads", 4)))
    print(
        f"V_F run: events={len(indices)} patients={len(patient_mapping)} "
        f"seeds={seeds} epochs={epochs} device={device}",
        flush=True,
    )

    target_oof = torch.empty_like(raw)
    linear_oof = torch.empty((len(indices), 19, 14, 6), dtype=torch.float64)
    time_oof = torch.empty_like(linear_oof)
    scaler_replay_max_abs = 0.0
    fold_records: dict[str, object] = {}
    fold_targets: dict[int, torch.Tensor] = {}
    fold_indices: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for fold in range(5):
        held = torch.nonzero(folds == fold, as_tuple=False).flatten()
        train = torch.nonzero(folds != fold, as_tuple=False).flatten()
        if held.numel() == 0 or train.numel() == 0:
            raise ValueError(f"fold {fold} lacks train or held events")
        scaler_payload = _load_json(paths[f"scaler_fold_{fold}"])
        scaler = scaler_from_receipt(scaler_payload)
        scaled = scaler.transform(raw)
        replay_error = float((scaled.index_select(0, held) - scaled_oof.index_select(0, held)).abs().max())
        scaler_replay_max_abs = max(scaler_replay_max_abs, replay_error)
        target_oof.index_copy_(0, held, scaled.index_select(0, held))
        coefficients = fit_patient_balanced_linear_ar(
            scaled, mask, patient_ids, train, ridge=ridge
        )
        linear_oof.index_copy_(
            0, held, predict_linear_ar(scaled.index_select(0, held), coefficients)
        )
        time_mean = fit_patient_balanced_time_only(scaled, mask, patient_ids, train)
        time_prediction = time_mean.view(1, 1, 14, 6).expand(len(held), 19, 14, 6)
        time_oof.index_copy_(0, held, time_prediction)
        fold_targets[fold] = scaled
        fold_indices[fold] = (train, held)
        fold_records[str(fold)] = {
            "train_events": len(train),
            "held_events": len(held),
            "train_patients": len(set(patient_ids.index_select(0, train).tolist())),
            "held_patients": len(set(patient_ids.index_select(0, held).tolist())),
            "scaler_replay_max_abs": replay_error,
            "linear_ar_coefficients": coefficients.tolist(),
            "time_only_mean": time_mean.tolist(),
        }
    scaler_replay_exact = scaler_replay_max_abs <= 1e-10
    if not scaler_replay_exact:
        raise ValueError(
            f"fold scaler replay differs from frozen OOF target: {scaler_replay_max_abs}"
        )

    seed_descriptor_predictions: list[torch.Tensor] = []
    seed_future_predictions: list[torch.Tensor] = []
    seed_metrics: dict[str, object] = {}
    for seed in seeds:
        descriptor_oof = torch.empty((len(indices), 19, 15, 6), dtype=torch.float32)
        future_oof = torch.empty((len(indices), 19, 14, 6), dtype=torch.float32)
        seed_history: dict[str, object] = {}
        for fold in range(5):
            train, held = fold_indices[fold]
            state, descriptor_prediction, future_prediction, history = _train_one_fold(
                tokens=selected_tokens,
                targets=fold_targets[fold],
                mask=mask,
                patient_ids=patient_ids,
                train_indices=train,
                held_indices=held,
                seed=seed,
                fold=fold,
                hidden_dim=hidden_dim,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                future_weight=future_weight,
                device=device,
            )
            descriptor_oof.index_copy_(0, held, descriptor_prediction)
            future_oof.index_copy_(0, held, future_prediction)
            checkpoint = temporary / "checkpoints" / f"seed_{seed}_fold_{fold}.safetensors"
            save_file(state, str(checkpoint))
            seed_history[str(fold)] = history
            print(
                f"seed={seed} fold={fold} complete "
                f"final_total={history[-1]['total']:.6f}",
                flush=True,
            )
        target_delta, future_mask = future_targets_and_mask(target_oof, mask)
        metrics = _patient_metrics(
            future_oof.to(torch.float64),
            descriptor_oof.to(torch.float64),
            target_delta,
            target_oof,
            future_mask,
            mask,
            patient_ids,
        )
        metrics["history"] = seed_history
        seed_metrics[str(seed)] = metrics
        seed_descriptor_predictions.append(descriptor_oof)
        seed_future_predictions.append(future_oof)

    descriptor_ensemble = torch.stack(seed_descriptor_predictions).mean(dim=0).to(torch.float64)
    future_ensemble = torch.stack(seed_future_predictions).mean(dim=0).to(torch.float64)
    target_delta, future_mask = future_targets_and_mask(target_oof, mask)
    ensemble_metrics = _patient_metrics(
        future_ensemble,
        descriptor_ensemble,
        target_delta,
        target_oof,
        future_mask,
        mask,
        patient_ids,
    )
    zeros = torch.zeros_like(target_delta)
    persistence_loss, persistence_patient = patient_macro_smooth_l1(
        zeros, target_delta, future_mask, patient_ids
    )
    linear_loss, linear_patient = patient_macro_smooth_l1(
        linear_oof, target_delta, future_mask, patient_ids
    )
    time_loss, time_patient = patient_macro_smooth_l1(
        time_oof, target_delta, future_mask, patient_ids
    )
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(int(config["evaluation"]["shuffle_seed"]))
    permutation = torch.randperm(target_delta.shape[2], generator=shuffle_generator)
    if torch.equal(permutation, torch.arange(target_delta.shape[2])):
        permutation = permutation.roll(1)
    shuffled_target = target_delta.index_select(2, permutation)
    shuffled_mask = future_mask.index_select(2, permutation)
    shuffled_loss, shuffled_patient = patient_macro_smooth_l1(
        future_ensemble, shuffled_target, shuffled_mask, patient_ids
    )
    ensemble_patient = {
        int(key): float(value)
        for key, value in ensemble_metrics["patient_future"].items()
    }
    replicates = 200 if smoke else int(config["evaluation"]["bootstrap_replicates"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    versus_persistence = bootstrap_paired_difference(
        ensemble_patient,
        persistence_patient,
        replicates=replicates,
        seed=bootstrap_seed,
    )
    versus_linear = bootstrap_paired_difference(
        ensemble_patient,
        linear_patient,
        replicates=replicates,
        seed=bootstrap_seed + 1,
    )
    coverage_complete = len(indices) == len(events) and len(patient_mapping) == 65
    passed, gates = temporal_predictability_decision(
        ensemble_loss=float(ensemble_metrics["future_smooth_l1"]),
        persistence_loss=persistence_loss,
        linear_ar_loss=linear_loss,
        shuffled_loss=shuffled_loss,
        versus_persistence=versus_persistence,
        versus_linear_ar=versus_linear,
        coverage_complete=coverage_complete,
        scaler_replay_exact=scaler_replay_exact,
        safety_passed=all(safety.values()),
    )
    if smoke:
        passed = False
        gates["formal_full_roster"] = False

    future_artifact = torch.maximum(artifact[:, :, :-1], artifact[:, :, 1:])
    observed_artifact = future_artifact[future_mask]
    quantiles = torch.quantile(observed_artifact, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    element_future = torch.nn.functional.smooth_l1_loss(
        future_ensemble, target_delta, reduction="none"
    ).mean(dim=-1)
    strata = []
    boundaries = [float("-inf"), *quantiles.tolist(), float("inf")]
    for index in range(4):
        stratum_mask = (
            future_mask
            & (future_artifact > boundaries[index])
            & (future_artifact <= boundaries[index + 1])
        )
        stratum_count = int(stratum_mask.sum())
        strata.append(
            {
                "quartile": index + 1,
                "count": stratum_count,
                "smooth_l1": (
                    float(element_future[stratum_mask].mean())
                    if stratum_count
                    else None
                ),
            }
        )

    parameter_count = sum(
        value.numel()
        for value in TemporalEvolutionHead(hidden_dim=hidden_dim).parameters()
    )
    result: dict[str, object] = {
        "schema_version": RESULT_SCHEMA,
        "status": "SMOKE_COMPLETED" if smoke else ("PASS" if passed else "NO_GO"),
        "qualified_scope": (
            "none_smoke_only"
            if smoke
            else (
                "target_free_temporal_predictability_only"
                if passed
                else "none_failed_target_free_temporal_predictability"
            )
        ),
        "claim_boundary": (
            "observable future descriptor change; not propagation, origin, onset channel, "
            "SOZ, or clinical artifact calibration"
        ),
        "config_sha256": config_sha,
        "execution": {
            "device": str(device),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "smoke": smoke,
            "event_count": len(indices),
            "patient_count": len(patient_mapping),
            "fold_event_counts": {
                str(fold): int((folds == fold).sum()) for fold in range(5)
            },
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "parameter_count": parameter_count,
        },
        "safety": {
            **safety,
            "tusz_involvement_targets_loaded": False,
            "soz_targets_loaded": False,
            "private_data_loaded": False,
            "source_token_purpose": token_index["purpose"],
            "read_only_purpose_extension": "target_free_temporal_future_qualification_only",
        },
        "folds": fold_records,
        "scaler_replay_max_abs": scaler_replay_max_abs,
        "seed_metrics": seed_metrics,
        "primary_ensemble": ensemble_metrics,
        "comparators": {
            "persistence_future_smooth_l1": persistence_loss,
            "linear_ar_future_smooth_l1": linear_loss,
            "time_only_future_smooth_l1": time_loss,
            "shuffled_next_future_smooth_l1": shuffled_loss,
            "shuffled_tile_permutation": permutation.tolist(),
        },
        "paired_bootstrap": {
            "replicates": replicates,
            "versus_persistence": versus_persistence,
            "versus_linear_ar": versus_linear,
        },
        "artifact_burden_quartiles_descriptive": strata,
        "gates": gates,
        "downstream_authorization": {
            "current_soz_ranker_changed": False,
            "soz_reasoner_ingestion_authorized": False,
            "clinical_propagation_wording_authorized": False,
            "complete_three_concept_claim_authorized": False,
        },
    }

    prediction_path = temporary / "oof_predictions.safetensors"
    save_file(
        {
            "descriptor_ensemble": descriptor_ensemble.to(torch.float32).contiguous(),
            "future_ensemble": future_ensemble.to(torch.float32).contiguous(),
            "descriptor_target": target_oof.to(torch.float32).contiguous(),
            "future_target": target_delta.to(torch.float32).contiguous(),
            "future_mask": future_mask.contiguous(),
            "linear_ar": linear_oof.to(torch.float32).contiguous(),
            "time_only": time_oof.to(torch.float32).contiguous(),
            "fold": folds.contiguous(),
            "patient_id": patient_ids.contiguous(),
        },
        str(prediction_path),
    )
    result_path = temporary / "result.json"
    _write_json(result_path, result)
    checkpoint_files = sorted((temporary / "checkpoints").glob("*.safetensors"))
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result_file": "result.json",
        "result_sha256": _file_sha256(result_path),
        "prediction_file": "oof_predictions.safetensors",
        "prediction_sha256": _file_sha256(prediction_path),
        "checkpoints": [
            {
                "path": f"checkpoints/{path.name}",
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in checkpoint_files
        ],
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": config_sha,
        "source_file_hashes": {
            name: _file_sha256(path)
            for name, path in paths.items()
            if not name.startswith("code:")
        },
        "code_file_hashes": {
            name.removeprefix("code:"): _file_sha256(path)
            for name, path in paths.items()
            if name.startswith("code:")
        },
        "token_tensor_roster_sha256": _sha256_bytes(_canonical_json(token_hashes)),
        "event_roster_sha256": _sha256_bytes(
            _canonical_json([row["event_id"] for row in selected_events])
        ),
        "patient_mapping": patient_mapping,
    }
    _write_json(temporary / "manifest.json", manifest)
    temporary.rename(output)
    print(
        f"completed: status={result['status']} output={output} "
        f"future_loss={ensemble_metrics['future_smooth_l1']:.6f}",
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args.config, args.output, smoke=bool(args.smoke), device_name=str(args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
