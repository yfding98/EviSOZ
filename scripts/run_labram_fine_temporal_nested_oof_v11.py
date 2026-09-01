#!/usr/bin/env python3
"""Run v11-A LaBraM + fine temporal evidence developmental nested OOF."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.deepsoz_target_v2 import (  # noqa: E402
    TARGET_V2_POLICY_SHA256,
    load_verified_deepsoz_target_v2_artifact,
)
from src.soz.fine_temporal_evidence import FINE_TEMPORAL_FEATURE_NAMES  # noqa: E402
from src.soz.metrics import (  # noqa: E402
    DEEPSOZ_STANDARD19_NEIGHBORS,
    deepsoz_style_top1_metrics,
    patient_localization_metrics,
)
from src.soz.v11_development_union import (  # noqa: E402
    EXPECTED_PUBLIC_DEVELOPMENT_UNION_MANIFEST_SHA256,
    load_public_development_union,
)
from src.soz.v11_reasoner import (  # noqa: E402
    FoldFeatureTransform,
    SharedPositiveSetReasoner,
    TransformedPatientFeatures,
    extract_block9_phase_contrasts,
    fit_fold_transform,
    jeffreys_reference_prior_logits,
    positive_set_mass_loss,
    robust_pool_complete_patient_bags,
)


PROTOCOL_PATH = (
    ROOT
    / "research/02_method/labram_fine_temporal_development_union_protocol_v11_20260811_zh.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "67c015240a39c57d0a3714f2b0eb448fbb72c1363d91d014a1f64fb20861e380"
)
DEFAULT_UNION = ROOT / "outputs/public_development_union_v11_20260811"
DEFAULT_FINE = ROOT / "outputs/public_development_fine_evidence_v11_20260811"
DEFAULT_PREFIX = ROOT / "outputs/public_development_labram_prefix_v11_20260811"
DEFAULT_TARGET = ROOT / "outputs/deepsoz_target_v2"
DEFAULT_SOURCE = (
    ROOT / "outputs/deepsoz_llm_tusz_all_607_20260801/source/TUH_manifest_final.csv"
)
DEFAULT_SPLIT = ROOT / "outputs/deepsoz_tusz_patient_splits_v1/split_manifest.csv"
DEFAULT_OUTPUT = ROOT / "outputs/labram_fine_temporal_nested_oof_v11_20260811"
DEFAULT_ANCHOR = ROOT / "outputs/labram_temporal_mil_nested_oof_v1_20260810"

EXPECTED_FINE_MANIFEST_SHA256 = (
    "60ce6c5af15dcff3a0c0dcbac1451f4d5cb3bb28e7b9c22180ab7adecfb417a2"
)
EXPECTED_FINE_TENSOR_FILE_SHA256 = (
    "24dc5da224c79446992cde08d800877ff1ea4349d217c225da95588c9e173bbb"
)
EXPECTED_PREFIX_MANIFEST_SHA256 = (
    "b3ce8913a33848b7a706f8b30ccedf09ad8b2f6ae27412b1ae56d187866ff71f"
)
EXPECTED_PREFIX_TENSOR_FILE_SHA256 = (
    "40396fabac11ead6ac870ee69f428951f0577445c291a45b58e37c8fc6bf12bc"
)
EXPECTED_TARGET_ARTIFACT_SHA256 = (
    "5c01591c20328fb60817099cac669032bd743e36f47df77ac390842e9a2c67ed"
)
EXPECTED_TARGET_SUMMARY_SHA256 = (
    "1def41d4af3b3446db8a64cac1db658eff9c32c574e838e3a3b8e9b1bb93ec39"
)
EXPECTED_TARGET_README_SHA256 = (
    "e8b88190b0c8b10f05f2a67ffe572aa64b3c4ee47d61b2a1ed01b95aa1520196"
)
EXPECTED_TARGET_RECEIPT_SHA256 = (
    "80f2b71cfdf23d604849b2d1a52cc36f0b01c593906e3cef74e79d425cc442d3"
)
EXPECTED_SOURCE_SHA256 = (
    "4d08552dbb94f1e8e8a3931249d2bd29538233e2282b8d21a39d0f5dd873fd5c"
)
EXPECTED_SPLIT_SHA256 = (
    "5062e894ec139ffaf7abc1b8f45b326f50a118cfcb8907bb25ff81dbbaa91d57"
)

SCHEMA = "soz_labram_fine_temporal_nested_oof_v11"
ARMS = {
    "fine_change_only": (False, True),
    "frozen_labram_only": (True, False),
    "full_frozen_labram_plus_fine": (True, True),
}
L2_CANDIDATES = (0.01, 0.05, 0.20)
OUTER_FOLDS = tuple(range(5))
INNER_FOLDS = tuple(range(4))
LBFGS_MAX_ITER = 100
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260811


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_manifest(path: Path, *, expected_sha: str) -> dict[str, object]:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha:
        raise ValueError(f"manifest SHA mismatch for {path}: {actual}")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or raw != _canonical_bytes(payload, newline=True):
        raise ValueError(f"manifest is not canonical JSON: {path}")
    return payload


def _require_target_free_cache(manifest: Mapping[str, object], *, label: str) -> None:
    access = manifest.get("access_receipt")
    if not isinstance(access, Mapping):
        raise TypeError(f"{label} lacks an access receipt")
    forbidden = (
        "deepsoz_target_values_loaded",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "historical_prediction_artifacts_loaded",
    )
    if any(access.get(key) is not False for key in forbidden):
        raise ValueError(f"{label} is not target/private-free")
    if manifest.get("event_count") != 988 or manifest.get("patient_count") != 102:
        raise ValueError(f"{label} does not cover the complete developmental union")


def _evaluate(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, object]:
    ranking = asdict(
        patient_localization_metrics(logits, targets, target_mask, k_values=(1, 3, 5))
    )
    top1 = asdict(deepsoz_style_top1_metrics(logits, targets, target_mask))
    return {
        "ranking": ranking,
        "top1": top1,
        "far_error_count": _far_error_count(logits, targets, target_mask),
    }


def _far_error_count(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> int:
    count = 0
    for patient in range(logits.shape[0]):
        observed = target_mask[patient]
        observed_indices = torch.nonzero(observed, as_tuple=False).flatten()
        predicted = int(observed_indices[logits[patient, observed].argmax()].item())
        positive = observed & (targets[patient] == 1)
        accepted = positive.clone()
        if int(positive.sum().item()) <= 4:
            for index in torch.nonzero(positive, as_tuple=False).flatten().tolist():
                accepted[list(DEEPSOZ_STANDARD19_NEIGHBORS[index])] = True
            accepted &= observed
        if not bool(accepted[predicted]):
            count += 1
    return count


def _patient_contributions(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    rows = {name: [] for name in ("strict", "relaxed", "macro_ap", "mrr")}
    for patient in range(logits.shape[0]):
        metrics = _evaluate(
            logits[patient : patient + 1],
            targets[patient : patient + 1],
            target_mask[patient : patient + 1],
        )
        rows["strict"].append(metrics["top1"]["strict_accuracy"])
        rows["relaxed"].append(metrics["top1"]["relaxed_accuracy"])
        rows["macro_ap"].append(metrics["ranking"]["macro_average_precision"])
        rows["mrr"].append(metrics["ranking"]["mean_reciprocal_rank"])
    return {
        name: torch.tensor(values, dtype=torch.float64)
        for name, values in rows.items()
    }


def _paired_bootstrap(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, object]:
    cand = _patient_contributions(candidate, targets, target_mask)
    base = _patient_contributions(baseline, targets, target_mask)
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    indices = torch.randint(
        0,
        candidate.shape[0],
        (BOOTSTRAP_REPLICATES, candidate.shape[0]),
        generator=generator,
    )
    result = {}
    for name in cand:
        difference = cand[name] - base[name]
        samples = difference[indices].mean(dim=1)
        result[name] = {
            "delta": float(difference.mean()),
            "ci95": [
                float(torch.quantile(samples, 0.025)),
                float(torch.quantile(samples, 0.975)),
            ],
        }
    return result


def _inner_assignments(
    outer_train_indices: Sequence[int],
    *,
    patient_ids: Sequence[str],
    event_counts: torch.Tensor,
    outer_fold: int,
) -> dict[int, int]:
    fold_events = [0] * len(INNER_FOLDS)
    fold_patients = [0] * len(INNER_FOLDS)
    assignment = {}
    salt = f"v11-inner|outer={outer_fold}|20260811"
    ordered = sorted(
        (int(index) for index in outer_train_indices),
        key=lambda index: (
            -int(event_counts[index]),
            hashlib.sha256(f"{salt}|{patient_ids[index]}".encode("ascii")).hexdigest(),
        ),
    )
    for index in ordered:
        tie = [
            hashlib.sha256(
                f"{salt}|{patient_ids[index]}|inner={fold}".encode("ascii")
            ).hexdigest()
            for fold in INNER_FOLDS
        ]
        fold = min(
            INNER_FOLDS,
            key=lambda value: (fold_events[value], fold_patients[value], tie[value]),
        )
        assignment[index] = fold
        fold_events[fold] += int(event_counts[index])
        fold_patients[fold] += 1
    if set(assignment) != set(int(value) for value in outer_train_indices):
        raise RuntimeError("inner patient assignment lost a patient")
    return assignment


@dataclass(frozen=True)
class _FitResult:
    logits: torch.Tensor
    state: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, object]


def _fit_reasoner(
    transformed: TransformedPatientFeatures,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    train_indices: Sequence[int],
    *,
    use_h: bool,
    use_fine: bool,
    l2: float,
    allow_candidate_subset: bool = False,
    fixed_prior_logits: torch.Tensor | None = None,
) -> _FitResult:
    indices = torch.tensor(tuple(train_indices), dtype=torch.long)
    if fixed_prior_logits is None:
        prior = jeffreys_reference_prior_logits(
            targets.index_select(0, indices),
            target_mask.index_select(0, indices),
            allow_candidate_subset=allow_candidate_subset,
        )
        prior_source = "fit_training_rows"
    else:
        if (
            not isinstance(fixed_prior_logits, torch.Tensor)
            or not fixed_prior_logits.is_floating_point()
            or tuple(fixed_prior_logits.shape) != (19,)
            or not torch.isfinite(fixed_prior_logits).all()
        ):
            raise ValueError("fixed_prior_logits must be finite floating [19]")
        prior = fixed_prior_logits.detach().cpu().contiguous()
        prior_source = "caller_frozen"
    model = SharedPositiveSetReasoner(prior, use_h=use_h, use_fine=use_fine)
    train_evidence = transformed.index_select(indices)
    train_targets = targets.index_select(0, indices)
    train_mask = target_mask.index_select(0, indices)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=LBFGS_MAX_ITER,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )
    closure_calls = 0
    first_loss: float | None = None

    def closure() -> torch.Tensor:
        nonlocal closure_calls, first_loss
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_evidence).logits
        set_loss = positive_set_mass_loss(
            logits,
            train_targets,
            train_mask,
            allow_candidate_subset=allow_candidate_subset,
        )
        penalty = sum(parameter.square().sum() for parameter in model.parameters())
        loss = set_loss + float(l2) * penalty
        if not torch.isfinite(loss):
            raise RuntimeError("v11 reasoner optimization became non-finite")
        loss.backward()
        closure_calls += 1
        if first_loss is None:
            first_loss = float(loss.detach())
        return loss

    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    final_train_logits = model(train_evidence).logits
    final_set_tensor = positive_set_mass_loss(
        final_train_logits,
        train_targets,
        train_mask,
        allow_candidate_subset=allow_candidate_subset,
    )
    final_penalty_tensor = sum(
        parameter.square().sum() for parameter in model.parameters()
    )
    final_total_tensor = final_set_tensor + float(l2) * final_penalty_tensor
    if not torch.isfinite(final_total_tensor):
        raise RuntimeError("v11 reasoner final objective became non-finite")
    final_total_tensor.backward()
    final_gradient_norm = float(
        torch.sqrt(
            sum(
                parameter.grad.detach().square().sum()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )
    )
    final_total = float(final_total_tensor.detach())
    if first_loss is not None and final_total > first_loss + 1e-6:
        raise RuntimeError("v11 reasoner final objective is worse than initialization")
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        all_logits = model(transformed).logits.detach().cpu()
        final_set = float(
            positive_set_mass_loss(
                all_logits.index_select(0, indices),
                train_targets,
                train_mask,
                allow_candidate_subset=allow_candidate_subset,
            )
        )
        final_penalty = float(
            sum(parameter.detach().square().sum() for parameter in model.parameters())
        )
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    assert first_loss is not None
    optimizer_state = next(iter(optimizer.state.values()), {})
    return _FitResult(
        logits=all_logits,
        state=state,
        diagnostics={
            "l2": l2,
            "prior_source": prior_source,
            "train_patient_count": len(indices),
            "trainable_parameter_count": model.n_trainable_parameters,
            "closure_calls": closure_calls,
            "first_total_loss": first_loss,
            "final_total_loss": final_total,
            "final_set_mass_loss": final_set,
            "final_l2_penalty_unweighted": final_penalty,
            "final_gradient_norm": final_gradient_norm,
            "optimizer_iterations": int(optimizer_state.get("n_iter", 0)),
            "optimizer_function_evaluations": int(
                optimizer_state.get("func_evals", closure_calls)
            ),
        },
    )


@dataclass(frozen=True)
class _InnerContext:
    fold: int
    train_indices: tuple[int, ...]
    held_indices: tuple[int, ...]
    transformed: TransformedPatientFeatures


def _select_l2(
    contexts: Sequence[_InnerContext],
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    use_h: bool,
    use_fine: bool,
) -> tuple[float, Mapping[str, object]]:
    all_outer_train = sorted(
        {index for context in contexts for index in context.held_indices}
    )
    position = {patient: row for row, patient in enumerate(all_outer_train)}
    candidates = {}
    for l2 in L2_CANDIDATES:
        oof = torch.full((len(all_outer_train), 19), torch.nan)
        fits = []
        for context in contexts:
            fitted = _fit_reasoner(
                context.transformed,
                targets,
                target_mask,
                context.train_indices,
                use_h=use_h,
                use_fine=use_fine,
                l2=l2,
            )
            for held in context.held_indices:
                oof[position[held]] = fitted.logits[held]
            fits.append(dict(fitted.diagnostics))
        if not torch.isfinite(oof).all():
            raise RuntimeError("inner OOF predictions are incomplete")
        selected = torch.tensor(all_outer_train, dtype=torch.long)
        metrics = _evaluate(
            oof,
            targets.index_select(0, selected),
            target_mask.index_select(0, selected),
        )
        candidates[str(l2)] = {"metrics": metrics, "fits": fits}
    selected_l2 = max(
        L2_CANDIDATES,
        key=lambda value: (
            candidates[str(value)]["metrics"]["top1"]["strict_accuracy"],
            candidates[str(value)]["metrics"]["ranking"]["macro_average_precision"],
            candidates[str(value)]["metrics"]["ranking"]["mean_reciprocal_rank"],
            -value,
        ),
    )
    return selected_l2, {
        "selection_order": "strict_then_macro_ap_then_mrr_then_lower_l2",
        "selected_l2": selected_l2,
        "candidates": candidates,
    }


def _complement_dropout_mask(
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    patient_ids: Sequence[str],
    train_indices: Sequence[int],
    *,
    outer_fold: int,
) -> torch.Tensor:
    result = target_mask.clone()
    for patient in train_indices:
        for channel in range(19):
            if not bool(result[patient, channel]) or targets[patient, channel] == 1:
                continue
            digest = hashlib.sha256(
                f"v11-complement-drop|{outer_fold}|{patient_ids[patient]}|{channel}".encode(
                    "ascii"
                )
            ).digest()
            if int.from_bytes(digest[:8], "big") % 10 == 0:
                result[patient, channel] = False
    if not (((targets == 1) & result).any(dim=1)).all():
        raise RuntimeError("complement dropout removed an observed positive")
    return result


def _transform_state(transform: FoldFeatureTransform) -> dict[str, torch.Tensor]:
    return {
        "transform.h_center": transform.h_center,
        "transform.h_scale": transform.h_scale,
        "transform.h_pca_mean": transform.h_pca_mean,
        "transform.h_components": transform.h_components,
        "transform.fine_center": transform.fine_center,
        "transform.fine_scale": transform.fine_scale,
    }


def _state_sha(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(_canonical_bytes({"shape": list(tensor.shape), "dtype": str(tensor.dtype)}))
        # ``Tensor.view(dtype)`` rejects zero-dimensional tensors (for
        # example the scalar ``config.l2`` stored in the final checkpoint).
        # Flatten first so scalars and higher-dimensional state entries share
        # the same byte-level hashing path.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run(args: argparse.Namespace) -> tuple[Mapping[str, object], Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    if _file_sha(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("v11 protocol changed after it was frozen")
    union = load_public_development_union(
        args.union_directory,
        expected_manifest_sha256=EXPECTED_PUBLIC_DEVELOPMENT_UNION_MANIFEST_SHA256,
    )
    fine_manifest = _load_json_manifest(
        args.fine_directory / "manifest.json",
        expected_sha=EXPECTED_FINE_MANIFEST_SHA256,
    )
    prefix_manifest = _load_json_manifest(
        args.prefix_directory / "manifest.json",
        expected_sha=EXPECTED_PREFIX_MANIFEST_SHA256,
    )
    _require_target_free_cache(fine_manifest, label="fine evidence")
    _require_target_free_cache(prefix_manifest, label="LaBraM prefix")
    union_event_ids = tuple(event.event_id for event in union.events)
    for label, manifest in (("fine", fine_manifest), ("prefix", prefix_manifest)):
        if tuple(str(value) for value in manifest.get("event_ids", ())) != union_event_ids:
            raise ValueError(f"{label} event order differs from the frozen union")
    fine_file = args.fine_directory / str(fine_manifest["tensor_file"])
    prefix_file = args.prefix_directory / str(prefix_manifest["tensor_file"])
    if _file_sha(fine_file) != EXPECTED_FINE_TENSOR_FILE_SHA256 or (
        _file_sha(prefix_file) != EXPECTED_PREFIX_TENSOR_FILE_SHA256
    ):
        raise ValueError("v11 evidence tensor file SHA mismatch")

    fine_payload = load_file(str(fine_file), device="cpu")
    fine_event = fine_payload["features"].detach()
    if tuple(fine_event.shape) != (988, 19, 20) or (
        tuple(fine_manifest["feature_names"]) != FINE_TEMPORAL_FEATURE_NAMES
    ):
        raise ValueError("v11 fine feature tensor/vocabulary changed")
    prefix_payload = load_file(str(prefix_file), device="cpu")
    prefix = prefix_payload["prefix_tokens"].detach()
    if tuple(prefix.shape) != (988, 15, 77, 200):
        raise ValueError("v11 LaBraM prefix tensor changed")
    h_event = extract_block9_phase_contrasts(prefix)
    del prefix, prefix_payload

    event_patient_index = torch.tensor(union.event_patient_index, dtype=torch.long)
    artifact_index = FINE_TEMPORAL_FEATURE_NAMES.index("artifact_burden_0_12s")
    reliability = (1.0 - fine_event[:, :, artifact_index]).clamp(0.0, 1.0)
    h_pool = robust_pool_complete_patient_bags(
        h_event, event_patient_index, len(union.patient_ids), reliability
    )
    fine_pool = robust_pool_complete_patient_bags(
        fine_event, event_patient_index, len(union.patient_ids), reliability
    )
    if not torch.equal(h_pool.event_counts, fine_pool.event_counts):
        raise RuntimeError("H/fine patient bags disagree")
    del h_event, fine_event, fine_payload

    # This is the first target-value read in the runner.  All candidates,
    # folds, raw features, caches, and protocol hashes are already frozen.
    target = load_verified_deepsoz_target_v2_artifact(
        args.target_directory,
        args.source_csv,
        args.split_csv,
        expected_target_artifact_sha256=EXPECTED_TARGET_ARTIFACT_SHA256,
        expected_summary_artifact_sha256=EXPECTED_TARGET_SUMMARY_SHA256,
        expected_readme_artifact_sha256=EXPECTED_TARGET_README_SHA256,
        expected_source_input_sha256=EXPECTED_SOURCE_SHA256,
        expected_split_input_sha256=EXPECTED_SPLIT_SHA256,
    )
    if target.receipt.receipt_sha256 != EXPECTED_TARGET_RECEIPT_SHA256 or (
        target.receipt.policy_sha256 != TARGET_V2_POLICY_SHA256
    ):
        raise ValueError("verified target receipt/policy changed")
    batch = target.registry.target_batch(union.patient_ids, require_eligible=True)
    targets = batch.values.cpu()
    target_mask = batch.mask.cpu()
    if bool(target_mask[:, 14].any()) or not (((targets == 1) & target_mask).any(dim=1)).all():
        raise ValueError("v11 target PZ/positive contract failed")

    h_patient = h_pool.features.cpu()
    fine_patient = fine_pool.features.cpu()
    patient_folds = torch.tensor(union.patient_folds, dtype=torch.long)
    oof = {
        "prevalence_only": torch.full((102, 19), torch.nan),
        **{name: torch.full((102, 19), torch.nan) for name in ARMS},
    }
    complement_oof = torch.full((102, 19), torch.nan)
    fold_results = []
    selected_l2_by_arm = {name: [] for name in ARMS}
    fold_strict = {name: [] for name in oof}
    outer_states: dict[str, torch.Tensor] = {}

    for outer_fold in OUTER_FOLDS:
        held = tuple(torch.nonzero(patient_folds == outer_fold, as_tuple=False).flatten().tolist())
        train = tuple(torch.nonzero(patient_folds != outer_fold, as_tuple=False).flatten().tolist())
        transform = fit_fold_transform(h_patient, fine_patient, train)
        transformed = transform.apply(h_patient, fine_patient)
        train_tensor = torch.tensor(train, dtype=torch.long)
        prior = jeffreys_reference_prior_logits(
            targets.index_select(0, train_tensor), target_mask.index_select(0, train_tensor)
        )
        held_tensor = torch.tensor(held, dtype=torch.long)
        oof["prevalence_only"].index_copy_(
            0, held_tensor, prior.expand(len(held), -1)
        )

        inner_assignment = _inner_assignments(
            train,
            patient_ids=union.patient_ids,
            event_counts=h_pool.event_counts,
            outer_fold=outer_fold,
        )
        inner_contexts = []
        inner_receipts = []
        for inner_fold in INNER_FOLDS:
            inner_held = tuple(index for index in train if inner_assignment[index] == inner_fold)
            inner_train = tuple(index for index in train if inner_assignment[index] != inner_fold)
            inner_transform = fit_fold_transform(h_patient, fine_patient, inner_train)
            inner_contexts.append(
                _InnerContext(
                    fold=inner_fold,
                    train_indices=inner_train,
                    held_indices=inner_held,
                    transformed=inner_transform.apply(h_patient, fine_patient),
                )
            )
            inner_receipts.append(
                {
                    "inner_fold": inner_fold,
                    "train_patient_count": len(inner_train),
                    "held_patient_count": len(inner_held),
                    "train_patient_ids": [union.patient_ids[index] for index in inner_train],
                    "held_patient_ids": [union.patient_ids[index] for index in inner_held],
                }
            )

        arm_rows = {}
        for arm, (use_h, use_fine) in ARMS.items():
            selected_l2, selection = _select_l2(
                inner_contexts,
                targets,
                target_mask,
                use_h=use_h,
                use_fine=use_fine,
            )
            selected_l2_by_arm[arm].append(selected_l2)
            fitted = _fit_reasoner(
                transformed,
                targets,
                target_mask,
                train,
                use_h=use_h,
                use_fine=use_fine,
                l2=selected_l2,
            )
            oof[arm].index_copy_(0, held_tensor, fitted.logits.index_select(0, held_tensor))
            held_metrics = _evaluate(
                fitted.logits.index_select(0, held_tensor),
                targets.index_select(0, held_tensor),
                target_mask.index_select(0, held_tensor),
            )
            fold_strict[arm].append(held_metrics["top1"]["strict_accuracy"])
            for name, value in fitted.state.items():
                outer_states[f"outer{outer_fold}.{arm}.{name}"] = value
            arm_rows[arm] = {
                "selected_l2": selected_l2,
                "inner_selection": selection,
                "fit": dict(fitted.diagnostics),
                "held_metrics": held_metrics,
            }

        prevalence_held = oof["prevalence_only"].index_select(0, held_tensor)
        prevalence_metrics = _evaluate(
            prevalence_held,
            targets.index_select(0, held_tensor),
            target_mask.index_select(0, held_tensor),
        )
        fold_strict["prevalence_only"].append(
            prevalence_metrics["top1"]["strict_accuracy"]
        )

        full_l2 = arm_rows["full_frozen_labram_plus_fine"]["selected_l2"]
        sensitivity_mask = _complement_dropout_mask(
            targets,
            target_mask,
            union.patient_ids,
            train,
            outer_fold=outer_fold,
        )
        sensitivity_fit = _fit_reasoner(
            transformed,
            targets,
            sensitivity_mask,
            train,
            use_h=True,
            use_fine=True,
            l2=full_l2,
            allow_candidate_subset=True,
        )
        complement_oof.index_copy_(
            0, held_tensor, sensitivity_fit.logits.index_select(0, held_tensor)
        )
        fold_results.append(
            {
                "outer_fold": outer_fold,
                "train_patient_count": len(train),
                "held_patient_count": len(held),
                "train_event_count": int(h_pool.event_counts[train_tensor].sum()),
                "held_event_count": int(h_pool.event_counts[held_tensor].sum()),
                "train_patient_ids": [union.patient_ids[index] for index in train],
                "held_patient_ids": [union.patient_ids[index] for index in held],
                "inner_folds": inner_receipts,
                "prevalence_held_metrics": prevalence_metrics,
                "arms": arm_rows,
                "complement_dropout": {
                    "drop_fraction_of_observed_complements": float(
                        ((target_mask & ~sensitivity_mask) & (targets == 0)).sum()
                        / ((target_mask & (targets == 0)).sum().clamp_min(1))
                    ),
                    "fit": dict(sensitivity_fit.diagnostics),
                },
            }
        )
        print(
            json.dumps(
                {
                    "outer_fold": outer_fold,
                    "status": "complete",
                    "held_patients": len(held),
                    "full_strict": arm_rows["full_frozen_labram_plus_fine"][
                        "held_metrics"
                    ]["top1"]["strict_accuracy"],
                    "full_l2": full_l2,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if any(not torch.isfinite(value).all() for value in (*oof.values(), complement_oof)):
        raise RuntimeError("v11 OOF prediction matrix is incomplete")
    metrics = {name: _evaluate(value, targets, target_mask) for name, value in oof.items()}
    complement_metrics = _evaluate(complement_oof, targets, target_mask)
    full_name = "full_frozen_labram_plus_fine"
    strongest_single = max(
        ("fine_change_only", "frozen_labram_only"),
        key=lambda name: (
            metrics[name]["top1"]["strict_accuracy"],
            metrics[name]["ranking"]["macro_average_precision"],
        ),
    )
    paired = {
        name: _paired_bootstrap(oof[full_name], oof[name], targets, target_mask)
        for name in ("fine_change_only", "frozen_labram_only", "prevalence_only")
    }
    full_top = oof[full_name].argmax(dim=1)
    complement_top = complement_oof.argmax(dim=1)
    complement_agreement = float((full_top == complement_top).float().mean())
    fold_nonlower = sum(
        full >= single
        for full, single in zip(fold_strict[full_name], fold_strict[strongest_single])
    )
    full_metrics = metrics[full_name]
    single_metrics = metrics[strongest_single]
    go_checks = {
        "strict_nonlower_than_fine": full_metrics["top1"]["strict_accuracy"]
        >= metrics["fine_change_only"]["top1"]["strict_accuracy"],
        "strict_nonlower_than_labram": full_metrics["top1"]["strict_accuracy"]
        >= metrics["frozen_labram_only"]["top1"]["strict_accuracy"],
        "relaxed_nonlower_than_strongest_single": full_metrics["top1"]["relaxed_accuracy"]
        >= single_metrics["top1"]["relaxed_accuracy"],
        "far_error_nonincreasing": full_metrics["far_error_count"]
        <= single_metrics["far_error_count"],
        "macro_ap_positive_increment": full_metrics["ranking"]["macro_average_precision"]
        > single_metrics["ranking"]["macro_average_precision"],
        "four_of_five_outer_folds_strict_nonlower": fold_nonlower >= 4,
        "bootstrap_does_not_show_clear_strict_harm": paired[strongest_single]["strict"][
            "ci95"
        ][1]
        >= 0.0,
    }
    go = all(go_checks.values())

    # Refit the predeclared full candidate on all 102 developmental patients.
    l2_counts = Counter(selected_l2_by_arm[full_name])
    final_l2 = max(
        L2_CANDIDATES,
        key=lambda value: (l2_counts[value], -abs(math.log(value / 0.05))),
    )
    all_indices = tuple(range(102))
    final_transform = fit_fold_transform(h_patient, fine_patient, all_indices)
    final_transformed = final_transform.apply(h_patient, fine_patient)
    final_fit = _fit_reasoner(
        final_transformed,
        targets,
        target_mask,
        all_indices,
        use_h=True,
        use_fine=True,
        l2=final_l2,
    )
    final_state = {
        **_transform_state(final_transform),
        **{f"reasoner.{name}": value for name, value in final_fit.state.items()},
        "config.l2": torch.tensor(final_l2, dtype=torch.float32),
    }

    anchor_comparison = None
    anchor_manifest_path = args.anchor_directory / "manifest.json"
    anchor_prediction_path = args.anchor_directory / "oof_predictions.safetensors"
    if anchor_manifest_path.is_file() and anchor_prediction_path.is_file():
        anchor_manifest = json.loads(anchor_manifest_path.read_text(encoding="utf-8"))
        anchor_ids = tuple(str(value) for value in anchor_manifest["patient_ids"])
        union_index = {patient: index for index, patient in enumerate(union.patient_ids)}
        selected_union = torch.tensor([union_index[patient] for patient in anchor_ids])
        anchor_payload = load_file(str(anchor_prediction_path), device="cpu")
        anchor_logits = anchor_payload["temporal_mil_exact"]
        v11_subset = oof[full_name].index_select(0, selected_union)
        subset_targets = targets.index_select(0, selected_union)
        subset_mask = target_mask.index_select(0, selected_union)
        anchor_comparison = {
            "scope": "original_65_only_training_cohorts_differ_developmental_comparison",
            "anchor_manifest_sha256": _file_sha(anchor_manifest_path),
            "anchor_predictions_sha256": _file_sha(anchor_prediction_path),
            "patient_count": len(anchor_ids),
            "anchor_metrics": _evaluate(anchor_logits, subset_targets, subset_mask),
            "v11_metrics": _evaluate(v11_subset, subset_targets, subset_mask),
            "paired_v11_minus_anchor": _paired_bootstrap(
                v11_subset, anchor_logits, subset_targets, subset_mask
            ),
        }

    oof_tensors = {
        **{f"oof.{name}": value for name, value in oof.items()},
        "oof.complement_dropout_full": complement_oof,
        "targets": targets,
        "target_mask": target_mask,
        "patient_folds": patient_folds,
        "patient_event_counts": h_pool.event_counts,
        "patient_mean_h_dispersion": h_pool.dispersion.mean(dim=(1, 2)),
        "patient_mean_fine_dispersion": fine_pool.dispersion.mean(dim=(1, 2)),
    }
    manifest = {
        "schema_version": SCHEMA,
        "status": "completed_internal_developmental_nested_oof",
        "decision": "GO_support_separate_fold_local_peft_trial" if go else (
            "NO_GO_keep_existing_temporal_mil_anchor"
        ),
        "claim_boundary": {
            "public_confirmation": False,
            "external_validation": False,
            "nested_oof_is_internal_developmental_estimate": True,
            "historical_source_eval_reclassified_developmental": True,
            "private_used": False,
            "private_reserved_for_frozen_zero_adaptation_transfer": True,
        },
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "foundation": {
            "backbone": "official_pretrained_LaBraM_Base_not_replaced",
            "trained_from_scratch": False,
            "foundation_trainable_parameters_v11_a": 0,
            "uniform_block9_prefix": True,
        },
        "patient_count": 102,
        "event_count": 988,
        "patient_ids": list(union.patient_ids),
        "event_counts": h_pool.event_counts.tolist(),
        "outer_folds": list(OUTER_FOLDS),
        "inner_fold_count": len(INNER_FOLDS),
        "arms": list(oof),
        "l2_candidates": list(L2_CANDIDATES),
        "selected_l2_by_arm": selected_l2_by_arm,
        "fold_results": fold_results,
        "metrics": metrics,
        "strongest_single_evidence_arm": strongest_single,
        "paired_full_minus_baselines": paired,
        "complement_dropout_sensitivity": {
            "metrics": complement_metrics,
            "top1_agreement_with_primary": complement_agreement,
        },
        "go_checks": go_checks,
        "go_all": go,
        "goal_thresholds_descriptive_only": {
            "strict_top1_ge_0_80": full_metrics["top1"]["strict_accuracy"] >= 0.80,
            "relaxed_top1_ge_0_85": full_metrics["top1"]["relaxed_accuracy"] >= 0.85,
            "thresholds_were_not_used_for_model_selection": True,
        },
        "anchor_comparison": anchor_comparison,
        "final_full_development_refit": {
            "selected_l2_by_outer_mode": final_l2,
            "outer_selected_l2_counts": {str(key): value for key, value in l2_counts.items()},
            "fit": dict(final_fit.diagnostics),
            "state_sha256": _state_sha(final_state),
            "foundation_weights_serialized": False,
        },
        "lineage": {
            "union_manifest_sha256": union.manifest_sha256,
            "fine_manifest_sha256": EXPECTED_FINE_MANIFEST_SHA256,
            "fine_tensor_file_sha256": EXPECTED_FINE_TENSOR_FILE_SHA256,
            "prefix_manifest_sha256": EXPECTED_PREFIX_MANIFEST_SHA256,
            "prefix_tensor_file_sha256": EXPECTED_PREFIX_TENSOR_FILE_SHA256,
            "target_artifact_sha256": target.receipt.target_artifact_sha256,
            "target_receipt_sha256": target.receipt.receipt_sha256,
            "target_policy_sha256": target.receipt.policy_sha256,
        },
        "access_receipt": {
            "target_values_loaded_only_after_protocol_folds_and_features_frozen": True,
            "source_eval_is_not_test": True,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "private_forward_count": 0,
            "llm_used_as_soz_predictor": False,
        },
    }
    return manifest, oof_tensors, final_state


def _publish(
    output_directory: Path,
    manifest: Mapping[str, object],
    oof_tensors: Mapping[str, torch.Tensor],
    final_state: Mapping[str, torch.Tensor],
) -> Path:
    target = Path(os.path.abspath(output_directory))
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        oof_path = staging / "oof_predictions.safetensors"
        final_path = staging / "final_checkpoint.safetensors"
        save_file(dict(oof_tensors), str(oof_path))
        save_file(dict(final_state), str(final_path))
        completed = dict(manifest)
        completed["files"] = {
            "oof_predictions.safetensors": {
                "sha256": _file_sha(oof_path),
                "size_bytes": oof_path.stat().st_size,
            },
            "final_checkpoint.safetensors": {
                "sha256": _file_sha(final_path),
                "size_bytes": final_path.stat().st_size,
            },
        }
        (staging / "manifest.json").write_bytes(_canonical_bytes(completed, newline=True))
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--union-directory", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--fine-directory", type=Path, default=DEFAULT_FINE)
    parser.add_argument("--prefix-directory", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--target-directory", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--anchor-directory", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    args = parse_args(argv)
    manifest, oof, final_state = run(args)
    path = _publish(args.output_directory, manifest, oof, final_state)
    completed = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    full = completed["metrics"]["full_frozen_labram_plus_fine"]
    print(
        json.dumps(
            {
                "status": completed["status"],
                "decision": completed["decision"],
                "path": str(path),
                "manifest_sha256": _file_sha(path / "manifest.json"),
                "strict_top1": full["top1"]["strict_accuracy"],
                "relaxed_top1": full["top1"]["relaxed_accuracy"],
                "macro_ap": full["ranking"]["macro_average_precision"],
                "private_used": False,
                "public_confirmation": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
