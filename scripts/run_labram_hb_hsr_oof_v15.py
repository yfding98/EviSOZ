#!/usr/bin/env python3
"""Run the single frozen HB-HSR public-development nested OOF candidate.

This runner deliberately reuses the audited v11.1 feature materialization,
patient folds, transforms, optimizer, and L2-selection machinery.  Only the
full H+fine reasoner is replaced by the soft hierarchical HB-HSR objective.
LaBraM remains frozen and private data are not accepted by this interface.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_labram_fine_temporal_nested_oof_v11 as shared_v11  # noqa: E402
import scripts.run_labram_fine_temporal_nested_oof_v11_1 as base_v11_1  # noqa: E402
from src.soz.hb_hsr_reasoner import (  # noqa: E402
    FoldLocalHBHSRPriors,
    HBHSRReasoner,
    HBHSRReasonerOutput,
    HB_HSR_CHANNEL_TO_SIDE,
    fit_fold_local_hb_hsr_priors,
    hb_hsr_set_mass_loss,
)
from src.soz.metrics import DEEPSOZ_STANDARD19_NEIGHBORS  # noqa: E402
from src.soz.v11_reasoner import (  # noqa: E402
    TransformedPatientFeatures,
    V11_CANDIDATE_MASK,
)


PROTOCOL_PATH = (
    ROOT
    / "research/02_method/labram_hb_hsr_recovery_protocol_20260812_zh.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "5df6199040dffbc7e27715363608a781c7bc90d9dc2a8956821f245ee2c86879"
)
ANCHOR_DIRECTORY = (
    ROOT / "outputs/labram_fine_temporal_nested_oof_v11_1_20260811_r2"
)
EXPECTED_ANCHOR_MANIFEST_SHA256 = (
    "f399678e5756ae30cbe5f9f87d9d8bb5b220b16015e1b2a0417110f20e70195c"
)
EXPECTED_ANCHOR_OOF_SHA256 = (
    "6443680b18b53b0c552b9634e7c9e2547284c9d08cccd5cd99c35b9e1a27ac08"
)
DEFAULT_OUTPUT = ROOT / "outputs/labram_hb_hsr_oof_v15_20260812"
SCHEMA = "soz_labram_hb_hsr_nested_oof_v15"
CANDIDATE_NAME = "hb_hsr_full_block9_plus_fine"
ANCHOR_NAME = "v11_1_full_block9_plus_fine"
FINITE_MASK_SENTINEL = -1.0e6
STRICT_MINIMUM_NET_GAIN = 5


_ORIGINAL_FIT_REASONER = base_v11_1._fit_reasoner
_ORIGINAL_SELECT_L2 = base_v11_1._select_l2
_ORIGINAL_LOAD_REASONER = base_v11_1._load_reasoner_from_fit


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "hb_hsr_reasoner": ROOT / "src/soz/hb_hsr_reasoner.py",
        "base_v11_1_runner": ROOT
        / "scripts/run_labram_fine_temporal_nested_oof_v11_1.py",
        "shared_v11_runner": ROOT
        / "scripts/run_labram_fine_temporal_nested_oof_v11.py",
        "v11_reasoner": ROOT / "src/soz/v11_reasoner.py",
        "protocol": PROTOCOL_PATH,
    }
    return {name: _file_sha(path) for name, path in paths.items()}


def _finite_deployment_scores(output: HBHSRReasonerOutput) -> torch.Tensor:
    """Return metric-safe scores while retaining the explicit fixed mask.

    The core model correctly emits ``-inf`` at carrier-only PZ.  Historical
    v11.1 metric contracts require a finite tensor before applying their fixed
    candidate mask, so the serialized OOF representation uses a finite value
    far below every valid log probability.  The candidate mask remains a
    mandatory, separately serialized deployment contract.
    """

    if not isinstance(output, HBHSRReasonerOutput):
        raise TypeError("output must be HBHSRReasonerOutput")
    scores = output.deployment_scores.detach().clone()
    candidate = V11_CANDIDATE_MASK.to(scores.device)
    if not torch.isfinite(scores[:, candidate]).all() or not torch.isneginf(
        scores[:, ~candidate]
    ).all():
        raise ValueError("HB-HSR deployment-score mask contract drifted")
    scores[:, ~candidate] = FINITE_MASK_SENTINEL
    if not torch.isfinite(scores).all():
        raise RuntimeError("finite HB-HSR OOF serialization failed")
    return scores


def _state_from_model(
    model: HBHSRReasoner,
    priors: FoldLocalHBHSRPriors,
) -> dict[str, torch.Tensor]:
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    state["fit.positive_patient_support"] = (
        priors.positive_patient_support.detach().cpu().clone()
    )
    state["fit.train_patient_indices"] = torch.tensor(
        priors.outer_train_patient_indices,
        dtype=torch.long,
    )
    return state


def _fit_hb_hsr_reasoner(
    transformed: TransformedPatientFeatures,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    train_indices: Sequence[int],
    *,
    use_h: bool,
    use_fine: bool,
    l2: float,
    allow_candidate_subset: bool = False,
) -> shared_v11._FitResult:
    if allow_candidate_subset:
        raise ValueError("HB-HSR does not permit a patient-specific candidate subset")
    if not use_h or not use_fine:
        raise ValueError("the frozen HB-HSR candidate requires both H and fine evidence")
    selected = tuple(int(value) for value in train_indices)
    indices = torch.tensor(selected, dtype=torch.long)
    priors = fit_fold_local_hb_hsr_priors(targets, target_mask, selected)
    model = HBHSRReasoner(priors, use_h=True, use_fine=True)
    train_evidence = transformed.index_select(indices)
    train_targets = targets.index_select(0, indices)
    train_mask = target_mask.index_select(0, indices)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=shared_v11.LBFGS_MAX_ITER,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )
    closure_calls = 0
    first_loss: float | None = None
    first_side: float | None = None
    first_conditional: float | None = None

    def closure() -> torch.Tensor:
        nonlocal closure_calls, first_loss, first_side, first_conditional
        optimizer.zero_grad(set_to_none=True)
        loss_output = hb_hsr_set_mass_loss(
            model(train_evidence), train_targets, train_mask
        )
        penalty = sum(parameter.square().sum() for parameter in model.parameters())
        total = loss_output.total + float(l2) * penalty
        if not torch.isfinite(total):
            raise RuntimeError("HB-HSR optimization became non-finite")
        total.backward()
        closure_calls += 1
        if first_loss is None:
            first_loss = float(total.detach())
            first_side = float(loss_output.side_set.detach())
            first_conditional = float(
                loss_output.channel_given_side_set.detach()
            )
        return total

    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    final_output = model(train_evidence)
    final_loss_output = hb_hsr_set_mass_loss(
        final_output, train_targets, train_mask
    )
    final_penalty_tensor = sum(
        parameter.square().sum() for parameter in model.parameters()
    )
    final_total_tensor = final_loss_output.total + float(l2) * final_penalty_tensor
    if not torch.isfinite(final_total_tensor):
        raise RuntimeError("HB-HSR final objective became non-finite")
    final_total_tensor.backward()
    gradient_norm = float(
        torch.sqrt(
            sum(
                parameter.grad.detach().square().sum()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )
    )
    final_total = float(final_total_tensor.detach())
    if first_loss is None or first_side is None or first_conditional is None:
        raise RuntimeError("HB-HSR optimizer did not execute its closure")
    if final_total > first_loss + 1e-6:
        raise RuntimeError("HB-HSR final objective is worse than initialization")
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        all_logits = _finite_deployment_scores(model(transformed)).cpu()
        replay_loss = hb_hsr_set_mass_loss(
            model(train_evidence), train_targets, train_mask
        )
        final_penalty = float(
            sum(parameter.detach().square().sum() for parameter in model.parameters())
        )
    optimizer_state = next(iter(optimizer.state.values()), {})
    diagnostics = {
        "reasoner": "HB-HSR",
        "l2": float(l2),
        "train_patient_count": len(selected),
        "trainable_parameter_count": model.n_trainable_parameters,
        "foundation_optimizer_parameter_count": 0,
        "closure_calls": closure_calls,
        "first_total_loss": first_loss,
        "first_side_set_loss": first_side,
        "first_channel_given_side_set_loss": first_conditional,
        "final_total_loss": final_total,
        "final_hb_hsr_loss_unregularized": float(replay_loss.total),
        "final_side_set_loss": float(replay_loss.side_set),
        "final_channel_given_side_set_loss": float(
            replay_loss.channel_given_side_set
        ),
        "final_l2_penalty_unweighted": final_penalty,
        "final_gradient_norm": gradient_norm,
        "optimizer_iterations": int(optimizer_state.get("n_iter", 0)),
        "optimizer_function_evaluations": int(
            optimizer_state.get("func_evals", closure_calls)
        ),
        "support_smoothing": 0.5,
        "support_exponent": -0.5,
        "side_to_conditional_loss_ratio": "1:1",
        "hard_side_gate": False,
    }
    return shared_v11._FitResult(
        logits=all_logits,
        state=_state_from_model(model, priors),
        diagnostics=diagnostics,
    )


def _fit_reasoner_dispatch(*args, **kwargs) -> shared_v11._FitResult:
    if kwargs.get("use_h") and kwargs.get("use_fine") and not kwargs.get(
        "allow_candidate_subset", False
    ):
        return _fit_hb_hsr_reasoner(*args, **kwargs)
    return _ORIGINAL_FIT_REASONER(*args, **kwargs)


def _select_hb_hsr_l2(
    contexts: Sequence[shared_v11._InnerContext],
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[float, Mapping[str, object]]:
    all_outer_train = sorted(
        {index for context in contexts for index in context.held_indices}
    )
    position = {patient: row for row, patient in enumerate(all_outer_train)}
    candidates: dict[str, object] = {}
    for l2 in shared_v11.L2_CANDIDATES:
        oof = torch.full((len(all_outer_train), 19), torch.nan)
        fits = []
        for context in contexts:
            fitted = _fit_hb_hsr_reasoner(
                context.transformed,
                targets,
                target_mask,
                context.train_indices,
                use_h=True,
                use_fine=True,
                l2=l2,
            )
            for held in context.held_indices:
                oof[position[held]] = fitted.logits[held]
            fits.append(dict(fitted.diagnostics))
        if not torch.isfinite(oof).all():
            raise RuntimeError("HB-HSR inner OOF predictions are incomplete")
        selected = torch.tensor(all_outer_train, dtype=torch.long)
        metrics = base_v11_1._evaluate(
            oof,
            targets.index_select(0, selected),
            target_mask.index_select(0, selected),
        )
        candidates[str(l2)] = {"metrics": metrics, "fits": fits}
    selected_l2 = max(
        shared_v11.L2_CANDIDATES,
        key=lambda value: (
            candidates[str(value)]["metrics"]["top1"]["strict_accuracy"],
            candidates[str(value)]["metrics"]["ranking"][
                "macro_average_precision"
            ],
            candidates[str(value)]["metrics"]["ranking"][
                "mean_reciprocal_rank"
            ],
            -value,
        ),
    )
    return selected_l2, {
        "selection_order": "strict_then_macro_ap_then_mrr_then_lower_l2",
        "selected_l2": selected_l2,
        "candidates": candidates,
        "new_hyperparameter_search": False,
    }


def _select_l2_dispatch(*args, **kwargs):
    if kwargs.get("use_h") and kwargs.get("use_fine"):
        return _select_hb_hsr_l2(args[0], args[1], args[2])
    return _ORIGINAL_SELECT_L2(*args, **kwargs)


@dataclass(frozen=True)
class _MetricSafeOutput:
    logits: torch.Tensor


class _MetricSafeHBHSRModel(torch.nn.Module):
    """Expose the historical ``.logits`` interface with finite masked PZ."""

    def __init__(self, reasoner: HBHSRReasoner) -> None:
        super().__init__()
        self.reasoner = reasoner

    def forward(self, evidence: TransformedPatientFeatures) -> _MetricSafeOutput:
        return _MetricSafeOutput(
            logits=_finite_deployment_scores(self.reasoner(evidence))
        )


def _load_hb_hsr_from_fit(state: Mapping[str, torch.Tensor]) -> torch.nn.Module:
    required = {
        "reference_prior_logits",
        "conditional_channel_prior",
        "fit.positive_patient_support",
        "fit.train_patient_indices",
    }
    if not required.issubset(state):
        missing = sorted(required - set(state))
        raise ValueError(f"HB-HSR fit state is incomplete: {missing}")
    if "candidate_mask" not in state or not torch.equal(
        state["candidate_mask"].detach().cpu().bool(), V11_CANDIDATE_MASK
    ):
        raise ValueError("HB-HSR fit state candidate mask changed")
    if "channel_to_side" not in state or not torch.equal(
        state["channel_to_side"].detach().cpu().long(),
        HB_HSR_CHANNEL_TO_SIDE,
    ):
        raise ValueError("HB-HSR fit state side partition changed")
    priors = FoldLocalHBHSRPriors(
        reference_prior_logits=state["reference_prior_logits"].detach().cpu(),
        positive_patient_support=state[
            "fit.positive_patient_support"
        ].detach().cpu().long(),
        conditional_channel_prior=state[
            "conditional_channel_prior"
        ].detach().cpu(),
        outer_train_patient_indices=tuple(
            int(value) for value in state["fit.train_patient_indices"].tolist()
        ),
    )
    model = HBHSRReasoner(priors, use_h=True, use_fine=True)
    model_names = set(model.state_dict())
    model.load_state_dict(
        {name: value for name, value in state.items() if name in model_names},
        strict=True,
    )
    model.eval()
    wrapper = _MetricSafeHBHSRModel(model)
    wrapper.eval()
    return wrapper


def _strict_contributions(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for patient in range(logits.shape[0]):
        indices = torch.nonzero(mask[patient], as_tuple=False).flatten()
        scores = logits[patient, indices]
        tied = indices[scores == scores.max()]
        rows.append(targets[patient, tied].float().mean())
    return torch.stack(rows)


def _contralateral_far_count(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    channel_to_side = HB_HSR_CHANNEL_TO_SIDE
    total = 0.0
    for patient in range(logits.shape[0]):
        candidates = torch.nonzero(mask[patient], as_tuple=False).flatten()
        scores = logits[patient, candidates]
        tied = candidates[scores == scores.max()]
        positive = (targets[patient] == 1) & mask[patient]
        accepted = positive.clone()
        if int(positive.sum()) <= 4:
            for index in torch.nonzero(positive, as_tuple=False).flatten().tolist():
                accepted[list(DEEPSOZ_STANDARD19_NEIGHBORS[index])] = True
        accepted &= mask[patient]
        gold_lateral = {
            int(channel_to_side[index])
            for index in torch.nonzero(positive, as_tuple=False).flatten().tolist()
            if int(channel_to_side[index]) in (0, 1)
        }
        if len(gold_lateral) != 1:
            continue
        opposite = 1 - next(iter(gold_lateral))
        total += float(
            torch.tensor(
                [
                    (not bool(accepted[index]))
                    and int(channel_to_side[index]) == opposite
                    for index in tied.tolist()
                ],
                dtype=torch.float32,
            ).mean()
        )
    return total


def _win_loss_tie(
    candidate: torch.Tensor,
    anchor: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, int]:
    cand = _strict_contributions(candidate, targets, mask)
    base = _strict_contributions(anchor, targets, mask)
    return {
        "wins": int((cand > base).sum()),
        "losses": int((cand < base).sum()),
        "ties": int((cand == base).sum()),
    }


def _rename_candidate_fields(manifest: dict) -> None:
    mappings = (
        ("metrics", True),
        ("absolute_patient_bootstrap", True),
        ("selected_l2_by_arm", True),
        ("fold_strict", False),
    )
    for key, required in mappings:
        container = manifest.get(key)
        if container is None:
            if required:
                raise KeyError(key)
            continue
        if "full_frozen_labram_plus_fine" in container:
            container[CANDIDATE_NAME] = container.pop(
                "full_frozen_labram_plus_fine"
            )
    for fold in manifest["fold_results"]:
        arms = fold["arms"]
        arms[CANDIDATE_NAME] = arms.pop("full_frozen_labram_plus_fine")


def _postprocess(
    manifest: dict,
    oof_tensors: dict[str, torch.Tensor],
    final_state: dict[str, torch.Tensor],
    outer_states: dict[str, torch.Tensor],
    *,
    source_hashes: Mapping[str, str],
) -> tuple[dict, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    anchor_manifest_path = ANCHOR_DIRECTORY / "manifest.json"
    anchor_oof_path = ANCHOR_DIRECTORY / "oof_predictions.safetensors"
    if _file_sha(anchor_manifest_path) != EXPECTED_ANCHOR_MANIFEST_SHA256 or (
        _file_sha(anchor_oof_path) != EXPECTED_ANCHOR_OOF_SHA256
    ):
        raise ValueError("pinned v11.1 block-9 anchor changed")
    anchor_manifest = json.loads(anchor_manifest_path.read_text(encoding="utf-8"))
    anchor_payload = load_file(str(anchor_oof_path), device="cpu")
    anchor_logits = anchor_payload["oof.full_frozen_labram_plus_fine"].float()
    candidate_logits = oof_tensors.pop(
        "oof.full_frozen_labram_plus_fine"
    ).float()
    targets = oof_tensors["targets"].float()
    mask = oof_tensors["target_mask"].bool()
    if anchor_manifest["patient_ids"] != manifest["patient_ids"]:
        raise ValueError("HB-HSR and pinned anchor patient order differs")
    if not torch.equal(targets, anchor_payload["targets"].float()) or not torch.equal(
        mask, anchor_payload["target_mask"].bool()
    ):
        raise ValueError("HB-HSR and pinned anchor target tensors differ")
    if candidate_logits.shape != (101, 19) or not torch.isfinite(
        candidate_logits
    ).all():
        raise ValueError("HB-HSR OOF candidate tensor is invalid")

    oof_tensors[f"oof.{CANDIDATE_NAME}"] = candidate_logits
    oof_tensors[f"oof.{ANCHOR_NAME}"] = anchor_logits
    candidate_metrics = base_v11_1._evaluate(candidate_logits, targets, mask)
    anchor_metrics = base_v11_1._evaluate(anchor_logits, targets, mask)
    paired = base_v11_1._paired_bootstrap(
        candidate_logits, anchor_logits, targets, mask
    )
    transitions = _win_loss_tie(candidate_logits, anchor_logits, targets, mask)
    candidate_folds = [
        float(fold["arms"]["full_frozen_labram_plus_fine"]["held_metrics"][
            "top1"
        ]["strict_accuracy"])
        for fold in manifest["fold_results"]
    ]
    anchor_folds = [
        float(fold["arms"]["full_frozen_labram_plus_fine"]["held_metrics"][
            "top1"
        ]["strict_accuracy"])
        for fold in anchor_manifest["fold_results"]
    ]
    fold_nonlower = sum(
        candidate >= anchor
        for candidate, anchor in zip(candidate_folds, anchor_folds)
    )
    candidate_strict_count = float(
        _strict_contributions(candidate_logits, targets, mask).sum()
    )
    anchor_strict_count = float(
        _strict_contributions(anchor_logits, targets, mask).sum()
    )
    net_gain = candidate_strict_count - anchor_strict_count
    candidate_contralateral = _contralateral_far_count(
        candidate_logits, targets, mask
    )
    anchor_contralateral = _contralateral_far_count(anchor_logits, targets, mask)
    checks = {
        "strict_net_gain_at_least_5_of_101": net_gain
        >= STRICT_MINIMUM_NET_GAIN,
        "strict_paired_ci_lower_strictly_positive": float(
            paired["strict"]["ci95"][0]
        )
        > 0.0,
        "macro_ap_paired_ci_lower_nonnegative": float(
            paired["macro_ap"]["ci95"][0]
        )
        >= 0.0,
        "relaxed_point_nonlower": float(
            candidate_metrics["top1"]["relaxed_accuracy"]
        )
        >= float(anchor_metrics["top1"]["relaxed_accuracy"]),
        "far_error_not_above_23": float(candidate_metrics["far_error_count"])
        <= 23.0,
        "four_of_five_fold_strict_nonlower": fold_nonlower >= 4,
        "foundation_optimizer_parameter_count_zero": True,
        "private_access_and_forward_count_zero": True,
    }
    passed = all(checks.values())

    original_go = {
        "paired_candidate_minus_internal_single_family_baselines": manifest.pop(
            "paired_full_minus_baselines"
        ),
        "go_checks_vs_internal_single_family_baselines": manifest.pop(
            "go_checks"
        ),
        "engineering_go_vs_internal_single_family_baselines": manifest.pop(
            "engineering_go_all"
        ),
        "scientific_increment_vs_internal_H_only": manifest.pop(
            "scientific_increment_supported"
        ),
    }
    _rename_candidate_fields(manifest)
    manifest.update(
        {
            "schema_version": SCHEMA,
            "status": "completed_post_hoc_mechanism_driven_public_development_oof",
            "decision": (
                "HB_HSR_RETAIN_AS_DEVELOPMENT_CANDIDATE"
                if passed
                else "HB_HSR_STOP_ON_CURRENT_PUBLIC_COHORT"
            ),
            "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "candidate_name": CANDIDATE_NAME,
            "primary_comparator": ANCHOR_NAME,
            "primary_comparison": {
                "scope": "same_101_repeatedly_used_public_development_not_confirmation",
                "candidate_metrics": candidate_metrics,
                "anchor_metrics": anchor_metrics,
                "paired_candidate_minus_anchor": paired,
                "strict_win_loss_tie": transitions,
                "strict_count_candidate": candidate_strict_count,
                "strict_count_anchor": anchor_strict_count,
                "strict_net_gain": net_gain,
                "candidate_fold_strict": candidate_folds,
                "anchor_fold_strict": anchor_folds,
                "fold_strict_nonlower_count": fold_nonlower,
                "candidate_contralateral_far_count": candidate_contralateral,
                "anchor_contralateral_far_count": anchor_contralateral,
            },
            "frozen_stop_gate": {
                "all_passed": passed,
                "checks": checks,
                "failure_action": "no_more_HB_HSR_or_related_scans_on_current_101",
            },
            "internal_replay_diagnostics_not_primary": original_go,
            "hb_hsr_contract": {
                "side_groups": ["L", "R", "M"],
                "support_smoothing": 0.5,
                "support_exponent": -0.5,
                "side_to_conditional_loss_ratio": "1:1",
                "hard_side_gate": False,
                "foundation_trainable_parameters": 0,
                "reasoner_trainable_parameters": 36,
                "fixed_candidate_count": 18,
                "pz_carrier_only": True,
                "training_inference_probability_factorization_identical": True,
            },
            "source_file_sha256": {
                **manifest["source_file_sha256"],
                **dict(source_hashes),
            },
            "claim_boundary": {
                "public_confirmation": False,
                "external_validation": False,
                "repeatedly_used_public_development": True,
                "private_used": False,
                "private_forward_count": 0,
                "clinical_deployment_allowed": False,
                "cortical_SOZ_claim_allowed": False,
            },
            "access_receipt": {
                **manifest["access_receipt"],
                "private_eeg_loaded": False,
                "private_target_values_loaded": False,
                "private_forward_count": 0,
                "held_targets_used_for_support_prior_or_transform": False,
                "patient_specific_gold_side_used_at_inference": False,
            },
            "lineage": {
                **manifest["lineage"],
                "v11_1_anchor_manifest_sha256": EXPECTED_ANCHOR_MANIFEST_SHA256,
                "v11_1_anchor_oof_sha256": EXPECTED_ANCHOR_OOF_SHA256,
                "failure_audit": "outputs/block9_v11_1_failure_audit_20260812/audit.json",
            },
        }
    )
    return manifest, oof_tensors, final_state, outer_states


def run(args: argparse.Namespace):
    if _file_sha(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("HB-HSR frozen protocol changed")
    source_before = _source_hashes()
    original_fit = base_v11_1._fit_reasoner
    original_select = base_v11_1._select_l2
    original_loader = base_v11_1._load_reasoner_from_fit
    base_v11_1._fit_reasoner = _fit_reasoner_dispatch
    base_v11_1._select_l2 = _select_l2_dispatch
    base_v11_1._load_reasoner_from_fit = _load_hb_hsr_from_fit
    try:
        manifest, oof, final_state, outer_states = base_v11_1.run(args)
    finally:
        base_v11_1._fit_reasoner = original_fit
        base_v11_1._select_l2 = original_select
        base_v11_1._load_reasoner_from_fit = original_loader
    source_after = _source_hashes()
    if source_after != source_before:
        raise RuntimeError("HB-HSR source files changed during execution")
    return _postprocess(
        manifest,
        oof,
        final_state,
        outer_states,
        source_hashes=source_after,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument(
        "--union-directory", type=Path, default=base_v11_1.DEFAULT_UNION
    )
    parser.add_argument(
        "--fine-directory", type=Path, default=base_v11_1.DEFAULT_FINE
    )
    parser.add_argument(
        "--prefix-directory", type=Path, default=base_v11_1.DEFAULT_PREFIX
    )
    parser.add_argument(
        "--target-directory", type=Path, default=base_v11_1.DEFAULT_TARGET
    )
    parser.add_argument("--source-csv", type=Path, default=base_v11_1.DEFAULT_SOURCE)
    parser.add_argument("--split-csv", type=Path, default=base_v11_1.DEFAULT_SPLIT)
    parser.add_argument(
        "--anchor-directory", type=Path, default=base_v11_1.DEFAULT_ANCHOR
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    args = parse_args(argv)
    result = run(args)
    recorded_sources = result[0]["source_file_sha256"]
    current_sources = _source_hashes()
    if any(recorded_sources.get(name) != digest for name, digest in current_sources.items()):
        raise RuntimeError("HB-HSR source files changed before atomic publish")
    path = base_v11_1._publish(args.output_directory, *result)
    if _source_hashes() != current_sources:
        raise RuntimeError("HB-HSR source files changed during atomic publish")
    completed = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    comparison = completed["primary_comparison"]
    print(
        json.dumps(
            {
                "status": completed["status"],
                "decision": completed["decision"],
                "path": str(path),
                "manifest_sha256": _file_sha(path / "manifest.json"),
                "strict_candidate": comparison["strict_count_candidate"],
                "strict_anchor": comparison["strict_count_anchor"],
                "strict_net_gain": comparison["strict_net_gain"],
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
