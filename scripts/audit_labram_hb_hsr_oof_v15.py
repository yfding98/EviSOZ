#!/usr/bin/env python3
"""Read-only publication and leakage audit of the completed HB-HSR OOF.

The training bundle is immutable and pinned byte-for-byte.  This audit does
not refit, rescore, choose a model, access raw EEG, or access private data.  It
recomputes primary metrics and the frozen STOP gate, verifies every outer-fold
support prior against outer-train targets, and documents publication-only
namespace/legacy-field issues separately from the primary OOF numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable, Mapping

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_labram_fine_temporal_nested_oof_v11_1 import (  # noqa: E402
    _evaluate,
    _paired_bootstrap,
)
from scripts.run_labram_hb_hsr_oof_v15 import (  # noqa: E402
    ANCHOR_NAME,
    CANDIDATE_NAME,
    _contralateral_far_count,
    _strict_contributions,
    _win_loss_tie,
)
from src.soz.hb_hsr_reasoner import (  # noqa: E402
    HB_HSR_CHANNEL_TO_SIDE,
    fit_fold_local_hb_hsr_priors,
)
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SOURCE = ROOT / "outputs/labram_hb_hsr_oof_v15_20260812"
ANCHOR = ROOT / "outputs/labram_fine_temporal_nested_oof_v11_1_20260811_r2"
OUTPUT = ROOT / "outputs/labram_hb_hsr_oof_v15_audit_20260812"
EXPECTED_SOURCE_FILES = {
    "manifest.json": "f2c2c00d0772cc3777d8d644f77f20f3f066bba4aa49d175e818d529cf34cd28",
    "oof_predictions.safetensors": "3e5a1b331b4b8e0c53bf500cb66996afae1180c02f0bfbc6afa9a24c255a0654",
    "outer_fold_states.safetensors": "e76f5eee39d77e7b3fe467a903882f81c9179337b630c7a28d0979bea9eecab8",
    "final_checkpoint.safetensors": "b9aae50354c0f64c7ec66032cc0522b89afb94baeb1424548a240c37502a7646",
}
EXPECTED_ANCHOR_FILES = {
    "manifest.json": "f399678e5756ae30cbe5f9f87d9d8bb5b220b16015e1b2a0417110f20e70195c",
    "oof_predictions.safetensors": "6443680b18b53b0c552b9634e7c9e2547284c9d08cccd5cd99c35b9e1a27ac08",
}
EXPECTED_RUNNER_SHA256 = (
    "b0ba5413b8461a0d4aac233271cc147a13491a9e27f746c1efd8e95833cdd2b0"
)
EXPECTED_CORE_SHA256 = (
    "34967a1888fc15cbf13d7108d958062f10ace3b57c661aa1f4e228cd65620331"
)
SCHEMA = "soz_labram_hb_hsr_v15_independent_read_only_audit_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_pinned(directory: Path, expected: Mapping[str, str]) -> None:
    for name, digest in expected.items():
        path = directory / name
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"pinned artifact changed: {path}")


def _close(left: float, right: float, *, atol: float = 1e-7) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= atol


def _metric_core(metrics: Mapping[str, object]) -> dict[str, float]:
    hit_at_k = metrics["ranking"]["hit_at_k"]
    hit3 = hit_at_k[3] if 3 in hit_at_k else hit_at_k["3"]
    hit5 = hit_at_k[5] if 5 in hit_at_k else hit_at_k["5"]
    return {
        "strict": float(metrics["top1"]["strict_accuracy"]),
        "relaxed": float(metrics["top1"]["relaxed_accuracy"]),
        "macro_ap": float(metrics["ranking"]["macro_average_precision"]),
        "mrr": float(metrics["ranking"]["mean_reciprocal_rank"]),
        "hit_at_3": float(hit3),
        "hit_at_5": float(hit5),
        "far_error_count": float(metrics["far_error_count"]),
    }


def _assert_metric_replay(
    replay: Mapping[str, object], recorded: Mapping[str, object]
) -> None:
    left = _metric_core(replay)
    right = _metric_core(recorded)
    for key in left:
        if not _close(left[key], right[key]):
            raise ValueError(f"metric replay mismatch for {key}: {left[key]} vs {right[key]}")


def _subset_metrics(
    indices: Iterable[int],
    candidate: torch.Tensor,
    anchor: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, object]:
    selected = torch.tensor(tuple(indices), dtype=torch.long)
    if selected.numel() < 1:
        return {"n_patients": 0, "candidate": None, "anchor": None}
    return {
        "n_patients": int(selected.numel()),
        "candidate": _metric_core(
            _evaluate(
                candidate.index_select(0, selected),
                targets.index_select(0, selected),
                mask.index_select(0, selected),
            )
        ),
        "anchor": _metric_core(
            _evaluate(
                anchor.index_select(0, selected),
                targets.index_select(0, selected),
                mask.index_select(0, selected),
            )
        ),
    }


def _fit_diagnostics_are_hb_safe(value: Mapping[str, object]) -> bool:
    return (
        value.get("reasoner") == "HB-HSR"
        and int(value.get("trainable_parameter_count", -1)) == 36
        and int(value.get("foundation_optimizer_parameter_count", -1)) == 0
        and value.get("hard_side_gate") is False
        and float(value.get("support_smoothing", math.nan)) == 0.5
        and float(value.get("support_exponent", math.nan)) == -0.5
        and value.get("side_to_conditional_loss_ratio") == "1:1"
    )


def _selected_l2_replays(arm: Mapping[str, object]) -> bool:
    selection = arm["inner_selection"]
    candidates = selection["candidates"]
    values = tuple(float(value) for value in candidates)
    selected = max(
        values,
        key=lambda value: (
            float(candidates[str(value)]["metrics"]["top1"]["strict_accuracy"]),
            float(
                candidates[str(value)]["metrics"]["ranking"][
                    "macro_average_precision"
                ]
            ),
            float(
                candidates[str(value)]["metrics"]["ranking"][
                    "mean_reciprocal_rank"
                ]
            ),
            -value,
        ),
    )
    return selected == float(arm["selected_l2"])


def _verify_outer_receipts(
    manifest: Mapping[str, object],
    states: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    mask: torch.Tensor,
    patient_folds: torch.Tensor,
) -> dict[str, object]:
    patient_ids = tuple(str(value) for value in manifest["patient_ids"])
    index_by_id = {patient: index for index, patient in enumerate(patient_ids)}
    fold_rows = []
    all_safe = True
    for fold in manifest["fold_results"]:
        outer = int(fold["outer_fold"])
        expected_train = tuple(
            torch.nonzero(patient_folds != outer, as_tuple=False).flatten().tolist()
        )
        expected_held = tuple(
            torch.nonzero(patient_folds == outer, as_tuple=False).flatten().tolist()
        )
        listed_train = tuple(index_by_id[str(value)] for value in fold["train_patient_ids"])
        listed_held = tuple(index_by_id[str(value)] for value in fold["held_patient_ids"])
        prefix = f"outer{outer}.full_frozen_labram_plus_fine."
        state_train = tuple(
            int(value) for value in states[prefix + "fit.train_patient_indices"].tolist()
        )
        recomputed = fit_fold_local_hb_hsr_priors(
            targets, mask, expected_train
        )
        support_equal = torch.equal(
            states[prefix + "fit.positive_patient_support"].long(),
            recomputed.positive_patient_support,
        )
        reference_equal = torch.allclose(
            states[prefix + "reference_prior_logits"].float(),
            recomputed.reference_prior_logits,
            atol=1e-7,
            rtol=1e-7,
        )
        conditional_equal = torch.allclose(
            states[prefix + "conditional_channel_prior"].float(),
            recomputed.conditional_channel_prior,
            atol=1e-7,
            rtol=1e-7,
        )
        fixed_mask = torch.equal(
            states[prefix + "candidate_mask"].bool(), V11_CANDIDATE_MASK
        )
        side_partition = torch.equal(
            states[prefix + "channel_to_side"].long(), HB_HSR_CHANNEL_TO_SIDE
        )
        arm = fold["arms"][CANDIDATE_NAME]
        outer_fit_safe = _fit_diagnostics_are_hb_safe(arm["fit"])
        inner_fit_safe = all(
            _fit_diagnostics_are_hb_safe(fit)
            for candidate in arm["inner_selection"]["candidates"].values()
            for fit in candidate["fits"]
        )
        inner_rosters_safe = True
        outer_train_set = set(expected_train)
        for inner in fold["inner_folds"]:
            inner_train = {index_by_id[str(value)] for value in inner["train_patient_ids"]}
            inner_held = {index_by_id[str(value)] for value in inner["held_patient_ids"]}
            inner_rosters_safe &= (
                not (inner_train & inner_held)
                and inner_train | inner_held == outer_train_set
                and inner_train <= outer_train_set
                and inner_held <= outer_train_set
            )
        l2_safe = _selected_l2_replays(arm)
        checks = {
            "manifest_train_roster_matches_fold": listed_train == expected_train,
            "manifest_held_roster_matches_fold": listed_held == expected_held,
            "serialized_prior_train_indices_match": state_train == expected_train,
            "positive_support_outer_train_only": support_equal,
            "jeffreys_prior_outer_train_only": reference_equal,
            "conditional_prior_outer_train_only": conditional_equal,
            "candidate_mask_fixed": fixed_mask,
            "side_partition_fixed": side_partition,
            "outer_fit_foundation_zero_and_contract_fixed": outer_fit_safe,
            "inner_fits_foundation_zero_and_contract_fixed": inner_fit_safe,
            "inner_rosters_nested_with_no_outer_held": inner_rosters_safe,
            "l2_selection_replays_frozen_order": l2_safe,
        }
        safe = all(checks.values())
        all_safe &= safe
        fold_rows.append({"outer_fold": outer, "all_passed": safe, "checks": checks})
    return {"all_outer_folds_passed": all_safe, "folds": fold_rows}


def _rare_strata(
    targets: torch.Tensor,
    patient_folds: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, list[int]]:
    candidate_indices = torch.nonzero(V11_CANDIDATE_MASK, as_tuple=False).flatten()
    exposed = []
    unexposed = []
    for patient in range(targets.shape[0]):
        train = patient_folds != patient_folds[patient]
        support = ((targets[train] == 1) & mask[train]).sum(dim=0)
        ordered = sorted(
            candidate_indices.tolist(),
            key=lambda index: (int(support[index]), int(index)),
        )
        bottom_six = torch.zeros(19, dtype=torch.bool)
        bottom_six[ordered[:6]] = True
        has_rare = bool(((targets[patient] == 1) & bottom_six).any())
        (exposed if has_rare else unexposed).append(patient)
    return {"contains_fold_local_bottom_six_positive": exposed, "other": unexposed}


def _side_strata(targets: torch.Tensor, mask: torch.Tensor) -> dict[str, list[int]]:
    result = {"left_only": [], "right_only": [], "midline_or_multiside": []}
    for patient in range(targets.shape[0]):
        positive = torch.nonzero(
            (targets[patient] == 1) & mask[patient], as_tuple=False
        ).flatten()
        sides = {int(HB_HSR_CHANNEL_TO_SIDE[index]) for index in positive.tolist()}
        if sides == {0}:
            result["left_only"].append(patient)
        elif sides == {1}:
            result["right_only"].append(patient)
        else:
            result["midline_or_multiside"].append(patient)
    return result


def _event_strata(event_counts: torch.Tensor) -> dict[str, list[int]]:
    return {
        "1": torch.nonzero(event_counts == 1, as_tuple=False).flatten().tolist(),
        "2": torch.nonzero(event_counts == 2, as_tuple=False).flatten().tolist(),
        "3_to_5": torch.nonzero(
            (event_counts >= 3) & (event_counts <= 5), as_tuple=False
        ).flatten().tolist(),
        "ge_6": torch.nonzero(event_counts >= 6, as_tuple=False).flatten().tolist(),
    }


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def main() -> int:
    _assert_pinned(SOURCE, EXPECTED_SOURCE_FILES)
    _assert_pinned(ANCHOR, EXPECTED_ANCHOR_FILES)
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_file_sha256"]["runner"] != EXPECTED_RUNNER_SHA256 or (
        manifest["source_file_sha256"]["hb_hsr_reasoner"] != EXPECTED_CORE_SHA256
    ):
        raise ValueError("recorded HB-HSR source lineage changed")
    if _sha256(ROOT / "scripts/run_labram_hb_hsr_oof_v15.py") != EXPECTED_RUNNER_SHA256 or (
        _sha256(ROOT / "src/soz/hb_hsr_reasoner.py") != EXPECTED_CORE_SHA256
    ):
        raise ValueError("current HB-HSR source no longer reproduces the run")

    tensors = load_file(str(SOURCE / "oof_predictions.safetensors"), device="cpu")
    states = load_file(str(SOURCE / "outer_fold_states.safetensors"), device="cpu")
    candidate = tensors[f"oof.{CANDIDATE_NAME}"].float()
    anchor = tensors[f"oof.{ANCHOR_NAME}"].float()
    targets = tensors["targets"].float()
    mask = tensors["target_mask"].bool()
    folds = tensors["patient_folds"].long()
    event_counts = tensors["patient_event_counts"].long()
    if tuple(candidate.shape) != (101, 19) or not torch.isfinite(candidate).all():
        raise ValueError("candidate prediction tensor is invalid")
    if not torch.equal(mask, V11_CANDIDATE_MASK.unsqueeze(0).expand_as(mask)):
        raise ValueError("fixed candidate mask changed")
    anchor_payload = load_file(str(ANCHOR / "oof_predictions.safetensors"), device="cpu")
    if not torch.equal(anchor, anchor_payload["oof.full_frozen_labram_plus_fine"]):
        raise ValueError("serialized anchor copy differs from pinned v11.1")

    candidate_metrics = _evaluate(candidate, targets, mask)
    anchor_metrics = _evaluate(anchor, targets, mask)
    recorded = manifest["primary_comparison"]
    _assert_metric_replay(candidate_metrics, recorded["candidate_metrics"])
    _assert_metric_replay(anchor_metrics, recorded["anchor_metrics"])
    paired = _paired_bootstrap(candidate, anchor, targets, mask)
    for endpoint in ("strict", "relaxed", "macro_ap", "mrr", "hit_at_3", "hit_at_5", "far_error"):
        for field in ("delta",):
            if not _close(
                float(paired[endpoint][field]),
                float(recorded["paired_candidate_minus_anchor"][endpoint][field]),
            ):
                raise ValueError(f"paired replay mismatch: {endpoint}/{field}")

    receipts = _verify_outer_receipts(manifest, states, targets, mask, folds)
    private_safe = (
        manifest["access_receipt"]["private_eeg_loaded"] is False
        and manifest["access_receipt"]["private_target_values_loaded"] is False
        and int(manifest["access_receipt"]["private_forward_count"]) == 0
        and manifest["claim_boundary"]["private_used"] is False
    )
    expected_candidate = float(_strict_contributions(candidate, targets, mask).sum())
    expected_anchor = float(_strict_contributions(anchor, targets, mask).sum())
    net_gain = expected_candidate - expected_anchor
    candidate_folds = tuple(float(value) for value in recorded["candidate_fold_strict"])
    anchor_folds = tuple(float(value) for value in recorded["anchor_fold_strict"])
    fold_nonlower = sum(left >= right for left, right in zip(candidate_folds, anchor_folds))
    derived_checks = {
        "strict_net_gain_at_least_5_of_101": net_gain >= 5.0,
        "strict_paired_ci_lower_strictly_positive": float(paired["strict"]["ci95"][0]) > 0.0,
        "macro_ap_paired_ci_lower_nonnegative": float(paired["macro_ap"]["ci95"][0]) >= 0.0,
        "relaxed_point_nonlower": float(candidate_metrics["top1"]["relaxed_accuracy"])
        >= float(anchor_metrics["top1"]["relaxed_accuracy"]),
        "far_error_not_above_23": float(candidate_metrics["far_error_count"]) <= 23.0,
        "four_of_five_fold_strict_nonlower": fold_nonlower >= 4,
        "foundation_optimizer_parameter_count_zero": receipts["all_outer_folds_passed"],
        "support_prior_transform_l2_held_label_isolation_verified": receipts[
            "all_outer_folds_passed"
        ],
        "private_access_and_forward_count_zero": private_safe,
    }
    decision = (
        "HB_HSR_RETAIN_AS_DEVELOPMENT_CANDIDATE"
        if all(derived_checks.values())
        else "HB_HSR_STOP_ON_CURRENT_PUBLIC_COHORT"
    )
    if decision != manifest["decision"]:
        raise ValueError("derived STOP decision differs from training manifest")

    rare_groups = _rare_strata(targets, folds, mask)
    side_groups = _side_strata(targets, mask)
    event_groups = _event_strata(event_counts)
    result = {
        "schema_version": SCHEMA,
        "status": "completed_independent_read_only_audit",
        "decision": decision,
        "scope": {
            "patient_count": 101,
            "event_count": 984,
            "model_fit_or_selection_performed": False,
            "raw_eeg_read": False,
            "private_read": False,
            "predictions_changed": False,
            "repeatedly_used_public_development_not_confirmation": True,
        },
        "pinned_source_bundle_sha256": EXPECTED_SOURCE_FILES,
        "pinned_anchor_sha256": EXPECTED_ANCHOR_FILES,
        "source_code_sha256": {
            "runner": EXPECTED_RUNNER_SHA256,
            "hb_hsr_reasoner": EXPECTED_CORE_SHA256,
        },
        "primary_replay": {
            "candidate_metrics": _metric_core(candidate_metrics),
            "anchor_metrics": _metric_core(anchor_metrics),
            "paired_candidate_minus_anchor": paired,
            "strict_expected_success_candidate": expected_candidate,
            "strict_expected_success_anchor": expected_anchor,
            "strict_net_gain": net_gain,
            "strict_win_loss_tie": _win_loss_tie(candidate, anchor, targets, mask),
            "candidate_contralateral_far_count": _contralateral_far_count(
                candidate, targets, mask
            ),
            "anchor_contralateral_far_count": _contralateral_far_count(
                anchor, targets, mask
            ),
            "candidate_fold_strict": list(candidate_folds),
            "anchor_fold_strict": list(anchor_folds),
            "fold_strict_nonlower_count": fold_nonlower,
        },
        "derived_frozen_stop_gate": {
            "all_passed": all(derived_checks.values()),
            "checks": derived_checks,
        },
        "outer_fold_leakage_and_state_receipts": receipts,
        "diagnostic_strata_post_hoc_not_for_selection": {
            "rare_positive": {
                name: _subset_metrics(indices, candidate, anchor, targets, mask)
                for name, indices in rare_groups.items()
            },
            "gold_scalp_coordinate_side": {
                name: _subset_metrics(indices, candidate, anchor, targets, mask)
                for name, indices in side_groups.items()
            },
            "event_count": {
                name: _subset_metrics(indices, candidate, anchor, targets, mask)
                for name, indices in event_groups.items()
            },
        },
        "publication_contract_findings": {
            "primary_predictions_and_metrics_valid": True,
            "outer_state_namespace_uses_legacy_full_arm_name": True,
            "namespace_mapping": {
                "outer{fold}.full_frozen_labram_plus_fine.*": f"outer{{fold}}.{CANDIDATE_NAME}.*"
            },
            "legacy_candidate_membership_sensitivity_is_not_HB_HSR": True,
            "legacy_sensitivity_excluded_from_HB_HSR_claims": True,
            "base_manifest_nested_full_arm_names_are_not_authoritative": True,
            "original_two_safety_gate_boole_were_hardcoded": True,
            "this_audit_rederived_all_safety_checks_fail_closed": True,
            "issues_affect_primary_oof_values": False,
            "issues_change_stop_decision": False,
        },
        "claim_boundary": {
            "private_used": False,
            "external_validation": False,
            "clinical_deployment_allowed": False,
            "further_HB_HSR_scans_on_current_101_allowed": False,
        },
    }

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not OUTPUT.parent.is_dir():
        raise FileNotFoundError(OUTPUT.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.tmp-", dir=OUTPUT.parent))
    published = False
    try:
        audit_path = staging / "audit.json"
        audit_path.write_bytes(_canonical_json(result))
        receipt = {
            "audit_sha256": _sha256(audit_path),
            "audit_size_bytes": audit_path.stat().st_size,
            "audit_script_sha256": _sha256(Path(__file__).resolve()),
        }
        (staging / "receipt.json").write_bytes(_canonical_json(receipt))
        os.replace(staging, OUTPUT)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({"path": str(OUTPUT), "decision": decision, "strict_net_gain": net_gain, "private_used": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
