#!/usr/bin/env python3
"""Run the frozen LaBraM 4 s versus native 8 s full-phase patient OOF test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_labram_native_context_source_train_v7 import (  # noqa: E402
    EXPECTED_EVENT_ORDER_SHA256,
    NATIVE_CONTEXT_CACHE_SCHEMA,
    load_native_context_cache,
)
from scripts.run_labram_frozen_h_nested_oof_v3 import (  # noqa: E402
    CROSSWALK_MANIFEST_SHA256,
    CROSSWALK_RECEIPT_SHA256,
    _load_frozen_h_source_train,
)
from scripts.run_labram_scalp_onset_contrast_oof_v6 import (  # noqa: E402
    BASE_SEED,
    EPOCHS,
    FULL_PHASE_CONTROL,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    WEIGHT_DECAY,
    _fit_candidate,
    _phase_availability,
    _predict_candidate,
)
from scripts.run_labram_temporal_mil_nested_oof_v1 import (  # noqa: E402
    _file_sha256,
    _indices_for_folds,
    _metrics,
)
from scripts.run_labram_v_directed_endpoint_oof_v5 import (  # noqa: E402
    _direction_payload,
    _paired_patient_bootstrap,
    _transition_diagnostic,
)
from src.soz.frozen_h_recovery import FrozenHPatientBatch  # noqa: E402
from src.soz.onset_contrast_recovery import (  # noqa: E402
    fit_onset_contrast_standardization,
    subset_onset_contrast_patient_batch,
)
from src.soz.safe_anchor_h_recovery import (  # noqa: E402
    within_tcp_edge_direction_metrics,
)
from src.soz.temporal_mil_recovery import (  # noqa: E402
    jeffreys_channel_prior_logits,
)


NATIVE_CONTEXT_OOF_SCHEMA = "soz_labram_native_8s_context_oof_v7"
PROTOCOL_PATH = (
    ROOT
    / "research/02_method/"
    "labram_native_8s_context_recovery_protocol_v7_20260811_zh.md"
)
MATERIALIZER_PATH = (
    ROOT / "scripts/materialize_labram_native_context_source_train_v7.py"
)
FOUNDATION_MODULE_PATH = ROOT / "src/soz/models/foundation.py"
V6_RESULT_PATH = ROOT / "outputs/labram_scalp_onset_contrast_oof_v6_1_20260811"
V6_MANIFEST_SHA256 = "4f7b1f2edeac292c34b0a3053873bf4eb50ed31d8d87149efc4c2e4546967230"
V6_PREDICTION_SHA256 = "130cf66701b2d8d45959eacf86948eefa743bf0b5a48a243ad29dc7994a0befa"
OUTER_FOLDS = tuple(range(5))
FOUR_SECOND = "labram4_nonoverlap_full_phase_h_v"
EIGHT_SECOND = "labram8_stride4_full_phase_h_v"
ANCHOR = "temporal_mil_exact_anchor"
V6_REPLAY_TOLERANCE = 1e-4


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _load_v6_comparators(
    full: FrozenHPatientBatch,
    patient_folds: Sequence[int],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    manifest_path = V6_RESULT_PATH / "manifest.json"
    prediction_path = V6_RESULT_PATH / "oof_predictions.safetensors"
    if _file_sha256(manifest_path) != V6_MANIFEST_SHA256:
        raise ValueError("v6.1 manifest changed")
    if _file_sha256(prediction_path) != V6_PREDICTION_SHA256:
        raise ValueError("v6.1 prediction file changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest["patient_ids"]) != full.base.patient_ids or tuple(
        manifest["patient_folds"]
    ) != tuple(patient_folds):
        raise ValueError("v6.1 patient carrier differs")
    if manifest.get("source_dev_forward_count") != 0 or manifest.get(
        "source_eval_used"
    ) is not False or manifest.get("private_used") is not False:
        raise ValueError("v6.1 comparator violates the development boundary")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required") from exc
    tensors = load_file(str(prediction_path), device="cpu")
    required = {
        "full_phase_h_v_matched",
        "temporal_mil_exact_anchor",
        "targets",
        "target_mask",
        "patient_folds",
    }
    if not required <= set(tensors):
        raise ValueError("v6.1 comparator tensors are incomplete")
    if not torch.equal(tensors["targets"], full.base.targets.cpu()) or not torch.equal(
        tensors["target_mask"], full.base.target_mask.cpu()
    ):
        raise ValueError("v6.1 target carrier differs")
    if not torch.equal(
        tensors["patient_folds"], torch.tensor(patient_folds, dtype=torch.int64)
    ):
        raise ValueError("v6.1 fold carrier differs")
    return {
        FOUR_SECOND: tensors["full_phase_h_v_matched"].float().contiguous(),
        ANCHOR: tensors["temporal_mil_exact_anchor"].float().contiguous(),
    }, manifest


def _load_matched_inputs(
    native_cache: Path,
    expected_native_manifest_sha256: str,
) -> tuple[
    FrozenHPatientBatch,
    FrozenHPatientBatch,
    tuple[int, ...],
    dict[str, torch.Tensor],
    dict[str, object],
]:
    full_4s, patient_folds, lineage = _load_frozen_h_source_train()
    cache = load_native_context_cache(
        native_cache,
        expected_manifest_sha256=expected_native_manifest_sha256,
    )
    manifest = cache.manifest
    checks = {
        "schema": manifest["schema_version"] == NATIVE_CONTEXT_CACHE_SCHEMA,
        "full scope": manifest["full_scope"] is True,
        "not smoke": manifest["smoke_only"] is False,
        "event count": manifest["event_count"] == full_4s.base.evidence.batch_size,
        "patient count": manifest["patient_count"] == len(full_4s.base.patient_ids),
        "event order": manifest["event_order_sha256"]
        == EXPECTED_EVENT_ORDER_SHA256,
        "crosswalk manifest": manifest["crosswalk_manifest_sha256"]
        == CROSSWALK_MANIFEST_SHA256,
        "crosswalk receipt": manifest["crosswalk_receipt_sha256"]
        == CROSSWALK_RECEIPT_SHA256,
        "lineage event order": lineage["crosswalk_event_order_sha256"]
        == manifest["event_order_sha256"],
        "patient order": tuple(manifest["patient_ids"])
        == full_4s.base.patient_ids,
        "no targets": manifest["deepsoz_target_values_loaded"] is False,
        "no dev": manifest["source_dev_used"] is False,
        "no eval": manifest["source_eval_used"] is False,
        "no private": manifest["private_used"] is False,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"native-context/full-source join failed: {failed}")
    node_tokens_8s = cache.tokens.reshape(
        cache.tokens.shape[0], 19, 15, 4, 200
    ).contiguous()
    full_8s = FrozenHPatientBatch(base=full_4s.base, node_tokens=node_tokens_8s)
    comparators, v6_manifest = _load_v6_comparators(full_4s, patient_folds)
    joined_lineage = {
        **lineage,
        "native_cache_path": str(cache.path.relative_to(ROOT)),
        "native_cache_manifest_sha256": cache.manifest_sha256,
        "native_cache_tensor_file_sha256": manifest["tensor_file_sha256"],
        "native_cache_event_order_sha256": manifest["event_order_sha256"],
        "native_cache_feature_receipt_sha256": manifest[
            "foundation_feature_receipt_sha256"
        ],
        "v6_manifest_sha256": V6_MANIFEST_SHA256,
        "v6_prediction_sha256": V6_PREDICTION_SHA256,
    }
    return full_4s, full_8s, patient_folds, comparators, {
        "lineage": joined_lineage,
        "native_manifest": dict(manifest),
        "v6_manifest_status": v6_manifest["status"],
    }


def _fit_and_predict(
    full: FrozenHPatientBatch,
    train_indices: Sequence[int],
    held_indices: Sequence[int],
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object], dict[str, object]]:
    train = subset_onset_contrast_patient_batch(full, train_indices)
    held = subset_onset_contrast_patient_batch(full, held_indices)
    standardization = fit_onset_contrast_standardization(train)
    prior = jeffreys_channel_prior_logits(train.base).detach().cpu()
    model, fit = _fit_candidate(
        train,
        standardization,
        prior,
        candidate=FULL_PHASE_CONTROL,
        seed=seed,
        device=device,
    )
    logits, diagnostics = _predict_candidate(model, held, device=device)
    del model, train, held
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return logits, fit, diagnostics


def _run(
    full_4s: FrozenHPatientBatch,
    full_8s: FrozenHPatientBatch,
    patient_folds: tuple[int, ...],
    frozen_comparators: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    patients = len(full_4s.base.patient_ids)
    oof = {
        FOUR_SECOND: torch.full((patients, 19), torch.nan),
        EIGHT_SECOND: torch.full((patients, 19), torch.nan),
    }
    outer_rows: list[dict[str, object]] = []
    fold_strict_nonlower_vs_4s = 0
    fold_strict_nonlower_vs_anchor = 0

    for fold in OUTER_FOLDS:
        train_indices = _indices_for_folds(
            patient_folds, tuple(value for value in OUTER_FOLDS if value != fold)
        )
        held_indices = _indices_for_folds(patient_folds, (fold,))
        seed = BASE_SEED + fold * 1000
        candidate_rows: dict[str, object] = {}
        for name, full in ((FOUR_SECOND, full_4s), (EIGHT_SECOND, full_8s)):
            logits, fit, diagnostics = _fit_and_predict(
                full,
                train_indices,
                held_indices,
                seed=seed,
                device=device,
            )
            oof[name][list(held_indices)] = logits
            held = subset_onset_contrast_patient_batch(full, held_indices)
            direction = within_tcp_edge_direction_metrics(
                logits, held.base.targets, held.base.target_mask
            )
            candidate_rows[name] = {
                "fit": fit,
                "diagnostics": diagnostics,
                "metrics": _metrics(
                    logits, held.base.targets, held.base.target_mask
                ),
                "within_tcp_direction": _direction_payload(direction),
            }
            del held
        held_index = torch.tensor(held_indices, dtype=torch.long)
        anchor_held = frozen_comparators[ANCHOR].index_select(0, held_index)
        held_base = subset_onset_contrast_patient_batch(full_4s, held_indices).base
        anchor_metrics = _metrics(
            anchor_held, held_base.targets, held_base.target_mask
        )
        anchor_direction = _direction_payload(
            within_tcp_edge_direction_metrics(
                anchor_held, held_base.targets, held_base.target_mask
            )
        )
        strict_4s = candidate_rows[FOUR_SECOND]["metrics"]["top1"][
            "strict_accuracy"
        ]
        strict_8s = candidate_rows[EIGHT_SECOND]["metrics"]["top1"][
            "strict_accuracy"
        ]
        strict_anchor = anchor_metrics["top1"]["strict_accuracy"]
        fold_strict_nonlower_vs_4s += int(strict_8s >= strict_4s)
        fold_strict_nonlower_vs_anchor += int(strict_8s >= strict_anchor)
        outer_rows.append(
            {
                "outer_fold": fold,
                "train_patient_count": len(train_indices),
                "held_patient_count": len(held_indices),
                "same_seed": seed,
                "candidates": candidate_rows,
                "anchor_metrics": anchor_metrics,
                "anchor_within_tcp_direction": anchor_direction,
                "primary_strict_nonlower_than_4s": strict_8s >= strict_4s,
                "primary_strict_nonlower_than_anchor": strict_8s >= strict_anchor,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "outer_complete",
                    "fold": fold,
                    "strict_4s": strict_4s,
                    "strict_8s": strict_8s,
                    "strict_anchor": strict_anchor,
                    "ap_8s": candidate_rows[EIGHT_SECOND]["metrics"]["ranking"][
                        "macro_average_precision"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if any(not torch.isfinite(value).all() for value in oof.values()):
        raise RuntimeError("native-context OOF left predictions unfilled")
    replay_error = float(
        (oof[FOUR_SECOND] - frozen_comparators[FOUR_SECOND]).abs().max()
    )
    if replay_error > V6_REPLAY_TOLERANCE:
        raise RuntimeError(
            f"4 s v6.1 replay drift {replay_error} exceeds {V6_REPLAY_TOLERANCE}"
        )

    scores = {
        FOUR_SECOND: oof[FOUR_SECOND],
        EIGHT_SECOND: oof[EIGHT_SECOND],
        ANCHOR: frozen_comparators[ANCHOR],
    }
    metrics = {
        name: _metrics(value, full_4s.base.targets, full_4s.base.target_mask)
        for name, value in scores.items()
    }
    directions = {
        name: _direction_payload(
            within_tcp_edge_direction_metrics(
                value, full_4s.base.targets, full_4s.base.target_mask
            )
        )
        for name, value in scores.items()
    }
    transition_vs_4s = _transition_diagnostic(
        scores[EIGHT_SECOND],
        scores[FOUR_SECOND],
        full_4s.base.targets,
        full_4s.base.target_mask,
    )
    transition_vs_anchor = _transition_diagnostic(
        scores[EIGHT_SECOND],
        scores[ANCHOR],
        full_4s.base.targets,
        full_4s.base.target_mask,
    )
    primary = metrics[EIGHT_SECOND]
    control = metrics[FOUR_SECOND]
    anchor = metrics[ANCHOR]
    primary_direction = directions[EIGHT_SECOND]["patient_macro_accuracy"]
    control_direction = directions[FOUR_SECOND]["patient_macro_accuracy"]
    anchor_direction = directions[ANCHOR]["patient_macro_accuracy"]
    interface_checks = {
        "strict_nonlower_than_4s": primary["top1"]["strict_accuracy"]
        >= control["top1"]["strict_accuracy"],
        "macro_ap_nonlower_than_4s": primary["ranking"]["macro_average_precision"]
        >= control["ranking"]["macro_average_precision"],
        "tcp_direction_nonlower_than_4s": primary_direction >= control_direction,
        "far_errors_nonincreasing_vs_4s": transition_vs_4s[
            "far_errors_nonincreasing"
        ],
        "strict_nonlower_in_at_least_3_of_5_folds_vs_4s": (
            fold_strict_nonlower_vs_4s >= 3
        ),
        "strict_or_ap_strictly_improved_vs_4s": (
            primary["top1"]["strict_accuracy"]
            > control["top1"]["strict_accuracy"]
            or primary["ranking"]["macro_average_precision"]
            > control["ranking"]["macro_average_precision"]
        ),
    }
    promotion_checks = {
        "strict_strictly_higher_than_anchor": primary["top1"]["strict_accuracy"]
        > anchor["top1"]["strict_accuracy"],
        "macro_ap_strictly_higher_than_anchor": primary["ranking"][
            "macro_average_precision"
        ]
        > anchor["ranking"]["macro_average_precision"],
        "tcp_direction_strictly_higher_than_anchor": primary_direction
        > anchor_direction,
        "far_errors_strictly_lower_than_anchor": transition_vs_anchor[
            "candidate_far_count"
        ]
        < transition_vs_anchor["anchor_far_count"],
        "strict_nonlower_in_at_least_3_of_5_folds_vs_anchor": (
            fold_strict_nonlower_vs_anchor >= 3
        ),
    }
    interface_go = all(interface_checks.values())
    promotion_go = all(promotion_checks.values())
    if interface_go and promotion_go:
        decision = "go_promote_native_8s_after_external_validation"
    elif interface_go:
        decision = "interface_go_but_keep_temporal_mil_exact_anchor"
    else:
        decision = "no_go_keep_temporal_mil_exact_next_minimal_peft"

    final_states: dict[str, torch.Tensor] = {}
    final_fits: dict[str, object] = {}
    for name, full in ((FOUR_SECOND, full_4s), (EIGHT_SECOND, full_8s)):
        standardization = fit_onset_contrast_standardization(full)
        prior = jeffreys_channel_prior_logits(full.base).detach().cpu()
        model, fit = _fit_candidate(
            full,
            standardization,
            prior,
            candidate=FULL_PHASE_CONTROL,
            seed=BASE_SEED + 99999,
            device=device,
        )
        for key, value in model.state_dict().items():
            final_states[f"{name}.{key}"] = value.detach().cpu().contiguous()
        final_fits[name] = fit
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "screen_kind": (
            "post_hoc_source_train_patient_oof_native_context_interface_test"
        ),
        "metrics": metrics,
        "within_tcp_direction": directions,
        "paired_bootstrap_8s_vs_4s": _paired_patient_bootstrap(
            scores[EIGHT_SECOND],
            scores[FOUR_SECOND],
            full_4s.base.targets,
            full_4s.base.target_mask,
        ),
        "paired_bootstrap_8s_vs_anchor": _paired_patient_bootstrap(
            scores[EIGHT_SECOND],
            scores[ANCHOR],
            full_4s.base.targets,
            full_4s.base.target_mask,
        ),
        "top1_transition_8s_vs_4s": transition_vs_4s,
        "top1_transition_8s_vs_anchor": transition_vs_anchor,
        "outer_folds": outer_rows,
        "fold_strict_nonlower_count_vs_4s": fold_strict_nonlower_vs_4s,
        "fold_strict_nonlower_count_vs_anchor": fold_strict_nonlower_vs_anchor,
        "v6_4s_replay_max_abs_error": replay_error,
        "v6_4s_replay_tolerance": V6_REPLAY_TOLERANCE,
        "interface_gate_checks": interface_checks,
        "interface_go": interface_go,
        "promotion_gate_checks": promotion_checks,
        "promotion_go": promotion_go,
        "decision": decision,
        "final_fits": final_fits,
    }
    tensors = {
        **{name: value.contiguous() for name, value in scores.items()},
        "v6_frozen_4s_comparator": frozen_comparators[FOUR_SECOND].contiguous(),
        "targets": full_4s.base.targets.cpu().contiguous(),
        "target_mask": full_4s.base.target_mask.cpu().contiguous(),
        "patient_folds": torch.tensor(patient_folds, dtype=torch.int64),
    }
    return result, tensors, final_states


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--native-cache", type=Path, required=True)
    parser.add_argument(
        "--expected-native-cache-manifest-sha256", type=str, required=True
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.preflight_only and args.output_directory is None:
        raise ValueError("full OOF requires --output-directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    full_4s, full_8s, patient_folds, comparators, receipts = _load_matched_inputs(
        args.native_cache,
        args.expected_native_cache_manifest_sha256.strip().lower(),
    )
    preflight = {
        "status": "ready_native_8s_matched_patient_oof",
        "schema_version": NATIVE_CONTEXT_OOF_SCHEMA,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": _file_sha256(PROTOCOL_PATH),
        "runner_sha256": _file_sha256(Path(__file__).resolve()),
        "materializer_sha256": _file_sha256(MATERIALIZER_PATH),
        "foundation_module_sha256": _file_sha256(FOUNDATION_MODULE_PATH),
        "device": str(device),
        "patient_count": len(full_4s.base.patient_ids),
        "event_count": full_4s.base.evidence.batch_size,
        "patient_ids": list(full_4s.base.patient_ids),
        "patient_folds": list(patient_folds),
        "fold_counts": {
            str(fold): sum(value == fold for value in patient_folds)
            for fold in OUTER_FOLDS
        },
        "phase_availability": _phase_availability(full_4s),
        "lineage": receipts["lineage"],
        "native_cache_contract": {
            key: receipts["native_manifest"][key]
            for key in (
                "context_seconds",
                "stride_seconds",
                "start_seconds",
                "coverage_counts",
                "aggregation",
                "pipeline_control_max_abs_error",
                "foundation_trainable_parameter_count",
                "raw_replay_verified",
            )
        },
        "config": {
            "primary": EIGHT_SECOND,
            "matched_control": FOUR_SECOND,
            "safe_anchor": ANCHOR,
            "candidate_selection": False,
            "outer_folds": list(OUTER_FOLDS),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "base_seed": BASE_SEED,
            "patient_event_pooling": "equal_event_logit_mean",
            "neighbor_training_auxiliary": False,
            "foundation_frozen": True,
        },
        "foundation_backbone": "official_pretrained_LaBraM_frozen_not_replaced",
        "foundation_trainable_parameter_count": 0,
        "source_dev_forward_count": 0,
        "source_eval_forward_count": 0,
        "private_forward_count": 0,
        "formal_promotion": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True), flush=True)
        return 0

    output = Path(os.path.abspath(args.output_directory))
    if output.name in {"", ".", ".."} or os.path.lexists(output):
        raise FileExistsError("OOF output exists or is invalid")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise FileNotFoundError(output.parent)
    result, tensors, final_states = _run(
        full_4s,
        full_8s,
        patient_folds,
        comparators,
        device=device,
    )
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required") from exc
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        prediction_path = staging / "oof_predictions.safetensors"
        checkpoint_path = staging / "final_checkpoints.safetensors"
        save_file(tensors, str(prediction_path))
        save_file(final_states, str(checkpoint_path))
        manifest = {
            **preflight,
            "status": "completed_development_only",
            "result": result,
            "files": {
                prediction_path.name: {
                    "sha256": _file_sha256(prediction_path),
                    "size_bytes": prediction_path.stat().st_size,
                },
                checkpoint_path.name: {
                    "sha256": _file_sha256(checkpoint_path),
                    "size_bytes": checkpoint_path.stat().st_size,
                },
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(_canonical_bytes(manifest))
        manifest_sha = _file_sha256(manifest_path)
        os.rename(staging, output)
        published = True
        print(
            json.dumps(
                {
                    "status": "completed_native_8s_context_oof",
                    "path": str(output),
                    "manifest_sha256": manifest_sha,
                    "decision": result["decision"],
                    "interface_go": result["interface_go"],
                    "promotion_go": result["promotion_go"],
                    "metrics": result["metrics"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
