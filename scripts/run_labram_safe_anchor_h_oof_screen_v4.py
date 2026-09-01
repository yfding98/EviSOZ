#!/usr/bin/env python3
"""Replay the post-hoc Top-1-safe LaBraM-H source-train OOF screen.

This command trains nothing.  It combines three immutable outer-OOF artifacts
with a target-free transform, verifies exact Top-1-set preservation, and only
then evaluates source-train labels.  Endpoint-flip proposals are saved as a
target-free diagnostic but are never applied because fresh inner-OOF stacking
features are absent and the within-TCP direction prerequisite failed.
"""

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

from scripts.run_labram_temporal_mil_nested_oof_v1 import (  # noqa: E402
    BASE_SEED,
    BOOTSTRAP_REPLICATES,
    _canonical_bytes,
    _file_sha256,
    _metrics,
)
from src.soz.geometry import STANDARD_19  # noqa: E402
from src.soz.safe_anchor_h_recovery import (  # noqa: E402
    SAFE_ANCHOR_H_RECOVERY_SCHEMA,
    prior_cancelled_log_probability_ratio,
    propose_tcp_endpoint_flips,
    top1_safe_bounded_residual,
    within_tcp_edge_direction_metrics,
)


PROTOCOL_PATH = (
    ROOT
    / "research/02_method/"
    "labram_safe_anchor_h_recovery_protocol_v4_20260811_zh.md"
)
TEMPORAL_PATH = ROOT / "outputs/labram_temporal_mil_nested_oof_v1_20260810"
FROZEN_H_PATH = ROOT / "outputs/labram_frozen_h_nested_oof_v3_20260810"
SHORTCUT_PATH = ROOT / "outputs/labram_frozen_h_shortcut_controls_v3_1_20260811"
SAFE_ANCHOR_MODULE_PATH = ROOT / "src/soz/safe_anchor_h_recovery.py"
EXPECTED_PROTOCOL_SHA256 = (
    "e91f02545d1b5f5acc43c54f9f82c8ad70317ae9d309862a3d7426a968d7853e"
)

PRIMARY_MASK_POLICY = {
    "carrier_channels": list(STANDARD_19),
    "evaluable_channels": [channel for channel in STANDARD_19 if channel != "PZ"],
    "excluded_channels": ["PZ"],
    "semantics": (
        "fixed DeepSOZ primary benchmark mask; standard-19 carrier with "
        "canonical PZ excluded because its duplicated source columns are ambiguous"
    ),
}
PRIMARY_MASK_RECEIPT_SHA256 = hashlib.sha256(
    _canonical_bytes(PRIMARY_MASK_POLICY)
).hexdigest()

TEMPORAL_MANIFEST_SHA256 = (
    "58cbfcc3d25e8ff4b13ab93e388e8aa5691e1c8fc9dc515ec2e8b51b226c9811"
)
TEMPORAL_PREDICTION_SHA256 = (
    "9373dc6bf269002c812ae26ca6ea8365b7518d3396037c4fc5b3a67603e1211d"
)
FROZEN_H_MANIFEST_SHA256 = (
    "285609d9bba2d17fc728541fadabb5a272ee36e0c0b53d2be5ac865648aa04b6"
)
FROZEN_H_PREDICTION_SHA256 = (
    "4bacebb5dd3f616ebd6a039d51364864791b954c27e8058dc536a3fc7b83c9d1"
)
SHORTCUT_MANIFEST_SHA256 = (
    "24d5c4a220f128613fb0a3d3b26233b8afeaac1b28ae78445b63a80648822fa2"
)
SHORTCUT_PREDICTION_SHA256 = (
    "c18579696da12a087567b8712214df5dde25c5e5f1a513c02c31c103349d8183"
)


def _strict_json(raw: bytes, *, field_name: str) -> object:
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant in {field_name}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} is not strict UTF-8 JSON") from exc
    return value


def _load_oof_artifact(
    directory: Path,
    *,
    expected_manifest_sha256: str,
    expected_prediction_sha256: str,
) -> tuple[Mapping[str, object], dict[str, torch.Tensor]]:
    manifest_path = directory / "manifest.json"
    prediction_path = directory / "oof_predictions.safetensors"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"OOF artifact is not a regular directory: {directory}")
    for path in (manifest_path, prediction_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"OOF member is not a regular file: {path}")
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError(f"OOF manifest SHA changed: {directory.name}")
    manifest = _strict_json(raw, field_name=f"{directory.name} manifest")
    if not isinstance(manifest, dict) or _canonical_bytes(manifest) != raw:
        raise ValueError(f"OOF manifest is not canonical: {directory.name}")
    if _file_sha256(prediction_path) != expected_prediction_sha256:
        raise ValueError(f"OOF prediction SHA changed: {directory.name}")
    file_record = manifest.get("files", {}).get("oof_predictions.safetensors")
    if not isinstance(file_record, dict) or file_record.get("sha256") != (
        expected_prediction_sha256
    ) or file_record.get("size_bytes") != prediction_path.stat().st_size:
        raise ValueError(f"OOF prediction receipt changed: {directory.name}")
    boundary = {
        "patient_count": 65,
        "event_count": 582,
        "source_dev_forward_count": 0,
        "source_dev_target_values_reachable": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_promotion": False,
    }
    changed = tuple(name for name, value in boundary.items() if manifest.get(name) != value)
    if changed:
        raise ValueError(f"OOF scientific boundary changed: {directory.name}: {changed}")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for safe OOF replay") from exc
    return manifest, load_file(str(prediction_path), device="cpu")


def _required_tensor(
    tensors: Mapping[str, torch.Tensor],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if name not in tensors:
        raise ValueError(f"required OOF tensor is missing: {name}")
    value = tensors[name]
    if tuple(value.shape) != shape or value.dtype != dtype:
        raise ValueError(f"OOF tensor schema changed: {name}")
    return value.contiguous()


def _direction_payload(report) -> dict[str, object]:
    return {
        "patient_macro_accuracy": report.patient_macro_accuracy,
        "eligible_patient_count": report.eligible_patient_count,
        "informative_pair_count": report.informative_pair_count,
        "semantics": (
            "within fixed TCP_20 edge; exactly one observed positive endpoint and "
            "one observed negative endpoint; patient-macro; ties equal 0.5; no "
            "pseudo-label creation"
        ),
    }


def _top_set(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    maximum = scores.masked_fill(~mask, -torch.inf).max(dim=1, keepdim=True).values
    return mask & (scores == maximum)


def _fixed_primary_evaluable_mask(patient_count: int) -> torch.Tensor:
    """Return the label-value-independent DeepSOZ primary deployment mask."""

    row = torch.tensor(
        [channel != "PZ" for channel in STANDARD_19], dtype=torch.bool
    )
    return row.unsqueeze(0).expand(patient_count, -1).clone()


def _paired_patient_bootstrap(
    candidate: torch.Tensor,
    anchor: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, object]:
    """Return deterministic patient-paired deltas for all reported endpoints."""

    rows: list[torch.Tensor] = []
    for patient in range(targets.shape[0]):
        index = slice(patient, patient + 1)
        candidate_metrics = _metrics(
            candidate[index], targets[index], target_mask[index]
        )
        anchor_metrics = _metrics(anchor[index], targets[index], target_mask[index])
        rows.append(
            torch.tensor(
                [
                    candidate_metrics["top1"]["strict_accuracy"]
                    - anchor_metrics["top1"]["strict_accuracy"],
                    candidate_metrics["top1"]["relaxed_accuracy"]
                    - anchor_metrics["top1"]["relaxed_accuracy"],
                    candidate_metrics["ranking"]["macro_average_precision"]
                    - anchor_metrics["ranking"]["macro_average_precision"],
                    candidate_metrics["ranking"]["mean_reciprocal_rank"]
                    - anchor_metrics["ranking"]["mean_reciprocal_rank"],
                    candidate_metrics["ranking"]["hit_at_k"][3]
                    - anchor_metrics["ranking"]["hit_at_k"][3],
                    candidate_metrics["ranking"]["hit_at_k"][5]
                    - anchor_metrics["ranking"]["hit_at_k"][5],
                ],
                dtype=torch.float64,
            )
        )
    row_deltas = torch.stack(rows)
    generator = torch.Generator().manual_seed(BASE_SEED)
    indices = torch.randint(
        0,
        targets.shape[0],
        (BOOTSTRAP_REPLICATES, targets.shape[0]),
        generator=generator,
    )
    samples = row_deltas[indices].mean(dim=1)
    point = row_deltas.mean(dim=0)
    names = (
        "strict_top1",
        "relaxed_top1",
        "macro_ap",
        "mrr",
        "hit_at_3",
        "hit_at_5",
    )
    return {
        name: {
            "delta": float(point[column]),
            "ci95": [
                float(torch.quantile(samples[:, column], 0.025)),
                float(torch.quantile(samples[:, column], 0.975)),
            ],
            "patient_improved_count": int((row_deltas[:, column] > 0).sum().item()),
            "patient_worsened_count": int((row_deltas[:, column] < 0).sum().item()),
            "patient_equal_count": int((row_deltas[:, column] == 0).sum().item()),
        }
        for column, name in enumerate(names)
    }


def _run() -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    temporal_manifest, temporal = _load_oof_artifact(
        TEMPORAL_PATH,
        expected_manifest_sha256=TEMPORAL_MANIFEST_SHA256,
        expected_prediction_sha256=TEMPORAL_PREDICTION_SHA256,
    )
    h_manifest, h_tensors = _load_oof_artifact(
        FROZEN_H_PATH,
        expected_manifest_sha256=FROZEN_H_MANIFEST_SHA256,
        expected_prediction_sha256=FROZEN_H_PREDICTION_SHA256,
    )
    shortcut_manifest, shortcut = _load_oof_artifact(
        SHORTCUT_PATH,
        expected_manifest_sha256=SHORTCUT_MANIFEST_SHA256,
        expected_prediction_sha256=SHORTCUT_PREDICTION_SHA256,
    )
    if temporal_manifest.get("patient_ids") != h_manifest.get("patient_ids") or (
        temporal_manifest.get("patient_ids") != shortcut_manifest.get("patient_ids")
    ):
        raise ValueError("OOF patient rosters differ")
    if temporal_manifest.get("patient_folds") != h_manifest.get("patient_folds") or (
        temporal_manifest.get("patient_folds")
        != shortcut_manifest.get("patient_folds")
    ):
        raise ValueError("OOF patient fold manifests differ")

    anchor = _required_tensor(
        temporal, "temporal_mil_exact", shape=(65, 19), dtype=torch.float32
    )
    targets = _required_tensor(
        temporal, "targets", shape=(65, 19), dtype=torch.float32
    )
    target_mask = _required_tensor(
        temporal, "target_mask", shape=(65, 19), dtype=torch.bool
    )
    deployment_prediction_mask = _fixed_primary_evaluable_mask(65)
    if not torch.equal(target_mask, deployment_prediction_mask):
        raise ValueError(
            "OOF target mask differs from the fixed 18-channel DeepSOZ primary mask"
        )
    patient_folds = _required_tensor(
        temporal, "patient_folds", shape=(65,), dtype=torch.int64
    )
    for name, tensors in (("frozen-H", h_tensors), ("shortcut", shortcut)):
        if not torch.equal(tensors.get("targets"), targets) or not torch.equal(
            tensors.get("target_mask"), target_mask
        ) or not torch.equal(tensors.get("patient_folds"), patient_folds):
            raise ValueError(f"{name} target/fold tensors differ from temporal anchor")

    h_probability = _required_tensor(
        h_tensors,
        "probability__frozen_h_uniform",
        shape=(65, 19),
        dtype=torch.float32,
    )
    q_probability = _required_tensor(
        shortcut,
        "probability__zero_h_q_only",
        shape=(65, 19),
        dtype=torch.float32,
    )

    # Score composition does not read target values or a patient-specific
    # availability pattern.  The fixed benchmark/deployment mask is derived
    # from the frozen PZ policy above and separately checked against the OOF
    # evaluation mask.
    h_residual = prior_cancelled_log_probability_ratio(
        h_probability, q_probability
    )
    safe = top1_safe_bounded_residual(
        anchor, h_residual, deployment_prediction_mask
    )
    proposal = propose_tcp_endpoint_flips(
        anchor, h_residual, deployment_prediction_mask
    )
    anchor_top = _top_set(anchor, deployment_prediction_mask)
    safe_top = _top_set(safe.scores, deployment_prediction_mask)
    top_set_change_count = int((anchor_top != safe_top).any(dim=1).sum().item())
    if top_set_change_count != 0:
        raise RuntimeError("safe OOF screen changed an anchor Top-1 set")

    anchor_metrics = _metrics(anchor, targets, target_mask)
    safe_metrics = _metrics(safe.scores, targets, target_mask)
    if safe_metrics["top1"]["strict_accuracy"] != anchor_metrics["top1"][
        "strict_accuracy"
    ] or safe_metrics["top1"]["relaxed_accuracy"] != anchor_metrics["top1"][
        "relaxed_accuracy"
    ]:
        raise RuntimeError("safe residual changed a Top-1 endpoint")

    anchor_direction = within_tcp_edge_direction_metrics(
        anchor, targets, target_mask
    )
    residual_direction = within_tcp_edge_direction_metrics(
        h_residual, targets, target_mask
    )
    safe_direction = within_tcp_edge_direction_metrics(
        safe.scores, targets, target_mask
    )
    ap_nonlower = safe_metrics["ranking"]["macro_average_precision"] >= (
        anchor_metrics["ranking"]["macro_average_precision"]
    )
    hit3_nonlower = safe_metrics["ranking"]["hit_at_k"][3] >= (
        anchor_metrics["ranking"]["hit_at_k"][3]
    )
    result = {
        "screen_kind": "post_hoc_source_train_outer_oof_mechanism_screen",
        "anchor": "temporal_mil_exact",
        "residual": (
            "log(patient_probability_frozen_h_uniform)_minus_"
            "log(patient_probability_fold_local_q_only)"
        ),
        "safe_candidate": "top1_safe_h_residual",
        "metrics": {
            "temporal_mil_exact": anchor_metrics,
            "top1_safe_h_residual": safe_metrics,
        },
        "safe_minus_anchor_paired_patient_bootstrap": _paired_patient_bootstrap(
            safe.scores, anchor, targets, target_mask
        ),
        "top1_invariant": {
            "complete_top_tie_set_change_count": top_set_change_count,
            "top_set_value_change_count": int(
                (safe.scores[anchor_top] != anchor[anchor_top]).sum().item()
            ),
            "strict_metric_identical": True,
            "relaxed_metric_identical": True,
            "changed_lower_rank_patient_count": safe.changed_patient_count,
            "budget_fraction": 0.5,
        },
        "within_tcp_direction_diagnostic": {
            "temporal_mil_exact": _direction_payload(anchor_direction),
            "h_prior_cancelled_residual": _direction_payload(residual_direction),
            "top1_safe_h_residual": _direction_payload(safe_direction),
            "h_residual_minus_anchor_accuracy": (
                residual_direction.patient_macro_accuracy
                - anchor_direction.patient_macro_accuracy
            ),
        },
        "secondary_ranking_gate": {
            "macro_ap_nonlower": ap_nonlower,
            "hit_at_3_nonlower": hit3_nonlower,
            "pass": ap_nonlower and hit3_nonlower,
            "interpretation": (
                "post-hoc point-estimate development screen; not a statistical "
                "noninferiority test"
            ),
            "status": (
                "secondary_ranking_point_estimate_screen_pass_post_hoc_only"
                if ap_nonlower and hit3_nonlower
                else "no_go_keep_temporal_exact"
            ),
        },
        "endpoint_flip": {
            "status": "blocked_not_applied",
            "target_free_proposal_count_diagnostic_only": proposal.proposal_count,
            "application_count": 0,
            "fresh_outer_train_inner_oof_gate_available": False,
            "within_tcp_direction_prerequisite_pass": (
                residual_direction.patient_macro_accuracy
                > anchor_direction.patient_macro_accuracy
            ),
            "reason": (
                "H residual direction is worse than temporal exact and existing "
                "global OOF rows cannot fit a strict nested endpoint gate"
            ),
        },
    }
    tensors = {
        "temporal_mil_exact": anchor,
        "top1_safe_h_residual": safe.scores.contiguous(),
        "h_prior_cancelled_log_probability_ratio": h_residual.contiguous(),
        "safe_residual_delta": safe.delta.contiguous(),
        "safe_margin_budget": safe.margin_budget.contiguous(),
        "anchor_top_set": safe.anchor_top_set.contiguous(),
        "tcp_endpoint_proposal": proposal.proposed.contiguous(),
        "tcp_endpoint_anchor_index": proposal.anchor_index.contiguous(),
        "tcp_endpoint_candidate_index": proposal.candidate_index.contiguous(),
        "tcp_endpoint_residual_margin": proposal.residual_margin.contiguous(),
        "targets": targets,
        "target_mask": target_mask,
        "deployment_prediction_mask": deployment_prediction_mask,
        "patient_folds": patient_folds,
    }
    return result, tensors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.preflight_only and args.output_directory is None:
        raise ValueError("full OOF screen requires --output-directory")
    protocol_sha256 = _file_sha256(PROTOCOL_PATH)
    if protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("safe-anchor v4 protocol SHA changed")
    preflight = {
        "status": "ready_post_hoc_safe_outer_oof_replay",
        "schema_version": SAFE_ANCHOR_H_RECOVERY_SCHEMA,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": protocol_sha256,
        "patient_count": 65,
        "event_count": 582,
        "foundation_trainable_parameter_count": 0,
        "gpu_training_run": False,
        "source_dev_forward_count": 0,
        "source_dev_target_values_reachable": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_promotion": False,
        "endpoint_flip_authorized": False,
        "primary_evaluable_mask": {
            **PRIMARY_MASK_POLICY,
            "receipt_sha256": PRIMARY_MASK_RECEIPT_SHA256,
        },
        "lineage": {
            "runner_sha256": _file_sha256(Path(__file__).resolve()),
            "safe_anchor_module_sha256": _file_sha256(SAFE_ANCHOR_MODULE_PATH),
            "temporal_manifest_sha256": TEMPORAL_MANIFEST_SHA256,
            "temporal_prediction_sha256": TEMPORAL_PREDICTION_SHA256,
            "frozen_h_manifest_sha256": FROZEN_H_MANIFEST_SHA256,
            "frozen_h_prediction_sha256": FROZEN_H_PREDICTION_SHA256,
            "shortcut_manifest_sha256": SHORTCUT_MANIFEST_SHA256,
            "shortcut_prediction_sha256": SHORTCUT_PREDICTION_SHA256,
        },
    }
    if args.preflight_only:
        # SHA and tensor replay belongs to the full read-only screen; this flag
        # only exposes the frozen plan without loading target tensors.
        print(json.dumps(preflight, sort_keys=True), flush=True)
        return 0

    output = Path(os.path.abspath(args.output_directory))
    if output.name in {"", ".", ".."} or os.path.lexists(output):
        raise FileExistsError(f"output already exists or is invalid: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("output parent must be a regular directory")
    for source in (PROTOCOL_PATH, TEMPORAL_PATH, FROZEN_H_PATH, SHORTCUT_PATH):
        resolved = source.resolve(strict=True)
        if output == resolved or output in resolved.parents or resolved in output.parents:
            raise ValueError("output path overlaps an immutable input")

    result, tensors = _run()
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for safe OOF publication") from exc
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        prediction_path = temporary / "oof_predictions.safetensors"
        save_file(tensors, str(prediction_path))
        patient_ids = json.loads((TEMPORAL_PATH / "manifest.json").read_text())["patient_ids"]
        manifest = {
            **preflight,
            "status": "completed_post_hoc_source_train_safe_oof_screen",
            "patient_ids": patient_ids,
            "result": result,
            "files": {
                "oof_predictions.safetensors": {
                    "sha256": _file_sha256(prediction_path),
                    "size_bytes": prediction_path.stat().st_size,
                }
            },
            "scientific_boundary": {
                "foundation_backbone": "official_LaBraM_frozen",
                "foundation_or_head_training_performed": False,
                "residual_is_not_a_named_clinical_concept": True,
                "safe_means_top1_set_invariant_not_ap_or_hit3_guaranteed": True,
                "post_hoc_after_v3_and_v3_1_results_were_visible": True,
                "fresh_nested_endpoint_gate_completed": False,
                "endpoint_flip_applied": False,
                "source_dev_used": False,
                "source_eval_used": False,
                "private_used": False,
                "formal_promotion": False,
            },
        }
        raw = _canonical_bytes(manifest)
        (temporary / "manifest.json").write_bytes(raw)
        os.rename(temporary, output)
        published = True
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "output_directory": str(output),
                    "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                    "metrics": result["metrics"],
                    "secondary_ranking_gate": result["secondary_ranking_gate"],
                    "endpoint_flip": result["endpoint_flip"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
