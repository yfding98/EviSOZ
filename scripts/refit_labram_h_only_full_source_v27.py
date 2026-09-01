#!/usr/bin/env python3
"""Fit the frozen-LaBraM H-only deployment reasoner on all legal public data.

The official pretrained LaBraM-Base is never updated.  The feature transform
and Jeffreys channel prior are fitted on the 102 fully observed DeepSOZ/TUSZ
development patients.  Nine disjoint masked-variable patients can affect only
the positive-set-mass reasoner loss.  This command has no private-data input.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_labram_fine_temporal_nested_oof_v11 import (  # noqa: E402
    _file_sha,
    _fit_reasoner,
    _state_sha,
    _transform_state,
)
import scripts.run_labram_masked_variable_auxiliary_oof_v17 as v17  # noqa: E402
from src.soz.geometry import STANDARD_19  # noqa: E402
from src.soz.v11_reasoner import (  # noqa: E402
    V11_CANDIDATE_MASK,
    apply_fixed_candidate_mask,
    fit_fold_transform,
    jeffreys_reference_prior_logits,
    positive_set_mass_loss,
)


SCHEMA = "soz_labram_h_only_full_source_refit_v27"
L2 = 0.20
DEFAULT_OUTPUT = ROOT / "outputs/labram_h_only_full_source_refit_v27_20260815"
EXPECTED_AUX_PREFIX_MANIFEST_SHA256 = (
    "32ae76e5151f997cad70cee71070646dad4c6febe70ced50b0e20574fc5e4ed9"
)
EXPECTED_AUX_PREFIX_TENSOR_SHA256 = (
    "23e9726f5456da2a79c4c17a1b697428ccbd5d5eb1d46d866337b20bc901ffc6"
)
EXPECTED_AUX_FINE_MANIFEST_SHA256 = (
    "79f97711738167fec8c596d74b7ac17332130acf24a5f2428ff4fb9cd150dc85"
)
EXPECTED_AUX_FINE_TENSOR_SHA256 = (
    "8bcdf782117e694331f7be0c24f97682f8ff077381b85b493bd95588ef7b1b7c"
)


def _public_metrics(
    logits: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor
) -> dict[str, float]:
    masked = apply_fixed_candidate_mask(logits)
    top1 = masked.argmax(dim=1)
    strict = targets.gather(1, top1[:, None]).squeeze(1) == 1
    positive_mass = torch.softmax(masked, dim=1).masked_fill(
        ~((targets == 1) & target_mask), 0.0
    ).sum(dim=1)
    return {
        "resubstitution_strict_top1": float(strict.float().mean()),
        "mean_positive_set_probability_mass": float(positive_mass.mean()),
    }


def run(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    stable = v17._load_stable_development(args)
    auxiliary = v17._load_auxiliary_targets(args, stable)
    prefix = v17._load_auxiliary_cache(
        args.aux_prefix_directory,
        expected_manifest_sha256=args.expected_aux_prefix_manifest_sha256,
        expected_tensor_sha256=args.expected_aux_prefix_tensor_sha256,
        schema=v17.PREFIX_SCHEMA,
        expected_keys=v17.PREFIX_TENSOR_KEYS,
        primary_key="prefix_tokens",
        auxiliary=auxiliary,
        label="auxiliary LaBraM prefix",
    )
    fine = v17._load_auxiliary_cache(
        args.aux_fine_directory,
        expected_manifest_sha256=args.expected_aux_fine_manifest_sha256,
        expected_tensor_sha256=args.expected_aux_fine_tensor_sha256,
        schema=v17.FINE_SCHEMA,
        expected_keys=v17.FINE_TENSOR_KEYS,
        primary_key="features",
        auxiliary=auxiliary,
        label="auxiliary fine evidence",
    )
    auxiliary_h, auxiliary_fine = v17._pool_auxiliary_features(
        auxiliary, prefix, fine
    )

    stable_count = len(stable.patient_ids)
    auxiliary_count = len(auxiliary.patient_ids)
    if stable_count != 102 or auxiliary_count != 9:
        raise RuntimeError("v27 source patient roster changed")
    if set(stable.patient_ids) & set(auxiliary.patient_ids):
        raise RuntimeError("stable and auxiliary patients overlap")

    stable_indices = tuple(range(stable_count))
    transform = fit_fold_transform(
        stable.h_patient, stable.fine_patient, stable_indices
    )
    prior = jeffreys_reference_prior_logits(
        stable.targets, stable.target_mask
    )
    combined_h = torch.cat((stable.h_patient, auxiliary_h), dim=0).contiguous()
    combined_fine = torch.cat(
        (stable.fine_patient, auxiliary_fine), dim=0
    ).contiguous()
    combined_targets = torch.cat(
        (stable.targets, auxiliary.targets), dim=0
    ).contiguous()
    combined_mask = torch.cat(
        (stable.target_mask, auxiliary.target_mask), dim=0
    ).contiguous()
    combined_indices = tuple(range(stable_count + auxiliary_count))
    transformed = transform.apply(combined_h, combined_fine)
    fit = _fit_reasoner(
        transformed,
        combined_targets,
        combined_mask,
        combined_indices,
        use_h=True,
        use_fine=False,
        l2=L2,
        allow_candidate_subset=True,
        fixed_prior_logits=prior,
    )
    if fit.diagnostics.get("trainable_parameter_count") != 16:
        raise RuntimeError("v27 reasoner capacity changed")
    if fit.diagnostics.get("prior_source") != "caller_frozen":
        raise RuntimeError("v27 prior was not frozen from stable patients")

    stable_logits = fit.logits[:stable_count].detach().cpu().contiguous()
    auxiliary_logits = fit.logits[stable_count:].detach().cpu().contiguous()
    objective = positive_set_mass_loss(
        fit.logits,
        combined_targets,
        combined_mask,
        allow_candidate_subset=True,
    )
    checkpoint = {
        **_transform_state(transform),
        **{f"reasoner.{name}": value for name, value in fit.state.items()},
        "config.l2": torch.tensor(L2, dtype=torch.float32),
        "config.candidate_mask": V11_CANDIDATE_MASK.clone(),
        "config.stable_patient_count": torch.tensor(stable_count, dtype=torch.long),
        "config.auxiliary_patient_count": torch.tensor(auxiliary_count, dtype=torch.long),
        "config.foundation_trainable_parameters": torch.tensor(0, dtype=torch.long),
        "config.reasoner_trainable_parameters": torch.tensor(16, dtype=torch.long),
    }
    outputs = {
        "stable_logits": stable_logits,
        "stable_probability": torch.softmax(
            apply_fixed_candidate_mask(stable_logits), dim=1
        ),
        "auxiliary_logits": auxiliary_logits,
        "stable_targets": stable.targets,
        "stable_target_mask": stable.target_mask,
        "auxiliary_targets": auxiliary.targets,
        "auxiliary_target_mask": auxiliary.target_mask,
    }
    manifest: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_public_only_full_refit_frozen_for_deployment",
        "model_role": "scalp_electrode_soz_reference_candidate_ranker",
        "target_definition": {
            "public_primary": "DeepSOZ clinician SOZ-channel positive set mapped to local TUSZ",
            "auxiliary": "masked-variable DeepSOZ channel labels with unknown entries excluded from loss",
            "not_targets": [
                "earliest scalp-visible channel",
                "TUSZ ictal involvement channel",
                "cortical SOZ",
                "epileptogenic zone",
                "surgical treatment target",
            ],
        },
        "architecture": {
            "raw_deployment_carrier": "standard19 monopolar CAR, 200 Hz, event window [-12,+48) s",
            "foundation": "official pretrained LaBraM-Base, fully frozen",
            "cached_prefix_shape_per_event": [15, 77, 200],
            "phase_contrast_shape_per_event": [19, 600],
            "patient_pooled_h_shape": [19, 600],
            "stable_only_robust_scaling": [600],
            "stable_only_pca_output_shape": [19, 16],
            "reasoner": "shared linear H-evidence scorer plus stable-only Jeffreys channel prior",
            "reasoner_trainable_parameters": 16,
            "output": "C18 candidate-masked softmax over physical scalp electrodes; PZ not evaluable",
            "information_flow": [
                "EEG event",
                "frozen LaBraM block-9 prefix",
                "three target-independent phase contrasts",
                "reliability-weighted complete patient bag",
                "stable-only transform",
                "shared H evidence plus anatomical prior",
                "C18 SOZ-reference candidate ranking",
            ],
        },
        "training": {
            "stable_patient_count": stable_count,
            "stable_event_count": int(stable.event_counts.sum()),
            "auxiliary_patient_count": auxiliary_count,
            "auxiliary_event_count": int(auxiliary.event_counts.sum()),
            "combined_patient_count": len(combined_indices),
            "loss": "equally patient-weighted positive-set probability-mass negative log likelihood plus L2",
            "l2": L2,
            "l2_selection_source": "precommitted majority value from v16 H-only five-fold source development",
            "final_positive_set_mass_loss_unregularized": float(objective),
            "fit_diagnostics": dict(fit.diagnostics),
            "stable_resubstitution_diagnostic_not_a_performance_claim": _public_metrics(
                stable_logits, stable.targets, stable.target_mask
            ),
        },
        "patient_ids": {
            "stable": list(stable.patient_ids),
            "auxiliary": list(auxiliary.patient_ids),
        },
        "channels": list(STANDARD_19),
        "lineage": {
            **dict(stable.lineage),
            "auxiliary_join_artifact_sha256": auxiliary.join.artifact_sha256,
            "auxiliary_join_receipt_sha256": auxiliary.join.receipt_sha256,
            "auxiliary_prefix_manifest_sha256": prefix.manifest_sha256,
            "auxiliary_prefix_tensor_file_sha256": prefix.tensor_file_sha256,
            "auxiliary_fine_manifest_sha256": fine.manifest_sha256,
            "auxiliary_fine_tensor_file_sha256": fine.tensor_file_sha256,
        },
        "access_receipt": {
            "private_path_argument_exposed": False,
            "private_raw_eeg_loaded": False,
            "private_cached_evidence_loaded": False,
            "private_target_values_loaded": False,
            "private_used_for_transform_prior_loss_or_model_selection": False,
            "foundation_training_performed": False,
            "foundation_trainable_parameters": 0,
            "reasoner_training_performed": True,
            "reasoner_trainable_parameters": 16,
            "stable_only_transform_fit": True,
            "stable_only_prior_fit": True,
            "auxiliary_partial_masks_used_only_in_reasoner_loss": True,
        },
        "claim_boundary": {
            "full_refit_training_metrics_are_generalization_metrics": False,
            "public_generalization_requires_existing_patient_oof_endpoint": True,
            "private_validation_must_be_run_only_after_this_artifact_is_frozen": True,
            "neighborhood4_is_strict_channel_accuracy": False,
        },
    }
    return manifest, {"checkpoint": checkpoint, "outputs": outputs}


def publish(
    output_directory: Path,
    manifest: Mapping[str, object],
    payloads: Mapping[str, Mapping[str, torch.Tensor]],
) -> Path:
    target = output_directory.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        checkpoint_path = staging / "checkpoint.safetensors"
        outputs_path = staging / "training_outputs.safetensors"
        save_file(dict(payloads["checkpoint"]), str(checkpoint_path))
        save_file(dict(payloads["outputs"]), str(outputs_path))
        completed = dict(manifest)
        completed["files"] = {
            "checkpoint.safetensors": {
                "sha256": _file_sha(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
                "state_sha256": _state_sha(payloads["checkpoint"]),
            },
            "training_outputs.safetensors": {
                "sha256": _file_sha(outputs_path),
                "size_bytes": outputs_path.stat().st_size,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(completed, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--union-directory", type=Path, default=v17.DEFAULT_UNION)
    parser.add_argument("--stable-fine-directory", type=Path, default=v17.DEFAULT_STABLE_FINE)
    parser.add_argument("--stable-prefix-directory", type=Path, default=v17.DEFAULT_STABLE_PREFIX)
    parser.add_argument("--legacy-fine-directory", type=Path, default=v17.DEFAULT_LEGACY_FINE)
    parser.add_argument("--legacy-prefix-directory", type=Path, default=v17.DEFAULT_LEGACY_PREFIX)
    parser.add_argument("--target-directory", type=Path, default=v17.DEFAULT_TARGET)
    parser.add_argument("--source-csv", type=Path, default=v17.DEFAULT_SOURCE)
    parser.add_argument("--split-csv", type=Path, default=v17.DEFAULT_SPLIT)
    parser.add_argument("--aux-join-directory", type=Path, default=v17.DEFAULT_AUX_JOIN)
    parser.add_argument("--aux-prefix-directory", type=Path, default=v17.DEFAULT_AUX_PREFIX)
    parser.add_argument("--aux-fine-directory", type=Path, default=v17.DEFAULT_AUX_FINE)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-protocol-sha256", default=v17.EXPECTED_PROTOCOL_SHA256)
    parser.add_argument(
        "--expected-union-manifest-sha256",
        default=v17.EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-stable-fine-manifest-sha256",
        default=v17.identity_v16.EXPECTED_FINE_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-stable-fine-tensor-sha256",
        default=v17.identity_v16.EXPECTED_FINE_TENSOR_SHA256,
    )
    parser.add_argument(
        "--expected-stable-prefix-manifest-sha256",
        default=v17.identity_v16.EXPECTED_PREFIX_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-stable-prefix-tensor-sha256",
        default=v17.identity_v16.EXPECTED_PREFIX_TENSOR_SHA256,
    )
    parser.add_argument(
        "--expected-aux-join-artifact-sha256",
        default=v17.EXPECTED_AUX_JOIN_ARTIFACT_SHA256,
    )
    parser.add_argument(
        "--expected-aux-admission-artifact-sha256",
        default=v17.EXPECTED_AUX_ADMISSION_ARTIFACT_SHA256,
    )
    parser.add_argument(
        "--expected-aux-prefix-manifest-sha256",
        default=EXPECTED_AUX_PREFIX_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-aux-prefix-tensor-sha256",
        default=EXPECTED_AUX_PREFIX_TENSOR_SHA256,
    )
    parser.add_argument(
        "--expected-aux-fine-manifest-sha256",
        default=EXPECTED_AUX_FINE_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-aux-fine-tensor-sha256",
        default=EXPECTED_AUX_FINE_TENSOR_SHA256,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    args = build_parser().parse_args(argv)
    manifest, payloads = run(args)
    output = publish(args.output_directory, manifest, payloads)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(output),
                "stable_patients": manifest["training"]["stable_patient_count"],
                "auxiliary_patients": manifest["training"]["auxiliary_patient_count"],
                "reasoner_trainable_parameters": 16,
                "private_used": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
