#!/usr/bin/env python3
"""Train one v13 independent-second control from fit-only artifacts.

The command accepts no legacy monolithic target snapshot, native-evaluation
input, I-gate signal/outcome source, DeepSOZ input, or private input.  Its
selective corpus loader opens only the exact k31 fit token bundles, and its
target loader accepts only a separately sealed physical fit-only artifact.
Training runs for the frozen 20 epochs and saves the fixed final epoch without
any evaluation, early stopping, selection, calibration, or threshold search.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Sequence


_REQUIRED_CUBLAS_WORKSPACE = ":4096:8"
_observed_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _observed_workspace is None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _REQUIRED_CUBLAS_WORKSPACE
elif _observed_workspace != _REQUIRED_CUBLAS_WORKSPACE:
    raise RuntimeError("v13 control requires CUBLAS_WORKSPACE_CONFIG=':4096:8'")

import torch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._v13_minimal_import import (  # noqa: E402
    assert_forbidden_v13_modules_absent,
    install_v13_minimal_soz_package,
)

_V13_MINIMAL_IMPORT_ACTIVE = install_v13_minimal_soz_package(ROOT)

from src.soz.cached_concept_training import (  # noqa: E402
    IctalTokenBagDataset,
    train_cached_ictal_epoch,
)
from src.soz.ictal_fit_primitives_v13 import (  # noqa: E402
    IctalTrainingConfig,
    LABRAM_K31_EXECUTION_RECEIPT_SCHEMA,
    ictal_determinism_runtime,
    ictal_head_state_sha256,
    validate_ictal_cuda_environment,
)
from src.soz.ictal_fit_only_consumer_v13 import (  # noqa: E402
    load_fit_only_target_artifact_v13,
)
from src.soz.ictal_fit_token_view_consumer_v13 import (  # noqa: E402
    load_fit_token_view_v13,
)
from src.soz.ictal_matched_control_v13 import (  # noqa: E402
    build_fit_only_token_bag_dataset_v13,
    save_matched_independent_control_v13,
    validate_matched_control_lineage,
)
from src.soz.models.concept_heads import IctalInvolvementHead  # noqa: E402

if _V13_MINIMAL_IMPORT_ACTIVE:
    assert_forbidden_v13_modules_absent()


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(value: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        required=True,
        choices=(*[f"fold{index}" for index in range(5)], "final"),
    )
    parser.add_argument("--fit-token-view-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-fit-token-view-manifest-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-fit-token-view-receipt-sha256", type=_sha256, required=True
    )
    parser.add_argument("--fit-only-target-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-fit-only-target-manifest-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-fit-only-target-receipt-sha256", type=_sha256, required=True
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _execution_receipt(
    config: IctalTrainingConfig, device: torch.device
) -> dict[str, object]:
    probe = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW(
        [probe], lr=float(config.learning_rate), weight_decay=float(config.weight_decay)
    )
    group = optimizer.param_groups[0]
    capability = None
    device_name = "cpu"
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(index)
        capability = list(torch.cuda.get_device_capability(index))
    config_payload = asdict(config)
    config_sha = hashlib.sha256(
        json.dumps(
            config_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": LABRAM_K31_EXECUTION_RECEIPT_SCHEMA,
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "device_type": device.type,
        "device_name": device_name,
        "compute_capability": capability,
        "optimizer_class": "torch.optim.AdamW",
        "optimizer_effective_hyperparameters": {
            "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
            "betas": [float(value) for value in group["betas"]],
            "eps": float(group["eps"]),
            "amsgrad": bool(group["amsgrad"]),
            "maximize": bool(group["maximize"]),
            "foreach": group.get("foreach"),
            "capturable": bool(group["capturable"]),
            "differentiable": bool(group["differentiable"]),
            "fused": group.get("fused"),
        },
        "training_config_sha256": config_sha,
    }


def _memoize_fit_only(dataset: IctalTokenBagDataset) -> IctalTokenBagDataset:
    """Load the already selective fit roster once for the fixed 20 epochs."""

    bags = {bag.patient_id: bag for bag in dataset.iter_epoch()}
    if tuple(sorted(bags)) != dataset.patient_ids:
        raise RuntimeError("Memoized fit-only dataset changed its patient roster")
    return IctalTokenBagDataset(
        dataset.patient_ids,
        bags.__getitem__,
        training_manifest_sha256=dataset.training_manifest_sha256,
        token_source_manifest_sha256=dataset.token_source_manifest_sha256,
        foundation_feature_receipt_sha256=dataset.foundation_feature_receipt_sha256,
        formal_token_corpus_verified=dataset.formal_token_corpus_verified,
        formal_token_corpus_index_sha256=dataset.formal_token_corpus_index_sha256,
        formal_token_corpus_training_bundle_manifest_sha256=(
            dataset.formal_token_corpus_training_bundle_manifest_sha256
        ),
        formal_token_corpus_event_roster_sha256=(
            dataset.formal_token_corpus_event_roster_sha256
        ),
        formal_token_corpus_patient_roster_sha256=(
            dataset.formal_token_corpus_patient_roster_sha256
        ),
        formal_token_corpus_tensor_roster_sha256=(
            dataset.formal_token_corpus_tensor_roster_sha256
        ),
        training_authorized=dataset.training_authorized,
    )


def _train_fit_only(
    dataset: IctalTokenBagDataset,
    *,
    config: IctalTrainingConfig,
    device: torch.device,
    selection: str,
) -> tuple[IctalInvolvementHead, dict[str, object]]:
    cuda_devices: list[int] = []
    if device.type == "cuda":
        validate_ictal_cuda_environment()
        cuda_devices = [
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        ]
    with ictal_determinism_runtime(config, execution_device_type=device.type):
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(config.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(config.seed)
            head = IctalInvolvementHead().to(device)
            initial_state = ictal_head_state_sha256(head)
            optimizer = torch.optim.AdamW(
                head.parameters(),
                lr=float(config.learning_rate),
                weight_decay=float(config.weight_decay),
            )
            epoch_rows = []
            for epoch in range(config.fixed_epochs):
                order = list(dataset.patient_ids)
                random.Random(config.seed + epoch).shuffle(order)
                output = train_cached_ictal_epoch(
                    head,
                    dataset,
                    optimizer,
                    patient_order=tuple(order),
                    max_grad_norm=config.max_grad_norm,
                    event_microbatch_size=config.event_microbatch_size,
                )
                epoch_rows.append(asdict(output))
                print(
                    json.dumps(
                        {
                            "stage": "v13_fit_only_control_training",
                            "selection": selection,
                            "epoch": epoch + 1,
                            "epochs": config.fixed_epochs,
                            "mean_patient_loss": output.mean_patient_loss,
                            "evaluation_performed": False,
                            "gate_opened": False,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            final_state = ictal_head_state_sha256(head)
    return head, {
        "initial_state_sha256": initial_state,
        "final_state_sha256": final_state,
        "epoch_rows": epoch_rows,
        "evaluation_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    if not _V13_MINIMAL_IMPORT_ACTIVE:
        raise RuntimeError("v13 formal trainer requires a clean minimal import process")
    assert_forbidden_v13_modules_absent()
    args = build_parser().parse_args(argv)
    output = Path(os.path.abspath(args.output_directory))
    if output.name in {"", ".", ".."} or not output.parent.is_dir():
        raise ValueError("v13 output requires a concrete path with an existing parent")
    if os.path.lexists(output):
        raise FileExistsError(f"v13 output already exists: {output}")
    device = torch.device(args.device)
    if not args.preflight_only and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")

    fit_targets = load_fit_only_target_artifact_v13(
        args.fit_only_target_bundle,
        expected_manifest_sha256=args.expected_fit_only_target_manifest_sha256,
        expected_receipt_sha256=args.expected_fit_only_target_receipt_sha256,
    )
    if fit_targets.manifest["selection"] != args.selection:
        raise ValueError("Fit-only target selection differs from CLI")
    fit_patients = tuple(fit_targets.manifest["fit_patient_ids"])
    gate_patients = tuple(
        fit_targets.manifest["i_gate_patient_ids_excluded_unopened"]
    )
    if set(fit_patients) & set(gate_patients) or len(gate_patients) != 12:
        raise ValueError("Brokered fit roster intersects the unopened I-gate")
    fit_token_view = load_fit_token_view_v13(
        args.fit_token_view_bundle,
        expected_manifest_sha256=args.expected_fit_token_view_manifest_sha256,
        expected_receipt_sha256=args.expected_fit_token_view_receipt_sha256,
    )
    if (
        fit_token_view.manifest["selection"] != args.selection
        or fit_token_view.manifest["matched_k31_manifest_sha256"]
        != fit_targets.manifest["matched_k31_manifest_sha256"]
        or tuple(fit_token_view.manifest["fit_patient_ids"]) != fit_patients
        or tuple(fit_token_view.manifest["i_gate_patient_ids_excluded_unopened"])
        != gate_patients
    ):
        raise ValueError("Physical fit-token view differs from fit target authority")
    fit_corpus = fit_token_view.corpus
    lineage = validate_matched_control_lineage(
        selection=args.selection,
        fit_only_targets=fit_targets,
        fit_token_view=fit_token_view,
    )
    dataset = build_fit_only_token_bag_dataset_v13(
        fit_corpus,
        fit_targets,
    )
    if dataset.patient_ids != fit_patients:
        raise ValueError("Fit-only dataset changed the exact k31 fit roster")
    preflight = {
        "schema_version": "soz_labram_ictal_matched_control_preflight_v13",
        "selection": lineage.selection,
        "oof_fold": lineage.oof_fold,
        "preflight_passed": True,
        "training_started": False,
        "development_confirmation_control": True,
        "matched_k31_manifest_sha256": lineage.matched_k31_manifest_sha256,
        "fit_patient_count": len(fit_patients),
        "fit_event_count": lineage.fit_event_count,
        "i_gate_patient_count": len(gate_patients),
        "fit_roster_exactly_matches_k31": True,
        "fit_gate_intersection_count": 0,
        "fit_only_target_artifact_loaded": True,
        "fit_target_values_loaded": True,
        "source_full_target_arrays_loaded": False,
        "source_full_target_arrays_mapped": False,
        "fit_token_bundles_opened": True,
        "physical_fit_token_view_loaded": True,
        "non_fit_token_bundles_opened": False,
        "source_full_corpus_root_reachable": False,
        "full_corpus_loader_imported": False,
        "legacy_k31_full_manifest_loaded": False,
        "legacy_k31_native_roster_or_metrics_loaded": False,
        "full_training_manifest_loaded": False,
        "gate_row_level_target_derived_hashes_counts_loaded": False,
        "source_full_target_file_or_snapshot_hashes_loaded": True,
        "native_evaluation_inputs_loaded": False,
        "native_evaluation_performed": False,
        "native_metrics_computed": False,
        "gate_opened": False,
        "i_gate_signal_or_tokens_opened": False,
        "i_gate_target_values_materialized": False,
        "i_gate_target_values_evaluated": False,
        "i_gate_outcomes_opened": False,
        "deepsoz_identity_outcome_prediction_reachable": False,
        "deepsoz_target_source_loaded": False,
        "deepsoz_soz_labels_used": False,
        "private_signal_identity_outcome_reachable": False,
        "private_labels_used": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True), flush=True)
        return 0

    fit_dataset = _memoize_fit_only(dataset)
    config = IctalTrainingConfig()
    if asdict(config) != dict(fit_targets.manifest["matched_training_config"]):
        raise ValueError("v13 optimization policy differs from brokered k31 authority")
    execution = _execution_receipt(config, device)
    head, training_run = _train_fit_only(
        fit_dataset,
        config=config,
        device=device,
        selection=args.selection,
    )
    saved = save_matched_independent_control_v13(
        output,
        lineage=lineage,
        head=head,
        training_config=asdict(config),
        execution_receipt=execution,
        training_run=training_run,
    )
    print(
        json.dumps(
            {
                **preflight,
                "training_started": True,
                "path": str(saved.path),
                "manifest_sha256": saved.manifest_sha256,
                "fixed_final_epoch_saved": True,
                "native_evaluation_performed": False,
                "native_metrics_computed": False,
                "gate_opened": False,
                "i_gate_outcomes_opened": False,
                "formal_promotion": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
