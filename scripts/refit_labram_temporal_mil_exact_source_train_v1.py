#!/usr/bin/env python3
"""Refit and seal the exact temporal-MIL anchor on full source-train only.

This is a deployment refit, not another model-selection experiment.  The
candidate, seed, epoch count, loss, and source scope are fixed in code.  Its
only data inputs are the physically source-train-only I/V capability and the
independent source-train target child.
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
    BASE_SEED as NESTED_BASE_SEED,
    TEMPORAL_EPOCHS as NESTED_TEMPORAL_EPOCHS,
    _fit_temporal,
    _tensor_state_sha256,
)
from src.soz.aggregation import aggregate_patient_logits  # noqa: E402
from src.soz.development_reasoner_training_v1_1 import (  # noqa: E402
    FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256,
)
from src.soz.source_train_iv_capability import (  # noqa: E402
    EXPECTED_SOURCE_TRAIN_EVENT_COUNT,
    EXPECTED_SOURCE_TRAIN_PATIENT_COUNT,
    load_and_join_source_train_iv_target_scope,
)
from src.soz.temporal_mil_recovery import (  # noqa: E402
    TEMPORAL_MIL_RECOVERY_SCHEMA,
    TemporalMILEvidenceReasoner,
    TemporalMILPatientBatch,
)


SCHEMA_VERSION = "soz_labram_temporal_mil_exact_full_source_train_refit_v1"
CANDIDATE = "temporal_mil_exact"
NEIGHBOR_WEIGHT = 0.0
EPOCHS = 100
SEED = 20360809  # Original nested runner's BASE_SEED + 99_999.

DEFAULT_SOURCE_TRAIN_IV = (
    ROOT / "outputs/labram_iv_source_train_only_capability_v1_20260811"
)
DEFAULT_SOURCE_TRAIN_IV_MANIFEST_SHA256 = (
    "ccd238b17e1da0aa24f2542a314c770900eeed71cbc31282a4acb76dcf957821"
)
DEFAULT_TRAIN_TARGET_SCOPE = (
    ROOT / "outputs/development_target_scope_v1_1_final_20260810/train"
)

CHECKPOINT_FILENAME = "final_checkpoint.safetensors"
PREDICTION_FILENAME = "source_train_resubstitution_predictions.safetensors"
MANIFEST_FILENAME = "manifest.json"
PREDICTION_KEYS = frozenset(
    {
        "event_logits",
        "patient_logits",
        "temporal_weights",
        "ictal_contribution",
        "evolution_contribution",
        "event_patient_index",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_specs(tensors: Mapping[str, torch.Tensor]) -> dict[str, object]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
        }
        for name, value in sorted(tensors.items())
    }


def _assert_frozen_training_contract() -> None:
    if NESTED_BASE_SEED != 20260810:
        raise RuntimeError("upstream temporal-MIL base seed changed")
    if NESTED_TEMPORAL_EPOCHS != EPOCHS:
        raise RuntimeError("upstream temporal-MIL epoch count changed")
    if SEED != NESTED_BASE_SEED + 99_999:
        raise RuntimeError("exact-anchor final-fit seed changed")
    if NEIGHBOR_WEIGHT != 0.0:
        raise RuntimeError("exact-anchor neighbor weight must remain zero")


def _load_source_train_only(
    *,
    source_train_iv: Path,
    expected_source_train_iv_manifest_sha256: str,
    train_target_scope: Path,
    expected_train_target_receipt_sha256: str,
) -> tuple[
    TemporalMILPatientBatch,
    tuple[int, ...],
    tuple[str, ...],
    dict[str, object],
]:
    """Open only the two closed source-train child bundles."""

    joined = load_and_join_source_train_iv_target_scope(
        source_train_iv,
        train_target_scope,
        expected_capability_manifest_sha256=(
            expected_source_train_iv_manifest_sha256
        ),
        expected_target_receipt_file_sha256=(
            expected_train_target_receipt_sha256
        ),
    )
    patient = joined.batch
    full = TemporalMILPatientBatch(
        evidence=patient.evidence,
        event_patient_index=patient.event_patient_index,
        patient_ids=patient.patient_ids,
        targets=patient.targets,
        target_mask=patient.target_mask,
    )
    patient_folds = tuple(int(value) for value in joined.patient_folds)
    event_ids = tuple(str(value) for value in patient.event_ids)
    if len(full.patient_ids) != EXPECTED_SOURCE_TRAIN_PATIENT_COUNT:
        raise RuntimeError("source-train patient count changed")
    if full.evidence.batch_size != EXPECTED_SOURCE_TRAIN_EVENT_COUNT:
        raise RuntimeError("source-train event count changed")
    if len(event_ids) != EXPECTED_SOURCE_TRAIN_EVENT_COUNT or len(
        set(event_ids)
    ) != len(event_ids):
        raise RuntimeError("source-train event roster is incomplete or duplicated")
    if len(patient_folds) != EXPECTED_SOURCE_TRAIN_PATIENT_COUNT or set(
        patient_folds
    ) != set(range(5)):
        raise RuntimeError("source-train frozen fold carrier changed")
    lineage = {
        "source_train_iv_manifest_sha256": joined.evidence_manifest_sha256,
        "source_train_iv_receipt_sha256": joined.evidence_receipt_sha256,
        "source_train_target_receipt_file_sha256": (
            joined.target_receipt_file_sha256
        ),
    }
    return full, patient_folds, event_ids, lineage


def _fit_exact(
    full: TemporalMILPatientBatch,
    *,
    device: torch.device,
) -> tuple[TemporalMILEvidenceReasoner, dict[str, object]]:
    _assert_frozen_training_contract()
    model, fit = _fit_temporal(
        full,
        neighbor_weight=NEIGHBOR_WEIGHT,
        seed=SEED,
        device=device,
    )
    if fit.get("seed") != SEED or fit.get("epochs") != EPOCHS:
        raise RuntimeError("exact-anchor fit receipt violates the frozen contract")
    if not isinstance(model, TemporalMILEvidenceReasoner):
        raise TypeError("exact-anchor fit returned an unexpected model type")
    return model, fit


def _target_excluding_predictions(
    model: TemporalMILEvidenceReasoner,
    full: TemporalMILPatientBatch,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create a fit-sanity payload without serializing any target tensor."""

    moved_evidence = full.evidence.to(device)
    moved_index = full.event_patient_index.to(device=device)
    model.eval()
    with torch.no_grad():
        output = model(moved_evidence)
        patient_logits = aggregate_patient_logits(
            output.event_logits, moved_index
        ).logits
    tensors = {
        "event_logits": output.event_logits.detach().cpu().contiguous(),
        "patient_logits": patient_logits.detach().cpu().contiguous(),
        "temporal_weights": output.temporal_weights.detach().cpu().contiguous(),
        "ictal_contribution": output.ictal_contribution.detach()
        .cpu()
        .contiguous(),
        "evolution_contribution": output.evolution_contribution.detach()
        .cpu()
        .contiguous(),
        "event_patient_index": full.event_patient_index.detach()
        .cpu()
        .contiguous(),
    }
    if set(tensors) != set(PREDICTION_KEYS):
        raise RuntimeError("target-excluding prediction schema changed")
    if any("target" in name.lower() or "mask" in name.lower() for name in tensors):
        raise RuntimeError("prediction payload contains a target-like key")
    if any(not torch.isfinite(value).all() for name, value in tensors.items() if name != "event_patient_index"):
        raise RuntimeError("exact-anchor prediction contains non-finite values")
    return tensors


def _absolute_without_symlink(path: Path, *, field_name: str) -> Path:
    result = Path(os.path.abspath(path))
    for component in (result, *result.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field_name} cannot traverse symlinks")
    return result


def _guard_output(output_directory: Path, inputs: Sequence[Path]) -> Path:
    output = _absolute_without_symlink(
        output_directory, field_name="exact-anchor output"
    )
    if output.name in {"", ".", ".."} or os.path.lexists(output):
        raise FileExistsError(f"output already exists or is invalid: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    for value in inputs:
        source = _absolute_without_symlink(value, field_name="exact-anchor input")
        source = source.resolve(strict=True)
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("output path overlaps an immutable input")
    return output


def _preflight(
    *,
    full: TemporalMILPatientBatch,
    patient_folds: tuple[int, ...],
    event_ids: tuple[str, ...],
    lineage: Mapping[str, object],
    device: torch.device,
    source_train_iv: Path,
    train_target_scope: Path,
) -> dict[str, object]:
    fold_counts = {
        str(fold): patient_folds.count(fold) for fold in sorted(set(patient_folds))
    }
    event_roster_sha256 = hashlib.sha256(
        _canonical_bytes(list(event_ids))
    ).hexdigest()
    patient_roster_sha256 = hashlib.sha256(
        _canonical_bytes(list(full.patient_ids))
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_full_source_train_exact_refit",
        "candidate": CANDIDATE,
        "device": str(device),
        "patient_count": len(full.patient_ids),
        "event_count": full.evidence.batch_size,
        "patient_fold_counts": fold_counts,
        "patient_roster_sha256": patient_roster_sha256,
        "event_roster_sha256": event_roster_sha256,
        "lineage": dict(lineage),
        "data_access": {
            "loaded_model_splits": ["source_train"],
            "source_train_iv_path": str(source_train_iv),
            "source_train_target_path": str(train_target_scope),
            "source_train_target_values_loaded_for_fit": True,
            "source_dev_signal_or_target_open_count": 0,
            "source_eval_signal_or_target_open_count": 0,
            "private_signal_or_target_open_count": 0,
            "other_split_input_path_count": 0,
        },
        "frozen_fit": {
            "seed": SEED,
            "epochs": EPOCHS,
            "neighbor_weight": NEIGHBOR_WEIGHT,
            "checkpoint_selection": "fixed_final_epoch_no_selection",
            "architecture_selection_in_this_run": False,
            "hyperparameter_selection_in_this_run": False,
        },
        "scientific_boundary": {
            "foundation_backbone": "frozen_LaBraM_not_replaced",
            "foundation_trainable_parameters": 0,
            "fit_scope": "full_source_train_only",
            "resubstitution_predictions_are_evaluation": False,
            "source_eval_used": False,
            "private_used": False,
            "confirmatory_result": False,
        },
    }


def _publish(
    output: Path,
    *,
    preflight: Mapping[str, object],
    model: TemporalMILEvidenceReasoner,
    fit: Mapping[str, object],
    predictions: Mapping[str, torch.Tensor],
    patient_ids: Sequence[str],
    patient_folds: Sequence[int],
) -> str:
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for exact-anchor publication") from exc

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        }
        checkpoint_path = temporary / CHECKPOINT_FILENAME
        prediction_path = temporary / PREDICTION_FILENAME
        save_file(state, str(checkpoint_path))
        save_file(dict(predictions), str(prediction_path))

        reloaded_state = load_file(str(checkpoint_path), device="cpu")
        if set(reloaded_state) != set(state) or _tensor_state_sha256(
            reloaded_state
        ) != _tensor_state_sha256(state):
            raise RuntimeError("serialized exact-anchor checkpoint did not round-trip")
        reloaded_model = TemporalMILEvidenceReasoner(torch.zeros(19))
        reloaded_model.load_state_dict(reloaded_state, strict=True)
        reloaded_predictions = load_file(str(prediction_path), device="cpu")
        if set(reloaded_predictions) != set(PREDICTION_KEYS):
            raise RuntimeError("serialized prediction payload changed schema")

        runner_path = Path(__file__).resolve(strict=True)
        reused_runner_path = (
            ROOT / "scripts/run_labram_temporal_mil_nested_oof_v1.py"
        ).resolve(strict=True)
        model_path = (ROOT / "src/soz/temporal_mil_recovery.py").resolve(
            strict=True
        )
        manifest = {
            **dict(preflight),
            "status": "completed_full_source_train_exact_refit",
            "fit": dict(fit),
            "model": {
                "class": "TemporalMILEvidenceReasoner",
                "trainable_parameter_count": model.n_trainable_parameters,
                "checkpoint_includes_train_target_derived_channel_prior": True,
                "checkpoint_state_sha256": _tensor_state_sha256(state),
                "checkpoint_tensor_specs": _tensor_specs(state),
            },
            "prediction_payload": {
                "role": "source_train_resubstitution_fit_sanity_only",
                "contains_target_values": False,
                "contains_target_mask": False,
                "eligible_as_evaluation": False,
                "tensor_specs": _tensor_specs(predictions),
            },
            "patient_ids": list(patient_ids),
            "patient_folds": [int(value) for value in patient_folds],
            "code_lineage": {
                "runner_path": str(runner_path.relative_to(ROOT)),
                "runner_sha256": _file_sha256(runner_path),
                "reused_fit_runner_path": str(reused_runner_path.relative_to(ROOT)),
                "reused_fit_runner_sha256": _file_sha256(reused_runner_path),
                "model_module_path": str(model_path.relative_to(ROOT)),
                "model_module_sha256": _file_sha256(model_path),
                "temporal_mil_schema": TEMPORAL_MIL_RECOVERY_SCHEMA,
            },
            "files": {
                CHECKPOINT_FILENAME: {
                    "sha256": _file_sha256(checkpoint_path),
                    "size_bytes": checkpoint_path.stat().st_size,
                },
                PREDICTION_FILENAME: {
                    "sha256": _file_sha256(prediction_path),
                    "size_bytes": prediction_path.stat().st_size,
                },
            },
        }
        raw = _canonical_bytes(manifest)
        (temporary / MANIFEST_FILENAME).write_bytes(raw)
        os.rename(temporary, output)
        published = True
        return hashlib.sha256(raw).hexdigest()
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--source-train-iv", type=Path, default=DEFAULT_SOURCE_TRAIN_IV
    )
    parser.add_argument(
        "--expected-source-train-iv-manifest-sha256",
        default=DEFAULT_SOURCE_TRAIN_IV_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--train-target-scope", type=Path, default=DEFAULT_TRAIN_TARGET_SCOPE
    )
    parser.add_argument(
        "--expected-train-target-receipt-sha256",
        default=FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256,
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _assert_frozen_training_contract()
    if not args.preflight_only and args.output_directory is None:
        raise ValueError("full refit requires --output-directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    source_train_iv = _absolute_without_symlink(
        args.source_train_iv, field_name="source-train I/V"
    )
    train_target_scope = _absolute_without_symlink(
        args.train_target_scope, field_name="source-train target"
    )
    full, patient_folds, event_ids, lineage = _load_source_train_only(
        source_train_iv=source_train_iv,
        expected_source_train_iv_manifest_sha256=(
            args.expected_source_train_iv_manifest_sha256
        ),
        train_target_scope=train_target_scope,
        expected_train_target_receipt_sha256=(
            args.expected_train_target_receipt_sha256
        ),
    )
    preflight = _preflight(
        full=full,
        patient_folds=patient_folds,
        event_ids=event_ids,
        lineage=lineage,
        device=device,
        source_train_iv=source_train_iv,
        train_target_scope=train_target_scope,
    )
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True), flush=True)
        return 0

    output = _guard_output(
        args.output_directory, (source_train_iv, train_target_scope)
    )
    model, fit = _fit_exact(full, device=device)
    predictions = _target_excluding_predictions(model, full, device=device)
    model = model.cpu()
    model.requires_grad_(False)
    manifest_sha256 = _publish(
        output,
        preflight=preflight,
        model=model,
        fit=fit,
        predictions=predictions,
        patient_ids=full.patient_ids,
        patient_folds=patient_folds,
    )
    print(
        json.dumps(
            {
                "status": "completed_full_source_train_exact_refit",
                "candidate": CANDIDATE,
                "output_directory": str(output),
                "manifest_sha256": manifest_sha256,
                "checkpoint": str(output / CHECKPOINT_FILENAME),
                "source_dev_used": False,
                "source_eval_used": False,
                "private_used": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
