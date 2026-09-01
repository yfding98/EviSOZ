#!/usr/bin/env python3
"""Audit the common17 trainable-parameter receipt correction.

The original replay read ``n_trainable_parameters`` after freezing the fitted
model and therefore recorded zero.  This audit proves that the corrected
replays only change that metadata: weights, predictions, losses, and metrics
remain identical, while the saved fitted states differ from seeded
initialization for every trainable tensor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from safetensors import safe_open
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sota_soz.train_common17_oracle_event_oof_v1 import (  # noqa: E402
    Common17EventSetReasoner,
)


SCHEMA = "clinical_eeg_common17_trainable_parameter_metadata_correction_v1"
ARMS = {
    "verified_strict": {
        "old": ROOT / "outputs/clinical_eeg_common17_oracle_event_oof_r3r2_20260824",
        "new": ROOT / "outputs/clinical_eeg_common17_oracle_event_oof_r3r3_20260824",
        "state_prefix": "strict_car17_labram",
    },
    "literal_duplicate_pz_or_sensitivity": {
        "old": ROOT
        / "outputs/clinical_eeg_common17_car17_literal_midline_oof_sensitivity_v1_20260824",
        "new": ROOT
        / "outputs/clinical_eeg_common17_car17_literal_midline_oof_sensitivity_v1r2_20260824",
        "state_prefix": "strict_car17_labram_literal_raw_FZ_PZ_OR_to_CZ",
    },
}
OUTPUT = (
    ROOT
    / "outputs/clinical_eeg_common17_trainable_parameter_metadata_correction_v1_20260824"
    / "receipt.json"
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
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


def _tensor_inventory(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path.resolve(strict=True)), framework="pt", device="cpu") as source:
        return {key: source.get_tensor(key) for key in source.keys()}


def _fold_projection(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "fold": int(row["fold"]),
            "seed": int(row["seed"]),
            "train_patients": int(row["train_patients"]),
            "held_patients": int(row["held_patients"]),
            "patient_overlap": int(row["patient_overlap"]),
            "fit_seed": int(row["fit"]["seed"]),
            "first_epoch_loss": float(row["fit"]["first_epoch_loss"]),
            "final_epoch_loss": float(row["fit"]["final_epoch_loss"]),
            "held_metrics": row["held_metrics"],
        }
        for row in manifest["folds"]
    ]


def _initialization_change_audit(
    tensors: dict[str, torch.Tensor], manifest: dict[str, Any], prefix: str
) -> list[dict[str, Any]]:
    result = []
    for row in manifest["folds"]:
        fold = int(row["fold"])
        seed = int(row["fit"]["seed"])
        state_prefix = f"{prefix}.fold{fold}."
        saved = {
            key.removeprefix(state_prefix): value
            for key, value in tensors.items()
            if key.startswith(state_prefix)
        }
        prior = saved["prior_logits"]
        projection = saved["projection.weight"]
        phase_logits = saved["phase_logits"]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            initialized = Common17EventSetReasoner(
                input_dim=int(projection.shape[1]),
                phase_count=int(phase_logits.numel()),
                latent_dim=int(projection.shape[0]),
                prior_logits=prior,
            )
        parameter_names = tuple(name for name, _ in initialized.named_parameters())
        if initialized.n_trainable_parameters != 6_967 or len(parameter_names) != 11:
            raise RuntimeError("Unexpected common17 reasoner parameter contract")
        initial_state = initialized.state_dict()
        changed = [name for name in parameter_names if not torch.equal(saved[name], initial_state[name])]
        unchanged = sorted(set(parameter_names) - set(changed))
        if len(changed) != 11 or unchanged:
            raise RuntimeError(f"Fold {fold} did not update every trainable tensor: {unchanged}")
        if not torch.equal(saved["prior_logits"], initial_state["prior_logits"]):
            raise RuntimeError("The non-trainable prior buffer unexpectedly changed")
        result.append(
            {
                "fold": fold,
                "seed": seed,
                "fit_time_trainable_parameters": initialized.n_trainable_parameters,
                "trainable_state_tensors": len(parameter_names),
                "trainable_state_tensors_changed_from_seeded_initialization": len(changed),
                "unchanged_trainable_state_tensors": unchanged,
                "non_trainable_prior_buffer_unchanged": True,
            }
        )
    return result


def _audit_arm(spec: dict[str, Any]) -> dict[str, Any]:
    old_dir = Path(spec["old"])
    new_dir = Path(spec["new"])
    old_manifest_path = old_dir / "manifest.json"
    new_manifest_path = new_dir / "manifest.json"
    old_tensor_path = old_dir / "oof_predictions_and_states.safetensors"
    new_tensor_path = new_dir / "oof_predictions_and_states.safetensors"
    old_manifest = _json(old_manifest_path)
    new_manifest = _json(new_manifest_path)

    old_counts = [int(row["fit"]["trainable_parameters"]) for row in old_manifest["folds"]]
    new_counts = [int(row["fit"]["trainable_parameters"]) for row in new_manifest["folds"]]
    if old_counts != [0] * 5 or new_counts != [6_967] * 5:
        raise RuntimeError(f"Unexpected old/new fit metadata: {old_counts} -> {new_counts}")
    if old_manifest["metrics"] != new_manifest["metrics"]:
        raise RuntimeError("Aggregate metrics changed during metadata correction replay")
    if _fold_projection(old_manifest) != _fold_projection(new_manifest):
        raise RuntimeError("Fold losses, splits, or held-out metrics changed")

    old_sha = _sha256(old_tensor_path)
    new_sha = _sha256(new_tensor_path)
    if old_sha != new_sha:
        raise RuntimeError("Old/new safetensors files are not byte-identical")
    old_tensors = _tensor_inventory(old_tensor_path)
    new_tensors = _tensor_inventory(new_tensor_path)
    if tuple(old_tensors) != tuple(new_tensors):
        raise RuntimeError("Old/new tensor key order differs")
    unequal = [key for key in old_tensors if not torch.equal(old_tensors[key], new_tensors[key])]
    if unequal:
        raise RuntimeError(f"Old/new tensor payload differs: {unequal}")

    initialization = _initialization_change_audit(
        new_tensors, new_manifest, str(spec["state_prefix"])
    )
    return {
        "old_frozen_directory": str(old_dir.resolve(strict=True)),
        "corrected_replay_directory": str(new_dir.resolve(strict=True)),
        "old_manifest": {"path": str(old_manifest_path), "sha256": _sha256(old_manifest_path)},
        "corrected_manifest": {
            "path": str(new_manifest_path),
            "sha256": _sha256(new_manifest_path),
        },
        "old_and_corrected_tensor_file_sha256": old_sha,
        "tensor_count": len(old_tensors),
        "all_tensor_keys_and_payloads_bitwise_identical": True,
        "aggregate_metrics_identical": True,
        "fold_splits_losses_and_held_metrics_identical": True,
        "old_fit_trainable_parameters_per_fold": old_counts,
        "corrected_fit_trainable_parameters_per_fold": new_counts,
        "seeded_initialization_change_evidence": initialization,
    }


def main() -> None:
    source = Path(__file__).resolve()
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "metadata_error_corrected_by_deterministic_replay",
        "error": {
            "affected_field": "folds[*].fit.trainable_parameters",
            "cause": (
                "The property counted parameters with requires_grad=True after "
                "model.eval().requires_grad_(False), so a fitted model was reported as zero."
            ),
            "fix": (
                "Cache model.n_trainable_parameters immediately after construction and before "
                "training/freezing; write that fit-time value to the fold receipt."
            ),
            "correct_fit_time_parameter_count": 6_967,
            "numerical_results_affected": False,
        },
        "arms": {name: _audit_arm(spec) for name, spec in ARMS.items()},
        "old_frozen_directories_modified": False,
        "audit_script": {"path": str(source), "sha256": _sha256(source)},
        "access_receipt": {
            "raw_EEG_loaded": False,
            "private_data_loaded": False,
            "edf_annotations_loaded": False,
            "excel_or_doctor_text_loaded": False,
        },
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "receipt_sha256": receipt["receipt_sha256"]}))


if __name__ == "__main__":
    main()
