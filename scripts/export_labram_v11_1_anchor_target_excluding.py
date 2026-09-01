#!/usr/bin/env python3
"""Export the frozen v11.1 CAR anchor into a target-excluding bridge.

The historical OOF safetensors container co-locates predictions with SOZ
targets.  MRSC must not consume that mixed container at inference time.  This
one-way exporter reads only the explicitly allow-listed prediction and roster
tensors, removes the non-candidate PZ carrier, binds the exact patient/event
order from the signal-only public-development union, and writes a new bundle
that contains no target or target-mask tensor.

This command does not evaluate predictions, choose a threshold, train a model,
or read private data.  Opening a historical mixed container is disclosed in
the output manifest; only the listed prediction tensors are materialized.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

from safetensors import safe_open
from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANCHOR = (
    ROOT / "outputs/labram_fine_temporal_nested_oof_v11_1_20260811_r2"
)
DEFAULT_UNION = ROOT / "outputs/public_development_union_v11_20260811"
DEFAULT_OUTPUT = ROOT / "outputs/labram_v11_1_anchor_target_excluding_20260812"

SCHEMA = "soz_labram_v11_1_target_excluding_anchor_bridge_v1"
ANCHOR_SCHEMA = "soz_labram_fine_temporal_nested_oof_v11_1"
UNION_SCHEMA = "soz_public_development_union_v11"
PATIENT_SCORE_KEY = "oof.full_frozen_labram_plus_fine"
EVENT_SCORE_KEY = "oof.event_full"
SOURCE_TENSOR_KEYS_READ = (
    PATIENT_SCORE_KEY,
    EVENT_SCORE_KEY,
    "patient_event_counts",
    "patient_folds",
    "config.candidate_mask",
)
EXPECTED_CANDIDATE_INDICES = tuple(index for index in range(19) if index != 14)


def _load_json(path: Path, *, name: str) -> dict[str, object]:
    source = path.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"{name} must be a canonical regular file")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must contain one JSON object")
    return payload


def _require_string_list(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    result = tuple(str(item).strip() for item in value)
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must be non-empty and unique")
    return result


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    fd, staging = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    except Exception:
        try:
            os.unlink(staging)
        except FileNotFoundError:
            pass
        raise


def export_target_excluding_anchor(
    *,
    anchor_directory: Path,
    union_directory: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Materialize an outcome-free, fixed-18 CAR anchor bundle."""

    anchor_root = anchor_directory.resolve(strict=True)
    union_root = union_directory.resolve(strict=True)
    if not anchor_root.is_dir() or anchor_root.is_symlink():
        raise ValueError("anchor_directory must be a canonical directory")
    if not union_root.is_dir() or union_root.is_symlink():
        raise ValueError("union_directory must be a canonical directory")
    target = output_directory.absolute()
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)

    anchor_manifest = _load_json(anchor_root / "manifest.json", name="anchor manifest")
    union_manifest = _load_json(union_root / "manifest.json", name="union manifest")
    if anchor_manifest.get("schema_version") != ANCHOR_SCHEMA:
        raise ValueError("Unsupported v11.1 anchor manifest")
    if anchor_manifest.get("status") != "completed_internal_developmental_nested_oof":
        raise ValueError("v11.1 anchor is not a completed developmental OOF bundle")
    if union_manifest.get("schema_version") != UNION_SCHEMA:
        raise ValueError("Unsupported signal-only public-development union")

    union_access = union_manifest.get("access_receipt")
    if not isinstance(union_access, Mapping):
        raise TypeError("Signal union lacks an access receipt")
    for field in (
        "deepsoz_target_values_loaded",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "source_eval_target_values_loaded",
        "prediction_artifacts_loaded",
    ):
        if union_access.get(field) is not False:
            raise ValueError(f"Signal union is not target/private free: {field}")

    patient_ids = _require_string_list(anchor_manifest.get("patient_ids"), name="patient_ids")
    if len(patient_ids) != 101 or anchor_manifest.get("primary_patient_count") != 101:
        raise ValueError("Frozen v11.1 complete-case patient roster changed")
    patient_to_index = {patient: index for index, patient in enumerate(patient_ids)}

    raw_events = union_manifest.get("events")
    if not isinstance(raw_events, list):
        raise TypeError("Signal union lacks event rows")
    event_ids: list[str] = []
    event_patient_indices: list[int] = []
    derived_counts = torch.zeros(len(patient_ids), dtype=torch.long)
    for raw in raw_events:
        if not isinstance(raw, Mapping):
            raise TypeError("Signal union contains a non-object event row")
        patient = str(raw.get("patient_id", "")).strip()
        if patient not in patient_to_index:
            continue
        event_id = str(raw.get("event_id", "")).strip()
        if not event_id:
            raise ValueError("Signal union contains an empty event identity")
        event_ids.append(event_id)
        patient_index = patient_to_index[patient]
        event_patient_indices.append(patient_index)
        derived_counts[patient_index] += 1
    if len(event_ids) != 984 or len(set(event_ids)) != len(event_ids):
        raise ValueError("Frozen v11.1 complete-case event roster changed")

    source_tensor = anchor_root / "oof_predictions.safetensors"
    if not source_tensor.is_file() or source_tensor.is_symlink():
        raise ValueError("Anchor prediction container must be a canonical regular file")
    with safe_open(str(source_tensor), framework="pt", device="cpu") as handle:
        available_keys = set(handle.keys())
        missing = tuple(key for key in SOURCE_TENSOR_KEYS_READ if key not in available_keys)
        if missing:
            raise ValueError(f"Anchor prediction container lacks required tensors: {missing}")
        patient_scores_19 = handle.get_tensor(PATIENT_SCORE_KEY).detach()
        event_scores_19 = handle.get_tensor(EVENT_SCORE_KEY).detach()
        patient_event_counts = handle.get_tensor("patient_event_counts").detach()
        patient_folds = handle.get_tensor("patient_folds").detach()
        candidate_mask = handle.get_tensor("config.candidate_mask").detach()

    if tuple(patient_scores_19.shape) != (101, 19):
        raise ValueError("Anchor patient scores must have shape [101,19]")
    if tuple(event_scores_19.shape) != (984, 19):
        raise ValueError("Anchor event scores must have shape [984,19]")
    if tuple(patient_event_counts.shape) != (101,) or patient_event_counts.dtype != torch.long:
        raise TypeError("Anchor patient_event_counts must be long [101]")
    if tuple(patient_folds.shape) != (101,) or patient_folds.dtype != torch.long:
        raise TypeError("Anchor patient_folds must be long [101]")
    if tuple(candidate_mask.shape) != (19,) or candidate_mask.dtype != torch.bool:
        raise TypeError("Anchor candidate mask must be bool [19]")
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if not torch.equal(
        candidate_indices,
        torch.tensor(EXPECTED_CANDIDATE_INDICES, dtype=torch.long),
    ):
        raise ValueError("Anchor must expose the frozen 18 candidates with PZ excluded")
    if not torch.equal(patient_event_counts.cpu(), derived_counts):
        raise ValueError("Anchor event counts disagree with the signal-only event roster")
    if int(patient_event_counts.sum()) != len(event_ids):
        raise ValueError("Anchor event counts do not cover all complete-case events")

    patient_scores = patient_scores_19.index_select(1, candidate_indices).contiguous()
    event_scores = event_scores_19.index_select(1, candidate_indices).contiguous()
    if not torch.isfinite(patient_scores).all() or not torch.isfinite(event_scores).all():
        raise ValueError("Fixed-18 anchor scores must be finite")
    candidate_names = tuple(
        name
        for index, name in enumerate(
            ("FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ", "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2")
        )
        if bool(candidate_mask[index])
    )
    if len(candidate_names) != 18 or "PZ" in candidate_names:
        raise ValueError("Fixed candidate ontology changed")

    target.mkdir()
    tensor_path = target / "anchor_scores.safetensors"
    try:
        save_file(
            {
                "car_patient_scores": patient_scores,
                "car_event_scores": event_scores,
                "event_patient_index": torch.tensor(event_patient_indices, dtype=torch.long),
                "patient_event_counts": patient_event_counts.cpu().contiguous(),
                "patient_folds": patient_folds.cpu().contiguous(),
                "candidate_indices": candidate_indices.cpu().long().contiguous(),
            },
            str(tensor_path),
        )
        manifest: dict[str, object] = {
            "schema_version": SCHEMA,
            "status": "completed_target_excluding_anchor_bridge",
            "model_lineage": "frozen_v11_1_full_frozen_labram_plus_fine_oof",
            "score_semantics": "uncalibrated_fixed_18_candidate_scores",
            "preprocessing_primary": "C-CAR19",
            "patient_count": len(patient_ids),
            "event_count": len(event_ids),
            "candidate_count": len(candidate_names),
            "candidate_channels": list(candidate_names),
            "patient_ids": list(patient_ids),
            "event_ids": event_ids,
            "tensor_file": tensor_path.name,
            "tensor_keys": [
                "candidate_indices",
                "car_event_scores",
                "car_patient_scores",
                "event_patient_index",
                "patient_event_counts",
                "patient_folds",
            ],
            "access_receipt": {
                "historical_mixed_prediction_container_opened": True,
                "historical_container_contains_target_tensors": (
                    "targets" in available_keys or "target_mask" in available_keys
                ),
                "source_tensor_keys_read": list(SOURCE_TENSOR_KEYS_READ),
                "target_tensor_values_loaded": False,
                "target_metrics_computed": False,
                "private_eeg_loaded": False,
                "private_target_values_loaded": False,
                "training_performed": False,
                "threshold_or_model_selection_performed": False,
            },
            "claim_boundary": {
                "developmental_oof_not_external_validation": True,
                "bridge_contains_no_soz_targets": True,
                "bridge_does_not_supply_ref19_scores": True,
                "bridge_does_not_calibrate_mrsc": True,
                "patient_folds_are_audit_only_forbidden_as_mrsc_features": True,
            },
        }
        _atomic_json(target / "manifest.json", manifest)
    except Exception:
        for child in target.iterdir():
            child.unlink()
        target.rmdir()
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-directory", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--union-directory", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = export_target_excluding_anchor(
        anchor_directory=args.anchor_directory,
        union_directory=args.union_directory,
        output_directory=args.output_directory,
    )
    print(json.dumps({
        "output": str(args.output_directory),
        "patient_count": manifest["patient_count"],
        "event_count": manifest["event_count"],
        "target_tensor_values_loaded": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
