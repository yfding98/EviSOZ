#!/usr/bin/env python3
"""Export the frozen identity-v16 C-CAR19 OOF scores without SOZ targets.

The v16 OOF container intentionally co-locates predictions and developmental
SOZ targets.  Downstream MRSC/report assembly must never consume that mixed
container.  This one-way exporter opens only a fixed allow-list of score and
roster tensors, removes the non-candidate PZ carrier, binds the result to the
target-free identity-v12 union, and publishes a new closed bundle.

No target tensor value is read.  This command does not train, calibrate,
select a model, evaluate an SOZ outcome, or access private data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.soz.data.identity_v12_cache_extension import (  # noqa: E402
    file_sha256,
    tensor_bitwise_equal,
    tensor_sha256,
)
from src.soz.data.public_development_union_identity_v12 import (  # noqa: E402
    EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_MANIFEST_SHA256,
    EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_PAYLOAD_SHA256,
    load_public_development_union_identity_v12,
)
from src.soz.geometry import STANDARD_19  # noqa: E402
from src.soz.v11_reasoner import (  # noqa: E402
    V11_CANDIDATE_INDICES,
    V11_CANDIDATE_MASK,
)


DEFAULT_SOURCE = ROOT / "outputs/labram_identity_recovery_closed_replay_v16_20260812"
DEFAULT_UNION = ROOT / "outputs/public_development_union_identity_v12_20260812"
DEFAULT_OUTPUT = ROOT / "outputs/labram_identity_v16_anchor_target_excluding_20260812"

SCHEMA = "soz_labram_identity_v16_target_excluding_anchor_bridge_v1"
SOURCE_SCHEMA = "soz_labram_identity_recovery_closed_replay_v16"
SOURCE_STATUS = "completed_internal_developmental_identity_recovery_closed_replay"
SOURCE_MANIFEST_SHA256 = "6b3eedd2af91f5d1905076a85c6990d35bea6b9e2b0d73fe062aa321f68562bb"
SOURCE_OOF_SHA256 = "3cf8b5b4659e3664cc8de1a9b1be7137c7bb3e5fac889482c112555c04ae456e"
PATIENT_SCORE_KEY = "oof.full_frozen_labram_plus_fine"
EVENT_SCORE_KEY = "oof.event_full"
PRIMARY_PATIENT_COUNT = 102
PRIMARY_EVENT_COUNT = 1145
BLOCKED_PATIENT_ID = "258"

SOURCE_TENSOR_KEYS_READ = (
    PATIENT_SCORE_KEY,
    EVENT_SCORE_KEY,
    "patient_event_counts",
    "patient_folds",
    "config.candidate_mask",
)

# The exact mixed-source vocabulary is pinned without opening the two target
# tensors.  An added private/label port therefore fails before any score read.
EXPECTED_MIXED_SOURCE_TENSOR_KEYS = frozenset(
    {
        "config.candidate_mask",
        "new10489_primary_index",
        "old101_primary_indices",
        "oof.event_full",
        "oof.fine_change_only",
        "oof.frozen_labram_only",
        "oof.full_frozen_labram_plus_fine",
        "oof.prevalence_only",
        "patient_artifact_quality",
        "patient_event_counts",
        "patient_folds",
        "target_mask",
        "targets",
    }
)
OUTPUT_TENSOR_KEYS = frozenset(
    {
        "candidate_indices",
        "car_event_scores",
        "car_patient_scores",
        "event_patient_index",
        "patient_event_counts",
        "patient_folds",
    }
)


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _canonical_json(path: Path, *, expected_sha256: str, name: str) -> dict[str, object]:
    source = path.resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{name} must be a canonical regular file")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{name} SHA256 mismatch")

    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate field {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{name} contains non-finite constant {value}")

    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict) or raw != _canonical_bytes(payload, newline=True):
        raise ValueError(f"{name} is not canonical JSON")
    return payload


def _publish_json(path: Path, payload: Mapping[str, object]) -> None:
    raw = _canonical_bytes(payload, newline=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _require_long(value: torch.Tensor, *, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    if value.dtype != torch.long or tuple(value.shape) != shape or value.requires_grad:
        raise TypeError(f"{name} must be detached torch.long with shape {list(shape)}")
    return value.detach().cpu().contiguous()


def _require_scores(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if (
        not value.is_floating_point()
        or tuple(value.shape) != shape
        or value.requires_grad
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be finite detached floating point {list(shape)}")
    return value.detach().cpu().contiguous()


def export_target_excluding_identity_v16_anchor(
    *,
    source_directory: Path,
    union_directory: Path,
    output_directory: Path,
    expected_source_manifest_sha256: str = SOURCE_MANIFEST_SHA256,
    expected_source_oof_sha256: str = SOURCE_OOF_SHA256,
) -> dict[str, object]:
    """Publish the fixed-C18 identity-v16 scores behind a target-free port."""

    source_root = source_directory.resolve(strict=True)
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("source_directory must be canonical and non-symlinked")
    union = load_public_development_union_identity_v12(
        union_directory,
        expected_manifest_sha256=(
            EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_MANIFEST_SHA256
        ),
        expected_payload_sha256=(
            EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_PAYLOAD_SHA256
        ),
    )
    manifest = _canonical_json(
        source_root / "manifest.json",
        expected_sha256=expected_source_manifest_sha256,
        name="identity-v16 source manifest",
    )
    if manifest.get("schema_version") != SOURCE_SCHEMA or (
        manifest.get("status") != SOURCE_STATUS
    ):
        raise ValueError("identity-v16 source schema/status changed")
    if (
        manifest.get("primary_patient_count") != PRIMARY_PATIENT_COUNT
        or manifest.get("primary_event_count") != PRIMARY_EVENT_COUNT
        or manifest.get("union_patient_count") != 103
        or manifest.get("union_event_count") != 1149
        or manifest.get("excluded_partial_reference_patients") != [BLOCKED_PATIENT_ID]
    ):
        raise ValueError("identity-v16 source cohort contract changed")
    foundation = manifest.get("foundation")
    if not isinstance(foundation, Mapping) or (
        foundation.get("backbone") != "official_pretrained_LaBraM_Base_not_replaced"
        or foundation.get("trained_from_scratch") is not False
        or foundation.get("foundation_trainable_parameters") != 0
    ):
        raise ValueError("identity-v16 LaBraM foundation contract changed")
    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping) or (
        lineage.get("union_manifest_sha256") != union.manifest_sha256
    ):
        raise ValueError("identity-v16 and identity-v12 union lineage differ")
    access = manifest.get("access_receipt")
    if not isinstance(access, Mapping) or any(
        access.get(field) is not False
        for field in (
            "private_eeg_loaded",
            "private_target_values_loaded",
            "foundation_training_performed",
            "llm_used_as_soz_predictor",
        )
    ):
        raise ValueError("identity-v16 source violates the private/foundation boundary")

    patient_ids = tuple(
        patient for patient in union.patient_ids if patient != BLOCKED_PATIENT_ID
    )
    blocked_events = tuple(
        event for event in union.events if event.patient_id == BLOCKED_PATIENT_ID
    )
    selected_events = tuple(
        event for event in union.events if event.patient_id != BLOCKED_PATIENT_ID
    )
    if (
        len(patient_ids) != PRIMARY_PATIENT_COUNT
        or len(selected_events) != PRIMARY_EVENT_COUNT
        or len(blocked_events) != 4
        or tuple(str(value) for value in manifest.get("patient_ids", ())) != patient_ids
    ):
        raise ValueError("identity-v16 target-free roster does not close")
    patient_position = {patient: index for index, patient in enumerate(patient_ids)}
    event_ids = tuple(event.event_id for event in selected_events)
    event_patient_index = torch.tensor(
        [patient_position[event.patient_id] for event in selected_events],
        dtype=torch.long,
    )
    derived_counts = torch.bincount(
        event_patient_index, minlength=PRIMARY_PATIENT_COUNT
    ).long()
    expected_folds = torch.tensor(
        [union.patient_folds[union.patient_index[patient]] for patient in patient_ids],
        dtype=torch.long,
    )
    if manifest.get("event_counts") != derived_counts.tolist() or (
        manifest.get("patient_folds") != expected_folds.tolist()
    ):
        raise ValueError("identity-v16 manifest routing differs from the v12 union")

    oof_path = source_root / "oof_predictions.safetensors"
    if oof_path.is_symlink() or not oof_path.is_file() or (
        file_sha256(oof_path) != expected_source_oof_sha256
    ):
        raise ValueError("identity-v16 OOF prediction file SHA256 changed")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("identity-v16 manifest lacks file receipts")
    oof_receipt = files.get(oof_path.name)
    if not isinstance(oof_receipt, Mapping) or (
        oof_receipt.get("sha256") != expected_source_oof_sha256
        or oof_receipt.get("size_bytes") != oof_path.stat().st_size
    ):
        raise ValueError("identity-v16 manifest OOF receipt changed")

    with safe_open(str(oof_path), framework="pt", device="cpu") as handle:
        available = frozenset(handle.keys())
        if available != EXPECTED_MIXED_SOURCE_TENSOR_KEYS:
            raise ValueError(
                "identity-v16 mixed tensor vocabulary changed: "
                f"missing={sorted(EXPECTED_MIXED_SOURCE_TENSOR_KEYS - available)}, "
                f"unexpected={sorted(available - EXPECTED_MIXED_SOURCE_TENSOR_KEYS)}"
            )
        # Do not call get_tensor for targets or target_mask.
        source_values = {
            key: handle.get_tensor(key).detach() for key in SOURCE_TENSOR_KEYS_READ
        }

    patient_scores_19 = _require_scores(
        source_values[PATIENT_SCORE_KEY],
        name=PATIENT_SCORE_KEY,
        shape=(PRIMARY_PATIENT_COUNT, 19),
    )
    event_scores_19 = _require_scores(
        source_values[EVENT_SCORE_KEY],
        name=EVENT_SCORE_KEY,
        shape=(PRIMARY_EVENT_COUNT, 19),
    )
    source_counts = _require_long(
        source_values["patient_event_counts"],
        name="patient_event_counts",
        shape=(PRIMARY_PATIENT_COUNT,),
    )
    source_folds = _require_long(
        source_values["patient_folds"],
        name="patient_folds",
        shape=(PRIMARY_PATIENT_COUNT,),
    )
    candidate_mask = source_values["config.candidate_mask"].detach().cpu()
    if not torch.equal(candidate_mask, V11_CANDIDATE_MASK):
        raise ValueError("identity-v16 fixed-C18 candidate mask changed")
    if not torch.equal(source_counts, derived_counts) or not torch.equal(
        source_folds, expected_folds
    ):
        raise ValueError("identity-v16 tensor routing differs from target-free union")

    candidate_indices = torch.tensor(V11_CANDIDATE_INDICES, dtype=torch.long)
    candidate_channels = tuple(STANDARD_19[index] for index in V11_CANDIDATE_INDICES)
    patient_scores = patient_scores_19.index_select(1, candidate_indices).contiguous()
    event_scores = event_scores_19.index_select(1, candidate_indices).contiguous()
    if not tensor_bitwise_equal(
        patient_scores, patient_scores_19[:, V11_CANDIDATE_MASK].contiguous()
    ) or not tensor_bitwise_equal(
        event_scores, event_scores_19[:, V11_CANDIDATE_MASK].contiguous()
    ):
        raise RuntimeError("fixed-C18 score extraction changed source score bytes")

    target = Path(os.path.abspath(output_directory))
    if os.path.lexists(target):
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        tensor_path = staging / "anchor_scores.safetensors"
        output_tensors = {
            "candidate_indices": candidate_indices,
            "car_event_scores": event_scores,
            "car_patient_scores": patient_scores,
            "event_patient_index": event_patient_index,
            "patient_event_counts": derived_counts,
            "patient_folds": expected_folds,
        }
        if frozenset(output_tensors) != OUTPUT_TENSOR_KEYS:
            raise RuntimeError("target-excluding output vocabulary changed")
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in output_tensors.items()},
            str(tensor_path),
        )
        tensor_file_sha = file_sha256(tensor_path)
        output_manifest: dict[str, object] = {
            "schema_version": SCHEMA,
            "status": "completed_target_excluding_identity_v16_anchor_bridge",
            "model_lineage": "identity_v16_frozen_outer_fold_full_labram_plus_fine_oof",
            "score_semantics": "uncalibrated_fixed_18_candidate_scores",
            "preprocessing_primary": "C-CAR19",
            "patient_count": PRIMARY_PATIENT_COUNT,
            "event_count": PRIMARY_EVENT_COUNT,
            "candidate_count": len(candidate_channels),
            "candidate_channels": list(candidate_channels),
            "patient_ids": list(patient_ids),
            "event_ids": list(event_ids),
            "tensor_file": tensor_path.name,
            "tensor_file_sha256": tensor_file_sha,
            "tensor_keys": sorted(OUTPUT_TENSOR_KEYS),
            "tensor_specs": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in sorted(output_tensors.items())
            },
            "score_integrity": {
                "car_patient_scores_tensor_sha256": tensor_sha256(patient_scores),
                "car_event_scores_tensor_sha256": tensor_sha256(event_scores),
                "fixed18_selection_elementwise_bitwise_equal_to_source": True,
                "score_or_ranking_change_performed": False,
            },
            "report_join_blocks": [
                {
                    "patient_id": event.patient_id,
                    "event_id": event.event_id,
                    "reason": "mrsc_anchor_identity_not_available",
                }
                for event in blocked_events
            ],
            "lineage": {
                "source_manifest_sha256": expected_source_manifest_sha256,
                "source_oof_predictions_sha256": expected_source_oof_sha256,
                "union_manifest_sha256": union.manifest_sha256,
                "union_payload_sha256": (
                    EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_PAYLOAD_SHA256
                ),
            },
            "access_receipt": {
                "historical_mixed_prediction_container_opened": True,
                "historical_container_contains_target_tensors": True,
                "source_tensor_vocabulary_inspected": sorted(available),
                "source_tensor_keys_read": list(SOURCE_TENSOR_KEYS_READ),
                "target_tensor_values_loaded": False,
                "target_metrics_computed": False,
                "private_eeg_loaded": False,
                "private_target_values_loaded": False,
                "training_performed": False,
                "calibration_performed": False,
                "threshold_or_model_selection_performed": False,
            },
            "claim_boundary": {
                "developmental_oof_not_external_validation": True,
                "bridge_contains_no_soz_targets": True,
                "bridge_does_not_supply_ref19_scores": True,
                "bridge_does_not_calibrate_mrsc": True,
                "patient_258_four_events_are_explicitly_blocked": True,
                "patient_folds_are_checkpoint_routing_only": True,
            },
        }
        _publish_json(staging / "manifest.json", output_manifest)
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return output_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--source-directory", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--union-directory", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-source-manifest-sha256", default=SOURCE_MANIFEST_SHA256
    )
    parser.add_argument("--expected-source-oof-sha256", default=SOURCE_OOF_SHA256)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = export_target_excluding_identity_v16_anchor(
        source_directory=args.source_directory,
        union_directory=args.union_directory,
        output_directory=args.output_directory,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        expected_source_oof_sha256=args.expected_source_oof_sha256,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_directory),
                "patient_count": manifest["patient_count"],
                "event_count": manifest["event_count"],
                "blocked_event_count": len(manifest["report_join_blocks"]),
                "target_tensor_values_loaded": False,
                "private_used": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
