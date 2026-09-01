#!/usr/bin/env python3
"""Run the formal source-train preprocessing-parity experiment.

This is a resumable, target-safe Stage-P runner.  It consumes only native
TUEV CE6 and TUSZ bipolar edge-time involvement labels.  DeepSOZ, SOZ labels,
source-dev/eval and private data are not accepted by the CLI.

The four deployable arms are extracted in one paired pass over each EDF.  A
single frozen LaBraM backbone uses position IDs derived from each record's
actual raw headers.  Five patient/content-component-disjoint folds train the
same lightweight morphology and ictal heads under the same fixed schedule.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Iterable, Mapping, Sequence


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/neurosoz-numba-cache")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.soz.data.tuev_morphology import (  # noqa: E402
    load_tuev_morphology_manifest,
)
from src.soz.data.tusz import load_tusz_ictal_involvement_target  # noqa: E402
from src.soz.data.tusz_training import (  # noqa: E402
    load_tusz_ictal_training_manifest,
    parse_tusz_official_train_path,
)
from src.soz.geometry import MORPHOLOGY_CLASSES  # noqa: E402
from src.soz.models.concept_heads import (  # noqa: E402
    IctalInvolvementHead,
    MorphologyEvidenceHead,
)
from src.soz.models.labram import (  # noqa: E402
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    LABRAM_RAW_HEADER_POSITION_BINDING_POLICY,
    OfficialLaBraMEncoder,
    bind_labram_record_positions,
)
from src.soz.preprocessing_arm_runtime import (  # noqa: E402
    OfficialReference23LaBraMEncoder,
    prepare_arm_interval,
    prepare_full_record_arm,
    read_physical_edf,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    DEPLOYABLE_PREPROCESSING_ARM_IDS,
    FROZEN_PREPROCESSING_ARM_SPEC_BY_ID,
    FROZEN_PREPROCESSING_ARM_SPECS_SHA256,
    LOCKED_PREPROCESSING_SELECTION_POLICY_RECEIPT_SHA256,
    OFFICIAL_REF23_CHANNELS,
    PREPROCESSING_ARM_IDS,
    PreprocessingArmResultReceipt,
    PreprocessingArmSelectionMetrics,
    PreprocessingArmSelectionNoGoError,
    PreprocessingNestedDevSourceRecord,
    PreprocessingParityProtocolReceipt,
    build_preprocessing_parity_nested_dev_manifest,
    evaluate_preprocessing_arm_selection,
    materialize_preprocessing_selection_bundle,
    preprocessing_foundation_policy_receipt_sha256,
)


RUN_SCHEMA = "soz_preprocessing_parity_formal_runner_v2"
EXTRACTION_SCHEMA = "soz_preprocessing_parity_record_extraction_v1"
RESULT_SCHEMA = "soz_preprocessing_parity_formal_run_result_v2"
SEED = 20260808
FOLD_COUNT = 5
MORPHOLOGY_EPOCHS = 20
ICTAL_EPOCHS = 20
BOOTSTRAP_REPLICATES = 2000
ARM_IDS = tuple(DEPLOYABLE_PREPROCESSING_ARM_IDS)


def _canonical_json(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _hash_payload(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _json_equivalent(left: object, right: object) -> bool:
    """Compare JSON contracts after tuple/list normalization."""

    return _canonical_json(left) == _canonical_json(right)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = _canonical_json(
        {"dtype": str(array.dtype), "shape": list(array.shape)}
    )
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _sha(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256")
    return normalized


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(payload, newline=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _log(event: str, **values: object) -> None:
    print(
        json.dumps(
            {"event": event, **values},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuev-root", type=Path, required=True)
    parser.add_argument("--tuev-manifest", type=Path, required=True)
    parser.add_argument("--tuev-bundle-sha256", type=_sha, required=True)
    parser.add_argument("--tuev-receipt-sha256", type=_sha, required=True)
    parser.add_argument("--tusz-root", type=Path, required=True)
    parser.add_argument("--tusz-manifest", type=Path, required=True)
    parser.add_argument("--tusz-bundle-sha256", type=_sha, required=True)
    parser.add_argument("--tusz-receipt-sha256", type=_sha, required=True)
    parser.add_argument("--labram-modeling", type=Path, required=True)
    parser.add_argument("--labram-checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--reuse-nominal-directory",
        type=Path,
        default=None,
        help=(
            "Reuse a completed v1 paired extraction, fixed-fold checkpoints, "
            "and nominal native-task endpoints. Only the corrected, "
            "label-preserving v2 robustness experiment is recomputed."
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--morphology-batch-size", type=int, default=4)
    parser.add_argument("--ictal-event-batch-size", type=int, default=1)
    parser.add_argument(
        "--stop-after-extraction",
        action="store_true",
        help="Publish the paired token corpus and stop before head optimization.",
    )
    return parser


def _bundle_directory(path: Path) -> Path:
    return path.parent if path.name == "manifest.json" else path


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _assign_folds(rows: Sequence[dict[str, object]]) -> dict[tuple[str, str], int]:
    keys = tuple(f"{row['dataset_id']}:{row['record_id']}" for row in rows)
    uf = _UnionFind(keys)
    patient_owner: dict[str, str] = {}
    content_owner: dict[str, str] = {}
    for key, row in zip(keys, rows):
        patient = str(row["patient_identity_key"])
        content = str(row["content_component_id"])
        if patient in patient_owner:
            uf.union(key, patient_owner[patient])
        else:
            patient_owner[patient] = key
        if content in content_owner:
            uf.union(key, content_owner[content])
        else:
            content_owner[content] = key
    components: dict[str, list[tuple[str, dict[str, object]]]] = defaultdict(list)
    for key, row in zip(keys, rows):
        components[uf.find(key)].append((key, row))

    def component_load(
        component: Sequence[tuple[str, dict[str, object]]],
    ) -> dict[str, float]:
        load = {
            "record:all": float(len(component)),
            "record:TUEV": 0.0,
            "record:TUSZ": 0.0,
            "item:TUEV": 0.0,
            "item:TUSZ": 0.0,
            "evaluation:TUEV": 0.0,
            "evaluation:TUSZ": 0.0,
        }
        evaluation_ids = {"TUEV": set(), "TUSZ": set()}
        for _, row in component:
            dataset = str(row["dataset_id"])
            load[f"record:{dataset}"] += 1.0
            load[f"item:{dataset}"] += float(row.get("item_count", 1))
            declared = row.get(
                "evaluation_unit_ids",
                (str(row["patient_identity_key"]),),
            )
            if not isinstance(declared, (tuple, list)) or not declared:
                raise ValueError("Fold row has no evaluation units")
            evaluation_ids[dataset].update(str(value) for value in declared)
        for dataset in ("TUEV", "TUSZ"):
            load[f"evaluation:{dataset}"] = float(len(evaluation_ids[dataset]))
        return load

    component_loads = {
        root: component_load(component) for root, component in components.items()
    }
    totals = {
        metric: sum(load[metric] for load in component_loads.values())
        for metric in next(iter(component_loads.values()))
    }
    active_metrics = tuple(metric for metric, total in totals.items() if total > 0)
    targets = {metric: totals[metric] / FOLD_COUNT for metric in active_metrics}
    ordered = sorted(
        components.items(),
        key=lambda pair: (
            -max(
                component_loads[pair[0]][metric] / targets[metric]
                for metric in active_metrics
            ),
            -sum(component_loads[pair[0]][metric] for metric in active_metrics),
            min(key for key, _ in pair[1]),
        ),
    )
    fold_loads = [
        {metric: 0.0 for metric in active_metrics} for _ in range(FOLD_COUNT)
    ]
    assigned: dict[tuple[str, str], int] = {}
    for root, component in ordered:
        contribution = component_loads[root]

        def assignment_score(candidate: int) -> tuple[float, float, float, int]:
            normalized = {
                (fold_index, metric): (
                    fold_loads[fold_index][metric]
                    + (contribution[metric] if fold_index == candidate else 0.0)
                )
                / targets[metric]
                for fold_index in range(FOLD_COUNT)
                for metric in active_metrics
            }
            maximum_load = max(normalized.values())
            squared_imbalance = sum(
                (value - 1.0) ** 2 for value in normalized.values()
            )
            candidate_load = sum(
                normalized[(candidate, metric)] for metric in active_metrics
            )
            return maximum_load, squared_imbalance, candidate_load, candidate

        fold = min(
            range(FOLD_COUNT),
            key=assignment_score,
        )
        for _, row in component:
            dataset = str(row["dataset_id"])
            record_id = str(row["record_id"])
            assigned[(dataset, record_id)] = fold
        for metric in active_metrics:
            fold_loads[fold][metric] += contribution[metric]
    for dataset in ("TUEV", "TUSZ"):
        if {fold for (source, _), fold in assigned.items() if source == dataset} != set(
            range(FOLD_COUNT)
        ):
            raise RuntimeError(f"{dataset} does not contribute records to every fold")
    return assigned


def _build_source_plan(tuev, tusz):
    fit_groups = set(tuev.fit_group_ids)
    candidate_tuev_records = {
        record.record_id: record
        for record in tuev.records
        if record.official_split == "train" and record.parent_group_id in fit_groups
    }
    tuev_groups = tuple(
        sorted(
            (
                group
                for group in tuev.interval_groups
                if group.parent_group_id in fit_groups
            ),
            key=lambda group: (group.record_id, group.start_sample, group.crop_id),
        )
    )
    used_tuev_records = {group.record_id for group in tuev_groups}
    if not used_tuev_records <= set(candidate_tuev_records):
        raise ValueError("TUEV native targets reference a non-fit source record")
    tuev_records = {
        record_id: candidate_tuev_records[record_id]
        for record_id in sorted(used_tuev_records)
    }
    tuev_groups_by_record: dict[str, list[object]] = defaultdict(list)
    for group in tuev_groups:
        tuev_groups_by_record[group.record_id].append(group)

    tusz_events = tuple(
        sorted(
            tusz.events,
            key=lambda event: (
                event.relative_edf_path,
                event.event_index,
                event.event_id,
            ),
        )
    )
    events_by_record: dict[str, list[object]] = defaultdict(list)
    for event in tusz_events:
        events_by_record[event.record_id].append(event)

    raw_rows: list[dict[str, object]] = []
    for record_id, record in sorted(tuev_records.items()):
        raw_rows.append(
            {
                "dataset_id": "TUEV",
                "task_family": "morphology_ce6",
                "record_id": record_id,
                "patient_identity_key": str(record.source_subject_id),
                "content_component_id": f"exact-edf:{record.edf_sha256}",
                "edf_sha256": record.edf_sha256,
                "source_record_receipt_sha256": _hash_payload(
                    record.canonical_payload
                ),
                "raw_qc_receipt_sha256": (
                    record.metadata.signal_qc_receipt_sha256
                ),
                "item_count": len(tuev_groups_by_record[record_id]),
                "evaluation_unit_ids": tuple(
                    sorted(
                        {
                            group.parent_group_id
                            for group in tuev_groups_by_record[record_id]
                        }
                    )
                ),
            }
        )
    for record_id, events in sorted(events_by_record.items()):
        first = events[0]
        if any(
            event.patient_id != first.patient_id
            or event.edf_sha256 != first.edf_sha256
            or event.public_record_sha256 != first.public_record_sha256
            for event in events
        ):
            raise ValueError("One TUSZ record contains inconsistent source identity")
        raw_qc = _hash_payload(
            tuple(event.signal_preflight_receipt_sha256 for event in events)
        )
        raw_rows.append(
            {
                "dataset_id": "TUSZ",
                "task_family": "ictal_native",
                "record_id": record_id,
                "patient_identity_key": first.patient_id,
                "content_component_id": f"exact-edf:{first.edf_sha256}",
                "edf_sha256": first.edf_sha256,
                "source_record_receipt_sha256": first.public_record_sha256,
                "raw_qc_receipt_sha256": raw_qc,
                "item_count": len(events),
                "evaluation_unit_ids": (first.patient_id,),
            }
        )
    folds = _assign_folds(raw_rows)
    typed_records = tuple(
        PreprocessingNestedDevSourceRecord(
            dataset_id=str(row["dataset_id"]),
            task_family=str(row["task_family"]),
            record_id=str(row["record_id"]),
            patient_identity_key=str(row["patient_identity_key"]),
            content_component_id=str(row["content_component_id"]),
            edf_sha256=str(row["edf_sha256"]),
            source_record_receipt_sha256=str(
                row["source_record_receipt_sha256"]
            ),
            raw_qc_receipt_sha256=str(row["raw_qc_receipt_sha256"]),
            common_raw_qc_eligible=True,
            raw_qc_exclusion_code=None,
            nested_dev_fold=folds[(str(row["dataset_id"]), str(row["record_id"]))],
        )
        for row in raw_rows
    )
    nested = build_preprocessing_parity_nested_dev_manifest(
        records=typed_records,
        tuev_source_manifest_receipt_sha256=tuev.manifest_sha256,
        tusz_source_manifest_receipt_sha256=tusz.manifest_sha256,
    )
    fold_by_record = {
        (record.dataset_id, record.record_id): record.nested_dev_fold
        for record in nested.records
    }
    tuev_items = tuple(
        {
            "index": index,
            "crop_id": group.crop_id,
            "record_id": group.record_id,
            "parent_group_id": group.parent_group_id,
            "start_sec": group.start_sample / 200.0,
            "stop_sec": group.stop_sample / 200.0,
            "fold": fold_by_record[("TUEV", group.record_id)],
        }
        for index, group in enumerate(tuev_groups)
    )
    tusz_items = tuple(
        {
            "index": index,
            "event_id": event.event_id,
            "record_id": event.record_id,
            "patient_id": event.patient_id,
            "event_index": event.event_index,
            "start_sec": event.event_t0_sec - 12.0,
            "stop_sec": event.event_t0_sec + 48.0,
            "fold": fold_by_record[("TUSZ", event.record_id)],
        }
        for index, event in enumerate(tusz_events)
    )
    return nested, tuev_records, tuev_groups, tusz_events, tuev_items, tusz_items


def _protocol(
    nested, *, robustness_protocol: str = "label_preserving_v2"
) -> PreprocessingParityProtocolReceipt:
    if robustness_protocol not in {"legacy_misaligned_v1", "label_preserving_v2"}:
        raise ValueError("Unsupported preprocessing robustness protocol")
    foundation_policy_receipt_sha256 = (
        preprocessing_foundation_policy_receipt_sha256(
            checkpoint_sha256=AUDITED_LABRAM_BASE_SHA256,
            modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
            position_binding_policy=LABRAM_RAW_HEADER_POSITION_BINDING_POLICY,
            record_specific_position_ids=True,
            token_dim=200,
            input_scale_from_volts=1e4,
        )
    )
    head_policy = {
        "morphology": "NodeToEdge(left,right,left-right)-MLP600x128-CE6",
        "ictal": "NodeToEdge(left,right,left-right)-MLP600x128-binary",
    }
    optimizer_policy = {
        "morphology": {
            "epochs": MORPHOLOGY_EPOCHS,
            "optimizer": "AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "group_balanced": True,
        },
        "ictal": {
            "epochs": ICTAL_EPOCHS,
            "optimizer": "AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "patient_balanced": True,
        },
    }
    evaluation = {
        "tuev": "held-group-equal CE6 class-macro F1",
        "tusz": "held-patient-macro masked native BCE",
        "bootstrap": "paired component/patient bootstrap percentile95",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": SEED,
        "arm_id_probe": "fold4-patient-disjoint linear token-summary probe",
    }
    if robustness_protocol == "legacy_misaligned_v1":
        evaluation.update(
            {
                "jitter_seconds": [-0.5, 0.5],
                "jitter_subset": (
                    "one deterministic eligible item per held group/patient"
                ),
            }
        )
    else:
        evaluation.update(
            {
                "robustness_protocol": "label_preserving_v2",
                "tuev_slot_shift": {
                    "event_slots": [0, 1, 2, 3],
                    "window_seconds": 4,
                    "target_semantics": (
                        "the same native one-second CE6 event is evaluated at "
                        "its matching output slot"
                    ),
                },
                "tusz_onset_shift": {
                    "window_shift_seconds": [-1, 1],
                    "target_semantics": (
                        "native one-second targets and masks are reindexed onto "
                        "the shifted signal grid; the one out-of-range boundary "
                        "bin is masked"
                    ),
                },
                "robustness_subset": (
                    "one deterministic item per context-eligible held group/patient; "
                    "all omissions and coverage are receipted before arm evaluation"
                ),
            }
        )
    return PreprocessingParityProtocolReceipt(
        nested_dev_manifest_receipt_sha256=nested.receipt_sha256,
        source_patient_roster_sha256=nested.source_patient_roster_sha256,
        content_component_split_receipt_sha256=(
            nested.content_component_split_receipt_sha256
        ),
        raw_qc_intersection_receipt_sha256=(
            nested.raw_qc_intersection_receipt_sha256
        ),
        foundation_feature_receipt_sha256=foundation_policy_receipt_sha256,
        tuev_source_manifest_receipt_sha256=(
            nested.tuev_source_manifest_receipt_sha256
        ),
        tusz_source_manifest_receipt_sha256=(
            nested.tusz_source_manifest_receipt_sha256
        ),
        head_architecture_receipt_sha256=_hash_payload(head_policy),
        optimizer_schedule_receipt_sha256=_hash_payload(optimizer_policy),
        seed_roster_receipt_sha256=_hash_payload((SEED,)),
        evaluation_policy_receipt_sha256=_hash_payload(evaluation),
        selection_policy_receipt_sha256=(
            LOCKED_PREPROCESSING_SELECTION_POLICY_RECEIPT_SHA256
        ),
    )


def _ensure_memmap(path: Path, *, shape: tuple[int, ...], dtype: np.dtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"Existing array contract changed: {path}")
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _array_paths(output: Path) -> dict[str, Path]:
    arrays = output / "arrays"
    values = {
        f"tuev_tokens_{arm}": arrays / f"tuev_tokens_{arm}.npy"
        for arm in ARM_IDS
    }
    values.update(
        {
            f"tusz_tokens_{arm}": arrays / f"tusz_tokens_{arm}.npy"
            for arm in ARM_IDS
        }
    )
    values.update(
        {
            "tuev_labels": arrays / "tuev_labels.npy",
            "tuev_mask": arrays / "tuev_mask.npy",
            "tuev_weights": arrays / "tuev_weights.npy",
            "tusz_targets": arrays / "tusz_targets.npy",
            "tusz_mask": arrays / "tusz_mask.npy",
        }
    )
    return values


def _open_arrays(output: Path, n_tuev: int, n_tusz: int):
    paths = _array_paths(output)
    arrays = {
        f"tuev_tokens_{arm}": _ensure_memmap(
            paths[f"tuev_tokens_{arm}"],
            shape=(n_tuev, 19, 1, 200),
            dtype=np.float32,
        )
        for arm in ARM_IDS
    }
    arrays.update(
        {
            f"tusz_tokens_{arm}": _ensure_memmap(
                paths[f"tusz_tokens_{arm}"],
                shape=(n_tusz, 19, 60, 200),
                dtype=np.float32,
            )
            for arm in ARM_IDS
        }
    )
    arrays["tuev_labels"] = _ensure_memmap(
        paths["tuev_labels"], shape=(n_tuev, 20), dtype=np.int64
    )
    arrays["tuev_mask"] = _ensure_memmap(
        paths["tuev_mask"], shape=(n_tuev, 20), dtype=np.bool_
    )
    arrays["tuev_weights"] = _ensure_memmap(
        paths["tuev_weights"], shape=(n_tuev, 20), dtype=np.float32
    )
    arrays["tusz_targets"] = _ensure_memmap(
        paths["tusz_targets"], shape=(n_tusz, 20, 60), dtype=np.float32
    )
    arrays["tusz_mask"] = _ensure_memmap(
        paths["tusz_mask"], shape=(n_tusz, 20, 60), dtype=np.bool_
    )
    return paths, arrays


def _stats(values: np.ndarray) -> dict[str, object]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "count": int(data.size),
        "sum": float(data.sum(dtype=np.float64)),
        "sum_squares": float(np.square(data).sum(dtype=np.float64)),
        "absolute_max": float(np.abs(data).max()),
    }


def _torch_from_numpy_safe(values: object) -> torch.Tensor:
    """Create a tensor without exposing PyTorch to read-only memmap views."""

    array = np.asarray(values)
    if not array.flags.writeable or not array.flags.c_contiguous:
        array = np.array(array, copy=True, order="C")
    return torch.from_numpy(array)


def _record_receipt_path(output: Path, dataset: str, record_id: str) -> Path:
    key = hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:24]
    return output / "record-receipts" / f"{dataset.lower()}-{key}.json"


def _load_completed_receipt(
    path: Path,
    *,
    dataset: str,
    record_id: str,
    indices: Sequence[int],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != EXTRACTION_SCHEMA
        or payload.get("dataset") != dataset
        or payload.get("record_id") != record_id
        or payload.get("item_indices") != list(indices)
    ):
        raise ValueError(f"Extraction receipt identity changed: {path}")
    declared = payload.get("array_slice_sha256")
    if not isinstance(declared, dict):
        raise TypeError("Extraction receipt lacks array slice hashes")
    for name, expected in declared.items():
        if name not in arrays:
            raise ValueError("Extraction receipt names an unknown array")
        actual = _hash_array(np.asarray(arrays[name][list(indices)]))
        if actual != expected:
            raise ValueError(f"Completed extraction slice changed: {name}")
    return payload


def _encode_four_second(
    encoder: OfficialLaBraMEncoder,
    data: np.ndarray,
    binding,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)).reshape(
        -1, 19, 4, 200
    ).to(device)
    with torch.inference_mode():
        return encoder.forward_with_record_binding(tensor, binding).cpu()


def _encode_sixty_second(
    encoder: OfficialLaBraMEncoder,
    data: np.ndarray,
    binding,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)).to(
        device
    )
    chunks = []
    with torch.inference_mode():
        for call in range(15):
            patch = tensor[:, :, call * 800 : (call + 1) * 800].reshape(
                tensor.shape[0], 19, 4, 200
            )
            chunks.append(encoder.forward_with_record_binding(patch, binding))
    return torch.cat(chunks, dim=2).cpu()


def _prepare_full_arms(raw) -> dict[str, object | None]:
    return {
        arm: (
            None if arm == "C-CAR19" else prepare_full_record_arm(raw, arm_id=arm)
        )
        for arm in ARM_IDS
    }


def _extract_tuev(
    *,
    root: Path,
    output: Path,
    records: Mapping[str, object],
    groups: Sequence[object],
    arrays: Mapping[str, np.ndarray],
    encoder: OfficialLaBraMEncoder,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, object], ...]:
    groups_by_record: dict[str, list[tuple[int, object]]] = defaultdict(list)
    for index, group in enumerate(groups):
        groups_by_record[group.record_id].append((index, group))
    receipts: list[dict[str, object]] = []
    for record_index, (record_id, indexed_groups) in enumerate(
        sorted(groups_by_record.items())
    ):
        indices = [index for index, _ in indexed_groups]
        receipt_path = _record_receipt_path(output, "TUEV", record_id)
        completed = _load_completed_receipt(
            receipt_path,
            dataset="TUEV",
            record_id=record_id,
            indices=indices,
            arrays=arrays,
        )
        if completed is not None:
            receipts.append(completed)
            _log("extract_skip", dataset="TUEV", record_id=record_id)
            continue
        record = records[record_id]
        source = (root / record.relative_edf_path).resolve(strict=True)
        if _hash_file(source) != record.edf_sha256:
            raise ValueError(f"TUEV EDF changed: {record_id}")
        raw = read_physical_edf(source, geometry="standard19")
        binding = bind_labram_record_positions(raw.channel_names)
        full = _prepare_full_arms(raw)
        input_stats = {arm: {"count": 0, "sum": 0.0, "sum_squares": 0.0, "absolute_max": 0.0} for arm in ARM_IDS}
        token_stats = {arm: {"count": 0, "sum": 0.0, "sum_squares": 0.0, "absolute_max": 0.0} for arm in ARM_IDS}
        for start in range(0, len(indexed_groups), batch_size):
            batch = indexed_groups[start : start + batch_size]
            prepared = []
            locations: list[tuple[int, str]] = []
            for item_index, group in batch:
                labels = np.zeros((20,), dtype=np.int64)
                mask = np.zeros((20,), dtype=np.bool_)
                weights = np.zeros((20,), dtype=np.float32)
                for target in group.targets:
                    labels[target.edge_index] = target.label_index
                    mask[target.edge_index] = True
                    weights[target.edge_index] = target.component_weight
                arrays["tuev_labels"][item_index] = labels
                arrays["tuev_mask"][item_index] = mask
                arrays["tuev_weights"][item_index] = weights
                for arm in ARM_IDS:
                    interval = prepare_arm_interval(
                        raw,
                        arm_id=arm,
                        start_sec=group.start_sample / 200.0,
                        stop_sec=group.stop_sample / 200.0,
                        full_record=full[arm],
                    )
                    prepared.append(interval.data_volts)
                    locations.append((item_index, arm))
                    stat = _stats(interval.data_volts)
                    for key in ("count", "sum", "sum_squares"):
                        input_stats[arm][key] += stat[key]
                    input_stats[arm]["absolute_max"] = max(
                        input_stats[arm]["absolute_max"], stat["absolute_max"]
                    )
            tokens = _encode_four_second(
                encoder, np.stack(prepared), binding, device
            ).numpy()
            for row, (item_index, arm) in enumerate(locations):
                selected = tokens[row, :, :1]
                arrays[f"tuev_tokens_{arm}"][item_index] = selected
                stat = _stats(selected)
                for key in ("count", "sum", "sum_squares"):
                    token_stats[arm][key] += stat[key]
                token_stats[arm]["absolute_max"] = max(
                    token_stats[arm]["absolute_max"], stat["absolute_max"]
                )
        for array in arrays.values():
            array.flush()
        names = [f"tuev_tokens_{arm}" for arm in ARM_IDS] + [
            "tuev_labels",
            "tuev_mask",
            "tuev_weights",
        ]
        payload = {
            "schema_version": EXTRACTION_SCHEMA,
            "dataset": "TUEV",
            "record_id": record_id,
            "edf_sha256": record.edf_sha256,
            "duration_sec": raw.duration_sec,
            "item_indices": indices,
            "position_binding": binding.to_dict(),
            "feature_receipt": encoder.feature_receipt_for_record_binding(
                binding
            ).to_dict(),
            "input_stats_by_arm": input_stats,
            "token_stats_by_arm": token_stats,
            "array_slice_sha256": {
                name: _hash_array(np.asarray(arrays[name][indices]))
                for name in names
            },
        }
        _atomic_json(receipt_path, payload)
        receipts.append(payload)
        _log(
            "extract_record",
            dataset="TUEV",
            record_id=record_id,
            records_done=record_index + 1,
            records_total=len(groups_by_record),
            item_count=len(indices),
        )
    return tuple(receipts)


def _extract_tusz(
    *,
    root: Path,
    output: Path,
    events: Sequence[object],
    arrays: Mapping[str, np.ndarray],
    encoder: OfficialLaBraMEncoder,
    device: torch.device,
    event_batch_size: int,
) -> tuple[dict[str, object], ...]:
    events_by_record: dict[str, list[tuple[int, object]]] = defaultdict(list)
    for index, event in enumerate(events):
        events_by_record[event.record_id].append((index, event))
    receipts: list[dict[str, object]] = []
    for record_index, (record_id, indexed_events) in enumerate(
        sorted(events_by_record.items())
    ):
        indices = [index for index, _ in indexed_events]
        receipt_path = _record_receipt_path(output, "TUSZ", record_id)
        completed = _load_completed_receipt(
            receipt_path,
            dataset="TUSZ",
            record_id=record_id,
            indices=indices,
            arrays=arrays,
        )
        if completed is not None:
            receipts.append(completed)
            _log("extract_skip", dataset="TUSZ", record_id=record_id)
            continue
        first_event = indexed_events[0][1]
        source = parse_tusz_official_train_path(
            root, first_event.relative_edf_path
        )
        if _hash_file(source.edf_path) != first_event.edf_sha256:
            raise ValueError(f"TUSZ EDF changed: {record_id}")
        raw = read_physical_edf(source.edf_path, geometry="standard19")
        binding = bind_labram_record_positions(raw.channel_names)
        full = _prepare_full_arms(raw)
        input_stats = {arm: {"count": 0, "sum": 0.0, "sum_squares": 0.0, "absolute_max": 0.0} for arm in ARM_IDS}
        token_stats = {arm: {"count": 0, "sum": 0.0, "sum_squares": 0.0, "absolute_max": 0.0} for arm in ARM_IDS}
        for start in range(0, len(indexed_events), event_batch_size):
            batch = indexed_events[start : start + event_batch_size]
            prepared = []
            locations: list[tuple[int, str]] = []
            for item_index, event in batch:
                target = load_tusz_ictal_involvement_target(
                    source.channel_annotation_path,
                    source.global_annotation_path,
                    event_index=event.event_index,
                    source_path=source.edf_path,
                )
                targets = target.targets.cpu().numpy().astype(np.float32)
                mask = target.source_target_mask.cpu().numpy().astype(np.bool_)
                arrays["tusz_targets"][item_index] = targets
                arrays["tusz_mask"][item_index] = mask
                for arm in ARM_IDS:
                    interval = prepare_arm_interval(
                        raw,
                        arm_id=arm,
                        start_sec=event.event_t0_sec - 12.0,
                        stop_sec=event.event_t0_sec + 48.0,
                        full_record=full[arm],
                    )
                    prepared.append(interval.data_volts)
                    locations.append((item_index, arm))
                    stat = _stats(interval.data_volts)
                    for key in ("count", "sum", "sum_squares"):
                        input_stats[arm][key] += stat[key]
                    input_stats[arm]["absolute_max"] = max(
                        input_stats[arm]["absolute_max"], stat["absolute_max"]
                    )
            tokens = _encode_sixty_second(
                encoder, np.stack(prepared), binding, device
            ).numpy()
            for row, (item_index, arm) in enumerate(locations):
                selected = tokens[row]
                arrays[f"tusz_tokens_{arm}"][item_index] = selected
                stat = _stats(selected)
                for key in ("count", "sum", "sum_squares"):
                    token_stats[arm][key] += stat[key]
                token_stats[arm]["absolute_max"] = max(
                    token_stats[arm]["absolute_max"], stat["absolute_max"]
                )
        for array in arrays.values():
            array.flush()
        names = [f"tusz_tokens_{arm}" for arm in ARM_IDS] + [
            "tusz_targets",
            "tusz_mask",
        ]
        payload = {
            "schema_version": EXTRACTION_SCHEMA,
            "dataset": "TUSZ",
            "record_id": record_id,
            "edf_sha256": first_event.edf_sha256,
            "duration_sec": raw.duration_sec,
            "item_indices": indices,
            "position_binding": binding.to_dict(),
            "feature_receipt": encoder.feature_receipt_for_record_binding(
                binding
            ).to_dict(),
            "input_stats_by_arm": input_stats,
            "token_stats_by_arm": token_stats,
            "array_slice_sha256": {
                name: _hash_array(np.asarray(arrays[name][indices]))
                for name in names
            },
        }
        _atomic_json(receipt_path, payload)
        receipts.append(payload)
        _log(
            "extract_record",
            dataset="TUSZ",
            record_id=record_id,
            records_done=record_index + 1,
            records_total=len(events_by_record),
            item_count=len(indices),
        )
    return tuple(receipts)


def _set_determinism(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _checkpoint_paths(output: Path, dataset: str, arm: str, fold: int):
    directory = output / "nested-checkpoints" / dataset.lower() / arm / f"fold-{fold}"
    return directory / "model.safetensors", directory / "receipt.json"


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(_canonical_json({"dtype": str(value.dtype), "shape": list(value.shape)}))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _save_checkpoint(
    output: Path,
    *,
    dataset: str,
    arm: str,
    fold: int,
    head: torch.nn.Module,
    config: Mapping[str, object],
    fit_ids: Sequence[str],
    held_ids: Sequence[str],
) -> dict[str, object]:
    from safetensors.torch import save_file

    model_path, receipt_path = _checkpoint_paths(output, dataset, arm, fold)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in head.state_dict().items()
    }
    temporary = model_path.parent / f".{model_path.name}.tmp-{os.getpid()}"
    save_file(state, str(temporary))
    os.replace(temporary, model_path)
    payload = {
        "schema_version": "soz_preprocessing_parity_nested_checkpoint_v1",
        "dataset": dataset,
        "arm_id": arm,
        "fold": fold,
        "config": dict(config),
        "config_sha256": _hash_payload(config),
        "fit_ids": list(sorted(fit_ids)),
        "held_ids": list(sorted(held_ids)),
        "fit_roster_sha256": _hash_payload(tuple(sorted(fit_ids))),
        "held_roster_sha256": _hash_payload(tuple(sorted(held_ids))),
        "model_file": model_path.name,
        "model_file_sha256": _hash_file(model_path),
        "state_sha256": _state_sha256(state),
    }
    _atomic_json(receipt_path, payload)
    return payload


def _load_checkpoint_if_valid(
    output: Path,
    *,
    dataset: str,
    arm: str,
    fold: int,
    head: torch.nn.Module,
    config: Mapping[str, object],
    fit_ids: Sequence[str],
    held_ids: Sequence[str],
) -> dict[str, object] | None:
    from safetensors.torch import load_file

    model_path, receipt_path = _checkpoint_paths(output, dataset, arm, fold)
    if not model_path.exists() or not receipt_path.exists():
        return None
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    checks = {
        "dataset": dataset,
        "arm_id": arm,
        "fold": fold,
        "config_sha256": _hash_payload(config),
        "fit_roster_sha256": _hash_payload(tuple(sorted(fit_ids))),
        "held_roster_sha256": _hash_payload(tuple(sorted(held_ids))),
        "model_file_sha256": _hash_file(model_path),
    }
    if any(payload.get(key) != value for key, value in checks.items()):
        raise ValueError(f"Nested checkpoint binding changed: {receipt_path}")
    state = load_file(str(model_path), device="cpu")
    if _state_sha256(state) != payload.get("state_sha256"):
        raise ValueError("Nested checkpoint state SHA changed")
    head.load_state_dict(state, strict=True)
    return payload


def _confusion_f1(confusion: np.ndarray) -> float:
    matrix = np.asarray(confusion, dtype=np.float64)
    values = []
    for label in range(6):
        tp = matrix[label, label]
        fp = matrix[:, label].sum() - tp
        fn = matrix[label, :].sum() - tp
        denominator = 2.0 * tp + fp + fn
        values.append(0.0 if denominator <= 0 else float(2.0 * tp / denominator))
    return float(sum(values) / len(values))


def _morphology_class_weights(
    labels: np.ndarray,
    masks: np.ndarray,
    weights: np.ndarray,
    indices: Sequence[int],
) -> torch.Tensor:
    mass = np.zeros(6, dtype=np.float64)
    for index in indices:
        observed = masks[index]
        np.add.at(mass, labels[index][observed], weights[index][observed])
    if np.any(mass <= 0):
        raise ValueError(f"Morphology fit fold lacks CE6 class support: {mass.tolist()}")
    values = mass.sum() / (6.0 * mass)
    values /= values.mean()
    values = np.minimum(values, 10.0)
    values /= values.mean()
    return torch.tensor(values, dtype=torch.float32)


def _evaluate_morphology_indices(
    head: MorphologyEvidenceHead,
    tokens: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    weights: np.ndarray,
    indices_by_group: Mapping[str, Sequence[int]],
    device: torch.device,
) -> dict[str, np.ndarray]:
    head.eval()
    result: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for group_id, indices in sorted(indices_by_group.items()):
            confusion = np.zeros((6, 6), dtype=np.float64)
            for start in range(0, len(indices), 64):
                selection = list(indices[start : start + 64])
                batch = _torch_from_numpy_safe(tokens[selection]).to(device)
                prediction = head(batch).squeeze(2).argmax(dim=-1).cpu().numpy()
                for row, item_index in enumerate(selection):
                    observed = masks[item_index]
                    truth = labels[item_index][observed]
                    guessed = prediction[row][observed]
                    mass = weights[item_index][observed]
                    np.add.at(confusion, (truth, guessed), mass)
            if confusion.sum() <= 0:
                raise RuntimeError("Held morphology group has no native targets")
            result[group_id] = confusion / confusion.sum()
    return result


def _train_morphology_oof(
    *,
    output: Path,
    arm: str,
    items: Sequence[Mapping[str, object]],
    arrays: Mapping[str, np.ndarray],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], tuple[dict[str, object], ...]]:
    tokens = arrays[f"tuev_tokens_{arm}"]
    labels = arrays["tuev_labels"]
    masks = arrays["tuev_mask"]
    weights = arrays["tuev_weights"]
    group_indices: dict[str, list[int]] = defaultdict(list)
    group_fold: dict[str, int] = {}
    for item in items:
        group_id = str(item["parent_group_id"])
        group_indices[group_id].append(int(item["index"]))
        previous = group_fold.setdefault(group_id, int(item["fold"]))
        if previous != int(item["fold"]):
            raise ValueError("One TUEV group crosses nested folds")
    config = {
        "epochs": MORPHOLOGY_EPOCHS,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "crop_microbatch_size": 32,
        "class_weight_cap": 10.0,
        "seed": SEED,
        "loss": "group_equal_overlap_component_weighted_CE6",
        "checkpoint_selection": "fixed_final_epoch",
    }
    oof: dict[str, np.ndarray] = {}
    checkpoint_receipts = []
    for fold in range(FOLD_COUNT):
        fit_groups = tuple(sorted(key for key, value in group_fold.items() if value != fold))
        held_groups = tuple(sorted(key for key, value in group_fold.items() if value == fold))
        fit_indices = [index for group in fit_groups for index in group_indices[group]]
        head = MorphologyEvidenceHead().to(device)
        checkpoint = _load_checkpoint_if_valid(
            output,
            dataset="TUEV",
            arm=arm,
            fold=fold,
            head=head,
            config=config,
            fit_ids=fit_groups,
            held_ids=held_groups,
        )
        if checkpoint is None:
            _set_determinism(SEED + fold, device)
            head = MorphologyEvidenceHead().to(device)
            class_weights = _morphology_class_weights(
                labels, masks, weights, fit_indices
            ).to(device)
            optimizer = torch.optim.AdamW(
                head.parameters(), lr=1e-3, weight_decay=1e-4
            )
            generator = random.Random(SEED + fold)
            for epoch in range(MORPHOLOGY_EPOCHS):
                order = list(fit_groups)
                generator.shuffle(order)
                epoch_loss = 0.0
                head.train()
                for group_id in order:
                    indices = group_indices[group_id]
                    denominator = float(
                        sum(weights[index][masks[index]].sum() for index in indices)
                    )
                    optimizer.zero_grad(set_to_none=True)
                    group_loss = 0.0
                    for start in range(0, len(indices), 32):
                        selection = indices[start : start + 32]
                        batch_tokens = _torch_from_numpy_safe(
                            tokens[selection]
                        ).to(device)
                        batch_labels = _torch_from_numpy_safe(
                            labels[selection]
                        ).to(device)
                        batch_mask = _torch_from_numpy_safe(
                            masks[selection]
                        ).to(device)
                        batch_weights = _torch_from_numpy_safe(
                            weights[selection]
                        ).to(device)
                        logits = head(batch_tokens).squeeze(2)
                        safe = torch.where(batch_mask, batch_labels, 0)
                        element = F.cross_entropy(
                            logits.movedim(-1, 1),
                            safe,
                            weight=class_weights,
                            reduction="none",
                        )
                        loss = (element * batch_weights)[batch_mask].sum() / denominator
                        loss.backward()
                        group_loss += float(loss.detach().cpu())
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += group_loss
                _log(
                    "train_epoch",
                    dataset="TUEV",
                    arm_id=arm,
                    fold=fold,
                    epoch=epoch,
                    mean_group_loss=epoch_loss / len(fit_groups),
                )
            checkpoint = _save_checkpoint(
                output,
                dataset="TUEV",
                arm=arm,
                fold=fold,
                head=head,
                config=config,
                fit_ids=fit_groups,
                held_ids=held_groups,
            )
        else:
            head.to(device)
            _log("checkpoint_resume", dataset="TUEV", arm_id=arm, fold=fold)
        checkpoint_receipts.append(checkpoint)
        held = {group: group_indices[group] for group in held_groups}
        predictions = _evaluate_morphology_indices(
            head, tokens, labels, masks, weights, held, device
        )
        if set(oof) & set(predictions):
            raise RuntimeError("Morphology OOF groups were predicted more than once")
        oof.update(predictions)
    if set(oof) != set(group_indices):
        raise RuntimeError("Morphology OOF prediction roster is incomplete")
    return oof, tuple(checkpoint_receipts)


def _evaluate_ictal_patients(
    head: IctalInvolvementHead,
    tokens: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    indices_by_patient: Mapping[str, Sequence[int]],
    device: torch.device,
) -> dict[str, float]:
    head.eval()
    result: dict[str, float] = {}
    with torch.inference_mode():
        for patient, indices in sorted(indices_by_patient.items()):
            numerator = 0.0
            denominator = 0
            for start in range(0, len(indices), 4):
                selection = list(indices[start : start + 4])
                batch_tokens = _torch_from_numpy_safe(tokens[selection]).to(device)
                batch_targets = _torch_from_numpy_safe(targets[selection]).to(device)
                batch_mask = _torch_from_numpy_safe(masks[selection]).to(device)
                logits = head(batch_tokens).squeeze(-1)
                observed = int(batch_mask.sum().item())
                if observed:
                    numerator += float(
                        F.binary_cross_entropy_with_logits(
                            logits[batch_mask],
                            batch_targets[batch_mask],
                            reduction="sum",
                        ).cpu()
                    )
                    denominator += observed
            if denominator < 1:
                raise RuntimeError("Held TUSZ patient has no explicit native cells")
            result[patient] = numerator / denominator
    return result


def _train_ictal_oof(
    *,
    output: Path,
    arm: str,
    items: Sequence[Mapping[str, object]],
    arrays: Mapping[str, np.ndarray],
    device: torch.device,
) -> tuple[dict[str, float], tuple[dict[str, object], ...]]:
    tokens = arrays[f"tusz_tokens_{arm}"]
    targets = arrays["tusz_targets"]
    masks = arrays["tusz_mask"]
    patient_indices: dict[str, list[int]] = defaultdict(list)
    patient_fold: dict[str, int] = {}
    for item in items:
        patient = str(item["patient_id"])
        patient_indices[patient].append(int(item["index"]))
        previous = patient_fold.setdefault(patient, int(item["fold"]))
        if previous != int(item["fold"]):
            raise ValueError("One TUSZ patient crosses nested folds")
    config = {
        "epochs": ICTAL_EPOCHS,
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "gradient_clip_norm": 1.0,
        "event_microbatch_size": 4,
        "seed": SEED,
        "loss": "unweighted_patient_macro_masked_bce",
        "checkpoint_selection": "fixed_final_epoch",
    }
    oof: dict[str, float] = {}
    checkpoint_receipts = []
    for fold in range(FOLD_COUNT):
        fit_patients = tuple(sorted(key for key, value in patient_fold.items() if value != fold))
        held_patients = tuple(sorted(key for key, value in patient_fold.items() if value == fold))
        head = IctalInvolvementHead().to(device)
        checkpoint = _load_checkpoint_if_valid(
            output,
            dataset="TUSZ",
            arm=arm,
            fold=fold,
            head=head,
            config=config,
            fit_ids=fit_patients,
            held_ids=held_patients,
        )
        if checkpoint is None:
            _set_determinism(SEED + fold, device)
            head = IctalInvolvementHead().to(device)
            optimizer = torch.optim.AdamW(
                head.parameters(), lr=1e-3, weight_decay=1e-2
            )
            generator = random.Random(SEED + fold)
            for epoch in range(ICTAL_EPOCHS):
                order = list(fit_patients)
                generator.shuffle(order)
                epoch_loss = 0.0
                head.train()
                for patient in order:
                    indices = patient_indices[patient]
                    patient_observed = int(sum(masks[index].sum() for index in indices))
                    if patient_observed < 1:
                        raise RuntimeError("Fit patient has no observed ictal labels")
                    optimizer.zero_grad(set_to_none=True)
                    patient_loss = 0.0
                    for start in range(0, len(indices), 4):
                        selection = indices[start : start + 4]
                        batch_tokens = _torch_from_numpy_safe(
                            tokens[selection]
                        ).to(device)
                        batch_targets = _torch_from_numpy_safe(
                            targets[selection]
                        ).to(device)
                        batch_mask = _torch_from_numpy_safe(
                            masks[selection]
                        ).to(device)
                        logits = head(batch_tokens).squeeze(-1)
                        loss = F.binary_cross_entropy_with_logits(
                            logits[batch_mask],
                            batch_targets[batch_mask],
                            reduction="sum",
                        ) / patient_observed
                        loss.backward()
                        patient_loss += float(loss.detach().cpu())
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += patient_loss
                _log(
                    "train_epoch",
                    dataset="TUSZ",
                    arm_id=arm,
                    fold=fold,
                    epoch=epoch,
                    mean_patient_loss=epoch_loss / len(fit_patients),
                )
            checkpoint = _save_checkpoint(
                output,
                dataset="TUSZ",
                arm=arm,
                fold=fold,
                head=head,
                config=config,
                fit_ids=fit_patients,
                held_ids=held_patients,
            )
        else:
            head.to(device)
            _log("checkpoint_resume", dataset="TUSZ", arm_id=arm, fold=fold)
        checkpoint_receipts.append(checkpoint)
        held = {patient: patient_indices[patient] for patient in held_patients}
        predictions = _evaluate_ictal_patients(
            head, tokens, targets, masks, held, device
        )
        if set(oof) & set(predictions):
            raise RuntimeError("Ictal OOF patients were predicted more than once")
        oof.update(predictions)
    if set(oof) != set(patient_indices):
        raise RuntimeError("Ictal OOF prediction roster is incomplete")
    return oof, tuple(checkpoint_receipts)


def _jitter_paths(output: Path, dataset: str) -> tuple[Path, Path]:
    directory = output / "jitter"
    return directory / f"{dataset.lower()}_tokens.npy", directory / f"{dataset.lower()}_receipt.json"


def _select_jitter_items(
    items: Sequence[Mapping[str, object]],
    *,
    dataset: str,
    identity_field: str,
    duration_by_record: Mapping[str, float],
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Choose the paired v2 robustness denominator without changing labels.

    TUEV needs three seconds of pre-event context so the same annotated
    one-second event can occupy slots 0--3 of a four-second encoder call.
    TUSZ shifts the full 60-second event-anchored window by one second in
    either direction.  The 30-second lower bound preserves the causal arm's
    real warm-up contract in every state.
    """

    if dataset not in {"TUEV", "TUSZ"}:
        raise ValueError("jitter dataset must be TUEV or TUSZ")
    by_identity: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for item in items:
        by_identity[str(item[identity_field])].append(item)
    selected = []
    omitted_identity_ids = []
    for identity, candidates in sorted(by_identity.items()):
        pre_shift = 3.0 if dataset == "TUEV" else 1.0
        post_shift = 0.0 if dataset == "TUEV" else 1.0
        eligible = [
            item
            for item in candidates
            if float(item["start_sec"]) - pre_shift >= 30.0
            and float(item["stop_sec"]) + post_shift
            <= duration_by_record[str(item["record_id"])] + 1e-9
        ]
        if not eligible:
            omitted_identity_ids.append(identity)
            continue
        chosen = min(eligible, key=lambda item: int(item["index"]))
        selected.append(dict(chosen))
    if not selected:
        raise ValueError(f"No {dataset} identity has complete v2 robustness context")
    selected_identity_ids = tuple(str(item[identity_field]) for item in selected)
    if selected_identity_ids != tuple(sorted(set(selected_identity_ids))):
        raise RuntimeError("Robustness selection identity roster is not canonical")
    receipt = {
        "schema_version": "soz_preprocessing_robustness_selection_v2",
        "dataset": dataset,
        "identity_field": identity_field,
        "source_identity_ids": sorted(by_identity),
        "selected_identity_ids": list(selected_identity_ids),
        "omitted_identity_ids": omitted_identity_ids,
        "source_identity_count": len(by_identity),
        "selected_identity_count": len(selected),
        "omitted_identity_count": len(omitted_identity_ids),
        "eligibility_policy": (
            "complete_real_context_for_all_label_preserving_states_and_30s_causal_warmup"
        ),
        "arm_specific_selection_forbidden": True,
    }
    _log(
        "robustness_eligibility",
        dataset=dataset,
        source_identity_count=len(by_identity),
        selected_identity_count=len(selected),
        omitted_identity_count=len(omitted_identity_ids),
    )
    return tuple(selected), receipt


def _materialize_jitter_tokens(
    *,
    dataset: str,
    root: Path,
    output: Path,
    selected_items: Sequence[Mapping[str, object]],
    selection_receipt: Mapping[str, object],
    source_objects: Sequence[object],
    tuev_records: Mapping[str, object] | None,
    encoder: OfficialLaBraMEncoder,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    if dataset not in {"TUEV", "TUSZ"}:
        raise ValueError("jitter dataset must be TUEV or TUSZ")
    tensor_path, receipt_path = _jitter_paths(output, dataset)
    if dataset == "TUEV":
        # Each state contains only the contextualized token at the slot that
        # corresponds to the unchanged one-second native event label.
        state_values = (0, 1, 2, 3)
        expected_shape = (len(selected_items), 4, len(ARM_IDS), 19, 1, 200)
    else:
        state_values = (-1, 1)
        expected_shape = (len(selected_items), 2, len(ARM_IDS), 19, 60, 200)
    selection_payload = [
        {
            "index": int(item["index"]),
            "record_id": str(item["record_id"]),
            "identity": str(
                item["parent_group_id"] if dataset == "TUEV" else item["patient_id"]
            ),
            "fold": int(item["fold"]),
        }
        for item in selected_items
    ]
    if (
        selection_receipt.get("dataset") != dataset
        or int(selection_receipt.get("selected_identity_count", -1))
        != len(selected_items)
        or selection_receipt.get("arm_specific_selection_forbidden") is not True
    ):
        raise ValueError("Robustness selection receipt disagrees with selected items")
    selection_receipt_sha256 = _hash_payload(selection_receipt)
    if tensor_path.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema_version")
            != "soz_preprocessing_parity_robustness_corpus_v2"
            or receipt.get("dataset") != dataset
            or receipt.get("shape") != list(expected_shape)
            or receipt.get("selection_sha256") != _hash_payload(selection_payload)
            or receipt.get("selection_receipt_sha256") != selection_receipt_sha256
            or receipt.get("tensor_file_sha256") != _hash_file(tensor_path)
        ):
            raise ValueError(f"Existing {dataset} jitter corpus changed")
        tensor = np.load(tensor_path, mmap_mode="r")
        if tensor.shape != expected_shape or _hash_array(tensor) != receipt.get(
            "tensor_sha256"
        ):
            raise ValueError(f"Existing {dataset} jitter tensor changed")
        _log("jitter_resume", dataset=dataset, item_count=len(selected_items))
        return tensor, receipt

    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = np.lib.format.open_memmap(
        tensor_path, mode="w+", dtype=np.float32, shape=expected_shape
    )
    object_by_index = {index: value for index, value in enumerate(source_objects)}
    selected_by_record: dict[str, list[tuple[int, Mapping[str, object]]]] = defaultdict(list)
    for row, item in enumerate(selected_items):
        selected_by_record[str(item["record_id"])].append((row, item))
    for record_number, (record_id, rows) in enumerate(sorted(selected_by_record.items())):
        first_object = object_by_index[int(rows[0][1]["index"])]
        if str(first_object.record_id) != record_id:
            raise ValueError(f"{dataset} jitter item/source index binding changed")
        if dataset == "TUEV":
            if tuev_records is None or record_id not in tuev_records:
                raise ValueError("TUEV jitter lacks its source-record receipt")
            source_record = tuev_records[record_id]
            source = (root / source_record.relative_edf_path).resolve(strict=True)
            expected_edf_sha = source_record.edf_sha256
        else:
            if tuev_records is not None:
                raise ValueError("TUSZ jitter must not receive TUEV source records")
            parsed = parse_tusz_official_train_path(
                root, first_object.relative_edf_path
            )
            source = parsed.edf_path
            expected_edf_sha = first_object.edf_sha256
        if _hash_file(source) != expected_edf_sha:
            raise ValueError(f"{dataset} jitter EDF changed: {record_id}")
        raw = read_physical_edf(source, geometry="standard19")
        binding = bind_labram_record_positions(raw.channel_names)
        full = _prepare_full_arms(raw)
        prepared = []
        locations: list[tuple[int, int, int, int]] = []
        for row, item in rows:
            for state_index, state_value in enumerate(state_values):
                for arm_index, arm in enumerate(ARM_IDS):
                    if dataset == "TUEV":
                        slot = int(state_value)
                        interval_start = float(item["start_sec"]) - slot
                        interval_stop = interval_start + 4.0
                    else:
                        slot = -1
                        interval_start = float(item["start_sec"]) + int(state_value)
                        interval_stop = float(item["stop_sec"]) + int(state_value)
                    interval = prepare_arm_interval(
                        raw,
                        arm_id=arm,
                        start_sec=interval_start,
                        stop_sec=interval_stop,
                        full_record=full[arm],
                    )
                    prepared.append(interval.data_volts)
                    locations.append((row, state_index, arm_index, slot))
        if dataset == "TUEV":
            encoded = _encode_four_second(encoder, np.stack(prepared), binding, device).numpy()
        else:
            encoded = _encode_sixty_second(
                encoder, np.stack(prepared), binding, device
            ).numpy()
        for values, location in zip(encoded, locations):
            row, state_index, arm_index, slot = location
            tensor[row, state_index, arm_index] = (
                values[:, slot : slot + 1] if dataset == "TUEV" else values
            )
        tensor.flush()
        _log(
            "jitter_record",
            dataset=dataset,
            record_id=record_id,
            records_done=record_number + 1,
            records_total=len(selected_by_record),
        )
    receipt = {
        "schema_version": "soz_preprocessing_parity_robustness_corpus_v2",
        "dataset": dataset,
        "label_preserving": True,
        "states": (
            ["event_at_slot_0", "event_at_slot_1", "event_at_slot_2", "event_at_slot_3"]
            if dataset == "TUEV"
            else ["window_minus_1_sec", "window_plus_1_sec"]
        ),
        "signal_grid_offsets_seconds": (
            [0.0, -1.0, -2.0, -3.0] if dataset == "TUEV" else [-1.0, 1.0]
        ),
        "target_projection": (
            "same_native_event_evaluated_at_matching_slot"
            if dataset == "TUEV"
            else "integer_bin_reindex_with_out_of_range_boundary_masked"
        ),
        "selection": selection_payload,
        "selection_sha256": _hash_payload(selection_payload),
        "selection_receipt": dict(selection_receipt),
        "selection_receipt_sha256": selection_receipt_sha256,
        "shape": list(expected_shape),
        "dtype": "float32",
        "tensor_file": tensor_path.name,
        "tensor_file_sha256": _hash_file(tensor_path),
        "tensor_sha256": _hash_array(tensor),
    }
    _atomic_json(receipt_path, receipt)
    return tensor, receipt


def _load_fold_head(
    output: Path,
    *,
    dataset: str,
    arm: str,
    fold: int,
    device: torch.device,
):
    from safetensors.torch import load_file

    model_path, receipt_path = _checkpoint_paths(output, dataset, arm, fold)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("model_file_sha256") != _hash_file(model_path):
        raise ValueError("Jitter evaluation checkpoint bytes changed")
    head = (
        MorphologyEvidenceHead() if dataset == "TUEV" else IctalInvolvementHead()
    ).to(device)
    state = load_file(str(model_path), device="cpu")
    if _state_sha256(state) != receipt.get("state_sha256"):
        raise ValueError("Jitter evaluation checkpoint state changed")
    head.load_state_dict(state, strict=True)
    head.eval()
    return head


def _project_tusz_integer_shift(
    targets: np.ndarray, masks: np.ndarray, *, shift_seconds: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reindex native one-second labels onto an integer-shifted signal grid."""

    if shift_seconds not in {-1, 1}:
        raise ValueError("TUSZ v2 robustness supports only -1/+1 second shifts")
    target = np.asarray(targets)
    mask = np.asarray(masks)
    if target.shape != (20, 60) or mask.shape != (20, 60):
        raise ValueError("TUSZ projection expects [20,60] target and mask")
    projected_target = np.zeros_like(target)
    projected_mask = np.zeros_like(mask)
    if shift_seconds == -1:
        # New bin j spans old bin j-1. New bin zero has no source label in
        # the fixed native [-12,+48) target and therefore stays masked.
        projected_target[:, 1:] = target[:, :-1]
        projected_mask[:, 1:] = mask[:, :-1]
    else:
        # New bin j spans old bin j+1. The final new bin is out of range.
        projected_target[:, :-1] = target[:, 1:]
        projected_mask[:, :-1] = mask[:, 1:]
    projected_target[~projected_mask] = 0
    return projected_target, projected_mask


def _evaluate_jitter(
    *,
    checkpoint_output: Path,
    arm: str,
    tuev_selected: Sequence[Mapping[str, object]],
    tuev_jitter: np.ndarray,
    tusz_selected: Sequence[Mapping[str, object]],
    tusz_jitter: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    device: torch.device,
) -> tuple[float, float, dict[str, object]]:
    """Evaluate the label-preserving v2 robustness states."""

    arm_index = ARM_IDS.index(arm)
    nominal_reference = np.stack(
        [
            np.asarray(
                arrays[f"tuev_tokens_{arm}"][int(item["index"])]
            )
            for item in tuev_selected
        ]
    )
    nominal_replay = np.asarray(tuev_jitter[:, 0, arm_index])
    replay_difference = nominal_replay.astype(np.float64) - nominal_reference.astype(
        np.float64
    )
    nominal_norm = float(np.linalg.norm(nominal_reference.astype(np.float64)))
    replay_norm = float(np.linalg.norm(nominal_replay.astype(np.float64)))
    replay_relative_l2 = float(np.linalg.norm(replay_difference) / nominal_norm)
    replay_cosine_distance = float(
        1.0
        - np.sum(
            nominal_replay.astype(np.float64)
            * nominal_reference.astype(np.float64)
        )
        / (replay_norm * nominal_norm)
    )
    replay_max_absolute_difference = float(np.abs(replay_difference).max())
    morphology_scores = []
    morphology_by_fold: dict[int, list[int]] = defaultdict(list)
    for row, item in enumerate(tuev_selected):
        morphology_by_fold[int(item["fold"])].append(row)
    for state_index in range(4):
        morphology_confusions: dict[str, np.ndarray] = {}
        for fold, rows in sorted(morphology_by_fold.items()):
            head = _load_fold_head(
                checkpoint_output,
                dataset="TUEV",
                arm=arm,
                fold=fold,
                device=device,
            )
            with torch.inference_mode():
                for start in range(0, len(rows), 64):
                    selection = rows[start : start + 64]
                    if state_index == 0:
                        # Score the exact frozen nominal corpus. The separately
                        # recomputed slot-0 replay is a receipted numerical
                        # diagnostic because GPU batch shape can change float32
                        # reduction order without changing signal semantics.
                        batch_values = nominal_reference[selection]
                    else:
                        batch_values = np.asarray(
                            tuev_jitter[selection, state_index, arm_index]
                        )
                    batch = _torch_from_numpy_safe(batch_values).to(device)
                    predictions = head(batch).squeeze(2).argmax(-1).cpu().numpy()
                    for local, row in enumerate(selection):
                        group_id = str(tuev_selected[row]["parent_group_id"])
                        confusion = morphology_confusions.setdefault(
                            group_id, np.zeros((6, 6), dtype=np.float64)
                        )
                        item_index = int(tuev_selected[row]["index"])
                        observed = arrays["tuev_mask"][item_index]
                        truth = arrays["tuev_labels"][item_index][observed]
                        guessed = predictions[local][observed]
                        mass = arrays["tuev_weights"][item_index][observed]
                        np.add.at(confusion, (truth, guessed), mass)
        if not morphology_confusions or any(
            confusion.sum() <= 0 for confusion in morphology_confusions.values()
        ):
            raise RuntimeError("TUEV slot-shift denominator is incomplete")
        group_equal_confusion = np.stack(
            [
                confusion / confusion.sum()
                for _, confusion in sorted(morphology_confusions.items())
            ]
        ).mean(axis=0)
        morphology_scores.append(_confusion_f1(group_equal_confusion))

    ictal_scores = []
    ictal_by_fold: dict[int, list[int]] = defaultdict(list)
    for row, item in enumerate(tusz_selected):
        ictal_by_fold[int(item["fold"])].append(row)
    for state_index, shift_seconds in enumerate((0, -1, 1)):
        patient_bces = []
        for fold, rows in sorted(ictal_by_fold.items()):
            head = _load_fold_head(
                checkpoint_output,
                dataset="TUSZ",
                arm=arm,
                fold=fold,
                device=device,
            )
            with torch.inference_mode():
                for row in rows:
                    item_index = int(tusz_selected[row]["index"])
                    if state_index == 0:
                        token_values = np.asarray(
                            arrays[f"tusz_tokens_{arm}"][item_index : item_index + 1]
                        )
                        target_values = np.asarray(
                            arrays["tusz_targets"][item_index]
                        )
                        mask_values = np.asarray(arrays["tusz_mask"][item_index])
                    else:
                        token_values = np.asarray(
                            tusz_jitter[
                                row,
                                state_index - 1,
                                arm_index : arm_index + 1,
                            ]
                        )
                        target_values, mask_values = _project_tusz_integer_shift(
                            arrays["tusz_targets"][item_index],
                            arrays["tusz_mask"][item_index],
                            shift_seconds=shift_seconds,
                        )
                    token = _torch_from_numpy_safe(token_values).to(device)
                    target = _torch_from_numpy_safe(target_values).to(device)
                    mask = _torch_from_numpy_safe(mask_values).to(device)
                    if not bool(mask.any()):
                        raise RuntimeError("Shifted TUSZ item has no projected labels")
                    logits = head(token).squeeze(0).squeeze(-1)
                    patient_bces.append(
                        float(
                            F.binary_cross_entropy_with_logits(
                                logits[mask], target[mask], reduction="mean"
                            ).cpu()
                        )
                    )
        if not patient_bces:
            raise RuntimeError("TUSZ onset-shift denominator is incomplete")
        ictal_scores.append(float(np.mean(patient_bces)))

    morphology_nominal = morphology_scores[0]
    ictal_nominal = ictal_scores[0]
    morphology_drop = max(
        0.0, max(morphology_nominal - score for score in morphology_scores[1:])
    )
    ictal_increase = max(
        0.0, max(score - ictal_nominal for score in ictal_scores[1:])
    )
    payload = {
        "schema_version": "soz_preprocessing_parity_robustness_evaluation_v2",
        "arm_id": arm,
        "label_preserving": True,
        "tuev_states": [
            "event_at_slot_0",
            "event_at_slot_1",
            "event_at_slot_2",
            "event_at_slot_3",
        ],
        "tusz_states": ["nominal", "window_minus_1_sec", "window_plus_1_sec"],
        "tuev_macro_f1_by_state": morphology_scores,
        "tusz_macro_bce_by_state": ictal_scores,
        "tuev_macro_f1_max_drop": morphology_drop,
        "tusz_macro_bce_max_increase": ictal_increase,
        "tusz_target_projection": (
            "integer_bin_reindex_with_out_of_range_boundary_masked"
        ),
        "tuev_slot0_numerical_replay": {
            "role": "diagnostic_not_selection_endpoint",
            "scored_nominal_source": "exact_frozen_nominal_token_corpus",
            "relative_l2_error": replay_relative_l2,
            "cosine_distance": replay_cosine_distance,
            "maximum_absolute_difference": replay_max_absolute_difference,
        },
    }
    return morphology_drop, ictal_increase, payload


def _bootstrap_metrics(
    morphology_by_arm: Mapping[str, Mapping[str, np.ndarray]],
    ictal_by_arm: Mapping[str, Mapping[str, float]],
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, object],
]:
    morphology_ids = tuple(sorted(next(iter(morphology_by_arm.values()))))
    ictal_ids = tuple(sorted(next(iter(ictal_by_arm.values()))))
    if any(tuple(sorted(values)) != morphology_ids for values in morphology_by_arm.values()):
        raise ValueError("Morphology arms have different paired group denominators")
    if any(tuple(sorted(values)) != ictal_ids for values in ictal_by_arm.values()):
        raise ValueError("Ictal arms have different paired patient denominators")
    morphology_arrays = {
        arm: np.stack([morphology_by_arm[arm][key] for key in morphology_ids])
        for arm in ARM_IDS
    }
    ictal_arrays = {
        arm: np.asarray([ictal_by_arm[arm][key] for key in ictal_ids], dtype=np.float64)
        for arm in ARM_IDS
    }
    morphology_points = {
        arm: _confusion_f1(values.mean(axis=0))
        for arm, values in morphology_arrays.items()
    }
    ictal_points = {
        arm: float(values.mean()) for arm, values in ictal_arrays.items()
    }
    rng = np.random.default_rng(SEED)
    morphology_differences = {
        arm: {reference: [] for reference in ARM_IDS} for arm in ARM_IDS
    }
    ictal_differences = {
        arm: {reference: [] for reference in ARM_IDS} for arm in ARM_IDS
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        morphology_sample = rng.integers(0, len(morphology_ids), len(morphology_ids))
        ictal_sample = rng.integers(0, len(ictal_ids), len(ictal_ids))
        sampled_f1 = {
            arm: _confusion_f1(values[morphology_sample].mean(axis=0))
            for arm, values in morphology_arrays.items()
        }
        sampled_bce = {
            arm: float(values[ictal_sample].mean())
            for arm, values in ictal_arrays.items()
        }
        for arm in ARM_IDS:
            for reference in ARM_IDS:
                morphology_differences[arm][reference].append(
                    sampled_f1[arm] - sampled_f1[reference]
                )
                ictal_differences[arm][reference].append(
                    sampled_bce[arm] - sampled_bce[reference]
                )
    morphology_lower = {
        arm: {
            reference: (
                0.0
                if arm == reference
                else float(
                    np.quantile(
                        morphology_differences[arm][reference], 0.025, method="linear"
                    )
                )
            )
            for reference in ARM_IDS
        }
        for arm in ARM_IDS
    }
    ictal_upper = {
        arm: {
            reference: (
                0.0
                if arm == reference
                else float(
                    np.quantile(
                        ictal_differences[arm][reference], 0.975, method="linear"
                    )
                )
            )
            for reference in ARM_IDS
        }
        for arm in ARM_IDS
    }
    receipt = {
        "schema_version": "soz_preprocessing_parity_paired_bootstrap_v1",
        "seed": SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "confidence_level": 0.95,
        "tuev_group_ids": list(morphology_ids),
        "tusz_patient_ids": list(ictal_ids),
        "tuev_point_macro_f1": morphology_points,
        "tusz_point_macro_bce": ictal_points,
        "tuev_difference_lower95": morphology_lower,
        "tusz_difference_upper95": ictal_upper,
    }
    return (
        morphology_points,
        ictal_points,
        morphology_lower,
        ictal_upper,
        receipt,
    )


def _arm_id_probe(
    arrays: Mapping[str, np.ndarray],
    items: Sequence[Mapping[str, object]],
) -> tuple[float, dict[str, object]]:
    first_by_patient: dict[str, Mapping[str, object]] = {}
    for item in items:
        first_by_patient.setdefault(str(item["patient_id"]), item)
    features = []
    labels = []
    folds = []
    patients = []
    for patient, item in sorted(first_by_patient.items()):
        index = int(item["index"])
        for arm_index, arm in enumerate(ARM_IDS):
            token = np.asarray(arrays[f"tusz_tokens_{arm}"][index], dtype=np.float64)
            features.append(
                np.concatenate(
                    (
                        token.mean(axis=(1, 2)),
                        token.std(axis=(1, 2)),
                        np.sqrt(np.square(token).mean(axis=(1, 2))),
                    )
                )
            )
            labels.append(arm_index)
            folds.append(int(item["fold"]))
            patients.append(patient)
    x = torch.tensor(np.stack(features), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    fold_tensor = torch.tensor(folds, dtype=torch.long)
    train = fold_tensor != 4
    test = fold_tensor == 4
    train_patients = {patient for patient, fold in zip(patients, folds) if fold != 4}
    test_patients = {patient for patient, fold in zip(patients, folds) if fold == 4}
    if not train.any() or not test.any() or train_patients & test_patients:
        raise ValueError("Arm-ID probe patient firewall failed")
    mean = x[train].mean(0)
    scale = x[train].std(0).clamp_min(1e-6)
    x = (x - mean) / scale
    _set_determinism(SEED, torch.device("cpu"))
    probe = torch.nn.Linear(x.shape[1], len(ARM_IDS))
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(x[train]), y[train])
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        prediction = probe(x[test]).argmax(-1)
    recalls = [
        float((prediction[y[test] == label] == label).float().mean())
        for label in range(len(ARM_IDS))
    ]
    balanced = float(sum(recalls) / len(recalls))
    receipt = {
        "schema_version": "soz_preprocessing_parity_arm_id_probe_v1",
        "feature": "per-channel token mean,std,rms from one event per patient",
        "model": "linear_softmax",
        "train_folds": [0, 1, 2, 3],
        "test_fold": 4,
        "patient_disjoint": True,
        "train_patient_count": len(train_patients),
        "test_patient_count": len(test_patients),
        "balanced_accuracy": balanced,
        "per_arm_recall": dict(zip(ARM_IDS, recalls)),
    }
    return balanced, receipt


def _aggregate_distribution_receipts(
    record_receipts: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    def aggregate(field: str):
        result = {}
        for arm in ARM_IDS:
            count = 0
            total = 0.0
            total_squares = 0.0
            absolute_max = 0.0
            for receipt in record_receipts:
                stats = receipt[field][arm]
                count += int(stats["count"])
                total += float(stats["sum"])
                total_squares += float(stats["sum_squares"])
                absolute_max = max(absolute_max, float(stats["absolute_max"]))
            mean = total / count
            variance = max(0.0, total_squares / count - mean * mean)
            result[arm] = {
                "count": count,
                "mean": mean,
                "std": math.sqrt(variance),
                "rms": math.sqrt(total_squares / count),
                "absolute_max": absolute_max,
            }
        return result

    input_payload = {
        "schema_version": "soz_preprocessing_parity_input_distribution_v1",
        "statistics_by_arm": aggregate("input_stats_by_arm"),
        "record_count": len(record_receipts),
    }
    token_payload = {
        "schema_version": "soz_preprocessing_parity_token_distribution_v1",
        "statistics_by_arm": aggregate("token_stats_by_arm"),
        "record_count": len(record_receipts),
    }
    return input_payload, token_payload


def _official_sanity(
    *,
    tuev_root: Path,
    tuev_records: Mapping[str, object],
    tuev_groups: Sequence[object],
    modeling: Path,
    checkpoint: Path,
    device: torch.device,
) -> dict[str, object]:
    group = next(
        value
        for value in tuev_groups
        if value.targets and value.targets[0].start_sample >= 3 * 200
    )
    record = tuev_records[group.record_id]
    source = (tuev_root / record.relative_edf_path).resolve(strict=True)
    if _hash_file(source) != record.edf_sha256:
        raise ValueError("O-REF sanity EDF changed")
    raw = read_physical_edf(source, geometry="official_ref23")
    full_a = prepare_full_record_arm(raw, arm_id="O-REF")
    full_b = prepare_full_record_arm(raw, arm_id="O-REF")
    target = group.targets[0]
    start = target.start_sample / 200.0 - 2.0
    stop = target.stop_sample / 200.0 + 2.0
    interval_a = prepare_arm_interval(
        raw, arm_id="O-REF", start_sec=start, stop_sec=stop, full_record=full_a
    )
    interval_b = prepare_arm_interval(
        raw, arm_id="O-REF", start_sec=start, stop_sec=stop, full_record=full_b
    )
    denominator = np.linalg.norm(interval_b.data_volts.astype(np.float64))
    signal_error = float(
        np.linalg.norm(
            interval_a.data_volts.astype(np.float64)
            - interval_b.data_volts.astype(np.float64)
        )
        / max(denominator, np.finfo(np.float64).tiny)
    )
    encoder = OfficialReference23LaBraMEncoder(
        modeling_path=modeling, checkpoint_path=checkpoint
    ).to(device)
    tensor_a = torch.from_numpy(interval_a.data_volts).reshape(1, 23, 5, 200).to(device)
    tensor_b = torch.from_numpy(interval_b.data_volts).reshape(1, 23, 5, 200).to(device)
    with torch.inference_mode():
        token_a = encoder(tensor_a).double().reshape(-1)
        token_b = encoder(tensor_b).double().reshape(-1)
    cosine_distance = float(
        1.0 - F.cosine_similarity(token_a[None], token_b[None]).item()
    )
    return {
        "schema_version": "soz_preprocessing_parity_official_exact_sanity_v1",
        "role": "independent_replay_of_frozen_official_geometry",
        "record_id": record.record_id,
        "crop_id": group.crop_id,
        "channels": list(OFFICIAL_REF23_CHANNELS),
        "signal_relative_l2_error": signal_error,
        "token_cosine_distance": max(0.0, cosine_distance),
        "finite": bool(torch.isfinite(token_a).all() and torch.isfinite(token_b).all()),
    }


def _publish_extraction_receipt(
    output: Path,
    paths: Mapping[str, Path],
    tuev_receipts: Sequence[Mapping[str, object]],
    tusz_receipts: Sequence[Mapping[str, object]],
    plan_sha256: str,
) -> dict[str, object]:
    target = output / "extraction-receipt.json"
    payload = {
        "schema_version": "soz_preprocessing_parity_paired_extraction_v1",
        "formal": True,
        "paired_arms": list(ARM_IDS),
        "plan_sha256": plan_sha256,
        "tuev_record_receipt_sha256s": [
            _hash_payload(value) for value in tuev_receipts
        ],
        "tusz_record_receipt_sha256s": [
            _hash_payload(value) for value in tusz_receipts
        ],
        "array_files": {
            name: {
                "path": str(path.relative_to(output)),
                "sha256": _hash_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in sorted(paths.items())
        },
    }
    payload["receipt_sha256"] = _hash_payload(payload)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("Formal paired extraction receipt changed")
    else:
        _atomic_json(target, payload)
    return payload


def _finalize_selection(
    *,
    output: Path,
    nested,
    protocol: PreprocessingParityProtocolReceipt,
    arrays: Mapping[str, np.ndarray],
    tuev_items: Sequence[Mapping[str, object]],
    tusz_items: Sequence[Mapping[str, object]],
    morphology_by_arm: Mapping[str, Mapping[str, np.ndarray]],
    ictal_by_arm: Mapping[str, Mapping[str, float]],
    jitter_by_arm: Mapping[str, tuple[float, float, Mapping[str, object]]],
    input_distribution: Mapping[str, object],
    token_distribution: Mapping[str, object],
    probe_accuracy: float,
    probe_receipt: Mapping[str, object],
    official: Mapping[str, object],
    extraction_receipt: Mapping[str, object],
    checkpoint_receipts: Mapping[str, object],
    prior_nominal_result: Mapping[str, object] | None = None,
) -> dict[str, object]:
    nominal_reuse_receipt = None
    if prior_nominal_result is None:
        (
            morphology_points,
            ictal_points,
            morphology_lower,
            ictal_upper,
            bootstrap_receipt,
        ) = _bootstrap_metrics(morphology_by_arm, ictal_by_arm)
        tuev_group_ids = tuple(sorted(next(iter(morphology_by_arm.values()))))
        tusz_patient_ids = tuple(sorted(next(iter(ictal_by_arm.values()))))
        denominator_payload = {
            "schema_version": "soz_preprocessing_parity_paired_denominator_v1",
            "tuev_group_ids": list(tuev_group_ids),
            "tusz_patient_ids": list(tusz_patient_ids),
            "tuev_content_component_ids": sorted(
                {
                    record.content_component_id
                    for record in nested.records
                    if record.dataset_id == "TUEV" and record.common_raw_qc_eligible
                }
            ),
            "tusz_explicit_cell_count": int(arrays["tusz_mask"].sum()),
            "paired_attrition_count": 0,
        }
    else:
        old_metrics = prior_nominal_result.get("metrics_by_arm")
        if not isinstance(old_metrics, Mapping) or tuple(sorted(old_metrics)) != tuple(
            sorted(PREPROCESSING_ARM_IDS)
        ):
            raise ValueError("Reusable nominal result lacks the five arm metrics")
        denominator_payload = prior_nominal_result.get("denominator_receipt")
        bootstrap_receipt = prior_nominal_result.get("bootstrap_receipt")
        if not isinstance(denominator_payload, Mapping) or not isinstance(
            bootstrap_receipt, Mapping
        ):
            raise ValueError("Reusable nominal result lacks paired receipts")
        denominator_payload = dict(denominator_payload)
        bootstrap_receipt = dict(bootstrap_receipt)
        tuev_group_ids = tuple(str(value) for value in denominator_payload["tuev_group_ids"])
        tusz_patient_ids = tuple(str(value) for value in denominator_payload["tusz_patient_ids"])
        if tuev_group_ids != tuple(sorted(set(tuev_group_ids))) or tusz_patient_ids != tuple(
            sorted(set(tusz_patient_ids))
        ):
            raise ValueError("Reusable nominal paired denominator is not canonical")
        expected_components = sorted(
            {
                record.content_component_id
                for record in nested.records
                if record.dataset_id == "TUEV" and record.common_raw_qc_eligible
            }
        )
        if denominator_payload.get("tuev_content_component_ids") != expected_components:
            raise ValueError("Reusable nominal TUEV component denominator changed")
        if int(denominator_payload.get("tusz_explicit_cell_count", -1)) != int(
            arrays["tusz_mask"].sum()
        ):
            raise ValueError("Reusable nominal TUSZ explicit-cell denominator changed")
        if int(denominator_payload.get("paired_attrition_count", -1)) != 0:
            raise ValueError("Reusable nominal result has paired attrition")
        morphology_points = {
            arm: float(old_metrics[arm]["tuev_macro_ce6_f1"]) for arm in ARM_IDS
        }
        ictal_points = {
            arm: float(old_metrics[arm]["tusz_native_macro_bce"]) for arm in ARM_IDS
        }
        morphology_lower = {
            arm: {
                str(reference): float(value)
                for reference, value in old_metrics[arm][
                    "tuev_f1_difference_lower95_by_reference"
                ].items()
            }
            for arm in ARM_IDS
        }
        ictal_upper = {
            arm: {
                str(reference): float(value)
                for reference, value in old_metrics[arm][
                    "tusz_bce_difference_upper95_by_reference"
                ].items()
            }
            for arm in ARM_IDS
        }
        nominal_reuse_receipt = {
            "schema_version": "soz_preprocessing_nominal_reuse_v1",
            "prior_result_sha256": str(prior_nominal_result["result_sha256"]),
            "reuse_scope": [
                "paired_extraction",
                "fixed_fold_checkpoints",
                "nominal_native_task_endpoints",
                "paired_bootstrap",
            ],
            "recomputed_scope": [
                "label_preserving_robustness_v2",
                "selection_decision",
            ],
        }
    denominator_sha = _hash_payload(denominator_payload)
    input_sha = _hash_payload(input_distribution)
    token_sha = _hash_payload(token_distribution)
    probe_sha = _hash_payload(probe_receipt)
    official_sha = _hash_payload(official)
    metrics = {}
    results = {}
    for arm in PREPROCESSING_ARM_IDS:
        sanity_only = arm == "O-REF"
        jitter = (0.0, 0.0, {"role": "not_applicable"}) if sanity_only else jitter_by_arm[arm]
        metric = PreprocessingArmSelectionMetrics(
            arm_id=arm,
            protocol_receipt_sha256=protocol.receipt_sha256,
            tuev_macro_ce6_f1=(0.0 if sanity_only else morphology_points[arm]),
            tusz_native_macro_bce=(0.0 if sanity_only else ictal_points[arm]),
            tuev_f1_difference_lower95_by_reference=(
                {reference: 0.0 for reference in ARM_IDS}
                if sanity_only
                else morphology_lower[arm]
            ),
            tusz_bce_difference_upper95_by_reference=(
                {reference: 0.0 for reference in ARM_IDS}
                if sanity_only
                else ictal_upper[arm]
            ),
            paired_denominator_receipt_sha256=denominator_sha,
            tuev_paired_patient_count=len(tuev_group_ids),
            tuev_paired_content_component_count=len(
                denominator_payload["tuev_content_component_ids"]
            ),
            tusz_paired_patient_count=len(tusz_patient_ids),
            tusz_paired_explicit_cell_count=int(
                denominator_payload["tusz_explicit_cell_count"]
            ),
            paired_attrition_count=0,
            tuev_jitter_macro_f1_max_drop=float(jitter[0]),
            tusz_jitter_macro_bce_max_increase=float(jitter[1]),
            arm_id_probe_balanced_accuracy=probe_accuracy,
            arm_id_probe_patient_disjoint=True,
            input_distribution_analysis_complete=True,
            token_distribution_analysis_complete=True,
            concept_endpoints_applicable=not sanity_only,
            official_signal_relative_l2_error=(
                float(official["signal_relative_l2_error"]) if sanity_only else None
            ),
            official_token_cosine_distance=(
                float(official["token_cosine_distance"]) if sanity_only else None
            ),
        )
        metrics[arm] = metric
        execution = {
            "arm_id": arm,
            "extraction_receipt_sha256": extraction_receipt["receipt_sha256"],
            "checkpoint_receipts": (
                checkpoint_receipts.get(arm, {}) if not sanity_only else {}
            ),
            "official_sanity_sha256": official_sha if sanity_only else None,
            "nominal_reuse_receipt_sha256": (
                _hash_payload(nominal_reuse_receipt)
                if nominal_reuse_receipt is not None
                else None
            ),
        }
        fidelity_m = {
            "arm_id": arm,
            "not_applicable": sanity_only,
            "tuev_macro_ce6_f1": 0.0 if sanity_only else morphology_points[arm],
        }
        fidelity_i = {
            "arm_id": arm,
            "not_applicable": sanity_only,
            "tusz_native_macro_bce": 0.0 if sanity_only else ictal_points[arm],
        }
        results[arm] = PreprocessingArmResultReceipt(
            arm_id=arm,
            protocol_receipt_sha256=protocol.receipt_sha256,
            arm_spec_receipt_sha256=(
                FROZEN_PREPROCESSING_ARM_SPEC_BY_ID[arm].receipt_sha256
            ),
            execution_receipt_sha256=_hash_payload(execution),
            paired_attrition_receipt_sha256=denominator_sha,
            input_distribution_receipt_sha256=input_sha,
            token_distribution_receipt_sha256=token_sha,
            tuev_ce6_fidelity_receipt_sha256=_hash_payload(fidelity_m),
            tusz_native_fidelity_receipt_sha256=_hash_payload(fidelity_i),
            onset_boundary_jitter_receipt_sha256=_hash_payload(jitter[2]),
            arm_id_shortcut_probe_receipt_sha256=probe_sha,
            metric_payload_receipt_sha256=metric.receipt_sha256,
            nested_dev_manifest_receipt_sha256=nested.receipt_sha256,
            source_patient_roster_sha256=nested.source_patient_roster_sha256,
            content_component_split_receipt_sha256=(
                nested.content_component_split_receipt_sha256
            ),
            raw_qc_intersection_receipt_sha256=(
                nested.raw_qc_intersection_receipt_sha256
            ),
            foundation_feature_receipt_sha256=(
                protocol.foundation_feature_receipt_sha256
            ),
            selection_policy_receipt_sha256=(
                protocol.selection_policy_receipt_sha256
            ),
        )
    selection_directory = output / "selection-capability"
    common_payload = {
        "schema_version": RESULT_SCHEMA,
        "formal": True,
        "training_completed": True,
        "selection_completed": True,
        "protocol_receipt_sha256": protocol.receipt_sha256,
        "nested_dev_manifest_receipt_sha256": nested.receipt_sha256,
        "metrics_by_arm": {
            arm: metrics[arm].to_payload() for arm in PREPROCESSING_ARM_IDS
        },
        "bootstrap_receipt": bootstrap_receipt,
        "denominator_receipt": denominator_payload,
        "input_distribution_receipt": input_distribution,
        "token_distribution_receipt": token_distribution,
        "arm_id_probe_receipt": probe_receipt,
        "official_sanity_receipt": official,
        "nominal_reuse_receipt": nominal_reuse_receipt,
    }
    try:
        decision = evaluate_preprocessing_arm_selection(
            protocol=protocol, arm_results=results, arm_metrics=metrics
        )
    except PreprocessingArmSelectionNoGoError as conflict:
        if selection_directory.exists():
            raise ValueError(
                "Formal NO-GO cannot coexist with a selection capability"
            ) from conflict
        no_go_receipt = conflict.to_payload()
        if conflict.receipt_sha256 != _hash_payload(no_go_receipt):
            raise RuntimeError("Formal NO-GO receipt hash is not reproducible")
        no_go_artifact = {
            **no_go_receipt,
            "receipt_sha256": conflict.receipt_sha256,
        }
        no_go_path = output / "selection-no-go.json"
        if no_go_path.exists():
            existing = json.loads(no_go_path.read_text(encoding="utf-8"))
            if not _json_equivalent(existing, no_go_artifact):
                raise ValueError("Existing formal NO-GO artifact changed")
        else:
            _atomic_json(no_go_path, no_go_artifact)
        payload = {
            **common_payload,
            "selection_status": "NO_GO",
            "downstream_training_authorized": False,
            "selected_arm_id": None,
            "selection_artifact_sha256": None,
            "selected_arm_result_receipt_sha256": None,
            "selection_no_go_receipt": no_go_receipt,
            "selection_no_go_receipt_sha256": conflict.receipt_sha256,
            "selection_no_go_artifact_sha256": _hash_file(no_go_path),
        }
        payload["result_sha256"] = _hash_payload(payload)
        _atomic_json(output / "formal-result.json", payload)
        return payload

    capability = materialize_preprocessing_selection_bundle(
        nested_dev_manifest=nested,
        protocol=protocol,
        arm_results=results,
        decision=decision,
        output_directory=selection_directory,
    )
    capability.assert_unchanged()
    payload = {
        **common_payload,
        "selection_status": "GO",
        "downstream_training_authorized": True,
        "selected_arm_id": capability.selected_arm_id,
        "selection_artifact_sha256": capability.selection_artifact_sha256,
        "selected_arm_result_receipt_sha256": (
            capability.selected_arm_result_receipt_sha256
        ),
        "selection_no_go_receipt": None,
        "selection_no_go_receipt_sha256": None,
        "selection_no_go_artifact_sha256": None,
    }
    payload["result_sha256"] = _hash_payload(payload)
    _atomic_json(output / "formal-result.json", payload)
    return payload


def _load_reusable_nominal_artifacts(
    *,
    directory: Path,
    nested,
    tuev_records: Mapping[str, object],
    tusz_events: Sequence[object],
) -> tuple[dict[str, object], dict[str, object], tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Reload and bind the completed v1 nominal run used by v2 robustness."""

    source = directory.resolve(strict=True)
    result_path = source / "formal-result.json"
    plan_path = source / "run-plan.json"
    extraction_path = source / "extraction-receipt.json"
    for path in (result_path, plan_path, extraction_path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Reusable nominal artifact is missing or symlinked: {path}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    claimed_result_sha = result.pop("result_sha256", None)
    if claimed_result_sha != _hash_payload(result):
        raise ValueError("Reusable nominal formal-result receipt changed")
    result["result_sha256"] = claimed_result_sha
    if (
        result.get("schema_version") != "soz_preprocessing_parity_formal_run_result_v1"
        or result.get("formal") is not True
        or result.get("training_completed") is not True
        or result.get("selection_completed") is not True
    ):
        raise ValueError("Reusable nominal directory is not a completed formal v1 run")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    claimed_plan_sha = plan.pop("plan_sha256", None)
    if claimed_plan_sha != _hash_payload(plan):
        raise ValueError("Reusable nominal run-plan receipt changed")
    plan["plan_sha256"] = claimed_plan_sha
    legacy_protocol = _protocol(nested, robustness_protocol="legacy_misaligned_v1")
    source_plan_checks = {
        "plan_nested_dev": (
            plan.get("nested_dev_manifest_receipt_sha256") == nested.receipt_sha256
        ),
        "plan_protocol_receipt": (
            plan.get("protocol_receipt_sha256") == legacy_protocol.receipt_sha256
        ),
        "plan_protocol_payload": _json_equivalent(
            plan.get("protocol"), asdict(legacy_protocol)
        ),
        "result_protocol_receipt": (
            result.get("protocol_receipt_sha256") == legacy_protocol.receipt_sha256
        ),
        "result_nested_dev": (
            result.get("nested_dev_manifest_receipt_sha256") == nested.receipt_sha256
        ),
    }
    failed_source_plan_checks = sorted(
        field for field, passed in source_plan_checks.items() if not passed
    )
    if failed_source_plan_checks:
        old_protocol_payload = plan.get("protocol")
        protocol_field_differences = []
        if isinstance(old_protocol_payload, Mapping):
            rebuilt_protocol_payload = asdict(legacy_protocol)
            protocol_field_differences = sorted(
                field
                for field in set(old_protocol_payload) | set(rebuilt_protocol_payload)
                if not _json_equivalent(
                    old_protocol_payload.get(field), rebuilt_protocol_payload.get(field)
                )
            )
        raise ValueError(
            "Reusable nominal run disagrees with the rebuilt source plan: "
            + ",".join(failed_source_plan_checks)
            + "; protocol_fields="
            + ",".join(protocol_field_differences)
        )

    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    extraction_claim = extraction.pop("receipt_sha256", None)
    if extraction_claim != _hash_payload(extraction):
        raise ValueError("Reusable nominal extraction receipt changed")
    extraction["receipt_sha256"] = extraction_claim
    if extraction.get("plan_sha256") != claimed_plan_sha:
        raise ValueError("Reusable nominal extraction binds another run plan")

    tuev_receipts = []
    for record_id in sorted(tuev_records):
        path = _record_receipt_path(source, "TUEV", record_id)
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("dataset") != "TUEV" or receipt.get("record_id") != record_id:
            raise ValueError("Reusable TUEV record receipt identity changed")
        tuev_receipts.append(receipt)
    tusz_record_ids = sorted({event.record_id for event in tusz_events})
    tusz_receipts = []
    for record_id in tusz_record_ids:
        path = _record_receipt_path(source, "TUSZ", record_id)
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("dataset") != "TUSZ" or receipt.get("record_id") != record_id:
            raise ValueError("Reusable TUSZ record receipt identity changed")
        tusz_receipts.append(receipt)
    if extraction.get("tuev_record_receipt_sha256s") != [
        _hash_payload(value) for value in tuev_receipts
    ] or extraction.get("tusz_record_receipt_sha256s") != [
        _hash_payload(value) for value in tusz_receipts
    ]:
        raise ValueError("Reusable nominal record-receipt roster changed")
    return result, extraction, tuple(tuev_receipts), tuple(tusz_receipts)


def main() -> int:
    args = _parser().parse_args()
    if args.morphology_batch_size < 1 or args.ictal_event_batch_size < 1:
        raise ValueError("Extraction batch sizes must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA formal run requested but CUDA is unavailable")
    output = args.output_directory.absolute()
    output.mkdir(parents=True, exist_ok=True)
    completed_path = output / "formal-result.json"
    if completed_path.exists():
        payload = json.loads(completed_path.read_text(encoding="utf-8"))
        claimed = payload.pop("result_sha256", None)
        if claimed != _hash_payload(payload):
            raise ValueError("Existing formal result receipt changed")
        payload["result_sha256"] = claimed
        _log("formal_resume_complete", **payload)
        return 0

    tuev = load_tuev_morphology_manifest(
        _bundle_directory(args.tuev_manifest),
        expected_bundle_manifest_sha256=args.tuev_bundle_sha256,
        expected_source_manifest_sha256=args.tuev_receipt_sha256,
    )
    tusz = load_tusz_ictal_training_manifest(
        _bundle_directory(args.tusz_manifest),
        expected_bundle_manifest_sha256=args.tusz_bundle_sha256,
        expected_source_manifest_sha256=args.tusz_receipt_sha256,
    )
    (
        nested,
        tuev_records,
        tuev_groups,
        tusz_events,
        tuev_items,
        tusz_items,
    ) = _build_source_plan(tuev, tusz)
    protocol = _protocol(nested, robustness_protocol="label_preserving_v2")
    reusable = None
    reusable_directory = None
    if args.reuse_nominal_directory is not None:
        reusable_directory = args.reuse_nominal_directory.resolve(strict=True)
        if reusable_directory == output.resolve():
            raise ValueError("v2 output must differ from the reusable v1 directory")
        reusable = _load_reusable_nominal_artifacts(
            directory=reusable_directory,
            nested=nested,
            tuev_records=tuev_records,
            tusz_events=tusz_events,
        )
    plan = {
        "schema_version": RUN_SCHEMA,
        "formal": True,
        "source_scope": "public_source_train_only",
        "private_data_used": False,
        "soz_labels_used": False,
        "tuev_bundle_sha256": args.tuev_bundle_sha256,
        "tuev_receipt_sha256": args.tuev_receipt_sha256,
        "tusz_bundle_sha256": args.tusz_bundle_sha256,
        "tusz_receipt_sha256": args.tusz_receipt_sha256,
        "labram_checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
        "labram_modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
        "arm_specs_sha256": FROZEN_PREPROCESSING_ARM_SPECS_SHA256,
        "nested_dev_manifest": nested.to_payload(),
        "nested_dev_manifest_receipt_sha256": nested.receipt_sha256,
        "protocol": asdict(protocol),
        "protocol_receipt_sha256": protocol.receipt_sha256,
        "nominal_reuse": (
            None
            if reusable is None
            else {
                "prior_result_sha256": reusable[0]["result_sha256"],
                "prior_extraction_receipt_sha256": reusable[1]["receipt_sha256"],
                "reuse_scope": (
                    "paired extraction, fixed-fold checkpoints, nominal endpoints, "
                    "and paired bootstrap only"
                ),
                "recomputed_scope": "label-preserving v2 robustness and selection",
            }
        ),
        "tuev_items": list(tuev_items),
        "tusz_items": list(tusz_items),
    }
    plan_sha = _hash_payload(plan)
    plan["plan_sha256"] = plan_sha
    plan_path = output / "run-plan.json"
    if plan_path.exists():
        if not _json_equivalent(
            json.loads(plan_path.read_text(encoding="utf-8")), plan
        ):
            raise ValueError("Formal preprocessing run plan changed")
    else:
        _atomic_json(plan_path, plan)
    _log(
        "formal_plan",
        plan_sha256=plan_sha,
        nested_dev_manifest_receipt_sha256=nested.receipt_sha256,
        protocol_receipt_sha256=protocol.receipt_sha256,
        tuev_crop_count=len(tuev_items),
        tusz_event_count=len(tusz_items),
        record_count=nested.record_count,
        fold_record_counts=list(nested.fold_record_counts),
    )

    if reusable is not None:
        prior_result, extraction, tuev_receipts, tusz_receipts = reusable
        _, arrays = _open_arrays(
            reusable_directory, len(tuev_items), len(tusz_items)
        )
        _set_determinism(SEED, device)
        encoder = OfficialLaBraMEncoder(
            modeling_path=args.labram_modeling,
            checkpoint_path=args.labram_checkpoint,
            tile_seconds=4,
        ).to(device)
        duration_tuev = {
            str(receipt["record_id"]): float(receipt["duration_sec"])
            for receipt in tuev_receipts
        }
        duration_tusz = {
            str(receipt["record_id"]): float(receipt["duration_sec"])
            for receipt in tusz_receipts
        }
        tuev_selected, tuev_selection_receipt = _select_jitter_items(
            tuev_items,
            dataset="TUEV",
            identity_field="parent_group_id",
            duration_by_record=duration_tuev,
        )
        tusz_selected, tusz_selection_receipt = _select_jitter_items(
            tusz_items,
            dataset="TUSZ",
            identity_field="patient_id",
            duration_by_record=duration_tusz,
        )
        tuev_jitter, tuev_jitter_receipt = _materialize_jitter_tokens(
            dataset="TUEV",
            root=args.tuev_root,
            output=output,
            selected_items=tuev_selected,
            selection_receipt=tuev_selection_receipt,
            source_objects=tuev_groups,
            tuev_records=tuev_records,
            encoder=encoder,
            device=device,
        )
        tusz_jitter, tusz_jitter_receipt = _materialize_jitter_tokens(
            dataset="TUSZ",
            root=args.tusz_root,
            output=output,
            selected_items=tusz_selected,
            selection_receipt=tusz_selection_receipt,
            source_objects=tusz_events,
            tuev_records=None,
            encoder=encoder,
            device=device,
        )
        jitter_by_arm = {}
        for arm in ARM_IDS:
            drop, increase, receipt = _evaluate_jitter(
                checkpoint_output=reusable_directory,
                arm=arm,
                tuev_selected=tuev_selected,
                tuev_jitter=tuev_jitter,
                tusz_selected=tusz_selected,
                tusz_jitter=tusz_jitter,
                arrays=arrays,
                device=device,
            )
            receipt = {
                **receipt,
                "tuev_jitter_corpus_sha256": _hash_payload(tuev_jitter_receipt),
                "tusz_jitter_corpus_sha256": _hash_payload(tusz_jitter_receipt),
            }
            jitter_by_arm[arm] = (drop, increase, receipt)
        result = _finalize_selection(
            output=output,
            nested=nested,
            protocol=protocol,
            arrays=arrays,
            tuev_items=tuev_items,
            tusz_items=tusz_items,
            morphology_by_arm={},
            ictal_by_arm={},
            jitter_by_arm=jitter_by_arm,
            input_distribution=prior_result["input_distribution_receipt"],
            token_distribution=prior_result["token_distribution_receipt"],
            probe_accuracy=float(
                prior_result["arm_id_probe_receipt"]["balanced_accuracy"]
            ),
            probe_receipt=prior_result["arm_id_probe_receipt"],
            official=prior_result["official_sanity_receipt"],
            extraction_receipt=extraction,
            checkpoint_receipts={
                arm: {"prior_result_sha256": prior_result["result_sha256"]}
                for arm in ARM_IDS
            },
            prior_nominal_result=prior_result,
        )
        _log(
            "formal_no_go"
            if result.get("selection_status") == "NO_GO"
            else "formal_complete",
            **result,
        )
        return 0

    paths, arrays = _open_arrays(output, len(tuev_items), len(tusz_items))
    _set_determinism(SEED, device)
    encoder = OfficialLaBraMEncoder(
        modeling_path=args.labram_modeling,
        checkpoint_path=args.labram_checkpoint,
        tile_seconds=4,
    ).to(device)
    tuev_receipts = _extract_tuev(
        root=args.tuev_root,
        output=output,
        records=tuev_records,
        groups=tuev_groups,
        arrays=arrays,
        encoder=encoder,
        device=device,
        batch_size=args.morphology_batch_size,
    )

    morphology_by_arm = {}
    morphology_checkpoints = {}
    if not args.stop_after_extraction:
        for arm in ARM_IDS:
            _log("training_start", dataset="TUEV", arm_id=arm)
            oof, checkpoints = _train_morphology_oof(
                output=output,
                arm=arm,
                items=tuev_items,
                arrays=arrays,
                device=device,
            )
            morphology_by_arm[arm] = oof
            morphology_checkpoints[arm] = checkpoints

    tusz_receipts = _extract_tusz(
        root=args.tusz_root,
        output=output,
        events=tusz_events,
        arrays=arrays,
        encoder=encoder,
        device=device,
        event_batch_size=args.ictal_event_batch_size,
    )
    extraction = _publish_extraction_receipt(
        output, paths, tuev_receipts, tusz_receipts, plan_sha
    )
    if args.stop_after_extraction:
        _log(
            "formal_extraction_complete",
            extraction_receipt_sha256=extraction["receipt_sha256"],
            training_started=False,
        )
        return 0

    ictal_by_arm = {}
    ictal_checkpoints = {}
    for arm in ARM_IDS:
        _log("training_start", dataset="TUSZ", arm_id=arm)
        oof, checkpoints = _train_ictal_oof(
            output=output,
            arm=arm,
            items=tusz_items,
            arrays=arrays,
            device=device,
        )
        ictal_by_arm[arm] = oof
        ictal_checkpoints[arm] = checkpoints

    duration_tuev = {
        str(receipt["record_id"]): float(receipt["duration_sec"])
        for receipt in tuev_receipts
    }
    duration_tusz = {
        str(receipt["record_id"]): float(receipt["duration_sec"])
        for receipt in tusz_receipts
    }
    tuev_selected, tuev_selection_receipt = _select_jitter_items(
        tuev_items,
        dataset="TUEV",
        identity_field="parent_group_id",
        duration_by_record=duration_tuev,
    )
    tusz_selected, tusz_selection_receipt = _select_jitter_items(
        tusz_items,
        dataset="TUSZ",
        identity_field="patient_id",
        duration_by_record=duration_tusz,
    )
    tuev_jitter, tuev_jitter_receipt = _materialize_jitter_tokens(
        dataset="TUEV",
        root=args.tuev_root,
        output=output,
        selected_items=tuev_selected,
        selection_receipt=tuev_selection_receipt,
        source_objects=tuev_groups,
        tuev_records=tuev_records,
        encoder=encoder,
        device=device,
    )
    tusz_jitter, tusz_jitter_receipt = _materialize_jitter_tokens(
        dataset="TUSZ",
        root=args.tusz_root,
        output=output,
        selected_items=tusz_selected,
        selection_receipt=tusz_selection_receipt,
        source_objects=tusz_events,
        tuev_records=None,
        encoder=encoder,
        device=device,
    )
    jitter_by_arm = {}
    for arm in ARM_IDS:
        drop, increase, receipt = _evaluate_jitter(
            checkpoint_output=output,
            arm=arm,
            tuev_selected=tuev_selected,
            tuev_jitter=tuev_jitter,
            tusz_selected=tusz_selected,
            tusz_jitter=tusz_jitter,
            arrays=arrays,
            device=device,
        )
        receipt = {
            **receipt,
            "tuev_jitter_corpus_sha256": _hash_payload(tuev_jitter_receipt),
            "tusz_jitter_corpus_sha256": _hash_payload(tusz_jitter_receipt),
        }
        jitter_by_arm[arm] = (drop, increase, receipt)

    all_record_receipts = tuple(tuev_receipts) + tuple(tusz_receipts)
    input_distribution, token_distribution = _aggregate_distribution_receipts(
        all_record_receipts
    )
    probe_accuracy, probe_receipt = _arm_id_probe(arrays, tusz_items)
    official = _official_sanity(
        tuev_root=args.tuev_root,
        tuev_records=tuev_records,
        tuev_groups=tuev_groups,
        modeling=args.labram_modeling,
        checkpoint=args.labram_checkpoint,
        device=device,
    )
    checkpoint_receipts = {
        arm: {
            "tuev": morphology_checkpoints[arm],
            "tusz": ictal_checkpoints[arm],
        }
        for arm in ARM_IDS
    }
    result = _finalize_selection(
        output=output,
        nested=nested,
        protocol=protocol,
        arrays=arrays,
        tuev_items=tuev_items,
        tusz_items=tusz_items,
        morphology_by_arm=morphology_by_arm,
        ictal_by_arm=ictal_by_arm,
        jitter_by_arm=jitter_by_arm,
        input_distribution=input_distribution,
        token_distribution=token_distribution,
        probe_accuracy=probe_accuracy,
        probe_receipt=probe_receipt,
        official=official,
        extraction_receipt=extraction,
        checkpoint_receipts=checkpoint_receipts,
    )
    _log(
        "formal_no_go"
        if result.get("selection_status") == "NO_GO"
        else "formal_complete",
        **result,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
