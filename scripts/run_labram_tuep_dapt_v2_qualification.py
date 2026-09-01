#!/usr/bin/env python3
"""Run the one locked TUH-internal representation qualification for DAPT-v2."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_labram_tuep_dapt_v2 as training_runner  # noqa: E402
from src.soz.data.labram_tuep_dapt_v2 import (  # noqa: E402
    EXPECTED_QUALIFICATION_PATIENTS,
    PatientUniformEpochSampler,
    TUEPDAPTV2WindowDataset,
    load_tuep_dapt_v2_manifest,
)
from src.soz.labram_source_dapt_qualification import (  # noqa: E402
    PairedReferenceQualificationDataset,
    fixed_mask_sha256,
    jensen_shannon_from_logits,
    ordered_window_identity,
    sha256_file,
    sha256_json,
)
from src.soz.labram_tuep_dapt_v2_qualification import (  # noqa: E402
    QUALIFICATION_PATIENTS,
    QUALIFICATION_SEED,
    QUALIFICATION_TOKENS_PER_WINDOW,
    QUALIFICATION_WINDOW_DRAWS,
    QUALIFICATION_WINDOWS_PER_PATIENT,
    QualificationArmStatistics,
    build_paired_metrics,
    build_qualification_artifact,
    canonical_json_bytes,
    patient_bootstrap_draws,
)
from src.soz.models.labram_peft import (  # noqa: E402
    LABRAM_PEFT_BLOCKS,
    LABRAM_PEFT_RANK,
    LABRAM_PEFT_TOKEN_DIM,
)
from src.soz.models.labram_source_dapt import (  # noqa: E402
    AUDITED_VQNSP_SHA256,
    OfficialFrozenLaBraMVQTokenizer,
    OfficialLaBraMSourceDAPT,
    exact_random_mask,
)


DEFAULT_MANIFEST = (
    ROOT / "outputs/labram_tuep_dapt_v2_manifest_20260811/manifest.json"
)
DEFAULT_TUEP_ROOT = Path("/mnt/hd1/dyf/dataset/tuh_eeg_epilepsy/v2.0.1")
DEFAULT_LABRAM_ROOT = Path("/mnt/hd1/dyf/workspace/LaBraM")
DEFAULT_SOURCE_RUN = ROOT / "outputs/labram_tuep_dapt_v2_20260811"
DEFAULT_OUTPUT = (
    ROOT / "outputs/labram_tuep_dapt_v2_qualification_v1_20260812"
)
OUTPUT_FILENAME = "qualification.json"
EXPECTED_MANIFEST_SHA256 = (
    "83abb54ec7a22820368ddc06d78d6f32338595d5405834ecf0d7daed980e165b"
)

WINDOW_SAMPLER_SEED = QUALIFICATION_SEED + 17
WINDOW_SAMPLER_EPOCH = 0
NUM_WORKERS = 2
IO_BATCH_SIZE = 4
EXPECTED_EPOCHS = 10
EXPECTED_ADAPTER_KEYS = {
    f"blocks.{block}.attn.qkv.lora_{factor}"
    for block in LABRAM_PEFT_BLOCKS
    for factor in ("A", "B")
}


@dataclass(frozen=True)
class AuthorizedCandidate:
    receipt: Mapping[str, object]
    receipt_path: Path
    receipt_sha256: str
    adapter_path: Path
    adapter_sha256: str
    adapter_state: Mapping[str, torch.Tensor]
    selected_epoch: int


@dataclass(frozen=True)
class PreparedQualificationWindows:
    primary_car: torch.Tensor
    sensitivity_source_reference: torch.Tensor
    position_ids: torch.Tensor
    patient_ids: tuple[str, ...]
    record_uids: tuple[str, ...]
    grid_indices: tuple[int, ...]
    source_references: tuple[str, ...]
    car_replay_max_abs_error_volts: float
    car_from_float32_source_reference_max_abs_error_volts: float

    def __post_init__(self) -> None:
        expected_signal = (QUALIFICATION_WINDOW_DRAWS, 19, 8, 200)
        if tuple(self.primary_car.shape) != expected_signal:
            raise ValueError("Prepared DAPT-v2 CAR windows must be [1152,19,8,200]")
        if tuple(self.sensitivity_source_reference.shape) != expected_signal:
            raise ValueError("Prepared DAPT-v2 source-reference windows are misaligned")
        if tuple(self.position_ids.shape) != (QUALIFICATION_WINDOW_DRAWS, 19):
            raise ValueError("Prepared DAPT-v2 positions must be [1152,19]")
        if self.position_ids.dtype != torch.long:
            raise TypeError("Prepared DAPT-v2 positions must be long")
        for values in (self.primary_car, self.sensitivity_source_reference):
            if not values.is_floating_point() or not torch.isfinite(values).all():
                raise ValueError("Prepared DAPT-v2 qualification signal is invalid")
        identities = (
            self.patient_ids,
            self.record_uids,
            self.grid_indices,
            self.source_references,
        )
        if any(len(values) != QUALIFICATION_WINDOW_DRAWS for values in identities):
            raise ValueError("Prepared DAPT-v2 qualification identities are misaligned")
        if set(self.source_references) - {"REF", "LE"}:
            raise ValueError("Prepared source reference must be REF or LE")
        if self.car_replay_max_abs_error_volts != 0.0:
            raise ValueError("DAPT-v2 qualification CAR replay must be bitwise exact")
        if (
            not math.isfinite(
                self.car_from_float32_source_reference_max_abs_error_volts
            )
            or self.car_from_float32_source_reference_max_abs_error_volts < 0.0
        ):
            raise ValueError("DAPT-v2 source-reference/CAR discrepancy is invalid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tuep-root", type=Path, default=DEFAULT_TUEP_ROOT)
    parser.add_argument("--labram-root", type=Path, default=DEFAULT_LABRAM_ROOT)
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--io-batch-size", type=int, default=IO_BATCH_SIZE)
    return parser.parse_args()


def validate_runtime_args(args: argparse.Namespace) -> None:
    expected_paths = (
        ("manifest", args.manifest, DEFAULT_MANIFEST),
        ("tuep_root", args.tuep_root, DEFAULT_TUEP_ROOT),
        ("labram_root", args.labram_root, DEFAULT_LABRAM_ROOT),
        ("source_run_dir", args.source_run_dir, DEFAULT_SOURCE_RUN),
        ("output_dir", args.output_dir, DEFAULT_OUTPUT),
    )
    for name, actual, expected in expected_paths:
        if actual.resolve() != expected.resolve():
            raise ValueError(f"Formal DAPT-v2 qualification {name} is locked to {expected}")
    if (
        args.device != "cuda"
        or args.num_workers != NUM_WORKERS
        or args.io_batch_size != IO_BATCH_SIZE
    ):
        raise ValueError(
            "Formal DAPT-v2 qualification is locked to CUDA, 2 workers, and IO batch 4"
        )
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite DAPT-v2 qualification output: {args.output_dir}"
        )


def _regular_json(path_value: Path) -> tuple[Mapping[str, object], str]:
    path = path_value.resolve(strict=True)
    if path_value.is_symlink() or not path.is_file():
        raise ValueError(f"DAPT-v2 receipt is not a regular file: {path_value}")
    content = path.read_bytes()
    payload = json.loads(content)
    if not isinstance(payload, Mapping):
        raise TypeError("DAPT-v2 run receipt must be a mapping")
    return payload, hashlib.sha256(content).hexdigest()


def _require_false(payload: Mapping[str, object], fields: Sequence[str]) -> None:
    for field in fields:
        if payload.get(field) is not False:
            raise ValueError(f"DAPT-v2 source safety field must be false: {field}")


def _validate_adapter_state(
    path: Path, expected_sha256: str
) -> Mapping[str, torch.Tensor]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Selected DAPT-v2 adapter is not a regular file")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Selected DAPT-v2 adapter hash differs from the receipt")
    state = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or set(state) != EXPECTED_ADAPTER_KEYS:
        raise ValueError("Selected DAPT-v2 adapter tensor roster changed")
    validated: dict[str, torch.Tensor] = {}
    effective_nonzero = False
    for block in LABRAM_PEFT_BLOCKS:
        a_key = f"blocks.{block}.attn.qkv.lora_A"
        b_key = f"blocks.{block}.attn.qkv.lora_B"
        a_value = state[a_key]
        b_value = state[b_key]
        if (
            not isinstance(a_value, torch.Tensor)
            or not isinstance(b_value, torch.Tensor)
            or tuple(a_value.shape) != (LABRAM_PEFT_RANK, LABRAM_PEFT_TOKEN_DIM)
            or tuple(b_value.shape)
            != (3 * LABRAM_PEFT_TOKEN_DIM, LABRAM_PEFT_RANK)
            or not a_value.is_floating_point()
            or not b_value.is_floating_point()
            or not torch.isfinite(a_value).all()
            or not torch.isfinite(b_value).all()
        ):
            raise ValueError("Selected DAPT-v2 adapter tensor is invalid")
        effective_nonzero = effective_nonzero or bool(
            torch.count_nonzero(b_value @ a_value).item()
        )
        validated[a_key] = a_value.detach().clone()
        validated[b_key] = b_value.detach().clone()
    if not effective_nonzero:
        raise ValueError(
            "Formal DAPT-v2 receipt did not select an effectively non-zero adapter"
        )
    return validated


def authorize_formal_candidate(
    *,
    source_run_dir: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> AuthorizedCandidate:
    """Authorize the candidate without opening any qualification signal."""

    run_dir = source_run_dir.resolve(strict=True)
    if source_run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("Formal DAPT-v2 source run is not a regular directory")
    receipt_path = run_dir / "run_receipt.json"
    receipt, receipt_sha = _regular_json(receipt_path)
    if (
        receipt.get("schema_version") != "soz_labram_tuep_dapt_v2_run_v1"
        or receipt.get("mode") != "formal_training"
        or receipt.get("training_started") is not True
        or receipt.get("training_completed") is not True
    ):
        raise ValueError("DAPT-v2 source run is not a completed formal training run")
    _require_false(
        receipt,
        (
            "target_values_loaded",
            "diagnostic_directory_labels_used",
            "private_data_loaded",
            "annotation_sidecars_opened",
            "annotation_times_used",
            "qualification_signal_split_loaded",
            "soz_promotion",
            "candidate_promotable",
            "representation_qualified",
        ),
    )
    if receipt.get("qualification_patient_signals_seen") != 0:
        raise ValueError("Training receipt already accessed qualification patients")

    manifest = manifest_path.resolve(strict=True)
    if manifest_path.is_symlink() or sha256_file(manifest) != expected_manifest_sha256:
        raise ValueError("Frozen DAPT-v2 manifest file/hash changed")
    if (
        receipt.get("manifest_path") != str(manifest)
        or receipt.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise ValueError("DAPT-v2 receipt is bound to a different manifest")
    config = receipt.get("run_config")
    if not isinstance(config, Mapping):
        raise TypeError("DAPT-v2 receipt lacks its run configuration")
    if (
        config.get("mode") != "formal_training"
        or config.get("manifest_sha256") != expected_manifest_sha256
        or config.get("signal_splits_loaded") != ["pretext_train", "pretext_dev"]
        or config.get("qualification_signal_split_loaded") is not False
        or config.get("epochs") != EXPECTED_EPOCHS
    ):
        raise ValueError("DAPT-v2 run configuration violated qualification isolation")

    baseline = receipt.get("zero_lora_pretext_dev")
    epochs = receipt.get("epochs")
    if not isinstance(baseline, Mapping) or not isinstance(epochs, list):
        raise TypeError("DAPT-v2 receipt lacks formal dev selection metrics")
    if len(epochs) != EXPECTED_EPOCHS:
        raise ValueError("DAPT-v2 formal receipt must contain exactly 10 epochs")
    eligible_count = 0
    for expected_epoch, row in enumerate(epochs):
        if (
            not isinstance(row, Mapping)
            or row.get("epoch") != expected_epoch
            or not isinstance(row.get("pretext_dev"), Mapping)
            or not isinstance(row.get("selection_eligibility"), Mapping)
        ):
            raise ValueError("DAPT-v2 epoch/dev selection row is malformed")
        recomputed = training_runner.evaluate_epoch_eligibility(
            baseline, row["pretext_dev"]
        )
        if dict(row["selection_eligibility"]) != recomputed:
            raise ValueError("DAPT-v2 stored dev eligibility does not replay")
        eligible_count += int(recomputed["eligible"])
    replay_selected = training_runner.select_epoch_by_frozen_rule(baseline, epochs)
    selected = receipt.get("best_epoch_by_frozen_eligibility_then_ce")

    # This is the fail-closed boundary.  It intentionally occurs before any
    # selected-adapter read and before manifest/dataset qualification loading.
    if (
        isinstance(selected, bool)
        or not isinstance(selected, int)
        or selected != replay_selected
        or not 0 <= selected < EXPECTED_EPOCHS
        or epochs[selected]["selection_eligibility"]["eligible"] is not True
        or receipt.get("selection_fallback_to_zero_lora") is not False
        or eligible_count < 1
        or receipt.get("eligible_epoch_count") != eligible_count
    ):
        raise ValueError(
            "Formal DAPT-v2 receipt has no dev-eligible non-zero adapter; "
            "qualification signal must remain unopened"
        )
    expected_best_ce = float(
        epochs[selected]["pretext_dev"]["patient_macro_official_ce"]
    )
    if receipt.get("best_eligible_patient_macro_official_ce") != expected_best_ce:
        raise ValueError("DAPT-v2 selected-epoch CE contradicts the receipt")

    raw_adapter_path = receipt.get("selected_adapter_path")
    adapter_sha = receipt.get("selected_adapter_sha256")
    if not isinstance(raw_adapter_path, str) or not isinstance(adapter_sha, str):
        raise TypeError("DAPT-v2 selected adapter lineage is missing")
    adapter_path = Path(raw_adapter_path)
    expected_adapter = (run_dir / "selected_lora.pt").resolve(strict=True)
    if (
        not adapter_path.is_absolute()
        or adapter_path.is_symlink()
        or adapter_path.resolve(strict=True) != expected_adapter
        or len(adapter_sha) != 64
        or any(character not in "0123456789abcdef" for character in adapter_sha)
    ):
        raise ValueError("DAPT-v2 selected adapter path/hash declaration is invalid")
    adapter_state = _validate_adapter_state(expected_adapter, adapter_sha)
    return AuthorizedCandidate(
        receipt=receipt,
        receipt_path=receipt_path.resolve(strict=True),
        receipt_sha256=receipt_sha,
        adapter_path=expected_adapter,
        adapter_sha256=adapter_sha,
        adapter_state=adapter_state,
        selected_epoch=selected,
    )


def load_authorized_qualification_dataset(
    args: argparse.Namespace,
) -> tuple[AuthorizedCandidate, object, TUEPDAPTV2WindowDataset]:
    """Preserve the authorization-before-qualification-signal ordering."""

    candidate = authorize_formal_candidate(
        source_run_dir=args.source_run_dir,
        manifest_path=args.manifest,
        expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
    )
    manifest = load_tuep_dapt_v2_manifest(
        args.manifest,
        tuep_root=args.tuep_root,
        verify_file_inventory=True,
    )
    if manifest.sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Runtime DAPT-v2 qualification manifest digest changed")
    for field in (
        "target_values_loaded",
        "diagnostic_directory_labels_used",
        "private_data_loaded",
        "annotation_sidecars_opened",
    ):
        if manifest.payload[field] is not False:
            raise ValueError(f"Qualification manifest safety flag changed: {field}")
    if manifest.payload["continuous_grid_contract"]["annotation_time_used"] is not False:
        raise ValueError("Qualification manifest used annotation time")
    dataset = TUEPDAPTV2WindowDataset(
        manifest, split="pretext_qualification"
    )
    expected_patients = set(
        manifest.payload["pretext_split_contract"]["qualification_patient_ids"]
    )
    if (
        len(expected_patients) != EXPECTED_QUALIFICATION_PATIENTS
        or set(dataset.patient_to_indices) != expected_patients
    ):
        raise RuntimeError("DAPT-v2 qualification patient roster changed")
    return candidate, manifest, dataset


def build_ordered_qualification_contract(
    dataset: TUEPDAPTV2WindowDataset,
) -> tuple[
    list[int],
    tuple[str, ...],
    list[dict[str, object]],
    tuple[str, ...],
]:
    sampler = PatientUniformEpochSampler(
        dataset,
        windows_per_patient=QUALIFICATION_WINDOWS_PER_PATIENT,
        seed=WINDOW_SAMPLER_SEED,
    )
    sampler.set_epoch(WINDOW_SAMPLER_EPOCH)
    ordered_indices = list(iter(sampler))
    identities: list[dict[str, object]] = []
    patient_draws: list[str] = []
    source_references: list[str] = []
    for dataset_index in ordered_indices:
        row, grid_index = dataset.locate(dataset_index)
        patient = str(row["patient_id"])
        patient_draws.append(patient)
        source_references.append(str(row["source_reference"]))
        identities.append(
            ordered_window_identity(patient, str(row["record_uid"]), grid_index)
        )
    patients = tuple(sorted(set(patient_draws)))
    counts = {patient: patient_draws.count(patient) for patient in patients}
    if (
        len(ordered_indices) != QUALIFICATION_WINDOW_DRAWS
        or len(patients) != QUALIFICATION_PATIENTS
        or set(counts.values()) != {QUALIFICATION_WINDOWS_PER_PATIENT}
        or set(source_references) - {"REF", "LE"}
    ):
        raise RuntimeError("Frozen DAPT-v2 36 x 32 qualification sampler changed")
    return ordered_indices, patients, identities, tuple(source_references)


def _reference_section(
    rows: Sequence[Mapping[str, object]],
    *,
    include_draw_count: bool,
    all_patients: Sequence[str],
) -> dict[str, object]:
    result: dict[str, object] = {}
    references_by_patient: dict[str, set[str]] = {
        patient: set() for patient in all_patients
    }
    for reference in ("REF", "LE"):
        selected = [row for row in rows if row["source_reference"] == reference]
        patients = sorted({str(row["patient_id"]) for row in selected})
        records = sorted({str(row["record_uid"]) for row in selected})
        for patient in patients:
            references_by_patient[patient].add(reference)
        item: dict[str, object] = {
            "patient_count": len(patients),
            "patient_ids_sha256": sha256_json(patients),
            "unique_record_count": len(records),
            "record_uids_sha256": sha256_json(records),
        }
        if include_draw_count:
            item["window_draw_count"] = len(selected)
        result[reference] = item
    composition = {"REF_only": 0, "LE_only": 0, "mixed_REF_LE": 0}
    for patient, references in references_by_patient.items():
        if references == {"REF"}:
            composition["REF_only"] += 1
        elif references == {"LE"}:
            composition["LE_only"] += 1
        elif references == {"REF", "LE"}:
            composition["mixed_REF_LE"] += 1
        else:
            raise RuntimeError(f"Qualification patient has no reference stratum: {patient}")
    result["patient_composition"] = composition
    return result


def build_reference_stratification(
    dataset: TUEPDAPTV2WindowDataset,
    ordered_indices: Sequence[int],
    patients: Sequence[str],
) -> dict[str, object]:
    inventory_rows = list(dataset.records)
    sampled_rows: list[Mapping[str, object]] = []
    for index in ordered_indices:
        row, _ = dataset.locate(int(index))
        sampled_rows.append(row)
    return {
        "qualification_eligible_inventory": _reference_section(
            inventory_rows,
            include_draw_count=False,
            all_patients=patients,
        ),
        "fixed_window_replay": _reference_section(
            sampled_rows,
            include_draw_count=True,
            all_patients=patients,
        ),
    }


def load_paired_windows(
    dataset: TUEPDAPTV2WindowDataset,
    ordered_indices: Sequence[int],
    source_references: Sequence[str],
    *,
    num_workers: int,
    io_batch_size: int,
) -> PreparedQualificationWindows:
    paired = PairedReferenceQualificationDataset(dataset, ordered_indices)
    loader = DataLoader(
        paired,
        batch_size=io_batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        multiprocessing_context=("spawn" if num_workers > 0 else None),
        drop_last=False,
    )
    car_batches: list[torch.Tensor] = []
    source_batches: list[torch.Tensor] = []
    position_batches: list[torch.Tensor] = []
    patient_ids: list[str] = []
    record_uids: list[str] = []
    grid_indices: list[int] = []
    maximum_replay_error = 0.0
    maximum_float32_error = 0.0
    for batch_index, batch in enumerate(loader):
        car_batches.append(batch["primary_car"].to(dtype=torch.float32).contiguous())
        source_batches.append(
            batch["sensitivity_ref"].to(dtype=torch.float32).contiguous()
        )
        position_batches.append(
            batch["position_ids"].to(dtype=torch.long).contiguous()
        )
        patient_ids.extend(str(value) for value in batch["patient_id"])
        record_uids.extend(str(value) for value in batch["record_uid"])
        grid_indices.extend(int(value) for value in batch["grid_index"].tolist())
        maximum_replay_error = max(
            maximum_replay_error,
            max(float(value) for value in batch["car_replay_max_abs_error_volts"].tolist()),
        )
        maximum_float32_error = max(
            maximum_float32_error,
            max(
                float(value)
                for value in batch[
                    "car_from_float32_ref_max_abs_error_volts"
                ].tolist()
            ),
        )
        if (batch_index + 1) % 32 == 0:
            print(
                f"qualification signal replay: {len(patient_ids)}/1152 windows",
                flush=True,
            )
    return PreparedQualificationWindows(
        primary_car=torch.cat(car_batches, dim=0),
        sensitivity_source_reference=torch.cat(source_batches, dim=0),
        position_ids=torch.cat(position_batches, dim=0),
        patient_ids=tuple(patient_ids),
        record_uids=tuple(record_uids),
        grid_indices=tuple(grid_indices),
        source_references=tuple(source_references),
        car_replay_max_abs_error_volts=maximum_replay_error,
        car_from_float32_source_reference_max_abs_error_volts=(
            maximum_float32_error
        ),
    )


def build_fixed_masks(
    prepared: PreparedQualificationWindows, *, device: torch.device
) -> torch.Tensor:
    masks: list[torch.Tensor] = []
    for record_uid, grid_index in zip(prepared.record_uids, prepared.grid_indices):
        digest = hashlib.sha256(
            f"{QUALIFICATION_SEED}\0qualification\0{record_uid}\0{grid_index}".encode(
                "ascii"
            )
        ).digest()
        mask_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        masks.append(exact_random_mask(1, seed=mask_seed, device=device)[0])
    result = torch.stack(masks)
    if tuple(result.shape) != (QUALIFICATION_WINDOW_DRAWS, 152) or not torch.all(
        result.sum(dim=1) == 76
    ):
        raise RuntimeError("DAPT-v2 fixed masks are not [1152,152] with 76 masked")
    return result


def build_fixed_targets(
    *,
    tokenizer: OfficialFrozenLaBraMVQTokenizer,
    prepared: PreparedQualificationWindows,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    tokenizer.eval()
    tokenizer._assert_frozen()
    targets: list[torch.Tensor] = []
    digest = hashlib.sha256()
    with torch.inference_mode():
        for ordinal in range(QUALIFICATION_WINDOW_DRAWS):
            car = prepared.primary_car[ordinal : ordinal + 1].to(
                device=device, dtype=torch.float32
            )
            positions = prepared.position_ids[ordinal].to(device=device)
            codes = tokenizer(car, positions)
            if tuple(codes.shape) != (1, 152):
                raise RuntimeError("Frozen VQ tokenizer target geometry changed")
            cpu_codes = codes.detach().cpu().to(dtype=torch.long).contiguous()
            digest.update(
                np.asarray(cpu_codes.numpy(), dtype="<i8", order="C").tobytes(
                    order="C"
                )
            )
            targets.append(cpu_codes[0])
            if (ordinal + 1) % 32 == 0:
                print(f"fixed VQ targets: {ordinal + 1}/1152 windows", flush=True)
    result = torch.stack(targets)
    if tuple(result.shape) != (QUALIFICATION_WINDOW_DRAWS, 152):
        raise RuntimeError("Fixed DAPT-v2 VQ target matrix changed")
    return result, digest.hexdigest()


def evaluate_arm(
    *,
    arm_name: str,
    model: OfficialLaBraMSourceDAPT,
    prepared: PreparedQualificationWindows,
    fixed_masks: torch.Tensor,
    fixed_targets: torch.Tensor,
    target_ids_sha256: str,
    ordered_patients: Sequence[str],
    device: torch.device,
) -> QualificationArmStatistics:
    """Evaluate one arm on exactly the same signals, targets, and masks."""

    model.eval()
    patient_to_index = {patient: index for index, patient in enumerate(ordered_patients)}
    patient_ce: dict[str, list[float]] = defaultdict(list)
    patient_accuracy: dict[str, list[float]] = defaultdict(list)
    patient_jsd: dict[str, list[float]] = defaultdict(list)
    prediction_counts = np.zeros((QUALIFICATION_PATIENTS, 8192), dtype=np.int64)
    with torch.inference_mode():
        for ordinal in range(QUALIFICATION_WINDOW_DRAWS):
            patient = prepared.patient_ids[ordinal]
            patient_index = patient_to_index[patient]
            car = prepared.primary_car[ordinal : ordinal + 1].to(
                device=device, dtype=torch.float32
            )
            source_reference = prepared.sensitivity_source_reference[
                ordinal : ordinal + 1
            ].to(device=device, dtype=torch.float32)
            positions = prepared.position_ids[ordinal].to(device=device)
            original_mask = fixed_masks[ordinal : ordinal + 1]
            codes = fixed_targets[ordinal : ordinal + 1].to(device=device)
            predicted = torch.full((1, 152), -1, dtype=torch.long, device=device)
            coverage = torch.zeros((1, 152), dtype=torch.uint8, device=device)
            ce_sum = 0.0
            correct_total = 0
            jsd_sum = 0.0
            for selected_positions in (original_mask, ~original_mask):
                if int(selected_positions.sum()) != 76:
                    raise RuntimeError("Complementary DAPT-v2 path must score 76 tokens")
                car_logits = model.forward_selected_logits(
                    car,
                    positions,
                    selected_positions,
                    selected_positions=selected_positions,
                )
                source_logits = model.forward_selected_logits(
                    source_reference,
                    positions,
                    selected_positions,
                    selected_positions=selected_positions,
                )
                targets = codes[selected_positions]
                if tuple(car_logits.shape) != (76, 8192) or tuple(targets.shape) != (76,):
                    raise RuntimeError("DAPT-v2 selected-logit geometry changed")
                ce_sum += float(F.cross_entropy(car_logits, targets, reduction="mean"))
                path_prediction = car_logits.argmax(dim=-1)
                correct_total += int((path_prediction == targets).sum())
                predicted[selected_positions] = path_prediction
                coverage[selected_positions] += 1
                jsd_sum += float(
                    jensen_shannon_from_logits(car_logits, source_logits).sum()
                )
            if not torch.all(coverage == 1) or torch.any(predicted < 0):
                raise RuntimeError("Complementary paths did not cover exactly 152 tokens")
            patient_ce[patient].append(ce_sum)
            patient_accuracy[patient].append(correct_total / 152.0)
            patient_jsd[patient].append(jsd_sum / 152.0)
            counts = torch.bincount(predicted.flatten(), minlength=8192).cpu().numpy()
            if int(counts.sum()) != QUALIFICATION_TOKENS_PER_WINDOW:
                raise RuntimeError("DAPT-v2 predicted-code accounting changed")
            prediction_counts[patient_index] += counts
            if (ordinal + 1) % 16 == 0:
                print(f"{arm_name}: {ordinal + 1}/1152 windows", flush=True)
    patients = tuple(ordered_patients)
    if (
        tuple(sorted(patient_ce)) != patients
        or set(patient_ce) != set(patient_accuracy)
        or set(patient_ce) != set(patient_jsd)
        or any(len(patient_ce[patient]) != 32 for patient in patients)
    ):
        raise RuntimeError("DAPT-v2 arm did not produce 32 windows for 36 patients")
    return QualificationArmStatistics(
        patient_ids=patients,
        patient_ce=np.asarray(
            [np.mean(patient_ce[patient]) for patient in patients],
            dtype=np.float64,
        ),
        patient_accuracy=np.asarray(
            [np.mean(patient_accuracy[patient]) for patient in patients],
            dtype=np.float64,
        ),
        patient_reference_jsd=np.asarray(
            [np.mean(patient_jsd[patient]) for patient in patients],
            dtype=np.float64,
        ),
        prediction_counts=prediction_counts,
        aggregate_prediction_counts=prediction_counts.sum(axis=0, dtype=np.int64),
        target_ids_sha256=target_ids_sha256,
    )


def _atomic_publish_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink() or path.parent.exists():
        raise FileExistsError(
            f"Refusing to reuse DAPT-v2 qualification output: {path.parent}"
        )
    publication_parent = path.parent.parent.resolve(strict=True)
    temporary_dir = publication_parent / f".{path.parent.name}.tmp-{os.getpid()}"
    if temporary_dir.exists() or temporary_dir.is_symlink():
        raise FileExistsError(f"Qualification temporary output exists: {temporary_dir}")
    temporary_dir.mkdir(mode=0o755)
    temporary_file = temporary_dir / path.name
    published = False
    try:
        with temporary_file.open("xb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = os.open(
            temporary_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.rename(temporary_dir, path.parent)
        published = True
    finally:
        if not published and temporary_dir.exists() and not temporary_dir.is_symlink():
            temporary_file.unlink(missing_ok=True)
            temporary_dir.rmdir()
    descriptor = os.open(
        publication_parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    args = parse_args()
    validate_runtime_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Formal DAPT-v2 qualification requires CUDA")
    torch.manual_seed(QUALIFICATION_SEED)
    np.random.seed(QUALIFICATION_SEED)
    torch.cuda.manual_seed_all(QUALIFICATION_SEED)
    device = torch.device("cuda")

    candidate, manifest, dataset = load_authorized_qualification_dataset(args)
    ordered_indices, patients, identities, source_references = (
        build_ordered_qualification_contract(dataset)
    )
    stratification = build_reference_stratification(
        dataset, ordered_indices, patients
    )
    prepared = load_paired_windows(
        dataset,
        ordered_indices,
        source_references,
        num_workers=args.num_workers,
        io_batch_size=args.io_batch_size,
    )
    observed_identities = [
        ordered_window_identity(patient, record_uid, grid_index)
        for patient, record_uid, grid_index in zip(
            prepared.patient_ids, prepared.record_uids, prepared.grid_indices
        )
    ]
    if observed_identities != identities:
        raise RuntimeError("Qualification loader order differs from frozen sampler")
    fixed_masks = build_fixed_masks(prepared, device=device)

    labram_root = args.labram_root.resolve(strict=True)
    model = OfficialLaBraMSourceDAPT(
        modeling_path=labram_root / "modeling_finetune.py",
        checkpoint_path=labram_root / "checkpoints/labram-base.pth",
    ).to(device=device, dtype=torch.float32)
    tokenizer = OfficialFrozenLaBraMVQTokenizer(
        official_root=labram_root,
        checkpoint_path=labram_root / "checkpoints/vqnsp.pth",
        expected_sha256=AUDITED_VQNSP_SHA256,
    ).to(device=device, dtype=torch.float32)
    for block in LABRAM_PEFT_BLOCKS:
        if torch.count_nonzero(model._lora(block).lora_B).item() != 0:
            raise RuntimeError("Exact zero-LoRA qualification arm is not zero")
    fixed_targets, target_digest = build_fixed_targets(
        tokenizer=tokenizer,
        prepared=prepared,
        device=device,
    )
    zero = evaluate_arm(
        arm_name="exact-zero-LoRA",
        model=model,
        prepared=prepared,
        fixed_masks=fixed_masks,
        fixed_targets=fixed_targets,
        target_ids_sha256=target_digest,
        ordered_patients=patients,
        device=device,
    )
    model.load_lora_state_dict(candidate.adapter_state)
    v2 = evaluate_arm(
        arm_name=f"selected-epoch-{candidate.selected_epoch}-DAPT-v2",
        model=model,
        prepared=prepared,
        fixed_masks=fixed_masks,
        fixed_targets=fixed_targets,
        target_ids_sha256=target_digest,
        ordered_patients=patients,
        device=device,
    )
    draws = patient_bootstrap_draws()
    paired = build_paired_metrics(zero, v2, draws=draws)

    # Re-bind the long run to the same immutable receipt/adapter before publish.
    replay_candidate = authorize_formal_candidate(
        source_run_dir=args.source_run_dir,
        manifest_path=args.manifest,
        expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
    )
    if (
        replay_candidate.receipt_sha256 != candidate.receipt_sha256
        or replay_candidate.adapter_sha256 != candidate.adapter_sha256
        or replay_candidate.selected_epoch != candidate.selected_epoch
    ):
        raise RuntimeError("DAPT-v2 source lineage changed during qualification")

    artifact = build_qualification_artifact(
        source_run_receipt_path=candidate.receipt_path,
        source_run_receipt_sha256=candidate.receipt_sha256,
        selected_adapter_path=candidate.adapter_path,
        selected_adapter_sha256=candidate.adapter_sha256,
        selected_epoch=candidate.selected_epoch,
        manifest_path=manifest.path,
        manifest_sha256=manifest.sha256,
        qualification_runner_sha256=sha256_file(Path(__file__)),
        qualification_statistics_sha256=sha256_file(
            ROOT / "src/soz/labram_tuep_dapt_v2_qualification.py"
        ),
        patient_ids=patients,
        ordered_window_identities=identities,
        unique_window_count=len(set(ordered_indices)),
        fixed_mask_sha256=fixed_mask_sha256(fixed_masks),
        reference_stratification=stratification,
        car_replay_max_abs_error_volts=prepared.car_replay_max_abs_error_volts,
        car_from_float32_source_reference_max_abs_error_volts=(
            prepared.car_from_float32_source_reference_max_abs_error_volts
        ),
        draws=draws,
        zero_statistics=zero,
        v2_statistics=v2,
        paired_metrics=paired,
    )
    output_path = args.output_dir / OUTPUT_FILENAME
    _atomic_publish_json(output_path, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
