#!/usr/bin/env python3
"""Run the one locked paired-dev representation qualification for LaBraM DAPT."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
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

from src.soz.data.labram_source_dapt import (  # noqa: E402
    PatientUniformEpochSampler,
    SourceDAPTWindowDataset,
    load_source_dapt_manifest,
)
from src.soz.labram_source_dapt_qualification import (  # noqa: E402
    PairedReferenceQualificationDataset,
    QualificationArmStatistics,
    build_paired_metrics,
    build_qualification_artifact,
    canonical_json_bytes,
    fixed_mask_sha256,
    jensen_shannon_from_logits,
    ordered_window_identity,
    patient_bootstrap_draws,
    sha256_file,
)
from src.soz.models.labram_peft import LABRAM_PEFT_BLOCKS  # noqa: E402
from src.soz.models.labram_source_dapt import (  # noqa: E402
    AUDITED_VQNSP_SHA256,
    OfficialFrozenLaBraMVQTokenizer,
    OfficialLaBraMSourceDAPT,
    exact_random_mask,
)


SOURCE_RUN_RECEIPT = ROOT / "outputs/labram_source_only_dapt_v1_20260811/run_receipt.json"
SELECTED_ADAPTER = ROOT / "outputs/labram_source_only_dapt_v1_20260811/selected_lora.pt"
MANIFEST = ROOT / "outputs/labram_source_only_dapt_manifest_v1_20260811/manifest.json"
DEEPSOZ_EXCLUSION_ROSTER = ROOT / "outputs/deepsoz_tusz_patient_splits_v1/split_manifest.csv"
TUSZ_EDF_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
LABRAM_ROOT = Path("/mnt/hd1/dyf/workspace/LaBraM")
OUTPUT_DIR = ROOT / "outputs/labram_source_only_dapt_qualification_v1_20260811"
OUTPUT_JSON = OUTPUT_DIR / "qualification.json"

EXPECTED_SOURCE_RUN_RECEIPT_SHA256 = (
    "3364146a822c17f8d3b40a845255ef9cb06890fd745dcf4fb57350207df3a94a"
)
EXPECTED_SELECTED_ADAPTER_SHA256 = (
    "69ad1fbc423616331a26850dd6917ee0759fa505bdc5d3f1868c0de080500ec0"
)
EXPECTED_MANIFEST_SHA256 = (
    "38900fec398899f1841705c757ade1ef3ab9f5486670bc1fd6aa0a6d4ecb0b10"
)
EXPECTED_IMPLEMENTATION_SHA256 = {
    "runner": "ac0815aa612062c8e2ca5864928c4769043a2d0853610bbe1779395bb47c474f",
    "data": "ff0392d1e4f6e2b42654c59d84f895430898cf915c0ccbf57bf90dd476b618ab",
    "model": "ffdd3c19905026f4e1e7a84ef7d45ac9bdd6b1fad5eea92330531a2a1c97e09c",
    "peft": "f01fd518747657fd981aa8d578e60675f3e0b59d1671ac6059c3bf1c8e3b95c7",
}
IMPLEMENTATION_PATHS = {
    "runner": ROOT / "scripts/run_labram_source_only_dapt.py",
    "data": ROOT / "src/soz/data/labram_source_dapt.py",
    "model": ROOT / "src/soz/models/labram_source_dapt.py",
    "peft": ROOT / "src/soz/models/labram_peft.py",
}

EXPECTED_PATIENT_IDS_SHA256 = (
    "57ab16ca2d01a83d506f1b633d25f21a92ccabd31bf2dddf4b5b8a9b517992dc"
)
EXPECTED_ORDERED_WINDOW_SHA256 = (
    "f5caf96d6ff275e035e642056dd21cb6f94f795ca80303d39e1f447c9a91e4b8"
)
EXPECTED_ORDERED_DRAWS = 384
EXPECTED_UNIQUE_WINDOWS = 367

EXPECTED_ZERO_CE = 9.891463227570057
EXPECTED_ZERO_ACCURACY = 0.14852316398658635
EXPECTED_DAPT_CE = 8.778516415506601
EXPECTED_DAPT_ACCURACY = 0.16826000607397873
REPLAY_CE_ABSOLUTE_TOLERANCE = 1e-6
REPLAY_ACCURACY_ABSOLUTE_TOLERANCE = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cuda",),
        default="cuda",
        help="Formal qualification is locked to the CUDA mask/replay path.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--io-batch-size", type=int, default=4)
    return parser.parse_args()


def validate_runtime_args(args: argparse.Namespace) -> None:
    if args.device != "cuda" or args.num_workers != 2 or args.io_batch_size != 4:
        raise ValueError(
            "Formal qualification is locked to CUDA, num_workers=2, io_batch_size=4"
        )


def _load_json_mapping(path: Path) -> Mapping[str, object]:
    if path.is_symlink():
        raise ValueError(f"Qualification lineage JSON cannot be a symlink: {path}")
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"Qualification lineage JSON is not a mapping: {path}")
    return value


def validate_source_run_lineage() -> Mapping[str, object]:
    """Bind qualification to the completed epoch-18 run and unchanged sources."""

    if sha256_file(SOURCE_RUN_RECEIPT) != EXPECTED_SOURCE_RUN_RECEIPT_SHA256:
        raise ValueError("Completed source-DAPT run receipt SHA-256 changed")
    if sha256_file(SELECTED_ADAPTER) != EXPECTED_SELECTED_ADAPTER_SHA256:
        raise ValueError("Completed source-DAPT selected adapter SHA-256 changed")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Completed source-DAPT manifest SHA-256 changed")
    current_implementation = {
        name: sha256_file(path) for name, path in IMPLEMENTATION_PATHS.items()
    }
    if current_implementation != EXPECTED_IMPLEMENTATION_SHA256:
        raise ValueError(
            "One or more completed-run implementation files changed: "
            f"{current_implementation}"
        )

    receipt = _load_json_mapping(SOURCE_RUN_RECEIPT)
    config = receipt.get("run_config")
    if not isinstance(config, Mapping):
        raise TypeError("Completed source-DAPT receipt lacks run_config")
    if config.get("implementation_sha256") != EXPECTED_IMPLEMENTATION_SHA256:
        raise ValueError("Source-DAPT run_config implementation lineage changed")
    frozen_receipt_contract = {
        "mode": "full_pretext_training",
        "training_started": True,
        "training_completed": True,
        "best_epoch_by_pretext_dev_loss_only": 18,
        "qualification_pending": True,
        "representation_qualified": False,
        "soz_promotion": False,
        "candidate_promotable": False,
        "target_values_loaded": False,
        "private_data_loaded": False,
        "annotation_times_used": False,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "selected_adapter_sha256": EXPECTED_SELECTED_ADAPTER_SHA256,
    }
    for field, expected in frozen_receipt_contract.items():
        if receipt.get(field) != expected:
            raise ValueError(
                f"Completed source-DAPT receipt field changed: {field}="
                f"{receipt.get(field)!r}, expected={expected!r}"
            )
    if Path(str(receipt.get("manifest_path"))).resolve(strict=True) != MANIFEST.resolve(
        strict=True
    ):
        raise ValueError("Completed source-DAPT receipt points to a different manifest")

    zero = receipt.get("zero_lora_pretext_dev")
    epoch_rows = receipt.get("epochs")
    if not isinstance(zero, Mapping) or not isinstance(epoch_rows, list):
        raise TypeError("Completed source-DAPT receipt lacks dev metrics")
    epoch18 = [
        row.get("pretext_dev")
        for row in epoch_rows
        if isinstance(row, Mapping) and row.get("epoch") == 18
    ]
    if len(epoch18) != 1 or not isinstance(epoch18[0], Mapping):
        raise ValueError("Completed source-DAPT receipt lacks exactly one epoch-18 dev row")
    expected_metrics = (
        (zero, EXPECTED_ZERO_CE, EXPECTED_ZERO_ACCURACY, "zero-LoRA"),
        (epoch18[0], EXPECTED_DAPT_CE, EXPECTED_DAPT_ACCURACY, "epoch-18"),
    )
    for metrics, expected_ce, expected_accuracy, label in expected_metrics:
        if metrics != {
            "patient_macro_pretext_loss": expected_ce,
            "patient_macro_code_accuracy": expected_accuracy,
            "patient_count": 12,
            "windows_per_patient": 32,
        }:
            raise ValueError(f"Completed source-DAPT {label} dev receipt changed")
    return receipt


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class PreparedQualificationWindows:
    primary_car: torch.Tensor
    sensitivity_ref: torch.Tensor
    position_ids: torch.Tensor
    patient_ids: tuple[str, ...]
    record_uids: tuple[str, ...]
    grid_indices: tuple[int, ...]
    car_replay_max_abs_error_volts: float
    car_from_float32_ref_max_abs_error_volts: float

    def __post_init__(self) -> None:
        n_windows = EXPECTED_ORDERED_DRAWS
        if self.primary_car.shape != (n_windows, 19, 8, 200):
            raise ValueError("Prepared primary qualification windows must be [384,19,8,200]")
        if self.sensitivity_ref.shape != self.primary_car.shape:
            raise ValueError("Prepared REF/CAR qualification views are not aligned")
        if self.position_ids.shape != (n_windows, 19):
            raise ValueError("Prepared qualification position IDs must be [384,19]")
        if not self.primary_car.is_floating_point() or not torch.isfinite(
            self.primary_car
        ).all():
            raise ValueError("Prepared primary qualification EEG is invalid")
        if not self.sensitivity_ref.is_floating_point() or not torch.isfinite(
            self.sensitivity_ref
        ).all():
            raise ValueError("Prepared sensitivity qualification EEG is invalid")
        if self.position_ids.dtype != torch.long:
            raise ValueError("Prepared qualification position IDs must be long")
        if any(
            len(values) != n_windows
            for values in (self.patient_ids, self.record_uids, self.grid_indices)
        ):
            raise ValueError("Prepared qualification identities are misaligned")
        if self.car_replay_max_abs_error_volts != 0.0:
            raise ValueError("Qualification CAR replay must be bitwise exact")
        if (
            not math.isfinite(self.car_from_float32_ref_max_abs_error_volts)
            or self.car_from_float32_ref_max_abs_error_volts < 0
        ):
            raise ValueError("Float32 REF-to-CAR numerical discrepancy is invalid")


def build_ordered_dev_contract(
    dataset: SourceDAPTWindowDataset,
) -> tuple[list[int], tuple[str, ...], list[dict[str, object]]]:
    sampler = PatientUniformEpochSampler(
        dataset,
        windows_per_patient=32,
        seed=20260811 + 17,
    )
    sampler.set_epoch(0)
    ordered_indices = list(iter(sampler))
    identities: list[dict[str, object]] = []
    patient_draws: list[str] = []
    for dataset_index in ordered_indices:
        row, grid_index = dataset.locate(dataset_index)
        patient = str(row["patient_id"])
        patient_draws.append(patient)
        identities.append(
            ordered_window_identity(patient, str(row["record_uid"]), grid_index)
        )
    patients = tuple(sorted(set(patient_draws)))
    counts = {patient: patient_draws.count(patient) for patient in patients}
    if (
        len(ordered_indices) != EXPECTED_ORDERED_DRAWS
        or len(set(ordered_indices)) != EXPECTED_UNIQUE_WINDOWS
        or len(patients) != 12
        or set(counts.values()) != {32}
        or _sha256_json(list(patients)) != EXPECTED_PATIENT_IDS_SHA256
        or _sha256_json(identities) != EXPECTED_ORDERED_WINDOW_SHA256
    ):
        raise ValueError(
            "Frozen qualification sampler/window identities changed: "
            f"draws={len(ordered_indices)}, unique={len(set(ordered_indices))}, "
            f"patients={len(patients)}, counts={counts}, "
            f"patient_sha={_sha256_json(list(patients))}, "
            f"window_sha={_sha256_json(identities)}"
        )
    return ordered_indices, patients, identities


def load_paired_windows(
    dataset: SourceDAPTWindowDataset,
    ordered_indices: Sequence[int],
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
    ref_batches: list[torch.Tensor] = []
    position_batches: list[torch.Tensor] = []
    patient_ids: list[str] = []
    record_uids: list[str] = []
    grid_indices: list[int] = []
    maximum_replay_error = 0.0
    maximum_float32_ref_car_error = 0.0
    for batch_index, batch in enumerate(loader):
        car_batches.append(batch["primary_car"].to(dtype=torch.float32).contiguous())
        ref_batches.append(batch["sensitivity_ref"].to(dtype=torch.float32).contiguous())
        position_batches.append(batch["position_ids"].to(dtype=torch.long).contiguous())
        patient_ids.extend(str(value) for value in batch["patient_id"])
        record_uids.extend(str(value) for value in batch["record_uid"])
        grid_indices.extend(int(value) for value in batch["grid_index"].tolist())
        maximum_replay_error = max(
            maximum_replay_error,
            max(
                float(value)
                for value in batch["car_replay_max_abs_error_volts"].tolist()
            ),
        )
        maximum_float32_ref_car_error = max(
            maximum_float32_ref_car_error,
            max(
                float(value)
                for value in batch[
                    "car_from_float32_ref_max_abs_error_volts"
                ].tolist()
            ),
        )
        if (batch_index + 1) % 16 == 0:
            print(
                f"paired-signal replay: {min(len(patient_ids), 384)}/384 windows",
                flush=True,
            )
    prepared = PreparedQualificationWindows(
        primary_car=torch.cat(car_batches, dim=0),
        sensitivity_ref=torch.cat(ref_batches, dim=0),
        position_ids=torch.cat(position_batches, dim=0),
        patient_ids=tuple(patient_ids),
        record_uids=tuple(record_uids),
        grid_indices=tuple(grid_indices),
        car_replay_max_abs_error_volts=maximum_replay_error,
        car_from_float32_ref_max_abs_error_volts=maximum_float32_ref_car_error,
    )
    return prepared


def build_fixed_cuda_masks(
    prepared: PreparedQualificationWindows, *, device: torch.device
) -> torch.Tensor:
    if device.type != "cuda":
        raise ValueError("Formal fixed masks must be generated with the CUDA RNG path")
    masks: list[torch.Tensor] = []
    for record_uid, grid_index in zip(prepared.record_uids, prepared.grid_indices):
        digest = hashlib.sha256(
            f"20260811\0{record_uid}\0{grid_index}".encode("ascii")
        ).digest()
        mask_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        masks.append(exact_random_mask(1, seed=mask_seed, device=device)[0])
    result = torch.stack(masks)
    if result.shape != (384, 152) or not torch.all(result.sum(dim=1) == 76):
        raise RuntimeError("Frozen CUDA fixed masks are not exactly [384,152] with 76 masked")
    return result


def evaluate_arm(
    *,
    arm_name: str,
    model: OfficialLaBraMSourceDAPT,
    tokenizer: OfficialFrozenLaBraMVQTokenizer,
    prepared: PreparedQualificationWindows,
    fixed_masks: torch.Tensor,
    ordered_patients: Sequence[str],
    device: torch.device,
) -> QualificationArmStatistics:
    """Replay one arm one window at a time, exactly matching runner dev semantics."""

    model.eval()
    tokenizer.eval()
    tokenizer._assert_frozen()
    patient_to_index = {patient: index for index, patient in enumerate(ordered_patients)}
    patient_ce: dict[str, list[float]] = defaultdict(list)
    patient_accuracy: dict[str, list[float]] = defaultdict(list)
    patient_jsd: dict[str, list[float]] = defaultdict(list)
    prediction_counts = np.zeros((12, 8192), dtype=np.int64)
    target_digest = hashlib.sha256()

    with torch.inference_mode():
        for ordinal in range(384):
            patient = prepared.patient_ids[ordinal]
            patient_index = patient_to_index[patient]
            car = prepared.primary_car[ordinal : ordinal + 1].to(
                device=device, dtype=torch.float32
            )
            ref = prepared.sensitivity_ref[ordinal : ordinal + 1].to(
                device=device, dtype=torch.float32
            )
            positions = prepared.position_ids[ordinal].to(device=device)
            original_mask = fixed_masks[ordinal : ordinal + 1]
            codes = tokenizer(car, positions)
            if codes.shape != (1, 152):
                raise RuntimeError("Frozen VQ tokenizer did not return one code per token")
            target_digest.update(
                np.asarray(codes.cpu().numpy(), dtype="<i8", order="C").tobytes(
                    order="C"
                )
            )

            predicted = torch.full(
                (1, 152), -1, dtype=torch.long, device=device
            )
            coverage = torch.zeros((1, 152), dtype=torch.uint8, device=device)
            ce_sum = 0.0
            correct_total = 0
            jsd_sum = 0.0
            for selected_positions in (original_mask, ~original_mask):
                if int(selected_positions.sum()) != 76:
                    raise RuntimeError("Each complementary pretext path must select 76 tokens")
                # In the official complementary objective, the same positions
                # are replaced by mask token and selected for scoring.
                car_logits = model.forward_selected_logits(
                    car,
                    positions,
                    selected_positions,
                    selected_positions=selected_positions,
                )
                ref_logits = model.forward_selected_logits(
                    ref,
                    positions,
                    selected_positions,
                    selected_positions=selected_positions,
                )
                targets = codes[selected_positions]
                if car_logits.shape != (76, 8192) or targets.shape != (76,):
                    raise RuntimeError("Qualification selected-logit geometry changed")
                # Q1 is the official sum of two independently mean-reduced CEs.
                ce_sum += float(F.cross_entropy(car_logits, targets, reduction="mean"))
                path_prediction = car_logits.argmax(dim=-1)
                correct_total += int((path_prediction == targets).sum())
                predicted[selected_positions] = path_prediction
                coverage[selected_positions] += 1
                # Q4 compares full 8192-way softmax distributions on the same
                # arm/mask/pass/token; VQ targets play no role in this JSD.
                jsd_sum += float(
                    jensen_shannon_from_logits(car_logits, ref_logits).sum()
                )

            if not torch.all(coverage == 1) or torch.any(predicted < 0):
                raise RuntimeError(
                    "The two complementary paths did not scatter back to exactly 152 tokens"
                )
            window_accuracy = correct_total / 152.0
            window_jsd = jsd_sum / 152.0
            patient_ce[patient].append(ce_sum)
            patient_accuracy[patient].append(window_accuracy)
            patient_jsd[patient].append(window_jsd)
            counts = torch.bincount(
                predicted.flatten(), minlength=8192
            ).cpu().numpy()
            if int(counts.sum()) != 152:
                raise RuntimeError("Q3 predicted-code accounting is not exactly 152/window")
            prediction_counts[patient_index] += counts
            if (ordinal + 1) % 16 == 0:
                print(f"{arm_name}: {ordinal + 1}/384 windows", flush=True)

    expected_patients = tuple(ordered_patients)
    if (
        tuple(sorted(patient_ce)) != expected_patients
        or set(patient_ce) != set(patient_accuracy)
        or set(patient_ce) != set(patient_jsd)
        or any(len(patient_ce[patient]) != 32 for patient in expected_patients)
    ):
        raise RuntimeError("Qualification arm did not produce 32 windows for all 12 patients")
    if not np.all(prediction_counts.sum(axis=1) == 32 * 152):
        raise RuntimeError("Q3 patient predicted-code counts are not 32 x 152")
    return QualificationArmStatistics(
        patient_ids=expected_patients,
        patient_ce=np.asarray(
            [np.mean(patient_ce[patient]) for patient in expected_patients],
            dtype=np.float64,
        ),
        patient_accuracy=np.asarray(
            [np.mean(patient_accuracy[patient]) for patient in expected_patients],
            dtype=np.float64,
        ),
        patient_reference_jsd=np.asarray(
            [np.mean(patient_jsd[patient]) for patient in expected_patients],
            dtype=np.float64,
        ),
        prediction_counts=prediction_counts,
        aggregate_prediction_counts=prediction_counts.sum(axis=0, dtype=np.int64),
        target_ids_sha256=target_digest.hexdigest(),
    )


def assert_receipt_metric_replay(
    statistics: QualificationArmStatistics,
    *,
    expected_ce: float,
    expected_accuracy: float,
    arm_name: str,
) -> None:
    actual_ce = float(np.mean(statistics.patient_ce, dtype=np.float64))
    actual_accuracy = float(
        np.mean(statistics.patient_accuracy, dtype=np.float64)
    )
    if not math.isclose(
        actual_ce,
        expected_ce,
        rel_tol=0.0,
        abs_tol=REPLAY_CE_ABSOLUTE_TOLERANCE,
    ):
        raise RuntimeError(
            f"{arm_name} patient-macro CE does not replay the source receipt: "
            f"actual={actual_ce:.17g}, expected={expected_ce:.17g}, "
            f"atol={REPLAY_CE_ABSOLUTE_TOLERANCE}"
        )
    if not math.isclose(
        actual_accuracy,
        expected_accuracy,
        rel_tol=0.0,
        abs_tol=REPLAY_ACCURACY_ABSOLUTE_TOLERANCE,
    ):
        raise RuntimeError(
            f"{arm_name} patient-macro accuracy does not replay the source receipt: "
            f"actual={actual_accuracy:.17g}, expected={expected_accuracy:.17g}, "
            f"atol={REPLAY_ACCURACY_ABSOLUTE_TOLERANCE}"
        )


def _atomic_publish_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink() or path.parent.exists():
        raise FileExistsError(
            f"Refusing to reuse the frozen qualification output path: {path.parent}"
        )
    publication_parent = path.parent.parent.resolve(strict=True)
    temporary_dir = publication_parent / f".{path.parent.name}.tmp-{os.getpid()}"
    if temporary_dir.exists() or temporary_dir.is_symlink():
        raise FileExistsError(f"Qualification temporary output already exists: {temporary_dir}")
    temporary_dir.mkdir(mode=0o755)
    content = canonical_json_bytes(payload)
    temporary_file = temporary_dir / path.name
    published = False
    try:
        with temporary_file.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = os.open(
            temporary_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        # The complete, fsynced artifact appears at its frozen directory path
        # in one metadata operation; an interrupted write never leaves a final
        # directory that blocks a clean retry.
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
        raise RuntimeError("Formal qualification requires the CUDA mask/replay path")
    if OUTPUT_DIR.exists() or OUTPUT_DIR.is_symlink():
        raise FileExistsError(f"Frozen qualification output already exists: {OUTPUT_DIR}")

    torch.manual_seed(20260811)
    np.random.seed(20260811)
    torch.cuda.manual_seed_all(20260811)
    device = torch.device(args.device)
    validate_source_run_lineage()
    manifest = load_source_dapt_manifest(
        MANIFEST,
        deepsoz_split_roster=DEEPSOZ_EXCLUSION_ROSTER,
        tusz_root=TUSZ_EDF_ROOT,
        verify_file_inventory=True,
    )
    if manifest.sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Runtime manifest loader did not replay the frozen digest")
    if (
        manifest.payload["target_values_loaded"] is not False
        or manifest.payload["private_data_loaded"] is not False
        or manifest.payload["annotation_sidecars_opened"] is not False
        or manifest.payload["continuous_grid_contract"]["annotation_time_used"]
        is not False
    ):
        raise ValueError("Source manifest safety boundary changed")
    dev_dataset = SourceDAPTWindowDataset(manifest, split="pretext_dev")
    ordered_indices, patients, window_identities = build_ordered_dev_contract(
        dev_dataset
    )
    prepared = load_paired_windows(
        dev_dataset,
        ordered_indices,
        num_workers=args.num_workers,
        io_batch_size=args.io_batch_size,
    )
    observed_identities = [
        ordered_window_identity(patient, record_uid, grid_index)
        for patient, record_uid, grid_index in zip(
            prepared.patient_ids, prepared.record_uids, prepared.grid_indices
        )
    ]
    if observed_identities != window_identities:
        raise RuntimeError("Signal-loader order differs from the frozen dev sampler order")
    fixed_masks = build_fixed_cuda_masks(prepared, device=device)
    masks_sha256 = fixed_mask_sha256(fixed_masks)

    labram_root = LABRAM_ROOT.resolve(strict=True)
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
            raise RuntimeError("Zero-LoRA qualification arm did not start at exact zero")

    zero = evaluate_arm(
        arm_name="zero-LoRA",
        model=model,
        tokenizer=tokenizer,
        prepared=prepared,
        fixed_masks=fixed_masks,
        ordered_patients=patients,
        device=device,
    )
    assert_receipt_metric_replay(
        zero,
        expected_ce=EXPECTED_ZERO_CE,
        expected_accuracy=EXPECTED_ZERO_ACCURACY,
        arm_name="zero-LoRA",
    )
    adapter_state = torch.load(
        SELECTED_ADAPTER.resolve(strict=True),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(adapter_state, Mapping):
        raise TypeError("Selected DAPT adapter is not a tensor mapping")
    model.load_lora_state_dict(adapter_state)
    dapt = evaluate_arm(
        arm_name="selected-epoch18-DAPT",
        model=model,
        tokenizer=tokenizer,
        prepared=prepared,
        fixed_masks=fixed_masks,
        ordered_patients=patients,
        device=device,
    )
    assert_receipt_metric_replay(
        dapt,
        expected_ce=EXPECTED_DAPT_CE,
        expected_accuracy=EXPECTED_DAPT_ACCURACY,
        arm_name="selected-epoch18-DAPT",
    )

    draws = patient_bootstrap_draws()
    paired_metrics = build_paired_metrics(zero, dapt, draws=draws)
    # Recheck immutable lineage after the long signal/model replay.
    validate_source_run_lineage()
    artifact = build_qualification_artifact(
        source_run_receipt_path=SOURCE_RUN_RECEIPT,
        source_run_receipt_sha256=EXPECTED_SOURCE_RUN_RECEIPT_SHA256,
        selected_adapter_path=SELECTED_ADAPTER,
        selected_adapter_sha256=EXPECTED_SELECTED_ADAPTER_SHA256,
        manifest_path=MANIFEST,
        manifest_sha256=EXPECTED_MANIFEST_SHA256,
        patient_ids=patients,
        ordered_window_identities=window_identities,
        masks_sha256=masks_sha256,
        car_replay_max_abs_error_volts=prepared.car_replay_max_abs_error_volts,
        car_from_float32_ref_max_abs_error_volts=(
            prepared.car_from_float32_ref_max_abs_error_volts
        ),
        draws=draws,
        paired_metrics=paired_metrics,
    )
    _atomic_publish_json(OUTPUT_JSON, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
