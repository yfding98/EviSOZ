#!/usr/bin/env python3
"""Run the single LaBraM-preserving BUNDL-style source-only qualification.

The runner never loads a DeepSOZ SOZ target or any private input.  It removes
every DeepSOZ identity reachable from the public 102-patient development
union, removes the historical I-dev/I-gate identities, and performs a fixed
five-fold OOF comparison on the remaining TUSZ patients.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Iterator, Mapping, Sequence


_REQUIRED_CUBLAS_WORKSPACE = ":4096:8"
_observed_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _observed_workspace is None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _REQUIRED_CUBLAS_WORKSPACE
elif _observed_workspace != _REQUIRED_CUBLAS_WORKSPACE:
    raise RuntimeError("BUNDL qualification requires CUBLAS_WORKSPACE_CONFIG=':4096:8'")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_tusz_ictal_token_cache import (  # noqa: E402
    load_formal_token_corpus,
)
from src.soz.bundl_ictal import (  # noqa: E402
    BUNDL_CLEAN_CORE_RADIUS_SECONDS,
    BUNDL_DROPOUT_PROBABILITY,
    BUNDL_MC_SAMPLES,
    BUNDL_NEGATIVE_LABEL_UNRELIABILITY,
    BUNDL_PROBABILITY_CEILING,
    BUNDL_PROBABILITY_FLOOR,
    BundlDropoutK31IctalHead,
    bundl_clean_core_mask,
    bundl_ictal_loss,
    patient_macro_masked_mean,
    sample_mcd_logits,
)
from src.soz.cached_concept_training import (  # noqa: E402
    IctalTokenBagDataset,
)
from src.soz.concept_metrics import patient_macro_ictal_metrics  # noqa: E402
from src.soz.concept_run import (  # noqa: E402
    IctalTrainingConfig,
    ictal_determinism_runtime,
    ictal_head_state_sha256,
    validate_ictal_cuda_environment,
)
from src.soz.data.tusz_training import (  # noqa: E402
    load_tusz_ictal_training_manifest,
)
from src.soz.ictal_target_snapshot import (  # noqa: E402
    build_tusz_ictal_token_bag_dataset_from_target_snapshot,
    load_verified_ictal_target_snapshot,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    load_preprocessing_selection_capability,
)


SCHEMA = "soz_labram_bundl_edge_time_source_only_oof_v1"
PROVISIONAL_PASS_STATUS = "provisional_internal_gates_passed_pending_trace_audit"
INTERNAL_FAIL_STATUS = "stop_source_native_probability_gate_failed"
FINAL_AUDIT_RELATIVE_PATH = (
    "scripts/audit_labram_bundl_ictal_trace_v1.py"
)
PROTOCOL_PATH = ROOT / "research/02_method/labram_next_single_candidate_protocol_20260812_zh.md"
PROTOCOL_SHA256 = "7560721b54bfd5c5c5247d53426c717391f8c766a1c677e3b3a77037e14bf537"

MASTER_MANIFEST = ROOT / "outputs/tusz_ictal_master_manifest_v4_1_20260809_current_preflight"
MASTER_BUNDLE_SHA256 = "73e821d08805c3a7e8ae75011dd98fe10c388d7291c74881286438e91cacc35f"
MASTER_SOURCE_SHA256 = "d5329b9231ecea7aaae6e126f5cd7a17a51f21b950025b32369592379acf8cb8"
TOKEN_CORPUS = ROOT / "outputs/tusz_ictal_token_corpus_formal_v4_20260809/master"
TOKEN_CORPUS_INDEX_SHA256 = "a7d672e3228cdc71fafb46e910033f6a5302a9e2e0a5f5716f7f4c8292ecfc26"
TARGET_SNAPSHOT = ROOT / "outputs/tusz_ictal_prediction_artifacts_formal_v4_20260809/final/native"
TARGET_SNAPSHOT_MANIFEST_SHA256 = "bc22681928e596ef6564af51f54215e96a9560a21cdeaedef043ccd324596cba"
TARGET_SNAPSHOT_RECEIPT_SHA256 = "e216338d5112a67d20fcba5d545834af2b84c8896a8713b9919866e839c7953a"
TRAINING_TARGETS_FILE_SHA256 = "99bc6250dfadb407fe890a39ef9fa00743968d7d1c6ce3e710d917c555294722"
TRAINING_TARGET_MASK_FILE_SHA256 = "e4a372ef95dc85be57077389d83ca1fbdd18c99489af090fe2755e4b0cc5da60"
PREPROCESSING_SELECTION = ROOT / "outputs/preprocessing_parity_formal_v2_1_20260809/selection-capability"
PREPROCESSING_SELECTION_SHA256 = "b4aa73bff2800f12186085976a5655db6882a38232d775d11234efa387171485"
PREPROCESSING_PROTOCOL_SHA256 = "9a75dd2f3293d4d944380c0d82dcfca6a95e332f3b999e32e52b15d89622a196"
IDENTITY_FIREWALL = (
    ROOT
    / "outputs/labram_bundl_identity_firewall_v1_20260812/identity_firewall.json"
)
IDENTITY_FIREWALL_SHA256 = "b1322d1bdae1608fe5040e2f1c40dadbfd04f6608d352be1421e694889b56942"
PUBLIC_UNION = ROOT / "outputs/public_development_union_v11_20260811/manifest.json"
PUBLIC_UNION_SHA256 = "89a9ca456c724c2dee4d14a2c0da5a1190e58f97ad602060f6dda5f619b97232"

DEFAULT_OUTPUT = ROOT / "outputs/labram_bundl_ictal_source_only_v1_20260812"
DEFAULT_TRACE = ROOT / "outputs/labram_bundl_ictal_source_only_v1_20260812.strace.raw"
PRELAUNCH_LEDGER = (
    ROOT / "outputs/labram_bundl_ictal_source_only_v1_prelaunch_ledger_20260812.json"
)
PRELAUNCH_LEDGER_SCHEMA = "soz_labram_bundl_source_only_prelaunch_ledger_v1"
FORMAL_RUNNER_ARGV = (
    "--device",
    "cuda",
    "--output-directory",
    str(DEFAULT_OUTPUT),
)
FORMAL_STRACE_ARGV = (
    "strace",
    "-f",
    "-s",
    "4096",
    "-yy",
    "-e",
    "trace=open,openat,openat2,statx,readlink",
    "-o",
    str(DEFAULT_TRACE),
    "python3",
    str(Path(__file__).resolve()),
    *FORMAL_RUNNER_ARGV,
)
SEED = 20260812
OUTER_FOLDS = 5
WARMUP_EPOCHS = 10
FINETUNE_EPOCHS = 10
BOOTSTRAP_REPLICATES = 10_000
FOLD_SALT = "labram-bundl-source-only-v1-20260812"
EVENT_MICROBATCH = 4
EXPECTED_HEAD_TRAINABLE_PARAMETERS = 81_665
PAIRED_METRIC_KEYS = (
    "bce",
    "brier",
    "auroc",
    "average_precision",
    "sensitivity_at_0_5",
    "false_positive_rate_at_0_5",
    "core_bce",
    "core_brier",
    "noncore_bce",
    "noncore_brier",
)
_EVENT_RE = re.compile(r"^([a-z]{8})_s[0-9]+_t[0-9]+__ev[0-9]+$")

EXPECTED_MASTER_COUNTS = (1_729_430, 502_093, 1_227_337)
EXPECTED_MASTER_CORE_COUNTS = (1_474_350, 411_359, 1_062_991)
EXPECTED_FIT_COUNTS = (94_958, 46_118, 48_840)
EXPECTED_FIT_CORE_COUNTS = (84_015, 41_366, 42_649)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _roster_sha256(values: tuple[str, ...]) -> str:
    return hashlib.sha256(_canonical_json_bytes(list(values))).hexdigest()


def _load_pinned_json(path: Path, expected_sha256: str) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved != path.resolve():
        raise ValueError(f"Pinned JSON is not a regular resolved file: {path}")
    raw = resolved.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Pinned JSON SHA mismatch: {path}")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Pinned JSON must be an object: {path}")
    return payload


def _ledger_code_paths() -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "bundl_ictal": ROOT / "src/soz/bundl_ictal.py",
        "concept_heads": ROOT / "src/soz/models/concept_heads.py",
        "concept_metrics": ROOT / "src/soz/concept_metrics.py",
        "cached_dataset": ROOT / "src/soz/cached_concept_training.py",
        "target_loader": ROOT / "src/soz/ictal_target_snapshot.py",
        "auditor": ROOT / FINAL_AUDIT_RELATIVE_PATH,
    }


def _expected_ledger_data_sha256() -> dict[str, str]:
    return {
        "master_bundle": MASTER_BUNDLE_SHA256,
        "master_source": MASTER_SOURCE_SHA256,
        "token_corpus_index": TOKEN_CORPUS_INDEX_SHA256,
        "target_snapshot_manifest": TARGET_SNAPSHOT_MANIFEST_SHA256,
        "target_snapshot_receipt": TARGET_SNAPSHOT_RECEIPT_SHA256,
        "training_targets_file": TRAINING_TARGETS_FILE_SHA256,
        "training_target_mask_file": TRAINING_TARGET_MASK_FILE_SHA256,
        "preprocessing_selection": PREPROCESSING_SELECTION_SHA256,
        "preprocessing_protocol": PREPROCESSING_PROTOCOL_SHA256,
        "identity_firewall": IDENTITY_FIREWALL_SHA256,
        "public_union": PUBLIC_UNION_SHA256,
    }


def _expected_formal_launch() -> dict[str, object]:
    return {
        "working_directory": str(ROOT),
        "python_executable": str(Path(sys.executable).resolve()),
        "runner_argv": list(FORMAL_RUNNER_ARGV),
        "strace_argv": list(FORMAL_STRACE_ARGV),
        "trace_path": str(DEFAULT_TRACE),
        "output_directory": str(DEFAULT_OUTPUT),
        "result_path": str(DEFAULT_OUTPUT / "result.json"),
        "audit_receipt_path": str(DEFAULT_OUTPUT / "post_run_trace_audit.json"),
    }


def _load_prelaunch_ledger() -> tuple[dict[str, object], str]:
    if PRELAUNCH_LEDGER.is_symlink():
        raise ValueError("Pre-launch ledger must not be a symlink")
    resolved = PRELAUNCH_LEDGER.resolve(strict=True)
    raw = resolved.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Pre-launch ledger must be an object")
    if set(payload) != {
        "schema_version",
        "code_sha256",
        "protocol_sha256",
        "data_sha256",
        "formal_launch",
    }:
        raise ValueError("Pre-launch ledger violates the closed top-level schema")
    if payload.get("schema_version") != PRELAUNCH_LEDGER_SCHEMA:
        raise ValueError("Unexpected pre-launch ledger schema")
    code_sha = payload.get("code_sha256")
    if not isinstance(code_sha, dict) or set(code_sha) != set(_ledger_code_paths()):
        raise ValueError("Pre-launch ledger code hash schema changed")
    observed_code_sha = {
        name: _file_sha256(path) for name, path in _ledger_code_paths().items()
    }
    if code_sha != observed_code_sha:
        raise ValueError("Pre-launch ledger code hash mismatch")
    if payload.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("Pre-launch ledger protocol hash mismatch")
    if payload.get("data_sha256") != _expected_ledger_data_sha256():
        raise ValueError("Pre-launch ledger data hash contract changed")
    if _file_sha256(TARGET_SNAPSHOT / "training_targets.npy") != TRAINING_TARGETS_FILE_SHA256:
        raise ValueError("Training target tensor file changed")
    if (
        _file_sha256(TARGET_SNAPSHOT / "training_target_mask.npy")
        != TRAINING_TARGET_MASK_FILE_SHA256
    ):
        raise ValueError("Training target mask file changed")
    if payload.get("formal_launch") != _expected_formal_launch():
        raise ValueError("Pre-launch ledger formal launch contract changed")
    return payload, hashlib.sha256(raw).hexdigest()


def _safe_new_output(path: Path) -> Path:
    target = Path(os.path.abspath(path))
    if target.name in {"", ".", ".."} or not target.parent.is_dir():
        raise ValueError("Output must be a concrete new directory under an existing parent")
    if os.path.lexists(target):
        raise FileExistsError(f"Output already exists: {target}")
    return target


def _validate_formal_launch(
    *, preflight_only: bool, device: torch.device, output: Path
) -> None:
    """Forbid CPU or alternate-output formal retries while allowing preflight."""

    if preflight_only:
        return
    if device.type != "cuda":
        raise ValueError("Formal BUNDL qualification is CUDA-only")
    requested = Path(os.path.abspath(output))
    frozen = Path(os.path.abspath(DEFAULT_OUTPUT))
    if requested != frozen:
        raise ValueError("Formal BUNDL qualification requires the frozen default output")


def _internal_result_status(passed: bool) -> str:
    """Return a non-final status; trace audit owns final qualification."""

    return PROVISIONAL_PASS_STATUS if passed else INTERNAL_FAIL_STATUS


def _validate_formal_runner_argv(
    *, preflight_only: bool, raw_argv: Sequence[str]
) -> None:
    if preflight_only:
        return
    if tuple(raw_argv) != FORMAL_RUNNER_ARGV:
        raise ValueError("Formal runner argv differs from the sealed launch ledger")
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Formal runner working directory differs from the sealed ledger")


def _execution_receipt(device: torch.device) -> dict[str, object]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Formal execution receipt requires an available CUDA device")
    index = device.index if device.index is not None else torch.cuda.current_device()
    receipt = {
        "python_version": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_index": int(index),
        "gpu_name": torch.cuda.get_device_name(index),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(index)),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    expected = {
        "cublas_workspace_config": _REQUIRED_CUBLAS_WORKSPACE,
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Formal CUDA determinism receipt violates the frozen policy")
    return receipt


def _seed_for(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def _set_torch_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _event_slices(n_events: int) -> Iterator[slice]:
    for start in range(0, n_events, EVENT_MICROBATCH):
        yield slice(start, min(start + EVENT_MICROBATCH, n_events))


def _label_counts(targets: torch.Tensor, mask: torch.Tensor) -> tuple[int, int, int]:
    observed = targets[mask]
    positive = int(observed.sum().item())
    total = int(observed.numel())
    return total, positive, total - positive


def _public_union_identities(payload: Mapping[str, object]) -> tuple[str, ...]:
    if payload.get("schema_version") != "soz_public_development_union_v11":
        raise ValueError("Unexpected public development union schema")
    if payload.get("patient_count") != 102 or payload.get("event_count") != 988:
        raise ValueError("Public development union count changed")
    events = payload.get("events")
    if not isinstance(events, list) or len(events) != 988:
        raise ValueError("Public development union event roster changed")
    identities: set[str] = set()
    for row in events:
        if not isinstance(row, Mapping):
            raise TypeError("Public union event row must be an object")
        match = _EVENT_RE.fullmatch(str(row.get("event_id", "")).strip())
        if match is None:
            raise ValueError("Cannot derive public patient identity from union event")
        identities.add(match.group(1))
    result = tuple(sorted(identities))
    if len(result) != 102:
        raise ValueError("Public union must reconstruct exactly 102 identities")
    return result


def _load_identity_firewall(
    payload: Mapping[str, object], master_patients: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    expected_fields = {
        "schema_version",
        "serialization",
        "source_split_schema_version",
        "source_split_sha256",
        "deepsoz_master_overlap_public_patient_ids",
        "deepsoz_master_overlap_roster_sha256",
        "i_dev_public_patient_ids",
        "i_dev_roster_sha256",
        "i_gate_public_patient_ids",
        "i_gate_roster_sha256",
        "rosters_pairwise_disjoint",
        "deepsoz_soz_target_values_exported",
        "private_values_exported",
        "label_counts_prevalence_balance_exported",
    }
    if set(payload) != expected_fields:
        raise ValueError("Identity firewall sidecar violates its closed schema")
    fixed = {
        "schema_version": "soz_labram_bundl_identity_firewall_v1",
        "serialization": "canonical_json_utf8_no_pickle",
        "source_split_schema_version": "soz_ictal_formal_v5_auxiliary_split_v1",
        "rosters_pairwise_disjoint": True,
        "deepsoz_soz_target_values_exported": False,
        "private_values_exported": False,
        "label_counts_prevalence_balance_exported": False,
    }
    if any(payload.get(key) != value for key, value in fixed.items()):
        raise ValueError("Identity firewall sidecar changed an access boundary")
    source_sha = str(payload.get("source_split_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ValueError("Identity firewall source receipt is not a SHA256")
    deepsoz = tuple(
        sorted(str(value) for value in payload["deepsoz_master_overlap_public_patient_ids"])
    )
    dev = tuple(sorted(str(value) for value in payload["i_dev_public_patient_ids"]))
    gate = tuple(sorted(str(value) for value in payload["i_gate_public_patient_ids"]))
    if len(deepsoz) != 65 or len(dev) != 12 or len(gate) != 12:
        raise ValueError("Identity firewall roster counts changed")
    receipts = (
        (deepsoz, payload["deepsoz_master_overlap_roster_sha256"]),
        (dev, payload["i_dev_roster_sha256"]),
        (gate, payload["i_gate_roster_sha256"]),
    )
    if any(_roster_sha256(roster) != receipt for roster, receipt in receipts):
        raise ValueError("Identity firewall roster receipt mismatch")
    rosters = (set(deepsoz), set(dev), set(gate))
    if any(left & right for index, left in enumerate(rosters) for right in rosters[index + 1 :]):
        raise ValueError("DeepSOZ/I-dev/I-gate identity rosters overlap")
    if not set(deepsoz + dev + gate) <= set(master_patients):
        raise ValueError("V5 identity firewall contains an unknown master patient")
    return deepsoz, dev, gate


def _fixed_folds(patient_ids: tuple[str, ...]) -> dict[int, tuple[str, ...]]:
    ordered = sorted(
        patient_ids,
        key=lambda patient: hashlib.sha256(
            f"{FOLD_SALT}|{patient}".encode("utf-8")
        ).hexdigest(),
    )
    result = {
        fold: tuple(sorted(ordered[fold::OUTER_FOLDS]))
        for fold in range(OUTER_FOLDS)
    }
    if sorted(len(values) for values in result.values()) != [8, 8, 8, 8, 8]:
        raise RuntimeError("Fixed source-only folds must contain eight patients each")
    if set().union(*(set(values) for values in result.values())) != set(patient_ids):
        raise RuntimeError("Fixed source-only folds do not cover the fit cohort")
    return result


def _memoized_dataset(
    dataset: IctalTokenBagDataset, patient_ids: tuple[str, ...]
) -> IctalTokenBagDataset:
    source = dataset.subset(patient_ids)
    bags = {bag.patient_id: bag for bag in source.iter_epoch()}
    if tuple(sorted(bags)) != source.patient_ids:
        raise RuntimeError("Memoized source-only dataset changed its patient roster")
    return IctalTokenBagDataset(
        source.patient_ids,
        bags.__getitem__,
        training_manifest_sha256=source.training_manifest_sha256,
        token_source_manifest_sha256=source.token_source_manifest_sha256,
        foundation_feature_receipt_sha256=source.foundation_feature_receipt_sha256,
        formal_token_corpus_verified=source.formal_token_corpus_verified,
        formal_token_corpus_index_sha256=source.formal_token_corpus_index_sha256,
        formal_token_corpus_training_bundle_manifest_sha256=(
            source.formal_token_corpus_training_bundle_manifest_sha256
        ),
        formal_token_corpus_event_roster_sha256=(
            source.formal_token_corpus_event_roster_sha256
        ),
        formal_token_corpus_patient_roster_sha256=(
            source.formal_token_corpus_patient_roster_sha256
        ),
        formal_token_corpus_tensor_roster_sha256=(
            source.formal_token_corpus_tensor_roster_sha256
        ),
        training_authorized=True,
    )


def _fold_training_view(
    dataset: IctalTokenBagDataset, train_patients: tuple[str, ...]
) -> IctalTokenBagDataset:
    """Return the exact 32-patient view required by complete-bag epochs."""

    expected = tuple(sorted(train_patients))
    view = dataset.subset(expected)
    if view.patient_ids != expected or len(view.patient_ids) != 32:
        raise RuntimeError("Fold training view must contain exactly the declared 32 patients")
    return view


def _core_by_patient(dataset: IctalTokenBagDataset) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for bag in dataset.iter_epoch():
        core = bundl_clean_core_mask(
            bag.targets,
            bag.target_mask,
            radius_seconds=BUNDL_CLEAN_CORE_RADIUS_SECONDS,
        ).contiguous()
        if not core.any():
            raise ValueError(f"Patient has no clean-core cell: {bag.patient_id}")
        result[bag.patient_id] = core
    return result


def _ordered_patients(
    patients: tuple[str, ...], *, fold: int, stage: str, epoch: int
) -> tuple[str, ...]:
    order = list(patients)
    random.Random(_seed_for(SEED, fold, stage, epoch, "patient-order")).shuffle(order)
    return tuple(order)


def _train_epoch(
    head: BundlDropoutK31IctalHead,
    dataset: IctalTokenBagDataset,
    optimizer: torch.optim.Optimizer,
    core_masks: Mapping[str, torch.Tensor],
    *,
    patient_order: tuple[str, ...],
    fold: int,
    epoch: int,
    stage: str,
    device: torch.device,
) -> dict[str, object]:
    if stage not in {"warmup_core_bce", "finetune_hard_bce", "finetune_bundl"}:
        raise ValueError("Unknown source-only training stage")
    head.train()
    patient_losses: list[float] = []
    n_events = 0
    n_selected = 0
    teacher_unreliability_sum = 0.0
    teacher_unreliability_count = 0
    for bag in dataset.iter_epoch(patient_order):
        optimizer.zero_grad(set_to_none=True)
        base_mask = core_masks[bag.patient_id] if stage == "warmup_core_bce" else bag.target_mask
        patient_selected = int(base_mask.sum().item())
        if patient_selected < 1:
            raise ValueError("A complete patient bag has no selected training cells")
        patient_loss = 0.0
        for event_slice in _event_slices(len(bag.event_ids)):
            tokens = torch.stack(
                [event.tokens for event in bag.token_events[event_slice]], dim=0
            ).to(device=device, non_blocking=True)
            targets = bag.targets[event_slice].to(device=device, non_blocking=True)
            selected_mask = base_mask[event_slice].to(device=device, non_blocking=True)
            micro_selected = int(selected_mask.sum().item())
            if micro_selected < 1:
                continue
            start = 0 if event_slice.start is None else int(event_slice.start)
            gradient_seed = _seed_for(
                SEED, fold, "shared-gradient-dropout", epoch, bag.patient_id, start
            )
            _set_torch_seed(gradient_seed, device)
            logits = head(tokens.detach())
            patient_ids = torch.zeros(
                logits.shape[0], dtype=torch.long, device=device
            )
            if stage == "finetune_bundl":
                teacher_seed = _seed_for(
                    SEED, fold, "bundl-teacher", epoch, bag.patient_id, start
                )
                mc_logits = sample_mcd_logits(
                    head,
                    tokens.detach(),
                    seed=teacher_seed,
                    n_samples=BUNDL_MC_SAMPLES,
                )
                loss_output = bundl_ictal_loss(
                    logits,
                    targets,
                    selected_mask,
                    patient_ids,
                    mc_logits,
                )
                micro_loss = loss_output.total
                positive = selected_mask & (targets == 1)
                if positive.any():
                    teacher_unreliability_sum += float(
                        loss_output.positive_label_unreliability[positive].sum().cpu()
                    )
                    teacher_unreliability_count += int(positive.sum().item())
            else:
                element = F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), targets, reduction="none"
                )
                micro_loss = patient_macro_masked_mean(
                    element, selected_mask, patient_ids
                )
            weight = micro_selected / patient_selected
            (micro_loss * weight).backward()
            patient_loss += float(micro_loss.detach().cpu()) * weight
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        patient_losses.append(patient_loss)
        n_events += len(bag.event_ids)
        n_selected += patient_selected
    return {
        "mean_patient_loss": sum(patient_losses) / len(patient_losses),
        "n_patients": len(patient_losses),
        "n_events": n_events,
        "n_selected_cells": n_selected,
        "mean_positive_label_unreliability": (
            None
            if teacher_unreliability_count == 0
            else teacher_unreliability_sum / teacher_unreliability_count
        ),
    }


def _fit_fold(
    dataset: IctalTokenBagDataset,
    core_masks: Mapping[str, torch.Tensor],
    train_patients: tuple[str, ...],
    *,
    fold: int,
    device: torch.device,
) -> tuple[
    BundlDropoutK31IctalHead,
    BundlDropoutK31IctalHead,
    dict[str, object],
]:
    training_dataset = _fold_training_view(dataset, train_patients)
    initialization_seed = _seed_for(SEED, fold, "initialization")
    _set_torch_seed(initialization_seed, device)
    warmup = BundlDropoutK31IctalHead().to(device)
    initial_sha = ictal_head_state_sha256(warmup)
    warm_optimizer = torch.optim.AdamW(
        warmup.parameters(), lr=1e-3, weight_decay=1e-2
    )
    warm_rows = []
    for epoch in range(WARMUP_EPOCHS):
        row = _train_epoch(
            warmup,
            training_dataset,
            warm_optimizer,
            core_masks,
            patient_order=_ordered_patients(
                train_patients, fold=fold, stage="warmup", epoch=epoch
            ),
            fold=fold,
            epoch=epoch,
            stage="warmup_core_bce",
            device=device,
        )
        warm_rows.append(row)
        print(
            json.dumps(
                {"fold": fold, "stage": "warmup_core_bce", "epoch": epoch + 1, **row},
                sort_keys=True,
            ),
            flush=True,
        )
    warm_sha = ictal_head_state_sha256(warmup)
    warm_state = copy.deepcopy(warmup.state_dict())
    warm_optimizer_state = copy.deepcopy(warm_optimizer.state_dict())

    control = BundlDropoutK31IctalHead().to(device)
    candidate = BundlDropoutK31IctalHead().to(device)
    control.load_state_dict(warm_state, strict=True)
    candidate.load_state_dict(warm_state, strict=True)
    control_optimizer = torch.optim.AdamW(
        control.parameters(), lr=1e-3, weight_decay=1e-2
    )
    candidate_optimizer = torch.optim.AdamW(
        candidate.parameters(), lr=1e-3, weight_decay=1e-2
    )
    control_optimizer.load_state_dict(copy.deepcopy(warm_optimizer_state))
    candidate_optimizer.load_state_dict(copy.deepcopy(warm_optimizer_state))
    control_rows = []
    candidate_rows = []
    for epoch in range(FINETUNE_EPOCHS):
        order = _ordered_patients(
            train_patients, fold=fold, stage="finetune", epoch=epoch
        )
        control_row = _train_epoch(
            control,
            training_dataset,
            control_optimizer,
            core_masks,
            patient_order=order,
            fold=fold,
            epoch=epoch,
            stage="finetune_hard_bce",
            device=device,
        )
        candidate_row = _train_epoch(
            candidate,
            training_dataset,
            candidate_optimizer,
            core_masks,
            patient_order=order,
            fold=fold,
            epoch=epoch,
            stage="finetune_bundl",
            device=device,
        )
        control_rows.append(control_row)
        candidate_rows.append(candidate_row)
        print(
            json.dumps(
                {
                    "fold": fold,
                    "stage": "paired_finetune",
                    "epoch": epoch + 1,
                    "control_loss": control_row["mean_patient_loss"],
                    "candidate_loss": candidate_row["mean_patient_loss"],
                    "candidate_positive_unreliability": candidate_row[
                        "mean_positive_label_unreliability"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return control, candidate, {
        "initial_state_sha256": initial_sha,
        "shared_warmup_state_sha256": warm_sha,
        "control_final_state_sha256": ictal_head_state_sha256(control),
        "candidate_final_state_sha256": ictal_head_state_sha256(candidate),
        "warmup_epochs": warm_rows,
        "control_finetune_epochs": control_rows,
        "candidate_finetune_epochs": candidate_rows,
    }


@torch.no_grad()
def _predict_patients(
    head: BundlDropoutK31IctalHead,
    dataset: IctalTokenBagDataset,
    patient_ids: tuple[str, ...],
    *,
    device: torch.device,
) -> dict[str, dict[str, object]]:
    head.eval()
    result: dict[str, dict[str, object]] = {}
    for bag in dataset.iter_subset(patient_ids):
        patient_logits = []
        for event_slice in _event_slices(len(bag.event_ids)):
            tokens = torch.stack(
                [event.tokens for event in bag.token_events[event_slice]], dim=0
            ).to(device=device, non_blocking=True)
            patient_logits.append(head(tokens.detach()).cpu())
        logits = torch.cat(patient_logits, dim=0).contiguous()
        if not torch.isfinite(logits).all():
            raise ValueError("OOF prediction contains a non-finite logit")
        result[bag.patient_id] = {
            "event_ids": bag.event_ids,
            "logits": logits,
            "targets": bag.targets.clone().contiguous(),
            "mask": bag.target_mask.clone().contiguous(),
        }
    return result


def _masked_bce_brier(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> tuple[float | None, float | None]:
    if not mask.any():
        return None, None
    selected_logits = logits.squeeze(-1)[mask]
    selected_targets = targets[mask]
    bce = F.binary_cross_entropy_with_logits(
        selected_logits, selected_targets, reduction="mean"
    )
    brier = ((selected_logits.sigmoid() - selected_targets) ** 2).mean()
    return float(bce), float(brier)


def _patient_metric_row(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    core: torch.Tensor,
) -> dict[str, float | int | None]:
    event_patient = torch.zeros(logits.shape[0], dtype=torch.long)
    metrics = patient_macro_ictal_metrics(
        logits, targets, mask, event_patient
    )
    probabilities = logits.squeeze(-1).sigmoid()
    predicted = probabilities >= 0.5
    positive = mask & (targets == 1)
    negative = mask & (targets == 0)
    sensitivity = (
        None
        if not positive.any()
        else float(predicted[positive].to(torch.float32).mean())
    )
    false_positive_rate = (
        None
        if not negative.any()
        else float(predicted[negative].to(torch.float32).mean())
    )
    core_bce, core_brier = _masked_bce_brier(logits, targets, core)
    noncore = mask & ~core
    noncore_bce, noncore_brier = _masked_bce_brier(logits, targets, noncore)
    return {
        "bce": metrics.patient_macro_bce,
        "brier": metrics.patient_macro_brier,
        "auroc": metrics.patient_macro_auroc,
        "average_precision": metrics.patient_macro_average_precision,
        "sensitivity_at_0_5": sensitivity,
        "false_positive_rate_at_0_5": false_positive_rate,
        "core_bce": core_bce,
        "core_brier": core_brier,
        "noncore_bce": noncore_bce,
        "noncore_brier": noncore_brier,
        "observed_cells": metrics.n_observed_labels,
        "positive_cells": metrics.n_positive_labels,
        "negative_cells": metrics.n_negative_labels,
        "core_cells": int(core.sum().item()),
        "noncore_cells": int(noncore.sum().item()),
    }


def _mean_defined(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else sum(values) / len(values)


def _macro_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    keys = (
        "bce",
        "brier",
        "auroc",
        "average_precision",
        "sensitivity_at_0_5",
        "false_positive_rate_at_0_5",
        "core_bce",
        "core_brier",
        "noncore_bce",
        "noncore_brier",
    )
    result: dict[str, object] = {key: _mean_defined(rows, key) for key in keys}
    result.update(
        {
            "patient_count": len(rows),
            "discrimination_patient_count": sum(
                row.get("auroc") is not None for row in rows
            ),
            "observed_cells": sum(int(row["observed_cells"]) for row in rows),
            "positive_cells": sum(int(row["positive_cells"]) for row in rows),
            "negative_cells": sum(int(row["negative_cells"]) for row in rows),
            "core_cells": sum(int(row["core_cells"]) for row in rows),
            "noncore_cells": sum(int(row["noncore_cells"]) for row in rows),
        }
    )
    return result


def _paired_bootstrap(
    control_rows: Mapping[str, Mapping[str, object]],
    candidate_rows: Mapping[str, Mapping[str, object]],
    key: str,
    *,
    seed: int,
) -> dict[str, object]:
    patients = [
        patient
        for patient in sorted(control_rows)
        if control_rows[patient].get(key) is not None
        and candidate_rows[patient].get(key) is not None
    ]
    if not patients:
        return {"patient_count": 0, "control": None, "candidate": None, "delta": None, "ci95": None}
    control = np.asarray([float(control_rows[p][key]) for p in patients], dtype=np.float64)
    candidate = np.asarray([float(candidate_rows[p][key]) for p in patients], dtype=np.float64)
    deltas = candidate - control
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(patients), size=(BOOTSTRAP_REPLICATES, len(patients)))
    bootstrap = deltas[sample].mean(axis=1)
    return {
        "patient_count": len(patients),
        "control": float(control.mean()),
        "candidate": float(candidate.mean()),
        "delta": float(deltas.mean()),
        "ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "replicates": BOOTSTRAP_REPLICATES,
        "cluster_unit": "patient",
        "seed": seed,
    }


def _validate_paired_gate_intervals(
    paired: Mapping[str, Mapping[str, object]],
) -> None:
    missing = tuple(
        key
        for key, value in paired.items()
        if not isinstance(value.get("ci95"), list) or len(value["ci95"]) != 2
    )
    if missing:
        raise RuntimeError(
            "Paired bootstrap produced no gated confidence interval for: "
            + ", ".join(missing)
        )


def _joint_noninferior_fold_count(
    fold_rows: Sequence[Mapping[str, object]],
) -> int:
    """Count folds where BCE and Brier are jointly no worse."""

    count = 0
    for row in fold_rows:
        control = row.get("control")
        candidate = row.get("candidate")
        if not isinstance(control, Mapping) or not isinstance(candidate, Mapping):
            raise TypeError("Fold result must contain control/candidate mappings")
        if (
            float(candidate["bce"]) <= float(control["bce"])
            and float(candidate["brier"]) <= float(control["brier"])
        ):
            count += 1
    return count


def _preflight() -> dict[str, object]:
    if _file_sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise ValueError("Frozen BUNDL protocol changed")
    ledger, ledger_sha256 = _load_prelaunch_ledger()
    union_payload = _load_pinned_json(PUBLIC_UNION, PUBLIC_UNION_SHA256)
    union_identities = _public_union_identities(union_payload)
    identity_payload = _load_pinned_json(
        IDENTITY_FIREWALL, IDENTITY_FIREWALL_SHA256
    )
    preprocessing = load_preprocessing_selection_capability(
        PREPROCESSING_SELECTION,
        expected_artifact_sha256=PREPROCESSING_SELECTION_SHA256,
        expected_protocol_receipt_sha256=PREPROCESSING_PROTOCOL_SHA256,
    )
    manifest = load_tusz_ictal_training_manifest(
        MASTER_MANIFEST,
        expected_bundle_manifest_sha256=MASTER_BUNDLE_SHA256,
        expected_source_manifest_sha256=MASTER_SOURCE_SHA256,
    )
    if len(manifest.patient_ids) != 129:
        raise ValueError("TUSZ master patient count changed")
    deepsoz, dev, gate = _load_identity_firewall(
        identity_payload, manifest.patient_ids
    )
    master_union_overlap = tuple(sorted(set(manifest.patient_ids) & set(union_identities)))
    if master_union_overlap != deepsoz:
        raise ValueError("Full 102-identity union disagrees with the identity-only firewall")
    auxiliary = tuple(sorted(set(manifest.patient_ids) - set(deepsoz)))
    fit = tuple(sorted(set(auxiliary) - set(dev) - set(gate)))
    if len(auxiliary) != 64 or len(fit) != 40 or set(fit) & set(union_identities):
        raise ValueError("Source-only 64/40 identity firewall failed")
    fit_event_count = sum(len(manifest.events_for_patient(patient)) for patient in fit)
    if fit_event_count != 105:
        raise ValueError("Source-only fit event count changed")

    corpus = load_formal_token_corpus(
        TOKEN_CORPUS,
        expected_index_sha256=TOKEN_CORPUS_INDEX_SHA256,
        preprocessing_selection=preprocessing,
    )
    snapshot = load_verified_ictal_target_snapshot(
        TARGET_SNAPSHOT,
        expected_manifest_sha256=TARGET_SNAPSHOT_MANIFEST_SHA256,
        expected_receipt_sha256=TARGET_SNAPSHOT_RECEIPT_SHA256,
    )
    master_counts = _label_counts(
        snapshot.training_targets, snapshot.training_target_mask
    )
    master_core = bundl_clean_core_mask(
        snapshot.training_targets,
        snapshot.training_target_mask,
        radius_seconds=BUNDL_CLEAN_CORE_RADIUS_SECONDS,
    )
    master_core_counts = _label_counts(snapshot.training_targets, master_core)
    if master_counts != EXPECTED_MASTER_COUNTS:
        raise ValueError(f"Master observed denominator changed: {master_counts}")
    if master_core_counts != EXPECTED_MASTER_CORE_COUNTS:
        raise ValueError(f"Master clean-core denominator changed: {master_core_counts}")

    fit_rows = [
        index
        for index, (_, patient) in enumerate(snapshot.training_event_rows)
        if patient in set(fit)
    ]
    if len(fit_rows) != fit_event_count:
        raise ValueError("Fit target snapshot event roster changed")
    fit_targets = snapshot.training_targets[fit_rows]
    fit_mask = snapshot.training_target_mask[fit_rows]
    fit_core = bundl_clean_core_mask(
        fit_targets, fit_mask, radius_seconds=BUNDL_CLEAN_CORE_RADIUS_SECONDS
    )
    fit_counts = _label_counts(fit_targets, fit_mask)
    fit_core_counts = _label_counts(fit_targets, fit_core)
    if fit_counts != EXPECTED_FIT_COUNTS:
        raise ValueError(f"Fit observed denominator changed: {fit_counts}")
    if fit_core_counts != EXPECTED_FIT_CORE_COUNTS:
        raise ValueError(f"Fit clean-core denominator changed: {fit_core_counts}")

    dataset = build_tusz_ictal_token_bag_dataset_from_target_snapshot(
        manifest, corpus, snapshot, patient_ids=fit
    )
    dataset = _memoized_dataset(dataset, fit)
    discrimination_patient_count = 0
    for bag in dataset.iter_epoch():
        observed = bag.targets[bag.target_mask]
        if bool((observed == 1).any()) and bool((observed == 0).any()):
            discrimination_patient_count += 1
    if discrimination_patient_count != 37:
        raise ValueError(
            "Fit patient-level discrimination denominator changed: "
            f"{discrimination_patient_count}"
        )
    first_bag = next(dataset.iter_epoch())
    if tuple(first_bag.token_events[0].tokens.shape) != (19, 60, 200):
        raise ValueError("Cached LaBraM input shape changed")
    cores = _core_by_patient(dataset)
    if sum(int(value.sum().item()) for value in cores.values()) != EXPECTED_FIT_CORE_COUNTS[0]:
        raise ValueError("Memoized clean-core count differs from the target snapshot")

    if (
        BUNDL_DROPOUT_PROBABILITY != 0.2
        or BUNDL_MC_SAMPLES != 10
        or BUNDL_NEGATIVE_LABEL_UNRELIABILITY != 0.001
        or BUNDL_PROBABILITY_FLOOR != 0.001
        or BUNDL_PROBABILITY_CEILING != 0.999
    ):
        raise ValueError("Frozen BUNDL constants changed")
    head_probe = BundlDropoutK31IctalHead()
    trainable_names = tuple(
        name for name, parameter in head_probe.named_parameters() if parameter.requires_grad
    )
    trainable_count = sum(
        parameter.numel() for parameter in head_probe.parameters() if parameter.requires_grad
    )
    if head_probe.context_seconds != 31 or trainable_count != EXPECTED_HEAD_TRAINABLE_PARAMETERS:
        raise ValueError("Frozen k31 head contract changed")
    if any(
        token in name.lower()
        for name in trainable_names
        for token in ("foundation", "backbone", "encoder", "labram")
    ):
        raise ValueError("A foundation parameter entered the BUNDL head")

    folds = _fixed_folds(fit)
    for fold, held in folds.items():
        held_bags = list(dataset.iter_subset(held))
        held_targets = torch.cat([bag.targets for bag in held_bags], dim=0)
        held_mask = torch.cat([bag.target_mask for bag in held_bags], dim=0)
        _, positive, negative = _label_counts(held_targets, held_mask)
        if positive < 1 or negative < 1:
            raise ValueError(f"Fold {fold} lacks a positive or negative target")
    return {
        "manifest": manifest,
        "corpus": corpus,
        "snapshot": snapshot,
        "dataset": dataset,
        "core_masks": cores,
        "fit_patients": fit,
        "deepsoz_union_identities": union_identities,
        "deepsoz_master_overlap": deepsoz,
        "i_dev_patients": dev,
        "i_gate_patients": gate,
        "folds": folds,
        "receipt": {
            "schema_version": "soz_labram_bundl_source_only_preflight_v1",
            "passed": True,
            "master_patients": len(manifest.patient_ids),
            "deepsoz_union_identity_count": len(union_identities),
            "deepsoz_master_overlap_excluded": len(deepsoz),
            "auxiliary_patients_after_deepsoz_exclusion": len(auxiliary),
            "historical_i_dev_excluded": len(dev),
            "historical_i_gate_excluded": len(gate),
            "fit_oof_patients": len(fit),
            "fit_oof_events": fit_event_count,
            "fit_discrimination_patients": discrimination_patient_count,
            "master_observed_counts": list(master_counts),
            "master_clean_core_counts": list(master_core_counts),
            "fit_observed_counts": list(fit_counts),
            "fit_clean_core_counts": list(fit_core_counts),
            "input_shape_per_event": [19, 60, 200],
            "logit_shape_per_event": [20, 60, 1],
            "temporal_resolution_seconds": 1,
            "deepsoz_target_values_loaded": False,
            "deepsoz_target_values_used": False,
            "private_inputs_loaded": False,
            "private_inputs_used": False,
            "historical_i_dev_or_gate_outcomes_loaded": False,
            "historical_i_dev_or_gate_outcomes_used": False,
            "missing_labels_imputed_as_negative": False,
            "foundation_optimizer_parameter_count": 0,
            "head_trainable_parameter_count": trainable_count,
            "k31_context_seconds": head_probe.context_seconds,
            "dropout_probability": BUNDL_DROPOUT_PROBABILITY,
            "mc_samples": BUNDL_MC_SAMPLES,
            "negative_label_unreliability": BUNDL_NEGATIVE_LABEL_UNRELIABILITY,
            "label_probability_floor": BUNDL_PROBABILITY_FLOOR,
            "label_probability_ceiling": BUNDL_PROBABILITY_CEILING,
            "protocol_sha256": PROTOCOL_SHA256,
            "prelaunch_ledger_sha256": ledger_sha256,
            "public_union_sha256": PUBLIC_UNION_SHA256,
            "identity_firewall_sidecar_sha256": IDENTITY_FIREWALL_SHA256,
            "identity_source_split_sha256": identity_payload["source_split_sha256"],
            "source_file_sha256": ledger["code_sha256"],
        },
    }


def _run(device: torch.device, output: Path) -> dict[str, object]:
    loaded = _preflight()
    dataset = loaded["dataset"]
    core_masks = loaded["core_masks"]
    fit_patients = loaded["fit_patients"]
    folds = loaded["folds"]
    assert isinstance(dataset, IctalTokenBagDataset)
    assert isinstance(core_masks, Mapping)
    assert isinstance(fit_patients, tuple)
    assert isinstance(folds, Mapping)

    all_control: dict[str, dict[str, object]] = {}
    all_candidate: dict[str, dict[str, object]] = {}
    fold_rows: list[dict[str, object]] = []
    state_tensors: dict[str, torch.Tensor] = {}
    config = IctalTrainingConfig(seed=SEED)
    if device.type == "cuda":
        validate_ictal_cuda_environment()
    execution = None
    with ictal_determinism_runtime(config, execution_device_type=device.type):
        execution = _execution_receipt(device)
        output.mkdir()
        for fold in range(OUTER_FOLDS):
            held = tuple(folds[fold])
            train = tuple(sorted(set(fit_patients) - set(held)))
            control, candidate, training = _fit_fold(
                dataset,
                core_masks,
                train,
                fold=fold,
                device=device,
            )
            control_predictions = _predict_patients(
                control, dataset, held, device=device
            )
            candidate_predictions = _predict_patients(
                candidate, dataset, held, device=device
            )
            control_patient_rows = {}
            candidate_patient_rows = {}
            for patient in held:
                control_row = control_predictions[patient]
                candidate_row = candidate_predictions[patient]
                if control_row["event_ids"] != candidate_row["event_ids"]:
                    raise RuntimeError("Matched arms produced different event rosters")
                targets = control_row["targets"]
                mask = control_row["mask"]
                if not torch.equal(targets, candidate_row["targets"]) or not torch.equal(
                    mask, candidate_row["mask"]
                ):
                    raise RuntimeError("Matched arms do not share hard targets/masks")
                core = core_masks[patient]
                control_patient_rows[patient] = _patient_metric_row(
                    control_row["logits"], targets, mask, core
                )
                candidate_patient_rows[patient] = _patient_metric_row(
                    candidate_row["logits"], targets, mask, core
                )
            fold_rows.append(
                {
                    "fold": fold,
                    "train_patient_count": len(train),
                    "held_patient_count": len(held),
                    "held_patient_ids": list(held),
                    "training": training,
                    "control": _macro_summary(list(control_patient_rows.values())),
                    "candidate": _macro_summary(list(candidate_patient_rows.values())),
                }
            )
            all_control.update(control_predictions)
            all_candidate.update(candidate_predictions)
            for name, tensor in control.state_dict().items():
                state_tensors[f"fold{fold}.control.{name}"] = tensor.detach().cpu().contiguous()
            for name, tensor in candidate.state_dict().items():
                state_tensors[f"fold{fold}.candidate.{name}"] = tensor.detach().cpu().contiguous()
            del control, candidate
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not isinstance(execution, dict):
        raise RuntimeError("Formal CUDA execution receipt was not materialized")

    if set(all_control) != set(fit_patients) or set(all_candidate) != set(fit_patients):
        raise RuntimeError("OOF predictions do not cover the fixed 40-patient cohort")
    control_patient_rows = {}
    candidate_patient_rows = {}
    event_ids: list[str] = []
    event_patient_indices: list[int] = []
    control_logits: list[torch.Tensor] = []
    candidate_logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    cores: list[torch.Tensor] = []
    for patient_index, patient in enumerate(fit_patients):
        control_row = all_control[patient]
        candidate_row = all_candidate[patient]
        patient_core = core_masks[patient]
        control_patient_rows[patient] = _patient_metric_row(
            control_row["logits"], control_row["targets"], control_row["mask"], patient_core
        )
        candidate_patient_rows[patient] = _patient_metric_row(
            candidate_row["logits"], candidate_row["targets"], candidate_row["mask"], patient_core
        )
        patient_events = tuple(control_row["event_ids"])
        event_ids.extend(patient_events)
        event_patient_indices.extend([patient_index] * len(patient_events))
        control_logits.append(control_row["logits"])
        candidate_logits.append(candidate_row["logits"])
        targets.append(control_row["targets"])
        masks.append(control_row["mask"])
        cores.append(patient_core)

    paired = {
        key: _paired_bootstrap(
            control_patient_rows,
            candidate_patient_rows,
            key,
            seed=SEED,
        )
        for key in PAIRED_METRIC_KEYS
    }
    _validate_paired_gate_intervals(paired)
    bce_fold_passes = sum(
        float(row["candidate"]["bce"]) <= float(row["control"]["bce"])
        for row in fold_rows
    )
    brier_fold_passes = sum(
        float(row["candidate"]["brier"]) <= float(row["control"]["brier"])
        for row in fold_rows
    )
    joint_fold_passes = _joint_noninferior_fold_count(fold_rows)

    def lower(key: str) -> float:
        return float(paired[key]["ci95"][0])

    def upper(key: str) -> float:
        return float(paired[key]["ci95"][1])

    gates = {
        "bce_point_improves": float(paired["bce"]["delta"]) < 0.0,
        "bce_ci_upper_nonpositive": upper("bce") <= 0.0,
        "brier_point_improves": float(paired["brier"]["delta"]) < 0.0,
        "brier_ci_upper_nonpositive": upper("brier") <= 0.0,
        "auroc_noninferiority": lower("auroc") >= -0.01,
        "average_precision_noninferiority": lower("average_precision") >= -0.01,
        "sensitivity_noninferiority": lower("sensitivity_at_0_5") >= -0.02,
        "false_positive_rate_noninferiority": upper("false_positive_rate_at_0_5") <= 0.02,
        "bce_and_brier_jointly_noninferior_at_least_four_of_five_folds": (
            joint_fold_passes >= 4
        ),
        "identity_firewall": True,
        "missing_mask_preserved": True,
        "foundation_optimizer_parameter_count_zero": True,
    }
    passed = all(gates.values())

    prediction_tensors = {
        "control_oof_logits": torch.cat(control_logits, dim=0).contiguous(),
        "candidate_oof_logits": torch.cat(candidate_logits, dim=0).contiguous(),
        "hard_targets": torch.cat(targets, dim=0).contiguous(),
        "target_mask": torch.cat(masks, dim=0).contiguous(),
        "clean_core_mask": torch.cat(cores, dim=0).contiguous(),
        "patient_index": torch.tensor(event_patient_indices, dtype=torch.int64),
    }
    if not torch.isfinite(prediction_tensors["control_oof_logits"]).all() or not torch.isfinite(
        prediction_tensors["candidate_oof_logits"]
    ).all():
        raise ValueError("Final OOF tensor is non-finite")
    prediction_path = output / "oof_predictions.safetensors"
    state_path = output / "outer_fold_states.safetensors"
    save_file(prediction_tensors, str(prediction_path))
    save_file(state_tensors, str(state_path))

    result = {
        "schema_version": SCHEMA,
        "status": _internal_result_status(passed),
        "final_qualification": False,
        "trace_audit_required": True,
        "trace_audit_completed": False,
        "final_qualification_authority": FINAL_AUDIT_RELATIVE_PATH,
        "candidate": "labram_k31_dropout_0_2_bundl_edge_time_extension",
        "matched_control": "labram_k31_dropout_0_2_hard_bce",
        "official_bundl_reproduction": False,
        "extension_semantics": "tusz_bipolar_edge_time_visible_ictal_involvement_not_soz",
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": PROTOCOL_SHA256,
        "prelaunch_ledger_sha256": loaded["receipt"]["prelaunch_ledger_sha256"],
        "preflight": loaded["receipt"],
        "training_contract": {
            "seed": SEED,
            "outer_folds": OUTER_FOLDS,
            "warmup_epochs_clean_core_hard_bce": WARMUP_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "dropout_probability": BUNDL_DROPOUT_PROBABILITY,
            "mc_samples": BUNDL_MC_SAMPLES,
            "negative_label_unreliability": BUNDL_NEGATIVE_LABEL_UNRELIABILITY,
            "clean_core_radius_seconds": BUNDL_CLEAN_CORE_RADIUS_SECONDS,
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "gradient_clip": 1.0,
            "event_microbatch": EVENT_MICROBATCH,
            "checkpoint_selection": "fixed_final_epoch_no_target_validation",
            "foundation_policy": "frozen_cached_tokens_no_foundation_optimizer",
        },
        "execution_receipt": execution,
        "control_oof": _macro_summary(list(control_patient_rows.values())),
        "candidate_oof": _macro_summary(list(candidate_patient_rows.values())),
        "paired_candidate_minus_control": paired,
        "fold_direction_counts": {
            "bce_noninferior_folds": bce_fold_passes,
            "brier_noninferior_folds": brier_fold_passes,
            "bce_and_brier_jointly_noninferior_folds": joint_fold_passes,
        },
        "folds": fold_rows,
        "gates": gates,
        "all_gates_passed": passed,
        "event_ids": event_ids,
        "fit_patient_ids": list(fit_patients),
        "prediction_file": {
            "filename": prediction_path.name,
            "sha256": _file_sha256(prediction_path),
        },
        "state_file": {
            "filename": state_path.name,
            "sha256": _file_sha256(state_path),
        },
        "deepsoz_target_values_loaded": False,
        "deepsoz_target_values_used": False,
        "private_inputs_loaded": False,
        "private_inputs_used": False,
        "historical_i_dev_or_gate_outcomes_loaded": False,
        "historical_i_dev_or_gate_outcomes_used": False,
        "formal_soz_promotion": False,
        "checkpoint_authorized_for_soz_reasoner": False,
        "claim_boundary": (
            "Passing qualifies only a source-native TUSZ edge-time involvement producer; "
            "it does not change or validate SOZ localization."
        ),
    }
    (output / "result.json").write_bytes(_canonical_json_bytes(result))
    print(
        json.dumps(
            {
                "status": result["status"],
                "all_gates_passed": passed,
                "output": str(output),
                "control_oof": result["control_oof"],
                "candidate_oof": result["candidate_oof"],
                "paired_candidate_minus_control": paired,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    _validate_formal_runner_argv(
        preflight_only=bool(args.preflight_only), raw_argv=raw_argv
    )
    device = torch.device(args.device)
    _validate_formal_launch(
        preflight_only=bool(args.preflight_only),
        device=device,
        output=args.output_directory,
    )
    if not args.preflight_only and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    if args.preflight_only:
        loaded = _preflight()
        print(json.dumps(loaded["receipt"], sort_keys=True), flush=True)
        return 0
    output = _safe_new_output(args.output_directory)
    _run(device, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
