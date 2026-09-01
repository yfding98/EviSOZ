#!/usr/bin/env python3
"""Run the frozen global-I-gated H+V positive-set endpoint residual v10.

The runner is deliberately source-train-only.  It consumes the already
frozen 65-patient/582-event LaBraM prefix, I/V evidence, DeepSOZ target scope,
five patient folds, and ``temporal_mil_exact`` OOF anchor.  There is one
prespecified candidate and two label-mask sensitivity refits; no development,
evaluation, private, or hyperparameter-selection port exists in the CLI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import stat
import sys
import tempfile
from typing import Callable, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_labram_endpoint_aligned_peft_oof_v8 import (  # noqa: E402
    DEFAULT_PREFIX_CACHE,
    DEFAULT_PREFIX_CACHE_MANIFEST_SHA256,
    DEFAULT_SOURCE_TRAIN_IV,
    DEFAULT_SOURCE_TRAIN_IV_MANIFEST_SHA256,
    DEFAULT_TARGET_SCOPE,
    TEMPORAL_ANCHOR,
    V7_COMPARATOR_PATH,
    V7_MANIFEST_SHA256,
    V7_PREDICTION_SHA256,
    _load_access_audit,
    _load_fixed_comparators,
    _load_inputs,
)
from scripts.run_labram_temporal_mil_nested_oof_v1 import (  # noqa: E402
    _file_sha256,
    _metrics,
    _tensor_state_sha256,
)
from scripts.run_labram_v_directed_endpoint_oof_v5 import (  # noqa: E402
    BASE_SEED as PAIRED_BOOTSTRAP_SEED,
    BOOTSTRAP_REPLICATES as PAIRED_BOOTSTRAP_REPLICATES,
    _paired_patient_bootstrap,
    _top1_states,
    _transition_diagnostic,
)
from src.soz.development_reasoner_training_v1_1 import (  # noqa: E402
    FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256,
)
from src.soz.geometry import (  # noqa: E402
    CHANNEL_INDEX,
    EVOLUTION_FEATURES,
    N_STANDARD_CHANNELS,
    STANDARD_19,
    TCP_20_EDGES,
)
from src.soz.positive_set_endpoint_residual import (  # noqa: E402
    EARLY_TILE_INDICES,
    POSITIVE_SET_ENDPOINT_FEATURE_DIM,
    POSITIVE_SET_ENDPOINT_L2_WEIGHT,
    POSITIVE_SET_ENDPOINT_RESIDUAL_SCHEMA,
    EarlyPreEndpointContrasts,
    PositiveSetEndpointFeatureBatch,
    PositiveSetEndpointFeatureState,
    PositiveSetEndpointResidual,
    SharedEarlyIGate,
    TargetMaskSensitivityKind,
    build_early_pre_endpoint_contrasts,
    build_shared_early_i_gate,
    fit_fold_positive_set_endpoint_features,
    positive_set_endpoint_objective,
    positive_set_endpoint_sensitivity_objective,
    transform_positive_set_endpoint_features,
)


SCHEMA_VERSION = "soz_labram_global_i_gated_positive_set_endpoint_oof_v10"
CANDIDATE_NAME = "global_i_gated_hv_positive_set_endpoint_residual_v10"
PROTOCOL_PATH = (
    ROOT
    / "research/02_method/"
    "labram_global_i_gated_positive_set_endpoint_residual_protocol_v10_20260811_zh.md"
)
MODULE_PATH = ROOT / "src/soz/positive_set_endpoint_residual.py"
RUNNER_PATH = Path(__file__).resolve()
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/labram_global_i_gated_positive_set_endpoint_oof_v10_20260811"
)
AUTHORIZATION_PATH = (
    ROOT
    / "outputs/labram_global_i_gated_positive_set_endpoint_v10_authorization_20260811.json"
)
LAUNCH_LEDGER_PATH = (
    ROOT
    / "outputs/labram_global_i_gated_positive_set_endpoint_oof_v10_launch_ledger_20260811.json"
)
AUTHORIZATION_SCHEMA = (
    "soz_labram_global_i_gated_positive_set_endpoint_v10_authorization_v1"
)
LAUNCH_LEDGER_SCHEMA = (
    "soz_labram_global_i_gated_positive_set_endpoint_v10_launch_ledger_v1"
)
AUTHORIZED_SCOPE = "source_train_exploratory_mechanism_oof_only"
DIRECT_RUNTIME_SCRIPT_PATHS = (
    ROOT / "scripts/run_labram_endpoint_aligned_peft_oof_v8.py",
    ROOT / "scripts/run_labram_temporal_mil_nested_oof_v1.py",
    ROOT / "scripts/run_labram_v_directed_endpoint_oof_v5.py",
)

OUTER_FOLDS = tuple(range(5))
OBJECTIVES: tuple[str, ...] = (
    "main",
    "known_positive_dropout",
    "one_zero_to_unknown_jackknife",
)
SENSITIVITY_KINDS: Mapping[str, TargetMaskSensitivityKind] = {
    "known_positive_dropout": "known_positive_to_unknown",
    "one_zero_to_unknown_jackknife": "observed_zero_to_unknown",
}
LBFGS_MAX_ITER = 200
LBFGS_MAX_EVAL = 250
LBFGS_HISTORY_SIZE = 50
FLIP_LOGIT_MARGIN = math.log(3.0)
MIN_PAIR_EVENT_COUNT = 2
ANCHOR_GAP_Z_MAX = 1.0
EXPECTED_PATIENT_COUNT = 65
EXPECTED_EVENT_COUNT = 582
EXPECTED_ANCHOR_STRICT_HITS = 42.0
EXPECTED_ANCHOR_RELAXED_HITS = 55.0
EXPECTED_HIT_ATOL = 1e-5
NONINFERIORITY_ATOL = 1e-8

K31_SCORE_MANIFEST = (
    ROOT / "outputs/labram_k31_development_scores_v1_2_20260810/manifest.json"
)
K31_SCORE_MANIFEST_SHA256 = (
    "9fad486f99bdd7fda706045918c83bd2608c091653f34bbe6a9cb62f7bfbfa66"
)
VAQ_MANIFEST = ROOT / "outputs/deepsoz_source_train_oof_vaq_v1_20260810/manifest.json"
VAQ_MANIFEST_SHA256 = (
    "40845ec7115d4f7c43de28ded0cb510843f26ec1907c5432e6a2026ebfde92c2"
)
H_CROSSWALK_RECEIPT = (
    ROOT / "outputs/labram_frozen_h_source_train_crosswalk_v1_20260810/receipt.json"
)
H_CROSSWALK_RECEIPT_SHA256 = (
    "4eec735065d93f761c1e17753977fe1f0e633d1fdbb6c6888f0af4eb78f6bbee"
)
TARGET_SCOPE_RECEIPT_SHA256 = (
    "90529bb91df657a27f52d82300ce13431c94d4a4b76f28691bea59eeddcde361"
)
EXPECTED_EVENT_ORDER_SHA256 = (
    "c45fe14fc4cdc1767710aa5bc22b3dce4cb08caa340f9e99a035bf134e59d434"
)
EXPECTED_I_TENSOR_SHA256 = (
    "279518be479d6106aaa8b3561c2082b35b1f107606d4d5349760d807d6ef8f98"
)
EXPECTED_V_TENSOR_SHA256 = (
    "f3c4ef270d1995ab1e79019474dabb778062fbf945512d6bddd390c2aa62379f"
)
EXPECTED_V_MASK_TENSOR_SHA256 = (
    "93bedbd38ee8d0354acea8f14d2f926e53b99e1b5d4662769d81f819df330097"
)
EXPECTED_LEGACY_POSITION_NAMES = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ",
    "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2",
)

# DeepSOZ's frozen source-train target carrier evaluates 18 electrodes, with
# PZ unavailable.  Define this from the ontology rather than inspecting a
# held patient's label values during proposal generation.
FIXED_EVALUABLE_CHANNEL_MASK = torch.tensor(
    [channel != "PZ" for channel in STANDARD_19], dtype=torch.bool
)

_SHA256_HEX = frozenset("0123456789abcdef")


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return text


def _absolute_no_symlink(path: str | Path, *, field: str) -> Path:
    result = Path(os.path.abspath(path))
    for component in (result, *result.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot traverse symlinks")
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_bytes_no_symlink(
    path: str | Path,
    *,
    field: str,
    max_bytes: int | None = None,
) -> tuple[Path, bytes]:
    resolved = _absolute_no_symlink(path, field=field)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"{field} must be a regular file")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if max_bytes is not None and size > max_bytes:
                raise ValueError(f"{field} is unexpectedly large")
    finally:
        os.close(descriptor)
    return resolved, b"".join(chunks)


@dataclass
class LaunchLedgerLease:
    """Exclusive process lease over the immutable launch ledger."""

    path: Path
    sha256: str
    launch_mode: str
    descriptor: int

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> "LaunchLedgerLease":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _configure_deterministic_runtime() -> None:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise RuntimeError(
                "v10 requires one interop thread before any parallel work starts"
            ) from exc
    torch.use_deterministic_algorithms(True)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("v10 deterministic CPU thread contract failed")
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("v10 deterministic algorithm contract failed")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _scope_sha256(values: Sequence[object]) -> str:
    return hashlib.sha256(_canonical_bytes(list(values))).hexdigest()


def _runtime_dependency_binding() -> dict[str, object]:
    """Hash the conservative local Python dependency closure and runtime."""

    candidates = set(DIRECT_RUNTIME_SCRIPT_PATHS)
    candidates.add(ROOT / "src/__init__.py")
    scripts_init = ROOT / "scripts/__init__.py"
    if scripts_init.exists():
        candidates.add(scripts_init)
    candidates.update((ROOT / "src/soz").rglob("*.py"))
    entries: list[dict[str, str]] = []
    for path in sorted(candidates, key=lambda value: str(value.relative_to(ROOT))):
        resolved, raw = _read_regular_bytes_no_symlink(
            path, field="runtime dependency"
        )
        entries.append(
            {
                "path": str(resolved.relative_to(ROOT)),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return {
        "bundle_policy": (
            "sorted_relative_path_and_raw_sha256_of_direct_imported_runners_"
            "plus_all_src_soz_python_v1"
        ),
        "file_count": len(entries),
        "bundle_sha256": hashlib.sha256(_canonical_bytes(entries)).hexdigest(),
        "entries": entries,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_build_config_sha256": hashlib.sha256(
            torch.__config__.show().encode("utf-8")
        ).hexdigest(),
        "safetensors_version": importlib_metadata.version("safetensors"),
    }


def _frozen_input_bindings(
    *,
    lineage: Mapping[str, object],
    frozen_lineage: Mapping[str, object],
) -> dict[str, object]:
    """Return the closed set of immutable signal/target/anchor inputs."""

    return {
        "prefix_cache_manifest_sha256": _require_sha256(
            lineage["prefix_cache_manifest_sha256"],
            field="prefix_cache_manifest_sha256",
        ),
        "prefix_cache_event_order_sha256": _require_sha256(
            lineage["prefix_cache_event_order_sha256"],
            field="prefix_cache_event_order_sha256",
        ),
        "source_train_iv_manifest_sha256": _require_sha256(
            lineage["source_train_iv_manifest_sha256"],
            field="source_train_iv_manifest_sha256",
        ),
        "source_train_iv_receipt_sha256": _require_sha256(
            lineage["source_train_iv_receipt_sha256"],
            field="source_train_iv_receipt_sha256",
        ),
        "source_train_target_receipt_sha256": _require_sha256(
            lineage["source_train_target_receipt_sha256"],
            field="source_train_target_receipt_sha256",
        ),
        "event_order_sha256": _require_sha256(
            frozen_lineage["event_order_sha256"], field="event_order_sha256"
        ),
        "k31_score_manifest_sha256": _require_sha256(
            frozen_lineage["k31_score_manifest_sha256"],
            field="k31_score_manifest_sha256",
        ),
        "vaq_manifest_sha256": _require_sha256(
            frozen_lineage["vaq_manifest_sha256"], field="vaq_manifest_sha256"
        ),
        "h_crosswalk_receipt_sha256": _require_sha256(
            frozen_lineage["h_crosswalk_receipt_sha256"],
            field="h_crosswalk_receipt_sha256",
        ),
        "target_scope_receipt_sha256": _require_sha256(
            frozen_lineage["target_scope_receipt_sha256"],
            field="target_scope_receipt_sha256",
        ),
        "ictal_tensor_sha256": _require_sha256(
            frozen_lineage["ictal_tensor_sha256"], field="ictal_tensor_sha256"
        ),
        "ictal_patient_excluded_recovery_manifest_sha256s": {
            str(fold): _require_sha256(
                frozen_lineage[
                    "ictal_patient_excluded_recovery_manifest_sha256s"
                ][str(fold)],
                field=f"ictal recovery fold {fold}",
            )
            for fold in OUTER_FOLDS
        },
        "evolution_tensor_sha256": _require_sha256(
            frozen_lineage["evolution_tensor_sha256"],
            field="evolution_tensor_sha256",
        ),
        "evolution_mask_tensor_sha256": _require_sha256(
            frozen_lineage["evolution_mask_tensor_sha256"],
            field="evolution_mask_tensor_sha256",
        ),
        "temporal_anchor_manifest_sha256": V7_MANIFEST_SHA256,
        "temporal_anchor_prediction_sha256": V7_PREDICTION_SHA256,
    }


def _authorization_payload(
    *,
    input_bindings: Mapping[str, object],
    patient_ids: Sequence[str],
    patient_folds: Sequence[int],
    event_patient_index: Sequence[int],
) -> dict[str, object]:
    """Construct the only authorization receipt accepted by the runner."""

    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "authorization_scope": AUTHORIZED_SCOPE,
        "authorized": True,
        "supersedes_legacy_development_only_for_this_scope": True,
        "legacy_artifacts_modified": False,
        "source_train_authorized": True,
        "source_dev_authorized": False,
        "source_eval_authorized": False,
        "private_authorized": False,
        "formal_reasoner_authorized": False,
        "formal_promotion_authorized": False,
        "performance_claim_authorized": False,
        "candidate_name": CANDIDATE_NAME,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": _file_sha256(PROTOCOL_PATH),
        "module_path": str(MODULE_PATH.relative_to(ROOT)),
        "module_sha256": _file_sha256(MODULE_PATH),
        "runner_path": str(RUNNER_PATH.relative_to(ROOT)),
        "runner_sha256": _file_sha256(RUNNER_PATH),
        "runtime_dependency_binding": _runtime_dependency_binding(),
        "output_directory": str(_absolute_no_symlink(DEFAULT_OUTPUT, field="default output")),
        "patient_count": EXPECTED_PATIENT_COUNT,
        "event_count": EXPECTED_EVENT_COUNT,
        "outer_folds": list(OUTER_FOLDS),
        "patient_roster_sha256": _scope_sha256(patient_ids),
        "patient_folds_sha256": _scope_sha256(patient_folds),
        "event_patient_index_sha256": _scope_sha256(event_patient_index),
        "standard_19_sha256": _scope_sha256(STANDARD_19),
        "input_bindings": dict(input_bindings),
    }


def _load_v10_authorization(
    path: str | Path,
    *,
    expected_payload: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    """Validate an exact, closed-schema superseding authorization receipt."""

    authorization = _absolute_no_symlink(path, field="v10 authorization receipt")
    if authorization.name in {"", ".", ".."}:
        raise ValueError("v10 authorization receipt requires a concrete file")
    authorization, raw = _read_regular_bytes_no_symlink(
        authorization,
        field="v10 authorization receipt",
        max_bytes=1024 * 1024,
    )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("v10 authorization receipt is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError("v10 authorization receipt must contain an object")
    expected = dict(expected_payload)
    if payload != expected:
        missing = sorted(set(expected) - set(payload))
        extra = sorted(set(payload) - set(expected))
        changed = sorted(
            key
            for key in set(expected) & set(payload)
            if payload[key] != expected[key]
        )
        raise ValueError(
            "v10 authorization receipt does not exactly bind this launch: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return payload, hashlib.sha256(raw).hexdigest()


def _launch_ledger_payload(
    *,
    authorization_sha256: str,
    authorization_payload: Mapping[str, object],
) -> dict[str, object]:
    """Build the deterministic immutable launch-consumption record."""

    return {
        "schema_version": LAUNCH_LEDGER_SCHEMA,
        "authorization_scope": AUTHORIZED_SCOPE,
        "state": "consumed_before_held_label_metrics",
        "launch_consumed": True,
        "ledger_created_before_held_label_metrics": True,
        "deterministic_exact_replay_only_after_runtime_failure": True,
        "authorization_path": str(
            _absolute_no_symlink(AUTHORIZATION_PATH, field="authorization path")
        ),
        "authorization_sha256": _require_sha256(
            authorization_sha256, field="authorization_sha256"
        ),
        "candidate_name": CANDIDATE_NAME,
        "protocol_sha256": authorization_payload["protocol_sha256"],
        "module_sha256": authorization_payload["module_sha256"],
        "runner_sha256": authorization_payload["runner_sha256"],
        "runtime_dependency_binding": authorization_payload[
            "runtime_dependency_binding"
        ],
        "output_directory": authorization_payload["output_directory"],
        "patient_count": authorization_payload["patient_count"],
        "event_count": authorization_payload["event_count"],
        "outer_folds": authorization_payload["outer_folds"],
        "patient_roster_sha256": authorization_payload["patient_roster_sha256"],
        "patient_folds_sha256": authorization_payload["patient_folds_sha256"],
        "event_patient_index_sha256": authorization_payload[
            "event_patient_index_sha256"
        ],
        "standard_19_sha256": authorization_payload["standard_19_sha256"],
        "input_bindings": authorization_payload["input_bindings"],
        "frozen_selection_policy": {
            "pre_tile_indices": [0, 1, 2],
            "early_tile_indices": [3, 4, 5],
            "tile_seconds": 4,
            "shared_i_gate": "mean_across_all_tcp20_edges_then_early_softmax",
            "h_pca_components": 8,
            "v_feature_dimension": 6,
            "node_feature_dimension": POSITIVE_SET_ENDPOINT_FEATURE_DIM,
            "tcp20_electrode_edges": [list(edge) for edge in TCP_20_EDGES],
            "fixed_evaluable_channel_mask": FIXED_EVALUABLE_CHANNEL_MASK.tolist(),
            "flip_logit_margin": FLIP_LOGIT_MARGIN,
            "minimum_pair_event_count": MIN_PAIR_EVENT_COUNT,
            "anchor_gap_z_max": ANCHOR_GAP_Z_MAX,
            "loeo_stability_required": True,
            "maximum_swaps_per_patient": 1,
        },
        "optimization": {
            "objective_names": list(OBJECTIVES),
            "l2_weight": POSITIVE_SET_ENDPOINT_L2_WEIGHT,
            "lbfgs_lr": 1.0,
            "lbfgs_max_iter": LBFGS_MAX_ITER,
            "lbfgs_max_eval": LBFGS_MAX_EVAL,
            "lbfgs_history_size": LBFGS_HISTORY_SIZE,
            "lbfgs_tolerance_grad": 1e-7,
            "lbfgs_tolerance_change": 1e-9,
            "lbfgs_line_search": "strong_wolfe",
            "initialization": "exact_zero",
            "device": "cpu",
            "dtype": "float32",
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "torch_deterministic_algorithms": True,
        },
        "bootstrap": {
            "replicates": PAIRED_BOOTSTRAP_REPLICATES,
            "seed": PAIRED_BOOTSTRAP_SEED,
            "interval": "two_sided_percentile_0.025_0.975",
            "quantile_interpolation": "torch_default_linear",
        },
        "go_no_go_thresholds": {
            "main_strict_hit_delta_minimum": 4,
            "applied_exact_to_nonexact_harm_maximum": 0,
            "strict_bootstrap_ci95_lower_strictly_above": 0.0,
            "ranking_and_relaxed_noninferiority_tolerance": NONINFERIORITY_ATOL,
            "far_error_count_nonincreasing": True,
            "strict_rescue_outer_fold_minimum": 2,
            "full_denominator_patient_count": EXPECTED_PATIENT_COUNT,
            "sensitivity_minimum_strict_gain": (
                "ceil(0.5 * main_strict_hit_delta)"
            ),
            "sensitivity_exact_harm_maximum": 0,
        },
    }


def _acquire_launch_ledger(
    ledger_path: str | Path,
    *,
    expected_payload: Mapping[str, object],
    output_directory: str | Path,
) -> LaunchLedgerLease:
    """Atomically consume the launch or acquire an exact crash-replay lease.

    The immutable file is never removed or overwritten.  A process holds an
    advisory exclusive lock until publication finishes, which rejects a
    concurrent process while still permitting an identical post-crash replay.
    """

    ledger = _absolute_no_symlink(ledger_path, field="v10 launch ledger")
    output = _absolute_no_symlink(output_directory, field="v10 formal output")
    if ledger.name in {"", ".", ".."}:
        raise ValueError("v10 launch ledger requires a concrete file")
    if not ledger.parent.is_dir() or ledger.parent.is_symlink():
        raise ValueError("v10 launch ledger parent must be a regular directory")
    if os.path.lexists(output):
        raise FileExistsError(f"formal v10 output already exists: {output}")
    temporary_prefix = f".{output.name}.tmp-"
    stale_temporary = tuple(
        sorted(
            path
            for path in output.parent.iterdir()
            if path.name.startswith(temporary_prefix)
        )
    )
    if stale_temporary:
        raise RuntimeError(
            "stale formal-output temporary path requires manual fail-closed audit: "
            f"{stale_temporary}"
        )
    raw = _canonical_bytes(dict(expected_payload))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(ledger, flags, 0o440)
    except FileExistsError:
        read_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            read_flags |= os.O_NOFOLLOW
        descriptor = os.open(ledger, read_flags)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            os.close(descriptor)
            raise RuntimeError("a v10 launch or replay is already in progress") from exc
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError("existing v10 launch ledger is not a regular file")
            existing = b""
            while True:
                chunk = os.read(descriptor, 64 * 1024)
                if not chunk:
                    break
                existing += chunk
                if len(existing) > 1024 * 1024:
                    raise ValueError("existing v10 launch ledger is unexpectedly large")
            if existing != raw:
                raise ValueError(
                    "existing v10 launch ledger differs byte-for-byte; replay denied"
                )
            if os.path.lexists(output):
                raise FileExistsError(f"formal v10 output already exists: {output}")
            stale_temporary = tuple(
                sorted(
                    path
                    for path in output.parent.iterdir()
                    if path.name.startswith(temporary_prefix)
                )
            )
            if stale_temporary:
                raise RuntimeError(
                    "exact replay denied by stale formal-output temporary path: "
                    f"{stale_temporary}"
                )
        except Exception:
            os.close(descriptor)
            raise
        return LaunchLedgerLease(
            path=ledger,
            sha256=hashlib.sha256(existing).hexdigest(),
            launch_mode="deterministic_exact_replay_after_incomplete_launch",
            descriptor=descriptor,
        )

    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("v10 launch-ledger write made no progress")
            offset += written
        os.fsync(descriptor)
        _fsync_directory(ledger.parent)
        if os.path.lexists(output):
            raise FileExistsError(f"formal v10 output already exists: {output}")
    except Exception:
        os.close(descriptor)
        raise
    return LaunchLedgerLease(
        path=ledger,
        sha256=hashlib.sha256(raw).hexdigest(),
        launch_mode="first_atomic_launch",
        descriptor=descriptor,
    )


def _load_json_with_sha256(path: Path, expected_sha256: str) -> Mapping[str, object]:
    _, raw = _read_regular_bytes_no_symlink(
        path,
        field="frozen lineage file",
        max_bytes=16 * 1024 * 1024,
    )
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"frozen lineage file changed: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"frozen lineage file must contain an object: {path}")
    return value


def _validate_frozen_lineage_contract(
    *,
    full,
    patient_folds: Sequence[int],
    event_ids: Sequence[str],
) -> dict[str, object]:
    """Prove I/V/H/target identity and leakage boundaries before OOF."""

    if _scope_sha256(event_ids) != EXPECTED_EVENT_ORDER_SHA256:
        raise ValueError("source-train event order changed")
    k31 = _load_json_with_sha256(K31_SCORE_MANIFEST, K31_SCORE_MANIFEST_SHA256)
    bindings = tuple(
        row for row in k31.get("producer_bindings", ())
        if isinstance(row, dict) and row.get("oof_fold") in OUTER_FOLDS
    )
    binding_folds = tuple(sorted(int(row["oof_fold"]) for row in bindings))
    k31_checks = {
        "five patient-excluded fold producers": binding_folds == OUTER_FOLDS,
        "source annotation targets absent at replay": (
            k31.get("source_annotation_targets_present") is False
        ),
        "source annotation coverage absent at replay": (
            k31.get("source_annotation_coverage_present") is False
        ),
        "DeepSOZ target unreachable": k31.get("deepsoz_target_values_reachable") is False,
        "deployment mask target independent": k31.get("deployment_mask_policy")
        == "all_replayed_producer_cells_available_independent_of_source_annotations",
        "retrospective context declared": k31.get("context_direction")
        == "symmetric_retrospective_not_causal",
        "each fold excludes DeepSOZ": all(
            row.get("deepsoz_target_values_reachable") is False
            and row.get("deepsoz_target_source_loaded") is False
            for row in bindings
        ),
    }
    failed = tuple(name for name, passed in k31_checks.items() if not passed)
    if failed:
        raise ValueError(f"I producer lineage contract failed: {failed}")

    vaq = _load_json_with_sha256(VAQ_MANIFEST, VAQ_MANIFEST_SHA256)
    vaq_checks = {
        "source-train split": vaq.get("model_split") == "source_train",
        "SOZ labels absent": vaq.get("contains_soz_labels") is False,
        "TUSZ channel targets absent": (
            vaq.get("contains_tusz_channel_targets_or_masks") is False
        ),
        "target vectors absent": vaq.get("target_vectors_loaded") is False,
        "other splits absent": vaq.get("source_dev_events_used") is False
        and vaq.get("source_eval_events_used") is False
        and vaq.get("private_events_used") is False,
        "feature names frozen": tuple(vaq.get("evolution_feature_names", ()))
        == tuple(EVOLUTION_FEATURES),
    }
    failed = tuple(name for name, passed in vaq_checks.items() if not passed)
    if failed:
        raise ValueError(f"V producer lineage contract failed: {failed}")

    crosswalk = _load_json_with_sha256(
        H_CROSSWALK_RECEIPT, H_CROSSWALK_RECEIPT_SHA256
    )
    events = tuple(crosswalk.get("events", ()))
    patient_by_event = tuple(
        full.patient_ids[int(index)] for index in full.event_patient_index.tolist()
    )
    fold_by_event = tuple(
        int(patient_folds[int(index)]) for index in full.event_patient_index.tolist()
    )
    crosswalk_checks = {
        "event count": len(events) == EXPECTED_EVENT_COUNT,
        "event ids": tuple(row.get("evidence_event_id") for row in events)
        == tuple(event_ids),
        "target patient ids": tuple(row.get("target_patient_id") for row in events)
        == patient_by_event,
        "fold ids": tuple(int(row.get("oof_fold", -1)) for row in events)
        == fold_by_event,
        "position names": all(
            tuple(row.get("labram_position_names", ()))
            == EXPECTED_LEGACY_POSITION_NAMES
            for row in events
        ),
        "position binding policy": all(
            row.get("labram_position_binding_policy")
            == "exact_raw_header_legacy_modern_alias_to_official_1020_id_v1"
            for row in events
        ),
        "processed shape": all(
            tuple(row.get("processed_window_shape", ())) == (19, 12000)
            for row in events
        ),
        "raw replay verified": crosswalk.get("raw_replay_verified") is True,
        "reshape policy": crosswalk.get("reshape_policy")
        == "channel_major_60_tokens_to_15_calls_x_4_slots_v1",
    }
    failed = tuple(name for name, passed in crosswalk_checks.items() if not passed)
    if failed:
        raise ValueError(f"H crosswalk lineage contract failed: {failed}")

    recovery_hashes: dict[str, str] = {}
    binding_by_fold = {int(row["oof_fold"]): row for row in bindings}
    for fold in OUTER_FOLDS:
        recovery_path = (
            ROOT
            / "outputs/labram_ictal_k31_oof_recovery_v1_2_20260810"
            / f"fold{fold}/recovery_run.json"
        )
        expected_recovery_sha = str(
            binding_by_fold[fold].get("recovery_run_manifest_sha256", "")
        )
        recovery = _load_json_with_sha256(recovery_path, expected_recovery_sha)
        held_public = set(recovery.get("held_out_exclusion_public_patient_ids", ()))
        training_public = set(recovery.get("training_public_patient_ids", ()))
        target_public = {
            str(row.get("public_patient_id"))
            for row in events
            if int(row.get("oof_fold", -1)) == fold
        }
        recovery_checks = {
            "fold identity": recovery.get("oof_fold") == fold,
            "target patients held out": target_public <= held_public,
            "training-held disjoint": not (training_public & held_public),
            "DeepSOZ unused": recovery.get("deepsoz_soz_labels_used") is False
            and recovery.get("deepsoz_target_source_loaded") is False
            and recovery.get("deepsoz_target_values_reachable") is False,
            "private unused": recovery.get("private_labels_used") is False,
            "missing TUSZ cells not negative-imputed": recovery.get(
                "missing_tusz_cells_imputed_as_negative"
            ) is False,
            "TUSZ involvement only for producer training": recovery.get(
                "tusz_ictal_involvement_targets_loaded"
            ) is True,
        }
        failed = tuple(
            name for name, passed in recovery_checks.items() if not passed
        )
        if failed:
            raise ValueError(
                f"I fold-{fold} patient-exclusion contract failed: {failed}"
            )
        recovery_hashes[str(fold)] = expected_recovery_sha

    target_receipt = _load_json_with_sha256(
        DEFAULT_TARGET_SCOPE / "receipt.json", TARGET_SCOPE_RECEIPT_SHA256
    )
    target_checks = {
        "source-train split": target_receipt.get("model_split") == "source_train",
        "patient roster": tuple(target_receipt.get("patient_ids", ()))
        == tuple(full.patient_ids),
        "standard-19 order": tuple(target_receipt.get("standard_19", ()))
        == tuple(STANDARD_19),
        "benchmark complement semantics": target_receipt.get("target_semantics")
        == "deepsoz_patient_reference_dataset_complement_v2",
        "other split targets absent": target_receipt.get(
            "other_split_target_payload_included"
        ) is False,
        "private absent": target_receipt.get("private_payload_included") is False,
    }
    failed = tuple(name for name, passed in target_checks.items() if not passed)
    if failed:
        raise ValueError(f"DeepSOZ target lineage contract failed: {failed}")

    capability_manifest = json.loads(
        (DEFAULT_SOURCE_TRAIN_IV / "manifest.json").read_text(encoding="utf-8")
    )
    tensor_specs = capability_manifest.get("tensor_specs", {})
    tensor_checks = {
        "I tensor": tensor_specs.get("ictal", {}).get("tensor_sha256")
        == EXPECTED_I_TENSOR_SHA256,
        "V tensor": tensor_specs.get("evolution", {}).get("tensor_sha256")
        == EXPECTED_V_TENSOR_SHA256,
        "V mask tensor": tensor_specs.get("evolution_mask", {}).get("tensor_sha256")
        == EXPECTED_V_MASK_TENSOR_SHA256,
    }
    failed = tuple(name for name, passed in tensor_checks.items() if not passed)
    if failed:
        raise ValueError(f"I/V tensor lineage contract failed: {failed}")
    return {
        "event_order_sha256": EXPECTED_EVENT_ORDER_SHA256,
        "k31_score_manifest_sha256": K31_SCORE_MANIFEST_SHA256,
        "vaq_manifest_sha256": VAQ_MANIFEST_SHA256,
        "h_crosswalk_receipt_sha256": H_CROSSWALK_RECEIPT_SHA256,
        "target_scope_receipt_sha256": TARGET_SCOPE_RECEIPT_SHA256,
        "ictal_tensor_sha256": EXPECTED_I_TENSOR_SHA256,
        "ictal_patient_excluded_recovery_manifest_sha256s": recovery_hashes,
        "evolution_tensor_sha256": EXPECTED_V_TENSOR_SHA256,
        "evolution_mask_tensor_sha256": EXPECTED_V_MASK_TENSOR_SHA256,
        "explicit_event_patient_fold_join": True,
        "labram_channel_major_mapping_verified": True,
    }


def _patient_indices_for_fold(
    patient_folds: Sequence[int], fold: int, *, held: bool
) -> tuple[int, ...]:
    return tuple(
        index
        for index, value in enumerate(patient_folds)
        if (int(value) == int(fold)) is held
    )


def _index_tensor(indices: Sequence[int], *, device: torch.device) -> torch.Tensor:
    return torch.tensor(tuple(int(value) for value in indices), dtype=torch.long, device=device)


def _subset_rows(value: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    return value.index_select(0, _index_tensor(indices, device=value.device))


def _subset_feature_batch(
    value: PositiveSetEndpointFeatureBatch,
    indices: Sequence[int],
) -> PositiveSetEndpointFeatureBatch:
    return PositiveSetEndpointFeatureBatch(
        values=_subset_rows(value.values, indices),
        node_mask=_subset_rows(value.node_mask, indices),
        event_counts=_subset_rows(value.event_counts, indices),
    )


def _subset_contrasts(
    value: EarlyPreEndpointContrasts,
    event_indices: torch.Tensor,
) -> EarlyPreEndpointContrasts:
    """Select events without recomputing or spatializing the shared I gate."""

    if event_indices.ndim != 1 or event_indices.dtype != torch.long:
        raise TypeError("event_indices must be long [E]")
    if event_indices.device != value.h.device:
        raise ValueError("event_indices and contrasts must share a device")
    gate = value.temporal_gate
    selected_gate = SharedEarlyIGate(
        weights=gate.weights.index_select(0, event_indices).contiguous(),
        global_support=gate.global_support.index_select(0, event_indices).contiguous(),
        early_tile_mask=gate.early_tile_mask.index_select(0, event_indices).contiguous(),
        event_valid=gate.event_valid.index_select(0, event_indices).contiguous(),
    )
    return EarlyPreEndpointContrasts(
        h=value.h.index_select(0, event_indices).contiguous(),
        v=value.v.index_select(0, event_indices).contiguous(),
        node_mask=value.node_mask.index_select(0, event_indices).contiguous(),
        temporal_gate=selected_gate,
    )


def _tcp_neighbours() -> tuple[tuple[int, ...], ...]:
    rows: list[set[int]] = [set() for _ in range(N_STANDARD_CHANNELS)]
    for left, right in TCP_20_EDGES:
        a = CHANNEL_INDEX[left]
        b = CHANNEL_INDEX[right]
        rows[a].add(b)
        rows[b].add(a)
    return tuple(tuple(sorted(row)) for row in rows)


TCP_NEIGHBOURS = _tcp_neighbours()


def _candidate_nodes(anchor_index: int, node_mask: torch.Tensor) -> tuple[int, ...]:
    """Return only the anchor and its direct TCP-20 physical neighbours."""

    if tuple(node_mask.shape) != (N_STANDARD_CHANNELS,) or node_mask.dtype != torch.bool:
        raise TypeError("node_mask must be bool [19]")
    if anchor_index < 0 or anchor_index >= N_STANDARD_CHANNELS:
        raise IndexError("anchor index is outside standard-19")
    fixed = FIXED_EVALUABLE_CHANNEL_MASK.to(node_mask.device)
    allowed = node_mask & fixed
    ordered = (anchor_index,) + TCP_NEIGHBOURS[anchor_index]
    return tuple(index for index in ordered if bool(allowed[index]))


def _unique_anchor_index(anchor_scores: torch.Tensor) -> tuple[int, bool]:
    if tuple(anchor_scores.shape) != (N_STANDARD_CHANNELS,) or not torch.isfinite(
        anchor_scores
    ).all():
        raise ValueError("anchor scores must be finite [19]")
    fixed = FIXED_EVALUABLE_CHANNEL_MASK.to(anchor_scores.device)
    row = anchor_scores.masked_fill(~fixed, -torch.inf)
    top = torch.nonzero(row == row.max(), as_tuple=False).flatten()
    return int(top[0]), int(top.numel()) == 1


def _anchor_gap_z(
    anchor_scores: torch.Tensor,
    anchor_index: int,
    candidate_index: int,
) -> float:
    fixed = FIXED_EVALUABLE_CHANNEL_MASK.to(anchor_scores.device)
    values = anchor_scores[fixed]
    scale = values.std(unbiased=False).clamp_min(1e-8)
    return float(
        ((anchor_scores[anchor_index] - anchor_scores[candidate_index]) / scale)
        .detach()
        .cpu()
    )


def _local_utility_winner(
    utility: torch.Tensor,
    anchor_index: int,
    node_mask: torch.Tensor,
) -> tuple[int, float, tuple[int, ...]] | None:
    """Deterministically rank the local TCP candidate set.

    The return margin is best minus runner-up utility.  A single available
    node has no checkable local alternative and therefore returns ``None``.
    """

    if tuple(utility.shape) != (N_STANDARD_CHANNELS,) or not torch.isfinite(
        utility
    ).all():
        raise ValueError("utility must be finite [19]")
    candidates = _candidate_nodes(anchor_index, node_mask)
    if anchor_index not in candidates or len(candidates) < 2:
        return None
    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=utility.device)
    values = utility.index_select(0, candidate_tensor)
    # Stable, ontology-index tie breaking is diagnostic only; the fixed margin
    # gate rejects every tied proposal.
    order = sorted(
        range(len(candidates)),
        key=lambda position: (-float(values[position]), candidates[position]),
    )
    if bool(values[order[0]] == values[order[1]]):
        return None
    winner = candidates[order[0]]
    margin = float(values[order[0]] - values[order[1]])
    return winner, margin, candidates


def _feature_eligible_patients(
    features: PositiveSetEndpointFeatureBatch,
    patient_indices: Sequence[int],
) -> tuple[int, ...]:
    """Exclude only patients with no target-free valid early/pre event.

    Availability is decided before looking at target values.  The production
    carrier normally makes every standard-19 node available together; if a
    future carrier has partial-node support that censors a known positive, the
    objective fails closed instead of silently performing label-based sample
    exclusion.
    """

    eligible: list[int] = []
    for patient in patient_indices:
        available = features.node_mask[int(patient)]
        if bool(available.any()):
            eligible.append(int(patient))
    return tuple(eligible)


def _objective_function(
    name: str,
) -> Callable[
    [PositiveSetEndpointResidual, PositiveSetEndpointFeatureBatch, torch.Tensor, torch.Tensor],
    object,
]:
    if name == "main":
        return positive_set_endpoint_objective
    if name not in SENSITIVITY_KINDS:
        raise ValueError(f"unknown frozen objective: {name}")
    kind = SENSITIVITY_KINDS[name]

    def objective(
        model: PositiveSetEndpointResidual,
        features: PositiveSetEndpointFeatureBatch,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
    ):
        return positive_set_endpoint_sensitivity_objective(
            model,
            features,
            targets,
            target_mask,
            kind=kind,
        )

    return objective


def _fit_residual(
    features: PositiveSetEndpointFeatureBatch,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    objective_name: str,
) -> tuple[PositiveSetEndpointResidual, dict[str, object]]:
    """Fit the only 14 parameters from exact zero initialization."""

    objective = _objective_function(objective_name)
    model = PositiveSetEndpointResidual()
    with torch.no_grad():
        model.endpoint_utility.weight.zero_()
    if model.n_trainable_parameters != POSITIVE_SET_ENDPOINT_FEATURE_DIM:
        raise RuntimeError("v10 residual parameter count changed")
    initial = objective(model, features, targets, target_mask)
    closure_count = 0
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=LBFGS_MAX_ITER,
        max_eval=LBFGS_MAX_EVAL,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=LBFGS_HISTORY_SIZE,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        nonlocal closure_count
        closure_count += 1
        optimizer.zero_grad(set_to_none=True)
        output = objective(model, features, targets, target_mask)
        output.total.backward()
        return output.total

    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    final = objective(model, features, targets, target_mask)
    if not torch.isfinite(final.total) or float(final.total) > float(initial.total) + 1e-6:
        raise RuntimeError("v10 deterministic LBFGS did not reduce its objective")
    parameter = model.endpoint_utility.weight
    if not torch.isfinite(parameter).all():
        raise RuntimeError("v10 LBFGS produced a non-finite residual weight")
    state = optimizer.state.get(parameter, {})
    iteration_count = int(state.get("n_iter", 0))
    if iteration_count > LBFGS_MAX_ITER:
        raise RuntimeError("v10 LBFGS exceeded its frozen iteration limit")
    fit = {
        "objective": objective_name,
        "optimizer": "full_batch_LBFGS_strong_wolfe",
        "initialization": "exact_zero",
        "max_iter": LBFGS_MAX_ITER,
        "max_eval": LBFGS_MAX_EVAL,
        "iteration_count": iteration_count,
        "closure_count": closure_count,
        "trainable_parameter_count": model.n_trainable_parameters,
        "patient_count": int(features.values.shape[0]),
        "initial": {
            "total": float(initial.total.detach()),
            "exact_set_mass": float(initial.exact_set_mass.detach()),
            "l2_penalty": float(initial.l2_penalty.detach()),
        },
        "final": {
            "total": float(final.total.detach()),
            "exact_set_mass": float(final.exact_set_mass.detach()),
            "l2_penalty": float(final.l2_penalty.detach()),
        },
        "weight_l2": float(parameter.detach().norm()),
    }
    model.eval()
    model.requires_grad_(False)
    return model, fit


def _loeo_winners(
    model: PositiveSetEndpointResidual,
    contrasts: EarlyPreEndpointContrasts,
    event_patient_index: torch.Tensor,
    patient_index: int,
    state: PositiveSetEndpointFeatureState,
    anchor_index: int,
) -> tuple[tuple[int, ...], int]:
    """Reaggregate one held patient after removing each event in turn."""

    common_event = contrasts.node_mask.all(dim=1)
    event_indices = torch.nonzero(
        (event_patient_index == int(patient_index)) & common_event,
        as_tuple=False,
    ).flatten()
    winners: list[int] = []
    undefined = 0
    if event_indices.numel() < 2:
        return (), int(event_indices.numel())
    for removed in event_indices.tolist():
        kept = event_indices[event_indices != removed]
        if kept.numel() < 1:
            undefined += 1
            continue
        selected = _subset_contrasts(contrasts, kept)
        local_patient_index = torch.zeros(
            kept.numel(), dtype=torch.long, device=kept.device
        )
        transformed = transform_positive_set_endpoint_features(
            selected,
            local_patient_index,
            1,
            state,
        )
        with torch.no_grad():
            utility = model.score_nodes(transformed.values)[0]
        ranked = _local_utility_winner(
            utility,
            anchor_index,
            transformed.node_mask[0],
        )
        if ranked is None:
            undefined += 1
        else:
            winners.append(int(ranked[0]))
    return tuple(winners), undefined


def _apply_fold_residual(
    model: PositiveSetEndpointResidual,
    features: PositiveSetEndpointFeatureBatch,
    anchor_scores: torch.Tensor,
    global_patient_indices: Sequence[int],
    contrasts: EarlyPreEndpointContrasts,
    event_patient_index: torch.Tensor,
    feature_state: PositiveSetEndpointFeatureState,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, object]]:
    """Apply the frozen margin/event/LOEO/gap policy without held labels."""

    held_count = len(global_patient_indices)
    if tuple(anchor_scores.shape) != (held_count, N_STANDARD_CHANNELS):
        raise ValueError("held anchor scores must have shape [P,19]")
    candidate = anchor_scores.clone()
    utility = torch.zeros_like(anchor_scores)
    anchor_index = torch.full((held_count,), -1, dtype=torch.long)
    proposed_index = torch.full((held_count,), -1, dtype=torch.long)
    margin = torch.full((held_count,), float("-inf"))
    gap = torch.full((held_count,), float("inf"))
    pair_events = torch.zeros(held_count, dtype=torch.long)
    available_events = torch.tensor(
        [
            int(
                contrasts.node_mask[event_patient_index == int(patient)]
                .any(dim=1)
                .sum()
            )
            for patient in global_patient_indices
        ],
        dtype=torch.long,
    )
    loeo_defined = torch.zeros(held_count, dtype=torch.long)
    loeo_undefined = torch.zeros(held_count, dtype=torch.long)
    loeo_stable = torch.zeros(held_count, dtype=torch.bool)
    candidate_available = torch.zeros(held_count, dtype=torch.bool)
    anchor_unique = torch.zeros(held_count, dtype=torch.bool)
    margin_pass = torch.zeros(held_count, dtype=torch.bool)
    event_count_pass = torch.zeros(held_count, dtype=torch.bool)
    gap_pass = torch.zeros(held_count, dtype=torch.bool)
    eligible = torch.zeros(held_count, dtype=torch.bool)
    applied = torch.zeros(held_count, dtype=torch.bool)

    with torch.no_grad():
        utility.copy_(model.score_nodes(features.values).cpu())
    for local, patient in enumerate(global_patient_indices):
        a, unique = _unique_anchor_index(anchor_scores[local])
        anchor_index[local] = a
        anchor_unique[local] = unique
        ranked = _local_utility_winner(
            utility[local], a, features.node_mask[local]
        )
        if ranked is None or not unique:
            continue
        q, local_margin, _ = ranked
        proposed_index[local] = q
        margin[local] = local_margin
        candidate_available[local] = True
        pair_events[local] = min(
            int(features.event_counts[local, a]),
            int(features.event_counts[local, q]),
        )
        margin_pass[local] = local_margin >= FLIP_LOGIT_MARGIN
        event_count_pass[local] = bool(
            available_events[local] >= MIN_PAIR_EVENT_COUNT
            and pair_events[local] >= MIN_PAIR_EVENT_COUNT
        )
        local_gap = _anchor_gap_z(anchor_scores[local], a, q)
        gap[local] = local_gap
        gap_pass[local] = q == a or local_gap <= ANCHOR_GAP_Z_MAX
        winners, undefined = _loeo_winners(
            model,
            contrasts,
            event_patient_index,
            int(patient),
            feature_state,
            a,
        )
        loeo_defined[local] = len(winners)
        loeo_undefined[local] = undefined
        # Avoid a vacuous pass: a candidate supported by >=2 paired events must
        # yield at least two checkable leave-one-event-out rankings.
        loeo_stable[local] = bool(
            undefined == 0
            and len(winners) == int(available_events[local])
            and len(winners) >= 2
            and all(value == q for value in winners)
        )
        eligible[local] = bool(
            margin_pass[local]
            and event_count_pass[local]
            and loeo_stable[local]
            and gap_pass[local]
        )
        if bool(eligible[local]) and q != a:
            candidate[local, a], candidate[local, q] = (
                anchor_scores[local, q],
                anchor_scores[local, a],
            )
            applied[local] = True

    if not torch.isfinite(candidate).all():
        raise RuntimeError("selective residual produced a non-finite score")
    diagnostics = {
        "patient_count": held_count,
        "unique_anchor_count": int(anchor_unique.sum()),
        "candidate_available_count": int(candidate_available.sum()),
        "margin_pass_count": int(margin_pass.sum()),
        "event_count_pass_count": int(event_count_pass.sum()),
        "loeo_stable_count": int(loeo_stable.sum()),
        "anchor_gap_pass_count": int(gap_pass.sum()),
        "eligible_count": int(eligible.sum()),
        "applied_swap_count": int(applied.sum()),
        "accepted_anchor_noop_count": int((eligible & ~applied).sum()),
        "residual_abstain_count": int((~eligible).sum()),
        "loeo_defined_total": int(loeo_defined.sum()),
        "loeo_undefined_total": int(loeo_undefined.sum()),
        "pair_event_count_min_when_available": (
            int(pair_events[candidate_available].min())
            if bool(candidate_available.any())
            else None
        ),
    }
    tensors = {
        "utility": utility.contiguous(),
        "anchor_index": anchor_index,
        "proposed_index": proposed_index,
        "candidate_margin": margin,
        "anchor_gap_z": gap,
        "pair_event_count": pair_events,
        "available_event_count": available_events,
        "loeo_defined_count": loeo_defined,
        "loeo_undefined_count": loeo_undefined,
        "loeo_stable": loeo_stable,
        "candidate_available": candidate_available,
        "anchor_unique": anchor_unique,
        "margin_pass": margin_pass,
        "event_count_pass": event_count_pass,
        "anchor_gap_pass": gap_pass,
        "eligible": eligible,
        "flip_applied": applied,
        "residual_abstain": ~eligible,
        "node_mask": features.node_mask.cpu().contiguous(),
        "event_counts": features.event_counts.cpu().contiguous(),
    }
    return candidate.contiguous(), tensors, diagnostics


def _flip_outcomes(
    anchor_index: torch.Tensor,
    proposed_index: torch.Tensor,
    applied: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, int]:
    rescue = harm = neutral = unavailable = 0
    for patient in torch.nonzero(applied, as_tuple=False).flatten().tolist():
        a = int(anchor_index[patient])
        q = int(proposed_index[patient])
        if a < 0 or q < 0 or not bool(target_mask[patient, a] and target_mask[patient, q]):
            unavailable += 1
            continue
        before = bool(targets[patient, a] == 1)
        after = bool(targets[patient, q] == 1)
        if after and not before:
            rescue += 1
        elif before and not after:
            harm += 1
        else:
            neutral += 1
    return {
        "applied": int(applied.sum()),
        "strict_rescue": rescue,
        "exact_to_nonexact_harm": harm,
        "neutral": neutral,
        "target_unavailable": unavailable,
        "net_exact": rescue - harm,
    }


def _transition_summary(
    candidate: torch.Tensor,
    anchor: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, object]:
    result = _transition_diagnostic(candidate, anchor, targets, target_mask)
    transitions = result["transitions"]
    neighbour_to_exact = transitions.get("tcp_neighbour_only->exact", 0) + transitions.get(
        "official_non_tcp_neighbour_only->exact", 0
    )
    exact_to_neighbour = transitions.get("exact->tcp_neighbour_only", 0) + transitions.get(
        "exact->official_non_tcp_neighbour_only", 0
    )
    result.update(
        {
            "all_neighbour_to_exact_rescue_count": neighbour_to_exact,
            "exact_to_any_neighbour_loss_count": exact_to_neighbour,
            "far_to_exact_rescue_count": transitions.get("far->exact", 0),
            "exact_to_far_loss_count": transitions.get("exact->far", 0),
        }
    )
    return result


def _top1_indices(
    scores: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    rows = scores.masked_fill(~target_mask, -torch.inf)
    return rows.argmax(dim=1)


def _weight_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator) <= 1e-12:
        return 1.0 if torch.allclose(left, right) else 0.0
    return float(torch.dot(left.flatten(), right.flatten()) / denominator)


def _fold_coefficient_stability(weights: Sequence[torch.Tensor]) -> dict[str, object]:
    matrix = torch.stack(tuple(value.float().flatten() for value in weights))
    norm = matrix.norm(dim=1, keepdim=True)
    normalized = matrix / norm.clamp_min(1e-12)
    cosine = normalized @ normalized.transpose(0, 1)
    off_diagonal = cosine[~torch.eye(len(weights), dtype=torch.bool)]
    sign = torch.sign(matrix)
    nonzero = sign != 0
    unanimous = ((sign == sign[:1]) | ~nonzero).all(dim=0) & nonzero.any(dim=0)
    return {
        "pairwise_cosine_mean": float(off_diagonal.mean()),
        "pairwise_cosine_min": float(off_diagonal.min()),
        "unanimous_nonzero_sign_fraction": float(unanimous.float().mean()),
        "fold_weight_l2": norm.squeeze(1).tolist(),
        "fold_h8_weight_l2": matrix[:, :8].norm(dim=1).tolist(),
        "fold_v6_weight_l2": matrix[:, 8:].norm(dim=1).tolist(),
        "semantics": "descriptive fold stability; not feature causality",
    }


def _expected_hits(metrics: Mapping[str, object], field: str) -> float:
    top1 = metrics["top1"]
    return float(top1[field]) * int(top1["n_samples"])


def _gate_decision(
    *,
    candidate_metrics: Mapping[str, object],
    anchor_metrics: Mapping[str, object],
    bootstrap: Mapping[str, object],
    transitions: Mapping[str, object],
    flip_outcomes: Mapping[str, int],
    fold_rescue_count: int,
    candidate_oof: torch.Tensor,
    sensitivity_rows: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    patients = int(candidate_oof.shape[0])
    strict_gain = _expected_hits(candidate_metrics, "strict_accuracy") - _expected_hits(
        anchor_metrics, "strict_accuracy"
    )
    checks = {
        "strict_net_gain_at_least_4_of_65": strict_gain + EXPECTED_HIT_ATOL >= 4.0,
        "applied_exact_to_nonexact_harm_is_zero": int(
            flip_outcomes["exact_to_nonexact_harm"]
        )
        == 0,
        "bootstrap_strict_delta_ci95_lower_strictly_above_zero": float(
            bootstrap["strict_top1"]["ci95"][0]
        )
        > 0.0,
        "macro_ap_nonlower_than_anchor": float(
            candidate_metrics["ranking"]["macro_average_precision"]
        )
        + NONINFERIORITY_ATOL
        >= float(anchor_metrics["ranking"]["macro_average_precision"]),
        "mrr_nonlower_than_anchor": float(
            candidate_metrics["ranking"]["mean_reciprocal_rank"]
        )
        + NONINFERIORITY_ATOL
        >= float(anchor_metrics["ranking"]["mean_reciprocal_rank"]),
        "relaxed_top1_nonlower_than_anchor": float(
            candidate_metrics["top1"]["relaxed_accuracy"]
        )
        + NONINFERIORITY_ATOL
        >= float(anchor_metrics["top1"]["relaxed_accuracy"]),
        "far_error_count_nonincreasing": int(transitions["candidate_far_count"])
        <= int(transitions["anchor_far_count"]),
        "strict_rescue_in_at_least_2_outer_folds": int(fold_rescue_count) >= 2,
        "complete_finite_65_patient_oof_full_denominator": patients
        == EXPECTED_PATIENT_COUNT
        and tuple(candidate_oof.shape) == (EXPECTED_PATIENT_COUNT, N_STANDARD_CHANNELS)
        and bool(torch.isfinite(candidate_oof).all()),
    }
    sensitivity_minimum_strict_gain = math.ceil(0.5 * strict_gain)
    for name in SENSITIVITY_KINDS:
        row = sensitivity_rows[name]
        checks[f"{name}_retains_at_least_half_main_strict_gain"] = (
            float(row["strict_hit_delta"]) + EXPECTED_HIT_ATOL
            >= sensitivity_minimum_strict_gain
        )
        checks[f"{name}_exact_harm_zero"] = int(
            row["flip_outcomes"]["exact_to_nonexact_harm"]
        ) == 0
    passed = all(checks.values())
    return {
        "checks": checks,
        "pass": passed,
        "status": (
            "go_mechanism_support_source_train_only"
            if passed
            else "no_go_keep_temporal_mil_exact"
        ),
        "strict_hit_delta": strict_gain,
        "sensitivity_minimum_strict_hit_delta": sensitivity_minimum_strict_gain,
        "sensitivity_retention_rule": "ceil(0.5 * main_strict_hit_delta)",
        "interpretation": (
            "exploratory repeatedly-developed source-train OOF mechanism gate; "
            "not confirmatory external validation"
        ),
    }


def _build_contrasts(full, prefix_tokens: torch.Tensor) -> EarlyPreEndpointContrasts:
    evidence = full.evidence
    # AQ is allowed only to remove support.  It never enters H/V utility.
    phase_mask = evidence.phase_mask & ~evidence.event_abstain.unsqueeze(1)
    gate = build_shared_early_i_gate(
        evidence.ictal,
        evidence.ictal_mask,
        phase_mask,
    )
    return build_early_pre_endpoint_contrasts(
        prefix_tokens,
        evidence.evolution,
        evidence.evolution_mask,
        phase_mask,
        gate,
    )


def _validate_fixed_target_mask(target_mask: torch.Tensor) -> None:
    expected = FIXED_EVALUABLE_CHANNEL_MASK.to(target_mask.device).view(1, -1)
    if not torch.equal(target_mask, expected.expand_as(target_mask)):
        raise ValueError("v10 requires the frozen PZ-only-unavailable target carrier")


def _run_oof(full, patient_folds, contrasts, anchor):
    patient_count = len(full.patient_ids)
    predictions = {
        name: torch.full_like(anchor, torch.nan) for name in OBJECTIVES
    }
    diagnostics: dict[str, dict[str, torch.Tensor]] = {
        name: {} for name in OBJECTIVES
    }
    fold_rows: list[dict[str, object]] = []
    fold_weights: dict[str, list[torch.Tensor]] = {
        name: [] for name in OBJECTIVES
    }
    tensor_keys = (
        "utility",
        "anchor_index",
        "proposed_index",
        "candidate_margin",
        "anchor_gap_z",
        "pair_event_count",
        "available_event_count",
        "loeo_defined_count",
        "loeo_undefined_count",
        "loeo_stable",
        "candidate_available",
        "anchor_unique",
        "margin_pass",
        "event_count_pass",
        "anchor_gap_pass",
        "eligible",
        "flip_applied",
        "residual_abstain",
        "node_mask",
        "event_counts",
    )
    for name in OBJECTIVES:
        for key in tensor_keys:
            if key in {"utility"}:
                value = torch.zeros(patient_count, N_STANDARD_CHANNELS)
            elif key in {"node_mask"}:
                value = torch.zeros(patient_count, N_STANDARD_CHANNELS, dtype=torch.bool)
            elif key in {"event_counts"}:
                value = torch.zeros(patient_count, N_STANDARD_CHANNELS, dtype=torch.long)
            elif key in {"anchor_index", "proposed_index"}:
                value = torch.full((patient_count,), -1, dtype=torch.long)
            elif key in {
                "pair_event_count",
                "available_event_count",
                "loeo_defined_count",
                "loeo_undefined_count",
            }:
                value = torch.zeros(patient_count, dtype=torch.long)
            elif key in {"candidate_margin"}:
                value = torch.full((patient_count,), float("-inf"))
            elif key in {"anchor_gap_z"}:
                value = torch.full((patient_count,), float("inf"))
            else:
                value = torch.zeros(patient_count, dtype=torch.bool)
            diagnostics[name][key] = value

    for fold in OUTER_FOLDS:
        train_indices = _patient_indices_for_fold(patient_folds, fold, held=False)
        held_indices = _patient_indices_for_fold(patient_folds, fold, held=True)
        features, feature_state = fit_fold_positive_set_endpoint_features(
            contrasts,
            full.event_patient_index,
            patient_count,
            train_indices,
        )
        train_eligible = _feature_eligible_patients(features, train_indices)
        if not train_eligible:
            raise RuntimeError("outer fold has no feature-eligible training patient")
        train_features = _subset_feature_batch(features, train_eligible)
        train_targets = _subset_rows(full.targets, train_eligible)
        train_target_mask = _subset_rows(full.target_mask, train_eligible)
        held_features = _subset_feature_batch(features, held_indices)
        held_anchor = _subset_rows(anchor, held_indices)
        objective_rows: dict[str, object] = {}

        for objective_name in OBJECTIVES:
            model, fit = _fit_residual(
                train_features,
                train_targets,
                train_target_mask,
                objective_name=objective_name,
            )
            candidate, local_tensors, policy = _apply_fold_residual(
                model,
                held_features,
                held_anchor,
                held_indices,
                contrasts,
                full.event_patient_index,
                feature_state,
            )
            predictions[objective_name][list(held_indices)] = candidate
            for key, value in local_tensors.items():
                diagnostics[objective_name][key][list(held_indices)] = value
            objective_rows[objective_name] = {
                "fit": fit,
                "policy": policy,
            }
            fold_weights[objective_name].append(
                model.endpoint_utility.weight.detach().cpu().flatten().contiguous()
            )

        fold_rows.append(
            {
                "outer_fold": fold,
                "train_patient_count": len(train_indices),
                "feature_eligible_train_patient_count": len(train_eligible),
                "feature_ineligible_train_patient_count": len(train_indices)
                - len(train_eligible),
                "held_patient_count": len(held_indices),
                "train_patient_roster_sha256": _scope_sha256(
                    tuple(full.patient_ids[index] for index in train_indices)
                ),
                "feature_eligible_train_roster_sha256": _scope_sha256(
                    tuple(full.patient_ids[index] for index in train_eligible)
                ),
                "feature_state": {
                    "h_pca_components": int(feature_state.h_components.shape[0]),
                    "feature_dimension": POSITIVE_SET_ENDPOINT_FEATURE_DIM,
                    "feature_scale_min": float(feature_state.feature_scale.min()),
                    "feature_scale_max": float(feature_state.feature_scale.max()),
                },
                "objectives": objective_rows,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "outer_predictions_complete_labels_unread",
                    "fold": fold,
                    "main_applied": objective_rows["main"]["policy"][
                        "applied_swap_count"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if any(not torch.isfinite(value).all() for value in predictions.values()):
        raise RuntimeError("v10 OOF left a patient prediction unfilled")

    # All three complete OOF score carriers now exist.  Held labels are first
    # consulted below, after prediction generation can no longer affect any
    # fold-local PCA, scaler, residual weight, or selective threshold.
    for fold, fold_row in zip(OUTER_FOLDS, fold_rows):
        held_indices = _patient_indices_for_fold(patient_folds, fold, held=True)
        held_targets = _subset_rows(full.targets, held_indices)
        held_target_mask = _subset_rows(full.target_mask, held_indices)
        held_anchor = _subset_rows(anchor, held_indices)
        anchor_fold_metrics = _metrics(
            held_anchor, held_targets, held_target_mask
        )
        for objective_name in OBJECTIVES:
            candidate = _subset_rows(predictions[objective_name], held_indices)
            metrics = _metrics(candidate, held_targets, held_target_mask)
            local_anchor_index = _subset_rows(
                diagnostics[objective_name]["anchor_index"], held_indices
            )
            local_proposed_index = _subset_rows(
                diagnostics[objective_name]["proposed_index"], held_indices
            )
            local_applied = _subset_rows(
                diagnostics[objective_name]["flip_applied"], held_indices
            )
            fold_row["objectives"][objective_name].update(
                {
                    "metrics": metrics,
                    "strict_hit_delta": _expected_hits(metrics, "strict_accuracy")
                    - _expected_hits(anchor_fold_metrics, "strict_accuracy"),
                    "flip_outcomes": _flip_outcomes(
                        local_anchor_index,
                        local_proposed_index,
                        local_applied,
                        held_targets,
                        held_target_mask,
                    ),
                }
            )

    anchor_metrics = _metrics(anchor, full.targets, full.target_mask)
    anchor_strict = _expected_hits(anchor_metrics, "strict_accuracy")
    anchor_relaxed = _expected_hits(anchor_metrics, "relaxed_accuracy")
    if abs(anchor_strict - EXPECTED_ANCHOR_STRICT_HITS) > EXPECTED_HIT_ATOL or abs(
        anchor_relaxed - EXPECTED_ANCHOR_RELAXED_HITS
    ) > EXPECTED_HIT_ATOL:
        raise RuntimeError("frozen temporal_mil_exact anchor counts changed")

    objective_results: dict[str, dict[str, object]] = {}
    for objective_name in OBJECTIVES:
        candidate = predictions[objective_name]
        metrics = _metrics(candidate, full.targets, full.target_mask)
        flip_outcomes = _flip_outcomes(
            diagnostics[objective_name]["anchor_index"],
            diagnostics[objective_name]["proposed_index"],
            diagnostics[objective_name]["flip_applied"],
            full.targets,
            full.target_mask,
        )
        objective_results[objective_name] = {
            "metrics": metrics,
            "strict_hit_delta": _expected_hits(metrics, "strict_accuracy")
            - anchor_strict,
            "flip_outcomes": flip_outcomes,
            "paired_patient_bootstrap": _paired_patient_bootstrap(
                candidate,
                anchor,
                full.targets,
                full.target_mask,
            ),
            "top1_transition_diagnostic": _transition_summary(
                candidate,
                anchor,
                full.targets,
                full.target_mask,
            ),
            "residual_policy": {
                "applied_swap_count": int(
                    diagnostics[objective_name]["flip_applied"].sum()
                ),
                "eligible_count": int(diagnostics[objective_name]["eligible"].sum()),
                "residual_abstain_count": int(
                    diagnostics[objective_name]["residual_abstain"].sum()
                ),
                "full_denominator_patient_count": patient_count,
            },
            "fold_coefficient_stability": _fold_coefficient_stability(
                fold_weights[objective_name]
            ),
        }

    main_top1 = _top1_indices(predictions["main"], full.target_mask)
    sensitivity_rows: dict[str, dict[str, object]] = {}
    for name in SENSITIVITY_KINDS:
        top1 = _top1_indices(predictions[name], full.target_mask)
        main_flips = diagnostics["main"]["flip_applied"]
        sensitivity_flips = diagnostics[name]["flip_applied"]
        cosines = [
            _weight_cosine(left, right)
            for left, right in zip(fold_weights["main"], fold_weights[name])
        ]
        sensitivity_rows[name] = {
            **objective_results[name],
            "top1_agreement_with_main": float((top1 == main_top1).float().mean()),
            "top1_disagreement_patient_ids": [
                full.patient_ids[index]
                for index in torch.nonzero(top1 != main_top1, as_tuple=False)
                .flatten()
                .tolist()
            ],
            "applied_flip_patient_ids": [
                full.patient_ids[index]
                for index in torch.nonzero(sensitivity_flips, as_tuple=False)
                .flatten()
                .tolist()
            ],
            "flip_set_symmetric_difference_from_main": [
                full.patient_ids[index]
                for index in torch.nonzero(
                    main_flips ^ sensitivity_flips, as_tuple=False
                )
                .flatten()
                .tolist()
            ],
            "fold_weight_cosine_to_main": cosines,
            "fold_weight_cosine_mean": sum(cosines) / len(cosines),
            "fold_weight_cosine_min": min(cosines),
        }

    main_result = objective_results["main"]
    fold_rescue_count = sum(
        int(row["objectives"]["main"]["flip_outcomes"]["strict_rescue"] > 0)
        for row in fold_rows
    )
    decision = _gate_decision(
        candidate_metrics=main_result["metrics"],
        anchor_metrics=anchor_metrics,
        bootstrap=main_result["paired_patient_bootstrap"],
        transitions=main_result["top1_transition_diagnostic"],
        flip_outcomes=main_result["flip_outcomes"],
        fold_rescue_count=fold_rescue_count,
        candidate_oof=predictions["main"],
        sensitivity_rows=sensitivity_rows,
    )
    result = {
        "screen_kind": (
            "single_frozen_source_train_patient_oof_mechanism_recovery;"
            "same_65_patients_previously_used_for_development"
        ),
        "anchor": {
            "name": "temporal_mil_exact",
            "metrics": anchor_metrics,
        },
        "main": main_result,
        "sensitivity": sensitivity_rows,
        "outer_folds": fold_rows,
        "strict_rescue_fold_count": fold_rescue_count,
        "frozen_go_no_go_gate": decision,
    }

    tensors: dict[str, torch.Tensor] = {
        CANDIDATE_NAME: predictions["main"].cpu().contiguous(),
        "temporal_mil_exact_anchor": anchor.cpu().contiguous(),
        "targets": full.targets.cpu().contiguous(),
        "target_mask": full.target_mask.cpu().contiguous(),
        "patient_folds": torch.tensor(patient_folds, dtype=torch.long),
    }
    for objective_name in OBJECTIVES:
        prediction_name = (
            CANDIDATE_NAME
            if objective_name == "main"
            else f"{CANDIDATE_NAME}__{objective_name}"
        )
        tensors[prediction_name] = predictions[objective_name].cpu().contiguous()
        for key, value in diagnostics[objective_name].items():
            tensors[f"{objective_name}__{key}"] = value.cpu().contiguous()
        tensors[f"{objective_name}__fold_weights"] = torch.stack(
            fold_weights[objective_name]
        ).contiguous()

    final_features, final_feature_state = fit_fold_positive_set_endpoint_features(
        contrasts,
        full.event_patient_index,
        patient_count,
        tuple(range(patient_count)),
    )
    final_eligible = _feature_eligible_patients(
        final_features, tuple(range(patient_count))
    )
    if not final_eligible:
        raise RuntimeError("full source-train has no feature-eligible patient")
    final_train_features = _subset_feature_batch(final_features, final_eligible)
    final_targets = _subset_rows(full.targets, final_eligible)
    final_target_mask = _subset_rows(full.target_mask, final_eligible)
    final_state = {
        "h_center": final_feature_state.h_center.cpu().contiguous(),
        "h_components": final_feature_state.h_components.cpu().contiguous(),
        "feature_mean": final_feature_state.feature_mean.cpu().contiguous(),
        "feature_scale": final_feature_state.feature_scale.cpu().contiguous(),
        "feature_eligible_patient_index": torch.tensor(
            final_eligible, dtype=torch.long
        ),
    }
    final_fit: dict[str, object] = {
        "feature_eligible_patient_count": len(final_eligible),
        "feature_ineligible_patient_count": patient_count - len(final_eligible),
        "feature_eligible_patient_roster_sha256": _scope_sha256(
            tuple(full.patient_ids[index] for index in final_eligible)
        ),
        "objectives": {},
    }
    for objective_name in OBJECTIVES:
        model, fit = _fit_residual(
            final_train_features,
            final_targets,
            final_target_mask,
            objective_name=objective_name,
        )
        final_state[f"{objective_name}__endpoint_utility.weight"] = (
            model.endpoint_utility.weight.detach().cpu().contiguous()
        )
        final_fit["objectives"][objective_name] = fit
    result["final_full_source_train_fit"] = final_fit
    return result, tensors, final_state


def _graph_payload() -> dict[str, object]:
    edges = tuple(
        (CHANNEL_INDEX[left], CHANNEL_INDEX[right]) for left, right in TCP_20_EDGES
    )
    return {
        "edge_count": len(edges),
        "electrode_edges": [list(edge) for edge in TCP_20_EDGES],
        "candidate_generation": "anchor_plus_direct_TCP20_physical_neighbours_only",
        "deepsoz_official_one_hop_used_for_candidate_generation": False,
        "semantics": "local selective safety graph; not propagation or anatomy truth",
    }


def _shared_i_gate_concentration_diagnostic(
    gate: SharedEarlyIGate,
) -> dict[str, object]:
    """Describe the target-free gate without treating it as an I ablation."""

    alpha = gate.weights[gate.event_valid][:, list(EARLY_TILE_INDICES)].float()
    if alpha.ndim != 2 or alpha.shape[0] < 1 or alpha.shape[1] != 3:
        raise ValueError("v10 requires at least one valid three-tile I gate")
    if not torch.allclose(
        alpha.sum(dim=1),
        torch.ones(alpha.shape[0], dtype=alpha.dtype, device=alpha.device),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("valid early I gates must sum to one")
    entropy = -(alpha * alpha.clamp_min(torch.finfo(alpha.dtype).tiny).log()).sum(
        dim=1
    )
    normalized_entropy = entropy / math.log(3.0)
    maximum_weight = alpha.max(dim=1).values
    effective_tile_number = entropy.exp()
    l1_from_uniform = (alpha - (1.0 / 3.0)).abs().sum(dim=1)

    def summary(value: torch.Tensor) -> dict[str, float]:
        return {
            "mean": float(value.mean()),
            "population_sd": float(value.std(unbiased=False)),
            "min": float(value.min()),
            "max": float(value.max()),
        }

    return {
        "valid_event_count": int(alpha.shape[0]),
        "early_tile_indices": list(EARLY_TILE_INDICES),
        "normalized_entropy": summary(normalized_entropy),
        "maximum_weight": summary(maximum_weight),
        "effective_tile_number": summary(effective_tile_number),
        "l1_distance_from_uniform": summary(l1_from_uniform),
        "enters_go_no_go_gate": False,
        "supports_standalone_i_effect_claim": False,
        "interpretation": (
            "target-free concentration diagnostic only; without a frozen uniform-gate "
            "counterfactual, performance belongs to the composite H+V+global-I residual"
        ),
    }


def _execute_authorized_oof_and_publish(
    *,
    output: Path,
    preflight: Mapping[str, object],
    full,
    patient_folds: Sequence[int],
    contrasts: EarlyPreEndpointContrasts,
    anchor: torch.Tensor,
) -> int:
    """Run and atomically publish after the caller holds the ledger lease."""

    result, tensors, final_state = _run_oof(
        full,
        patient_folds,
        contrasts,
        anchor,
    )
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required") from exc
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        prediction_path = temporary / "oof_predictions.safetensors"
        checkpoint_path = temporary / "final_checkpoint.safetensors"
        save_file(tensors, str(prediction_path))
        save_file(final_state, str(checkpoint_path))
        manifest = {
            **preflight,
            "status": "completed_exploratory_source_train_patient_oof",
            "patient_ids": list(full.patient_ids),
            "patient_folds": list(patient_folds),
            "result": result,
            "files": {
                "oof_predictions.safetensors": {
                    "sha256": _file_sha256(prediction_path),
                    "size_bytes": prediction_path.stat().st_size,
                },
                "final_checkpoint.safetensors": {
                    "sha256": _file_sha256(checkpoint_path),
                    "size_bytes": checkpoint_path.stat().st_size,
                    "state_sha256": _tensor_state_sha256(final_state),
                },
            },
            "scientific_boundary": {
                "foundation_replaced": False,
                "foundation_trainable_parameter_count": 0,
                "labram_feature": "frozen_block9_prefix_H_only",
                "ictal_role": "global_shared_temporal_gate_only",
                "ictal_semantics": (
                    "retrospective_scalp_visible_ictal_involvement_not_SOZ_or_onset_channel"
                ),
                "evolution_semantics": (
                    "signal_observable_descriptors_not_propagation_or_origin_truth"
                ),
                "deepsoz_zero_semantics": (
                    "benchmark_complement_not_clinically_verified_non_SOZ"
                ),
                "candidate_graph_semantics": (
                    "direct_TCP20_local_safety_prior_not_propagation"
                ),
                "source_dev_used": False,
                "source_eval_used": False,
                "private_used": False,
                "task_conditioning": (
                    "confirmed_seizure_TUSZ_global_t0_t1_not_end_to_end_detection"
                ),
                "reference_robustness_assessed": False,
                "preprocessing_view": "single_causal_C_CAR19",
                "same_source_train_patients_reused_after_prior_development": True,
                "formal_promotion": False,
            },
        }
        raw = _canonical_bytes(manifest)
        (temporary / "manifest.json").write_bytes(raw)
        os.rename(temporary, output)
        _fsync_directory(output.parent)
        published = True
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "output_directory": str(output),
                    "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                    "main_metrics": result["main"]["metrics"],
                    "main_transitions": result["main"][
                        "top1_transition_diagnostic"
                    ],
                    "decision": result["frozen_go_no_go_gate"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = _absolute_no_symlink(args.output_directory, field="v10 output directory")
    default_output = _absolute_no_symlink(DEFAULT_OUTPUT, field="default v10 output")
    if output != default_output:
        raise ValueError(
            "v10 is authorized only for its unique frozen default output directory"
        )
    _configure_deterministic_runtime()
    access_audit = _load_access_audit()
    cache, full, patient_folds, event_ids, lineage = _load_inputs(
        prefix_cache_path=DEFAULT_PREFIX_CACHE,
        expected_prefix_manifest_sha256=DEFAULT_PREFIX_CACHE_MANIFEST_SHA256,
        source_train_iv_path=DEFAULT_SOURCE_TRAIN_IV,
        expected_source_train_iv_manifest_sha256=(
            DEFAULT_SOURCE_TRAIN_IV_MANIFEST_SHA256
        ),
        target_scope_path=DEFAULT_TARGET_SCOPE,
        expected_target_receipt_sha256=(
            FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256
        ),
        require_full_scope=True,
    )
    _validate_fixed_target_mask(full.target_mask)
    if len(full.patient_ids) != EXPECTED_PATIENT_COUNT or (
        full.evidence.batch_size != EXPECTED_EVENT_COUNT
    ):
        raise ValueError("v10 requires the frozen 65-patient/582-event source-train")
    frozen_lineage = _validate_frozen_lineage_contract(
        full=full,
        patient_folds=patient_folds,
        event_ids=event_ids,
    )
    input_bindings = _frozen_input_bindings(
        lineage=lineage,
        frozen_lineage=frozen_lineage,
    )
    contrasts = _build_contrasts(full, cache.tokens)
    common_event_contract = contrasts.node_mask.all(dim=1) | ~contrasts.node_mask.any(
        dim=1
    )
    if not bool(common_event_contract.all()):
        raise ValueError("v10 forbids channel-specific event subsets")
    event_valid = contrasts.temporal_gate.event_valid
    patient_valid_event_count = torch.zeros(
        len(full.patient_ids), dtype=torch.long
    )
    patient_valid_event_count.index_add_(
        0,
        full.event_patient_index.cpu(),
        contrasts.node_mask.any(dim=1).cpu().long(),
    )
    patient_node_event_count = torch.zeros(
        len(full.patient_ids), N_STANDARD_CHANNELS, dtype=torch.long
    )
    patient_node_event_count.index_add_(
        0,
        full.event_patient_index.cpu(),
        contrasts.node_mask.cpu().long(),
    )
    feature_eligible = patient_valid_event_count >= 1
    feature_eligible_all_nodes = (
        patient_node_event_count[feature_eligible] > 0
    ).all()
    if not bool(feature_eligible_all_nodes):
        raise ValueError("feature-eligible patients must have a common all-19 carrier")
    i_gate_concentration = _shared_i_gate_concentration_diagnostic(
        contrasts.temporal_gate
    )
    preflight = {
        "status": "ready_single_frozen_source_train_patient_oof",
        "schema_version": SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "residual_schema_version": POSITIVE_SET_ENDPOINT_RESIDUAL_SCHEMA,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": _file_sha256(PROTOCOL_PATH),
        "patient_count": len(full.patient_ids),
        "event_count": full.evidence.batch_size,
        "fold_counts": {
            str(fold): sum(value == fold for value in patient_folds)
            for fold in OUTER_FOLDS
        },
        "event_order_sha256": _scope_sha256(event_ids),
        "carrier_preflight": {
            "shared_i_gate_shape": list(contrasts.temporal_gate.weights.shape),
            "shared_i_gate_has_spatial_axis": False,
            "shared_i_gate_concentration": i_gate_concentration,
            "h_contrast_shape": list(contrasts.h.shape),
            "v_contrast_shape": list(contrasts.v.shape),
            "valid_i_gate_event_count": int(event_valid.sum()),
            "invalid_i_gate_event_count": int((~event_valid).sum()),
            "valid_residual_event_count": int(contrasts.node_mask.all(dim=1).sum()),
            "common_all19_or_none_event_contract": True,
            "complete_tcp20_each_early_tile_required": True,
            "valid_event_node_count": int(contrasts.node_mask.sum()),
            "patient_with_at_least_1_valid_event_count": int(
                (patient_valid_event_count >= 1).sum()
            ),
            "patient_with_at_least_2_valid_events_count": int(
                (patient_valid_event_count >= 2).sum()
            ),
            "patient_valid_event_count": patient_valid_event_count.tolist(),
            "eligibility_used_target_values": False,
            "aq_value_used_as_spatial_feature": False,
            "aq_event_abstention_applied_as_support_removal": True,
        },
        "lineage": {
            **lineage,
            **frozen_lineage,
            "module_sha256": _file_sha256(MODULE_PATH),
            "runner_sha256": _file_sha256(RUNNER_PATH),
            "comparator": {
                "artifact_path": str(V7_COMPARATOR_PATH.relative_to(ROOT)),
                "manifest_sha256": V7_MANIFEST_SHA256,
                "prediction_sha256": V7_PREDICTION_SHA256,
            },
            "access_audit": access_audit,
        },
        "config": {
            "outer_folds": list(OUTER_FOLDS),
            "candidate_count": 1,
            "objective_names": list(OBJECTIVES),
            "h_pca_components": 8,
            "v_feature_dimension": 6,
            "node_feature_dimension": POSITIVE_SET_ENDPOINT_FEATURE_DIM,
            "trainable_parameter_count_per_fit": POSITIVE_SET_ENDPOINT_FEATURE_DIM,
            "bias": False,
            "l2_weight": POSITIVE_SET_ENDPOINT_L2_WEIGHT,
            "lbfgs_max_iter": LBFGS_MAX_ITER,
            "lbfgs_max_eval": LBFGS_MAX_EVAL,
            "lbfgs_tolerance_grad": 1e-7,
            "lbfgs_tolerance_change": 1e-9,
            "lbfgs_history_size": LBFGS_HISTORY_SIZE,
            "lbfgs_line_search": "strong_wolfe",
            "optimization_device": "cpu",
            "optimization_dtype": "float32",
            "lbfgs_initialization": "exact_zero",
            "flip_logit_margin": FLIP_LOGIT_MARGIN,
            "minimum_pair_event_count": MIN_PAIR_EVENT_COUNT,
            "anchor_gap_z_max": ANCHOR_GAP_Z_MAX,
            "loeo_stability_required": True,
            "maximum_swaps_per_patient": 1,
            "threshold_scan": False,
            "hyperparameter_scan": False,
            "channel_identity_feature": False,
            "foundation_trainable_parameter_count": 0,
            "paired_bootstrap_replicates": PAIRED_BOOTSTRAP_REPLICATES,
            "paired_bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
            "paired_bootstrap_interval": "two_sided_percentile_0.025_0.975",
            "paired_bootstrap_quantile_interpolation": "torch_default_linear",
            "noninferiority_tolerance": NONINFERIORITY_ATOL,
        },
        "candidate_graph": _graph_payload(),
        "foundation_backbone": "official_pretrained_LaBraM_Base_not_replaced_frozen",
        "source_dev_forward_count": 0,
        "source_eval_forward_count": 0,
        "private_forward_count": 0,
        "formal_promotion": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True), flush=True)
        return 0

    if output.name in {"", ".", ".."} or os.path.lexists(output):
        raise FileExistsError(f"output already exists or is invalid: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("output parent must be a regular directory")
    for source in (
        PROTOCOL_PATH,
        MODULE_PATH,
        K31_SCORE_MANIFEST,
        VAQ_MANIFEST,
        H_CROSSWALK_RECEIPT,
        DEFAULT_PREFIX_CACHE,
        DEFAULT_SOURCE_TRAIN_IV,
        DEFAULT_TARGET_SCOPE,
    ):
        resolved = source.resolve(strict=True)
        if output == resolved or output in resolved.parents or resolved in output.parents:
            raise ValueError("output path overlaps an immutable input")

    authorization_expected = _authorization_payload(
        input_bindings=input_bindings,
        patient_ids=full.patient_ids,
        patient_folds=patient_folds,
        event_patient_index=full.event_patient_index.cpu().tolist(),
    )
    authorization, authorization_sha256 = _load_v10_authorization(
        AUTHORIZATION_PATH,
        expected_payload=authorization_expected,
    )
    ledger_expected = _launch_ledger_payload(
        authorization_sha256=authorization_sha256,
        authorization_payload=authorization,
    )
    lease = _acquire_launch_ledger(
        LAUNCH_LEDGER_PATH,
        expected_payload=ledger_expected,
        output_directory=output,
    )
    try:
        # This loader verifies the target carrier and computes the first held-
        # label anchor metric.  It must therefore remain after ledger acquire.
        comparators, comparator_receipt = _load_fixed_comparators(
            full, patient_folds
        )
        expected_comparator = preflight["lineage"]["comparator"]
        if any(
            comparator_receipt.get(key) != expected_comparator[key]
            for key in ("artifact_path", "manifest_sha256", "prediction_sha256")
        ):
            raise ValueError("temporal anchor receipt changed after launch consumption")
        anchor = comparators[TEMPORAL_ANCHOR]
        authorized_preflight = dict(preflight)
        authorized_preflight["governance"] = {
            "authorization_scope": AUTHORIZED_SCOPE,
            "authorization_path": str(AUTHORIZATION_PATH.relative_to(ROOT)),
            "authorization_sha256": authorization_sha256,
            "launch_ledger_path": str(lease.path.relative_to(ROOT)),
            "launch_ledger_sha256": lease.sha256,
            "ledger_created_before_held_label_metrics": True,
            "deterministic_exact_replay_policy_enforced": True,
            "formal_reasoner_authorized": False,
            "formal_promotion_authorized": False,
        }
        print(
            json.dumps(
                {
                    "stage": "authorized_launch_consumed_before_held_label_metrics",
                    "launch_mode": lease.launch_mode,
                    "launch_ledger_sha256": lease.sha256,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return _execute_authorized_oof_and_publish(
            output=output,
            preflight=authorized_preflight,
            full=full,
            patient_folds=patient_folds,
            contrasts=contrasts,
            anchor=anchor,
        )
    finally:
        lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
