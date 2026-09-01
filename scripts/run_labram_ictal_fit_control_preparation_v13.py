#!/usr/bin/env python3
"""Run the six authorized v13 fit-only capacity controls in frozen order.

This orchestration has no gate input, target-join input, native-evaluation
input, DeepSOZ input, or private input.  It first verifies a canonical
authorization and all pinned file hashes.  Unless ``--preflight-only`` is
used, it exclusively creates the authorized output root, then materializes
physical fit targets, physical fit tokens, the fixed capacity-matched primary
control, and the naked secondary control for ``fold0..fold4,final``.  Existing
outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
_SELECTION_ORDER = ("fold0", "fold1", "fold2", "fold3", "fold4", "final")
_CONTROL_ORDER = ("capacity", "naked")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_AUTHORIZATION_BYTES = 4 * 1024 * 1024
_BWRAP_EXECUTABLE = Path("/usr/bin/bwrap")
_STRACE_EXECUTABLE = Path("/usr/bin/strace")
_SANDBOX_PROJECT_ROOT = Path("/opt/soz-v13-project")
_SANDBOX_PYTHON_ROOT = Path("/opt/soz-v13-python")
_SANDBOX_FIT_TARGET = Path("/inputs/fit-target")
_SANDBOX_FIT_TOKEN = Path("/inputs/fit-token")
_SANDBOX_OUTPUT_PARENT = Path("/output")
_SANDBOX_OUTPUT_BUNDLE = _SANDBOX_OUTPUT_PARENT / "bundle"
_MOUNT_MANIFEST_SCHEMA = "soz_ictal_fit_control_trainer_mount_manifest_v13_1_v3"
_PREFLIGHT_SCHEMA = "soz_ictal_fit_control_preflight_v13_1_v3"
_EXECUTION_SCHEMA = "soz_ictal_fit_control_execution_receipt_v13_1_v3"
_AUTHORIZATION_SCHEMA = "soz_ictal_fit_control_authorization_v13_1_v3"
_BROKER_AUDIT_SCHEMA = "soz_ictal_fit_target_broker_trace_audit_v13_1_v3"
_STRACE_STRING_LIMIT_BYTES = 4096
_V2_AUTHORIZATION_PATH = (
    "outputs/labram_k31_v13_1_fit_control_authorization_20260811_v2.json"
)
_V2_AUTHORIZATION_SHA256 = (
    "0bdbea8fb8990c35a9714b03a3d27c18a20906c087ede72f90bc670b0deb0997"
)
_V2_PARTIAL_OUTPUT_ROOT = (
    "outputs/labram_k31_v13_1_fit_controls_authorized_20260811_v2"
)
_V2_SUPERSESSION_SIDECAR_PATH = (
    "outputs/labram_k31_v13_1_fit_control_authorization_20260811_v2."
    "superseded_by_v3.json"
)
_V2_PARTIAL_FILE_SHA256 = {
    "broker_logs/fold0.log": (
        "8a30c6377a9b7d7720f8632709a9afa430a32052d761342859570e95f6cae154"
    ),
    "broker_traces/fold0.strace": (
        "95435b25f8bc381c538cbccb9326e6ce9ee98faccb61f6335e4b447c150840f2"
    ),
    "fit_targets/fold0/fit_target_mask.npy": (
        "e95d34702f2f0517ab2f6487e381483ec7e5f389e329e26f96b8f0f3ce427559"
    ),
    "fit_targets/fold0/fit_targets.npy": (
        "b026bed74de2007903f6c87cc2b816dc4df932b86ac7fb325f4515da343ebee0"
    ),
    "fit_targets/fold0/manifest.json": (
        "2724d97047ee5d781b897e64007271c47bede02c17b9123d6a3751cfd7152688"
    ),
    "fit_targets/fold0/receipt.json": (
        "43f6eb88f2e7ad42b3626f54020cda17de8136316a50b0ad711abba06ba84351"
    ),
}
_FROZEN_PROTOCOL_PATH = (
    "research/02_method/"
    "labram_k31_source_native_confirmation_protocol_v13_20260811_zh.md"
)
_FROZEN_PROTOCOL_SHA256 = (
    "d109c9ba8841ec7277260138f3e6d4111caf5ec9016e5cd451265cf87fa8759b"
)
_FROZEN_AMENDMENT_PATH = (
    "research/02_method/"
    "labram_k31_source_native_confirmation_protocol_v13_1_amendment_20260811_zh.md"
)
_FROZEN_AMENDMENT_SHA256 = (
    "1747cdf701ad13be849a1e12a0fed511d3cf17d6dbd9ac1e7e3c6066cdf3c968"
)
_TEST_COMMAND = (
    "pytest -q tests/test_ictal_matched_control_v13.py "
    "tests/test_ictal_fit_control_orchestrator_v13.py"
)
# Update only after the combined suite is run; the authorization generator must
# execute the command rather than trusting this declaration on its own.
_EXPECTED_TEST_PASSED = 48

# Exact consumer-only projection.  In particular, there is no broker,
# materializer, native evaluator, target snapshot, DeepSOZ, private-data, or
# broad ``src/soz/__init__.py`` capability in the trainer namespace.
_TRAINER_CODE_RELATIVE_FILES = (
    "src/__init__.py",
    "scripts/_v13_minimal_import.py",
    "scripts/train_labram_ictal_capacity_matched_channel_control_v13.py",
    "scripts/train_labram_ictal_matched_independent_control_v13.py",
    "src/soz/cached_concept_training.py",
    "src/soz/concept_losses.py",
    "src/soz/concept_metrics.py",
    "src/soz/concept_token_io.py",
    "src/soz/concept_training.py",
    "src/soz/formal_token_corpus.py",
    "src/soz/geometry.py",
    "src/soz/ictal_fit_only_consumer_v13.py",
    "src/soz/ictal_fit_primitives_v13.py",
    "src/soz/ictal_fit_token_view_consumer_v13.py",
    "src/soz/ictal_matched_control_v13.py",
    "src/soz/models/concept_heads.py",
    "src/soz/models/foundation.py",
    "src/soz/models/labram.py",
)
_REQUIRED_AUTHORIZED_CODE_FILES = (
    "scripts/authorize_labram_ictal_fit_control_preparation_v13_1.py",
    "scripts/run_labram_ictal_fit_control_preparation_v13.py",
    "scripts/materialize_labram_ictal_fit_only_targets_v13.py",
    "scripts/materialize_labram_ictal_fit_token_view_v13.py",
    "scripts/train_labram_ictal_capacity_matched_channel_control_v13.py",
    "scripts/train_labram_ictal_matched_independent_control_v13.py",
    "scripts/_v13_minimal_import.py",
    "src/soz/cached_concept_training.py",
    "src/soz/ictal_fit_primitives_v13.py",
    "src/soz/ictal_fit_only_targets_v13.py",
    "src/soz/ictal_fit_only_consumer_v13.py",
    "src/soz/ictal_fit_token_view_v13.py",
    "src/soz/ictal_fit_token_view_consumer_v13.py",
    "src/soz/ictal_matched_control_v13.py",
    "src/soz/models/concept_heads.py",
    "tests/test_ictal_matched_control_v13.py",
    "tests/test_ictal_fit_control_orchestrator_v13.py",
)
_TRAINER_BY_CONTROL = {
    "capacity": "scripts/train_labram_ictal_capacity_matched_channel_control_v13.py",
    "naked": "scripts/train_labram_ictal_matched_independent_control_v13.py",
}
_FROZEN_EXECUTION_POLICY = {
    "selection_order": list(_SELECTION_ORDER),
    "control_candidates": [
        "labram_capacity_matched_channel_only_residual_control",
        "secondary_unmatched_capacity_diagnostic",
    ],
    "control_checkpoint_count": 12,
    "head_parameter_counts": {
        "naked_independent": 77313,
        "capacity_matched": 81665,
        "k31_temporal": 81665,
        "capacity_extra_over_independent": 4352,
        "k31_extra_over_independent": 4352,
    },
    "target_semantics": "tusz_bipolar_edge_time_involvement_not_soz",
    "fixed_epochs": 20,
    "seed": 20260808,
    "optimizer": "torch.optim.AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
    "eps": 1e-08,
    "loss": "unweighted_patient_macro_masked_bce",
    "event_microbatch_size": 4,
    "patient_order": "sorted_then_random.Random(seed+epoch).shuffle",
    "precision": "torch_float32_no_amp",
    "device": "cuda",
    "fixed_final_epoch": True,
    "early_stopping": False,
    "calibration": False,
    "evaluation_performed": False,
    "gate_opened": False,
    "i_gate_signal_or_tokens_opened": False,
    "i_gate_target_values_materialized": False,
    "outer_broker_workspace_and_source_data_reachable": True,
    "outer_broker_source_target_snapshot_reachable": True,
    "outer_broker_source_training_token_corpus_reachable": True,
    "outer_broker_gate_target_values_read": False,
    "trainer_forbidden_data_mounted": False,
    "trainer_source_broker_imported": False,
    "trainer_deepsoz_or_private_mounted": False,
    "trainer_native_or_source_evaluation_mounted": False,
    "mandatory_bwrap": True,
    "target_broker_strace_required": True,
    "target_broker_trace_audit_required": True,
    "target_broker_trace_count": 6,
    "target_broker_gate_range_overlap_allowed": False,
    "overwrite": False,
}


def _expected_sandbox_policy() -> dict[str, object]:
    return {
        "schema_version": "soz_ictal_fit_control_bwrap_policy_v13_1_v3",
        "required": True,
        "implementation": "bubblewrap",
        "executable": str(_BWRAP_EXECUTABLE),
        "executable_sha256": _file_sha256(_BWRAP_EXECUTABLE),
        "no_fallback": True,
        "unshare_all": True,
        "network_unshared": True,
        "project_root_mounted": False,
        "project_code_projection": "individual_read_only_files",
        "input_mount_scope": "current_selection_fit_only",
        "output_mount_scope": "dedicated_control_parent",
        "proc_mounted": True,
        "minimal_dev_mounted": True,
        "tmpfs_tmp_mounted": True,
        "cuda_devices": "explicit_existing_nvidia_device_nodes",
        "mount_manifest_schema": _MOUNT_MANIFEST_SCHEMA,
        "strace_required": True,
        "strace_executable": str(_STRACE_EXECUTABLE),
        "strace_executable_sha256": _file_sha256(_STRACE_EXECUTABLE),
        "strace_trace_expression": "openat,openat2,pread64",
        "strace_string_limit_bytes": _STRACE_STRING_LIMIT_BYTES,
        "strace_no_fallback": True,
    }


def _expected_target_broker_trace_policy() -> dict[str, object]:
    """Return the closed host-level broker syscall-audit policy."""

    return {
        "schema_version": "soz_ictal_fit_target_broker_trace_policy_v13_1_v3",
        "required": True,
        "host_level": True,
        "selection_count": len(_SELECTION_ORDER),
        "executable": str(_STRACE_EXECUTABLE),
        "executable_sha256": _file_sha256(_STRACE_EXECUTABLE),
        "arguments": [
            "-f",
            "-qq",
            "-s",
            str(_STRACE_STRING_LIMIT_BYTES),
            "-e",
            "trace=openat,openat2,pread64",
        ],
        "trace_path_reserved_with_o_excl": True,
        "trace_expression": "openat,openat2,pread64",
        "string_limit_bytes": _STRACE_STRING_LIMIT_BYTES,
        "source_files": ["training_targets.npy", "training_target_mask.npy"],
        "required_read_pattern": "exact_selected_fit_rows_only",
        "i_gate_range_overlap_allowed": False,
        "extra_or_missing_source_pread_allowed": False,
        "short_source_pread_allowed": False,
        "unfinished_or_unparsed_relevant_syscall_allowed": False,
        "audit_receipt_schema": _BROKER_AUDIT_SCHEMA,
        "audit_receipt_published_with_o_excl": True,
        "no_fallback": True,
    }


def _expected_supersession() -> dict[str, object]:
    """Bind v3 to the immutable v2 authorization and preserved failed run."""

    return {
        "status": "SUPERSEDED_AFTER_FAILED_PARTIAL_EXECUTION",
        "prior_authorization": {
            "path": _V2_AUTHORIZATION_PATH,
            "sha256": _V2_AUTHORIZATION_SHA256,
        },
        "prior_authorization_bytes_modified": False,
        "prior_output": {
            "path": _V2_PARTIAL_OUTPUT_ROOT,
            "preserved": True,
            "file_sha256": dict(_V2_PARTIAL_FILE_SHA256),
            "execution_receipt_present": False,
            "completed_fit_target_selections": ["fold0"],
            "target_broker_trace_formally_audited": False,
            "trainer_subprocesses_started": 0,
            "training_started": False,
        },
        "failure": {
            "stage": "fold0_target_broker_lineage_audit",
            "exception_type": "ValueError",
            "message": "fit-target source snapshot lineage changed",
            "root_cause": (
                "physical_manifest_sha_compared_to_reserialized_json_sha"
            ),
            "i_gate_target_values_read": False,
            "deepsoz_or_private_data_read": False,
        },
        "supersession_sidecar": _V2_SUPERSESSION_SIDECAR_PATH,
        "replacement": "v13.1-v3-physical-manifest-sha-lineage-fix",
    }


def _verify_superseded_v2_state() -> None:
    """Fail closed unless every retained v2 byte still matches the failure record."""

    prior = _workspace_path(
        _V2_AUTHORIZATION_PATH,
        field="supersession.prior_authorization.path",
        directory=False,
    )
    if _file_sha256(prior) != _V2_AUTHORIZATION_SHA256:
        raise ValueError("superseded v2 authorization bytes changed")
    prior_root = _workspace_path(
        _V2_PARTIAL_OUTPUT_ROOT,
        field="supersession.prior_output.path",
        directory=True,
    )
    observed: set[str] = set()
    for candidate in prior_root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError("superseded v2 output contains a symlink")
        if candidate.is_file():
            observed.add(candidate.relative_to(prior_root).as_posix())
    if observed != set(_V2_PARTIAL_FILE_SHA256):
        raise ValueError("superseded v2 partial file roster changed")
    for relative, expected in _V2_PARTIAL_FILE_SHA256.items():
        if _file_sha256(prior_root / relative) != expected:
            raise ValueError(f"superseded v2 partial file changed: {relative}")
    if (prior_root / "execution_receipt.json").exists():
        raise ValueError("superseded v2 unexpectedly contains an execution receipt")


def _assert_disjoint_from_superseded_v2(value: Path, *, field: str) -> Path:
    """Reject a new artifact path that could mutate the preserved v2 tree."""

    candidate = Path(os.path.abspath(value))
    protected = (ROOT / _V2_PARTIAL_OUTPUT_ROOT).resolve(strict=True)
    if (
        candidate == protected
        or protected in candidate.parents
        or candidate in protected.parents
    ):
        raise ValueError(f"{field} overlaps the preserved v2 output root")
    return candidate


def _assert_direct_formal_output_child(value: Path, *, field: str) -> Path:
    """Confine every new formal artifact/root to one direct ``outputs`` child."""

    candidate = _assert_disjoint_from_superseded_v2(value, field=field)
    formal_parent = (ROOT / "outputs").resolve(strict=True)
    if candidate.parent != formal_parent:
        raise ValueError(f"{field} must be a direct child of the workspace outputs root")
    return candidate


_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "authorization_status",
        "protocol",
        "amendment",
        "execution_policy",
        "sandbox_policy",
        "target_broker_trace_policy",
        "supersession",
        "preprocessing",
        "source_target_snapshot",
        "selections",
        "code_files",
        "test_receipt",
        "output_root",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object, *, field: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_canonical_new_json(path: Path, payload: Mapping[str, object]) -> str:
    """Publish one canonical receipt with kernel ``O_EXCL`` no-replace."""

    raw = _canonical_json_bytes(dict(payload))
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return hashlib.sha256(raw).hexdigest()


def _create_empty_new_file(path: Path) -> None:
    """Reserve a trace path with O_EXCL before handing it to strace."""

    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.parent.is_dir() or source.parent.is_symlink():
        raise ValueError("trace log requires a regular existing parent")
    descriptor = os.open(source, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _audit_strace_log(
    path: Path, *, forbidden_roots: Sequence[Path]
) -> dict[str, object]:
    """Seal the open-file trace and fail on any declared forbidden data root."""

    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_file() or source.resolve() != source:
        raise RuntimeError("trainer strace log is not a regular immutable-path file")
    size = source.stat().st_size
    if size < 1:
        raise RuntimeError("trainer strace log is empty; ptrace confinement is unproven")
    raw = source.read_bytes()
    if b"openat(" not in raw and b"openat2(" not in raw:
        raise RuntimeError("trainer strace log contains no open-file syscall")
    text = raw.decode("utf-8", errors="replace")
    normalized_forbidden = []
    for value in forbidden_roots:
        forbidden = Path(os.path.abspath(value)).resolve()
        normalized_forbidden.append(str(forbidden))
        if str(forbidden) in text:
            raise RuntimeError(f"trainer trace opened a forbidden data root: {forbidden}")
    return {
        "strace_log_sha256": hashlib.sha256(raw).hexdigest(),
        "strace_log_size_bytes": size,
        "strace_open_file_syscall_present": True,
        "forbidden_root_count_checked": len(normalized_forbidden),
        "forbidden_root_open_detected": False,
    }


def _read_canonical_json_metadata(
    path: Path, *, expected_sha256: str, field: str
) -> tuple[dict[str, object], str]:
    """Read one bounded metadata file without opening any tensor payload."""

    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_file() or source.resolve() != source:
        raise ValueError(f"{field} must be a regular absolute file")
    metadata_before = source.stat()
    if not 1 <= metadata_before.st_size <= _MAX_AUTHORIZATION_BYTES:
        raise ValueError(f"{field} has an invalid metadata size")
    raw = source.read_bytes()
    metadata_after = source.stat()
    identity_before = (
        metadata_before.st_dev,
        metadata_before.st_ino,
        metadata_before.st_size,
        metadata_before.st_mtime_ns,
    )
    identity_after = (
        metadata_after.st_dev,
        metadata_after.st_ino,
        metadata_after.st_size,
        metadata_after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise RuntimeError(f"{field} changed while its metadata was read")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _sha256(expected_sha256, field=f"{field}.sha256"):
        raise ValueError(f"{field} SHA mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is invalid JSON") from exc
    if (
        not isinstance(payload, dict)
        or raw not in {_canonical_json_bytes(payload), _canonical_json_bytes(payload) + b"\n"}
    ):
        raise ValueError(f"{field} is not canonical JSON metadata")
    return payload, digest


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _broker_source_layout(
    source_bundle: Path,
    *,
    tensor_record: object,
    tensor_name: str,
    row_count: int,
    expected_filename: str,
    expected_dtype: str,
    row_size_bytes: int,
) -> dict[str, object]:
    """Reconstruct a source NPY layout from sealed metadata only."""

    if not isinstance(tensor_record, Mapping):
        raise TypeError(f"{tensor_name} tensor record must be a mapping")
    record = dict(tensor_record)
    required = {
        "dtype",
        "file_sha256",
        "file_size_bytes",
        "filename",
        "shape",
        "tensor_sha256",
    }
    if set(record) != required:
        raise ValueError(f"{tensor_name} tensor record violates its closed schema")
    if (
        record["filename"] != expected_filename
        or record["dtype"] != expected_dtype
        or record["shape"] != [row_count, 20, 60]
    ):
        raise ValueError(f"{tensor_name} source layout changed")
    declared_size = record["file_size_bytes"]
    if (
        isinstance(declared_size, bool)
        or not isinstance(declared_size, int)
        or declared_size <= row_count * row_size_bytes
    ):
        raise ValueError(f"{tensor_name} source size is invalid")
    _sha256(record["file_sha256"], field=f"{tensor_name}.file_sha256")
    _sha256(record["tensor_sha256"], field=f"{tensor_name}.tensor_sha256")
    data_offset = declared_size - row_count * row_size_bytes
    if not 1 <= data_offset <= 64 * 1024:
        raise ValueError(f"{tensor_name} inferred NPY header size is invalid")
    lexical_source_path = source_bundle / expected_filename
    if lexical_source_path.is_symlink():
        raise ValueError(f"{tensor_name} source file cannot be a symlink")
    source_path = lexical_source_path.resolve()
    if (
        source_bundle not in source_path.parents
        or source_path.is_symlink()
        or not source_path.is_file()
        or source_path.stat().st_size != declared_size
    ):
        raise ValueError(f"{tensor_name} source file metadata changed")
    return {
        "path": source_path,
        "row_size_bytes": row_size_bytes,
        "data_offset_bytes": data_offset,
        "declared_file_sha256": record["file_sha256"],
    }


def _target_broker_range_plan(
    *,
    source_bundle: Path,
    source_manifest: Mapping[str, object],
    source_manifest_sha256: str,
    fit_manifest: Mapping[str, object],
    selection: str,
) -> dict[str, object]:
    """Reconstruct selected and I-gate byte ranges without reading array rows."""

    if source_manifest.get("schema_version") != "soz_ictal_native_prediction_artifact_v1":
        raise ValueError("source target snapshot schema changed")
    if (
        fit_manifest.get("schema_version")
        != "soz_ictal_fit_only_target_artifact_v13_1"
        or fit_manifest.get("target_semantics")
        != _FROZEN_EXECUTION_POLICY["target_semantics"]
    ):
        raise ValueError("fit-target schema or target semantics changed")
    source_rows_value = source_manifest.get("training_event_rows")
    if not isinstance(source_rows_value, list) or not source_rows_value:
        raise ValueError("source training_event_rows is missing")
    source_rows: list[tuple[str, str]] = []
    for value in source_rows_value:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("source training_event_rows schema changed")
        event_id, patient_id = value
        if not isinstance(event_id, str) or not event_id or not isinstance(patient_id, str) or not patient_id:
            raise ValueError("source training_event_rows identity is invalid")
        source_rows.append((event_id, patient_id))
    if len({event_id for event_id, _ in source_rows}) != len(source_rows):
        raise ValueError("source training_event_rows contains duplicate events")
    tensor_files = source_manifest.get("tensor_files")
    if not isinstance(tensor_files, Mapping):
        raise TypeError("source tensor_files must be a mapping")
    target_layout = _broker_source_layout(
        source_bundle,
        tensor_record=tensor_files.get("training_targets"),
        tensor_name="training_targets",
        row_count=len(source_rows),
        expected_filename="training_targets.npy",
        expected_dtype="torch.float32",
        row_size_bytes=20 * 60 * 4,
    )
    mask_layout = _broker_source_layout(
        source_bundle,
        tensor_record=tensor_files.get("training_target_mask"),
        tensor_name="training_target_mask",
        row_count=len(source_rows),
        expected_filename="training_target_mask.npy",
        expected_dtype="torch.bool",
        row_size_bytes=20 * 60,
    )
    if (
        fit_manifest.get("source_target_snapshot_manifest_sha256")
        != _sha256(
            source_manifest_sha256,
            field="source_target_snapshot.manifest.sha256",
        )
        or fit_manifest.get("source_training_targets_declared_file_sha256")
        != target_layout["declared_file_sha256"]
        or fit_manifest.get("source_training_target_mask_declared_file_sha256")
        != mask_layout["declared_file_sha256"]
    ):
        raise ValueError("fit-target source snapshot lineage changed")
    if fit_manifest.get("selection") != selection:
        raise ValueError("fit-target manifest selection differs from broker audit")
    fit_rows_value = fit_manifest.get("fit_event_rows")
    fit_count = fit_manifest.get("fit_event_count")
    if (
        not isinstance(fit_rows_value, list)
        or isinstance(fit_count, bool)
        or not isinstance(fit_count, int)
        or fit_count < 1
        or len(fit_rows_value) != fit_count
    ):
        raise ValueError("fit-target event roster changed")
    selected_indices: list[int] = []
    selected_patients: set[str] = set()
    for value in fit_rows_value:
        if not isinstance(value, list) or len(value) != 7:
            raise ValueError("fit-target event row schema changed")
        event_id, patient_id, source_index = value[:3]
        if (
            not isinstance(event_id, str)
            or not isinstance(patient_id, str)
            or isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or not 0 <= source_index < len(source_rows)
            or source_rows[source_index] != (event_id, patient_id)
        ):
            raise ValueError("fit-target event/source identity changed")
        selected_indices.append(source_index)
        selected_patients.add(patient_id)
    if selected_indices != sorted(selected_indices) or len(set(selected_indices)) != len(selected_indices):
        raise ValueError("fit-target source indices are not strictly ordered and unique")
    fit_patients = fit_manifest.get("fit_patient_ids")
    gate_patients = fit_manifest.get("i_gate_patient_ids_excluded_unopened")
    if (
        not isinstance(fit_patients, list)
        or not fit_patients
        or any(not isinstance(value, str) or not value for value in fit_patients)
        or len(set(fit_patients)) != len(fit_patients)
        or set(fit_patients) != selected_patients
        or not isinstance(gate_patients, list)
        or len(gate_patients) != 12
        or any(not isinstance(value, str) or not value for value in gate_patients)
        or len(set(gate_patients)) != len(gate_patients)
        or set(fit_patients) & set(gate_patients)
    ):
        raise ValueError("fit/I-gate patient firewall changed")
    gate_set = set(gate_patients)
    gate_indices = [
        index for index, (_, patient_id) in enumerate(source_rows) if patient_id in gate_set
    ]
    if not gate_indices or set(selected_indices) & set(gate_indices):
        raise ValueError("selected source rows overlap or omit the I-gate roster")

    def ranges(layout: Mapping[str, object], indices: Sequence[int]) -> list[list[int]]:
        offset = int(layout["data_offset_bytes"])
        size = int(layout["row_size_bytes"])
        return [[index, offset + index * size, offset + (index + 1) * size] for index in indices]

    selected_target_ranges = ranges(target_layout, selected_indices)
    selected_mask_ranges = ranges(mask_layout, selected_indices)
    gate_target_ranges = ranges(target_layout, gate_indices)
    gate_mask_ranges = ranges(mask_layout, gate_indices)
    expected_hashes = {
        "selected_source_row_indices_sha256": _canonical_sha256(selected_indices),
        "selected_target_byte_ranges_sha256": _canonical_sha256(selected_target_ranges),
        "selected_mask_byte_ranges_sha256": _canonical_sha256(selected_mask_ranges),
    }
    for field, expected in expected_hashes.items():
        if fit_manifest.get(field) != expected:
            raise ValueError(f"fit-target manifest {field} differs from reconstructed ranges")
    return {
        "target_layout": target_layout,
        "mask_layout": mask_layout,
        "selected_target_ranges": selected_target_ranges,
        "selected_mask_ranges": selected_mask_ranges,
        "gate_target_ranges": gate_target_ranges,
        "gate_mask_ranges": gate_mask_ranges,
        "selected_source_row_indices_sha256": expected_hashes[
            "selected_source_row_indices_sha256"
        ],
    }


_TRACE_LINE_RE = re.compile(
    r"^\s*(?:(?P<pid>[0-9]+)\s+|\[pid\s+(?P<bracket_pid>[0-9]+)\]\s+)?(?P<body>.*)$"
)
_TRACE_OPEN_RE = re.compile(
    r'^(?P<name>openat|openat2)\([^,]+,\s*(?P<path>"(?:\\.|[^"\\])*").*\)\s*=\s*(?P<fd>-?[0-9]+)(?:\s+.*)?$'
)
_TRACE_PREAD_RE = re.compile(
    r"^pread64\(\s*(?P<fd>[0-9]+)\s*,.*," 
    r"\s*(?P<count>0x[0-9a-fA-F]+|[0-9]+)\s*,"
    r"\s*(?P<offset>0x[0-9a-fA-F]+|[0-9]+)\s*\)\s*="
    r"\s*(?P<returned>-?[0-9]+)(?:\s+.*)?$"
)
_TRACE_UNFINISHED_RE = re.compile(
    r"^(?P<prefix>(?P<name>openat|openat2|pread64)\(.*)\s+<unfinished \.\.\.>$"
)
_TRACE_RESUMED_RE = re.compile(
    r"^<\.\.\.\s+(?P<name>openat|openat2|pread64)\s+resumed>"
    r"(?P<suffix>.*)$"
)


def _reassemble_target_broker_trace(
    text: str,
) -> list[tuple[int, str, str]]:
    """Strictly pair strace unfinished/resumed records by PID and syscall."""

    pending: dict[str, tuple[str, str, int]] = {}
    complete: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        matched_line = _TRACE_LINE_RE.fullmatch(line)
        if matched_line is None:
            raise RuntimeError(f"unparsed target-broker strace line {line_number}")
        pid = matched_line.group("pid") or matched_line.group("bracket_pid") or "main"
        body = matched_line.group("body")
        unfinished = _TRACE_UNFINISHED_RE.fullmatch(body)
        if unfinished is not None:
            if pid in pending:
                raise RuntimeError("duplicate unfinished relevant syscall is not auditable")
            pending[pid] = (
                unfinished.group("name"),
                unfinished.group("prefix"),
                line_number,
            )
            continue
        resumed = _TRACE_RESUMED_RE.fullmatch(body)
        if resumed is not None:
            prior = pending.pop(pid, None)
            if prior is None or prior[0] != resumed.group("name"):
                raise RuntimeError("resumed relevant syscall is not auditable")
            complete.append((prior[2], pid, prior[1] + resumed.group("suffix")))
            continue
        if pid in pending and body.startswith(("openat(", "openat2(", "pread64(")):
            raise RuntimeError(
                "completed relevant syscall while the same PID is unfinished"
            )
        if (
            any(name in body for name in ("openat(", "openat2(", "pread64("))
            and ("<unfinished ...>" in body or "resumed>" in body)
        ) or re.search(r"<\.\.\.\s+(?:openat|openat2|pread64)\s+resumed>", body):
            raise RuntimeError("unfinished/resumed relevant syscall is not auditable")
        complete.append((line_number, pid, body))
    if pending:
        raise RuntimeError("unfinished/resumed relevant syscall is not auditable")
    return complete


def _parse_target_broker_trace(
    raw: bytes, *, target_path: Path, mask_path: Path
) -> dict[str, object]:
    """Parse source-file descriptor mappings and all relevant pread64 calls."""

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("target-broker strace is not UTF-8") from exc
    relevant = {
        os.path.normpath(str(target_path)): "target",
        os.path.normpath(str(mask_path)): "mask",
    }
    descriptor_paths: dict[tuple[str, int], str] = {}
    calls: dict[str, list[tuple[int, int, int]]] = {"target": [], "mask": []}
    open_counts = {"target": 0, "mask": 0}
    for line_number, pid, body in _reassemble_target_broker_trace(text):
        if body.startswith(("openat(", "openat2(")):
            opened = _TRACE_OPEN_RE.fullmatch(body)
            if opened is None:
                raise RuntimeError(f"unparsed open-file syscall at trace line {line_number}")
            try:
                decoded_path = ast.literal_eval(opened.group("path"))
            except (SyntaxError, ValueError) as exc:
                raise RuntimeError("strace open path cannot be decoded") from exc
            if not isinstance(decoded_path, str):
                raise RuntimeError("strace open path is not a string")
            fd = int(opened.group("fd"))
            if fd >= 0:
                normalized = os.path.normpath(decoded_path)
                descriptor_paths[(pid, fd)] = normalized
                kind = relevant.get(normalized)
                if kind is not None:
                    open_counts[kind] += 1
            continue
        if body.startswith("pread64("):
            pread = _TRACE_PREAD_RE.fullmatch(body)
            if pread is None:
                raise RuntimeError(f"unparsed pread64 syscall at trace line {line_number}")
            fd = int(pread.group("fd"))
            opened_path = descriptor_paths.get((pid, fd))
            kind = relevant.get(opened_path) if opened_path is not None else None
            if kind is not None:
                calls[kind].append(
                    (
                        int(pread.group("offset"), 0),
                        int(pread.group("count"), 0),
                        int(pread.group("returned")),
                    )
                )
    if open_counts["target"] < 1 or open_counts["mask"] < 1:
        raise RuntimeError("target-broker trace omitted a source target/mask open")
    return {"calls": calls, "open_counts": open_counts}


def _audit_target_broker_strace(
    trace_path: Path,
    *,
    selection: str,
    source_snapshot_bundle: Path,
    expected_source_manifest_sha256: str,
    expected_source_receipt_sha256: str,
    fit_target_bundle: Path,
    expected_fit_manifest_sha256: str,
    expected_fit_receipt_sha256: str,
    authorization_sha256: str,
) -> dict[str, object]:
    """Prove one broker read exactly selected rows and no I-gate byte range."""

    if selection not in _SELECTION_ORDER:
        raise ValueError("broker audit selection is invalid")
    source_bundle = _regular_absolute_directory(
        source_snapshot_bundle, field="source_snapshot_bundle"
    )
    fit_bundle = _regular_absolute_directory(fit_target_bundle, field="fit_target_bundle")
    source_manifest, source_manifest_sha = _read_canonical_json_metadata(
        source_bundle / "manifest.json",
        expected_sha256=expected_source_manifest_sha256,
        field="source_target_snapshot.manifest",
    )
    source_receipt, source_receipt_sha = _read_canonical_json_metadata(
        source_bundle / "receipt.json",
        expected_sha256=expected_source_receipt_sha256,
        field="source_target_snapshot.receipt",
    )
    fit_manifest, fit_manifest_sha = _read_canonical_json_metadata(
        fit_bundle / "manifest.json",
        expected_sha256=expected_fit_manifest_sha256,
        field="fit_target.manifest",
    )
    fit_receipt, fit_receipt_sha = _read_canonical_json_metadata(
        fit_bundle / "receipt.json",
        expected_sha256=expected_fit_receipt_sha256,
        field="fit_target.receipt",
    )
    if (
        source_receipt.get("schema_version")
        != "soz_ictal_native_prediction_bundle_receipt_v1"
        or source_receipt.get("artifact_sha256") != source_manifest_sha
    ):
        raise ValueError("source target snapshot receipt does not bind its manifest")
    if fit_receipt != {
        "schema_version": "soz_ictal_fit_only_target_receipt_v13_1",
        "artifact_sha256": fit_manifest_sha,
        "artifact_size_bytes": (fit_bundle / "manifest.json").stat().st_size,
    }:
        raise ValueError("fit-target receipt does not bind its manifest")
    if (
        fit_manifest.get("source_target_snapshot_manifest_sha256")
        != source_manifest_sha
        or fit_manifest.get("source_target_snapshot_receipt_sha256")
        != source_receipt_sha
    ):
        raise ValueError("fit-target manifest does not bind the audited source snapshot")
    authorization_sha = _sha256(
        authorization_sha256, field="broker_audit.authorization_sha256"
    )
    plan = _target_broker_range_plan(
        source_bundle=source_bundle,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha,
        fit_manifest=fit_manifest,
        selection=selection,
    )
    trace = Path(os.path.abspath(trace_path))
    if trace.is_symlink() or not trace.is_file() or trace.resolve() != trace:
        raise RuntimeError("target-broker strace must be a regular absolute file")
    before = trace.stat()
    if before.st_size < 1:
        raise RuntimeError("target-broker strace is empty")
    raw = trace.read_bytes()
    after = trace.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("target-broker strace changed while audited")
    parsed = _parse_target_broker_trace(
        raw,
        target_path=Path(str(plan["target_layout"]["path"])),
        mask_path=Path(str(plan["mask_layout"]["path"])),
    )

    def expected_calls(ranges: Sequence[Sequence[int]]) -> list[tuple[int, int, int]]:
        return [(int(start), int(stop) - int(start), int(stop) - int(start)) for _, start, stop in ranges]

    target_expected = expected_calls(plan["selected_target_ranges"])
    mask_expected = expected_calls(plan["selected_mask_ranges"])
    target_actual = parsed["calls"]["target"]
    mask_actual = parsed["calls"]["mask"]

    def overlap_count(
        calls: Sequence[tuple[int, int, int]], gate_ranges: Sequence[Sequence[int]]
    ) -> int:
        return sum(
            1
            for offset, count, _ in calls
            for _, gate_start, gate_stop in gate_ranges
            if offset < int(gate_stop) and offset + count > int(gate_start)
        )

    gate_overlap_count = overlap_count(
        target_actual, plan["gate_target_ranges"]
    ) + overlap_count(mask_actual, plan["gate_mask_ranges"])
    if gate_overlap_count:
        raise RuntimeError("target-broker pread64 overlaps an I-gate byte range")
    if target_actual != target_expected:
        raise RuntimeError("target-broker target pread64 calls differ from selected fit rows")
    if mask_actual != mask_expected:
        raise RuntimeError("target-broker mask pread64 calls differ from selected fit rows")
    trace_sha = hashlib.sha256(raw).hexdigest()
    return {
        "schema_version": _BROKER_AUDIT_SCHEMA,
        "selection": selection,
        "authorization_sha256": authorization_sha,
        "source_target_snapshot_manifest_sha256": source_manifest_sha,
        "source_target_snapshot_receipt_sha256": source_receipt_sha,
        "fit_target_manifest_sha256": fit_manifest_sha,
        "fit_target_receipt_sha256": fit_receipt_sha,
        "selected_source_row_indices_sha256": plan[
            "selected_source_row_indices_sha256"
        ],
        "selected_target_byte_ranges_sha256": _canonical_sha256(
            plan["selected_target_ranges"]
        ),
        "selected_target_range_count": len(plan["selected_target_ranges"]),
        "selected_mask_byte_ranges_sha256": _canonical_sha256(
            plan["selected_mask_ranges"]
        ),
        "selected_mask_range_count": len(plan["selected_mask_ranges"]),
        "i_gate_target_byte_ranges_sha256": _canonical_sha256(
            plan["gate_target_ranges"]
        ),
        "i_gate_target_range_count": len(plan["gate_target_ranges"]),
        "i_gate_mask_byte_ranges_sha256": _canonical_sha256(
            plan["gate_mask_ranges"]
        ),
        "i_gate_mask_range_count": len(plan["gate_mask_ranges"]),
        "source_target_open_count": parsed["open_counts"]["target"],
        "source_mask_open_count": parsed["open_counts"]["mask"],
        "traced_target_pread64_call_count": len(target_actual),
        "traced_mask_pread64_call_count": len(mask_actual),
        "exact_selected_pread64_match": True,
        "i_gate_range_overlap_count": 0,
        "i_gate_target_or_mask_bytes_read": False,
        "unfinished_or_unparsed_relevant_syscall_detected": False,
        "strace_expression": "openat,openat2,pread64",
        "strace_string_limit_bytes": _STRACE_STRING_LIMIT_BYTES,
        "strace_log_reserved_with_o_excl": True,
        "strace_log_sha256": trace_sha,
        "strace_log_size_bytes": len(raw),
        "audit_passed": True,
    }


def _regular_absolute_directory(path: Path, *, field: str) -> Path:
    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_dir() or source.resolve() != source:
        raise ValueError(f"{field} must be a regular absolute directory")
    return source


def _discover_cuda_device_nodes() -> tuple[Path, ...]:
    """Return only explicit NVIDIA character devices; never broad-bind /dev."""

    candidates = set(Path("/dev").glob("nvidia*"))
    candidates.update(Path("/dev/nvidia-caps").glob("nvidia-cap*"))
    candidates.update(Path("/dev/dri").glob("renderD*"))
    devices = []
    for candidate in sorted(candidates, key=str):
        try:
            mode = candidate.stat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISCHR(mode):
            devices.append(candidate)
    if not devices:
        raise RuntimeError(
            "mandatory CUDA sandbox has no explicit NVIDIA device nodes to mount"
        )
    return tuple(devices)


def _trainer_code_projection() -> tuple[tuple[Path, Path], ...]:
    projection = []
    for relative in _TRAINER_CODE_RELATIVE_FILES:
        source = (ROOT / relative).resolve()
        if (
            ROOT not in source.parents
            or source.is_symlink()
            or not source.is_file()
        ):
            raise RuntimeError(f"trainer code projection is invalid: {relative}")
        projection.append((source, _SANDBOX_PROJECT_ROOT / relative))
    return tuple(projection)


def _mount_parent_directories(destinations: Sequence[Path]) -> tuple[Path, ...]:
    parents: set[Path] = set()
    for destination in destinations:
        current = destination.parent
        while current != Path("/"):
            parents.add(current)
            current = current.parent
    return tuple(sorted(parents, key=lambda item: (len(item.parts), str(item))))


def _assert_closed_trainer_mounts(
    mounts: Sequence[Mapping[str, object]],
    *,
    fit_target: Path,
    fit_token: Path,
    output_parent: Path,
) -> None:
    """Reject broad workspace/data mounts before bwrap can be invoked."""

    exact_workspace_allowlist = {
        *(source for source, _ in _trainer_code_projection()),
        fit_target,
        fit_token,
        output_parent,
    }
    destinations: set[str] = set()
    for mount in mounts:
        if set(mount) != {"kind", "source", "destination", "mode"}:
            raise RuntimeError("trainer mount record violates its closed schema")
        destination = str(mount["destination"])
        if destination in destinations:
            raise RuntimeError("trainer mount destinations must be unique")
        destinations.add(destination)
        source = Path(str(mount["source"]))
        resolved = source.resolve()
        if resolved == ROOT or (ROOT in resolved.parents and resolved not in exact_workspace_allowlist):
            raise RuntimeError(f"broad or unauthorized workspace mount rejected: {source}")
    required = {
        str(_SANDBOX_FIT_TARGET),
        str(_SANDBOX_FIT_TOKEN),
        str(_SANDBOX_OUTPUT_PARENT),
    }
    if not required.issubset(destinations):
        raise RuntimeError("trainer sandbox omitted a required isolated mount")
    if any(
        mount["mode"] == "rw"
        and str(mount["destination"]) != str(_SANDBOX_OUTPUT_PARENT)
        for mount in mounts
    ):
        raise RuntimeError("trainer sandbox has a write mount outside its output parent")


def _build_trainer_sandbox_invocation(
    *,
    selection: str,
    control: str,
    fit_target_bundle: Path,
    fit_token_bundle: Path,
    output_parent: Path,
    target_manifest_sha256: str,
    target_receipt_sha256: str,
    token_manifest_sha256: str,
    token_receipt_sha256: str,
    authorization_sha256: str,
    trace_log_path: Path,
    cuda_device_nodes: Sequence[Path],
) -> tuple[list[str], dict[str, object]]:
    """Build the only permitted formal trainer command and mount receipt.

    This is deliberately a pure builder.  Formal execution obtains
    ``cuda_device_nodes`` exclusively from :func:`_discover_cuda_device_nodes`.
    There is no unsandboxed trainer fallback.
    """

    if selection not in _SELECTION_ORDER:
        raise ValueError("trainer selection is not in the frozen six-selection order")
    if control not in _CONTROL_ORDER:
        raise ValueError("trainer control must be capacity or naked")
    fit_target = _regular_absolute_directory(
        fit_target_bundle, field="fit_target_bundle"
    )
    fit_token = _regular_absolute_directory(fit_token_bundle, field="fit_token_bundle")
    output = _regular_absolute_directory(output_parent, field="output_parent")
    trace_log = Path(os.path.abspath(trace_log_path))
    if (
        trace_log.name in {"", ".", ".."}
        or trace_log.is_symlink()
        or not trace_log.parent.is_dir()
        or trace_log.parent.is_symlink()
        or os.path.lexists(trace_log)
    ):
        raise FileExistsError("trainer strace path must be new with a regular parent")
    if os.path.lexists(output / "bundle"):
        raise FileExistsError(f"trainer output already exists: {output / 'bundle'}")
    for value, field in (
        (target_manifest_sha256, "target_manifest_sha256"),
        (target_receipt_sha256, "target_receipt_sha256"),
        (token_manifest_sha256, "token_manifest_sha256"),
        (token_receipt_sha256, "token_receipt_sha256"),
        (authorization_sha256, "authorization_sha256"),
    ):
        _sha256(value, field=field)

    python_runtime = Path(sys.prefix).resolve()
    if (
        python_runtime == ROOT
        or ROOT in python_runtime.parents
        or not python_runtime.is_dir()
        or not (python_runtime / "bin" / "python3").exists()
    ):
        raise RuntimeError("Python runtime cannot be projected into the trainer sandbox")
    system_sources = (Path("/usr"), Path("/etc/ld.so.cache"))
    if any(not source.exists() for source in system_sources):
        raise RuntimeError("required read-only system runtime path is missing")
    code_projection = _trainer_code_projection()
    devices = tuple(Path(value) for value in cuda_device_nodes)
    if not devices or len(set(devices)) != len(devices):
        raise RuntimeError("formal CUDA sandbox requires unique explicit device nodes")
    if any(
        not device.is_absolute()
        or not (
            str(device).startswith("/dev/nvidia")
            or str(device).startswith("/dev/dri/renderD")
        )
        for device in devices
    ):
        raise RuntimeError("CUDA sandbox device allowlist contains a non-NVIDIA node")

    mounts: list[dict[str, object]] = [
        {"kind": "system_runtime", "source": "/usr", "destination": "/usr", "mode": "ro"},
        {
            "kind": "system_runtime",
            "source": "/etc/ld.so.cache",
            "destination": "/etc/ld.so.cache",
            "mode": "ro",
        },
        {
            "kind": "python_cuda_runtime",
            "source": str(python_runtime),
            "destination": str(_SANDBOX_PYTHON_ROOT),
            "mode": "ro",
        },
    ]
    mounts.extend(
        {
            "kind": "project_code_file",
            "source": str(source),
            "destination": str(destination),
            "mode": "ro",
        }
        for source, destination in code_projection
    )
    mounts.extend(
        (
            {
                "kind": "selection_fit_target",
                "source": str(fit_target),
                "destination": str(_SANDBOX_FIT_TARGET),
                "mode": "ro",
            },
            {
                "kind": "selection_fit_token",
                "source": str(fit_token),
                "destination": str(_SANDBOX_FIT_TOKEN),
                "mode": "ro",
            },
            {
                "kind": "dedicated_control_output_parent",
                "source": str(output),
                "destination": str(_SANDBOX_OUTPUT_PARENT),
                "mode": "rw",
            },
        )
    )
    mounts.extend(
        {
            "kind": "cuda_device",
            "source": str(device),
            "destination": str(device),
            "mode": "device",
        }
        for device in devices
    )
    _assert_closed_trainer_mounts(
        mounts, fit_target=fit_target, fit_token=fit_token, output_parent=output
    )

    bind_destinations = [
        Path(str(mount["destination"]))
        for mount in mounts
        if mount["kind"] not in {"system_runtime", "cuda_device"}
    ]
    command = [
        str(_BWRAP_EXECUTABLE),
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--clearenv",
        "--ro-bind",
        "/usr",
        "/usr",
        "--dir",
        "/etc",
        "--ro-bind",
        "/etc/ld.so.cache",
        "/etc/ld.so.cache",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib64",
        "/lib64",
    ]
    for parent in _mount_parent_directories(bind_destinations):
        command.extend(("--dir", str(parent)))
    command.extend(
        (
            "--ro-bind",
            str(python_runtime),
            str(_SANDBOX_PYTHON_ROOT),
        )
    )
    for source, destination in code_projection:
        command.extend(("--ro-bind", str(source), str(destination)))
    command.extend(
        (
            "--ro-bind",
            str(fit_target),
            str(_SANDBOX_FIT_TARGET),
            "--ro-bind",
            str(fit_token),
            str(_SANDBOX_FIT_TOKEN),
            "--bind",
            str(output),
            str(_SANDBOX_OUTPUT_PARENT),
            "--proc",
            "/proc",
            "--dev",
            "/dev",
        )
    )
    device_parents = sorted(
        {device.parent for device in devices if device.parent != Path("/dev")},
        key=lambda item: (len(item.parts), str(item)),
    )
    for parent in device_parents:
        command.extend(("--dir", str(parent)))
    for device in devices:
        command.extend(("--dev-bind", str(device), str(device)))
    command.extend(
        (
            "--tmpfs",
            "/tmp",
            "--setenv",
            "PATH",
            f"{_SANDBOX_PYTHON_ROOT}/bin:/usr/bin:/bin",
            "--setenv",
            "PYTHONPATH",
            str(_SANDBOX_PROJECT_ROOT),
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "CUBLAS_WORKSPACE_CONFIG",
            ":4096:8",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--chdir",
            str(_SANDBOX_PROJECT_ROOT),
        )
    )
    inner_command = [
        str(_SANDBOX_PYTHON_ROOT / "bin" / "python3"),
        str(_SANDBOX_PROJECT_ROOT / _TRAINER_BY_CONTROL[control]),
        "--selection",
        selection,
        "--fit-token-view-bundle",
        str(_SANDBOX_FIT_TOKEN),
        "--expected-fit-token-view-manifest-sha256",
        token_manifest_sha256,
        "--expected-fit-token-view-receipt-sha256",
        token_receipt_sha256,
        "--fit-only-target-bundle",
        str(_SANDBOX_FIT_TARGET),
        "--expected-fit-only-target-manifest-sha256",
        target_manifest_sha256,
        "--expected-fit-only-target-receipt-sha256",
        target_receipt_sha256,
        "--output-directory",
        str(_SANDBOX_OUTPUT_BUNDLE),
        "--device",
        "cuda",
    ]
    command.extend(inner_command)
    command = [
        str(_STRACE_EXECUTABLE),
        "-f",
        "-qq",
        "-s",
        str(_STRACE_STRING_LIMIT_BYTES),
        "-e",
        "trace=openat,openat2,pread64",
        "-o",
        str(trace_log),
        "--",
        *command,
    ]
    manifest = {
        "schema_version": _MOUNT_MANIFEST_SCHEMA,
        "authorization_sha256": authorization_sha256,
        "selection": selection,
        "control": control,
        "trainer": _TRAINER_BY_CONTROL[control],
        "sandbox_required": True,
        "sandbox_implementation": "bubblewrap",
        "sandbox_executable": str(_BWRAP_EXECUTABLE),
        "sandbox_executable_sha256": _file_sha256(_BWRAP_EXECUTABLE),
        "no_unsandboxed_fallback": True,
        "unshare_all": True,
        "network_unshared": True,
        "project_root_mounted": False,
        "outer_broker_workspace_and_source_data_reachable": True,
        "outer_broker_gate_target_values_read": False,
        "trainer_forbidden_data_mounted": False,
        "trainer_source_broker_imported": False,
        "trainer_deepsoz_or_private_mounted": False,
        "trainer_native_or_source_evaluation_mounted": False,
        "trainer_master_token_corpus_mounted": False,
        "trainer_k31_recovery_bundle_mounted": False,
        "proc_mounted": True,
        "minimal_dev_mounted": True,
        "tmpfs_tmp_mounted": True,
        "strace_required": True,
        "strace_executable": str(_STRACE_EXECUTABLE),
        "strace_executable_sha256": _file_sha256(_STRACE_EXECUTABLE),
        "strace_trace_expression": "openat,openat2,pread64",
        "strace_string_limit_bytes": _STRACE_STRING_LIMIT_BYTES,
        "strace_log": str(trace_log),
        "strace_log_reserved_with_o_excl": True,
        "strace_no_fallback": True,
        "mounts": mounts,
        "inner_command": inner_command,
    }
    return command, manifest


def _workspace_path(value: object, *, field: str, directory: bool) -> Path:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise ValueError(f"{field} must be a workspace-relative path")
    path = (ROOT / value).resolve()
    if ROOT not in path.parents:
        raise ValueError(f"{field} escapes the workspace")
    if directory and not path.is_dir():
        raise ValueError(f"{field} directory is missing")
    if not directory and not path.is_file():
        raise ValueError(f"{field} file is missing")
    return path


def _strict_authorization(
    path: Path, *, expected_sha256: str
) -> tuple[dict[str, object], str]:
    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_file() or source.resolve() != source:
        raise ValueError("authorization must be a regular absolute file")
    raw = source.read_bytes()
    if not 1 <= len(raw) <= _MAX_AUTHORIZATION_BYTES:
        raise ValueError("authorization size is invalid")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _sha256(expected_sha256, field="expected_authorization_sha256"):
        raise ValueError("authorization SHA mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("authorization is invalid JSON") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != _AUTHORIZATION_FIELDS
        or _canonical_json_bytes(payload) != raw
    ):
        raise ValueError("authorization violates its canonical closed schema")
    return payload, digest


def _verify_file_record(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    record = dict(value)
    if set(record) != {"path", "sha256"}:
        raise ValueError(f"{field} violates its closed schema")
    path = _workspace_path(record["path"], field=f"{field}.path", directory=False)
    digest = _sha256(record["sha256"], field=f"{field}.sha256")
    if _file_sha256(path) != digest:
        raise ValueError(f"{field} file SHA changed")
    return {"path": str(record["path"]), "sha256": digest}


def _validate_authorization(payload: Mapping[str, object]) -> dict[str, object]:
    authorization = dict(payload)
    if authorization["schema_version"] != _AUTHORIZATION_SCHEMA:
        raise ValueError("unsupported fit-control authorization schema")
    if authorization["authorization_status"] != (
        "AUTHORIZED_FIT_ONLY_PREPARATION_NO_GATE_NO_EVALUATION"
    ):
        raise ValueError("fit-control preparation is not explicitly authorized")
    protocol = _verify_file_record(authorization["protocol"], field="protocol")
    if protocol != {"path": _FROZEN_PROTOCOL_PATH, "sha256": _FROZEN_PROTOCOL_SHA256}:
        raise ValueError("authorization does not bind the frozen v13 protocol")
    amendment = _verify_file_record(authorization["amendment"], field="amendment")
    if amendment != {
        "path": _FROZEN_AMENDMENT_PATH,
        "sha256": _FROZEN_AMENDMENT_SHA256,
    }:
        raise ValueError("authorization does not bind the frozen v13.1 amendment")
    policy = authorization["execution_policy"]
    if not isinstance(policy, Mapping):
        raise TypeError("execution_policy must be a mapping")
    if dict(policy) != _FROZEN_EXECUTION_POLICY:
        raise ValueError("execution_policy differs from the frozen v13.1 policy")
    sandbox_policy = authorization["sandbox_policy"]
    if not isinstance(sandbox_policy, Mapping) or dict(sandbox_policy) != _expected_sandbox_policy():
        raise ValueError("sandbox_policy differs from mandatory v13.1 bubblewrap")
    broker_policy = authorization["target_broker_trace_policy"]
    if (
        not isinstance(broker_policy, Mapping)
        or dict(broker_policy) != _expected_target_broker_trace_policy()
    ):
        raise ValueError("target_broker_trace_policy differs from mandatory v13.1 audit")
    supersession = authorization["supersession"]
    if not isinstance(supersession, Mapping) or dict(supersession) != _expected_supersession():
        raise ValueError("authorization does not immutably supersede failed v2")
    _verify_superseded_v2_state()
    preprocessing = authorization["preprocessing"]
    if not isinstance(preprocessing, Mapping) or set(preprocessing) != {
        "bundle",
        "selection_artifact_sha256",
        "protocol_receipt_sha256",
    }:
        raise ValueError("preprocessing authorization violates its closed schema")
    _workspace_path(
        preprocessing["bundle"], field="preprocessing.bundle", directory=True
    )
    _sha256(
        preprocessing["selection_artifact_sha256"],
        field="preprocessing.selection_artifact_sha256",
    )
    _sha256(
        preprocessing["protocol_receipt_sha256"],
        field="preprocessing.protocol_receipt_sha256",
    )
    snapshot = authorization["source_target_snapshot"]
    if not isinstance(snapshot, Mapping) or set(snapshot) != {
        "bundle",
        "manifest_sha256",
        "receipt_sha256",
    }:
        raise ValueError("source_target_snapshot violates its closed schema")
    snapshot_path = _workspace_path(
        snapshot["bundle"], field="source_target_snapshot.bundle", directory=True
    )
    for name in ("manifest", "receipt"):
        digest = _sha256(snapshot[f"{name}_sha256"], field=f"snapshot.{name}_sha256")
        if _file_sha256(snapshot_path / f"{name}.json") != digest:
            raise ValueError(f"source target snapshot {name} SHA changed")
    selections = authorization["selections"]
    if not isinstance(selections, list) or len(selections) != len(_SELECTION_ORDER):
        raise ValueError("authorization must contain exactly six selections")
    normalized_rows = []
    expected_row_fields = {
        "selection",
        "k31_bundle",
        "k31_manifest_sha256",
        "k31_checkpoint",
        "k31_checkpoint_sha256",
        "training_manifest_bundle",
        "training_manifest_bundle_sha256",
        "training_manifest_source_sha256",
        "training_token_corpus",
        "training_token_corpus_index_sha256",
        "fit_target_output",
        "fit_token_output",
        "capacity_control_output",
        "independent_control_output",
    }
    for expected_selection, value in zip(_SELECTION_ORDER, selections, strict=True):
        if not isinstance(value, Mapping) or set(value) != expected_row_fields:
            raise ValueError("selection authorization violates its closed schema")
        row = dict(value)
        if row["selection"] != expected_selection:
            raise ValueError("selection execution order changed")
        k31 = _workspace_path(
            row["k31_bundle"], field=f"{expected_selection}.k31_bundle", directory=True
        )
        checkpoint = _workspace_path(
            row["k31_checkpoint"],
            field=f"{expected_selection}.k31_checkpoint",
            directory=False,
        )
        training = _workspace_path(
            row["training_manifest_bundle"],
            field=f"{expected_selection}.training_manifest_bundle",
            directory=True,
        )
        corpus = _workspace_path(
            row["training_token_corpus"],
            field=f"{expected_selection}.training_token_corpus",
            directory=True,
        )
        file_checks = (
            (k31 / "recovery_run.json", "k31_manifest_sha256"),
            (checkpoint, "k31_checkpoint_sha256"),
            (training / "manifest.json", "training_manifest_bundle_sha256"),
            (training / "receipt.json", "training_manifest_source_sha256"),
            (corpus / "index.json", "training_token_corpus_index_sha256"),
        )
        for file, name in file_checks:
            digest = _sha256(row[name], field=f"{expected_selection}.{name}")
            if _file_sha256(file) != digest:
                raise ValueError(f"{expected_selection} input SHA changed: {name}")
        recovery_payload = json.loads((k31 / "recovery_run.json").read_text("utf-8"))
        if not isinstance(recovery_payload, Mapping):
            raise ValueError(f"{expected_selection} k31 manifest is not an object")
        declared_checkpoint = recovery_payload.get("checkpoint_filename")
        declared_checkpoint_sha = recovery_payload.get("checkpoint_sha256")
        if (
            checkpoint.parent != k31
            or checkpoint.name != declared_checkpoint
            or row["k31_checkpoint_sha256"] != declared_checkpoint_sha
        ):
            raise ValueError(f"{expected_selection} k31 checkpoint lineage changed")
        for name in (
            "fit_target_output",
            "fit_token_output",
            "capacity_control_output",
            "independent_control_output",
        ):
            output = (ROOT / str(row[name])).resolve()
            if ROOT not in output.parents or output.name in {"", ".", ".."}:
                raise ValueError(f"{expected_selection}.{name} is unsafe")
            if os.path.lexists(output):
                raise FileExistsError(f"authorized output already exists: {output}")
        normalized_rows.append(row)
    code_files = authorization["code_files"]
    if not isinstance(code_files, list) or len(code_files) < len(
        _REQUIRED_AUTHORIZED_CODE_FILES
    ):
        raise ValueError("authorization omits the required v13.1 code-file roster")
    verified_code = [
        _verify_file_record(value, field=f"code_files[{index}]")
        for index, value in enumerate(code_files)
    ]
    verified_paths = tuple(record["path"] for record in verified_code)
    if len(set(verified_paths)) != len(verified_paths) or not set(
        _REQUIRED_AUTHORIZED_CODE_FILES
    ).issubset(verified_paths):
        raise ValueError("authorization code-file roster is duplicate or incomplete")
    test_receipt = authorization["test_receipt"]
    if not isinstance(test_receipt, Mapping) or dict(test_receipt) != {
        "command": _TEST_COMMAND,
        "expected_passed": _EXPECTED_TEST_PASSED,
        "passed": True,
    }:
        raise ValueError("test receipt differs from the authorized suite")
    output_root = (ROOT / str(authorization["output_root"])).resolve()
    if ROOT not in output_root.parents or output_root.name in {"", ".", ".."}:
        raise ValueError("output_root is unsafe")
    _assert_direct_formal_output_child(output_root, field="output_root")
    if os.path.lexists(output_root):
        raise FileExistsError(f"authorized output root already exists: {output_root}")
    for row in normalized_rows:
        selection_name = str(row["selection"])
        expected_outputs = {
            "fit_target_output": output_root / "fit_targets" / selection_name,
            "fit_token_output": output_root / "fit_token_views" / selection_name,
            "capacity_control_output": (
                output_root / "capacity_controls" / selection_name / "bundle"
            ),
            "independent_control_output": (
                output_root / "naked_controls" / selection_name / "bundle"
            ),
        }
        for field, expected in expected_outputs.items():
            if (ROOT / str(row[field])).resolve() != expected:
                raise ValueError(f"{selection_name}.{field} violates category layout")
    return {
        **authorization,
        "protocol": protocol,
        "amendment": amendment,
        "selections": normalized_rows,
        "code_files": verified_code,
    }


def _run(
    command: list[str], *, log_path: Path | None = None
) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if log_path is not None:
        source = Path(os.path.abspath(log_path))
        if not source.parent.is_dir() or os.path.lexists(source):
            raise FileExistsError(f"subprocess log path is not new: {source}")
        with source.open("xb") as handle:
            handle.write(completed.stdout.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    if completed.returncode:
        raise RuntimeError(f"authorized subprocess failed ({completed.returncode})")
    rows = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    if not rows:
        raise RuntimeError("authorized subprocess emitted no JSON receipt")
    payload = json.loads(rows[-1])
    if not isinstance(payload, dict):
        raise RuntimeError("authorized subprocess receipt is not an object")
    return payload


def _target_command(
    row: Mapping[str, object], authorization: Mapping[str, object], *, preflight: bool
) -> list[str]:
    snapshot = authorization["source_target_snapshot"]
    command = [
        sys.executable,
        "scripts/materialize_labram_ictal_fit_only_targets_v13.py",
        "--selection",
        str(row["selection"]),
        "--k31-v1-2-bundle",
        str(row["k31_bundle"]),
        "--expected-k31-v1-2-manifest-sha256",
        str(row["k31_manifest_sha256"]),
        "--training-manifest-bundle",
        str(row["training_manifest_bundle"]),
        "--expected-training-manifest-bundle-sha256",
        str(row["training_manifest_bundle_sha256"]),
        "--expected-training-manifest-source-sha256",
        str(row["training_manifest_source_sha256"]),
        "--expected-training-token-corpus-index-sha256",
        str(row["training_token_corpus_index_sha256"]),
        "--source-formal-v4-target-snapshot",
        str(snapshot["bundle"]),
        "--expected-source-target-snapshot-manifest-sha256",
        str(snapshot["manifest_sha256"]),
        "--expected-source-target-snapshot-receipt-sha256",
        str(snapshot["receipt_sha256"]),
        "--output-directory",
        (
            f"outputs/.v13_fit_control_preflight_{row['selection']}"
            if preflight
            else str(row["fit_target_output"])
        ),
    ]
    if preflight:
        command.append("--preflight-only")
    return command


def _build_target_broker_trace_invocation(
    command: Sequence[str], *, trace_log_path: Path
) -> list[str]:
    """Wrap only a formal target materialization in mandatory host strace."""

    values = [str(value) for value in command]
    if (
        len(values) < 3
        or Path(values[1]).name
        != "materialize_labram_ictal_fit_only_targets_v13.py"
        or "--preflight-only" in values
        or "--output-directory" not in values
    ):
        raise ValueError("broker strace can wrap only formal target materialization")
    trace = Path(os.path.abspath(trace_log_path))
    if (
        trace.name in {"", ".", ".."}
        or trace.is_symlink()
        or not trace.parent.is_dir()
        or trace.parent.is_symlink()
        or os.path.lexists(trace)
    ):
        raise FileExistsError("target-broker strace path must be new")
    return [
        str(_STRACE_EXECUTABLE),
        "-f",
        "-qq",
        "-s",
        str(_STRACE_STRING_LIMIT_BYTES),
        "-e",
        "trace=openat,openat2,pread64",
        "-o",
        str(trace),
        "--",
        *values,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw, authorization_sha = _strict_authorization(
        args.authorization, expected_sha256=args.expected_authorization_sha256
    )
    authorization = _validate_authorization(raw)
    preflight_rows = []
    for row in authorization["selections"]:
        receipt = _run(_target_command(row, authorization, preflight=True))
        if (
            receipt.get("preflight_passed") is not True
            or receipt.get("training_started") is not False
            or receipt.get("materialization_started") is not False
            or receipt.get("source_data_rows_read") != 0
        ):
            raise RuntimeError("fit-target preflight did not remain training-free")
        preflight_rows.append(
            {
                "selection": row["selection"],
                "fit_patient_count": receipt["fit_patient_count"],
                "fit_event_count": receipt["fit_event_count"],
                "source_data_rows_read": 0,
                "k31_manifest_sha256": row["k31_manifest_sha256"],
                "training_manifest_bundle_sha256": row[
                    "training_manifest_bundle_sha256"
                ],
                "training_manifest_source_sha256": row[
                    "training_manifest_source_sha256"
                ],
                "training_token_corpus_index_sha256": row[
                    "training_token_corpus_index_sha256"
                ],
            }
        )
    summary = {
        "schema_version": _PREFLIGHT_SCHEMA,
        "authorization_sha256": authorization_sha,
        "preflight_passed": True,
        "selection_order": list(_SELECTION_ORDER),
        "selection_rows": preflight_rows,
        "training_started": False,
        "trainer_subprocesses_started": 0,
        "target_broker_subprocesses_started": 0,
        "source_data_rows_read": 0,
        "mandatory_bwrap": True,
        "unsandboxed_trainer_fallback": False,
        "outer_broker_workspace_and_source_data_reachable": True,
        "outer_broker_gate_target_values_read": False,
        "gate_opened": False,
        "i_gate_target_values_materialized": False,
        "overwrite": False,
    }
    if args.preflight_only:
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0

    cuda_device_nodes = _discover_cuda_device_nodes()
    output_root = (ROOT / str(authorization["output_root"])).resolve()
    output_root.mkdir(parents=False, exist_ok=False)
    for category in (
        "fit_targets",
        "fit_token_views",
        "capacity_controls",
        "naked_controls",
        "broker_traces",
        "broker_logs",
        "broker_audit_receipts",
        "mount_receipts",
        "logs",
    ):
        (output_root / category).mkdir(exist_ok=False)
    artifact_rows = []
    forbidden_trainer_roots = [
        ROOT / str(authorization["preprocessing"]["bundle"]),
        ROOT / str(authorization["source_target_snapshot"]["bundle"]),
    ]
    for authorization_row in authorization["selections"]:
        forbidden_trainer_roots.extend(
            (
                ROOT / str(authorization_row["k31_bundle"]),
                ROOT / str(authorization_row["training_manifest_bundle"]),
                ROOT / str(authorization_row["training_token_corpus"]),
            )
        )
    for row in authorization["selections"]:
        selection_name = str(row["selection"])
        broker_trace_path = output_root / "broker_traces" / f"{selection_name}.strace"
        broker_command = _build_target_broker_trace_invocation(
            _target_command(row, authorization, preflight=False),
            trace_log_path=broker_trace_path,
        )
        _create_empty_new_file(broker_trace_path)
        target_receipt = _run(
            broker_command,
            log_path=output_root / "broker_logs" / f"{selection_name}.log",
        )
        fit_target = ROOT / str(row["fit_target_output"])
        target_manifest_sha = _file_sha256(fit_target / "manifest.json")
        target_receipt_sha = _file_sha256(fit_target / "receipt.json")
        if target_receipt.get("manifest_sha256") != target_manifest_sha:
            raise RuntimeError("fit-target publication SHA differs from receipt")
        broker_audit = _audit_target_broker_strace(
            broker_trace_path,
            selection=selection_name,
            source_snapshot_bundle=ROOT
            / str(authorization["source_target_snapshot"]["bundle"]),
            expected_source_manifest_sha256=str(
                authorization["source_target_snapshot"]["manifest_sha256"]
            ),
            expected_source_receipt_sha256=str(
                authorization["source_target_snapshot"]["receipt_sha256"]
            ),
            fit_target_bundle=fit_target,
            expected_fit_manifest_sha256=target_manifest_sha,
            expected_fit_receipt_sha256=target_receipt_sha,
            authorization_sha256=authorization_sha,
        )
        broker_audit_path = (
            output_root / "broker_audit_receipts" / f"{selection_name}.json"
        )
        broker_audit_sha = _write_canonical_new_json(
            broker_audit_path, broker_audit
        )
        preprocessing = authorization["preprocessing"]
        token_command = [
            sys.executable,
            "scripts/materialize_labram_ictal_fit_token_view_v13.py",
            "--selection",
            str(row["selection"]),
            "--source-training-token-corpus",
            str(row["training_token_corpus"]),
            "--expected-source-training-token-corpus-index-sha256",
            str(row["training_token_corpus_index_sha256"]),
            "--preprocessing-selection-bundle",
            str(preprocessing["bundle"]),
            "--expected-preprocessing-selection-artifact-sha256",
            str(preprocessing["selection_artifact_sha256"]),
            "--expected-preprocessing-protocol-receipt-sha256",
            str(preprocessing["protocol_receipt_sha256"]),
            "--fit-only-target-bundle",
            str(row["fit_target_output"]),
            "--expected-fit-only-target-manifest-sha256",
            target_manifest_sha,
            "--expected-fit-only-target-receipt-sha256",
            target_receipt_sha,
            "--output-directory",
            str(row["fit_token_output"]),
        ]
        token_receipt = _run(token_command)
        fit_token = ROOT / str(row["fit_token_output"])
        token_manifest_sha = _file_sha256(fit_token / "manifest.json")
        token_receipt_sha = _file_sha256(fit_token / "receipt.json")
        if token_receipt.get("manifest_sha256") != token_manifest_sha:
            raise RuntimeError("fit-token publication SHA differs from receipt")
        mount_receipt_root = output_root / "mount_receipts" / selection_name
        log_root = output_root / "logs" / selection_name
        mount_receipt_root.mkdir(exist_ok=False)
        log_root.mkdir(exist_ok=False)
        control_results: dict[str, tuple[str, str, str, int]] = {}
        for control_kind, output_field in (
            ("capacity", "capacity_control_output"),
            ("naked", "independent_control_output"),
        ):
            control_bundle = (ROOT / str(row[output_field])).resolve()
            control_parent = control_bundle.parent
            control_parent.mkdir(exist_ok=False)
            trace_path = log_root / f"{control_kind}.open_files.strace"
            sandbox_command, mount_manifest = _build_trainer_sandbox_invocation(
                selection=selection_name,
                control=control_kind,
                fit_target_bundle=fit_target,
                fit_token_bundle=fit_token,
                output_parent=control_parent,
                target_manifest_sha256=target_manifest_sha,
                target_receipt_sha256=target_receipt_sha,
                token_manifest_sha256=token_manifest_sha,
                token_receipt_sha256=token_receipt_sha,
                authorization_sha256=authorization_sha,
                trace_log_path=trace_path,
                cuda_device_nodes=cuda_device_nodes,
            )
            _create_empty_new_file(trace_path)
            mount_path = mount_receipt_root / f"{control_kind}.json"
            mount_sha = _write_canonical_new_json(mount_path, mount_manifest)
            control_receipt = _run(
                sandbox_command, log_path=log_root / f"{control_kind}.log"
            )
            control_manifest_sha = _file_sha256(
                control_bundle / "control_run.json"
            )
            if control_receipt.get("manifest_sha256") != control_manifest_sha:
                raise RuntimeError(f"{control_kind}-control SHA differs from receipt")
            trace_receipt = _audit_strace_log(
                trace_path, forbidden_roots=forbidden_trainer_roots
            )
            control_results[control_kind] = (
                control_manifest_sha,
                mount_sha,
                str(trace_receipt["strace_log_sha256"]),
                int(trace_receipt["strace_log_size_bytes"]),
            )
        (
            control_manifest_sha,
            capacity_mount_sha,
            capacity_trace_sha,
            capacity_trace_size,
        ) = control_results["capacity"]
        (
            independent_manifest_sha,
            naked_mount_sha,
            naked_trace_sha,
            naked_trace_size,
        ) = control_results["naked"]
        artifact_rows.append(
            {
                "selection": row["selection"],
                "fit_target_manifest_sha256": target_manifest_sha,
                "fit_target_receipt_sha256": target_receipt_sha,
                "target_broker_strace_log_sha256": broker_audit[
                    "strace_log_sha256"
                ],
                "target_broker_strace_log_size_bytes": broker_audit[
                    "strace_log_size_bytes"
                ],
                "target_broker_audit_receipt_sha256": broker_audit_sha,
                "target_broker_exact_selected_pread64_match": True,
                "target_broker_i_gate_range_overlap_count": 0,
                "fit_token_manifest_sha256": token_manifest_sha,
                "fit_token_receipt_sha256": token_receipt_sha,
                "capacity_control_manifest_sha256": control_manifest_sha,
                "independent_control_manifest_sha256": independent_manifest_sha,
                "capacity_mount_manifest_sha256": capacity_mount_sha,
                "naked_mount_manifest_sha256": naked_mount_sha,
                "capacity_strace_log_sha256": capacity_trace_sha,
                "capacity_strace_log_size_bytes": capacity_trace_size,
                "naked_strace_log_sha256": naked_trace_sha,
                "naked_strace_log_size_bytes": naked_trace_size,
                "forbidden_root_open_detected": False,
                "training_started": True,
                "evaluation_performed": False,
                "gate_opened": False,
            }
        )
    execution = {
        "schema_version": _EXECUTION_SCHEMA,
        "authorization_sha256": authorization_sha,
        "selection_order": list(_SELECTION_ORDER),
        "artifact_rows": artifact_rows,
        "all_six_completed": True,
        "checkpoint_count": 12,
        "trainer_subprocess_count": 12,
        "target_broker_subprocess_count": 6,
        "target_broker_strace_trace_count": 6,
        "target_broker_audit_receipt_count": 6,
        "all_target_brokers_exact_pread64_audited": True,
        "target_broker_i_gate_range_overlap_count": 0,
        "target_broker_i_gate_target_or_mask_bytes_read": False,
        "all_trainers_bubblewrap_confined": True,
        "unsandboxed_trainer_fallback": False,
        "outer_broker_workspace_and_source_data_reachable": True,
        "outer_broker_gate_target_values_read": False,
        "trainer_forbidden_data_mounted": False,
        "trainer_source_broker_imported": False,
        "evaluation_performed": False,
        "gate_opened": False,
        "i_gate_target_values_materialized": False,
        "overwrite": False,
    }
    receipt_path = output_root / "execution_receipt.json"
    with receipt_path.open("xb") as handle:
        handle.write(_canonical_json_bytes(execution))
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(output_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(
        json.dumps(
            {
                **summary,
                "training_started": True,
                "all_six_completed": True,
                "checkpoint_count": 12,
                "execution_receipt": str(receipt_path.relative_to(ROOT)),
                "execution_receipt_sha256": _file_sha256(receipt_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
