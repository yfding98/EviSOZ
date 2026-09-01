#!/usr/bin/env python3
"""Materialize a resumable A0 native-12 BA-IEG source-train dataset.

This runner is intentionally limited to the A0 upper-bound experiment.  It
uses frozen public TUSZ seizure intervals for oracle navigation, never claims
a detector-frozen candidate set, and labels the default 12/48-second support
as an *initial bootstrap watchdog only*.  The final iterative rule-adaptive
window is not materialized by this script.

Each selected record is loaded canonically once, processed into per-event
native-12 P0/projection-v2 artifacts, and committed atomically only after all
expected events have either succeeded or acquired a typed failure terminal.
The very large raw-dependency provenance object is stored as deterministic
disk-v3 gzip carrying the exact disk-v2 canonical JSON bytes and semantic
artifact identity.  Resume re-hashes every referenced byte before reusing a
committed record; legacy disk-v2 ``raw_dependency.json`` records remain valid.
No EDF annotations, spreadsheets, clinical text, seizure type, channel target,
report model, Qwen service, or private data input is accepted by the CLI.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.ba_ieg_a0_native12_dataset_v1 import (  # noqa: E402
    build_ba_ieg_a0_artifact_reference_v1,
    build_ba_ieg_a0_boundary_target_v1,
    build_ba_ieg_a0_event_failure_terminal_v1,
    build_ba_ieg_a0_event_success_terminal_v1,
    build_ba_ieg_a0_native12_dataset_manifest_v2,
    build_ba_ieg_a0_record_terminal_v1,
    validate_ba_ieg_a0_native12_dataset_manifest_v2,
    verify_ba_ieg_a0_record_terminal_artifacts_v1,
    write_ba_ieg_a0_native12_dataset_manifest_v2,
)
from src.clinical_eeg_long_recording.ba_ieg_a0_navigation_window_v1 import (  # noqa: E402
    BAIEGA0NavigationSupportPolicyV1,
    build_ba_ieg_a0_canonical_identity_binding_v1,
    build_ba_ieg_a0_navigation_window_v1,
    validate_ba_ieg_a0_canonical_identity_binding_v1,
)
from src.clinical_eeg_long_recording.ba_ieg_a0_oracle_navigation_candidate_roster_v1 import (  # noqa: E402
    load_ba_ieg_a0_oracle_navigation_candidate_roster_v1,
)
from src.clinical_eeg_long_recording.ba_ieg_deterministic_target_projection_disk_v1 import (  # noqa: E402
    write_ba_ieg_deterministic_target_projection_disk_v1,
)
from src.clinical_eeg_long_recording.ba_ieg_event_model_input_projection_v2 import (  # noqa: E402
    project_ba_ieg_event_model_input_v2,
)
from src.clinical_eeg_long_recording.ba_ieg_p0_raw_dependency_projection_disk_v3 import (  # noqa: E402
    write_ba_ieg_p0_raw_dependency_projection_disk_sidecar_v3,
)
from src.clinical_eeg_long_recording.ba_ieg_permission_split_segmental_disk_training_v1 import (  # noqa: E402
    ba_ieg_segmental_event_input_metadata_v1,
    ba_ieg_segmental_event_tensor_arrays_v1,
)
from src.clinical_eeg_long_recording.ba_ieg_training_contract import (  # noqa: E402
    BA_IEG_P0_VIEW_PROFILE_NATIVE_12,
    materialize_ba_ieg_p0_event_tokens,
)
from src.clinical_eeg_long_recording.canonical_edf_materialization import (  # noqa: E402
    CanonicalEDFConfig,
    load_canonical_edf_views,
)
from src.clinical_eeg_long_recording.deepsoz_tusz_identity_binding_v1 import (  # noqa: E402
    load_deepsoz_tusz_source_train_identity_binding_v1,
)


_DEFAULT_ROSTER = (
    ROOT
    / "outputs/ba_ieg_a0_oracle_navigation_candidate_roster_v1_20260824r1/candidate_roster.json"
)
_DEFAULT_IDENTITY = (
    ROOT
    / "outputs/deepsoz_tusz_source_train_identity_binding_v1_20260823/identity_binding.json"
)
_DEFAULT_OUTPUT = ROOT / "outputs/ba_ieg_a0_native12_dataset_v1"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    if size < 1:
        raise ValueError(f"artifact is empty: {path.name}")
    return size, digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"append-only artifact exists: {path}")
    descriptor = -1
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.tmp-", dir=path.parent
        )
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None and os.path.lexists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_no_clobber(path, _canonical_json_bytes(payload))


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Stable, non-pickle NPZ for target-free event input tensors."""

    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(arrays):
            array = np.ascontiguousarray(arrays[name])
            npy = io.BytesIO()
            np.lib.format.write_array(npy, array, allow_pickle=False)
            info = zipfile.ZipInfo(
                f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, npy.getvalue())
    return stream.getvalue()


def _artifact_reference(
    *,
    attempt_root: Path,
    relative_path: str,
    kind: str,
) -> dict[str, Any]:
    path = attempt_root / relative_path
    size, digest = _file_sha256(path)
    return build_ba_ieg_a0_artifact_reference_v1(
        kind=kind,
        relative_path=relative_path,
        file_size_bytes=size,
        file_sha256=digest,
    )


def _trusted_view_registry(bundle) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for role in (
        "onset_causal",
        "context_offline",
        "findings_native_morphology",
    ):
        for reference in ("referential", "tcp_bipolar", "car", "laplacian"):
            receipt = bundle.task_reference_views[role][reference].receipt
            view_id = str(receipt["view_id"])
            if view_id in registry:
                raise ValueError("canonical trusted-view registry repeats a view ID")
            registry[view_id] = receipt
    return registry


def _a0_input_metadata(
    model_event,
    *,
    navigation_window_receipt_sha256: str,
    p0_materialization_receipt_sha256: str,
) -> dict[str, Any]:
    # Reuse the exact event metadata projection, but remove the legacy field
    # name before persistence so A0 cannot masquerade as adaptive detection.
    legacy = ba_ieg_segmental_event_input_metadata_v1(
        model_event,
        adaptive_acquisition_receipt_sha256=(
            navigation_window_receipt_sha256
        ),
    )
    event_metadata = deepcopy(legacy["event_metadata"])
    registered = event_metadata.pop("adaptive_window_receipt_sha256")
    if registered != navigation_window_receipt_sha256:
        raise ValueError("A0 event input lost its navigation-window binding")
    event_metadata["navigation_window_receipt_sha256"] = registered
    return {
        "schema_version": "ba_ieg_a0_target_free_event_input_metadata_v1",
        "event_model_input_receipt_sha256": model_event.input_receipt_sha256,
        "source_p0_materialization_receipt_sha256": (
            p0_materialization_receipt_sha256
        ),
        "navigation_arm": "A0_conditional_on_oracle_navigation",
        "evaluation_semantics": "conditional_on_seizure_interval_upper_bound",
        "support_role": "initial_bootstrap_watchdog_only",
        "event_metadata": event_metadata,
        "scope_receipt": {
            "model_input_target_free": True,
            "deterministic_target_embedded": False,
            "boundary_target_embedded": False,
            "detector_receipt_claimed": False,
            "final_rule_adaptive_support_materialized": False,
            "edf_annotation_used": False,
            "spreadsheet_used": False,
            "clinical_text_used": False,
        },
    }


def _projection_receipt_payload(projection) -> dict[str, Any]:
    body = {
        "schema_version": "ba_ieg_projection_v2_detached_receipt_v1",
        "projection_v2_receipt_sha256": projection.receipt_sha256,
        "source_p0_materialization_receipt_sha256": (
            projection.source_p0_materialization_receipt_sha256
        ),
        "event_model_input_receipt_sha256": (
            projection.model_input_event.input_receipt_sha256
        ),
        "deterministic_target_sidecar_receipt_sha256": (
            projection.deterministic_target_sidecar.receipt_sha256
        ),
        "deterministic_target_receipt_sha256": (
            projection.deterministic_target_sidecar.target_receipt_sha256
        ),
        "raw_dependency_sidecar_sha256": (
            projection.raw_sample_dependency_sidecar_sha256
        ),
        "scope_receipt": projection.scope_receipt,
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def _events_by_record(roster: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result = {row["model_recording_id"]: [] for row in roster["records"]}
    for event in roster["events"]:
        result[event["model_recording_id"]].append(event)
    return result


def _p0_failure_code(receipt: Mapping[str, Any]) -> str:
    mapping = {
        "invalid_canonical_bundle": "p0_invalid_canonical_bundle",
        "invalid_a0_navigation_window": "p0_invalid_a0_navigation_window",
        "a0_navigation_argument_mismatch": "p0_invalid_a0_navigation_window",
        "a0_canonical_identity_mismatch": "p0_a0_identity_binding_mismatch",
        "recording_clock_mismatch": "p0_recording_clock_mismatch",
        "event_interval_unavailable": "p0_event_interval_unavailable",
        "view_clock_or_reference_mismatch": (
            "p0_view_clock_or_reference_mismatch"
        ),
        "no_evidence_eligible_tokens": "p0_no_evidence_eligible_tokens",
        "tokenization_failed": "p0_tokenization_failed",
    }
    return mapping.get(str(receipt.get("failure_code")), "unexpected_event_failure")


def _source_edf(tusz_root: Path, source_recording_id: str) -> Path:
    root = tusz_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("--tusz-root must be an existing directory")
    logical = root / source_recording_id
    parent = logical.parent.resolve(strict=True)
    if parent != root and root not in parent.parents:
        raise ValueError("TUSZ source recording path escapes --tusz-root")
    path = parent / logical.name
    if not path.is_file() or path.is_symlink():
        raise ValueError("TUSZ source EDF must be a non-symlink regular file")
    return path


def _load_committed_record(
    *,
    output_root: Path,
    record_id: str,
    roster: Mapping[str, Any],
) -> dict[str, Any]:
    record_dir = output_root / "records" / record_id
    if record_dir.is_symlink() or not record_dir.is_dir():
        raise ValueError("committed record path is not a trusted directory")
    commit_path = record_dir / "record_terminal.json"
    raw = commit_path.read_bytes()
    parsed = json.loads(raw.decode("utf-8", errors="strict"))
    if _canonical_json_bytes(parsed) != raw:
        raise ValueError("record terminal JSON is not canonical")
    terminal = verify_ba_ieg_a0_record_terminal_artifacts_v1(
        output_root, parsed, candidate_roster=roster
    )
    identity_receipt = terminal[
        "canonical_identity_binding_receipt_sha256"
    ]
    identity_path = record_dir / "canonical_identity_binding.json"
    if identity_receipt is None:
        if os.path.lexists(identity_path):
            raise ValueError("failed record unexpectedly acquired identity artifact")
    else:
        identity_raw = identity_path.read_bytes()
        identity_payload = json.loads(identity_raw.decode("utf-8", errors="strict"))
        if identity_raw != _canonical_json_bytes(identity_payload):
            raise ValueError("record canonical identity artifact is not canonical JSON")
        validated_identity = validate_ba_ieg_a0_canonical_identity_binding_v1(
            identity_payload, candidate_roster=roster
        )
        if validated_identity["receipt_sha256"] != identity_receipt:
            raise ValueError("record canonical identity artifact was rebound")
    return terminal


def _materialize_success_event(
    *,
    attempt_root: Path,
    record_row: Mapping[str, Any],
    event_row: Mapping[str, Any],
    roster: Mapping[str, Any],
    canonical_binding: Mapping[str, Any],
    canonical_bundle,
    trusted_views: Mapping[str, Mapping[str, Any]],
    annotation_resolution_seconds: float,
) -> dict[str, Any]:
    record_id = str(record_row["model_recording_id"])
    event_id = str(event_row["model_event_id"])
    event_relative = f"records/{record_id}/events/{event_id}"
    event_dir = attempt_root / event_relative
    event_dir.mkdir(parents=True, exist_ok=False)

    window = build_ba_ieg_a0_navigation_window_v1(
        candidate_roster=roster,
        canonical_identity_binding=canonical_binding,
        model_event_id=event_id,
    )
    result = materialize_ba_ieg_p0_event_tokens(
        canonical_bundle,
        None,
        None,
        event_id=event_id,
        recording_id=canonical_binding["canonical_signal"]["recording_id"],
        patient_uid=record_row["patient_uid"],
        model_split="source_train",
        view_profile=BA_IEG_P0_VIEW_PROFILE_NATIVE_12,
        a0_navigation_window_receipt=window,
    )
    if result.receipt["status"] != "materialized":
        return build_ba_ieg_a0_event_failure_terminal_v1(
            event_row=event_row,
            failure_code=_p0_failure_code(result.receipt),
            failure_stage=str(result.receipt["failure_stage"]),
            canonical_identity_binding_receipt_sha256=(
                canonical_binding["receipt_sha256"]
            ),
        )
    projection = project_ba_ieg_event_model_input_v2(
        result,
        canonical_signal_receipt=(
            canonical_bundle.canonical_record.canonical_receipt
        ),
        trusted_view_receipts=trusted_views,
    )
    model_event = projection.model_input_event

    paths = {
        "canonical_identity_binding": f"{event_relative}/canonical_identity_binding.json",
        "navigation_window": f"{event_relative}/navigation_window.json",
        "p0_receipt": f"{event_relative}/p0_receipt.json",
        "input_metadata": f"{event_relative}/input_metadata.json",
        "input_tensors": f"{event_relative}/input_tensors.npz",
        "projection_v2_receipt": f"{event_relative}/projection_v2_receipt.json",
        "raw_dependency": f"{event_relative}/raw_dependency.json.gz",
        "raw_dependency_reference": f"{event_relative}/raw_dependency_reference.json",
        "deterministic_target_metadata": f"{event_relative}/deterministic_target.json",
        "deterministic_target_tensors": f"{event_relative}/deterministic_target.npz",
        "deterministic_target_reference": f"{event_relative}/deterministic_target_reference.json",
        "boundary_target": f"{event_relative}/boundary_target.json",
    }
    _write_json(
        attempt_root / paths["canonical_identity_binding"], canonical_binding
    )
    _write_json(attempt_root / paths["navigation_window"], window)
    _write_json(attempt_root / paths["p0_receipt"], result.receipt)
    input_metadata = _a0_input_metadata(
        model_event,
        navigation_window_receipt_sha256=window["receipt_sha256"],
        p0_materialization_receipt_sha256=result.receipt["receipt_sha256"],
    )
    _write_json(attempt_root / paths["input_metadata"], input_metadata)
    input_arrays = ba_ieg_segmental_event_tensor_arrays_v1(model_event)
    _write_no_clobber(
        attempt_root / paths["input_tensors"],
        _deterministic_npz_bytes(input_arrays),
    )
    _write_json(
        attempt_root / paths["projection_v2_receipt"],
        _projection_receipt_payload(projection),
    )
    raw_reference = (
        write_ba_ieg_p0_raw_dependency_projection_disk_sidecar_v3(
            attempt_root,
            paths["raw_dependency"],
            projection,
            canonical_signal_receipt=(
                canonical_bundle.canonical_record.canonical_receipt
            ),
            trusted_view_receipts=trusted_views,
        )
    )
    _write_json(
        attempt_root / paths["raw_dependency_reference"], raw_reference
    )
    deterministic_reference = (
        write_ba_ieg_deterministic_target_projection_disk_v1(
            attempt_root,
            json_relative_path=paths["deterministic_target_metadata"],
            npz_relative_path=paths["deterministic_target_tensors"],
            projection=projection,
        )
    )
    _write_json(
        attempt_root / paths["deterministic_target_reference"],
        deterministic_reference,
    )
    boundary = build_ba_ieg_a0_boundary_target_v1(
        event_row=event_row,
        navigation_window=window,
        source_p0_materialization_receipt_sha256=result.receipt[
            "receipt_sha256"
        ],
        source_event_model_input_receipt_sha256=(
            model_event.input_receipt_sha256
        ),
        annotation_resolution_seconds=annotation_resolution_seconds,
    )
    _write_json(attempt_root / paths["boundary_target"], boundary)
    artifacts = {
        kind: _artifact_reference(
            attempt_root=attempt_root,
            relative_path=relative_path,
            kind=kind,
        )
        for kind, relative_path in paths.items()
    }
    return build_ba_ieg_a0_event_success_terminal_v1(
        event_row=event_row,
        canonical_identity_binding_receipt_sha256=canonical_binding[
            "receipt_sha256"
        ],
        a0_navigation_window_receipt_sha256=window["receipt_sha256"],
        p0_materialization_receipt_sha256=result.receipt["receipt_sha256"],
        event_model_input_receipt_sha256=model_event.input_receipt_sha256,
        projection_v2_receipt_sha256=projection.receipt_sha256,
        raw_dependency_sidecar_sha256=(
            projection.raw_sample_dependency_sidecar_sha256
        ),
        deterministic_target_sidecar_receipt_sha256=(
            projection.deterministic_target_sidecar.receipt_sha256
        ),
        deterministic_target_receipt_sha256=(
            projection.deterministic_target_sidecar.target_receipt_sha256
        ),
        boundary_target_receipt_sha256=boundary["receipt_sha256"],
        artifacts=artifacts,
    )


def _materialize_record(
    *,
    output_root: Path,
    roster: Mapping[str, Any],
    identity_binding: Mapping[str, Any],
    tusz_root: Path,
    record_row: Mapping[str, Any],
    event_rows: Sequence[Mapping[str, Any]],
    annotation_resolution_seconds: float,
) -> dict[str, Any]:
    record_id = str(record_row["model_recording_id"])
    staging_root = output_root / "staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    attempt_root = Path(
        tempfile.mkdtemp(prefix=f"attempt-{record_id}-", dir=staging_root)
    )
    record_dir = attempt_root / "records" / record_id
    record_dir.mkdir(parents=True)
    canonical_binding: dict[str, Any] | None = None
    try:
        source_edf = _source_edf(
            tusz_root, str(record_row["source_recording_id"])
        )
        bundle = load_canonical_edf_views(source_edf, config=CanonicalEDFConfig())
    except Exception:
        failures = [
            build_ba_ieg_a0_event_failure_terminal_v1(
                event_row=event,
                failure_code="record_canonical_edf_load_failed",
                failure_stage="canonical_edf_load",
                canonical_identity_binding_receipt_sha256=None,
            )
            for event in event_rows
        ]
        terminal = build_ba_ieg_a0_record_terminal_v1(
            record_row=record_row,
            event_rows=event_rows,
            event_terminals=failures,
            canonical_identity_binding_receipt_sha256=None,
            record_failure_code="canonical_edf_load_failed",
            record_failure_stage="canonical_edf_load",
        )
    else:
        try:
            canonical_binding = build_ba_ieg_a0_canonical_identity_binding_v1(
                candidate_roster=roster,
                source_identity_binding=identity_binding,
                model_recording_id=record_id,
                source_edf_path=source_edf,
                canonical_bundle=bundle,
            )
        except Exception:
            failures = [
                build_ba_ieg_a0_event_failure_terminal_v1(
                    event_row=event,
                    failure_code="record_canonical_identity_binding_failed",
                    failure_stage="canonical_identity_binding",
                    canonical_identity_binding_receipt_sha256=None,
                )
                for event in event_rows
            ]
            terminal = build_ba_ieg_a0_record_terminal_v1(
                record_row=record_row,
                event_rows=event_rows,
                event_terminals=failures,
                canonical_identity_binding_receipt_sha256=None,
                record_failure_code="canonical_identity_binding_failed",
                record_failure_stage="canonical_identity_binding",
            )
        else:
            _write_json(record_dir / "canonical_identity_binding.json", canonical_binding)
            trusted_views = _trusted_view_registry(bundle)
            event_terminals: list[dict[str, Any]] = []
            for event in event_rows:
                try:
                    event_terminal = _materialize_success_event(
                        attempt_root=attempt_root,
                        record_row=record_row,
                        event_row=event,
                        roster=roster,
                        canonical_binding=canonical_binding,
                        canonical_bundle=bundle,
                        trusted_views=trusted_views,
                        annotation_resolution_seconds=(
                            annotation_resolution_seconds
                        ),
                    )
                except Exception:
                    event_terminal = build_ba_ieg_a0_event_failure_terminal_v1(
                        event_row=event,
                        failure_code="unexpected_event_failure",
                        failure_stage="event_materialization",
                        canonical_identity_binding_receipt_sha256=(
                            canonical_binding["receipt_sha256"]
                        ),
                    )
                event_terminals.append(event_terminal)
            terminal = build_ba_ieg_a0_record_terminal_v1(
                record_row=record_row,
                event_rows=event_rows,
                event_terminals=event_terminals,
                canonical_identity_binding_receipt_sha256=canonical_binding[
                    "receipt_sha256"
                ],
            )
    _write_json(record_dir / "record_terminal.json", terminal)
    final_dir = output_root / "records" / record_id
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(final_dir):
        raise FileExistsError("record destination appeared during atomic commit")
    os.rename(record_dir, final_dir)
    _fsync_directory(final_dir.parent)
    return _load_committed_record(
        output_root=output_root, record_id=record_id, roster=roster
    )


def _select_records(
    roster: Mapping[str, Any],
    *,
    requested_ids: Sequence[str],
    maximum_records: int | None,
) -> list[dict[str, Any]]:
    records = list(roster["records"])
    if requested_ids:
        if maximum_records is not None:
            raise ValueError("--record-id and --maximum-records are mutually exclusive")
        requested = list(requested_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("--record-id cannot repeat a record")
        lookup = {row["model_recording_id"]: row for row in records}
        missing = [item for item in requested if item not in lookup]
        if missing:
            raise ValueError(f"unknown A0 --record-id: {missing[0]}")
        selected = [row for row in records if row["model_recording_id"] in requested]
        if [row["model_recording_id"] for row in selected] != [
            item for item in lookup if item in set(requested)
        ]:
            raise RuntimeError("record selection order drifted")
        return selected
    if maximum_records is None:
        return records
    if maximum_records < 1 or maximum_records > len(records):
        raise ValueError("--maximum-records must lie within the A0 record roster")
    return records[:maximum_records]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0-roster", type=Path, default=_DEFAULT_ROSTER)
    parser.add_argument("--identity-binding", type=Path, default=_DEFAULT_IDENTITY)
    parser.add_argument("--tusz-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--maximum-records", type=int)
    parser.add_argument("--record-id", action="append", default=[])
    parser.add_argument(
        "--annotation-resolution-seconds", type=float, default=0.001
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    roster = load_ba_ieg_a0_oracle_navigation_candidate_roster_v1(
        args.a0_roster
    )
    identity = load_deepsoz_tusz_source_train_identity_binding_v1(
        args.identity_binding
    )
    if identity["receipt_sha256"] != roster["identity_binding_sha256"]:
        raise ValueError("A0 roster and identity binding disagree")
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not 0 < float(args.annotation_resolution_seconds) < 60:
        raise ValueError("--annotation-resolution-seconds must lie in (0,60)")
    records = _select_records(
        roster,
        requested_ids=args.record_id,
        maximum_records=args.maximum_records,
    )
    events_by_record = _events_by_record(roster)
    terminals: list[dict[str, Any]] = []
    for record in records:
        record_id = record["model_recording_id"]
        final_dir = output_root / "records" / record_id
        if os.path.lexists(final_dir):
            if not args.resume:
                raise FileExistsError(
                    f"committed record exists; pass --resume: {record_id}"
                )
            terminal = _load_committed_record(
                output_root=output_root,
                record_id=record_id,
                roster=roster,
            )
        else:
            terminal = _materialize_record(
                output_root=output_root,
                roster=roster,
                identity_binding=identity,
                tusz_root=args.tusz_root,
                record_row=record,
                event_rows=events_by_record[record_id],
                annotation_resolution_seconds=(
                    args.annotation_resolution_seconds
                ),
            )
        terminals.append(terminal)
    manifest = build_ba_ieg_a0_native12_dataset_manifest_v2(
        candidate_roster=roster, record_terminals=terminals
    )
    manifest_path = output_root / "dataset_manifest.json"
    if os.path.lexists(manifest_path):
        if not args.resume:
            raise FileExistsError("dataset manifest exists; pass --resume")
        raw = manifest_path.read_bytes()
        existing = json.loads(raw.decode("utf-8", errors="strict"))
        validate_ba_ieg_a0_native12_dataset_manifest_v2(
            existing, candidate_roster=roster
        )
        if raw != _canonical_json_bytes(existing) or existing != manifest:
            raise ValueError("existing dataset manifest disagrees with resumed bytes")
    else:
        write_ba_ieg_a0_native12_dataset_manifest_v2(
            manifest, manifest_path, candidate_roster=roster
        )
    print(
        json.dumps(
            {
                "output": str(manifest_path),
                "receipt_sha256": manifest["receipt_sha256"],
                "dataset_state": manifest["dataset_state"],
                "counts": manifest["counts"],
                "evaluation_semantics": manifest["evaluation_semantics"],
                "support_role": "initial_bootstrap_watchdog_only",
                "final_rule_adaptive_support_materialized": False,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
