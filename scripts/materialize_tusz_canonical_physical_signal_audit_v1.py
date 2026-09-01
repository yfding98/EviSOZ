#!/usr/bin/env python3
"""Resume canonical-physical TUSZ audit shards and aggregate them.

Each shard stores one immutable outcome JSON per selected analysis identity.
An interrupted invocation can therefore resume without replaying successful
records.  A manifest is published only after every selected identity has a
terminal success/failure outcome.  Aggregation requires every partition index
exactly once and authorizes the physical analysis projection only when there
are no failures.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.tusz_canonical_physical_signal_audit_v1 import (  # noqa: E402
    build_tusz_canonical_physical_analysis_projection_v1,
    build_tusz_canonical_physical_duplicate_audit_v1,
    build_tusz_canonical_physical_failure_outcome_v1,
    build_tusz_canonical_physical_shard_v1,
    build_tusz_canonical_physical_shards_v1,
    materialize_tusz_canonical_physical_outcome_v1,
    select_tusz_canonical_physical_shard_rows_v1,
    validate_tusz_canonical_physical_outcome_v1,
    validate_tusz_canonical_physical_shard_v1,
)
from src.clinical_eeg_long_recording.tusz_complete_detector_roster_v2 import (  # noqa: E402
    validate_tusz_analysis_identity_projection_v2,
    validate_tusz_complete_detector_roster_v2,
)


DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_ROSTER = Path("outputs/tusz_complete_detector_roster_v2_20260823/roster.json")
DEFAULT_PROJECTION = Path(
    "outputs/tusz_complete_detector_roster_v2_20260823/analysis_projection.json"
)


def _load_json(path: Path) -> Any:
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"JSON artifact is empty: {path}")
    return json.loads(raw.decode("utf-8"))


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise ValueError(f"append-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Hard-link publication is an atomic no-replace operation on the
            # same filesystem.  Unlike a check followed by os.replace(), two
            # concurrent writers can never silently overwrite one another.
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise ValueError(
                f"append-only output appeared during write: {path}"
            ) from exc
        Path(temporary_name).unlink()
        temporary_name = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _inputs(arguments: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    roster = validate_tusz_complete_detector_roster_v2(_load_json(arguments.roster))
    projection = validate_tusz_analysis_identity_projection_v2(
        _load_json(arguments.projection),
        source_roster=roster,
    )
    return roster, projection


def _shard_directory(output_root: Path, count: int, index: int) -> Path:
    return output_root / f"shard-{index:05d}-of-{count:05d}"


def _resolve_append_only_generation_root(path: Path) -> Path:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise ValueError(
                "append-only audit generation root or ancestor is a symbolic link"
            )
    return absolute.resolve(strict=False)


def _outcome_path(directory: Path, ordinal: int, identity: str) -> Path:
    return directory / "outcomes" / f"{ordinal:06d}-{identity}.json"


def _run_shard(arguments: argparse.Namespace) -> int:
    roster, projection = _inputs(arguments)
    arguments.output_root = _resolve_append_only_generation_root(arguments.output_root)
    selected = select_tusz_canonical_physical_shard_rows_v1(
        projection,
        shard_count=arguments.shard_count,
        shard_index=arguments.shard_index,
        source_roster=roster,
    )
    directory = _shard_directory(
        arguments.output_root,
        arguments.shard_count,
        arguments.shard_index,
    )
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = validate_tusz_canonical_physical_shard_v1(
            _load_json(manifest_path),
            source_roster=roster,
            source_projection=projection,
        )
        print(
            json.dumps(
                {
                    "status": "already_complete",
                    "manifest": str(manifest_path),
                    "shard_id": manifest["shard_id"],
                    **manifest["outcome_inventory"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    outcomes: list[dict[str, Any]] = []
    new_count = 0
    limited = False
    for ordinal, row in enumerate(selected):
        path = _outcome_path(directory, ordinal, row["analysis_identity_id"])
        if path.exists():
            outcome = validate_tusz_canonical_physical_outcome_v1(
                _load_json(path),
                projection_row=row,
                source_projection_receipt_sha256=projection["receipt_sha256"],
            )
        elif (
            arguments.max_records_this_run is not None
            and new_count >= arguments.max_records_this_run
        ):
            limited = True
            break
        else:
            try:
                outcome = materialize_tusz_canonical_physical_outcome_v1(
                    projection_row=row,
                    source_projection_receipt_sha256=projection["receipt_sha256"],
                    tusz_root=arguments.tusz_root,
                )
            except Exception as exc:  # terminal denominator, no free-form text
                outcome = build_tusz_canonical_physical_failure_outcome_v1(
                    projection_row=row,
                    source_projection_receipt_sha256=projection["receipt_sha256"],
                    failure_stage="container_hash_canonical_materialization_or_binding",
                    exception_type=type(exc).__name__,
                )
            _write_new_json(path, outcome)
            new_count += 1
        outcomes.append(outcome)

    if len(outcomes) == len(selected):
        manifest = build_tusz_canonical_physical_shard_v1(
            source_roster=roster,
            source_projection=projection,
            shard_count=arguments.shard_count,
            shard_index=arguments.shard_index,
            outcomes=outcomes,
        )
        _write_new_json(manifest_path, manifest)
        status = "complete"
        inventory: Mapping[str, object] = manifest["outcome_inventory"]
    else:
        status = "partial_limit_reached" if limited else "partial"
        inventory = {
            "selected_identity_count": len(selected),
            "terminal_outcome_count": len(outcomes),
            "remaining_identity_count": len(selected) - len(outcomes),
        }
    print(
        json.dumps(
            {
                "status": status,
                "shard_index": arguments.shard_index,
                "shard_count": arguments.shard_count,
                "new_outcome_count": new_count,
                "manifest": str(manifest_path) if manifest_path.exists() else None,
                **inventory,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _worker_materialize(request: tuple[dict[str, Any], str, str]) -> dict[str, Any]:
    """Process-pool worker; failures remain typed terminal outcomes."""

    row, projection_receipt_sha256, tusz_root = request
    try:
        return materialize_tusz_canonical_physical_outcome_v1(
            projection_row=row,
            source_projection_receipt_sha256=projection_receipt_sha256,
            tusz_root=tusz_root,
        )
    except Exception as exc:
        return build_tusz_canonical_physical_failure_outcome_v1(
            projection_row=row,
            source_projection_receipt_sha256=projection_receipt_sha256,
            failure_stage="container_hash_canonical_materialization_or_binding",
            exception_type=type(exc).__name__,
        )


def _seed_successes_from_predecessor(
    *,
    predecessor_root: Path,
    output_root: Path,
    selected_by_shard: Sequence[Sequence[Mapping[str, Any]]],
    source_projection_receipt_sha256: str,
) -> dict[str, Any]:
    """Copy only validated successes into a new append-only generation."""

    predecessor = predecessor_root.resolve(strict=True)
    target = _resolve_append_only_generation_root(output_root)
    if (
        predecessor == target
        or predecessor in target.parents
        or target in predecessor.parents
    ):
        raise ValueError("predecessor and new audit generation roots must be disjoint")
    shard_count = len(selected_by_shard)
    terminal_receipts: list[str] = []
    success_receipts: list[str] = []
    success_identities: list[str] = []
    failure_identities: list[str] = []
    imported_count = 0
    for shard_index, selected in enumerate(selected_by_shard):
        predecessor_directory = _shard_directory(
            predecessor,
            shard_count,
            shard_index,
        )
        target_directory = _shard_directory(target, shard_count, shard_index)
        for ordinal, row in enumerate(selected):
            source_path = _outcome_path(
                predecessor_directory,
                ordinal,
                row["analysis_identity_id"],
            )
            if not source_path.exists():
                continue
            outcome = validate_tusz_canonical_physical_outcome_v1(
                _load_json(source_path),
                projection_row=row,
                source_projection_receipt_sha256=source_projection_receipt_sha256,
            )
            terminal_receipts.append(outcome["receipt_sha256"])
            if outcome["terminal_status"] == "failure":
                failure_identities.append(row["analysis_identity_id"])
                continue
            success_receipts.append(outcome["receipt_sha256"])
            success_identities.append(row["analysis_identity_id"])
            target_path = _outcome_path(
                target_directory,
                ordinal,
                row["analysis_identity_id"],
            )
            if target_path.exists():
                existing = validate_tusz_canonical_physical_outcome_v1(
                    _load_json(target_path),
                    projection_row=row,
                    source_projection_receipt_sha256=(source_projection_receipt_sha256),
                )
                if existing != outcome:
                    raise ValueError(
                        "new audit generation contains a different seeded outcome"
                    )
            else:
                _write_new_json(target_path, outcome)
                imported_count += 1
    body: dict[str, Any] = {
        "schema_version": "tusz_canonical_physical_audit_success_seed_v1",
        "method_id": "validated_success_only_new_append_only_generation_v1",
        "source_analysis_projection_receipt_sha256": (source_projection_receipt_sha256),
        "shard_count": shard_count,
        "predecessor_output_root": str(predecessor),
        "new_output_root": str(target),
        "predecessor_terminal_outcome_count": len(terminal_receipts),
        "predecessor_success_count": len(success_receipts),
        "predecessor_failure_count": len(failure_identities),
        "predecessor_terminal_outcome_receipt_roster_sha256": _canonical_sha256(
            terminal_receipts
        ),
        "seeded_success_identity_roster_sha256": _canonical_sha256(success_identities),
        "seeded_success_outcome_receipt_roster_sha256": _canonical_sha256(
            success_receipts
        ),
        "predecessor_failure_identity_ids": failure_identities,
        "terminal_failures_imported_as_success": False,
        "predecessor_files_overwritten": False,
        "new_generation_existing_files_overwritten": False,
        "reference_or_annotation_inputs_read": False,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    receipt_path = target / "predecessor_success_seed_receipt.json"
    if receipt_path.exists():
        if _load_json(receipt_path) != body:
            raise ValueError("predecessor success seed receipt drifted")
    else:
        _write_new_json(receipt_path, body)
    return {**body, "newly_imported_success_count": imported_count}


def _run_all(arguments: argparse.Namespace) -> int:
    """Validate the source inventory once, then execute all shards in a pool."""

    roster, projection = _inputs(arguments)
    arguments.output_root = _resolve_append_only_generation_root(arguments.output_root)
    selected_by_shard: list[list[dict[str, Any]]] = [
        [] for _ in range(arguments.shard_count)
    ]
    for global_index, row in enumerate(projection["records"]):
        selected_by_shard[global_index % arguments.shard_count].append(row)
    seed_summary: dict[str, Any] | None = None
    if arguments.seed_successes_from is not None:
        seed_summary = _seed_successes_from_predecessor(
            predecessor_root=arguments.seed_successes_from,
            output_root=arguments.output_root,
            selected_by_shard=selected_by_shard,
            source_projection_receipt_sha256=projection["receipt_sha256"],
        )
    outcomes_by_shard: list[dict[str, dict[str, Any]]] = [
        {} for _ in range(arguments.shard_count)
    ]
    pending: list[tuple[int, int, dict[str, Any], Path]] = []
    completed_manifests = 0
    for shard_index, selected in enumerate(selected_by_shard):
        directory = _shard_directory(
            arguments.output_root,
            arguments.shard_count,
            shard_index,
        )
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = validate_tusz_canonical_physical_shard_v1(
                _load_json(manifest_path),
                source_roster=roster,
                source_projection=projection,
            )
            outcomes_by_shard[shard_index] = {
                row["analysis_identity_id"]: row for row in manifest["outcomes"]
            }
            completed_manifests += 1
            continue
        for ordinal, row in enumerate(selected):
            path = _outcome_path(directory, ordinal, row["analysis_identity_id"])
            if path.exists():
                outcome = validate_tusz_canonical_physical_outcome_v1(
                    _load_json(path),
                    projection_row=row,
                    source_projection_receipt_sha256=projection["receipt_sha256"],
                )
                outcomes_by_shard[shard_index][row["analysis_identity_id"]] = outcome
            else:
                pending.append((shard_index, ordinal, row, path))

    total_missing_before_limit = len(pending)
    if arguments.max_records_this_run is not None:
        pending = pending[: arguments.max_records_this_run]
    new_success = 0
    new_failure = 0
    if pending:
        with ProcessPoolExecutor(max_workers=arguments.worker_count) as executor:
            future_bindings = {
                executor.submit(
                    _worker_materialize,
                    (
                        row,
                        projection["receipt_sha256"],
                        str(arguments.tusz_root),
                    ),
                ): (shard_index, row, path)
                for shard_index, _ordinal, row, path in pending
            }
            for future in as_completed(future_bindings):
                shard_index, row, path = future_bindings[future]
                outcome = future.result()
                _write_new_json(path, outcome)
                outcomes_by_shard[shard_index][row["analysis_identity_id"]] = outcome
                if outcome["terminal_status"] == "success":
                    new_success += 1
                else:
                    new_failure += 1

    all_outcomes_complete = all(
        len(outcomes_by_shard[index]) == len(selected_by_shard[index])
        for index in range(arguments.shard_count)
    )
    if all_outcomes_complete:
        ordered_outcomes_by_shard = [
            [
                outcomes_by_shard[shard_index][row["analysis_identity_id"]]
                for row in selected
            ]
            for shard_index, selected in enumerate(selected_by_shard)
        ]
        manifests = build_tusz_canonical_physical_shards_v1(
            source_roster=roster,
            source_projection=projection,
            outcomes_by_shard=ordered_outcomes_by_shard,
        )
        for shard_index, manifest in enumerate(manifests):
            manifest_path = (
                _shard_directory(
                    arguments.output_root,
                    arguments.shard_count,
                    shard_index,
                )
                / "manifest.json"
            )
            if not manifest_path.exists():
                _write_new_json(manifest_path, manifest)
                completed_manifests += 1

    remaining = total_missing_before_limit - len(pending)
    if arguments.max_records_this_run is None:
        remaining = sum(
            len(selected_by_shard[index]) - len(outcomes_by_shard[index])
            for index in range(arguments.shard_count)
        )
    print(
        json.dumps(
            {
                "status": (
                    "all_shards_complete"
                    if completed_manifests == arguments.shard_count
                    else "partial"
                ),
                "source_analysis_identity_count": len(projection["records"]),
                "shard_count": arguments.shard_count,
                "completed_shard_manifest_count": completed_manifests,
                "new_terminal_outcome_count": len(pending),
                "new_success_count": new_success,
                "new_failure_count": new_failure,
                "remaining_identity_count": remaining,
                "worker_count": arguments.worker_count,
                "output_root": str(arguments.output_root),
                "predecessor_success_seed": seed_summary,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _discover_shards(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.glob("shard-*-of-*/manifest.json"))
    if not paths:
        raise ValueError("no completed canonical physical shard manifests found")
    return [_load_json(path) for path in paths]


def _run_aggregate(arguments: argparse.Namespace) -> int:
    roster, projection = _inputs(arguments)
    audit = build_tusz_canonical_physical_duplicate_audit_v1(
        source_roster=roster,
        source_projection=projection,
        shards=_discover_shards(arguments.shards_root),
    )
    _write_new_json(arguments.audit_output, audit)
    complete = audit["scope_receipt"][
        "canonical_physical_signal_duplicate_audit_complete"
    ]
    projection_output: str | None = None
    if complete and arguments.physical_projection_output is not None:
        physical_projection = build_tusz_canonical_physical_analysis_projection_v1(
            audit=audit,
            source_roster=roster,
            source_projection=projection,
        )
        _write_new_json(arguments.physical_projection_output, physical_projection)
        projection_output = str(arguments.physical_projection_output)
    summary = {
        "audit_output": str(arguments.audit_output),
        "audit_id": audit["audit_id"],
        "canonical_physical_signal_duplicate_audit_complete": complete,
        "physical_projection_output": projection_output,
        **audit["physical_equivalence_inventory"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if complete else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run resumable TUSZ canonical-physical duplicate audit"
    )
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--projection", type=Path, default=DEFAULT_PROJECTION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    shard = subparsers.add_parser("shard")
    shard.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--shard-count", type=int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--max-records-this-run", type=int)
    shard.set_defaults(handler=_run_shard)

    run_all = subparsers.add_parser("run-all")
    run_all.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    run_all.add_argument("--output-root", type=Path, required=True)
    run_all.add_argument("--shard-count", type=int, default=32)
    run_all.add_argument("--worker-count", type=int, default=4)
    run_all.add_argument("--max-records-this-run", type=int)
    run_all.add_argument("--seed-successes-from", type=Path)
    run_all.set_defaults(handler=_run_all)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--shards-root", type=Path, required=True)
    aggregate.add_argument("--audit-output", type=Path, required=True)
    aggregate.add_argument("--physical-projection-output", type=Path)
    aggregate.set_defaults(handler=_run_aggregate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if getattr(arguments, "max_records_this_run", None) is not None:
        if arguments.max_records_this_run < 1:
            raise SystemExit("--max-records-this-run must be positive")
    if getattr(arguments, "worker_count", None) is not None:
        if not 1 <= arguments.worker_count <= 32:
            raise SystemExit("--worker-count must lie in [1,32]")
    if getattr(arguments, "shard_count", None) is not None:
        if arguments.shard_count < 1:
            raise SystemExit("--shard-count must be positive")
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
