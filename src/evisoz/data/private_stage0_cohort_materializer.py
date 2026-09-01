"""Patient-split-aware real private Stage-0 cohort materialization."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping

from .artifact_ref import (
    build_json_artifact_ref,
    canonical_json_bytes,
    canonical_json_sha256,
)
from .opaque_reference_authority import (
    validate_private_opaque_reference_authority,
)
from .private_stage0_split import build_private_patient_linkage_group
from .real_stage0_materializer import load_real_stage0_event_carrier
from .split_ledger import validate_split_roster
from .stage0_dual_montage_cache import (
    materialize_stage0_dual_montage_cache_to_disk,
    open_stage0_dual_montage_cache_from_disk,
)


PRIVATE_STAGE0_COHORT_SCHEMA_VERSION = "evisoz_private_real_stage0_cohort_v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_PLACEHOLDER = "0" * 64


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError("Stage-0 cohort JSON input must be an object")
    return value


def _safe_edf(root: Path, value: object) -> Path:
    relative = PurePosixPath(str(value).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".edf":
        raise ValueError("private Stage-0 roster contains an unsafe EDF path")
    source = root.joinpath(*relative.parts)
    if source.is_symlink():
        raise ValueError("private Stage-0 EDF source must not be a symbolic link")
    resolved = source.resolve(strict=True)
    resolved.relative_to(root)
    return resolved


def _failure_code(error: BaseException) -> str:
    message = str(error)
    if "discontinuous EDF+D" in message:
        return "edf_discontinuous_clock"
    if "causal warmup" in message:
        return "insufficient_causal_warmup"
    if "complete +48-second context" in message:
        return "insufficient_post_context"
    if "opaque reference" in message or "reference authority" in message:
        return "reference_authority_rejected"
    if isinstance(error, OSError):
        return "edf_io_error"
    if isinstance(error, (ValueError, TypeError)):
        return "signal_or_contract_rejected"
    return "unexpected_materialization_error"


def _load_roster(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "event_id",
        "patient_id",
        "relative_edf_path",
        "global_event_t0_sec",
        "time_source",
        "time_support_preeligible",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("private Stage-0 signal roster fields drifted")
    if len({row["event_id"] for row in rows}) != len(rows):
        raise ValueError("private Stage-0 signal roster event IDs are duplicated")
    return rows


def materialize_private_stage0_cohort(
    *,
    signal_roster_path: str | Path,
    eeg_root: str | Path,
    split_roster_path: str | Path,
    split_manifest_path: str | Path,
    reference_authority_path: str | Path,
    output: str | Path,
    limit: int | None = None,
    raise_on_event_error: bool = False,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, Any]:
    """Materialize all time-supported events and persist closed exclusions."""

    roster_path = Path(signal_roster_path).resolve(strict=True)
    root = Path(eeg_root).resolve(strict=True)
    split = _json(Path(split_roster_path))
    split_manifest = _json(Path(split_manifest_path))
    authority = validate_private_opaque_reference_authority(
        _json(Path(reference_authority_path))
    )
    rows = _load_roster(roster_path)
    roster_sha256 = _sha256_file(roster_path)
    if split_manifest.get("input_signal_roster_sha256") != roster_sha256:
        raise ValueError("private Stage-0 split belongs to another signal roster")
    expected_split_ref = build_json_artifact_ref(
        split,
        artifact_kind="split_roster",
        payload_schema_version="evisoz_split_roster_v1",
    )
    if split_manifest.get("split_roster_ref") != expected_split_ref:
        raise ValueError("private Stage-0 split manifest binding drifted")

    patient_keys = sorted({row["patient_id"] for row in rows})
    trusted_groups = {
        group["linkage_group_id"]: group
        for group in (
            build_private_patient_linkage_group(patient_key)
            for patient_key in patient_keys
        )
    }
    split = validate_split_roster(split, trusted_linkage_groups=trusted_groups)
    assignment_by_group = {
        row["linkage_group_id"]: row for row in split["assignments"]
    }
    patient_context: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for patient_key in patient_keys:
        group = build_private_patient_linkage_group(patient_key)
        patient_context[patient_key] = (
            group,
            assignment_by_group[group["linkage_group_id"]],
        )

    selected: list[dict[str, str]] = []
    preexcluded: list[dict[str, object]] = []
    for row in rows:
        event_id = row["event_id"]
        patient_id = row["patient_id"]
        if _SAFE_ID.fullmatch(event_id) is None or _SAFE_ID.fullmatch(patient_id) is None:
            raise ValueError("private Stage-0 pseudonymous identifiers are invalid")
        try:
            onset = float(row["global_event_t0_sec"])
        except (TypeError, ValueError):
            onset = float("nan")
        if row["time_support_preeligible"] != "1" or not math.isfinite(onset):
            group, assignment = patient_context[patient_id]
            preexcluded.append(
                {
                    "event_id": event_id,
                    "patient_id": patient_id,
                    "linkage_group_id": group["linkage_group_id"],
                    "evisoz_role": assignment["evisoz_role"],
                    "outer_holdout_fold": assignment["outer_holdout_fold"],
                    "reason_code": "time_support_not_preeligible",
                }
            )
            continue
        selected.append(row)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        selected = selected[:limit]

    target = Path(output).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    committed = False
    materialized: list[dict[str, object]] = []
    runtime_excluded: list[dict[str, object]] = []
    try:
        events_root = staging / "events"
        events_root.mkdir()
        for ordinal, row in enumerate(selected, start=1):
            event_id = row["event_id"]
            patient_id = row["patient_id"]
            group, assignment = patient_context[patient_id]
            source = _safe_edf(root, row["relative_edf_path"])
            anchor_quality = (
                "exact" if row["time_source"] == "exact_sz_marker" else "approximate"
            )
            anchor_payload = {
                "schema_version": "evisoz_private_known_seizure_anchor_selection_v1",
                "event_id": event_id,
                "global_event_t0_seconds": float(row["global_event_t0_sec"]),
                "anchor_quality": anchor_quality,
                "source_signal_roster_sha256": roster_sha256,
                "clinical_target_values_used": False,
            }
            anchor_ref = build_json_artifact_ref(
                anchor_payload,
                artifact_kind="analysis_selection_receipt",
                payload_schema_version=str(anchor_payload["schema_version"]),
            )
            try:
                result = load_real_stage0_event_carrier(
                    source,
                    float(row["global_event_t0_sec"]),
                    dataset_id="private",
                    sample_id=event_id,
                    event_id=event_id,
                    linkage_group_id=group["linkage_group_id"],
                    source_patient_sha256=group["members"][0][
                        "source_patient_sha256"
                    ],
                    anchor_source_ref=anchor_ref,
                    anchor_quality=anchor_quality,
                    opaque_reference_authority=authority,
                )
                event_root = events_root / event_id
                event_root.mkdir()
                cache = materialize_stage0_dual_montage_cache_to_disk(
                    event_root / "dual_montage",
                    result.carrier,
                )
                preprocessing_bytes = canonical_json_bytes(
                    dict(result.preprocessing_receipt)
                )
                (event_root / "preprocessing_receipt.json").write_bytes(
                    preprocessing_bytes
                )
                materialized.append(
                    {
                        "event_id": event_id,
                        "patient_id": patient_id,
                        "linkage_group_id": group["linkage_group_id"],
                        "evisoz_role": assignment["evisoz_role"],
                        "outer_holdout_fold": assignment["outer_holdout_fold"],
                        "anchor_quality": anchor_quality,
                        "source_edf_sha256": result.preprocessing_receipt[
                            "source_edf_sha256"
                        ],
                        "preprocessing_receipt_sha256": result.preprocessing_receipt[
                            "receipt_sha256"
                        ],
                        "cache_materialization_receipt_sha256": cache.materialization_receipt[
                            "receipt_sha256"
                        ],
                        "reference_route": result.preprocessing_receipt[
                            "reference_route"
                        ],
                        "relative_cache_path": f"events/{event_id}/dual_montage",
                        "relative_preprocessing_receipt_path": (
                            f"events/{event_id}/preprocessing_receipt.json"
                        ),
                    }
                )
            except Exception as exc:  # closed aggregate attrition receipt
                if raise_on_event_error:
                    raise
                runtime_excluded.append(
                    {
                        "event_id": event_id,
                        "patient_id": patient_id,
                        "linkage_group_id": group["linkage_group_id"],
                        "evisoz_role": assignment["evisoz_role"],
                        "outer_holdout_fold": assignment["outer_holdout_fold"],
                        "reason_code": _failure_code(exc),
                        "exception_type": type(exc).__name__,
                    }
                )
            if progress is not None:
                progress(
                    {
                        "complete": ordinal,
                        "selected": len(selected),
                        "materialized": len(materialized),
                        "runtime_excluded": len(runtime_excluded),
                    }
                )

        materialized.sort(key=lambda row: str(row["event_id"]))
        preexcluded.sort(key=lambda row: str(row["event_id"]))
        runtime_excluded.sort(key=lambda row: str(row["event_id"]))
        role_counts = Counter(row["evisoz_role"] for row in materialized)
        exclusion_counts = Counter(
            row["reason_code"] for row in [*preexcluded, *runtime_excluded]
        )
        manifest: dict[str, Any] = {
            "schema_version": PRIVATE_STAGE0_COHORT_SCHEMA_VERSION,
            "status": "completed_real_private_stage0_materialization",
            "signal_roster_sha256": roster_sha256,
            "split_roster_ref": expected_split_ref,
            "reference_authority_ref": build_json_artifact_ref(
                authority,
                artifact_kind="reference_authority",
                payload_schema_version=str(authority["schema_version"]),
            ),
            "candidate_event_count": len(selected),
            "materialized_event_count": len(materialized),
            "preexcluded_event_count": len(preexcluded),
            "runtime_excluded_event_count": len(runtime_excluded),
            "materialized_role_event_counts": dict(sorted(role_counts.items())),
            "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
            "events": materialized,
            "preexcluded_events": preexcluded,
            "runtime_excluded_events": runtime_excluded,
            "access_receipt": {
                "known_seizure_anchor_selection_used": True,
                "raw_private_eeg_samples_used": True,
                "edf_signal_headers_used": True,
                "edf_annotations_used": False,
                "target_ledger_opened": False,
                "clinical_target_values_used": False,
                "physician_reports_used": False,
                "knowledge_base_used": False,
                "training_performed": False,
            },
            "claim_boundary": {
                "opaque_reference_is_header_proven": False,
                "edf_discontinuity_was_repaired_or_concatenated": False,
                "spherical_interpolation_used": False,
                "tcp22_is_signed_endpoint_difference": True,
                "tcp22_is_cortical_source_localization": False,
            },
            "receipt_sha256": _PLACEHOLDER,
        }
        manifest["receipt_sha256"] = canonical_json_sha256(manifest)
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        staging.rename(target)
        committed = True
        return manifest
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


def validate_private_stage0_cohort_artifact(root: str | Path) -> dict[str, Any]:
    """Replay every real event cache and return an aggregate validation receipt."""

    cohort_root = Path(root).resolve(strict=True)
    manifest_path = cohort_root / "manifest.json"
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest.decode("utf-8"))
    if raw_manifest != canonical_json_bytes(manifest):
        raise ValueError("private Stage-0 cohort manifest is not canonical JSON")
    if (
        manifest.get("schema_version") != PRIVATE_STAGE0_COHORT_SCHEMA_VERSION
        or manifest.get("status") != "completed_real_private_stage0_materialization"
    ):
        raise ValueError("private Stage-0 cohort schema/status drifted")
    replay = dict(manifest)
    replay["receipt_sha256"] = _PLACEHOLDER
    if manifest.get("receipt_sha256") != canonical_json_sha256(replay):
        raise ValueError("private Stage-0 cohort receipt hash drifted")
    events = manifest.get("events")
    if (
        not isinstance(events, list)
        or len(events) != manifest.get("materialized_event_count")
        or len({row.get("event_id") for row in events}) != len(events)
    ):
        raise ValueError("private Stage-0 cohort event inventory drifted")

    role_counts: Counter[object] = Counter()
    reference_routes: Counter[object] = Counter()
    edge_state_counts: Counter[object] = Counter()
    for row in events:
        if type(row) is not dict or _SAFE_ID.fullmatch(str(row.get("event_id", ""))) is None:
            raise ValueError("private Stage-0 cohort event row is invalid")
        event_root = cohort_root / "events" / str(row["event_id"])
        if event_root.resolve(strict=True).parent != (cohort_root / "events").resolve(strict=True):
            raise ValueError("private Stage-0 event directory escaped its cohort")
        preprocessing_path = event_root / "preprocessing_receipt.json"
        preprocessing_raw = preprocessing_path.read_bytes()
        preprocessing = json.loads(preprocessing_raw.decode("utf-8"))
        if preprocessing_raw != canonical_json_bytes(preprocessing):
            raise ValueError("private Stage-0 preprocessing receipt is not canonical JSON")
        preprocessing_replay = dict(preprocessing)
        preprocessing_replay["receipt_sha256"] = _PLACEHOLDER
        if (
            preprocessing.get("receipt_sha256")
            != canonical_json_sha256(preprocessing_replay)
            or preprocessing.get("receipt_sha256")
            != row.get("preprocessing_receipt_sha256")
            or preprocessing.get("reference_route")
            != "protocol_authorized_opaque_common_reference"
        ):
            raise ValueError("private Stage-0 preprocessing receipt drifted")
        opened = open_stage0_dual_montage_cache_from_disk(event_root / "dual_montage")
        if (
            opened.materialization_receipt["receipt_sha256"]
            != row.get("cache_materialization_receipt_sha256")
            or opened.montage_receipt["permissions"][
                "residual_main_analysis_eligible"
            ]
            is not True
            or opened.montage_receipt["permissions"][
                "tcp22_standalone_evidence_available"
            ]
            is not True
        ):
            raise ValueError("private Stage-0 cache permission/receipt drifted")
        states = [
            support["support_state"]
            for support in opened.montage_receipt["edge_support"]
        ]
        if states != [
            "exact_derived_from_protocol_authorized_opaque_common_reference"
        ] * 22:
            raise ValueError("private Stage-0 TCP22 support route drifted")
        role_counts[row["evisoz_role"]] += 1
        reference_routes[preprocessing["reference_route"]] += 1
        edge_state_counts.update(states)

    if dict(sorted(role_counts.items())) != manifest.get(
        "materialized_role_event_counts"
    ):
        raise ValueError("private Stage-0 role counts do not replay")
    result: dict[str, Any] = {
        "schema_version": "evisoz_private_real_stage0_cohort_validation_v1",
        "status": "passed_full_event_cache_replay",
        "cohort_manifest_sha256": _sha256_file(manifest_path),
        "cohort_receipt_sha256": manifest["receipt_sha256"],
        "validated_event_count": len(events),
        "role_event_counts": dict(sorted(role_counts.items())),
        "reference_route_counts": dict(sorted(reference_routes.items())),
        "tcp22_edge_support_state_counts": dict(sorted(edge_state_counts.items())),
        "source_paths_or_raw_signal_labels_serialized": False,
        "receipt_sha256": _PLACEHOLDER,
    }
    result["receipt_sha256"] = canonical_json_sha256(result)
    return result


__all__ = [
    "PRIVATE_STAGE0_COHORT_SCHEMA_VERSION",
    "materialize_private_stage0_cohort",
    "validate_private_stage0_cohort_artifact",
]
