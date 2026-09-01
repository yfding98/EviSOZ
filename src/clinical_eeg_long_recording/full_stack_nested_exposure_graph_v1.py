"""Materialize a patient-closed full-stack nested exposure graph.

This module is a lineage planner and verifier, not a trainer.  It binds the
canonical TUSZ physical five-fold plan, the source-train-only DeepSOZ identity
bridge, target-free DeepSOZ audit artifacts, and a byte-identity-only TUEV
snapshot.  It never reads seizure/SOZ labels, EDF annotations, clinical text,
spreadsheets, private data, or a source-eval reference.

The graph deliberately distinguishes five detector checkpoint *slots* per arm
from a legal full-stack OOF inventory.  A legal outer-fold prediction also
requires outer-test-closed preprocessing, SSL (if any), detector, core, ranker,
router-target endpoint, router, and calibrator artifacts.  Missing external
patient/content authority fails closed to an empty exposure roster.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Final, Mapping, Sequence

from .deepsoz_published_external_exposure_attestation_v1 import (
    validate_deepsoz_published_external_exposure_attestation_v1,
)
from .deepsoz_tusz_identity_binding_v1 import (
    validate_deepsoz_tusz_source_train_identity_binding_v1,
)
from .tusz_canonical_physical_signal_audit_v1 import (
    validate_tusz_canonical_physical_analysis_projection_v1,
)
from .tusz_detector_cleanroom_fold_plan_v1 import (
    validate_tusz_detector_cleanroom_fold_plan_v1,
)


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
FULL_STACK_NESTED_EXPOSURE_PLAN_V1_SCHEMA: Final[str] = (
    "clinical_eeg_full_stack_nested_exposure_plan_v1"
)
FULL_STACK_NESTED_EXPOSURE_REGISTRY_V1_SCHEMA: Final[str] = (
    "clinical_eeg_full_stack_nested_exposure_registry_v1"
)
FULL_STACK_NESTED_EXPOSURE_PLAN_V1_ID: Final[str] = (
    "CLINICAL-EEG-FULL-STACK-NESTED-EXPOSURE-PLAN-V1-20260824"
)
DEFAULT_PLAN_PATH: Final[Path] = (
    ROOT / "configs" / "clinical_eeg_full_stack_nested_exposure_plan_v1.json"
)
DEFAULT_REGISTRY_PATH: Final[Path] = (
    ROOT
    / "outputs"
    / "clinical_eeg_full_stack_nested_exposure_graph_v1_20260824r1"
    / "exposure_registry.json"
)

DEFAULT_PLAN_SHA256: Final[str] = (
    "ec99c5be4eb40ec9a7a64672184d3d18b398db099cb69c249f0a28412db1745c"
)

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_TUEV_TRAIN_PATIENT_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9]{8}$")
_TUEV_EVAL_SESSION_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9]{3}$")
_REQUIRED_DATASETS: Final[tuple[str, ...]] = (
    "TUSZ",
    "SzCORE",
    "DeepSOZ",
    "TUEV",
    "TUAR",
)
_REQUIRED_EXPOSURE_STAGES: Final[tuple[str, ...]] = (
    "detector",
    "core",
    "ranker",
    "router",
    "SSL",
    "weak_label",
    "calibrator",
)
_DETECTOR_ARMS: Final[tuple[str, ...]] = (
    "seizuretransformer_cleanroom_retrained_v1",
    "eventnet_cleanroom_retrained_v1",
)
_FORBIDDEN_FORWARD_SOURCES: Final[tuple[str, ...]] = (
    "EDF_annotations",
    "annotation_channels",
    "Excel_or_spreadsheet_fields",
    "doctor_labels_or_reports",
    "clinical_history",
    "video_or_behavior",
    "sleep_staging_or_activation",
    "provocation_or_activation",
    "ECG_EMG_EOG",
    "LLM_output_as_evidence",
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _identifier(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{context} must be a normalized non-empty string")
    return value


def _strict_object(
    value: object, expected_fields: set[str] | frozenset[str], context: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(expected_fields):
        raise ValueError(f"{context} fields drifted")
    return deepcopy(value)


def _file_sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _regular_file(path: str | Path, context: str) -> Path:
    lexical = Path(path).absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    resolved = lexical.resolve(strict=True)
    if resolved != lexical:
        raise ValueError(f"{context} must use its canonical path")
    return resolved


def _regular_directory(path: str | Path, context: str) -> Path:
    lexical = Path(path).absolute()
    if lexical.is_symlink() or not lexical.is_dir():
        raise ValueError(f"{context} must be a regular non-symlink directory")
    resolved = lexical.resolve(strict=True)
    if resolved != lexical:
        raise ValueError(f"{context} must use its canonical path")
    return resolved


def _load_json_file(path: str | Path, context: str) -> tuple[dict[str, Any], Path]:
    source = _regular_file(path, context)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise TypeError(f"{context} must contain one JSON object")
    return payload, source


def _relative_or_absolute_source_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _source_binding(
    *, role: str, path: Path, schema_version: str, semantic_id: str, receipt: str
) -> dict[str, Any]:
    return {
        "role": role,
        "path": _relative_or_absolute_source_path(path),
        "file_sha256": _file_sha256(path),
        "schema_version": schema_version,
        "semantic_id": semantic_id,
        "semantic_receipt_sha256": _sha256(receipt, f"{role} receipt"),
    }


def validate_full_stack_nested_exposure_plan_v1(
    value: Mapping[str, Any],
    *,
    trusted_plan_sha256: str = DEFAULT_PLAN_SHA256,
) -> dict[str, Any]:
    """Validate the additive policy plan and its parent binding."""

    required = {
        "schema_version",
        "plan_id",
        "status",
        "parent_binding",
        "population_authority",
        "split_contract",
        "nested_schedule",
        "stage_contract",
        "dataset_contract",
        "forward_source_firewall",
        "detector_vs_full_stack_oof",
        "current_permissions",
        "plan_sha256",
    }
    plan = _strict_object(value, required, "full-stack exposure plan")
    if plan["schema_version"] != FULL_STACK_NESTED_EXPOSURE_PLAN_V1_SCHEMA:
        raise ValueError("full-stack exposure plan schema drifted")
    if plan["plan_id"] != FULL_STACK_NESTED_EXPOSURE_PLAN_V1_ID:
        raise ValueError("full-stack exposure plan ID drifted")
    if plan["status"] != "additive_plan_materialized_execution_untrained":
        raise ValueError("full-stack exposure plan status drifted")
    declared = _sha256(plan["plan_sha256"], "plan receipt")
    body = deepcopy(plan)
    body.pop("plan_sha256")
    if _canonical_sha256(body) != declared:
        raise ValueError("full-stack exposure plan receipt does not replay")
    if trusted_plan_sha256 != "PLAN-SHA256-PENDING" and declared != _sha256(
        trusted_plan_sha256, "trusted plan receipt"
    ):
        raise ValueError("full-stack exposure plan is not the trusted plan")

    parent = _strict_object(
        plan["parent_binding"],
        {"path", "file_sha256", "schema_version", "semantic_receipt_sha256"},
        "full-stack parent binding",
    )
    parent_path = ROOT / parent["path"]
    if parent_path.is_symlink() or not parent_path.is_file():
        raise ValueError("full-stack parent binding is unavailable")
    if _file_sha256(parent_path) != _sha256(
        parent["file_sha256"], "parent file SHA-256"
    ):
        raise ValueError("full-stack parent binding file drifted")
    parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        parent_payload.get("schema_version") != parent["schema_version"]
        or parent_payload.get("addendum_sha256")
        != parent["semantic_receipt_sha256"]
    ):
        raise ValueError("full-stack parent semantic binding drifted")

    authority = plan["population_authority"]
    if authority != {
        "outer_plan_id": "TUSZDETCLEANFOLDV1-c4808802f6ab2626332782b9",
        "split_unit": "patient",
        "outer_fold_count": 5,
        "source_train_patients": 579,
        "source_train_physical_records": 4664,
        "source_dev_reference_role": "closed_during_nested_training",
        "source_eval_reference_role": "zero_access",
    }:
        raise ValueError("full-stack population authority drifted")
    split = plan["split_contract"]
    if (
        split.get("outer_test_patient_may_reach_fit_selection_or_calibration")
        is not False
        or split.get("unknown_external_exposure_authorized") is not False
        or split.get("downstream_row_requires_upstream_patient_exclusion") is not True
    ):
        raise ValueError("full-stack patient firewall was weakened")
    if split.get("unresolved_patient_or_content_identity_action") != (
        "empty_exposure_roster_and_training_block"
    ):
        raise ValueError("unresolved cross-dataset exposure no longer fails closed")

    schedule = plan["nested_schedule"]
    if schedule != {
        "partition_groups": "five_held_out_groups_from_bound_TUSZ_outer_plan",
        "outer_train_group_count": 4,
        "feature_row_crossfit": "leave_one_remaining_outer_group_out",
        "router_target_endpoint_crossfit": (
            "leave_target_group_out_then_build_endpoint_on_three_groups_using_"
            "three_way_detector_row_crossfit"
        ),
        "calibrator_prediction_crossfit": (
            "leave_one_remaining_outer_group_out_full_stack_predictions_only"
        ),
        "final_outer_inference": (
            "freeze_all_stage_artifacts_before_prediction_first_outer_test"
        ),
    }:
        raise ValueError("full-stack nested schedule drifted")
    stages = plan["stage_contract"]
    if not isinstance(stages, list) or [row.get("stage") for row in stages] != list(
        _REQUIRED_EXPOSURE_STAGES
    ):
        raise ValueError("full-stack stage roster drifted")
    for row in stages:
        if set(row) != {
            "stage",
            "outer_test_fit_exposure_allowed",
            "fold_scoped_artifact_required",
            "actual_exposure_receipt_required",
        }:
            raise ValueError("full-stack stage contract fields drifted")
        if row["outer_test_fit_exposure_allowed"] is not False:
            raise ValueError("outer-test fit exposure was opened")
        if row["fold_scoped_artifact_required"] is not True:
            raise ValueError("fold-scoped stage artifact was weakened")
        if row["actual_exposure_receipt_required"] is not True:
            raise ValueError("actual stage exposure receipt was weakened")

    datasets = plan["dataset_contract"]
    if not isinstance(datasets, list) or [row.get("dataset_id") for row in datasets] != list(
        _REQUIRED_DATASETS
    ):
        raise ValueError("cross-dataset contract roster drifted")
    for row in datasets:
        if row.get("patient_identity_required") is not True:
            raise ValueError("cross-dataset patient identity requirement weakened")
        if row.get("exact_content_identity_required") is not True:
            raise ValueError("cross-dataset exact-content requirement weakened")
        if row.get("near_or_partial_overlap_audit_required") is not True:
            raise ValueError("cross-dataset partial-overlap requirement weakened")
        if row.get("unresolved_identity_may_train") is not False:
            raise ValueError("unresolved cross-dataset identity may train")

    firewall = plan["forward_source_firewall"]
    if firewall.get("allowed") != [
        "canonical_physical_EEG_samples",
        "recording_relative_sampling_clock",
        "allowlisted_signal_acquisition_metadata",
        "EEG_derived_QC_and_support_lineage",
    ]:
        raise ValueError("EEG-only forward allowlist drifted")
    if tuple(firewall.get("forbidden", ())) != _FORBIDDEN_FORWARD_SOURCES:
        raise ValueError("EEG-only forward denylist drifted")

    distinction = plan["detector_vs_full_stack_oof"]
    if distinction != {
        "detector_checkpoint_slots_per_arm": 5,
        "detector_arm_count": 2,
        "five_detector_checkpoints_per_arm_equal_full_stack_OOF": False,
        "detector_only_OOF_may_be_reported_as_full_stack_OOF": False,
        "full_stack_requires": [
            "fold_scoped_preprocessing",
            "fold_scoped_SSL_or_explicit_no_SSL_receipt",
            "fold_scoped_detector",
            "inner_OOF_candidate_and_feature_rows",
            "fold_scoped_core",
            "fold_scoped_ranker",
            "cross_fitted_router_targets",
            "fold_scoped_router",
            "cross_fitted_calibrator_inputs",
            "fold_scoped_calibrator",
            "prediction_first_outer_test_inventory",
        ],
    }:
        raise ValueError("detector/full-stack OOF distinction drifted")
    permissions = plan["current_permissions"]
    for field in (
        "model_training_authorized",
        "source_eval_reference_open_authorized",
        "full_stack_OOF_claim_authorized",
        "performance_claim_authorized",
        "clinical_use_authorized",
    ):
        if permissions.get(field) is not False:
            raise ValueError(f"premature full-stack permission opened: {field}")
    return plan


def load_full_stack_nested_exposure_plan_v1(
    path: str | Path = DEFAULT_PLAN_PATH,
) -> dict[str, Any]:
    payload, _ = _load_json_file(path, "full-stack exposure plan")
    return validate_full_stack_nested_exposure_plan_v1(payload)


def _validate_deepsoz_union_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = deepcopy(dict(value))
    if manifest.get("schema_version") != "soz_public_development_union_identity_v12":
        raise ValueError("DeepSOZ public-development identity manifest drifted")
    if manifest.get("patient_count") != 103 or manifest.get("event_count") != 1149:
        raise ValueError("DeepSOZ public-development identity denominator drifted")
    if manifest.get("outer_fold_count") != 5:
        raise ValueError("DeepSOZ public-development fold count drifted")
    declared = _sha256(
        manifest.get("manifest_payload_sha256"), "DeepSOZ union payload receipt"
    )
    body = deepcopy(manifest)
    body.pop("manifest_payload_sha256")
    if _canonical_sha256(body) != declared:
        raise ValueError("DeepSOZ union payload receipt does not replay")
    access = manifest.get("access_receipt")
    if not isinstance(access, Mapping):
        raise TypeError("DeepSOZ union access receipt is missing")
    for forbidden in (
        "deepsoz_target_values_loaded",
        "prediction_artifacts_loaded",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "raw_eeg_loaded",
    ):
        if access.get(forbidden) is not False:
            raise PermissionError(f"DeepSOZ union opened forbidden source: {forbidden}")
    return manifest


def _scan_tuev_eeg_identity(edf_root: str | Path) -> dict[str, Any]:
    """Hash only TUEV EDF containers; REC/LAB/HTK files remain unopened."""

    root = _regular_directory(edf_root, "TUEV EDF root")
    if root.name != "edf":
        raise ValueError("TUEV identity root must be the canonical edf directory")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.edf"), key=lambda item: item.as_posix()):
        if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
            raise ValueError("TUEV EDF identity may not traverse a symlink")
        relative = path.relative_to(root)
        if len(relative.parts) != 3:
            raise ValueError("TUEV EDF identity path depth drifted")
        split, group, basename = relative.parts
        if split == "train":
            if _TUEV_TRAIN_PATIENT_RE.fullmatch(group) is None:
                raise ValueError("TUEV train patient identity is not canonical")
            patient_id: str | None = group
            group_kind = "visible_train_patient"
        elif split == "eval":
            if _TUEV_EVAL_SESSION_RE.fullmatch(group) is None:
                raise ValueError("TUEV eval session identity is not canonical")
            patient_id = None
            group_kind = "opaque_eval_session_not_patient_authority"
        else:
            raise ValueError("TUEV official split drifted")
        if PurePosixPath(basename).suffix.lower() != ".edf":
            raise ValueError("TUEV EDF basename drifted")
        rows.append(
            {
                "relative_edf_path": relative.as_posix(),
                "official_split": split,
                "group_id": group,
                "group_kind": group_kind,
                "patient_id": patient_id,
                "container_bytes": path.stat().st_size,
                "container_sha256": _file_sha256(path),
            }
        )
    if not rows:
        raise ValueError("TUEV identity scan found no EDF containers")
    train_patients = sorted(
        {str(row["patient_id"]) for row in rows if row["patient_id"] is not None}
    )
    eval_sessions = sorted(
        {str(row["group_id"]) for row in rows if row["official_split"] == "eval"}
    )
    container_hashes: dict[str, list[str]] = {}
    for row in rows:
        container_hashes.setdefault(str(row["container_sha256"]), []).append(
            str(row["relative_edf_path"])
        )
    duplicate_classes = [
        {"container_sha256": digest, "relative_edf_paths": sorted(paths)}
        for digest, paths in sorted(container_hashes.items())
        if len(paths) > 1
    ]
    return {
        "root": str(root),
        "record_count": len(rows),
        "train_record_count": sum(row["official_split"] == "train" for row in rows),
        "eval_record_count": sum(row["official_split"] == "eval" for row in rows),
        "train_patient_count": len(train_patients),
        "eval_opaque_session_count": len(eval_sessions),
        "train_patient_ids": train_patients,
        "train_patient_roster_sha256": _canonical_sha256(train_patients),
        "eval_session_ids": eval_sessions,
        "eval_session_roster_sha256": _canonical_sha256(eval_sessions),
        "record_identity_roster_sha256": _canonical_sha256(rows),
        "container_sha256_roster_sha256": _canonical_sha256(
            [row["container_sha256"] for row in rows]
        ),
        "unique_exact_container_count": len(container_hashes),
        "exact_duplicate_classes": duplicate_classes,
        "exact_duplicate_class_roster_sha256": _canonical_sha256(duplicate_classes),
        "records": rows,
    }


def _partition_groups(tusz_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups = []
    for fold in tusz_plan["folds"]:
        held = fold["held_out_roster"]
        group_id = f"G{fold['fold_id']}"
        groups.append(
            {
                "group_id": group_id,
                "source_fold_id": fold["fold_id"],
                "patient_count": held["patient_count"],
                "patient_ids": list(held["patient_ids"]),
                "patient_roster_sha256": held["patient_roster_sha256"],
                "analysis_identity_count": held["recording_count"],
                "analysis_identity_roster_sha256": held[
                    "analysis_identity_roster_sha256"
                ],
                "duration_seconds_fraction": list(held["duration_seconds_fraction"]),
            }
        )
    return groups


def _crossfit_rows(group_ids: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "held_group_id": held,
            "fit_group_ids": [group for group in group_ids if group != held],
        }
        for held in group_ids
    ]


def _router_target_rows(group_ids: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for target in group_ids:
        endpoint_groups = [group for group in group_ids if group != target]
        rows.append(
            {
                "target_group_id": target,
                "endpoint_fit_group_ids": endpoint_groups,
                "endpoint_detector_feature_crossfit": _crossfit_rows(
                    endpoint_groups
                ),
                "target_patient_absent_from_endpoint_exposure": True,
            }
        )
    return rows


def _stage_node(
    *,
    node_id: str,
    stage: str,
    dependencies: Sequence[str],
    fit_groups: Sequence[str],
    inference_groups: Sequence[str] = (),
    external_dataset_ids: Sequence[str] = (),
    external_candidate_local_patient_ids: Sequence[str] = (),
    external_exposure_local_patient_ids: Sequence[str] = (),
    status: str = "planned_unmaterialized",
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "stage": stage,
        "dependency_node_ids": list(dependencies),
        "tusz_fit_group_ids": list(fit_groups),
        "tusz_inference_group_ids": list(inference_groups),
        "external_dataset_ids": list(external_dataset_ids),
        "external_candidate_local_patient_ids": list(
            external_candidate_local_patient_ids
        ),
        "external_exposure_local_patient_ids": list(
            external_exposure_local_patient_ids
        ),
        "materialization_status": status,
    }


def _outer_fold_graph(
    *,
    outer_fold_id: int,
    group_ids: Sequence[str],
    group_patient_lookup: Mapping[str, set[str]],
    deepsoz_local_patients: set[str],
) -> dict[str, Any]:
    test_group = f"G{outer_fold_id}"
    outer_train = [group for group in group_ids if group != test_group]
    outer_train_patients = set().union(
        *(group_patient_lookup[group] for group in outer_train)
    )
    deepsoz_candidates = sorted(deepsoz_local_patients & outer_train_patients)
    prefix = f"outer-{outer_fold_id}"
    source = f"{prefix}:tusz_signal_source"
    weak = f"{prefix}:deepsoz_weak_label_gate"
    ssl = f"{prefix}:ssl_fit_or_no_ssl_receipt"
    preprocess = f"{prefix}:preprocessing_fit"
    detector_nodes = [
        f"{prefix}:detector:{arm}" for arm in _DETECTOR_ARMS
    ]
    candidate_rows = f"{prefix}:inner_detector_oof_candidate_rows"
    core = f"{prefix}:core"
    ranker = f"{prefix}:ranker"
    router_targets = f"{prefix}:router_target_endpoint_bank"
    router = f"{prefix}:router"
    calibrator = f"{prefix}:calibrator"
    frozen = f"{prefix}:frozen_stack"
    inference = f"{prefix}:outer_test_inference"
    nodes = [
        _stage_node(
            node_id=source,
            stage="signal_source",
            dependencies=(),
            fit_groups=outer_train,
        ),
        _stage_node(
            node_id=weak,
            stage="weak_label",
            dependencies=(source,),
            fit_groups=outer_train,
            external_dataset_ids=("DeepSOZ",),
            external_candidate_local_patient_ids=deepsoz_candidates,
            external_exposure_local_patient_ids=(),
            status="identity_candidates_registered_targets_unopened_no_exposure",
        ),
        _stage_node(
            node_id=ssl,
            stage="SSL",
            dependencies=(source,),
            fit_groups=outer_train,
            status="fold_local_scratch_or_explicit_no_SSL_only_external_weights_blocked",
        ),
        _stage_node(
            node_id=preprocess,
            stage="preprocessing",
            dependencies=(source,),
            fit_groups=outer_train,
        ),
    ]
    for node_id in detector_nodes:
        nodes.append(
            _stage_node(
                node_id=node_id,
                stage="detector",
                dependencies=(preprocess, ssl),
                fit_groups=outer_train,
            )
        )
    nodes.extend(
        [
            _stage_node(
                node_id=candidate_rows,
                stage="inner_detector_OOF_rows",
                dependencies=tuple(detector_nodes),
                fit_groups=outer_train,
                status="schedule_materialized_rows_not_generated",
            ),
            _stage_node(
                node_id=core,
                stage="core",
                dependencies=(candidate_rows, weak),
                fit_groups=outer_train,
            ),
            _stage_node(
                node_id=ranker,
                stage="ranker",
                dependencies=(core,),
                fit_groups=outer_train,
            ),
            _stage_node(
                node_id=router_targets,
                stage="router_target_endpoint",
                dependencies=(candidate_rows, core, ranker),
                fit_groups=outer_train,
                status="nested_schedule_materialized_targets_not_generated",
            ),
            _stage_node(
                node_id=router,
                stage="router",
                dependencies=(router_targets,),
                fit_groups=outer_train,
            ),
            _stage_node(
                node_id=calibrator,
                stage="calibrator",
                dependencies=(ranker, router),
                fit_groups=outer_train,
                status="crossfit_schedule_materialized_inputs_not_generated",
            ),
            _stage_node(
                node_id=frozen,
                stage="frozen_stack",
                dependencies=tuple(
                    [preprocess, ssl, weak, *detector_nodes, core, ranker, router, calibrator]
                ),
                fit_groups=outer_train,
            ),
            _stage_node(
                node_id=inference,
                stage="outer_test_inference",
                dependencies=(frozen,),
                fit_groups=(),
                inference_groups=(test_group,),
                status="planned_prediction_first_no_artifacts",
            ),
        ]
    )
    return {
        "outer_fold_id": outer_fold_id,
        "outer_test_group_id": test_group,
        "outer_train_group_ids": outer_train,
        "inner_feature_row_crossfit": _crossfit_rows(outer_train),
        "router_counterfactual_target_crossfit": _router_target_rows(outer_train),
        "calibrator_prediction_crossfit": _crossfit_rows(outer_train),
        "nodes": nodes,
        "outer_inference_node_id": inference,
    }


def build_full_stack_nested_exposure_registry_v1(
    *,
    plan: Mapping[str, Any],
    tusz_fold_plan: Mapping[str, Any],
    tusz_physical_projection: Mapping[str, Any],
    deepsoz_identity_binding: Mapping[str, Any],
    deepsoz_public_union_manifest: Mapping[str, Any],
    deepsoz_external_attestation: Mapping[str, Any],
    source_paths: Mapping[str, Path],
    tuev_edf_root: str | Path,
    tuev_readme_path: str | Path,
    tuar_metadata_root: str | Path,
    tuar_audit_receipt_path: str | Path,
    szcore_candidate_roots: Sequence[str | Path],
) -> dict[str, Any]:
    """Build the real roster-level graph without opening any target/reference."""

    validated_plan = validate_full_stack_nested_exposure_plan_v1(plan)
    validated_tusz = validate_tusz_detector_cleanroom_fold_plan_v1(tusz_fold_plan)
    physical = validate_tusz_canonical_physical_analysis_projection_v1(
        tusz_physical_projection
    )
    deepsoz = deepcopy(dict(deepsoz_identity_binding))
    validate_deepsoz_tusz_source_train_identity_binding_v1(deepsoz)
    deepsoz_union = _validate_deepsoz_union_manifest(deepsoz_public_union_manifest)
    external = validate_deepsoz_published_external_exposure_attestation_v1(
        deepsoz_external_attestation
    )
    if validated_tusz["plan_id"] != validated_plan["population_authority"][
        "outer_plan_id"
    ]:
        raise ValueError("bound TUSZ outer plan ID drifted")
    if physical["projection_id"] != validated_tusz["source_binding"][
        "source_canonical_physical_projection_id"
    ] or physical["receipt_sha256"] != validated_tusz["source_binding"][
        "source_canonical_physical_projection_receipt_sha256"
    ]:
        raise ValueError("TUSZ physical projection disagrees with outer plan")

    path_roles = {
        "tusz_fold_plan",
        "tusz_physical_projection",
        "deepsoz_identity_binding",
        "deepsoz_public_union_manifest",
        "deepsoz_external_attestation",
    }
    if set(source_paths) != path_roles:
        raise ValueError("full-stack source path roles drifted")
    source_files = {
        role: _regular_file(path, role) for role, path in source_paths.items()
    }
    source_bindings = [
        _source_binding(
            role="tusz_fold_plan",
            path=source_files["tusz_fold_plan"],
            schema_version=validated_tusz["schema_version"],
            semantic_id=validated_tusz["plan_id"],
            receipt=validated_tusz["receipt_sha256"],
        ),
        _source_binding(
            role="tusz_physical_projection",
            path=source_files["tusz_physical_projection"],
            schema_version=physical["schema_version"],
            semantic_id=physical["projection_id"],
            receipt=physical["receipt_sha256"],
        ),
        _source_binding(
            role="deepsoz_identity_binding",
            path=source_files["deepsoz_identity_binding"],
            schema_version=deepsoz["schema_version"],
            semantic_id=deepsoz["binding_id"],
            receipt=deepsoz["receipt_sha256"],
        ),
        _source_binding(
            role="deepsoz_public_union_manifest",
            path=source_files["deepsoz_public_union_manifest"],
            schema_version=deepsoz_union["schema_version"],
            semantic_id="DEEPSOZ-PUBLIC-DEVELOPMENT-UNION-IDENTITY-V12",
            receipt=deepsoz_union["manifest_payload_sha256"],
        ),
        _source_binding(
            role="deepsoz_external_attestation",
            path=source_files["deepsoz_external_attestation"],
            schema_version=external["schema_version"],
            semantic_id=external["attestation_id"],
            receipt=external["receipt_sha256"],
        ),
    ]

    groups = _partition_groups(validated_tusz)
    group_ids = [row["group_id"] for row in groups]
    group_patient_lookup = {
        row["group_id"]: set(row["patient_ids"]) for row in groups
    }
    source_train_patients = set().union(*group_patient_lookup.values())
    if len(source_train_patients) != 579:
        raise ValueError("TUSZ partition groups do not cover 579 unique patients")

    physical_by_path = {row["local_edf_path"]: row for row in physical["records"]}
    for row in deepsoz["records"]:
        physical_row = physical_by_path.get(row["tusz_recording_id"])
        if physical_row is None:
            raise ValueError("DeepSOZ identity record is absent from physical projection")
        if (
            physical_row["local_patient_id"] != row["local_patient_id"]
            or physical_row["source_edf_container_sha256"]
            != row["source_container_sha256"]
        ):
            raise ValueError("DeepSOZ identity record crosses TUSZ physical identity")
    deepsoz_mapping = []
    for row in deepsoz["patients"]:
        local_id = row["local_patient_id"]
        memberships = [
            group_id
            for group_id, patients in group_patient_lookup.items()
            if local_id in patients
        ]
        if len(memberships) != 1:
            raise ValueError("DeepSOZ source-train patient has no unique outer group")
        deepsoz_mapping.append(
            {
                "deepsoz_patient_id": row["deepsoz_patient_id"],
                "local_patient_id": local_id,
                "outer_group_id": memberships[0],
                "source_record_count": row["source_record_count"],
            }
        )
    deepsoz_mapping.sort(key=lambda row: int(row["deepsoz_patient_id"]))
    deepsoz_local_patients = {row["local_patient_id"] for row in deepsoz_mapping}

    tuev = _scan_tuev_eeg_identity(tuev_edf_root)
    tuev_readme = _regular_file(tuev_readme_path, "TUEV README")
    tusz_patients_by_split = {
        split: {
            row["local_patient_id"]
            for row in physical["records"]
            if row["model_split"] == split
        }
        for split in ("source_train", "source_dev", "source_eval")
    }
    tuev_train_patients = set(tuev["train_patient_ids"])
    tuev_patient_overlap = {
        split: sorted(tuev_train_patients & patients)
        for split, patients in tusz_patients_by_split.items()
    }
    tusz_hash_split: dict[str, set[str]] = {
        split: {
            row["source_edf_container_sha256"]
            for row in physical["records"]
            if row["model_split"] == split
        }
        for split in ("source_train", "source_dev", "source_eval")
    }
    tuev_hashes = {row["container_sha256"] for row in tuev["records"]}
    tuev_exact_overlap = {
        split: sorted(tuev_hashes & hashes)
        for split, hashes in tusz_hash_split.items()
    }
    tuev_summary = {key: value for key, value in tuev.items() if key != "records"}
    tuev_summary.update(
        {
            "readme_path": str(tuev_readme),
            "readme_sha256": _file_sha256(tuev_readme),
            "tusz_visible_patient_overlap_by_split": tuev_patient_overlap,
            "tusz_exact_container_overlap_by_split": tuev_exact_overlap,
            "decoded_sample_near_or_partial_overlap_audit_complete": False,
            "REC_or_other_label_files_opened": False,
        }
    )
    for group in groups:
        group["deepsoz_source_train_overlay_patient_ids"] = sorted(
            set(group["patient_ids"]) & deepsoz_local_patients
        )
        group["tuev_train_visible_overlap_patient_ids"] = sorted(
            set(group["patient_ids"]) & tuev_train_patients
        )

    tuar_root = _regular_directory(tuar_metadata_root, "TUAR metadata root")
    tuar_audit = _regular_file(tuar_audit_receipt_path, "TUAR metadata audit")
    if tuar_audit.parent != tuar_root:
        raise ValueError("TUAR audit receipt is outside its metadata root")
    tuar_edf_count = sum(
        1 for path in tuar_root.rglob("*") if path.is_file() and path.suffix.lower() == ".edf"
    )
    if tuar_edf_count != 0:
        raise ValueError("TUAR metadata-only root unexpectedly contains EDF")
    szcore_rows = []
    for candidate in szcore_candidate_roots:
        path = Path(candidate).absolute()
        szcore_rows.append({"path": str(path), "exists": path.exists()})
    if not szcore_rows or any(row["exists"] for row in szcore_rows):
        raise ValueError("SzCORE candidate root state requires a new admitted manifest")

    dataset_registry = [
        {
            "dataset_id": "TUSZ",
            "role": "primary_long_EEG_and_outer_patient_partition",
            "manifest_status": "canonical_physical_identity_and_outer_plan_materialized",
            "patient_identity_status": "materialized",
            "content_identity_status": "exact_decoded_physical_equivalence_materialized",
            "patient_count": 579,
            "record_count": 4664,
            "near_or_partial_overlap_status": "not_materialized",
            "actual_training_exposure_status": "none_plan_only",
            "training_authorized_by_registry": False,
        },
        {
            "dataset_id": "SzCORE",
            "role": "future_event_interval_external_dataset",
            "manifest_status": "no_local_admitted_manifest_or_signal_root",
            "patient_identity_status": "missing",
            "content_identity_status": "missing",
            "checked_candidate_roots": szcore_rows,
            "near_or_partial_overlap_status": "missing",
            "actual_training_exposure_status": "empty_fail_closed",
            "training_authorized_by_registry": False,
        },
        {
            "dataset_id": "DeepSOZ",
            "role": "TUSZ_overlay_incomplete_positive_weak_spatial_label_candidate",
            "manifest_status": "target_free_source_train_identity_bridge_materialized",
            "patient_identity_status": "70_patients_exactly_bound_to_TUSZ_source_train",
            "content_identity_status": "318_exact_TUSZ_EDF_containers_bound",
            "source_train_patient_count": 70,
            "source_train_record_count": 318,
            "source_train_patient_group_mapping": deepsoz_mapping,
            "public_union_audit": {
                "patients": deepsoz_union["patient_count"],
                "events": deepsoz_union["event_count"],
                "status": "historical_development_union_not_admitted_to_new_outer_plan",
            },
            "published_weight_audit": {
                "folds": external["counts"]["published_folds"],
                "clean_room_verified": external["evidence_gates"][
                    "clean_room_verified"
                ],
                "strict_g0a_verified": external["evidence_gates"][
                    "strict_g0a_verified"
                ],
                "status": "descriptive_control_not_full_stack_exposure_authority",
            },
            "weak_label_values_opened": False,
            "same_patient_training_label_and_outer_test_GT_allowed": False,
            "actual_training_exposure_status": "empty_targets_unopened",
            "training_authorized_by_registry": False,
        },
        {
            "dataset_id": "TUEV",
            "role": "future_morphology_or_QC_auxiliary_only",
            "manifest_status": "EEG_byte_identity_snapshot_materialized_clinical_label_manifest_absent",
            "patient_identity_status": "train_visible_eval_opaque",
            "content_identity_status": "exact_EDF_container_only",
            "identity_snapshot": tuev_summary,
            "near_or_partial_overlap_status": "not_materialized",
            "actual_training_exposure_status": "empty_fail_closed",
            "training_authorized_by_registry": False,
        },
        {
            "dataset_id": "TUAR",
            "role": "future_QC_artifact_auxiliary_only",
            "manifest_status": "metadata_only_no_EEG_or_label_manifest",
            "patient_identity_status": "published_counts_only_not_materialized",
            "content_identity_status": "missing_no_EDF",
            "metadata_root": str(tuar_root),
            "metadata_audit_path": str(tuar_audit),
            "metadata_audit_sha256": _file_sha256(tuar_audit),
            "local_EDF_count": tuar_edf_count,
            "near_or_partial_overlap_status": "missing",
            "actual_training_exposure_status": "empty_fail_closed",
            "training_authorized_by_registry": False,
        },
    ]

    outer_folds = [
        _outer_fold_graph(
            outer_fold_id=index,
            group_ids=group_ids,
            group_patient_lookup=group_patient_lookup,
            deepsoz_local_patients=deepsoz_local_patients,
        )
        for index in range(5)
    ]
    body: dict[str, Any] = {
        "schema_version": FULL_STACK_NESTED_EXPOSURE_REGISTRY_V1_SCHEMA,
        "registry_id": "FULLSTACKEXPREGV1-PENDING",
        "status": "patient_rosters_and_nested_schedule_materialized_artifacts_untrained",
        "plan_binding": {
            "plan_id": validated_plan["plan_id"],
            "plan_sha256": validated_plan["plan_sha256"],
        },
        "source_bindings": source_bindings,
        "dataset_registry": dataset_registry,
        "partition_groups": groups,
        "outer_folds": outer_folds,
        "artifact_inventory": {
            "detector_arms": [
                {
                    "provider_id": arm,
                    "outer_checkpoint_slot_count": 5,
                    "materialized_checkpoint_count": 0,
                    "detector_only_OOF_inventory_exists": False,
                }
                for arm in _DETECTOR_ARMS
            ],
            "preprocessing_artifact_count": 0,
            "SSL_artifact_count": 0,
            "core_checkpoint_count": 0,
            "ranker_checkpoint_count": 0,
            "router_target_inventory_count": 0,
            "router_checkpoint_count": 0,
            "calibrator_checkpoint_count": 0,
            "prediction_first_outer_test_inventory_count": 0,
            "five_detector_checkpoints_per_arm_equal_full_stack_OOF": False,
            "legal_full_stack_OOF_inventory_exists": False,
        },
        "external_pretrained_artifact_registry": [
            {
                "artifact_family": family,
                "status": "not_admitted_training_exposure_unverified",
                "may_reach_outer_stack": False,
            }
            for family in ("TFM", "TimeFilter", "CBraMod", "LaBraM")
        ],
        "data_access_receipt": {
            "TUSZ_source_eval_reference_opened": False,
            "TUSZ_source_dev_reference_opened": False,
            "DeepSOZ_channel_target_values_opened": False,
            "TUEV_REC_or_label_files_opened": False,
            "TUAR_EEG_or_label_files_opened": False,
            "SzCORE_reference_opened": False,
            "EDF_annotations_opened": False,
            "private_data_opened": False,
            "clinical_text_or_spreadsheet_opened": False,
            "TUEV_raw_EEG_bytes_opened_for_identity_hash_only": True,
            "model_training_executed": False,
            "model_inference_executed": False,
        },
        "known_missing_exposure_closures": [
            "TUSZ_and_TUEV_reencoded_decoded_sample_near_duplicate_crop_shift_resample_and_partial_overlap_audit",
            "TUEV_eval_session_to_patient_identity_authority",
            "TUEV_authorized_clinical_label_manifest_and_fold_scoped_exposure_receipts",
            "TUAR_EEG_label_manifest_patient_and_content_identity",
            "SzCORE_signal_reference_manifest_patient_and_content_identity",
            "fold_scoped_TUSZ_detection_reference_authority",
            "actual_fold_scoped_preprocessing_SSL_detector_core_ranker_router_and_calibrator_receipts",
            "inner_OOF_candidate_feature_router_target_and_calibrator_prediction_inventories",
            "external_pretrained_TFM_TimeFilter_CBraMod_and_LaBraM_training_exposure_attestations",
        ],
        "scientific_permissions": {
            "detector_training_authorized": False,
            "full_stack_training_authorized": False,
            "source_eval_reference_open_authorized": False,
            "claim_detector_OOF_as_full_stack_OOF": False,
            "claim_full_stack_OOF_performance": False,
            "claim_medical_or_clinical_performance": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    id_body = deepcopy(body)
    id_body["registry_id"] = "FULLSTACKEXPREGV1-PENDING"
    id_body["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    body["registry_id"] = "FULLSTACKEXPREGV1-" + _canonical_sha256(id_body)[:24]
    receipt_body = deepcopy(body)
    receipt_body["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    body["receipt_sha256"] = _canonical_sha256(receipt_body)
    return validate_full_stack_nested_exposure_registry_v1(body)


def _reachable_ancestors(nodes: Mapping[str, Mapping[str, Any]], root: str) -> set[str]:
    active: set[str] = set()
    visiting: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("full-stack exposure graph contains a cycle")
        if node_id in active:
            return
        if node_id not in nodes:
            raise ValueError("full-stack exposure graph has a missing dependency")
        visiting.add(node_id)
        for dependency in nodes[node_id]["dependency_node_ids"]:
            visit(str(dependency))
        visiting.remove(node_id)
        active.add(node_id)

    visit(root)
    return active


def validate_full_stack_nested_exposure_registry_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay receipts and enforce patient-unreachability for every stage."""

    required = {
        "schema_version",
        "registry_id",
        "status",
        "plan_binding",
        "source_bindings",
        "dataset_registry",
        "partition_groups",
        "outer_folds",
        "artifact_inventory",
        "external_pretrained_artifact_registry",
        "data_access_receipt",
        "known_missing_exposure_closures",
        "scientific_permissions",
        "receipt_sha256",
    }
    registry = _strict_object(value, required, "full-stack exposure registry")
    if registry["schema_version"] != FULL_STACK_NESTED_EXPOSURE_REGISTRY_V1_SCHEMA:
        raise ValueError("full-stack exposure registry schema drifted")
    if registry["status"] != (
        "patient_rosters_and_nested_schedule_materialized_artifacts_untrained"
    ):
        raise ValueError("full-stack exposure registry status drifted")
    receipt = _sha256(registry["receipt_sha256"], "registry receipt")
    receipt_body = deepcopy(registry)
    receipt_body["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if _canonical_sha256(receipt_body) != receipt:
        raise ValueError("full-stack exposure registry receipt does not replay")
    id_body = deepcopy(registry)
    id_body["registry_id"] = "FULLSTACKEXPREGV1-PENDING"
    id_body["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if registry["registry_id"] != "FULLSTACKEXPREGV1-" + _canonical_sha256(
        id_body
    )[:24]:
        raise ValueError("full-stack exposure registry ID drifted")
    plan = load_full_stack_nested_exposure_plan_v1()
    if registry["plan_binding"] != {
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
    }:
        raise ValueError("full-stack registry plan binding drifted")

    bindings = registry["source_bindings"]
    expected_roles = [
        "tusz_fold_plan",
        "tusz_physical_projection",
        "deepsoz_identity_binding",
        "deepsoz_public_union_manifest",
        "deepsoz_external_attestation",
    ]
    if not isinstance(bindings, list) or [row.get("role") for row in bindings] != expected_roles:
        raise ValueError("full-stack source binding roster drifted")
    for row in bindings:
        if set(row) != {
            "role",
            "path",
            "file_sha256",
            "schema_version",
            "semantic_id",
            "semantic_receipt_sha256",
        }:
            raise ValueError("full-stack source binding fields drifted")
        _sha256(row["file_sha256"], "source file SHA-256")
        _sha256(row["semantic_receipt_sha256"], "source semantic receipt")

    datasets = registry["dataset_registry"]
    if not isinstance(datasets, list) or [row.get("dataset_id") for row in datasets] != list(
        _REQUIRED_DATASETS
    ):
        raise ValueError("materialized dataset registry roster drifted")
    by_dataset = {row["dataset_id"]: row for row in datasets}
    if by_dataset["TUSZ"]["patient_count"] != 579 or by_dataset["TUSZ"][
        "record_count"
    ] != 4664:
        raise ValueError("materialized TUSZ denominator drifted")
    if by_dataset["DeepSOZ"]["source_train_patient_count"] != 70 or by_dataset[
        "DeepSOZ"
    ]["source_train_record_count"] != 318:
        raise ValueError("materialized DeepSOZ source-train bridge drifted")
    if by_dataset["DeepSOZ"]["weak_label_values_opened"] is not False:
        raise PermissionError("DeepSOZ weak-label values were prematurely opened")
    for dataset_id in ("SzCORE", "TUEV", "TUAR"):
        if by_dataset[dataset_id]["training_authorized_by_registry"] is not False:
            raise PermissionError(f"unclosed external dataset was authorized: {dataset_id}")
        if by_dataset[dataset_id]["actual_training_exposure_status"] != (
            "empty_fail_closed"
        ):
            raise PermissionError(f"unclosed external exposure is not empty: {dataset_id}")
    tuev_snapshot = by_dataset["TUEV"]["identity_snapshot"]
    if (
        tuev_snapshot["REC_or_other_label_files_opened"] is not False
        or tuev_snapshot["decoded_sample_near_or_partial_overlap_audit_complete"]
        is not False
    ):
        raise PermissionError("TUEV identity-only boundary drifted")
    if by_dataset["TUAR"]["local_EDF_count"] != 0:
        raise PermissionError("TUAR metadata-only boundary drifted")

    groups = registry["partition_groups"]
    if not isinstance(groups, list) or [row.get("group_id") for row in groups] != [
        "G0",
        "G1",
        "G2",
        "G3",
        "G4",
    ]:
        raise ValueError("outer partition group roster drifted")
    group_patients: dict[str, set[str]] = {}
    for row in groups:
        patients = list(row["patient_ids"])
        if patients != sorted(set(patients)) or len(patients) != row["patient_count"]:
            raise ValueError("outer partition patient roster drifted")
        if _canonical_sha256(patients) != row["patient_roster_sha256"]:
            raise ValueError("outer partition patient receipt drifted")
        group_patients[row["group_id"]] = set(patients)
    for left_index, left in enumerate(groups):
        for right in groups[left_index + 1 :]:
            if group_patients[left["group_id"]] & group_patients[right["group_id"]]:
                raise ValueError("outer partition patient groups overlap")
    union_patients = set().union(*group_patients.values())
    if len(union_patients) != 579 or sum(
        row["analysis_identity_count"] for row in groups
    ) != 4664:
        raise ValueError("outer partition union denominator drifted")

    deepsoz_mapping = by_dataset["DeepSOZ"]["source_train_patient_group_mapping"]
    if not isinstance(deepsoz_mapping, list) or len(deepsoz_mapping) != 70:
        raise ValueError("DeepSOZ patient-group mapping drifted")
    deepsoz_by_group: dict[str, set[str]] = {group: set() for group in group_patients}
    for row in deepsoz_mapping:
        local = row["local_patient_id"]
        group = row["outer_group_id"]
        if group not in group_patients or local not in group_patients[group]:
            raise ValueError("DeepSOZ patient-group mapping crosses patients")
        deepsoz_by_group[group].add(local)
    if len(set().union(*deepsoz_by_group.values())) != 70:
        raise ValueError("DeepSOZ patient-group mapping repeats patients")

    folds = registry["outer_folds"]
    if not isinstance(folds, list) or [row.get("outer_fold_id") for row in folds] != list(
        range(5)
    ):
        raise ValueError("outer full-stack fold roster drifted")
    all_groups = list(group_patients)
    for fold in folds:
        fold_id = fold["outer_fold_id"]
        test_group = f"G{fold_id}"
        outer_train = [group for group in all_groups if group != test_group]
        if fold["outer_test_group_id"] != test_group or fold[
            "outer_train_group_ids"
        ] != outer_train:
            raise ValueError("outer full-stack group assignment drifted")
        if fold["inner_feature_row_crossfit"] != _crossfit_rows(outer_train):
            raise ValueError("inner detector feature-row crossfit drifted")
        if fold["router_counterfactual_target_crossfit"] != _router_target_rows(
            outer_train
        ):
            raise ValueError("router target nested crossfit drifted")
        if fold["calibrator_prediction_crossfit"] != _crossfit_rows(outer_train):
            raise ValueError("calibrator prediction crossfit drifted")

        test_patients = group_patients[test_group]
        nodes_list = fold["nodes"]
        if not isinstance(nodes_list, list):
            raise TypeError("outer-fold nodes must be a list")
        nodes = {row["node_id"]: row for row in nodes_list}
        if len(nodes) != len(nodes_list):
            raise ValueError("outer-fold graph repeats a node")
        reachable = _reachable_ancestors(nodes, fold["outer_inference_node_id"])
        reachable_stages = {nodes[node]["stage"] for node in reachable}
        if not set(_REQUIRED_EXPOSURE_STAGES).issubset(reachable_stages):
            raise ValueError("outer inference does not bind every exposure stage")
        for node in nodes_list:
            expected_node_fields = {
                "node_id",
                "stage",
                "dependency_node_ids",
                "tusz_fit_group_ids",
                "tusz_inference_group_ids",
                "external_dataset_ids",
                "external_candidate_local_patient_ids",
                "external_exposure_local_patient_ids",
                "materialization_status",
            }
            if set(node) != expected_node_fields:
                raise ValueError("outer-fold exposure node fields drifted")
            fit_groups = node["tusz_fit_group_ids"]
            if len(fit_groups) != len(set(fit_groups)) or any(
                group not in outer_train for group in fit_groups
            ):
                raise PermissionError(
                    f"outer-test patient can reach {node['stage']} fit exposure"
                )
            fit_patients = set().union(
                *(group_patients[group] for group in fit_groups), set()
            )
            if fit_patients & test_patients:
                raise PermissionError(
                    f"outer-test patient can reach {node['stage']} patient exposure"
                )
            external_candidates = set(node["external_candidate_local_patient_ids"])
            external_exposure = set(node["external_exposure_local_patient_ids"])
            if external_candidates & test_patients or external_exposure & test_patients:
                raise PermissionError(
                    f"outer-test patient can reach {node['stage']} external exposure"
                )
            if not external_exposure.issubset(external_candidates):
                raise PermissionError("actual external exposure exceeds candidate roster")
            if node["stage"] == "weak_label":
                expected_candidates = set().union(
                    *(deepsoz_by_group[group] for group in outer_train), set()
                )
                if external_candidates != expected_candidates or external_exposure:
                    raise PermissionError("DeepSOZ weak-label gate drifted")
            if node["stage"] != "outer_test_inference" and node[
                "tusz_inference_group_ids"
            ]:
                raise PermissionError("fit-stage node gained outer inference authority")
        inference_node = nodes[fold["outer_inference_node_id"]]
        if inference_node["tusz_fit_group_ids"] or inference_node[
            "tusz_inference_group_ids"
        ] != [test_group]:
            raise PermissionError("outer inference node exposure semantics drifted")

    inventory = registry["artifact_inventory"]
    if [row["provider_id"] for row in inventory["detector_arms"]] != list(
        _DETECTOR_ARMS
    ):
        raise ValueError("detector arm inventory drifted")
    for arm in inventory["detector_arms"]:
        if arm["outer_checkpoint_slot_count"] != 5:
            raise ValueError("detector checkpoint slot count drifted")
        if arm["materialized_checkpoint_count"] != 0 or arm[
            "detector_only_OOF_inventory_exists"
        ] is not False:
            raise PermissionError("untrained detector artifacts were overstated")
    if inventory["five_detector_checkpoints_per_arm_equal_full_stack_OOF"] is not False:
        raise PermissionError("five detector checkpoints became full-stack OOF")
    if inventory["legal_full_stack_OOF_inventory_exists"] is not False:
        raise PermissionError("full-stack OOF inventory was prematurely opened")
    for family in registry["external_pretrained_artifact_registry"]:
        if family["status"] != "not_admitted_training_exposure_unverified" or family[
            "may_reach_outer_stack"
        ] is not False:
            raise PermissionError("unverified external SSL artifact can reach outer stack")
    for field, observed in registry["data_access_receipt"].items():
        if field == "TUEV_raw_EEG_bytes_opened_for_identity_hash_only":
            if observed is not True:
                raise ValueError("TUEV identity hash access receipt drifted")
        elif observed is not False:
            raise PermissionError(f"forbidden data access opened: {field}")
    for field, observed in registry["scientific_permissions"].items():
        if observed is not False:
            raise PermissionError(f"premature scientific permission opened: {field}")
    return registry


def build_full_stack_nested_exposure_registry_from_paths_v1(
    *,
    plan_path: str | Path,
    tusz_fold_plan_path: str | Path,
    tusz_physical_projection_path: str | Path,
    deepsoz_identity_binding_path: str | Path,
    deepsoz_public_union_manifest_path: str | Path,
    deepsoz_external_attestation_path: str | Path,
    tuev_edf_root: str | Path,
    tuev_readme_path: str | Path,
    tuar_metadata_root: str | Path,
    tuar_audit_receipt_path: str | Path,
    szcore_candidate_roots: Sequence[str | Path],
) -> dict[str, Any]:
    plan, plan_source = _load_json_file(plan_path, "full-stack exposure plan")
    # The plan source is loaded only to enforce a canonical trusted payload;
    # registry source_bindings intentionally start at the empirical manifests.
    _ = plan_source
    tusz_plan, tusz_plan_source = _load_json_file(
        tusz_fold_plan_path, "TUSZ fold plan"
    )
    physical, physical_source = _load_json_file(
        tusz_physical_projection_path, "TUSZ physical projection"
    )
    deepsoz, deepsoz_source = _load_json_file(
        deepsoz_identity_binding_path, "DeepSOZ identity binding"
    )
    union, union_source = _load_json_file(
        deepsoz_public_union_manifest_path, "DeepSOZ public union"
    )
    external, external_source = _load_json_file(
        deepsoz_external_attestation_path, "DeepSOZ external attestation"
    )
    return build_full_stack_nested_exposure_registry_v1(
        plan=plan,
        tusz_fold_plan=tusz_plan,
        tusz_physical_projection=physical,
        deepsoz_identity_binding=deepsoz,
        deepsoz_public_union_manifest=union,
        deepsoz_external_attestation=external,
        source_paths={
            "tusz_fold_plan": tusz_plan_source,
            "tusz_physical_projection": physical_source,
            "deepsoz_identity_binding": deepsoz_source,
            "deepsoz_public_union_manifest": union_source,
            "deepsoz_external_attestation": external_source,
        },
        tuev_edf_root=tuev_edf_root,
        tuev_readme_path=tuev_readme_path,
        tuar_metadata_root=tuar_metadata_root,
        tuar_audit_receipt_path=tuar_audit_receipt_path,
        szcore_candidate_roots=szcore_candidate_roots,
    )


def replay_full_stack_nested_exposure_registry_from_paths_v1(
    registry: Mapping[str, Any],
    **source_arguments: Any,
) -> dict[str, Any]:
    validated = validate_full_stack_nested_exposure_registry_v1(registry)
    rebuilt = build_full_stack_nested_exposure_registry_from_paths_v1(
        **source_arguments
    )
    if rebuilt != validated:
        raise ValueError("full-stack registry does not replay from bound sources")
    return validated


def materialize_full_stack_nested_exposure_registry_v1(
    registry: Mapping[str, Any], output_path: str | Path
) -> Path:
    validated = validate_full_stack_nested_exposure_registry_v1(registry)
    destination = Path(output_path).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"append-only exposure registry exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(validated, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"append-only exposure registry appeared: {destination}")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return destination


def load_full_stack_nested_exposure_registry_v1(
    path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    payload, _ = _load_json_file(path, "full-stack exposure registry")
    return validate_full_stack_nested_exposure_registry_v1(payload)


__all__ = [
    "DEFAULT_PLAN_PATH",
    "DEFAULT_PLAN_SHA256",
    "DEFAULT_REGISTRY_PATH",
    "FULL_STACK_NESTED_EXPOSURE_PLAN_V1_ID",
    "FULL_STACK_NESTED_EXPOSURE_PLAN_V1_SCHEMA",
    "FULL_STACK_NESTED_EXPOSURE_REGISTRY_V1_SCHEMA",
    "build_full_stack_nested_exposure_registry_from_paths_v1",
    "build_full_stack_nested_exposure_registry_v1",
    "load_full_stack_nested_exposure_plan_v1",
    "load_full_stack_nested_exposure_registry_v1",
    "materialize_full_stack_nested_exposure_registry_v1",
    "replay_full_stack_nested_exposure_registry_from_paths_v1",
    "validate_full_stack_nested_exposure_plan_v1",
    "validate_full_stack_nested_exposure_registry_v1",
]
