"""Machine-readable execution plan for the EviSOZ-LM research route.

The plan is deliberately separate from the aggregate Stage-0 gate.  It does
not authorize anything: it converts the current gate into an auditable list
of runnable/blocked stages and frozen experiment controls.  In particular,
all training and Qwen stages remain fail-closed while Stage 0 is ``NO_GO``.
The output contains only schema/status/action metadata and never copies
private patient identifiers or physician report text.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from src.evisoz.data.artifact_ref import canonical_json_sha256
from src.evisoz.data.stage0_gate import validate_stage0_gate
from src.evisoz.models.clinical_evidence import (
    validate_structured_evidence_pipeline_config,
)


EXECUTION_PLAN_SCHEMA_VERSION = "evisoz_execution_plan_v1"
_HASH_PLACEHOLDER = "0" * 64
_ID_PREFIX = "EVISOZ-PLAN-"

_STAGE_DEFINITIONS = (
    (
        "stage0_real_data",
        "Stage-0 real data and baseline closure",
        (
            "schema_registry",
            "public_v29_reference",
            "private_real_dual_montage",
            "private_field_envelopes",
            "private_report_linkage",
            "private_report_text_release",
            "knowledge_authority",
            "public_auxiliary_patient_exposure_ledger",
            "offline_teacher_and_derived_candidates",
            "bound_evidence_materialization",
            "findings_claim_graph_and_reports",
            "clean_freeze_audit",
        ),
    ),
    (
        "stage1_evidence_representation",
        "Clinical motif and sparse temporal-spatial evidence representation",
        (
            "offline_teacher_and_derived_candidates",
            "private_field_envelopes",
        ),
    ),
    (
        "stage2_soz_localization",
        "Residual SOZ localization with frozen canonical v29 reference",
        (
            "private_field_envelopes",
            "public_auxiliary_patient_exposure_ledger",
        ),
    ),
    (
        "stage3_qwen_text",
        "Evidence/claim graph to Qwen text-side adaptation",
        (
            "private_report_linkage",
            "private_report_text_release",
            "knowledge_authority",
        ),
    ),
    (
        "stage4_eeg_to_qwen",
        "EEG-to-Qwen connector and clause-level alignment",
        (
            "private_report_text_release",
            "offline_teacher_and_derived_candidates",
        ),
    ),
    (
        "stage5_joint_evaluation",
        "Patient-disjoint localization, report, calibration, and feedback evaluation",
        (
            "private_report_text_release",
            "public_auxiliary_patient_exposure_ledger",
        ),
    ),
)

_EXPERIMENTS = (
    ("A", "frozen canonical v29 H/D baseline", "baseline"),
    ("B", "A + Clinical Motif Adapter", "evidence"),
    ("C", "B + Evidence Query Decoder", "evidence"),
    ("D", "C + CerebraGloss candidate pretraining", "teacher"),
    ("E", "C + ELM semantic pretraining", "teacher"),
    ("F", "C + deterministic/teacher evidence fusion", "evidence"),
    ("G", "F + structured report generation", "report"),
    ("H", "G + knowledge/eeg constraints", "report"),
    ("I", "H + evidence-guided masking", "feedback"),
    ("J", "I + one-shot grounded report feedback", "feedback"),
)

_REPORT_CONTROLS = (
    "correct_report",
    "patient_shuffled_report",
    "left_right_swapped_report",
    "onset_spread_swapped_report",
    "top1_only_report",
    "without_knowledge_base",
)


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    body = _hash_source(value)
    body["plan_id"] = "CONTENT-ADDRESS-PENDING"
    return body


def _stage_status(
    *,
    plan_stage: str,
    gate_status: str,
    check_statuses: Mapping[str, str],
    required_checks: tuple[str, ...],
) -> tuple[str, list[str]]:
    # ``QUALIFIED_GO`` is a passing Stage-0 status for artifacts that have
    # completed their contract/replay checks but carry a documented scope
    # limitation (for example, the private dual-montage cache or the
    # findings/report shadow).  It must not be confused with evaluator-only,
    # partial, or failed checks.  The aggregate gate already uses this same
    # distinction when building ``blocking_check_ids``.
    passing_statuses = {"GO", "QUALIFIED_GO"}
    missing = sorted(
        check
        for check in required_checks
        if check_statuses.get(check) not in passing_statuses
    )
    if plan_stage == "stage0_real_data":
        return ("GO" if not missing and gate_status == "GO" else "NO_GO", missing)
    if gate_status != "GO":
        return ("BLOCKED_BY_STAGE0", missing or ["stage0_overall"])
    # The current repository contains contract/shadow implementations for
    # downstream stages, not an authorized training run.  Keep that
    # distinction explicit even if a future gate becomes GO.
    return ("READY_FOR_AUTHORIZED_RUN", missing)


def build_evisoz_execution_plan(
    gate: Mapping[str, object],
    *,
    pipeline_config: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build a content-addressed, fail-closed EviSOZ execution plan."""

    validated_gate = validate_stage0_gate(dict(gate))
    validated_config = None
    if pipeline_config is not None:
        validated_config = validate_structured_evidence_pipeline_config(
            dict(pipeline_config)
        )
    check_statuses = {
        str(row["check_id"]): str(row["status"])
        for row in validated_gate["checks"]
    }
    stage_rows: list[dict[str, Any]] = []
    for stage_id, title, required_checks in _STAGE_DEFINITIONS:
        status, missing = _stage_status(
            plan_stage=stage_id,
            gate_status=str(validated_gate["status"]),
            check_statuses=check_statuses,
            required_checks=required_checks,
        )
        stage_rows.append(
            {
                "stage_id": stage_id,
                "title": title,
                "status": status,
                "required_stage0_checks": list(required_checks),
                "missing_or_non_go_checks": missing,
                "formal_training_allowed": status == "READY_FOR_AUTHORIZED_RUN"
                and str(validated_gate["status"]) == "GO",
            }
        )

    experiment_rows = [
        {
            "id": experiment_id,
            "description": description,
            "family": family,
            "status": "BLOCKED_BY_STAGE0"
            if validated_gate["status"] != "GO"
            else "PLANNED_NOT_RUN",
            "uses_private_training_or_report_eval": experiment_id not in {"A", "B", "C"},
        }
        for experiment_id, description, family in _EXPERIMENTS
    ]
    body: dict[str, Any] = {
        "schema_version": EXECUTION_PLAN_SCHEMA_VERSION,
        "plan_id": _HASH_PLACEHOLDER,
        "status": "STAGE0_NO_GO" if validated_gate["status"] != "GO" else "STAGE0_GO_DOWNSTREAM_NOT_RUN",
        "stage0_gate_id": validated_gate["gate_id"],
        "stage0_gate_receipt_sha256": validated_gate["receipt_sha256"],
        "baseline": {
            "model": "canonical_v29_H_D",
            "route": "standard19_car_exact_reference",
            "frozen": True,
            "tcp22_role": "signed_bipolar_edge_evidence_only",
            "missing_channel_policy": "explicit_mask_only",
            "interpolation": "independent_ablation_only",
        },
        "stage0_check_statuses": dict(sorted(check_statuses.items())),
        "blocking_check_ids": list(validated_gate["blocking_check_ids"]),
        "authorized_next_actions": list(validated_gate["authorized_next_actions"]),
        "prohibited_actions": list(validated_gate["prohibited_actions"]),
        "stages": stage_rows,
        "experiments": experiment_rows,
        "report_controls": [
            {
                "control_id": control,
                "status": "BLOCKED_BY_STAGE0"
                if validated_gate["status"] != "GO"
                else "PLANNED_NOT_RUN",
            }
            for control in _REPORT_CONTROLS
        ],
        "entrypoints": {
            "stage0_gate": "scripts/materialize_evisoz_stage0_gate_v1.py",
            "real_cohort_validation": "scripts/validate_private_evisoz_real_stage0_cohort_v1.py",
            "loader_replay": "scripts/replay_evisoz_bound_evidence_loader_v1.py",
            "real_labram_shadow": "scripts/run_evisoz_real_labram_shadow_v1.py",
            "real_shadow_inference": "scripts/run_evisoz_real_shadow_inference_v1.py",
            "structured_smoke": "scripts/smoke_evisoz_structured_evidence_pipeline_v1.py",
            "patient_qwen_shadow": "scripts/materialize_evisoz_qwen_patient_shadow_v1.py",
        },
        "permissions": {
            "training_authorized": validated_gate["status"] == "GO",
            "qwen_generation_authorized": False,
            "private_report_language_evaluation_authorized": False,
            "clinical_deployment_authorized": False,
            "knowledge_creates_patient_facts": False,
        },
        "pipeline_config": (
            {
                "schema_version": validated_config["schema_version"],
                "status": validated_config["status"],
            }
            if validated_config is not None
            else None
        ),
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["plan_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_evisoz_execution_plan(body)


def validate_evisoz_execution_plan(value: object) -> dict[str, Any]:
    """Validate an execution plan and its fail-closed status invariants."""

    required = {
        "schema_version",
        "plan_id",
        "status",
        "stage0_gate_id",
        "stage0_gate_receipt_sha256",
        "baseline",
        "stage0_check_statuses",
        "blocking_check_ids",
        "authorized_next_actions",
        "prohibited_actions",
        "stages",
        "experiments",
        "report_controls",
        "entrypoints",
        "permissions",
        "pipeline_config",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("EviSOZ execution plan fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != EXECUTION_PLAN_SCHEMA_VERSION:
        raise ValueError("EviSOZ execution plan schema drifted")
    if data["status"] not in {"STAGE0_NO_GO", "STAGE0_GO_DOWNSTREAM_NOT_RUN"}:
        raise ValueError("EviSOZ execution plan status drifted")
    if not isinstance(data["stage0_gate_id"], str) or not data["stage0_gate_id"]:
        raise ValueError("execution plan gate ID is invalid")
    if (
        not isinstance(data["stage0_gate_receipt_sha256"], str)
        or len(data["stage0_gate_receipt_sha256"]) != 64
    ):
        raise ValueError("execution plan gate receipt is invalid")
    baseline = data["baseline"]
    if baseline != {
        "model": "canonical_v29_H_D",
        "route": "standard19_car_exact_reference",
        "frozen": True,
        "tcp22_role": "signed_bipolar_edge_evidence_only",
        "missing_channel_policy": "explicit_mask_only",
        "interpolation": "independent_ablation_only",
    }:
        raise ValueError("execution plan baseline contract drifted")
    check_statuses = data["stage0_check_statuses"]
    if type(check_statuses) is not dict or any(
        status not in {"GO", "QUALIFIED_GO", "EVALUATOR_ONLY_GO", "PARTIAL", "NO_GO"}
        for status in check_statuses.values()
    ):
        raise ValueError("execution plan check statuses are invalid")
    if data["blocking_check_ids"] != sorted(data["blocking_check_ids"]):
        raise ValueError("execution plan blocking checks are not sorted")
    if data["status"] == "STAGE0_NO_GO" and not data["prohibited_actions"]:
        raise ValueError("NO_GO execution plan lacks prohibited actions")
    stages = data["stages"]
    if not isinstance(stages, list) or [row.get("stage_id") for row in stages] != [
        definition[0] for definition in _STAGE_DEFINITIONS
    ]:
        raise ValueError("execution plan stage roster drifted")
    for row in stages:
        if type(row) is not dict or set(row) != {
            "stage_id",
            "title",
            "status",
            "required_stage0_checks",
            "missing_or_non_go_checks",
            "formal_training_allowed",
        }:
            raise ValueError("execution plan stage fields drifted")
        if row["status"] not in {"GO", "NO_GO", "BLOCKED_BY_STAGE0", "READY_FOR_AUTHORIZED_RUN"}:
            raise ValueError("execution plan stage status drifted")
        if row["formal_training_allowed"] and data["status"] != "STAGE0_GO_DOWNSTREAM_NOT_RUN":
            raise ValueError("execution plan opened training under Stage-0 NO_GO")
    if not isinstance(data["experiments"], list) or [row.get("id") for row in data["experiments"]] != [
        item[0] for item in _EXPERIMENTS
    ]:
        raise ValueError("execution plan experiment roster drifted")
    if not isinstance(data["report_controls"], list) or [row.get("control_id") for row in data["report_controls"]] != list(_REPORT_CONTROLS):
        raise ValueError("execution plan report control roster drifted")
    permissions = data["permissions"]
    if permissions != {
        "training_authorized": data["status"] == "STAGE0_GO_DOWNSTREAM_NOT_RUN",
        "qwen_generation_authorized": False,
        "private_report_language_evaluation_authorized": False,
        "clinical_deployment_authorized": False,
        "knowledge_creates_patient_facts": False,
    }:
        raise ValueError("execution plan permissions drifted")
    if data["plan_id"] != _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]:
        raise ValueError("execution plan ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("execution plan receipt drifted")
    return data


__all__ = [
    "EXECUTION_PLAN_SCHEMA_VERSION",
    "build_evisoz_execution_plan",
    "validate_evisoz_execution_plan",
]
