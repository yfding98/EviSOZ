"""Fail-closed validation for the detached clinical EEG experiment freeze.

The v1 manifest is deliberately detached from the method-policy JSON.  Its
current checked-in instance is a *draft*: it records unresolved freeze fields
and a previously touched source-evaluation engineering-smoke recording, but it
does not authorize a frozen experiment or any production/private report route.

JSON Schema validation closes the wire format.  The checks below close the
cross-field semantics that JSON Schema cannot express, especially lockbox
access, empty clinical-term promotion, receipt-backed gates, and the rule that
a draft can never authorize production.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator, FormatChecker


EXPERIMENT_FREEZE_SCHEMA_VERSION = "clinical_eeg_experiment_freeze_v1"

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _ROOT / "schemas" / "clinical_eeg_experiment_freeze_v1.schema.json"
_DEFAULT_MANIFEST_PATH = (
    _ROOT / "configs" / "clinical_eeg_experiment_freeze_v1.json"
)

_EXPECTED_COHORT_ROLES = {
    "source_train",
    "source_dev",
    "calibration",
    "locked_source_eval",
    "external_test",
    "private_post_freeze_eval",
}
_EXPECTED_PROMOTION_GATES = {
    "GATE_DETECTOR_OPERATING_POINT_AND_LOCKED_SOURCE_EVAL",
    "GATE_ADAPTIVE_FIXED_COMPUTE_BUDGET",
    "GATE_ATOMIC_TERM_QUALIFICATION",
    "GATE_ENDPOINT_ANALYSIS_PLAN",
    "GATE_CLAIM_RENDER_PUBLIC_SHADOW_ROUTE",
    "GATE_REPRESENTATIVE_DUAL_EXPERT_READER_STUDY",
}
_EXPECTED_PROMOTION_GATE_RECEIPT_KINDS = {
    "GATE_DETECTOR_OPERATING_POINT_AND_LOCKED_SOURCE_EVAL": {
        "detector_operating_point_receipt",
        "locked_source_eval_benchmark_receipt",
    },
    "GATE_ADAPTIVE_FIXED_COMPUTE_BUDGET": {
        "adaptive_budget_profile_receipt",
        "compute_matched_ablation_receipt",
    },
    "GATE_ATOMIC_TERM_QUALIFICATION": {
        "patient_disjoint_term_qualification_receipt",
        "term_opportunity_and_sensitivity_receipt",
        "event_findings_atom_roster_closure_receipt",
        "independent_event_findings_denominator_receipt",
    },
    "GATE_ENDPOINT_ANALYSIS_PLAN": {
        "endpoint_formula_registry_receipt",
        "multiplicity_and_failure_policy_receipt",
    },
    "GATE_CLAIM_RENDER_PUBLIC_SHADOW_ROUTE": {
        "source_bound_claim_graph_receipt",
        "deterministic_fallback_and_roster_closure_receipt",
    },
    "GATE_REPRESENTATIVE_DUAL_EXPERT_READER_STUDY": {
        "reader_study_roster_and_blinding_receipt",
        "reader_study_result_receipt",
    },
}
_REQUIRED_RECORDED_ENGINEERING_SMOKE = {
    "access_id": "LOCKBOX-ACCESS-DEEPSOZ-ENGINEERING-SMOKE-0001",
    "patient_reference": "source_eval_patient_16",
    "recording_reference": "aaaaaaaq_s006_t000",
    "source_receipt_ids": {
        "DSZBATCH-a7e110cd76153a7fe3d4ad47",
        "DSZOOF-058523cfafd4765deedc5aca",
    },
}
_ENDPOINT_ROLES = {
    "safety_gates": "safety_gate",
    "primary": "primary",
    "key_secondary": "key_secondary",
    "exploratory": "exploratory",
}


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _path(error: Any) -> str:
    parts = [str(item) for item in error.absolute_path]
    return ".".join(parts) if parts else "$"


def _reject_nonfinite(value: object, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def _unique(values: Iterable[str], context: str) -> set[str]:
    rows = list(values)
    if len(rows) != len(set(rows)):
        raise ValueError(f"{context} contains duplicate IDs")
    return set(rows)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_artifact_binding(
    value: Mapping[str, object],
    context: str,
    *,
    require_hash: bool,
) -> None:
    relative = Path(str(value["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context}.path must be a workspace-relative path")
    resolved = _ROOT / relative
    if not resolved.is_file():
        raise ValueError(f"{context}.path does not exist: {relative}")
    expected = value["sha256"]
    if require_hash and expected is None:
        raise ValueError(f"{context}.sha256 is required for a frozen manifest")
    if expected is not None and str(expected) != _file_sha256(resolved):
        raise ValueError(f"{context}.sha256 does not match {relative}")


def _validate_freeze_identity(payload: Mapping[str, object]) -> None:
    state = str(payload["state"])
    identity = payload["freeze_identity"]
    assert isinstance(identity, Mapping)
    if identity["state"] != state:
        raise ValueError("freeze_identity.state must equal manifest state")

    require_hash = state == "frozen"
    _validate_artifact_binding(
        identity["policy_binding"],
        "freeze_identity.policy_binding",
        require_hash=require_hash,
    )
    _validate_artifact_binding(
        identity["schema_binding"],
        "freeze_identity.schema_binding",
        require_hash=require_hash,
    )

    if state == "draft":
        if identity["frozen_at"] is not None:
            raise ValueError("a draft manifest cannot have frozen_at")
    elif state == "frozen":
        required = {
            "frozen_at": identity["frozen_at"],
            "git_commit": identity["git_commit"],
            "environment_lock_sha256": identity["environment_lock_sha256"],
        }
        missing = sorted(key for key, value in required.items() if value is None)
        if missing:
            raise ValueError(
                "frozen manifest is missing freeze identity fields: "
                + ", ".join(missing)
            )
        if not identity["random_seed_set"]:
            raise ValueError("frozen manifest requires a non-empty random_seed_set")


def _validate_cohorts_and_lockbox(payload: Mapping[str, object]) -> None:
    block = payload["cohorts_and_lockbox"]
    assert isinstance(block, Mapping)
    cohorts = block["cohorts"]
    assert isinstance(cohorts, list)
    roles = _unique((str(row["role"]) for row in cohorts), "cohort roles")
    if roles != _EXPECTED_COHORT_ROLES:
        raise ValueError(
            "cohort roles must exactly match the six registered experiment roles"
        )

    for row in cohorts:
        if row["status"] == "not_materialized":
            unresolved = (
                "roster_path",
                "roster_sha256",
                "patient_count",
                "recording_count",
                "event_count",
                "seizure_free_recording_count",
            )
            if any(row[key] is not None for key in unresolved):
                raise ValueError(
                    f"cohort {row['role']} is not_materialized but has frozen roster data"
                )
        if row["status"] == "locked":
            required = ("roster_path", "roster_sha256", "patient_count", "recording_count")
            if any(row[key] is None for key in required):
                raise ValueError(
                    f"locked cohort {row['role']} is missing roster identity or counts"
                )

    ledger = block["access_ledger"]
    assert isinstance(ledger, list)
    _unique((str(row["access_id"]) for row in ledger), "lockbox access IDs")
    if bool(ledger) != bool(block["source_eval_accessed"]):
        raise ValueError(
            "source_eval_accessed must exactly reflect whether access_ledger is non-empty"
        )
    if ledger and block["pristine_untouched_lockbox_claim_authorized"]:
        raise ValueError(
            "an accessed source-eval lockbox cannot be claimed pristine and untouched"
        )

    for row in ledger:
        access_type = row["access_type"]
        if access_type == "engineering_smoke":
            if any(
                bool(row[key])
                for key in (
                    "used_for_training",
                    "used_for_threshold_selection",
                    "used_for_model_selection",
                )
            ):
                raise ValueError(
                    f"engineering smoke {row['access_id']} cannot be used for fitting or selection"
                )
            if row["disposition"] not in {
                "exclude_from_final_untouched_lockbox_unless_predeclared_exception",
                "predeclared_engineering_exception",
            }:
                raise ValueError(
                    f"engineering smoke {row['access_id']} lacks an exclusion/exception disposition"
                )

    recorded_smoke = {
        str(row["access_id"]): row
        for row in ledger
        if row["access_type"] == "engineering_smoke"
    }.get(_REQUIRED_RECORDED_ENGINEERING_SMOKE["access_id"])
    if recorded_smoke is None:
        raise ValueError(
            "the known DeepSOZ source-eval engineering smoke access must remain in the lockbox ledger"
        )
    for key in ("patient_reference", "recording_reference"):
        if recorded_smoke[key] != _REQUIRED_RECORDED_ENGINEERING_SMOKE[key]:
            raise ValueError(f"known DeepSOZ engineering smoke has an incorrect {key}")
    if set(recorded_smoke["source_receipt_ids"]) != _REQUIRED_RECORDED_ENGINEERING_SMOKE[
        "source_receipt_ids"
    ]:
        raise ValueError("known DeepSOZ engineering-smoke receipt IDs changed")


def _validate_detector(payload: Mapping[str, object]) -> None:
    detector = payload["detector_operating_point"]
    lockbox = payload["cohorts_and_lockbox"]
    assert isinstance(detector, Mapping)
    assert isinstance(lockbox, Mapping)
    state = detector["state"]

    selected = detector["selected_provider_id"]
    if selected is not None and selected not in detector["candidate_provider_ids"]:
        raise ValueError("selected detector provider is not registered as a candidate")

    smoke_accessed = any(
        row["access_type"] == "engineering_smoke"
        for row in lockbox["access_ledger"]
    )
    if bool(detector["source_eval_accessed_for_engineering_smoke"]) != smoke_accessed:
        raise ValueError(
            "detector source_eval_accessed_for_engineering_smoke disagrees with lockbox ledger"
        )
    selection_accessed = any(
        bool(row["used_for_threshold_selection"])
        or bool(row["used_for_model_selection"])
        for row in lockbox["access_ledger"]
    )
    if bool(detector["source_eval_opened_for_selection"]) != selection_accessed:
        raise ValueError(
            "detector source_eval_opened_for_selection disagrees with lockbox ledger"
        )

    unresolved = (
        "selected_provider_id",
        "primary_false_alarms_per_24h",
        "selected_decoder_policy_id",
        "selected_threshold",
        "source_eval_prediction_artifact_sha256",
    )
    if state == "not_selected":
        if any(detector[key] is not None for key in unresolved):
            raise ValueError("not_selected detector operating point has selected values")
        if detector["promotion_eligible"]:
            raise ValueError("not_selected detector operating point cannot be promotion eligible")
    elif state == "frozen":
        if payload["state"] != "frozen":
            raise ValueError("detector operating point cannot freeze under a non-frozen manifest")
        if any(detector[key] is None for key in unresolved):
            raise ValueError("frozen detector operating point is incomplete")
        if not detector["promotion_eligible"]:
            raise ValueError("frozen detector operating point must explicitly pass its gate")


def _validate_adaptive_budget(payload: Mapping[str, object]) -> None:
    budget = payload["adaptive_fixed_compute_budget"]
    assert isinstance(budget, Mapping)
    selected_fields = (
        "selected_fine_eeg_seconds_per_event",
        "selected_tokens_per_event",
        "selected_incremental_gpu_seconds_per_eeg_hour",
        "selected_max_left_seconds",
        "selected_max_right_seconds",
        "budget_profile_id",
        "budget_receipt_id",
    )
    if budget["state"] == "draft_unfrozen":
        if any(budget[key] is not None for key in selected_fields):
            raise ValueError("draft adaptive budget contains selected freeze values")
        if budget["selected_action_delta_seconds"]:
            raise ValueError("draft adaptive budget cannot contain selected action deltas")
        if budget["promotion_eligible"]:
            raise ValueError("draft adaptive budget cannot be promotion eligible")
    elif budget["state"] == "frozen":
        if payload["state"] != "frozen":
            raise ValueError("adaptive budget cannot freeze under a non-frozen manifest")
        if any(budget[key] is None for key in selected_fields):
            raise ValueError("frozen adaptive fixed-compute budget is incomplete")
        if not budget["selected_action_delta_seconds"]:
            raise ValueError("frozen adaptive budget requires selected action deltas")
        if not budget["promotion_eligible"]:
            raise ValueError("frozen adaptive budget must explicitly pass its gate")
    if not budget["compute_match_required"]:
        raise ValueError("adaptive ablations must remain compute matched")
    if not budget["whole_record_navigation_cost_reported_separately"]:
        raise ValueError("whole-record navigation cost must be reported separately")


def _validate_term_qualification(payload: Mapping[str, object]) -> None:
    block = payload["term_qualification"]
    assert isinstance(block, Mapping)
    report_terms = set(block["report_eligible_term_allowlist"])
    candidate_terms = set(block["candidate_only_term_allowlist"])
    forbidden = set(block["forbidden_surface_terms"])
    if report_terms.intersection(forbidden) or candidate_terms.intersection(forbidden):
        raise ValueError("qualified/candidate terms intersect forbidden surface terms")
    if block["state"] == "no_terms_qualified":
        if report_terms:
            raise ValueError("no_terms_qualified requires an empty report allowlist")
        if block["term_qualification_receipt_ids"]:
            raise ValueError("no_terms_qualified cannot cite qualification receipts")
        if block["report_eligible_event_outcome_allowlist"]:
            raise ValueError("no_terms_qualified cannot promote qualified event outcomes")
        if block["promotion_eligible"]:
            raise ValueError("no_terms_qualified cannot be promotion eligible")
    elif block["state"] == "qualified":
        if payload["state"] != "frozen":
            raise ValueError("terms cannot be qualified under a non-frozen manifest")
        required = (
            block["ontology_registry_path"],
            block["ontology_registry_sha256"],
            block["term_qualification_receipt_ids"],
            block["report_eligible_term_allowlist"],
        )
        if any(not value for value in required):
            raise ValueError("qualified term registry/allowlist/receipts are incomplete")
        if not block["promotion_eligible"]:
            raise ValueError("qualified terms must explicitly pass their promotion gate")


def _validate_endpoint_hierarchy(payload: Mapping[str, object]) -> None:
    hierarchy = payload["endpoint_hierarchy"]
    assert isinstance(hierarchy, Mapping)
    endpoint_ids: list[str] = []
    for collection, expected_role in _ENDPOINT_ROLES.items():
        for row in hierarchy[collection]:
            endpoint_ids.append(str(row["endpoint_id"]))
            if row["role"] != expected_role:
                raise ValueError(
                    f"endpoint {row['endpoint_id']} is in {collection} but has role {row['role']}"
                )
            if bool(row["estimable"]) != (row["status"] == "frozen_estimable"):
                raise ValueError(
                    f"endpoint {row['endpoint_id']} has inconsistent estimable/status fields"
                )
    _unique(endpoint_ids, "endpoint IDs")

    multiplicity = hierarchy["multiplicity"]
    if hierarchy["state"] == "draft_unfrozen":
        if payload["state"] != "draft":
            raise ValueError("draft endpoint hierarchy requires a draft manifest")
        for collection in _ENDPOINT_ROLES:
            for row in hierarchy[collection]:
                if row["status"] != "draft_not_estimable":
                    raise ValueError("draft endpoint hierarchy cannot contain frozen endpoints")
                if row["formula_version"] is not None or row["analysis_profile_id"] is not None:
                    raise ValueError(
                        "draft endpoint hierarchy cannot masquerade as a frozen analysis plan"
                    )
        if any(multiplicity[key] is not None for key in multiplicity):
            raise ValueError("draft endpoint hierarchy cannot freeze multiplicity fields")
    elif hierarchy["state"] == "frozen":
        if payload["state"] != "frozen":
            raise ValueError("endpoint hierarchy cannot freeze under a non-frozen manifest")
        for row in hierarchy["safety_gates"] + hierarchy["primary"]:
            if not row["estimable"]:
                raise ValueError("frozen safety/primary endpoints must be estimable")
            if row["formula_version"] is None or row["analysis_profile_id"] is None:
                raise ValueError("frozen safety/primary endpoint lacks formula/profile")
        if any(multiplicity[key] is None for key in multiplicity):
            raise ValueError("frozen endpoint hierarchy requires a multiplicity plan")


def _validate_promotion_and_production(payload: Mapping[str, object]) -> None:
    gates = payload["promotion_gates"]
    assert isinstance(gates, list)
    gate_ids = _unique((str(row["gate_id"]) for row in gates), "promotion gate IDs")
    if gate_ids != _EXPECTED_PROMOTION_GATES:
        raise ValueError("promotion gate IDs do not match the registered v1 gate set")

    for row in gates:
        receipt_kinds = set(str(value) for value in row["required_receipt_kinds"])
        if receipt_kinds != _EXPECTED_PROMOTION_GATE_RECEIPT_KINDS[row["gate_id"]]:
            raise ValueError(
                f"promotion gate {row['gate_id']} required receipt kinds do not "
                "match the registered v1 contract"
            )
        if bool(row["passed"]) != (row["status"] == "passed"):
            raise ValueError(f"promotion gate {row['gate_id']} has inconsistent status/passed")
        if row["status"] == "not_evaluated":
            unresolved = (
                row["evidence_receipt_ids"],
                row["evaluated_against_manifest_id"],
                row["evaluator_code_sha256"],
                row["evaluated_at"],
            )
            if any(value not in (None, []) for value in unresolved):
                raise ValueError(
                    f"not-evaluated promotion gate {row['gate_id']} contains evaluation evidence"
                )
        if row["status"] == "passed":
            required = (
                row["evidence_receipt_ids"],
                row["evaluated_against_manifest_id"],
                row["evaluator_code_sha256"],
                row["evaluated_at"],
            )
            if any(not value for value in required):
                raise ValueError(
                    f"passed promotion gate {row['gate_id']} lacks a receipt-backed evaluation"
                )
            if row["evaluated_against_manifest_id"] != payload["manifest_id"]:
                raise ValueError(
                    f"promotion gate {row['gate_id']} was evaluated against another manifest"
                )

    authorization = payload["production_authorization"]
    assert isinstance(authorization, Mapping)
    if payload["state"] != "frozen":
        if any(
            bool(authorization[key])
            for key in ("requested", "authorized", "production_route_connected")
        ) or authorization["authorization_receipt_id"] is not None:
            raise ValueError("draft/invalidated manifest cannot request or authorize production")
        if any(row["passed"] for row in gates):
            raise ValueError("draft/invalidated manifest cannot contain passed promotion gates")
    if authorization["authorized"] and not authorization["requested"]:
        raise ValueError("production cannot be authorized without an explicit request")
    if authorization["production_route_connected"] and not authorization["authorized"]:
        raise ValueError("production route cannot connect without authorization")
    if authorization["authorized"]:
        if authorization["authorization_receipt_id"] is None:
            raise ValueError("production authorization lacks an authorization receipt")
        if not all(row["passed"] for row in gates):
            raise ValueError("production authorization requires every promotion gate to pass")
        if not all(
            (
                payload["detector_operating_point"]["promotion_eligible"],
                payload["adaptive_fixed_compute_budget"]["promotion_eligible"],
                payload["term_qualification"]["promotion_eligible"],
            )
        ):
            raise ValueError("production authorization requires all method components to promote")


def validate_clinical_eeg_experiment_freeze_v1(value: object) -> dict[str, Any]:
    """Validate and defensively copy one detached experiment-freeze manifest."""

    if type(value) is not dict:
        raise TypeError("clinical EEG experiment-freeze manifest must be an object")
    candidate: dict[str, Any] = deepcopy(value)
    _reject_nonfinite(candidate)
    errors = sorted(
        _schema_validator().iter_errors(candidate),
        key=lambda item: list(item.path),
    )
    if errors:
        rendered = "; ".join(
            f"{_path(error)}: {error.message}" for error in errors[:8]
        )
        if len(errors) > 8:
            rendered += f"; ... {len(errors) - 8} more error(s)"
        raise ValueError(
            "clinical_eeg_experiment_freeze_v1 schema validation failed: "
            + rendered
        )

    _validate_freeze_identity(candidate)
    _validate_cohorts_and_lockbox(candidate)
    _validate_detector(candidate)
    _validate_adaptive_budget(candidate)
    _validate_term_qualification(candidate)
    _validate_endpoint_hierarchy(candidate)
    _validate_promotion_and_production(candidate)
    return candidate


def load_clinical_eeg_experiment_freeze_v1(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate the checked-in draft or another detached manifest."""

    resolved = _DEFAULT_MANIFEST_PATH if path is None else Path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return validate_clinical_eeg_experiment_freeze_v1(payload)


def assert_clinical_eeg_experiment_freeze_allows_production(
    value: object,
) -> dict[str, Any]:
    """Return a validated frozen manifest or fail closed for production use."""

    candidate = validate_clinical_eeg_experiment_freeze_v1(value)
    if candidate["state"] != "frozen":
        raise ValueError("production requires state=frozen; draft is never sufficient")
    authorization = candidate["production_authorization"]
    if not (
        authorization["requested"]
        and authorization["authorized"]
        and authorization["production_route_connected"]
        and authorization["authorization_receipt_id"]
    ):
        raise ValueError("frozen manifest does not contain complete production authorization")
    return candidate


__all__ = [
    "EXPERIMENT_FREEZE_SCHEMA_VERSION",
    "assert_clinical_eeg_experiment_freeze_allows_production",
    "load_clinical_eeg_experiment_freeze_v1",
    "validate_clinical_eeg_experiment_freeze_v1",
]
