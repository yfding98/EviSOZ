"""Evidence-bound aggregate Stage-0 gate for EviSOZ-LM."""

from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.validate_eeg_knowledge_system import validate_knowledge_system
from src.evisoz.baseline.v29_public_cache_materializer import (
    validate_public_v29_cache_materialization_receipt,
)
from src.evisoz.data.artifact_ref import (
    build_json_artifact_ref,
    canonical_json_sha256,
)
from src.evisoz.data.clean_freeze import (
    CLEAN_FREEZE_AUDIT_SCHEMA_VERSION,
    validate_clean_freeze_audit,
)
from src.evisoz.data.private_physician_reports import (
    PHYSICIAN_REPORT_INVENTORY_SCHEMA_VERSION,
    validate_private_physician_report_inventory,
)
from src.evisoz.data.private_stage0_cohort_materializer import (
    validate_private_stage0_cohort_artifact,
)
from src.evisoz.data.private_stage0_split import (
    build_private_patient_linkage_group,
)
from src.evisoz.data.public_exposure_projection import (
    PUBLIC_EXPOSURE_PROJECTION_SCHEMA_VERSION,
    validate_public_auxiliary_exposure_projection,
)
from src.evisoz.data.public_auxiliary_field_release import (
    PUBLIC_AUXILIARY_FIELD_RELEASE_SCHEMA_VERSION,
    validate_public_auxiliary_field_release,
)
from src.evisoz.data.public_v29_tusz_crosswalk import (
    PUBLIC_V29_TUSZ_CROSSWALK_SCHEMA_VERSION,
    validate_public_v29_tusz_crosswalk,
)
from src.evisoz.data.public_overlap_audit import (
    PUBLIC_OVERLAP_AUDIT_SCHEMA_VERSION,
    validate_public_overlap_audit_receipt,
)
from src.evisoz.data.private_report_mapping_intake import (
    PRIVATE_REPORT_MAPPING_INTAKE_SCHEMA_VERSION,
    validate_private_report_mapping_intake,
)
from src.evisoz.data.private_report_exclusion import (
    PRIVATE_REPORT_EXCLUSION_SCHEMA_VERSION,
    validate_private_report_exclusion,
)
from src.evisoz.data.private_physician_report_release import (
    PHYSICIAN_REPORT_RELEASE_SCHEMA_VERSION,
    validate_private_physician_report_release,
)
from src.evisoz.data.schema_registry import validate_schema_registry
from src.evisoz.data.split_ledger import validate_split_roster
from src.evisoz.forge.private_report_deidentification import (
    PRIVATE_REPORT_DEID_CANDIDATES_SCHEMA_VERSION,
    validate_private_report_deidentification_candidates,
)
from src.evisoz.forge.private_stage0_examples import (
    PRIVATE_STAGE0_EXAMPLES_SCHEMA_VERSION,
    validate_private_stage0_examples_materialization,
)
from src.evisoz.forge.deterministic_signal_candidates import (
    CANDIDATE_MATERIALIZATION_SCHEMA_VERSION,
    validate_deterministic_signal_candidate_materialization,
)
from src.evisoz.forge.candidate_exposure_ledger import (
    CANDIDATE_EXPOSURE_LEDGER_SCHEMA_VERSION,
    validate_candidate_exposure_ledger,
)
from src.evisoz.forge.teacher_candidates import (
    TEACHER_CANDIDATE_MATERIALIZATION_SCHEMA_VERSION,
    validate_teacher_candidate_materialization,
)
from src.evisoz.forge.findings_claims_reports import (
    MATERIALIZATION_SCHEMA_VERSION as FINDINGS_CLAIM_REPORT_MATERIALIZATION_SCHEMA_VERSION,
    validate_findings_claim_report_materialization,
)
from src.evisoz.forge.evidence_binding import (
    BOUND_MATERIALIZATION_SCHEMA_VERSION,
    validate_bound_evidence_materialization,
)


STAGE0_GATE_SCHEMA_VERSION = "evisoz_stage0_gate_v1"
_HASH_PLACEHOLDER = "0" * 64
_GATE_ID_PREFIX = "EVISOZ-STAGE0-"
_PROHIBITED_ACTIONS = [
    "query_decoder_or_residual_formal_training",
    "nonzero_residual_gate",
    "large_scale_teacher_inference",
    "qwen_sft_or_eeg_to_qwen_alignment",
    "private_label_training",
    "physician_report_training_or_language_evaluation",
]
_AUTHORIZED_NEXT_ACTIONS = [
    "resolve_report_associations_with_authoritative_mapping",
    "perform_manual_deidentification_review",
    "obtain_private_data_governance_training_authorization",
    "complete_public_auxiliary_near_partial_overlap_audit",
    "obtain_tuev_eval_patient_identity_authority",
    "materialize_small_calibration_only_teacher_candidates",
    "materialize_fold_local_candidate_calibration_receipts",
    "perform_clean_freeze_audit",
]


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Stage-0 gate JSON input must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError("Stage-0 gate JSON input must be an object")
    return value


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    result = _hash_source(value)
    result["gate_id"] = "CONTENT-ADDRESS-PENDING"
    return result


def _ref(payload: object, *, kind: str, schema_version: str) -> dict[str, Any]:
    return build_json_artifact_ref(
        payload,
        artifact_kind=kind,
        payload_schema_version=schema_version,
    )


def _trusted_private_groups(signal_roster_path: Path) -> dict[str, dict[str, Any]]:
    with signal_roster_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    patient_ids = sorted({str(row.get("patient_id", "")) for row in rows})
    if not patient_ids or "" in patient_ids:
        raise ValueError("Stage-0 gate private signal roster is invalid")
    groups = [build_private_patient_linkage_group(patient_id) for patient_id in patient_ids]
    return {group["linkage_group_id"]: group for group in groups}


def build_stage0_gate(
    *,
    repository_root: Path,
    schema_registry_path: Path,
    public_v29_root: Path,
    private_real_cohort_root: Path,
    private_split_roster_path: Path,
    private_signal_roster_path: Path,
    private_examples_root: Path,
    private_report_inventory_path: Path,
    private_report_deid_root: Path,
    knowledge_root: Path,
    public_exposure_projection_path: Path,
    deterministic_signal_candidates_root: Path,
    findings_claim_report_root: Path,
    public_v29_tusz_crosswalk_path: Path | None = None,
    public_auxiliary_field_release_path: Path | None = None,
    public_overlap_audit_path: Path | None = None,
    candidate_exposure_ledger_root: Path | None = None,
    private_report_mapping_intake_root: Path | None = None,
    private_report_exclusion_path: Path | None = None,
    private_report_release_path: Path | None = None,
    bound_evidence_root: Path | None = None,
    teacher_cerebragloss_root: Path | None = None,
    teacher_elm_root: Path | None = None,
    clean_freeze_audit_path: Path | None = None,
) -> dict[str, Any]:
    """Replay completed artifacts and emit the current fail-closed gate."""

    repository = repository_root.resolve(strict=True)
    schema_registry = validate_schema_registry(
        _json(schema_registry_path.resolve(strict=True)),
        repository_root=repository,
    )
    public_receipt_path = (
        public_v29_root.resolve(strict=True) / "audit" / "materialization_receipt.json"
    )
    public_receipt = validate_public_v29_cache_materialization_receipt(
        _json(public_receipt_path),
        root=public_v29_root,
    )
    private_cohort = validate_private_stage0_cohort_artifact(private_real_cohort_root)
    private_cohort_manifest = _json(
        private_real_cohort_root.resolve(strict=True) / "manifest.json"
    )
    trusted_groups = _trusted_private_groups(private_signal_roster_path)
    split = validate_split_roster(
        _json(private_split_roster_path),
        trusted_linkage_groups=trusted_groups,
    )
    examples = validate_private_stage0_examples_materialization(
        _json(private_examples_root / "manifest.json"),
        output_root=private_examples_root,
        split_roster=split,
        trusted_groups=trusted_groups,
        cohort_root=private_real_cohort_root,
    )
    report_inventory = validate_private_physician_report_inventory(
        _json(private_report_inventory_path)
    )
    private_report_exclusion: dict[str, Any] | None = None
    if private_report_exclusion_path is not None:
        exclusion_path = private_report_exclusion_path.resolve()
        if exclusion_path.is_file() and not exclusion_path.is_symlink():
            private_report_exclusion = validate_private_report_exclusion(
                _json(exclusion_path), report_inventory=report_inventory
            )
    report_mapping_intake: dict[str, Any] | None = None
    if private_report_mapping_intake_root is not None:
        intake_path = private_report_mapping_intake_root.resolve() / "intake.json"
        if intake_path.is_file() and not intake_path.is_symlink():
            report_mapping_intake = validate_private_report_mapping_intake(
                _json(intake_path)
            )
            expected_inventory_ref = _ref(
                report_inventory,
                kind="physician_report_inventory",
                schema_version=PHYSICIAN_REPORT_INVENTORY_SCHEMA_VERSION,
            )
            if report_mapping_intake["inventory_ref"] != expected_inventory_ref:
                raise ValueError("private report mapping intake inventory binding drifted")
            expected_split_ref = _ref(
                split,
                kind="split_roster",
                schema_version="evisoz_split_roster_v1",
            )
            if report_mapping_intake["split_roster_ref"] != expected_split_ref:
                raise ValueError("private report mapping intake split binding drifted")
            unresolved_by_id = {
                str(row["report_id"]): row["document_ref"]
                for row in report_inventory["reports"]
                if row["association"]["status"] == "unresolved"
            }
            intake_by_id = {
                str(row["report_id"]): row["document_ref"]
                for row in report_mapping_intake["requests"]
            }
            if intake_by_id != unresolved_by_id:
                raise ValueError("private report mapping intake request roster drifted")
    unresolved_report_ids = {
        str(row["report_id"])
        for row in report_inventory["reports"]
        if row["association"]["status"] == "unresolved"
    }
    excluded_report_ids = (
        {str(row["report_id"]) for row in private_report_exclusion["entries"]}
        if private_report_exclusion is not None
        else set()
    )
    if excluded_report_ids and excluded_report_ids != unresolved_report_ids:
        raise ValueError(
            "private report exclusion must cover every unresolved report and no resolved report"
        )
    deid_candidates = validate_private_report_deidentification_candidates(
        _json(private_report_deid_root / "manifest.json"),
        output_root=private_report_deid_root,
    )
    private_report_release: dict[str, Any] | None = None
    if private_report_release_path is not None:
        release_path = private_report_release_path.resolve()
        if release_path.is_symlink() or not release_path.is_file():
            raise ValueError("private physician report release must be a regular JSON file")
        private_report_release = validate_private_physician_report_release(
            _json(release_path),
            candidate_bundle=deid_candidates,
            candidate_output_root=private_report_deid_root,
        )
    knowledge = validate_knowledge_system(knowledge_root)
    exposure = validate_public_auxiliary_exposure_projection(
        _json(public_exposure_projection_path)
    )
    crosswalk: dict[str, Any] | None = None
    if public_v29_tusz_crosswalk_path is not None:
        candidate = public_v29_tusz_crosswalk_path.resolve()
        if candidate.is_file() and not candidate.is_symlink():
            crosswalk = validate_public_v29_tusz_crosswalk(_json(candidate))
    field_release: dict[str, Any] | None = None
    if public_auxiliary_field_release_path is not None:
        candidate = public_auxiliary_field_release_path.resolve()
        if candidate.is_file() and not candidate.is_symlink():
            field_release = validate_public_auxiliary_field_release(_json(candidate))
    public_overlap_audit: dict[str, Any] | None = None
    if public_overlap_audit_path is not None:
        candidate = public_overlap_audit_path.resolve()
        if candidate.is_file() and not candidate.is_symlink():
            public_overlap_audit = validate_public_overlap_audit_receipt(_json(candidate))
            expected_projection_ref = _ref(
                exposure,
                kind="public_auxiliary_exposure_projection",
                schema_version=PUBLIC_EXPOSURE_PROJECTION_SCHEMA_VERSION,
            )
            if public_overlap_audit["source_projection_ref"] != expected_projection_ref:
                raise ValueError("public overlap audit source projection binding drifted")
    deterministic_candidates = validate_deterministic_signal_candidate_materialization(
        _json(deterministic_signal_candidates_root / "manifest.json"),
        output_root=deterministic_signal_candidates_root,
        real_cohort_root=private_real_cohort_root,
        replay_features=False,
    )
    candidate_exposure_ledger: dict[str, Any] | None = None
    if candidate_exposure_ledger_root is not None:
        ledger_path = candidate_exposure_ledger_root.resolve() / "ledger.json"
        if ledger_path.is_file() and not ledger_path.is_symlink():
            candidate_exposure_ledger = validate_candidate_exposure_ledger(
                _json(ledger_path)
            )
    teacher_materializations: dict[str, dict[str, Any]] = {}
    for teacher_id, root in (
        ("cerebragloss", teacher_cerebragloss_root),
        ("elm", teacher_elm_root),
    ):
        if root is None:
            continue
        candidate_root = root.resolve()
        manifest_path = (
            candidate_root / "manifest.json"
            if candidate_root.is_dir()
            else candidate_root
        )
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError(f"{teacher_id} teacher materialization manifest is missing")
        materialization = validate_teacher_candidate_materialization(
            _json(manifest_path)
        )
        if materialization["teacher_id"] != teacher_id:
            raise ValueError(f"{teacher_id} teacher materialization ID drifted")
        teacher_materializations[teacher_id] = materialization
    findings_claim_reports = validate_findings_claim_report_materialization(
        _json(findings_claim_report_root / "manifest.json"),
        output_root=findings_claim_report_root,
    )
    bound_evidence: dict[str, Any] | None = None
    if bound_evidence_root is not None:
        bound_root = bound_evidence_root.resolve(strict=True)
        bound_evidence = validate_bound_evidence_materialization(
            _json(bound_root / "manifest.json"),
            output_root=bound_root,
        )

    clean_freeze_audit: dict[str, Any] | None = None
    if clean_freeze_audit_path is not None:
        audit_path = clean_freeze_audit_path.resolve()
        if audit_path.is_symlink() or not audit_path.is_file():
            raise ValueError("clean-freeze audit path must be a regular JSON file")
        clean_freeze_audit = validate_clean_freeze_audit(_json(audit_path))

    # The legacy candidate ledger predates the typed teacher manifests and
    # therefore always lists both teacher-missing codes.  When a validated
    # teacher manifest is supplied, retain only independent ledger closures
    # (such as calibration) and derive teacher-missing codes from the actual
    # manifest set.
    offline_candidate_blockers = (
        ["candidate_exposure_ledger_not_materialized"]
        if candidate_exposure_ledger is None
        else [
            code
            for code in candidate_exposure_ledger["missing_closure_codes"]
            if code
            not in {
                "cerebragloss_candidate_artifact_missing",
                "elm_candidate_artifact_missing",
            }
        ]
    )
    if "cerebragloss" not in teacher_materializations:
        offline_candidate_blockers.append("cerebragloss_candidate_artifact_missing")
    if "elm" not in teacher_materializations:
        offline_candidate_blockers.append("elm_candidate_artifact_missing")
    if any(
        "fold_local_calibration_receipts_missing"
        in materialization["missing_closure_codes"]
        for materialization in teacher_materializations.values()
    ) or deterministic_candidates["counts"]["fold_local_calibration_receipt_count"] == 0:
        offline_candidate_blockers.append("fold_local_calibration_receipts_missing")
    teacher_candidate_counts = {
        teacher_id: int(materialization["counts"]["candidate_count"])
        for teacher_id, materialization in teacher_materializations.items()
    }

    public_auxiliary_blockers = [
        "auxiliary_field_releases_not_materialized",
        "near_or_partial_overlap_closure_incomplete",
        "public_v29_to_tusz_crosswalk_not_materialized",
        "tuev_eval_patient_identity_opaque",
    ]
    if crosswalk is not None and crosswalk["status"] == "complete_audit_only_training_disabled":
        public_auxiliary_blockers.remove("public_v29_to_tusz_crosswalk_not_materialized")
    if (
        field_release is not None
        and field_release["status"] == "capability_catalog_materialized_training_disabled"
    ):
        public_auxiliary_blockers.remove("auxiliary_field_releases_not_materialized")
    if public_overlap_audit is not None and public_overlap_audit["status"] == "complete":
        public_auxiliary_blockers.remove("near_or_partial_overlap_closure_incomplete")
        public_auxiliary_blockers.remove("tuev_eval_patient_identity_opaque")
    deid_counts = deid_candidates["counts"]
    release_counts = (
        private_report_release["counts"] if private_report_release is not None else {}
    )
    release_complete = (
        private_report_release is not None
        and release_counts.get("released_row_count") == deid_counts["candidate_count"]
        and release_counts.get("development_qwen_training_count")
        == deid_counts["split_role_candidate_counts"].get("development_cv", 0)
        and release_counts.get("locked_language_evaluation_count")
        == deid_counts["split_role_candidate_counts"].get("locked_test", 0)
    )
    checks: list[dict[str, object]] = [
        {
            "check_id": "schema_registry",
            "status": "GO",
            "evidence_ref": _ref(
                schema_registry,
                kind="schema_registry",
                schema_version="evisoz_schema_registry_v1",
            ),
            "facts": {
                "entry_count": len(schema_registry["entries"]),
                "registry_id": schema_registry["registry_id"],
            },
            "blocker_codes": [],
        },
        {
            "check_id": "public_v29_reference",
            "status": "GO",
            "evidence_ref": _ref(
                public_receipt,
                kind="public_v29_materialization_receipt",
                schema_version=str(public_receipt["schema_version"]),
            ),
            "facts": {
                "patient_count": public_receipt["patient_count"],
                "unit_count": public_receipt["unit_count"],
                "alpha_zero_hard_bypass_replayed": public_receipt[
                    "alpha_zero_hard_bypass_replayed"
                ],
            },
            "blocker_codes": [],
        },
        {
            "check_id": "private_real_dual_montage",
            "status": "QUALIFIED_GO",
            "evidence_ref": _ref(
                private_cohort,
                kind="private_real_stage0_cohort_validation",
                schema_version=str(private_cohort["schema_version"]),
            ),
            "facts": {
                "materialized_event_count": private_cohort["validated_event_count"],
                "runtime_excluded_event_count": private_cohort_manifest[
                    "runtime_excluded_event_count"
                ],
                "reference_route": "protocol_authorized_opaque_common_reference",
            },
            "blocker_codes": [
                "edf_reference_token_unobservable",
                "locked_test_has_prior_frozen_v29_exposure",
            ],
        },
        {
            "check_id": "private_field_envelopes",
            "status": "EVALUATOR_ONLY_GO",
            "evidence_ref": _ref(
                examples,
                kind="private_stage0_examples_materialization",
                schema_version=PRIVATE_STAGE0_EXAMPLES_SCHEMA_VERSION,
            ),
            "facts": {
                "event_count": examples["counts"]["event_count"],
                "enabled_loss_port_event_counts": examples["counts"][
                    "enabled_loss_port_event_counts"
                ],
                "private_training_authority_present": examples["release_policy"][
                    "private_training_authority_present"
                ],
            },
            "blocker_codes": ["private_data_governance_training_authority_missing"],
        },
        {
            "check_id": "private_report_linkage",
            "status": (
                "GO"
                if not unresolved_report_ids or excluded_report_ids == unresolved_report_ids
                else "PARTIAL"
            ),
            "evidence_ref": _ref(
                report_inventory,
                kind="physician_report_inventory",
                schema_version=PHYSICIAN_REPORT_INVENTORY_SCHEMA_VERSION,
            ),
            "facts": {
                "report_count": report_inventory["counts"]["report_count"],
                "linked_high_confidence_count": report_inventory["counts"][
                    "association_status_counts"
                ].get("linked_high_confidence", 0),
                "unresolved_count": report_inventory["counts"][
                    "association_status_counts"
                ].get("unresolved", 0),
                "excluded_unresolved_count": len(excluded_report_ids),
                "exclusion_status": (
                    private_report_exclusion["decision"]["decision_type"]
                    if private_report_exclusion is not None
                    else "not_materialized"
                ),
                "exclusion_id": (
                    private_report_exclusion["exclusion_id"]
                    if private_report_exclusion is not None
                    else None
                ),
                "mapping_intake_status": (
                    report_mapping_intake["status"]
                    if report_mapping_intake is not None
                    else "not_materialized"
                ),
                "mapping_intake_id": (
                    report_mapping_intake["intake_id"]
                    if report_mapping_intake is not None
                    else None
                ),
                "mapping_intake_receipt_sha256": (
                    report_mapping_intake["receipt_sha256"]
                    if report_mapping_intake is not None
                    else None
                ),
                "mapping_intake_ref": (
                    _ref(
                        report_mapping_intake,
                        kind="private_report_mapping_intake",
                        schema_version=PRIVATE_REPORT_MAPPING_INTAKE_SCHEMA_VERSION,
                    )
                    if report_mapping_intake is not None
                    else None
                ),
            },
            "blocker_codes": (
                []
                if not unresolved_report_ids
                or excluded_report_ids == unresolved_report_ids
                else ["unresolved_physician_report_associations"]
            ),
        },
        {
            "check_id": "private_report_text_release",
            "status": "GO" if release_complete else "NO_GO",
            "evidence_ref": (
                _ref(
                    private_report_release,
                    kind="private_physician_report_release",
                    schema_version=PHYSICIAN_REPORT_RELEASE_SCHEMA_VERSION,
                )
                if private_report_release is not None
                else _ref(
                    deid_candidates,
                    kind="physician_report_deidentification_candidates",
                    schema_version=PRIVATE_REPORT_DEID_CANDIDATES_SCHEMA_VERSION,
                )
            ),
            "facts": {
                "candidate_count": deid_counts["candidate_count"],
                "automated_phi_scan_pass_count": deid_counts["automated_phi_scan_pass_count"],
                "manual_review_pass_count": (
                    release_counts.get("released_row_count", 0)
                    if private_report_release is not None
                    else deid_counts["manual_review_pass_count"]
                ),
                "qwen_training_release_count": (
                    release_counts.get("development_qwen_training_count", 0)
                    if private_report_release is not None
                    else deid_counts["development_qwen_training_release_count"]
                ),
                "locked_language_evaluation_release_count": (
                    release_counts.get("locked_language_evaluation_count", 0)
                    if private_report_release is not None
                    else deid_counts["locked_language_evaluation_release_count"]
                ),
                "release_complete": release_complete,
            },
            "blocker_codes": []
            if release_complete
            else [
                "manual_deidentification_review_missing",
                "development_and_evaluator_report_releases_missing",
            ],
        },
        {
            "check_id": "knowledge_authority",
            "status": "GO",
            "evidence_ref": _ref(
                knowledge,
                kind="knowledge_validation_receipt",
                schema_version="evisoz_eeg_knowledge_validation_projection_v1",
            ),
            "facts": {
                "knowledge_card_count": knowledge["knowledge_card_count"],
                "clinically_reviewed_card_count": knowledge[
                    "clinically_reviewed_card_count"
                ],
                "patient_fact_creation_allowed": knowledge[
                    "patient_fact_creation_allowed"
                ],
                "clinical_deployment_allowed": knowledge[
                    "clinical_deployment_allowed"
                ],
            },
            "blocker_codes": [],
        },
        {
            "check_id": "public_auxiliary_patient_exposure_ledger",
            "status": "PARTIAL",
            "evidence_ref": _ref(
                exposure,
                kind="public_auxiliary_exposure_projection",
                schema_version=PUBLIC_EXPOSURE_PROJECTION_SCHEMA_VERSION,
            ),
            "facts": {
                "status": exposure["status"],
                "patient_count": exposure["counts"][
                    "tusz_source_train_patient_count"
                ],
                "deepsoz_overlap_patient_count": exposure["counts"][
                    "deepsoz_source_train_overlap_patient_count"
                ],
                "tuev_overlap_patient_count": exposure["counts"][
                    "tuev_train_visible_overlap_patient_count"
                ],
                "training_authorized": exposure["permissions"][
                    "training_authorized_by_projection"
                ],
                "public_v29_tusz_crosswalk_status": (
                    crosswalk["status"] if crosswalk is not None else "not_materialized"
                ),
                "public_v29_tusz_crosswalk_patient_count": (
                    crosswalk["counts"]["v29_patient_count"]
                    if crosswalk is not None
                    else 0
                ),
                "public_v29_tusz_crosswalk_ref": (
                    _ref(
                        crosswalk,
                        kind="public_v29_tusz_crosswalk",
                        schema_version=PUBLIC_V29_TUSZ_CROSSWALK_SCHEMA_VERSION,
                    )
                    if crosswalk is not None
                    else None
                ),
                "public_auxiliary_field_release_status": (
                    field_release["status"] if field_release is not None else "not_materialized"
                ),
                "public_auxiliary_field_release_ref": (
                    _ref(
                        field_release,
                        kind="public_auxiliary_field_release",
                        schema_version=PUBLIC_AUXILIARY_FIELD_RELEASE_SCHEMA_VERSION,
                    )
                    if field_release is not None
                    else None
                ),
                "public_overlap_audit_status": (
                    public_overlap_audit["status"] if public_overlap_audit is not None else "not_materialized"
                ),
                "public_overlap_audit_ref": (
                    _ref(
                        public_overlap_audit,
                        kind="public_overlap_audit_receipt",
                        schema_version=PUBLIC_OVERLAP_AUDIT_SCHEMA_VERSION,
                    )
                    if public_overlap_audit is not None
                    else None
                ),
            },
            "blocker_codes": public_auxiliary_blockers,
        },
        {
            "check_id": "offline_teacher_and_derived_candidates",
            "status": "PARTIAL",
            "evidence_ref": _ref(
                candidate_exposure_ledger or deterministic_candidates,
                kind=(
                    "candidate_exposure_ledger"
                    if candidate_exposure_ledger is not None
                    else "deterministic_signal_candidate_materialization"
                ),
                schema_version=(
                    CANDIDATE_EXPOSURE_LEDGER_SCHEMA_VERSION
                    if candidate_exposure_ledger is not None
                    else CANDIDATE_MATERIALIZATION_SCHEMA_VERSION
                ),
            ),
            "facts": {
                "cerebragloss_candidate_count": teacher_candidate_counts.get(
                    "cerebragloss", 0
                ),
                "elm_candidate_count": teacher_candidate_counts.get("elm", 0),
                "deterministic_signal_candidate_count": deterministic_candidates[
                    "counts"
                ]["candidate_count"],
                "deterministic_signal_event_count": deterministic_candidates[
                    "counts"
                ]["event_count"],
                "deterministic_signal_candidate_concept_counts": deterministic_candidates[
                    "counts"
                ]["candidate_concept_counts"],
                "fold_local_calibration_receipt_count": deterministic_candidates[
                    "counts"
                ]["fold_local_calibration_receipt_count"],
                "node_localization_supervision_candidate_count": deterministic_candidates[
                    "counts"
                ]["node_localization_supervision_candidate_count"],
                "candidate_exposure_ledger_status": (
                    candidate_exposure_ledger["status"]
                    if candidate_exposure_ledger is not None
                    else "not_materialized"
                ),
                "candidate_exposure_ledger_id": (
                    candidate_exposure_ledger["ledger_id"]
                    if candidate_exposure_ledger is not None
                    else None
                ),
                "candidate_exposure_ledger_receipt_sha256": (
                    candidate_exposure_ledger["receipt_sha256"]
                    if candidate_exposure_ledger is not None
                    else None
                ),
                "candidate_exposure_training_authorized": (
                    candidate_exposure_ledger["permissions"]["training_authorized"]
                    if candidate_exposure_ledger is not None
                    else False
                ),
                "teacher_candidate_materialization_status": {
                    teacher_id: materialization["status"]
                    for teacher_id, materialization in sorted(
                        teacher_materializations.items()
                    )
                },
                "teacher_candidate_materialization_refs": {
                    teacher_id: _ref(
                        materialization,
                        kind="teacher_candidate_materialization",
                        schema_version=TEACHER_CANDIDATE_MATERIALIZATION_SCHEMA_VERSION,
                    )
                    for teacher_id, materialization in sorted(
                        teacher_materializations.items()
                    )
                },
            },
            "blocker_codes": sorted(set(offline_candidate_blockers)),
        },
        {
            "check_id": "findings_claim_graph_and_reports",
            "status": "QUALIFIED_GO",
            "evidence_ref": _ref(
                findings_claim_reports,
                kind="findings_claim_report_materialization",
                schema_version=FINDINGS_CLAIM_REPORT_MATERIALIZATION_SCHEMA_VERSION,
            ),
            "facts": {
                "event_findings_count": findings_claim_reports["counts"][
                    "event_findings_count"
                ],
                "reference_claim_graph_count": findings_claim_reports["counts"][
                    "reference_claim_graph_count"
                ],
                "signal_candidate_claim_graph_count": findings_claim_reports[
                    "counts"
                ]["signal_candidate_claim_graph_count"],
                "canonical_report_count": findings_claim_reports["counts"][
                    "canonical_report_count"
                ],
                "knowledge_selection_receipt_count": findings_claim_reports[
                    "counts"
                ]["knowledge_selection_receipt_count"],
                "physician_authored_report_count": findings_claim_reports[
                    "counts"
                ]["physician_authored_report_count"],
                "clinical_report_release": findings_claim_reports["permissions"][
                    "canonical_report_is_clinical_release"
                ],
            },
            "blocker_codes": [],
        },
    ]
    if bound_evidence is not None:
        checks.append(
            {
                "check_id": "bound_evidence_materialization",
                "status": "GO",
                "evidence_ref": _ref(
                    bound_evidence,
                    kind="bound_evidence_materialization",
                    schema_version=BOUND_MATERIALIZATION_SCHEMA_VERSION,
                ),
                "facts": {
                    "event_count": bound_evidence["counts"]["event_count"],
                    "development_event_count": bound_evidence["counts"]["development_event_count"],
                    "locked_test_event_count": bound_evidence["counts"]["locked_test_event_count"],
                    "training_authorized_event_count": bound_evidence["counts"]["training_authorized_event_count"],
                    "physician_report_text_released_count": bound_evidence["counts"]["physician_report_text_released_count"],
                    "shadow_only": True,
                },
                "blocker_codes": [],
            }
        )
    if clean_freeze_audit is not None:
        checks.append(
            {
                "check_id": "clean_freeze_audit",
                "status": clean_freeze_audit["status"],
                "evidence_ref": _ref(
                    clean_freeze_audit,
                    kind="holdout_freeze_audit",
                    schema_version=CLEAN_FREEZE_AUDIT_SCHEMA_VERSION,
                ),
                "facts": {
                    "git_clean": clean_freeze_audit["git_snapshot"]["clean"],
                    "contract_count": len(clean_freeze_audit["required_contracts"]),
                    "non_authorizing": clean_freeze_audit["non_authorizing"],
                },
                "blocker_codes": sorted(
                    code
                    for row in clean_freeze_audit["checks"]
                    for code in row["blocker_codes"]
                ),
            }
        )
    blocking_check_ids = sorted(
        row["check_id"]
        for row in checks
        if row["status"] in {"NO_GO", "PARTIAL", "EVALUATOR_ONLY_GO"}
    )
    overall_status = "GO" if not blocking_check_ids else "NO_GO"
    body: dict[str, Any] = {
        "schema_version": STAGE0_GATE_SCHEMA_VERSION,
        "gate_id": "CONTENT-ADDRESS-PENDING",
        "status": overall_status,
        "checks": checks,
        "blocking_check_ids": blocking_check_ids,
        "prohibited_actions": list(_PROHIBITED_ACTIONS) if overall_status == "NO_GO" else [],
        "authorized_next_actions": list(_AUTHORIZED_NEXT_ACTIONS) if overall_status == "NO_GO" else [],
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["gate_id"] = _GATE_ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_stage0_gate(body)


def validate_stage0_gate(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "gate_id",
        "status",
        "checks",
        "blocking_check_ids",
        "prohibited_actions",
        "authorized_next_actions",
        "receipt_sha256",
    }:
        raise ValueError("EviSOZ Stage-0 gate fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != STAGE0_GATE_SCHEMA_VERSION:
        raise ValueError("EviSOZ Stage-0 gate schema drifted")
    checks = data["checks"]
    if not isinstance(checks, list) or not checks:
        raise ValueError("EviSOZ Stage-0 gate checks are empty")
    ids = [row.get("check_id") for row in checks if isinstance(row, dict)]
    if len(ids) != len(checks) or len(ids) != len(set(ids)):
        raise ValueError("EviSOZ Stage-0 gate check IDs are invalid")
    for row in checks:
        if set(row) != {
            "check_id",
            "status",
            "evidence_ref",
            "facts",
            "blocker_codes",
        }:
            raise ValueError("EviSOZ Stage-0 gate check fields drifted")
        if row["status"] not in {
            "GO",
            "QUALIFIED_GO",
            "EVALUATOR_ONLY_GO",
            "PARTIAL",
            "NO_GO",
        }:
            raise ValueError("EviSOZ Stage-0 gate check status is invalid")
        if not isinstance(row["facts"], dict) or not isinstance(row["blocker_codes"], list):
            raise ValueError("EviSOZ Stage-0 gate check facts/blockers are invalid")
        if row["status"] == "GO" and row["blocker_codes"]:
            raise ValueError("EviSOZ Stage-0 GO check has blockers")
    expected_blocking = sorted(
        row["check_id"]
        for row in checks
        if row["status"] in {"NO_GO", "PARTIAL", "EVALUATOR_ONLY_GO"}
    )
    if data["blocking_check_ids"] != expected_blocking:
        raise ValueError("EviSOZ Stage-0 blocking check roster drifted")
    expected_status = "GO" if not expected_blocking else "NO_GO"
    if data["status"] != expected_status:
        raise ValueError("EviSOZ Stage-0 overall status is inconsistent")
    if expected_status == "NO_GO" and not data["prohibited_actions"]:
        raise ValueError("EviSOZ Stage-0 NO_GO lacks prohibited actions")
    expected_id = _GATE_ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["gate_id"] != expected_id:
        raise ValueError("EviSOZ Stage-0 gate ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("EviSOZ Stage-0 gate hash drifted")
    return data


__all__ = [
    "STAGE0_GATE_SCHEMA_VERSION",
    "build_stage0_gate",
    "validate_stage0_gate",
]
