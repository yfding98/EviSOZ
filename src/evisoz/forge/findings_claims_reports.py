"""Stage-0 Findings/claim/report projection for EviSOZ.

This is an adapter layer, not a second EEG measurement ontology.  It keeps
the private field release (evaluator-only), deterministic signal candidates,
offline teacher candidates (both soft auxiliary), and generated text in
separate lanes.  Knowledge cards are selected only to constrain terminology
and safety boundaries; they never add patient facts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from scripts.validate_eeg_knowledge_system import validate_knowledge_system
from src.evisoz.data.artifact_ref import (
    build_json_artifact_ref,
    canonical_json_bytes,
    canonical_json_sha256,
    validate_artifact_ref,
    verify_artifact_content,
)
from src.evisoz.data.dataset_policy import validate_field_release
from src.evisoz.data.private_stage0_cohort_materializer import (
    PRIVATE_STAGE0_COHORT_SCHEMA_VERSION,
    validate_private_stage0_cohort_artifact,
)
from src.evisoz.data.stage0_dual_montage_cache import (
    MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
)
from src.evisoz.forge.deterministic_signal_candidates import (
    CANDIDATE_CACHE_SCHEMA_VERSION,
    CANDIDATE_MATERIALIZATION_SCHEMA_VERSION,
    validate_deterministic_signal_candidate_cache,
    validate_deterministic_signal_candidate_materialization,
)
from src.evisoz.forge.teacher_candidates import (
    TEACHER_CANDIDATE_CACHE_SCHEMA_VERSION,
    TEACHER_CANDIDATE_MATERIALIZATION_SCHEMA_VERSION,
    TEACHER_IDS,
    validate_teacher_candidate_cache,
    validate_teacher_candidate_materialization,
)
from src.evisoz.forge.private_stage0_examples import (
    PRIVATE_STAGE0_EXAMPLES_SCHEMA_VERSION,
)


EVENT_FINDINGS_SCHEMA_VERSION = "evisoz_event_findings_projection_v1"
REFERENCE_GRAPH_SCHEMA_VERSION = "evisoz_reference_claim_graph_v1"
SIGNAL_GRAPH_SCHEMA_VERSION = "evisoz_signal_candidate_claim_graph_v1"
KNOWLEDGE_SELECTION_SCHEMA_VERSION = "evisoz_knowledge_selection_receipt_v1"
CANONICAL_REPORT_SCHEMA_VERSION = "evisoz_canonical_report_v1"
MATERIALIZATION_SCHEMA_VERSION = "evisoz_findings_claim_report_materialization_v1"

_HASH_PLACEHOLDER = "0" * 64
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_SIGNAL_CANDIDATE_POLICY = {
    "clinical_labels_evaluation_only": True,
    "direct_measurements_must_be_replayable": True,
    "teacher_candidates_soft_only": True,
    "derived_candidates_soft_only": True,
    "training_label_loss_allowed": False,
    "prompt_or_rag_can_receive_label_values": False,
    "signal_candidate_claims_may_be_reported_as_possible": True,
    "signal_candidate_claims_may_supervise_node_localization": False,
    "knowledge_can_create_patient_fact": False,
    "generated_text_can_create_patient_fact": False,
}
_REPORT_SAFETY = {
    "patient_facts_added": False,
    "clinical_soz_confirmed": False,
    "cortical_source_confirmed": False,
    "surgical_target_proposed": False,
    "diagnosis_generated": False,
    "treatment_recommendation_generated": False,
    "requires_physician_review": True,
}


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _finite_tree(value: object, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _id_source(value: Mapping[str, object], key: str) -> dict[str, object]:
    result = _hash_source(value)
    result[key] = _PENDING_ID
    return result


def _artifact(value: object, *, kind: str, schema: str) -> dict[str, Any]:
    return build_json_artifact_ref(
        _plain(value), artifact_kind=kind, payload_schema_version=schema
    )


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON artifact must be a regular file: {path}")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if type(value) is not dict:
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _candidate_row_sha(row: Mapping[str, object]) -> str:
    return canonical_json_sha256(_plain(row))


def _teacher_candidate_row_sha(row: Mapping[str, object]) -> str:
    """Hash the normalized teacher lane row without its provenance hash."""

    source = {
        key: row[key]
        for key in (
            "candidate_id", "teacher_id", "concept", "support_kind",
            "support_view", "support_units", "support_interval_seconds",
            "confidence", "probability_semantics",
            "source_teacher_candidate_cache_ref", "authority", "status",
            "calibration_state", "permitted_uses", "prohibited_uses",
        )
    }
    return canonical_json_sha256(_plain(source))


def _teacher_candidate_finding_rows(
    *,
    event_row: Mapping[str, object],
    teacher_candidate_cache: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    """Project a validated teacher cache into an isolated, non-label lane."""

    if teacher_candidate_cache is None:
        return []
    cache = validate_teacher_candidate_cache(dict(teacher_candidate_cache))
    if cache["event_id"] != event_row["event_id"]:
        raise ValueError("teacher candidate event binding drifted")
    if cache["linkage_group_id"] != event_row["linkage_group_id"]:
        raise ValueError("teacher candidate linkage binding drifted")
    if cache["outer_holdout_fold"] != event_row["outer_holdout_fold"]:
        raise ValueError("teacher candidate fold binding drifted")
    if event_row["evisoz_role"] != "development_cv" or cache["evisoz_role"] != "development_cv":
        raise ValueError("teacher candidates may not enter locked-test Findings")
    cache_ref = _artifact(
        cache,
        kind="teacher_candidate_cache",
        schema=TEACHER_CANDIDATE_CACHE_SCHEMA_VERSION,
    )
    rows: list[dict[str, object]] = []
    for source in cache["candidate_rows"]:
        row: dict[str, object] = {
            "candidate_id": source["candidate_id"],
            "teacher_id": cache["teacher_id"],
            "concept": source["concept"],
            "support_kind": source["support_kind"],
            "support_view": source["support_view"],
            "support_units": source["support_units"],
            "support_interval_seconds": source["support_interval_seconds"],
            "confidence": source["confidence"],
            "probability_semantics": source["probability_semantics"],
            "source_teacher_candidate_cache_ref": cache_ref,
            "authority": "offline_teacher",
            "status": "candidate_only",
            "calibration_state": "uncalibrated",
            "permitted_uses": ["soft_auxiliary"],
            "prohibited_uses": [
                "clinical_label",
                "measured_fact",
                "node_localization_supervision",
                "endpoint_expansion_from_edge",
            ],
        }
        row["row_sha256"] = _teacher_candidate_row_sha(row)
        rows.append(row)
    rows.sort(key=lambda row: (str(row["concept"]), str(row["candidate_id"])))
    return rows


def build_event_findings_projection(
    *,
    event_row: Mapping[str, object],
    training_example: Mapping[str, object],
    field_release: Mapping[str, object],
    candidate_cache: Mapping[str, object],
    teacher_candidate_cache: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Project one event into typed Findings lanes without copying new facts."""

    event_id = str(event_row["event_id"])
    linkage_group_id = str(event_row["linkage_group_id"])
    fields = [deepcopy(dict(row)) for row in field_release["fields"]]
    fields.sort(key=lambda row: str(row["field_id"]))
    clinical = [
        {
            "field_id": row["field_id"],
            "field_path": row["field_path"],
            "state": row["state"],
            "authority": row["authority"],
            "quality_tier": row["quality_tier"],
            "semantic_role": row["semantic_role"],
            "value_ref": row["value_ref"],
            "value_payload": row["value_payload"],
            "claim_permission": row["claim_permission"],
            "loss_permissions": row["loss_permissions"],
            "source_lane": "clinical_labels",
        }
        for row in fields
    ]
    candidates = []
    candidate_cache_ref = _artifact(
        candidate_cache,
        kind="deterministic_signal_candidate_cache",
        schema=CANDIDATE_CACHE_SCHEMA_VERSION,
    )
    for row in candidate_cache["candidate_rows"]:
        candidates.append(
            {
                "candidate_id": row["candidate_id"],
                "concept": row["concept"],
                "support_view": row["support_view"],
                "support_units": row["support_units"],
                "support_interval_seconds": row["support_interval_seconds"],
                "heuristic_score": row["heuristic_score"],
                "rule_id": row["rule_id"],
                "shared_electrode": row["shared_electrode"],
                "row_sha256": _candidate_row_sha(row),
                "source_candidate_cache_ref": candidate_cache_ref,
                "authority": "signal_derived",
                "status": "derived_candidate",
                "calibration_state": "uncalibrated",
                "permitted_uses": ["soft_auxiliary"],
                "prohibited_uses": [
                    "clinical_label",
                    "measured_fact",
                    "node_localization_supervision",
                ],
            }
        )
    teacher_candidates = _teacher_candidate_finding_rows(
        event_row=event_row,
        teacher_candidate_cache=teacher_candidate_cache,
    )
    candidates.sort(key=lambda row: (str(row["concept"]), str(row["candidate_id"])))
    source_refs = {
        "event_identity": training_example["artifact_refs"]["event_identity"],
        "field_release": training_example["artifact_refs"]["field_release"],
        "training_example": _artifact(
            training_example,
            kind="evisoz_training_example",
            schema=PRIVATE_STAGE0_EXAMPLES_SCHEMA_VERSION.replace(
                "private_real_stage0_examples_materialization_v1", "training_example_v1"
            ),
        ),
        "montage_derivation": training_example["artifact_refs"]["montage_derivation"],
        "dual_montage_cache": candidate_cache["dual_montage_cache_ref"],
        "deterministic_candidate_cache": candidate_cache_ref,
    }
    body: dict[str, Any] = {
        "schema_version": EVENT_FINDINGS_SCHEMA_VERSION,
        "findings_id": _PENDING_ID,
        "event_id": event_id,
        "linkage_group_id": linkage_group_id,
        "evisoz_role": event_row["evisoz_role"],
        "outer_holdout_fold": event_row["outer_holdout_fold"],
        "source_refs": source_refs,
        "lanes": {
            "clinical_labels": clinical,
            "direct_measurements": [],
            "teacher_candidates": teacher_candidates,
            "derived_candidates": candidates,
            "physician_authored_text": [],
            "generated_text": [],
        },
        "permissions": deepcopy(_SIGNAL_CANDIDATE_POLICY),
        "counts": {
            "clinical_label_rows": len(clinical),
            "direct_measurement_rows": 0,
            "teacher_candidate_rows": len(teacher_candidates),
            "derived_candidate_rows": len(candidates),
            "physician_authored_text_rows": 0,
            "generated_text_rows": 0,
        },
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["findings_id"] = "EVISOZ-FIND-" + canonical_json_sha256(
        _id_source(body, "findings_id")
    )[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_event_findings_projection(body)


def validate_event_findings_projection(value: object) -> dict[str, Any]:
    required = {
        "schema_version", "findings_id", "event_id", "linkage_group_id",
        "evisoz_role", "outer_holdout_fold", "source_refs", "lanes",
        "permissions", "counts", "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("EviSOZ event Findings fields drifted")
    data = deepcopy(value)
    _finite_tree(data)
    if data["schema_version"] != EVENT_FINDINGS_SCHEMA_VERSION:
        raise ValueError("EviSOZ event Findings schema drifted")
    if data["evisoz_role"] not in {"development_cv", "locked_test"}:
        raise ValueError("EviSOZ event Findings split role drifted")
    if not isinstance(data["event_id"], str) or not data["event_id"]:
        raise ValueError("EviSOZ event Findings event ID is invalid")
    if not isinstance(data["source_refs"], dict):
        raise ValueError("EviSOZ event Findings source refs are missing")
    for key, kind in {
        "event_identity": "event_identity",
        "field_release": "field_release",
        "training_example": "evisoz_training_example",
        "montage_derivation": "montage_derivation_receipt",
        "dual_montage_cache": "dual_montage_cache_materialization_receipt",
        "deterministic_candidate_cache": "deterministic_signal_candidate_cache",
    }.items():
        ref = validate_artifact_ref(data["source_refs"][key])
        if ref["artifact_kind"] != kind:
            raise ValueError(f"EviSOZ event Findings source ref kind drifted: {key}")
    lanes = data["lanes"]
    if type(lanes) is not dict or set(lanes) != {
        "clinical_labels", "direct_measurements", "teacher_candidates",
        "derived_candidates", "physician_authored_text", "generated_text",
    }:
        raise ValueError("EviSOZ event Findings lanes drifted")
    for key in lanes:
        if not isinstance(lanes[key], list):
            raise ValueError(f"EviSOZ event Findings lane is not an array: {key}")
    if lanes["direct_measurements"] or lanes["physician_authored_text"] or lanes["generated_text"]:
        raise ValueError("Stage-0 projection unexpectedly contains unapproved non-derived facts")
    for row in lanes["clinical_labels"]:
        if type(row) is not dict or row.get("source_lane") != "clinical_labels":
            raise ValueError("clinical label lane provenance drifted")
    for row in lanes["teacher_candidates"]:
        expected = {
            "candidate_id", "teacher_id", "concept", "support_kind",
            "support_view", "support_units", "support_interval_seconds",
            "confidence", "probability_semantics",
            "source_teacher_candidate_cache_ref", "row_sha256", "authority",
            "status", "calibration_state", "permitted_uses", "prohibited_uses",
        }
        if type(row) is not dict or set(row) != expected:
            raise ValueError("teacher candidate lane fields drifted")
        if row["teacher_id"] not in TEACHER_IDS:
            raise ValueError("teacher candidate lane teacher identity drifted")
        ref = validate_artifact_ref(row["source_teacher_candidate_cache_ref"])
        if ref["artifact_kind"] != "teacher_candidate_cache" or ref["payload_schema_version"] != TEACHER_CANDIDATE_CACHE_SCHEMA_VERSION:
            raise ValueError("teacher candidate lane source ref drifted")
        if (
            row["authority"] != "offline_teacher"
            or row["status"] != "candidate_only"
            or row["calibration_state"] != "uncalibrated"
            or row["permitted_uses"] != ["soft_auxiliary"]
            or "node_localization_supervision" not in row["prohibited_uses"]
            or "endpoint_expansion_from_edge" not in row["prohibited_uses"]
        ):
            raise ValueError("teacher candidate lane authority drifted")
        confidence = row["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("teacher candidate lane confidence drifted")
        if row["row_sha256"] != _teacher_candidate_row_sha(row):
            raise ValueError("teacher candidate lane row hash drifted")
    candidate_ids: set[str] = set()
    for row in lanes["derived_candidates"]:
        expected = {
            "candidate_id", "concept", "support_view", "support_units",
            "support_interval_seconds", "heuristic_score", "rule_id",
            "shared_electrode", "row_sha256", "source_candidate_cache_ref",
            "authority", "status", "calibration_state", "permitted_uses",
            "prohibited_uses",
        }
        if type(row) is not dict or set(row) != expected:
            raise ValueError("derived candidate lane fields drifted")
        if row["candidate_id"] in candidate_ids:
            raise ValueError("derived candidate IDs are duplicated")
        candidate_ids.add(row["candidate_id"])
        if (
            row["authority"] != "signal_derived"
            or row["status"] != "derived_candidate"
            or row["calibration_state"] != "uncalibrated"
            or row["permitted_uses"] != ["soft_auxiliary"]
            or "node_localization_supervision" not in row["prohibited_uses"]
        ):
            raise ValueError("derived candidate lane authority drifted")
        ref = validate_artifact_ref(row["source_candidate_cache_ref"])
        if ref["artifact_kind"] != "deterministic_signal_candidate_cache":
            raise ValueError("derived candidate source ref kind drifted")
        source_row = {
            key: row[key]
            for key in (
                "candidate_id", "concept", "support_view", "support_units",
                "support_interval_seconds", "heuristic_score", "rule_id",
                "shared_electrode", "authority", "status", "calibration_state",
                "permitted_uses", "prohibited_uses",
            )
        }
        if row["row_sha256"] != canonical_json_sha256(source_row):
            raise ValueError("derived candidate row hash drifted")
    if data["permissions"] != _SIGNAL_CANDIDATE_POLICY:
        raise ValueError("EviSOZ event Findings permission policy drifted")
    expected_counts = {
        "clinical_label_rows": len(lanes["clinical_labels"]),
        "direct_measurement_rows": 0,
        "teacher_candidate_rows": len(lanes["teacher_candidates"]),
        "derived_candidate_rows": len(lanes["derived_candidates"]),
        "physician_authored_text_rows": 0,
        "generated_text_rows": 0,
    }
    if data["counts"] != expected_counts:
        raise ValueError("EviSOZ event Findings counts drifted")
    expected_id = "EVISOZ-FIND-" + canonical_json_sha256(
        _id_source(data, "findings_id")
    )[:24]
    if data["findings_id"] != expected_id:
        raise ValueError("EviSOZ event Findings ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("EviSOZ event Findings receipt drifted")
    return data


def build_reference_claim_graph(findings: Mapping[str, object]) -> dict[str, Any]:
    claims: list[dict[str, object]] = []
    for row in findings["lanes"]["clinical_labels"]:
        if row["state"] != "provided" or row["value_payload"] is None:
            continue
        claim: dict[str, object] = {
            "claim_id": _PENDING_ID,
            "event_id": findings["event_id"],
            "field_id": row["field_id"],
            "field_path": row["field_path"],
            "semantic_role": row["semantic_role"],
            "state": row["state"],
            "authority": row["authority"],
            "quality_tier": row["quality_tier"],
            "value_ref": row["value_ref"],
            "value_payload": row["value_payload"],
            "claim_permission": row["claim_permission"],
            "assertion_level": "dataset_or_physician_label",
            "allowed_uses": ["evaluation_only"],
            "prohibited_uses": ["training", "prompt", "rag", "report_generation"],
            "source_field_id": row["field_id"],
        }
        claim["claim_id"] = "EVISOZ-REFCLAIM-" + canonical_json_sha256(
            _id_source(claim, "claim_id")
        )[:24]
        claims.append(claim)
    claims.sort(key=lambda row: str(row["claim_id"]))
    body: dict[str, Any] = {
        "schema_version": REFERENCE_GRAPH_SCHEMA_VERSION,
        "graph_id": _PENDING_ID,
        "graph_role": "reference_evaluator_only",
        "event_id": findings["event_id"],
        "linkage_group_id": findings["linkage_group_id"],
        "source_findings_ref": _artifact(
            findings, kind="evisoz_event_findings_projection", schema=EVENT_FINDINGS_SCHEMA_VERSION
        ),
        "claims": claims,
        "permissions": {
            "evaluator_only": True,
            "training_allowed": False,
            "prompt_or_rag_allowed": False,
            "report_generation_allowed": False,
            "may_create_signal_fact": False,
        },
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["graph_id"] = "EVISOZ-REFGRAPH-" + canonical_json_sha256(
        _id_source(body, "graph_id")
    )[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_reference_claim_graph(body)


def validate_reference_claim_graph(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version", "graph_id", "graph_role", "event_id", "linkage_group_id",
        "source_findings_ref", "claims", "permissions", "receipt_sha256",
    }:
        raise ValueError("EviSOZ reference claim graph fields drifted")
    data = deepcopy(value)
    _finite_tree(data)
    if data["schema_version"] != REFERENCE_GRAPH_SCHEMA_VERSION or data["graph_role"] != "reference_evaluator_only":
        raise ValueError("EviSOZ reference graph identity drifted")
    ref = validate_artifact_ref(data["source_findings_ref"])
    if ref["artifact_kind"] != "evisoz_event_findings_projection":
        raise ValueError("EviSOZ reference graph source kind drifted")
    if data["permissions"] != {
        "evaluator_only": True,
        "training_allowed": False,
        "prompt_or_rag_allowed": False,
        "report_generation_allowed": False,
        "may_create_signal_fact": False,
    }:
        raise ValueError("EviSOZ reference graph permission drifted")
    claim_ids: set[str] = set()
    for claim in data["claims"]:
        if type(claim) is not dict:
            raise ValueError("EviSOZ reference graph claim is invalid")
        if claim["claim_id"] in claim_ids:
            raise ValueError("EviSOZ reference graph claim IDs are duplicated")
        claim_ids.add(claim["claim_id"])
        if claim["allowed_uses"] != ["evaluation_only"] or "training" not in claim["prohibited_uses"]:
            raise ValueError("EviSOZ reference graph claim permission drifted")
        if claim["assertion_level"] != "dataset_or_physician_label":
            raise ValueError("EviSOZ reference graph assertion level drifted")
        expected_id = "EVISOZ-REFCLAIM-" + canonical_json_sha256(
            _id_source(claim, "claim_id")
        )[:24]
        if claim["claim_id"] != expected_id:
            raise ValueError("EviSOZ reference graph claim identity drifted")
    expected_id = "EVISOZ-REFGRAPH-" + canonical_json_sha256(
        _id_source(data, "graph_id")
    )[:24]
    if data["graph_id"] != expected_id:
        raise ValueError("EviSOZ reference graph ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("EviSOZ reference graph receipt drifted")
    return data


def build_signal_candidate_claim_graph(
    *,
    linkage_group_id: str,
    event_findings: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    refs = [
        _artifact(item, kind="evisoz_event_findings_projection", schema=EVENT_FINDINGS_SCHEMA_VERSION)
        for item in event_findings
    ]
    refs.sort(key=lambda ref: str(ref["content_hash"]["sha256"]))
    claims: list[dict[str, object]] = []
    concept_event_counts: Counter[str] = Counter()
    unit_counts: Counter[str] = Counter()
    for findings in sorted(event_findings, key=lambda row: str(row["event_id"])):
        candidates_with_lane = [
            (candidate, "derived_candidates")
            for candidate in findings["lanes"]["derived_candidates"]
        ] + [
            (candidate, "teacher_candidates")
            for candidate in findings["lanes"]["teacher_candidates"]
        ]
        for candidate, source_lane in candidates_with_lane:
            if source_lane == "derived_candidates":
                heuristic_score = candidate["heuristic_score"]
                rule_id = candidate["rule_id"]
                shared_electrode = candidate["shared_electrode"]
                assertion_level = "derived_candidate"
                source_cache_ref = candidate["source_candidate_cache_ref"]
            else:
                heuristic_score = candidate["confidence"]
                rule_id = f"teacher:{candidate['teacher_id']}"
                shared_electrode = False
                assertion_level = "teacher_candidate"
                source_cache_ref = candidate["source_teacher_candidate_cache_ref"]
            claim = {
                "claim_id": _PENDING_ID,
                "event_id": findings["event_id"],
                "concept": candidate["concept"],
                "support_view": candidate["support_view"],
                "support_units": candidate["support_units"],
                "support_interval_seconds": candidate["support_interval_seconds"],
                "heuristic_score": heuristic_score,
                "rule_id": rule_id,
                "shared_electrode": shared_electrode,
                "source_candidate_id": candidate["candidate_id"],
                "source_candidate_row_sha256": candidate["row_sha256"],
                "source_candidate_cache_ref": source_cache_ref,
                "source_lane": source_lane,
                "assertion_level": assertion_level,
                "calibration_state": "uncalibrated",
                "allowed_uses": ["soft_auxiliary", "candidate_shadow_report"],
                "prohibited_uses": ["clinical_label", "measured_fact", "node_localization_supervision"],
            }
            if source_lane == "teacher_candidates":
                claim.update({
                    "teacher_id": candidate["teacher_id"],
                    "confidence": candidate["confidence"],
                    "probability_semantics": candidate["probability_semantics"],
                    "authority": candidate["authority"],
                    "status": candidate["status"],
                    "prohibited_uses": [
                        "clinical_label",
                        "measured_fact",
                        "node_localization_supervision",
                        "endpoint_expansion_from_edge",
                    ],
                })
            claim["claim_id"] = "EVISOZ-SIGCLAIM-" + canonical_json_sha256(
                _id_source(claim, "claim_id")
            )[:24]
            claims.append(claim)
            concept_event_counts[str(candidate["concept"])] += 1
            unit_counts.update(str(item) for item in candidate["support_units"])
    claims.sort(key=lambda row: (str(row["event_id"]), str(row["concept"]), str(row["claim_id"])))
    body: dict[str, Any] = {
        "schema_version": SIGNAL_GRAPH_SCHEMA_VERSION,
        "graph_id": _PENDING_ID,
        "graph_role": "signal_derived_shadow",
        "linkage_group_id": linkage_group_id,
        "event_findings_refs": refs,
        "claims": claims,
        "patient_summary": {
            "event_count": len(event_findings),
            "candidate_claim_count": len(claims),
            "candidate_concept_counts": dict(sorted(concept_event_counts.items())),
            "candidate_support_unit_counts": dict(sorted(unit_counts.items())),
            "localization_state": "not_assessable_pending_calibrated_localization_model",
            "patient_level_soz_conclusion": None,
            "cross_event_conflict_policy": "do_not_average_uncalibrated_candidates_into_soz",
        },
        "permissions": {
            "signal_candidate_shadow_report_allowed": True,
            "training_allowed": False,
            "node_localization_supervision_allowed": False,
            "clinical_soz_conclusion_allowed": False,
            "knowledge_can_create_patient_fact": False,
        },
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["graph_id"] = "EVISOZ-SIGGRAPH-" + canonical_json_sha256(
        _id_source(body, "graph_id")
    )[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_signal_candidate_claim_graph(body)


def validate_signal_candidate_claim_graph(value: object) -> dict[str, Any]:
    required = {
        "schema_version", "graph_id", "graph_role", "linkage_group_id",
        "event_findings_refs", "claims", "patient_summary", "permissions",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("EviSOZ signal claim graph fields drifted")
    data = deepcopy(value)
    _finite_tree(data)
    if data["schema_version"] != SIGNAL_GRAPH_SCHEMA_VERSION or data["graph_role"] != "signal_derived_shadow":
        raise ValueError("EviSOZ signal claim graph identity drifted")
    if not isinstance(data["event_findings_refs"], list) or not data["event_findings_refs"]:
        raise ValueError("EviSOZ signal claim graph event refs are empty")
    for ref in data["event_findings_refs"]:
        if validate_artifact_ref(ref)["artifact_kind"] != "evisoz_event_findings_projection":
            raise ValueError("EviSOZ signal claim graph event ref kind drifted")
    for claim in data["claims"]:
        if claim["assertion_level"] not in {"derived_candidate", "teacher_candidate"} or claim["calibration_state"] != "uncalibrated":
            raise ValueError("EviSOZ signal claim graph assertion drifted")
        if "node_localization_supervision" not in claim["prohibited_uses"]:
            raise ValueError("EviSOZ signal claim graph localization permission drifted")
        lane = claim.get("source_lane", "derived_candidates")
        if lane == "teacher_candidates":
            if claim["assertion_level"] != "teacher_candidate":
                raise ValueError("teacher signal claim assertion lane drifted")
            if claim.get("teacher_id") not in TEACHER_IDS:
                raise ValueError("teacher signal claim identity drifted")
            if claim.get("authority") != "offline_teacher" or claim.get("status") != "candidate_only":
                raise ValueError("teacher signal claim authority drifted")
            if "endpoint_expansion_from_edge" not in claim["prohibited_uses"]:
                raise ValueError("teacher signal claim edge policy drifted")
        elif lane == "derived_candidates":
            if claim["assertion_level"] != "derived_candidate":
                raise ValueError("derived signal claim assertion lane drifted")
        else:
            raise ValueError("signal claim source lane drifted")
        expected_id = "EVISOZ-SIGCLAIM-" + canonical_json_sha256(
            _id_source(claim, "claim_id")
        )[:24]
        if claim["claim_id"] != expected_id:
            raise ValueError("EviSOZ signal claim graph claim identity drifted")
    if data["patient_summary"]["patient_level_soz_conclusion"] is not None:
        raise ValueError("Stage-0 signal graph created a SOZ conclusion")
    if data["permissions"] != {
        "signal_candidate_shadow_report_allowed": True,
        "training_allowed": False,
        "node_localization_supervision_allowed": False,
        "clinical_soz_conclusion_allowed": False,
        "knowledge_can_create_patient_fact": False,
    }:
        raise ValueError("EviSOZ signal claim graph permission drifted")
    expected_id = "EVISOZ-SIGGRAPH-" + canonical_json_sha256(
        _id_source(data, "graph_id")
    )[:24]
    if data["graph_id"] != expected_id:
        raise ValueError("EviSOZ signal claim graph ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("EviSOZ signal claim graph receipt drifted")
    return data


def _load_knowledge_bundle(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    validation = validate_knowledge_system(root)
    manifest = _read_json(root / "manifest.json")
    card_path = root / str(manifest["active_entrypoints"]["knowledge_cards"])
    cards: dict[str, dict[str, Any]] = {}
    for line in card_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        card = json.loads(line)
        cards[str(card["card_id"])] = card
    if len(cards) != validation["knowledge_card_count"]:
        raise ValueError("knowledge card count drifted during selection")
    return manifest, cards


def build_knowledge_selection_receipt(
    *,
    signal_graph: Mapping[str, object],
    knowledge_manifest: Mapping[str, object],
    cards: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    concepts = {str(row["concept"]) for row in signal_graph["claims"]}
    selected_ids: set[str] = {
        "CARD.LOC.EARLIEST_SCALP_VISIBLE_CHANGE",
        "CARD.CLIN.SOZ_EZ_BOUNDARY",
    }
    reasons = ["always_select_scalp_visible_onset_boundary", "always_select_soz_ez_boundary"]
    if "possible_phase_reversal" in concepts:
        selected_ids.update({"CARD.ELEC.PHASE_REVERSAL", "CARD.ELEC.BIPOLAR_DERIVATION"})
        reasons.append("phase_reversal_candidate_requires_shared_edge_boundary")
    if "frequency_evolution_present" in concepts:
        selected_ids.add("CARD.ICTAL.DEFINITE_EVOLUTION.ACNS2021")
        reasons.append("frequency_evolution_candidate_requires_duration_and_profile_boundary")
    if concepts.intersection({"possible_rhythmic_theta", "possible_rhythmic_delta"}):
        selected_ids.add("CARD.VAR.TEMPORAL_BENIGN_RHYTHMS")
        reasons.append("rhythmic_candidate_requires_benign_rhythm_differential")
    if signal_graph["patient_summary"]["event_count"] > 1:
        selected_ids.add("CARD.LOC.MULTI_EVENT_CONCORDANCE")
        reasons.append("multi_event_graph_requires_concordance_boundary")
    if "possible_attenuation" in concepts or "possible_LVFA" in concepts:
        reasons.append("attenuation_or_fast_activity_remains_candidate_only")
    missing = sorted(selected_ids.difference(cards))
    if missing:
        raise ValueError(f"knowledge selection cards missing: {missing}")
    selected_cards: list[dict[str, object]] = []
    source_ids: set[str] = set()
    for card_id in sorted(selected_ids):
        card = cards[card_id]
        role = "safety_boundary" if card["claim_type"] == "safety_boundary" or card_id == "CARD.ELEC.PHASE_REVERSAL" else (
            "differential" if card["claim_type"] == "differential" else "primary"
        )
        selected_cards.append({
            "card_id": card_id,
            "card_sha256": hashlib.sha256(canonical_json_bytes(card)).hexdigest(),
            "role": role,
            "score": 1.0 if role == "safety_boundary" else 0.8,
        })
        source_ids.update(str(row["source_id"]) for row in card["source_refs"])
    base = {
        "schema_version": "eeg_knowledge_selection_receipt_v2",
        "knowledge_version": knowledge_manifest["knowledge_version"],
        "bundle_sha256": knowledge_manifest["active_bundle_sha256"],
        "profile_id": "evisoz_signal_candidate_shadow_v1",
        "query_fact_ids": sorted(str(row["claim_id"]) for row in signal_graph["claims"]),
        "selected_cards": selected_cards,
        "selected_source_ids": sorted(source_ids),
        "selection_reasons": reasons,
        "patient_fact_creation_allowed": False,
    }
    graph_ref = _artifact(signal_graph, kind="evisoz_signal_candidate_claim_graph", schema=SIGNAL_GRAPH_SCHEMA_VERSION)
    body: dict[str, Any] = {
        "schema_version": KNOWLEDGE_SELECTION_SCHEMA_VERSION,
        "selection_id": _PENDING_ID,
        "graph_ref": graph_ref,
        "knowledge_manifest_sha256": knowledge_manifest["active_bundle_sha256"],
        "knowledge_version": knowledge_manifest["knowledge_version"],
        "profile_id": "evisoz_signal_candidate_shadow_v1",
        "query_claim_ids": base["query_fact_ids"],
        "selected_cards": selected_cards,
        "selected_source_ids": sorted(source_ids),
        "selection_reasons": reasons,
        "patient_fact_creation_allowed": False,
        "can_add_patient_fact": False,
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["selection_id"] = "EVISOZ-KNOWSEL-" + canonical_json_sha256(
        _id_source(body, "selection_id")
    )[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_knowledge_selection_receipt(body, trusted_graph=signal_graph)


def validate_knowledge_selection_receipt(
    value: object,
    *,
    trusted_graph: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version", "selection_id", "graph_ref", "knowledge_manifest_sha256",
        "knowledge_version", "profile_id", "query_claim_ids", "selected_cards",
        "selected_source_ids", "selection_reasons", "patient_fact_creation_allowed",
        "can_add_patient_fact", "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("EviSOZ knowledge selection fields drifted")
    data = deepcopy(value)
    _finite_tree(data)
    if data["schema_version"] != KNOWLEDGE_SELECTION_SCHEMA_VERSION:
        raise ValueError("EviSOZ knowledge selection schema drifted")
    graph_ref = validate_artifact_ref(data["graph_ref"])
    if graph_ref["artifact_kind"] != "evisoz_signal_candidate_claim_graph":
        raise ValueError("EviSOZ knowledge selection graph kind drifted")
    if trusted_graph is not None:
        verify_artifact_content(graph_ref, trusted_graph)
        expected = sorted(str(row["claim_id"]) for row in trusted_graph["claims"])
        if data["query_claim_ids"] != expected:
            raise ValueError("EviSOZ knowledge query claims drifted")
    if data["patient_fact_creation_allowed"] is not False or data["can_add_patient_fact"] is not False:
        raise ValueError("EviSOZ knowledge selection can create patient facts")
    if not data["selected_cards"] or not data["selected_source_ids"]:
        raise ValueError("EviSOZ knowledge selection is empty")
    for card in data["selected_cards"]:
        if set(card) != {"card_id", "card_sha256", "role", "score"}:
            raise ValueError("EviSOZ selected card fields drifted")
        if card["role"] not in {"primary", "differential", "safety_boundary"}:
            raise ValueError("EviSOZ selected card role drifted")
    expected_id = "EVISOZ-KNOWSEL-" + canonical_json_sha256(
        _id_source(data, "selection_id")
    )[:24]
    if data["selection_id"] != expected_id:
        raise ValueError("EviSOZ knowledge selection ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("EviSOZ knowledge selection receipt drifted")
    return data


def _top_claims_by_event(graph: Mapping[str, object], event_id: str) -> list[Mapping[str, object]]:
    rows = [row for row in graph["claims"] if row["event_id"] == event_id]
    rows = sorted(rows, key=lambda row: (-float(row["heuristic_score"]), str(row["concept"]), str(row["claim_id"])))
    selected: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        concept = str(row["concept"])
        if concept in seen:
            continue
        seen.add(concept)
        selected.append(row)
        if len(selected) >= 8:
            break
    return selected


def build_canonical_report(
    *,
    signal_graph: Mapping[str, object],
    knowledge_selection: Mapping[str, object],
) -> dict[str, Any]:
    all_claim_ids = [str(row["claim_id"]) for row in signal_graph["claims"]]
    card_ids = [str(row["card_id"]) for row in knowledge_selection["selected_cards"]]
    sections: list[dict[str, object]] = [
        {
            "section_id": "analysis_scope",
            "text_zh": (
                f"本研究性投影基于 {signal_graph['patient_summary']['event_count']} 次已知发作片段，"
                "不执行连续脑电发作检测；下述内容仅为未校准的信号规则候选。"
            ),
            "claim_ids": [],
            "knowledge_card_ids": [],
        },
        {
            "section_id": "technical_conditions",
            "text_zh": "输入同时保留 Standard19-CAR 电极视图与 signed TCP22 双极边视图；缺失支持不会被补写为存在。",
            "claim_ids": [],
            "knowledge_card_ids": [item for item in card_ids if item in {"CARD.ELEC.BIPOLAR_DERIVATION", "CARD.LOC.MISSING_COVERAGE"}],
        },
    ]
    event_ids = sorted({str(row["event_id"]) for row in signal_graph["claims"]})
    for ordinal, event_id in enumerate(event_ids, start=1):
        selected = _top_claims_by_event(signal_graph, event_id)
        fragments = []
        claim_ids = []
        for claim in selected:
            supports = "、".join(str(item) for item in claim["support_units"])
            fragments.append(f"{claim['concept']}（{supports}，规则分数 {float(claim['heuristic_score']):.3g}）")
            claim_ids.append(str(claim["claim_id"]))
        text = (
            f"第 {ordinal} 次事件（{event_id}）触发的信号候选："
            + ("；".join(fragments) if fragments else "未触发本候选规则")
            + "。这些条目不是临床标签，也不等同于 SOZ。"
        )
        sections.append({
            "section_id": f"event_{ordinal}_findings",
            "text_zh": text,
            "claim_ids": claim_ids,
            "knowledge_card_ids": card_ids,
        })
    sections.extend([
        {
            "section_id": "cross_event_summary",
            "text_zh": "跨事件仅汇总候选触发的重复性，不把未校准候选平均或升级为患者级 SOZ 结论。",
            "claim_ids": all_claim_ids,
            "knowledge_card_ids": [item for item in card_ids if item == "CARD.LOC.MULTI_EVENT_CONCORDANCE"],
        },
        {
            "section_id": "impression",
            "text_zh": "当前产物只形成 signal-derived shadow evidence，尚未形成可校准的头皮起始定位或非侵入性 SOZ 假设；需后续定位模型和神经电生理医师复核。",
            "claim_ids": [],
            "knowledge_card_ids": [item for item in card_ids if item in {"CARD.LOC.EARLIEST_SCALP_VISIBLE_CHANGE", "CARD.CLIN.SOZ_EZ_BOUNDARY"}],
        },
        {
            "section_id": "limitations",
            "text_zh": "仅依据头皮 EEG 的研究性候选；未结合 iEEG、MRI、PET、手术范围或术后结局。相位反转、衰减、低电压快活动、节律和频率变化均需独立资格与伪迹鉴别，不能证明皮层源、致痫区或手术靶点。",
            "claim_ids": [],
            "knowledge_card_ids": [item for item in card_ids if item in {"CARD.CLIN.SOZ_EZ_BOUNDARY", "CARD.ELEC.PHASE_REVERSAL"}],
        },
    ])
    body: dict[str, Any] = {
        "schema_version": CANONICAL_REPORT_SCHEMA_VERSION,
        "report_id": _PENDING_ID,
        "linkage_group_id": signal_graph["linkage_group_id"],
        "report_scope": "signal_candidate_shadow",
        "status": "research_shadow_not_clinical",
        "source_graph_ref": _artifact(signal_graph, kind="evisoz_signal_candidate_claim_graph", schema=SIGNAL_GRAPH_SCHEMA_VERSION),
        "knowledge_selection_ref": _artifact(knowledge_selection, kind="evisoz_knowledge_selection_receipt", schema=KNOWLEDGE_SELECTION_SCHEMA_VERSION),
        "sections": sections,
        "safety": deepcopy(_REPORT_SAFETY),
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["report_id"] = "EVISOZ-REPORT-" + canonical_json_sha256(
        _id_source(body, "report_id")
    )[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_canonical_report(body, trusted_graph=signal_graph, trusted_selection=knowledge_selection)


def validate_canonical_report(
    value: object,
    *,
    trusted_graph: Mapping[str, object] | None = None,
    trusted_selection: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version", "report_id", "linkage_group_id", "report_scope", "status",
        "source_graph_ref", "knowledge_selection_ref", "sections", "safety", "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("EviSOZ canonical report fields drifted")
    data = deepcopy(value)
    _finite_tree(data)
    if data["schema_version"] != CANONICAL_REPORT_SCHEMA_VERSION or data["report_scope"] != "signal_candidate_shadow" or data["status"] != "research_shadow_not_clinical":
        raise ValueError("EviSOZ canonical report identity drifted")
    graph_ref = validate_artifact_ref(data["source_graph_ref"])
    selection_ref = validate_artifact_ref(data["knowledge_selection_ref"])
    if graph_ref["artifact_kind"] != "evisoz_signal_candidate_claim_graph" or selection_ref["artifact_kind"] != "evisoz_knowledge_selection_receipt":
        raise ValueError("EviSOZ canonical report source kind drifted")
    if trusted_graph is not None:
        verify_artifact_content(graph_ref, trusted_graph)
    if trusted_selection is not None:
        verify_artifact_content(selection_ref, trusted_selection)
    if data["safety"] != _REPORT_SAFETY:
        raise ValueError("EviSOZ canonical report safety policy drifted")
    required_sections = {"analysis_scope", "technical_conditions", "cross_event_summary", "impression", "limitations"}
    section_ids = [str(row.get("section_id")) for row in data["sections"]]
    if not required_sections.issubset(section_ids) or len(section_ids) != len(set(section_ids)):
        raise ValueError("EviSOZ canonical report section roster drifted")
    for section in data["sections"]:
        if type(section) is not dict or set(section) != {"section_id", "text_zh", "claim_ids", "knowledge_card_ids"}:
            raise ValueError("EviSOZ canonical report section fields drifted")
        if not isinstance(section["text_zh"], str) or not section["text_zh"]:
            raise ValueError("EviSOZ canonical report section text is empty")
    expected_id = "EVISOZ-REPORT-" + canonical_json_sha256(
        _id_source(data, "report_id")
    )[:24]
    if data["report_id"] != expected_id:
        raise ValueError("EviSOZ canonical report ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("EviSOZ canonical report receipt drifted")
    return data


def _materialization_id_source(value: Mapping[str, object]) -> dict[str, object]:
    return _id_source(value, "materialization_id")


def build_findings_claim_report_materialization(
    *,
    private_examples_root: str | Path,
    deterministic_candidates_root: str | Path,
    knowledge_root: str | Path,
    output: str | Path,
    teacher_candidates_root: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize event Findings, evaluator graphs, shadow graphs and reports."""

    examples_root = Path(private_examples_root).resolve(strict=True)
    candidates_root = Path(deterministic_candidates_root).resolve(strict=True)
    knowledge_path = Path(knowledge_root).resolve(strict=True)
    examples_manifest = _read_json(examples_root / "manifest.json")
    if examples_manifest.get("schema_version") != PRIVATE_STAGE0_EXAMPLES_SCHEMA_VERSION:
        raise ValueError("private examples manifest schema drifted")
    candidates_manifest = _read_json(candidates_root / "manifest.json")
    validate_deterministic_signal_candidate_materialization(
        candidates_manifest,
        output_root=candidates_root,
        replay_features=False,
    )
    teacher_cache_by_event: dict[str, dict[str, Any]] = {}
    teacher_materialization: dict[str, Any] | None = None
    teacher_root: Path | None = None
    if teacher_candidates_root is not None:
        teacher_root = Path(teacher_candidates_root).resolve(strict=True)
        teacher_materialization = _read_json(teacher_root / "manifest.json")
        validate_teacher_candidate_materialization(
            teacher_materialization,
            output_root=str(teacher_root),
        )
        for row in teacher_materialization["events"]:
            event_id = str(row["event_id"])
            if event_id in teacher_cache_by_event:
                raise ValueError(f"duplicate teacher candidate event: {event_id}")
            cache_path = _safe_candidate_path(teacher_root, row["relative_cache_path"])
            cache = _read_json(cache_path)
            checked = validate_teacher_candidate_cache(cache)
            if (
                checked["event_id"] != event_id
                or checked["linkage_group_id"] != row["linkage_group_id"]
                or checked["outer_holdout_fold"] != row["outer_holdout_fold"]
            ):
                raise ValueError("teacher candidate materialization event binding drifted")
            teacher_cache_by_event[event_id] = checked
    knowledge_manifest, cards = _load_knowledge_bundle(knowledge_path)
    event_findings: dict[str, dict[str, Any]] = {}
    reference_graphs: dict[str, dict[str, Any]] = {}
    candidate_event_rows = {
        str(row["event_id"]): row for row in candidates_manifest["events"]
    }
    for event_row in examples_manifest["events"]:
        event_id = str(event_row["event_id"])
        if event_id not in candidate_event_rows:
            raise ValueError(f"missing deterministic candidate event: {event_id}")
        event_root = examples_root / "events" / event_id
        field_release = _read_json(event_root / "field_release.json")
        training_example = _read_json(event_root / "training_example.json")
        validate_field_release(field_release)
        candidate_path = _safe_candidate_path(candidates_root, candidate_event_rows[event_id]["relative_candidate_cache_path"])
        candidate_cache = _read_json(candidate_path)
        validate_deterministic_signal_candidate_cache(candidate_cache)
        findings = build_event_findings_projection(
            event_row=event_row,
            training_example=training_example,
            field_release=field_release,
            candidate_cache=candidate_cache,
            teacher_candidate_cache=teacher_cache_by_event.get(event_id),
        )
        graph = build_reference_claim_graph(findings)
        event_findings[event_id] = findings
        reference_graphs[event_id] = graph

    patients: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event_id, findings in event_findings.items():
        patients[str(findings["linkage_group_id"])].append(findings)
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    committed = False
    try:
        event_rows: list[dict[str, object]] = []
        patient_rows: list[dict[str, object]] = []
        event_graph_refs: dict[str, dict[str, object]] = {}
        for event_id in sorted(event_findings):
            findings = event_findings[event_id]
            graph = reference_graphs[event_id]
            event_dir = staging / "events" / event_id
            event_dir.mkdir(parents=True, exist_ok=False)
            (event_dir / "findings.json").write_bytes(canonical_json_bytes(findings))
            (event_dir / "reference_claim_graph.json").write_bytes(canonical_json_bytes(graph))
            findings_ref = _artifact(findings, kind="evisoz_event_findings_projection", schema=EVENT_FINDINGS_SCHEMA_VERSION)
            reference_ref = _artifact(graph, kind="evisoz_reference_claim_graph", schema=REFERENCE_GRAPH_SCHEMA_VERSION)
            event_graph_refs[event_id] = reference_ref
            source = next(row for row in examples_manifest["events"] if str(row["event_id"]) == event_id)
            event_rows.append({
                "event_id": event_id,
                "linkage_group_id": findings["linkage_group_id"],
                "evisoz_role": source["evisoz_role"],
                "outer_holdout_fold": source["outer_holdout_fold"],
                "findings_ref": findings_ref,
                "reference_claim_graph_ref": reference_ref,
                "relative_findings_path": f"events/{event_id}/findings.json",
                "relative_reference_claim_graph_path": f"events/{event_id}/reference_claim_graph.json",
                "derived_candidate_count": findings["counts"]["derived_candidate_rows"],
                "reference_claim_count": len(graph["claims"]),
            })
        for linkage_group_id in sorted(patients):
            findings_list = sorted(patients[linkage_group_id], key=lambda row: str(row["event_id"]))
            graph = build_signal_candidate_claim_graph(
                linkage_group_id=linkage_group_id, event_findings=findings_list
            )
            selection = build_knowledge_selection_receipt(
                signal_graph=graph,
                knowledge_manifest=knowledge_manifest,
                cards=cards,
            )
            report = build_canonical_report(signal_graph=graph, knowledge_selection=selection)
            patient_dir = staging / "patients" / linkage_group_id
            patient_dir.mkdir(parents=True, exist_ok=False)
            (patient_dir / "signal_candidate_claim_graph.json").write_bytes(canonical_json_bytes(graph))
            (patient_dir / "knowledge_selection.json").write_bytes(canonical_json_bytes(selection))
            (patient_dir / "canonical_report.json").write_bytes(canonical_json_bytes(report))
            patient_rows.append({
                "linkage_group_id": linkage_group_id,
                "event_count": len(findings_list),
                "signal_candidate_claim_graph_ref": _artifact(graph, kind="evisoz_signal_candidate_claim_graph", schema=SIGNAL_GRAPH_SCHEMA_VERSION),
                "knowledge_selection_ref": _artifact(selection, kind="evisoz_knowledge_selection_receipt", schema=KNOWLEDGE_SELECTION_SCHEMA_VERSION),
                "canonical_report_ref": _artifact(report, kind="evisoz_canonical_report", schema=CANONICAL_REPORT_SCHEMA_VERSION),
                "relative_signal_candidate_claim_graph_path": f"patients/{linkage_group_id}/signal_candidate_claim_graph.json",
                "relative_knowledge_selection_path": f"patients/{linkage_group_id}/knowledge_selection.json",
                "relative_canonical_report_path": f"patients/{linkage_group_id}/canonical_report.json",
                "candidate_claim_count": len(graph["claims"]),
            })
        event_rows.sort(key=lambda row: str(row["event_id"]))
        patient_rows.sort(key=lambda row: str(row["linkage_group_id"]))
        manifest: dict[str, Any] = {
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "materialization_id": _PENDING_ID,
            "status": "complete_signal_shadow_evaluator_reference_materialization",
            "source_refs": {
                "private_examples_manifest": _artifact(examples_manifest, kind="evisoz_private_real_examples_manifest", schema=PRIVATE_STAGE0_EXAMPLES_SCHEMA_VERSION),
                "deterministic_candidates_manifest": _artifact(candidates_manifest, kind="deterministic_signal_candidate_materialization", schema=CANDIDATE_MATERIALIZATION_SCHEMA_VERSION),
                "knowledge_manifest": _artifact(knowledge_manifest, kind="eeg_knowledge_manifest", schema="eeg_external_knowledge_manifest_v2"),
            },
            "event_rows": event_rows,
            "patient_rows": patient_rows,
            "counts": {
                "event_findings_count": len(event_rows),
                "reference_claim_graph_count": len(event_rows),
                "signal_candidate_claim_graph_count": len(patient_rows),
                "knowledge_selection_receipt_count": len(patient_rows),
                "canonical_report_count": len(patient_rows),
                "physician_authored_report_count": 0,
                "generated_text_fact_count": 0,
                "reference_claim_count": sum(int(row["reference_claim_count"]) for row in event_rows),
                "signal_candidate_claim_count": sum(int(row["candidate_claim_count"]) for row in patient_rows),
            },
            "permissions": {
                "reference_claim_graph_training_allowed": False,
                "reference_claim_graph_prompt_or_rag_allowed": False,
                "signal_candidate_claim_graph_training_allowed": False,
                "signal_candidate_claim_graph_node_localization_supervision_allowed": False,
                "knowledge_can_create_patient_fact": False,
                "canonical_report_is_clinical_release": False,
                "physician_text_training_allowed": False,
            },
            "receipt_sha256": _HASH_PLACEHOLDER,
        }
        if teacher_materialization is not None:
            manifest["source_refs"]["teacher_candidate_materialization"] = _artifact(
                teacher_materialization,
                kind="teacher_candidate_materialization",
                schema=TEACHER_CANDIDATE_MATERIALIZATION_SCHEMA_VERSION,
            )
        manifest["materialization_id"] = "EVISOZ-FCR-" + canonical_json_sha256(
            _materialization_id_source(manifest)
        )[:24]
        manifest["receipt_sha256"] = canonical_json_sha256(_hash_source(manifest))
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        validate_findings_claim_report_materialization(manifest, output_root=staging)
        os.rename(staging, destination)
        committed = True
        return manifest
    finally:
        if not committed:
            shutil.rmtree(staging, ignore_errors=True)


def _safe_candidate_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise TypeError("candidate path must be a string")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or not parsed.parts or ".." in parsed.parts:
        raise ValueError("candidate path is unsafe")
    path = root.joinpath(*parsed.parts)
    resolved = path.resolve(strict=True)
    resolved.relative_to(root.resolve(strict=True))
    if path.is_symlink() or not path.is_file():
        raise ValueError("candidate cache must be a regular file")
    return resolved


def validate_findings_claim_report_materialization(
    value: object,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version", "materialization_id", "status", "source_refs", "event_rows",
        "patient_rows", "counts", "permissions", "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("EviSOZ Findings/claim/report materialization fields drifted")
    data = deepcopy(value)
    _finite_tree(data)
    if data["schema_version"] != MATERIALIZATION_SCHEMA_VERSION or data["status"] != "complete_signal_shadow_evaluator_reference_materialization":
        raise ValueError("EviSOZ Findings/claim/report materialization identity drifted")
    for ref in data["source_refs"].values():
        validate_artifact_ref(ref)
    if data["permissions"] != {
        "reference_claim_graph_training_allowed": False,
        "reference_claim_graph_prompt_or_rag_allowed": False,
        "signal_candidate_claim_graph_training_allowed": False,
        "signal_candidate_claim_graph_node_localization_supervision_allowed": False,
        "knowledge_can_create_patient_fact": False,
        "canonical_report_is_clinical_release": False,
        "physician_text_training_allowed": False,
    }:
        raise ValueError("EviSOZ Findings/claim/report permissions drifted")
    root = Path(output_root).resolve(strict=True) if output_root is not None else None
    expected_files = {"manifest.json"}
    event_count = 0
    patient_count = 0
    reference_claim_count = 0
    signal_claim_count = 0
    for row in data["event_rows"]:
        if type(row) is not dict:
            raise ValueError("EviSOZ event materialization row is invalid")
        findings = graph = None
        if root is not None:
            findings_path = _safe_candidate_path(root, row["relative_findings_path"])
            graph_path = _safe_candidate_path(root, row["relative_reference_claim_graph_path"])
            expected_files.update({str(PurePosixPath(row["relative_findings_path"])), str(PurePosixPath(row["relative_reference_claim_graph_path"]))})
            findings = validate_event_findings_projection(_read_json(findings_path))
            graph = validate_reference_claim_graph(_read_json(graph_path))
            verify_artifact_content(row["findings_ref"], findings)
            verify_artifact_content(row["reference_claim_graph_ref"], graph)
        if findings is not None and graph is not None:
            event_count += 1
            reference_claim_count += len(graph["claims"])
            if row["derived_candidate_count"] != findings["counts"]["derived_candidate_rows"] or row["reference_claim_count"] != len(graph["claims"]):
                raise ValueError("EviSOZ event materialization summary drifted")
    for row in data["patient_rows"]:
        if root is not None:
            graph_path = _safe_candidate_path(root, row["relative_signal_candidate_claim_graph_path"])
            selection_path = _safe_candidate_path(root, row["relative_knowledge_selection_path"])
            report_path = _safe_candidate_path(root, row["relative_canonical_report_path"])
            expected_files.update({str(PurePosixPath(row["relative_signal_candidate_claim_graph_path"])), str(PurePosixPath(row["relative_knowledge_selection_path"])), str(PurePosixPath(row["relative_canonical_report_path"]))})
            graph = validate_signal_candidate_claim_graph(_read_json(graph_path))
            selection = validate_knowledge_selection_receipt(_read_json(selection_path))
            report = validate_canonical_report(_read_json(report_path))
            verify_artifact_content(row["signal_candidate_claim_graph_ref"], graph)
            verify_artifact_content(row["knowledge_selection_ref"], selection)
            verify_artifact_content(row["canonical_report_ref"], report)
            patient_count += 1
            signal_claim_count += len(graph["claims"])
            if row["candidate_claim_count"] != len(graph["claims"]):
                raise ValueError("EviSOZ patient materialization summary drifted")
    if root is not None:
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        if actual != expected_files:
            raise ValueError("EviSOZ Findings/claim/report file inventory drifted")
    expected_counts = {
        "event_findings_count": len(data["event_rows"]) if root is None else event_count,
        "reference_claim_graph_count": len(data["event_rows"]),
        "signal_candidate_claim_graph_count": len(data["patient_rows"]),
        "knowledge_selection_receipt_count": len(data["patient_rows"]),
        "canonical_report_count": len(data["patient_rows"]),
        "physician_authored_report_count": 0,
        "generated_text_fact_count": 0,
        "reference_claim_count": data["counts"]["reference_claim_count"] if root is None else reference_claim_count,
        "signal_candidate_claim_count": data["counts"]["signal_candidate_claim_count"] if root is None else signal_claim_count,
    }
    # In memory validation still checks the declared counters against roster
    # cardinalities; disk validation additionally recomputes the claim totals.
    if data["counts"]["event_findings_count"] != len(data["event_rows"]) or data["counts"]["reference_claim_graph_count"] != len(data["event_rows"]):
        raise ValueError("EviSOZ event Findings denominator drifted")
    if data["counts"]["signal_candidate_claim_graph_count"] != len(data["patient_rows"]) or data["counts"]["canonical_report_count"] != len(data["patient_rows"]):
        raise ValueError("EviSOZ patient report denominator drifted")
    if root is not None and data["counts"] != expected_counts:
        raise ValueError("EviSOZ Findings/claim/report counts drifted")
    expected_id = "EVISOZ-FCR-" + canonical_json_sha256(
        _materialization_id_source(data)
    )[:24]
    if data["materialization_id"] != expected_id:
        raise ValueError("EviSOZ Findings/claim/report materialization ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("EviSOZ Findings/claim/report materialization receipt drifted")
    return data


__all__ = [
    "EVENT_FINDINGS_SCHEMA_VERSION",
    "REFERENCE_GRAPH_SCHEMA_VERSION",
    "SIGNAL_GRAPH_SCHEMA_VERSION",
    "KNOWLEDGE_SELECTION_SCHEMA_VERSION",
    "CANONICAL_REPORT_SCHEMA_VERSION",
    "MATERIALIZATION_SCHEMA_VERSION",
    "build_event_findings_projection",
    "validate_event_findings_projection",
    "build_reference_claim_graph",
    "validate_reference_claim_graph",
    "build_signal_candidate_claim_graph",
    "validate_signal_candidate_claim_graph",
    "build_knowledge_selection_receipt",
    "validate_knowledge_selection_receipt",
    "build_canonical_report",
    "validate_canonical_report",
    "build_findings_claim_report_materialization",
    "validate_findings_claim_report_materialization",
]
