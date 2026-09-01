"""Bind existing EviSOZ artifacts into one training/report input envelope.

This is a reference-only join: it stores content-addressed references to the
already validated event envelope, Findings, claim graphs, knowledge selection,
and canonical shadow report.  It deliberately does not copy or promote their
patient facts, teacher candidates, or physician text.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from pathlib import PurePosixPath
import tempfile
from typing import Any, Mapping

from src.evisoz.data.artifact_ref import (
    build_json_artifact_ref,
    canonical_json_sha256,
    validate_artifact_ref,
    verify_artifact_content,
)
from src.evisoz.forge.findings_claims_reports import (
    CANONICAL_REPORT_SCHEMA_VERSION,
    EVENT_FINDINGS_SCHEMA_VERSION,
    KNOWLEDGE_SELECTION_SCHEMA_VERSION,
    REFERENCE_GRAPH_SCHEMA_VERSION,
    SIGNAL_GRAPH_SCHEMA_VERSION,
    validate_canonical_report,
    validate_event_findings_projection,
    validate_knowledge_selection_receipt,
    validate_reference_claim_graph,
    validate_signal_candidate_claim_graph,
)
from src.evisoz.forge.private_stage0_examples import (
    PRIVATE_STAGE0_EXAMPLES_SCHEMA_VERSION,
)
from src.evisoz.forge.training_example import (
    TRAINING_EXAMPLE_SCHEMA_VERSION,
    validate_training_example,
)
from src.evisoz.data.event_identity import validate_event_identity
from src.evisoz.data.private_stage0_split import build_private_patient_linkage_group
from src.evisoz.data.split_ledger import validate_split_roster
from src.evisoz.data.tcp22_views import validate_montage_derivation_receipt
from src.evisoz.data.dataset_policy import validate_field_release


BOUND_EVIDENCE_SCHEMA_VERSION = "evisoz_bound_evidence_example_v1"
BOUND_MATERIALIZATION_SCHEMA_VERSION = "evisoz_bound_evidence_materialization_v1"
PHYSICIAN_REPORT_RELEASE_SCHEMA_VERSION = "evisoz_private_physician_report_release_v1"
_HASH_PLACEHOLDER = "0" * 64
_ID_PREFIX = "EVISOZ-BOUND-"
_MATERIALIZATION_ID_PREFIX = "EVISOZ-BOUNDEVID-"


_PERMISSIONS = {
    "training_allowed": False,
    "node_localization_supervision_allowed": False,
    "report_text_loss_allowed": False,
    "prompt_or_rag_allowed": False,
    "knowledge_can_create_patient_fact": False,
    "generated_text_can_create_patient_fact": False,
}


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _id_source(value: Mapping[str, object], key: str = "bound_example_id") -> dict[str, object]:
    body = _hash_source(value)
    body[key] = "CONTENT-ADDRESS-PENDING"
    return body


def _ref(value: Mapping[str, object], *, kind: str, schema: str) -> dict[str, Any]:
    return build_json_artifact_ref(
        value,
        artifact_kind=kind,
        payload_schema_version=schema,
    )


def _validate_training_example_minimal(value: object) -> dict[str, Any]:
    """Validate a persisted example when trusted source payloads are unavailable.

    The materializer normally supplies the full trusted context below.  This
    fallback keeps the pure builder useful for contract tests while still
    checking the closed envelope identity, receipt and all artifact kinds.
    """

    if type(value) is not dict:
        raise TypeError("training example must be an object")
    data = deepcopy(value)
    required = {
        "schema_version", "example_id", "sample_id", "event_id", "dataset_id",
        "linkage_group_id", "anchor", "split_assignment", "report_scope",
        "artifact_refs", "field_state_counts", "unavailable_field_ids",
        "enabled_loss_ports", "safety_contract", "receipt_sha256",
    }
    if set(data) != required or data["schema_version"] != TRAINING_EXAMPLE_SCHEMA_VERSION:
        raise ValueError("training example envelope fields drifted")
    if not isinstance(data["example_id"], str) or not data["example_id"].startswith("EVISOZ-EX-"):
        raise ValueError("training example ID is invalid")
    refs = data["artifact_refs"]
    if type(refs) is not dict or set(refs) != {"event_identity", "split_roster", "montage_derivation", "field_release"}:
        raise ValueError("training example artifact refs drifted")
    expected_kinds = {
        "event_identity": "event_identity",
        "split_roster": "split_roster",
        "montage_derivation": "montage_derivation_receipt",
        "field_release": "field_release",
    }
    for key, kind in expected_kinds.items():
        ref = validate_artifact_ref(refs[key])
        if ref["artifact_kind"] != kind:
            raise ValueError(f"training example artifact kind drifted: {key}")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("training example receipt drifted")
    return data


def _safe_json_path(root: Path, relative: object) -> Path:
    """Resolve a manifest path without allowing traversal or symlinks."""

    if not isinstance(relative, str):
        raise TypeError("manifest relative path must be a string")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or not parsed.parts or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError("manifest relative path is unsafe")
    candidate = root.joinpath(*parsed.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("manifest JSON artifact missing") from exc
    resolved.relative_to(root.resolve(strict=True))
    if candidate.is_symlink() or not resolved.is_file():
        raise ValueError("manifest JSON artifact must be a regular file")
    return resolved


def build_bound_evidence_example(
    *,
    training_example: Mapping[str, object],
    event_findings: Mapping[str, object],
    reference_claim_graph: Mapping[str, object],
    signal_candidate_claim_graph: Mapping[str, object] | None = None,
    knowledge_selection: Mapping[str, object] | None = None,
    canonical_report: Mapping[str, object] | None = None,
    training_example_ref: Mapping[str, object] | None = None,
    training_validation_context: Mapping[str, object] | None = None,
    dual_montage_cache_ref: Mapping[str, object] | None = None,
    physician_report_release: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Create one content-addressed, Stage-0 shadow evidence envelope."""

    if training_validation_context is None:
        base = _validate_training_example_minimal(training_example)
    else:
        context = dict(training_validation_context)
        base = validate_training_example(dict(training_example), **context)
    findings = validate_event_findings_projection(dict(event_findings))
    reference = validate_reference_claim_graph(dict(reference_claim_graph))
    if findings["event_id"] != base["event_id"] or reference["event_id"] != base["event_id"]:
        raise ValueError("bound evidence event identity drifted")
    if findings["linkage_group_id"] != base["linkage_group_id"] or reference["linkage_group_id"] != base["linkage_group_id"]:
        raise ValueError("bound evidence linkage group drifted")
    if training_example_ref is None:
        training_example_ref = _ref(
            base,
            kind="training_example",
            schema=PRIVATE_STAGE0_EXAMPLES_SCHEMA_VERSION.replace(
                "private_real_stage0_examples_materialization_v1", "training_example_v1"
            ),
        )
    base_ref = validate_artifact_ref(training_example_ref)
    if base_ref["artifact_kind"] != "training_example":
        raise ValueError("bound evidence training example ref kind drifted")
    if signal_candidate_claim_graph is not None:
        signal_graph = validate_signal_candidate_claim_graph(dict(signal_candidate_claim_graph))
        if signal_graph["linkage_group_id"] != base["linkage_group_id"]:
            raise ValueError("bound evidence patient graph linkage drifted")
    else:
        signal_graph = None
    if knowledge_selection is not None:
        selection = validate_knowledge_selection_receipt(dict(knowledge_selection), trusted_graph=signal_graph)
    else:
        selection = None
    if canonical_report is not None:
        report = validate_canonical_report(
            dict(canonical_report),
            trusted_graph=signal_graph,
            trusted_selection=selection,
        )
    else:
        report = None
    if dual_montage_cache_ref is None and signal_graph is not None:
        graph_refs = signal_graph.get("source_refs", {})
        if isinstance(graph_refs, Mapping):
            dual_montage_cache_ref = graph_refs.get("dual_montage_cache")
    if dual_montage_cache_ref is None:
        raise ValueError("bound evidence requires a dual montage cache reference")
    dual_montage_cache_ref = validate_artifact_ref(dual_montage_cache_ref)
    if dual_montage_cache_ref["artifact_kind"] != "dual_montage_cache_materialization_receipt":
        raise ValueError("bound evidence dual montage cache ref kind drifted")

    # A physician-authored report release is an optional, report-only lane.
    # The release manifest is deliberately referenced rather than copied, and
    # its rows are checked only for the current event's linkage/role here.  The
    # full release validator (including raw-byte/manual-review replay) is run
    # by the materializer before this function is called.
    physician_release_ref = None
    physician_lane_state = "not_released"
    if physician_report_release is not None:
        release = _validate_physician_report_release_metadata(
            physician_report_release,
            linkage_group_id=str(base["linkage_group_id"]),
            evisoz_role=str(base["split_assignment"]["evisoz_role"]),
        )
        physician_rows = _matching_physician_report_rows(
            release,
            linkage_group_id=str(base["linkage_group_id"]),
            evisoz_role=str(base["split_assignment"]["evisoz_role"]),
        )
        if physician_rows:
            physician_release_ref = _ref(
                release,
                kind="private_physician_report_release",
                schema=PHYSICIAN_REPORT_RELEASE_SCHEMA_VERSION,
            )
            purposes = {str(row["purpose"]) for row in physician_rows}
            physician_lane_state = (
                "released_for_qwen_text_training"
                if "qwen_text_training" in purposes
                else "released_for_language_evaluation"
            )
    source_refs: dict[str, object] = {
        "training_example": base_ref,
        "event_findings": _ref(findings, kind="evisoz_event_findings_projection", schema=EVENT_FINDINGS_SCHEMA_VERSION),
        "reference_claim_graph": _ref(reference, kind="evisoz_reference_claim_graph", schema=REFERENCE_GRAPH_SCHEMA_VERSION),
        "field_release": base["artifact_refs"]["field_release"],
        "montage_derivation": base["artifact_refs"]["montage_derivation"],
        "dual_montage_cache": dual_montage_cache_ref,
        "patient_signal_graph": _ref(signal_graph, kind="evisoz_signal_candidate_claim_graph", schema=SIGNAL_GRAPH_SCHEMA_VERSION) if signal_graph is not None else None,
        "knowledge_selection": _ref(selection, kind="evisoz_knowledge_selection_receipt", schema=KNOWLEDGE_SELECTION_SCHEMA_VERSION) if selection is not None else None,
        "canonical_report": _ref(report, kind="evisoz_canonical_report", schema=CANONICAL_REPORT_SCHEMA_VERSION) if report is not None else None,
        "physician_report_release": physician_release_ref,
    }
    body: dict[str, Any] = {
        "schema_version": BOUND_EVIDENCE_SCHEMA_VERSION,
        "bound_example_id": _HASH_PLACEHOLDER,
        "event_id": base["event_id"],
        "linkage_group_id": base["linkage_group_id"],
        "evisoz_role": base["split_assignment"]["evisoz_role"],
        "outer_holdout_fold": base["split_assignment"]["outer_holdout_fold"],
        "status": "stage0_shadow_bound",
        "source_refs": source_refs,
        "lanes": {
            "clinical_labels": {"source": "field_release", "state": "evaluator_only"},
            "direct_measurements": {"source": "event_findings", "state": "not_released"},
            "teacher_candidates": {
                "source": "event_findings" if findings["lanes"]["teacher_candidates"] else "candidate_exposure_ledger",
                "state": "soft_auxiliary_uncalibrated" if findings["lanes"]["teacher_candidates"] else "absent",
            },
            "derived_candidates": {"source": "event_findings", "state": "soft_auxiliary_uncalibrated"},
            "physician_authored_text": {"source": "private_report_release", "state": physician_lane_state},
            "generated_text": {"source": "canonical_report", "state": "shadow_only" if report is not None else "absent"},
        },
        "permissions": deepcopy(_PERMISSIONS),
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["bound_example_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_bound_evidence_example(body)


def _validate_physician_report_release_metadata(
    value: Mapping[str, object],
    *,
    linkage_group_id: str,
    evisoz_role: str,
) -> dict[str, Any]:
    """Validate the non-text portion of a physician release for one event.

    Full release validation needs the candidate bundle and candidate text
    root, so it is performed by ``materialize_bound_evidence_examples``.
    This helper is intentionally conservative for callers that already hold a
    validated release: it checks the immutable identity, permissions, rows and
    event-level split binding without opening report text.
    """

    if type(value) is not dict:
        raise TypeError("physician report release must be an object")
    release = deepcopy(dict(value))
    if release.get("schema_version") != PHYSICIAN_REPORT_RELEASE_SCHEMA_VERSION:
        raise ValueError("physician report release schema drifted")
    if release.get("permissions") != {
        "physician_authored_text_released": True,
        "qwen_text_training_allowed": True,
        "locked_language_evaluation_allowed": True,
        "report_text_can_supervise_localization": False,
        "generated_text_is_not_physician_authored": True,
        "raw_patient_identifiers_stored": False,
    }:
        raise ValueError("physician report release permissions drifted")
    rows = release.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("physician report release rows are empty")
    for row in rows:
        if type(row) is not dict:
            raise ValueError("physician report release row drifted")
        if row.get("linkage_group_id") == linkage_group_id and row.get("evisoz_role") == evisoz_role:
            if row.get("purpose") not in {"qwen_text_training", "language_evaluation"}:
                raise ValueError("physician report release purpose drifted")
    return release


def _matching_physician_report_rows(
    release: Mapping[str, object],
    *,
    linkage_group_id: str,
    evisoz_role: str,
) -> list[dict[str, Any]]:
    rows = release.get("rows")
    if not isinstance(rows, list):
        return []
    matched = [
        deepcopy(dict(row))
        for row in rows
        if isinstance(row, Mapping)
        and row.get("linkage_group_id") == linkage_group_id
        and row.get("evisoz_role") == evisoz_role
    ]
    return sorted(matched, key=lambda row: str(row.get("candidate_id", "")))


def validate_bound_evidence_example(value: object) -> dict[str, Any]:
    required = {
        "schema_version", "bound_example_id", "event_id", "linkage_group_id",
        "evisoz_role", "outer_holdout_fold", "status", "source_refs", "lanes",
        "permissions", "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("bound evidence example fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != BOUND_EVIDENCE_SCHEMA_VERSION or data["status"] != "stage0_shadow_bound":
        raise ValueError("bound evidence example identity drifted")
    if data["evisoz_role"] not in {"development_cv", "locked_test"}:
        raise ValueError("bound evidence split role drifted")
    refs = data["source_refs"]
    if type(refs) is not dict or set(refs) not in ({
        "training_example", "event_findings", "reference_claim_graph", "field_release",
        "montage_derivation", "dual_montage_cache", "patient_signal_graph",
        "knowledge_selection", "canonical_report",
    }, {
        "training_example", "event_findings", "reference_claim_graph", "field_release",
        "montage_derivation", "dual_montage_cache", "patient_signal_graph",
        "knowledge_selection", "canonical_report", "physician_report_release",
    }):
        raise ValueError("bound evidence source refs drifted")
    kinds = {
        "training_example": "training_example",
        "event_findings": "evisoz_event_findings_projection",
        "reference_claim_graph": "evisoz_reference_claim_graph",
        "field_release": "field_release",
        "montage_derivation": "montage_derivation_receipt",
        "dual_montage_cache": "dual_montage_cache_materialization_receipt",
    }
    for key, kind in kinds.items():
        ref = validate_artifact_ref(refs[key])
        if ref["artifact_kind"] != kind:
            raise ValueError(f"bound evidence ref kind drifted: {key}")
    for key, kind in {
        "patient_signal_graph": "evisoz_signal_candidate_claim_graph",
        "knowledge_selection": "evisoz_knowledge_selection_receipt",
        "canonical_report": "evisoz_canonical_report",
    }.items():
        if refs[key] is not None and validate_artifact_ref(refs[key])["artifact_kind"] != kind:
            raise ValueError(f"bound evidence optional ref kind drifted: {key}")
    if "physician_report_release" in refs:
        physician_ref = refs["physician_report_release"]
        physician_lane = data["lanes"]["physician_authored_text"]
        if physician_ref is not None:
            if validate_artifact_ref(physician_ref)["artifact_kind"] != "private_physician_report_release":
                raise ValueError("bound evidence physician report release ref kind drifted")
        elif physician_lane["state"] != "not_released":
            raise ValueError("released physician text requires a release reference")
    lanes = data["lanes"]
    expected_lanes = {
        "clinical_labels", "direct_measurements", "teacher_candidates",
        "derived_candidates", "physician_authored_text", "generated_text",
    }
    if type(lanes) is not dict or set(lanes) != expected_lanes:
        raise ValueError("bound evidence lanes drifted")
    for row in lanes.values():
        if type(row) is not dict or set(row) != {"source", "state"}:
            raise ValueError("bound evidence lane row drifted")
    if lanes["clinical_labels"]["state"] != "evaluator_only":
        raise ValueError("bound evidence label state drifted")
    physician_lane = lanes["physician_authored_text"]
    if physician_lane["source"] != "private_report_release":
        raise ValueError("bound evidence physician text source drifted")
    if physician_lane["state"] not in {
        "not_released",
        "released_for_qwen_text_training",
        "released_for_language_evaluation",
    }:
        raise ValueError("bound evidence physician text state drifted")
    if physician_lane["state"] != "not_released" and (
        "physician_report_release" not in refs or refs["physician_report_release"] is None
    ):
        raise ValueError("released physician text is missing its release reference")
    if "physician_report_release" in refs and physician_lane["state"] == "not_released" and refs["physician_report_release"] is not None:
        raise ValueError("bound evidence unreleased physician text has a release reference")
    teacher_lane = lanes["teacher_candidates"]
    if teacher_lane["state"] not in {"absent", "soft_auxiliary_uncalibrated"}:
        raise ValueError("bound evidence teacher state drifted")
    if teacher_lane["state"] == "absent" and teacher_lane["source"] != "candidate_exposure_ledger":
        raise ValueError("bound evidence absent teacher source drifted")
    if teacher_lane["state"] == "soft_auxiliary_uncalibrated" and teacher_lane["source"] != "event_findings":
        raise ValueError("bound evidence teacher source drifted")
    if data["evisoz_role"] == "locked_test" and teacher_lane["state"] != "absent":
        raise ValueError("bound evidence locked test contains teacher candidates")
    if data["permissions"] != _PERMISSIONS:
        raise ValueError("bound evidence permission policy drifted")
    expected_id = _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["bound_example_id"] != expected_id:
        raise ValueError("bound evidence ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("bound evidence receipt drifted")
    return data


def materialize_bound_evidence_examples(
    *,
    private_examples_root: str | Path,
    findings_claim_report_root: str | Path,
    output: str | Path,
    private_cohort_root: str | Path | None = None,
    split_roster_path: str | Path | None = None,
    physician_report_release_root: str | Path | None = None,
    physician_report_candidate_root: str | Path | None = None,
) -> dict[str, Any]:
    """Join existing Stage-0 artifacts without enabling any loss port."""

    examples_root = Path(private_examples_root).resolve(strict=True)
    findings_root = Path(findings_claim_report_root).resolve(strict=True)
    output_root = Path(output).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    examples_manifest = json.loads((examples_root / "manifest.json").read_text(encoding="utf-8"))
    findings_manifest = json.loads((findings_root / "manifest.json").read_text(encoding="utf-8"))
    if examples_manifest.get("schema_version") != PRIVATE_STAGE0_EXAMPLES_SCHEMA_VERSION:
        raise ValueError("private examples manifest schema drifted")
    if findings_manifest.get("schema_version") != "evisoz_findings_claim_report_materialization_v1":
        raise ValueError("Findings/claim/report manifest schema drifted")

    # The examples manifest intentionally carries references rather than paths.
    # Resolve the immutable cohort and split ledgers explicitly (or from the
    # repository's standard Stage-0 locations) so every envelope is replayed
    # against its original trusted authorities.
    cohort_root = (
        Path(private_cohort_root).resolve(strict=True)
        if private_cohort_root is not None
        else examples_root.parent / "evisoz_stage0_private_real_dual_montage_v1_20260831"
    )
    split_path = (
        Path(split_roster_path).resolve(strict=True)
        if split_roster_path is not None
        else examples_root.parent / "evisoz_stage0_private_split_v1_20260831" / "split_roster.json"
    )
    cohort_manifest_path = cohort_root / "manifest.json"
    if not cohort_manifest_path.is_file() or not split_path.is_file():
        raise FileNotFoundError("trusted Stage-0 cohort or split roster is unavailable")
    cohort_manifest = json.loads(cohort_manifest_path.read_text(encoding="utf-8"))
    split_roster = json.loads(split_path.read_text(encoding="utf-8"))
    cohort_rows = {str(row["event_id"]): row for row in cohort_manifest.get("events", [])}
    patient_keys = {
        str(row["patient_id"])
        for key in ("events", "preexcluded_events", "runtime_excluded_events")
        for row in cohort_manifest.get(key, [])
        if row.get("patient_id") is not None
    }
    trusted_groups = {
        group["linkage_group_id"]: group
        for group in (build_private_patient_linkage_group(key) for key in sorted(patient_keys))
    }
    split_roster = validate_split_roster(split_roster, trusted_linkage_groups=trusted_groups)
    physician_release: dict[str, Any] | None = None
    if physician_report_release_root is not None:
        release_root = Path(physician_report_release_root).resolve(strict=True)
        if release_root.is_symlink() or not release_root.is_dir():
            raise ValueError("physician report release root must be a regular directory")
        release_path = release_root / "release.json"
        if release_path.is_symlink() or not release_path.is_file():
            raise ValueError("physician report release manifest is missing")
        candidate_root = (
            Path(physician_report_candidate_root).resolve(strict=True)
            if physician_report_candidate_root is not None
            else examples_root.parent / "evisoz_stage0_private_report_deid_candidates_v1_20260831"
        )
        candidate_manifest_path = candidate_root / "manifest.json"
        if candidate_manifest_path.is_symlink() or not candidate_manifest_path.is_file():
            raise ValueError("physician report candidate manifest is missing")
        from src.evisoz.data.private_physician_report_release import (
            validate_private_physician_report_release,
        )
        physician_release = validate_private_physician_report_release(
            json.loads(release_path.read_text(encoding="utf-8")),
            candidate_bundle=json.loads(candidate_manifest_path.read_text(encoding="utf-8")),
            candidate_output_root=candidate_root,
        )
    event_rows = {str(row["event_id"]): row for row in findings_manifest["event_rows"]}
    patient_rows = {str(row["linkage_group_id"]): row for row in findings_manifest["patient_rows"]}
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    bound_rows: list[dict[str, Any]] = []
    expected_files = {"manifest.json"}
    try:
        for base_row in sorted(examples_manifest["events"], key=lambda row: str(row["event_id"])):
            event_id = str(base_row["event_id"])
            if event_id not in event_rows:
                raise ValueError(f"bound evidence missing Findings event: {event_id}")
            event_dir = examples_root / "events" / event_id
            base = json.loads((event_dir / "training_example.json").read_text(encoding="utf-8"))
            field_release = json.loads((event_dir / "field_release.json").read_text(encoding="utf-8"))
            cohort_row = cohort_rows.get(event_id)
            if cohort_row is None:
                raise ValueError(f"bound evidence missing cohort event: {event_id}")
            cache_root = cohort_root / str(cohort_row["relative_cache_path"])
            identity = validate_event_identity(json.loads((cache_root / "sidecars" / "event_identity.json").read_text(encoding="utf-8")))
            montage = validate_montage_derivation_receipt(
                json.loads((cache_root / "sidecars" / "montage_receipt.json").read_text(encoding="utf-8")),
                trusted_event_identity=identity,
            )
            trusted_values = {
                str(row["value_ref"]["artifact_id"]): row["value_payload"]
                for row in field_release.get("fields", [])
                if row.get("value_ref") is not None
            }
            field_release = validate_field_release(
                field_release,
                trusted_event_identity=identity,
                trusted_values_by_artifact_id=trusted_values,
            )
            training_context = {
                "split_roster": split_roster,
                "trusted_linkage_groups": trusted_groups,
                "event_identity": identity,
                "montage_receipt": montage,
                "field_release": field_release,
            }
            finding_row = event_rows[event_id]
            findings = json.loads(_safe_json_path(findings_root, finding_row["relative_findings_path"]).read_text(encoding="utf-8"))
            reference = json.loads(_safe_json_path(findings_root, finding_row["relative_reference_claim_graph_path"]).read_text(encoding="utf-8"))
            patient = patient_rows.get(str(base_row["linkage_group_id"]))
            signal_graph = selection = report = None
            if patient is not None:
                patient_dir = findings_root / "patients" / str(base_row["linkage_group_id"])
                signal_graph = json.loads(_safe_json_path(findings_root, patient["relative_signal_candidate_claim_graph_path"]).read_text(encoding="utf-8"))
                selection = json.loads(_safe_json_path(findings_root, patient["relative_knowledge_selection_path"]).read_text(encoding="utf-8"))
                report = json.loads(_safe_json_path(findings_root, patient["relative_canonical_report_path"]).read_text(encoding="utf-8"))
            bound = build_bound_evidence_example(
                training_example=base,
                event_findings=findings,
                reference_claim_graph=reference,
                signal_candidate_claim_graph=signal_graph,
                knowledge_selection=selection,
                canonical_report=report,
                training_example_ref=base_row["training_example_ref"],
                training_validation_context=training_context,
                dual_montage_cache_ref=findings.get("source_refs", {}).get("dual_montage_cache"),
                physician_report_release=physician_release,
            )
            relative = f"events/{event_id}/bound_evidence.json"
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=False)
            path.write_text(json.dumps(bound, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            expected_files.add(relative)
            bound_rows.append({
                "event_id": event_id,
                "linkage_group_id": bound["linkage_group_id"],
                "evisoz_role": bound["evisoz_role"],
                "outer_holdout_fold": bound["outer_holdout_fold"],
                "bound_evidence_ref": _ref(bound, kind="evisoz_bound_evidence_example", schema=BOUND_EVIDENCE_SCHEMA_VERSION),
                "relative_path": relative,
            })
        bound_rows.sort(key=lambda row: str(row["event_id"]))
        manifest: dict[str, Any] = {
            "schema_version": BOUND_MATERIALIZATION_SCHEMA_VERSION,
            "materialization_id": _HASH_PLACEHOLDER,
            "status": "stage0_shadow_bound_examples_materialized",
            "source_refs": {
                "private_examples_manifest": build_json_artifact_ref(examples_manifest, artifact_kind="private_real_examples_manifest", payload_schema_version=PRIVATE_STAGE0_EXAMPLES_SCHEMA_VERSION),
                "findings_claim_report_manifest": build_json_artifact_ref(findings_manifest, artifact_kind="findings_claim_report_materialization", payload_schema_version="evisoz_findings_claim_report_materialization_v1"),
                "physician_report_release": (
                    _ref(
                        physician_release,
                        kind="private_physician_report_release",
                        schema=PHYSICIAN_REPORT_RELEASE_SCHEMA_VERSION,
                    )
                    if physician_release is not None
                    else None
                ),
            },
            "rows": bound_rows,
            "counts": {
                "event_count": len(bound_rows),
                "development_event_count": sum(row["evisoz_role"] == "development_cv" for row in bound_rows),
                "locked_test_event_count": sum(row["evisoz_role"] == "locked_test" for row in bound_rows),
                "training_authorized_event_count": 0,
                "physician_report_text_released_count": sum(
                    json.loads(
                        (staging / f"events/{row['event_id']}/bound_evidence.json").read_text(encoding="utf-8")
                    )["lanes"]["physician_authored_text"]["state"] != "not_released"
                    for row in bound_rows
                ),
            },
            "permissions": deepcopy(_PERMISSIONS),
            "receipt_sha256": _HASH_PLACEHOLDER,
        }
        manifest["materialization_id"] = _MATERIALIZATION_ID_PREFIX + canonical_json_sha256(_id_source(manifest, "materialization_id"))[:24]
        manifest["receipt_sha256"] = canonical_json_sha256(_hash_source(manifest))
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        actual_files = {
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        if actual_files != expected_files:
            raise ValueError("bound evidence materialization file inventory drifted")
        # Validate the complete staged tree before publication so a failed
        # semantic check can never leave an invalid destination behind.
        validate_bound_evidence_materialization(manifest, output_root=staging)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output_root)
        return validate_bound_evidence_materialization(manifest, output_root=output_root)
    except Exception:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_bound_evidence_materialization(value: object, *, output_root: str | Path | None = None) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version", "materialization_id", "status", "source_refs", "rows", "counts", "permissions", "receipt_sha256",
    }:
        raise ValueError("bound evidence materialization fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != BOUND_MATERIALIZATION_SCHEMA_VERSION or data["status"] != "stage0_shadow_bound_examples_materialized":
        raise ValueError("bound evidence materialization identity drifted")
    for key, ref in data["source_refs"].items():
        if key in {
            "patient_signal_graph",
            "knowledge_selection",
            "canonical_report",
            "physician_report_release",
        } and ref is None:
            continue
        validate_artifact_ref(ref)
    root: Path | None = None
    expected_files: set[str] | None = None
    if output_root is not None:
        root = Path(output_root).resolve(strict=True)
        manifest_path = root / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("bound evidence materialization manifest file missing")
        expected_files = {"manifest.json"}
    rows = data["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("bound evidence materialization rows are empty")
    seen: set[str] = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"event_id", "linkage_group_id", "evisoz_role", "outer_holdout_fold", "bound_evidence_ref", "relative_path"}:
            raise ValueError("bound evidence materialization row fields drifted")
        if row["event_id"] in seen:
            raise ValueError("bound evidence materialization events duplicated")
        seen.add(row["event_id"])
        ref = validate_artifact_ref(row["bound_evidence_ref"])
        if ref["artifact_kind"] != "evisoz_bound_evidence_example":
            raise ValueError("bound evidence materialization row ref kind drifted")
        if root is not None:
            path = _safe_json_path(root, row["relative_path"])
            expected_files.add(PurePosixPath(row["relative_path"]).as_posix())
            payload = json.loads(path.read_text(encoding="utf-8"))
            bound = validate_bound_evidence_example(payload)
            verify_artifact_content(row["bound_evidence_ref"], bound)
            if bound["event_id"] != row["event_id"] or bound["linkage_group_id"] != row["linkage_group_id"]:
                raise ValueError("bound evidence materialization row identity drifted")
    if root is not None and expected_files is not None:
        actual_files = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        if actual_files != expected_files:
            raise ValueError("bound evidence materialization file inventory drifted")
    counts = data["counts"]
    expected = {
        "event_count": len(rows),
        "development_event_count": sum(row["evisoz_role"] == "development_cv" for row in rows),
        "locked_test_event_count": sum(row["evisoz_role"] == "locked_test" for row in rows),
        "training_authorized_event_count": 0,
    }
    if type(counts) is not dict or set(counts) != {
        "event_count", "development_event_count", "locked_test_event_count",
        "training_authorized_event_count", "physician_report_text_released_count",
    }:
        raise ValueError("bound evidence materialization count fields drifted")
    released_count = counts["physician_report_text_released_count"]
    if type(released_count) is not int or released_count < 0 or released_count > len(rows):
        raise ValueError("bound evidence physician report release count drifted")
    if root is not None:
        observed_released_count = 0
        for row in rows:
            payload = json.loads(_safe_json_path(root, row["relative_path"]).read_text(encoding="utf-8"))
            if payload["lanes"]["physician_authored_text"]["state"] != "not_released":
                observed_released_count += 1
        if released_count != observed_released_count:
            raise ValueError("bound evidence physician report release count does not match rows")
    if data["permissions"] != _PERMISSIONS:
        raise ValueError("bound evidence materialization permissions drifted")
    if counts["event_count"] != expected["event_count"] or counts["development_event_count"] != expected["development_event_count"] or counts["locked_test_event_count"] != expected["locked_test_event_count"] or counts["training_authorized_event_count"] != expected["training_authorized_event_count"]:
        raise ValueError("bound evidence materialization counts/permissions drifted")
    expected_id = _MATERIALIZATION_ID_PREFIX + canonical_json_sha256(_id_source(data, "materialization_id"))[:24]
    if data["materialization_id"] != expected_id or data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("bound evidence materialization receipt drifted")
    return data


__all__ = [
    "BOUND_EVIDENCE_SCHEMA_VERSION",
    "BOUND_MATERIALIZATION_SCHEMA_VERSION",
    "build_bound_evidence_example",
    "materialize_bound_evidence_examples",
    "validate_bound_evidence_example",
    "validate_bound_evidence_materialization",
]
