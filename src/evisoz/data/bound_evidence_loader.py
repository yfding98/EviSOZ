"""Fail-closed replay loader for real EviSOZ bound evidence.

The loader is deliberately a read-only bridge from the Stage-0 bound
materialization to model consumers.  It replays every content-addressed
reference against the original private example, Findings/claim/report and
dual-montage roots.  It never opens physician-authored DOCX files and never
returns a train-authorized view while Stage-0 is closed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

from src.evisoz.data.artifact_ref import (
    build_json_artifact_ref,
    canonical_json_sha256,
    validate_artifact_ref,
    verify_artifact_content,
)
from src.evisoz.data.event_identity import validate_event_identity
from src.evisoz.data.private_stage0_split import build_private_patient_linkage_group
from src.evisoz.data.split_ledger import validate_split_roster
from src.evisoz.data.stage0_dual_montage_cache import (
    OpenedStage0DualMontageCache,
    open_stage0_dual_montage_cache_from_disk,
)
from src.evisoz.data.tcp22_views import validate_montage_derivation_receipt
from src.evisoz.data.dataset_policy import validate_field_release
from src.evisoz.forge.evidence_binding import (
    BOUND_EVIDENCE_SCHEMA_VERSION,
    BOUND_MATERIALIZATION_SCHEMA_VERSION,
    validate_bound_evidence_example,
    validate_bound_evidence_materialization,
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
from src.evisoz.forge.private_stage0_examples import PRIVATE_STAGE0_EXAMPLES_SCHEMA_VERSION
from src.evisoz.forge.training_example import (
    TRAINING_EXAMPLE_SCHEMA_VERSION,
    validate_training_example,
)


BOUND_LOADER_RECEIPT_SCHEMA_VERSION = "evisoz_bound_evidence_loader_receipt_v1"
_HASH_PLACEHOLDER = "0" * 64
_ID_PREFIX = "EVISOZ-BOUND-LOADER-"


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    body = _hash_source(value)
    body["loader_id"] = "CONTENT-ADDRESS-PENDING"
    return body


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON artifact must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _plain(value: object) -> object:
    """Convert validated frozen mappings/tuples back to strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _safe_json_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise TypeError("manifest relative path must be a string")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or not parsed.parts or any(
        part in {"", ".", ".."} for part in parsed.parts
    ):
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


def _safe_directory(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise TypeError("manifest relative directory must be a string")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or not parsed.parts or any(
        part in {"", ".", ".."} for part in parsed.parts
    ):
        raise ValueError("manifest relative directory is unsafe")
    candidate = root.joinpath(*parsed.parts)
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(root.resolve(strict=True))
    if candidate.is_symlink() or not resolved.is_dir():
        raise ValueError("manifest directory must be a regular directory")
    return resolved


@dataclass(frozen=True)
class BoundEvidenceRecord:
    """One fully replayed event and its lazily check-out-able cache."""

    bound_evidence: Mapping[str, Any]
    training_example: Mapping[str, Any]
    field_release: Mapping[str, Any]
    event_findings: Mapping[str, Any]
    reference_claim_graph: Mapping[str, Any]
    signal_candidate_claim_graph: Mapping[str, Any] | None
    knowledge_selection: Mapping[str, Any] | None
    canonical_report: Mapping[str, Any] | None
    event_identity: Mapping[str, Any]
    montage_receipt: Mapping[str, Any]
    cache: OpenedStage0DualMontageCache

    @property
    def event_id(self) -> str:
        return str(self.bound_evidence["event_id"])

    @property
    def linkage_group_id(self) -> str:
        return str(self.bound_evidence["linkage_group_id"])

    @property
    def evisoz_role(self) -> str:
        return str(self.bound_evidence["evisoz_role"])

    def checkout_inputs(self) -> dict[str, Any]:
        """Return fresh tensors plus masks; every checkout is clone-isolated."""

        v29 = self.cache.checkout_v29_reference()
        tcp_context = self.cache.checkout_tcp22_context()
        tcp_onset = self.cache.checkout_tcp22_onset()
        views = self.montage_receipt.get("views", {})
        if not isinstance(views, Mapping):
            raise ValueError("montage receipt views are missing")
        standard_mask = views.get("v29_reference", {}).get("unit_observed_mask")
        edge_mask = views.get("tcp22_context", {}).get("unit_observed_mask")
        if not isinstance(standard_mask, list) or not isinstance(edge_mask, list):
            raise ValueError("montage view masks are missing")
        return {
            "v29_reference": v29,
            "tcp22_context": tcp_context,
            "tcp22_onset": tcp_onset,
            "standard19_observed_mask": tuple(bool(item) for item in standard_mask),
            "tcp22_observed_mask": tuple(bool(item) for item in edge_mask),
            "event_identity": deepcopy(dict(self.event_identity)),
            "montage_receipt": deepcopy(dict(self.montage_receipt)),
        }


def _trusted_groups(cohort_manifest: Mapping[str, object]) -> dict[str, dict[str, Any]]:
    patient_keys = {
        str(row["patient_id"])
        for key in ("events", "preexcluded_events", "runtime_excluded_events")
        for row in cohort_manifest.get(key, [])
        if isinstance(row, Mapping) and row.get("patient_id") is not None
    }
    if not patient_keys:
        raise ValueError("trusted cohort has no patient keys")
    groups = [build_private_patient_linkage_group(key) for key in sorted(patient_keys)]
    return {str(group["linkage_group_id"]): group for group in groups}


def _verify_ref(payload: Mapping[str, object], ref: Mapping[str, object], *, kind: str, schema: str) -> None:
    expected = build_json_artifact_ref(payload, artifact_kind=kind, payload_schema_version=schema)
    if expected != validate_artifact_ref(ref):
        raise ValueError(f"bound evidence source reference drifted: {kind}")


def load_bound_evidence_record(
    *,
    bound_evidence_root: str | Path,
    private_examples_root: str | Path,
    findings_claim_report_root: str | Path,
    private_cohort_root: str | Path,
    split_roster_path: str | Path,
    event_id: str,
) -> BoundEvidenceRecord:
    """Replay one event against all trusted Stage-0 authorities."""

    bound_root = Path(bound_evidence_root).resolve(strict=True)
    examples_root = Path(private_examples_root).resolve(strict=True)
    findings_root = Path(findings_claim_report_root).resolve(strict=True)
    cohort_root = Path(private_cohort_root).resolve(strict=True)
    split_path = Path(split_roster_path).resolve(strict=True)
    manifest = validate_bound_evidence_materialization(
        _read_json(bound_root / "manifest.json"), output_root=bound_root
    )
    row = next((item for item in manifest["rows"] if str(item["event_id"]) == event_id), None)
    if row is None:
        raise KeyError(f"bound evidence event is not present: {event_id}")
    bound = validate_bound_evidence_example(
        _read_json(_safe_json_path(bound_root, row["relative_path"]))
    )
    if bound["event_id"] != event_id:
        raise ValueError("bound evidence event ID drifted")

    examples_manifest = _read_json(examples_root / "manifest.json")
    findings_manifest = _read_json(findings_root / "manifest.json")
    cohort_manifest = _read_json(cohort_root / "manifest.json")
    split = _read_json(split_path)
    groups = _trusted_groups(cohort_manifest)
    split = validate_split_roster(split, trusted_linkage_groups=groups)
    example_row = next(
        (item for item in examples_manifest["events"] if str(item["event_id"]) == event_id),
        None,
    )
    finding_row = next(
        (item for item in findings_manifest["event_rows"] if str(item["event_id"]) == event_id),
        None,
    )
    if example_row is None or finding_row is None:
        raise ValueError("bound evidence source event roster is incomplete")
    example_dir = examples_root / "events" / event_id
    training_example = _read_json(example_dir / "training_example.json")
    field_release = _read_json(example_dir / "field_release.json")
    cohort_row = next(
        (item for item in cohort_manifest["events"] if str(item["event_id"]) == event_id),
        None,
    )
    if cohort_row is None:
        raise ValueError("bound evidence cohort event is missing")
    cache_root = _safe_directory(cohort_root, cohort_row["relative_cache_path"])
    event_identity = validate_event_identity(
        _read_json(cache_root / "sidecars" / "event_identity.json")
    )
    montage = validate_montage_derivation_receipt(
        _read_json(cache_root / "sidecars" / "montage_receipt.json"),
        trusted_event_identity=event_identity,
    )
    trusted_values = {
        str(item["value_ref"]["artifact_id"]): item["value_payload"]
        for item in field_release.get("fields", [])
        if item.get("value_ref") is not None
    }
    field_release = validate_field_release(
        field_release,
        trusted_event_identity=event_identity,
        trusted_values_by_artifact_id=trusted_values,
    )
    training_example = validate_training_example(
        training_example,
        split_roster=split,
        trusted_linkage_groups=groups,
        event_identity=event_identity,
        montage_receipt=montage,
        field_release=field_release,
    )
    _verify_ref(training_example, bound["source_refs"]["training_example"], kind="training_example", schema=TRAINING_EXAMPLE_SCHEMA_VERSION)
    _verify_ref(field_release, bound["source_refs"]["field_release"], kind="field_release", schema="evisoz_field_release_v1")
    # The bound reference is for the complete montage derivation receipt,
    # not for the event-identity sidecar.  Keeping these authorities separate
    # prevents a valid event identity from masking a montage/clock/mask drift.
    _verify_ref(montage, bound["source_refs"]["montage_derivation"], kind="montage_derivation_receipt", schema="evisoz_montage_derivation_receipt_v1")

    findings = validate_event_findings_projection(
        _read_json(_safe_json_path(findings_root, finding_row["relative_findings_path"]))
    )
    reference = validate_reference_claim_graph(
        _read_json(_safe_json_path(findings_root, finding_row["relative_reference_claim_graph_path"]))
    )
    _verify_ref(findings, bound["source_refs"]["event_findings"], kind="evisoz_event_findings_projection", schema=EVENT_FINDINGS_SCHEMA_VERSION)
    _verify_ref(reference, bound["source_refs"]["reference_claim_graph"], kind="evisoz_reference_claim_graph", schema=REFERENCE_GRAPH_SCHEMA_VERSION)
    signal_graph = selection = report = None
    patient_row = next(
        (item for item in findings_manifest["patient_rows"] if str(item["linkage_group_id"]) == bound["linkage_group_id"]),
        None,
    )
    if patient_row is not None:
        signal_graph = validate_signal_candidate_claim_graph(
            _read_json(_safe_json_path(findings_root, patient_row["relative_signal_candidate_claim_graph_path"]))
        )
        selection = validate_knowledge_selection_receipt(
            _read_json(_safe_json_path(findings_root, patient_row["relative_knowledge_selection_path"])),
            trusted_graph=signal_graph,
        )
        report = validate_canonical_report(
            _read_json(_safe_json_path(findings_root, patient_row["relative_canonical_report_path"])),
            trusted_graph=signal_graph,
            trusted_selection=selection,
        )
        _verify_ref(signal_graph, bound["source_refs"]["patient_signal_graph"], kind="evisoz_signal_candidate_claim_graph", schema=SIGNAL_GRAPH_SCHEMA_VERSION)
        _verify_ref(selection, bound["source_refs"]["knowledge_selection"], kind="evisoz_knowledge_selection_receipt", schema=KNOWLEDGE_SELECTION_SCHEMA_VERSION)
        _verify_ref(report, bound["source_refs"]["canonical_report"], kind="evisoz_canonical_report", schema=CANONICAL_REPORT_SCHEMA_VERSION)
    elif any(bound["source_refs"][key] is not None for key in ("patient_signal_graph", "knowledge_selection", "canonical_report")):
        raise ValueError("bound evidence patient-level optional refs have no patient row")

    cache = open_stage0_dual_montage_cache_from_disk(cache_root)
    cache_ref = build_json_artifact_ref(
        _plain(cache.materialization_receipt),
        artifact_kind="dual_montage_cache_materialization_receipt",
        payload_schema_version="evisoz_dual_montage_cache_materialization_receipt_v1",
    )
    if cache_ref != validate_artifact_ref(bound["source_refs"]["dual_montage_cache"]):
        raise ValueError("bound evidence dual montage cache reference drifted")
    return BoundEvidenceRecord(
        bound_evidence=bound,
        training_example=training_example,
        field_release=field_release,
        event_findings=findings,
        reference_claim_graph=reference,
        signal_candidate_claim_graph=signal_graph,
        knowledge_selection=selection,
        canonical_report=report,
        event_identity=event_identity,
        montage_receipt=montage,
        cache=cache,
    )


def iter_bound_evidence_records(
    *,
    bound_evidence_root: str | Path,
    private_examples_root: str | Path,
    findings_claim_report_root: str | Path,
    private_cohort_root: str | Path,
    split_roster_path: str | Path,
    evisoz_role: str | None = None,
    limit: int | None = None,
) -> Iterator[BoundEvidenceRecord]:
    """Yield validated records in manifest order, optionally role-filtered."""

    bound_root = Path(bound_evidence_root).resolve(strict=True)
    manifest = validate_bound_evidence_materialization(
        _read_json(bound_root / "manifest.json"), output_root=bound_root
    )
    if evisoz_role is not None and evisoz_role not in {"development_cv", "locked_test"}:
        raise ValueError("unsupported EviSOZ role filter")
    rows = [
        row for row in manifest["rows"]
        if evisoz_role is None or row["evisoz_role"] == evisoz_role
    ]
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        rows = rows[:limit]
    for row in rows:
        yield load_bound_evidence_record(
            bound_evidence_root=bound_root,
            private_examples_root=private_examples_root,
            findings_claim_report_root=findings_claim_report_root,
            private_cohort_root=private_cohort_root,
            split_roster_path=split_roster_path,
            event_id=str(row["event_id"]),
        )


def build_bound_evidence_loader_receipt(
    *,
    bound_evidence_root: str | Path,
    private_examples_root: str | Path,
    findings_claim_report_root: str | Path,
    private_cohort_root: str | Path,
    split_roster_path: str | Path,
    evisoz_role: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Replay records and emit a non-authorizing loader receipt."""

    records = list(
        iter_bound_evidence_records(
            bound_evidence_root=bound_evidence_root,
            private_examples_root=private_examples_root,
            findings_claim_report_root=findings_claim_report_root,
            private_cohort_root=private_cohort_root,
            split_roster_path=split_roster_path,
            evisoz_role=evisoz_role,
            limit=limit,
        )
    )
    if not records:
        raise ValueError("bound evidence loader selected no records")
    bound_manifest = _read_json(Path(bound_evidence_root).resolve(strict=True) / "manifest.json")
    body: dict[str, Any] = {
        "schema_version": BOUND_LOADER_RECEIPT_SCHEMA_VERSION,
        "loader_id": _HASH_PLACEHOLDER,
        "status": "real_bound_evidence_replay_only",
        "source_refs": {
            "bound_evidence_manifest": build_json_artifact_ref(bound_manifest, artifact_kind="bound_evidence_materialization", payload_schema_version=BOUND_MATERIALIZATION_SCHEMA_VERSION),
        },
        "selection": {
            "evisoz_role": evisoz_role,
            "limit": limit,
            "event_ids": [record.event_id for record in records],
            "linkage_group_ids": sorted({record.linkage_group_id for record in records}),
        },
        "counts": {
            "event_count": len(records),
            "patient_count": len({record.linkage_group_id for record in records}),
        },
        "runtime_policy": {
            "physician_report_text_opened": False,
            "canonical_shadow_report_opened": True,
            "teacher_runtime_opened": False,
            "training_allowed": False,
            "prompt_or_rag_allowed": False,
            "node_localization_supervision_allowed": False,
        },
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["loader_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_bound_evidence_loader_receipt(body)


def validate_bound_evidence_loader_receipt(value: object) -> dict[str, Any]:
    required = {"schema_version", "loader_id", "status", "source_refs", "selection", "counts", "runtime_policy", "receipt_sha256"}
    if type(value) is not dict or set(value) != required:
        raise ValueError("bound evidence loader receipt fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != BOUND_LOADER_RECEIPT_SCHEMA_VERSION or data["status"] != "real_bound_evidence_replay_only":
        raise ValueError("bound evidence loader receipt identity drifted")
    if not isinstance(data["selection"].get("event_ids"), list) or not data["selection"]["event_ids"]:
        raise ValueError("bound evidence loader receipt selection is empty")
    if data["counts"] != {
        "event_count": len(data["selection"]["event_ids"]),
        "patient_count": len(set(data["selection"]["linkage_group_ids"])),
    }:
        raise ValueError("bound evidence loader receipt counts drifted")
    if data["runtime_policy"] != {
        "physician_report_text_opened": False,
        "canonical_shadow_report_opened": True,
        "teacher_runtime_opened": False,
        "training_allowed": False,
        "prompt_or_rag_allowed": False,
        "node_localization_supervision_allowed": False,
    }:
        raise ValueError("bound evidence loader runtime policy drifted")
    for ref in data["source_refs"].values():
        validate_artifact_ref(ref)
    expected_id = _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["loader_id"] != expected_id or data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("bound evidence loader receipt hash drifted")
    return data


__all__ = [
    "BOUND_LOADER_RECEIPT_SCHEMA_VERSION",
    "BoundEvidenceRecord",
    "build_bound_evidence_loader_receipt",
    "iter_bound_evidence_records",
    "load_bound_evidence_record",
    "validate_bound_evidence_loader_receipt",
]
