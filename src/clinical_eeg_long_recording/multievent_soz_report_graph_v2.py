"""Lossless public/synthetic shadow report graph for event Findings v3.

The legacy ``clinical_eeg_multievent_soz_report_v1`` graph and its renderer
remain frozen.  They cannot preserve all four Finding states, raw-sample
dependencies, the five evidence-permission roles, or the explicit v3 signal
differential and event outcome.  This module therefore materializes a new
shadow-only graph whose first invariant is exact source replay.

Each embedded ``event_eeg_findings_v3`` payload is untrusted input: the
validator always re-runs the v3 validator, recomputes every materialized
projection and content seal, and optionally requires byte-equivalent trusted
source graphs supplied by the host.  Source objects are retained verbatim;
normalized nodes and edges carry JSON pointers plus object hashes.

The graph does not call an LLM and does not render a clinical report.  Qwen is
restricted to a future closed-claim lexicalizer with deterministic fallback.
Only ``public`` and ``synthetic`` route scopes are accepted.  Private EEG,
annotations, spreadsheets, physician labels, clinical text and production
Qwen routes are deliberately outside this module.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from .event_findings_v3_validation import validate_event_eeg_findings_v3_payload


MULTIEVENT_SOZ_REPORT_GRAPH_V2_SCHEMA_VERSION = (
    "clinical_eeg_multievent_soz_report_graph_v2"
)
MULTIEVENT_SOZ_REPORT_GRAPH_V2_ROUTE_ID = (
    "public_synthetic_lossless_report_graph_v2_shadow"
)
MULTIEVENT_SOZ_REPORT_GRAPH_V2_PRIVATE_ROUTE_CONNECTED = False
MULTIEVENT_SOZ_REPORT_GRAPH_V2_QWEN_PRODUCTION_ROUTE_CONNECTED = False

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = (
    _ROOT
    / "schemas"
    / "clinical_eeg_multievent_soz_report_graph_v2.schema.json"
)
_V3_SCHEMA_PATH = (
    _ROOT / "schemas" / "clinical_eeg_event_findings_v3.schema.json"
)
_V2_SCHEMA_PATH = (
    _ROOT / "schemas" / "clinical_eeg_event_findings_v2.schema.json"
)

_ROUTE_SCOPES = {"public", "synthetic"}
_ROLE_ORDER = (
    "ictal_pattern_qualification",
    "onset_time_support",
    "onset_topography_support",
    "course_or_spread_support",
    "counterevidence",
)
_QUALIFIED_EVENT_STATUSES = {
    "qualified_electrographic_event",
    "qualified_electrographic_seizure",
}
_SPATIAL_AXES = {"lead", "electrode", "region", "laterality"}
_CLAIM_BOUNDARY = (
    "research_scalp_visible_onset_topology_not_cortical_soz_ez_or_surgical_target"
)
_INFERENCE_EXCLUSIONS: Mapping[str, bool] = {
    "edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_used": False,
    "ecg_emg_eog_used": False,
    "sleep_staging_used": False,
    "provocation_used": False,
}
_LEFT_ELECTRODES = {
    "FP1", "F3", "F7", "C3", "T3", "T5", "T7", "P3", "P7", "O1",
    "A1", "M1",
}
_RIGHT_ELECTRODES = {
    "FP2", "F4", "F8", "C4", "T4", "T6", "T8", "P4", "P8", "O2",
    "A2", "M2",
}
_MIDLINE_ELECTRODES = {"FZ", "CZ", "PZ", "OZ"}
_ELECTRODE_TO_REGION = {
    "FP1": "frontal",
    "FP2": "frontal",
    "F3": "frontal",
    "F4": "frontal",
    "F7": "frontal",
    "F8": "frontal",
    "FZ": "frontal",
    "C3": "central",
    "C4": "central",
    "CZ": "central",
    "T3": "temporal",
    "T4": "temporal",
    "T5": "temporal",
    "T6": "temporal",
    "T7": "temporal",
    "T8": "temporal",
    "P7": "temporal",
    "P8": "temporal",
    "P3": "parietal",
    "P4": "parietal",
    "PZ": "parietal",
    "O1": "occipital",
    "O2": "occipital",
    "OZ": "occipital",
}


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    v2_schema = json.loads(_V2_SCHEMA_PATH.read_text(encoding="utf-8"))
    v3_schema = json.loads(_V3_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    registry = Registry().with_resources(
        [
            (str(v2_schema["$id"]), Resource.from_contents(v2_schema)),
            (str(v3_schema["$id"]), Resource.from_contents(v3_schema)),
        ]
    )
    return Draft202012Validator(schema, registry=registry)


def _schema_path(error: object) -> str:
    parts = [str(value) for value in getattr(error, "absolute_path", ())]
    return ".".join(parts) if parts else "$"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _bounded_id(prefix: str, value: object) -> str:
    return f"{prefix}-{_sha256(value)[:24]}"


def _seal(value: dict[str, Any], field: str, domain: str) -> None:
    value[field] = "0" * 64
    value[field] = _sha256({"binding_domain": domain, "value": value})


def _reject_nonfinite(value: object, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def _source_ref(
    *, object_kind: str, object_id: str, json_pointer: str, value: object
) -> dict[str, str]:
    return {
        "object_kind": object_kind,
        "object_id": object_id,
        "json_pointer": json_pointer,
        "object_sha256": _sha256(value),
    }


def _validate_record_context(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("record_context must be an object")
    expected = {
        "record_id",
        "canonical_signal_sha256",
        "recording_duration_seconds",
        "source_inference_exclusions",
    }
    if set(value) != expected:
        raise ValueError("record_context keys are not closed")
    record_id = value["record_id"]
    signal_sha256 = value["canonical_signal_sha256"]
    duration = value["recording_duration_seconds"]
    if not isinstance(record_id, str) or not record_id:
        raise TypeError("record_context.record_id must be non-empty")
    if (
        not isinstance(signal_sha256, str)
        or len(signal_sha256) != 64
        or any(char not in "0123456789abcdef" for char in signal_sha256)
    ):
        raise ValueError("record_context canonical signal hash is invalid")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise TypeError("recording duration must be numeric")
    duration_float = float(duration)
    if not math.isfinite(duration_float) or duration_float < 0.0:
        raise ValueError("recording duration must be finite and non-negative")
    exclusions = _require_explicit_eeg_only_exclusions(
        value["source_inference_exclusions"],
        context="record_context.source_inference_exclusions",
    )
    return {
        "record_id": record_id,
        "canonical_signal_sha256": signal_sha256,
        "recording_duration_seconds": duration_float,
        "source_inference_exclusions": exclusions,
    }


def _require_explicit_eeg_only_exclusions(
    value: object, *, context: str
) -> dict[str, bool]:
    """Require affirmative source receipts; ``unknown`` is not ``False``."""

    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    expected = set(_INFERENCE_EXCLUSIONS)
    if set(value) != expected:
        raise ValueError(f"{context} keys are not closed")
    unresolved = sorted(key for key, item in value.items() if item is not False)
    if unresolved:
        raise ValueError(
            "EEG-only report graph requires every source inference exclusion "
            f"to be explicitly false; unresolved fields at {context}: {unresolved}"
        )
    return {key: False for key in _INFERENCE_EXCLUSIONS}


def _validation_kwargs(
    *,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ),
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, object]:
    return {
        "trusted_producer_receipts": trusted_producer_receipts,
        "trusted_calibration_receipts": trusted_calibration_receipts,
        "trusted_capability_qualification_receipts": (
            trusted_capability_qualification_receipts
        ),
        "trusted_sensitivity_receipts": trusted_sensitivity_receipts,
        "trusted_term_decision_receipts": trusted_term_decision_receipts,
        "trusted_registry_bindings": trusted_registry_bindings,
    }


def _dependency_index(
    source: Mapping[str, Any], source_index: int
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    objects: dict[str, dict[str, Any]] = {}
    refs: dict[str, dict[str, str]] = {}

    def add(dependency: object, pointer: str) -> None:
        if dependency is None:
            return
        if not isinstance(dependency, Mapping):
            raise TypeError("raw_sample_dependency must be an object or null")
        item = deepcopy(dict(dependency))
        dependency_id = str(item["dependency_id"])
        previous = objects.get(dependency_id)
        if previous is not None and previous != item:
            raise ValueError(
                f"raw dependency {dependency_id!r} has inconsistent objects"
            )
        objects[dependency_id] = item
        refs.setdefault(
            dependency_id,
            _source_ref(
                object_kind="raw_sample_dependency",
                object_id=dependency_id,
                json_pointer=pointer,
                value=item,
            ),
        )

    prefix = f"/source_event_graphs/{source_index}/event_findings_v3"
    for finding_index, finding in enumerate(source["findings"]):
        for measurement_index, measurement in enumerate(finding["measurements"]):
            add(
                measurement["source_binding"]["raw_sample_dependency"],
                (
                    f"{prefix}/findings/{finding_index}/measurements/"
                    f"{measurement_index}/source_binding/raw_sample_dependency"
                ),
            )
    for waveform_index, waveform in enumerate(source["waveform_evidence"]):
        add(
            waveform["raw_sample_dependency"],
            f"{prefix}/waveform_evidence/{waveform_index}/raw_sample_dependency",
        )
    return objects, refs


def _finding_nodes(
    source: Mapping[str, Any], source_index: int
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
]:
    event_id = str(source["event_id"])
    dependency_objects, dependency_refs = _dependency_index(source, source_index)
    nodes: list[dict[str, Any]] = []
    node_by_evidence: dict[str, dict[str, Any]] = {}
    finding_refs: dict[str, dict[str, str]] = {}
    prefix = f"/source_event_graphs/{source_index}/event_findings_v3"
    for finding_index, finding in enumerate(source["findings"]):
        evidence_id = str(finding["evidence_id"])
        finding_ref = _source_ref(
            object_kind="finding",
            object_id=evidence_id,
            json_pointer=f"{prefix}/findings/{finding_index}",
            value=finding,
        )
        dependency_ids = [str(item) for item in finding["raw_sample_dependency_ids"]]
        missing = sorted(set(dependency_ids).difference(dependency_objects))
        if missing:
            raise ValueError(
                f"finding {evidence_id!r} references unavailable raw dependencies: {missing}"
            )
        node = {
            "node_id": _bounded_id(
                "FINDINGNODE", {"event_id": event_id, "evidence_id": evidence_id}
            ),
            "event_id": event_id,
            "evidence_id": evidence_id,
            "source_finding_ref": finding_ref,
            "assertion_level": str(finding["assertion_level"]),
            "finding_status": str(finding["status"]),
            "intrinsic_evidence_role": str(finding["intrinsic_evidence_role"]),
            "signal_temporal_context": str(finding["signal_temporal_context"]),
            "term_id": str(finding["term"]["term_id"]),
            "evaluation_opportunity_id": str(finding["evaluation_opportunity_id"]),
            "term_decision_receipt_id": (
                None
                if finding["term_decision_receipt_id"] is None
                else str(finding["term_decision_receipt_id"])
            ),
            "waveform_evidence_ids": [
                str(item) for item in finding["waveform_evidence_ids"]
            ],
            "raw_sample_dependencies": [
                deepcopy(dependency_objects[item]) for item in dependency_ids
            ],
            "source_finding_sha256": _sha256(finding),
        }
        if evidence_id in node_by_evidence:
            raise ValueError(f"duplicate finding evidence ID {evidence_id!r}")
        nodes.append(node)
        node_by_evidence[evidence_id] = node
        finding_refs[evidence_id] = finding_ref
    return nodes, node_by_evidence, finding_refs, dependency_refs


def _raw_dependency_ids_for_evidence(
    evidence_ids: Iterable[str], finding_nodes: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    return sorted(
        {
            str(dependency["dependency_id"])
            for evidence_id in evidence_ids
            for dependency in finding_nodes[str(evidence_id)][
                "raw_sample_dependencies"
            ]
        }
    )


def _future_free_onset_authorized(node: Mapping[str, Any]) -> bool:
    dependencies = node["raw_sample_dependencies"]
    return bool(dependencies) and all(
        dependency["dependency_status"]
        in {"bounded_past_and_present", "exact_instantaneous"}
        and dependency["view_role"] == "onset_causal"
        and dependency["dependency_policy"] == "past_and_present_only"
        and not dependency["future_sample_access"]
        and dependency["onset_evidence_authorized"]
        and dependency["onset_support_eligible"]
        for dependency in dependencies
    )


def _canonical_electrode(value: str) -> str | None:
    """Return one exact ontology electrode; never parse an opaque substring."""

    normalized = value.upper()
    allowed = _LEFT_ELECTRODES | _RIGHT_ELECTRODES | _MIDLINE_ELECTRODES
    return normalized if normalized in allowed else None


def _canonical_lead_endpoints(value: str) -> tuple[str, str] | None:
    """Parse only an exact two-endpoint ``A-B`` scalp lead identifier."""

    parts = value.upper().split("-")
    if len(parts) != 2:
        return None
    left = _canonical_electrode(parts[0])
    right = _canonical_electrode(parts[1])
    if left is None or right is None:
        return None
    return left, right


def _electrode_laterality(token: str) -> str | None:
    if token in _LEFT_ELECTRODES:
        return "left"
    if token in _RIGHT_ELECTRODES:
        return "right"
    if token in _MIDLINE_ELECTRODES:
        return "midline"
    return None


def _target_region(value: str) -> tuple[str, str | None]:
    normalized = value.lower().replace("-", "_")
    for prefix in ("left", "right", "midline", "bilateral"):
        marker = f"{prefix}_"
        if normalized.startswith(marker):
            return normalized[len(marker):], prefix
    return normalized, None


def _constructive_step(
    *, source_type: str, source_id: str, target_type: str, target_id: str
) -> dict[str, str] | None:
    source_normalized = source_id.upper()
    target_normalized = target_id.upper()
    if source_type == target_type and source_normalized == target_normalized:
        rule = "identity"
    elif source_type == "electrode" and target_type == "region":
        token = _canonical_electrode(source_id)
        if token is None:
            return None
        target_region, target_laterality = _target_region(target_id)
        if _ELECTRODE_TO_REGION.get(token) != target_region:
            return None
        if target_laterality is not None and (
            target_laterality == "bilateral"
            or _electrode_laterality(token) != target_laterality
        ):
            return None
        rule = "electrode_to_region"
    elif source_type == "electrode" and target_type == "laterality":
        token = _canonical_electrode(source_id)
        if token is None or _electrode_laterality(token) != target_id.lower():
            return None
        rule = "electrode_to_laterality"
    elif source_type == "lead" and target_type == "laterality":
        endpoints = _canonical_lead_endpoints(source_id)
        if endpoints is None:
            return None
        lateralities = {_electrode_laterality(token) for token in endpoints}
        lateralities.discard(None)
        if lateralities != {target_id.lower()}:
            return None
        rule = "same_side_lead_to_laterality"
    elif source_type == "region" and target_type == "laterality":
        _region, source_laterality = _target_region(source_id)
        if source_laterality != target_id.lower() or source_laterality == "bilateral":
            return None
        rule = "region_prefix_to_laterality"
    else:
        return None
    return {
        "source_unit_type": source_type,
        "source_unit_id": source_id,
        "target_resolution": target_type,
        "target_entity_id": target_id,
        "rule": rule,
    }


def _spatial_receipts(
    source: Mapping[str, Any],
    source_index: int,
    finding_nodes: Mapping[str, Mapping[str, Any]],
    finding_refs: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    event_id = str(source["event_id"])
    qualification = source["event_qualification"]
    if (
        qualification["status"] not in _QUALIFIED_EVENT_STATUSES
        or not qualification["supporting_evidence_ids"]
    ):
        # A causal field observation may remain structured event evidence, but
        # it is not an onset/SOZ-candidate claim without explicit ictal-pattern
        # qualification in the same source graph.
        return [], {}
    findings = {str(item["evidence_id"]): item for item in source["findings"]}
    receipts: list[dict[str, Any]] = []
    receipt_by_relation: dict[str, str] = {}
    prefix = f"/source_event_graphs/{source_index}/event_findings_v3"
    for relation_index, relation in enumerate(source["hypothesis_evidence_relations"]):
        if relation["relation"] != "supports" or relation["axis"] not in _SPATIAL_AXES:
            continue
        evidence_ids = [str(item) for item in relation["evidence_ids"]]
        if not evidence_ids or any(
            finding_nodes[item]["finding_status"] != "present"
            or finding_nodes[item]["intrinsic_evidence_role"] != "onset_eligible"
            or not _future_free_onset_authorized(finding_nodes[item])
            for item in evidence_ids
        ):
            continue
        target_type = str(relation["axis"])
        target_id = str(relation["candidate_id"])
        steps: list[dict[str, str]] = []
        spatial_refs: list[dict[str, str]] = []
        entailing_evidence_ids: list[str] = []
        for evidence_id in evidence_ids:
            finding = findings[evidence_id]
            local_match = False
            finding_index = next(
                index
                for index, item in enumerate(source["findings"])
                if item["evidence_id"] == evidence_id
            )
            for spatial_index, spatial in enumerate(finding["spatial_support"]):
                if not spatial["evidence_eligible"]:
                    continue
                if spatial["observation_status"] not in {"observed", "derived"}:
                    continue
                if spatial["mapping_status"] not in {"direct", "field_qualified"}:
                    continue
                step = _constructive_step(
                    source_type=str(spatial["unit_type"]),
                    source_id=str(spatial["id"]),
                    target_type=target_type,
                    target_id=target_id,
                )
                if step is None:
                    continue
                steps.append(step)
                spatial_refs.append(
                    _source_ref(
                        object_kind="spatial_support",
                        object_id=(
                            f"{evidence_id}:spatial:{spatial_index}"
                        ),
                        json_pointer=(
                            f"{prefix}/findings/{finding_index}/"
                            f"spatial_support/{spatial_index}"
                        ),
                        value=spatial,
                    )
                )
                local_match = True
            if local_match:
                entailing_evidence_ids.append(evidence_id)
        if not steps:
            continue
        relation_id = str(relation["relation_id"])
        relation_ref = _source_ref(
            object_kind="hypothesis_evidence_relation",
            object_id=relation_id,
            json_pointer=(
                f"{prefix}/hypothesis_evidence_relations/{relation_index}"
            ),
            value=relation,
        )
        receipt = {
            "receipt_id": _bounded_id(
                "SPATIALRECEIPT",
                {
                    "event_id": event_id,
                    "relation_id": relation_id,
                    "target_type": target_type,
                    "target_id": target_id,
                },
            ),
            "event_id": event_id,
            "source_relation_id": relation_id,
            "target_resolution": target_type,
            "target_entity_id": target_id,
            "supporting_evidence_ids": sorted(set(entailing_evidence_ids)),
            "source_refs": [
                _source_ref(
                    object_kind="event_qualification",
                    object_id=f"{event_id}:event_qualification",
                    json_pointer=f"{prefix}/event_qualification",
                    value=qualification,
                ),
                relation_ref,
                *[deepcopy(finding_refs[item]) for item in sorted(set(entailing_evidence_ids))],
                *spatial_refs,
            ],
            "entailment_path": steps,
            "decision": "constructively_entailed_at_selected_resolution",
            "receipt_sha256": "",
        }
        _seal(
            receipt,
            "receipt_sha256",
            "clinical-eeg-constructive-spatial-resolution-receipt-v2",
        )
        receipts.append(receipt)
        receipt_by_relation[relation_id] = str(receipt["receipt_id"])
    return receipts, receipt_by_relation


def _permission_edges(
    source: Mapping[str, Any],
    source_index: int,
    finding_nodes: Mapping[str, Mapping[str, Any]],
    finding_refs: Mapping[str, Mapping[str, str]],
    dependency_refs: Mapping[str, Mapping[str, str]],
    receipt_by_relation: Mapping[str, str],
    spatial_receipts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    event_id = str(source["event_id"])
    prefix = f"/source_event_graphs/{source_index}/event_findings_v3"
    result: list[dict[str, Any]] = []
    keys: set[tuple[object, ...]] = set()

    def emit(
        *,
        role: str,
        evidence_ids: Sequence[str],
        source_refs: Sequence[Mapping[str, str]],
        derivation_rule_id: str,
        spatial_receipt_id: str | None = None,
    ) -> None:
        ids = sorted(set(str(item) for item in evidence_ids))
        if not ids:
            return
        missing = sorted(set(ids).difference(finding_nodes))
        if missing:
            raise ValueError(f"permission edge references unknown evidence: {missing}")
        raw_ids = _raw_dependency_ids_for_evidence(ids, finding_nodes)
        key = (role, tuple(ids), derivation_rule_id, spatial_receipt_id)
        if key in keys:
            return
        keys.add(key)
        edge_id = _bounded_id(
            "PERMISSIONEDGE",
            {
                "event_id": event_id,
                "role": role,
                "evidence_ids": ids,
                "derivation_rule_id": derivation_rule_id,
                "spatial_receipt_id": spatial_receipt_id,
            },
        )
        result.append(
            {
                "permission_edge_id": edge_id,
                "role": role,
                "event_id": event_id,
                "evidence_ids": ids,
                "source_refs": [deepcopy(dict(item)) for item in source_refs],
                "derivation_rule_id": derivation_rule_id,
                "raw_sample_dependency_ids": raw_ids,
                "constructive_spatial_receipt_id": spatial_receipt_id,
                "authorization": "authorized_for_named_role_only",
            }
        )

    qualification = source["event_qualification"]
    qualification_ids = [str(item) for item in qualification["supporting_evidence_ids"]]
    if (
        qualification["status"] in _QUALIFIED_EVENT_STATUSES
        and qualification_ids
        and all(
            finding_nodes[item]["finding_status"] == "present"
            for item in qualification_ids
        )
    ):
        emit(
            role="ictal_pattern_qualification",
            evidence_ids=qualification_ids,
            source_refs=[
                _source_ref(
                    object_kind="event_qualification",
                    object_id=f"{event_id}:event_qualification",
                    json_pointer=f"{prefix}/event_qualification",
                    value=qualification,
                ),
                *[deepcopy(finding_refs[item]) for item in qualification_ids],
            ],
            derivation_rule_id="explicit_v3_event_qualification_support_v1",
        )

    for evidence_id, node in finding_nodes.items():
        raw_ids = [
            str(item["dependency_id"]) for item in node["raw_sample_dependencies"]
        ]
        dependency_source_refs = [deepcopy(dependency_refs[item]) for item in raw_ids]
        if (
            node["finding_status"] == "present"
            and node["intrinsic_evidence_role"] == "onset_eligible"
            and _future_free_onset_authorized(node)
        ):
            emit(
                role="onset_time_support",
                evidence_ids=[evidence_id],
                source_refs=[deepcopy(finding_refs[evidence_id]), *dependency_source_refs],
                derivation_rule_id="future_free_causal_onset_interval_v1",
            )
        if (
            node["finding_status"] == "present"
            and node["intrinsic_evidence_role"]
            in {"early_context", "later_involvement"}
        ):
            emit(
                role="course_or_spread_support",
                evidence_ids=[evidence_id],
                source_refs=[deepcopy(finding_refs[evidence_id]), *dependency_source_refs],
                derivation_rule_id=(
                    "typed_early_or_later_event_course_evidence_v1"
                ),
            )

    for relation_index, relation in enumerate(source["hypothesis_evidence_relations"]):
        relation_id = str(relation["relation_id"])
        evidence_ids = [str(item) for item in relation["evidence_ids"]]
        relation_ref = _source_ref(
            object_kind="hypothesis_evidence_relation",
            object_id=relation_id,
            json_pointer=(
                f"{prefix}/hypothesis_evidence_relations/{relation_index}"
            ),
            value=relation,
        )
        spatial_receipt_id = receipt_by_relation.get(relation_id)
        if relation["relation"] == "supports" and spatial_receipt_id is not None:
            entailing_ids = next(
                receipt["supporting_evidence_ids"]
                for receipt in spatial_receipts
                if receipt["receipt_id"] == spatial_receipt_id
            )
            raw_ids = _raw_dependency_ids_for_evidence(entailing_ids, finding_nodes)
            emit(
                role="onset_topography_support",
                evidence_ids=entailing_ids,
                source_refs=[
                    _source_ref(
                        object_kind="event_qualification",
                        object_id=f"{event_id}:event_qualification",
                        json_pointer=f"{prefix}/event_qualification",
                        value=qualification,
                    ),
                    relation_ref,
                    *[deepcopy(finding_refs[item]) for item in entailing_ids],
                    *[deepcopy(dependency_refs[item]) for item in raw_ids],
                ],
                derivation_rule_id=(
                    "future_free_onset_plus_constructive_spatial_entailment_v1"
                ),
                spatial_receipt_id=spatial_receipt_id,
            )
        if relation["relation"] == "contradicts":
            emit(
                role="counterevidence",
                evidence_ids=evidence_ids,
                source_refs=[
                    relation_ref,
                    *[deepcopy(finding_refs[item]) for item in evidence_ids],
                ],
                derivation_rule_id="explicit_v3_hypothesis_contradiction_v1",
            )

    for hypothesis_index, hypothesis in enumerate(
        source["competing_hypotheses"]["hypotheses"]
    ):
        evidence_ids = [
            str(item) for item in hypothesis["contradictory_evidence_ids"]
        ]
        if not evidence_ids:
            continue
        emit(
            role="counterevidence",
            evidence_ids=evidence_ids,
            source_refs=[
                _source_ref(
                    object_kind="competing_hypothesis",
                    object_id=str(hypothesis["hypothesis_id"]),
                    json_pointer=(
                        f"{prefix}/competing_hypotheses/hypotheses/"
                        f"{hypothesis_index}"
                    ),
                    value=hypothesis,
                ),
                *[deepcopy(finding_refs[item]) for item in evidence_ids],
            ],
            derivation_rule_id="explicit_v3_competing_hypothesis_counterevidence_v1",
        )
    return result


def _event_state_node(
    source: Mapping[str, Any], source_index: int
) -> dict[str, Any]:
    event_id = str(source["event_id"])
    prefix = f"/source_event_graphs/{source_index}/event_findings_v3"
    result = {
        "node_id": _bounded_id("EVENTSTATE", {"event_id": event_id}),
        "event_id": event_id,
        "event_qualification": deepcopy(source["event_qualification"]),
        "competing_hypotheses": deepcopy(source["competing_hypotheses"]),
        "event_outcome": deepcopy(source["event_outcome"]),
        "source_refs": [
            _source_ref(
                object_kind="event_qualification",
                object_id=f"{event_id}:event_qualification",
                json_pointer=f"{prefix}/event_qualification",
                value=source["event_qualification"],
            ),
            _source_ref(
                object_kind="competing_hypotheses",
                object_id=f"{event_id}:competing_hypotheses",
                json_pointer=f"{prefix}/competing_hypotheses",
                value=source["competing_hypotheses"],
            ),
            _source_ref(
                object_kind="event_outcome",
                object_id=f"{event_id}:event_outcome",
                json_pointer=f"{prefix}/event_outcome",
                value=source["event_outcome"],
            ),
        ],
        "state_sha256": "",
    }
    _seal(result, "state_sha256", "clinical-eeg-event-state-node-v2")
    return result


def _qualified_absence_receipt(
    source: Mapping[str, Any], finding: Mapping[str, Any]
) -> tuple[bool, dict[str, Any] | None, int | None]:
    if (
        finding["status"] != "absent_with_opportunity"
        or finding["assertion_level"] != "report_eligible_automated"
        or finding["term_decision_receipt_id"] is None
    ):
        return False, None, None
    receipt_id = str(finding["term_decision_receipt_id"])
    for index, receipt in enumerate(source["term_decision_receipts"]):
        if receipt["receipt_id"] != receipt_id:
            continue
        eligible = (
            receipt["term_id"] == finding["term"]["term_id"]
            and receipt["asserted_status"] == "absent_with_opportunity"
            and receipt["decision"] == "qualified"
            and receipt["capability_receipt_id"]
            == finding["capability_receipt_id"]
            and receipt["sensitivity_receipt_id"]
            == finding["sensitivity_receipt_id"]
        )
        return eligible, deepcopy(receipt), index
    return False, None, None


def _finding_render_disposition(
    source: Mapping[str, Any], finding: Mapping[str, Any]
) -> str:
    status = str(finding["status"])
    assertion = str(finding["assertion_level"])
    if status == "present":
        if assertion == "model_candidate":
            return "research_candidate_surface_allowed"
        return "positive_surface_allowed"
    if status == "absent_with_opportunity":
        eligible, _receipt, _index = _qualified_absence_receipt(source, finding)
        return (
            "explicit_absence_surface_allowed"
            if eligible
            else "structured_evidence_only"
        )
    if status == "uncertain":
        return "uncertainty_surface_only"
    return "not_evaluable_surface_only"


def _competing_hypothesis_surface_supported(
    hypothesis: Mapping[str, Any],
    *,
    event_id: str,
    present_finding_ids: set[str],
    permission_edges: Sequence[Mapping[str, Any]],
) -> bool:
    """Authorize only a completely covered, ictal-qualified differential."""

    supporting_evidence_ids = {
        str(item) for item in hypothesis["supporting_evidence_ids"]
    }
    if (
        hypothesis["disposition"] not in {"supported", "possible"}
        or not supporting_evidence_ids
        or not supporting_evidence_ids.issubset(present_finding_ids)
        or hypothesis["category"] != "cerebral_ictal"
        or not hypothesis["onset_claim_eligible"]
    ):
        return False
    supporting_permission_edges = [
        edge
        for edge in permission_edges
        if edge["event_id"] == event_id
        and edge["role"] != "counterevidence"
        and set(str(item) for item in edge["evidence_ids"]).intersection(
            supporting_evidence_ids
        )
    ]
    covered_support_ids = {
        str(evidence_id)
        for edge in supporting_permission_edges
        for evidence_id in edge["evidence_ids"]
    }
    return supporting_evidence_ids.issubset(covered_support_ids) and any(
        edge["role"] == "ictal_pattern_qualification"
        for edge in supporting_permission_edges
    )


def _claims(
    sources: Sequence[Mapping[str, Any]],
    finding_refs_by_event: Mapping[str, Mapping[str, Mapping[str, str]]],
    permission_edges: Sequence[Mapping[str, Any]],
    spatial_receipts: Sequence[Mapping[str, Any]],
    derivation_dag: Mapping[str, Any],
    record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    edge_ids_by_evidence: dict[tuple[str, str], list[str]] = {}
    for edge in permission_edges:
        for evidence_id in edge["evidence_ids"]:
            edge_ids_by_evidence.setdefault(
                (str(edge["event_id"]), str(evidence_id)), []
            ).append(str(edge["permission_edge_id"]))
    result: list[dict[str, Any]] = []
    permission_role_by_edge = {
        str(item["permission_edge_id"]): str(item["role"])
        for item in permission_edges
    }
    event_dag_node = {
        str(item["source_ids"][0]): item
        for item in derivation_dag["nodes"]
        if item["node_type"] == "event"
    }
    source_index_by_event = {
        str(source["event_id"]): index for index, source in enumerate(sources)
    }
    source_by_event = {str(source["event_id"]): source for source in sources}

    def root_ref(event_id: str) -> dict[str, str]:
        source = source_by_event[event_id]
        return _source_ref(
            object_kind="event_findings_v3",
            object_id=event_id,
            json_pointer=(
                f"/source_event_graphs/{source_index_by_event[event_id]}/"
                "event_findings_v3"
            ),
            value=source,
        )

    def add(row: dict[str, Any]) -> None:
        event_id = row.get("event_id")
        row.setdefault("layer", "event")
        row.setdefault(
            "dag_node_id",
            None if event_id is None else event_dag_node[str(event_id)]["node_id"],
        )
        row.setdefault("mode_id", None)
        row.setdefault("record_id", str(record["record_id"]))
        row.setdefault(
            "required_permission_roles",
            sorted(
                {
                    permission_role_by_edge[str(edge_id)]
                    for edge_id in row["permission_edge_ids"]
                }
            ),
        )
        row.setdefault("conclusion_authorization", "source_bound_atomic")
        row.setdefault("typed_conclusion", None)
        _seal(row, "claim_sha256", "clinical-eeg-report-graph-claim-v2")
        result.append(row)

    for source_index, source in enumerate(sources):
        event_id = str(source["event_id"])
        prefix = f"/source_event_graphs/{source_index}/event_findings_v3"
        finding_refs = finding_refs_by_event[event_id]
        for finding_index, finding in enumerate(source["findings"]):
            evidence_id = str(finding["evidence_id"])
            source_refs = [deepcopy(finding_refs[evidence_id])]
            absence_ok, absence_receipt, receipt_index = _qualified_absence_receipt(
                source, finding
            )
            if absence_ok and absence_receipt is not None and receipt_index is not None:
                source_refs.append(
                    _source_ref(
                        object_kind="term_decision_receipt",
                        object_id=str(absence_receipt["receipt_id"]),
                        json_pointer=(
                            f"{prefix}/term_decision_receipts/{receipt_index}"
                        ),
                        value=absence_receipt,
                    )
                )
            add(
                {
                    "claim_id": _bounded_id(
                        "CLAIM",
                        {
                            "kind": "finding_state",
                            "event_id": event_id,
                            "evidence_id": evidence_id,
                        },
                    ),
                    "claim_kind": "finding_state",
                    "event_id": event_id,
                    "source_evidence_ids": [evidence_id],
                    "source_refs": source_refs,
                    "assertion_level": str(finding["assertion_level"]),
                    "finding_status": str(finding["status"]),
                    "target_resolution": None,
                    "target_entity_id": None,
                    "permission_edge_ids": sorted(
                        edge_ids_by_evidence.get((event_id, evidence_id), [])
                    ),
                    "constructive_spatial_receipt_id": None,
                    "waveform_evidence_ids": [
                        str(item) for item in finding["waveform_evidence_ids"]
                    ],
                    "render_disposition": _finding_render_disposition(
                        source, finding
                    ),
                    "claim_sha256": "",
                }
            )

        event_outcome = source["event_outcome"]
        outcome = str(event_outcome["outcome"])
        outcome_render = (
            "research_candidate_surface_allowed"
            if outcome in _QUALIFIED_EVENT_STATUSES | {"candidate_only"}
            else "not_evaluable_surface_only"
            if outcome in {"obscured_by_artifact", "not_possible_to_determine"}
            else "structured_evidence_only"
        )
        add(
            {
                "claim_id": _bounded_id(
                    "CLAIM",
                    {"kind": "event_outcome", "event_id": event_id, "outcome": outcome},
                ),
                "claim_kind": "event_outcome",
                "event_id": event_id,
                "source_evidence_ids": [
                    str(item) for item in event_outcome["evidence_ids"]
                ],
                "source_refs": [
                    _source_ref(
                        object_kind="event_outcome",
                        object_id=f"{event_id}:event_outcome",
                        json_pointer=f"{prefix}/event_outcome",
                        value=event_outcome,
                    )
                ],
                "assertion_level": None,
                "finding_status": None,
                "target_resolution": None,
                "target_entity_id": None,
                "permission_edge_ids": sorted(
                    {
                        edge_id
                        for evidence_id in event_outcome["evidence_ids"]
                        for edge_id in edge_ids_by_evidence.get(
                            (event_id, str(evidence_id)), []
                        )
                    }
                ),
                "constructive_spatial_receipt_id": None,
                "waveform_evidence_ids": [],
                "render_disposition": outcome_render,
                "claim_sha256": "",
            }
        )
        for hypothesis_index, hypothesis in enumerate(
            source["competing_hypotheses"]["hypotheses"]
        ):
            supporting_evidence_ids = [
                str(item) for item in hypothesis["supporting_evidence_ids"]
            ]
            evidence_ids = sorted(
                {
                    *supporting_evidence_ids,
                    *[str(item) for item in hypothesis["contradictory_evidence_ids"]],
                }
            )
            present_finding_ids = {
                str(item["evidence_id"])
                for item in source["findings"]
                if item["status"] == "present"
            }
            surface_supported = _competing_hypothesis_surface_supported(
                hypothesis,
                event_id=event_id,
                present_finding_ids=present_finding_ids,
                permission_edges=permission_edges,
            )
            add(
                {
                    "claim_id": _bounded_id(
                        "CLAIM",
                        {
                            "kind": "competing_hypothesis",
                            "event_id": event_id,
                            "hypothesis_id": hypothesis["hypothesis_id"],
                        },
                    ),
                    "claim_kind": "competing_hypothesis",
                    "event_id": event_id,
                    "source_evidence_ids": evidence_ids,
                    "source_refs": [
                        _source_ref(
                            object_kind="competing_hypothesis",
                            object_id=str(hypothesis["hypothesis_id"]),
                            json_pointer=(
                                f"{prefix}/competing_hypotheses/hypotheses/"
                                f"{hypothesis_index}"
                            ),
                            value=hypothesis,
                        )
                    ],
                    "assertion_level": None,
                    "finding_status": None,
                    "target_resolution": None,
                    "target_entity_id": None,
                    "permission_edge_ids": sorted(
                        {
                            edge_id
                            for evidence_id in evidence_ids
                            for edge_id in edge_ids_by_evidence.get(
                                (event_id, evidence_id), []
                            )
                        }
                    ),
                    "constructive_spatial_receipt_id": None,
                    "waveform_evidence_ids": [],
                    "render_disposition": (
                        "research_candidate_surface_allowed"
                        if surface_supported
                        else "structured_evidence_only"
                    ),
                    "claim_sha256": "",
                }
            )

    edge_by_spatial_receipt = {
        str(edge["constructive_spatial_receipt_id"]): str(
            edge["permission_edge_id"]
        )
        for edge in permission_edges
        if edge["role"] == "onset_topography_support"
        and edge["constructive_spatial_receipt_id"] is not None
    }
    ictal_edge_ids_by_event: dict[str, list[str]] = {}
    for edge in permission_edges:
        if edge["role"] == "ictal_pattern_qualification":
            ictal_edge_ids_by_event.setdefault(str(edge["event_id"]), []).append(
                str(edge["permission_edge_id"])
            )
    for receipt in spatial_receipts:
        receipt_id = str(receipt["receipt_id"])
        event_id = str(receipt["event_id"])
        ictal_edge_ids = ictal_edge_ids_by_event.get(event_id, [])
        if not ictal_edge_ids:
            raise ValueError(
                "constructive spatial receipt lacks ictal-pattern qualification"
            )
        supporting_ids = set(str(item) for item in receipt["supporting_evidence_ids"])
        onset_time_edges = [
            edge
            for edge in permission_edges
            if edge["event_id"] == event_id
            and edge["role"] == "onset_time_support"
            and supporting_ids.intersection(
                str(item) for item in edge["evidence_ids"]
            )
        ]
        onset_time_covered = {
            str(evidence_id)
            for edge in onset_time_edges
            for evidence_id in edge["evidence_ids"]
        }
        if not supporting_ids.issubset(onset_time_covered):
            raise ValueError(
                "constructive spatial receipt lacks complete onset-time support"
            )
        onset_time_edge_ids = [
            str(edge["permission_edge_id"]) for edge in onset_time_edges
        ]
        add(
            {
                "claim_id": _bounded_id(
                    "CLAIM",
                    {"kind": "spatial_candidate", "receipt_id": receipt_id},
                ),
                "claim_kind": "scalp_onset_spatial_candidate",
                "event_id": event_id,
                "source_evidence_ids": deepcopy(receipt["supporting_evidence_ids"]),
                "source_refs": deepcopy(receipt["source_refs"]),
                "assertion_level": "model_candidate",
                "finding_status": "present",
                "target_resolution": str(receipt["target_resolution"]),
                "target_entity_id": str(receipt["target_entity_id"]),
                "permission_edge_ids": [
                    *ictal_edge_ids,
                    *onset_time_edge_ids,
                    edge_by_spatial_receipt[receipt_id],
                ],
                "constructive_spatial_receipt_id": receipt_id,
                "waveform_evidence_ids": [],
                "conclusion_authorization": "constructive_event_candidate",
                "render_disposition": "research_candidate_surface_allowed",
                "claim_sha256": "",
            }
        )

    for source in sources:
        event_id = str(source["event_id"])
        event_node = event_dag_node[event_id]
        event_edge_ids = [
            str(item["permission_edge_id"])
            for item in permission_edges
            if item["event_id"] == event_id
        ]
        add(
            {
                "claim_id": _bounded_id(
                    "CLAIM",
                    {"kind": "event_scalp_hypothesis", "event_id": event_id},
                ),
                "claim_kind": "event_scalp_onset_hypothesis",
                "event_id": event_id,
                "source_evidence_ids": sorted(
                    {
                        str(evidence_id)
                        for edge in permission_edges
                        if edge["event_id"] == event_id
                        for evidence_id in edge["evidence_ids"]
                    }
                ),
                "source_refs": [root_ref(event_id)],
                "assertion_level": "model_candidate",
                "finding_status": "present",
                "target_resolution": None,
                "target_entity_id": None,
                "permission_edge_ids": event_edge_ids,
                "constructive_spatial_receipt_id": None,
                "waveform_evidence_ids": [],
                "conclusion_authorization": (
                    "source_event_hypothesis_only_not_record_diagnosis"
                ),
                "typed_conclusion": deepcopy(event_node["typed_payload"]),
                "render_disposition": "structured_evidence_only",
                "claim_sha256": "",
            }
        )

    for dag_node in derivation_dag["nodes"]:
        node_type = str(dag_node["node_type"])
        if node_type not in {"mode", "record"}:
            continue
        if node_type == "mode":
            event_ids = [str(item) for item in dag_node["source_ids"]]
            claim_kind = "mode_conclusion_status"
            layer = "mode"
            mode_id: str | None = str(dag_node["node_id"])
        else:
            event_ids = [str(source["event_id"]) for source in sources]
            claim_kind = "record_impression_status"
            layer = "record"
            mode_id = None
        leaf_edges = [
            item
            for item in permission_edges
            if str(item["event_id"]) in set(event_ids)
        ]
        leaf_edge_ids = [str(item["permission_edge_id"]) for item in leaf_edges]
        add(
            {
                "claim_id": _bounded_id(
                    "CLAIM",
                    {"kind": claim_kind, "dag_node_id": dag_node["node_id"]},
                ),
                "claim_kind": claim_kind,
                "layer": layer,
                "dag_node_id": str(dag_node["node_id"]),
                "event_id": None,
                "mode_id": mode_id,
                "record_id": str(record["record_id"]),
                "source_evidence_ids": sorted(
                    {
                        str(evidence_id)
                        for edge in leaf_edges
                        for evidence_id in edge["evidence_ids"]
                    }
                ),
                "source_refs": (
                    [root_ref(event_id) for event_id in event_ids]
                    if event_ids
                    else [
                        _source_ref(
                            object_kind="record_context",
                            object_id=str(record["record_id"]),
                            json_pointer="/record",
                            value=record,
                        )
                    ]
                ),
                "assertion_level": None,
                "finding_status": None,
                "target_resolution": None,
                "target_entity_id": None,
                "permission_edge_ids": leaf_edge_ids,
                "constructive_spatial_receipt_id": None,
                "waveform_evidence_ids": [],
                "required_permission_roles": sorted(
                    {str(item["role"]) for item in leaf_edges}
                ),
                "conclusion_authorization": (
                    "not_authorized_missing_mode_aware_mil_receipt"
                ),
                "typed_conclusion": deepcopy(dag_node["typed_payload"]),
                "render_disposition": "not_evaluable_surface_only",
                "claim_sha256": "",
            }
        )
    return result


def _waveform_panels(
    sources: Sequence[Mapping[str, Any]], claims: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        event_id = str(source["event_id"])
        prefix = f"/source_event_graphs/{source_index}/event_findings_v3"
        findings = {str(item["evidence_id"]): item for item in source["findings"]}
        finding_index = {
            str(item["evidence_id"]): index
            for index, item in enumerate(source["findings"])
        }
        claim_by_evidence = {
            str(claim["source_evidence_ids"][0]): claim
            for claim in claims
            if claim["event_id"] == event_id
            and claim["claim_kind"] == "finding_state"
            and len(claim["source_evidence_ids"]) == 1
            and claim["finding_status"] == "present"
        }
        for waveform_index, waveform in enumerate(source["waveform_evidence"]):
            if not waveform["evidence_eligible"]:
                continue
            waveform_id = str(waveform["waveform_evidence_id"])
            evidence_ids = sorted(
                evidence_id
                for evidence_id, finding in findings.items()
                if finding["status"] == "present"
                and waveform_id in finding["waveform_evidence_ids"]
                and evidence_id in claim_by_evidence
            )
            if not evidence_ids:
                continue
            claim_ids = sorted(
                str(claim_by_evidence[item]["claim_id"]) for item in evidence_ids
            )
            claim_evidence_bindings = sorted(
                (
                    {
                        "claim_id": str(claim_by_evidence[evidence_id]["claim_id"]),
                        "finding_evidence_id": evidence_id,
                    }
                    for evidence_id in evidence_ids
                ),
                key=lambda item: (item["claim_id"], item["finding_evidence_id"]),
            )
            source_refs = [
                _source_ref(
                    object_kind="waveform_evidence",
                    object_id=waveform_id,
                    json_pointer=f"{prefix}/waveform_evidence/{waveform_index}",
                    value=waveform,
                ),
                *[
                    _source_ref(
                        object_kind="finding",
                        object_id=evidence_id,
                        json_pointer=(
                            f"{prefix}/findings/{finding_index[evidence_id]}"
                        ),
                        value=findings[evidence_id],
                    )
                    for evidence_id in evidence_ids
                ],
            ]
            dependency = waveform["raw_sample_dependency"]
            panel = {
                "panel_id": _bounded_id(
                    "WAVEPANEL",
                    {"event_id": event_id, "waveform_evidence_id": waveform_id},
                ),
                "event_id": event_id,
                "claim_ids": claim_ids,
                "finding_evidence_ids": evidence_ids,
                "claim_evidence_bindings": claim_evidence_bindings,
                "waveform_evidence_id": waveform_id,
                "source_refs": source_refs,
                "interval": deepcopy(waveform["interval"]),
                "unit_ids": [str(item) for item in waveform["unit_ids"]],
                "view_role": str(waveform["view_role"]),
                "raw_sample_dependency_id": (
                    None if dependency is None else str(dependency["dependency_id"])
                ),
                "render_authorization": "claim_and_evidence_double_closed_eeg_only",
                "panel_sha256": "",
            }
            _seal(panel, "panel_sha256", "clinical-eeg-waveform-panel-v2")
            result.append(panel)
    return result


def _mode_signature(source: Mapping[str, Any]) -> dict[str, Any]:
    hypothesis = source["scalp_onset_hypothesis"]
    selected_hypothesis_id = source["competing_hypotheses"][
        "selected_hypothesis_id"
    ]
    selected_competing = next(
        (
            item
            for item in source["competing_hypotheses"]["hypotheses"]
            if item["hypothesis_id"] == selected_hypothesis_id
        ),
        None,
    )
    candidates = sorted(
        (
            {
                "candidate_type": str(item["candidate_type"]),
                "candidate_id": str(item["candidate_id"]),
                "rank": int(item["rank"]),
            }
            for item in hypothesis["candidate_scores"]
            if int(item["rank"]) == 1
        ),
        key=lambda item: (item["candidate_type"], item["candidate_id"]),
    )
    return {
        "event_outcome": str(source["event_outcome"]["outcome"]),
        "localization_status": str(hypothesis["localization_status"]),
        "selected_resolution": str(hypothesis["selected_resolution"]),
        "phenotype": hypothesis["phenotype"],
        "rank_one_candidates": candidates,
        "selected_competing_category": (
            None if selected_competing is None else selected_competing["category"]
        ),
        "selected_competing_term_id": (
            None if selected_competing is None else selected_competing["term_id"]
        ),
    }


def _derivation_dag(
    sources: Sequence[Mapping[str, Any]],
    record: Mapping[str, Any],
    permission_edges: Sequence[Mapping[str, Any]],
    spatial_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    event_nodes: list[dict[str, Any]] = []
    event_node_ids: dict[str, str] = {}
    root_refs: dict[str, dict[str, str]] = {}
    signature_groups: dict[str, list[str]] = {}
    for index, source in enumerate(sources):
        event_id = str(source["event_id"])
        event_node_id = _bounded_id("DAGEVENT", {"event_id": event_id})
        event_node_ids[event_id] = event_node_id
        signature = _mode_signature(source)
        signature_sha = _sha256(signature)
        signature_groups.setdefault(signature_sha, []).append(event_id)
        root_refs[event_id] = _source_ref(
            object_kind="event_findings_v3",
            object_id=event_id,
            json_pointer=f"/source_event_graphs/{index}/event_findings_v3",
            value=source,
        )
        typed_payload = {
            "payload_type": "event",
            "conclusion_authorization": (
                "source_event_hypothesis_only_not_record_diagnosis"
            ),
            "scalp_onset_hypothesis": deepcopy(
                source["scalp_onset_hypothesis"]
            ),
            "event_outcome": deepcopy(source["event_outcome"]),
            "source_permission_edge_ids": [
                str(item["permission_edge_id"])
                for item in permission_edges
                if item["event_id"] == event_id
            ],
            "constructive_spatial_receipt_ids": [
                str(item["receipt_id"])
                for item in spatial_receipts
                if item["event_id"] == event_id
            ],
        }
        event_nodes.append(
            {
                "node_id": event_node_id,
                "node_type": "event",
                "source_ids": [event_id],
                "typed_payload": typed_payload,
                "node_value_sha256": _sha256(typed_payload),
            }
        )

    mode_nodes: list[dict[str, Any]] = []
    mode_id_by_signature: dict[str, str] = {}
    for signature_sha, event_ids in signature_groups.items():
        mode_id = _bounded_id(
            "DAGMODE",
            {"signature_sha256": signature_sha, "event_ids": event_ids},
        )
        mode_id_by_signature[signature_sha] = mode_id
        typed_payload = {
            "payload_type": "mode",
            "conclusion_authorization": (
                "not_authorized_missing_mode_aware_mil_receipt"
            ),
            "event_ids": deepcopy(event_ids),
            "source_signature_sha256": signature_sha,
            "phenotype": None,
            "selected_resolution": None,
            "ranked_candidates": [],
            "uncertainty_status": (
                "not_assessed_without_mode_aware_mil_receipt"
            ),
            "spatial_resolution_status": (
                "not_authorized_no_constructive_mode_receipt"
            ),
            "required_receipt_type": (
                "patient_disjoint_calibrated_mode_aware_mil_receipt"
            ),
            "authorization_receipt_id": None,
        }
        mode_nodes.append(
            {
                "node_id": mode_id,
                "node_type": "mode",
                "source_ids": deepcopy(event_ids),
                "typed_payload": typed_payload,
                "node_value_sha256": _sha256(typed_payload),
            }
        )
    record_node_id = _bounded_id(
        "DAGRECORD", {"record_id": record["record_id"]}
    )
    record_typed_payload = {
        "payload_type": "record",
        "conclusion_authorization": (
            "not_authorized_missing_mode_aware_mil_receipt"
        ),
        "mode_node_ids": [item["node_id"] for item in mode_nodes],
        "phenotype": None,
        "selected_resolution": None,
        "ranked_candidates": [],
        "uncertainty_status": (
            "not_assessed_without_mode_aware_mil_receipt"
        ),
        "spatial_resolution_status": (
            "not_authorized_no_constructive_record_receipt"
        ),
        "required_receipt_type": (
            "patient_disjoint_calibrated_mode_aware_mil_receipt"
        ),
        "authorization_receipt_id": None,
    }
    record_node = {
        "node_id": record_node_id,
        "node_type": "record",
        "source_ids": [item["node_id"] for item in mode_nodes],
        "typed_payload": record_typed_payload,
        "node_value_sha256": _sha256(record_typed_payload),
    }
    edges: list[dict[str, Any]] = []
    for source in sources:
        event_id = str(source["event_id"])
        signature_sha = _sha256(_mode_signature(source))
        mode_id = mode_id_by_signature[signature_sha]
        edges.append(
            {
                "edge_id": _bounded_id(
                    "DAGEDGE",
                    {"event_id": event_id, "mode_id": mode_id},
                ),
                "source_node_id": event_node_ids[event_id],
                "target_node_id": mode_id,
                "relation": "event_to_mode",
                "source_refs": [deepcopy(root_refs[event_id])],
                "derivation_rule_id": "exact_eeg_source_signature_partition_v1",
            }
        )
    for mode_node in mode_nodes:
        mode_event_ids = [str(item) for item in mode_node["source_ids"]]
        edges.append(
            {
                "edge_id": _bounded_id(
                    "DAGEDGE",
                    {"mode_id": mode_node["node_id"], "record_id": record["record_id"]},
                ),
                "source_node_id": str(mode_node["node_id"]),
                "target_node_id": record_node_id,
                "relation": "mode_to_record",
                "source_refs": [deepcopy(root_refs[item]) for item in mode_event_ids],
                "derivation_rule_id": "complete_mode_roster_to_record_v1",
            }
        )
    dag = {
        "mode_semantics": (
            "exact_eeg_source_signature_partition_not_calibrated_clinical_mode_inference"
        ),
        "nodes": [*event_nodes, *mode_nodes, record_node],
        "edges": edges,
        "topological_order": [
            *[item["node_id"] for item in event_nodes],
            *[item["node_id"] for item in mode_nodes],
            record_node_id,
        ],
        "dag_sha256": "",
    }
    _seal(dag, "dag_sha256", "clinical-eeg-event-mode-record-dag-v2")
    return dag


def materialize_multievent_soz_report_graph_v2(
    event_findings_v3: Sequence[object],
    *,
    route_scope: str = "synthetic",
    record_context: Mapping[str, object] | None = None,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Materialize a deterministic, lossless v3-to-report-graph v2 shadow."""

    if route_scope not in _ROUTE_SCOPES:
        raise ValueError("report graph v2 route_scope must be public or synthetic")
    if isinstance(event_findings_v3, (str, bytes)) or not isinstance(
        event_findings_v3, Sequence
    ):
        raise TypeError("event_findings_v3 must be an ordered sequence")
    kwargs = _validation_kwargs(
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    sources = [
        validate_event_eeg_findings_v3_payload(item, **kwargs)
        for item in event_findings_v3
    ]
    source_exclusions = [
        _require_explicit_eeg_only_exclusions(
            source["provenance"]["inference_exclusions"],
            context=(
                f"event_findings_v3[{index}].provenance.inference_exclusions"
            ),
        )
        for index, source in enumerate(sources)
    ]
    event_ids = [str(item["event_id"]) for item in sources]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event_findings_v3 contains duplicate event IDs")

    if sources:
        first = sources[0]
        record = {
            "record_id": str(first["provenance"]["record_id"]),
            "canonical_signal_sha256": str(
                first["provenance"]["canonical_signal_sha256"]
            ),
            "recording_duration_seconds": float(
                first["coordinates"]["recording_duration_seconds"]
            ),
            "source_inference_exclusions": deepcopy(source_exclusions[0]),
        }
        for source_index, source in enumerate(sources[1:], start=1):
            candidate = {
                "record_id": str(source["provenance"]["record_id"]),
                "canonical_signal_sha256": str(
                    source["provenance"]["canonical_signal_sha256"]
                ),
                "recording_duration_seconds": float(
                    source["coordinates"]["recording_duration_seconds"]
                ),
                "source_inference_exclusions": deepcopy(
                    source_exclusions[source_index]
                ),
            }
            if candidate != record:
                raise ValueError("all event graphs must bind the same EEG record")
        if record_context is not None and _validate_record_context(record_context) != record:
            raise ValueError("record_context conflicts with embedded event graphs")
    else:
        if record_context is None:
            raise ValueError("zero-event graph materialization requires record_context")
        record = _validate_record_context(record_context)

    source_wrappers = [
        {
            "event_id": str(source["event_id"]),
            "source_event_findings_v3_sha256": _sha256(source),
            "event_findings_v3": deepcopy(source),
        }
        for source in sources
    ]
    source_roster_sha256 = _sha256(
        {
            "binding_domain": "clinical-eeg-report-graph-v2-source-roster",
            "record": record,
            "ordered_event_sources": [
                {
                    "event_id": item["event_id"],
                    "source_event_findings_v3_sha256": item[
                        "source_event_findings_v3_sha256"
                    ],
                }
                for item in source_wrappers
            ],
        }
    )

    all_finding_nodes: list[dict[str, Any]] = []
    event_state_nodes: list[dict[str, Any]] = []
    all_spatial_receipts: list[dict[str, Any]] = []
    all_permission_edges: list[dict[str, Any]] = []
    finding_refs_by_event: dict[str, dict[str, dict[str, str]]] = {}
    for source_index, source in enumerate(sources):
        event_id = str(source["event_id"])
        nodes, node_map, finding_refs, dependency_refs = _finding_nodes(
            source, source_index
        )
        spatial_receipts, receipt_by_relation = _spatial_receipts(
            source, source_index, node_map, finding_refs
        )
        permission_edges = _permission_edges(
            source,
            source_index,
            node_map,
            finding_refs,
            dependency_refs,
            receipt_by_relation,
            spatial_receipts,
        )
        all_finding_nodes.extend(nodes)
        event_state_nodes.append(_event_state_node(source, source_index))
        all_spatial_receipts.extend(spatial_receipts)
        all_permission_edges.extend(permission_edges)
        finding_refs_by_event[event_id] = finding_refs

    role_roster = []
    for role in _ROLE_ORDER:
        edge_ids = [
            str(item["permission_edge_id"])
            for item in all_permission_edges
            if item["role"] == role
        ]
        role_roster.append(
            {
                "role": role,
                "status": (
                    "materialized" if edge_ids else "not_expressed_by_source"
                ),
                "edge_ids": edge_ids,
            }
        )

    derivation_dag = _derivation_dag(
        sources,
        record,
        all_permission_edges,
        all_spatial_receipts,
    )
    claims = _claims(
        sources,
        finding_refs_by_event,
        all_permission_edges,
        all_spatial_receipts,
        derivation_dag,
        record,
    )
    panels = _waveform_panels(sources, claims)
    graph: dict[str, Any] = {
        "schema_version": MULTIEVENT_SOZ_REPORT_GRAPH_V2_SCHEMA_VERSION,
        "graph_id": _bounded_id(
            "REPORTGRAPH",
            {"record_id": record["record_id"], "source_roster": source_roster_sha256},
        ),
        "route_boundary": {
            "route_id": MULTIEVENT_SOZ_REPORT_GRAPH_V2_ROUTE_ID,
            "route_scope": route_scope,
            "claim_boundary": _CLAIM_BOUNDARY,
            "eeg_signal_only": True,
            "embedded_sources_are_untrusted_until_revalidated": True,
            "private_route_connected": (
                MULTIEVENT_SOZ_REPORT_GRAPH_V2_PRIVATE_ROUTE_CONNECTED
            ),
            "qwen_production_route_connected": (
                MULTIEVENT_SOZ_REPORT_GRAPH_V2_QWEN_PRODUCTION_ROUTE_CONNECTED
            ),
            "clinical_use_authorized": False,
            "inference_exclusions": dict(_INFERENCE_EXCLUSIONS),
        },
        "record": record,
        "source_event_graphs": source_wrappers,
        "source_event_roster_sha256": source_roster_sha256,
        "finding_evidence_nodes": all_finding_nodes,
        "event_state_nodes": event_state_nodes,
        "evidence_permission_role_roster": role_roster,
        "evidence_permission_edges": all_permission_edges,
        "constructive_spatial_resolution_receipts": all_spatial_receipts,
        "derivation_dag": derivation_dag,
        "claims": claims,
        "waveform_panels": panels,
        "lexicalization_policy": {
            "qwen_role": "closed_claim_graph_lexicalizer_only",
            "qwen_may_receive_raw_eeg_or_external_context": False,
            "qwen_may_add_or_strengthen_claims": False,
            "qwen_may_change_negation_or_uncertainty": False,
            "qwen_may_change_spatial_resolution": False,
            "deterministic_fallback_required": True,
            "generation_failure_may_cancel_report": False,
            "absent_surface_requires_qualified_term_receipt": True,
            "waveform_panels_require_claim_and_evidence_closure": True,
        },
        "graph_sha256": "",
    }
    _seal(graph, "graph_sha256", "clinical-eeg-multievent-soz-report-graph-v2")
    return graph


def _trusted_source_map(
    value: Mapping[str, object] | Sequence[object],
) -> tuple[dict[str, object], list[str] | None]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}, None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("trusted_source_event_findings_v3 must be a mapping or sequence")
    result: dict[str, object] = {}
    order: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("trusted source event graph must be an object")
        event_id = str(item.get("event_id", ""))
        if not event_id or event_id in result:
            raise ValueError("trusted source event roster has invalid or duplicate IDs")
        result[event_id] = item
        order.append(event_id)
    return result, order


def validate_multievent_soz_report_graph_v2(
    payload: object,
    *,
    trusted_source_event_findings_v3: (
        Mapping[str, object] | Sequence[object] | None
    ) = None,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Fail closed and source-replay every report graph v2 field."""

    if type(payload) is not dict:
        raise TypeError("report graph v2 payload must be an object")
    data = deepcopy(payload)
    _reject_nonfinite(data)
    errors = sorted(
        _schema_validator().iter_errors(data),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        first = errors[0]
        raise ValueError(
            "report graph v2 schema validation failed at "
            f"{_schema_path(first)}: {first.message}"
        )
    kwargs = _validation_kwargs(
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    embedded_sources = [
        validate_event_eeg_findings_v3_payload(
            item["event_findings_v3"], **kwargs
        )
        for item in data["source_event_graphs"]
    ]
    embedded_ids = [str(item["event_id"]) for item in embedded_sources]
    if trusted_source_event_findings_v3 is not None:
        trusted_map, trusted_order = _trusted_source_map(
            trusted_source_event_findings_v3
        )
        if set(trusted_map) != set(embedded_ids):
            raise ValueError("trusted source event roster does not close embedded roster")
        if trusted_order is not None and trusted_order != embedded_ids:
            raise ValueError("trusted source event order differs from embedded roster")
        for source in embedded_sources:
            event_id = str(source["event_id"])
            trusted_validated = validate_event_eeg_findings_v3_payload(
                trusted_map[event_id], **kwargs
            )
            if source != trusted_validated:
                raise ValueError(
                    f"embedded event {event_id!r} does not replay trusted v3 source"
                )

    rebuilt = materialize_multievent_soz_report_graph_v2(
        embedded_sources,
        route_scope=str(data["route_boundary"]["route_scope"]),
        record_context=data["record"],
        **kwargs,
    )
    if data != rebuilt:
        raise ValueError(
            "report graph v2 does not replay from independently validated v3 sources"
        )
    return data


def replay_event_findings_v3_from_report_graph_v2(
    payload: object,
    **validation_kwargs: object,
) -> list[dict[str, Any]]:
    """Return exact ordered v3 sources after full graph validation."""

    validated = validate_multievent_soz_report_graph_v2(
        payload, **validation_kwargs
    )
    return [
        deepcopy(item["event_findings_v3"])
        for item in validated["source_event_graphs"]
    ]


__all__ = [
    "MULTIEVENT_SOZ_REPORT_GRAPH_V2_PRIVATE_ROUTE_CONNECTED",
    "MULTIEVENT_SOZ_REPORT_GRAPH_V2_QWEN_PRODUCTION_ROUTE_CONNECTED",
    "MULTIEVENT_SOZ_REPORT_GRAPH_V2_ROUTE_ID",
    "MULTIEVENT_SOZ_REPORT_GRAPH_V2_SCHEMA_VERSION",
    "materialize_multievent_soz_report_graph_v2",
    "replay_event_findings_v3_from_report_graph_v2",
    "validate_multievent_soz_report_graph_v2",
]
