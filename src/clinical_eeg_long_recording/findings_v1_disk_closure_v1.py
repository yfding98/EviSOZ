"""Source-replayable disk closure for the ten Findings-v1 required queries.

This module closes an engineering path that was intentionally left open by
the Findings-v1 composer addendum.  It turns validated native-waveform and
proposal artifacts into ten deterministic query artifacts, binds those
artifacts to the event-only twelve-slot card projection, and projects only
future-free onset evidence into a research scalp-onset/SOZ EvidenceGraph.

The closure is deliberately narrower than clinical qualification:

* record-level interictal material remains solely in the record context card;
* all query artifacts remain measured or research candidates;
* uncalibrated scores are never serialized as probabilities;
* untrusted upstream candidate/spatial numeric scores never enter ranking;
* ranking is replayed only as a typed count of unique, candidate-matched,
  future-free causal evidence leaves with complete raw-dependency receipts;
* clinical/report allowlists remain empty;
* course, evolution, later-involvement and record-context evidence cannot
  become positive onset/SOZ evidence;
* every logical object and every on-disk JSON file is content addressed and
  replayed from its typed sources.

The public writer is append-only/no-clobber.  The reader rejects symlinks,
unknown files, duplicate JSON keys, non-canonical file bytes and any source or
dependency drift.  It performs no EDF/annotation/spreadsheet I/O.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .acns_frequency_evolution_candidate import ACNSFrequencyEvolutionCandidate
from .ba_ieg_multireference_field import (
    validate_ba_ieg_multireference_field_result,
)
from .deterministic_event_morphology_primitives_v1 import (
    validate_event_morphology_primitive_supervision_v1,
)
from .deterministic_periodicity_candidate import (
    validate_deterministic_periodicity_candidate,
)
from .event_baseline_context_comparability import (
    validate_event_baseline_context_comparability_receipt,
)
from .event_card_projection_v2 import (
    materialize_event_card_projection_v2,
    validate_event_card_projection_v2,
)
from .event_component_cycle_element_query_adapter_v1 import (
    EVENT_COMPONENT_CYCLE_ELEMENT_QUERY_ADAPTER_METHOD_ID,
    S06_REQUIRED_QUERY_IDS,
    materialize_event_component_cycle_element_query_adapter_v1,
)
from .event_evolution_recovery_query_bridge_v1 import (
    EventChangePointProposal,
    TQ_EVOLUTION_FREQUENCY,
    TQ_EVOLUTION_LOCATION,
    TQ_EVOLUTION_MORPHOLOGY,
    TQ_RETURN_COMPARABLE_BACKGROUND,
    compose_frequency_evolution_query_ledger_v1,
    compose_location_evolution_query_ledger_v1,
    compose_morphology_evolution_query_ledger_v1,
    compose_return_to_comparable_background_query_ledger_v1,
    validate_event_evolution_recovery_query_ledger_v1,
)
from .event_findings_v3_validation import validate_event_eeg_findings_v3_payload
from .event_physical_amplitude_query_adapter_v1 import (
    EVENT_PHYSICAL_AMPLITUDE_QUERY_ADAPTER_METHOD_ID,
    S04_REQUIRED_QUERY_IDS,
    materialize_event_physical_amplitude_query_adapter_v1,
)
from .event_waveform_rhythm_query_bridge_v1 import (
    materialize_event_waveform_rhythm_query_bridge_v1,
    validate_event_waveform_rhythm_query_bridge_v1,
)
from .record_non_event_context_card_v1 import (
    validate_record_non_event_context_card_v1,
)


FINDINGS_V1_REQUIRED_QUERY_BUNDLE_SCHEMA_VERSION = (
    "clinical_eeg_findings_v1_required_query_bundle_v1"
)
FINDINGS_TO_SOZ_EVIDENCE_GRAPH_SCHEMA_VERSION = (
    "clinical_eeg_findings_to_research_soz_evidence_graph_v1"
)
FINDINGS_V1_DISK_CLOSURE_SCHEMA_VERSION = "clinical_eeg_findings_v1_disk_closure_v1"
FINDINGS_V1_DISK_MANIFEST_SCHEMA_VERSION = (
    "clinical_eeg_findings_v1_disk_closure_manifest_v1"
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHA_RE = re.compile(r"^[a-f0-9]{64}$")
_QUERY_FILENAME_RE = re.compile(r"^TQ-[A-Z0-9-]+\.json$")

REQUIRED_QUERY_SLOT: Mapping[str, str] = {
    "TQ-EVENT-AMPLITUDE-COURSE": "S04_PHYSICAL_AMPLITUDE",
    "TQ-EVENT-RHYTHMICITY-COURSE": "S06_RHYTHMICITY_PERIODICITY",
    TQ_EVOLUTION_FREQUENCY: "S09_CHANGE_POINTS_EVOLUTION",
    TQ_EVOLUTION_LOCATION: "S09_CHANGE_POINTS_EVOLUTION",
    TQ_EVOLUTION_MORPHOLOGY: "S09_CHANGE_POINTS_EVOLUTION",
    "TQ-PERIODIC-ELEMENT-INSTANCE": "S06_RHYTHMICITY_PERIODICITY",
    "TQ-PHYSICAL-AMPLITUDE-PROFILE": "S04_PHYSICAL_AMPLITUDE",
    TQ_RETURN_COMPARABLE_BACKGROUND: "S12_COMPARABLE_BACKGROUND_RECOVERY",
    "TQ-RHYTHMIC-RUN-INSTANCE": "S06_RHYTHMICITY_PERIODICITY",
    "TQ-SHARP-CONTOURED-ICTAL-COMPONENT-INSTANCE": (
        "S05_WAVEFORM_MORPHOLOGY"
    ),
}
REQUIRED_QUERY_IDS = tuple(sorted(REQUIRED_QUERY_SLOT))

_QUERY_AUTHORIZATION: Mapping[str, object] = {
    "deterministic_or_research_candidate_only": True,
    "clinical_term_qualification_authorized": False,
    "clinical_absence_authorized": False,
    "onset_support_authorized": False,
    "soz_support_authorized": False,
    "report_promotion_authorized": False,
    "qwen_authorized": False,
    "report_eligible_term_allowlist": [],
}
_GRAPH_AUTHORIZATION: Mapping[str, object] = {
    "research_scalp_visible_onset_ranking_only": True,
    "clinical_soz_claim_authorized": False,
    "cortical_soz_or_ez_claim_authorized": False,
    "bipolar_endpoint_attribution_authorized": False,
    "late_or_context_positive_support_authorized": False,
    "source_candidate_score_reuse_authorized": False,
    "ranking_requires_replayable_future_free_causal_score_leaves": True,
    "uncalibrated_probability_authorized": False,
    "report_promotion_authorized": False,
    "qwen_authorized": False,
    "report_eligible_term_allowlist": [],
}
_CAUSAL_SCORE_POLICY: Mapping[str, object] = {
    "policy_id": "FUTURE-FREE-CAUSAL-SPATIAL-LEAF-COUNT-V1",
    "candidate_axes": ["laterality", "region", "lead", "electrode"],
    "finding_contract": {
        "status": "present",
        "intrinsic_evidence_role": "onset_eligible",
        "signal_temporal_context": "candidate_emergence",
    },
    "raw_dependency_contract": {
        "minimum_receipt_count": 1,
        "view_role": "onset_causal",
        "dependency_policy": "past_and_present_only",
        "future_sample_access": False,
        "onset_evidence_authorized": True,
        "onset_support_eligible": True,
    },
    "spatial_support_contract": {
        "candidate_type_and_id_must_match": True,
        "evidence_eligible": True,
        "source_support_score_consumed": False,
    },
    "leaf_value_formula": "one_unit_mass_per_unique_permitted_evidence_leaf",
    "candidate_aggregation_formula": (
        "math.fsum(unique_per_evidence_leaf_unit_masses)"
    ),
    "rank_formula": (
        "per_axis_descending_recomputed_score_then_candidate_id"
    ),
    "disallowed_score_inputs": [
        "source_candidate_score",
        "source_candidate_rank",
        "source_finding_spatial_support_score",
        "course_or_evolution_query",
        "later_involvement_finding",
        "offline_or_record_context",
    ],
    "uncalibrated_not_probability": True,
}
_FIREWALL: Mapping[str, bool] = {
    "eeg_signal_and_allowlisted_acquisition_metadata_only": True,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_or_behavior_used": False,
    "sleep_staging_used": False,
    "provocation_used": False,
    "ecg_emg_eog_used": False,
    "qwen_used": False,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_file_bytes(value: object) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _domain_sha256(domain: str, value: object) -> str:
    return _sha256({"domain": domain, "value": value})


def _self_hash(value: Mapping[str, object], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _sha256(body)


def _identifier(value: object, context: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a canonical identifier")
    return value


def _hash(value: object, context: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _reject_nonfinite(value: object, context: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{context}[{index}]")


def _no_duplicate_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json_bytes(value: bytes, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(value, object_pairs_hook=_no_duplicate_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not strict UTF-8 JSON") from error
    if type(payload) is not dict:
        raise TypeError(f"{context} must contain one JSON object")
    if value != _canonical_file_bytes(payload):
        raise ValueError(f"{context} is not canonical JSON plus one newline")
    return payload


def _frequency_candidate_to_dict(candidate: ACNSFrequencyEvolutionCandidate) -> dict[str, Any]:
    if not isinstance(candidate, ACNSFrequencyEvolutionCandidate):
        raise TypeError("frequency_candidate must be ACNSFrequencyEvolutionCandidate")
    payload = candidate.to_dict()
    if payload["receipt_sha256"] != candidate.receipt_sha256:
        raise ValueError("frequency candidate receipt drifted")
    return deepcopy(payload)


def _frequency_candidate_from_dict(value: object) -> ACNSFrequencyEvolutionCandidate:
    if type(value) is not dict:
        raise TypeError("frequency candidate source must be an object")
    payload = deepcopy(value)
    expected_keys = {
        "schema_version",
        "event_id",
        "source_binding_sha256",
        "policy_id",
        "policy_sha256",
        "assertion_level",
        "status",
        "controlled_term",
        "clinical_term_qualified",
        "target_domain_qualification_required",
        "intrinsic_evidence_role",
        "onset_support_eligible",
        "soz_support_eligible",
        "amplitude_only_change_used",
        "reason_codes",
        "states",
        "transitions",
        "selected_state_ids",
        "selected_transition_indices",
        "direction",
        "minimum_conservative_frequency_step_hz",
        "minimum_conservative_cycles_per_state",
        "maximum_selected_unchanged_seconds",
        "raw_dependency_sha256s",
        "scope_receipt",
        "receipt_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("frequency candidate source fields drifted")
    candidate = ACNSFrequencyEvolutionCandidate(
        event_id=payload["event_id"],
        source_binding_sha256=payload["source_binding_sha256"],
        policy_sha256=payload["policy_sha256"],
        status=payload["status"],
        reason_codes=tuple(payload["reason_codes"]),
        states=tuple(payload["states"]),
        transitions=tuple(payload["transitions"]),
        selected_state_ids=tuple(payload["selected_state_ids"]),
        selected_transition_indices=tuple(payload["selected_transition_indices"]),
        direction=payload["direction"],
        minimum_conservative_frequency_step_hz=(
            payload["minimum_conservative_frequency_step_hz"]
        ),
        minimum_conservative_cycles_per_state=(
            payload["minimum_conservative_cycles_per_state"]
        ),
        maximum_selected_unchanged_seconds=(
            payload["maximum_selected_unchanged_seconds"]
        ),
        raw_dependency_sha256s=tuple(payload["raw_dependency_sha256s"]),
    )
    if candidate.to_dict() != payload:
        raise ValueError("frequency candidate does not replay from disk source")
    return candidate


def _change_point_to_dict(value: EventChangePointProposal) -> dict[str, Any]:
    if not isinstance(value, EventChangePointProposal):
        raise TypeError("morphology change points must be EventChangePointProposal")
    return deepcopy(value.to_dict())


def _change_point_from_dict(value: object) -> EventChangePointProposal:
    if type(value) is not dict:
        raise TypeError("change-point source must be an object")
    payload = deepcopy(value)
    candidate = EventChangePointProposal(
        proposal_id=payload["proposal_id"],
        event_id=payload["event_id"],
        source_signal_sha256=payload["source_signal_sha256"],
        change_interval_seconds=tuple(payload["change_interval_seconds"]),
        proposal_status=payload["proposal_status"],
        source_receipt_sha256=payload["source_receipt_sha256"],
        source_evidence_id=payload["source_evidence_id"],
        source_temporal_role=payload["source_temporal_role"],
        future_sample_access=payload["future_sample_access"],
        boundary_resolution_seconds=payload["boundary_resolution_seconds"],
        source_binding_sha256s=tuple(payload["source_binding_sha256s"]),
        raw_dependency_sha256s=tuple(payload["raw_dependency_sha256s"]),
    )
    if candidate.to_dict() != payload:
        raise ValueError("change-point proposal does not replay from disk source")
    return candidate


def _query_dependencies(query: Mapping[str, object]) -> tuple[list[str], list[str]]:
    if "source_artifact_bindings" in query:
        source = sorted(
            {
                _hash(row["source_artifact_sha256"], "source artifact")
                for row in query["source_artifact_bindings"]
            }
        )
        raw = sorted(
            {
                _hash(item, "raw dependency")
                for item in query.get("raw_dependency_sha256s", [])
            }
        )
        return source, raw
    lineage = query["lineage"]
    source = sorted(
        {
            *(_hash(item, "source receipt") for item in lineage["source_receipt_sha256s"]),
            *(_hash(item, "source binding") for item in lineage["source_binding_sha256s"]),
        }
    )
    raw = sorted(
        _hash(item, "raw dependency") for item in lineage["raw_dependency_sha256s"]
    )
    return source, raw


def _compose_required_queries(source_artifacts: Mapping[str, object]) -> tuple[
    dict[str, Any], list[dict[str, Any]]
]:
    legacy_source_keys = {
        "waveform_morphology_receipt",
        "periodicity_candidates",
        "frequency_candidate",
        "morphology_evolution_receipt",
        "morphology_change_points",
        "multireference_field_result",
        "baseline_context_receipt",
    }
    optional_source_keys = {
        "physical_amplitude_findings_receipt",
        "component_cycle_element_ledger_receipt",
    }
    observed_source_keys = set(source_artifacts)
    if (
        not legacy_source_keys.issubset(observed_source_keys)
        or observed_source_keys - legacy_source_keys - optional_source_keys
    ):
        raise ValueError("required-query source artifact roster drifted")
    waveform_morphology = validate_event_morphology_primitive_supervision_v1(
        source_artifacts["waveform_morphology_receipt"]
    )
    periodicity = [
        validate_deterministic_periodicity_candidate(row)
        for row in source_artifacts["periodicity_candidates"]
    ]
    frequency = _frequency_candidate_from_dict(source_artifacts["frequency_candidate"])
    morphology_evolution = validate_event_morphology_primitive_supervision_v1(
        source_artifacts["morphology_evolution_receipt"]
    )
    change_points = [
        _change_point_from_dict(row)
        for row in source_artifacts["morphology_change_points"]
    ]
    field_result = validate_ba_ieg_multireference_field_result(
        source_artifacts["multireference_field_result"]
    )
    baseline = validate_event_baseline_context_comparability_receipt(
        source_artifacts["baseline_context_receipt"]
    )

    waveform_bundle = materialize_event_waveform_rhythm_query_bridge_v1(
        morphology_primitive_receipt=waveform_morphology,
        periodicity_candidates=periodicity,
    )
    waveform_bundle = validate_event_waveform_rhythm_query_bridge_v1(waveform_bundle)
    ledgers = [
        compose_frequency_evolution_query_ledger_v1(frequency),
        compose_morphology_evolution_query_ledger_v1(
            morphology_evolution,
            change_points=change_points,
        ),
        compose_location_evolution_query_ledger_v1(field_result),
        compose_return_to_comparable_background_query_ledger_v1(baseline),
    ]
    ledgers = [
        validate_event_evolution_recovery_query_ledger_v1(row) for row in ledgers
    ]
    query_rows = [deepcopy(row) for row in waveform_bundle["query_results"]] + ledgers
    if "physical_amplitude_findings_receipt" in source_artifacts:
        amplitude_adapter = materialize_event_physical_amplitude_query_adapter_v1(
            source_artifacts["physical_amplitude_findings_receipt"]
        )
        replacement = {
            str(row["term_query_id"]): deepcopy(row)
            for row in amplitude_adapter["query_results"]
        }
        if tuple(sorted(replacement)) != tuple(sorted(S04_REQUIRED_QUERY_IDS)):
            raise ValueError("native S04 adapter query roster drifted")
        query_rows = [
            replacement.get(str(row.get("term_query_id", row.get("query_id"))), row)
            for row in query_rows
        ]
    if "component_cycle_element_ledger_receipt" in source_artifacts:
        s06_adapter = materialize_event_component_cycle_element_query_adapter_v1(
            source_artifacts["component_cycle_element_ledger_receipt"]
        )
        replacement = {
            str(row["term_query_id"]): deepcopy(row)
            for row in s06_adapter["query_results"]
        }
        if tuple(sorted(replacement)) != tuple(sorted(S06_REQUIRED_QUERY_IDS)):
            raise ValueError("native S06 adapter query roster drifted")
        query_rows = [
            replacement.get(str(row.get("term_query_id", row.get("query_id"))), row)
            for row in query_rows
        ]
    query_ids = [
        str(row.get("term_query_id", row.get("query_id"))) for row in query_rows
    ]
    if sorted(query_ids) != list(REQUIRED_QUERY_IDS) or len(set(query_ids)) != 10:
        raise ValueError("required-query composer did not close the exact ten-query roster")
    event_ids = {
        str(waveform_bundle["event_id"]),
        *(str(row["event_id"]) for row in ledgers),
    }
    if len(event_ids) != 1:
        raise ValueError(f"required-query sources cross event identities: {sorted(event_ids)}")

    event_id = str(waveform_bundle["event_id"])
    recording_id = str(waveform_bundle["recording_id"])
    signal_sha256 = str(waveform_bundle["source_signal_sha256"])
    canonical_receipt_sha256 = str(waveform_morphology["canonical_receipt_sha256"])
    identity_checks = {
        "morphology_evolution_receipt.event_id": (
            str(morphology_evolution["event_id"]),
            event_id,
        ),
        "morphology_evolution_receipt.recording_id": (
            str(morphology_evolution["recording_id"]),
            recording_id,
        ),
        "morphology_evolution_receipt.source_signal_sha256": (
            str(morphology_evolution["source_signal_sha256"]),
            signal_sha256,
        ),
        "morphology_evolution_receipt.canonical_receipt_sha256": (
            str(morphology_evolution["canonical_receipt_sha256"]),
            canonical_receipt_sha256,
        ),
        "frequency_candidate.event_id": (str(frequency.event_id), event_id),
        "multireference_field_result.event_id": (
            str(field_result["event_id"]),
            event_id,
        ),
        "multireference_field_result.recording_id": (
            str(field_result["recording_id"]),
            recording_id,
        ),
        "multireference_field_result.canonical_receipt_sha256": (
            str(field_result["canonical_receipt_sha256"]),
            canonical_receipt_sha256,
        ),
        "baseline_context_receipt.event_id": (
            str(baseline["event_binding"]["event_id"]),
            event_id,
        ),
        "baseline_context_receipt.recording_id": (
            str(baseline["event_binding"]["recording_id"]),
            recording_id,
        ),
        "baseline_context_receipt.canonical_signal_sha256": (
            str(baseline["event_binding"]["canonical_signal_sha256"]),
            signal_sha256,
        ),
    }
    if "physical_amplitude_findings_receipt" in source_artifacts:
        amplitude = amplitude_adapter
        identity_checks.update(
            {
                "physical_amplitude_findings_receipt.event_id": (
                    str(amplitude["event_id"]),
                    event_id,
                ),
                "physical_amplitude_findings_receipt.recording_id": (
                    str(amplitude["recording_id"]),
                    recording_id,
                ),
                "physical_amplitude_findings_receipt.source_signal_sha256": (
                    str(amplitude["source_signal_sha256"]),
                    signal_sha256,
                ),
                "physical_amplitude_findings_receipt.canonical_receipt_sha256": (
                    str(amplitude["canonical_receipt_sha256"]),
                    canonical_receipt_sha256,
                ),
                "physical_amplitude_findings_receipt.canonical_signal_id": (
                    str(amplitude["canonical_signal_id"]),
                    str(waveform_bundle["canonical_signal_id"]),
                ),
            }
        )
    if "component_cycle_element_ledger_receipt" in source_artifacts:
        s06 = s06_adapter
        identity_checks.update(
            {
                "component_cycle_element_ledger_receipt.event_id": (
                    str(s06["event_id"]),
                    event_id,
                ),
                "component_cycle_element_ledger_receipt.recording_id": (
                    str(s06["recording_id"]),
                    recording_id,
                ),
                "component_cycle_element_ledger_receipt.source_signal_sha256": (
                    str(s06["source_signal_sha256"]),
                    signal_sha256,
                ),
                "component_cycle_element_ledger_receipt.canonical_receipt_sha256": (
                    str(s06["canonical_receipt_sha256"]),
                    canonical_receipt_sha256,
                ),
                "component_cycle_element_ledger_receipt.canonical_signal_id": (
                    str(s06["canonical_signal_id"]),
                    str(waveform_bundle["canonical_signal_id"]),
                ),
            }
        )
    for context, (observed, expected) in identity_checks.items():
        if observed != expected:
            raise ValueError(
                f"{context} crosses the required-query source identity: "
                f"{observed!r} != {expected!r}"
            )
    for index, row in enumerate(periodicity):
        if str(row["event_id"]) != event_id:
            raise ValueError(
                f"periodicity_candidates[{index}].event_id crosses event identity"
            )
        if str(row["source_binding"]["canonical_signal_sha256"]) != signal_sha256:
            raise ValueError(
                "periodicity candidate crosses the canonical signal identity"
            )
    return waveform_bundle, query_rows


def _build_required_query_artifacts(
    *,
    waveform_bundle: Mapping[str, Any],
    query_rows: Sequence[Mapping[str, Any]],
    source_closure_sha256: str,
) -> list[dict[str, Any]]:
    by_query = {
        str(row.get("term_query_id", row.get("query_id"))): row
        for row in query_rows
    }
    artifacts: list[dict[str, Any]] = []
    for query_id in REQUIRED_QUERY_IDS:
        query = deepcopy(dict(by_query[query_id]))
        query_sha256 = str(
            query.get("query_result_sha256", query.get("receipt_sha256"))
        )
        _hash(query_sha256, f"{query_id} query payload")
        source_hashes, raw_hashes = _query_dependencies(query)
        artifact: dict[str, Any] = {
            "schema_version": "clinical_eeg_findings_v1_required_query_artifact_v1",
            "artifact_id": "FQART-"
            + _domain_sha256(
                "clinical-eeg-findings-v1-required-query-artifact-id-v1",
                {
                    "query_id": query_id,
                    "query_payload_sha256": query_sha256,
                    "source_closure_sha256": source_closure_sha256,
                },
            )[:24],
            "event_id": str(waveform_bundle["event_id"]),
            "recording_id": str(waveform_bundle["recording_id"]),
            "canonical_signal_sha256": str(waveform_bundle["source_signal_sha256"]),
            "term_query_id": query_id,
            "event_card_slot_id": REQUIRED_QUERY_SLOT[query_id],
            "query_payload_kind": (
                "native_s04_physical_amplitude_query_result"
                if query.get("projection_method_id")
                == EVENT_PHYSICAL_AMPLITUDE_QUERY_ADAPTER_METHOD_ID
                else "native_s06_component_cycle_element_query_result"
                if query.get("projection_method_id")
                == EVENT_COMPONENT_CYCLE_ELEMENT_QUERY_ADAPTER_METHOD_ID
                else "waveform_rhythm_query_result"
                if "term_query_id" in query
                else "evolution_recovery_query_ledger"
            ),
            "query_payload": query,
            "query_payload_sha256": query_sha256,
            "source_closure_sha256": source_closure_sha256,
            "source_or_binding_sha256s": source_hashes,
            "raw_dependency_sha256s": raw_hashes,
            "calibration": {
                "status": "not_evaluable_uncalibrated",
                "probability": None,
                "calibration_receipt_sha256": None,
            },
            "authorization": deepcopy(_QUERY_AUTHORIZATION),
            "artifact_sha256": "",
        }
        artifact["artifact_sha256"] = _self_hash(artifact, "artifact_sha256")
        artifacts.append(artifact)
    return artifacts


def materialize_findings_v1_required_query_bundle_v1(
    *,
    waveform_morphology_receipt: object,
    periodicity_candidates: Sequence[object],
    frequency_candidate: ACNSFrequencyEvolutionCandidate,
    morphology_evolution_receipt: object,
    morphology_change_points: Sequence[EventChangePointProposal],
    multireference_field_result: object,
    baseline_context_receipt: object,
    physical_amplitude_findings_receipt: object | None = None,
    component_cycle_element_ledger_receipt: object | None = None,
) -> dict[str, Any]:
    """Compose and content-bind the exact ten Findings-v1 query artifacts."""

    source_artifacts: dict[str, object] = {
        "waveform_morphology_receipt": deepcopy(waveform_morphology_receipt),
        "periodicity_candidates": [deepcopy(row) for row in periodicity_candidates],
        "frequency_candidate": _frequency_candidate_to_dict(frequency_candidate),
        "morphology_evolution_receipt": deepcopy(morphology_evolution_receipt),
        "morphology_change_points": [
            _change_point_to_dict(row) for row in morphology_change_points
        ],
        "multireference_field_result": deepcopy(multireference_field_result),
        "baseline_context_receipt": deepcopy(baseline_context_receipt),
    }
    if physical_amplitude_findings_receipt is not None:
        source_artifacts["physical_amplitude_findings_receipt"] = deepcopy(
            physical_amplitude_findings_receipt
        )
    if component_cycle_element_ledger_receipt is not None:
        source_artifacts["component_cycle_element_ledger_receipt"] = deepcopy(
            component_cycle_element_ledger_receipt
        )
    waveform_bundle, query_rows = _compose_required_queries(source_artifacts)
    source_closure_sha256 = _domain_sha256(
        "clinical-eeg-findings-v1-required-query-source-closure-v1",
        source_artifacts,
    )
    artifacts = _build_required_query_artifacts(
        waveform_bundle=waveform_bundle,
        query_rows=query_rows,
        source_closure_sha256=source_closure_sha256,
    )

    bundle: dict[str, Any] = {
        "schema_version": FINDINGS_V1_REQUIRED_QUERY_BUNDLE_SCHEMA_VERSION,
        "bundle_id": "FQBU-"
        + _domain_sha256(
            "clinical-eeg-findings-v1-required-query-bundle-id-v1",
            {
                "source_closure_sha256": source_closure_sha256,
                "artifact_sha256s": [row["artifact_sha256"] for row in artifacts],
            },
        )[:24],
        "event_id": str(waveform_bundle["event_id"]),
        "recording_id": str(waveform_bundle["recording_id"]),
        "canonical_signal_sha256": str(waveform_bundle["source_signal_sha256"]),
        "source_artifacts": source_artifacts,
        "source_closure_sha256": source_closure_sha256,
        "waveform_bridge_receipt_sha256": str(waveform_bundle["receipt_sha256"]),
        "query_artifacts": artifacts,
        "closure": {
            "expected_query_count": 10,
            "observed_query_count": 10,
            "query_ids": list(REQUIRED_QUERY_IDS),
            "every_query_exactly_once": True,
            "all_sources_replayed": True,
            "all_artifacts_content_bound": True,
            "uncalibrated_probabilities_null": True,
            "report_allowlist_empty": True,
        },
        "firewall": deepcopy(_FIREWALL),
        "authorization": deepcopy(_QUERY_AUTHORIZATION),
        "bundle_sha256": "",
    }
    bundle["bundle_sha256"] = _self_hash(bundle, "bundle_sha256")
    return validate_findings_v1_required_query_bundle_v1(bundle)


def validate_findings_v1_required_query_bundle_v1(value: object) -> dict[str, Any]:
    """Replay all ten query artifacts from the embedded typed source objects."""

    if type(value) is not dict:
        raise TypeError("required-query bundle must be an object")
    candidate = deepcopy(value)
    _reject_nonfinite(candidate)
    required = {
        "schema_version",
        "bundle_id",
        "event_id",
        "recording_id",
        "canonical_signal_sha256",
        "source_artifacts",
        "source_closure_sha256",
        "waveform_bridge_receipt_sha256",
        "query_artifacts",
        "closure",
        "firewall",
        "authorization",
        "bundle_sha256",
    }
    if set(candidate) != required:
        raise ValueError("required-query bundle fields drifted")
    if candidate["schema_version"] != FINDINGS_V1_REQUIRED_QUERY_BUNDLE_SCHEMA_VERSION:
        raise ValueError("required-query bundle schema drifted")
    for key in ("event_id", "recording_id"):
        _identifier(candidate[key], key)
    for key in (
        "canonical_signal_sha256",
        "source_closure_sha256",
        "waveform_bridge_receipt_sha256",
        "bundle_sha256",
    ):
        _hash(candidate[key], key)
    if candidate["firewall"] != _FIREWALL or candidate["authorization"] != _QUERY_AUTHORIZATION:
        raise ValueError("required-query firewall/authorization drifted")
    waveform_bundle, query_rows = _compose_required_queries(candidate["source_artifacts"])
    source_closure = _domain_sha256(
        "clinical-eeg-findings-v1-required-query-source-closure-v1",
        candidate["source_artifacts"],
    )
    if source_closure != candidate["source_closure_sha256"]:
        raise ValueError("required-query source closure hash drifted")
    artifacts = candidate["query_artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 10:
        raise ValueError("required-query bundle must contain ten artifacts")
    observed_ids = [str(row.get("term_query_id")) for row in artifacts]
    if observed_ids != list(REQUIRED_QUERY_IDS):
        raise ValueError("required-query artifact order/roster drifted")
    expected_artifacts = _build_required_query_artifacts(
        waveform_bundle=waveform_bundle,
        query_rows=query_rows,
        source_closure_sha256=source_closure,
    )
    if artifacts != expected_artifacts:
        raise ValueError("required-query artifacts do not replay exactly from sources")
    expected_closure = {
        "expected_query_count": 10,
        "observed_query_count": 10,
        "query_ids": list(REQUIRED_QUERY_IDS),
        "every_query_exactly_once": True,
        "all_sources_replayed": True,
        "all_artifacts_content_bound": True,
        "uncalibrated_probabilities_null": True,
        "report_allowlist_empty": True,
    }
    if candidate["closure"] != expected_closure:
        raise ValueError("required-query closure receipt drifted")
    expected_bundle_id = "FQBU-" + _domain_sha256(
        "clinical-eeg-findings-v1-required-query-bundle-id-v1",
        {
            "source_closure_sha256": source_closure,
            "artifact_sha256s": [row["artifact_sha256"] for row in artifacts],
        },
    )[:24]
    if candidate["bundle_id"] != expected_bundle_id:
        raise ValueError("required-query bundle ID drifted")
    if candidate["waveform_bridge_receipt_sha256"] != waveform_bundle["receipt_sha256"]:
        raise ValueError("required-query waveform bridge binding drifted")
    if candidate["event_id"] != waveform_bundle["event_id"]:
        raise ValueError("required-query event identity drifted")
    if candidate["recording_id"] != waveform_bundle["recording_id"]:
        raise ValueError("required-query recording identity drifted")
    if candidate["canonical_signal_sha256"] != waveform_bundle["source_signal_sha256"]:
        raise ValueError("required-query signal identity drifted")
    if candidate["bundle_sha256"] != _self_hash(candidate, "bundle_sha256"):
        raise ValueError("required-query bundle self hash drifted")
    return candidate


def _finding_raw_dependencies(
    event_source: Mapping[str, Any], finding: Mapping[str, Any]
) -> list[dict[str, Any]]:
    waveform_by_id = {
        str(row["waveform_evidence_id"]): row
        for row in event_source["waveform_evidence"]
    }
    dependencies: dict[str, dict[str, Any]] = {}
    for measurement in finding["measurements"]:
        raw = measurement["source_binding"].get("raw_sample_dependency")
        if isinstance(raw, Mapping):
            dependency_id = str(raw["dependency_id"])
            receipt = deepcopy(dict(raw))
            previous = dependencies.get(dependency_id)
            if previous is not None and previous != receipt:
                raise ValueError("raw dependency ID resolves to conflicting receipts")
            dependencies[dependency_id] = receipt
    for waveform_id in finding["waveform_evidence_ids"]:
        raw = waveform_by_id[str(waveform_id)].get("raw_sample_dependency")
        if isinstance(raw, Mapping):
            dependency_id = str(raw["dependency_id"])
            receipt = deepcopy(dict(raw))
            previous = dependencies.get(dependency_id)
            if previous is not None and previous != receipt:
                raise ValueError("raw dependency ID resolves to conflicting receipts")
            dependencies[dependency_id] = receipt
    return [dependencies[key] for key in sorted(dependencies)]


def _onset_permission(
    event_source: Mapping[str, Any], finding: Mapping[str, Any]
) -> tuple[bool, list[str], list[dict[str, Any]]]:
    reasons: list[str] = []
    if finding["status"] != "present":
        reasons.append("finding_not_present")
    if finding["intrinsic_evidence_role"] != "onset_eligible":
        reasons.append("intrinsic_role_not_onset_eligible")
    if finding["signal_temporal_context"] != "candidate_emergence":
        reasons.append("signal_context_not_candidate_emergence")
    dependencies = _finding_raw_dependencies(event_source, finding)
    if not dependencies:
        reasons.append("no_replayable_raw_dependency")
    if list(finding["raw_sample_dependency_ids"]) != [
        str(row["dependency_id"]) for row in dependencies
    ]:
        reasons.append("raw_dependency_receipt_roster_drifted")
    for dependency in dependencies:
        if dependency["view_role"] != "onset_causal":
            reasons.append("dependency_not_onset_causal")
        if dependency["dependency_policy"] != "past_and_present_only":
            reasons.append("dependency_not_past_and_present_only")
        if dependency["future_sample_access"] is not False:
            reasons.append("future_sample_access_present")
        if dependency["onset_evidence_authorized"] is not True:
            reasons.append("dependency_onset_not_authorized")
        if dependency["onset_support_eligible"] is not True:
            reasons.append("dependency_onset_support_ineligible")
    return not reasons, sorted(set(reasons)), dependencies


def _causal_score_policy_receipt() -> dict[str, Any]:
    receipt = deepcopy(dict(_CAUSAL_SCORE_POLICY))
    receipt["policy_sha256"] = ""
    receipt["policy_sha256"] = _self_hash(receipt, "policy_sha256")
    return receipt


def _causal_spatial_score_leaf(
    *,
    finding: Mapping[str, Any],
    source_finding_sha256: str,
    candidate: Mapping[str, Any],
    source_relations: Sequence[Mapping[str, Any]],
    raw_dependency_receipts: Sequence[Mapping[str, Any]],
    score_policy_sha256: str,
) -> dict[str, Any] | None:
    """Bind one candidate-matched score input to causal raw receipts only."""

    axis = str(candidate["axis"])
    candidate_type = str(candidate["candidate_type"])
    candidate_id = str(candidate["candidate_id"])
    if axis not in _CAUSAL_SCORE_POLICY["candidate_axes"]:
        return None
    matching_support = [
        row
        for row in finding["spatial_support"]
        if row["unit_type"] == candidate_type and row["id"] == candidate_id
    ]
    if len(matching_support) != 1:
        return None
    source_support = matching_support[0]
    if source_support["evidence_eligible"] is not True:
        return None
    support = {
        "unit_type": str(source_support["unit_type"]),
        "id": str(source_support["id"]),
        "mapping_status": str(source_support["mapping_status"]),
        "observation_status": str(source_support["observation_status"]),
        "evidence_eligible": True,
        "missing_reason_codes": deepcopy(source_support["missing_reason_codes"]),
        "field_observation": deepcopy(source_support["field_observation"]),
        "source_support_score_consumed": False,
    }
    leaf_value = 1.0
    raw_receipts = [deepcopy(dict(row)) for row in raw_dependency_receipts]
    if not raw_receipts:
        return None
    raw_receipt_refs = [
        {
            "dependency_id": str(row["dependency_id"]),
            "dependency_sha256": str(row["dependency_sha256"]),
        }
        for row in raw_receipts
    ]
    relation_refs = [
        {
            "relation_id": str(row["relation_id"]),
            "relation_sha256": _domain_sha256(
                "clinical-eeg-causal-score-leaf-source-relation-v1", row
            ),
        }
        for row in sorted(source_relations, key=lambda item: str(item["relation_id"]))
    ]
    support_sha256 = _domain_sha256(
        "clinical-eeg-causal-score-leaf-spatial-support-v1", support
    )
    score_input_binding_sha256 = _domain_sha256(
        "clinical-eeg-causal-score-leaf-input-binding-v1",
        {
            "score_policy_sha256": score_policy_sha256,
            "source_finding_sha256": source_finding_sha256,
            "source_relation_receipts": relation_refs,
            "source_spatial_support_sha256": support_sha256,
            "raw_dependency_receipts": raw_receipt_refs,
        },
    )
    leaf: dict[str, Any] = {
        "schema_version": "clinical_eeg_future_free_causal_score_leaf_v1",
        "leaf_id": "CSLEAF-"
        + _domain_sha256(
            "clinical-eeg-future-free-causal-score-leaf-id-v1",
            {
                "axis": axis,
                "candidate_type": candidate_type,
                "candidate_id": candidate_id,
                "evidence_id": finding["evidence_id"],
                "score_input_binding_sha256": score_input_binding_sha256,
            },
        )[:24],
        "candidate": {
            "axis": axis,
            "candidate_type": candidate_type,
            "candidate_id": candidate_id,
        },
        "source_evidence_id": str(finding["evidence_id"]),
        "source_finding_sha256": source_finding_sha256,
        "source_relation_receipts": relation_refs,
        "source_spatial_support": support,
        "source_spatial_support_sha256": support_sha256,
        "raw_dependency_receipts": raw_receipts,
        "raw_dependency_receipt_refs": raw_receipt_refs,
        "score_input_binding_sha256": score_input_binding_sha256,
        "leaf_value": leaf_value,
        "leaf_value_semantics": (
            "one_unit_mass_for_unique_candidate_matched_future_free_causal_"
            "evidence_leaf"
        ),
        "permission": {
            "all_raw_dependencies_future_free_causal": True,
            "matching_spatial_support_evidence_eligible": True,
            "source_spatial_support_score_consumed": False,
            "source_candidate_score_consumed": False,
            "source_candidate_rank_consumed": False,
            "course_later_or_context_consumed": False,
        },
        "leaf_receipt_sha256": "",
    }
    leaf["leaf_receipt_sha256"] = _self_hash(leaf, "leaf_receipt_sha256")
    return leaf


def materialize_findings_to_research_soz_evidence_graph_v1(
    *,
    event_findings_v3: object,
    event_card_projection_v2: object,
    required_query_bundle_v1: object,
    registry_closure_receipt_v1: object,
    record_context_card_id: str,
    trusted_registry_closure_receipt_sha256: str | None = None,
    **trusted_validation_context: Any,
) -> dict[str, Any]:
    """Project future-free onset evidence after replaying the complete Event Card."""

    event_source = validate_event_eeg_findings_v3_payload(
        event_findings_v3, **trusted_validation_context
    )
    query_bundle = validate_findings_v1_required_query_bundle_v1(
        required_query_bundle_v1
    )
    card = validate_event_card_projection_v2(
        event_card_projection_v2,
        source_event_findings_v3=event_source,
        source_registry_closure_receipt_v1=registry_closure_receipt_v1,
        record_context_card_id=record_context_card_id,
        trusted_registry_closure_receipt_sha256=(
            trusted_registry_closure_receipt_sha256
        ),
        **trusted_validation_context,
    )
    if query_bundle["event_id"] != event_source["event_id"]:
        raise ValueError("SOZ graph required-query/event identity drifted")
    if query_bundle["canonical_signal_sha256"] != event_source["provenance"][
        "canonical_signal_sha256"
    ]:
        raise ValueError("SOZ graph required-query signal identity drifted")
    projected_by_id = {
        str(row["evidence_id"]): row for row in card["event_findings"]
    }
    source_by_id = {
        str(row["evidence_id"]): row for row in event_source["findings"]
    }
    if set(projected_by_id) - set(source_by_id):
        raise ValueError("SOZ graph Event Card cites unknown source Findings")
    if len(projected_by_id) != len(card["event_findings"]):
        raise ValueError("SOZ graph Event Card duplicates an evidence ID")
    for evidence_id, projected in projected_by_id.items():
        source_finding = source_by_id[evidence_id]
        if projected.get("finding") != source_finding:
            raise ValueError("SOZ graph Event Card Finding payload drifted")
        if projected.get("source_finding_sha256") != _domain_sha256(
            "clinical-eeg-event-card-v2-source-finding", source_finding
        ):
            raise ValueError("SOZ graph Event Card Finding hash drifted")

    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    permission_by_id: dict[
        str, tuple[bool, list[dict[str, Any]]]
    ] = {}
    for evidence_id in sorted(projected_by_id):
        finding = source_by_id[evidence_id]
        allowed, reasons, raw_receipts = _onset_permission(event_source, finding)
        permission_by_id[evidence_id] = (allowed, raw_receipts)
        row = {
            "evidence_id": evidence_id,
            "source_finding_sha256": str(
                projected_by_id[evidence_id]["source_finding_sha256"]
            ),
            "assertion_level": str(finding["assertion_level"]),
            "intrinsic_evidence_role": str(finding["intrinsic_evidence_role"]),
            "signal_temporal_context": str(finding["signal_temporal_context"]),
            "raw_dependency_sha256s": [
                str(receipt["dependency_sha256"])
                for receipt in raw_receipts
            ],
            "raw_dependency_receipt_refs": [
                {
                    "dependency_id": str(receipt["dependency_id"]),
                    "dependency_sha256": str(receipt["dependency_sha256"]),
                }
                for receipt in raw_receipts
            ],
            "research_scalp_onset_support_authorized": bool(allowed),
            "clinical_soz_support_authorized": False,
            "reason_codes": reasons,
        }
        (eligible if allowed else excluded).append(row)

    eligible_ids = {row["evidence_id"] for row in eligible}
    relation_by_id = {
        str(row["relation_id"]): row
        for row in event_source["hypothesis_evidence_relations"]
    }
    score_policy = _causal_score_policy_receipt()
    score_leaves: list[dict[str, Any]] = []
    score_receipts: list[dict[str, Any]] = []
    ranking_candidates: list[dict[str, Any]] = []
    for candidate in event_source["scalp_onset_hypothesis"]["candidate_scores"]:
        if candidate["score_semantics"] != "uncalibrated_ranking_score":
            raise ValueError("SOZ graph candidate score semantics drifted")
        if candidate["calibration_receipt_id"] is not None:
            raise ValueError("v1 disk closure does not accept calibrated score claims")
        relations_by_evidence: dict[str, list[Mapping[str, Any]]] = {}
        for relation_id in candidate["supporting_relation_ids"]:
            relation = relation_by_id.get(str(relation_id))
            if relation is None:
                raise ValueError("SOZ graph candidate cites an unknown relation")
            if relation["relation"] != "supports":
                raise ValueError("SOZ graph supporting relation is not supportive")
            if (
                relation["axis"] != candidate["axis"]
                or relation["candidate_type"] != candidate["candidate_type"]
                or relation["candidate_id"] != candidate["candidate_id"]
            ):
                raise ValueError("SOZ graph relation/candidate target drifted")
            for evidence_id in relation["evidence_ids"]:
                evidence_key = str(evidence_id)
                if evidence_key in eligible_ids:
                    relations_by_evidence.setdefault(evidence_key, []).append(
                        relation
                    )
        candidate_leaves: list[dict[str, Any]] = []
        for evidence_id in sorted(relations_by_evidence):
            allowed, raw_receipts = permission_by_id[evidence_id]
            if not allowed:
                continue
            leaf = _causal_spatial_score_leaf(
                finding=source_by_id[evidence_id],
                source_finding_sha256=str(
                    projected_by_id[evidence_id]["source_finding_sha256"]
                ),
                candidate=candidate,
                source_relations=relations_by_evidence[evidence_id],
                raw_dependency_receipts=raw_receipts,
                score_policy_sha256=str(score_policy["policy_sha256"]),
            )
            if leaf is not None:
                candidate_leaves.append(leaf)
        if not candidate_leaves:
            continue
        try:
            ranking_score = math.fsum(
                float(row["leaf_value"]) for row in candidate_leaves
            )
        except OverflowError as error:
            raise ValueError("causal score leaf aggregation overflowed") from error
        if not math.isfinite(ranking_score):
            raise ValueError("causal score leaf aggregation is non-finite")
        leaf_receipt_sha256s = [
            str(row["leaf_receipt_sha256"]) for row in candidate_leaves
        ]
        score_receipt: dict[str, Any] = {
            "schema_version": (
                "clinical_eeg_future_free_causal_leaf_count_score_receipt_v1"
            ),
            "candidate": {
                "axis": str(candidate["axis"]),
                "candidate_type": str(candidate["candidate_type"]),
                "candidate_id": str(candidate["candidate_id"]),
            },
            "score_policy_sha256": str(score_policy["policy_sha256"]),
            "unique_score_leaf_receipt_sha256s": leaf_receipt_sha256s,
            "aggregation_formula": (
                "math.fsum(unique_per_evidence_leaf_unit_masses)"
            ),
            "ranking_score": ranking_score,
            "score_semantics": (
                "uncalibrated_future_free_causal_evidence_leaf_count"
            ),
            "source_numeric_scores_consumed": False,
            "course_later_or_context_consumed": False,
            "receipt_sha256": "",
        }
        score_receipt["receipt_sha256"] = _self_hash(
            score_receipt, "receipt_sha256"
        )
        score_leaves.extend(candidate_leaves)
        score_receipts.append(score_receipt)
        ranking_candidates.append(
            {
                "rank": 0,
                "axis": str(candidate["axis"]),
                "candidate_type": str(candidate["candidate_type"]),
                "candidate_id": str(candidate["candidate_id"]),
                "ranking_score": ranking_score,
                "score_semantics": (
                    "uncalibrated_future_free_causal_evidence_leaf_count"
                ),
                "ranking_score_policy_sha256": str(
                    score_policy["policy_sha256"]
                ),
                "ranking_score_receipt_sha256": str(
                    score_receipt["receipt_sha256"]
                ),
                "score_leaf_receipt_sha256s": leaf_receipt_sha256s,
                "probability": None,
                "calibration_status": "not_evaluable_uncalibrated",
                "calibration_receipt_sha256": None,
                "supporting_relation_ids": sorted(
                    {
                        str(receipt["relation_id"])
                        for leaf in candidate_leaves
                        for receipt in leaf["source_relation_receipts"]
                    }
                ),
                "supporting_onset_evidence_ids": [
                    str(row["source_evidence_id"]) for row in candidate_leaves
                ],
                "source_candidate_score_consumed": False,
                "source_candidate_rank_consumed": False,
                "clinical_soz_claim_authorized": False,
            }
        )
    rankings: list[dict[str, Any]] = []
    for axis in sorted({str(row["axis"]) for row in ranking_candidates}):
        axis_rows = sorted(
            (row for row in ranking_candidates if row["axis"] == axis),
            key=lambda row: (-float(row["ranking_score"]), row["candidate_id"]),
        )
        for rank, row in enumerate(axis_rows, start=1):
            row["rank"] = rank
            rankings.append(row)
    rankings.sort(key=lambda row: (row["rank"], row["axis"], row["candidate_id"]))
    score_leaves.sort(
        key=lambda row: (
            row["candidate"]["axis"],
            row["candidate"]["candidate_id"],
            row["source_evidence_id"],
        )
    )
    score_receipts.sort(
        key=lambda row: (
            row["candidate"]["axis"],
            row["candidate"]["candidate_id"],
        )
    )

    course_refs = [
        {
            "term_query_id": str(row["term_query_id"]),
            "artifact_id": str(row["artifact_id"]),
            "artifact_sha256": str(row["artifact_sha256"]),
            "event_card_slot_id": str(row["event_card_slot_id"]),
            "positive_onset_support_authorized": False,
            "positive_soz_support_authorized": False,
        }
        for row in query_bundle["query_artifacts"]
    ]
    status = "ranked_research_candidates" if rankings else "not_evaluable"
    reasons = [] if rankings else ["no_replayable_future_free_causal_score_leaf"]
    graph: dict[str, Any] = {
        "schema_version": FINDINGS_TO_SOZ_EVIDENCE_GRAPH_SCHEMA_VERSION,
        "graph_id": "F2SOZ-"
        + _domain_sha256(
            "clinical-eeg-findings-to-research-soz-graph-id-v1",
            {
                "event_id": event_source["event_id"],
                "event_card_projection_sha256": card["projection_sha256"],
                "required_query_bundle_sha256": query_bundle["bundle_sha256"],
            },
        )[:24],
        "owner": {
            "event_id": str(event_source["event_id"]),
            "recording_id": str(event_source["provenance"]["record_id"]),
            "canonical_signal_sha256": str(
                event_source["provenance"]["canonical_signal_sha256"]
            ),
        },
        "claim_boundary": (
            "research_scalp_visible_ictal_onset_topography_not_cortical_soz_ez"
        ),
        "source_bindings": {
            "event_findings_v3_sha256": _domain_sha256(
                "clinical-eeg-findings-to-soz-source-v3", event_source
            ),
            "event_card_projection_sha256": str(card["projection_sha256"]),
            "required_query_bundle_sha256": str(query_bundle["bundle_sha256"]),
            "record_context_card_id": str(
                card["record_context_reference"]["card_id"]
            ),
            "record_context_payload_consumed": False,
        },
        "event_qualification": {
            "source_status": str(event_source["event_qualification"]["status"]),
            "research_ranking_status": status,
            "clinical_term_qualified": False,
            "report_promotion_authorized": False,
            "reason_codes": reasons,
        },
        "eligible_future_free_onset_evidence": eligible,
        "excluded_event_evidence": excluded,
        "causal_leaf_count_score_policy_receipt": score_policy,
        "future_free_causal_score_leaf_receipts": score_leaves,
        "candidate_leaf_count_score_receipts": score_receipts,
        "candidate_ranking": rankings,
        "course_query_artifact_catalog": course_refs,
        "uncertainty": {
            "status": "not_evaluable_uncalibrated",
            "probability": None,
            "calibration_receipt_sha256": None,
            "source_component_scores_copied_as_probabilities": False,
            "source_candidate_or_spatial_support_scores_consumed": False,
            "leaf_count_is_model_probability_or_final_soz_score": False,
        },
        "firewall": deepcopy(_FIREWALL),
        "authorization": deepcopy(_GRAPH_AUTHORIZATION),
        "graph_sha256": "",
    }
    graph["graph_sha256"] = _self_hash(graph, "graph_sha256")
    return graph


def validate_findings_to_research_soz_evidence_graph_v1(
    value: object,
    *,
    event_findings_v3: object,
    event_card_projection_v2: object,
    required_query_bundle_v1: object,
    registry_closure_receipt_v1: object,
    record_context_card_id: str,
    trusted_registry_closure_receipt_sha256: str | None = None,
    **trusted_validation_context: Any,
) -> dict[str, Any]:
    """Rebuild the complete graph and reject source/self-rehashed drift."""

    if type(value) is not dict:
        raise TypeError("Findings-to-SOZ graph must be an object")
    candidate = deepcopy(value)
    _reject_nonfinite(candidate)
    if candidate.get("schema_version") != FINDINGS_TO_SOZ_EVIDENCE_GRAPH_SCHEMA_VERSION:
        raise ValueError("Findings-to-SOZ graph schema drifted")
    if (
        candidate.get("firewall") != _FIREWALL
        or candidate.get("authorization") != _GRAPH_AUTHORIZATION
    ):
        raise ValueError("Findings-to-SOZ graph firewall/authorization drifted")
    if candidate.get("graph_sha256") != _self_hash(candidate, "graph_sha256"):
        raise ValueError("Findings-to-SOZ graph self hash drifted")
    expected = materialize_findings_to_research_soz_evidence_graph_v1(
        event_findings_v3=event_findings_v3,
        event_card_projection_v2=event_card_projection_v2,
        required_query_bundle_v1=required_query_bundle_v1,
        registry_closure_receipt_v1=registry_closure_receipt_v1,
        record_context_card_id=record_context_card_id,
        trusted_registry_closure_receipt_sha256=(
            trusted_registry_closure_receipt_sha256
        ),
        **trusted_validation_context,
    )
    if candidate != expected:
        raise ValueError(
            "Findings-to-SOZ graph does not replay exactly from typed sources"
        )
    return candidate


def _query_index(event_card: Mapping[str, Any], query_bundle: Mapping[str, Any]) -> dict[str, Any]:
    query_slot = {
        str(row["term_query_id"]): str(slot["slot_id"])
        for slot in event_card["slots"]
        for row in slot["operational_queries"]
    }
    missing = sorted(set(REQUIRED_QUERY_IDS) - set(query_slot))
    if missing:
        raise ValueError(f"Event Card v2 lost required event queries: {missing}")
    rows = []
    for artifact in query_bundle["query_artifacts"]:
        query_id = str(artifact["term_query_id"])
        if query_slot[query_id] != artifact["event_card_slot_id"]:
            raise ValueError(f"{query_id}: query artifact/Event Card slot mismatch")
        rows.append(
            {
                "term_query_id": query_id,
                "slot_id": query_slot[query_id],
                "artifact_id": str(artifact["artifact_id"]),
                "artifact_sha256": str(artifact["artifact_sha256"]),
                "report_promotion_authorized": False,
                "onset_support_authorized": False,
                "soz_support_authorized": False,
            }
        )
    index: dict[str, Any] = {
        "schema_version": "clinical_eeg_event_card_required_query_index_v1",
        "event_card_projection_id": str(event_card["projection_id"]),
        "event_card_projection_sha256": str(event_card["projection_sha256"]),
        "required_query_bundle_id": str(query_bundle["bundle_id"]),
        "required_query_bundle_sha256": str(query_bundle["bundle_sha256"]),
        "query_artifacts": rows,
        "record_context_payload_embedded": False,
        "report_eligible_term_allowlist": [],
        "index_sha256": "",
    }
    index["index_sha256"] = _self_hash(index, "index_sha256")
    return index


def materialize_findings_v1_disk_closure_v1(
    *,
    event_findings_v3: object,
    registry_closure_receipt_v1: object,
    record_context_card_v1: object,
    waveform_morphology_receipt: object,
    periodicity_candidates: Sequence[object],
    frequency_candidate: ACNSFrequencyEvolutionCandidate,
    morphology_evolution_receipt: object,
    morphology_change_points: Sequence[EventChangePointProposal],
    multireference_field_result: object,
    baseline_context_receipt: object,
    physical_amplitude_findings_receipt: object | None = None,
    component_cycle_element_ledger_receipt: object | None = None,
    trusted_registry_closure_receipt_sha256: str | None = None,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Materialize the complete in-memory closure before any disk write."""

    validation_context = {
        "trusted_producer_receipts": trusted_producer_receipts,
        "trusted_calibration_receipts": trusted_calibration_receipts,
        "trusted_capability_qualification_receipts": trusted_capability_qualification_receipts,
        "trusted_sensitivity_receipts": trusted_sensitivity_receipts,
        "trusted_term_decision_receipts": trusted_term_decision_receipts,
        "trusted_registry_bindings": trusted_registry_bindings,
    }
    event_source = validate_event_eeg_findings_v3_payload(
        event_findings_v3, **validation_context
    )
    context = validate_record_non_event_context_card_v1(record_context_card_v1)
    if context["owner"]["recording_id"] != event_source["provenance"]["record_id"]:
        raise ValueError("record context/Event Findings recording identity drifted")
    if context["owner"]["source_signal_sha256"] != event_source["provenance"][
        "canonical_signal_sha256"
    ]:
        raise ValueError("record context/Event Findings signal identity drifted")
    query_bundle = materialize_findings_v1_required_query_bundle_v1(
        waveform_morphology_receipt=waveform_morphology_receipt,
        periodicity_candidates=periodicity_candidates,
        frequency_candidate=frequency_candidate,
        morphology_evolution_receipt=morphology_evolution_receipt,
        morphology_change_points=morphology_change_points,
        multireference_field_result=multireference_field_result,
        baseline_context_receipt=baseline_context_receipt,
        physical_amplitude_findings_receipt=physical_amplitude_findings_receipt,
        component_cycle_element_ledger_receipt=(
            component_cycle_element_ledger_receipt
        ),
    )
    if query_bundle["event_id"] != event_source["event_id"]:
        raise ValueError("required-query/Event Findings event identity drifted")
    if query_bundle["recording_id"] != event_source["provenance"]["record_id"]:
        raise ValueError("required-query/Event Findings recording identity drifted")
    if query_bundle["canonical_signal_sha256"] != event_source["provenance"][
        "canonical_signal_sha256"
    ]:
        raise ValueError("required-query/Event Findings signal identity drifted")
    event_card = materialize_event_card_projection_v2(
        event_findings_v3=event_source,
        registry_closure_receipt_v1=registry_closure_receipt_v1,
        record_context_card_id=context["card_id"],
        trusted_registry_closure_receipt_sha256=(
            trusted_registry_closure_receipt_sha256
        ),
        **validation_context,
    )
    query_index = _query_index(event_card, query_bundle)
    graph = materialize_findings_to_research_soz_evidence_graph_v1(
        event_findings_v3=event_source,
        event_card_projection_v2=event_card,
        required_query_bundle_v1=query_bundle,
        registry_closure_receipt_v1=registry_closure_receipt_v1,
        record_context_card_id=context["card_id"],
        trusted_registry_closure_receipt_sha256=(
            trusted_registry_closure_receipt_sha256
        ),
        **validation_context,
    )
    closure: dict[str, Any] = {
        "schema_version": FINDINGS_V1_DISK_CLOSURE_SCHEMA_VERSION,
        "closure_id": "FDCLOSE-"
        + _domain_sha256(
            "clinical-eeg-findings-v1-disk-closure-id-v1",
            {
                "event_findings_v3_sha256": _sha256(event_source),
                "query_bundle_sha256": query_bundle["bundle_sha256"],
                "context_card_sha256": context["card_sha256"],
                "event_card_projection_sha256": event_card["projection_sha256"],
                "query_index_sha256": query_index["index_sha256"],
                "soz_graph_sha256": graph["graph_sha256"],
            },
        )[:24],
        "owner": {
            "event_id": str(event_source["event_id"]),
            "recording_id": str(event_source["provenance"]["record_id"]),
            "canonical_signal_sha256": str(
                event_source["provenance"]["canonical_signal_sha256"]
            ),
        },
        "source_event_findings_v3": deepcopy(event_source),
        "source_registry_closure_receipt_v1": deepcopy(
            registry_closure_receipt_v1
        ),
        "required_query_bundle": query_bundle,
        "record_context_card": context,
        "event_card_projection": event_card,
        "event_card_query_index": query_index,
        "findings_to_soz_evidence_graph": graph,
        "closure": {
            "ten_required_queries_materialized": True,
            "ten_required_queries_source_replayed": True,
            "twelve_event_slots_materialized": True,
            "record_interictal_excluded_from_event_card": True,
            "six_record_context_slots_reference_only": True,
            "findings_to_soz_graph_materialized": True,
            "only_future_free_onset_evidence_supports_ranking": True,
            "ranking_score_replayed_from_typed_causal_leaf_count_receipts": True,
            "source_numeric_ranking_scores_consumed": False,
            "uncalibrated_probabilities_null": True,
            "report_allowlist_empty": True,
            "clinical_term_qualification_opened": False,
        },
        "firewall": deepcopy(_FIREWALL),
        "authorization": deepcopy(_GRAPH_AUTHORIZATION),
        "closure_sha256": "",
    }
    closure["closure_sha256"] = _self_hash(closure, "closure_sha256")
    return closure


def validate_findings_v1_disk_closure_v1(
    value: object,
    *,
    trusted_registry_closure_receipt_sha256: str | None = None,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Validate every layer and its exact cross-layer content binding."""

    if type(value) is not dict:
        raise TypeError("Findings disk closure must be an object")
    candidate = deepcopy(value)
    _reject_nonfinite(candidate)
    if candidate.get("schema_version") != FINDINGS_V1_DISK_CLOSURE_SCHEMA_VERSION:
        raise ValueError("Findings disk closure schema drifted")
    if candidate.get("firewall") != _FIREWALL or candidate.get("authorization") != _GRAPH_AUTHORIZATION:
        raise ValueError("Findings disk closure firewall/authorization drifted")
    if candidate.get("closure_sha256") != _self_hash(candidate, "closure_sha256"):
        raise ValueError("Findings disk closure self hash drifted")
    validation_context = {
        "trusted_producer_receipts": trusted_producer_receipts,
        "trusted_calibration_receipts": trusted_calibration_receipts,
        "trusted_capability_qualification_receipts": trusted_capability_qualification_receipts,
        "trusted_sensitivity_receipts": trusted_sensitivity_receipts,
        "trusted_term_decision_receipts": trusted_term_decision_receipts,
        "trusted_registry_bindings": trusted_registry_bindings,
    }
    event_source = validate_event_eeg_findings_v3_payload(
        candidate["source_event_findings_v3"], **validation_context
    )
    query_bundle = validate_findings_v1_required_query_bundle_v1(
        candidate["required_query_bundle"]
    )
    context = validate_record_non_event_context_card_v1(
        candidate["record_context_card"]
    )
    event_card = validate_event_card_projection_v2(
        candidate["event_card_projection"],
        source_event_findings_v3=event_source,
        source_registry_closure_receipt_v1=candidate[
            "source_registry_closure_receipt_v1"
        ],
        record_context_card_id=context["card_id"],
        trusted_registry_closure_receipt_sha256=(
            trusted_registry_closure_receipt_sha256
        ),
        **validation_context,
    )
    expected_index = _query_index(event_card, query_bundle)
    if candidate["event_card_query_index"] != expected_index:
        raise ValueError("Event Card/query artifact index does not replay")
    graph = validate_findings_to_research_soz_evidence_graph_v1(
        candidate["findings_to_soz_evidence_graph"],
        event_findings_v3=event_source,
        event_card_projection_v2=event_card,
        required_query_bundle_v1=query_bundle,
        registry_closure_receipt_v1=candidate[
            "source_registry_closure_receipt_v1"
        ],
        record_context_card_id=context["card_id"],
        trusted_registry_closure_receipt_sha256=(
            trusted_registry_closure_receipt_sha256
        ),
        **validation_context,
    )
    if graph != candidate["findings_to_soz_evidence_graph"]:
        raise ValueError("Findings-to-SOZ graph drifted")
    owner = candidate["owner"]
    if owner != {
        "event_id": event_source["event_id"],
        "recording_id": event_source["provenance"]["record_id"],
        "canonical_signal_sha256": event_source["provenance"]["canonical_signal_sha256"],
    }:
        raise ValueError("Findings disk closure owner drifted")
    if candidate["closure"] != {
        "ten_required_queries_materialized": True,
        "ten_required_queries_source_replayed": True,
        "twelve_event_slots_materialized": True,
        "record_interictal_excluded_from_event_card": True,
        "six_record_context_slots_reference_only": True,
        "findings_to_soz_graph_materialized": True,
        "only_future_free_onset_evidence_supports_ranking": True,
        "ranking_score_replayed_from_typed_causal_leaf_count_receipts": True,
        "source_numeric_ranking_scores_consumed": False,
        "uncalibrated_probabilities_null": True,
        "report_allowlist_empty": True,
        "clinical_term_qualification_opened": False,
    }:
        raise ValueError("Findings disk closure status receipt drifted")
    sources = query_bundle["source_artifacts"]
    expected = materialize_findings_v1_disk_closure_v1(
        event_findings_v3=event_source,
        registry_closure_receipt_v1=candidate[
            "source_registry_closure_receipt_v1"
        ],
        record_context_card_v1=context,
        waveform_morphology_receipt=sources["waveform_morphology_receipt"],
        periodicity_candidates=sources["periodicity_candidates"],
        frequency_candidate=_frequency_candidate_from_dict(
            sources["frequency_candidate"]
        ),
        morphology_evolution_receipt=sources[
            "morphology_evolution_receipt"
        ],
        morphology_change_points=[
            _change_point_from_dict(row)
            for row in sources["morphology_change_points"]
        ],
        multireference_field_result=sources["multireference_field_result"],
        baseline_context_receipt=sources["baseline_context_receipt"],
        physical_amplitude_findings_receipt=sources.get(
            "physical_amplitude_findings_receipt"
        ),
        component_cycle_element_ledger_receipt=sources.get(
            "component_cycle_element_ledger_receipt"
        ),
        trusted_registry_closure_receipt_sha256=(
            trusted_registry_closure_receipt_sha256
        ),
        **validation_context,
    )
    if candidate != expected:
        raise ValueError("Findings disk closure does not replay exactly from sources")
    return candidate


def _logical_files(value: Mapping[str, Any]) -> dict[str, object]:
    query_bundle = value["required_query_bundle"]
    files: dict[str, object] = {
        "sources/event_findings_v3.json": value["source_event_findings_v3"],
        "sources/required_query_sources.json": query_bundle["source_artifacts"],
        "registry/minimum_event_card_closure_v1.json": value[
            "source_registry_closure_receipt_v1"
        ],
        "record/context_card_v1.json": value["record_context_card"],
        "event/event_card_v2.json": value["event_card_projection"],
        "event/event_card_query_index_v1.json": value["event_card_query_index"],
        "event/findings_to_soz_evidence_graph_v1.json": value[
            "findings_to_soz_evidence_graph"
        ],
    }
    for row in query_bundle["query_artifacts"]:
        query_id = str(row["term_query_id"])
        if _QUERY_FILENAME_RE.fullmatch(query_id + ".json") is None:
            raise ValueError("query ID cannot be used as a safe disk filename")
        files[f"event/queries/{query_id}.json"] = row
    return files


def write_findings_v1_disk_closure_v1(
    value: object,
    output_directory: str | Path,
    **validation_context: Any,
) -> dict[str, Any]:
    """Write an exact no-clobber JSON bundle and return its disk manifest."""

    checked = validate_findings_v1_disk_closure_v1(
        value, **validation_context
    )
    root = Path(output_directory)
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise ValueError("Findings disk output must be a regular directory")
        if any(root.iterdir()):
            raise FileExistsError("Findings disk output directory is not empty")
    else:
        root.mkdir(parents=True, exist_ok=False)
    logical = _logical_files(checked)
    inventory: list[dict[str, object]] = []
    for relative in sorted(logical):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload_bytes = _canonical_file_bytes(logical[relative])
        with target.open("xb") as handle:
            handle.write(payload_bytes)
        inventory.append(
            {
                "relative_path": relative,
                "byte_count": len(payload_bytes),
                "file_sha256": _sha256_bytes(payload_bytes),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": FINDINGS_V1_DISK_MANIFEST_SCHEMA_VERSION,
        "closure_id": str(checked["closure_id"]),
        "closure_sha256": str(checked["closure_sha256"]),
        "file_inventory": inventory,
        "expected_file_count": len(inventory),
        "unknown_files_authorized": False,
        "symlinks_authorized": False,
        "report_artifacts_written": False,
        "report_eligible_term_allowlist": [],
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = _self_hash(manifest, "manifest_sha256")
    manifest_path = root / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(_canonical_file_bytes(manifest))
    return manifest


def read_findings_v1_disk_closure_v1(
    output_directory: str | Path,
    *,
    trusted_manifest_sha256: str | None = None,
    **validation_context: Any,
) -> dict[str, Any]:
    """Read, byte-verify, reconstruct and replay one disk closure."""

    root = Path(output_directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Findings disk bundle root must be a regular directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Findings disk manifest must be a regular file")
    manifest = _read_json_bytes(manifest_path.read_bytes(), "manifest.json")
    manifest_fields = {
        "schema_version",
        "closure_id",
        "closure_sha256",
        "file_inventory",
        "expected_file_count",
        "unknown_files_authorized",
        "symlinks_authorized",
        "report_artifacts_written",
        "report_eligible_term_allowlist",
        "manifest_sha256",
    }
    if set(manifest) != manifest_fields:
        raise ValueError("Findings disk manifest fields drifted")
    if manifest.get("schema_version") != FINDINGS_V1_DISK_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Findings disk manifest schema drifted")
    _identifier(manifest["closure_id"], "manifest closure_id")
    _hash(manifest["closure_sha256"], "manifest closure_sha256")
    _hash(manifest["manifest_sha256"], "manifest_sha256")
    if manifest.get("manifest_sha256") != _self_hash(manifest, "manifest_sha256"):
        raise ValueError("Findings disk manifest self hash drifted")
    if trusted_manifest_sha256 is not None and manifest["manifest_sha256"] != _hash(
        trusted_manifest_sha256, "trusted_manifest_sha256"
    ):
        raise ValueError("Findings disk manifest is not host trusted")
    if (
        manifest["unknown_files_authorized"] is not False
        or manifest["symlinks_authorized"] is not False
        or manifest["report_artifacts_written"] is not False
        or manifest["report_eligible_term_allowlist"] != []
    ):
        raise ValueError("Findings disk manifest authorization drifted")
    inventory = manifest["file_inventory"]
    if not isinstance(inventory, list):
        raise TypeError("Findings disk inventory must be an array")
    if type(manifest["expected_file_count"]) is not int or (
        manifest["expected_file_count"] != len(inventory)
    ):
        raise ValueError("Findings disk manifest file count drifted")
    inventory_paths: list[str] = []
    for index, row in enumerate(inventory):
        if type(row) is not dict or set(row) != {
            "relative_path",
            "byte_count",
            "file_sha256",
        }:
            raise ValueError(f"Findings disk inventory row {index} fields drifted")
        relative = str(row["relative_path"])
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or relative_path.as_posix() != relative
            or not relative_path.parts
            or any(part in {"", ".", ".."} for part in relative_path.parts)
            or relative == "manifest.json"
        ):
            raise ValueError("Findings disk inventory path is not canonical relative")
        if type(row["byte_count"]) is not int or row["byte_count"] <= 0:
            raise ValueError("Findings disk inventory byte count is invalid")
        _hash(row["file_sha256"], "inventory file_sha256")
        inventory_paths.append(relative)
    if inventory_paths != sorted(inventory_paths) or len(set(inventory_paths)) != len(
        inventory_paths
    ):
        raise ValueError("Findings disk inventory order/roster drifted")
    expected_paths = {"manifest.json"} | {
        str(row["relative_path"]) for row in inventory
    }
    actual_paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Findings disk bundle contains a symlink")
        if path.is_file():
            actual_paths.add(path.relative_to(root).as_posix())
    if actual_paths != expected_paths:
        raise ValueError(
            "Findings disk file roster drifted: "
            f"missing={sorted(expected_paths-actual_paths)}, "
            f"extra={sorted(actual_paths-expected_paths)}"
        )
    payload_by_path: dict[str, dict[str, Any]] = {}
    for row in inventory:
        relative = str(row["relative_path"])
        target = root / relative
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"disk artifact {relative!r} is not a regular file")
        data = target.read_bytes()
        if len(data) != int(row["byte_count"]) or _sha256_bytes(data) != row[
            "file_sha256"
        ]:
            raise ValueError(f"disk artifact {relative!r} content hash drifted")
        payload_by_path[relative] = _read_json_bytes(data, relative)
    query_artifacts = [
        payload_by_path[f"event/queries/{query_id}.json"]
        for query_id in REQUIRED_QUERY_IDS
    ]
    query_sources = payload_by_path["sources/required_query_sources.json"]
    query_bundle: dict[str, Any] = {
        "schema_version": FINDINGS_V1_REQUIRED_QUERY_BUNDLE_SCHEMA_VERSION,
        "bundle_id": payload_by_path["event/event_card_query_index_v1.json"][
            "required_query_bundle_id"
        ],
        "event_id": query_artifacts[0]["event_id"],
        "recording_id": query_artifacts[0]["recording_id"],
        "canonical_signal_sha256": query_artifacts[0]["canonical_signal_sha256"],
        "source_artifacts": query_sources,
        "source_closure_sha256": query_artifacts[0]["source_closure_sha256"],
        "waveform_bridge_receipt_sha256": "",
        "query_artifacts": query_artifacts,
        "closure": {
            "expected_query_count": 10,
            "observed_query_count": 10,
            "query_ids": list(REQUIRED_QUERY_IDS),
            "every_query_exactly_once": True,
            "all_sources_replayed": True,
            "all_artifacts_content_bound": True,
            "uncalibrated_probabilities_null": True,
            "report_allowlist_empty": True,
        },
        "firewall": deepcopy(_FIREWALL),
        "authorization": deepcopy(_QUERY_AUTHORIZATION),
        "bundle_sha256": payload_by_path["event/event_card_query_index_v1.json"][
            "required_query_bundle_sha256"
        ],
    }
    waveform_bundle, _ = _compose_required_queries(query_sources)
    query_bundle["waveform_bridge_receipt_sha256"] = waveform_bundle[
        "receipt_sha256"
    ]
    query_bundle = validate_findings_v1_required_query_bundle_v1(query_bundle)
    closure: dict[str, Any] = {
        "schema_version": FINDINGS_V1_DISK_CLOSURE_SCHEMA_VERSION,
        "closure_id": manifest["closure_id"],
        "owner": payload_by_path["event/findings_to_soz_evidence_graph_v1.json"][
            "owner"
        ],
        "source_event_findings_v3": payload_by_path["sources/event_findings_v3.json"],
        "source_registry_closure_receipt_v1": payload_by_path[
            "registry/minimum_event_card_closure_v1.json"
        ],
        "required_query_bundle": query_bundle,
        "record_context_card": payload_by_path["record/context_card_v1.json"],
        "event_card_projection": payload_by_path["event/event_card_v2.json"],
        "event_card_query_index": payload_by_path[
            "event/event_card_query_index_v1.json"
        ],
        "findings_to_soz_evidence_graph": payload_by_path[
            "event/findings_to_soz_evidence_graph_v1.json"
        ],
        "closure": {
            "ten_required_queries_materialized": True,
            "ten_required_queries_source_replayed": True,
            "twelve_event_slots_materialized": True,
            "record_interictal_excluded_from_event_card": True,
            "six_record_context_slots_reference_only": True,
            "findings_to_soz_graph_materialized": True,
            "only_future_free_onset_evidence_supports_ranking": True,
            "ranking_score_replayed_from_typed_causal_leaf_count_receipts": True,
            "source_numeric_ranking_scores_consumed": False,
            "uncalibrated_probabilities_null": True,
            "report_allowlist_empty": True,
            "clinical_term_qualification_opened": False,
        },
        "firewall": deepcopy(_FIREWALL),
        "authorization": deepcopy(_GRAPH_AUTHORIZATION),
        "closure_sha256": manifest["closure_sha256"],
    }
    return validate_findings_v1_disk_closure_v1(
        closure, **validation_context
    )


__all__ = [
    "FINDINGS_TO_SOZ_EVIDENCE_GRAPH_SCHEMA_VERSION",
    "FINDINGS_V1_DISK_CLOSURE_SCHEMA_VERSION",
    "FINDINGS_V1_DISK_MANIFEST_SCHEMA_VERSION",
    "FINDINGS_V1_REQUIRED_QUERY_BUNDLE_SCHEMA_VERSION",
    "REQUIRED_QUERY_IDS",
    "REQUIRED_QUERY_SLOT",
    "materialize_findings_to_research_soz_evidence_graph_v1",
    "materialize_findings_v1_disk_closure_v1",
    "materialize_findings_v1_required_query_bundle_v1",
    "read_findings_v1_disk_closure_v1",
    "validate_findings_to_research_soz_evidence_graph_v1",
    "validate_findings_v1_disk_closure_v1",
    "validate_findings_v1_required_query_bundle_v1",
    "write_findings_v1_disk_closure_v1",
]
