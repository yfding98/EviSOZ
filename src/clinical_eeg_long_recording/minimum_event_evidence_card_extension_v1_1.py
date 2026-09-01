"""Candidate-blind additive Minimum Event Evidence Card v1.1 contract.

The frozen v1 registry closes the original 28 core atoms, 12 repeatable child
rosters and 41 operational queries.  It intentionally does not pretend that
the later method additions -- possible/unequivocal onset, emergence manner,
event-duration intervals, spatial recruitment, termination field, spatial
post-event changes and recording burden -- were already machine closed.

This module adds those questions without changing the v1 registry or any of
its hashes.  Fourteen questions belong to an event card; three recording
aggregate questions live in a separate destination and may not be copied into
each event.  The registry is candidate blind and structural only.  It cannot
inspect an event payload, change a status/assertion level, authorize an
absence, qualify a term, promote report text, infer cortical propagation
velocity, or authorize an SOZ/EZ claim.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .minimum_event_evidence_card_registry_v1 import (
    DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_REGISTRY_SHA256_V1,
    load_minimum_event_evidence_card_registry_v1,
    validate_minimum_event_evidence_card_registry_v1,
)


MINIMUM_EVENT_EVIDENCE_CARD_EXTENSION_REGISTRY_SCHEMA_VERSION_V1_1 = (
    "clinical_eeg_minimum_event_evidence_card_extension_registry_v1_1"
)
MINIMUM_EVENT_EVIDENCE_CARD_EXTENSION_CLOSURE_RECEIPT_SCHEMA_VERSION_V1_1 = (
    "clinical_eeg_minimum_event_evidence_card_extension_closure_receipt_v1_1"
)
MINIMUM_EVENT_EVIDENCE_CARD_EXTENSION_REGISTRY_ID_V1_1 = (
    "CLINICAL-EEG-MINIMUM-EVENT-EVIDENCE-CARD-EXTENSION-REGISTRY-V1.1"
)
MINIMUM_EVENT_EVIDENCE_CARD_EXTENSION_CLOSURE_METHOD_ID_V1_1 = (
    "CANDIDATE-BLIND-MINIMUM-EVENT-EVIDENCE-CARD-EXTENSION-CLOSURE-V1.1"
)
DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_EXTENSION_REGISTRY_SHA256_V1_1 = (
    "ee82600c8c7b0c19ef78e9fbf19817f86a1f4d19790debf18bb5346142d77656"
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_EXTENSION_REGISTRY_PATH_V1_1 = (
    _ROOT
    / "configs"
    / "clinical_eeg_minimum_event_evidence_card_extension_registry_v1_1.json"
)
MINIMUM_EVENT_EVIDENCE_CARD_EXTENSION_REGISTRY_SCHEMA_PATH_V1_1 = (
    _ROOT
    / "schemas"
    / "clinical_eeg_minimum_event_evidence_card_extension_registry_v1_1.schema.json"
)
MINIMUM_EVENT_EVIDENCE_CARD_EXTENSION_CLOSURE_RECEIPT_SCHEMA_PATH_V1_1 = (
    _ROOT
    / "schemas"
    / "clinical_eeg_minimum_event_evidence_card_extension_closure_receipt_v1_1.schema.json"
)

_DESTINATION_IDS = (
    "S01_SOURCE_EVALUABILITY",
    "S02_EVENT_BOUNDARY",
    "S03_FREQUENCY",
    "S04_PHYSICAL_AMPLITUDE",
    "S05_WAVEFORM_MORPHOLOGY",
    "S06_RHYTHMICITY_PERIODICITY",
    "S07_EARLIEST_VISIBLE_SET",
    "S08_SPATIAL_FIELD_REFERENCE_STABILITY",
    "S09_CHANGE_POINTS_EVOLUTION",
    "S10_LATER_INVOLVEMENT",
    "S11_TERMINATION_POST_EVENT",
    "S12_COMPARABLE_BACKGROUND_RECOVERY",
    "R01_RECORD_EVENT_ROSTER_AND_BURDEN",
)
_EVENT_DESTINATION_IDS = frozenset(_DESTINATION_IDS[:12])
_RECORD_DESTINATION_ID = _DESTINATION_IDS[-1]
_EXPECTED_QUERY_IDS = (
    "MEC11-EVENT-DURATION-INTERVAL",
    "MEC11-EVENT-EARLIEST-POSSIBLE-ONSET",
    "MEC11-EVENT-EARLIEST-UNEQUIVOCAL-ONSET",
    "MEC11-EVENT-EMERGENCE-MANNER",
    "MEC11-ONSET-PER-UNIT-POSSIBLE-INTERVAL",
    "MEC11-ONSET-PER-UNIT-UNEQUIVOCAL-INTERVAL",
    "MEC11-POST-EVENT-LATERALIZED-SLOWING-DURATION",
    "MEC11-POST-EVENT-LATERALIZED-SLOWING-FIELD",
    "MEC11-POST-EVENT-LATERALIZED-SUPPRESSION-DURATION",
    "MEC11-POST-EVENT-LATERALIZED-SUPPRESSION-FIELD",
    "MEC11-RECORD-DETECTED-QUALIFIED-EVENT-COUNT",
    "MEC11-RECORD-INTER-EVENT-INTERVAL-DISTRIBUTION",
    "MEC11-RECORD-QUALIFIED-ICTAL-PATTERN-BURDEN",
    "MEC11-SPATIAL-RECRUITMENT-LATENCY",
    "MEC11-SPATIAL-RECRUITMENT-RATE",
    "MEC11-TERMINATION-ASYNCHRONY",
    "MEC11-TERMINATION-FIELD",
)
_EXPECTED_DESTINATION_COUNTS = {
    "S02_EVENT_BOUNDARY": 4,
    "S07_EARLIEST_VISIBLE_SET": 2,
    "S10_LATER_INVOLVEMENT": 2,
    "S11_TERMINATION_POST_EVENT": 6,
    "R01_RECORD_EVENT_ROSTER_AND_BURDEN": 3,
}
_EXPECTED_ASSERTION_LEVEL_DOMAIN = [
    "measured",
    "model_candidate",
    "report_eligible_automated",
]
_EXPECTED_STATUS_DOMAIN = [
    "present",
    "absent_with_opportunity",
    "uncertain",
    "not_evaluable",
]
_SOURCE_FIREWALL = {
    "private_data_used": False,
    "event_findings_payload_used": False,
    "findings_candidates_used": False,
    "payload_evaluation_opportunities_used": False,
    "event_outcome_used": False,
    "scalp_onset_hypothesis_used": False,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_used": False,
    "sleep_staging_used": False,
    "provocation_used": False,
    "ecg_emg_eog_used": False,
    "qwen_used": False,
}
_AUTHORIZATION = {
    "registry_is_structural_only": True,
    "status_or_assertion_mutation_authorized": False,
    "clinical_absence_authorized": False,
    "report_promotion_authorized": False,
    "clinical_correctness_claimed": False,
    "cortical_propagation_velocity_claim_authorized": False,
    "soz_or_ez_claim_authorized": False,
}
_EXPECTED_QUERY_ROWS_SHA256_V1_1 = (
    "3c76ca4a192dce0797f2607bd32ee62b4baaf599868a3e8b0f2f084a6a75836f"
)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _self_hash(value: Mapping[str, object], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _canonical_sha256(body)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if type(value) is not dict:
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _schema_errors(value: object, path: Path) -> list[str]:
    schema = _read_json(path)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: list(item.path),
    )
    rendered: list[str] = []
    for error in errors[:16]:
        pointer = "/" + "/".join(str(part) for part in error.path)
        rendered.append(f"{pointer}: {error.message}")
    if len(errors) > 16:
        rendered.append(f"... {len(errors) - 16} more error(s)")
    return rendered


def _require_sha256(value: object, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _require_id_array(value: object, context: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be an ID array")
    result: list[str] = []
    for raw in value:
        if type(raw) is not str or _ID_RE.fullmatch(raw) is None:
            raise ValueError(f"{context} contains a non-canonical ID")
        result.append(raw)
    if len(result) != len(set(result)):
        raise ValueError(f"{context} must be unique")
    return result


def _base_registry(
    value: Mapping[str, object] | None,
    *,
    trusted_base_registry_sha256: str | None,
) -> dict[str, Any]:
    if value is None:
        if trusted_base_registry_sha256 is None:
            trusted_base_registry_sha256 = (
                DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_REGISTRY_SHA256_V1
            )
        return load_minimum_event_evidence_card_registry_v1(
            trusted_registry_sha256=trusted_base_registry_sha256
        )
    if trusted_base_registry_sha256 is None:
        trusted_base_registry_sha256 = (
            DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_REGISTRY_SHA256_V1
        )
    return validate_minimum_event_evidence_card_registry_v1(
        dict(value),
        trusted_registry_sha256=trusted_base_registry_sha256,
    )


def _base_binding(value: Mapping[str, object]) -> dict[str, str]:
    return {
        "registry_id": str(value["registry_id"]),
        "registry_sha256": str(value["registry_sha256"]),
    }


def _assert_acyclic_extension_dependencies(
    rows_by_id: Mapping[str, Mapping[str, object]],
) -> None:
    state: dict[str, int] = {}

    def visit(query_id: str, trail: tuple[str, ...]) -> None:
        marker = state.get(query_id, 0)
        if marker == 1:
            raise ValueError(
                "v1.1 extension dependency graph contains a cycle: "
                + " -> ".join((*trail, query_id))
            )
        if marker == 2:
            return
        state[query_id] = 1
        for dependency in rows_by_id[query_id]["derived_from_query_ids"]:
            dependency_id = str(dependency)
            if dependency_id in rows_by_id:
                visit(dependency_id, (*trail, query_id))
        state[query_id] = 2

    for query_id in rows_by_id:
        visit(query_id, ())


def _validate_registry(
    value: object,
    *,
    trusted_registry_sha256: str | None,
    base_registry: Mapping[str, object] | None,
    trusted_base_registry_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(value) is not dict:
        raise TypeError("Minimum Event Evidence Card v1.1 registry must be an object")
    candidate = deepcopy(value)
    errors = _schema_errors(
        candidate,
        MINIMUM_EVENT_EVIDENCE_CARD_EXTENSION_REGISTRY_SCHEMA_PATH_V1_1,
    )
    if errors:
        raise ValueError(
            "Minimum Event Evidence Card v1.1 registry schema validation failed: "
            + "; ".join(errors)
        )
    expected_hash = _self_hash(candidate, "registry_sha256")
    if candidate["registry_sha256"] != expected_hash:
        raise ValueError("Minimum Event Evidence Card v1.1 registry SHA-256 mismatch")
    if trusted_registry_sha256 is not None and expected_hash != _require_sha256(
        trusted_registry_sha256, "trusted_registry_sha256"
    ):
        raise ValueError(
            "Minimum Event Evidence Card v1.1 registry is not host trusted"
        )

    base = _base_registry(
        base_registry,
        trusted_base_registry_sha256=trusted_base_registry_sha256,
    )
    if candidate["base_registry"] != _base_binding(base):
        raise ValueError(
            "Minimum Event Evidence Card v1.1 base-registry binding drifted"
        )
    if candidate["source_firewall"] != _SOURCE_FIREWALL:
        raise ValueError("Minimum Event Evidence Card v1.1 source firewall drifted")
    if candidate["authorization"] != _AUTHORIZATION:
        raise ValueError("Minimum Event Evidence Card v1.1 authorization drifted")
    if tuple(candidate["destination_order"]) != _DESTINATION_IDS:
        raise ValueError("Minimum Event Evidence Card v1.1 destination order drifted")

    query_rows = list(candidate["query_specs"])
    query_ids = tuple(str(row["extension_query_id"]) for row in query_rows)
    if query_ids != _EXPECTED_QUERY_IDS:
        raise ValueError("Minimum Event Evidence Card v1.1 query roster drifted")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Minimum Event Evidence Card v1.1 query IDs must be unique")
    rows_by_id = {str(row["extension_query_id"]): row for row in query_rows}

    base_query_ids = {
        str(query_id)
        for slot in base["slots"]
        for query_id in slot["operational_query_ids"]
    }
    known_dependencies = base_query_ids | set(rows_by_id)
    destination_counts: dict[str, int] = {}
    event_count = 0
    record_count = 0
    for row in query_rows:
        query_id = str(row["extension_query_id"])
        destination_id = str(row["destination_id"])
        analysis_unit = str(row["analysis_unit"])
        destination_counts[destination_id] = (
            destination_counts.get(destination_id, 0) + 1
        )
        if analysis_unit == "event":
            event_count += 1
            if destination_id not in _EVENT_DESTINATION_IDS:
                raise ValueError(
                    f"{query_id} event query is outside the twelve-slot card"
                )
        elif analysis_unit == "recording":
            record_count += 1
            if destination_id != _RECORD_DESTINATION_ID:
                raise ValueError(
                    f"{query_id} recording query was copied into an event slot"
                )
        else:  # JSON Schema closes this branch; keep defense in depth.
            raise ValueError(f"{query_id} has an invalid analysis unit")

        dependencies = _require_id_array(
            row["derived_from_query_ids"],
            f"{query_id}.derived_from_query_ids",
        )
        if not dependencies:
            raise ValueError(f"{query_id} must retain at least one source dependency")
        missing = sorted(set(dependencies) - known_dependencies)
        if missing:
            raise ValueError(f"{query_id} has unresolved dependencies: {missing}")
        receipts = _require_id_array(
            row["required_receipt_kinds"],
            f"{query_id}.required_receipt_kinds",
        )
        if not receipts:
            raise ValueError(f"{query_id} must require at least one receipt kind")
        if row["assertion_level_domain"] != _EXPECTED_ASSERTION_LEVEL_DOMAIN:
            raise ValueError(f"{query_id} assertion-level domain drifted")
        if row["status_domain"] != _EXPECTED_STATUS_DOMAIN:
            raise ValueError(f"{query_id} four-state status domain drifted")
        if row["implementation_status"] != "unimplemented_not_evaluable":
            raise ValueError(f"{query_id} cannot silently claim implementation")
        if row["report_promotion_authorized"] is not False:
            raise ValueError(f"{query_id} unexpectedly authorizes report promotion")

    if (event_count, record_count) != (14, 3):
        raise ValueError(
            "Minimum Event Evidence Card v1.1 analysis-unit counts drifted"
        )
    actual_nonzero_counts = {
        key: value for key, value in destination_counts.items() if value > 0
    }
    if actual_nonzero_counts != _EXPECTED_DESTINATION_COUNTS:
        raise ValueError("Minimum Event Evidence Card v1.1 destination mapping drifted")
    _assert_acyclic_extension_dependencies(rows_by_id)
    if _canonical_sha256(query_rows) != _EXPECTED_QUERY_ROWS_SHA256_V1_1:
        raise ValueError(
            "Minimum Event Evidence Card v1.1 frozen query semantics drifted"
        )
    return candidate, base


def validate_minimum_event_evidence_card_extension_registry_v1_1(
    value: object,
    *,
    trusted_registry_sha256: str | None = None,
    base_registry: Mapping[str, object] | None = None,
    trusted_base_registry_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the additive event/record query registry and its trust roots."""

    candidate, _ = _validate_registry(
        value,
        trusted_registry_sha256=trusted_registry_sha256,
        base_registry=base_registry,
        trusted_base_registry_sha256=trusted_base_registry_sha256,
    )
    return candidate


def load_minimum_event_evidence_card_extension_registry_v1_1(
    path: str
    | Path = (DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_EXTENSION_REGISTRY_PATH_V1_1),
    *,
    trusted_registry_sha256: str | None = None,
) -> dict[str, Any]:
    """Load the checked-in v1.1 extension under the default host trust anchor."""

    if trusted_registry_sha256 is None:
        trusted_registry_sha256 = (
            DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_EXTENSION_REGISTRY_SHA256_V1_1
        )
    return validate_minimum_event_evidence_card_extension_registry_v1_1(
        _read_json(Path(path)),
        trusted_registry_sha256=trusted_registry_sha256,
    )


def _registry_bundle(
    registry: Mapping[str, object] | None,
    *,
    trusted_registry_sha256: str | None,
    base_registry: Mapping[str, object] | None,
    trusted_base_registry_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if registry is None:
        registry_value = _read_json(
            DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_EXTENSION_REGISTRY_PATH_V1_1
        )
        if trusted_registry_sha256 is None:
            trusted_registry_sha256 = (
                DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_EXTENSION_REGISTRY_SHA256_V1_1
            )
    else:
        registry_value = dict(registry)
        if trusted_registry_sha256 is None:
            trusted_registry_sha256 = (
                DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_EXTENSION_REGISTRY_SHA256_V1_1
            )
    return _validate_registry(
        registry_value,
        trusted_registry_sha256=trusted_registry_sha256,
        base_registry=base_registry,
        trusted_base_registry_sha256=trusted_base_registry_sha256,
    )


def _closure_receipt_body(
    registry: Mapping[str, object],
) -> dict[str, Any]:
    rows_by_destination: dict[str, list[Mapping[str, object]]] = {
        destination_id: [] for destination_id in _DESTINATION_IDS
    }
    for row in registry["query_specs"]:
        rows_by_destination[str(row["destination_id"])].append(row)

    destinations: list[dict[str, object]] = []
    for index, destination_id in enumerate(_DESTINATION_IDS, start=1):
        query_rows: list[dict[str, object]] = []
        for row in rows_by_destination[destination_id]:
            query_rows.append(
                {
                    "extension_query_id": str(row["extension_query_id"]),
                    "analysis_unit": str(row["analysis_unit"]),
                    "destination_id": str(row["destination_id"]),
                    "output_type": str(row["output_type"]),
                    "derived_from_query_ids": list(row["derived_from_query_ids"]),
                    "required_receipt_kinds": list(row["required_receipt_kinds"]),
                    "assertion_level_domain": list(row["assertion_level_domain"]),
                    "status_domain": list(row["status_domain"]),
                    "evidence_role_ceiling": str(row["evidence_role_ceiling"]),
                    "source_implementation_status": str(row["implementation_status"]),
                    "source_report_promotion_authorized": bool(
                        row["report_promotion_authorized"]
                    ),
                }
            )
        destinations.append(
            {
                "destination_index": index,
                "destination_id": destination_id,
                "analysis_unit": (
                    "recording" if destination_id == _RECORD_DESTINATION_ID else "event"
                ),
                "extension_queries": query_rows,
            }
        )

    return {
        "schema_version": (
            MINIMUM_EVENT_EVIDENCE_CARD_EXTENSION_CLOSURE_RECEIPT_SCHEMA_VERSION_V1_1
        ),
        "method_id": MINIMUM_EVENT_EVIDENCE_CARD_EXTENSION_CLOSURE_METHOD_ID_V1_1,
        "registry_id": str(registry["registry_id"]),
        "registry_sha256": str(registry["registry_sha256"]),
        "base_registry": deepcopy(registry["base_registry"]),
        "source_firewall": deepcopy(_SOURCE_FIREWALL),
        "candidate_blind_denominator": True,
        "destinations": destinations,
        "summary": {
            "destination_count": 13,
            "legacy_event_slot_count": 12,
            "recording_aggregate_destination_count": 1,
            "extension_query_count": 17,
            "event_extension_query_count": 14,
            "recording_extension_query_count": 3,
            "retained_unimplemented_query_count": 17,
            "report_promoted_query_count": 0,
            "all_extension_queries_mapped_exactly_once": True,
            "dependencies_resolved": True,
            "dependency_graph_acyclic": True,
            "four_state_missingness_preserved": True,
            "event_and_recording_outputs_separated": True,
        },
        "authorization": deepcopy(_AUTHORIZATION),
    }


def materialize_minimum_event_evidence_card_extension_closure_receipt_v1_1(
    *,
    registry: Mapping[str, object] | None = None,
    trusted_registry_sha256: str | None = None,
    base_registry: Mapping[str, object] | None = None,
    trusted_base_registry_sha256: str | None = None,
) -> dict[str, Any]:
    """Materialize the deterministic v1.1 closure receipt without event data."""

    registry_value, base = _registry_bundle(
        registry,
        trusted_registry_sha256=trusted_registry_sha256,
        base_registry=base_registry,
        trusted_base_registry_sha256=trusted_base_registry_sha256,
    )
    receipt = _closure_receipt_body(registry_value)
    receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")
    return validate_minimum_event_evidence_card_extension_closure_receipt_v1_1(
        receipt,
        registry=registry_value,
        trusted_registry_sha256=str(registry_value["registry_sha256"]),
        base_registry=base,
        trusted_base_registry_sha256=str(base["registry_sha256"]),
    )


def validate_minimum_event_evidence_card_extension_closure_receipt_v1_1(
    value: object,
    *,
    registry: Mapping[str, object] | None = None,
    trusted_registry_sha256: str | None = None,
    base_registry: Mapping[str, object] | None = None,
    trusted_base_registry_sha256: str | None = None,
    trusted_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay the v1.1 registry and fail closed on any receipt mutation."""

    if type(value) is not dict:
        raise TypeError("Minimum Event Evidence Card v1.1 receipt must be an object")
    candidate = deepcopy(value)
    errors = _schema_errors(
        candidate,
        MINIMUM_EVENT_EVIDENCE_CARD_EXTENSION_CLOSURE_RECEIPT_SCHEMA_PATH_V1_1,
    )
    if errors:
        raise ValueError(
            "Minimum Event Evidence Card v1.1 receipt schema validation failed: "
            + "; ".join(errors)
        )
    expected_hash = _self_hash(candidate, "receipt_sha256")
    if candidate["receipt_sha256"] != expected_hash:
        raise ValueError("Minimum Event Evidence Card v1.1 receipt SHA-256 mismatch")
    if trusted_receipt_sha256 is not None and expected_hash != _require_sha256(
        trusted_receipt_sha256, "trusted_receipt_sha256"
    ):
        raise ValueError("Minimum Event Evidence Card v1.1 receipt is not trusted")

    registry_value, _ = _registry_bundle(
        registry,
        trusted_registry_sha256=trusted_registry_sha256,
        base_registry=base_registry,
        trusted_base_registry_sha256=trusted_base_registry_sha256,
    )
    expected = _closure_receipt_body(registry_value)
    expected["receipt_sha256"] = _self_hash(expected, "receipt_sha256")
    if candidate != expected:
        raise ValueError(
            "Minimum Event Evidence Card v1.1 receipt does not exactly replay"
        )
    return candidate
