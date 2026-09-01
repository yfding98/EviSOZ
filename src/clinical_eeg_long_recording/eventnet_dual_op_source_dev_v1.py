"""Prediction-first EventNet decoder-grid bridge to the v1.2 dual-OP scorer.

The public workflow has two irreversible stages.  Stage 1 accepts only an
already validated, reference-free EventNet decoder grid, converts every
merged alarm under every policy into the provider-neutral dual operating-point
schema, and content-binds that complete cross product.  Stage 2 first replays
that exact freeze and only then derives and opens public TUSZ ``dev/*.csv_bi``
references containing global ``TERM,seiz`` intervals.

No channel annotation, EDF annotation stream, spreadsheet, physician label,
clinical text, private datum or source-evaluation reference has an API slot.
The EventNet bridge consumes no channel identity at all and therefore cannot
split a whole bipolar lead into electrode endpoints.  Its alarm-start anchor
is a detector candidate anchor, never a qualified clinical onset.

The query interval is an explicitly limited cost proxy: the nominal
``[-12,+48]`` initial watchdog around the detector anchor, widened only when
needed to contain the complete merged-alarm search envelope.  It is not the
real cost of v1.2 adaptive outer acquisition, so this bridge cannot grant a
Navigation qualification or promote a provider.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from .continuous_detection_source_dev_join import (
    parse_tusz_term_seiz_reference_bytes,
    read_source_dev_reference_bytes,
    source_dev_reference_relative_path,
)
from .detector_dual_operating_point_v1 import (
    score_detector_dual_op_v1,
    validate_detector_dual_op_diagnostic_v1,
    validate_detector_prediction_inventory_v1,
)
from .eventnet_full_record_adapter import EVENTNET_PROVIDER_ID
from .eventnet_native_decoder_grid import (
    ValidatedEventNetDecoderGridBundle,
    revalidate_eventnet_native_decoder_grid_bundle_without_references,
)


EVENTNET_DUAL_OP_FREEZE_SCHEMA_VERSION = (
    "eventnet_decoder_grid_dual_op_prediction_freeze_v1"
)
EVENTNET_DUAL_OP_FREEZE_METHOD_ID = (
    "eventnet_merged_alarm_to_complete_reference_free_dual_op_inventory_v1"
)
EVENTNET_DUAL_OP_REFERENCE_JOIN_SCHEMA_VERSION = (
    "eventnet_dual_op_source_dev_reference_join_v1"
)
EVENTNET_DUAL_OP_CALIBRATION_SCHEMA_VERSION = (
    "eventnet_dual_op_source_dev_calibration_wrapper_v1"
)
EVENTNET_DUAL_OP_CALIBRATION_METHOD_ID = (
    "postfreeze_exact_term_seiz_dual_operating_point_diagnostic_v1"
)

EVENTNET_DUAL_OP_INITIAL_WATCHDOG_LEFT_SECONDS = 12.0
EVENTNET_DUAL_OP_INITIAL_WATCHDOG_RIGHT_SECONDS = 48.0

EVENTNET_DUAL_OP_FREEZE_FILENAME = "dual_op_prediction_freeze_receipt.json"
EVENTNET_DUAL_OP_REFERENCE_JOIN_FILENAME = "dual_op_reference_join_receipt.json"
EVENTNET_DUAL_OP_CALIBRATION_FILENAME = "dual_op_calibration_receipt.json"

_REFERENCE_ACCESS_NONE = {
    "reference_path_argument_accepted": False,
    "reference_paths_derived": 0,
    "reference_files_opened": 0,
    "edf_annotations_opened": 0,
    "excel_files_opened": 0,
    "physician_or_clinical_text_opened": 0,
    "private_data_opened": 0,
    "source_eval_opened": 0,
}
_FREEZE_SCOPE = {
    "complete_eventnet_record_policy_cross_product_required": True,
    "reference_free_grid_revalidated_before_adapter_mapping": True,
    "global_or_channel_reference_used": False,
    "edf_annotations_used": False,
    "excel_or_physician_labels_used": False,
    "clinical_text_used": False,
    "private_data_used": False,
    "source_eval_used": False,
    "channel_or_lead_payload_consumed": False,
    "whole_bipolar_lead_endpoint_split_allowed": False,
    "provider_promotion_authorized": False,
}
_CANDIDATE_MAPPING_CONTRACT = {
    "source_unit": "one_provider_native_merged_alarm",
    "event_interval": "merged_alarm_start_stop_seconds",
    "anchor": "merged_alarm_start_seconds_detector_candidate_not_clinical_onset",
    "ranking_score": "merged_alarm_maximum_center_probability",
    "search_envelope": "complete_merged_alarm_interval",
    "nominal_initial_watchdog_relative_to_anchor_seconds": [-12.0, 48.0],
    "query_interval": (
        "record_clipped_union_of_nominal_initial_watchdog_and_complete_"
        "merged_alarm_search_envelope"
    ),
    "query_cost_semantics": (
        "initial_watchdog_proxy_not_v1_2_adaptive_outer_acquisition_cost"
    ),
    "alarm_start_is_qualified_clinical_onset": False,
    "candidate_is_confirmed_seizure": False,
    "channel_or_lead_transformation": "none",
    "whole_bipolar_lead_endpoint_split_allowed": False,
}
_REFERENCE_JOIN_SCOPE = {
    "prediction_freeze_revalidated_before_reference_path_derivation": True,
    "source_dev_global_term_seiz_intervals_only": True,
    "channel_annotations_used": False,
    "edf_annotations_used": False,
    "excel_or_physician_labels_used": False,
    "clinical_text_used": False,
    "private_data_used": False,
    "source_eval_used": False,
    "detector_or_decoder_rerun": False,
    "frozen_predictions_mutated": False,
}
_CALIBRATION_LIMITATIONS = {
    "alarm_start_anchor_is_qualified_clinical_onset": False,
    "query_cost_is_v1_2_adaptive_outer_acquisition_cost": False,
    "query_cost_is_initial_watchdog_proxy": True,
    "navigation_query_cost_qualification_eligible": False,
    "whole_bipolar_lead_endpoint_split_performed": False,
    "findings_or_soz_evidence_authorized": False,
    "provider_promotion_authorized": False,
    "clinical_or_production_permission": False,
    "sota_claim_authorized": False,
}


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eventnet_dual_op_adapter_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    if len(value) > 1024 or any(ord(character) < 32 for character in value):
        raise ValueError(f"{context} is invalid")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TypeError(f"{context} must be a positive integer")
    return value


def _safe_source_dev_recording_identity_without_reference_derivation(
    recording_id: str,
) -> None:
    """Validate only the frozen identity; do not construct a reference path."""

    identifier = _identifier(recording_id, "EventNet recording ID")
    if "\\" in identifier:
        raise ValueError("EventNet source-dev recording ID must use POSIX separators")
    relative = PurePosixPath(identifier)
    lowered = tuple(part.lower() for part in relative.parts)
    if (
        relative.is_absolute()
        or len(relative.parts) < 2
        or relative.parts[0] != "dev"
        or ".." in relative.parts
        or "." in relative.parts
        or relative.suffix.lower() != ".edf"
        or any(
            part in {"train", "eval", "source_eval", "private", "private_inference"}
            for part in lowered[1:]
        )
    ):
        raise ValueError("EventNet recording is not a safe source_dev EDF identity")


def _candidate_from_merged_alarm(
    alarm: Mapping[str, Any],
    *,
    duration_seconds: float,
) -> dict[str, Any]:
    alarm_id = _identifier(alarm.get("alarm_id"), "EventNet merged alarm ID")
    start = float(alarm["start_offset_seconds"])
    stop = float(alarm["stop_offset_seconds"])
    score = float(alarm["maximum_center_probability"])
    if (
        not all(math.isfinite(value) for value in (start, stop, score))
        or start < 0.0
        or stop <= start
        or stop > duration_seconds + 1e-9
        or not 0.0 <= score <= 1.0
    ):
        raise ValueError("EventNet merged alarm cannot enter the dual-OP adapter")
    # EventNet's native decoder exposes no qualified onset boundary.  Its
    # earliest merged-alarm support is kept only as a detector candidate anchor.
    anchor = start
    watchdog_start = max(0.0, anchor - EVENTNET_DUAL_OP_INITIAL_WATCHDOG_LEFT_SECONDS)
    watchdog_stop = min(
        duration_seconds, anchor + EVENTNET_DUAL_OP_INITIAL_WATCHDOG_RIGHT_SECONDS
    )
    query_start = min(watchdog_start, start)
    query_stop = max(watchdog_stop, stop)
    if query_stop <= query_start:
        raise ValueError("EventNet dual-OP query support is empty")
    return {
        "candidate_id": alarm_id,
        "start_seconds": start,
        "stop_seconds": stop,
        "anchor_seconds": anchor,
        "ranking_score": score,
        "search_envelope_start_seconds": start,
        "search_envelope_stop_seconds": stop,
        "query_start_seconds": query_start,
        "query_stop_seconds": query_stop,
    }


@dataclass(frozen=True, slots=True)
class FrozenEventNetDualOpPredictionInventoryV1:
    """Immutable Stage-1 carrier; references cannot enter its constructor API."""

    prediction_inventory_json: str
    freeze_receipt_json: str

    def prediction_inventory(self) -> dict[str, Any]:
        value = json.loads(self.prediction_inventory_json)
        if type(value) is not dict:
            raise RuntimeError("sealed dual-OP prediction inventory is corrupted")
        return validate_detector_prediction_inventory_v1(value)

    def freeze_receipt(self) -> dict[str, Any]:
        value = json.loads(self.freeze_receipt_json)
        if type(value) is not dict:
            raise RuntimeError("sealed EventNet dual-OP freeze receipt is corrupted")
        return value


def validate_eventnet_dual_op_prediction_freeze_receipt_v1(
    value: object,
    *,
    prediction_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "method_id",
        "provider_id",
        "split",
        "adapter_code_sha256",
        "upstream_grid_bundle_id",
        "upstream_grid_bundle_receipt_sha256",
        "upstream_grid_definition_sha256",
        "upstream_prediction_row_roster_sha256",
        "prediction_inventory_receipt_sha256",
        "prediction_row_roster_sha256",
        "recording_count",
        "patient_count",
        "policy_count",
        "prediction_row_count",
        "candidate_count",
        "candidate_mapping_contract",
        "scope_receipt",
        "reference_access",
        "prediction_inventory_materialized_before_reference_access",
        "navigation_query_cost_qualification_eligible",
        "qualification_granted",
        "provider_promotion_authorized",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("EventNet dual-OP freeze receipt fields drifted")
    data = deepcopy(value)
    inventory = validate_detector_prediction_inventory_v1(dict(prediction_inventory))
    if (
        data["schema_version"] != EVENTNET_DUAL_OP_FREEZE_SCHEMA_VERSION
        or data["method_id"] != EVENTNET_DUAL_OP_FREEZE_METHOD_ID
        or data["provider_id"] != EVENTNET_PROVIDER_ID
        or data["split"] != "source_dev"
        or data["candidate_mapping_contract"] != _CANDIDATE_MAPPING_CONTRACT
        or data["scope_receipt"] != _FREEZE_SCOPE
        or data["reference_access"] != _REFERENCE_ACCESS_NONE
        or data["prediction_inventory_materialized_before_reference_access"] is not True
        or data["navigation_query_cost_qualification_eligible"] is not False
        or data["qualification_granted"] is not False
        or data["provider_promotion_authorized"] is not False
    ):
        raise ValueError("EventNet dual-OP freeze identity or firewall drifted")
    for field in (
        "adapter_code_sha256",
        "upstream_grid_bundle_receipt_sha256",
        "upstream_grid_definition_sha256",
        "upstream_prediction_row_roster_sha256",
        "prediction_inventory_receipt_sha256",
        "prediction_row_roster_sha256",
        "receipt_sha256",
    ):
        _sha256(data[field], f"EventNet dual-OP freeze {field}")
    _identifier(data["upstream_grid_bundle_id"], "upstream grid bundle ID")
    for field in (
        "recording_count",
        "patient_count",
        "policy_count",
        "prediction_row_count",
    ):
        _positive_integer(data[field], f"EventNet dual-OP freeze {field}")
    if (
        isinstance(data["candidate_count"], bool)
        or not isinstance(data["candidate_count"], int)
        or data["candidate_count"] < 0
    ):
        raise TypeError("EventNet dual-OP candidate count must be non-negative")
    rows = inventory["prediction_rows"]
    expected = {
        "provider_id": inventory["provider_id"],
        "prediction_inventory_receipt_sha256": inventory["receipt_sha256"],
        "prediction_row_roster_sha256": inventory["prediction_row_roster_sha256"],
        "recording_count": len(inventory["expected_recording_ids"]),
        "patient_count": len({str(row["patient_id"]) for row in rows}),
        "policy_count": len(inventory["expected_policy_ids"]),
        "prediction_row_count": len(rows),
        "candidate_count": sum(len(row["candidates"]) for row in rows),
    }
    for field, expected_value in expected.items():
        if data[field] != expected_value:
            raise ValueError(
                f"EventNet dual-OP freeze {field} disagrees with inventory"
            )
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("EventNet dual-OP freeze receipt hash drifted")
    return data


def freeze_eventnet_decoder_grid_for_dual_op_v1(
    grid_bundle: ValidatedEventNetDecoderGridBundle,
) -> FrozenEventNetDualOpPredictionInventoryV1:
    """Map and seal a complete decoder grid without accepting a reference."""

    frozen_grid = revalidate_eventnet_native_decoder_grid_bundle_without_references(
        grid_bundle
    )
    grid_receipt = frozen_grid.bundle_receipt()
    grid_definition = frozen_grid.grid_definition()
    rows: list[dict[str, Any]] = []
    upstream_row_roster: list[list[str]] = []
    for prediction in frozen_grid.predictions:
        _safe_source_dev_recording_identity_without_reference_derivation(
            prediction.recording_id
        )
        alarms = prediction.merged_alarms()
        candidates = [
            _candidate_from_merged_alarm(
                alarm,
                duration_seconds=float(prediction.recording_duration_seconds),
            )
            for alarm in alarms
        ]
        rows.append(
            {
                "provider_id": EVENTNET_PROVIDER_ID,
                "patient_id": prediction.patient_alias,
                "recording_id": prediction.recording_id,
                "split": "source_dev",
                "duration_seconds": float(prediction.recording_duration_seconds),
                "policy_id": prediction.policy_id,
                "processing_status": "completed",
                "modeled_duration_seconds": float(
                    prediction.recording_duration_seconds
                ),
                "failure_code": None,
                "candidates": candidates,
            }
        )
        upstream_row_roster.append(
            [
                prediction.recording_id,
                prediction.policy_id,
                prediction.prediction_row_receipt_sha256,
            ]
        )
    recording_ids = sorted({str(row["recording_id"]) for row in rows})
    policy_ids = sorted({str(row["policy_id"]) for row in rows})
    # Imported lazily only for construction so validation remains delegated to
    # the provider-neutral schema implementation.
    from .detector_dual_operating_point_v1 import (  # noqa: PLC0415
        freeze_detector_prediction_inventory_v1,
    )

    inventory = freeze_detector_prediction_inventory_v1(
        provider_id=EVENTNET_PROVIDER_ID,
        rows=rows,
        expected_recording_ids=recording_ids,
        expected_policy_ids=policy_ids,
    )
    body: dict[str, Any] = {
        "schema_version": EVENTNET_DUAL_OP_FREEZE_SCHEMA_VERSION,
        "method_id": EVENTNET_DUAL_OP_FREEZE_METHOD_ID,
        "provider_id": EVENTNET_PROVIDER_ID,
        "split": "source_dev",
        "adapter_code_sha256": eventnet_dual_op_adapter_code_sha256(),
        "upstream_grid_bundle_id": grid_receipt["bundle_id"],
        "upstream_grid_bundle_receipt_sha256": grid_receipt["receipt_sha256"],
        "upstream_grid_definition_sha256": grid_receipt.get(
            "grid_definition_sha256", _canonical_sha256(grid_definition)
        ),
        "upstream_prediction_row_roster_sha256": _canonical_sha256(upstream_row_roster),
        "prediction_inventory_receipt_sha256": inventory["receipt_sha256"],
        "prediction_row_roster_sha256": inventory["prediction_row_roster_sha256"],
        "recording_count": len(recording_ids),
        "patient_count": len({str(row["patient_id"]) for row in rows}),
        "policy_count": len(policy_ids),
        "prediction_row_count": len(rows),
        "candidate_count": sum(len(row["candidates"]) for row in rows),
        "candidate_mapping_contract": deepcopy(_CANDIDATE_MAPPING_CONTRACT),
        "scope_receipt": deepcopy(_FREEZE_SCOPE),
        "reference_access": deepcopy(_REFERENCE_ACCESS_NONE),
        "prediction_inventory_materialized_before_reference_access": True,
        "navigation_query_cost_qualification_eligible": False,
        "qualification_granted": False,
        "provider_promotion_authorized": False,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    receipt = validate_eventnet_dual_op_prediction_freeze_receipt_v1(
        body,
        prediction_inventory=inventory,
    )
    return FrozenEventNetDualOpPredictionInventoryV1(
        prediction_inventory_json=_canonical_json(inventory),
        freeze_receipt_json=_canonical_json(receipt),
    )


def revalidate_frozen_eventnet_dual_op_prediction_inventory_v1(
    value: object,
) -> FrozenEventNetDualOpPredictionInventoryV1:
    frozen, _, _ = _validated_frozen_components(value)
    return frozen


def _validated_frozen_components(
    value: object,
) -> tuple[FrozenEventNetDualOpPredictionInventoryV1, dict[str, Any], dict[str, Any],]:
    if type(value) is not FrozenEventNetDualOpPredictionInventoryV1:
        raise TypeError("source-dev join requires the exact sealed EventNet freeze")
    inventory = value.prediction_inventory()
    receipt = validate_eventnet_dual_op_prediction_freeze_receipt_v1(
        value.freeze_receipt(),
        prediction_inventory=inventory,
    )
    return value, inventory, receipt


def write_eventnet_dual_op_prediction_freeze_append_only(
    value: FrozenEventNetDualOpPredictionInventoryV1,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Persist the small Stage-1 receipt before any reference is opened."""

    _, _, receipt = _validated_frozen_components(value)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("EventNet dual-OP output must be a new directory")
    output.mkdir(parents=True, exist_ok=False)
    path = output / EVENTNET_DUAL_OP_FREEZE_FILENAME
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
        handle.write("\n")
    return {
        "filename": EVENTNET_DUAL_OP_FREEZE_FILENAME,
        "file_sha256": _file_sha256(path),
        "freeze_receipt_sha256": receipt["receipt_sha256"],
        "reference_files_opened_before_write": 0,
    }


def _build_reference_join(
    frozen: FrozenEventNetDualOpPredictionInventoryV1,
    *,
    source_dev_reference_root: str | Path,
    reference_reader: Callable[[Path, PurePosixPath], bytes],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any],]:
    # Security ordering: this complete replay precedes even reference path
    # derivation or root resolution.
    _, inventory, freeze_receipt = _validated_frozen_components(frozen)
    rows = inventory["prediction_rows"]
    metadata: dict[str, tuple[str, float]] = {}
    for row in rows:
        recording_id = str(row["recording_id"])
        prior = metadata.setdefault(
            recording_id,
            (str(row["patient_id"]), float(row["duration_seconds"])),
        )
        if prior != (str(row["patient_id"]), float(row["duration_seconds"])):
            raise ValueError("dual-OP prediction metadata differs across policies")

    root = Path(source_dev_reference_root)
    references: dict[str, dict[str, Any]] = {}
    file_inventory: list[dict[str, Any]] = []
    event_inventory: list[list[Any]] = []
    ignored_count = 0
    for recording_id in sorted(metadata):
        patient_id, duration = metadata[recording_id]
        relative = source_dev_reference_relative_path(recording_id)
        payload = reference_reader(root, relative)
        parsed = parse_tusz_term_seiz_reference_bytes(
            payload,
            duration_seconds=duration,
        )
        events = parsed.events()
        references[recording_id] = {
            "patient_id": patient_id,
            "recording_id": recording_id,
            "split": "source_dev",
            "duration_seconds": duration,
            "reference_events": events,
        }
        file_inventory.append(
            {
                "recording_id": recording_id,
                "reference_relative_path": str(relative),
                "reference_file_sha256": parsed.reference_file_sha256,
                "selected_term_seiz_row_count": parsed.selected_term_seiz_row_count,
                "ignored_non_term_seiz_row_count": (
                    parsed.ignored_non_term_seiz_row_count
                ),
            }
        )
        ignored_count += parsed.ignored_non_term_seiz_row_count
        for event in events:
            event_inventory.append(
                [
                    recording_id,
                    float(event["start_seconds"]),
                    float(event["stop_seconds"]),
                ]
            )
    joined = [
        {
            # The provider-neutral scorer deep-copies during validation.  Keep
            # shared immutable inputs here to avoid a third full candidate
            # inventory in memory on the 65,952-row official-dev grid.
            "prediction_row": row,
            "reference_row": references[str(row["recording_id"])],
        }
        for row in rows
    ]
    join_body: dict[str, Any] = {
        "schema_version": EVENTNET_DUAL_OP_REFERENCE_JOIN_SCHEMA_VERSION,
        "provider_id": EVENTNET_PROVIDER_ID,
        "split": "source_dev",
        "prediction_freeze_receipt_sha256": freeze_receipt["receipt_sha256"],
        "prediction_inventory_receipt_sha256": inventory["receipt_sha256"],
        "prediction_row_roster_sha256": inventory["prediction_row_roster_sha256"],
        "reference_file_inventory_sha256": _canonical_sha256(file_inventory),
        "reference_event_inventory_sha256": _canonical_sha256(event_inventory),
        "recording_count": len(metadata),
        "patient_count": len({patient for patient, _ in metadata.values()}),
        "policy_count": len(inventory["expected_policy_ids"]),
        "joined_row_count": len(joined),
        "reference_files_opened": len(file_inventory),
        "selected_term_seiz_event_count": len(event_inventory),
        "ignored_non_term_seiz_row_count": ignored_count,
        "seizure_free_recording_count": sum(
            not reference["reference_events"] for reference in references.values()
        ),
        "prediction_freeze_validated_before_first_reference_path_derivation": True,
        "prediction_freeze_validated_before_first_reference_open": True,
        "joined_predictions_exactly_replayed_from_freeze": True,
        "scope_receipt": deepcopy(_REFERENCE_JOIN_SCOPE),
        "qualification_granted": False,
        "provider_promotion_authorized": False,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    join_body["receipt_sha256"] = _canonical_sha256(join_body)
    return (
        inventory,
        freeze_receipt,
        joined,
        validate_eventnet_dual_op_reference_join_receipt_v1(
            join_body,
            prediction_inventory=inventory,
        ),
    )


def validate_eventnet_dual_op_reference_join_receipt_v1(
    value: object,
    *,
    prediction_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "provider_id",
        "split",
        "prediction_freeze_receipt_sha256",
        "prediction_inventory_receipt_sha256",
        "prediction_row_roster_sha256",
        "reference_file_inventory_sha256",
        "reference_event_inventory_sha256",
        "recording_count",
        "patient_count",
        "policy_count",
        "joined_row_count",
        "reference_files_opened",
        "selected_term_seiz_event_count",
        "ignored_non_term_seiz_row_count",
        "seizure_free_recording_count",
        "prediction_freeze_validated_before_first_reference_path_derivation",
        "prediction_freeze_validated_before_first_reference_open",
        "joined_predictions_exactly_replayed_from_freeze",
        "scope_receipt",
        "qualification_granted",
        "provider_promotion_authorized",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("EventNet dual-OP reference-join receipt fields drifted")
    data = deepcopy(value)
    inventory = validate_detector_prediction_inventory_v1(dict(prediction_inventory))
    if (
        data["schema_version"] != EVENTNET_DUAL_OP_REFERENCE_JOIN_SCHEMA_VERSION
        or data["provider_id"] != EVENTNET_PROVIDER_ID
        or data["split"] != "source_dev"
        or data["prediction_inventory_receipt_sha256"] != inventory["receipt_sha256"]
        or data["prediction_row_roster_sha256"]
        != inventory["prediction_row_roster_sha256"]
        or data["scope_receipt"] != _REFERENCE_JOIN_SCOPE
        or data["prediction_freeze_validated_before_first_reference_path_derivation"]
        is not True
        or data["prediction_freeze_validated_before_first_reference_open"] is not True
        or data["joined_predictions_exactly_replayed_from_freeze"] is not True
        or data["qualification_granted"] is not False
        or data["provider_promotion_authorized"] is not False
    ):
        raise ValueError("EventNet dual-OP reference join identity drifted")
    for field in (
        "prediction_freeze_receipt_sha256",
        "prediction_inventory_receipt_sha256",
        "prediction_row_roster_sha256",
        "reference_file_inventory_sha256",
        "reference_event_inventory_sha256",
        "receipt_sha256",
    ):
        _sha256(data[field], f"EventNet dual-OP join {field}")
    for field in (
        "recording_count",
        "patient_count",
        "policy_count",
        "joined_row_count",
        "reference_files_opened",
    ):
        _positive_integer(data[field], f"EventNet dual-OP join {field}")
    for field in (
        "selected_term_seiz_event_count",
        "ignored_non_term_seiz_row_count",
        "seizure_free_recording_count",
    ):
        if (
            isinstance(data[field], bool)
            or not isinstance(data[field], int)
            or data[field] < 0
        ):
            raise TypeError(f"EventNet dual-OP join {field} must be non-negative")
    if (
        data["recording_count"] != len(inventory["expected_recording_ids"])
        or data["patient_count"]
        != len({str(row["patient_id"]) for row in inventory["prediction_rows"]})
        or data["policy_count"] != len(inventory["expected_policy_ids"])
        or data["joined_row_count"] != len(inventory["prediction_rows"])
        or data["reference_files_opened"] != data["recording_count"]
        or data["seizure_free_recording_count"] > data["recording_count"]
    ):
        raise ValueError("EventNet dual-OP reference join cardinality drifted")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("EventNet dual-OP reference join receipt hash drifted")
    return data


@dataclass(frozen=True, slots=True)
class EventNetDualOpSourceDevCalibrationV1:
    calibration_receipt_json: str
    reference_join_receipt_json: str

    def calibration_receipt(self) -> dict[str, Any]:
        value = json.loads(self.calibration_receipt_json)
        if type(value) is not dict:
            raise RuntimeError("sealed EventNet dual-OP calibration is corrupted")
        return value

    def reference_join_receipt(self) -> dict[str, Any]:
        value = json.loads(self.reference_join_receipt_json)
        if type(value) is not dict:
            raise RuntimeError("sealed EventNet dual-OP join receipt is corrupted")
        return value


def validate_eventnet_dual_op_calibration_receipt_v1(value: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "method_id",
        "provider_id",
        "split",
        "prediction_freeze_receipt_sha256",
        "reference_join_receipt_sha256",
        "dual_op_diagnostic",
        "technical_coverage_qualification_eligible",
        "navigation_query_cost_qualification_eligible",
        "limitations",
        "qualification_granted",
        "descriptive_research_primary_selected",
        "provider_promotion_authorized",
        "clinical_or_production_permission",
        "sota_claim_authorized",
        "status",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("EventNet dual-OP calibration wrapper fields drifted")
    data = deepcopy(value)
    diagnostic = validate_detector_dual_op_diagnostic_v1(data["dual_op_diagnostic"])
    if (
        data["schema_version"] != EVENTNET_DUAL_OP_CALIBRATION_SCHEMA_VERSION
        or data["method_id"] != EVENTNET_DUAL_OP_CALIBRATION_METHOD_ID
        or data["provider_id"] != EVENTNET_PROVIDER_ID
        or data["split"] != "source_dev"
        or data["technical_coverage_qualification_eligible"]
        is not bool(diagnostic["qualification_eligible"])
        or data["navigation_query_cost_qualification_eligible"] is not False
        or data["limitations"] != _CALIBRATION_LIMITATIONS
        or data["qualification_granted"] is not False
        or data["descriptive_research_primary_selected"] is not False
        or data["provider_promotion_authorized"] is not False
        or data["clinical_or_production_permission"] is not False
        or data["sota_claim_authorized"] is not False
        or data["status"]
        != "source_dev_dual_op_diagnostic_only_navigation_cost_proxy_unqualified"
    ):
        raise ValueError("EventNet dual-OP calibration permissions drifted")
    for field in (
        "prediction_freeze_receipt_sha256",
        "reference_join_receipt_sha256",
        "receipt_sha256",
    ):
        _sha256(data[field], f"EventNet dual-OP calibration {field}")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("EventNet dual-OP calibration receipt hash drifted")
    return data


def calibrate_eventnet_decoder_grid_dual_op_source_dev_v1(
    frozen_prediction_inventory: FrozenEventNetDualOpPredictionInventoryV1,
    *,
    source_dev_reference_root: str | Path,
    reference_reader: Callable[[Path, PurePosixPath], bytes] = (
        read_source_dev_reference_bytes
    ),
) -> EventNetDualOpSourceDevCalibrationV1:
    """Open exact public dev truth only after the adapter freeze replays."""

    inventory, freeze_receipt, joined, join_receipt = _build_reference_join(
        frozen_prediction_inventory,
        source_dev_reference_root=source_dev_reference_root,
        reference_reader=reference_reader,
    )
    diagnostic = score_detector_dual_op_v1(
        frozen_prediction_inventory=inventory,
        joined_rows=joined,
    )
    body: dict[str, Any] = {
        "schema_version": EVENTNET_DUAL_OP_CALIBRATION_SCHEMA_VERSION,
        "method_id": EVENTNET_DUAL_OP_CALIBRATION_METHOD_ID,
        "provider_id": EVENTNET_PROVIDER_ID,
        "split": "source_dev",
        "prediction_freeze_receipt_sha256": freeze_receipt["receipt_sha256"],
        "reference_join_receipt_sha256": join_receipt["receipt_sha256"],
        "dual_op_diagnostic": diagnostic,
        "technical_coverage_qualification_eligible": bool(
            diagnostic["qualification_eligible"]
        ),
        "navigation_query_cost_qualification_eligible": False,
        "limitations": deepcopy(_CALIBRATION_LIMITATIONS),
        "qualification_granted": False,
        "descriptive_research_primary_selected": False,
        "provider_promotion_authorized": False,
        "clinical_or_production_permission": False,
        "sota_claim_authorized": False,
        "status": (
            "source_dev_dual_op_diagnostic_only_navigation_cost_proxy_unqualified"
        ),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    calibration = validate_eventnet_dual_op_calibration_receipt_v1(body)
    return EventNetDualOpSourceDevCalibrationV1(
        calibration_receipt_json=_canonical_json(calibration),
        reference_join_receipt_json=_canonical_json(join_receipt),
    )


def write_eventnet_dual_op_calibration_append_only(
    value: EventNetDualOpSourceDevCalibrationV1,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Append Stage-2 receipts to a directory containing only Stage 1."""

    if type(value) is not EventNetDualOpSourceDevCalibrationV1:
        raise TypeError("EventNet dual-OP writer requires the sealed result type")
    calibration = validate_eventnet_dual_op_calibration_receipt_v1(
        value.calibration_receipt()
    )
    join = value.reference_join_receipt()
    _sha256(join.get("receipt_sha256"), "reference join receipt")
    if calibration["reference_join_receipt_sha256"] != join["receipt_sha256"]:
        raise ValueError("EventNet calibration and reference join receipts disagree")
    output = Path(output_directory)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("EventNet dual-OP output must be a regular directory")
    if {path.name for path in output.iterdir()} != {EVENTNET_DUAL_OP_FREEZE_FILENAME}:
        raise FileExistsError(
            "EventNet dual-OP Stage-2 output is not an untouched Stage-1 directory"
        )
    join_path = output / EVENTNET_DUAL_OP_REFERENCE_JOIN_FILENAME
    calibration_path = output / EVENTNET_DUAL_OP_CALIBRATION_FILENAME
    with join_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(join, ensure_ascii=False, sort_keys=True, indent=2))
        handle.write("\n")
    with calibration_path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(calibration, ensure_ascii=False, sort_keys=True, indent=2)
        )
        handle.write("\n")
    return {
        "reference_join_filename": EVENTNET_DUAL_OP_REFERENCE_JOIN_FILENAME,
        "reference_join_file_sha256": _file_sha256(join_path),
        "calibration_filename": EVENTNET_DUAL_OP_CALIBRATION_FILENAME,
        "calibration_file_sha256": _file_sha256(calibration_path),
        "qualification_granted": False,
        "provider_promotion_authorized": False,
    }


__all__ = [
    "EVENTNET_DUAL_OP_CALIBRATION_FILENAME",
    "EVENTNET_DUAL_OP_CALIBRATION_METHOD_ID",
    "EVENTNET_DUAL_OP_CALIBRATION_SCHEMA_VERSION",
    "EVENTNET_DUAL_OP_FREEZE_FILENAME",
    "EVENTNET_DUAL_OP_FREEZE_METHOD_ID",
    "EVENTNET_DUAL_OP_FREEZE_SCHEMA_VERSION",
    "EVENTNET_DUAL_OP_INITIAL_WATCHDOG_LEFT_SECONDS",
    "EVENTNET_DUAL_OP_INITIAL_WATCHDOG_RIGHT_SECONDS",
    "EVENTNET_DUAL_OP_REFERENCE_JOIN_FILENAME",
    "EVENTNET_DUAL_OP_REFERENCE_JOIN_SCHEMA_VERSION",
    "EventNetDualOpSourceDevCalibrationV1",
    "FrozenEventNetDualOpPredictionInventoryV1",
    "calibrate_eventnet_decoder_grid_dual_op_source_dev_v1",
    "eventnet_dual_op_adapter_code_sha256",
    "freeze_eventnet_decoder_grid_for_dual_op_v1",
    "revalidate_frozen_eventnet_dual_op_prediction_inventory_v1",
    "validate_eventnet_dual_op_calibration_receipt_v1",
    "validate_eventnet_dual_op_prediction_freeze_receipt_v1",
    "validate_eventnet_dual_op_reference_join_receipt_v1",
    "write_eventnet_dual_op_calibration_append_only",
    "write_eventnet_dual_op_prediction_freeze_append_only",
]
