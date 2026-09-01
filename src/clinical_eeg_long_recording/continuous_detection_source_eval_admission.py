"""Fail-closed admission for one-shot continuous-detector source evaluation.

The source-evaluation scorer must not trust a request-layer provider name,
operating-point identifier, or ``frozen=True`` boolean.  This module binds an
already materialized prediction inventory to a replayed source-development
calibration receipt, the exact patient/recording split roster, a complete
ledger of prior source-evaluation access, and one decoder receipt per expected
recording.  Reference seizure intervals are deliberately absent from the
artifact.

The artifact is research infrastructure, not an independent production trust
root.  It prevents accidental or request-layer bypass of the local lockbox;
production promotion remains governed by the separate provider authority.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .continuous_detection import CONTINUOUS_DETECTION_METHOD_ID
from .continuous_detection_roster import (
    validate_continuous_detector_split_roster,
)
from .detector_provider_contract import validate_provider_definition


SOURCE_EVAL_LOCKBOX_LEDGER_SCHEMA_VERSION = (
    "continuous_detection_source_eval_lockbox_access_ledger_v1"
)
SOURCE_EVAL_LOCKBOX_LEDGER_METHOD_ID = "complete_known_pre_scoring_access_inventory_v1"
SOURCE_EVAL_ADMISSION_SCHEMA_VERSION = (
    "continuous_detection_source_eval_admission_v1"
)
SOURCE_EVAL_ADMISSION_METHOD_ID = (
    "replayed_calibration_decoder_prediction_roster_lockbox_closure_v1"
)

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_LOCKBOX_ACCESS_KINDS = frozenset(
    {
        "signal_engineering_smoke",
        "posterior_engineering_smoke",
        "decoder_engineering_smoke",
        "reference_scoring_or_inspection",
        "other_source_eval_access",
    }
)
_LOCKBOX_DISPOSITION = "exclude_patient_from_untouched_source_eval"
_COMPLETED_OUTCOMES = frozenset(
    {"completed_with_alarms", "completed_zero_alarm"}
)


def _validate_calibration_receipt(payload: object) -> dict[str, Any]:
    # Calibration imports the benchmark metric implementation.  Importing it
    # lazily keeps the benchmark -> admission edge acyclic during collection.
    from .continuous_detection_calibration import (
        validate_continuous_detection_calibration_receipt,
    )

    return validate_continuous_detection_calibration_receipt(payload)


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
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _normalize_events(value: object, *, context: str) -> list[dict[str, float]]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be an array")
    events: list[dict[str, float]] = []
    previous_stop = 0.0
    for index, raw in enumerate(value):
        if type(raw) is not dict or set(raw) != {"start_seconds", "stop_seconds"}:
            raise ValueError(f"{context}[{index}] has invalid fields")
        start = _finite(raw["start_seconds"], f"{context}[{index}] start")
        stop = _finite(raw["stop_seconds"], f"{context}[{index}] stop")
        if start < 0 or stop <= start:
            raise ValueError(f"{context}[{index}] is invalid")
        if index and start < previous_stop - 1e-9:
            raise ValueError(f"{context} must be sorted and non-overlapping")
        events.append({"start_seconds": start, "stop_seconds": stop})
        previous_stop = stop
    return events


def normalize_source_eval_prediction_inventory(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the reference-free prediction projection used by admission."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise TypeError("source-eval prediction rows must be a non-empty sequence")
    inventory: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"source-eval prediction row {index} must be an object")
        recording_id = _identifier(
            raw.get("recording_id"), f"source-eval prediction row {index} recording"
        )
        if recording_id in seen:
            raise ValueError("source-eval prediction recording IDs must be unique")
        seen.add(recording_id)
        if "predicted_events" not in raw:
            raise ValueError("source-eval prediction row lacks predicted_events")
        inventory.append(
            {
                "recording_id": recording_id,
                "predicted_events": _normalize_events(
                    raw["predicted_events"],
                    context=f"source-eval prediction row {index} events",
                ),
            }
        )
    inventory.sort(key=lambda row: row["recording_id"])
    return inventory


def build_source_eval_lockbox_access_ledger(
    *,
    dataset_id: str,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a content-bound identity-only ledger of all known prior access."""

    dataset = _identifier(dataset_id, "lockbox dataset_id")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TypeError("lockbox entries must be a sequence")
    normalized: list[dict[str, Any]] = []
    access_ids: set[str] = set()
    for index, raw in enumerate(entries):
        required = {
            "access_id",
            "patient_id",
            "recording_id",
            "access_kind",
            "selection_or_threshold_tuning_used",
        }
        if type(raw) is not dict or set(raw) != required:
            raise ValueError(f"lockbox entry {index} has missing or unknown fields")
        access_id = _identifier(raw["access_id"], f"lockbox entry {index} access_id")
        if access_id in access_ids:
            raise ValueError("lockbox access IDs must be unique")
        access_ids.add(access_id)
        access_kind = _identifier(
            raw["access_kind"], f"lockbox entry {index} access_kind"
        )
        if access_kind not in _LOCKBOX_ACCESS_KINDS:
            raise ValueError("lockbox access kind is unsupported")
        if type(raw["selection_or_threshold_tuning_used"]) is not bool:
            raise TypeError("lockbox tuning-use flag must be boolean")
        normalized.append(
            {
                "access_id": access_id,
                "patient_id": _identifier(
                    raw["patient_id"], f"lockbox entry {index} patient_id"
                ),
                "recording_id": _identifier(
                    raw["recording_id"], f"lockbox entry {index} recording_id"
                ),
                "access_kind": access_kind,
                "selection_or_threshold_tuning_used": raw[
                    "selection_or_threshold_tuning_used"
                ],
                "required_disposition": _LOCKBOX_DISPOSITION,
            }
        )
    normalized.sort(key=lambda row: (row["patient_id"], row["recording_id"], row["access_id"]))
    patients = sorted({row["patient_id"] for row in normalized})
    recordings = sorted({row["recording_id"] for row in normalized})
    body: dict[str, Any] = {
        "schema_version": SOURCE_EVAL_LOCKBOX_LEDGER_SCHEMA_VERSION,
        "ledger_id": "SOURCE-EVAL-LOCKBOX-LEDGER-PENDING",
        "method_id": SOURCE_EVAL_LOCKBOX_LEDGER_METHOD_ID,
        "dataset_id": dataset,
        "inventory_status": "complete_known_pre_scoring_access_inventory",
        "entries": normalized,
        "touched_patient_ids": patients,
        "touched_recording_ids": recordings,
        "touched_patient_roster_sha256": _canonical_sha256(patients),
        "touched_recording_roster_sha256": _canonical_sha256(recordings),
        "scope_receipt": {
            "identity_and_access_metadata_only": True,
            "reference_event_intervals_embedded": False,
            "soz_or_channel_labels_embedded": False,
            "empty_entries_means_no_known_prior_access": True,
            "overlapping_patient_requires_exclusion_from_untouched_lockbox": True,
            "production_or_sota_claim_authorized": False,
        },
    }
    body["ledger_id"] = "CONTLOCK-" + _canonical_sha256(body)[:24]
    return validate_source_eval_lockbox_access_ledger(body)


def validate_source_eval_lockbox_access_ledger(payload: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "ledger_id",
        "method_id",
        "dataset_id",
        "inventory_status",
        "entries",
        "touched_patient_ids",
        "touched_recording_ids",
        "touched_patient_roster_sha256",
        "touched_recording_roster_sha256",
        "scope_receipt",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("source-eval lockbox ledger has missing or unknown fields")
    data = deepcopy(payload)
    if (
        data["schema_version"] != SOURCE_EVAL_LOCKBOX_LEDGER_SCHEMA_VERSION
        or data["method_id"] != SOURCE_EVAL_LOCKBOX_LEDGER_METHOD_ID
        or data["inventory_status"]
        != "complete_known_pre_scoring_access_inventory"
    ):
        raise ValueError("source-eval lockbox ledger schema/method/status drifted")
    _identifier(data["dataset_id"], "lockbox dataset_id")
    expected_scope = {
        "identity_and_access_metadata_only": True,
        "reference_event_intervals_embedded": False,
        "soz_or_channel_labels_embedded": False,
        "empty_entries_means_no_known_prior_access": True,
        "overlapping_patient_requires_exclusion_from_untouched_lockbox": True,
        "production_or_sota_claim_authorized": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("source-eval lockbox ledger scope drifted")
    entries = data["entries"]
    if not isinstance(entries, list):
        raise TypeError("source-eval lockbox entries must be an array")
    normalized: list[dict[str, Any]] = []
    access_ids: set[str] = set()
    for index, raw in enumerate(entries):
        required_entry = {
            "access_id",
            "patient_id",
            "recording_id",
            "access_kind",
            "selection_or_threshold_tuning_used",
            "required_disposition",
        }
        if type(raw) is not dict or set(raw) != required_entry:
            raise ValueError(f"lockbox entry {index} drifted")
        access_id = _identifier(raw["access_id"], f"lockbox entry {index} access_id")
        if access_id in access_ids:
            raise ValueError("lockbox access IDs must be unique")
        access_ids.add(access_id)
        if raw["access_kind"] not in _LOCKBOX_ACCESS_KINDS:
            raise ValueError("lockbox access kind drifted")
        if raw["required_disposition"] != _LOCKBOX_DISPOSITION:
            raise ValueError("lockbox access disposition drifted")
        if type(raw["selection_or_threshold_tuning_used"]) is not bool:
            raise TypeError("lockbox tuning-use flag must be boolean")
        normalized.append(
            {
                "access_id": access_id,
                "patient_id": _identifier(raw["patient_id"], "lockbox patient_id"),
                "recording_id": _identifier(
                    raw["recording_id"], "lockbox recording_id"
                ),
                "access_kind": raw["access_kind"],
                "selection_or_threshold_tuning_used": raw[
                    "selection_or_threshold_tuning_used"
                ],
                "required_disposition": _LOCKBOX_DISPOSITION,
            }
        )
    normalized.sort(key=lambda row: (row["patient_id"], row["recording_id"], row["access_id"]))
    if normalized != entries:
        raise ValueError("source-eval lockbox entries are not canonical")
    patients = sorted({row["patient_id"] for row in normalized})
    recordings = sorted({row["recording_id"] for row in normalized})
    if data["touched_patient_ids"] != patients or data["touched_recording_ids"] != recordings:
        raise ValueError("source-eval lockbox touched rosters drifted")
    if (
        data["touched_patient_roster_sha256"] != _canonical_sha256(patients)
        or data["touched_recording_roster_sha256"] != _canonical_sha256(recordings)
    ):
        raise ValueError("source-eval lockbox roster hash drifted")
    digest = deepcopy(data)
    digest["ledger_id"] = "SOURCE-EVAL-LOCKBOX-LEDGER-PENDING"
    if data["ledger_id"] != "CONTLOCK-" + _canonical_sha256(digest)[:24]:
        raise ValueError("source-eval lockbox ledger is not content-bound")
    return data


def _selected_operating_point(
    calibration: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if calibration["source_eval_use_authorized"] is not True:
        raise ValueError("calibration does not authorize source_eval")
    if (
        calibration["constraint_status"] != "met_selected_one_operating_point"
        or calibration["patient_isolation_status"] != "verified_no_patient_overlap"
        or calibration["source_dev_inventory_status"]
        != "verified_complete_expected_source_dev_inventory"
    ):
        raise ValueError("calibration lacks source_eval admission prerequisites")
    selected = calibration["selected_operating_point"]
    required_selected = {
        "operating_point_id",
        "candidate_id",
        "decoder_policy",
        "pooled_metrics",
        "patient_macro_metrics",
    }
    if type(selected) is not dict or set(selected) != required_selected:
        raise ValueError("calibration has no exact selected operating point")
    matches = [
        candidate
        for candidate in calibration["candidate_results"]
        if candidate.get("candidate_id") == selected["candidate_id"]
    ]
    if len(matches) != 1:
        raise ValueError("selected operating point candidate is not unique")
    candidate = matches[0]
    required_candidate = {
        "candidate_id",
        "decoder_policy",
        "decoder_policy_sha256",
        "pooled_metrics",
        "patient_macro_metrics",
        "coverage_accounting",
        "high_recall_constraints_met",
    }
    if type(candidate) is not dict or set(candidate) != required_candidate:
        raise ValueError("selected calibration candidate fields drifted")
    policy_sha256 = _canonical_sha256(candidate["decoder_policy"])
    if (
        candidate["decoder_policy_sha256"] != policy_sha256
        or candidate["candidate_id"] != "CONTCAND-" + policy_sha256[:20]
        or candidate["high_recall_constraints_met"] is not True
        or selected["decoder_policy"] != candidate["decoder_policy"]
        or selected["pooled_metrics"] != candidate["pooled_metrics"]
        or selected["patient_macro_metrics"] != candidate["patient_macro_metrics"]
    ):
        raise ValueError("selected operating point does not replay its candidate")
    selection = calibration["selection_definition"]
    pooled = candidate["pooled_metrics"].get("event_sensitivity")
    macro = candidate["patient_macro_metrics"].get("event_sensitivity_macro")
    if (
        pooled is None
        or macro is None
        or _finite(pooled, "selected pooled sensitivity") + 1e-12
        < _finite(selection["minimum_pooled_event_sensitivity"], "pooled floor")
        or _finite(macro, "selected macro sensitivity") + 1e-12
        < _finite(
            selection["minimum_patient_macro_event_sensitivity"], "macro floor"
        )
    ):
        raise ValueError("selected operating point does not satisfy recall floors")
    expected_operating_point_id = "CONTOP-" + _canonical_sha256(
        {
            "provider_id": calibration["provider_id"],
            "source_dev_rows_sha256": calibration["input_rows_sha256"],
            "candidate_id": candidate["candidate_id"],
            "selection_method_id": calibration["method_id"],
        }
    )[:24]
    if selected["operating_point_id"] != expected_operating_point_id:
        raise ValueError("selected operating-point ID does not replay")
    return deepcopy(selected), deepcopy(candidate), policy_sha256


def _validate_decoding_receipts(
    value: object,
    *,
    provider_id: str,
    operating_point_id: str,
    decoder_policy_sha256: str,
    prediction_by_recording: Mapping[str, list[dict[str, float]]],
) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(value, list) or not value:
        raise ValueError("source-eval admission needs decoder receipts")
    required = {
        "recording_id",
        "provider_id",
        "operating_point_id",
        "provider_execution_receipt_id",
        "provider_execution_receipt_sha256",
        "full_record_result_id",
        "full_record_result_sha256",
        "outcome_status",
        "decoder_method_id",
        "decoder_code_sha256",
        "decoder_policy_sha256",
        "decoding_receipt_id",
        "decoding_receipt_sha256",
        "predicted_events_sha256",
    }
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    decoder_code_hashes: set[str] = set()
    for index, raw in enumerate(value):
        if type(raw) is not dict or set(raw) != required:
            raise ValueError(f"decoder receipt {index} has missing or unknown fields")
        row = deepcopy(raw)
        recording_id = _identifier(row["recording_id"], "decoder recording_id")
        if recording_id in seen or recording_id not in prediction_by_recording:
            raise ValueError("decoder receipt recording roster is invalid")
        seen.add(recording_id)
        if row["provider_id"] != provider_id or row["operating_point_id"] != operating_point_id:
            raise ValueError("decoder receipt provider/operating point mismatch")
        if row["decoder_method_id"] != CONTINUOUS_DETECTION_METHOD_ID:
            raise ValueError("decoder receipt method mismatch")
        decoder_code_hashes.add(_sha256(row["decoder_code_sha256"], "decoder code hash"))
        if row["decoder_policy_sha256"] != decoder_policy_sha256:
            raise ValueError("decoder receipt policy hash mismatch")
        for field in (
            "provider_execution_receipt_sha256",
            "full_record_result_sha256",
            "decoding_receipt_sha256",
        ):
            _sha256(row[field], field)
        for field in (
            "provider_execution_receipt_id",
            "full_record_result_id",
            "decoding_receipt_id",
        ):
            _identifier(row[field], field)
        expected_events_hash = _canonical_sha256(prediction_by_recording[recording_id])
        if row["predicted_events_sha256"] != expected_events_hash:
            raise ValueError("decoder receipt prediction-event hash mismatch")
        events_present = bool(prediction_by_recording[recording_id])
        expected_outcome = (
            "completed_with_alarms" if events_present else "completed_zero_alarm"
        )
        if row["outcome_status"] not in _COMPLETED_OUTCOMES or row["outcome_status"] != expected_outcome:
            raise ValueError("partial/failed decoder outcome cannot enter source_eval scoring")
        normalized.append(row)
    normalized.sort(key=lambda row: row["recording_id"])
    if normalized != value:
        raise ValueError("source-eval decoder receipts are not canonical")
    if seen != set(prediction_by_recording):
        raise ValueError("source-eval decoder receipt roster is incomplete")
    if len(decoder_code_hashes) != 1:
        raise ValueError("decoder code changed within source_eval predictions")
    return normalized, next(iter(decoder_code_hashes))


@dataclass(frozen=True, slots=True)
class ValidatedSourceEvalAdmission:
    """Opaque, replayed admission accepted by the source-eval scorer."""

    _canonical_payload_json: str

    def payload(self) -> dict[str, Any]:
        value = json.loads(self._canonical_payload_json)
        if type(value) is not dict:
            raise RuntimeError("validated admission payload corrupted")
        return value


def build_continuous_detection_source_eval_admission(
    *,
    provider_definition: Mapping[str, Any],
    calibration_receipt: Mapping[str, Any],
    split_roster_receipt: Mapping[str, Any],
    lockbox_access_ledger: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    decoding_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build admission after EEG-only predictions freeze and before references open."""

    provider = validate_provider_definition(dict(provider_definition))
    calibration = _validate_calibration_receipt(dict(calibration_receipt))
    roster = validate_continuous_detector_split_roster(dict(split_roster_receipt))
    lockbox = validate_source_eval_lockbox_access_ledger(dict(lockbox_access_ledger))
    selected, _, policy_sha256 = _selected_operating_point(calibration)
    predictions = normalize_source_eval_prediction_inventory(prediction_rows)
    prediction_by_recording = {
        row["recording_id"]: row["predicted_events"] for row in predictions
    }
    decoder_rows, decoder_code_sha256 = _validate_decoding_receipts(
        list(decoding_receipts),
        provider_id=calibration["provider_id"],
        operating_point_id=selected["operating_point_id"],
        decoder_policy_sha256=policy_sha256,
        prediction_by_recording=prediction_by_recording,
    )
    body: dict[str, Any] = {
        "schema_version": SOURCE_EVAL_ADMISSION_SCHEMA_VERSION,
        "admission_id": "SOURCE-EVAL-ADMISSION-PENDING",
        "method_id": SOURCE_EVAL_ADMISSION_METHOD_ID,
        "admission_status": "authorized_for_one_shot_source_eval_scoring",
        "evaluation_split": "source_eval",
        "provider_definition": provider,
        "provider_definition_sha256": _canonical_sha256(provider),
        "calibration_receipt": calibration,
        "calibration_receipt_sha256": _canonical_sha256(calibration),
        "split_roster_receipt": roster,
        "split_roster_receipt_sha256": _canonical_sha256(roster),
        "lockbox_access_ledger": lockbox,
        "lockbox_access_ledger_sha256": _canonical_sha256(lockbox),
        "provider_id": calibration["provider_id"],
        "operating_point_id": selected["operating_point_id"],
        "decoder_method_id": CONTINUOUS_DETECTION_METHOD_ID,
        "decoder_code_sha256": decoder_code_sha256,
        "decoder_policy": deepcopy(selected["decoder_policy"]),
        "decoder_policy_sha256": policy_sha256,
        "decoding_receipts": decoder_rows,
        "decoding_receipt_roster_sha256": _canonical_sha256(decoder_rows),
        "prediction_inventory": predictions,
        "prediction_inventory_sha256": _canonical_sha256(predictions),
        "source_dev_patient_roster_sha256": roster["split_rosters"]["source_dev"][
            "patient_roster_sha256"
        ],
        "source_dev_recording_roster_sha256": roster["split_rosters"]["source_dev"][
            "recording_roster_sha256"
        ],
        "source_eval_patient_roster_sha256": roster["split_rosters"]["source_eval"][
            "patient_roster_sha256"
        ],
        "source_eval_recording_roster_sha256": roster["split_rosters"]["source_eval"][
            "recording_roster_sha256"
        ],
        "scope_receipt": {
            "calibration_receipt_replayed": True,
            "selected_operating_point_replayed": True,
            "source_eval_authorization_replayed": True,
            "prediction_inventory_frozen_before_reference_scoring": True,
            "reference_event_intervals_embedded": False,
            "touched_lockbox_patient_or_recording_overlap": False,
            "provider_decoder_policy_and_rosters_cross_bound": True,
            "partial_or_technical_failures_silently_dropped": False,
            "production_or_sota_claim_authorized": False,
        },
    }
    body["admission_id"] = "CONTEVALADMIT-" + _canonical_sha256(body)[:24]
    return validate_continuous_detection_source_eval_admission(body).payload()


def validate_continuous_detection_source_eval_admission(
    payload: object,
) -> ValidatedSourceEvalAdmission:
    required = {
        "schema_version",
        "admission_id",
        "method_id",
        "admission_status",
        "evaluation_split",
        "provider_definition",
        "provider_definition_sha256",
        "calibration_receipt",
        "calibration_receipt_sha256",
        "split_roster_receipt",
        "split_roster_receipt_sha256",
        "lockbox_access_ledger",
        "lockbox_access_ledger_sha256",
        "provider_id",
        "operating_point_id",
        "decoder_method_id",
        "decoder_code_sha256",
        "decoder_policy",
        "decoder_policy_sha256",
        "decoding_receipts",
        "decoding_receipt_roster_sha256",
        "prediction_inventory",
        "prediction_inventory_sha256",
        "source_dev_patient_roster_sha256",
        "source_dev_recording_roster_sha256",
        "source_eval_patient_roster_sha256",
        "source_eval_recording_roster_sha256",
        "scope_receipt",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("source-eval admission has missing or unknown fields")
    data = deepcopy(payload)
    if (
        data["schema_version"] != SOURCE_EVAL_ADMISSION_SCHEMA_VERSION
        or data["method_id"] != SOURCE_EVAL_ADMISSION_METHOD_ID
        or data["admission_status"]
        != "authorized_for_one_shot_source_eval_scoring"
        or data["evaluation_split"] != "source_eval"
    ):
        raise ValueError("source-eval admission schema/method/status drifted")
    provider = validate_provider_definition(data["provider_definition"])
    calibration = _validate_calibration_receipt(data["calibration_receipt"])
    roster = validate_continuous_detector_split_roster(data["split_roster_receipt"])
    lockbox = validate_source_eval_lockbox_access_ledger(
        data["lockbox_access_ledger"]
    )
    for value, expected, context in (
        (data["provider_definition_sha256"], _canonical_sha256(provider), "provider definition"),
        (data["calibration_receipt_sha256"], _canonical_sha256(calibration), "calibration receipt"),
        (data["split_roster_receipt_sha256"], _canonical_sha256(roster), "split roster"),
        (data["lockbox_access_ledger_sha256"], _canonical_sha256(lockbox), "lockbox ledger"),
    ):
        if value != expected:
            raise ValueError(f"source-eval admission {context} hash mismatch")
    selected, _, policy_sha256 = _selected_operating_point(calibration)
    if provider["provider_id"] != calibration["provider_id"] or data["provider_id"] != calibration["provider_id"]:
        raise ValueError("source-eval admission provider mismatch")
    if provider["implementation_status"] not in {"runnable_research", "production_qualified"}:
        raise ValueError("source-eval provider is not locally runnable")
    if data["operating_point_id"] != selected["operating_point_id"]:
        raise ValueError("source-eval admission operating point mismatch")
    if data["decoder_method_id"] != CONTINUOUS_DETECTION_METHOD_ID:
        raise ValueError("source-eval admission decoder method mismatch")
    _sha256(data["decoder_code_sha256"], "decoder code SHA-256")
    if data["decoder_policy"] != selected["decoder_policy"] or data["decoder_policy_sha256"] != policy_sha256:
        raise ValueError("source-eval admission decoder policy mismatch")
    split_rosters = roster["split_rosters"]
    if "source_dev" not in split_rosters or "source_eval" not in split_rosters:
        raise ValueError("source-eval admission roster lacks source_dev/source_eval")
    source_dev = split_rosters["source_dev"]
    source_eval = split_rosters["source_eval"]
    expected_roster_bindings = {
        "source_dev_patient_roster_sha256": source_dev["patient_roster_sha256"],
        "source_dev_recording_roster_sha256": source_dev["recording_roster_sha256"],
        "source_eval_patient_roster_sha256": source_eval["patient_roster_sha256"],
        "source_eval_recording_roster_sha256": source_eval["recording_roster_sha256"],
    }
    for field, expected in expected_roster_bindings.items():
        if data[field] != expected:
            raise ValueError(f"source-eval admission {field} mismatch")
    if (
        calibration["development_patient_roster_sha256"]
        != source_dev["patient_roster_sha256"]
        or calibration["expected_source_dev_recording_roster_sha256"]
        != source_dev["recording_roster_sha256"]
        or calibration["evaluation_patient_roster_sha256"]
        != source_eval["patient_roster_sha256"]
    ):
        raise ValueError("calibration and split-roster hashes disagree")
    touched_patients = set(lockbox["touched_patient_ids"])
    touched_recordings = set(lockbox["touched_recording_ids"])
    patient_overlap = touched_patients.intersection(source_eval["patient_ids"])
    recording_overlap = touched_recordings.intersection(source_eval["recording_ids"])
    if patient_overlap or recording_overlap:
        raise ValueError(
            "touched lockbox patient/recording remains in source_eval roster"
        )
    predictions = normalize_source_eval_prediction_inventory(data["prediction_inventory"])
    if predictions != data["prediction_inventory"] or data["prediction_inventory_sha256"] != _canonical_sha256(predictions):
        raise ValueError("source-eval prediction inventory hash/canonicalization drifted")
    if [row["recording_id"] for row in predictions] != source_eval["recording_ids"]:
        raise ValueError("source-eval prediction inventory does not equal frozen roster")
    prediction_by_recording = {
        row["recording_id"]: row["predicted_events"] for row in predictions
    }
    decoder_rows, decoder_code_sha256 = _validate_decoding_receipts(
        data["decoding_receipts"],
        provider_id=data["provider_id"],
        operating_point_id=data["operating_point_id"],
        decoder_policy_sha256=data["decoder_policy_sha256"],
        prediction_by_recording=prediction_by_recording,
    )
    if (
        data["decoder_code_sha256"] != decoder_code_sha256
        or data["decoding_receipt_roster_sha256"]
        != _canonical_sha256(decoder_rows)
    ):
        raise ValueError("source-eval decoder receipt/code hash mismatch")
    expected_scope = {
        "calibration_receipt_replayed": True,
        "selected_operating_point_replayed": True,
        "source_eval_authorization_replayed": True,
        "prediction_inventory_frozen_before_reference_scoring": True,
        "reference_event_intervals_embedded": False,
        "touched_lockbox_patient_or_recording_overlap": False,
        "provider_decoder_policy_and_rosters_cross_bound": True,
        "partial_or_technical_failures_silently_dropped": False,
        "production_or_sota_claim_authorized": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("source-eval admission scope drifted")
    digest = deepcopy(data)
    digest["admission_id"] = "SOURCE-EVAL-ADMISSION-PENDING"
    if data["admission_id"] != "CONTEVALADMIT-" + _canonical_sha256(digest)[:24]:
        raise ValueError("source-eval admission is not content-bound")
    return ValidatedSourceEvalAdmission(_canonical_json(data))


def authorize_source_eval_prediction_inventory(
    admission: ValidatedSourceEvalAdmission,
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Check frozen predictions before any source-eval reference is opened."""

    if type(admission) is not ValidatedSourceEvalAdmission:
        raise TypeError("source_eval requires an exact validated admission")
    replayed = validate_continuous_detection_source_eval_admission(
        admission.payload()
    ).payload()
    observed = normalize_source_eval_prediction_inventory(prediction_rows)
    if observed != replayed["prediction_inventory"]:
        raise ValueError("prediction file disagrees with source-eval admission")
    return replayed


def authorize_source_eval_benchmark_rows(
    admission: ValidatedSourceEvalAdmission,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind post-admission benchmark rows without reading their references here."""

    replayed = authorize_source_eval_prediction_inventory(admission, rows)
    patient_ids: set[str] = set()
    recording_ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"benchmark row {index} must be an object")
        if row.get("split") != "source_eval":
            raise ValueError("source-eval admission cannot authorize another split")
        patient_ids.add(_identifier(row.get("patient_id"), "benchmark patient_id"))
        recording_ids.append(
            _identifier(row.get("recording_id"), "benchmark recording_id")
        )
    roster = replayed["split_roster_receipt"]["split_rosters"]["source_eval"]
    if sorted(recording_ids) != roster["recording_ids"] or sorted(patient_ids) != roster["patient_ids"]:
        raise ValueError("benchmark patient/recording roster disagrees with admission")
    return replayed


def source_eval_admission_benchmark_binding(
    admission_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the minimal admission provenance stored in a benchmark receipt."""

    data = dict(admission_payload)
    return {
        "admission_id": data["admission_id"],
        "admission_sha256": _canonical_sha256(data),
        "provider_id": data["provider_id"],
        "operating_point_id": data["operating_point_id"],
        "calibration_receipt_id": data["calibration_receipt"][
            "calibration_receipt_id"
        ],
        "calibration_receipt_sha256": data["calibration_receipt_sha256"],
        "provider_definition_sha256": data["provider_definition_sha256"],
        "decoder_method_id": data["decoder_method_id"],
        "decoder_code_sha256": data["decoder_code_sha256"],
        "decoder_policy_sha256": data["decoder_policy_sha256"],
        "decoding_receipt_roster_sha256": data[
            "decoding_receipt_roster_sha256"
        ],
        "lockbox_access_ledger_sha256": data["lockbox_access_ledger_sha256"],
        "source_dev_patient_roster_sha256": data[
            "source_dev_patient_roster_sha256"
        ],
        "source_dev_recording_roster_sha256": data[
            "source_dev_recording_roster_sha256"
        ],
        "source_eval_patient_roster_sha256": data[
            "source_eval_patient_roster_sha256"
        ],
        "source_eval_recording_roster_sha256": data[
            "source_eval_recording_roster_sha256"
        ],
    }


__all__ = [
    "SOURCE_EVAL_ADMISSION_METHOD_ID",
    "SOURCE_EVAL_ADMISSION_SCHEMA_VERSION",
    "SOURCE_EVAL_LOCKBOX_LEDGER_METHOD_ID",
    "SOURCE_EVAL_LOCKBOX_LEDGER_SCHEMA_VERSION",
    "ValidatedSourceEvalAdmission",
    "authorize_source_eval_benchmark_rows",
    "authorize_source_eval_prediction_inventory",
    "build_continuous_detection_source_eval_admission",
    "build_source_eval_lockbox_access_ledger",
    "normalize_source_eval_prediction_inventory",
    "source_eval_admission_benchmark_binding",
    "validate_continuous_detection_source_eval_admission",
    "validate_source_eval_lockbox_access_ledger",
]
