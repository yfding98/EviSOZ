"""Prediction-first G0 A1 candidate roster and post-freeze target join.

This module closes a narrow but important executable gap in the clinical EEG
v1.3-min method contract.  It does *not* run a detector and it does not bless a
provider.  Instead, it gives an already materialized patient-OOF detector a
strict, content-addressed denominator with the following properties:

* every expected source-train record is represented exactly once;
* completed zero-candidate, partial-coverage, and technical-failure records are
  retained rather than disappearing from the denominator;
* detector candidates and deterministic reference-blind random supports are
  frozen before any public seizure interval is accepted;
* the prediction fold set must exactly equal the patient-held-out fold set;
* the only post-freeze reference surface is a global event-interval roster; and
* matched, duplicate/fragment, near-event, false-candidate, and random-
  background strata are derived by one explicit matching policy.

The random-background label describes a *proposal origin*, not an assumed
negative.  If a reference-blind random support collides with a seizure after
the freeze barrier, it is correctly relabelled as a matched positive (or a
hard candidate) while retaining its origin.  This avoids a subtle source of
false-negative label noise.

The output is a data/lineage contract only.  It does not authorize model
training, G0 promotion, detector/SOZ performance claims, or clinical use.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence


BA_IEG_G0_A1_INVENTORY_SCHEMA_V1: Final[
    str
] = "ba_ieg_g0_a1_source_train_oof_inventory_v1"
BA_IEG_G0_A1_RANDOM_POLICY_SCHEMA_V1: Final[
    str
] = "ba_ieg_g0_a1_reference_blind_random_background_policy_v1"
BA_IEG_G0_A1_PREDICTION_ROSTER_SCHEMA_V1: Final[
    str
] = "ba_ieg_g0_a1_prediction_first_candidate_roster_v1"
BA_IEG_G0_A1_MATCH_POLICY_SCHEMA_V1: Final[
    str
] = "ba_ieg_g0_a1_postfreeze_interval_match_policy_v1"
BA_IEG_G0_A1_REFERENCE_ROSTER_SCHEMA_V1: Final[
    str
] = "ba_ieg_g0_a1_public_interval_reference_roster_v1"
BA_IEG_G0_A1_TARGET_JOIN_SCHEMA_V1: Final[
    str
] = "ba_ieg_g0_a1_postfreeze_candidate_target_join_v1"

BA_IEG_G0_A1_RECORD_OUTCOMES: Final[tuple[str, ...]] = (
    "completed_with_candidates",
    "completed_zero_candidate",
    "partial_coverage",
    "technical_failure",
)
BA_IEG_G0_A1_CANDIDATE_ORIGINS: Final[tuple[str, ...]] = (
    "detector_proposal",
    "candidate_blind_random_background",
)
BA_IEG_G0_A1_TRAINING_CLASSES: Final[tuple[str, ...]] = (
    "matched_true_event",
    "unmatched_false_candidate",
    "fragmented_or_duplicate_hard_candidate",
    "near_event_hard_candidate",
    "candidate_blind_random_background",
)

_SHA256_ALPHABET: Final[frozenset[str]] = frozenset("0123456789abcdef")
_INVENTORY_RECORD_FIELDS: Final[set[str]] = {
    "patient_uid",
    "recording_id",
    "recording_duration_seconds",
    "source_signal_sha256",
    "held_out_fold_ids",
    "patient_fold_binding_sha256",
}
_PREDICTION_RECORD_FIELDS: Final[set[str]] = {
    "patient_uid",
    "recording_id",
    "recording_duration_seconds",
    "source_signal_sha256",
    "inference_fold_ids",
    "prediction_artifact_sha256",
    "prediction_result_receipt_sha256",
    "outcome",
    "failure_stage",
    "candidates",
}
_DETECTOR_CANDIDATE_FIELDS: Final[set[str]] = {
    "candidate_id",
    "start_offset_seconds",
    "stop_offset_seconds",
    "anchor_offset_seconds",
    "score",
    "decision_available_offset_seconds",
    "candidate_receipt_sha256",
}
_REFERENCE_RECORD_FIELDS: Final[set[str]] = {
    "patient_uid",
    "recording_id",
    "recording_duration_seconds",
    "reference_coverage_status",
    "annotation_timestamp_resolution_seconds",
    "source_reference_receipt_sha256",
    "seizure_intervals",
}
_REFERENCE_EVENT_FIELDS: Final[set[str]] = {
    "public_event_id",
    "onset_recording_seconds",
    "offset_recording_seconds",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_object(value: object, fields: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{context} fields drifted")
    return deepcopy(value)


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value).difference(_SHA256_ALPHABET)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TypeError(f"{context} must be a positive integer")
    return value


def _fold_ids(value: object, context: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{context} must be a non-empty list")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value
    ):
        raise TypeError(f"{context} values must be non-negative integers")
    if value != sorted(set(value)):
        raise ValueError(f"{context} must be unique and canonically sorted")
    return list(value)


def _seal(body: Mapping[str, Any], *, id_field: str, prefix: str) -> dict[str, Any]:
    result = deepcopy(dict(body))
    result[id_field] = f"{prefix}-PENDING"
    result["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    id_source = deepcopy(result)
    result[id_field] = prefix + "-" + _canonical_sha256(id_source)[:24]
    hash_source = deepcopy(result)
    hash_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["receipt_sha256"] = _canonical_sha256(hash_source)
    return result


def _replay_seal(
    value: Mapping[str, Any], *, id_field: str, prefix: str, context: str
) -> None:
    expected = _seal(value, id_field=id_field, prefix=prefix)
    if (
        value[id_field] != expected[id_field]
        or value["receipt_sha256"] != expected["receipt_sha256"]
    ):
        raise ValueError(f"{context} content address does not replay")


def _validate_inventory_record(value: object, index: int) -> dict[str, Any]:
    row = _strict_object(value, _INVENTORY_RECORD_FIELDS, f"inventory record {index}")
    duration = _finite(row["recording_duration_seconds"], "recording duration")
    if duration <= 0:
        raise ValueError("recording duration must be positive")
    return {
        "patient_uid": _identifier(row["patient_uid"], "patient UID"),
        "recording_id": _identifier(row["recording_id"], "recording ID"),
        "recording_duration_seconds": duration,
        "source_signal_sha256": _sha256(row["source_signal_sha256"], "source signal"),
        "held_out_fold_ids": _fold_ids(row["held_out_fold_ids"], "held-out folds"),
        "patient_fold_binding_sha256": _sha256(
            row["patient_fold_binding_sha256"], "patient-fold binding"
        ),
    }


def build_ba_ieg_g0_a1_oof_inventory_v1(
    records: Sequence[Mapping[str, Any]],
    *,
    fold_assignment_receipt_sha256: str,
) -> dict[str, Any]:
    """Freeze the complete source-train patient/fold/signal denominator."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("inventory records must be a sequence")
    normalized = [
        _validate_inventory_record(dict(row), index)
        for index, row in enumerate(records)
    ]
    if not normalized:
        raise ValueError("G0 A1 inventory must contain at least one record")
    normalized.sort(key=lambda row: (row["patient_uid"], row["recording_id"]))
    record_ids = [row["recording_id"] for row in normalized]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("G0 A1 inventory repeats a recording")
    patient_fold_sets: dict[str, tuple[tuple[int, ...], str]] = {}
    for row in normalized:
        current = (
            tuple(row["held_out_fold_ids"]),
            row["patient_fold_binding_sha256"],
        )
        previous = patient_fold_sets.setdefault(row["patient_uid"], current)
        if previous != current:
            raise ValueError("one patient has inconsistent held-out fold bindings")
    body = {
        "schema_version": BA_IEG_G0_A1_INVENTORY_SCHEMA_V1,
        "inventory_id": "BAIEG-G0-A1-INVENTORY-PENDING",
        "model_split": "source_train",
        "fold_assignment_receipt_sha256": _sha256(
            fold_assignment_receipt_sha256, "fold-assignment receipt"
        ),
        "records": normalized,
        "counts": {
            "patients": len(patient_fold_sets),
            "records": len(normalized),
        },
        "scope_receipt": {
            "identity_signal_and_patient_fold_metadata_only": True,
            "public_event_intervals_opened": 0,
            "edf_annotations_opened": 0,
            "channel_or_soz_targets_opened": 0,
            "private_doctor_or_clinical_text_opened": 0,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    result = _seal(body, id_field="inventory_id", prefix="BAIEGG0A1INV")
    validate_ba_ieg_g0_a1_oof_inventory_v1(result)
    return result


def validate_ba_ieg_g0_a1_oof_inventory_v1(payload: object) -> dict[str, Any]:
    fields = {
        "schema_version",
        "inventory_id",
        "model_split",
        "fold_assignment_receipt_sha256",
        "records",
        "counts",
        "scope_receipt",
        "receipt_sha256",
    }
    data = _strict_object(payload, fields, "G0 A1 OOF inventory")
    if (
        data["schema_version"] != BA_IEG_G0_A1_INVENTORY_SCHEMA_V1
        or data["model_split"] != "source_train"
    ):
        raise ValueError("G0 A1 OOF inventory schema/split drifted")
    _sha256(data["fold_assignment_receipt_sha256"], "fold-assignment receipt")
    if not isinstance(data["records"], list) or not data["records"]:
        raise ValueError("G0 A1 OOF inventory has no records")
    rows = [
        _validate_inventory_record(row, index)
        for index, row in enumerate(data["records"])
    ]
    if rows != sorted(rows, key=lambda row: (row["patient_uid"], row["recording_id"])):
        raise ValueError("G0 A1 OOF inventory is not canonically sorted")
    if len({row["recording_id"] for row in rows}) != len(rows):
        raise ValueError("G0 A1 OOF inventory repeats a recording")
    by_patient: dict[str, tuple[tuple[int, ...], str]] = {}
    for row in rows:
        current = (tuple(row["held_out_fold_ids"]), row["patient_fold_binding_sha256"])
        if (
            row["patient_uid"] in by_patient
            and by_patient[row["patient_uid"]] != current
        ):
            raise ValueError("G0 A1 patient fold binding drifted")
        by_patient[row["patient_uid"]] = current
    if data["counts"] != {"patients": len(by_patient), "records": len(rows)}:
        raise ValueError("G0 A1 inventory counts do not replay")
    expected_scope = {
        "identity_signal_and_patient_fold_metadata_only": True,
        "public_event_intervals_opened": 0,
        "edf_annotations_opened": 0,
        "channel_or_soz_targets_opened": 0,
        "private_doctor_or_clinical_text_opened": 0,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("G0 A1 inventory firewall drifted")
    _replay_seal(
        data, id_field="inventory_id", prefix="BAIEGG0A1INV", context="inventory"
    )
    return data


def build_ba_ieg_g0_a1_random_background_policy_v1(
    *,
    seed_sha256: str,
    samples_per_completed_record: int,
    support_seconds: float,
) -> dict[str, Any]:
    samples = _positive_integer(
        samples_per_completed_record, "random samples per completed record"
    )
    support = _finite(support_seconds, "random support seconds")
    if support <= 0:
        raise ValueError("random support seconds must be positive")
    body = {
        "schema_version": BA_IEG_G0_A1_RANDOM_POLICY_SCHEMA_V1,
        "policy_id": "BAIEG-G0-A1-RANDOM-PENDING",
        "seed_sha256": _sha256(seed_sha256, "random policy seed"),
        "samples_per_completed_record": samples,
        "support_seconds": support,
        "eligible_record_outcomes": [
            "completed_with_candidates",
            "completed_zero_candidate",
        ],
        "sampling_method": (
            "sha256_counter_uniform_start_from_record_duration_without_reference_or_detector_candidates"
        ),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    result = _seal(body, id_field="policy_id", prefix="BAIEGG0A1RND")
    validate_ba_ieg_g0_a1_random_background_policy_v1(result)
    return result


def validate_ba_ieg_g0_a1_random_background_policy_v1(
    payload: object,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "policy_id",
        "seed_sha256",
        "samples_per_completed_record",
        "support_seconds",
        "eligible_record_outcomes",
        "sampling_method",
        "receipt_sha256",
    }
    data = _strict_object(payload, fields, "G0 A1 random-background policy")
    if data["schema_version"] != BA_IEG_G0_A1_RANDOM_POLICY_SCHEMA_V1:
        raise ValueError("random-background policy schema drifted")
    _sha256(data["seed_sha256"], "random policy seed")
    _positive_integer(data["samples_per_completed_record"], "random sample count")
    if _finite(data["support_seconds"], "random support seconds") <= 0:
        raise ValueError("random support seconds must be positive")
    if data["eligible_record_outcomes"] != [
        "completed_with_candidates",
        "completed_zero_candidate",
    ]:
        raise ValueError("random-background eligible outcome roster drifted")
    if data["sampling_method"] != (
        "sha256_counter_uniform_start_from_record_duration_without_reference_or_detector_candidates"
    ):
        raise ValueError("random-background sampling method drifted")
    _replay_seal(
        data, id_field="policy_id", prefix="BAIEGG0A1RND", context="random policy"
    )
    return data


def _normalize_detector_candidate(
    value: object, *, duration: float, record_id: str, index: int
) -> dict[str, Any]:
    row = _strict_object(
        value, _DETECTOR_CANDIDATE_FIELDS, f"detector candidate {record_id}:{index}"
    )
    start = _finite(row["start_offset_seconds"], "candidate start")
    stop = _finite(row["stop_offset_seconds"], "candidate stop")
    anchor = _finite(row["anchor_offset_seconds"], "candidate anchor")
    available = _finite(
        row["decision_available_offset_seconds"], "candidate decision availability"
    )
    if start < 0 or stop <= start or stop > duration:
        raise ValueError("detector candidate support is outside its recording")
    if anchor < start or anchor > stop:
        raise ValueError("detector candidate anchor is outside its support")
    if available < stop or available > duration:
        raise ValueError("detector candidate decision availability is invalid")
    return {
        "candidate_id": _identifier(row["candidate_id"], "candidate ID"),
        "start_offset_seconds": start,
        "stop_offset_seconds": stop,
        "anchor_offset_seconds": anchor,
        "score": _finite(row["score"], "candidate score"),
        "decision_available_offset_seconds": available,
        "candidate_receipt_sha256": _sha256(
            row["candidate_receipt_sha256"], "candidate receipt"
        ),
    }


def _uniform_start(seed: str, recording_id: str, ordinal: int, maximum: float) -> float:
    digest = hashlib.sha256(
        _canonical_json_bytes(
            {
                "seed_sha256": seed,
                "recording_id": recording_id,
                "ordinal": ordinal,
            }
        )
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return unit * maximum


def _random_candidates(
    record: Mapping[str, Any], policy: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    if record["outcome"] not in policy["eligible_record_outcomes"]:
        return [], 0
    duration = float(record["recording_duration_seconds"])
    support = float(policy["support_seconds"])
    if duration + 1e-12 < support:
        return [], int(policy["samples_per_completed_record"])
    rows: list[dict[str, Any]] = []
    maximum = max(0.0, duration - support)
    for ordinal in range(1, int(policy["samples_per_completed_record"]) + 1):
        start = _uniform_start(
            str(policy["seed_sha256"]), str(record["recording_id"]), ordinal, maximum
        )
        stop = min(duration, start + support)
        body = {
            "schema": "ba_ieg_g0_a1_reference_blind_random_candidate_v1",
            "policy_receipt_sha256": policy["receipt_sha256"],
            "patient_uid": record["patient_uid"],
            "recording_id": record["recording_id"],
            "ordinal": ordinal,
            "start_offset_seconds": start,
            "stop_offset_seconds": stop,
            "anchor_offset_seconds": 0.5 * (start + stop),
            "source_signal_sha256": record["source_signal_sha256"],
        }
        receipt = _canonical_sha256(body)
        rows.append(
            {
                "candidate_id": "G0RND-" + receipt[:24],
                "patient_uid": record["patient_uid"],
                "recording_id": record["recording_id"],
                "origin": "candidate_blind_random_background",
                "start_offset_seconds": start,
                "stop_offset_seconds": stop,
                "anchor_offset_seconds": body["anchor_offset_seconds"],
                "score": None,
                "decision_available_offset_seconds": stop,
                "source_candidate_receipt_sha256": receipt,
                "random_policy_receipt_sha256": policy["receipt_sha256"],
            }
        )
    rows.sort(
        key=lambda row: (
            row["start_offset_seconds"],
            row["candidate_id"],
        )
    )
    return rows, 0


def _validate_prediction_record(
    value: object, *, inventory: Mapping[str, Any], index: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row = _strict_object(value, _PREDICTION_RECORD_FIELDS, f"prediction record {index}")
    patient = _identifier(row["patient_uid"], "prediction patient UID")
    recording = _identifier(row["recording_id"], "prediction recording ID")
    expected = inventory.get(recording)
    if expected is None:
        raise ValueError("prediction record is absent from the frozen inventory")
    duration = _finite(row["recording_duration_seconds"], "prediction duration")
    signal_hash = _sha256(row["source_signal_sha256"], "prediction source signal")
    folds = _fold_ids(row["inference_fold_ids"], "prediction inference folds")
    if (
        patient != expected["patient_uid"]
        or abs(duration - float(expected["recording_duration_seconds"])) > 1e-9
        or signal_hash != expected["source_signal_sha256"]
        or folds != expected["held_out_fold_ids"]
    ):
        raise ValueError(
            "prediction record does not replay patient-held-out inventory identity"
        )
    outcome = row["outcome"]
    if outcome not in BA_IEG_G0_A1_RECORD_OUTCOMES:
        raise ValueError("prediction record outcome is unsupported")
    failure = row["failure_stage"]
    if outcome == "technical_failure":
        _identifier(failure, "technical failure stage")
    elif failure is not None:
        raise ValueError("non-failure prediction record cannot claim a failure stage")
    if not isinstance(row["candidates"], list):
        raise TypeError("prediction candidates must be a list")
    detector = [
        _normalize_detector_candidate(
            item, duration=duration, record_id=recording, index=candidate_index
        )
        for candidate_index, item in enumerate(row["candidates"])
    ]
    detector.sort(
        key=lambda item: (
            item["start_offset_seconds"],
            item["anchor_offset_seconds"],
            item["candidate_id"],
        )
    )
    if len({item["candidate_id"] for item in detector}) != len(detector):
        raise ValueError("prediction record repeats a detector candidate")
    if outcome == "completed_with_candidates" and not detector:
        raise ValueError("completed-with-candidates record has no detector candidate")
    if outcome == "completed_zero_candidate" and detector:
        raise ValueError("zero-candidate record contains detector candidates")
    if outcome == "technical_failure" and detector:
        raise ValueError("technical-failure record cannot expose detector candidates")
    normalized_record = {
        "patient_uid": patient,
        "recording_id": recording,
        "recording_duration_seconds": duration,
        "source_signal_sha256": signal_hash,
        "inference_fold_ids": folds,
        "prediction_artifact_sha256": _sha256(
            row["prediction_artifact_sha256"], "prediction artifact"
        ),
        "prediction_result_receipt_sha256": _sha256(
            row["prediction_result_receipt_sha256"], "prediction result receipt"
        ),
        "outcome": outcome,
        "failure_stage": failure,
    }
    candidates = [
        {
            "candidate_id": item["candidate_id"],
            "patient_uid": patient,
            "recording_id": recording,
            "origin": "detector_proposal",
            "start_offset_seconds": item["start_offset_seconds"],
            "stop_offset_seconds": item["stop_offset_seconds"],
            "anchor_offset_seconds": item["anchor_offset_seconds"],
            "score": item["score"],
            "decision_available_offset_seconds": item[
                "decision_available_offset_seconds"
            ],
            "source_candidate_receipt_sha256": item["candidate_receipt_sha256"],
            "random_policy_receipt_sha256": None,
        }
        for item in detector
    ]
    return normalized_record, candidates


def build_ba_ieg_g0_a1_prediction_roster_v1(
    *,
    inventory: Mapping[str, Any],
    prediction_records: Sequence[Mapping[str, Any]],
    provider_id: str,
    provider_prediction_receipt_sha256: str,
    decoder_policy_receipt_sha256: str,
    random_background_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze all OOF predictions and random supports before target access."""

    source_inventory = validate_ba_ieg_g0_a1_oof_inventory_v1(dict(inventory))
    random_policy = validate_ba_ieg_g0_a1_random_background_policy_v1(
        dict(random_background_policy)
    )
    if not isinstance(prediction_records, Sequence) or isinstance(
        prediction_records, (str, bytes)
    ):
        raise TypeError("prediction records must be a sequence")
    expected = {row["recording_id"]: row for row in source_inventory["records"]}
    records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    short_random_opportunities = 0
    for index, raw in enumerate(prediction_records):
        record, detector = _validate_prediction_record(
            dict(raw), inventory=expected, index=index
        )
        random_rows, skipped = _random_candidates(record, random_policy)
        short_random_opportunities += skipped
        candidates.extend(detector)
        candidates.extend(random_rows)
        record["detector_candidate_ids"] = [item["candidate_id"] for item in detector]
        record["random_background_candidate_ids"] = [
            item["candidate_id"] for item in random_rows
        ]
        records.append(record)
    records.sort(key=lambda row: (row["patient_uid"], row["recording_id"]))
    if {row["recording_id"] for row in records} != set(expected) or len(records) != len(
        expected
    ):
        raise ValueError(
            "prediction records do not equal the complete frozen inventory"
        )
    if len({row["recording_id"] for row in records}) != len(records):
        raise ValueError("prediction roster repeats a recording")
    candidates.sort(
        key=lambda row: (
            row["patient_uid"],
            row["recording_id"],
            row["start_offset_seconds"],
            row["origin"],
            row["candidate_id"],
        )
    )
    candidate_ids = [row["candidate_id"] for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("G0 A1 candidate IDs are not globally unique")
    outcome_counts = {
        outcome: sum(row["outcome"] == outcome for row in records)
        for outcome in BA_IEG_G0_A1_RECORD_OUTCOMES
    }
    origin_counts = {
        origin: sum(row["origin"] == origin for row in candidates)
        for origin in BA_IEG_G0_A1_CANDIDATE_ORIGINS
    }
    body = {
        "schema_version": BA_IEG_G0_A1_PREDICTION_ROSTER_SCHEMA_V1,
        "roster_id": "BAIEG-G0-A1-PREDICTIONS-PENDING",
        "model_split": "source_train",
        "inventory_receipt_sha256": source_inventory["receipt_sha256"],
        "fold_assignment_receipt_sha256": source_inventory[
            "fold_assignment_receipt_sha256"
        ],
        "provider_id": _identifier(provider_id, "provider ID"),
        "provider_prediction_receipt_sha256": _sha256(
            provider_prediction_receipt_sha256, "provider prediction receipt"
        ),
        "decoder_policy_receipt_sha256": _sha256(
            decoder_policy_receipt_sha256, "decoder policy receipt"
        ),
        "random_background_policy": random_policy,
        "records": records,
        "candidates": candidates,
        "counts": {
            "patients": len({row["patient_uid"] for row in records}),
            "records": len(records),
            "record_outcomes": outcome_counts,
            "candidates": len(candidates),
            "candidate_origins": origin_counts,
            "random_opportunities_skipped_record_too_short": short_random_opportunities,
        },
        "scope_receipt": {
            "patient_oof_inference_fold_equals_held_out_fold_set": True,
            "complete_inventory_retained": True,
            "zero_candidate_records_retained": True,
            "partial_coverage_records_retained": True,
            "technical_failures_retained": True,
            "random_background_selected_without_reference_or_detector_candidates": True,
            "prediction_freeze_before_target_join": True,
            "public_event_intervals_opened": 0,
            "edf_annotations_opened": 0,
            "channel_or_soz_targets_opened": 0,
            "private_doctor_or_clinical_text_opened": 0,
            "candidate_roster_embeds_raw_eeg_or_p0_payload": False,
            "content_addressed_signal_references_only": True,
            "bulk_materialization_storage_budget_receipt_available": False,
            "bulk_materialization_authorized": False,
            "training_authorized": False,
            "g0_promotion_authorized": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    result = _seal(body, id_field="roster_id", prefix="BAIEGG0A1ROSTER")
    validate_ba_ieg_g0_a1_prediction_roster_v1(result)
    return result


def validate_ba_ieg_g0_a1_prediction_roster_v1(payload: object) -> dict[str, Any]:
    fields = {
        "schema_version",
        "roster_id",
        "model_split",
        "inventory_receipt_sha256",
        "fold_assignment_receipt_sha256",
        "provider_id",
        "provider_prediction_receipt_sha256",
        "decoder_policy_receipt_sha256",
        "random_background_policy",
        "records",
        "candidates",
        "counts",
        "scope_receipt",
        "receipt_sha256",
    }
    data = _strict_object(payload, fields, "G0 A1 prediction roster")
    if (
        data["schema_version"] != BA_IEG_G0_A1_PREDICTION_ROSTER_SCHEMA_V1
        or data["model_split"] != "source_train"
    ):
        raise ValueError("G0 A1 prediction roster schema/split drifted")
    for field in (
        "inventory_receipt_sha256",
        "fold_assignment_receipt_sha256",
        "provider_prediction_receipt_sha256",
        "decoder_policy_receipt_sha256",
    ):
        _sha256(data[field], field)
    _identifier(data["provider_id"], "provider ID")
    policy = validate_ba_ieg_g0_a1_random_background_policy_v1(
        data["random_background_policy"]
    )
    if not isinstance(data["records"], list) or not data["records"]:
        raise ValueError("G0 A1 prediction roster has no records")
    record_fields = {
        "patient_uid",
        "recording_id",
        "recording_duration_seconds",
        "source_signal_sha256",
        "inference_fold_ids",
        "prediction_artifact_sha256",
        "prediction_result_receipt_sha256",
        "outcome",
        "failure_stage",
        "detector_candidate_ids",
        "random_background_candidate_ids",
    }
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(data["records"]):
        row = _strict_object(raw, record_fields, f"frozen prediction record {index}")
        duration = _finite(row["recording_duration_seconds"], "recording duration")
        if duration <= 0:
            raise ValueError("recording duration must be positive")
        outcome = row["outcome"]
        if outcome not in BA_IEG_G0_A1_RECORD_OUTCOMES:
            raise ValueError("frozen prediction record outcome drifted")
        if outcome == "technical_failure":
            _identifier(row["failure_stage"], "failure stage")
        elif row["failure_stage"] is not None:
            raise ValueError("non-failure frozen record has a failure stage")
        for key in ("detector_candidate_ids", "random_background_candidate_ids"):
            if not isinstance(row[key], list) or len(row[key]) != len(set(row[key])):
                raise ValueError(f"{key} must be a unique list")
            for item in row[key]:
                _identifier(item, key)
        records.append(
            {
                **row,
                "patient_uid": _identifier(row["patient_uid"], "patient UID"),
                "recording_id": _identifier(row["recording_id"], "recording ID"),
                "recording_duration_seconds": duration,
                "source_signal_sha256": _sha256(
                    row["source_signal_sha256"], "source signal"
                ),
                "inference_fold_ids": _fold_ids(
                    row["inference_fold_ids"], "inference folds"
                ),
                "prediction_artifact_sha256": _sha256(
                    row["prediction_artifact_sha256"], "prediction artifact"
                ),
                "prediction_result_receipt_sha256": _sha256(
                    row["prediction_result_receipt_sha256"], "prediction result receipt"
                ),
            }
        )
    if records != sorted(
        records, key=lambda row: (row["patient_uid"], row["recording_id"])
    ):
        raise ValueError("frozen prediction records are not canonically sorted")
    if len({row["recording_id"] for row in records}) != len(records):
        raise ValueError("frozen prediction records repeat a recording")
    candidate_fields = {
        "candidate_id",
        "patient_uid",
        "recording_id",
        "origin",
        "start_offset_seconds",
        "stop_offset_seconds",
        "anchor_offset_seconds",
        "score",
        "decision_available_offset_seconds",
        "source_candidate_receipt_sha256",
        "random_policy_receipt_sha256",
    }
    record_by_id = {row["recording_id"]: row for row in records}
    candidates: list[dict[str, Any]] = []
    for index, raw in enumerate(data["candidates"]):
        row = _strict_object(raw, candidate_fields, f"frozen candidate {index}")
        recording = _identifier(row["recording_id"], "candidate recording ID")
        record = record_by_id.get(recording)
        if record is None or row["patient_uid"] != record["patient_uid"]:
            raise ValueError("frozen candidate crosses record/patient identity")
        origin = row["origin"]
        if origin not in BA_IEG_G0_A1_CANDIDATE_ORIGINS:
            raise ValueError("frozen candidate origin drifted")
        start = _finite(row["start_offset_seconds"], "candidate start")
        stop = _finite(row["stop_offset_seconds"], "candidate stop")
        anchor = _finite(row["anchor_offset_seconds"], "candidate anchor")
        available = _finite(
            row["decision_available_offset_seconds"], "candidate decision availability"
        )
        duration = float(record["recording_duration_seconds"])
        if (
            start < 0
            or stop <= start
            or stop > duration
            or anchor < start
            or anchor > stop
            or available < stop
            or available > duration
        ):
            raise ValueError("frozen candidate time support is invalid")
        if origin == "detector_proposal":
            _finite(row["score"], "detector candidate score")
            if row["random_policy_receipt_sha256"] is not None:
                raise ValueError(
                    "detector proposal cannot carry a random policy receipt"
                )
        else:
            if row["score"] is not None:
                raise ValueError("random background cannot carry detector confidence")
            if row["random_policy_receipt_sha256"] != policy["receipt_sha256"]:
                raise ValueError("random background policy binding drifted")
        _sha256(row["source_candidate_receipt_sha256"], "source candidate receipt")
        candidates.append(row)
    expected_order = sorted(
        candidates,
        key=lambda row: (
            row["patient_uid"],
            row["recording_id"],
            row["start_offset_seconds"],
            row["origin"],
            row["candidate_id"],
        ),
    )
    if candidates != expected_order:
        raise ValueError("frozen candidates are not canonically sorted")
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise ValueError("frozen candidate IDs are not globally unique")
    candidates_by_record: dict[str, dict[str, list[str]]] = {
        row["recording_id"]: {origin: [] for origin in BA_IEG_G0_A1_CANDIDATE_ORIGINS}
        for row in records
    }
    for row in candidates:
        candidates_by_record[row["recording_id"]][row["origin"]].append(
            row["candidate_id"]
        )
    for record in records:
        observed = candidates_by_record[record["recording_id"]]
        if record["detector_candidate_ids"] != observed["detector_proposal"]:
            raise ValueError("record detector candidate roster does not replay")
        if (
            record["random_background_candidate_ids"]
            != observed["candidate_blind_random_background"]
        ):
            raise ValueError("record random candidate roster does not replay")
        detector_count = len(observed["detector_proposal"])
        if record["outcome"] == "completed_with_candidates" and detector_count == 0:
            raise ValueError(
                "completed-with-candidates record lost its detector candidates"
            )
        if (
            record["outcome"] in {"completed_zero_candidate", "technical_failure"}
            and detector_count
        ):
            raise ValueError("zero/failure record acquired detector candidates")
        expected_random, skipped = _random_candidates(record, policy)
        observed_random = [
            row
            for row in candidates
            if row["recording_id"] == record["recording_id"]
            and row["origin"] == "candidate_blind_random_background"
        ]
        if observed_random != expected_random:
            raise ValueError(
                "reference-blind random candidates do not replay from policy"
            )
    outcome_counts = {
        outcome: sum(row["outcome"] == outcome for row in records)
        for outcome in BA_IEG_G0_A1_RECORD_OUTCOMES
    }
    origin_counts = {
        origin: sum(row["origin"] == origin for row in candidates)
        for origin in BA_IEG_G0_A1_CANDIDATE_ORIGINS
    }
    skipped_count = sum(_random_candidates(row, policy)[1] for row in records)
    expected_counts = {
        "patients": len({row["patient_uid"] for row in records}),
        "records": len(records),
        "record_outcomes": outcome_counts,
        "candidates": len(candidates),
        "candidate_origins": origin_counts,
        "random_opportunities_skipped_record_too_short": skipped_count,
    }
    if data["counts"] != expected_counts:
        raise ValueError("G0 A1 prediction roster counts do not replay")
    expected_scope = {
        "patient_oof_inference_fold_equals_held_out_fold_set": True,
        "complete_inventory_retained": True,
        "zero_candidate_records_retained": True,
        "partial_coverage_records_retained": True,
        "technical_failures_retained": True,
        "random_background_selected_without_reference_or_detector_candidates": True,
        "prediction_freeze_before_target_join": True,
        "public_event_intervals_opened": 0,
        "edf_annotations_opened": 0,
        "channel_or_soz_targets_opened": 0,
        "private_doctor_or_clinical_text_opened": 0,
        "candidate_roster_embeds_raw_eeg_or_p0_payload": False,
        "content_addressed_signal_references_only": True,
        "bulk_materialization_storage_budget_receipt_available": False,
        "bulk_materialization_authorized": False,
        "training_authorized": False,
        "g0_promotion_authorized": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("G0 A1 prediction roster firewall/authority drifted")
    _replay_seal(
        data,
        id_field="roster_id",
        prefix="BAIEGG0A1ROSTER",
        context="prediction roster",
    )
    return data


def build_ba_ieg_g0_a1_match_policy_v1(
    *,
    minimum_temporal_iou: float,
    maximum_anchor_to_onset_seconds: float,
    near_event_margin_seconds: float,
) -> dict[str, Any]:
    iou = _finite(minimum_temporal_iou, "minimum temporal IoU")
    onset = _finite(maximum_anchor_to_onset_seconds, "anchor-onset tolerance")
    near = _finite(near_event_margin_seconds, "near-event margin")
    if not 0 < iou <= 1 or onset < 0 or near < 0:
        raise ValueError("post-freeze matching thresholds are invalid")
    body = {
        "schema_version": BA_IEG_G0_A1_MATCH_POLICY_SCHEMA_V1,
        "policy_id": "BAIEG-G0-A1-MATCH-PENDING",
        "minimum_temporal_iou": iou,
        "maximum_anchor_to_onset_seconds": onset,
        "near_event_margin_seconds": near,
        "positive_edge_rule": "temporal_iou_or_anchor_to_reference_onset",
        "one_to_one_rule": (
            "greedy_iou_desc_anchor_error_asc_candidate_id_event_id_without_detector_score"
        ),
        "unmatched_overlap_or_positive_edge": "fragmented_or_duplicate_hard_candidate",
        "random_collision_is_assigned_by_reference_relation_not_forced_negative": True,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    result = _seal(body, id_field="policy_id", prefix="BAIEGG0A1MATCH")
    validate_ba_ieg_g0_a1_match_policy_v1(result)
    return result


def validate_ba_ieg_g0_a1_match_policy_v1(payload: object) -> dict[str, Any]:
    fields = {
        "schema_version",
        "policy_id",
        "minimum_temporal_iou",
        "maximum_anchor_to_onset_seconds",
        "near_event_margin_seconds",
        "positive_edge_rule",
        "one_to_one_rule",
        "unmatched_overlap_or_positive_edge",
        "random_collision_is_assigned_by_reference_relation_not_forced_negative",
        "receipt_sha256",
    }
    data = _strict_object(payload, fields, "G0 A1 match policy")
    if data["schema_version"] != BA_IEG_G0_A1_MATCH_POLICY_SCHEMA_V1:
        raise ValueError("G0 A1 match policy schema drifted")
    if not 0 < _finite(data["minimum_temporal_iou"], "minimum IoU") <= 1:
        raise ValueError("minimum IoU is invalid")
    if _finite(data["maximum_anchor_to_onset_seconds"], "onset tolerance") < 0:
        raise ValueError("onset tolerance is invalid")
    if _finite(data["near_event_margin_seconds"], "near-event margin") < 0:
        raise ValueError("near-event margin is invalid")
    if (
        data["positive_edge_rule"] != "temporal_iou_or_anchor_to_reference_onset"
        or data["one_to_one_rule"]
        != "greedy_iou_desc_anchor_error_asc_candidate_id_event_id_without_detector_score"
        or data["unmatched_overlap_or_positive_edge"]
        != "fragmented_or_duplicate_hard_candidate"
        or data[
            "random_collision_is_assigned_by_reference_relation_not_forced_negative"
        ]
        is not True
    ):
        raise ValueError("G0 A1 match-policy semantics drifted")
    _replay_seal(
        data, id_field="policy_id", prefix="BAIEGG0A1MATCH", context="match policy"
    )
    return data


def _normalize_reference_record(value: object, index: int) -> dict[str, Any]:
    row = _strict_object(value, _REFERENCE_RECORD_FIELDS, f"reference record {index}")
    duration = _finite(row["recording_duration_seconds"], "reference duration")
    if duration <= 0:
        raise ValueError("reference duration must be positive")
    coverage = row["reference_coverage_status"]
    if coverage not in {"complete_recording", "incomplete"}:
        raise ValueError("reference coverage status is unsupported")
    resolution = _finite(
        row["annotation_timestamp_resolution_seconds"], "reference resolution"
    )
    if resolution <= 0:
        raise ValueError("reference timestamp resolution must be positive")
    if not isinstance(row["seizure_intervals"], list):
        raise TypeError("reference seizure intervals must be a list")
    intervals: list[dict[str, Any]] = []
    previous_stop = -math.inf
    for event_index, raw in enumerate(row["seizure_intervals"]):
        event = _strict_object(
            raw, _REFERENCE_EVENT_FIELDS, f"reference event {index}:{event_index}"
        )
        onset = _finite(event["onset_recording_seconds"], "reference onset")
        offset = _finite(event["offset_recording_seconds"], "reference offset")
        if onset < 0 or offset <= onset or offset > duration or onset < previous_stop:
            raise ValueError(
                "reference intervals are invalid, overlapping, or unsorted"
            )
        intervals.append(
            {
                "public_event_id": _identifier(
                    event["public_event_id"], "public event ID"
                ),
                "onset_recording_seconds": onset,
                "offset_recording_seconds": offset,
            }
        )
        previous_stop = offset
    if len({event["public_event_id"] for event in intervals}) != len(intervals):
        raise ValueError("reference record repeats a public event ID")
    return {
        "patient_uid": _identifier(row["patient_uid"], "reference patient UID"),
        "recording_id": _identifier(row["recording_id"], "reference recording ID"),
        "recording_duration_seconds": duration,
        "reference_coverage_status": coverage,
        "annotation_timestamp_resolution_seconds": resolution,
        "source_reference_receipt_sha256": _sha256(
            row["source_reference_receipt_sha256"], "source reference receipt"
        ),
        "seizure_intervals": intervals,
    }


def build_ba_ieg_g0_a1_reference_roster_v1(
    records: Sequence[Mapping[str, Any]],
    *,
    prediction_roster_receipt_sha256: str,
) -> dict[str, Any]:
    """Seal the exact post-freeze public interval denominator."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("reference records must be a sequence")
    rows = [
        _normalize_reference_record(dict(row), index)
        for index, row in enumerate(records)
    ]
    if not rows:
        raise ValueError("reference roster must contain at least one record")
    rows.sort(key=lambda row: (row["patient_uid"], row["recording_id"]))
    if len({row["recording_id"] for row in rows}) != len(rows):
        raise ValueError("reference roster repeats a recording")
    body = {
        "schema_version": BA_IEG_G0_A1_REFERENCE_ROSTER_SCHEMA_V1,
        "reference_roster_id": "BAIEG-G0-A1-REFERENCE-PENDING",
        "prediction_roster_receipt_sha256": _sha256(
            prediction_roster_receipt_sha256, "prediction roster receipt"
        ),
        "records": rows,
        "counts": {
            "records": len(rows),
            "complete_reference_records": sum(
                row["reference_coverage_status"] == "complete_recording" for row in rows
            ),
            "incomplete_reference_records": sum(
                row["reference_coverage_status"] == "incomplete" for row in rows
            ),
            "public_events": sum(len(row["seizure_intervals"]) for row in rows),
        },
        "scope_receipt": {
            "global_public_event_intervals_only": True,
            "channel_or_soz_targets_opened": 0,
            "edf_annotations_opened": 0,
            "private_doctor_or_clinical_text_opened": 0,
            "reference_join_is_post_prediction_freeze": True,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    result = _seal(body, id_field="reference_roster_id", prefix="BAIEGG0A1REF")
    validate_ba_ieg_g0_a1_reference_roster_v1(result)
    return result


def validate_ba_ieg_g0_a1_reference_roster_v1(payload: object) -> dict[str, Any]:
    fields = {
        "schema_version",
        "reference_roster_id",
        "prediction_roster_receipt_sha256",
        "records",
        "counts",
        "scope_receipt",
        "receipt_sha256",
    }
    data = _strict_object(payload, fields, "G0 A1 reference roster")
    if data["schema_version"] != BA_IEG_G0_A1_REFERENCE_ROSTER_SCHEMA_V1:
        raise ValueError("G0 A1 reference roster schema drifted")
    _sha256(data["prediction_roster_receipt_sha256"], "prediction roster receipt")
    if not isinstance(data["records"], list) or not data["records"]:
        raise ValueError("G0 A1 reference roster has no records")
    rows = [
        _normalize_reference_record(row, index)
        for index, row in enumerate(data["records"])
    ]
    if rows != sorted(rows, key=lambda row: (row["patient_uid"], row["recording_id"])):
        raise ValueError("G0 A1 reference roster is not canonically sorted")
    if len({row["recording_id"] for row in rows}) != len(rows):
        raise ValueError("G0 A1 reference roster repeats a recording")
    expected_counts = {
        "records": len(rows),
        "complete_reference_records": sum(
            row["reference_coverage_status"] == "complete_recording" for row in rows
        ),
        "incomplete_reference_records": sum(
            row["reference_coverage_status"] == "incomplete" for row in rows
        ),
        "public_events": sum(len(row["seizure_intervals"]) for row in rows),
    }
    if data["counts"] != expected_counts:
        raise ValueError("G0 A1 reference roster counts do not replay")
    if data["scope_receipt"] != {
        "global_public_event_intervals_only": True,
        "channel_or_soz_targets_opened": 0,
        "edf_annotations_opened": 0,
        "private_doctor_or_clinical_text_opened": 0,
        "reference_join_is_post_prediction_freeze": True,
    }:
        raise ValueError("G0 A1 reference roster firewall drifted")
    _replay_seal(
        data,
        id_field="reference_roster_id",
        prefix="BAIEGG0A1REF",
        context="reference roster",
    )
    return data


def _pair_metrics(
    candidate: Mapping[str, Any], event: Mapping[str, Any]
) -> dict[str, float]:
    start = float(candidate["start_offset_seconds"])
    stop = float(candidate["stop_offset_seconds"])
    onset = float(event["onset_recording_seconds"])
    offset = float(event["offset_recording_seconds"])
    intersection = max(0.0, min(stop, offset) - max(start, onset))
    union = max(stop, offset) - min(start, onset)
    iou = 0.0 if union <= 0 else intersection / union
    if stop < onset:
        gap = onset - stop
    elif offset < start:
        gap = start - offset
    else:
        gap = 0.0
    return {
        "intersection_seconds": intersection,
        "temporal_iou": iou,
        "anchor_to_onset_seconds": abs(
            float(candidate["anchor_offset_seconds"]) - onset
        ),
        "interval_gap_seconds": gap,
    }


def _classify_record_candidates(
    candidates: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[str]]:
    pair_rows: list[tuple[float, float, str, str, dict[str, float]]] = []
    all_metrics: dict[tuple[str, str], dict[str, float]] = {}
    for candidate in candidates:
        for event in events:
            metrics = _pair_metrics(candidate, event)
            key = (candidate["candidate_id"], event["public_event_id"])
            all_metrics[key] = metrics
            eligible = metrics["temporal_iou"] >= float(
                policy["minimum_temporal_iou"]
            ) or metrics["anchor_to_onset_seconds"] <= float(
                policy["maximum_anchor_to_onset_seconds"]
            )
            # Random supports are training controls, not detector proposals.
            # A chance collision may receive a positive training relation, but
            # it must never consume a one-to-one detector/reference match or
            # improve the detector event-sensitivity denominator.
            if eligible and candidate["origin"] == "detector_proposal":
                pair_rows.append(
                    (
                        -metrics["temporal_iou"],
                        metrics["anchor_to_onset_seconds"],
                        candidate["candidate_id"],
                        event["public_event_id"],
                        metrics,
                    )
                )
    matched_candidates: set[str] = set()
    matched_events: set[str] = set()
    match_by_candidate: dict[str, tuple[str, dict[str, float]]] = {}
    for _, _, candidate_id, event_id, metrics in sorted(pair_rows):
        if candidate_id in matched_candidates or event_id in matched_events:
            continue
        matched_candidates.add(candidate_id)
        matched_events.add(event_id)
        match_by_candidate[candidate_id] = (event_id, metrics)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        matched = match_by_candidate.get(candidate_id)
        if matched is not None:
            event_id, metrics = matched
            training_class = "matched_true_event"
        else:
            metrics_by_event = [
                (event, all_metrics[(candidate_id, event["public_event_id"])])
                for event in events
            ]
            eligible_or_overlap = [
                (event, metric)
                for event, metric in metrics_by_event
                if metric["intersection_seconds"] > 0
                or metric["temporal_iou"] >= float(policy["minimum_temporal_iou"])
                or metric["anchor_to_onset_seconds"]
                <= float(policy["maximum_anchor_to_onset_seconds"])
            ]
            positive_edges = [
                (event, metric)
                for event, metric in metrics_by_event
                if metric["temporal_iou"] >= float(policy["minimum_temporal_iou"])
                or metric["anchor_to_onset_seconds"]
                <= float(policy["maximum_anchor_to_onset_seconds"])
            ]
            if (
                candidate["origin"] == "candidate_blind_random_background"
                and positive_edges
            ):
                event, metrics = min(
                    positive_edges,
                    key=lambda item: (
                        -item[1]["temporal_iou"],
                        item[1]["anchor_to_onset_seconds"],
                        item[0]["public_event_id"],
                    ),
                )
                event_id = event["public_event_id"]
                training_class = "matched_true_event"
            elif eligible_or_overlap:
                event, metrics = min(
                    eligible_or_overlap,
                    key=lambda item: (
                        -item[1]["temporal_iou"],
                        item[1]["anchor_to_onset_seconds"],
                        item[0]["public_event_id"],
                    ),
                )
                event_id = event["public_event_id"]
                training_class = "fragmented_or_duplicate_hard_candidate"
            elif metrics_by_event:
                event, metrics = min(
                    metrics_by_event,
                    key=lambda item: (
                        item[1]["interval_gap_seconds"],
                        item[1]["anchor_to_onset_seconds"],
                        item[0]["public_event_id"],
                    ),
                )
                event_id = event["public_event_id"]
                if metrics["interval_gap_seconds"] <= float(
                    policy["near_event_margin_seconds"]
                ):
                    training_class = "near_event_hard_candidate"
                else:
                    event_id = None
                    training_class = (
                        "candidate_blind_random_background"
                        if candidate["origin"] == "candidate_blind_random_background"
                        else "unmatched_false_candidate"
                    )
            else:
                event_id = None
                metrics = {
                    "intersection_seconds": 0.0,
                    "temporal_iou": 0.0,
                    "anchor_to_onset_seconds": None,
                    "interval_gap_seconds": None,
                }
                training_class = (
                    "candidate_blind_random_background"
                    if candidate["origin"] == "candidate_blind_random_background"
                    else "unmatched_false_candidate"
                )
        rows.append(
            {
                **deepcopy(dict(candidate)),
                "target_status": "evaluable_complete_reference",
                "training_class": training_class,
                "matched_or_nearest_public_event_id": event_id,
                "intersection_seconds": metrics["intersection_seconds"],
                "temporal_iou": metrics["temporal_iou"],
                "anchor_to_onset_seconds": metrics["anchor_to_onset_seconds"],
                "interval_gap_seconds": metrics["interval_gap_seconds"],
            }
        )
    return rows, matched_events


def build_ba_ieg_g0_a1_postfreeze_target_join_v1(
    *,
    prediction_roster: Mapping[str, Any],
    reference_roster: Mapping[str, Any],
    match_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Join public global intervals only after the complete prediction freeze."""

    predictions = validate_ba_ieg_g0_a1_prediction_roster_v1(dict(prediction_roster))
    references = validate_ba_ieg_g0_a1_reference_roster_v1(dict(reference_roster))
    policy = validate_ba_ieg_g0_a1_match_policy_v1(dict(match_policy))
    if references["prediction_roster_receipt_sha256"] != predictions["receipt_sha256"]:
        raise ValueError("reference roster is not bound to this prediction freeze")
    prediction_records = {row["recording_id"]: row for row in predictions["records"]}
    reference_records = {row["recording_id"]: row for row in references["records"]}
    if set(prediction_records) != set(reference_records):
        raise ValueError(
            "post-freeze reference records do not equal prediction denominator"
        )
    candidates_by_record: dict[str, list[dict[str, Any]]] = {
        recording_id: [] for recording_id in prediction_records
    }
    for row in predictions["candidates"]:
        candidates_by_record[row["recording_id"]].append(row)
    candidate_targets: list[dict[str, Any]] = []
    record_denominator: list[dict[str, Any]] = []
    matched_event_ids: set[str] = set()
    all_event_ids: set[str] = set()
    for recording_id in sorted(prediction_records):
        prediction = prediction_records[recording_id]
        reference = reference_records[recording_id]
        if (
            prediction["patient_uid"] != reference["patient_uid"]
            or abs(
                float(prediction["recording_duration_seconds"])
                - float(reference["recording_duration_seconds"])
            )
            > 1e-9
        ):
            raise ValueError(
                "post-freeze reference crosses patient/record time identity"
            )
        events = reference["seizure_intervals"]
        namespaced_events = {
            recording_id + "::" + event["public_event_id"] for event in events
        }
        if all_event_ids.intersection(namespaced_events):
            raise ValueError("post-freeze reference repeats a namespaced public event")
        all_event_ids.update(namespaced_events)
        candidates = candidates_by_record[recording_id]
        evaluable = (
            prediction["outcome"]
            in {"completed_with_candidates", "completed_zero_candidate"}
            and reference["reference_coverage_status"] == "complete_recording"
        )
        if evaluable:
            classified, matched = _classify_record_candidates(
                candidates, events, policy
            )
            matched_event_ids.update(recording_id + "::" + item for item in matched)
            candidate_targets.extend(classified)
            record_matched = len(matched)
        else:
            candidate_targets.extend(
                {
                    **deepcopy(candidate),
                    "target_status": "not_evaluable_prediction_or_reference_coverage",
                    "training_class": None,
                    "matched_or_nearest_public_event_id": None,
                    "intersection_seconds": None,
                    "temporal_iou": None,
                    "anchor_to_onset_seconds": None,
                    "interval_gap_seconds": None,
                }
                for candidate in candidates
            )
            record_matched = 0
        record_denominator.append(
            {
                "patient_uid": prediction["patient_uid"],
                "recording_id": recording_id,
                "prediction_outcome": prediction["outcome"],
                "reference_coverage_status": reference["reference_coverage_status"],
                "detector_candidate_count": len(prediction["detector_candidate_ids"]),
                "random_background_candidate_count": len(
                    prediction["random_background_candidate_ids"]
                ),
                "public_event_count": len(events),
                "matched_public_event_count": record_matched,
                "missed_public_event_count": (
                    len(events) - record_matched if evaluable else None
                ),
                "candidate_target_evaluable": evaluable,
            }
        )
    candidate_targets.sort(
        key=lambda row: (
            row["patient_uid"],
            row["recording_id"],
            row["start_offset_seconds"],
            row["origin"],
            row["candidate_id"],
        )
    )
    class_counts = {
        name: sum(row["training_class"] == name for row in candidate_targets)
        for name in BA_IEG_G0_A1_TRAINING_CLASSES
    }
    not_evaluable_candidates = sum(
        row["training_class"] is None for row in candidate_targets
    )
    evaluable_events = sum(
        row["public_event_count"]
        for row in record_denominator
        if row["candidate_target_evaluable"]
    )
    matched_events = sum(
        row["matched_public_event_count"]
        for row in record_denominator
        if row["candidate_target_evaluable"]
    )
    body = {
        "schema_version": BA_IEG_G0_A1_TARGET_JOIN_SCHEMA_V1,
        "join_id": "BAIEG-G0-A1-JOIN-PENDING",
        "prediction_roster_receipt_sha256": predictions["receipt_sha256"],
        "reference_roster_receipt_sha256": references["receipt_sha256"],
        "match_policy": policy,
        "record_denominator": record_denominator,
        "candidate_targets": candidate_targets,
        "counts": {
            "patients": predictions["counts"]["patients"],
            "records": len(record_denominator),
            "candidates": len(candidate_targets),
            "candidate_training_classes": class_counts,
            "not_evaluable_candidates": not_evaluable_candidates,
            "evaluable_public_events": evaluable_events,
            "all_public_events": sum(
                row["public_event_count"] for row in record_denominator
            ),
            "public_events_on_non_evaluable_records": sum(
                row["public_event_count"]
                for row in record_denominator
                if not row["candidate_target_evaluable"]
            ),
            "matched_public_events": matched_events,
            "missed_public_events": evaluable_events - matched_events,
            "zero_detector_candidate_records": sum(
                row["prediction_outcome"] == "completed_zero_candidate"
                for row in record_denominator
            ),
            "technical_failure_records": sum(
                row["prediction_outcome"] == "technical_failure"
                for row in record_denominator
            ),
        },
        "scope_receipt": {
            "prediction_roster_frozen_before_reference_join": True,
            "candidate_selection_or_score_changed_by_reference": False,
            "random_origin_not_assumed_negative": True,
            "complete_record_denominator_retained": True,
            "zero_candidate_and_failure_records_retained": True,
            "global_public_event_intervals_only": True,
            "channel_or_soz_targets_opened": 0,
            "edf_annotations_opened": 0,
            "private_doctor_or_clinical_text_opened": 0,
            "training_authorized": False,
            "g0_promotion_authorized": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    result = _seal(body, id_field="join_id", prefix="BAIEGG0A1JOIN")
    validate_ba_ieg_g0_a1_postfreeze_target_join_v1(result)
    return result


def validate_ba_ieg_g0_a1_postfreeze_target_join_v1(payload: object) -> dict[str, Any]:
    fields = {
        "schema_version",
        "join_id",
        "prediction_roster_receipt_sha256",
        "reference_roster_receipt_sha256",
        "match_policy",
        "record_denominator",
        "candidate_targets",
        "counts",
        "scope_receipt",
        "receipt_sha256",
    }
    data = _strict_object(payload, fields, "G0 A1 post-freeze target join")
    if data["schema_version"] != BA_IEG_G0_A1_TARGET_JOIN_SCHEMA_V1:
        raise ValueError("G0 A1 target join schema drifted")
    _sha256(data["prediction_roster_receipt_sha256"], "prediction roster receipt")
    _sha256(data["reference_roster_receipt_sha256"], "reference roster receipt")
    validate_ba_ieg_g0_a1_match_policy_v1(data["match_policy"])
    if (
        not isinstance(data["record_denominator"], list)
        or not data["record_denominator"]
    ):
        raise ValueError("G0 A1 target join has no record denominator")
    record_fields = {
        "patient_uid",
        "recording_id",
        "prediction_outcome",
        "reference_coverage_status",
        "detector_candidate_count",
        "random_background_candidate_count",
        "public_event_count",
        "matched_public_event_count",
        "missed_public_event_count",
        "candidate_target_evaluable",
    }
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(data["record_denominator"]):
        row = _strict_object(raw, record_fields, f"joined record denominator {index}")
        _identifier(row["patient_uid"], "joined patient UID")
        _identifier(row["recording_id"], "joined recording ID")
        if row["prediction_outcome"] not in BA_IEG_G0_A1_RECORD_OUTCOMES:
            raise ValueError("joined prediction outcome drifted")
        if row["reference_coverage_status"] not in {"complete_recording", "incomplete"}:
            raise ValueError("joined reference coverage status drifted")
        for name in (
            "detector_candidate_count",
            "random_background_candidate_count",
            "public_event_count",
            "matched_public_event_count",
        ):
            if (
                isinstance(row[name], bool)
                or not isinstance(row[name], int)
                or row[name] < 0
            ):
                raise TypeError(f"{name} must be a non-negative integer")
        if type(row["candidate_target_evaluable"]) is not bool:
            raise TypeError("candidate_target_evaluable must be boolean")
        if row["candidate_target_evaluable"]:
            if (
                not isinstance(row["missed_public_event_count"], int)
                or row["missed_public_event_count"] < 0
                or row["matched_public_event_count"] + row["missed_public_event_count"]
                != row["public_event_count"]
            ):
                raise ValueError(
                    "joined matched/missed event denominator is inconsistent"
                )
        elif row["missed_public_event_count"] is not None:
            raise ValueError("non-evaluable record cannot claim a missed-event count")
        records.append(row)
    if records != sorted(records, key=lambda row: row["recording_id"]):
        raise ValueError("joined record denominator is not canonically sorted")
    if len({row["recording_id"] for row in records}) != len(records):
        raise ValueError("joined record denominator repeats a recording")
    candidate_fields = {
        "candidate_id",
        "patient_uid",
        "recording_id",
        "origin",
        "start_offset_seconds",
        "stop_offset_seconds",
        "anchor_offset_seconds",
        "score",
        "decision_available_offset_seconds",
        "source_candidate_receipt_sha256",
        "random_policy_receipt_sha256",
        "target_status",
        "training_class",
        "matched_or_nearest_public_event_id",
        "intersection_seconds",
        "temporal_iou",
        "anchor_to_onset_seconds",
        "interval_gap_seconds",
    }
    record_ids = {row["recording_id"] for row in records}
    targets: list[dict[str, Any]] = []
    for index, raw in enumerate(data["candidate_targets"]):
        row = _strict_object(raw, candidate_fields, f"joined candidate target {index}")
        if row["recording_id"] not in record_ids:
            raise ValueError("joined candidate target is outside record denominator")
        if row["origin"] not in BA_IEG_G0_A1_CANDIDATE_ORIGINS:
            raise ValueError("joined candidate origin drifted")
        training_class = row["training_class"]
        if (
            training_class is not None
            and training_class not in BA_IEG_G0_A1_TRAINING_CLASSES
        ):
            raise ValueError("joined candidate training class drifted")
        if training_class is None:
            if row["target_status"] != "not_evaluable_prediction_or_reference_coverage":
                raise ValueError("null candidate target lacks not-evaluable status")
            if any(
                row[name] is not None
                for name in (
                    "matched_or_nearest_public_event_id",
                    "intersection_seconds",
                    "temporal_iou",
                    "anchor_to_onset_seconds",
                    "interval_gap_seconds",
                )
            ):
                raise ValueError(
                    "not-evaluable candidate cannot carry reference metrics"
                )
        elif row["target_status"] != "evaluable_complete_reference":
            raise ValueError("evaluable candidate target status drifted")
        targets.append(row)
    expected_target_order = sorted(
        targets,
        key=lambda row: (
            row["patient_uid"],
            row["recording_id"],
            row["start_offset_seconds"],
            row["origin"],
            row["candidate_id"],
        ),
    )
    if targets != expected_target_order:
        raise ValueError("joined candidate targets are not canonically sorted")
    class_counts = {
        name: sum(row["training_class"] == name for row in targets)
        for name in BA_IEG_G0_A1_TRAINING_CLASSES
    }
    evaluable_events = sum(
        row["public_event_count"]
        for row in records
        if row["candidate_target_evaluable"]
    )
    matched_events = sum(
        row["matched_public_event_count"]
        for row in records
        if row["candidate_target_evaluable"]
    )
    expected_counts = {
        "patients": len({row["patient_uid"] for row in records}),
        "records": len(records),
        "candidates": len(targets),
        "candidate_training_classes": class_counts,
        "not_evaluable_candidates": sum(
            row["training_class"] is None for row in targets
        ),
        "evaluable_public_events": evaluable_events,
        "all_public_events": sum(row["public_event_count"] for row in records),
        "public_events_on_non_evaluable_records": sum(
            row["public_event_count"]
            for row in records
            if not row["candidate_target_evaluable"]
        ),
        "matched_public_events": matched_events,
        "missed_public_events": evaluable_events - matched_events,
        "zero_detector_candidate_records": sum(
            row["prediction_outcome"] == "completed_zero_candidate" for row in records
        ),
        "technical_failure_records": sum(
            row["prediction_outcome"] == "technical_failure" for row in records
        ),
    }
    if data["counts"] != expected_counts:
        raise ValueError("G0 A1 target-join counts do not replay")
    expected_scope = {
        "prediction_roster_frozen_before_reference_join": True,
        "candidate_selection_or_score_changed_by_reference": False,
        "random_origin_not_assumed_negative": True,
        "complete_record_denominator_retained": True,
        "zero_candidate_and_failure_records_retained": True,
        "global_public_event_intervals_only": True,
        "channel_or_soz_targets_opened": 0,
        "edf_annotations_opened": 0,
        "private_doctor_or_clinical_text_opened": 0,
        "training_authorized": False,
        "g0_promotion_authorized": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("G0 A1 target-join firewall/authority drifted")
    _replay_seal(
        data, id_field="join_id", prefix="BAIEGG0A1JOIN", context="target join"
    )
    return data


__all__ = [
    "BA_IEG_G0_A1_INVENTORY_SCHEMA_V1",
    "BA_IEG_G0_A1_RANDOM_POLICY_SCHEMA_V1",
    "BA_IEG_G0_A1_PREDICTION_ROSTER_SCHEMA_V1",
    "BA_IEG_G0_A1_MATCH_POLICY_SCHEMA_V1",
    "BA_IEG_G0_A1_REFERENCE_ROSTER_SCHEMA_V1",
    "BA_IEG_G0_A1_TARGET_JOIN_SCHEMA_V1",
    "BA_IEG_G0_A1_RECORD_OUTCOMES",
    "BA_IEG_G0_A1_CANDIDATE_ORIGINS",
    "BA_IEG_G0_A1_TRAINING_CLASSES",
    "build_ba_ieg_g0_a1_oof_inventory_v1",
    "validate_ba_ieg_g0_a1_oof_inventory_v1",
    "build_ba_ieg_g0_a1_random_background_policy_v1",
    "validate_ba_ieg_g0_a1_random_background_policy_v1",
    "build_ba_ieg_g0_a1_prediction_roster_v1",
    "validate_ba_ieg_g0_a1_prediction_roster_v1",
    "build_ba_ieg_g0_a1_match_policy_v1",
    "validate_ba_ieg_g0_a1_match_policy_v1",
    "build_ba_ieg_g0_a1_reference_roster_v1",
    "validate_ba_ieg_g0_a1_reference_roster_v1",
    "build_ba_ieg_g0_a1_postfreeze_target_join_v1",
    "validate_ba_ieg_g0_a1_postfreeze_target_join_v1",
]
