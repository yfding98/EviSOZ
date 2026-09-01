"""Recording-level, EEG-only non-event context card shadow contract.

This module is deliberately separate from the repeatable event Findings card.
It materializes exactly one context card per physical recording and closes six
record-level questions: evaluable non-event opportunity, background spectrum,
regional slowing, persistent asymmetry/attenuation, interictal sharp/IED
candidate burden, and concordance/discordance with already-existing onset
modes.  The card cannot create an event, onset/SOZ evidence, change a mode
ranking, promote report text, read private labels, or invoke Qwen.  The EEG
signal itself may come from a private, public, or synthetic domain; that
signal-domain permission never permits private labels, annotations, clinical
text, or other patient-side information.

The implementation performs no dataset I/O.  It accepts only content-bound,
host-constructed evidence bindings and returns a deterministic JSON object.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator


RECORD_NON_EVENT_CONTEXT_CARD_POLICY_SCHEMA_VERSION_V1 = (
    "clinical_eeg_record_non_event_context_card_policy_v1"
)
RECORD_NON_EVENT_CONTEXT_CARD_SCHEMA_VERSION_V1 = (
    "clinical_eeg_record_non_event_context_card_v1"
)
RECORD_NON_EVENT_CONTEXT_CARD_POLICY_ID_V1 = (
    "CLINICAL-EEG-RECORD-NON-EVENT-CONTEXT-CARD-POLICY-V1"
)
DEFAULT_RECORD_NON_EVENT_CONTEXT_CARD_POLICY_SHA256_V1 = (
    "4cd26cd8cdff78ce99533f8c6072c3c61d398fb7ca3db5b20bcb3f7d40c1e4e6"
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORD_NON_EVENT_CONTEXT_CARD_POLICY_PATH_V1 = (
    _ROOT / "configs" / "clinical_eeg_record_non_event_context_card_policy_v1.json"
)
RECORD_NON_EVENT_CONTEXT_CARD_POLICY_SCHEMA_PATH_V1 = (
    _ROOT
    / "schemas"
    / "clinical_eeg_record_non_event_context_card_policy_v1.schema.json"
)
RECORD_NON_EVENT_CONTEXT_CARD_SCHEMA_PATH_V1 = (
    _ROOT / "schemas" / "clinical_eeg_record_non_event_context_card_v1.schema.json"
)

_SLOT_IDS = (
    "RNE01_OPPORTUNITY_PROTECTION_QC",
    "RNE02_BACKGROUND_SPECTRAL_SPATIAL",
    "RNE03_FOCAL_REGIONAL_SLOWING",
    "RNE04_PERSISTENT_ASYMMETRY_ATTENUATION",
    "RNE05_INTERICTAL_IED_BURDEN",
    "RNE06_MODE_CONCORDANCE_DISCORDANCE",
)
_STATUS_VOCABULARY = (
    "present",
    "absent_with_opportunity",
    "uncertain",
    "not_evaluable",
)
_ASSERTION_VOCABULARY = (
    "measured",
    "model_candidate",
    "report_eligible_automated",
)
_SLOT_ASSERTIONS = {
    _SLOT_IDS[0]: ("measured",),
    _SLOT_IDS[1]: ("measured",),
    _SLOT_IDS[2]: ("measured", "model_candidate"),
    _SLOT_IDS[3]: ("measured", "model_candidate"),
    _SLOT_IDS[4]: ("measured", "model_candidate"),
    _SLOT_IDS[5]: ("model_candidate",),
}
_SLOT_EVIDENCE_KINDS = {
    _SLOT_IDS[0]: (
        "event_protection_interval_union_receipt",
        "non_event_opportunity_interval_union_receipt",
        "quality_exclusion_interval_union_receipt",
    ),
    _SLOT_IDS[1]: (
        "background_spatial_distribution_measurement",
        "background_spectral_measurement",
    ),
    _SLOT_IDS[2]: ("focal_regional_slowing_candidate",),
    _SLOT_IDS[3]: (
        "non_event_attenuation_candidate",
        "persistent_frequency_asymmetry_candidate",
        "persistent_physical_amplitude_asymmetry_candidate",
    ),
    _SLOT_IDS[4]: (
        "interictal_ied_candidate",
        "interictal_sharp_wave_candidate",
        "interictal_spike_candidate",
        "non_event_burden_measurement",
    ),
    _SLOT_IDS[5]: ("onset_mode_context_comparison",),
}
_SEMANTIC_CEILINGS = {
    "event_protection_interval_union_receipt": "technical_scope_receipt",
    "non_event_opportunity_interval_union_receipt": "technical_scope_receipt",
    "quality_exclusion_interval_union_receipt": "technical_scope_receipt",
    "background_spectral_measurement": "physical_measurement",
    "background_spatial_distribution_measurement": "physical_measurement",
    "focal_regional_slowing_candidate": "model_candidate",
    "persistent_frequency_asymmetry_candidate": "model_candidate",
    "persistent_physical_amplitude_asymmetry_candidate": "model_candidate",
    "non_event_attenuation_candidate": "model_candidate",
    "interictal_spike_candidate": "model_candidate",
    "interictal_sharp_wave_candidate": "model_candidate",
    "interictal_ied_candidate": "model_candidate",
    "non_event_burden_measurement": "physical_measurement",
    "onset_mode_context_comparison": "non_event_context_relation_only",
}
_EEG_SIGNAL_DOMAIN_POLICY = {
    "allowed_domains": ["private", "public", "synthetic"],
    "domain_declaration_required": True,
    "private_eeg_signal_authorized": True,
    "private_labels_annotations_clinical_text_authorized": False,
}
_ALLOWED_EEG_SIGNAL_DOMAINS = frozenset(
    _EEG_SIGNAL_DOMAIN_POLICY["allowed_domains"]
)
_SOURCE_FIREWALL = {
    "private_auxiliary_data_used": False,
    "private_labels_used": False,
    "private_annotations_used": False,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "clinical_reports_used": False,
    "patient_metadata_used": False,
    "video_or_behavior_used": False,
    "sleep_staging_used": False,
    "provocation_used": False,
    "ecg_emg_eog_used": False,
    "qwen_used": False,
    "production_route_used": False,
}
_CARDINALITY_POLICY = {
    "owner_kind": "recording",
    "expected_cards_per_recording": 1,
    "cardinality_key_fields": [
        "policy_sha256",
        "recording_id",
        "source_signal_sha256",
    ],
    "event_id_field_authorized": False,
    "event_scoped_materialization_authorized": False,
    "copy_into_event_card_authorized": False,
    "event_cards_may_reference_card_id_only": True,
    "duplicate_recording_cards_fail_closed": True,
}
_ABSENCE_POLICY = {
    "absent_with_opportunity_requires_complete_opportunity": True,
    "absent_with_opportunity_requires_sensitivity_receipt": True,
    "not_evaluable_requires_reason": True,
    "uncertain_requires_reason": True,
    "missing_candidate_is_absence": False,
    "detector_low_or_segmental_s0_is_non_event_truth": False,
}
_CLOSURE_POLICY = {
    "expected_slot_count": 6,
    "every_slot_present_exactly_once": True,
    "slot_order_frozen": True,
    "every_evidence_binding_referenced_exactly_once": True,
    "evidence_binding_may_cross_slots": False,
    "unimplemented_slots_must_be_retained_not_evaluable": True,
}
_AUTHORIZATION = {
    "shadow_contract_only": True,
    "clinical_correctness_claimed": False,
    "training_authorized": False,
    "production_connection_authorized": False,
    "qwen_authorized": False,
    "report_promotion_authorized": False,
    "clinical_absence_promotion_authorized": False,
    "onset_time_claim_authorized": False,
    "onset_topography_claim_authorized": False,
    "soz_or_ez_positive_support_authorized": False,
    "mode_ranking_change_authorized": False,
    "non_event_context_may_create_ictal_onset": False,
    "non_event_context_may_create_propagation": False,
}
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


def _schema_errors(value: object, schema_path: Path) -> list[str]:
    validator = Draft202012Validator(_read_json(schema_path))
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
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


def _require_id(value: object, context: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a canonical identifier")
    return value


def _require_sorted_unique_ids(value: object, context: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be an ID array")
    result = [_require_id(item, context) for item in value]
    if result != sorted(result) or len(result) != len(set(result)):
        raise ValueError(f"{context} must be sorted and unique")
    return result


def validate_record_non_event_context_card_policy_v1(
    value: object,
    *,
    trusted_policy_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the frozen six-slot policy and its host trust binding."""

    if type(value) is not dict:
        raise TypeError("record non-event context card policy must be an object")
    candidate = deepcopy(value)
    errors = _schema_errors(
        candidate, RECORD_NON_EVENT_CONTEXT_CARD_POLICY_SCHEMA_PATH_V1
    )
    if errors:
        raise ValueError("policy schema validation failed: " + "; ".join(errors))
    expected_hash = _self_hash(candidate, "policy_sha256")
    if candidate["policy_sha256"] != expected_hash:
        raise ValueError("record non-event context policy SHA-256 mismatch")
    if trusted_policy_sha256 is not None and expected_hash != _require_sha256(
        trusted_policy_sha256, "trusted_policy_sha256"
    ):
        raise ValueError("record non-event context policy is not host trusted")

    if tuple(candidate["status_vocabulary"]) != _STATUS_VOCABULARY:
        raise ValueError("status vocabulary drifted")
    if tuple(candidate["assertion_vocabulary"]) != _ASSERTION_VOCABULARY:
        raise ValueError("assertion vocabulary drifted")
    if candidate["source_scope"] != "eeg_signal_only":
        raise ValueError("EEG signal-only source scope drifted")
    if candidate["eeg_signal_domain_policy"] != _EEG_SIGNAL_DOMAIN_POLICY:
        raise ValueError("EEG signal domain policy drifted")
    if candidate["source_firewall"] != _SOURCE_FIREWALL:
        raise ValueError("EEG-only source firewall drifted")
    if candidate["cardinality_policy"] != _CARDINALITY_POLICY:
        raise ValueError("one-card-per-recording policy drifted")
    if candidate["absence_policy"] != _ABSENCE_POLICY:
        raise ValueError("four-state absence policy drifted")
    if candidate["closure_policy"] != _CLOSURE_POLICY:
        raise ValueError("six-slot closure policy drifted")
    if candidate["authorization"] != _AUTHORIZATION:
        raise ValueError("no-onset/no-report authorization drifted")
    if tuple(candidate["slot_order"]) != _SLOT_IDS:
        raise ValueError("record non-event slot order drifted")

    slots = list(candidate["slots"])
    if [int(row["slot_index"]) for row in slots] != list(range(1, 7)):
        raise ValueError("record non-event slots must be indexed 1--6")
    if tuple(str(row["slot_id"]) for row in slots) != _SLOT_IDS:
        raise ValueError("record non-event slots are not exactly closed")
    flattened: list[str] = []
    for row in slots:
        slot_id = str(row["slot_id"])
        if tuple(row["allowed_assertion_levels"]) != _SLOT_ASSERTIONS[slot_id]:
            raise ValueError(f"{slot_id}: assertion ceiling drifted")
        if tuple(row["allowed_evidence_kinds"]) != _SLOT_EVIDENCE_KINDS[slot_id]:
            raise ValueError(f"{slot_id}: evidence-kind partition drifted")
        if row["scope_id"] != "record_non_event_context":
            raise ValueError(f"{slot_id}: scope drifted")
        if row["onset_support_permission"] != "forbidden":
            raise ValueError(f"{slot_id}: onset support must remain forbidden")
        if row["report_promotion_authorized"] is not False:
            raise ValueError(f"{slot_id}: report promotion must remain forbidden")
        flattened.extend(str(item) for item in row["allowed_evidence_kinds"])
    if len(flattened) != len(set(flattened)) or set(flattened) != set(
        _SEMANTIC_CEILINGS
    ):
        raise ValueError("evidence kinds are not an exact once-only slot partition")
    registry = {
        str(row["evidence_kind"]): str(row["semantic_ceiling"])
        for row in candidate["evidence_kind_registry"]
    }
    if len(registry) != len(candidate["evidence_kind_registry"]):
        raise ValueError("evidence-kind registry contains duplicates")
    if registry != _SEMANTIC_CEILINGS:
        raise ValueError("evidence-kind semantic ceilings drifted")
    return candidate


def load_record_non_event_context_card_policy_v1(
    path: str | Path = DEFAULT_RECORD_NON_EVENT_CONTEXT_CARD_POLICY_PATH_V1,
    *,
    trusted_policy_sha256: str | None = None,
) -> dict[str, Any]:
    """Load the checked-in policy under the default host trust anchor."""

    if trusted_policy_sha256 is None:
        trusted_policy_sha256 = DEFAULT_RECORD_NON_EVENT_CONTEXT_CARD_POLICY_SHA256_V1
    return validate_record_non_event_context_card_policy_v1(
        _read_json(Path(path)), trusted_policy_sha256=trusted_policy_sha256
    )


def _resolve_policy(
    policy: Mapping[str, object] | None,
    trusted_policy_sha256: str | None,
) -> dict[str, Any]:
    if policy is None:
        return load_record_non_event_context_card_policy_v1(
            trusted_policy_sha256=trusted_policy_sha256
        )
    if trusted_policy_sha256 is None:
        trusted_policy_sha256 = DEFAULT_RECORD_NON_EVENT_CONTEXT_CARD_POLICY_SHA256_V1
    return validate_record_non_event_context_card_policy_v1(
        dict(policy), trusted_policy_sha256=trusted_policy_sha256
    )


def _cardinality_key(
    *, policy_sha256: str, recording_id: str, source_signal_sha256: str
) -> str:
    return _canonical_sha256(
        {
            "policy_sha256": policy_sha256,
            "recording_id": recording_id,
            "source_signal_sha256": source_signal_sha256,
        }
    )


def _card_id(cardinality_key: str) -> str:
    digest = _canonical_sha256(
        {
            "method_id": "RECORD-NON-EVENT-CONTEXT-CARD-MATERIALIZER-V1",
            "cardinality_key": cardinality_key,
        }
    )
    return "RNECARD-" + digest[:24]


def _default_slot_input(
    slot_id: str, evidence_rows: Sequence[Mapping[str, object]]
) -> dict[str, Any]:
    evidence_ids = sorted(str(row["evidence_id"]) for row in evidence_rows)
    if evidence_ids:
        assertion = (
            "model_candidate"
            if any(row["assertion_level"] == "model_candidate" for row in evidence_rows)
            else "measured"
        )
        return {
            "status": "present",
            "assertion_level": assertion,
            "evidence_ids": evidence_ids,
            "opportunity_complete": False,
            "opportunity_receipt_ids": [],
            "sensitivity_receipt_ids": [],
            "reason_codes": [],
        }
    return {
        "status": "not_evaluable",
        "assertion_level": _SLOT_ASSERTIONS[slot_id][0],
        "evidence_ids": [],
        "opportunity_complete": False,
        "opportunity_receipt_ids": [],
        "sensitivity_receipt_ids": [],
        "reason_codes": ["producer_not_connected_shadow"],
    }


def materialize_record_non_event_context_card_v1(
    *,
    recording_id: str,
    source_signal_sha256: str,
    canonical_clock_id: str,
    recording_duration_seconds: float,
    eeg_signal_domain: str,
    evidence_bindings: Sequence[Mapping[str, object]] = (),
    slot_inputs: Mapping[str, Mapping[str, object]] | None = None,
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
) -> dict[str, Any]:
    """Materialize one deterministic card; missing slots remain not-evaluable."""

    checked_policy = _resolve_policy(policy, trusted_policy_sha256)
    recording_id = _require_id(recording_id, "recording_id")
    source_signal_sha256 = _require_sha256(source_signal_sha256, "source_signal_sha256")
    canonical_clock_id = _require_id(canonical_clock_id, "canonical_clock_id")
    if eeg_signal_domain not in _ALLOWED_EEG_SIGNAL_DOMAINS:
        raise ValueError(
            "eeg_signal_domain must be one of private, public, or synthetic"
        )
    if type(recording_duration_seconds) not in (int, float) or (
        float(recording_duration_seconds) <= 0
    ):
        raise ValueError("recording_duration_seconds must be positive")

    evidence = [deepcopy(dict(row)) for row in evidence_bindings]
    evidence.sort(key=lambda row: str(row.get("evidence_id", "")))
    kind_to_slot = {
        kind: slot_id
        for slot_id, kinds in _SLOT_EVIDENCE_KINDS.items()
        for kind in kinds
    }
    by_slot: dict[str, list[dict[str, Any]]] = {slot_id: [] for slot_id in _SLOT_IDS}
    for row in evidence:
        kind = str(row.get("evidence_kind", ""))
        if kind not in kind_to_slot:
            raise ValueError(f"unknown record non-event evidence kind {kind!r}")
        by_slot[kind_to_slot[kind]].append(row)

    supplied = {} if slot_inputs is None else dict(slot_inputs)
    extras = sorted(set(supplied) - set(_SLOT_IDS))
    if extras:
        raise ValueError(f"slot_inputs contains unknown slots: {extras}")
    slots: list[dict[str, Any]] = []
    for index, slot_id in enumerate(_SLOT_IDS, start=1):
        state = _default_slot_input(slot_id, by_slot[slot_id])
        if slot_id in supplied:
            state.update(deepcopy(dict(supplied[slot_id])))
        for key in (
            "evidence_ids",
            "opportunity_receipt_ids",
            "sensitivity_receipt_ids",
            "reason_codes",
        ):
            state[key] = sorted(set(str(item) for item in state.get(key, [])))
        slots.append(
            {
                "slot_index": index,
                "slot_id": slot_id,
                **state,
                "onset_support_permission": "forbidden",
                "report_promotion_authorized": False,
            }
        )

    policy_sha256 = str(checked_policy["policy_sha256"])
    cardinality_key = _cardinality_key(
        policy_sha256=policy_sha256,
        recording_id=recording_id,
        source_signal_sha256=source_signal_sha256,
    )
    card: dict[str, Any] = {
        "schema_version": RECORD_NON_EVENT_CONTEXT_CARD_SCHEMA_VERSION_V1,
        "card_id": _card_id(cardinality_key),
        "card_status": "draft_eeg_signal_domain_shadow",
        "source_scope": "eeg_signal_only",
        "eeg_signal_domain": eeg_signal_domain,
        "policy_binding": {
            "policy_id": RECORD_NON_EVENT_CONTEXT_CARD_POLICY_ID_V1,
            "policy_sha256": policy_sha256,
        },
        "owner": {
            "owner_kind": "recording",
            "recording_id": recording_id,
            "source_signal_sha256": source_signal_sha256,
            "canonical_clock_id": canonical_clock_id,
            "recording_duration_seconds": float(recording_duration_seconds),
            "cardinality_key": cardinality_key,
            "card_instance_ordinal": 1,
            "event_scoped": False,
            "shared_with_event_cards_by_reference_only": True,
        },
        "source_firewall": deepcopy(_SOURCE_FIREWALL),
        "slots": slots,
        "evidence_bindings": evidence,
        "closure": {
            "expected_slot_count": 6,
            "observed_slot_count": 6,
            "every_slot_present_exactly_once": True,
            "slot_order_matches_policy": True,
            "every_evidence_binding_referenced_exactly_once": True,
            "record_scope_only": True,
            "single_cardinality_key_for_recording": True,
            "event_payload_embedded": False,
        },
        "authorization": deepcopy(_AUTHORIZATION),
    }
    card["card_sha256"] = _self_hash(card, "card_sha256")
    return validate_record_non_event_context_card_v1(
        card,
        policy=checked_policy,
        trusted_policy_sha256=policy_sha256,
    )


def validate_record_non_event_context_card_v1(
    value: object,
    *,
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
) -> dict[str, Any]:
    """Strictly replay a card and reject scope, absence, or permission drift."""

    if type(value) is not dict:
        raise TypeError("record non-event context card must be an object")
    candidate = deepcopy(value)
    errors = _schema_errors(candidate, RECORD_NON_EVENT_CONTEXT_CARD_SCHEMA_PATH_V1)
    if errors:
        raise ValueError("card schema validation failed: " + "; ".join(errors))
    checked_policy = _resolve_policy(policy, trusted_policy_sha256)
    policy_sha256 = str(checked_policy["policy_sha256"])
    if candidate["policy_binding"] != {
        "policy_id": RECORD_NON_EVENT_CONTEXT_CARD_POLICY_ID_V1,
        "policy_sha256": policy_sha256,
    }:
        raise ValueError("card policy binding drifted")
    if candidate["source_scope"] != "eeg_signal_only":
        raise ValueError("card EEG signal-only source scope drifted")
    if candidate["eeg_signal_domain"] not in _ALLOWED_EEG_SIGNAL_DOMAINS:
        raise ValueError("card EEG signal domain is not policy allowed")
    if candidate["source_firewall"] != _SOURCE_FIREWALL:
        raise ValueError("card EEG-only source firewall drifted")
    if candidate["authorization"] != _AUTHORIZATION:
        raise ValueError("card no-onset/no-report authorization drifted")
    expected_closure = {
        "expected_slot_count": 6,
        "observed_slot_count": 6,
        "every_slot_present_exactly_once": True,
        "slot_order_matches_policy": True,
        "every_evidence_binding_referenced_exactly_once": True,
        "record_scope_only": True,
        "single_cardinality_key_for_recording": True,
        "event_payload_embedded": False,
    }
    if candidate["closure"] != expected_closure:
        raise ValueError("card closure receipt drifted")

    owner = candidate["owner"]
    recording_id = _require_id(owner["recording_id"], "owner.recording_id")
    signal_hash = _require_sha256(
        owner["source_signal_sha256"], "owner.source_signal_sha256"
    )
    expected_key = _cardinality_key(
        policy_sha256=policy_sha256,
        recording_id=recording_id,
        source_signal_sha256=signal_hash,
    )
    if owner["cardinality_key"] != expected_key:
        raise ValueError("recording cardinality key does not replay")
    if candidate["card_id"] != _card_id(expected_key):
        raise ValueError("record non-event card ID does not replay")

    slots = list(candidate["slots"])
    if [int(row["slot_index"]) for row in slots] != list(range(1, 7)) or tuple(
        str(row["slot_id"]) for row in slots
    ) != _SLOT_IDS:
        raise ValueError("card must contain the six policy slots exactly once in order")
    evidence_rows = list(candidate["evidence_bindings"])
    evidence_ids = [str(row["evidence_id"]) for row in evidence_rows]
    if evidence_ids != sorted(evidence_ids) or len(evidence_ids) != len(
        set(evidence_ids)
    ):
        raise ValueError("evidence bindings must be sorted and unique")
    evidence_by_id = {str(row["evidence_id"]): row for row in evidence_rows}
    kind_to_slot = {
        kind: slot_id
        for slot_id, kinds in _SLOT_EVIDENCE_KINDS.items()
        for kind in kinds
    }
    references: list[str] = []
    for slot in slots:
        slot_id = str(slot["slot_id"])
        assertion = str(slot["assertion_level"])
        if assertion not in _SLOT_ASSERTIONS[slot_id]:
            raise ValueError(f"{slot_id}: assertion level is not policy allowed")
        ids = _require_sorted_unique_ids(
            slot["evidence_ids"], f"{slot_id}.evidence_ids"
        )
        _require_sorted_unique_ids(
            slot["opportunity_receipt_ids"], f"{slot_id}.opportunity_receipt_ids"
        )
        _require_sorted_unique_ids(
            slot["sensitivity_receipt_ids"], f"{slot_id}.sensitivity_receipt_ids"
        )
        reasons = _require_sorted_unique_ids(
            slot["reason_codes"], f"{slot_id}.reason_codes"
        )
        references.extend(ids)
        status = str(slot["status"])
        if status == "present" and not ids:
            raise ValueError(f"{slot_id}: present requires evidence")
        if status in {"uncertain", "not_evaluable"} and not reasons:
            raise ValueError(f"{slot_id}: {status} requires a reason")
        if status == "absent_with_opportunity":
            if slot["opportunity_complete"] is not True:
                raise ValueError(f"{slot_id}: absence requires complete opportunity")
            if not slot["opportunity_receipt_ids"]:
                raise ValueError(f"{slot_id}: absence requires an opportunity receipt")
            if not slot["sensitivity_receipt_ids"]:
                raise ValueError(f"{slot_id}: absence requires a sensitivity receipt")
        owned_rows: list[Mapping[str, object]] = []
        for evidence_id in ids:
            if evidence_id not in evidence_by_id:
                raise ValueError(
                    f"{slot_id}: references unknown evidence {evidence_id!r}"
                )
            row = evidence_by_id[evidence_id]
            if kind_to_slot.get(str(row["evidence_kind"])) != slot_id:
                raise ValueError(f"{slot_id}: evidence kind belongs to another slot")
            owned_rows.append(row)
        if owned_rows:
            expected_assertion = (
                "model_candidate"
                if any(
                    row["assertion_level"] == "model_candidate" for row in owned_rows
                )
                else "measured"
            )
            if assertion != expected_assertion:
                raise ValueError(
                    f"{slot_id}: slot assertion does not summarize evidence"
                )

    counts = Counter(references)
    if set(counts) != set(evidence_by_id) or any(
        count != 1 for count in counts.values()
    ):
        raise ValueError("every evidence binding must be referenced exactly once")
    for row in evidence_rows:
        context = str(row["evidence_id"])
        if (
            row["recording_id"] != recording_id
            or row["source_signal_sha256"] != signal_hash
        ):
            raise ValueError(f"{context}: recording/source identity drifted")
        if row["scope_id"] != "record_non_event_context":
            raise ValueError(f"{context}: event-scoped evidence is forbidden")
        if row["intrinsic_evidence_role"] not in {"non_event_context", "limitation"}:
            raise ValueError(f"{context}: onset/event evidence role is forbidden")
        if row["onset_support_permission"] != "forbidden":
            raise ValueError(f"{context}: onset support is forbidden")
        if row["mode_ranking_change_authorized"] is not False:
            raise ValueError(f"{context}: mode ranking change is forbidden")
        if row["report_promotion_authorized"] is not False:
            raise ValueError(f"{context}: report promotion is forbidden")
        mode_fields = {"mode_id", "context_relation", "comparison_basis"}
        present_mode_fields = mode_fields.intersection(row)
        if row["evidence_kind"] == "onset_mode_context_comparison":
            if present_mode_fields != mode_fields:
                raise ValueError(f"{context}: mode comparison fields are incomplete")
        elif present_mode_fields:
            raise ValueError(f"{context}: mode fields are comparison-only")

    expected_self_hash = _self_hash(candidate, "card_sha256")
    if candidate["card_sha256"] != expected_self_hash:
        raise ValueError("record non-event card SHA-256 mismatch")
    return candidate


def validate_record_non_event_context_card_collection_v1(
    values: Sequence[object],
    *,
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Validate a collection and reject a second card for the same recording."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("record non-event context card collection must be an array")
    checked_policy = _resolve_policy(policy, trusted_policy_sha256)
    result = [
        validate_record_non_event_context_card_v1(
            value,
            policy=checked_policy,
            trusted_policy_sha256=str(checked_policy["policy_sha256"]),
        )
        for value in values
    ]
    keys = [str(row["owner"]["cardinality_key"]) for row in result]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError(
            "duplicate record non-event context card cardinality keys: "
            + ", ".join(duplicates)
        )
    return result
