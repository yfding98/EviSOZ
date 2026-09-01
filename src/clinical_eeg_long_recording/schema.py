"""Strict recording-level contracts for long-term scalp EEG reporting.

The contracts in this module deliberately keep three concerns separate:

* a full-recording detector proposes *review candidates*, never seizures;
* each fixed ``[-12, +48]`` second segment carries an independently validated
  ``clinical_eeg_report_v1`` EEG fact ledger and presentation-only receipts;
* recording-level aggregation is implemented in :mod:`.aggregation`.

All public validators accept untrusted ``dict`` payloads and return canonical
deep copies.  Unknown fields, non-finite numbers, host paths, free-form
side-channel prose, unfrozen detector claims, and research SOZ promotion into
clinical facts fail closed.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
from typing import Any, Mapping, Sequence

from src.clinical_eeg_report import canonicalize_electrode, validate_report_payload


DETECTION_MANIFEST_SCHEMA_VERSION = "long_term_seizure_detection_manifest_v1"
EVENT_SEGMENT_RECEIPT_SCHEMA_VERSION = "long_term_event_segment_receipt_v1"
LONG_TERM_BUNDLE_SCHEMA_VERSION = "trustworthy_long_term_clinical_eeg_bundle_v1"
FILTERED_LONG_TERM_BUNDLE_SCHEMA_VERSION = (
    "trustworthy_long_term_clinical_eeg_bundle_v2_signal_eligibility_partition"
)

FIXED_EVENT_WINDOW_SECONDS = (-12.0, 48.0)
FIXED_SEGMENT_DURATION_SECONDS = 60.0
FIXED_EVENT_ANCHOR_OFFSET_SECONDS = 12.0
REQUIRED_CAUSAL_WARMUP_SECONDS = 30.0
MINIMUM_ANALYZABLE_ANCHOR_SECONDS = (
    FIXED_EVENT_ANCHOR_OFFSET_SECONDS + REQUIRED_CAUSAL_WARMUP_SECONDS
)
CANDIDATE_SEMANTICS = "review_candidate_not_confirmed_seizure"
BOUNDARY_POLICY = "require_full_fixed_window_no_padding"
WAVEFORM_SELECTION_POLICY = "report_fact_evidence_ids_only_no_llm"
SOZ_INTERPRETATION_STATUS = "research_scalp_electrode_ranking_not_clinical_soz"

_TOLERANCE_SECONDS = 1e-6
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PATH_LIKE_RE = re.compile(
    r"(?:^/|^[A-Za-z]:[\\/]|\\|(?:^|/)\.\.(?:/|$)|file://|https?://)",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_RAW_FILE_REFERENCE_RE = re.compile(
    r"(?:[/\\]|\.(?:edf|bdf|xls|xlsx|csv|doc|docx)(?:$|[^A-Za-z0-9]))",
    re.IGNORECASE,
)
_SOZ_PROMOTION_RE = re.compile(
    r"(?:\bSOZ\b|seizure\s*onset\s*zone|致痫区|致痫灶|治疗靶点|"
    r"皮层(?:发作)?起始|(?:通道|电极)(?:候选)?排序)",
    re.IGNORECASE,
)

_DETECTOR_ROLES = (
    "deployment_qualified",
    "research_candidate",
    "heuristic_preselector",
)
_PROMOTION_STATUSES = (
    "passed_external_promotion_gate",
    "not_deployment_qualified",
    "not_evaluated_for_deployment",
)
_OPERATING_POINT_SOURCES = (
    "external_validation_frozen",
    "internal_validation_frozen",
    "engineering_heuristic_frozen",
)
_RAW_ALARM_DECISIONS = (
    "retained_for_merge",
    "rejected_below_threshold",
    "suppressed_by_artifact_gate",
    "suppressed_by_refractory_policy",
)
_MERGE_CANDIDATE_DECISIONS = (
    "selected_for_event_analysis",
    "suppressed_by_merge_policy",
    "rejected_insufficient_fixed_window",
    "review_only_not_segmented",
)


def _strict_object(
    value: object,
    *,
    required: Sequence[str],
    context: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    keys = set(value)
    expected = set(required)
    missing = expected.difference(keys)
    extra = keys.difference(expected)
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{context} has unknown keys: {sorted(extra)}")
    return deepcopy(value)


def _controlled_string(value: object, context: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty, trimmed string")
    if len(value) > maximum:
        raise ValueError(f"{context} must be at most {maximum} characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{context} contains control characters")
    if _PATH_LIKE_RE.search(value):
        raise ValueError(f"{context} must not contain a raw path or URL")
    if _EMAIL_RE.search(value):
        raise ValueError(f"{context} must not contain direct contact information")
    return value


def _identifier(value: object, context: str) -> str:
    text = _controlled_string(value, context, maximum=128)
    if _ID_RE.fullmatch(text) is None:
        raise ValueError(f"{context} has an invalid identifier: {text!r}")
    return text


def _enum(value: object, allowed: Sequence[str], context: str) -> str:
    text = _controlled_string(value, context)
    if text not in allowed:
        raise ValueError(f"{context} must be one of {tuple(allowed)}")
    return text


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be 64 lowercase hexadecimal characters")
    return value


def _finite_number(
    value: object,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{context} must be <= {maximum}")
    if exclusive_minimum is not None and number <= exclusive_minimum:
        raise ValueError(f"{context} must be > {exclusive_minimum}")
    return number


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return value


def _boolean(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{context} must be a boolean")
    return value


def _same_number(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= _TOLERANCE_SECONDS


def _number_pair(value: object, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{context} must be a two-number array")
    return [
        _finite_number(value[0], f"{context}[0]"),
        _finite_number(value[1], f"{context}[1]"),
    ]


def _identifier_list(
    value: object,
    context: str,
    *,
    allow_empty: bool,
) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "an array" if allow_empty else "a non-empty array"
        raise TypeError(f"{context} must be {qualifier}")
    result = [_identifier(item, f"{context}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise ValueError(f"{context} contains duplicate identifiers")
    return result


def _constant(value: object, expected: object, context: str) -> object:
    if value != expected or type(value) is not type(expected):
        raise ValueError(f"{context} must be {expected!r}")
    return deepcopy(expected)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_payload_sha256(value: object) -> str:
    """Return the stable SHA-256 of an already canonical JSON payload."""

    return _canonical_sha256(deepcopy(value))


def _validate_scan_coverage(
    value: object,
    *,
    duration: float,
) -> list[dict[str, float]]:
    if not isinstance(value, list) or not value:
        raise TypeError("scan_coverage_intervals must be a non-empty array")
    intervals: list[dict[str, float]] = []
    expected_start = 0.0
    for index, raw in enumerate(value):
        data = _strict_object(
            raw,
            required=("start_offset_seconds", "stop_offset_seconds"),
            context=f"scan_coverage_intervals[{index}]",
        )
        start = _finite_number(
            data["start_offset_seconds"],
            f"scan_coverage_intervals[{index}].start_offset_seconds",
            minimum=0,
            maximum=duration,
        )
        stop = _finite_number(
            data["stop_offset_seconds"],
            f"scan_coverage_intervals[{index}].stop_offset_seconds",
            minimum=0,
            maximum=duration,
        )
        if stop <= start:
            raise ValueError("scan coverage intervals must have positive duration")
        if not _same_number(start, expected_start):
            raise ValueError(
                "scan coverage must be ordered, contiguous, and start at recording offset 0"
            )
        intervals.append(
            {"start_offset_seconds": start, "stop_offset_seconds": stop}
        )
        expected_start = stop
    if not _same_number(expected_start, duration):
        raise ValueError("scan coverage must end at recording_duration_seconds")
    return intervals


def _validate_detector_receipt(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=(
            "detector_id",
            "detector_role",
            "weights_sha256",
            "code_sha256",
            "policy_sha256",
            "operating_point",
            "promotion_status",
            "promotion_receipt_sha256",
            "annotations_used",
            "labels_used",
        ),
        context="detector_receipt",
    )
    detector_id = _identifier(data["detector_id"], "detector_receipt.detector_id")
    role = _enum(data["detector_role"], _DETECTOR_ROLES, "detector_receipt.detector_role")
    operating_raw = _strict_object(
        data["operating_point"],
        required=(
            "operating_point_id",
            "threshold",
            "score_direction",
            "selection_source",
            "frozen_before_recording",
        ),
        context="detector_receipt.operating_point",
    )
    operating_point = {
        "operating_point_id": _identifier(
            operating_raw["operating_point_id"],
            "detector_receipt.operating_point.operating_point_id",
        ),
        "threshold": _finite_number(
            operating_raw["threshold"],
            "detector_receipt.operating_point.threshold",
        ),
        "score_direction": _enum(
            operating_raw["score_direction"],
            ("greater_or_equal", "less_or_equal"),
            "detector_receipt.operating_point.score_direction",
        ),
        "selection_source": _enum(
            operating_raw["selection_source"],
            _OPERATING_POINT_SOURCES,
            "detector_receipt.operating_point.selection_source",
        ),
        "frozen_before_recording": _boolean(
            operating_raw["frozen_before_recording"],
            "detector_receipt.operating_point.frozen_before_recording",
        ),
    }
    if operating_point["frozen_before_recording"] is not True:
        raise ValueError("detector operating point must be frozen before inference")

    promotion_status = _enum(
        data["promotion_status"],
        _PROMOTION_STATUSES,
        "detector_receipt.promotion_status",
    )
    raw_promotion_hash = data["promotion_receipt_sha256"]
    promotion_hash = (
        None
        if raw_promotion_hash is None
        else _sha256(raw_promotion_hash, "detector_receipt.promotion_receipt_sha256")
    )
    selection_source = operating_point["selection_source"]
    if role == "deployment_qualified":
        if (
            promotion_status != "passed_external_promotion_gate"
            or promotion_hash is None
            or selection_source != "external_validation_frozen"
        ):
            raise ValueError(
                "deployment_qualified requires an external frozen operating point "
                "and a passed promotion receipt"
            )
    elif role == "research_candidate":
        if promotion_status != "not_deployment_qualified":
            raise ValueError(
                "research_candidate must remain explicitly not deployment qualified"
            )
        if selection_source == "engineering_heuristic_frozen":
            raise ValueError("research_candidate cannot claim a heuristic operating point")
    else:
        if (
            promotion_status != "not_evaluated_for_deployment"
            or promotion_hash is not None
            or selection_source != "engineering_heuristic_frozen"
        ):
            raise ValueError(
                "heuristic_preselector must remain not evaluated for deployment"
            )

    if _boolean(data["annotations_used"], "detector_receipt.annotations_used") is not False:
        raise ValueError("full-recording detection must not use EDF annotations")
    if _boolean(data["labels_used"], "detector_receipt.labels_used") is not False:
        raise ValueError("full-recording detection must not use event labels")

    return {
        "detector_id": detector_id,
        "detector_role": role,
        "weights_sha256": _sha256(
            data["weights_sha256"], "detector_receipt.weights_sha256"
        ),
        "code_sha256": _sha256(data["code_sha256"], "detector_receipt.code_sha256"),
        "policy_sha256": _sha256(
            data["policy_sha256"], "detector_receipt.policy_sha256"
        ),
        "operating_point": operating_point,
        "promotion_status": promotion_status,
        "promotion_receipt_sha256": promotion_hash,
        "annotations_used": False,
        "labels_used": False,
    }


def _validate_alarm_like(
    value: object,
    *,
    context: str,
    id_key: str,
    duration: float,
    decisions: Sequence[str],
) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=(
            id_key,
            "start_offset_seconds",
            "stop_offset_seconds",
            "anchor_offset_seconds",
            "score",
            "support_count",
            "decision_available",
            "decision_available_offset_seconds",
            "decision",
            "member_ids",
            "semantics",
        ),
        context=context,
    )
    start = _finite_number(
        data["start_offset_seconds"],
        f"{context}.start_offset_seconds",
        minimum=0,
        maximum=duration,
    )
    stop = _finite_number(
        data["stop_offset_seconds"],
        f"{context}.stop_offset_seconds",
        minimum=0,
        maximum=duration,
    )
    if stop <= start:
        raise ValueError(f"{context} must have positive duration")
    anchor = _finite_number(
        data["anchor_offset_seconds"],
        f"{context}.anchor_offset_seconds",
        minimum=start,
        maximum=stop,
    )
    support_count = _integer(data["support_count"], f"{context}.support_count")
    member_ids = _identifier_list(
        data["member_ids"], f"{context}.member_ids", allow_empty=False
    )
    if support_count != len(member_ids):
        raise ValueError(f"{context}.support_count must equal the number of member_ids")
    available = _boolean(data["decision_available"], f"{context}.decision_available")
    available_offset = _finite_number(
        data["decision_available_offset_seconds"],
        f"{context}.decision_available_offset_seconds",
        minimum=0,
        maximum=duration,
    )
    if available_offset < stop - _TOLERANCE_SECONDS:
        raise ValueError(
            f"{context}.decision_available_offset_seconds must not precede support stop"
        )
    if available:
        decision = _enum(data["decision"], decisions, f"{context}.decision")
    else:
        if data["decision"] is not None:
            raise ValueError(f"{context}.decision must be null when unavailable")
        decision = None
    semantics = _enum(data["semantics"], (CANDIDATE_SEMANTICS,), f"{context}.semantics")
    return {
        id_key: _identifier(data[id_key], f"{context}.{id_key}"),
        "start_offset_seconds": start,
        "stop_offset_seconds": stop,
        "anchor_offset_seconds": anchor,
        "score": _finite_number(data["score"], f"{context}.score"),
        "support_count": support_count,
        "decision_available": available,
        "decision_available_offset_seconds": available_offset,
        "decision": decision,
        "member_ids": member_ids,
        "semantics": semantics,
    }


def validate_long_term_seizure_detection_manifest(payload: object) -> dict[str, Any]:
    """Validate a full-recording detector manifest and return a canonical copy."""

    data = _strict_object(
        payload,
        required=(
            "schema_version",
            "manifest_id",
            "recording_id",
            "patient_pseudonym",
            "source_signal_sha256",
            "recording_duration_seconds",
            "scan_coverage_intervals",
            "detector_receipt",
            "candidate_semantics",
            "raw_alarms",
            "merge_candidates",
        ),
        context="long-term seizure detection manifest",
    )
    schema_version = _enum(
        data["schema_version"],
        (DETECTION_MANIFEST_SCHEMA_VERSION,),
        "manifest.schema_version",
    )
    duration = _finite_number(
        data["recording_duration_seconds"],
        "manifest.recording_duration_seconds",
        exclusive_minimum=0,
    )
    coverage = _validate_scan_coverage(data["scan_coverage_intervals"], duration=duration)
    detector_receipt = _validate_detector_receipt(data["detector_receipt"])
    semantics = _enum(
        data["candidate_semantics"],
        (CANDIDATE_SEMANTICS,),
        "manifest.candidate_semantics",
    )

    raw_alarm_values = data["raw_alarms"]
    if not isinstance(raw_alarm_values, list):
        raise TypeError("manifest.raw_alarms must be an array")
    raw_alarms = [
        _validate_alarm_like(
            item,
            context=f"manifest.raw_alarms[{index}]",
            id_key="alarm_id",
            duration=duration,
            decisions=_RAW_ALARM_DECISIONS,
        )
        for index, item in enumerate(raw_alarm_values)
    ]
    raw_ids = [item["alarm_id"] for item in raw_alarms]
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError("manifest.raw_alarms contains duplicate alarm_id values")
    raw_alarms.sort(
        key=lambda item: (
            item["start_offset_seconds"],
            item["anchor_offset_seconds"],
            item["alarm_id"],
        )
    )

    candidate_values = data["merge_candidates"]
    if not isinstance(candidate_values, list):
        raise TypeError("manifest.merge_candidates must be an array")
    merge_candidates = [
        _validate_alarm_like(
            item,
            context=f"manifest.merge_candidates[{index}]",
            id_key="candidate_id",
            duration=duration,
            decisions=_MERGE_CANDIDATE_DECISIONS,
        )
        for index, item in enumerate(candidate_values)
    ]
    candidate_ids = [item["candidate_id"] for item in merge_candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("manifest.merge_candidates contains duplicate candidate_id values")
    merge_candidates.sort(
        key=lambda item: (
            item["start_offset_seconds"],
            item["anchor_offset_seconds"],
            item["candidate_id"],
        )
    )
    raw_id_set = set(raw_ids)
    raw_by_id = {item["alarm_id"]: item for item in raw_alarms}
    for candidate in merge_candidates:
        if not candidate["member_ids"]:
            raise ValueError("every merged candidate must identify at least one raw alarm")
        unknown = set(candidate["member_ids"]).difference(raw_id_set)
        if unknown:
            raise ValueError(
                f"merged candidate references unknown raw alarms: {sorted(unknown)}"
            )
        member_alarms = [raw_by_id[member_id] for member_id in candidate["member_ids"]]
        if candidate["start_offset_seconds"] > min(
            member["start_offset_seconds"] for member in member_alarms
        ) + _TOLERANCE_SECONDS or candidate["stop_offset_seconds"] < max(
            member["stop_offset_seconds"] for member in member_alarms
        ) - _TOLERANCE_SECONDS:
            raise ValueError("merged candidate interval must cover every member alarm")
        latest_member_decision = max(
            member["decision_available_offset_seconds"] for member in member_alarms
        )
        if (
            candidate["decision_available_offset_seconds"]
            < latest_member_decision - _TOLERANCE_SECONDS
        ):
            raise ValueError(
                "merged candidate decision became available before a member alarm decision"
            )
        if candidate["decision"] == "selected_for_event_analysis":
            anchor = candidate["anchor_offset_seconds"]
            if (
                anchor < MINIMUM_ANALYZABLE_ANCHOR_SECONDS - _TOLERANCE_SECONDS
                or duration - anchor < FIXED_EVENT_WINDOW_SECONDS[1] - _TOLERANCE_SECONDS
            ):
                raise ValueError(
                    "selected candidate lacks the full fixed [-12,+48] second "
                    "context or required causal warmup"
                )

    return {
        "schema_version": schema_version,
        "manifest_id": _identifier(data["manifest_id"], "manifest.manifest_id"),
        "recording_id": _identifier(data["recording_id"], "manifest.recording_id"),
        "patient_pseudonym": _identifier(
            data["patient_pseudonym"], "manifest.patient_pseudonym"
        ),
        "source_signal_sha256": _sha256(
            data["source_signal_sha256"], "manifest.source_signal_sha256"
        ),
        "recording_duration_seconds": duration,
        "scan_coverage_intervals": coverage,
        "detector_receipt": detector_receipt,
        "candidate_semantics": semantics,
        "raw_alarms": raw_alarms,
        "merge_candidates": merge_candidates,
    }


def _safe_relative_png(value: object, context: str) -> str:
    text = _controlled_string(value, context, maximum=512)
    if ":" in text or "\\" in text or text.startswith("/"):
        raise ValueError(f"{context} must be a safe POSIX relative path")
    parts = text.split("/")
    if any(
        part in {"", ".", ".."} or _PATH_SEGMENT_RE.fullmatch(part) is None
        for part in parts
    ):
        raise ValueError(f"{context} contains an unsafe path segment")
    path = PurePosixPath(text)
    if path.is_absolute() or path.as_posix() != text or path.suffix != ".png":
        raise ValueError(f"{context} must be a canonical relative lowercase PNG path")
    return text


def _validate_waveform_attachment(
    value: object,
    *,
    report_payload: Mapping[str, Any],
    eeg_event_id: str,
    source_signal_sha256: str,
    preprocessing_receipt_sha256: str,
    processed_window_sha256: str,
) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=(
            "attachment_id",
            "evidence_id",
            "fact_ids",
            "eeg_event_id",
            "figure_file",
            "figure_sha256",
            "source_signal_sha256",
            "preprocessing_receipt_sha256",
            "processed_window_sha256",
            "event_window_seconds",
            "event_anchor_offset_seconds",
            "evidence_interval_seconds_relative_to_anchor",
            "selection_policy",
            "sent_to_llm",
        ),
        context="waveform_attachment",
    )
    attachment_event_id = _identifier(
        data["eeg_event_id"], "waveform_attachment.eeg_event_id"
    )
    if attachment_event_id != eeg_event_id:
        raise ValueError("waveform attachment EEG event does not match the segment")
    attachment_source_hash = _sha256(
        data["source_signal_sha256"], "waveform_attachment.source_signal_sha256"
    )
    attachment_preprocess_hash = _sha256(
        data["preprocessing_receipt_sha256"],
        "waveform_attachment.preprocessing_receipt_sha256",
    )
    attachment_processed_hash = _sha256(
        data["processed_window_sha256"],
        "waveform_attachment.processed_window_sha256",
    )
    if attachment_source_hash != source_signal_sha256:
        raise ValueError("waveform attachment source_signal_sha256 mismatch")
    if attachment_preprocess_hash != preprocessing_receipt_sha256:
        raise ValueError("waveform attachment preprocessing receipt hash mismatch")
    if attachment_processed_hash != processed_window_sha256:
        raise ValueError("waveform attachment processed window hash mismatch")

    event_window = _number_pair(
        data["event_window_seconds"], "waveform_attachment.event_window_seconds"
    )
    if not all(
        _same_number(actual, expected)
        for actual, expected in zip(event_window, FIXED_EVENT_WINDOW_SECONDS)
    ):
        raise ValueError("waveform attachment must use the fixed [-12,+48] window")
    anchor = _finite_number(
        data["event_anchor_offset_seconds"],
        "waveform_attachment.event_anchor_offset_seconds",
    )
    if not _same_number(anchor, FIXED_EVENT_ANCHOR_OFFSET_SECONDS):
        raise ValueError("waveform attachment anchor must be 12 seconds from segment start")
    evidence_interval = _number_pair(
        data["evidence_interval_seconds_relative_to_anchor"],
        "waveform_attachment.evidence_interval_seconds_relative_to_anchor",
    )
    if (
        evidence_interval[0] < FIXED_EVENT_WINDOW_SECONDS[0] - _TOLERANCE_SECONDS
        or evidence_interval[1] > FIXED_EVENT_WINDOW_SECONDS[1] + _TOLERANCE_SECONDS
        or evidence_interval[1] <= evidence_interval[0]
    ):
        raise ValueError("waveform evidence interval falls outside the fixed event window")

    evidence_id = _identifier(data["evidence_id"], "waveform_attachment.evidence_id")
    fact_ids = _identifier_list(
        data["fact_ids"], "waveform_attachment.fact_ids", allow_empty=False
    )
    raw_facts = report_payload["facts"]
    facts_by_id = {item["fact_id"]: item for item in raw_facts}
    unknown_fact_ids = set(fact_ids).difference(facts_by_id)
    if unknown_fact_ids:
        raise ValueError(
            f"waveform attachment references unknown facts: {sorted(unknown_fact_ids)}"
        )
    for fact_id in fact_ids:
        fact = facts_by_id[fact_id]
        if fact.get("eeg_event_id") != eeg_event_id or fact.get("section") != "ictal":
            raise ValueError("waveform attachment fact_ids must remain within one EEG event")
        if evidence_id not in fact["evidence_ids"]:
            raise ValueError("waveform evidence_id is absent from a declared report fact")
    actual_fact_ids = {
        fact["fact_id"] for fact in raw_facts if evidence_id in fact["evidence_ids"]
    }
    if actual_fact_ids != set(fact_ids):
        raise ValueError(
            "waveform attachment fact_ids must exactly match facts using its evidence_id"
        )
    if _enum(
        data["selection_policy"],
        (WAVEFORM_SELECTION_POLICY,),
        "waveform_attachment.selection_policy",
    ) != WAVEFORM_SELECTION_POLICY:
        raise AssertionError("unreachable selection policy branch")
    if _boolean(data["sent_to_llm"], "waveform_attachment.sent_to_llm") is not False:
        raise ValueError("waveform attachments must not be sent to the language model")

    return {
        "attachment_id": _identifier(
            data["attachment_id"], "waveform_attachment.attachment_id"
        ),
        "evidence_id": evidence_id,
        "fact_ids": fact_ids,
        "eeg_event_id": attachment_event_id,
        "figure_file": _safe_relative_png(
            data["figure_file"], "waveform_attachment.figure_file"
        ),
        "figure_sha256": _sha256(
            data["figure_sha256"], "waveform_attachment.figure_sha256"
        ),
        "source_signal_sha256": attachment_source_hash,
        "preprocessing_receipt_sha256": attachment_preprocess_hash,
        "processed_window_sha256": attachment_processed_hash,
        "event_window_seconds": event_window,
        "event_anchor_offset_seconds": anchor,
        "evidence_interval_seconds_relative_to_anchor": evidence_interval,
        "selection_policy": WAVEFORM_SELECTION_POLICY,
        "sent_to_llm": False,
    }


def _validate_research_soz_ranking(
    value: object,
    *,
    processed_window_sha256: str,
) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=(
            "receipt_id",
            "method_id",
            "model_sha256",
            "input_processed_window_sha256",
            "interpretation_status",
            "ranked_electrodes",
            "used_in_clinical_facts",
            "used_in_impression",
            "sent_to_llm",
        ),
        context="research_soz_ranking_receipt",
    )
    input_hash = _sha256(
        data["input_processed_window_sha256"],
        "research_soz_ranking_receipt.input_processed_window_sha256",
    )
    if input_hash != processed_window_sha256:
        raise ValueError("research SOZ ranking input hash does not match the segment")
    status = _enum(
        data["interpretation_status"],
        (SOZ_INTERPRETATION_STATUS,),
        "research_soz_ranking_receipt.interpretation_status",
    )
    raw_ranking = data["ranked_electrodes"]
    if not isinstance(raw_ranking, list) or not raw_ranking:
        raise TypeError("research_soz_ranking_receipt.ranked_electrodes must be non-empty")
    ranking: list[dict[str, Any]] = []
    seen_electrodes: set[str] = set()
    for index, raw in enumerate(raw_ranking):
        item = _strict_object(
            raw,
            required=("rank", "electrode", "score"),
            context=f"research_soz_ranking_receipt.ranked_electrodes[{index}]",
        )
        rank = _integer(
            item["rank"],
            f"research_soz_ranking_receipt.ranked_electrodes[{index}].rank",
            minimum=1,
        )
        if rank != index + 1:
            raise ValueError("research SOZ ranks must be contiguous and start at 1")
        electrode = canonicalize_electrode(item["electrode"])
        if electrode in seen_electrodes:
            raise ValueError("research SOZ ranking contains duplicate electrodes")
        seen_electrodes.add(electrode)
        ranking.append(
            {
                "rank": rank,
                "electrode": electrode,
                "score": _finite_number(
                    item["score"],
                    f"research_soz_ranking_receipt.ranked_electrodes[{index}].score",
                    minimum=0,
                    maximum=1,
                ),
            }
        )
    scores = [item["score"] for item in ranking]
    if any(left < right for left, right in zip(scores, scores[1:])):
        raise ValueError("research SOZ ranking scores must be non-increasing")
    for key in ("used_in_clinical_facts", "used_in_impression", "sent_to_llm"):
        if _boolean(data[key], f"research_soz_ranking_receipt.{key}") is not False:
            raise ValueError(f"research SOZ ranking {key} must remain false")
    return {
        "receipt_id": _identifier(
            data["receipt_id"], "research_soz_ranking_receipt.receipt_id"
        ),
        "method_id": _identifier(
            data["method_id"], "research_soz_ranking_receipt.method_id"
        ),
        "model_sha256": _sha256(
            data["model_sha256"], "research_soz_ranking_receipt.model_sha256"
        ),
        "input_processed_window_sha256": input_hash,
        "interpretation_status": status,
        "ranked_electrodes": ranking,
        "used_in_clinical_facts": False,
        "used_in_impression": False,
        "sent_to_llm": False,
    }


def _nested_strings(value: object) -> Sequence[str]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(
            text for child in value.values() for text in _nested_strings(child)
        )
    if isinstance(value, (list, tuple)):
        return tuple(text for child in value for text in _nested_strings(child))
    return ()


def _audit_clinical_fact_boundary(report_payload: Mapping[str, Any]) -> None:
    """Keep research rankings, raw files, and direct PHI out of fact ledgers."""

    for fact in report_payload["facts"]:
        for text in _nested_strings(fact):
            if _SOZ_PROMOTION_RE.search(text):
                raise ValueError("research SOZ ranking must not appear in clinical facts")
            if _PATH_LIKE_RE.search(text) or _RAW_FILE_REFERENCE_RE.search(text):
                raise ValueError("clinical facts must not contain raw source paths")
            if _EMAIL_RE.search(text):
                raise ValueError("clinical facts must not contain direct contact information")


def _validate_event_report(
    value: object,
    *,
    patient_pseudonym: str,
    eeg_event_id: str,
) -> tuple[dict[str, Any], float, float]:
    report = validate_report_payload(value).to_dict()
    if report["patient_pseudonym"] != patient_pseudonym:
        raise ValueError("event report patient_pseudonym does not match the segment")
    if report["eeg_event_ids"] != [eeg_event_id]:
        raise ValueError("event report must contain exactly the segment EEG event")
    _audit_clinical_fact_boundary(report)
    duration_facts = [
        fact
        for fact in report["facts"]
        if fact["fact_type"] == "recording_duration"
        and fact["state"] in {"present", "uncertain"}
        and isinstance(fact["value"], dict)
    ]
    if len(duration_facts) != 1 or not _same_number(
        duration_facts[0]["value"]["duration_seconds"],
        FIXED_SEGMENT_DURATION_SECONDS,
    ):
        raise ValueError("event report recording_duration must describe the 60-second segment")
    occurrences = [
        fact
        for fact in report["facts"]
        if fact.get("eeg_event_id") == eeg_event_id
        and fact["fact_type"] == "electrographic_event_occurrence"
    ]
    if len(occurrences) != 1:
        raise ValueError("event report requires exactly one occurrence fact")
    occurrence = occurrences[0]["value"]
    if occurrence.get("time_coordinate") != "segment_start_seconds":
        raise ValueError(
            "event report occurrence must explicitly use time_coordinate="
            "segment_start_seconds"
        )
    start = _finite_number(
        occurrence["start_offset_seconds"],
        "event report occurrence start_offset_seconds",
        minimum=0,
        maximum=FIXED_SEGMENT_DURATION_SECONDS,
    )
    event_duration = _finite_number(
        occurrence["duration_seconds"],
        "event report occurrence duration_seconds",
        exclusive_minimum=0,
    )
    stop = start + event_duration
    if stop > FIXED_SEGMENT_DURATION_SECONDS + _TOLERANCE_SECONDS:
        raise ValueError("event report occurrence falls outside the 60-second segment")
    return report, start, stop


def _validate_segment_components(
    *,
    event_report_payload: object,
    waveform_attachment: object,
    research_soz_ranking_receipt: object,
    patient_pseudonym: str,
    eeg_event_id: str,
    source_signal_sha256: str,
    preprocessing_receipt_sha256: str,
    processed_window_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float, float]:
    report, local_start, local_stop = _validate_event_report(
        event_report_payload,
        patient_pseudonym=patient_pseudonym,
        eeg_event_id=eeg_event_id,
    )
    waveform = _validate_waveform_attachment(
        waveform_attachment,
        report_payload=report,
        eeg_event_id=eeg_event_id,
        source_signal_sha256=source_signal_sha256,
        preprocessing_receipt_sha256=preprocessing_receipt_sha256,
        processed_window_sha256=processed_window_sha256,
    )
    evidence_start, evidence_stop = waveform[
        "evidence_interval_seconds_relative_to_anchor"
    ]
    if not _same_number(local_start, evidence_start + FIXED_EVENT_ANCHOR_OFFSET_SECONDS):
        raise ValueError(
            "event report segment-local start does not match anchor-relative waveform evidence"
        )
    if not _same_number(local_stop, evidence_stop + FIXED_EVENT_ANCHOR_OFFSET_SECONDS):
        raise ValueError(
            "event report segment-local stop does not match anchor-relative waveform evidence"
        )
    ranking = _validate_research_soz_ranking(
        research_soz_ranking_receipt,
        processed_window_sha256=processed_window_sha256,
    )
    return report, waveform, ranking, local_start, local_stop


def validate_long_term_event_segment_receipt(payload: object) -> dict[str, Any]:
    """Validate one fixed-window event analysis receipt and return a copy."""

    data = _strict_object(
        payload,
        required=(
            "schema_version",
            "segment_receipt_id",
            "recording_id",
            "patient_pseudonym",
            "source_signal_sha256",
            "recording_duration_seconds",
            "candidate_id",
            "eeg_event_id",
            "candidate_anchor_offset_seconds",
            "requested_window_seconds",
            "segment_start_offset_seconds",
            "segment_stop_offset_seconds",
            "warmup_seconds_available",
            "post_anchor_seconds_available",
            "boundary_policy",
            "processed_window_sha256",
            "preprocessing_receipt_sha256",
            "event_report_payload",
            "waveform_attachment",
            "research_soz_ranking_receipt",
        ),
        context="long-term event segment receipt",
    )
    schema_version = _enum(
        data["schema_version"],
        (EVENT_SEGMENT_RECEIPT_SCHEMA_VERSION,),
        "segment.schema_version",
    )
    recording_id = _identifier(data["recording_id"], "segment.recording_id")
    patient = _identifier(data["patient_pseudonym"], "segment.patient_pseudonym")
    source_hash = _sha256(data["source_signal_sha256"], "segment.source_signal_sha256")
    duration = _finite_number(
        data["recording_duration_seconds"],
        "segment.recording_duration_seconds",
        exclusive_minimum=0,
    )
    candidate_id = _identifier(data["candidate_id"], "segment.candidate_id")
    event_id = _identifier(data["eeg_event_id"], "segment.eeg_event_id")
    anchor = _finite_number(
        data["candidate_anchor_offset_seconds"],
        "segment.candidate_anchor_offset_seconds",
        minimum=0,
        maximum=duration,
    )
    requested_window = _number_pair(
        data["requested_window_seconds"], "segment.requested_window_seconds"
    )
    if not all(
        _same_number(actual, expected)
        for actual, expected in zip(requested_window, FIXED_EVENT_WINDOW_SECONDS)
    ):
        raise ValueError("segment must request the fixed [-12,+48] second window")
    segment_start = _finite_number(
        data["segment_start_offset_seconds"],
        "segment.segment_start_offset_seconds",
        minimum=0,
        maximum=duration,
    )
    segment_stop = _finite_number(
        data["segment_stop_offset_seconds"],
        "segment.segment_stop_offset_seconds",
        minimum=0,
        maximum=duration,
    )
    if not _same_number(segment_start, anchor + FIXED_EVENT_WINDOW_SECONDS[0]):
        raise ValueError("segment_start must equal candidate anchor minus 12 seconds")
    if not _same_number(segment_stop, anchor + FIXED_EVENT_WINDOW_SECONDS[1]):
        raise ValueError("segment_stop must equal candidate anchor plus 48 seconds")
    if not _same_number(segment_stop - segment_start, FIXED_SEGMENT_DURATION_SECONDS):
        raise ValueError("event segment must contain exactly 60 seconds")
    warmup = _finite_number(
        data["warmup_seconds_available"],
        "segment.warmup_seconds_available",
        minimum=0,
    )
    post = _finite_number(
        data["post_anchor_seconds_available"],
        "segment.post_anchor_seconds_available",
        minimum=0,
    )
    if not _same_number(warmup, anchor):
        raise ValueError("warmup_seconds_available must be measured from recording start")
    if not _same_number(post, duration - anchor):
        raise ValueError("post_anchor_seconds_available must be measured to recording stop")
    if (
        warmup < MINIMUM_ANALYZABLE_ANCHOR_SECONDS - _TOLERANCE_SECONDS
        or post < 48.0 - _TOLERANCE_SECONDS
    ):
        raise ValueError("candidate lacks sufficient warmup or post-anchor context")
    boundary_policy = _enum(
        data["boundary_policy"], (BOUNDARY_POLICY,), "segment.boundary_policy"
    )
    processed_hash = _sha256(
        data["processed_window_sha256"], "segment.processed_window_sha256"
    )
    preprocess_hash = _sha256(
        data["preprocessing_receipt_sha256"],
        "segment.preprocessing_receipt_sha256",
    )
    report, waveform, ranking, _, _ = _validate_segment_components(
        event_report_payload=data["event_report_payload"],
        waveform_attachment=data["waveform_attachment"],
        research_soz_ranking_receipt=data["research_soz_ranking_receipt"],
        patient_pseudonym=patient,
        eeg_event_id=event_id,
        source_signal_sha256=source_hash,
        preprocessing_receipt_sha256=preprocess_hash,
        processed_window_sha256=processed_hash,
    )
    return {
        "schema_version": schema_version,
        "segment_receipt_id": _identifier(
            data["segment_receipt_id"], "segment.segment_receipt_id"
        ),
        "recording_id": recording_id,
        "patient_pseudonym": patient,
        "source_signal_sha256": source_hash,
        "recording_duration_seconds": duration,
        "candidate_id": candidate_id,
        "eeg_event_id": event_id,
        "candidate_anchor_offset_seconds": anchor,
        "requested_window_seconds": requested_window,
        "segment_start_offset_seconds": segment_start,
        "segment_stop_offset_seconds": segment_stop,
        "warmup_seconds_available": warmup,
        "post_anchor_seconds_available": post,
        "boundary_policy": boundary_policy,
        "processed_window_sha256": processed_hash,
        "preprocessing_receipt_sha256": preprocess_hash,
        "event_report_payload": report,
        "waveform_attachment": waveform,
        "research_soz_ranking_receipt": ranking,
    }


# Compatibility aliases make the dict-in/dict-out nature explicit to callers.
validate_long_term_seizure_detection_manifest_payload = (
    validate_long_term_seizure_detection_manifest
)
validate_long_term_event_segment_receipt_payload = validate_long_term_event_segment_receipt


__all__ = [
    "BOUNDARY_POLICY",
    "CANDIDATE_SEMANTICS",
    "DETECTION_MANIFEST_SCHEMA_VERSION",
    "EVENT_SEGMENT_RECEIPT_SCHEMA_VERSION",
    "FIXED_EVENT_ANCHOR_OFFSET_SECONDS",
    "FIXED_EVENT_WINDOW_SECONDS",
    "FIXED_SEGMENT_DURATION_SECONDS",
    "FILTERED_LONG_TERM_BUNDLE_SCHEMA_VERSION",
    "MINIMUM_ANALYZABLE_ANCHOR_SECONDS",
    "REQUIRED_CAUSAL_WARMUP_SECONDS",
    "LONG_TERM_BUNDLE_SCHEMA_VERSION",
    "SOZ_INTERPRETATION_STATUS",
    "WAVEFORM_SELECTION_POLICY",
    "canonical_payload_sha256",
    "validate_long_term_event_segment_receipt",
    "validate_long_term_event_segment_receipt_payload",
    "validate_long_term_seizure_detection_manifest",
    "validate_long_term_seizure_detection_manifest_payload",
]
