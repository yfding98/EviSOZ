"""Strict de-identified source context for a long-term clinical EEG record.

``clinical_eeg_long_term_context_v1`` is a sidecar, not an EEG fact ledger.
It deliberately keeps source annotations and manually transcribed spreadsheet
observations outside seizure detection, SOZ ranking, generated narrative, and
the EEG impression.  A deterministic renderer may show the closed codes in a
separately labelled source-context column.

Raw EDF annotation descriptions are accepted only by the in-memory builder.
They are reduced with conservative offline rules to a closed vocabulary and
are never copied into the returned object.  Spreadsheet text is never accepted
at all: an Excel observation can enter only as typed EEG fields accompanied by
an explicit, de-identified review binding.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from src.clinical_eeg_report.schema import (
    canonicalize_derivation,
    canonicalize_electrode,
)


LONG_TERM_CLINICAL_CONTEXT_SCHEMA = "clinical_eeg_long_term_context_v1"
CLINICAL_EEG_LONG_TERM_CONTEXT_SCHEMA = LONG_TERM_CLINICAL_CONTEXT_SCHEMA

EVENT_WINDOW_START_SECONDS = -12.0
EVENT_WINDOW_END_SECONDS = 48.0

EEG_POINT_MARKER_TYPES = frozenset(
    {"event_marker", "eeg_event_marker", "end_marker"}
)
SOURCE_BEHAVIOR_TYPES = frozenset(
    {
        "motor_activity",
        "behavioral_arrest",
        "responsiveness_change",
        "vocalization",
        "eye_or_head_deviation",
        "automotor_activity",
        "autonomic_or_salivation",
    }
)
ANNOTATION_TYPES = (
    "event_marker",
    "eeg_event_marker",
    "end_marker",
    "motor_activity",
    "behavioral_arrest",
    "responsiveness_change",
    "vocalization",
    "eye_or_head_deviation",
    "automotor_activity",
    "autonomic_or_salivation",
)

EXCEL_LATERALITIES = frozenset(
    {"left", "right", "bilateral", "midline", "none", "indeterminate"}
)
EXCEL_REGIONS = frozenset(
    {
        "frontal",
        "temporal",
        "central",
        "parietal",
        "occipital",
        "frontotemporal",
        "centrotemporal",
        "temporoparietal",
        "posterior",
        "diffuse",
        "midline",
        "unknown",
    }
)
EXCEL_PATTERNS = frozenset(
    {
        "low_voltage_fast_activity",
        "rhythmic_activity",
        "repetitive_spikes",
        "electrodecrement",
        "attenuation",
        "spike",
        "sharp_wave",
        "spike_and_slow_wave",
        "sharp_and_slow_wave",
        "polyspike",
        "fast_activity",
        "theta_activity",
        "delta_activity",
        "mixed",
        "indeterminate",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RECORDING_ID_RE = re.compile(r"^(?:PRIV|SYNTH|DEID)-R[A-Z0-9._-]{1,55}$")
_PATIENT_ID_RE = re.compile(r"^(?:PRIV|SYNTH|DEID)-P[A-Z0-9._-]{1,55}$")
_EVENT_ID_RE = re.compile(r"^(?:PRIV|SYNTH|DEID)-E[A-Z0-9._-]{1,55}$")
_SZ_SLOT_RE = re.compile(r"^SZ\d+(?:[-_]\d+)?$", re.IGNORECASE)
_ANNOTATION_ID_RE = re.compile(r"^CTXANN-[0-9a-f]{24}$")
_OBSERVATION_ID_RE = re.compile(r"^XLSOBS-[0-9a-f]{24}$")

_EVENT_MARKER_RE = re.compile(
    r"^(?:(?:SZ|SEIZURE)\s*[-_:]?\s*\d+(?:[-_]\d+)?|"
    r"PATIENT\s*EVENT(?:\s*[-_:]?\s*\d+(?:[-_]\d+)?)?)$",
    re.IGNORECASE,
)
_EEG_EVENT_MARKER_RE = re.compile(
    r"(?:^|[^A-Z0-9])(?:EEG\s*(?:SZ|SEIZURE)|脑电(?:图)?发作)(?:$|[^A-Z0-9])",
    re.IGNORECASE,
)
_END_MARKER_RE = re.compile(
    r"^(?:END|STOP|SZ\s*END|SEIZURE\s*END|EEG\s*(?:SZ\s*)?END|"
    r"结束|终止|发作结束|脑电(?:图)?发作结束)$",
    re.IGNORECASE,
)

# A row mentioning one of these out-of-scope domains is excluded wholesale.
# This is intentionally conservative: a source statement is never split in a
# way that could make a sleep/provocation/cardiac observation appear EEG-only.
_OUT_OF_SCOPE_RE = re.compile(
    r"(?:睡眠|入睡|睡着|睡期|觉醒期|困倦|"
    r"诱发(?:试验)?|过度换气|闪光刺激|光刺激|睁闭眼|"
    r"心电|心率|脉搏|肌电|"
    r"阻抗|校准|电极检查|记录开始|"
    r"(?<![A-Za-z0-9])(?:sleep|asleep|drows(?:y|iness)?|REM|N[123]|"
    r"hyperventilation|photic|ECG|EKG|EMG|heart\s*rate|pulse|"
    r"\d+(?:\.\d+)?\s*bpm|impedance|calibration|electrode\s*check|"
    r"recording\s*start(?:ed)?|montage|clip\s*note|administrative)"
    r"(?![A-Za-z0-9]))",
    re.IGNORECASE,
)
_PROMPT_INJECTION_RE = re.compile(
    r"(?:ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?|"
    r"system\s+prompt|developer\s+message|prompt\s*injection|"
    r"execute\s+(?:this\s+)?command|"
    r"忽略.{0,12}(?:指令|提示)|系统提示|开发者消息|执行.{0,8}命令)",
    re.IGNORECASE,
)
_UNCERTAINTY_RE = re.compile(
    r"(?:疑似|可疑|可能|不确定|待定|[?？]|questionable|possible|uncertain)", re.IGNORECASE
)

_BEHAVIOR_PATTERNS: Mapping[str, tuple[re.Pattern[str], ...]] = {
    "motor_activity": (
        re.compile(
            r"(?:肢体.{0,5}(?:抽动|抖动|强直|阵挛)|抽搐|肌张力障碍|"
            r"(?:非对称|双侧|四肢|面部|面肌|口角|上肢|下肢|手|足)?.{0,3}(?:强直|阵挛)|"
            r"过度运动|复杂运动|蹬踏|翻身|"
            r"(?<![A-Za-z0-9])(?:motor\s+activity|jerk(?:ing)?|shak(?:e|ing)|"
            r"tonic|clonic|myoclonic|convulsion)(?![A-Za-z0-9]))",
            re.IGNORECASE,
        ),
    ),
    "behavioral_arrest": (
        re.compile(
            r"(?:行为停止|动作(?:停止|停顿|减少|减缓)|活动停止|运动减少|呆滞|凝视|"
            r"(?<![A-Za-z0-9])behavioral\s+arrest(?![A-Za-z0-9]))",
            re.IGNORECASE,
        ),
    ),
    "responsiveness_change": (
        re.compile(
            r"(?:呼之不应|不能应答|不能说名字|意识(?:不清|恢复|半恢复)|"
            r"反应(?:性)?(?:下降|减弱|改变|消失)|应答(?:下降|消失)|"
            r"(?<![A-Za-z0-9])(?:unresponsive|not\s+responding|"
            r"responsiveness\s+(?:change|decrease|loss))(?![A-Za-z0-9]))",
            re.IGNORECASE,
        ),
    ),
    "vocalization": (
        re.compile(
            r"(?:发声|喊叫|叫喊|尖叫|发笑|哭泣|胡言乱语|"
            r"(?<![A-Za-z0-9])vocali[sz]ation(?![A-Za-z0-9]))",
            re.IGNORECASE,
        ),
    ),
    "eye_or_head_deviation": (
        re.compile(
            r"(?:(?:头|眼|头眼|双眼).{0,8}(?:偏向|偏转|偏斜|向左|向右|左视|右视|凝视)|"
            r"(?:向左|向右|左侧|右侧).{0,3}凝视|"
            r"(?<![A-Za-z0-9])(?:eye|head|eye\s+and\s+head)\s+deviation"
            r"(?![A-Za-z0-9]))",
            re.IGNORECASE,
        ),
    ),
    "automotor_activity": (
        re.compile(
            r"(?:自动症?|自动运动|口咽自动|口部反复动作|咂嘴|抿嘴|瘪嘴|咀嚼|吞咽|摸索|"
            r"(?<![A-Za-z0-9])(?:automotor(?:\s+activity)?|automatism|"
            r"lip\s+smacking|fumbling)(?![A-Za-z0-9]))",
            re.IGNORECASE,
        ),
    ),
    "autonomic_or_salivation": (
        re.compile(
            r"(?:流涎|(?<![A-Za-z0-9])(?:salivation|drooling)(?![A-Za-z0-9]))",
            re.IGNORECASE,
        ),
    ),
}

_TOP_LEVEL_KEYS = {
    "schema_version",
    "recording_id",
    "patient_id",
    "source_signal_sha256",
    "recording_duration_seconds",
    "source_receipts",
    "annotations",
    "excel_onset_observations",
    "event_associations",
    "unbound_annotation_ids",
    "frozen_event_registry_receipt",
    "association_policy",
    "exclusion_summary",
    "claim_boundary",
}
_ANNOTATION_KEYS = {
    "annotation_id",
    "annotation_type",
    "annotation_scope",
    "recording_offset_seconds",
    "source_row",
    "source_row_sha256",
    "source_file_sha256",
    "uncertain",
    "verification_status",
    "llm_eligible",
}
_EXCEL_KEYS = {
    "observation_id",
    "workbook_sha256",
    "sheet_index",
    "source_row",
    "source_row_sha256",
    "sz_slot",
    "recording_id",
    "eeg_event_id",
    "binding_verified",
    "physician_verified",
    "physician_signed_review_sha256",
    "typed_eeg_fields",
    "uncertain",
    "verification_status",
    "raw_text_included",
    "llm_eligible",
}
_TYPED_EEG_KEYS = {
    "electrodes",
    "derivations",
    "laterality",
    "regions",
    "patterns",
}
_ASSOCIATION_KEYS = {
    "eeg_event_id",
    "recording_id",
    "patient_id",
    "source_signal_sha256",
    "event_anchor_recording_seconds",
    "annotation_links",
    "excel_observation_ids",
    "annotation_used_for_detection",
    "annotation_used_for_ranking",
    "annotation_used_for_narrative",
    "annotation_used_for_impression",
}

_ASSOCIATION_POLICY = {
    "post_freeze_only": True,
    "temporal_window_seconds": {
        "start": EVENT_WINDOW_START_SECONDS,
        "end": EVENT_WINDOW_END_SECONDS,
    },
    "ambiguous_temporal_matches_remain_unbound": True,
    "unbound_annotations_create_events": False,
}
_CLAIM_BOUNDARY = {
    "raw_annotation_text_released": False,
    "source_path_released": False,
    "direct_identity_released": False,
    "sleep_context_included": False,
    "provocation_context_included": False,
    "cardiac_or_emg_context_included": False,
    "annotations_used_for_detection": False,
    "annotations_used_for_ranking": False,
    "annotations_used_for_narrative": False,
    "annotations_used_for_impression": False,
    "excel_used_for_detection": False,
    "excel_used_for_ranking": False,
    "excel_used_for_narrative": False,
    "excel_used_for_impression": False,
    "unbound_annotations_create_events": False,
    "excel_automatically_bound": False,
    "physician_verification_inferred": False,
    "llm_access_allowed": False,
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_object(
    value: object,
    *,
    required: set[str],
    optional: set[str] | None = None,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    allowed = required | (optional or set())
    keys = set(value)
    missing = required - keys
    extra = keys - allowed
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{context} has unknown keys: {sorted(extra)}")
    return {str(key): deepcopy(item) for key, item in value.items()}


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _identifier(value: object, pattern: re.Pattern[str], context: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{context} must be a supported de-identified ID")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{context} must be an integer")
    try:
        number = int(value)  # type: ignore[arg-type]
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} must be an integer") from exc
    if not math.isfinite(numeric) or not math.isclose(
        numeric, float(number), rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(f"{context} must be an integer")
    if number < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return number


def _number(value: object, context: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{context} must be numeric")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    if number < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return number


def _bool(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{context} must be boolean")
    return value


def _recording_offset(value: object, duration: float, context: str) -> float:
    offset = _number(value, context)
    if offset > duration:
        raise ValueError(f"{context} exceeds recording duration")
    return offset


def _annotation_scope(annotation_type: str) -> str:
    if annotation_type in EEG_POINT_MARKER_TYPES:
        return "eeg_point_marker"
    if annotation_type in SOURCE_BEHAVIOR_TYPES:
        return "source_observed_behavior"
    raise ValueError("unsupported annotation type")


def _description_is_excluded(description: str) -> bool:
    return bool(
        _OUT_OF_SCOPE_RE.search(description) or _PROMPT_INJECTION_RE.search(description)
    )


def classify_edf_annotation_description(description: str) -> tuple[str, ...]:
    """Reduce one raw description to zero or more closed, non-text codes.

    The function is deliberately high precision.  Any mention of sleep,
    provocation, ECG/heart rate, EMG, acquisition administration, or prompt
    injection makes the whole row ineligible.
    """

    if not isinstance(description, str):
        raise TypeError("EDF annotation description must be a string")
    text = re.sub(r"\s+", " ", description.strip())
    if not text or _description_is_excluded(text):
        return ()
    matches: set[str] = set()
    if _END_MARKER_RE.fullmatch(text):
        matches.add("end_marker")
    elif _EVENT_MARKER_RE.fullmatch(text):
        matches.add("event_marker")
    elif _EEG_EVENT_MARKER_RE.search(text):
        matches.add("eeg_event_marker")
    for code, patterns in _BEHAVIOR_PATTERNS.items():
        if any(pattern.search(text) for pattern in patterns):
            matches.add(code)
    return tuple(code for code in ANNOTATION_TYPES if code in matches)


def _canonical_string_list(
    value: object,
    *,
    allowed: frozenset[str],
    context: str,
) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or item not in allowed:
            raise ValueError(f"{context} contains an unsupported value")
        if item in result:
            raise ValueError(f"{context} contains duplicates")
        result.append(item)
    return sorted(result)


def _typed_eeg_fields(value: object, *, normalize: bool) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=_TYPED_EEG_KEYS,
        context="typed_eeg_fields",
    )
    electrodes = data["electrodes"]
    derivations = data["derivations"]
    if not isinstance(electrodes, list) or not isinstance(derivations, list):
        raise TypeError("typed EEG electrodes and derivations must be lists")
    normalized_electrodes: list[str] = []
    for raw in electrodes:
        canonical = canonicalize_electrode(raw)
        if not normalize and raw != canonical:
            raise ValueError("typed EEG electrodes must already be canonical")
        if canonical in normalized_electrodes:
            raise ValueError("typed EEG electrodes contain duplicates")
        normalized_electrodes.append(canonical)
    normalized_derivations: list[str] = []
    for raw in derivations:
        canonical = canonicalize_derivation(raw)
        if not normalize and raw != canonical:
            raise ValueError("typed EEG derivations must already be canonical")
        if canonical in normalized_derivations:
            raise ValueError("typed EEG derivations contain duplicates")
        normalized_derivations.append(canonical)
    laterality = data["laterality"]
    if not isinstance(laterality, str) or laterality not in EXCEL_LATERALITIES:
        raise ValueError("typed EEG laterality is unsupported")
    regions = _canonical_string_list(
        data["regions"], allowed=EXCEL_REGIONS, context="typed_eeg_fields.regions"
    )
    patterns = _canonical_string_list(
        data["patterns"], allowed=EXCEL_PATTERNS, context="typed_eeg_fields.patterns"
    )
    if not normalized_electrodes and not normalized_derivations and not regions and not patterns:
        raise ValueError("typed EEG fields contain no EEG observation")
    return {
        "electrodes": sorted(normalized_electrodes),
        "derivations": sorted(normalized_derivations),
        "laterality": laterality,
        "regions": regions,
        "patterns": patterns,
    }


def _annotation_id_payload(
    *,
    recording_id: str,
    source_file_sha256: str,
    source_row: int,
    source_row_sha256: str,
    recording_offset_seconds: float,
    annotation_type: str,
) -> dict[str, object]:
    return {
        "recording_id": recording_id,
        "source_file_sha256": source_file_sha256,
        "source_row": source_row,
        "source_row_sha256": source_row_sha256,
        "recording_offset_seconds": recording_offset_seconds,
        "annotation_type": annotation_type,
    }


def _validate_annotation(
    value: object,
    *,
    recording_id: str,
    duration: float,
    edf_annotations_sha256: str,
) -> dict[str, Any]:
    data = _strict_object(
        value, required=_ANNOTATION_KEYS, context="long-term context annotation"
    )
    annotation_id = data["annotation_id"]
    if not isinstance(annotation_id, str) or _ANNOTATION_ID_RE.fullmatch(annotation_id) is None:
        raise ValueError("annotation_id is invalid")
    annotation_type = data["annotation_type"]
    if annotation_type not in set(ANNOTATION_TYPES):
        raise ValueError("annotation_type is unsupported")
    if data["annotation_scope"] != _annotation_scope(annotation_type):
        raise ValueError("annotation scope/type mismatch")
    offset = _recording_offset(
        data["recording_offset_seconds"], duration, "annotation recording offset"
    )
    source_row = _integer(data["source_row"], "annotation source row", minimum=1)
    row_sha = _sha256(data["source_row_sha256"], "annotation source row SHA-256")
    file_sha = _sha256(data["source_file_sha256"], "annotation source file SHA-256")
    if file_sha != edf_annotations_sha256:
        raise ValueError("annotation source file receipt mismatch")
    expected_id = "CTXANN-" + _canonical_sha256(
        _annotation_id_payload(
            recording_id=recording_id,
            source_file_sha256=file_sha,
            source_row=source_row,
            source_row_sha256=row_sha,
            recording_offset_seconds=offset,
            annotation_type=annotation_type,
        )
    )[:24]
    if annotation_id != expected_id:
        raise ValueError("annotation_id does not bind its provenance")
    uncertain = _bool(data["uncertain"], "annotation uncertain")
    if data["verification_status"] != "source_transcribed_unverified":
        raise ValueError("source annotation verification was promoted")
    if data["llm_eligible"] is not False:
        raise ValueError("source annotation cannot enter the LLM")
    return {
        "annotation_id": annotation_id,
        "annotation_type": annotation_type,
        "annotation_scope": _annotation_scope(annotation_type),
        "recording_offset_seconds": offset,
        "source_row": source_row,
        "source_row_sha256": row_sha,
        "source_file_sha256": file_sha,
        "uncertain": uncertain,
        "verification_status": "source_transcribed_unverified",
        "llm_eligible": False,
    }


def _observation_id_payload(value: Mapping[str, object]) -> dict[str, object]:
    return {
        key: deepcopy(value[key])
        for key in (
            "workbook_sha256",
            "sheet_index",
            "source_row",
            "source_row_sha256",
            "sz_slot",
            "recording_id",
            "eeg_event_id",
            "binding_verified",
            "physician_verified",
            "physician_signed_review_sha256",
            "typed_eeg_fields",
            "uncertain",
        )
    }


def _validate_excel_observation(
    value: object,
    *,
    recording_id: str,
    workbook_hashes: set[str],
    normalize_typed_fields: bool = False,
) -> dict[str, Any]:
    data = _strict_object(
        value, required=_EXCEL_KEYS, context="Excel onset observation"
    )
    observation_id = data["observation_id"]
    if not isinstance(observation_id, str) or _OBSERVATION_ID_RE.fullmatch(observation_id) is None:
        raise ValueError("Excel observation_id is invalid")
    workbook_sha = _sha256(data["workbook_sha256"], "Excel workbook SHA-256")
    if workbook_sha not in workbook_hashes:
        raise ValueError("Excel observation workbook is not receipted")
    sheet_index = _integer(data["sheet_index"], "Excel sheet index")
    source_row = _integer(data["source_row"], "Excel source row", minimum=1)
    source_row_sha = _sha256(data["source_row_sha256"], "Excel source row SHA-256")
    sz_slot = data["sz_slot"]
    if not isinstance(sz_slot, str) or _SZ_SLOT_RE.fullmatch(sz_slot) is None:
        raise ValueError("Excel SZ slot is invalid")
    if sz_slot != sz_slot.upper():
        raise ValueError("Excel SZ slot must be canonical uppercase")
    bound_recording = _identifier(
        data["recording_id"], _RECORDING_ID_RE, "Excel recording_id"
    )
    if bound_recording != recording_id:
        raise ValueError("Excel observation recording binding mismatch")
    eeg_event_id = _identifier(data["eeg_event_id"], _EVENT_ID_RE, "Excel eeg_event_id")
    if data["binding_verified"] is not True:
        raise ValueError("Excel observation requires an explicit verified binding")
    physician_verified = _bool(
        data["physician_verified"], "Excel physician_verified"
    )
    signed_receipt = data["physician_signed_review_sha256"]
    if physician_verified:
        signed_receipt = _sha256(
            signed_receipt, "physician signed review SHA-256"
        )
        expected_verification = "physician_verified"
    else:
        if signed_receipt is not None:
            raise ValueError("unsigned Excel observation cannot carry a physician receipt")
        expected_verification = "source_transcribed_unverified"
    if data["verification_status"] != expected_verification:
        raise ValueError("Excel verification status/receipt mismatch")
    typed_fields = _typed_eeg_fields(
        data["typed_eeg_fields"], normalize=normalize_typed_fields
    )
    uncertain = _bool(data["uncertain"], "Excel observation uncertain")
    if data["raw_text_included"] is not False:
        raise ValueError("Excel observation leaks free text")
    if data["llm_eligible"] is not False:
        raise ValueError("Excel observation cannot enter the LLM")
    normalized_for_id: dict[str, object] = {
        "workbook_sha256": workbook_sha,
        "sheet_index": sheet_index,
        "source_row": source_row,
        "source_row_sha256": source_row_sha,
        "sz_slot": sz_slot,
        "recording_id": bound_recording,
        "eeg_event_id": eeg_event_id,
        "binding_verified": True,
        "physician_verified": physician_verified,
        "physician_signed_review_sha256": signed_receipt,
        "typed_eeg_fields": typed_fields,
        "uncertain": uncertain,
    }
    expected_id = "XLSOBS-" + _canonical_sha256(
        _observation_id_payload(normalized_for_id)
    )[:24]
    if observation_id != expected_id:
        raise ValueError("Excel observation_id does not bind its provenance")
    return {
        "observation_id": observation_id,
        **normalized_for_id,
        "verification_status": expected_verification,
        "raw_text_included": False,
        "llm_eligible": False,
    }


def _source_receipts(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required={"edf_annotations_sha256", "workbook_sha256s"},
        context="long-term context source_receipts",
    )
    edf_sha = _sha256(data["edf_annotations_sha256"], "EDF annotations SHA-256")
    workbook_values = data["workbook_sha256s"]
    if not isinstance(workbook_values, list):
        raise TypeError("workbook_sha256s must be a list")
    workbook_hashes = [
        _sha256(item, "Excel workbook SHA-256") for item in workbook_values
    ]
    if len(workbook_hashes) != len(set(workbook_hashes)):
        raise ValueError("workbook_sha256s contains duplicates")
    return {
        "edf_annotations_sha256": edf_sha,
        "workbook_sha256s": sorted(workbook_hashes),
    }


def _normalize_frozen_events(
    frozen_events: Sequence[Mapping[str, object]],
    *,
    recording_id: str,
    patient_id: str,
    source_signal_sha256: str,
    duration: float,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(frozen_events):
        event = _strict_object(
            raw,
            required={
                "eeg_event_id",
                "recording_id",
                "patient_id",
                "source_signal_sha256",
                "event_anchor_recording_seconds",
            },
            context=f"frozen event {index}",
        )
        event_id = _identifier(event["eeg_event_id"], _EVENT_ID_RE, "eeg_event_id")
        if _identifier(event["recording_id"], _RECORDING_ID_RE, "event recording_id") != recording_id:
            raise ValueError("frozen event recording binding mismatch")
        if _identifier(event["patient_id"], _PATIENT_ID_RE, "event patient_id") != patient_id:
            raise ValueError("frozen event patient binding mismatch")
        if _sha256(event["source_signal_sha256"], "event signal SHA-256") != source_signal_sha256:
            raise ValueError("frozen event source-signal binding mismatch")
        anchor = _recording_offset(
            event["event_anchor_recording_seconds"], duration, "event anchor"
        )
        normalized.append(
            {
                "eeg_event_id": event_id,
                "recording_id": recording_id,
                "patient_id": patient_id,
                "source_signal_sha256": source_signal_sha256,
                "event_anchor_recording_seconds": anchor,
            }
        )
    event_ids = [item["eeg_event_id"] for item in normalized]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("frozen events repeat an eeg_event_id")
    anchors = [item["event_anchor_recording_seconds"] for item in normalized]
    if len(anchors) != len(set(anchors)):
        raise ValueError("frozen events repeat an event anchor")
    return sorted(
        normalized,
        key=lambda item: (item["event_anchor_recording_seconds"], item["eeg_event_id"]),
    )


def _event_registry_receipt(
    events: Sequence[Mapping[str, object]],
    *,
    recording_id: str,
    patient_id: str,
    source_signal_sha256: str,
) -> dict[str, object]:
    registry_payload = {
        "recording_id": recording_id,
        "patient_id": patient_id,
        "source_signal_sha256": source_signal_sha256,
        "events": [
            {
                "eeg_event_id": item["eeg_event_id"],
                "event_anchor_recording_seconds": item[
                    "event_anchor_recording_seconds"
                ],
            }
            for item in events
        ],
    }
    return {
        "post_freeze": True,
        "event_count": len(events),
        "registry_sha256": _canonical_sha256(registry_payload),
    }


def _associate_context(
    context: Mapping[str, object],
    *,
    frozen_events: Sequence[Mapping[str, object]],
    explicit_annotation_bindings: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    result = deepcopy(dict(context))
    recording_id = result["recording_id"]
    patient_id = result["patient_id"]
    signal_sha = result["source_signal_sha256"]
    duration = float(result["recording_duration_seconds"])
    events = _normalize_frozen_events(
        frozen_events,
        recording_id=recording_id,
        patient_id=patient_id,
        source_signal_sha256=signal_sha,
        duration=duration,
    )
    event_ids = {item["eeg_event_id"] for item in events}
    annotations = result["annotations"]
    annotation_by_id = {item["annotation_id"]: item for item in annotations}
    explicit_by_annotation: dict[str, str] = {}
    for index, raw in enumerate(explicit_annotation_bindings):
        binding = _strict_object(
            raw,
            required={"annotation_id", "eeg_event_id"},
            context=f"explicit annotation binding {index}",
        )
        annotation_id = binding["annotation_id"]
        if not isinstance(annotation_id, str) or annotation_id not in annotation_by_id:
            raise ValueError("explicit binding references an unknown annotation_id")
        event_id = _identifier(binding["eeg_event_id"], _EVENT_ID_RE, "bound eeg_event_id")
        if event_id not in event_ids:
            raise ValueError("explicit binding references an unknown frozen event")
        if annotation_id in explicit_by_annotation:
            raise ValueError("annotation has duplicate explicit bindings")
        explicit_by_annotation[annotation_id] = event_id

    links_by_event: dict[str, list[dict[str, str]]] = {
        event_id: [] for event_id in event_ids
    }
    unbound: list[str] = []
    for annotation in annotations:
        annotation_id = annotation["annotation_id"]
        if annotation_id in explicit_by_annotation:
            links_by_event[explicit_by_annotation[annotation_id]].append(
                {
                    "annotation_id": annotation_id,
                    "association_method": "explicit_binding",
                }
            )
            continue
        offset = float(annotation["recording_offset_seconds"])
        temporal_matches = [
            event["eeg_event_id"]
            for event in events
            if EVENT_WINDOW_START_SECONDS
            <= offset - float(event["event_anchor_recording_seconds"])
            <= EVENT_WINDOW_END_SECONDS
        ]
        if len(temporal_matches) == 1:
            links_by_event[temporal_matches[0]].append(
                {
                    "annotation_id": annotation_id,
                    "association_method": "temporal_window",
                }
            )
        else:
            unbound.append(annotation_id)

    excel_by_event: dict[str, list[str]] = {event_id: [] for event_id in event_ids}
    for observation in result["excel_onset_observations"]:
        event_id = observation["eeg_event_id"]
        if event_id not in event_ids:
            raise ValueError("Excel binding references an unknown frozen event")
        excel_by_event[event_id].append(observation["observation_id"])

    associations: list[dict[str, Any]] = []
    for event in events:
        event_id = event["eeg_event_id"]
        associations.append(
            {
                **event,
                "annotation_links": sorted(
                    links_by_event[event_id], key=lambda item: item["annotation_id"]
                ),
                "excel_observation_ids": sorted(excel_by_event[event_id]),
                "annotation_used_for_detection": False,
                "annotation_used_for_ranking": False,
                "annotation_used_for_narrative": False,
                "annotation_used_for_impression": False,
            }
        )
    result["event_associations"] = associations
    result["unbound_annotation_ids"] = sorted(unbound)
    result["frozen_event_registry_receipt"] = _event_registry_receipt(
        events,
        recording_id=recording_id,
        patient_id=patient_id,
        source_signal_sha256=signal_sha,
    )
    return result


def _validate_associations(
    associations: object,
    *,
    annotations: Sequence[Mapping[str, object]],
    excel_observations: Sequence[Mapping[str, object]],
    recording_id: str,
    patient_id: str,
    source_signal_sha256: str,
    duration: float,
    declared_unbound: object,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(associations, list):
        raise TypeError("event_associations must be a list")
    annotation_by_id = {item["annotation_id"]: item for item in annotations}
    observation_by_id = {item["observation_id"]: item for item in excel_observations}
    normalized: list[dict[str, Any]] = []
    linked_annotations: dict[str, tuple[str, str]] = {}
    linked_observations: dict[str, str] = {}
    for index, raw in enumerate(associations):
        data = _strict_object(
            raw,
            required=_ASSOCIATION_KEYS,
            context=f"event association {index}",
        )
        event_id = _identifier(data["eeg_event_id"], _EVENT_ID_RE, "associated eeg_event_id")
        if _identifier(data["recording_id"], _RECORDING_ID_RE, "association recording_id") != recording_id:
            raise ValueError("event association recording binding mismatch")
        if _identifier(data["patient_id"], _PATIENT_ID_RE, "association patient_id") != patient_id:
            raise ValueError("event association patient binding mismatch")
        if _sha256(data["source_signal_sha256"], "association signal SHA-256") != source_signal_sha256:
            raise ValueError("event association source-signal binding mismatch")
        anchor = _recording_offset(
            data["event_anchor_recording_seconds"], duration, "associated event anchor"
        )
        links = data["annotation_links"]
        if not isinstance(links, list):
            raise TypeError("annotation_links must be a list")
        normalized_links: list[dict[str, str]] = []
        for link in links:
            link_data = _strict_object(
                link,
                required={"annotation_id", "association_method"},
                context="annotation association link",
            )
            annotation_id = link_data["annotation_id"]
            if not isinstance(annotation_id, str) or annotation_id not in annotation_by_id:
                raise ValueError("association references an unknown annotation")
            method = link_data["association_method"]
            if method not in {"temporal_window", "explicit_binding"}:
                raise ValueError("annotation association method is unsupported")
            if annotation_id in linked_annotations:
                raise ValueError("annotation is associated more than once")
            linked_annotations[annotation_id] = (event_id, method)
            normalized_links.append(
                {"annotation_id": annotation_id, "association_method": method}
            )
        observation_ids = data["excel_observation_ids"]
        if not isinstance(observation_ids, list):
            raise TypeError("excel_observation_ids must be a list")
        normalized_observation_ids: list[str] = []
        for observation_id in observation_ids:
            if not isinstance(observation_id, str) or observation_id not in observation_by_id:
                raise ValueError("association references an unknown Excel observation")
            if observation_by_id[observation_id]["eeg_event_id"] != event_id:
                raise ValueError("Excel observation/event association mismatch")
            if observation_id in linked_observations:
                raise ValueError("Excel observation is associated more than once")
            linked_observations[observation_id] = event_id
            normalized_observation_ids.append(observation_id)
        for flag in (
            "annotation_used_for_detection",
            "annotation_used_for_ranking",
            "annotation_used_for_narrative",
            "annotation_used_for_impression",
        ):
            if data[flag] is not False:
                raise ValueError(f"{flag} must remain false")
        normalized.append(
            {
                "eeg_event_id": event_id,
                "recording_id": recording_id,
                "patient_id": patient_id,
                "source_signal_sha256": source_signal_sha256,
                "event_anchor_recording_seconds": anchor,
                "annotation_links": sorted(
                    normalized_links, key=lambda item: item["annotation_id"]
                ),
                "excel_observation_ids": sorted(normalized_observation_ids),
                "annotation_used_for_detection": False,
                "annotation_used_for_ranking": False,
                "annotation_used_for_narrative": False,
                "annotation_used_for_impression": False,
            }
        )
    event_ids = [item["eeg_event_id"] for item in normalized]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event_associations repeats an eeg_event_id")
    anchors = [item["event_anchor_recording_seconds"] for item in normalized]
    if len(anchors) != len(set(anchors)):
        raise ValueError("event_associations repeats an event anchor")

    # A temporal link must have exactly one possible frozen event.  Ambiguous
    # overlaps remain on the recording timeline instead of being guessed.
    for annotation_id, (linked_event_id, method) in linked_annotations.items():
        if method != "temporal_window":
            continue
        offset = float(annotation_by_id[annotation_id]["recording_offset_seconds"])
        candidates = [
            item["eeg_event_id"]
            for item in normalized
            if EVENT_WINDOW_START_SECONDS
            <= offset - float(item["event_anchor_recording_seconds"])
            <= EVENT_WINDOW_END_SECONDS
        ]
        if candidates != [linked_event_id]:
            raise ValueError("temporal association is outside or ambiguous within the event window")

    if set(linked_observations) != set(observation_by_id):
        raise ValueError("every Excel observation must bind exactly one frozen event")
    if not isinstance(declared_unbound, list):
        raise TypeError("unbound_annotation_ids must be a list")
    unbound: list[str] = []
    for annotation_id in declared_unbound:
        if not isinstance(annotation_id, str) or annotation_id not in annotation_by_id:
            raise ValueError("unbound_annotation_ids references an unknown annotation")
        if annotation_id in unbound:
            raise ValueError("unbound_annotation_ids contains duplicates")
        unbound.append(annotation_id)
    expected_unbound = set(annotation_by_id) - set(linked_annotations)
    if set(unbound) != expected_unbound:
        raise ValueError("bound/unbound annotation partition is incomplete")
    return (
        sorted(
            normalized,
            key=lambda item: (
                item["event_anchor_recording_seconds"],
                item["eeg_event_id"],
            ),
        ),
        sorted(unbound),
    )


def validate_long_term_clinical_context(
    value: object,
    *,
    expected_recording_id: str | None = None,
    expected_patient_id: str | None = None,
    expected_source_signal_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate and canonicalize a ``clinical_eeg_long_term_context_v1`` sidecar."""

    data = _strict_object(
        value, required=_TOP_LEVEL_KEYS, context="long-term clinical context"
    )
    if data["schema_version"] != LONG_TERM_CLINICAL_CONTEXT_SCHEMA:
        raise ValueError("long-term clinical context schema drifted")
    recording_id = _identifier(data["recording_id"], _RECORDING_ID_RE, "recording_id")
    patient_id = _identifier(data["patient_id"], _PATIENT_ID_RE, "patient_id")
    signal_sha = _sha256(data["source_signal_sha256"], "source signal SHA-256")
    duration = _number(data["recording_duration_seconds"], "recording duration")
    if duration <= 0:
        raise ValueError("recording duration must be positive")
    if expected_recording_id is not None and recording_id != _identifier(
        expected_recording_id, _RECORDING_ID_RE, "expected recording_id"
    ):
        raise ValueError("long-term context recording binding mismatch")
    if expected_patient_id is not None and patient_id != _identifier(
        expected_patient_id, _PATIENT_ID_RE, "expected patient_id"
    ):
        raise ValueError("long-term context patient binding mismatch")
    if expected_source_signal_sha256 is not None and signal_sha != _sha256(
        expected_source_signal_sha256, "expected source signal SHA-256"
    ):
        raise ValueError("long-term context source-signal binding mismatch")
    receipts = _source_receipts(data["source_receipts"])

    raw_annotations = data["annotations"]
    if not isinstance(raw_annotations, list):
        raise TypeError("annotations must be a list")
    annotations = [
        _validate_annotation(
            item,
            recording_id=recording_id,
            duration=duration,
            edf_annotations_sha256=receipts["edf_annotations_sha256"],
        )
        for item in raw_annotations
    ]
    annotation_ids = [item["annotation_id"] for item in annotations]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("annotations contains duplicate annotation IDs")
    row_state: dict[int, tuple[str, float, bool]] = {}
    row_hash_owner: dict[str, int] = {}
    row_types: set[tuple[int, str]] = set()
    for item in annotations:
        row = item["source_row"]
        state = (
            item["source_row_sha256"],
            item["recording_offset_seconds"],
            item["uncertain"],
        )
        if row in row_state and row_state[row] != state:
            raise ValueError("annotation items from one source row disagree")
        row_state[row] = state
        row_sha = item["source_row_sha256"]
        if row_sha in row_hash_owner and row_hash_owner[row_sha] != row:
            raise ValueError("annotation source row SHA-256 is reused")
        row_hash_owner[row_sha] = row
        row_type = (row, item["annotation_type"])
        if row_type in row_types:
            raise ValueError("annotation row repeats a closed type")
        row_types.add(row_type)
    annotations.sort(
        key=lambda item: (
            item["recording_offset_seconds"],
            item["source_row"],
            ANNOTATION_TYPES.index(item["annotation_type"]),
        )
    )

    raw_excel = data["excel_onset_observations"]
    if not isinstance(raw_excel, list):
        raise TypeError("excel_onset_observations must be a list")
    workbook_hashes = set(receipts["workbook_sha256s"])
    excel_observations = [
        _validate_excel_observation(
            item, recording_id=recording_id, workbook_hashes=workbook_hashes
        )
        for item in raw_excel
    ]
    observation_ids = [item["observation_id"] for item in excel_observations]
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError("Excel observations contain duplicate IDs")
    source_bindings = [
        (
            item["workbook_sha256"],
            item["sheet_index"],
            item["source_row"],
            item["sz_slot"],
        )
        for item in excel_observations
    ]
    if len(source_bindings) != len(set(source_bindings)):
        raise ValueError("Excel review source binding is duplicated")
    excel_observations.sort(
        key=lambda item: (
            item["workbook_sha256"],
            item["sheet_index"],
            item["source_row"],
            item["sz_slot"],
        )
    )

    associations, unbound = _validate_associations(
        data["event_associations"],
        annotations=annotations,
        excel_observations=excel_observations,
        recording_id=recording_id,
        patient_id=patient_id,
        source_signal_sha256=signal_sha,
        duration=duration,
        declared_unbound=data["unbound_annotation_ids"],
    )
    registry_receipt = _strict_object(
        data["frozen_event_registry_receipt"],
        required={"post_freeze", "event_count", "registry_sha256"},
        context="frozen event registry receipt",
    )
    if registry_receipt["post_freeze"] is not True:
        raise ValueError("event registry must be frozen before context association")
    event_count = _integer(
        registry_receipt["event_count"], "frozen event registry count"
    )
    registry_sha = _sha256(
        registry_receipt["registry_sha256"], "frozen event registry SHA-256"
    )
    expected_registry_receipt = _event_registry_receipt(
        associations,
        recording_id=recording_id,
        patient_id=patient_id,
        source_signal_sha256=signal_sha,
    )
    if (
        event_count != expected_registry_receipt["event_count"]
        or registry_sha != expected_registry_receipt["registry_sha256"]
    ):
        raise ValueError("frozen event registry receipt does not match associations")
    if data["association_policy"] != _ASSOCIATION_POLICY:
        raise ValueError("long-term context association policy drifted")

    exclusion = _strict_object(
        data["exclusion_summary"],
        required={
            "raw_annotation_rows",
            "mapped_source_rows",
            "mapped_annotation_items",
            "excluded_non_eeg_or_administrative_rows",
            "unmapped_rows",
            "excel_review_bindings",
        },
        context="long-term context exclusion_summary",
    )
    exclusion = {
        key: _integer(value, f"exclusion_summary.{key}")
        for key, value in exclusion.items()
    }
    if exclusion["mapped_source_rows"] != len(row_state):
        raise ValueError("mapped source row count is inconsistent")
    if exclusion["mapped_annotation_items"] != len(annotations):
        raise ValueError("mapped annotation item count is inconsistent")
    if exclusion["excel_review_bindings"] != len(excel_observations):
        raise ValueError("Excel binding count is inconsistent")
    if exclusion["raw_annotation_rows"] != (
        exclusion["mapped_source_rows"]
        + exclusion["excluded_non_eeg_or_administrative_rows"]
        + exclusion["unmapped_rows"]
    ):
        raise ValueError("annotation exclusion partition is inconsistent")
    if data["claim_boundary"] != _CLAIM_BOUNDARY:
        raise ValueError("long-term context claim boundary drifted")

    return {
        "schema_version": LONG_TERM_CLINICAL_CONTEXT_SCHEMA,
        "recording_id": recording_id,
        "patient_id": patient_id,
        "source_signal_sha256": signal_sha,
        "recording_duration_seconds": duration,
        "source_receipts": receipts,
        "annotations": annotations,
        "excel_onset_observations": excel_observations,
        "event_associations": associations,
        "unbound_annotation_ids": unbound,
        "frozen_event_registry_receipt": expected_registry_receipt,
        "association_policy": deepcopy(_ASSOCIATION_POLICY),
        "exclusion_summary": exclusion,
        "claim_boundary": deepcopy(_CLAIM_BOUNDARY),
    }


def _build_excel_observation(
    raw: Mapping[str, object],
    *,
    recording_id: str,
    workbook_hashes: set[str],
) -> dict[str, Any]:
    data = _strict_object(
        raw,
        required={
            "workbook_sha256",
            "sheet_index",
            "source_row",
            "source_row_sha256",
            "sz_slot",
            "recording_id",
            "eeg_event_id",
            "binding_verified",
            "physician_verified",
            "physician_signed_review_sha256",
            "typed_eeg_fields",
            "uncertain",
        },
        context="Excel review binding input",
    )
    workbook_sha = _sha256(data["workbook_sha256"], "Excel workbook SHA-256")
    if workbook_sha not in workbook_hashes:
        raise ValueError("Excel review binding workbook is not receipted")
    sheet_index = _integer(data["sheet_index"], "Excel sheet index")
    source_row = _integer(data["source_row"], "Excel source row", minimum=1)
    row_sha = _sha256(data["source_row_sha256"], "Excel row SHA-256")
    sz_slot = data["sz_slot"]
    if not isinstance(sz_slot, str) or _SZ_SLOT_RE.fullmatch(sz_slot) is None:
        raise ValueError("Excel SZ slot is invalid")
    sz_slot = sz_slot.upper()
    bound_recording = _identifier(
        data["recording_id"], _RECORDING_ID_RE, "Excel recording_id"
    )
    if bound_recording != recording_id:
        raise ValueError("Excel review binding targets another recording")
    event_id = _identifier(data["eeg_event_id"], _EVENT_ID_RE, "Excel eeg_event_id")
    if data["binding_verified"] is not True:
        raise ValueError("unverified Excel mapping cannot enter the context")
    physician_verified = _bool(data["physician_verified"], "physician_verified")
    signed_receipt = data["physician_signed_review_sha256"]
    if physician_verified:
        signed_receipt = _sha256(signed_receipt, "physician signed review SHA-256")
        verification_status = "physician_verified"
    else:
        if signed_receipt is not None:
            raise ValueError("physician receipt requires physician_verified=true")
        verification_status = "source_transcribed_unverified"
    typed_fields = _typed_eeg_fields(data["typed_eeg_fields"], normalize=True)
    uncertain = _bool(data["uncertain"], "Excel uncertainty")
    id_payload: dict[str, object] = {
        "workbook_sha256": workbook_sha,
        "sheet_index": sheet_index,
        "source_row": source_row,
        "source_row_sha256": row_sha,
        "sz_slot": sz_slot,
        "recording_id": recording_id,
        "eeg_event_id": event_id,
        "binding_verified": True,
        "physician_verified": physician_verified,
        "physician_signed_review_sha256": signed_receipt,
        "typed_eeg_fields": typed_fields,
        "uncertain": uncertain,
    }
    return {
        "observation_id": "XLSOBS-"
        + _canonical_sha256(_observation_id_payload(id_payload))[:24],
        **id_payload,
        "verification_status": verification_status,
        "raw_text_included": False,
        "llm_eligible": False,
    }


def build_long_term_clinical_context(
    *,
    recording_id: str,
    patient_id: str,
    source_signal_sha256: str,
    recording_duration_seconds: object,
    edf_annotations_sha256: str,
    annotation_rows: Sequence[Mapping[str, object]],
    workbook_sha256s: Sequence[str] = (),
    excel_review_bindings: Sequence[Mapping[str, object]] = (),
    frozen_events: Sequence[Mapping[str, object]] = (),
    annotation_source_row_bindings: Sequence[Mapping[str, object]] = (),
) -> dict[str, Any]:
    """Build the sidecar without retaining raw annotations or spreadsheet text.

    ``annotation_rows`` accepts only ``source_row``, ``source_row_sha256``,
    ``recording_offset_seconds``, ``description``, and optional ``uncertain``.
    ``excel_review_bindings`` accepts typed EEG fields only.  Frozen events are
    supplied by the upstream detector/SOZ pipeline; annotations can associate
    to those events but can never create one.
    """

    normalized_recording_id = _identifier(
        recording_id, _RECORDING_ID_RE, "recording_id"
    )
    normalized_patient_id = _identifier(patient_id, _PATIENT_ID_RE, "patient_id")
    signal_sha = _sha256(source_signal_sha256, "source signal SHA-256")
    duration = _number(recording_duration_seconds, "recording duration")
    if duration <= 0:
        raise ValueError("recording duration must be positive")
    edf_sha = _sha256(edf_annotations_sha256, "EDF annotations SHA-256")
    workbook_hashes = [_sha256(item, "Excel workbook SHA-256") for item in workbook_sha256s]
    if len(workbook_hashes) != len(set(workbook_hashes)):
        raise ValueError("workbook_sha256s contains duplicates")

    annotations: list[dict[str, Any]] = []
    seen_rows: set[int] = set()
    seen_row_hashes: set[str] = set()
    excluded_rows = 0
    unmapped_rows = 0
    mapped_rows = 0
    for index, raw in enumerate(annotation_rows):
        row = _strict_object(
            raw,
            required={
                "source_row",
                "source_row_sha256",
                "recording_offset_seconds",
                "description",
            },
            optional={"uncertain"},
            context=f"EDF annotation input row {index}",
        )
        source_row = _integer(row["source_row"], "EDF annotation source row", minimum=1)
        row_sha = _sha256(row["source_row_sha256"], "EDF annotation row SHA-256")
        if source_row in seen_rows or row_sha in seen_row_hashes:
            raise ValueError("EDF annotation source row/hash is duplicated")
        seen_rows.add(source_row)
        seen_row_hashes.add(row_sha)
        offset = _recording_offset(
            row["recording_offset_seconds"], duration, "EDF annotation recording offset"
        )
        description = row["description"]
        if not isinstance(description, str):
            raise TypeError("EDF annotation description must be a string")
        input_uncertain = _bool(row.get("uncertain", False), "EDF annotation uncertainty")
        if _description_is_excluded(description):
            excluded_rows += 1
            continue
        codes = classify_edf_annotation_description(description)
        if not codes:
            unmapped_rows += 1
            continue
        mapped_rows += 1
        uncertain = input_uncertain or bool(_UNCERTAINTY_RE.search(description))
        for annotation_type in codes:
            id_payload = _annotation_id_payload(
                recording_id=normalized_recording_id,
                source_file_sha256=edf_sha,
                source_row=source_row,
                source_row_sha256=row_sha,
                recording_offset_seconds=offset,
                annotation_type=annotation_type,
            )
            annotations.append(
                {
                    "annotation_id": "CTXANN-" + _canonical_sha256(id_payload)[:24],
                    "annotation_type": annotation_type,
                    "annotation_scope": _annotation_scope(annotation_type),
                    "recording_offset_seconds": offset,
                    "source_row": source_row,
                    "source_row_sha256": row_sha,
                    "source_file_sha256": edf_sha,
                    "uncertain": uncertain,
                    "verification_status": "source_transcribed_unverified",
                    "llm_eligible": False,
                }
            )
    annotations.sort(
        key=lambda item: (
            item["recording_offset_seconds"],
            item["source_row"],
            ANNOTATION_TYPES.index(item["annotation_type"]),
        )
    )

    excel_observations = [
        _build_excel_observation(
            item,
            recording_id=normalized_recording_id,
            workbook_hashes=set(workbook_hashes),
        )
        for item in excel_review_bindings
    ]
    source_bindings = [
        (
            item["workbook_sha256"],
            item["sheet_index"],
            item["source_row"],
            item["sz_slot"],
        )
        for item in excel_observations
    ]
    if len(source_bindings) != len(set(source_bindings)):
        raise ValueError("Excel review source binding is duplicated")

    context: dict[str, Any] = {
        "schema_version": LONG_TERM_CLINICAL_CONTEXT_SCHEMA,
        "recording_id": normalized_recording_id,
        "patient_id": normalized_patient_id,
        "source_signal_sha256": signal_sha,
        "recording_duration_seconds": duration,
        "source_receipts": {
            "edf_annotations_sha256": edf_sha,
            "workbook_sha256s": sorted(workbook_hashes),
        },
        "annotations": annotations,
        "excel_onset_observations": excel_observations,
        "event_associations": [],
        "unbound_annotation_ids": [item["annotation_id"] for item in annotations],
        "frozen_event_registry_receipt": _event_registry_receipt(
            [],
            recording_id=normalized_recording_id,
            patient_id=normalized_patient_id,
            source_signal_sha256=signal_sha,
        ),
        "association_policy": deepcopy(_ASSOCIATION_POLICY),
        "exclusion_summary": {
            "raw_annotation_rows": len(annotation_rows),
            "mapped_source_rows": mapped_rows,
            "mapped_annotation_items": len(annotations),
            "excluded_non_eeg_or_administrative_rows": excluded_rows,
            "unmapped_rows": unmapped_rows,
            "excel_review_bindings": len(excel_observations),
        },
        "claim_boundary": deepcopy(_CLAIM_BOUNDARY),
    }

    # Source-row bindings are converted only after closed annotation IDs exist.
    explicit_annotation_bindings: list[dict[str, str]] = []
    bound_source_rows: set[int] = set()
    for index, raw in enumerate(annotation_source_row_bindings):
        binding = _strict_object(
            raw,
            required={"source_row", "eeg_event_id"},
            context=f"annotation source-row binding {index}",
        )
        source_row = _integer(binding["source_row"], "bound annotation source row", minimum=1)
        if source_row in bound_source_rows:
            raise ValueError("annotation source row has duplicate explicit bindings")
        bound_source_rows.add(source_row)
        event_id = _identifier(binding["eeg_event_id"], _EVENT_ID_RE, "bound eeg_event_id")
        matching_ids = [
            item["annotation_id"]
            for item in annotations
            if item["source_row"] == source_row
        ]
        if not matching_ids:
            raise ValueError("explicit source-row binding has no mapped annotation")
        explicit_annotation_bindings.extend(
            {"annotation_id": annotation_id, "eeg_event_id": event_id}
            for annotation_id in matching_ids
        )

    context = _associate_context(
        context,
        frozen_events=frozen_events,
        explicit_annotation_bindings=explicit_annotation_bindings,
    )
    return validate_long_term_clinical_context(context)


def associate_long_term_context_events(
    context: object,
    *,
    frozen_events: Sequence[Mapping[str, object]],
    explicit_annotation_bindings: Sequence[Mapping[str, object]] = (),
) -> dict[str, Any]:
    """Re-associate a frozen sidecar to a frozen detector event registry.

    This operation never alters annotations or Excel observations and creates
    exactly one association record per supplied frozen event.  An annotation
    outside a unique ``[-12,+48]``-second window stays on the recording-level
    timeline unless an explicit annotation-ID binding is supplied.
    """

    normalized = validate_long_term_clinical_context(context)
    associated = _associate_context(
        normalized,
        frozen_events=frozen_events,
        explicit_annotation_bindings=explicit_annotation_bindings,
    )
    return validate_long_term_clinical_context(associated)


__all__ = [
    "ANNOTATION_TYPES",
    "CLINICAL_EEG_LONG_TERM_CONTEXT_SCHEMA",
    "EEG_POINT_MARKER_TYPES",
    "EVENT_WINDOW_END_SECONDS",
    "EVENT_WINDOW_START_SECONDS",
    "EXCEL_LATERALITIES",
    "EXCEL_PATTERNS",
    "EXCEL_REGIONS",
    "LONG_TERM_CLINICAL_CONTEXT_SCHEMA",
    "SOURCE_BEHAVIOR_TYPES",
    "associate_long_term_context_events",
    "build_long_term_clinical_context",
    "classify_edf_annotation_description",
    "validate_long_term_clinical_context",
]
