"""Strict clinical EEG fact ledger for ``clinical_eeg_report_v1``.

This module is deliberately independent from the research SOZ pipeline.  It
defines the patient-specific facts that a deterministic renderer or a
fact-constrained language model may consume.  It does not infer diagnoses.

The ledger distinguishes an assessed negative finding (``absent``) from a
finding that was not recorded, could not be assessed, or remains uncertain.
Every assessed fact is bound to provenance and evidence identifiers.  Final
impression facts are accepted only after physician verification.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math
import re
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "clinical_eeg_report_v1"


class FactState(str, Enum):
    """Epistemic state of one patient-specific observation."""

    PRESENT = "present"
    ABSENT = "absent"
    NOT_RECORDED = "not_recorded"
    NOT_ASSESSABLE = "not_assessable"
    UNCERTAIN = "uncertain"


class FactSection(str, Enum):
    METADATA = "metadata"
    BACKGROUND = "background"
    INTERICTAL = "interictal"
    ICTAL = "ictal"
    IMPRESSION = "impression"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    ALGORITHM_CANDIDATE = "algorithm_candidate"
    TECHNOLOGIST_VERIFIED = "technologist_verified"
    PHYSICIAN_VERIFIED = "physician_verified"


class ProvenanceSource(str, Enum):
    ACQUISITION_SYSTEM = "acquisition_system"
    SIGNAL_ALGORITHM = "signal_algorithm"
    TECHNOLOGIST_OBSERVATION = "technologist_observation"
    PHYSICIAN_OBSERVATION = "physician_observation"


# The canonical vocabulary follows current 10-20/10-10 labels.  Legacy
# temporal and auricular names are normalized at ingestion and are never
# emitted by ``to_dict``.
CANONICAL_ELECTRODES = frozenset(
    {
        "FP1", "FPZ", "FP2",
        "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
        "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10",
        "FT9", "FT7", "FT5", "FT3", "FT1", "FT2", "FT4", "FT6", "FT8", "FT10",
        "T9", "T7", "T8", "T10",
        "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
        "TP9", "TP7", "TP5", "TP3", "TP1", "TP2", "TP4", "TP6", "TP8", "TP10",
        "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
        "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10",
        "O1", "OZ", "O2", "IZ", "M1", "M2",
    }
)

ELECTRODE_ALIASES: Mapping[str, str] = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
    "A1": "M1",
    "A2": "M2",
}


FACT_TYPE_TO_SECTION: Mapping[str, FactSection] = {
    # Recording metadata and technique.
    "recording_modality": FactSection.METADATA,
    "recording_duration": FactSection.METADATA,
    "electrode_setup": FactSection.METADATA,
    "acquisition_settings": FactSection.METADATA,
    "recording_quality": FactSection.METADATA,
    # Background.
    "posterior_dominant_rhythm": FactSection.BACKGROUND,
    "background_organization": FactSection.BACKGROUND,
    "background_slowing": FactSection.BACKGROUND,
    "background_asymmetry": FactSection.BACKGROUND,
    "artifact_observation": FactSection.BACKGROUND,
    # Interictal findings.
    "epileptiform_discharge": FactSection.INTERICTAL,
    "periodic_pattern": FactSection.INTERICTAL,
    "rhythmic_delta_activity": FactSection.INTERICTAL,
    "normal_variant_or_uncertain_pattern": FactSection.INTERICTAL,
    # Ictal EEG timeline.
    "electrographic_event_occurrence": FactSection.ICTAL,
    "source_eeg_annotation_timing": FactSection.ICTAL,
    "algorithmic_sustained_eeg_change": FactSection.ICTAL,
    "later_scalp_visible_eeg_change": FactSection.ICTAL,
    "ictal_onset_pattern": FactSection.ICTAL,
    "ictal_evolution": FactSection.ICTAL,
    "ictal_spread": FactSection.ICTAL,
    "ictal_termination": FactSection.ICTAL,
    "postictal_pattern": FactSection.ICTAL,
    # Physician conclusions.
    "study_classification": FactSection.IMPRESSION,
    "interictal_impression": FactSection.IMPRESSION,
    "ictal_eeg_impression": FactSection.IMPRESSION,
    "recording_limitation": FactSection.IMPRESSION,
}

FACT_TYPE_LABEL_ZH: Mapping[str, str] = {
    "recording_modality": "记录类型",
    "recording_duration": "记录时长",
    "electrode_setup": "电极及导联设置",
    "acquisition_settings": "采集参数",
    "recording_quality": "脑电记录质量",
    "posterior_dominant_rhythm": "后部优势节律",
    "background_organization": "背景活动组织",
    "background_slowing": "背景慢波",
    "background_asymmetry": "背景不对称",
    "artifact_observation": "伪迹",
    "epileptiform_discharge": "癫痫样放电",
    "periodic_pattern": "周期性放电模式",
    "rhythmic_delta_activity": "节律性慢波活动",
    "normal_variant_or_uncertain_pattern": "正常变异或意义不确定图形",
    "electrographic_event_occurrence": "脑电事件",
    "source_eeg_annotation_timing": "原始EDF脑电标注时间",
    "algorithmic_sustained_eeg_change": "算法标记的持续脑电波形变化",
    "later_scalp_visible_eeg_change": "后续头皮可见脑电变化（仅时序关系）",
    "ictal_onset_pattern": "发作期脑电起始模式",
    "ictal_evolution": "发作期脑电演变",
    "ictal_spread": "发作期头皮可见空间扩展",
    "ictal_termination": "发作期脑电终止",
    "postictal_pattern": "发作后脑电",
    "study_classification": "检查总体分类",
    "interictal_impression": "发作间期印象",
    "ictal_eeg_impression": "发作期脑电印象",
    "recording_limitation": "记录局限性",
}


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")
_NON_EEG_TEXT_RE = re.compile(
    r"(?:临床|视频|症状|意识|行为|自动症|呼之不应|惯常|心电|心率|肌电|眼电|"
    r"病史|病历|用药|服药|药物|诊断|既往|转诊|影像|头颅|患者|受检者|"
    r"睡眠|困倦|诱发试验|过度换气|闪光刺激|睁闭眼|"
    r"(?<![A-Za-z0-9])(?:video|clinical|semiology|awareness|responsiveness|"
    r"habitual|ECG|EKG|EMG|EOG|MRI|CT|sleep|drowsy|drowsiness|N[123]|REM|"
    r"hyperventilation|photic)(?![A-Za-z0-9]))",
    re.IGNORECASE,
)

_NON_EEG_ID_RE = re.compile(
    r"(?:VIDEO|ECG|EKG|EMG|EOG|CLIN(?:ICAL)?|SEMI(?:OLOGY)?)",
    re.IGNORECASE,
)

_NEUTRAL_TEMPORAL_PROMOTION_RE = re.compile(
    r"(?:发作(?:期(?:脑电)?)?起始|脑电(?:发作)?起始|"
    r"(?:发作期|脑电)(?:变化)?起点|临床(?:确认的?)?起始|"
    r"(?:皮层|癫痫灶|SOZ)(?:起始|起点|起源)|最早(?:头皮可见|电极|导联)|"
    r"传播|扩散|蔓延|空间扩展|"
    r"(?:^|[^A-Za-z0-9])(?:onset|origin|propagation|spread)"
    r"(?:$|[^A-Za-z0-9]))",
    re.IGNORECASE,
)


def canonicalize_electrode(label: str) -> str:
    """Return a current canonical label for one scalp/reference electrode."""

    if not isinstance(label, str) or not label.strip():
        raise TypeError("electrode label must be a non-empty string")
    normalized = re.sub(r"\s+", "", label).upper()
    if normalized.startswith("EEG"):
        normalized = normalized[3:]
    for suffix in ("-REF", "_REF", "-LE", "_LE"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    normalized = ELECTRODE_ALIASES.get(normalized, normalized)
    if normalized not in CANONICAL_ELECTRODES:
        raise ValueError(f"unsupported electrode label: {label!r}")
    return normalized


def canonicalize_derivation(label: str) -> str:
    """Canonicalize a two-electrode bipolar derivation."""

    if not isinstance(label, str) or not label.strip():
        raise TypeError("derivation label must be a non-empty string")
    normalized = _DASH_RE.sub("-", label.strip())
    parts = [item.strip() for item in normalized.split("-")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"derivation must contain exactly two electrodes: {label!r}")
    first, second = (canonicalize_electrode(item) for item in parts)
    if first == second:
        raise ValueError("derivation endpoints must be distinct")
    return f"{first}-{second}"


def _strict_object(
    value: object,
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    keys = set(value)
    required_set = set(required)
    allowed = required_set.union(optional)
    missing = required_set.difference(keys)
    extra = keys.difference(allowed)
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{context} has unknown keys: {sorted(extra)}")
    return deepcopy(value)


def _nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} must be a non-empty string")
    return value.strip()


def _eeg_only_text(value: object, context: str) -> str:
    text = _nonempty_string(value, context)
    if len(text) > 2000:
        raise ValueError(f"{context} must be at most 2000 characters")
    if any(ord(character) < 32 and character not in "\t\n\r" for character in text):
        raise ValueError(f"{context} contains control characters")
    if _NON_EEG_TEXT_RE.search(text):
        raise ValueError(f"{context} contains a non-EEG assertion")
    return text


def _neutral_temporal_eeg_text(value: object, context: str) -> str:
    """Validate wording that may describe timing but never onset or spread."""

    text = _eeg_only_text(value, context)
    if _NEUTRAL_TEMPORAL_PROMOTION_RE.search(text):
        raise ValueError(
            f"{context} upgrades a neutral temporal EEG observation to onset or propagation"
        )
    return text


def _identifier(value: object, context: str, *, source: bool = False) -> str:
    text = _nonempty_string(value, context)
    regex = _SOURCE_ID_RE if source else _ID_RE
    if regex.fullmatch(text) is None:
        raise ValueError(f"{context} has an invalid identifier: {text!r}")
    return text


def _eeg_source_identifier(value: object, context: str) -> str:
    text = _identifier(value, context, source=True)
    if _NON_EEG_ID_RE.search(text):
        raise ValueError(f"{context} identifies a non-EEG source")
    return text


def _enum(value: object, allowed: Sequence[str], context: str) -> str:
    text = _nonempty_string(value, context)
    if text not in allowed:
        raise ValueError(f"{context} must be one of {tuple(allowed)}")
    return text


def _number(
    value: object,
    context: str,
    *,
    minimum: float | None = None,
    exclusive_minimum: float | None = None,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and float(value) < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    if exclusive_minimum is not None and float(value) <= exclusive_minimum:
        raise ValueError(f"{context} must be > {exclusive_minimum}")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return value


def _string_list(
    value: object,
    context: str,
    *,
    allowed: Sequence[str] | None = None,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise TypeError(f"{context} must be {qualifier}")
    result: list[str] = []
    for index, raw in enumerate(value):
        item = _nonempty_string(raw, f"{context}[{index}]")
        if allowed is not None and item not in allowed:
            raise ValueError(f"{context}[{index}] must be one of {tuple(allowed)}")
        if item in result:
            raise ValueError(f"{context} contains duplicate value: {item!r}")
        result.append(item)
    return result


def _id_list(
    value: object,
    context: str,
    *,
    allow_empty: bool = False,
    source: bool = False,
) -> list[str]:
    raw = _string_list(value, context, allow_empty=allow_empty)
    result = [_identifier(item, context, source=source) for item in raw]
    if len(result) != len(set(result)):
        raise ValueError(f"{context} contains duplicate identifiers")
    return result


def _electrode_list(value: object, context: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise TypeError(f"{context} must be {qualifier}")
    result = [canonicalize_electrode(item) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{context} contains duplicate canonical electrodes")
    return result


def _derivation_list(value: object, context: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise TypeError(f"{context} must be {qualifier}")
    result = [canonicalize_derivation(item) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{context} contains duplicate canonical derivations")
    return result


def _numeric_range(
    value: object,
    context: str,
    *,
    minimum: float = 0.0,
    exclusive_minimum: bool = False,
) -> dict[str, int | float]:
    data = _strict_object(value, required=("min", "max"), context=context)
    kwargs = {"exclusive_minimum": minimum} if exclusive_minimum else {"minimum": minimum}
    low = _number(data["min"], f"{context}.min", **kwargs)
    high = _number(data["max"], f"{context}.max", **kwargs)
    if float(low) > float(high):
        raise ValueError(f"{context}.min must be <= max")
    return {"min": low, "max": high}


def _timestamp(value: object, context: str) -> str:
    text = _nonempty_string(value, context)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{context} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{context} must include a timezone")
    return text


_LATERALITY = ("left", "right", "bilateral", "midline", "none", "indeterminate")
_DISTRIBUTION = (
    "focal", "multifocal", "hemispheric", "bilateral_independent",
    "bilateral_synchronous", "generalized", "diffuse",
)
_REGIONS = (
    "frontal", "temporal", "central", "parietal", "occipital",
    "frontotemporal", "centrotemporal", "temporoparietal", "posterior",
    "diffuse", "midline", "unknown",
)
_FREQUENCY_BANDS = ("delta", "theta", "alpha", "beta", "gamma", "broadband", "unknown")
_RHYTHMICITY = ("rhythmic", "quasi_rhythmic", "nonrhythmic", "indeterminate")
_SYMMETRY = ("symmetric", "asymmetric", "indeterminate")
_SIGNAL_QUALIFICATION_KEYS = (
    "producer_id",
    "policy_sha256",
    "artifact_gate_passed",
    "sustained_change_gate_passed",
    "reproducibility_gate_passed",
    "source_signal_only",
    "external_context_used",
    "research_ranking_used",
    "morphology_terms_qualified",
    "spatial_spread_terms_qualified",
)


def _simple_enum_value(value: object, *, key: str, allowed: Sequence[str], context: str) -> dict[str, Any]:
    data = _strict_object(value, required=(key,), context=context)
    data[key] = _enum(data[key], allowed, f"{context}.{key}")
    return data


def _validate_recording_duration(value: object) -> dict[str, Any]:
    data = _strict_object(value, required=("duration_seconds",), context="recording_duration.value")
    data["duration_seconds"] = _number(data["duration_seconds"], "duration_seconds", exclusive_minimum=0)
    return data


def _validate_electrode_setup(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("system", "electrodes", "montages"),
        optional=("reference",),
        context="electrode_setup.value",
    )
    data["system"] = _enum(
        data["system"],
        ("international_10_20", "international_10_10", "custom"),
        "electrode_setup.system",
    )
    data["electrodes"] = _electrode_list(data["electrodes"], "electrode_setup.electrodes")
    data["montages"] = _string_list(
        data["montages"],
        "electrode_setup.montages",
        allowed=(
            "longitudinal_bipolar",
            "transverse_bipolar",
            "common_average",
            "linked_mastoids",
            "ipsilateral_mastoid",
            "referential",
            "laplacian",
            "custom_scalp_eeg",
        ),
    )
    if "reference" in data:
        reference = _nonempty_string(data["reference"], "electrode_setup.reference")
        if reference not in {
            "average",
            "linked_mastoids",
            "ipsilateral_mastoid",
            "source_reference",
            "unknown",
        }:
            reference = canonicalize_electrode(reference)
        data["reference"] = reference
    return data


def _validate_acquisition_settings(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("sampling_rate_hz", "low_cut_hz", "high_cut_hz"),
        optional=("notch_hz",),
        context="acquisition_settings.value",
    )
    data["sampling_rate_hz"] = _number(data["sampling_rate_hz"], "sampling_rate_hz", exclusive_minimum=0)
    data["low_cut_hz"] = _number(data["low_cut_hz"], "low_cut_hz", minimum=0)
    data["high_cut_hz"] = _number(data["high_cut_hz"], "high_cut_hz", exclusive_minimum=0)
    if float(data["low_cut_hz"]) >= float(data["high_cut_hz"]):
        raise ValueError("low_cut_hz must be below high_cut_hz")
    if float(data["high_cut_hz"]) > float(data["sampling_rate_hz"]) / 2:
        raise ValueError("high_cut_hz must not exceed the Nyquist frequency")
    if "notch_hz" in data:
        data["notch_hz"] = _number(data["notch_hz"], "notch_hz", exclusive_minimum=0)
    return data


def _validate_recording_quality(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("quality",),
        optional=("interpretable_fraction", "affected_assessments"),
        context="recording_quality.value",
    )
    data["quality"] = _enum(
        data["quality"],
        ("good", "limited", "unusable"),
        "recording_quality.quality",
    )
    if "interpretable_fraction" in data:
        fraction = _number(
            data["interpretable_fraction"],
            "recording_quality.interpretable_fraction",
            minimum=0,
        )
        if float(fraction) > 1:
            raise ValueError("recording_quality.interpretable_fraction must be <= 1")
        data["interpretable_fraction"] = fraction
    if "affected_assessments" in data:
        data["affected_assessments"] = _string_list(
            data["affected_assessments"],
            "recording_quality.affected_assessments",
            allowed=("background", "interictal", "ictal"),
        )
    return data


def _validate_pdr(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("frequency_hz", "symmetry"),
        optional=("amplitude_uv", "maximal_electrodes"),
        context="posterior_dominant_rhythm.value",
    )
    data["frequency_hz"] = _numeric_range(data["frequency_hz"], "pdr.frequency_hz", exclusive_minimum=True)
    data["symmetry"] = _enum(data["symmetry"], _SYMMETRY, "pdr.symmetry")
    if "amplitude_uv" in data:
        data["amplitude_uv"] = _numeric_range(data["amplitude_uv"], "pdr.amplitude_uv")
    if "maximal_electrodes" in data:
        data["maximal_electrodes"] = _electrode_list(data["maximal_electrodes"], "pdr.maximal_electrodes")
    return data


def _validate_background_organization(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("organization", "continuity", "voltage"),
        context="background_organization.value",
    )
    data["organization"] = _enum(data["organization"], ("well_organized", "fairly_organized", "poorly_organized", "disorganized"), "background.organization")
    data["continuity"] = _enum(data["continuity"], ("continuous", "discontinuous", "burst_suppression", "suppressed"), "background.continuity")
    data["voltage"] = _enum(data["voltage"], ("normal", "low", "high", "mixed", "indeterminate"), "background.voltage")
    return data


def _validate_background_slowing(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("distribution", "laterality", "regions", "electrodes", "frequency_band", "rhythmicity"),
        optional=("frequency_hz", "amplitude_uv", "maximal_electrodes"),
        context="background_slowing.value",
    )
    data["distribution"] = _enum(data["distribution"], _DISTRIBUTION, "background_slowing.distribution")
    data["laterality"] = _enum(data["laterality"], _LATERALITY, "background_slowing.laterality")
    data["regions"] = _string_list(data["regions"], "background_slowing.regions", allowed=_REGIONS)
    data["electrodes"] = _electrode_list(data["electrodes"], "background_slowing.electrodes", allow_empty=True)
    data["frequency_band"] = _enum(data["frequency_band"], _FREQUENCY_BANDS, "background_slowing.frequency_band")
    data["rhythmicity"] = _enum(data["rhythmicity"], _RHYTHMICITY, "background_slowing.rhythmicity")
    if "frequency_hz" in data:
        data["frequency_hz"] = _numeric_range(data["frequency_hz"], "background_slowing.frequency_hz", exclusive_minimum=True)
    if "amplitude_uv" in data:
        data["amplitude_uv"] = _numeric_range(data["amplitude_uv"], "background_slowing.amplitude_uv")
    if "maximal_electrodes" in data:
        data["maximal_electrodes"] = _electrode_list(data["maximal_electrodes"], "background_slowing.maximal_electrodes")
    return data


def _validate_background_asymmetry(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("lower_amplitude_side", "affected_electrodes", "descriptor"),
        optional=("amplitude_difference_percent",),
        context="background_asymmetry.value",
    )
    data["lower_amplitude_side"] = _enum(data["lower_amplitude_side"], ("left", "right", "indeterminate"), "background_asymmetry.lower_amplitude_side")
    data["affected_electrodes"] = _electrode_list(data["affected_electrodes"], "background_asymmetry.affected_electrodes")
    data["descriptor"] = _enum(data["descriptor"], ("mild", "moderate", "marked", "indeterminate"), "background_asymmetry.descriptor")
    if "amplitude_difference_percent" in data:
        number = _number(data["amplitude_difference_percent"], "background_asymmetry.amplitude_difference_percent", minimum=0)
        if float(number) > 100:
            raise ValueError("amplitude_difference_percent must be <= 100")
        data["amplitude_difference_percent"] = number
    return data


def _validate_artifact(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("artifact_types", "burden", "affected_electrodes"),
        context="artifact_observation.value",
    )
    data["artifact_types"] = _string_list(
        data["artifact_types"],
        "artifact.artifact_types",
        allowed=("eye_movement", "blink", "muscle", "movement", "electrode", "sweat", "pulse", "line_noise", "other"),
    )
    data["burden"] = _enum(data["burden"], ("minimal", "mild", "moderate", "severe", "intermittent", "continuous"), "artifact.burden")
    data["affected_electrodes"] = _electrode_list(data["affected_electrodes"], "artifact.affected_electrodes", allow_empty=True)
    return data


def _validate_prevalence(value: object, context: str) -> dict[str, Any]:
    data = _strict_object(value, required=("descriptor",), optional=("count", "rate_per_hour"), context=context)
    data["descriptor"] = _enum(data["descriptor"], ("single", "rare", "occasional", "frequent", "abundant", "continuous", "clustered"), f"{context}.descriptor")
    if "count" in data:
        data["count"] = _integer(data["count"], f"{context}.count", minimum=1)
    if "rate_per_hour" in data:
        data["rate_per_hour"] = _number(data["rate_per_hour"], f"{context}.rate_per_hour", exclusive_minimum=0)
    return data


def _validate_interictal(value: object, *, family: str) -> dict[str, Any]:
    morphology_allowed = (
        "spike", "sharp_wave", "spike_and_slow_wave", "sharp_and_slow_wave",
        "polyspike", "polyspike_and_slow_wave", "periodic_discharge",
        "rhythmic_delta", "other",
    )
    data = _strict_object(
        value,
        required=("morphology", "distribution", "laterality", "regions", "electrodes", "maximal_electrodes", "prevalence", "rhythmicity"),
        optional=("amplitude_uv", "frequency_hz", "spread_to_electrodes", "derivations"),
        context=f"{family}.value",
    )
    data["morphology"] = _enum(data["morphology"], morphology_allowed, f"{family}.morphology")
    data["distribution"] = _enum(data["distribution"], _DISTRIBUTION, f"{family}.distribution")
    data["laterality"] = _enum(data["laterality"], _LATERALITY, f"{family}.laterality")
    data["regions"] = _string_list(data["regions"], f"{family}.regions", allowed=_REGIONS)
    data["electrodes"] = _electrode_list(data["electrodes"], f"{family}.electrodes", allow_empty=True)
    data["maximal_electrodes"] = _electrode_list(data["maximal_electrodes"], f"{family}.maximal_electrodes", allow_empty=True)
    data["prevalence"] = _validate_prevalence(data["prevalence"], f"{family}.prevalence")
    data["rhythmicity"] = _enum(data["rhythmicity"], _RHYTHMICITY, f"{family}.rhythmicity")
    if not data["regions"] and not data["electrodes"]:
        raise ValueError(f"{family} must identify at least one region or electrode")
    if "amplitude_uv" in data:
        data["amplitude_uv"] = _numeric_range(data["amplitude_uv"], f"{family}.amplitude_uv")
    if "frequency_hz" in data:
        data["frequency_hz"] = _numeric_range(data["frequency_hz"], f"{family}.frequency_hz", exclusive_minimum=True)
    if "spread_to_electrodes" in data:
        data["spread_to_electrodes"] = _electrode_list(data["spread_to_electrodes"], f"{family}.spread_to_electrodes", allow_empty=True)
    if "derivations" in data:
        data["derivations"] = _derivation_list(data["derivations"], f"{family}.derivations", allow_empty=True)
    return data


def _validate_electrographic_event_occurrence(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("event_number", "start_offset_seconds", "duration_seconds", "event_class"),
        optional=("time_coordinate",),
        context="electrographic_event_occurrence.value",
    )
    data["event_number"] = _integer(
        data["event_number"], "electrographic_event_occurrence.event_number", minimum=1
    )
    data["start_offset_seconds"] = _number(
        data["start_offset_seconds"],
        "electrographic_event_occurrence.start_offset_seconds",
        minimum=0,
    )
    data["duration_seconds"] = _number(
        data["duration_seconds"],
        "electrographic_event_occurrence.duration_seconds",
        exclusive_minimum=0,
    )
    data["event_class"] = _enum(
        data["event_class"],
        (
            "electrographic_seizure",
            "electrographic_event",
            "uncertain_electrographic_pattern",
        ),
        "electrographic_event_occurrence.event_class",
    )
    # Historical full-record ledgers used recording-start coordinates and did
    # not carry this field.  Event-window adapters must state their coordinate
    # explicitly so a 60-second clip offset can never be rendered as a
    # recording-level timestamp.
    if "time_coordinate" in data:
        data["time_coordinate"] = _enum(
            data["time_coordinate"],
            ("recording_start_seconds", "segment_start_seconds"),
            "electrographic_event_occurrence.time_coordinate",
        )
    return data


def _validate_source_eeg_annotation_timing(value: object) -> dict[str, Any]:
    """Validate point annotations without promoting them to event boundaries.

    Private EDF annotations are heterogeneous point markers.  Their time field
    is an annotation coordinate, not evidence by itself of electrographic
    onset, termination, or duration.  This fact therefore carries only a
    controlled marker kind and its original-recording coordinate, together
    with explicit non-promotion constants.
    """

    data = _strict_object(
        value,
        required=(
            "time_coordinate",
            "markers",
            "point_markers_only",
            "onset_confirmed",
            "termination_confirmed",
            "duration_derived",
        ),
        context="source_eeg_annotation_timing.value",
    )
    data["time_coordinate"] = _enum(
        data["time_coordinate"],
        ("original_recording_start_seconds",),
        "source_eeg_annotation_timing.time_coordinate",
    )
    raw_markers = data["markers"]
    if not isinstance(raw_markers, list) or not raw_markers:
        raise TypeError(
            "source_eeg_annotation_timing.markers must be a non-empty list"
        )
    markers: list[dict[str, Any]] = []
    seen_kinds: set[str] = set()
    for index, raw in enumerate(raw_markers):
        marker = _strict_object(
            raw,
            required=("marker_kind", "recording_offset_seconds", "point_marker"),
            context=f"source_eeg_annotation_timing.markers[{index}]",
        )
        kind = _enum(
            marker["marker_kind"],
            ("event_marker", "eeg_event_marker", "end_marker"),
            f"source_eeg_annotation_timing.markers[{index}].marker_kind",
        )
        if kind in seen_kinds:
            raise ValueError(
                "source_eeg_annotation_timing contains duplicate marker kinds"
            )
        seen_kinds.add(kind)
        if marker["point_marker"] is not True:
            raise ValueError(
                "source_eeg_annotation_timing markers must remain point markers"
            )
        markers.append(
            {
                "marker_kind": kind,
                "recording_offset_seconds": _number(
                    marker["recording_offset_seconds"],
                    (
                        "source_eeg_annotation_timing."
                        f"markers[{index}].recording_offset_seconds"
                    ),
                    minimum=0,
                ),
                "point_marker": True,
            }
        )
    offsets = {
        marker["marker_kind"]: float(marker["recording_offset_seconds"])
        for marker in markers
    }
    if (
        "eeg_event_marker" in offsets
        and "end_marker" in offsets
        and offsets["end_marker"] <= offsets["eeg_event_marker"]
    ):
        raise ValueError(
            "source EEG end marker must follow its EEG event marker"
        )
    for key in (
        "point_markers_only",
        "onset_confirmed",
        "termination_confirmed",
        "duration_derived",
    ):
        expected = key == "point_markers_only"
        if data[key] is not expected:
            raise ValueError(
                f"source_eeg_annotation_timing.{key} must be {expected!r}"
            )
        data[key] = expected
    data["markers"] = markers
    return data


def _validate_signal_finding_qualification(
    value: object,
    *,
    context: str,
) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=_SIGNAL_QUALIFICATION_KEYS,
        context=context,
    )
    data["producer_id"] = _nonempty_string(
        data["producer_id"], f"{context}.producer_id"
    )
    policy_sha256 = _nonempty_string(
        data["policy_sha256"], f"{context}.policy_sha256"
    )
    if re.fullmatch(r"[0-9a-f]{64}", policy_sha256) is None:
        raise ValueError(f"{context}.policy_sha256 must be lowercase SHA-256")
    data["policy_sha256"] = policy_sha256
    for key in _SIGNAL_QUALIFICATION_KEYS[2:]:
        if not isinstance(data[key], bool):
            raise TypeError(f"{context}.{key} must be boolean")
    for key in (
        "artifact_gate_passed",
        "sustained_change_gate_passed",
        "reproducibility_gate_passed",
        "source_signal_only",
    ):
        if data[key] is not True:
            raise ValueError(f"{context}.{key} must be true")
    for key in ("external_context_used", "research_ranking_used"):
        if data[key] is not False:
            raise ValueError(f"{context}.{key} must be false")
    return data


def _validate_algorithmic_sustained_eeg_change(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("start_offset_seconds", "end_offset_seconds", "derivations"),
        optional=(
            "electrodes",
            "maximal_electrodes",
            "laterality",
            "distribution",
            "regions",
            "frequency_hz",
            "frequency_band",
            "amplitude_uv",
            "rhythmicity",
            "quantitative_trajectory",
            "later_derivation_changes",
            "candidate_return_to_baseline_offset_seconds",
            "qualification",
        ),
        context="algorithmic_sustained_eeg_change.value",
    )
    data["start_offset_seconds"] = _number(
        data["start_offset_seconds"],
        "algorithmic_sustained_eeg_change.start_offset_seconds",
        minimum=0,
    )
    data["end_offset_seconds"] = _number(
        data["end_offset_seconds"],
        "algorithmic_sustained_eeg_change.end_offset_seconds",
        minimum=0,
    )
    if float(data["end_offset_seconds"]) <= float(data["start_offset_seconds"]):
        raise ValueError(
            "algorithmic_sustained_eeg_change.end_offset_seconds must be greater "
            "than start_offset_seconds"
        )
    data["derivations"] = _derivation_list(
        data["derivations"],
        "algorithmic_sustained_eeg_change.derivations",
    )
    qualified_descriptors = {
        "electrodes",
        "maximal_electrodes",
        "laterality",
        "distribution",
        "regions",
        "frequency_hz",
        "frequency_band",
        "amplitude_uv",
        "rhythmicity",
        "quantitative_trajectory",
        "later_derivation_changes",
        "candidate_return_to_baseline_offset_seconds",
    }
    if qualified_descriptors.intersection(data) and "qualification" not in data:
        raise ValueError(
            "algorithmic_sustained_eeg_change quantitative descriptors require "
            "qualification"
        )
    if "electrodes" in data:
        data["electrodes"] = _electrode_list(
            data["electrodes"],
            "algorithmic_sustained_eeg_change.electrodes",
        )
    if "maximal_electrodes" in data:
        data["maximal_electrodes"] = _electrode_list(
            data["maximal_electrodes"],
            "algorithmic_sustained_eeg_change.maximal_electrodes",
        )
        if "electrodes" not in data or not set(data["maximal_electrodes"]).issubset(
            data["electrodes"]
        ):
            raise ValueError(
                "algorithmic_sustained_eeg_change maximal_electrodes must be a "
                "subset of electrodes"
            )
    if "laterality" in data:
        data["laterality"] = _enum(
            data["laterality"],
            _LATERALITY,
            "algorithmic_sustained_eeg_change.laterality",
        )
    if "distribution" in data:
        data["distribution"] = _enum(
            data["distribution"],
            _DISTRIBUTION,
            "algorithmic_sustained_eeg_change.distribution",
        )
    if "regions" in data:
        data["regions"] = _string_list(
            data["regions"],
            "algorithmic_sustained_eeg_change.regions",
            allowed=_REGIONS,
        )
    if "frequency_hz" in data:
        data["frequency_hz"] = _numeric_range(
            data["frequency_hz"],
            "algorithmic_sustained_eeg_change.frequency_hz",
            exclusive_minimum=True,
        )
    if "frequency_band" in data:
        data["frequency_band"] = _enum(
            data["frequency_band"],
            _FREQUENCY_BANDS,
            "algorithmic_sustained_eeg_change.frequency_band",
        )
    if "amplitude_uv" in data:
        data["amplitude_uv"] = _numeric_range(
            data["amplitude_uv"],
            "algorithmic_sustained_eeg_change.amplitude_uv",
        )
    if "rhythmicity" in data:
        data["rhythmicity"] = _enum(
            data["rhythmicity"],
            _RHYTHMICITY,
            "algorithmic_sustained_eeg_change.rhythmicity",
        )
    if "quantitative_trajectory" in data:
        trajectory = _strict_object(
            data["quantitative_trajectory"],
            required=(
                "comparison_offset_seconds",
                "change_dimensions",
                "early_frequency_hz",
                "late_frequency_hz",
                "early_amplitude_uv",
                "late_amplitude_uv",
                "amplitude_change_alone_is_not_ictal_evolution",
            ),
            context="algorithmic_sustained_eeg_change.quantitative_trajectory",
        )
        trajectory["comparison_offset_seconds"] = _number(
            trajectory["comparison_offset_seconds"],
            "algorithmic_sustained_eeg_change.trajectory.comparison_offset_seconds",
            exclusive_minimum=0,
        )
        trajectory["change_dimensions"] = _string_list(
            trajectory["change_dimensions"],
            "algorithmic_sustained_eeg_change.trajectory.change_dimensions",
            allowed=("frequency", "amplitude"),
        )
        for key in ("early_frequency_hz", "late_frequency_hz"):
            trajectory[key] = _number(
                trajectory[key],
                f"algorithmic_sustained_eeg_change.trajectory.{key}",
                exclusive_minimum=0,
            )
        for key in ("early_amplitude_uv", "late_amplitude_uv"):
            trajectory[key] = _number(
                trajectory[key],
                f"algorithmic_sustained_eeg_change.trajectory.{key}",
                minimum=0,
            )
        if trajectory["amplitude_change_alone_is_not_ictal_evolution"] is not True:
            raise ValueError(
                "algorithmic sustained-change trajectory must retain the "
                "non-ictal-evolution boundary"
            )
        data["quantitative_trajectory"] = trajectory
    if "later_derivation_changes" in data:
        observations = data["later_derivation_changes"]
        if not isinstance(observations, list) or not observations:
            raise TypeError(
                "algorithmic_sustained_eeg_change.later_derivation_changes "
                "must be a non-empty list"
            )
        normalized_observations: list[dict[str, Any]] = []
        seen_derivations: set[str] = set()
        for index, raw in enumerate(observations):
            observation = _strict_object(
                raw,
                required=("derivation", "delay_seconds"),
                context=(
                    "algorithmic_sustained_eeg_change."
                    f"later_derivation_changes[{index}]"
                ),
            )
            derivation = canonicalize_derivation(observation["derivation"])
            if derivation in seen_derivations:
                raise ValueError(
                    "later_derivation_changes contains a duplicate derivation"
                )
            seen_derivations.add(derivation)
            normalized_observations.append(
                {
                    "derivation": derivation,
                    "delay_seconds": _number(
                        observation["delay_seconds"],
                        (
                            "algorithmic_sustained_eeg_change."
                            f"later_derivation_changes[{index}].delay_seconds"
                        ),
                        exclusive_minimum=0,
                    ),
                }
            )
        data["later_derivation_changes"] = normalized_observations
    if "candidate_return_to_baseline_offset_seconds" in data:
        data["candidate_return_to_baseline_offset_seconds"] = _number(
            data["candidate_return_to_baseline_offset_seconds"],
            (
                "algorithmic_sustained_eeg_change."
                "candidate_return_to_baseline_offset_seconds"
            ),
            exclusive_minimum=0,
        )
    if "qualification" in data:
        data["qualification"] = _validate_signal_finding_qualification(
            data["qualification"],
            context="algorithmic_sustained_eeg_change.qualification",
        )
    return data


def _validate_later_scalp_visible_eeg_change(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("observations", "temporal_relation_only"),
        optional=("qualification",),
        context="later_scalp_visible_eeg_change.value",
    )
    observations = data["observations"]
    if not isinstance(observations, list) or not observations:
        raise TypeError(
            "later_scalp_visible_eeg_change.observations must be a non-empty list"
        )
    normalized_observations: list[dict[str, Any]] = []
    seen_electrodes: set[str] = set()
    for index, raw in enumerate(observations):
        observation = _strict_object(
            raw,
            required=("electrode", "delay_seconds"),
            context=f"later_scalp_visible_eeg_change.observations[{index}]",
        )
        electrode = canonicalize_electrode(observation["electrode"])
        if electrode in seen_electrodes:
            raise ValueError(
                "later_scalp_visible_eeg_change.observations contains duplicate "
                f"electrode: {electrode!r}"
            )
        seen_electrodes.add(electrode)
        normalized_observations.append(
            {
                "electrode": electrode,
                "delay_seconds": _number(
                    observation["delay_seconds"],
                    (
                        "later_scalp_visible_eeg_change."
                        f"observations[{index}].delay_seconds"
                    ),
                    exclusive_minimum=0,
                ),
            }
        )
    if data["temporal_relation_only"] is not True:
        raise ValueError(
            "later_scalp_visible_eeg_change.temporal_relation_only must be true"
        )
    data["observations"] = normalized_observations
    data["temporal_relation_only"] = True
    if "qualification" in data:
        data["qualification"] = _validate_signal_finding_qualification(
            data["qualification"],
            context="later_scalp_visible_eeg_change.qualification",
        )
    return data


def _validate_ictal_onset(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("onset_offset_seconds", "onset_type", "laterality", "distribution", "regions", "electrodes", "maximal_electrodes", "morphology", "rhythmicity"),
        optional=("frequency_hz", "amplitude_uv", "derivations"),
        context="ictal_onset_pattern.value",
    )
    data["onset_offset_seconds"] = _number(data["onset_offset_seconds"], "ictal_onset.onset_offset_seconds", minimum=0)
    data["onset_type"] = _enum(data["onset_type"], ("low_voltage_fast_activity", "rhythmic_activity", "repetitive_spikes", "electrodecrement", "attenuation", "irregular_activity", "other"), "ictal_onset.onset_type")
    data["laterality"] = _enum(data["laterality"], _LATERALITY, "ictal_onset.laterality")
    data["distribution"] = _enum(data["distribution"], _DISTRIBUTION, "ictal_onset.distribution")
    data["regions"] = _string_list(data["regions"], "ictal_onset.regions", allowed=_REGIONS)
    data["electrodes"] = _electrode_list(data["electrodes"], "ictal_onset.electrodes", allow_empty=True)
    data["maximal_electrodes"] = _electrode_list(data["maximal_electrodes"], "ictal_onset.maximal_electrodes", allow_empty=True)
    data["morphology"] = _enum(data["morphology"], ("spike", "sharp_wave", "spike_and_slow_wave", "sharp_and_slow_wave", "polyspike", "fast_activity", "theta_activity", "delta_activity", "attenuation", "mixed", "other"), "ictal_onset.morphology")
    data["rhythmicity"] = _enum(data["rhythmicity"], _RHYTHMICITY, "ictal_onset.rhythmicity")
    if not data["regions"] and not data["electrodes"]:
        raise ValueError("ictal_onset must identify at least one region or electrode")
    if "frequency_hz" in data:
        data["frequency_hz"] = _numeric_range(data["frequency_hz"], "ictal_onset.frequency_hz", exclusive_minimum=True)
    if "amplitude_uv" in data:
        data["amplitude_uv"] = _numeric_range(data["amplitude_uv"], "ictal_onset.amplitude_uv")
    if "derivations" in data:
        data["derivations"] = _derivation_list(data["derivations"], "ictal_onset.derivations", allow_empty=True)
    return data


def _validate_ictal_evolution(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("sequence_index", "onset_offset_seconds", "change_dimensions"),
        optional=(
            "frequency_hz",
            "amplitude_uv",
            "morphology",
            "electrodes",
            "regions",
            "laterality",
            "qualification",
        ),
        context="ictal_evolution.value",
    )
    data["sequence_index"] = _integer(data["sequence_index"], "ictal_evolution.sequence_index", minimum=1)
    data["onset_offset_seconds"] = _number(data["onset_offset_seconds"], "ictal_evolution.onset_offset_seconds", minimum=0)
    dimensions = _string_list(data["change_dimensions"], "ictal_evolution.change_dimensions", allowed=("frequency", "amplitude", "morphology", "spatial_distribution"))
    data["change_dimensions"] = dimensions
    expected_key = {"frequency": "frequency_hz", "amplitude": "amplitude_uv", "morphology": "morphology", "spatial_distribution": "electrodes"}
    for dimension in dimensions:
        if expected_key[dimension] not in data:
            raise ValueError(f"ictal_evolution {dimension} requires {expected_key[dimension]}")
    if "frequency_hz" in data:
        data["frequency_hz"] = _numeric_range(data["frequency_hz"], "ictal_evolution.frequency_hz", exclusive_minimum=True)
    if "amplitude_uv" in data:
        data["amplitude_uv"] = _numeric_range(data["amplitude_uv"], "ictal_evolution.amplitude_uv")
    if "morphology" in data:
        data["morphology"] = _enum(
            data["morphology"],
            (
                "spike",
                "sharp_wave",
                "spike_and_slow_wave",
                "sharp_and_slow_wave",
                "polyspike",
                "fast_activity",
                "theta_activity",
                "delta_activity",
                "attenuation",
                "mixed",
                "other",
            ),
            "ictal_evolution.morphology",
        )
    if "electrodes" in data:
        data["electrodes"] = _electrode_list(data["electrodes"], "ictal_evolution.electrodes")
    if "regions" in data:
        data["regions"] = _string_list(data["regions"], "ictal_evolution.regions", allowed=_REGIONS)
    if "laterality" in data:
        data["laterality"] = _enum(data["laterality"], _LATERALITY, "ictal_evolution.laterality")
    if "qualification" in data:
        data["qualification"] = _validate_signal_finding_qualification(
            data["qualification"],
            context="ictal_evolution.qualification",
        )
        if (
            "morphology" in dimensions
            and data["qualification"]["morphology_terms_qualified"] is not True
        ):
            raise ValueError(
                "ictal_evolution morphology requires a morphology-qualified producer"
            )
        if (
            "spatial_distribution" in dimensions
            and data["qualification"]["spatial_spread_terms_qualified"] is not True
        ):
            raise ValueError(
                "ictal_evolution spatial distribution requires a spatial-qualified producer"
            )
    return data


def _validate_ictal_spread(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("onset_offset_seconds", "from_electrodes", "to_electrodes", "to_regions", "laterality"),
        context="ictal_spread.value",
    )
    data["onset_offset_seconds"] = _number(data["onset_offset_seconds"], "ictal_spread.onset_offset_seconds", minimum=0)
    data["from_electrodes"] = _electrode_list(data["from_electrodes"], "ictal_spread.from_electrodes")
    data["to_electrodes"] = _electrode_list(data["to_electrodes"], "ictal_spread.to_electrodes")
    if set(data["from_electrodes"]) == set(data["to_electrodes"]):
        raise ValueError("ictal_spread from/to electrodes must differ")
    data["to_regions"] = _string_list(data["to_regions"], "ictal_spread.to_regions", allowed=_REGIONS)
    data["laterality"] = _enum(data["laterality"], _LATERALITY, "ictal_spread.laterality")
    return data


def _validate_ictal_termination(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("offset_seconds", "duration_seconds"),
        optional=("qualification",),
        context="ictal_termination.value",
    )
    data["offset_seconds"] = _number(data["offset_seconds"], "ictal_termination.offset_seconds", minimum=0)
    data["duration_seconds"] = _number(data["duration_seconds"], "ictal_termination.duration_seconds", exclusive_minimum=0)
    if "qualification" in data:
        data["qualification"] = _validate_signal_finding_qualification(
            data["qualification"],
            context="ictal_termination.qualification",
        )
    return data


def _validate_postictal(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("onset_offset_seconds", "pattern", "laterality", "regions", "electrodes"),
        optional=("duration_seconds",),
        context="postictal_pattern.value",
    )
    data["onset_offset_seconds"] = _number(data["onset_offset_seconds"], "postictal.onset_offset_seconds", minimum=0)
    data["pattern"] = _enum(data["pattern"], ("suppression", "attenuation", "slowing", "periodic_discharge", "return_to_baseline", "other"), "postictal.pattern")
    data["laterality"] = _enum(data["laterality"], _LATERALITY, "postictal.laterality")
    data["regions"] = _string_list(data["regions"], "postictal.regions", allowed=_REGIONS)
    data["electrodes"] = _electrode_list(data["electrodes"], "postictal.electrodes", allow_empty=True)
    if "duration_seconds" in data:
        data["duration_seconds"] = _number(data["duration_seconds"], "postictal.duration_seconds", exclusive_minimum=0)
    return data


def _validate_impression(
    value: object,
    *,
    classification: bool = False,
    ictal_eeg: bool = False,
) -> dict[str, Any]:
    required = ["statement", "supported_fact_ids"]
    optional: list[str] = []
    if classification:
        required.append("classification")
    if ictal_eeg:
        required.append("eeg_event_ids")
    data = _strict_object(value, required=required, optional=optional, context="impression.value")
    data["statement"] = _eeg_only_text(data["statement"], "impression.statement")
    data["supported_fact_ids"] = _id_list(data["supported_fact_ids"], "impression.supported_fact_ids")
    if classification:
        data["classification"] = _enum(data["classification"], ("normal", "abnormal", "limited", "indeterminate"), "impression.classification")
    if ictal_eeg:
        data["eeg_event_ids"] = _id_list(
            data["eeg_event_ids"], "impression.eeg_event_ids"
        )
    return data


def _validate_normal_variant_or_uncertain_pattern(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=(
            "pattern_code",
            "classification",
            "laterality",
            "electrodes",
        ),
        context="normal_variant_or_uncertain_pattern.value",
    )
    data["pattern_code"] = _enum(
        data["pattern_code"],
        (
            "lambda_waves",
            "wicket_waves",
            "SREDA",
            "breach_rhythm",
            "uncertain_sharp_transient",
            "other",
        ),
        "normal_variant_or_uncertain_pattern.pattern_code",
    )
    data["classification"] = _enum(
        data["classification"],
        ("normal_variant", "uncertain_significance"),
        "normal_variant_or_uncertain_pattern.classification",
    )
    data["laterality"] = _enum(
        data["laterality"],
        _LATERALITY,
        "normal_variant_or_uncertain_pattern.laterality",
    )
    data["electrodes"] = _electrode_list(
        data["electrodes"],
        "normal_variant_or_uncertain_pattern.electrodes",
        allow_empty=True,
    )
    return data


FactValidator = Callable[[object], dict[str, Any]]


FACT_VALUE_VALIDATORS: Mapping[str, FactValidator] = {
    "recording_modality": lambda value: _simple_enum_value(
        value,
        key="modality",
        allowed=(
            "routine_scalp_eeg",
            "ambulatory_scalp_eeg",
            "long_term_scalp_eeg",
            "continuous_scalp_eeg",
        ),
        context="recording_modality.value",
    ),
    "recording_duration": _validate_recording_duration,
    "electrode_setup": _validate_electrode_setup,
    "acquisition_settings": _validate_acquisition_settings,
    "recording_quality": _validate_recording_quality,
    "posterior_dominant_rhythm": _validate_pdr,
    "background_organization": _validate_background_organization,
    "background_slowing": _validate_background_slowing,
    "background_asymmetry": _validate_background_asymmetry,
    "artifact_observation": _validate_artifact,
    "epileptiform_discharge": lambda value: _validate_interictal(value, family="epileptiform_discharge"),
    "periodic_pattern": lambda value: _validate_interictal(value, family="periodic_pattern"),
    "rhythmic_delta_activity": lambda value: _validate_interictal(value, family="rhythmic_delta_activity"),
    "normal_variant_or_uncertain_pattern": _validate_normal_variant_or_uncertain_pattern,
    "electrographic_event_occurrence": _validate_electrographic_event_occurrence,
    "source_eeg_annotation_timing": _validate_source_eeg_annotation_timing,
    "algorithmic_sustained_eeg_change": _validate_algorithmic_sustained_eeg_change,
    "later_scalp_visible_eeg_change": _validate_later_scalp_visible_eeg_change,
    "ictal_onset_pattern": _validate_ictal_onset,
    "ictal_evolution": _validate_ictal_evolution,
    "ictal_spread": _validate_ictal_spread,
    "ictal_termination": _validate_ictal_termination,
    "postictal_pattern": _validate_postictal,
    "study_classification": lambda value: _validate_impression(value, classification=True),
    "interictal_impression": _validate_impression,
    "ictal_eeg_impression": lambda value: _validate_impression(value, ictal_eeg=True),
    "recording_limitation": _validate_impression,
}


@dataclass(frozen=True)
class FactProvenance:
    source_type: ProvenanceSource
    source_id: str
    method: str
    acquired_at: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_type, ProvenanceSource):
            raise TypeError("provenance.source_type must be ProvenanceSource")
        object.__setattr__(
            self,
            "source_id",
            _eeg_source_identifier(self.source_id, "provenance.source_id"),
        )
        object.__setattr__(self, "method", _eeg_only_text(self.method, "provenance.method"))
        if self.acquired_at is not None:
            object.__setattr__(self, "acquired_at", _timestamp(self.acquired_at, "provenance.acquired_at"))

    @classmethod
    def from_dict(cls, value: object) -> "FactProvenance":
        data = _strict_object(
            value,
            required=("source_type", "source_id", "method"),
            optional=("acquired_at",),
            context="provenance",
        )
        try:
            source_type = ProvenanceSource(data["source_type"])
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported provenance.source_type") from exc
        return cls(
            source_type=source_type,
            source_id=data["source_id"],
            method=data["method"],
            acquired_at=data.get("acquired_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "method": self.method,
        }
        if self.acquired_at is not None:
            result["acquired_at"] = self.acquired_at
        return result


@dataclass(frozen=True)
class FactVerification:
    status: VerificationStatus
    verified_by: str | None = None
    verified_at: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, VerificationStatus):
            raise TypeError("verification.status must be VerificationStatus")
        requires_verifier = self.status in {
            VerificationStatus.TECHNOLOGIST_VERIFIED,
            VerificationStatus.PHYSICIAN_VERIFIED,
        }
        if requires_verifier:
            if self.verified_by is None or self.verified_at is None:
                raise ValueError("verified facts require verified_by and verified_at")
            object.__setattr__(self, "verified_by", _identifier(self.verified_by, "verification.verified_by", source=True))
            object.__setattr__(self, "verified_at", _timestamp(self.verified_at, "verification.verified_at"))
        elif self.verified_by is not None or self.verified_at is not None:
            raise ValueError("unverified/algorithm facts cannot name a verifier")

    @classmethod
    def from_dict(cls, value: object) -> "FactVerification":
        data = _strict_object(
            value,
            required=("status",),
            optional=("verified_by", "verified_at"),
            context="verification",
        )
        try:
            status = VerificationStatus(data["status"])
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported verification.status") from exc
        return cls(status=status, verified_by=data.get("verified_by"), verified_at=data.get("verified_at"))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"status": self.status.value}
        if self.verified_by is not None:
            result["verified_by"] = self.verified_by
            result["verified_at"] = self.verified_at
        return result


@dataclass(frozen=True)
class AtomicFact:
    fact_id: str
    section: FactSection
    fact_type: str
    state: FactState
    value: Mapping[str, Any] | None
    provenance: FactProvenance
    verification: FactVerification
    evidence_ids: tuple[str, ...]
    eeg_event_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_id", _identifier(self.fact_id, "fact_id"))
        if not isinstance(self.section, FactSection):
            raise TypeError("section must be FactSection")
        if not isinstance(self.state, FactState):
            raise TypeError("state must be FactState")
        if not isinstance(self.provenance, FactProvenance):
            raise TypeError("provenance must be FactProvenance")
        if not isinstance(self.verification, FactVerification):
            raise TypeError("verification must be FactVerification")
        expected_section = FACT_TYPE_TO_SECTION.get(self.fact_type)
        if expected_section is None:
            raise ValueError(f"unsupported fact_type: {self.fact_type!r}")
        if self.section is not expected_section:
            raise ValueError(
                f"fact_type {self.fact_type!r} belongs to section {expected_section.value!r}"
            )
        if self.section is FactSection.ICTAL:
            if self.eeg_event_id is None:
                raise ValueError("ictal facts require eeg_event_id")
        elif self.eeg_event_id is not None:
            raise ValueError(f"{self.section.value} facts cannot carry eeg_event_id")
        if self.eeg_event_id is not None:
            object.__setattr__(
                self,
                "eeg_event_id",
                _identifier(self.eeg_event_id, "eeg_event_id"),
            )

        if not isinstance(self.evidence_ids, tuple):
            raise TypeError("evidence_ids must be a tuple")
        evidence = tuple(
            _eeg_source_identifier(item, "evidence_id") for item in self.evidence_ids
        )
        if len(evidence) != len(set(evidence)):
            raise ValueError("evidence_ids contain duplicates")
        if self.state not in {FactState.NOT_RECORDED, FactState.NOT_ASSESSABLE} and not evidence:
            raise ValueError("assessed facts require at least one evidence_id")
        object.__setattr__(self, "evidence_ids", evidence)

        if self.state in {FactState.PRESENT, FactState.UNCERTAIN}:
            if self.value is None:
                raise ValueError(f"{self.state.value} facts require a typed value")
            if not isinstance(self.value, dict):
                raise TypeError("fact.value must be an object")
            raw_value = deepcopy(self.value)
            text_zh = raw_value.pop("text_zh", None)
            normalized = FACT_VALUE_VALIDATORS[self.fact_type](raw_value)
            if text_zh is not None:
                text_validator = (
                    _neutral_temporal_eeg_text
                    if self.fact_type
                    in {
                        "algorithmic_sustained_eeg_change",
                        "later_scalp_visible_eeg_change",
                    }
                    else _eeg_only_text
                )
                normalized["text_zh"] = text_validator(text_zh, "fact.value.text_zh")
            object.__setattr__(self, "value", normalized)
        elif self.value is not None:
            raise ValueError(f"{self.state.value} facts must use value=null")

        if self.section is FactSection.IMPRESSION:
            if self.state is not FactState.PRESENT:
                raise ValueError("impression facts must be present")
            if self.verification.status is not VerificationStatus.PHYSICIAN_VERIFIED:
                raise ValueError("impression facts require physician_verified status")
        if self.fact_type == "electrographic_event_occurrence" and self.state not in {
            FactState.PRESENT,
            FactState.UNCERTAIN,
        }:
            raise ValueError(
                "electrographic_event_occurrence must be present or uncertain"
            )
        if self.fact_type == "electrographic_event_occurrence" and self.value is not None:
            is_uncertain_class = (
                self.value["event_class"] == "uncertain_electrographic_pattern"
            )
            if is_uncertain_class != (self.state is FactState.UNCERTAIN):
                raise ValueError(
                    "uncertain_electrographic_pattern requires state=uncertain; "
                    "other event classes require state=present"
                )
            if self.value["event_class"] == "electrographic_seizure":
                if self.verification.status is not VerificationStatus.PHYSICIAN_VERIFIED:
                    raise ValueError(
                        "electrographic_seizure occurrence requires physician_verified status"
                    )
                if float(self.value["duration_seconds"]) < 10:
                    raise ValueError(
                        "electrographic_seizure must last at least 10 seconds; "
                        "otherwise downgrade its event_class"
                    )

    @classmethod
    def from_dict(cls, value: object) -> "AtomicFact":
        data = _strict_object(
            value,
            required=("fact_id", "section", "fact_type", "state", "value", "provenance", "verification", "evidence_ids"),
            optional=("eeg_event_id",),
            context="fact",
        )
        try:
            section = FactSection(data["section"])
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported fact.section") from exc
        try:
            state = FactState(data["state"])
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported fact.state") from exc
        evidence_ids = _id_list(
            data["evidence_ids"],
            "fact.evidence_ids",
            allow_empty=True,
            source=True,
        )
        return cls(
            fact_id=data["fact_id"],
            section=section,
            fact_type=_nonempty_string(data["fact_type"], "fact.fact_type"),
            state=state,
            value=data["value"],
            provenance=FactProvenance.from_dict(data["provenance"]),
            verification=FactVerification.from_dict(data["verification"]),
            evidence_ids=tuple(evidence_ids),
            eeg_event_id=data.get("eeg_event_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "fact_id": self.fact_id,
            "section": self.section.value,
            "fact_type": self.fact_type,
            "state": self.state.value,
            "value": deepcopy(self.value),
            "provenance": self.provenance.to_dict(),
            "verification": self.verification.to_dict(),
            "evidence_ids": list(self.evidence_ids),
        }
        if self.eeg_event_id is not None:
            result["eeg_event_id"] = self.eeg_event_id
        return result


def _referenced_ids(fact: AtomicFact) -> tuple[str, ...]:
    if fact.value is None:
        return ()
    references: list[str] = []
    for key in ("target_fact_ids", "abnormality_fact_ids", "supported_fact_ids"):
        raw = fact.value.get(key)
        if isinstance(raw, list):
            references.extend(raw)
    return tuple(references)


@dataclass(frozen=True)
class ClinicalEEGReport:
    schema_version: str
    report_id: str
    patient_pseudonym: str
    facts: tuple[AtomicFact, ...]
    eeg_event_ids: tuple[str, ...]
    impression_fact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        object.__setattr__(self, "report_id", _identifier(self.report_id, "report_id", source=True))
        object.__setattr__(self, "patient_pseudonym", _identifier(self.patient_pseudonym, "patient_pseudonym", source=True))
        if not isinstance(self.facts, tuple) or not self.facts:
            raise TypeError("facts must be a non-empty tuple")
        if not all(isinstance(fact, AtomicFact) for fact in self.facts):
            raise TypeError("facts must contain AtomicFact values")
        by_id = {fact.fact_id: fact for fact in self.facts}
        if len(by_id) != len(self.facts):
            raise ValueError("fact_id values must be unique")

        event_ids = tuple(_identifier(item, "eeg_event_id") for item in self.eeg_event_ids)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("eeg_event_ids contain duplicates")
        object.__setattr__(self, "eeg_event_ids", event_ids)
        used_event_ids = {
            fact.eeg_event_id for fact in self.facts if fact.eeg_event_id is not None
        }
        if set(event_ids) != used_event_ids:
            raise ValueError("eeg_event_ids must exactly match event-linked facts")
        event_numbers: list[int] = []
        for eeg_event_id in event_ids:
            occurrences = [
                fact for fact in self.facts
                if fact.eeg_event_id == eeg_event_id
                and fact.fact_type == "electrographic_event_occurrence"
            ]
            if len(occurrences) != 1:
                raise ValueError(
                    f"EEG event {eeg_event_id!r} requires exactly one "
                    "electrographic_event_occurrence fact"
                )
            occurrence_value = occurrences[0].value
            if occurrence_value is None:  # guarded by AtomicFact, kept defensive
                raise ValueError("electrographic event occurrence requires a value")
            event_numbers.append(int(occurrence_value["event_number"]))
            if occurrence_value["event_class"] == "electrographic_seizure":
                required_components = {
                    "ictal_onset_pattern",
                    "ictal_evolution",
                    "ictal_termination",
                }
                present_components = {
                    fact.fact_type
                    for fact in self.facts
                    if fact.eeg_event_id == eeg_event_id
                    and fact.state is FactState.PRESENT
                    and fact.value is not None
                }
                missing_components = required_components.difference(
                    present_components
                )
                if missing_components:
                    raise ValueError(
                        f"electrographic_seizure {eeg_event_id!r} lacks required "
                        f"EEG facts {sorted(missing_components)}; downgrade its "
                        "event_class when seizure criteria are not met"
                    )
        if len(event_numbers) != len(set(event_numbers)):
            raise ValueError("electrographic event numbers must be unique")

        impression_ids = tuple(_identifier(item, "impression_fact_id") for item in self.impression_fact_ids)
        if len(impression_ids) != len(set(impression_ids)):
            raise ValueError("impression_fact_ids contain duplicates")
        object.__setattr__(self, "impression_fact_ids", impression_ids)
        actual_impressions = {
            fact.fact_id for fact in self.facts if fact.section is FactSection.IMPRESSION
        }
        if set(impression_ids) != actual_impressions:
            raise ValueError("impression_fact_ids must exactly match impression facts")

        for fact in self.facts:
            for reference in _referenced_ids(fact):
                if reference == fact.fact_id:
                    raise ValueError(f"fact {fact.fact_id!r} cannot support itself")
                target = by_id.get(reference)
                if target is None:
                    raise ValueError(f"fact {fact.fact_id!r} references unknown fact {reference!r}")
                if fact.section is FactSection.IMPRESSION and target.section is FactSection.IMPRESSION:
                    raise ValueError("impression facts must be supported by non-impression facts")
            if fact.fact_type == "ictal_eeg_impression" and fact.value is not None:
                referenced_events = set(fact.value["eeg_event_ids"])
                if not referenced_events.issubset(event_ids):
                    raise ValueError(
                        "ictal_eeg_impression references an unknown eeg_event_id"
                    )
                supported = {
                    by_id[fact_id].eeg_event_id
                    for fact_id in fact.value["supported_fact_ids"]
                    if by_id[fact_id].section is FactSection.ICTAL
                }
                if not referenced_events.issubset(supported):
                    raise ValueError(
                        "ictal_eeg_impression must support every referenced EEG event"
                    )

    @classmethod
    def from_dict(cls, value: object) -> "ClinicalEEGReport":
        data = _strict_object(
            value,
            required=(
                "schema_version",
                "report_id",
                "patient_pseudonym",
                "facts",
                "eeg_event_ids",
                "impression_fact_ids",
            ),
            context="clinical EEG report",
        )
        if not isinstance(data["facts"], list) or not data["facts"]:
            raise TypeError("clinical EEG report facts must be a non-empty list")
        facts = tuple(AtomicFact.from_dict(item) for item in data["facts"])
        events = _id_list(data["eeg_event_ids"], "eeg_event_ids", allow_empty=True)
        impressions = _id_list(data["impression_fact_ids"], "impression_fact_ids", allow_empty=True)
        return cls(
            schema_version=data["schema_version"],
            report_id=data["report_id"],
            patient_pseudonym=data["patient_pseudonym"],
            facts=facts,
            eeg_event_ids=tuple(events),
            impression_fact_ids=tuple(impressions),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "patient_pseudonym": self.patient_pseudonym,
            "facts": [fact.to_dict() for fact in self.facts],
            "eeg_event_ids": list(self.eeg_event_ids),
            "impression_fact_ids": list(self.impression_fact_ids),
        }

    def facts_for_section(self, section: FactSection | str) -> tuple[AtomicFact, ...]:
        """Return facts in source order for one report section."""

        try:
            selected = section if isinstance(section, FactSection) else FactSection(section)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported section: {section!r}") from exc
        return tuple(fact for fact in self.facts if fact.section is selected)


def validate_report_payload(payload: object) -> ClinicalEEGReport:
    """Parse, canonicalize, and cross-validate an untrusted JSON payload."""

    return ClinicalEEGReport.from_dict(payload)


def fact_observation_text_zh(fact: AtomicFact) -> str:
    """Return grounded Chinese observation text for prompts or fallback prose.

    A supplied ``value.text_zh`` remains explicitly qualified when it has not
    been verified by a physician.  For facts without prose (including null
    negative/missing states), the function returns a deterministic state
    statement and never turns missing evidence into a negative finding.
    """

    if not isinstance(fact, AtomicFact):
        raise TypeError("fact must be AtomicFact")
    label = FACT_TYPE_LABEL_ZH[fact.fact_type]
    supplied = None if fact.value is None else fact.value.get("text_zh")
    if supplied is not None:
        base = str(supplied)
    elif fact.state is FactState.PRESENT:
        base = f"{label}：已记录为存在，具体结构化参数见事实值。"
    elif fact.state is FactState.ABSENT:
        base = f"{label}：已评估，未见相应表现。"
    elif fact.state is FactState.NOT_RECORDED:
        base = f"{label}：未记录。"
    elif fact.state is FactState.NOT_ASSESSABLE:
        base = f"{label}：现有记录无法评估。"
    else:
        base = f"{label}：现有证据不确定，具体结构化参数见事实值。"

    if fact.verification.status is VerificationStatus.PHYSICIAN_VERIFIED:
        return base
    if fact.verification.status is VerificationStatus.TECHNOLOGIST_VERIFIED:
        return f"技师已核对、尚未经医师确认：{base}"
    if fact.verification.status is VerificationStatus.ALGORITHM_CANDIDATE:
        return f"算法候选、尚未经医师确认：{base}"
    return f"未核实观察：{base}"


__all__ = [
    "SCHEMA_VERSION",
    "CANONICAL_ELECTRODES",
    "ELECTRODE_ALIASES",
    "FACT_TYPE_TO_SECTION",
    "FACT_TYPE_LABEL_ZH",
    "FactState",
    "FactSection",
    "VerificationStatus",
    "ProvenanceSource",
    "FactProvenance",
    "FactVerification",
    "AtomicFact",
    "ClinicalEEGReport",
    "canonicalize_electrode",
    "canonicalize_derivation",
    "fact_observation_text_zh",
    "validate_report_payload",
]
