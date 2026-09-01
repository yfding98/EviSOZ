"""Immutable canonical-signal and task-specific EEG view contracts.

This module is deliberately a metadata/receipt layer.  It does not open EDF
files, call an annotation API, run a detector, or transform a signal tensor.
Instead, it binds one physical recording clock to independently versioned
detector, boundary, Findings, spatial and display views and validates the
properties needed before a derived tensor can support EEG evidence.

The contract is fail-closed:

* physical time is an integer sample-edge mapping on a rational clock;
* annotations, spreadsheets, clinical text, video and labels are forbidden;
* reference matrices, transform specifications, masks and tensors are hashed;
* padding, filter edges and quality intervals cannot support evidence;
* unobserved or imputed units are never evidence eligible; and
* detector-provider tensors never become clinical Findings evidence; and
* progressive-window cache reuse is allowed only for immutable global tiles.

All public validators return deep copies.  A validated receipt can therefore
be safely retained while the caller continues to mutate its input object.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


CANONICAL_SIGNAL_SCHEMA_VERSION = "canonical_eeg_signal_v1"
TRANSFORM_SPEC_SCHEMA_VERSION = "clinical_eeg_signal_transform_v1"
SIGNAL_VIEW_SCHEMA_VERSION = "clinical_eeg_signal_view_v3"
SIGNAL_VIEW_DAG_SCHEMA_VERSION = "clinical_eeg_signal_view_dag_v1"
CROSS_VIEW_ALIGNMENT_SCHEMA_VERSION = "clinical_eeg_cross_view_alignment_v1"
VIEW_EVIDENCE_ELIGIBILITY_SCHEMA_VERSION = "view_evidence_eligibility_v1"
CACHE_EXTENSION_SCHEMA_VERSION = "clinical_eeg_view_cache_extension_v1"
TEMPORAL_EVIDENCE_SCHEMA_VERSION = "clinical_eeg_temporal_evidence_v1"

COORDINATE_SYSTEM = "recording_relative_seconds"
GLOBAL_CLOCK_POLICY = "integer_sample_edges_on_rational_global_clock_v1"
CACHE_EXPANSION_POLICY = "immutable_global_output_tiles_append_or_reveal_v1"
NONREUSABLE_CACHE_POLICY = "interval_bound_recompute_required_v1"
HIGH_FREQUENCY_MIN_HZ = 40.0
DETECTOR_PROVIDER_EVIDENCE_REASON_CODE = "detector_provider_view_not_clinical_evidence"
ONSET_FIR_RESPONSE_AUTHORIZATION_SOFTWARE_KEY = (
    "fir_response_target_band_claim_authorized"
)
ONSET_FIR_RESPONSE_UNQUALIFIED_REASON_CODE = "causal_onset_response_unqualified"
ONSET_FIR_CLINICAL_ADMISSION_AUTHORIZATION_SOFTWARE_KEY = (
    "fir_clinical_onset_support_authorized"
)
ONSET_FIR_CLINICAL_ADMISSION_UNQUALIFIED_REASON_CODE = (
    "causal_onset_clinical_admission_unqualified"
)
CANONICAL_EDF_ONSET_TRANSFORM_NAME = (
    "edf_to_onset_causal_linear_phase_fir_referential_v1"
)

TASK_ROLES = (
    "detector_provider",
    "detector_native",
    "boundary_coarse",
    "findings_native",
    "findings_clinical",
    "findings_native_morphology",
    "onset_causal",
    "context_offline",
    "spatial_reference",
    "waveform_display",
)
EVIDENCE_FAMILIES = (
    "amplitude",
    "morphology",
    "spectral",
    "spatial_field",
    "high_frequency",
    "waveform",
)
QUALITY_KINDS = (
    "missing",
    "flat",
    "clipping",
    "step",
    "line_noise",
    "gap",
    "other_signal_quality",
)
_FIREWALL = {
    "eeg_samples_used": True,
    "edf_annotation_api_called": False,
    "edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "video_used": False,
    "sleep_or_activation_labels_used": False,
}
_SHA256_LENGTH = 64
_TOL = 1e-8
_UNIT_TO_VOLTS = {"V": 1.0, "mV": 1e-3, "uV": 1e-6}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(value: object, context: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not _is_sha256(value):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return str(value)


def _nonempty(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    return value


def _strict_dict(value: object, required: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    if set(value) != required:
        missing = sorted(required.difference(value))
        unknown = sorted(set(value).difference(required))
        raise ValueError(
            f"{context} has missing or unknown fields; "
            f"missing={missing}, unknown={unknown}"
        )
    return deepcopy(value)


def _finite(value: object, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return result


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{context} must be an integer >= {minimum}")
    return value


def _unique_strings(
    values: object, context: str, *, nonempty: bool = True
) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{context} must be an array")
    result = [
        _nonempty(item, f"{context}[{index}]") for index, item in enumerate(values)
    ]
    if nonempty and not result:
        raise ValueError(f"{context} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{context} contains duplicates")
    return result


def _rate(numerator: object, denominator: object, context: str) -> tuple[int, int]:
    num = _integer(numerator, f"{context}.numerator", minimum=1)
    den = _integer(denominator, f"{context}.denominator", minimum=1)
    if math.gcd(num, den) != 1:
        raise ValueError(f"{context} must be reduced to lowest terms")
    return num, den


def _clock_rate(clock: Mapping[str, object], context: str) -> tuple[int, int]:
    return _rate(
        clock["sampling_rate_numerator"],
        clock["sampling_rate_denominator"],
        context,
    )


def _sample_edge_to_seconds(index: int, numerator: int, denominator: int) -> float:
    return float(Fraction(index * denominator, numerator))


def _seconds_to_sample_edge(
    seconds: object,
    numerator: int,
    denominator: int,
    *,
    context: str,
    rounding: str = "exact",
) -> int:
    value = _finite(seconds, context, minimum=0.0)
    position = value * numerator / denominator
    if rounding == "exact":
        result = int(round(position))
        if abs(position - result) > _TOL:
            raise ValueError(f"{context} is not aligned to a sample edge")
        return result
    if rounding == "floor":
        return int(math.floor(position + _TOL))
    if rounding == "ceil":
        return int(math.ceil(position - _TOL))
    if rounding == "nearest":
        return int(round(position))
    raise ValueError("rounding must be exact, floor, ceil or nearest")


def _integer_interval(
    value: object,
    context: str,
    *,
    lower: int = 0,
    upper: int | None = None,
    positive: bool = True,
) -> tuple[int, int]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise TypeError(f"{context} must be a two-item integer interval")
    start = _integer(value[0], f"{context}[0]", minimum=lower)
    stop = _integer(value[1], f"{context}[1]", minimum=lower)
    if (positive and stop <= start) or (not positive and stop < start):
        raise ValueError(f"{context} is not ordered")
    if upper is not None and stop > upper:
        raise ValueError(f"{context} lies outside [0, {upper}]")
    return start, stop


def _float_interval(
    value: object,
    context: str,
    *,
    lower: float = 0.0,
    upper: float | None = None,
) -> tuple[float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise TypeError(f"{context} must be a two-item interval")
    start = _finite(value[0], f"{context}[0]", minimum=lower)
    stop = _finite(value[1], f"{context}[1]", minimum=lower)
    if stop <= start + _TOL:
        raise ValueError(f"{context} must have positive duration")
    if upper is not None and stop > upper + _TOL:
        raise ValueError(f"{context} lies outside [0, {upper}]")
    return start, stop


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _covers(carrier: tuple[int, int], target: tuple[int, int]) -> bool:
    return carrier[0] <= target[0] and carrier[1] >= target[1]


def _canonical_identity(data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": data["schema_version"],
        "recording_id": data["recording_id"],
        "source_signal_sha256": data["source_signal_sha256"],
        "recording_duration_seconds": data["recording_duration_seconds"],
        "coordinate_system": data["coordinate_system"],
        "clock_policy": data["clock_policy"],
        "channels": data["channels"],
        "annotation_firewall": data["annotation_firewall"],
    }


def build_canonical_signal_receipt(
    *,
    recording_id: str,
    source_signal_sha256: str,
    recording_duration_seconds: float,
    channels: Sequence[Mapping[str, object]],
    quality_primitives: Sequence[Mapping[str, object]] = (),
) -> dict[str, Any]:
    """Build one immutable physical-signal root without reading an EDF."""

    body: dict[str, Any] = {
        "schema_version": CANONICAL_SIGNAL_SCHEMA_VERSION,
        "canonical_signal_id": "CONTENT-ADDRESS-PENDING",
        "recording_id": recording_id,
        "source_signal_sha256": source_signal_sha256,
        "recording_duration_seconds": float(recording_duration_seconds),
        "coordinate_system": COORDINATE_SYSTEM,
        "clock_policy": GLOBAL_CLOCK_POLICY,
        "channels": [deepcopy(dict(item)) for item in channels],
        "quality_primitives": [deepcopy(dict(item)) for item in quality_primitives],
        "annotation_firewall": deepcopy(_FIREWALL),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["canonical_signal_id"] = (
        "CANONICAL-" + _canonical_sha256(_canonical_identity(body))[:24]
    )
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_canonical_signal_receipt(body)


def validate_canonical_signal_receipt(payload: object) -> dict[str, Any]:
    """Validate the immutable mother-signal metadata and physical clocks."""

    required = {
        "schema_version",
        "canonical_signal_id",
        "recording_id",
        "source_signal_sha256",
        "recording_duration_seconds",
        "coordinate_system",
        "clock_policy",
        "channels",
        "quality_primitives",
        "annotation_firewall",
        "receipt_sha256",
    }
    data = _strict_dict(payload, required, "canonical signal receipt")
    if data["schema_version"] != CANONICAL_SIGNAL_SCHEMA_VERSION:
        raise ValueError("unsupported canonical signal schema")
    _nonempty(data["recording_id"], "canonical recording_id")
    _sha256(data["source_signal_sha256"], "canonical source_signal_sha256")
    duration = _finite(
        data["recording_duration_seconds"],
        "canonical recording_duration_seconds",
        minimum=_TOL,
    )
    if data["coordinate_system"] != COORDINATE_SYSTEM:
        raise ValueError("canonical coordinate system drifted")
    if data["clock_policy"] != GLOBAL_CLOCK_POLICY:
        raise ValueError("canonical clock policy drifted")
    if data["annotation_firewall"] != _FIREWALL:
        raise ValueError("canonical signal violates the annotation firewall")

    channel_required = {
        "channel_id",
        "raw_label",
        "canonical_name",
        "source_physical_unit",
        "scale_to_volts",
        "sample_rate_numerator",
        "sample_rate_denominator",
        "sample_count",
        "observed",
        "imputed",
        "acquisition_highpass_hz",
        "acquisition_lowpass_hz",
        "reference_label",
    }
    if not isinstance(data["channels"], list) or not data["channels"]:
        raise TypeError("canonical channels must be a non-empty array")
    channel_ids: set[str] = set()
    normalized_channels: list[dict[str, Any]] = []
    for index, raw in enumerate(data["channels"]):
        channel = _strict_dict(raw, channel_required, f"canonical channels[{index}]")
        channel_id = _nonempty(channel["channel_id"], f"channels[{index}].channel_id")
        if channel_id in channel_ids:
            raise ValueError("canonical channel IDs must be unique")
        channel_ids.add(channel_id)
        _nonempty(channel["raw_label"], f"channels[{index}].raw_label")
        _nonempty(channel["canonical_name"], f"channels[{index}].canonical_name")
        unit = channel["source_physical_unit"]
        if unit not in _UNIT_TO_VOLTS:
            raise ValueError(f"channels[{index}] has unsupported physical unit")
        scale = _finite(channel["scale_to_volts"], f"channels[{index}].scale_to_volts")
        if not math.isclose(
            scale, _UNIT_TO_VOLTS[str(unit)], rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(
                f"channels[{index}] scale_to_volts disagrees with its unit"
            )
        numerator, denominator = _rate(
            channel["sample_rate_numerator"],
            channel["sample_rate_denominator"],
            f"channels[{index}].sample_rate",
        )
        count = _integer(channel["sample_count"], f"channels[{index}].sample_count")
        if (
            type(channel["observed"]) is not bool
            or type(channel["imputed"]) is not bool
        ):
            raise TypeError(f"channels[{index}] observed/imputed must be boolean")
        if channel["imputed"] is not False:
            raise ValueError(
                "canonical mother signal must never contain imputed samples"
            )
        if channel["observed"]:
            end_seconds = _sample_edge_to_seconds(count, numerator, denominator)
            if count <= 0 or abs(end_seconds - duration) > _TOL:
                raise ValueError(
                    f"channels[{index}] sample clock does not end at recording duration"
                )
        elif count != 0:
            raise ValueError("unobserved canonical channels must have sample_count=0")
        nyquist = 0.5 * numerator / denominator
        highpass = channel["acquisition_highpass_hz"]
        lowpass = channel["acquisition_lowpass_hz"]
        if highpass is not None:
            highpass = _finite(
                highpass, f"channels[{index}].acquisition_highpass_hz", minimum=0.0
            )
            if highpass >= nyquist:
                raise ValueError(
                    f"channels[{index}] acquisition highpass exceeds Nyquist"
                )
        if lowpass is not None:
            lowpass = _finite(
                lowpass, f"channels[{index}].acquisition_lowpass_hz", minimum=_TOL
            )
            if lowpass > nyquist + _TOL:
                raise ValueError(
                    f"channels[{index}] acquisition lowpass exceeds Nyquist"
                )
        if highpass is not None and lowpass is not None and highpass >= lowpass:
            raise ValueError(f"channels[{index}] acquisition bandwidth is empty")
        _nonempty(channel["reference_label"], f"channels[{index}].reference_label")
        normalized_channels.append(channel)
    data["channels"] = normalized_channels

    primitive_required = {
        "quality_id",
        "channel_ids",
        "start_recording_seconds",
        "stop_recording_seconds",
        "kind",
        "severity",
        "disabled_evidence_families",
    }
    if not isinstance(data["quality_primitives"], list):
        raise TypeError("canonical quality_primitives must be an array")
    quality_ids: set[str] = set()
    normalized_primitives: list[dict[str, Any]] = []
    channels_by_id = {str(row["channel_id"]): row for row in normalized_channels}
    for index, raw in enumerate(data["quality_primitives"]):
        primitive = _strict_dict(
            raw,
            primitive_required,
            f"canonical quality_primitives[{index}]",
        )
        quality_id = _nonempty(
            primitive["quality_id"], f"quality_primitives[{index}].quality_id"
        )
        if quality_id in quality_ids:
            raise ValueError("canonical quality IDs must be unique")
        quality_ids.add(quality_id)
        affected = _unique_strings(
            primitive["channel_ids"], f"quality_primitives[{index}].channel_ids"
        )
        if not set(affected).issubset(channel_ids):
            raise ValueError(f"quality_primitives[{index}] references unknown channels")
        start, stop = _float_interval(
            [
                primitive["start_recording_seconds"],
                primitive["stop_recording_seconds"],
            ],
            f"quality_primitives[{index}].interval",
            upper=duration,
        )
        if primitive["kind"] not in QUALITY_KINDS:
            raise ValueError(f"quality_primitives[{index}] has unsupported kind")
        if primitive["severity"] not in {"limited", "unusable"}:
            raise ValueError(f"quality_primitives[{index}] has unsupported severity")
        disabled = _unique_strings(
            primitive["disabled_evidence_families"],
            f"quality_primitives[{index}].disabled_evidence_families",
        )
        if not set(disabled).issubset(EVIDENCE_FAMILIES):
            raise ValueError(f"quality_primitives[{index}] disables unknown families")
        if primitive["severity"] == "unusable" and set(disabled) != set(
            EVIDENCE_FAMILIES
        ):
            raise ValueError(
                "unusable quality primitives must disable every evidence family"
            )
        for channel_id in affected:
            channel = channels_by_id[channel_id]
            if not channel["observed"]:
                continue
            numerator = int(channel["sample_rate_numerator"])
            denominator = int(channel["sample_rate_denominator"])
            _seconds_to_sample_edge(
                start,
                numerator,
                denominator,
                context=f"quality_primitives[{index}].start",
            )
            _seconds_to_sample_edge(
                stop,
                numerator,
                denominator,
                context=f"quality_primitives[{index}].stop",
            )
        normalized_primitives.append(primitive)
    data["quality_primitives"] = normalized_primitives

    expected_id = "CANONICAL-" + _canonical_sha256(_canonical_identity(data))[:24]
    if data["canonical_signal_id"] != expected_id:
        raise ValueError("canonical signal ID does not bind its physical identity")
    _sha256(data["receipt_sha256"], "canonical receipt_sha256")
    digest_source = deepcopy(data)
    digest_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("canonical signal receipt hash does not bind its content")
    return data


def canonical_sample_index_to_recording_seconds(
    canonical_receipt: object,
    *,
    channel_id: str,
    sample_index: int,
) -> float:
    canonical = validate_canonical_signal_receipt(canonical_receipt)
    channel = next(
        (row for row in canonical["channels"] if row["channel_id"] == channel_id),
        None,
    )
    if channel is None:
        raise KeyError(f"unknown canonical channel: {channel_id}")
    if not channel["observed"]:
        raise ValueError("unobserved canonical channel has no physical sample mapping")
    index = _integer(sample_index, "sample_index")
    if index > channel["sample_count"]:
        raise ValueError("sample_index lies outside the canonical channel")
    return _sample_edge_to_seconds(
        index,
        int(channel["sample_rate_numerator"]),
        int(channel["sample_rate_denominator"]),
    )


def recording_seconds_to_canonical_sample_index(
    canonical_receipt: object,
    *,
    channel_id: str,
    recording_seconds: float,
    rounding: str = "exact",
) -> int:
    canonical = validate_canonical_signal_receipt(canonical_receipt)
    channel = next(
        (row for row in canonical["channels"] if row["channel_id"] == channel_id),
        None,
    )
    if channel is None:
        raise KeyError(f"unknown canonical channel: {channel_id}")
    if not channel["observed"]:
        raise ValueError("unobserved canonical channel has no physical sample mapping")
    index = _seconds_to_sample_edge(
        recording_seconds,
        int(channel["sample_rate_numerator"]),
        int(channel["sample_rate_denominator"]),
        context="recording_seconds",
        rounding=rounding,
    )
    if index > channel["sample_count"]:
        raise ValueError("recording_seconds lies outside the canonical channel")
    return index


def build_transform_spec(
    *,
    transform_name: str,
    input_unit_ids: Sequence[str],
    output_unit_ids: Sequence[str],
    source_sampling_rate: tuple[int, int],
    output_sampling_rate: tuple[int, int],
    resample_up: int,
    resample_down: int,
    resampler_implementation: str,
    anti_alias_filter: str,
    anti_alias_lowpass_hz: float | None,
    filter_family: str,
    filter_order: int | None,
    highpass_hz: float | None,
    lowpass_hz: float | None,
    phase_policy: str,
    normalization_method: str,
    normalization_source: str,
    clipping_applied: bool,
    clipping_policy: str,
    clipping_source: str,
    reference_type: str,
    reference_matrix: Sequence[Sequence[float]],
    edge_policy: str,
    edge_left_invalid_samples: int,
    edge_right_invalid_samples: int,
    software_versions: Mapping[str, str],
) -> dict[str, Any]:
    """Build a content-addressed transform specification.

    ``input_unit_ids`` refer either to canonical physical channels or to the
    output units of explicitly bound parent views.  The reference matrix rows
    are ordered exactly like ``output_unit_ids`` and columns like
    ``input_unit_ids``.
    """

    input_ids = list(input_unit_ids)
    output_ids = list(output_unit_ids)
    matrix = [[float(value) for value in row] for row in reference_matrix]
    reference_payload = {
        "reference_type": reference_type,
        "input_unit_ids": input_ids,
        "output_unit_ids": output_ids,
        "matrix": matrix,
    }
    body: dict[str, Any] = {
        "schema_version": TRANSFORM_SPEC_SCHEMA_VERSION,
        "transform_name": transform_name,
        "input_unit_ids": input_ids,
        "output_unit_ids": output_ids,
        "source_clock": {
            "sampling_rate_numerator": int(source_sampling_rate[0]),
            "sampling_rate_denominator": int(source_sampling_rate[1]),
        },
        "output_clock": {
            "sampling_rate_numerator": int(output_sampling_rate[0]),
            "sampling_rate_denominator": int(output_sampling_rate[1]),
            "global_origin_recording_seconds": 0.0,
        },
        "resampler": {
            "implementation": resampler_implementation,
            "up": int(resample_up),
            "down": int(resample_down),
            "anti_alias_filter": anti_alias_filter,
            "anti_alias_lowpass_hz": anti_alias_lowpass_hz,
        },
        "filter": {
            "family": filter_family,
            "order": filter_order,
            "highpass_hz": highpass_hz,
            "lowpass_hz": lowpass_hz,
            "phase_policy": phase_policy,
        },
        "normalization": {
            "method": normalization_method,
            "source": normalization_source,
            "preserves_physical_amplitude": (
                normalization_method in {"none", "physical_unit_scale_only"}
                and normalization_source in {"none", "channel_metadata"}
            ),
        },
        "clipping": {
            "applied": clipping_applied,
            "policy": clipping_policy,
            "source": clipping_source,
        },
        "reference": {
            **reference_payload,
            "matrix_sha256": _canonical_sha256(reference_payload),
        },
        "edge_handling": {
            "policy": edge_policy,
            "left_invalid_samples": int(edge_left_invalid_samples),
            "right_invalid_samples": int(edge_right_invalid_samples),
        },
        "software_versions": deepcopy(dict(software_versions)),
        "transform_spec_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["transform_spec_sha256"] = _canonical_sha256(body)
    return validate_transform_spec(body)


def validate_transform_spec(payload: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "transform_name",
        "input_unit_ids",
        "output_unit_ids",
        "source_clock",
        "output_clock",
        "resampler",
        "filter",
        "normalization",
        "clipping",
        "reference",
        "edge_handling",
        "software_versions",
        "transform_spec_sha256",
    }
    data = _strict_dict(payload, required, "signal transform spec")
    if data["schema_version"] != TRANSFORM_SPEC_SCHEMA_VERSION:
        raise ValueError("unsupported signal transform schema")
    _nonempty(data["transform_name"], "transform_name")
    input_ids = _unique_strings(data["input_unit_ids"], "transform input_unit_ids")
    output_ids = _unique_strings(data["output_unit_ids"], "transform output_unit_ids")

    clock_required = {"sampling_rate_numerator", "sampling_rate_denominator"}
    source_clock = _strict_dict(data["source_clock"], clock_required, "source_clock")
    source_num, source_den = _clock_rate(source_clock, "source_clock")
    output_clock = _strict_dict(
        data["output_clock"],
        clock_required | {"global_origin_recording_seconds"},
        "output_clock",
    )
    output_num, output_den = _clock_rate(output_clock, "output_clock")
    if float(output_clock["global_origin_recording_seconds"]) != 0.0:
        raise ValueError("derived views must use the recording-global clock origin")

    resampler = _strict_dict(
        data["resampler"],
        {"implementation", "up", "down", "anti_alias_filter", "anti_alias_lowpass_hz"},
        "resampler",
    )
    _nonempty(resampler["implementation"], "resampler implementation")
    _nonempty(resampler["anti_alias_filter"], "resampler anti_alias_filter")
    up, down = _rate(resampler["up"], resampler["down"], "resampler ratio")
    if source_num * up * output_den != output_num * down * source_den:
        raise ValueError("resampler ratio does not reproduce the output clock")
    output_nyquist = 0.5 * output_num / output_den
    anti_alias = resampler["anti_alias_lowpass_hz"]
    if up != down:
        anti_alias = _finite(anti_alias, "anti_alias_lowpass_hz", minimum=_TOL)
        if anti_alias > output_nyquist + _TOL:
            raise ValueError("anti-alias lowpass exceeds output Nyquist")
    elif anti_alias is not None:
        anti_alias = _finite(anti_alias, "anti_alias_lowpass_hz", minimum=_TOL)

    filter_spec = _strict_dict(
        data["filter"],
        {"family", "order", "highpass_hz", "lowpass_hz", "phase_policy"},
        "filter",
    )
    if filter_spec["family"] not in {
        "none",
        "butterworth",
        "fir",
        "provider_native",
        "other_versioned",
    }:
        raise ValueError("filter family is unsupported")
    if filter_spec["family"] == "none":
        if filter_spec["order"] is not None:
            raise ValueError("filter order must be null when family=none")
    else:
        _integer(filter_spec["order"], "filter order", minimum=1)
    highpass = filter_spec["highpass_hz"]
    lowpass = filter_spec["lowpass_hz"]
    if highpass is not None:
        highpass = _finite(highpass, "filter highpass_hz", minimum=0.0)
    if lowpass is not None:
        lowpass = _finite(lowpass, "filter lowpass_hz", minimum=_TOL)
        if lowpass > output_nyquist + _TOL:
            raise ValueError("filter lowpass exceeds output Nyquist")
    if highpass is not None and lowpass is not None and highpass >= lowpass:
        raise ValueError("filter bandwidth is empty")
    if filter_spec["phase_policy"] not in {
        "none",
        "offline_zero_phase",
        "causal_with_group_delay_receipt",
        "provider_native_versioned",
    }:
        raise ValueError("filter phase policy is unsupported")

    normalization = _strict_dict(
        data["normalization"],
        {"method", "source", "preserves_physical_amplitude"},
        "normalization",
    )
    if normalization["method"] not in {
        "none",
        "physical_unit_scale_only",
        "zscore",
        "robust_scale",
        "provider_native_versioned",
    }:
        raise ValueError("normalization method is unsupported")
    if normalization["source"] not in {
        "none",
        "channel_metadata",
        "recording",
        "tile",
        "event",
        "provider_native",
    }:
        raise ValueError("normalization source is unsupported")
    expected_preservation = normalization["method"] in {
        "none",
        "physical_unit_scale_only",
    } and normalization["source"] in {"none", "channel_metadata"}
    if normalization["preserves_physical_amplitude"] is not expected_preservation:
        raise ValueError("normalization physical-amplitude flag is inconsistent")

    clipping = _strict_dict(
        data["clipping"], {"applied", "policy", "source"}, "clipping"
    )
    if type(clipping["applied"]) is not bool:
        raise TypeError("clipping.applied must be boolean")
    if clipping["applied"]:
        _nonempty(clipping["policy"], "clipping policy")
        if clipping["policy"] == "none" or clipping["source"] not in {
            "recording",
            "tile",
            "event",
            "provider_native",
        }:
            raise ValueError("applied clipping requires a versioned non-none source")
    elif clipping["policy"] != "none" or clipping["source"] != "none":
        raise ValueError("disabled clipping must use policy/source=none")

    reference = _strict_dict(
        data["reference"],
        {
            "reference_type",
            "input_unit_ids",
            "output_unit_ids",
            "matrix",
            "matrix_sha256",
        },
        "reference",
    )
    _nonempty(reference["reference_type"], "reference_type")
    if (
        reference["input_unit_ids"] != input_ids
        or reference["output_unit_ids"] != output_ids
    ):
        raise ValueError("reference unit order must equal transform unit order")
    if not isinstance(reference["matrix"], list) or len(reference["matrix"]) != len(
        output_ids
    ):
        raise ValueError("reference matrix row count is invalid")
    normalized_matrix: list[list[float]] = []
    for row_index, raw_row in enumerate(reference["matrix"]):
        if not isinstance(raw_row, list) or len(raw_row) != len(input_ids):
            raise ValueError("reference matrix column count is invalid")
        row = [
            _finite(value, f"reference.matrix[{row_index}][{column_index}]")
            for column_index, value in enumerate(raw_row)
        ]
        if not any(abs(value) > _TOL for value in row):
            raise ValueError("reference matrix contains an all-zero output row")
        normalized_matrix.append(row)
    reference_payload = {
        "reference_type": reference["reference_type"],
        "input_unit_ids": input_ids,
        "output_unit_ids": output_ids,
        "matrix": normalized_matrix,
    }
    _sha256(reference["matrix_sha256"], "reference matrix_sha256")
    if reference["matrix_sha256"] != _canonical_sha256(reference_payload):
        raise ValueError("reference matrix hash does not bind its content")

    edge = _strict_dict(
        data["edge_handling"],
        {"policy", "left_invalid_samples", "right_invalid_samples"},
        "edge_handling",
    )
    if edge["policy"] not in {
        "none",
        "global_recording_edges",
        "interval_local_edges",
    }:
        raise ValueError("edge handling policy is unsupported")
    left_edge = _integer(edge["left_invalid_samples"], "left_invalid_samples")
    right_edge = _integer(edge["right_invalid_samples"], "right_invalid_samples")
    if edge["policy"] == "none" and (left_edge or right_edge):
        raise ValueError("edge samples require a non-none edge policy")

    if filter_spec["phase_policy"] == "causal_with_group_delay_receipt":
        # The onset path deliberately accepts only a linear-phase FIR whose
        # constant delay is recoverable from its even order.  A centered
        # resampler, bidirectional IIR, or an unspecified provider filter
        # would silently move activity across an onset boundary.
        order = int(filter_spec["order"] or 0)
        if filter_spec["family"] != "fir" or order < 2 or order % 2:
            raise ValueError(
                "causal onset filters require a positive even-order linear-phase FIR"
            )
        if (
            up != 1
            or down != 1
            or resampler["implementation"] != "none"
            or resampler["anti_alias_filter"] != "none"
        ):
            raise ValueError("causal onset transforms cannot use a centered resampler")
        if edge["policy"] != "global_recording_edges":
            raise ValueError("causal onset warm-up must be bound to the recording edge")
        if left_edge < order or right_edge != 0:
            raise ValueError(
                "causal onset edge receipt must mask FIR warm-up and no future edge"
            )

    if type(data["software_versions"]) is not dict or not data["software_versions"]:
        raise TypeError("software_versions must be a non-empty object")
    for key, value in data["software_versions"].items():
        _nonempty(key, "software_versions key")
        _nonempty(value, f"software_versions[{key!r}]")

    _sha256(data["transform_spec_sha256"], "transform_spec_sha256")
    digest_source = deepcopy(data)
    digest_source["transform_spec_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["transform_spec_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("transform spec hash does not bind its content")
    return data


def _family_rows(unit: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = unit["evidence_eligibility"]
    return {str(row["family"]): row for row in rows}


def _canonical_source_catalog(
    canonical: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for channel in canonical["channels"]:
        numerator = int(channel["sample_rate_numerator"])
        denominator = int(channel["sample_rate_denominator"])
        nyquist = 0.5 * numerator / denominator
        lower = float(channel["acquisition_highpass_hz"] or 0.0)
        upper = min(float(channel["acquisition_lowpass_hz"] or nyquist), nyquist)
        observed = bool(channel["observed"])
        catalog[str(channel["channel_id"])] = {
            "unit_id": str(channel["channel_id"]),
            "clock": (numerator, denominator),
            "observed": observed,
            "imputed": False,
            "canonical_source_channel_ids": [str(channel["channel_id"])],
            "effective_bandwidth_hz": [lower, upper],
            "family_eligibility": {family: observed for family in EVIDENCE_FAMILIES},
            "amplitude_preserved": observed,
        }
    return catalog


def _basic_view_hash_check(
    view: object,
    canonical: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    if type(view) is not dict:
        raise TypeError(f"{context} must be an object")
    data = deepcopy(view)
    if data.get("schema_version") != SIGNAL_VIEW_SCHEMA_VERSION:
        raise ValueError(f"{context} has an unsupported schema")
    if data.get("canonical_signal_id") != canonical["canonical_signal_id"]:
        raise ValueError(f"{context} belongs to a different canonical signal")
    if data.get("canonical_receipt_sha256") != canonical["receipt_sha256"]:
        raise ValueError(f"{context} does not bind the canonical receipt")
    _sha256(data.get("receipt_sha256"), f"{context}.receipt_sha256")
    digest_source = deepcopy(data)
    digest_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest_source):
        raise ValueError(f"{context} receipt hash does not bind its content")
    return data


def _source_catalog(
    canonical: Mapping[str, Any],
    parent_bindings: Sequence[Mapping[str, Any]],
    trusted_parent_views: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not parent_bindings:
        return _canonical_source_catalog(canonical), []
    catalog: dict[str, dict[str, Any]] = {}
    normalized_parents: list[dict[str, Any]] = []
    for index, raw_binding in enumerate(parent_bindings):
        binding = _strict_dict(
            raw_binding,
            {
                "view_id",
                "view_receipt_id",
                "receipt_sha256",
                "cache_namespace_sha256",
            },
            f"parent_view_bindings[{index}]",
        )
        parent_id = _nonempty(
            binding["view_id"], f"parent_view_bindings[{index}].view_id"
        )
        if parent_id not in trusted_parent_views:
            raise ValueError(
                f"parent view {parent_id!r} is not host-supplied and trusted"
            )
        parent = _basic_view_hash_check(
            trusted_parent_views[parent_id],
            canonical,
            context=f"trusted parent view {parent_id!r}",
        )
        expected = {
            "view_id": parent["view_id"],
            "view_receipt_id": parent["view_receipt_id"],
            "receipt_sha256": parent["receipt_sha256"],
            "cache_namespace_sha256": parent["cache"]["cache_namespace_sha256"],
        }
        if binding != expected:
            raise ValueError(f"parent view binding for {parent_id!r} drifted")
        transform = validate_transform_spec(parent["transform_spec"])
        clock = _clock_rate(
            transform["output_clock"], f"parent {parent_id} output_clock"
        )
        for unit in parent["output_units"]:
            unit_id = str(unit["unit_id"])
            if unit_id in catalog:
                raise ValueError("parent views expose duplicate output unit IDs")
            families = _family_rows(unit)
            catalog[unit_id] = {
                "unit_id": unit_id,
                "clock": clock,
                "observed": bool(unit["observed"]),
                "imputed": bool(unit["imputed"]),
                "canonical_source_channel_ids": list(
                    unit["canonical_source_channel_ids"]
                ),
                "effective_bandwidth_hz": list(unit["effective_bandwidth_hz"]),
                "family_eligibility": {
                    family: bool(families[family]["eligible"])
                    for family in EVIDENCE_FAMILIES
                },
                "amplitude_preserved": bool(families["amplitude"]["eligible"]),
                "parent_view": parent,
            }
        normalized_parents.append(parent)
    if len({item["view_id"] for item in normalized_parents}) != len(normalized_parents):
        raise ValueError("parent view bindings contain duplicate view IDs")
    return catalog, normalized_parents


def _source_bandwidth(
    sources: Sequence[Mapping[str, Any]],
    transform: Mapping[str, Any],
) -> tuple[float, float]:
    lower = max(float(source["effective_bandwidth_hz"][0]) for source in sources)
    upper = min(float(source["effective_bandwidth_hz"][1]) for source in sources)
    highpass = transform["filter"]["highpass_hz"]
    lowpass = transform["filter"]["lowpass_hz"]
    anti_alias = transform["resampler"]["anti_alias_lowpass_hz"]
    output_num, output_den = _clock_rate(transform["output_clock"], "output_clock")
    upper = min(upper, 0.5 * output_num / output_den)
    if highpass is not None:
        lower = max(lower, float(highpass))
    if lowpass is not None:
        upper = min(upper, float(lowpass))
    if anti_alias is not None:
        upper = min(upper, float(anti_alias))
    if upper <= lower + _TOL:
        raise ValueError("derived unit has no usable effective bandwidth")
    return lower, upper


def _eligibility(
    *,
    task_role: str,
    observed: bool,
    imputed: bool,
    physical_unit: str,
    sources: Sequence[Mapping[str, Any]],
    bandwidth: tuple[float, float],
    transform: Mapping[str, Any],
    high_frequency_qualification_sha256: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase_safe = transform["filter"]["phase_policy"] in {
        "none",
        "offline_zero_phase",
        "causal_with_group_delay_receipt",
    }
    amplitude_preserved = (
        bool(transform["normalization"]["preserves_physical_amplitude"])
        and not bool(transform["clipping"]["applied"])
        and physical_unit == "V"
        and all(bool(source["amplitude_preserved"]) for source in sources)
    )
    for family in EVIDENCE_FAMILIES:
        reasons: list[str] = []
        if task_role in {"detector_provider", "detector_native"}:
            # Detector-native tensors and their posterior are consumed only by
            # the independent detection/boundary contracts.  Even a provider
            # transform that happens to preserve volts or bandwidth cannot be
            # promoted into clinical Findings evidence.
            reasons.append(DETECTOR_PROVIDER_EVIDENCE_REASON_CODE)
        if not observed:
            reasons.append("unit_unobserved")
        if imputed:
            reasons.append("unit_imputed")
        if any(not source["observed"] or source["imputed"] for source in sources):
            reasons.append("source_unobserved_or_imputed")
        if any(not source["family_eligibility"][family] for source in sources):
            reasons.append("source_family_ineligible")
        if family == "amplitude" and not amplitude_preserved:
            reasons.append("physical_amplitude_not_preserved")
        if family == "morphology" and not phase_safe:
            reasons.append("phase_response_not_morphology_qualified")
        if family == "high_frequency":
            if high_frequency_qualification_sha256 is None:
                reasons.append("high_frequency_qualification_absent")
            if bandwidth[1] < HIGH_FREQUENCY_MIN_HZ - _TOL:
                reasons.append("high_frequency_bandwidth_insufficient")
        reasons = sorted(set(reasons))
        rows.append(
            {
                "family": family,
                "eligible": not reasons,
                "effective_bandwidth_hz": [bandwidth[0], bandwidth[1]],
                "reason_codes": reasons,
            }
        )
    return rows


def _total_output_samples(
    canonical: Mapping[str, Any], transform: Mapping[str, Any]
) -> int:
    numerator, denominator = _clock_rate(transform["output_clock"], "output_clock")
    return _seconds_to_sample_edge(
        canonical["recording_duration_seconds"],
        numerator,
        denominator,
        context="recording duration on output clock",
    )


def _expected_padding_intervals(
    *,
    data_samples: int,
    left_padding: int,
    right_padding: int,
) -> list[list[int]]:
    result: list[list[int]] = []
    if left_padding:
        result.append([0, left_padding])
    valid_stop = left_padding + data_samples
    if right_padding:
        result.append([valid_stop, valid_stop + right_padding])
    return result


def _expected_edge_intervals(
    *,
    selected: tuple[int, int],
    total_output_samples: int,
    left_padding: int,
    transform: Mapping[str, Any],
) -> list[list[int]]:
    policy = transform["edge_handling"]["policy"]
    left = int(transform["edge_handling"]["left_invalid_samples"])
    right = int(transform["edge_handling"]["right_invalid_samples"])
    data_samples = selected[1] - selected[0]
    if left + right >= data_samples and (left or right):
        raise ValueError("edge-invalid samples consume the entire selected interval")
    mark_left = policy == "interval_local_edges" or (
        policy == "global_recording_edges" and selected[0] == 0
    )
    mark_right = policy == "interval_local_edges" or (
        policy == "global_recording_edges" and selected[1] == total_output_samples
    )
    result: list[list[int]] = []
    if mark_left and left:
        result.append([left_padding, left_padding + left])
    valid_stop = left_padding + data_samples
    if mark_right and right:
        result.append([valid_stop - right, valid_stop])
    return result


def _canonical_quality_masks(
    canonical: Mapping[str, Any],
    output_units: Sequence[Mapping[str, Any]],
    *,
    task_role: str,
    selected: tuple[int, int],
    left_padding: int,
    transform: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project canonical QC through the actual temporal transform support.

    A raw bad interval is not, in general, the bad interval in a filtered
    tensor.  For the onset-authorized one-sided FIR the exact dependency is
    finite: a contaminated raw sample can affect the current output and the
    following ``filter_order`` outputs.  A zero-phase IIR (and a centered
    resampler whose finite kernel is not receipted) has no exact finite
    support in the current contract.  Such a view therefore fails closed to
    the complete selected interval instead of pretending that the raw QC
    interval or an arbitrary guard is sufficient.

    Detector-native/provider roles deliberately retain their provider
    contract here; this hardening applies only to canonical clinical task
    views.  A derived spatial-reference view receives the parent's already
    projected masks separately in :func:`_inherited_parent_quality_masks`.
    """

    numerator, denominator = _clock_rate(transform["output_clock"], "output_clock")
    phase = str(transform["filter"]["phase_policy"])
    family = str(transform["filter"]["family"])
    resample_up = int(transform["resampler"]["up"])
    resample_down = int(transform["resampler"]["down"])
    support_policy = "instantaneous_no_temporal_dilation_v1"
    support_left = 0
    support_right = 0
    full_selected_fail_closed = False
    if task_role == "onset_causal":
        if (
            phase != "causal_with_group_delay_receipt"
            or family != "fir"
            or resample_up != 1
            or resample_down != 1
        ):
            raise ValueError(
                "onset_causal QC support requires an exact non-resampled causal FIR"
            )
        support_policy = "exact_causal_fir_raw_to_output_support_v1"
        support_right = int(transform["filter"]["order"] or 0)
    elif task_role == "context_offline" and (
        phase == "offline_zero_phase" or resample_up != resample_down
    ):
        support_policy = "full_selected_fail_closed_nonfinite_or_unreceipted_support_v1"
        full_selected_fail_closed = True

    rows: list[dict[str, Any]] = []
    for unit in output_units:
        sources = set(str(item) for item in unit["canonical_source_channel_ids"])
        for primitive in canonical["quality_primitives"]:
            if sources.isdisjoint(str(item) for item in primitive["channel_ids"]):
                continue
            start_global = _seconds_to_sample_edge(
                primitive["start_recording_seconds"],
                numerator,
                denominator,
                context=f"quality {primitive['quality_id']} start on output clock",
                rounding="floor",
            )
            stop_global = _seconds_to_sample_edge(
                primitive["stop_recording_seconds"],
                numerator,
                denominator,
                context=f"quality {primitive['quality_id']} stop on output clock",
                rounding="ceil",
            )
            if full_selected_fail_closed:
                projected = selected
            else:
                projected = (
                    max(0, start_global - support_left),
                    stop_global + support_right,
                )
            overlap = (
                max(projected[0], selected[0]),
                min(projected[1], selected[1]),
            )
            if overlap[1] <= overlap[0]:
                continue
            reason_codes = [
                f"canonical_quality:{primitive['kind']}",
                f"quality_id:{primitive['quality_id']}",
            ]
            if task_role in {"onset_causal", "context_offline"}:
                reason_codes.extend(
                    [
                        f"quality_support_policy:{support_policy}",
                        f"quality_support_task_role:{task_role}",
                        (
                            "quality_support_transform_sha256:"
                            f"{transform['transform_spec_sha256']}"
                        ),
                        f"quality_support_left_output_samples:{support_left}",
                        f"quality_support_right_output_samples:{support_right}",
                    ]
                )
            if len(sources) > 1:
                reason_codes.append(
                    "derived_reference_nonzero_carrier_contamination_v1"
                )
            rows.append(
                {
                    "unit_id": unit["unit_id"],
                    "tensor_sample_interval": [
                        left_padding + overlap[0] - selected[0],
                        left_padding + overlap[1] - selected[0],
                    ],
                    "severity": primitive["severity"],
                    "disabled_evidence_families": list(
                        primitive["disabled_evidence_families"]
                    ),
                    "reason_codes": reason_codes,
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["unit_id"]),
            int(row["tensor_sample_interval"][0]),
            int(row["tensor_sample_interval"][1]),
            tuple(row["disabled_evidence_families"]),
        )
    )
    return rows


def _inherited_parent_quality_masks(
    output_units: Sequence[Mapping[str, Any]],
    *,
    task_role: str,
    selected: tuple[int, int],
    left_padding: int,
    transform: Mapping[str, Any],
    source_catalog: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Carry parent QC support into reference children without shrinking it.

    The canonical quality primitive alone is insufficient for a child of a
    filtered parent: recomputing the primitive through the child's
    instantaneous reference matrix would discard the parent's FIR/IIR
    support dilation.  This projection maps every non-zero parent carrier's
    mask through recording-relative time and binds the parent receipt and
    carrier identity in the child reason codes.
    """

    if task_role != "spatial_reference":
        return []
    output_num, output_den = _clock_rate(transform["output_clock"], "output_clock")
    instantaneous = (
        transform["filter"]["family"] == "none"
        and transform["filter"]["phase_policy"] == "none"
        and int(transform["resampler"]["up"]) == 1
        and int(transform["resampler"]["down"]) == 1
    )
    rows: list[dict[str, Any]] = []
    for output_unit in output_units:
        output_unit_id = str(output_unit["unit_id"])
        for source_unit_id in output_unit["source_unit_ids"]:
            source = source_catalog[str(source_unit_id)]
            parent = source.get("parent_view")
            if parent is None:
                continue
            parent_transform = validate_transform_spec(parent["transform_spec"])
            parent_num, parent_den = _clock_rate(
                parent_transform["output_clock"], "parent output_clock"
            )
            parent_selected_start = int(
                parent["coordinates"]["selected_global_output_sample_interval"][0]
            )
            parent_valid_start = int(
                parent["tensor_layout"]["valid_data_tensor_sample_interval"][0]
            )
            for parent_mask in parent["masks"]["quality_invalid_intervals"]:
                if str(parent_mask["unit_id"]) != str(source_unit_id):
                    continue
                parent_interval = parent_mask["tensor_sample_interval"]
                parent_global_start = (
                    parent_selected_start + int(parent_interval[0]) - parent_valid_start
                )
                parent_global_stop = (
                    parent_selected_start + int(parent_interval[1]) - parent_valid_start
                )
                start_seconds = _sample_edge_to_seconds(
                    parent_global_start, parent_num, parent_den
                )
                stop_seconds = _sample_edge_to_seconds(
                    parent_global_stop, parent_num, parent_den
                )
                if instantaneous:
                    child_global_start = _seconds_to_sample_edge(
                        start_seconds,
                        output_num,
                        output_den,
                        context="parent quality start on child clock",
                        rounding="floor",
                    )
                    child_global_stop = _seconds_to_sample_edge(
                        stop_seconds,
                        output_num,
                        output_den,
                        context="parent quality stop on child clock",
                        rounding="ceil",
                    )
                    overlap = (
                        max(child_global_start, selected[0]),
                        min(child_global_stop, selected[1]),
                    )
                    inherited_policy = "exact_instantaneous_reference_parent_support_v1"
                else:
                    # The current schema has no general finite-kernel support
                    # ledger for a filtered/resampled child.  Never shrink a
                    # trusted parent mask based on an unreceipted approximation.
                    overlap = selected
                    inherited_policy = (
                        "full_selected_fail_closed_child_support_unreceipted_v1"
                    )
                if overlap[1] <= overlap[0]:
                    continue
                reason_codes = list(parent_mask["reason_codes"])
                reason_codes.extend(
                    [
                        "parent_quality_mask_propagated_v1",
                        f"parent_view_id:{parent['view_id']}",
                        f"parent_view_receipt_sha256:{parent['receipt_sha256']}",
                        f"parent_unit_id:{source_unit_id}",
                        f"child_quality_support_policy:{inherited_policy}",
                    ]
                )
                if len(output_unit["source_unit_ids"]) > 1:
                    reason_codes.append(
                        "derived_reference_nonzero_carrier_contamination_v1"
                    )
                rows.append(
                    {
                        "unit_id": output_unit_id,
                        "tensor_sample_interval": [
                            left_padding + overlap[0] - selected[0],
                            left_padding + overlap[1] - selected[0],
                        ],
                        "severity": parent_mask["severity"],
                        "disabled_evidence_families": list(
                            parent_mask["disabled_evidence_families"]
                        ),
                        "reason_codes": sorted(set(reason_codes)),
                    }
                )
    rows.sort(
        key=lambda row: (
            str(row["unit_id"]),
            int(row["tensor_sample_interval"][0]),
            int(row["tensor_sample_interval"][1]),
            tuple(row["disabled_evidence_families"]),
        )
    )
    return rows


def _cache_is_expansion_safe(transform: Mapping[str, Any]) -> bool:
    return (
        transform["normalization"]["source"] != "event"
        and transform["clipping"]["source"] != "event"
        and transform["edge_handling"]["policy"] != "interval_local_edges"
    )


def _cache_namespace(
    *,
    canonical: Mapping[str, Any],
    transform: Mapping[str, Any],
    parent_bindings: Sequence[Mapping[str, Any]],
    selected: tuple[int, int],
    expansion_safe: bool,
) -> str:
    material: dict[str, Any] = {
        "canonical_signal_id": canonical["canonical_signal_id"],
        "canonical_receipt_sha256": canonical["receipt_sha256"],
        "transform_spec_sha256": transform["transform_spec_sha256"],
        "parent_cache_namespace_sha256s": sorted(
            str(item["cache_namespace_sha256"]) for item in parent_bindings
        ),
    }
    if not expansion_safe:
        material["interval_bound_global_output_sample_interval"] = list(selected)
    return _canonical_sha256(material)


def _temporal_contract_fields() -> set[str]:
    return {
        "schema_version",
        "dependency_policy",
        "future_sample_access",
        "onset_evidence_authorized",
        "warm_up_samples",
        "warm_up_recording_seconds",
        "group_delay_samples",
        "group_delay_recording_seconds",
        "delay_correction_policy",
        "latest_raw_support_offset_samples",
        "raw_support_end_policy",
        "authorization_reason_codes",
    }


def _trusted_parent_temporal_contract(
    parent: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    contract = _strict_dict(
        parent.get("temporal_evidence"),
        _temporal_contract_fields(),
        f"{context}.temporal_evidence",
    )
    if contract["schema_version"] != TEMPORAL_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(f"{context} has an unsupported temporal-evidence schema")
    if contract["dependency_policy"] not in {
        "instantaneous",
        "past_and_present_only",
        "bidirectional_or_unknown",
    }:
        raise ValueError(f"{context} has an unsupported dependency policy")
    if type(contract["future_sample_access"]) is not bool:
        raise TypeError(f"{context}.future_sample_access must be boolean")
    if type(contract["onset_evidence_authorized"]) is not bool:
        raise TypeError(f"{context}.onset_evidence_authorized must be boolean")
    _integer(contract["warm_up_samples"], f"{context}.warm_up_samples")
    _finite(
        contract["warm_up_recording_seconds"],
        f"{context}.warm_up_recording_seconds",
        minimum=0.0,
    )
    _finite(
        contract["group_delay_samples"], f"{context}.group_delay_samples", minimum=0.0
    )
    _finite(
        contract["group_delay_recording_seconds"],
        f"{context}.group_delay_recording_seconds",
        minimum=0.0,
    )
    if contract["delay_correction_policy"] not in {
        "none",
        "report_constant_processing_latency_no_timestamp_advance_v1",
    }:
        raise ValueError(f"{context} has an unsupported delay-correction policy")
    raw_support_offset = contract["latest_raw_support_offset_samples"]
    if raw_support_offset is not None:
        if isinstance(raw_support_offset, bool) or not isinstance(
            raw_support_offset, int
        ):
            raise TypeError(
                f"{context}.latest_raw_support_offset_samples must be integer or null"
            )
        if raw_support_offset > 0:
            raise ValueError("onset evidence cannot depend on a future raw sample")
    if contract["raw_support_end_policy"] not in {
        "at_or_before_unshifted_evidence_sample_v1",
        "future_dependent_context_not_onset_eligible_v1",
    }:
        raise ValueError(f"{context} has an unsupported raw-support policy")
    _unique_strings(
        contract["authorization_reason_codes"],
        f"{context}.authorization_reason_codes",
        nonempty=not bool(contract["onset_evidence_authorized"]),
    )
    if contract["onset_evidence_authorized"] and contract["authorization_reason_codes"]:
        raise ValueError("authorized onset evidence cannot carry denial reason codes")
    if contract["future_sample_access"] and contract["onset_evidence_authorized"]:
        raise ValueError("future-dependent views cannot authorize onset evidence")
    return contract


def _view_temporal_evidence(
    *,
    task_role: str,
    transform: Mapping[str, Any],
    parent_views: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive the effective time-direction contract for one signal view.

    This is deliberately computed from the content-addressed transform and
    trusted parent receipts; callers cannot promote an offline view into an
    onset carrier by setting a boolean in report-generation code.
    """

    phase = str(transform["filter"]["phase_policy"])
    filter_family = str(transform["filter"]["family"])
    order = int(transform["filter"]["order"] or 0)
    output_num, output_den = _clock_rate(transform["output_clock"], "output_clock")
    output_rate_hz = output_num / output_den
    edge_left = int(transform["edge_handling"]["left_invalid_samples"])

    parent_contracts = [
        _trusted_parent_temporal_contract(
            parent,
            context=f"parent view {parent.get('view_id', '<unknown>')!r}",
        )
        for parent in parent_views
    ]
    parent_future = any(bool(item["future_sample_access"]) for item in parent_contracts)
    parent_warm_seconds = max(
        (float(item["warm_up_recording_seconds"]) for item in parent_contracts),
        default=0.0,
    )
    parent_delay_seconds = max(
        (float(item["group_delay_recording_seconds"]) for item in parent_contracts),
        default=0.0,
    )
    parent_onset_authorized = bool(parent_contracts) and all(
        bool(item["onset_evidence_authorized"]) for item in parent_contracts
    )
    response_authorization = transform["software_versions"].get(
        ONSET_FIR_RESPONSE_AUTHORIZATION_SOFTWARE_KEY
    )
    if response_authorization not in {None, "true", "false"}:
        raise ValueError(
            "causal FIR response authorization must be encoded as true or false"
        )
    direct_response_authorized = response_authorization != "false"
    clinical_admission_authorization = transform["software_versions"].get(
        ONSET_FIR_CLINICAL_ADMISSION_AUTHORIZATION_SOFTWARE_KEY
    )
    if clinical_admission_authorization not in {None, "true", "false"}:
        raise ValueError(
            "causal FIR clinical admission authorization must be encoded as "
            "true or false"
        )
    # The canonical EDF transform must never regain clinical permission by
    # omitting the new marker.  Generic noncanonical causal views retain the
    # pre-extension behavior; archived canonical 101-tap receipts remain
    # auditable because their independent response marker is already false.
    if transform["transform_name"] == CANONICAL_EDF_ONSET_TRANSFORM_NAME:
        direct_clinical_admission_authorized = (
            clinical_admission_authorization == "true"
        )
    else:
        direct_clinical_admission_authorized = (
            clinical_admission_authorization != "false"
        )
    inherited_response_unqualified = any(
        ONSET_FIR_RESPONSE_UNQUALIFIED_REASON_CODE in item["authorization_reason_codes"]
        for item in parent_contracts
    )
    inherited_clinical_admission_unqualified = any(
        ONSET_FIR_CLINICAL_ADMISSION_UNQUALIFIED_REASON_CODE
        in item["authorization_reason_codes"]
        for item in parent_contracts
    )

    if phase == "causal_with_group_delay_receipt":
        current_future = False
        current_warm_seconds = edge_left / output_rate_hz
        current_delay_seconds = (order / 2.0) / output_rate_hz
        current_dependency = "past_and_present_only"
    elif phase == "offline_zero_phase":
        current_future = True
        current_warm_seconds = edge_left / output_rate_hz
        current_delay_seconds = 0.0
        current_dependency = "bidirectional_or_unknown"
    elif phase == "provider_native_versioned":
        current_future = True
        current_warm_seconds = edge_left / output_rate_hz
        current_delay_seconds = 0.0
        current_dependency = "bidirectional_or_unknown"
    else:
        current_future = False
        # A pure reference transform copies the parent's carrier mask; its
        # edge receipt does not represent a second warm-up period.
        current_warm_seconds = 0.0 if parent_contracts else edge_left / output_rate_hz
        current_delay_seconds = 0.0
        current_dependency = "instantaneous"

    future = parent_future or current_future
    warm_seconds = max(parent_warm_seconds, current_warm_seconds)
    delay_seconds = parent_delay_seconds + current_delay_seconds
    warm_position = warm_seconds * output_rate_hz
    warm_samples = int(round(warm_position))
    if abs(warm_position - warm_samples) > _TOL:
        raise ValueError("temporal warm-up is not aligned to the output clock")
    delay_samples = delay_seconds * output_rate_hz

    direct_onset = (
        not parent_contracts
        and task_role == "onset_causal"
        and phase == "causal_with_group_delay_receipt"
        and direct_response_authorized
        and direct_clinical_admission_authorized
    )
    inherited_onset = (
        bool(parent_contracts)
        and parent_onset_authorized
        and task_role == "spatial_reference"
        and phase == "none"
        and filter_family == "none"
        and transform["resampler"]["up"] == 1
        and transform["resampler"]["down"] == 1
    )
    onset_authorized = (direct_onset or inherited_onset) and not future

    reasons: list[str] = []
    if not onset_authorized:
        if future:
            reasons.append("future_sample_access")
        if (
            task_role == "onset_causal" and response_authorization == "false"
        ) or inherited_response_unqualified:
            reasons.append(ONSET_FIR_RESPONSE_UNQUALIFIED_REASON_CODE)
        elif (
            task_role == "onset_causal" and not direct_clinical_admission_authorized
        ) or inherited_clinical_admission_unqualified:
            reasons.append(ONSET_FIR_CLINICAL_ADMISSION_UNQUALIFIED_REASON_CODE)
        elif task_role == "context_offline":
            reasons.append("offline_context_view_not_onset_authorized")
        elif task_role == "findings_native_morphology":
            reasons.append("native_morphology_view_not_onset_authorized")
        elif task_role in {"detector_provider", "detector_native"}:
            reasons.append("detector_provider_view_not_onset_authorized")
        elif parent_contracts and not parent_onset_authorized:
            reasons.append("parent_view_not_onset_authorized")
        else:
            reasons.append("task_role_not_onset_authorized")

    if future:
        dependency = "bidirectional_or_unknown"
    elif current_dependency == "past_and_present_only" or any(
        item["dependency_policy"] == "past_and_present_only"
        for item in parent_contracts
    ):
        dependency = "past_and_present_only"
    else:
        dependency = "instantaneous"

    correction = (
        "report_constant_processing_latency_no_timestamp_advance_v1"
        if delay_seconds > _TOL and not future
        else "none"
    )
    return {
        "schema_version": TEMPORAL_EVIDENCE_SCHEMA_VERSION,
        "dependency_policy": dependency,
        "future_sample_access": future,
        "onset_evidence_authorized": onset_authorized,
        "warm_up_samples": warm_samples,
        "warm_up_recording_seconds": warm_seconds,
        "group_delay_samples": delay_samples,
        "group_delay_recording_seconds": delay_seconds,
        "delay_correction_policy": correction,
        "latest_raw_support_offset_samples": None if future else 0,
        "raw_support_end_policy": (
            "future_dependent_context_not_onset_eligible_v1"
            if future
            else "at_or_before_unshifted_evidence_sample_v1"
        ),
        "authorization_reason_codes": sorted(set(reasons)),
    }


def _enforce_task_view_contract(
    *,
    task_role: str,
    transform: Mapping[str, Any],
    parent_views: Sequence[Mapping[str, Any]],
    temporal: Mapping[str, Any],
) -> None:
    phase = transform["filter"]["phase_policy"]
    if (
        task_role
        in {
            "findings_native_morphology",
            "onset_causal",
            "context_offline",
        }
        and parent_views
    ):
        raise ValueError(
            f"{task_role} must be derived directly from the canonical mother signal"
        )
    if task_role == "findings_native_morphology":
        if (
            transform["filter"]["family"] != "none"
            or phase != "none"
            or transform["resampler"]["up"] != 1
            or transform["resampler"]["down"] != 1
            or transform["normalization"]["method"]
            not in {"none", "physical_unit_scale_only"}
            or transform["clipping"]["applied"]
        ):
            raise ValueError(
                "native morphology must preserve native samples and physical amplitude"
            )
        if temporal["onset_evidence_authorized"]:
            raise ValueError("native morphology cannot authorize onset timing")
    elif task_role == "onset_causal":
        if phase != "causal_with_group_delay_receipt":
            raise ValueError(
                "onset_causal requires a one-sided delay-receipted transform"
            )
        if temporal["future_sample_access"]:
            raise ValueError("onset_causal must be future-free")
        allowed_denials = {
            (ONSET_FIR_RESPONSE_UNQUALIFIED_REASON_CODE,),
            (ONSET_FIR_CLINICAL_ADMISSION_UNQUALIFIED_REASON_CODE,),
        }
        if (
            not temporal["onset_evidence_authorized"]
            and tuple(temporal["authorization_reason_codes"]) not in allowed_denials
        ):
            raise ValueError(
                "onset_causal may lose authorization only after a bound FIR "
                "response or clinical-admission qualification failure"
            )
        if temporal["delay_correction_policy"] != (
            "report_constant_processing_latency_no_timestamp_advance_v1"
        ):
            raise ValueError(
                "onset_causal must report latency without advancing timestamps"
            )
        if (
            temporal["latest_raw_support_offset_samples"] is None
            or temporal["latest_raw_support_offset_samples"] > 0
        ):
            raise ValueError("onset_causal raw support extends after evidence time")
    elif task_role == "context_offline":
        if phase != "offline_zero_phase":
            raise ValueError(
                "context_offline requires an explicitly bidirectional transform"
            )
        if (
            not temporal["future_sample_access"]
            or temporal["onset_evidence_authorized"]
        ):
            raise ValueError(
                "context_offline can describe evolution but cannot authorize onset"
            )
    elif task_role == "spatial_reference" and temporal["onset_evidence_authorized"]:
        if (
            len(parent_views) != 1
            or not parent_views[0]["temporal_evidence"]["onset_evidence_authorized"]
        ):
            raise ValueError("a spatial view cannot create onset authorization")


def _view_digest_id(body: Mapping[str, Any]) -> str:
    digest_source = deepcopy(body)
    digest_source["view_receipt_id"] = "CONTENT-ADDRESS-PENDING"
    digest_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    return "VIEWREC-" + _canonical_sha256(digest_source)[:24]


def _derive_output_units(
    definitions: Sequence[Mapping[str, Any]],
    *,
    task_role: str,
    transform: Mapping[str, Any],
    source_catalog: Mapping[str, Mapping[str, Any]],
    high_frequency_qualification_sha256: str | None,
) -> list[dict[str, Any]]:
    input_ids = list(transform["input_unit_ids"])
    amplitude_preserved = (
        transform["normalization"]["preserves_physical_amplitude"]
        and not transform["clipping"]["applied"]
    )
    output_units: list[dict[str, Any]] = []
    matrix = transform["reference"]["matrix"]
    for definition, row in zip(definitions, matrix):
        source_ids = [
            input_id
            for input_id, coefficient in zip(input_ids, row)
            if abs(float(coefficient)) > _TOL
        ]
        sources = [source_catalog[item] for item in source_ids]
        if definition["observed"] and any(
            not source["observed"] or source["imputed"] for source in sources
        ):
            raise ValueError("observed output unit depends on unobserved/imputed input")
        if amplitude_preserved and definition["physical_unit"] != "V":
            raise ValueError(
                "physical-amplitude-preserving view outputs must use volts"
            )
        if not amplitude_preserved and definition["physical_unit"] != "dimensionless":
            raise ValueError("normalized/clipped view outputs must be dimensionless")
        canonical_sources = sorted(
            {
                channel_id
                for source in sources
                for channel_id in source["canonical_source_channel_ids"]
            }
        )
        bandwidth = _source_bandwidth(sources, transform)
        eligibility = _eligibility(
            task_role=task_role,
            observed=bool(definition["observed"]),
            imputed=bool(definition["imputed"]),
            physical_unit=str(definition["physical_unit"]),
            sources=sources,
            bandwidth=bandwidth,
            transform=transform,
            high_frequency_qualification_sha256=high_frequency_qualification_sha256,
        )
        output_units.append(
            {
                **definition,
                "source_unit_ids": source_ids,
                "canonical_source_channel_ids": canonical_sources,
                "effective_bandwidth_hz": [bandwidth[0], bandwidth[1]],
                "evidence_eligible": bool(definition["observed"])
                and not bool(definition["imputed"])
                and task_role not in {"detector_provider", "detector_native"},
                "evidence_eligibility": eligibility,
            }
        )
    return output_units


def build_signal_view_receipt(
    canonical_receipt: object,
    *,
    view_id: str,
    task_role: str,
    transform_spec: object,
    output_unit_definitions: Sequence[Mapping[str, object]],
    selected_global_output_sample_interval: tuple[int, int],
    processed_view_sha256: str,
    padding_left_samples: int = 0,
    padding_right_samples: int = 0,
    additional_quality_masks: Sequence[Mapping[str, object]] = (),
    cache_tile_size_samples: int,
    cache_tiles: Sequence[Mapping[str, object]],
    parent_views: Sequence[Mapping[str, object]] = (),
    high_frequency_qualification_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one task-specific view receipt without touching signal samples."""

    canonical = validate_canonical_signal_receipt(canonical_receipt)
    transform = validate_transform_spec(transform_spec)
    parent_mapping = {
        _nonempty(item.get("view_id"), "parent view_id"): deepcopy(dict(item))
        for item in parent_views
    }
    if len(parent_mapping) != len(parent_views):
        raise ValueError("parent_views contain duplicate view IDs")
    parent_bindings = [
        {
            "view_id": parent["view_id"],
            "view_receipt_id": parent["view_receipt_id"],
            "receipt_sha256": parent["receipt_sha256"],
            "cache_namespace_sha256": parent["cache"]["cache_namespace_sha256"],
        }
        for parent in parent_views
    ]
    source_catalog, normalized_parents = _source_catalog(
        canonical, parent_bindings, parent_mapping
    )
    input_ids = list(transform["input_unit_ids"])
    if not set(input_ids).issubset(source_catalog):
        raise ValueError("transform input units are not present in its source node(s)")
    source_clocks = {source_catalog[item]["clock"] for item in input_ids}
    if len(source_clocks) != 1 or next(iter(source_clocks)) != _clock_rate(
        transform["source_clock"], "source_clock"
    ):
        raise ValueError("transform source clock disagrees with its input units")

    selected = (
        int(selected_global_output_sample_interval[0]),
        int(selected_global_output_sample_interval[1]),
    )
    total_samples = _total_output_samples(canonical, transform)
    selected = _integer_interval(
        selected,
        "selected_global_output_sample_interval",
        upper=total_samples,
    )
    output_num, output_den = _clock_rate(transform["output_clock"], "output_clock")
    selected_time = (
        _sample_edge_to_seconds(selected[0], output_num, output_den),
        _sample_edge_to_seconds(selected[1], output_num, output_den),
    )
    for parent in normalized_parents:
        parent_transform = validate_transform_spec(parent["transform_spec"])
        parent_num, parent_den = _clock_rate(
            parent_transform["output_clock"], "parent output_clock"
        )
        parent_selected = parent["coordinates"][
            "selected_global_output_sample_interval"
        ]
        parent_time = (
            _sample_edge_to_seconds(int(parent_selected[0]), parent_num, parent_den),
            _sample_edge_to_seconds(int(parent_selected[1]), parent_num, parent_den),
        )
        if (
            selected_time[0] < parent_time[0] - _TOL
            or selected_time[1] > parent_time[1] + _TOL
        ):
            raise ValueError("child view interval lies outside a bound parent view")

    if high_frequency_qualification_sha256 is not None:
        _sha256(
            high_frequency_qualification_sha256,
            "high_frequency_qualification_sha256",
        )
    definition_required = {
        "unit_id",
        "unit_type",
        "physical_unit",
        "observed",
        "imputed",
    }
    definitions: list[dict[str, Any]] = []
    for index, raw in enumerate(output_unit_definitions):
        definition = _strict_dict(
            raw, definition_required, f"output_unit_definitions[{index}]"
        )
        _nonempty(definition["unit_id"], f"output_unit_definitions[{index}].unit_id")
        if definition["unit_type"] not in {"electrode", "lead", "virtual"}:
            raise ValueError("output unit type is unsupported")
        if definition["physical_unit"] not in {"V", "dimensionless"}:
            raise ValueError("output unit physical_unit is unsupported")
        if (
            type(definition["observed"]) is not bool
            or type(definition["imputed"]) is not bool
        ):
            raise TypeError("output unit observed/imputed must be boolean")
        if definition["imputed"] and definition["observed"]:
            raise ValueError("an imputed output unit cannot be marked observed")
        definitions.append(definition)
    if [item["unit_id"] for item in definitions] != list(transform["output_unit_ids"]):
        raise ValueError("output unit definitions must follow transform output order")
    if len({item["unit_id"] for item in definitions}) != len(definitions):
        raise ValueError("output unit IDs must be unique")

    output_units = _derive_output_units(
        definitions,
        task_role=task_role,
        transform=transform,
        source_catalog=source_catalog,
        high_frequency_qualification_sha256=high_frequency_qualification_sha256,
    )
    temporal_evidence = _view_temporal_evidence(
        task_role=task_role,
        transform=transform,
        parent_views=normalized_parents,
    )
    _enforce_task_view_contract(
        task_role=task_role,
        transform=transform,
        parent_views=normalized_parents,
        temporal=temporal_evidence,
    )

    left_padding = _integer(padding_left_samples, "padding_left_samples")
    right_padding = _integer(padding_right_samples, "padding_right_samples")
    data_samples = selected[1] - selected[0]
    tensor_samples = left_padding + data_samples + right_padding
    padding_intervals = _expected_padding_intervals(
        data_samples=data_samples,
        left_padding=left_padding,
        right_padding=right_padding,
    )
    edge_intervals = _expected_edge_intervals(
        selected=selected,
        total_output_samples=total_samples,
        left_padding=left_padding,
        transform=transform,
    )
    quality_masks = _canonical_quality_masks(
        canonical,
        output_units,
        task_role=task_role,
        selected=selected,
        left_padding=left_padding,
        transform=transform,
    )
    quality_masks.extend(
        _inherited_parent_quality_masks(
            output_units,
            task_role=task_role,
            selected=selected,
            left_padding=left_padding,
            transform=transform,
            source_catalog=source_catalog,
        )
    )
    quality_masks.extend(deepcopy(dict(item)) for item in additional_quality_masks)
    quality_masks.sort(
        key=lambda row: (
            str(row.get("unit_id", "")),
            int(row.get("tensor_sample_interval", [0, 0])[0]),
            int(row.get("tensor_sample_interval", [0, 0])[1]),
            tuple(row.get("disabled_evidence_families", [])),
        )
    )
    masks_core = {
        "padding_intervals": padding_intervals,
        "edge_invalid_intervals": edge_intervals,
        "quality_invalid_intervals": quality_masks,
    }
    masks = {**masks_core, "mask_sha256": _canonical_sha256(masks_core)}

    expansion_safe = _cache_is_expansion_safe(transform)
    cache_namespace = _cache_namespace(
        canonical=canonical,
        transform=transform,
        parent_bindings=parent_bindings,
        selected=selected,
        expansion_safe=expansion_safe,
    )
    cache = {
        "cache_namespace_sha256": cache_namespace,
        "expansion_safe": expansion_safe,
        "expansion_policy": (
            CACHE_EXPANSION_POLICY if expansion_safe else NONREUSABLE_CACHE_POLICY
        ),
        "tile_size_samples": int(cache_tile_size_samples),
        "tiles": [deepcopy(dict(item)) for item in cache_tiles],
    }
    body: dict[str, Any] = {
        "schema_version": SIGNAL_VIEW_SCHEMA_VERSION,
        "view_receipt_id": "CONTENT-ADDRESS-PENDING",
        "view_id": view_id,
        "task_role": task_role,
        "canonical_signal_id": canonical["canonical_signal_id"],
        "canonical_receipt_sha256": canonical["receipt_sha256"],
        "parent_view_bindings": parent_bindings,
        "transform_spec": transform,
        "temporal_evidence": temporal_evidence,
        "output_units": output_units,
        "coordinates": {
            "coordinate_system": COORDINATE_SYSTEM,
            "clock_policy": GLOBAL_CLOCK_POLICY,
            "selected_global_output_sample_interval": list(selected),
            "selected_recording_seconds": [selected_time[0], selected_time[1]],
        },
        "tensor_layout": {
            "axis_order": ["unit", "time"],
            "data_sample_count": data_samples,
            "padding_left_samples": left_padding,
            "padding_right_samples": right_padding,
            "tensor_sample_count": tensor_samples,
            "valid_data_tensor_sample_interval": [
                left_padding,
                left_padding + data_samples,
            ],
        },
        "masks": masks,
        "cache": cache,
        "processed_view_sha256": processed_view_sha256,
        "high_frequency_qualification_sha256": high_frequency_qualification_sha256,
        "annotation_firewall": deepcopy(_FIREWALL),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["view_receipt_id"] = _view_digest_id(body)
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_signal_view_receipt(
        body,
        canonical,
        trusted_parent_views=parent_mapping,
    )


def _validate_output_units(
    raw_units: object,
    *,
    task_role: str,
    transform: Mapping[str, Any],
    source_catalog: Mapping[str, Mapping[str, Any]],
    high_frequency_qualification_sha256: str | None,
) -> list[dict[str, Any]]:
    if not isinstance(raw_units, list) or not raw_units:
        raise TypeError("signal view output_units must be a non-empty array")
    required = {
        "unit_id",
        "unit_type",
        "physical_unit",
        "observed",
        "imputed",
        "source_unit_ids",
        "canonical_source_channel_ids",
        "effective_bandwidth_hz",
        "evidence_eligible",
        "evidence_eligibility",
    }
    units: list[dict[str, Any]] = []
    definitions: list[dict[str, Any]] = []
    family_required = {
        "family",
        "eligible",
        "effective_bandwidth_hz",
        "reason_codes",
    }
    for index, raw in enumerate(raw_units):
        unit = _strict_dict(raw, required, f"output_units[{index}]")
        unit_id = _nonempty(unit["unit_id"], f"output_units[{index}].unit_id")
        if unit["unit_type"] not in {"electrode", "lead", "virtual"}:
            raise ValueError(f"output_units[{index}] has unsupported unit_type")
        if unit["physical_unit"] not in {"V", "dimensionless"}:
            raise ValueError(f"output_units[{index}] has unsupported physical_unit")
        if type(unit["observed"]) is not bool or type(unit["imputed"]) is not bool:
            raise TypeError(f"output_units[{index}] observed/imputed must be boolean")
        if unit["imputed"] and unit["observed"]:
            raise ValueError("an imputed output unit cannot be observed")
        source_ids = _unique_strings(
            unit["source_unit_ids"], f"output_units[{index}].source_unit_ids"
        )
        if not set(source_ids).issubset(source_catalog):
            raise ValueError(f"output_units[{index}] references unknown source units")
        canonical_ids = _unique_strings(
            unit["canonical_source_channel_ids"],
            f"output_units[{index}].canonical_source_channel_ids",
        )
        bandwidth = _float_interval(
            unit["effective_bandwidth_hz"],
            f"output_units[{index}].effective_bandwidth_hz",
        )
        if type(unit["evidence_eligible"]) is not bool:
            raise TypeError(f"output_units[{index}].evidence_eligible must be boolean")
        if (not unit["observed"] or unit["imputed"]) and unit["evidence_eligible"]:
            raise ValueError(
                "unobserved/imputed output units cannot be evidence eligible"
            )
        if not isinstance(unit["evidence_eligibility"], list):
            raise TypeError(
                f"output_units[{index}].evidence_eligibility must be an array"
            )
        families: list[str] = []
        for family_index, raw_family in enumerate(unit["evidence_eligibility"]):
            family = _strict_dict(
                raw_family,
                family_required,
                f"output_units[{index}].evidence_eligibility[{family_index}]",
            )
            if family["family"] not in EVIDENCE_FAMILIES:
                raise ValueError("output unit contains an unknown evidence family")
            families.append(str(family["family"]))
            if type(family["eligible"]) is not bool:
                raise TypeError("evidence family eligible must be boolean")
            family_bandwidth = _float_interval(
                family["effective_bandwidth_hz"],
                "evidence family effective_bandwidth_hz",
            )
            if any(abs(a - b) > _TOL for a, b in zip(family_bandwidth, bandwidth)):
                raise ValueError("evidence-family bandwidth disagrees with its unit")
            reasons = _unique_strings(
                family["reason_codes"],
                "evidence family reason_codes",
                nonempty=False,
            )
            if family["eligible"] and reasons:
                raise ValueError("eligible evidence family cannot carry reason codes")
            if not family["eligible"] and not reasons:
                raise ValueError("ineligible evidence family requires reason codes")
        if families != list(EVIDENCE_FAMILIES):
            raise ValueError("output unit must cover evidence families in frozen order")
        units.append(unit)
        definitions.append(
            {
                "unit_id": unit_id,
                "unit_type": unit["unit_type"],
                "physical_unit": unit["physical_unit"],
                "observed": unit["observed"],
                "imputed": unit["imputed"],
            }
        )
    if [item["unit_id"] for item in units] != list(transform["output_unit_ids"]):
        raise ValueError("output unit order disagrees with transform output order")
    if len({item["unit_id"] for item in units}) != len(units):
        raise ValueError("output unit IDs must be unique")
    expected = _derive_output_units(
        definitions,
        task_role=task_role,
        transform=transform,
        source_catalog=source_catalog,
        high_frequency_qualification_sha256=high_frequency_qualification_sha256,
    )
    if units != expected:
        raise ValueError("output-unit provenance, bandwidth or eligibility drifted")
    return units


def _validate_quality_mask_row(
    raw: object,
    *,
    index: int,
    unit_ids: set[str],
    data_interval: tuple[int, int],
) -> dict[str, Any]:
    row = _strict_dict(
        raw,
        {
            "unit_id",
            "tensor_sample_interval",
            "severity",
            "disabled_evidence_families",
            "reason_codes",
        },
        f"quality_invalid_intervals[{index}]",
    )
    if row["unit_id"] not in unit_ids:
        raise ValueError("quality mask references an unknown output unit")
    interval = _integer_interval(
        row["tensor_sample_interval"],
        f"quality_invalid_intervals[{index}].tensor_sample_interval",
        upper=data_interval[1],
    )
    if not _covers(data_interval, interval):
        raise ValueError("quality-invalid interval lies outside observed data")
    if row["severity"] not in {"limited", "unusable"}:
        raise ValueError("quality-invalid interval has unsupported severity")
    disabled = _unique_strings(
        row["disabled_evidence_families"],
        f"quality_invalid_intervals[{index}].disabled_evidence_families",
    )
    if not set(disabled).issubset(EVIDENCE_FAMILIES):
        raise ValueError("quality mask disables an unknown evidence family")
    if row["severity"] == "unusable" and set(disabled) != set(EVIDENCE_FAMILIES):
        raise ValueError("unusable quality masks must disable every evidence family")
    _unique_strings(
        row["reason_codes"], f"quality_invalid_intervals[{index}].reason_codes"
    )
    return row


def _validate_masks(
    masks_raw: object,
    *,
    canonical: Mapping[str, Any],
    output_units: Sequence[Mapping[str, Any]],
    selected: tuple[int, int],
    total_output_samples: int,
    tensor_layout: Mapping[str, Any],
    transform: Mapping[str, Any],
    task_role: str,
    source_catalog: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    masks = _strict_dict(
        masks_raw,
        {
            "padding_intervals",
            "edge_invalid_intervals",
            "quality_invalid_intervals",
            "mask_sha256",
        },
        "signal view masks",
    )
    left_padding = int(tensor_layout["padding_left_samples"])
    right_padding = int(tensor_layout["padding_right_samples"])
    data_samples = selected[1] - selected[0]
    expected_padding = _expected_padding_intervals(
        data_samples=data_samples,
        left_padding=left_padding,
        right_padding=right_padding,
    )
    if masks["padding_intervals"] != expected_padding:
        raise ValueError("padding mask does not exactly cover every padding sample")
    expected_edges = _expected_edge_intervals(
        selected=selected,
        total_output_samples=total_output_samples,
        left_padding=left_padding,
        transform=transform,
    )
    if masks["edge_invalid_intervals"] != expected_edges:
        raise ValueError("edge-invalid mask disagrees with the transform policy")
    data_interval = tuple(tensor_layout["valid_data_tensor_sample_interval"])
    unit_ids = {str(unit["unit_id"]) for unit in output_units}
    if not isinstance(masks["quality_invalid_intervals"], list):
        raise TypeError("quality_invalid_intervals must be an array")
    quality_rows = [
        _validate_quality_mask_row(
            raw,
            index=index,
            unit_ids=unit_ids,
            data_interval=(int(data_interval[0]), int(data_interval[1])),
        )
        for index, raw in enumerate(masks["quality_invalid_intervals"])
    ]
    expected_order = sorted(
        quality_rows,
        key=lambda row: (
            str(row["unit_id"]),
            int(row["tensor_sample_interval"][0]),
            int(row["tensor_sample_interval"][1]),
            tuple(row["disabled_evidence_families"]),
        ),
    )
    if quality_rows != expected_order:
        raise ValueError("quality-invalid intervals must use deterministic order")
    required_quality = _canonical_quality_masks(
        canonical,
        output_units,
        task_role=task_role,
        selected=selected,
        left_padding=left_padding,
        transform=transform,
    )
    required_quality.extend(
        _inherited_parent_quality_masks(
            output_units,
            task_role=task_role,
            selected=selected,
            left_padding=left_padding,
            transform=transform,
            source_catalog=source_catalog,
        )
    )
    for required in required_quality:
        if required not in quality_rows:
            raise ValueError(
                "canonical QC primitive was not propagated to the view mask"
            )
    _sha256(masks["mask_sha256"], "view mask_sha256")
    mask_core = {
        "padding_intervals": masks["padding_intervals"],
        "edge_invalid_intervals": masks["edge_invalid_intervals"],
        "quality_invalid_intervals": quality_rows,
    }
    if masks["mask_sha256"] != _canonical_sha256(mask_core):
        raise ValueError("view mask hash does not bind its content")
    return masks


def _validate_cache(
    cache_raw: object,
    *,
    canonical: Mapping[str, Any],
    transform: Mapping[str, Any],
    parent_bindings: Sequence[Mapping[str, Any]],
    selected: tuple[int, int],
    total_output_samples: int,
) -> dict[str, Any]:
    cache = _strict_dict(
        cache_raw,
        {
            "cache_namespace_sha256",
            "expansion_safe",
            "expansion_policy",
            "tile_size_samples",
            "tiles",
        },
        "signal view cache",
    )
    if type(cache["expansion_safe"]) is not bool:
        raise TypeError("cache.expansion_safe must be boolean")
    expected_safe = _cache_is_expansion_safe(transform)
    if cache["expansion_safe"] is not expected_safe:
        raise ValueError("cache expansion-safety flag is inconsistent")
    expected_policy = (
        CACHE_EXPANSION_POLICY if expected_safe else NONREUSABLE_CACHE_POLICY
    )
    if cache["expansion_policy"] != expected_policy:
        raise ValueError("cache expansion policy is inconsistent")
    _sha256(cache["cache_namespace_sha256"], "cache namespace SHA-256")
    expected_namespace = _cache_namespace(
        canonical=canonical,
        transform=transform,
        parent_bindings=parent_bindings,
        selected=selected,
        expansion_safe=expected_safe,
    )
    if cache["cache_namespace_sha256"] != expected_namespace:
        raise ValueError("cache namespace does not bind canonical/transform semantics")
    tile_size = _integer(
        cache["tile_size_samples"], "cache tile_size_samples", minimum=1
    )
    if not isinstance(cache["tiles"], list) or not cache["tiles"]:
        raise TypeError("cache tiles must be a non-empty array")
    tile_required = {
        "tile_index",
        "global_output_sample_interval",
        "signal_sha256",
        "quality_mask_sha256",
    }
    normalized_tiles: list[dict[str, Any]] = []
    tile_indices: set[int] = set()
    for row_index, raw in enumerate(cache["tiles"]):
        tile = _strict_dict(raw, tile_required, f"cache.tiles[{row_index}]")
        tile_index = _integer(
            tile["tile_index"], f"cache.tiles[{row_index}].tile_index"
        )
        if tile_index in tile_indices:
            raise ValueError("cache tile indices must be unique")
        tile_indices.add(tile_index)
        interval = _integer_interval(
            tile["global_output_sample_interval"],
            f"cache.tiles[{row_index}].global_output_sample_interval",
            upper=total_output_samples,
        )
        expected_interval = (
            tile_index * tile_size,
            min((tile_index + 1) * tile_size, total_output_samples),
        )
        if interval != expected_interval:
            raise ValueError("cache tile interval is not on the immutable global grid")
        _sha256(tile["signal_sha256"], f"cache.tiles[{row_index}].signal_sha256")
        _sha256(
            tile["quality_mask_sha256"],
            f"cache.tiles[{row_index}].quality_mask_sha256",
        )
        normalized_tiles.append(tile)
    if [tile["tile_index"] for tile in normalized_tiles] != sorted(tile_indices):
        raise ValueError("cache tiles must be sorted by tile_index")
    required_indices = set(
        range(selected[0] // tile_size, (selected[1] - 1) // tile_size + 1)
    )
    if not required_indices.issubset(tile_indices):
        raise ValueError("cache tiles do not cover the selected signal interval")
    return cache


def validate_signal_view_receipt(
    payload: object,
    canonical_receipt: object,
    *,
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Strictly validate one task-specific view and its evidence boundary.

    Parent view objects are host-supplied, not accepted from the payload.  A
    complete graph should normally be checked with :func:`build_signal_view_dag`;
    direct canonical children need no parent registry.
    """

    canonical = validate_canonical_signal_receipt(canonical_receipt)
    required = {
        "schema_version",
        "view_receipt_id",
        "view_id",
        "task_role",
        "canonical_signal_id",
        "canonical_receipt_sha256",
        "parent_view_bindings",
        "transform_spec",
        "temporal_evidence",
        "output_units",
        "coordinates",
        "tensor_layout",
        "masks",
        "cache",
        "processed_view_sha256",
        "high_frequency_qualification_sha256",
        "annotation_firewall",
        "receipt_sha256",
    }
    data = _strict_dict(payload, required, "signal view receipt")
    if data["schema_version"] != SIGNAL_VIEW_SCHEMA_VERSION:
        raise ValueError("unsupported signal view schema")
    _nonempty(data["view_id"], "signal view_id")
    if data["task_role"] not in TASK_ROLES:
        raise ValueError("signal view task_role is unsupported")
    if data["canonical_signal_id"] != canonical["canonical_signal_id"]:
        raise ValueError("signal view belongs to a different canonical signal")
    if data["canonical_receipt_sha256"] != canonical["receipt_sha256"]:
        raise ValueError("signal view does not bind the canonical receipt")
    if data["annotation_firewall"] != _FIREWALL:
        raise ValueError("signal view violates the annotation firewall")
    if not isinstance(data["parent_view_bindings"], list):
        raise TypeError("parent_view_bindings must be an array")
    trusted = (
        {}
        if trusted_parent_views is None
        else {
            str(key): deepcopy(dict(value))
            for key, value in trusted_parent_views.items()
        }
    )
    source_catalog, normalized_parents = _source_catalog(
        canonical,
        data["parent_view_bindings"],
        trusted,
    )
    if any(parent["view_id"] == data["view_id"] for parent in normalized_parents):
        raise ValueError("signal view cannot be its own parent")

    transform = validate_transform_spec(data["transform_spec"])
    input_ids = list(transform["input_unit_ids"])
    if not set(input_ids).issubset(source_catalog):
        raise ValueError(
            "transform input units are absent from canonical/parent sources"
        )
    clocks = {source_catalog[item]["clock"] for item in input_ids}
    if len(clocks) != 1 or next(iter(clocks)) != _clock_rate(
        transform["source_clock"], "source_clock"
    ):
        raise ValueError("signal view source clocks are inconsistent")
    total_samples = _total_output_samples(canonical, transform)

    expected_temporal = _view_temporal_evidence(
        task_role=str(data["task_role"]),
        transform=transform,
        parent_views=normalized_parents,
    )
    supplied_temporal = _trusted_parent_temporal_contract(
        {"temporal_evidence": data["temporal_evidence"]},
        context="signal view",
    )
    if supplied_temporal != expected_temporal:
        raise ValueError("signal view temporal-evidence contract drifted")
    _enforce_task_view_contract(
        task_role=str(data["task_role"]),
        transform=transform,
        parent_views=normalized_parents,
        temporal=supplied_temporal,
    )

    coordinates = _strict_dict(
        data["coordinates"],
        {
            "coordinate_system",
            "clock_policy",
            "selected_global_output_sample_interval",
            "selected_recording_seconds",
        },
        "signal view coordinates",
    )
    if (
        coordinates["coordinate_system"] != COORDINATE_SYSTEM
        or coordinates["clock_policy"] != GLOBAL_CLOCK_POLICY
    ):
        raise ValueError("signal view coordinate policy drifted")
    selected = _integer_interval(
        coordinates["selected_global_output_sample_interval"],
        "selected_global_output_sample_interval",
        upper=total_samples,
    )
    output_num, output_den = _clock_rate(transform["output_clock"], "output_clock")
    expected_seconds = [
        _sample_edge_to_seconds(selected[0], output_num, output_den),
        _sample_edge_to_seconds(selected[1], output_num, output_den),
    ]
    if coordinates["selected_recording_seconds"] != expected_seconds:
        raise ValueError("signal view sample interval is not reversibly bound to time")
    for parent in normalized_parents:
        parent_transform = validate_transform_spec(parent["transform_spec"])
        parent_num, parent_den = _clock_rate(
            parent_transform["output_clock"], "parent output_clock"
        )
        parent_interval = parent["coordinates"][
            "selected_global_output_sample_interval"
        ]
        parent_seconds = (
            _sample_edge_to_seconds(int(parent_interval[0]), parent_num, parent_den),
            _sample_edge_to_seconds(int(parent_interval[1]), parent_num, parent_den),
        )
        if (
            expected_seconds[0] < parent_seconds[0] - _TOL
            or expected_seconds[1] > parent_seconds[1] + _TOL
        ):
            raise ValueError("child signal view lies outside a parent interval")

    high_frequency_receipt = _sha256(
        data["high_frequency_qualification_sha256"],
        "high_frequency_qualification_sha256",
        nullable=True,
    )
    output_units = _validate_output_units(
        data["output_units"],
        task_role=str(data["task_role"]),
        transform=transform,
        source_catalog=source_catalog,
        high_frequency_qualification_sha256=high_frequency_receipt,
    )

    layout = _strict_dict(
        data["tensor_layout"],
        {
            "axis_order",
            "data_sample_count",
            "padding_left_samples",
            "padding_right_samples",
            "tensor_sample_count",
            "valid_data_tensor_sample_interval",
        },
        "signal view tensor_layout",
    )
    if layout["axis_order"] != ["unit", "time"]:
        raise ValueError("signal view tensor axis order drifted")
    data_count = selected[1] - selected[0]
    left_padding = _integer(layout["padding_left_samples"], "padding_left_samples")
    right_padding = _integer(layout["padding_right_samples"], "padding_right_samples")
    expected_layout = {
        "axis_order": ["unit", "time"],
        "data_sample_count": data_count,
        "padding_left_samples": left_padding,
        "padding_right_samples": right_padding,
        "tensor_sample_count": left_padding + data_count + right_padding,
        "valid_data_tensor_sample_interval": [
            left_padding,
            left_padding + data_count,
        ],
    }
    if layout != expected_layout:
        raise ValueError("signal view tensor layout is inconsistent")
    _validate_masks(
        data["masks"],
        canonical=canonical,
        output_units=output_units,
        selected=selected,
        total_output_samples=total_samples,
        tensor_layout=layout,
        transform=transform,
        task_role=str(data["task_role"]),
        source_catalog=source_catalog,
    )
    _validate_cache(
        data["cache"],
        canonical=canonical,
        transform=transform,
        parent_bindings=data["parent_view_bindings"],
        selected=selected,
        total_output_samples=total_samples,
    )
    _sha256(data["processed_view_sha256"], "processed_view_sha256")

    _nonempty(data["view_receipt_id"], "view_receipt_id")
    expected_id = _view_digest_id(data)
    if data["view_receipt_id"] != expected_id:
        raise ValueError("signal view receipt ID does not bind its content")
    _sha256(data["receipt_sha256"], "signal view receipt_sha256")
    digest_source = deepcopy(data)
    digest_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("signal view receipt hash does not bind its content")
    return data


def view_tensor_index_to_recording_seconds(
    view_receipt: object,
    *,
    tensor_sample_index: int,
) -> float:
    """Map a valid data sample edge to recording-relative seconds.

    Padding has no physical time and is rejected.  The right edge of the last
    valid sample is accepted, which is useful for half-open intervals.
    """

    if type(view_receipt) is not dict:
        raise TypeError("view_receipt must be an object")
    view = deepcopy(view_receipt)
    transform = validate_transform_spec(view.get("transform_spec"))
    layout = view.get("tensor_layout")
    coordinates = view.get("coordinates")
    if not isinstance(layout, Mapping) or not isinstance(coordinates, Mapping):
        raise ValueError("view receipt lacks coordinate metadata")
    index = _integer(tensor_sample_index, "tensor_sample_index")
    valid_start, valid_stop = (
        int(item) for item in layout["valid_data_tensor_sample_interval"]
    )
    if index < valid_start or index > valid_stop:
        raise ValueError("padding has no recording-relative time coordinate")
    selected_start = int(coordinates["selected_global_output_sample_interval"][0])
    global_index = selected_start + index - valid_start
    numerator, denominator = _clock_rate(transform["output_clock"], "output_clock")
    return _sample_edge_to_seconds(global_index, numerator, denominator)


def recording_seconds_to_view_tensor_index(
    view_receipt: object,
    *,
    recording_seconds: float,
    rounding: str = "exact",
) -> int:
    """Map physical time to a non-padding tensor sample edge."""

    if type(view_receipt) is not dict:
        raise TypeError("view_receipt must be an object")
    view = deepcopy(view_receipt)
    transform = validate_transform_spec(view.get("transform_spec"))
    layout = view.get("tensor_layout")
    coordinates = view.get("coordinates")
    if not isinstance(layout, Mapping) or not isinstance(coordinates, Mapping):
        raise ValueError("view receipt lacks coordinate metadata")
    numerator, denominator = _clock_rate(transform["output_clock"], "output_clock")
    global_index = _seconds_to_sample_edge(
        recording_seconds,
        numerator,
        denominator,
        context="recording_seconds",
        rounding=rounding,
    )
    selected_start, selected_stop = (
        int(item) for item in coordinates["selected_global_output_sample_interval"]
    )
    if global_index < selected_start or global_index > selected_stop:
        raise ValueError("recording_seconds lies outside the selected view interval")
    valid_start = int(layout["valid_data_tensor_sample_interval"][0])
    return valid_start + global_index - selected_start
