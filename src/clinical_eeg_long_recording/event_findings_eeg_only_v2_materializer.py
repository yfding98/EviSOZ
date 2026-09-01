"""Fail-closed materialization for the EEG-only event Findings v2 profile.

This module has two deliberately separate responsibilities:

* build content-addressed common-17 raw/display waveform artifacts from a
  verified EDF and a verified, already-materialized event support receipt;
* project only explicitly allow-listed numerical fields from that receipt
  into :mod:`event_findings_eeg_only_v2`.

The projection layer never opens an EDF, annotation API, TERM/SzCORE sidecar,
spreadsheet, clinical text, patient header, behaviour/video stream or LLM.
The waveform builder opens only the 17 directly observed ``-REF`` EEG signal
channels and acquisition calibration fields through the narrow reader used by
the real-EDF adaptive rollout.  FZ/PZ are neither read nor synthesized.

Positive observations require a verified real waveform manifest.  Without
one, every category is emitted as ``not_evaluable`` with the typed reason
``waveform_artifact_unmaterialized``.  A URI or digest is never guessed from
a filename or copied from an unchecked JSON field.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Final, Mapping, Sequence
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .adaptive_native_evidence_common17 import (
    ADAPTIVE_NATIVE_EVIDENCE_METHOD_ID,
    ADAPTIVE_NATIVE_EVIDENCE_SCHEMA_VERSION,
    COMMON17_CHANNELS,
    COMMON17_TCP_EDGES,
    validate_common17_adaptive_native_event_evidence,
)
from .event_findings_eeg_only_v2 import (
    EVENT_FINDINGS_EEG_ONLY_V2_REGISTRY_ID,
    EVENT_FINDINGS_EEG_ONLY_V2_SCHEMA_VERSION,
    REQUIRED_CATEGORIES,
    load_event_findings_eeg_only_v2_registry,
    validate_event_findings_eeg_only_v2,
)
from .tusz_real_edf_adaptive_findings_v1 import (
    DirectObservedCommon17EDFQueryReader,
    TUSZ_REAL_EDF_ADAPTIVE_ROLLOUT_SCHEMA,
)


PROJECTION_ALLOWLIST_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_event_findings_eeg_only_v2_projection_allowlist_v1"
)
PROJECTION_ALLOWLIST_ID: Final[str] = (
    "CLINICAL-EEG-EVENT-FINDINGS-EEG-ONLY-V2-PROJECTION-ALLOWLIST-V1"
)
WAVEFORM_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_common17_waveform_artifact_manifest_v1"
)
WAVEFORM_BUILDER_ID: Final[str] = (
    "EEG-ONLY-DIRECT-COMMON17-WAVEFORM-ARTIFACT-BUILDER-V1"
)
PROJECTION_MATERIALIZER_ID: Final[str] = (
    "EEG-ONLY-EXPLICIT-ALLOWLIST-PROJECTION-MATERIALIZER-V1"
)

_ROOT = Path(__file__).resolve().parents[2]
_ALLOWLIST_PATH = (
    _ROOT
    / "configs"
    / "clinical_eeg_event_findings_eeg_only_v2_projection_allowlist.json"
)
_REGISTRY_PATH = (
    _ROOT / "configs" / "clinical_eeg_event_findings_eeg_only_v2_registry.json"
)
_TIME_TOL = 1e-6
_SHA_CHARS = frozenset("0123456789abcdef")
_BAND_FEATURES: Final[tuple[str, ...]] = (
    "delta_relative_power",
    "theta_relative_power",
    "alpha_relative_power",
    "beta_relative_power",
    "gamma_relative_power",
)
_SOURCE_UNIT_TO_PROFILE_UNIT: Final[dict[str, str]] = {
    "spectral_concentration_ratio": "ratio",
}
_KNOWN_PROJECTORS: Final[frozenset[str]] = frozenset(
    {
        "background_band_relative_power",
        "native_candidate_vector",
        "quantitative_change_trajectory",
        "earliest_scalp_change_field_candidate",
        "earliest_field_graph_contiguity",
        "return_to_baseline_candidate",
        "reference_distribution_js_similarity",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"hash target is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA_CHARS for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"existing content-addressed artifact differs: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, payload: object) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_bytes(path, encoded)


def _safe_relative_edf(root: Path, relative: object) -> tuple[str, Path]:
    posix = PurePosixPath(str(relative))
    if posix.is_absolute() or ".." in posix.parts or posix.suffix.lower() != ".edf":
        raise ValueError("source relative EDF path is unsafe")
    source = root.joinpath(*posix.parts).resolve(strict=True)
    source.relative_to(root)
    if source.is_symlink() or not source.is_file():
        raise ValueError("source EDF must be a regular non-symlinked file")
    return posix.as_posix(), source


def _verify_source_rollout_receipt(payload: object) -> dict[str, Any]:
    """Verify the immutable outer receipt and its authoritative inner wire."""

    if type(payload) is not dict:
        raise TypeError("source rollout receipt must be an object")
    required = {
        "schema_version",
        "receipt_sha256",
        "rollout_id",
        "manifest_sha256",
        "source",
        "selection_only",
        "reader_receipt",
        "event_findings_evidence",
        "scope_receipt",
        "claim_limits",
    }
    if set(payload) != required:
        raise ValueError("source rollout receipt fields drifted")
    result = deepcopy(payload)
    if result["schema_version"] != TUSZ_REAL_EDF_ADAPTIVE_ROLLOUT_SCHEMA:
        raise ValueError("unsupported source rollout schema")
    expected = _canonical_sha256(
        {key: value for key, value in result.items() if key != "receipt_sha256"}
    )
    if result["receipt_sha256"] != expected:
        raise ValueError("source rollout receipt content hash mismatch")

    evidence = validate_common17_adaptive_native_event_evidence(
        result["event_findings_evidence"]
    )
    source = result["source"]
    reader = result["reader_receipt"]
    if not isinstance(source, dict) or not isinstance(reader, dict):
        raise TypeError("source or reader receipt is malformed")
    if source.get("recording_id") != evidence["recording_id"]:
        raise ValueError("outer and inner recording identities differ")
    if source.get("event_id") != evidence["event_id"]:
        raise ValueError("outer and inner event identities differ")
    if reader.get("common17_channel_order") != list(COMMON17_CHANNELS):
        raise ValueError("source reader is not exact common17")
    if reader.get("source_edf_sha256") != source.get("edf_sha256"):
        raise ValueError("source EDF hash binding drifted")
    if reader.get("source_sampling_rate_hz") != evidence["acquisition"].get(
        "sampling_rate_hz"
    ):
        raise ValueError("source sampling rate binding drifted")
    if reader.get("recording_sample_count") != evidence["acquisition"].get(
        "recording_sample_count"
    ):
        raise ValueError("source recording length binding drifted")
    forbidden_true = {
        "FZ_PZ_samples_read",
        "non_common17_signal_samples_read",
        "EDF_annotation_API_called",
        "patient_header_API_called",
        "target_sidecar_opened",
    }
    if any(reader.get(field) is not False for field in forbidden_true):
        raise ValueError("source reader receipt crossed the EEG-only firewall")
    scope = result["scope_receipt"]
    if not isinstance(scope, dict):
        raise TypeError("source scope receipt is malformed")
    scope_false = {
        "TERM_or_other_target_sidecar_opened_at_runtime",
        "EDF_annotations_opened",
        "SOZ_or_channel_target_opened",
        "clinical_text_or_spreadsheet_opened",
        "patient_header_fields_opened",
        "non_common17_signal_samples_read",
        "FZ_or_PZ_samples_read",
        "zero_fill_interpolation_or_montage_synthesis_used",
        "feature_threshold_training_used",
    }
    if any(scope.get(field) is not False for field in scope_false):
        raise ValueError("source scope receipt crossed a forbidden input boundary")
    return result


def load_verified_source_rollout_receipt(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise ValueError("source rollout receipt must be a regular file")
    return _verify_source_rollout_receipt(
        json.loads(source.read_text(encoding="utf-8"))
    )


@lru_cache(maxsize=1)
def load_projection_allowlist(
    path: str | Path = _ALLOWLIST_PATH,
) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "allowlist_id",
        "source_schema_version",
        "source_evidence_schema_version",
        "profile_binding_paths",
        "projection_rules",
        "category_fallback_metrics",
        "runtime_forbidden_inputs",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("projection allowlist fields drifted")
    if value["schema_version"] != PROJECTION_ALLOWLIST_SCHEMA_VERSION:
        raise ValueError("projection allowlist schema drifted")
    if value["allowlist_id"] != PROJECTION_ALLOWLIST_ID:
        raise ValueError("projection allowlist identity drifted")
    if value["source_schema_version"] != TUSZ_REAL_EDF_ADAPTIVE_ROLLOUT_SCHEMA:
        raise ValueError("projection allowlist source schema drifted")
    if value["source_evidence_schema_version"] != ADAPTIVE_NATIVE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("projection allowlist evidence schema drifted")

    registry = {
        str(row["metric_id"]): row
        for row in load_event_findings_eeg_only_v2_registry()["capabilities"]
    }
    bindings = value["profile_binding_paths"]
    if (
        not isinstance(bindings, list)
        or len(bindings) != len(set(bindings))
        or any(not isinstance(item, str) or not item.startswith("/") for item in bindings)
    ):
        raise ValueError("profile binding path allowlist is invalid")
    seen_rules: set[str] = set()
    seen_categories: set[str] = set()
    for rule in value["projection_rules"]:
        if not isinstance(rule, dict):
            raise TypeError("projection rule must be an object")
        expected_fields = {
            "rule_id",
            "category",
            "metric_id",
            "projector",
            "source_paths",
        }
        if rule.get("projector") == "native_candidate_vector":
            expected_fields.add("source_feature")
        if set(rule) != expected_fields:
            raise ValueError(f"projection rule fields drifted: {rule.get('rule_id')}")
        rule_id = str(rule["rule_id"])
        category = str(rule["category"])
        metric_id = str(rule["metric_id"])
        if rule_id in seen_rules or category in seen_categories:
            raise ValueError("projection rules duplicate an ID or category")
        seen_rules.add(rule_id)
        seen_categories.add(category)
        capability = registry.get(metric_id)
        if capability is None or capability["category"] != category:
            raise ValueError(f"projection rule is absent from registry: {rule_id}")
        if rule["projector"] not in _KNOWN_PROJECTORS:
            raise ValueError(f"projection rule uses an unknown projector: {rule_id}")
        paths = rule["source_paths"]
        if (
            not isinstance(paths, list)
            or not paths
            or len(paths) != len(set(paths))
            or any(not isinstance(item, str) or not item.startswith("/") for item in paths)
        ):
            raise ValueError(f"projection rule has invalid source paths: {rule_id}")

    fallback = value["category_fallback_metrics"]
    if not isinstance(fallback, dict) or tuple(fallback) != REQUIRED_CATEGORIES:
        raise ValueError("fallback metrics must exactly cover the frozen categories")
    for category, metric_id in fallback.items():
        capability = registry.get(str(metric_id))
        if capability is None or capability["category"] != category:
            raise ValueError(f"fallback metric disagrees with registry: {category}")
        if "not_evaluable" not in capability["allowed_evidence_levels"]:
            raise ValueError(f"fallback metric cannot abstain: {category}")
    return deepcopy(value)


def _json_pointer(payload: Mapping[str, Any], pointer: str) -> Any:
    current: Any = payload
    for token in pointer.split("/")[1:]:
        key = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(pointer)
        current = current[key]
    return deepcopy(current)


def _projection_view(
    payload: Mapping[str, Any], allowlist: Mapping[str, Any]
) -> dict[str, Any]:
    paths = list(allowlist["profile_binding_paths"])
    for rule in allowlist["projection_rules"]:
        paths.extend(rule["source_paths"])
    result: dict[str, Any] = {}
    for path in dict.fromkeys(paths):
        try:
            result[path] = _json_pointer(payload, path)
        except KeyError:
            # A verified source may explicitly carry ``null`` for an
            # unqualified final-evidence family.  The allowlist projection
            # translates that absence to a typed not-evaluable row.
            result[path] = None
    return result


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _npy_bytes(np.asarray(arrays[name])))
    return stream.getvalue()


def _canonical_signal_sha256(
    *,
    samples_uv: np.ndarray,
    valid_sample_mask: np.ndarray,
    sampling_rate_hz: float,
    interval_samples: Sequence[int],
) -> str:
    signal = np.asarray(samples_uv, dtype="<f4", order="C")
    valid = np.asarray(valid_sample_mask, dtype=np.uint8, order="C")
    header = {
        "domain": "direct-observed-common17-native-referential-uv-f32-v1",
        "channel_order": list(COMMON17_CHANNELS),
        "sampling_rate_hz": float(sampling_rate_hz),
        "interval_samples": [int(interval_samples[0]), int(interval_samples[1])],
        "shape": list(signal.shape),
        "valid_mask_dtype": "uint8",
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(header))
    digest.update(signal.tobytes(order="C"))
    digest.update(valid.tobytes(order="C"))
    return digest.hexdigest()


def _render_waveform_png(
    *,
    samples_uv: np.ndarray,
    valid_sample_mask: np.ndarray,
    sampling_rate_hz: float,
    interval_seconds: Sequence[float],
    recording_id: str,
) -> tuple[bytes, dict[str, Any]]:
    """Render a deterministic display-only overview without clinical markup."""

    width = 2400
    margin_left = 150
    margin_right = 40
    margin_top = 90
    margin_bottom = 70
    track_height = 96
    height = margin_top + margin_bottom + track_height * len(COMMON17_CHANNELS)
    plot_width = width - margin_left - margin_right
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    signal = np.asarray(samples_uv, dtype=np.float64)
    valid = np.asarray(valid_sample_mask, dtype=bool)
    centered = np.zeros_like(signal)
    channel_centers: list[float] = []
    absolute_values: list[np.ndarray] = []
    for channel in range(signal.shape[0]):
        usable = signal[channel, valid[channel]]
        center = float(np.median(usable)) if usable.size else 0.0
        channel_centers.append(center)
        centered[channel] = signal[channel] - center
        if usable.size:
            absolute_values.append(np.abs(usable - center))
    pooled = np.concatenate(absolute_values) if absolute_values else np.asarray([1.0])
    robust_uv = max(1.0, float(np.quantile(pooled, 0.99)))
    pixels_per_uv = 0.36 * track_height / robust_uv

    draw.text(
        (margin_left, 20),
        (
            f"common17 EEG-only waveform | record={recording_id} | "
            f"t={float(interval_seconds[0]):.6f}-{float(interval_seconds[1]):.6f}s | "
            f"fs={float(sampling_rate_hz):g}Hz"
        ),
        fill="black",
        font=font,
    )
    duration = float(interval_seconds[1]) - float(interval_seconds[0])
    for second in range(int(math.floor(duration)) + 1):
        x = margin_left + int(round(second / max(duration, 1e-12) * plot_width))
        draw.line((x, margin_top, x, height - margin_bottom), fill=(230, 230, 230))
        draw.text(
            (x + 2, height - margin_bottom + 8),
            f"+{second}s",
            fill=(80, 80, 80),
            font=font,
        )

    sample_count = signal.shape[1]
    edges = np.linspace(0, sample_count, plot_width + 1, dtype=np.int64)
    for channel, name in enumerate(COMMON17_CHANNELS):
        baseline_y = margin_top + channel * track_height + track_height // 2
        draw.line(
            (margin_left, baseline_y, width - margin_right, baseline_y),
            fill=(242, 242, 242),
        )
        draw.text((20, baseline_y - 6), name, fill="black", font=font)
        previous: tuple[int, int] | None = None
        for pixel in range(plot_width):
            start = int(edges[pixel])
            stop = int(edges[pixel + 1])
            if stop <= start:
                stop = min(sample_count, start + 1)
            local_valid = valid[channel, start:stop]
            if not local_valid.any():
                previous = None
                continue
            values = centered[channel, start:stop][local_valid]
            mean_value = float(np.mean(values))
            minimum = float(np.min(values))
            maximum = float(np.max(values))
            x = margin_left + pixel
            y = int(round(baseline_y - mean_value * pixels_per_uv))
            y_min = int(round(baseline_y - maximum * pixels_per_uv))
            y_max = int(round(baseline_y - minimum * pixels_per_uv))
            floor = margin_top + channel * track_height + 2
            ceiling = floor + track_height - 4
            y = min(max(y, floor), ceiling)
            y_min = min(max(y_min, floor), ceiling)
            y_max = min(max(y_max, floor), ceiling)
            draw.line((x, y_min, x, y_max), fill=(30, 80, 160))
            if previous is not None:
                draw.line((previous[0], previous[1], x, y), fill=(0, 35, 100))
            previous = (x, y)

    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=9)
    return stream.getvalue(), {
        "renderer": "Pillow-fixed-minmax-envelope-v1",
        "image_size_pixels": [width, height],
        "display_only_per_channel_median_centering": True,
        "display_robust_99pct_uv": round(robust_uv, 6),
        "pixels_per_uv": round(pixels_per_uv, 9),
        "clinical_markup_used": False,
        "annotation_text_used": False,
    }


def materialize_common17_waveform_artifacts(
    *,
    source_receipt: Mapping[str, Any],
    source_receipt_path: str | Path,
    tusz_root: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Build deterministic raw NPZ and PNG artifacts from one verified EDF."""

    verified = _verify_source_rollout_receipt(source_receipt)
    receipt_path = Path(source_receipt_path).resolve(strict=True)
    on_disk = load_verified_source_rollout_receipt(receipt_path)
    if on_disk["receipt_sha256"] != verified["receipt_sha256"]:
        raise ValueError("in-memory and on-disk source receipts differ")

    root = Path(tusz_root).resolve(strict=True)
    relative, edf_path = _safe_relative_edf(root, verified["source"]["relative_edf_path"])
    evidence = verified["event_findings_evidence"]
    support = evidence["final_variable_support"]
    interval_samples = support["interval_samples"]
    if (
        not isinstance(interval_samples, list)
        or len(interval_samples) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in interval_samples)
        or interval_samples[1] <= interval_samples[0]
    ):
        raise ValueError("verified source support lacks a valid sample interval")

    with DirectObservedCommon17EDFQueryReader(
        edf_path,
        expected_edf_sha256=str(verified["source"]["edf_sha256"]),
    ) as reader:
        if reader.sampling_rate_hz != float(evidence["acquisition"]["sampling_rate_hz"]):
            raise ValueError("EDF sampling rate differs from verified source evidence")
        if reader.recording_sample_count != int(evidence["acquisition"]["recording_sample_count"]):
            raise ValueError("EDF sample count differs from verified source evidence")
        chunk = reader(int(interval_samples[0]), int(interval_samples[1]))
        reader_audit = reader.receipt()

    samples_uv = np.ascontiguousarray(chunk.signal_volts * 1.0e6, dtype=np.float32)
    valid_mask = np.ascontiguousarray(chunk.valid_sample_mask, dtype=bool)
    if samples_uv.shape != valid_mask.shape or samples_uv.shape[0] != len(COMMON17_CHANNELS):
        raise RuntimeError("waveform builder produced an invalid common17 array")
    if not np.isfinite(samples_uv).all():
        raise ValueError("waveform builder encountered non-finite EEG samples")
    rate = float(reader_audit["source_sampling_rate_hz"])
    interval_seconds = [
        float(interval_samples[0]) / rate,
        float(interval_samples[1]) / rate,
    ]
    source_support_seconds = support["interval_recording_seconds"]
    if any(
        abs(float(left) - float(right)) > max(_TIME_TOL, 0.51 / rate)
        for left, right in zip(interval_seconds, source_support_seconds)
    ):
        raise ValueError("source support seconds and sample interval disagree")

    canonical_signal_sha = _canonical_signal_sha256(
        samples_uv=samples_uv,
        valid_sample_mask=valid_mask,
        sampling_rate_hz=rate,
        interval_samples=interval_samples,
    )
    raw_arrays = {
        "channel_order": np.asarray(COMMON17_CHANNELS, dtype="<U3"),
        "interval_samples": np.asarray(interval_samples, dtype="<i8"),
        "samples_uv": np.asarray(samples_uv, dtype="<f4"),
        "sampling_rate_hz": np.asarray([rate], dtype="<f8"),
        "valid_sample_mask": valid_mask.astype(np.uint8),
    }
    raw_bytes = _deterministic_npz(raw_arrays)
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    png_bytes, display_receipt = _render_waveform_png(
        samples_uv=samples_uv,
        valid_sample_mask=valid_mask,
        sampling_rate_hz=rate,
        interval_seconds=interval_seconds,
        recording_id=str(evidence["recording_id"]),
    )
    png_sha = hashlib.sha256(png_bytes).hexdigest()
    raw_name = f"common17_raw_{raw_sha}.npz"
    png_name = f"common17_waveform_{png_sha}.png"
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(output / raw_name, raw_bytes)
    _atomic_bytes(output / png_name, png_bytes)

    event_digest = _canonical_sha256(
        {
            "recording_id": evidence["recording_id"],
            "source_receipt_sha256": verified["receipt_sha256"],
            "canonical_signal_sha256": canonical_signal_sha,
            "interval_samples": interval_samples,
        }
    )
    event_id = f"EEG-EVENT-{event_digest[:24].upper()}"
    raw_dependency_id = f"RAWDEP-{raw_sha[:24].upper()}"
    usable_fraction = float(np.mean(valid_mask))
    body: dict[str, Any] = {
        "schema_version": WAVEFORM_MANIFEST_SCHEMA_VERSION,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "builder_id": WAVEFORM_BUILDER_ID,
        "builder_version": "v1",
        "event_id": event_id,
        "recording_id": str(evidence["recording_id"]),
        "source_binding": {
            "source_rollout_schema_version": verified["schema_version"],
            "source_receipt_content_sha256": verified["receipt_sha256"],
            "source_receipt_file_sha256": _file_sha256(receipt_path),
            "source_edf_relative_path": relative,
            "source_edf_sha256": str(verified["source"]["edf_sha256"]),
        },
        "signal_contract": {
            "channel_order": list(COMMON17_CHANNELS),
            "reference": "native_referential",
            "physical_unit": "uV",
            "sampling_rate_hz": rate,
            "interval_samples": [int(interval_samples[0]), int(interval_samples[1])],
            "interval_recording_seconds": interval_seconds,
            "sample_count_per_channel": int(samples_uv.shape[1]),
            "canonical_signal_sha256": canonical_signal_sha,
            "FZ_PZ_samples_read": False,
            "zero_fill_interpolation_or_synthesis_used": False,
        },
        "quality_summary": {
            "usable_signal_fraction": round(usable_fraction, 12),
            "invalid_signal_fraction": round(1.0 - usable_fraction, 12),
            "valid_mask_semantics": "finite_and_sustained_exact_ADC_rail_mask_v1",
        },
        "artifacts": {
            "raw_npz": {
                "relative_path": raw_name,
                "sha256": raw_sha,
                "byte_count": len(raw_bytes),
                "raw_sample_dependency_id": raw_dependency_id,
                "canonical_signal_sha256": canonical_signal_sha,
            },
            "display_png": {
                "relative_path": png_name,
                "sha256": png_sha,
                "byte_count": len(png_bytes),
                "raw_sample_dependency_id": raw_dependency_id,
                "canonical_signal_sha256": canonical_signal_sha,
                "display_receipt": display_receipt,
            },
        },
        "reader_audit": {
            "method_id": reader_audit["method_id"],
            "selected_raw_names": reader_audit["selected_raw_names"],
            "selected_edf_indices": reader_audit["selected_edf_indices"],
            "query_count": reader_audit["query_count"],
            "queried_intervals_samples": reader_audit["queried_intervals_samples"],
            "EDF_annotation_API_called": False,
            "patient_header_API_called": False,
            "target_sidecar_opened": False,
        },
        "scope_receipt": {
            "direct_common17_EEG_samples_used": True,
            "acquisition_signal_headers_used": True,
            "edf_annotations_used": False,
            "term_or_szcore_sidecar_used": False,
            "soz_or_channel_labels_used": False,
            "spreadsheet_used": False,
            "doctor_or_clinical_text_used": False,
            "patient_header_fields_used": False,
            "video_or_behavior_used": False,
            "sleep_or_provocation_used": False,
            "ecg_emg_eog_used": False,
            "llm_used": False,
        },
    }
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    _atomic_json(output / "manifest.json", body)
    return validate_common17_waveform_artifact_manifest(
        body,
        manifest_path=output / "manifest.json",
        source_receipt=verified,
        require_files=True,
    )


def validate_common17_waveform_artifact_manifest(
    payload: object,
    *,
    manifest_path: str | Path | None = None,
    source_receipt: Mapping[str, Any] | None = None,
    require_files: bool = False,
) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError("waveform artifact manifest must be an object")
    result = deepcopy(payload)
    required = {
        "schema_version",
        "receipt_sha256",
        "builder_id",
        "builder_version",
        "event_id",
        "recording_id",
        "source_binding",
        "signal_contract",
        "quality_summary",
        "artifacts",
        "reader_audit",
        "scope_receipt",
    }
    if set(result) != required:
        raise ValueError("waveform artifact manifest fields drifted")
    if result["schema_version"] != WAVEFORM_MANIFEST_SCHEMA_VERSION:
        raise ValueError("waveform artifact manifest schema drifted")
    if result["builder_id"] != WAVEFORM_BUILDER_ID or result["builder_version"] != "v1":
        raise ValueError("waveform artifact builder identity drifted")
    expected = _canonical_sha256(
        {key: value for key, value in result.items() if key != "receipt_sha256"}
    )
    if result["receipt_sha256"] != expected:
        raise ValueError("waveform artifact manifest content hash mismatch")
    contract = result["signal_contract"]
    if contract.get("channel_order") != list(COMMON17_CHANNELS):
        raise ValueError("waveform manifest is not exact common17")
    if contract.get("reference") != "native_referential" or contract.get("physical_unit") != "uV":
        raise ValueError("waveform manifest signal contract drifted")
    if contract.get("FZ_PZ_samples_read") is not False or contract.get(
        "zero_fill_interpolation_or_synthesis_used"
    ) is not False:
        raise ValueError("waveform manifest synthesized or read excluded channels")
    _sha(contract.get("canonical_signal_sha256"), "canonical signal SHA")
    rate = _finite(contract.get("sampling_rate_hz"), "sampling rate")
    interval_samples = contract.get("interval_samples")
    if (
        not isinstance(interval_samples, list)
        or len(interval_samples) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in interval_samples)
        or interval_samples[1] <= interval_samples[0]
    ):
        raise ValueError("waveform sample interval is invalid")
    if contract.get("sample_count_per_channel") != interval_samples[1] - interval_samples[0]:
        raise ValueError("waveform sample count does not close")
    interval_seconds = contract.get("interval_recording_seconds")
    if not isinstance(interval_seconds, list) or len(interval_seconds) != 2:
        raise ValueError("waveform time interval is invalid")
    expected_seconds = [interval_samples[0] / rate, interval_samples[1] / rate]
    if any(abs(float(a) - float(b)) > _TIME_TOL for a, b in zip(interval_seconds, expected_seconds)):
        raise ValueError("waveform time and sample intervals differ")
    quality = result["quality_summary"]
    usable = _finite(quality.get("usable_signal_fraction"), "usable fraction")
    invalid = _finite(quality.get("invalid_signal_fraction"), "invalid fraction")
    if not 0.0 <= usable <= 1.0 or not 0.0 <= invalid <= 1.0 or abs(usable + invalid - 1.0) > 2e-9:
        raise ValueError("waveform quality fractions do not close")

    artifacts = result["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {"raw_npz", "display_png"}:
        raise ValueError("waveform artifact roster drifted")
    raw = artifacts["raw_npz"]
    png = artifacts["display_png"]
    for name, row, suffix in (("raw_npz", raw, ".npz"), ("display_png", png, ".png")):
        relative = PurePosixPath(str(row.get("relative_path")))
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != suffix:
            raise ValueError(f"{name} artifact path is unsafe")
        _sha(row.get("sha256"), f"{name} artifact SHA")
        if row.get("canonical_signal_sha256") != contract["canonical_signal_sha256"]:
            raise ValueError(f"{name} canonical signal binding drifted")
        if not isinstance(row.get("byte_count"), int) or row["byte_count"] <= 0:
            raise ValueError(f"{name} byte count is invalid")
    if raw.get("raw_sample_dependency_id") != png.get("raw_sample_dependency_id"):
        raise ValueError("raw and display artifacts have different sample dependencies")

    scope = result["scope_receipt"]
    forbidden = {
        "edf_annotations_used",
        "term_or_szcore_sidecar_used",
        "soz_or_channel_labels_used",
        "spreadsheet_used",
        "doctor_or_clinical_text_used",
        "patient_header_fields_used",
        "video_or_behavior_used",
        "sleep_or_provocation_used",
        "ecg_emg_eog_used",
        "llm_used",
    }
    if any(scope.get(field) is not False for field in forbidden):
        raise ValueError("waveform artifact manifest crossed the EEG-only firewall")
    if source_receipt is not None:
        source = _verify_source_rollout_receipt(source_receipt)
        binding = result["source_binding"]
        if binding.get("source_receipt_content_sha256") != source["receipt_sha256"]:
            raise ValueError("waveform manifest source receipt binding drifted")
        if binding.get("source_edf_sha256") != source["source"]["edf_sha256"]:
            raise ValueError("waveform manifest source EDF binding drifted")
        if result["recording_id"] != source["event_findings_evidence"]["recording_id"]:
            raise ValueError("waveform manifest recording binding drifted")
        if interval_samples != source["event_findings_evidence"]["final_variable_support"][
            "interval_samples"
        ]:
            raise ValueError("waveform manifest support differs from source evidence")

    if require_files:
        if manifest_path is None:
            raise ValueError("manifest_path is required when artifact files are verified")
        source_path = Path(manifest_path).resolve(strict=True)
        base = source_path.parent
        raw_path = (base / str(raw["relative_path"])).resolve(strict=True)
        png_path = (base / str(png["relative_path"])).resolve(strict=True)
        raw_path.relative_to(base)
        png_path.relative_to(base)
        if _file_sha256(raw_path) != raw["sha256"] or raw_path.stat().st_size != raw["byte_count"]:
            raise ValueError("raw waveform artifact content differs from manifest")
        if _file_sha256(png_path) != png["sha256"] or png_path.stat().st_size != png["byte_count"]:
            raise ValueError("PNG waveform artifact content differs from manifest")
        with np.load(raw_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "channel_order",
                "interval_samples",
                "samples_uv",
                "sampling_rate_hz",
                "valid_sample_mask",
            }:
                raise ValueError("raw NPZ array roster drifted")
            channel_order = tuple(str(item) for item in archive["channel_order"].tolist())
            stored_interval = archive["interval_samples"].astype(np.int64).tolist()
            stored_rate = float(archive["sampling_rate_hz"].reshape(-1)[0])
            samples_uv = np.asarray(archive["samples_uv"], dtype=np.float32)
            valid_mask = np.asarray(archive["valid_sample_mask"], dtype=np.uint8)
        if channel_order != COMMON17_CHANNELS or stored_interval != interval_samples or stored_rate != rate:
            raise ValueError("raw NPZ metadata differs from manifest")
        if samples_uv.shape != (len(COMMON17_CHANNELS), contract["sample_count_per_channel"]):
            raise ValueError("raw NPZ signal shape differs from manifest")
        if valid_mask.shape != samples_uv.shape or not set(np.unique(valid_mask)).issubset({0, 1}):
            raise ValueError("raw NPZ valid mask is invalid")
        observed_sha = _canonical_signal_sha256(
            samples_uv=samples_uv,
            valid_sample_mask=valid_mask.astype(bool),
            sampling_rate_hz=rate,
            interval_samples=interval_samples,
        )
        if observed_sha != contract["canonical_signal_sha256"]:
            raise ValueError("raw NPZ canonical signal hash differs from manifest")
        with Image.open(png_path) as image:
            image.verify()
    return result


def load_verified_waveform_artifact_manifest(
    path: str | Path,
    *,
    source_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise ValueError("waveform manifest must be a regular file")
    return validate_common17_waveform_artifact_manifest(
        json.loads(source.read_text(encoding="utf-8")),
        manifest_path=source,
        source_receipt=source_receipt,
        require_files=True,
    )


def _source_dependency_sha(source: Mapping[str, Any]) -> str:
    evidence = source["event_findings_evidence"]
    return _canonical_sha256(
        {
            "domain": "verified-adaptive-raw-chunk-dependency-v1",
            "source_edf_sha256": source["source"]["edf_sha256"],
            "channel_order": list(COMMON17_CHANNELS),
            "sampling_rate_hz": evidence["acquisition"]["sampling_rate_hz"],
            "support": evidence["final_variable_support"]["interval_samples"],
            "raw_chunk_sha256": [
                row["raw_eeg_chunk_sha256"] for row in evidence["query_trace"]
            ],
        }
    )


def _baseline_interval(view: Mapping[str, Any]) -> list[float] | None:
    value = view.get(
        "/event_findings_evidence/final_evidence/robust_matched_baseline"
    )
    if not isinstance(value, Mapping) or value.get("status") != "qualified_robust_matched_baseline":
        return None
    interval = value.get("pool_interval_recording_seconds")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or float(interval[1]) <= float(interval[0])
    ):
        return None
    return [float(interval[0]), float(interval[1])]


def _global_spatial(electrodes: Sequence[str] = COMMON17_CHANNELS) -> dict[str, Any]:
    return {
        "electrodes": list(electrodes),
        "derived_lead_ids": [],
        "regions": [],
        "laterality": "not_applicable",
        "spatial_scope": "global",
    }


def _field_spatial(channels: Sequence[str]) -> dict[str, Any]:
    ordered = [channel for channel in COMMON17_CHANNELS if channel in set(channels)]
    left = {"FP1", "F7", "F3", "T7", "C3", "P7", "P3", "O1"}
    right = {"FP2", "F4", "F8", "C4", "T8", "P4", "P8", "O2"}
    selected = set(ordered)
    if selected and selected <= left:
        laterality = "left"
    elif selected and selected <= right:
        laterality = "right"
    elif selected == {"CZ"}:
        laterality = "midline"
    elif selected & left and selected & right:
        laterality = "bilateral"
    else:
        laterality = "indeterminate"
    return {
        "electrodes": ordered,
        "derived_lead_ids": [],
        "regions": [],
        "laterality": laterality,
        "spatial_scope": "electrode",
    }


def _measurement(
    *,
    value_type: str,
    value: object,
    unit: str,
    labels: Sequence[str] = (),
    baseline_relation: str,
    baseline_interval: Sequence[float] | None,
) -> dict[str, Any]:
    return {
        "value_type": value_type,
        "value": value,
        "unit_id": unit,
        "dimension_labels": list(labels),
        "baseline_relation": baseline_relation,
        "baseline_interval_seconds": (
            None if baseline_interval is None else list(baseline_interval)
        ),
    }


def _positive_row(
    *,
    rule: Mapping[str, Any],
    source_sha: str,
    waveform_id: str,
    query_interval: Sequence[float],
    evidence_interval: Sequence[float],
    resolution_seconds: float,
    spatial: Mapping[str, Any],
    measurement: Mapping[str, Any],
    usable_fraction: float,
    confidence_score: float | None,
) -> dict[str, Any]:
    registry = {
        row["metric_id"]: row
        for row in load_event_findings_eeg_only_v2_registry()["capabilities"]
    }
    capability = registry[rule["metric_id"]]
    level = (
        "direct_measurement"
        if capability["maturity"] == "replayable_measurement"
        else "algorithmic_inference"
    )
    if level == "direct_measurement":
        confidence = {
            "score": None,
            "semantics": "not_available",
            "calibration_receipt_id": None,
        }
        assertion = "observed"
        surface = "numeric_measurement_only"
    else:
        if confidence_score is None or not 0.0 <= float(confidence_score) <= 1.0:
            raise ValueError(f"{rule['rule_id']} lacks a bounded source-derived score")
        confidence = {
            "score": round(float(confidence_score), 9),
            "semantics": "uncalibrated_score",
            "calibration_receipt_id": None,
        }
        assertion = "candidate_present"
        surface = "research_candidate_only"
    return {
        "observation_id": f"OBS-{rule['rule_id']}",
        "category": rule["category"],
        "metric_id": rule["metric_id"],
        "evidence_level": level,
        "assertion_status": assertion,
        "temporal_support": {
            "query_interval_seconds": list(query_interval),
            "evidence_interval_seconds": list(evidence_interval),
            "resolution_seconds": float(resolution_seconds),
        },
        "spatial_support": dict(spatial),
        "measurement": dict(measurement),
        "quality": {
            "status": "passed" if usable_fraction >= 0.95 else "limited",
            "usable_fraction": round(float(usable_fraction), 9),
            "artifact_overlap_fraction": round(float(1.0 - usable_fraction), 9),
            "reason_codes": [] if usable_fraction >= 0.95 else ["limited_valid_sample_fraction"],
        },
        "confidence": confidence,
        "source_binding": {
            "producer_id": "ADAPTIVE-NATIVE-EVIDENCE",
            "maturity": capability["maturity"],
            "source_evidence_ids": [
                f"SRC-{source_sha[:20].upper()}-{str(rule['rule_id']).split('-')[0]}"
            ],
            "sensitivity_receipt_id": None,
        },
        "waveform_evidence_ids": [waveform_id],
        "surface_policy": surface,
        "reason_codes": ["explicit_projection_allowlist_v1"],
    }


def _not_evaluable_row(
    *,
    category: str,
    metric_id: str,
    query_interval: Sequence[float],
    resolution_seconds: float,
    reason: str,
) -> dict[str, Any]:
    registry = {
        row["metric_id"]: row
        for row in load_event_findings_eeg_only_v2_registry()["capabilities"]
    }
    capability = registry[metric_id]
    return {
        "observation_id": f"OBS-NE-{category.upper().replace('_', '-')}",
        "category": category,
        "metric_id": metric_id,
        "evidence_level": "not_evaluable",
        "assertion_status": "not_evaluable",
        "temporal_support": {
            "query_interval_seconds": list(query_interval),
            "evidence_interval_seconds": None,
            "resolution_seconds": float(resolution_seconds),
        },
        "spatial_support": {
            "electrodes": [],
            "derived_lead_ids": [],
            "regions": [],
            "laterality": "not_applicable",
            "spatial_scope": "none",
        },
        "measurement": _measurement(
            value_type="none",
            value=None,
            unit="not_applicable",
            baseline_relation="not_applicable",
            baseline_interval=None,
        ),
        "quality": {
            "status": "not_evaluable",
            "usable_fraction": 0.0,
            "artifact_overlap_fraction": 0.0,
            "reason_codes": [reason],
        },
        "confidence": {
            "score": None,
            "semantics": "not_available",
            "calibration_receipt_id": None,
        },
        "source_binding": {
            "producer_id": "CAPABILITY-REGISTRY",
            "maturity": capability["maturity"],
            "source_evidence_ids": [],
            "sensitivity_receipt_id": None,
        },
        "waveform_evidence_ids": [],
        "surface_policy": "technical_limitation_only",
        "reason_codes": [reason],
    }


def _clipped_interval(
    start: float,
    stop: float,
    query_interval: Sequence[float],
) -> list[float] | None:
    left = max(float(query_interval[0]), float(start))
    right = min(float(query_interval[1]), float(stop))
    if right <= left + _TIME_TOL:
        return None
    return [left, right]


def _max_source_posterior(view: Mapping[str, Any]) -> float | None:
    rows = view.get(
        "/event_findings_evidence/final_evidence/per_channel_evidence"
    )
    if not isinstance(rows, list) or not rows:
        return None
    values = [
        float(row["peak_algorithmic_change_posterior"])
        for row in rows
        if isinstance(row, Mapping)
        and row.get("evaluable") is True
        and isinstance(row.get("peak_algorithmic_change_posterior"), (int, float))
    ]
    return min(1.0, max(values)) if values else None


def _project_rule(
    *,
    rule: Mapping[str, Any],
    view: Mapping[str, Any],
    source_sha: str,
    waveform_id: str,
    query_interval: Sequence[float],
    resolution_seconds: float,
    usable_fraction: float,
) -> dict[str, Any] | None:
    projector = rule["projector"]
    baseline = _baseline_interval(view)
    primitives = view.get(
        "/event_findings_evidence/final_evidence/native_primitives/channel_feature_summaries"
    )

    if projector == "background_band_relative_power":
        if baseline is None or not isinstance(primitives, Mapping):
            return None
        labels: list[str] = []
        values: list[float] = []
        electrodes: list[str] = []
        for channel in COMMON17_CHANNELS:
            channel_row = primitives.get(channel)
            if not isinstance(channel_row, Mapping):
                continue
            complete = True
            local: list[float] = []
            for feature in _BAND_FEATURES:
                item = channel_row.get(feature)
                if not isinstance(item, Mapping) or item.get("available") is not True:
                    complete = False
                    break
                local.append(_finite(item.get("baseline_median"), feature))
            if complete:
                electrodes.append(channel)
                for feature, value in zip(_BAND_FEATURES, local):
                    labels.append(f"{channel}-{feature.split('_')[0]}")
                    values.append(round(value, 9))
        if not values:
            return None
        return _positive_row(
            rule=rule,
            source_sha=source_sha,
            waveform_id=waveform_id,
            query_interval=query_interval,
            evidence_interval=baseline,
            resolution_seconds=resolution_seconds,
            spatial=_global_spatial(electrodes),
            measurement=_measurement(
                value_type="vector",
                value=values,
                unit="relative_power",
                labels=labels,
                baseline_relation="absolute",
                baseline_interval=None,
            ),
            usable_fraction=usable_fraction,
            confidence_score=None,
        )

    if projector == "native_candidate_vector":
        if baseline is None or not isinstance(primitives, Mapping):
            return None
        feature = str(rule["source_feature"])
        labels: list[str] = []
        values: list[float] = []
        unit: str | None = None
        for channel in COMMON17_CHANNELS:
            channel_row = primitives.get(channel)
            item = channel_row.get(feature) if isinstance(channel_row, Mapping) else None
            if not isinstance(item, Mapping) or item.get("available") is not True:
                continue
            observed_unit = str(item.get("unit"))
            observed_unit = _SOURCE_UNIT_TO_PROFILE_UNIT.get(
                observed_unit, observed_unit
            )
            if unit is None:
                unit = observed_unit
            elif unit != observed_unit:
                raise ValueError(f"{rule['rule_id']} source units are mixed")
            labels.append(channel)
            values.append(round(_finite(item.get("candidate_peak"), feature), 9))
        if not values or unit is None:
            return None
        return _positive_row(
            rule=rule,
            source_sha=source_sha,
            waveform_id=waveform_id,
            query_interval=query_interval,
            evidence_interval=query_interval,
            resolution_seconds=resolution_seconds,
            spatial=_global_spatial(labels),
            measurement=_measurement(
                value_type="vector",
                value=values,
                unit=unit,
                labels=labels,
                baseline_relation="relative_to_local_baseline",
                baseline_interval=baseline,
            ),
            usable_fraction=usable_fraction,
            confidence_score=None,
        )

    if projector == "quantitative_change_trajectory":
        rows = view.get(
            "/event_findings_evidence/final_evidence/change_trajectory"
        )
        window = view.get(
            "/event_findings_evidence/final_evidence/window_seconds"
        )
        step = view.get("/event_findings_evidence/final_evidence/step_seconds")
        if baseline is None or not isinstance(rows, list) or not rows or window is None or step is None:
            return None
        values = [
            round(_finite(row["global_change_score"], "global change score"), 9)
            for row in rows
        ]
        labels = [f"W{index:04d}" for index in range(len(values))]
        interval = _clipped_interval(
            float(rows[0]["window_start_recording_seconds"]),
            float(rows[-1]["window_start_recording_seconds"]) + float(window),
            query_interval,
        )
        score = _max_source_posterior(view)
        if interval is None or score is None:
            return None
        return _positive_row(
            rule=rule,
            source_sha=source_sha,
            waveform_id=waveform_id,
            query_interval=query_interval,
            evidence_interval=interval,
            resolution_seconds=float(step),
            spatial=_global_spatial(),
            measurement=_measurement(
                value_type="vector",
                value=values,
                unit="score",
                labels=labels,
                baseline_relation="delta_from_baseline",
                baseline_interval=baseline,
            ),
            usable_fraction=usable_fraction,
            confidence_score=score,
        )

    if projector == "earliest_scalp_change_field_candidate":
        onset = view.get(
            "/event_findings_evidence/final_evidence/onset_candidate"
        )
        field = view.get(
            "/event_findings_evidence/final_evidence/earliest_field"
        )
        window = view.get(
            "/event_findings_evidence/final_evidence/window_seconds"
        )
        rows = view.get(
            "/event_findings_evidence/final_evidence/per_channel_evidence"
        )
        if baseline is None or not isinstance(onset, Mapping) or not isinstance(field, Mapping) or window is None:
            return None
        channels = field.get("channels")
        if not isinstance(channels, list) or not channels:
            return None
        onset_seconds = _finite(onset.get("recording_seconds"), "onset candidate time")
        interval = _clipped_interval(onset_seconds, onset_seconds + float(window), query_interval)
        if interval is None or not isinstance(rows, list):
            return None
        masses = {
            str(row.get("channel")): float(row.get("onset_spatial_posterior_mass"))
            for row in rows
            if isinstance(row, Mapping)
            and isinstance(row.get("onset_spatial_posterior_mass"), (int, float))
        }
        score = min(1.0, sum(masses.get(str(channel), 0.0) for channel in channels))
        return _positive_row(
            rule=rule,
            source_sha=source_sha,
            waveform_id=waveform_id,
            query_interval=query_interval,
            evidence_interval=interval,
            resolution_seconds=resolution_seconds,
            spatial=_field_spatial(channels),
            measurement=_measurement(
                value_type="scalar",
                value=round(onset_seconds, 9),
                unit="s",
                baseline_relation="not_applicable",
                baseline_interval=None,
            ),
            usable_fraction=usable_fraction,
            confidence_score=score,
        )

    if projector == "earliest_field_graph_contiguity":
        onset = view.get(
            "/event_findings_evidence/final_evidence/onset_candidate"
        )
        field = view.get(
            "/event_findings_evidence/final_evidence/earliest_field"
        )
        connectivity = view.get(
            "/event_findings_evidence/final_evidence/spatial_connectivity"
        )
        window = view.get(
            "/event_findings_evidence/final_evidence/window_seconds"
        )
        if baseline is None or not isinstance(onset, Mapping) or not isinstance(field, Mapping) or not isinstance(connectivity, Mapping) or window is None:
            return None
        channels = field.get("channels")
        fraction = connectivity.get("dominant_component_fraction")
        if not isinstance(channels, list) or not channels or not isinstance(fraction, (int, float)):
            return None
        onset_seconds = _finite(onset.get("recording_seconds"), "onset candidate time")
        interval = _clipped_interval(onset_seconds, onset_seconds + float(window), query_interval)
        if interval is None:
            return None
        score = min(1.0, max(0.0, float(fraction)))
        return _positive_row(
            rule=rule,
            source_sha=source_sha,
            waveform_id=waveform_id,
            query_interval=query_interval,
            evidence_interval=interval,
            resolution_seconds=resolution_seconds,
            spatial=_field_spatial(channels),
            measurement=_measurement(
                value_type="scalar",
                value=round(score, 9),
                unit="ratio",
                baseline_relation="not_applicable",
                baseline_interval=None,
            ),
            usable_fraction=usable_fraction,
            confidence_score=score,
        )

    if projector == "return_to_baseline_candidate":
        onset = view.get(
            "/event_findings_evidence/final_evidence/onset_candidate"
        )
        evolution = view.get(
            "/event_findings_evidence/final_evidence/evolution"
        )
        if baseline is None or not isinstance(onset, Mapping) or not isinstance(evolution, Mapping):
            return None
        relative = evolution.get("candidate_return_relative_to_onset_seconds")
        absolute = evolution.get("candidate_return_to_baseline_recording_seconds")
        similarity = evolution.get("posterior_saturation_similarity")
        if not isinstance(relative, (int, float)) or not isinstance(absolute, (int, float)) or not isinstance(similarity, (int, float)):
            return None
        onset_seconds = _finite(onset.get("recording_seconds"), "onset candidate time")
        interval = _clipped_interval(onset_seconds, float(absolute), query_interval)
        if interval is None:
            return None
        score = min(1.0, max(0.0, float(similarity)))
        return _positive_row(
            rule=rule,
            source_sha=source_sha,
            waveform_id=waveform_id,
            query_interval=query_interval,
            evidence_interval=interval,
            resolution_seconds=resolution_seconds,
            spatial=_global_spatial(),
            measurement=_measurement(
                value_type="scalar",
                value=round(float(relative), 9),
                unit="s",
                baseline_relation="not_applicable",
                baseline_interval=None,
            ),
            usable_fraction=usable_fraction,
            confidence_score=score,
        )

    if projector == "reference_distribution_js_similarity":
        stability = view.get(
            "/event_findings_evidence/final_evidence/reference_stability"
        )
        if baseline is None or not isinstance(stability, Mapping):
            return None
        similarity = stability.get("minimum_similarity")
        if not isinstance(similarity, (int, float)):
            return None
        return _positive_row(
            rule=rule,
            source_sha=source_sha,
            waveform_id=waveform_id,
            query_interval=query_interval,
            evidence_interval=query_interval,
            resolution_seconds=resolution_seconds,
            spatial=_global_spatial(),
            measurement=_measurement(
                value_type="scalar",
                value=round(float(similarity), 9),
                unit="ratio",
                baseline_relation="not_applicable",
                baseline_interval=None,
            ),
            usable_fraction=usable_fraction,
            confidence_score=None,
        )
    raise ValueError(f"unhandled projector: {projector}")


def materialize_event_findings_eeg_only_v2(
    *,
    source_receipt: Mapping[str, Any],
    waveform_manifest: Mapping[str, Any] | None = None,
    waveform_manifest_path: str | Path | None = None,
    allowlist_path: str | Path = _ALLOWLIST_PATH,
) -> dict[str, Any]:
    """Project one verified source receipt into the uniform v2 profile."""

    source = _verify_source_rollout_receipt(source_receipt)
    allowlist = load_projection_allowlist(allowlist_path)
    view = _projection_view(source, allowlist)
    evidence = source["event_findings_evidence"]
    rate = float(evidence["acquisition"]["sampling_rate_hz"])
    source_sha = str(evidence["receipt_sha256"])
    support = evidence["final_variable_support"]
    query_interval = [float(value) for value in support["interval_recording_seconds"]]
    resolution = max(1.0 / rate, float(evidence["policy"].get("step_seconds", 0.5)))

    verified_waveform: dict[str, Any] | None = None
    waveform_evidence: list[dict[str, Any]] = []
    waveform_id: str | None = None
    usable_fraction = 0.0
    if waveform_manifest is not None:
        if waveform_manifest_path is None:
            raise ValueError("waveform_manifest_path is required for real artifact verification")
        verified_waveform = validate_common17_waveform_artifact_manifest(
            waveform_manifest,
            manifest_path=waveform_manifest_path,
            source_receipt=source,
            require_files=True,
        )
        contract = verified_waveform["signal_contract"]
        canonical_signal_sha = str(contract["canonical_signal_sha256"])
        waveform_id = f"WF-{verified_waveform['receipt_sha256'][:24].upper()}"
        usable_fraction = float(
            verified_waveform["quality_summary"]["usable_signal_fraction"]
        )
        png = verified_waveform["artifacts"]["display_png"]
        manifest_base = Path(waveform_manifest_path).resolve(strict=True).parent
        png_path = (manifest_base / str(png["relative_path"])).resolve(strict=True)
        waveform_evidence.append(
            {
                "waveform_evidence_id": waveform_id,
                "canonical_signal_sha256": canonical_signal_sha,
                "interval_seconds": list(contract["interval_recording_seconds"]),
                "electrodes": list(COMMON17_CHANNELS),
                "derived_lead_ids": [],
                "reference": "native_referential",
                "sampling_rate_hz": rate,
                "artifact_uri": str(png_path),
                "artifact_sha256": str(png["sha256"]),
                "raw_sample_dependency_id": str(png["raw_sample_dependency_id"]),
            }
        )
    else:
        canonical_signal_sha = _source_dependency_sha(source)

    observations_by_category: dict[str, dict[str, Any]] = {}
    if verified_waveform is not None and waveform_id is not None:
        for rule in allowlist["projection_rules"]:
            row = _project_rule(
                rule=rule,
                view=view,
                source_sha=source_sha,
                waveform_id=waveform_id,
                query_interval=query_interval,
                resolution_seconds=resolution,
                usable_fraction=usable_fraction,
            )
            if row is not None:
                observations_by_category[str(rule["category"])] = row

    for category in REQUIRED_CATEGORIES:
        if category in observations_by_category:
            continue
        if verified_waveform is None:
            reason = "waveform_artifact_unmaterialized"
        elif category not in {rule["category"] for rule in allowlist["projection_rules"]}:
            reason = "no_explicit_projection_allowlist_mapping"
        else:
            reason = "source_metric_not_available_or_unqualified"
        observations_by_category[category] = _not_evaluable_row(
            category=category,
            metric_id=str(allowlist["category_fallback_metrics"][category]),
            query_interval=query_interval,
            resolution_seconds=resolution,
            reason=reason,
        )

    registry_sha = _file_sha256(_REGISTRY_PATH)
    module_sha = _file_sha256(Path(__file__))
    producers = [
        {
            "producer_id": "ADAPTIVE-NATIVE-EVIDENCE",
            "version": "v1",
            "artifact_sha256": source_sha,
        },
        {
            "producer_id": "CAPABILITY-REGISTRY",
            "version": "v2",
            "artifact_sha256": registry_sha,
        },
        {
            "producer_id": "EEG-ONLY-V2-PROJECTION-MATERIALIZER",
            "version": "v1",
            "artifact_sha256": module_sha,
        },
    ]
    if verified_waveform is not None:
        producers.append(
            {
                "producer_id": "WAVEFORM-ARTIFACT-BUILDER",
                "version": "v1",
                "artifact_sha256": str(verified_waveform["receipt_sha256"]),
            }
        )

    onset = evidence["final_evidence"].get("onset_candidate")
    if isinstance(onset, Mapping) and isinstance(onset.get("recording_seconds"), (int, float)):
        navigation_anchor = float(onset["recording_seconds"])
        anchor_semantics = "signal_reestimated_candidate"
    else:
        navigation_anchor = 0.5 * (query_interval[0] + query_interval[1])
        anchor_semantics = "technical_support_midpoint_navigation_only"
    navigation_anchor = min(max(navigation_anchor, query_interval[0]), query_interval[1])
    profile_event_id = (
        str(verified_waveform["event_id"])
        if verified_waveform is not None
        else f"EEG-EVENT-{_canonical_sha256({'source': source_sha, 'support': query_interval})[:24].upper()}"
    )
    derived_leads = [
        {"lead_id": f"{left}-{right}", "anode": left, "cathode": right}
        for left, right in COMMON17_TCP_EDGES
    ]
    profile: dict[str, Any] = {
        "schema_version": EVENT_FINDINGS_EEG_ONLY_V2_SCHEMA_VERSION,
        "event_id": profile_event_id,
        "record_id": str(evidence["recording_id"]),
        "registry_id": EVENT_FINDINGS_EEG_ONLY_V2_REGISTRY_ID,
        "signal_contract": {
            "electrode_space": "common17",
            "observed_electrodes": list(COMMON17_CHANNELS),
            "derived_leads": derived_leads,
            "sampling_rate_hz": rate,
            "physical_unit": "uV",
            "analysis_references": [
                "native_referential",
                "common_average",
                "tcp_bipolar",
            ],
            "highpass_hz": None,
            "lowpass_hz": None,
            "line_frequency_hz": None,
            "missing_electrodes": ["FZ", "PZ"],
            "imputed_electrodes": [],
        },
        "analysis_window": {
            "recording_duration_seconds": float(
                evidence["acquisition"]["recording_duration_seconds"]
            ),
            "navigation_anchor_seconds": navigation_anchor,
            "query_interval_seconds": query_interval,
            "left_censored": query_interval[0] <= _TIME_TOL,
            "right_censored": (
                query_interval[1]
                >= float(evidence["acquisition"]["recording_duration_seconds"])
                - _TIME_TOL
            ),
            "anchor_semantics": anchor_semantics,
        },
        "provenance": {
            "canonical_signal_sha256": canonical_signal_sha,
            "preprocess_receipt_id": f"COMMON17-NATIVE-{source['receipt_sha256'][:20].upper()}",
            "producers": producers,
            "inference_firewall": {
                "eeg_samples_used": True,
                "edf_annotations_used": False,
                "spreadsheet_used": False,
                "doctor_text_used": False,
                "clinical_metadata_used": False,
                "video_or_behavior_used": False,
                "sleep_stage_used": False,
                "provocation_used": False,
                "ecg_emg_eog_used": False,
                "llm_used": False,
            },
            "prediction_frozen_before_reference_join": True,
            "source_event_findings_v2_id": None,
        },
        "category_coverage": list(REQUIRED_CATEGORIES),
        "observations": [observations_by_category[item] for item in REQUIRED_CATEGORIES],
        "waveform_evidence": waveform_evidence,
        "evaluation_binding": {
            "reference_join_status": "not_joined",
            "join_stage": "none",
            "reference_used_for_inference": False,
            "prediction_payload_sha256": "0" * 64,
            "reference_artifact_sha256": None,
            "szcore_event_match_id": None,
        },
        "limitations": [
            {
                "code": "scalp_visible_not_cortical_soz",
                "scope": "clinical_claim",
                "text_zh": "头皮可见最早变化候选不等同于皮层 SOZ、EZ 或手术靶点。",
            },
            {
                "code": "research_proxy_not_clinical_term",
                "scope": "finding",
                "text_zh": "研究代理仅保留候选语义，不升级为临床 onset、演变或传播。",
            },
            {
                "code": "legacy_support_smoke_not_detector_benchmark",
                "scope": "evaluation",
                "text_zh": "该冻结支持范围仅用于真实 EEG 物化冒烟，不构成检测器性能证据。",
            },
        ],
    }
    if verified_waveform is None:
        profile["limitations"].append(
            {
                "code": "waveform_artifact_unmaterialized",
                "scope": "signal",
                "text_zh": "未提供可校验的真实波形制品，所有观察均按不可评价输出。",
            }
        )
    freeze_payload = deepcopy(profile)
    freeze_payload.pop("evaluation_binding")
    profile["evaluation_binding"]["prediction_payload_sha256"] = _canonical_sha256(
        freeze_payload
    )
    return validate_event_findings_eeg_only_v2(profile)


__all__ = [
    "PROJECTION_ALLOWLIST_ID",
    "PROJECTION_ALLOWLIST_SCHEMA_VERSION",
    "PROJECTION_MATERIALIZER_ID",
    "WAVEFORM_BUILDER_ID",
    "WAVEFORM_MANIFEST_SCHEMA_VERSION",
    "load_projection_allowlist",
    "load_verified_source_rollout_receipt",
    "load_verified_waveform_artifact_manifest",
    "materialize_common17_waveform_artifacts",
    "materialize_event_findings_eeg_only_v2",
    "validate_common17_waveform_artifact_manifest",
]
