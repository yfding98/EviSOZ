"""Target-blind real-EDF adapter for common-17 adaptive event evidence.

The adapter is deliberately narrow.  It reads only the 17 directly observed
scalp electrodes required by :mod:`adaptive_native_evidence_common17`, in the
exact intervals requested by that module.  It never opens an EDF annotation
channel, TERM/CSV sidecar, spreadsheet, patient header field, clinical text,
or SOZ target.  A frozen TERM ``seiz`` onset may appear in the *input manifest*
solely as a navigation anchor; it is never supplied to a feature computation
as a target or label.

The signal path preserves the EDF native sample rate and referential voltage.
No zero fill, interpolation, montage synthesis, filtering, or resampling is
performed in this rollout adapter.  Cross-reference measurements are produced
inside the adaptive evidence core from the same directly observed samples.
This is a real-signal feasibility rollout, not a detector benchmark and not a
claim that adaptive support is superior to any fixed-window comparator.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Final, Mapping, Sequence

import numpy as np
import pyedflib

from src.soz.geometry import normalize_electrode_name

from .adaptive_native_evidence_common17 import (
    COMMON17_CHANNELS,
    AdaptiveNativeEvidencePolicy,
    DEFAULT_ADAPTIVE_NATIVE_EVIDENCE_POLICY,
    NativeEEGQueryChunk,
    materialize_common17_adaptive_native_event_evidence,
    validate_common17_adaptive_native_event_evidence,
)


TUSZ_REAL_EDF_ADAPTIVE_MANIFEST_SCHEMA: Final[str] = (
    "clinical_eeg_tusz_real_edf_adaptive_findings_manifest_v1"
)
TUSZ_REAL_EDF_ADAPTIVE_ROLLOUT_SCHEMA: Final[str] = (
    "clinical_eeg_tusz_real_edf_adaptive_findings_rollout_v1"
)
TUSZ_REAL_EDF_ADAPTIVE_COHORT_RECEIPT_SCHEMA: Final[str] = (
    "clinical_eeg_tusz_real_edf_adaptive_findings_cohort_receipt_v1"
)
TUSZ_REAL_EDF_ADAPTER_METHOD_ID: Final[str] = (
    "TUSZ-DIRECT-OBSERVED-COMMON17-NATIVE-QUERY-READER-V1"
)

_ENTRY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "rollout_id",
        "event_id",
        "recording_id",
        "official_split",
        "relative_edf_path",
        "edf_sha256",
        "navigation_anchor_recording_seconds",
        "selection_reference_duration_seconds",
        "selection_duration_stratum",
        "selection_roles",
        "expected_source_sampling_rate_hz",
        "expected_recording_sample_count",
        "expected_FZ_PZ_observation_state",
    }
)
_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "cohort_id",
        "common17_channel_order",
        "selection_contract",
        "entries",
    }
)
_ALLOWED_ROLES: Final[frozenset[str]] = frozenset(
    {
        "short_reference_duration",
        "medium_reference_duration",
        "long_reference_duration",
        "same_recording_multi_event",
        "recording_start_censor",
        "recording_stop_censor",
        "natural_FZ_PZ_absence",
        "non_256_hz",
    }
)
_ALLOWED_MIDLINE_STATES: Final[frozenset[str]] = frozenset(
    {"both_observed_but_excluded", "both_naturally_absent"}
)
_UNIT_TO_VOLTS: Final[dict[str, float]] = {
    "v": 1.0,
    "mv": 1.0e-3,
    "uv": 1.0e-6,
}
_SCOPE_RECEIPT: Final[dict[str, object]] = {
    "edf_signal_header_allowlist_used": True,
    "direct_common17_EEG_samples_used": True,
    "acquisition_parameters_used": True,
    "eeg_derived_ADC_rail_QC_used": True,
    "frozen_TERM_seiz_onset_used_for_navigation_only": True,
    "TERM_or_other_target_sidecar_opened_at_runtime": False,
    "EDF_annotations_opened": False,
    "SOZ_or_channel_target_opened": False,
    "clinical_text_or_spreadsheet_opened": False,
    "patient_header_fields_opened": False,
    "non_common17_signal_samples_read": False,
    "FZ_or_PZ_samples_read": False,
    "zero_fill_interpolation_or_montage_synthesis_used": False,
    "feature_threshold_training_used": False,
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed identifier")
    if len(value) > 180 or any(token in value for token in ("/", "\\")):
        raise ValueError(f"{field} is not a safe identifier")
    return value


def _finite(value: object, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return result


def _unit_scale(value: object) -> float:
    normalized = str(value).strip().lower().replace("µ", "u").replace("μ", "u")
    try:
        return _UNIT_TO_VOLTS[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported EDF EEG physical unit: {value!r}") from error


def _safe_source_path(root: Path, relative: object) -> tuple[str, Path]:
    candidate = PurePosixPath(str(relative))
    if candidate.is_absolute() or ".." in candidate.parts or candidate.suffix.lower() != ".edf":
        raise ValueError("relative_edf_path must be a safe relative EDF path")
    source = root.joinpath(*candidate.parts).resolve(strict=True)
    source.relative_to(root)
    return candidate.as_posix(), source


def _run_mask(values: np.ndarray, minimum_run_samples: int) -> np.ndarray:
    """Return samples belonging to true runs of at least ``minimum`` length."""

    flags = np.asarray(values, dtype=bool)
    result = np.zeros(flags.shape, dtype=bool)
    if not flags.any():
        return result
    padded = np.pad(flags.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    for start, stop in changes.reshape(-1, 2):
        if stop - start >= minimum_run_samples:
            result[start:stop] = True
    return result


class DirectObservedCommon17EDFQueryReader:
    """Incremental, exact-interval reader for native common-17 EDF EEG."""

    def __init__(
        self,
        edf_path: str | Path,
        *,
        expected_edf_sha256: str | None = None,
        reader_factory: Callable[[str], Any] = pyedflib.EdfReader,
        verify_file_sha256: bool = True,
    ) -> None:
        self.path = Path(edf_path).resolve(strict=True)
        if expected_edf_sha256 is not None and (
            not isinstance(expected_edf_sha256, str)
            or len(expected_edf_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_edf_sha256)
        ):
            raise ValueError("expected_edf_sha256 must be a lowercase SHA-256")
        self.edf_sha256 = (
            _file_sha256(self.path) if verify_file_sha256 or expected_edf_sha256 else None
        )
        if expected_edf_sha256 is not None and self.edf_sha256 != expected_edf_sha256:
            raise ValueError("source EDF SHA-256 differs from the frozen manifest")
        self._reader = reader_factory(str(self.path))
        self._closed = False
        self._query_calls: list[dict[str, object]] = []
        try:
            labels = tuple(str(value).strip() for value in self._reader.getSignalLabels())
            candidates: dict[str, list[int]] = {channel: [] for channel in COMMON17_CHANNELS}
            for index, label in enumerate(labels):
                canonical = normalize_electrode_name(label)
                if canonical in candidates:
                    candidates[canonical].append(index)
            missing = [channel for channel, rows in candidates.items() if not rows]
            duplicates = {
                channel: [labels[index] for index in rows]
                for channel, rows in candidates.items()
                if len(rows) > 1
            }
            if missing or duplicates:
                raise ValueError(
                    "EDF lacks unambiguous directly observed common17 support; "
                    f"missing={missing}, duplicates={duplicates}"
                )
            self.selected_indices = tuple(candidates[channel][0] for channel in COMMON17_CHANNELS)
            self.selected_raw_names = tuple(labels[index] for index in self.selected_indices)
            references = tuple(
                "REF" if label.upper().replace("_", "-").endswith("-REF") else ""
                for label in self.selected_raw_names
            )
            if any(reference != "REF" for reference in references):
                raise ValueError("v1 real rollout requires uniform directly observed -REF channels")
            rates = tuple(
                float(self._reader.getSampleFrequency(index)) for index in self.selected_indices
            )
            if any(not math.isfinite(value) or value < 10.0 for value in rates) or len(set(rates)) != 1:
                raise ValueError("common17 source sampling rates are invalid or mixed")
            self.sampling_rate_hz = rates[0]
            all_counts = self._reader.getNSamples()
            counts = tuple(int(all_counts[index]) for index in self.selected_indices)
            if any(value < int(round(2.0 * self.sampling_rate_hz)) for value in counts) or len(set(counts)) != 1:
                raise ValueError("common17 source sample counts are invalid or mixed")
            self.recording_sample_count = counts[0]
            self.recording_duration_seconds = self.recording_sample_count / self.sampling_rate_hz
            self._unit_scales = np.asarray(
                [_unit_scale(self._reader.getPhysicalDimension(index)) for index in self.selected_indices],
                dtype=np.float64,
            )
            self._physical_min_volts = np.asarray(
                [
                    float(self._reader.getPhysicalMinimum(index)) * self._unit_scales[row]
                    for row, index in enumerate(self.selected_indices)
                ],
                dtype=np.float64,
            )
            self._physical_max_volts = np.asarray(
                [
                    float(self._reader.getPhysicalMaximum(index)) * self._unit_scales[row]
                    for row, index in enumerate(self.selected_indices)
                ],
                dtype=np.float64,
            )
            canonical_all = {normalize_electrode_name(label) for label in labels}
            self.fz_pz_observation_state = (
                "both_observed_but_excluded"
                if {"FZ", "PZ"} <= canonical_all
                else "both_naturally_absent"
                if not ({"FZ", "PZ"} & canonical_all)
                else "partial_midline_observation"
            )
        except Exception:
            self._reader.close()
            self._closed = True
            raise

    def __enter__(self) -> "DirectObservedCommon17EDFQueryReader":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    @property
    def calls(self) -> tuple[dict[str, object], ...]:
        return tuple(deepcopy(self._query_calls))

    def close(self) -> None:
        if not self._closed:
            self._reader.close()
            self._closed = True

    def __call__(self, start_sample: int, stop_sample: int) -> NativeEEGQueryChunk:
        if self._closed:
            raise RuntimeError("EDF query reader is closed")
        if (
            isinstance(start_sample, bool)
            or isinstance(stop_sample, bool)
            or not isinstance(start_sample, int)
            or not isinstance(stop_sample, int)
            or not 0 <= start_sample < stop_sample <= self.recording_sample_count
        ):
            raise ValueError("EDF query interval is outside the common17 recording")
        count = stop_sample - start_sample
        physical = np.stack(
            [
                np.asarray(self._reader.readSignal(index, start_sample, count), dtype=np.float64)
                for index in self.selected_indices
            ]
        )
        if physical.shape != (len(COMMON17_CHANNELS), count):
            raise RuntimeError("EDF returned an unexpected common17 query shape")
        signal = np.ascontiguousarray(physical * self._unit_scales[:, None])
        valid = np.isfinite(signal)
        # Only sustained exact ADC-rail occupancy is censored.  The 100-ms run
        # is an acquisition-QC engineering rule, not a clinical amplitude norm.
        minimum_rail_run = max(2, int(math.ceil(0.100 * self.sampling_rate_hz)))
        for channel in range(len(COMMON17_CHANNELS)):
            span = max(
                self._physical_max_volts[channel] - self._physical_min_volts[channel],
                1.0e-12,
            )
            tolerance = max(span * 1.0e-9, 1.0e-12)
            at_rail = np.isclose(
                signal[channel], self._physical_min_volts[channel], rtol=0.0, atol=tolerance
            ) | np.isclose(
                signal[channel], self._physical_max_volts[channel], rtol=0.0, atol=tolerance
            )
            valid[channel, _run_mask(at_rail, minimum_rail_run)] = False
        self._query_calls.append(
            {
                "query_index": len(self._query_calls),
                "interval_samples": [start_sample, stop_sample],
                "sample_count_per_channel": count,
                "readSignal_call_count": len(COMMON17_CHANNELS),
                "selected_edf_indices": list(self.selected_indices),
                "non_common17_signal_read_count": 0,
                "invalid_sample_fraction": round(float(1.0 - np.mean(valid)), 9),
            }
        )
        return NativeEEGQueryChunk(signal_volts=signal, valid_sample_mask=valid)

    def receipt(self) -> dict[str, object]:
        if not self._query_calls:
            raise ValueError("reader receipt requested before any EEG query")
        unique_intervals = [row["interval_samples"] for row in self._query_calls]
        total_samples = sum(int(row["sample_count_per_channel"]) for row in self._query_calls)
        return {
            "method_id": TUSZ_REAL_EDF_ADAPTER_METHOD_ID,
            "source_edf_sha256": self.edf_sha256,
            "common17_channel_order": list(COMMON17_CHANNELS),
            "selected_raw_names": list(self.selected_raw_names),
            "selected_edf_indices": list(self.selected_indices),
            "source_reference": "uniform_REF",
            "source_sampling_rate_hz": self.sampling_rate_hz,
            "recording_sample_count": self.recording_sample_count,
            "recording_duration_seconds": round(self.recording_duration_seconds, 6),
            "FZ_PZ_observation_state": self.fz_pz_observation_state,
            "FZ_PZ_samples_read": False,
            "non_common17_signal_samples_read": False,
            "EDF_annotation_API_called": False,
            "patient_header_API_called": False,
            "target_sidecar_opened": False,
            "preprocessing": {
                "physical_unit_conversion_to_volts": True,
                "native_sample_rate_preserved": True,
                "filtering_used": False,
                "resampling_used": False,
                "re_reference_before_core_used": False,
                "zero_fill_or_interpolation_used": False,
                "sample_QC": "finite_and_sustained_exact_ADC_rail_mask_v1",
                "sample_QC_threshold_training_used": False,
            },
            "query_count": len(self._query_calls),
            "queried_intervals_samples": deepcopy(unique_intervals),
            "total_unique_physical_samples_per_channel": total_samples,
            "readSignal_call_count": len(COMMON17_CHANNELS) * len(self._query_calls),
            "full_recording_preloaded": False,
        }


def validate_tusz_real_edf_adaptive_manifest(payload: object) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != _MANIFEST_FIELDS:
        raise ValueError("real-EDF adaptive manifest fields drifted")
    manifest = deepcopy(payload)
    if manifest["schema_version"] != TUSZ_REAL_EDF_ADAPTIVE_MANIFEST_SCHEMA:
        raise ValueError("real-EDF adaptive manifest schema drifted")
    _identifier(manifest["cohort_id"], "cohort_id")
    if manifest["common17_channel_order"] != list(COMMON17_CHANNELS):
        raise ValueError("real-EDF adaptive manifest is not exact common17")
    selection = manifest["selection_contract"]
    expected_selection = {
        "official_split": "dev_only",
        "anchor_source": "frozen_TERM_seiz_reference_selection_only",
        "reference_duration_used_for_offline_stratification_only": True,
        "runtime_target_sidecar_access": False,
        "anchor_is_not_channel_target_or_feature": True,
        "feature_or_stop_threshold_training_on_this_cohort": False,
        "clinical_text_or_SOZ_label_used": False,
    }
    if selection != expected_selection:
        raise ValueError("real-EDF adaptive selection firewall drifted")
    entries = manifest["entries"]
    if not isinstance(entries, list) or len(entries) < 4:
        raise ValueError("real-EDF adaptive cohort is too small")
    seen: set[str] = set()
    roles: set[str] = set()
    recording_counts: dict[str, int] = {}
    for entry in entries:
        if type(entry) is not dict or set(entry) != _ENTRY_FIELDS:
            raise ValueError("real-EDF adaptive entry fields drifted")
        rollout = _identifier(entry["rollout_id"], "rollout_id")
        if rollout in seen:
            raise ValueError("real-EDF adaptive rollout IDs are duplicated")
        seen.add(rollout)
        _identifier(entry["event_id"], "event_id")
        recording = _identifier(entry["recording_id"], "recording_id")
        recording_counts[recording] = recording_counts.get(recording, 0) + 1
        if entry["official_split"] != "dev":
            raise ValueError("real-EDF adaptive frozen cohort must remain dev-only")
        relative = PurePosixPath(str(entry["relative_edf_path"]))
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".edf":
            raise ValueError("real-EDF adaptive source path is unsafe")
        digest = entry["edf_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("real-EDF adaptive source hash is invalid")
        _finite(entry["navigation_anchor_recording_seconds"], "navigation anchor")
        duration = _finite(entry["selection_reference_duration_seconds"], "selection duration")
        if duration <= 0.0 or entry["selection_duration_stratum"] not in {"short", "medium", "long"}:
            raise ValueError("real-EDF adaptive selection duration stratum is invalid")
        entry_roles = entry["selection_roles"]
        if not isinstance(entry_roles, list) or not entry_roles or not set(entry_roles) <= _ALLOWED_ROLES:
            raise ValueError("real-EDF adaptive selection roles are invalid")
        roles.update(entry_roles)
        rate = _finite(entry["expected_source_sampling_rate_hz"], "expected rate", minimum=10.0)
        samples = entry["expected_recording_sample_count"]
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < int(rate * 2):
            raise ValueError("real-EDF adaptive expected sample count is invalid")
        if entry["expected_FZ_PZ_observation_state"] not in _ALLOWED_MIDLINE_STATES:
            raise ValueError("real-EDF adaptive expected midline state is invalid")
    required_roles = {
        "short_reference_duration",
        "medium_reference_duration",
        "long_reference_duration",
        "same_recording_multi_event",
        "recording_start_censor",
        "recording_stop_censor",
        "natural_FZ_PZ_absence",
        "non_256_hz",
    }
    if not required_roles <= roles or max(recording_counts.values()) < 2:
        raise ValueError("real-EDF adaptive cohort does not cover the frozen strata")
    return manifest


def load_tusz_real_edf_adaptive_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    return validate_tusz_real_edf_adaptive_manifest(value)


def materialize_tusz_real_edf_adaptive_entry(
    *,
    entry: Mapping[str, object],
    tusz_root: str | Path,
    manifest_sha256: str,
    policy: AdaptiveNativeEvidencePolicy = DEFAULT_ADAPTIVE_NATIVE_EVIDENCE_POLICY,
    reader_factory: Callable[[str], Any] = pyedflib.EdfReader,
    verify_file_sha256: bool = True,
) -> dict[str, Any]:
    root = Path(tusz_root).resolve(strict=True)
    relative, source = _safe_source_path(root, entry["relative_edf_path"])
    with DirectObservedCommon17EDFQueryReader(
        source,
        expected_edf_sha256=str(entry["edf_sha256"]),
        reader_factory=reader_factory,
        verify_file_sha256=verify_file_sha256,
    ) as reader:
        expected_rate = float(entry["expected_source_sampling_rate_hz"])
        if abs(reader.sampling_rate_hz - expected_rate) > 1.0e-9:
            raise ValueError("source sampling rate differs from frozen manifest")
        if reader.recording_sample_count != int(entry["expected_recording_sample_count"]):
            raise ValueError("recording sample count differs from frozen manifest")
        if reader.fz_pz_observation_state != entry["expected_FZ_PZ_observation_state"]:
            raise ValueError("source FZ/PZ observation state differs from frozen manifest")
        evidence = materialize_common17_adaptive_native_event_evidence(
            event_id=str(entry["event_id"]),
            recording_id=str(entry["recording_id"]),
            navigation_anchor_recording_seconds=float(
                entry["navigation_anchor_recording_seconds"]
            ),
            sampling_rate_hz=reader.sampling_rate_hz,
            recording_sample_count=reader.recording_sample_count,
            query_reader=reader,
            policy=policy,
        )
        reader_receipt = reader.receipt()
    validate_common17_adaptive_native_event_evidence(evidence)
    trace_intervals = [row["action"]["interval_samples"] for row in evidence["query_trace"]]
    if reader_receipt["queried_intervals_samples"] != trace_intervals:
        raise RuntimeError("EDF reader calls differ from the adaptive query trace")
    if reader_receipt["total_unique_physical_samples_per_channel"] != evidence[
        "final_variable_support"
    ]["unique_physical_samples_per_channel"]:
        raise RuntimeError("EDF reader sample ledger does not close")
    body: dict[str, Any] = {
        "schema_version": TUSZ_REAL_EDF_ADAPTIVE_ROLLOUT_SCHEMA,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "rollout_id": str(entry["rollout_id"]),
        "manifest_sha256": manifest_sha256,
        "source": {
            "official_split": "dev",
            "relative_edf_path": relative,
            "edf_sha256": str(entry["edf_sha256"]),
            "recording_id": str(entry["recording_id"]),
            "event_id": str(entry["event_id"]),
        },
        "selection_only": {
            "navigation_anchor_recording_seconds": float(
                entry["navigation_anchor_recording_seconds"]
            ),
            "reference_duration_seconds": float(
                entry["selection_reference_duration_seconds"]
            ),
            "duration_stratum": str(entry["selection_duration_stratum"]),
            "roles": list(entry["selection_roles"]),
            "passed_to_feature_extractor": False,
            "passed_to_stop_policy": False,
        },
        "reader_receipt": reader_receipt,
        "event_findings_evidence": evidence,
        "scope_receipt": deepcopy(_SCOPE_RECEIPT),
        "claim_limits": {
            "real_EEG_rollout_completed": True,
            "detector_performance_measured": False,
            "SOZ_accuracy_measured": False,
            "adaptive_beats_fixed_window_claim_authorized": False,
            "protected_clinical_terms_authorized": False,
            "scalp_change_candidate_is_diagnosis": False,
        },
    }
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    return body


def summarize_tusz_real_edf_adaptive_rollouts(
    *, manifest_sha256: str, rollouts: Sequence[Mapping[str, object]]
) -> dict[str, Any]:
    if not rollouts:
        raise ValueError("cannot summarize an empty real-EDF adaptive rollout")
    supports: list[float] = []
    left_queries = 0
    right_queries = 0
    censor_counts: dict[str, int] = {}
    statuses: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    for rollout in rollouts:
        evidence = rollout["event_findings_evidence"]
        support = evidence["final_variable_support"]
        interval = support["interval_recording_seconds"]
        seconds = float(interval[1]) - float(interval[0])
        supports.append(seconds)
        sides = [row["action"]["side"] for row in evidence["query_trace"]]
        left = sides.count("left")
        right = sides.count("right")
        left_queries += left
        right_queries += right
        status = str(evidence["status"])
        statuses[status] = statuses.get(status, 0) + 1
        typed: list[str] = []
        for side in ("left", "right"):
            closure = support["side_closure"][side]
            for reason in closure["reason_codes"]:
                censor_counts[reason] = censor_counts.get(reason, 0) + 1
                typed.append(f"{side}:{reason}")
        rows.append(
            {
                "rollout_id": rollout["rollout_id"],
                "status": status,
                "support_seconds": round(seconds, 6),
                "left_query_count": left,
                "right_query_count": right,
                "typed_stops": typed,
                "query_count": len(evidence["query_trace"]),
                "left_extent_seconds": support["left_extent_seconds"],
                "right_extent_seconds": support["right_extent_seconds"],
            }
        )
    body: dict[str, Any] = {
        "schema_version": TUSZ_REAL_EDF_ADAPTIVE_COHORT_RECEIPT_SCHEMA,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "manifest_sha256": manifest_sha256,
        "rollout_count": len(rollouts),
        "summary": {
            "status_counts": statuses,
            "support_seconds": {
                "minimum": round(min(supports), 6),
                "median": round(float(np.median(supports)), 6),
                "maximum": round(max(supports), 6),
                "total": round(sum(supports), 6),
            },
            "left_query_count": left_queries,
            "right_query_count": right_queries,
            "typed_stop_counts": censor_counts,
        },
        "rollouts": rows,
        "scope_receipt": deepcopy(_SCOPE_RECEIPT),
        "interpretation": {
            "real_EDF_incremental_retrieval_demonstrated": True,
            "fixed_window_comparator_run": False,
            "adaptive_superiority_claim_authorized": False,
            "clinical_or_SOZ_accuracy_claim_authorized": False,
        },
    }
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    return body


__all__ = [
    "TUSZ_REAL_EDF_ADAPTIVE_MANIFEST_SCHEMA",
    "TUSZ_REAL_EDF_ADAPTIVE_ROLLOUT_SCHEMA",
    "TUSZ_REAL_EDF_ADAPTIVE_COHORT_RECEIPT_SCHEMA",
    "TUSZ_REAL_EDF_ADAPTER_METHOD_ID",
    "DirectObservedCommon17EDFQueryReader",
    "load_tusz_real_edf_adaptive_manifest",
    "materialize_tusz_real_edf_adaptive_entry",
    "summarize_tusz_real_edf_adaptive_rollouts",
    "validate_tusz_real_edf_adaptive_manifest",
]
