"""Real-TUSZ adapter for the target-blind common-17 support comparator.

The extraction manifest contains a frozen navigation anchor but no seizure
offset, channel target, SOZ label, clinical text, or annotation payload.  Each
event opens one EDF signal reader and exposes only the exact intervals queried
by the five support arms.  EDF annotations and TERM/CSV sidecars are never
opened by this module.

Reference-interval evaluation is intentionally absent here.  It is performed
only on content-addressed event receipts by the separate post-freeze API in
``common17_support_policy_comparator_v1``.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Final, Mapping, MutableMapping, Sequence

import pyedflib

from .adaptive_native_evidence_common17 import COMMON17_CHANNELS, NativeEEGQueryChunk
from .common17_support_policy_comparator_v1 import (
    STRATEGY_ORDER,
    materialize_common17_support_policy_comparison_v1,
    summarize_common17_support_policy_comparison_cohort_v1,
    validate_common17_support_policy_comparison_v1,
)
from .tusz_real_edf_adaptive_findings_v1 import (
    DirectObservedCommon17EDFQueryReader,
)


TUSZ_REAL_EDF_SUPPORT_COMPARISON_MANIFEST_SCHEMA: Final[str] = (
    "clinical_eeg_tusz_real_edf_support_comparison_manifest_v1"
)
TUSZ_REAL_EDF_SUPPORT_COMPARISON_EVENT_SCHEMA: Final[str] = (
    "clinical_eeg_tusz_real_edf_support_comparison_event_v1"
)
TUSZ_REAL_EDF_SUPPORT_COMPARISON_COHORT_SCHEMA: Final[str] = (
    "clinical_eeg_tusz_real_edf_support_comparison_cohort_v1"
)
TUSZ_REAL_EDF_SUPPORT_COMPARISON_ADAPTER_METHOD_ID: Final[str] = (
    "TUSZ-COMMON17-SINGLE-OPEN-MULTI-SUPPORT-QUERY-ADAPTER-V1"
)

_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "cohort_id",
        "common17_channel_order",
        "selection_contract",
        "source_bindings",
        "cohort_statistics",
        "entries",
    }
)
_ENTRY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "ordinal",
        "rollout_id",
        "event_id",
        "recording_id",
        "patient_group_id",
        "official_split",
        "relative_edf_path",
        "edf_sha256",
        "navigation_anchor_recording_seconds",
    }
)
_EXPECTED_SELECTION_CONTRACT: Final[dict[str, object]] = {
    "official_split": "source_dev_only",
    "population": "all_events_in_frozen_common17_oracle_roster_source_dev",
    "patient_disjoint_from_source_train_and_source_eval": True,
    "navigation_anchor_source": "frozen_global_TERM_seiz_onset_navigation_only",
    "seizure_offset_present_in_extraction_manifest": False,
    "runtime_TERM_or_annotation_sidecar_access": False,
    "window_or_stopping_selection_uses_reference": False,
    "channel_or_SOZ_target_used": False,
    "clinical_text_used": False,
    "source_eval_opened": False,
}
_SCOPE_RECEIPT: Final[dict[str, object]] = {
    "direct_common17_EEG_samples_used": True,
    "acquisition_parameters_used": True,
    "EEG_derived_ADC_rail_QC_used": True,
    "navigation_anchor_used": True,
    "TERM_or_other_target_sidecar_opened_at_runtime": False,
    "EDF_annotation_API_called": False,
    "patient_header_API_called": False,
    "SOZ_or_channel_target_opened": False,
    "clinical_text_or_spreadsheet_opened": False,
    "non_common17_signal_samples_read": False,
    "FZ_or_PZ_samples_read": False,
    "zero_fill_interpolation_or_montage_synthesis_used": False,
    "source_eval_opened": False,
}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    if len(value) > 220 or any(character in value for character in ("/", "\\")):
        raise ValueError(f"{name} is not a safe identifier")
    return value


def _finite(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _safe_source_path(root: Path, relative: object) -> tuple[str, Path]:
    value = PurePosixPath(str(relative))
    if (
        value.is_absolute()
        or ".." in value.parts
        or value.suffix.lower() != ".edf"
        or not value.parts
        or value.parts[0] != "dev"
    ):
        raise ValueError("relative_edf_path must be a safe source-dev EDF path")
    source = root.joinpath(*value.parts).resolve(strict=True)
    source.relative_to(root)
    return value.as_posix(), source


class _StrategyQueryView:
    """Per-arm ledger over one shared annotation-blind EDF signal reader."""

    def __init__(self, reader: DirectObservedCommon17EDFQueryReader) -> None:
        self.reader = reader
        self.calls: list[tuple[int, int]] = []

    def __call__(self, start: int, stop: int) -> NativeEEGQueryChunk:
        self.calls.append((start, stop))
        return self.reader(start, stop)


def validate_tusz_real_edf_support_comparison_manifest_v1(
    payload: object,
) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != _MANIFEST_FIELDS:
        raise ValueError("TUSZ support-comparison manifest fields drifted")
    data = deepcopy(payload)
    if data["schema_version"] != TUSZ_REAL_EDF_SUPPORT_COMPARISON_MANIFEST_SCHEMA:
        raise ValueError("TUSZ support-comparison manifest schema drifted")
    _identifier(data["cohort_id"], "cohort_id")
    if data["common17_channel_order"] != list(COMMON17_CHANNELS):
        raise ValueError("TUSZ support-comparison manifest is not exact common-17")
    if data["selection_contract"] != _EXPECTED_SELECTION_CONTRACT:
        raise ValueError("TUSZ support-comparison selection firewall drifted")
    bindings = data["source_bindings"]
    if not isinstance(bindings, dict) or set(bindings) != {
        "common17_phase_manifest_sha256",
        "complete_roster_projection_sha256",
        "event_input_table_sha256",
    }:
        raise ValueError("TUSZ support-comparison source bindings drifted")
    for field, value in bindings.items():
        _digest(value, field)
    entries = data["entries"]
    if not isinstance(entries, list) or len(entries) < 5:
        raise ValueError("TUSZ support-comparison cohort is too small")
    event_ids: set[str] = set()
    rollout_ids: set[str] = set()
    patient_groups: set[str] = set()
    recordings: set[str] = set()
    for ordinal, entry in enumerate(entries):
        if type(entry) is not dict or set(entry) != _ENTRY_FIELDS:
            raise ValueError("TUSZ support-comparison entry fields drifted")
        if entry["ordinal"] != ordinal:
            raise ValueError("TUSZ support-comparison ordinals are not contiguous")
        event = _identifier(entry["event_id"], "event_id")
        rollout = _identifier(entry["rollout_id"], "rollout_id")
        recording = _identifier(entry["recording_id"], "recording_id")
        patient = _identifier(entry["patient_group_id"], "patient_group_id")
        if event in event_ids or rollout in rollout_ids:
            raise ValueError("TUSZ support-comparison event identities are duplicated")
        event_ids.add(event)
        rollout_ids.add(rollout)
        patient_groups.add(patient)
        recordings.add(recording)
        if entry["official_split"] != "dev":
            raise ValueError("TUSZ support-comparison manifest escaped source-dev")
        relative = PurePosixPath(str(entry["relative_edf_path"]))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix.lower() != ".edf"
            or not relative.parts
            or relative.parts[0] != "dev"
        ):
            raise ValueError("TUSZ support-comparison path is unsafe or not dev")
        if relative.stem != recording:
            raise ValueError("recording_id differs from the EDF basename")
        _digest(entry["edf_sha256"], "edf_sha256")
        _finite(entry["navigation_anchor_recording_seconds"], "navigation anchor")
    statistics = data["cohort_statistics"]
    expected_statistics = {
        "event_count": len(entries),
        "recording_count": len(recordings),
        "patient_group_count": len(patient_groups),
        "one_official_split": True,
        "patient_overlap_with_source_train": 0,
        "patient_overlap_with_source_eval": 0,
    }
    if statistics != expected_statistics:
        raise ValueError("TUSZ support-comparison cohort statistics drifted")
    return data


def load_tusz_real_edf_support_comparison_manifest_v1(
    path: str | Path,
) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    return validate_tusz_real_edf_support_comparison_manifest_v1(
        json.loads(source.read_text(encoding="utf-8"))
    )


def materialize_tusz_real_edf_support_comparison_entry_v1(
    *,
    entry: Mapping[str, object],
    tusz_root: str | Path,
    manifest_sha256: str,
    verified_source_sha256_cache: MutableMapping[str, str] | None = None,
    verify_file_sha256: bool = True,
    reader_factory: Callable[[str], Any] = pyedflib.EdfReader,
) -> dict[str, Any]:
    """Run all five arms while opening no target or annotation source."""

    if type(entry) is not dict or set(entry) != _ENTRY_FIELDS:
        raise ValueError("TUSZ support-comparison entry fields drifted")
    root = Path(tusz_root).resolve(strict=True)
    relative, source = _safe_source_path(root, entry["relative_edf_path"])
    expected_sha = _digest(entry["edf_sha256"], "edf_sha256")
    cache = verified_source_sha256_cache
    cache_key = str(source)
    cache_hit = cache is not None and cache_key in cache
    if cache_hit:
        observed_sha = _digest(cache[cache_key], "cached EDF SHA-256")
    elif verify_file_sha256:
        observed_sha = _file_sha256(source)
        if cache is not None:
            cache[cache_key] = observed_sha
    else:
        observed_sha = expected_sha
    if observed_sha != expected_sha:
        raise ValueError("source EDF SHA-256 differs from the frozen manifest")

    views: dict[str, _StrategyQueryView] = {}
    with DirectObservedCommon17EDFQueryReader(
        source,
        expected_edf_sha256=None,
        reader_factory=reader_factory,
        verify_file_sha256=False,
    ) as reader:
        def factory(strategy_id: str) -> _StrategyQueryView:
            if strategy_id in views:
                raise RuntimeError("support strategy requested more than one reader view")
            view = _StrategyQueryView(reader)
            views[strategy_id] = view
            return view

        comparison = materialize_common17_support_policy_comparison_v1(
            event_id=str(entry["event_id"]),
            recording_id=str(entry["recording_id"]),
            navigation_anchor_recording_seconds=float(
                entry["navigation_anchor_recording_seconds"]
            ),
            sampling_rate_hz=reader.sampling_rate_hz,
            recording_sample_count=reader.recording_sample_count,
            query_reader_factory=factory,
        )
        selected_raw_names = list(reader.selected_raw_names)
        selected_indices = list(reader.selected_indices)
        midline_state = reader.fz_pz_observation_state
        rate = reader.sampling_rate_hz
        samples = reader.recording_sample_count
        aggregate_calls = reader.calls
    validate_common17_support_policy_comparison_v1(comparison)

    arms = {row["strategy_id"]: row for row in comparison["arms"]}
    if set(views) != set(STRATEGY_ORDER):
        raise RuntimeError("real EDF adapter did not instantiate every strategy")
    arm_ledgers: dict[str, object] = {}
    flattened: list[tuple[int, int]] = []
    for strategy_id in STRATEGY_ORDER:
        calls = views[strategy_id].calls
        expected_calls = [tuple(value) for value in arms[strategy_id]["query_intervals_samples"]]
        if calls != expected_calls:
            raise RuntimeError("real EDF queries differ from the support-arm ledger")
        flattened.extend(calls)
        arm_ledgers[strategy_id] = {
            "query_count": len(calls),
            "query_intervals_samples": [list(value) for value in calls],
            "readSignal_call_count": len(COMMON17_CHANNELS) * len(calls),
            "non_common17_signal_read_count": 0,
        }
    if tuple(flattened) != tuple(
        (int(row["interval_samples"][0]), int(row["interval_samples"][1]))
        for row in aggregate_calls
    ):
        raise RuntimeError("shared EDF reader ledger differs from strategy views")

    body: dict[str, Any] = {
        "schema_version": TUSZ_REAL_EDF_SUPPORT_COMPARISON_EVENT_SCHEMA,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "method_id": TUSZ_REAL_EDF_SUPPORT_COMPARISON_ADAPTER_METHOD_ID,
        "rollout_id": str(entry["rollout_id"]),
        "manifest_sha256": _digest(manifest_sha256, "manifest_sha256"),
        "source": {
            "official_split": "dev",
            "relative_edf_path": relative,
            "edf_sha256": expected_sha,
            "recording_id": str(entry["recording_id"]),
            "event_id": str(entry["event_id"]),
            "patient_group_id": str(entry["patient_group_id"]),
            "source_hash_verified": verify_file_sha256 or cache_hit,
            "source_hash_cache_hit": cache_hit,
        },
        "selection_only": {
            # Bind the receipt to the effective sample-grid coordinate used by
            # the numerical kernel, rather than the pre-quantization TERM
            # decimal carried by the extraction manifest.
            "navigation_anchor_recording_seconds": float(
                comparison["navigation_anchor_recording_seconds"]
            ),
            "passed_to_feature_extractor_as_navigation_coordinate_only": True,
            "seizure_offset_available_at_runtime": False,
            "channel_or_SOZ_target_available_at_runtime": False,
            "patient_group_passed_to_feature_extractor": False,
        },
        "reader_receipt": {
            "source_sampling_rate_hz": rate,
            "recording_sample_count": samples,
            "common17_channel_order": list(COMMON17_CHANNELS),
            "selected_raw_names": selected_raw_names,
            "selected_edf_indices": selected_indices,
            "FZ_PZ_observation_state": midline_state,
            "FZ_PZ_samples_read": False,
            "non_common17_signal_samples_read": False,
            "EDF_annotation_API_called": False,
            "patient_header_API_called": False,
            "full_recording_preloaded": False,
            "single_EDF_open_shared_across_support_arms": True,
            "per_strategy": arm_ledgers,
            "total_readSignal_call_count": len(COMMON17_CHANNELS) * len(flattened),
        },
        "event_comparison_receipt": comparison,
        "scope_receipt": deepcopy(_SCOPE_RECEIPT),
        "claim_limits": {
            "real_EEG_support_comparison_completed": True,
            "oracle_navigation_anchor_is_detector_performance": False,
            "high_budget_shadow_is_ground_truth": False,
            "SOZ_accuracy_measured": False,
            "adaptive_superiority_claim_authorized_from_one_event": False,
        },
    }
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    return validate_tusz_real_edf_support_comparison_event_v1(body)


def validate_tusz_real_edf_support_comparison_event_v1(
    payload: object,
) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError("real EDF support-comparison event receipt must be an object")
    required = {
        "schema_version",
        "receipt_sha256",
        "method_id",
        "rollout_id",
        "manifest_sha256",
        "source",
        "selection_only",
        "reader_receipt",
        "event_comparison_receipt",
        "scope_receipt",
        "claim_limits",
    }
    if set(payload) != required:
        raise ValueError("real EDF support-comparison event fields drifted")
    data = deepcopy(payload)
    if data["schema_version"] != TUSZ_REAL_EDF_SUPPORT_COMPARISON_EVENT_SCHEMA:
        raise ValueError("real EDF support-comparison event schema drifted")
    if data["method_id"] != TUSZ_REAL_EDF_SUPPORT_COMPARISON_ADAPTER_METHOD_ID:
        raise ValueError("real EDF support-comparison adapter method drifted")
    _identifier(data["rollout_id"], "rollout_id")
    _digest(data["manifest_sha256"], "manifest_sha256")
    source = data["source"]
    comparison = validate_common17_support_policy_comparison_v1(
        data["event_comparison_receipt"]
    )
    if (
        source.get("official_split") != "dev"
        or source.get("event_id") != comparison["event_id"]
        or source.get("recording_id") != comparison["recording_id"]
    ):
        raise ValueError("real EDF source identity differs from comparison receipt")
    if data["selection_only"] != {
        "navigation_anchor_recording_seconds": float(
            comparison["navigation_anchor_recording_seconds"]
        ),
        "passed_to_feature_extractor_as_navigation_coordinate_only": True,
        "seizure_offset_available_at_runtime": False,
        "channel_or_SOZ_target_available_at_runtime": False,
        "patient_group_passed_to_feature_extractor": False,
    }:
        raise ValueError("real EDF support-comparison selection firewall drifted")
    reader = data["reader_receipt"]
    if reader.get("common17_channel_order") != list(COMMON17_CHANNELS):
        raise ValueError("real EDF reader is not common-17")
    if (
        reader.get("FZ_PZ_samples_read") is not False
        or reader.get("non_common17_signal_samples_read") is not False
        or reader.get("EDF_annotation_API_called") is not False
        or reader.get("patient_header_API_called") is not False
        or reader.get("full_recording_preloaded") is not False
    ):
        raise ValueError("real EDF reader violated the signal-only firewall")
    per_strategy = reader.get("per_strategy")
    if not isinstance(per_strategy, dict) or set(per_strategy) != set(STRATEGY_ORDER):
        raise ValueError("real EDF reader ledger lacks support strategies")
    arms = {row["strategy_id"]: row for row in comparison["arms"]}
    for strategy_id in STRATEGY_ORDER:
        ledger = per_strategy[strategy_id]
        if ledger.get("query_intervals_samples") != arms[strategy_id][
            "query_intervals_samples"
        ]:
            raise ValueError("real EDF reader and support query ledgers differ")
    if data["scope_receipt"] != _SCOPE_RECEIPT:
        raise ValueError("real EDF support comparison violated its scope")
    expected = _canonical_sha256(
        {key: value for key, value in data.items() if key != "receipt_sha256"}
    )
    if data["receipt_sha256"] != expected:
        raise ValueError("real EDF support-comparison event hash mismatch")
    return data


def summarize_tusz_real_edf_support_comparison_cohort_v1(
    *,
    manifest_sha256: str,
    event_receipts: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    rows = [validate_tusz_real_edf_support_comparison_event_v1(row) for row in event_receipts]
    if not rows:
        raise ValueError("cannot summarize an empty real EDF support cohort")
    if any(row["manifest_sha256"] != manifest_sha256 for row in rows):
        raise ValueError("real EDF event receipts do not share one manifest")
    comparisons = [row["event_comparison_receipt"] for row in rows]
    groups = {
        str(row["source"]["event_id"]): str(row["source"]["patient_group_id"])
        for row in rows
    }
    target_blind = summarize_common17_support_policy_comparison_cohort_v1(
        receipts=comparisons,
        group_id_by_event_id=groups,
    )
    body: dict[str, Any] = {
        "schema_version": TUSZ_REAL_EDF_SUPPORT_COMPARISON_COHORT_SCHEMA,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "method_id": TUSZ_REAL_EDF_SUPPORT_COMPARISON_ADAPTER_METHOD_ID,
        "manifest_sha256": _digest(manifest_sha256, "manifest_sha256"),
        "event_count": len(rows),
        "recording_count": len({row["source"]["recording_id"] for row in rows}),
        "patient_group_count": len({row["source"]["patient_group_id"] for row in rows}),
        "target_blind_comparison_summary": target_blind,
        "event_receipts": [
            {
                "rollout_id": row["rollout_id"],
                "event_id": row["source"]["event_id"],
                "receipt_sha256": row["receipt_sha256"],
            }
            for row in rows
        ],
        "scope_receipt": deepcopy(_SCOPE_RECEIPT),
        "interpretation": {
            "real_EDF_same_kernel_support_ablation_completed": True,
            "metrics_are_target_blind": True,
            "postfreeze_reference_metrics_included": False,
            "source_eval_opened": False,
        },
    }
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    return body


__all__ = [
    "TUSZ_REAL_EDF_SUPPORT_COMPARISON_ADAPTER_METHOD_ID",
    "TUSZ_REAL_EDF_SUPPORT_COMPARISON_COHORT_SCHEMA",
    "TUSZ_REAL_EDF_SUPPORT_COMPARISON_EVENT_SCHEMA",
    "TUSZ_REAL_EDF_SUPPORT_COMPARISON_MANIFEST_SCHEMA",
    "load_tusz_real_edf_support_comparison_manifest_v1",
    "materialize_tusz_real_edf_support_comparison_entry_v1",
    "summarize_tusz_real_edf_support_comparison_cohort_v1",
    "validate_tusz_real_edf_support_comparison_event_v1",
    "validate_tusz_real_edf_support_comparison_manifest_v1",
]
