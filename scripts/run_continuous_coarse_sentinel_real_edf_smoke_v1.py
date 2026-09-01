#!/usr/bin/env python3
"""Run a two-record, target-blind real-EDF sentinel engineering smoke.

This smoke deliberately uses a recording-clock horizon frozen independently
of detector output, annotations, labels, or clinical information.  Its source
API surface is restricted to the EDF acquisition header and the 17 directly
observed common17 signal channels.  It verifies engineering behavior only:
gap-free 1/4/16-second screening, query-only proposal permissions, and exact
raw-EEG/QC content replay.  It is not a seizure-detection, onset, Findings, or
SOZ efficacy evaluation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pyedflib


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.continuous_coarse_sentinel_cache_v1 import (  # noqa: E402
    COMMON17_CHANNELS,
    materialize_common17_continuous_coarse_sentinel_cache_v1,
    validate_common17_continuous_coarse_sentinel_cache_v1,
)
from src.clinical_eeg_long_recording.tusz_real_edf_adaptive_findings_v1 import (  # noqa: E402
    DirectObservedCommon17EDFQueryReader,
)


SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_continuous_coarse_sentinel_real_edf_smoke_cohort_v1"
)
METHOD_ID: Final[str] = (
    "COMMON17-CONTINUOUS-SENTINEL-TARGET-BLIND-REAL-EDF-SMOKE-V1"
)
DEFAULT_SOURCE_MANIFEST: Final[Path] = (
    ROOT / "configs/clinical_eeg_adaptive_support_v2_real_edf_smoke_v1.json"
)
DEFAULT_TUSZ_ROOT: Final[Path] = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT: Final[Path] = (
    ROOT
    / "outputs/clinical_eeg_continuous_coarse_sentinel_real_edf_smoke_v1_20260825"
)

# The interval is frozen on the recording clock, before any EDF signal is read.
# Fractional edges exercise clipped first/last cells at all three scales.
HORIZON_START_SECONDS: Final[float] = 17.25
HORIZON_STOP_SECONDS: Final[float] = 48.75
EXPECTED_RECORDING_IDS: Final[tuple[str, str]] = (
    "aaaaaajy_s003_t001",
    "aaaaahie_s007_t011",
)
ROSTER_ENTRY_ALLOWLIST: Final[tuple[str, ...]] = (
    "recording_id",
    "official_split",
    "relative_edf_path",
    "edf_sha256",
    "expected_source_sampling_rate_hz",
    "expected_recording_sample_count",
    "expected_FZ_PZ_observation_state",
)
_ALLOWED_MIDLINE_STATES: Final[frozenset[str]] = frozenset(
    {"both_observed_but_excluded", "both_naturally_absent"}
)
_EDF_API_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {
        "getSignalLabels",
        "getSampleFrequency",
        "getNSamples",
        "getPhysicalDimension",
        "getPhysicalMinimum",
        "getPhysicalMaximum",
        "readSignal",
        "close",
    }
)
_RUN_SCOPE: Final[dict[str, object]] = {
    "engineering_real_EDF_smoke_only": True,
    "seizure_detection_efficacy_evaluated": False,
    "onset_accuracy_evaluated": False,
    "Findings_or_clinical_terms_produced": False,
    "SOZ_or_channel_localization_evaluated": False,
    "detector_score_or_anchor_used": False,
    "TERM_CSV_BI_or_reference_interval_opened": False,
    "EDF_annotation_API_called": False,
    "spreadsheet_Excel_or_doctor_text_opened": False,
    "clinical_history_video_or_behavior_opened": False,
    "source_eval_opened": False,
    "LLM_used": False,
}
_CLAIM_LIMITS: Final[dict[str, object]] = {
    "real_common17_engineering_path_verified": True,
    "continuous_coverage_implementation_verified": True,
    "raw_and_QC_exact_replay_verified": True,
    "proposal_permission_firewall_verified": True,
    "detector_provider_qualified": False,
    "sentinel_detection_performance_claim_authorized": False,
    "Findings_or_SOZ_efficacy_claim_authorized": False,
    "clinical_deployment_allowed": False,
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


def _array_sha256(value: np.ndarray, *, domain: str) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            # The sentinel validator freezes the semantic primitive roster in
            # insertion order, so preserve it across disk replay.  Content
            # hashes remain canonical and key-order independent.
            json.dump(value, handle, ensure_ascii=False, sort_keys=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _safe_edf_path(root: Path, value: object) -> tuple[str, Path]:
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".edf":
        raise ValueError("real-EDF smoke path is not a safe relative EDF path")
    resolved = root.joinpath(*relative.parts).resolve(strict=True)
    resolved.relative_to(root)
    return relative.as_posix(), resolved


def _project_signal_only_roster(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Load only the signal/acquisition projection of the pre-existing roster."""

    source = path.resolve(strict=True)
    if source.suffix.lower() != ".json":
        raise ValueError("smoke roster must be JSON")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "clinical_eeg_adaptive_support_v2_real_edf_smoke_manifest_v1"
    ):
        raise ValueError("pre-existing real-EDF smoke roster schema drifted")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != 2:
        raise ValueError("real-EDF sentinel smoke requires the frozen two-record roster")
    entries: list[dict[str, Any]] = []
    for raw in raw_entries:
        if not isinstance(raw, dict) or not set(ROSTER_ENTRY_ALLOWLIST).issubset(raw):
            raise ValueError("real-EDF roster lacks its signal-only projection")
        row = {field: deepcopy(raw[field]) for field in ROSTER_ENTRY_ALLOWLIST}
        if row["official_split"] != "dev":
            raise ValueError("real-EDF engineering smoke must remain source-dev only")
        if row["expected_FZ_PZ_observation_state"] not in _ALLOWED_MIDLINE_STATES:
            raise ValueError("real-EDF midline observation state is unsupported")
        entries.append(row)
    if tuple(row["recording_id"] for row in entries) != EXPECTED_RECORDING_IDS:
        raise ValueError("pre-existing real-EDF smoke roster identity drifted")
    return _file_sha256(source), entries


class _AllowlistedEDFAPI:
    """Fail-closed pyEDFlib proxy that exposes no annotation/header-text API."""

    def __init__(self, path: str) -> None:
        self._reader = pyedflib.EdfReader(path)
        self.api_call_counts = {name: 0 for name in sorted(_EDF_API_ALLOWLIST)}
        self.read_signal_indices: list[int] = []
        self.read_signal_intervals: list[list[int]] = []
        self.closed = False

    def __getattr__(self, name: str) -> Any:
        if name not in _EDF_API_ALLOWLIST:
            raise RuntimeError(f"EDF API is outside the strict signal allowlist: {name}")
        target = getattr(self._reader, name)

        def guarded(*args: object, **kwargs: object) -> Any:
            self.api_call_counts[name] += 1
            if name == "readSignal":
                if len(args) < 3 or kwargs:
                    raise RuntimeError("readSignal must use explicit index/start/count")
                index, start, count = (int(args[0]), int(args[1]), int(args[2]))
                self.read_signal_indices.append(index)
                self.read_signal_intervals.append([start, start + count])
            result = target(*args, **kwargs)
            if name == "close":
                self.closed = True
            return result

        return guarded

    def audit(self) -> dict[str, Any]:
        return {
            "EDF_API_allowlist": sorted(_EDF_API_ALLOWLIST),
            "api_call_counts": deepcopy(self.api_call_counts),
            "readSignal_indices": list(self.read_signal_indices),
            "readSignal_intervals_samples": deepcopy(self.read_signal_intervals),
            "forbidden_API_call_count": 0,
            "readAnnotations_API_exposed": False,
            "patient_or_clinical_header_API_exposed": False,
            "closed": self.closed,
        }


def _partition_audit(
    intervals: Sequence[Sequence[int]], *, start: int, stop: int
) -> dict[str, int | bool]:
    cursor = start
    total = 0
    for interval in intervals:
        if (
            len(interval) != 2
            or isinstance(interval[0], bool)
            or isinstance(interval[1], bool)
            or not isinstance(interval[0], int)
            or not isinstance(interval[1], int)
            or interval[0] != cursor
            or interval[1] <= interval[0]
            or interval[1] > stop
        ):
            raise ValueError("sentinel scale is not a gap-free legal-horizon partition")
        total += interval[1] - interval[0]
        cursor = interval[1]
    if cursor != stop or total != stop - start:
        raise ValueError("sentinel scale does not cover the full legal horizon")
    return {
        "cell_count": len(intervals),
        "covered_samples_per_channel": total,
        "expected_samples_per_channel": stop - start,
        "starts_at_legal_horizon": bool(intervals and intervals[0][0] == start),
        "stops_at_legal_horizon": bool(intervals and intervals[-1][1] == stop),
        "gap_samples": 0,
        "overlap_samples": 0,
        "exact_partition": True,
    }


def _materialize_once(
    *, entry: Mapping[str, Any], edf_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    guards: list[_AllowlistedEDFAPI] = []

    def factory(path: str) -> _AllowlistedEDFAPI:
        guard = _AllowlistedEDFAPI(path)
        guards.append(guard)
        return guard

    captured: dict[str, np.ndarray] = {}
    with DirectObservedCommon17EDFQueryReader(
        edf_path,
        expected_edf_sha256=None,
        reader_factory=factory,
        verify_file_sha256=False,
    ) as reader:
        rate = float(entry["expected_source_sampling_rate_hz"])
        count = int(entry["expected_recording_sample_count"])
        if (
            reader.sampling_rate_hz != rate
            or reader.recording_sample_count != count
            or reader.fz_pz_observation_state
            != entry["expected_FZ_PZ_observation_state"]
        ):
            raise ValueError("real EDF acquisition differs from its frozen roster")
        start = int(round(HORIZON_START_SECONDS * rate))
        stop = int(round(HORIZON_STOP_SECONDS * rate))
        if (
            not math.isclose(start / rate, HORIZON_START_SECONDS, abs_tol=1.0e-12)
            or not math.isclose(stop / rate, HORIZON_STOP_SECONDS, abs_tol=1.0e-12)
            or not 0 <= start < stop <= count
        ):
            raise ValueError("frozen clock horizon is not exactly representable in EDF")

        def query(start_sample: int, stop_sample: int) -> dict[str, np.ndarray]:
            if captured:
                raise RuntimeError("continuous sentinel requested EDF more than once")
            chunk = reader(start_sample, stop_sample)
            captured["signal"] = np.ascontiguousarray(chunk.signal_volts)
            captured["qc"] = np.ascontiguousarray(chunk.valid_sample_mask, dtype=bool)
            return {
                "signal_volts": captured["signal"],
                "valid_sample_mask": captured["qc"],
            }

        sentinel = materialize_common17_continuous_coarse_sentinel_cache_v1(
            recording_id=str(entry["recording_id"]),
            candidate_group_id=(
                f"{entry['recording_id']}__target_blind_clock_horizon_17p25_48p75"
            ),
            horizon_start_sample=start,
            horizon_stop_sample=stop,
            recording_sample_count=count,
            sampling_rate_hz=rate,
            query_reader=query,
            channel_order=COMMON17_CHANNELS,
        )
        reader_receipt = reader.receipt()
    if len(guards) != 1 or not guards[0].closed:
        raise RuntimeError("strict EDF API wrapper lifecycle drifted")
    api_audit = guards[0].audit()
    if (
        reader_receipt["source_edf_sha256"] is not None
        or reader_receipt["EDF_annotation_API_called"] is not False
        or reader_receipt["patient_header_API_called"] is not False
        or reader_receipt["target_sidecar_opened"] is not False
        or reader_receipt["FZ_PZ_samples_read"] is not False
        or reader_receipt["non_common17_signal_samples_read"] is not False
        or reader_receipt["common17_channel_order"] != list(COMMON17_CHANNELS)
        or api_audit["readSignal_indices"] != reader_receipt["selected_edf_indices"]
        or set(api_audit["readSignal_indices"])
        != set(reader_receipt["selected_edf_indices"])
        or api_audit["api_call_counts"]["readSignal"] != len(COMMON17_CHANNELS)
    ):
        raise ValueError("strict real-EDF common17 reader firewall failed")
    independent = {
        "signal_sha256": _array_sha256(
            captured["signal"].astype("<f8", copy=False),
            domain="common17-continuous-coarse-sentinel-volts-v1",
        ),
        "valid_sample_mask_sha256": _array_sha256(
            captured["qc"].astype(np.uint8),
            domain="common17-continuous-coarse-sentinel-qc-v1",
        ),
    }
    if sentinel["source_binding"]["signal_sha256"] != independent["signal_sha256"]:
        raise ValueError("sentinel raw EEG content hash is not independently reproducible")
    if (
        sentinel["source_binding"]["valid_sample_mask_sha256"]
        != independent["valid_sample_mask_sha256"]
    ):
        raise ValueError("sentinel EEG-derived QC hash is not independently reproducible")
    return (
        validate_common17_continuous_coarse_sentinel_cache_v1(sentinel),
        reader_receipt,
        api_audit,
        independent,
    )


def _record_receipt(
    *,
    entry: Mapping[str, Any],
    relative_edf_path: str,
    sentinel: Mapping[str, Any],
    replay: Mapping[str, Any],
    reader_receipt: Mapping[str, Any],
    replay_reader_receipt: Mapping[str, Any],
    api_audit: Mapping[str, Any],
    replay_api_audit: Mapping[str, Any],
    independent: Mapping[str, str],
    replay_independent: Mapping[str, str],
    sentinel_relative_path: str,
    sentinel_file_sha256: str,
) -> dict[str, Any]:
    if sentinel != replay:
        raise ValueError("real EDF sentinel receipt is not exactly replayable")
    if independent != replay_independent:
        raise ValueError("real EDF raw EEG/QC hashes are not exactly replayable")
    if reader_receipt != replay_reader_receipt or api_audit != replay_api_audit:
        raise ValueError("real EDF reader/API audit is not exactly replayable")
    start, stop = sentinel["legal_horizon"]["interval_samples"]
    scale_cells = {
        "1": sentinel["base_cells_1s"],
        **sentinel["aggregate_cells"],
    }
    coverage = {
        scale: _partition_audit(
            [row["interval_samples"] for row in cells], start=start, stop=stop
        )
        for scale, cells in scale_cells.items()
    }
    for scale, cells in scale_cells.items():
        if len(sentinel["screening_transitions"][scale]) != max(0, len(cells) - 1):
            raise ValueError("continuous sentinel skipped an adjacent-cell boundary")
    proposals = sentinel["native_query_proposals"]
    proposal_only = all(
        row["permission"] == "native_query_only"
        and row["clinical_assertion_authorized"] is False
        and row["trigger_native_query"] is True
        for row in proposals
    )
    authorization = sentinel["authorization"]
    if (
        not proposal_only
        or authorization["may_trigger_downstream_native_query"] is not True
        or authorization["may_assert_eeg_finding"] is not False
        or authorization["may_assert_onset_or_offset"] is not False
        or authorization["may_rank_channels_regions_or_laterality"] is not False
        or authorization["may_assert_SOZ_EZ_or_diagnosis"] is not False
        or sentinel["compute_ledger"]["downstream_native_fine_compute"]["executed"]
        is not False
    ):
        raise ValueError("continuous sentinel proposal permission escalated")
    return {
        "recording_id": entry["recording_id"],
        "official_split": entry["official_split"],
        "source": {
            "relative_edf_path": relative_edf_path,
            "frozen_roster_edf_sha256": entry["edf_sha256"],
            "whole_EDF_SHA256_recomputed_at_runtime": False,
            "whole_EDF_hash_scan_performed": False,
            "sampling_rate_hz": sentinel["acquisition"]["sampling_rate_hz"],
            "recording_sample_count": sentinel["acquisition"]["recording_sample_count"],
            "FZ_PZ_observation_state": reader_receipt["FZ_PZ_observation_state"],
        },
        "legal_horizon": {
            "interval_samples": [start, stop],
            "interval_recording_seconds": sentinel["legal_horizon"][
                "interval_recording_seconds"
            ],
            "sample_count_per_channel": stop - start,
            "selection_source": "frozen_recording_clock_constant_before_EDF_read",
            "detector_anchor_or_target_used": False,
        },
        "sentinel_receipt": {
            "relative_path": sentinel_relative_path,
            "file_sha256": sentinel_file_sha256,
            "content_receipt_sha256": sentinel["receipt_sha256"],
            "signal_sha256": sentinel["source_binding"]["signal_sha256"],
            "valid_sample_mask_sha256": sentinel["source_binding"][
                "valid_sample_mask_sha256"
            ],
        },
        "coverage_recomputed": coverage,
        "every_adjacent_cell_screened": {
            scale: len(sentinel["screening_transitions"][scale])
            == max(0, len(cells) - 1)
            for scale, cells in scale_cells.items()
        },
        "reader_firewall": {
            "common17_channel_order": reader_receipt["common17_channel_order"],
            "selected_raw_names": reader_receipt["selected_raw_names"],
            "selected_edf_indices": reader_receipt["selected_edf_indices"],
            "FZ_PZ_samples_read": False,
            "non_common17_signal_samples_read": False,
            "EDF_annotation_API_called": False,
            "patient_header_API_called": False,
            "target_sidecar_opened": False,
            "primary_reader_receipt_sha256": _canonical_sha256(reader_receipt),
            "replay_reader_receipt_sha256": _canonical_sha256(replay_reader_receipt),
            "primary_API_audit": deepcopy(api_audit),
            "replay_API_audit": deepcopy(replay_api_audit),
        },
        "query_proposal_firewall": {
            "proposal_count": len(proposals),
            "proposal_ids": [row["proposal_id"] for row in proposals],
            "permission": "native_query_only",
            "all_proposals_request_native_EEG_only": proposal_only,
            "Finding_assertion_authorized": False,
            "onset_or_offset_assertion_authorized": False,
            "SOZ_or_channel_rank_assertion_authorized": False,
            "downstream_native_query_executed_by_sentinel": False,
        },
        "exact_replay": {
            "independent_primary_signal_sha256": independent["signal_sha256"],
            "independent_replay_signal_sha256": replay_independent["signal_sha256"],
            "independent_primary_QC_sha256": independent[
                "valid_sample_mask_sha256"
            ],
            "independent_replay_QC_sha256": replay_independent[
                "valid_sample_mask_sha256"
            ],
            "raw_signal_hash_exact_match": True,
            "QC_hash_exact_match": True,
            "sentinel_content_receipt_exact_match": True,
            "reader_receipt_exact_match": True,
            "API_audit_exact_match": True,
        },
    }


def validate_cohort_receipt_v1(payload: object) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError("real-EDF sentinel cohort receipt must be an object")
    data = deepcopy(payload)
    if set(data) != {
        "schema_version",
        "method_id",
        "receipt_sha256",
        "source_roster_binding",
        "horizon_policy",
        "common17_contract",
        "run_scope",
        "records",
        "summary",
        "claim_limits",
    }:
        raise ValueError("real-EDF sentinel cohort receipt fields drifted")
    if data["schema_version"] != SCHEMA_VERSION or data["method_id"] != METHOD_ID:
        raise ValueError("real-EDF sentinel cohort method binding drifted")
    if data["run_scope"] != _RUN_SCOPE or data["claim_limits"] != _CLAIM_LIMITS:
        raise ValueError("real-EDF sentinel scope or claims escalated")
    if data["common17_contract"] != {
        "channel_order": list(COMMON17_CHANNELS),
        "directly_observed_only": True,
        "FZ_PZ_samples_read": False,
        "zero_fill_interpolation_or_montage_synthesis_used": False,
        "native_sampling_rate_preserved": True,
    }:
        raise ValueError("real-EDF sentinel common17 contract drifted")
    if data["horizon_policy"] != {
        "start_recording_seconds": HORIZON_START_SECONDS,
        "stop_recording_seconds": HORIZON_STOP_SECONDS,
        "duration_seconds": HORIZON_STOP_SECONDS - HORIZON_START_SECONDS,
        "frozen_before_signal_read": True,
        "recording_clock_aligned_not_detector_or_target_aligned": True,
    }:
        raise ValueError("real-EDF sentinel horizon policy drifted")
    binding = data["source_roster_binding"]
    if (
        type(binding) is not dict
        or binding.get("source_manifest_relative_path")
        != "configs/clinical_eeg_adaptive_support_v2_real_edf_smoke_v1.json"
        or binding.get("projected_entry_field_allowlist") != list(ROSTER_ENTRY_ALLOWLIST)
        or binding.get("ignored_prediction_or_detector_fields_opened") is not False
        or binding.get("referenced_prediction_files_opened") is not False
        or binding.get("recording_ids") != list(EXPECTED_RECORDING_IDS)
    ):
        raise ValueError("real-EDF signal-only roster projection drifted")
    records = data["records"]
    if not isinstance(records, list) or [row.get("recording_id") for row in records] != list(
        EXPECTED_RECORDING_IDS
    ):
        raise ValueError("real-EDF sentinel record denominator drifted")
    for row in records:
        if set(row) != {
            "recording_id",
            "official_split",
            "source",
            "legal_horizon",
            "sentinel_receipt",
            "coverage_recomputed",
            "every_adjacent_cell_screened",
            "reader_firewall",
            "query_proposal_firewall",
            "exact_replay",
        }:
            raise ValueError("real-EDF sentinel record fields drifted")
        if row["official_split"] != "dev":
            raise ValueError("real-EDF sentinel smoke opened a non-dev record")
        source = row["source"]
        if set(source) != {
            "relative_edf_path",
            "frozen_roster_edf_sha256",
            "whole_EDF_SHA256_recomputed_at_runtime",
            "whole_EDF_hash_scan_performed",
            "sampling_rate_hz",
            "recording_sample_count",
            "FZ_PZ_observation_state",
        } or source["FZ_PZ_observation_state"] not in _ALLOWED_MIDLINE_STATES:
            raise ValueError("real-EDF sentinel source fields drifted")
        if (
            source["whole_EDF_SHA256_recomputed_at_runtime"] is not False
            or source["whole_EDF_hash_scan_performed"] is not False
        ):
            raise ValueError("real-EDF sentinel performed an unauthorized whole-file scan")
        legal = row["legal_horizon"]
        if set(legal) != {
            "interval_samples",
            "interval_recording_seconds",
            "sample_count_per_channel",
            "selection_source",
            "detector_anchor_or_target_used",
        } or legal["interval_recording_seconds"] != [
            HORIZON_START_SECONDS,
            HORIZON_STOP_SECONDS,
        ]:
            raise ValueError("real-EDF sentinel record horizon drifted")
        if (
            legal["selection_source"]
            != "frozen_recording_clock_constant_before_EDF_read"
            or legal["detector_anchor_or_target_used"] is not False
            or legal["sample_count_per_channel"]
            != legal["interval_samples"][1] - legal["interval_samples"][0]
        ):
            raise ValueError("real-EDF sentinel record horizon semantics drifted")
        sentinel_binding = row["sentinel_receipt"]
        if set(sentinel_binding) != {
            "relative_path",
            "file_sha256",
            "content_receipt_sha256",
            "signal_sha256",
            "valid_sample_mask_sha256",
        }:
            raise ValueError("real-EDF sentinel receipt binding drifted")
        for field in (
            "file_sha256",
            "content_receipt_sha256",
            "signal_sha256",
            "valid_sample_mask_sha256",
        ):
            value = sentinel_binding[field]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("real-EDF sentinel hash binding is invalid")
        if set(row["coverage_recomputed"]) != {"1", "4", "16"}:
            raise ValueError("real-EDF sentinel scale roster drifted")
        if any(
            set(audit)
            != {
                "cell_count",
                "covered_samples_per_channel",
                "expected_samples_per_channel",
                "starts_at_legal_horizon",
                "stops_at_legal_horizon",
                "gap_samples",
                "overlap_samples",
                "exact_partition",
            }
            or audit.get("exact_partition") is not True
            or audit.get("starts_at_legal_horizon") is not True
            or audit.get("stops_at_legal_horizon") is not True
            or audit.get("gap_samples") != 0
            or audit.get("overlap_samples") != 0
            or audit.get("covered_samples_per_channel")
            != legal["sample_count_per_channel"]
            or audit.get("expected_samples_per_channel")
            != legal["sample_count_per_channel"]
            for audit in row["coverage_recomputed"].values()
        ):
            raise ValueError("real-EDF sentinel coverage is not exact")
        if row["every_adjacent_cell_screened"] != {
            "1": True,
            "4": True,
            "16": True,
        }:
            raise ValueError("real-EDF sentinel skipped an adjacent boundary")
        firewall = row["reader_firewall"]
        if (
            set(firewall)
            != {
                "common17_channel_order",
                "selected_raw_names",
                "selected_edf_indices",
                "FZ_PZ_samples_read",
                "non_common17_signal_samples_read",
                "EDF_annotation_API_called",
                "patient_header_API_called",
                "target_sidecar_opened",
                "primary_reader_receipt_sha256",
                "replay_reader_receipt_sha256",
                "primary_API_audit",
                "replay_API_audit",
            }
            or firewall["common17_channel_order"] != list(COMMON17_CHANNELS)
            or len(firewall["selected_raw_names"]) != len(COMMON17_CHANNELS)
            or len(firewall["selected_edf_indices"]) != len(COMMON17_CHANNELS)
            or len(set(firewall["selected_edf_indices"])) != len(COMMON17_CHANNELS)
            or firewall["FZ_PZ_samples_read"] is not False
            or firewall["non_common17_signal_samples_read"] is not False
            or firewall["EDF_annotation_API_called"] is not False
            or firewall["patient_header_API_called"] is not False
            or firewall["target_sidecar_opened"] is not False
            or firewall["primary_API_audit"] != firewall["replay_API_audit"]
            or firewall["primary_API_audit"]["forbidden_API_call_count"] != 0
            or firewall["primary_API_audit"]["readAnnotations_API_exposed"] is not False
        ):
            raise ValueError("real-EDF sentinel reader firewall drifted")
        api = firewall["primary_API_audit"]
        expected_counts = {
            "close": 1,
            "getNSamples": 1,
            "getPhysicalDimension": len(COMMON17_CHANNELS),
            "getPhysicalMaximum": len(COMMON17_CHANNELS),
            "getPhysicalMinimum": len(COMMON17_CHANNELS),
            "getSampleFrequency": len(COMMON17_CHANNELS),
            "getSignalLabels": 1,
            "readSignal": len(COMMON17_CHANNELS),
        }
        if (
            set(api)
            != {
                "EDF_API_allowlist",
                "api_call_counts",
                "readSignal_indices",
                "readSignal_intervals_samples",
                "forbidden_API_call_count",
                "readAnnotations_API_exposed",
                "patient_or_clinical_header_API_exposed",
                "closed",
            }
            or api["EDF_API_allowlist"] != sorted(_EDF_API_ALLOWLIST)
            or api["api_call_counts"] != expected_counts
            or api["readSignal_indices"] != firewall["selected_edf_indices"]
            or api["readSignal_intervals_samples"]
            != [legal["interval_samples"]] * len(COMMON17_CHANNELS)
            or api["patient_or_clinical_header_API_exposed"] is not False
            or api["closed"] is not True
        ):
            raise ValueError("real-EDF sentinel API audit drifted")
        proposal = row["query_proposal_firewall"]
        if (
            set(proposal)
            != {
                "proposal_count",
                "proposal_ids",
                "permission",
                "all_proposals_request_native_EEG_only",
                "Finding_assertion_authorized",
                "onset_or_offset_assertion_authorized",
                "SOZ_or_channel_rank_assertion_authorized",
                "downstream_native_query_executed_by_sentinel",
            }
            or proposal["proposal_count"] != len(proposal["proposal_ids"])
            or proposal["proposal_ids"]
            != [f"NQ{index:06d}" for index in range(proposal["proposal_count"])]
            or proposal["permission"] != "native_query_only"
            or proposal["all_proposals_request_native_EEG_only"] is not True
            or proposal["Finding_assertion_authorized"] is not False
            or proposal["onset_or_offset_assertion_authorized"] is not False
            or proposal["SOZ_or_channel_rank_assertion_authorized"] is not False
            or proposal["downstream_native_query_executed_by_sentinel"] is not False
        ):
            raise ValueError("real-EDF sentinel proposal permission escalated")
        replay = row["exact_replay"]
        if set(replay) != {
            "independent_primary_signal_sha256",
            "independent_replay_signal_sha256",
            "independent_primary_QC_sha256",
            "independent_replay_QC_sha256",
            "raw_signal_hash_exact_match",
            "QC_hash_exact_match",
            "sentinel_content_receipt_exact_match",
            "reader_receipt_exact_match",
            "API_audit_exact_match",
        }:
            raise ValueError("real-EDF sentinel replay fields drifted")
        if (
            replay["independent_primary_signal_sha256"]
            != replay["independent_replay_signal_sha256"]
            or replay["independent_primary_signal_sha256"]
            != sentinel_binding["signal_sha256"]
            or replay["independent_primary_QC_sha256"]
            != replay["independent_replay_QC_sha256"]
            or replay["independent_primary_QC_sha256"]
            != sentinel_binding["valid_sample_mask_sha256"]
            or not all(value is True for value in replay.values() if isinstance(value, bool))
        ):
            raise ValueError("real-EDF sentinel replay failed")
    expected_summary = {
        "record_count": 2,
        "all_records_completed": True,
        "all_1s_4s_16s_partitions_gap_free": True,
        "all_common17_samples_only": True,
        "all_FZ_PZ_samples_unread": True,
        "all_forbidden_source_APIs_unopened": True,
        "all_proposals_query_only": True,
        "all_raw_and_QC_hashes_exactly_replayed": True,
        "engineering_smoke_passed": True,
    }
    if data["summary"] != expected_summary:
        raise ValueError("real-EDF sentinel cohort summary drifted")
    expected_hash = _canonical_sha256(
        {key: value for key, value in data.items() if key != "receipt_sha256"}
    )
    if data["receipt_sha256"] != expected_hash:
        raise ValueError("real-EDF sentinel cohort content hash mismatch")
    return data


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    source_manifest = arguments.source_manifest.resolve(strict=True)
    roster_sha256, entries = _project_signal_only_roster(source_manifest)
    tusz_root = arguments.tusz_root.resolve(strict=True)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for entry in entries:
        relative_edf_path, edf_path = _safe_edf_path(
            tusz_root, entry["relative_edf_path"]
        )
        primary = _materialize_once(entry=entry, edf_path=edf_path)
        replay = _materialize_once(entry=entry, edf_path=edf_path)
        sentinel_path = (
            output / "records" / str(entry["recording_id"]) / "sentinel_receipt.json"
        )
        _atomic_json(sentinel_path, primary[0])
        sentinel_relative_path = sentinel_path.relative_to(output).as_posix()
        records.append(
            _record_receipt(
                entry=entry,
                relative_edf_path=relative_edf_path,
                sentinel=primary[0],
                replay=replay[0],
                reader_receipt=primary[1],
                replay_reader_receipt=replay[1],
                api_audit=primary[2],
                replay_api_audit=replay[2],
                independent=primary[3],
                replay_independent=replay[3],
                sentinel_relative_path=sentinel_relative_path,
                sentinel_file_sha256=_file_sha256(sentinel_path),
            )
        )
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "source_roster_binding": {
            "source_manifest_relative_path": source_manifest.relative_to(ROOT).as_posix(),
            "source_manifest_file_sha256": roster_sha256,
            "projected_entry_field_allowlist": list(ROSTER_ENTRY_ALLOWLIST),
            "ignored_prediction_or_detector_fields_opened": False,
            "referenced_prediction_files_opened": False,
            "recording_ids": list(EXPECTED_RECORDING_IDS),
        },
        "horizon_policy": {
            "start_recording_seconds": HORIZON_START_SECONDS,
            "stop_recording_seconds": HORIZON_STOP_SECONDS,
            "duration_seconds": HORIZON_STOP_SECONDS - HORIZON_START_SECONDS,
            "frozen_before_signal_read": True,
            "recording_clock_aligned_not_detector_or_target_aligned": True,
        },
        "common17_contract": {
            "channel_order": list(COMMON17_CHANNELS),
            "directly_observed_only": True,
            "FZ_PZ_samples_read": False,
            "zero_fill_interpolation_or_montage_synthesis_used": False,
            "native_sampling_rate_preserved": True,
        },
        "run_scope": deepcopy(_RUN_SCOPE),
        "records": records,
        "summary": {
            "record_count": 2,
            "all_records_completed": True,
            "all_1s_4s_16s_partitions_gap_free": True,
            "all_common17_samples_only": True,
            "all_FZ_PZ_samples_unread": True,
            "all_forbidden_source_APIs_unopened": True,
            "all_proposals_query_only": True,
            "all_raw_and_QC_hashes_exactly_replayed": True,
            "engineering_smoke_passed": True,
        },
        "claim_limits": deepcopy(_CLAIM_LIMITS),
    }
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    receipt = validate_cohort_receipt_v1(body)
    _atomic_json(output / "receipt.json", receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST
    )
    value.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
