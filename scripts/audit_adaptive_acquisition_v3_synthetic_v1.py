#!/usr/bin/env python3
"""Run the narrow target-blind synthetic audit for adaptive acquisition v3.

This runner exercises acquisition geometry and the onset-causal evidence
firewall only.  The inputs are deterministic engineering signals; it does not
read source-dev, TERM, SOZ targets, annotations, EDF sidecars, or clinical text
and cannot establish detector or localization efficacy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Final

import numpy as np

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.adaptive_acquisition_v3 import (
    DEFAULT_POLICY,
    METHOD_ID,
    _canonical_sha256,
    _contract,
    materialize_common17_adaptive_acquisition_v3,
)
from src.clinical_eeg_long_recording.adaptive_native_evidence_common17 import (
    COMMON17_CHANNELS,
    _array_sha256,
)
from src.clinical_eeg_long_recording.adaptive_support_v2 import (
    ADAPTIVE_SUPPORT_V2_METHOD_ID,
    materialize_common17_adaptive_support_v2,
)


DEFAULT_OUTPUT: Final[Path] = (
    ROOT
    / "outputs/clinical_eeg_adaptive_acquisition_v3_synthetic_audit_v1_20260825"
)
RATE: Final[float] = 32.0
ANCHOR_SECONDS: Final[float] = 320.0
RECORDING_SECONDS: Final[float] = 640.0


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reader(signal: np.ndarray):
    def read(start: int, stop: int) -> np.ndarray:
        if not 0 <= start < stop <= signal.shape[1]:
            raise ValueError("synthetic audit query lies outside the recording")
        return np.ascontiguousarray(signal[:, start:stop])

    return read


def _background() -> np.ndarray:
    samples = int(round(RECORDING_SECONDS * RATE))
    time = np.arange(samples, dtype=np.float64) / RATE
    rows = []
    for index in range(len(COMMON17_CHANNELS)):
        phase = 0.19 * index
        rows.append(
            4.0e-6 * np.sin(2.0 * np.pi * 2.0 * time + phase)
            + 0.8e-6 * np.cos(2.0 * np.pi * 5.0 * time - phase)
        )
    return np.ascontiguousarray(np.stack(rows, axis=0))


def _add_change(
    signal: np.ndarray,
    *,
    onset_relative_seconds: float,
    offset_relative_seconds: float,
    channel_indices: tuple[int, ...],
    amplitude_uv: float = 120.0,
    frequency_hz: float = 7.0,
) -> None:
    time = np.arange(signal.shape[1], dtype=np.float64) / RATE
    changed = (time >= ANCHOR_SECONDS + onset_relative_seconds) & (
        time < ANCHOR_SECONDS + offset_relative_seconds
    )
    phases = np.linspace(0.0, 1.4, num=len(channel_indices), endpoint=True)
    for index, phase in zip(channel_indices, phases):
        signal[index, changed] += amplitude_uv * 1.0e-6 * np.sin(
            2.0 * np.pi * frequency_hz * time[changed] + phase
        )


def _v3(
    signal: np.ndarray,
    *,
    event_id: str,
    envelopes: tuple[tuple[float, float], ...],
) -> dict[str, Any]:
    return materialize_common17_adaptive_acquisition_v3(
        event_id=event_id,
        recording_id="SYNTHETIC-RECORDING",
        navigation_anchor_recording_seconds=ANCHOR_SECONDS,
        sampling_rate_hz=RATE,
        recording_sample_count=signal.shape[1],
        query_reader=_reader(signal),
        frozen_detector_candidate_envelopes_recording_seconds=envelopes,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _build() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    near_anchor_envelope = ((ANCHOR_SECONDS - 1.0, ANCHOR_SECONDS + 1.0),)

    separated_signal = _background()
    _add_change(
        separated_signal,
        onset_relative_seconds=-36.0,
        offset_relative_seconds=-31.0,
        channel_indices=(2, 6, 11),
    )
    separated_v3 = _v3(
        separated_signal,
        event_id="SYNTH-V3-SEPARATED-ISLAND",
        envelopes=near_anchor_envelope,
    )
    separated_v2 = materialize_common17_adaptive_support_v2(
        event_id="SYNTH-V2-SEPARATED-ISLAND",
        recording_id="SYNTHETIC-RECORDING",
        navigation_anchor_recording_seconds=ANCHOR_SECONDS,
        sampling_rate_hz=RATE,
        recording_sample_count=separated_signal.shape[1],
        query_reader=_reader(separated_signal),
        frozen_detector_candidate_envelopes_recording_seconds=near_anchor_envelope,
    )
    sentinel = next(
        row
        for row in separated_v3["sparse_sentinel_screening"]["rows"]
        if row["relative_interval_seconds"] == [-34.0, -30.0]
    )
    separated_candidate = separated_v3["selected_candidate"]
    _require(
        separated_v2["final_evidence"]["onset_candidate"] is None,
        "v2 unexpectedly recovered the separated island",
    )
    _require(
        sentinel["status"] == "trigger_dense_refinement",
        "the frozen sentinel did not trigger refinement",
    )
    _require(
        separated_candidate is not None
        and -37.0
        <= separated_candidate["relative_to_navigation_anchor_seconds"]
        <= -35.0,
        "v3 did not qualify the earlier separated change",
    )

    gap_signal = _background()
    _add_change(
        gap_signal,
        onset_relative_seconds=-41.0,
        offset_relative_seconds=-36.0,
        channel_indices=(3, 7, 12),
    )
    gap_v3 = _v3(
        gap_signal,
        event_id="SYNTH-V3-ENVELOPE-GAP",
        envelopes=((ANCHOR_SECONDS - 45.0, ANCHOR_SECONDS - 32.0),),
    )
    gap_candidate = gap_v3["selected_candidate"]
    _require(
        all(
            row["status"] == "screen_clear"
            for row in gap_v3["sparse_sentinel_screening"]["rows"]
        ),
        "the detector-envelope gap fixture was not sentinel-clear",
    )
    _require(
        gap_candidate is not None
        and -42.0 <= gap_candidate["relative_to_navigation_anchor_seconds"] <= -40.0,
        "mandatory detector-envelope coverage did not recover the gap change",
    )

    early_signal = _background()
    _add_change(
        early_signal,
        onset_relative_seconds=-5.0,
        offset_relative_seconds=-1.0,
        channel_indices=(2, 6, 11),
    )
    late_signal = early_signal.copy()
    _add_change(
        late_signal,
        onset_relative_seconds=2.0,
        offset_relative_seconds=7.0,
        channel_indices=(4, 8, 13, 16),
        amplitude_uv=180.0,
        frequency_hz=6.0,
    )
    early_v3 = _v3(
        early_signal,
        event_id="SYNTH-V3-EARLY-ONLY",
        envelopes=near_anchor_envelope,
    )
    late_v3 = _v3(
        late_signal,
        event_id="SYNTH-V3-WITH-LATE-SPREAD",
        envelopes=near_anchor_envelope,
    )
    early_ranking = early_v3["positive_onset_channel_ranking"]
    late_ranking = late_v3["positive_onset_channel_ranking"]
    _require(
        early_v3["selected_candidate"]["recording_seconds"]
        == late_v3["selected_candidate"]["recording_seconds"],
        "late evidence changed the selected earlier candidate",
    )
    _require(
        early_ranking["ranking_sha256"] == late_ranking["ranking_sha256"]
        and early_ranking["candidate_locked_prefix_audit"]
        == late_ranking["candidate_locked_prefix_audit"],
        "late evidence escaped into candidate-locked positive ranking",
    )

    budget = separated_v3["query_budget_ledger"]
    comparator = budget["budget_matched_contiguous_comparator"]
    _require(
        budget["phase_sum_equals_analysis_unique_samples"] is True
        and comparator["executed"] is False
        and comparator["exact_match_to_analysis_unique_samples"] is True
        and comparator["samples_per_channel"]
        == budget["analysis_unique_samples_per_channel"],
        "the exact analysis/comparator budget ledger did not close",
    )

    artifacts = {
        "separated_island_v3.json": separated_v3,
        "separated_island_v2.json": separated_v2,
        "detector_envelope_gap_v3.json": gap_v3,
        "early_only_v3.json": early_v3,
        "late_spread_v3.json": late_v3,
    }
    summary: dict[str, Any] = {
        "schema_version": "clinical_eeg_adaptive_acquisition_v3_synthetic_audit_v1",
        "audit_id": "COMMON17-ADAPTIVE-ACQUISITION-V3-TARGET-BLIND-SYNTHETIC-AUDIT",
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "method_id": METHOD_ID,
        "design_contract_sha256": _canonical_sha256(_contract()),
        "policy_sha256": DEFAULT_POLICY.sha256,
        "v2_comparator_method_id": ADAPTIVE_SUPPORT_V2_METHOD_ID,
        "synthetic_signal_contract": {
            "sampling_rate_hz": RATE,
            "recording_seconds": RECORDING_SECONDS,
            "navigation_anchor_recording_seconds": ANCHOR_SECONDS,
            "channel_order": list(COMMON17_CHANNELS),
            "signal_unit": "V",
            "raw_signals_persisted": False,
            "separated_island_signal_sha256": _array_sha256(
                separated_signal.astype("<f8", copy=False),
                prefix="common17-v3-synthetic-separated",
            ),
            "detector_envelope_gap_signal_sha256": _array_sha256(
                gap_signal.astype("<f8", copy=False),
                prefix="common17-v3-synthetic-gap",
            ),
            "early_only_signal_sha256": _array_sha256(
                early_signal.astype("<f8", copy=False),
                prefix="common17-v3-synthetic-early",
            ),
            "late_spread_signal_sha256": _array_sha256(
                late_signal.astype("<f8", copy=False),
                prefix="common17-v3-synthetic-late",
            ),
        },
        "checks": {
            "v2_misses_quiet_gap_separated_earlier_island": True,
            "frozen_sentinel_requests_dense_refinement": True,
            "v3_qualifies_earlier_candidate_outside_probe": True,
            "mandatory_detector_envelope_covers_sentinel_gap": True,
            "analysis_query_intervals_are_physically_deduplicated": True,
            "phase_incremental_sum_equals_analysis_union": True,
            "unexecuted_comparator_has_exact_analysis_sample_budget": True,
            "late_spread_cannot_change_candidate_locked_positive_rank": True,
            "candidate_locked_raw_prefix_is_hashed": True,
        },
        "case_summaries": {
            "separated_island": {
                "engineered_change_relative_seconds": [-36.0, -31.0],
                "v2_candidate": None,
                "v2_left_extent_seconds": separated_v2[
                    "adaptive_analysis_support"
                ]["left_extent_seconds"],
                "triggering_sentinel_relative_seconds": [-34.0, -30.0],
                "v3_candidate_relative_seconds": separated_candidate[
                    "relative_to_navigation_anchor_seconds"
                ],
                "analysis_unique_samples_per_channel": budget[
                    "analysis_unique_samples_per_channel"
                ],
                "matched_comparator_samples_per_channel": comparator[
                    "samples_per_channel"
                ],
            },
            "detector_envelope_gap": {
                "engineered_change_relative_seconds": [-41.0, -36.0],
                "all_sentinels_clear": True,
                "v3_candidate_relative_seconds": gap_candidate[
                    "relative_to_navigation_anchor_seconds"
                ],
            },
            "late_evidence_firewall": {
                "candidate_recording_seconds": early_v3["selected_candidate"][
                    "recording_seconds"
                ],
                "ranking_sha256": early_ranking["ranking_sha256"],
                "candidate_locked_prefix_audit": early_ranking[
                    "candidate_locked_prefix_audit"
                ],
            },
        },
        "scope_receipt": {
            "deterministic_synthetic_common17_EEG_only": True,
            "frozen_detector_anchor_and_envelope_navigation_only": True,
            "TERM_or_reference_seizure_intervals_used": False,
            "SOZ_or_channel_targets_used": False,
            "EDF_annotations_or_sidecars_used": False,
            "clinical_text_or_spreadsheets_used": False,
            "source_dev_or_source_eval_used": False,
        },
        "claim_limits": {
            "engineering_behavior_audit_only": True,
            "detector_or_localization_efficacy_estimated": False,
            "adaptive_superiority_authorized": False,
            "clinical_deployment_allowed": False,
        },
    }
    return summary, artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        summary, artifacts = _build()
        artifact_hashes: dict[str, str] = {}
        for name, payload in artifacts.items():
            path = staging / "cases" / name
            _write_json(path, payload)
            artifact_hashes[f"cases/{name}"] = _file_sha256(path)
        summary["artifact_sha256"] = artifact_hashes
        summary["receipt_sha256"] = _canonical_sha256(
            {
                key: value
                for key, value in summary.items()
                if key != "receipt_sha256"
            }
        )
        _write_json(staging / "receipt.json", summary)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(output / "receipt.json")


if __name__ == "__main__":
    main()
