#!/usr/bin/env python3
"""Serve the raw-EEG-first, target-blind qualified-report reader study.

Independent reader roles open only their own annotation JSONL.  The report
card is withheld until the raw-only phase is validated and locked.  The server
never opens DeepSOZ targets, private data, TUSZ channel annotations, model
scores, correctness metrics, or the other reader's file.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import threading
import traceback
from typing import Mapping, Sequence
import urllib.parse

import numpy as np
from jsonschema import Draft202012Validator
from scipy.signal import butter, resample_poly, sosfiltfilt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_trustworthy_soz_report_reader_study_v1 import (  # noqa: E402
    ANNOTATION_SCHEMA,
    PACK_SCHEMA,
)
from src.soz.geometry import STANDARD_19, normalize_electrode_name  # noqa: E402


DEFAULT_PACK = ROOT / "outputs/trustworthy_soz_report_reader_study_v1_20260815"
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_TEMPLATE = (
    ROOT / "research/00_problem_definition/trustworthy_soz_report_reader.html"
)
ROLES = ("reader_a", "reader_b")
LAYERS = ("event_clause_factuality", "patient_candidate_utility")
UNIT_TO_UV = {"v": 1e6, "mv": 1e3, "uv": 1.0}
_REVIEWER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,63}$")

LONGITUDINAL_BIPOLAR = (
    ("FP1", "F7"),
    ("F7", "T7"),
    ("T7", "P7"),
    ("P7", "O1"),
    ("FP2", "F8"),
    ("F8", "T8"),
    ("T8", "P8"),
    ("P8", "O2"),
    ("FP1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("FP2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
)
TRANSVERSE_BIPOLAR = (
    ("FP1", "FP2"),
    ("F7", "F3"),
    ("F3", "FZ"),
    ("FZ", "F4"),
    ("F4", "F8"),
    ("T7", "C3"),
    ("C3", "CZ"),
    ("CZ", "C4"),
    ("C4", "T8"),
    ("P7", "P3"),
    ("P3", "PZ"),
    ("PZ", "P4"),
    ("P4", "P8"),
    ("O1", "O2"),
)
WAVEFORM_MODES = {
    "longitudinal_bipolar": LONGITUDINAL_BIPOLAR,
    "transverse_bipolar": TRANSVERSE_BIPOLAR,
    "average_reference": None,
}

RAW_FIELDS = (
    "raw_only_signal_assessable",
    "raw_only_assessability_reason",
    "raw_only_review_duration_sec",
    "raw_only_key_findings_not_for_model",
    "raw_only_candidate_action",
    "raw_only_candidate_channels",
    "reviewed_event_case_ids",
)
REPORT_FIELDS = (
    "report_review_duration_sec",
    "clause_ratings",
    "important_omission",
    "omission_categories",
    "omission_text_not_for_model",
    "candidate_eeg_consistency_likert_1_to_5",
    "candidate_review_usefulness_likert_1_to_5",
    "candidate_burden_acceptable",
    "abstention_display_appropriate",
    "candidate_action_after_report",
    "candidate_channels_after_report",
    "safe_without_edit",
    "overall_modification_count",
    "overstatement_present",
    "free_text_note_not_for_model",
)
RAW_CLIENT_FIELDS = (
    "schema_version",
    "case_id",
    "layer",
    "presentation_order",
    "reviewer_id",
    "review_status",
    "raw_phase_locked",
    "raw_phase_locked_at",
    "report_revealed_at",
    *RAW_FIELDS,
)


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required JSON is missing or symlinked: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required JSONL is missing or symlinked: {path.name}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL row {line_number} is not an object: {path.name}")
        rows.append(value)
    return rows


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"reader linkage is missing or symlinked: {path.name}")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _safe_edf(root: Path, value: object) -> Path:
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".edf":
        raise ValueError("unsafe reader-pack EDF path")
    candidate = root.joinpath(*relative.parts)
    for component in (candidate, *candidate.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("report reader cannot traverse a symlinked EDF path")
    resolved = candidate.resolve(strict=True)
    if resolved.relative_to(root).as_posix() != relative.as_posix():
        raise ValueError("report-reader EDF path escaped the pinned TUSZ root")
    return resolved


def _unit_scale_to_uv(value: object) -> float:
    key = str(value).strip().lower().replace("µ", "u").replace("μ", "u")
    try:
        return UNIT_TO_UV[key]
    except KeyError as error:
        raise ValueError(f"unsupported EDF physical unit: {value!r}") from error


def _finite_nonnegative(value: object, *, name: str, strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (strictly_positive and result <= 0):
        raise ValueError(f"{name} is outside its valid range")
    return result


def _validate_raw_lock(
    row: Mapping[str, object], *, expected_events: set[str]
) -> None:
    assessable = row.get("raw_only_signal_assessable")
    reason = row.get("raw_only_assessability_reason")
    if not isinstance(assessable, bool):
        raise ValueError("raw-only signal assessability must be completed")
    valid_reasons = {
        "assessable",
        "artifact_obscured",
        "recording_truncated",
        "missing_channel_or_reference",
        "insufficient_context",
        "other",
    }
    if reason not in valid_reasons:
        raise ValueError("raw-only assessability reason is incomplete")
    if assessable != (reason == "assessable"):
        raise ValueError("raw-only assessability and reason disagree")
    _finite_nonnegative(
        row.get("raw_only_review_duration_sec"),
        name="raw-only review duration",
        strictly_positive=True,
    )
    reviewed = row.get("reviewed_event_case_ids")
    if not isinstance(reviewed, list) or set(reviewed) != expected_events:
        raise ValueError("every case event must be reviewed before report reveal")
    action = row.get("raw_only_candidate_action")
    channels = row.get("raw_only_candidate_channels")
    if not isinstance(channels, list) or len(channels) != len(set(channels)) or any(
        channel not in STANDARD_19 or channel == "PZ" for channel in channels
    ):
        raise ValueError("raw-only candidate channels are invalid")
    if row.get("layer") == "patient_candidate_utility":
        if action not in {"display_candidate", "abstain", "indeterminate"}:
            raise ValueError("patient-candidate raw-only action is incomplete")
        if action == "display_candidate" and not channels:
            raise ValueError("display_candidate requires at least one raw-only channel")
        if action != "display_candidate" and channels:
            raise ValueError("non-display raw-only action cannot carry channels")
    elif action is not None or channels:
        raise ValueError("event-clause cases cannot collect a patient candidate")


def _validate_completed(
    row: Mapping[str, object], card: Mapping[str, object], *, expected_events: set[str]
) -> None:
    _validate_raw_lock(row, expected_events=expected_events)
    if row.get("raw_phase_locked") is not True:
        raise ValueError("raw phase must remain locked")
    _finite_nonnegative(
        row.get("report_review_duration_sec"),
        name="report review duration",
        strictly_positive=True,
    )
    card_clauses = card.get("clauses")
    ratings = row.get("clause_ratings")
    if not isinstance(card_clauses, list) or not isinstance(ratings, list):
        raise ValueError("clause ratings are unavailable")
    expected = [
        (str(clause["clause_id"]), str(clause["clause_type"]))
        for clause in card_clauses
    ]
    observed = [
        (str(rating.get("clause_id")), str(rating.get("clause_type")))
        for rating in ratings
        if isinstance(rating, Mapping)
    ]
    if observed != expected or len(ratings) != len(expected):
        raise ValueError("clause-rating identity/order differs from the sealed report card")
    for rating in ratings:
        if not isinstance(rating, Mapping):
            raise TypeError("clause rating is not an object")
        support = rating.get("support")
        material = rating.get("clinically_material_error")
        action = rating.get("proposed_action")
        if support not in {
            "supported",
            "partially_supported",
            "unsupported",
            "not_assessable",
        }:
            raise ValueError("every report clause requires a support state")
        if support == "not_assessable":
            if material is not None or action != "not_assessable":
                raise ValueError("not-assessable clause semantics are inconsistent")
        elif not isinstance(material, bool) or action not in {
            "retain",
            "minor_edit",
            "major_edit",
            "delete",
        }:
            raise ValueError("assessable clause lacks error/action review")
    omission = row.get("important_omission")
    categories = row.get("omission_categories")
    if not isinstance(omission, bool) or not isinstance(categories, list):
        raise ValueError("important-omission review is incomplete")
    if omission and not categories:
        raise ValueError("important omission requires at least one category")
    if not omission and categories:
        raise ValueError("negative omission review cannot carry categories")
    if not isinstance(row.get("safe_without_edit"), bool):
        raise ValueError("safe-without-edit review is incomplete")
    if not isinstance(row.get("overstatement_present"), bool):
        raise ValueError("overstatement review is incomplete")
    modifications = row.get("overall_modification_count")
    if not isinstance(modifications, int) or isinstance(modifications, bool) or modifications < 0:
        raise ValueError("overall modification count is incomplete")

    if row.get("layer") == "patient_candidate_utility":
        after_action = row.get("candidate_action_after_report")
        after_channels = row.get("candidate_channels_after_report")
        if after_action not in {"display_candidate", "abstain", "indeterminate"}:
            raise ValueError("post-report candidate action is incomplete")
        if not isinstance(after_channels, list) or len(after_channels) != len(set(after_channels)):
            raise ValueError("post-report candidate channel carrier is invalid")
        if any(channel not in STANDARD_19 or channel == "PZ" for channel in after_channels):
            raise ValueError("post-report candidate channel is outside C18")
        if after_action == "display_candidate" and not after_channels:
            raise ValueError("post-report display action requires at least one channel")
        if after_action != "display_candidate" and after_channels:
            raise ValueError("post-report non-display action cannot carry channels")
        usefulness = row.get("candidate_review_usefulness_likert_1_to_5")
        if not isinstance(usefulness, int) or not 1 <= usefulness <= 5:
            raise ValueError("candidate/abstention usefulness must be rated")
        card_action = card.get("candidate_display_action")
        if card_action == "display_candidate":
            consistency = row.get("candidate_eeg_consistency_likert_1_to_5")
            if not isinstance(consistency, int) or not 1 <= consistency <= 5:
                raise ValueError("displayed candidate EEG consistency must be rated")
            if not isinstance(row.get("candidate_burden_acceptable"), bool):
                raise ValueError("displayed candidate burden must be rated")
            if row.get("abstention_display_appropriate") != "not_applicable":
                raise ValueError("displayed candidate must mark abstention as not applicable")
        elif card_action == "localization_abstain":
            if row.get("candidate_eeg_consistency_likert_1_to_5") is not None:
                raise ValueError("abstention has no candidate EEG-consistency score")
            if row.get("candidate_burden_acceptable") is not None:
                raise ValueError("abstention has no candidate burden")
            if row.get("abstention_display_appropriate") not in {
                "yes",
                "no",
                "indeterminate",
            }:
                raise ValueError("abstention appropriateness must be rated")
        else:
            raise ValueError("sealed report card candidate action is invalid")
    else:
        forbidden_values = (
            row.get("candidate_eeg_consistency_likert_1_to_5"),
            row.get("candidate_review_usefulness_likert_1_to_5"),
            row.get("candidate_burden_acceptable"),
            row.get("abstention_display_appropriate"),
            row.get("candidate_action_after_report"),
        )
        if any(value is not None for value in forbidden_values) or row.get(
            "candidate_channels_after_report"
        ):
            raise ValueError("event-clause cases cannot collect candidate-utility endpoints")


class ReportReaderStore:
    """Role-scoped report-reader state with server-enforced phase locking."""

    def __init__(
        self,
        *,
        reader_pack: Path,
        tusz_root: Path,
        role: str,
        reviewer_id: str,
    ) -> None:
        if role not in ROLES:
            raise ValueError("unsupported report-reader role")
        if _REVIEWER_RE.fullmatch(reviewer_id) is None:
            raise ValueError("reviewer_id must be a 2-64 character pseudonym")
        self.root = reader_pack.resolve(strict=True)
        self.tusz_root = tusz_root.resolve(strict=True)
        self.role = role
        self.reviewer_id = reviewer_id
        self._lock = threading.Lock()

        manifest = _read_json(self.root / "manifest.json")
        if manifest.get("schema_version") != PACK_SCHEMA:
            raise ValueError("report reader requires the original v1 pack")
        access = manifest.get("access_receipt")
        if not isinstance(access, Mapping) or any(
            access.get(field) is not False
            for field in (
                "deepsoz_target_values_loaded",
                "private_eeg_or_target_loaded",
                "tusz_channel_time_target_values_loaded",
                "model_correctness_or_outcome_metrics_loaded",
                "training_calibration_or_model_selection_performed",
                "llm_annotation_performed",
            )
        ):
            raise ValueError("report-reader target-free access contract changed")
        self.cards = _read_jsonl(self.root / "report_cards.jsonl")
        self.card_by_case = {str(card["case_id"]): card for card in self.cards}
        if len(self.card_by_case) != len(self.cards):
            raise ValueError("duplicate report-reader case ID")
        for card in self.cards:
            if (
                card.get("schema_version") != "trustworthy_soz_report_reader_card_v1"
                or card.get("layer") not in LAYERS
                or card.get("candidate_display_action")
                not in {"display_candidate", "localization_abstain"}
            ):
                raise ValueError("report-reader card schema/layer/action drifted")
        self.linkage_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in _csv_rows(self.root / "case_linkage.csv"):
            case_id = str(row.get("case_id", ""))
            if case_id not in self.card_by_case:
                raise ValueError("case linkage references an unknown report card")
            self.linkage_by_case[case_id].append(row)
        if set(self.linkage_by_case) != set(self.card_by_case):
            raise ValueError("report cards and case linkage have different rosters")
        self.event_by_ui_id: dict[str, dict[str, str]] = {}
        self.ui_events_by_case: dict[str, list[dict[str, object]]] = {}
        for case_id, rows in self.linkage_by_case.items():
            rows.sort(key=lambda row: int(row["event_bundle_index"]))
            card = self.card_by_case[case_id]
            if len(rows) != card.get("linked_event_count") or any(
                row.get("layer") != card.get("layer") for row in rows
            ):
                raise ValueError("report-card event count/layer disagrees with linkage")
            events: list[dict[str, object]] = []
            for index, row in enumerate(rows, start=1):
                ui_event_id = f"{case_id}-E{index:03d}"
                if ui_event_id in self.event_by_ui_id:
                    raise AssertionError("duplicate opaque UI event ID")
                self.event_by_ui_id[ui_event_id] = row
                onset = float(row["global_event_t0_sec"])
                stop = float(row["global_event_stop_sec"])
                events.append(
                    {
                        "event_case_id": ui_event_id,
                        "event_duration_sec": stop - onset,
                        "navigation_min_relative_sec": max(-30.0, -onset),
                        "navigation_max_relative_sec": stop - onset + 60.0,
                        "event_anchor_semantics": (
                            "global seizure interval for navigation only; not SOZ or first-visible truth"
                        ),
                    }
                )
            self.ui_events_by_case[case_id] = events

        self.annotation_path = self.root / f"{role}_annotations.jsonl"
        self.annotation_rows = _read_jsonl(self.annotation_path)
        self.annotation_by_case = {
            str(row["case_id"]): row for row in self.annotation_rows
        }
        if len(self.annotation_by_case) != len(self.annotation_rows) or set(
            self.annotation_by_case
        ) != set(self.card_by_case):
            raise ValueError("role-scoped annotation roster disagrees with report cards")
        self.annotation_validator = Draft202012Validator(
            _read_json(self.root / "annotation_schema.json")
        )
        for case_id, row in self.annotation_by_case.items():
            if row.get("layer") != self.card_by_case[case_id].get("layer"):
                raise ValueError("annotation layer disagrees with report card")
            self._validate_schema(row)

    def _case(self, case_id: object) -> str:
        value = str(case_id)
        if value not in self.card_by_case:
            raise ValueError("unknown blinded report-reader case ID")
        return value

    def metadata(self) -> dict[str, object]:
        cases = []
        for case_id, row in sorted(
            self.annotation_by_case.items(),
            key=lambda item: int(item[1].get("presentation_order", 10**9)),
        ):
            cases.append(
                {
                    "case_id": case_id,
                    "layer": row.get("layer"),
                    "presentation_order": row.get("presentation_order"),
                    "event_count": len(self.ui_events_by_case[case_id]),
                    "review_status": row.get("review_status"),
                    "raw_phase_locked": row.get("raw_phase_locked") is True,
                }
            )
        return {
            "schema_version": "trustworthy_soz_report_reader_ui_v1",
            "role": self.role,
            "reviewer_id": self.reviewer_id,
            "blinding": {
                "patient_identity_exposed": False,
                "edf_path_exposed": False,
                "deepsoz_target_exposed": False,
                "tusz_channel_involvement_exposed": False,
                "private_data_exposed": False,
                "model_score_or_correctness_exposed": False,
                "other_reader_exposed": False,
                "report_exposed_before_raw_lock": False,
            },
            "standard_19": list(STANDARD_19),
            "candidate_channels": [channel for channel in STANDARD_19 if channel != "PZ"],
            "waveform_modes": list(WAVEFORM_MODES),
            "cases": cases,
        }

    def case_payload(self, case_id: object) -> dict[str, object]:
        case = self._case(case_id)
        annotation = self.annotation_by_case[case]
        locked = annotation.get("raw_phase_locked") is True
        result: dict[str, object] = {
            "case_id": case,
            "layer": annotation.get("layer"),
            "events": self.ui_events_by_case[case],
            "annotation": self._annotation_for_client(annotation),
            "report_revealed": locked,
        }
        if locked:
            result["report_card"] = self.card_by_case[case]
        return result

    def _edf_for_event(self, event_case_id: object) -> tuple[Path, dict[str, str]]:
        event_id = str(event_case_id)
        try:
            row = self.event_by_ui_id[event_id]
        except KeyError as error:
            raise ValueError("unknown blinded report-reader event ID") from error
        return _safe_edf(self.tusz_root, row["relative_edf_path"]), row

    def waveform(
        self,
        *,
        event_case_id: object,
        mode: str,
        relative_start_sec: float,
        window_sec: float,
        max_points: int,
    ) -> dict[str, object]:
        if mode not in WAVEFORM_MODES:
            raise ValueError("unsupported waveform mode")
        if not math.isfinite(relative_start_sec) or not math.isfinite(window_sec):
            raise ValueError("waveform coordinates must be finite")
        if not 2.0 <= window_sec <= 120.0 or not 500 <= max_points <= 12000:
            raise ValueError("waveform window/max_points are outside review limits")
        edf, event = self._edf_for_event(event_case_id)
        event_t0 = float(event["global_event_t0_sec"])
        requested_start = event_t0 + relative_start_sec
        try:
            import pyedflib
        except ImportError as error:
            raise RuntimeError("pyedflib is required for the report reader") from error
        reader = pyedflib.EdfReader(str(edf))
        try:
            labels = tuple(str(value).strip() for value in reader.getSignalLabels())
            candidates: dict[str, list[int]] = {channel: [] for channel in STANDARD_19}
            for index, label in enumerate(labels):
                canonical = normalize_electrode_name(label)
                if canonical in candidates:
                    candidates[canonical].append(index)
            if any(len(indices) != 1 for indices in candidates.values()):
                raise ValueError("EDF no longer has one direct channel per standard-19 electrode")
            indices = tuple(candidates[channel][0] for channel in STANDARD_19)
            rates = tuple(float(reader.getSampleFrequency(index)) for index in indices)
            if any(not math.isfinite(value) or value <= 90.0 for value in rates) or len(
                set(rates)
            ) != 1:
                raise ValueError("report reader requires one valid standard-19 sampling rate")
            sfreq = rates[0]
            counts = tuple(int(reader.getNSamples()[index]) for index in indices)
            if len(set(counts)) != 1:
                raise ValueError("report-reader standard-19 sample counts differ")
            duration = counts[0] / sfreq
            absolute_start = min(max(0.0, requested_start), max(0.0, duration - 0.1))
            absolute_stop = min(duration, absolute_start + window_sec)
            margin = min(10.0, absolute_start, max(0.0, duration - absolute_stop))
            read_start = max(0, int(math.floor((absolute_start - margin) * sfreq)))
            read_stop = min(counts[0], int(math.ceil((absolute_stop + margin) * sfreq)))
            n_read = read_stop - read_start
            scales = np.asarray(
                [_unit_scale_to_uv(reader.getPhysicalDimension(index)) for index in indices]
            )
            raw = np.stack(
                [
                    np.asarray(reader.readSignal(index, read_start, n_read), dtype=np.float64)
                    for index in indices
                ]
            )
            raw_uv = raw * scales[:, None]
            sos = butter(4, (0.5, 45.0), btype="bandpass", fs=sfreq, output="sos")
            filtered = sosfiltfilt(sos, raw_uv, axis=1)
            crop_start = int(round((absolute_start - read_start / sfreq) * sfreq))
            crop_stop = crop_start + int(round((absolute_stop - absolute_start) * sfreq))
            values = filtered[:, crop_start:crop_stop]
            car = values - values.mean(axis=0, keepdims=True)
            pairs = WAVEFORM_MODES[mode]
            if pairs is None:
                display = car
                names = list(STANDARD_19)
            else:
                lookup = {channel: index for index, channel in enumerate(STANDARD_19)}
                display = np.stack(
                    [car[lookup[left]] - car[lookup[right]] for left, right in pairs]
                )
                names = [f"{left}-{right}" for left, right in pairs]
            target_rate = min(
                sfreq,
                200.0,
                max(20.0, max_points / max(absolute_stop - absolute_start, 0.1)),
            )
            ratio = Fraction(target_rate / sfreq).limit_denominator(1000)
            if ratio.numerator != ratio.denominator:
                display = resample_poly(
                    display, ratio.numerator, ratio.denominator, axis=1
                )
            times = np.linspace(
                absolute_start - event_t0,
                absolute_stop - event_t0,
                display.shape[1],
                endpoint=False,
                dtype=np.float64,
            )
        finally:
            reader.close()
        return {
            "event_case_id": str(event_case_id),
            "mode": mode,
            "channel_names": names,
            "unit": "microvolt",
            "relative_start_sec": float(times[0]),
            "relative_stop_sec": float(absolute_stop - event_t0),
            "global_interval_relative_sec": [
                0.0,
                float(event["global_event_stop_sec"]) - event_t0,
            ],
            "times_sec": np.round(times, 4).tolist(),
            "traces_uv": np.round(np.clip(display, -2000.0, 2000.0), 3).tolist(),
            "display_processing": (
                "direct_standard19_microvolt;4th_order_zero_phase_0.5-45Hz;"
                "longitudinal_or_transverse_bipolar_or_CAR19;review_only_not_model_input"
            ),
        }

    def save_raw(self, payload: Mapping[str, object], *, lock_phase: bool) -> dict[str, object]:
        case = self._case(payload.get("case_id"))
        with self._lock:
            existing = self.annotation_by_case[case]
            if existing.get("review_status") == "completed":
                raise ValueError("completed reader records are immutable")
            if existing.get("raw_phase_locked") is True:
                raise ValueError("raw-only fields are immutable after report reveal")
            updated = dict(existing)
            for field in RAW_FIELDS:
                if field in payload:
                    updated[field] = payload[field]
            updated["reviewer_id"] = self.reviewer_id
            expected_events = {
                str(event["event_case_id"]) for event in self.ui_events_by_case[case]
            }
            reviewed = updated.get("reviewed_event_case_ids")
            if not isinstance(reviewed, list) or set(reviewed) - expected_events:
                raise ValueError("reviewed-event carrier includes an event outside this case")
            if lock_phase:
                _validate_raw_lock(updated, expected_events=expected_events)
                now = datetime.now(timezone.utc).isoformat()
                updated["raw_phase_locked"] = True
                updated["raw_phase_locked_at"] = now
                updated["report_revealed_at"] = now
            self._replace(case, updated)
        return self._result(case, updated, raw_phase_just_locked=lock_phase)

    def save_report(
        self, payload: Mapping[str, object], *, finalize: bool
    ) -> dict[str, object]:
        case = self._case(payload.get("case_id"))
        with self._lock:
            existing = self.annotation_by_case[case]
            if existing.get("review_status") == "completed":
                raise ValueError("completed reader records are immutable")
            if existing.get("raw_phase_locked") is not True:
                raise ValueError("report phase is unavailable until raw-only lock")
            updated = dict(existing)
            for field in REPORT_FIELDS:
                if field in payload:
                    updated[field] = payload[field]
            updated["reviewer_id"] = self.reviewer_id
            expected_events = {
                str(event["event_case_id"]) for event in self.ui_events_by_case[case]
            }
            if finalize:
                _validate_completed(
                    updated,
                    self.card_by_case[case],
                    expected_events=expected_events,
                )
                updated["review_status"] = "completed"
                updated["review_completed_at"] = datetime.now(timezone.utc).isoformat()
            self._replace(case, updated)
        return self._result(case, updated, raw_phase_just_locked=False)

    def _replace(self, case: str, updated: dict[str, object]) -> None:
        self._validate_schema(updated)
        row_index = next(
            index for index, row in enumerate(self.annotation_rows) if row.get("case_id") == case
        )
        self.annotation_rows[row_index] = updated
        self.annotation_by_case[case] = updated
        _write_jsonl_atomic(self.annotation_path, self.annotation_rows)

    def _validate_schema(self, row: Mapping[str, object]) -> None:
        errors = sorted(
            self.annotation_validator.iter_errors(row),
            key=lambda error: tuple(str(value) for value in error.absolute_path),
        )
        if errors:
            path = ".".join(str(value) for value in errors[0].absolute_path)
            raise ValueError(
                f"report-reader annotation schema violation at {path or '<root>'}: "
                f"{errors[0].message}"
            )

    def _result(
        self,
        case: str,
        annotation: Mapping[str, object],
        *,
        raw_phase_just_locked: bool,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "case_id": case,
            "status": annotation.get("review_status"),
            "raw_phase_locked": annotation.get("raw_phase_locked") is True,
            "raw_phase_just_locked": raw_phase_just_locked,
            "annotation": self._annotation_for_client(annotation),
        }
        if annotation.get("raw_phase_locked") is True:
            result["report_card"] = self.card_by_case[case]
        return result

    @staticmethod
    def _annotation_for_client(
        annotation: Mapping[str, object]
    ) -> dict[str, object]:
        if annotation.get("raw_phase_locked") is True:
            return dict(annotation)
        return {
            field: annotation.get(field)
            for field in RAW_CLIENT_FIELDS
        }


class ReportReaderHandler(BaseHTTPRequestHandler):
    store: ReportReaderStore
    html: str

    def log_message(self, fmt: str, *args: object) -> None:
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value: object, status: int = 200) -> None:
        self._send(
            status,
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length < 1 or length > 2_000_000:
            raise ValueError("invalid report-reader request size")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise TypeError("report-reader request body must be an object")
        return value

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send(200, self.html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/meta":
                self._json(self.store.metadata())
                return
            query = urllib.parse.parse_qs(parsed.query)
            if parsed.path == "/api/case":
                self._json(self.store.case_payload(query.get("case_id", [""])[0]))
                return
            if parsed.path == "/api/waveform":
                self._json(
                    self.store.waveform(
                        event_case_id=query.get("event_case_id", [""])[0],
                        mode=query.get("mode", ["longitudinal_bipolar"])[0],
                        relative_start_sec=float(query.get("start", ["-12"])[0]),
                        window_sec=float(query.get("window", ["30"])[0]),
                        max_points=int(query.get("max_points", ["5000"])[0]),
                    )
                )
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")
        except (ValueError, TypeError) as error:
            self._json({"ok": False, "error": str(error)}, status=400)
        except Exception as error:
            traceback.print_exc()
            self._json({"ok": False, "error": str(error)}, status=500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/api/raw-draft":
                result = self.store.save_raw(self._body(), lock_phase=False)
            elif parsed.path == "/api/lock-raw":
                result = self.store.save_raw(self._body(), lock_phase=True)
            elif parsed.path == "/api/report-draft":
                result = self.store.save_report(self._body(), finalize=False)
            elif parsed.path == "/api/finalize":
                result = self.store.save_report(self._body(), finalize=True)
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            self._json({"ok": True, **result})
        except (ValueError, TypeError) as error:
            self._json({"ok": False, "error": str(error)}, status=400)
        except Exception as error:
            traceback.print_exc()
            self._json({"ok": False, "error": str(error)}, status=500)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--reader-pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    store = ReportReaderStore(
        reader_pack=args.reader_pack,
        tusz_root=args.tusz_root,
        role=args.role,
        reviewer_id=args.reviewer_id,
    )
    template = args.template.resolve(strict=True).read_text(encoding="utf-8")
    ReportReaderHandler.store = store
    ReportReaderHandler.html = template
    server = ThreadingHTTPServer((args.host, args.port), ReportReaderHandler)
    server.verbose = args.verbose
    print(f"Trustworthy SOZ report reader: http://{args.host}:{args.port}/")
    print(f"role={args.role} cases={len(store.card_by_case)}")
    print(f"annotation_file={store.annotation_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping report reader")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
