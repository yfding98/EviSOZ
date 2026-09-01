#!/usr/bin/env python3
"""Serve a blinded expert reader/adjudicator UI for the EEG-only S1 cohort.

The server has three mutually exclusive roles: ``reader_a``, ``reader_b``
and ``adjudicator``.  Independent-reader roles open only their own cohort
JSONL file.  The adjudicator opens both reader files only after both records
for a case are completed.  No endpoint exposes TUSZ patient IDs, EDF paths,
DeepSOZ labels, TUSZ per-channel involvement, model predictions, private data,
or another independent reader's result.

Official global event intervals are navigation anchors only.  Waveforms are
read directly from the EDF, converted to microvolts, filtered for display,
and exposed as standard-19 CAR or TCP20 bipolar traces.  Display filtering is
not a model input and never creates an SOZ target.
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
from scipy.signal import butter, resample_poly, sosfiltfilt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_tusz_eeg_only_s1_reader_pack import (  # noqa: E402
    ADJUDICATION_SCHEMA,
    ANNOTATION_SCHEMA,
    ELECTRODE_STATES,
    EVIDENCE_BASES,
    SCALP_VISIBILITY,
    SET_EXHAUSTIVENESS,
    STANDARD_19,
    TARGET_AVAILABILITY,
    validate_completed_adjudication,
    validate_completed_annotation,
)
from src.soz.geometry import TCP_20_EDGES, normalize_electrode_name  # noqa: E402


DEFAULT_PACK = ROOT / "outputs/tusz_eeg_only_s1_reader_pack_v1_20260813"
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_TEMPLATE = ROOT / "research/00_problem_definition/s1_expert_reader.html"
PACK_SCHEMA = "tusz_eeg_only_patient_s1_reader_pack_v1"
ROLES = ("reader_a", "reader_b", "adjudicator")
COHORTS = ("s1_development", "s1_calibration", "s1_locked")
PATIENT_EVENT_CONSISTENCY = (
    "consistent",
    "partially_consistent",
    "heterogeneous",
    "indeterminate",
)
UNIT_TO_UV = {"v": 1e6, "mv": 1e3, "uv": 1.0}
_REVIEWER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,63}$")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Annotation carrier is missing or is a symlink: {path.name}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL row {line_number}: {path.name}") from error
        if not isinstance(row, dict):
            raise TypeError(f"JSONL row {line_number} is not an object: {path.name}")
        rows.append(row)
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
        raise ValueError(f"Reader-pack linkage is missing or is a symlink: {path.name}")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _annotation_filename(role: str, cohort: str) -> str:
    if role in {"reader_a", "reader_b"}:
        return f"{role}_{cohort}.jsonl"
    return f"adjudication_{cohort}.jsonl"


def _safe_edf(root: Path, value: object) -> Path:
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".edf":
        raise ValueError("Unsafe reader-pack EDF path")
    candidate = root.joinpath(*relative.parts)
    for component in (candidate, *candidate.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("Expert reader cannot traverse a symlinked EDF path")
    resolved = candidate.resolve(strict=True)
    if resolved.relative_to(root).as_posix() != relative.as_posix():
        raise ValueError("Expert-reader EDF path escaped the pinned TUSZ root")
    return resolved


def _unit_scale_to_uv(value: object) -> float:
    key = str(value).strip().lower().replace("µ", "u").replace("μ", "u")
    try:
        return UNIT_TO_UV[key]
    except KeyError as error:
        raise ValueError(f"Unsupported EDF physical unit: {value!r}") from error


def _disagreement_domains(
    reader_a: Mapping[str, object], reader_b: Mapping[str, object]
) -> list[str]:
    result: list[str] = []
    comparisons = (
        ("target_availability", "target_availability"),
        ("candidate_positive_set", "candidate_positive_electrodes"),
        ("set_exhaustiveness", "set_exhaustive"),
        ("spread", "known_spread_electrodes"),
        ("event_consistency", "patient_event_consistency"),
        ("confidence", "label_confidence"),
    )
    for domain, field in comparisons:
        left, right = reader_a.get(field), reader_b.get(field)
        differs = set(left) != set(right) if isinstance(left, list) and isinstance(right, list) else left != right
        if differs:
            result.append(domain)
    left_states, right_states = reader_a.get("electrode_states"), reader_b.get("electrode_states")
    if isinstance(left_states, Mapping) and isinstance(right_states, Mapping) and any(
        left_states.get(electrode) != right_states.get(electrode)
        and {
            left_states.get(electrode),
            right_states.get(electrode),
        }
        & {"reviewed_not_candidate", "unknown", "unavailable"}
        for electrode in STANDARD_19
    ):
        result.append("reviewed_negative_or_unknown")
    return result


class S1ExpertReaderStore:
    """Role-scoped mutable annotation store with a blinded waveform API."""

    def __init__(
        self,
        *,
        reader_pack: Path,
        tusz_root: Path,
        role: str,
        cohort: str,
        reviewer_id: str,
    ) -> None:
        if role not in ROLES or cohort not in COHORTS:
            raise ValueError("Unsupported S1 expert-reader role or cohort")
        if _REVIEWER_RE.fullmatch(reviewer_id) is None:
            raise ValueError("reviewer_id must be a 2-64 character pseudonym")
        self.root = reader_pack.resolve(strict=True)
        self.tusz_root = tusz_root.resolve(strict=True)
        self.role = role
        self.cohort = cohort
        self.reviewer_id = reviewer_id
        self._lock = threading.Lock()

        manifest = _read_json(self.root / "manifest.json")
        if manifest.get("schema_version") != PACK_SCHEMA:
            raise ValueError("Expert reader requires the original EEG-only S1 pack")
        access = manifest.get("access_receipt")
        if not isinstance(access, Mapping) or any(
            access.get(field) is not False
            for field in (
                "deepsoz_target_values_loaded",
                "model_predictions_loaded",
                "private_eeg_loaded",
                "private_target_loaded",
                "automatic_soz_annotation_performed",
                "tusz_channel_time_target_values_used_for_selection_or_s1_labels",
                "tusz_channel_time_target_values_exported",
            )
        ):
            raise ValueError("Reader-pack target-free access contract changed")

        patient_rows = [
            row for row in _csv_rows(self.root / "patient_linkage.csv") if row.get("cohort") == cohort
        ]
        event_rows = [
            row for row in _csv_rows(self.root / "event_linkage.csv") if row.get("cohort") == cohort
        ]
        self.patient_by_case = {str(row["case_id"]): row for row in patient_rows}
        self.events_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
        self.event_by_id: dict[str, dict[str, str]] = {}
        for row in event_rows:
            case_id, event_id = str(row["case_id"]), str(row["event_case_id"])
            if case_id not in self.patient_by_case or event_id in self.event_by_id:
                raise ValueError("S1 expert-reader event linkage is inconsistent")
            self.events_by_case[case_id].append(row)
            self.event_by_id[event_id] = row
        for rows in self.events_by_case.values():
            rows.sort(key=lambda row: str(row["event_case_id"]))

        self.annotation_path = self.root / _annotation_filename(role, cohort)
        self.annotation_rows = _read_jsonl(self.annotation_path)
        self.annotation_by_case = {str(row["case_id"]): row for row in self.annotation_rows}
        if set(self.annotation_by_case) != set(self.patient_by_case):
            raise ValueError("Role-scoped annotation roster disagrees with patient linkage")

        self.reader_a_by_case: dict[str, dict[str, object]] = {}
        self.reader_b_by_case: dict[str, dict[str, object]] = {}
        if role == "adjudicator":
            self.reader_a_by_case = {
                str(row["case_id"]): row
                for row in _read_jsonl(self.root / _annotation_filename("reader_a", cohort))
            }
            self.reader_b_by_case = {
                str(row["case_id"]): row
                for row in _read_jsonl(self.root / _annotation_filename("reader_b", cohort))
            }

    def _case(self, case_id: object) -> str:
        value = str(case_id)
        if value not in self.patient_by_case:
            raise ValueError("Unknown blinded S1 case ID")
        return value

    def metadata(self) -> dict[str, object]:
        cases = []
        for case_id, row in sorted(
            self.annotation_by_case.items(),
            key=lambda item: int(item[1].get("presentation_order", 10**9)),
        ):
            status_field = "adjudication_status" if self.role == "adjudicator" else "review_status"
            cases.append(
                {
                    "case_id": case_id,
                    "presentation_order": row.get("presentation_order"),
                    "event_count": len(self.events_by_case[case_id]),
                    "status": row.get(status_field),
                    "ready_for_adjudication": (
                        self.role != "adjudicator" or self._readers_complete(case_id)
                    ),
                }
            )
        return {
            "schema_version": "tusz_eeg_only_s1_expert_reader_ui_v1",
            "role": self.role,
            "cohort": self.cohort,
            "reviewer_id": self.reviewer_id,
            "blinding": {
                "patient_identity_exposed": False,
                "edf_path_exposed": False,
                "deepsoz_target_exposed": False,
                "tusz_channel_involvement_exposed": False,
                "model_prediction_exposed": False,
                "other_reader_exposed_during_independent_read": False,
                "private_data_exposed": False,
            },
            "standard_19": list(STANDARD_19),
            "electrode_states": list(ELECTRODE_STATES),
            "target_availability": list(TARGET_AVAILABILITY),
            "scalp_visibility": list(SCALP_VISIBILITY),
            "set_exhaustiveness": list(SET_EXHAUSTIVENESS),
            "evidence_bases": list(EVIDENCE_BASES),
            "event_consistency": list(PATIENT_EVENT_CONSISTENCY),
            "cases": cases,
        }

    def _readers_complete(self, case_id: str) -> bool:
        return (
            self.reader_a_by_case.get(case_id, {}).get("review_status") == "completed"
            and self.reader_b_by_case.get(case_id, {}).get("review_status") == "completed"
        )

    def case_payload(self, case_id: object) -> dict[str, object]:
        case = self._case(case_id)
        if self.role == "adjudicator" and not self._readers_complete(case):
            raise ValueError("Both independent reads must close before adjudication")
        events = []
        for row in self.events_by_case[case]:
            onset = float(row["global_event_t0_sec"])
            stop = float(row["global_event_stop_sec"])
            events.append(
                {
                    "event_case_id": row["event_case_id"],
                    "event_duration_sec": stop - onset,
                    "navigation_min_relative_sec": max(-30.0, -onset),
                    "navigation_max_relative_sec": stop - onset + 60.0,
                    "event_anchor_semantics": "global seizure interval for navigation only; not an SOZ answer",
                }
            )
        result: dict[str, object] = {
            "case_id": case,
            "events": events,
            "annotation": self.annotation_by_case[case],
        }
        if self.role == "adjudicator":
            left, right = self.reader_a_by_case[case], self.reader_b_by_case[case]
            result["independent_reads"] = {
                "reader_a": left,
                "reader_b": right,
                "disagreement_domains": _disagreement_domains(left, right),
            }
        return result

    def _edf_for_event(self, event_case_id: object) -> tuple[Path, dict[str, str]]:
        event_id = str(event_case_id)
        try:
            row = self.event_by_id[event_id]
        except KeyError as error:
            raise ValueError("Unknown blinded S1 event ID") from error
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
        if mode not in {"monopolar_car19", "bipolar_tcp20"}:
            raise ValueError("Unsupported waveform mode")
        if not math.isfinite(relative_start_sec) or not math.isfinite(window_sec):
            raise ValueError("Waveform coordinates must be finite")
        if not 2.0 <= window_sec <= 120.0 or not 500 <= max_points <= 12000:
            raise ValueError("Waveform window/max_points are outside the review limits")
        edf, event = self._edf_for_event(event_case_id)
        event_t0 = float(event["global_event_t0_sec"])
        requested_start = event_t0 + relative_start_sec

        try:
            import pyedflib
        except ImportError as error:
            raise RuntimeError("pyedflib is required for the S1 expert reader") from error
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
            if any(not math.isfinite(value) or value <= 90.0 for value in rates) or len(set(rates)) != 1:
                raise ValueError("Expert reader requires one valid standard-19 sampling rate")
            sfreq = rates[0]
            counts = tuple(int(reader.getNSamples()[index]) for index in indices)
            if len(set(counts)) != 1:
                raise ValueError("Expert-reader standard-19 sample counts differ")
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
                [np.asarray(reader.readSignal(index, read_start, n_read), dtype=np.float64) for index in indices]
            )
            raw_uv = raw * scales[:, None]
            sos = butter(4, (0.5, 45.0), btype="bandpass", fs=sfreq, output="sos")
            try:
                filtered = sosfiltfilt(sos, raw_uv, axis=1)
            except ValueError as error:
                raise ValueError(
                    "Requested expert-review window is too short for stable display filtering"
                ) from error
            crop_start = int(round((absolute_start - read_start / sfreq) * sfreq))
            crop_stop = crop_start + int(round((absolute_stop - absolute_start) * sfreq))
            values = filtered[:, crop_start:crop_stop]
            car = values - values.mean(axis=0, keepdims=True)
            if mode == "monopolar_car19":
                display = car
                names = list(STANDARD_19)
            else:
                lookup = {channel: index for index, channel in enumerate(STANDARD_19)}
                display = np.stack(
                    [car[lookup[left]] - car[lookup[right]] for left, right in TCP_20_EDGES]
                )
                names = [f"{left}-{right}" for left, right in TCP_20_EDGES]
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
                "CAR19_or_TCP20;review_only_not_model_input"
            ),
        }

    def save(self, payload: Mapping[str, object], *, finalize: bool) -> dict[str, object]:
        case = self._case(payload.get("case_id"))
        with self._lock:
            existing = self.annotation_by_case[case]
            status_field = "adjudication_status" if self.role == "adjudicator" else "review_status"
            if existing.get(status_field) == "completed":
                raise ValueError("Completed independent/adjudicated records are immutable")
            updated = dict(existing)
            common_fields = (
                "scalp_visible_localizing_evidence",
                "target_availability",
                "electrode_states",
                "set_exhaustive",
                "evidence_bases",
                "known_spread_assessable",
                "known_spread_electrodes",
                "patient_event_consistency",
                "label_confidence",
            )
            for field in common_fields:
                if field in payload:
                    updated[field] = payload[field]
            states = updated.get("electrode_states")
            if not isinstance(states, Mapping) or set(states) != set(STANDARD_19):
                raise ValueError("Every standard-19 electrode requires an explicit state carrier")
            if any(value not in (*ELECTRODE_STATES, None) for value in states.values()):
                raise ValueError("Electrode state is outside the frozen four-state vocabulary")
            positives = [channel for channel in STANDARD_19 if states.get(channel) == "candidate_positive"]
            updated["candidate_positive_electrodes"] = positives

            if self.role == "adjudicator":
                if not self._readers_complete(case):
                    raise ValueError("Both independent reads must close before adjudication")
                updated["adjudicator_id"] = self.reviewer_id
                updated["reader_a_record_available"] = True
                updated["reader_b_record_available"] = True
                updated["all_available_events_reviewed"] = bool(
                    payload.get("all_available_events_reviewed")
                )
                updated["adjudication_rationale_not_for_model"] = str(
                    payload.get("adjudication_rationale_not_for_model", "")
                )
                updated["disagreement_domains"] = _disagreement_domains(
                    self.reader_a_by_case[case], self.reader_b_by_case[case]
                )
                if finalize:
                    updated["adjudication_status"] = "completed"
                    updated["adjudication_completed_at"] = datetime.now(timezone.utc).isoformat()
                    validate_completed_adjudication(
                        updated,
                        self.reader_a_by_case[case],
                        self.reader_b_by_case[case],
                    )
            else:
                expected_events = {
                    str(row["event_case_id"]) for row in self.events_by_case[case]
                }
                reviewed = payload.get("reviewed_event_case_ids", [])
                if not isinstance(reviewed, list) or set(reviewed) - expected_events:
                    raise ValueError("Reviewed-event carrier contains an event outside this case")
                updated["reviewed_event_case_ids"] = sorted(set(str(value) for value in reviewed))
                updated["all_available_events_reviewed"] = set(updated["reviewed_event_case_ids"]) == expected_events
                updated["reviewer_id"] = self.reviewer_id
                updated["unavailability_reason"] = str(payload.get("unavailability_reason", ""))
                updated["free_text_note_not_for_model"] = str(
                    payload.get("free_text_note_not_for_model", "")
                )
                if finalize:
                    updated["review_status"] = "completed"
                    updated["review_completed_at"] = datetime.now(timezone.utc).isoformat()
                    validate_completed_annotation(updated, expected_events)

            row_index = next(
                index for index, row in enumerate(self.annotation_rows) if row.get("case_id") == case
            )
            self.annotation_rows[row_index] = updated
            self.annotation_by_case[case] = updated
            _write_jsonl_atomic(self.annotation_path, self.annotation_rows)
        return {
            "case_id": case,
            "finalized": finalize,
            "status": updated[status_field],
            "annotation": updated,
        }


class S1ExpertReaderHandler(BaseHTTPRequestHandler):
    store: S1ExpertReaderStore
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
            raise ValueError("Invalid expert-reader request size")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise TypeError("Expert-reader request body must be an object")
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
                        mode=query.get("mode", ["bipolar_tcp20"])[0],
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
            if parsed.path not in {"/api/draft", "/api/finalize"}:
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            result = self.store.save(
                self._body(), finalize=parsed.path == "/api/finalize"
            )
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
    parser.add_argument("--cohort", choices=COHORTS, default="s1_development")
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    store = S1ExpertReaderStore(
        reader_pack=args.reader_pack,
        tusz_root=args.tusz_root,
        role=args.role,
        cohort=args.cohort,
        reviewer_id=args.reviewer_id,
    )
    template = args.template.resolve(strict=True).read_text(encoding="utf-8")
    S1ExpertReaderHandler.store = store
    S1ExpertReaderHandler.html = template
    server = ThreadingHTTPServer((args.host, args.port), S1ExpertReaderHandler)
    server.verbose = args.verbose
    print(f"S1 blinded expert reader: http://{args.host}:{args.port}/")
    print(f"role={args.role} cohort={args.cohort} cases={len(store.patient_by_case)}")
    print(f"annotation_file={store.annotation_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping S1 expert reader")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
