#!/usr/bin/env python3
"""Serve the phase-locked three-arm SOZ workflow/automation-bias MRMC study."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import traceback
from typing import Any, Mapping, Sequence
import urllib.parse

from scripts.audit_trustworthy_soz_workflow_mrmc_study_v1 import (
    ACTIONS,
    ARMS,
    _validate_completed,
)
from scripts.serve_trustworthy_soz_report_reader_study_v1 import (
    DEFAULT_TUSZ_ROOT,
    ReportReaderStore,
    WAVEFORM_MODES,
    _REVIEWER_RE,
    _csv_rows,
    _read_json,
    _read_jsonl,
    _safe_edf,
    _write_jsonl_atomic,
)
from src.soz.geometry import STANDARD_19


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "outputs/trustworthy_soz_workflow_mrmc_study_v1_1_20260816"
DEFAULT_TEMPLATE = ROOT / "research/00_problem_definition/trustworthy_soz_workflow_mrmc_reader.html"
ROLES = ("reader_a", "reader_b", "reader_c")
RAW_FIELDS = (
    "raw_signal_assessable",
    "raw_review_time_sec",
    "reviewed_event_case_ids",
    "raw_candidate_action",
    "raw_candidate_channels",
    "raw_confidence_1_to_5",
)
FINAL_FIELDS = (
    "post_intervention_review_time_sec",
    "final_candidate_action",
    "final_candidate_channels",
    "final_confidence_1_to_5",
    "assistance_helpfulness_1_to_5",
    "assistance_harmfulness_1_to_5",
    "report_overstatement_present",
    "would_use_in_research_review",
    "free_text_not_for_training",
)


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _valid_channels(value: object, action: object, name: str) -> list[str]:
    if not isinstance(value, list) or len(value) != len(set(value)):
        raise ValueError(f"{name} must be a unique list")
    if any(channel not in STANDARD_19 or channel == "PZ" for channel in value):
        raise ValueError(f"{name} must remain in C18")
    if action == "display_candidate" and not value:
        raise ValueError(f"{name} is required for display_candidate")
    if action != "display_candidate" and value:
        raise ValueError(f"{name} must be empty for a non-display action")
    return [str(channel) for channel in value]


class WorkflowMRMCStore(ReportReaderStore):
    """Role-scoped state. The outcome allocation key is intentionally never opened."""

    def __init__(self, *, reader_pack: Path, tusz_root: Path, role: str, reviewer_id: str) -> None:
        if role not in ROLES:
            raise ValueError("unsupported workflow-reader role")
        if _REVIEWER_RE.fullmatch(reviewer_id) is None:
            raise ValueError("reviewer_id must be a 2-64 character pseudonym")
        self.root = reader_pack.resolve(strict=True)
        self.tusz_root = tusz_root.resolve(strict=True)
        self.role = role
        self.reviewer_id = reviewer_id
        self._lock = threading.Lock()

        manifest = _read_json(self.root / "manifest.json")
        if manifest.get("schema_version") != "trustworthy_soz_workflow_mrmc_pack_v1_1":
            raise ValueError("workflow reader requires the v1.1 phase-lock pack")
        access = manifest.get("access_receipt")
        if not isinstance(access, Mapping) or access.get("target_used_for_training_calibration_or_model_selection") is not False:
            raise ValueError("workflow pack target-use boundary changed")
        self.raw_cards = _read_jsonl(self.root / "raw_case_cards.jsonl")
        self.raw_card_by_case = {str(row["case_id"]): row for row in self.raw_cards}
        if len(self.raw_card_by_case) != len(self.raw_cards):
            raise ValueError("duplicate workflow case ID")

        interventions = _read_jsonl(self.root / f"{role}_interventions.jsonl")
        self.intervention_by_case = {str(row["case_id"]): row for row in interventions}
        if len(self.intervention_by_case) != len(interventions) or set(self.intervention_by_case) != set(self.raw_card_by_case):
            raise ValueError("role intervention roster differs from raw cases")
        for case_id, intervention in self.intervention_by_case.items():
            arm = intervention.get("assigned_arm")
            if arm not in ARMS or intervention.get("must_remain_hidden_until_raw_phase_lock") is not True:
                raise ValueError(f"invalid intervention assignment for {case_id}")
            candidate = intervention.get("candidate")
            report = intervention.get("report_text_zh")
            if arm == "raw_only" and (candidate is not None or report is not None):
                raise ValueError("raw-only intervention leaks assistance")
            if arm == "candidate_only" and (not isinstance(candidate, Mapping) or report is not None):
                raise ValueError("candidate-only intervention content mismatch")
            if arm == "candidate_plus_report" and (not isinstance(candidate, Mapping) or not isinstance(report, str) or not report):
                raise ValueError("candidate+report intervention content mismatch")

        self.linkage_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in _csv_rows(self.root / "case_linkage.csv"):
            case_id = str(row.get("case_id", ""))
            if case_id not in self.raw_card_by_case:
                raise ValueError("linkage references unknown workflow case")
            self.linkage_by_case[case_id].append(row)
        if set(self.linkage_by_case) != set(self.raw_card_by_case):
            raise ValueError("raw cases and event linkage have different rosters")
        self.event_by_ui_id: dict[str, dict[str, str]] = {}
        self.ui_events_by_case: dict[str, list[dict[str, object]]] = {}
        for case_id, rows in self.linkage_by_case.items():
            rows.sort(key=lambda row: int(row["event_bundle_index"]))
            if len(rows) != int(self.raw_card_by_case[case_id]["linked_event_count"]):
                raise ValueError("linked event count differs from raw card")
            events: list[dict[str, object]] = []
            for index, row in enumerate(rows, start=1):
                event_case_id = f"{case_id}-E{index:03d}"
                self.event_by_ui_id[event_case_id] = row
                onset = float(row["global_event_t0_sec"])
                stop = float(row["global_event_stop_sec"])
                events.append(
                    {
                        "event_case_id": event_case_id,
                        "event_duration_sec": stop - onset,
                        "navigation_min_relative_sec": max(-30.0, -onset),
                        "navigation_max_relative_sec": stop - onset + 60.0,
                    }
                )
            self.ui_events_by_case[case_id] = events

        self.annotation_path = self.root / f"{role}_annotations.jsonl"
        self.annotation_rows = _read_jsonl(self.annotation_path)
        self.annotation_by_case = {str(row["case_id"]): row for row in self.annotation_rows}
        if len(self.annotation_by_case) != len(self.annotation_rows) or set(self.annotation_by_case) != set(self.raw_card_by_case):
            raise ValueError("role annotation roster differs from workflow cases")
        for case_id, row in self.annotation_by_case.items():
            if row.get("schema_version") != "trustworthy_soz_workflow_mrmc_annotation_v1_1":
                raise ValueError("workflow annotation schema drifted")
            if row.get("assigned_arm") != self.intervention_by_case[case_id].get("assigned_arm"):
                raise ValueError("annotation/intervention arm mismatch")
            if row.get("reviewer_id") not in {role, reviewer_id}:
                raise ValueError("annotation belongs to another reader")

    def metadata(self) -> dict[str, object]:
        cases = []
        for case_id, row in sorted(self.annotation_by_case.items(), key=lambda item: int(item[1]["presentation_order"])):
            cases.append(
                {
                    "case_id": case_id,
                    "presentation_order": row["presentation_order"],
                    "event_count": len(self.ui_events_by_case[case_id]),
                    "review_status": row["review_status"],
                    "raw_phase_locked": row["raw_phase_locked"] is True,
                }
            )
        return {
            "schema_version": "trustworthy_soz_workflow_mrmc_ui_v1",
            "role": self.role,
            "reviewer_id": self.reviewer_id,
            "candidate_channels": [channel for channel in STANDARD_19 if channel != "PZ"],
            "waveform_modes": list(WAVEFORM_MODES),
            "cases": cases,
            "blinding": {
                "outcome_or_reference_loaded": False,
                "other_reader_loaded": False,
                "assigned_arm_exposed_before_raw_lock": False,
                "candidate_or_report_exposed_before_raw_lock": False,
            },
        }

    def _case(self, value: object) -> str:
        case_id = str(value)
        if case_id not in self.raw_card_by_case:
            raise ValueError("unknown blinded workflow case")
        return case_id

    def case_payload(self, case_id: object) -> dict[str, object]:
        case = self._case(case_id)
        annotation = self.annotation_by_case[case]
        locked = annotation.get("raw_phase_locked") is True
        result: dict[str, object] = {
            "case_id": case,
            "raw_case_card": self.raw_card_by_case[case],
            "events": self.ui_events_by_case[case],
            "raw_phase_locked": locked,
            "annotation": self._annotation_for_client(annotation),
        }
        if locked:
            result["intervention"] = self.intervention_by_case[case]
        return result

    def _edf_for_event(self, event_case_id: object) -> tuple[Path, dict[str, str]]:
        event_id = str(event_case_id)
        if event_id not in self.event_by_ui_id:
            raise ValueError("unknown blinded workflow event")
        row = self.event_by_ui_id[event_id]
        return _safe_edf(self.tusz_root, row["relative_edf_path"]), row

    def _expected_events(self, case: str) -> set[str]:
        return {str(event["event_case_id"]) for event in self.ui_events_by_case[case]}

    def _validate_raw(self, row: Mapping[str, object], case: str, *, complete: bool) -> None:
        reviewed = row.get("reviewed_event_case_ids")
        if not isinstance(reviewed, list) or len(reviewed) != len(set(reviewed)) or set(reviewed) - self._expected_events(case):
            raise ValueError("reviewed events contain an invalid workflow event")
        action = row.get("raw_candidate_action")
        channels = row.get("raw_candidate_channels")
        if action is not None and action not in ACTIONS:
            raise ValueError("invalid raw candidate action")
        if channels is not None:
            _valid_channels(channels, action, "raw_candidate_channels")
        if complete:
            if row.get("raw_signal_assessable") not in {True, False}:
                raise ValueError("raw signal assessability is required")
            _positive_number(row.get("raw_review_time_sec"), "raw review time")
            if set(reviewed) != self._expected_events(case):
                raise ValueError("every linked event must be reviewed before intervention reveal")
            if action not in ACTIONS:
                raise ValueError("raw candidate action is required")
            _valid_channels(channels, action, "raw_candidate_channels")
            if row.get("raw_confidence_1_to_5") not in {1, 2, 3, 4, 5}:
                raise ValueError("raw confidence is required")

    def save_raw(self, payload: Mapping[str, object], *, lock_phase: bool) -> dict[str, object]:
        case = self._case(payload.get("case_id"))
        with self._lock:
            current = self.annotation_by_case[case]
            if current.get("review_status") == "completed":
                raise ValueError("completed workflow records are immutable")
            if current.get("raw_phase_locked") is True:
                raise ValueError("raw fields are immutable after intervention reveal")
            updated = dict(current)
            for field in RAW_FIELDS:
                if field in payload:
                    updated[field] = payload[field]
            updated["reviewer_id"] = self.reviewer_id
            self._validate_raw(updated, case, complete=lock_phase)
            if lock_phase:
                now = datetime.now(timezone.utc).isoformat()
                updated["raw_phase_locked"] = True
                updated["raw_phase_locked_at"] = now
                updated["intervention_revealed_at"] = now
            self._replace(case, updated)
        return self._result(case, updated)

    def save_final(self, payload: Mapping[str, object], *, finalize: bool) -> dict[str, object]:
        case = self._case(payload.get("case_id"))
        with self._lock:
            current = self.annotation_by_case[case]
            if current.get("review_status") == "completed":
                raise ValueError("completed workflow records are immutable")
            if current.get("raw_phase_locked") is not True:
                raise ValueError("intervention phase is unavailable before raw lock")
            updated = dict(current)
            for field in FINAL_FIELDS:
                if field in payload:
                    updated[field] = payload[field]
            updated["reviewer_id"] = self.reviewer_id
            if finalize:
                self._validate_raw(updated, case, complete=True)
                updated["review_completed_at"] = datetime.now(timezone.utc).isoformat()
                _validate_completed(updated)
                updated["review_status"] = "completed"
            self._replace(case, updated)
        return self._result(case, updated)

    def _replace(self, case: str, updated: dict[str, object]) -> None:
        index = next(index for index, row in enumerate(self.annotation_rows) if row.get("case_id") == case)
        self.annotation_rows[index] = updated
        self.annotation_by_case[case] = updated
        _write_jsonl_atomic(self.annotation_path, self.annotation_rows)

    def _result(self, case: str, annotation: Mapping[str, object]) -> dict[str, object]:
        result: dict[str, object] = {
            "case_id": case,
            "review_status": annotation.get("review_status"),
            "raw_phase_locked": annotation.get("raw_phase_locked") is True,
            "annotation": self._annotation_for_client(annotation),
        }
        if annotation.get("raw_phase_locked") is True:
            result["intervention"] = self.intervention_by_case[case]
        return result

    @staticmethod
    def _annotation_for_client(annotation: Mapping[str, object]) -> dict[str, object]:
        if annotation.get("raw_phase_locked") is True:
            return dict(annotation)
        allowed = {
            "schema_version",
            "case_id",
            "reviewer_id",
            "presentation_order",
            "review_status",
            "raw_phase_locked",
            "raw_phase_locked_at",
            *RAW_FIELDS,
        }
        return {field: annotation.get(field) for field in allowed}


class WorkflowMRMCHandler(BaseHTTPRequestHandler):
    store: WorkflowMRMCStore
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
        self._send(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

    def _body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length < 1 or length > 2_000_000:
            raise ValueError("invalid workflow-reader request size")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise TypeError("workflow-reader body must be an object")
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
        try:
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/raw-draft":
                result = self.store.save_raw(self._body(), lock_phase=False)
            elif path == "/api/lock-raw":
                result = self.store.save_raw(self._body(), lock_phase=True)
            elif path == "/api/final-draft":
                result = self.store.save_final(self._body(), finalize=False)
            elif path == "/api/finalize":
                result = self.store.save_final(self._body(), finalize=True)
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
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    store = WorkflowMRMCStore(reader_pack=args.reader_pack, tusz_root=args.tusz_root, role=args.role, reviewer_id=args.reviewer_id)
    WorkflowMRMCHandler.store = store
    WorkflowMRMCHandler.html = args.template.resolve(strict=True).read_text(encoding="utf-8")
    server = ThreadingHTTPServer((args.host, args.port), WorkflowMRMCHandler)
    server.verbose = args.verbose
    print(f"Trustworthy SOZ workflow MRMC reader: http://{args.host}:{args.port}/")
    print(f"role={args.role} cases={len(store.raw_card_by_case)}")
    print(f"annotation_file={store.annotation_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping workflow MRMC reader")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
