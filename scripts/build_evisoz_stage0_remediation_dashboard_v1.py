#!/usr/bin/env python3
"""Build a self-contained EviSOZ Stage-0 remediation dashboard HTML file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.artifact_ref import canonical_json_bytes


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_state(root: Path, gate_path: Path, packet_root: Path) -> dict[str, Any]:
    gate = _load(gate_path)
    teacher = _load(packet_root / "teacher_artifact_inventory.json")
    mapping = _load(packet_root / "private_report_mapping_resolution_packet.json")
    review = _load(packet_root / "private_report_manual_review_matrix.json")
    public = _load(packet_root / "public_overlap_audit_request.json")
    cg = next((item for item in teacher["teachers"] if item["teacher_id"] == "cerebragloss"), {})
    elm = next((item for item in teacher["teachers"] if item["teacher_id"] == "elm"), {})
    checks = {row["check_id"]: row["status"] for row in gate["checks"]}
    return {
        "gate": {
            "status": gate["status"],
            "source_gate": gate_path.relative_to(root).as_posix(),
            "blocking_check_ids": gate["blocking_check_ids"],
            "check_statuses": checks,
        },
        "teacher": {
            "cerebragloss_candidate_count": cg.get("candidate_count", 0),
            "cerebragloss_event_count": cg.get("candidate_event_count", 0),
            "elm_candidate_count": elm.get("candidate_count", 0),
            "fold_local_calibration_receipt_count": teacher["fold_local_calibration"].get("current_receipt_count", 0),
        },
        "mapping": mapping,
        "manual_review": review,
        "public": {
            "tusz_source_train_patient_count": public["known_closed_inputs"].get("tusz_source_train_patient_count", 0),
            "deepsoz_exact_overlap_patient_count": public["known_closed_inputs"].get("deepsoz_exact_overlap_patient_count", 0),
            "tuev_train_visible_overlap_patient_count": public["known_closed_inputs"].get("tuev_train_visible_overlap_patient_count", 0),
            "tuev_identity_status": public["remaining_audit_requests"][1].get("current_registry_status", "opaque"),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--gate", type=Path, default=ROOT / "outputs/evisoz_stage0_gate_v1_20260901_r38/gate.json")
    parser.add_argument("--packet", type=Path, default=ROOT / "outputs/evisoz_stage0_remediation_packet_v1_20260901_r6")
    parser.add_argument("--template", type=Path, default=ROOT / "code/data_preprocess/templates/evisoz_stage0_remediation_dashboard.html")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evisoz_stage0_remediation_packet_v1_20260901_r6/evisoz_stage0_remediation_dashboard.html")
    args = parser.parse_args(argv)
    template = args.template.resolve(strict=True).read_text(encoding="utf-8")
    marker = "__INITIAL_STATE__"
    if marker not in template:
        raise ValueError("dashboard template is missing state marker")
    state = build_state(ROOT, args.gate.resolve(strict=True), args.packet.resolve(strict=True))
    state_json = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    # Keep the embedded JSON inert if a future source field contains HTML.
    state_json = state_json.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    rendered = template.replace(marker, state_json)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"output": str(output), "gate_status": state["gate"]["status"], "blocking_check_ids": state["gate"]["blocking_check_ids"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
