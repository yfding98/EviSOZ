#!/usr/bin/env python3
"""Materialize an explicit quarantine receipt for unresolved reports.

This command reads only the controlled inventory and emits report IDs and
content hashes.  It never copies or edits source DOCX/EDF files and never
creates a patient linkage, split assignment, or release decision.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.private_physician_reports import validate_private_physician_report_inventory  # noqa: E402
from src.evisoz.data.private_report_exclusion import build_private_report_exclusion  # noqa: E402


DEFAULT_BUNDLE = ROOT / "outputs/private_public_mapping_split_deid_v1_20260901_r4"
DEFAULT_OUTPUT = DEFAULT_BUNDLE / "private_reports/exclusion_manifest.json"


def _json(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"expected regular JSON file: {path}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_BUNDLE / "private_reports/inventory.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--operator", default="dyf")
    parser.add_argument("--recorded-at-utc", default=None)
    parser.add_argument(
        "--reason",
        action="append",
        default=[],
        metavar="REPORT_ID=CODE",
        help="Optional reason override; repeat once per excluded report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inventory = validate_private_physician_report_inventory(_json(args.inventory))
    reasons = {
        "EVISOZ-PRPT-32efa0b02b8149ca70779b11": "edf_unreadable_invalid_startdate_and_name_alias",
        "EVISOZ-PRPT-3754cbc80cd1f59d67031247": "edf_missing",
        "EVISOZ-PRPT-4e79c3dac42502339e5787e5": "edf_missing_and_event_record_absent",
    }
    for override in args.reason:
        report_id, separator, code = override.partition("=")
        if not separator or not report_id or not code:
            raise SystemExit("--reason must use REPORT_ID=CODE")
        reasons[report_id] = code
    unresolved = [
        row for row in inventory["reports"] if row["association"]["status"] == "unresolved"
    ]
    entries = [
        {
            "report_id": row["report_id"],
            "document_ref": row["document_ref"],
            "exclusion_status": "excluded_unresolved",
            "exclusion_code": reasons.get(str(row["report_id"]), "unresolved_identity_or_signal_unavailable"),
            "downstream_policy": {
                "create_linkage": False,
                "create_split_assignment": False,
                "admit_to_signal_preprocessing": False,
                "admit_to_event_training": False,
                "admit_to_qwen_training": False,
                "admit_to_language_evaluation": False,
            },
        }
        for row in unresolved
    ]
    receipt = build_private_report_exclusion(
        report_inventory=inventory,
        entries=entries,
        operator=args.operator,
        recorded_at_utc=args.recorded_at_utc or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "operational_quarantine_materialized", "exclusion_id": receipt["exclusion_id"], "excluded_report_ids": [row["report_id"] for row in receipt["entries"]], "output": str(output)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
