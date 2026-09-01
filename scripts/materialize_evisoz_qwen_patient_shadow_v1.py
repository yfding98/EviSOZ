#!/usr/bin/env python3
"""Materialize a replayable patient-level Qwen shadow packet bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.bound_evidence_loader import (  # noqa: E402
    build_bound_evidence_loader_receipt,
    iter_bound_evidence_records,
)
from src.evisoz.evaluation.bound_evidence_eval import (  # noqa: E402
    validate_bound_evidence_shadow_evaluation,
)
from src.evisoz.evaluation.patient_qwen_shadow_eval import (  # noqa: E402
    evaluate_bound_patient_qwen_shadow_inputs,
    validate_patient_qwen_shadow_evaluation,
)
from src.evisoz.reporting.qwen_patient_shadow_materialization import (  # noqa: E402
    build_qwen_patient_shadow_materialization,
    validate_qwen_patient_shadow_materialization,
)
from src.evisoz.reporting.qwen_patient_input import validate_qwen_patient_input  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"shadow JSON must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"shadow JSON must be an object: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument(
        "--shadow-root",
        type=Path,
        default=ROOT / "outputs/evisoz_stage0_shadow_inference_smoke_v1_20260901_r10",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/evisoz_stage0_qwen_patient_shadow_v1_20260901_r5",
    )
    parser.add_argument("--limit", type=int, default=88)
    parser.add_argument(
        "--evisoz-role",
        choices=("development_cv", "locked_test"),
        default=None,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 1 or args.output.exists() or args.output.is_symlink():
        raise ValueError("limit must be positive and output must not already exist")
    shadow_root = args.shadow_root.resolve(strict=True)
    source_loader = _read_json(shadow_root / "loader_receipt.json")
    source_evaluation = validate_bound_evidence_shadow_evaluation(
        _read_json(shadow_root / "evaluation.json")
    )
    roots = {
        "bound_evidence_root": ROOT / "outputs/evisoz_stage0_bound_evidence_v1_20260901_r27",
        "private_examples_root": ROOT / "outputs/evisoz_stage0_private_real_examples_v1_20260831",
        "findings_claim_report_root": ROOT / "outputs/evisoz_stage0_findings_claim_reports_v1_20260901_r3",
        "private_cohort_root": ROOT / "outputs/evisoz_stage0_private_real_dual_montage_v1_20260831",
        "split_roster_path": ROOT / "outputs/evisoz_stage0_private_split_v1_20260831/split_roster.json",
        "evisoz_role": args.evisoz_role,
        "limit": args.limit,
    }
    records = list(iter_bound_evidence_records(**roots))
    if not records:
        raise ValueError("loader selected no records")
    loader = build_bound_evidence_loader_receipt(**roots)
    if loader != source_loader:
        raise ValueError("shadow source loader receipt is stale or mismatched")
    if source_evaluation["source"]["event_ids"] != loader["selection"]["event_ids"]:
        raise ValueError("shadow evaluation is not bound to the current loader roster")

    groups = sorted({record.linkage_group_id for record in records})
    packets: dict[str, dict[str, Any]] = {}
    for group_id in groups:
        relative = PurePosixPath("patients") / group_id / "qwen_patient_input.json"
        packet = validate_qwen_patient_input(_read_json(shadow_root / relative))
        if packet["linkage_group_id"] != group_id:
            raise ValueError("source patient packet linkage drifted")
        packets[group_id] = packet

    manifest = build_qwen_patient_shadow_materialization(
        records=records,
        patient_packets=packets,
        loader_receipt=loader,
        shadow_evaluation=source_evaluation,
    )
    patient_evaluation = evaluate_bound_patient_qwen_shadow_inputs(
        records,
        packets,
        loader_receipt_sha256=loader["receipt_sha256"],
    )
    args.output.mkdir(parents=True)
    for group_id in groups:
        target = args.output / "patients" / group_id
        target.mkdir(parents=True)
        (target / "qwen_patient_input.json").write_text(
            json.dumps(packets[group_id], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "patient_qwen_evaluation.json").write_text(
        json.dumps(patient_evaluation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    replayed = validate_qwen_patient_shadow_materialization(
        _read_json(args.output / "manifest.json"),
        trusted_packets=packets,
        trusted_loader_receipt=loader,
        trusted_evaluation=source_evaluation,
        output_root=args.output,
    )
    validated_patient_evaluation = validate_patient_qwen_shadow_evaluation(
        _read_json(args.output / "patient_qwen_evaluation.json")
    )
    print(
        json.dumps(
            {
                "status": replayed["status"],
                "materialization_id": replayed["materialization_id"],
                "counts": replayed["counts"],
                "receipt_sha256": replayed["receipt_sha256"],
                "patient_evaluation_id": validated_patient_evaluation["evaluation_id"],
                "patient_evaluation_receipt_sha256": validated_patient_evaluation["receipt_sha256"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
