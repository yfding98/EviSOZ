#!/usr/bin/env python3
"""Replay and materialize the aggregate EviSOZ Stage-0 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.stage0_gate import build_stage0_gate  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evisoz_stage0_gate_v1_20260831")
    parser.add_argument("--schema-registry", type=Path, default=ROOT / "configs/evisoz_schema_registry_v1.json")
    parser.add_argument("--public-v29-root", type=Path, default=ROOT / "outputs/evisoz_v29_public_held_fold_cache_v2_20260831")
    parser.add_argument("--private-real-cohort", type=Path, default=ROOT / "outputs/evisoz_stage0_private_real_dual_montage_v1_20260831")
    parser.add_argument("--private-split-roster", type=Path, default=ROOT / "outputs/evisoz_stage0_private_split_v1_20260831/split_roster.json")
    parser.add_argument("--private-signal-roster", type=Path, default=ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/signal_roster.csv")
    parser.add_argument("--private-target-ledger", type=Path, default=ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/target_ledger.csv")
    parser.add_argument("--private-source-manifest", type=Path, default=ROOT / "outputs/soz_pre/private_edf_soz_manifest.csv")
    parser.add_argument("--private-examples", type=Path, default=ROOT / "outputs/evisoz_stage0_private_real_examples_v1_20260831")
    parser.add_argument("--private-report-inventory", type=Path, default=ROOT / "outputs/evisoz_stage0_private_physician_report_inventory_v1_20260831/inventory.json")
    parser.add_argument("--private-report-deid", type=Path, default=ROOT / "outputs/evisoz_stage0_private_report_deid_candidates_v1_20260831")
    parser.add_argument("--private-report-mapping-intake", type=Path, default=ROOT / "outputs/evisoz_stage0_private_report_mapping_intake_v1_20260831")
    parser.add_argument("--private-report-exclusion", type=Path, default=ROOT / "outputs/private_public_mapping_split_deid_v1_20260901_r4/private_reports/exclusion_manifest.json", help="explicit unresolved-report quarantine receipt")
    parser.add_argument("--private-report-release", type=Path, help="optional externally authorized physician-report release receipt")
    parser.add_argument("--private-training-authorization", type=Path, help="optional external data-controller authorization for private clinical-label loss ports")
    parser.add_argument("--knowledge-root", type=Path, default=ROOT / "knowledge/eeg")
    parser.add_argument("--public-exposure-projection", type=Path, default=ROOT / "outputs/evisoz_public_auxiliary_exposure_projection_v1_20260831/projection.json")
    parser.add_argument("--public-v29-tusz-crosswalk", type=Path, default=ROOT / "outputs/evisoz_public_v29_tusz_crosswalk_v1_20260831/crosswalk.json")
    parser.add_argument("--public-auxiliary-field-release", type=Path, default=ROOT / "outputs/evisoz_public_auxiliary_field_release_v1_20260831/field_release.json")
    parser.add_argument("--public-overlap-audit", type=Path, help="optional dataset-authoritative overlap/identity audit receipt")
    parser.add_argument("--deterministic-signal-candidates", type=Path, default=ROOT / "outputs/evisoz_stage0_deterministic_signal_candidates_v1_20260831")
    parser.add_argument("--candidate-exposure-ledger", type=Path, default=ROOT / "outputs/evisoz_candidate_exposure_ledger_v1_20260831")
    parser.add_argument(
        "--teacher-cerebragloss",
        type=Path,
        help="optional validated development-only CerebraGloss candidate materialization",
    )
    parser.add_argument(
        "--teacher-elm",
        type=Path,
        help="optional validated development-only ELM candidate materialization",
    )
    parser.add_argument(
        "--clean-freeze-audit",
        type=Path,
        default=ROOT / "outputs/evisoz_clean_freeze_audit_v1_20260901_r2/audit.json",
        help="optional non-authorizing clean-freeze audit receipt",
    )
    parser.add_argument("--findings-claim-reports", type=Path, default=ROOT / "outputs/evisoz_stage0_findings_claim_reports_v1_20260831")
    parser.add_argument("--bound-evidence", type=Path, default=ROOT / "outputs/evisoz_stage0_bound_evidence_v1_20260831_r23")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    gate = build_stage0_gate(
        repository_root=ROOT,
        schema_registry_path=args.schema_registry,
        public_v29_root=args.public_v29_root,
        private_real_cohort_root=args.private_real_cohort,
        private_split_roster_path=args.private_split_roster,
        private_signal_roster_path=args.private_signal_roster,
        private_target_ledger_path=args.private_target_ledger,
        private_source_manifest_path=args.private_source_manifest,
        private_examples_root=args.private_examples,
        private_report_inventory_path=args.private_report_inventory,
        private_report_deid_root=args.private_report_deid,
        private_report_mapping_intake_root=args.private_report_mapping_intake,
        private_report_exclusion_path=args.private_report_exclusion,
        private_report_release_path=args.private_report_release,
        private_training_authorization_path=args.private_training_authorization,
        knowledge_root=args.knowledge_root,
        public_exposure_projection_path=args.public_exposure_projection,
        public_v29_tusz_crosswalk_path=args.public_v29_tusz_crosswalk,
        public_auxiliary_field_release_path=args.public_auxiliary_field_release,
        public_overlap_audit_path=args.public_overlap_audit,
        deterministic_signal_candidates_root=args.deterministic_signal_candidates,
        candidate_exposure_ledger_root=args.candidate_exposure_ledger,
        findings_claim_report_root=args.findings_claim_reports,
        bound_evidence_root=args.bound_evidence,
        teacher_cerebragloss_root=args.teacher_cerebragloss,
        teacher_elm_root=args.teacher_elm,
        clean_freeze_audit_path=args.clean_freeze_audit,
    )
    args.output.mkdir(parents=True)
    (args.output / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": gate["status"],
                "gate_id": gate["gate_id"],
                "check_statuses": {
                    row["check_id"]: row["status"] for row in gate["checks"]
                },
                "blocking_check_ids": gate["blocking_check_ids"],
                "receipt_sha256": gate["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
