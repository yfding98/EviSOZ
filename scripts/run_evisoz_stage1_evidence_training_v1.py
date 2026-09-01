#!/usr/bin/env python3
"""Run one guarded Stage-1 EviSOZ evidence-training epoch.

The command is intentionally useful in the current blocked state: it reads
only the gate/config JSON and writes a content-addressed blocked receipt.  It
does not construct a model, optimizer, teacher runtime, Qwen instance, or
training loader until the aggregate Stage-0 gate is GO and the pipeline
config is explicitly opened for training.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.artifact_ref import canonical_json_sha256  # noqa: E402
from src.evisoz.training.evidence_trainer import run_stage1_evidence_epoch  # noqa: E402
from src.evisoz.training.stage0_guard import Stage0TrainingBlocked  # noqa: E402
from src.evisoz.training.training_receipts import validate_stage1_training_block_receipt  # noqa: E402


DEFAULT_GATE = ROOT / "outputs/evisoz_stage0_gate_v1_20260901_r59/gate.json"
DEFAULT_CONFIG = ROOT / "configs/evisoz_structured_evidence_pipeline_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/evisoz_stage1_evidence_training_v1_20260901_r3"
DEFAULT_BOUND = ROOT / "outputs/evisoz_stage0_bound_evidence_v1_20260901_r50"
DEFAULT_EXAMPLES = ROOT / "outputs/evisoz_stage0_private_real_examples_v1_20260831"
DEFAULT_FINDINGS = ROOT / "outputs/evisoz_stage0_findings_claim_reports_v1_20260901_r3"
DEFAULT_COHORT = ROOT / "outputs/evisoz_stage0_private_real_dual_montage_v1_20260831"
DEFAULT_SPLIT = ROOT / "outputs/evisoz_stage0_private_split_v1_20260831/split_roster.json"
DEFAULT_MODELING = ROOT / "third_party/labram/modeling_finetune.py"
DEFAULT_CHECKPOINT = ROOT / "models/canonical_v29_h_d/labram-base.pth"


def _hash_source(value: dict[str, Any]) -> dict[str, Any]:
    body = deepcopy(value)
    body["receipt_sha256"] = "0" * 64
    return body


def _blocked_receipt(gate: dict[str, Any], *, error: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "evisoz_stage1_training_block_receipt_v1",
        "status": "blocked_before_model_or_loader_construction",
        "stage0_gate_id": gate.get("gate_id"),
        "stage0_status": gate.get("status"),
        "blocking_check_ids": list(gate.get("blocking_check_ids", [])),
        "error": error,
        "runtime": {
            "model_constructed": False,
            "optimizer_constructed": False,
            "training_loader_opened": False,
            "teacher_runtime_opened": False,
            "qwen_generation": False,
            "residual_enabled": False,
        },
        "receipt_sha256": "0" * 64,
    }
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--pipeline-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bound-evidence", type=Path, default=DEFAULT_BOUND)
    parser.add_argument("--private-examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--findings", type=Path, default=DEFAULT_FINDINGS)
    parser.add_argument("--private-cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--labram-modeling", type=Path, default=DEFAULT_MODELING)
    parser.add_argument("--labram-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    gate = json.loads(args.gate.resolve(strict=True).read_text(encoding="utf-8"))
    config = json.loads(args.pipeline_config.resolve(strict=True).read_text(encoding="utf-8"))
    try:
        result = run_stage1_evidence_epoch(
            gate=gate,
            pipeline_config=config,
            bound_evidence_root=str(args.bound_evidence),
            private_examples_root=str(args.private_examples),
            findings_claim_report_root=str(args.findings),
            private_cohort_root=str(args.private_cohort),
            split_roster_path=str(args.split),
            modeling_path=str(args.labram_modeling),
            checkpoint_path=str(args.labram_checkpoint),
            learning_rate=args.learning_rate,
            device=args.device,
        )
        output = dict(result)
        output["schema_version"] = "evisoz_stage1_training_epoch_receipt_v1"
        output["receipt_sha256"] = "0" * 64
        output["receipt_sha256"] = canonical_json_sha256(_hash_source(output))
    except Stage0TrainingBlocked as exc:
        output = _blocked_receipt(gate, error=str(exc))
        validate_stage1_training_block_receipt(output)
    args.output.mkdir(parents=True)
    (args.output / "receipt.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": output["status"], "output": str(args.output), "stage0_status": output.get("stage0_status", gate.get("status"))}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
