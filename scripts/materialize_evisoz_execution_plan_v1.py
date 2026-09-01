#!/usr/bin/env python3
"""Materialize the current fail-closed EviSOZ execution plan.

This is a reproducibility entry point, not a training launcher.  It reads a
validated Stage-0 gate and pipeline contract, records which stages and
experiments are runnable, and writes a content-addressed plan.  In the
current ``NO_GO`` state it performs no model/data-loader construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.stage0_gate import validate_stage0_gate  # noqa: E402
from src.evisoz.training.execution_plan import (  # noqa: E402
    build_evisoz_execution_plan,
)


DEFAULT_GATE = ROOT / "outputs/evisoz_stage0_gate_v1_20260901_r31/gate.json"
DEFAULT_CONFIG = ROOT / "configs/evisoz_structured_evidence_pipeline_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/evisoz_execution_plan_v1_20260901_r2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--pipeline-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    gate = json.loads(args.gate.resolve(strict=True).read_text(encoding="utf-8"))
    config = json.loads(
        args.pipeline_config.resolve(strict=True).read_text(encoding="utf-8")
    )
    validated_gate = validate_stage0_gate(gate)
    plan = build_evisoz_execution_plan(gate, pipeline_config=config)
    args.output.mkdir(parents=True)
    (args.output / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": plan["status"],
                "stage0_status": validated_gate["status"],
                "plan_id": plan["plan_id"],
                "receipt_sha256": plan["receipt_sha256"],
                "blocking_check_ids": plan["blocking_check_ids"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
