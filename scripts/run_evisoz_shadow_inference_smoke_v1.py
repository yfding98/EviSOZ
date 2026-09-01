#!/usr/bin/env python3
"""Run a deterministic loader-backed EviSOZ shadow inference smoke test."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))
if "code" not in sys.modules or not hasattr(sys.modules["code"], "__path__"):
    code_init = ROOT / "code" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "code", code_init, submodule_search_locations=[str(ROOT / "code")]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot initialize repository code package")
    module = importlib.util.module_from_spec(spec)
    sys.modules["code"] = module
    spec.loader.exec_module(module)

from soz_pre.constants import TCP_CHANNELS  # noqa: E402
from src.evisoz.data.bound_evidence_loader import (  # noqa: E402
    build_bound_evidence_loader_receipt,
    iter_bound_evidence_records,
)
from src.evisoz.evaluation.bound_evidence_eval import (  # noqa: E402
    evaluate_bound_evidence_shadow_predictions,
)
from src.evisoz.inference.shadow import (  # noqa: E402
    run_bound_evidence_shadow_inference,
)
from src.evisoz.inference.patient import (  # noqa: E402
    aggregate_bound_shadow_predictions,
    build_bound_patient_qwen_shadow_inputs,
)
from src.evisoz.models.clinical_evidence import EviSOZEvidencePipeline  # noqa: E402


def _encoder(waveform: torch.Tensor):
    pooled = F.adaptive_avg_pool1d(waveform.unsqueeze(1), 8).squeeze(1)
    return pooled.unsqueeze(-1).repeat(1, 1, 128)


def _baseline(waveform: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    del observed
    return torch.zeros(waveform.shape[0], dtype=torch.float32)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--bound-evidence",
        type=Path,
        default=ROOT / "outputs/evisoz_stage0_bound_evidence_v1_20260901_r50",
    )
    parser.add_argument(
        "--private-examples",
        type=Path,
        default=ROOT / "outputs/evisoz_stage0_private_real_examples_v1_20260831",
    )
    parser.add_argument(
        "--findings-claim-reports",
        type=Path,
        default=ROOT / "outputs/evisoz_stage0_findings_claim_reports_v1_20260901_r3",
    )
    parser.add_argument(
        "--private-cohort",
        type=Path,
        default=ROOT / "outputs/evisoz_stage0_private_real_dual_montage_v1_20260831",
    )
    parser.add_argument(
        "--split-roster",
        type=Path,
        default=ROOT / "outputs/evisoz_stage0_private_split_v1_20260831/split_roster.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/evisoz_stage0_shadow_inference_smoke_v1_20260901_r11",
    )
    args = parser.parse_args(argv)
    if args.limit < 1 or args.output.exists() or args.output.is_symlink():
        raise ValueError("limit must be positive and output must not already exist")

    roots = {
        "bound_evidence_root": args.bound_evidence,
        "private_examples_root": args.private_examples,
        "findings_claim_report_root": args.findings_claim_reports,
        "private_cohort_root": args.private_cohort,
        "split_roster_path": args.split_roster,
    }
    records = list(iter_bound_evidence_records(**roots, limit=args.limit))
    torch.manual_seed(20260901)
    results = [
        run_bound_evidence_shadow_inference(
            record,
            node_encoder=_encoder,
            edge_encoder=_encoder,
            baseline_inference=_baseline,
            evidence_pipeline=EviSOZEvidencePipeline(128, query_heads=4),
            node_units=(
                "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ",
                "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
            ),
            edge_units=TCP_CHANNELS,
        )
        for record in records
    ]
    loader_receipt = build_bound_evidence_loader_receipt(**roots, limit=args.limit)
    evaluation = evaluate_bound_evidence_shadow_predictions(
        records,
        results,
        loader_receipt=loader_receipt,
    )
    patient_aggregates = aggregate_bound_shadow_predictions(records, results)
    patient_qwen_inputs = build_bound_patient_qwen_shadow_inputs(records)
    args.output.mkdir(parents=True)
    for result in results:
        event_dir = args.output / "events" / result.event_id
        event_dir.mkdir(parents=True)
        (event_dir / "predicted_evidence.json").write_text(
            json.dumps(result.predicted_evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (event_dir / "report_plan.json").write_text(
            json.dumps(result.report_plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if result.qwen_structured_input is not None:
            (event_dir / "qwen_structured_input.json").write_text(
                json.dumps(
                    result.qwen_structured_input,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    for linkage_group_id, packet in patient_qwen_inputs.items():
        patient_dir = args.output / "patients" / linkage_group_id
        patient_dir.mkdir(parents=True)
        (patient_dir / "qwen_patient_input.json").write_text(
            json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (args.output / "loader_receipt.json").write_text(
        json.dumps(loader_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    serializable_aggregates = {
        key: {
            **value,
            "patient_probability": (
                value["patient_probability"].detach().cpu().tolist()
                if isinstance(value["patient_probability"], torch.Tensor)
                else None
            ),
            "weights": value["weights"].detach().cpu().tolist(),
        }
        for key, value in patient_aggregates.items()
    }
    (args.output / "patient_aggregates.json").write_text(
        json.dumps(serializable_aggregates, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": evaluation["status"], "counts": evaluation["counts"], "metrics": evaluation["metrics"], "patient_qwen_input_count": len(patient_qwen_inputs), "output": str(args.output)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
