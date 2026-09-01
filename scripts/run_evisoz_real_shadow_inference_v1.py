#!/usr/bin/env python3
"""Run the real-data EviSOZ shadow evidence/report pipeline.

This entry point is deliberately *not* a training or clinical-report
launcher.  It replays the content-addressed Stage-0 loader, encodes the
Standard19/CAR and signed TCP22 views with the real shadow adapter, runs the
untrained evidence decoder with the frozen canonical-v29 identity reference,
and materializes candidate-only packets/report plans.  The residual remains
an exact identity and no physician report, teacher runtime, knowledge-card
text, or Qwen generation is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch
from safetensors.torch import load_file

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
from src.evisoz.data.artifact_ref import canonical_json_sha256  # noqa: E402
from src.evisoz.data.bound_evidence_loader import (  # noqa: E402
    build_bound_evidence_loader_receipt,
    iter_bound_evidence_records,
)
from src.evisoz.evaluation.bound_evidence_eval import (  # noqa: E402
    evaluate_bound_evidence_shadow_predictions,
)
from src.evisoz.inference.patient import (  # noqa: E402
    aggregate_bound_shadow_predictions,
    build_bound_patient_qwen_shadow_inputs,
)
from src.evisoz.inference.shadow import (  # noqa: E402
    run_bound_evidence_shadow_inference,
)
from src.evisoz.models.clinical_evidence import EviSOZEvidencePipeline  # noqa: E402
from src.evisoz.models.real_signal_adapter import (  # noqa: E402
    RealDualMontageTokenAdapter,
    TOKEN_DIM,
)


STANDARD19 = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ",
    "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
)
DEFAULT_BOUND_EVIDENCE = ROOT / "outputs/evisoz_stage0_bound_evidence_v1_20260901_r50"
DEFAULT_PRIVATE_EXAMPLES = ROOT / "outputs/evisoz_stage0_private_real_examples_v1_20260831"
DEFAULT_FINDINGS = ROOT / "outputs/evisoz_stage0_findings_claim_reports_v1_20260901_r3"
DEFAULT_PRIVATE_COHORT = ROOT / "outputs/evisoz_stage0_private_real_dual_montage_v1_20260831"
DEFAULT_SPLIT = ROOT / "outputs/evisoz_stage0_private_split_v1_20260831/split_roster.json"
DEFAULT_LABRAM_MODELING = ROOT / "third_party/labram/modeling_finetune.py"
DEFAULT_LABRAM_CHECKPOINT = ROOT / "models/canonical_v29_h_d/labram-base.pth"
DEFAULT_V29_DIR = ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
DEFAULT_OUTPUT = ROOT / "outputs/evisoz_real_shadow_inference_v1_20260901_r1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _build_baseline_lookup(v29_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    manifest_path = (v29_dir / "manifest.json").resolve(strict=True)
    tensor_path = (v29_dir / "predictions.safetensors").resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = manifest.get("events")
    if not isinstance(events, list) or len(events) != 88:
        raise ValueError("frozen private v29 manifest must contain the audited 88 events")
    tensors = load_file(str(tensor_path), device="cpu")
    probabilities = tensors.get("private_portable_equal_probability")
    candidate_mask = tensors.get("candidate_mask")
    if not isinstance(probabilities, torch.Tensor) or tuple(probabilities.shape) != (88, 19):
        raise ValueError("private v29 probability tensor shape drifted")
    if not isinstance(candidate_mask, torch.Tensor) or tuple(candidate_mask.shape) != (19,):
        raise ValueError("private v29 candidate mask shape drifted")
    if not torch.isfinite(probabilities).all() or not torch.allclose(
        probabilities.sum(dim=1), torch.ones(88), atol=1e-6, rtol=0
    ):
        raise ValueError("private v29 baseline probabilities are not normalized")
    lookup = {
        str(row["event_id"]): torch.log(probabilities[index].clamp_min(1e-12)).float()
        for index, row in enumerate(events)
    }
    return lookup, {
        "method_id": "canonical_v29_equal_H_D_probability_ensemble",
        "manifest_sha256": _sha256_file(manifest_path),
        "predictions_sha256": _sha256_file(tensor_path),
        "candidate_mask_sha256": _tensor_sha256(candidate_mask),
        "target_values_loaded": manifest.get("access_receipt", {}).get("private_targets_loaded", False),
        "baseline_role": "frozen_v29_identity_reference_only",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--limit", type=int, default=88)
    parser.add_argument("--bound-evidence", type=Path, default=DEFAULT_BOUND_EVIDENCE)
    parser.add_argument("--private-examples", type=Path, default=DEFAULT_PRIVATE_EXAMPLES)
    parser.add_argument("--findings", type=Path, default=DEFAULT_FINDINGS)
    parser.add_argument("--private-cohort", type=Path, default=DEFAULT_PRIVATE_COHORT)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--labram-modeling", type=Path, default=DEFAULT_LABRAM_MODELING)
    parser.add_argument("--labram-checkpoint", type=Path, default=DEFAULT_LABRAM_CHECKPOINT)
    parser.add_argument("--v29-dir", type=Path, default=DEFAULT_V29_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.limit < 1 or args.output.exists() or args.output.is_symlink():
        raise ValueError("limit must be positive and output must not already exist")

    roots = {
        "bound_evidence_root": args.bound_evidence,
        "private_examples_root": args.private_examples,
        "findings_claim_report_root": args.findings,
        "private_cohort_root": args.private_cohort,
        "split_roster_path": args.split,
    }
    records = list(iter_bound_evidence_records(**roots, limit=args.limit))
    if not records:
        raise ValueError("bound evidence loader selected no records")
    baseline_lookup, baseline_receipt = _build_baseline_lookup(args.v29_dir)
    adapter = RealDualMontageTokenAdapter(
        modeling_path=args.labram_modeling,
        checkpoint_path=args.labram_checkpoint,
        onset_start_seconds=12.0,
        projection_mode="fixed_shadow",
    )
    pipeline = EviSOZEvidencePipeline(TOKEN_DIM, query_heads=4).eval()
    results = []
    for record in records:
        baseline = baseline_lookup.get(record.event_id)
        if baseline is None:
            raise ValueError(f"event {record.event_id} is absent from frozen v29 baseline")

        def baseline_inference(_waveform: torch.Tensor, _observed: torch.Tensor, value=baseline) -> torch.Tensor:
            return value.clone()

        result = run_bound_evidence_shadow_inference(
            record,
            node_encoder=lambda waveform: adapter.encode_node_view(waveform.unsqueeze(0)),
            edge_encoder=lambda waveform: adapter.encode_edge_view(waveform.unsqueeze(0)),
            baseline_inference=baseline_inference,
            evidence_pipeline=pipeline,
            node_units=STANDARD19,
            edge_units=TCP_CHANNELS,
            stage0_status="NO_GO",
        )
        if not torch.equal(result.baseline_logits, baseline.unsqueeze(0)):
            raise RuntimeError("baseline lookup changed during real shadow inference")
        if not torch.equal(result.baseline_logits, result.baseline_logits + result.residual_delta):
            raise RuntimeError("real shadow residual identity failed")
        results.append(result)

    loader_receipt = build_bound_evidence_loader_receipt(**roots, limit=args.limit)
    evaluation = evaluate_bound_evidence_shadow_predictions(records, results, loader_receipt=loader_receipt)
    patient_aggregates = aggregate_bound_shadow_predictions(records, results)
    patient_qwen_inputs = build_bound_patient_qwen_shadow_inputs(records)
    args.output.mkdir(parents=True)
    for result in results:
        event_dir = args.output / "events" / result.event_id
        event_dir.mkdir(parents=True)
        (event_dir / "predicted_evidence.json").write_text(
            json.dumps(result.predicted_evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (event_dir / "report_plan.json").write_text(
            json.dumps(result.report_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if result.qwen_structured_input is not None:
            (event_dir / "qwen_structured_input.json").write_text(
                json.dumps(result.qwen_structured_input, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    for linkage_group_id, packet in patient_qwen_inputs.items():
        patient_dir = args.output / "patients" / linkage_group_id
        patient_dir.mkdir(parents=True)
        (patient_dir / "qwen_patient_input.json").write_text(
            json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    serializable_aggregates = {
        key: {
            **value,
            "patient_probability": value["patient_probability"].detach().cpu().tolist()
            if isinstance(value["patient_probability"], torch.Tensor) else None,
            "weights": value["weights"].detach().cpu().tolist(),
        }
        for key, value in patient_aggregates.items()
    }
    (args.output / "loader_receipt.json").write_text(json.dumps(loader_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "patient_aggregates.json").write_text(json.dumps(serializable_aggregates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": "evisoz_real_shadow_inference_receipt_v1",
        "status": "real_data_shadow_evidence_and_report_plan_only",
        "stage0_status": "NO_GO",
        "event_count": len(results),
        "patient_count": len(patient_qwen_inputs),
        "source": {
            "bound_evidence_root": str(args.bound_evidence),
            "loader_receipt_sha256": loader_receipt["receipt_sha256"],
            "evaluation_receipt_sha256": evaluation["receipt_sha256"],
        },
        "baseline": baseline_receipt,
        "adapter": {
            "node": "official_labram_base_patch200_200",
            "node_units": list(STANDARD19),
            "edge": "signed_tcp22_independent_temporal_patch_encoder",
            "edge_units": list(TCP_CHANNELS),
            "token_dim": TOKEN_DIM,
            "projection_mode": "fixed_shadow",
            "endpoint_expansion": False,
        },
        "safety": {
            "formal_training": False,
            "residual_enabled": False,
            "residual_identity_all_events": True,
            "teacher_runtime_opened": False,
            "physician_report_text_opened": False,
            "canonical_shadow_report_opened": True,
            "knowledge_card_text_opened": False,
            "qwen_generation": False,
            "tcp22_edge_to_node_label_expansion": False,
            "predicted_packets_are_patient_facts": False,
        },
        "receipt_sha256": "0" * 64,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "event_count": len(results), "patient_count": len(patient_qwen_inputs), "output": str(args.output)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
