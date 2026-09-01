#!/usr/bin/env python3
"""Run a bounded real-data LaBraM/TCP22 EviSOZ shadow forward.

This is a Stage-0 integration receipt, not a training entry point.  It reads
only the content-addressed bound-evidence loader, uses the audited frozen
official LaBraM encoder for the Standard19/CAR node view, and uses a separate
deterministic temporal-patch encoder for signed TCP22 edges.  The latter is a
shape/transport adapter until a Stage-1 edge encoder is authorised.  The
frozen v29 private probability artifact is used only as the baseline identity
reference; no residual is enabled and no private text is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))
# Some environments preload Python's stdlib ``code`` module.  The repository
# uses ``code.soz_pre`` as a namespace, so install the local package explicitly
# before importing any EviSOZ modules.
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

from src.evisoz.data.bound_evidence_loader import (  # noqa: E402
    iter_bound_evidence_records,
)
from src.evisoz.data.artifact_ref import canonical_json_sha256  # noqa: E402
from src.evisoz.inference.shadow import run_bound_evidence_shadow_inference  # noqa: E402
from src.evisoz.models.clinical_evidence import EviSOZEvidencePipeline  # noqa: E402
from src.evisoz.models.real_signal_adapter import (  # noqa: E402
    RealDualMontageTokenAdapter,
    TOKEN_DIM,
)
from soz_pre.constants import TCP_CHANNELS  # noqa: E402


DEFAULT_BOUND_EVIDENCE = ROOT / "outputs/evisoz_stage0_bound_evidence_v1_20260901_r50"
DEFAULT_PRIVATE_EXAMPLES = ROOT / "outputs/evisoz_stage0_private_real_examples_v1_20260831"
DEFAULT_FINDINGS = ROOT / "outputs/evisoz_stage0_findings_claim_reports_v1_20260901_r3"
DEFAULT_PRIVATE_COHORT = ROOT / "outputs/evisoz_stage0_private_real_dual_montage_v1_20260831"
DEFAULT_SPLIT = ROOT / "outputs/evisoz_stage0_private_split_v1_20260831/split_roster.json"
DEFAULT_LABRAM_MODELING = ROOT / "third_party/labram/modeling_finetune.py"
DEFAULT_LABRAM_CHECKPOINT = ROOT / "models/canonical_v29_h_d/labram-base.pth"
DEFAULT_V29_DIR = ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
DEFAULT_OUTPUT = ROOT / "outputs/evisoz_real_labram_shadow_v1_20260901"

STANDARD19 = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ",
    "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
)
LABRAM_TOKEN_DIM = 200
SAMPLE_RATE_HZ = 200
ONSET_START_SECONDS = 12.0
PATCH_SECONDS = 4
PATCH_SAMPLES = PATCH_SECONDS * SAMPLE_RATE_HZ


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
    receipt = {
        "method_id": "canonical_v29_equal_H_D_probability_ensemble",
        "combination_space": manifest.get("ensemble", {}).get("combination_space"),
        "event_count": len(events),
        "manifest_sha256": _sha256_file(manifest_path),
        "predictions_sha256": _sha256_file(tensor_path),
        "candidate_mask_sha256": _tensor_sha256(candidate_mask),
        "target_values_loaded": manifest.get("access_receipt", {}).get("private_targets_loaded", False),
        "baseline_role": "frozen_v29_identity_reference_only",
    }
    return lookup, receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--limit", type=int, default=2)
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
    baseline_lookup, baseline_receipt = _build_baseline_lookup(args.v29_dir)
    adapter = RealDualMontageTokenAdapter(
        modeling_path=args.labram_modeling,
        checkpoint_path=args.labram_checkpoint,
        onset_start_seconds=ONSET_START_SECONDS,
        projection_mode="fixed_shadow",
    )
    node_encoder = lambda waveform: adapter.encode_node_view(waveform.unsqueeze(0))
    edge_encoder = lambda waveform: adapter.encode_edge_view(waveform.unsqueeze(0))
    torch.manual_seed(20260901)
    pipeline = EviSOZEvidencePipeline(TOKEN_DIM, query_heads=4).eval()
    event_rows: list[dict[str, Any]] = []
    for record in records:
        baseline = baseline_lookup.get(record.event_id)
        if baseline is None:
            raise ValueError(f"event {record.event_id} is absent from frozen v29 baseline")

        def baseline_inference(_waveform: torch.Tensor, _observed: torch.Tensor, value=baseline) -> torch.Tensor:
            return value.clone()

        result = run_bound_evidence_shadow_inference(
            record,
            node_encoder=node_encoder,
            edge_encoder=edge_encoder,
            baseline_inference=baseline_inference,
            evidence_pipeline=pipeline,
            node_units=STANDARD19,
            edge_units=TCP_CHANNELS,
            stage0_status="NO_GO",
        )
        expected_baseline = baseline.unsqueeze(0)
        if not torch.equal(result.baseline_logits, expected_baseline):
            raise RuntimeError("baseline lookup changed during shadow forward")
        if not torch.equal(result.baseline_logits, result.baseline_logits + result.residual_delta):
            raise RuntimeError("shadow residual identity failed")
        event_rows.append(
            {
                "event_id": record.event_id,
                "linkage_group_id": record.linkage_group_id,
                "evisoz_role": record.evisoz_role,
                "standard19_shape": [1, 19, PATCH_SECONDS, TOKEN_DIM],
                "tcp22_shape": [1, 22, PATCH_SECONDS, TOKEN_DIM],
                "standard19_observed_count": sum(record.checkout_inputs()["standard19_observed_mask"]),
                "tcp22_observed_count": sum(record.checkout_inputs()["tcp22_observed_mask"]),
                "baseline_logits_sha256": _tensor_sha256(result.baseline_logits),
                "residual_delta_sha256": _tensor_sha256(result.residual_delta),
                "residual_gate_sha256": _tensor_sha256(result.residual_gate),
                "residual_identity": True,
                "qwen_generation": False,
            }
        )
    args.output.mkdir(parents=True)
    receipt = {
        "schema_version": "evisoz_real_labram_shadow_receipt_v1",
        "status": "real_data_shadow_forward_only",
        "stage0_status": "NO_GO",
        "event_count": len(event_rows),
        "events": event_rows,
        "window": {
            "source_view": "v29_reference/tcp22_context",
            "analysis_interval_seconds": [-12.0, 48.0],
            "onset_start_seconds": ONSET_START_SECONDS,
            "patch_seconds": PATCH_SECONDS,
            "sampling_rate_hz": SAMPLE_RATE_HZ,
        },
        "node_encoder": {
            "implementation": "official_labram_base_patch200_200",
            "token_dim": LABRAM_TOKEN_DIM,
            "projection": "fixed_adaptive_average_pool_200_to_128_shadow_only",
            "receipt": adapter.labram.receipt.to_dict(),
        },
        "edge_encoder": {
            "implementation": adapter.receipt.edge_encoder,
            "edge_count": 22,
            "token_dim": TOKEN_DIM,
            "endpoint_expansion": False,
            "trainable": False,
        },
        "baseline": baseline_receipt,
        "safety": {
            "formal_training": False,
            "residual_enabled": False,
            "teacher_runtime_opened": False,
            "physician_report_text_opened": False,
            "knowledge_prompt_opened": False,
            "tcp22_edge_to_node_label_expansion": False,
        },
        "receipt_sha256": "0" * 64,
    }
    receipt_for_hash = dict(receipt)
    receipt["receipt_sha256"] = canonical_json_sha256(receipt_for_hash)
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": receipt["status"], "event_count": len(event_rows), "output": str(args.output)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
