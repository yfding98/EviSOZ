#!/usr/bin/env python3
"""Run a synthetic, Stage-0-safe smoke test for the EviSOZ evidence path.

This command deliberately never opens an EDF, physician report, teacher
checkpoint, or knowledge file.  It validates the checked-in configuration and
exercises masks, TCP22 edge tokens, query outputs, and the exact v29 residual
bypass while Stage 0 is not GO.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))
# Running this file directly can import Python's stdlib ``code`` module
# before the repository package.  EviSOZ's TCP22 registry lives under the
# repository package, so install that package explicitly before importing
# any EviSOZ module.  This changes import resolution only; the smoke test
# still never opens EEG, report, teacher, or knowledge data.
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

from src.evisoz.models.clinical_evidence import (  # noqa: E402
    EviSOZEvidencePipeline,
    validate_structured_evidence_pipeline_config,
)
from src.evisoz.models.predicted_evidence import build_predicted_evidence_packet  # noqa: E402
from src.evisoz.reporting.predicted_report_plan import build_predicted_report_plan  # noqa: E402


CONFIG_PATH = ROOT / "configs/evisoz_structured_evidence_pipeline_v1.json"
SCHEMA_PATH = ROOT / "schemas/evisoz_structured_evidence_pipeline_config_v1.schema.json"


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.resolve(strict=True).read_text(encoding="utf-8"))
    schema = json.loads(args.schema.resolve(strict=True).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(config))
    if errors:
        raise ValueError(f"pipeline config schema validation failed: {errors[0].message}")
    validate_structured_evidence_pipeline_config(config)

    torch.manual_seed(0)
    batch, cells, dim = 1, 6, 128
    node = torch.randn(batch, 19, cells, dim)
    edge = torch.randn(batch, 22, cells, dim)
    node_mask = torch.ones(batch, 19, cells, dtype=torch.bool)
    edge_mask = torch.ones(batch, 22, cells, dtype=torch.bool)
    node_mask[0, 18] = False  # explicit missing PZ example
    edge_mask[0, 0] = False   # explicit missing TCP22 edge example
    z0 = torch.randn(batch, 19)
    candidate_mask = torch.ones(batch, 19, dtype=torch.bool)
    eligible = torch.ones(batch, dtype=torch.bool)

    model = EviSOZEvidencePipeline(dim, query_heads=8)
    evidence, z1, delta, gate = model(
        node,
        node_mask,
        edge_tokens=edge,
        edge_mask=edge_mask,
        z0_node=z0,
        candidate_mask=candidate_mask,
        residual_mode_eligible=eligible,
        residual_enabled=False,
        alpha=0.0,
        stage0_status=str(config["status"]),
    )
    if z1 is None or delta is None or gate is None:
        raise AssertionError("smoke pipeline did not return residual bypass tensors")
    if not torch.equal(z1, z0) or delta.any() or gate.any():
        raise AssertionError("Stage-0 blocked smoke path changed frozen v29 logits")
    tensors = {
        "evidence_tokens": evidence.evidence_tokens,
        "onset_logits": evidence.onset_logits,
        "spread_logits": evidence.spread_logits,
        "motif_logits": evidence.motif_logits,
    }
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise AssertionError("synthetic evidence outputs contain non-finite values")
    packet = build_predicted_evidence_packet(
        event_id="SYNTH-EVENT-1",
        output=evidence,
        node_mask=node_mask,
        edge_mask=edge_mask,
        candidate_node_mask=candidate_mask[0],
        node_units=[f"N{i}" for i in range(19)],
        edge_units=[f"E{i}" for i in range(22)],
        stage0_status="GO" if config["status"] == "training_enabled" else "NO_GO",
    )
    plan = build_predicted_report_plan(packet, knowledge_card_ids=("CARD.CLIN.SOZ_EZ_BOUNDARY",))

    result = {
        "status": "synthetic_smoke_pass",
        "stage0_status": config["status"],
        "baseline": config["baseline"]["model"],
        "node_shape": list(node.shape),
        "edge_shape": list(edge.shape),
        "missing_node_cells": int((~node_mask).sum().item()),
        "missing_edge_cells": int((~edge_mask).sum().item()),
        "residual_bypassed": True,
        "residual_nonzero": False,
        "query_names": list(evidence.query_names),
        "predicted_evidence_packet_id": packet["packet_id"],
        "predicted_report_plan_id": plan["plan_id"],
        "report_mode": plan["permissions"]["qwen_may_lexicalize_only"],
    }
    if args.output is not None:
        output = args.output.resolve()
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
