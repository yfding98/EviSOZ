#!/usr/bin/env python3
"""Run a model-free synthetic smoke for the EEG-to-Qwen connector.

This command exercises only tensor contracts.  It does not import a Qwen
runtime, load a checkpoint, construct an optimizer, update parameters, read
patient data, or authorize generation.  It is therefore permitted while the
aggregate Stage-0 gate is ``NO_GO``.
"""

from __future__ import annotations

import argparse
import json
import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ``code`` is also a Python standard-library module.  The clean worktree's
# ``code.soz_pre`` package must be explicitly bootstrapped before importing
# ``src.evisoz`` so a fresh interpreter cannot resolve the stdlib module.
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))
if "code" not in sys.modules or not hasattr(sys.modules["code"], "__path__"):
    code_init = ROOT / "code" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "code", code_init, submodule_search_locations=[str(ROOT / "code")]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot initialize repository code package")
    code_module = importlib.util.module_from_spec(spec)
    sys.modules["code"] = code_module
    spec.loader.exec_module(code_module)

from src.evisoz.data.artifact_ref import canonical_json_sha256  # noqa: E402
from src.evisoz.models.qwen_connector import (  # noqa: E402
    DEFAULT_EVIDENCE_TOKEN_COUNT,
    QWEN3_8_27B_HIDDEN_SIZE,
    EvidenceTokenResampler,
    assemble_qwen_embedding_inputs,
    clause_mil_alignment_loss,
    evidence_guided_mask,
)


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body["receipt_sha256"] = "0" * 64
    body["receipt_sha256"] = canonical_json_sha256(body)
    return body


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def build_smoke() -> dict[str, Any]:
    torch.manual_seed(20260901)
    batch, evidence_length, evidence_dim = 2, 7, 128
    text_length = 5
    evidence = torch.randn(batch, evidence_length, evidence_dim)
    evidence_token_mask = torch.tensor(
        [
            [True, True, False, True, False, False, False],
            [False, False, False, False, False, False, False],
        ],
        dtype=torch.bool,
    )

    # The second row is deliberately all-masked to exercise the finite zero
    # sentinel path without creating an evidence fact.
    resampler = EvidenceTokenResampler(
        evidence_dim,
        output_dim=QWEN3_8_27B_HIDDEN_SIZE,
        token_count=DEFAULT_EVIDENCE_TOKEN_COUNT,
    )
    with torch.inference_mode():
        qwen_evidence = resampler(evidence, evidence_token_mask)
    _require(
        tuple(qwen_evidence.shape)
        == (batch, DEFAULT_EVIDENCE_TOKEN_COUNT, QWEN3_8_27B_HIDDEN_SIZE),
        "resampler output shape drifted",
    )
    _require(torch.isfinite(qwen_evidence).all().item(), "resampler output is non-finite")

    text_embeddings = torch.randn(batch, text_length, QWEN3_8_27B_HIDDEN_SIZE)
    text_attention_mask = torch.ones(batch, text_length, dtype=torch.bool)
    eeg_attention_mask = torch.cat(
        (
            evidence_token_mask,
            torch.zeros(
                batch,
                DEFAULT_EVIDENCE_TOKEN_COUNT - evidence_length,
                dtype=torch.bool,
            ),
        ),
        dim=1,
    )
    inputs, attention, modality = assemble_qwen_embedding_inputs(
        text_embeddings,
        qwen_evidence,
        text_attention_mask,
        eeg_attention_mask=eeg_attention_mask,
        insertion_index=2,
    )
    _require(
        tuple(inputs.shape)
        == (batch, text_length + DEFAULT_EVIDENCE_TOKEN_COUNT, QWEN3_8_27B_HIDDEN_SIZE),
        "assembled embedding shape drifted",
    )
    _require(
        tuple(attention.shape) == (batch, text_length + DEFAULT_EVIDENCE_TOKEN_COUNT),
        "assembled attention mask shape drifted",
    )
    _require(
        tuple(modality.shape) == (batch, text_length + DEFAULT_EVIDENCE_TOKEN_COUNT),
        "assembled modality mask shape drifted",
    )
    _require(
        int(modality.sum().item()) == batch * DEFAULT_EVIDENCE_TOKEN_COUNT,
        "EEG modality slots are not isolated",
    )

    local_tokens = torch.randn(batch, 6, QWEN3_8_27B_HIDDEN_SIZE)
    clause_embeddings = torch.randn(batch, 3, QWEN3_8_27B_HIDDEN_SIZE)
    token_mask = torch.tensor(
        [[True, True, True, True, False, False], [True, True, False, True, True, False]],
        dtype=torch.bool,
    )
    positive_mask = torch.zeros(batch, 3, 6, dtype=torch.bool)
    positive_mask[:, 0, 0] = True
    positive_mask[:, 0, 1] = True  # multi-positive clause, not single-patch
    positive_mask[:, 1, 2] = True
    positive_mask[:, 2, 3] = True
    loss = clause_mil_alignment_loss(
        local_tokens,
        clause_embeddings,
        positive_mask,
        token_mask=token_mask,
    )
    _require(torch.isfinite(loss).item(), "MIL loss is non-finite")

    masked, selected = evidence_guided_mask(
        local_tokens,
        token_mask,
        mask_probability=1.0,
    )
    _require(torch.equal(selected, token_mask), "evidence-guided selection drifted")
    _require(not bool((selected & ~token_mask).any().item()), "non-evidence token was selected")
    _require(
        torch.equal(masked[~token_mask], local_tokens[~token_mask]),
        "non-evidence token values were changed",
    )

    return _seal(
        {
            "schema_version": "evisoz_qwen_connector_synthetic_smoke_v1",
            "status": "synthetic_qwen_connector_smoke_pass",
            "runtime": {
                "qwen_runtime_imported": False,
                "qwen_checkpoint_loaded": False,
                "generation_performed": False,
                "optimizer_constructed": False,
                "parameter_update_performed": False,
                "patient_data_opened": False,
            },
            "contracts": {
                "evidence_input_shape": [batch, evidence_length, evidence_dim],
                "resampled_evidence_shape": list(qwen_evidence.shape),
                "assembled_embedding_shape": list(inputs.shape),
                "assembled_attention_shape": list(attention.shape),
                "assembled_modality_shape": list(modality.shape),
                "mil_loss_finite": True,
                "multi_positive_clause_exercised": True,
                "mask_isolation_verified": True,
                "all_masked_row_finite": True,
            },
            "stage0_status": "NO_GO_training_closed",
            "training_authorized": False,
            "report_text_loss_authorized": False,
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = build_smoke()
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
