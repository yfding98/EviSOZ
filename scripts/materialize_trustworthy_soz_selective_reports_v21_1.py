#!/usr/bin/env python3
"""Materialize target-blind candidate-or-abstain reports for v21.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Sequence

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_trustworthy_soz_selective_v21_1 import (  # noqa: E402
    _probability_and_margin,
)
from src.soz.geometry import STANDARD_19  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs/trustworthy_soz_selective_reporting_v21_1.json"
DEFAULT_PUBLIC_TENSORS = (
    ROOT
    / "outputs/labram_identity_recovery_closed_replay_v16_20260812/"
    "oof_predictions.safetensors"
)
DEFAULT_PUBLIC_MANIFEST = (
    ROOT / "outputs/labram_identity_recovery_closed_replay_v16_20260812/manifest.json"
)
DEFAULT_PRIVATE_TENSORS = (
    ROOT / "outputs/trustworthy_soz_candidate_v21_20260815/predictions.safetensors"
)
DEFAULT_PRIVATE_MANIFEST = (
    ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814/manifest.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_selective_reports_v21_1_20260815"


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_public(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with safe_open(str(path.resolve(strict=True)), framework="pt", device="cpu") as source:
        logits = source.get_tensor("oof.frozen_labram_only").float()
        candidate_mask = source.get_tensor("config.candidate_mask").bool()
    return _probability_and_margin(logits, candidate_mask, values_are_logits=True)[0], candidate_mask


def _load_private(path: Path, candidate_mask: torch.Tensor) -> torch.Tensor:
    with safe_open(str(path.resolve(strict=True)), framework="pt", device="cpu") as source:
        probability = source.get_tensor("private_h_only_probability").float()
    return _probability_and_margin(
        probability, candidate_mask, values_are_logits=False
    )[0]


def _decision_record(
    *,
    cohort: str,
    unit_id: str,
    patient_id: str,
    probability: torch.Tensor,
    candidate_mask: torch.Tensor,
    threshold: float,
    top_k: int,
    pass_reason_code: str,
    abstain_reason_code: str,
) -> dict[str, object]:
    evaluable = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    scores = probability.index_select(0, evaluable)
    top = torch.topk(scores, k=min(top_k, int(evaluable.numel())))
    channel_indices = evaluable.index_select(0, top.indices)
    margin = float(top.values[0] - top.values[1])
    passed = margin >= threshold
    if passed:
        displayed = [
            {
                "channel": STANDARD_19[int(index)],
                "normalized_candidate_score": float(score),
            }
            for index, score in zip(channel_indices.tolist(), top.values.tolist())
        ]
        reason_codes = [pass_reason_code]
        clinical_text = (
            "冻结模型在C18空间显示头皮电极SOZ-reference候选："
            + "、".join(item["channel"] for item in displayed)
            + "。该候选需医生结合完整EEG与临床信息复核；margin不是错误概率。"
        )
        sentence_fact_map = [
            {
                "sentence": 1,
                "fact_paths": [
                    "decision.action",
                    "decision.displayed_candidates",
                    "uncertainty.top1_top2_margin",
                    "uncertainty.frozen_threshold",
                ],
            },
            {
                "sentence": 2,
                "fact_paths": [
                    "claim_boundary.clinician_review_required",
                    "claim_boundary.margin_is_error_probability",
                ],
            },
        ]
    else:
        displayed = []
        reason_codes = [abstain_reason_code]
        clinical_text = (
            f"Top1-Top2 score margin={margin:.6f}低于冻结阈值{threshold:.6f}，"
            "系统对SOZ候选定位弃权且不显示隐藏排名。弃权不表示不存在SOZ。"
        )
        sentence_fact_map = [
            {
                "sentence": 1,
                "fact_paths": [
                    "decision.action",
                    "decision.reason_codes",
                    "uncertainty.top1_top2_margin",
                    "uncertainty.frozen_threshold",
                ],
            },
            {
                "sentence": 2,
                "fact_paths": ["claim_boundary.abstention_means_no_soz"],
            },
        ]
    return {
        "schema_version": "trustworthy_soz_selective_report_v21_1",
        "cohort": cohort,
        "unit_id": unit_id,
        "patient_id": patient_id,
        "decision": {
            "action": "display_candidate" if passed else "localization_abstain",
            "reason_codes": reason_codes,
            "displayed_candidates": displayed,
        },
        "uncertainty": {
            "metric": "candidate_masked_softmax_top1_minus_top2_margin",
            "top1_top2_margin": margin,
            "frozen_threshold": threshold,
            "threshold_comparison": "greater_than_or_equal",
            "calibrated_error_probability": None,
        },
        "claim_boundary": {
            "output": "scalp_electrode_clinical_reference_candidate",
            "clinician_review_required": True,
            "margin_is_error_probability": False,
            "clinical_risk_control_guarantee": False,
            "abstention_means_no_soz": False,
            "concepts_causally_explain_h_only_score": False,
        },
        "clinical_text_zh": clinical_text,
        "sentence_fact_map": sentence_fact_map,
    }


def _write_jsonl(path: Path, records: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def run(args: argparse.Namespace) -> dict[str, object]:
    protocol = _read_json(args.protocol)
    if protocol.get("schema_version") != "trustworthy_soz_selective_reporting_protocol_v21_1":
        raise ValueError("wrong selective-reporting protocol schema")
    threshold = float(protocol["absolute_margin_threshold"])
    top_k = int(protocol["display_top_k"])
    pass_code = str(protocol["pass_reason_code"])
    abstain_code = str(protocol["abstain_reason_code"])

    public_probability, candidate_mask = _load_public(args.public_tensors)
    public_manifest = _read_json(args.public_manifest)
    public_ids = public_manifest.get("patient_ids")
    if not isinstance(public_ids, list) or len(public_ids) != len(public_probability):
        raise ValueError("public patient roster does not match H-only predictions")
    if len({str(value) for value in public_ids}) != len(public_ids):
        raise ValueError("public patient roster contains duplicates")

    private_probability = _load_private(args.private_tensors, candidate_mask)
    private_manifest = _read_json(args.private_manifest)
    private_events = private_manifest.get("events")
    if not isinstance(private_events, list) or len(private_events) != len(private_probability):
        raise ValueError("private target-blind event roster does not match predictions")

    public_records = [
        _decision_record(
            cohort="public_deepsoz_development",
            unit_id=str(patient_id),
            patient_id=str(patient_id),
            probability=public_probability[index],
            candidate_mask=candidate_mask,
            threshold=threshold,
            top_k=top_k,
            pass_reason_code=pass_code,
            abstain_reason_code=abstain_code,
        )
        for index, patient_id in enumerate(public_ids)
    ]
    private_records = [
        _decision_record(
            cohort="private_post_open_exploratory",
            unit_id=str(event["event_id"]),
            patient_id=str(event["patient_id"]),
            probability=private_probability[index],
            candidate_mask=candidate_mask,
            threshold=threshold,
            top_k=top_k,
            pass_reason_code=pass_code,
            abstain_reason_code=abstain_code,
        )
        for index, event in enumerate(private_events)
    ]

    def _count(records: Sequence[dict[str, object]], action: str) -> int:
        return sum(record["decision"]["action"] == action for record in records)

    payload = {
        "schema_version": "trustworthy_soz_selective_reporting_manifest_v21_1",
        "status": "target_blind_selective_reports_materialized",
        "arm": protocol["arm"],
        "threshold": threshold,
        "counts": {
            "public_total": len(public_records),
            "public_display_candidate": _count(public_records, "display_candidate"),
            "public_localization_abstain": _count(public_records, "localization_abstain"),
            "private_total": len(private_records),
            "private_display_candidate": _count(private_records, "display_candidate"),
            "private_localization_abstain": _count(private_records, "localization_abstain"),
        },
        "access_receipt": {
            "soz_target_tensor_loaded": False,
            "private_target_ledger_loaded": False,
            "private_evaluation_rows_loaded": False,
            "routing_uses_only_frozen_scores_and_absolute_margin_threshold": True,
        },
        "source_sha256": {
            "protocol": _sha256(args.protocol),
            "public_tensors": _sha256(args.public_tensors),
            "public_manifest": _sha256(args.public_manifest),
            "private_tensors": _sha256(args.private_tensors),
            "private_manifest": _sha256(args.private_manifest),
        },
        "files": {
            "public": "public_reports.jsonl",
            "private": "private_reports.jsonl",
        },
        "claim_boundary": protocol["claim_boundary"],
    }

    target = args.output.resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        _write_jsonl(staging / "public_reports.jsonl", public_records)
        _write_jsonl(staging / "private_reports.jsonl", private_records)
        (staging / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--public-tensors", type=Path, default=DEFAULT_PUBLIC_TENSORS)
    parser.add_argument("--public-manifest", type=Path, default=DEFAULT_PUBLIC_MANIFEST)
    parser.add_argument("--private-tensors", type=Path, default=DEFAULT_PRIVATE_TENSORS)
    parser.add_argument("--private-manifest", type=Path, default=DEFAULT_PRIVATE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
