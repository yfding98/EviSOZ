#!/usr/bin/env python3
"""Freeze a public-only H-anchor margin threshold, then audit private transfer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_private_labram_zero_adaptation_v18 import (  # noqa: E402
    _aggregate,
)
from src.soz.metrics import deepsoz_style_top1_metrics  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs/trustworthy_soz_selective_v21_1.json"
DEFAULT_PUBLIC = (
    ROOT
    / "outputs/labram_identity_recovery_closed_replay_v16_20260812/"
    "oof_predictions.safetensors"
)
DEFAULT_PRIVATE = ROOT / "outputs/trustworthy_soz_candidate_v21_20260815"
DEFAULT_PRIVATE_MANIFEST = (
    ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814/manifest.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_selective_v21_1_20260815"


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _probability_and_margin(
    values: torch.Tensor, candidate_mask: torch.Tensor, *, values_are_logits: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if values_are_logits:
        probability = torch.softmax(values.masked_fill(~candidate_mask, -torch.inf), dim=1)
    else:
        probability = values.clone()
        probability[:, ~candidate_mask] = 0.0
        probability /= probability.sum(dim=1, keepdim=True)
    top2 = torch.topk(probability[:, candidate_mask], k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    if not torch.isfinite(probability).all() or not torch.isfinite(margin).all():
        raise RuntimeError("selective probability or margin is non-finite")
    return probability.contiguous(), margin.contiguous()


def _public_metrics(
    probability: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    retained: torch.Tensor,
) -> dict[str, object]:
    indices = torch.nonzero(retained, as_tuple=False).flatten()
    metric = deepsoz_style_top1_metrics(
        probability.index_select(0, indices),
        targets.index_select(0, indices),
        target_mask.index_select(0, indices),
        max_positive_for_neighbor=4,
    )
    return {
        "retained_count": int(indices.numel()),
        "coverage": float(indices.numel() / probability.shape[0]),
        "exact": metric.strict_accuracy,
        "neighborhood4": metric.relaxed_accuracy,
    }


def _load_private_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.resolve(strict=True).open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("private selective row must be an object")
            rows.append(value)
    if len(rows) != 51:
        raise RuntimeError("private v21 primary rows drifted")
    return rows


def _private_metrics(
    retained_event_ids: set[str], rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    selected = [row for row in rows if str(row["event_id"]) in retained_event_ids]
    if not selected:
        return {
            "event_count": 0,
            "patient_count": 0,
            "event_micro": None,
            "patient_macro": None,
            "patient_cluster_bootstrap_ci95": None,
        }
    metric_names = (
        "exact",
        "relaxed_neighbor4",
        "hit_at_3",
        "hit_at_5",
        "mrr",
        "positive_recall_at_3",
        "positive_recall_at_5",
        "laterality_agreement",
        "bucket_agreement",
        "significant_over_spread_pairwise",
    )
    return _aggregate(selected, metric_names)


def run(args: argparse.Namespace) -> dict[str, object]:
    protocol = _read_json(args.protocol)
    if protocol.get("schema_version") != "trustworthy_soz_selective_protocol_v21_1":
        raise ValueError("wrong selective v21.1 protocol schema")
    public = load_file(str(args.public.resolve(strict=True)))
    candidate_mask = public["config.candidate_mask"].bool()
    public_probability, public_margin = _probability_and_margin(
        public["oof.frozen_labram_only"].float(),
        candidate_mask,
        values_are_logits=True,
    )
    targets = public["targets"].float()
    target_mask = public["target_mask"].bool()
    quantiles = tuple(float(value) for value in protocol["public_threshold_quantiles"])
    required_accuracy = float(
        protocol["required_public_neighborhood4_accuracy_strictly_greater_than"]
    )
    minimum_count = int(protocol["minimum_retained_public_patients"])
    candidates = []
    for quantile in quantiles:
        threshold = float(
            torch.quantile(
                public_margin,
                torch.tensor(quantile),
                interpolation="higher",
            )
        )
        retained = public_margin >= threshold
        metrics = _public_metrics(
            public_probability, targets, target_mask, retained
        )
        candidates.append(
            {
                "quantile": quantile,
                "threshold": threshold,
                **metrics,
                "passes": (
                    metrics["retained_count"] >= minimum_count
                    and metrics["neighborhood4"] > required_accuracy
                ),
            }
        )
    passing = [row for row in candidates if bool(row["passes"])]
    selected = (
        sorted(
            passing,
            key=lambda row: (-int(row["retained_count"]), float(row["threshold"])),
        )[0]
        if passing
        else None
    )

    # Everything above this line is public-only threshold selection.
    private_tensors = load_file(
        str((args.private / "predictions.safetensors").resolve(strict=True))
    )
    private_probability, private_margin = _probability_and_margin(
        private_tensors["private_h_only_probability"].float(),
        candidate_mask,
        values_are_logits=False,
    )
    del private_probability
    private_manifest = _read_json(args.private_manifest)
    events = private_manifest["events"]
    if not isinstance(events, list) or len(events) != private_margin.numel():
        raise ValueError("private selective event roster drifted")
    if selected is None:
        retained_event_ids: set[str] = set()
    else:
        retained_event_ids = {
            str(events[index]["event_id"])
            for index in torch.nonzero(
                private_margin >= float(selected["threshold"]), as_tuple=False
            ).flatten().tolist()
        }

    # Private target-derived rows are loaded only after the public threshold
    # and target-blind retained event roster are fixed.
    private_rows = _load_private_rows(args.private / "evaluation_rows.jsonl")
    private_full = _private_metrics(
        {str(row["event_id"]) for row in private_rows}, private_rows
    )
    private_selective = _private_metrics(retained_event_ids, private_rows)
    payload = {
        "schema_version": "trustworthy_soz_selective_result_v21_1",
        "status": "public_developmental_threshold_private_exploratory_audit_complete",
        "arm": protocol["arm"],
        "confidence": protocol["confidence"],
        "public_candidate_operating_points": candidates,
        "selected_public_operating_point": selected,
        "public_selection_succeeded": selected is not None,
        "private_full_coverage": private_full,
        "private_selective": private_selective,
        "private_target_blind_retained_signal_event_count": len(retained_event_ids),
        "claim_audit": {
            "public_threshold_selected_without_private_targets": True,
            "private_threshold_tuned": False,
            "private_previously_opened": True,
            "external_confirmation": False,
            "public_and_private_statistical_units_match": False,
            "clinical_risk_control_guarantee": False,
        },
    }
    target = args.output.resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / "result.json").write_text(
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
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--private", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--private-manifest", type=Path, default=DEFAULT_PRIVATE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    selected = result["selected_public_operating_point"]
    private = result["private_selective"]
    print(
        json.dumps(
            {
                "public_selection_succeeded": result["public_selection_succeeded"],
                "public_operating_point": selected,
                "private_retained_primary_events": private["event_count"],
                "private_event_micro_relaxed": (
                    None
                    if private["event_micro"] is None
                    else private["event_micro"]["relaxed_neighbor4"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
