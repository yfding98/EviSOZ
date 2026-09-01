#!/usr/bin/env python3
"""Audit frozen SOZ abstention with full risk--coverage curves, without selection."""

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

from scripts.audit_trustworthy_soz_selective_v21_1 import (  # noqa: E402
    _probability_and_margin,
)
from src.soz.metrics import deepsoz_style_top1_metrics  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs/trustworthy_soz_risk_coverage_audit_v22_3.json"
DEFAULT_PUBLIC = (
    ROOT
    / "outputs/labram_identity_recovery_closed_replay_v16_20260812/"
    "oof_predictions.safetensors"
)
DEFAULT_SELECTIVE_RESULT = (
    ROOT / "outputs/trustworthy_soz_selective_v21_1_20260815/result.json"
)
DEFAULT_PRIVATE = ROOT / "outputs/trustworthy_soz_candidate_v21_20260815"
DEFAULT_PRIVATE_MANIFEST = (
    ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814/manifest.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_risk_coverage_v22_3_20260815"


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.resolve(strict=True).open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected JSONL object: {path}")
            rows.append(value)
    return rows


def _validate_rows(confidence: np.ndarray, correct: np.ndarray) -> None:
    if confidence.ndim != 1 or correct.ndim != 1:
        raise ValueError("confidence and correct must be one-dimensional")
    if confidence.shape != correct.shape or confidence.size < 2:
        raise ValueError("confidence and correct must have equal length >= 2")
    if not np.isfinite(confidence).all() or not np.isfinite(correct).all():
        raise ValueError("risk--coverage inputs must be finite")
    if np.any((correct < 0.0) | (correct > 1.0)):
        raise ValueError("correct values must lie in [0,1]")


def _aurc_values(confidence: np.ndarray, correct: np.ndarray) -> tuple[float, float, float]:
    _validate_rows(confidence, correct)
    order = np.argsort(-confidence, kind="stable")
    errors = 1.0 - correct[order]
    denominators = np.arange(1, errors.size + 1, dtype=np.float64)
    aurc = float(np.mean(np.cumsum(errors) / denominators))
    oracle_errors = np.sort(1.0 - correct, kind="stable")
    oracle = float(np.mean(np.cumsum(oracle_errors) / denominators))
    return aurc, oracle, aurc - oracle


def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray) -> dict[str, object]:
    """Return every observed coverage point and discrete AURC/eAURC."""

    aurc, oracle, eaurc = _aurc_values(confidence, correct)
    order = np.argsort(-confidence, kind="stable")
    ordered_confidence = confidence[order]
    errors = 1.0 - correct[order]
    cumulative_risk = np.cumsum(errors) / np.arange(1, errors.size + 1)
    points = [
        {
            "retained_count": index + 1,
            "coverage": float((index + 1) / errors.size),
            "minimum_margin": float(ordered_confidence[index]),
            "accuracy": float(1.0 - cumulative_risk[index]),
            "selective_risk": float(cumulative_risk[index]),
        }
        for index in range(errors.size)
    ]
    return {
        "n": int(errors.size),
        "full_coverage_accuracy": float(np.mean(correct)),
        "aurc": aurc,
        "oracle_aurc_same_labels": oracle,
        "eaurc": eaurc,
        "points": points,
    }


def frozen_working_point(
    confidence: np.ndarray,
    correctness: Mapping[str, np.ndarray],
    threshold: float,
) -> dict[str, object]:
    retained = confidence >= threshold
    if not retained.any() or retained.all():
        raise ValueError("frozen working point must retain and abstain on at least one row")
    result: dict[str, object] = {
        "threshold": float(threshold),
        "total_count": int(confidence.size),
        "accepted_count": int(retained.sum()),
        "abstained_count": int((~retained).sum()),
        "coverage": float(retained.mean()),
    }
    endpoint_rows: dict[str, object] = {}
    for name, correct in correctness.items():
        _validate_rows(confidence, correct)
        accepted_accuracy = float(np.mean(correct[retained]))
        abstained_accuracy = float(np.mean(correct[~retained]))
        accepted_risk = 1.0 - accepted_accuracy
        abstained_risk = 1.0 - abstained_accuracy
        endpoint_rows[name] = {
            "accepted_accuracy": accepted_accuracy,
            "accepted_risk": accepted_risk,
            "abstained_accuracy": abstained_accuracy,
            "abstained_risk": abstained_risk,
            "abstained_minus_accepted_risk": abstained_risk - accepted_risk,
        }
    result["endpoints"] = endpoint_rows
    return result


def _bootstrap_indices(
    groups: np.ndarray, *, replicates: int, seed: int
) -> list[np.ndarray]:
    if groups.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError("bootstrap requires at least two groups")
    group_rows = {value: np.flatnonzero(groups == value) for value in unique}
    generator = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for _ in range(replicates):
        selected = generator.choice(unique, size=unique.size, replace=True)
        samples.append(np.concatenate([group_rows[value] for value in selected]))
    return samples


def _interval(values: list[float], *, expected: int) -> dict[str, object]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError("bootstrap produced no finite values")
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return {
        "ci95": [float(lower), float(upper)],
        "valid_replicates": int(finite.size),
        "requested_replicates": int(expected),
    }


def bootstrap_audit(
    confidence: np.ndarray,
    strict_correct: np.ndarray,
    relaxed_correct: np.ndarray,
    groups: np.ndarray,
    *,
    threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    metrics: dict[str, list[float]] = {
        "strict_aurc": [],
        "strict_eaurc": [],
        "relaxed_aurc": [],
        "relaxed_eaurc": [],
        "coverage": [],
        "strict_abstained_minus_accepted_risk": [],
        "relaxed_abstained_minus_accepted_risk": [],
    }
    for indices in _bootstrap_indices(groups, replicates=replicates, seed=seed):
        sampled_confidence = confidence[indices]
        sampled_strict = strict_correct[indices]
        sampled_relaxed = relaxed_correct[indices]
        strict_aurc, _, strict_eaurc = _aurc_values(
            sampled_confidence, sampled_strict
        )
        relaxed_aurc, _, relaxed_eaurc = _aurc_values(
            sampled_confidence, sampled_relaxed
        )
        metrics["strict_aurc"].append(strict_aurc)
        metrics["strict_eaurc"].append(strict_eaurc)
        metrics["relaxed_aurc"].append(relaxed_aurc)
        metrics["relaxed_eaurc"].append(relaxed_eaurc)
        retained = sampled_confidence >= threshold
        metrics["coverage"].append(float(retained.mean()))
        if retained.any() and not retained.all():
            for name, correct in (
                ("strict", sampled_strict),
                ("relaxed", sampled_relaxed),
            ):
                accepted_risk = 1.0 - float(np.mean(correct[retained]))
                abstained_risk = 1.0 - float(np.mean(correct[~retained]))
                metrics[f"{name}_abstained_minus_accepted_risk"].append(
                    abstained_risk - accepted_risk
                )
    return {
        name: _interval(values, expected=replicates)
        for name, values in metrics.items()
    }


def _public_correctness(
    probability: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    strict: list[float] = []
    relaxed: list[float] = []
    for index in range(probability.shape[0]):
        metric = deepsoz_style_top1_metrics(
            probability[index : index + 1],
            targets[index : index + 1],
            target_mask[index : index + 1],
            max_positive_for_neighbor=4,
        )
        strict.append(metric.strict_accuracy)
        relaxed.append(metric.relaxed_accuracy)
    return np.asarray(strict), np.asarray(relaxed)


def _private_arrays(
    *,
    private_dir: Path,
    manifest_path: Path,
    candidate_mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tensors = load_file(str((private_dir / "predictions.safetensors").resolve(strict=True)))
    _, all_margin = _probability_and_margin(
        tensors["private_h_only_probability"].float(),
        candidate_mask,
        values_are_logits=False,
    )
    manifest = _read_json(manifest_path)
    events = manifest.get("events")
    if not isinstance(events, list) or len(events) != all_margin.numel():
        raise ValueError("private manifest/prediction roster drifted")
    margin_by_event = {
        str(row["event_id"]): float(all_margin[index])
        for index, row in enumerate(events)
    }
    rows = _read_jsonl(private_dir / "evaluation_rows.jsonl")
    if len(rows) != 51:
        raise RuntimeError("private primary evaluation roster drifted")
    event_ids = [str(row["event_id"]) for row in rows]
    if len(set(event_ids)) != len(event_ids) or any(
        event_id not in margin_by_event for event_id in event_ids
    ):
        raise ValueError("private evaluation event identity drifted")
    confidence = np.asarray([margin_by_event[event_id] for event_id in event_ids])
    strict = np.asarray([float(row["exact"]) for row in rows])
    relaxed = np.asarray([float(row["relaxed_neighbor4"]) for row in rows])
    groups = np.asarray([str(row["patient_id"]) for row in rows])
    return confidence, strict, relaxed, groups


def run(args: argparse.Namespace) -> dict[str, object]:
    protocol = _read_json(args.protocol)
    if protocol.get("schema_version") != "trustworthy_soz_risk_coverage_audit_protocol_v22_3":
        raise ValueError("wrong v22.3 risk--coverage protocol schema")
    selective = _read_json(args.selective_result)
    if selective.get("schema_version") != "trustworthy_soz_selective_result_v21_1":
        raise ValueError("wrong frozen selective result schema")
    selected = selective.get("selected_public_operating_point")
    if not isinstance(selected, dict):
        raise ValueError("frozen selective result has no selected operating point")
    threshold = float(selected["threshold"])

    public = load_file(str(args.public.resolve(strict=True)))
    candidate_mask = public["config.candidate_mask"].bool()
    public_probability, public_margin_tensor = _probability_and_margin(
        public["oof.frozen_labram_only"].float(),
        candidate_mask,
        values_are_logits=True,
    )
    public_strict, public_relaxed = _public_correctness(
        public_probability, public["targets"].float(), public["target_mask"].bool()
    )
    public_margin = public_margin_tensor.detach().cpu().numpy().astype(np.float64)

    private_margin, private_strict, private_relaxed, private_groups = _private_arrays(
        private_dir=args.private,
        manifest_path=args.private_manifest,
        candidate_mask=candidate_mask,
    )

    bootstrap = protocol.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise TypeError("bootstrap protocol must be an object")
    replicates = (
        int(args.bootstrap_replicates)
        if args.bootstrap_replicates is not None
        else int(bootstrap["replicates"])
    )
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    seed = int(bootstrap["seed"])

    public_fixed = frozen_working_point(
        public_margin,
        {"strict": public_strict, "neighborhood4": public_relaxed},
        threshold,
    )
    private_fixed = frozen_working_point(
        private_margin,
        {"strict": private_strict, "neighborhood4": private_relaxed},
        threshold,
    )
    if public_fixed["accepted_count"] != 81 or private_fixed["accepted_count"] != 43:
        raise RuntimeError("frozen evaluable working-point counts drifted")

    payload: dict[str, object] = {
        "schema_version": "trustworthy_soz_risk_coverage_audit_result_v22_3",
        "status": "completed_post_open_descriptive_risk_coverage_audit",
        "arm": protocol["arm"],
        "confidence": protocol["confidence"],
        "frozen_threshold": threshold,
        "public_patient_level": {
            "strict": risk_coverage_curve(public_margin, public_strict),
            "neighborhood4": risk_coverage_curve(public_margin, public_relaxed),
            "frozen_working_point": public_fixed,
            "bootstrap_ci95": bootstrap_audit(
                public_margin,
                public_strict,
                public_relaxed,
                np.arange(public_margin.size),
                threshold=threshold,
                replicates=replicates,
                seed=seed,
            ),
        },
        "private_event_level_patient_clustered": {
            "event_count": int(private_margin.size),
            "patient_count": int(np.unique(private_groups).size),
            "strict": risk_coverage_curve(private_margin, private_strict),
            "neighborhood4": risk_coverage_curve(private_margin, private_relaxed),
            "frozen_working_point": private_fixed,
            "bootstrap_ci95": bootstrap_audit(
                private_margin,
                private_strict,
                private_relaxed,
                private_groups,
                threshold=threshold,
                replicates=replicates,
                seed=seed + 1,
            ),
        },
        "access_and_claim_audit": {
            "training_performed": False,
            "model_or_threshold_selection_performed": False,
            "frozen_threshold_changed": False,
            "private_threshold_tuned": False,
            "private_previously_opened": True,
            "public_is_repeatedly_used_development": True,
            "clinical_or_conformal_risk_guarantee": False,
            "full_coverage_results_remain_primary": True,
            "public_and_private_statistical_units_match": False,
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
    parser.add_argument("--selective-result", type=Path, default=DEFAULT_SELECTIVE_RESULT)
    parser.add_argument("--private", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--private-manifest", type=Path, default=DEFAULT_PRIVATE_MANIFEST)
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    public = result["public_patient_level"]
    private = result["private_event_level_patient_clustered"]
    print(
        json.dumps(
            {
                "public_relaxed_aurc": public["neighborhood4"]["aurc"],
                "public_relaxed_risk_gap": public["frozen_working_point"]["endpoints"]["neighborhood4"]["abstained_minus_accepted_risk"],
                "private_relaxed_aurc": private["neighborhood4"]["aurc"],
                "private_relaxed_risk_gap": private["frozen_working_point"]["endpoints"]["neighborhood4"]["abstained_minus_accepted_risk"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
