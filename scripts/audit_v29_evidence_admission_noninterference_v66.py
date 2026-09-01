#!/usr/bin/env python3
"""Audit the frozen-v29 evidence gate and sidecar non-interference.

No EEG, public/private reference, foundation forward, fitting, thresholding,
or model selection is performed.  The audit (1) stress-injects arbitrary
failed/description-only sidecar evidence around the already frozen public and
private probabilities and (2) validates the admission policy on controlled
positive, chance, shifted, sparse, shortcut, and description-only scenarios.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
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

from src.soz.evidence_admission import (  # noqa: E402
    ControlledQualificationPolicy,
    EvidenceStatus,
    apply_formal_v29_firewall,
    formal_v29_evidence_receipts,
)


SCHEMA = "trustworthy_soz_v29_evidence_admission_noninterference_v66"
DEFAULT_PUBLIC = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_PRIVATE = ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_v29_evidence_admission_noninterference_v66_20260816"
SIMULATION_REPETITIONS = 5_000
INJECTION_TRIALS = 4_096
SEED = 20260866


def _upper_zero_failure_bound(total: int, *, alpha: float = 0.05) -> float:
    if total <= 0:
        raise ValueError("total must be positive")
    return float(1.0 - alpha ** (1.0 / total))


def _load_probabilities(public_directory: Path, private_directory: Path) -> tuple[torch.Tensor, torch.Tensor]:
    public = load_file(
        str((public_directory / "oof_predictions.safetensors").resolve(strict=True)),
        device="cpu",
    )["oof.portable_equal_ensemble_probability"].float()
    private = load_file(
        str((private_directory / "predictions.safetensors").resolve(strict=True)),
        device="cpu",
    )["private_portable_equal_probability"].float()
    if tuple(public.shape) != (102, 19) or tuple(private.shape) != (88, 19):
        raise ValueError("formal v29 probability roster changed")
    return public, private


def _injection_audit(probability: torch.Tensor, *, seed: int, trials: int) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    receipts = formal_v29_evidence_receipts()
    variants: tuple[object, ...] = (
        torch.randn(probability.shape, generator=generator),
        torch.full(probability.shape, 1e30),
        torch.full(probability.shape, -1e30),
        torch.full(probability.shape, float("nan")),
        torch.full(probability.shape, float("inf")),
        {"forged_candidates": ["FP1", "P8"], "gold_hit": True},
        {"known_spread": ["F7"], "clinical_confidence": 1.0},
        "propagation and cortical onset assertion",
    )
    blocked = ("M_morphology", "I_ictal_involvement", "V_learned_future", "uncertainty_proxy")
    violations = 0
    description_release_failures = 0
    for trial in range(trials):
        payload = {family: variants[(trial + offset) % len(variants)] for offset, family in enumerate(blocked)}
        payload["V_direct_waveform"] = {"target_blind_waveform_description": trial}
        payload["unknown_unauthorized_family"] = {"candidate": "F7"}
        after, released = apply_formal_v29_firewall(probability, payload, receipts)
        if not torch.equal(after, probability):
            violations += 1
        if set(released) != {"V_direct_waveform"}:
            description_release_failures += 1
    return {
        "trials": trials,
        "ranking_violations": violations,
        "description_permission_failures": description_release_failures,
        "zero_violation_exact_binomial_upper95": _upper_zero_failure_bound(trials),
        "blocked_families": list(blocked),
        "description_only_family": "V_direct_waveform",
    }


def _controlled_scenarios(*, repetitions: int, seed: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    policy = ControlledQualificationPolicy()
    rng = np.random.default_rng(seed)
    specifications = {
        "qualified_signal": {"native_p": 0.95, "transport_p": 0.90, "coverage": 0.95, "shortcut": True, "semantic": True},
        "native_chance": {"native_p": 0.50, "transport_p": 0.50, "coverage": 0.95, "shortcut": True, "semantic": True},
        "source_valid_transport_reversal": {"native_p": 0.90, "transport_p": 0.20, "coverage": 0.95, "shortcut": True, "semantic": True},
        "insufficient_coverage": {"native_p": 0.90, "transport_p": 0.90, "coverage": 0.50, "shortcut": True, "semantic": True},
        "shortcut_contaminated": {"native_p": 0.95, "transport_p": 0.95, "coverage": 0.95, "shortcut": False, "semantic": True},
        "target_blind_description": {"native_p": 0.95, "transport_p": 0.95, "coverage": 0.95, "shortcut": True, "semantic": False},
    }
    expected = {
        "qualified_signal": EvidenceStatus.ADMITTED_CLINICAL_CONCEPT,
        "native_chance": EvidenceStatus.FAIL_NATIVE,
        "source_valid_transport_reversal": EvidenceStatus.NO_GO,
        "insufficient_coverage": EvidenceStatus.NO_GO,
        "shortcut_contaminated": EvidenceStatus.NO_GO,
        "target_blind_description": EvidenceStatus.DESCRIPTION_ONLY,
    }
    rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    unsafe = set(specifications) - {"qualified_signal", "target_blind_description"}
    false_admissions = 0
    unsafe_decisions = 0
    for name, spec in specifications.items():
        counts: Counter[str] = Counter()
        correct = 0
        for repetition in range(repetitions):
            native_success = int(rng.binomial(102, spec["native_p"]))
            transport_success = int(rng.binomial(23, spec["transport_p"]))
            decision = policy.decide(
                native_successes=native_success,
                native_total=102,
                transport_successes=transport_success,
                transport_total=23,
                coverage=float(spec["coverage"]),
                shortcut_control_passed=bool(spec["shortcut"]),
                patient_semantic_claim=bool(spec["semantic"]),
            )
            counts[decision.value] += 1
            correct += int(decision == expected[name])
            if name in unsafe:
                unsafe_decisions += 1
                false_admissions += int(decision == EvidenceStatus.ADMITTED_CLINICAL_CONCEPT)
            rows.append({
                "scenario": name,
                "repetition": repetition,
                "native_successes": native_success,
                "transport_successes": transport_success,
                "decision": decision.value,
            })
        summaries[name] = {
            "expected": expected[name].value,
            "decision_counts": dict(sorted(counts.items())),
            "expected_decision_rate": correct / repetitions,
        }

    # Fixed transport harm demonstration.  This does not touch v29 or SOZ data.
    transport_n = 100_000
    truth = rng.integers(0, 2, size=transport_n)
    base_correct = rng.random(transport_n) < 0.70
    base = np.where(base_correct, truth, 1 - truth)
    reversed_correct = rng.random(transport_n) < 0.20
    shifted = np.where(reversed_correct, truth, 1 - truth)
    qualified_correct = rng.random(transport_n) < 0.90
    qualified = np.where(qualified_correct, truth, 1 - truth)
    harm = {
        "base_only_accuracy": float(np.mean(base == truth)),
        "always_admit_shifted_concept_accuracy": float(np.mean(shifted == truth)),
        "qualification_blocks_shifted_concept_accuracy": float(np.mean(base == truth)),
        "always_admit_qualified_concept_accuracy": float(np.mean(qualified == truth)),
        "qualification_admits_qualified_concept_accuracy": float(np.mean(qualified == truth)),
    }
    return {
        "repetitions_per_scenario": repetitions,
        "native_control_patients": 102,
        "transport_control_patients": 23,
        "policy": {"chance_level": policy.chance_level, "minimum_coverage": policy.minimum_coverage},
        "scenarios": summaries,
        "unsafe_false_admission_rate": false_admissions / unsafe_decisions,
        "always_admit_unsafe_rate": 1.0,
        "transport_harm_control": harm,
    }, rows


def run(*, public_directory: Path, private_directory: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    public, private = _load_probabilities(public_directory, private_directory)
    controlled, rows = _controlled_scenarios(repetitions=SIMULATION_REPETITIONS, seed=SEED)
    receipts = formal_v29_evidence_receipts()
    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_frozen_v29_evidence_admission_noninterference_audit",
        "formal_evidence_contract": [
            {
                "family": row.family,
                "status": row.status.value,
                "localization_access": row.localization_access,
                "report_permission": row.report_permission.value,
                "qualification_source": row.qualification_source,
            }
            for row in receipts
        ],
        "noninterference": {
            "public": _injection_audit(public, seed=SEED + 1, trials=INJECTION_TRIALS),
            "private": _injection_audit(private, seed=SEED + 2, trials=INJECTION_TRIALS),
        },
        "controlled_policy_validation": controlled,
        "access_receipt": {
            "frozen_public_probability_loaded": True,
            "frozen_private_target_blind_probability_loaded": True,
            "public_or_private_reference_loaded": False,
            "private_significant_or_spread_loaded": False,
            "raw_EEG_or_foundation_forward_loaded": False,
            "training_calibration_threshold_or_model_selection_performed": False,
            "formal_v29_modified_selected_or_ensembled": False,
        },
        "interpretation_boundary": {
            "controlled_binary_policy_replaces_native_M_I_V_gates": False,
            "zero_injection_violations_prove_all_software_or_clinical_safety": False,
            "description_only_evidence_explains_v29_scores": False,
            "allowed_claim": "the frozen evidence-access contract blocks audited failed sidecars from changing v29 rankings, and the controlled gate rejects known unsafe evidence conditions",
        },
        "files": {"controlled_rows": "controlled_qualification_rows.jsonl"},
    }
    return result, rows


def publish(output: Path, result: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (staging / "controlled_qualification_rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--public-directory", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--private-directory", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, rows = run(public_directory=args.public_directory, private_directory=args.private_directory)
    output = publish(args.output, result, rows)
    print(json.dumps({
        "output": str(output),
        "public_ranking_violations": result["noninterference"]["public"]["ranking_violations"],
        "private_ranking_violations": result["noninterference"]["private"]["ranking_violations"],
        "unsafe_false_admission_rate": result["controlled_policy_validation"]["unsafe_false_admission_rate"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
