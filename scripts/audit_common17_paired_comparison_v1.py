#!/usr/bin/env python3
"""Patient-paired strict CAR17 versus aligned official-DeepSOZ audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
from safetensors import safe_open
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sota_soz.train_common17_oracle_event_oof_v1 import (  # noqa: E402
    induced_common17_neighbors,
)
from scripts.audit_common17_soz_raw_path_v1 import COMMON17  # noqa: E402
from src.soz.geometry import STANDARD_19  # noqa: E402


SCHEMA = "clinical_eeg_common17_patient_paired_comparison_v1"
DEFAULT_STRICT = ROOT / "outputs/clinical_eeg_common17_oracle_event_oof_r3r2_20260824/oof_predictions_and_states.safetensors"
DEFAULT_LITERAL = ROOT / "outputs/clinical_eeg_common17_car17_literal_midline_oof_sensitivity_v1_20260824/oof_predictions_and_states.safetensors"
DEFAULT_OFFICIAL = ROOT / "outputs/deepsoz_official_local_oof_full.json"
DEFAULT_ROSTER = ROOT / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815/manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/clinical_eeg_common17_paired_comparison_v1_20260824/receipt.json"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _load(path: Path, keys: Iterable[str]) -> dict[str, torch.Tensor]:
    requested = tuple(keys)
    with safe_open(str(path.resolve(strict=True)), framework="pt", device="cpu") as source:
        missing = sorted(set(requested) - set(source.keys()))
        if missing:
            raise KeyError(f"Missing tensors in {path}: {missing}")
        return {key: source.get_tensor(key) for key in requested}


def _acceptable(
    positive: torch.Tensor,
    graph: Sequence[Sequence[int]],
    enabled: bool,
) -> torch.Tensor:
    value = positive.clone()
    if enabled:
        for index in torch.nonzero(positive, as_tuple=False).flatten().tolist():
            if graph[index]:
                value[list(graph[index])] = True
    return value


def _outcomes(
    probability: torch.Tensor,
    targets: torch.Tensor,
    pre_count: torch.Tensor,
    graph: Sequence[Sequence[int]],
) -> dict[str, torch.Tensor]:
    top = torch.argsort(probability, dim=1, descending=True, stable=True)[:, 0]
    top_value = probability.gather(1, top[:, None]).squeeze(1)
    if bool(((probability == top_value[:, None]).sum(dim=1) != 1).any()):
        raise RuntimeError("Paired audit requires unique Top-1 predictions")
    exact = targets[torch.arange(len(targets)), top].bool()
    result = {"exact_top1": exact}
    for name, gate in (("deepsoz_N2", 2), ("deepsoz_N4", 4)):
        result[name] = torch.stack(
            [
                _acceptable(targets[row] == 1, graph, int(pre_count[row]) <= gate)[top[row]]
                for row in range(len(targets))
            ]
        ).bool()
    return result


def _mcnemar_exact(strict_only: int, comparator_only: int) -> float:
    discordant = strict_only + comparator_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(strict_only, comparator_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _bootstrap_delta_ci(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    seed: int,
    replicates: int = 100_000,
) -> list[float]:
    delta = (first.to(torch.int8) - second.to(torch.int8)).numpy().astype(np.float64)
    generator = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    batch = 10_000
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        indices = generator.integers(0, len(delta), size=(stop - start, len(delta)))
        estimates[start:stop] = delta[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return [float(lower), float(upper)]


def _paired(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    seed: int,
) -> dict[str, Any]:
    both = int((first & second).sum())
    first_only = int((first & ~second).sum())
    second_only = int((~first & second).sum())
    neither = int((~first & ~second).sum())
    total = len(first)
    return {
        "two_by_two": {
            "both_correct": both,
            "strict_only_correct": first_only,
            "official_only_correct": second_only,
            "both_incorrect": neither,
            "total": total,
        },
        "strict_successes": int(first.sum()),
        "official_successes": int(second.sum()),
        "paired_rate_difference_strict_minus_official": (int(first.sum()) - int(second.sum())) / total,
        "patient_bootstrap_percentile_95_ci": _bootstrap_delta_ci(first, second, seed=seed),
        "patient_bootstrap_replicates": 100_000,
        "patient_bootstrap_seed": seed,
        "mcnemar_exact_two_sided_p": _mcnemar_exact(first_only, second_only),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    strict = _load(
        args.strict,
        (
            "targets",
            "target_mask",
            "pre_mapping_positive_count",
            "common17_standard19_indices",
            "oof_probability.strict_car17_labram",
        ),
    )
    targets = strict["targets"].float()
    mask = strict["target_mask"].bool()
    pre_count = strict["pre_mapping_positive_count"].long()
    indices = strict["common17_standard19_indices"].long()
    if tuple(targets.shape) != (102, 17) or not bool(mask.all()):
        raise RuntimeError("Strict common17 target carrier is incomplete")
    expected = torch.tensor([STANDARD_19.index(channel) for channel in COMMON17])
    if not torch.equal(indices, expected):
        raise RuntimeError("Strict common17 channel order drifted")
    graph = induced_common17_neighbors(indices)

    official = _json(args.official)
    roster = _json(args.roster)
    patient_ids = [str(value) for value in roster["patient_ids"]]
    predictions = official["held_out_ensemble_predictions"]
    if tuple(official["preprocessing"]["channels"]) != tuple(STANDARD_19):
        raise RuntimeError("Official prediction channel space is ambiguous")
    if [str(row["patient_id"]) for row in predictions] != patient_ids:
        raise RuntimeError("Official/current patient order differs")
    official19 = torch.tensor([row["score"] for row in predictions], dtype=torch.float64)
    official17 = official19.index_select(1, indices)
    strict_outcomes = _outcomes(
        strict["oof_probability.strict_car17_labram"].float(), targets, pre_count, graph
    )
    official_outcomes = _outcomes(official17, targets, pre_count, graph)
    paired = {
        endpoint: _paired(
            strict_outcomes[endpoint],
            official_outcomes[endpoint],
            seed=20260824 + ordinal,
        )
        for ordinal, endpoint in enumerate(("exact_top1", "deepsoz_N2", "deepsoz_N4"))
    }

    literal = _load(
        args.literal,
        ("targets", "target_mask", "pre_mapping_positive_count", "patient_folds", "oof_probability"),
    )
    literal_outcomes = _outcomes(
        literal["oof_probability"].float(),
        literal["targets"].float(),
        literal["pre_mapping_positive_count"].long(),
        graph,
    )
    official_literal_outcomes = _outcomes(
        official17,
        literal["targets"].float(),
        literal["pre_mapping_positive_count"].long(),
        graph,
    )
    literal_counts = {
        endpoint: {
            "successes": int(values.sum()),
            "total": len(values),
            "rate": float(values.float().mean()),
        }
        for endpoint, values in literal_outcomes.items()
    }
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "pass_patient_paired_read_only_audit",
        "patient_count": 102,
        "patient_order_exactly_aligned": True,
        "strict_vs_official_same_common17_GT_graph_and_gates": True,
        "strict_vs_official": paired,
        "multiplicity_note": "three correlated descriptive endpoints; p-values are unadjusted and do not establish superiority",
        "inferential_conclusion": {
            "exact_top1_superiority_established_at_unadjusted_0_05": paired["exact_top1"]["mcnemar_exact_two_sided_p"] < 0.05,
            "deepsoz_N2_superiority_established_at_unadjusted_0_05": paired["deepsoz_N2"]["mcnemar_exact_two_sided_p"] < 0.05,
            "deepsoz_N4_superiority_established_at_unadjusted_0_05": paired["deepsoz_N4"]["mcnemar_exact_two_sided_p"] < 0.05,
            "point_estimate_only_superiority_claim_allowed": False,
        },
        "literal_raw_duplicate_PZ_OR_sensitivity": {
            "metrics": literal_counts,
            "target_endpoint_identical_to_verified_primary": False,
            "reason": "two discordant duplicate-PZ patients are literal positives only in this sensitivity arm",
            "formal_paired_significance_against_verified_primary_not_interpretable_as_same_GT_endpoint": True,
            "literal_strict_vs_official_on_same_literal_GT": {
                endpoint: _paired(
                    literal_outcomes[endpoint],
                    official_literal_outcomes[endpoint],
                    seed=20260924 + ordinal,
                )
                for ordinal, endpoint in enumerate(("exact_top1", "deepsoz_N2", "deepsoz_N4"))
            },
        },
        "lineage": {
            "strict_tensor": {"path": str(args.strict), "sha256": _sha256(args.strict)},
            "literal_tensor": {"path": str(args.literal), "sha256": _sha256(args.literal)},
            "official_oof": {"path": str(args.official), "sha256": _sha256(args.official)},
            "patient_roster": {"path": str(args.roster), "sha256": _sha256(args.roster)},
            "script": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        },
        "access_receipt": {
            "private_data_loaded": False,
            "training_performed": False,
            "frozen_artifacts_modified": False,
        },
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--literal", type=Path, default=DEFAULT_LITERAL)
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = run(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output), "receipt_sha256": receipt["receipt_sha256"]}))


if __name__ == "__main__":
    main()
