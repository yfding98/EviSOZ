#!/usr/bin/env python3
"""Re-score frozen local official-DeepSOZ OOF predictions on common17."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

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


SCHEMA = "deepsoz_official_local_oof_common17_comparator_v1"
DEFAULT_OFFICIAL = ROOT / "outputs/deepsoz_official_local_oof_full.json"
DEFAULT_OOF = ROOT / "outputs/clinical_eeg_common17_oracle_event_oof_r3r2_20260824/oof_predictions_and_states.safetensors"
DEFAULT_ROSTER = ROOT / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815/manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/deepsoz_official_common17_comparator_v1_20260824/receipt.json"


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
            raise KeyError(f"Missing tensors: {missing}")
        return {key: source.get_tensor(key) for key in requested}


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {
        "successes": successes,
        "total": total,
        "rate": p,
        "wilson_95_ci": [center - half, center + half],
    }


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    official = _json(args.official)
    roster = _json(args.roster)
    patient_ids = [str(value) for value in roster["patient_ids"]]
    official_channels = tuple(str(value) for value in official["preprocessing"]["channels"])
    if official_channels != tuple(STANDARD_19):
        raise RuntimeError("Official score channel space cannot be aligned unambiguously")
    predictions = official.get("held_out_ensemble_predictions")
    if not isinstance(predictions, list) or len(predictions) != 102:
        raise RuntimeError("Frozen official held-out ensemble prediction roster is incomplete")
    prediction_ids = [str(row["patient_id"]) for row in predictions]
    if prediction_ids != patient_ids:
        raise RuntimeError("Official/current patient order differs; fail closed")
    scores19 = torch.tensor([row["score"] for row in predictions], dtype=torch.float64)
    if tuple(scores19.shape) != (102, 19) or not torch.isfinite(scores19).all():
        raise RuntimeError("Official score carrier is not finite [102,19]")

    payload = _load(
        args.oof,
        (
            "targets",
            "target_mask",
            "pre_mapping_positive_count",
            "common17_standard19_indices",
        ),
    )
    indices = payload["common17_standard19_indices"].long()
    expected = torch.tensor([STANDARD_19.index(channel) for channel in COMMON17])
    if not torch.equal(indices, expected):
        raise RuntimeError("Current common17 channel index carrier is ambiguous")
    targets = payload["targets"].float()
    mask = payload["target_mask"].bool()
    pre_count = payload["pre_mapping_positive_count"].long()
    if tuple(targets.shape) != (102, 17) or not bool(mask.all()):
        raise RuntimeError("Mapped common17 GT is not complete [102,17]")
    scores = scores19.index_select(1, indices)
    if not torch.equal(scores, scores19[:, indices]):
        raise RuntimeError("Prediction transform did more than delete FZ/PZ")
    graph = induced_common17_neighbors(indices)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    top = order[:, 0]
    top_value = scores.gather(1, top[:, None]).squeeze(1)
    if bool(((scores == top_value[:, None]).sum(dim=1) != 1).any()):
        raise RuntimeError("Official common17 scores contain Top-1 ties")
    ranked = targets.gather(1, order)
    exact = ranked[:, 0].bool()
    hit3 = (ranked[:, :3].sum(dim=1) > 0)
    hit5 = (ranked[:, :5].sum(dim=1) > 0)
    first = ranked.argmax(dim=1)
    n2 = torch.stack(
        [
            _acceptable(targets[row] == 1, graph, int(pre_count[row]) <= 2)[top[row]]
            for row in range(102)
        ]
    ).bool()
    n4 = torch.stack(
        [
            _acceptable(targets[row] == 1, graph, int(pre_count[row]) <= 4)[top[row]]
            for row in range(102)
        ]
    ).bool()
    rank_histogram = Counter(int(value) + 1 for value in first.tolist())
    metrics = {
        "exact_top1": _wilson(int(exact.sum()), 102),
        "accuracy": _wilson(int(exact.sum()), 102),
        "hit_at_3": _wilson(int(hit3.sum()), 102),
        "hit_at_5": _wilson(int(hit5.sum()), 102),
        "mrr": float((1.0 / (first.double() + 1.0)).mean()),
        "first_positive_rank_histogram": {
            str(rank): count for rank, count in sorted(rank_histogram.items())
        },
        "deepsoz_N2": {
            **_wilson(int(n2.sum()), 102),
            "eligible_for_neighbor_expansion": int((pre_count <= 2).sum()),
        },
        "deepsoz_N4": {
            **_wilson(int(n4.sum()), 102),
            "eligible_for_neighbor_expansion": int((pre_count <= 4).sum()),
        },
    }
    frozen = official["held_out_ensemble_metrics"]
    frozen_identity = {
        "exact": int(exact.sum()) == int(frozen["exact_n"]),
        "N2": int(n2.sum()) == int(frozen["neighborhood2_n"]),
        "N4": int(n4.sum()) == int(frozen["neighborhood4_n"]),
    }
    if not all(frozen_identity.values()):
        raise RuntimeError("Common17 comparator disagrees with frozen official success counts")
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "pass_strictly_aligned_read_only_comparator",
        "patient_order_exactly_equal": True,
        "patient_count": 102,
        "official_score_channel_order_exactly_STANDARD_19": True,
        "common17_channels": list(COMMON17),
        "prediction_transform": {
            "operation": "delete FZ and PZ score axes by unambiguous index selection",
            "FZ_or_PZ_score_mapped_into_CZ": False,
            "renormalization_applied": False,
        },
        "ground_truth_transform": {
            "operation": "use frozen current GT with observed FZ/PZ positives OR-mapped to CZ then delete FZ/PZ",
            "mapping_before_positive_count_used_for_N2_N4_gate": True,
        },
        "neighbor_graph": "DeepSOZ standard19 one-hop table induced on common17",
        "metrics": metrics,
        "frozen_official_exact_N2_N4_success_counts_identical": frozen_identity,
        "claim_boundary": {
            "source_is_local_official_DeepSOZ_weight_transfer_OOF": True,
            "exact_original_paper_data_reproduction": False,
            "N2_or_N4_is_exact_electrode_accuracy": False,
            "oracle_event_localization_not_end_to_end_detection": True,
        },
        "lineage": {
            "official_oof_json": {"path": str(args.official), "sha256": _sha256(args.official)},
            "current_common17_target_tensor": {"path": str(args.oof), "sha256": _sha256(args.oof)},
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
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
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
