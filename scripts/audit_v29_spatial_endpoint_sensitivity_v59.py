#!/usr/bin/env python3
"""Audit frozen v29 strict/N2/N4 and graph-distance endpoint sensitivity.

The same frozen public OOF and private post-open predictions are evaluated
without retraining, thresholding, routing, or model selection.  Strict exact
positive-set Top-1 remains primary.  DeepSOZ-style N2/N4 are reported beside
an unconditional one-hop description and the minimum undirected graph-hop
distance to any reference-positive electrode.  Known private spread channels
never enter an official acceptable set.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict, deque
import csv
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

from src.soz.geometry import CHANNEL_INDEX, STANDARD_19  # noqa: E402
from src.soz.metrics import DEEPSOZ_STANDARD19_NEIGHBORS  # noqa: E402


SCHEMA = "trustworthy_soz_v29_spatial_endpoint_sensitivity_v59"
DEFAULT_PUBLIC = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_PUBLIC_V16 = ROOT / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815"
DEFAULT_PRIVATE = ROOT / "outputs/trustworthy_soz_private_frozen_publication_v36_20260816"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_v29_spatial_endpoint_sensitivity_v59_20260816"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260859
CANDIDATES = tuple(channel for channel in STANDARD_19 if channel != "PZ")


def _accepted(
    positives: set[int],
    spread: set[int],
    evaluable: set[int],
    *,
    max_positive: int,
) -> set[int]:
    result = set(positives)
    if len(positives) <= max_positive:
        for index in positives:
            result.update(DEEPSOZ_STANDARD19_NEIGHBORS[index])
    result.intersection_update(evaluable)
    result.difference_update(spread - positives)
    return result


def _undirected_adjacency() -> tuple[frozenset[int], ...]:
    rows = [set() for _ in STANDARD_19]
    for left, neighbors in enumerate(DEEPSOZ_STANDARD19_NEIGHBORS):
        for right in neighbors:
            rows[left].add(int(right))
            rows[int(right)].add(left)
    return tuple(frozenset(row) for row in rows)


UNDIRECTED_ADJACENCY = _undirected_adjacency()


def _minimum_graph_hops(top: int, positives: set[int]) -> int:
    if top in positives:
        return 0
    queue: deque[tuple[int, int]] = deque([(top, 0)])
    visited = {top}
    while queue:
        node, distance = queue.popleft()
        for neighbor in UNDIRECTED_ADJACENCY[node]:
            if neighbor in positives:
                return distance + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    raise RuntimeError("standard-19 neighbor graph is disconnected")


def _endpoint_row(
    *,
    unit_id: str,
    patient_id: str,
    top1: str,
    positive_channels: Sequence[str],
    spread_channels: Sequence[str],
    first_positive_rank: int,
) -> dict[str, object]:
    if top1 not in CHANNEL_INDEX:
        raise ValueError("top1 is outside standard-19")
    positives = {CHANNEL_INDEX[str(value)] for value in positive_channels}
    spread = {CHANNEL_INDEX[str(value)] for value in spread_channels}
    evaluable = {CHANNEL_INDEX[value] for value in CANDIDATES}
    if not positives or not positives <= evaluable or positives & spread:
        raise ValueError("positive/spread/evaluable sets are invalid")
    top = CHANNEL_INDEX[top1]
    if top not in evaluable:
        raise ValueError("top1 is outside frozen C18")
    n2_set = _accepted(positives, spread, evaluable, max_positive=2)
    n4_set = _accepted(positives, spread, evaluable, max_positive=4)
    unconditional = set(positives)
    for index in positives:
        unconditional.update(DEEPSOZ_STANDARD19_NEIGHBORS[index])
    unconditional.intersection_update(evaluable)
    distance = _minimum_graph_hops(top, positives)
    strict = float(top in positives)
    n2 = float(top in n2_set)
    n4 = float(top in n4_set)
    return {
        "unit_id": str(unit_id),
        "patient_id": str(patient_id),
        "top1": top1,
        "positive_channels": list(positive_channels),
        "known_spread_channels": list(spread_channels),
        "positive_set_size": len(positives),
        "strict": strict,
        "official_N2": n2,
        "official_N4": n4,
        "N2_neighbor_only": float(n2 == 1.0 and strict == 0.0),
        "N4_neighbor_only": float(n4 == 1.0 and strict == 0.0),
        "N4_far": 1.0 - n4,
        "unconditional_directed_one_hop": float(top in unconditional),
        "minimum_undirected_graph_hops": distance,
        "undirected_distance_le_1": float(distance <= 1),
        "known_spread_top1": float(top in spread),
        "first_positive_rank": int(first_positive_rank),
    }


def _public_rows(
    public_directory: Path, public_v16_directory: Path
) -> list[dict[str, object]]:
    v16 = json.loads(
        (public_v16_directory / "manifest.json").resolve(strict=True).read_text(encoding="utf-8")
    )
    patient_ids = [str(value) for value in v16.get("patient_ids", ())]
    if len(patient_ids) != 102:
        raise ValueError("public frozen patient roster changed")
    payload = load_file(
        str((public_directory / "oof_predictions.safetensors").resolve(strict=True)),
        device="cpu",
    )
    probability = payload["oof.portable_equal_ensemble_probability"].float()
    targets = payload["targets"].float()
    mask = payload["target_mask"].bool()
    if tuple(probability.shape) != (102, 19):
        raise ValueError("public v29 probability shape changed")
    rows: list[dict[str, object]] = []
    for index, patient_id in enumerate(patient_ids):
        evaluable = torch.nonzero(mask[index], as_tuple=False).flatten()
        order = evaluable[
            torch.argsort(probability[index, evaluable], descending=True, stable=True)
        ]
        positives = set(
            torch.nonzero((targets[index] == 1) & mask[index], as_tuple=False)
            .flatten()
            .tolist()
        )
        first_rank = min(position + 1 for position, value in enumerate(order.tolist()) if value in positives)
        rows.append(
            _endpoint_row(
                unit_id=f"PUBLIC-{index:03d}",
                patient_id=patient_id,
                top1=STANDARD_19[int(order[0])],
                positive_channels=[STANDARD_19[value] for value in sorted(positives)],
                spread_channels=(),
                first_positive_rank=first_rank,
            )
        )
    return rows


def _private_rows(private_directory: Path) -> list[dict[str, object]]:
    path = (private_directory / "private_event_error_audit.csv").resolve(strict=True)
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            positives = tuple(str(value) for value in ast.literal_eval(raw["positive_channels"]))
            spread = tuple(str(value) for value in ast.literal_eval(raw["known_spread_channels"]))
            row = _endpoint_row(
                unit_id=str(raw["unit_id"]),
                patient_id=str(raw["patient_id"]),
                top1=str(raw["top1"]),
                positive_channels=positives,
                spread_channels=spread,
                first_positive_rank=int(raw["first_positive_rank"]),
            )
            if row["strict"] != float(raw["strict"]) or row["official_N4"] != (
                float(raw["strict"]) + float(raw["neighbor_only"])
            ):
                raise ValueError("private v59 endpoint does not reproduce v36")
            rows.append(row)
    if len(rows) != 51 or len({row["patient_id"] for row in rows}) != 23:
        raise ValueError("private v36 primary roster changed")
    return rows


METRICS = (
    "strict",
    "official_N2",
    "official_N4",
    "N2_neighbor_only",
    "N4_neighbor_only",
    "N4_far",
    "unconditional_directed_one_hop",
    "undirected_distance_le_1",
    "known_spread_top1",
)


def _patient_equal(rows: Sequence[Mapping[str, object]], key: str) -> float:
    bags: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        bags[str(row["patient_id"])].append(float(row[key]))
    return float(np.mean([np.mean(values) for values in bags.values()]))


def _cluster_bootstrap(
    rows: Sequence[Mapping[str, object]], *, key: str, seed: int
) -> list[float]:
    bags: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        bags[str(row["patient_id"])].append(float(row[key]))
    values = np.asarray([np.mean(bags[name]) for name in sorted(bags)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    return [float(value) for value in np.quantile(values[sampled].mean(axis=1), (0.025, 0.975))]


def _summary(rows: Sequence[Mapping[str, object]], *, seed: int) -> dict[str, object]:
    result: dict[str, object] = {
        "units": len(rows),
        "patients": len({str(row["patient_id"]) for row in rows}),
        "N2_neighbor_eligible_units": sum(int(row["positive_set_size"]) <= 2 for row in rows),
        "N4_neighbor_eligible_units": sum(int(row["positive_set_size"]) <= 4 for row in rows),
    }
    for offset, key in enumerate(METRICS):
        result[key] = {
            "unit_micro": float(np.mean([float(row[key]) for row in rows])),
            "patient_equal": _patient_equal(rows, key),
            "patient_cluster_bootstrap_ci95": _cluster_bootstrap(
                rows, key=key, seed=seed + offset
            ),
        }
    distances = np.asarray(
        [int(row["minimum_undirected_graph_hops"]) for row in rows], dtype=np.int64
    )
    counts = Counter(
        str(value) if value <= 2 else ">=3" for value in distances.tolist()
    )
    result["minimum_undirected_graph_hops"] = {
        "mean_unit_micro": float(distances.mean()),
        "median_unit_micro": float(np.median(distances)),
        "counts": {key: counts.get(key, 0) for key in ("0", "1", "2", ">=3")},
    }
    ranks = np.asarray([int(row["first_positive_rank"]) for row in rows])
    result["candidate_burden"] = {
        "median_first_positive_rank": float(np.median(ranks)),
        "mean_first_positive_rank": float(ranks.mean()),
        "above_top5": int((ranks > 5).sum()),
    }
    return result


def _size_strata(rows: Sequence[Mapping[str, object]], *, seed: int) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        size = int(row["positive_set_size"])
        key = "1" if size == 1 else "2" if size == 2 else "3-4" if size <= 4 else ">=5"
        groups[key].append(row)
    return {
        key: _summary(values, seed=seed + 100 * index)
        for index, (key, values) in enumerate(sorted(groups.items()))
    }


def run(
    *, public_directory: Path, public_v16_directory: Path, private_directory: Path
) -> tuple[dict[str, object], list[dict[str, object]]]:
    public_rows = _public_rows(public_directory, public_v16_directory)
    private_rows = _private_rows(private_directory)
    public_summary = _summary(public_rows, seed=BOOTSTRAP_SEED)
    private_summary = _summary(private_rows, seed=BOOTSTRAP_SEED + 10_000)
    if public_summary["strict"]["unit_micro"] != 54 / 102 or public_summary[
        "official_N4"
    ]["unit_micro"] != 78 / 102:
        raise RuntimeError("public v59 did not reproduce formal v29 endpoints")
    if private_summary["strict"]["unit_micro"] != 25 / 51 or private_summary[
        "official_N4"
    ]["unit_micro"] != 38 / 51:
        raise RuntimeError("private v59 did not reproduce formal v29 endpoints")

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_frozen_v29_spatial_endpoint_sensitivity",
        "primary_endpoint": "strict_positive_set_physical_electrode_top1",
        "secondary_endpoints": {
            "official_N2": "exact, or directed one-hop when positive-set size <=2",
            "official_N4": "exact, or directed one-hop when positive-set size <=4",
            "unconditional_directed_one_hop": "distance description without cardinality eligibility",
            "minimum_undirected_graph_hops": "descriptive topology distance, not millimeters",
        },
        "public": {
            "role": "consumed_adaptive_patient_OOF_development",
            "summary": public_summary,
            "positive_set_size_strata": _size_strata(
                public_rows, seed=BOOTSTRAP_SEED + 20_000
            ),
        },
        "private": {
            "role": "post_open_zero_adaptation_transport",
            "summary": private_summary,
            "positive_set_size_strata": _size_strata(
                private_rows, seed=BOOTSTRAP_SEED + 30_000
            ),
        },
        "graph_contract": {
            "directed_official_neighbor_table": [list(row) for row in DEEPSOZ_STANDARD19_NEIGHBORS],
            "distance_graph": "symmetrized_union_of_official_directed_edges",
            "PZ_can_be_an_intermediate_distance_node": True,
            "PZ_is_output_candidate": False,
            "known_spread_removed_from_official_N2_N4": True,
        },
        "access_receipt": {
            "frozen_predictions_loaded": True,
            "public_target_loaded_for_read_only_evaluation": True,
            "opened_private_reference_loaded_for_read_only_evaluation": True,
            "training_calibration_routing_or_threshold_selection_performed": False,
            "private_used_to_change_endpoint_definition": False,
        },
        "interpretation_boundary": {
            "official_N2_or_N4_is_strict_accuracy": False,
            "graph_hops_are_physical_millimeters_or_cortical_distance": False,
            "one_hop_neighbor_is_biologically_equivalent_SOZ": False,
            "private_is_fresh_external_validation": False,
            "allowed_claim": (
                "the apparent v29 agreement changes by the explicitly reported "
                "spatial tolerance and positive-set cardinality convention"
            ),
        },
        "bootstrap": {
            "unit": "patient_cluster",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confirmatory_p_values": False,
        },
        "files": {"unit_table": "spatial_endpoint_rows.csv"},
    }
    rows = [
        {"cohort": "public", **row} for row in public_rows
    ] + [{"cohort": "private", **row} for row in private_rows]
    return result, rows


def publish(
    *, output: Path, result: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        fieldnames = list(rows[0])
        with (staging / "spatial_endpoint_rows.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False)
                        if isinstance(value, list)
                        else value
                        for key, value in row.items()
                    }
                )
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--public-v16", type=Path, default=DEFAULT_PUBLIC_V16)
    parser.add_argument("--private", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, rows = run(
        public_directory=args.public,
        public_v16_directory=args.public_v16,
        private_directory=args.private,
    )
    output = publish(output=args.output, result=result, rows=rows)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "public_N2": result["public"]["summary"]["official_N2"]["unit_micro"],
                "public_N4": result["public"]["summary"]["official_N4"]["unit_micro"],
                "private_N2": result["private"]["summary"]["official_N2"]["unit_micro"],
                "private_N4": result["private"]["summary"]["official_N4"]["unit_micro"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
