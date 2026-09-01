#!/usr/bin/env python3
"""Select the frozen morphology candidate rule from source-train OOF scores.

This command reuses the five patient/group-held-out C-CAR19 morphology heads
from the formal preprocessing comparison.  Their architecture, optimizer,
epochs, source groups and native CE6 targets match the final M0 head.  No
official TUEV evaluation, TUSZ, DeepSOZ SOZ, or private label is read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import tempfile


_REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _REQUIRED_CUBLAS_WORKSPACE_CONFIG)

import numpy as np
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.soz.data.tuev_morphology import load_tuev_morphology_manifest  # noqa: E402
from src.soz.geometry import MORPHOLOGY_CLASSES  # noqa: E402
from src.soz.models.concept_heads import MorphologyEvidenceHead  # noqa: E402


SEED = 20260808
BOOTSTRAP_REPLICATES = 2000
LOCAL_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
CONFLICT_THRESHOLDS = tuple(round(0.05 + 0.05 * index, 2) for index in range(10))
LOCAL_LABELS = frozenset({"SPSW", "PLED"})
CONFLICT_SCORE_LABELS = ("GPED", "EYEM", "ARTF")
NONLOCAL_LABELS = frozenset({"GPED", "EYEM", "ARTF", "BCKG"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--parity-directory",
        type=Path,
        default=Path("outputs/preprocessing_parity_formal_v1_20260809"),
    )
    parser.add_argument(
        "--holding-manifest",
        type=Path,
        default=Path("outputs/tuev_morphology_holding_manifest_v3_20260810"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/tuev_morphology_oof_thresholds_v1_20260810"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _load_oof_probabilities(
    parity: Path,
    items: list[dict[str, object]],
    tokens: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if tuple(tokens.shape) != (len(items), 19, 1, 200):
        raise ValueError("C-CAR19 morphology token geometry is not [N,19,1,200]")
    group_fold: dict[str, int] = {}
    indices_by_fold: dict[int, list[int]] = {fold: [] for fold in range(5)}
    for item in items:
        index = int(item["index"])
        if index < 0 or index >= len(items):
            raise ValueError("TUEV item index is out of range")
        group = str(item["parent_group_id"])
        fold = int(item["fold"])
        if fold not in indices_by_fold:
            raise ValueError("TUEV OOF fold must be 0--4")
        previous = group_fold.setdefault(group, fold)
        if previous != fold:
            raise ValueError("One morphology group crosses OOF folds")
        indices_by_fold[fold].append(index)
    if tuple(sorted(int(item["index"]) for item in items)) != tuple(range(len(items))):
        raise ValueError("TUEV item indices are not a complete canonical roster")

    probabilities = np.empty((len(items), 20, 6), dtype=np.float32)
    all_groups = set(group_fold)
    held_union: set[str] = set()
    for fold in range(5):
        checkpoint = parity / "nested-checkpoints" / "tuev" / "C-CAR19" / f"fold-{fold}"
        receipt = _load_json(checkpoint / "receipt.json")
        if (
            receipt.get("dataset") != "TUEV"
            or receipt.get("arm_id") != "C-CAR19"
            or int(receipt.get("fold", -1)) != fold
        ):
            raise ValueError("Morphology OOF checkpoint identity changed")
        config = receipt.get("config")
        if not isinstance(config, dict) or (
            config.get("epochs") != 20
            or config.get("loss") != "group_equal_overlap_component_weighted_CE6"
            or config.get("checkpoint_selection") != "fixed_final_epoch"
        ):
            raise ValueError("Morphology OOF checkpoint uses another training protocol")
        held = tuple(str(value) for value in receipt.get("held_ids", ()))
        fit = tuple(str(value) for value in receipt.get("fit_ids", ()))
        expected_held = tuple(sorted(group for group, value in group_fold.items() if value == fold))
        if held != expected_held or set(fit) & set(held) or set(fit) | set(held) != all_groups:
            raise ValueError("Morphology OOF checkpoint violates its group firewall")
        if held_union & set(held):
            raise ValueError("A morphology group is held out more than once")
        held_union.update(held)

        head = MorphologyEvidenceHead().to(device)
        state = load_file(str(checkpoint / "model.safetensors"), device="cpu")
        head.load_state_dict(state, strict=True)
        head.eval()
        indices = indices_by_fold[fold]
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                selection = indices[start : start + batch_size]
                batch = torch.from_numpy(np.array(tokens[selection], copy=True)).to(device)
                output = head(batch).squeeze(2).softmax(dim=-1).cpu().numpy()
                probabilities[selection] = output
        del head
    if held_union != all_groups or not np.isfinite(probabilities).all():
        raise RuntimeError("Morphology OOF prediction roster is incomplete")
    return probabilities


def _component_rows(
    *,
    items: list[dict[str, object]],
    labels: np.ndarray,
    masks: np.ndarray,
    weights: np.ndarray,
    probabilities: np.ndarray,
    holding_manifest,
) -> list[dict[str, object]]:
    expected_shape = (len(items), 20)
    if labels.shape != expected_shape or masks.shape != expected_shape or weights.shape != expected_shape:
        raise ValueError("TUEV native target arrays must be [N,20]")
    by_crop = {group.crop_id: group for group in holding_manifest.interval_groups}
    conflict_indices = tuple(MORPHOLOGY_CLASSES.index(name) for name in CONFLICT_SCORE_LABELS)
    rows: list[dict[str, object]] = []
    observed_cells = 0
    for item in items:
        index = int(item["index"])
        crop_id = str(item["crop_id"])
        group_id = str(item["parent_group_id"])
        fold = int(item["fold"])
        interval = by_crop.get(crop_id)
        if interval is None or interval.parent_group_id != group_id:
            raise ValueError("Parity crop was swapped across the holding manifest")
        targets_by_edge = {target.edge_index: target for target in interval.targets}
        for edge_index in np.flatnonzero(masks[index]):
            observed_cells += 1
            target = targets_by_edge.get(int(edge_index))
            if target is None:
                raise ValueError("Parity mask exposes a cell absent from native TUEV targets")
            label_index = int(labels[index, edge_index])
            if label_index != target.label_index:
                raise ValueError("Parity label differs from the native TUEV target")
            expected_weight = target.component_weight
            if not math.isclose(
                float(weights[index, edge_index]), expected_weight, rel_tol=0.0, abs_tol=1e-7
            ):
                raise ValueError("Parity overlap-component weight changed")
            probability = probabilities[index, edge_index]
            rows.append(
                {
                    "overlap_component_id": target.overlap_component_id,
                    "group_id": group_id,
                    "fold": fold,
                    "crop_id": crop_id,
                    "edge_index": int(edge_index),
                    "label": target.label_name,
                    "local_score": float(
                        probability[MORPHOLOGY_CLASSES.index("SPSW")]
                        + probability[MORPHOLOGY_CLASSES.index("PLED")]
                    ),
                    "conflict_score": float(probability[list(conflict_indices)].sum()),
                }
            )
    if observed_cells != int(masks.sum()) or not rows:
        raise RuntimeError("Morphology OOF target-cell extraction is incomplete")
    return rows


def _collapse_components(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    components: dict[str, dict[str, object]] = {}
    for row in rows:
        component_id = str(row["overlap_component_id"])
        value = components.setdefault(
            component_id,
            {
                "overlap_component_id": component_id,
                "group_id": str(row["group_id"]),
                "fold": int(row["fold"]),
                "label": str(row["label"]),
                "local_scores": [],
                "conflict_scores": [],
            },
        )
        identity = (value["group_id"], value["fold"], value["label"])
        observed = (str(row["group_id"]), int(row["fold"]), str(row["label"]))
        if identity != observed:
            raise ValueError("One overlap component crosses group/fold/label identity")
        value["local_scores"].append(float(row["local_score"]))
        value["conflict_scores"].append(float(row["conflict_score"]))
    return [components[key] for key in sorted(components)]


def _select_thresholds(components: list[dict[str, object]]) -> dict[str, object]:
    groups = tuple(sorted({str(value["group_id"]) for value in components}))
    group_index = {group: index for index, group in enumerate(groups)}
    bootstrap_rng = np.random.default_rng(SEED)
    bootstrap_groups = bootstrap_rng.integers(
        0, len(groups), size=(BOOTSTRAP_REPLICATES, len(groups)), dtype=np.int32
    )
    grid: list[dict[str, object]] = []
    for tau_local in LOCAL_THRESHOLDS:
        for tau_conflict in CONFLICT_THRESHOLDS:
            group_counts = np.zeros((len(groups), 4), dtype=np.float64)
            for component in components:
                local_scores = np.asarray(component["local_scores"], dtype=np.float64)
                conflict_scores = np.asarray(component["conflict_scores"], dtype=np.float64)
                predicted = bool(
                    np.any(
                        (local_scores >= tau_local)
                        & (conflict_scores <= tau_conflict)
                    )
                )
                label = str(component["label"])
                local_truth = label in LOCAL_LABELS
                nonlocal_truth = label in NONLOCAL_LABELS
                if not local_truth and not nonlocal_truth:
                    raise ValueError("Unexpected CE6 component label")
                row = group_counts[group_index[str(component["group_id"])]]
                row[0] += float(predicted and local_truth)  # true localizing
                row[1] += float(predicted)  # predicted localizing
                row[2] += float(predicted and nonlocal_truth)  # false localizing
                row[3] += float(nonlocal_truth)  # nonlocal denominator
            totals = group_counts.sum(axis=0)
            precision = float(totals[0] / totals[1]) if totals[1] else 0.0
            conflict_rate = float(totals[2] / totals[3]) if totals[3] else 1.0
            sampled = group_counts[bootstrap_groups].sum(axis=1)
            bootstrap_precision = np.divide(
                sampled[:, 0],
                sampled[:, 1],
                out=np.zeros(BOOTSTRAP_REPLICATES, dtype=np.float64),
                where=sampled[:, 1] > 0,
            )
            bootstrap_conflict = np.divide(
                sampled[:, 2],
                sampled[:, 3],
                out=np.ones(BOOTSTRAP_REPLICATES, dtype=np.float64),
                where=sampled[:, 3] > 0,
            )
            precision_lower95 = float(np.quantile(bootstrap_precision, 0.025))
            conflict_upper95 = float(np.quantile(bootstrap_conflict, 0.975))
            predicted_components = int(totals[1])
            predicted_groups = int((group_counts[:, 1] > 0).sum())
            qualifies = bool(
                precision_lower95 >= 0.80
                and predicted_components >= 100
                and predicted_groups >= 20
                and conflict_upper95 <= 0.10
            )
            grid.append(
                {
                    "tau_local": tau_local,
                    "tau_conflict": tau_conflict,
                    "precision": precision,
                    "precision_lower95": precision_lower95,
                    "conflict_false_localizing_rate": conflict_rate,
                    "conflict_false_localizing_upper95": conflict_upper95,
                    "predicted_localizing_component_count": predicted_components,
                    "predicted_localizing_group_count": predicted_groups,
                    "qualifies": qualifies,
                }
            )
    eligible = [value for value in grid if bool(value["qualifies"])]
    selected = (
        max(
            eligible,
            key=lambda value: (
                int(value["predicted_localizing_component_count"]),
                float(value["tau_local"]),
                -float(value["tau_conflict"]),
            ),
        )
        if eligible
        else None
    )
    return {
        "schema_version": "soz_tuev_morphology_oof_threshold_selection_v1",
        "selection_status": "GO" if selected is not None else "NO_GO",
        "selected_thresholds": selected,
        "source_train_group_count": len(groups),
        "overlap_component_count": len(components),
        "bootstrap_seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "threshold_grid": grid,
        "gate": {
            "precision_lower95_minimum": 0.80,
            "predicted_localizing_component_count_minimum": 100,
            "predicted_localizing_group_count_minimum": 20,
            "conflict_false_localizing_upper95_maximum": 0.10,
        },
        "data_boundary": {
            "uses_source_train_tuev_only": True,
            "uses_official_tuev_eval": False,
            "uses_tusz": False,
            "uses_deepsoz_soz_labels": False,
            "uses_private": False,
        },
    }


def main() -> int:
    args = _parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    output = args.output_directory.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"OOF threshold output already exists: {output}")
    parity = args.parity_directory.absolute()
    plan = _load_json(parity / "run-plan.json")
    raw_items = plan.get("tuev_items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Parity run plan lacks TUEV items")
    items = [dict(value) for value in raw_items]
    arrays = parity / "arrays"
    tokens = np.load(arrays / "tuev_tokens_C-CAR19.npy", mmap_mode="r")
    labels = np.load(arrays / "tuev_labels.npy", mmap_mode="r")
    masks = np.load(arrays / "tuev_mask.npy", mmap_mode="r")
    weights = np.load(arrays / "tuev_weights.npy", mmap_mode="r")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    probabilities = _load_oof_probabilities(
        parity,
        items,
        tokens,
        device=device,
        batch_size=args.batch_size,
    )
    holding = load_tuev_morphology_manifest(args.holding_manifest)
    rows = _component_rows(
        items=items,
        labels=labels,
        masks=masks,
        weights=weights,
        probabilities=probabilities,
        holding_manifest=holding,
    )
    components = _collapse_components(rows)
    result = _select_thresholds(components)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        with (staging / "oof_occurrence_predictions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            fields = (
                "overlap_component_id",
                "group_id",
                "fold",
                "crop_id",
                "edge_index",
                "label",
                "local_score",
                "conflict_score",
            )
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        (staging / "threshold_selection.json").write_text(
            json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({key: result[key] for key in (
        "selection_status",
        "selected_thresholds",
        "source_train_group_count",
        "overlap_component_count",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
