#!/usr/bin/env python3
"""Evaluate frozen event phenotypes against native TUSZ edge-time labels.

This is a source-task semantic check only.  Native channel annotations encode
scalp-visible ictal involvement, not SOZ or propagation.  Unknown/masked cells
are never treated as negatives, and no threshold is selected by this command.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.soz.data.tusz import (  # noqa: E402
    TUSZ_WINDOW_START_SEC,
    load_tusz_ictal_involvement_target,
)
from src.soz.geometry import TCP_20_EDGES  # noqa: E402


DEFAULT_PHENOTYPE = ROOT / "outputs/event_phenotype_source_only_n64_20260811.json"
DEFAULT_SOURCE_RECEIPT = (
    ROOT
    / "outputs/tusz_ictal_master_manifest_v4_20260809_preflight/receipt.json"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT = (
    ROOT / "outputs/event_phenotype_tusz_native_alignment_n64_20260811.json"
)
OUTPUT_SCHEMA = "soz_event_phenotype_tusz_native_alignment_v1"
EDGE_NAMES = tuple(f"{left}-{right}" for left, right in TCP_20_EDGES)
EDGE_INDEX = {name: index for index, name in enumerate(EDGE_NAMES)}


def _object(path: Path) -> dict[str, object]:
    source = path.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError("Input must be a canonical regular JSON file")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Input JSON must contain one object")
    return value


def _safe_edf(root: Path, relative_value: object) -> Path:
    relative = PurePosixPath(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".edf":
        raise ValueError("Unsafe relative EDF path")
    source = root.joinpath(*relative.parts)
    for component in (source, *source.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("EDF path cannot traverse symlinks")
    resolved = source.resolve(strict=True)
    if resolved.relative_to(root).as_posix() != relative.as_posix():
        raise ValueError("EDF path escaped the pinned TUSZ root")
    return resolved


def _fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "minimum": float(tensor.min().item()),
        "median": float(tensor.median().item()),
        "mean": float(tensor.mean().item()),
        "maximum": float(tensor.max().item()),
    }


def evaluate(
    *,
    phenotype_path: Path,
    source_receipt_path: Path,
    tusz_root: Path,
    output: Path,
) -> dict[str, object]:
    phenotype_artifact = _object(phenotype_path)
    if phenotype_artifact.get("schema_version") != (
        "soz_event_phenotype_source_only_audit_v1"
    ):
        raise ValueError("Unexpected event-phenotype audit schema")
    access = phenotype_artifact.get("access_receipt")
    if not isinstance(access, Mapping) or any(
        access.get(field) is not False
        for field in (
            "tusz_native_target_values_loaded",
            "deepsoz_target_values_loaded",
            "private_eeg_loaded",
            "private_target_values_loaded",
            "training_performed",
            "threshold_selection_performed",
        )
    ):
        raise ValueError("Phenotype artifact does not prove target-free construction")
    rows = phenotype_artifact.get("events")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Phenotype artifact has no event rows")

    source_receipt = _object(source_receipt_path)
    source_events = source_receipt.get("events")
    if not isinstance(source_events, list):
        raise TypeError("Frozen TUSZ source receipt lacks events")
    source_by_id = {
        str(row["event_id"]): row
        for row in source_events
        if isinstance(row, Mapping) and "event_id" in row
    }
    if len(source_by_id) != len(source_events):
        raise ValueError("Frozen TUSZ source receipt repeats/invalidates event IDs")

    raw_root = tusz_root.resolve(strict=True)
    if not raw_root.is_dir() or raw_root.is_symlink():
        raise ValueError("TUSZ root must be a canonical directory")
    target = output.absolute()
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)

    evaluated_rows: list[dict[str, object]] = []
    observed_predicted_cells = 0
    positive_predicted_cells = 0
    contemporaneous_evaluable_events = 0
    contemporaneous_hit_events = 0
    native_early_evaluable_events = 0
    native_early_overlap_events = 0
    involvement_0_12_evaluable_events = 0
    involvement_0_12_overlap_events = 0
    timing_errors: list[float] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Phenotype event row must be an object")
        event_id = str(row.get("event_id", ""))
        source = source_by_id.get(event_id)
        if not isinstance(source, Mapping):
            raise ValueError(f"Phenotype event is absent from source receipt: {event_id}")
        if str(row.get("relative_edf_path")) != str(source.get("relative_edf_path")):
            raise ValueError("Phenotype/source EDF identity mismatch")
        edf = _safe_edf(raw_root, source.get("relative_edf_path"))
        native = load_tusz_ictal_involvement_target(
            edf.with_suffix(".csv"),
            edf.with_suffix(".csv_bi"),
            event_index=int(source["event_index"]),
            source_path=edf,
        )
        if abs(float(source["event_t0_sec"]) - native.event_t0_sec) > 1e-9:
            raise ValueError("Native target and frozen source event t0 differ")
        if row.get("status") != "reportable":
            evaluated_rows.append(
                {
                    "event_id": event_id,
                    "status": "phenotype_abstained",
                    "reason_codes": row.get("reason_codes"),
                }
            )
            continue
        phenotype = row.get("phenotype")
        if not isinstance(phenotype, Mapping):
            raise ValueError("Reportable phenotype row lacks structured facts")
        predicted_edges = tuple(str(value) for value in phenotype["first_visible_derivations"])
        if not predicted_edges or any(value not in EDGE_INDEX for value in predicted_edges):
            raise ValueError("Phenotype contains a noncanonical first-visible edge")
        predicted_latency = float(phenotype["onset_start_sec"]) - native.event_t0_sec
        bin_index = int(math.floor(predicted_latency - TUSZ_WINDOW_START_SEC + 1e-9))
        if not 0 <= bin_index < native.targets.shape[1]:
            raise ValueError("Predicted first-visible time lies outside native target grid")
        predicted_indices = torch.tensor(
            [EDGE_INDEX[value] for value in predicted_edges], dtype=torch.long
        )
        cell_mask = native.source_target_mask[predicted_indices, bin_index]
        cell_target = native.targets[predicted_indices, bin_index]
        observed = int(cell_mask.sum().item())
        positive = int(((cell_target == 1) & cell_mask).sum().item())
        observed_predicted_cells += observed
        positive_predicted_cells += positive
        if observed > 0:
            contemporaneous_evaluable_events += 1
            if positive > 0:
                contemporaneous_hit_events += 1

        nonnegative_bin = max(0, int(round(-TUSZ_WINDOW_START_SEC)))
        post_mask = native.source_target_mask[:, nonnegative_bin:]
        post_positive = (native.targets[:, nonnegative_bin:] == 1) & post_mask
        positive_coordinates = torch.where(post_positive)
        native_early_edges: tuple[str, ...] = ()
        native_earliest_sec: float | None = None
        if positive_coordinates[0].numel() > 0:
            earliest_native_bin = int(positive_coordinates[1].min().item())
            early_edge_indices = torch.where(post_positive[:, earliest_native_bin])[0]
            native_early_edges = tuple(
                EDGE_NAMES[int(index)] for index in early_edge_indices.tolist()
            )
            native_earliest_sec = float(earliest_native_bin)
            native_early_evaluable_events += 1
            if set(predicted_edges) & set(native_early_edges):
                native_early_overlap_events += 1
            timing_errors.append(predicted_latency - native_earliest_sec)

        early_stop_bin = min(native.targets.shape[1], nonnegative_bin + 12)
        involvement_positive = (
            (native.targets[:, nonnegative_bin:early_stop_bin] == 1)
            & native.source_target_mask[:, nonnegative_bin:early_stop_bin]
        ).any(dim=1)
        involvement_edges = tuple(
            EDGE_NAMES[int(index)] for index in torch.where(involvement_positive)[0].tolist()
        )
        if involvement_edges:
            involvement_0_12_evaluable_events += 1
            if set(predicted_edges) & set(involvement_edges):
                involvement_0_12_overlap_events += 1

        evaluated_rows.append(
            {
                "event_id": event_id,
                "status": "evaluated_reportable",
                "predicted_first_visible_edges": list(predicted_edges),
                "predicted_latency_sec": predicted_latency,
                "native_contemporaneous_bin_index": bin_index,
                "observed_predicted_cell_count": observed,
                "positive_predicted_cell_count": positive,
                "native_earliest_positive_sec": native_earliest_sec,
                "native_early_edges": list(native_early_edges),
                "native_involvement_edges_0_12s": list(involvement_edges),
            }
        )

    result: dict[str, object] = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "frozen_source_native_semantic_alignment_not_soz_evaluation",
        "target_semantics": "tusz_bipolar_edge_ictal_involvement_not_soz",
        "access_receipt": {
            "tusz_native_target_values_loaded": True,
            "deepsoz_target_values_loaded": False,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "training_performed": False,
            "threshold_selection_performed": False,
            "soz_model_selection_performed": False,
        },
        "metrics": {
            "contemporaneous_predicted_cell_precision": {
                "numerator": positive_predicted_cells,
                "denominator": observed_predicted_cells,
                "value": _fraction(positive_predicted_cells, observed_predicted_cells),
            },
            "contemporaneous_event_hit": {
                "numerator": contemporaneous_hit_events,
                "denominator": contemporaneous_evaluable_events,
                "value": _fraction(
                    contemporaneous_hit_events, contemporaneous_evaluable_events
                ),
            },
            "native_earliest_edge_overlap": {
                "numerator": native_early_overlap_events,
                "denominator": native_early_evaluable_events,
                "value": _fraction(
                    native_early_overlap_events, native_early_evaluable_events
                ),
            },
            "native_involvement_0_12s_overlap": {
                "numerator": involvement_0_12_overlap_events,
                "denominator": involvement_0_12_evaluable_events,
                "value": _fraction(
                    involvement_0_12_overlap_events,
                    involvement_0_12_evaluable_events,
                ),
            },
            "predicted_minus_native_earliest_time_sec": _summary(timing_errors),
        },
        "events": evaluated_rows,
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phenotype", type=Path, default=DEFAULT_PHENOTYPE)
    parser.add_argument("--source-receipt", type=Path, default=DEFAULT_SOURCE_RECEIPT)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = evaluate(
        phenotype_path=args.phenotype,
        source_receipt_path=args.source_receipt,
        tusz_root=args.tusz_root,
        output=args.output,
    )
    print(json.dumps({"output": str(args.output), "metrics": result["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
