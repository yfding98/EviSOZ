#!/usr/bin/env python3
"""Materialize private event signal descriptors without opening SOZ targets.

The input is the frozen target-blind private evidence cache. The output only
describes algorithm-detected sustained bipolar changes and later scalp-visible
node changes. It deliberately emits no rhythm class, propagation, artifact
subtype, physical-electrode onset, or SOZ statement because those producers
did not pass independent native/reader qualification.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

from safetensors.torch import load_file
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.soz.fine_temporal_evidence import (  # noqa: E402
    FINE_STRIDE_SECONDS,
    FINE_SUSTAINED_WINDOWS,
    FINE_WINDOW_SECONDS,
)
from src.soz.geometry import STANDARD_19, TCP_20_EDGES  # noqa: E402


DEFAULT_INPUT = ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814"
DEFAULT_OUTPUT = ROOT / "outputs/private_event_descriptors_target_blind_v24_20260815"
INPUT_SCHEMA = "soz_private_target_blind_labram_evidence_v18"
OUTPUT_SCHEMA = "soz_private_event_descriptors_target_blind_v24"
ROW_SCHEMA = "soz_private_event_descriptor_target_blind_v24"
EARLIEST_TIE_SEC = FINE_STRIDE_SECONDS
LATER_MIN_DELAY_SEC = 1.0


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _validate_input(manifest: Mapping[str, object], tensors: Mapping[str, torch.Tensor]) -> list[Mapping[str, object]]:
    if manifest.get("schema_version") != INPUT_SCHEMA:
        raise ValueError("private target-blind evidence schema drifted")
    access = manifest.get("access_receipt")
    if not isinstance(access, Mapping):
        raise TypeError("private evidence lacks access receipt")
    forbidden_true = (
        "target_ledger_opened",
        "private_target_values_loaded",
        "deepsoz_target_values_loaded",
        "model_predictions_loaded",
        "foundation_training_performed",
        "reasoner_training_performed",
    )
    if any(access.get(field) is not False for field in forbidden_true):
        raise ValueError("private descriptor input crossed a target/model/training firewall")
    events = manifest.get("events")
    if not isinstance(events, list) or not events:
        raise TypeError("private evidence has no event roster")
    required = {
        "bipolar_change_detected": (len(events), len(TCP_20_EDGES)),
        "bipolar_change_latency_sec": (len(events), len(TCP_20_EDGES)),
        "node_change_detected": (len(events), len(STANDARD_19)),
        "node_change_latency_sec": (len(events), len(STANDARD_19)),
    }
    for name, shape in required.items():
        tensor = tensors.get(name)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"private descriptor tensor shape drifted: {name}")
        if name.endswith("latency_sec"):
            detected_name = name.replace("_latency_sec", "_detected")
            detected = tensors[detected_name].bool()
            if not torch.isfinite(tensor[detected]).all():
                raise ValueError(
                    f"detected private descriptor latency is non-finite: {name}"
                )
    return events


def _edge_name(index: int) -> str:
    left, right = TCP_20_EDGES[index]
    return f"{left}-{right}"


def _row(
    event: Mapping[str, object],
    tensors: Mapping[str, torch.Tensor],
    index: int,
) -> dict[str, object]:
    event_id = str(event.get("event_id", "")).strip()
    patient_id = str(event.get("patient_id", "")).strip()
    if not event_id or not patient_id:
        raise ValueError("private descriptor event identity is incomplete")
    edge_detected = tensors["bipolar_change_detected"][index].bool()
    edge_latency = tensors["bipolar_change_latency_sec"][index].float()
    detected_edges = torch.where(edge_detected)[0].tolist()
    first_latency: float | None = None
    first_edges: list[str] = []
    if detected_edges:
        first_latency = min(float(edge_latency[edge]) for edge in detected_edges)
        first_edges = [
            _edge_name(edge)
            for edge in detected_edges
            if abs(float(edge_latency[edge]) - first_latency) <= EARLIEST_TIE_SEC
        ]
    interval: list[float] | None = None
    later: list[dict[str, object]] = []
    if first_latency is not None:
        interval = [
            first_latency,
            first_latency
            + FINE_WINDOW_SECONDS
            + (FINE_SUSTAINED_WINDOWS - 1) * FINE_STRIDE_SECONDS,
        ]
        node_detected = tensors["node_change_detected"][index].bool()
        node_latency = tensors["node_change_latency_sec"][index].float()
        for channel_index in torch.where(node_detected)[0].tolist():
            delay = float(node_latency[channel_index]) - first_latency
            if delay >= LATER_MIN_DELAY_SEC:
                later.append(
                    {
                        "channel": STANDARD_19[channel_index],
                        "delay_sec": delay,
                    }
                )
        later.sort(key=lambda item: (float(item["delay_sec"]), str(item["channel"])))
    return {
        "schema_version": ROW_SCHEMA,
        "event_id": event_id,
        "patient_id": patient_id,
        "algorithmic_sustained_change": {
            "status": "detected" if first_latency is not None else "not_detected",
            "support_interval_sec_relative_to_clinical_event_anchor": interval,
            "bipolar_derivation_candidates": first_edges,
            "physical_electrode_onset_truth": False,
            "soz_onset_truth": False,
        },
        "later_scalp_visible_change_candidates": later[:5],
        "qualification": {
            "rhythm_or_frequency_phrase": "withheld_native_precision_gate_failed",
            "propagation_phrase": "forbidden_no_propagation_labels",
            "artifact_type_or_severity": "unavailable_not_reader_qualified",
            "montage_consistency": "unavailable_single_unverified_reference",
        },
        "lineage": {
            "source": "frozen_private_target_blind_fine_temporal_evidence_v18",
            "private_soz_target_used": False,
            "deepsoz_target_used": False,
            "model_prediction_used": False,
        },
    }


def materialize(input_directory: Path, output_directory: Path) -> dict[str, object]:
    source = input_directory.resolve(strict=True)
    manifest = _json(source / "manifest.json")
    tensor_file = manifest.get("tensor_file")
    if not isinstance(tensor_file, str) or Path(tensor_file).name != tensor_file:
        raise ValueError("unsafe private evidence tensor basename")
    tensors = load_file(str((source / tensor_file).resolve(strict=True)))
    events = _validate_input(manifest, tensors)
    rows = [_row(event, tensors, index) for index, event in enumerate(events)]
    if len({str(row["event_id"]) for row in rows}) != len(rows):
        raise ValueError("private descriptor event IDs are not unique")
    counts: Counter[str] = Counter()
    for row in rows:
        detected = row["algorithmic_sustained_change"]["status"] == "detected"
        counts["sustained_change_detected"] += int(detected)
        counts["sustained_change_not_detected"] += int(not detected)
        counts["with_later_visible_candidate"] += int(
            bool(row["later_scalp_visible_change_candidates"])
        )
    target = output_directory.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        with (staging / "descriptors.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        output_manifest: dict[str, object] = {
            "schema_version": OUTPUT_SCHEMA,
            "status": "completed_target_blind_algorithmic_descriptors_not_clinical_onset",
            "event_count": len(rows),
            "patient_count": len({str(row["patient_id"]) for row in rows}),
            "counts": dict(sorted(counts.items())),
            "descriptor_file": "descriptors.jsonl",
            "access_receipt": {
                "private_signal_evidence_loaded": True,
                "private_soz_targets_loaded": False,
                "deepsoz_targets_loaded": False,
                "model_predictions_loaded": False,
                "training_or_threshold_selection_performed": False,
            },
            "claim_boundary": {
                "algorithmic_change_is_physical_electrode_onset": False,
                "later_visible_is_propagation": False,
                "rhythm_band_reportable": False,
                "artifact_or_montage_statement_reportable": False,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(output_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return output_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--input-directory", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = materialize(args.input_directory, args.output_directory)
    print(json.dumps({"output": str(args.output_directory), **result["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
