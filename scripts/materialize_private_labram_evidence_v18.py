#!/usr/bin/env python3
"""Materialize target-blind LaBraM and fine evidence for private v18.

Only ``signal_roster.csv`` is opened.  The sibling target ledger is not read.
Each successful event is reduced immediately to compact block-9 phase
contrasts and target-free temporal descriptors; full prefix tensors are never
written, which keeps the private transfer path feasible on a nearly full disk.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time
from typing import Mapping

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.edf import (  # noqa: E402
    CausalEDFConfig,
    EDFEventEligibilityError,
    load_standard19_edf_event,
)
from src.soz.fine_temporal_evidence import (  # noqa: E402
    FINE_TEMPORAL_FEATURE_NAMES,
    extract_fine_temporal_evidence,
)
from src.soz.models.labram import (  # noqa: E402
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    bind_labram_record_positions,
)
from src.soz.models.labram_peft import (  # noqa: E402
    OfficialLaBraMFrozenPrefixEncoder,
)
from src.soz.v11_reasoner import extract_block9_phase_contrasts  # noqa: E402


SCHEMA = "soz_private_target_blind_labram_evidence_v18"
SMOKE_SCHEMA = "soz_private_target_blind_labram_evidence_v18_smoke"
DEFAULT_BUNDLE = ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814"
DEFAULT_MODELING = Path("/mnt/hd1/dyf/workspace/LaBraM/modeling_finetune.py")
DEFAULT_CHECKPOINT = Path("/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth")
DEFAULT_OUTPUT = ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814"
TENSOR_FILE = "evidence.safetensors"
FORBIDDEN_TARGET_FIELDS = frozenset(
    {
        "candidate_positive_electrodes",
        "standard19_positive_electrodes",
        "known_spread_electrodes",
        "outside_head_positive_electrodes",
        "diffuse_spread_present",
        "duplicate_conflict",
        "significant_spread_overlap",
        "has_c18_positive",
        "positive_set_exhaustiveness",
        "primary_reference_preeligible",
        "expanded_reference_preeligible",
    }
)


def _read_manifest(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != (
        "soz_private_zero_adaptation_bundle_v18"
    ):
        raise ValueError("private v18 bundle manifest schema mismatch")
    return value


def _read_signal_roster(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({row.get("event_id", "") for row in rows}) != len(rows):
        raise ValueError("private signal roster is empty or duplicated")
    if FORBIDDEN_TARGET_FIELDS & set(rows[0]):
        raise ValueError("private signal roster unexpectedly contains target fields")
    return rows


def _safe_private_edf(root: Path, value: object) -> Path:
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".edf":
        raise ValueError("unsafe private EDF path")
    source = root.joinpath(*relative.parts)
    resolved = source.resolve(strict=True)
    resolved.relative_to(root)
    return resolved


def _split_calls(eeg: torch.Tensor) -> torch.Tensor:
    if tuple(eeg.shape) != (19, 12_000) or eeg.dtype != torch.float32:
        raise ValueError("private event must be float32 [19,12000]")
    return eeg.reshape(19, 15, 4, 200).permute(1, 0, 2, 3).contiguous()


def materialize(
    bundle: Path,
    modeling: Path,
    checkpoint: Path,
    output: Path,
    *,
    device: torch.device,
    limit: int | None,
    progress_every: int,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    manifest = _read_manifest(bundle / "manifest.json")
    roster = _read_signal_roster(bundle / "signal_roster.csv")
    selected = [row for row in roster if row["time_support_preeligible"] == "1"]
    if len(selected) != int(manifest["counts"]["time_support_preeligible"]):
        raise ValueError("private signal roster time-support count drifted")
    full_scope = limit is None
    if limit is not None:
        if limit < 1 or limit >= len(selected):
            raise ValueError("--limit must be a positive smoke prefix")
        selected = selected[:limit]
    eeg_root = Path(str(manifest["eeg_root"])).resolve(strict=True)
    encoder = OfficialLaBraMFrozenPrefixEncoder(
        modeling_path=modeling.resolve(strict=True),
        checkpoint_path=checkpoint.resolve(strict=True),
        expected_sha256=AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
    ).to(device).eval()
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("private v18 foundation prefix exposes trainable parameters")

    config = CausalEDFConfig(reference_policy="unlabeled_common_car19")
    h_rows: list[torch.Tensor] = []
    fine_rows: list[torch.Tensor] = []
    composite_rows: list[torch.Tensor] = []
    frequency_rows: list[torch.Tensor] = []
    node_detected_rows: list[torch.Tensor] = []
    node_latency_rows: list[torch.Tensor] = []
    edge_detected_rows: list[torch.Tensor] = []
    edge_latency_rows: list[torch.Tensor] = []
    successful: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    exclusion_counts: Counter[str] = Counter()
    centers: torch.Tensor | None = None
    started = time.monotonic()

    for ordinal, row in enumerate(selected, start=1):
        event_id = row["event_id"]
        try:
            source = _safe_private_edf(eeg_root, row["relative_edf_path"])
            loaded = load_standard19_edf_event(
                source,
                float(row["global_event_t0_sec"]),
                config=config,
            )
            calls = _split_calls(loaded.window.data).to(device)
            binding = bind_labram_record_positions(
                loaded.edf_receipt.raw_channel_names,
                semantic_channels=loaded.edf_receipt.semantic_channels,
            )
            with torch.inference_mode():
                prefix = encoder.forward_with_record_binding(calls, binding)
            prefix = prefix.detach().cpu().float().contiguous()
            if tuple(prefix.shape) != (15, 77, 200) or not torch.isfinite(prefix).all():
                raise RuntimeError("private block-9 prefix shape/value contract failed")
            h = extract_block9_phase_contrasts(prefix.unsqueeze(0))[0]
            fine = extract_fine_temporal_evidence(loaded.window.data)
            if centers is None:
                centers = fine.window_center_sec.detach().cpu().contiguous()
            elif not torch.equal(centers, fine.window_center_sec.cpu()):
                raise RuntimeError("private fine temporal grid drifted")
            h_rows.append(h)
            fine_rows.append(fine.features.detach().cpu().contiguous())
            composite_rows.append(fine.composite_trace.detach().cpu().contiguous())
            frequency_rows.append(
                fine.dominant_frequency_hz.detach().cpu().contiguous()
            )
            node_detected_rows.append(
                fine.node_change_detected.detach().cpu().contiguous()
            )
            node_latency_rows.append(
                fine.node_change_latency_sec.detach().cpu().contiguous()
            )
            edge_detected_rows.append(
                fine.bipolar_change_detected.detach().cpu().contiguous()
            )
            edge_latency_rows.append(
                fine.bipolar_change_latency_sec.detach().cpu().contiguous()
            )
            successful.append(
                {
                    "event_id": event_id,
                    "patient_id": row["patient_id"],
                    "primary_anchor_preeligible": int(row["primary_anchor_preeligible"]),
                    "expanded_anchor_preeligible": int(row["expanded_anchor_preeligible"]),
                    "source_reference_policy": loaded.signal_receipt.reference_policy,
                    "output_reference": loaded.signal_receipt.output_reference,
                    "source_sfreq_hz": loaded.edf_receipt.source_sfreq_hz,
                }
            )
        except EDFEventEligibilityError as exc:
            reason = exc.code
            exclusion_counts[reason] += 1
            excluded.append({"event_id": event_id, "patient_id": row["patient_id"], "reason": reason})
        except OSError:
            reason = "edf_reader_oserror"
            exclusion_counts[reason] += 1
            excluded.append({"event_id": event_id, "patient_id": row["patient_id"], "reason": reason})
        if ordinal % progress_every == 0 or ordinal == len(selected):
            elapsed = time.monotonic() - started
            print(
                f"private-evidence {ordinal}/{len(selected)} "
                f"success={len(successful)} excluded={len(excluded)} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    if not successful or centers is None:
        raise RuntimeError("private evidence materialization produced no events")
    tensors = {
        "h_event": torch.stack(h_rows).float().contiguous(),
        "fine_event": torch.stack(fine_rows).float().contiguous(),
        "composite_trace": torch.stack(composite_rows).float().contiguous(),
        "dominant_frequency_hz": torch.stack(frequency_rows).float().contiguous(),
        "node_change_detected": torch.stack(node_detected_rows).bool().contiguous(),
        "node_change_latency_sec": torch.stack(node_latency_rows).float().contiguous(),
        "bipolar_change_detected": torch.stack(edge_detected_rows).bool().contiguous(),
        "bipolar_change_latency_sec": torch.stack(edge_latency_rows).float().contiguous(),
        "window_center_sec": centers.float().contiguous(),
    }
    expected = {
        "h_event": (len(successful), 19, 600),
        "fine_event": (len(successful), 19, len(FINE_TEMPORAL_FEATURE_NAMES)),
        "node_change_detected": (len(successful), 19),
        "node_change_latency_sec": (len(successful), 19),
        "bipolar_change_detected": (len(successful), 20),
        "bipolar_change_latency_sec": (len(successful), 20),
    }
    for name, shape in expected.items():
        if tuple(tensors[name].shape) != shape:
            raise RuntimeError(f"private evidence tensor {name} shape drifted")
    for name in ("h_event", "fine_event", "composite_trace", "dominant_frequency_hz", "window_center_sec"):
        if not torch.isfinite(tensors[name]).all():
            raise RuntimeError(f"private evidence tensor {name} is non-finite")

    output.mkdir(parents=True)
    save_file(tensors, str(output / TENSOR_FILE))
    result: dict[str, object] = {
        "schema_version": SCHEMA if full_scope else SMOKE_SCHEMA,
        "bundle": str(bundle),
        "tensor_file": TENSOR_FILE,
        "input_time_supported_event_count": len(selected),
        "successful_event_count": len(successful),
        "excluded_event_count": len(excluded),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "events": successful,
        "excluded_events": excluded,
        "feature_names": list(FINE_TEMPORAL_FEATURE_NAMES),
        "tensor_shapes": {name: list(value.shape) for name, value in tensors.items()},
        "preprocessing": {
            "reference_policy": "unlabeled_common_car19",
            "common_reference_assumption_proven_by_header": False,
            "car19_required": True,
            "window_sec": [-12.0, 48.0],
            "output_sfreq_hz": 200.0,
        },
        "access_receipt": {
            "signal_roster_opened": True,
            "target_ledger_opened": False,
            "private_eeg_loaded": True,
            "private_target_values_loaded": False,
            "deepsoz_target_values_loaded": False,
            "model_predictions_loaded": False,
            "foundation_training_performed": False,
            "reasoner_training_performed": False,
        },
        "elapsed_sec": time.monotonic() - started,
    }
    (output / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--modeling", type=Path, default=DEFAULT_MODELING)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    result = materialize(
        args.bundle,
        args.modeling,
        args.checkpoint,
        args.output,
        device=torch.device(device_name),
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(json.dumps({key: result[key] for key in (
        "schema_version", "successful_event_count", "excluded_event_count",
        "exclusion_counts", "elapsed_sec")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
