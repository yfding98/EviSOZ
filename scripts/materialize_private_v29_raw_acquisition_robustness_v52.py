#!/usr/bin/env python3
"""Materialize target-blind private v29 raw acquisition robustness conditions.

The frozen v29 model is replayed on all 88 private events under a fixed grid:
identity; event-anchor shifts of -5, -2, +2, and +5 seconds; global amplitude
scales of 0.5 and 2.0; and exhaustive full-window removal of each C18 candidate
channel.  Channel removal uses the same instantaneous all-channel mean followed
by CAR19 as v49.  Every condition is materialized before the historically open
private significant/spread reference is read by the separate audit stage.

This is a post-open, target-blind model robustness audit.  It performs no model
training, threshold selection, report editing, or clinical uncertainty
calibration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_private_labram_evidence_v18 import (  # noqa: E402
    DEFAULT_BUNDLE,
    DEFAULT_CHECKPOINT,
    DEFAULT_MODELING,
    _read_manifest,
    _read_signal_roster,
    _safe_private_edf,
)
from scripts.materialize_private_v29_raw_channel_time_interventions_v49 import (  # noqa: E402
    _forward_prefix,
    _remove_channel_interval,
    _v29_probability,
)
from src.soz.data.edf import CausalEDFConfig, load_standard19_edf_event  # noqa: E402
from src.soz.geometry import STANDARD_19  # noqa: E402
from src.soz.models.labram import (  # noqa: E402
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    bind_labram_record_positions,
)
from src.soz.models.labram_peft import OfficialLaBraMFrozenPrefixEncoder  # noqa: E402
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SCHEMA = "trustworthy_soz_private_v29_raw_acquisition_robustness_v52"
DEFAULT_V16 = ROOT / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815"
DEFAULT_V28 = ROOT / "outputs/labram_rank1_direct_token_oof_v28_20260815"
DEFAULT_PREDICTION = (
    ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/trustworthy_soz_private_v29_raw_acquisition_robustness_v52_20260816"
)
ANCHOR_SHIFTS_SEC = (-5.0, -2.0, 2.0, 5.0)
AMPLITUDE_SCALES = (0.5, 2.0)
FORWARD_CHUNK_CALLS = 45
CANDIDATE_INDICES = torch.nonzero(V11_CANDIDATE_MASK, as_tuple=False).flatten().long()
CANDIDATE_CHANNELS = tuple(STANDARD_19[index] for index in CANDIDATE_INDICES.tolist())


def _number_id(value: float) -> str:
    sign = "m" if value < 0 else "p"
    magnitude = str(abs(value)).replace(".", "p")
    if magnitude.endswith("p0"):
        magnitude = magnitude[:-2]
    return f"{sign}{magnitude}"


def _scale_id(value: float) -> str:
    return str(value).replace(".", "p")


def _condition_ids() -> tuple[str, ...]:
    return (
        "identity",
        *(f"anchor_shift_{_number_id(value)}s" for value in ANCHOR_SHIFTS_SEC),
        *(f"amplitude_scale_{_scale_id(value)}" for value in AMPLITUDE_SCALES),
        *(f"drop_{channel}" for channel in CANDIDATE_CHANNELS),
    )


CONDITION_IDS = _condition_ids()


def _build_nonanchor_conditions(eeg: torch.Tensor) -> torch.Tensor:
    if tuple(eeg.shape) != (19, 12_000) or eeg.dtype != torch.float32:
        raise ValueError("private event must be float32 [19,12000]")
    values = [eeg * float(scale) for scale in AMPLITUDE_SCALES]
    values.extend(
        _remove_channel_interval(eeg, int(channel), 0, 12_000)
        for channel in CANDIDATE_INDICES.tolist()
    )
    result = torch.stack(values).float().contiguous()
    expected = len(AMPLITUDE_SCALES) + len(CANDIDATE_CHANNELS)
    if tuple(result.shape) != (expected, 19, 12_000):
        raise RuntimeError("non-anchor robustness condition shape drifted")
    if not torch.isfinite(result).all():
        raise RuntimeError("non-anchor robustness condition is non-finite")
    return result


def materialize(
    *,
    bundle_directory: Path,
    modeling_path: Path,
    checkpoint_path: Path,
    v16_directory: Path,
    v28_directory: Path,
    prediction_directory: Path,
    device: torch.device,
    limit: int | None,
    progress_every: int,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    bundle = _read_manifest((bundle_directory / "manifest.json").resolve(strict=True))
    roster = _read_signal_roster((bundle_directory / "signal_roster.csv").resolve(strict=True))
    roster_by_event = {str(row["event_id"]): row for row in roster}
    prediction_manifest_path = (prediction_directory / "manifest.json").resolve(strict=True)
    prediction_tensor_path = (
        prediction_directory / "predictions.safetensors"
    ).resolve(strict=True)
    prediction_manifest = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
    events = prediction_manifest.get("events")
    if not isinstance(events, list) or len(events) != 88:
        raise ValueError("private frozen prediction roster changed")
    prediction = load_file(str(prediction_tensor_path), device="cpu")
    original = prediction["private_portable_equal_probability"].float()
    if limit is not None:
        if limit < 1 or limit >= len(events):
            raise ValueError("--limit must be a strict positive smoke prefix")
        events = events[:limit]
        original = original[:limit]

    h_state_path = (v16_directory / "outer_fold_states.safetensors").resolve(strict=True)
    d_state_path = (v28_directory / "model_and_oof.safetensors").resolve(strict=True)
    h_states = load_file(str(h_state_path), device="cpu")
    d_states = load_file(str(d_state_path), device="cpu")
    encoder = OfficialLaBraMFrozenPrefixEncoder(
        modeling_path=modeling_path.resolve(strict=True),
        checkpoint_path=checkpoint_path.resolve(strict=True),
        expected_sha256=AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
    ).to(device).eval()
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("foundation encoder exposes trainable parameters")

    eeg_root = Path(str(bundle["eeg_root"])).resolve(strict=True)
    config = CausalEDFConfig(reference_policy="unlabeled_common_car19")
    probability_rows: list[torch.Tensor] = []
    h_rows: list[torch.Tensor] = []
    d_rows: list[torch.Tensor] = []
    signal_change_rows: list[torch.Tensor] = []
    event_receipts: list[dict[str, object]] = []
    started = time.monotonic()
    for ordinal, event in enumerate(events):
        event_id = str(event["event_id"])
        row = roster_by_event.get(event_id)
        if row is None:
            raise ValueError(f"prediction event lacks target-blind signal row: {event_id}")
        source = _safe_private_edf(eeg_root, row["relative_edf_path"])
        anchor = float(row["global_event_t0_sec"])
        loaded = load_standard19_edf_event(source, anchor, config=config)
        eeg = loaded.window.data.float().contiguous()
        condition_signals = [eeg]
        for shift in ANCHOR_SHIFTS_SEC:
            shifted = load_standard19_edf_event(source, anchor + shift, config=config)
            if (
                shifted.edf_receipt.raw_channel_names != loaded.edf_receipt.raw_channel_names
                or shifted.edf_receipt.semantic_channels != loaded.edf_receipt.semantic_channels
            ):
                raise RuntimeError("anchor shift changed channel identity")
            condition_signals.append(shifted.window.data.float().contiguous())
        condition_signals.extend(_build_nonanchor_conditions(eeg).unbind(dim=0))
        conditions = torch.stack(condition_signals).float().contiguous()
        if tuple(conditions.shape) != (len(CONDITION_IDS), 19, 12_000):
            raise RuntimeError("raw acquisition robustness grid shape drifted")
        binding = bind_labram_record_positions(
            loaded.edf_receipt.raw_channel_names,
            semantic_channels=loaded.edf_receipt.semantic_channels,
        )
        prefix = _forward_prefix(encoder, conditions, binding, device=device)
        probability, h_fold, d_fold = _v29_probability(
            prefix, h_states=h_states, d_states=d_states
        )
        probability_rows.append(probability)
        h_rows.append(h_fold)
        d_rows.append(d_fold)
        signal_change_rows.append(
            (conditions - conditions[0:1]).square().mean(dim=(1, 2)).sqrt()
        )
        event_receipts.append(
            {
                "event_id": event_id,
                "patient_id": str(event["patient_id"]),
                "source_sfreq_hz": float(loaded.edf_receipt.source_sfreq_hz),
                "output_reference": loaded.signal_receipt.output_reference,
            }
        )
        if (ordinal + 1) % progress_every == 0 or ordinal + 1 == len(events):
            print(
                f"raw-v52 {ordinal + 1}/{len(events)} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    probability_tensor = torch.stack(probability_rows).contiguous()
    identity_difference = float((probability_tensor[:, 0] - original).abs().max())
    if identity_difference > 1e-5:
        raise ValueError(f"raw identity replay drifted: {identity_difference}")
    tensors = {
        "probability": probability_tensor,
        "H_fold_probability": torch.stack(h_rows).contiguous(),
        "D_fold_probability": torch.stack(d_rows).contiguous(),
        "raw_rms_change": torch.stack(signal_change_rows).float().contiguous(),
        "candidate_mask": V11_CANDIDATE_MASK.clone(),
    }
    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": (
            "completed_target_blind_private_raw_acquisition_robustness"
            if limit is None
            else "completed_target_blind_private_raw_acquisition_robustness_smoke"
        ),
        "event_count": len(events),
        "condition_count_per_event": len(CONDITION_IDS),
        "condition_ids": list(CONDITION_IDS),
        "anchor_shifts_sec": list(ANCHOR_SHIFTS_SEC),
        "amplitude_scales": list(AMPLITUDE_SCALES),
        "dropped_candidate_channels": list(CANDIDATE_CHANNELS),
        "channel_removal": "instantaneous_all19_channel_mean_then_reCAR19_full_window",
        "identity_replay_max_absolute_probability_difference": identity_difference,
        "events": event_receipts,
        "tensor_file": "raw_acquisition_robustness_predictions.safetensors",
        "access_receipt": {
            "private_raw_EEG_loaded": True,
            "private_target_or_spread_ledger_loaded": False,
            "DeepSOZ_target_loaded": False,
            "foundation_forward_performed": True,
            "foundation_or_reasoner_training_performed": False,
            "condition_threshold_report_or_model_selected": False,
        },
        "interpretation_boundary": {
            "post_open_target_blind_private_robustness_audit": True,
            "anchor_shift_is_clinical_onset_uncertainty_model": False,
            "channel_mean_replacement_may_be_out_of_distribution": True,
            "amplitude_scale_is_label_preserving_sensitivity_only": True,
            "clinical_safety_or_input_contract_qualified": False,
            "private_is_fresh_validation": False,
        },
        "elapsed_sec": time.monotonic() - started,
    }
    return result, tensors


def publish(
    *, output: Path, result: Mapping[str, object], tensors: Mapping[str, torch.Tensor]
) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / "manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        save_file(dict(tensors), str(staging / "raw_acquisition_robustness_predictions.safetensors"))
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--modeling", type=Path, default=DEFAULT_MODELING)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--v16", type=Path, default=DEFAULT_V16)
    parser.add_argument("--v28", type=Path, default=DEFAULT_V28)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    result, tensors = materialize(
        bundle_directory=args.bundle,
        modeling_path=args.modeling,
        checkpoint_path=args.checkpoint,
        v16_directory=args.v16,
        v28_directory=args.v28,
        prediction_directory=args.prediction,
        device=torch.device(device_name),
        limit=args.limit,
        progress_every=args.progress_every,
    )
    output = publish(output=args.output, result=result, tensors=tensors)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "event_count": result["event_count"],
                "condition_count": result["condition_count_per_event"],
                "identity_difference": result[
                    "identity_replay_max_absolute_probability_difference"
                ],
                "target_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
