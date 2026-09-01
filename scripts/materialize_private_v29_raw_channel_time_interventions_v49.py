#!/usr/bin/env python3
"""Materialize target-blind private v29 raw channel-time interventions.

For every one of the 88 already frozen private v29 events, the original Top-1
channel is selected without reading the private reference.  Four non-Top-1 C18
channels are sampled from a fixed target-blind RNG.  The selected channel and
each matched control are replaced by the instantaneous across-channel mean and
the result is re-CARed in four prespecified windows: full [-12,+48), pre
[-12,0), early [0,+12), and late [+12,+48) seconds.  Identity is included,
yielding 21 conditions per event.

Each perturbed raw signal is forwarded through the frozen official LaBraM
block-9 encoder and the frozen v29 H/D heads.  This stage never opens SOZ or
private significant/spread labels and performs no training, thresholding or
intervention selection.
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

import scripts.run_labram_rank1_direct_token_oof_v28 as v28  # noqa: E402
from scripts.audit_labram_v29_h_carrier_stress_v43 import (  # noqa: E402
    _private_h_fold_probability,
)
from scripts.materialize_private_labram_evidence_v18 import (  # noqa: E402
    DEFAULT_BUNDLE,
    DEFAULT_CHECKPOINT,
    DEFAULT_MODELING,
    _read_manifest,
    _read_signal_roster,
    _safe_private_edf,
)
from scripts.predict_private_labram_portable_equal_v29 import (  # noqa: E402
    _direct_probability,
)
from src.soz.data.edf import CausalEDFConfig, load_standard19_edf_event  # noqa: E402
from src.soz.models.labram import (  # noqa: E402
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    bind_labram_record_positions,
)
from src.soz.models.labram_peft import (  # noqa: E402
    OfficialLaBraMFrozenPrefixEncoder,
)
from src.soz.v11_reasoner import (  # noqa: E402
    V11_CANDIDATE_MASK,
    extract_block9_phase_contrasts,
)


SCHEMA = "trustworthy_soz_private_v29_raw_channel_time_interventions_v49"
DEFAULT_V16 = (
    ROOT / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815"
)
DEFAULT_V28 = ROOT / "outputs/labram_rank1_direct_token_oof_v28_20260815"
DEFAULT_PREDICTION = (
    ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/trustworthy_soz_private_v29_raw_channel_time_interventions_v49_20260816"
)
SEED = 20260819
CONTROL_COUNT = 4
FORWARD_CHUNK_CALLS = 45
PHASES = {
    "full": (0, 12_000, (-12.0, 48.0)),
    "pre": (0, 2_400, (-12.0, 0.0)),
    "early": (2_400, 4_800, (0.0, 12.0)),
    "late": (4_800, 12_000, (12.0, 48.0)),
}
CANDIDATE_INDICES = torch.nonzero(
    V11_CANDIDATE_MASK, as_tuple=False
).flatten().long()


def _intervention_ids() -> tuple[str, ...]:
    values = ["identity"]
    for phase in PHASES:
        values.append(f"top1_{phase}_removed")
        values.extend(
            f"control{control}_{phase}_removed"
            for control in range(CONTROL_COUNT)
        )
    return tuple(values)


INTERVENTION_IDS = _intervention_ids()


def _split_calls(eeg: torch.Tensor) -> torch.Tensor:
    if tuple(eeg.shape) != (19, 12_000) or eeg.dtype != torch.float32:
        raise ValueError("event must be float32 [19,12000]")
    return eeg.reshape(19, 15, 4, 200).permute(1, 0, 2, 3).contiguous()


def _remove_channel_interval(
    eeg: torch.Tensor, channel: int, start: int, stop: int
) -> torch.Tensor:
    if channel not in CANDIDATE_INDICES.tolist() or not (0 <= start < stop <= 12_000):
        raise ValueError("raw intervention channel or interval is invalid")
    result = eeg.clone()
    replacement = eeg.mean(dim=0)
    result[channel, start:stop] = replacement[start:stop]
    result -= result.mean(dim=0, keepdim=True)
    if not torch.isfinite(result).all():
        raise RuntimeError("raw intervention is non-finite")
    return result.contiguous()


def _control_channels(top1: int, *, event_ordinal: int) -> torch.Tensor:
    available = CANDIDATE_INDICES[CANDIDATE_INDICES != top1]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + event_ordinal)
    selected = available.index_select(
        0, torch.randperm(len(available), generator=generator)[:CONTROL_COUNT]
    )
    if len(torch.unique(selected)) != CONTROL_COUNT or bool((selected == top1).any()):
        raise RuntimeError("target-blind control selection failed")
    return selected


def _build_interventions(
    eeg: torch.Tensor, top1: int, controls: torch.Tensor
) -> torch.Tensor:
    values = [eeg]
    for _, (start, stop, _) in PHASES.items():
        values.append(_remove_channel_interval(eeg, top1, start, stop))
        values.extend(
            _remove_channel_interval(eeg, int(channel), start, stop)
            for channel in controls.tolist()
        )
    result = torch.stack(values).float().contiguous()
    if tuple(result.shape) != (len(INTERVENTION_IDS), 19, 12_000):
        raise RuntimeError("raw intervention stack shape drifted")
    return result


def _forward_prefix(
    encoder: OfficialLaBraMFrozenPrefixEncoder,
    interventions: torch.Tensor,
    binding: object,
    *,
    device: torch.device,
) -> torch.Tensor:
    condition_count = len(interventions)
    if condition_count < 1:
        raise ValueError("at least one raw intervention is required")
    calls = torch.stack([_split_calls(value) for value in interventions]).reshape(
        -1, 19, 4, 200
    )
    rows: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(calls), FORWARD_CHUNK_CALLS):
            rows.append(
                encoder.forward_with_record_binding(
                    calls[start : start + FORWARD_CHUNK_CALLS].to(device), binding
                )
                .detach()
                .cpu()
                .float()
            )
    prefix = torch.cat(rows).reshape(condition_count, 15, 77, 200)
    if not torch.isfinite(prefix).all():
        raise RuntimeError("intervention prefix is non-finite")
    return prefix.contiguous()


def _v29_probability(
    prefix: torch.Tensor,
    *,
    h_states: Mapping[str, torch.Tensor],
    d_states: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h = extract_block9_phase_contrasts(prefix)
    d = v28.extract_rank1_phase_features(prefix)
    h_fold = _private_h_fold_probability(h, h_states)
    d_fold = torch.stack(
        [_direct_probability(d, d_states, fold) for fold in range(5)], dim=1
    )
    combined = (0.5 * h_fold + 0.5 * d_fold).mean(dim=1)
    return combined.contiguous(), h_fold.contiguous(), d_fold.contiguous()


def materialize(
    *,
    bundle_directory: Path,
    modeling_path: Path,
    checkpoint_path: Path,
    v16_directory: Path,
    v28_directory: Path,
    prediction_directory: Path,
    output: Path,
    device: torch.device,
    limit: int | None,
    progress_every: int,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    bundle = _read_manifest((bundle_directory / "manifest.json").resolve(strict=True))
    roster = _read_signal_roster(
        (bundle_directory / "signal_roster.csv").resolve(strict=True)
    )
    roster_by_event = {str(row["event_id"]): row for row in roster}
    prediction_manifest_path = (
        prediction_directory / "manifest.json"
    ).resolve(strict=True)
    prediction_tensor_path = (
        prediction_directory / "predictions.safetensors"
    ).resolve(strict=True)
    prediction_manifest = json.loads(
        prediction_manifest_path.read_text(encoding="utf-8")
    )
    events = prediction_manifest.get("events")
    if not isinstance(events, list) or len(events) != 88:
        raise ValueError("private frozen prediction roster changed")
    prediction = load_file(str(prediction_tensor_path), device="cpu")
    original = prediction["private_portable_equal_probability"].float()
    original_top1 = original.masked_fill(
        ~V11_CANDIDATE_MASK, -torch.inf
    ).argmax(dim=1)
    if limit is not None:
        if limit < 1 or limit >= len(events):
            raise ValueError("--limit must be a strict positive smoke prefix")
        events = events[:limit]
        original = original[:limit]
        original_top1 = original_top1[:limit]

    h_state_path = (v16_directory / "outer_fold_states.safetensors").resolve(
        strict=True
    )
    d_state_path = (v28_directory / "model_and_oof.safetensors").resolve(
        strict=True
    )
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
    top1_rows: list[int] = []
    control_rows: list[torch.Tensor] = []
    signal_change_rows: list[torch.Tensor] = []
    event_receipts: list[dict[str, object]] = []
    started = time.monotonic()
    for ordinal, event in enumerate(events):
        event_id = str(event["event_id"])
        row = roster_by_event.get(event_id)
        if row is None:
            raise ValueError(f"prediction event lacks target-blind signal row: {event_id}")
        source = _safe_private_edf(eeg_root, row["relative_edf_path"])
        loaded = load_standard19_edf_event(
            source, float(row["global_event_t0_sec"]), config=config
        )
        eeg = loaded.window.data.float().contiguous()
        top1 = int(original_top1[ordinal])
        controls = _control_channels(top1, event_ordinal=ordinal)
        interventions = _build_interventions(eeg, top1, controls)
        binding = bind_labram_record_positions(
            loaded.edf_receipt.raw_channel_names,
            semantic_channels=loaded.edf_receipt.semantic_channels,
        )
        prefix = _forward_prefix(
            encoder, interventions, binding, device=device
        )
        probability, h_fold, d_fold = _v29_probability(
            prefix, h_states=h_states, d_states=d_states
        )
        rms_change = (
            (interventions - interventions[0:1]).square().mean(dim=(1, 2)).sqrt()
        )
        probability_rows.append(probability)
        h_rows.append(h_fold)
        d_rows.append(d_fold)
        top1_rows.append(top1)
        control_rows.append(controls)
        signal_change_rows.append(rms_change)
        event_receipts.append(
            {
                "event_id": event_id,
                "patient_id": str(event["patient_id"]),
                "top1_channel_index": top1,
                "control_channel_indices": controls.tolist(),
                "source_sfreq_hz": float(loaded.edf_receipt.source_sfreq_hz),
                "output_reference": loaded.signal_receipt.output_reference,
            }
        )
        if (ordinal + 1) % progress_every == 0 or ordinal + 1 == len(events):
            print(
                f"raw-v49 {ordinal + 1}/{len(events)} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    probability_tensor = torch.stack(probability_rows).contiguous()
    identity_difference = float(
        (probability_tensor[:, 0] - original).abs().max()
    )
    if identity_difference > 1e-5:
        raise ValueError(f"raw identity replay drifted: {identity_difference}")
    tensors = {
        "probability": probability_tensor,
        "H_fold_probability": torch.stack(h_rows).contiguous(),
        "D_fold_probability": torch.stack(d_rows).contiguous(),
        "original_top1_channel": torch.tensor(top1_rows, dtype=torch.long),
        "control_channels": torch.stack(control_rows).long().contiguous(),
        "raw_rms_change": torch.stack(signal_change_rows).float().contiguous(),
        "candidate_mask": V11_CANDIDATE_MASK.clone(),
    }
    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": (
            "completed_target_blind_private_raw_intervention_materialization"
            if limit is None
            else "completed_target_blind_private_raw_intervention_smoke"
        ),
        "event_count": len(events),
        "intervention_count_per_event": len(INTERVENTION_IDS),
        "intervention_ids": list(INTERVENTION_IDS),
        "phase_windows_sec": {
            name: list(values[2]) for name, values in PHASES.items()
        },
        "control_count_per_event": CONTROL_COUNT,
        "control_selection": "fixed_seed_uniform_without_replacement_from_non_top1_C18",
        "replacement": "instantaneous_all19_channel_mean_then_reCAR19",
        "candidate_selection": "frozen_v29_original_top1_without_target_access",
        "identity_replay_max_absolute_probability_difference": identity_difference,
        "events": event_receipts,
        "tensor_file": "raw_intervention_predictions.safetensors",
        "source_files": {
            "signal_bundle": str(bundle_directory),
            "frozen_private_prediction": str(prediction_tensor_path.relative_to(ROOT)),
            "H_fold_states": str(h_state_path.relative_to(ROOT)),
            "D_fold_states": str(d_state_path.relative_to(ROOT)),
        },
        "access_receipt": {
            "private_raw_EEG_loaded": True,
            "private_signal_roster_loaded": True,
            "private_target_or_spread_ledger_loaded": False,
            "DeepSOZ_target_loaded": False,
            "frozen_prediction_loaded_for_target_blind_top1_selection": True,
            "foundation_forward_performed": True,
            "foundation_or_reasoner_training_performed": False,
            "intervention_threshold_or_report_selected": False,
        },
        "interpretation_boundary": {
            "raw_counterfactual_reliance_audit": True,
            "replacement_may_be_out_of_distribution": True,
            "matched_non_top1_channel_controls_included": True,
            "phase_names_are_clinical_onset_or_propagation_labels": False,
            "specific_reported_waveform_interval_tested": False,
            "biological_causality_established": False,
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
        save_file(dict(tensors), str(staging / "raw_intervention_predictions.safetensors"))
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
        output=args.output,
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
