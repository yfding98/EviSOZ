#!/usr/bin/env python3
"""Materialize target-free public/private 25-Hz CAR19 waveforms for v54.

The exact frozen 60-second standard-19 preprocessing used by the current
method is replayed.  A fixed polyphase anti-aliasing resample converts 200 Hz
to 25 Hz.  Public seizures are averaged equally within each of the 103 union
patients; all 88 private events remain event level.  No public SOZ target or
private significant/spread reference is read in this materialization stage.
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

import numpy as np
from scipy.signal import resample_poly
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_private_labram_evidence_v18 import (  # noqa: E402
    DEFAULT_BUNDLE,
    _read_manifest,
    _read_signal_roster,
    _safe_private_edf,
)
from scripts.materialize_public_development_fine_evidence_identity_v12 import (  # noqa: E402
    DEFAULT_TUSZ_ROOT,
    _safe_edf,
)
from scripts.predict_private_labram_portable_equal_v29 import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_PRIVATE_PREDICTION,
)
from src.soz.data.edf import CausalEDFConfig, load_standard19_edf_event  # noqa: E402
from src.soz.data.public_development_union_identity_v12 import (  # noqa: E402
    load_public_development_union_identity_v12,
)


SCHEMA = "trustworthy_soz_public_private_raw25_waveforms_v54"
DEFAULT_UNION = ROOT / "outputs/public_development_union_identity_v12_20260812"
DEFAULT_PUBLIC_SIGNAL = (
    ROOT
    / "outputs/deepsoz_signal_preflight_identity_v3_20260812/deepsoz_signal_preflight_identity_v3.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_public_private_raw25_waveforms_v54_20260816"
OUTPUT_SFREQ_HZ = 25.0
DOWNSAMPLE_FACTOR = 8


def _raw25(eeg: torch.Tensor) -> torch.Tensor:
    if tuple(eeg.shape) != (19, 12_000) or eeg.dtype != torch.float32:
        raise ValueError("raw25 input must be float32 [19,12000]")
    value = resample_poly(
        eeg.detach().cpu().numpy(),
        up=1,
        down=DOWNSAMPLE_FACTOR,
        axis=-1,
        window=("kaiser", 5.0),
        padtype="constant",
    )
    result = torch.from_numpy(np.asarray(value, dtype=np.float32)).contiguous()
    if tuple(result.shape) != (19, 1_500) or not torch.isfinite(result).all():
        raise RuntimeError("raw25 resampling contract failed")
    return result


def materialize(
    *,
    union_directory: Path,
    public_signal_path: Path,
    tusz_root: Path,
    private_bundle_directory: Path,
    private_prediction_directory: Path,
    public_limit: int | None,
    private_limit: int | None,
    progress_every: int,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    union = load_public_development_union_identity_v12(union_directory)
    signal = json.loads(public_signal_path.resolve(strict=True).read_text(encoding="utf-8"))
    config_payload = signal.get("receipt", {}).get("preprocess_config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("public signal preflight lacks preprocess_config")
    public_config = CausalEDFConfig(**dict(config_payload))
    public_events = list(union.events)
    if public_limit is not None:
        if public_limit < 1 or public_limit >= len(public_events):
            raise ValueError("--public-limit must be a strict positive smoke prefix")
        public_events = public_events[:public_limit]

    public_root = Path(os.path.abspath(tusz_root)).resolve(strict=True)
    patient_index = union.patient_index
    public_sum = torch.zeros((len(union.patient_ids), 19, 1_500), dtype=torch.float64)
    public_count = torch.zeros(len(union.patient_ids), dtype=torch.long)
    started = time.monotonic()
    for ordinal, event in enumerate(public_events, start=1):
        source = _safe_edf(public_root, event.relative_edf_path)
        loaded = load_standard19_edf_event(source, event.global_t0_sec, config=public_config)
        waveform = _raw25(loaded.window.data.float())
        index = patient_index[event.patient_id]
        public_sum[index] += waveform.double()
        public_count[index] += 1
        if ordinal % progress_every == 0 or ordinal == len(public_events):
            print(
                f"raw25-public {ordinal}/{len(public_events)} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    observed_public = public_count > 0
    public_waveform = torch.zeros_like(public_sum, dtype=torch.float32)
    public_waveform[observed_public] = (
        public_sum[observed_public]
        / public_count[observed_public].double().view(-1, 1, 1)
    ).float()
    if public_limit is None and not bool(observed_public.all()):
        raise RuntimeError("formal public raw25 materialization lost a patient")

    private_bundle = _read_manifest(
        (private_bundle_directory / "manifest.json").resolve(strict=True)
    )
    private_roster_all = _read_signal_roster(
        (private_bundle_directory / "signal_roster.csv").resolve(strict=True)
    )
    private_roster_by_event = {
        str(row["event_id"]): row for row in private_roster_all
    }
    private_prediction_manifest = json.loads(
        (private_prediction_directory / "manifest.json")
        .resolve(strict=True)
        .read_text(encoding="utf-8")
    )
    private_prediction_events = private_prediction_manifest.get("events")
    if not isinstance(private_prediction_events, list) or len(private_prediction_events) != 88:
        raise ValueError("private frozen v29 event roster changed")
    private_roster = []
    for event in private_prediction_events:
        row = private_roster_by_event.get(str(event["event_id"]))
        if row is None or str(row["patient_id"]) != str(event["patient_id"]):
            raise ValueError("private frozen event lacks aligned target-blind signal row")
        private_roster.append(row)
    if private_limit is not None:
        if private_limit < 1 or private_limit >= len(private_roster):
            raise ValueError("--private-limit must be a strict positive smoke prefix")
        private_roster = private_roster[:private_limit]
    private_root = Path(str(private_bundle["eeg_root"])).resolve(strict=True)
    private_config = CausalEDFConfig(reference_policy="unlabeled_common_car19")
    private_rows: list[torch.Tensor] = []
    private_events: list[dict[str, object]] = []
    for ordinal, row in enumerate(private_roster, start=1):
        source = _safe_private_edf(private_root, row["relative_edf_path"])
        loaded = load_standard19_edf_event(
            source, float(row["global_event_t0_sec"]), config=private_config
        )
        private_rows.append(_raw25(loaded.window.data.float()))
        private_events.append(
            {
                "event_id": str(row["event_id"]),
                "patient_id": str(row["patient_id"]),
                "source_sfreq_hz": float(loaded.edf_receipt.source_sfreq_hz),
                "output_reference": loaded.signal_receipt.output_reference,
            }
        )
        if ordinal % progress_every == 0 or ordinal == len(private_roster):
            print(
                f"raw25-private {ordinal}/{len(private_roster)} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    private_waveform = torch.stack(private_rows).float().contiguous()

    formal = public_limit is None and private_limit is None
    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": (
            "completed_target_free_public_private_raw25_waveforms"
            if formal
            else "completed_target_free_public_private_raw25_waveforms_smoke"
        ),
        "preprocessing": {
            "input_shape": [19, 12_000],
            "input_sfreq_hz": 200.0,
            "output_shape": [19, 1_500],
            "output_sfreq_hz": OUTPUT_SFREQ_HZ,
            "resampling": "scipy_resample_poly_up1_down8_kaiser_beta5",
            "public_patient_pooling": "equal_mean_over_complete_seizure_bag",
            "private_unit": "event",
        },
        "public": {
            "patient_ids": list(union.patient_ids),
            "patient_folds": list(union.patient_folds),
            "event_count": len(public_events),
            "patient_event_counts": public_count.tolist(),
        },
        "private": {
            "event_count": len(private_events),
            "events": private_events,
        },
        "tensor_file": "raw25_waveforms.safetensors",
        "access_receipt": {
            "public_raw_EEG_loaded": True,
            "public_SOZ_target_loaded": False,
            "private_raw_EEG_loaded": True,
            "private_significant_or_spread_reference_loaded": False,
            "foundation_forward_or_training_performed": False,
            "baseline_training_performed": False,
        },
        "interpretation_boundary": {
            "raw25_is_fixed_low_bandwidth_neural_baseline_input": True,
            "raw25_is_full_bandwidth_200Hz_EEG": False,
            "public_patient_waveform_average_may_smear_event_timing": True,
            "private_is_fresh_validation": False,
        },
        "elapsed_sec": time.monotonic() - started,
    }
    tensors = {
        "public.patient_waveform": public_waveform.contiguous(),
        "public.patient_event_count": public_count.contiguous(),
        "private.event_waveform": private_waveform,
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
        save_file(dict(tensors), str(staging / "raw25_waveforms.safetensors"))
        (staging / "manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--union", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--public-signal", type=Path, default=DEFAULT_PUBLIC_SIGNAL)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--private-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--private-prediction", type=Path, default=DEFAULT_PRIVATE_PREDICTION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--public-limit", type=int)
    parser.add_argument("--private-limit", type=int)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, tensors = materialize(
        union_directory=args.union,
        public_signal_path=args.public_signal,
        tusz_root=args.tusz_root,
        private_bundle_directory=args.private_bundle,
        private_prediction_directory=args.private_prediction,
        public_limit=args.public_limit,
        private_limit=args.private_limit,
        progress_every=args.progress_every,
    )
    output = publish(output=args.output, result=result, tensors=tensors)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "public_events": result["public"]["event_count"],
                "private_events": result["private"]["event_count"],
                "target_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
