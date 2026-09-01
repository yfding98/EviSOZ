#!/usr/bin/env python3
"""Materialize target-isolated public/private 200-Hz CAR19 event waveforms.

The exact frozen 60-second standard-19 preprocessing is replayed.  Public
events remain event level so the raw comparator can aggregate nonlinear event
predictions within patient bags.  No public SOZ values or private
significant/spread reference is read by this script.
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


SCHEMA = "trustworthy_soz_public_private_raw200_events_v60"
DEFAULT_UNION = ROOT / "outputs/public_development_union_identity_v12_20260812"
DEFAULT_PUBLIC_SIGNAL = (
    ROOT
    / "outputs/deepsoz_signal_preflight_identity_v3_20260812/"
    "deepsoz_signal_preflight_identity_v3.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_raw200_events_v60_20260816"


def _microvolts(eeg: torch.Tensor) -> torch.Tensor:
    if tuple(eeg.shape) != (19, 12_000) or eeg.dtype != torch.float32:
        raise ValueError("raw200 input must be float32 [19,12000]")
    result = eeg.mul(1_000_000.0).contiguous()
    if not torch.isfinite(result).all():
        raise RuntimeError("raw200 microvolt conversion produced non-finite values")
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
    started = time.monotonic()
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
    public_rows: list[torch.Tensor] = []
    public_patient_index: list[int] = []
    public_receipts: list[dict[str, object]] = []
    peak_microvolts = 0.0
    for ordinal, event in enumerate(public_events, start=1):
        source = _safe_edf(public_root, event.relative_edf_path)
        loaded = load_standard19_edf_event(source, event.global_t0_sec, config=public_config)
        waveform = _microvolts(loaded.window.data.float())
        public_rows.append(waveform)
        public_patient_index.append(union.patient_index[event.patient_id])
        public_receipts.append(
            {
                "event_id": event.event_id,
                "patient_id": event.patient_id,
                "source_sfreq_hz": float(loaded.edf_receipt.source_sfreq_hz),
                "output_reference": loaded.signal_receipt.output_reference,
            }
        )
        peak_microvolts = max(peak_microvolts, float(waveform.abs().max()))
        if ordinal % progress_every == 0 or ordinal == len(public_events):
            print(
                f"raw200-public {ordinal}/{len(public_events)} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    public_waveform = torch.stack(public_rows).float().contiguous()
    public_event_patient_index = torch.tensor(public_patient_index, dtype=torch.long)

    private_bundle = _read_manifest(
        (private_bundle_directory / "manifest.json").resolve(strict=True)
    )
    private_roster_all = _read_signal_roster(
        (private_bundle_directory / "signal_roster.csv").resolve(strict=True)
    )
    private_roster_by_event = {str(row["event_id"]): row for row in private_roster_all}
    prediction_manifest = json.loads(
        (private_prediction_directory / "manifest.json")
        .resolve(strict=True)
        .read_text(encoding="utf-8")
    )
    prediction_events = prediction_manifest.get("events")
    if not isinstance(prediction_events, list) or len(prediction_events) != 88:
        raise ValueError("private frozen v29 event roster changed")
    private_roster = []
    for event in prediction_events:
        row = private_roster_by_event.get(str(event["event_id"]))
        if row is None or str(row["patient_id"]) != str(event["patient_id"]):
            raise ValueError("private frozen event lacks aligned signal row")
        private_roster.append(row)
    if private_limit is not None:
        if private_limit < 1 or private_limit >= len(private_roster):
            raise ValueError("--private-limit must be a strict positive smoke prefix")
        private_roster = private_roster[:private_limit]

    private_root = Path(str(private_bundle["eeg_root"])).resolve(strict=True)
    private_config = CausalEDFConfig(reference_policy="unlabeled_common_car19")
    private_rows: list[torch.Tensor] = []
    private_receipts: list[dict[str, object]] = []
    for ordinal, row in enumerate(private_roster, start=1):
        source = _safe_private_edf(private_root, row["relative_edf_path"])
        loaded = load_standard19_edf_event(
            source, float(row["global_event_t0_sec"]), config=private_config
        )
        waveform = _microvolts(loaded.window.data.float())
        private_rows.append(waveform)
        private_receipts.append(
            {
                "event_id": str(row["event_id"]),
                "patient_id": str(row["patient_id"]),
                "source_sfreq_hz": float(loaded.edf_receipt.source_sfreq_hz),
                "output_reference": loaded.signal_receipt.output_reference,
            }
        )
        peak_microvolts = max(peak_microvolts, float(waveform.abs().max()))
        if ordinal % progress_every == 0 or ordinal == len(private_roster):
            print(
                f"raw200-private {ordinal}/{len(private_roster)} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    private_waveform = torch.stack(private_rows).float().contiguous()

    formal = public_limit is None and private_limit is None
    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": (
            "completed_reference_isolated_raw200_event_waveforms"
            if formal
            else "completed_reference_isolated_raw200_event_waveforms_smoke"
        ),
        "protocol": "research/02_method/post_open_fixed_audit_extensions_v60_20260816_zh.md",
        "preprocessing": {
            "shape": [19, 12_000],
            "sampling_frequency_hz": 200.0,
            "reference": "common_average_standard19",
            "window_sec": [-12.0, 48.0],
            "stored_unit": "microvolt_float32",
            "public_unit": "event",
            "private_unit": "event",
            "peak_absolute_microvolts": peak_microvolts,
        },
        "public": {
            "patient_ids": list(union.patient_ids),
            "patient_folds": list(union.patient_folds),
            "event_count": len(public_receipts),
            "events": public_receipts,
        },
        "private": {
            "event_count": len(private_receipts),
            "events": private_receipts,
        },
        "tensor_file": "raw200_events.safetensors",
        "access_receipt": {
            "public_raw_EEG_loaded": True,
            "public_SOZ_target_values_loaded": False,
            "private_raw_EEG_loaded": True,
            "private_significant_or_spread_reference_loaded": False,
            "foundation_forward_or_training_performed": False,
            "baseline_training_performed": False,
        },
        "interpretation_boundary": {
            "materialization_executed_after_private_reference_opening": True,
            "code_path_is_reference_isolated": True,
            "fresh_or_target_blind_confirmation": False,
            "private_may_select_model_or_hyperparameters": False,
        },
        "elapsed_sec": time.monotonic() - started,
    }
    tensors = {
        "public.event_waveform_microvolts": public_waveform,
        "public.event_patient_union_index": public_event_patient_index,
        "private.event_waveform_microvolts": private_waveform,
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
        save_file(dict(tensors), str(staging / "raw200_events.safetensors"))
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
    parser.add_argument("--private-prediction", type=Path, default=DEFAULT_PRIVATE_PREDICTION)
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
                "reference_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
