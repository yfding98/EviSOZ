#!/usr/bin/env python3
"""Materialize compact private rank-1 phase tokens without reading targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_private_labram_evidence_v18 import (  # noqa: E402
    DEFAULT_BUNDLE,
    DEFAULT_CHECKPOINT,
    DEFAULT_MODELING,
    FORBIDDEN_TARGET_FIELDS,
    _read_manifest,
    _read_signal_roster,
    _safe_private_edf,
    _split_calls,
)
from scripts.run_labram_rank1_direct_token_oof_v28 import (  # noqa: E402
    extract_rank1_phase_features,
)
from src.soz.data.edf import CausalEDFConfig, load_standard19_edf_event  # noqa: E402
from src.soz.models.labram import (  # noqa: E402
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    bind_labram_record_positions,
)
from src.soz.models.labram_peft import OfficialLaBraMFrozenPrefixEncoder  # noqa: E402


SCHEMA = "soz_private_target_blind_rank1_phase_v29"
DEFAULT_EXPECTED = ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814/manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/private_target_blind_rank1_phase_v29_20260815"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def materialize(args: argparse.Namespace) -> tuple[dict[str, object], torch.Tensor]:
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    bundle_manifest = _read_manifest(args.bundle / "manifest.json")
    roster = _read_signal_roster(args.bundle / "signal_roster.csv")
    if FORBIDDEN_TARGET_FIELDS & set(roster[0]):
        raise ValueError("private signal roster contains target fields")
    roster_by_id = {row["event_id"]: row for row in roster}
    expected_manifest = _json(args.expected_manifest)
    expected_events = expected_manifest.get("events")
    access = expected_manifest.get("access_receipt")
    if not isinstance(expected_events, list) or len(expected_events) != 88:
        raise ValueError("expected target-blind private event roster changed")
    if not isinstance(access, Mapping) or access.get("target_ledger_opened") is not False:
        raise ValueError("expected private roster is not target blind")
    event_ids = [str(value["event_id"]) for value in expected_events]
    if any(value not in roster_by_id for value in event_ids):
        raise ValueError("target-blind event is absent from signal roster")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    encoder = OfficialLaBraMFrozenPrefixEncoder(
        modeling_path=args.modeling.resolve(strict=True),
        checkpoint_path=args.checkpoint.resolve(strict=True),
        expected_sha256=AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
    ).to(device).eval()
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("private v29 foundation exposes trainable parameters")
    eeg_root = Path(str(bundle_manifest["eeg_root"])).resolve(strict=True)
    config = CausalEDFConfig(reference_policy="unlabeled_common_car19")
    rows: list[torch.Tensor] = []
    started = time.monotonic()
    for ordinal, event_id in enumerate(event_ids, start=1):
        source_row = roster_by_id[event_id]
        source = _safe_private_edf(eeg_root, source_row["relative_edf_path"])
        loaded = load_standard19_edf_event(
            source, float(source_row["global_event_t0_sec"]), config=config
        )
        calls = _split_calls(loaded.window.data).to(device)
        binding = bind_labram_record_positions(
            loaded.edf_receipt.raw_channel_names,
            semantic_channels=loaded.edf_receipt.semantic_channels,
        )
        with torch.inference_mode():
            prefix = encoder.forward_with_record_binding(calls, binding)
        prefix = prefix.detach().cpu().float().contiguous()
        rows.append(extract_rank1_phase_features(prefix.unsqueeze(0))[0])
        if ordinal % args.progress_every == 0 or ordinal == len(event_ids):
            print(
                json.dumps(
                    {
                        "stage": "private_target_blind_phase",
                        "complete": ordinal,
                        "total": len(event_ids),
                        "elapsed_sec": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    phase = torch.stack(rows).float().contiguous()
    if tuple(phase.shape) != (88, 19, 5, 200) or not torch.isfinite(phase).all():
        raise RuntimeError("private v29 phase feature contract failed")
    manifest: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_target_blind_private_phase_materialization",
        "event_count": 88,
        "events": expected_events,
        "tensor_file": "phase_features.safetensors",
        "tensor_shape": [88, 19, 5, 200],
        "preprocessing": expected_manifest.get("preprocessing"),
        "access_receipt": {
            "signal_roster_loaded": True,
            "raw_private_eeg_loaded": True,
            "existing_target_blind_event_manifest_loaded": True,
            "private_target_ledger_path_argument_exposed": False,
            "private_target_values_loaded": False,
            "private_prediction_or_metric_loaded": False,
            "foundation_training_performed": False,
            "foundation_trainable_parameters": 0,
            "reasoner_training_performed": False,
        },
        "claim_boundary": {
            "phase_boundaries_are_clinical_soz_onset": False,
            "phase_features_are_propagation_labels": False,
        },
    }
    return manifest, phase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--modeling", type=Path, default=DEFAULT_MODELING)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--expected-manifest", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, phase = materialize(args)
    args.output.mkdir(parents=True)
    save_file({"phase_features": phase}, str(args.output / "phase_features.safetensors"))
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "events": 88, "private_targets_loaded": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
