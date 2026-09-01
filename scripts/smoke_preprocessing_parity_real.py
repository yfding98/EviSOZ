#!/usr/bin/env python3
"""Real-EDF/CUDA smoke for every frozen preprocessing-parity geometry.

This command performs one genuine optimizer update for the morphology and
ictal heads under each deployable arm.  It is intentionally a smoke receipt,
not a formal arm-selection result and cannot issue a preprocessing capability.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/neurosoz-numba-cache")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.soz.data.tuev_morphology import load_tuev_morphology_manifest  # noqa: E402
from src.soz.data.tusz import load_tusz_ictal_involvement_target  # noqa: E402
from src.soz.data.tusz_training import (  # noqa: E402
    load_tusz_ictal_training_manifest,
    parse_tusz_official_train_path,
)
from src.soz.models.concept_heads import (  # noqa: E402
    IctalInvolvementHead,
    MorphologyEvidenceHead,
)
from src.soz.models.foundation import TiledFoundationEncoder  # noqa: E402
from src.soz.models.labram import OfficialLaBraMEncoder  # noqa: E402
from src.soz.preprocessing_arm_runtime import (  # noqa: E402
    OfficialReference23LaBraMEncoder,
    prepare_arm_interval,
    prepare_full_record_arm,
    read_physical_edf,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    DEPLOYABLE_PREPROCESSING_ARM_IDS,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuev-root", type=Path, required=True)
    parser.add_argument("--tuev-manifest", type=Path, required=True)
    parser.add_argument("--tuev-bundle-sha256", required=True)
    parser.add_argument("--tuev-receipt-sha256", required=True)
    parser.add_argument("--tusz-root", type=Path, required=True)
    parser.add_argument("--tusz-manifest", type=Path, required=True)
    parser.add_argument("--tusz-bundle-sha256", required=True)
    parser.add_argument("--tusz-receipt-sha256", required=True)
    parser.add_argument("--labram-modeling", type=Path, required=True)
    parser.add_argument("--labram-checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def _bundle_directory(path: Path) -> Path:
    """Accept either the strict bundle directory or its manifest.json member."""

    return path.parent if path.name == "manifest.json" else path


def _morphology_target(group, device: torch.device):
    labels = torch.zeros((1, 20, 1), dtype=torch.long, device=device)
    mask = torch.zeros((1, 20, 1), dtype=torch.bool, device=device)
    for target in group.targets:
        labels[0, target.edge_index, 0] = target.label_index
        mask[0, target.edge_index, 0] = True
    if not mask.any():
        raise ValueError("Selected morphology smoke crop has no native target")
    return labels, mask


def _one_step(head, logits, loss):
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def main() -> int:
    args = _parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but torch.cuda is unavailable")
    torch.manual_seed(20260808)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260808)
    torch.use_deterministic_algorithms(True)

    tuev = load_tuev_morphology_manifest(
        _bundle_directory(args.tuev_manifest),
        expected_bundle_manifest_sha256=args.tuev_bundle_sha256,
        expected_source_manifest_sha256=args.tuev_receipt_sha256,
    )
    tusz = load_tusz_ictal_training_manifest(
        _bundle_directory(args.tusz_manifest),
        expected_bundle_manifest_sha256=args.tusz_bundle_sha256,
        expected_source_manifest_sha256=args.tusz_receipt_sha256,
    )
    tuev_records = {record.record_id: record for record in tuev.records}
    morphology_group = next(
        group
        for group in tuev.interval_groups
        if tuev_records[group.record_id].official_split == "train"
        and group.start_sample >= 30 * 200
    )
    morphology_record = tuev_records[morphology_group.record_id]
    tuev_edf = args.tuev_root / morphology_record.relative_edf_path
    ictal_event = tusz.events[0]
    tusz_source = parse_tusz_official_train_path(
        args.tusz_root, ictal_event.relative_edf_path
    )

    encoder = OfficialLaBraMEncoder(
        modeling_path=args.labram_modeling,
        checkpoint_path=args.labram_checkpoint,
        tile_seconds=4,
    ).to(device)
    tiled = TiledFoundationEncoder(encoder, n_calls=15).to(device)
    encoder.eval()
    tiled.eval()
    morphology_raw = read_physical_edf(tuev_edf, geometry="standard19")
    ictal_raw = read_physical_edf(tusz_source.edf_path, geometry="standard19")
    native = load_tusz_ictal_involvement_target(
        tusz_source.channel_annotation_path,
        tusz_source.global_annotation_path,
        event_index=ictal_event.event_index,
        source_path=tusz_source.edf_path,
    )
    native_targets = native.targets.unsqueeze(0).to(device)
    native_mask = native.source_target_mask.unsqueeze(0).to(device)
    morphology_labels, morphology_mask = _morphology_target(
        morphology_group, device
    )

    arms: dict[str, object] = {}
    for arm_id in DEPLOYABLE_PREPROCESSING_ARM_IDS:
        morphology_full = (
            None
            if arm_id == "C-CAR19"
            else prepare_full_record_arm(morphology_raw, arm_id=arm_id)
        )
        morphology = prepare_arm_interval(
            morphology_raw,
            arm_id=arm_id,
            start_sec=morphology_group.start_sample / 200.0,
            stop_sec=morphology_group.stop_sample / 200.0,
            full_record=morphology_full,
        )
        morph_input = torch.from_numpy(morphology.data_volts).reshape(
            1, 19, 4, 200
        ).to(device)
        with torch.inference_mode():
            morphology_tokens = encoder(morph_input)[:, :, :1].detach()
        morphology_head = MorphologyEvidenceHead().to(device)
        morphology_logits = morphology_head(morphology_tokens)
        morphology_loss = F.cross_entropy(
            morphology_logits[morphology_mask],
            morphology_labels[morphology_mask],
        )
        _one_step(morphology_head, morphology_logits, morphology_loss)

        ictal_full = (
            None
            if arm_id == "C-CAR19"
            else prepare_full_record_arm(ictal_raw, arm_id=arm_id)
        )
        ictal = prepare_arm_interval(
            ictal_raw,
            arm_id=arm_id,
            start_sec=ictal_event.event_t0_sec - 12.0,
            stop_sec=ictal_event.event_t0_sec + 48.0,
            full_record=ictal_full,
        )
        ictal_input = torch.from_numpy(ictal.data_volts).unsqueeze(0).to(device)
        with torch.inference_mode():
            ictal_tokens = tiled(ictal_input).detach()
        ictal_head = IctalInvolvementHead().to(device)
        ictal_logits = ictal_head(ictal_tokens).squeeze(-1)
        ictal_loss = F.binary_cross_entropy_with_logits(
            ictal_logits[native_mask], native_targets[native_mask]
        )
        _one_step(ictal_head, ictal_logits, ictal_loss)
        arms[arm_id] = {
            "morphology_loss": float(morphology_loss.detach().cpu()),
            "ictal_loss": float(ictal_loss.detach().cpu()),
            "morphology_token_shape": list(morphology_tokens.shape),
            "ictal_token_shape": list(ictal_tokens.shape),
            "native_ictal_cells": int(native_mask.sum().item()),
        }

    official_raw = read_physical_edf(tuev_edf, geometry="official_ref23")
    official_full = prepare_full_record_arm(official_raw, arm_id="O-REF")
    target = morphology_group.targets[0]
    official = prepare_arm_interval(
        official_raw,
        arm_id="O-REF",
        start_sec=target.start_sample / 200.0 - 2.0,
        stop_sec=target.stop_sample / 200.0 + 2.0,
        full_record=official_full,
    )
    official_encoder = OfficialReference23LaBraMEncoder(
        modeling_path=args.labram_modeling,
        checkpoint_path=args.labram_checkpoint,
    ).to(device)
    official_input = torch.from_numpy(official.data_volts).reshape(
        1, 23, 5, 200
    ).to(device)
    with torch.inference_mode():
        official_tokens = official_encoder(official_input)
    payload = {
        "schema_version": "soz_preprocessing_parity_real_smoke_v1",
        "formal": False,
        "training_started": True,
        "optimizer_steps": 2 * len(DEPLOYABLE_PREPROCESSING_ARM_IDS),
        "device": str(device),
        "tuev_record_id": morphology_record.record_id,
        "tuev_crop_id": morphology_group.crop_id,
        "tusz_event_id": ictal_event.event_id,
        "arms": arms,
        "official_sanity": {
            "input_shape": list(official_input.shape),
            "token_shape": list(official_tokens.shape),
            "finite": bool(torch.isfinite(official_tokens).all().item()),
        },
    }
    print(json.dumps(payload, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
