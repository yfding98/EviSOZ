#!/usr/bin/env python3
"""Materialize and rank detector-selected long-EEG windows with frozen v29."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Initialize the long-recording package before importing the v29 module.  The
# package re-exports event materialization helpers that also depend on v29;
# bootstrapping it first prevents the standalone CLI from observing a partially
# initialized v29 module.
import src.clinical_eeg_long_recording  # noqa: E402,F401

from src.soz.v29_long_recording_inference import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_DIRECT_STATES,
    DEFAULT_H_STATES,
    DEFAULT_MODELING,
    DEFAULT_PUBLIC_FREEZE_MANIFEST,
    TENSOR_FILE,
    run_filtered_frozen_v29_candidate_batch,
)


def _strict_json(path: Path) -> object:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"JSON input must be a regular non-symlinked file: {path}")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"JSON contains duplicate key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"JSON contains invalid constant {value!r}")

    return json.loads(
        resolved.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=invalid_constant,
    )


def _publish(
    output: Path,
    selection: dict[str, Any],
    manifest: dict[str, Any] | None,
    tensors: dict[str, torch.Tensor],
) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        selection_path = staging / "analysis_selection_manifest.json"
        selection_path.write_text(
            json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.chmod(selection_path, 0o600)
        if manifest is None:
            if tensors:
                raise ValueError("zero-analyzable selection must not publish tensors")
            no_ranking_path = staging / "no_ranking_receipt.json"
            no_ranking_path.write_text(
                json.dumps(
                    {
                        "schema_version": "soz_v29_no_analyzable_candidates_v1",
                        "status": "completed_no_analyzable_candidates",
                        "recording_id": selection["recording_id"],
                        "patient_pseudonym": selection["patient_pseudonym"],
                        "analysis_selection_sha256": hashlib.sha256(
                            json.dumps(
                                selection,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            ).encode("utf-8")
                        ).hexdigest(),
                        "detector_selected_count": selection[
                            "detector_selected_count"
                        ],
                        "analyzable_count": 0,
                        "rejected_count": selection["rejected_count"],
                        "model_loaded": False,
                        "placeholder_tensor_generated": False,
                        "rejection_is_not_no_seizure": True,
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(no_ranking_path, 0o600)
        else:
            tensor_path = staging / TENSOR_FILE
            manifest_path = staging / "manifest.json"
            save_file(tensors, str(tensor_path))
            manifest_path.write_text(
                json.dumps(
                    manifest, ensure_ascii=False, indent=2, allow_nan=False
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(tensor_path, 0o600)
            os.chmod(manifest_path, 0o600)
        os.replace(staging, target)
        os.chmod(target, 0o700)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--recording-edf", type=Path, required=True)
    parser.add_argument("--detection-manifest", type=Path, required=True)
    parser.add_argument(
        "--event-id-assignment",
        "--event-registry",
        dest="event_id_assignment",
        type=Path,
        required=True,
        help=(
            "Strict candidate/event assignment or "
            "clinical_eeg_detector_aligned_frozen_event_registry_v1 JSON"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modeling", type=Path, default=DEFAULT_MODELING)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--direct-states", type=Path, default=DEFAULT_DIRECT_STATES)
    parser.add_argument("--h-states", type=Path, default=DEFAULT_H_STATES)
    parser.add_argument(
        "--public-freeze-manifest",
        type=Path,
        default=DEFAULT_PUBLIC_FREEZE_MANIFEST,
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    detection_manifest = _strict_json(args.detection_manifest)
    event_id_assignment = _strict_json(args.event_id_assignment)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    selection, manifest, tensors = run_filtered_frozen_v29_candidate_batch(
        recording_path=args.recording_edf,
        detection_manifest=detection_manifest,
        event_id_assignment=event_id_assignment,
        device=torch.device(device_name),
        modeling_path=args.modeling,
        checkpoint_path=args.checkpoint,
        direct_states_path=args.direct_states,
        h_states_path=args.h_states,
        public_freeze_manifest_path=args.public_freeze_manifest,
    )
    published = _publish(args.output, selection, manifest, tensors)
    print(
        json.dumps(
            {
                "output": str(published),
                "detector_selected_count": selection[
                    "detector_selected_count"
                ],
                "event_count": selection["analyzable_count"],
                "rejected_count": selection["rejected_count"],
                "method_id": manifest["method_id"] if manifest else None,
                "model_loaded": manifest is not None,
                "research_only": True,
                "legacy_88_event_roster_loaded": False,
                "target_values_loaded": False,
                "final_segment_generated": False,
                "waveform_figure_generated": False,
                "clinical_report_facts_generated": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
