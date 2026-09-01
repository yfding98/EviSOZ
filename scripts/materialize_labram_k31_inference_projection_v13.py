#!/usr/bin/env python3
"""Broker six legacy v1.2 k31 runs into one minimal inference projection.

This is the only v13 process that imports the historical recovery loader.  It
therefore discloses that full legacy manifests, native-evaluation roster
metadata, training-run metrics, and checkpoint weights were loaded.  It never
opens a target snapshot, gate signal/token, clinical identity/outcome, or model
forward.  The atomically published projection excludes those legacy metadata
payloads and retains only inference-critical identities and weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.ictal_k31_inference_projection_v13 import (  # noqa: E402
    EXPECTED_CANDIDATE,
    EXPECTED_CONTEXT_DIRECTION,
    EXPECTED_CONTEXT_SECONDS,
    EXPECTED_HEAD_CONFIG,
    EXPECTED_PRODUCER_ORDER,
    EXPECTED_TARGET_SEMANTICS,
    LegacyK31ProjectionSourceV13,
    publish_k31_inference_projection_v13,
)
from src.soz.ictal_recovery_oof_v1_2 import (  # noqa: E402
    LABRAM_K31_OOF_RUN_SCHEMA_V1_2,
    load_labram_k31_oof_recovery_run_v1_2,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(value: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-v5-split-sha256", type=_sha256, required=True)
    for selection in EXPECTED_PRODUCER_ORDER:
        parser.add_argument(f"--legacy-{selection}", type=Path, required=True)
        parser.add_argument(
            f"--expected-legacy-{selection}-manifest-sha256",
            type=_sha256,
            required=True,
        )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _progress(stage: str, **extra: object) -> None:
    print(
        json.dumps(
            {
                "stage": stage,
                "legacy_full_manifests_loaded": stage
                != "before_legacy_bundle_load",
                "legacy_native_evaluation_roster_metadata_loaded": stage
                != "before_legacy_bundle_load",
                "legacy_training_run_metrics_loaded": stage
                != "before_legacy_bundle_load",
                "legacy_checkpoint_weights_loaded": stage
                != "before_legacy_bundle_load",
                "broker_target_snapshot_files_opened": False,
                "broker_target_values_loaded": False,
                "broker_target_masks_loaded": False,
                "broker_gate_signal_or_tokens_loaded": False,
                "broker_forward_performed": False,
                "broker_evaluation_performed": False,
                **extra,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _load_legacy_runs(args: argparse.Namespace):
    return tuple(
        load_labram_k31_oof_recovery_run_v1_2(
            getattr(args, f"legacy_{selection}"),
            expected_manifest_sha256=getattr(
                args, f"expected_legacy_{selection}_manifest_sha256"
            ),
        )
        for selection in EXPECTED_PRODUCER_ORDER
    )


def _projection_sources(
    runs: Sequence[object], *, expected_v5_split_sha256: str
) -> tuple[LegacyK31ProjectionSourceV13, ...]:
    values = tuple(runs)
    if len(values) != 6:
        raise ValueError("Broker requires exactly six strict legacy runs")
    sources: list[LegacyK31ProjectionSourceV13] = []
    first_gate: tuple[str, ...] | None = None
    first_gate_sha: str | None = None
    for index, (selection, run) in enumerate(
        zip(EXPECTED_PRODUCER_ORDER, values, strict=True)
    ):
        manifest = run.manifest
        expected_fold = None if selection == "final" else index
        fixed = {
            "schema_version": LABRAM_K31_OOF_RUN_SCHEMA_V1_2,
            "selection": selection,
            "oof_fold": expected_fold,
            "candidate": EXPECTED_CANDIDATE,
            "context_seconds": EXPECTED_CONTEXT_SECONDS,
            "context_direction": EXPECTED_CONTEXT_DIRECTION,
            "target_semantics": EXPECTED_TARGET_SEMANTICS,
            "head_config": EXPECTED_HEAD_CONFIG,
            "v5_split_sha256": expected_v5_split_sha256,
            "tusz_ictal_involvement_targets_loaded": True,
            "i_gate_outcomes_opened": False,
            "formal_promotion": False,
            "checkpoint_authorized_for_formal_evidence_or_reasoner": False,
        }
        if any(manifest.get(field) != expected for field, expected in fixed.items()):
            raise ValueError(f"Legacy {selection} changed a frozen boundary")
        gate = tuple(manifest["i_gate_patient_ids_excluded_unopened"])
        gate_sha = str(manifest["i_gate_patient_roster_sha256"])
        if first_gate is None:
            first_gate, first_gate_sha = gate, gate_sha
        elif gate != first_gate or gate_sha != first_gate_sha:
            raise ValueError("Legacy producers do not share one I-gate roster")
        fit = tuple(manifest["training_public_patient_ids"])
        if set(fit) & set(gate):
            raise ValueError(f"Legacy {selection} fit roster intersects the I-gate")
        checkpoint_path = run.path / str(manifest["checkpoint_filename"])
        sources.append(
            LegacyK31ProjectionSourceV13(
                selection=selection,
                oof_fold=expected_fold,
                legacy_recovery_manifest_sha256=run.manifest_sha256,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=str(manifest["checkpoint_sha256"]),
                head_state_sha256=str(manifest["head_state_sha256"]),
                fit_patient_ids=fit,
                fit_patient_roster_sha256=str(
                    manifest["training_public_roster_sha256"]
                ),
                gate_patient_ids=gate,
                gate_patient_roster_sha256=gate_sha,
                v5_split_sha256=str(manifest["v5_split_sha256"]),
            )
        )
    return tuple(sources)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _progress("before_legacy_bundle_load", projection_published=False)
    runs = _load_legacy_runs(args)
    _progress("strict_legacy_bundles_loaded", projection_published=False)
    sources = _projection_sources(
        runs, expected_v5_split_sha256=args.expected_v5_split_sha256
    )
    del runs
    if args.preflight_only:
        _progress(
            "projection_broker_preflight_complete",
            projection_published=False,
            producer_count=len(sources),
            gate_patient_count=len(sources[0].gate_patient_ids),
        )
        return 0
    artifact = publish_k31_inference_projection_v13(
        args.output_directory, sources=sources
    )
    _progress(
        "minimal_inference_projection_atomically_published",
        projection_published=True,
        path=str(artifact.path),
        manifest_sha256=artifact.manifest_sha256,
        producer_count=len(artifact.producers),
        gate_patient_count=len(artifact.gate_patient_ids),
        v13_execution_hold=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
