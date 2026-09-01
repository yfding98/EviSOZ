#!/usr/bin/env python3
"""Materialize one fold/phase detector reference authority.

The command accepts only one outer fold and one of ``selection_fit``,
``inner_validation`` or ``final_refit``.  The underlying authority has no
route to source-dev, source-eval, private, or outer-held-out references.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from src.clinical_eeg_long_recording.detector_fold_reference_authority_v1 import (
    DEFAULT_REGISTRY_PATH,
    REFERENCE_PHASES_V1,
    ROOT,
    authorize_detector_fold_reference_phase_receipt_v1,
    materialize_detector_fold_reference_phase_v1,
    validate_detector_fold_reference_authority_registry_v1,
)


DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")


def _write_new_json(path: Path, value: object) -> None:
    target = path.resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite authority receipt: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json_object(path: Path | None, context: str) -> dict | None:
    if path is None:
        return None
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{context} must be a regular non-symlink JSON file")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{context} must contain a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--phase", choices=REFERENCE_PHASES_V1, required=True)
    parser.add_argument(
        "--phase-gate-receipt-sha256",
        help=(
            "deprecated fail-closed input; bare gate hashes are never accepted"
        ),
    )
    parser.add_argument("--controller-bundle-root", type=Path)
    parser.add_argument("--controller-ledger-relative-path")
    parser.add_argument("--prior-selection-fit-phase-receipt", type=Path)
    parser.add_argument("--prior-inner-validation-phase-receipt", type=Path)
    parser.add_argument(
        "--prior-inner-controller-ledger-relative-path",
        help=(
            "required for final_refit: ledger used to actual-byte replay and "
            "reauthorize the serialized prior inner-validation receipt"
        ),
    )
    parser.add_argument("--selected-epoch-metric-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    registry_path = args.registry.resolve(strict=True)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    fold_path = ROOT / registry["fold_plan_binding"]["path"]
    fold_plan = json.loads(fold_path.read_text(encoding="utf-8"))
    registry = validate_detector_fold_reference_authority_registry_v1(
        registry, fold_plan=fold_plan, verify_bound_files=True
    )
    prior_selection = _read_json_object(
        args.prior_selection_fit_phase_receipt,
        "prior selection-fit phase receipt",
    )
    prior_inner = _read_json_object(
        args.prior_inner_validation_phase_receipt,
        "prior inner-validation phase receipt",
    )
    selected_metric = _read_json_object(
        args.selected_epoch_metric_receipt,
        "selected-epoch metric receipt",
    )
    prior_selection_authority = None
    if prior_selection is not None:
        prior_selection_authority = (
            authorize_detector_fold_reference_phase_receipt_v1(
                prior_selection,
                fold_plan=fold_plan,
                registry=registry,
                replay_reference_root=args.reference_root,
            )
        )
    prior_inner_authority = None
    if prior_inner is not None:
        if prior_selection_authority is None:
            raise PermissionError(
                "reauthorizing prior inner-validation requires prior selection-fit authority"
            )
        if (
            args.controller_bundle_root is None
            or args.prior_inner_controller_ledger_relative_path is None
        ):
            raise PermissionError(
                "reauthorizing prior inner-validation requires its controller bundle/ledger"
            )
        prior_inner_authority = authorize_detector_fold_reference_phase_receipt_v1(
            prior_inner,
            fold_plan=fold_plan,
            registry=registry,
            replay_reference_root=args.reference_root,
            controller_bundle_root=args.controller_bundle_root,
            controller_ledger_relative_path=(
                args.prior_inner_controller_ledger_relative_path
            ),
            prior_selection_fit_authority=prior_selection_authority,
        )
    authority = materialize_detector_fold_reference_phase_v1(
        fold_plan=fold_plan,
        registry=registry,
        reference_root=args.reference_root,
        outer_fold_id=args.outer_fold,
        phase=args.phase,
        phase_gate_receipt_sha256=args.phase_gate_receipt_sha256,
        controller_bundle_root=args.controller_bundle_root,
        controller_ledger_relative_path=args.controller_ledger_relative_path,
        prior_selection_fit_phase_receipt=prior_selection_authority,
        prior_inner_validation_phase_receipt=prior_inner_authority,
        selected_epoch_metric_receipt=selected_metric,
    )
    receipt = authority.to_receipt()
    _write_new_json(args.output, receipt)
    print(
        json.dumps(
            {
                "authority_id": receipt["authority_id"],
                "receipt_sha256": receipt["receipt_sha256"],
                "outer_fold_id": receipt["outer_fold_id"],
                "phase": receipt["phase"],
                "recording_count": receipt["authorized_roster"]["recording_count"],
                "reference_files_opened": receipt["reference_open_log"][
                    "reference_files_opened"
                ],
                "outer_heldout_reference_files_opened": 0,
                "source_eval_reference_files_opened": 0,
                "private_reference_files_opened": 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
