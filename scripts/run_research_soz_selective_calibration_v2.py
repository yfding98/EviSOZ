#!/usr/bin/env python3
"""Freeze, calibrate, project, or formally evaluate research scalp-SOZ v2.

Prediction freezing reads only validated research prediction/strength JSON
sidecars.  Development weak labels are imported by a separate command and can
never enter prediction freezing or projection.  Formal source-eval scoring is
refused unless an external one-shot ledger receipt binds the frozen
calibrator, prediction cohort, and already-authorized label artifact.

This command never reads EDF/BDF, EDF annotations, spreadsheets, or clinical
narrative.  It does not provide a command that creates a source-eval release
receipt; that authority must remain outside the evaluation process.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.research_soz_selective_calibration_v2 import (  # noqa: E402
    SOURCE_DEV,
    SOURCE_EVAL,
    apply_selective_soz_calibrator,
    build_frozen_eeg_only_prediction_cohort,
    build_tusz_scalp_weak_label_cohort,
    evaluate_frozen_selective_soz_calibrator,
    fit_selective_soz_calibrator,
    validate_frozen_eeg_only_prediction_cohort,
    validate_selective_soz_calibrator,
    validate_selective_soz_evaluation,
    validate_tusz_scalp_weak_label_cohort,
)


FREEZE_REQUEST_SCHEMA_VERSION = "research_soz_selective_freeze_request_v2"
SOURCE_DEV_LABEL_REQUEST_SCHEMA_VERSION = (
    "research_soz_selective_source_dev_weak_label_request_v2"
)
SOURCE_EVAL_RELEASE_RECEIPT_SCHEMA_VERSION = (
    "research_soz_selective_source_eval_one_shot_release_v2"
)
SOURCE_EVAL_RELEASE_SCHEMA_VERSION = (
    "research_soz_selective_source_eval_release_artifact_v2"
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_SUFFIXES = frozenset(
    {".edf", ".bdf", ".csv", ".tsv", ".xls", ".xlsx", ".xlsm", ".ods"}
)


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"JSON contains invalid non-finite constant {value!r}")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_keys(value: object, expected: set[str], context: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ValueError(f"{context} is missing keys: {missing}")
    if unknown:
        raise ValueError(f"{context} contains unknown keys: {unknown}")
    return value


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be an opaque identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _read_json(path: Path, context: str) -> dict[str, Any]:
    if path.suffix.lower() != ".json" or path.suffix.lower() in _FORBIDDEN_SUFFIXES:
        raise ValueError(f"{context} must be a JSON artifact")
    if path.is_symlink():
        raise ValueError(f"{context} must not be a symlink")
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{context} must be a regular file")
    raw = resolved.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from error
    if type(payload) is not dict:
        raise TypeError(f"{context} must contain a JSON object")
    return payload


def _regular_root(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("artifact root must not be a symlink")
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_dir():
        raise ValueError("artifact root must be a regular directory")
    return resolved


def _safe_relative_json(value: object, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a relative JSON path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix.lower() != ".json"
    ):
        raise ValueError(f"{context} must be a safe relative JSON path")
    return path


def _read_under_root(root: Path, relative: PurePosixPath, context: str) -> dict[str, Any]:
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{context} path must not contain symlinks")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{context} escapes the artifact root") from error
    return _read_json(candidate, context)


def _preflight_new_json_output(path: Path) -> Path:
    if path.suffix.lower() != ".json":
        raise ValueError("output must use a .json suffix")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists: {path}")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("output parent must be a regular directory")
    return parent / path.name


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = _preflight_new_json_output(path)
    parent = destination.parent
    raw = _pretty_json_bytes(payload)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        temporary.unlink()
    except BaseException:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise


def freeze_prediction_cohort(
    *,
    request_path: Path,
    artifact_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    request = _strict_keys(
        _read_json(request_path, "freeze request"),
        {"schema_version", "dataset_id", "partition", "records"},
        "freeze request",
    )
    if request["schema_version"] != FREEZE_REQUEST_SCHEMA_VERSION:
        raise ValueError("unexpected freeze request schema")
    _identifier(request["dataset_id"], "freeze request dataset_id")
    records = request["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("freeze request records must be non-empty")
    root = _regular_root(artifact_root)
    inputs: list[dict[str, Any]] = []
    record_keys = {
        "patient_id",
        "recording_id",
        "prediction_relative_path",
        "strength_relative_path",
    }
    for index, raw_record in enumerate(records):
        record = _strict_keys(raw_record, record_keys, f"freeze records[{index}]")
        patient_id = _identifier(record["patient_id"], f"records[{index}].patient_id")
        recording_id = _identifier(
            record["recording_id"], f"records[{index}].recording_id"
        )
        prediction_path = _safe_relative_json(
            record["prediction_relative_path"],
            f"records[{index}].prediction_relative_path",
        )
        strength_path = _safe_relative_json(
            record["strength_relative_path"],
            f"records[{index}].strength_relative_path",
        )
        inputs.append(
            {
                "patient_id": patient_id,
                "recording_id": recording_id,
                "prediction": _read_under_root(
                    root, prediction_path, f"records[{index}] prediction"
                ),
                "strength": _read_under_root(
                    root, strength_path, f"records[{index}] strength"
                ),
            }
        )
    cohort = build_frozen_eeg_only_prediction_cohort(
        inputs,
        dataset_id=request["dataset_id"],
        partition=request["partition"],
    )
    _write_new_json(output_path, cohort)
    return cohort


def materialize_source_dev_weak_labels(
    *,
    prediction_cohort_path: Path,
    request_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    predictions = validate_frozen_eeg_only_prediction_cohort(
        _read_json(prediction_cohort_path, "source-dev prediction cohort")
    )
    if predictions["partition"] != SOURCE_DEV:
        raise ValueError("label materialization command is restricted to source_dev")
    request = _strict_keys(
        _read_json(request_path, "source-dev weak-label request"),
        {"schema_version", "records"},
        "source-dev weak-label request",
    )
    if request["schema_version"] != SOURCE_DEV_LABEL_REQUEST_SCHEMA_VERSION:
        raise ValueError("unexpected source-dev weak-label request schema")
    if not isinstance(request["records"], list) or not request["records"]:
        raise ValueError("source-dev weak-label request records must be non-empty")
    labels = build_tusz_scalp_weak_label_cohort(predictions, request["records"])
    _write_new_json(output_path, labels)
    return labels


def fit_calibrator_release(
    *,
    source_dev_prediction_path: Path,
    source_dev_label_path: Path,
    locked_source_eval_prediction_path: Path,
    output_path: Path,
    stronger_max_patient_macro_risk: float,
    limited_max_patient_macro_risk: float,
    minimum_accepted_patients: int,
) -> dict[str, Any]:
    dev = validate_frozen_eeg_only_prediction_cohort(
        _read_json(source_dev_prediction_path, "source-dev prediction cohort")
    )
    labels = validate_tusz_scalp_weak_label_cohort(
        _read_json(source_dev_label_path, "source-dev weak-label cohort")
    )
    locked_eval = validate_frozen_eeg_only_prediction_cohort(
        _read_json(
            locked_source_eval_prediction_path,
            "locked source-eval prediction cohort",
        )
    )
    calibrator = fit_selective_soz_calibrator(
        dev,
        labels,
        locked_source_eval_predictions=locked_eval,
        stronger_max_patient_macro_risk=stronger_max_patient_macro_risk,
        limited_max_patient_macro_risk=limited_max_patient_macro_risk,
        minimum_accepted_patients=minimum_accepted_patients,
    )
    _write_new_json(output_path, calibrator)
    return calibrator


def project_cohort_release(
    *,
    calibrator_path: Path,
    prediction_cohort_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    calibrator = validate_selective_soz_calibrator(
        _read_json(calibrator_path, "selective calibrator")
    )
    predictions = validate_frozen_eeg_only_prediction_cohort(
        _read_json(prediction_cohort_path, "prediction cohort")
    )
    projection = apply_selective_soz_calibrator(calibrator, predictions)
    _write_new_json(output_path, projection)
    return projection


def _validate_source_eval_release_receipt(
    value: object,
    *,
    calibrator: Mapping[str, Any],
    predictions: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _strict_keys(
        value,
        {
            "schema_version",
            "release_id",
            "calibrator_content_sha256",
            "source_eval_prediction_cohort_sha256",
            "source_eval_label_cohort_sha256",
            "external_ledger_entry_sha256",
            "prior_formal_release_count",
            "one_shot_authorized",
            "model_and_threshold_selection_complete",
            "content_sha256",
        },
        "source-eval release receipt",
    )
    if receipt["schema_version"] != SOURCE_EVAL_RELEASE_RECEIPT_SCHEMA_VERSION:
        raise ValueError("unexpected source-eval release receipt schema")
    _identifier(receipt["release_id"], "source-eval release_id")
    for name in (
        "calibrator_content_sha256",
        "source_eval_prediction_cohort_sha256",
        "source_eval_label_cohort_sha256",
        "external_ledger_entry_sha256",
        "content_sha256",
    ):
        _sha256(receipt[name], f"release receipt {name}")
    hashable = deepcopy(dict(receipt))
    saved_hash = hashable.pop("content_sha256")
    if _content_sha256(hashable) != saved_hash:
        raise ValueError("source-eval release receipt content hash mismatch")
    if receipt["calibrator_content_sha256"] != calibrator["content_sha256"]:
        raise ValueError("release receipt does not bind the supplied calibrator")
    if receipt["source_eval_prediction_cohort_sha256"] != predictions[
        "content_sha256"
    ]:
        raise ValueError("release receipt does not bind the source-eval predictions")
    if predictions["partition"] != SOURCE_EVAL:
        raise ValueError("release receipt can authorize source_eval only")
    if type(receipt["prior_formal_release_count"]) is not int or receipt[
        "prior_formal_release_count"
    ] != 0:
        raise ValueError("source-eval has already been formally released")
    if receipt["one_shot_authorized"] is not True:
        raise ValueError("source-eval one-shot release is not authorized")
    if receipt["model_and_threshold_selection_complete"] is not True:
        raise ValueError("model and threshold selection must finish before release")
    return deepcopy(dict(receipt))


def release_source_eval_evaluation(
    *,
    calibrator_path: Path,
    source_eval_prediction_path: Path,
    source_eval_label_path: Path,
    external_release_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Perform an externally authorized one-shot source-eval release."""

    # Refuse an unusable/reused destination before opening any evaluation label.
    _preflight_new_json_output(output_path)
    calibrator = validate_selective_soz_calibrator(
        _read_json(calibrator_path, "selective calibrator")
    )
    predictions = validate_frozen_eeg_only_prediction_cohort(
        _read_json(source_eval_prediction_path, "source-eval prediction cohort")
    )
    receipt = _validate_source_eval_release_receipt(
        _read_json(external_release_receipt_path, "external release receipt"),
        calibrator=calibrator,
        predictions=predictions,
    )
    # The label artifact is opened only after external authorization has passed.
    labels = validate_tusz_scalp_weak_label_cohort(
        _read_json(source_eval_label_path, "authorized source-eval weak-label cohort")
    )
    if labels["content_sha256"] != receipt["source_eval_label_cohort_sha256"]:
        raise ValueError("release receipt does not bind the source-eval labels")
    evaluation = evaluate_frozen_selective_soz_calibrator(
        calibrator,
        predictions,
        labels,
        evaluation_run_id=receipt["release_id"],
    )
    envelope_without_hash = {
        "schema_version": SOURCE_EVAL_RELEASE_SCHEMA_VERSION,
        "external_release_receipt_content_sha256": receipt["content_sha256"],
        "evaluation": evaluation,
        "release_boundary": {
            "external_one_shot_receipt_required_and_validated": True,
            "source_eval_labels_opened_after_authorization": True,
            "model_or_threshold_selection_performed": False,
            "edf_annotations_used": False,
            "excel_fields_used": False,
            "soft_spread_used": False,
        },
    }
    envelope = validate_source_eval_release_artifact({
        **envelope_without_hash,
        "content_sha256": _content_sha256(envelope_without_hash),
    })
    _write_new_json(output_path, envelope)
    return envelope


def validate_source_eval_release_artifact(value: object) -> dict[str, Any]:
    envelope = _strict_keys(
        value,
        {
            "schema_version",
            "external_release_receipt_content_sha256",
            "evaluation",
            "release_boundary",
            "content_sha256",
        },
        "source-eval release artifact",
    )
    if envelope["schema_version"] != SOURCE_EVAL_RELEASE_SCHEMA_VERSION:
        raise ValueError("unexpected source-eval release artifact schema")
    _sha256(
        envelope["external_release_receipt_content_sha256"],
        "external release receipt content hash",
    )
    _sha256(envelope["content_sha256"], "source-eval release content hash")
    hashable = deepcopy(dict(envelope))
    saved_hash = hashable.pop("content_sha256")
    if _content_sha256(hashable) != saved_hash:
        raise ValueError("source-eval release artifact content hash mismatch")
    validate_selective_soz_evaluation(envelope["evaluation"])
    expected_boundary = {
        "external_one_shot_receipt_required_and_validated": True,
        "source_eval_labels_opened_after_authorization": True,
        "model_or_threshold_selection_performed": False,
        "edf_annotations_used": False,
        "excel_fields_used": False,
        "soft_spread_used": False,
    }
    if envelope["release_boundary"] != expected_boundary:
        raise ValueError("source-eval release boundary is unsafe")
    return deepcopy(dict(envelope))


def _finite_rate_arg(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return number


def _positive_int_arg(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze-predictions", allow_abbrev=False)
    freeze.add_argument("--request", type=Path, required=True)
    freeze.add_argument("--artifact-root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    labels = commands.add_parser("materialize-source-dev-labels", allow_abbrev=False)
    labels.add_argument("--prediction-cohort", type=Path, required=True)
    labels.add_argument("--request", type=Path, required=True)
    labels.add_argument("--output", type=Path, required=True)

    fit = commands.add_parser("fit", allow_abbrev=False)
    fit.add_argument("--source-dev-predictions", type=Path, required=True)
    fit.add_argument("--source-dev-labels", type=Path, required=True)
    fit.add_argument("--locked-source-eval-predictions", type=Path, required=True)
    fit.add_argument("--stronger-max-patient-macro-risk", type=_finite_rate_arg, default=0.20)
    fit.add_argument("--limited-max-patient-macro-risk", type=_finite_rate_arg, default=0.40)
    fit.add_argument("--minimum-accepted-patients", type=_positive_int_arg, default=5)
    fit.add_argument("--output", type=Path, required=True)

    project = commands.add_parser("project", allow_abbrev=False)
    project.add_argument("--calibrator", type=Path, required=True)
    project.add_argument("--prediction-cohort", type=Path, required=True)
    project.add_argument("--output", type=Path, required=True)

    evaluate = commands.add_parser("evaluate-source-eval", allow_abbrev=False)
    evaluate.add_argument("--calibrator", type=Path, required=True)
    evaluate.add_argument("--source-eval-predictions", type=Path, required=True)
    evaluate.add_argument("--source-eval-labels", type=Path, required=True)
    evaluate.add_argument("--external-release-receipt", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "freeze-predictions":
        result = freeze_prediction_cohort(
            request_path=args.request,
            artifact_root=args.artifact_root,
            output_path=args.output,
        )
    elif args.command == "materialize-source-dev-labels":
        result = materialize_source_dev_weak_labels(
            prediction_cohort_path=args.prediction_cohort,
            request_path=args.request,
            output_path=args.output,
        )
    elif args.command == "fit":
        result = fit_calibrator_release(
            source_dev_prediction_path=args.source_dev_predictions,
            source_dev_label_path=args.source_dev_labels,
            locked_source_eval_prediction_path=args.locked_source_eval_predictions,
            output_path=args.output,
            stronger_max_patient_macro_risk=args.stronger_max_patient_macro_risk,
            limited_max_patient_macro_risk=args.limited_max_patient_macro_risk,
            minimum_accepted_patients=args.minimum_accepted_patients,
        )
    elif args.command == "project":
        result = project_cohort_release(
            calibrator_path=args.calibrator,
            prediction_cohort_path=args.prediction_cohort,
            output_path=args.output,
        )
    else:
        result = release_source_eval_evaluation(
            calibrator_path=args.calibrator,
            source_eval_prediction_path=args.source_eval_predictions,
            source_eval_label_path=args.source_eval_labels,
            external_release_receipt_path=args.external_release_receipt,
            output_path=args.output,
        )
    print(
        json.dumps(
            {
                "command": args.command,
                "output": str(args.output.resolve()),
                "schema_version": result["schema_version"],
                "content_sha256": result["content_sha256"],
                "edf_annotations_used": False,
                "excel_fields_used": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
