#!/usr/bin/env python3
"""Materialize non-destructive research SOZ prediction sidecars.

The command reads exactly one allowed source artifact per recording:
``records/<recording_id>/report/bundle.json``.  From that object, the
aggregation API consumes only the existing per-event C18 research-ranking
receipts.  EDF/BDF files, EDF annotations, spreadsheets, doctor labels,
post-freeze evaluation data, and narrative text are never generation inputs.

The source batch is immutable.  Results are atomically published to a new
output root and include one validated Top-k prediction plus one explicitly
uncalibrated descriptive-strength sidecar for every eligible bundle.  Missing
or invalid inputs are retained in the cohort manifest with stable skip codes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.research_soz_evidence import (  # noqa: E402
    DESCRIPTIVE_EVIDENCE_LEVELS,
    RESEARCH_SOZ_EVIDENCE_POLICY_ID,
    classify_research_soz_descriptive_strength,
)
from src.clinical_eeg_long_recording.research_soz_prediction import (  # noqa: E402
    C18_ELECTRODES,
    DEFAULT_JS_THRESHOLD,
    RESEARCH_SOZ_PREDICTION_METHOD_ID,
    aggregate_research_soz_rankings_from_bundle,
    validate_research_soz_prediction_artifact,
)


SCHEMA_VERSION = "private_long_recording_research_soz_sidecar_batch_v1_1"
STATUS = "completed_research_soz_sidecar_batch"
PREDICTION_FILENAME = "research_soz_prediction.json"
STRENGTH_FILENAME = "research_soz_descriptive_strength.json"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_FORBIDDEN_SOURCE_SUFFIXES = frozenset(
    {".edf", ".bdf", ".csv", ".tsv", ".xls", ".xlsx", ".xlsm", ".ods"}
)


class BundleSkipError(ValueError):
    """Expected per-record exclusion with a stable public reason code."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(detail or code)
        self.code = code
        self.detail = detail or code


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise BundleSkipError("bundle_json_duplicate_key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise BundleSkipError("bundle_json_nonfinite_number", value)


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
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _content_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _failure_fingerprint(stage: str, error: BaseException) -> str:
    safe = f"{stage}:{type(error).__name__}:{str(error)}".encode("utf-8")
    return hashlib.sha256(safe).hexdigest()


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise BundleSkipError("invalid_recording_identifier", context)
    return value


def _regular_input_root(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("input root must not be a symlink")
    root = path.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("input root must be a regular directory")
    records = root / "records"
    if records.is_symlink() or not records.is_dir():
        raise ValueError("input root must contain a regular records directory")
    return root


def _ensure_independent_output(input_root: Path, output_root: Path) -> Path:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("output root already exists; source/sidecars are immutable")
    parent = output_root.parent.resolve(strict=True)
    candidate = parent / output_root.name
    try:
        candidate.relative_to(input_root)
    except ValueError:
        return candidate
    raise ValueError("output root must be independent of the input batch root")


def _read_bundle(path: Path) -> tuple[dict[str, Any], str]:
    if path.suffix.lower() in _FORBIDDEN_SOURCE_SUFFIXES:
        raise BundleSkipError("forbidden_source_type")
    if path.is_symlink():
        raise BundleSkipError("bundle_symlink_prohibited")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise BundleSkipError("bundle_not_found") from error
    if resolved.is_symlink() or not resolved.is_file():
        raise BundleSkipError("bundle_not_regular_file")
    raw = resolved.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_invalid_constant,
        )
    except BundleSkipError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleSkipError("bundle_json_invalid") from error
    if type(value) is not dict:
        raise BundleSkipError("bundle_json_not_object")
    return value, _sha256_bytes(raw)


def _preflight_bundle(
    bundle: Mapping[str, Any], expected_recording_id: str
) -> tuple[int, int]:
    recording_id = bundle.get("recording_id")
    if recording_id != expected_recording_id:
        raise BundleSkipError("bundle_recording_id_mismatch")
    events = bundle.get("events")
    if not isinstance(events, list):
        raise BundleSkipError("bundle_events_not_list")
    if not events:
        raise BundleSkipError("no_event_rankings")
    if "event_count" in bundle and bundle["event_count"] != len(events):
        raise BundleSkipError("bundle_event_count_mismatch")
    explicit_weight_count = 0
    for event in events:
        if not isinstance(event, Mapping):
            raise BundleSkipError("bundle_event_not_object")
        receipt = event.get("research_soz_ranking_receipt")
        if not isinstance(receipt, Mapping):
            raise BundleSkipError("event_research_ranking_missing")
        if "evidence_weight" in receipt:
            explicit_weight_count += 1
    return len(events), explicit_weight_count


def _write_json(path: Path, value: object) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace sidecar artifact {path.name}")
    raw = _pretty_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _sha256_bytes(raw)


def _write_json_in_existing_directory(path: Path, value: object) -> str:
    return _write_json(path, value)


def _empty_channel_counts() -> dict[str, int]:
    return {electrode: 0 for electrode in C18_ELECTRODES}


def _record_directories(input_root: Path) -> list[Path]:
    records_root = input_root / "records"
    result: list[Path] = []
    for child in sorted(records_root.iterdir(), key=lambda item: item.name):
        if child.is_symlink():
            raise ValueError("record directory symlinks are prohibited")
        if child.is_dir():
            _identifier(child.name, "record directory name")
            result.append(child)
    if not result:
        raise ValueError("input records directory contains no record directories")
    return result


def _skip_row(
    recording_id: str,
    reason: str,
    *,
    bundle_relative_path: str | None = None,
    bundle_sha256: str | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "recording_id": recording_id,
        "status": "skipped",
        "skip_reason": reason,
        "source_bundle_relative_path": bundle_relative_path,
        "source_bundle_file_sha256": bundle_sha256,
    }
    if error is not None:
        row["validation_error_class"] = type(error).__name__
        row["validation_error_fingerprint"] = _failure_fingerprint(reason, error)
    return row


def _materialize_into(
    input_root: Path,
    staging_root: Path,
    *,
    top_k: int,
    js_threshold: float,
) -> dict[str, Any]:
    record_rows: list[dict[str, Any]] = []
    skip_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    top1_counts = _empty_channel_counts()
    top_k_counts = _empty_channel_counts()
    by_rank_counts = {
        str(rank): _empty_channel_counts() for rank in range(1, top_k + 1)
    }
    bundle_count = 0
    generated_count = 0
    input_event_ranking_count = 0
    explicit_weight_event_count = 0
    default_unit_weight_event_count = 0

    for record_directory in _record_directories(input_root):
        recording_id = record_directory.name
        relative_bundle = PurePosixPath("records") / recording_id / "report" / "bundle.json"
        bundle_path = record_directory / "report" / "bundle.json"
        if not bundle_path.exists() and not bundle_path.is_symlink():
            technical_receipt = record_directory / "technical_failure_receipt.json"
            reason = (
                "technical_unassessable_bundle_absent"
                if technical_receipt.exists() and not technical_receipt.is_symlink()
                else "bundle_absent"
            )
            skip_counts[reason] += 1
            record_rows.append(_skip_row(recording_id, reason))
            continue
        bundle_count += 1
        bundle_sha256: str | None = None
        try:
            bundle, bundle_sha256 = _read_bundle(bundle_path)
            event_count, explicit_weight_count = _preflight_bundle(
                bundle, recording_id
            )
            prediction = aggregate_research_soz_rankings_from_bundle(
                bundle,
                top_k=top_k,
                js_threshold=js_threshold,
            )
            prediction = validate_research_soz_prediction_artifact(prediction)
            strength = classify_research_soz_descriptive_strength(
                prediction,
                recording_id=recording_id,
            )
        except BundleSkipError as error:
            skip_counts[error.code] += 1
            record_rows.append(
                _skip_row(
                    recording_id,
                    error.code,
                    bundle_relative_path=relative_bundle.as_posix(),
                    bundle_sha256=bundle_sha256,
                    error=error,
                )
            )
            continue
        except (TypeError, ValueError, KeyError) as error:
            reason = "event_research_ranking_validation_failed"
            skip_counts[reason] += 1
            record_rows.append(
                _skip_row(
                    recording_id,
                    reason,
                    bundle_relative_path=relative_bundle.as_posix(),
                    bundle_sha256=bundle_sha256,
                    error=error,
                )
            )
            continue

        output_record_directory = staging_root / "records" / recording_id
        prediction_relative = (
            PurePosixPath("records") / recording_id / PREDICTION_FILENAME
        )
        strength_relative = (
            PurePosixPath("records") / recording_id / STRENGTH_FILENAME
        )
        prediction_file_sha256 = _write_json(
            output_record_directory / PREDICTION_FILENAME, prediction
        )
        strength_file_sha256 = _write_json_in_existing_directory(
            output_record_directory / STRENGTH_FILENAME, strength
        )

        ranked = prediction["ranked_hypotheses"]
        if len(ranked) != top_k:
            raise AssertionError("validated prediction did not preserve Top-k")
        for row in ranked:
            rank = int(row["rank"])
            electrode = str(row["electrode"])
            top_k_counts[electrode] += 1
            by_rank_counts[str(rank)][electrode] += 1
        top1_electrode = str(ranked[0]["electrode"])
        top1_counts[top1_electrode] += 1
        evidence_level = str(strength["evidence_level"])
        evidence_counts[evidence_level] += 1
        generated_count += 1
        input_event_ranking_count += event_count
        explicit_weight_event_count += explicit_weight_count
        default_unit_weight_event_count += event_count - explicit_weight_count
        record_rows.append(
            {
                "recording_id": recording_id,
                "status": "completed",
                "source_bundle_relative_path": relative_bundle.as_posix(),
                "source_bundle_file_sha256": bundle_sha256,
                "input_event_ranking_count": event_count,
                "explicit_evidence_weight_event_count": explicit_weight_count,
                "default_unit_weight_event_count": (
                    event_count - explicit_weight_count
                ),
                "prediction_artifact_relative_path": prediction_relative.as_posix(),
                "prediction_artifact_id": prediction["artifact_id"],
                "prediction_content_sha256": prediction["content_sha256"],
                "prediction_file_sha256": prediction_file_sha256,
                "descriptive_strength_relative_path": strength_relative.as_posix(),
                "descriptive_strength_content_sha256": strength["content_sha256"],
                "descriptive_strength_file_sha256": strength_file_sha256,
                "top_k_covered": True,
                "top1_electrode": top1_electrode,
                "ranked_electrodes": [row["electrode"] for row in ranked],
                "evidence_level": evidence_level,
                "deterministic_research_conclusion": strength[
                    "deterministic_research_conclusion"
                ]["text"],
                "llm_input_eligible": strength["llm_projection_receipt"][
                    "llm_input_eligible"
                ],
                "llm_invoked": strength["llm_projection_receipt"]["llm_invoked"],
                "llm_may_add_facts": strength["llm_projection_receipt"][
                    "llm_may_add_facts"
                ],
            }
        )

    input_record_count = len(record_rows)
    skipped_count = input_record_count - generated_count
    if sum(skip_counts.values()) != skipped_count:
        raise AssertionError("skip-reason accounting does not close")
    if sum(evidence_counts.values()) != generated_count:
        raise AssertionError("evidence-level accounting does not close")
    if sum(top1_counts.values()) != generated_count:
        raise AssertionError("Top-1 distribution does not close")
    if sum(top_k_counts.values()) != generated_count * top_k:
        raise AssertionError("Top-k distribution does not close")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "source_batch_root_name": input_root.name,
        "input_record_count": input_record_count,
        "bundle_count": bundle_count,
        "generated_prediction_count": generated_count,
        "top_k_covered_record_count": generated_count,
        "deterministic_research_conclusion_count": generated_count,
        "llm_input_eligible_record_count": generated_count,
        "llm_invoked_record_count": 0,
        "skipped_record_count": skipped_count,
        "input_event_ranking_count": input_event_ranking_count,
        "explicit_evidence_weight_event_count": explicit_weight_event_count,
        "default_unit_weight_event_count": default_unit_weight_event_count,
        "top_k": top_k,
        "js_threshold": js_threshold,
        "prediction_method_id": RESEARCH_SOZ_PREDICTION_METHOD_ID,
        "descriptive_evidence_policy_id": RESEARCH_SOZ_EVIDENCE_POLICY_ID,
        "evidence_level_counts": {
            level: evidence_counts[level] for level in DESCRIPTIVE_EVIDENCE_LEVELS
        },
        "skip_reason_counts": dict(sorted(skip_counts.items())),
        "candidate_channel_distribution": {
            "candidate_space": list(C18_ELECTRODES),
            "top1_record_counts": top1_counts,
            "top_k_occurrence_counts": top_k_counts,
            "rank_position_record_counts": by_rank_counts,
        },
        "records": record_rows,
        "calibration_receipt": {
            "status": "not_attached",
            "receipt": None,
            "intended_source": "patient_disjoint_tusz_source_development_partition",
            "source_evaluation_partition_must_remain_frozen": True,
            "private_cohort_used_to_tune_descriptive_cutpoints": False,
            "clinical_probability_interpretation_permitted": False,
        },
        "scope_receipt": {
            "source_file_content_pattern_opened": (
                "records/<recording_id>/report/bundle.json_only"
            ),
            "bundle_fields_used_for_prediction": (
                "events[*].eeg_event_id_and_research_soz_ranking_receipt_only"
            ),
            "bundle_nonranking_fields_ignored_by_prediction": True,
            "existing_source_batch_modified": False,
            "raw_eeg_used": False,
            "edf_annotations_used": False,
            "excel_fields_used": False,
            "doctor_labels_used": False,
            "postfreeze_evaluation_used": False,
            "free_text_used_for_prediction": False,
            "qwen_service_called": False,
            "llm_may_add_facts": False,
            "cortical_soz_or_epileptogenic_zone_claim_permitted": False,
            "top_k_is_research_scalp_eeg_ranked_hypothesis": True,
        },
    }
    manifest["content_sha256"] = _content_sha256(manifest)
    _write_json(staging_root / "cohort_summary.json", manifest)
    return manifest


def materialize_research_soz_sidecars(
    input_root: Path,
    output_root: Path,
    *,
    top_k: int = 5,
    js_threshold: float = DEFAULT_JS_THRESHOLD,
) -> dict[str, Any]:
    """Publish a new immutable sidecar batch and return its cohort manifest."""

    input_root = _regular_input_root(Path(input_root))
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(
        C18_ELECTRODES
    ):
        raise ValueError("top_k must be an integer from one through C18 size")
    if (
        isinstance(js_threshold, bool)
        or not isinstance(js_threshold, (int, float))
        or not math.isfinite(float(js_threshold))
        or not 0.0 <= float(js_threshold) <= 1.0
    ):
        raise ValueError("js_threshold must be a finite rate from zero to one")
    output_root = _ensure_independent_output(input_root, Path(output_root))
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    try:
        manifest = _materialize_into(
            input_root,
            staging,
            top_k=top_k,
            js_threshold=float(js_threshold),
        )
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Existing private long-recording report batch root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New independent sidecar batch root; it must not already exist",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--js-threshold",
        type=float,
        default=DEFAULT_JS_THRESHOLD,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = materialize_research_soz_sidecars(
        args.input_root,
        args.output_root,
        top_k=args.top_k,
        js_threshold=args.js_threshold,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "input_record_count": manifest["input_record_count"],
                "generated_prediction_count": manifest[
                    "generated_prediction_count"
                ],
                "skipped_record_count": manifest["skipped_record_count"],
                "evidence_level_counts": manifest["evidence_level_counts"],
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
