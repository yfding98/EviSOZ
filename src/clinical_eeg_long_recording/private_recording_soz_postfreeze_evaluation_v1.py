"""Prediction-first record-level evaluation for private long-EEG SOZ ranks.

This module evaluates the already frozen, cross-event record-level scalp EEG
rankings.  Its trust-boundary ordering is deliberate:

1. validate the prediction cohort self-hash and every completed prediction
   sidecar/file/content hash;
2. freeze the exact recording roster in memory; and only then
3. open the post-freeze, PHI-free doctor-label release bundle.

Raw spreadsheets, EDF annotations, EDF signals, report prose and source
clinical text are neither accepted nor opened.  Physician ``significant``
electrodes form the hard positive set.  ``spread`` electrodes are evaluated
as a separate soft endpoint and are never promoted into the hard set.

The available predictions contain Top-5 only.  Consequently the reported
MRR is explicitly truncated MRR@5: a relevant hard electrode not present in
the frozen Top-5 receives zero reciprocal rank.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from statistics import mean
from typing import Any, Mapping, Sequence

from src.clinical_eeg_report.schema import canonicalize_electrode

from .postfreeze_doctor_label_bundle import (
    SCHEMA_VERSION as DOCTOR_BUNDLE_SCHEMA_VERSION,
    validate_postfreeze_doctor_label_bundle,
)
from .research_soz_prediction import (
    C18_ELECTRODES,
    RESEARCH_SOZ_PREDICTION_METHOD_ID,
    validate_research_soz_prediction_artifact,
)


SCHEMA_VERSION = "private_recording_soz_postfreeze_evaluation_v1"
PREDICTION_COHORT_SCHEMA_VERSION = (
    "private_long_recording_research_soz_sidecar_batch_v1_1"
)
STATUS = "completed_prediction_first_postfreeze_evaluation"
MRR_POLICY_ID = "truncated_mrr_at_frozen_top5_missing_or_miss_zero_v1"
REGION_MAPPING_POLICY_ID = "frozen_c18_five_region_compatibility_v1"
LATERALITY_MAPPING_POLICY_ID = "canonical_10_20_electrode_side_v1"

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_RECORDING_ID_RE = re.compile(r"^PRIV-RH[A-F0-9]{16}$")

# This reproduces the frozen research ranker's five-region projection already
# used by postfreeze_evaluation.py.  It is intentionally not replaced by a
# post-hoc clinically nicer mapping after references are seen.
_ELECTRODE_TO_MODEL_REGION: Mapping[str, str] = {
    "FP1": "left_frontal",
    "F3": "left_frontal",
    "F7": "left_frontal",
    "FP2": "right_frontal",
    "F4": "right_frontal",
    "F8": "right_frontal",
    "T7": "left_temporal",
    "P7": "left_temporal",
    "M1": "left_temporal",
    "T8": "right_temporal",
    "P8": "right_temporal",
    "M2": "right_temporal",
    "FZ": "central_parietal",
    "CZ": "central_parietal",
    "PZ": "central_parietal",
    "C3": "central_parietal",
    "C4": "central_parietal",
    "P3": "central_parietal",
    "P4": "central_parietal",
    "O1": "central_parietal",
    "O2": "central_parietal",
}
_MODEL_REGION_TO_CLINICAL_REGIONS: Mapping[str, tuple[str, ...]] = {
    "left_frontal": ("frontal",),
    "right_frontal": ("frontal",),
    "left_temporal": ("temporal",),
    "right_temporal": ("temporal",),
    "central_parietal": ("central", "parietal"),
}
_ELECTRODE_TO_LATERALITY: Mapping[str, str] = {
    "FP1": "left",
    "F3": "left",
    "F7": "left",
    "T7": "left",
    "P7": "left",
    "M1": "left",
    "C3": "left",
    "P3": "left",
    "O1": "left",
    "FP2": "right",
    "F4": "right",
    "F8": "right",
    "T8": "right",
    "P8": "right",
    "M2": "right",
    "C4": "right",
    "P4": "right",
    "O2": "right",
    "FZ": "midline",
    "CZ": "midline",
    "PZ": "midline",
}
_REGION_EXPANSION: Mapping[str, tuple[str, ...]] = {
    "frontotemporal": ("frontal", "temporal"),
    "centrotemporal": ("central", "temporal"),
    "temporoparietal": ("temporal", "parietal"),
    "posterior": ("parietal", "occipital"),
}


class DuplicateKeyError(ValueError):
    """Raised when a supposedly immutable JSON object repeats a key."""


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _canonical_bytes(value: object) -> bytes:
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


def _pretty_bytes(value: object) -> bytes:
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
    return _sha256_bytes(_canonical_bytes(value))


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be lowercase SHA-256")
    return value


def _require_recording_id(value: object, context: str) -> str:
    if not isinstance(value, str) or _RECORDING_ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be an opaque private recording ID")
    return value


def _regular_json_bytes(path: Path, context: str) -> bytes:
    if path.is_symlink():
        raise ValueError(f"{context} must not be a symlink")
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{context} must be a regular file")
    return resolved.read_bytes()


def _decode_json(raw: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _resolved_child(root: Path, relative: object, context: str) -> Path:
    if not isinstance(relative, str):
        raise TypeError(f"{context} must be a relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"{context} escapes the frozen sidecar root")
    candidate = root.joinpath(*pure.parts)
    parent = candidate.parent.resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    try:
        parent.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{context} escapes the frozen sidecar root") from error
    return candidate


def _validated_prediction_snapshot(summary_path: Path) -> dict[str, Any]:
    """Validate all prediction bytes before any reference path is touched."""

    summary_path = Path(summary_path)
    summary_raw = _regular_json_bytes(summary_path, "prediction cohort summary")
    summary = _decode_json(summary_raw, "prediction cohort summary")
    if summary.get("schema_version") != PREDICTION_COHORT_SCHEMA_VERSION:
        raise ValueError("unexpected prediction cohort schema version")
    if summary.get("prediction_method_id") != RESEARCH_SOZ_PREDICTION_METHOD_ID:
        raise ValueError("unexpected frozen prediction method")
    saved_content_hash = _require_sha256(
        summary.get("content_sha256"), "prediction cohort content_sha256"
    )
    hashable = dict(summary)
    hashable.pop("content_sha256")
    if _content_sha256(hashable) != saved_content_hash:
        raise ValueError("prediction cohort content hash mismatch")
    scope = summary.get("scope_receipt")
    if not isinstance(scope, Mapping):
        raise TypeError("prediction cohort scope_receipt must be an object")
    for key in (
        "edf_annotations_used",
        "excel_fields_used",
        "doctor_labels_used",
        "postfreeze_evaluation_used",
        "free_text_used_for_prediction",
    ):
        if scope.get(key) is not False:
            raise ValueError(f"prediction cohort unsafe scope flag: {key}")

    records = summary.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("prediction cohort records must be a non-empty array")
    if summary.get("input_record_count") != len(records):
        raise ValueError("prediction cohort input_record_count mismatch")
    top_k = summary.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k != 5:
        raise ValueError("this endpoint requires the frozen Top-5 cohort")

    root = summary_path.resolve(strict=True).parent
    by_record: dict[str, dict[str, Any]] = {}
    completed_count = 0
    skipped_count = 0
    sidecar_receipts: list[dict[str, str]] = []
    for index, raw_row in enumerate(records):
        if not isinstance(raw_row, Mapping):
            raise TypeError(f"prediction records[{index}] must be an object")
        row = dict(raw_row)
        recording_id = _require_recording_id(
            row.get("recording_id"), f"prediction records[{index}].recording_id"
        )
        if recording_id in by_record:
            raise ValueError("prediction cohort has duplicate recording IDs")
        status = row.get("status")
        if status == "completed":
            completed_count += 1
            relative = row.get("prediction_artifact_relative_path")
            expected_relative = f"records/{recording_id}/research_soz_prediction.json"
            if relative != expected_relative:
                raise ValueError("prediction artifact relative path drifted")
            artifact_path = _resolved_child(
                root, relative, "prediction artifact relative path"
            )
            artifact_raw = _regular_json_bytes(
                artifact_path, f"prediction artifact for {recording_id}"
            )
            artifact_file_hash = _sha256_bytes(artifact_raw)
            if artifact_file_hash != _require_sha256(
                row.get("prediction_file_sha256"),
                f"prediction file hash for {recording_id}",
            ):
                raise ValueError("prediction sidecar file hash mismatch")
            artifact = validate_research_soz_prediction_artifact(
                _decode_json(artifact_raw, f"prediction artifact for {recording_id}")
            )
            if artifact["content_sha256"] != _require_sha256(
                row.get("prediction_content_sha256"),
                f"prediction content hash for {recording_id}",
            ):
                raise ValueError("prediction sidecar content hash mismatch")
            if artifact["artifact_id"] != row.get("prediction_artifact_id"):
                raise ValueError("prediction artifact ID mismatch")
            ranking = [
                canonicalize_electrode(item["electrode"])
                for item in artifact["ranked_hypotheses"]
            ]
            if ranking != row.get("ranked_electrodes"):
                raise ValueError("cohort embedded ranking differs from sidecar")
            if len(ranking) != top_k or len(set(ranking)) != top_k:
                raise ValueError("completed prediction does not contain unique Top-5")
            if row.get("top1_electrode") != ranking[0]:
                raise ValueError("cohort Top-1 differs from sidecar")
            if row.get("top_k_covered") is not True:
                raise ValueError("completed row lost Top-k coverage receipt")
            by_record[recording_id] = {
                "status": "completed",
                "ranking": ranking,
                "skip_reason": None,
            }
            sidecar_receipts.append(
                {
                    "recording_id": recording_id,
                    "relative_path": str(relative),
                    "file_sha256": artifact_file_hash,
                    "content_sha256": artifact["content_sha256"],
                }
            )
        elif status == "skipped":
            skipped_count += 1
            reason = row.get("skip_reason")
            if not isinstance(reason, str) or not reason:
                raise ValueError("skipped prediction row lacks a reason")
            if "ranked_electrodes" in row or "prediction_artifact_relative_path" in row:
                raise ValueError("skipped prediction row unexpectedly contains output")
            by_record[recording_id] = {
                "status": "skipped",
                "ranking": [],
                "skip_reason": reason,
            }
        else:
            raise ValueError("unknown prediction cohort record status")

    if summary.get("generated_prediction_count") != completed_count:
        raise ValueError("prediction generated count mismatch")
    if summary.get("skipped_record_count") != skipped_count:
        raise ValueError("prediction skipped count mismatch")
    if completed_count + skipped_count != len(records):
        raise AssertionError("prediction status accounting did not close")
    embedded_skip_counts = dict(summary.get("skip_reason_counts", {}))
    observed_skip_counts = Counter(
        row["skip_reason"] for row in by_record.values() if row["status"] == "skipped"
    )
    if embedded_skip_counts != dict(sorted(observed_skip_counts.items())):
        raise ValueError("prediction skip reason accounting mismatch")

    roster = sorted(by_record)
    return {
        "records": by_record,
        "roster": roster,
        "roster_sha256": _content_sha256(roster),
        "input_record_count": len(roster),
        "generated_prediction_count": completed_count,
        "skipped_record_count": skipped_count,
        "summary_file_sha256": _sha256_bytes(summary_raw),
        "summary_content_sha256": saved_content_hash,
        "prediction_sidecar_count": completed_count,
        "prediction_sidecar_set_sha256": _content_sha256(sidecar_receipts),
        "prediction_method_id": summary["prediction_method_id"],
        "top_k": top_k,
        "skip_reason_counts": dict(sorted(observed_skip_counts.items())),
    }


def _validated_reference_snapshot(
    doctor_bundle_path: Path,
    expected_roster: Sequence[str],
) -> dict[str, Any]:
    raw = _regular_json_bytes(Path(doctor_bundle_path), "doctor-label bundle")
    bundle = validate_postfreeze_doctor_label_bundle(
        _decode_json(raw, "doctor-label bundle")
    )
    if bundle.get("schema_version") != DOCTOR_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unexpected doctor-label bundle schema version")
    records = bundle.get("records")
    if not isinstance(records, list):
        raise TypeError("doctor-label records must be an array")
    if bundle.get("record_count") != len(records):
        raise ValueError("doctor-label record_count mismatch")
    by_record: dict[str, dict[str, Any]] = {}
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"doctor-label records[{index}] must be an object")
        record = dict(raw_record)
        recording_id = _require_recording_id(
            record.get("recording_id"),
            f"doctor-label records[{index}].recording_id",
        )
        if recording_id in by_record:
            raise ValueError("doctor-label bundle has duplicate recording IDs")
        by_record[recording_id] = record
    observed_roster = sorted(by_record)
    expected = sorted(expected_roster)
    if observed_roster != expected:
        missing = sorted(set(expected) - set(observed_roster))
        extra = sorted(set(observed_roster) - set(expected))
        raise ValueError(
            "prediction/reference roster mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    return {
        "records": by_record,
        "roster": observed_roster,
        "roster_sha256": _content_sha256(observed_roster),
        "file_sha256": _sha256_bytes(raw),
        "content_sha256": _content_sha256(bundle),
        "label_release_id": bundle["label_release_id"],
        "record_count": len(observed_roster),
    }


def _canonical_electrode_set(values: Sequence[object]) -> list[str]:
    result: set[str] = set()
    for value in values:
        electrode = canonicalize_electrode(value)
        result.add(electrode)
    return sorted(result)


def _expand_laterality(value: object) -> set[str]:
    if value == "bilateral":
        return {"left", "right"}
    if value in {"left", "right", "midline"}:
        return {str(value)}
    return set()


def _expand_regions(values: object) -> set[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return set()
    result: set[str] = set()
    for value in values:
        if value in {None, "unknown"}:
            continue
        region = str(value)
        result.update(_REGION_EXPANSION.get(region, (region,)))
    return result


def _reference_for_record(record: Mapping[str, Any]) -> dict[str, Any]:
    raw_labels = record.get("doctor_labels")
    if not isinstance(raw_labels, list):
        raise TypeError("doctor_labels must be an array")
    eligible = [
        dict(label)
        for label in raw_labels
        if isinstance(label, Mapping) and label.get("evaluation_eligible") is True
    ]
    hard: list[object] = []
    spread: list[object] = []
    laterality: set[str] = set()
    regions: set[str] = set()
    for label in eligible:
        physician = label.get("physician_channel_reference")
        if isinstance(physician, Mapping) and physician.get("status") == "available":
            hard.extend(physician.get("significant_electrodes", []))
            spread.extend(physician.get("spread_electrodes", []))
        onset = label.get("onset")
        if isinstance(onset, Mapping) and onset.get("status") == "available":
            laterality.update(_expand_laterality(onset.get("laterality")))
            regions.update(_expand_regions(onset.get("regions")))
    hard_set = _canonical_electrode_set(hard)
    spread_set = _canonical_electrode_set(spread)
    return {
        "patient_pseudonym": record.get("patient_pseudonym"),
        "doctor_label_status": record.get("doctor_label_status"),
        "eligible_doctor_label_count": len(eligible),
        "hard_significant_electrodes": hard_set,
        "soft_spread_electrodes": spread_set,
        "hard_spread_overlap_electrodes": sorted(
            set(hard_set).intersection(spread_set)
        ),
        "laterality_values": sorted(laterality),
        "region_values": sorted(regions),
    }


def _rank_metrics(ranking: Sequence[str], labels: Sequence[str]) -> dict[str, float]:
    relevant = set(labels)
    if not relevant:
        raise ValueError("rank metrics require a non-empty reference")
    top1 = float(bool(ranking[:1] and set(ranking[:1]).intersection(relevant)))
    top3 = float(bool(set(ranking[:3]).intersection(relevant)))
    top5 = float(bool(set(ranking[:5]).intersection(relevant)))
    first = next(
        (
            rank
            for rank, electrode in enumerate(ranking[:5], 1)
            if electrode in relevant
        ),
        None,
    )
    return {
        "top1_accuracy": top1,
        "hit_at_3": top3,
        "hit_at_5": top5,
        "mrr_at_5": 1.0 / first if first is not None else 0.0,
    }


def _semantic_prediction(ranking: Sequence[str]) -> dict[str, list[str]]:
    if not ranking:
        return {"laterality": [], "regions": []}
    electrode = ranking[0]
    laterality = _ELECTRODE_TO_LATERALITY.get(electrode)
    model_region = _ELECTRODE_TO_MODEL_REGION.get(electrode)
    return {
        "laterality": [laterality] if laterality is not None else [],
        "regions": list(_MODEL_REGION_TO_CLINICAL_REGIONS.get(model_region, ())),
    }


def _mean_metric(rows: Sequence[Mapping[str, float]], key: str) -> float | None:
    if not rows:
        return None
    value = sum(float(row[key]) for row in rows) / len(rows)
    if not math.isfinite(value):
        raise ValueError("non-finite aggregate metric")
    return round(value, 12)


def _wilson_95(successes: int, denominator: int) -> dict[str, float] | None:
    """Two-sided Wilson score interval with z=1.959963984540054."""

    if denominator == 0:
        return None
    if not 0 <= successes <= denominator:
        raise ValueError("Wilson successes must be within the denominator")
    z = 1.959963984540054
    n = float(denominator)
    p = successes / n
    denominator_term = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator_term
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator_term
    return {
        "lower": round(max(0.0, centre - radius), 12),
        "upper": round(min(1.0, centre + radius), 12),
        "method": "wilson_score_two_sided_95_percent",
    }


def _patient_macro_rank_metrics(
    rows: Sequence[Mapping[str, Any]], metric_row_key: str
) -> dict[str, Any]:
    by_patient: dict[str, list[Mapping[str, float]]] = {}
    for row in rows:
        patient = row.get("patient_pseudonym")
        if not isinstance(patient, str) or not patient:
            raise ValueError("patient pseudonym missing from an applicable record")
        by_patient.setdefault(patient, []).append(row[metric_row_key])
    keys = ("top1_accuracy", "hit_at_3", "hit_at_5", "mrr_at_5")
    return {
        "patient_denominator": len(by_patient),
        "record_denominator": len(rows),
        **{
            key: (
                round(
                    mean(
                        mean(float(metric[key]) for metric in metrics)
                        for metrics in by_patient.values()
                    ),
                    12,
                )
                if by_patient
                else None
            )
            for key in keys
        },
    }


def _patient_macro_semantic(
    rows: Sequence[Mapping[str, Any]], reference_key: str, prediction_key: str
) -> dict[str, Any]:
    by_patient: dict[str, list[float]] = {}
    for row in rows:
        patient = row.get("patient_pseudonym")
        if not isinstance(patient, str) or not patient:
            raise ValueError("patient pseudonym missing from an applicable record")
        compatible = float(
            bool(set(row[reference_key]).intersection(row[prediction_key]))
        )
        by_patient.setdefault(patient, []).append(compatible)
    return {
        "patient_denominator": len(by_patient),
        "record_denominator": len(rows),
        "compatible_accuracy": (
            round(mean(mean(values) for values in by_patient.values()), 12)
            if by_patient
            else None
        ),
    }


def _rank_metric_summary(
    rows: Sequence[Mapping[str, Any]],
    label_key: str,
) -> dict[str, Any]:
    applicable = [row for row in rows if row[label_key]]
    metric_row_key = f"{label_key}_rank_metrics"
    conditional_rows = [
        row for row in applicable if row["prediction_status"] == "completed"
    ]

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        values = [row[metric_row_key] for row in selected]
        top1_count = int(sum(item["top1_accuracy"] for item in values))
        hit3_count = int(sum(item["hit_at_3"] for item in values))
        hit5_count = int(sum(item["hit_at_5"] for item in values))
        return {
            "denominator": len(values),
            "top1_correct_count": top1_count,
            "top1_accuracy": _mean_metric(values, "top1_accuracy"),
            "top1_wilson_95_ci": _wilson_95(top1_count, len(values)),
            "hit_at_3_correct_count": hit3_count,
            "hit_at_3": _mean_metric(values, "hit_at_3"),
            "hit_at_3_wilson_95_ci": _wilson_95(hit3_count, len(values)),
            "hit_at_5_correct_count": hit5_count,
            "hit_at_5": _mean_metric(values, "hit_at_5"),
            "hit_at_5_wilson_95_ci": _wilson_95(hit5_count, len(values)),
            "mrr_at_5": _mean_metric(values, "mrr_at_5"),
            "patient_macro": _patient_macro_rank_metrics(selected, metric_row_key),
        }

    return {
        "applicable_gt_record_count": len(applicable),
        "not_applicable_gt_record_count": len(rows) - len(applicable),
        "forced_full_gt_coverage": summarize(applicable),
        "conditional_on_prediction": summarize(conditional_rows),
        "missing_prediction_in_applicable_gt_count": sum(
            row["prediction_status"] != "completed" for row in applicable
        ),
        "mrr_policy_id": MRR_POLICY_ID,
    }


def _semantic_summary(
    rows: Sequence[Mapping[str, Any]],
    reference_key: str,
    prediction_key: str,
) -> dict[str, Any]:
    applicable = [row for row in rows if row[reference_key]]

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        compatible = sum(
            bool(set(row[reference_key]).intersection(row[prediction_key]))
            for row in selected
        )
        return {
            "denominator": len(selected),
            "compatible_count": int(compatible),
            "compatible_accuracy": (
                round(compatible / len(selected), 12) if selected else None
            ),
            "compatible_wilson_95_ci": _wilson_95(int(compatible), len(selected)),
            "patient_macro": _patient_macro_semantic(
                selected, reference_key, prediction_key
            ),
        }

    return {
        "applicable_gt_record_count": len(applicable),
        "not_applicable_gt_record_count": len(rows) - len(applicable),
        "forced_full_gt_coverage": summarize(applicable),
        "conditional_on_prediction": summarize(
            [row for row in applicable if row["prediction_status"] == "completed"]
        ),
        "missing_prediction_in_applicable_gt_count": sum(
            row["prediction_status"] != "completed" for row in applicable
        ),
    }


def _uniform_without_replacement_hit_probability(
    positive_count: int, k: int, *, candidate_count: int
) -> float:
    if not 0 <= positive_count <= candidate_count:
        raise ValueError("uniform baseline positive count is out of range")
    if positive_count == 0:
        return 0.0
    if candidate_count - positive_count < k:
        return 1.0
    return 1.0 - (
        math.comb(candidate_count - positive_count, k) / math.comb(candidate_count, k)
    )


def _uniform_c18_baseline(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Analytic baseline fixed before labels: uniform random C18 permutation."""

    applicable = [row for row in rows if row["hard_significant_electrodes"]]
    n_candidates = len(C18_ELECTRODES)
    per_record: list[dict[str, float]] = []
    by_patient: dict[str, list[dict[str, float]]] = {}
    for row in applicable:
        positive_count = len(row["hard_gt_in_candidate_space"])
        metric = {
            "top1_accuracy": positive_count / n_candidates,
            "hit_at_3": _uniform_without_replacement_hit_probability(
                positive_count, 3, candidate_count=n_candidates
            ),
            "hit_at_5": _uniform_without_replacement_hit_probability(
                positive_count, 5, candidate_count=n_candidates
            ),
        }
        per_record.append(metric)
        by_patient.setdefault(row["patient_pseudonym"], []).append(metric)
    return {
        "baseline_id": "analytic_uniform_random_c18_permutation_v1",
        "uses_test_labels_to_select_a_fixed_electrode": False,
        "candidate_count": n_candidates,
        "record_denominator": len(per_record),
        "record_micro_expected": {
            key: round(mean(item[key] for item in per_record), 12)
            if per_record
            else None
            for key in ("top1_accuracy", "hit_at_3", "hit_at_5")
        },
        "patient_macro_expected": {
            "patient_denominator": len(by_patient),
            **{
                key: (
                    round(
                        mean(
                            mean(item[key] for item in values)
                            for values in by_patient.values()
                        ),
                        12,
                    )
                    if by_patient
                    else None
                )
                for key in ("top1_accuracy", "hit_at_3", "hit_at_5")
            },
        },
        "interpretation": (
            "analytic expectation under a uniformly random full C18 ranking; "
            "not a test-derived best-electrode baseline"
        ),
    }


def evaluate_validated_snapshots(
    prediction_snapshot: Mapping[str, Any],
    reference_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate two validated snapshots without changing either artifact."""

    prediction_roster = list(prediction_snapshot["roster"])
    reference_roster = list(reference_snapshot["roster"])
    if prediction_roster != reference_roster:
        raise ValueError("validated prediction/reference rosters are not identical")
    prediction_records = prediction_snapshot["records"]
    reference_records = reference_snapshot["records"]
    rows: list[dict[str, Any]] = []
    for recording_id in prediction_roster:
        prediction = prediction_records[recording_id]
        reference = _reference_for_record(reference_records[recording_id])
        ranking = list(prediction["ranking"])
        semantic = _semantic_prediction(ranking)
        hard = reference["hard_significant_electrodes"]
        spread = reference["soft_spread_electrodes"]
        rows.append(
            {
                "recording_id": recording_id,
                "prediction_status": prediction["status"],
                "prediction_skip_reason": prediction["skip_reason"],
                "ranked_electrodes": ranking,
                **reference,
                "hard_gt_in_candidate_space": sorted(
                    set(hard).intersection(C18_ELECTRODES)
                ),
                "hard_gt_outside_candidate_space": sorted(
                    set(hard) - set(C18_ELECTRODES)
                ),
                "predicted_laterality_values": semantic["laterality"],
                "predicted_region_values": semantic["regions"],
                "hard_significant_electrodes_rank_metrics": (
                    _rank_metrics(ranking, hard) if hard else None
                ),
                "soft_spread_electrodes_rank_metrics": (
                    _rank_metrics(ranking, spread) if spread else None
                ),
            }
        )

    label_status_counts = Counter(row["doctor_label_status"] for row in rows)
    output_status_counts = Counter(row["prediction_status"] for row in rows)
    hard_summary = _rank_metric_summary(rows, "hard_significant_electrodes")
    spread_summary = _rank_metric_summary(rows, "soft_spread_electrodes")
    laterality_summary = _semantic_summary(
        rows, "laterality_values", "predicted_laterality_values"
    )
    region_summary = _semantic_summary(rows, "region_values", "predicted_region_values")
    hard_applicable = [row for row in rows if row["hard_significant_electrodes"]]
    hard_impossible = [
        row for row in hard_applicable if not row["hard_gt_in_candidate_space"]
    ]
    zero_output = [row for row in rows if row["prediction_status"] != "completed"]
    frozen_top1_distribution = Counter(
        row["ranked_electrodes"][0]
        for row in rows
        if row["prediction_status"] == "completed"
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "evaluation_unit": "one_unique_long_eeg_recording_cross_event_aggregate",
        "prediction_interpretation": (
            "research_scalp_visible_ictal_onset_topography_channel_ranking"
        ),
        "input_receipts": {
            "prediction_cohort": {
                key: deepcopy(prediction_snapshot[key])
                for key in (
                    "summary_file_sha256",
                    "summary_content_sha256",
                    "prediction_sidecar_count",
                    "prediction_sidecar_set_sha256",
                    "prediction_method_id",
                    "top_k",
                )
            },
            "doctor_label_release": {
                key: deepcopy(reference_snapshot[key])
                for key in ("file_sha256", "content_sha256", "label_release_id")
            },
        },
        "prediction_first_gate": {
            "logical_stage_order": [
                "validate_prediction_cohort_and_all_prediction_sidecar_hashes",
                "freeze_exact_prediction_roster",
                "open_and_validate_postfreeze_doctor_label_bundle",
                "verify_exact_roster_equality",
                "score_without_tuning",
            ],
            "prediction_validated_before_first_reference_byte_opened": True,
            "every_completed_prediction_sidecar_hash_validated": True,
            "threshold_or_ranking_changed_after_reference_open": False,
            "raw_excel_opened": False,
            "edf_annotation_opened": False,
        },
        "roster_validation": {
            "prediction_record_count": len(prediction_roster),
            "reference_record_count": len(reference_roster),
            "exact_set_equal": True,
            "prediction_roster_sha256": prediction_snapshot["roster_sha256"],
            "reference_roster_sha256": reference_snapshot["roster_sha256"],
        },
        "coverage": {
            "full_cohort_record_count": len(rows),
            "generated_prediction_count": output_status_counts["completed"],
            "zero_output_count": len(zero_output),
            "prediction_coverage_rate": round(
                output_status_counts["completed"] / len(rows), 12
            ),
            "prediction_status_counts": dict(sorted(output_status_counts.items())),
            "prediction_skip_reason_counts": dict(
                sorted(
                    Counter(
                        row["prediction_skip_reason"]
                        for row in zero_output
                        if row["prediction_skip_reason"] is not None
                    ).items()
                )
            ),
            "doctor_label_status_counts": dict(sorted(label_status_counts.items())),
            "label_missing_count": label_status_counts["not_available"],
            "source_conflict_count": label_status_counts["source_conflict"],
            "ambiguous_mapping_count": label_status_counts["ambiguous_mapping"],
            "available_label_record_count": label_status_counts["available"],
            "eligible_label_but_hard_gt_empty_record_count": sum(
                row["eligible_doctor_label_count"] > 0
                and not row["hard_significant_electrodes"]
                for row in rows
            ),
            "hard_gt_record_count": len(hard_applicable),
            "hard_gt_entirely_outside_c18_count": len(hard_impossible),
            "hard_gt_with_any_outside_c18_count": sum(
                bool(row["hard_gt_outside_candidate_space"]) for row in hard_applicable
            ),
            "hard_spread_overlap_record_count": sum(
                bool(row["hard_spread_overlap_electrodes"]) for row in rows
            ),
        },
        "metrics": {
            "hard_significant_electrodes": hard_summary,
            "soft_spread_electrodes_separate_endpoint": spread_summary,
            "laterality_compatible": laterality_summary,
            "region_compatible": region_summary,
        },
        "baselines_and_prediction_bias_diagnostics": {
            "uniform_c18_analytic_baseline": _uniform_c18_baseline(rows),
            "frozen_prediction_top1_distribution": dict(
                sorted(frozen_top1_distribution.items())
            ),
            "test_derived_most_frequent_gt_electrode_baseline_reported": False,
            "reason": (
                "choosing a fixed electrode from this evaluation release would "
                "tune on the test labels"
            ),
        },
        "mapping_rules": {
            "laterality_policy_id": LATERALITY_MAPPING_POLICY_ID,
            "laterality": {
                "left": sorted(
                    key
                    for key, value in _ELECTRODE_TO_LATERALITY.items()
                    if value == "left"
                ),
                "right": sorted(
                    key
                    for key, value in _ELECTRODE_TO_LATERALITY.items()
                    if value == "right"
                ),
                "midline": sorted(
                    key
                    for key, value in _ELECTRODE_TO_LATERALITY.items()
                    if value == "midline"
                ),
                "doctor_bilateral_expands_to": ["left", "right"],
                "compatibility_rule": "nonempty_set_intersection",
            },
            "region_policy_id": REGION_MAPPING_POLICY_ID,
            "electrode_to_model_region": dict(_ELECTRODE_TO_MODEL_REGION),
            "model_region_to_clinical_regions": {
                key: list(value)
                for key, value in _MODEL_REGION_TO_CLINICAL_REGIONS.items()
            },
            "doctor_compound_region_expansion": {
                key: list(value) for key, value in _REGION_EXPANSION.items()
            },
            "compatibility_rule": "nonempty_set_intersection",
        },
        "endpoint_rules": {
            "hard_gt": (
                "deduplicated union of significant_electrodes from all "
                "evaluation_eligible doctor_labels in the same recording"
            ),
            "soft_spread_gt": (
                "deduplicated union of spread_electrodes scored separately; "
                "never unioned into hard_gt"
            ),
            "unlisted_electrodes_are_negative": False,
            "hard_gt_empty": "not_applicable",
            "applicable_gt_without_prediction": "score_zero_in_forced_denominator",
            "conditional_endpoint": "applicable_gt_and_completed_prediction_only",
            "mrr": MRR_POLICY_ID,
        },
        "records": rows,
        "claim_boundary": {
            "prediction_or_threshold_tuned_on_this_doctor_release": False,
            "doctor_label_used_for_inference": False,
            "spread_electrodes_promoted_to_hard_gt": False,
            "raw_excel_text_included": False,
            "raw_edf_annotation_included": False,
            "clinical_soz_claim_permitted": False,
        },
    }
    body_hash = _content_sha256(result)
    result["evaluation_id"] = "PRIV-SOZ-EVAL-" + body_hash[:24]
    result["content_sha256"] = _content_sha256(result)
    return result


def evaluate_private_recording_soz_postfreeze(
    *,
    prediction_cohort_path: str | Path,
    doctor_bundle_path: str | Path,
) -> dict[str, Any]:
    """Run the ordered prediction-first evaluation from two frozen inputs."""

    # Do not stat, resolve or open doctor_bundle_path before this completes.
    prediction = _validated_prediction_snapshot(Path(prediction_cohort_path))
    reference = _validated_reference_snapshot(
        Path(doctor_bundle_path), prediction["roster"]
    )
    if prediction["roster_sha256"] != reference["roster_sha256"]:
        raise ValueError("prediction/reference roster hash mismatch")
    return evaluate_validated_snapshots(prediction, reference)


def render_chinese_report(artifact: Mapping[str, Any]) -> str:
    coverage = artifact["coverage"]
    hard = artifact["metrics"]["hard_significant_electrodes"]
    soft = artifact["metrics"]["soft_spread_electrodes_separate_endpoint"]
    laterality = artifact["metrics"]["laterality_compatible"]
    region = artifact["metrics"]["region_compatible"]
    uniform = artifact["baselines_and_prediction_bias_diagnostics"][
        "uniform_c18_analytic_baseline"
    ]
    top1_distribution = artifact["baselines_and_prediction_bias_diagnostics"][
        "frozen_prediction_top1_distribution"
    ]

    def pct(value: object) -> str:
        return "NA" if value is None else f"{100.0 * float(value):.2f}%"

    def metric_row(title: str, value: Mapping[str, Any]) -> str:
        forced = value["forced_full_gt_coverage"]
        conditional = value["conditional_on_prediction"]
        return (
            f"| {title} | forced | {forced['denominator']} | "
            f"{pct(forced['top1_accuracy'])} | {pct(forced['hit_at_3'])} | "
            f"{pct(forced['hit_at_5'])} | "
            f"{forced['mrr_at_5'] if forced['mrr_at_5'] is not None else 'NA'} |\n"
            f"| {title} | conditional | {conditional['denominator']} | "
            f"{pct(conditional['top1_accuracy'])} | "
            f"{pct(conditional['hit_at_3'])} | "
            f"{pct(conditional['hit_at_5'])} | "
            f"{conditional['mrr_at_5'] if conditional['mrr_at_5'] is not None else 'NA'} |"
        )

    laterality_forced = laterality["forced_full_gt_coverage"]
    laterality_cond = laterality["conditional_on_prediction"]
    region_forced = region["forced_full_gt_coverage"]
    region_cond = region["conditional_on_prediction"]
    lines = [
        "# 私有长程 EEG 记录级冻结 SOZ 排名评价 v1",
        "",
        "## 结论",
        "",
        (
            f"141 条冻结记录名册完全一致；生成记录级 Top-5 的记录为 "
            f"{coverage['generated_prediction_count']} 条，零输出 "
            f"{coverage['zero_output_count']} 条，覆盖率 "
            f"{pct(coverage['prediction_coverage_rate'])}。"
        ),
        "",
        ("以下 SOZ 仅指研究性的头皮 EEG 可见发作起始拓扑/通道排序，" "不等同于皮层 SOZ 或致痫区临床结论。"),
        "",
        "## 通道排名指标",
        "",
        "| 端点 | 分母策略 | n | Top-1 | Hit@3 | Hit@5 | MRR@5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        metric_row("hard significant", hard),
        metric_row("soft spread（独立端点）", soft),
        "",
        (
            "forced 分母包含所有 hard GT 可用记录；其中无预测的记录按 0 分。"
            "conditional 只包含 hard GT 可用且确有冻结预测的记录。"
            "现有预测只保存 Top-5，因此 MRR 是 MRR@5，Top-5 外统一为 0。"
        ),
        "",
        (
            "hard forced 的记录级 Top-1 Wilson 95% CI："
            f"{json.dumps(hard['forced_full_gt_coverage']['top1_wilson_95_ci'], ensure_ascii=False, sort_keys=True)}；"
            "患者等权 macro Top-1："
            f"{pct(hard['forced_full_gt_coverage']['patient_macro']['top1_accuracy'])} "
            f"(patients={hard['forced_full_gt_coverage']['patient_macro']['patient_denominator']})。"
        ),
        "",
        "## 侧别与区域兼容准确度",
        "",
        "| 端点 | forced n | forced compatible | conditional n | conditional compatible |",
        "|---|---:|---:|---:|---:|",
        (
            f"| 侧别 | {laterality_forced['denominator']} | "
            f"{pct(laterality_forced['compatible_accuracy'])} | "
            f"{laterality_cond['denominator']} | "
            f"{pct(laterality_cond['compatible_accuracy'])} |"
        ),
        (
            f"| 区域 | {region_forced['denominator']} | "
            f"{pct(region_forced['compatible_accuracy'])} | "
            f"{region_cond['denominator']} | "
            f"{pct(region_cond['compatible_accuracy'])} |"
        ),
        "",
        (
            "侧别 forced Wilson 95% CI："
            f"{json.dumps(laterality_forced['compatible_wilson_95_ci'], ensure_ascii=False, sort_keys=True)}；"
            "区域 forced Wilson 95% CI："
            f"{json.dumps(region_forced['compatible_wilson_95_ci'], ensure_ascii=False, sort_keys=True)}。"
        ),
        "",
        "兼容规则固定为预测集合与医生结构化 GT 集合非空交集：",
        "",
        "- 侧别：10–20 奇数为左、偶数为右，FZ/CZ/PZ 为中线；医生 bilateral 展开为 left+right。",
        "- 区域：沿用冻结 C18 排名器既有五区域投影；额叶电极→frontal，T7/P7→left_temporal，T8/P8→right_temporal，C/P/O 与中线组→central+parietal。",
        "- 医生复合区域：frontotemporal、centrotemporal、temporoparietal、posterior 分别展开后求交；unknown 不进入分母。",
        "",
        "## 覆盖与失败分解",
        "",
        f"- 医生标签 available：{coverage['available_label_record_count']}；缺失：{coverage['label_missing_count']}；source conflict：{coverage['source_conflict_count']}；ambiguous mapping：{coverage['ambiguous_mapping_count']}。",
        f"- hard GT 非空：{coverage['hard_gt_record_count']}；有合格标签但 hard GT 为空：{coverage['eligible_label_but_hard_gt_empty_record_count']}（不适用，不计 0）。",
        f"- hard GT 全部位于 C18 候选空间外：{coverage['hard_gt_entirely_outside_c18_count']}；至少含一个候选空间外电极：{coverage['hard_gt_with_any_outside_c18_count']}。",
        f"- hard 与 spread 标签重叠的记录：{coverage['hard_spread_overlap_record_count']}；spread 仍只在独立软端点评价，未并入 hard。",
        f"- 零输出原因：{json.dumps(coverage['prediction_skip_reason_counts'], ensure_ascii=False, sort_keys=True)}。",
        "",
        "## 不使用测试标签调参的先验基线与偏置诊断",
        "",
        (
            "C18 均匀随机完整排序的解析期望（不是从本测试集挑出的固定通道）："
            f"Top-1={pct(uniform['record_micro_expected']['top1_accuracy'])}，"
            f"Hit@3={pct(uniform['record_micro_expected']['hit_at_3'])}，"
            f"Hit@5={pct(uniform['record_micro_expected']['hit_at_5'])}。"
        ),
        (
            "冻结模型 Top-1 输出分布仅作偏置诊断，不称为 baseline："
            f"{json.dumps(top1_distribution, ensure_ascii=False, sort_keys=True)}。"
        ),
        "",
        "## 防泄漏与可复核性",
        "",
        "评估器先校验 cohort 自哈希以及每个已完成 prediction sidecar 的文件哈希、内容哈希和嵌入排名，再打开医生标签 bundle；没有读取原始 Excel、EDF annotation、EDF 信号或临床自由文本，也没有在开 GT 后改变阈值或排序。",
        "",
        f"- evaluation_id：`{artifact['evaluation_id']}`",
        f"- evaluation content SHA-256：`{artifact['content_sha256']}`",
        f"- prediction cohort file SHA-256：`{artifact['input_receipts']['prediction_cohort']['summary_file_sha256']}`",
        f"- doctor release file SHA-256：`{artifact['input_receipts']['doctor_label_release']['file_sha256']}`",
        "",
    ]
    return "\n".join(lines)


def write_append_only(path: str | Path, raw: bytes) -> str:
    """Write one immutable artifact using O_EXCL and fsync."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise FileExistsError(f"append-only output exists: {destination}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Never unlink a possibly visible append-only artifact here.
        raise
    return _sha256_bytes(raw)


def write_evaluation_artifacts_append_only(
    artifact: Mapping[str, Any],
    *,
    output_json: str | Path,
    output_report: str | Path,
) -> dict[str, str]:
    """Write JSON and Chinese report without replacing existing artifacts."""

    json_path = Path(output_json)
    report_path = Path(output_report)
    for path in (json_path, report_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"append-only output exists: {path}")
    json_hash = write_append_only(json_path, _pretty_bytes(artifact))
    report_hash = write_append_only(
        report_path, render_chinese_report(artifact).encode("utf-8")
    )
    return {
        "json_file_sha256": json_hash,
        "report_file_sha256": report_hash,
    }


__all__ = [
    "LATERALITY_MAPPING_POLICY_ID",
    "MRR_POLICY_ID",
    "REGION_MAPPING_POLICY_ID",
    "SCHEMA_VERSION",
    "STATUS",
    "evaluate_private_recording_soz_postfreeze",
    "evaluate_validated_snapshots",
    "render_chinese_report",
    "write_evaluation_artifacts_append_only",
]
