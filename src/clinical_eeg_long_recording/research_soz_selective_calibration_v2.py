"""Patient-disjoint selective calibration for research scalp-SOZ rankings.

This module is deliberately downstream of the frozen EEG-only artifacts from
``research_soz_prediction`` and ``research_soz_evidence``.  It keeps a Top-k
ranking at full coverage and calibrates only the *wording tier* attached to
that ranking.  The label-free evidence ordering score is not a probability.

DeepSOZ/TUSZ labels enter through a separate, post-freeze API.  They are
treated as patient/record-level clinical scalp-electrode weak labels, may
contain more than one hard-positive electrode, and are used only to select
thresholds on ``source_dev`` or to score a once-frozen ``source_eval`` cohort.
Soft spread electrodes, EDF annotations, spreadsheets, doctor narrative, and
free text are not accepted by any prediction-side schema.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from statistics import mean
from typing import Any, Mapping, Sequence

from .research_soz_evidence import (
    LIMITED_CROSS_EVENT_CONSISTENCY,
    MULTIMODAL_OR_WEAK_RANKED_HYPOTHESES,
    STABLE_LEADING_CANDIDATE,
    validate_research_soz_descriptive_strength,
)
from .research_soz_prediction import (
    C18_ELECTRODES,
    validate_research_soz_prediction_artifact,
)


FROZEN_EEG_ONLY_COHORT_SCHEMA_VERSION = (
    "clinical_eeg_research_soz_frozen_eeg_only_cohort_v2"
)
TUSZ_WEAK_LABEL_COHORT_SCHEMA_VERSION = (
    "clinical_eeg_research_soz_tusz_scalp_weak_labels_v2"
)
SELECTIVE_CALIBRATOR_SCHEMA_VERSION = (
    "clinical_eeg_research_soz_selective_calibrator_v2"
)
SELECTIVE_PROJECTION_SCHEMA_VERSION = (
    "clinical_eeg_research_soz_selective_projection_v2"
)
SELECTIVE_EVALUATION_SCHEMA_VERSION = (
    "clinical_eeg_research_soz_selective_evaluation_v2"
)

EVIDENCE_ORDER_SCORE_METHOD_ID = "eeg_only_descriptive_order_score_v2"
SELECTIVE_CALIBRATION_POLICY_ID = (
    "patient_macro_top1_risk_max_coverage_thresholds_v2"
)
DEEPSOZ_TUSZ_DATASET_ID = "deepsoz_tusz"
DEEPSOZ_GT_SEMANTICS = (
    "clinical_scalp_electrode_weak_label_not_event_level_cortical_soz"
)

SOURCE_DEV = "source_dev"
SOURCE_EVAL = "source_eval"
DEPLOYMENT = "deployment_eeg_only"
ALLOWED_PARTITIONS = frozenset({SOURCE_DEV, SOURCE_EVAL, DEPLOYMENT})

STRONGER_EVIDENCE = "stronger_evidence_research_candidate"
LIMITED_EVIDENCE = "limited_evidence_research_candidate"
WEAK_EVIDENCE = "weak_or_inconsistent_research_candidate"
CALIBRATED_EVIDENCE_LEVELS = (
    STRONGER_EVIDENCE,
    LIMITED_EVIDENCE,
    WEAK_EVIDENCE,
)

_C18_INDEX = {electrode: index for index, electrode in enumerate(C18_ELECTRODES)}
_SHA256_HEX = frozenset("0123456789abcdef")
_IDENTIFIER_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
)


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


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    *,
    keys: frozenset[str],
    context: str,
) -> None:
    missing = sorted(keys - set(value))
    unknown = sorted(set(value) - keys)
    if missing:
        raise ValueError(f"{context} is missing required keys: {missing}")
    if unknown:
        raise ValueError(f"{context} contains unknown keys: {unknown}")


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError(f"{context} must be a non-empty identifier <= 128 characters")
    if value[0] not in _IDENTIFIER_CHARACTERS or any(
        character not in _IDENTIFIER_CHARACTERS for character in value
    ):
        raise ValueError(f"{context} must be an opaque identifier, not prose or a path")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 hex digest")
    return value


def _finite_rate(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{context} must be a finite value in [0, 1]")
    return result


def _positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _with_content_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result["content_sha256"] = _content_sha256(result)
    return result


def _validate_content_hash(payload: Mapping[str, Any], context: str) -> None:
    saved = _sha256(payload.get("content_sha256"), f"{context}.content_sha256")
    hashable = deepcopy(dict(payload))
    hashable.pop("content_sha256", None)
    if _content_sha256(hashable) != saved:
        raise ValueError(f"{context} content hash mismatch")


def _patient_token(patient_id: str) -> str:
    return _content_sha256({"namespace": "patient_split_v2", "patient_id": patient_id})


def _canonical_electrodes(
    value: object,
    context: str,
    *,
    minimum_length: int = 1,
) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum_length:
        raise ValueError(f"{context} must be a non-empty electrode list")
    if any(not isinstance(item, str) or item not in _C18_INDEX for item in value):
        raise ValueError(f"{context} must contain canonical C18 electrodes only")
    if len(value) != len(set(value)):
        raise ValueError(f"{context} must not contain duplicate electrodes")
    return list(value)


_LEVEL_VALUE = {
    STABLE_LEADING_CANDIDATE: 1.0,
    LIMITED_CROSS_EVENT_CONSISTENCY: 0.5,
    MULTIMODAL_OR_WEAK_RANKED_HYPOTHESES: 0.0,
}


def _evidence_order_score(features: Mapping[str, Any]) -> float:
    """Return a predeclared label-free ordering score, never a probability."""

    event_support = min(1.0, float(features["input_event_count"]) / 3.0)
    single_mode = 0.0 if bool(features["multimodal"]) else 1.0
    sharpness = (
        (1.0 - float(features["normalized_entropy"]))
        + min(1.0, 4.0 * float(features["top1_margin"]))
    ) / 2.0
    repeatability = mean(
        (
            float(features["top1_support_rate"]),
            float(features["top3_support_rate"]),
            float(features["jensen_shannon_consistency"]),
        )
    )
    structure = mean((single_mode, sharpness, event_support))
    level_value = _LEVEL_VALUE[str(features["descriptive_evidence_level"])]
    score = mean((repeatability, structure, level_value))
    return round(min(1.0, max(0.0, score)), 12)


def build_frozen_eeg_only_prediction_cohort(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str,
    partition: str,
) -> dict[str, Any]:
    """Freeze compact prediction records from validated EEG-only artifacts.

    Each input record accepts exactly ``patient_id``, ``recording_id``,
    ``prediction``, and ``strength``.  Consequently annotation, spreadsheet,
    label, and narrative side channels fail closed as unknown keys.
    """

    dataset_id = _identifier(dataset_id, "dataset_id")
    if partition not in ALLOWED_PARTITIONS:
        raise ValueError(f"partition must be one of {sorted(ALLOWED_PARTITIONS)}")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence")
    if not records:
        raise ValueError("records must not be empty")

    compact: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"records[{index}]")
        _strict_keys(
            record,
            keys=frozenset({"patient_id", "recording_id", "prediction", "strength"}),
            context=f"records[{index}]",
        )
        patient_id = _identifier(record["patient_id"], f"records[{index}].patient_id")
        recording_id = _identifier(
            record["recording_id"], f"records[{index}].recording_id"
        )
        identity = (patient_id, recording_id)
        if identity in identities:
            raise ValueError("prediction cohort contains duplicate patient/recording IDs")
        identities.add(identity)

        prediction = validate_research_soz_prediction_artifact(record["prediction"])
        strength = validate_research_soz_descriptive_strength(record["strength"])
        if int(prediction["top_k"]) < 3:
            raise ValueError("prediction top_k must be at least three for hit@3 evaluation")
        if strength["prediction_artifact_id"] != prediction["artifact_id"] or strength[
            "prediction_content_sha256"
        ] != prediction["content_sha256"]:
            raise ValueError("strength is not bound to the supplied prediction artifact")
        if strength["recording_id"] not in (None, recording_id):
            raise ValueError("strength recording_id does not match the cohort record")

        ranked = [str(row["electrode"]) for row in prediction["ranked_hypotheses"]]
        bound_ranked = strength["deterministic_research_conclusion"]["binding"][
            "ranked_electrodes"
        ]
        if ranked != bound_ranked:
            raise ValueError("strength Top-k does not match the supplied prediction")
        inputs = strength["descriptive_inputs"]
        diagnostics = prediction["aggregate_diagnostics"]
        consistency = prediction["cross_event_consistency"]
        expected = {
            "input_event_count": prediction["input_event_count"],
            "top1_support_rate": diagnostics["top1_support_rate"],
            "top3_support_rate": diagnostics["top3_support_rate"],
            "mode_cluster_count": consistency["mode_cluster_count"],
            "multimodal": consistency["multimodal"],
            "jensen_shannon_consistency": consistency["jensen_shannon_consistency"],
            "normalized_entropy": diagnostics["normalized_entropy"],
            "top1_margin": diagnostics["top1_margin"],
        }
        if any(inputs[key] != value for key, value in expected.items()):
            raise ValueError("strength descriptive inputs disagree with prediction diagnostics")

        features = {
            **expected,
            "descriptive_evidence_level": strength["evidence_level"],
        }
        compact.append(
            {
                "patient_id": patient_id,
                "recording_id": recording_id,
                "prediction_artifact_id": prediction["artifact_id"],
                "prediction_content_sha256": prediction["content_sha256"],
                "strength_content_sha256": strength["content_sha256"],
                "ranked_electrodes": ranked,
                "evidence_features": features,
                "evidence_order_score": _evidence_order_score(features),
            }
        )

    compact.sort(key=lambda row: (row["patient_id"], row["recording_id"]))
    payload = {
        "schema_version": FROZEN_EEG_ONLY_COHORT_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "partition": partition,
        "record_count": len(compact),
        "patient_count": len({row["patient_id"] for row in compact}),
        "evidence_order_score_method_id": EVIDENCE_ORDER_SCORE_METHOD_ID,
        "records": compact,
        "input_boundary": {
            "validated_research_prediction_and_strength_only": True,
            "raw_eeg_used_here": False,
            "ground_truth_used": False,
            "edf_annotations_used": False,
            "excel_fields_used": False,
            "doctor_labels_used": False,
            "free_text_used": False,
            "evidence_order_score_is_probability": False,
        },
    }
    return validate_frozen_eeg_only_prediction_cohort(_with_content_hash(payload))


def validate_frozen_eeg_only_prediction_cohort(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _mapping(payload, "frozen EEG-only prediction cohort")
    keys = frozenset(
        {
            "schema_version",
            "dataset_id",
            "partition",
            "record_count",
            "patient_count",
            "evidence_order_score_method_id",
            "records",
            "input_boundary",
            "content_sha256",
        }
    )
    _strict_keys(payload, keys=keys, context="prediction cohort")
    if payload["schema_version"] != FROZEN_EEG_ONLY_COHORT_SCHEMA_VERSION:
        raise ValueError("unexpected frozen prediction cohort schema")
    _identifier(payload["dataset_id"], "prediction cohort dataset_id")
    if payload["partition"] not in ALLOWED_PARTITIONS:
        raise ValueError("unexpected prediction cohort partition")
    if payload["evidence_order_score_method_id"] != EVIDENCE_ORDER_SCORE_METHOD_ID:
        raise ValueError("unexpected evidence order score method")
    _validate_content_hash(payload, "prediction cohort")

    records = payload["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("prediction cohort records must be non-empty")
    record_count = _positive_integer(
        payload["record_count"], "prediction cohort record_count"
    )
    patient_count = _positive_integer(
        payload["patient_count"], "prediction cohort patient_count"
    )
    if record_count != len(records):
        raise ValueError("prediction cohort record_count mismatch")
    record_keys = frozenset(
        {
            "patient_id",
            "recording_id",
            "prediction_artifact_id",
            "prediction_content_sha256",
            "strength_content_sha256",
            "ranked_electrodes",
            "evidence_features",
            "evidence_order_score",
        }
    )
    feature_keys = frozenset(
        {
            "input_event_count",
            "top1_support_rate",
            "top3_support_rate",
            "mode_cluster_count",
            "multimodal",
            "jensen_shannon_consistency",
            "normalized_entropy",
            "top1_margin",
            "descriptive_evidence_level",
        }
    )
    identities: list[tuple[str, str]] = []
    patients: set[str] = set()
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"prediction cohort records[{index}]")
        _strict_keys(record, keys=record_keys, context=f"prediction records[{index}]")
        patient_id = _identifier(record["patient_id"], f"records[{index}].patient_id")
        recording_id = _identifier(
            record["recording_id"], f"records[{index}].recording_id"
        )
        identities.append((patient_id, recording_id))
        patients.add(patient_id)
        _identifier(record["prediction_artifact_id"], "prediction_artifact_id")
        _sha256(record["prediction_content_sha256"], "prediction_content_sha256")
        _sha256(record["strength_content_sha256"], "strength_content_sha256")
        _canonical_electrodes(
            record["ranked_electrodes"], "ranked_electrodes", minimum_length=3
        )
        features = _mapping(record["evidence_features"], "evidence_features")
        _strict_keys(features, keys=feature_keys, context="evidence_features")
        _positive_integer(features["input_event_count"], "input_event_count")
        _positive_integer(features["mode_cluster_count"], "mode_cluster_count")
        if not isinstance(features["multimodal"], bool):
            raise TypeError("multimodal must be boolean")
        if features["multimodal"] is not (features["mode_cluster_count"] > 1):
            raise ValueError("multimodal does not match mode_cluster_count")
        for name in (
            "top1_support_rate",
            "top3_support_rate",
            "jensen_shannon_consistency",
            "normalized_entropy",
            "top1_margin",
        ):
            _finite_rate(features[name], f"evidence_features.{name}")
        if features["descriptive_evidence_level"] not in _LEVEL_VALUE:
            raise ValueError("unknown descriptive evidence level")
        score = _finite_rate(record["evidence_order_score"], "evidence_order_score")
        if not math.isclose(score, _evidence_order_score(features), abs_tol=1e-12):
            raise ValueError("evidence_order_score is not reproducible from EEG-only inputs")
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise ValueError("prediction cohort records must be unique and canonically sorted")
    if patient_count != len(patients):
        raise ValueError("prediction cohort patient_count mismatch")

    boundary = _mapping(payload["input_boundary"], "prediction cohort input_boundary")
    boundary_keys = frozenset(
        {
            "validated_research_prediction_and_strength_only",
            "raw_eeg_used_here",
            "ground_truth_used",
            "edf_annotations_used",
            "excel_fields_used",
            "doctor_labels_used",
            "free_text_used",
            "evidence_order_score_is_probability",
        }
    )
    _strict_keys(boundary, keys=boundary_keys, context="prediction input_boundary")
    if boundary["validated_research_prediction_and_strength_only"] is not True or any(
        boundary[name] is not False
        for name in boundary_keys - {"validated_research_prediction_and_strength_only"}
    ):
        raise ValueError("prediction cohort admits a prohibited input or probability claim")
    return deepcopy(dict(payload))


def build_tusz_scalp_weak_label_cohort(
    prediction_cohort: Mapping[str, Any],
    labels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind hard-positive DeepSOZ/TUSZ scalp labels to a frozen cohort.

    Label rows accept no soft-spread field.  Multiple hard-positive electrodes
    are supported and a prediction is correct when any hard positive is hit.
    """

    cohort = validate_frozen_eeg_only_prediction_cohort(prediction_cohort)
    if cohort["dataset_id"] != DEEPSOZ_TUSZ_DATASET_ID:
        raise ValueError("TUSZ weak labels require the deepsoz_tusz dataset_id")
    if cohort["partition"] not in {SOURCE_DEV, SOURCE_EVAL}:
        raise ValueError("TUSZ weak labels are restricted to source_dev/source_eval")
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        raise TypeError("labels must be a sequence")

    expected = {
        (row["patient_id"], row["recording_id"]) for row in cohort["records"]
    }
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    label_keys = frozenset({"patient_id", "recording_id", "hard_positive_electrodes"})
    for index, raw_label in enumerate(labels):
        label = _mapping(raw_label, f"labels[{index}]")
        _strict_keys(label, keys=label_keys, context=f"labels[{index}]")
        patient_id = _identifier(label["patient_id"], f"labels[{index}].patient_id")
        recording_id = _identifier(
            label["recording_id"], f"labels[{index}].recording_id"
        )
        identity = (patient_id, recording_id)
        if identity in seen:
            raise ValueError("weak-label cohort contains duplicate identities")
        seen.add(identity)
        electrodes = _canonical_electrodes(
            label["hard_positive_electrodes"],
            f"labels[{index}].hard_positive_electrodes",
        )
        electrodes.sort(key=_C18_INDEX.__getitem__)
        rows.append(
            {
                "patient_id": patient_id,
                "recording_id": recording_id,
                "hard_positive_electrodes": electrodes,
            }
        )
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            f"weak labels must exactly match the frozen cohort; missing={missing}, extra={extra}"
        )
    rows.sort(key=lambda row: (row["patient_id"], row["recording_id"]))
    payload = {
        "schema_version": TUSZ_WEAK_LABEL_COHORT_SCHEMA_VERSION,
        "dataset_id": DEEPSOZ_TUSZ_DATASET_ID,
        "partition": cohort["partition"],
        "prediction_cohort_content_sha256": cohort["content_sha256"],
        "record_count": len(rows),
        "patient_count": len({row["patient_id"] for row in rows}),
        "gt_semantics": DEEPSOZ_GT_SEMANTICS,
        "records": rows,
        "label_boundary": {
            "hard_positive_set_only": True,
            "multiple_hard_positive_electrodes_supported": True,
            "soft_spread_electrodes_used": False,
            "event_level_ground_truth_claimed": False,
            "cortical_soz_ground_truth_claimed": False,
            "used_to_modify_frozen_predictions": False,
        },
    }
    return validate_tusz_scalp_weak_label_cohort(_with_content_hash(payload))


def validate_tusz_scalp_weak_label_cohort(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(payload, "TUSZ weak-label cohort")
    keys = frozenset(
        {
            "schema_version",
            "dataset_id",
            "partition",
            "prediction_cohort_content_sha256",
            "record_count",
            "patient_count",
            "gt_semantics",
            "records",
            "label_boundary",
            "content_sha256",
        }
    )
    _strict_keys(payload, keys=keys, context="weak-label cohort")
    if payload["schema_version"] != TUSZ_WEAK_LABEL_COHORT_SCHEMA_VERSION:
        raise ValueError("unexpected TUSZ weak-label schema")
    if payload["dataset_id"] != DEEPSOZ_TUSZ_DATASET_ID:
        raise ValueError("unexpected weak-label dataset")
    if payload["partition"] not in {SOURCE_DEV, SOURCE_EVAL}:
        raise ValueError("weak labels must be source_dev or source_eval")
    _sha256(
        payload["prediction_cohort_content_sha256"],
        "prediction_cohort_content_sha256",
    )
    if payload["gt_semantics"] != DEEPSOZ_GT_SEMANTICS:
        raise ValueError("weak labels were promoted beyond scalp-electrode semantics")
    _validate_content_hash(payload, "weak-label cohort")
    rows = payload["records"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("weak-label records must be non-empty")
    record_count = _positive_integer(payload["record_count"], "weak-label record_count")
    patient_count = _positive_integer(
        payload["patient_count"], "weak-label patient_count"
    )
    if record_count != len(rows):
        raise ValueError("weak-label record_count mismatch")
    row_keys = frozenset({"patient_id", "recording_id", "hard_positive_electrodes"})
    identities: list[tuple[str, str]] = []
    patients: set[str] = set()
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"weak-label records[{index}]")
        _strict_keys(row, keys=row_keys, context=f"weak-label records[{index}]")
        patient_id = _identifier(row["patient_id"], "weak-label patient_id")
        recording_id = _identifier(row["recording_id"], "weak-label recording_id")
        identities.append((patient_id, recording_id))
        patients.add(patient_id)
        electrodes = _canonical_electrodes(
            row["hard_positive_electrodes"], "hard_positive_electrodes"
        )
        if electrodes != sorted(electrodes, key=_C18_INDEX.__getitem__):
            raise ValueError("hard-positive electrodes must be canonically sorted")
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise ValueError("weak-label rows must be unique and canonically sorted")
    if patient_count != len(patients):
        raise ValueError("weak-label patient_count mismatch")
    boundary = _mapping(payload["label_boundary"], "label_boundary")
    boundary_keys = frozenset(
        {
            "hard_positive_set_only",
            "multiple_hard_positive_electrodes_supported",
            "soft_spread_electrodes_used",
            "event_level_ground_truth_claimed",
            "cortical_soz_ground_truth_claimed",
            "used_to_modify_frozen_predictions",
        }
    )
    _strict_keys(boundary, keys=boundary_keys, context="label_boundary")
    if boundary["hard_positive_set_only"] is not True or boundary[
        "multiple_hard_positive_electrodes_supported"
    ] is not True or any(
        boundary[name] is not False
        for name in (
            "soft_spread_electrodes_used",
            "event_level_ground_truth_claimed",
            "cortical_soz_ground_truth_claimed",
            "used_to_modify_frozen_predictions",
        )
    ):
        raise ValueError("weak-label boundary is unsafe")
    return deepcopy(dict(payload))


def _join_outcomes(
    prediction_cohort: Mapping[str, Any],
    label_cohort: Mapping[str, Any],
) -> list[dict[str, Any]]:
    predictions = validate_frozen_eeg_only_prediction_cohort(prediction_cohort)
    labels = validate_tusz_scalp_weak_label_cohort(label_cohort)
    if labels["prediction_cohort_content_sha256"] != predictions["content_sha256"]:
        raise ValueError("weak-label cohort is not bound to this prediction cohort")
    if labels["partition"] != predictions["partition"]:
        raise ValueError("prediction and label partitions do not match")
    by_identity = {
        (row["patient_id"], row["recording_id"]): row for row in labels["records"]
    }
    outcomes: list[dict[str, Any]] = []
    for prediction in predictions["records"]:
        identity = (prediction["patient_id"], prediction["recording_id"])
        label = by_identity.get(identity)
        if label is None:
            raise ValueError("labels do not cover every frozen prediction")
        positives = set(label["hard_positive_electrodes"])
        first_rank = next(
            (
                rank
                for rank, electrode in enumerate(prediction["ranked_electrodes"], start=1)
                if electrode in positives
            ),
            None,
        )
        outcomes.append(
            {
                "patient_id": prediction["patient_id"],
                "recording_id": prediction["recording_id"],
                "evidence_order_score": prediction["evidence_order_score"],
                "top1_correct": int(first_rank == 1),
                "hit_at_3": int(first_rank is not None and first_rank <= 3),
                "reciprocal_rank": 0.0 if first_rank is None else 1.0 / first_rank,
            }
        )
    return outcomes


def _metric_triplet(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        "top1_accuracy": mean(float(row["top1_correct"]) for row in rows),
        "hit_at_3": mean(float(row["hit_at_3"]) for row in rows),
        "mean_reciprocal_rank": mean(float(row["reciprocal_rank"]) for row in rows),
    }


def _validate_metric_triplet(value: object, context: str) -> None:
    metrics = _mapping(value, context)
    keys = frozenset({"top1_accuracy", "hit_at_3", "mean_reciprocal_rank"})
    _strict_keys(metrics, keys=keys, context=context)
    for name in keys:
        _finite_rate(metrics[name], f"{context}.{name}")


def _validate_full_coverage_metrics(value: object, context: str) -> None:
    metrics = _mapping(value, context)
    keys = frozenset(
        {
            "record_count",
            "patient_count",
            "coverage",
            "record_micro",
            "patient_macro",
            "metric_semantics",
        }
    )
    _strict_keys(metrics, keys=keys, context=context)
    record_count = _positive_integer(metrics["record_count"], f"{context}.record_count")
    patient_count = _positive_integer(
        metrics["patient_count"], f"{context}.patient_count"
    )
    if patient_count > record_count:
        raise ValueError(f"{context}.patient_count exceeds record_count")
    if _finite_rate(metrics["coverage"], f"{context}.coverage") != 1.0:
        raise ValueError(f"{context} must report full coverage")
    _validate_metric_triplet(metrics["record_micro"], f"{context}.record_micro")
    _validate_metric_triplet(metrics["patient_macro"], f"{context}.patient_macro")
    semantics = _mapping(metrics["metric_semantics"], f"{context}.metric_semantics")
    expected_semantics = {
        "correct_if_any_hard_positive_is_hit": True,
        "mrr_unretrieved_hard_positive_value": 0.0,
        "mrr_truncated_at_emitted_top_k": True,
        "soft_spread_used": False,
        "gt_is_scalp_electrode_weak_label": True,
    }
    if semantics != expected_semantics:
        raise ValueError(f"{context} metric semantics are unsafe")


def _validate_selective_curve(
    value: object,
    context: str,
    *,
    expected_record_count: int | None = None,
    expected_patient_count: int | None = None,
) -> None:
    curve = _mapping(value, context)
    keys = frozenset(
        {
            "risk_definition",
            "ordering_score_semantics",
            "tie_policy",
            "integration",
            "aurc",
            "patient_macro_aurc_over_record_coverage",
            "points",
        }
    )
    _strict_keys(curve, keys=keys, context=context)
    if curve["risk_definition"] != "one_minus_top1_any_hard_positive_accuracy":
        raise ValueError(f"{context} has an unexpected risk definition")
    if curve["ordering_score_semantics"] != (
        "label_free_evidence_ordering_not_probability"
    ):
        raise ValueError(f"{context} promotes the ordering score to probability")
    if curve["tie_policy"] != "accept_all_equal_scores_at_inclusive_threshold":
        raise ValueError(f"{context} has an unsafe tie policy")
    if curve["integration"] != "right_step_over_record_coverage":
        raise ValueError(f"{context} has an unexpected AURC integration rule")
    saved_aurc = _finite_rate(curve["aurc"], f"{context}.aurc")
    saved_patient_aurc = _finite_rate(
        curve["patient_macro_aurc_over_record_coverage"],
        f"{context}.patient_macro_aurc_over_record_coverage",
    )
    points = curve["points"]
    if not isinstance(points, list) or not points:
        raise ValueError(f"{context}.points must be non-empty")
    point_keys = frozenset(
        {
            "threshold_inclusive",
            "accepted_record_count",
            "accepted_patient_count",
            "record_coverage",
            "patient_coverage",
            "top1_selective_risk",
            "patient_macro_top1_selective_risk",
        }
    )
    previous_threshold = math.inf
    previous_record_count = 0
    previous_patient_count = 0
    previous_coverage = 0.0
    recomputed_aurc = 0.0
    recomputed_patient_aurc = 0.0
    for index, raw_point in enumerate(points):
        point = _mapping(raw_point, f"{context}.points[{index}]")
        _strict_keys(point, keys=point_keys, context=f"{context}.points[{index}]")
        threshold = _finite_rate(
            point["threshold_inclusive"],
            f"{context}.points[{index}].threshold_inclusive",
        )
        if threshold >= previous_threshold:
            raise ValueError(f"{context} thresholds must be strictly decreasing")
        previous_threshold = threshold
        record_count = _positive_integer(
            point["accepted_record_count"],
            f"{context}.points[{index}].accepted_record_count",
        )
        patient_count = _positive_integer(
            point["accepted_patient_count"],
            f"{context}.points[{index}].accepted_patient_count",
        )
        if record_count <= previous_record_count or patient_count < previous_patient_count:
            raise ValueError(f"{context} accepted sets must grow monotonically")
        previous_record_count = record_count
        previous_patient_count = patient_count
        coverage = _finite_rate(
            point["record_coverage"], f"{context}.points[{index}].record_coverage"
        )
        patient_coverage = _finite_rate(
            point["patient_coverage"], f"{context}.points[{index}].patient_coverage"
        )
        if coverage <= previous_coverage:
            raise ValueError(f"{context} record coverage must be strictly increasing")
        if expected_record_count is not None and not math.isclose(
            coverage, record_count / expected_record_count, abs_tol=1e-12
        ):
            raise ValueError(f"{context} record coverage disagrees with count")
        if expected_patient_count is not None and not math.isclose(
            patient_coverage, patient_count / expected_patient_count, abs_tol=1e-12
        ):
            raise ValueError(f"{context} patient coverage disagrees with count")
        micro_risk = _finite_rate(
            point["top1_selective_risk"],
            f"{context}.points[{index}].top1_selective_risk",
        )
        patient_risk = _finite_rate(
            point["patient_macro_top1_selective_risk"],
            f"{context}.points[{index}].patient_macro_top1_selective_risk",
        )
        delta = coverage - previous_coverage
        recomputed_aurc += delta * micro_risk
        recomputed_patient_aurc += delta * patient_risk
        previous_coverage = coverage
    if expected_record_count is not None and previous_record_count != expected_record_count:
        raise ValueError(f"{context} does not end at the full record count")
    if expected_patient_count is not None and previous_patient_count != expected_patient_count:
        raise ValueError(f"{context} does not end at the full patient count")
    if not math.isclose(previous_coverage, 1.0, abs_tol=1e-12):
        raise ValueError(f"{context} does not reach full coverage")
    if not math.isclose(saved_aurc, recomputed_aurc, abs_tol=1e-12) or not math.isclose(
        saved_patient_aurc, recomputed_patient_aurc, abs_tol=1e-12
    ):
        raise ValueError(f"{context} AURC is not reproducible from its curve")


def full_coverage_topk_metrics(
    prediction_cohort: Mapping[str, Any],
    label_cohort: Mapping[str, Any],
) -> dict[str, Any]:
    """Score every record; no evidence tier is allowed to remove its Top-k."""

    rows = _join_outcomes(prediction_cohort, label_cohort)
    patients = sorted({str(row["patient_id"]) for row in rows})
    patient_metrics = [
        _metric_triplet([row for row in rows if row["patient_id"] == patient_id])
        for patient_id in patients
    ]
    patient_macro = {
        name: mean(float(metrics[name]) for metrics in patient_metrics)
        for name in ("top1_accuracy", "hit_at_3", "mean_reciprocal_rank")
    }
    return {
        "record_count": len(rows),
        "patient_count": len(patients),
        "coverage": 1.0,
        "record_micro": _metric_triplet(rows),
        "patient_macro": patient_macro,
        "metric_semantics": {
            "correct_if_any_hard_positive_is_hit": True,
            "mrr_unretrieved_hard_positive_value": 0.0,
            "mrr_truncated_at_emitted_top_k": True,
            "soft_spread_used": False,
            "gt_is_scalp_electrode_weak_label": True,
        },
    }


def selective_risk_coverage(
    prediction_cohort: Mapping[str, Any],
    label_cohort: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute tie-safe Top-1 selective risk and AURC over all score levels."""

    rows = _join_outcomes(prediction_cohort, label_cohort)
    total_records = len(rows)
    total_patients = len({row["patient_id"] for row in rows})
    thresholds = sorted(
        {float(row["evidence_order_score"]) for row in rows}, reverse=True
    )
    points: list[dict[str, Any]] = []
    previous_coverage = 0.0
    aurc = 0.0
    patient_macro_aurc = 0.0
    for threshold in thresholds:
        accepted = [
            row for row in rows if float(row["evidence_order_score"]) >= threshold
        ]
        accepted_patients = sorted({str(row["patient_id"]) for row in accepted})
        micro_accuracy = mean(float(row["top1_correct"]) for row in accepted)
        per_patient_accuracy = [
            mean(
                float(row["top1_correct"])
                for row in accepted
                if row["patient_id"] == patient_id
            )
            for patient_id in accepted_patients
        ]
        patient_macro_accuracy = mean(per_patient_accuracy)
        coverage = len(accepted) / total_records
        micro_risk = 1.0 - micro_accuracy
        patient_macro_risk = 1.0 - patient_macro_accuracy
        delta = coverage - previous_coverage
        aurc += delta * micro_risk
        patient_macro_aurc += delta * patient_macro_risk
        previous_coverage = coverage
        points.append(
            {
                "threshold_inclusive": threshold,
                "accepted_record_count": len(accepted),
                "accepted_patient_count": len(accepted_patients),
                "record_coverage": coverage,
                "patient_coverage": len(accepted_patients) / total_patients,
                "top1_selective_risk": micro_risk,
                "patient_macro_top1_selective_risk": patient_macro_risk,
            }
        )
    return {
        "risk_definition": "one_minus_top1_any_hard_positive_accuracy",
        "ordering_score_semantics": "label_free_evidence_ordering_not_probability",
        "tie_policy": "accept_all_equal_scores_at_inclusive_threshold",
        "integration": "right_step_over_record_coverage",
        "aurc": aurc,
        "patient_macro_aurc_over_record_coverage": patient_macro_aurc,
        "points": points,
    }


def _select_working_point(
    curve: Mapping[str, Any],
    *,
    maximum_patient_macro_risk: float,
    minimum_accepted_patients: int,
    minimum_coverage: float = 0.0,
) -> dict[str, Any]:
    eligible = [
        point
        for point in curve["points"]
        if point["accepted_patient_count"] >= minimum_accepted_patients
        and point["record_coverage"] >= minimum_coverage
        and point["patient_macro_top1_selective_risk"]
        <= maximum_patient_macro_risk
    ]
    if not eligible:
        return {
            "threshold_inclusive": None,
            "accepted_record_count": 0,
            "accepted_patient_count": 0,
            "record_coverage": 0.0,
            "patient_coverage": 0.0,
            "top1_selective_risk": None,
            "patient_macro_top1_selective_risk": None,
            "target_met": False,
        }
    chosen = max(
        eligible,
        key=lambda point: (
            float(point["record_coverage"]),
            int(point["accepted_patient_count"]),
            -float(point["threshold_inclusive"]),
        ),
    )
    return {**deepcopy(chosen), "target_met": True}


def fit_selective_soz_calibrator(
    source_dev_predictions: Mapping[str, Any],
    source_dev_labels: Mapping[str, Any],
    *,
    locked_source_eval_predictions: Mapping[str, Any],
    stronger_max_patient_macro_risk: float = 0.20,
    limited_max_patient_macro_risk: float = 0.40,
    minimum_accepted_patients: int = 5,
) -> dict[str, Any]:
    """Fit two wording thresholds on source-dev and lock source-eval inputs."""

    dev = validate_frozen_eeg_only_prediction_cohort(source_dev_predictions)
    labels = validate_tusz_scalp_weak_label_cohort(source_dev_labels)
    locked_eval = validate_frozen_eeg_only_prediction_cohort(
        locked_source_eval_predictions
    )
    if dev["dataset_id"] != DEEPSOZ_TUSZ_DATASET_ID or locked_eval[
        "dataset_id"
    ] != DEEPSOZ_TUSZ_DATASET_ID:
        raise ValueError("calibration is restricted to the deepsoz_tusz source cohort")
    if dev["partition"] != SOURCE_DEV or labels["partition"] != SOURCE_DEV:
        raise ValueError("calibrator fitting is restricted to source_dev labels")
    if locked_eval["partition"] != SOURCE_EVAL:
        raise ValueError("locked source-eval predictions must use source_eval partition")
    if labels["prediction_cohort_content_sha256"] != dev["content_sha256"]:
        raise ValueError("source-dev labels are not bound to source-dev predictions")
    dev_patients = {str(row["patient_id"]) for row in dev["records"]}
    eval_patients = {str(row["patient_id"]) for row in locked_eval["records"]}
    overlap = sorted(dev_patients & eval_patients)
    if overlap:
        raise ValueError(f"source-dev/source-eval patient leakage detected: {overlap}")

    stronger_risk = _finite_rate(
        stronger_max_patient_macro_risk, "stronger_max_patient_macro_risk"
    )
    limited_risk = _finite_rate(
        limited_max_patient_macro_risk, "limited_max_patient_macro_risk"
    )
    if stronger_risk > limited_risk:
        raise ValueError("stronger risk target must not exceed limited risk target")
    minimum_patients = _positive_integer(
        minimum_accepted_patients, "minimum_accepted_patients"
    )
    if minimum_patients > int(dev["patient_count"]):
        raise ValueError("minimum_accepted_patients exceeds source-dev patient count")

    full_metrics = full_coverage_topk_metrics(dev, labels)
    curve = selective_risk_coverage(dev, labels)
    stronger = _select_working_point(
        curve,
        maximum_patient_macro_risk=stronger_risk,
        minimum_accepted_patients=minimum_patients,
    )
    limited = _select_working_point(
        curve,
        maximum_patient_macro_risk=limited_risk,
        minimum_accepted_patients=minimum_patients,
        minimum_coverage=float(stronger["record_coverage"]),
    )
    if stronger["threshold_inclusive"] is not None and limited[
        "threshold_inclusive"
    ] is not None and float(limited["threshold_inclusive"]) > float(
        stronger["threshold_inclusive"]
    ):
        raise AssertionError("limited threshold must cover the stronger tier")

    input_receipt = {
        "policy_id": SELECTIVE_CALIBRATION_POLICY_ID,
        "source_dev_prediction_sha256": dev["content_sha256"],
        "source_dev_label_sha256": labels["content_sha256"],
        "locked_source_eval_prediction_sha256": locked_eval["content_sha256"],
        "stronger_max_patient_macro_risk": stronger_risk,
        "limited_max_patient_macro_risk": limited_risk,
        "minimum_accepted_patients": minimum_patients,
    }
    input_hash = _content_sha256(input_receipt)
    payload = {
        "schema_version": SELECTIVE_CALIBRATOR_SCHEMA_VERSION,
        "calibrator_id": f"SOZ-SEL-V2-{input_hash[:20]}",
        "policy_id": SELECTIVE_CALIBRATION_POLICY_ID,
        "evidence_order_score_method_id": EVIDENCE_ORDER_SCORE_METHOD_ID,
        "source_dataset_id": DEEPSOZ_TUSZ_DATASET_ID,
        "source_dev_prediction_cohort_sha256": dev["content_sha256"],
        "source_dev_label_cohort_sha256": labels["content_sha256"],
        "locked_source_eval_prediction_cohort_sha256": locked_eval[
            "content_sha256"
        ],
        "split_receipt": {
            "source_dev_patient_tokens": sorted(_patient_token(p) for p in dev_patients),
            "source_eval_patient_tokens": sorted(_patient_token(p) for p in eval_patients),
            "patient_overlap_count": 0,
            "patient_disjoint": True,
        },
        "risk_targets": {
            "stronger_max_patient_macro_top1_risk": stronger_risk,
            "limited_max_patient_macro_top1_risk": limited_risk,
            "minimum_accepted_patients": minimum_patients,
        },
        "frozen_working_points": {
            STRONGER_EVIDENCE: stronger,
            LIMITED_EVIDENCE: limited,
            WEAK_EVIDENCE: {
                "threshold_inclusive": 0.0,
                "top_k_still_required": True,
                "risk_target_not_claimed": True,
            },
        },
        "source_dev_full_coverage_metrics": full_metrics,
        "source_dev_risk_coverage_summary": {
            "aurc": curve["aurc"],
            "patient_macro_aurc_over_record_coverage": curve[
                "patient_macro_aurc_over_record_coverage"
            ],
            "point_count": len(curve["points"]),
        },
        "label_semantics": {
            "ground_truth": DEEPSOZ_GT_SEMANTICS,
            "multiple_hard_positive_electrodes_supported": True,
            "soft_spread_used": False,
        },
        "protocol_boundary": {
            "predictions_frozen_before_labels_joined": True,
            "thresholds_fitted_on_source_dev_only": True,
            "source_eval_labels_accessed_during_fit": False,
            "source_eval_predictions_hash_locked_before_evaluation": True,
            "source_eval_model_or_threshold_selection_permitted": False,
            "top_k_emitted_at_every_evidence_level": True,
            "evidence_order_score_is_probability": False,
            "clinical_diagnosis_claim_permitted": False,
        },
    }
    return validate_selective_soz_calibrator(_with_content_hash(payload))


def _validate_working_point(value: object, context: str) -> None:
    point = _mapping(value, context)
    keys = frozenset(
        {
            "threshold_inclusive",
            "accepted_record_count",
            "accepted_patient_count",
            "record_coverage",
            "patient_coverage",
            "top1_selective_risk",
            "patient_macro_top1_selective_risk",
            "target_met",
        }
    )
    _strict_keys(point, keys=keys, context=context)
    if not isinstance(point["target_met"], bool):
        raise TypeError(f"{context}.target_met must be boolean")
    threshold = point["threshold_inclusive"]
    if threshold is not None:
        _finite_rate(threshold, f"{context}.threshold_inclusive")
    for name in ("record_coverage", "patient_coverage"):
        _finite_rate(point[name], f"{context}.{name}")
    for name in ("accepted_record_count", "accepted_patient_count"):
        if isinstance(point[name], bool) or not isinstance(point[name], int) or point[name] < 0:
            raise ValueError(f"{context}.{name} must be a nonnegative integer")
    for name in ("top1_selective_risk", "patient_macro_top1_selective_risk"):
        if point[name] is not None:
            _finite_rate(point[name], f"{context}.{name}")
    if point["target_met"] is not (threshold is not None):
        raise ValueError(f"{context} target state does not match threshold")


def validate_selective_soz_calibrator(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(payload, "selective SOZ calibrator")
    keys = frozenset(
        {
            "schema_version",
            "calibrator_id",
            "policy_id",
            "evidence_order_score_method_id",
            "source_dataset_id",
            "source_dev_prediction_cohort_sha256",
            "source_dev_label_cohort_sha256",
            "locked_source_eval_prediction_cohort_sha256",
            "split_receipt",
            "risk_targets",
            "frozen_working_points",
            "source_dev_full_coverage_metrics",
            "source_dev_risk_coverage_summary",
            "label_semantics",
            "protocol_boundary",
            "content_sha256",
        }
    )
    _strict_keys(payload, keys=keys, context="selective calibrator")
    if payload["schema_version"] != SELECTIVE_CALIBRATOR_SCHEMA_VERSION:
        raise ValueError("unexpected selective calibrator schema")
    _identifier(payload["calibrator_id"], "calibrator_id")
    if payload["policy_id"] != SELECTIVE_CALIBRATION_POLICY_ID:
        raise ValueError("unexpected selective calibration policy")
    if payload["evidence_order_score_method_id"] != EVIDENCE_ORDER_SCORE_METHOD_ID:
        raise ValueError("unexpected evidence ordering method")
    if payload["source_dataset_id"] != DEEPSOZ_TUSZ_DATASET_ID:
        raise ValueError("unexpected calibration source dataset")
    for name in (
        "source_dev_prediction_cohort_sha256",
        "source_dev_label_cohort_sha256",
        "locked_source_eval_prediction_cohort_sha256",
    ):
        _sha256(payload[name], name)
    _validate_content_hash(payload, "selective calibrator")

    split = _mapping(payload["split_receipt"], "split_receipt")
    split_keys = frozenset(
        {
            "source_dev_patient_tokens",
            "source_eval_patient_tokens",
            "patient_overlap_count",
            "patient_disjoint",
        }
    )
    _strict_keys(split, keys=split_keys, context="split_receipt")
    for name in ("source_dev_patient_tokens", "source_eval_patient_tokens"):
        tokens = split[name]
        if not isinstance(tokens, list) or not tokens or tokens != sorted(set(tokens)):
            raise ValueError(f"{name} must be a sorted non-empty unique list")
        for token in tokens:
            _sha256(token, name)
    if set(split["source_dev_patient_tokens"]) & set(split["source_eval_patient_tokens"]):
        raise ValueError("calibrator contains patient leakage")
    if (
        type(split["patient_overlap_count"]) is not int
        or split["patient_overlap_count"] != 0
        or split["patient_disjoint"] is not True
    ):
        raise ValueError("calibrator does not certify patient-disjoint partitions")

    targets = _mapping(payload["risk_targets"], "risk_targets")
    target_keys = frozenset(
        {
            "stronger_max_patient_macro_top1_risk",
            "limited_max_patient_macro_top1_risk",
            "minimum_accepted_patients",
        }
    )
    _strict_keys(targets, keys=target_keys, context="risk_targets")
    stronger_risk = _finite_rate(
        targets["stronger_max_patient_macro_top1_risk"], "stronger risk"
    )
    limited_risk = _finite_rate(
        targets["limited_max_patient_macro_top1_risk"], "limited risk"
    )
    if stronger_risk > limited_risk:
        raise ValueError("calibrator risk targets are reversed")
    minimum_patients = _positive_integer(
        targets["minimum_accepted_patients"], "minimum patients"
    )

    _validate_full_coverage_metrics(
        payload["source_dev_full_coverage_metrics"],
        "source_dev_full_coverage_metrics",
    )
    full_metrics = payload["source_dev_full_coverage_metrics"]
    if full_metrics["patient_count"] != len(split["source_dev_patient_tokens"]):
        raise ValueError("source-dev metrics disagree with the patient split receipt")
    summary = _mapping(
        payload["source_dev_risk_coverage_summary"],
        "source_dev_risk_coverage_summary",
    )
    summary_keys = frozenset(
        {"aurc", "patient_macro_aurc_over_record_coverage", "point_count"}
    )
    _strict_keys(summary, keys=summary_keys, context="source_dev_risk_coverage_summary")
    _finite_rate(summary["aurc"], "source_dev_risk_coverage_summary.aurc")
    _finite_rate(
        summary["patient_macro_aurc_over_record_coverage"],
        "source_dev_risk_coverage_summary.patient_macro_aurc_over_record_coverage",
    )
    _positive_integer(summary["point_count"], "source_dev risk curve point_count")

    points = _mapping(payload["frozen_working_points"], "frozen_working_points")
    if set(points) != set(CALIBRATED_EVIDENCE_LEVELS):
        raise ValueError("frozen working points must define exactly three tiers")
    _validate_working_point(points[STRONGER_EVIDENCE], STRONGER_EVIDENCE)
    _validate_working_point(points[LIMITED_EVIDENCE], LIMITED_EVIDENCE)
    weak = _mapping(points[WEAK_EVIDENCE], WEAK_EVIDENCE)
    if weak != {
        "threshold_inclusive": 0.0,
        "top_k_still_required": True,
        "risk_target_not_claimed": True,
    }:
        raise ValueError("weak tier must retain Top-k without a risk claim")
    strong_threshold = points[STRONGER_EVIDENCE]["threshold_inclusive"]
    limited_threshold = points[LIMITED_EVIDENCE]["threshold_inclusive"]
    if strong_threshold is not None and limited_threshold is not None and float(
        strong_threshold
    ) < float(limited_threshold):
        raise ValueError("stronger threshold must be at least the limited threshold")
    for tier, risk_limit in (
        (STRONGER_EVIDENCE, stronger_risk),
        (LIMITED_EVIDENCE, limited_risk),
    ):
        point = points[tier]
        if point["target_met"]:
            if point["accepted_patient_count"] < minimum_patients:
                raise ValueError(f"{tier} violates minimum accepted-patient policy")
            if point["patient_macro_top1_selective_risk"] > risk_limit:
                raise ValueError(f"{tier} violates its frozen source-dev risk target")
            if point["accepted_record_count"] > full_metrics["record_count"] or point[
                "accepted_patient_count"
            ] > full_metrics["patient_count"]:
                raise ValueError(f"{tier} accepted count exceeds source-dev cohort")
            if not math.isclose(
                point["record_coverage"],
                point["accepted_record_count"] / full_metrics["record_count"],
                abs_tol=1e-12,
            ) or not math.isclose(
                point["patient_coverage"],
                point["accepted_patient_count"] / full_metrics["patient_count"],
                abs_tol=1e-12,
            ):
                raise ValueError(f"{tier} coverage disagrees with accepted counts")
    if points[LIMITED_EVIDENCE]["record_coverage"] < points[STRONGER_EVIDENCE][
        "record_coverage"
    ]:
        raise ValueError("limited tier must cover at least the stronger working point")

    semantics = _mapping(payload["label_semantics"], "label_semantics")
    if semantics != {
        "ground_truth": DEEPSOZ_GT_SEMANTICS,
        "multiple_hard_positive_electrodes_supported": True,
        "soft_spread_used": False,
    }:
        raise ValueError("calibrator label semantics are unsafe")
    boundary = _mapping(payload["protocol_boundary"], "protocol_boundary")
    expected_boundary = {
        "predictions_frozen_before_labels_joined": True,
        "thresholds_fitted_on_source_dev_only": True,
        "source_eval_labels_accessed_during_fit": False,
        "source_eval_predictions_hash_locked_before_evaluation": True,
        "source_eval_model_or_threshold_selection_permitted": False,
        "top_k_emitted_at_every_evidence_level": True,
        "evidence_order_score_is_probability": False,
        "clinical_diagnosis_claim_permitted": False,
    }
    if boundary != expected_boundary:
        raise ValueError("calibrator protocol boundary is unsafe")
    input_receipt = {
        "policy_id": SELECTIVE_CALIBRATION_POLICY_ID,
        "source_dev_prediction_sha256": payload[
            "source_dev_prediction_cohort_sha256"
        ],
        "source_dev_label_sha256": payload["source_dev_label_cohort_sha256"],
        "locked_source_eval_prediction_sha256": payload[
            "locked_source_eval_prediction_cohort_sha256"
        ],
        "stronger_max_patient_macro_risk": stronger_risk,
        "limited_max_patient_macro_risk": limited_risk,
        "minimum_accepted_patients": minimum_patients,
    }
    expected_id = f"SOZ-SEL-V2-{_content_sha256(input_receipt)[:20]}"
    if payload["calibrator_id"] != expected_id:
        raise ValueError("calibrator_id is not reproducible from frozen inputs")
    return deepcopy(dict(payload))


def _tier_for_score(score: float, calibrator: Mapping[str, Any]) -> str:
    points = calibrator["frozen_working_points"]
    strong = points[STRONGER_EVIDENCE]["threshold_inclusive"]
    limited = points[LIMITED_EVIDENCE]["threshold_inclusive"]
    if strong is not None and score >= float(strong):
        return STRONGER_EVIDENCE
    if limited is not None and score >= float(limited):
        return LIMITED_EVIDENCE
    return WEAK_EVIDENCE


def _wording_zh(tier: str, ranked: Sequence[str]) -> str:
    joined = "、".join(ranked)
    top1 = ranked[0]
    if tier == STRONGER_EVIDENCE:
        qualifier = "跨事件证据相对较强"
    elif tier == LIMITED_EVIDENCE:
        qualifier = "存在一定支持，但跨事件证据有限"
    else:
        qualifier = "证据较弱或发作模式不一致"
    return (
        f"研究性头皮 EEG 起始候选通道排序为：{joined}；首位为 {top1}，"
        f"{qualifier}。该排序仅供研究和医生复核，不等同于皮层 SOZ、"
        "致痫区或治疗靶点。"
    )


def apply_selective_soz_calibrator(
    calibrator: Mapping[str, Any],
    prediction_cohort: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply frozen thresholds using prediction/strength-derived inputs only."""

    calibration = validate_selective_soz_calibrator(calibrator)
    cohort = validate_frozen_eeg_only_prediction_cohort(prediction_cohort)
    partition = cohort["partition"]
    if partition == SOURCE_DEV and cohort["content_sha256"] != calibration[
        "source_dev_prediction_cohort_sha256"
    ]:
        raise ValueError("unrecognized source-dev prediction cohort")
    if partition == SOURCE_EVAL and cohort["content_sha256"] != calibration[
        "locked_source_eval_prediction_cohort_sha256"
    ]:
        raise ValueError("source-eval prediction cohort differs from the pre-locked hash")
    patient_tokens = {_patient_token(str(row["patient_id"])) for row in cohort["records"]}
    if partition == DEPLOYMENT:
        benchmark_tokens = set(
            calibration["split_receipt"]["source_dev_patient_tokens"]
        ) | set(calibration["split_receipt"]["source_eval_patient_tokens"])
        if patient_tokens & benchmark_tokens:
            raise ValueError("deployment cohort overlaps a calibration/evaluation patient")

    rows: list[dict[str, Any]] = []
    for prediction in cohort["records"]:
        score = float(prediction["evidence_order_score"])
        tier = _tier_for_score(score, calibration)
        ranked = list(prediction["ranked_electrodes"])
        rows.append(
            {
                "patient_id": prediction["patient_id"],
                "recording_id": prediction["recording_id"],
                "prediction_content_sha256": prediction["prediction_content_sha256"],
                "evidence_order_score": score,
                "evidence_level": tier,
                "ranked_electrodes": ranked,
                "top_k_output_retained": True,
                "reporting_phrase_zh": _wording_zh(tier, ranked),
            }
        )
    payload = {
        "schema_version": SELECTIVE_PROJECTION_SCHEMA_VERSION,
        "calibrator_id": calibration["calibrator_id"],
        "calibrator_content_sha256": calibration["content_sha256"],
        "prediction_cohort_content_sha256": cohort["content_sha256"],
        "dataset_id": cohort["dataset_id"],
        "partition": partition,
        "record_count": len(rows),
        "records": rows,
        "inference_boundary": {
            "input_is_frozen_eeg_only_prediction_and_strength": True,
            "labels_read_during_projection": False,
            "edf_annotations_used": False,
            "excel_fields_used": False,
            "doctor_labels_used": False,
            "top_k_output_required": True,
            "tier_controls_wording_not_prediction": True,
            "evidence_order_score_is_probability": False,
        },
    }
    return validate_selective_soz_projection(_with_content_hash(payload))


def validate_selective_soz_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(payload, "selective SOZ projection")
    keys = frozenset(
        {
            "schema_version",
            "calibrator_id",
            "calibrator_content_sha256",
            "prediction_cohort_content_sha256",
            "dataset_id",
            "partition",
            "record_count",
            "records",
            "inference_boundary",
            "content_sha256",
        }
    )
    _strict_keys(payload, keys=keys, context="selective projection")
    if payload["schema_version"] != SELECTIVE_PROJECTION_SCHEMA_VERSION:
        raise ValueError("unexpected selective projection schema")
    _identifier(payload["calibrator_id"], "projection calibrator_id")
    _sha256(payload["calibrator_content_sha256"], "calibrator_content_sha256")
    _sha256(
        payload["prediction_cohort_content_sha256"],
        "prediction_cohort_content_sha256",
    )
    _identifier(payload["dataset_id"], "projection dataset_id")
    if payload["partition"] not in ALLOWED_PARTITIONS:
        raise ValueError("unexpected projection partition")
    _validate_content_hash(payload, "selective projection")
    rows = payload["records"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("projection records must be non-empty")
    record_count = _positive_integer(payload["record_count"], "projection record_count")
    if record_count != len(rows):
        raise ValueError("projection record count mismatch")
    row_keys = frozenset(
        {
            "patient_id",
            "recording_id",
            "prediction_content_sha256",
            "evidence_order_score",
            "evidence_level",
            "ranked_electrodes",
            "top_k_output_retained",
            "reporting_phrase_zh",
        }
    )
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"projection records[{index}]")
        _strict_keys(row, keys=row_keys, context=f"projection records[{index}]")
        _identifier(row["patient_id"], "projection patient_id")
        _identifier(row["recording_id"], "projection recording_id")
        _sha256(row["prediction_content_sha256"], "prediction_content_sha256")
        _finite_rate(row["evidence_order_score"], "evidence_order_score")
        if row["evidence_level"] not in CALIBRATED_EVIDENCE_LEVELS:
            raise ValueError("unexpected calibrated evidence level")
        ranked = _canonical_electrodes(
            row["ranked_electrodes"], "projection ranked_electrodes", minimum_length=3
        )
        if row["top_k_output_retained"] is not True:
            raise ValueError("selective projection suppressed a Top-k output")
        if row["reporting_phrase_zh"] != _wording_zh(row["evidence_level"], ranked):
            raise ValueError("projection wording is not deterministically fact-bound")
    boundary = _mapping(payload["inference_boundary"], "inference_boundary")
    expected = {
        "input_is_frozen_eeg_only_prediction_and_strength": True,
        "labels_read_during_projection": False,
        "edf_annotations_used": False,
        "excel_fields_used": False,
        "doctor_labels_used": False,
        "top_k_output_required": True,
        "tier_controls_wording_not_prediction": True,
        "evidence_order_score_is_probability": False,
    }
    if boundary != expected:
        raise ValueError("selective projection boundary is unsafe")
    return deepcopy(dict(payload))


def evaluate_frozen_selective_soz_calibrator(
    calibrator: Mapping[str, Any],
    source_eval_predictions: Mapping[str, Any],
    source_eval_labels: Mapping[str, Any],
    *,
    evaluation_run_id: str,
) -> dict[str, Any]:
    """Score the pre-locked source-eval cohort without refitting or selection."""

    calibration = validate_selective_soz_calibrator(calibrator)
    predictions = validate_frozen_eeg_only_prediction_cohort(source_eval_predictions)
    labels = validate_tusz_scalp_weak_label_cohort(source_eval_labels)
    evaluation_run_id = _identifier(evaluation_run_id, "evaluation_run_id")
    if predictions["partition"] != SOURCE_EVAL or labels["partition"] != SOURCE_EVAL:
        raise ValueError("formal evaluation requires source_eval predictions and labels")
    if predictions["content_sha256"] != calibration[
        "locked_source_eval_prediction_cohort_sha256"
    ]:
        raise ValueError("source-eval predictions do not match the pre-locked cohort")
    if labels["prediction_cohort_content_sha256"] != predictions["content_sha256"]:
        raise ValueError("source-eval labels are not bound to the locked predictions")
    eval_tokens = sorted(
        {_patient_token(str(row["patient_id"])) for row in predictions["records"]}
    )
    if eval_tokens != calibration["split_receipt"]["source_eval_patient_tokens"]:
        raise ValueError("source-eval patient membership differs from calibration lock")
    if set(eval_tokens) & set(calibration["split_receipt"]["source_dev_patient_tokens"]):
        raise ValueError("source-eval contains source-dev patients")

    projection = apply_selective_soz_calibrator(calibration, predictions)
    full_metrics = full_coverage_topk_metrics(predictions, labels)
    curve = selective_risk_coverage(predictions, labels)
    outcomes = _join_outcomes(predictions, labels)
    tier_by_identity = {
        (row["patient_id"], row["recording_id"]): row["evidence_level"]
        for row in projection["records"]
    }
    tier_metrics: dict[str, Any] = {}
    for tier in CALIBRATED_EVIDENCE_LEVELS:
        selected = [
            row
            for row in outcomes
            if tier_by_identity[(row["patient_id"], row["recording_id"])] == tier
        ]
        if not selected:
            tier_metrics[tier] = {
                "record_count": 0,
                "patient_count": 0,
                "record_micro": None,
                "patient_macro": None,
            }
            continue
        patients = sorted({str(row["patient_id"]) for row in selected})
        patient_values = [
            _metric_triplet(
                [row for row in selected if row["patient_id"] == patient_id]
            )
            for patient_id in patients
        ]
        tier_metrics[tier] = {
            "record_count": len(selected),
            "patient_count": len(patients),
            "record_micro": _metric_triplet(selected),
            "patient_macro": {
                name: mean(float(value[name]) for value in patient_values)
                for name in ("top1_accuracy", "hit_at_3", "mean_reciprocal_rank")
            },
        }

    payload = {
        "schema_version": SELECTIVE_EVALUATION_SCHEMA_VERSION,
        "evaluation_run_id": evaluation_run_id,
        "calibrator_id": calibration["calibrator_id"],
        "calibrator_content_sha256": calibration["content_sha256"],
        "source_eval_prediction_cohort_sha256": predictions["content_sha256"],
        "source_eval_label_cohort_sha256": labels["content_sha256"],
        "projection_content_sha256": projection["content_sha256"],
        "gt_semantics": DEEPSOZ_GT_SEMANTICS,
        "full_coverage_topk_metrics": full_metrics,
        "selective_risk_coverage": curve,
        "disjoint_evidence_tier_metrics": tier_metrics,
        "evaluation_boundary": {
            "source_eval_predictions_prelocked": True,
            "source_eval_labels_used_for_scoring_only": True,
            "model_selection_performed": False,
            "threshold_selection_performed": False,
            "calibrator_refit_performed": False,
            "one_shot_source_eval_release_required": True,
            "stateful_external_access_ledger_required": True,
            "stateless_module_claims_reentry_enforcement": False,
            "top_k_scored_at_full_coverage": True,
            "soft_spread_used": False,
            "gt_is_scalp_electrode_weak_label": True,
        },
    }
    return validate_selective_soz_evaluation(_with_content_hash(payload))


def validate_selective_soz_evaluation(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(payload, "selective SOZ evaluation")
    keys = frozenset(
        {
            "schema_version",
            "evaluation_run_id",
            "calibrator_id",
            "calibrator_content_sha256",
            "source_eval_prediction_cohort_sha256",
            "source_eval_label_cohort_sha256",
            "projection_content_sha256",
            "gt_semantics",
            "full_coverage_topk_metrics",
            "selective_risk_coverage",
            "disjoint_evidence_tier_metrics",
            "evaluation_boundary",
            "content_sha256",
        }
    )
    _strict_keys(payload, keys=keys, context="selective evaluation")
    if payload["schema_version"] != SELECTIVE_EVALUATION_SCHEMA_VERSION:
        raise ValueError("unexpected selective evaluation schema")
    _identifier(payload["evaluation_run_id"], "evaluation_run_id")
    _identifier(payload["calibrator_id"], "evaluation calibrator_id")
    for name in (
        "calibrator_content_sha256",
        "source_eval_prediction_cohort_sha256",
        "source_eval_label_cohort_sha256",
        "projection_content_sha256",
    ):
        _sha256(payload[name], name)
    if payload["gt_semantics"] != DEEPSOZ_GT_SEMANTICS:
        raise ValueError("evaluation GT semantics are unsafe")
    _validate_content_hash(payload, "selective evaluation")
    _validate_full_coverage_metrics(
        payload["full_coverage_topk_metrics"], "full_coverage_topk_metrics"
    )
    full_metrics = payload["full_coverage_topk_metrics"]
    _validate_selective_curve(
        payload["selective_risk_coverage"],
        "selective_risk_coverage",
        expected_record_count=full_metrics["record_count"],
        expected_patient_count=full_metrics["patient_count"],
    )
    tiers = _mapping(payload["disjoint_evidence_tier_metrics"], "tier metrics")
    if set(tiers) != set(CALIBRATED_EVIDENCE_LEVELS):
        raise ValueError("evaluation tier metrics are incomplete")
    tier_record_total = 0
    tier_keys = frozenset(
        {"record_count", "patient_count", "record_micro", "patient_macro"}
    )
    for tier in CALIBRATED_EVIDENCE_LEVELS:
        metrics = _mapping(tiers[tier], f"tier metrics.{tier}")
        _strict_keys(metrics, keys=tier_keys, context=f"tier metrics.{tier}")
        record_count = metrics["record_count"]
        patient_count = metrics["patient_count"]
        if (
            isinstance(record_count, bool)
            or not isinstance(record_count, int)
            or record_count < 0
            or isinstance(patient_count, bool)
            or not isinstance(patient_count, int)
            or patient_count < 0
            or patient_count > record_count
        ):
            raise ValueError(f"tier metrics.{tier} has invalid counts")
        tier_record_total += record_count
        if record_count == 0:
            if patient_count != 0 or metrics["record_micro"] is not None or metrics[
                "patient_macro"
            ] is not None:
                raise ValueError(f"empty tier metrics.{tier} must contain null metrics")
        else:
            _validate_metric_triplet(
                metrics["record_micro"], f"tier metrics.{tier}.record_micro"
            )
            _validate_metric_triplet(
                metrics["patient_macro"], f"tier metrics.{tier}.patient_macro"
            )
    if tier_record_total != full_metrics["record_count"]:
        raise ValueError("disjoint tier record counts do not cover source-eval exactly")
    boundary = _mapping(payload["evaluation_boundary"], "evaluation_boundary")
    expected = {
        "source_eval_predictions_prelocked": True,
        "source_eval_labels_used_for_scoring_only": True,
        "model_selection_performed": False,
        "threshold_selection_performed": False,
        "calibrator_refit_performed": False,
        "one_shot_source_eval_release_required": True,
        "stateful_external_access_ledger_required": True,
        "stateless_module_claims_reentry_enforcement": False,
        "top_k_scored_at_full_coverage": True,
        "soft_spread_used": False,
        "gt_is_scalp_electrode_weak_label": True,
    }
    if boundary != expected:
        raise ValueError("source-eval boundary permits selection or leakage")
    return deepcopy(dict(payload))


__all__ = [
    "ALLOWED_PARTITIONS",
    "CALIBRATED_EVIDENCE_LEVELS",
    "DEEPSOZ_GT_SEMANTICS",
    "DEEPSOZ_TUSZ_DATASET_ID",
    "DEPLOYMENT",
    "EVIDENCE_ORDER_SCORE_METHOD_ID",
    "FROZEN_EEG_ONLY_COHORT_SCHEMA_VERSION",
    "LIMITED_EVIDENCE",
    "SELECTIVE_CALIBRATION_POLICY_ID",
    "SELECTIVE_CALIBRATOR_SCHEMA_VERSION",
    "SELECTIVE_EVALUATION_SCHEMA_VERSION",
    "SELECTIVE_PROJECTION_SCHEMA_VERSION",
    "SOURCE_DEV",
    "SOURCE_EVAL",
    "STRONGER_EVIDENCE",
    "TUSZ_WEAK_LABEL_COHORT_SCHEMA_VERSION",
    "WEAK_EVIDENCE",
    "apply_selective_soz_calibrator",
    "build_frozen_eeg_only_prediction_cohort",
    "build_tusz_scalp_weak_label_cohort",
    "evaluate_frozen_selective_soz_calibrator",
    "fit_selective_soz_calibrator",
    "full_coverage_topk_metrics",
    "selective_risk_coverage",
    "validate_frozen_eeg_only_prediction_cohort",
    "validate_selective_soz_calibrator",
    "validate_selective_soz_evaluation",
    "validate_selective_soz_projection",
    "validate_tusz_scalp_weak_label_cohort",
]
