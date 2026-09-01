"""Common-17 channel and evaluation contract for long-EEG SOZ experiments.

The experiment removes FZ and PZ from the canonical 19-electrode model head
and retains CZ.  Reference labels on FZ/PZ are *experimentally* remapped to
CZ, then deduplicated.  This is a target-space compatibility experiment, not
an assertion that FZ, PZ and CZ are physiologically interchangeable.

DeepSOZ N2/N4 are reported only as neighbour-tolerant sensitivity endpoints.
Their one-hop graph is the induced subgraph of the published STANDARD_19
lookup: deleting FZ/PZ never creates a new edge through a deleted node.  The
N2/N4 gate uses the number of unique hard positives before remapping.  Known
soft/spread electrodes are excluded from neighbour-relaxed successes.

The evaluator exposes both forced and conditional views.  Forced evaluation
scores an applicable reference with no prediction as zero; conditional
evaluation includes only records with a non-empty common-17 ranking.  Patient
macro metrics first average records within patient and then patients equally.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.soz.geometry import STANDARD_19, normalize_electrode_name
from src.soz.metrics import DEEPSOZ_STANDARD19_NEIGHBORS


SCHEMA_VERSION = "clinical_eeg_common17_experiment_v1"
POLICY_ID = "standard19_minus_fz_pz_labels_to_cz_deepsoz_induced_v1"
REMOVED_ELECTRODES: tuple[str, ...] = ("FZ", "PZ")
LABEL_REMAP: Mapping[str, str] = {"FZ": "CZ", "PZ": "CZ"}
COMMON_17: tuple[str, ...] = tuple(
    electrode for electrode in STANDARD_19 if electrode not in REMOVED_ELECTRODES
)
COMMON_17_INDEX: Mapping[str, int] = {
    electrode: index for index, electrode in enumerate(COMMON_17)
}

_STANDARD19_INDEX = {electrode: index for index, electrode in enumerate(STANDARD_19)}
_CURRENT_PRIVATE_METRICS_SCHEMA = "private_recording_soz_postfreeze_evaluation_v1"


def _build_induced_neighbors() -> Mapping[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    retained = set(COMMON_17)
    for electrode in COMMON_17:
        original_index = _STANDARD19_INDEX[electrode]
        result[electrode] = tuple(
            STANDARD_19[neighbor_index]
            for neighbor_index in DEEPSOZ_STANDARD19_NEIGHBORS[original_index]
            if STANDARD_19[neighbor_index] in retained
        )
    return result


# Rows are indexed by the true electrode and contain acceptable predicted
# electrodes.  The published lookup is directed and is preserved as-is.
DEEPSOZ_COMMON17_INDUCED_NEIGHBORS: Mapping[str, tuple[str, ...]] = (
    _build_induced_neighbors()
)


@dataclass(frozen=True)
class Common17Reference:
    """A remapped reference with explicit pre-remap gate provenance."""

    hard_before_remap: tuple[str, ...]
    soft_before_remap: tuple[str, ...]
    hard: tuple[str, ...]
    soft: tuple[str, ...]
    hard_positive_count_before_remap: int
    hard_positive_count_after_remap: int


def _canonical_unique(
    values: Iterable[object],
    *,
    context: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{context} must be an electrode sequence")
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        electrode = normalize_electrode_name(raw)
        if electrode not in _STANDARD19_INDEX:
            raise ValueError(f"{context} contains an electrode outside STANDARD_19: {raw!r}")
        if electrode not in seen:
            result.append(electrode)
            seen.add(electrode)
    return tuple(result)


def project_reference_to_common17(
    hard_electrodes: Iterable[object],
    soft_electrodes: Iterable[object] = (),
) -> Common17Reference:
    """Map FZ/PZ reference labels to CZ, deduplicate, and give hard priority."""

    hard_before = _canonical_unique(hard_electrodes, context="hard reference")
    soft_before = _canonical_unique(soft_electrodes, context="soft reference")
    hard = tuple(dict.fromkeys(LABEL_REMAP.get(item, item) for item in hard_before))
    hard_set = set(hard)
    soft = tuple(
        item
        for item in dict.fromkeys(LABEL_REMAP.get(item, item) for item in soft_before)
        if item not in hard_set
    )
    if any(item not in COMMON_17_INDEX for item in (*hard, *soft)):
        raise RuntimeError("common-17 label projection escaped its output space")
    return Common17Reference(
        hard_before_remap=hard_before,
        soft_before_remap=soft_before,
        hard=hard,
        soft=soft,
        hard_positive_count_before_remap=len(hard_before),
        hard_positive_count_after_remap=len(hard),
    )


def project_ranking_to_common17(ranked_electrodes: Iterable[object]) -> tuple[str, ...]:
    """Delete FZ/PZ predictions; never remap model predictions to CZ."""

    canonical = _canonical_unique(ranked_electrodes, context="prediction ranking")
    return tuple(item for item in canonical if item in COMMON_17_INDEX)


def _nonempty_identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{context} must be a non-empty trimmed string")
    return value


def _normalize_k_values(k_values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(k_values, (str, bytes)):
        raise TypeError("k_values must be a sequence of integers")
    values: set[int] = set()
    for raw in k_values:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError("k_values must contain integers")
        if raw < 1 or raw > len(COMMON_17):
            raise ValueError("k_values must be within the common-17 output space")
        values.add(raw)
    if not values:
        raise ValueError("k_values must not be empty")
    return tuple(sorted(values))


def _relaxed_top1_hit(
    top1: str | None,
    reference: Common17Reference,
    *,
    max_positive_before_remap: int,
) -> float:
    if top1 is None:
        return 0.0
    acceptable = set(reference.hard)
    if reference.hard_positive_count_before_remap <= max_positive_before_remap:
        for true_electrode in reference.hard:
            acceptable.update(DEEPSOZ_COMMON17_INDUCED_NEIGHBORS[true_electrode])
    acceptable.difference_update(reference.soft)
    return float(top1 in acceptable)


def _score_record(
    *,
    record_id: str,
    patient_id: str,
    ranking: tuple[str, ...],
    reference: Common17Reference,
    k_values: tuple[int, ...],
    prediction_available: bool,
    prefix_censored_after_projection: bool,
) -> dict[str, Any]:
    top1 = ranking[0] if ranking else None
    hard = set(reference.hard)
    first_relevant_rank = next(
        (rank for rank, electrode in enumerate(ranking, start=1) if electrode in hard),
        None,
    )
    return {
        "recording_id": record_id,
        "patient_id": patient_id,
        "prediction_available": prediction_available,
        "ranking": list(ranking),
        "hard_before_remap": list(reference.hard_before_remap),
        "hard": list(reference.hard),
        "soft": list(reference.soft),
        "hard_positive_count_before_remap": reference.hard_positive_count_before_remap,
        "hard_positive_count_after_remap": reference.hard_positive_count_after_remap,
        "prefix_censored_after_projection": prefix_censored_after_projection,
        "exact_top1": float(top1 in hard) if top1 is not None else 0.0,
        "hit_at_k": {
            str(k): float(bool(hard.intersection(ranking[:k]))) for k in k_values
        },
        "mrr": 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
        "deepsoz_n2_top1": _relaxed_top1_hit(
            top1, reference, max_positive_before_remap=2
        ),
        "deepsoz_n4_top1": _relaxed_top1_hit(
            top1, reference, max_positive_before_remap=4
        ),
        "n2_neighbor_gate_open": reference.hard_positive_count_before_remap <= 2,
        "n4_neighbor_gate_open": reference.hard_positive_count_before_remap <= 4,
    }


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else math.fsum(values) / len(values)


def _aggregate(rows: Sequence[Mapping[str, Any]], k_values: tuple[int, ...]) -> dict[str, Any]:
    by_patient: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_patient.setdefault(str(row["patient_id"]), []).append(row)

    def value(row: Mapping[str, Any], field: str) -> float:
        return float(row[field])

    def patient_macro(field: str) -> float:
        return _mean(
            [_mean([value(row, field) for row in patient_rows]) for patient_rows in by_patient.values()]
        )

    hit = {
        str(k): _mean([float(row["hit_at_k"][str(k)]) for row in rows])
        for k in k_values
    }
    patient_hit = {
        str(k): _mean(
            [
                _mean([float(row["hit_at_k"][str(k)]) for row in patient_rows])
                for patient_rows in by_patient.values()
            ]
        )
        for k in k_values
    }
    return {
        "denominator": len(rows),
        "patient_denominator": len(by_patient),
        "exact_top1_accuracy": _mean([value(row, "exact_top1") for row in rows]),
        "hit_at_k": hit,
        "mrr": _mean([value(row, "mrr") for row in rows]),
        "deepsoz_n2_top1_accuracy": _mean(
            [value(row, "deepsoz_n2_top1") for row in rows]
        ),
        "deepsoz_n4_top1_accuracy": _mean(
            [value(row, "deepsoz_n4_top1") for row in rows]
        ),
        "n2_neighbor_gate_open_count": sum(
            bool(row["n2_neighbor_gate_open"]) for row in rows
        ),
        "n4_neighbor_gate_open_count": sum(
            bool(row["n4_neighbor_gate_open"]) for row in rows
        ),
        "prefix_censored_after_projection_count": sum(
            bool(row["prefix_censored_after_projection"]) for row in rows
        ),
        "patient_macro": {
            "exact_top1_accuracy": patient_macro("exact_top1"),
            "hit_at_k": patient_hit,
            "mrr": patient_macro("mrr"),
            "deepsoz_n2_top1_accuracy": patient_macro("deepsoz_n2_top1"),
            "deepsoz_n4_top1_accuracy": patient_macro("deepsoz_n4_top1"),
        },
    }


def evaluate_common17_records(
    records: Sequence[Mapping[str, Any]],
    *,
    k_values: Sequence[int] = (1, 3, 5),
    source_prediction_top_k: int | None = None,
) -> dict[str, Any]:
    """Evaluate current private-metrics-style records on the common-17 space.

    Required record fields are ``recording_id``, ``patient_pseudonym``,
    ``prediction_status``, ``ranked_electrodes``,
    ``hard_significant_electrodes`` and ``soft_spread_electrodes``.  Extra
    fields are ignored so the function can directly consume the frozen
    private post-freeze metrics artifact's ``records`` array.
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of objects")
    ks = _normalize_k_values(k_values)
    if source_prediction_top_k is not None:
        if (
            isinstance(source_prediction_top_k, bool)
            or not isinstance(source_prediction_top_k, int)
            or source_prediction_top_k < max(ks)
        ):
            raise ValueError("source_prediction_top_k must cover every requested k")

    scored: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    applicable_count = 0
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise TypeError(f"records[{index}] must be an object")
        record_id = _nonempty_identifier(raw.get("recording_id"), "recording_id")
        patient_id = _nonempty_identifier(
            raw.get("patient_pseudonym"), "patient_pseudonym"
        )
        if record_id in seen_record_ids:
            raise ValueError(f"duplicate recording_id: {record_id}")
        seen_record_ids.add(record_id)
        status = raw.get("prediction_status")
        if status not in {"completed", "skipped"}:
            raise ValueError("prediction_status must be completed or skipped")
        ranking_raw = raw.get("ranked_electrodes")
        if not isinstance(ranking_raw, list):
            raise TypeError("ranked_electrodes must be a list")
        if status == "skipped" and ranking_raw:
            raise ValueError("a skipped prediction must have an empty ranking")
        ranking = project_ranking_to_common17(ranking_raw)
        prediction_available = status == "completed" and bool(ranking)

        hard_raw = raw.get("hard_significant_electrodes")
        soft_raw = raw.get("soft_spread_electrodes")
        if not isinstance(hard_raw, list) or not isinstance(soft_raw, list):
            raise TypeError("hard and soft references must be lists")
        reference = project_reference_to_common17(hard_raw, soft_raw)
        if not reference.hard:
            continue
        applicable_count += 1

        prefix_censored = bool(
            source_prediction_top_k is not None
            and status == "completed"
            and len(ranking_raw) == source_prediction_top_k
            and len(ranking) < source_prediction_top_k
        )
        scored.append(
            _score_record(
                record_id=record_id,
                patient_id=patient_id,
                ranking=ranking if prediction_available else (),
                reference=reference,
                k_values=ks,
                prediction_available=prediction_available,
                prefix_censored_after_projection=prefix_censored,
            )
        )

    forced = scored
    conditional = [row for row in scored if row["prediction_available"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "candidate_space": list(COMMON_17),
        "removed_model_channels": list(REMOVED_ELECTRODES),
        "reference_label_remap": dict(LABEL_REMAP),
        "deepsoz_neighbor_graph_policy": (
            "published_standard19_directed_lookup_induced_subgraph_no_deleted_node_closure"
        ),
        "n2_n4_are_strict_accuracy": False,
        "n2_gate_max_hard_positive_count_before_remap": 2,
        "n4_gate_max_hard_positive_count_before_remap": 4,
        "soft_known_spread_excluded_from_relaxed_success": True,
        "hard_label_priority_over_soft_after_remap": True,
        "input_record_count": len(records),
        "applicable_hard_reference_record_count": applicable_count,
        "source_prediction_top_k": source_prediction_top_k,
        "requested_k_values": list(ks),
        "forced_full_gt_coverage": _aggregate(forced, ks),
        "conditional_on_prediction": _aggregate(conditional, ks),
        "records": scored,
    }


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def evaluate_private_metrics_file(
    path: str | Path,
    *,
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Load and re-evaluate the current private post-freeze metrics records."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("private metrics path must be a regular non-symlink file")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("private metrics file must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise TypeError("private metrics artifact must be an object")
    if payload.get("schema_version") != _CURRENT_PRIVATE_METRICS_SCHEMA:
        raise ValueError("unexpected private metrics schema_version")
    records = payload.get("records")
    if not isinstance(records, list):
        raise TypeError("private metrics records must be a list")
    receipts = payload.get("input_receipts")
    if not isinstance(receipts, Mapping):
        raise TypeError("private metrics input_receipts must be an object")
    prediction_receipt = receipts.get("prediction_cohort")
    if not isinstance(prediction_receipt, Mapping):
        raise TypeError("private metrics prediction cohort receipt must be an object")
    source_top_k = prediction_receipt.get("top_k")
    if isinstance(source_top_k, bool) or not isinstance(source_top_k, int):
        raise TypeError("private metrics source top_k must be an integer")
    result = evaluate_common17_records(
        records,
        k_values=k_values,
        source_prediction_top_k=source_top_k,
    )
    result["source_artifact"] = {
        "schema_version": payload["schema_version"],
        "evaluation_unit": payload.get("evaluation_unit"),
        "ranking_prefix_only": True,
        "projection_censoring_interpretation": (
            "when a removed channel occurs in a frozen top-k prefix, Hit@k is a lower bound "
            "because rank k+1 was not retained"
        ),
    }
    return result


if len(COMMON_17) != 17 or "CZ" not in COMMON_17_INDEX:
    raise RuntimeError("common-17 must retain CZ and contain exactly 17 electrodes")
if any(item in COMMON_17_INDEX for item in REMOVED_ELECTRODES):
    raise RuntimeError("common-17 still contains a removed midline electrode")
if set(DEEPSOZ_COMMON17_INDUCED_NEIGHBORS) != set(COMMON_17):
    raise RuntimeError("DeepSOZ common-17 induced graph is incomplete")


__all__ = [
    "COMMON_17",
    "COMMON_17_INDEX",
    "DEEPSOZ_COMMON17_INDUCED_NEIGHBORS",
    "LABEL_REMAP",
    "POLICY_ID",
    "REMOVED_ELECTRODES",
    "SCHEMA_VERSION",
    "Common17Reference",
    "evaluate_common17_records",
    "evaluate_private_metrics_file",
    "project_ranking_to_common17",
    "project_reference_to_common17",
]
