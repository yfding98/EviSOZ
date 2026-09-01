"""Pure-EEG, research-only aggregation of cross-event scalp rankings.

The input is limited to one complete C18 electrode ranking per EEG event and
an optional signal-evidence weight.  Model scores are retained only as
uncalibrated input proxies; cross-event aggregation and Jensen--Shannon (JS)
comparisons use reciprocal-rank mass so that incompatible score scales are
never silently interpreted as probabilities.

The output is an ordinal *scalp-electrode ranked hypothesis*.  It must not be
promoted to a cortical seizure-onset-zone or epileptogenic-zone conclusion,
and it never consumes ground truth, EDF annotations, spreadsheets, doctor
labels, or free-form clinical text.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from statistics import mean, median
from typing import Any, Mapping, Sequence

from src.soz.geometry import STANDARD_19


RESEARCH_SOZ_PREDICTION_SCHEMA_VERSION = (
    "clinical_eeg_research_cross_event_scalp_ranking_v1"
)
RESEARCH_SOZ_PREDICTION_METHOD_ID = (
    "weighted_reciprocal_rank_js_complete_link_v1"
)
C18_ELECTRODES: tuple[str, ...] = tuple(
    electrode for electrode in STANDARD_19 if electrode != "PZ"
)
DISPLAY_TIERS: tuple[str, ...] = (
    "high_ranked_hypothesis",
    "moderate_ranked_hypothesis",
    "low_ranked_hypothesis",
)
DEFAULT_JS_THRESHOLD = 0.12

_C18_INDEX = {electrode: index for index, electrode in enumerate(C18_ELECTRODES)}
_SHA256_HEX = frozenset("0123456789abcdef")
_EVENT_ALLOWED_KEYS = frozenset(
    {"event_id", "ranked_electrodes", "evidence_weight", "model_sha256"}
)
_RECEIPT_ALLOWED_KEYS = frozenset(
    {
        "receipt_id",
        "method_id",
        "model_sha256",
        "input_processed_window_sha256",
        "interpretation_status",
        "ranked_electrodes",
        "evidence_weight",
        "used_in_clinical_facts",
        "used_in_impression",
        "sent_to_llm",
    }
)
_RECEIPT_REQUIRED_KEYS = _RECEIPT_ALLOWED_KEYS - {"evidence_weight"}


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    allowed: frozenset[str],
    context: str,
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        raise ValueError(f"{context} is missing required keys: {missing}")
    if unknown:
        raise ValueError(f"{context} contains unknown keys: {unknown}")


def _finite_number(
    value: object,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{context} must be <= {maximum}")
    return result


def _positive_integer(value: object, context: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value < 1 or value > maximum:
        raise ValueError(f"{context} must be between 1 and {maximum}")
    return value


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise TypeError(f"{context} must be a non-empty string of at most 128 characters")
    allowed = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
    )
    if value[0] not in allowed or any(character not in allowed for character in value):
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


def _validate_ranking(value: object, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(C18_ELECTRODES):
        raise ValueError(
            f"{context} must contain exactly the complete C18 ranking"
        )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous_score = math.inf
    for index, raw_item in enumerate(value, start=1):
        item = _mapping(raw_item, f"{context}[{index - 1}]")
        _strict_keys(
            item,
            required=frozenset({"rank", "electrode", "score"}),
            allowed=frozenset({"rank", "electrode", "score"}),
            context=f"{context}[{index - 1}]",
        )
        if isinstance(item["rank"], bool) or item["rank"] != index:
            raise ValueError(f"{context} ranks must be contiguous and start at 1")
        electrode = item["electrode"]
        if not isinstance(electrode, str) or electrode not in _C18_INDEX:
            raise ValueError(
                f"{context}[{index - 1}].electrode must be a canonical C18 electrode"
            )
        if electrode in seen:
            raise ValueError(f"{context} contains duplicate electrodes")
        seen.add(electrode)
        score = _finite_number(
            item["score"],
            f"{context}[{index - 1}].score",
            minimum=0.0,
            maximum=1.0,
        )
        if score > previous_score:
            raise ValueError(f"{context} scores must be non-increasing")
        previous_score = score
        result.append({"rank": index, "electrode": electrode, "score": score})
    if seen != set(C18_ELECTRODES):
        raise ValueError(f"{context} must cover canonical C18 exactly once")
    return result


def _validate_events(events: object) -> list[dict[str, Any]]:
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise TypeError("events must be a sequence")
    if not events:
        raise ValueError("events must contain at least one EEG event ranking")
    validated: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for index, raw_event in enumerate(events, start=1):
        event = _mapping(raw_event, f"events[{index - 1}]")
        _strict_keys(
            event,
            required=frozenset({"ranked_electrodes"}),
            allowed=_EVENT_ALLOWED_KEYS,
            context=f"events[{index - 1}]",
        )
        event_id = _identifier(
            event.get("event_id", f"EVENT-{index:04d}"),
            f"events[{index - 1}].event_id",
        )
        if event_id in seen_event_ids:
            raise ValueError("events contain duplicate event_id values")
        seen_event_ids.add(event_id)
        evidence_weight = _finite_number(
            event.get("evidence_weight", 1.0),
            f"events[{index - 1}].evidence_weight",
            minimum=0.0,
            maximum=1.0,
        )
        model_sha256 = event.get("model_sha256")
        if model_sha256 is not None:
            model_sha256 = _sha256(
                model_sha256, f"events[{index - 1}].model_sha256"
            )
        validated.append(
            {
                "event_id": event_id,
                "evidence_weight": evidence_weight,
                "model_sha256": model_sha256,
                "ranked_electrodes": _validate_ranking(
                    event["ranked_electrodes"],
                    f"events[{index - 1}].ranked_electrodes",
                ),
            }
        )
    return validated


def _rank_maps(
    events: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, int]], list[dict[str, float]]]:
    ranks: list[dict[str, int]] = []
    scores: list[dict[str, float]] = []
    for event in events:
        ranks.append(
            {item["electrode"]: int(item["rank"]) for item in event["ranked_electrodes"]}
        )
        scores.append(
            {
                item["electrode"]: float(item["score"])
                for item in event["ranked_electrodes"]
            }
        )
    return ranks, scores


def _effective_weights(events: Sequence[Mapping[str, Any]]) -> tuple[list[float], bool]:
    provided = [float(event["evidence_weight"]) for event in events]
    if sum(provided) > 0.0:
        return provided, False
    return [1.0 for _ in events], True


def _weighted_median(values: Sequence[int], weights: Sequence[float]) -> float:
    if len(values) != len(weights) or not values:
        raise ValueError("weighted median requires aligned non-empty values")
    if sum(weights) <= 0.0:
        return float(median(values))
    ordered = sorted(zip(values, weights), key=lambda pair: pair[0])
    threshold = sum(weights) / 2.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    raise AssertionError("weighted median traversal did not terminate")


def _event_rank_mass(rank_map: Mapping[str, int]) -> list[float]:
    unnormalized = [1.0 / float(rank_map[electrode]) for electrode in C18_ELECTRODES]
    total = sum(unnormalized)
    return [value / total for value in unnormalized]


def _js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(C18_ELECTRODES) or len(right) != len(C18_ELECTRODES):
        raise ValueError("JS operands must be C18 rank-mass vectors")
    midpoint = [(a + b) / 2.0 for a, b in zip(left, right)]

    def _kl(values: Sequence[float]) -> float:
        return sum(
            value * math.log(value / middle)
            for value, middle in zip(values, midpoint)
            if value > 0.0
        )

    normalized = 0.5 * (_kl(left) + _kl(right)) / math.log(2.0)
    return min(1.0, max(0.0, normalized))


def _pairwise_js(
    masses: Sequence[Sequence[float]],
) -> tuple[dict[tuple[int, int], float], list[float]]:
    by_pair: dict[tuple[int, int], float] = {}
    values: list[float] = []
    for left in range(len(masses)):
        for right in range(left + 1, len(masses)):
            value = _js_divergence(masses[left], masses[right])
            by_pair[(left, right)] = value
            values.append(value)
    return by_pair, values


def _complete_link_clusters(
    event_count: int,
    pairwise: Mapping[tuple[int, int], float],
    threshold: float,
) -> list[list[int]]:
    clusters: list[list[int]] = [[index] for index in range(event_count)]

    def _distance(left: Sequence[int], right: Sequence[int]) -> float:
        return max(
            pairwise[(min(a, b), max(a, b))]
            for a in left
            for b in right
        )

    while len(clusters) > 1:
        candidates: list[tuple[float, int, int]] = []
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                distance = _distance(clusters[left], clusters[right])
                if distance <= threshold:
                    candidates.append((distance, left, right))
        if not candidates:
            break
        _, left, right = min(candidates)
        merged = sorted(clusters[left] + clusters[right])
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in (left, right)
        ]
        clusters.append(merged)
        clusters.sort(key=lambda cluster: (min(cluster), tuple(cluster)))
    return clusters


def _display_tier(rank: int) -> str:
    if rank == 1:
        return "high_ranked_hypothesis"
    if rank <= 3:
        return "moderate_ranked_hypothesis"
    return "low_ranked_hypothesis"


def _ranked_rows(
    events: Sequence[Mapping[str, Any]],
    event_indices: Sequence[int],
    *,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rank_maps, score_maps = _rank_maps(events)
    selected_events = [events[index] for index in event_indices]
    effective_weights, fallback = _effective_weights(selected_events)
    total_effective_weight = sum(effective_weights)
    provided_weight_total = sum(
        float(events[index]["evidence_weight"]) for index in event_indices
    )

    rows: list[dict[str, Any]] = []
    for electrode in C18_ELECTRODES:
        ranks = [rank_maps[index][electrode] for index in event_indices]
        scores = [score_maps[index][electrode] for index in event_indices]
        top1_count = sum(rank == 1 for rank in ranks)
        top3_count = sum(rank <= 3 for rank in ranks)
        weighted_top1 = sum(
            weight for rank, weight in zip(ranks, effective_weights) if rank == 1
        )
        weighted_top3 = sum(
            weight for rank, weight in zip(ranks, effective_weights) if rank <= 3
        )
        aggregate_proxy = sum(
            weight / float(rank) for rank, weight in zip(ranks, effective_weights)
        ) / total_effective_weight
        mean_input_score = sum(
            weight * score for score, weight in zip(scores, effective_weights)
        ) / total_effective_weight
        rows.append(
            {
                "electrode": electrode,
                "aggregate_rank_proxy": aggregate_proxy,
                "top1_support_count": top1_count,
                "top1_support_rate": top1_count / len(event_indices),
                "top3_support_count": top3_count,
                "top3_support_rate": top3_count / len(event_indices),
                "evidence_weighted_top1_support_rate": (
                    None if fallback else weighted_top1 / total_effective_weight
                ),
                "evidence_weighted_top3_support_rate": (
                    None if fallback else weighted_top3 / total_effective_weight
                ),
                "mean_rank": float(mean(ranks)),
                "median_rank": float(median(ranks)),
                "evidence_weighted_mean_rank": sum(
                    rank * weight for rank, weight in zip(ranks, effective_weights)
                )
                / total_effective_weight,
                "evidence_weighted_median_rank": _weighted_median(
                    ranks, effective_weights
                ),
                "mean_input_score_proxy": mean_input_score,
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["aggregate_rank_proxy"]),
            -int(row["top1_support_count"]),
            -int(row["top3_support_count"]),
            _C18_INDEX[str(row["electrode"])],
        )
    )
    ranked: list[dict[str, Any]] = []
    for rank, row in enumerate(rows[:top_k], start=1):
        ranked.append({"rank": rank, **row, "display_tier": _display_tier(rank)})
    return ranked, {
        "provided_evidence_weight_total": provided_weight_total,
        "effective_weight_total": total_effective_weight,
        "equal_weight_fallback_applied": fallback,
    }


def aggregate_research_soz_rankings(
    events: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 5,
    js_threshold: float = DEFAULT_JS_THRESHOLD,
) -> dict[str, Any]:
    """Aggregate complete per-event C18 rankings into a research artifact.

    Valid input always yields exactly ``top_k`` ranked scalp hypotheses.  If
    every evidence weight is zero, ordering falls back to equal-weight rank
    aggregation and the fallback is explicit; weighted support rates remain
    ``None`` rather than being fabricated.
    """

    top_k = _positive_integer(top_k, "top_k", maximum=len(C18_ELECTRODES))
    js_threshold = _finite_number(
        js_threshold, "js_threshold", minimum=0.0, maximum=1.0
    )
    validated_events = _validate_events(events)
    rank_maps, _ = _rank_maps(validated_events)
    masses = [_event_rank_mass(rank_map) for rank_map in rank_maps]
    pairwise, pair_values = _pairwise_js(masses)
    mean_js = float(mean(pair_values)) if pair_values else 0.0
    clusters = _complete_link_clusters(
        len(validated_events), pairwise, js_threshold
    )
    diagnostic_ranked_hypotheses, weight_policy = _ranked_rows(
        validated_events,
        list(range(len(validated_events))),
        top_k=max(top_k, 2),
    )
    ranked_hypotheses = diagnostic_ranked_hypotheses[:top_k]

    effective_weights, _ = _effective_weights(validated_events)
    total_effective_weight = sum(effective_weights)
    aggregate_mass = [
        sum(
            weight * masses[event_index][electrode_index]
            for event_index, weight in enumerate(effective_weights)
        )
        / total_effective_weight
        for electrode_index in range(len(C18_ELECTRODES))
    ]
    entropy = -sum(value * math.log(value) for value in aggregate_mass)
    normalized_entropy = entropy / math.log(len(C18_ELECTRODES))
    top1_margin = (
        float(diagnostic_ranked_hypotheses[0]["aggregate_rank_proxy"])
        - float(diagnostic_ranked_hypotheses[1]["aggregate_rank_proxy"])
    )

    mode_rows: list[dict[str, Any]] = []
    for cluster_number, event_indices in enumerate(clusters, start=1):
        cluster_ranked, cluster_weight_policy = _ranked_rows(
            validated_events, event_indices, top_k=top_k
        )
        mode_rows.append(
            {
                "cluster_id": f"MODE-{cluster_number:03d}",
                "event_ids": [
                    validated_events[index]["event_id"] for index in event_indices
                ],
                "event_count": len(event_indices),
                "provided_evidence_weight_total": cluster_weight_policy[
                    "provided_evidence_weight_total"
                ],
                "ranked_hypotheses": cluster_ranked,
            }
        )

    model_hashes = sorted(
        {
            str(event["model_sha256"])
            for event in validated_events
            if event["model_sha256"] is not None
        }
    )
    input_receipt = {
        "method_id": RESEARCH_SOZ_PREDICTION_METHOD_ID,
        "candidate_space": list(C18_ELECTRODES),
        "top_k": top_k,
        "js_threshold": js_threshold,
        "events": validated_events,
    }
    input_hash = _content_sha256(input_receipt)
    artifact: dict[str, Any] = {
        "schema_version": RESEARCH_SOZ_PREDICTION_SCHEMA_VERSION,
        "artifact_id": f"EEG-RANK-{input_hash[:20]}",
        "method_id": RESEARCH_SOZ_PREDICTION_METHOD_ID,
        "candidate_space": list(C18_ELECTRODES),
        "input_event_count": len(validated_events),
        "top_k": top_k,
        "js_threshold": js_threshold,
        "input_content_sha256": input_hash,
        "model_sha256s": model_hashes,
        "effective_weight_policy": weight_policy,
        "ranked_hypotheses": ranked_hypotheses,
        "aggregate_diagnostics": {
            "top1_electrode": ranked_hypotheses[0]["electrode"],
            "top1_support_rate": ranked_hypotheses[0]["top1_support_rate"],
            "top3_support_rate": ranked_hypotheses[0]["top3_support_rate"],
            "normalized_entropy": normalized_entropy,
            "top1_margin": top1_margin,
        },
        "cross_event_consistency": {
            "pair_count": len(pair_values),
            "mean_pairwise_js_divergence": mean_js,
            "minimum_pairwise_js_divergence": min(pair_values) if pair_values else 0.0,
            "maximum_pairwise_js_divergence": max(pair_values) if pair_values else 0.0,
            "jensen_shannon_consistency": 1.0 - mean_js,
            "cluster_linkage": "complete",
            "cluster_threshold_inclusive": js_threshold,
            "mode_cluster_count": len(clusters),
            "multimodal": len(clusters) > 1,
        },
        "event_mode_clusters": mode_rows,
        "claim_boundary": {
            "input_scope": "eeg_event_c18_rankings_and_signal_evidence_weights_only",
            "aggregation_semantics": "ordinal_reciprocal_rank_proxy",
            "display_tier_semantics": "ordinal_rank_band_not_confidence",
            "same_model_calibration_artifact_input": False,
            "rank_proxy_calibrated": False,
            "probability_claim_prohibited": True,
            "cortical_soz_or_epileptogenic_zone_claim_prohibited": True,
            "clinical_findings_or_impression_use_prohibited": True,
            "ground_truth_used": False,
            "edf_annotations_used": False,
            "excel_fields_used": False,
            "doctor_labels_used": False,
            "free_text_used": False,
        },
    }
    artifact["content_sha256"] = _content_sha256(artifact)
    return validate_research_soz_prediction_artifact(artifact)


def aggregate_research_soz_rankings_from_bundle(
    bundle: Mapping[str, Any],
    *,
    top_k: int = 5,
    js_threshold: float = DEFAULT_JS_THRESHOLD,
) -> dict[str, Any]:
    """Extract only ``events[*].research_soz_ranking_receipt`` and aggregate.

    All other bundle content is ignored and therefore cannot influence the
    content hash.  Each receipt must retain the existing research-only safety
    flags; missing rankings fail closed instead of silently changing the event
    denominator.
    """

    bundle = _mapping(bundle, "bundle")
    raw_events = bundle.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("bundle.events must be a non-empty list")
    if "event_count" in bundle and bundle["event_count"] != len(raw_events):
        raise ValueError("bundle.event_count does not match bundle.events")

    extracted: list[dict[str, Any]] = []
    for index, raw_event in enumerate(raw_events, start=1):
        event = _mapping(raw_event, f"bundle.events[{index - 1}]")
        receipt = _mapping(
            event.get("research_soz_ranking_receipt"),
            f"bundle.events[{index - 1}].research_soz_ranking_receipt",
        )
        _strict_keys(
            receipt,
            required=_RECEIPT_REQUIRED_KEYS,
            allowed=_RECEIPT_ALLOWED_KEYS,
            context=f"bundle.events[{index - 1}].research_soz_ranking_receipt",
        )
        if (
            receipt["interpretation_status"]
            != "research_scalp_electrode_ranking_not_clinical_soz"
        ):
            raise ValueError("bundle research ranking lost its non-clinical boundary")
        for flag in ("used_in_clinical_facts", "used_in_impression", "sent_to_llm"):
            if receipt[flag] is not False:
                raise ValueError(f"bundle research ranking {flag} must remain false")
        event_identifier = event.get("eeg_event_id")
        if event_identifier is None:
            event_number = event.get("event_number", index)
            if isinstance(event_number, bool) or not isinstance(event_number, int):
                raise TypeError("bundle event_number must be an integer")
            event_identifier = f"EVENT-{event_number:04d}"
        extracted.append(
            {
                "event_id": event_identifier,
                "ranked_electrodes": deepcopy(receipt["ranked_electrodes"]),
                "evidence_weight": receipt.get("evidence_weight", 1.0),
                "model_sha256": receipt["model_sha256"],
            }
        )
    return aggregate_research_soz_rankings(
        extracted, top_k=top_k, js_threshold=js_threshold
    )


def _nested_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for child in value.values() for text in _nested_strings(child)]
    if isinstance(value, list):
        return [text for child in value for text in _nested_strings(child)]
    return []


def _validate_hypothesis_rows(
    value: object,
    *,
    top_k: int,
    event_count: int,
    context: str,
) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != top_k:
        raise ValueError(f"{context} must contain exactly top_k hypotheses")
    required = frozenset(
        {
            "rank",
            "electrode",
            "aggregate_rank_proxy",
            "top1_support_count",
            "top1_support_rate",
            "top3_support_count",
            "top3_support_rate",
            "evidence_weighted_top1_support_rate",
            "evidence_weighted_top3_support_rate",
            "mean_rank",
            "median_rank",
            "evidence_weighted_mean_rank",
            "evidence_weighted_median_rank",
            "mean_input_score_proxy",
            "display_tier",
        }
    )
    rows: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    previous_proxy = math.inf
    for index, raw_row in enumerate(value, start=1):
        row = _mapping(raw_row, f"{context}[{index - 1}]")
        _strict_keys(
            row,
            required=required,
            allowed=required,
            context=f"{context}[{index - 1}]",
        )
        if row["rank"] != index:
            raise ValueError(f"{context} ranks must be contiguous")
        electrode = row["electrode"]
        if electrode not in _C18_INDEX or electrode in seen:
            raise ValueError(f"{context} contains an invalid electrode")
        seen.add(str(electrode))
        if row["display_tier"] != _display_tier(index):
            raise ValueError(f"{context} display_tier is not permitted")
        proxy = _finite_number(
            row["aggregate_rank_proxy"],
            f"{context}[{index - 1}].aggregate_rank_proxy",
            minimum=0.0,
            maximum=1.0,
        )
        if proxy > previous_proxy:
            raise ValueError(f"{context} aggregate proxies must be non-increasing")
        previous_proxy = proxy
        for prefix, maximum_rank in (
            ("mean_rank", len(C18_ELECTRODES)),
            ("median_rank", len(C18_ELECTRODES)),
            ("evidence_weighted_mean_rank", len(C18_ELECTRODES)),
            ("evidence_weighted_median_rank", len(C18_ELECTRODES)),
        ):
            _finite_number(
                row[prefix],
                f"{context}[{index - 1}].{prefix}",
                minimum=1.0,
                maximum=float(maximum_rank),
            )
        _finite_number(
            row["mean_input_score_proxy"],
            f"{context}[{index - 1}].mean_input_score_proxy",
            minimum=0.0,
            maximum=1.0,
        )
        for support_name in ("top1", "top3"):
            count_key = f"{support_name}_support_count"
            rate_key = f"{support_name}_support_rate"
            count = row[count_key]
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                or count > event_count
            ):
                raise ValueError(f"{context} has an invalid {count_key}")
            rate = _finite_number(
                row[rate_key],
                f"{context}[{index - 1}].{rate_key}",
                minimum=0.0,
                maximum=1.0,
            )
            if not math.isclose(rate, count / event_count, abs_tol=1e-12):
                raise ValueError(f"{context} {rate_key} does not match its count")
            weighted_key = f"evidence_weighted_{support_name}_support_rate"
            weighted = row[weighted_key]
            if weighted is not None:
                _finite_number(
                    weighted,
                    f"{context}[{index - 1}].{weighted_key}",
                    minimum=0.0,
                    maximum=1.0,
                )
        rows.append(row)
    return rows


def validate_research_soz_prediction_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate hashes and the non-clinical semantic boundary of an artifact."""

    payload = _mapping(payload, "research SOZ prediction artifact")
    required = frozenset(
        {
            "schema_version",
            "artifact_id",
            "method_id",
            "candidate_space",
            "input_event_count",
            "top_k",
            "js_threshold",
            "input_content_sha256",
            "model_sha256s",
            "effective_weight_policy",
            "ranked_hypotheses",
            "aggregate_diagnostics",
            "cross_event_consistency",
            "event_mode_clusters",
            "claim_boundary",
            "content_sha256",
        }
    )
    _strict_keys(payload, required=required, allowed=required, context="artifact")
    if payload["schema_version"] != RESEARCH_SOZ_PREDICTION_SCHEMA_VERSION:
        raise ValueError("unexpected research SOZ prediction schema version")
    if payload["method_id"] != RESEARCH_SOZ_PREDICTION_METHOD_ID:
        raise ValueError("unexpected research SOZ prediction method")
    _identifier(payload["artifact_id"], "artifact.artifact_id")
    _sha256(payload["input_content_sha256"], "artifact.input_content_sha256")
    saved_hash = _sha256(payload["content_sha256"], "artifact.content_sha256")
    hashable = dict(payload)
    hashable.pop("content_sha256")
    if _content_sha256(hashable) != saved_hash:
        raise ValueError("research SOZ prediction artifact content hash mismatch")
    if payload["candidate_space"] != list(C18_ELECTRODES):
        raise ValueError("artifact candidate_space must be canonical C18")
    if (
        isinstance(payload["input_event_count"], bool)
        or not isinstance(payload["input_event_count"], int)
        or payload["input_event_count"] < 1
    ):
        raise ValueError("artifact input_event_count must be positive")
    top_k = _positive_integer(
        payload["top_k"], "artifact.top_k", maximum=len(C18_ELECTRODES)
    )
    _finite_number(
        payload["js_threshold"],
        "artifact.js_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    event_count = int(payload["input_event_count"])
    ranked = _validate_hypothesis_rows(
        payload["ranked_hypotheses"],
        top_k=top_k,
        event_count=event_count,
        context="artifact.ranked_hypotheses",
    )
    model_hashes = payload["model_sha256s"]
    if (
        not isinstance(model_hashes, list)
        or model_hashes != sorted(set(model_hashes))
    ):
        raise ValueError("artifact.model_sha256s must be a sorted unique list")
    for index, model_hash in enumerate(model_hashes):
        _sha256(model_hash, f"artifact.model_sha256s[{index}]")

    weight_policy = _mapping(
        payload["effective_weight_policy"], "artifact.effective_weight_policy"
    )
    weight_keys = frozenset(
        {
            "provided_evidence_weight_total",
            "effective_weight_total",
            "equal_weight_fallback_applied",
        }
    )
    _strict_keys(
        weight_policy,
        required=weight_keys,
        allowed=weight_keys,
        context="artifact.effective_weight_policy",
    )
    provided_weight = _finite_number(
        weight_policy["provided_evidence_weight_total"],
        "artifact provided evidence weight",
        minimum=0.0,
        maximum=float(event_count),
    )
    effective_weight = _finite_number(
        weight_policy["effective_weight_total"],
        "artifact effective evidence weight",
        minimum=0.0,
        maximum=float(event_count),
    )
    fallback = weight_policy["equal_weight_fallback_applied"]
    if not isinstance(fallback, bool):
        raise TypeError("artifact equal_weight_fallback_applied must be boolean")
    if fallback != (provided_weight == 0.0):
        raise ValueError("artifact equal-weight fallback does not match provided weight")
    if effective_weight <= 0.0:
        raise ValueError("artifact effective evidence weight must be positive")

    diagnostics = _mapping(
        payload["aggregate_diagnostics"], "artifact.aggregate_diagnostics"
    )
    diagnostic_keys = frozenset(
        {
            "top1_electrode",
            "top1_support_rate",
            "top3_support_rate",
            "normalized_entropy",
            "top1_margin",
        }
    )
    _strict_keys(
        diagnostics,
        required=diagnostic_keys,
        allowed=diagnostic_keys,
        context="artifact.aggregate_diagnostics",
    )
    if diagnostics["top1_electrode"] != ranked[0]["electrode"]:
        raise ValueError("artifact top1 diagnostic does not match ranking")
    for key in (
        "top1_support_rate",
        "top3_support_rate",
        "normalized_entropy",
        "top1_margin",
    ):
        _finite_number(
            diagnostics[key],
            f"artifact.aggregate_diagnostics.{key}",
            minimum=0.0,
            maximum=1.0,
        )
    if not math.isclose(
        float(diagnostics["top1_support_rate"]),
        float(ranked[0]["top1_support_rate"]),
        abs_tol=1e-12,
    ) or not math.isclose(
        float(diagnostics["top3_support_rate"]),
        float(ranked[0]["top3_support_rate"]),
        abs_tol=1e-12,
    ):
        raise ValueError("artifact support diagnostics do not match top hypothesis")

    consistency = _mapping(
        payload["cross_event_consistency"], "artifact.cross_event_consistency"
    )
    consistency_keys = frozenset(
        {
            "pair_count",
            "mean_pairwise_js_divergence",
            "minimum_pairwise_js_divergence",
            "maximum_pairwise_js_divergence",
            "jensen_shannon_consistency",
            "cluster_linkage",
            "cluster_threshold_inclusive",
            "mode_cluster_count",
            "multimodal",
        }
    )
    _strict_keys(
        consistency,
        required=consistency_keys,
        allowed=consistency_keys,
        context="artifact.cross_event_consistency",
    )
    if consistency["pair_count"] != event_count * (event_count - 1) // 2:
        raise ValueError("artifact JS pair_count is inconsistent")
    for key in (
        "mean_pairwise_js_divergence",
        "minimum_pairwise_js_divergence",
        "maximum_pairwise_js_divergence",
        "jensen_shannon_consistency",
    ):
        _finite_number(
            consistency[key],
            f"artifact.cross_event_consistency.{key}",
            minimum=0.0,
            maximum=1.0,
        )
    if not math.isclose(
        float(consistency["jensen_shannon_consistency"]),
        1.0 - float(consistency["mean_pairwise_js_divergence"]),
        abs_tol=1e-12,
    ):
        raise ValueError("artifact JS consistency must equal one minus mean divergence")
    if consistency["cluster_linkage"] != "complete":
        raise ValueError("artifact clustering must use complete linkage")
    if not math.isclose(
        float(consistency["cluster_threshold_inclusive"]),
        float(payload["js_threshold"]),
        abs_tol=1e-12,
    ):
        raise ValueError("artifact cluster threshold does not match input policy")

    clusters = payload["event_mode_clusters"]
    if not isinstance(clusters, list) or not clusters:
        raise ValueError("artifact event_mode_clusters must be non-empty")
    if consistency["mode_cluster_count"] != len(clusters):
        raise ValueError("artifact mode_cluster_count does not match clusters")
    if consistency["multimodal"] is not (len(clusters) > 1):
        raise ValueError("artifact multimodal flag does not match clusters")
    cluster_keys = frozenset(
        {
            "cluster_id",
            "event_ids",
            "event_count",
            "provided_evidence_weight_total",
            "ranked_hypotheses",
        }
    )
    all_cluster_event_ids: list[str] = []
    cluster_event_total = 0
    for cluster_index, raw_cluster in enumerate(clusters, start=1):
        cluster = _mapping(
            raw_cluster, f"artifact.event_mode_clusters[{cluster_index - 1}]"
        )
        _strict_keys(
            cluster,
            required=cluster_keys,
            allowed=cluster_keys,
            context=f"artifact.event_mode_clusters[{cluster_index - 1}]",
        )
        if cluster["cluster_id"] != f"MODE-{cluster_index:03d}":
            raise ValueError("artifact cluster IDs must be canonical and contiguous")
        cluster_count = cluster["event_count"]
        if (
            isinstance(cluster_count, bool)
            or not isinstance(cluster_count, int)
            or cluster_count < 1
        ):
            raise ValueError("artifact cluster event_count must be positive")
        event_ids = cluster["event_ids"]
        if not isinstance(event_ids, list) or len(event_ids) != cluster_count:
            raise ValueError("artifact cluster event_ids do not match event_count")
        for event_index, event_id in enumerate(event_ids):
            all_cluster_event_ids.append(
                _identifier(
                    event_id,
                    f"artifact cluster {cluster_index} event_ids[{event_index}]",
                )
            )
        _finite_number(
            cluster["provided_evidence_weight_total"],
            "artifact cluster provided evidence weight",
            minimum=0.0,
            maximum=float(cluster_count),
        )
        _validate_hypothesis_rows(
            cluster["ranked_hypotheses"],
            top_k=top_k,
            event_count=cluster_count,
            context=f"artifact.event_mode_clusters[{cluster_index - 1}].ranked_hypotheses",
        )
        cluster_event_total += cluster_count
    if cluster_event_total != event_count or len(all_cluster_event_ids) != len(
        set(all_cluster_event_ids)
    ):
        raise ValueError("artifact clusters must partition all input events exactly once")
    boundary = _mapping(payload["claim_boundary"], "artifact.claim_boundary")
    required_false = (
        "same_model_calibration_artifact_input",
        "rank_proxy_calibrated",
        "ground_truth_used",
        "edf_annotations_used",
        "excel_fields_used",
        "doctor_labels_used",
        "free_text_used",
    )
    if any(boundary.get(key) is not False for key in required_false):
        raise ValueError("artifact claim boundary admits a prohibited input or calibration")
    required_true = (
        "probability_claim_prohibited",
        "cortical_soz_or_epileptogenic_zone_claim_prohibited",
        "clinical_findings_or_impression_use_prohibited",
    )
    if any(boundary.get(key) is not True for key in required_true):
        raise ValueError("artifact claim boundary permits an unsafe clinical claim")
    prohibited_surface = ("皮层soz", "致痫区", "epileptogenic zone", "cortical soz")
    for text in _nested_strings(payload):
        lowered = text.lower()
        if any(term in lowered for term in prohibited_surface):
            raise ValueError("artifact contains a prohibited cortical localization claim")
    return deepcopy(dict(payload))


__all__ = [
    "C18_ELECTRODES",
    "DEFAULT_JS_THRESHOLD",
    "DISPLAY_TIERS",
    "RESEARCH_SOZ_PREDICTION_METHOD_ID",
    "RESEARCH_SOZ_PREDICTION_SCHEMA_VERSION",
    "aggregate_research_soz_rankings",
    "aggregate_research_soz_rankings_from_bundle",
    "validate_research_soz_prediction_artifact",
]
