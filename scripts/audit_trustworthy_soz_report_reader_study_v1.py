#!/usr/bin/env python3
"""Audit completed target-blind qualified-report reader annotations.

The audit preserves the patient/case as the resampling unit.  Both readers and
all clauses for a sampled case travel together in every bootstrap replicate.
It never opens EEG, DeepSOZ/private targets, TUSZ channel annotations, model
scores, correctness metrics, or report-source training artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import statistics
from typing import Callable, Iterable, Mapping, Sequence, TypeVar


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "outputs/trustworthy_soz_report_reader_study_v1_20260815"
DEFAULT_OUTPUT = (
    ROOT / "outputs/trustworthy_soz_report_reader_study_v1_preflight_20260815.json"
)

from scripts.build_trustworthy_soz_report_reader_study_v1 import PACK_SCHEMA  # noqa: E402
from scripts.serve_trustworthy_soz_report_reader_study_v1 import (  # noqa: E402
    _validate_completed,
)


AUDIT_SCHEMA = "trustworthy_soz_report_reader_study_audit_v1"
LAYERS = ("event_clause_factuality", "patient_candidate_utility")
SIGNAL_FACT_CLAUSE_TYPES = {
    "event_scalp_evidence",
    "later_visible_order",
    "artifact_qualification",
}
SUPPORT_CATEGORIES = (
    "supported",
    "partially_supported",
    "unsupported",
    "not_assessable",
)
T = TypeVar("T")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.resolve(strict=True).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _unique(rows: Iterable[Mapping[str, object]], key: str, *, name: str) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for raw in rows:
        value = raw.get(key)
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f"{name} has missing/duplicate {key}")
        result[value] = dict(raw)
    return result


def _percentile(values: Sequence[float], probability: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    location = (len(finite) - 1) * probability
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return finite[lower]
    weight = location - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _interval(values: Sequence[float]) -> list[float] | None:
    low = _percentile(values, 0.025)
    high = _percentile(values, 0.975)
    return None if low is None or high is None else [low, high]


def _flatten_selected(
    by_case: Mapping[str, Sequence[T]], selected_cases: Sequence[str]
) -> list[T]:
    return [value for case_id in selected_cases for value in by_case[case_id]]


def _cluster_bootstrap(
    by_case: Mapping[str, Sequence[T]],
    metric: Callable[[Sequence[T]], float | None],
    *,
    iterations: int,
    seed: int,
) -> list[float]:
    cases = sorted(by_case)
    if not cases:
        return []
    rng = random.Random(seed)
    result: list[float] = []
    for _ in range(iterations):
        selected = [rng.choice(cases) for _ in cases]
        value = metric(_flatten_selected(by_case, selected))
        if value is not None and math.isfinite(value):
            result.append(float(value))
    return result


def _ratio(
    values: Sequence[T],
    numerator: Callable[[T], bool],
    denominator: Callable[[T], bool] = lambda _: True,
) -> float | None:
    eligible = [value for value in values if denominator(value)]
    return None if not eligible else sum(numerator(value) for value in eligible) / len(eligible)


def _ratio_summary(
    by_case: Mapping[str, Sequence[T]],
    numerator: Callable[[T], bool],
    denominator: Callable[[T], bool] = lambda _: True,
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    values = _flatten_selected(by_case, sorted(by_case))
    eligible = [value for value in values if denominator(value)]
    count = sum(numerator(value) for value in eligible)
    metric = lambda rows: _ratio(rows, numerator, denominator)
    estimate = metric(values)
    bootstrap = _cluster_bootstrap(
        by_case, metric, iterations=iterations, seed=seed
    )
    return {
        "numerator": count,
        "denominator": len(eligible),
        "estimate": estimate,
        "patient_cluster_bootstrap_ci95": _interval(bootstrap),
        "valid_bootstrap_replicates": len(bootstrap),
    }


def _numeric_summary(
    by_case: Mapping[str, Sequence[float]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    values = [float(value) for value in _flatten_selected(by_case, sorted(by_case))]
    if not values:
        return {"count": 0, "mean": None, "median": None, "median_ci95": None}
    bootstrap = _cluster_bootstrap(
        by_case,
        lambda rows: statistics.median(float(value) for value in rows),
        iterations=iterations,
        seed=seed,
    )
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "patient_cluster_bootstrap_median_ci95": _interval(bootstrap),
        "valid_bootstrap_replicates": len(bootstrap),
    }


def _cohen_kappa(
    pairs: Sequence[tuple[object, object]],
    categories: Sequence[object],
    *,
    quadratic: bool = False,
) -> float | None:
    if not pairs:
        return None
    index = {value: position for position, value in enumerate(categories)}
    if any(left not in index or right not in index for left, right in pairs):
        raise ValueError("agreement pair is outside its frozen category vocabulary")
    count = len(pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    if quadratic:
        scale = max(1, len(categories) - 1)
        weight = lambda left, right: 1.0 - ((index[left] - index[right]) / scale) ** 2
    else:
        weight = lambda left, right: float(left == right)
    observed = sum(weight(left, right) for left, right in pairs) / count
    expected = sum(
        (left_counts[left] / count)
        * (right_counts[right] / count)
        * weight(left, right)
        for left in categories
        for right in categories
    )
    return None if math.isclose(expected, 1.0) else (observed - expected) / (1.0 - expected)


def _agreement_summary(
    by_case: Mapping[str, Sequence[tuple[object, object]]],
    categories: Sequence[object],
    *,
    quadratic: bool,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    pairs = _flatten_selected(by_case, sorted(by_case))
    raw = _ratio(pairs, lambda pair: pair[0] == pair[1])
    kappa = _cohen_kappa(pairs, categories, quadratic=quadratic)
    raw_boot = _cluster_bootstrap(
        by_case,
        lambda rows: _ratio(rows, lambda pair: pair[0] == pair[1]),
        iterations=iterations,
        seed=seed,
    )
    kappa_boot = _cluster_bootstrap(
        by_case,
        lambda rows: _cohen_kappa(rows, categories, quadratic=quadratic),
        iterations=iterations,
        seed=seed + 1,
    )
    return {
        "pair_count": len(pairs),
        "raw_agreement": raw,
        "raw_agreement_patient_cluster_ci95": _interval(raw_boot),
        "cohen_kappa": kappa,
        "cohen_kappa_patient_cluster_ci95": _interval(kappa_boot),
        "quadratic_weighting": quadratic,
    }


def _icc_a1(pairs: Sequence[tuple[float, float]]) -> float | None:
    n = len(pairs)
    k = 2
    if n < 2:
        return None
    matrix = [[float(left), float(right)] for left, right in pairs]
    row_means = [statistics.fmean(row) for row in matrix]
    column_means = [statistics.fmean(row[column] for row in matrix) for column in range(k)]
    grand = statistics.fmean(value for row in matrix for value in row)
    ms_rows = k * sum((value - grand) ** 2 for value in row_means) / (n - 1)
    ms_columns = n * sum((value - grand) ** 2 for value in column_means) / (k - 1)
    residual = sum(
        (matrix[row][column] - row_means[row] - column_means[column] + grand) ** 2
        for row in range(n)
        for column in range(k)
    )
    ms_error = residual / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return None if math.isclose(denominator, 0.0) else (ms_rows - ms_error) / denominator


def _icc_summary(
    by_case: Mapping[str, Sequence[tuple[float, float]]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    pairs = _flatten_selected(by_case, sorted(by_case))
    icc = _icc_a1(pairs)
    bootstrap = _cluster_bootstrap(
        by_case, _icc_a1, iterations=iterations, seed=seed
    )
    differences = [right - left for left, right in pairs]
    return {
        "pair_count": len(pairs),
        "icc_a_1_absolute_agreement": icc,
        "patient_cluster_bootstrap_ci95": _interval(bootstrap),
        "reader_b_minus_reader_a_median": (
            statistics.median(differences) if differences else None
        ),
    }


def _pack_state(pack: Path) -> dict[str, object]:
    manifest = _read_json(pack / "manifest.json")
    if manifest.get("schema_version") != PACK_SCHEMA:
        raise ValueError("reader-study manifest schema drifted")
    access = manifest.get("access_receipt")
    if not isinstance(access, Mapping) or any(
        access.get(field) is not False
        for field in (
            "deepsoz_target_values_loaded",
            "private_eeg_or_target_loaded",
            "tusz_channel_time_target_values_loaded",
            "model_correctness_or_outcome_metrics_loaded",
            "training_calibration_or_model_selection_performed",
            "llm_annotation_performed",
        )
    ):
        raise ValueError("reader-study target-free access contract failed")
    cards = _unique(_read_jsonl(pack / "report_cards.jsonl"), "case_id", name="report card")
    readers = {
        role: _unique(
            _read_jsonl(pack / f"{role}_annotations.jsonl"),
            "case_id",
            name=f"{role} annotation",
        )
        for role in ("reader_a", "reader_b")
    }
    if any(set(rows) != set(cards) for rows in readers.values()):
        raise ValueError("reader annotation rosters differ from report cards")
    linkage = _csv_rows(pack / "case_linkage.csv")
    linkage_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    patient_case: dict[str, str] = {}
    for row in linkage:
        case_id = row.get("case_id", "")
        patient_id = row.get("public_patient_id", "")
        if case_id not in cards or not patient_id:
            raise ValueError("reader linkage contains unknown/missing identity")
        if case_id in patient_case and patient_case[case_id] != patient_id:
            raise ValueError("one reader case maps to multiple patients")
        patient_case[case_id] = patient_id
        linkage_by_case[case_id].append(row)
    if set(linkage_by_case) != set(cards) or len(set(patient_case.values())) != len(cards):
        raise ValueError("reader cases are not one-to-one with unique patients")
    expected_events = {
        case_id: {f"{case_id}-E{index:03d}" for index in range(1, len(rows) + 1)}
        for case_id, rows in linkage_by_case.items()
    }
    return {
        "manifest": manifest,
        "cards": cards,
        "readers": readers,
        "expected_events": expected_events,
    }


def _completion_receipt(state: Mapping[str, object]) -> dict[str, object]:
    cards = state["cards"]
    readers = state["readers"]
    if not isinstance(cards, Mapping) or not isinstance(readers, Mapping):
        raise TypeError("reader pack state is invalid")
    by_role: dict[str, object] = {}
    completed_all = True
    for role in ("reader_a", "reader_b"):
        rows = readers[role]
        if not isinstance(rows, Mapping):
            raise TypeError("reader rows are invalid")
        completed = [case_id for case_id, row in rows.items() if row.get("review_status") == "completed"]
        raw_locked = [case_id for case_id, row in rows.items() if row.get("raw_phase_locked") is True]
        by_role[role] = {
            "total": len(rows),
            "raw_phase_locked": len(raw_locked),
            "completed": len(completed),
            "pending": len(rows) - len(completed),
        }
        completed_all &= len(completed) == len(rows)
    return {
        "all_independent_annotations_completed": completed_all,
        "by_role": by_role,
        "case_count": len(cards),
    }


def _validate_all_completed(state: Mapping[str, object]) -> None:
    cards = state["cards"]
    readers = state["readers"]
    expected_events = state["expected_events"]
    if not all(isinstance(value, Mapping) for value in (cards, readers, expected_events)):
        raise TypeError("reader pack state is invalid")
    reviewer_ids: dict[str, set[str]] = defaultdict(set)
    for role in ("reader_a", "reader_b"):
        for case_id, row in readers[role].items():
            if row.get("review_status") != "completed":
                raise ValueError("independent reader annotations are incomplete")
            _validate_completed(
                row,
                cards[case_id],
                expected_events=set(expected_events[case_id]),
            )
            reviewer = row.get("reviewer_id")
            if not isinstance(reviewer, str) or not reviewer:
                raise ValueError("completed annotation lacks reviewer identity")
            reviewer_ids[role].add(reviewer)
    if len(reviewer_ids["reader_a"]) != 1 or len(reviewer_ids["reader_b"]) != 1:
        raise ValueError("each independent role must use exactly one reviewer identity")
    if reviewer_ids["reader_a"] == reviewer_ids["reader_b"]:
        raise ValueError("reader A and reader B must be distinct clinicians")


def _clause_metrics(
    state: Mapping[str, object], *, iterations: int, seed: int
) -> tuple[dict[str, object], dict[str, Sequence[Mapping[str, object]]]]:
    cards = state["cards"]
    readers = state["readers"]
    observations: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_group_case: dict[tuple[str, str], dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    signal_by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
    for role in ("reader_a", "reader_b"):
        for case_id, row in readers[role].items():
            layer = str(row["layer"])
            for rating in row["clause_ratings"]:
                value = dict(rating)
                value["reader_role"] = role
                value["case_id"] = case_id
                clause_type = str(value["clause_type"])
                observations[case_id].append(value)
                by_group_case[(layer, clause_type)][case_id].append(value)
                if layer == "event_clause_factuality" and clause_type in SIGNAL_FACT_CLAUSE_TYPES:
                    signal_by_case[case_id].append(value)
    groups: dict[str, object] = {}
    for group_index, ((layer, clause_type), by_case) in enumerate(
        sorted(by_group_case.items()), start=1
    ):
        values = _flatten_selected(by_case, sorted(by_case))
        support_counts = Counter(str(value["support"]) for value in values)
        by_reader_case: dict[str, dict[str, list[dict[str, object]]]] = {
            role: defaultdict(list) for role in ("reader_a", "reader_b")
        }
        for case_id, case_values in by_case.items():
            for value in case_values:
                by_reader_case[str(value["reader_role"])][case_id].append(value)
        groups[f"{layer}/{clause_type}"] = {
            "case_count": len(by_case),
            "reader_clause_count": len(values),
            "support_counts": dict(sorted(support_counts.items())),
            "strict_supported_precision": _ratio_summary(
                by_case,
                lambda value: value["support"] == "supported",
                lambda value: value["support"] != "not_assessable",
                iterations=iterations,
                seed=seed + 10 * group_index,
            ),
            "unsupported_clause_rate": _ratio_summary(
                by_case,
                lambda value: value["support"] == "unsupported",
                lambda value: value["support"] != "not_assessable",
                iterations=iterations,
                seed=seed + 10 * group_index + 1,
            ),
            "clinically_material_error_rate": _ratio_summary(
                by_case,
                lambda value: value["clinically_material_error"] is True,
                lambda value: value["support"] != "not_assessable",
                iterations=iterations,
                seed=seed + 10 * group_index + 2,
            ),
            "assessable_coverage": _ratio_summary(
                by_case,
                lambda value: value["support"] != "not_assessable",
                iterations=iterations,
                seed=seed + 10 * group_index + 3,
            ),
            "proposed_action_counts": dict(
                sorted(Counter(str(value["proposed_action"]) for value in values).items())
            ),
            "by_reader": {
                role: {
                    "strict_supported_precision": _ratio_summary(
                        reader_cases,
                        lambda value: value["support"] == "supported",
                        lambda value: value["support"] != "not_assessable",
                        iterations=iterations,
                        seed=seed + 1000 + 20 * group_index + role_index,
                    ),
                    "unsupported_clause_rate": _ratio_summary(
                        reader_cases,
                        lambda value: value["support"] == "unsupported",
                        lambda value: value["support"] != "not_assessable",
                        iterations=iterations,
                        seed=seed + 2000 + 20 * group_index + role_index,
                    ),
                    "clinically_material_error_rate": _ratio_summary(
                        reader_cases,
                        lambda value: value["clinically_material_error"] is True,
                        lambda value: value["support"] != "not_assessable",
                        iterations=iterations,
                        seed=seed + 3000 + 20 * group_index + role_index,
                    ),
                    "assessable_coverage": _ratio_summary(
                        reader_cases,
                        lambda value: value["support"] != "not_assessable",
                        iterations=iterations,
                        seed=seed + 4000 + 20 * group_index + role_index,
                    ),
                }
                for role_index, (role, reader_cases) in enumerate(
                    by_reader_case.items(), start=1
                )
            },
        }
    signal_by_reader_case: dict[str, dict[str, list[dict[str, object]]]] = {
        role: defaultdict(list) for role in ("reader_a", "reader_b")
    }
    for case_id, case_values in signal_by_case.items():
        for value in case_values:
            signal_by_reader_case[str(value["reader_role"])][case_id].append(value)
    signal_summary = {
        "included_clause_types": sorted(SIGNAL_FACT_CLAUSE_TYPES),
        "case_count": len(signal_by_case),
        "strict_supported_precision": _ratio_summary(
            signal_by_case,
            lambda value: value["support"] == "supported",
            lambda value: value["support"] != "not_assessable",
            iterations=iterations,
            seed=seed + 7001,
        ),
        "clinically_material_error_rate": _ratio_summary(
            signal_by_case,
            lambda value: value["clinically_material_error"] is True,
            lambda value: value["support"] != "not_assessable",
            iterations=iterations,
            seed=seed + 7002,
        ),
        "assessable_coverage": _ratio_summary(
            signal_by_case,
            lambda value: value["support"] != "not_assessable",
            iterations=iterations,
            seed=seed + 7003,
        ),
        "by_reader": {
            role: {
                "strict_supported_precision": _ratio_summary(
                    reader_cases,
                    lambda value: value["support"] == "supported",
                    lambda value: value["support"] != "not_assessable",
                    iterations=iterations,
                    seed=seed + 7100 + role_index,
                ),
                "clinically_material_error_rate": _ratio_summary(
                    reader_cases,
                    lambda value: value["clinically_material_error"] is True,
                    lambda value: value["support"] != "not_assessable",
                    iterations=iterations,
                    seed=seed + 7200 + role_index,
                ),
                "assessable_coverage": _ratio_summary(
                    reader_cases,
                    lambda value: value["support"] != "not_assessable",
                    iterations=iterations,
                    seed=seed + 7300 + role_index,
                ),
            }
            for role_index, (role, reader_cases) in enumerate(
                signal_by_reader_case.items(), start=1
            )
        },
    }
    return {"by_layer_and_clause_type": groups, "signal_fact_primary": signal_summary}, observations


def _report_metrics(
    state: Mapping[str, object], *, iterations: int, seed: int
) -> dict[str, object]:
    readers = state["readers"]
    cards = state["cards"]
    result: dict[str, object] = {}
    for layer_index, layer in enumerate(LAYERS, start=1):
        rows_by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
        for role in ("reader_a", "reader_b"):
            for case_id, row in readers[role].items():
                if row["layer"] == layer:
                    value = dict(row)
                    value["reader_role"] = role
                    rows_by_case[case_id].append(value)
        raw_time = {
            case_id: [float(row["raw_only_review_duration_sec"]) for row in rows]
            for case_id, rows in rows_by_case.items()
        }
        report_time = {
            case_id: [float(row["report_review_duration_sec"]) for row in rows]
            for case_id, rows in rows_by_case.items()
        }
        total_time = {
            case_id: [
                float(row["raw_only_review_duration_sec"])
                + float(row["report_review_duration_sec"])
                for row in rows
            ]
            for case_id, rows in rows_by_case.items()
        }
        layer_result: dict[str, object] = {
            "case_count": len(rows_by_case),
            "reader_case_count": sum(len(rows) for rows in rows_by_case.values()),
            "safe_without_edit": _ratio_summary(
                rows_by_case,
                lambda row: row["safe_without_edit"] is True,
                iterations=iterations,
                seed=seed + 100 * layer_index,
            ),
            "important_omission": _ratio_summary(
                rows_by_case,
                lambda row: row["important_omission"] is True,
                iterations=iterations,
                seed=seed + 100 * layer_index + 1,
            ),
            "overstatement_present": _ratio_summary(
                rows_by_case,
                lambda row: row["overstatement_present"] is True,
                iterations=iterations,
                seed=seed + 100 * layer_index + 2,
            ),
            "modification_count": _numeric_summary(
                {
                    case_id: [float(row["overall_modification_count"]) for row in rows]
                    for case_id, rows in rows_by_case.items()
                },
                iterations=iterations,
                seed=seed + 100 * layer_index + 3,
            ),
            "raw_only_review_duration_sec": _numeric_summary(
                raw_time,
                iterations=iterations,
                seed=seed + 100 * layer_index + 4,
            ),
            "report_review_duration_sec": _numeric_summary(
                report_time,
                iterations=iterations,
                seed=seed + 100 * layer_index + 5,
            ),
            "total_review_duration_sec": _numeric_summary(
                total_time,
                iterations=iterations,
                seed=seed + 100 * layer_index + 6,
            ),
        }
        layer_result["by_reader"] = {
            role: {
                "safe_without_edit": _ratio_summary(
                    {
                        case_id: [row]
                        for case_id, rows in rows_by_case.items()
                        for row in rows
                        if row["reader_role"] == role
                    },
                    lambda row: row["safe_without_edit"] is True,
                    iterations=iterations,
                    seed=seed + 300 * layer_index + role_index,
                ),
                "important_omission": _ratio_summary(
                    {
                        case_id: [row]
                        for case_id, rows in rows_by_case.items()
                        for row in rows
                        if row["reader_role"] == role
                    },
                    lambda row: row["important_omission"] is True,
                    iterations=iterations,
                    seed=seed + 400 * layer_index + role_index,
                ),
                "overstatement_present": _ratio_summary(
                    {
                        case_id: [row]
                        for case_id, rows in rows_by_case.items()
                        for row in rows
                        if row["reader_role"] == role
                    },
                    lambda row: row["overstatement_present"] is True,
                    iterations=iterations,
                    seed=seed + 500 * layer_index + role_index,
                ),
            }
            for role_index, role in enumerate(("reader_a", "reader_b"), start=1)
        }
        strata: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for case_id, rows in rows_by_case.items():
            action = str(cards[case_id]["candidate_display_action"])
            strata[f"candidate_display_action={action}"][case_id].extend(rows)
            if layer == "event_clause_factuality":
                report_status = str(cards[case_id]["report_status"])
                phenotype = (
                    "reportable"
                    if "event_phenotype_reportable" in report_status
                    else "abstained"
                )
                strata[f"event_phenotype_action={phenotype}"][case_id].extend(rows)
        layer_result["prespecified_strata"] = {
            stratum: {
                "case_count": len(by_case),
                "reader_case_count": sum(len(rows) for rows in by_case.values()),
                "safe_without_edit": _ratio_summary(
                    by_case,
                    lambda row: row["safe_without_edit"] is True,
                    iterations=iterations,
                    seed=seed + 6000 + 10 * stratum_index,
                ),
                "important_omission": _ratio_summary(
                    by_case,
                    lambda row: row["important_omission"] is True,
                    iterations=iterations,
                    seed=seed + 6001 + 10 * stratum_index,
                ),
                "overstatement_present": _ratio_summary(
                    by_case,
                    lambda row: row["overstatement_present"] is True,
                    iterations=iterations,
                    seed=seed + 6002 + 10 * stratum_index,
                ),
            }
            for stratum_index, (stratum, by_case) in enumerate(
                sorted(strata.items()), start=1
            )
        }
        if layer == "patient_candidate_utility":
            display_by_case = {
                case_id: rows
                for case_id, rows in rows_by_case.items()
                if cards[case_id]["candidate_display_action"] == "display_candidate"
            }
            abstain_by_case = {
                case_id: rows
                for case_id, rows in rows_by_case.items()
                if cards[case_id]["candidate_display_action"] == "localization_abstain"
            }
            usefulness = {
                case_id: [float(row["candidate_review_usefulness_likert_1_to_5"]) for row in rows]
                for case_id, rows in rows_by_case.items()
            }
            consistency = {
                case_id: [float(row["candidate_eeg_consistency_likert_1_to_5"]) for row in rows]
                for case_id, rows in display_by_case.items()
            }
            action_change = lambda row: row["raw_only_candidate_action"] != row[
                "candidate_action_after_report"
            ]
            channel_jaccard: dict[str, list[float]] = defaultdict(list)
            for case_id, rows in rows_by_case.items():
                for row in rows:
                    before = set(row["raw_only_candidate_channels"])
                    after = set(row["candidate_channels_after_report"])
                    if before or after:
                        channel_jaccard[case_id].append(
                            len(before & after) / len(before | after)
                        )
            layer_result["candidate_utility"] = {
                "usefulness_likert": _numeric_summary(
                    usefulness,
                    iterations=iterations,
                    seed=seed + 901,
                ),
                "usefulness_at_least_4": _ratio_summary(
                    rows_by_case,
                    lambda row: row["candidate_review_usefulness_likert_1_to_5"] >= 4,
                    iterations=iterations,
                    seed=seed + 902,
                ),
                "display_candidate_eeg_consistency_likert": _numeric_summary(
                    consistency,
                    iterations=iterations,
                    seed=seed + 903,
                ),
                "display_candidate_burden_acceptable": _ratio_summary(
                    display_by_case,
                    lambda row: row["candidate_burden_acceptable"] is True,
                    iterations=iterations,
                    seed=seed + 904,
                ),
                "abstention_display_appropriate": _ratio_summary(
                    abstain_by_case,
                    lambda row: row["abstention_display_appropriate"] == "yes",
                    iterations=iterations,
                    seed=seed + 905,
                ),
                "raw_to_report_action_change": _ratio_summary(
                    rows_by_case,
                    action_change,
                    iterations=iterations,
                    seed=seed + 906,
                ),
                "raw_to_report_candidate_channel_jaccard": _numeric_summary(
                    channel_jaccard,
                    iterations=iterations,
                    seed=seed + 907,
                ),
                "interpretation": (
                    "perceived_eeg_compatibility_and_workflow_utility_only_not_soz_accuracy"
                ),
            }
        result[layer] = layer_result
    return result


def _agreement_metrics(
    state: Mapping[str, object], *, iterations: int, seed: int
) -> dict[str, object]:
    readers = state["readers"]
    cards = state["cards"]
    support_by_case: dict[str, list[tuple[object, object]]] = defaultdict(list)
    material_by_case: dict[str, list[tuple[object, object]]] = defaultdict(list)
    report_pairs: dict[str, dict[str, list[tuple[object, object]]]] = {
        field: defaultdict(list)
        for field in ("safe_without_edit", "important_omission", "overstatement_present")
    }
    usefulness_by_case: dict[str, list[tuple[object, object]]] = defaultdict(list)
    consistency_by_case: dict[str, list[tuple[object, object]]] = defaultdict(list)
    raw_time: dict[str, list[tuple[float, float]]] = defaultdict(list)
    report_time: dict[str, list[tuple[float, float]]] = defaultdict(list)
    total_time: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for case_id in sorted(cards):
        left = readers["reader_a"][case_id]
        right = readers["reader_b"][case_id]
        left_ratings = {rating["clause_id"]: rating for rating in left["clause_ratings"]}
        right_ratings = {rating["clause_id"]: rating for rating in right["clause_ratings"]}
        if set(left_ratings) != set(right_ratings):
            raise ValueError("reader clause rosters differ")
        for clause_id in sorted(left_ratings):
            left_rating, right_rating = left_ratings[clause_id], right_ratings[clause_id]
            support_by_case[case_id].append(
                (left_rating["support"], right_rating["support"])
            )
            if (
                left_rating["support"] != "not_assessable"
                and right_rating["support"] != "not_assessable"
            ):
                material_by_case[case_id].append(
                    (
                        left_rating["clinically_material_error"],
                        right_rating["clinically_material_error"],
                    )
                )
        for field, by_case in report_pairs.items():
            by_case[case_id].append((left[field], right[field]))
        raw_pair = (
            float(left["raw_only_review_duration_sec"]),
            float(right["raw_only_review_duration_sec"]),
        )
        report_pair = (
            float(left["report_review_duration_sec"]),
            float(right["report_review_duration_sec"]),
        )
        raw_time[case_id].append(raw_pair)
        report_time[case_id].append(report_pair)
        total_time[case_id].append(
            (raw_pair[0] + report_pair[0], raw_pair[1] + report_pair[1])
        )
        if left["layer"] == "patient_candidate_utility":
            usefulness_by_case[case_id].append(
                (
                    left["candidate_review_usefulness_likert_1_to_5"],
                    right["candidate_review_usefulness_likert_1_to_5"],
                )
            )
            if cards[case_id]["candidate_display_action"] == "display_candidate":
                consistency_by_case[case_id].append(
                    (
                        left["candidate_eeg_consistency_likert_1_to_5"],
                        right["candidate_eeg_consistency_likert_1_to_5"],
                    )
                )
    result = {
        "clause_support": _agreement_summary(
            support_by_case,
            SUPPORT_CATEGORIES,
            quadratic=False,
            iterations=iterations,
            seed=seed + 1,
        ),
        "clinically_material_error": _agreement_summary(
            material_by_case,
            (False, True),
            quadratic=False,
            iterations=iterations,
            seed=seed + 2,
        ),
        "report_level_binary": {
            field: _agreement_summary(
                by_case,
                (False, True),
                quadratic=False,
                iterations=iterations,
                seed=seed + 10 + index,
            )
            for index, (field, by_case) in enumerate(report_pairs.items())
        },
        "candidate_usefulness_likert": _agreement_summary(
            usefulness_by_case,
            (1, 2, 3, 4, 5),
            quadratic=True,
            iterations=iterations,
            seed=seed + 30,
        ),
        "display_candidate_eeg_consistency_likert": _agreement_summary(
            consistency_by_case,
            (1, 2, 3, 4, 5),
            quadratic=True,
            iterations=iterations,
            seed=seed + 31,
        ),
        "review_time_icc": {
            "raw_only": _icc_summary(
                raw_time, iterations=iterations, seed=seed + 40
            ),
            "report_phase": _icc_summary(
                report_time, iterations=iterations, seed=seed + 41
            ),
            "total": _icc_summary(
                total_time, iterations=iterations, seed=seed + 42
            ),
        },
    }
    return result


def _qualification(
    clause_metrics: Mapping[str, object], report_metrics: Mapping[str, object]
) -> dict[str, object]:
    signal = clause_metrics["signal_fact_primary"]
    strict_ci = signal["strict_supported_precision"]["patient_cluster_bootstrap_ci95"]
    material_ci = signal["clinically_material_error_rate"]["patient_cluster_bootstrap_ci95"]
    gates: dict[str, dict[str, object]] = {
        "signal_fact_supported_precision": {
            "rule": "patient_cluster_ci95_lower_greater_than_or_equal_0.90",
            "observed_interval": strict_ci,
            "passed": bool(strict_ci is not None and strict_ci[0] >= 0.90),
        },
        "signal_fact_material_error": {
            "rule": "patient_cluster_ci95_upper_less_than_or_equal_0.05",
            "observed_interval": material_ci,
            "passed": bool(material_ci is not None and material_ci[1] <= 0.05),
        },
    }
    for layer in LAYERS:
        safe_ci = report_metrics[layer]["safe_without_edit"][
            "patient_cluster_bootstrap_ci95"
        ]
        omission_ci = report_metrics[layer]["important_omission"][
            "patient_cluster_bootstrap_ci95"
        ]
        gates[f"{layer}_safe_without_edit"] = {
            "rule": "patient_cluster_ci95_lower_greater_than_or_equal_0.80",
            "observed_interval": safe_ci,
            "passed": bool(safe_ci is not None and safe_ci[0] >= 0.80),
        }
        gates[f"{layer}_important_omission"] = {
            "rule": "patient_cluster_ci95_upper_less_than_or_equal_0.10",
            "observed_interval": omission_ci,
            "passed": bool(omission_ci is not None and omission_ci[1] <= 0.10),
        }
    overall = all(gate["passed"] for gate in gates.values())
    return {
        "gates": gates,
        "all_prespecified_report_qualification_gates_passed": overall,
        "allowed_claim_if_passed": (
            "specified_report_clauses_passed_prespecified_two_reader_factuality_gates"
        ),
        "forbidden_inference": (
            "reader_qualification_does_not_establish_soz_accuracy_score_faithfulness_or_cortical_truth"
        ),
    }


def audit_reader_study(
    *,
    reader_pack: Path,
    output_path: Path,
    preflight: bool,
    bootstrap_iterations: int = 5000,
    seed: int = 20260815,
) -> dict[str, object]:
    if bootstrap_iterations < 1000:
        raise ValueError("reader-study audit requires at least 1000 bootstrap iterations")
    state = _pack_state(reader_pack)
    completion = _completion_receipt(state)
    base: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA,
        "reader_pack_status": state["manifest"].get("status"),
        "completion": completion,
        "bootstrap_contract": {
            "unit": "patient_case",
            "both_readers_and_all_case_clauses_resampled_together": True,
            "iterations": bootstrap_iterations,
            "seed": seed,
            "interval": "percentile_95",
        },
        "access_receipt": {
            "reader_annotations_loaded": True,
            "sealed_report_cards_loaded": True,
            "data_manager_linkage_loaded_for_one_patient_per_case_validation_only": True,
            "raw_eeg_loaded": False,
            "deepsoz_or_private_target_loaded": False,
            "tusz_channel_time_annotation_loaded": False,
            "model_score_margin_or_correctness_loaded": False,
            "training_calibration_model_selection_or_report_rewrite_performed": False,
            "llm_used": False,
        },
    }
    if preflight:
        base.update(
            {
                "status": (
                    "independent_clinician_annotations_complete_audit_not_run"
                    if completion["all_independent_annotations_completed"]
                    else "pending_independent_clinician_annotations"
                ),
                "outcome_metrics_computed": False,
                "qualification_decision": "not_evaluable",
            }
        )
        _write_json(output_path, base)
        return base
    if not completion["all_independent_annotations_completed"]:
        raise ValueError(
            "independent reader annotations are incomplete; use --preflight for a pending receipt"
        )
    _validate_all_completed(state)
    clause_metrics, _ = _clause_metrics(
        state, iterations=bootstrap_iterations, seed=seed
    )
    report_metrics = _report_metrics(
        state, iterations=bootstrap_iterations, seed=seed + 10000
    )
    agreement = _agreement_metrics(
        state, iterations=bootstrap_iterations, seed=seed + 20000
    )
    qualification = _qualification(clause_metrics, report_metrics)
    base.update(
        {
            "status": "completed_two_reader_target_blind_report_audit",
            "outcome_metrics_computed": True,
            "clause_metrics": clause_metrics,
            "report_metrics": report_metrics,
            "reader_agreement": agreement,
            "qualification": qualification,
            "scientific_boundary": {
                "event_clause_factuality_is_not_soz_accuracy": True,
                "candidate_usefulness_is_not_candidate_correctness": True,
                "reader_entered_candidates_are_not_gold_or_training_labels": True,
                "no_report_time_saving_claim_without_randomized_control": True,
                "raw_independent_reads_must_be_reported_alongside_any_future_adjudication": True,
            },
        }
    )
    _write_json(output_path, base)
    return base


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--reader-pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    result = audit_reader_study(
        reader_pack=args.reader_pack,
        output_path=args.output,
        preflight=args.preflight,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
