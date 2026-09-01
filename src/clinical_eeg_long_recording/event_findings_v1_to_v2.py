"""Explicit fail-closed migration from event Findings v1 to v2.

The adapter preserves replayable v1 numbers and identifiers, but does not
pretend that v1 recorded facts it did not contain.  In particular it cannot
recover causal/offline view roles, channel imputation provenance, S0--S3
posteriors, the expanded EEG-only input firewall, target-relative
counterevidence, four-state IFCN atoms, or v2 qualification metrics.  Those
fields become ``unknown``/``not_evaluable`` and all scalp-onset and clinical
surface authorization is withheld.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .event_finding_term_registry import EVENT_FINDING_TERM_FAMILIES
from .event_findings_validation import validate_event_eeg_findings_payload
from .event_findings_v2_validation import (
    validate_event_eeg_findings_v2_payload,
)


EVENT_FINDINGS_V1_TO_V2_MIGRATOR_ID = (
    "event_eeg_findings_v1_to_v2_fail_closed_v1"
)

_UNIT_MAP = {
    "ratio": "ratio",
    "unitless": "unitless",
    "hz": "hertz",
    "hertz": "hertz",
    "uv": "microvolt",
    "µv": "microvolt",
    "μv": "microvolt",
    "microvolt": "microvolt",
    "s": "second",
    "sec": "second",
    "second": "second",
    "seconds": "second",
    "ms": "millisecond",
    "millisecond": "millisecond",
    "milliseconds": "millisecond",
    "uv_per_sample": "microvolt_per_sample",
    "uv/sample": "microvolt_per_sample",
}

_FEATURE_FAMILIES = (
    "quality",
    "spectral",
    "rhythm",
    "morphology",
    "evolution",
    "spatial_field",
    "spatial_recruitment",
    "termination_recovery",
    "high_frequency",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _identifier(value: object, *, prefix: str = "ID") -> str:
    text = re.sub(r"[^A-Za-z0-9._:-]+", "-", str(value)).strip("-._:")
    if not text:
        text = prefix
    if not text[0].isalnum():
        text = f"{prefix}-{text}"
    return text[:256]


def _span(value: Mapping[str, object] | None) -> tuple[float, float] | None:
    if value is None:
        return None
    return float(value["start"]), float(value["stop"])


def _overlap_fraction(
    span: tuple[float, float] | None, zone: tuple[float, float]
) -> float:
    if span is None:
        return 0.0
    overlap = max(0.0, min(span[1], zone[1]) - max(span[0], zone[0]))
    return overlap / (span[1] - span[0])


def _normalize_time_interval(value: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        "lower": float(value["lower"]),
        "upper": float(value["upper"]),
        "resolution_seconds": float(value["resolution_seconds"]),
        "calibration_status": str(
            value.get("calibration_status", "legacy_unverified")
        ),
    }
    if "median" in value:
        result["median"] = float(value["median"])
    if "coverage" in value:
        result["coverage"] = float(value["coverage"])
    return result


def _legacy_boundary(
    value: Mapping[str, object],
    *,
    boundary_name: str,
) -> dict[str, object]:
    # A v1 interval proves that a legacy estimator emitted numbers.  It does
    # not prove v2's causal-view semantics, so observed/estimated v1
    # boundaries are deliberately indeterminate after migration.
    if "status" not in value:
        return {
            "status": "indeterminate",
            "interval": None,
            "censoring_reason_codes": [
                f"legacy_v1_{boundary_name}_semantics_not_recoverable"
            ],
        }
    status = str(value["status"])
    interval = value.get("interval")
    if status == "censored":
        return {
            "status": "censored",
            "interval": (
                _normalize_time_interval(interval)  # type: ignore[arg-type]
                if isinstance(interval, Mapping)
                else None
            ),
            "censoring_reason_codes": [f"legacy_v1_{boundary_name}_censored"],
        }
    if status == "not_observed":
        return {
            "status": "not_observed",
            "interval": None,
            "censoring_reason_codes": [],
        }
    return {
        "status": "indeterminate",
        "interval": None,
        "censoring_reason_codes": [
            f"legacy_v1_{boundary_name}_semantics_not_recoverable"
        ],
    }


def _legacy_interval_bounds(value: Mapping[str, object]) -> tuple[float, float] | None:
    interval: object
    if "status" in value:
        interval = value.get("interval")
    else:
        interval = value
    if not isinstance(interval, Mapping):
        return None
    return float(interval["lower"]), float(interval["upper"])


def _project_protection_zone(
    payload: Mapping[str, Any],
) -> tuple[float, float]:
    window = payload["window"]
    final_start, final_stop = (float(item) for item in window["final_interval"])
    onset = _legacy_interval_bounds(window["onset_interval"])
    offset = _legacy_interval_bounds(window["offset_interval"])
    lower = onset[0] if onset is not None else final_start
    upper = offset[1] if offset is not None else final_stop
    lower = max(final_start, lower)
    upper = min(final_stop, upper)
    if upper <= lower:
        return final_start, final_stop
    return lower, upper


def _term_ref(term: str) -> dict[str, str]:
    if term in {"spike", "sharp_wave"}:
        return {
            "term_id": term,
            "ontology_id": "IFCN_EEG_GLOSSARY",
            "source_id": "KANE_IFCN_2017",
            "source_version": "2017",
            "operational_rule_id": "legacy_v1_term_registry_projection",
        }
    if term == "interictal_epileptiform_discharge":
        return {
            "term_id": term,
            "ontology_id": "IFCN_IED_CRITERIA",
            "source_id": "KURAL_IFCN_VALIDATION_2020",
            "source_version": "2020",
            "operational_rule_id": "legacy_v1_term_registry_projection",
        }
    if term in {"definite_evolution", "electrographic_seizure"}:
        return {
            "term_id": term,
            "ontology_id": "ACNS_CRITICAL_CARE_EEG_TERMINOLOGY",
            "source_id": "HIRSCH_ACNS_2021",
            "source_version": "2021",
            "operational_rule_id": "legacy_v1_term_registry_projection",
        }
    return {
        "term_id": _identifier(term, prefix="TERM"),
        "ontology_id": "PROJECT_EVENT_FINDING_TERMS",
        "source_id": "EVENT_FINDING_TERM_REGISTRY",
        "source_version": "v1",
        "operational_rule_id": "legacy_v1_term_registry_projection",
    }


def _unit_id(value: object) -> tuple[str, str]:
    normalized = str(value).strip().lower()
    unit_id = _UNIT_MAP.get(normalized)
    if unit_id is None:
        return "unknown_unit", "unknown"
    return unit_id, "registered"


def _opportunity_from_finding(
    finding: Mapping[str, Any],
    *,
    usable_fraction: float,
) -> dict[str, Any]:
    evidence_id = str(finding["evidence_id"])
    measurements = finding["measurements"]
    source_view_ids = sorted(
        {
            str(row["source_binding"]["source_view_id"])
            for row in measurements
        }
    )
    bandwidths = {
        tuple(float(item) for item in row["source_binding"]["effective_bandwidth_hz"])
        for row in measurements
    }
    quality_hashes = {
        str(row["source_binding"]["quality_mask_sha256"])
        for row in measurements
    }
    not_evaluable = finding["status"] == "not_evaluable"
    return {
        "evaluation_opportunity_id": _identifier(f"OPP-{evidence_id}"),
        "family": str(finding["family"]),
        "term_id": _term_ref(str(finding["term"]))["term_id"],
        "interval": deepcopy(finding["time_interval"]) if not not_evaluable else None,
        "spatial_unit_keys": sorted(
            _identifier(f"{row['unit_type']}:{row['id']}")
            for row in finding["spatial_support"]
        ),
        "source_view_ids": source_view_ids,
        "status": "not_evaluable" if not_evaluable else "limited",
        "usable_fraction": 0.0 if not_evaluable else float(usable_fraction),
        "effective_bandwidth_hz": (
            list(next(iter(bandwidths))) if len(bandwidths) == 1 else None
        ),
        "quality_mask_sha256": (
            next(iter(quality_hashes)) if len(quality_hashes) == 1 else None
        ),
        "reason_codes": [
            "legacy_v1_not_evaluable"
            if not_evaluable
            else "legacy_v1_opportunity_sensitivity_not_qualified"
        ],
    }


def _placeholder_opportunity(
    *,
    family: str,
    status: str,
    final_interval: Sequence[object],
) -> dict[str, Any]:
    not_evaluable = status == "not_evaluable"
    return {
        "evaluation_opportunity_id": _identifier(f"OPP-FAMILY-{family}"),
        "family": family,
        "term_id": "family_evaluation",
        "interval": (
            None
            if not_evaluable
            else {
                "start": float(final_interval[0]),
                "stop": float(final_interval[1]),
                "resolution_seconds": 1.0,
            }
        ),
        "spatial_unit_keys": [],
        "source_view_ids": [],
        "status": "not_evaluable" if not_evaluable else "limited",
        "usable_fraction": 0.0,
        "effective_bandwidth_hz": None,
        "quality_mask_sha256": None,
        "reason_codes": [
            "legacy_v1_family_not_evaluable"
            if not_evaluable
            else "legacy_v1_family_opportunity_not_recoverable"
        ],
    }


def migrate_event_eeg_findings_v1_to_v2(
    value: object,
    *,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_term_decision_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
) -> dict[str, Any]:
    """Validate v1, migrate replayable fields, and validate a fail-closed v2.

    No optional argument can authorize a v2 clinical or SOZ claim.  The v1
    registries are accepted solely to validate the source payload before its
    legacy clinical receipts are intentionally discarded.
    """

    source = validate_event_eeg_findings_payload(
        value,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_term_decision_receipts=trusted_term_decision_receipts,
    )
    event_id = str(source["event_id"])
    signal_sha = str(source["provenance"]["signal_sha256"])
    usable_fraction = float(source["quality"]["usable_fraction"])
    final_interval = source["window"]["final_interval"]
    zone_start, zone_stop = _project_protection_zone(source)
    zone_id = _identifier(f"PZ-MIGRATED-{event_id}")
    loss_codes = {
        "legacy_v1_extended_input_firewall_unknown",
        "legacy_v1_observation_imputation_status_unknown",
        "legacy_v1_view_role_causality_unknown",
        "legacy_v1_raw_sample_dependency_unrecoverable",
        "legacy_v1_state_posteriors_unavailable",
        "legacy_v1_target_relative_counterevidence_unrecoverable",
        "legacy_v1_capability_metrics_unrecoverable",
        "legacy_v1_field_observation_details_unrecoverable",
        "legacy_v1_spatial_hypothesis_withheld",
        "legacy_v1_protection_zone_projected",
        "legacy_v1_registries_unverified",
    }
    if source["qualification_receipts"] or source["term_decision_receipts"]:
        loss_codes.add("legacy_v1_atomic_term_criteria_unrecoverable")

    term_registry_projection = {
        key: sorted(value)
        for key, value in sorted(EVENT_FINDING_TERM_FAMILIES.items())
    }
    unit_registry_projection = dict(sorted(_UNIT_MAP.items()))

    input_units: list[dict[str, Any]] = []
    for row in source["montage"]["input_units"]:
        available = bool(row["available"])
        migrated = {
            "unit_id": str(row["unit_id"]),
            "unit_type": str(row["unit_type"]),
            "canonical_name": str(row["canonical_name"]),
            "observation_status": "unknown" if available else "missing",
            "evidence_eligible": False,
            "missing_reason_codes": [
                "legacy_v1_observation_or_imputation_status_unknown"
                if available
                else "legacy_v1_input_unavailable"
            ],
            "imputation_receipt_id": None,
            "laterality": str(row["laterality"]),
        }
        if "source_name" in row:
            migrated["source_name"] = str(row["source_name"])
        if "region" in row:
            migrated["region"] = str(row["region"])
        input_units.append(migrated)

    opportunities = [
        _opportunity_from_finding(finding, usable_fraction=usable_fraction)
        for finding in source["findings"]
    ]
    opportunity_ids_by_family: dict[str, list[str]] = {
        family: [] for family in _FEATURE_FAMILIES
    }
    for opportunity in opportunities:
        opportunity_ids_by_family[str(opportunity["family"])].append(
            str(opportunity["evaluation_opportunity_id"])
        )

    v1_feature = {
        ("spatial_recruitment" if row["family"] == "recruitment" else str(row["family"])): row
        for row in source["quality"]["feature_availability"]
    }
    feature_evaluability: list[dict[str, Any]] = []
    for family in _FEATURE_FAMILIES:
        legacy = v1_feature.get(family)
        legacy_status = str(legacy["status"]) if legacy is not None else "limited"
        status = "not_evaluable" if legacy_status == "not_evaluable" else "limited"
        if not opportunity_ids_by_family[family]:
            placeholder = _placeholder_opportunity(
                family=family,
                status=status,
                final_interval=final_interval,
            )
            opportunities.append(placeholder)
            opportunity_ids_by_family[family].append(
                str(placeholder["evaluation_opportunity_id"])
            )
        reason_codes = set(
            str(item) for item in (legacy["reason_codes"] if legacy else [])
        )
        reason_codes.add(
            "legacy_v1_family_not_evaluable"
            if status == "not_evaluable"
            else "legacy_v1_family_opportunity_not_recoverable"
        )
        feature_evaluability.append(
            {
                "family": family,
                "status": status,
                "reason_codes": sorted(reason_codes),
                "evaluation_opportunity_ids": sorted(
                    opportunity_ids_by_family[family]
                ),
            }
        )

    migrated_findings: list[dict[str, Any]] = []
    for finding_index, finding in enumerate(source["findings"]):
        evidence_id = str(finding["evidence_id"])
        finding_span = _span(finding["time_interval"])
        overlap = _overlap_fraction(finding_span, (zone_start, zone_stop))
        if finding["status"] == "not_evaluable":
            role = "limitation"
            temporal_context = "unknown"
            ownership_status = "unknown"
            owner_event_ids: list[str] = []
        elif overlap <= 1e-6:
            role = "non_event_context"
            temporal_context = "outside_candidate_protection"
            ownership_status = "outside_protection"
            owner_event_ids = []
        else:
            role = {
                "onset_support": "early_context",
                "spread_support": "later_involvement",
                "contradiction": "limitation",
                "context_only": "early_context",
            }[str(finding["evidence_role"])]
            temporal_context = (
                "late_involvement"
                if finding["evidence_role"] == "spread_support"
                else "unknown"
            )
            ownership_status = "event_owned"
            owner_event_ids = [event_id]

        migrated_measurements: list[dict[str, Any]] = []
        for measurement_index, measurement in enumerate(finding["measurements"]):
            unit_id, unit_status = _unit_id(measurement["unit"])
            binding = measurement["source_binding"]
            migrated_measurements.append(
                {
                    "measurement_id": _identifier(
                        f"MEAS-{evidence_id}-{measurement_index + 1}"
                    ),
                    "name_id": _identifier(measurement["name"], prefix="MEASURE"),
                    "value": float(measurement["value"]),
                    "unit_id": unit_id,
                    "unit_registry_status": unit_status,
                    "baseline_delta": (
                        float(measurement["baseline_delta"])
                        if measurement.get("baseline_delta") is not None
                        else None
                    ),
                    "numerical_uncertainty": {
                        "status": "legacy_unknown",
                        "lower": None,
                        "upper": None,
                        "coverage": None,
                        "calibration_receipt_id": None,
                    },
                    "producer_type": "deterministic_signal_measurement",
                    "source_binding": {
                        "canonical_signal_sha256": signal_sha,
                        "source_view_id": str(binding["source_view_id"]),
                        "view_role": "unknown",
                        "view_receipt_id": str(binding["view_receipt_id"]),
                        "view_receipt_sha256": str(binding["view_receipt_sha256"]),
                        "transform_spec_sha256": str(binding["transform_spec_sha256"]),
                        "processed_view_sha256": str(binding["processed_view_sha256"]),
                        "source_unit_ids": [str(item) for item in binding["source_unit_ids"]],
                        "recording_interval": [float(item) for item in binding["recording_interval"]],
                        "tensor_sample_interval": [int(item) for item in binding["tensor_sample_interval"]],
                        "effective_bandwidth_hz": [float(item) for item in binding["effective_bandwidth_hz"]],
                        "reference_type": str(binding["reference_type"]),
                        "evidence_family": str(binding["evidence_family"]),
                        "quality_mask_sha256": str(binding["quality_mask_sha256"]),
                        "edge_mask_sha256": None,
                        "padding_mask_sha256": None,
                        "imputation_mask_sha256": None,
                        "evidence_eligible": False,
                        "ineligibility_reason_codes": [
                            "legacy_v1_view_role_and_imputation_unknown"
                        ],
                        "background_reference_ids": [
                            str(item) for item in binding["background_reference_ids"]
                        ],
                        "method_id": str(binding["method_id"]),
                        "policy_sha256": str(binding["policy_sha256"]),
                        # v1 did not preserve the raw-sample support, processing
                        # latency, or view-receipt ancestry needed to reconstruct
                        # a trustworthy dependency interval.  Keep this null;
                        # the v2 validator permits it only on a fail-closed
                        # migration and never as onset-positive evidence.
                        "raw_sample_dependency": None,
                    },
                }
            )

        migrated_findings.append(
            {
                "evidence_id": evidence_id,
                "finding_group_id": _identifier(f"GROUP-{evidence_id}"),
                "family": str(finding["family"]),
                "term": _term_ref(str(finding["term"])),
                "assertion_level": (
                    "model_candidate"
                    if finding["assertion_level"] == "clinically_qualified"
                    else str(finding["assertion_level"])
                ),
                "status": str(finding["status"]),
                "intrinsic_evidence_role": role,
                "signal_temporal_context": temporal_context,
                "ownership": {
                    "owner_event_ids": owner_event_ids,
                    "event_group_id": None,
                    "protection_zone_id": zone_id,
                    "ownership_status": ownership_status,
                    "protection_zone_overlap_fraction": overlap,
                },
                "state_membership": {"S0": 0.0, "S1": 0.0, "S2": 0.0, "S3": 0.0},
                "time_interval": deepcopy(finding["time_interval"]),
                "spatial_support": [
                    {
                        "unit_type": str(row["unit_type"]),
                        "id": str(row["id"]),
                        "mapping_status": (
                            "candidate_only"
                            if row["mapping_status"] == "field_qualified"
                            else str(row["mapping_status"])
                        ),
                        "observation_status": "unknown",
                        "evidence_eligible": False,
                        "missing_reason_codes": [
                            "legacy_v1_spatial_observation_provenance_unknown"
                        ],
                        "support_score": (
                            float(row["support_score"])
                            if "support_score" in row
                            else None
                        ),
                        "field_observation": None,
                    }
                    for row in finding["spatial_support"]
                ],
                "measurements": migrated_measurements,
                "uncertainty": deepcopy(finding["uncertainty"]),
                "evaluation_opportunity_id": _identifier(f"OPP-{evidence_id}"),
                "capability_receipt_id": None,
                "sensitivity_receipt_id": None,
                "term_decision_receipt_id": None,
                "waveform_evidence_ids": [
                    str(item) for item in finding["waveform_evidence_ids"]
                ],
                "raw_sample_dependency_ids": [],
            }
        )

    # Per-unit v1 intervals cannot be promoted: causal-view identity and
    # four-state opportunity/sensitivity are absent.  Keep only the spatial
    # inventory and explicit not-evaluable state.
    per_unit_involvement: list[dict[str, Any]] = []
    for row_index, row in enumerate(source["spatial_onset"]["per_unit_intervals"]):
        opportunity_id = _identifier(
            f"OPP-UNIT-{row['unit_type']}-{row['unit_id']}-{row_index + 1}"
        )
        opportunity = {
            "evaluation_opportunity_id": opportunity_id,
            "family": "spatial_recruitment",
            "term_id": "unit_involvement",
            "interval": None,
            "spatial_unit_keys": [
                _identifier(f"{row['unit_type']}:{row['unit_id']}")
            ],
            "source_view_ids": [],
            "status": "not_evaluable",
            "usable_fraction": 0.0,
            "effective_bandwidth_hz": None,
            "quality_mask_sha256": None,
            "reason_codes": ["legacy_v1_causal_unit_onset_not_recoverable"],
        }
        opportunities.append(opportunity)
        opportunity_ids_by_family["spatial_recruitment"].append(opportunity_id)
        per_unit_involvement.append(
            {
                "unit_type": str(row["unit_type"]),
                "unit_id": str(row["unit_id"]),
                "status": "not_evaluable",
                "interval": None,
                "evaluation_opportunity_id": opportunity_id,
                "sensitivity_receipt_id": None,
                "evidence_ids": [],
            }
        )
    for row in feature_evaluability:
        if row["family"] == "spatial_recruitment":
            row["evaluation_opportunity_ids"] = sorted(
                opportunity_ids_by_family["spatial_recruitment"]
            )

    # Remove any legacy background interval that intersects the projected
    # event protection zone.  Such a segment cannot simultaneously be a v2
    # background prototype.  The removal is recorded rather than hidden.
    def keep_background(interval: Sequence[object]) -> bool:
        start, stop = float(interval[0]), float(interval[1])
        return max(start, zone_start) >= min(stop, zone_stop) - 1e-6

    local_background = [
        [float(item) for item in row]
        for row in source["context"]["local_background_intervals"]
        if keep_background(row)
    ]
    distant_background = [
        [float(item) for item in row]
        for row in source["context"]["distant_background_intervals"]
        if keep_background(row)
    ]
    original_background_count = len(source["context"]["local_background_intervals"]) + len(source["context"]["distant_background_intervals"])
    if len(local_background) + len(distant_background) != original_background_count:
        loss_codes.add("legacy_v1_background_overlapping_projected_protection_removed")
    has_background = bool(local_background or distant_background)

    migrated_waveforms = [
        {
            "waveform_evidence_id": str(row["waveform_evidence_id"]),
            "interval": [float(item) for item in row["interval"]],
            "unit_ids": [str(item) for item in row["unit_ids"]],
            "source_view_id": "LEGACY-VIEW-UNKNOWN",
            "view_role": "unknown",
            "view_receipt_id": None,
            "view_receipt_sha256": None,
            "processed_view_sha256": None,
            "quality_mask_sha256": None,
            "evidence_eligible": False,
            "ineligibility_reason_codes": [
                "legacy_v1_waveform_view_binding_not_recorded"
            ],
            "render_policy": str(row["render_policy"]),
            "canonical_signal_sha256": signal_sha,
            "raw_sample_dependency": None,
        }
        for row in source["waveform_evidence"]
    ]

    limitations = [deepcopy(row) for row in source["limitations"]]
    if not any(row["code"] == "legacy_v1_migration_loss" for row in limitations):
        limitations.append(
            {
                "code": "legacy_v1_migration_loss",
                "scope": "migration",
                "text_zh": "该事件由旧版证据合同保守迁移；无法恢复的视图因果性、状态后验、插补来源、术语四态资格和目标相对证据均不用于临床措辞或头皮起始定位。",
            }
        )

    result: dict[str, Any] = {
        "schema_version": "event_eeg_findings_v2",
        "event_id": event_id,
        "provenance": {
            "record_id": str(source["provenance"]["record_id"]),
            "canonical_signal_sha256": signal_sha,
            "preprocess_receipt_id": str(source["provenance"]["preprocess_receipt_id"]),
            "model_ids": [str(item) for item in source["provenance"]["model_ids"]],
            "policy_sha256": str(source["provenance"]["policy_sha256"]),
            "inference_exclusions": {
                "edf_annotations_used": False,
                "excel_used": False,
                "doctor_labels_used": False,
                "clinical_text_used": False,
                "patient_metadata_used": "unknown",
                "video_used": "unknown",
                "ecg_emg_eog_used": "unknown",
                "sleep_staging_used": "unknown",
                "provocation_used": "unknown",
            },
        },
        "coordinates": deepcopy(source["coordinates"]),
        "registry_bindings": {
            "term_registry": {
                "registry_id": "EVENT-FINDING-TERM-REGISTRY-V1-PROJECTION",
                "version": "v1",
                "registry_sha256": _sha256(term_registry_projection),
                "trust_status": "legacy_unverified",
            },
            "unit_registry": {
                "registry_id": "EVENT-FINDING-UNIT-MAP-V1-PROJECTION",
                "version": "v1",
                "registry_sha256": _sha256(unit_registry_projection),
                "trust_status": "legacy_unverified",
            },
        },
        "montage": {
            "analysis_reference": str(source["montage"]["analysis_reference"]),
            "input_units": input_units,
            "electrode_ids": [str(item) for item in source["montage"]["electrode_ids"]],
            "lead_definitions": deepcopy(source["montage"]["lead_definitions"]),
            "reference_perturbations_evaluated": [
                str(item)
                for item in source["montage"].get(
                    "reference_perturbations_evaluated", []
                )
            ],
        },
        "window": {
            "search_interval": [float(item) for item in source["window"]["search_interval"]],
            "final_interval": [float(item) for item in final_interval],
            "protection_zone": {
                "protection_zone_id": zone_id,
                "interval": [zone_start, zone_stop],
                "policy_sha256": str(source["provenance"]["policy_sha256"]),
            },
            "onset_boundary": _legacy_boundary(
                source["window"]["onset_interval"], boundary_name="onset"
            ),
            "offset_boundary": _legacy_boundary(
                source["window"]["offset_interval"], boundary_name="offset"
            ),
            "state_posterior_status": "not_evaluable",
            "state_segments": [],
            "state_path_receipt_id": None,
            "left_censored": bool(source["window"]["left_censored"]),
            "right_censored": bool(source["window"]["right_censored"]),
            "search_cap_censored": bool(source["window"]["search_cap_censored"]),
            "merge_split_status": str(source["window"]["merge_split_status"]),
        },
        "context": {
            "queried_intervals": [
                [float(item) for item in row]
                for row in source["context"]["queried_intervals"]
            ],
            "local_background_intervals": local_background,
            "distant_background_intervals": distant_background,
            "background_status": (
                str(source["context"]["background_status"])
                if has_background
                else "unavailable"
            ),
            "background_bank_id": (
                source["context"]["background_bank_id"] if has_background else None
            ),
            "selection_receipt_id": (
                source["context"]["selection_receipt_id"] if has_background else None
            ),
            "selection_scope": "eeg_detector_quality_only",
            "contamination_risk": (
                float(source["context"]["contamination_risk"])
                if has_background
                else 1.0
            ),
        },
        "quality": {
            "usable_fraction": 0.0,
            "per_unit": [
                {
                    "unit_id": str(row["unit_id"]),
                    "usable_fraction": 0.0,
                    "status": (
                        "unknown"
                        if input_units[index]["observation_status"] == "unknown"
                        else "not_observed"
                    ),
                    "evidence_eligible": False,
                    "reason_codes": [
                        "legacy_v1_observation_or_imputation_status_unknown"
                        if input_units[index]["observation_status"] == "unknown"
                        else "legacy_v1_input_unavailable"
                    ],
                }
                for index, row in enumerate(source["quality"]["per_unit"])
            ],
            "artifact_intervals": deepcopy(source["quality"]["artifact_intervals"]),
            "feature_evaluability": feature_evaluability,
        },
        "event_qualification": {
            "status": "not_evaluable",
            "qualification_receipt_id": None,
            "supporting_evidence_ids": [],
            "reason_codes": ["legacy_v1_event_qualification_not_recoverable"],
        },
        "producer_receipts": [],
        "calibration_receipts": [],
        "capability_qualification_receipts": [],
        "sensitivity_receipts": [],
        "term_decision_receipts": [],
        "evaluation_opportunities": opportunities,
        "findings": migrated_findings,
        "scalp_onset_hypothesis": {
            "hypothesis_id": _identifier(f"HYP-MIGRATED-{event_id}"),
            "layer": "research_ai_hypothesis",
            "claim_boundary": "research_scalp_eeg_onset_candidate_not_cortical_soz",
            "event_id": event_id,
            "localization_status": "not_evaluable",
            "selected_resolution": "none",
            "phenotype": "not_evaluable",
            "candidate_scores": [],
            "per_unit_involvement": per_unit_involvement,
            "involvement_order": [],
            "reason_codes": ["legacy_v1_causal_spatial_evidence_not_recoverable"],
            "model_receipt_id": None,
        },
        "hypothesis_evidence_relations": [],
        "waveform_evidence": migrated_waveforms,
        "limitations": limitations,
        "migration": {
            "source_schema_version": "event_eeg_findings_v1",
            "migration_status": "lossy_fail_closed",
            "source_payload_sha256": _sha256(source),
            "migrator_id": EVENT_FINDINGS_V1_TO_V2_MIGRATOR_ID,
            "loss_codes": sorted(loss_codes),
        },
    }
    return validate_event_eeg_findings_v2_payload(result)


__all__ = [
    "EVENT_FINDINGS_V1_TO_V2_MIGRATOR_ID",
    "migrate_event_eeg_findings_v1_to_v2",
]
