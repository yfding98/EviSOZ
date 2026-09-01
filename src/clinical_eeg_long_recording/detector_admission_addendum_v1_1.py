"""Fail-closed validator for the long-EEG detector admission addendum v1.1.

The addendum corrects one narrow defect in the older BA-IEG core freeze: the
formal false-alarm hard gate is the rate of *all unmatched alarms*, while the
background-only rate remains a required secondary decomposition.  It also
freezes the complete official-development identity denominator and requires
three distinct scorers.  Loading this method contract does not qualify a
detector operating point or authorize clinical/production use.

``receipt_sha256`` is the SHA-256 of canonical JSON after deleting only that
field (UTF-8, sorted keys, compact separators, finite JSON numbers).  The
previously circulated ``21cb5477...`` value belongs to an earlier content
snapshot and is deliberately not accepted for the current contract.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final, Mapping, Sequence

from .tusz_complete_detector_roster_v2 import (
    validate_tusz_analysis_identity_projection_v2,
)


DETECTOR_ADMISSION_ADDENDUM_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_detector_admission_addendum_v1_1"
)
DETECTOR_ADMISSION_ADDENDUM_ID: Final[str] = (
    "CLINICAL-EEG-DETECTOR-ADMISSION-ADDENDUM-V1.1-20260824"
)
TRUSTED_DETECTOR_ADMISSION_ADDENDUM_RECEIPT_SHA256: Final[str] = (
    "f24bdd1c23d900d184af09292cff488fc8d9527b4ad3b3286e9f3646c77ca17a"
)
STALE_DRAFT_DETECTOR_ADMISSION_RECEIPT_SHA256: Final[str] = (
    "21cb5477e8950b80f3379e4721028e119428c5c9c1c60e8fb15a75a8ac9b69a1"
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DETECTOR_ADMISSION_ADDENDUM_PATH: Final[Path] = (
    _ROOT / "configs" / "clinical_eeg_detector_admission_addendum_v1_1.json"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "addendum_id",
    "status",
    "superseded_contract",
    "official_dev_denominator",
    "provider_roles",
    "hard_gates",
    "required_scorers",
    "duration_strata",
    "selection_status",
    "scientific_permissions",
    "source_firewall",
    "receipt_sha256",
}
_SUPERSEDED_KEYS = {
    "path",
    "contract_sha256",
    "superseded_hard_gate_field",
    "replacement_hard_gate_field",
    "background_only_metric_disposition",
    "reason",
}
_DENOMINATOR_KEYS = {
    "identity_projection_path",
    "identity_projection_receipt_sha256",
    "patient_count",
    "recording_count",
    "duration_hours",
    "seizure_event_count",
    "recordings_with_seizure",
    "recordings_without_seizure",
    "reference_count_binding_status",
}
_PROVIDER_ROLE_KEYS = {
    "current_engineering_provider_id",
    "accuracy_primary_provider_id",
    "first_accuracy_challenger_id",
    "first_efficiency_control_id",
    "target_linked_secondary_provider_id",
    "serial_cascade_authorized_in_primary_benchmark",
    "final_provider_selection",
}
_FORMAL_HARD_GATE_KEYS = {
    "pooled_event_sensitivity_minimum",
    "patient_macro_event_sensitivity_minimum",
    "all_unmatched_alarms_per_24h_maximum",
    "warm_end_to_end_rtf_maximum",
}
_REQUIRED_WORKLOAD_METRIC_KEYS = {
    "background_only_false_alarms_per_24h_required_as_secondary_metric",
    "time_in_warning_required",
    "candidate_count_required",
    "queried_eeg_seconds_required",
}
_HARD_GATE_KEYS = _FORMAL_HARD_GATE_KEYS | _REQUIRED_WORKLOAD_METRIC_KEYS
_EXPECTED_SCORERS = (
    "ONSET-NAVIGATION",
    "STRICT-OVERLAP",
    "SZCORE-COMPAT",
)
_EXPECTED_DURATION_STRATA = (
    "complete_official_dev",
    "native_recording_at_least_30_minutes",
    "native_recording_at_least_60_minutes",
)
_EXPECTED_SCIENTIFIC_PERMISSIONS: Mapping[str, bool] = {
    "prediction_first_before_reference_join": True,
    "full_denominator_includes_zero_alarm_partial_and_technical_failure": True,
    "cross_edf_concatenation_to_manufacture_long_recordings": False,
    "generic_inventory_closure_is_detection_performance": False,
    "upstream_reported_metrics_are_local_reproduction": False,
    "sota_claim_authorized": False,
    "clinical_or_production_use_authorized": False,
}
_EXPECTED_SOURCE_FIREWALL: Mapping[str, bool] = {
    "eeg_samples_used": True,
    "edf_signal_header_used": True,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "video_or_behavior_used": False,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def detector_admission_addendum_self_sha256(value: Mapping[str, object]) -> str:
    """Hash canonical contract content after deleting ``receipt_sha256``."""

    if not isinstance(value, Mapping):
        raise TypeError("detector admission addendum must be an object")
    body = deepcopy(dict(value))
    body.pop("receipt_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def _no_duplicate_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _strict_object(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    actual = set(value)
    missing = keys - actual
    unknown = actual - keys
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown keys: {sorted(unknown)}")
    return deepcopy(value)


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _exact_positive_integer(value: object, expected: int, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{context} must equal {expected}")


def _exact_finite_number(value: object, expected: float, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    observed = float(value)
    if not math.isfinite(observed) or observed != expected:
        raise ValueError(f"{context} must equal {expected}")


def _safe_project_file(relative: object, context: str) -> Path:
    if not isinstance(relative, str):
        raise TypeError(f"{context} must be a relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{context} is not a canonical relative path")
    root = _ROOT.resolve(strict=True)
    unresolved = root.joinpath(*pure.parts)
    if unresolved.is_symlink():
        raise ValueError(f"{context} must not be a symlink")
    candidate = unresolved.resolve(strict=True)
    candidate.relative_to(root)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{context} must resolve to a regular non-symlink file")
    return candidate


def _load_strict_json(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{context} contains non-finite JSON token {token}")
            ),
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{context} is not valid UTF-8") from error
    if type(value) is not dict:
        raise TypeError(f"{context} must contain a JSON object")
    return value


def _validate_superseded_contract(value: object) -> dict[str, Any]:
    row = _strict_object(value, _SUPERSEDED_KEYS, "superseded_contract")
    expected = {
        "path": "configs/clinical_eeg_ba_ieg_v1_core_freeze.json",
        "contract_sha256": (
            "d02cb0044555195cb697b4e3e210f0dd6d55378d693c1e35f6a778420a09be91"
        ),
        "superseded_hard_gate_field": (
            "background_only_false_alarms_per_24h_maximum"
        ),
        "replacement_hard_gate_field": "all_unmatched_alarms_per_24h_maximum",
        "background_only_metric_disposition": (
            "required_secondary_report_not_a_standalone_hard_gate"
        ),
        "reason": (
            "duplicate_and_fragment_alarms_overlapping_reference_events_are_real_"
            "navigation_workload"
        ),
    }
    if row != expected:
        raise ValueError("superseded detector-gate correction drifted")
    core_path = _safe_project_file(row["path"], "superseded_contract.path")
    core = _load_strict_json(core_path, "superseded detector contract")
    if core.get("contract_sha256") != row["contract_sha256"]:
        raise ValueError("superseded detector contract hash binding drifted")
    core_body = deepcopy(core)
    core_body.pop("contract_sha256", None)
    if hashlib.sha256(_canonical_json_bytes(core_body)).hexdigest() != row[
        "contract_sha256"
    ]:
        raise ValueError("superseded detector contract self-hash replay failed")
    return row


def _validate_projection_binding(denominator: Mapping[str, Any]) -> None:
    path = _safe_project_file(
        denominator["identity_projection_path"],
        "official_dev_denominator.identity_projection_path",
    )
    projection = validate_tusz_analysis_identity_projection_v2(
        _load_strict_json(path, "official-dev identity projection")
    )
    if projection.get("receipt_sha256") != denominator[
        "identity_projection_receipt_sha256"
    ]:
        raise ValueError("official-dev identity projection receipt binding drifted")
    summary = projection.get("split_summaries", {}).get("source_dev")
    if type(summary) is not dict:
        raise ValueError("official-dev identity projection lacks source_dev summary")
    expected_summary = {
        "analysis_identity_count": 1832,
        "analysis_patient_alias_count": 53,
        "audit_official_path_count": 1832,
        "official_split": "dev",
        "path_count_closure_verified": True,
        "quarantined_path_count": 0,
        "same_patient_alias_excluded_path_count": 0,
    }
    if any(summary.get(key) != expected for key, expected in expected_summary.items()):
        raise ValueError("official-dev identity projection summary drifted")
    records = projection.get("records")
    if not isinstance(records, list):
        raise TypeError("official-dev identity projection records must be a list")
    dev = [row for row in records if row.get("model_split") == "source_dev"]
    if len(dev) != 1832:
        raise ValueError("official-dev identity projection record denominator drifted")
    identities = {row.get("analysis_identity_id") for row in dev}
    patients = {row.get("local_patient_id") for row in dev}
    if len(identities) != 1832 or len(patients) != 53:
        raise ValueError("official-dev identity or patient denominator drifted")
    if any(
        row.get("official_split") != "dev"
        or row.get("analysis_unit_weight") != 1
        for row in dev
    ):
        raise ValueError("official-dev identity projection weight/split drifted")
    reference = projection.get("reference_access_receipt")
    if type(reference) is not dict or any(
        reference.get(key) not in (False, 0)
        for key in (
            "csv_bi_bytes_read",
            "csv_bi_contents_read",
            "csv_bi_files_opened",
            "edf_annotations_read",
            "reference_files_opened",
            "reference_path_argument_accepted",
            "seizure_interval_or_label_values_read",
            "spreadsheet_or_clinical_text_read",
        )
    ):
        raise ValueError("official-dev identity projection opened reference data")


def _validate_official_dev_denominator(
    value: object, *, verify_projection_binding: bool
) -> dict[str, Any]:
    row = _strict_object(value, _DENOMINATOR_KEYS, "official_dev_denominator")
    if row["identity_projection_path"] != (
        "outputs/tusz_complete_detector_roster_v2_20260823/analysis_projection.json"
    ):
        raise ValueError("official-dev identity projection path drifted")
    _sha256(
        row["identity_projection_receipt_sha256"],
        "official_dev_denominator.identity_projection_receipt_sha256",
    )
    if row["identity_projection_receipt_sha256"] != (
        "f987f38d7b550737e60e16601d457297ca91353b762a9e0e3b31a4908439b672"
    ):
        raise ValueError("official-dev identity projection receipt drifted")
    for key, expected in (
        ("patient_count", 53),
        ("recording_count", 1832),
        ("seizure_event_count", 1075),
        ("recordings_with_seizure", 325),
        ("recordings_without_seizure", 1507),
    ):
        _exact_positive_integer(row[key], expected, f"official_dev_denominator.{key}")
    _exact_finite_number(
        row["duration_hours"], 435.548, "official_dev_denominator.duration_hours"
    )
    if row["recordings_with_seizure"] + row["recordings_without_seizure"] != row[
        "recording_count"
    ]:
        raise ValueError("official-dev seizure-bearing/free counts do not close")
    if row["reference_count_binding_status"] != (
        "audited_local_denominator_content_binding_receipt_pending"
    ):
        raise ValueError("official-dev reference-count binding status was promoted")
    if verify_projection_binding:
        _validate_projection_binding(row)
    return row


def _validate_provider_roles(value: object) -> dict[str, Any]:
    row = _strict_object(value, _PROVIDER_ROLE_KEYS, "provider_roles")
    expected = {
        "current_engineering_provider_id": "eventnet_event_boundary_shadow_v1",
        "accuracy_primary_provider_id": None,
        "first_accuracy_challenger_id": "seizuretransformer_timestep_shadow_v1",
        "first_efficiency_control_id": "rest_fft_shadow_v1",
        "target_linked_secondary_provider_id": "deepsoz_temporal_oof_candidate_v1",
        "serial_cascade_authorized_in_primary_benchmark": False,
        "final_provider_selection": "complete_official_dev_same_protocol_pareto_pending",
    }
    if row != expected:
        raise ValueError("detector provider-role freeze drifted")
    return row


def _validate_hard_gates(value: object) -> dict[str, Any]:
    row = _strict_object(value, _HARD_GATE_KEYS, "hard_gates")
    for key, expected in (
        ("pooled_event_sensitivity_minimum", 0.90),
        ("patient_macro_event_sensitivity_minimum", 0.85),
        ("all_unmatched_alarms_per_24h_maximum", 12.0),
        ("warm_end_to_end_rtf_maximum", 0.05),
    ):
        _exact_finite_number(row[key], expected, f"hard_gates.{key}")
    for key in _REQUIRED_WORKLOAD_METRIC_KEYS:
        if row[key] is not True:
            raise ValueError(f"hard_gates.{key} must remain true")
    if "background_only_false_alarms_per_24h_maximum" in row:
        raise ValueError("background-only false alarms cannot replace all-unmatched gate")
    return row


def validate_clinical_eeg_detector_admission_addendum_v1_1(
    value: object,
    *,
    verify_projection_binding: bool = True,
) -> dict[str, Any]:
    """Validate and return a defensive copy of the frozen addendum."""

    candidate = _strict_object(value, _TOP_LEVEL_KEYS, "detector admission addendum")
    receipt = _sha256(candidate["receipt_sha256"], "receipt_sha256")
    computed = detector_admission_addendum_self_sha256(candidate)
    if receipt != computed:
        raise ValueError("detector admission addendum canonical self-hash mismatch")
    if receipt == STALE_DRAFT_DETECTOR_ADMISSION_RECEIPT_SHA256:
        raise ValueError("stale detector admission addendum draft hash is forbidden")
    if candidate["schema_version"] != DETECTOR_ADMISSION_ADDENDUM_SCHEMA_VERSION:
        raise ValueError("detector admission addendum schema version drifted")
    if candidate["addendum_id"] != DETECTOR_ADMISSION_ADDENDUM_ID:
        raise ValueError("detector admission addendum ID drifted")
    if candidate["status"] != "method_selection_frozen_performance_not_established":
        raise ValueError("detector admission performance status was promoted")

    candidate["superseded_contract"] = _validate_superseded_contract(
        candidate["superseded_contract"]
    )
    candidate["official_dev_denominator"] = _validate_official_dev_denominator(
        candidate["official_dev_denominator"],
        verify_projection_binding=verify_projection_binding,
    )
    candidate["provider_roles"] = _validate_provider_roles(candidate["provider_roles"])
    candidate["hard_gates"] = _validate_hard_gates(candidate["hard_gates"])
    if tuple(candidate["required_scorers"]) != _EXPECTED_SCORERS:
        raise ValueError("all three frozen detector scorers are required exactly once")
    if len(set(candidate["required_scorers"])) != len(_EXPECTED_SCORERS):
        raise ValueError("detector scorer roster contains duplicates")
    if tuple(candidate["duration_strata"]) != _EXPECTED_DURATION_STRATA:
        raise ValueError("detector native-duration strata drifted")
    if candidate["selection_status"] != "no_qualified_operating_point":
        raise ValueError("detector selection status was promoted without evidence")
    if candidate["scientific_permissions"] != _EXPECTED_SCIENTIFIC_PERMISSIONS:
        raise ValueError("detector scientific permissions drifted")
    if candidate["source_firewall"] != _EXPECTED_SOURCE_FIREWALL:
        raise ValueError("detector source firewall drifted")
    return deepcopy(candidate)


def load_clinical_eeg_detector_admission_addendum_v1_1(
    path: str | Path = DEFAULT_DETECTOR_ADMISSION_ADDENDUM_PATH,
    *,
    verify_projection_binding: bool = True,
) -> dict[str, Any]:
    """Load strict UTF-8 JSON and validate the detector addendum."""

    candidate = _load_strict_json(Path(path), "detector admission addendum file")
    return validate_clinical_eeg_detector_admission_addendum_v1_1(
        candidate,
        verify_projection_binding=verify_projection_binding,
    )


__all__ = [
    "DEFAULT_DETECTOR_ADMISSION_ADDENDUM_PATH",
    "DETECTOR_ADMISSION_ADDENDUM_ID",
    "DETECTOR_ADMISSION_ADDENDUM_SCHEMA_VERSION",
    "STALE_DRAFT_DETECTOR_ADMISSION_RECEIPT_SHA256",
    "TRUSTED_DETECTOR_ADMISSION_ADDENDUM_RECEIPT_SHA256",
    "detector_admission_addendum_self_sha256",
    "load_clinical_eeg_detector_admission_addendum_v1_1",
    "validate_clinical_eeg_detector_admission_addendum_v1_1",
]
