#!/usr/bin/env python3
"""Validate EEG-SOZ report/gold JSON beyond what JSON Schema can express."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMAS = {
    "report": ROOT / "reports/schemas/eeg_soz_evidence_report.schema.json",
    "gold": ROOT / "reports/schemas/eeg_soz_gold_annotation.schema.json",
}

DIRECT_IDENTIFIER_PATTERNS = (
    re.compile(
        r"(?:姓名|患者姓名|MRN|病案号|住院号|身份证号|"
        r"patient\s*name|patient|name|medical\s*record\s*number)"
        r"\s*[:：#=]\s*(?!not_recorded\b|unknown\b|未提供\b)\S+",
        re.I,
    ),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
               r"(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?!\d)"),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?:file://|/mnt/|/home/|[A-Za-z]:\\)", re.I),
)
PHI_SCAN_EXEMPT_FIELD_SUFFIXES = (
    "_sha256",
    "_hash",
)
INVASIVE_REVIEWER_ROLES = {
    "clinical_neurophysiologist",
    "epileptologist",
}
OUTCOME_REVIEWER_ROLES = {
    "epileptologist",
    "neurosurgeon",
    "other_physician",
}
TRUSTED_ATTESTATIONS = {
    "verified",
    "signed",
}


def _walk(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            yield from _walk(child, f"{path}[{idx}]")


def _interval_errors(
    document: Any,
    *,
    duration_s: float | None,
    sfreq: float | None,
) -> list[str]:
    errors: list[str] = []
    for path, value in _walk(document):
        if not isinstance(value, dict):
            continue
        if {"lower_s", "upper_s", "reference"}.issubset(value):
            lower = float(value["lower_s"])
            upper = float(value["upper_s"])
            if upper < lower:
                errors.append(f"{path}: upper_s ({upper}) is below lower_s ({lower})")
            if (
                duration_s is not None
                and value.get("reference") == "record_start"
                and upper > duration_s
            ):
                errors.append(
                    f"{path}: record-relative upper_s ({upper}) exceeds "
                    f"record duration ({duration_s})"
                )
            sample_lower = value.get("sample_lower")
            sample_upper = value.get("sample_upper")
            if sample_lower is not None and sample_upper is not None:
                if int(sample_upper) < int(sample_lower):
                    errors.append(
                        f"{path}: sample_upper ({sample_upper}) is below "
                        f"sample_lower ({sample_lower})"
                    )
                if sfreq and value.get("reference") == "record_start":
                    tolerance_s = max(2.0 / sfreq, 1e-6)
                    if abs(int(sample_lower) / sfreq - lower) > tolerance_s:
                        errors.append(
                            f"{path}: sample_lower is inconsistent with lower_s "
                            f"at sampling_rate_hz={sfreq}"
                        )
                    if abs(int(sample_upper) / sfreq - upper) > tolerance_s:
                        errors.append(
                            f"{path}: sample_upper is inconsistent with upper_s "
                            f"at sampling_rate_hz={sfreq}"
                        )
    return errors


def _phi_errors(document: Any) -> list[str]:
    errors: list[str] = []
    for path, value in _walk(document):
        if not isinstance(value, str):
            continue
        field_name = path.rsplit(".", maxsplit=1)[-1]
        if field_name.endswith(PHI_SCAN_EXEMPT_FIELD_SUFFIXES):
            continue
        for pattern in DIRECT_IDENTIFIER_PATTERNS:
            if pattern.search(value):
                errors.append(
                    f"{path}: potential direct identifier or filesystem path detected"
                )
                break
    return errors


def _unique_id_map(
    items: Any, id_key: str, path: str, errors: list[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(items, list):
        return result
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        item_id = item.get(id_key)
        if not isinstance(item_id, str):
            continue
        if item_id in result:
            errors.append(f"{path}[{idx}].{id_key}: duplicate ID {item_id!r}")
        result[item_id] = item
    return result


def _supported_by_errors(document: Any, evidence_ids: set[str]) -> list[str]:
    errors: list[str] = []
    for path, value in _walk(document):
        if not path.endswith(".supported_by") or not isinstance(value, list):
            continue
        missing = sorted(
            ref for ref in value if isinstance(ref, str) and ref not in evidence_ids
        )
        if missing:
            errors.append(f"{path}: unknown Evidence IDs {missing}")
    return errors


def _subset_errors(
    values: Any,
    allowed: set[str],
    path: str,
) -> list[str]:
    if not isinstance(values, list):
        return []
    unknown = sorted(
        value for value in values if isinstance(value, str) and value not in allowed
    )
    return [f"{path}: electrodes absent from recording/source {unknown}"] if unknown else []


def _reviewer_ids(
    attestations: Any,
    path: str,
    errors: list[str],
) -> set[str]:
    reviewer_ids: set[str] = set()
    if not isinstance(attestations, list):
        return reviewer_ids
    for idx, item in enumerate(attestations):
        if not isinstance(item, dict):
            continue
        reviewer_id = item.get("reviewer_id")
        if not isinstance(reviewer_id, str):
            continue
        if reviewer_id in reviewer_ids:
            errors.append(
                f"{path}[{idx}].reviewer_id: duplicate reviewer {reviewer_id!r}"
            )
        reviewer_ids.add(reviewer_id)
    return reviewer_ids


def _qualified_reviewer_ids(
    attestations: Any,
    *,
    roles: set[str],
) -> set[str]:
    if not isinstance(attestations, list):
        return set()
    return {
        item["reviewer_id"]
        for item in attestations
        if isinstance(item, dict)
        and isinstance(item.get("reviewer_id"), str)
        and item.get("role") in roles
        and item.get("attestation") in TRUSTED_ATTESTATIONS
    }


def _structured_support_errors(
    item: Any,
    path: str,
    evidence: dict[str, dict[str, Any]],
    *,
    interval_key: str | None = None,
    physical_key: str | None = None,
    required_representations: set[str] | None = None,
) -> list[str]:
    """Check that structured findings are actually covered by cited evidence."""
    if not isinstance(item, dict):
        return []
    errors: list[str] = []
    refs = [
        evidence[ref]
        for ref in item.get("supported_by") or []
        if isinstance(ref, str) and ref in evidence
    ]
    if not refs:
        return errors

    if interval_key and isinstance(item.get(interval_key), dict):
        interval = item[interval_key]
        lower = float(interval["lower_s"])
        upper = float(interval["upper_s"])
        reference = interval.get("reference")
        containing = []
        for ref in refs:
            ref_interval = ref.get("interval")
            if not isinstance(ref_interval, dict):
                continue
            if ref_interval.get("reference") != reference:
                continue
            if (
                float(ref_interval["lower_s"]) <= lower
                and float(ref_interval["upper_s"]) >= upper
            ):
                containing.append(ref)
        if not containing:
            errors.append(
                f"{path}.{interval_key}: no cited Evidence interval with the same "
                "time reference contains this finding interval"
            )

    if physical_key:
        claimed = set(item.get(physical_key) or [])
        available = {
            electrode
            for ref in refs
            for electrode in ref.get("physical_electrodes") or []
            if isinstance(electrode, str)
        }
        missing = sorted(claimed - available)
        if missing:
            errors.append(
                f"{path}.{physical_key}: cited Evidence does not contain electrodes "
                f"{missing}"
            )

    if required_representations:
        usable_representations = {
            ref.get("representation")
            for ref in refs
            if ref.get("quality_state") != "unusable"
        }
        if not usable_representations.intersection(required_representations):
            errors.append(
                f"{path}.supported_by: no usable cited Evidence has one of the "
                f"required representations {sorted(required_representations)}"
            )
    return errors


def _report_semantics(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    recording = document.get("recording") or {}
    duration_s = float(recording.get("duration_s", 0)) or None
    sfreq = float(recording.get("sampling_rate_hz", 0)) or None
    record_id = recording.get("record_id")
    electrodes = set(recording.get("physical_electrodes") or [])

    errors.extend(_interval_errors(document, duration_s=duration_s, sfreq=sfreq))
    errors.extend(_phi_errors(document))

    evidence = _unique_id_map(
        document.get("evidence"), "evidence_id", "$.evidence", errors
    )
    _unique_id_map(document.get("events"), "event_id", "$.events", errors)
    _unique_id_map(document.get("claims"), "claim_id", "$.claims", errors)
    errors.extend(_supported_by_errors(document, set(evidence)))

    for idx, item in enumerate(document.get("evidence") or []):
        if not isinstance(item, dict):
            continue
        if item.get("record_id") != record_id:
            errors.append(
                f"$.evidence[{idx}].record_id: does not match $.recording.record_id"
            )
        errors.extend(
            _subset_errors(
                item.get("physical_electrodes"),
                electrodes,
                f"$.evidence[{idx}].physical_electrodes",
            )
        )
        representation = item.get("representation")
        attestations = item.get("reviewer_attestations") or []
        _reviewer_ids(
            attestations,
            f"$.evidence[{idx}].reviewer_attestations",
            errors,
        )
        qualified_invasive_reviewers = _qualified_reviewer_ids(
            attestations,
            roles=INVASIVE_REVIEWER_ROLES,
        )
        qualified_outcome_reviewers = _qualified_reviewer_ids(
            attestations,
            roles=OUTCOME_REVIEWER_ROLES,
        )
        if representation in {"invasive_eeg", "surgical_outcome"}:
            if item.get("source") == "model":
                errors.append(
                    f"$.evidence[{idx}].source: model-generated evidence cannot "
                    f"establish {representation}"
                )
            if not item.get("source_artifact_id") or not item.get("source_sha256"):
                errors.append(
                    f"$.evidence[{idx}]: strong clinical evidence requires a "
                    "pseudonymous source artifact ID and SHA-256"
                )
            if item.get("quality_state") not in {"usable", "partially_usable"}:
                errors.append(
                    f"$.evidence[{idx}].quality_state: strong evidence cannot "
                    "be unusable or uncertain"
                )
        if (
            representation == "invasive_eeg"
            and len(qualified_invasive_reviewers) < 2
        ):
            errors.append(
                f"$.evidence[{idx}].reviewer_attestations: invasive evidence "
                "requires at least two distinct verified/signed reviewers with "
                "clinical neurophysiology or epileptology roles"
            )
        if (
            representation == "surgical_outcome"
            and len(qualified_outcome_reviewers) < 1
        ):
            errors.append(
                f"$.evidence[{idx}].reviewer_attestations: surgical outcome "
                "requires at least one verified/signed outcome physician"
            )

    for event_idx, event in enumerate(document.get("events") or []):
        if not isinstance(event, dict):
            continue
        _unique_id_map(
            event.get("differential_explanations"),
            "explanation_id",
            f"$.events[{event_idx}].differential_explanations",
            errors,
        )
        onset = event.get("scalp_ictal_onset") or {}
        errors.extend(
            _subset_errors(
                onset.get("earliest_physical_electrodes"),
                electrodes,
                f"$.events[{event_idx}].scalp_ictal_onset."
                "earliest_physical_electrodes",
            )
        )
        if onset.get("state") == "observed":
            errors.extend(
                _structured_support_errors(
                    onset,
                    f"$.events[{event_idx}].scalp_ictal_onset",
                    evidence,
                    interval_key="onset_interval",
                    physical_key="earliest_physical_electrodes",
                    required_representations={
                        "raw_waveform",
                        "waveform_image",
                        "time_frequency",
                        "topography",
                    },
                )
            )
        clinical_onset = event.get("clinical_onset") or {}
        if clinical_onset.get("state") in {"present", "uncertain"}:
            errors.extend(
                _structured_support_errors(
                    clinical_onset,
                    f"$.events[{event_idx}].clinical_onset",
                    evidence,
                    interval_key="interval",
                    required_representations={"video", "clinical_document"},
                )
            )
        for step_idx, step in enumerate(event.get("propagation") or []):
            if isinstance(step, dict):
                errors.extend(
                    _subset_errors(
                        step.get("physical_electrodes"),
                        electrodes,
                        f"$.events[{event_idx}].propagation[{step_idx}]."
                        "physical_electrodes",
                    )
                )
                errors.extend(
                    _structured_support_errors(
                        step,
                        f"$.events[{event_idx}].propagation[{step_idx}]",
                        evidence,
                        interval_key="interval",
                        physical_key="physical_electrodes",
                        required_representations={
                            "raw_waveform",
                            "waveform_image",
                            "time_frequency",
                            "topography",
                        },
                    )
                )

        hypothesis = event.get("cortical_soz_localization_hypothesis") or {}
        tier = hypothesis.get("evidence_tier")
        referenced = [
            evidence[ref]
            for ref in hypothesis.get("supported_by") or []
            if ref in evidence
        ]
        trusted_invasive = [
            item
            for item in referenced
            if item.get("representation") == "invasive_eeg"
            and item.get("source") != "model"
            and item.get("quality_state") in {"usable", "partially_usable"}
            and len(
                _qualified_reviewer_ids(
                    item.get("reviewer_attestations"),
                    roles=INVASIVE_REVIEWER_ROLES,
                )
            )
            >= 2
        ]
        trusted_outcome = [
            item
            for item in referenced
            if item.get("representation") == "surgical_outcome"
            and item.get("source") != "model"
            and item.get("quality_state") in {"usable", "partially_usable"}
            and bool(
                _qualified_reviewer_ids(
                    item.get("reviewer_attestations"),
                    roles=OUTCOME_REVIEWER_ROLES,
                )
            )
        ]
        if tier == "G3" and not trusted_invasive:
            errors.append(
                f"$.events[{event_idx}].cortical_soz_localization_hypothesis: "
                "G3 requires referenced, externally sourced invasive_eeg evidence "
                "attested by two distinct reviewers"
            )
        if tier == "G4":
            if not trusted_invasive or not trusted_outcome:
                errors.append(
                    f"$.events[{event_idx}].cortical_soz_localization_hypothesis: "
                    "G4 requires trusted invasive_eeg and surgical_outcome evidence"
                )

    status = document.get("report_status")
    provenance = document.get("provenance") or {}
    attestations = provenance.get("reviewer_attestations") or []
    attestation_ids = _reviewer_ids(
        attestations,
        "$.provenance.reviewer_attestations",
        errors,
    )
    audit = provenance.get("audit_log") or []
    for idx, entry in enumerate(audit):
        if not isinstance(entry, dict):
            continue
        if entry.get("action") in {"reviewed", "signed"}:
            if entry.get("actor_type") != "human":
                errors.append(
                    f"$.provenance.audit_log[{idx}].actor_type: review/sign actions "
                    "must be performed by a human"
                )
            if entry.get("actor_id") not in attestation_ids:
                errors.append(
                    f"$.provenance.audit_log[{idx}].actor_id: review/sign actor "
                    "must match a reviewer attestation"
                )
    if status in {"physician_reviewed", "physician_signed"}:
        expected_attestation = (
            {"signed"}
            if status == "physician_signed"
            else {"reviewed", "signed"}
        )
        qualified = {
            item.get("reviewer_id")
            for item in attestations
            if isinstance(item, dict)
            and item.get("role")
            in {
                "clinical_neurophysiologist",
                "epileptologist",
                "other_physician",
            }
            and item.get("attestation") in expected_attestation
        }
        if not qualified:
            errors.append(
                "$.provenance.reviewer_attestations: physician-reviewed/signed "
                "reports require a qualified physician attestation"
            )
        expected_action = (
            "signed" if status == "physician_signed" else "reviewed"
        )
        matching_audit = any(
            isinstance(item, dict)
            and item.get("actor_type") == "human"
            and item.get("actor_id") in qualified
            and item.get("action") == expected_action
            for item in audit
        )
        if not matching_audit:
            errors.append(
                "$.provenance.audit_log: report status lacks a matching physician "
                "review/sign action"
            )

    return errors


def _gold_semantics(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    source = document.get("source") or {}
    duration_s = float(source.get("duration_s", 0)) or None
    sfreq = float(source.get("sampling_rate_hz", 0)) or None
    electrodes = set(source.get("physical_electrodes") or [])

    errors.extend(_interval_errors(document, duration_s=duration_s, sfreq=sfreq))
    errors.extend(_phi_errors(document))

    scalp = document.get("scalp_event") or {}
    errors.extend(
        _subset_errors(
            scalp.get("earliest_physical_electrodes"),
            electrodes,
            "$.scalp_event.earliest_physical_electrodes",
        )
    )
    for idx, step in enumerate(scalp.get("early_spread") or []):
        if isinstance(step, dict):
            errors.extend(
                _subset_errors(
                    step.get("physical_electrodes"),
                    electrodes,
                    f"$.scalp_event.early_spread[{idx}].physical_electrodes",
                )
            )

    tiers = set(document.get("evidence_tiers") or [])
    if "G3" in tiers:
        invasive = document.get("invasive_event") or {}
        if invasive.get("state") != "present":
            errors.append("$.invasive_event.state: G3 requires present invasive evidence")
        if int(invasive.get("typical_seizures_reviewed", 0)) < 2:
            errors.append(
                "$.invasive_event.typical_seizures_reviewed: G3 requires at least 2"
            )
        if int(invasive.get("independent_reviewer_count", 0)) < 2:
            errors.append(
                "$.invasive_event.independent_reviewer_count: G3 requires at least 2"
            )
        independent_reviews = invasive.get("independent_reviews") or []
        reviewer_ids = _reviewer_ids(
            independent_reviews,
            "$.invasive_event.independent_reviews",
            errors,
        )
        if len(reviewer_ids) < 2:
            errors.append(
                "$.invasive_event.independent_reviews: G3 requires at least two "
                "distinct invasive reviewers"
            )
        declared_count = int(invasive.get("independent_reviewer_count", 0))
        if declared_count != len(reviewer_ids):
            errors.append(
                "$.invasive_event.independent_reviewer_count: must equal the "
                "number of distinct independent review records"
            )
    if "G4" in tiers:
        outcome = document.get("surgery_and_outcome") or {}
        if outcome.get("state") != "present":
            errors.append("$.surgery_and_outcome.state: G4 requires present outcome")
        if float(outcome.get("follow_up_months", 0)) < 12:
            errors.append(
                "$.surgery_and_outcome.follow_up_months: G4 requires at least 12 months"
            )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate EEG-SOZ JSON Schema and cross-field clinical semantics"
    )
    parser.add_argument("document", type=Path)
    parser.add_argument("--kind", choices=("report", "gold"), required=True)
    parser.add_argument("--schema", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema_path = args.schema or DEFAULT_SCHEMAS[args.kind]
    document = json.loads(args.document.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = [
        f"{'.'.join(str(part) for part in error.absolute_path) or '$'}: "
        f"{error.message}"
        for error in sorted(validator.iter_errors(document), key=lambda e: list(e.path))
    ]
    if isinstance(document, dict):
        errors.extend(
            _report_semantics(document)
            if args.kind == "report"
            else _gold_semantics(document)
        )
    else:
        errors.append("$: document must be a JSON object")

    if errors:
        print(f"FAIL: {len(errors)} validation error(s)", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"PASS: {args.kind} schema and semantic validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
