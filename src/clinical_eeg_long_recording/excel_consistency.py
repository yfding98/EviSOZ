"""Post-freeze comparison of EEG onset facts with a PHI-free Excel review.

This module is intentionally outside the report materialization path.  It
first verifies that the authoritative EEG bundle and both rendered report
bodies are already frozen by the report manifest.  Only then may it load an
optional, closed-vocabulary Excel review JSON and publish a separate
consistency artifact.  The review can never add facts to the EEG bundle,
change the report body, select an event, rank an electrode, or reach an LLM.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from src.clinical_eeg_report.schema import (
    canonicalize_derivation,
    canonicalize_electrode,
    validate_report_payload,
)

from .aggregation import validate_trustworthy_long_term_clinical_eeg_bundle


EXCEL_REVIEW_SCHEMA_VERSION = "clinical_eeg_excel_onset_review_v1"
CONSISTENCY_ARTIFACT_SCHEMA_VERSION = (
    "postfreeze_clinical_eeg_excel_consistency_v1"
)
REPORT_MATERIALIZATION_SCHEMA_VERSION = (
    "trustworthy_long_term_clinical_eeg_materialization_v1"
)

COMPARISON_FIELDS = (
    "laterality",
    "regions",
    "patterns",
    "electrodes",
    "derivations",
)
COMPARISON_STATUSES = frozenset(
    {"match", "partial_match", "mismatch", "not_available"}
)

_LATERALITIES = frozenset(
    {"left", "right", "bilateral", "midline", "none", "indeterminate"}
)
_REGIONS = frozenset(
    {
        "frontal",
        "temporal",
        "central",
        "parietal",
        "occipital",
        "frontotemporal",
        "centrotemporal",
        "temporoparietal",
        "posterior",
        "diffuse",
        "midline",
        "unknown",
    }
)
_PATTERNS = frozenset(
    {
        "low_voltage_fast_activity",
        "rhythmic_activity",
        "repetitive_spikes",
        "electrodecrement",
        "attenuation",
        "irregular_activity",
        "spike",
        "sharp_wave",
        "spike_and_slow_wave",
        "sharp_and_slow_wave",
        "polyspike",
        "fast_activity",
        "theta_activity",
        "delta_activity",
        "mixed",
        "indeterminate",
        "other",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REVIEW_ID_RE = re.compile(r"^XLSREVIEW-[0-9a-f]{24}$")
_OBSERVATION_ID_RE = re.compile(r"^XLSOBS-[0-9a-f]{24}$")

_REVIEW_KEYS = {
    "schema_version",
    "review_id",
    "recording_id",
    "bundle_id",
    "observations",
    "claim_boundary",
}
_OBSERVATION_KEYS = {
    "observation_id",
    "eeg_event_id",
    "binding_verified",
    "typed_eeg_fields",
}
_CLAIM_BOUNDARY = {
    "raw_excel_text_included": False,
    "direct_identity_included": False,
    "source_path_included": False,
    "edf_annotation_included": False,
    "used_for_report_generation": False,
    "used_for_renderer": False,
    "used_for_llm": False,
}


def _strict_object(
    value: object,
    *,
    keys: set[str],
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    actual = set(value)
    missing = keys - actual
    extra = actual - keys
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{context} has unknown keys: {sorted(extra)}")
    return {str(key): deepcopy(item) for key, item in value.items()}


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a closed de-identified identifier")
    return value


def _opaque_identifier(
    value: object,
    pattern: re.Pattern[str],
    context: str,
) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{context} must be an opaque hash-derived identifier")
    return value


def _sha256_value(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON input must be a regular non-symlink file: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError(f"JSON input must be an object: {path.name}")
    return value


def _optional_closed_list(
    value: object,
    *,
    allowed: frozenset[str],
    context: str,
) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise TypeError(f"{context} must be null or a non-empty list")
    result: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or raw not in allowed:
            raise ValueError(f"{context} contains an unsupported controlled code")
        if raw in result:
            raise ValueError(f"{context} contains duplicate values")
        result.append(raw)
    return result


def _optional_electrodes(value: object, context: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise TypeError(f"{context} must be null or a non-empty list")
    result: list[str] = []
    for raw in value:
        canonical = canonicalize_electrode(raw)
        if canonical in result:
            raise ValueError(f"{context} contains duplicate canonical electrodes")
        result.append(canonical)
    return result


def _optional_derivations(value: object, context: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise TypeError(f"{context} must be null or a non-empty list")
    result: list[str] = []
    for raw in value:
        canonical = canonicalize_derivation(raw)
        if canonical in result:
            raise ValueError(f"{context} contains duplicate canonical derivations")
        result.append(canonical)
    return result


def validate_excel_onset_review(
    value: object,
    *,
    expected_recording_id: str,
    expected_bundle_id: str,
    expected_event_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate a narrow PHI-free, free-text-free Excel review payload."""

    data = _strict_object(value, keys=_REVIEW_KEYS, context="Excel onset review")
    if data["schema_version"] != EXCEL_REVIEW_SCHEMA_VERSION:
        raise ValueError(
            f"Excel onset review must use {EXCEL_REVIEW_SCHEMA_VERSION}"
        )
    data["review_id"] = _opaque_identifier(
        data["review_id"], _REVIEW_ID_RE, "review_id"
    )
    data["recording_id"] = _identifier(data["recording_id"], "recording_id")
    data["bundle_id"] = _identifier(data["bundle_id"], "bundle_id")
    if data["recording_id"] != expected_recording_id:
        raise ValueError("Excel review recording_id does not match the frozen bundle")
    if data["bundle_id"] != expected_bundle_id:
        raise ValueError("Excel review bundle_id does not match the frozen bundle")

    boundary = _strict_object(
        data["claim_boundary"],
        keys=set(_CLAIM_BOUNDARY),
        context="Excel review claim_boundary",
    )
    for key, expected in _CLAIM_BOUNDARY.items():
        if boundary[key] is not expected:
            raise ValueError(f"Excel review claim_boundary.{key} must be {expected}")
    data["claim_boundary"] = dict(_CLAIM_BOUNDARY)

    raw_observations = data["observations"]
    if not isinstance(raw_observations, list):
        raise TypeError("Excel onset review observations must be a list")
    event_ids = set(expected_event_ids)
    seen_observations: set[str] = set()
    seen_events: set[str] = set()
    observations: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_observations):
        item = _strict_object(
            raw,
            keys=_OBSERVATION_KEYS,
            context=f"Excel onset review observations[{index}]",
        )
        observation_id = _opaque_identifier(
            item["observation_id"],
            _OBSERVATION_ID_RE,
            f"observations[{index}].observation_id",
        )
        event_id = _identifier(
            item["eeg_event_id"], f"observations[{index}].eeg_event_id"
        )
        if observation_id in seen_observations:
            raise ValueError("Excel onset review contains duplicate observation IDs")
        if event_id in seen_events:
            raise ValueError("Excel onset review contains repeated event bindings")
        if event_id not in event_ids:
            raise ValueError("Excel onset review references an unknown EEG event")
        if item["binding_verified"] is not True:
            raise ValueError("Excel onset review event binding must be verified")
        seen_observations.add(observation_id)
        seen_events.add(event_id)

        typed = _strict_object(
            item["typed_eeg_fields"],
            keys=set(COMPARISON_FIELDS),
            context=f"observations[{index}].typed_eeg_fields",
        )
        laterality = typed["laterality"]
        if laterality is not None and laterality not in _LATERALITIES:
            raise ValueError("typed laterality is not a supported controlled code")
        observations.append(
            {
                "observation_id": observation_id,
                "eeg_event_id": event_id,
                "binding_verified": True,
                "typed_eeg_fields": {
                    "laterality": laterality,
                    "regions": _optional_closed_list(
                        typed["regions"],
                        allowed=_REGIONS,
                        context="typed regions",
                    ),
                    "patterns": _optional_closed_list(
                        typed["patterns"],
                        allowed=_PATTERNS,
                        context="typed patterns",
                    ),
                    "electrodes": _optional_electrodes(
                        typed["electrodes"], "typed electrodes"
                    ),
                    "derivations": _optional_derivations(
                        typed["derivations"], "typed derivations"
                    ),
                },
            }
        )
    observations.sort(key=lambda item: expected_event_ids.index(item["eeg_event_id"]))
    data["observations"] = observations
    return data


def _onset_eeg_values(event: Mapping[str, Any]) -> dict[str, list[str] | None]:
    report = validate_report_payload(event["event_report_payload"]).to_dict()
    event_id = str(event["eeg_event_id"])
    onset_facts = [
        fact
        for fact in report["facts"]
        if fact["fact_type"] == "ictal_onset_pattern"
        and fact.get("eeg_event_id") == event_id
    ]
    if len(onset_facts) != 1:
        return {field: None for field in COMPARISON_FIELDS}
    value = onset_facts[0]["value"]

    patterns: list[str] = []
    for key in ("onset_type", "morphology"):
        raw = value.get(key)
        if isinstance(raw, str) and raw not in patterns:
            patterns.append(raw)
    electrodes: list[str] = []
    for key in ("electrodes", "maximal_electrodes"):
        for raw in value.get(key, []):
            if raw not in electrodes:
                electrodes.append(raw)
    return {
        "laterality": [value["laterality"]],
        "regions": list(value["regions"]) or None,
        "patterns": patterns or None,
        "electrodes": electrodes or None,
        "derivations": list(value.get("derivations", [])) or None,
    }


def _review_values(
    observation: Mapping[str, Any] | None,
) -> dict[str, list[str] | None]:
    if observation is None:
        return {field: None for field in COMPARISON_FIELDS}
    typed = observation["typed_eeg_fields"]
    laterality = typed["laterality"]
    return {
        "laterality": [laterality] if laterality is not None else None,
        "regions": deepcopy(typed["regions"]),
        "patterns": deepcopy(typed["patterns"]),
        "electrodes": deepcopy(typed["electrodes"]),
        "derivations": deepcopy(typed["derivations"]),
    }


def _field_comparison(
    field: str,
    eeg_values: list[str] | None,
    excel_values: list[str] | None,
) -> dict[str, Any]:
    eeg = sorted(eeg_values or [])
    excel = sorted(excel_values or [])
    if not eeg or not excel:
        status = "not_available"
        overlap: list[str] = []
    else:
        overlap = sorted(set(eeg).intersection(excel))
        if eeg == excel:
            status = "match"
        elif overlap:
            status = "partial_match"
        else:
            status = "mismatch"
    if status not in COMPARISON_STATUSES:
        raise AssertionError("unreachable comparison status")
    return {
        "field": field,
        "status": status,
        "eeg_available": bool(eeg),
        "excel_available": bool(excel),
        "eeg_values": eeg,
        "excel_values": excel,
        "overlap_values": overlap,
    }


def compare_frozen_bundle_with_excel_review(
    bundle: object,
    excel_review: object | None,
) -> dict[str, Any]:
    """Return event-by-event comparisons without mutating either input."""

    frozen = validate_trustworthy_long_term_clinical_eeg_bundle(bundle)
    event_ids = [str(event["eeg_event_id"]) for event in frozen["events"]]
    review = (
        validate_excel_onset_review(
            excel_review,
            expected_recording_id=str(frozen["recording_id"]),
            expected_bundle_id=str(frozen["bundle_id"]),
            expected_event_ids=event_ids,
        )
        if excel_review is not None
        else None
    )
    observation_by_event = {
        str(item["eeg_event_id"]): item for item in review["observations"]
    } if review is not None else {}

    status_counts = {status: 0 for status in sorted(COMPARISON_STATUSES)}
    events: list[dict[str, Any]] = []
    for event in frozen["events"]:
        event_id = str(event["eeg_event_id"])
        observation = observation_by_event.get(event_id)
        eeg = _onset_eeg_values(event)
        excel = _review_values(observation)
        comparisons = [
            _field_comparison(field, eeg[field], excel[field])
            for field in COMPARISON_FIELDS
        ]
        for comparison in comparisons:
            status_counts[comparison["status"]] += 1
        events.append(
            {
                "event_number": int(event["event_number"]),
                "eeg_event_id": event_id,
                "excel_observation_status": (
                    "available" if observation is not None else "not_available"
                ),
                "excel_observation_id": (
                    str(observation["observation_id"])
                    if observation is not None
                    else None
                ),
                "comparisons": comparisons,
            }
        )
    return {
        "recording_id": frozen["recording_id"],
        "bundle_id": frozen["bundle_id"],
        "event_count": len(events),
        "excel_review_status": "available" if review is not None else "not_available",
        "events": events,
        "comparison_summary": {
            "field_comparison_count": len(events) * len(COMPARISON_FIELDS),
            "status_counts": status_counts,
        },
        "validated_excel_review": review,
    }


def verify_frozen_eeg_report_bundle(
    report_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Verify and load an EEG-only report bundle before any sidecar input."""
    if report_dir.is_symlink():
        raise ValueError("report bundle directory must not be a symlink")
    root = report_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("report bundle directory must be a regular directory")
    manifest_path = root / "manifest.json"
    bundle_path = root / "bundle.json"
    manifest = _json_object(manifest_path)
    if manifest.get("schema_version") != REPORT_MATERIALIZATION_SCHEMA_VERSION:
        raise ValueError("report manifest schema is unsupported")
    if manifest.get("status") != "completed_unsigned_ai_draft":
        raise ValueError("report bundle must be completely materialized before comparison")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TypeError("report manifest artifacts must be an object")
    if "manifest.json" in artifacts:
        raise ValueError("report manifest must not claim a self hash")
    required = ("bundle.json", "report.html", "report.docx")
    hashes: dict[str, str] = {}
    for relative in required:
        expected = _sha256_value(artifacts.get(relative), f"artifacts.{relative}")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"frozen report artifact is missing: {relative}")
        actual = _file_sha256(path)
        if actual != expected:
            raise ValueError(f"frozen report artifact hash mismatch: {relative}")
        hashes[relative] = actual
    context_path = root / "context.json"
    if (
        "context.json" in artifacts
        or context_path.exists()
        or context_path.is_symlink()
    ):
        raise ValueError("legacy source context is forbidden in an EEG-only report bundle")
    source_receipts = manifest.get("source_receipts")
    if isinstance(source_receipts, Mapping) and source_receipts.get(
        "context_sha256"
    ) is not None:
        raise ValueError("report manifest must not bind a source context receipt")
    scope = manifest.get("scope_receipt")
    if not isinstance(scope, Mapping):
        raise TypeError("report manifest scope_receipt must be an object")
    if scope.get("eeg_signal_only_generation") is not True:
        raise ValueError(
            "frozen report scope_receipt.eeg_signal_only_generation must be true"
        )
    for key in (
        "external_edf_annotations_loaded",
        "excel_observations_loaded",
        "source_context_joined_post_freeze",
    ):
        if scope.get(key) is not False:
            raise ValueError(f"frozen report scope_receipt.{key} must be false")
    bundle = validate_trustworthy_long_term_clinical_eeg_bundle(
        _json_object(bundle_path)
    )
    for event in bundle["events"]:
        report = validate_report_payload(event["event_report_payload"]).to_dict()
        if any(
            fact["fact_type"] == "source_eeg_annotation_timing"
            for fact in report["facts"]
        ):
            raise ValueError(
                "frozen EEG-only report contains a source annotation timing fact"
            )
    for key in ("recording_id", "bundle_id", "event_count"):
        if manifest.get(key) != bundle.get(key):
            raise ValueError(f"report manifest {key} does not match bundle.json")
    return manifest, bundle, hashes


def materialize_postfreeze_excel_consistency(
    *,
    report_bundle_dir: str | Path,
    excel_review_path: str | Path | None,
    output_path: str | Path,
) -> dict[str, Any]:
    """Publish one private, separate consistency artifact atomically."""

    raw_report_dir = Path(report_bundle_dir)
    if raw_report_dir.is_symlink():
        raise ValueError("report bundle directory must not be a symlink")
    report_dir = raw_report_dir.resolve(strict=True)
    raw_target = Path(output_path)
    if raw_target.is_symlink():
        raise FileExistsError(raw_target)
    target = raw_target.resolve()
    try:
        target.relative_to(report_dir)
    except ValueError:
        pass
    else:
        raise ValueError("consistency artifact must be outside the frozen report bundle")
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)

    manifest, bundle, report_hashes = verify_frozen_eeg_report_bundle(report_dir)
    raw_review_path = Path(excel_review_path) if excel_review_path is not None else None
    if raw_review_path is not None and raw_review_path.is_symlink():
        raise ValueError("Excel review must not be a symlink")
    review_path = (
        raw_review_path.resolve(strict=True) if raw_review_path is not None else None
    )
    review_payload = _json_object(review_path) if review_path is not None else None
    comparison = compare_frozen_bundle_with_excel_review(bundle, review_payload)
    review_hash = _file_sha256(review_path) if review_path is not None else None
    validated_review = comparison.pop("validated_excel_review")

    artifact = {
        "schema_version": CONSISTENCY_ARTIFACT_SCHEMA_VERSION,
        "status": "completed_postfreeze_comparison",
        "recording_id": comparison["recording_id"],
        "bundle_id": comparison["bundle_id"],
        "event_count": comparison["event_count"],
        "frozen_report_receipt": {
            "materialization_manifest_sha256": _file_sha256(
                report_dir / "manifest.json"
            ),
            "bundle_sha256": report_hashes["bundle.json"],
            "report_html_sha256": report_hashes["report.html"],
            "report_docx_sha256": report_hashes["report.docx"],
            "report_status": manifest["status"],
            "report_frozen_before_excel_loaded": True,
        },
        "excel_review_receipt": {
            "status": comparison["excel_review_status"],
            "source_file_sha256": review_hash,
            "validated_payload_sha256": (
                _canonical_sha256(validated_review)
                if validated_review is not None
                else None
            ),
            "raw_excel_text_loaded": False,
            "source_path_persisted": False,
        },
        "events": comparison["events"],
        "comparison_summary": comparison["comparison_summary"],
        "comparison_policy": {
            "eeg_fact_type": "ictal_onset_pattern",
            "field_projection": {
                "laterality": ["laterality"],
                "regions": ["regions"],
                "patterns": ["onset_type", "morphology"],
                "electrodes": ["electrodes", "maximal_electrodes"],
                "derivations": ["derivations"],
            },
            "values_compared_as_canonical_sets": True,
            "missing_either_side_status": "not_available",
            "nonempty_nonexact_intersection_status": "partial_match",
            "empty_intersection_status": "mismatch",
        },
        "claim_boundary": {
            "postfreeze_only": True,
            "comparison_fields": list(COMPARISON_FIELDS),
            "ictal_onset_pattern_facts_only": True,
            "edf_annotations_loaded": False,
            "report_bundle_modified": False,
            "eeg_facts_modified": False,
            "impression_modified": False,
            "waveform_selection_modified": False,
            "detector_or_ranking_modified": False,
            "renderer_used_excel": False,
            "llm_used_excel": False,
            "diagnostic_claim_generated": False,
        },
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                artifact,
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        os.chmod(target, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()
    return deepcopy(artifact)


__all__ = [
    "COMPARISON_FIELDS",
    "CONSISTENCY_ARTIFACT_SCHEMA_VERSION",
    "EXCEL_REVIEW_SCHEMA_VERSION",
    "compare_frozen_bundle_with_excel_review",
    "materialize_postfreeze_excel_consistency",
    "validate_excel_onset_review",
    "verify_frozen_eeg_report_bundle",
]
