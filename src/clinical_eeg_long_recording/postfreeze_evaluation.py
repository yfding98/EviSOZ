"""Independent post-freeze evaluation for long-recording EEG reports.

The report bundle is the authoritative, EEG-signal-only output.  This module
verifies its manifest and every frozen body hash *before* opening an optional
typed review sidecar.  The sidecar is useful for retrospective evaluation of
Excel onset labels and physician electrode references, but it is never an
input to detection, event selection, SOZ ranking, clinical facts, the LLM, or
either report renderer.

Only closed, de-identified values are accepted.  Raw Excel cells, EDF
annotations, identities, paths, clinical narratives and free text have no
field in the schema.  Missing labels remain ``"not_available"`` and are
reported in coverage denominators instead of being silently scored as zero or
dropped from the audit.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from src.clinical_eeg_report.schema import (
    canonicalize_electrode,
    validate_report_payload,
)

from .aggregation import validate_trustworthy_long_term_clinical_eeg_bundle
from .excel_consistency import verify_frozen_eeg_report_bundle


POSTFREEZE_EVALUATION_INPUT_SCHEMA_VERSION = (
    "postfreeze_clinical_eeg_evaluation_input_v1"
)
POSTFREEZE_EVALUATION_ARTIFACT_SCHEMA_VERSION = (
    "postfreeze_clinical_eeg_evaluation_v1"
)
PHYSICIAN_SPREAD_SOFT_WEIGHT = 0.35
RESEARCH_RANKING_INTERPRETATION_STATUS = (
    "research_scalp_electrode_ranking_not_clinical_soz"
)

SEMANTIC_FIELDS = ("laterality", "regions", "onset_uncertainty")
SEMANTIC_SOURCES = (
    "report_onset_conclusion",
    "ictal_onset_pattern",
    "report_signal_change",
    "research_soz_top1_projection",
)
COMPARISON_STATUSES = frozenset(
    {"match", "partial_match", "mismatch", "not_available"}
)

HARD_METRICS = (
    "top1_hit",
    "top3_hit",
    "top5_hit",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "mrr",
    "average_precision",
)
SOFT_METRICS = (
    "weighted_recall_at_1",
    "weighted_recall_at_3",
    "weighted_recall_at_5",
    "linear_gain_ndcg_at_1",
    "linear_gain_ndcg_at_3",
    "linear_gain_ndcg_at_5",
    "top1_gain",
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
_ONSET_UNCERTAINTY = frozenset(
    {"clear", "uncertain_or_unclear", "indeterminate"}
)

_STANDARD_19 = frozenset(
    {
        "FP1",
        "FP2",
        "F7",
        "F8",
        "F3",
        "F4",
        "FZ",
        "C3",
        "C4",
        "CZ",
        "T7",
        "T8",
        "P7",
        "P8",
        "P3",
        "P4",
        "PZ",
        "O1",
        "O2",
    }
)
_REFERENCE_SCOPES: Mapping[str, frozenset[str]] = {
    "standard_19_monopolar_electrodes": _STANDARD_19,
    "standard_19_plus_m1_m2_reference_electrodes": _STANDARD_19
    | frozenset({"M1", "M2"}),
}

# This is the frozen research ranker's exclusive five-region label protocol.
# The temporal aliases T3/T4/T5/T6 and A1/A2 have already been canonicalized.
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
_MODEL_REGION_TO_CLINICAL_REGIONS: Mapping[str, tuple[str, ...]] = {
    "left_frontal": ("frontal",),
    "right_frontal": ("frontal",),
    "left_temporal": ("temporal",),
    "right_temporal": ("temporal",),
    "central_parietal": ("central", "parietal"),
}
_REGION_EXPANSION: Mapping[str, tuple[str, ...]] = {
    "frontotemporal": ("frontal", "temporal"),
    "centrotemporal": ("central", "temporal"),
    "temporoparietal": ("temporal", "parietal"),
    "posterior": ("parietal", "occipital"),
}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EVALUATION_ID_RE = re.compile(r"^EVAL-[0-9a-f]{24}$")
_EXCEL_REVIEW_ID_RE = re.compile(r"^XLSREVIEW-[0-9a-f]{24}$")
_SOZ_REFERENCE_ID_RE = re.compile(r"^SOZREF-[0-9a-f]{24}$")
_UNMATCHED_REFERENCE_ID_RE = re.compile(r"^UNMATCHED-[0-9a-f]{24}$")

_INPUT_KEYS = {
    "schema_version",
    "evaluation_id",
    "recording_id",
    "bundle_id",
    "events",
    "unmatched_references",
    "claim_boundary",
}
_EVENT_KEYS = {
    "eeg_event_id",
    "event_binding_verified",
    "excel_onset_review",
    "physician_channel_reference",
}
_UNMATCHED_KEYS = {
    "unmatched_reference_id",
    "excel_onset_review",
    "physician_channel_reference",
}
_EXCEL_KEYS = {
    "review_id",
    "laterality",
    "regions",
    "onset_uncertainty",
}
_PHYSICIAN_REFERENCE_KEYS = {
    "reference_id",
    "reference_scope",
    "review_status",
    "reference_completeness",
    "significant_electrodes",
    "spread_electrodes",
}
_INPUT_CLAIM_BOUNDARY = {
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


def _optional_regions(value: object, context: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise TypeError(f"{context} must be null or a non-empty list")
    result: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or raw not in _REGIONS:
            raise ValueError(f"{context} contains an unsupported controlled code")
        if raw in result:
            raise ValueError(f"{context} contains duplicate values")
        result.append(raw)
    return result


def _optional_electrodes(
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
        canonical = canonicalize_electrode(raw)
        if canonical not in allowed:
            raise ValueError(f"{context} contains an electrode outside reference_scope")
        if canonical in result:
            raise ValueError(f"{context} contains duplicate canonical electrodes")
        result.append(canonical)
    return result


def _validate_excel_review(
    value: object | None,
    *,
    context: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    data = _strict_object(value, keys=_EXCEL_KEYS, context=context)
    review_id = _opaque_identifier(
        data["review_id"], _EXCEL_REVIEW_ID_RE, f"{context}.review_id"
    )
    laterality = data["laterality"]
    if laterality is not None and laterality not in _LATERALITIES:
        raise ValueError(f"{context}.laterality is not a controlled code")
    uncertainty = data["onset_uncertainty"]
    if uncertainty is not None and uncertainty not in _ONSET_UNCERTAINTY:
        raise ValueError(f"{context}.onset_uncertainty is not a controlled code")
    return {
        "review_id": review_id,
        "laterality": laterality,
        "regions": _optional_regions(data["regions"], f"{context}.regions"),
        "onset_uncertainty": uncertainty,
    }


def _validate_physician_reference(
    value: object | None,
    *,
    context: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    data = _strict_object(
        value,
        keys=_PHYSICIAN_REFERENCE_KEYS,
        context=context,
    )
    reference_id = _opaque_identifier(
        data["reference_id"],
        _SOZ_REFERENCE_ID_RE,
        f"{context}.reference_id",
    )
    scope = data["reference_scope"]
    if scope not in _REFERENCE_SCOPES:
        raise ValueError(f"{context}.reference_scope is not supported")
    if data["review_status"] != "physician_typed_reference":
        raise ValueError(
            f"{context}.review_status must be physician_typed_reference"
        )
    completeness = data["reference_completeness"]
    if completeness not in {"exhaustive", "positive_only_unknown_complement"}:
        raise ValueError(
            f"{context}.reference_completeness is not a controlled code"
        )
    allowed = _REFERENCE_SCOPES[scope]
    significant = _optional_electrodes(
        data["significant_electrodes"],
        allowed=allowed,
        context=f"{context}.significant_electrodes",
    )
    spread = _optional_electrodes(
        data["spread_electrodes"],
        allowed=allowed,
        context=f"{context}.spread_electrodes",
    )
    if significant is None and spread is None:
        raise ValueError(
            f"{context} must contain significant or spread electrode labels"
        )
    significant_set = set(significant or [])
    # A channel explicitly identified as significant is always hard GT.  It is
    # removed from the weaker spread set instead of being counted twice.
    resolved_spread = [item for item in spread or [] if item not in significant_set]
    return {
        "reference_id": reference_id,
        "reference_scope": scope,
        "review_status": "physician_typed_reference",
        "reference_completeness": completeness,
        "significant_electrodes": significant,
        "spread_electrodes": resolved_spread if spread is not None else None,
        "hard_overrides_spread": True,
    }


def validate_postfreeze_evaluation_input(
    value: object,
    *,
    expected_recording_id: str,
    expected_bundle_id: str,
    expected_event_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate a PHI-free evaluation sidecar against frozen event IDs."""

    data = _strict_object(
        value,
        keys=_INPUT_KEYS,
        context="post-freeze evaluation input",
    )
    if data["schema_version"] != POSTFREEZE_EVALUATION_INPUT_SCHEMA_VERSION:
        raise ValueError(
            "post-freeze evaluation input must use "
            f"{POSTFREEZE_EVALUATION_INPUT_SCHEMA_VERSION}"
        )
    data["evaluation_id"] = _opaque_identifier(
        data["evaluation_id"], _EVALUATION_ID_RE, "evaluation_id"
    )
    data["recording_id"] = _identifier(data["recording_id"], "recording_id")
    data["bundle_id"] = _identifier(data["bundle_id"], "bundle_id")
    if data["recording_id"] != expected_recording_id:
        raise ValueError("evaluation recording_id does not match the frozen bundle")
    if data["bundle_id"] != expected_bundle_id:
        raise ValueError("evaluation bundle_id does not match the frozen bundle")

    boundary = _strict_object(
        data["claim_boundary"],
        keys=set(_INPUT_CLAIM_BOUNDARY),
        context="evaluation claim_boundary",
    )
    for key, expected in _INPUT_CLAIM_BOUNDARY.items():
        if boundary[key] is not expected:
            raise ValueError(f"evaluation claim_boundary.{key} must be {expected}")
    data["claim_boundary"] = dict(_INPUT_CLAIM_BOUNDARY)

    expected_ids = list(expected_event_ids)
    expected_id_set = set(expected_ids)
    raw_events = data["events"]
    if not isinstance(raw_events, list):
        raise TypeError("evaluation events must be a list")
    seen_event_ids: set[str] = set()
    seen_review_ids: set[str] = set()
    seen_reference_ids: set[str] = set()
    events: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_events):
        context = f"evaluation events[{index}]"
        item = _strict_object(raw, keys=_EVENT_KEYS, context=context)
        event_id = _identifier(item["eeg_event_id"], f"{context}.eeg_event_id")
        if event_id not in expected_id_set:
            raise ValueError("evaluation references an unknown EEG event")
        if event_id in seen_event_ids:
            raise ValueError("evaluation contains repeated EEG event bindings")
        if item["event_binding_verified"] is not True:
            raise ValueError("evaluation event binding must be verified")
        excel = _validate_excel_review(
            item["excel_onset_review"],
            context=f"{context}.excel_onset_review",
        )
        physician = _validate_physician_reference(
            item["physician_channel_reference"],
            context=f"{context}.physician_channel_reference",
        )
        if excel is None and physician is None:
            raise ValueError("a bound evaluation event must contain a typed reference")
        if excel is not None:
            if excel["review_id"] in seen_review_ids:
                raise ValueError("evaluation contains a duplicate Excel review ID")
            seen_review_ids.add(excel["review_id"])
        if physician is not None:
            if physician["reference_id"] in seen_reference_ids:
                raise ValueError("evaluation contains a duplicate SOZ reference ID")
            seen_reference_ids.add(physician["reference_id"])
        seen_event_ids.add(event_id)
        events.append(
            {
                "eeg_event_id": event_id,
                "event_binding_verified": True,
                "excel_onset_review": excel,
                "physician_channel_reference": physician,
            }
        )
    events.sort(key=lambda item: expected_ids.index(item["eeg_event_id"]))
    data["events"] = events

    raw_unmatched = data["unmatched_references"]
    if not isinstance(raw_unmatched, list):
        raise TypeError("evaluation unmatched_references must be a list")
    seen_unmatched_ids: set[str] = set()
    unmatched: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_unmatched):
        context = f"evaluation unmatched_references[{index}]"
        item = _strict_object(raw, keys=_UNMATCHED_KEYS, context=context)
        unmatched_id = _opaque_identifier(
            item["unmatched_reference_id"],
            _UNMATCHED_REFERENCE_ID_RE,
            f"{context}.unmatched_reference_id",
        )
        if unmatched_id in seen_unmatched_ids:
            raise ValueError("evaluation contains duplicate unmatched reference IDs")
        excel = _validate_excel_review(
            item["excel_onset_review"],
            context=f"{context}.excel_onset_review",
        )
        physician = _validate_physician_reference(
            item["physician_channel_reference"],
            context=f"{context}.physician_channel_reference",
        )
        if excel is None and physician is None:
            raise ValueError("an unmatched record must contain a typed reference")
        if excel is not None:
            if excel["review_id"] in seen_review_ids:
                raise ValueError("evaluation contains a duplicate Excel review ID")
            seen_review_ids.add(excel["review_id"])
        if physician is not None:
            if physician["reference_id"] in seen_reference_ids:
                raise ValueError("evaluation contains a duplicate SOZ reference ID")
            seen_reference_ids.add(physician["reference_id"])
        seen_unmatched_ids.add(unmatched_id)
        unmatched.append(
            {
                "unmatched_reference_id": unmatched_id,
                "binding_status": "unmatched_reference_no_eeg_event",
                "excel_onset_review": excel,
                "physician_channel_reference": physician,
            }
        )
    data["unmatched_references"] = unmatched
    return data


def _expanded_laterality(value: object) -> list[str]:
    if value == "bilateral":
        return ["left", "right"]
    if value in {"left", "right", "midline"}:
        return [str(value)]
    return []


def _expanded_regions(values: object) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    result: list[str] = []
    for raw in values:
        if raw in {"unknown", None}:
            continue
        expanded = _REGION_EXPANSION.get(str(raw), (str(raw),))
        for item in expanded:
            if item not in result:
                result.append(item)
    return sorted(result)


def _expanded_uncertainty(value: object) -> list[str]:
    if value in {"clear", "uncertain_or_unclear"}:
        return [str(value)]
    return []


def _semantic_values(
    *,
    laterality: object,
    regions: object,
    onset_uncertainty: object,
) -> dict[str, list[str]]:
    return {
        "laterality": _expanded_laterality(laterality),
        "regions": _expanded_regions(regions),
        "onset_uncertainty": _expanded_uncertainty(onset_uncertainty),
    }


def _semantic_comparison(
    field: str,
    eeg_values: Sequence[str],
    excel_values: Sequence[str],
) -> dict[str, Any]:
    eeg = sorted(set(eeg_values))
    excel = sorted(set(excel_values))
    overlap = sorted(set(eeg).intersection(excel)) if eeg and excel else []
    if not eeg or not excel:
        status = "not_available"
    elif eeg == excel:
        status = "match"
    elif overlap and field != "onset_uncertainty":
        status = "partial_match"
    else:
        status = "mismatch"
    if status not in COMPARISON_STATUSES:
        raise AssertionError("unreachable semantic comparison status")
    return {
        "field": field,
        "status": status,
        "eeg_available": bool(eeg),
        "excel_available": bool(excel),
        "eeg_values": eeg,
        "excel_values": excel,
        "overlap_values": overlap,
    }


def _ictal_onset_source(event: Mapping[str, Any]) -> dict[str, Any]:
    event_id = str(event["eeg_event_id"])
    report = validate_report_payload(event["event_report_payload"]).to_dict()
    matches = [
        fact
        for fact in report["facts"]
        if fact["fact_type"] == "ictal_onset_pattern"
        and fact.get("eeg_event_id") == event_id
    ]
    if len(matches) != 1:
        return {
            "source_status": "not_available",
            "fact_id": "not_available",
            "interpretation_status": (
                "structured_report_fact_not_independent_ground_truth"
            ),
            "verification_status": "not_available",
            "semantic_values": _semantic_values(
                laterality=None,
                regions=None,
                onset_uncertainty=None,
            ),
        }
    fact = matches[0]
    uncertainty = {
        "present": "clear",
        "uncertain": "uncertain_or_unclear",
    }.get(str(fact["state"]))
    value = fact["value"] if isinstance(fact.get("value"), Mapping) else {}
    return {
        "source_status": "available",
        "fact_id": fact["fact_id"],
        "interpretation_status": (
            "structured_report_fact_not_independent_ground_truth"
        ),
        "verification_status": fact["verification"]["status"],
        "semantic_values": _semantic_values(
            laterality=value.get("laterality"),
            regions=value.get("regions"),
            onset_uncertainty=uncertainty,
        ),
    }


def _report_onset_conclusion_source(event: Mapping[str, Any]) -> dict[str, Any]:
    """Project the frozen report's explicit onset-certainty conclusion.

    This is deliberately distinct from both the raw ``ictal_onset_pattern``
    fact and the neutral sustained-signal-change source.  In the deterministic
    long-recording report, an event with no physician-confirmed onset pattern
    is explicitly described as a non-confirmed/uncertain onset.  Previously
    that rendered conclusion was lost during evaluation: an Excel
    ``uncertain_or_unclear`` label was compared with an empty value even when
    the frozen report explicitly carried the same uncertainty boundary.

    The projection is made only from the already frozen, validated structured
    ledger.  It never parses Excel text or report prose, and it never changes
    the report.  Lack of an onset fact alone is not enough: a bound
    ``electrographic_event_occurrence`` fact must establish that this is a
    reported event for which the renderer emits the onset boundary.
    """

    event_id = str(event["eeg_event_id"])
    report = validate_report_payload(event["event_report_payload"]).to_dict()
    onset_facts = [
        fact
        for fact in report["facts"]
        if fact["fact_type"] == "ictal_onset_pattern"
        and fact.get("eeg_event_id") == event_id
    ]
    occurrence_facts = [
        fact
        for fact in report["facts"]
        if fact["fact_type"] == "electrographic_event_occurrence"
        and fact.get("eeg_event_id") == event_id
    ]

    uncertainty: str | None = None
    projection_basis = "not_available"
    basis_fact_ids: list[str] = []
    if len(onset_facts) == 1:
        onset = onset_facts[0]
        state = str(onset["state"])
        verification = str(onset["verification"]["status"])
        basis_fact_ids.append(str(onset["fact_id"]))
        if state == "present" and verification == "physician_verified":
            uncertainty = "clear"
            projection_basis = "physician_verified_present_onset_pattern"
        elif state in {"present", "uncertain", "absent", "not_assessable"}:
            # The frozen automatic impression does not promote an unverified
            # or algorithm-candidate pattern to a confirmed onset.  Explicit
            # absent/not-assessable states likewise express an unclear onset.
            uncertainty = "uncertain_or_unclear"
            projection_basis = (
                "unconfirmed_or_explicitly_unclear_onset_pattern"
            )
        # ``not_recorded`` is missing evidence, not an uncertainty assertion.
    elif not onset_facts and len(occurrence_facts) == 1:
        occurrence = occurrence_facts[0]
        if str(occurrence["state"]) in {"present", "uncertain"}:
            uncertainty = "uncertain_or_unclear"
            projection_basis = "reported_event_without_qualified_onset_pattern"
            basis_fact_ids.append(str(occurrence["fact_id"]))

    return {
        "source_status": "available" if uncertainty is not None else "not_available",
        "basis_fact_ids": basis_fact_ids,
        "projection_basis": projection_basis,
        "interpretation_status": (
            "structured_frozen_report_onset_certainty_not_independent_ground_truth"
        ),
        "semantic_values": _semantic_values(
            laterality=None,
            regions=None,
            onset_uncertainty=uncertainty,
        ),
    }


def _report_signal_change_source(event: Mapping[str, Any]) -> dict[str, Any]:
    """Project a qualified neutral signal change without promoting it to onset."""

    event_id = str(event["eeg_event_id"])
    report = validate_report_payload(event["event_report_payload"]).to_dict()
    matches = [
        fact
        for fact in report["facts"]
        if fact["fact_type"] == "algorithmic_sustained_eeg_change"
        and fact.get("eeg_event_id") == event_id
    ]
    if len(matches) != 1:
        return {
            "source_status": "not_available",
            "fact_id": "not_available",
            "interpretation_status": (
                "neutral_qualified_signal_change_not_confirmed_onset"
            ),
            "semantic_values": _semantic_values(
                laterality=None,
                regions=None,
                onset_uncertainty=None,
            ),
        }
    fact = matches[0]
    value = fact["value"]
    laterality = value.get("laterality")
    regions = value.get("regions")
    semantic_values = _semantic_values(
        laterality=laterality,
        regions=regions,
        # A neutral sustained signal change does not assert onset clarity.
        onset_uncertainty=None,
    )
    spatial_available = bool(
        semantic_values["laterality"] or semantic_values["regions"]
    )
    return {
        "source_status": (
            "available" if spatial_available else "spatial_semantics_not_available"
        ),
        "fact_id": fact["fact_id"],
        "interpretation_status": (
            "neutral_qualified_signal_change_not_confirmed_onset"
        ),
        "semantic_values": semantic_values,
    }


def _ranking_source(event: Mapping[str, Any]) -> dict[str, Any]:
    receipt = event["research_soz_ranking_receipt"]
    ranking = receipt["ranked_electrodes"]
    if not ranking:
        return {
            "source_status": "abstained",
            "top1_electrode": "not_available",
            "model_region": "not_available",
            "semantic_values": _semantic_values(
                laterality=None,
                regions=None,
                onset_uncertainty=None,
            ),
        }
    electrode = canonicalize_electrode(ranking[0]["electrode"])
    model_region = _ELECTRODE_TO_MODEL_REGION.get(electrode)
    laterality = _ELECTRODE_TO_LATERALITY.get(electrode)
    clinical_regions = _MODEL_REGION_TO_CLINICAL_REGIONS.get(model_region, ())
    source_status = (
        "available" if model_region is not None and laterality is not None
        else "projection_not_available"
    )
    return {
        "source_status": source_status,
        "top1_electrode": electrode,
        "model_region": model_region,
        "semantic_values": _semantic_values(
            laterality=laterality,
            regions=clinical_regions,
            # Ranking scores do not encode an onset-clear/unclear assertion.
            onset_uncertainty=None,
        ),
    }


def _excel_semantic_values(
    excel_review: Mapping[str, Any] | None,
) -> dict[str, list[str]]:
    if excel_review is None:
        return _semantic_values(
            laterality=None,
            regions=None,
            onset_uncertainty=None,
        )
    return _semantic_values(
        laterality=excel_review["laterality"],
        regions=excel_review["regions"],
        onset_uncertainty=excel_review["onset_uncertainty"],
    )


def _clean_float(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("evaluation metric must be finite")
    return round(float(value), 12)


def _hard_metrics(
    ranking: Sequence[str],
    hard_labels: Sequence[str] | None,
    *,
    reference_completeness: str | None,
) -> dict[str, float | str]:
    if not hard_labels or not ranking:
        return {metric: "not_available" for metric in HARD_METRICS}
    relevant = set(hard_labels)
    values: dict[str, float | str] = {}
    for k in (1, 3, 5):
        prefix = list(ranking[:k])
        hits = sum(item in relevant for item in prefix)
        values[f"top{k}_hit"] = float(hits > 0)
        values[f"recall_at_{k}"] = (
            _clean_float(hits / len(relevant))
            if reference_completeness == "exhaustive"
            else "not_available"
        )
    first_rank = next(
        (index for index, item in enumerate(ranking, start=1) if item in relevant),
        None,
    )
    values["mrr"] = _clean_float(1.0 / first_rank) if first_rank else 0.0
    relevant_seen = 0
    precision_sum = 0.0
    for rank, electrode in enumerate(ranking, start=1):
        if electrode in relevant:
            relevant_seen += 1
            precision_sum += relevant_seen / rank
    values["average_precision"] = (
        _clean_float(precision_sum / len(relevant))
        if reference_completeness == "exhaustive"
        else "not_available"
    )
    return {metric: values[metric] for metric in HARD_METRICS}


def _dcg(gains: Sequence[float]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))


def _soft_metrics(
    ranking: Sequence[str],
    label_weights: Mapping[str, float],
    *,
    reference_completeness: str | None,
) -> dict[str, float | str]:
    if (
        not label_weights
        or not ranking
        or reference_completeness != "exhaustive"
    ):
        return {metric: "not_available" for metric in SOFT_METRICS}
    total_gain = sum(label_weights.values())
    values: dict[str, float | str] = {}
    ideal_all = sorted(label_weights.values(), reverse=True)
    for k in (1, 3, 5):
        actual_gains = [label_weights.get(item, 0.0) for item in ranking[:k]]
        values[f"weighted_recall_at_{k}"] = _clean_float(
            sum(actual_gains) / total_gain
        )
        ideal_gains = ideal_all[:k]
        ideal = _dcg(ideal_gains)
        values[f"linear_gain_ndcg_at_{k}"] = (
            _clean_float(_dcg(actual_gains) / ideal) if ideal > 0 else 0.0
        )
    values["top1_gain"] = _clean_float(label_weights.get(ranking[0], 0.0))
    return {metric: values[metric] for metric in SOFT_METRICS}


def _metric_summary(
    events: Sequence[Mapping[str, Any]],
    *,
    family: str,
    metrics: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in metrics:
        available = [
            item["soz_ranking_evaluation"][family][metric]
            for item in events
            if item["soz_ranking_evaluation"][family][metric]
            != "not_available"
        ]
        result[metric] = {
            "status": "available" if available else "not_available",
            "available_event_count": len(available),
            "not_available_event_count": len(events) - len(available),
            "macro_average": (
                _clean_float(sum(float(value) for value in available) / len(available))
                if available
                else "not_available"
            ),
        }
    return result


def _unmatched_reference_audit(
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose coverage without publishing unbound label values as evidence."""

    result: list[dict[str, Any]] = []
    for item in items:
        excel = item["excel_onset_review"]
        physician = item["physician_channel_reference"]
        result.append(
            {
                "unmatched_reference_id": item["unmatched_reference_id"],
                "binding_status": "unmatched_reference_no_eeg_event",
                "excel_onset_review_status": (
                    "available" if excel is not None else "not_available"
                ),
                "excel_review_id": (
                    excel["review_id"] if excel is not None else "not_available"
                ),
                "physician_channel_reference_status": (
                    "available" if physician is not None else "not_available"
                ),
                "physician_reference_id": (
                    physician["reference_id"]
                    if physician is not None
                    else "not_available"
                ),
                "reference_completeness": (
                    physician["reference_completeness"]
                    if physician is not None
                    else "not_available"
                ),
                "excluded_from_semantic_metrics": True,
                "excluded_from_soz_metrics": True,
            }
        )
    return result


def evaluate_frozen_bundle_with_typed_references(
    bundle: object,
    evaluation_input: object | None,
) -> dict[str, Any]:
    """Evaluate semantic consistency and research ranking without mutation."""

    frozen = validate_trustworthy_long_term_clinical_eeg_bundle(bundle)
    event_ids = [str(event["eeg_event_id"]) for event in frozen["events"]]
    validated = (
        validate_postfreeze_evaluation_input(
            evaluation_input,
            expected_recording_id=str(frozen["recording_id"]),
            expected_bundle_id=str(frozen["bundle_id"]),
            expected_event_ids=event_ids,
        )
        if evaluation_input is not None
        else None
    )
    binding_by_event = {
        item["eeg_event_id"]: item for item in validated["events"]
    } if validated is not None else {}

    semantic_counts: dict[str, dict[str, dict[str, int]]] = {
        source: {
            field: {status: 0 for status in sorted(COMPARISON_STATUSES)}
            for field in SEMANTIC_FIELDS
        }
        for source in SEMANTIC_SOURCES
    }
    events: list[dict[str, Any]] = []
    abstained = 0
    for event in frozen["events"]:
        event_id = str(event["eeg_event_id"])
        binding = binding_by_event.get(event_id)
        excel = binding["excel_onset_review"] if binding is not None else None
        physician = (
            binding["physician_channel_reference"] if binding is not None else None
        )
        excel_values = _excel_semantic_values(excel)
        report_onset_conclusion = _report_onset_conclusion_source(event)
        onset_source = _ictal_onset_source(event)
        signal_change_source = _report_signal_change_source(event)
        ranking_source = _ranking_source(event)
        if ranking_source["source_status"] == "abstained":
            abstained += 1

        semantic_sources: dict[str, Any] = {}
        for source_name, source in (
            ("report_onset_conclusion", report_onset_conclusion),
            ("ictal_onset_pattern", onset_source),
            ("report_signal_change", signal_change_source),
            ("research_soz_top1_projection", ranking_source),
        ):
            comparisons = {
                field: _semantic_comparison(
                    field,
                    source["semantic_values"][field],
                    excel_values[field],
                )
                for field in SEMANTIC_FIELDS
            }
            for field, comparison in comparisons.items():
                semantic_counts[source_name][field][comparison["status"]] += 1
            source_output = {
                key: deepcopy(value)
                for key, value in source.items()
                if key != "semantic_values"
            }
            source_output["comparisons"] = comparisons
            semantic_sources[source_name] = source_output

        ranking = [
            canonicalize_electrode(item["electrode"])
            for item in event["research_soz_ranking_receipt"]["ranked_electrodes"]
        ]
        hard_labels = (
            physician["significant_electrodes"] if physician is not None else None
        )
        spread_labels = (
            physician["spread_electrodes"] if physician is not None else None
        )
        reference_completeness = (
            physician["reference_completeness"]
            if physician is not None
            else None
        )
        label_weights = {
            electrode: 1.0 for electrode in hard_labels or []
        }
        for electrode in spread_labels or []:
            label_weights.setdefault(electrode, PHYSICIAN_SPREAD_SOFT_WEIGHT)
        hard = _hard_metrics(
            ranking,
            hard_labels,
            reference_completeness=reference_completeness,
        )
        soft = _soft_metrics(
            ranking,
            label_weights,
            reference_completeness=reference_completeness,
        )
        events.append(
            {
                "event_number": int(event["event_number"]),
                "eeg_event_id": event_id,
                "event_binding_status": (
                    "verified" if binding is not None else "not_available"
                ),
                "excel_onset_review_status": (
                    "available" if excel is not None else "not_available"
                ),
                "physician_channel_reference_status": (
                    "available" if physician is not None else "not_available"
                ),
                "semantic_consistency": semantic_sources,
                "soz_ranking_evaluation": {
                    "interpretation_status": RESEARCH_RANKING_INTERPRETATION_STATUS,
                    "ranking_status": ranking_source["source_status"],
                    "reference_completeness": (
                        reference_completeness
                        if reference_completeness is not None
                        else "not_available"
                    ),
                    "unmarked_electrode_semantics": (
                        "true_negative_within_reference_scope"
                        if reference_completeness == "exhaustive"
                        else (
                            "unknown_not_negative"
                            if reference_completeness
                            == "positive_only_unknown_complement"
                            else "not_available"
                        )
                    ),
                    "ranked_electrodes": ranking,
                    "significant_electrodes": (
                        deepcopy(hard_labels)
                        if hard_labels is not None
                        else "not_available"
                    ),
                    "spread_electrodes": (
                        deepcopy(spread_labels)
                        if spread_labels is not None
                        else "not_available"
                    ),
                    "label_weights": (
                        dict(sorted(label_weights.items()))
                        if label_weights
                        else "not_available"
                    ),
                    "hard_metrics": hard,
                    "soft_metrics": soft,
                },
            }
        )

    unmatched = validated["unmatched_references"] if validated is not None else []
    sidecar_bound = len(binding_by_event)
    soz_bound = sum(
        item["physician_channel_reference"] is not None
        for item in binding_by_event.values()
    )
    excel_bound = sum(
        item["excel_onset_review"] is not None
        for item in binding_by_event.values()
    )
    hard_available = sum(
        item["soz_ranking_evaluation"]["hard_metrics"]["top1_hit"]
        != "not_available"
        for item in events
    )
    soft_available = sum(
        item["soz_ranking_evaluation"]["soft_metrics"]["top1_gain"]
        != "not_available"
        for item in events
    )
    eligible = len(events)
    exhaustive_references = sum(
        item["physician_channel_reference"] is not None
        and item["physician_channel_reference"]["reference_completeness"]
        == "exhaustive"
        for item in binding_by_event.values()
    )
    positive_only_references = sum(
        item["physician_channel_reference"] is not None
        and item["physician_channel_reference"]["reference_completeness"]
        == "positive_only_unknown_complement"
        for item in binding_by_event.values()
    )
    unmatched_exhaustive = sum(
        item["physician_channel_reference"] is not None
        and item["physician_channel_reference"]["reference_completeness"]
        == "exhaustive"
        for item in unmatched
    )
    unmatched_positive_only = sum(
        item["physician_channel_reference"] is not None
        and item["physician_channel_reference"]["reference_completeness"]
        == "positive_only_unknown_complement"
        for item in unmatched
    )
    coverage = {
        "eligible_event_count": eligible,
        "sidecar_bound_event_count": sidecar_bound,
        "excel_review_bound_event_count": excel_bound,
        "bound_event_count": soz_bound,
        "missing_reference_event_count": eligible - soz_bound,
        "excess_candidate_event_count": eligible - soz_bound,
        "unmatched_reference_record_count": len(unmatched),
        "unmatched_excel_review_record_count": sum(
            item["excel_onset_review"] is not None for item in unmatched
        ),
        "unmatched_ground_truth_record_count": sum(
            item["physician_channel_reference"] is not None for item in unmatched
        ),
        "abstained_event_count": abstained,
        "exhaustive_reference_event_count": exhaustive_references,
        "positive_only_unknown_complement_event_count": positive_only_references,
        "unmatched_exhaustive_ground_truth_record_count": unmatched_exhaustive,
        "unmatched_positive_only_ground_truth_record_count": unmatched_positive_only,
        "hard_metric_available_event_count": hard_available,
        "hard_metric_not_available_event_count": eligible - hard_available,
        "soft_metric_available_event_count": soft_available,
        "soft_metric_not_available_event_count": eligible - soft_available,
    }
    exhaustive_events = [
        item
        for item in events
        if item["soz_ranking_evaluation"]["reference_completeness"]
        == "exhaustive"
    ]
    positive_only_events = [
        item
        for item in events
        if item["soz_ranking_evaluation"]["reference_completeness"]
        == "positive_only_unknown_complement"
    ]
    return {
        "recording_id": frozen["recording_id"],
        "bundle_id": frozen["bundle_id"],
        "event_count": eligible,
        "evaluation_input_status": (
            "available" if validated is not None else "not_available"
        ),
        "events": events,
        "coverage": coverage,
        "semantic_consistency_summary": semantic_counts,
        "hard_metric_summary": _metric_summary(
            events,
            family="hard_metrics",
            metrics=HARD_METRICS,
        ),
        "hard_metric_summary_by_reference_completeness": {
            "exhaustive": _metric_summary(
                exhaustive_events,
                family="hard_metrics",
                metrics=HARD_METRICS,
            ),
            "positive_only_unknown_complement": _metric_summary(
                positive_only_events,
                family="hard_metrics",
                metrics=HARD_METRICS,
            ),
        },
        "soft_metric_summary": _metric_summary(
            events,
            family="soft_metrics",
            metrics=SOFT_METRICS,
        ),
        "unmatched_references": _unmatched_reference_audit(unmatched),
        "validated_evaluation_input": validated,
    }


def _atomic_private_json(target: Path, value: object) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        os.chmod(target, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize_postfreeze_clinical_eeg_evaluation(
    *,
    report_bundle_dir: str | Path,
    evaluation_input_path: str | Path | None,
    output_path: str | Path,
) -> dict[str, Any]:
    """Verify the frozen report, then publish a separate private evaluation."""

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
        raise ValueError("evaluation artifact must be outside the frozen report bundle")
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)

    # Security and causality boundary: no sidecar path is resolved or opened
    # until all frozen EEG report bodies and the EEG-only scope are verified.
    manifest, bundle, report_hashes = verify_frozen_eeg_report_bundle(report_dir)

    raw_input_path = (
        Path(evaluation_input_path) if evaluation_input_path is not None else None
    )
    if raw_input_path is not None and raw_input_path.is_symlink():
        raise ValueError("evaluation input must not be a symlink")
    input_path = (
        raw_input_path.resolve(strict=True) if raw_input_path is not None else None
    )
    input_payload = _json_object(input_path) if input_path is not None else None
    result = evaluate_frozen_bundle_with_typed_references(bundle, input_payload)
    validated_input = result.pop("validated_evaluation_input")

    artifact = {
        "schema_version": POSTFREEZE_EVALUATION_ARTIFACT_SCHEMA_VERSION,
        "status": "completed_postfreeze_evaluation",
        "recording_id": result["recording_id"],
        "bundle_id": result["bundle_id"],
        "event_count": result["event_count"],
        "frozen_report_receipt": {
            "materialization_manifest_sha256": _file_sha256(
                report_dir / "manifest.json"
            ),
            "bundle_sha256": report_hashes["bundle.json"],
            "report_html_sha256": report_hashes["report.html"],
            "report_docx_sha256": report_hashes["report.docx"],
            "report_status": manifest["status"],
            "report_frozen_before_evaluation_input_loaded": True,
        },
        "evaluation_input_receipt": {
            "status": result["evaluation_input_status"],
            "source_file_sha256": (
                _file_sha256(input_path) if input_path is not None else None
            ),
            "validated_payload_sha256": (
                _canonical_sha256(validated_input)
                if validated_input is not None
                else None
            ),
            "evaluation_id": (
                validated_input["evaluation_id"]
                if validated_input is not None
                else None
            ),
            "raw_excel_text_loaded": False,
            "raw_edf_annotation_loaded": False,
            "source_path_persisted": False,
        },
        "events": result["events"],
        "coverage": result["coverage"],
        "semantic_consistency_summary": result["semantic_consistency_summary"],
        "hard_metric_summary": result["hard_metric_summary"],
        "hard_metric_summary_by_reference_completeness": result[
            "hard_metric_summary_by_reference_completeness"
        ],
        "soft_metric_summary": result["soft_metric_summary"],
        "unmatched_references": result["unmatched_references"],
        "evaluation_policy": {
            "excel_onset_fields": list(SEMANTIC_FIELDS),
            "semantic_sources": list(SEMANTIC_SOURCES),
            "ictal_onset_fact_interpretation": (
                "structured_report_fact_not_independent_ground_truth"
            ),
            "ictal_onset_uncertainty_projection": {
                "present": "clear",
                "uncertain": "uncertain_or_unclear",
                "absent_not_recorded_not_assessable": "not_available",
            },
            "primary_onset_uncertainty_source": "report_onset_conclusion",
            "report_onset_conclusion_projection": {
                "physician_verified_present_onset_pattern": "clear",
                "unconfirmed_present_onset_pattern": "uncertain_or_unclear",
                "uncertain_absent_or_not_assessable_onset_pattern": (
                    "uncertain_or_unclear"
                ),
                "reported_event_without_qualified_onset_pattern": (
                    "uncertain_or_unclear"
                ),
                "not_recorded_or_no_reported_event": "not_available",
            },
            "report_onset_conclusion_uses_structured_frozen_facts_only": True,
            "report_onset_conclusion_parses_report_or_excel_free_text": False,
            "report_signal_change_interpretation": (
                "neutral_qualified_signal_change_not_confirmed_onset"
            ),
            "report_signal_change_uncertainty_projection_forbidden": True,
            "composite_region_expansion": {
                key: list(value) for key, value in _REGION_EXPANSION.items()
            },
            "unknown_none_or_indeterminate_is_not_available": True,
            "ranking_top1_projection_protocol": (
                "exclusive_standard_19_five_region_protocol"
            ),
            "ranking_interpretation_status": (
                RESEARCH_RANKING_INTERPRETATION_STATUS
            ),
            "physician_significant_electrode_role": "hard_ground_truth",
            "physician_significant_electrode_weight": 1.0,
            "physician_spread_electrode_role": "soft_label",
            "physician_spread_electrode_weight": PHYSICIAN_SPREAD_SOFT_WEIGHT,
            "hard_overrides_spread": True,
            "missing_labels_are_not_scored_as_zero": True,
            "macro_average_available_events_only": True,
            "coverage_counts_retain_missing_and_unmatched_records": True,
            "reference_completeness_required": True,
            "positive_only_unknown_complement_unmarked_channels": "unknown_not_negative",
            "positive_only_available_hard_metrics": [
                "top1_hit",
                "top3_hit",
                "top5_hit",
                "mrr",
            ],
            "positive_only_hard_metric_interpretation": (
                "known_positive_retrieval_only_not_negative_class_evaluation"
            ),
            "exhaustive_reference_required_for": [
                "recall_at_1",
                "recall_at_3",
                "recall_at_5",
                "average_precision",
                "weighted_recall_at_1",
                "weighted_recall_at_3",
                "weighted_recall_at_5",
                "linear_gain_ndcg_at_1",
                "linear_gain_ndcg_at_3",
                "linear_gain_ndcg_at_5",
                "top1_gain",
            ],
            "negative_class_metrics_reported": False,
            "auroc_specificity_fpr_or_accuracy_reported": False,
        },
        "claim_boundary": {
            "postfreeze_only": True,
            "typed_deidentified_reference_only": True,
            "raw_excel_text_loaded": False,
            "edf_annotations_loaded": False,
            "report_bundle_modified": False,
            "eeg_facts_modified": False,
            "impression_modified": False,
            "waveform_selection_modified": False,
            "detector_modified": False,
            "research_ranking_modified": False,
            "renderer_used_evaluation_input": False,
            "llm_used_evaluation_input": False,
            "diagnostic_claim_generated": False,
        },
    }
    _atomic_private_json(target, artifact)
    return deepcopy(artifact)


__all__ = [
    "COMPARISON_STATUSES",
    "HARD_METRICS",
    "PHYSICIAN_SPREAD_SOFT_WEIGHT",
    "POSTFREEZE_EVALUATION_ARTIFACT_SCHEMA_VERSION",
    "POSTFREEZE_EVALUATION_INPUT_SCHEMA_VERSION",
    "SEMANTIC_FIELDS",
    "SEMANTIC_SOURCES",
    "SOFT_METRICS",
    "evaluate_frozen_bundle_with_typed_references",
    "materialize_postfreeze_clinical_eeg_evaluation",
    "validate_postfreeze_evaluation_input",
]
