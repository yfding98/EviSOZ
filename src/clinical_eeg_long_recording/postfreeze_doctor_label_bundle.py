"""Post-freeze doctor-label publication for private long-recording EEG reports.

This module is deliberately outside every EEG generation path.  It first
validates a completed, release-audited cohort and re-hashes every selected
report artifact.  Only after that phase has closed may it open the private
doctor workbooks.  Workbook values are reduced to controlled codes and
evaluation-only electrode sets; raw cells, identities, paths, sheet names and
row numbers never enter the published artifact.

The doctor ``SZ`` slot describes the physician's retrospective conclusion for
the source recording.  It is therefore associated with the unique-signal
``recording_id`` and is never bound to a detector candidate or an automatically
created ``eeg_event_id``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from code.soz_pre.build_private_edf_soz_manifest import (
    DoctorEvent,
    _load_doctor_xlsx,
)
from code.soz_pre.utils import (
    clean_cell,
    normalize_patient_name,
    parse_electrodes,
    parse_sz_ids_from_stem,
)
from scripts import audit_private_long_recording_report_release_v1 as release_audit
from scripts import combine_private_long_recording_report_recovery_v1 as overlay
from scripts import materialize_private_long_recording_reports_v1 as batch
from src.clinical_eeg_report.schema import canonicalize_electrode
from src.clinical_eeg_long_recording.aggregation import (
    validate_trustworthy_long_term_clinical_eeg_bundle,
)
from src.clinical_eeg_long_recording.report_outcome import (
    COMPLETED_INSUFFICIENT_EVIDENCE,
    COMPLETED_LOCALIZABLE,
    COMPLETED_NONLOCALIZABLE,
    classify_recording_eeg_outcome,
)


SCHEMA_VERSION = "private_postfreeze_doctor_label_release_bundle_v1"
STATUS = "completed_postfreeze_doctor_label_release"
MAPPING_POLICY_ID = "normalized_leaf_subject_plus_sz_slot_to_unique_signal_v1"
PROJECTION_POLICY_ID = "doctor_onset_closed_vocabulary_projection_v1"
COMPARISON_POLICY_ID = "selective_recording_level_fact_consistency_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LABEL_ID_RE = re.compile(r"^DRLBL-[0-9a-f]{24}$")
_WORKBOOK_ID_RE = re.compile(r"^DRWB-[0-9a-f]{24}$")
_SZ_SLOT_RE = re.compile(r"^SZ[1-9][0-9]*$")

_LATERALITIES = frozenset(
    {"left", "right", "bilateral", "midline", "indeterminate"}
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
_UNCERTAINTIES = frozenset(
    {"clear", "uncertain_or_unclear", "indeterminate"}
)
_COMPARISON_STATUSES = frozenset(
    {"match", "partial_match", "mismatch", "not_available"}
)
_EVALUATION_DISPOSITIONS = frozenset(
    {
        "exact_or_compatible",
        "mismatch",
        "generated_abstention",
        "label_missing",
        "technical_unassessable",
        "source_conflict",
        "ambiguous_mapping",
    }
)
_DOCTOR_REFERENCE_STATUSES = frozenset(
    {"doctor_clear", "doctor_uncertain", "label_missing"}
)
_GENERATED_REPORT_STATUSES = frozenset(
    {
        "generated_localization",
        "generated_nonfocal_conclusion",
        "generated_abstention",
        "technical_unassessable",
    }
)
_ALIGNMENT_CODES = frozenset(
    {
        "exact_spatial_match",
        "compatible_spatial_overlap",
        "spatial_mismatch",
        "uncertainty_aligned",
        "doctor_uncertain_but_generated_localization",
        "generated_abstention_not_scored_as_conflict",
        "label_not_comparable",
        "technical_unassessable",
        "source_conflict_not_scored",
    }
)

_STANDARD_19_PLUS_M1_M2 = frozenset(
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
        "M1",
        "M2",
    }
)

_UNCLEAR_MARKERS = (
    "起始不清",
    "起始不明确",
    "起始未明",
    "起始难",
    "不清楚",
    "不明确",
    "无法判断",
    "难以判断",
    "难判断",
    "不能判断",
    "无法定位",
    "不能定位",
    "不易定位",
    "未见明确",
    "未能明确",
    "不确定",
    "不详",
)
_DIFFUSE_MARKERS = ("弥漫", "广泛", "全导", "全脑")

_REGION_ZH = {
    "frontal": "额区",
    "temporal": "颞区",
    "central": "中央区",
    "parietal": "顶区",
    "occipital": "枕区",
    "frontotemporal": "额颞区",
    "centrotemporal": "中央颞区",
    "temporoparietal": "颞顶区",
    "posterior": "后头部",
    "diffuse": "弥漫/广泛头皮分布",
    "midline": "中线区",
    "unknown": "区域未结构化",
}
_LATERALITY_ZH = {
    "left": "左侧",
    "right": "右侧",
    "bilateral": "双侧",
    "midline": "中线",
    "indeterminate": "侧别未结构化",
}

_FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "patient_name",
        "patient_key",
        "edf_relative_path",
        "edf_path",
        "workbook_path",
        "source_file",
        "sheet_name",
        "source_row",
        "source_row_number",
        "onset_text",
        "onset_description",
        "raw_significant",
        "raw_spread",
        "raw_text",
    }
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"JSON contains invalid constant {value!r}")


def _json_snapshot(path: Path) -> tuple[dict[str, Any], tuple[Path, str]]:
    if path.is_symlink():
        raise ValueError("JSON input must not be a symlink")
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("JSON input must be a regular file")
    raw = resolved.read_bytes()
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=_invalid_constant,
    )
    if not isinstance(value, dict):
        raise TypeError("JSON input must be an object")
    return value, (resolved, _sha256_bytes(raw))


def _safe_relative(value: object, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{context} is unsafe")
    return relative


def _regular_root(path: Path, context: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{context} must not be a symlink")
    root = path.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{context} must be a regular directory")
    return root


def _resolve_regular(root: Path, relative: PurePosixPath) -> Path:
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("artifact path traverses a symlink")
    resolved = cursor.resolve(strict=True)
    resolved.relative_to(root)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("artifact must be a regular file")
    return resolved


def _assert_snapshots_unchanged(snapshots: Iterable[tuple[Path, str]]) -> None:
    for path, expected in snapshots:
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError("a frozen report/source snapshot changed during publication")


def _atomic_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_doctor_electrodes(values: Iterable[str]) -> tuple[list[str], int]:
    result: list[str] = []
    excluded = 0
    for raw in values:
        if str(raw).upper() == "DIFFUSE":
            excluded += 1
            continue
        try:
            canonical = canonicalize_electrode(str(raw))
        except (TypeError, ValueError):
            excluded += 1
            continue
        if canonical not in _STANDARD_19_PLUS_M1_M2:
            excluded += 1
            continue
        if canonical not in result:
            result.append(canonical)
    return sorted(result), excluded


def _electrode_spatial_projection(text: str) -> tuple[set[str], set[str]]:
    token_candidates = list(parse_electrodes(text))
    token_candidates.extend(
        re.findall(
            r"(?<![A-Z0-9])(?:FP[12Z]|FZ|F[3478]|CZ|C[34]|PZ|P[34]|"
            r"O[12Z]|T[345678]|A[12]|M[12])(?![A-Z0-9])",
            text.upper(),
        )
    )
    electrodes, _ = _canonical_doctor_electrodes(token_candidates)
    lateralities: set[str] = set()
    regions: set[str] = set()
    left = {"FP1", "F7", "F3", "C3", "T7", "P7", "P3", "O1", "M1"}
    right = {"FP2", "F8", "F4", "C4", "T8", "P8", "P4", "O2", "M2"}
    midline = {"FZ", "CZ", "PZ"}
    for electrode in electrodes:
        if electrode in left:
            lateralities.add("left")
        elif electrode in right:
            lateralities.add("right")
        elif electrode in midline:
            lateralities.add("midline")
        if electrode.startswith("FP") or electrode.startswith("F"):
            regions.add("frontal")
        elif electrode in {"T7", "T8", "P7", "P8", "M1", "M2"}:
            regions.add("temporal")
        elif electrode.startswith("C"):
            regions.add("central")
        elif electrode.startswith("P"):
            regions.add("parietal")
        elif electrode.startswith("O"):
            regions.add("occipital")
    return lateralities, regions


def _display_onset_projection(
    *, laterality: str, regions: Sequence[str], onset_uncertainty: str
) -> str:
    if onset_uncertainty == "uncertain_or_unclear":
        return "医生标签（结构化）：侧别=未定；区域=未定；起始清晰度=不清/不确定。"
    if onset_uncertainty == "indeterminate":
        return "医生标签（结构化）：起始字段=未提供。"
    region_text = "/".join(_REGION_ZH[item] for item in regions) or "未定"
    laterality_text = _LATERALITY_ZH[laterality]
    return (
        "医生标签（结构化）："
        f"侧别={laterality_text}；区域={region_text}；起始清晰度=明确。"
    )


def project_doctor_onset_text(value: object) -> dict[str, Any]:
    """Reduce one raw onset cell to a closed, non-verbatim representation."""

    text = clean_cell(value)
    if not text:
        return {
            "status": "not_available",
            "laterality": "indeterminate",
            "regions": [],
            "onset_uncertainty": "indeterminate",
            "projection_method": PROJECTION_POLICY_ID,
            "display_zh": _display_onset_projection(
                laterality="indeterminate",
                regions=[],
                onset_uncertainty="indeterminate",
            ),
            "raw_text_included": False,
        }

    uncertain = any(marker in text for marker in _UNCLEAR_MARKERS)
    has_left = "左" in text
    has_right = "右" in text
    has_bilateral = any(marker in text for marker in ("双侧", "两侧", "双半球"))
    has_midline = any(marker in text for marker in ("中线", "旁中线"))
    electrode_lateralities, electrode_regions = _electrode_spatial_projection(text)
    has_left = has_left or "left" in electrode_lateralities
    has_right = has_right or "right" in electrode_lateralities
    has_midline = has_midline or "midline" in electrode_lateralities
    if has_bilateral or (has_left and has_right):
        laterality = "bilateral"
    elif has_left:
        laterality = "left"
    elif has_right:
        laterality = "right"
    elif has_midline:
        laterality = "midline"
    else:
        laterality = "indeterminate"

    regions: list[str] = []

    def add(region: str) -> None:
        if region not in regions:
            regions.append(region)

    compound_terms = (
        ("frontotemporal", ("额颞", "颞额")),
        ("centrotemporal", ("中央颞", "颞中央")),
        ("temporoparietal", ("颞顶", "顶颞")),
        ("posterior", ("后头", "后部")),
    )
    for region, markers in compound_terms:
        if any(marker in text for marker in markers):
            add(region)
    simple_terms = (
        ("frontal", "额"),
        ("temporal", "颞"),
        ("central", "中央"),
        ("parietal", "顶"),
        ("occipital", "枕"),
    )
    for region, marker in simple_terms:
        covered_by_compound = any(
            compound_marker in text and marker in compound_marker
            for _, compound_markers in compound_terms
            for compound_marker in compound_markers
        )
        if marker in text and not covered_by_compound:
            add(region)
    for region in sorted(electrode_regions):
        add(region)
    if any(marker in text for marker in _DIFFUSE_MARKERS):
        add("diffuse")
    if has_midline and not regions:
        add("midline")
    regions = [region for region in regions if region in _REGIONS and region != "unknown"]
    onset_uncertainty = "uncertain_or_unclear" if uncertain else "clear"
    if uncertain:
        # An explicitly unclear onset must not quietly retain a weaker spatial
        # phrase as if it were a physician-confirmed localization reference.
        laterality = "indeterminate"
        regions = []
    projection = {
        "status": "available",
        "laterality": laterality,
        "regions": regions,
        "onset_uncertainty": onset_uncertainty,
        "projection_method": PROJECTION_POLICY_ID,
        "display_zh": _display_onset_projection(
            laterality=laterality,
            regions=regions,
            onset_uncertainty=onset_uncertainty,
        ),
        "raw_text_included": False,
    }
    validate_doctor_onset_projection(projection)
    return projection


def validate_doctor_onset_projection(value: object) -> dict[str, Any]:
    keys = {
        "status",
        "laterality",
        "regions",
        "onset_uncertainty",
        "projection_method",
        "display_zh",
        "raw_text_included",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("doctor onset projection shape drifted")
    if value["status"] not in {"available", "not_available"}:
        raise ValueError("doctor onset projection status is unsupported")
    if value["laterality"] not in _LATERALITIES:
        raise ValueError("doctor onset projection laterality is unsupported")
    if not isinstance(value["regions"], list) or any(
        region not in _REGIONS for region in value["regions"]
    ) or len(value["regions"]) != len(set(value["regions"])):
        raise ValueError("doctor onset projection regions are unsupported")
    if value["onset_uncertainty"] not in _UNCERTAINTIES:
        raise ValueError("doctor onset projection uncertainty is unsupported")
    if value["projection_method"] != PROJECTION_POLICY_ID:
        raise ValueError("doctor onset projection policy drifted")
    if not isinstance(value["display_zh"], str) or not value["display_zh"]:
        raise ValueError("doctor onset display projection is missing")
    if value["raw_text_included"] is not False:
        raise ValueError("raw doctor onset text must not be included")
    if value["status"] == "not_available" and (
        value["laterality"] != "indeterminate"
        or value["regions"]
        or value["onset_uncertainty"] != "indeterminate"
    ):
        raise ValueError("missing doctor onset projection carries a typed conclusion")
    if value["status"] == "available" and value["onset_uncertainty"] == "indeterminate":
        raise ValueError("available doctor onset projection has indeterminate availability")
    if value["onset_uncertainty"] == "uncertain_or_unclear" and (
        value["laterality"] != "indeterminate" or value["regions"]
    ):
        raise ValueError("unclear doctor onset must not carry a spatial conclusion")
    if value["display_zh"] != _display_onset_projection(
        laterality=str(value["laterality"]),
        regions=list(value["regions"]),
        onset_uncertainty=str(value["onset_uncertainty"]),
    ):
        raise ValueError("doctor onset display is not the deterministic code projection")
    return deepcopy(dict(value))


def _qualified_onset_values(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for event in bundle["events"]:
        event_id = event["eeg_event_id"]
        for fact in event["event_report_payload"]["facts"]:
            if fact.get("fact_type") != "ictal_onset_pattern" or fact.get(
                "eeg_event_id"
            ) != event_id:
                continue
            verification = fact.get("verification")
            value = fact.get("value")
            if not isinstance(verification, Mapping) or not isinstance(value, Mapping):
                continue
            physician_verified = verification.get("status") == "physician_verified"
            qualification = value.get("qualification")
            algorithm_qualified = isinstance(qualification, Mapping) and all(
                (
                    qualification.get("electrographic_seizure_gate_passed") is True,
                    qualification.get("spatial_field_gate_passed") is True,
                    qualification.get("artifact_gate_passed") is True,
                    qualification.get("source_signal_only") is True,
                    qualification.get("external_context_used") is False,
                    qualification.get("research_ranking_used") is False,
                    qualification.get("promotion_status")
                    == "passed_external_validation_gate",
                )
            )
            if physician_verified or algorithm_qualified:
                result.append(value)
    return result


def _report_fact_projection(
    manifest: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, Any]:
    outcome = classify_recording_eeg_outcome(bundle)
    if manifest.get("diagnostic_outcome") != outcome:
        raise ValueError("report diagnostic outcome is not reproducible from frozen facts")
    status = outcome["report_status"]
    if status == COMPLETED_LOCALIZABLE:
        signatures: list[tuple[str, set[str]]] = []
        for value in _qualified_onset_values(bundle):
            laterality = value.get("laterality")
            regions = value.get("regions")
            if laterality not in {"left", "right", "midline"}:
                continue
            if not isinstance(regions, list):
                continue
            normalized = {str(item) for item in regions if str(item) in _REGIONS}
            if normalized:
                signatures.append((str(laterality), normalized))
        if not signatures:
            raise ValueError("localizable report lacks a qualified spatial signature")
        lateralities = {item[0] for item in signatures}
        common_regions = set(signatures[0][1])
        for _, regions in signatures[1:]:
            common_regions.intersection_update(regions)
        if len(lateralities) != 1 or not common_regions:
            raise ValueError("localizable report has no reproducible spatial consensus")
        disposition = "localized"
        laterality = next(iter(lateralities))
        regions = sorted(common_regions)
    elif status == COMPLETED_NONLOCALIZABLE:
        affirmative_nonfocal = (
            "qualified_events_have_bilateral_synchronous_or_diffuse_scalp_distribution"
            in outcome["evidence_reasons"]
            and bool(outcome["nonfocal_supporting_event_ids"])
            and len(outcome["nonfocal_supporting_event_ids"])
            == outcome["qualified_event_count"]
        )
        disposition = "nonfocal" if affirmative_nonfocal else "nonlocalizable"
        laterality = None
        regions = ["diffuse"] if affirmative_nonfocal else []
    elif status == COMPLETED_INSUFFICIENT_EVIDENCE:
        disposition = "insufficient_evidence"
        laterality = None
        regions = []
    else:
        raise ValueError("EEG report diagnostic status is unsupported")
    return {
        "status": "available",
        "diagnostic_status": status,
        "localization_disposition": disposition,
        "laterality": laterality,
        "regions": regions,
        "projection_source": "frozen_eeg_fact_ledger_only",
        "doctor_label_used": False,
    }


def _technical_report_projection(diagnostic_status: str) -> dict[str, Any]:
    return {
        "status": "not_available",
        "diagnostic_status": diagnostic_status,
        "localization_disposition": "technical_unassessable",
        "laterality": None,
        "regions": [],
        "projection_source": "technical_report_has_no_eeg_fact_projection",
        "doctor_label_used": False,
    }


def _field_comparison(
    field: str, report_values: Sequence[str], doctor_values: Sequence[str]
) -> dict[str, Any]:
    report = sorted(set(report_values))
    doctor = sorted(set(doctor_values))
    if not report or not doctor:
        status = "not_available"
        overlap: list[str] = []
    else:
        overlap = sorted(set(report).intersection(doctor))
        if report == doctor:
            status = "match"
        elif overlap:
            status = "partial_match"
        else:
            status = "mismatch"
    return {
        "field": field,
        "status": status,
        "report_values": report,
        "doctor_values": doctor,
        "overlap_values": overlap,
    }


def compare_report_with_doctor_onset(
    report: Mapping[str, Any], doctor: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare only a frozen report projection with one recording-level label."""

    doctor_projection = validate_doctor_onset_projection(doctor)
    disposition = report.get("localization_disposition")
    doctor_uncertainty = doctor_projection["onset_uncertainty"]
    if doctor_projection["status"] != "available":
        uncertainty_status = "not_available"
        reason = "doctor_onset_not_available"
    elif disposition == "technical_unassessable":
        uncertainty_status = "not_available"
        reason = "technical_report_has_no_eeg_conclusion"
    elif doctor_uncertainty == "uncertain_or_unclear":
        if disposition in {"localized", "nonfocal"}:
            uncertainty_status = "mismatch"
            reason = "doctor_unclear_but_report_localized"
        else:
            uncertainty_status = "match"
            reason = "doctor_unclear_and_report_abstained"
    elif doctor_uncertainty == "clear" and disposition in {"localized", "nonfocal"}:
        uncertainty_status = "match"
        reason = "both_sources_express_onset_distribution_conclusion"
    elif doctor_uncertainty == "clear":
        uncertainty_status = "not_available"
        reason = "report_abstained_from_spatial_conclusion"
    else:
        uncertainty_status = "not_available"
        reason = "doctor_uncertainty_indeterminate"

    laterality = _field_comparison(
        "laterality",
        [str(report["laterality"])]
        if disposition in {"localized", "nonfocal"} and report.get("laterality")
        else [],
        [doctor_projection["laterality"]]
        if doctor_projection["laterality"] != "indeterminate"
        else [],
    )
    regions = _field_comparison(
        "regions",
        list(report.get("regions", []))
        if disposition in {"localized", "nonfocal"}
        else [],
        list(doctor_projection["regions"]),
    )
    if doctor_uncertainty == "uncertain_or_unclear":
        overall = uncertainty_status
    elif uncertainty_status != "match":
        overall = "not_available" if uncertainty_status == "not_available" else "mismatch"
    else:
        spatial = [laterality["status"], regions["status"]]
        comparable = [item for item in spatial if item != "not_available"]
        if "mismatch" in comparable:
            overall = "mismatch"
        elif "partial_match" in comparable:
            overall = "partial_match"
        elif comparable and all(item == "match" for item in comparable):
            overall = "match"
        else:
            overall = "not_available"
    if overall not in _COMPARISON_STATUSES:
        raise AssertionError("unreachable consistency status")
    if doctor_projection["status"] != "available":
        doctor_reference_status = "label_missing"
    elif doctor_uncertainty == "uncertain_or_unclear":
        doctor_reference_status = "doctor_uncertain"
    else:
        doctor_reference_status = "doctor_clear"
    if disposition == "technical_unassessable":
        generated_report_status = "technical_unassessable"
        evaluation_disposition = "technical_unassessable"
        alignment_code = "technical_unassessable"
    elif disposition in {"localized", "nonfocal"}:
        generated_report_status = (
            "generated_localization"
            if disposition == "localized"
            else "generated_nonfocal_conclusion"
        )
        if doctor_reference_status == "label_missing":
            evaluation_disposition = "label_missing"
            alignment_code = "label_not_comparable"
        elif doctor_reference_status == "doctor_uncertain":
            evaluation_disposition = "mismatch"
            alignment_code = "doctor_uncertain_but_generated_localization"
        elif overall == "mismatch":
            evaluation_disposition = "mismatch"
            alignment_code = "spatial_mismatch"
        elif overall == "partial_match":
            evaluation_disposition = "exact_or_compatible"
            alignment_code = "compatible_spatial_overlap"
        elif overall == "match":
            evaluation_disposition = "exact_or_compatible"
            alignment_code = "exact_spatial_match"
        else:
            evaluation_disposition = "label_missing"
            alignment_code = "label_not_comparable"
    else:
        generated_report_status = "generated_abstention"
        if doctor_reference_status == "doctor_uncertain":
            evaluation_disposition = "exact_or_compatible"
            alignment_code = "uncertainty_aligned"
        elif doctor_reference_status == "label_missing":
            evaluation_disposition = "label_missing"
            alignment_code = "label_not_comparable"
        else:
            evaluation_disposition = "generated_abstention"
            alignment_code = "generated_abstention_not_scored_as_conflict"
    return {
        "policy_id": COMPARISON_POLICY_ID,
        "overall_status": overall,
        "evaluation_disposition": evaluation_disposition,
        "doctor_reference_status": doctor_reference_status,
        "generated_report_status": generated_report_status,
        "alignment_code": alignment_code,
        "reason_code": reason,
        "onset_uncertainty": {
            "status": uncertainty_status,
            "report_disposition": disposition,
            "doctor_value": doctor_uncertainty,
        },
        "spatial_fields": [laterality, regions],
        "missing_prediction_is_not_scored_as_mismatch": True,
        "doctor_unclear_and_report_abstention_is_compatible": True,
    }


def _source_conflict_comparison(
    report: Mapping[str, Any], doctor: Mapping[str, Any]
) -> dict[str, Any]:
    comparison = compare_report_with_doctor_onset(report, doctor)
    comparison["overall_status"] = "not_available"
    comparison["reason_code"] = "conflicting_doctor_sources_not_scored"
    comparison["evaluation_disposition"] = "source_conflict"
    comparison["alignment_code"] = "source_conflict_not_scored"
    comparison["onset_uncertainty"]["status"] = "not_available"
    for field in comparison["spatial_fields"]:
        field["status"] = "not_available"
        field["overlap_values"] = []
    comparison["source_conflict_present"] = True
    return comparison


def _validate_release_audit(
    value: Mapping[str, Any],
    *,
    inventory: Mapping[str, Any],
    inventory_sha256: str,
    coverage_sha256: str,
    coverage_kind: str,
) -> None:
    if value.get("schema_version") != release_audit.SCHEMA_VERSION:
        raise ValueError("release audit schema is unsupported")
    body = {key: item for key, item in value.items() if key != "audit_id"}
    expected_id = release_audit.AUDIT_ID_PREFIX + _canonical_sha256(body)[:24]
    if value.get("audit_id") != expected_id:
        raise ValueError("release audit ID does not bind its content")
    if (
        value.get("audit_mode") != release_audit.COHORT_AUDIT_MODE
        or value.get("status") != release_audit.PASS_STATUS
        or value.get("release_ready") is not True
        or value.get("coverage_kind") != coverage_kind
        or value.get("inventory_id") != inventory["inventory_id"]
        or value.get("recording_unit_policy") != "unique_signal_sha256_v1"
    ):
        raise ValueError("a passing final cohort release audit is required")
    receipts = value.get("source_receipts")
    if not isinstance(receipts, Mapping) or (
        receipts.get("inventory_manifest_sha256") != inventory_sha256
        or receipts.get("coverage_manifest_sha256") != coverage_sha256
    ):
        raise ValueError("release audit source hashes do not bind current inputs")
    counts = value.get("cohort_counts")
    if not isinstance(counts, Mapping) or any(
        (
            counts.get("expected_record_count") != inventory["record_count"],
            counts.get("expected_subject_count") != inventory["subject_count"],
            counts.get("pending_or_not_run_count") != 0,
            counts.get("completed_eeg_reports_failed") != 0,
            counts.get("technical_reports_failed") != 0,
        )
    ):
        raise ValueError("release audit does not cover the complete cohort")
    checks = value.get("checks")
    required_checks = (
        "inventory_schema_and_binding_validated",
        "coverage_schema_and_binding_validated",
        "all_completed_eeg_reports_valid",
        "all_technical_reports_valid",
        "dataset_artifact_coverage_complete",
        "html_docx_json_and_waveforms_revalidated",
        "source_artifact_snapshots_unchanged",
    )
    if not isinstance(checks, Mapping) or any(checks.get(key) is not True for key in required_checks):
        raise ValueError("release audit required checks are incomplete")
    scope = value.get("scope_receipt")
    required_scope = {
        "edf_signal_files_read": False,
        "edf_annotations_read": False,
        "excel_or_workbook_read": False,
        "onset_label_or_ground_truth_read": False,
        "report_artifacts_read_only": True,
        "report_artifacts_modified": False,
    }
    if not isinstance(scope, Mapping) or any(
        scope.get(key) != expected for key, expected in required_scope.items()
    ):
        raise ValueError("release audit violates the pre-label access boundary")


def _selected_manifest_relative(
    row: Mapping[str, Any], *, coverage_kind: str
) -> PurePosixPath:
    recording_id = str(row["recording_id"])
    if coverage_kind == "combined":
        return _safe_relative(row["report_manifest_relative_path"], "report manifest")
    if row["diagnostic_status"] in batch.COMPLETED_DIAGNOSTIC_STATUSES:
        return PurePosixPath("records") / recording_id / "report" / "manifest.json"
    technical = _safe_relative(
        row["technical_artifact_relative_dir"], "technical report directory"
    )
    return PurePosixPath("records") / recording_id / technical / "manifest.json"


def _artifact_root_alias(row: Mapping[str, Any], coverage_kind: str) -> str:
    return "full" if coverage_kind == "full" else str(row["artifact_source"])


def _parse_json_bytes(raw: bytes, context: str) -> dict[str, Any]:
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=_invalid_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _freeze_selected_reports(
    *,
    inventory: Mapping[str, Any],
    coverage: Mapping[str, Any],
    coverage_kind: str,
    roots: Mapping[str, Path],
) -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
    inventory_by_id = {
        str(item["recording_id"]): item for item in inventory["records"]
    }
    receipts: list[dict[str, Any]] = []
    snapshots: list[tuple[Path, str]] = []
    for row in coverage["records"]:
        recording_id = str(row["recording_id"])
        root_alias = _artifact_root_alias(row, coverage_kind)
        if root_alias not in roots:
            raise ValueError(f"missing selected report root alias {root_alias!r}")
        root = roots[root_alias]
        manifest_relative = _selected_manifest_relative(
            row, coverage_kind=coverage_kind
        )
        manifest_path = _resolve_regular(root, manifest_relative)
        manifest_raw = manifest_path.read_bytes()
        manifest_sha = _sha256_bytes(manifest_raw)
        manifest = _parse_json_bytes(manifest_raw, "selected report manifest")
        snapshots.append((manifest_path, manifest_sha))
        if coverage_kind == "combined" and row["report_manifest_sha256"] != manifest_sha:
            raise ValueError("combined coverage selected report hash drifted")
        inventory_record = inventory_by_id[recording_id]
        diagnostic = str(row["diagnostic_status"])
        if diagnostic in batch.COMPLETED_DIAGNOSTIC_STATUSES:
            overlay._validate_eeg_report(  # noqa: SLF001
                manifest, inventory_record=inventory_record, row=row
            )
            report_kind = "eeg_report"
            required = {"bundle.json", "report.html", "report.docx"}
        elif diagnostic == overlay.TECHNICAL_STATUS:
            overlay._validate_technical_report(  # noqa: SLF001
                manifest, inventory_record=inventory_record, row=row
            )
            report_kind = "technical_unassessable_report"
            required = {"report.json", "report.html"}
        else:
            raise ValueError("coverage contains a non-completed report")
        artifacts = manifest["artifacts"]
        if not isinstance(artifacts, Mapping) or not required.issubset(artifacts):
            raise ValueError("selected report is missing a required body artifact")
        artifact_hashes: dict[str, str] = {}
        parsed_bundle: dict[str, Any] | None = None
        for relative_text, expected_sha in sorted(artifacts.items()):
            if not isinstance(expected_sha, str) or _SHA256_RE.fullmatch(expected_sha) is None:
                raise ValueError("selected report carries an invalid artifact hash")
            relative = _safe_relative(relative_text, "report artifact")
            artifact_path = _resolve_regular(manifest_path.parent, relative)
            raw = artifact_path.read_bytes()
            actual_sha = _sha256_bytes(raw)
            if actual_sha != expected_sha:
                raise ValueError("selected report artifact hash mismatch")
            snapshots.append((artifact_path, actual_sha))
            artifact_hashes[relative.as_posix()] = actual_sha
            if relative.as_posix() == "bundle.json":
                parsed_bundle = _parse_json_bytes(raw, "frozen EEG bundle")
        if report_kind == "eeg_report":
            if parsed_bundle is None:
                raise ValueError("EEG report has no frozen bundle.json")
            normalized_bundle = validate_trustworthy_long_term_clinical_eeg_bundle(
                parsed_bundle
            )
            if normalized_bundle["recording_id"] != recording_id:
                raise ValueError("EEG bundle recording identity drifted")
            report_projection = _report_fact_projection(manifest, normalized_bundle)
        else:
            report_projection = _technical_report_projection(diagnostic)
        html_relative = manifest_relative.parent / "report.html"
        receipt = {
            "recording_id": recording_id,
            "patient_pseudonym": inventory_record["patient_pseudonym"],
            "artifact_source": root_alias,
            "report_kind": report_kind,
            "diagnostic_status": diagnostic,
            "event_count": int(row["event_count"]),
            "report_manifest_relative_path": manifest_relative.as_posix(),
            "report_manifest_sha256": manifest_sha,
            "report_html_relative_path": html_relative.as_posix(),
            "report_html_sha256": artifact_hashes["report.html"],
            "artifact_hash_set_sha256": _canonical_sha256(artifact_hashes),
            "report_projection": report_projection,
        }
        receipts.append(receipt)
    if len(receipts) != inventory["record_count"]:
        raise ValueError("frozen report receipts do not span the inventory")
    return receipts, snapshots


def _freeze_report_release(
    *,
    inventory_path: Path,
    coverage_path: Path,
    release_audit_path: Path,
    full_root: Path | None,
    primary_root: Path | None,
    recovery_root: Path | None,
    remediation_root: Path | None,
) -> dict[str, Any]:
    inventory_value, inventory_snapshot = _json_snapshot(inventory_path)
    inventory = batch.validate_inventory(inventory_value)
    if (
        inventory["recording_unit_policy"] != "unique_signal_sha256_v1"
        or inventory["source_rejections"]
        or any(item["inventory_validation_status"] != batch.READY for item in inventory["records"])
    ):
        raise ValueError("doctor-label release requires a ready unique-signal inventory")
    signal_hashes = [str(item["source_signal_sha256"]) for item in inventory["records"]]
    if len(signal_hashes) != len(set(signal_hashes)):
        raise ValueError("inventory repeats a complete EEG signal SHA-256")

    coverage_value, coverage_snapshot = _json_snapshot(coverage_path)
    coverage_kind, coverage = release_audit._validate_coverage_input(  # noqa: SLF001
        coverage_value,
        inventory=inventory,
        inventory_sha256=inventory_snapshot[1],
    )
    if coverage.get("dataset_artifact_coverage_complete") is not True or coverage.get(
        "pending_or_not_run_count"
    ) != 0:
        raise ValueError("doctor-label release requires complete report artifact coverage")

    audit_value, audit_snapshot = _json_snapshot(release_audit_path)
    _validate_release_audit(
        audit_value,
        inventory=inventory,
        inventory_sha256=inventory_snapshot[1],
        coverage_sha256=coverage_snapshot[1],
        coverage_kind=coverage_kind,
    )
    if coverage_kind == "full":
        if any(root is not None for root in (primary_root, recovery_root, remediation_root)):
            raise ValueError("primary/recovery/remediation roots require combined coverage")
        root_value = full_root if full_root is not None else coverage_snapshot[0].parent
        roots = {"full": _regular_root(root_value, "full report root")}
    else:
        if full_root is not None:
            raise ValueError("full root applies only to full coverage")
        supplied = {
            "primary": primary_root,
            "recovery": recovery_root,
            "remediation": remediation_root,
        }
        selected = {
            str(row["artifact_source"]) for row in coverage["records"]
        }
        if any(supplied.get(alias) is None for alias in selected):
            raise ValueError("combined coverage requires every selected artifact root")
        roots = {
            alias: _regular_root(supplied[alias], f"{alias} report root")  # type: ignore[arg-type]
            for alias in selected
        }

    report_receipts, report_snapshots = _freeze_selected_reports(
        inventory=inventory,
        coverage=coverage,
        coverage_kind=coverage_kind,
        roots=roots,
    )
    snapshots = [inventory_snapshot, coverage_snapshot, audit_snapshot, *report_snapshots]
    _assert_snapshots_unchanged(snapshots)
    return {
        "inventory": inventory,
        "coverage": coverage,
        "coverage_kind": coverage_kind,
        "roots": roots,
        "report_receipts": report_receipts,
        "snapshots": snapshots,
        "source_receipts": {
            "inventory_manifest_sha256": inventory_snapshot[1],
            "coverage_manifest_sha256": coverage_snapshot[1],
            "release_audit_sha256": audit_snapshot[1],
            "release_audit_id": audit_value["audit_id"],
            "report_manifest_set_sha256": _canonical_sha256(
                [
                    {
                        "recording_id": item["recording_id"],
                        "report_manifest_sha256": item["report_manifest_sha256"],
                    }
                    for item in report_receipts
                ]
            ),
            "report_artifact_set_sha256": _canonical_sha256(
                [
                    {
                        "recording_id": item["recording_id"],
                        "artifact_hash_set_sha256": item["artifact_hash_set_sha256"],
                    }
                    for item in report_receipts
                ]
            ),
        },
    }


def _read_doctor_workbooks(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[Path, str]]]:
    if not paths:
        raise ValueError("at least one private doctor workbook is required")
    events: list[dict[str, Any]] = []
    workbook_receipts: list[dict[str, Any]] = []
    snapshots: list[tuple[Path, str]] = []
    seen_hashes: set[str] = set()
    for raw_path in paths:
        if raw_path.is_symlink():
            raise ValueError("doctor workbook must not be a symlink")
        path = raw_path.resolve(strict=True)
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {
            ".xls",
            ".xlsx",
            ".xlsm",
        }:
            raise ValueError("doctor workbook must be a regular supported Excel file")
        workbook_sha = _sha256_file(path)
        if workbook_sha in seen_hashes:
            continue
        seen_hashes.add(workbook_sha)
        snapshots.append((path, workbook_sha))
        parsed = _load_doctor_xlsx(path)
        workbook_id = "DRWB-" + workbook_sha[:24]
        source_event_count = 0
        for item in parsed:
            if not isinstance(item, DoctorEvent):
                raise TypeError("doctor workbook parser returned an unsupported row")
            patient_key = normalize_patient_name(item.patient_name)
            slot = str(item.sz_id).upper()
            if not patient_key or _SZ_SLOT_RE.fullmatch(slot) is None:
                continue
            sheet_token = str(item.source_file).rsplit(":", 1)[-1]
            source_locator_fingerprint = _canonical_sha256(
                {
                    "workbook_sha256": workbook_sha,
                    "sheet_token_sha256": hashlib.sha256(
                        sheet_token.encode("utf-8")
                    ).hexdigest(),
                    "source_row": int(item.source_row),
                    "source_event_slot": slot,
                }
            )
            onset = project_doctor_onset_text(item.onset_text)
            significant, excluded_significant = _canonical_doctor_electrodes(
                item.significant or []
            )
            spread, excluded_spread = _canonical_doctor_electrodes(item.spread or [])
            raw_spread = clean_cell(item.raw_spread)
            diffuse_spread = any(marker in raw_spread for marker in _DIFFUSE_MARKERS)
            structured = {
                "onset": onset,
                "physician_channel_reference": {
                    "status": "available"
                    if significant or spread or diffuse_spread
                    else "not_available",
                    "reference_scope": "standard_19_plus_m1_m2_electrodes",
                    "significant_electrodes": significant,
                    "spread_electrodes": spread,
                    "diffuse_spread_present": diffuse_spread,
                    "excluded_out_of_scope_significant_token_count": excluded_significant,
                    "excluded_out_of_scope_spread_token_count": excluded_spread,
                    "significant_semantics": "hard_positive_only_unknown_complement",
                    "spread_semantics": "soft_positive_not_onset",
                    "evaluation_only": True,
                    "eligible_for_report_body": False,
                    "eligible_for_llm": False,
                },
            }
            projection_sha = _canonical_sha256(structured)
            label_body = {
                "workbook_id": workbook_id,
                "source_event_slot": slot,
                "source_locator_fingerprint": source_locator_fingerprint,
                "structured_projection_sha256": projection_sha,
            }
            events.append(
                {
                    "patient_key": patient_key,
                    "source_event_slot": slot,
                    "label_id": "DRLBL-" + _canonical_sha256(label_body)[:24],
                    "source_receipt": {
                        "workbook_id": workbook_id,
                        "workbook_sha256": workbook_sha,
                        "source_locator_fingerprint": source_locator_fingerprint,
                        "structured_projection_sha256": projection_sha,
                        "source_path_included": False,
                        "sheet_name_or_row_number_included": False,
                    },
                    **structured,
                }
            )
            source_event_count += 1
        workbook_receipts.append(
            {
                "workbook_id": workbook_id,
                "workbook_sha256": workbook_sha,
                "structured_source_event_count": source_event_count,
                "source_path_included": False,
                "raw_cell_values_included": False,
            }
        )
    return events, workbook_receipts, snapshots


def _public_label(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label_id": event["label_id"],
        "source_event_slot": event["source_event_slot"],
        "source_receipt": deepcopy(event["source_receipt"]),
        "onset": deepcopy(event["onset"]),
        "physician_channel_reference": deepcopy(
            event["physician_channel_reference"]
        ),
        "equivalent_source_receipts": [
            {
                "label_id": event["label_id"],
                "source_receipt": deepcopy(event["source_receipt"]),
            }
        ],
    }


def _associate_labels(
    *,
    inventory: Mapping[str, Any],
    report_receipts: Sequence[Mapping[str, Any]],
    doctor_events: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for event in doctor_events:
        by_key[(str(event["patient_key"]), str(event["source_event_slot"]))].append(
            event
        )
    report_by_id = {str(item["recording_id"]): item for item in report_receipts}
    recording_count_by_key: Counter[tuple[str, str]] = Counter()
    for inventory_record in inventory["records"]:
        relative = PurePosixPath(str(inventory_record["edf_relative_path"]))
        patient_key = normalize_patient_name(relative.parent.name)
        for slot in parse_sz_ids_from_stem(relative.stem):
            recording_count_by_key[(patient_key, slot)] += 1
    used_label_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    comparison_counts: Counter[str] = Counter()
    conflict_count = 0
    for inventory_record in inventory["records"]:
        recording_id = str(inventory_record["recording_id"])
        relative = PurePosixPath(str(inventory_record["edf_relative_path"]))
        patient_key = normalize_patient_name(relative.parent.name)
        slots = parse_sz_ids_from_stem(relative.stem)
        candidates: list[Mapping[str, Any]] = []
        for slot in slots:
            if recording_count_by_key[(patient_key, slot)] == 1:
                candidates.extend(by_key.get((patient_key, slot), []))
        source_label_key_present = any(by_key.get((patient_key, slot)) for slot in slots)
        ambiguous_mapping = any(
            recording_count_by_key[(patient_key, slot)] > 1
            and bool(by_key.get((patient_key, slot)))
            for slot in slots
        )
        if ambiguous_mapping:
            candidates = []
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            grouped[str(candidate["source_event_slot"])].append(candidate)
        labels: list[dict[str, Any]] = []
        record_conflict = False
        report_receipt = report_by_id[recording_id]
        for slot in slots:
            variants = grouped.get(slot, [])
            if not variants:
                continue
            projections: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for variant in variants:
                projection_key = _canonical_sha256(
                    {
                        "onset": variant["onset"],
                        "physician_channel_reference": variant[
                            "physician_channel_reference"
                        ],
                    }
                )
                projections[projection_key].append(variant)
            if len(projections) != 1:
                record_conflict = True
                conflict_count += 1
                for equivalent_variants in sorted(
                    projections.values(), key=lambda items: str(items[0]["label_id"])
                ):
                    selected = equivalent_variants[0]
                    public = _public_label(selected)
                    public["duplicate_equivalent_source_count"] = len(
                        equivalent_variants
                    )
                    public["equivalent_source_receipts"] = [
                        {
                            "label_id": item["label_id"],
                            "source_receipt": deepcopy(item["source_receipt"]),
                        }
                        for item in equivalent_variants
                    ]
                    public["source_conflict_variant"] = True
                    public["evaluation_eligible"] = False
                    public["fact_consistency"] = _source_conflict_comparison(
                        report_receipt["report_projection"], public["onset"]
                    )
                    comparison_counts["not_available"] += 1
                    labels.append(public)
                    used_label_ids.update(
                        str(item["label_id"]) for item in equivalent_variants
                    )
                continue
            equivalent_variants = next(iter(projections.values()))
            selected = equivalent_variants[0]
            public = _public_label(selected)
            public["duplicate_equivalent_source_count"] = len(equivalent_variants)
            public["equivalent_source_receipts"] = [
                {
                    "label_id": item["label_id"],
                    "source_receipt": deepcopy(item["source_receipt"]),
                }
                for item in equivalent_variants
            ]
            public["source_conflict_variant"] = False
            public["evaluation_eligible"] = True
            public["fact_consistency"] = compare_report_with_doctor_onset(
                report_receipt["report_projection"], public["onset"]
            )
            comparison_counts[public["fact_consistency"]["overall_status"]] += 1
            labels.append(public)
            used_label_ids.update(
                str(item["label_id"]) for item in equivalent_variants
            )
        if ambiguous_mapping:
            label_status = "ambiguous_mapping"
        elif record_conflict:
            label_status = "source_conflict"
        elif labels:
            label_status = "available"
        else:
            label_status = "not_available"
        label_dispositions = {
            str(item["fact_consistency"]["evaluation_disposition"])
            for item in labels
        }
        if (
            report_receipt["report_projection"]["localization_disposition"]
            == "technical_unassessable"
        ):
            record_consistency_disposition = "technical_unassessable"
        elif ambiguous_mapping:
            record_consistency_disposition = "ambiguous_mapping"
        elif record_conflict:
            record_consistency_disposition = "source_conflict"
        elif not labels:
            record_consistency_disposition = "label_missing"
        elif "mismatch" in label_dispositions:
            record_consistency_disposition = "mismatch"
        elif "generated_abstention" in label_dispositions:
            record_consistency_disposition = "generated_abstention"
        elif "label_missing" in label_dispositions:
            record_consistency_disposition = "label_missing"
        else:
            record_consistency_disposition = "exact_or_compatible"
        mapping_key_fingerprint = _canonical_sha256(
            {
                "inventory_id": inventory["inventory_id"],
                "recording_id": recording_id,
                "source_signal_sha256": inventory_record["source_signal_sha256"],
                "source_event_slots": slots,
                "normalized_patient_key_sha256": hashlib.sha256(
                    patient_key.encode("utf-8")
                ).hexdigest(),
            }
        )
        records.append(
            {
                "recording_id": recording_id,
                "patient_pseudonym": inventory_record["patient_pseudonym"],
                "report_receipt": {
                    key: deepcopy(report_receipt[key])
                    for key in (
                        "artifact_source",
                        "report_kind",
                        "diagnostic_status",
                        "event_count",
                        "report_manifest_relative_path",
                        "report_manifest_sha256",
                        "report_html_relative_path",
                        "report_html_sha256",
                        "artifact_hash_set_sha256",
                        "report_projection",
                    )
                },
                "doctor_label_status": label_status,
                "doctor_label_count": len(labels),
                "doctor_labels": labels,
                "record_consistency_disposition": record_consistency_disposition,
                "mapping_receipt": {
                    "policy_id": MAPPING_POLICY_ID,
                    "mapping_key_fingerprint": mapping_key_fingerprint,
                    "source_event_slot_count": len(slots),
                    "maximum_unique_signal_count_for_mapping_key": max(
                        (
                            recording_count_by_key[(patient_key, slot)]
                            for slot in slots
                        ),
                        default=0,
                    ),
                    "doctor_source_label_key_present": source_label_key_present,
                    "associated_at_recording_level": True,
                    "bound_to_detector_candidate_event": False,
                    "raw_patient_identity_included": False,
                    "private_edf_path_included": False,
                    "source_signal_deduplicated_before_association": True,
                },
            }
        )
    unmatched = len({str(item["label_id"]) for item in doctor_events} - used_label_ids)
    published_associations = sum(item["doctor_label_count"] for item in records)
    distinct_published_label_ids = {
        str(label["label_id"])
        for record in records
        for label in record["doctor_labels"]
    }
    summary = {
        "record_count": len(records),
        "record_with_doctor_label_count": sum(
            item["doctor_label_status"] == "available" for item in records
        ),
        "record_with_any_published_doctor_label_count": sum(
            bool(item["doctor_labels"]) for item in records
        ),
        "record_without_doctor_label_count": sum(
            item["doctor_label_status"] == "not_available" for item in records
        ),
        "record_with_source_conflict_count": sum(
            item["doctor_label_status"] == "source_conflict" for item in records
        ),
        "record_with_ambiguous_mapping_count": sum(
            item["doctor_label_status"] == "ambiguous_mapping" for item in records
        ),
        "published_doctor_label_count": published_associations,
        "published_doctor_label_association_count": published_associations,
        "distinct_associated_source_label_count": len(used_label_ids),
        "distinct_published_label_id_count": len(distinct_published_label_ids),
        "reused_source_label_association_count": (
            published_associations - len(distinct_published_label_ids)
        ),
        "unmatched_structured_source_label_count": unmatched,
        "conflicting_source_slot_count": conflict_count,
        "fact_consistency_status_counts": {
            status: comparison_counts[status] for status in sorted(_COMPARISON_STATUSES)
        },
        "fact_consistency_disposition_counts": {
            status: sum(
                label["fact_consistency"]["evaluation_disposition"] == status
                for record in records
                for label in record["doctor_labels"]
            )
            for status in sorted(_EVALUATION_DISPOSITIONS)
        },
        "doctor_reference_status_counts": {
            status: sum(
                label["fact_consistency"]["doctor_reference_status"] == status
                for record in records
                for label in record["doctor_labels"]
            )
            for status in sorted(_DOCTOR_REFERENCE_STATUSES)
        },
        "record_consistency_disposition_counts": {
            status: sum(
                record["record_consistency_disposition"] == status
                for record in records
            )
            for status in sorted(_EVALUATION_DISPOSITIONS)
        },
    }
    return records, summary


def _walk_output(value: object) -> Iterable[tuple[str | None, object]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _walk_output(item)
    elif isinstance(value, list):
        for item in value:
            yield None, item
            yield from _walk_output(item)


def _run_leakage_gate(
    value: Mapping[str, Any],
    *,
    raw_inventory_paths: Sequence[str],
    raw_patient_keys: Sequence[str],
    workbook_paths: Sequence[Path],
) -> None:
    forbidden_strings = {
        text
        for text in (
            *raw_inventory_paths,
            *raw_patient_keys,
            *(str(path) for path in workbook_paths),
        )
        if text
    }
    for key, item in _walk_output(value):
        if key in _FORBIDDEN_OUTPUT_KEYS:
            raise ValueError(f"published doctor-label bundle contains forbidden key {key!r}")
        if isinstance(item, str):
            if item in forbidden_strings:
                raise ValueError("published doctor-label bundle contains a raw source value")
            if (
                item.startswith("/")
                or re.match(r"^[A-Za-z]:[\\/]", item) is not None
                or "/mnt/" in item
                or "\\" in item
            ):
                raise ValueError("published doctor-label bundle contains a private source path")


def _strict_mapping(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{context} has missing or unknown keys")
    return deepcopy(dict(value))


def _sha256_value(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _nonnegative_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _validate_report_projection(value: object) -> dict[str, Any]:
    data = _strict_mapping(
        value,
        {
            "status",
            "diagnostic_status",
            "localization_disposition",
            "laterality",
            "regions",
            "projection_source",
            "doctor_label_used",
        },
        "report projection",
    )
    disposition = data["localization_disposition"]
    expected_by_disposition = {
        "localized": ("available", COMPLETED_LOCALIZABLE),
        "nonfocal": ("available", COMPLETED_NONLOCALIZABLE),
        "nonlocalizable": ("available", COMPLETED_NONLOCALIZABLE),
        "insufficient_evidence": ("available", COMPLETED_INSUFFICIENT_EVIDENCE),
        "technical_unassessable": (
            "not_available",
            overlay.TECHNICAL_STATUS,
        ),
    }
    if disposition not in expected_by_disposition or (
        data["status"], data["diagnostic_status"]
    ) != expected_by_disposition[disposition]:
        raise ValueError("report projection disposition/status drifted")
    regions = data["regions"]
    if not isinstance(regions, list) or any(
        region not in _REGIONS for region in regions
    ) or len(regions) != len(set(regions)):
        raise ValueError("report projection regions are unsupported")
    if disposition == "localized":
        if data["laterality"] not in {"left", "right", "midline"} or not regions:
            raise ValueError("localized report projection lacks spatial facts")
        if data["projection_source"] != "frozen_eeg_fact_ledger_only":
            raise ValueError("localized report projection source drifted")
    elif disposition == "nonfocal":
        if data["laterality"] is not None or regions != ["diffuse"]:
            raise ValueError("nonfocal report projection lacks its closed distribution")
        if data["projection_source"] != "frozen_eeg_fact_ledger_only":
            raise ValueError("nonfocal report projection source drifted")
    elif data["laterality"] is not None or regions:
        raise ValueError("nonlocalized report projection carries spatial facts")
    if data["doctor_label_used"] is not False:
        raise ValueError("report projection must not use doctor labels")
    return data


def _validate_channel_reference(value: object) -> dict[str, Any]:
    data = _strict_mapping(
        value,
        {
            "status",
            "reference_scope",
            "significant_electrodes",
            "spread_electrodes",
            "diffuse_spread_present",
            "excluded_out_of_scope_significant_token_count",
            "excluded_out_of_scope_spread_token_count",
            "significant_semantics",
            "spread_semantics",
            "evaluation_only",
            "eligible_for_report_body",
            "eligible_for_llm",
        },
        "physician channel reference",
    )
    if data["status"] not in {"available", "not_available"} or data[
        "reference_scope"
    ] != "standard_19_plus_m1_m2_electrodes":
        raise ValueError("physician channel reference status/scope drifted")
    for key in ("significant_electrodes", "spread_electrodes"):
        values = data[key]
        if not isinstance(values, list) or any(
            item not in _STANDARD_19_PLUS_M1_M2 for item in values
        ) or values != sorted(set(values)):
            raise ValueError("physician channel reference electrodes are invalid")
    for key in (
        "excluded_out_of_scope_significant_token_count",
        "excluded_out_of_scope_spread_token_count",
    ):
        _nonnegative_integer(data[key], key)
    if not isinstance(data["diffuse_spread_present"], bool):
        raise TypeError("diffuse_spread_present must be boolean")
    available = bool(
        data["significant_electrodes"]
        or data["spread_electrodes"]
        or data["diffuse_spread_present"]
    )
    if data["status"] != ("available" if available else "not_available"):
        raise ValueError("physician channel reference availability drifted")
    if (
        data["significant_semantics"]
        != "hard_positive_only_unknown_complement"
        or data["spread_semantics"] != "soft_positive_not_onset"
        or data["evaluation_only"] is not True
        or data["eligible_for_report_body"] is not False
        or data["eligible_for_llm"] is not False
    ):
        raise ValueError("physician channel reference boundary drifted")
    return data


def _validate_label_source_receipt(value: object) -> dict[str, Any]:
    data = _strict_mapping(
        value,
        {
            "workbook_id",
            "workbook_sha256",
            "source_locator_fingerprint",
            "structured_projection_sha256",
            "source_path_included",
            "sheet_name_or_row_number_included",
        },
        "doctor label source receipt",
    )
    if not isinstance(data["workbook_id"], str) or _WORKBOOK_ID_RE.fullmatch(
        data["workbook_id"]
    ) is None:
        raise ValueError("doctor workbook ID is invalid")
    workbook_sha = _sha256_value(data["workbook_sha256"], "workbook SHA-256")
    if data["workbook_id"] != "DRWB-" + workbook_sha[:24]:
        raise ValueError("doctor workbook ID does not bind its hash")
    _sha256_value(data["source_locator_fingerprint"], "source locator fingerprint")
    _sha256_value(
        data["structured_projection_sha256"], "structured projection SHA-256"
    )
    if (
        data["source_path_included"] is not False
        or data["sheet_name_or_row_number_included"] is not False
    ):
        raise ValueError("doctor label source locator leaked raw provenance")
    return data


def _validate_fact_consistency(
    value: object, *, source_conflict: bool
) -> dict[str, Any]:
    base_keys = {
        "policy_id",
        "overall_status",
        "evaluation_disposition",
        "doctor_reference_status",
        "generated_report_status",
        "alignment_code",
        "reason_code",
        "onset_uncertainty",
        "spatial_fields",
        "missing_prediction_is_not_scored_as_mismatch",
        "doctor_unclear_and_report_abstention_is_compatible",
    }
    keys = base_keys | ({"source_conflict_present"} if source_conflict else set())
    data = _strict_mapping(value, keys, "fact consistency")
    if (
        data["policy_id"] != COMPARISON_POLICY_ID
        or data["overall_status"] not in _COMPARISON_STATUSES
        or data["evaluation_disposition"] not in _EVALUATION_DISPOSITIONS
        or data["doctor_reference_status"] not in _DOCTOR_REFERENCE_STATUSES
        or data["generated_report_status"] not in _GENERATED_REPORT_STATUSES
        or data["alignment_code"] not in _ALIGNMENT_CODES
        or not isinstance(data["reason_code"], str)
        or _IDENTIFIER_RE.fullmatch(data["reason_code"]) is None
        or data["missing_prediction_is_not_scored_as_mismatch"] is not True
        or data["doctor_unclear_and_report_abstention_is_compatible"] is not True
    ):
        raise ValueError("fact consistency policy/status drifted")
    uncertainty = _strict_mapping(
        data["onset_uncertainty"],
        {"status", "report_disposition", "doctor_value"},
        "fact consistency onset uncertainty",
    )
    if (
        uncertainty["status"] not in {"match", "mismatch", "not_available"}
        or uncertainty["report_disposition"]
        not in {
            "localized",
            "nonfocal",
            "nonlocalizable",
            "insufficient_evidence",
            "technical_unassessable",
        }
        or uncertainty["doctor_value"] not in _UNCERTAINTIES
    ):
        raise ValueError("fact consistency uncertainty values drifted")
    raw_fields = data["spatial_fields"]
    if not isinstance(raw_fields, list) or len(raw_fields) != 2:
        raise ValueError("fact consistency must contain two spatial fields")
    fields: list[dict[str, Any]] = []
    for expected_field, raw in zip(("laterality", "regions"), raw_fields, strict=True):
        field = _strict_mapping(
            raw,
            {"field", "status", "report_values", "doctor_values", "overlap_values"},
            "fact consistency spatial field",
        )
        if field["field"] != expected_field or field["status"] not in _COMPARISON_STATUSES:
            raise ValueError("fact consistency spatial field status drifted")
        allowed = _LATERALITIES if expected_field == "laterality" else _REGIONS
        for list_key in ("report_values", "doctor_values", "overlap_values"):
            items = field[list_key]
            if not isinstance(items, list) or any(item not in allowed for item in items):
                raise ValueError("fact consistency spatial values are invalid")
            if items != sorted(set(items)):
                raise ValueError("fact consistency spatial values repeat")
        if not set(field["overlap_values"]).issubset(
            set(field["report_values"]).intersection(field["doctor_values"])
        ):
            raise ValueError("fact consistency overlap is invalid")
        fields.append(field)
    data["onset_uncertainty"] = uncertainty
    data["spatial_fields"] = fields
    if source_conflict and (
        data["source_conflict_present"] is not True
        or data["overall_status"] != "not_available"
        or data["evaluation_disposition"] != "source_conflict"
        or data["alignment_code"] != "source_conflict_not_scored"
        or data["reason_code"] != "conflicting_doctor_sources_not_scored"
        or uncertainty["status"] != "not_available"
        or any(field["status"] != "not_available" for field in fields)
    ):
        raise ValueError("conflicting doctor source was not excluded from scoring")
    return data


def _validate_doctor_label(value: object) -> dict[str, Any]:
    data = _strict_mapping(
        value,
        {
            "label_id",
            "source_event_slot",
            "source_receipt",
            "onset",
            "physician_channel_reference",
            "equivalent_source_receipts",
            "duplicate_equivalent_source_count",
            "source_conflict_variant",
            "evaluation_eligible",
            "fact_consistency",
        },
        "doctor label",
    )
    if not isinstance(data["label_id"], str) or _LABEL_ID_RE.fullmatch(
        data["label_id"]
    ) is None:
        raise ValueError("doctor label ID is invalid")
    slot = data["source_event_slot"]
    if not isinstance(slot, str) or _SZ_SLOT_RE.fullmatch(slot) is None:
        raise ValueError("doctor source event slot is invalid")
    source = _validate_label_source_receipt(data["source_receipt"])
    onset = validate_doctor_onset_projection(data["onset"])
    channels = _validate_channel_reference(data["physician_channel_reference"])
    expected_projection_sha = _canonical_sha256(
        {"onset": onset, "physician_channel_reference": channels}
    )
    if source["structured_projection_sha256"] != expected_projection_sha:
        raise ValueError("doctor label source receipt does not bind its projection")
    expected_label_id = "DRLBL-" + _canonical_sha256(
        {
            "workbook_id": source["workbook_id"],
            "source_event_slot": slot,
            "source_locator_fingerprint": source["source_locator_fingerprint"],
            "structured_projection_sha256": expected_projection_sha,
        }
    )[:24]
    if data["label_id"] != expected_label_id:
        raise ValueError("doctor label ID does not bind its source/projection")
    raw_equivalent = data["equivalent_source_receipts"]
    if not isinstance(raw_equivalent, list) or not raw_equivalent:
        raise ValueError("doctor label has no equivalent-source receipt")
    equivalents: list[dict[str, Any]] = []
    for raw in raw_equivalent:
        equivalent = _strict_mapping(
            raw, {"label_id", "source_receipt"}, "equivalent source receipt"
        )
        equivalent_source = _validate_label_source_receipt(
            equivalent["source_receipt"]
        )
        if equivalent_source["structured_projection_sha256"] != expected_projection_sha:
            raise ValueError("equivalent source receipt projection differs")
        equivalent_expected_id = "DRLBL-" + _canonical_sha256(
            {
                "workbook_id": equivalent_source["workbook_id"],
                "source_event_slot": slot,
                "source_locator_fingerprint": equivalent_source[
                    "source_locator_fingerprint"
                ],
                "structured_projection_sha256": expected_projection_sha,
            }
        )[:24]
        if equivalent["label_id"] != equivalent_expected_id:
            raise ValueError("equivalent doctor label ID drifted")
        equivalents.append(
            {"label_id": equivalent_expected_id, "source_receipt": equivalent_source}
        )
    if (
        data["duplicate_equivalent_source_count"] != len(equivalents)
        or equivalents[0]["label_id"] != data["label_id"]
        or equivalents[0]["source_receipt"] != source
        or len({item["label_id"] for item in equivalents}) != len(equivalents)
    ):
        raise ValueError("equivalent doctor source counts/order drifted")
    if not isinstance(data["source_conflict_variant"], bool) or not isinstance(
        data["evaluation_eligible"], bool
    ):
        raise TypeError("doctor label conflict/evaluation flags must be boolean")
    if data["evaluation_eligible"] is data["source_conflict_variant"]:
        raise ValueError("conflicting doctor label must be evaluation-ineligible")
    consistency = _validate_fact_consistency(
        data["fact_consistency"], source_conflict=data["source_conflict_variant"]
    )
    return {
        **data,
        "source_receipt": source,
        "onset": onset,
        "physician_channel_reference": channels,
        "equivalent_source_receipts": equivalents,
        "fact_consistency": consistency,
    }


def validate_postfreeze_doctor_label_bundle(value: object) -> dict[str, Any]:
    """Strictly validate the single-file viewer/evaluation release contract."""

    top_keys = {
        "schema_version",
        "status",
        "inventory_id",
        "recording_unit_policy",
        "record_count",
        "subject_count",
        "coverage_kind",
        "source_receipts",
        "association_summary",
        "records",
        "publication_policy",
        "claim_boundary",
        "leakage_gate",
        "label_release_id",
    }
    data = _strict_mapping(value, top_keys, "doctor label release bundle")
    if (
        data["schema_version"] != SCHEMA_VERSION
        or data["status"] != STATUS
        or not isinstance(data["inventory_id"], str)
        or not data["inventory_id"].startswith("PLINV-")
        or data["recording_unit_policy"] != "unique_signal_sha256_v1"
        or data["coverage_kind"] not in {"full", "combined"}
    ):
        raise ValueError("doctor label release identity/policy drifted")
    record_count = _nonnegative_integer(data["record_count"], "record_count")
    subject_count = _nonnegative_integer(data["subject_count"], "subject_count")
    if record_count < 1 or subject_count < 1:
        raise ValueError("doctor label release cohort must be nonempty")

    raw_records = data["records"]
    if not isinstance(raw_records, list) or len(raw_records) != record_count:
        raise ValueError("doctor label release records do not span the cohort")
    records: list[dict[str, Any]] = []
    recording_ids: set[str] = set()
    all_source_label_ids: set[str] = set()
    all_public_label_ids: list[str] = []
    consistency_counts: Counter[str] = Counter()
    conflict_slot_count = 0
    for raw_record in raw_records:
        record = _strict_mapping(
            raw_record,
            {
                "recording_id",
                "patient_pseudonym",
                "report_receipt",
                "doctor_label_status",
                "doctor_label_count",
                "doctor_labels",
                "record_consistency_disposition",
                "mapping_receipt",
            },
            "doctor label release record",
        )
        recording_id = record["recording_id"]
        patient = record["patient_pseudonym"]
        if (
            not isinstance(recording_id, str)
            or _IDENTIFIER_RE.fullmatch(recording_id) is None
            or not isinstance(patient, str)
            or _IDENTIFIER_RE.fullmatch(patient) is None
            or recording_id in recording_ids
        ):
            raise ValueError("doctor label release record identity is invalid")
        recording_ids.add(recording_id)
        report = _strict_mapping(
            record["report_receipt"],
            {
                "artifact_source",
                "report_kind",
                "diagnostic_status",
                "event_count",
                "report_manifest_relative_path",
                "report_manifest_sha256",
                "report_html_relative_path",
                "report_html_sha256",
                "artifact_hash_set_sha256",
                "report_projection",
            },
            "record report receipt",
        )
        expected_sources = (
            {"full"}
            if data["coverage_kind"] == "full"
            else {"primary", "recovery", "remediation"}
        )
        if report["artifact_source"] not in expected_sources:
            raise ValueError("record report artifact source is invalid")
        if report["report_kind"] not in {
            "eeg_report",
            "technical_unassessable_report",
        }:
            raise ValueError("record report kind is invalid")
        _nonnegative_integer(report["event_count"], "report event_count")
        for key in (
            "report_manifest_sha256",
            "report_html_sha256",
            "artifact_hash_set_sha256",
        ):
            _sha256_value(report[key], key)
        manifest_relative = _safe_relative(
            report["report_manifest_relative_path"], "report manifest locator"
        )
        html_relative = _safe_relative(
            report["report_html_relative_path"], "report HTML locator"
        )
        expected_prefix = PurePosixPath("records") / recording_id
        if (
            manifest_relative.name != "manifest.json"
            or html_relative != manifest_relative.parent / "report.html"
            or manifest_relative.parts[:2] != expected_prefix.parts
        ):
            raise ValueError("record report locator does not bind its recording")
        projection = _validate_report_projection(report["report_projection"])
        if projection["diagnostic_status"] != report["diagnostic_status"]:
            raise ValueError("report projection and receipt diagnostic status differ")
        expected_kind = (
            "technical_unassessable_report"
            if projection["localization_disposition"] == "technical_unassessable"
            else "eeg_report"
        )
        if report["report_kind"] != expected_kind:
            raise ValueError("report projection and report kind differ")
        report["report_projection"] = projection

        raw_labels = record["doctor_labels"]
        if not isinstance(raw_labels, list):
            raise TypeError("record doctor_labels must be an array")
        normalized_labels = [_validate_doctor_label(item) for item in raw_labels]
        if (
            record["doctor_label_count"] != len(normalized_labels)
            or len({item["label_id"] for item in normalized_labels})
            != len(normalized_labels)
        ):
            raise ValueError("record doctor label count/identity drifted")
        for label in normalized_labels:
            all_public_label_ids.append(label["label_id"])
            all_source_label_ids.update(
                item["label_id"] for item in label["equivalent_source_receipts"]
            )
            consistency_counts[label["fact_consistency"]["overall_status"]] += 1
        has_conflict = any(item["source_conflict_variant"] for item in normalized_labels)
        conflict_slot_count += len(
            {
                item["source_event_slot"]
                for item in normalized_labels
                if item["source_conflict_variant"]
            }
        )
        expected_status = (
            "ambiguous_mapping"
            if (
                record["mapping_receipt"].get(
                    "maximum_unique_signal_count_for_mapping_key"
                )
                if isinstance(record["mapping_receipt"], Mapping)
                else 0
            )
            > 1
            and (
                record["mapping_receipt"].get("doctor_source_label_key_present")
                is True
                if isinstance(record["mapping_receipt"], Mapping)
                else False
            )
            else "source_conflict"
            if has_conflict
            else "available"
            if normalized_labels
            else "not_available"
        )
        if record["doctor_label_status"] != expected_status:
            raise ValueError("record doctor label status drifted")
        if record["record_consistency_disposition"] not in _EVALUATION_DISPOSITIONS:
            raise ValueError("record consistency disposition is unsupported")
        label_dispositions = {
            item["fact_consistency"]["evaluation_disposition"]
            for item in normalized_labels
        }
        if projection["localization_disposition"] == "technical_unassessable":
            expected_record_disposition = "technical_unassessable"
        elif expected_status == "ambiguous_mapping":
            expected_record_disposition = "ambiguous_mapping"
        elif has_conflict:
            expected_record_disposition = "source_conflict"
        elif not normalized_labels:
            expected_record_disposition = "label_missing"
        elif "mismatch" in label_dispositions:
            expected_record_disposition = "mismatch"
        elif "generated_abstention" in label_dispositions:
            expected_record_disposition = "generated_abstention"
        elif "label_missing" in label_dispositions:
            expected_record_disposition = "label_missing"
        else:
            expected_record_disposition = "exact_or_compatible"
        if record["record_consistency_disposition"] != expected_record_disposition:
            raise ValueError("record consistency disposition drifted")
        mapping = _strict_mapping(
            record["mapping_receipt"],
            {
                "policy_id",
                "mapping_key_fingerprint",
                "source_event_slot_count",
                "maximum_unique_signal_count_for_mapping_key",
                "doctor_source_label_key_present",
                "associated_at_recording_level",
                "bound_to_detector_candidate_event",
                "raw_patient_identity_included",
                "private_edf_path_included",
                "source_signal_deduplicated_before_association",
            },
            "record mapping receipt",
        )
        if (
            mapping["policy_id"] != MAPPING_POLICY_ID
            or _SHA256_RE.fullmatch(str(mapping["mapping_key_fingerprint"])) is None
            or _nonnegative_integer(
                mapping["source_event_slot_count"], "source_event_slot_count"
            )
            < 1
            or not isinstance(mapping["doctor_source_label_key_present"], bool)
            or _nonnegative_integer(
                mapping["maximum_unique_signal_count_for_mapping_key"],
                "maximum_unique_signal_count_for_mapping_key",
            )
            < 1
            or mapping["associated_at_recording_level"] is not True
            or mapping["bound_to_detector_candidate_event"] is not False
            or mapping["raw_patient_identity_included"] is not False
            or mapping["private_edf_path_included"] is not False
            or mapping["source_signal_deduplicated_before_association"] is not True
        ):
            raise ValueError("record mapping receipt boundary drifted")
        record["report_receipt"] = report
        record["doctor_labels"] = normalized_labels
        record["mapping_receipt"] = mapping
        records.append(record)

    summary = _strict_mapping(
        data["association_summary"],
        {
            "record_count",
            "record_with_doctor_label_count",
            "record_with_any_published_doctor_label_count",
            "record_without_doctor_label_count",
            "record_with_source_conflict_count",
            "record_with_ambiguous_mapping_count",
            "published_doctor_label_count",
            "published_doctor_label_association_count",
            "distinct_associated_source_label_count",
            "distinct_published_label_id_count",
            "reused_source_label_association_count",
            "unmatched_structured_source_label_count",
            "conflicting_source_slot_count",
            "fact_consistency_status_counts",
            "fact_consistency_disposition_counts",
            "doctor_reference_status_counts",
            "record_consistency_disposition_counts",
        },
        "association summary",
    )
    association_count = len(all_public_label_ids)
    distinct_public_count = len(set(all_public_label_ids))
    expected_summary = {
        "record_count": record_count,
        "record_with_doctor_label_count": sum(
            item["doctor_label_status"] == "available" for item in records
        ),
        "record_with_any_published_doctor_label_count": sum(
            bool(item["doctor_labels"]) for item in records
        ),
        "record_without_doctor_label_count": sum(
            item["doctor_label_status"] == "not_available" for item in records
        ),
        "record_with_source_conflict_count": sum(
            item["doctor_label_status"] == "source_conflict" for item in records
        ),
        "record_with_ambiguous_mapping_count": sum(
            item["doctor_label_status"] == "ambiguous_mapping" for item in records
        ),
        "published_doctor_label_count": association_count,
        "published_doctor_label_association_count": association_count,
        "distinct_associated_source_label_count": len(all_source_label_ids),
        "distinct_published_label_id_count": distinct_public_count,
        "reused_source_label_association_count": association_count
        - distinct_public_count,
        "conflicting_source_slot_count": conflict_slot_count,
        "fact_consistency_status_counts": {
            status: consistency_counts[status]
            for status in sorted(_COMPARISON_STATUSES)
        },
        "fact_consistency_disposition_counts": {
            status: sum(
                label["fact_consistency"]["evaluation_disposition"] == status
                for record in records
                for label in record["doctor_labels"]
            )
            for status in sorted(_EVALUATION_DISPOSITIONS)
        },
        "doctor_reference_status_counts": {
            status: sum(
                label["fact_consistency"]["doctor_reference_status"] == status
                for record in records
                for label in record["doctor_labels"]
            )
            for status in sorted(_DOCTOR_REFERENCE_STATUSES)
        },
        "record_consistency_disposition_counts": {
            status: sum(
                record["record_consistency_disposition"] == status
                for record in records
            )
            for status in sorted(_EVALUATION_DISPOSITIONS)
        },
    }
    for key, expected in expected_summary.items():
        if summary[key] != expected:
            raise ValueError(f"association summary {key} drifted")
    unmatched = _nonnegative_integer(
        summary["unmatched_structured_source_label_count"],
        "unmatched_structured_source_label_count",
    )

    source_receipts = _strict_mapping(
        data["source_receipts"],
        {
            "inventory_manifest_sha256",
            "coverage_manifest_sha256",
            "release_audit_sha256",
            "release_audit_id",
            "report_manifest_set_sha256",
            "report_artifact_set_sha256",
            "workbooks",
            "workbook_set_sha256",
            "mapping_provenance_sha256",
            "source_paths_persisted",
        },
        "label release source receipts",
    )
    for key in (
        "inventory_manifest_sha256",
        "coverage_manifest_sha256",
        "release_audit_sha256",
        "report_manifest_set_sha256",
        "report_artifact_set_sha256",
        "workbook_set_sha256",
        "mapping_provenance_sha256",
    ):
        _sha256_value(source_receipts[key], key)
    if (
        not isinstance(source_receipts["release_audit_id"], str)
        or re.fullmatch(r"PLRAUD-[0-9a-f]{24}", source_receipts["release_audit_id"])
        is None
        or source_receipts["source_paths_persisted"] is not False
    ):
        raise ValueError("label release source/audit receipt drifted")
    raw_workbooks = source_receipts["workbooks"]
    if not isinstance(raw_workbooks, list) or not raw_workbooks:
        raise ValueError("label release has no workbook receipts")
    workbooks: list[dict[str, Any]] = []
    for raw in raw_workbooks:
        workbook = _strict_mapping(
            raw,
            {
                "workbook_id",
                "workbook_sha256",
                "structured_source_event_count",
                "source_path_included",
                "raw_cell_values_included",
            },
            "workbook receipt",
        )
        workbook_sha = _sha256_value(
            workbook["workbook_sha256"], "workbook receipt SHA-256"
        )
        if (
            workbook["workbook_id"] != "DRWB-" + workbook_sha[:24]
            or _nonnegative_integer(
                workbook["structured_source_event_count"],
                "structured_source_event_count",
            )
            < 0
            or workbook["source_path_included"] is not False
            or workbook["raw_cell_values_included"] is not False
        ):
            raise ValueError("workbook receipt boundary drifted")
        workbooks.append(workbook)
    if len({item["workbook_sha256"] for item in workbooks}) != len(workbooks):
        raise ValueError("workbook receipts repeat a source file")
    if source_receipts["workbook_set_sha256"] != _canonical_sha256(workbooks):
        raise ValueError("workbook-set receipt drifted")
    if sum(item["structured_source_event_count"] for item in workbooks) != (
        len(all_source_label_ids) + unmatched
    ):
        raise ValueError("workbook source-event accounting does not close")
    workbook_ids = {item["workbook_id"] for item in workbooks}
    if any(
        equivalent["source_receipt"]["workbook_id"] not in workbook_ids
        for record in records
        for label in record["doctor_labels"]
        for equivalent in label["equivalent_source_receipts"]
    ):
        raise ValueError("doctor label references an undeclared workbook")
    if source_receipts["report_manifest_set_sha256"] != _canonical_sha256(
        [
            {
                "recording_id": item["recording_id"],
                "report_manifest_sha256": item["report_receipt"][
                    "report_manifest_sha256"
                ],
            }
            for item in records
        ]
    ) or source_receipts["report_artifact_set_sha256"] != _canonical_sha256(
        [
            {
                "recording_id": item["recording_id"],
                "artifact_hash_set_sha256": item["report_receipt"][
                    "artifact_hash_set_sha256"
                ],
            }
            for item in records
        ]
    ):
        raise ValueError("report hash-set receipt drifted")
    if source_receipts["mapping_provenance_sha256"] != _canonical_sha256(
        [item["mapping_receipt"] for item in records]
    ):
        raise ValueError("mapping provenance receipt drifted")
    source_receipts["workbooks"] = workbooks

    expected_publication_policy = {
        "mapping_policy_id": MAPPING_POLICY_ID,
        "onset_projection_policy_id": PROJECTION_POLICY_ID,
        "fact_consistency_policy_id": COMPARISON_POLICY_ID,
        "doctor_label_unit": "source_recording_sz_slot",
        "report_unit": "unique_complete_eeg_signal_sha256",
        "missing_doctor_label_allowed": True,
        "conflicting_source_label_fails_closed_per_record": True,
        "doctor_label_is_not_bound_to_detected_candidate_event": True,
        "ambiguous_cross_signal_mapping_fails_closed_per_record": True,
        "one_source_label_reused_across_unique_signals": False,
        "significant_electrodes_are_hard_evaluation_reference": True,
        "spread_electrodes_are_soft_evaluation_reference": True,
        "spread_electrodes_are_not_onset_positive": True,
        "unlisted_electrodes_are_not_negative": True,
    }
    if data["publication_policy"] != expected_publication_policy:
        raise ValueError("doctor label publication policy drifted")
    expected_claim_boundary = {
        "reports_and_hashes_frozen_before_workbook_open": True,
        "final_release_audit_pass_required": True,
        "report_artifacts_modified": False,
        "generation_pipeline_read_workbook": False,
        "generation_pipeline_read_edf_annotation": False,
        "doctor_label_used_for_detection": False,
        "doctor_label_used_for_candidate_selection": False,
        "doctor_label_used_for_soz_ranking": False,
        "doctor_label_used_for_findings": False,
        "doctor_label_used_for_impression": False,
        "doctor_label_used_for_renderer": False,
        "doctor_label_used_for_llm": False,
        "raw_onset_free_text_included": False,
        "raw_significant_or_spread_text_included": False,
        "raw_patient_identity_included": False,
        "private_edf_path_included": False,
        "workbook_path_sheet_or_row_included": False,
        "edf_annotation_loaded": False,
        "structured_labels_postfreeze_sidecar_only": True,
    }
    if data["claim_boundary"] != expected_claim_boundary:
        raise ValueError("doctor label claim boundary drifted")
    expected_leakage_gate = {
        "status": "passed",
        "forbidden_output_keys_checked": True,
        "raw_identity_and_private_path_exact_values_checked": True,
        "absolute_private_path_pattern_checked": True,
        "closed_vocabulary_onset_projection_only": True,
        "channel_reference_confined_to_evaluation_only_sidecar": True,
    }
    if data["leakage_gate"] != expected_leakage_gate:
        raise ValueError("doctor label leakage gate drifted")
    body = {key: item for key, item in data.items() if key != "label_release_id"}
    if data["label_release_id"] != "DRREL-" + _canonical_sha256(body)[:24]:
        raise ValueError("doctor label release ID does not bind its content")
    _run_leakage_gate(
        data,
        raw_inventory_paths=[],
        raw_patient_keys=[],
        workbook_paths=[],
    )
    return {
        **data,
        "source_receipts": source_receipts,
        "association_summary": summary,
        "records": records,
    }


def materialize_postfreeze_doctor_label_bundle(
    *,
    inventory_path: str | Path,
    coverage_path: str | Path,
    release_audit_path: str | Path,
    workbook_paths: Sequence[str | Path],
    output_path: str | Path,
    full_root: str | Path | None = None,
    primary_root: str | Path | None = None,
    recovery_root: str | Path | None = None,
    remediation_root: str | Path | None = None,
) -> dict[str, Any]:
    """Publish a PHI-free doctor-label sidecar after report freeze verification."""

    raw_output = Path(output_path)
    if raw_output.exists() or raw_output.is_symlink():
        raise FileExistsError(raw_output)
    freeze = _freeze_report_release(
        inventory_path=Path(inventory_path),
        coverage_path=Path(coverage_path),
        release_audit_path=Path(release_audit_path),
        full_root=Path(full_root) if full_root is not None else None,
        primary_root=Path(primary_root) if primary_root is not None else None,
        recovery_root=Path(recovery_root) if recovery_root is not None else None,
        remediation_root=Path(remediation_root) if remediation_root is not None else None,
    )
    output = raw_output.resolve()
    for root in freeze["roots"].values():
        if output == root or output.is_relative_to(root):
            raise ValueError("doctor-label sidecar must be outside every frozen report tree")

    # Trust-boundary ordering is intentional: no workbook is resolved, hashed,
    # or parsed until the complete report/release snapshot above has passed.
    workbook_path_objects = [Path(item) for item in workbook_paths]
    doctor_events, workbook_receipts, workbook_snapshots = _read_doctor_workbooks(
        workbook_path_objects
    )
    records, association_summary = _associate_labels(
        inventory=freeze["inventory"],
        report_receipts=freeze["report_receipts"],
        doctor_events=doctor_events,
    )
    source_receipts = {
        **freeze["source_receipts"],
        "workbooks": workbook_receipts,
        "workbook_set_sha256": _canonical_sha256(workbook_receipts),
        "mapping_provenance_sha256": _canonical_sha256(
            [item["mapping_receipt"] for item in records]
        ),
        "source_paths_persisted": False,
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "inventory_id": freeze["inventory"]["inventory_id"],
        "recording_unit_policy": "unique_signal_sha256_v1",
        "record_count": freeze["inventory"]["record_count"],
        "subject_count": freeze["inventory"]["subject_count"],
        "coverage_kind": freeze["coverage_kind"],
        "source_receipts": source_receipts,
        "association_summary": association_summary,
        "records": records,
        "publication_policy": {
            "mapping_policy_id": MAPPING_POLICY_ID,
            "onset_projection_policy_id": PROJECTION_POLICY_ID,
            "fact_consistency_policy_id": COMPARISON_POLICY_ID,
            "doctor_label_unit": "source_recording_sz_slot",
            "report_unit": "unique_complete_eeg_signal_sha256",
            "missing_doctor_label_allowed": True,
            "conflicting_source_label_fails_closed_per_record": True,
            "doctor_label_is_not_bound_to_detected_candidate_event": True,
            "ambiguous_cross_signal_mapping_fails_closed_per_record": True,
            "one_source_label_reused_across_unique_signals": False,
            "significant_electrodes_are_hard_evaluation_reference": True,
            "spread_electrodes_are_soft_evaluation_reference": True,
            "spread_electrodes_are_not_onset_positive": True,
            "unlisted_electrodes_are_not_negative": True,
        },
        "claim_boundary": {
            "reports_and_hashes_frozen_before_workbook_open": True,
            "final_release_audit_pass_required": True,
            "report_artifacts_modified": False,
            "generation_pipeline_read_workbook": False,
            "generation_pipeline_read_edf_annotation": False,
            "doctor_label_used_for_detection": False,
            "doctor_label_used_for_candidate_selection": False,
            "doctor_label_used_for_soz_ranking": False,
            "doctor_label_used_for_findings": False,
            "doctor_label_used_for_impression": False,
            "doctor_label_used_for_renderer": False,
            "doctor_label_used_for_llm": False,
            "raw_onset_free_text_included": False,
            "raw_significant_or_spread_text_included": False,
            "raw_patient_identity_included": False,
            "private_edf_path_included": False,
            "workbook_path_sheet_or_row_included": False,
            "edf_annotation_loaded": False,
            "structured_labels_postfreeze_sidecar_only": True,
        },
        "leakage_gate": {
            "status": "passed",
            "forbidden_output_keys_checked": True,
            "raw_identity_and_private_path_exact_values_checked": True,
            "absolute_private_path_pattern_checked": True,
            "closed_vocabulary_onset_projection_only": True,
            "channel_reference_confined_to_evaluation_only_sidecar": True,
        },
    }
    artifact = {
        **body,
        "label_release_id": "DRREL-" + _canonical_sha256(body)[:24],
    }
    artifact = validate_postfreeze_doctor_label_bundle(artifact)
    _run_leakage_gate(
        artifact,
        raw_inventory_paths=[
            str(item["edf_relative_path"]) for item in freeze["inventory"]["records"]
        ],
        raw_patient_keys=[
            str(item["patient_key"]) for item in doctor_events
        ],
        workbook_paths=workbook_path_objects,
    )
    all_snapshots = [*freeze["snapshots"], *workbook_snapshots]
    _assert_snapshots_unchanged(all_snapshots)
    _atomic_json(output, artifact)
    return deepcopy(artifact)


__all__ = [
    "COMPARISON_POLICY_ID",
    "MAPPING_POLICY_ID",
    "PROJECTION_POLICY_ID",
    "SCHEMA_VERSION",
    "STATUS",
    "compare_report_with_doctor_onset",
    "materialize_postfreeze_doctor_label_bundle",
    "project_doctor_onset_text",
    "validate_postfreeze_doctor_label_bundle",
    "validate_doctor_onset_projection",
]
