"""Fail-closed target-free OOF clinical-report assembly.

The v3 assembler joins two independently sealed, target-free artifacts:

* the frozen LaBraM v11.1 MRSC C-CAR19/C-REF19 score artifact; and
* a target-free, paired-reference event-phenotype artifact.

It never opens a DeepSOZ target container or private data.  C-CAR19 patient
and event scores are read-only inputs and are bound by tensor/file digests.
An event that cannot be joined by exact patient/event identity is retained as
a blocked record instead of being silently dropped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Mapping, Sequence

from safetensors import safe_open
import torch

from .clinical_reporting import (
    ClinicalReportFactsV2,
    EventReferenceConsistencyReceipt,
    EventScalpPhenotypeAbstention,
    EventScalpPhenotypeEvidence,
    EvidenceProvenanceReceipt,
    GroundedChineseDiagnosticReport,
    PatientSOZReferenceRanking,
    UncertaintyDecomposition,
    derive_spatial_report,
    render_grounded_chinese_diagnostic_report,
)
from .event_phenotype_producer import EVENT_PHENOTYPE_PRODUCER_SCHEMA
from .final_score_reference_disagreement import (
    FINAL_SCORE_REFERENCE_SOURCE_SCHEMA,
    FINAL_SCORE_REFERENCE_SOURCE_STATUS,
    FINAL_SCORE_SOURCE_TENSOR_KEYS,
    MRSC_CANDIDATE_CHANNELS,
    SOURCE_TENSOR_FILENAME,
    load_final_score_reference_disagreement_receipt,
)
from .final_score_reference_reporting import (
    attach_final_score_reference_disagreement_to_clinical_facts,
)
from .geometry import CHANNEL_INDEX, N_STANDARD_CHANNELS
from .later_visible_region_producer import (
    LATER_VISIBLE_REGION_PRODUCER_SCHEMA,
    LATER_VISIBLE_REGION_RECEIPT_SCHEMA,
    LaterVisibleRegionReceipt,
)
from .later_visible_region_reporting import (
    attach_later_visible_region_to_clinical_facts,
)
from .preprocessing_arm_runtime import CAUSAL_REFERENCE_PAIR_SCHEMA


ASSEMBLER_SCHEMA = "soz_target_free_oof_report_assembler_v3"
ASSEMBLER_STATUS = "completed_target_free_oof_reports_all_rankings_abstain"
PHENOTYPE_SCHEMA = "soz_deepsoz_event_phenotype_target_free_oof_v1"
PHENOTYPE_STATUS = (
    "completed_target_free_development_signal_application_not_evaluation"
)
IDENTITY_POLICY = (
    "exact_mrsc_patient_event_plus_phenotype_internal_signal_receipts_v1"
)
PREDICTION_ROLE = "developmental_oof_target_free_report_draft_not_evaluation"
MODEL_ID = "official-LaBraM-Base-frozen-block9-v11.1"
MODEL_VERSION = "v11.1-oof"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PATIENT_RE = re.compile(r"^[0-9]+$")
_LOCAL_PATIENT_RE = re.compile(r"^[a-z0-9]+$")
_EVENT_RE = re.compile(
    r"^(?P<local>[a-z0-9]+)_s\d+_t\d+__ev(?P<index>\d{4})$"
)
_REASON_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_FROZEN_LATER_VISIBLE_UNAVAILABLE_REASONS = frozenset(
    {
        "primary_event_phenotype_abstained",
        "no_observed_later_visible_derivations",
    }
)

_PHENOTYPE_TOP_LEVEL_KEYS = frozenset(
    {
        "access_receipt",
        "counts",
        "elapsed_sec",
        "event_reference_temporal_tolerance_sec",
        "events",
        "later_visible_region_producer_schema",
        "later_visible_region_receipt_schema",
        "producer_schema",
        "reference_pair_schema",
        "schema_version",
        "scientific_boundary",
        "source_preflight",
        "status",
    }
)
_EVENT_REQUIRED_KEYS = frozenset(
    {
        "abstention",
        "edf_receipt_sha256",
        "edf_sha256",
        "event_id",
        "event_record_sha256",
        "event_reference_consistency_receipt",
        "event_reference_consistency_receipt_sha256",
        "global_event_index",
        "global_stop_sec",
        "global_t0_sec",
        "later_visible_region",
        "local_patient_id",
        "model_split",
        "official_split",
        "ordinal",
        "patient_id",
        "phenotype",
        "primary_arm",
        "processed_window_sha256",
        "reason_codes",
        "relative_edf_path",
        "sensitivity_arm",
        "signal_receipt_sha256",
        "slot_availability",
        "status",
    }
)
_EVENT_OPTIONAL_KEYS: frozenset[str] = frozenset()
_ACCESS_REQUIRED_FALSE = frozenset(
    {
        "deepsoz_target_values_loaded",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "localization_scores_loaded",
        "training_performed",
    }
)
_ACCESS_OPTIONAL_FALSE = frozenset(
    {
        "target_tensor_values_loaded",
        "target_labels_loaded",
        "deepsoz_target_fields_accessed",
        "tusz_channel_target_values_loaded",
        "tusz_native_target_values_loaded",
        "localization_scores_used",
        "model_selection_performed",
        "threshold_selection_or_calibration_performed",
        "calibration_performed",
        "soz_outcome_metrics_computed",
    }
)
_FORBIDDEN_CORTICAL_CLAIMS = (
    "皮层SOZ位于",
    "皮层SOZ可疑位于",
    "皮层SOZ定位为",
    "皮层SOZ首位候选",
    "手术靶点位于",
)
_SLOT_FIELDS = frozenset(
    {
        "artifact_assessment",
        "first_visible_derivations",
        "frequency_range_hz",
        "later_visible_delay",
        "later_visible_destination",
        "later_visible_region",
        "montage_stability",
        "rhythm_state",
        "sustained_change_interval",
    }
)


class TargetFreeOOFReportAssemblyError(ValueError):
    """Base class for a v3 assembly contract failure."""


class UnsafeTargetFreeArtifactError(TargetFreeOOFReportAssemblyError):
    """Raised when an artifact violates the target/private firewall."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = _canonical_bytes(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
        }
    )
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    raw = tensor.view(torch.uint8).numpy().tobytes()
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TargetFreeOOFReportAssemblyError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TargetFreeOOFReportAssemblyError(f"{name} must be non-empty text")
    return value.strip()


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TargetFreeOOFReportAssemblyError(f"{name} must be a mapping")
    return value


def _require_sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TargetFreeOOFReportAssemblyError(f"{name} must be a sequence")
    return value


def _require_reason_codes(value: object, *, name: str) -> tuple[str, ...]:
    raw = _require_sequence(value, name=name)
    result = tuple(_require_text(item, name=f"{name} item") for item in raw)
    if len(set(result)) != len(result) or any(
        _REASON_RE.fullmatch(item) is None for item in result
    ):
        raise TargetFreeOOFReportAssemblyError(
            f"{name} must contain unique stable tokens"
        )
    return result


def _canonical_regular_file(path: str | Path, *, name: str) -> Path:
    lexical = Path(os.path.abspath(path))
    for component in (lexical, *lexical.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise UnsafeTargetFreeArtifactError(f"{name} cannot traverse symlinks")
    try:
        resolved = lexical.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TargetFreeOOFReportAssemblyError(f"{name} does not exist") from exc
    if not resolved.is_file():
        raise TargetFreeOOFReportAssemblyError(f"{name} must be a regular file")
    return resolved


def _canonical_directory(path: str | Path, *, name: str) -> Path:
    lexical = Path(os.path.abspath(path))
    for component in (lexical, *lexical.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise UnsafeTargetFreeArtifactError(f"{name} cannot traverse symlinks")
    try:
        resolved = lexical.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TargetFreeOOFReportAssemblyError(f"{name} does not exist") from exc
    if not resolved.is_dir():
        raise TargetFreeOOFReportAssemblyError(f"{name} must be a directory")
    return resolved


def _load_json_with_anchor(
    path: str | Path,
    *,
    expected_sha256: str,
    name: str,
) -> tuple[Path, Mapping[str, object], str]:
    expected = _require_sha256(expected_sha256, name=f"expected {name} sha256")
    source = _canonical_regular_file(path, name=name)
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise UnsafeTargetFreeArtifactError(f"{name} hash mismatch")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TargetFreeOOFReportAssemblyError(
            f"{name} is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise TargetFreeOOFReportAssemblyError(f"{name} must contain one object")
    return source, payload, actual


def _validate_relative_edf(value: object) -> str:
    text = _require_text(value, name="relative_edf_path")
    relative = PurePosixPath(text)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".edf":
        raise TargetFreeOOFReportAssemblyError("relative_edf_path is unsafe")
    return relative.as_posix()


def _parse_exact_dataclass(
    value: object,
    *,
    cls: type,
    name: str,
    tuple_fields: Sequence[str] = (),
) -> object:
    row = _require_mapping(value, name=name)
    expected = {field.name for field in fields(cls)}
    if set(row) != expected:
        raise UnsafeTargetFreeArtifactError(f"{name} schema drifted")
    converted = dict(row)
    for field_name in tuple_fields:
        converted[field_name] = tuple(
            _require_sequence(converted[field_name], name=f"{name}.{field_name}")
        )
    return cls(**converted)


def _parse_evidence_receipt(
    value: object,
    *,
    expected_montages: tuple[str, ...],
) -> EvidenceProvenanceReceipt:
    receipt = _parse_exact_dataclass(
        value,
        cls=EvidenceProvenanceReceipt,
        name="event evidence receipt",
        tuple_fields=("montages",),
    )
    assert isinstance(receipt, EvidenceProvenanceReceipt)
    if receipt.time_coordinate_semantics != "recording_start_seconds":
        raise UnsafeTargetFreeArtifactError(
            "event time must use recording_start_seconds"
        )
    if receipt.causal_prefix_safe is not True:
        raise UnsafeTargetFreeArtifactError("event evidence must be prefix-safe")
    if receipt.montages != expected_montages:
        raise UnsafeTargetFreeArtifactError(
            "event evidence montage binding changed"
        )
    if any(
        value is not None
        for value in (
            receipt.reference_pair_schema_version,
            receipt.reference_pair_role,
            receipt.reference_primary_arm_id,
            receipt.reference_sensitivity_arm_id,
            receipt.reference_disagreement_metric_id,
            receipt.reference_disagreement_receipt_sha256,
        )
    ):
        raise UnsafeTargetFreeArtifactError(
            "event phenotype cannot carry the older representation receipt"
        )
    return receipt


def _parse_phenotype(
    value: object,
    *,
    expected_montages: tuple[str, ...],
) -> EventScalpPhenotypeEvidence:
    row = _require_mapping(value, name="phenotype")
    expected = {field.name for field in fields(EventScalpPhenotypeEvidence)}
    if set(row) != expected:
        raise UnsafeTargetFreeArtifactError("phenotype schema drifted")
    converted = dict(row)
    converted["receipt"] = _parse_evidence_receipt(
        converted["receipt"], expected_montages=expected_montages
    )
    for name in (
        "first_visible_derivations",
        "later_visible_derivations",
        "artifact_types",
    ):
        converted[name] = tuple(
            _require_sequence(converted[name], name=f"phenotype.{name}")
        )
    if converted["frequency_range_hz"] is not None:
        converted["frequency_range_hz"] = tuple(
            _require_sequence(
                converted["frequency_range_hz"],
                name="phenotype.frequency_range_hz",
            )
        )
    result = EventScalpPhenotypeEvidence(**converted)
    if (
        result.artifact_assessed is not None
        or result.artifact_types
        or result.artifact_burden is not None
    ):
        raise UnsafeTargetFreeArtifactError(
            "unqualified artifact fields must remain empty"
        )
    return result


def _parse_abstention(
    value: object,
    *,
    expected_montages: tuple[str, ...],
) -> EventScalpPhenotypeAbstention:
    row = _require_mapping(value, name="abstention")
    expected = {field.name for field in fields(EventScalpPhenotypeAbstention)}
    if set(row) != expected:
        raise UnsafeTargetFreeArtifactError("abstention schema drifted")
    converted = dict(row)
    converted["receipt"] = _parse_evidence_receipt(
        converted["receipt"], expected_montages=expected_montages
    )
    converted["reason_codes"] = tuple(
        _require_sequence(converted["reason_codes"], name="abstention.reason_codes")
    )
    return EventScalpPhenotypeAbstention(**converted)


def _parse_event_reference(value: object) -> EventReferenceConsistencyReceipt:
    result = _parse_exact_dataclass(
        value,
        cls=EventReferenceConsistencyReceipt,
        name="event reference-consistency receipt",
        tuple_fields=(
            "primary_first_visible_derivations",
            "sensitivity_first_visible_derivations",
            "reason_codes",
        ),
    )
    assert isinstance(result, EventReferenceConsistencyReceipt)
    return result


def _parse_later_region(value: object) -> LaterVisibleRegionReceipt:
    result = _parse_exact_dataclass(
        value,
        cls=LaterVisibleRegionReceipt,
        name="later-visible region receipt",
        tuple_fields=(
            "observed_derivations",
            "canonical_derivations",
            "support_regions",
            "support_lateralities",
        ),
    )
    assert isinstance(result, LaterVisibleRegionReceipt)
    return result


@dataclass(frozen=True)
class _PhenotypeArm:
    status: str
    reason_codes: tuple[str, ...]
    detected_bipolar_edge_count: int
    event: EventScalpPhenotypeEvidence | EventScalpPhenotypeAbstention


def _parse_phenotype_arm(
    value: object,
    *,
    arm_id: str,
) -> _PhenotypeArm:
    row = _require_mapping(value, name=f"{arm_id} phenotype arm")
    expected = {
        "abstention",
        "arm_id",
        "detected_bipolar_edge_count",
        "phenotype",
        "processed_window_sha256",
        "reason_codes",
        "status",
    }
    if set(row) != expected:
        raise UnsafeTargetFreeArtifactError(f"{arm_id} arm schema drifted")
    if row.get("arm_id") != arm_id:
        raise UnsafeTargetFreeArtifactError(f"{arm_id} arm identity changed")
    _require_sha256(
        row.get("processed_window_sha256"),
        name=f"{arm_id}.processed_window_sha256",
    )
    status = _require_text(row.get("status"), name=f"{arm_id}.status")
    reasons = _require_reason_codes(
        row.get("reason_codes"), name=f"{arm_id}.reason_codes"
    )
    count = row.get("detected_bipolar_edge_count")
    if type(count) is not int or not 0 <= count <= 20:
        raise TargetFreeOOFReportAssemblyError(
            f"{arm_id} detected_bipolar_edge_count is invalid"
        )
    expected_montages = (arm_id,)
    if status == "reportable":
        if row.get("phenotype") is None or row.get("abstention") is not None or reasons:
            raise TargetFreeOOFReportAssemblyError(
                f"{arm_id} reportable arm payload is inconsistent"
            )
        event: EventScalpPhenotypeEvidence | EventScalpPhenotypeAbstention = (
            _parse_phenotype(
                row.get("phenotype"), expected_montages=expected_montages
            )
        )
    elif status == "abstained":
        if row.get("phenotype") is not None or row.get("abstention") is None or not reasons:
            raise TargetFreeOOFReportAssemblyError(
                f"{arm_id} abstained arm payload is inconsistent"
            )
        event = _parse_abstention(
            row.get("abstention"), expected_montages=expected_montages
        )
        if event.reason_codes != reasons:
            raise TargetFreeOOFReportAssemblyError(
                f"{arm_id} abstention reasons disagree"
            )
    else:
        raise TargetFreeOOFReportAssemblyError(f"{arm_id} status is invalid")
    if isinstance(event, EventScalpPhenotypeAbstention) and (
        event.detected_bipolar_edge_count != count
    ):
        raise TargetFreeOOFReportAssemblyError(
            f"{arm_id} detected-edge count disagrees"
        )
    if isinstance(event, EventScalpPhenotypeEvidence) and (
        event.montage_stability is not None
        or event.later_visible_region_zh is not None
    ):
        raise UnsafeTargetFreeArtifactError(
            f"{arm_id} arm must precede report-only reference/region binding"
        )
    return _PhenotypeArm(
        status=status,
        reason_codes=reasons,
        detected_bipolar_edge_count=count,
        event=event,
    )


@dataclass(frozen=True)
class _LaterVisibleBinding:
    status: str
    reason_codes: tuple[str, ...]
    receipt: LaterVisibleRegionReceipt | None


def _parse_later_visible_binding(value: object) -> _LaterVisibleBinding:
    row = _require_mapping(value, name="later_visible_region")
    if set(row) != {"status", "reason_codes", "receipt", "receipt_sha256"}:
        raise UnsafeTargetFreeArtifactError("later-visible binding schema drifted")
    status = _require_text(row.get("status"), name="later_visible_region.status")
    reasons = _require_reason_codes(
        row.get("reason_codes"), name="later_visible_region.reason_codes"
    )
    raw_receipt = row.get("receipt")
    raw_sha = row.get("receipt_sha256")
    if status == "mapped":
        if raw_receipt is None or raw_sha is None or reasons:
            raise TargetFreeOOFReportAssemblyError(
                "mapped later-visible binding is incomplete"
            )
        receipt = _parse_later_region(raw_receipt)
        declared_sha = _require_sha256(
            raw_sha, name="later_visible_region.receipt_sha256"
        )
        if receipt.receipt_sha256 != declared_sha:
            raise TargetFreeOOFReportAssemblyError(
                "later-visible receipt digest mismatch"
            )
    elif status == "unavailable":
        if (
            raw_receipt is not None
            or raw_sha is not None
            or len(reasons) != 1
            or reasons[0] not in _FROZEN_LATER_VISIBLE_UNAVAILABLE_REASONS
        ):
            raise TargetFreeOOFReportAssemblyError(
                "unavailable later-visible binding is not a frozen producer abstention"
            )
        receipt = None
    elif status == "abstained":
        if raw_receipt is not None or raw_sha is not None or not reasons:
            raise TargetFreeOOFReportAssemblyError(
                "abstained later-visible binding is inconsistent"
            )
        receipt = None
    else:
        raise TargetFreeOOFReportAssemblyError(
            "later-visible binding status is invalid"
        )
    return _LaterVisibleBinding(status=status, reason_codes=reasons, receipt=receipt)


@dataclass(frozen=True)
class _PhenotypeEvent:
    patient_id: str
    local_patient_id: str
    event_id: str
    relative_edf_path: str
    global_t0_sec: float
    global_stop_sec: float
    global_event_index: int
    edf_sha256: str
    edf_receipt_sha256: str
    event_record_sha256: str
    signal_receipt_sha256: str
    processed_window_sha256: str
    model_split: str
    official_split: str
    ordinal: int
    status: str
    reason_codes: tuple[str, ...]
    primary_arm: _PhenotypeArm
    sensitivity_arm: _PhenotypeArm
    event: EventScalpPhenotypeEvidence | EventScalpPhenotypeAbstention
    event_reference_receipt: EventReferenceConsistencyReceipt
    later_visible_region_receipt: LaterVisibleRegionReceipt | None


def _parse_phenotype_event(value: object) -> _PhenotypeEvent:
    row = _require_mapping(value, name="phenotype event")
    keys = set(row)
    if not _EVENT_REQUIRED_KEYS.issubset(keys) or not keys.issubset(
        _EVENT_REQUIRED_KEYS | _EVENT_OPTIONAL_KEYS
    ):
        raise UnsafeTargetFreeArtifactError("phenotype event schema drifted")
    patient_id = _require_text(row.get("patient_id"), name="patient_id")
    if _PATIENT_RE.fullmatch(patient_id) is None:
        raise TargetFreeOOFReportAssemblyError("patient_id must be numeric text")
    local_patient = _require_text(
        row.get("local_patient_id"), name="local_patient_id"
    )
    if _LOCAL_PATIENT_RE.fullmatch(local_patient) is None:
        raise TargetFreeOOFReportAssemblyError("local_patient_id is invalid")
    event_id = _require_text(row.get("event_id"), name="event_id")
    match = _EVENT_RE.fullmatch(event_id)
    if match is None:
        raise TargetFreeOOFReportAssemblyError("event_id is not canonical")
    if match.group("local") != local_patient:
        raise TargetFreeOOFReportAssemblyError(
            "event_id and local_patient_id disagree"
        )
    global_index = row.get("global_event_index")
    if type(global_index) is not int or global_index < 0:
        raise TargetFreeOOFReportAssemblyError("global_event_index is invalid")
    if int(match.group("index")) != global_index:
        raise TargetFreeOOFReportAssemblyError(
            "event_id and global_event_index disagree"
        )
    relative_edf = _validate_relative_edf(row.get("relative_edf_path"))
    if local_patient not in PurePosixPath(relative_edf).parts:
        raise TargetFreeOOFReportAssemblyError(
            "relative_edf_path and local_patient_id disagree"
        )
    global_t0 = row.get("global_t0_sec")
    if (
        isinstance(global_t0, bool)
        or not isinstance(global_t0, (int, float))
        or not math.isfinite(float(global_t0))
        or float(global_t0) < 0
    ):
        raise TargetFreeOOFReportAssemblyError("global_t0_sec is invalid")
    global_stop = row.get("global_stop_sec")
    if (
        isinstance(global_stop, bool)
        or not isinstance(global_stop, (int, float))
        or not math.isfinite(float(global_stop))
        or float(global_stop) <= float(global_t0)
    ):
        raise TargetFreeOOFReportAssemblyError("global_stop_sec is invalid")
    edf_sha = _require_sha256(row.get("edf_sha256"), name="edf_sha256")
    edf_receipt_sha = _require_sha256(
        row.get("edf_receipt_sha256"), name="edf_receipt_sha256"
    )
    event_record_sha = _require_sha256(
        row.get("event_record_sha256"), name="event_record_sha256"
    )
    signal_receipt_sha = _require_sha256(
        row.get("signal_receipt_sha256"), name="signal_receipt_sha256"
    )
    processed_sha = _require_sha256(
        row.get("processed_window_sha256"), name="processed_window_sha256"
    )
    model_split = _require_text(row.get("model_split"), name="model_split")
    official_split = _require_text(row.get("official_split"), name="official_split")
    ordinal = row.get("ordinal")
    if type(ordinal) is not int or ordinal < 0:
        raise TargetFreeOOFReportAssemblyError("ordinal is invalid")
    status = _require_text(row.get("status"), name="status")
    if status not in {"reportable", "abstained"}:
        raise TargetFreeOOFReportAssemblyError("event status is invalid")
    reasons = _require_reason_codes(row.get("reason_codes"), name="reason_codes")
    primary_arm = _parse_phenotype_arm(row.get("primary_arm"), arm_id="C-CAR19")
    sensitivity_arm = _parse_phenotype_arm(
        row.get("sensitivity_arm"), arm_id="C-REF19"
    )
    primary_arm_payload = _require_mapping(
        row.get("primary_arm"), name="primary_arm"
    )
    if primary_arm_payload.get("processed_window_sha256") != processed_sha:
        raise TargetFreeOOFReportAssemblyError(
            "top-level and C-CAR19 processed-window hashes disagree"
        )
    phenotype_raw = row.get("phenotype")
    abstention_raw = row.get("abstention")
    if status == "reportable":
        if phenotype_raw is None or abstention_raw is not None or reasons:
            raise TargetFreeOOFReportAssemblyError(
                "reportable row must contain only a phenotype"
            )
        event: EventScalpPhenotypeEvidence | EventScalpPhenotypeAbstention = (
            _parse_phenotype(
                phenotype_raw, expected_montages=("C-CAR19", "C-REF19")
            )
        )
    else:
        if phenotype_raw is not None or abstention_raw is None or not reasons:
            raise TargetFreeOOFReportAssemblyError(
                "abstained row must contain only a typed abstention"
            )
        event = _parse_abstention(
            abstention_raw, expected_montages=("C-CAR19", "C-REF19")
        )
        if event.reason_codes != reasons:
            raise TargetFreeOOFReportAssemblyError(
                "row and typed abstention reason codes disagree"
            )
    event_reference = _parse_event_reference(
        row.get("event_reference_consistency_receipt")
    )
    if not math.isclose(
        event_reference.temporal_alignment_tolerance_sec,
        0.25,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise UnsafeTargetFreeArtifactError(
            "event reference temporal tolerance changed"
        )
    declared_event_reference_sha = _require_sha256(
        row.get("event_reference_consistency_receipt_sha256"),
        name="event_reference_consistency_receipt_sha256",
    )
    if event_reference.receipt_sha256 != declared_event_reference_sha:
        raise TargetFreeOOFReportAssemblyError(
            "event reference-consistency receipt digest mismatch"
        )
    later_binding = _parse_later_visible_binding(row.get("later_visible_region"))
    later = later_binding.receipt
    if later_binding.status == "unavailable":
        if isinstance(event, EventScalpPhenotypeAbstention):
            expected_unavailable_reason = "primary_event_phenotype_abstained"
        elif event.later_visible_derivations:
            raise TargetFreeOOFReportAssemblyError(
                "observed later-visible derivations cannot be marked unavailable"
            )
        else:
            expected_unavailable_reason = (
                "no_observed_later_visible_derivations"
            )
        if later_binding.reason_codes != (expected_unavailable_reason,):
            raise TargetFreeOOFReportAssemblyError(
                "later-visible unavailable reason disagrees with phenotype status"
            )
    slot_availability = _require_mapping(
        row.get("slot_availability"), name="slot_availability"
    )
    if set(slot_availability) != set(_SLOT_FIELDS) or any(
        type(value) is not bool for value in slot_availability.values()
    ):
        raise UnsafeTargetFreeArtifactError("slot-availability schema drifted")
    if slot_availability.get("artifact_assessment") is not False:
        raise UnsafeTargetFreeArtifactError(
            "artifact_assessment slot must remain unavailable"
        )

    receipt = event.receipt
    identity_pairs = (
        (receipt.patient_pseudonym, patient_id),
        (receipt.event_pseudonym, event_id),
        (receipt.signal_artifact_sha256, edf_sha),
        (event_reference.patient_pseudonym, patient_id),
        (event_reference.event_pseudonym, event_id),
        (event_reference.signal_artifact_sha256, edf_sha),
        (event_reference.primary_evidence_artifact_sha256, receipt.evidence_artifact_sha256),
        (event_reference.primary_result_status, status),
        (primary_arm.event.receipt.patient_pseudonym, patient_id),
        (primary_arm.event.receipt.event_pseudonym, event_id),
        (primary_arm.event.receipt.signal_artifact_sha256, edf_sha),
        (sensitivity_arm.event.receipt.patient_pseudonym, patient_id),
        (sensitivity_arm.event.receipt.event_pseudonym, event_id),
        (sensitivity_arm.event.receipt.signal_artifact_sha256, edf_sha),
        (event_reference.primary_result_status, primary_arm.status),
        (event_reference.sensitivity_result_status, sensitivity_arm.status),
        (
            event_reference.primary_evidence_artifact_sha256,
            primary_arm.event.receipt.evidence_artifact_sha256,
        ),
        (
            event_reference.sensitivity_evidence_artifact_sha256,
            sensitivity_arm.event.receipt.evidence_artifact_sha256,
        ),
    )
    if any(actual != expected for actual, expected in identity_pairs):
        raise TargetFreeOOFReportAssemblyError(
            "event row and typed receipt identity disagree"
        )
    if isinstance(event, EventScalpPhenotypeEvidence):
        if event.montage_stability != event_reference.montage_stability:
            raise TargetFreeOOFReportAssemblyError(
                "event phenotype and reference receipt stability disagree"
            )
        if event.onset_start_sec + 1e-9 < float(global_t0):
            raise TargetFreeOOFReportAssemblyError(
                "record-start phenotype time precedes the indexed event anchor"
            )
        expected_slots = {
            "artifact_assessment": False,
            "first_visible_derivations": bool(event.first_visible_derivations),
            "frequency_range_hz": event.frequency_range_hz is not None,
            "later_visible_delay": event.later_visible_delay_sec is not None,
            "later_visible_destination": bool(
                event.later_visible_derivations or event.later_visible_region_zh
            ),
            "later_visible_region": later is not None,
            "montage_stability": event.montage_stability is not None,
            "rhythm_state": event.rhythm_state is not None,
            "sustained_change_interval": True,
        }
        if dict(slot_availability) != expected_slots:
            raise TargetFreeOOFReportAssemblyError(
                "slot availability does not replay typed phenotype facts"
            )
    elif event_reference.montage_stability is not None:
        raise TargetFreeOOFReportAssemblyError(
            "an abstained phenotype cannot carry montage stability"
        )
    elif any(slot_availability.values()):
        raise TargetFreeOOFReportAssemblyError(
            "abstained event cannot expose report fact slots"
        )
    if later is not None and (
        later.patient_pseudonym != patient_id
        or later.event_pseudonym != event_id
    ):
        raise TargetFreeOOFReportAssemblyError(
            "later-visible receipt identity mismatch"
        )
    if isinstance(event, EventScalpPhenotypeAbstention) and later is not None:
        raise TargetFreeOOFReportAssemblyError(
            "an abstained event cannot carry a later-visible region"
        )
    if (
        isinstance(event, EventScalpPhenotypeEvidence)
        and event.later_visible_region_zh is not None
        and (
            later is None
            or event.later_visible_region_zh != later.later_visible_region_zh
        )
    ):
        raise TargetFreeOOFReportAssemblyError(
            "pre-bound later-visible region disagrees with its receipt"
        )

    # The top-level event is the C-CAR19 arm after two report-only adapters:
    # paired-reference binding, then optional deterministic later-region binding.
    primary_event = primary_arm.event
    if type(primary_event) is not type(event):
        raise TargetFreeOOFReportAssemblyError(
            "reference binding changed primary event status"
        )
    if isinstance(event, EventScalpPhenotypeEvidence):
        assert isinstance(primary_event, EventScalpPhenotypeEvidence)
        expected_bound = replace(
            primary_event,
            receipt=replace(
                primary_event.receipt,
                montages=("C-CAR19", "C-REF19"),
            ),
            montage_stability=event_reference.montage_stability,
            later_visible_region_zh=event.later_visible_region_zh,
        )
        if event != expected_bound:
            raise TargetFreeOOFReportAssemblyError(
                "bound event changed non-reference phenotype facts"
            )
    else:
        assert isinstance(primary_event, EventScalpPhenotypeAbstention)
        expected_bound_abstention = replace(
            primary_event,
            receipt=replace(
                primary_event.receipt,
                montages=("C-CAR19", "C-REF19"),
            ),
        )
        if event != expected_bound_abstention:
            raise TargetFreeOOFReportAssemblyError(
                "bound abstention changed primary facts"
            )

    return _PhenotypeEvent(
        patient_id=patient_id,
        local_patient_id=local_patient,
        event_id=event_id,
        relative_edf_path=relative_edf,
        global_t0_sec=float(global_t0),
        global_stop_sec=float(global_stop),
        global_event_index=global_index,
        edf_sha256=edf_sha,
        edf_receipt_sha256=edf_receipt_sha,
        event_record_sha256=event_record_sha,
        signal_receipt_sha256=signal_receipt_sha,
        processed_window_sha256=processed_sha,
        model_split=model_split,
        official_split=official_split,
        ordinal=ordinal,
        status=status,
        reason_codes=reasons,
        primary_arm=primary_arm,
        sensitivity_arm=sensitivity_arm,
        event=event,
        event_reference_receipt=event_reference,
        later_visible_region_receipt=later,
    )


@dataclass(frozen=True)
class _PhenotypeBundle:
    raw_events: tuple[Mapping[str, object], ...]
    artifact_sha256: str


def _load_phenotype_bundle(
    path: str | Path,
    *,
    expected_sha256: str,
) -> _PhenotypeBundle:
    _, payload, actual_sha = _load_json_with_anchor(
        path,
        expected_sha256=expected_sha256,
        name="target-free phenotype artifact",
    )
    if set(payload) != set(_PHENOTYPE_TOP_LEVEL_KEYS):
        raise UnsafeTargetFreeArtifactError(
            "target-free phenotype top-level schema drifted"
        )
    if payload.get("schema_version") != PHENOTYPE_SCHEMA:
        raise UnsafeTargetFreeArtifactError("unsupported phenotype schema")
    if payload.get("producer_schema") != EVENT_PHENOTYPE_PRODUCER_SCHEMA:
        raise UnsafeTargetFreeArtifactError("phenotype producer schema changed")
    if payload.get("reference_pair_schema") != CAUSAL_REFERENCE_PAIR_SCHEMA:
        raise UnsafeTargetFreeArtifactError("phenotype reference-pair schema changed")
    if payload.get("later_visible_region_producer_schema") != (
        LATER_VISIBLE_REGION_PRODUCER_SCHEMA
    ) or payload.get("later_visible_region_receipt_schema") != (
        LATER_VISIBLE_REGION_RECEIPT_SCHEMA
    ):
        raise UnsafeTargetFreeArtifactError("later-visible region schema changed")
    if payload.get("status") != PHENOTYPE_STATUS:
        raise UnsafeTargetFreeArtifactError("phenotype artifact is not completed")
    elapsed = payload.get("elapsed_sec")
    tolerance = payload.get("event_reference_temporal_tolerance_sec")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0
        or isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isclose(float(tolerance), 0.25, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise UnsafeTargetFreeArtifactError(
            "phenotype runtime/tolerance contract changed"
        )
    access = _require_mapping(payload.get("access_receipt"), name="access_receipt")
    source_preflight = _require_mapping(
        payload.get("source_preflight"), name="source_preflight"
    )
    missing = _ACCESS_REQUIRED_FALSE - set(access)
    if missing:
        raise UnsafeTargetFreeArtifactError(
            "phenotype access receipt lacks mandatory firewall declarations"
        )
    for name in _ACCESS_REQUIRED_FALSE | (_ACCESS_OPTIONAL_FALSE & set(access)):
        if access.get(name) is not False:
            raise UnsafeTargetFreeArtifactError(
                f"phenotype access firewall failed: {name}"
            )
    scientific = _require_mapping(
        payload.get("scientific_boundary"), name="scientific_boundary"
    )
    expected_scientific = {
        "allowed_use": "report_fact_availability_or_abstention_only",
        "measurement": "target_free_scalp_visible_event_phenotype",
        "cortical_soz_claim_allowed": False,
        "earliest_physical_electrode_claim_allowed": False,
        "propagation_truth_claim_allowed": False,
        "soz_score_modification_allowed": False,
    }
    if scientific != expected_scientific:
        raise UnsafeTargetFreeArtifactError(
            "phenotype scientific boundary changed"
        )
    for name in (
        "artifact_sha256",
        "eligible_event_roster_sha256",
        "preprocess_config_sha256",
        "receipt_sha256",
    ):
        _require_sha256(source_preflight.get(name), name=f"source_preflight.{name}")
    for name in ("eligible_event_count", "eligible_patient_count"):
        value = source_preflight.get(name)
        if type(value) is not int or value < 1:
            raise UnsafeTargetFreeArtifactError(
                f"source_preflight.{name} is invalid"
            )
    raw_events = tuple(
        _require_mapping(value, name="events item")
        for value in _require_sequence(payload.get("events"), name="events")
    )
    counts = _require_mapping(payload.get("counts"), name="counts")
    declared_event_count = counts.get("materialized_events")
    if type(declared_event_count) is not int or declared_event_count != len(raw_events):
        raise UnsafeTargetFreeArtifactError("phenotype event count does not replay")
    materialized_patients = counts.get("materialized_patients")
    if type(materialized_patients) is not int or materialized_patients < 1:
        raise UnsafeTargetFreeArtifactError("phenotype patient count is invalid")
    if counts.get("input_signal_eligible_events") != source_preflight.get(
        "eligible_event_count"
    ) or counts.get("input_signal_eligible_patients") != source_preflight.get(
        "eligible_patient_count"
    ):
        raise UnsafeTargetFreeArtifactError(
            "phenotype and source-preflight counts disagree"
        )
    event_ids: list[str] = []
    for row in raw_events:
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise UnsafeTargetFreeArtifactError(
                "every phenotype row needs an event identity"
            )
        event_ids.append(event_id)
    if len(set(event_ids)) != len(event_ids):
        raise UnsafeTargetFreeArtifactError("phenotype event roster has duplicates")
    patient_ids = {
        str(row.get("patient_id"))
        for row in raw_events
        if isinstance(row.get("patient_id"), str)
    }
    if len(patient_ids) != materialized_patients:
        raise UnsafeTargetFreeArtifactError(
            "phenotype materialized patient count does not replay"
        )
    return _PhenotypeBundle(raw_events=raw_events, artifact_sha256=actual_sha)


@dataclass(frozen=True)
class _MRSCBundle:
    directory: Path
    manifest_sha256: str
    tensor_sha256: str
    model_lineage: str
    patient_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    event_patient_index: torch.Tensor
    car_patient_scores: torch.Tensor
    car_event_scores: torch.Tensor
    ranking_ambiguity: torch.Tensor
    event_dispersion: torch.Tensor
    event_dispersion_estimable: torch.Tensor
    final_score_reference_disagreement: torch.Tensor
    abstain: torch.Tensor

    @property
    def event_to_index(self) -> Mapping[str, int]:
        return {event_id: index for index, event_id in enumerate(self.event_ids)}

    @property
    def patient_to_index(self) -> Mapping[str, int]:
        return {patient_id: index for index, patient_id in enumerate(self.patient_ids)}


def _load_mrsc_bundle(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_tensor_sha256: str,
) -> _MRSCBundle:
    root = _canonical_directory(directory, name="MRSC directory")
    manifest_path, manifest, manifest_sha = _load_json_with_anchor(
        root / "manifest.json",
        expected_sha256=expected_manifest_sha256,
        name="MRSC manifest",
    )
    if manifest_path.parent != root:
        raise UnsafeTargetFreeArtifactError("MRSC manifest escaped its directory")
    if manifest.get("schema_version") != FINAL_SCORE_REFERENCE_SOURCE_SCHEMA or (
        manifest.get("status") != FINAL_SCORE_REFERENCE_SOURCE_STATUS
    ):
        raise UnsafeTargetFreeArtifactError("unsupported MRSC source contract")
    if manifest.get("tensor_file") != SOURCE_TENSOR_FILENAME:
        raise UnsafeTargetFreeArtifactError("MRSC tensor filename changed")
    if tuple(manifest.get("candidate_channels", ())) != MRSC_CANDIDATE_CHANNELS:
        raise UnsafeTargetFreeArtifactError("MRSC candidate order changed")
    access = _require_mapping(manifest.get("access_receipt"), name="MRSC access")
    for name in (
        "target_tensor_values_loaded",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "training_performed",
        "model_selection_performed",
        "threshold_selection_or_calibration_performed",
        "soz_outcome_metrics_computed",
        "label_based_subgrouping_performed",
    ):
        if access.get(name) is not False:
            raise UnsafeTargetFreeArtifactError(f"MRSC is not target-free: {name}")
    parity = _require_mapping(manifest.get("score_parity"), name="score_parity")
    for name in (
        "car_patient_bitwise_equal_before_after_mrsc",
        "car_event_bitwise_equal_before_after_mrsc",
        "r2_outer_state_car_replay_gate_passed",
    ):
        if parity.get(name) is not True:
            raise UnsafeTargetFreeArtifactError(f"MRSC score parity failed: {name}")
    if parity.get("maximum_absolute_car_score_change") != 0.0:
        raise UnsafeTargetFreeArtifactError("MRSC changed primary CAR scores")
    contract = _require_mapping(manifest.get("mrsc_contract"), name="mrsc_contract")
    if contract.get("selective_threshold_defined") is not False or (
        contract.get("all_patients_fail_closed") is not True
    ):
        raise UnsafeTargetFreeArtifactError("MRSC threshold/abstention contract changed")
    if contract.get("abstention_reason_vocabulary") != [
        "selective_threshold_undefined"
    ]:
        raise UnsafeTargetFreeArtifactError("MRSC abstention vocabulary changed")

    patient_ids = tuple(
        _require_text(value, name="MRSC patient_id")
        for value in _require_sequence(manifest.get("patient_ids"), name="patient_ids")
    )
    event_ids = tuple(
        _require_text(value, name="MRSC event_id")
        for value in _require_sequence(manifest.get("event_ids"), name="event_ids")
    )
    patient_count = manifest.get("patient_count")
    event_count = manifest.get("event_count")
    if (
        type(patient_count) is not int
        or type(event_count) is not int
        or patient_count < 1
        or event_count < patient_count
        or len(patient_ids) != patient_count
        or len(event_ids) != event_count
        or len(set(patient_ids)) != patient_count
        or len(set(event_ids)) != event_count
    ):
        raise UnsafeTargetFreeArtifactError("MRSC roster/count contract failed")

    tensor_path = _canonical_regular_file(
        root / SOURCE_TENSOR_FILENAME, name="MRSC tensor"
    )
    if tensor_path.parent != root:
        raise UnsafeTargetFreeArtifactError("MRSC tensor escaped its directory")
    expected_tensor = _require_sha256(
        expected_tensor_sha256, name="expected MRSC tensor sha256"
    )
    tensor_sha = _file_sha256(tensor_path)
    if not hmac.compare_digest(tensor_sha, expected_tensor):
        raise UnsafeTargetFreeArtifactError("MRSC tensor hash mismatch")
    specs = _require_mapping(manifest.get("tensor_specs"), name="tensor_specs")
    if set(specs) != set(FINAL_SCORE_SOURCE_TENSOR_KEYS):
        raise UnsafeTargetFreeArtifactError("MRSC tensor spec vocabulary changed")
    required = (
        "event_patient_index",
        "car_patient_scores_preserved",
        "car_event_scores_preserved",
        "mrsc_ranking_ambiguity",
        "mrsc_event_dispersion",
        "mrsc_event_dispersion_estimable",
        "mrsc_final_score_reference_disagreement",
        "mrsc_abstain",
        "mrsc_abstention_reason_flags",
    )
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if keys != set(FINAL_SCORE_SOURCE_TENSOR_KEYS):
            raise UnsafeTargetFreeArtifactError("MRSC tensor vocabulary changed")
        if any("target" in key.lower() for key in keys):
            raise UnsafeTargetFreeArtifactError("MRSC co-serializes target-like tensors")
        for name in required:
            value = handle.get_tensor(name).detach().cpu().contiguous()
            spec = _require_mapping(specs.get(name), name=f"tensor_specs.{name}")
            if list(value.shape) != spec.get("shape") or str(value.dtype) != spec.get(
                "dtype"
            ):
                raise UnsafeTargetFreeArtifactError(
                    f"MRSC tensor/spec mismatch: {name}"
                )
            tensors[name] = value
    expected_shapes = {
        "event_patient_index": (event_count,),
        "car_patient_scores_preserved": (patient_count, 18),
        "car_event_scores_preserved": (event_count, 18),
        "mrsc_ranking_ambiguity": (patient_count,),
        "mrsc_event_dispersion": (patient_count,),
        "mrsc_event_dispersion_estimable": (patient_count,),
        "mrsc_final_score_reference_disagreement": (patient_count,),
        "mrsc_abstain": (patient_count,),
        "mrsc_abstention_reason_flags": (patient_count, 1),
    }
    for name, shape in expected_shapes.items():
        if tuple(tensors[name].shape) != shape:
            raise UnsafeTargetFreeArtifactError(f"MRSC tensor shape changed: {name}")
    if tensors["event_patient_index"].dtype != torch.long:
        raise UnsafeTargetFreeArtifactError("MRSC event routing must be int64")
    routing = tensors["event_patient_index"]
    if int(routing.min()) != 0 or int(routing.max()) != patient_count - 1:
        raise UnsafeTargetFreeArtifactError("MRSC event routing is not contiguous")
    for name in (
        "car_patient_scores_preserved",
        "car_event_scores_preserved",
    ):
        if tensors[name].dtype != torch.float32 or not bool(
            torch.isfinite(tensors[name]).all()
        ):
            raise UnsafeTargetFreeArtifactError(f"MRSC scores are invalid: {name}")
    for name in (
        "mrsc_ranking_ambiguity",
        "mrsc_event_dispersion",
        "mrsc_final_score_reference_disagreement",
    ):
        if tensors[name].dtype != torch.float64 or not bool(
            torch.isfinite(tensors[name]).all()
        ):
            raise UnsafeTargetFreeArtifactError(
                f"MRSC uncertainty is invalid: {name}"
            )
    for name in (
        "mrsc_event_dispersion_estimable",
        "mrsc_abstain",
        "mrsc_abstention_reason_flags",
    ):
        if tensors[name].dtype != torch.bool:
            raise UnsafeTargetFreeArtifactError(f"MRSC flag must be bool: {name}")
    if not bool(tensors["mrsc_abstain"].all()) or not bool(
        tensors["mrsc_abstention_reason_flags"].all()
    ):
        raise UnsafeTargetFreeArtifactError(
            "selective_threshold_undefined must abstain for every patient"
        )
    return _MRSCBundle(
        directory=root,
        manifest_sha256=manifest_sha,
        tensor_sha256=tensor_sha,
        model_lineage=_require_text(manifest.get("model_lineage"), name="model_lineage"),
        patient_ids=patient_ids,
        event_ids=event_ids,
        event_patient_index=routing,
        car_patient_scores=tensors["car_patient_scores_preserved"],
        car_event_scores=tensors["car_event_scores_preserved"],
        ranking_ambiguity=tensors["mrsc_ranking_ambiguity"],
        event_dispersion=tensors["mrsc_event_dispersion"],
        event_dispersion_estimable=tensors["mrsc_event_dispersion_estimable"],
        final_score_reference_disagreement=tensors[
            "mrsc_final_score_reference_disagreement"
        ],
        abstain=tensors["mrsc_abstain"],
    )


@dataclass(frozen=True)
class TargetFreeOOFAssemblyReceiptV3:
    patient_id: str
    local_patient_id: str
    event_id: str
    relative_edf_path: str
    global_t0_sec: float
    global_stop_sec: float
    global_event_index: int
    edf_sha256: str
    edf_receipt_sha256: str
    event_record_sha256: str
    signal_receipt_sha256: str
    processed_window_sha256: str
    phenotype_artifact_sha256: str
    mrsc_manifest_sha256: str
    mrsc_tensor_sha256: str
    outer_state_container_sha256: str
    car_patient_score_sha256: str
    car_event_score_sha256: str
    final_score_reference_receipt_sha256: str
    event_reference_consistency_receipt_sha256: str
    aggregation_receipt_sha256: str
    later_visible_region_receipt_sha256: str | None = None
    identity_policy: str = IDENTITY_POLICY
    prediction_role: str = PREDICTION_ROLE
    car_scores_changed_by_assembler: bool = False
    selective_threshold_defined: bool = False
    abstain: bool = True
    deepsoz_target_values_loaded: bool = False
    private_data_loaded: bool = False
    localization_scores_used_by_event_producer: bool = False
    training_performed: bool = False
    schema_version: str = ASSEMBLER_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "patient_id",
            "local_patient_id",
            "event_id",
            "relative_edf_path",
        ):
            _require_text(getattr(self, name), name=name)
        if (
            not math.isfinite(self.global_t0_sec)
            or self.global_t0_sec < 0
            or not math.isfinite(self.global_stop_sec)
            or self.global_stop_sec <= self.global_t0_sec
        ):
            raise TargetFreeOOFReportAssemblyError("global event interval is invalid")
        if self.global_event_index < 0:
            raise TargetFreeOOFReportAssemblyError("global_event_index is invalid")
        for name in (
            "edf_sha256",
            "edf_receipt_sha256",
            "event_record_sha256",
            "signal_receipt_sha256",
            "processed_window_sha256",
            "phenotype_artifact_sha256",
            "mrsc_manifest_sha256",
            "mrsc_tensor_sha256",
            "outer_state_container_sha256",
            "car_patient_score_sha256",
            "car_event_score_sha256",
            "final_score_reference_receipt_sha256",
            "event_reference_consistency_receipt_sha256",
            "aggregation_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if self.later_visible_region_receipt_sha256 is not None:
            _require_sha256(
                self.later_visible_region_receipt_sha256,
                name="later_visible_region_receipt_sha256",
            )
        if self.identity_policy != IDENTITY_POLICY or self.prediction_role != PREDICTION_ROLE:
            raise TargetFreeOOFReportAssemblyError("assembly policy changed")
        for name in (
            "car_scores_changed_by_assembler",
            "selective_threshold_defined",
            "deepsoz_target_values_loaded",
            "private_data_loaded",
            "localization_scores_used_by_event_producer",
            "training_performed",
        ):
            if getattr(self, name) is not False:
                raise UnsafeTargetFreeArtifactError(f"{name} must remain false")
        if self.abstain is not True:
            raise UnsafeTargetFreeArtifactError("uncalibrated v3 ranking must abstain")
        if self.schema_version != ASSEMBLER_SCHEMA:
            raise TargetFreeOOFReportAssemblyError("assembly receipt schema changed")


@dataclass(frozen=True)
class AssembledTargetFreeOOFReportV3:
    patient_id: str
    local_patient_id: str
    event_id: str
    typed_facts: ClinicalReportFactsV2
    report: GroundedChineseDiagnosticReport
    assembly_receipt: TargetFreeOOFAssemblyReceiptV3
    status: str = "assembled_abstained_target_free_oof_draft"

    def __post_init__(self) -> None:
        receipt = self.assembly_receipt
        if not isinstance(receipt, TargetFreeOOFAssemblyReceiptV3):
            raise TypeError("assembly_receipt has the wrong type")
        if not isinstance(self.typed_facts, ClinicalReportFactsV2):
            raise TypeError("typed_facts has the wrong type")
        if not isinstance(self.report, GroundedChineseDiagnosticReport):
            raise TypeError("report has the wrong type")
        expected = (
            (self.patient_id, receipt.patient_id),
            (self.local_patient_id, receipt.local_patient_id),
            (self.event_id, receipt.event_id),
            (self.typed_facts.patient_ranking.patient_pseudonym, self.patient_id),
            (self.typed_facts.event_phenotype.receipt.event_pseudonym, self.event_id),
            (
                self.typed_facts.final_score_reference_disagreement_receipt.receipt_sha256,
                receipt.final_score_reference_receipt_sha256,
            ),
            (
                self.typed_facts.event_reference_consistency_receipt.receipt_sha256,
                receipt.event_reference_consistency_receipt_sha256,
            ),
        )
        if any(actual != wanted for actual, wanted in expected):
            raise TargetFreeOOFReportAssemblyError("assembled identity/receipt mismatch")
        uncertainty = self.typed_facts.patient_ranking.uncertainty
        if not uncertainty.abstain or uncertainty.abstention_reason_codes != (
            "selective_threshold_undefined",
        ):
            raise UnsafeTargetFreeArtifactError("v3 must preserve MRSC abstention")
        if "abstained" not in self.report.report_status:
            raise UnsafeTargetFreeArtifactError("rendered v3 report must abstain")
        if "SOZ-reference首位候选" not in self.report.text and (
            "SOZ-reference并列首位候选" not in self.report.text
        ):
            raise UnsafeTargetFreeArtifactError(
                "physical-electrode candidate wording is not scoped to SOZ-reference"
            )
        event = self.typed_facts.event_phenotype
        if isinstance(event, EventScalpPhenotypeEvidence):
            if "自记录起" not in self.report.text:
                raise UnsafeTargetFreeArtifactError(
                    "event timing lacks an explicit recording-start origin"
                )
            if (
                event.artifact_assessed is not None
                or event.artifact_types
                or event.artifact_burden is not None
            ):
                raise UnsafeTargetFreeArtifactError("artifact fact escaped empty policy")
        if any(claim in self.report.text for claim in _FORBIDDEN_CORTICAL_CLAIMS):
            raise UnsafeTargetFreeArtifactError("forbidden cortical/surgical claim")
        if self.status != "assembled_abstained_target_free_oof_draft":
            raise TargetFreeOOFReportAssemblyError("assembled status changed")


@dataclass(frozen=True)
class BlockedTargetFreeOOFReportV3:
    patient_id: str
    local_patient_id: str
    event_id: str
    reason_codes: tuple[str, ...]
    status: str = "blocked_fail_closed"

    def __post_init__(self) -> None:
        for name in ("patient_id", "local_patient_id", "event_id"):
            _require_text(getattr(self, name), name=name)
        if not self.reason_codes or len(set(self.reason_codes)) != len(
            self.reason_codes
        ) or any(_REASON_RE.fullmatch(code) is None for code in self.reason_codes):
            raise TargetFreeOOFReportAssemblyError("blocked reasons are invalid")


@dataclass(frozen=True)
class TargetFreeOOFReportBatchV3:
    records: tuple[
        AssembledTargetFreeOOFReportV3 | BlockedTargetFreeOOFReportV3, ...
    ]
    phenotype_event_count: int
    mrsc_event_count: int
    phenotype_artifact_sha256: str
    mrsc_manifest_sha256: str
    mrsc_tensor_sha256: str
    car_patient_score_tensor_sha256_before: str
    car_patient_score_tensor_sha256_after: str
    car_event_score_tensor_sha256_before: str
    car_event_score_tensor_sha256_after: str
    status: str = ASSEMBLER_STATUS
    schema_version: str = ASSEMBLER_SCHEMA

    def __post_init__(self) -> None:
        if not self.records:
            raise TargetFreeOOFReportAssemblyError("report batch cannot be empty")
        identities = tuple(record.event_id for record in self.records)
        if len(set(identities)) != len(identities):
            raise TargetFreeOOFReportAssemblyError("report event appears more than once")
        if self.phenotype_event_count < 1 or self.mrsc_event_count < 1:
            raise TargetFreeOOFReportAssemblyError("input event counts are invalid")
        for name in (
            "phenotype_artifact_sha256",
            "mrsc_manifest_sha256",
            "mrsc_tensor_sha256",
            "car_patient_score_tensor_sha256_before",
            "car_patient_score_tensor_sha256_after",
            "car_event_score_tensor_sha256_before",
            "car_event_score_tensor_sha256_after",
        ):
            _require_sha256(getattr(self, name), name=name)
        if self.car_patient_score_tensor_sha256_before != (
            self.car_patient_score_tensor_sha256_after
        ) or self.car_event_score_tensor_sha256_before != (
            self.car_event_score_tensor_sha256_after
        ):
            raise UnsafeTargetFreeArtifactError("assembler changed preserved CAR scores")
        if self.status != ASSEMBLER_STATUS or self.schema_version != ASSEMBLER_SCHEMA:
            raise TargetFreeOOFReportAssemblyError("batch contract changed")

    @property
    def assembled_count(self) -> int:
        return sum(
            isinstance(record, AssembledTargetFreeOOFReportV3)
            for record in self.records
        )

    @property
    def blocked_count(self) -> int:
        return len(self.records) - self.assembled_count

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "scientific_boundary": {
                "prediction_role": PREDICTION_ROLE,
                "evaluation_eligible": False,
                "clinical_deployment_eligible": False,
                "all_patient_rankings_abstain": True,
                "selective_threshold_defined": False,
                "physical_electrode_wording": "SOZ-reference candidate only",
                "cortical_soz_claim_allowed": False,
                "artifact_fact_available": False,
                "signal_identity_cross_checked_to_mrsc": False,
                "signal_identity_scope": (
                    "phenotype_internal_edf_processed_window_and_typed_receipts"
                ),
            },
            "access_receipt": {
                "deepsoz_target_values_loaded": False,
                "private_eeg_loaded": False,
                "private_target_values_loaded": False,
                "localization_scores_used_by_event_producer": False,
                "raw_eeg_loaded_by_assembler": False,
                "training_performed": False,
                "model_selection_performed": False,
                "calibration_performed": False,
            },
            "counts": {
                "phenotype_events": self.phenotype_event_count,
                "mrsc_events": self.mrsc_event_count,
                "output_records": len(self.records),
                "assembled": self.assembled_count,
                "blocked": self.blocked_count,
            },
            "score_parity": {
                "car_patient_score_tensor_sha256_before": (
                    self.car_patient_score_tensor_sha256_before
                ),
                "car_patient_score_tensor_sha256_after": (
                    self.car_patient_score_tensor_sha256_after
                ),
                "car_event_score_tensor_sha256_before": (
                    self.car_event_score_tensor_sha256_before
                ),
                "car_event_score_tensor_sha256_after": (
                    self.car_event_score_tensor_sha256_after
                ),
                "patient_scores_elementwise_unchanged": True,
                "event_scores_elementwise_unchanged": True,
            },
            "input_artifact_sha256s": {
                "phenotype": self.phenotype_artifact_sha256,
                "mrsc_manifest": self.mrsc_manifest_sha256,
                "mrsc_tensor": self.mrsc_tensor_sha256,
            },
            "records": [asdict(record) for record in self.records],
        }


def _blocked_from_raw(
    row: Mapping[str, object], reason: str
) -> BlockedTargetFreeOOFReportV3:
    patient = row.get("patient_id")
    local = row.get("local_patient_id")
    event = row.get("event_id")
    return BlockedTargetFreeOOFReportV3(
        patient_id=patient if isinstance(patient, str) and patient else "unknown",
        local_patient_id=local if isinstance(local, str) and local else "unknown",
        event_id=event if isinstance(event, str) and event else _canonical_sha256(row),
        reason_codes=(reason,),
    )


def _render_v3(facts: ClinicalReportFactsV2) -> GroundedChineseDiagnosticReport:
    report = render_grounded_chinese_diagnostic_report(facts)
    replacements = (
        ("当前并列首位候选为", "当前SOZ-reference并列首位候选为"),
        ("当前首位候选为", "当前SOZ-reference首位候选为"),
        (
            "患者级排序仅是临床参考头皮电极SOZ假设",
            "患者级排序仅为头皮物理电极SOZ-reference候选排序",
        ),
        (
            "checkpoint_sha256=",
            "oof_outer_state_container_sha256=",
        ),
    )

    def scoped(text: str) -> str:
        for source, target in replacements:
            text = text.replace(source, target)
        return text

    return replace(
        report,
        text=scoped(report.text),
        localization_phrase=scoped(report.localization_phrase),
        patient_ranking_phrase=scoped(report.patient_ranking_phrase),
        limitation_phrase=scoped(report.limitation_phrase),
        model_identity=scoped(report.model_identity),
    )


def _assemble_event(
    event: _PhenotypeEvent,
    *,
    mrsc: _MRSCBundle,
    phenotype_artifact_sha256: str,
    outer_state_container_sha256: str,
) -> AssembledTargetFreeOOFReportV3:
    event_index = mrsc.event_to_index[event.event_id]
    patient_index = int(mrsc.event_patient_index[event_index])
    expected_patient = mrsc.patient_ids[patient_index]
    if event.patient_id != expected_patient:
        raise TargetFreeOOFReportAssemblyError(
            "phenotype patient and MRSC event routing disagree"
        )
    patient_events = tuple(
        mrsc.event_ids[index]
        for index in torch.nonzero(
            mrsc.event_patient_index == patient_index, as_tuple=False
        )
        .flatten()
        .tolist()
    )
    final_reference = load_final_score_reference_disagreement_receipt(
        mrsc.directory,
        patient_pseudonym=event.patient_id,
        aggregation_event_ids=patient_events,
        expected_source_manifest_sha256=mrsc.manifest_sha256,
        expected_source_tensor_sha256=mrsc.tensor_sha256,
    )
    source_patient_scores = mrsc.car_patient_scores[patient_index]
    receipted_patient_scores = torch.tensor(
        [row.primary_score for row in final_reference.candidate_score_summary],
        dtype=torch.float32,
    )
    if not torch.equal(source_patient_scores, receipted_patient_scores):
        raise UnsafeTargetFreeArtifactError(
            "final-score receipt changed C-CAR19 patient scores"
        )
    source_event_score_sha = _tensor_sha256(mrsc.car_event_scores[event_index])
    event_score_receipts = dict(final_reference.primary_event_score_sha256s)
    if event_score_receipts.get(event.event_id) != source_event_score_sha:
        raise UnsafeTargetFreeArtifactError(
            "final-score receipt changed C-CAR19 event scores"
        )
    if not math.isclose(
        float(mrsc.final_score_reference_disagreement[patient_index]),
        final_reference.final_score_reference_disagreement,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise UnsafeTargetFreeArtifactError(
            "MRSC and final-score receipt disagreement metric differ"
        )

    full_scores = torch.zeros(N_STANDARD_CHANNELS, dtype=torch.float32)
    candidate_mask = torch.zeros(N_STANDARD_CHANNELS, dtype=torch.bool)
    for candidate_index, channel in enumerate(MRSC_CANDIDATE_CHANNELS):
        standard_index = CHANNEL_INDEX[channel]
        full_scores[standard_index] = source_patient_scores[candidate_index]
        candidate_mask[standard_index] = True
    spatial = derive_spatial_report(
        full_scores,
        candidate_mask,
        score_semantics="uncalibrated_localization_score",
    )
    spatial = replace(
        spatial,
        claim_boundary=(
            "This is a scalp-electrode SOZ-reference candidate ranking; it is "
            "not an invasive cortical SOZ, epileptogenic zone, or treatment target."
        ),
    )
    if final_reference.primary_top1_channel not in spatial.top_channels:
        raise UnsafeTargetFreeArtifactError("preserved CAR Top-1 changed")
    aggregation_receipt = _canonical_sha256(
        {
            "patient_id": event.patient_id,
            "aggregation_method": "patient_equal_event_mean",
            "aggregation_event_ids": patient_events,
            "car_patient_score_sha256": _tensor_sha256(source_patient_scores),
            "car_event_score_sha256s": final_reference.primary_event_score_sha256s,
            "mrsc_manifest_sha256": mrsc.manifest_sha256,
            "mrsc_tensor_sha256": mrsc.tensor_sha256,
        }
    )
    event_dispersion = (
        float(mrsc.event_dispersion[patient_index])
        if bool(mrsc.event_dispersion_estimable[patient_index])
        else None
    )
    ranking = PatientSOZReferenceRanking(
        spatial_report=spatial,
        patient_pseudonym=event.patient_id,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        # This field binds the five-fold OOF outer-state container.  It is
        # intentionally not presented as a single full-refit checkpoint.
        model_checkpoint_sha256=outer_state_container_sha256,
        aggregation_method="patient_equal_event_mean",
        aggregation_event_count=len(patient_events),
        aggregation_event_ids=patient_events,
        aggregation_receipt_sha256=aggregation_receipt,
        uncertainty=UncertaintyDecomposition(
            ranking_ambiguity=float(mrsc.ranking_ambiguity[patient_index]),
            within_patient_event_dispersion=event_dispersion,
            signal_quality_uncertainty=None,
            montage_disagreement=None,
            final_score_reference_disagreement=None,
            epistemic_uncertainty=None,
            abstain=True,
            abstention_reason_codes=("selective_threshold_undefined",),
        ),
    )
    prebound_later = (
        isinstance(event.event, EventScalpPhenotypeEvidence)
        and event.event.later_visible_region_zh is not None
    )
    facts = ClinicalReportFactsV2(
        event_phenotype=event.event,
        patient_ranking=ranking,
        later_visible_region_receipt=(
            event.later_visible_region_receipt if prebound_later else None
        ),
        event_reference_consistency_receipt=event.event_reference_receipt,
    )
    if event.later_visible_region_receipt is not None and not prebound_later:
        facts = attach_later_visible_region_to_clinical_facts(
            facts, event.later_visible_region_receipt
        )
    facts = attach_final_score_reference_disagreement_to_clinical_facts(
        facts, final_reference
    )
    report = _render_v3(facts)
    receipt = TargetFreeOOFAssemblyReceiptV3(
        patient_id=event.patient_id,
        local_patient_id=event.local_patient_id,
        event_id=event.event_id,
        relative_edf_path=event.relative_edf_path,
        global_t0_sec=event.global_t0_sec,
        global_stop_sec=event.global_stop_sec,
        global_event_index=event.global_event_index,
        edf_sha256=event.edf_sha256,
        edf_receipt_sha256=event.edf_receipt_sha256,
        event_record_sha256=event.event_record_sha256,
        signal_receipt_sha256=event.signal_receipt_sha256,
        processed_window_sha256=event.processed_window_sha256,
        phenotype_artifact_sha256=phenotype_artifact_sha256,
        mrsc_manifest_sha256=mrsc.manifest_sha256,
        mrsc_tensor_sha256=mrsc.tensor_sha256,
        outer_state_container_sha256=outer_state_container_sha256,
        car_patient_score_sha256=_tensor_sha256(source_patient_scores),
        car_event_score_sha256=source_event_score_sha,
        final_score_reference_receipt_sha256=final_reference.receipt_sha256,
        event_reference_consistency_receipt_sha256=(
            event.event_reference_receipt.receipt_sha256
        ),
        aggregation_receipt_sha256=aggregation_receipt,
        later_visible_region_receipt_sha256=(
            None
            if event.later_visible_region_receipt is None
            else event.later_visible_region_receipt.receipt_sha256
        ),
    )
    return AssembledTargetFreeOOFReportV3(
        patient_id=event.patient_id,
        local_patient_id=event.local_patient_id,
        event_id=event.event_id,
        typed_facts=facts,
        report=report,
        assembly_receipt=receipt,
    )


def assemble_target_free_oof_reports_v3(
    *,
    mrsc_directory: str | Path,
    expected_mrsc_manifest_sha256: str,
    expected_mrsc_tensor_sha256: str,
    phenotype_artifact: str | Path,
    expected_phenotype_sha256: str,
    outer_state_container_sha256: str,
) -> TargetFreeOOFReportBatchV3:
    """Assemble target-free OOF reports without fitting, scoring, or calibration."""

    outer_state_sha = _require_sha256(
        outer_state_container_sha256, name="outer_state_container_sha256"
    )
    mrsc = _load_mrsc_bundle(
        mrsc_directory,
        expected_manifest_sha256=expected_mrsc_manifest_sha256,
        expected_tensor_sha256=expected_mrsc_tensor_sha256,
    )
    phenotype = _load_phenotype_bundle(
        phenotype_artifact,
        expected_sha256=expected_phenotype_sha256,
    )
    patient_scores_before = _tensor_sha256(mrsc.car_patient_scores)
    event_scores_before = _tensor_sha256(mrsc.car_event_scores)

    raw_by_event = {str(row["event_id"]): row for row in phenotype.raw_events}
    records: list[
        AssembledTargetFreeOOFReportV3 | BlockedTargetFreeOOFReportV3
    ] = []
    consumed: set[str] = set()
    for event_index, event_id in enumerate(mrsc.event_ids):
        raw = raw_by_event.get(event_id)
        patient_id = mrsc.patient_ids[int(mrsc.event_patient_index[event_index])]
        local_id = event_id.split("_", 1)[0]
        if raw is None:
            records.append(
                BlockedTargetFreeOOFReportV3(
                    patient_id=patient_id,
                    local_patient_id=local_id,
                    event_id=event_id,
                    reason_codes=("phenotype_event_missing",),
                )
            )
            continue
        consumed.add(event_id)
        try:
            parsed = _parse_phenotype_event(raw)
        except (TargetFreeOOFReportAssemblyError, TypeError, ValueError):
            records.append(_blocked_from_raw(raw, "phenotype_contract_invalid"))
            continue
        if parsed.patient_id != patient_id:
            records.append(
                BlockedTargetFreeOOFReportV3(
                    patient_id=parsed.patient_id,
                    local_patient_id=parsed.local_patient_id,
                    event_id=parsed.event_id,
                    reason_codes=("mrsc_patient_event_identity_mismatch",),
                )
            )
            continue
        try:
            records.append(
                _assemble_event(
                    parsed,
                    mrsc=mrsc,
                    phenotype_artifact_sha256=phenotype.artifact_sha256,
                    outer_state_container_sha256=outer_state_sha,
                )
            )
        except (TargetFreeOOFReportAssemblyError, TypeError, ValueError):
            records.append(
                BlockedTargetFreeOOFReportV3(
                    patient_id=parsed.patient_id,
                    local_patient_id=parsed.local_patient_id,
                    event_id=parsed.event_id,
                    reason_codes=("typed_receipt_or_score_binding_failed",),
                )
            )

    for event_id, raw in raw_by_event.items():
        if event_id in consumed:
            continue
        try:
            parsed = _parse_phenotype_event(raw)
            records.append(
                BlockedTargetFreeOOFReportV3(
                    patient_id=parsed.patient_id,
                    local_patient_id=parsed.local_patient_id,
                    event_id=parsed.event_id,
                    reason_codes=("mrsc_anchor_identity_not_available",),
                )
            )
        except (TargetFreeOOFReportAssemblyError, TypeError, ValueError):
            records.append(_blocked_from_raw(raw, "phenotype_contract_invalid"))

    patient_scores_after = _tensor_sha256(mrsc.car_patient_scores)
    event_scores_after = _tensor_sha256(mrsc.car_event_scores)
    if _file_sha256(mrsc.directory / SOURCE_TENSOR_FILENAME) != mrsc.tensor_sha256:
        raise UnsafeTargetFreeArtifactError("MRSC tensor changed during assembly")
    return TargetFreeOOFReportBatchV3(
        records=tuple(records),
        phenotype_event_count=len(phenotype.raw_events),
        mrsc_event_count=len(mrsc.event_ids),
        phenotype_artifact_sha256=phenotype.artifact_sha256,
        mrsc_manifest_sha256=mrsc.manifest_sha256,
        mrsc_tensor_sha256=mrsc.tensor_sha256,
        car_patient_score_tensor_sha256_before=patient_scores_before,
        car_patient_score_tensor_sha256_after=patient_scores_after,
        car_event_score_tensor_sha256_before=event_scores_before,
        car_event_score_tensor_sha256_after=event_scores_after,
    )


def write_target_free_oof_report_batch_v3(
    batch: TargetFreeOOFReportBatchV3,
    output: str | Path,
) -> str:
    """Atomically publish one canonical JSON batch and refuse overwrite."""

    if not isinstance(batch, TargetFreeOOFReportBatchV3):
        raise TypeError("batch must be TargetFreeOOFReportBatchV3")
    target = Path(os.path.abspath(output))
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    raw = _canonical_bytes(batch.to_payload()) + b"\n"
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-", dir=target.parent
    )
    os.close(descriptor)
    staging = Path(staging_name)
    published = False
    try:
        staging.write_bytes(raw)
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            staging.unlink()
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "ASSEMBLER_SCHEMA",
    "ASSEMBLER_STATUS",
    "AssembledTargetFreeOOFReportV3",
    "BlockedTargetFreeOOFReportV3",
    "PHENOTYPE_SCHEMA",
    "PHENOTYPE_STATUS",
    "TargetFreeOOFAssemblyReceiptV3",
    "TargetFreeOOFReportAssemblyError",
    "TargetFreeOOFReportBatchV3",
    "UnsafeTargetFreeArtifactError",
    "assemble_target_free_oof_reports_v3",
    "write_target_free_oof_report_batch_v3",
]
