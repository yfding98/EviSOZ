"""Read-only, fail-closed assembly of grounded public-data reports.

The assembler is intentionally narrow.  It accepts only the sealed,
target-excluding ``temporal_mil_exact`` source-train refit artifact and joins
it to target-free event-phenotype, public signal-roster, and reference-audit
artifacts.  Historical OOF payloads that co-serialize targets are rejected
before any tensor is read.

The accepted ranking payload is a source-train *resubstitution fit-sanity*
artifact, not an evaluation artifact.  Consequently every assembled patient
ranking abstains and is labelled for manual review.  This module does not
read raw EEG, DeepSOZ target values, TUSZ channel-annotation values, or any
private signal/annotation artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Mapping, Sequence

import torch

from .clinical_reporting import (
    ClinicalReportFactsV2,
    EvidenceProvenanceReceipt,
    EventScalpPhenotypeEvidence,
    GroundedChineseDiagnosticReport,
    PatientSOZReferenceRanking,
    UncertaintyDecomposition,
    derive_spatial_report,
    render_grounded_chinese_diagnostic_report,
)
from .geometry import N_STANDARD_CHANNELS
from .later_visible_region_producer import (
    build_later_visible_region_receipt,
    produce_later_visible_region,
)
from .later_visible_region_reporting import (
    attach_later_visible_region_to_clinical_facts,
)
from .reference_disagreement import (
    REFERENCE_AUDIT_SCHEMA,
    REFERENCE_AUDIT_STATUS,
    build_reference_disagreement_receipt,
)
from .reference_reporting import attach_reference_disagreement_to_clinical_facts


PUBLIC_REPORT_ASSEMBLER_SCHEMA = "soz_public_report_assembler_v1"
PUBLIC_REPORT_BATCH_STATUS = (
    "public_source_train_resubstitution_audit_only_not_evaluation"
)
SAFE_RANKING_SCHEMA = "soz_labram_temporal_mil_exact_full_source_train_refit_v1"
SAFE_RANKING_STATUS = "completed_full_source_train_exact_refit"
SAFE_RANKING_CANDIDATE = "temporal_mil_exact"
SAFE_PREDICTION_ROLE = "source_train_resubstitution_fit_sanity_only"
SAFE_PREDICTION_FILENAME = "source_train_resubstitution_predictions.safetensors"
SAFE_CHECKPOINT_FILENAME = "final_checkpoint.safetensors"
SAFE_RANKING_KEYS = frozenset(
    {
        "event_logits",
        "patient_logits",
        "temporal_weights",
        "ictal_contribution",
        "evolution_contribution",
        "event_patient_index",
    }
)
UNSAFE_COLOCATED_TARGET_RANKING_SCHEMAS = frozenset(
    {
        "soz_labram_evidence_temporal_mil_recovery_v1",
        "soz_labram_robust_temporal_mil_direct_oof_v14",
    }
)
SOURCE_TRAIN_CAPABILITY_SCHEMA = "soz_source_train_only_iv_capability_v1"
SOURCE_TRAIN_EVENT_ROSTER_SCHEMA = "soz_source_train_only_iv_event_roster_v1"
PUBLIC_UNION_SCHEMA = "soz_public_development_union_v11"
PHENOTYPE_AUDIT_SCHEMA = "soz_event_phenotype_source_only_audit_v1"
PHENOTYPE_AUDIT_STATUS = "target_free_source_only_descriptive_audit"
PHENOTYPE_PRODUCER_SCHEMA = "soz_target_free_event_scalp_phenotype_producer_v1"
PHENOTYPE_PRODUCER_POLICY = (
    "target_free_sustained_bipolar_change_then_local_spectral_gate_v1"
)
IDENTITY_BINDING_POLICY = (
    "exact_relative_edf_path_global_t0_global_event_index_and_signal_sha256_v1"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GLOBAL_EVENT_RE = re.compile(
    r"^(?P<stem>[a-z0-9]+_s\d+_t\d+)__global_ictal_(?P<index>\d{4})$"
)
_CANONICAL_EVENT_RE = re.compile(
    r"^(?P<stem>[a-z0-9]+_s\d+_t\d+)__ev(?P<index>\d{4})$"
)
_FORBIDDEN_REPORT_CLAIMS = (
    "首先出现",
    "最早可见",
    "最早物理电极",
    "皮层SOZ可疑位于",
    "皮层SOZ位于",
)


class PublicReportAssemblyError(ValueError):
    """Base class for an assembly contract failure."""


class UnsafeArtifactError(PublicReportAssemblyError):
    """Raised before a target-bearing or otherwise unsafe artifact is read."""


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


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PublicReportAssemblyError(f"{name} must be a lowercase SHA-256")
    return value


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicReportAssemblyError(f"{name} must be non-empty text")
    return value.strip()


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PublicReportAssemblyError(f"{name} must be a mapping")
    return value


def _require_sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PublicReportAssemblyError(f"{name} must be a sequence")
    return value


def _require_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise PublicReportAssemblyError(f"{name} must be bool")
    return value


def _canonical_regular_file(path: str | Path, *, name: str) -> Path:
    lexical = Path(os.path.abspath(path))
    for component in (lexical, *lexical.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise UnsafeArtifactError(f"{name} cannot traverse symlinks")
    try:
        resolved = lexical.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PublicReportAssemblyError(f"{name} does not exist") from exc
    if not resolved.is_file():
        raise PublicReportAssemblyError(f"{name} must be a regular file")
    return resolved


def _canonical_directory(path: str | Path, *, name: str) -> Path:
    lexical = Path(os.path.abspath(path))
    for component in (lexical, *lexical.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise UnsafeArtifactError(f"{name} cannot traverse symlinks")
    try:
        resolved = lexical.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PublicReportAssemblyError(f"{name} does not exist") from exc
    if not resolved.is_dir():
        raise PublicReportAssemblyError(f"{name} must be a directory")
    return resolved


def _load_json_file(
    path: str | Path, *, name: str
) -> tuple[Path, bytes, Mapping[str, object]]:
    source = _canonical_regular_file(path, name=name)
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicReportAssemblyError(f"{name} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise PublicReportAssemblyError(f"{name} must contain one JSON object")
    return source, raw, value


def _require_file_receipt(
    directory: Path,
    files: Mapping[str, object],
    filename: str,
    *,
    inspect_safetensor_header_first: bool = False,
) -> tuple[Path, str]:
    row = _require_mapping(files.get(filename), name=f"files.{filename}")
    expected_sha = _require_sha256(row.get("sha256"), name=f"{filename}.sha256")
    expected_size = row.get("size_bytes")
    if type(expected_size) is not int or expected_size < 1:
        raise PublicReportAssemblyError(f"{filename}.size_bytes is invalid")
    source = _canonical_regular_file(directory / filename, name=filename)
    if source.parent != directory:
        raise UnsafeArtifactError(f"{filename} escaped its sealed directory")
    if source.stat().st_size != expected_size:
        raise UnsafeArtifactError(f"{filename} size disagrees with its manifest")
    if inspect_safetensor_header_first:
        _inspect_safe_prediction_header(source)
    actual_sha = _file_sha256(source)
    if actual_sha != expected_sha:
        raise UnsafeArtifactError(f"{filename} digest disagrees with its manifest")
    return source, actual_sha


def _inspect_safe_prediction_header(path: Path) -> tuple[str, ...]:
    """Reject co-located target tensors before reading any tensor value."""

    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise RuntimeError("safetensors is required for report assembly") from exc
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = tuple(handle.keys())
    key_set = set(keys)
    target_like = tuple(
        sorted(
            key
            for key in keys
            if "target" in key.lower() or key.lower().endswith("_mask")
        )
    )
    if target_like:
        raise UnsafeArtifactError(
            "Ranking tensor artifact co-serializes target/mask tensors and is "
            f"blocked before tensor loading: {','.join(target_like)}"
        )
    if key_set != set(SAFE_RANKING_KEYS):
        raise UnsafeArtifactError(
            "Ranking tensor artifact does not have the sealed target-excluding schema"
        )
    return keys


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


def _validate_relative_edf(value: object) -> str:
    text = _require_text(value, name="relative_edf_path")
    relative = PurePosixPath(text)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".edf":
        raise PublicReportAssemblyError("relative_edf_path is unsafe")
    if len(relative.parts) < 3:
        raise PublicReportAssemblyError("relative_edf_path lacks patient identity")
    return relative.as_posix()


def _parse_evidence_receipt(value: object) -> EvidenceProvenanceReceipt:
    row = _require_mapping(value, name="phenotype.receipt")
    expected = {field.name for field in fields(EvidenceProvenanceReceipt)}
    if set(row) != expected:
        raise UnsafeArtifactError("event-evidence receipt schema drifted")
    converted = dict(row)
    converted["montages"] = tuple(
        _require_sequence(converted["montages"], name="receipt.montages")
    )
    return EvidenceProvenanceReceipt(**converted)


def _parse_event_phenotype(value: object) -> EventScalpPhenotypeEvidence:
    row = _require_mapping(value, name="phenotype")
    expected = {field.name for field in fields(EventScalpPhenotypeEvidence)}
    if set(row) != expected:
        raise UnsafeArtifactError("event phenotype schema drifted")
    converted = dict(row)
    converted["receipt"] = _parse_evidence_receipt(converted["receipt"])
    for name in (
        "first_visible_derivations",
        "later_visible_derivations",
        "artifact_types",
    ):
        converted[name] = tuple(_require_sequence(converted[name], name=name))
    frequency = converted["frequency_range_hz"]
    if frequency is not None:
        converted["frequency_range_hz"] = tuple(
            _require_sequence(frequency, name="frequency_range_hz")
        )
    event = EventScalpPhenotypeEvidence(**converted)
    # The sealed v1 producer has no validated artifact, regional-destination,
    # or paired-reference producer.  Those optional typed slots are useful for
    # later schemas, but accepting them from this historical artifact would
    # launder manually inserted facts into a clinical sentence.
    receipt = event.receipt
    expected_receipt_contract = (
        (receipt.extractor_model_id, "target-free-fine-temporal-evidence"),
        (receipt.extractor_model_version, PHENOTYPE_PRODUCER_SCHEMA),
        (receipt.evidence_generation_policy, PHENOTYPE_PRODUCER_POLICY),
        (receipt.montages, ("C-CAR19",)),
    )
    if any(actual != expected for actual, expected in expected_receipt_contract):
        raise UnsafeArtifactError("event phenotype receipt violates v1 producer contract")
    unvalidated_slots = (
        event.later_visible_region_zh is not None,
        event.montage_stability is not None,
        event.artifact_assessed is not None,
        bool(event.artifact_types),
        event.artifact_burden is not None,
    )
    if any(unvalidated_slots):
        raise UnsafeArtifactError(
            "event phenotype v1 populated a fact slot without a validated producer"
        )
    return event


@dataclass(frozen=True)
class PublicReportAssemblyReceipt:
    """Lineage proving an exact source-event to ranking-event identity join."""

    source_patient_pseudonym: str
    source_event_pseudonym: str
    canonical_patient_pseudonym: str
    canonical_event_pseudonym: str
    relative_edf_path: str
    global_t0_sec: float
    global_event_index: int
    signal_artifact_sha256: str
    source_event_receipt_sha256: str
    canonical_pre_reference_event_receipt_sha256: str
    phenotype_artifact_sha256: str
    public_union_manifest_sha256: str
    ranking_manifest_sha256: str
    ranking_prediction_sha256: str
    ranking_checkpoint_sha256: str
    ranking_roster_manifest_sha256: str
    ranking_event_roster_sha256: str
    reference_audit_artifact_sha256: str
    reference_disagreement_receipt_sha256: str
    patient_score_tensor_sha256: str
    aggregation_receipt_sha256: str
    later_visible_region_receipt_sha256: str | None = None
    identity_binding_policy: str = IDENTITY_BINDING_POLICY
    prediction_role: str = SAFE_PREDICTION_ROLE
    evaluation_eligible: bool = False
    deepsoz_target_values_loaded_by_assembler: bool = False
    tusz_channel_target_values_loaded_by_assembler: bool = False
    private_data_loaded_by_assembler: bool = False
    schema_version: str = PUBLIC_REPORT_ASSEMBLER_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "source_patient_pseudonym",
            "source_event_pseudonym",
            "canonical_patient_pseudonym",
            "canonical_event_pseudonym",
            "relative_edf_path",
        ):
            _require_text(getattr(self, name), name=name)
        if not isinstance(self.global_t0_sec, (int, float)) or not math.isfinite(
            float(self.global_t0_sec)
        ):
            raise PublicReportAssemblyError("global_t0_sec must be finite")
        if type(self.global_event_index) is not int or self.global_event_index < 0:
            raise PublicReportAssemblyError("global_event_index is invalid")
        for name in (
            "signal_artifact_sha256",
            "source_event_receipt_sha256",
            "canonical_pre_reference_event_receipt_sha256",
            "phenotype_artifact_sha256",
            "public_union_manifest_sha256",
            "ranking_manifest_sha256",
            "ranking_prediction_sha256",
            "ranking_checkpoint_sha256",
            "ranking_roster_manifest_sha256",
            "ranking_event_roster_sha256",
            "reference_audit_artifact_sha256",
            "reference_disagreement_receipt_sha256",
            "patient_score_tensor_sha256",
            "aggregation_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if self.later_visible_region_receipt_sha256 is not None:
            _require_sha256(
                self.later_visible_region_receipt_sha256,
                name="later_visible_region_receipt_sha256",
            )
        if self.identity_binding_policy != IDENTITY_BINDING_POLICY:
            raise PublicReportAssemblyError("identity binding policy changed")
        if self.prediction_role != SAFE_PREDICTION_ROLE:
            raise PublicReportAssemblyError("unsafe prediction role")
        for name in (
            "evaluation_eligible",
            "deepsoz_target_values_loaded_by_assembler",
            "tusz_channel_target_values_loaded_by_assembler",
            "private_data_loaded_by_assembler",
        ):
            if type(getattr(self, name)) is not bool:
                raise PublicReportAssemblyError(f"{name} must be bool")
            if getattr(self, name):
                raise UnsafeArtifactError(f"{name} must remain false")
        if self.schema_version != PUBLIC_REPORT_ASSEMBLER_SCHEMA:
            raise PublicReportAssemblyError("unsupported assembly receipt schema")


@dataclass(frozen=True)
class AssembledPublicDiagnosticReport:
    source_patient_pseudonym: str
    source_event_pseudonym: str
    canonical_patient_pseudonym: str
    canonical_event_pseudonym: str
    typed_facts: ClinicalReportFactsV2
    report: GroundedChineseDiagnosticReport
    assembly_receipt: PublicReportAssemblyReceipt
    status: str = "assembled_abstained_public_resubstitution_draft"

    def __post_init__(self) -> None:
        if not isinstance(self.typed_facts, ClinicalReportFactsV2):
            raise TypeError("typed_facts must be ClinicalReportFactsV2")
        if not isinstance(self.report, GroundedChineseDiagnosticReport):
            raise TypeError("report must be GroundedChineseDiagnosticReport")
        if not isinstance(self.assembly_receipt, PublicReportAssemblyReceipt):
            raise TypeError("assembly_receipt must be PublicReportAssemblyReceipt")
        receipt = self.assembly_receipt
        expected = (
            (self.source_patient_pseudonym, receipt.source_patient_pseudonym),
            (self.source_event_pseudonym, receipt.source_event_pseudonym),
            (self.canonical_patient_pseudonym, receipt.canonical_patient_pseudonym),
            (self.canonical_event_pseudonym, receipt.canonical_event_pseudonym),
            (
                self.typed_facts.patient_ranking.patient_pseudonym,
                self.canonical_patient_pseudonym,
            ),
            (
                self.typed_facts.event_phenotype.receipt.event_pseudonym,
                self.canonical_event_pseudonym,
            ),
            (
                self.typed_facts.patient_ranking.aggregation_receipt_sha256,
                receipt.aggregation_receipt_sha256,
            ),
            (
                self.report.aggregation_receipt_sha256,
                receipt.aggregation_receipt_sha256,
            ),
            (
                self.report.reference_disagreement_receipt_sha256,
                receipt.reference_disagreement_receipt_sha256,
            ),
            (
                self.report.later_visible_region_receipt_sha256,
                receipt.later_visible_region_receipt_sha256,
            ),
        )
        if any(left != right for left, right in expected):
            raise PublicReportAssemblyError("assembled report identity/receipt mismatch")
        if not self.typed_facts.patient_ranking.uncertainty.abstain:
            raise UnsafeArtifactError("public resubstitution report must abstain")
        if "abstained" not in self.report.report_status:
            raise UnsafeArtifactError("rendered public resubstitution report must abstain")
        if self.status != "assembled_abstained_public_resubstitution_draft":
            raise PublicReportAssemblyError("assembled report status changed")


@dataclass(frozen=True)
class BlockedPublicDiagnosticReport:
    source_patient_pseudonym: str
    source_event_pseudonym: str
    reason_codes: tuple[str, ...]
    status: str = "blocked_fail_closed"

    def __post_init__(self) -> None:
        _require_text(self.source_patient_pseudonym, name="source_patient_pseudonym")
        _require_text(self.source_event_pseudonym, name="source_event_pseudonym")
        if (
            not self.reason_codes
            or len(set(self.reason_codes)) != len(self.reason_codes)
            or any(re.fullmatch(r"[a-z][a-z0-9_]*", value) is None for value in self.reason_codes)
        ):
            raise PublicReportAssemblyError("invalid blocked-report reason codes")


@dataclass(frozen=True)
class PublicReportAssemblyBatch:
    records: tuple[
        AssembledPublicDiagnosticReport | BlockedPublicDiagnosticReport, ...
    ]
    input_artifact_sha256s: tuple[tuple[str, str], ...]
    status: str = PUBLIC_REPORT_BATCH_STATUS
    schema_version: str = PUBLIC_REPORT_ASSEMBLER_SCHEMA

    def __post_init__(self) -> None:
        if self.status != PUBLIC_REPORT_BATCH_STATUS:
            raise PublicReportAssemblyError("public report batch status changed")
        if self.schema_version != PUBLIC_REPORT_ASSEMBLER_SCHEMA:
            raise PublicReportAssemblyError("public report batch schema changed")
        if any(
            not isinstance(
                record,
                (AssembledPublicDiagnosticReport, BlockedPublicDiagnosticReport),
            )
            for record in self.records
        ):
            raise TypeError("records contain an unsupported report type")
        identities = tuple(record.source_event_pseudonym for record in self.records)
        if len(set(identities)) != len(identities):
            raise PublicReportAssemblyError("source event appears more than once")
        if not self.records:
            raise PublicReportAssemblyError("report batch cannot be empty")
        names = tuple(name for name, _ in self.input_artifact_sha256s)
        if len(set(names)) != len(names):
            raise PublicReportAssemblyError("duplicate input artifact identity")
        for name, digest in self.input_artifact_sha256s:
            _require_text(name, name="input artifact name")
            _require_sha256(digest, name=f"{name}.sha256")

    @property
    def assembled_count(self) -> int:
        return sum(
            isinstance(record, AssembledPublicDiagnosticReport)
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
                "ranking_role": SAFE_PREDICTION_ROLE,
                "evaluation_eligible": False,
                "clinical_deployment_eligible": False,
                "all_patient_rankings_abstain": True,
                "event_wording": "fixed_algorithm_sustained_change_candidate",
                "cortical_soz_claim_allowed": False,
            },
            "access_receipt": {
                "deepsoz_target_values_loaded": False,
                "tusz_channel_target_values_loaded": False,
                "private_eeg_loaded": False,
                "private_target_values_loaded": False,
                "raw_eeg_loaded": False,
                "historical_target_colocated_oof_predictions_loaded": False,
                "target_excluding_resubstitution_predictions_loaded": True,
                "training_performed": False,
                "model_selection_performed": False,
            },
            "counts": {
                "input_event_count": len(self.records),
                "assembled": self.assembled_count,
                "blocked": self.blocked_count,
            },
            "input_artifact_sha256s": dict(self.input_artifact_sha256s),
            "records": [asdict(record) for record in self.records],
        }


@dataclass(frozen=True)
class _RankingBundle:
    patient_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    event_patient_ids: tuple[str, ...]
    event_patient_index: torch.Tensor
    event_logits: torch.Tensor
    patient_logits: torch.Tensor
    model_checkpoint_sha256: str
    ranking_manifest_sha256: str
    prediction_artifact_sha256: str
    roster_manifest_sha256: str
    roster_artifact_sha256: str


@dataclass(frozen=True)
class _PublicUnion:
    events_by_identity: Mapping[tuple[str, float, int], Mapping[str, object]]
    artifact_sha256: str


def _load_source_train_roster(
    directory: Path,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    str,
    str,
]:
    _, manifest_raw, manifest = _load_json_file(
        directory / "manifest.json", name="source-train capability manifest"
    )
    if manifest.get("schema_version") != SOURCE_TRAIN_CAPABILITY_SCHEMA:
        raise UnsafeArtifactError("unsupported source-train capability schema")
    expected_contract = {
        "model_split": "source_train",
        "source_train_only": True,
        "target_values_loaded": False,
        "private_used": False,
        "source_eval_used": False,
    }
    for name, expected in expected_contract.items():
        if manifest.get(name) != expected:
            raise UnsafeArtifactError(
                f"source-train capability contract mismatch: {name}"
            )
    files = _require_mapping(manifest.get("files"), name="capability.files")
    events_path, events_sha = _require_file_receipt(
        directory, files, "events.json"
    )
    _, events_raw, roster = _load_json_file(
        events_path, name="source-train event roster"
    )
    if hashlib.sha256(events_raw).hexdigest() != events_sha:
        raise UnsafeArtifactError("event roster changed between verification and read")
    if roster.get("schema_version") != SOURCE_TRAIN_EVENT_ROSTER_SCHEMA:
        raise UnsafeArtifactError("unsupported source-train event-roster schema")
    if roster.get("model_split") != "source_train":
        raise UnsafeArtifactError("event roster is not source_train-only")
    raw_events = _require_sequence(roster.get("events"), name="roster.events")
    event_ids: list[str] = []
    patient_ids: list[str] = []
    for raw_event in raw_events:
        event = _require_mapping(raw_event, name="roster.event")
        if set(event) != {"event_id", "oof_fold", "patient_id"}:
            raise UnsafeArtifactError("source-train event-roster row schema drifted")
        event_id = _require_text(event.get("event_id"), name="event_id")
        patient_id = _require_text(event.get("patient_id"), name="patient_id")
        if _CANONICAL_EVENT_RE.fullmatch(event_id) is None:
            raise PublicReportAssemblyError("ranking event ID is not canonical")
        fold = event.get("oof_fold")
        if type(fold) is not int or fold not in range(5):
            raise PublicReportAssemblyError("ranking event fold is invalid")
        event_ids.append(event_id)
        patient_ids.append(patient_id)
    expected_count = manifest.get("event_count")
    if type(expected_count) is not int or len(event_ids) != expected_count:
        raise UnsafeArtifactError("source-train event count disagrees with manifest")
    if len(set(event_ids)) != len(event_ids):
        raise UnsafeArtifactError("source-train event roster contains duplicates")
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    return tuple(event_ids), tuple(patient_ids), manifest_sha, events_sha


def _validate_tensor_specs(
    manifest: Mapping[str, object],
    tensor_shapes: Mapping[str, tuple[int, ...]],
    tensor_dtypes: Mapping[str, str],
) -> None:
    prediction = _require_mapping(
        manifest.get("prediction_payload"), name="prediction_payload"
    )
    specs = _require_mapping(prediction.get("tensor_specs"), name="tensor_specs")
    if set(specs) != set(SAFE_RANKING_KEYS):
        raise UnsafeArtifactError("prediction tensor specification schema drifted")
    for name in SAFE_RANKING_KEYS:
        row = _require_mapping(specs.get(name), name=f"tensor_specs.{name}")
        shape = tuple(_require_sequence(row.get("shape"), name=f"{name}.shape"))
        dtype = _require_text(row.get("dtype"), name=f"{name}.dtype")
        if shape != tensor_shapes[name] or dtype != tensor_dtypes[name]:
            raise UnsafeArtifactError(f"prediction tensor disagrees with manifest: {name}")


def load_target_excluding_historical_ranking(
    ranking_directory: str | Path,
    roster_directory: str | Path,
) -> _RankingBundle:
    """Load only the sealed target-excluding historical ranking payload.

    Known historical OOF schemas are rejected from their JSON manifest alone;
    their target-colocated safetensors files are never opened or hashed.
    """

    directory = _canonical_directory(ranking_directory, name="ranking directory")
    _, manifest_raw, manifest = _load_json_file(
        directory / "manifest.json", name="ranking manifest"
    )
    schema = manifest.get("schema_version")
    if schema in UNSAFE_COLOCATED_TARGET_RANKING_SCHEMAS:
        raise UnsafeArtifactError(
            "Historical OOF ranking artifact is blocked: its prediction payload "
            "co-serializes DeepSOZ target values/target masks.  The assembler "
            "will not open that tensor file."
        )
    if schema != SAFE_RANKING_SCHEMA:
        raise UnsafeArtifactError("unsupported ranking manifest schema")
    if manifest.get("status") != SAFE_RANKING_STATUS:
        raise UnsafeArtifactError("ranking artifact is not the sealed completed refit")
    if manifest.get("candidate") != SAFE_RANKING_CANDIDATE:
        raise UnsafeArtifactError("ranking artifact is not temporal_mil_exact")
    prediction_contract = _require_mapping(
        manifest.get("prediction_payload"), name="prediction_payload"
    )
    expected_prediction = {
        "role": SAFE_PREDICTION_ROLE,
        "contains_target_values": False,
        "contains_target_mask": False,
        "eligible_as_evaluation": False,
    }
    for name, expected in expected_prediction.items():
        if prediction_contract.get(name) != expected:
            raise UnsafeArtifactError(f"unsafe prediction contract: {name}")
    scientific = _require_mapping(
        manifest.get("scientific_boundary"), name="scientific_boundary"
    )
    expected_scientific = {
        "fit_scope": "full_source_train_only",
        "resubstitution_predictions_are_evaluation": False,
        "source_eval_used": False,
        "private_used": False,
        "confirmatory_result": False,
    }
    for name, expected in expected_scientific.items():
        if scientific.get(name) != expected:
            raise UnsafeArtifactError(f"unsafe ranking scientific boundary: {name}")
    access = _require_mapping(manifest.get("data_access"), name="data_access")
    for name in (
        "source_dev_signal_or_target_open_count",
        "source_eval_signal_or_target_open_count",
        "private_signal_or_target_open_count",
        "other_split_input_path_count",
    ):
        if access.get(name) != 0:
            raise UnsafeArtifactError(f"ranking fit accessed forbidden scope: {name}")
    if access.get("loaded_model_splits") != ["source_train"]:
        raise UnsafeArtifactError("ranking fit is not source_train-only")

    roster_root = _canonical_directory(roster_directory, name="roster directory")
    event_ids, event_patient_ids, roster_manifest_sha, roster_sha = (
        _load_source_train_roster(roster_root)
    )
    lineage = _require_mapping(manifest.get("lineage"), name="ranking.lineage")
    if lineage.get("source_train_iv_manifest_sha256") != roster_manifest_sha:
        raise UnsafeArtifactError("ranking and source-train roster lineage disagree")

    files = _require_mapping(manifest.get("files"), name="ranking.files")
    prediction_path, prediction_sha = _require_file_receipt(
        directory,
        files,
        SAFE_PREDICTION_FILENAME,
        inspect_safetensor_header_first=True,
    )
    _, checkpoint_sha = _require_file_receipt(
        directory, files, SAFE_CHECKPOINT_FILENAME
    )
    ranking_manifest_sha = hashlib.sha256(manifest_raw).hexdigest()

    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for report assembly") from exc
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(prediction_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(SAFE_RANKING_KEYS):
            raise UnsafeArtifactError("prediction header changed after verification")
        for name in ("event_logits", "patient_logits", "event_patient_index"):
            tensors[name] = handle.get_tensor(name).detach().cpu().contiguous()
        tensor_shapes = {
            name: tuple(handle.get_slice(name).get_shape())
            for name in SAFE_RANKING_KEYS
        }
        tensor_dtypes = {
            name: str(handle.get_slice(name).get_dtype()).lower()
            for name in SAFE_RANKING_KEYS
        }
    if _file_sha256(prediction_path) != prediction_sha:
        raise UnsafeArtifactError("prediction artifact changed while being read")
    dtype_alias = {"f32": "float32", "i64": "int64"}
    tensor_dtypes = {
        name: dtype_alias.get(dtype, dtype) for name, dtype in tensor_dtypes.items()
    }
    _validate_tensor_specs(manifest, tensor_shapes, tensor_dtypes)

    event_logits = tensors["event_logits"]
    patient_logits = tensors["patient_logits"]
    event_patient_index = tensors["event_patient_index"]
    patient_ids_raw = _require_sequence(manifest.get("patient_ids"), name="patient_ids")
    patient_ids = tuple(_require_text(value, name="patient_id") for value in patient_ids_raw)
    if tuple(event_logits.shape) != (len(event_ids), N_STANDARD_CHANNELS):
        raise UnsafeArtifactError("event logits do not match source-train roster")
    if tuple(patient_logits.shape) != (len(patient_ids), N_STANDARD_CHANNELS):
        raise UnsafeArtifactError("patient logits do not match patient roster")
    if event_patient_index.dtype != torch.long or tuple(event_patient_index.shape) != (
        len(event_ids),
    ):
        raise UnsafeArtifactError("event_patient_index must be int64 [E]")
    if event_logits.dtype != torch.float32 or patient_logits.dtype != torch.float32:
        raise UnsafeArtifactError("ranking logits must be float32")
    if not torch.isfinite(event_logits).all() or not torch.isfinite(patient_logits).all():
        raise UnsafeArtifactError("ranking logits contain non-finite values")
    if len(set(patient_ids)) != len(patient_ids):
        raise UnsafeArtifactError("patient roster contains duplicates")
    if len(event_ids) < len(patient_ids) or event_patient_index.numel() == 0:
        raise UnsafeArtifactError("ranking event-to-patient carrier is incomplete")
    if int(event_patient_index.min()) != 0 or int(event_patient_index.max()) != len(
        patient_ids
    ) - 1:
        raise UnsafeArtifactError("ranking event-to-patient carrier is out of range")
    for position, patient_id in enumerate(event_patient_ids):
        carried = patient_ids[int(event_patient_index[position])]
        if carried != patient_id:
            raise UnsafeArtifactError("ranking carrier and event roster patient disagree")
    replay = torch.stack(
        [
            event_logits[event_patient_index == patient_index].mean(dim=0)
            for patient_index in range(len(patient_ids))
        ]
    )
    if not torch.allclose(replay, patient_logits, atol=1e-6, rtol=1e-6):
        raise UnsafeArtifactError("patient logits do not replay by equal-event mean")
    expected_event_roster_sha = _canonical_sha256(list(event_ids))
    expected_patient_roster_sha = _canonical_sha256(list(patient_ids))
    if manifest.get("event_roster_sha256") != expected_event_roster_sha:
        raise UnsafeArtifactError("ranking event roster digest disagrees")
    if manifest.get("patient_roster_sha256") != expected_patient_roster_sha:
        raise UnsafeArtifactError("ranking patient roster digest disagrees")
    return _RankingBundle(
        patient_ids=patient_ids,
        event_ids=event_ids,
        event_patient_ids=event_patient_ids,
        event_patient_index=event_patient_index,
        event_logits=event_logits,
        patient_logits=patient_logits,
        model_checkpoint_sha256=checkpoint_sha,
        ranking_manifest_sha256=ranking_manifest_sha,
        prediction_artifact_sha256=prediction_sha,
        roster_manifest_sha256=roster_manifest_sha,
        roster_artifact_sha256=roster_sha,
    )


def _load_public_union(path: str | Path) -> _PublicUnion:
    _, raw, payload = _load_json_file(path, name="public development union")
    if payload.get("schema_version") != PUBLIC_UNION_SCHEMA:
        raise UnsafeArtifactError("unsupported public union schema")
    if payload.get("cohort_name") != "public_development_union":
        raise UnsafeArtifactError("public union cohort identity changed")
    access = _require_mapping(payload.get("access_receipt"), name="union.access_receipt")
    expected_access = {
        "deepsoz_target_values_loaded": False,
        "prediction_artifacts_loaded": False,
        "private_eeg_loaded": False,
        "private_target_values_loaded": False,
        "raw_eeg_loaded": False,
        "signal_metadata_loaded": True,
        "source_eval_target_values_loaded": False,
    }
    if dict(access) != expected_access:
        raise UnsafeArtifactError("public union is not the sealed target-free metadata view")
    events = _require_sequence(payload.get("events"), name="union.events")
    expected_keys = {
        "edf_receipt_sha256",
        "edf_sha256",
        "event_id",
        "event_record_sha256",
        "global_event_index",
        "global_stop_sec",
        "global_t0_sec",
        "legacy_model_split",
        "official_split",
        "ordinal",
        "outer_fold",
        "patient_id",
        "processed_window_dtype",
        "processed_window_sha256",
        "processed_window_shape",
        "relative_edf_path",
        "signal_receipt_sha256",
    }
    index: dict[tuple[str, float, int], Mapping[str, object]] = {}
    ordered_event_ids: list[str] = []
    patient_ids: set[str] = set()
    for raw_event in events:
        event = _require_mapping(raw_event, name="union.event")
        if set(event) != expected_keys:
            raise UnsafeArtifactError("public union event schema drifted")
        event_id = _require_text(event.get("event_id"), name="union.event_id")
        match = _CANONICAL_EVENT_RE.fullmatch(event_id)
        if match is None:
            raise PublicReportAssemblyError("public union event ID is not canonical")
        relative = _validate_relative_edf(event.get("relative_edf_path"))
        global_t0 = event.get("global_t0_sec")
        global_index = event.get("global_event_index")
        if (
            isinstance(global_t0, bool)
            or not isinstance(global_t0, (int, float))
            or not math.isfinite(float(global_t0))
        ):
            raise PublicReportAssemblyError("union global_t0_sec is invalid")
        if type(global_index) is not int or global_index < 0:
            raise PublicReportAssemblyError("union global_event_index is invalid")
        stem = PurePosixPath(relative).stem
        if match.group("stem") != stem or int(match.group("index")) != global_index:
            raise UnsafeArtifactError("union event identity is internally inconsistent")
        _require_sha256(event.get("edf_sha256"), name="union.edf_sha256")
        patient_id = _require_text(event.get("patient_id"), name="union.patient_id")
        key = (relative, float(global_t0), global_index)
        if key in index:
            raise UnsafeArtifactError("public union has a duplicate signal identity")
        index[key] = event
        ordered_event_ids.append(event_id)
        patient_ids.add(patient_id)
    if payload.get("event_count") != len(events):
        raise UnsafeArtifactError("public union event count disagrees")
    if payload.get("patient_count") != len(patient_ids):
        raise UnsafeArtifactError("public union patient count disagrees")
    declared_ids = tuple(
        _require_text(value, name="union.declared_event_id")
        for value in _require_sequence(payload.get("event_ids"), name="union.event_ids")
    )
    if tuple(ordered_event_ids) != declared_ids:
        raise UnsafeArtifactError("public union declared event order disagrees")
    return _PublicUnion(
        events_by_identity=index,
        artifact_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _load_phenotype_payload(
    path: str | Path,
) -> tuple[Mapping[str, object], str]:
    _, raw, payload = _load_json_file(path, name="event phenotype audit")
    if payload.get("schema_version") != PHENOTYPE_AUDIT_SCHEMA:
        raise UnsafeArtifactError("unsupported event phenotype audit schema")
    if payload.get("status") != PHENOTYPE_AUDIT_STATUS:
        raise UnsafeArtifactError("event phenotype audit is not completed")
    if payload.get("producer_schema") != PHENOTYPE_PRODUCER_SCHEMA:
        raise UnsafeArtifactError("event phenotype producer schema changed")
    access = _require_mapping(
        payload.get("access_receipt"), name="phenotype.access_receipt"
    )
    expected_false = (
        "tusz_native_target_values_loaded",
        "deepsoz_target_values_loaded",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "training_performed",
        "threshold_selection_performed",
    )
    for name in expected_false:
        if _require_bool(access.get(name), name=f"access_receipt.{name}"):
            raise UnsafeArtifactError(f"phenotype artifact accessed forbidden data: {name}")
    events = _require_sequence(payload.get("events"), name="phenotype.events")
    if access.get("selected_event_count") != len(events):
        raise UnsafeArtifactError("phenotype selected-event count disagrees")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_reference_payload(
    path: str | Path,
) -> tuple[Mapping[str, object], str]:
    _, raw, payload = _load_json_file(path, name="reference robustness audit")
    if payload.get("schema_version") != REFERENCE_AUDIT_SCHEMA:
        raise UnsafeArtifactError("unsupported reference audit schema")
    if payload.get("status") != REFERENCE_AUDIT_STATUS:
        raise UnsafeArtifactError("reference audit is not completed audit-only output")
    access = _require_mapping(payload.get("access_receipt"), name="reference.access")
    for name in (
        "deepsoz_target_values_loaded",
        "tusz_native_target_values_loaded",
        "private_target_values_loaded",
        "private_eeg_loaded",
        "training_performed",
        "model_selection_performed",
    ):
        if _require_bool(access.get(name), name=f"reference.access.{name}"):
            raise UnsafeArtifactError(f"reference audit accessed forbidden data: {name}")
    return payload, hashlib.sha256(raw).hexdigest()


def _patient_event_ids(bundle: _RankingBundle, patient_index: int) -> tuple[str, ...]:
    return tuple(
        event_id
        for event_id, carrier in zip(bundle.event_ids, bundle.event_patient_index.tolist())
        if int(carrier) == patient_index
    )


def _ranking_aggregation_receipt(
    bundle: _RankingBundle,
    *,
    patient_id: str,
    patient_index: int,
    event_ids: tuple[str, ...],
) -> tuple[dict[str, object], str]:
    score = bundle.patient_logits[patient_index]
    payload = {
        "schema_version": "soz_public_patient_ranking_aggregation_receipt_v1",
        "patient_pseudonym": patient_id,
        "patient_index": patient_index,
        "aggregation_method": "patient_equal_event_mean",
        "aggregation_event_ids": list(event_ids),
        "score_semantics": "uncalibrated_localization_score",
        "patient_score_tensor_sha256": _tensor_sha256(score),
        "ranking_manifest_sha256": bundle.ranking_manifest_sha256,
        "ranking_prediction_sha256": bundle.prediction_artifact_sha256,
        "ranking_roster_manifest_sha256": bundle.roster_manifest_sha256,
        "ranking_event_roster_sha256": bundle.roster_artifact_sha256,
        "prediction_role": SAFE_PREDICTION_ROLE,
        "evaluation_eligible": False,
    }
    return payload, _canonical_sha256(payload)


def _match_public_identity(
    phenotype_row: Mapping[str, object],
    public_union: _PublicUnion,
) -> Mapping[str, object] | None:
    relative = _validate_relative_edf(phenotype_row.get("relative_edf_path"))
    global_t0 = phenotype_row.get("global_t0_sec")
    if (
        isinstance(global_t0, bool)
        or not isinstance(global_t0, (int, float))
        or not math.isfinite(float(global_t0))
    ):
        raise PublicReportAssemblyError("phenotype global_t0_sec is invalid")
    source_event = _require_text(phenotype_row.get("event_id"), name="source event_id")
    match = _GLOBAL_EVENT_RE.fullmatch(source_event)
    if match is None:
        raise PublicReportAssemblyError("phenotype event ID is not canonical source form")
    global_index = int(match.group("index"))
    if match.group("stem") != PurePosixPath(relative).stem:
        raise UnsafeArtifactError("phenotype event ID and EDF path disagree")
    source_patient = _require_text(
        phenotype_row.get("patient_id"), name="source patient_id"
    )
    if match.group("stem").split("_", 1)[0] != source_patient:
        raise UnsafeArtifactError("phenotype event and source patient disagree")
    return public_union.events_by_identity.get((relative, float(global_t0), global_index))


def _normalize_event_identity(
    event: EventScalpPhenotypeEvidence,
    *,
    source_row: Mapping[str, object],
    union_row: Mapping[str, object],
) -> tuple[EventScalpPhenotypeEvidence, EvidenceProvenanceReceipt]:
    source_receipt = event.receipt
    source_patient = _require_text(source_row.get("patient_id"), name="source patient")
    source_event = _require_text(source_row.get("event_id"), name="source event")
    canonical_patient = _require_text(union_row.get("patient_id"), name="canonical patient")
    canonical_event = _require_text(union_row.get("event_id"), name="canonical event")
    if source_receipt.patient_pseudonym != source_patient:
        raise UnsafeArtifactError("source receipt patient does not match phenotype row")
    if source_receipt.event_pseudonym != source_event:
        raise UnsafeArtifactError("source receipt event does not match phenotype row")
    signal_sha = _require_sha256(union_row.get("edf_sha256"), name="union.edf_sha256")
    if source_receipt.signal_artifact_sha256 != signal_sha:
        raise UnsafeArtifactError("phenotype and public union signal digests disagree")
    canonical_receipt = replace(
        source_receipt,
        patient_pseudonym=canonical_patient,
        event_pseudonym=canonical_event,
    )
    return replace(event, receipt=canonical_receipt), source_receipt


def _assert_report_claim_boundary(report: GroundedChineseDiagnosticReport) -> None:
    if any(phrase in report.text for phrase in _FORBIDDEN_REPORT_CLAIMS):
        raise UnsafeArtifactError("renderer emitted a forbidden clinical claim")
    required = (
        "固定算法最先检出的持续变化候选",
        "不是SOZ概率、定位准确率或预处理选臂依据",
        "不等同于侵入式皮层SOZ",
        "不得单独作为手术决策依据",
    )
    if any(phrase not in report.text for phrase in required):
        raise UnsafeArtifactError("renderer omitted a required claim boundary")
    if "source_train_resubstitution_fit_sanity_only_not_evaluation" not in report.text:
        raise UnsafeArtifactError("renderer omitted the resubstitution-only model identity")
    if "模型已拒答" not in report.text:
        raise UnsafeArtifactError("resubstitution ranking must remain abstained")


def assemble_public_grounded_reports(
    *,
    phenotype_artifact: str | Path,
    ranking_directory: str | Path,
    ranking_roster_directory: str | Path,
    public_union_manifest: str | Path,
    reference_audit_artifact: str | Path,
) -> PublicReportAssemblyBatch:
    """Assemble every phenotype row, explicitly blocking unsupported rows."""

    ranking = load_target_excluding_historical_ranking(
        ranking_directory, ranking_roster_directory
    )
    union = _load_public_union(public_union_manifest)
    phenotype_payload, phenotype_sha = _load_phenotype_payload(phenotype_artifact)
    reference_payload, reference_sha = _load_reference_payload(
        reference_audit_artifact
    )
    raw_rows = _require_sequence(phenotype_payload.get("events"), name="phenotype.events")
    ranking_event_positions = {
        event_id: position for position, event_id in enumerate(ranking.event_ids)
    }
    patient_positions = {
        patient_id: position for position, patient_id in enumerate(ranking.patient_ids)
    }
    records: list[
        AssembledPublicDiagnosticReport | BlockedPublicDiagnosticReport
    ] = []
    expected_row_keys = {
        "detected_bipolar_edge_count",
        "event_id",
        "global_t0_sec",
        "patient_id",
        "phenotype",
        "reason_codes",
        "relative_edf_path",
        "status",
    }
    for raw_row in raw_rows:
        row = _require_mapping(raw_row, name="phenotype.event")
        if set(row) != expected_row_keys:
            raise UnsafeArtifactError("phenotype event-row schema drifted")
        source_patient = _require_text(row.get("patient_id"), name="source patient")
        source_event = _require_text(row.get("event_id"), name="source event")
        union_row = _match_public_identity(row, union)
        if union_row is None:
            records.append(
                BlockedPublicDiagnosticReport(
                    source_patient_pseudonym=source_patient,
                    source_event_pseudonym=source_event,
                    reason_codes=("no_exact_public_signal_identity",),
                )
            )
            continue
        canonical_event = _require_text(union_row.get("event_id"), name="canonical event")
        canonical_patient = _require_text(
            union_row.get("patient_id"), name="canonical patient"
        )
        event_position = ranking_event_positions.get(canonical_event)
        patient_position = patient_positions.get(canonical_patient)
        if event_position is None or patient_position is None:
            records.append(
                BlockedPublicDiagnosticReport(
                    source_patient_pseudonym=source_patient,
                    source_event_pseudonym=source_event,
                    reason_codes=("event_absent_from_safe_ranking_roster",),
                )
            )
            continue
        if int(ranking.event_patient_index[event_position]) != patient_position:
            raise UnsafeArtifactError("canonical event-to-patient mapping disagrees")
        if row.get("status") == "abstained":
            if row.get("phenotype") is not None:
                raise UnsafeArtifactError("abstained phenotype row contains facts")
            records.append(
                BlockedPublicDiagnosticReport(
                    source_patient_pseudonym=source_patient,
                    source_event_pseudonym=source_event,
                    reason_codes=("source_abstention_lacks_evidence_receipt",),
                )
            )
            continue
        if row.get("status") != "reportable":
            raise UnsafeArtifactError("unsupported phenotype row status")
        event = _parse_event_phenotype(row.get("phenotype"))
        normalized_event, source_receipt = _normalize_event_identity(
            event, source_row=row, union_row=union_row
        )
        aggregation_event_ids = _patient_event_ids(ranking, patient_position)
        if canonical_event not in aggregation_event_ids:
            raise UnsafeArtifactError("report event is absent from patient aggregation")
        aggregation_payload, aggregation_sha = _ranking_aggregation_receipt(
            ranking,
            patient_id=canonical_patient,
            patient_index=patient_position,
            event_ids=aggregation_event_ids,
        )
        patient_score_sha = _require_sha256(
            aggregation_payload.get("patient_score_tensor_sha256"),
            name="patient_score_tensor_sha256",
        )
        spatial = derive_spatial_report(
            ranking.patient_logits[patient_position],
            torch.ones(N_STANDARD_CHANNELS, dtype=torch.bool),
            score_semantics="uncalibrated_localization_score",
        )
        patient_ranking = PatientSOZReferenceRanking(
            spatial_report=spatial,
            patient_pseudonym=canonical_patient,
            model_id="LaBraM-temporal_mil_exact",
            model_version=(
                "仅限source-train重代入拟合自检，不是评价；"
                "source_train_resubstitution_fit_sanity_only_not_evaluation"
            ),
            model_checkpoint_sha256=ranking.model_checkpoint_sha256,
            aggregation_method="patient_equal_event_mean",
            aggregation_event_count=len(aggregation_event_ids),
            aggregation_event_ids=aggregation_event_ids,
            aggregation_receipt_sha256=aggregation_sha,
            uncertainty=UncertaintyDecomposition(
                abstain=True,
                abstention_reason_codes=(
                    "source_train_resubstitution_only",
                    "uncertainty_not_materialized",
                ),
            ),
        )
        facts = ClinicalReportFactsV2(
            event_phenotype=normalized_event,
            patient_ranking=patient_ranking,
            require_causal_prefix_safe=True,
        )
        if normalized_event.later_visible_derivations:
            try:
                region_production = produce_later_visible_region(
                    normalized_event.later_visible_derivations
                )
                region_receipt = build_later_visible_region_receipt(
                    region_production,
                    normalized_event.receipt,
                )
                facts = attach_later_visible_region_to_clinical_facts(
                    facts,
                    region_receipt,
                )
            except (TypeError, ValueError):
                records.append(
                    BlockedPublicDiagnosticReport(
                        source_patient_pseudonym=source_patient,
                        source_event_pseudonym=source_event,
                        reason_codes=(
                            "later_visible_region_receipt_binding_failed",
                        ),
                    )
                )
                continue
        try:
            reference = build_reference_disagreement_receipt(
                reference_payload,
                source_audit_artifact_sha256=reference_sha,
                patient_pseudonym=canonical_patient,
                aggregation_event_ids=aggregation_event_ids,
            )
            facts = attach_reference_disagreement_to_clinical_facts(facts, reference)
        except (TypeError, ValueError):
            records.append(
                BlockedPublicDiagnosticReport(
                    source_patient_pseudonym=source_patient,
                    source_event_pseudonym=source_event,
                    reason_codes=("reference_receipt_binding_failed",),
                )
            )
            continue
        report = render_grounded_chinese_diagnostic_report(facts)
        _assert_report_claim_boundary(report)
        source_receipt_sha = _canonical_sha256(asdict(source_receipt))
        canonical_unbound_receipt_sha = _canonical_sha256(
            asdict(normalized_event.receipt)
        )
        union_global_index = union_row.get("global_event_index")
        if type(union_global_index) is not int:
            raise UnsafeArtifactError("union global event index changed")
        assembly_receipt = PublicReportAssemblyReceipt(
            source_patient_pseudonym=source_patient,
            source_event_pseudonym=source_event,
            canonical_patient_pseudonym=canonical_patient,
            canonical_event_pseudonym=canonical_event,
            relative_edf_path=_validate_relative_edf(row.get("relative_edf_path")),
            global_t0_sec=float(row["global_t0_sec"]),
            global_event_index=union_global_index,
            signal_artifact_sha256=source_receipt.signal_artifact_sha256,
            source_event_receipt_sha256=source_receipt_sha,
            canonical_pre_reference_event_receipt_sha256=(
                canonical_unbound_receipt_sha
            ),
            phenotype_artifact_sha256=phenotype_sha,
            public_union_manifest_sha256=union.artifact_sha256,
            ranking_manifest_sha256=ranking.ranking_manifest_sha256,
            ranking_prediction_sha256=ranking.prediction_artifact_sha256,
            ranking_checkpoint_sha256=ranking.model_checkpoint_sha256,
            ranking_roster_manifest_sha256=ranking.roster_manifest_sha256,
            ranking_event_roster_sha256=ranking.roster_artifact_sha256,
            reference_audit_artifact_sha256=reference_sha,
            reference_disagreement_receipt_sha256=(
                facts.reference_disagreement_receipt.receipt_sha256
            ),
            patient_score_tensor_sha256=patient_score_sha,
            aggregation_receipt_sha256=aggregation_sha,
            later_visible_region_receipt_sha256=(
                None
                if facts.later_visible_region_receipt is None
                else facts.later_visible_region_receipt.receipt_sha256
            ),
        )
        records.append(
            AssembledPublicDiagnosticReport(
                source_patient_pseudonym=source_patient,
                source_event_pseudonym=source_event,
                canonical_patient_pseudonym=canonical_patient,
                canonical_event_pseudonym=canonical_event,
                typed_facts=facts,
                report=report,
                assembly_receipt=assembly_receipt,
            )
        )
    input_hashes = (
        ("phenotype_artifact", phenotype_sha),
        ("public_union_manifest", union.artifact_sha256),
        ("ranking_manifest", ranking.ranking_manifest_sha256),
        ("ranking_prediction", ranking.prediction_artifact_sha256),
        ("ranking_checkpoint", ranking.model_checkpoint_sha256),
        ("ranking_roster_manifest", ranking.roster_manifest_sha256),
        ("ranking_event_roster", ranking.roster_artifact_sha256),
        ("reference_audit_artifact", reference_sha),
    )
    return PublicReportAssemblyBatch(
        records=tuple(records), input_artifact_sha256s=input_hashes
    )


def write_public_report_batch(
    batch: PublicReportAssemblyBatch, output_path: str | Path
) -> str:
    """Atomically publish a new JSON result without modifying any input."""

    if not isinstance(batch, PublicReportAssemblyBatch):
        raise TypeError("batch must be PublicReportAssemblyBatch")
    output = Path(os.path.abspath(output_path))
    for component in (output, *output.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise UnsafeArtifactError("output cannot traverse symlinks")
    if os.path.lexists(output):
        raise FileExistsError(output)
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir():
        raise PublicReportAssemblyError("output parent must be a directory")
    raw = _canonical_bytes(batch.to_payload())
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and refuses to replace a file that
        # appears after the initial existence check.
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "AssembledPublicDiagnosticReport",
    "BlockedPublicDiagnosticReport",
    "PUBLIC_REPORT_ASSEMBLER_SCHEMA",
    "PUBLIC_REPORT_BATCH_STATUS",
    "PublicReportAssemblyBatch",
    "PublicReportAssemblyError",
    "PublicReportAssemblyReceipt",
    "UnsafeArtifactError",
    "assemble_public_grounded_reports",
    "load_target_excluding_historical_ranking",
    "write_public_report_batch",
]
