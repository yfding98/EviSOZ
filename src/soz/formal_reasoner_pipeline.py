"""Closed formal bridge from verified timelines to reasoner calibration.

The lower-level evidence and calibration modules intentionally remain useful
for unit testing and research diagnostics.  Their APIs therefore accept
individual tensors and caller-provided receipt digests.  Those APIs are not a
formal training authority.  This module is the production boundary that:

* derives current offset and previous-seizure state only from a strictly
  replayed :class:`VerifiedDeepSOZSignalPreflightBundle`;
* publishes caches only from opaque M/I/V producer outputs issued by strict
  artifact replay, while deriving all event, fold, extractor, authorization,
  temporal, and content receipts internally;
* joins targets only through a strictly verified target-v2 artifact and only
  after the evidence cache has crossed the authorization firewall; and
* computes source-development logits, targets, masks, and receipt hashes
  internally after the reasoner has been frozen.

No public function in this module accepts raw logits, target tensors, cache
receipt SHA values, a seizure duration, a previous-seizure gap, or an
``EvidenceBatch`` for formal publication.  Importantly, this module does not
"bless" caller tensors by attaching the lineage expected by an authorization.
Until each active family has a first-party issuer that replays and binds its
actual checkpoint, token, scaler, event, mask, and tensor receipts, formal
evidence publication fails closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import tempfile
from typing import Mapping, Sequence

import torch

from .data.batching import EvidenceEvent
from .data.deepsoz import normalize_patient_id
from .data.deepsoz_signal_preflight import (
    DEEPSOZ_SIGNAL_SOURCE,
    VerifiedDeepSOZSignalPreflightBundle,
)
from .data.deepsoz_target_v2 import VerifiedDeepSOZTargetV2Artifact
from .data.provenance import (
    EventInputRecord,
    EventInputRegistry,
    EvidenceCacheReceipt,
    EventTemporalProvenanceReceipt,
    evidence_batch_sha256,
    ictal_phase_mask_sha256,
    patient_roster_sha256,
)
from .evidence import EvidenceBatch
from .evidence_authorization import (
    AuthorizedEvidenceEvent,
    AuthorizedPatientBagDataset,
    FamilyProducerAuthorization,
    OOFEvidenceAuthorization,
    authorize_evidence_event,
    build_event_temporal_provenance_receipts,
    build_oof_evidence_authorization,
    load_authorized_evidence_cache,
)
from .evidence_io import save_evidence_cache
from .losses import PatientLevelSOZObjective
from .models.reasoner import AdditiveEvidenceReasoner
from .reasoner_calibration import (
    FrozenReasonerCheckpoint,
    GlobalAffineSOZCalibrator,
    ReasonerCalibrationData,
    build_reasoner_calibration_data,
    fit_global_affine_calibrator,
    freeze_reasoner_checkpoint,
    reasoner_state_sha256,
)
from .temporal_masks import (
    OFFSET_TIME_TOLERANCE_SEC,
    OffsetAwarePhaseMasks,
    build_offset_aware_phase_masks,
)
from .training import ReasonerEpochOutput, train_formal_reasoner_epoch


FORMAL_TIMELINE_CONTEXT_SCHEMA = "soz_verified_record_local_timeline_context_v2"
# ``global`` in the upstream TUSZ fields names the record-level global
# annotation layer (for example ``*.csv_bi``).  Its seconds are not a
# patient-global clock and cannot order events from different EDF records.
FORMAL_TIMELINE_SCOPE = "record_local_official_global_annotation_clock"
FORMAL_PRIOR_STOP_POLICY = (
    "cumulative_max_stop_over_all_lower_index_events_in_same_source_record"
)
FORMAL_EVIDENCE_CACHE_SET_SCHEMA = "soz_formal_evidence_cache_set_v1"
FORMAL_EVIDENCE_CACHE_INDEX_FILENAME = "formal_evidence_cache_set.json"
FORMAL_REASONER_DATASET_SCHEMA = "soz_verified_reasoner_dataset_v1"
FORMAL_REASONER_FIT_SCHEMA = "soz_verified_reasoner_fit_v1"
FORMAL_FROZEN_REASONER_SCHEMA = "soz_verified_frozen_reasoner_v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MODEL_SPLITS = ("source_train", "source_dev", "source_eval")
_REASONER_TARGET_SPLITS = ("source_train", "source_dev")
# Training-time cache materialization deliberately has the same split boundary
# as reasoner target joins.  source_eval requires a future one-shot release
# capability bound to the frozen reasoner, calibrator, and evaluation protocol;
# it must never be materialized through this development API.
_TRAINING_CACHE_SPLITS = _REASONER_TARGET_SPLITS
_SELECTION_KEYS = (0, 1, 2, 3, 4, None)
_MAX_INDEX_BYTES = 64 * 1024 * 1024
_TIMELINE_MARKER = object()
_CACHE_SET_MARKER = object()
_DATASET_MARKER = object()
_FIT_MARKER = object()
_FROZEN_MARKER = object()

_FORMAL_REASONER_FIT_POLICY = {
    "seed": 20260808,
    "epochs": 20,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    "hidden_dim": 16,
    "ranking_weight": 0.25,
    "ranking_margin": 0.0,
    "optimizer": "AdamW",
    "patient_order_policy": "deterministic_complete_roster_per_epoch",
    "calibration_during_fit": False,
}
_FORMAL_CALIBRATOR_MAX_STEPS = 300
_FORMAL_CALIBRATOR_LEARNING_RATE = 0.05

# This blocker may only be removed in code when morphology/ictal/evolution each
# expose a strict replay issuer whose opaque output binds the actual artifact
# bytes listed below.  There is intentionally no runtime enable flag,
# registration hook, or caller-provided override.
_STRICT_FAMILY_PRODUCER_REPLAY_BLOCKERS = (
    "strict_family_specific_opaque_output_capability",
    "actual_checkpoint_and_training_run_replay",
    "actual_event_token_and_preprocess_replay",
    "actual_scaler_replay_or_authenticated_not_applicable_receipt",
    "event_record_tensor_and_mask_sha_binding",
    "authenticated_per_event_unavailable_reason_receipt",
)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Formal pipeline provenance is not canonical JSON") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


# These digests are the pre-registered optimization policies.  They are
# constants derived from closed canonical payloads, not values supplied at a
# training call after source-dev results are visible.
FORMAL_REASONER_FIT_POLICY_SHA256 = _canonical_sha256(
    _FORMAL_REASONER_FIT_POLICY
)
FORMAL_CALIBRATOR_FIT_POLICY_SHA256 = _canonical_sha256(
    {
        "optimizer": "Adam",
        "execution": "cpu_float64",
        "max_steps": _FORMAL_CALIBRATOR_MAX_STEPS,
        "learning_rate": _FORMAL_CALIBRATOR_LEARNING_RATE,
        "parameterization": "global_log_temperature_and_bias",
    }
)


def _require_sha256(value: object, *, field_name: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def _require_text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _nonnegative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _safe_relative_path(value: object, *, field_name: str) -> str:
    text = _require_text(value, field_name=field_name)
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} must be a safe relative POSIX path")
    return path.as_posix()


def _tensor_sha256(name: str, tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    metadata = f"{name}|{tuple(value.shape)}|{value.dtype}".encode("ascii")
    digest.update(len(metadata).to_bytes(4, "little"))
    digest.update(metadata)
    raw = value.view(torch.uint8).numpy().tobytes()
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_signal_binding(
    target: VerifiedDeepSOZTargetV2Artifact,
    signal: VerifiedDeepSOZSignalPreflightBundle,
) -> None:
    if not isinstance(target, VerifiedDeepSOZTargetV2Artifact):
        raise TypeError("target must be a strictly verified target-v2 artifact")
    if not isinstance(signal, VerifiedDeepSOZSignalPreflightBundle):
        raise TypeError("signal must be a strictly verified signal-preflight bundle")
    receipt = signal.receipt
    target_receipt = target.receipt
    checks = {
        "target-v2 receipt": (
            receipt["verified_target_v2_receipt_sha256"]
            == target_receipt.receipt_sha256
        ),
        "target-v2 artifact": (
            receipt["verified_target_v2_artifact_sha256"]
            == target_receipt.target_artifact_sha256
        ),
        "target-v2 policy": (
            receipt["verified_target_v2_policy_sha256"]
            == target_receipt.policy_sha256
        ),
        "DeepSOZ source": (
            receipt["deepsoz_source_sha256"] == target_receipt.source_input_sha256
        ),
        "target split manifest": (
            receipt["split_manifest_sha256"] == target_receipt.split_input_sha256
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "Signal-preflight and verified target-v2 artifact were swapped or "
            f"rebuilt from different inputs: {failed}"
        )


@dataclass(frozen=True)
class VerifiedTimelineEvent:
    """Target-free timing extracted from one complete record-local timeline.

    The historical ``global_*`` field names refer to the official TUSZ global
    annotation layer within one EDF record.  They do not imply that clocks in
    two different EDF records can be aligned for the same patient.
    """

    event_id: str
    patient_id: str
    model_split: str
    timeline_scope: str
    prior_stop_policy: str
    event_record_sha256: str
    deepsoz_source_record_sha256: str
    global_timeline_receipt_sha256: str
    global_event_index: int
    global_t0_sec: float
    global_stop_sec: float
    seizure_duration_sec: float
    previous_global_stop_sec: float | None
    previous_seizure_gap_sec: float | None
    previous_overlaps_current: bool
    relative_edf_path: str
    preprocess_receipt_sha256: str
    processed_window_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _require_text(self.event_id, field_name="event_id"))
        object.__setattr__(self, "patient_id", normalize_patient_id(self.patient_id))
        if self.model_split not in _MODEL_SPLITS:
            raise ValueError("Verified timeline event uses an unsupported model split")
        if self.timeline_scope != FORMAL_TIMELINE_SCOPE:
            raise ValueError("Verified event cannot claim a patient-global timeline")
        if self.prior_stop_policy != FORMAL_PRIOR_STOP_POLICY:
            raise ValueError("Verified event uses the wrong prior-stop policy")
        for name in (
            "event_record_sha256",
            "deepsoz_source_record_sha256",
            "global_timeline_receipt_sha256",
            "preprocess_receipt_sha256",
            "processed_window_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "global_event_index",
            _nonnegative_int(self.global_event_index, field_name="global_event_index"),
        )
        start = _finite_float(self.global_t0_sec, field_name="global_t0_sec")
        stop = _finite_float(self.global_stop_sec, field_name="global_stop_sec")
        duration = _finite_float(
            self.seizure_duration_sec, field_name="seizure_duration_sec"
        )
        if stop <= start or duration <= 0 or abs(duration - (stop - start)) > 1e-6:
            raise ValueError("Verified seizure duration must equal global stop minus t0")
        object.__setattr__(self, "global_t0_sec", start)
        object.__setattr__(self, "global_stop_sec", stop)
        object.__setattr__(self, "seizure_duration_sec", duration)
        if self.previous_global_stop_sec is None:
            if self.previous_seizure_gap_sec is not None or self.previous_overlaps_current:
                raise ValueError("No-previous timeline state is internally inconsistent")
        else:
            previous_stop = _finite_float(
                self.previous_global_stop_sec, field_name="previous_global_stop_sec"
            )
            if self.previous_seizure_gap_sec is None:
                raise ValueError("A previous event requires an explicit non-negative gap")
            gap = _finite_float(
                self.previous_seizure_gap_sec,
                field_name="previous_seizure_gap_sec",
            )
            if gap < 0:
                raise ValueError("Previous-seizure gap cannot be negative")
            expected_overlap = previous_stop > start + OFFSET_TIME_TOLERANCE_SEC
            expected_gap = max(0.0, start - previous_stop)
            if abs(gap - expected_gap) > 1e-6:
                raise ValueError("Previous-seizure gap was not derived from global times")
            if self.previous_overlaps_current != expected_overlap:
                raise ValueError("Previous-current overlap state disagrees with global times")
            object.__setattr__(self, "previous_global_stop_sec", previous_stop)
            object.__setattr__(self, "previous_seizure_gap_sec", gap)
        if not isinstance(self.previous_overlaps_current, bool):
            raise TypeError("previous_overlaps_current must be bool")
        object.__setattr__(
            self,
            "relative_edf_path",
            _safe_relative_path(self.relative_edf_path, field_name="relative_edf_path"),
        )


def _context_payload(
    *,
    target_v2_receipt_sha256: str,
    signal_preflight_receipt_sha256: str,
    signal_preflight_artifact_sha256: str,
    timeline_scope: str,
    prior_stop_policy: str,
    event_registry: EventInputRegistry,
    timeline_events: Sequence[VerifiedTimelineEvent],
    temporal_receipts: Sequence[EventTemporalProvenanceReceipt],
) -> dict[str, object]:
    return {
        "schema_version": FORMAL_TIMELINE_CONTEXT_SCHEMA,
        "timeline_scope": timeline_scope,
        "prior_stop_policy": prior_stop_policy,
        "target_v2_receipt_sha256": target_v2_receipt_sha256,
        "signal_preflight_receipt_sha256": signal_preflight_receipt_sha256,
        "signal_preflight_artifact_sha256": signal_preflight_artifact_sha256,
        "event_registry_sha256": event_registry.manifest_sha256,
        "split_manifest_sha256": event_registry.split_manifest_sha256,
        "events": [asdict(event) for event in timeline_events],
        "temporal_receipt_sha256s": [
            receipt.receipt_sha256 for receipt in temporal_receipts
        ],
    }


@dataclass(frozen=True, init=False)
class VerifiedGlobalTimelineContext:
    """Opaque record-local current/prior timing capability from strict replay.

    The class name is retained for API compatibility.  ``global`` means the
    official global annotation layer inside each TUSZ EDF, not a patient-wide
    clock.  Records are deliberately not linked unless an independently
    verified cross-record clock becomes available in a future schema.
    """

    target_v2_receipt_sha256: str
    signal_preflight_receipt_sha256: str
    signal_preflight_artifact_sha256: str
    timeline_scope: str
    prior_stop_policy: str
    event_registry: EventInputRegistry
    timeline_events: tuple[VerifiedTimelineEvent, ...]
    phase_masks: OffsetAwarePhaseMasks
    temporal_receipts: tuple[EventTemporalProvenanceReceipt, ...]
    receipt_sha256: str

    def __init__(
        self,
        *,
        _verification_marker: object,
        target_v2_receipt_sha256: str,
        signal_preflight_receipt_sha256: str,
        signal_preflight_artifact_sha256: str,
        event_registry: EventInputRegistry,
        timeline_events: Sequence[VerifiedTimelineEvent],
        phase_masks: OffsetAwarePhaseMasks,
        temporal_receipts: Sequence[EventTemporalProvenanceReceipt],
    ) -> None:
        if _verification_marker is not _TIMELINE_MARKER:
            raise TypeError(
                "VerifiedGlobalTimelineContext can only be issued by strict builder"
            )
        if not isinstance(event_registry, EventInputRegistry):
            raise TypeError("event_registry must be EventInputRegistry")
        if not isinstance(phase_masks, OffsetAwarePhaseMasks):
            raise TypeError("phase_masks must be OffsetAwarePhaseMasks")
        events = tuple(timeline_events)
        receipts = tuple(temporal_receipts)
        event_ids = tuple(record.event_id for record in event_registry)
        if tuple(event.event_id for event in events) != event_ids:
            raise ValueError("Verified timeline events do not equal the event registry")
        if tuple(receipt.event_id for receipt in receipts) != event_ids:
            raise ValueError("Temporal receipts do not equal the event registry")
        if len(event_ids) != phase_masks.ictal_phase_mask.shape[0]:
            raise ValueError("Phase-mask batch does not equal the event registry")
        target_sha = _require_sha256(
            target_v2_receipt_sha256, field_name="target_v2_receipt_sha256"
        )
        signal_sha = _require_sha256(
            signal_preflight_receipt_sha256,
            field_name="signal_preflight_receipt_sha256",
        )
        artifact_sha = _require_sha256(
            signal_preflight_artifact_sha256,
            field_name="signal_preflight_artifact_sha256",
        )
        payload = _context_payload(
            target_v2_receipt_sha256=target_sha,
            signal_preflight_receipt_sha256=signal_sha,
            signal_preflight_artifact_sha256=artifact_sha,
            timeline_scope=FORMAL_TIMELINE_SCOPE,
            prior_stop_policy=FORMAL_PRIOR_STOP_POLICY,
            event_registry=event_registry,
            timeline_events=events,
            temporal_receipts=receipts,
        )
        object.__setattr__(self, "target_v2_receipt_sha256", target_sha)
        object.__setattr__(self, "signal_preflight_receipt_sha256", signal_sha)
        object.__setattr__(self, "signal_preflight_artifact_sha256", artifact_sha)
        object.__setattr__(self, "timeline_scope", FORMAL_TIMELINE_SCOPE)
        object.__setattr__(self, "prior_stop_policy", FORMAL_PRIOR_STOP_POLICY)
        object.__setattr__(self, "event_registry", event_registry)
        object.__setattr__(self, "timeline_events", events)
        object.__setattr__(self, "phase_masks", phase_masks)
        object.__setattr__(self, "temporal_receipts", receipts)
        object.__setattr__(self, "receipt_sha256", _canonical_sha256(payload))

    def timeline_event(self, event_id: object) -> VerifiedTimelineEvent:
        key = str(event_id).strip()
        matches = tuple(event for event in self.timeline_events if event.event_id == key)
        if len(matches) != 1:
            raise KeyError(f"Unknown verified timeline event: {key}")
        return matches[0]

    def temporal_receipt(self, event_id: object) -> EventTemporalProvenanceReceipt:
        key = str(event_id).strip()
        matches = tuple(
            receipt for receipt in self.temporal_receipts if receipt.event_id == key
        )
        if len(matches) != 1:
            raise KeyError(f"Unknown temporal receipt event: {key}")
        return matches[0]

    def phase_mask(self, event_id: object) -> torch.Tensor:
        key = str(event_id).strip()
        event_ids = tuple(event.event_id for event in self.timeline_events)
        try:
            index = event_ids.index(key)
        except ValueError as exc:
            raise KeyError(f"Unknown phase-mask event: {key}") from exc
        return self.phase_masks.ictal_phase_mask[index].detach().clone()

    def assert_unchanged(self) -> None:
        if tuple(record.event_id for record in self.event_registry) != tuple(
            event.event_id for event in self.timeline_events
        ):
            raise ValueError("Verified timeline event registry changed")
        for index, receipt in enumerate(self.temporal_receipts):
            if receipt.ictal_phase_mask_sha256 != ictal_phase_mask_sha256(
                self.phase_masks.ictal_phase_mask[index]
            ):
                raise ValueError("Verified timeline phase masks changed after issuance")
        payload = _context_payload(
            target_v2_receipt_sha256=self.target_v2_receipt_sha256,
            signal_preflight_receipt_sha256=self.signal_preflight_receipt_sha256,
            signal_preflight_artifact_sha256=self.signal_preflight_artifact_sha256,
            timeline_scope=self.timeline_scope,
            prior_stop_policy=self.prior_stop_policy,
            event_registry=self.event_registry,
            timeline_events=self.timeline_events,
            temporal_receipts=self.temporal_receipts,
        )
        if _canonical_sha256(payload) != self.receipt_sha256:
            raise ValueError("Verified global timeline context changed after issuance")


def _timeline_candidate(row: Mapping[str, object]) -> dict[str, object]:
    start = _finite_float(row["global_t0_sec"], field_name="global_t0_sec")
    stop = _finite_float(row["global_stop_sec"], field_name="global_stop_sec")
    if stop <= start:
        raise ValueError("Global seizure stop must be later than global t0")
    return {
        "event_id": _require_text(row["event_id"], field_name="event_id"),
        "event_record_sha256": _require_sha256(
            row["event_record_sha256"], field_name="event_record_sha256"
        ),
        "deepsoz_source_record_sha256": _require_sha256(
            row["deepsoz_source_record_sha256"],
            field_name="deepsoz_source_record_sha256",
        ),
        "annotation_pair_sha256": _require_sha256(
            row["annotation_pair_sha256"], field_name="annotation_pair_sha256"
        ),
        "patient_id": normalize_patient_id(row["patient_id"]),
        "local_patient_id": _require_text(
            row["local_patient_id"], field_name="local_patient_id"
        ),
        "official_split": _require_text(
            row["official_split"], field_name="official_split"
        ),
        "model_split": _require_text(row["model_split"], field_name="model_split"),
        "relative_edf_path": _safe_relative_path(
            row["relative_edf_path"], field_name="relative_edf_path"
        ),
        "deepsoz_record": _require_text(
            row["deepsoz_record"], field_name="deepsoz_record"
        ),
        "global_event_index": _nonnegative_int(
            row["global_event_index"], field_name="global_event_index"
        ),
        "global_t0_sec": start,
        "global_stop_sec": stop,
    }


def build_verified_global_timeline_context(
    verified_target_v2: VerifiedDeepSOZTargetV2Artifact,
    signal_preflight: VerifiedDeepSOZSignalPreflightBundle,
) -> VerifiedGlobalTimelineContext:
    """Derive masks from complete per-record official annotation timelines.

    ``window_stop_sec`` is used only to validate the frozen 60-second crop.  It
    is never read as a seizure offset.  Current duration is always
    ``global_stop_sec - global_t0_sec``.  Previous-event state is reconstructed
    from the union of eligible events and signal-excluded candidate events, so
    an excluded prior event cannot silently turn pre-anchor context into a
    baseline.  For nested or overlapping annotations, the prior boundary is
    the maximum stop over *all* lower-index events in the same source record,
    not merely the immediately preceding row.

    Upstream ``global_*`` seconds are record-relative.  Events from different
    EDF records are never ordered here, even when they belong to one patient;
    doing so would falsely assert a patient-global clock that the verified
    inputs do not provide.
    """

    _target_signal_binding(verified_target_v2, signal_preflight)
    receipt = signal_preflight.receipt
    accepted_raw = tuple(receipt["events"])
    excluded_raw = tuple(receipt["exclusions"])
    if not accepted_raw:
        raise ValueError("Verified signal-preflight has no eligible events")

    accepted_candidates = {
        str(row["event_id"]): _timeline_candidate(row) for row in accepted_raw
    }
    excluded_candidates = {
        str(row["event_id"]): _timeline_candidate(row) for row in excluded_raw
    }
    if set(accepted_candidates) & set(excluded_candidates):
        raise ValueError("One event cannot be both eligible and excluded")
    candidates = {**accepted_candidates, **excluded_candidates}
    if len(candidates) != len(accepted_raw) + len(excluded_raw):
        raise ValueError("Verified global timeline contains duplicate event IDs")

    groups: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates.values():
        groups.setdefault(
            str(candidate["deepsoz_source_record_sha256"]), []
        ).append(candidate)
    timeline_sha_by_event: dict[str, str] = {}
    prior_max_stop_by_event: dict[str, float | None] = {}
    for source_record_sha, rows in groups.items():
        ordered = tuple(sorted(rows, key=lambda row: int(row["global_event_index"])))
        indices = tuple(int(row["global_event_index"]) for row in ordered)
        if indices != tuple(range(len(ordered))):
            raise ValueError(
                "Verified candidate union does not enumerate a complete global timeline"
            )
        identity_fields = (
            "patient_id",
            "local_patient_id",
            "official_split",
            "model_split",
            "relative_edf_path",
            "deepsoz_record",
            "annotation_pair_sha256",
        )
        for name in identity_fields:
            if len({str(row[name]) for row in ordered}) != 1:
                raise ValueError(f"Global timeline group changes {name}")
        starts = tuple(float(row["global_t0_sec"]) for row in ordered)
        if any(right < left - 1e-6 for left, right in zip(starts, starts[1:])):
            raise ValueError("Global event indices are not chronological")
        timeline_payload = {
            "schema_version": "soz_verified_record_local_tusz_timeline_v2",
            "timeline_scope": FORMAL_TIMELINE_SCOPE,
            "prior_stop_policy": FORMAL_PRIOR_STOP_POLICY,
            "signal_preflight_receipt_sha256": signal_preflight.receipt_sha256,
            "signal_preflight_artifact_sha256": signal_preflight.artifact_sha256,
            "deepsoz_source_record_sha256": source_record_sha,
            "annotation_pair_sha256": ordered[0]["annotation_pair_sha256"],
            "patient_id": ordered[0]["patient_id"],
            "model_split": ordered[0]["model_split"],
            "relative_edf_path": ordered[0]["relative_edf_path"],
            "events": [
                {
                    "event_id": row["event_id"],
                    "event_record_sha256": row["event_record_sha256"],
                    "global_event_index": row["global_event_index"],
                    "global_t0_sec": row["global_t0_sec"],
                    "global_stop_sec": row["global_stop_sec"],
                }
                for row in ordered
            ],
        }
        timeline_sha = _canonical_sha256(timeline_payload)
        running_prior_max_stop: float | None = None
        for row in ordered:
            event_id = str(row["event_id"])
            timeline_sha_by_event[event_id] = timeline_sha
            prior_max_stop_by_event[event_id] = running_prior_max_stop
            row_stop = float(row["global_stop_sec"])
            running_prior_max_stop = (
                row_stop
                if running_prior_max_stop is None
                else max(running_prior_max_stop, row_stop)
            )

    registry_records: list[EventInputRecord] = []
    timeline_events: list[VerifiedTimelineEvent] = []
    accepted_by_id = {str(row["event_id"]): row for row in accepted_raw}
    for event_id in sorted(accepted_by_id):
        raw = accepted_by_id[event_id]
        candidate = accepted_candidates[event_id]
        patient_id = str(candidate["patient_id"])
        reference = verified_target_v2.registry.get(patient_id)
        if not reference.eligible_for_localization:
            raise ValueError("Signal-preflight includes an ineligible target-v2 patient")
        if (
            reference.model_split != candidate["model_split"]
            or reference.official_split != candidate["official_split"]
        ):
            raise ValueError("Signal event and verified target-v2 split disagree")
        start = float(candidate["global_t0_sec"])
        stop = float(candidate["global_stop_sec"])
        window_start = _finite_float(
            raw["window_start_sec"], field_name="window_start_sec"
        )
        window_stop = _finite_float(
            raw["window_stop_sec"], field_name="window_stop_sec"
        )
        if (
            abs(window_start - (start - 12.0)) > 1e-6
            or abs(window_stop - (start + 48.0)) > 1e-6
        ):
            raise ValueError("Verified event crop drifted from fixed [-12,+48)")
        record = EventInputRecord(
            event_id=event_id,
            patient_id=patient_id,
            source=DEEPSOZ_SIGNAL_SOURCE,
            official_split=str(candidate["official_split"]),
            model_split=str(candidate["model_split"]),
            local_edf_path=str(candidate["relative_edf_path"]),
            t0_sec=start,
            window_start_sec=window_start,
            window_stop_sec=window_stop,
            record_sha256=str(candidate["event_record_sha256"]),
        )
        registry_records.append(record)
        # Historical public field names say ``previous``/``global``.  The
        # value is actually the cumulative maximum stop across all prior
        # events on this one record-local official annotation clock.
        previous_stop = prior_max_stop_by_event[event_id]
        previous_gap = (
            None if previous_stop is None else max(0.0, start - previous_stop)
        )
        timeline_events.append(
            VerifiedTimelineEvent(
                event_id=event_id,
                patient_id=patient_id,
                model_split=record.model_split,
                timeline_scope=FORMAL_TIMELINE_SCOPE,
                prior_stop_policy=FORMAL_PRIOR_STOP_POLICY,
                event_record_sha256=record.record_sha256,
                deepsoz_source_record_sha256=str(
                    candidate["deepsoz_source_record_sha256"]
                ),
                global_timeline_receipt_sha256=timeline_sha_by_event[event_id],
                global_event_index=int(candidate["global_event_index"]),
                global_t0_sec=start,
                global_stop_sec=stop,
                seizure_duration_sec=stop - start,
                previous_global_stop_sec=previous_stop,
                previous_seizure_gap_sec=previous_gap,
                previous_overlaps_current=(
                    previous_stop is not None
                    and previous_stop > start + OFFSET_TIME_TOLERANCE_SEC
                ),
                relative_edf_path=record.local_edf_path,
                preprocess_receipt_sha256=_require_sha256(
                    raw["preprocess_config_sha256"],
                    field_name="preprocess_config_sha256",
                ),
                processed_window_sha256=_require_sha256(
                    raw["processed_window_sha256"],
                    field_name="processed_window_sha256",
                ),
            )
        )

    event_registry = EventInputRegistry(
        registry_records,
        manifest_sha256=_require_sha256(
            receipt["event_inputs_sha256"], field_name="event_inputs_sha256"
        ),
        split_manifest_sha256=_require_sha256(
            receipt["split_manifest_sha256"], field_name="split_manifest_sha256"
        ),
    )
    event_ids = tuple(record.event_id for record in event_registry)
    timeline_by_id = {event.event_id: event for event in timeline_events}
    ordered_timeline = tuple(timeline_by_id[event_id] for event_id in event_ids)
    phase_masks = build_offset_aware_phase_masks(
        [event.seizure_duration_sec for event in ordered_timeline],
        offset_trustworthy=[True] * len(ordered_timeline),
        previous_seizure_gap_sec=[
            event.previous_seizure_gap_sec for event in ordered_timeline
        ],
        previous_timeline_trustworthy=[True] * len(ordered_timeline),
    )
    temporal_receipts = build_event_temporal_provenance_receipts(
        event_ids,
        phase_masks,
        global_timeline_receipt_sha256s=[
            event.global_timeline_receipt_sha256 for event in ordered_timeline
        ],
    )
    context = VerifiedGlobalTimelineContext(
        _verification_marker=_TIMELINE_MARKER,
        target_v2_receipt_sha256=verified_target_v2.receipt.receipt_sha256,
        signal_preflight_receipt_sha256=signal_preflight.receipt_sha256,
        signal_preflight_artifact_sha256=signal_preflight.artifact_sha256,
        event_registry=event_registry,
        timeline_events=ordered_timeline,
        phase_masks=phase_masks,
        temporal_receipts=temporal_receipts,
    )
    context.assert_unchanged()
    return context


def build_formal_oof_evidence_authorization(
    verified_target_v2: VerifiedDeepSOZTargetV2Artifact,
    timeline_context: VerifiedGlobalTimelineContext,
    *,
    active_families: Sequence[str],
    family_lineages: Sequence[FamilyProducerAuthorization],
) -> OOFEvidenceAuthorization:
    """Build OOF authority without accepting another registry or timeline."""

    if not isinstance(verified_target_v2, VerifiedDeepSOZTargetV2Artifact):
        raise TypeError("verified_target_v2 must be a verified target-v2 artifact")
    if not isinstance(timeline_context, VerifiedGlobalTimelineContext):
        raise TypeError("timeline_context must be a verified global timeline")
    timeline_context.assert_unchanged()
    if (
        verified_target_v2.receipt.receipt_sha256
        != timeline_context.target_v2_receipt_sha256
    ):
        raise ValueError("Target-v2 artifact was swapped after timeline verification")
    return build_oof_evidence_authorization(
        verified_target_v2.registry,
        timeline_context.event_registry,
        active_families=active_families,
        family_lineages=family_lineages,
        event_temporal_provenance=timeline_context.temporal_receipts,
    )


@dataclass(frozen=True, init=False)
class VerifiedMIVEvidenceEvent:
    """Reserved fused output of strict per-family producer replay.

    There is deliberately no constructor or generic issuer in the current
    implementation.  In particular, an arbitrary :class:`EvidenceBatch`
    cannot be converted into this capability and then given the lineage that
    an authorization *expected*.  A future implementation must consume one
    opaque output from every active family, independently cross-check each
    actual checkpoint/token/scaler/event/tensor receipt, and preserve an
    authenticated unavailable-reason receipt for any all-mask zero-filled
    family.  Until those family-specific issuers exist, publication is blocked.
    """

    event_id: str
    evidence: EvidenceBatch
    evidence_sha256: str
    timeline_context_sha256: str
    authorization_sha256: str
    producer_lineage_sha256s: tuple[str, ...]

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "VerifiedMIVEvidenceEvent has no generic issuer: strict family-specific "
            "producer replay capabilities are not implemented"
        )

    def assert_unchanged(self) -> None:
        if evidence_batch_sha256(self.evidence) != self.evidence_sha256:
            raise ValueError("Verified M/I/V evidence changed after producer issuance")


def _require_strict_family_producer_replay() -> None:
    """Fail closed until family-specific artifact replay issuers exist.

    ``active_families`` means that a producer is authorized, not that every
    event must contain an observed value.  The eventual issuers must therefore
    support both observed output and authenticated unavailable output.  The
    latter is finite zero-fill with an all-false family mask plus a replayed
    reason receipt; inventing a positive mask merely to pass a gate is invalid.
    """

    raise RuntimeError(
        "Formal evidence cache publication is disabled until strict "
        "family-specific producer replay issuers exist; unresolved="
        f"{_STRICT_FAMILY_PRODUCER_REPLAY_BLOCKERS}"
    )


def _cache_index_payload(
    *,
    target_v2_receipt_sha256: str,
    timeline_context_sha256: str,
    signal_preflight_receipt_sha256: str,
    authorization_sha256: str,
    model_split: str,
    patient_ids: Sequence[str],
    entries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    event_ids = tuple(str(entry["event_id"]) for entry in entries)
    return {
        "schema_version": FORMAL_EVIDENCE_CACHE_SET_SCHEMA,
        "serialization": "canonical_json_utf8_no_pickle",
        "target_values_present": False,
        "raw_eeg_present": False,
        "foundation_latents_present": False,
        "source_annotation_coverage_present": False,
        "target_v2_receipt_sha256": target_v2_receipt_sha256,
        "timeline_context_sha256": timeline_context_sha256,
        "signal_preflight_receipt_sha256": signal_preflight_receipt_sha256,
        "authorization_sha256": authorization_sha256,
        "model_split": model_split,
        "patient_ids": list(patient_ids),
        "patient_roster_sha256": patient_roster_sha256(patient_ids),
        "event_ids": list(event_ids),
        "event_roster_sha256": _canonical_sha256(event_ids),
        "event_count": len(event_ids),
        "events": [dict(entry) for entry in entries],
    }


def _safe_output_directory(path: str | Path) -> Path:
    target = Path(os.path.abspath(path))
    if target.name in {"", ".", ".."}:
        raise ValueError("Formal cache output requires a concrete directory")
    for component in (target.parent, *target.parent.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("Formal cache output parent cannot traverse symlinks")
    if not target.parent.is_dir():
        raise FileNotFoundError("Formal cache output parent does not exist")
    if os.path.lexists(target):
        raise FileExistsError("Formal cache output already exists")
    return target


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, init=False)
class VerifiedEvidenceCacheSet:
    """Opaque complete-split cache roster returned by publish/strict load."""

    path: Path
    index_sha256: str
    target_v2_receipt_sha256: str
    timeline_context_sha256: str
    signal_preflight_receipt_sha256: str
    authorization_sha256: str
    model_split: str
    patient_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    event_manifest_sha256s: tuple[tuple[str, str], ...]
    events: tuple[AuthorizedEvidenceEvent, ...] = field(repr=False)

    def __init__(
        self,
        *,
        _verification_marker: object,
        path: Path,
        index_sha256: str,
        target_v2_receipt_sha256: str,
        timeline_context_sha256: str,
        signal_preflight_receipt_sha256: str,
        authorization_sha256: str,
        model_split: str,
        patient_ids: Sequence[object],
        event_ids: Sequence[object],
        event_manifest_sha256s: Sequence[tuple[str, str]],
        events: Sequence[AuthorizedEvidenceEvent],
    ) -> None:
        if _verification_marker is not _CACHE_SET_MARKER:
            raise TypeError("VerifiedEvidenceCacheSet can only be issued by strict IO")
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("Formal cache-set path must be absolute")
        if model_split not in _TRAINING_CACHE_SPLITS:
            raise ValueError(
                "Training-time formal cache sets allow source_train/source_dev "
                "only; source_eval requires a one-shot frozen release capability"
            )
        patients = tuple(sorted(normalize_patient_id(value) for value in patient_ids))
        ids = tuple(str(value).strip() for value in event_ids)
        manifests = tuple((str(key).strip(), str(value).strip()) for key, value in event_manifest_sha256s)
        authorized = tuple(events)
        if not patients or len(set(patients)) != len(patients):
            raise ValueError("Cache-set patient roster must be non-empty and unique")
        if not ids or tuple(sorted(ids)) != ids or len(set(ids)) != len(ids):
            raise ValueError("Cache-set event roster must be sorted and unique")
        if tuple(key for key, _ in manifests) != ids:
            raise ValueError("Cache-set manifest roster does not equal event roster")
        if tuple(event.event.event_id for event in authorized) != ids:
            raise ValueError("Authorized event roster does not equal cache-set roster")
        if any(event.model_split != model_split for event in authorized):
            raise ValueError("Cache-set authorized events mix model splits")
        for _, manifest_sha in manifests:
            _require_sha256(manifest_sha, field_name="event_manifest_sha256")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "index_sha256",
            _require_sha256(index_sha256, field_name="index_sha256"),
        )
        for name, value in (
            ("target_v2_receipt_sha256", target_v2_receipt_sha256),
            ("timeline_context_sha256", timeline_context_sha256),
            ("signal_preflight_receipt_sha256", signal_preflight_receipt_sha256),
            ("authorization_sha256", authorization_sha256),
        ):
            object.__setattr__(
                self, name, _require_sha256(value, field_name=name)
            )
        object.__setattr__(self, "model_split", model_split)
        object.__setattr__(self, "patient_ids", patients)
        object.__setattr__(self, "event_ids", ids)
        object.__setattr__(self, "event_manifest_sha256s", manifests)
        object.__setattr__(self, "events", authorized)

    def assert_unchanged(
        self,
        verified_target_v2: VerifiedDeepSOZTargetV2Artifact,
        timeline_context: VerifiedGlobalTimelineContext,
        authorization: OOFEvidenceAuthorization,
    ) -> None:
        if not isinstance(verified_target_v2, VerifiedDeepSOZTargetV2Artifact):
            raise TypeError("verified_target_v2 must be verified target-v2")
        if not isinstance(timeline_context, VerifiedGlobalTimelineContext):
            raise TypeError("timeline_context must be verified")
        if not isinstance(authorization, OOFEvidenceAuthorization):
            raise TypeError("authorization must be OOFEvidenceAuthorization")
        timeline_context.assert_unchanged()
        if verified_target_v2.receipt.receipt_sha256 != self.target_v2_receipt_sha256:
            raise ValueError("Cache set belongs to another target-v2 artifact")
        if timeline_context.receipt_sha256 != self.timeline_context_sha256:
            raise ValueError("Cache set belongs to another verified timeline")
        if authorization.authorization_sha256 != self.authorization_sha256:
            raise ValueError("Cache set belongs to another OOF authorization")
        for event in self.events:
            authorize_evidence_event(
                event.event,
                verified_target_v2.registry,
                timeline_context.event_registry,
                authorization,
            )


def _validate_publication_inputs(
    verified_target_v2: VerifiedDeepSOZTargetV2Artifact,
    timeline_context: VerifiedGlobalTimelineContext,
    authorization: OOFEvidenceAuthorization,
    producer_events: Sequence[VerifiedMIVEvidenceEvent],
    *,
    model_split: str,
) -> tuple[VerifiedMIVEvidenceEvent, ...]:
    if model_split not in _TRAINING_CACHE_SPLITS:
        raise ValueError(
            "Training-time formal evidence publication allows source_train/"
            "source_dev only; source_eval/private release is forbidden"
        )
    if not isinstance(verified_target_v2, VerifiedDeepSOZTargetV2Artifact):
        raise TypeError("verified_target_v2 must be verified target-v2")
    if not isinstance(timeline_context, VerifiedGlobalTimelineContext):
        raise TypeError("timeline_context must be verified")
    if not isinstance(authorization, OOFEvidenceAuthorization):
        raise TypeError("authorization must be OOFEvidenceAuthorization")
    if any(not isinstance(event, VerifiedMIVEvidenceEvent) for event in producer_events):
        raise TypeError(
            "Formal cache publication requires opaque VerifiedMIVEvidenceEvent; "
            "direct EvidenceBatch injection is forbidden"
        )
    timeline_context.assert_unchanged()
    if verified_target_v2.receipt.receipt_sha256 != timeline_context.target_v2_receipt_sha256:
        raise ValueError("Target-v2 artifact was swapped after timeline verification")
    if authorization.event_registry_sha256 != timeline_context.event_registry.manifest_sha256:
        raise ValueError("OOF authorization belongs to another event registry")
    if authorization.split_manifest_sha256 != timeline_context.event_registry.split_manifest_sha256:
        raise ValueError("OOF authorization belongs to another split manifest")
    expected_ids = tuple(
        record.event_id
        for record in timeline_context.event_registry
        if record.model_split == model_split
    )
    by_id = {event.event_id: event for event in producer_events}
    if len(by_id) != len(producer_events) or set(by_id) != set(expected_ids):
        raise ValueError(
            "Formal evidence publication requires the complete registered split; "
            f"missing={sorted(set(expected_ids)-set(by_id))[:5]}, "
            f"extra={sorted(set(by_id)-set(expected_ids))[:5]}"
        )
    ordered = tuple(by_id[event_id] for event_id in expected_ids)
    for event in ordered:
        event.assert_unchanged()
        if event.timeline_context_sha256 != timeline_context.receipt_sha256:
            raise ValueError("Producer evidence belongs to another verified timeline")
        if event.authorization_sha256 != authorization.authorization_sha256:
            raise ValueError("Producer evidence belongs to another authorization")
        record = timeline_context.event_registry.get(event.event_id)
        key = (
            dict(authorization.source_train_patient_folds)[record.patient_id]
            if model_split == "source_train"
            else None
        )
        expected_lineages = tuple(
            authorization.lineage(
                family,
                authorization.lineage_key_for_event(family, key),
            ).lineage_sha256
            for family in authorization.active_families
        )
        if event.producer_lineage_sha256s != expected_lineages:
            raise ValueError("Producer evidence fold/checkpoint lineage changed")
        if not torch.equal(
            event.evidence.ictal_phase_mask[0],
            timeline_context.phase_mask(event.event_id),
        ):
            raise ValueError("Producer evidence temporal phase mask changed")
    return ordered


def publish_formal_evidence_cache_set(
    verified_target_v2: VerifiedDeepSOZTargetV2Artifact,
    timeline_context: VerifiedGlobalTimelineContext,
    authorization: OOFEvidenceAuthorization,
    producer_events: Sequence[VerifiedMIVEvidenceEvent],
    output_directory: str | Path,
    *,
    model_split: str,
) -> VerifiedEvidenceCacheSet:
    """Atomically publish one complete split without caller-supplied receipts."""

    if model_split not in _TRAINING_CACHE_SPLITS:
        raise ValueError(
            "Training-time formal evidence publication allows source_train/"
            "source_dev only; source_eval/private release is forbidden"
        )
    _require_strict_family_producer_replay()

    ordered = _validate_publication_inputs(
        verified_target_v2,
        timeline_context,
        authorization,
        producer_events,
        model_split=model_split,
    )
    target = _safe_output_directory(output_directory)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    published = False
    entries: list[dict[str, object]] = []
    try:
        for index, producer_event in enumerate(ordered):
            record = timeline_context.event_registry.get(producer_event.event_id)
            key = (
                dict(authorization.source_train_patient_folds)[record.patient_id]
                if model_split == "source_train"
                else None
            )
            receipt = EvidenceCacheReceipt(
                event_id=record.event_id,
                event_registry_sha256=timeline_context.event_registry.manifest_sha256,
                event_record_sha256=record.record_sha256,
                evidence_sha256=producer_event.evidence_sha256,
                extractors=tuple(
                    authorization.lineage(
                        family,
                        authorization.lineage_key_for_event(family, key),
                    ).extractor
                    for family in authorization.active_families
                ),
                authorization_sha256=authorization.authorization_sha256,
                temporal_provenance=timeline_context.temporal_receipt(record.event_id),
            )
            event = EvidenceEvent(record.event_id, producer_event.evidence, receipt)
            authorize_evidence_event(
                event,
                verified_target_v2.registry,
                timeline_context.event_registry,
                authorization,
            )
            directory_name = (
                f"event-{index:06d}-{hashlib.sha256(record.event_id.encode('utf-8')).hexdigest()[:16]}"
            )
            artifact = save_evidence_cache(
                staging / directory_name,
                producer_event.evidence,
                receipt,
            )
            entries.append(
                {
                    "event_id": record.event_id,
                    "event_record_sha256": record.record_sha256,
                    "cache_directory": directory_name,
                    "cache_manifest_sha256": artifact.manifest_sha256,
                    "evidence_sha256": producer_event.evidence_sha256,
                    "temporal_receipt_sha256": receipt.temporal_provenance.receipt_sha256,
                    "producer_lineage_sha256s": list(
                        producer_event.producer_lineage_sha256s
                    ),
                }
            )
        patient_ids = timeline_context.event_registry.patient_ids_for_split(model_split)
        payload = _cache_index_payload(
            target_v2_receipt_sha256=verified_target_v2.receipt.receipt_sha256,
            timeline_context_sha256=timeline_context.receipt_sha256,
            signal_preflight_receipt_sha256=timeline_context.signal_preflight_receipt_sha256,
            authorization_sha256=authorization.authorization_sha256,
            model_split=model_split,
            patient_ids=patient_ids,
            entries=entries,
        )
        encoded = _canonical_json_bytes(payload)
        if len(encoded) > _MAX_INDEX_BYTES:
            raise ValueError("Formal evidence cache index exceeds size limit")
        index_path = staging / FORMAL_EVIDENCE_CACHE_INDEX_FILENAME
        with index_path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(staging)
        if os.path.lexists(target):
            raise FileExistsError("Formal cache output already exists")
        os.rename(staging, target)
        published = True
        _fsync_directory(target.parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return load_formal_evidence_cache_set(
        target,
        verified_target_v2,
        timeline_context,
        authorization,
        expected_index_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _closed_index(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("Formal evidence cache index must be a JSON object")
    expected = {
        "schema_version",
        "serialization",
        "target_values_present",
        "raw_eeg_present",
        "foundation_latents_present",
        "source_annotation_coverage_present",
        "target_v2_receipt_sha256",
        "timeline_context_sha256",
        "signal_preflight_receipt_sha256",
        "authorization_sha256",
        "model_split",
        "patient_ids",
        "patient_roster_sha256",
        "event_ids",
        "event_roster_sha256",
        "event_count",
        "events",
    }
    if set(payload) != expected:
        raise ValueError("Formal evidence cache index violates its closed schema")
    if payload["schema_version"] != FORMAL_EVIDENCE_CACHE_SET_SCHEMA:
        raise ValueError("Unsupported formal evidence cache-set schema")
    if payload["serialization"] != "canonical_json_utf8_no_pickle":
        raise ValueError("Formal evidence cache index uses unsafe serialization")
    for field_name in (
        "target_values_present",
        "raw_eeg_present",
        "foundation_latents_present",
        "source_annotation_coverage_present",
    ):
        if payload[field_name] is not False:
            raise ValueError(f"Forbidden cache payload declared: {field_name}")
    return payload


def load_formal_evidence_cache_set(
    output_directory: str | Path,
    verified_target_v2: VerifiedDeepSOZTargetV2Artifact,
    timeline_context: VerifiedGlobalTimelineContext,
    authorization: OOFEvidenceAuthorization,
    *,
    expected_index_sha256: str,
) -> VerifiedEvidenceCacheSet:
    """Strictly load an externally pinned complete cache-set index."""

    _require_strict_family_producer_replay()

    root = Path(os.path.abspath(output_directory))
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("Formal evidence cache set must be a canonical directory")
    index_path = root / FORMAL_EVIDENCE_CACHE_INDEX_FILENAME
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("Formal evidence cache set lacks a regular index")
    raw = index_path.read_bytes()
    if not 1 <= len(raw) <= _MAX_INDEX_BYTES:
        raise ValueError("Formal evidence cache index size is invalid")
    actual_index_sha = hashlib.sha256(raw).hexdigest()
    if actual_index_sha != _require_sha256(
        expected_index_sha256, field_name="expected_index_sha256"
    ):
        raise ValueError("Formal evidence cache index SHA-256 mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Formal evidence cache index is not UTF-8 JSON") from exc
    if _canonical_json_bytes(payload) != raw:
        raise ValueError("Formal evidence cache index is not canonical JSON")
    payload = _closed_index(payload)
    if payload["target_v2_receipt_sha256"] != verified_target_v2.receipt.receipt_sha256:
        raise ValueError("Cache index belongs to another target-v2 artifact")
    if payload["timeline_context_sha256"] != timeline_context.receipt_sha256:
        raise ValueError("Cache index belongs to another verified timeline")
    if payload["signal_preflight_receipt_sha256"] != timeline_context.signal_preflight_receipt_sha256:
        raise ValueError("Cache index belongs to another signal-preflight replay")
    if payload["authorization_sha256"] != authorization.authorization_sha256:
        raise ValueError("Cache index belongs to another OOF authorization")
    model_split = str(payload["model_split"])
    if model_split not in _TRAINING_CACHE_SPLITS:
        raise ValueError(
            "Training-time cache index cannot contain source_eval/private evidence"
        )
    patient_ids = tuple(str(value) for value in payload["patient_ids"])
    expected_patients = timeline_context.event_registry.patient_ids_for_split(model_split)
    if patient_ids != expected_patients or payload["patient_roster_sha256"] != patient_roster_sha256(patient_ids):
        raise ValueError("Cache index patient roster is incomplete or changed")
    event_ids = tuple(str(value) for value in payload["event_ids"])
    expected_ids = tuple(
        record.event_id
        for record in timeline_context.event_registry
        if record.model_split == model_split
    )
    if event_ids != expected_ids or payload["event_roster_sha256"] != _canonical_sha256(event_ids):
        raise ValueError("Cache index event roster is incomplete or changed")
    if payload["event_count"] != len(event_ids):
        raise ValueError("Cache index event count changed")
    event_rows = payload["events"]
    if not isinstance(event_rows, list) or len(event_rows) != len(event_ids):
        raise ValueError("Cache index event rows do not equal its roster")
    expected_event_fields = {
        "event_id",
        "event_record_sha256",
        "cache_directory",
        "cache_manifest_sha256",
        "evidence_sha256",
        "temporal_receipt_sha256",
        "producer_lineage_sha256s",
    }
    loaded: list[AuthorizedEvidenceEvent] = []
    manifests: list[tuple[str, str]] = []
    expected_entries = {FORMAL_EVIDENCE_CACHE_INDEX_FILENAME}
    for event_id, row in zip(event_ids, event_rows):
        if not isinstance(row, dict) or set(row) != expected_event_fields:
            raise ValueError("Cache index event row violates its closed schema")
        if row["event_id"] != event_id:
            raise ValueError("Cache index event row order changed")
        record = timeline_context.event_registry.get(event_id)
        if row["event_record_sha256"] != record.record_sha256:
            raise ValueError("Cache index event-record SHA changed")
        directory_name = str(row["cache_directory"])
        if not re.fullmatch(r"event-[0-9]{6}-[0-9a-f]{16}", directory_name):
            raise ValueError("Cache index event directory is not canonical")
        expected_entries.add(directory_name)
        manifest_sha = _require_sha256(
            row["cache_manifest_sha256"], field_name="cache_manifest_sha256"
        )
        event = load_authorized_evidence_cache(
            str(root / directory_name),
            verified_target_v2.registry,
            timeline_context.event_registry,
            authorization,
            expected_manifest_sha256=manifest_sha,
        )
        if event.event.cache_receipt.evidence_sha256 != row["evidence_sha256"]:
            raise ValueError("Cache index evidence SHA changed")
        temporal = event.event.cache_receipt.temporal_provenance
        if temporal is None or temporal.receipt_sha256 != row["temporal_receipt_sha256"]:
            raise ValueError("Cache index temporal receipt changed")
        record_key = (
            dict(authorization.source_train_patient_folds)[record.patient_id]
            if model_split == "source_train"
            else None
        )
        expected_lineages = [
            authorization.lineage(family, record_key).lineage_sha256
            for family in authorization.active_families
        ]
        if row["producer_lineage_sha256s"] != expected_lineages:
            raise ValueError("Cache index producer lineage changed")
        loaded.append(event)
        manifests.append((event_id, manifest_sha))
    if {item.name for item in root.iterdir()} != expected_entries:
        raise ValueError("Formal evidence cache set contains missing or unknown entries")
    cache_set = VerifiedEvidenceCacheSet(
        _verification_marker=_CACHE_SET_MARKER,
        path=root,
        index_sha256=actual_index_sha,
        target_v2_receipt_sha256=verified_target_v2.receipt.receipt_sha256,
        timeline_context_sha256=timeline_context.receipt_sha256,
        signal_preflight_receipt_sha256=timeline_context.signal_preflight_receipt_sha256,
        authorization_sha256=authorization.authorization_sha256,
        model_split=model_split,
        patient_ids=patient_ids,
        event_ids=event_ids,
        event_manifest_sha256s=manifests,
        events=loaded,
    )
    cache_set.assert_unchanged(verified_target_v2, timeline_context, authorization)
    return cache_set


def _reasoner_dataset_payload(
    *,
    target_v2_receipt_sha256: str,
    timeline_context_sha256: str,
    authorization_sha256: str,
    cache_set_index_sha256: str,
    model_split: str,
    patient_ids: Sequence[str],
    event_ids: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": FORMAL_REASONER_DATASET_SCHEMA,
        "target_v2_receipt_sha256": target_v2_receipt_sha256,
        "timeline_context_sha256": timeline_context_sha256,
        "authorization_sha256": authorization_sha256,
        "cache_set_index_sha256": cache_set_index_sha256,
        "model_split": model_split,
        "patient_ids": list(patient_ids),
        "patient_roster_sha256": patient_roster_sha256(patient_ids),
        "event_ids": list(event_ids),
        "event_roster_sha256": _canonical_sha256(tuple(event_ids)),
        "target_source": "verified_deepsoz_target_v2_only",
        "evidence_source": "authorized_complete_cache_set_only",
    }


@dataclass(frozen=True, init=False)
class VerifiedReasonerDatasetBundle:
    """Opaque train/dev target join; source-eval construction is forbidden."""

    dataset: AuthorizedPatientBagDataset = field(repr=False)
    target_v2_receipt_sha256: str
    timeline_context_sha256: str
    authorization_sha256: str
    cache_set_index_sha256: str
    model_split: str
    patient_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    receipt_sha256: str
    _verified_target_v2: VerifiedDeepSOZTargetV2Artifact = field(repr=False)
    _timeline_context: VerifiedGlobalTimelineContext = field(repr=False)
    _authorization: OOFEvidenceAuthorization = field(repr=False)
    _cache_set: VerifiedEvidenceCacheSet = field(repr=False)

    def __init__(
        self,
        *,
        _verification_marker: object,
        dataset: AuthorizedPatientBagDataset,
        verified_target_v2: VerifiedDeepSOZTargetV2Artifact,
        timeline_context: VerifiedGlobalTimelineContext,
        authorization: OOFEvidenceAuthorization,
        cache_set: VerifiedEvidenceCacheSet,
    ) -> None:
        if _verification_marker is not _DATASET_MARKER:
            raise TypeError(
                "VerifiedReasonerDatasetBundle can only be issued by strict join"
            )
        if not isinstance(dataset, AuthorizedPatientBagDataset):
            raise TypeError("dataset must be AuthorizedPatientBagDataset")
        if dataset.model_split not in _REASONER_TARGET_SPLITS:
            raise ValueError("Reasoner target joins are restricted to source_train/dev")
        patients = tuple(dataset.patient_ids)
        events = tuple(cache_set.event_ids)
        payload = _reasoner_dataset_payload(
            target_v2_receipt_sha256=verified_target_v2.receipt.receipt_sha256,
            timeline_context_sha256=timeline_context.receipt_sha256,
            authorization_sha256=authorization.authorization_sha256,
            cache_set_index_sha256=cache_set.index_sha256,
            model_split=dataset.model_split,
            patient_ids=patients,
            event_ids=events,
        )
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(
            self,
            "target_v2_receipt_sha256",
            verified_target_v2.receipt.receipt_sha256,
        )
        object.__setattr__(self, "timeline_context_sha256", timeline_context.receipt_sha256)
        object.__setattr__(self, "authorization_sha256", authorization.authorization_sha256)
        object.__setattr__(self, "cache_set_index_sha256", cache_set.index_sha256)
        object.__setattr__(self, "model_split", dataset.model_split)
        object.__setattr__(self, "patient_ids", patients)
        object.__setattr__(self, "event_ids", events)
        object.__setattr__(self, "receipt_sha256", _canonical_sha256(payload))
        object.__setattr__(self, "_verified_target_v2", verified_target_v2)
        object.__setattr__(self, "_timeline_context", timeline_context)
        object.__setattr__(self, "_authorization", authorization)
        object.__setattr__(self, "_cache_set", cache_set)

    def assert_unchanged(self) -> None:
        self._cache_set.assert_unchanged(
            self._verified_target_v2,
            self._timeline_context,
            self._authorization,
        )
        if self.dataset.model_split != self.model_split:
            raise ValueError("Verified reasoner dataset split changed")
        if tuple(self.dataset.patient_ids) != self.patient_ids:
            raise ValueError("Verified reasoner dataset patient roster changed")
        payload = _reasoner_dataset_payload(
            target_v2_receipt_sha256=self.target_v2_receipt_sha256,
            timeline_context_sha256=self.timeline_context_sha256,
            authorization_sha256=self.authorization_sha256,
            cache_set_index_sha256=self.cache_set_index_sha256,
            model_split=self.model_split,
            patient_ids=self.patient_ids,
            event_ids=self.event_ids,
        )
        if _canonical_sha256(payload) != self.receipt_sha256:
            raise ValueError("Verified reasoner dataset receipt changed")


def build_verified_reasoner_dataset(
    verified_target_v2: VerifiedDeepSOZTargetV2Artifact,
    timeline_context: VerifiedGlobalTimelineContext,
    authorization: OOFEvidenceAuthorization,
    cache_set: VerifiedEvidenceCacheSet,
    *,
    model_split: str,
) -> VerifiedReasonerDatasetBundle:
    """Join authorized caches to target-v2 without accepting another target."""

    if model_split not in _REASONER_TARGET_SPLITS:
        raise ValueError(
            "Formal reasoner datasets allow source_train/source_dev only; "
            "source_eval and private targets are never opened here"
        )
    if not isinstance(cache_set, VerifiedEvidenceCacheSet):
        raise TypeError("cache_set must be a strictly verified cache-set capability")
    if cache_set.model_split != model_split:
        raise ValueError("Cache-set split differs from requested reasoner split")
    if verified_target_v2.receipt.receipt_sha256 != timeline_context.target_v2_receipt_sha256:
        raise ValueError("Verified target-v2 artifact was swapped")
    cache_set.assert_unchanged(verified_target_v2, timeline_context, authorization)
    dataset = AuthorizedPatientBagDataset(
        cache_set.events,
        verified_target_v2.registry,
        timeline_context.event_registry,
        authorization,
        expected_model_split=model_split,
    )
    expected_patients = timeline_context.event_registry.patient_ids_for_split(model_split)
    if tuple(dataset.patient_ids) != expected_patients:
        raise ValueError("Reasoner dataset does not cover the complete verified split")
    bundle = VerifiedReasonerDatasetBundle(
        _verification_marker=_DATASET_MARKER,
        dataset=dataset,
        verified_target_v2=verified_target_v2,
        timeline_context=timeline_context,
        authorization=authorization,
        cache_set=cache_set,
    )
    bundle.assert_unchanged()
    return bundle


@dataclass(frozen=True)
class FormalReasonerFitConfig:
    """Unique pre-registered small-reasoner optimization policy.

    The type remains in receipts for auditability, but it is not a tuning
    surface: every field must equal the one protocol-registered value.
    """

    seed: int = 20260808
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    hidden_dim: int = 16
    ranking_weight: float = 0.25
    ranking_margin: float = 0.0
    optimizer: str = "AdamW"
    patient_order_policy: str = "deterministic_complete_roster_per_epoch"
    calibration_during_fit: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Reasoner seed must be a non-negative integer")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or not 1 <= self.epochs <= 10_000:
            raise ValueError("Reasoner epochs must lie in [1,10000]")
        for name in ("learning_rate", "weight_decay", "max_grad_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.hidden_dim != 16:
            raise ValueError("Formal reasoner hidden_dim is frozen at 16")
        if self.ranking_weight != 0.25 or self.ranking_margin != 0.0:
            raise ValueError("Formal reasoner objective policy cannot change")
        if self.optimizer != "AdamW":
            raise ValueError("Formal reasoner optimizer is frozen to AdamW")
        if self.patient_order_policy != "deterministic_complete_roster_per_epoch":
            raise ValueError("Formal reasoner patient-order policy cannot change")
        if self.calibration_during_fit is not False:
            raise ValueError("Calibration during reasoner fitting is forbidden")
        if asdict(self) != _FORMAL_REASONER_FIT_POLICY:
            raise ValueError(
                "Formal reasoner optimization is frozen to the unique "
                "pre-registered policy"
            )

    @property
    def receipt_sha256(self) -> str:
        receipt = _canonical_sha256(asdict(self))
        if receipt != FORMAL_REASONER_FIT_POLICY_SHA256:
            raise RuntimeError("Formal reasoner policy digest drifted")
        return receipt


@dataclass(frozen=True)
class FormalReasonerEpochReceipt:
    epoch_index: int
    patient_order_sha256: str
    mean_total_loss: float
    mean_bce_loss: float
    mean_ranking_loss: float
    patient_count: int
    event_count: int

    def __post_init__(self) -> None:
        if isinstance(self.epoch_index, bool) or not isinstance(self.epoch_index, int) or self.epoch_index < 0:
            raise ValueError("epoch_index must be a non-negative integer")
        _require_sha256(self.patient_order_sha256, field_name="patient_order_sha256")
        for name in ("mean_total_loss", "mean_bce_loss", "mean_ranking_loss"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.patient_count < 1 or self.event_count < 1:
            raise ValueError("Reasoner epoch receipt requires patients and events")


@dataclass(frozen=True)
class FormalReasonerFitReceipt:
    target_v2_receipt_sha256: str
    timeline_context_sha256: str
    authorization_sha256: str
    source_train_dataset_receipt_sha256: str
    source_train_patient_ids: tuple[str, ...]
    source_train_roster_sha256: str
    config: FormalReasonerFitConfig
    config_sha256: str
    initial_state_sha256: str
    final_state_sha256: str
    parameter_count: int
    epochs: tuple[FormalReasonerEpochReceipt, ...]
    schema_version: str = FORMAL_REASONER_FIT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "target_v2_receipt_sha256",
            "timeline_context_sha256",
            "authorization_sha256",
            "source_train_dataset_receipt_sha256",
            "source_train_roster_sha256",
            "config_sha256",
            "initial_state_sha256",
            "final_state_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), field_name=name)
            )
        roster = tuple(sorted(normalize_patient_id(value) for value in self.source_train_patient_ids))
        if not roster or len(set(roster)) != len(roster):
            raise ValueError("Reasoner fit requires a unique source-train roster")
        object.__setattr__(self, "source_train_patient_ids", roster)
        if self.source_train_roster_sha256 != patient_roster_sha256(roster):
            raise ValueError("Reasoner fit source-train roster SHA mismatch")
        if not isinstance(self.config, FormalReasonerFitConfig):
            raise TypeError("Reasoner fit config has the wrong type")
        if self.config_sha256 != self.config.receipt_sha256:
            raise ValueError("Reasoner fit config SHA mismatch")
        if len(self.epochs) != self.config.epochs or tuple(
            receipt.epoch_index for receipt in self.epochs
        ) != tuple(range(self.config.epochs)):
            raise ValueError("Reasoner fit epoch ledger is incomplete")
        if self.parameter_count < 1 or self.parameter_count >= 50_000:
            raise ValueError("Reasoner parameter count violates the capacity gate")
        if self.schema_version != FORMAL_REASONER_FIT_SCHEMA:
            raise ValueError("Unsupported formal reasoner fit schema")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True, init=False)
class VerifiedReasonerFit:
    """Opaque trained state whose run receipt was derived, never injected."""

    model: AdditiveEvidenceReasoner = field(repr=False)
    receipt: FormalReasonerFitReceipt

    def __init__(
        self,
        *,
        _verification_marker: object,
        model: AdditiveEvidenceReasoner,
        receipt: FormalReasonerFitReceipt,
    ) -> None:
        if _verification_marker is not _FIT_MARKER:
            raise TypeError("VerifiedReasonerFit can only be issued by formal fitter")
        if not isinstance(model, AdditiveEvidenceReasoner):
            raise TypeError("model must be AdditiveEvidenceReasoner")
        if not isinstance(receipt, FormalReasonerFitReceipt):
            raise TypeError("receipt must be FormalReasonerFitReceipt")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "receipt", receipt)
        self.assert_unchanged()

    def assert_unchanged(self) -> None:
        if self.model.training:
            raise ValueError("Completed reasoner fit must remain in eval mode")
        if reasoner_state_sha256(self.model) != self.receipt.final_state_sha256:
            raise ValueError("Reasoner state changed after formal fitting")


def _epoch_patient_order(
    patient_ids: Sequence[str], *, seed: int, epoch: int
) -> tuple[str, ...]:
    order = list(patient_ids)
    random.Random((seed << 20) ^ epoch).shuffle(order)
    return tuple(order)


def fit_verified_reasoner(
    training_bundle: VerifiedReasonerDatasetBundle,
    *,
    device: str | torch.device = "cpu",
) -> VerifiedReasonerFit:
    """Only formal fit entry: construct the pre-registered fit internally."""

    if not isinstance(training_bundle, VerifiedReasonerDatasetBundle):
        raise TypeError("training_bundle must be a verified reasoner dataset")
    if training_bundle.model_split != "source_train":
        raise ValueError("Reasoner fitting is restricted to source_train")
    config = FormalReasonerFitConfig()
    execution_device = torch.device(device)
    if execution_device.type not in {"cpu", "cuda"}:
        raise ValueError("Reasoner fit device must be cpu or cuda")
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    training_bundle.assert_unchanged()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = AdditiveEvidenceReasoner(hidden_dim=config.hidden_dim).to(execution_device)
    objective = PatientLevelSOZObjective(
        ranking_weight=config.ranking_weight,
        ranking_margin=config.ranking_margin,
        require_positive=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    initial_sha = reasoner_state_sha256(model)
    epoch_receipts: list[FormalReasonerEpochReceipt] = []
    for epoch in range(config.epochs):
        training_bundle.assert_unchanged()
        order = _epoch_patient_order(
            training_bundle.patient_ids, seed=config.seed, epoch=epoch
        )
        output: ReasonerEpochOutput = train_formal_reasoner_epoch(
            model,
            training_bundle.dataset,
            optimizer,
            objective,
            patient_order=order,
            max_grad_norm=config.max_grad_norm,
        )
        if output.patient_ids != order:
            raise RuntimeError("Reasoner epoch did not preserve its complete patient order")
        epoch_receipts.append(
            FormalReasonerEpochReceipt(
                epoch_index=epoch,
                patient_order_sha256=_canonical_sha256(order),
                mean_total_loss=output.mean_total_loss,
                mean_bce_loss=output.mean_bce_loss,
                mean_ranking_loss=output.mean_ranking_loss,
                patient_count=output.n_patients,
                event_count=output.n_events,
            )
        )
    model.eval()
    final_sha = reasoner_state_sha256(model)
    receipt = FormalReasonerFitReceipt(
        target_v2_receipt_sha256=training_bundle.target_v2_receipt_sha256,
        timeline_context_sha256=training_bundle.timeline_context_sha256,
        authorization_sha256=training_bundle.authorization_sha256,
        source_train_dataset_receipt_sha256=training_bundle.receipt_sha256,
        source_train_patient_ids=training_bundle.patient_ids,
        source_train_roster_sha256=patient_roster_sha256(training_bundle.patient_ids),
        config=config,
        config_sha256=config.receipt_sha256,
        initial_state_sha256=initial_sha,
        final_state_sha256=final_sha,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        epochs=tuple(epoch_receipts),
    )
    return VerifiedReasonerFit(
        _verification_marker=_FIT_MARKER,
        model=model,
        receipt=receipt,
    )


@dataclass(frozen=True, init=False)
class VerifiedFrozenReasoner:
    """Opaque freeze boundary binding fit, train roster, and exact dev cache."""

    checkpoint: FrozenReasonerCheckpoint = field(repr=False)
    fit_receipt_sha256: str
    target_v2_receipt_sha256: str
    timeline_context_sha256: str
    authorization_sha256: str
    source_dev_dataset_receipt_sha256: str
    source_dev_cache_set_index_sha256: str
    receipt_sha256: str

    def __init__(
        self,
        *,
        _verification_marker: object,
        checkpoint: FrozenReasonerCheckpoint,
        fit_receipt_sha256: str,
        target_v2_receipt_sha256: str,
        timeline_context_sha256: str,
        authorization_sha256: str,
        source_dev_dataset_receipt_sha256: str,
        source_dev_cache_set_index_sha256: str,
    ) -> None:
        if _verification_marker is not _FROZEN_MARKER:
            raise TypeError(
                "VerifiedFrozenReasoner can only be issued by formal freezer"
            )
        if not isinstance(checkpoint, FrozenReasonerCheckpoint):
            raise TypeError("checkpoint must be FrozenReasonerCheckpoint")
        values = {
            "fit_receipt_sha256": fit_receipt_sha256,
            "target_v2_receipt_sha256": target_v2_receipt_sha256,
            "timeline_context_sha256": timeline_context_sha256,
            "authorization_sha256": authorization_sha256,
            "source_dev_dataset_receipt_sha256": source_dev_dataset_receipt_sha256,
            "source_dev_cache_set_index_sha256": source_dev_cache_set_index_sha256,
        }
        for name, value in values.items():
            values[name] = _require_sha256(value, field_name=name)
        payload = {
            "schema_version": FORMAL_FROZEN_REASONER_SCHEMA,
            "frozen_checkpoint_receipt_sha256": checkpoint.receipt.receipt_sha256,
            **values,
        }
        object.__setattr__(self, "checkpoint", checkpoint)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "receipt_sha256", _canonical_sha256(payload))
        self.assert_unchanged()

    def assert_unchanged(self) -> None:
        self.checkpoint.assert_unchanged()
        payload = {
            "schema_version": FORMAL_FROZEN_REASONER_SCHEMA,
            "frozen_checkpoint_receipt_sha256": self.checkpoint.receipt.receipt_sha256,
            "fit_receipt_sha256": self.fit_receipt_sha256,
            "target_v2_receipt_sha256": self.target_v2_receipt_sha256,
            "timeline_context_sha256": self.timeline_context_sha256,
            "authorization_sha256": self.authorization_sha256,
            "source_dev_dataset_receipt_sha256": self.source_dev_dataset_receipt_sha256,
            "source_dev_cache_set_index_sha256": self.source_dev_cache_set_index_sha256,
        }
        if _canonical_sha256(payload) != self.receipt_sha256:
            raise ValueError("Verified frozen-reasoner receipt changed")


def freeze_verified_reasoner(
    fitted_reasoner: VerifiedReasonerFit,
    development_bundle: VerifiedReasonerDatasetBundle,
) -> VerifiedFrozenReasoner:
    """Freeze using only internally derived run/auth/train/dev receipts."""

    if not isinstance(fitted_reasoner, VerifiedReasonerFit):
        raise TypeError("fitted_reasoner must be VerifiedReasonerFit")
    if not isinstance(development_bundle, VerifiedReasonerDatasetBundle):
        raise TypeError("development_bundle must be VerifiedReasonerDatasetBundle")
    fitted_reasoner.assert_unchanged()
    development_bundle.assert_unchanged()
    if development_bundle.model_split != "source_dev":
        raise ValueError("Reasoner freeze requires the complete source_dev bundle")
    fit = fitted_reasoner.receipt
    checks = {
        "target-v2": (
            fit.target_v2_receipt_sha256
            == development_bundle.target_v2_receipt_sha256
        ),
        "verified timeline": (
            fit.timeline_context_sha256
            == development_bundle.timeline_context_sha256
        ),
        "evidence authorization": (
            fit.authorization_sha256 == development_bundle.authorization_sha256
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Development bundle was swapped before freeze: {failed}")
    checkpoint = freeze_reasoner_checkpoint(
        fitted_reasoner.model,
        training_run_receipt_sha256=fit.receipt_sha256,
        evidence_authorization_sha256=fit.authorization_sha256,
        source_train_patient_ids=fit.source_train_patient_ids,
        source_dev_patient_ids=development_bundle.patient_ids,
    )
    return VerifiedFrozenReasoner(
        _verification_marker=_FROZEN_MARKER,
        checkpoint=checkpoint,
        fit_receipt_sha256=fit.receipt_sha256,
        target_v2_receipt_sha256=fit.target_v2_receipt_sha256,
        timeline_context_sha256=fit.timeline_context_sha256,
        authorization_sha256=fit.authorization_sha256,
        source_dev_dataset_receipt_sha256=development_bundle.receipt_sha256,
        source_dev_cache_set_index_sha256=development_bundle.cache_set_index_sha256,
    )


def build_verified_reasoner_calibration_data(
    frozen_reasoner: VerifiedFrozenReasoner,
    development_bundle: VerifiedReasonerDatasetBundle,
) -> ReasonerCalibrationData:
    """Run frozen reasoner on complete source-dev and derive every tensor/SHA.

    The signature deliberately has no logits, targets, masks, patient IDs, or
    receipt SHA parameters.  All are reconstructed from the frozen reasoner,
    the verified target-v2 registry, and the authorized complete dev bags.
    """

    if not isinstance(frozen_reasoner, VerifiedFrozenReasoner):
        raise TypeError("frozen_reasoner must be VerifiedFrozenReasoner")
    if not isinstance(development_bundle, VerifiedReasonerDatasetBundle):
        raise TypeError("development_bundle must be VerifiedReasonerDatasetBundle")
    frozen_reasoner.assert_unchanged()
    development_bundle.assert_unchanged()
    if development_bundle.model_split != "source_dev":
        raise ValueError(
            "Calibration data are restricted to source_dev; source_eval/private forbidden"
        )
    checks = {
        "target-v2": (
            frozen_reasoner.target_v2_receipt_sha256
            == development_bundle.target_v2_receipt_sha256
        ),
        "verified timeline": (
            frozen_reasoner.timeline_context_sha256
            == development_bundle.timeline_context_sha256
        ),
        "evidence authorization": (
            frozen_reasoner.authorization_sha256
            == development_bundle.authorization_sha256
        ),
        "development dataset": (
            frozen_reasoner.source_dev_dataset_receipt_sha256
            == development_bundle.receipt_sha256
        ),
        "development cache set": (
            frozen_reasoner.source_dev_cache_set_index_sha256
            == development_bundle.cache_set_index_sha256
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Development evidence was swapped after freeze: {failed}")

    model = frozen_reasoner.checkpoint.model
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:  # pragma: no cover - reasoner always has parameters
        raise RuntimeError("Frozen reasoner has no parameters") from exc
    patient_ids: list[str] = []
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    aggregation_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for raw_batch in development_bundle.dataset.iter_epoch(
            development_bundle.patient_ids
        ):
            if len(raw_batch.patient_ids) != 1:
                raise RuntimeError("Formal dev dataset must yield one complete patient bag")
            batch = raw_batch.to(device)
            reasoner = model(batch.evidence)
            aggregation = batch.aggregate(reasoner.event_logits)
            patient_ids.extend(batch.patient_ids)
            logits.append(aggregation.logits.detach().cpu())
            targets.append(batch.targets.detach().cpu())
            masks.append(batch.target_mask.detach().cpu())
            aggregation_rows.append(
                {
                    "patient_id": batch.patient_ids[0],
                    "event_ids": list(batch.event_ids),
                    "event_count": int(aggregation.event_counts[0].item()),
                }
            )
    roster = tuple(patient_ids)
    if roster != development_bundle.patient_ids:
        raise RuntimeError("Source-dev forward pass did not preserve canonical roster")
    raw_logits = torch.cat(logits, dim=0)
    target_values = torch.cat(targets, dim=0)
    target_mask = torch.cat(masks, dim=0)
    aggregation_receipt_sha = _canonical_sha256(
        {
            "schema_version": "soz_verified_source_dev_patient_aggregation_v1",
            "frozen_reasoner_receipt_sha256": frozen_reasoner.receipt_sha256,
            "development_dataset_receipt_sha256": development_bundle.receipt_sha256,
            "policy": "equal_event_mean_raw_logits",
            "patients": aggregation_rows,
            "raw_logits_sha256": _tensor_sha256("raw_patient_logits", raw_logits),
        }
    )
    data = build_reasoner_calibration_data(
        frozen_reasoner.checkpoint,
        patient_ids=roster,
        raw_patient_logits=raw_logits,
        targets=target_values,
        target_mask=target_mask,
        evidence_authorization_sha256=development_bundle.authorization_sha256,
        verified_target_v2_receipt_sha256=(
            development_bundle.target_v2_receipt_sha256
        ),
        authorized_dev_cache_receipt_sha256=(
            development_bundle.cache_set_index_sha256
        ),
        patient_aggregation_receipt_sha256=aggregation_receipt_sha,
    )
    frozen_reasoner.assert_unchanged()
    return data


def fit_verified_global_affine_calibrator(
    frozen_reasoner: VerifiedFrozenReasoner,
    development_bundle: VerifiedReasonerDatasetBundle,
) -> GlobalAffineSOZCalibrator:
    """Fit the unique pre-registered calibrator on internally built dev data."""

    data = build_verified_reasoner_calibration_data(
        frozen_reasoner, development_bundle
    )
    calibrator = fit_global_affine_calibrator(
        frozen_reasoner.checkpoint,
        data,
        max_steps=_FORMAL_CALIBRATOR_MAX_STEPS,
        learning_rate=_FORMAL_CALIBRATOR_LEARNING_RATE,
    )
    receipt = calibrator.receipt
    policy_sha = _canonical_sha256(
        {
            "optimizer": "Adam",
            "execution": "cpu_float64",
            "max_steps": receipt.optimizer_steps,
            "learning_rate": receipt.optimizer_learning_rate,
            "parameterization": "global_log_temperature_and_bias",
        }
    )
    if policy_sha != FORMAL_CALIBRATOR_FIT_POLICY_SHA256:
        raise RuntimeError("Formal calibrator optimization policy drifted")
    return calibrator


__all__ = [
    "FORMAL_CALIBRATOR_FIT_POLICY_SHA256",
    "FORMAL_EVIDENCE_CACHE_INDEX_FILENAME",
    "FORMAL_EVIDENCE_CACHE_SET_SCHEMA",
    "FORMAL_FROZEN_REASONER_SCHEMA",
    "FORMAL_REASONER_DATASET_SCHEMA",
    "FORMAL_REASONER_FIT_SCHEMA",
    "FORMAL_REASONER_FIT_POLICY_SHA256",
    "FORMAL_PRIOR_STOP_POLICY",
    "FORMAL_TIMELINE_SCOPE",
    "FORMAL_TIMELINE_CONTEXT_SCHEMA",
    "FormalReasonerEpochReceipt",
    "FormalReasonerFitConfig",
    "FormalReasonerFitReceipt",
    "VerifiedEvidenceCacheSet",
    "VerifiedFrozenReasoner",
    "VerifiedGlobalTimelineContext",
    "VerifiedMIVEvidenceEvent",
    "VerifiedReasonerDatasetBundle",
    "VerifiedReasonerFit",
    "VerifiedTimelineEvent",
    "build_formal_oof_evidence_authorization",
    "build_verified_global_timeline_context",
    "build_verified_reasoner_calibration_data",
    "build_verified_reasoner_dataset",
    "fit_verified_global_affine_calibrator",
    "fit_verified_reasoner",
    "freeze_verified_reasoner",
    "load_formal_evidence_cache_set",
    "publish_formal_evidence_cache_set",
]
