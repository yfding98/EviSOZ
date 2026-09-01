"""Strict development-only score bundles for the LaBraM k31 recovery head.

This module deliberately does not issue a formal evidence or reasoner
authorization.  It replays the five patient-held-out recovery heads over the
signal-eligible ``source_train`` events and the final recovery head over the
evaluation-only ``source_dev`` events.  The result is retrospective TUSZ
bipolar edge-time involvement evidence, never a localization, origin, or
spread target.

The target firewall is structural:

* the OOF protocol is loaded from its canonical, target-free protocol JSON;
  neither the DeepSOZ source CSV nor a target-v2 target table is accepted;
* the signal-preflight receipt supplies only event timing/split identities;
* token bundles contain frozen LaBraM features and no targets; and
* deployment masks are all-true producer-availability masks.  Source
  annotation targets and annotation-coverage masks are neither read nor
  serialized.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping, Sequence

import torch

from .concept_oof import (
    ICTAL_CONCEPT_FAMILY,
    ICTAL_OOF_PROTOCOL_ARTIFACT_FILENAME,
    ICTAL_OOF_PROTOCOL_ARTIFACT_SCHEMA,
    IctalConceptOOFPlanReceipt,
    IctalConceptOOFProtocolReceipt,
)
from .data.deepsoz_signal_preflight import VerifiedDeepSOZSignalPreflightBundle
from .formal_token_corpus import VerifiedFormalTokenCorpusArtifact
from .ictal_native_eval import VerifiedIctalNativeEvalTokenCorpusArtifact
from .ictal_prediction_artifacts import (
    _canonical_json_bytes,
    _formal_token_event_id_for_timeline_record,
    _fsync_directory,
    _fsync_file,
    _load_formal_probe_tokens,
    _load_native_probe_tokens,
    _read_regular_bytes,
    _read_tensor,
    _safe_output,
    _strict_directory,
    _tensor_sha256,
    _write_tensor,
)
from .ictal_recovery_oof import (
    LABRAM_K31_CANDIDATE,
    LABRAM_K31_CONTEXT_SECONDS,
    LoadedLaBraMK31OOFRecoveryRun,
    load_labram_k31_oof_recovery_run,
)
from .temporal_masks import build_offset_aware_phase_masks


LABRAM_K31_DEVELOPMENT_SCORE_SCHEMA = (
    "soz_labram_k31_development_score_bundle_v1"
)
LABRAM_K31_DEVELOPMENT_SCORE_RECEIPT_SCHEMA = (
    "soz_labram_k31_development_score_receipt_v1"
)
LABRAM_K31_DEVELOPMENT_SCORE_PURPOSE = (
    "development_only_retrospective_tusz_bipolar_edge_time_involvement"
)
LABRAM_K31_SCORE_SEMANTICS = (
    "retrospective_tusz_bipolar_edge_time_involvement_probability_not_localization"
)
LABRAM_K31_SCORE_TRANSFORM = "sigmoid_of_frozen_k31_head_logit"
LABRAM_K31_DEPLOYMENT_MASK_POLICY = (
    "all_replayed_producer_cells_available_independent_of_source_annotations"
)
LABRAM_K31_PHASE_MASK_POLICY = (
    "fifteen_four_second_bins_from_verified_record_local_event_timeline"
)

MANIFEST_FILENAME = "manifest.json"
RECEIPT_FILENAME = "receipt.json"
SOURCE_TRAIN_SCORE_FILENAME = "source_train_oof_scores.npy"
SOURCE_TRAIN_DEPLOYMENT_FILENAME = "source_train_deployment_mask.npy"
SOURCE_TRAIN_PHASE_FILENAME = "source_train_ictal_phase_mask.npy"
SOURCE_DEV_SCORE_FILENAME = "source_dev_final_scores.npy"
SOURCE_DEV_DEPLOYMENT_FILENAME = "source_dev_deployment_mask.npy"
SOURCE_DEV_PHASE_FILENAME = "source_dev_ictal_phase_mask.npy"

_SELECTIONS = ("fold0", "fold1", "fold2", "fold3", "fold4", "final")
_MAX_PROTOCOL_BYTES = 128 * 1024 * 1024
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SCORE_BATCH_SIZE = 8
_ARTIFACT_MARKER = object()

_TENSOR_FILENAMES = {
    "source_train_oof_scores": SOURCE_TRAIN_SCORE_FILENAME,
    "source_train_deployment_mask": SOURCE_TRAIN_DEPLOYMENT_FILENAME,
    "source_train_ictal_phase_mask": SOURCE_TRAIN_PHASE_FILENAME,
    "source_dev_final_scores": SOURCE_DEV_SCORE_FILENAME,
    "source_dev_deployment_mask": SOURCE_DEV_DEPLOYMENT_FILENAME,
    "source_dev_ictal_phase_mask": SOURCE_DEV_PHASE_FILENAME,
}

_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "candidate",
        "context_seconds",
        "context_direction",
        "development_only",
        "architecture_selected_after_opened_i_dev",
        "formal_promotion",
        "authorized_for_formal_evidence_or_reasoner",
        "score_semantics",
        "score_transform",
        "score_shape",
        "temporal_resolution_seconds",
        "edge_axis_size",
        "producer_bindings",
        "oof_protocol_lineage",
        "signal_timeline_lineage",
        "source_train_corpus_lineage",
        "source_dev_corpus_lineage",
        "source_train_event_rows",
        "source_train_patient_rows",
        "source_dev_event_rows",
        "source_dev_patient_rows",
        "tensor_files",
        "deployment_mask_policy",
        "phase_mask_policy",
        "target_vectors_loaded",
        "target_values_present",
        "source_annotation_targets_present",
        "source_annotation_coverage_present",
        "private_data_used",
        "source_eval_signals_or_events_used",
    }
)

_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_sha256",
        "oof_protocol_receipt_sha256",
        "signal_timeline_receipt_sha256",
        "source_train_token_corpus_index_sha256",
        "source_dev_token_corpus_index_sha256",
        "producer_binding_receipt_sha256",
        "source_train_event_row_receipt_sha256",
        "source_dev_event_row_receipt_sha256",
        "tensor_sha256s",
    }
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _strict_json(raw: bytes, *, field: str) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{field} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        raise ValueError(f"{field} contains forbidden constant {value}")

    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not strict UTF-8 JSON") from exc


def _closed_mapping(
    value: object, expected: set[str] | frozenset[str], *, field: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{field} violates its closed schema")
    return value


@dataclass(frozen=True)
class TargetFreeOOFProtocolView:
    """Verified OOF identity/fold view containing no localization targets."""

    path: Path
    artifact_sha256: str
    receipt: IctalConceptOOFProtocolReceipt
    fold_plan_receipts: tuple[IctalConceptOOFPlanReceipt, ...]
    final_plan_receipt: IctalConceptOOFPlanReceipt

    @property
    def receipt_sha256(self) -> str:
        return self.receipt.receipt_sha256

    @property
    def crosswalk(self) -> Mapping[str, str]:
        return dict(self.receipt.target_public_crosswalk)

    def fold_for_target(self, patient_id: str) -> int:
        matches = tuple(
            int(plan.oof_fold)
            for plan in self.fold_plan_receipts
            if patient_id in plan.held_out_target_patient_ids
        )
        if len(matches) != 1:
            raise KeyError(f"Target identity has no unique OOF fold: {patient_id}")
        return matches[0]

    def assert_unchanged(self) -> None:
        replay = load_target_free_ictal_oof_protocol(
            self.path,
            expected_artifact_sha256=self.artifact_sha256,
            expected_protocol_receipt_sha256=self.receipt_sha256,
        )
        if (
            replay.receipt != self.receipt
            or replay.fold_plan_receipts != self.fold_plan_receipts
            or replay.final_plan_receipt != self.final_plan_receipt
        ):
            raise ValueError("Target-free OOF protocol changed after verification")


def _plan_receipt(value: object, *, field: str) -> IctalConceptOOFPlanReceipt:
    payload = _closed_mapping(
        value,
        {name for name in IctalConceptOOFPlanReceipt.__dataclass_fields__},
        field=field,
    )
    try:
        receipt = IctalConceptOOFPlanReceipt(**payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not a valid target-free OOF receipt") from exc
    if _canonical_sha256(payload) != receipt.receipt_sha256:
        raise ValueError(f"{field} changed under canonical reconstruction")
    return receipt


def load_target_free_ictal_oof_protocol(
    bundle_directory: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_protocol_receipt_sha256: str,
) -> TargetFreeOOFProtocolView:
    """Load the pinned OOF protocol without opening a DeepSOZ target source."""

    source = _strict_directory(
        bundle_directory, {ICTAL_OOF_PROTOCOL_ARTIFACT_FILENAME}
    )
    path = source / ICTAL_OOF_PROTOCOL_ARTIFACT_FILENAME
    raw = _read_regular_bytes(path, maximum_bytes=_MAX_PROTOCOL_BYTES)
    artifact_sha = hashlib.sha256(raw).hexdigest()
    if artifact_sha != _require_sha256(
        expected_artifact_sha256, field="expected_artifact_sha256"
    ):
        raise ValueError("OOF protocol artifact SHA mismatch")
    payload = _strict_json(raw, field="OOF protocol")
    if _canonical_json_bytes(payload) != raw:
        raise ValueError("OOF protocol is not canonical JSON")
    artifact = _closed_mapping(
        payload,
        {
            "schema_version",
            "protocol_sha256",
            "public_ledger_build_sha256",
            "split_manifest_sha256",
            "protocol",
        },
        field="OOF protocol artifact",
    )
    if artifact["schema_version"] != ICTAL_OOF_PROTOCOL_ARTIFACT_SCHEMA:
        raise ValueError("Unsupported OOF protocol artifact schema")
    protocol_payload = _closed_mapping(
        artifact["protocol"],
        {"fold_plans", "final_plan", "receipt"},
        field="OOF protocol payload",
    )
    receipt_payload = _closed_mapping(
        protocol_payload["receipt"],
        {name for name in IctalConceptOOFProtocolReceipt.__dataclass_fields__},
        field="OOF protocol receipt",
    )
    try:
        receipt = IctalConceptOOFProtocolReceipt(**receipt_payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("OOF protocol receipt failed target-free validation") from exc
    expected_receipt = _require_sha256(
        expected_protocol_receipt_sha256,
        field="expected_protocol_receipt_sha256",
    )
    if (
        receipt.receipt_sha256 != expected_receipt
        or artifact["protocol_sha256"] != expected_receipt
        or _canonical_sha256(receipt_payload) != expected_receipt
    ):
        raise ValueError("OOF protocol receipt SHA mismatch")
    if (
        artifact["public_ledger_build_sha256"]
        != receipt.public_ledger_build_sha256
        or artifact["split_manifest_sha256"] != receipt.split_manifest_sha256
    ):
        raise ValueError("OOF artifact and receipt lineage disagree")

    raw_folds = protocol_payload["fold_plans"]
    if not isinstance(raw_folds, list) or len(raw_folds) != 5:
        raise ValueError("OOF protocol must contain five folds")
    folds: list[IctalConceptOOFPlanReceipt] = []
    for index, raw_plan in enumerate(raw_folds):
        plan = _closed_mapping(
            raw_plan, {"cohorts", "receipt"}, field=f"fold plan {index}"
        )
        fold_receipt = _plan_receipt(
            plan["receipt"], field=f"fold plan {index} receipt"
        )
        if fold_receipt.oof_fold != index:
            raise ValueError("OOF fold plans are not canonically ordered")
        folds.append(fold_receipt)
    raw_final = _closed_mapping(
        protocol_payload["final_plan"],
        {"cohorts", "receipt"},
        field="final plan",
    )
    final = _plan_receipt(raw_final["receipt"], field="final plan receipt")
    if final.oof_fold is not None:
        raise ValueError("Final OOF plan unexpectedly has a fold")

    fold_hashes = tuple((index, plan.receipt_sha256) for index, plan in enumerate(folds))
    if fold_hashes != receipt.fold_plan_receipt_sha256s:
        raise ValueError("OOF fold-plan receipts do not bind the protocol")
    if final.receipt_sha256 != receipt.final_plan_receipt_sha256:
        raise ValueError("OOF final-plan receipt does not bind the protocol")
    crosswalk = dict(receipt.target_public_crosswalk)
    source_train = set(receipt.source_train_patient_ids)
    held_once: list[str] = []
    for plan in folds:
        if (
            set(plan.training_target_patient_ids)
            | set(plan.held_out_target_patient_ids)
            != source_train
            or set(plan.training_target_patient_ids)
            & set(plan.held_out_target_patient_ids)
        ):
            raise ValueError("OOF fold does not partition source-train identities")
        if set(plan.held_out_public_patient_keys) != {
            crosswalk[value] for value in plan.held_out_target_patient_ids
        }:
            raise ValueError("OOF fold public crosswalk changed")
        held_once.extend(plan.held_out_target_patient_ids)
    if len(held_once) != len(set(held_once)) or set(held_once) != source_train:
        raise ValueError("Source-train identities are not held out exactly once")
    if set(final.training_target_patient_ids) != source_train:
        raise ValueError("Final plan does not train on all source-train identities")
    if set(final.held_out_target_patient_ids) != (
        set(receipt.source_dev_patient_ids) | set(receipt.source_eval_patient_ids)
    ):
        raise ValueError("Final plan held-out split identities changed")
    return TargetFreeOOFProtocolView(
        path=source,
        artifact_sha256=artifact_sha,
        receipt=receipt,
        fold_plan_receipts=tuple(folds),
        final_plan_receipt=final,
    )


@dataclass(frozen=True)
class SignalTimelineRow:
    event_id: str
    target_patient_id: str
    public_patient_id: str
    model_split: str
    relative_edf_path: str
    global_event_index: int
    event_record_sha256: str
    processed_window_sha256: str
    duration_sec: float
    previous_gap_sec: float | None

    @property
    def local_edf_path(self) -> str:
        return self.relative_edf_path


@dataclass(frozen=True)
class TargetFreeSignalTimelineView:
    source_train_rows: tuple[SignalTimelineRow, ...]
    source_dev_rows: tuple[SignalTimelineRow, ...]
    source_train_phase_mask: torch.Tensor
    source_dev_phase_mask: torch.Tensor
    lineage: Mapping[str, object]
    receipt_sha256: str


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def build_target_free_signal_timeline_view(
    signal: VerifiedDeepSOZSignalPreflightBundle,
    protocol: TargetFreeOOFProtocolView,
    *,
    expected_target_v2_artifact_sha256: str,
    expected_target_v2_receipt_sha256: str,
    expected_target_v2_policy_sha256: str,
) -> TargetFreeSignalTimelineView:
    """Build train/dev timing identities without loading target-v2 values."""

    if not isinstance(signal, VerifiedDeepSOZSignalPreflightBundle):
        raise TypeError("signal must be a strictly loaded signal-preflight bundle")
    if not isinstance(protocol, TargetFreeOOFProtocolView):
        raise TypeError("protocol must be a target-free verified OOF view")
    protocol.assert_unchanged()
    receipt = signal.receipt
    bindings = {
        "verified_target_v2_artifact_sha256": _require_sha256(
            expected_target_v2_artifact_sha256,
            field="expected_target_v2_artifact_sha256",
        ),
        "verified_target_v2_receipt_sha256": _require_sha256(
            expected_target_v2_receipt_sha256,
            field="expected_target_v2_receipt_sha256",
        ),
        "verified_target_v2_policy_sha256": _require_sha256(
            expected_target_v2_policy_sha256,
            field="expected_target_v2_policy_sha256",
        ),
        "split_manifest_sha256": protocol.receipt.split_manifest_sha256,
    }
    for field, expected in bindings.items():
        if receipt.get(field) != expected:
            raise ValueError(f"Signal timeline changed target-free binding {field}")

    crosswalk = protocol.crosswalk
    split_rosters = {
        str(row[0]): tuple(str(value) for value in row[1])
        for row in receipt["eligible_split_patient_ids"]
    }
    expected_splits = {
        "source_train": set(protocol.receipt.source_train_patient_ids),
        "source_dev": set(protocol.receipt.source_dev_patient_ids),
        "source_eval": set(protocol.receipt.source_eval_patient_ids),
    }
    if set(split_rosters) != set(expected_splits):
        raise ValueError("Signal-preflight split roster keys changed")
    for split, roster in split_rosters.items():
        if split not in expected_splits or not set(roster) <= expected_splits[split]:
            raise ValueError("Signal-preflight split roster differs from OOF protocol")

    candidates: dict[str, dict[str, object]] = {}
    for raw in (*receipt["events"], *receipt["exclusions"]):
        # Evaluation event metadata is deliberately not consumed by this
        # development materializer.  The OOF protocol's split identity is
        # retained above solely as an exclusion guard.
        if str(raw["model_split"]) == "source_eval":
            continue
        event_id = str(raw["event_id"])
        if event_id in candidates:
            raise ValueError("Signal timeline contains duplicate candidate events")
        start = _finite_float(raw["global_t0_sec"], field="global_t0_sec")
        stop = _finite_float(raw["global_stop_sec"], field="global_stop_sec")
        if stop <= start:
            raise ValueError("Signal timeline event has non-positive duration")
        patient_id = str(raw["patient_id"])
        split = str(raw["model_split"])
        if split not in expected_splits or patient_id not in expected_splits[split]:
            raise ValueError("Signal event identity differs from OOF split identity")
        candidates[event_id] = {
            "raw": raw,
            "start": start,
            "stop": stop,
            "source_record": str(raw["deepsoz_source_record_sha256"]),
        }

    prior_stop: dict[str, float | None] = {}
    by_record: dict[str, list[dict[str, object]]] = {}
    for item in candidates.values():
        by_record.setdefault(str(item["source_record"]), []).append(item)
    for values in by_record.values():
        ordered = sorted(values, key=lambda item: int(item["raw"]["global_event_index"]))
        indices = tuple(int(item["raw"]["global_event_index"]) for item in ordered)
        if indices != tuple(range(len(ordered))):
            raise ValueError("Signal receipt does not contain a complete record timeline")
        running: float | None = None
        for item in ordered:
            event_id = str(item["raw"]["event_id"])
            prior_stop[event_id] = running
            stop = float(item["stop"])
            running = stop if running is None else max(running, stop)

    accepted = {str(raw["event_id"]): raw for raw in receipt["events"]}
    for split in ("source_train", "source_dev"):
        observed = {
            str(raw["patient_id"])
            for raw in receipt["events"]
            if str(raw["model_split"]) == split
        }
        if observed != set(split_rosters[split]):
            raise ValueError("Signal event patients disagree with split roster")
    rows_by_split: dict[str, list[SignalTimelineRow]] = {
        "source_train": [],
        "source_dev": [],
    }
    for event_id, raw in accepted.items():
        split = str(raw["model_split"])
        if split not in rows_by_split:
            continue
        patient_id = str(raw["patient_id"])
        start = float(candidates[event_id]["start"])
        stop = float(candidates[event_id]["stop"])
        previous = prior_stop[event_id]
        rows_by_split[split].append(
            SignalTimelineRow(
                event_id=event_id,
                target_patient_id=patient_id,
                public_patient_id=str(crosswalk[patient_id]),
                model_split=split,
                relative_edf_path=str(raw["relative_edf_path"]),
                global_event_index=int(raw["global_event_index"]),
                event_record_sha256=_require_sha256(
                    raw["event_record_sha256"], field="event_record_sha256"
                ),
                processed_window_sha256=_require_sha256(
                    raw["processed_window_sha256"], field="processed_window_sha256"
                ),
                duration_sec=stop - start,
                previous_gap_sec=(
                    None if previous is None else max(0.0, start - previous)
                ),
            )
        )
    train_rows = tuple(
        sorted(rows_by_split["source_train"], key=lambda row: (row.target_patient_id, row.event_id))
    )
    dev_rows = tuple(
        sorted(rows_by_split["source_dev"], key=lambda row: (row.target_patient_id, row.event_id))
    )
    if not train_rows or not dev_rows:
        raise ValueError("Target-free timeline requires source-train and source-dev events")

    def phase(rows: tuple[SignalTimelineRow, ...]) -> torch.Tensor:
        value = build_offset_aware_phase_masks(
            [row.duration_sec for row in rows],
            offset_trustworthy=[True] * len(rows),
            previous_seizure_gap_sec=[row.previous_gap_sec for row in rows],
            previous_timeline_trustworthy=[True] * len(rows),
        ).ictal_phase_mask.to(torch.bool).contiguous()
        if tuple(value.shape) != (len(rows), 15):
            raise ValueError("Target-free timeline emitted an invalid phase mask")
        return value

    train_phase = phase(train_rows)
    dev_phase = phase(dev_rows)
    lineage = {
        "signal_preflight_artifact_sha256": signal.artifact_sha256,
        "signal_preflight_receipt_sha256": signal.receipt_sha256,
        "verified_target_v2_artifact_sha256": bindings[
            "verified_target_v2_artifact_sha256"
        ],
        "verified_target_v2_receipt_sha256": bindings[
            "verified_target_v2_receipt_sha256"
        ],
        "verified_target_v2_policy_sha256": bindings[
            "verified_target_v2_policy_sha256"
        ],
        "event_inputs_sha256": receipt["event_inputs_sha256"],
        "split_manifest_sha256": receipt["split_manifest_sha256"],
        "preprocess_config_sha256": receipt["preprocess_config_sha256"],
        "source_train_event_count": len(train_rows),
        "source_dev_event_count": len(dev_rows),
        "source_eval_events_used": False,
        "target_vectors_loaded": False,
        "source_train_phase_mask_sha256": _tensor_sha256(
            "source_train_ictal_phase_mask", train_phase
        ),
        "source_dev_phase_mask_sha256": _tensor_sha256(
            "source_dev_ictal_phase_mask", dev_phase
        ),
        "event_identity_receipt_sha256": _canonical_sha256(
            [
                [
                    row.event_id,
                    row.target_patient_id,
                    row.public_patient_id,
                    row.model_split,
                    row.relative_edf_path,
                    row.global_event_index,
                    row.event_record_sha256,
                    row.processed_window_sha256,
                ]
                for row in (*train_rows, *dev_rows)
            ]
        ),
    }
    return TargetFreeSignalTimelineView(
        source_train_rows=train_rows,
        source_dev_rows=dev_rows,
        source_train_phase_mask=train_phase,
        source_dev_phase_mask=dev_phase,
        lineage=lineage,
        receipt_sha256=_canonical_sha256(lineage),
    )


def _strict_recovery_runs(
    values: Sequence[LoadedLaBraMK31OOFRecoveryRun],
) -> tuple[LoadedLaBraMK31OOFRecoveryRun, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("recovery_runs must be a sequence")
    indexed: dict[str, LoadedLaBraMK31OOFRecoveryRun] = {}
    for value in values:
        if not isinstance(value, LoadedLaBraMK31OOFRecoveryRun):
            raise TypeError("recovery run must come from the strict loader")
        replay = load_labram_k31_oof_recovery_run(
            value.path, expected_manifest_sha256=value.manifest_sha256
        )
        selection = str(replay.manifest["selection"])
        if selection in indexed:
            raise ValueError(f"Duplicate recovery producer {selection}")
        indexed[selection] = replay
    if set(indexed) != set(_SELECTIONS):
        raise ValueError("Development scores require fold0..fold4 and final exactly once")
    return tuple(indexed[selection] for selection in _SELECTIONS)


def _validate_run_and_corpus_lineage(
    runs: tuple[LoadedLaBraMK31OOFRecoveryRun, ...],
    protocol: TargetFreeOOFProtocolView,
    timeline: TargetFreeSignalTimelineView,
    source_train_corpus: VerifiedFormalTokenCorpusArtifact,
    source_dev_corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
) -> None:
    if not isinstance(source_train_corpus, VerifiedFormalTokenCorpusArtifact):
        raise TypeError("source_train_corpus must be a strict formal token corpus")
    if not isinstance(source_dev_corpus, VerifiedIctalNativeEvalTokenCorpusArtifact):
        raise TypeError("source_dev_corpus must be a strict evaluation-only token corpus")
    if (
        source_train_corpus.training_bundle_manifest_sha256
        != source_train_corpus.master_bundle_manifest_sha256
        or source_train_corpus.training_source_manifest_sha256
        != source_train_corpus.master_source_manifest_sha256
    ):
        raise ValueError("Source-train scoring requires the verified master token corpus")
    if not source_dev_corpus.evaluation_only or source_dev_corpus.training_authorized:
        raise ValueError("Source-dev scoring requires an evaluation-only token corpus")
    if source_dev_corpus.signal_preflight_receipt_sha256 != timeline.lineage[
        "signal_preflight_receipt_sha256"
    ]:
        raise ValueError("Source-dev token corpus uses another signal timeline")
    if len(
        {
            (
                run.manifest["target_snapshot_manifest_sha256"],
                run.manifest["target_snapshot_receipt_sha256"],
            )
            for run in runs
        }
    ) != 1:
        raise ValueError("Recovery producers use different frozen target snapshots")
    crosswalk = protocol.crosswalk
    by_selection = {str(run.manifest["selection"]): run for run in runs}
    for fold in range(5):
        run = by_selection[f"fold{fold}"]
        plan = protocol.fold_plan_receipts[fold]
        checks = {
            "OOF artifact": run.manifest["oof_protocol_artifact_sha256"]
            == protocol.artifact_sha256,
            "OOF receipt": run.manifest["oof_protocol_receipt_sha256"]
            == protocol.receipt_sha256,
            "OOF plan": run.manifest["oof_plan_receipt_sha256"]
            == plan.receipt_sha256,
            "split": run.manifest["split_manifest_sha256"]
            == protocol.receipt.split_manifest_sha256,
            "native corpus": run.manifest["native_evaluation_corpus_index_sha256"]
            == source_train_corpus.index_sha256,
            "native manifest": run.manifest["native_evaluation_manifest_sha256"]
            == source_train_corpus.master_source_manifest_sha256,
            "held-out public roster": set(
                run.manifest["held_out_exclusion_public_patient_ids"]
            )
            == {crosswalk[value] for value in plan.held_out_target_patient_ids},
        }
        failed = tuple(label for label, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"fold{fold} recovery lineage failed: {failed}")
    final = by_selection["final"]
    checks = {
        "OOF artifact": final.manifest["oof_protocol_artifact_sha256"]
        == protocol.artifact_sha256,
        "OOF receipt": final.manifest["oof_protocol_receipt_sha256"]
        == protocol.receipt_sha256,
        "OOF plan": final.manifest["oof_plan_receipt_sha256"]
        == protocol.final_plan_receipt.receipt_sha256,
        "split": final.manifest["split_manifest_sha256"]
        == protocol.receipt.split_manifest_sha256,
        "native corpus": final.manifest["native_evaluation_corpus_index_sha256"]
        == source_dev_corpus.index_sha256,
        "native manifest": final.manifest["native_evaluation_manifest_sha256"]
        == source_dev_corpus.manifest_receipt_sha256,
    }
    failed = tuple(label for label, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"final recovery lineage failed: {failed}")
    if set(final.manifest["native_evaluation_public_patient_ids"]) != {
        row.public_patient_id for row in timeline.source_dev_rows
    }:
        raise ValueError("Final producer native patient roster differs from source-dev")
    for row in timeline.source_train_rows:
        fold = protocol.fold_for_target(row.target_patient_id)
        if row.public_patient_id not in set(
            by_selection[f"fold{fold}"].manifest[
                "native_evaluation_public_patient_ids"
            ]
        ):
            raise ValueError("Source-train event is absent from its held-out producer")
    if {row.public_patient_id for row in timeline.source_dev_rows} != {
        binding.public_patient_id for binding in source_dev_corpus.events
    }:
        raise ValueError("Source-dev timeline and token-corpus patient rosters differ")


@dataclass(frozen=True)
class _DevelopmentScoreGrid:
    source_train_scores: torch.Tensor
    source_train_deployment_mask: torch.Tensor
    source_train_phase_mask: torch.Tensor
    source_train_event_rows: tuple[Mapping[str, object], ...]
    source_train_patient_rows: tuple[Mapping[str, object], ...]
    source_dev_scores: torch.Tensor
    source_dev_deployment_mask: torch.Tensor
    source_dev_phase_mask: torch.Tensor
    source_dev_event_rows: tuple[Mapping[str, object], ...]
    source_dev_patient_rows: tuple[Mapping[str, object], ...]
    foundation_lineage: Mapping[str, str]


def _score_items(
    items: Sequence[tuple[int, LoadedLaBraMK31OOFRecoveryRun, object]],
) -> torch.Tensor:
    if not items:
        raise ValueError("Cannot score an empty event sequence")
    result: list[torch.Tensor | None] = [None] * len(items)
    grouped: dict[str, list[tuple[int, object]]] = {}
    runs: dict[str, LoadedLaBraMK31OOFRecoveryRun] = {}
    for index, run, token in items:
        selection = str(run.manifest["selection"])
        grouped.setdefault(selection, []).append((index, token))
        runs[selection] = run
    for selection, values in grouped.items():
        head = runs[selection].head.cpu().eval()
        for start in range(0, len(values), _SCORE_BATCH_SIZE):
            chunk = values[start : start + _SCORE_BATCH_SIZE]
            tokens = torch.stack(
                [value.tokens.to(torch.float32) for _, value in chunk], dim=0
            ).contiguous()
            with torch.inference_mode():
                logits = head(tokens)
                probabilities = head.probabilities(logits).squeeze(-1).cpu()
            if tuple(probabilities.shape[1:]) != (20, 60):
                raise ValueError("k31 producer emitted a non-[20,60] score")
            for local, (output_index, _) in enumerate(chunk):
                value = probabilities[local].to(torch.float32).contiguous()
                if not torch.isfinite(value).all() or not torch.all(
                    (value >= 0.0) & (value <= 1.0)
                ):
                    raise ValueError("k31 producer emitted an invalid probability")
                result[output_index] = value
    if any(value is None for value in result):
        raise RuntimeError("k31 score generation left an event unscored")
    return torch.stack([value for value in result if value is not None], dim=0)


def _patient_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    grouped: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (
            str(row["target_patient_id"]),
            str(row["public_patient_id"]),
            str(row["producer_selection"]),
        )
        grouped[key] = grouped.get(key, 0) + 1
    return tuple(
        {
            "target_patient_id": target,
            "public_patient_id": public,
            "producer_selection": selection,
            "oof_fold": (
                None if selection == "final" else int(selection.removeprefix("fold"))
            ),
            "event_count": count,
        }
        for (target, public, selection), count in sorted(grouped.items())
    )


def _generate_score_grid(
    runs: tuple[LoadedLaBraMK31OOFRecoveryRun, ...],
    protocol: TargetFreeOOFProtocolView,
    timeline: TargetFreeSignalTimelineView,
    source_train_corpus: VerifiedFormalTokenCorpusArtifact,
    source_dev_corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
) -> _DevelopmentScoreGrid:
    _validate_run_and_corpus_lineage(
        runs, protocol, timeline, source_train_corpus, source_dev_corpus
    )
    by_selection = {str(run.manifest["selection"]): run for run in runs}
    train_tokens, train_foundation_sha = _load_formal_probe_tokens(
        source_train_corpus
    )
    dev_tokens = _load_native_probe_tokens(source_dev_corpus)
    if set(dev_tokens) != {row.event_id for row in timeline.source_dev_rows}:
        raise ValueError("Source-dev token corpus is not the complete timeline roster")

    train_items: list[tuple[int, LoadedLaBraMK31OOFRecoveryRun, object]] = []
    train_event_rows: list[Mapping[str, object]] = []
    for index, row in enumerate(timeline.source_train_rows):
        fold = protocol.fold_for_target(row.target_patient_id)
        token_event_id = _formal_token_event_id_for_timeline_record(row, row)
        if token_event_id not in train_tokens:
            raise ValueError("Source-train master token corpus omits a timeline event")
        train_items.append((index, by_selection[f"fold{fold}"], train_tokens[token_event_id]))
        train_event_rows.append(
            {
                "event_id": row.event_id,
                "token_event_id": token_event_id,
                "target_patient_id": row.target_patient_id,
                "public_patient_id": row.public_patient_id,
                "oof_fold": fold,
                "producer_selection": f"fold{fold}",
            }
        )
    dev_items: list[tuple[int, LoadedLaBraMK31OOFRecoveryRun, object]] = []
    dev_event_rows: list[Mapping[str, object]] = []
    for index, row in enumerate(timeline.source_dev_rows):
        binding = dev_tokens[row.event_id]
        if binding.event_id != row.event_id:
            raise ValueError("Source-dev token identity changed")
        dev_items.append((index, by_selection["final"], binding))
        dev_event_rows.append(
            {
                "event_id": row.event_id,
                "token_event_id": row.event_id,
                "target_patient_id": row.target_patient_id,
                "public_patient_id": row.public_patient_id,
                "oof_fold": None,
                "producer_selection": "final",
            }
        )
    first_train = train_items[0][2]
    if (
        train_foundation_sha != first_train.foundation_feature_receipt_sha256
        or train_foundation_sha != source_dev_corpus.foundation_feature_receipt_sha256
        or first_train.foundation_checkpoint_sha256
        != source_dev_corpus.foundation_checkpoint_sha256
    ):
        raise ValueError("Source-train/source-dev frozen LaBraM lineage differs")
    foundation = {
        "feature_receipt_sha256": train_foundation_sha,
        "checkpoint_sha256": first_train.foundation_checkpoint_sha256,
        "modeling_sha256": first_train.foundation_feature_receipt.modeling_sha256,
    }
    if foundation["modeling_sha256"] != source_dev_corpus.foundation_modeling_sha256:
        raise ValueError("Source-train/source-dev LaBraM modeling lineage differs")

    train_scores = _score_items(train_items)
    dev_scores = _score_items(dev_items)
    train_mask = torch.ones_like(train_scores, dtype=torch.bool)
    dev_mask = torch.ones_like(dev_scores, dtype=torch.bool)
    return _DevelopmentScoreGrid(
        source_train_scores=train_scores,
        source_train_deployment_mask=train_mask,
        source_train_phase_mask=timeline.source_train_phase_mask,
        source_train_event_rows=tuple(train_event_rows),
        source_train_patient_rows=_patient_rows(train_event_rows),
        source_dev_scores=dev_scores,
        source_dev_deployment_mask=dev_mask,
        source_dev_phase_mask=timeline.source_dev_phase_mask,
        source_dev_event_rows=tuple(dev_event_rows),
        source_dev_patient_rows=_patient_rows(dev_event_rows),
        foundation_lineage=foundation,
    )


def _producer_bindings(
    runs: Sequence[LoadedLaBraMK31OOFRecoveryRun],
) -> list[dict[str, object]]:
    return [
        {
            "selection": run.manifest["selection"],
            "oof_fold": run.manifest["oof_fold"],
            "recovery_run_manifest_sha256": run.manifest_sha256,
            "checkpoint_sha256": run.manifest["checkpoint_sha256"],
            "head_state_sha256": run.manifest["head_state_sha256"],
            "oof_plan_receipt_sha256": run.manifest[
                "oof_plan_receipt_sha256"
            ],
            "training_manifest_sha256": run.manifest[
                "training_manifest_sha256"
            ],
            "training_corpus_index_sha256": run.manifest[
                "training_corpus_index_sha256"
            ],
            "native_evaluation_manifest_sha256": run.manifest[
                "native_evaluation_manifest_sha256"
            ],
            "native_evaluation_corpus_index_sha256": run.manifest[
                "native_evaluation_corpus_index_sha256"
            ],
            "target_snapshot_manifest_sha256": run.manifest[
                "target_snapshot_manifest_sha256"
            ],
            "target_snapshot_receipt_sha256": run.manifest[
                "target_snapshot_receipt_sha256"
            ],
        }
        for run in runs
    ]


def _protocol_lineage(protocol: TargetFreeOOFProtocolView) -> dict[str, object]:
    return {
        "artifact_sha256": protocol.artifact_sha256,
        "receipt_sha256": protocol.receipt_sha256,
        "split_manifest_sha256": protocol.receipt.split_manifest_sha256,
        "public_ledger_build_sha256": protocol.receipt.public_ledger_build_sha256,
        "ledger_sha256": protocol.receipt.ledger_sha256,
        "ledger_receipt_sha256": protocol.receipt.ledger_receipt_sha256,
        "target_public_crosswalk_sha256": protocol.receipt.target_public_crosswalk_sha256,
        "fold_plan_receipt_sha256s": [
            [int(fold), digest]
            for fold, digest in protocol.receipt.fold_plan_receipt_sha256s
        ],
        "final_plan_receipt_sha256": protocol.receipt.final_plan_receipt_sha256,
        "target_vectors_loaded": False,
    }


def _train_corpus_lineage(
    corpus: VerifiedFormalTokenCorpusArtifact,
    foundation: Mapping[str, str],
) -> dict[str, object]:
    return {
        "index_sha256": corpus.index_sha256,
        "master_bundle_manifest_sha256": corpus.master_bundle_manifest_sha256,
        "master_source_manifest_sha256": corpus.master_source_manifest_sha256,
        "training_bundle_manifest_sha256": corpus.training_bundle_manifest_sha256,
        "training_source_manifest_sha256": corpus.training_source_manifest_sha256,
        "preprocessing_selection_artifact_sha256": corpus.preprocessing_selection_artifact_sha256,
        "preprocessing_selection_bundle_receipt_sha256": corpus.preprocessing_selection_bundle_receipt_sha256,
        "preprocessing_protocol_receipt_sha256": corpus.preprocessing_protocol_receipt_sha256,
        "preprocessing_selected_arm_result_receipt_sha256": corpus.preprocessing_selected_arm_result_receipt_sha256,
        "event_roster_sha256": corpus.event_roster_sha256,
        "patient_roster_sha256": corpus.patient_roster_sha256,
        "patient_event_roster_sha256": corpus.patient_event_roster_sha256,
        "tensor_roster_sha256": corpus.tensor_roster_sha256,
        "foundation": dict(foundation),
    }


def _dev_corpus_lineage(
    corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
) -> dict[str, object]:
    return {
        "index_sha256": corpus.index_sha256,
        "manifest_artifact_sha256": corpus.manifest_artifact_sha256,
        "manifest_receipt_sha256": corpus.manifest_receipt_sha256,
        "signal_preflight_artifact_sha256": corpus.signal_preflight_artifact_sha256,
        "signal_preflight_receipt_sha256": corpus.signal_preflight_receipt_sha256,
        "event_roster_sha256": corpus.event_roster_sha256,
        "patient_roster_sha256": corpus.patient_roster_sha256,
        "patient_event_roster_sha256": corpus.patient_event_roster_sha256,
        "tensor_roster_sha256": corpus.tensor_roster_sha256,
        "foundation": {
            "feature_receipt_sha256": corpus.foundation_feature_receipt_sha256,
            "checkpoint_sha256": corpus.foundation_checkpoint_sha256,
            "modeling_sha256": corpus.foundation_modeling_sha256,
        },
        "evaluation_only": corpus.evaluation_only,
        "training_authorized": corpus.training_authorized,
    }


def _manifest_payload(
    *,
    runs: tuple[LoadedLaBraMK31OOFRecoveryRun, ...],
    protocol: TargetFreeOOFProtocolView,
    timeline: TargetFreeSignalTimelineView,
    source_train_corpus: VerifiedFormalTokenCorpusArtifact,
    source_dev_corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
    grid: _DevelopmentScoreGrid,
    tensor_records: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": LABRAM_K31_DEVELOPMENT_SCORE_SCHEMA,
        "purpose": LABRAM_K31_DEVELOPMENT_SCORE_PURPOSE,
        "candidate": LABRAM_K31_CANDIDATE,
        "context_seconds": LABRAM_K31_CONTEXT_SECONDS,
        "context_direction": "symmetric_retrospective_not_causal",
        "development_only": True,
        "architecture_selected_after_opened_i_dev": True,
        "formal_promotion": False,
        "authorized_for_formal_evidence_or_reasoner": False,
        "score_semantics": LABRAM_K31_SCORE_SEMANTICS,
        "score_transform": LABRAM_K31_SCORE_TRANSFORM,
        "score_shape": ["event", 20, 60],
        "temporal_resolution_seconds": 1,
        "edge_axis_size": 20,
        "producer_bindings": _producer_bindings(runs),
        "oof_protocol_lineage": _protocol_lineage(protocol),
        "signal_timeline_lineage": {
            **dict(timeline.lineage),
            "receipt_sha256": timeline.receipt_sha256,
        },
        "source_train_corpus_lineage": _train_corpus_lineage(
            source_train_corpus, grid.foundation_lineage
        ),
        "source_dev_corpus_lineage": _dev_corpus_lineage(source_dev_corpus),
        "source_train_event_rows": list(grid.source_train_event_rows),
        "source_train_patient_rows": list(grid.source_train_patient_rows),
        "source_dev_event_rows": list(grid.source_dev_event_rows),
        "source_dev_patient_rows": list(grid.source_dev_patient_rows),
        "tensor_files": dict(tensor_records),
        "deployment_mask_policy": LABRAM_K31_DEPLOYMENT_MASK_POLICY,
        "phase_mask_policy": LABRAM_K31_PHASE_MASK_POLICY,
        "target_vectors_loaded": False,
        "target_values_present": False,
        "source_annotation_targets_present": False,
        "source_annotation_coverage_present": False,
        "private_data_used": False,
        "source_eval_signals_or_events_used": False,
    }


def _tensor_values(grid: _DevelopmentScoreGrid) -> dict[str, torch.Tensor]:
    return {
        "source_train_oof_scores": grid.source_train_scores,
        "source_train_deployment_mask": grid.source_train_deployment_mask,
        "source_train_ictal_phase_mask": grid.source_train_phase_mask,
        "source_dev_final_scores": grid.source_dev_scores,
        "source_dev_deployment_mask": grid.source_dev_deployment_mask,
        "source_dev_ictal_phase_mask": grid.source_dev_phase_mask,
    }


@dataclass(frozen=True, init=False)
class VerifiedLaBraMK31DevelopmentScoreArtifact:
    """Opaque development score artifact; explicitly not formal authority."""

    path: Path
    artifact_sha256: str
    receipt_sha256: str
    manifest: Mapping[str, object]
    source_train_scores: torch.Tensor
    source_train_deployment_mask: torch.Tensor
    source_train_phase_mask: torch.Tensor
    source_dev_scores: torch.Tensor
    source_dev_deployment_mask: torch.Tensor
    source_dev_phase_mask: torch.Tensor

    def __init__(self, *, _verification_marker: object, **values: object) -> None:
        if _verification_marker is not _ARTIFACT_MARKER:
            raise TypeError(
                "VerifiedLaBraMK31DevelopmentScoreArtifact can only be issued by the strict loader"
            )
        for field, value in values.items():
            object.__setattr__(self, field, value)


def _expected_files() -> set[str]:
    return {MANIFEST_FILENAME, RECEIPT_FILENAME, *_TENSOR_FILENAMES.values()}


def _read_bundle(
    path: str | Path,
    *,
    recovery_runs: Sequence[LoadedLaBraMK31OOFRecoveryRun],
    protocol: TargetFreeOOFProtocolView,
    timeline: TargetFreeSignalTimelineView,
    source_train_corpus: VerifiedFormalTokenCorpusArtifact,
    source_dev_corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> VerifiedLaBraMK31DevelopmentScoreArtifact:
    runs = _strict_recovery_runs(recovery_runs)
    replay = _generate_score_grid(
        runs, protocol, timeline, source_train_corpus, source_dev_corpus
    )
    source = _strict_directory(path, _expected_files())
    manifest_raw = _read_regular_bytes(
        source / MANIFEST_FILENAME, maximum_bytes=_MAX_MANIFEST_BYTES
    )
    receipt_raw = _read_regular_bytes(
        source / RECEIPT_FILENAME, maximum_bytes=_MAX_MANIFEST_BYTES
    )
    artifact_sha = hashlib.sha256(manifest_raw).hexdigest()
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    if artifact_sha != _require_sha256(
        expected_artifact_sha256, field="expected_artifact_sha256"
    ):
        raise ValueError("Development score artifact SHA mismatch")
    if receipt_sha != _require_sha256(
        expected_receipt_sha256, field="expected_receipt_sha256"
    ):
        raise ValueError("Development score receipt SHA mismatch")
    manifest = _closed_mapping(
        _strict_json(manifest_raw, field="development score manifest"),
        _MANIFEST_FIELDS,
        field="development score manifest",
    )
    receipt = _closed_mapping(
        _strict_json(receipt_raw, field="development score receipt"),
        _RECEIPT_FIELDS,
        field="development score receipt",
    )
    if _canonical_json_bytes(manifest) != manifest_raw or _canonical_json_bytes(
        receipt
    ) != receipt_raw:
        raise ValueError("Development score JSON is not canonical")
    tensor_records = manifest.get("tensor_files")
    if not isinstance(tensor_records, dict) or set(tensor_records) != set(
        _TENSOR_FILENAMES
    ):
        raise ValueError("Development score tensor roster changed")
    tensors = {
        name: _read_tensor(
            source,
            name=name,
            record=tensor_records[name],
            expected_filename=filename,
        )
        for name, filename in _TENSOR_FILENAMES.items()
    }
    expected_tensors = _tensor_values(replay)
    if any(
        not torch.equal(tensors[name], expected_tensors[name])
        for name in _TENSOR_FILENAMES
    ):
        raise ValueError("Stored development scores differ from strict checkpoint/token replay")
    expected_static = _manifest_payload(
        runs=runs,
        protocol=protocol,
        timeline=timeline,
        source_train_corpus=source_train_corpus,
        source_dev_corpus=source_dev_corpus,
        grid=replay,
        tensor_records=tensor_records,
    )
    if manifest != expected_static:
        raise ValueError("Development score manifest changed strict lineage or boundaries")
    tensor_hashes = {
        name: _tensor_sha256(name, tensors[name]) for name in _TENSOR_FILENAMES
    }
    expected_receipt = {
        "schema_version": LABRAM_K31_DEVELOPMENT_SCORE_RECEIPT_SCHEMA,
        "artifact_sha256": artifact_sha,
        "oof_protocol_receipt_sha256": protocol.receipt_sha256,
        "signal_timeline_receipt_sha256": timeline.receipt_sha256,
        "source_train_token_corpus_index_sha256": source_train_corpus.index_sha256,
        "source_dev_token_corpus_index_sha256": source_dev_corpus.index_sha256,
        "producer_binding_receipt_sha256": _canonical_sha256(
            _producer_bindings(runs)
        ),
        "source_train_event_row_receipt_sha256": _canonical_sha256(
            list(replay.source_train_event_rows)
        ),
        "source_dev_event_row_receipt_sha256": _canonical_sha256(
            list(replay.source_dev_event_rows)
        ),
        "tensor_sha256s": tensor_hashes,
    }
    if receipt != expected_receipt:
        raise ValueError("Development score receipt does not bind exact replay")
    return VerifiedLaBraMK31DevelopmentScoreArtifact(
        _verification_marker=_ARTIFACT_MARKER,
        path=source,
        artifact_sha256=artifact_sha,
        receipt_sha256=receipt_sha,
        manifest=manifest,
        source_train_scores=tensors["source_train_oof_scores"],
        source_train_deployment_mask=tensors[
            "source_train_deployment_mask"
        ],
        source_train_phase_mask=tensors["source_train_ictal_phase_mask"],
        source_dev_scores=tensors["source_dev_final_scores"],
        source_dev_deployment_mask=tensors["source_dev_deployment_mask"],
        source_dev_phase_mask=tensors["source_dev_ictal_phase_mask"],
    )


def load_labram_k31_development_score_artifact(
    path: str | Path,
    *,
    recovery_runs: Sequence[LoadedLaBraMK31OOFRecoveryRun],
    protocol: TargetFreeOOFProtocolView,
    timeline: TargetFreeSignalTimelineView,
    source_train_corpus: VerifiedFormalTokenCorpusArtifact,
    source_dev_corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> VerifiedLaBraMK31DevelopmentScoreArtifact:
    return _read_bundle(
        path,
        recovery_runs=recovery_runs,
        protocol=protocol,
        timeline=timeline,
        source_train_corpus=source_train_corpus,
        source_dev_corpus=source_dev_corpus,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
    )


def materialize_labram_k31_development_scores(
    *,
    recovery_runs: Sequence[LoadedLaBraMK31OOFRecoveryRun],
    protocol: TargetFreeOOFProtocolView,
    timeline: TargetFreeSignalTimelineView,
    source_train_corpus: VerifiedFormalTokenCorpusArtifact,
    source_dev_corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
    output_directory: str | Path,
) -> VerifiedLaBraMK31DevelopmentScoreArtifact:
    """Replay target-free scores; accepts no caller scores, masks, or labels."""

    runs = _strict_recovery_runs(recovery_runs)
    grid = _generate_score_grid(
        runs, protocol, timeline, source_train_corpus, source_dev_corpus
    )
    target = _safe_output(output_directory)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    published = False
    try:
        tensor_records = {
            name: _write_tensor(
                temporary / _TENSOR_FILENAMES[name], name, value
            )
            for name, value in _tensor_values(grid).items()
        }
        manifest = _manifest_payload(
            runs=runs,
            protocol=protocol,
            timeline=timeline,
            source_train_corpus=source_train_corpus,
            source_dev_corpus=source_dev_corpus,
            grid=grid,
            tensor_records=tensor_records,
        )
        manifest_raw = _canonical_json_bytes(manifest)
        artifact_sha = hashlib.sha256(manifest_raw).hexdigest()
        receipt = {
            "schema_version": LABRAM_K31_DEVELOPMENT_SCORE_RECEIPT_SCHEMA,
            "artifact_sha256": artifact_sha,
            "oof_protocol_receipt_sha256": protocol.receipt_sha256,
            "signal_timeline_receipt_sha256": timeline.receipt_sha256,
            "source_train_token_corpus_index_sha256": source_train_corpus.index_sha256,
            "source_dev_token_corpus_index_sha256": source_dev_corpus.index_sha256,
            "producer_binding_receipt_sha256": _canonical_sha256(
                _producer_bindings(runs)
            ),
            "source_train_event_row_receipt_sha256": _canonical_sha256(
                list(grid.source_train_event_rows)
            ),
            "source_dev_event_row_receipt_sha256": _canonical_sha256(
                list(grid.source_dev_event_rows)
            ),
            "tensor_sha256s": {
                name: _tensor_sha256(name, value)
                for name, value in _tensor_values(grid).items()
            },
        }
        receipt_raw = _canonical_json_bytes(receipt)
        receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
        (temporary / MANIFEST_FILENAME).write_bytes(manifest_raw)
        (temporary / RECEIPT_FILENAME).write_bytes(receipt_raw)
        _fsync_file(temporary / MANIFEST_FILENAME)
        _fsync_file(temporary / RECEIPT_FILENAME)
        _fsync_directory(temporary)
        if os.path.lexists(target):
            raise FileExistsError(f"Development score output already exists: {target}")
        os.rename(temporary, target)
        published = True
        _fsync_directory(target.parent)
        return load_labram_k31_development_score_artifact(
            target,
            recovery_runs=runs,
            protocol=protocol,
            timeline=timeline,
            source_train_corpus=source_train_corpus,
            source_dev_corpus=source_dev_corpus,
            expected_artifact_sha256=artifact_sha,
            expected_receipt_sha256=receipt_sha,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


__all__ = [
    "LABRAM_K31_DEVELOPMENT_SCORE_PURPOSE",
    "LABRAM_K31_DEVELOPMENT_SCORE_RECEIPT_SCHEMA",
    "LABRAM_K31_DEVELOPMENT_SCORE_SCHEMA",
    "SignalTimelineRow",
    "TargetFreeOOFProtocolView",
    "TargetFreeSignalTimelineView",
    "VerifiedLaBraMK31DevelopmentScoreArtifact",
    "build_target_free_signal_timeline_view",
    "load_labram_k31_development_score_artifact",
    "load_target_free_ictal_oof_protocol",
    "materialize_labram_k31_development_scores",
]
