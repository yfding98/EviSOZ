"""Development-only LaBraM k31 scores from closed v1.2 producers.

This module deliberately reuses the target-free OOF protocol, signal timeline,
and frozen-token corpus implementation in :mod:`ictal_recovery_evidence` while
giving v1.2 producer bundles a separate, closed score-artifact capability.
Legacy v1/v1.1 recovery objects and bundles are not accepted.

Only two grids can be emitted:

* patient-OOF ``source_train`` scores with shape ``[E, 20, 60]``; and
* ``final``-producer ``source_dev`` scores with shape ``[E, 20, 60]``.

No DeepSOZ SOZ values, private labels/data, source-eval signal/event, source
annotation target, or caller-provided score/mask is an input to the
materializer.  The artifact remains retrospective ictal-involvement evidence
and is not authorized for formal promotion or SOZ reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import torch

from . import ictal_recovery_evidence as _shared
from .formal_token_corpus import VerifiedFormalTokenCorpusArtifact
from .ictal_native_eval import VerifiedIctalNativeEvalTokenCorpusArtifact
from .ictal_recovery_oof_v1_2 import (
    LABRAM_K31_OOF_RUN_SCHEMA_V1_2,
    LoadedLaBraMK31OOFRecoveryRunV12,
    load_labram_k31_oof_recovery_run_v1_2,
)


# These aliases are intentionally the one shared target-free implementation,
# rather than forks that could drift into reading another split or target.
SignalTimelineRow = _shared.SignalTimelineRow
TargetFreeOOFProtocolView = _shared.TargetFreeOOFProtocolView
TargetFreeSignalTimelineView = _shared.TargetFreeSignalTimelineView
build_target_free_signal_timeline_view = (
    _shared.build_target_free_signal_timeline_view
)
load_target_free_ictal_oof_protocol = _shared.load_target_free_ictal_oof_protocol

LABRAM_K31_DEVELOPMENT_SCORE_SCHEMA_V1_2 = (
    "soz_labram_k31_development_score_bundle_v1_2"
)
LABRAM_K31_DEVELOPMENT_SCORE_RECEIPT_SCHEMA_V1_2 = (
    "soz_labram_k31_development_score_receipt_v1_2"
)
LABRAM_K31_DEVELOPMENT_SCORE_PURPOSE_V1_2 = (
    "development_only_v1_2_retrospective_tusz_bipolar_edge_time_involvement"
)

MANIFEST_FILENAME = _shared.MANIFEST_FILENAME
RECEIPT_FILENAME = _shared.RECEIPT_FILENAME
SOURCE_TRAIN_SCORE_FILENAME = _shared.SOURCE_TRAIN_SCORE_FILENAME
SOURCE_TRAIN_DEPLOYMENT_FILENAME = _shared.SOURCE_TRAIN_DEPLOYMENT_FILENAME
SOURCE_TRAIN_PHASE_FILENAME = _shared.SOURCE_TRAIN_PHASE_FILENAME
SOURCE_DEV_SCORE_FILENAME = _shared.SOURCE_DEV_SCORE_FILENAME
SOURCE_DEV_DEPLOYMENT_FILENAME = _shared.SOURCE_DEV_DEPLOYMENT_FILENAME
SOURCE_DEV_PHASE_FILENAME = _shared.SOURCE_DEV_PHASE_FILENAME

_SELECTIONS = _shared._SELECTIONS
_TENSOR_FILENAMES = _shared._TENSOR_FILENAMES
_MAX_MANIFEST_BYTES = _shared._MAX_MANIFEST_BYTES
_ARTIFACT_MARKER_V1_2 = object()
_MANIFEST_FIELDS_V1_2 = frozenset(
    {
        *_shared._MANIFEST_FIELDS,
        "producer_bundle_schema_version",
        "v5_split_sha256",
        "deepsoz_target_source_loaded",
        "deepsoz_target_values_reachable",
        "tusz_ictal_involvement_targets_loaded",
        "reasoner_authorized",
    }
)
_RECEIPT_FIELDS_V1_2 = _shared._RECEIPT_FIELDS


def _strict_v12_recovery_runs(
    values: Sequence[LoadedLaBraMK31OOFRecoveryRunV12],
) -> tuple[LoadedLaBraMK31OOFRecoveryRunV12, ...]:
    """Replay exactly six v1.2 bundles; reject legacy loaded capabilities."""

    if isinstance(values, (str, bytes)):
        raise TypeError("v1.2 recovery_runs must be a sequence")
    indexed: dict[str, LoadedLaBraMK31OOFRecoveryRunV12] = {}
    for value in values:
        if type(value) is not LoadedLaBraMK31OOFRecoveryRunV12:
            raise TypeError(
                "recovery run must come from the strict v1.2 loader; "
                "v1/v1.1 capabilities are forbidden"
            )
        replay = load_labram_k31_oof_recovery_run_v1_2(
            value.path, expected_manifest_sha256=value.manifest_sha256
        )
        if type(replay) is not LoadedLaBraMK31OOFRecoveryRunV12:
            raise TypeError("v1.2 loader returned an invalid capability")
        if replay.manifest["schema_version"] != LABRAM_K31_OOF_RUN_SCHEMA_V1_2:
            raise ValueError("Recovery producer is not schema v1.2")
        selection = str(replay.manifest["selection"])
        if selection in indexed:
            raise ValueError(f"Duplicate v1.2 recovery producer {selection}")
        indexed[selection] = replay
    if set(indexed) != set(_SELECTIONS):
        raise ValueError("v1.2 scores require fold0..fold4 and final exactly once")
    ordered = tuple(indexed[selection] for selection in _SELECTIONS)
    v5_splits = {str(run.manifest["v5_split_sha256"]) for run in ordered}
    if len(v5_splits) != 1:
        raise ValueError("v1.2 recovery producers use different frozen v5 splits")
    return ordered


def _generate_score_grid_v1_2(
    runs: tuple[LoadedLaBraMK31OOFRecoveryRunV12, ...],
    protocol: TargetFreeOOFProtocolView,
    timeline: TargetFreeSignalTimelineView,
    source_train_corpus: VerifiedFormalTokenCorpusArtifact,
    source_dev_corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
) -> _shared._DevelopmentScoreGrid:
    """Use the shared target-free corpus replay with v1.2 heads only."""

    # The shared implementation is producer-structural after its caller has
    # issued a strict capability: it reads only manifest lineage, frozen heads,
    # target-free timeline rows, and target-free token corpora.
    return _shared._generate_score_grid(
        runs, protocol, timeline, source_train_corpus, source_dev_corpus
    )


def _producer_bindings_v1_2(
    runs: Sequence[LoadedLaBraMK31OOFRecoveryRunV12],
) -> list[dict[str, object]]:
    bindings: list[dict[str, object]] = []
    for run in runs:
        manifest = run.manifest
        bindings.append(
            {
                "selection": manifest["selection"],
                "oof_fold": manifest["oof_fold"],
                "producer_bundle_schema_version": manifest["schema_version"],
                "recovery_run_manifest_sha256": run.manifest_sha256,
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "head_state_sha256": manifest["head_state_sha256"],
                "oof_plan_receipt_sha256": manifest[
                    "oof_plan_receipt_sha256"
                ],
                "training_manifest_sha256": manifest[
                    "training_manifest_sha256"
                ],
                "training_corpus_index_sha256": manifest[
                    "training_corpus_index_sha256"
                ],
                "native_evaluation_manifest_sha256": manifest[
                    "native_evaluation_manifest_sha256"
                ],
                "native_evaluation_corpus_index_sha256": manifest[
                    "native_evaluation_corpus_index_sha256"
                ],
                "target_snapshot_manifest_sha256": manifest[
                    "target_snapshot_manifest_sha256"
                ],
                "target_snapshot_receipt_sha256": manifest[
                    "target_snapshot_receipt_sha256"
                ],
                "v5_split_sha256": manifest["v5_split_sha256"],
                "execution_receipt_sha256": manifest[
                    "execution_receipt_sha256"
                ],
                "deepsoz_target_source_loaded": manifest[
                    "deepsoz_target_source_loaded"
                ],
                "deepsoz_target_values_reachable": manifest[
                    "deepsoz_target_values_reachable"
                ],
                "tusz_ictal_involvement_targets_loaded": manifest[
                    "tusz_ictal_involvement_targets_loaded"
                ],
                "formal_promotion": manifest["formal_promotion"],
                "checkpoint_authorized_for_formal_evidence_or_reasoner": manifest[
                    "checkpoint_authorized_for_formal_evidence_or_reasoner"
                ],
            }
        )
    return bindings


def _manifest_payload_v1_2(
    *,
    runs: tuple[LoadedLaBraMK31OOFRecoveryRunV12, ...],
    protocol: TargetFreeOOFProtocolView,
    timeline: TargetFreeSignalTimelineView,
    source_train_corpus: VerifiedFormalTokenCorpusArtifact,
    source_dev_corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
    grid: _shared._DevelopmentScoreGrid,
    tensor_records: Mapping[str, object],
) -> dict[str, object]:
    payload = _shared._manifest_payload(
        runs=runs,
        protocol=protocol,
        timeline=timeline,
        source_train_corpus=source_train_corpus,
        source_dev_corpus=source_dev_corpus,
        grid=grid,
        tensor_records=tensor_records,
    )
    payload.update(
        {
            "schema_version": LABRAM_K31_DEVELOPMENT_SCORE_SCHEMA_V1_2,
            "purpose": LABRAM_K31_DEVELOPMENT_SCORE_PURPOSE_V1_2,
            "producer_bindings": _producer_bindings_v1_2(runs),
            "producer_bundle_schema_version": LABRAM_K31_OOF_RUN_SCHEMA_V1_2,
            "v5_split_sha256": runs[0].manifest["v5_split_sha256"],
            "deepsoz_target_source_loaded": False,
            "deepsoz_target_values_reachable": False,
            "tusz_ictal_involvement_targets_loaded": True,
            "reasoner_authorized": False,
        }
    )
    if set(payload) != set(_MANIFEST_FIELDS_V1_2):
        raise RuntimeError("Internal v1.2 score manifest schema drift")
    return payload


@dataclass(frozen=True, init=False)
class VerifiedLaBraMK31DevelopmentScoreArtifactV12:
    """Opaque v1.2 development scores without formal/reasoner authority."""

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
        if _verification_marker is not _ARTIFACT_MARKER_V1_2:
            raise TypeError(
                "VerifiedLaBraMK31DevelopmentScoreArtifactV12 can only be "
                "issued by the strict v1.2 loader"
            )
        for field, value in values.items():
            object.__setattr__(self, field, value)


def _expected_files() -> set[str]:
    return {MANIFEST_FILENAME, RECEIPT_FILENAME, *_TENSOR_FILENAMES.values()}


def _read_bundle_v1_2(
    path: str | Path,
    *,
    recovery_runs: Sequence[LoadedLaBraMK31OOFRecoveryRunV12],
    protocol: TargetFreeOOFProtocolView,
    timeline: TargetFreeSignalTimelineView,
    source_train_corpus: VerifiedFormalTokenCorpusArtifact,
    source_dev_corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> VerifiedLaBraMK31DevelopmentScoreArtifactV12:
    runs = _strict_v12_recovery_runs(recovery_runs)
    replay = _generate_score_grid_v1_2(
        runs, protocol, timeline, source_train_corpus, source_dev_corpus
    )
    source = _shared._strict_directory(path, _expected_files())
    manifest_raw = _shared._read_regular_bytes(
        source / MANIFEST_FILENAME, maximum_bytes=_MAX_MANIFEST_BYTES
    )
    receipt_raw = _shared._read_regular_bytes(
        source / RECEIPT_FILENAME, maximum_bytes=_MAX_MANIFEST_BYTES
    )
    artifact_sha = hashlib.sha256(manifest_raw).hexdigest()
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    if artifact_sha != _shared._require_sha256(
        expected_artifact_sha256, field="expected_artifact_sha256"
    ):
        raise ValueError("v1.2 development score artifact SHA mismatch")
    if receipt_sha != _shared._require_sha256(
        expected_receipt_sha256, field="expected_receipt_sha256"
    ):
        raise ValueError("v1.2 development score receipt SHA mismatch")
    manifest = _shared._closed_mapping(
        _shared._strict_json(manifest_raw, field="v1.2 development score manifest"),
        _MANIFEST_FIELDS_V1_2,
        field="v1.2 development score manifest",
    )
    receipt = _shared._closed_mapping(
        _shared._strict_json(receipt_raw, field="v1.2 development score receipt"),
        _RECEIPT_FIELDS_V1_2,
        field="v1.2 development score receipt",
    )
    if _shared._canonical_json_bytes(manifest) != manifest_raw or (
        _shared._canonical_json_bytes(receipt) != receipt_raw
    ):
        raise ValueError("v1.2 development score JSON is not canonical")
    fixed_boundaries = {
        "schema_version": LABRAM_K31_DEVELOPMENT_SCORE_SCHEMA_V1_2,
        "producer_bundle_schema_version": LABRAM_K31_OOF_RUN_SCHEMA_V1_2,
        "development_only": True,
        "formal_promotion": False,
        "authorized_for_formal_evidence_or_reasoner": False,
        "reasoner_authorized": False,
        "target_vectors_loaded": False,
        "target_values_present": False,
        "source_annotation_targets_present": False,
        "source_annotation_coverage_present": False,
        "private_data_used": False,
        "source_eval_signals_or_events_used": False,
        "deepsoz_target_source_loaded": False,
        "deepsoz_target_values_reachable": False,
        "tusz_ictal_involvement_targets_loaded": True,
    }
    if any(manifest.get(key) != value for key, value in fixed_boundaries.items()):
        raise ValueError("v1.2 development score scientific boundary changed")
    tensor_records = manifest.get("tensor_files")
    if not isinstance(tensor_records, dict) or set(tensor_records) != set(
        _TENSOR_FILENAMES
    ):
        raise ValueError("v1.2 development score tensor roster changed")
    tensors = {
        name: _shared._read_tensor(
            source,
            name=name,
            record=tensor_records[name],
            expected_filename=filename,
        )
        for name, filename in _TENSOR_FILENAMES.items()
    }
    expected_tensors = _shared._tensor_values(replay)
    if any(
        not torch.equal(tensors[name], expected_tensors[name])
        for name in _TENSOR_FILENAMES
    ):
        raise ValueError(
            "Stored v1.2 scores differ from strict checkpoint/token replay"
        )
    expected_manifest = _manifest_payload_v1_2(
        runs=runs,
        protocol=protocol,
        timeline=timeline,
        source_train_corpus=source_train_corpus,
        source_dev_corpus=source_dev_corpus,
        grid=replay,
        tensor_records=tensor_records,
    )
    if manifest != expected_manifest:
        raise ValueError("v1.2 score manifest changed strict lineage or boundaries")
    tensor_hashes = {
        name: _shared._tensor_sha256(name, tensors[name])
        for name in _TENSOR_FILENAMES
    }
    expected_receipt = {
        "schema_version": LABRAM_K31_DEVELOPMENT_SCORE_RECEIPT_SCHEMA_V1_2,
        "artifact_sha256": artifact_sha,
        "oof_protocol_receipt_sha256": protocol.receipt_sha256,
        "signal_timeline_receipt_sha256": timeline.receipt_sha256,
        "source_train_token_corpus_index_sha256": source_train_corpus.index_sha256,
        "source_dev_token_corpus_index_sha256": source_dev_corpus.index_sha256,
        "producer_binding_receipt_sha256": _shared._canonical_sha256(
            _producer_bindings_v1_2(runs)
        ),
        "source_train_event_row_receipt_sha256": _shared._canonical_sha256(
            list(replay.source_train_event_rows)
        ),
        "source_dev_event_row_receipt_sha256": _shared._canonical_sha256(
            list(replay.source_dev_event_rows)
        ),
        "tensor_sha256s": tensor_hashes,
    }
    if receipt != expected_receipt:
        raise ValueError("v1.2 score receipt does not bind exact replay")
    return VerifiedLaBraMK31DevelopmentScoreArtifactV12(
        _verification_marker=_ARTIFACT_MARKER_V1_2,
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


def load_labram_k31_development_score_artifact_v1_2(
    path: str | Path,
    *,
    recovery_runs: Sequence[LoadedLaBraMK31OOFRecoveryRunV12],
    protocol: TargetFreeOOFProtocolView,
    timeline: TargetFreeSignalTimelineView,
    source_train_corpus: VerifiedFormalTokenCorpusArtifact,
    source_dev_corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> VerifiedLaBraMK31DevelopmentScoreArtifactV12:
    """Strictly reload and reproduce a v1.2 development score artifact."""

    return _read_bundle_v1_2(
        path,
        recovery_runs=recovery_runs,
        protocol=protocol,
        timeline=timeline,
        source_train_corpus=source_train_corpus,
        source_dev_corpus=source_dev_corpus,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
    )


def materialize_labram_k31_development_scores_v1_2(
    *,
    recovery_runs: Sequence[LoadedLaBraMK31OOFRecoveryRunV12],
    protocol: TargetFreeOOFProtocolView,
    timeline: TargetFreeSignalTimelineView,
    source_train_corpus: VerifiedFormalTokenCorpusArtifact,
    source_dev_corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
    output_directory: str | Path,
) -> VerifiedLaBraMK31DevelopmentScoreArtifactV12:
    """Replay v1.2 producers without accepting scores, masks, or labels."""

    runs = _strict_v12_recovery_runs(recovery_runs)
    grid = _generate_score_grid_v1_2(
        runs, protocol, timeline, source_train_corpus, source_dev_corpus
    )
    target = _shared._safe_output(output_directory)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    published = False
    try:
        tensor_records = {
            name: _shared._write_tensor(
                temporary / _TENSOR_FILENAMES[name], name, value
            )
            for name, value in _shared._tensor_values(grid).items()
        }
        manifest = _manifest_payload_v1_2(
            runs=runs,
            protocol=protocol,
            timeline=timeline,
            source_train_corpus=source_train_corpus,
            source_dev_corpus=source_dev_corpus,
            grid=grid,
            tensor_records=tensor_records,
        )
        manifest_raw = _shared._canonical_json_bytes(manifest)
        artifact_sha = hashlib.sha256(manifest_raw).hexdigest()
        receipt = {
            "schema_version": LABRAM_K31_DEVELOPMENT_SCORE_RECEIPT_SCHEMA_V1_2,
            "artifact_sha256": artifact_sha,
            "oof_protocol_receipt_sha256": protocol.receipt_sha256,
            "signal_timeline_receipt_sha256": timeline.receipt_sha256,
            "source_train_token_corpus_index_sha256": source_train_corpus.index_sha256,
            "source_dev_token_corpus_index_sha256": source_dev_corpus.index_sha256,
            "producer_binding_receipt_sha256": _shared._canonical_sha256(
                _producer_bindings_v1_2(runs)
            ),
            "source_train_event_row_receipt_sha256": _shared._canonical_sha256(
                list(grid.source_train_event_rows)
            ),
            "source_dev_event_row_receipt_sha256": _shared._canonical_sha256(
                list(grid.source_dev_event_rows)
            ),
            "tensor_sha256s": {
                name: _shared._tensor_sha256(name, value)
                for name, value in _shared._tensor_values(grid).items()
            },
        }
        receipt_raw = _shared._canonical_json_bytes(receipt)
        receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
        (temporary / MANIFEST_FILENAME).write_bytes(manifest_raw)
        (temporary / RECEIPT_FILENAME).write_bytes(receipt_raw)
        _shared._fsync_file(temporary / MANIFEST_FILENAME)
        _shared._fsync_file(temporary / RECEIPT_FILENAME)
        _shared._fsync_directory(temporary)
        if os.path.lexists(target):
            raise FileExistsError(
                f"v1.2 development score output already exists: {target}"
            )
        os.rename(temporary, target)
        published = True
        _shared._fsync_directory(target.parent)
        return load_labram_k31_development_score_artifact_v1_2(
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
    "LABRAM_K31_DEVELOPMENT_SCORE_PURPOSE_V1_2",
    "LABRAM_K31_DEVELOPMENT_SCORE_RECEIPT_SCHEMA_V1_2",
    "LABRAM_K31_DEVELOPMENT_SCORE_SCHEMA_V1_2",
    "SignalTimelineRow",
    "TargetFreeOOFProtocolView",
    "TargetFreeSignalTimelineView",
    "VerifiedLaBraMK31DevelopmentScoreArtifactV12",
    "build_target_free_signal_timeline_view",
    "load_labram_k31_development_score_artifact_v1_2",
    "load_target_free_ictal_oof_protocol",
    "materialize_labram_k31_development_scores_v1_2",
]
