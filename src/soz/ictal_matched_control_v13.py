"""Strict fit-only matched independent-second control for v13.

The model-training process accepts only two physical inputs: a fit-only
target/event artifact and a selectively opened fit-token view.  It does not
load the legacy k31 manifest (which contains native roster/metric metadata),
the original fit+gate TUSZ manifest (which contains row-level gate-derived
hashes/counts), native evaluation data, I-gate signal/outcome, DeepSOZ, or
private data.  The exact k31 fit authority is projected into the fit-only
artifact by the preceding broker.  That projection honestly retains pinned
full-source snapshot/file identities, but no gate row-level target receipt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from .cached_concept_training import IctalTokenBagDataset, IctalTokenPatientBag
from .concept_token_io import load_labram_concept_tokens
from .formal_token_corpus import VerifiedFormalTokenCorpusSubsetArtifact
from .ictal_fit_only_consumer_v13 import (
    FitOnlyTargetEventV13,
    LoadedFitOnlyTargetArtifactV13,
)
from .ictal_fit_token_view_consumer_v13 import LoadedFitTokenViewV13
from .ictal_fit_primitives_v13 import (
    IctalTrainingConfig,
    LABRAM_K31_TARGET_SEMANTICS,
    canonical_json_bytes as _canonical_json_bytes,
    file_sha256 as _file_sha256,
    ictal_head_state_sha256,
    patient_roster as _patient_roster,
    patient_roster_sha256,
    require_sha256 as _require_sha256,
    safe_new_output as _safe_new_output,
    selection as _selection,
    validate_epoch_payload as _validated_epoch_payload,
    validate_execution_receipt as _validate_execution_receipt,
)
from .models.concept_heads import (
    CapacityMatchedChannelResidualIctalInvolvementHead,
    IctalInvolvementHead,
)


V13_MATCHED_CONTROL_SCHEMA = "soz_labram_ictal_independent_control_v13_1"
V13_MATCHED_CONTROL_CANDIDATE = "labram_independent_second_matched_control"
V13_MATCHED_CONTROL_ROLE = "secondary_unmatched_capacity_diagnostic"
V13_CAPACITY_MATCHED_CONTROL_SCHEMA = (
    "soz_labram_ictal_capacity_matched_channel_control_v13_1"
)
V13_CAPACITY_MATCHED_CONTROL_CANDIDATE = (
    "labram_capacity_matched_channel_only_residual_control"
)
V13_CAPACITY_MATCHED_CONTROL_ROLE = "primary_capacity_matched_confirmation_control"
V13_MANIFEST_FILENAME = "control_run.json"
V13_CHECKPOINT_FILENAME = "model.safetensors"
V13_INDEPENDENT_HEAD_PARAMETER_COUNT = 77_313
V13_K31_EXTRA_PARAMETER_COUNT = 4_352
V13_CAPACITY_MATCHED_HEAD_PARAMETER_COUNT = (
    V13_INDEPENDENT_HEAD_PARAMETER_COUNT + V13_K31_EXTRA_PARAMETER_COUNT
)
_MAX_JSON_BYTES = 16 * 1024 * 1024
_TRAINING_RUN_FIELDS = frozenset(
    {
        "initial_state_sha256",
        "final_state_sha256",
        "epoch_rows",
        "evaluation_performed",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "candidate",
        "control_role",
        "selection",
        "oof_fold",
        "head_class",
        "head_temporal_mixing",
        "target_semantics",
        "development_only",
        "development_confirmation_control",
        "candidate_selection_authorized",
        "architecture_selected_after_opened_i_dev",
        "formal_promotion",
        "checkpoint_authorized_for_formal_evidence_or_reasoner",
        "gate_opened",
        "i_gate_signal_or_tokens_opened",
        "i_gate_outcomes_opened",
        "i_gate_target_values_materialized",
        "i_gate_target_values_evaluated",
        "non_fit_token_bundles_opened",
        "legacy_k31_full_manifest_loaded",
        "legacy_k31_native_roster_or_metrics_loaded",
        "full_training_manifest_loaded",
        "gate_row_level_target_derived_hashes_counts_loaded",
        "source_full_target_file_or_snapshot_hashes_loaded",
        "native_evaluation_inputs_loaded",
        "native_evaluation_performed",
        "native_metrics_computed",
        "deepsoz_identity_outcome_prediction_reachable",
        "deepsoz_target_source_loaded",
        "deepsoz_target_values_reachable",
        "deepsoz_soz_labels_used",
        "private_signal_identity_outcome_reachable",
        "private_labels_used",
        "tusz_ictal_involvement_targets_loaded",
        "missing_tusz_cells_imputed_as_negative",
        "fit_only_target_artifact_loaded",
        "fit_only_event_manifest_loaded",
        "source_full_target_arrays_loaded",
        "source_full_target_arrays_mapped",
        "matched_k31_manifest_sha256",
        "matched_k31_checkpoint_sha256",
        "matched_k31_candidate",
        "matched_training_config_exact",
        "matched_training_roster_exact",
        "matched_training_manifest_exact",
        "matched_training_corpus_exact",
        "matched_target_rows_exact",
        "training_manifest_bundle_sha256",
        "training_manifest_sha256",
        "training_corpus_index_sha256",
        "fit_only_target_manifest_sha256",
        "fit_only_target_receipt_sha256",
        "fit_token_view_manifest_sha256",
        "fit_token_view_receipt_sha256",
        "source_target_snapshot_manifest_sha256",
        "source_target_snapshot_receipt_sha256",
        "fit_token_subset_patient_roster_sha256",
        "fit_token_subset_event_roster_sha256",
        "fit_token_subset_tensor_roster_sha256",
        "fit_event_count",
        "training_public_patient_ids",
        "training_public_roster_sha256",
        "i_gate_patient_ids_excluded_unopened",
        "i_gate_patient_roster_sha256",
        "training_config",
        "execution_receipt",
        "execution_receipt_sha256",
        "training_run",
        "head_config",
        "head_total_parameter_count",
        "head_extra_parameter_count_over_independent",
        "k31_extra_parameter_count",
        "capacity_matched_to_k31",
        "head_state_sha256",
        "checkpoint_filename",
        "checkpoint_sha256",
    }
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is required")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"v13 control output already exists: {target}")
        raise OSError(error, os.strerror(error), str(target))


def _validate_fit_only_training_run(
    value: object,
    *,
    head_state_sha256: str,
    training_patient_count: int,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("training_run must be a mapping")
    run = dict(value)
    if set(run) != _TRAINING_RUN_FIELDS or run["evaluation_performed"] is not False:
        raise ValueError("v13 training_run violates the fit-only schema")
    initial = _require_sha256(
        run["initial_state_sha256"], field="training_run.initial_state_sha256"
    )
    final = _require_sha256(
        run["final_state_sha256"], field="training_run.final_state_sha256"
    )
    if final != _require_sha256(head_state_sha256, field="head_state_sha256"):
        raise ValueError("v13 training final state differs from the checkpoint")
    if initial == final:
        raise ValueError("v13 training shows no optimizer-induced state change")
    rows = run["epoch_rows"]
    if not isinstance(rows, list) or len(rows) != IctalTrainingConfig().fixed_epochs:
        raise ValueError("v13 training must retain every fixed epoch")
    epochs = [
        _validated_epoch_payload(
            row,
            field=f"training_run.epoch_rows[{index}]",
            expected_patient_count=training_patient_count,
        )
        for index, row in enumerate(rows)
    ]
    return {
        "initial_state_sha256": initial,
        "final_state_sha256": final,
        "epoch_rows": epochs,
        "evaluation_performed": False,
    }


@dataclass(frozen=True)
class MatchedControlLineage:
    selection: str
    oof_fold: int | None
    matched_k31_manifest_sha256: str
    matched_k31_checkpoint_sha256: str
    training_manifest_bundle_sha256: str
    training_manifest_sha256: str
    training_corpus_index_sha256: str
    fit_only_target_manifest_sha256: str
    fit_only_target_receipt_sha256: str
    fit_token_view_manifest_sha256: str
    fit_token_view_receipt_sha256: str
    source_target_snapshot_manifest_sha256: str
    source_target_snapshot_receipt_sha256: str
    fit_token_subset_patient_roster_sha256: str
    fit_token_subset_event_roster_sha256: str
    fit_token_subset_tensor_roster_sha256: str
    fit_event_count: int
    training_public_patient_ids: tuple[str, ...]
    i_gate_patient_ids_excluded_unopened: tuple[str, ...]


def validate_matched_control_lineage(
    *,
    selection: str,
    fit_only_targets: LoadedFitOnlyTargetArtifactV13,
    fit_token_view: LoadedFitTokenViewV13,
) -> MatchedControlLineage:
    """Validate the brokered minimal authority without loading legacy k31."""

    if not isinstance(fit_only_targets, LoadedFitOnlyTargetArtifactV13):
        raise TypeError("fit_only_targets must come from the strict v13 loader")
    if not isinstance(fit_token_view, LoadedFitTokenViewV13):
        raise TypeError("fit_token_view must come from the physical-view loader")
    fit_token_corpus = fit_token_view.corpus
    target = fit_only_targets.manifest
    normalized, fold = _selection(selection)
    if target.get("selection") != normalized or target.get("oof_fold") != fold:
        raise ValueError("Fit-only target selection differs from the requested control")
    if target.get("matched_training_config") != asdict(IctalTrainingConfig()):
        raise ValueError("Brokered k31 training config differs from the frozen policy")
    fit = _patient_roster(
        target.get("fit_patient_ids"), field="fit_patient_ids", allow_empty=False
    )
    gate = _patient_roster(
        target.get("i_gate_patient_ids_excluded_unopened"),
        field="i_gate_patient_ids_excluded_unopened",
        allow_empty=False,
    )
    if len(gate) != 12 or set(fit) & set(gate):
        raise ValueError("Matched control fit/I-gate firewall failed")
    if (
        target.get("trainer_requires_full_training_manifest") is not False
        or target.get("trainer_requires_full_k31_manifest") is not False
        or fit_token_view.manifest.get("trainer_source_full_corpus_root_reachable")
        is not False
        or fit_token_view.manifest.get("trainer_imports_full_corpus_loader") is not False
        or fit_token_view.manifest.get("matched_k31_manifest_sha256")
        != target.get("matched_k31_manifest_sha256")
        or fit_token_corpus.selected_patient_ids != fit
        or fit_token_corpus.unselected_event_bundles_opened is not False
        or fit_token_corpus.index_sha256 != target.get("training_corpus_index_sha256")
        or fit_token_corpus.training_bundle_manifest_sha256
        != target.get("training_manifest_bundle_sha256")
        or fit_token_corpus.training_source_manifest_sha256
        != target.get("training_manifest_sha256")
    ):
        raise ValueError("Fit-only token corpus differs from brokered k31 authority")
    target_rows = tuple((event.event_id, event.patient_id) for event in fit_only_targets.events)
    token_rows = tuple((event.event_id, event.patient_id) for event in fit_token_corpus.events)
    if target_rows != token_rows or len(target_rows) != int(target["fit_event_count"]):
        raise ValueError("Fit-only target/token event rosters differ")
    return MatchedControlLineage(
        selection=normalized,
        oof_fold=fold,
        matched_k31_manifest_sha256=_require_sha256(
            target.get("matched_k31_manifest_sha256"),
            field="matched_k31_manifest_sha256",
        ),
        matched_k31_checkpoint_sha256=_require_sha256(
            target.get("matched_k31_checkpoint_sha256"),
            field="matched_k31_checkpoint_sha256",
        ),
        training_manifest_bundle_sha256=_require_sha256(
            target.get("training_manifest_bundle_sha256"),
            field="training_manifest_bundle_sha256",
        ),
        training_manifest_sha256=_require_sha256(
            target.get("training_manifest_sha256"),
            field="training_manifest_sha256",
        ),
        training_corpus_index_sha256=_require_sha256(
            target.get("training_corpus_index_sha256"),
            field="training_corpus_index_sha256",
        ),
        fit_only_target_manifest_sha256=fit_only_targets.manifest_sha256,
        fit_only_target_receipt_sha256=fit_only_targets.receipt_sha256,
        fit_token_view_manifest_sha256=fit_token_view.manifest_sha256,
        fit_token_view_receipt_sha256=fit_token_view.receipt_sha256,
        source_target_snapshot_manifest_sha256=_require_sha256(
            target.get("source_target_snapshot_manifest_sha256"),
            field="source_target_snapshot_manifest_sha256",
        ),
        source_target_snapshot_receipt_sha256=_require_sha256(
            target.get("source_target_snapshot_receipt_sha256"),
            field="source_target_snapshot_receipt_sha256",
        ),
        fit_token_subset_patient_roster_sha256=(
            fit_token_corpus.selected_patient_roster_sha256
        ),
        fit_token_subset_event_roster_sha256=(
            fit_token_corpus.selected_event_roster_sha256
        ),
        fit_token_subset_tensor_roster_sha256=(
            fit_token_corpus.selected_tensor_roster_sha256
        ),
        fit_event_count=len(target_rows),
        training_public_patient_ids=fit,
        i_gate_patient_ids_excluded_unopened=gate,
    )


def build_fit_only_token_bag_dataset_v13(
    corpus: VerifiedFormalTokenCorpusSubsetArtifact,
    targets: LoadedFitOnlyTargetArtifactV13,
) -> IctalTokenBagDataset:
    """Join fit-only event metadata, target rows, and selective token bundles."""

    if not isinstance(corpus, VerifiedFormalTokenCorpusSubsetArtifact):
        raise TypeError("corpus must come from the selective strict loader")
    if not isinstance(targets, LoadedFitOnlyTargetArtifactV13):
        raise TypeError("targets must come from the fit-only strict loader")
    manifest = targets.manifest
    fit = _patient_roster(
        manifest.get("fit_patient_ids"), field="fit_patient_ids", allow_empty=False
    )
    gate = _patient_roster(
        manifest.get("i_gate_patient_ids_excluded_unopened"),
        field="i_gate_patient_ids_excluded_unopened",
        allow_empty=False,
    )
    if (
        set(fit) & set(gate)
        or corpus.selected_patient_ids != fit
        or corpus.unselected_event_bundles_opened is not False
        or corpus.training_source_manifest_sha256
        != manifest.get("training_manifest_sha256")
        or targets.snapshot.training_manifest_sha256
        != manifest.get("training_manifest_sha256")
    ):
        raise ValueError("Fit-only dataset lineage/firewall differs")
    target_events = targets.events
    target_rows = tuple((event.event_id, event.patient_id) for event in target_events)
    token_rows = tuple((event.event_id, event.patient_id) for event in corpus.events)
    if target_rows != token_rows:
        raise ValueError("Fit-only token and event-manifest rosters differ")
    binding_by_event = {binding.event_id: binding for binding in corpus.events}
    target_index = {event.event_id: index for index, event in enumerate(target_events)}
    first = corpus.events[0]
    first_token = load_labram_concept_tokens(
        first.bundle_path,
        expected_manifest_sha256=first.bundle_manifest_sha256,
    )
    foundation_receipt = first_token.foundation_feature_receipt_sha256
    foundation_checkpoint = first_token.foundation_checkpoint_sha256
    training_manifest_sha = str(manifest["training_manifest_sha256"])

    def patient_events(patient_id: str) -> tuple[FitOnlyTargetEventV13, ...]:
        events = tuple(event for event in target_events if event.patient_id == patient_id)
        if not events:
            raise KeyError(f"Patient is absent from fit-only events: {patient_id}")
        return events

    def load_patient(patient_id: str) -> IctalTokenPatientBag:
        if patient_id not in set(fit):
            raise KeyError(f"Patient is outside the v13 fit-only view: {patient_id}")
        events = patient_events(patient_id)
        token_events = []
        patient_targets = []
        patient_masks = []
        for event in events:
            binding = binding_by_event[event.event_id]
            if (
                binding.patient_id != patient_id
                or binding.event_record_sha256 != event.event_record_sha256
                or binding.preprocess_receipt_sha256
                != event.preprocess_receipt_sha256
            ):
                raise ValueError("Fit-only token binding differs from event manifest")
            token = load_labram_concept_tokens(
                binding.bundle_path,
                expected_manifest_sha256=binding.bundle_manifest_sha256,
            )
            checks = (
                token.event_id == event.event_id,
                token.source_concept_manifest_sha256 == training_manifest_sha,
                token.event_record_sha256 == event.event_record_sha256,
                token.preprocess_receipt_sha256 == event.preprocess_receipt_sha256,
                token.foundation_feature_receipt_sha256 == foundation_receipt,
                token.foundation_checkpoint_sha256 == foundation_checkpoint,
            )
            if not all(checks):
                raise ValueError("Fit-only token/event lineage mismatch")
            row = target_index[event.event_id]
            token_events.append(token)
            patient_targets.append(targets.snapshot.training_targets[row].clone())
            patient_masks.append(targets.snapshot.training_target_mask[row].clone())
        event_ids = tuple(event.event_id for event in events)
        return IctalTokenPatientBag(
            patient_id=patient_id,
            event_ids=event_ids,
            expected_event_ids=event_ids,
            training_manifest_sha256=training_manifest_sha,
            expected_event_record_sha256s=tuple(
                event.event_record_sha256 for event in events
            ),
            token_events=tuple(token_events),
            targets=torch.stack(patient_targets, dim=0).to(torch.float32),
            target_mask=torch.stack(patient_masks, dim=0).to(torch.bool),
        )

    return IctalTokenBagDataset(
        fit,
        load_patient,
        training_manifest_sha256=training_manifest_sha,
        token_source_manifest_sha256=training_manifest_sha,
        foundation_feature_receipt_sha256=foundation_receipt,
        formal_token_corpus_verified=True,
        formal_token_corpus_index_sha256=corpus.index_sha256,
        formal_token_corpus_training_bundle_manifest_sha256=(
            corpus.training_bundle_manifest_sha256
        ),
        formal_token_corpus_event_roster_sha256=corpus.selected_event_roster_sha256,
        formal_token_corpus_patient_roster_sha256=(
            corpus.selected_patient_roster_sha256
        ),
        formal_token_corpus_tensor_roster_sha256=(
            corpus.selected_tensor_roster_sha256
        ),
    )


@dataclass(frozen=True)
class LoadedMatchedIndependentControlV13:
    path: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    head: IctalInvolvementHead


@dataclass(frozen=True)
class LoadedCapacityMatchedChannelControlV13:
    path: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    head: CapacityMatchedChannelResidualIctalInvolvementHead


def _save_control_v13(
    output_directory: str | Path,
    *,
    lineage: MatchedControlLineage,
    head: IctalInvolvementHead,
    training_config: Mapping[str, object],
    execution_receipt: Mapping[str, object],
    training_run: Mapping[str, object],
    schema: str,
    candidate: str,
    control_role: str,
    exact_head_type: type[IctalInvolvementHead],
    head_class_name: str,
    temporal_mixing: str,
    head_config: Mapping[str, object],
    expected_total_parameter_count: int,
    capacity_matched_to_k31: bool,
) -> LoadedMatchedIndependentControlV13 | LoadedCapacityMatchedChannelControlV13:
    if not isinstance(lineage, MatchedControlLineage):
        raise TypeError("lineage must be MatchedControlLineage")
    if type(head) is not exact_head_type:
        raise TypeError(f"v13 control requires the exact {head_class_name} class")
    first_layer = head.adapter[0]
    if (
        not isinstance(first_layer, torch.nn.Linear)
        or int(head.edge_tokens.token_dim) != 200
        or int(first_layer.out_features) != 128
    ):
        raise ValueError("v13 control head configuration changed")
    parameter_count = sum(parameter.numel() for parameter in head.parameters())
    if parameter_count != expected_total_parameter_count:
        raise ValueError("v13 control head parameter count changed")
    config = dict(training_config)
    if config != asdict(IctalTrainingConfig()):
        raise ValueError("v13 control training config is not exactly matched")
    state_sha = ictal_head_state_sha256(head)
    run = _validate_fit_only_training_run(
        training_run,
        head_state_sha256=state_sha,
        training_patient_count=len(lineage.training_public_patient_ids),
    )
    execution = _validate_execution_receipt(execution_receipt, training_config=config)
    target = _safe_new_output(output_directory)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.v13-", dir=target.parent))
    published = False
    try:
        checkpoint = staging / V13_CHECKPOINT_FILENAME
        save_file(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in head.state_dict().items()
            },
            str(checkpoint),
        )
        checkpoint_sha = _file_sha256(checkpoint)
        fit = lineage.training_public_patient_ids
        gate = lineage.i_gate_patient_ids_excluded_unopened
        payload = {
            "schema_version": schema,
            "candidate": candidate,
            "control_role": control_role,
            "selection": lineage.selection,
            "oof_fold": lineage.oof_fold,
            "head_class": head_class_name,
            "head_temporal_mixing": temporal_mixing,
            "target_semantics": LABRAM_K31_TARGET_SEMANTICS,
            "development_only": True,
            "development_confirmation_control": True,
            "candidate_selection_authorized": False,
            "architecture_selected_after_opened_i_dev": True,
            "formal_promotion": False,
            "checkpoint_authorized_for_formal_evidence_or_reasoner": False,
            "gate_opened": False,
            "i_gate_signal_or_tokens_opened": False,
            "i_gate_outcomes_opened": False,
            "i_gate_target_values_materialized": False,
            "i_gate_target_values_evaluated": False,
            "non_fit_token_bundles_opened": False,
            "legacy_k31_full_manifest_loaded": False,
            "legacy_k31_native_roster_or_metrics_loaded": False,
            "full_training_manifest_loaded": False,
            "gate_row_level_target_derived_hashes_counts_loaded": False,
            "source_full_target_file_or_snapshot_hashes_loaded": True,
            "native_evaluation_inputs_loaded": False,
            "native_evaluation_performed": False,
            "native_metrics_computed": False,
            "deepsoz_identity_outcome_prediction_reachable": False,
            "deepsoz_target_source_loaded": False,
            "deepsoz_target_values_reachable": False,
            "deepsoz_soz_labels_used": False,
            "private_signal_identity_outcome_reachable": False,
            "private_labels_used": False,
            "tusz_ictal_involvement_targets_loaded": True,
            "missing_tusz_cells_imputed_as_negative": False,
            "fit_only_target_artifact_loaded": True,
            "fit_only_event_manifest_loaded": True,
            "source_full_target_arrays_loaded": False,
            "source_full_target_arrays_mapped": False,
            "matched_k31_manifest_sha256": lineage.matched_k31_manifest_sha256,
            "matched_k31_checkpoint_sha256": lineage.matched_k31_checkpoint_sha256,
            "matched_k31_candidate": "labram_temporal_residual_k31",
            "matched_training_config_exact": True,
            "matched_training_roster_exact": True,
            "matched_training_manifest_exact": True,
            "matched_training_corpus_exact": True,
            "matched_target_rows_exact": True,
            "training_manifest_bundle_sha256": lineage.training_manifest_bundle_sha256,
            "training_manifest_sha256": lineage.training_manifest_sha256,
            "training_corpus_index_sha256": lineage.training_corpus_index_sha256,
            "fit_only_target_manifest_sha256": lineage.fit_only_target_manifest_sha256,
            "fit_only_target_receipt_sha256": lineage.fit_only_target_receipt_sha256,
            "fit_token_view_manifest_sha256": lineage.fit_token_view_manifest_sha256,
            "fit_token_view_receipt_sha256": lineage.fit_token_view_receipt_sha256,
            "source_target_snapshot_manifest_sha256": lineage.source_target_snapshot_manifest_sha256,
            "source_target_snapshot_receipt_sha256": lineage.source_target_snapshot_receipt_sha256,
            "fit_token_subset_patient_roster_sha256": lineage.fit_token_subset_patient_roster_sha256,
            "fit_token_subset_event_roster_sha256": lineage.fit_token_subset_event_roster_sha256,
            "fit_token_subset_tensor_roster_sha256": lineage.fit_token_subset_tensor_roster_sha256,
            "fit_event_count": lineage.fit_event_count,
            "training_public_patient_ids": list(fit),
            "training_public_roster_sha256": patient_roster_sha256(fit),
            "i_gate_patient_ids_excluded_unopened": list(gate),
            "i_gate_patient_roster_sha256": patient_roster_sha256(gate),
            "training_config": config,
            "execution_receipt": execution,
            "execution_receipt_sha256": _canonical_sha256(execution),
            "training_run": run,
            "head_config": dict(head_config),
            "head_total_parameter_count": parameter_count,
            "head_extra_parameter_count_over_independent": (
                parameter_count - V13_INDEPENDENT_HEAD_PARAMETER_COUNT
            ),
            "k31_extra_parameter_count": V13_K31_EXTRA_PARAMETER_COUNT,
            "capacity_matched_to_k31": capacity_matched_to_k31,
            "head_state_sha256": state_sha,
            "checkpoint_filename": V13_CHECKPOINT_FILENAME,
            "checkpoint_sha256": checkpoint_sha,
        }
        raw = _canonical_json_bytes(payload)
        if set(payload) != _MANIFEST_FIELDS or not 1 <= len(raw) <= _MAX_JSON_BYTES:
            raise ValueError("v13 control manifest violates its closed schema")
        (staging / V13_MANIFEST_FILENAME).write_bytes(raw)
        _fsync_file(checkpoint)
        _fsync_file(staging / V13_MANIFEST_FILENAME)
        _fsync_directory(staging)
        _rename_noreplace(staging, target)
        _fsync_directory(target.parent)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return _load_control_v13(
        target,
        expected_manifest_sha256=_file_sha256(target / V13_MANIFEST_FILENAME),
        schema=schema,
        candidate=candidate,
        control_role=control_role,
        exact_head_type=exact_head_type,
        loaded_type=(
            LoadedCapacityMatchedChannelControlV13
            if exact_head_type is CapacityMatchedChannelResidualIctalInvolvementHead
            else LoadedMatchedIndependentControlV13
        ),
        head_class_name=head_class_name,
        temporal_mixing=temporal_mixing,
        head_config=dict(head_config),
        expected_total_parameter_count=expected_total_parameter_count,
        capacity_matched_to_k31=capacity_matched_to_k31,
    )


def _load_control_v13(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    schema: str,
    candidate: str,
    control_role: str,
    exact_head_type: type[IctalInvolvementHead],
    loaded_type: type[
        LoadedMatchedIndependentControlV13 | LoadedCapacityMatchedChannelControlV13
    ],
    head_class_name: str,
    temporal_mixing: str,
    head_config: Mapping[str, object],
    expected_total_parameter_count: int,
    capacity_matched_to_k31: bool,
) -> LoadedMatchedIndependentControlV13 | LoadedCapacityMatchedChannelControlV13:
    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_dir() or source.resolve() != source:
        raise ValueError("v13 control bundle must be a regular absolute directory")
    if {item.name for item in source.iterdir()} != {
        V13_MANIFEST_FILENAME,
        V13_CHECKPOINT_FILENAME,
    }:
        raise ValueError("v13 control bundle has missing or unknown files")
    raw = (source / V13_MANIFEST_FILENAME).read_bytes()
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("v13 control manifest is invalid JSON") from exc
    if (
        not isinstance(manifest, dict)
        or set(manifest) != _MANIFEST_FIELDS
        or _canonical_json_bytes(manifest) != raw
    ):
        raise ValueError("v13 control manifest violates its canonical schema")
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None and manifest_sha != _require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise ValueError("v13 control manifest SHA mismatch")
    fixed = {
        "schema_version": schema,
        "candidate": candidate,
        "control_role": control_role,
        "head_class": head_class_name,
        "head_temporal_mixing": temporal_mixing,
        "target_semantics": LABRAM_K31_TARGET_SEMANTICS,
        "development_only": True,
        "development_confirmation_control": True,
        "candidate_selection_authorized": False,
        "architecture_selected_after_opened_i_dev": True,
        "formal_promotion": False,
        "checkpoint_authorized_for_formal_evidence_or_reasoner": False,
        "gate_opened": False,
        "i_gate_signal_or_tokens_opened": False,
        "i_gate_outcomes_opened": False,
        "i_gate_target_values_materialized": False,
        "i_gate_target_values_evaluated": False,
        "non_fit_token_bundles_opened": False,
        "legacy_k31_full_manifest_loaded": False,
        "legacy_k31_native_roster_or_metrics_loaded": False,
        "full_training_manifest_loaded": False,
        "gate_row_level_target_derived_hashes_counts_loaded": False,
        "source_full_target_file_or_snapshot_hashes_loaded": True,
        "native_evaluation_inputs_loaded": False,
        "native_evaluation_performed": False,
        "native_metrics_computed": False,
        "deepsoz_identity_outcome_prediction_reachable": False,
        "deepsoz_target_source_loaded": False,
        "deepsoz_target_values_reachable": False,
        "deepsoz_soz_labels_used": False,
        "private_signal_identity_outcome_reachable": False,
        "private_labels_used": False,
        "tusz_ictal_involvement_targets_loaded": True,
        "missing_tusz_cells_imputed_as_negative": False,
        "fit_only_target_artifact_loaded": True,
        "fit_only_event_manifest_loaded": True,
        "source_full_target_arrays_loaded": False,
        "source_full_target_arrays_mapped": False,
        "matched_k31_candidate": "labram_temporal_residual_k31",
        "matched_training_config_exact": True,
        "matched_training_roster_exact": True,
        "matched_training_manifest_exact": True,
        "matched_training_corpus_exact": True,
        "matched_target_rows_exact": True,
        "head_total_parameter_count": expected_total_parameter_count,
        "head_extra_parameter_count_over_independent": (
            expected_total_parameter_count - V13_INDEPENDENT_HEAD_PARAMETER_COUNT
        ),
        "k31_extra_parameter_count": V13_K31_EXTRA_PARAMETER_COUNT,
        "capacity_matched_to_k31": capacity_matched_to_k31,
        "checkpoint_filename": V13_CHECKPOINT_FILENAME,
    }
    if any(manifest.get(field) != value for field, value in fixed.items()):
        raise ValueError("v13 control changed a scientific/access boundary")
    _, fold = _selection(manifest.get("selection"))
    if manifest.get("oof_fold") != fold:
        raise ValueError("v13 control selection/fold mismatch")
    hashes = (
        "matched_k31_manifest_sha256",
        "matched_k31_checkpoint_sha256",
        "training_manifest_bundle_sha256",
        "training_manifest_sha256",
        "training_corpus_index_sha256",
        "fit_only_target_manifest_sha256",
        "fit_only_target_receipt_sha256",
        "fit_token_view_manifest_sha256",
        "fit_token_view_receipt_sha256",
        "source_target_snapshot_manifest_sha256",
        "source_target_snapshot_receipt_sha256",
        "fit_token_subset_patient_roster_sha256",
        "fit_token_subset_event_roster_sha256",
        "fit_token_subset_tensor_roster_sha256",
        "training_public_roster_sha256",
        "i_gate_patient_roster_sha256",
        "execution_receipt_sha256",
        "head_state_sha256",
        "checkpoint_sha256",
    )
    for field in hashes:
        _require_sha256(manifest.get(field), field=field)
    fit = _patient_roster(
        manifest.get("training_public_patient_ids"),
        field="training_public_patient_ids",
        allow_empty=False,
    )
    gate = _patient_roster(
        manifest.get("i_gate_patient_ids_excluded_unopened"),
        field="i_gate_patient_ids_excluded_unopened",
        allow_empty=False,
    )
    if (
        len(gate) != 12
        or set(fit) & set(gate)
        or patient_roster_sha256(fit) != manifest["training_public_roster_sha256"]
        or patient_roster_sha256(gate) != manifest["i_gate_patient_roster_sha256"]
    ):
        raise ValueError("v13 control patient firewall failed")
    if (
        isinstance(manifest.get("fit_event_count"), bool)
        or not isinstance(manifest.get("fit_event_count"), int)
        or manifest["fit_event_count"] < 1
    ):
        raise ValueError("v13 fit event count is invalid")
    config = dict(manifest.get("training_config", {}))
    if config != asdict(IctalTrainingConfig()):
        raise ValueError("v13 training config changed")
    run = _validate_fit_only_training_run(
        manifest.get("training_run"),
        head_state_sha256=str(manifest["head_state_sha256"]),
        training_patient_count=len(fit),
    )
    execution = _validate_execution_receipt(
        manifest.get("execution_receipt"), training_config=config
    )
    if _canonical_sha256(execution) != manifest["execution_receipt_sha256"]:
        raise ValueError("v13 execution receipt SHA mismatch")
    if manifest.get("head_config") != dict(head_config):
        raise ValueError("v13 head configuration changed")
    checkpoint = source / V13_CHECKPOINT_FILENAME
    if _file_sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("v13 checkpoint SHA mismatch")
    state = load_file(str(checkpoint), device="cpu")
    head = exact_head_type(token_dim=200, hidden_dim=128)
    if sum(parameter.numel() for parameter in head.parameters()) != (
        expected_total_parameter_count
    ):
        raise ValueError("v13 reconstructed head parameter count changed")
    expected = head.state_dict()
    if set(state) != set(expected):
        raise ValueError("v13 checkpoint tensor names changed")
    for name, reference in expected.items():
        value = state[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"v13 checkpoint tensor changed: {name}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"v13 checkpoint tensor is non-finite: {name}")
    head.load_state_dict(state, strict=True)
    if ictal_head_state_sha256(head) != manifest["head_state_sha256"]:
        raise ValueError("v13 head-state receipt mismatch")
    head.eval()
    return loaded_type(
        path=source,
        manifest={**manifest, "training_config": config, "training_run": run},
        manifest_sha256=manifest_sha,
        head=head,
    )


def save_matched_independent_control_v13(
    output_directory: str | Path,
    *,
    lineage: MatchedControlLineage,
    head: IctalInvolvementHead,
    training_config: Mapping[str, object],
    execution_receipt: Mapping[str, object],
    training_run: Mapping[str, object],
) -> LoadedMatchedIndependentControlV13:
    """Seal the secondary, capacity-unmatched naked-head diagnostic."""

    loaded = _save_control_v13(
        output_directory,
        lineage=lineage,
        head=head,
        training_config=training_config,
        execution_receipt=execution_receipt,
        training_run=training_run,
        schema=V13_MATCHED_CONTROL_SCHEMA,
        candidate=V13_MATCHED_CONTROL_CANDIDATE,
        control_role=V13_MATCHED_CONTROL_ROLE,
        exact_head_type=IctalInvolvementHead,
        head_class_name="IctalInvolvementHead",
        temporal_mixing="none_independent_seconds_on_cached_tokens",
        head_config={"token_dim": 200, "hidden_dim": 128},
        expected_total_parameter_count=V13_INDEPENDENT_HEAD_PARAMETER_COUNT,
        capacity_matched_to_k31=False,
    )
    if not isinstance(loaded, LoadedMatchedIndependentControlV13):
        raise RuntimeError("v13 independent loader returned the wrong artifact type")
    return loaded


def load_matched_independent_control_v13(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> LoadedMatchedIndependentControlV13:
    loaded = _load_control_v13(
        path,
        expected_manifest_sha256=expected_manifest_sha256,
        schema=V13_MATCHED_CONTROL_SCHEMA,
        candidate=V13_MATCHED_CONTROL_CANDIDATE,
        control_role=V13_MATCHED_CONTROL_ROLE,
        exact_head_type=IctalInvolvementHead,
        loaded_type=LoadedMatchedIndependentControlV13,
        head_class_name="IctalInvolvementHead",
        temporal_mixing="none_independent_seconds_on_cached_tokens",
        head_config={"token_dim": 200, "hidden_dim": 128},
        expected_total_parameter_count=V13_INDEPENDENT_HEAD_PARAMETER_COUNT,
        capacity_matched_to_k31=False,
    )
    if not isinstance(loaded, LoadedMatchedIndependentControlV13):
        raise RuntimeError("v13 independent loader returned the wrong artifact type")
    return loaded


def save_capacity_matched_channel_control_v13(
    output_directory: str | Path,
    *,
    lineage: MatchedControlLineage,
    head: CapacityMatchedChannelResidualIctalInvolvementHead,
    training_config: Mapping[str, object],
    execution_receipt: Mapping[str, object],
    training_run: Mapping[str, object],
) -> LoadedCapacityMatchedChannelControlV13:
    """Seal the primary capacity-matched, no-time-mixing comparator."""

    config = {
        "token_dim": 200,
        "hidden_dim": 128,
        "groups": 4,
        "kernel_size": 1,
        "bias": False,
        "residual": True,
        "activation": "gelu",
        "normalization": "layernorm_128",
    }
    loaded = _save_control_v13(
        output_directory,
        lineage=lineage,
        head=head,
        training_config=training_config,
        execution_receipt=execution_receipt,
        training_run=training_run,
        schema=V13_CAPACITY_MATCHED_CONTROL_SCHEMA,
        candidate=V13_CAPACITY_MATCHED_CONTROL_CANDIDATE,
        control_role=V13_CAPACITY_MATCHED_CONTROL_ROLE,
        exact_head_type=CapacityMatchedChannelResidualIctalInvolvementHead,
        head_class_name="CapacityMatchedChannelResidualIctalInvolvementHead",
        temporal_mixing=(
            "none_grouped_hidden_channel_k1_per_edge_second_no_cross_second"
        ),
        head_config=config,
        expected_total_parameter_count=V13_CAPACITY_MATCHED_HEAD_PARAMETER_COUNT,
        capacity_matched_to_k31=True,
    )
    if not isinstance(loaded, LoadedCapacityMatchedChannelControlV13):
        raise RuntimeError("v13 capacity-matched loader returned the wrong artifact type")
    return loaded


def load_capacity_matched_channel_control_v13(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> LoadedCapacityMatchedChannelControlV13:
    config = {
        "token_dim": 200,
        "hidden_dim": 128,
        "groups": 4,
        "kernel_size": 1,
        "bias": False,
        "residual": True,
        "activation": "gelu",
        "normalization": "layernorm_128",
    }
    loaded = _load_control_v13(
        path,
        expected_manifest_sha256=expected_manifest_sha256,
        schema=V13_CAPACITY_MATCHED_CONTROL_SCHEMA,
        candidate=V13_CAPACITY_MATCHED_CONTROL_CANDIDATE,
        control_role=V13_CAPACITY_MATCHED_CONTROL_ROLE,
        exact_head_type=CapacityMatchedChannelResidualIctalInvolvementHead,
        loaded_type=LoadedCapacityMatchedChannelControlV13,
        head_class_name="CapacityMatchedChannelResidualIctalInvolvementHead",
        temporal_mixing=(
            "none_grouped_hidden_channel_k1_per_edge_second_no_cross_second"
        ),
        head_config=config,
        expected_total_parameter_count=V13_CAPACITY_MATCHED_HEAD_PARAMETER_COUNT,
        capacity_matched_to_k31=True,
    )
    if not isinstance(loaded, LoadedCapacityMatchedChannelControlV13):
        raise RuntimeError("v13 capacity-matched loader returned the wrong artifact type")
    return loaded


__all__ = (
    "LoadedCapacityMatchedChannelControlV13",
    "LoadedMatchedIndependentControlV13",
    "MatchedControlLineage",
    "V13_CHECKPOINT_FILENAME",
    "V13_CAPACITY_MATCHED_CONTROL_CANDIDATE",
    "V13_CAPACITY_MATCHED_CONTROL_ROLE",
    "V13_CAPACITY_MATCHED_CONTROL_SCHEMA",
    "V13_CAPACITY_MATCHED_HEAD_PARAMETER_COUNT",
    "V13_INDEPENDENT_HEAD_PARAMETER_COUNT",
    "V13_K31_EXTRA_PARAMETER_COUNT",
    "V13_MANIFEST_FILENAME",
    "V13_MATCHED_CONTROL_CANDIDATE",
    "V13_MATCHED_CONTROL_ROLE",
    "V13_MATCHED_CONTROL_SCHEMA",
    "build_fit_only_token_bag_dataset_v13",
    "load_capacity_matched_channel_control_v13",
    "load_matched_independent_control_v13",
    "save_capacity_matched_channel_control_v13",
    "save_matched_independent_control_v13",
    "validate_matched_control_lineage",
)
