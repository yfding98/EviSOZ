"""Strict artifacts for the post-formal-v5 LaBraM k31 OOF recovery.

The formal-v4 ictal checkpoint schema intentionally describes only the
independent-second head.  A long-context recovery model must therefore not be
saved under that schema or silently passed off as a promoted formal producer.
This module gives the single prespecified k31 candidate its own closed,
development-only artifact format while retaining the patient/fold lineage
needed for later out-of-fold evidence generation.
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
from safetensors.torch import load_file, save_file

from .concept_metrics import IctalConceptMetrics
from .concept_run import IctalTrainingConfig, ictal_head_state_sha256
from .concept_training import IctalEpochOutput
from .models.concept_heads import LongContextTemporalResidualIctalInvolvementHead


LABRAM_K31_OOF_RUN_SCHEMA = "soz_labram_k31_ictal_oof_recovery_run_v1_1"
LABRAM_K31_OOF_MANIFEST_FILENAME = "recovery_run.json"
LABRAM_K31_OOF_CHECKPOINT_FILENAME = "model.safetensors"
LABRAM_K31_CANDIDATE = "labram_temporal_residual_k31"
LABRAM_K31_CONTEXT_SECONDS = 31
LABRAM_K31_TARGET_SEMANTICS = "tusz_bipolar_edge_time_involvement_not_soz"

_SELECTION_RE = re.compile(r"fold([0-4])|final")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_TRAINING_RUN_FIELDS = frozenset(
    {
        "initial_state_sha256",
        "final_state_sha256",
        "epoch_rows",
        "evaluation_epoch",
        "metrics",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "candidate",
        "selection",
        "oof_fold",
        "context_seconds",
        "context_direction",
        "target_semantics",
        "development_only",
        "architecture_selected_after_opened_i_dev",
        "formal_promotion",
        "checkpoint_authorized_for_formal_evidence_or_reasoner",
        "deepsoz_soz_labels_used",
        "private_labels_used",
        "missing_tusz_cells_imputed_as_negative",
        "i_gate_outcomes_opened",
        "split_manifest_sha256",
        "oof_protocol_artifact_sha256",
        "oof_protocol_receipt_sha256",
        "oof_plan_receipt_sha256",
        "training_manifest_sha256",
        "training_corpus_index_sha256",
        "target_snapshot_manifest_sha256",
        "target_snapshot_receipt_sha256",
        "native_evaluation_manifest_sha256",
        "native_evaluation_corpus_index_sha256",
        "training_public_patient_ids",
        "training_public_roster_sha256",
        "held_out_exclusion_public_patient_ids",
        "held_out_exclusion_public_roster_sha256",
        "native_evaluation_public_patient_ids",
        "native_evaluation_public_roster_sha256",
        "i_gate_patient_ids_excluded_unopened",
        "i_gate_patient_roster_sha256",
        "training_config",
        "training_run",
        "head_config",
        "head_state_sha256",
        "checkpoint_filename",
        "checkpoint_sha256",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return text


def patient_roster_sha256(values: Sequence[object]) -> str:
    roster = _patient_roster(values, field="patient_roster", allow_empty=False)
    return _sha256_bytes(_canonical_json_bytes(roster))


def _patient_roster(
    values: Sequence[object], *, field: str, allow_empty: bool
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a patient sequence")
    roster = tuple(str(value).strip() for value in values)
    if (not allow_empty and not roster) or any(not value for value in roster):
        raise ValueError(f"{field} must contain non-empty patient IDs")
    if roster != tuple(sorted(roster)) or len(set(roster)) != len(roster):
        raise ValueError(f"{field} must be unique and sorted")
    return roster


def _selection(value: object) -> tuple[str, int | None]:
    text = str(value).strip().lower()
    match = _SELECTION_RE.fullmatch(text)
    if match is None:
        raise ValueError("selection must be fold0..fold4 or final")
    return text, None if text == "final" else int(match.group(1))


def _safe_new_output(value: str | Path) -> Path:
    target = Path(os.path.abspath(value))
    if target.name in {"", ".", ".."} or not target.parent.is_dir():
        raise ValueError("Recovery output requires a concrete path with an existing parent")
    if os.path.lexists(target):
        raise FileExistsError(f"Recovery output already exists: {target}")
    return target


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_epoch_payload(
    value: object, *, field: str, expected_patient_count: int
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an epoch mapping")
    payload = dict(value)
    expected_fields = {
        "mean_patient_loss",
        "n_patients",
        "n_events",
        "n_observed_labels",
    }
    if set(payload) != expected_fields:
        raise ValueError(f"{field} violates the closed epoch schema")
    try:
        epoch = IctalEpochOutput(**payload)
    except TypeError as exc:
        raise ValueError(f"{field} is not a valid ictal epoch receipt") from exc
    if (
        isinstance(epoch.mean_patient_loss, bool)
        or not isinstance(epoch.mean_patient_loss, (int, float))
        or not math.isfinite(float(epoch.mean_patient_loss))
        or float(epoch.mean_patient_loss) < 0.0
    ):
        raise ValueError(f"{field}.mean_patient_loss must be finite and non-negative")
    for name in ("n_patients", "n_events", "n_observed_labels"):
        observed = getattr(epoch, name)
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 1:
            raise ValueError(f"{field}.{name} must be a positive integer")
    if epoch.n_patients != expected_patient_count:
        raise ValueError(f"{field}.n_patients disagrees with its patient roster")
    return asdict(epoch)


def _validated_training_metadata(
    training_config: object,
    training_run: object,
    *,
    head_state_sha256: str,
    training_patient_count: int,
    native_patient_count: int,
) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(training_config, Mapping):
        raise TypeError("training_config must be a mapping")
    config = dict(training_config)
    expected_config = asdict(IctalTrainingConfig())
    if config != expected_config:
        raise ValueError("Recovery training config changed from the frozen policy")
    if not isinstance(training_run, Mapping):
        raise TypeError("training_run must be a mapping")
    run = dict(training_run)
    if set(run) != _TRAINING_RUN_FIELDS:
        raise ValueError("training_run violates the closed recovery-run schema")
    initial_state = _require_sha256(
        run["initial_state_sha256"], field="training_run.initial_state_sha256"
    )
    final_state = _require_sha256(
        run["final_state_sha256"], field="training_run.final_state_sha256"
    )
    if final_state != _require_sha256(
        head_state_sha256, field="head_state_sha256"
    ):
        raise ValueError("training_run final state does not match the recovery head")
    if initial_state == final_state:
        raise ValueError("training_run shows no optimizer-induced state change")
    epoch_rows = run["epoch_rows"]
    if not isinstance(epoch_rows, list) or len(epoch_rows) != IctalTrainingConfig().fixed_epochs:
        raise ValueError("training_run must contain every frozen training epoch")
    normalized_epochs = [
        _validated_epoch_payload(
            row,
            field=f"training_run.epoch_rows[{index}]",
            expected_patient_count=training_patient_count,
        )
        for index, row in enumerate(epoch_rows)
    ]
    evaluation_epoch = _validated_epoch_payload(
        run["evaluation_epoch"],
        field="training_run.evaluation_epoch",
        expected_patient_count=native_patient_count,
    )
    if not isinstance(run["metrics"], Mapping):
        raise TypeError("training_run.metrics must be a mapping")
    try:
        metrics = IctalConceptMetrics(**dict(run["metrics"]))
    except TypeError as exc:
        raise ValueError("training_run.metrics violates the closed metric schema") from exc
    if metrics.n_patients != native_patient_count:
        raise ValueError("training_run metrics disagree with the native patient roster")
    if metrics.n_observed_labels != evaluation_epoch["n_observed_labels"]:
        raise ValueError("training_run evaluation label counts disagree")
    if (
        abs(
            float(metrics.patient_macro_bce)
            - float(evaluation_epoch["mean_patient_loss"])
        )
        > 1e-6
    ):
        raise ValueError("training_run native BCE disagrees with evaluation loss")
    return config, {
        "initial_state_sha256": initial_state,
        "final_state_sha256": final_state,
        "epoch_rows": normalized_epochs,
        "evaluation_epoch": evaluation_epoch,
        "metrics": asdict(metrics),
    }


@dataclass(frozen=True)
class LoadedLaBraMK31OOFRecoveryRun:
    path: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    head: LongContextTemporalResidualIctalInvolvementHead


def save_labram_k31_oof_recovery_run(
    output_directory: str | Path,
    *,
    selection: str,
    head: LongContextTemporalResidualIctalInvolvementHead,
    split_manifest_sha256: str,
    oof_protocol_artifact_sha256: str,
    oof_protocol_receipt_sha256: str,
    oof_plan_receipt_sha256: str,
    training_manifest_sha256: str,
    training_corpus_index_sha256: str,
    target_snapshot_manifest_sha256: str,
    target_snapshot_receipt_sha256: str,
    native_evaluation_manifest_sha256: str,
    native_evaluation_corpus_index_sha256: str,
    training_public_patient_ids: Sequence[object],
    held_out_exclusion_public_patient_ids: Sequence[object],
    native_evaluation_public_patient_ids: Sequence[object],
    i_gate_patient_ids_excluded_unopened: Sequence[object],
    training_config: Mapping[str, object],
    training_run: Mapping[str, object],
) -> LoadedLaBraMK31OOFRecoveryRun:
    """Atomically save one non-promoted, fold-bound k31 recovery head."""

    if type(head) is not LongContextTemporalResidualIctalInvolvementHead:
        raise TypeError("Recovery artifact requires the exact prespecified k31 head")
    normalized_selection, oof_fold = _selection(selection)
    training = _patient_roster(
        training_public_patient_ids,
        field="training_public_patient_ids",
        allow_empty=False,
    )
    held_out = _patient_roster(
        held_out_exclusion_public_patient_ids,
        field="held_out_exclusion_public_patient_ids",
        allow_empty=False,
    )
    native = _patient_roster(
        native_evaluation_public_patient_ids,
        field="native_evaluation_public_patient_ids",
        allow_empty=False,
    )
    gate = _patient_roster(
        i_gate_patient_ids_excluded_unopened,
        field="i_gate_patient_ids_excluded_unopened",
        allow_empty=False,
    )
    if len(gate) != 12:
        raise ValueError("Recovery must preserve the exact 12-patient unopened I-gate")
    if set(training) & set(held_out):
        raise ValueError("OOF held-out patients leaked into recovery fitting")
    if set(training) & set(native):
        raise ValueError("Native-evaluation patients leaked into recovery fitting")
    if set(training) & set(gate):
        raise ValueError("I-gate patients leaked into recovery fitting")
    if set(native) & set(gate):
        raise ValueError("I-gate outcomes were included in native evaluation")

    first_layer = head.adapter[0]
    if not isinstance(first_layer, torch.nn.Linear):
        raise TypeError("Recovery head adapter changed")
    token_dim = int(head.edge_tokens.token_dim)
    hidden_dim = int(first_layer.out_features)
    if token_dim != 200 or hidden_dim != 128 or head.context_seconds != 31:
        raise ValueError("Recovery head differs from the frozen k31 architecture")
    head_state_sha = ictal_head_state_sha256(head)
    normalized_config, normalized_run = _validated_training_metadata(
        training_config,
        training_run,
        head_state_sha256=head_state_sha,
        training_patient_count=len(training),
        native_patient_count=len(native),
    )

    target = _safe_new_output(output_directory)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        checkpoint_path = staging / LABRAM_K31_OOF_CHECKPOINT_FILENAME
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in head.state_dict().items()
        }
        save_file(state, str(checkpoint_path))
        checkpoint_sha256 = _file_sha256(checkpoint_path)
        payload = {
            "schema_version": LABRAM_K31_OOF_RUN_SCHEMA,
            "candidate": LABRAM_K31_CANDIDATE,
            "selection": normalized_selection,
            "oof_fold": oof_fold,
            "context_seconds": LABRAM_K31_CONTEXT_SECONDS,
            "context_direction": "symmetric_retrospective_not_causal_onset",
            "target_semantics": LABRAM_K31_TARGET_SEMANTICS,
            "development_only": True,
            "architecture_selected_after_opened_i_dev": True,
            "formal_promotion": False,
            "checkpoint_authorized_for_formal_evidence_or_reasoner": False,
            "deepsoz_soz_labels_used": False,
            "private_labels_used": False,
            "missing_tusz_cells_imputed_as_negative": False,
            "i_gate_outcomes_opened": False,
            "split_manifest_sha256": _require_sha256(
                split_manifest_sha256, field="split_manifest_sha256"
            ),
            "oof_protocol_artifact_sha256": _require_sha256(
                oof_protocol_artifact_sha256,
                field="oof_protocol_artifact_sha256",
            ),
            "oof_protocol_receipt_sha256": _require_sha256(
                oof_protocol_receipt_sha256,
                field="oof_protocol_receipt_sha256",
            ),
            "oof_plan_receipt_sha256": _require_sha256(
                oof_plan_receipt_sha256, field="oof_plan_receipt_sha256"
            ),
            "training_manifest_sha256": _require_sha256(
                training_manifest_sha256, field="training_manifest_sha256"
            ),
            "training_corpus_index_sha256": _require_sha256(
                training_corpus_index_sha256,
                field="training_corpus_index_sha256",
            ),
            "target_snapshot_manifest_sha256": _require_sha256(
                target_snapshot_manifest_sha256,
                field="target_snapshot_manifest_sha256",
            ),
            "target_snapshot_receipt_sha256": _require_sha256(
                target_snapshot_receipt_sha256,
                field="target_snapshot_receipt_sha256",
            ),
            "native_evaluation_manifest_sha256": _require_sha256(
                native_evaluation_manifest_sha256,
                field="native_evaluation_manifest_sha256",
            ),
            "native_evaluation_corpus_index_sha256": _require_sha256(
                native_evaluation_corpus_index_sha256,
                field="native_evaluation_corpus_index_sha256",
            ),
            "training_public_patient_ids": list(training),
            "training_public_roster_sha256": patient_roster_sha256(training),
            "held_out_exclusion_public_patient_ids": list(held_out),
            "held_out_exclusion_public_roster_sha256": patient_roster_sha256(held_out),
            "native_evaluation_public_patient_ids": list(native),
            "native_evaluation_public_roster_sha256": patient_roster_sha256(native),
            "i_gate_patient_ids_excluded_unopened": list(gate),
            "i_gate_patient_roster_sha256": patient_roster_sha256(gate),
            "training_config": normalized_config,
            "training_run": normalized_run,
            "head_config": {"token_dim": token_dim, "hidden_dim": hidden_dim},
            "head_state_sha256": head_state_sha,
            "checkpoint_filename": LABRAM_K31_OOF_CHECKPOINT_FILENAME,
            "checkpoint_sha256": checkpoint_sha256,
        }
        raw = _canonical_json_bytes(payload)
        if not 1 <= len(raw) <= _MAX_MANIFEST_BYTES:
            raise ValueError("Recovery manifest has an invalid size")
        (staging / LABRAM_K31_OOF_MANIFEST_FILENAME).write_bytes(raw)
        os.rename(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return load_labram_k31_oof_recovery_run(
        target,
        expected_manifest_sha256=_sha256_bytes(raw),
    )


def load_labram_k31_oof_recovery_run(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> LoadedLaBraMK31OOFRecoveryRun:
    """Strictly load and replay one k31 recovery bundle."""

    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_dir() or source.resolve() != source:
        raise ValueError("Recovery bundle must be a regular absolute directory")
    if {item.name for item in source.iterdir()} != {
        LABRAM_K31_OOF_MANIFEST_FILENAME,
        LABRAM_K31_OOF_CHECKPOINT_FILENAME,
    }:
        raise ValueError("Recovery bundle has missing or unknown files")
    raw = (source / LABRAM_K31_OOF_MANIFEST_FILENAME).read_bytes()
    if not 1 <= len(raw) <= _MAX_MANIFEST_BYTES:
        raise ValueError("Recovery manifest has an invalid size")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Recovery manifest is not valid UTF-8 JSON") from exc
    if (
        not isinstance(manifest, dict)
        or set(manifest) != _MANIFEST_FIELDS
        or _canonical_json_bytes(manifest) != raw
    ):
        raise ValueError("Recovery manifest violates its closed canonical schema")
    actual_manifest_sha = _sha256_bytes(raw)
    if expected_manifest_sha256 is not None and actual_manifest_sha != _require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise ValueError("Recovery manifest SHA mismatch")
    if manifest.get("schema_version") != LABRAM_K31_OOF_RUN_SCHEMA:
        raise ValueError("Unsupported recovery schema")
    selection, oof_fold = _selection(manifest.get("selection"))
    if manifest.get("oof_fold") != oof_fold:
        raise ValueError("Recovery selection and OOF fold disagree")
    fixed = {
        "candidate": LABRAM_K31_CANDIDATE,
        "context_seconds": LABRAM_K31_CONTEXT_SECONDS,
        "context_direction": "symmetric_retrospective_not_causal_onset",
        "target_semantics": LABRAM_K31_TARGET_SEMANTICS,
        "development_only": True,
        "architecture_selected_after_opened_i_dev": True,
        "formal_promotion": False,
        "checkpoint_authorized_for_formal_evidence_or_reasoner": False,
        "deepsoz_soz_labels_used": False,
        "private_labels_used": False,
        "missing_tusz_cells_imputed_as_negative": False,
        "i_gate_outcomes_opened": False,
        "checkpoint_filename": LABRAM_K31_OOF_CHECKPOINT_FILENAME,
    }
    if any(manifest.get(key) != value for key, value in fixed.items()):
        raise ValueError("Recovery manifest changed a frozen scientific boundary")
    for field in (
        "split_manifest_sha256",
        "oof_protocol_artifact_sha256",
        "oof_protocol_receipt_sha256",
        "oof_plan_receipt_sha256",
        "training_manifest_sha256",
        "training_corpus_index_sha256",
        "target_snapshot_manifest_sha256",
        "target_snapshot_receipt_sha256",
        "native_evaluation_manifest_sha256",
        "native_evaluation_corpus_index_sha256",
        "training_public_roster_sha256",
        "held_out_exclusion_public_roster_sha256",
        "native_evaluation_public_roster_sha256",
        "i_gate_patient_roster_sha256",
        "head_state_sha256",
        "checkpoint_sha256",
    ):
        _require_sha256(manifest.get(field), field=field)
    rosters = {
        "training_public_patient_ids": _patient_roster(
            manifest["training_public_patient_ids"],
            field="training_public_patient_ids",
            allow_empty=False,
        ),
        "held_out_exclusion_public_patient_ids": _patient_roster(
            manifest["held_out_exclusion_public_patient_ids"],
            field="held_out_exclusion_public_patient_ids",
            allow_empty=False,
        ),
        "native_evaluation_public_patient_ids": _patient_roster(
            manifest["native_evaluation_public_patient_ids"],
            field="native_evaluation_public_patient_ids",
            allow_empty=False,
        ),
        "i_gate_patient_ids_excluded_unopened": _patient_roster(
            manifest["i_gate_patient_ids_excluded_unopened"],
            field="i_gate_patient_ids_excluded_unopened",
            allow_empty=False,
        ),
    }
    receipt_fields = {
        "training_public_patient_ids": "training_public_roster_sha256",
        "held_out_exclusion_public_patient_ids": (
            "held_out_exclusion_public_roster_sha256"
        ),
        "native_evaluation_public_patient_ids": (
            "native_evaluation_public_roster_sha256"
        ),
        "i_gate_patient_ids_excluded_unopened": "i_gate_patient_roster_sha256",
    }
    for roster_field, receipt_field in receipt_fields.items():
        if patient_roster_sha256(rosters[roster_field]) != manifest[receipt_field]:
            raise ValueError(f"{roster_field} receipt mismatch")
    training = set(rosters["training_public_patient_ids"])
    if training & set(rosters["held_out_exclusion_public_patient_ids"]):
        raise ValueError("Recovery reload found OOF training leakage")
    if training & set(rosters["native_evaluation_public_patient_ids"]):
        raise ValueError("Recovery reload found native-evaluation training leakage")
    gate = set(rosters["i_gate_patient_ids_excluded_unopened"])
    if len(gate) != 12 or training & gate:
        raise ValueError("Recovery reload found I-gate training leakage")
    if gate & set(rosters["native_evaluation_public_patient_ids"]):
        raise ValueError("Recovery reload found opened I-gate outcomes")

    _validated_training_metadata(
        manifest["training_config"],
        manifest["training_run"],
        head_state_sha256=str(manifest["head_state_sha256"]),
        training_patient_count=len(rosters["training_public_patient_ids"]),
        native_patient_count=len(rosters["native_evaluation_public_patient_ids"]),
    )

    head_config = manifest.get("head_config")
    if head_config != {"token_dim": 200, "hidden_dim": 128}:
        raise ValueError("Recovery head configuration changed")
    checkpoint_path = source / LABRAM_K31_OOF_CHECKPOINT_FILENAME
    if _file_sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise ValueError("Recovery checkpoint SHA mismatch")
    state = load_file(str(checkpoint_path), device="cpu")
    head = LongContextTemporalResidualIctalInvolvementHead(
        token_dim=200, hidden_dim=128
    )
    expected_state = head.state_dict()
    if set(state) != set(expected_state):
        raise ValueError("Recovery checkpoint tensor names changed")
    for name, expected in expected_state.items():
        observed = state[name]
        if observed.shape != expected.shape or observed.dtype != expected.dtype:
            raise ValueError(f"Recovery checkpoint tensor changed: {name}")
        if observed.is_floating_point() and not torch.isfinite(observed).all():
            raise ValueError(f"Recovery checkpoint contains non-finite values: {name}")
    head.load_state_dict(state, strict=True)
    if ictal_head_state_sha256(head) != manifest["head_state_sha256"]:
        raise ValueError("Recovery head-state receipt mismatch")
    head.eval()
    return LoadedLaBraMK31OOFRecoveryRun(
        path=source,
        manifest=manifest,
        manifest_sha256=actual_manifest_sha,
        head=head,
    )
