"""Formal fold-bound production training for the TUSZ ictal concept.

This module is intentionally narrow: it trains only the lightweight
``IctalInvolvementHead`` from a verified formal-v3 LaBraM token corpus.  SOZ
labels are never accepted as a training or native-evaluation argument.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Mapping, Sequence

import torch

from .cached_concept_training import evaluate_cached_ictal_patients
from .concept_checkpoint import (
    IctalConceptCheckpointArtifact,
    LoadedIctalConceptCheckpoint,
    load_ictal_concept_checkpoint,
    save_ictal_concept_checkpoint,
)
from .concept_metrics import IctalConceptMetrics
from .concept_oof import (
    IctalConceptOOFPlan,
    IctalConceptOOFProtocolArtifact,
)
from .concept_run import (
    ICTAL_IDENTITY_SCALER_SHA256,
    IctalDeterminismPolicyReceipt,
    IctalTrainingConfig,
    ictal_determinism_runtime,
    ictal_head_state_sha256,
    load_ictal_training_run_receipt,
    save_ictal_training_run_receipt,
    train_fixed_epoch_ictal_head,
    validate_ictal_cuda_environment,
)
from .concept_token_io import load_labram_concept_tokens
from .data.tusz_training import TUSZIctalTrainingManifest
from .formal_token_corpus import VerifiedFormalTokenCorpusArtifact
from .ictal_native_eval import (
    VerifiedIctalNativeEvalManifestArtifact,
    VerifiedIctalNativeEvalTokenCorpusArtifact,
    build_ictal_native_eval_token_bag_dataset,
)
from .ictal_gate_policy import VerifiedIctalPromotionGatePolicyArtifact
from .models.concept_heads import IctalInvolvementHead
from .tusz_token_dataset import build_tusz_ictal_token_bag_dataset


ICTAL_PRODUCTION_RUN_SCHEMA = "soz_ictal_production_run_v3"
ICTAL_PRODUCTION_RUN_FILENAME = "production_run.json"
ICTAL_PRODUCTION_CHECKPOINT_DIRECTORY = "checkpoint"
ICTAL_NATIVE_TARGET_SEMANTICS = "tusz_bipolar_edge_time_involvement_not_soz"
ICTAL_FIXED_HEAD_HIDDEN_DIM = 128
ICTAL_PRODUCTION_CONFIG = IctalTrainingConfig()
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SELECTION_RE = re.compile(r"fold([0-4])")
_MAX_PRODUCTION_MANIFEST_BYTES = 4 * 1024 * 1024
_PRODUCTION_FIELDS = frozenset(
    {
        "schema_version",
        "selection",
        "oof_fold",
        "split_manifest_sha256",
        "oof_protocol_artifact_sha256",
        "oof_protocol_receipt_sha256",
        "oof_plan_receipt_sha256",
        "promotion_gate_policy_artifact_sha256",
        "promotion_gate_policy_bundle_receipt_sha256",
        "promotion_gate_policy_receipt_sha256",
        "promotion_gate_policy_document_sha256",
        "training_config",
        "training_config_sha256",
        "determinism_policy",
        "determinism_policy_sha256",
        "training_manifest_sha256",
        "training_corpus_index_sha256",
        "training_run_receipt_sha256",
        "training_source_public_patient_ids",
        "training_source_public_roster_sha256",
        "held_out_exclusion_public_patient_ids",
        "held_out_exclusion_public_roster_sha256",
        "native_evaluation_role",
        "native_evaluation_manifest_sha256",
        "native_evaluation_corpus_index_sha256",
        "native_evaluation_public_patient_ids",
        "native_evaluation_public_roster_sha256",
        "native_unevaluable_public_patient_ids",
        "native_unevaluable_public_roster_sha256",
        "native_unevaluable_omission_rows",
        "native_unevaluable_omission_reason_counts",
        "native_unevaluable_omission_roster_sha256",
        "native_metrics",
        "checkpoint_directory",
        "checkpoint_manifest_sha256",
        "checkpoint_sha256",
    }
)
_NATIVE_UNEVALUABLE_OMISSION_ROW_FIELDS = frozenset(
    {
        "patient_id",
        "relative_edf_path",
        "public_record_sha256",
        "reasons",
        "event_id",
    }
)
_NATIVE_UNEVALUABLE_REASON_COUNT_FIELDS = frozenset(
    {"patient_id", "reason_counts"}
)
_NATIVE_UNEVALUABLE_REASON_FIELDS = frozenset({"reason", "count"})
_NATIVE_UNEVALUABLE_OMISSION_SCHEMA = (
    "soz_ictal_native_unevaluable_omission_roster_v1"
)
_DETERMINISM_POLICY_FIELDS = frozenset(
    field.name for field in fields(IctalDeterminismPolicyReceipt)
)
_NATIVE_METRIC_FIELDS = frozenset(
    {
        "target_semantics",
        "deepsoz_soz_labels_used",
        "missing_tusz_bins_imputed_as_negative",
        "mean_patient_loss",
        "n_events",
        *IctalConceptMetrics.__dataclass_fields__,
    }
)


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Ictal production receipt is not canonical JSON data") from exc
    return (encoded + "\n").encode("utf-8")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
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


def _public_roster_sha256(values: Sequence[str]) -> str:
    roster = tuple(sorted(str(value).strip() for value in values))
    if not roster or any(not value for value in roster) or len(set(roster)) != len(roster):
        raise ValueError("Public patient roster must be non-empty, unique, and trimmed")
    return _canonical_sha256(roster)


def _attrition_public_roster_sha256(values: Sequence[str]) -> str:
    """Hash a canonical attrition roster, including the valid empty roster."""

    roster = tuple(sorted(str(value).strip() for value in values))
    if any(not value for value in roster) or len(set(roster)) != len(roster):
        raise ValueError("Attrition patient roster must be unique and trimmed")
    return _canonical_sha256(roster)


def _canonical_public_roster(
    values: Sequence[str], *, field: str, allow_empty: bool
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a patient sequence")
    roster = tuple(str(value) for value in values)
    if (not allow_empty and not roster) or any(
        not value or value != value.strip() for value in roster
    ):
        qualifier = "possibly empty" if allow_empty else "non-empty"
        raise ValueError(f"{field} must be a {qualifier}, trimmed patient roster")
    if tuple(sorted(roster)) != roster or len(set(roster)) != len(roster):
        raise ValueError(f"{field} must be unique and canonically sorted")
    return roster


def _native_unevaluable_omission_roster_sha256(
    rows: Sequence[tuple[str, str, str, tuple[str, ...], str | None]],
) -> str:
    return _canonical_sha256(
        {
            "schema_version": _NATIVE_UNEVALUABLE_OMISSION_SCHEMA,
            "omissions": tuple(rows),
        }
    )


def _native_unevaluable_reason_counts(
    rows: Sequence[tuple[str, str, str, tuple[str, ...], str | None]],
) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
    by_patient: dict[str, dict[str, int]] = {}
    for patient_id, _, _, reasons, _ in rows:
        patient_counts = by_patient.setdefault(patient_id, {})
        for reason in reasons:
            patient_counts[reason] = patient_counts.get(reason, 0) + 1
    return tuple(
        (
            patient_id,
            tuple(sorted(reason_counts.items())),
        )
        for patient_id, reason_counts in sorted(by_patient.items())
    )


def _native_unevaluable_omission_payload(
    rows: Sequence[tuple[str, str, str, tuple[str, ...], str | None]],
) -> list[dict[str, object]]:
    return [
        {
            "patient_id": patient_id,
            "relative_edf_path": relative_edf_path,
            "public_record_sha256": public_record_sha256,
            "reasons": list(reasons),
            "event_id": event_id,
        }
        for patient_id, relative_edf_path, public_record_sha256, reasons, event_id in rows
    ]


def _native_unevaluable_reason_count_payload(
    counts: Sequence[tuple[str, tuple[tuple[str, int], ...]]],
) -> list[dict[str, object]]:
    return [
        {
            "patient_id": patient_id,
            "reason_counts": [
                {"reason": reason, "count": count}
                for reason, count in reason_counts
            ],
        }
        for patient_id, reason_counts in counts
    ]


def _parse_native_unevaluable_omission_payload(
    value: object,
) -> tuple[tuple[str, str, str, tuple[str, ...], str | None], ...]:
    if not isinstance(value, list):
        raise TypeError("native_unevaluable_omission_rows must be a JSON array")
    rows: list[tuple[str, str, str, tuple[str, ...], str | None]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != (
            _NATIVE_UNEVALUABLE_OMISSION_ROW_FIELDS
        ):
            raise ValueError(
                "Native-unevaluable omission row violates its closed schema"
            )
        patient_id = item.get("patient_id")
        relative_edf_path = item.get("relative_edf_path")
        if (
            not isinstance(patient_id, str)
            or not patient_id
            or patient_id != patient_id.strip()
        ):
            raise ValueError(f"Omission row {index} has an invalid patient_id")
        if (
            not isinstance(relative_edf_path, str)
            or not relative_edf_path
            or relative_edf_path != relative_edf_path.strip()
        ):
            raise ValueError(f"Omission row {index} has an invalid EDF path")
        relative = PurePosixPath(relative_edf_path)
        if (
            relative.is_absolute()
            or len(relative.parts) != 5
            or relative.parts[0] != "train"
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError(f"Omission row {index} has a non-canonical EDF path")
        public_record_sha256 = _require_sha256(
            item.get("public_record_sha256"),
            field=f"native_unevaluable_omission_rows[{index}].public_record_sha256",
        )
        reasons_value = item.get("reasons")
        if not isinstance(reasons_value, list) or any(
            not isinstance(reason, str)
            or not reason
            or reason != reason.strip()
            for reason in reasons_value
        ):
            raise TypeError(f"Omission row {index} reasons must be strings")
        reasons = tuple(reasons_value)
        if not reasons or tuple(sorted(set(reasons))) != reasons:
            raise ValueError(
                f"Omission row {index} reasons must be non-empty, unique, and sorted"
            )
        event_id = item.get("event_id")
        if event_id is not None and (
            not isinstance(event_id, str)
            or not event_id
            or event_id != event_id.strip()
        ):
            raise ValueError(f"Omission row {index} has an invalid event_id")
        rows.append(
            (
                patient_id,
                relative_edf_path,
                public_record_sha256,
                reasons,
                event_id,
            )
        )
    result = tuple(rows)
    canonical = tuple(
        sorted(
            result,
            key=lambda row: (
                row[0], row[1], "" if row[4] is None else row[4], row[3]
            ),
        )
    )
    if result != canonical or len(set(result)) != len(result):
        raise ValueError("Native-unevaluable omission rows must be unique and sorted")
    return result


def _parse_native_unevaluable_reason_count_payload(
    value: object,
) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
    if not isinstance(value, list):
        raise TypeError(
            "native_unevaluable_omission_reason_counts must be a JSON array"
        )
    rows: list[tuple[str, tuple[tuple[str, int], ...]]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != (
            _NATIVE_UNEVALUABLE_REASON_COUNT_FIELDS
        ):
            raise ValueError("Native attrition reason-count row violates its schema")
        patient_id = item.get("patient_id")
        if (
            not isinstance(patient_id, str)
            or not patient_id
            or patient_id != patient_id.strip()
        ):
            raise ValueError(f"Reason-count row {index} has an invalid patient_id")
        counts_value = item.get("reason_counts")
        if not isinstance(counts_value, list) or not counts_value:
            raise ValueError(f"Reason-count row {index} must contain reasons")
        reason_counts: list[tuple[str, int]] = []
        for count_index, count_item in enumerate(counts_value):
            if not isinstance(count_item, dict) or set(count_item) != (
                _NATIVE_UNEVALUABLE_REASON_FIELDS
            ):
                raise ValueError("Native attrition reason entry violates its schema")
            reason = count_item.get("reason")
            count = count_item.get("count")
            if (
                not isinstance(reason, str)
                or not reason
                or reason != reason.strip()
            ):
                raise ValueError(
                    f"Reason-count row {index}:{count_index} has an invalid reason"
                )
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(
                    f"Reason-count row {index}:{count_index} has an invalid count"
                )
            reason_counts.append((reason, count))
        canonical_counts = tuple(sorted(reason_counts))
        if tuple(reason_counts) != canonical_counts or len(
            {reason for reason, _ in reason_counts}
        ) != len(reason_counts):
            raise ValueError("Native attrition reasons must be unique and sorted")
        rows.append((patient_id, canonical_counts))
    result = tuple(rows)
    if result != tuple(sorted(result)) or len({row[0] for row in result}) != len(
        result
    ):
        raise ValueError("Native attrition reason counts must be unique and sorted")
    return result


def _parse_determinism_policy(
    value: object,
) -> IctalDeterminismPolicyReceipt:
    if not isinstance(value, dict) or set(value) != _DETERMINISM_POLICY_FIELDS:
        raise ValueError("Production determinism policy violates its closed schema")
    try:
        return IctalDeterminismPolicyReceipt(**value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Production determinism policy is invalid") from exc


def tusz_native_annotation_roster_sha256(
    manifest: TUSZIctalTrainingManifest,
) -> str:
    """Hash every native target/sidecar receipt used by one training corpus."""

    if not isinstance(manifest, TUSZIctalTrainingManifest):
        raise TypeError("manifest must be TUSZIctalTrainingManifest")
    rows = tuple(
        (
            event.event_id,
            event.channel_annotation_sha256,
            event.global_annotation_sha256,
            event.annotation_pair_sha256,
            event.target_sha256,
            event.target_mask_sha256,
            event.bin_states_sha256,
        )
        for event in manifest
    )
    return _canonical_sha256(
        {
            "schema_version": "soz_tusz_native_ictal_annotation_roster_v1",
            "target_semantics": ICTAL_NATIVE_TARGET_SEMANTICS,
            "events": rows,
        }
    )


def _selection_plan(
    protocol_artifact: IctalConceptOOFProtocolArtifact,
    selection: str,
) -> tuple[IctalConceptOOFPlan, int | None, str]:
    if not isinstance(protocol_artifact, IctalConceptOOFProtocolArtifact):
        raise TypeError("protocol_artifact must be a strict-loader OOF artifact")
    normalized = str(selection).strip().lower()
    if normalized == "final":
        return protocol_artifact.protocol.final_plan, None, normalized
    match = _SELECTION_RE.fullmatch(normalized)
    if match is None:
        raise ValueError("selection must be fold0, fold1, fold2, fold3, fold4, or final")
    fold = int(match.group(1))
    return protocol_artifact.protocol.for_fold(fold), fold, normalized


@dataclass(frozen=True)
class ValidatedIctalProductionSelection:
    plan: IctalConceptOOFPlan
    oof_fold: int | None
    selection: str
    promotion_gate_policy_artifact_sha256: str
    promotion_gate_policy_bundle_receipt_sha256: str
    promotion_gate_policy_receipt_sha256: str
    promotion_gate_policy_document_sha256: str
    held_out_exclusion_public_patient_ids: tuple[str, ...]
    held_out_exclusion_public_roster_sha256: str
    native_evaluation_role: str
    native_evaluation_public_patient_ids: tuple[str, ...]
    native_evaluation_public_roster_sha256: str
    native_unevaluable_public_patient_ids: tuple[str, ...]
    native_unevaluable_public_roster_sha256: str
    native_unevaluable_omission_rows: tuple[
        tuple[str, str, str, tuple[str, ...], str | None], ...
    ]
    native_unevaluable_omission_reason_counts: tuple[
        tuple[str, tuple[tuple[str, int], ...]], ...
    ]
    native_unevaluable_omission_roster_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.plan, IctalConceptOOFPlan):
            raise TypeError("plan must be IctalConceptOOFPlan")
        if self.plan.oof_fold != self.oof_fold:
            raise ValueError("Selected OOF fold disagrees with its plan")
        for field in (
            "promotion_gate_policy_artifact_sha256",
            "promotion_gate_policy_bundle_receipt_sha256",
            "promotion_gate_policy_receipt_sha256",
            "promotion_gate_policy_document_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if self.native_evaluation_role not in {
            "source_train_oof_fold_heldout_native_tusz",
            "source_dev_native_tusz",
        }:
            raise ValueError("Unexpected native-evaluation role")
        held_out = _canonical_public_roster(
            self.held_out_exclusion_public_patient_ids,
            field="held_out_exclusion_public_patient_ids",
            allow_empty=False,
        )
        if self.held_out_exclusion_public_roster_sha256 != _public_roster_sha256(
            held_out
        ):
            raise ValueError("Held-out exclusion public roster SHA mismatch")
        native_evaluable = _canonical_public_roster(
            self.native_evaluation_public_patient_ids,
            field="native_evaluation_public_patient_ids",
            allow_empty=False,
        )
        if self.native_evaluation_public_roster_sha256 != _public_roster_sha256(
            native_evaluable
        ):
            raise ValueError("Native-evaluation public roster SHA mismatch")
        native_unevaluable = _canonical_public_roster(
            self.native_unevaluable_public_patient_ids,
            field="native_unevaluable_public_patient_ids",
            allow_empty=True,
        )
        if self.native_unevaluable_public_roster_sha256 != (
            _attrition_public_roster_sha256(native_unevaluable)
        ):
            raise ValueError("Native-unevaluable public roster SHA mismatch")
        if set(native_evaluable) & set(native_unevaluable):
            raise ValueError("Native evaluable and unevaluable rosters overlap")
        if not set(native_evaluable) <= set(held_out):
            raise ValueError("Native-evaluation roster is outside held-out exclusion")
        if self.oof_fold is None:
            if native_unevaluable or self.native_unevaluable_omission_rows:
                raise ValueError("Final source-dev evaluation cannot claim fold attrition")
        elif set(held_out) != set(native_evaluable) | set(native_unevaluable):
            raise ValueError(
                "Fold held-out exclusion must partition into evaluable and unevaluable"
            )
        rows = tuple(self.native_unevaluable_omission_rows)
        if rows != tuple(
            sorted(
                rows,
                key=lambda row: (
                    row[0], row[1], "" if row[4] is None else row[4], row[3]
                ),
            )
        ):
            raise ValueError("Native-unevaluable omission rows are not canonical")
        row_patients = {row[0] for row in rows}
        if row_patients != set(native_unevaluable):
            raise ValueError(
                "Every native-unevaluable patient requires exact omission rows"
            )
        expected_counts = _native_unevaluable_reason_counts(rows)
        if self.native_unevaluable_omission_reason_counts != expected_counts:
            raise ValueError("Native-unevaluable omission reason counts mismatch")
        if self.native_unevaluable_omission_roster_sha256 != (
            _native_unevaluable_omission_roster_sha256(rows)
        ):
            raise ValueError("Native-unevaluable omission roster SHA mismatch")


def _master_native_attrition_proof(
    manifest: TUSZIctalTrainingManifest,
    native_unevaluable_patient_ids: Sequence[str],
) -> tuple[tuple[str, str, str, tuple[str, ...], str | None], ...]:
    """Prove that fold attrition is solely complete master-manifest omission."""

    patients = _canonical_public_roster(
        native_unevaluable_patient_ids,
        field="native_unevaluable_patient_ids",
        allow_empty=True,
    )
    if not patients:
        return ()
    patient_set = set(patients)
    if patient_set & {event.patient_id for event in manifest.events}:
        raise ValueError(
            "A native-unevaluable patient has an eligible master-manifest event"
        )

    all_source_ids = set(manifest.authorized_source_record_sha256s) | set(
        manifest.excluded_source_record_sha256s
    )
    source_patient: dict[str, str] = {}
    event_source_ids_by_patient: dict[str, set[str]] = {}
    omission_source_ids_by_patient: dict[str, set[str]] = {}
    for event in manifest.events:
        previous = source_patient.setdefault(
            event.public_record_sha256, event.patient_id
        )
        if previous != event.patient_id:
            raise ValueError("Master manifest has contradictory source patients")
        event_source_ids_by_patient.setdefault(event.patient_id, set()).add(
            event.public_record_sha256
        )
    for omission in manifest.omissions:
        previous = source_patient.setdefault(
            omission.public_record_sha256, omission.patient_id
        )
        if previous != omission.patient_id:
            raise ValueError("Master manifest has contradictory source patients")
        omission_source_ids_by_patient.setdefault(omission.patient_id, set()).add(
            omission.public_record_sha256
        )
    if set(source_patient) != all_source_ids:
        raise ValueError("Master manifest source omission accounting is incomplete")

    excluded_source_ids = set(manifest.excluded_source_record_sha256s)
    for patient_id in patients:
        patient_source_ids = {
            source_id
            for source_id, source_patient_id in source_patient.items()
            if source_patient_id == patient_id
        }
        if not patient_source_ids:
            raise ValueError(
                "Held-out patient is absent from both master events and omissions: "
                f"{patient_id}"
            )
        if patient_source_ids & excluded_source_ids:
            raise ValueError(
                "Fold attrition must use master-authorized sources, not protected "
                f"source omissions: {patient_id}"
            )
        if event_source_ids_by_patient.get(patient_id):
            raise ValueError(
                "A native-unevaluable patient has an eligible master-manifest event"
            )
        if omission_source_ids_by_patient.get(patient_id, set()) != patient_source_ids:
            raise ValueError(
                "Native-unevaluable patient lacks complete master omission accounting: "
                f"{patient_id}"
            )

    selected_rows = (
        (
            omission.patient_id,
            omission.relative_edf_path,
            omission.public_record_sha256,
            omission.reasons,
            omission.event_id,
        )
        for omission in manifest.omissions
        if omission.patient_id in patient_set
    )
    rows = tuple(
        sorted(
            selected_rows,
            key=lambda row: (
                row[0], row[1], "" if row[4] is None else row[4], row[3]
            ),
        )
    )
    if {row[0] for row in rows} != patient_set:
        raise ValueError("Every native-unevaluable patient requires master omissions")
    return rows


def _prove_training_held_out_exclusion(
    manifest: TUSZIctalTrainingManifest,
    held_out_patient_ids: Sequence[str],
) -> None:
    """Recheck event- and source-level isolation for every protected patient."""

    held_out = set(
        _canonical_public_roster(
            held_out_patient_ids,
            field="held_out_patient_ids",
            allow_empty=False,
        )
    )
    if held_out & {event.patient_id for event in manifest.events}:
        raise ValueError("Selected plan's protected public patients entered training")
    source_patient: dict[str, str] = {}
    for item in (*manifest.events, *manifest.omissions):
        previous = source_patient.setdefault(item.public_record_sha256, item.patient_id)
        if previous != item.patient_id:
            raise ValueError("Training manifest has contradictory source patients")
    held_out_source_ids = {
        source_id
        for source_id, patient_id in source_patient.items()
        if patient_id in held_out
    }
    if held_out_source_ids - set(manifest.excluded_source_record_sha256s):
        raise ValueError("A held-out patient's source remains training-authorized")


def validate_ictal_production_selection(
    *,
    promotion_gate_policy_artifact: (
        VerifiedIctalPromotionGatePolicyArtifact | None
    ) = None,
    expected_promotion_gate_policy_artifact_sha256: str | None = None,
    expected_promotion_gate_policy_bundle_receipt_sha256: str | None = None,
    protocol_artifact: IctalConceptOOFProtocolArtifact,
    expected_protocol_artifact_sha256: str,
    expected_protocol_receipt_sha256: str,
    expected_split_manifest_sha256: str,
    selection: str,
    training_manifest: TUSZIctalTrainingManifest,
    training_corpus: VerifiedFormalTokenCorpusArtifact,
    expected_training_corpus_index_sha256: str,
    native_evaluation_manifest: (
        TUSZIctalTrainingManifest | VerifiedIctalNativeEvalManifestArtifact
    ),
    native_evaluation_corpus: (
        VerifiedFormalTokenCorpusArtifact
        | VerifiedIctalNativeEvalTokenCorpusArtifact
    ),
    expected_native_evaluation_corpus_index_sha256: str,
) -> ValidatedIctalProductionSelection:
    """Fail closed before optimization on fold, split, corpus, or roster drift."""

    if not isinstance(
        promotion_gate_policy_artifact,
        VerifiedIctalPromotionGatePolicyArtifact,
    ):
        raise RuntimeError(
            "Formal ictal training requires an externally frozen gate-policy "
            "artifact issued by the strict loader"
        )
    if (
        expected_promotion_gate_policy_artifact_sha256 is None
        or expected_promotion_gate_policy_bundle_receipt_sha256 is None
    ):
        raise RuntimeError(
            "Formal ictal training requires externally pinned gate-policy "
            "artifact and bundle-receipt SHA256 values"
        )
    promotion_gate_policy_artifact.assert_unchanged()
    if promotion_gate_policy_artifact.artifact_sha256 != _require_sha256(
        expected_promotion_gate_policy_artifact_sha256,
        field="expected_promotion_gate_policy_artifact_sha256",
    ):
        raise ValueError("Ictal training gate-policy artifact SHA mismatch")
    if promotion_gate_policy_artifact.receipt_sha256 != _require_sha256(
        expected_promotion_gate_policy_bundle_receipt_sha256,
        field="expected_promotion_gate_policy_bundle_receipt_sha256",
    ):
        raise ValueError("Ictal training gate-policy bundle receipt SHA mismatch")
    if protocol_artifact.artifact_sha256 != _require_sha256(
        expected_protocol_artifact_sha256,
        field="expected_protocol_artifact_sha256",
    ):
        raise ValueError("OOF protocol artifact SHA mismatch")
    if protocol_artifact.protocol_sha256 != _require_sha256(
        expected_protocol_receipt_sha256,
        field="expected_protocol_receipt_sha256",
    ):
        raise ValueError("OOF protocol receipt SHA mismatch")
    split_sha = _require_sha256(
        expected_split_manifest_sha256,
        field="expected_split_manifest_sha256",
    )
    if protocol_artifact.protocol.receipt.split_manifest_sha256 != split_sha:
        raise ValueError("OOF protocol uses the wrong target split manifest")
    if not isinstance(training_manifest, TUSZIctalTrainingManifest):
        raise TypeError("Training manifest must be a TUSZ training manifest")
    if not isinstance(training_corpus, VerifiedFormalTokenCorpusArtifact):
        raise TypeError("Training corpus must be a strict-loader formal artifact")
    if training_corpus.index_sha256 != _require_sha256(
        expected_training_corpus_index_sha256,
        field="expected_training_corpus_index_sha256",
    ):
        raise ValueError("Training token-corpus index SHA mismatch")
    if not isinstance(
        native_evaluation_corpus,
        (VerifiedFormalTokenCorpusArtifact, VerifiedIctalNativeEvalTokenCorpusArtifact),
    ):
        raise TypeError("Native-evaluation corpus must be a strict-loader artifact")
    if native_evaluation_corpus.index_sha256 != _require_sha256(
        expected_native_evaluation_corpus_index_sha256,
        field="expected_native_evaluation_corpus_index_sha256",
    ):
        raise ValueError("Native-evaluation token-corpus index SHA mismatch")
    if training_corpus.index_sha256 == native_evaluation_corpus.index_sha256:
        raise ValueError("Training and native-evaluation corpus indices must differ")
    if not training_manifest.preflight_performed:
        raise ValueError("Formal training requires signal preflight")
    training_preprocess_config = asdict(training_manifest.preprocess_config)
    if isinstance(
        native_evaluation_manifest, VerifiedIctalNativeEvalManifestArtifact
    ):
        native_preprocess_config = dict(
            native_evaluation_manifest.manifest.preprocess_config
        )
    elif isinstance(native_evaluation_manifest, TUSZIctalTrainingManifest):
        native_preprocess_config = asdict(
            native_evaluation_manifest.preprocess_config
        )
    else:
        raise TypeError(
            "Native-evaluation manifest must be a formal TUSZ or "
            "evaluation-only manifest"
        )
    if training_preprocess_config != native_preprocess_config:
        raise ValueError(
            "Training and native-evaluation preprocessing configurations differ"
        )
    if training_corpus.training_source_manifest_sha256 != (
        training_manifest.manifest_sha256
    ):
        raise ValueError("Training corpus is bound to a different TUSZ manifest")
    plan, fold, normalized_selection = _selection_plan(
        protocol_artifact, selection
    )
    if plan.receipt.split_manifest_sha256 != split_sha:
        raise ValueError("Selected plan uses the wrong target split manifest")
    if training_manifest.cohort_receipt.receipt_sha256 != (
        plan.training_cohort.receipt.receipt_sha256
    ):
        raise ValueError("Training manifest does not belong to the selected OOF plan")
    held_out_exclusion_patients = tuple(
        sorted(plan.held_out_public_patient_keys)
    )
    if fold is None:
        if not isinstance(
            native_evaluation_manifest, VerifiedIctalNativeEvalManifestArtifact
        ) or not isinstance(
            native_evaluation_corpus, VerifiedIctalNativeEvalTokenCorpusArtifact
        ):
            raise TypeError(
                "Final selection requires the source-dev evaluation-only manifest and corpus"
            )
        if training_manifest.derived_from_manifest_sha256 is not None:
            raise ValueError("Final selection requires the non-derived master manifest")
        if training_corpus.training_source_manifest_sha256 != (
            training_corpus.master_source_manifest_sha256
        ):
            raise ValueError("Final selection requires a master-role token corpus")
        crosswalk = dict(protocol_artifact.protocol.receipt.target_public_crosswalk)
        native_patients = tuple(
            sorted(
                crosswalk[target_id]
                for target_id in protocol_artifact.protocol.receipt.source_dev_patient_ids
            )
        )
        if native_evaluation_manifest.manifest.patient_ids != native_patients:
            raise ValueError(
                "Final native evaluation must equal the complete source-dev public roster"
            )
        if native_evaluation_manifest.manifest.target_patient_ids != tuple(
            sorted(protocol_artifact.protocol.receipt.source_dev_patient_ids)
        ):
            raise ValueError(
                "Final native evaluation changed the complete source-dev target roster"
            )
        if (
            native_evaluation_corpus.manifest_artifact_sha256
            != native_evaluation_manifest.artifact_sha256
            or native_evaluation_corpus.manifest_receipt_sha256
            != native_evaluation_manifest.receipt_sha256
            or native_evaluation_corpus.signal_preflight_artifact_sha256
            != native_evaluation_manifest.manifest.source_signal_preflight_artifact_sha256
            or native_evaluation_corpus.signal_preflight_receipt_sha256
            != native_evaluation_manifest.manifest.source_signal_preflight_receipt_sha256
        ):
            raise ValueError(
                "Final evaluation corpus is bound to another manifest or signal bundle"
            )
        native_unevaluable_patients: tuple[str, ...] = ()
        native_unevaluable_omission_rows: tuple[
            tuple[str, str, str, tuple[str, ...], str | None], ...
        ] = ()
        native_role = "source_dev_native_tusz"
    else:
        if not isinstance(native_evaluation_manifest, TUSZIctalTrainingManifest) or not isinstance(
            native_evaluation_corpus, VerifiedFormalTokenCorpusArtifact
        ):
            raise TypeError(
                "Fold selections require the formal master/fold TUSZ evaluation corpus"
            )
        if not native_evaluation_manifest.preflight_performed:
            raise ValueError("Fold native evaluation requires signal preflight")
        if native_evaluation_manifest.derived_from_manifest_sha256 is not None:
            raise ValueError(
                "Fold native evaluation requires the non-derived master manifest"
            )
        if native_evaluation_corpus.training_source_manifest_sha256 != (
            native_evaluation_manifest.manifest_sha256
        ):
            raise ValueError("Fold evaluation corpus is bound to a different manifest")
        if native_evaluation_corpus.training_source_manifest_sha256 != (
            native_evaluation_corpus.master_source_manifest_sha256
        ):
            raise ValueError(
                "Fold native evaluation requires a master-role token corpus"
            )
        if training_manifest.derived_from_manifest_sha256 != (
            training_corpus.master_source_manifest_sha256
        ):
            raise ValueError("Fold manifest does not derive from the corpus master")
        if training_manifest.derived_from_manifest_sha256 != (
            native_evaluation_manifest.manifest_sha256
        ) or training_corpus.master_source_manifest_sha256 != (
            native_evaluation_manifest.manifest_sha256
        ):
            raise ValueError(
                "Fold training and native evaluation do not share the exact master manifest"
            )
        if training_corpus.training_source_manifest_sha256 == (
            training_corpus.master_source_manifest_sha256
        ):
            raise ValueError("Fold selection cannot use a master-role token corpus")
        master_event_patients = set(native_evaluation_manifest.patient_ids)
        native_patients = tuple(
            patient_id
            for patient_id in held_out_exclusion_patients
            if patient_id in master_event_patients
        )
        if not native_patients:
            raise ValueError(
                "Fold native evaluation has no evaluable held-out patients"
            )
        native_unevaluable_patients = tuple(
            patient_id
            for patient_id in held_out_exclusion_patients
            if patient_id not in master_event_patients
        )
        native_unevaluable_omission_rows = _master_native_attrition_proof(
            native_evaluation_manifest,
            native_unevaluable_patients,
        )
        native_role = "source_train_oof_fold_heldout_native_tusz"

    training_patients = set(training_manifest.patient_ids)
    native_patient_set = set(native_patients)
    if training_patients & native_patient_set:
        raise ValueError("Native-evaluation patients leaked into concept fitting")
    evaluation_patient_ids = (
        native_evaluation_manifest.manifest.patient_ids
        if isinstance(
            native_evaluation_manifest, VerifiedIctalNativeEvalManifestArtifact
        )
        else native_evaluation_manifest.patient_ids
    )
    missing_native = tuple(sorted(native_patient_set - set(evaluation_patient_ids)))
    if missing_native:
        raise ValueError(
            "Native-evaluation corpus omits selected held-out patients: "
            f"{missing_native}"
        )
    _prove_training_held_out_exclusion(
        training_manifest,
        held_out_exclusion_patients,
    )
    native_unevaluable_reason_counts = _native_unevaluable_reason_counts(
        native_unevaluable_omission_rows
    )
    return ValidatedIctalProductionSelection(
        plan=plan,
        oof_fold=fold,
        selection=normalized_selection,
        promotion_gate_policy_artifact_sha256=(
            promotion_gate_policy_artifact.artifact_sha256
        ),
        promotion_gate_policy_bundle_receipt_sha256=(
            promotion_gate_policy_artifact.receipt_sha256
        ),
        promotion_gate_policy_receipt_sha256=(
            promotion_gate_policy_artifact.policy_receipt_sha256
        ),
        promotion_gate_policy_document_sha256=(
            promotion_gate_policy_artifact.policy_document_sha256
        ),
        held_out_exclusion_public_patient_ids=held_out_exclusion_patients,
        held_out_exclusion_public_roster_sha256=_public_roster_sha256(
            held_out_exclusion_patients
        ),
        native_evaluation_role=native_role,
        native_evaluation_public_patient_ids=native_patients,
        native_evaluation_public_roster_sha256=_public_roster_sha256(
            native_patients
        ),
        native_unevaluable_public_patient_ids=native_unevaluable_patients,
        native_unevaluable_public_roster_sha256=(
            _attrition_public_roster_sha256(native_unevaluable_patients)
        ),
        native_unevaluable_omission_rows=native_unevaluable_omission_rows,
        native_unevaluable_omission_reason_counts=(
            native_unevaluable_reason_counts
        ),
        native_unevaluable_omission_roster_sha256=(
            _native_unevaluable_omission_roster_sha256(
                native_unevaluable_omission_rows
            )
        ),
    )


@dataclass(frozen=True)
class IctalProductionRunArtifact:
    path: Path
    manifest_sha256: str
    checkpoint: IctalConceptCheckpointArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("path must be an absolute pathlib.Path")
        _require_sha256(self.manifest_sha256, field="manifest_sha256")
        if not isinstance(self.checkpoint, IctalConceptCheckpointArtifact):
            raise TypeError("checkpoint must be IctalConceptCheckpointArtifact")


@dataclass(frozen=True)
class LoadedIctalProductionRun:
    path: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    checkpoint: LoadedIctalConceptCheckpoint


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_output(path: str | Path) -> Path:
    target = Path(os.path.abspath(path))
    if target.name in {"", ".", ".."}:
        raise ValueError("Production output requires a concrete directory")
    for component in (target.parent, *target.parent.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("Production output parent cannot traverse symlinks")
    if not target.parent.is_dir():
        raise FileNotFoundError("Production output parent does not exist")
    if os.path.lexists(target):
        raise FileExistsError(f"Production output already exists: {target}")
    return target


def _native_metrics_payload(
    metrics: IctalConceptMetrics,
    *,
    mean_patient_loss: float,
    n_events: int,
) -> dict[str, object]:
    return {
        "target_semantics": ICTAL_NATIVE_TARGET_SEMANTICS,
        "deepsoz_soz_labels_used": False,
        "missing_tusz_bins_imputed_as_negative": False,
        "mean_patient_loss": float(mean_patient_loss),
        "n_events": int(n_events),
        **asdict(metrics),
    }


def train_formal_ictal_production_run(
    *,
    promotion_gate_policy_artifact: (
        VerifiedIctalPromotionGatePolicyArtifact | None
    ) = None,
    expected_promotion_gate_policy_artifact_sha256: str | None = None,
    expected_promotion_gate_policy_bundle_receipt_sha256: str | None = None,
    protocol_artifact: IctalConceptOOFProtocolArtifact,
    expected_protocol_artifact_sha256: str,
    expected_protocol_receipt_sha256: str,
    expected_split_manifest_sha256: str,
    selection: str,
    training_manifest: TUSZIctalTrainingManifest,
    training_corpus: VerifiedFormalTokenCorpusArtifact,
    expected_training_corpus_index_sha256: str,
    native_evaluation_manifest: (
        TUSZIctalTrainingManifest | VerifiedIctalNativeEvalManifestArtifact
    ),
    native_evaluation_corpus: (
        VerifiedFormalTokenCorpusArtifact
        | VerifiedIctalNativeEvalTokenCorpusArtifact
    ),
    expected_native_evaluation_corpus_index_sha256: str,
    edf_root: str | Path,
    output_directory: str | Path,
    device: str | torch.device = "cuda",
) -> IctalProductionRunArtifact:
    """Train, evaluate on native labels, and atomically publish one producer."""

    execution_device = torch.device(device)
    if execution_device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if execution_device.type == "cuda":
        validate_ictal_cuda_environment()
    target = _safe_output(output_directory)
    training_config = ICTAL_PRODUCTION_CONFIG
    selection_receipt = validate_ictal_production_selection(
        promotion_gate_policy_artifact=promotion_gate_policy_artifact,
        expected_promotion_gate_policy_artifact_sha256=(
            expected_promotion_gate_policy_artifact_sha256
        ),
        expected_promotion_gate_policy_bundle_receipt_sha256=(
            expected_promotion_gate_policy_bundle_receipt_sha256
        ),
        protocol_artifact=protocol_artifact,
        expected_protocol_artifact_sha256=expected_protocol_artifact_sha256,
        expected_protocol_receipt_sha256=expected_protocol_receipt_sha256,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        selection=selection,
        training_manifest=training_manifest,
        training_corpus=training_corpus,
        expected_training_corpus_index_sha256=(
            expected_training_corpus_index_sha256
        ),
        native_evaluation_manifest=native_evaluation_manifest,
        native_evaluation_corpus=native_evaluation_corpus,
        expected_native_evaluation_corpus_index_sha256=(
            expected_native_evaluation_corpus_index_sha256
        ),
    )
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    training_dataset = build_tusz_ictal_token_bag_dataset(
        training_manifest, edf_root, training_corpus
    )
    if isinstance(
        native_evaluation_manifest, VerifiedIctalNativeEvalManifestArtifact
    ):
        if not isinstance(
            native_evaluation_corpus, VerifiedIctalNativeEvalTokenCorpusArtifact
        ):
            raise TypeError("Evaluation-only manifest requires evaluation-only corpus")
        evaluation_dataset = build_ictal_native_eval_token_bag_dataset(
            native_evaluation_manifest, edf_root, native_evaluation_corpus
        )
        native_evaluation_manifest_sha256 = (
            native_evaluation_manifest.receipt_sha256
        )
    else:
        if not isinstance(native_evaluation_corpus, VerifiedFormalTokenCorpusArtifact):
            raise TypeError("Formal fold evaluation manifest requires formal corpus")
        evaluation_dataset = build_tusz_ictal_token_bag_dataset(
            native_evaluation_manifest, edf_root, native_evaluation_corpus
        )
        native_evaluation_manifest_sha256 = (
            native_evaluation_manifest.manifest_sha256
        )
    if training_dataset.foundation_feature_receipt_sha256 != (
        evaluation_dataset.foundation_feature_receipt_sha256
    ):
        raise ValueError("Training and native-evaluation corpora use different foundations")

    first_binding = training_corpus.events[0]
    first_token = load_labram_concept_tokens(
        first_binding.bundle_path,
        expected_manifest_sha256=first_binding.bundle_manifest_sha256,
    )
    foundation_receipt = first_token.foundation_feature_receipt
    if first_token.foundation_feature_receipt_sha256 != (
        training_dataset.foundation_feature_receipt_sha256
    ):
        raise ValueError("Foundation feature receipt drifted after dataset construction")

    if execution_device.type == "cuda":
        cuda_index = (
            execution_device.index
            if execution_device.index is not None
            else torch.cuda.current_device()
        )
        cuda_devices = [cuda_index]
    else:
        cuda_devices = []
    with ictal_determinism_runtime(
        training_config,
        execution_device_type=execution_device.type,
    ) as production_determinism_policy:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(training_config.seed)
            if execution_device.type == "cuda":
                torch.cuda.manual_seed_all(training_config.seed)
            head = IctalInvolvementHead(
                token_dim=foundation_receipt.token_dim,
                hidden_dim=ICTAL_FIXED_HEAD_HIDDEN_DIM,
            ).to(execution_device)
            run_receipt = train_fixed_epoch_ictal_head(
                head,
                training_dataset,
                config=training_config,
                split_manifest_sha256=expected_split_manifest_sha256,
                oof_protocol_receipt_sha256=(
                    protocol_artifact.protocol.receipt.receipt_sha256
                ),
                oof_plan_receipt_sha256=(
                    selection_receipt.plan.receipt.receipt_sha256
                ),
                oof_fold=selection_receipt.oof_fold,
                training_target_patient_ids=(
                    selection_receipt.plan.training_target_patient_ids
                ),
                held_out_target_patient_ids=(
                    selection_receipt.plan.held_out_target_patient_ids
                ),
                training_target_roster_sha256=(
                    selection_receipt.plan.receipt.training_target_roster_sha256
                ),
                held_out_target_roster_sha256=(
                    selection_receipt.plan.receipt.held_out_target_roster_sha256
                ),
            )
            if run_receipt.determinism_policy != production_determinism_policy:
                raise RuntimeError(
                    "Training run changed the frozen production determinism policy"
                )
            native_epoch, native_metrics = evaluate_cached_ictal_patients(
                head,
                evaluation_dataset,
                selection_receipt.native_evaluation_public_patient_ids,
                event_microbatch_size=training_config.event_microbatch_size,
            )

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        if not isinstance(
            promotion_gate_policy_artifact,
            VerifiedIctalPromotionGatePolicyArtifact,
        ):
            raise RuntimeError("Validated ictal gate-policy capability was lost")
        promotion_gate_policy_artifact.assert_unchanged()
        transient_run = save_ictal_training_run_receipt(
            staging / ".training-run-source", run_receipt
        )
        checkpoint = save_ictal_concept_checkpoint(
            staging / ICTAL_PRODUCTION_CHECKPOINT_DIRECTORY,
            head,
            foundation_feature_receipt=foundation_receipt,
            tusz_annotation_sha256=tusz_native_annotation_roster_sha256(
                training_manifest
            ),
            tusz_manifest_sha256=training_manifest.manifest_sha256,
            split_manifest_sha256=expected_split_manifest_sha256,
            oof_plan_receipt_sha256=(
                selection_receipt.plan.receipt.receipt_sha256
            ),
            oof_protocol_receipt_sha256=(
                protocol_artifact.protocol.receipt.receipt_sha256
            ),
            training_run_artifact=transient_run,
            scaler_sha256=ICTAL_IDENTITY_SCALER_SHA256,
            training_target_patient_ids=(
                selection_receipt.plan.training_target_patient_ids
            ),
            held_out_target_patient_ids=(
                selection_receipt.plan.held_out_target_patient_ids
            ),
            training_target_roster_sha256=(
                selection_receipt.plan.receipt.training_target_roster_sha256
            ),
            held_out_target_roster_sha256=(
                selection_receipt.plan.receipt.held_out_target_roster_sha256
            ),
            oof_fold=selection_receipt.oof_fold,
            epoch=training_config.fixed_epochs - 1,
            seed=training_config.seed,
        )
        shutil.rmtree(transient_run.path)
        native_payload = _native_metrics_payload(
            native_metrics,
            mean_patient_loss=native_epoch.mean_patient_loss,
            n_events=native_epoch.n_events,
        )
        payload = {
            "schema_version": ICTAL_PRODUCTION_RUN_SCHEMA,
            "selection": selection_receipt.selection,
            "oof_fold": selection_receipt.oof_fold,
            "split_manifest_sha256": expected_split_manifest_sha256,
            "oof_protocol_artifact_sha256": protocol_artifact.artifact_sha256,
            "oof_protocol_receipt_sha256": protocol_artifact.protocol_sha256,
            "oof_plan_receipt_sha256": (
                selection_receipt.plan.receipt.receipt_sha256
            ),
            "promotion_gate_policy_artifact_sha256": (
                selection_receipt.promotion_gate_policy_artifact_sha256
            ),
            "promotion_gate_policy_bundle_receipt_sha256": (
                selection_receipt.promotion_gate_policy_bundle_receipt_sha256
            ),
            "promotion_gate_policy_receipt_sha256": (
                selection_receipt.promotion_gate_policy_receipt_sha256
            ),
            "promotion_gate_policy_document_sha256": (
                selection_receipt.promotion_gate_policy_document_sha256
            ),
            "training_config": asdict(training_config),
            "training_config_sha256": training_config.receipt_sha256,
            "determinism_policy": asdict(run_receipt.determinism_policy),
            "determinism_policy_sha256": (
                run_receipt.determinism_policy_sha256
            ),
            "training_manifest_sha256": training_manifest.manifest_sha256,
            "training_corpus_index_sha256": training_corpus.index_sha256,
            "training_run_receipt_sha256": run_receipt.receipt_sha256,
            "training_source_public_patient_ids": list(
                training_dataset.patient_ids
            ),
            "training_source_public_roster_sha256": _public_roster_sha256(
                training_dataset.patient_ids
            ),
            "held_out_exclusion_public_patient_ids": list(
                selection_receipt.held_out_exclusion_public_patient_ids
            ),
            "held_out_exclusion_public_roster_sha256": (
                selection_receipt.held_out_exclusion_public_roster_sha256
            ),
            "native_evaluation_role": (
                selection_receipt.native_evaluation_role
            ),
            "native_evaluation_manifest_sha256": (
                native_evaluation_manifest_sha256
            ),
            "native_evaluation_corpus_index_sha256": (
                native_evaluation_corpus.index_sha256
            ),
            "native_evaluation_public_patient_ids": list(
                selection_receipt.native_evaluation_public_patient_ids
            ),
            "native_evaluation_public_roster_sha256": (
                selection_receipt.native_evaluation_public_roster_sha256
            ),
            "native_unevaluable_public_patient_ids": list(
                selection_receipt.native_unevaluable_public_patient_ids
            ),
            "native_unevaluable_public_roster_sha256": (
                selection_receipt.native_unevaluable_public_roster_sha256
            ),
            "native_unevaluable_omission_rows": (
                _native_unevaluable_omission_payload(
                    selection_receipt.native_unevaluable_omission_rows
                )
            ),
            "native_unevaluable_omission_reason_counts": (
                _native_unevaluable_reason_count_payload(
                    selection_receipt.native_unevaluable_omission_reason_counts
                )
            ),
            "native_unevaluable_omission_roster_sha256": (
                selection_receipt.native_unevaluable_omission_roster_sha256
            ),
            "native_metrics": native_payload,
            "checkpoint_directory": ICTAL_PRODUCTION_CHECKPOINT_DIRECTORY,
            "checkpoint_manifest_sha256": checkpoint.manifest_sha256,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
        }
        encoded = _canonical_json_bytes(payload)
        if not 1 <= len(encoded) <= _MAX_PRODUCTION_MANIFEST_BYTES:
            raise ValueError("Production-run manifest has an invalid size")
        manifest_path = staging / ICTAL_PRODUCTION_RUN_FILENAME
        manifest_path.write_bytes(encoded)
        _fsync_file(manifest_path)
        _fsync_directory(staging / ICTAL_PRODUCTION_CHECKPOINT_DIRECTORY)
        _fsync_directory(staging)
        load_ictal_concept_checkpoint(
            staging / ICTAL_PRODUCTION_CHECKPOINT_DIRECTORY,
            expected_manifest_sha256=checkpoint.manifest_sha256,
        )
        if os.path.lexists(target):
            raise FileExistsError(f"Production output already exists: {target}")
        os.rename(staging, target)
        published = True
        _fsync_directory(target.parent)
        return IctalProductionRunArtifact(
            path=target,
            manifest_sha256=hashlib.sha256(encoded).hexdigest(),
            checkpoint=IctalConceptCheckpointArtifact(
                path=target / ICTAL_PRODUCTION_CHECKPOINT_DIRECTORY,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
                manifest_sha256=checkpoint.manifest_sha256,
            ),
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def load_ictal_production_run(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> LoadedIctalProductionRun:
    """Strictly reload the atomic production bundle and its safe checkpoint."""

    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_dir() or source.resolve() != source:
        raise ValueError("Ictal production bundle must be a regular directory")
    if {item.name for item in source.iterdir()} != {
        ICTAL_PRODUCTION_CHECKPOINT_DIRECTORY,
        ICTAL_PRODUCTION_RUN_FILENAME,
    }:
        raise ValueError("Ictal production bundle has missing or unknown files")
    manifest_path = source / ICTAL_PRODUCTION_RUN_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Ictal production manifest must be a regular file")
    raw = manifest_path.read_bytes()
    if not 1 <= len(raw) <= _MAX_PRODUCTION_MANIFEST_BYTES:
        raise ValueError("Ictal production manifest has an invalid size")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Ictal production manifest is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or _canonical_json_bytes(payload) != raw:
        raise ValueError("Ictal production manifest is not canonical JSON")
    if set(payload) != _PRODUCTION_FIELDS:
        raise ValueError("Ictal production manifest violates its closed schema")
    if payload.get("schema_version") != ICTAL_PRODUCTION_RUN_SCHEMA:
        raise ValueError("Unsupported ictal production-run schema")
    for sha_field in (
        "split_manifest_sha256",
        "oof_protocol_artifact_sha256",
        "oof_protocol_receipt_sha256",
        "oof_plan_receipt_sha256",
        "promotion_gate_policy_artifact_sha256",
        "promotion_gate_policy_bundle_receipt_sha256",
        "promotion_gate_policy_receipt_sha256",
        "promotion_gate_policy_document_sha256",
        "training_config_sha256",
        "determinism_policy_sha256",
        "training_manifest_sha256",
        "training_corpus_index_sha256",
        "training_run_receipt_sha256",
        "training_source_public_roster_sha256",
        "held_out_exclusion_public_roster_sha256",
        "native_evaluation_manifest_sha256",
        "native_evaluation_corpus_index_sha256",
        "native_evaluation_public_roster_sha256",
        "native_unevaluable_public_roster_sha256",
        "native_unevaluable_omission_roster_sha256",
        "checkpoint_manifest_sha256",
        "checkpoint_sha256",
    ):
        _require_sha256(payload.get(sha_field), field=sha_field)
    if payload.get("checkpoint_directory") != ICTAL_PRODUCTION_CHECKPOINT_DIRECTORY:
        raise ValueError("Production checkpoint directory is not canonical")
    selection = str(payload.get("selection"))
    if selection == "final":
        if payload.get("oof_fold") is not None or payload.get(
            "native_evaluation_role"
        ) != "source_dev_native_tusz":
            raise ValueError("Final selection has invalid fold/evaluation semantics")
    else:
        match = _SELECTION_RE.fullmatch(selection)
        if (
            match is None
            or payload.get("oof_fold") != int(match.group(1))
            or payload.get("native_evaluation_role")
            != "source_train_oof_fold_heldout_native_tusz"
        ):
            raise ValueError("Fold selection has invalid fold/evaluation semantics")
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None and actual_sha != _require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise ValueError("Ictal production manifest SHA mismatch")
    checkpoint = load_ictal_concept_checkpoint(
        source / ICTAL_PRODUCTION_CHECKPOINT_DIRECTORY,
        expected_manifest_sha256=_require_sha256(
            payload.get("checkpoint_manifest_sha256"),
            field="checkpoint_manifest_sha256",
        ),
    )
    checks = {
        "checkpoint_sha256": (
            checkpoint.checkpoint_sha256 == payload.get("checkpoint_sha256")
        ),
        "split_manifest_sha256": (
            checkpoint.metadata["split_manifest_sha256"]
            == payload.get("split_manifest_sha256")
        ),
        "oof_fold": checkpoint.metadata["oof_fold"] == payload.get("oof_fold"),
        "oof_plan_receipt_sha256": (
            checkpoint.metadata["oof_plan_receipt_sha256"]
            == payload.get("oof_plan_receipt_sha256")
        ),
        "oof_protocol_receipt_sha256": (
            checkpoint.metadata["oof_protocol_receipt_sha256"]
            == payload.get("oof_protocol_receipt_sha256")
        ),
        "training_manifest_sha256": (
            checkpoint.metadata["tusz_manifest_sha256"]
            == payload.get("training_manifest_sha256")
        ),
        "training_corpus_index_sha256": (
            checkpoint.metadata["formal_token_corpus_index_sha256"]
            == payload.get("training_corpus_index_sha256")
        ),
        "determinism_policy": (
            checkpoint.metadata["determinism_policy"]
            == payload.get("determinism_policy")
        ),
        "determinism_policy_sha256": (
            checkpoint.metadata["determinism_policy_sha256"]
            == payload.get("determinism_policy_sha256")
        ),
    }
    failed = tuple(field for field, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            f"Production manifest disagrees with its checkpoint: {failed}"
        )
    metrics = payload.get("native_metrics")
    if not isinstance(metrics, dict) or set(metrics) != _NATIVE_METRIC_FIELDS:
        raise TypeError("native_metrics must be a JSON object")
    if metrics.get("target_semantics") != ICTAL_NATIVE_TARGET_SEMANTICS:
        raise ValueError("Production metrics use the wrong native target semantics")
    if metrics.get("deepsoz_soz_labels_used") is not False:
        raise ValueError("Native concept metrics may not use DeepSOZ SOZ labels")
    if metrics.get("missing_tusz_bins_imputed_as_negative") is not False:
        raise ValueError("Native concept metrics may not impute missing bins")
    mean_loss = metrics.get("mean_patient_loss")
    n_events = metrics.get("n_events")
    if (
        isinstance(mean_loss, bool)
        or not isinstance(mean_loss, (int, float))
        or not math.isfinite(float(mean_loss))
        or float(mean_loss) < 0
    ):
        raise ValueError("Native mean patient loss must be finite and non-negative")
    if isinstance(n_events, bool) or not isinstance(n_events, int) or n_events < 1:
        raise ValueError("Native metric event count must be positive")
    metric_fields = {
        field: metrics[field]
        for field in IctalConceptMetrics.__dataclass_fields__
    }
    IctalConceptMetrics(**metric_fields)
    config_payload = payload.get("training_config")
    if not isinstance(config_payload, dict):
        raise TypeError("training_config must be a JSON object")
    try:
        config = IctalTrainingConfig(**config_payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("Production training config is invalid") from exc
    if config != ICTAL_PRODUCTION_CONFIG:
        raise ValueError("Production training config is not the frozen policy")
    if config.receipt_sha256 != payload.get("training_config_sha256"):
        raise ValueError("Production training config SHA mismatch")
    determinism_policy = _parse_determinism_policy(
        payload.get("determinism_policy")
    )
    if payload.get("determinism_policy_sha256") != (
        determinism_policy.receipt_sha256
    ):
        raise ValueError("Production determinism policy SHA mismatch")
    policy_config_checks = {
        "cublas_workspace_config": (
            determinism_policy.required_cublas_workspace_config
            == config.cublas_workspace_config
        ),
        "deterministic_algorithms": (
            determinism_policy.deterministic_algorithms_enabled
            == config.deterministic_algorithms
        ),
        "deterministic_warn_only": (
            determinism_policy.deterministic_algorithms_warn_only
            == config.deterministic_warn_only
        ),
        "cudnn_deterministic": (
            determinism_policy.cudnn_deterministic
            == config.cudnn_deterministic
        ),
        "cudnn_benchmark": (
            determinism_policy.cudnn_benchmark == config.cudnn_benchmark
        ),
        "cuda_matmul_allow_tf32": (
            determinism_policy.cuda_matmul_allow_tf32
            == config.cuda_matmul_allow_tf32
        ),
        "cudnn_allow_tf32": (
            determinism_policy.cudnn_allow_tf32 == config.cudnn_allow_tf32
        ),
    }
    failed_policy_config = tuple(
        field for field, passed in policy_config_checks.items() if not passed
    )
    if failed_policy_config:
        raise ValueError(
            "Production determinism policy disagrees with config: "
            f"{failed_policy_config}"
        )
    public_patients = payload.get("training_source_public_patient_ids")
    held_out_patients = payload.get("held_out_exclusion_public_patient_ids")
    native_patients = payload.get("native_evaluation_public_patient_ids")
    native_unevaluable_patients = payload.get(
        "native_unevaluable_public_patient_ids"
    )
    rosters = (
        public_patients,
        held_out_patients,
        native_patients,
        native_unevaluable_patients,
    )
    if any(
        not isinstance(roster, list)
        or any(not isinstance(patient_id, str) for patient_id in roster)
        for roster in rosters
    ):
        raise TypeError("Production patient rosters must be JSON arrays")
    public_roster = _canonical_public_roster(
        tuple(public_patients),
        field="training_source_public_patient_ids",
        allow_empty=False,
    )
    held_out_roster = _canonical_public_roster(
        tuple(held_out_patients),
        field="held_out_exclusion_public_patient_ids",
        allow_empty=False,
    )
    native_roster = _canonical_public_roster(
        tuple(native_patients),
        field="native_evaluation_public_patient_ids",
        allow_empty=False,
    )
    native_unevaluable_roster = _canonical_public_roster(
        tuple(native_unevaluable_patients),
        field="native_unevaluable_public_patient_ids",
        allow_empty=True,
    )
    if payload.get("training_source_public_roster_sha256") != (
        _public_roster_sha256(public_roster)
    ):
        raise ValueError("Training source-public roster SHA mismatch")
    if payload.get("held_out_exclusion_public_roster_sha256") != (
        _public_roster_sha256(held_out_roster)
    ):
        raise ValueError("Held-out exclusion public roster SHA mismatch")
    if payload.get("native_evaluation_public_roster_sha256") != (
        _public_roster_sha256(native_roster)
    ):
        raise ValueError("Native-evaluation public roster SHA mismatch")
    if payload.get("native_unevaluable_public_roster_sha256") != (
        _attrition_public_roster_sha256(native_unevaluable_roster)
    ):
        raise ValueError("Native-unevaluable public roster SHA mismatch")
    if set(public_roster) & set(held_out_roster):
        raise ValueError("Held-out exclusion patients overlap concept fitting")
    if set(native_roster) & set(native_unevaluable_roster):
        raise ValueError("Native evaluable and unevaluable rosters overlap")
    if not set(native_roster) <= set(held_out_roster):
        raise ValueError("Native evaluation lies outside held-out exclusion")
    if selection == "final":
        if native_unevaluable_roster:
            raise ValueError("Final source-dev evaluation cannot claim fold attrition")
    elif set(held_out_roster) != set(native_roster) | set(
        native_unevaluable_roster
    ):
        raise ValueError(
            "Fold held-out exclusion must partition into evaluable and unevaluable"
        )

    omission_rows = _parse_native_unevaluable_omission_payload(
        payload.get("native_unevaluable_omission_rows")
    )
    omission_reason_counts = _parse_native_unevaluable_reason_count_payload(
        payload.get("native_unevaluable_omission_reason_counts")
    )
    if {row[0] for row in omission_rows} != set(native_unevaluable_roster):
        raise ValueError(
            "Every native-unevaluable patient requires exact omission rows"
        )
    if omission_reason_counts != _native_unevaluable_reason_counts(omission_rows):
        raise ValueError("Native-unevaluable omission reason counts mismatch")
    if payload.get("native_unevaluable_omission_roster_sha256") != (
        _native_unevaluable_omission_roster_sha256(omission_rows)
    ):
        raise ValueError("Native-unevaluable omission roster SHA mismatch")
    if metrics["n_patients"] != len(native_roster):
        raise ValueError("Native metric patient count disagrees with its roster")
    embedded_run = load_ictal_training_run_receipt(
        source
        / ICTAL_PRODUCTION_CHECKPOINT_DIRECTORY
        / "training_run",
        expected_training_run_receipt_sha256=_require_sha256(
            payload.get("training_run_receipt_sha256"),
            field="training_run_receipt_sha256",
        ),
    )
    run = embedded_run.training_run_receipt
    run_checks = {
        "training_config": run.config == config,
        "determinism_policy": run.determinism_policy == determinism_policy,
        "determinism_policy_sha256": (
            run.determinism_policy_sha256
            == payload["determinism_policy_sha256"]
        ),
        "split_manifest_sha256": (
            run.split_manifest_sha256 == payload["split_manifest_sha256"]
        ),
        "oof_protocol_receipt_sha256": (
            run.oof_protocol_receipt_sha256
            == payload["oof_protocol_receipt_sha256"]
        ),
        "oof_plan_receipt_sha256": (
            run.oof_plan_receipt_sha256 == payload["oof_plan_receipt_sha256"]
        ),
        "oof_fold": run.oof_fold == payload["oof_fold"],
        "training_manifest_sha256": (
            run.training_manifest_sha256 == payload["training_manifest_sha256"]
        ),
        "training_corpus_index_sha256": (
            run.formal_token_corpus_index_sha256
            == payload["training_corpus_index_sha256"]
        ),
        "source_public_patient_ids": (
            run.concept_training_patient_ids == public_roster
        ),
        "source_public_roster_sha256": (
            run.concept_training_patient_roster_sha256
            == payload["training_source_public_roster_sha256"]
        ),
        "final_head_state_sha256": (
            run.final_head_state_sha256 == ictal_head_state_sha256(checkpoint.head)
        ),
    }
    failed_run = tuple(field for field, passed in run_checks.items() if not passed)
    if failed_run:
        raise ValueError(
            f"Production manifest disagrees with its training run: {failed_run}"
        )
    return LoadedIctalProductionRun(
        path=source,
        manifest=payload,
        manifest_sha256=actual_sha,
        checkpoint=checkpoint,
    )


__all__: Sequence[str] = (
    "ICTAL_FIXED_HEAD_HIDDEN_DIM",
    "ICTAL_NATIVE_TARGET_SEMANTICS",
    "ICTAL_PRODUCTION_CONFIG",
    "ICTAL_PRODUCTION_RUN_SCHEMA",
    "IctalProductionRunArtifact",
    "LoadedIctalProductionRun",
    "ValidatedIctalProductionSelection",
    "load_ictal_production_run",
    "train_formal_ictal_production_run",
    "tusz_native_annotation_roster_sha256",
    "validate_ictal_production_selection",
)
