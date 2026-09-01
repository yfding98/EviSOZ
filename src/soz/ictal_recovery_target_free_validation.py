"""Target-free validation for LaBraM ictal recovery producers.

This module validates the same fold/native corpus boundaries as the formal
ictal-production validator without accepting a DeepSOZ source table or a
``DeepSOZReferenceRegistry``.  Its only target-side input is the canonical OOF
protocol JSON loaded through :func:`load_target_free_ictal_oof_protocol`.
That JSON contains identities, split assignments and public-patient
crosswalks, but no SOZ electrode vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .concept_oof import IctalConceptOOFPlanReceipt
from .data.tusz_training import TUSZIctalTrainingManifest
from .formal_token_corpus import VerifiedFormalTokenCorpusArtifact
from .ictal_gate_policy import VerifiedIctalPromotionGatePolicyArtifact
from .ictal_native_eval import (
    VerifiedIctalNativeEvalManifestArtifact,
    VerifiedIctalNativeEvalTokenCorpusArtifact,
)
from .ictal_production import (
    _attrition_public_roster_sha256,
    _master_native_attrition_proof,
    _native_unevaluable_reason_counts,
    _native_unevaluable_omission_roster_sha256,
    _prove_training_held_out_exclusion,
    _public_roster_sha256,
)
from .ictal_recovery_evidence import TargetFreeOOFProtocolView


@dataclass(frozen=True)
class TargetFreeValidatedIctalRecoverySelection:
    """Validated fold/final selection containing identities but no SOZ target."""

    plan_receipt: IctalConceptOOFPlanReceipt
    oof_fold: int | None
    selection: str
    promotion_gate_policy_artifact_sha256: str
    promotion_gate_policy_bundle_receipt_sha256: str
    held_out_exclusion_public_patient_ids: tuple[str, ...]
    native_evaluation_public_patient_ids: tuple[str, ...]
    native_unevaluable_public_patient_ids: tuple[str, ...]
    native_unevaluable_omission_rows: tuple[
        tuple[str, str, str, tuple[str, ...], str | None], ...
    ]


def _selection_plan(
    protocol: TargetFreeOOFProtocolView, selection: object
) -> tuple[IctalConceptOOFPlanReceipt, int | None, str]:
    if not isinstance(protocol, TargetFreeOOFProtocolView):
        raise TypeError("protocol must come from the target-free strict loader")
    normalized = str(selection).strip().lower()
    if normalized == "final":
        return protocol.final_plan_receipt, None, normalized
    if len(normalized) != 5 or not normalized.startswith("fold"):
        raise ValueError("selection must be fold0..fold4 or final")
    suffix = normalized[-1]
    if suffix not in "01234":
        raise ValueError("selection must be fold0..fold4 or final")
    fold = int(suffix)
    return protocol.fold_plan_receipts[fold], fold, normalized


def _expected_cohort_receipt(plan: IctalConceptOOFPlanReceipt) -> str:
    bindings = dict(plan.cohort_bindings)
    role = "train" if plan.oof_fold is not None else "dev"
    try:
        return bindings[role]
    except KeyError as exc:  # pragma: no cover - plan dataclass already closes this
        raise ValueError("Target-free OOF plan lacks its training cohort binding") from exc


def validate_target_free_ictal_recovery_selection(
    *,
    promotion_gate_policy_artifact: VerifiedIctalPromotionGatePolicyArtifact,
    expected_promotion_gate_policy_artifact_sha256: str,
    expected_promotion_gate_policy_bundle_receipt_sha256: str,
    protocol: TargetFreeOOFProtocolView,
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
) -> TargetFreeValidatedIctalRecoverySelection:
    """Fail closed on the production lineage without loading DeepSOZ targets."""

    if not isinstance(
        promotion_gate_policy_artifact, VerifiedIctalPromotionGatePolicyArtifact
    ):
        raise RuntimeError("Recovery requires the strict gate-policy capability")
    promotion_gate_policy_artifact.assert_unchanged()
    if (
        promotion_gate_policy_artifact.artifact_sha256
        != expected_promotion_gate_policy_artifact_sha256
        or promotion_gate_policy_artifact.receipt_sha256
        != expected_promotion_gate_policy_bundle_receipt_sha256
    ):
        raise ValueError("Recovery gate-policy lineage mismatch")
    protocol.assert_unchanged()
    if (
        protocol.artifact_sha256 != expected_protocol_artifact_sha256
        or protocol.receipt_sha256 != expected_protocol_receipt_sha256
        or protocol.receipt.split_manifest_sha256 != expected_split_manifest_sha256
    ):
        raise ValueError("Target-free OOF protocol lineage mismatch")
    plan, fold, normalized = _selection_plan(protocol, selection)
    if plan.split_manifest_sha256 != expected_split_manifest_sha256:
        raise ValueError("Selected target-free plan uses another split")

    if not isinstance(training_manifest, TUSZIctalTrainingManifest):
        raise TypeError("training_manifest must be a TUSZ training manifest")
    if not isinstance(training_corpus, VerifiedFormalTokenCorpusArtifact):
        raise TypeError("training_corpus must be a strict formal token corpus")
    if training_corpus.index_sha256 != expected_training_corpus_index_sha256:
        raise ValueError("Training token-corpus index SHA mismatch")
    if not isinstance(
        native_evaluation_corpus,
        (VerifiedFormalTokenCorpusArtifact, VerifiedIctalNativeEvalTokenCorpusArtifact),
    ):
        raise TypeError("native evaluation corpus must be a strict-loader artifact")
    if (
        native_evaluation_corpus.index_sha256
        != expected_native_evaluation_corpus_index_sha256
    ):
        raise ValueError("Native token-corpus index SHA mismatch")
    if training_corpus.index_sha256 == native_evaluation_corpus.index_sha256:
        raise ValueError("Training and native corpora must be distinct")
    if not training_manifest.preflight_performed:
        raise ValueError("Recovery training manifest lacks signal preflight")
    if training_corpus.training_source_manifest_sha256 != training_manifest.manifest_sha256:
        raise ValueError("Training corpus binds another manifest")
    if training_manifest.cohort_receipt.receipt_sha256 != _expected_cohort_receipt(plan):
        raise ValueError("Training manifest does not belong to the selected OOF plan")
    if tuple(training_manifest.authorized_source_record_sha256s) != tuple(
        plan.authorized_record_sha256s
    ):
        raise ValueError("Training manifest authorized records differ from the OOF plan")

    if isinstance(native_evaluation_manifest, VerifiedIctalNativeEvalManifestArtifact):
        native_preprocess = dict(native_evaluation_manifest.manifest.preprocess_config)
    elif isinstance(native_evaluation_manifest, TUSZIctalTrainingManifest):
        from dataclasses import asdict

        native_preprocess = asdict(native_evaluation_manifest.preprocess_config)
    else:
        raise TypeError("native evaluation manifest has the wrong strict type")
    from dataclasses import asdict

    if asdict(training_manifest.preprocess_config) != native_preprocess:
        raise ValueError("Training/native preprocessing configurations differ")

    held = tuple(sorted(plan.held_out_public_patient_keys))
    if fold is None:
        if not isinstance(
            native_evaluation_manifest, VerifiedIctalNativeEvalManifestArtifact
        ) or not isinstance(
            native_evaluation_corpus, VerifiedIctalNativeEvalTokenCorpusArtifact
        ):
            raise TypeError("Final recovery requires evaluation-only source-dev inputs")
        if training_manifest.derived_from_manifest_sha256 is not None:
            raise ValueError("Final recovery requires the master training manifest")
        if (
            training_corpus.training_source_manifest_sha256
            != training_corpus.master_source_manifest_sha256
        ):
            raise ValueError("Final recovery requires the master token corpus")
        crosswalk = protocol.crosswalk
        native = tuple(
            sorted(crosswalk[value] for value in protocol.receipt.source_dev_patient_ids)
        )
        if native_evaluation_manifest.manifest.patient_ids != native:
            raise ValueError("Final native roster is not complete source-dev")
        if native_evaluation_manifest.manifest.target_patient_ids != tuple(
            sorted(protocol.receipt.source_dev_patient_ids)
        ):
            raise ValueError("Final target identity roster is not complete source-dev")
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
            raise ValueError("Final native corpus binds another signal/manifest")
        unevaluable: tuple[str, ...] = ()
        omission_rows: tuple[
            tuple[str, str, str, tuple[str, ...], str | None], ...
        ] = ()
    else:
        if not isinstance(native_evaluation_manifest, TUSZIctalTrainingManifest) or not isinstance(
            native_evaluation_corpus, VerifiedFormalTokenCorpusArtifact
        ):
            raise TypeError("Fold recovery requires master TUSZ native inputs")
        if (
            not native_evaluation_manifest.preflight_performed
            or native_evaluation_manifest.derived_from_manifest_sha256 is not None
        ):
            raise ValueError("Fold native evaluation requires the preflighted master")
        if (
            native_evaluation_corpus.training_source_manifest_sha256
            != native_evaluation_manifest.manifest_sha256
            or native_evaluation_corpus.training_source_manifest_sha256
            != native_evaluation_corpus.master_source_manifest_sha256
        ):
            raise ValueError("Fold native corpus is not the exact master corpus")
        if (
            training_manifest.derived_from_manifest_sha256
            != native_evaluation_manifest.manifest_sha256
            or training_corpus.master_source_manifest_sha256
            != native_evaluation_manifest.manifest_sha256
            or training_corpus.training_source_manifest_sha256
            == training_corpus.master_source_manifest_sha256
        ):
            raise ValueError("Fold training/native manifests do not share one master")
        master_patients = set(native_evaluation_manifest.patient_ids)
        native = tuple(value for value in held if value in master_patients)
        if not native:
            raise ValueError("Fold has no native-evaluable held-out patient")
        unevaluable = tuple(value for value in held if value not in master_patients)
        omission_rows = _master_native_attrition_proof(
            native_evaluation_manifest, unevaluable
        )

    if set(training_manifest.patient_ids) & set(native):
        raise ValueError("Native-evaluation patient leaked into fitting")
    evaluation_patients: Sequence[str] = (
        native_evaluation_manifest.manifest.patient_ids
        if isinstance(native_evaluation_manifest, VerifiedIctalNativeEvalManifestArtifact)
        else native_evaluation_manifest.patient_ids
    )
    if set(native) - set(evaluation_patients):
        raise ValueError("Native corpus omits a selected held-out patient")
    _prove_training_held_out_exclusion(training_manifest, held)
    # Recompute these receipts now so callers cannot silently drop attrition.
    _public_roster_sha256(held)
    _public_roster_sha256(native)
    _attrition_public_roster_sha256(unevaluable)
    _native_unevaluable_reason_counts(omission_rows)
    _native_unevaluable_omission_roster_sha256(omission_rows)
    return TargetFreeValidatedIctalRecoverySelection(
        plan_receipt=plan,
        oof_fold=fold,
        selection=normalized,
        promotion_gate_policy_artifact_sha256=(
            promotion_gate_policy_artifact.artifact_sha256
        ),
        promotion_gate_policy_bundle_receipt_sha256=(
            promotion_gate_policy_artifact.receipt_sha256
        ),
        held_out_exclusion_public_patient_ids=held,
        native_evaluation_public_patient_ids=native,
        native_unevaluable_public_patient_ids=unevaluable,
        native_unevaluable_omission_rows=omission_rows,
    )


__all__ = (
    "TargetFreeValidatedIctalRecoverySelection",
    "validate_target_free_ictal_recovery_selection",
)
