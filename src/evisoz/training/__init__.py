"""Fail-closed training entry points for the EviSOZ research route."""

from .stage0_guard import (
    Stage0TrainingBlocked,
    require_stage0_training_authorized,
)
from .loader_entrypoint import open_authorized_training_records
from .authorized_epoch import run_authorized_evidence_epoch
from .typed_loss import TypedLossContractError, compute_typed_evidence_losses
from .targets import SLOT_VOCABULARIES, batch_typed_loss_targets, build_typed_loss_targets
from .evidence_trainer import forward_real_evidence_record, run_stage1_evidence_epoch
from .stage1_objectives import (
    channel_edge_dropout_consistency_loss,
    compute_authorized_stage1_objective,
    compute_stage1_objective_bundle,
    masked_latent_reconstruction_loss,
    montage_consistency_loss,
    motif_soft_target_loss,
    motif_teacher_kl_loss,
)
from .residual_trainer import (
    baseline_preservation_kl_loss,
    compute_residual_localization_objective,
    residual_node_localization_loss,
    run_authorized_residual_epoch,
)
from .grounding_feedback import apply_one_shot_grounded_feedback, evaluate_grounding_gate
from .training_receipts import (
    STAGE1_TRAINING_BLOCK_RECEIPT_SCHEMA_VERSION,
    validate_stage1_training_block_receipt,
)
from .execution_plan import (
    EXECUTION_PLAN_SCHEMA_VERSION,
    build_evisoz_execution_plan,
    validate_evisoz_execution_plan,
)

__all__ = [
    "Stage0TrainingBlocked",
    "require_stage0_training_authorized",
    "open_authorized_training_records",
    "run_authorized_evidence_epoch",
    "TypedLossContractError",
    "compute_typed_evidence_losses",
    "SLOT_VOCABULARIES",
    "batch_typed_loss_targets",
    "build_typed_loss_targets",
    "forward_real_evidence_record",
    "run_stage1_evidence_epoch",
    "channel_edge_dropout_consistency_loss",
    "compute_authorized_stage1_objective",
    "compute_stage1_objective_bundle",
    "masked_latent_reconstruction_loss",
    "montage_consistency_loss",
    "motif_soft_target_loss",
    "motif_teacher_kl_loss",
    "baseline_preservation_kl_loss",
    "compute_residual_localization_objective",
    "residual_node_localization_loss",
    "run_authorized_residual_epoch",
    "apply_one_shot_grounded_feedback",
    "evaluate_grounding_gate",
    "STAGE1_TRAINING_BLOCK_RECEIPT_SCHEMA_VERSION",
    "validate_stage1_training_block_receipt",
    "EXECUTION_PLAN_SCHEMA_VERSION",
    "build_evisoz_execution_plan",
    "validate_evisoz_execution_plan",
]
