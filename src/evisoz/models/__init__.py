"""Trainable EviSOZ evidence components.

These modules operate on already-materialized, typed EEG tokens.  They do not
open private reports, teacher models, or the frozen v29 target resources.
"""

from .clinical_evidence import (
    ClinicalEvidenceOutput,
    ClinicalEvidenceQueryDecoder,
    ClinicalMotifAdapter,
    EviSOZEvidencePipeline,
    PatientEvidenceAggregator,
    ResidualSOZHead,
    SparseTemporalSpatialEvidenceEncoder,
    MOTIF_NAMES,
    STRUCTURED_EVIDENCE_PIPELINE_CONFIG_SCHEMA_VERSION,
    validate_structured_evidence_pipeline_config,
)
from .qwen_connector import (
    DEFAULT_EVIDENCE_TOKEN_COUNT,
    EvidenceTokenResampler,
    QWEN3_8_27B_HIDDEN_SIZE,
    assemble_qwen_embedding_inputs,
    clause_mil_alignment_loss,
    evidence_guided_mask,
)
from .predicted_evidence import (
    PREDICTED_EVIDENCE_SCHEMA_VERSION,
    build_predicted_evidence_packet,
    validate_predicted_evidence_packet,
)
from .real_signal_adapter import (
    NODE_INPUT_DIM,
    PATCH_SAMPLES,
    PATCH_SECONDS,
    RealDualMontageTokenAdapter,
    RealSignalAdapterReceipt,
    SAMPLE_RATE_HZ,
    STANDARD19_COUNT,
    TCP22_COUNT,
    TOKEN_DIM,
    project_token_dimension,
)

__all__ = [
    "ClinicalEvidenceOutput",
    "ClinicalEvidenceQueryDecoder",
    "ClinicalMotifAdapter",
    "EviSOZEvidencePipeline",
    "PatientEvidenceAggregator",
    "ResidualSOZHead",
    "SparseTemporalSpatialEvidenceEncoder",
    "MOTIF_NAMES",
    "STRUCTURED_EVIDENCE_PIPELINE_CONFIG_SCHEMA_VERSION",
    "validate_structured_evidence_pipeline_config",
    "DEFAULT_EVIDENCE_TOKEN_COUNT",
    "EvidenceTokenResampler",
    "QWEN3_8_27B_HIDDEN_SIZE",
    "assemble_qwen_embedding_inputs",
    "clause_mil_alignment_loss",
    "evidence_guided_mask",
    "PREDICTED_EVIDENCE_SCHEMA_VERSION",
    "build_predicted_evidence_packet",
    "validate_predicted_evidence_packet",
    "NODE_INPUT_DIM",
    "PATCH_SAMPLES",
    "PATCH_SECONDS",
    "RealDualMontageTokenAdapter",
    "RealSignalAdapterReceipt",
    "SAMPLE_RATE_HZ",
    "STANDARD19_COUNT",
    "TCP22_COUNT",
    "TOKEN_DIM",
    "project_token_dimension",
]
