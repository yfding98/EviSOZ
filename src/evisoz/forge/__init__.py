"""Offline SOZ-Forge contracts."""

from .training_example import (
    TRAINING_EXAMPLE_SCHEMA_VERSION,
    build_training_example,
    validate_training_example,
)
from .evidence_binding import (
    BOUND_EVIDENCE_SCHEMA_VERSION,
    BOUND_MATERIALIZATION_SCHEMA_VERSION,
    build_bound_evidence_example,
    materialize_bound_evidence_examples,
    validate_bound_evidence_example,
    validate_bound_evidence_materialization,
)
from .teacher_candidates import (
    TEACHER_CANDIDATE_CACHE_SCHEMA_VERSION,
    TEACHER_CANDIDATE_MATERIALIZATION_SCHEMA_VERSION,
    build_teacher_candidate_cache,
    build_teacher_candidate_materialization,
    validate_teacher_candidate_cache,
    validate_teacher_candidate_materialization,
)

__all__ = [
    "TRAINING_EXAMPLE_SCHEMA_VERSION",
    "build_training_example",
    "validate_training_example",
    "BOUND_EVIDENCE_SCHEMA_VERSION",
    "BOUND_MATERIALIZATION_SCHEMA_VERSION",
    "build_bound_evidence_example",
    "materialize_bound_evidence_examples",
    "validate_bound_evidence_example",
    "validate_bound_evidence_materialization",
    "TEACHER_CANDIDATE_CACHE_SCHEMA_VERSION",
    "TEACHER_CANDIDATE_MATERIALIZATION_SCHEMA_VERSION",
    "build_teacher_candidate_cache",
    "build_teacher_candidate_materialization",
    "validate_teacher_candidate_cache",
    "validate_teacher_candidate_materialization",
]
