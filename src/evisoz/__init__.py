"""EviSOZ-LM isolated implementation layer.

The package starts with Stage-0 contracts.  Model and reporting modules are
added only after the content-addressed data, channel and montage contracts
pass their fail-closed tests.
"""

from .data.artifact_ref import (
    ARTIFACT_REF_SCHEMA_VERSION,
    build_json_artifact_ref,
    build_raw_artifact_ref,
    validate_artifact_ref,
)

__all__ = [
    "ARTIFACT_REF_SCHEMA_VERSION",
    "build_json_artifact_ref",
    "build_raw_artifact_ref",
    "validate_artifact_ref",
]
