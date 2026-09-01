"""Content-addressed synthetic runtime probe for the public ELM encoder.

The probe is deliberately weaker than teacher-candidate admission.  It only
proves that the pinned public source and externally stored checkpoints can
perform a deterministic CPU forward pass on synthetic tensors.  It never
opens EviSOZ EEG, reports, labels or a teacher training loader.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

from src.evisoz.data.artifact_ref import (
    canonical_json_sha256,
    validate_artifact_ref,
)


ELM_RUNTIME_PROBE_SCHEMA_VERSION = "evisoz_elm_runtime_probe_v1"
ELM_RUNTIME_PROBE_STATUS = "synthetic_forward_pass_only"
_HASH_PLACEHOLDER = "0" * 64
_ID_PREFIX = "EVISOZ-ELM-PROBE-"


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    body = _hash_source(value)
    body["probe_id"] = "CONTENT-ADDRESS-PENDING"
    return body


def _finite_tree(value: object, context: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{context}[{index}]")


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def build_elm_runtime_probe_receipt(
    *,
    source: Mapping[str, object],
    variants: Sequence[Mapping[str, object]],
    probe: Mapping[str, object],
    safety: Mapping[str, object],
    _skip_validation: bool = False,
) -> dict[str, Any]:
    """Build and validate one synthetic ELM runtime receipt."""

    if type(source) is not dict or set(source) != {
        "repository", "commit", "source_commit_verified", "source_root",
        "model_artifacts", "software",
    }:
        raise ValueError("ELM runtime source fields drifted")
    if source["repository"] != "https://github.com/SamGijsen/ELM":
        raise ValueError("ELM repository drifted")
    if source["commit"] != "fcd929a57ce3dc9a409be37a71f4ee80ee59979d":
        raise ValueError("ELM source commit drifted")
    if source["source_commit_verified"] is not True:
        raise ValueError("ELM source commit was not verified")
    if not isinstance(source["source_root"], str) or not source["source_root"]:
        raise ValueError("ELM source root is invalid")
    artifacts = source["model_artifacts"]
    if type(artifacts) is not dict or set(artifacts) != {"5s", "60s"}:
        raise ValueError("ELM model artifact roster drifted")
    for variant in ("5s", "60s"):
        row = artifacts[variant]
        if type(row) is not dict or set(row) != {"config_ref", "checkpoint_ref"}:
            raise ValueError("ELM model artifact row drifted")
        validate_artifact_ref(row["config_ref"])
        validate_artifact_ref(row["checkpoint_ref"])
    software = source["software"]
    if type(software) is not dict or set(software) != {
        "python", "torch", "source_requirements_sha256"
    }:
        raise ValueError("ELM software manifest drifted")
    if not isinstance(software["python"], str) or not software["python"]:
        raise ValueError("ELM Python version is invalid")
    if not isinstance(software["torch"], str) or not software["torch"]:
        raise ValueError("ELM torch version is invalid")
    _sha256(software["source_requirements_sha256"], "source requirements SHA-256")

    rows = [dict(row) for row in variants]
    if [row.get("variant") for row in rows] != ["5s", "60s"]:
        raise ValueError("ELM runtime variants must be ordered 5s, 60s")
    for row in rows:
        if set(row) != {
            "variant", "input_shape", "raw_embedding_shape",
            "projected_embedding_shape", "finite", "repeat_exact",
        }:
            raise ValueError("ELM runtime variant fields drifted")
        if row["input_shape"] not in ([1, 20, 500], [1, 20, 6000]):
            raise ValueError("ELM runtime input shape drifted")
        if row["raw_embedding_shape"] != [1, 96] or row["projected_embedding_shape"] != [1, 256]:
            raise ValueError("ELM runtime output shape drifted")
        if row["finite"] is not True or row["repeat_exact"] is not True:
            raise ValueError("ELM runtime numerical probe did not pass")

    if type(probe) is not dict or set(probe) != {
        "input_kind", "batch_size", "patient_data_opened", "forward_count",
    }:
        raise ValueError("ELM runtime probe fields drifted")
    if probe != {
        "input_kind": "synthetic_zeros",
        "batch_size": 1,
        "patient_data_opened": False,
        "forward_count": 4,
    }:
        raise ValueError("ELM runtime probe safety/input contract drifted")
    if type(safety) is not dict or set(safety) != {
        "weights_loaded_for_synthetic_forward", "candidate_cache_materialized",
        "large_scale_teacher_inference", "training", "optimizer", "patient_data",
        "physician_report_text", "qwen_generation", "training_authorized",
    }:
        raise ValueError("ELM runtime safety fields drifted")
    if safety != {
        "weights_loaded_for_synthetic_forward": True,
        "candidate_cache_materialized": False,
        "large_scale_teacher_inference": False,
        "training": False,
        "optimizer": False,
        "patient_data": False,
        "physician_report_text": False,
        "qwen_generation": False,
        "training_authorized": False,
    }:
        raise ValueError("ELM runtime safety contract drifted")

    body: dict[str, Any] = {
        "schema_version": ELM_RUNTIME_PROBE_SCHEMA_VERSION,
        "probe_id": _HASH_PLACEHOLDER,
        "status": ELM_RUNTIME_PROBE_STATUS,
        "source": deepcopy(dict(source)),
        "variants": rows,
        "probe": deepcopy(dict(probe)),
        "safety": deepcopy(dict(safety)),
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["probe_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return body if _skip_validation else validate_elm_runtime_probe_receipt(body)


def validate_elm_runtime_probe_receipt(value: object) -> dict[str, Any]:
    required = {
        "schema_version", "probe_id", "status", "source", "variants", "probe",
        "safety", "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("ELM runtime receipt fields drifted")
    data = deepcopy(value)
    _finite_tree(data)
    if data["schema_version"] != ELM_RUNTIME_PROBE_SCHEMA_VERSION:
        raise ValueError("ELM runtime receipt schema drifted")
    if data["status"] != ELM_RUNTIME_PROBE_STATUS:
        raise ValueError("ELM runtime receipt status drifted")
    # Reuse the strict builder contract while avoiding recursive finalization.
    built = build_elm_runtime_probe_receipt(
        source=data["source"],
        variants=data["variants"],
        probe=data["probe"],
        safety=data["safety"],
        _skip_validation=True,
    )
    if data["probe_id"] != built["probe_id"] or data["receipt_sha256"] != built["receipt_sha256"]:
        raise ValueError("ELM runtime receipt hash drifted")
    return data


__all__ = [
    "ELM_RUNTIME_PROBE_SCHEMA_VERSION",
    "ELM_RUNTIME_PROBE_STATUS",
    "build_elm_runtime_probe_receipt",
    "validate_elm_runtime_probe_receipt",
]
