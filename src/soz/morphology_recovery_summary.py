"""Strict artifact for the paired development-only morphology OOF summary."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping

from .morphology_recovery import MORPHOLOGY_RECOVERY_PROTOCOL_SHA256


MORPHOLOGY_RECOVERY_SUMMARY_SCHEMA = (
    "soz_labram_morphology_hierarchical_oof_paired_summary_v1"
)
MORPHOLOGY_RECOVERY_SUMMARY_FILENAME = "development_summary.json"
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_FIELDS = frozenset(
    {
        "schema_version",
        "development_only",
        "formal_promotion",
        "dense_deployment_authorized",
        "soz_reasoner_authorized",
        "official_tuev_eval_used",
        "threshold_selection_performed",
        "training_performed_by_summary",
        "comparison_scope",
        "target_semantics",
        "candidate_name",
        "baseline_name",
        "protocol_sha256",
        "preflight_receipt_sha256",
        "source_plan_sha256",
        "source_files_sha256",
        "source_item_count",
        "source_group_count",
        "observed_cell_count",
        "source_item_roster_sha256",
        "source_group_roster_sha256",
        "oof_prediction_coverage_complete",
        "candidate_fold_manifests",
        "baseline_fold_artifacts",
        "bootstrap",
        "metrics",
        "interpretation_boundary",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: object, field: str) -> str:
    text = str(value)
    if not _SHA_RE.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return text


def _finite_tree(value: object, field: str = "metrics") -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        return value
    if isinstance(value, list):
        return [_finite_tree(item, f"{field}[]") for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _finite_tree(item, f"{field}.{key}")
            for key, item in value.items()
        }
    raise TypeError(f"Unsupported value at {field}")


def validate_morphology_recovery_summary(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("Morphology recovery summary must be a mapping")
    payload = dict(value)
    if set(payload) != _FIELDS:
        raise ValueError(
            "Closed morphology summary fields changed; "
            f"missing={sorted(_FIELDS-set(payload))}, extra={sorted(set(payload)-_FIELDS)}"
        )
    if payload["schema_version"] != MORPHOLOGY_RECOVERY_SUMMARY_SCHEMA:
        raise ValueError("Unexpected morphology recovery summary schema")
    required_true = {"development_only", "oof_prediction_coverage_complete"}
    required_false = {
        "formal_promotion",
        "dense_deployment_authorized",
        "soz_reasoner_authorized",
        "official_tuev_eval_used",
        "threshold_selection_performed",
        "training_performed_by_summary",
    }
    if any(payload[field] is not True for field in required_true) or any(
        payload[field] is not False for field in required_false
    ):
        raise ValueError("Morphology recovery summary safety flags changed")
    expected_text = {
        "comparison_scope": "same_source_train_items_complete_group_oof_paired",
        "target_semantics": "tuev_native_ce6_bipolar_edge_not_soz",
        "candidate_name": "labram_frozen_shared_adapter_ce6_plus_three_roles",
        "baseline_name": "labram_frozen_independent_ce6_m0",
        "interpretation_boundary": (
            "retrospective_development_comparison_not_dense_M1_or_SOZ_evidence"
        ),
    }
    for field, expected in expected_text.items():
        if payload[field] != expected:
            raise ValueError(f"Morphology recovery summary {field} changed")
    if _require_sha(payload["protocol_sha256"], "protocol_sha256") != MORPHOLOGY_RECOVERY_PROTOCOL_SHA256:
        raise ValueError("Morphology recovery summary protocol changed")
    for field in (
        "preflight_receipt_sha256",
        "source_plan_sha256",
        "source_item_roster_sha256",
        "source_group_roster_sha256",
    ):
        _require_sha(payload[field], field)
    sources = payload["source_files_sha256"]
    if not isinstance(sources, Mapping) or set(sources) != {
        "run_plan",
        "tokens",
        "labels",
        "mask",
        "weights",
    }:
        raise ValueError("Morphology recovery summary source roster changed")
    for field, sha in sources.items():
        _require_sha(sha, f"source_files_sha256.{field}")
    for field in ("source_item_count", "source_group_count", "observed_cell_count"):
        if isinstance(payload[field], bool) or not isinstance(payload[field], int) or payload[field] < 1:
            raise ValueError(f"Morphology recovery summary {field} must be positive")
    candidate = payload["candidate_fold_manifests"]
    baseline = payload["baseline_fold_artifacts"]
    if not isinstance(candidate, list) or len(candidate) != 5:
        raise ValueError("Morphology recovery summary requires five candidate folds")
    if not isinstance(baseline, list) or len(baseline) != 5:
        raise ValueError("Morphology recovery summary requires five baseline folds")
    for fold, row in enumerate(candidate):
        if not isinstance(row, Mapping) or set(row) != {"fold", "manifest_file_sha256"}:
            raise ValueError("Candidate fold summary row changed")
        if row["fold"] != fold:
            raise ValueError("Candidate folds must be ordered 0--4")
        _require_sha(row["manifest_file_sha256"], "candidate manifest SHA")
    for fold, row in enumerate(baseline):
        expected = {
            "fold",
            "receipt_file_sha256",
            "checkpoint_file_sha256",
            "state_sha256",
        }
        if not isinstance(row, Mapping) or set(row) != expected or row["fold"] != fold:
            raise ValueError("Baseline folds must use the closed ordered schema")
        for name in expected - {"fold"}:
            _require_sha(row[name], f"baseline fold {fold} {name}")
    if payload["bootstrap"] != {
        "unit": "tuev_parent_group",
        "paired": True,
        "replicates": 2000,
        "seed": 20260808,
        "interval": "percentile_2.5_97.5",
    }:
        raise ValueError("Morphology recovery bootstrap policy changed")
    metrics = _finite_tree(payload["metrics"])
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Morphology recovery summary metrics are empty")
    payload["metrics"] = metrics
    return payload


@dataclass(frozen=True)
class LoadedMorphologyRecoverySummary:
    path: Path
    payload: dict[str, object]
    file_sha256: str


def save_morphology_recovery_summary(
    output_directory: str | Path, payload: Mapping[str, object]
) -> LoadedMorphologyRecoverySummary:
    validated = validate_morphology_recovery_summary(payload)
    target = Path(os.path.abspath(output_directory))
    if target.name in {"", ".", ".."} or not target.parent.is_dir():
        raise ValueError("Morphology summary requires a concrete existing parent")
    if os.path.lexists(target):
        raise FileExistsError(f"Morphology summary output already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        summary = temporary / MORPHOLOGY_RECOVERY_SUMMARY_FILENAME
        summary.write_bytes(_canonical_bytes(validated) + b"\n")
        with summary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_morphology_recovery_summary(target)


def load_morphology_recovery_summary(
    value: str | Path, *, expected_file_sha256: str | None = None
) -> LoadedMorphologyRecoverySummary:
    path = Path(value).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("Morphology recovery summary artifact must be a directory")
    summary = path / MORPHOLOGY_RECOVERY_SUMMARY_FILENAME
    if not summary.is_file() or summary.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Morphology recovery summary file is missing or oversized")
    sha = _file_sha256(summary)
    if expected_file_sha256 is not None and sha != _require_sha(
        expected_file_sha256, "expected_file_sha256"
    ):
        raise ValueError("Morphology recovery summary is not the expected artifact")
    payload = validate_morphology_recovery_summary(
        json.loads(summary.read_text(encoding="utf-8"))
    )
    return LoadedMorphologyRecoverySummary(path=path, payload=payload, file_sha256=sha)


__all__ = [
    "MORPHOLOGY_RECOVERY_SUMMARY_FILENAME",
    "MORPHOLOGY_RECOVERY_SUMMARY_SCHEMA",
    "LoadedMorphologyRecoverySummary",
    "load_morphology_recovery_summary",
    "save_morphology_recovery_summary",
    "validate_morphology_recovery_summary",
]

