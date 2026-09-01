"""Target-isolated dry-run entry for formal common17/LB16 ST16 training.

The module deliberately separates the two numeric process roles:

* a training process is allowed to receive exactly one labelled source-train
  manifest; and
* a prediction process is allowed to receive exactly one EEG-only source-dev
  roster.

No function accepts both artifacts.  The dry-run does not open an EDF, target
sidecar, checkpoint, CUDA context, or source-eval artifact.  It audits the
existing numerical runner and materializes the complete 4,664-record training
denominator, including a separate short-record context-padding arm.  Until
that arm is integrated and replayed on real EDFs, the returned launch gate is
NO-GO; this prevents an excluded 228-record subset from being described as
"complete training".
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

from scripts.build_detector_cleanroom_physical_isolation_v1 import (
    assert_source_dev_eeg_only,
    validate_source_dev_roster,
    validate_source_train_manifest,
)


SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_st16_common17_cleanroom_formal_entry_v1"
)
TRAIN_MANIFEST_SCHEMA: Final[str] = (
    "clinical_eeg_detector_source_train_labeled_manifest_v1"
)
DEV_ROSTER_SCHEMA: Final[str] = (
    "clinical_eeg_detector_source_dev_eeg_only_prediction_roster_v1"
)
TRAIN_DRY_RUN_SCHEMA: Final[str] = (
    "clinical_eeg_st16_common17_formal_training_dry_run_receipt_v1"
)
DEV_DRY_RUN_SCHEMA: Final[str] = (
    "clinical_eeg_st16_common17_formal_dev_prediction_dry_run_receipt_v1"
)
AUDIT_RECEIPT_SCHEMA: Final[str] = (
    "clinical_eeg_st16_common17_cleanroom_formal_entry_audit_receipt_v1"
)
ADMISSION_LEDGER_SCHEMA: Final[str] = (
    "clinical_eeg_st16_common17_training_record_admission_ledger_v1"
)
PENDING: Final[str] = "CONTENT-ADDRESS-PENDING"
TARGET_FS_HZ: Final[int] = 256
TILE_SAMPLES: Final[int] = 15_360
SHORT_POLICY_ID: Final[str] = "ST16-short-reflect-context-valid-support-mask-v1"

_TARGET_BEARING_FRAGMENTS: Final[tuple[str, ...]] = (
    "target",
    "label",
    "reference",
    "csv_bi",
    "seizure",
    "ictal",
    "onset",
    "offset",
    "annotation",
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    if result.get("content_sha256") != PENDING:
        raise ValueError("content-address input lacks pending marker")
    result["content_sha256"] = canonical_sha256(result)
    return result


def _validate_content_address(value: Mapping[str, Any], context: str) -> None:
    observed = value.get("content_sha256")
    if not isinstance(observed, str) or len(observed) != 64:
        raise ValueError(f"{context} lacks a content SHA-256")
    replay = deepcopy(dict(value))
    replay["content_sha256"] = PENDING
    if canonical_sha256(replay) != observed:
        raise ValueError(f"{context} content hash drifted")


def _load_json_regular_file(path: Path, *, expected_sha256: str, role: str) -> dict:
    source = path.resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise PermissionError(f"{role} artifact must be a regular non-symlink file")
    observed_sha = file_sha256(source)
    if observed_sha != expected_sha256:
        raise PermissionError(f"{role} artifact file hash drifted")
    try:
        value = json.loads(source.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} artifact is not valid JSON") from exc
    if type(value) is not dict:
        raise TypeError(f"{role} artifact must be a JSON object")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise PermissionError("formal-entry config must be a regular file")
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("formal-entry config schema drifted")
    if value.get("method_id") != "ST16-C17-LB16-cleanroom-formal-entry-v1":
        raise ValueError("formal-entry method drifted")
    if value["model"]["target_sampling_rate_hz"] != TARGET_FS_HZ:
        raise ValueError("formal-entry target clock drifted")
    if value["model"]["tile_samples"] != TILE_SAMPLES:
        raise ValueError("formal-entry tile geometry drifted")
    if value["short_record_arm"]["policy_id"] != SHORT_POLICY_ID:
        raise ValueError("formal-entry short-record policy drifted")
    if value["dataset_roles"]["source_eval_open_allowed"] is not False:
        raise PermissionError("source-eval must remain closed")
    if value["claim_limits"]["formal_gpu_launch_currently_authorized"] is not False:
        raise PermissionError("this v1 config is dry-run-only")
    for section in (
        "architecture_source",
        "transform_registry",
        "transform_implementation",
        "streaming_ola_implementation",
    ):
        binding = value["model"][section]
        bound_path = Path(binding["path"])
        if not bound_path.is_absolute():
            bound_path = source.parents[1] / bound_path
        if file_sha256(bound_path.resolve(strict=True)) != binding["file_sha256"]:
            raise PermissionError(f"formal-entry {section} binding drifted")
    return deepcopy(value)


def _resolve_bound_data_path(
    config_path: str | Path, relative_path: str
) -> Path:
    source = Path(config_path).resolve(strict=True)
    path = Path(relative_path)
    if not path.is_absolute():
        path = source.parents[1] / path
    return path.resolve(strict=True)


def load_training_manifest_for_process(
    config_path: str | Path,
    manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the sole target-bearing artifact admitted to a training process."""

    config = load_config(config_path)
    contract = config["dataset_roles"]["training"]
    expected_path = _resolve_bound_data_path(config_path, contract["only_manifest_path"])
    supplied = Path(manifest_path).resolve(strict=True)
    # Reject a dev/train role swap before opening the supplied bytes.
    if supplied != expected_path:
        raise PermissionError("training process may open only the bound train manifest")
    raw = _load_json_regular_file(
        supplied,
        expected_sha256=contract["file_sha256"],
        role="training manifest",
    )
    if raw.get("schema_version") != TRAIN_MANIFEST_SCHEMA:
        raise PermissionError("training process received a non-train artifact")
    manifest = validate_source_train_manifest(raw)
    if (
        manifest.get("split") != contract["required_split"]
        or manifest.get("content_sha256") != contract["content_sha256"]
        or len(manifest["records"]) != contract["expected_recording_count"]
        or manifest["inventory"]["patient_count"] != contract["expected_patient_count"]
    ):
        raise PermissionError("training manifest denominator or binding drifted")
    return config, manifest


def load_dev_roster_for_process(
    config_path: str | Path,
    roster_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the sole dataset artifact admitted to a dev prediction process."""

    config = load_config(config_path)
    contract = config["dataset_roles"]["source_dev_prediction"]
    expected_path = _resolve_bound_data_path(config_path, contract["only_roster_path"])
    supplied = Path(roster_path).resolve(strict=True)
    # Reject role confusion before reading attacker-controlled bytes.
    if supplied != expected_path:
        raise PermissionError("dev prediction may open only the bound EEG-only roster")
    raw = _load_json_regular_file(
        supplied,
        expected_sha256=contract["file_sha256"],
        role="source-dev EEG-only roster",
    )
    if raw.get("schema_version") != DEV_ROSTER_SCHEMA:
        raise PermissionError("dev prediction process received a non-dev artifact")
    assert_source_dev_eeg_only(raw)
    roster = validate_source_dev_roster(raw)
    if (
        roster.get("split") != contract["required_split"]
        or roster.get("content_sha256") != contract["content_sha256"]
        or len(roster["records"]) != contract["expected_recording_count"]
        or roster["inventory"]["patient_count"] != contract["expected_patient_count"]
    ):
        raise PermissionError("source-dev roster denominator or binding drifted")
    return config, roster


def provider_target_sample_count(row: Mapping[str, Any]) -> int:
    raw_rate = row.get("sampling_rate_hz_fraction")
    if (
        not isinstance(raw_rate, list)
        or len(raw_rate) != 2
        or type(raw_rate[0]) is not int
        or type(raw_rate[1]) is not int
        or raw_rate[0] <= 0
        or raw_rate[1] <= 0
    ):
        raise ValueError("record sampling rate is malformed")
    source_count = row.get("sample_count")
    if isinstance(source_count, bool) or not isinstance(source_count, int) or source_count < 1:
        raise ValueError("record sample count is malformed")
    rate = Fraction(raw_rate[0], raw_rate[1])
    ratio = Fraction(TARGET_FS_HZ, 1) / rate
    return (source_count * ratio.numerator) // ratio.denominator


def deterministic_reflect_context_pad(
    signal: object,
    *,
    output_samples: int = TILE_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Pad only model context; return an immutable original-support mask.

    NumPy ``reflect`` mirrors without repeating the edge sample.  The mask is
    one only where a target/loss/metric is physically observed.  Padding is
    therefore unavailable to target construction and is never clinical EEG
    evidence, even though the fixed-shape model may use it as context.
    """

    array = np.asarray(signal)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError("short-arm signal must have shape [channels,time>=2]")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("short-arm signal must be floating point")
    if not np.isfinite(array).all():
        raise ValueError("short-arm signal contains nonfinite values")
    if isinstance(output_samples, bool) or not isinstance(output_samples, int):
        raise TypeError("output_samples must be an integer")
    valid = int(array.shape[1])
    if output_samples < valid:
        raise ValueError("short-arm may not truncate observed support")
    padding = output_samples - valid
    carrier = np.ascontiguousarray(array, dtype="<f4")
    if padding:
        carrier = np.pad(carrier, ((0, 0), (0, padding)), mode="reflect")
        carrier = np.ascontiguousarray(carrier, dtype="<f4")
    mask = np.zeros(output_samples, dtype=np.uint8)
    mask[:valid] = 1
    input_hash = hashlib.sha256(
        np.ascontiguousarray(array, dtype="<f4").tobytes(order="C")
    ).hexdigest()
    output_hash = hashlib.sha256(carrier.tobytes(order="C")).hexdigest()
    mask_hash = hashlib.sha256(mask.tobytes(order="C")).hexdigest()
    ledger = _content_address(
        {
            "schema_version": "st16_short_reflect_context_padding_ledger_v1",
            "policy_id": SHORT_POLICY_ID,
            "input_shape": list(array.shape),
            "output_shape": list(carrier.shape),
            "valid_support_sample_range": [0, valid],
            "context_padding_sample_range": (
                [] if padding == 0 else [valid, output_samples]
            ),
            "valid_sample_count": valid,
            "context_padding_sample_count": padding,
            "reflect_mode_without_endpoint_duplication": True,
            "padding_is_observed_EEG": False,
            "padding_may_receive_target_or_metric_weight": False,
            "input_float32_payload_sha256": input_hash,
            "output_float32_payload_sha256": output_hash,
            "valid_support_mask_uint8_sha256": mask_hash,
            "content_sha256": PENDING,
        }
    )
    carrier.setflags(write=False)
    mask.setflags(write=False)
    return carrier, mask, ledger


def masked_dense_binary_cross_entropy(
    probability: object,
    target: object,
    valid_support_mask: object,
    *,
    positive_weight: float,
) -> float:
    """Reference short-arm loss proving padding has exactly zero weight."""

    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(target, dtype=np.uint8)
    mask = np.asarray(valid_support_mask, dtype=np.uint8)
    if p.shape != y.shape or p.shape != mask.shape or p.ndim != 1:
        raise ValueError("probability, target and valid mask must share one axis")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite in [0,1]")
    if np.any(~np.isin(y, [0, 1])) or np.any(~np.isin(mask, [0, 1])):
        raise ValueError("target and valid-support mask must be binary")
    if not math.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError("positive_weight must be finite and positive")
    observed = mask.astype(bool)
    if not observed.any():
        raise ValueError("loss requires observed support")
    clipped = np.clip(p[observed], 1e-7, 1.0 - 1e-7)
    truth = y[observed]
    weights = np.where(truth == 1, positive_weight, 1.0)
    numerator = -np.sum(
        weights
        * (truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped)),
        dtype=np.float64,
    )
    return float(numerator / np.sum(weights, dtype=np.float64))


def _record_admission_row(row: Mapping[str, Any]) -> dict[str, Any]:
    target_count = provider_target_sample_count(row)
    short = target_count < TILE_SAMPLES
    intervals = row.get("global_TERM_seiz_intervals_seconds")
    if not isinstance(intervals, list):
        raise TypeError("training record lacks embedded TERM,seiz intervals")
    return {
        "analysis_identity_id": row["analysis_identity_id"],
        "local_patient_id": row["local_patient_id"],
        "source_row_sha256": row["row_sha256"],
        "provider_target_sample_count": target_count,
        "terminal_admission_state": (
            "planned_short_reflect_context_valid_support_mask"
            if short
            else "planned_native_fully_observed_tile"
        ),
        "valid_support_sample_count": target_count,
        "model_context_padding_sample_count": max(0, TILE_SAMPLES - target_count),
        "loss_mask_one_count": target_count,
        "loss_mask_zero_count": max(0, TILE_SAMPLES - target_count),
        "positive_interval_count": len(intervals),
        "planned_to_contribute_gradient": True,
        "actual_gradient_contribution_observed": False,
    }


def build_training_dry_run(
    *,
    config_path: str | Path,
    train_manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a complete training denominator without opening EEG or sidecars."""

    config, manifest = load_training_manifest_for_process(
        config_path, train_manifest_path
    )
    rows = [_record_admission_row(row) for row in manifest["records"]]
    if len({row["analysis_identity_id"] for row in rows}) != len(rows):
        raise ValueError("training admission ledger repeats an analysis identity")
    rows.sort(key=lambda row: row["analysis_identity_id"])
    short = [
        row
        for row in rows
        if row["terminal_admission_state"]
        == "planned_short_reflect_context_valid_support_mask"
    ]
    native = [
        row
        for row in rows
        if row["terminal_admission_state"]
        == "planned_native_fully_observed_tile"
    ]
    ledger = _content_address(
        {
            "schema_version": ADMISSION_LEDGER_SCHEMA,
            "method_id": config["method_id"],
            "source_train_manifest_content_sha256": manifest["content_sha256"],
            "recording_count": len(rows),
            "native_record_count": len(native),
            "short_context_arm_record_count": len(short),
            "terminal_admission_state_count": len(rows),
            "all_records_have_exactly_one_terminal_admission_state": len(rows)
            == len(manifest["records"]),
            "planned_gradient_contributing_record_count": sum(
                bool(row["planned_to_contribute_gradient"]) for row in rows
            ),
            "actual_gradient_contributing_record_count": 0,
            "actual_gradient_contribution_not_claimed_by_dry_run": True,
            "records": rows,
            "content_sha256": PENDING,
        }
    )
    protocol = config["training_protocol"]
    run_specs = []
    for index, base_seed in enumerate(protocol["independent_seed_base_values"]):
        run_specs.append(
            {
                "replicate_index": index,
                "base_seed": base_seed,
                "best_seed_selection_allowed": False,
                "selection_maximum_epochs": protocol["selection_maximum_epochs"],
                "selection_minimum_epochs": protocol["selection_minimum_epochs"],
                "early_stop_patience_epochs": protocol[
                    "early_stop_patience_epochs"
                ],
                "final_refit_epoch_count_source": protocol[
                    "final_refit_epoch_count_source"
                ],
                "source_dev_target_access_allowed": False,
                "checkpoint_binding_required": {
                    "model_payload_sha256": True,
                    "architecture_source_sha256": config["model"][
                        "architecture_source"
                    ]["file_sha256"],
                    "transform_registry_sha256": config["model"][
                        "transform_registry"
                    ]["file_sha256"],
                    "transform_implementation_sha256": config["model"][
                        "transform_implementation"
                    ]["file_sha256"],
                    "training_manifest_file_sha256": config["dataset_roles"][
                        "training"
                    ]["file_sha256"],
                    "training_manifest_content_sha256": manifest[
                        "content_sha256"
                    ],
                },
            }
        )
    receipt = _content_address(
        {
            "schema_version": TRAIN_DRY_RUN_SCHEMA,
            "method_id": config["method_id"],
            "status": (
                "no_go_formal_gpu_training_pending_short_arm_real_edf_"
                "transform_and_trainer_mask_integration"
            ),
            "process_role": "training_dry_run",
            "dataset_artifact_opened": "source_train_labeled_manifest_only",
            "source_train_manifest_path": str(
                Path(train_manifest_path).resolve(strict=True)
            ),
            "source_train_manifest_file_sha256": file_sha256(train_manifest_path),
            "source_train_manifest_content_sha256": manifest["content_sha256"],
            "source_train_recording_count": len(rows),
            "source_train_patient_count": manifest["inventory"]["patient_count"],
            "native_record_count": len(native),
            "short_context_arm_record_count": len(short),
            "short_context_arm_positive_record_count": sum(
                row["positive_interval_count"] > 0 for row in short
            ),
            "record_terminal_admission_denominator_count": len(rows),
            "planned_gradient_contributing_record_count": len(rows),
            "actual_gradient_contributing_record_count": 0,
            "actual_gradient_contribution_requires_training_receipts": True,
            "admission_ledger_content_sha256": ledger["content_sha256"],
            "short_record_policy_id": SHORT_POLICY_ID,
            "short_record_pure_padding_and_mask_primitive_implemented": True,
            "short_record_real_edf_transform_integrated": False,
            "short_record_masked_loss_integrated_in_existing_trainer": False,
            "existing_exploratory_runner_audit": {
                "accepts_new_cleanroom_train_manifest_natively": False,
                "retains_short_records_for_gradient": False,
                "supports_multiple_independent_base_seeds": False,
                "checkpoint_container_matches_formal_registry_safetensors": False,
                "may_be_used_as_formal_executor_without_adapter_changes": False,
            },
            "run_specs": run_specs,
            "transform_hash_bindings": {
                "architecture": config["model"]["architecture_source"],
                "registry": config["model"]["transform_registry"],
                "implementation": config["model"]["transform_implementation"],
                "streaming_ola": config["model"][
                    "streaming_ola_implementation"
                ],
            },
            "permissions": {
                "EDF_signal_opened": False,
                "source_train_target_sidecar_opened": False,
                "source_dev_artifact_opened": False,
                "source_dev_target_opened": False,
                "source_eval_opened": False,
                "CUDA_initialized": False,
                "training_started": False,
                "vLLM_stopped_or_mutated": False,
            },
            "claim_limits": {
                "complete_training_claim_authorized": False,
                "checkpoint_exists": False,
                "performance_estimated": False,
                "clinical_use_authorized": False,
            },
            "content_sha256": PENDING,
        }
    )
    return receipt, ledger


def _contains_target_bearing_fragment(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lower = str(key).lower()
            if (
                lower == "term"
                or "_term_" in lower
                or lower.startswith("term_")
                or lower.endswith("_term")
                or any(fragment in lower for fragment in _TARGET_BEARING_FRAGMENTS)
            ):
                return True
            if _contains_target_bearing_fragment(child):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_target_bearing_fragment(child) for child in value)
    elif isinstance(value, str):
        lower = value.lower()
        return (
            lower == "term"
            or "_term_" in lower
            or lower.startswith("term_")
            or lower.endswith("_term")
            or any(fragment in lower for fragment in _TARGET_BEARING_FRAGMENTS)
        )
    return False


def build_dev_prediction_dry_run(
    *,
    config_path: str | Path,
    dev_roster_path: str | Path,
) -> dict[str, Any]:
    """Build a dev prediction-first denominator from EEG-only rows."""

    config, roster = load_dev_roster_for_process(config_path, dev_roster_path)
    identity_rows = [
        {
            "analysis_identity_id": row["analysis_identity_id"],
            "source_row_sha256": row["row_sha256"],
            "expected_sample_count": row["sample_count"],
            "terminal_prediction_state": "pending_numeric_prediction",
        }
        for row in roster["records"]
    ]
    if _contains_target_bearing_fragment(identity_rows):
        raise PermissionError("dev prediction dry-run leaked a target-bearing value")
    identity_rows.sort(key=lambda row: row["analysis_identity_id"])
    return _content_address(
        {
            "schema_version": DEV_DRY_RUN_SCHEMA,
            "method_id": config["method_id"],
            "status": "ready_for_target_free_prediction_after_checkpoint_exists",
            "process_role": "source_dev_prediction_dry_run",
            "dataset_artifact_opened": "source_dev_eeg_only_prediction_roster_only",
            "source_dev_roster_path": str(Path(dev_roster_path).resolve(strict=True)),
            "source_dev_roster_file_sha256": file_sha256(dev_roster_path),
            "source_dev_roster_content_sha256": roster["content_sha256"],
            "expected_recording_count": len(identity_rows),
            "expected_patient_count": roster["inventory"]["patient_count"],
            "exactly_one_pending_terminal_state_per_record": True,
            "prediction_identity_roster_sha256": canonical_sha256(identity_rows),
            "target_bearing_field_or_value_count": 0,
            "prediction_rows_embedded": False,
            "checkpoint_required_and_not_loaded_by_dry_run": True,
            "permissions": {
                "EDF_signal_opened": False,
                "source_train_artifact_opened": False,
                "source_dev_target_path_resolved": False,
                "source_dev_target_opened": False,
                "source_eval_opened": False,
                "CUDA_initialized": False,
                "vLLM_stopped_or_mutated": False,
            },
            "content_sha256": PENDING,
        }
    )


def validate_checkpoint_binding_receipt(
    value: Mapping[str, Any], *, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail-close a future checkpoint sidecar before dev inference."""

    required = {
        "schema_version",
        "model_payload_path",
        "model_payload_sha256",
        "architecture_source_sha256",
        "transform_registry_sha256",
        "transform_implementation_sha256",
        "training_manifest_file_sha256",
        "training_manifest_content_sha256",
        "base_seed",
        "completed_epoch_count",
        "source_dev_target_open_count",
        "source_eval_open_count",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("checkpoint binding receipt fields drifted")
    receipt = deepcopy(dict(value))
    _validate_content_address(receipt, "checkpoint binding receipt")
    expected = {
        "architecture_source_sha256": config["model"]["architecture_source"][
            "file_sha256"
        ],
        "transform_registry_sha256": config["model"]["transform_registry"][
            "file_sha256"
        ],
        "transform_implementation_sha256": config["model"][
            "transform_implementation"
        ]["file_sha256"],
        "training_manifest_file_sha256": config["dataset_roles"]["training"][
            "file_sha256"
        ],
        "training_manifest_content_sha256": config["dataset_roles"]["training"][
            "content_sha256"
        ],
    }
    if any(receipt[key] != expected[key] for key in expected):
        raise PermissionError("checkpoint architecture/transform/train binding drifted")
    if receipt["base_seed"] not in config["training_protocol"][
        "independent_seed_base_values"
    ]:
        raise PermissionError("checkpoint seed is outside the frozen seed roster")
    if (
        not isinstance(receipt["completed_epoch_count"], int)
        or receipt["completed_epoch_count"] < 1
        or receipt["source_dev_target_open_count"] != 0
        or receipt["source_eval_open_count"] != 0
    ):
        raise PermissionError("checkpoint is incomplete or target isolation failed")
    payload_path = Path(receipt["model_payload_path"]).resolve(strict=True)
    if payload_path.is_symlink() or not payload_path.is_file():
        raise PermissionError("checkpoint payload must be a regular file")
    if file_sha256(payload_path) != receipt["model_payload_sha256"]:
        raise PermissionError("checkpoint payload byte hash drifted")
    return receipt


def audit_materialized_dry_runs(
    *,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Replay both dry-run receipts as a non-numeric, target-free auditor."""

    config = load_config(config_path)
    output = Path(output_dir).resolve(strict=True)
    if output.is_symlink() or not output.is_dir():
        raise PermissionError("dry-run output directory must be regular")
    paths = {
        "training_dry_run": output / "training_dry_run_receipt.json",
        "training_record_admission_ledger": output
        / "training_record_admission_ledger.json",
        "dev_prediction_dry_run": output / "dev_prediction_dry_run_receipt.json",
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for semantic, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise PermissionError(f"{semantic} artifact is missing or unsafe")
        value = json.loads(path.read_text(encoding="utf-8"))
        if type(value) is not dict:
            raise TypeError(f"{semantic} artifact must be an object")
        _validate_content_address(value, semantic)
        artifacts[semantic] = value
    training = artifacts["training_dry_run"]
    ledger = artifacts["training_record_admission_ledger"]
    dev = artifacts["dev_prediction_dry_run"]
    if (
        training.get("schema_version") != TRAIN_DRY_RUN_SCHEMA
        or ledger.get("schema_version") != ADMISSION_LEDGER_SCHEMA
        or dev.get("schema_version") != DEV_DRY_RUN_SCHEMA
        or training.get("method_id") != config["method_id"]
        or ledger.get("method_id") != config["method_id"]
        or dev.get("method_id") != config["method_id"]
    ):
        raise PermissionError("materialized dry-run schema or method drifted")
    if training["admission_ledger_content_sha256"] != ledger["content_sha256"]:
        raise PermissionError("training receipt/record ledger binding drifted")
    if (
        training["source_train_recording_count"] != 4664
        or ledger["recording_count"] != 4664
        or ledger["terminal_admission_state_count"] != 4664
        or dev["expected_recording_count"] != 1821
        or dev["target_bearing_field_or_value_count"] != 0
    ):
        raise PermissionError("formal-entry denominator drifted")
    if any(training["permissions"].values()) or any(dev["permissions"].values()):
        raise PermissionError("dry-run unexpectedly opened data or initialized compute")
    artifact_bindings = {
        semantic: {
            "path": str(path),
            "file_sha256": file_sha256(path),
            "content_sha256": artifacts[semantic]["content_sha256"],
        }
        for semantic, path in paths.items()
    }
    root = Path(__file__).resolve().parents[2]
    implementation_path = Path(__file__).resolve(strict=True)
    cli_path = root / "scripts/run_st16_common17_cleanroom_formal_v1.py"
    config_source = Path(config_path).resolve(strict=True)
    return _content_address(
        {
            "schema_version": AUDIT_RECEIPT_SCHEMA,
            "method_id": config["method_id"],
            "status": (
                "pass_dry_run_and_target_isolation_formal_gpu_training_"
                "remains_no_go"
            ),
            "auditor_role": "offline_non_numeric_receipt_replay",
            "config_binding": {
                "path": str(config_source),
                "file_sha256": file_sha256(config_source),
            },
            "implementation_bindings": {
                "formal_entry_module": {
                    "path": str(implementation_path),
                    "file_sha256": file_sha256(implementation_path),
                },
                "formal_entry_cli": {
                    "path": str(cli_path),
                    "file_sha256": file_sha256(cli_path),
                },
            },
            "artifact_bindings": artifact_bindings,
            "audit": {
                "source_train_terminal_admission_count": 4664,
                "source_train_planned_gradient_contribution_count": 4664,
                "source_train_actual_gradient_contribution_count": 0,
                "short_context_arm_planned_count": training[
                    "short_context_arm_record_count"
                ],
                "source_dev_EEG_only_roster_count": 1821,
                "source_dev_target_bearing_field_or_value_count": 0,
                "source_dev_target_open_count": 0,
                "source_eval_open_count": 0,
                "CUDA_initialized": False,
                "training_started": False,
                "vLLM_stopped_or_mutated": False,
            },
            "formal_launch_gate": {
                "status": "no_go",
                "reason": (
                    "short-record reflect-context transform and valid-support "
                    "masked loss are not integrated/replayed on real EDF training"
                ),
                "may_call_4664_records_actual_gradient_contributors": False,
            },
            "claim_limits": {
                "dry_run_only": True,
                "checkpoint_exists": False,
                "performance_estimated": False,
                "clinical_use_authorized": False,
            },
            "content_sha256": PENDING,
        }
    )


__all__ = [
    "ADMISSION_LEDGER_SCHEMA",
    "AUDIT_RECEIPT_SCHEMA",
    "DEV_DRY_RUN_SCHEMA",
    "DEV_ROSTER_SCHEMA",
    "SCHEMA_VERSION",
    "SHORT_POLICY_ID",
    "TILE_SAMPLES",
    "TRAIN_DRY_RUN_SCHEMA",
    "TRAIN_MANIFEST_SCHEMA",
    "build_dev_prediction_dry_run",
    "build_training_dry_run",
    "audit_materialized_dry_runs",
    "canonical_sha256",
    "deterministic_reflect_context_pad",
    "file_sha256",
    "load_config",
    "load_dev_roster_for_process",
    "load_training_manifest_for_process",
    "masked_dense_binary_cross_entropy",
    "provider_target_sample_count",
    "validate_checkpoint_binding_receipt",
]
