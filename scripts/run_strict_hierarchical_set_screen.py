#!/usr/bin/env python3
"""Plan and execute the locked, test-blind hierarchical-set validation screen.

The default ``plan`` mode is non-executing.  It validates the immutable data
contract, writes the exact command matrix and reports a conservative runtime
estimate.  ``smoke`` executes one one-epoch protocol check, while ``screen``
executes the complete predeclared matrix.  Every execution uses the same fixed
28 training patients and the union of the three locked five-patient validation
panels.  Panel-specific checkpoint selection is deliberately impossible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = Path("outputs/cpbf_hierarchical_set_20260717/protocol/LOCKED_PROTOCOL.json")
DEFAULT_PREPROCESSED = Path("outputs/tfm_soz/private_0622_fix_regiongt_rows119_segments_15s")
DEFAULT_SCREENING_ROOT = Path("outputs/cpbf_hierarchical_set_20260717/screening")
TRAINER = Path("code/tfm_soz/train_private_soz_segments.py")
GATE_SCRIPT = Path("scripts/gate_strict_hierarchical_set_screen.py")
MODEL = Path("code/tfm_soz/model.py")
CONSTANTS = Path("code/tfm_soz/constants.py")
DATASET = Path("code/tfm_soz/dataset.py")

EXPECTED_MANIFEST_BASENAME = "private_sz_union_relabel_manifest_0622_fix_region_annotation.csv"
EXPECTED_MANIFEST_SHA256 = "e2adde43fa40cb1dc79feef26685c5f0a209f3b931058d201250681e70e40cae"
EXPECTED_INDEX_SHA256 = "a042313c89d98229f4d572a13a99392b4ec70a74294fc72a7b9974f214eda43c"
EXPECTED_SUMMARY_SHA256 = "a32a0601ee8cd2cd1d122f47ebb50a9484de9992af961bbd39ae8b6e76ff4c04"
EXPECTED_PROTOCOL_LINEAGE_ARTIFACTS = (
    "manifest_audit.json",
    "manifest_lineage.json",
    "validation_panels.json",
    "validation_panel_assignments.csv",
)
EXPECTED_SEEDS = (2058, 2059, 2060)
EXPECTED_PANEL_SEED = 20260717

EXPECTED_PANELS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    (
        ("validation_panel_A", ("刘娟", "吴斯龙", "江仁坤", "赖冬微", "黄亦奇")),
        ("validation_panel_B", ("彭文", "曾静君", "杜克华", "高萌", "黄邦洲")),
        ("validation_panel_C", ("杨祚明", "王用琼", "确干", "郑震宇", "龙凤")),
    )
)
EXPECTED_PANEL_EVENT_COUNTS = {name: 14 for name in EXPECTED_PANELS}
EXPECTED_TRAIN_PATIENTS = (
    "刘定治",
    "吴卫东",
    "周良贵",
    "庄芷端",
    "廖佳",
    "徐曼舒",
    "朱涵栖",
    "李伟恺",
    "李杰",
    "杨朵",
    "杨梦超",
    "查吉秀",
    "梁晓嘉",
    "江文杰",
    "汪云飞",
    "洪莲",
    "苏永源",
    "蒋璐",
    "薛少林",
    "郑剑敏",
    "陈健豪",
    "陈妙玲",
    "陈芳",
    "韩雪",
    "高资波",
    "黄建和",
    "黄荷霞",
    "龙娇",
)

EXPECTED_METRIC_ESTIMANDS: dict[str, Any] = {
    "channel_top1": "exact raw available-channel argmax hit in event soz_bipolar set",
    "region_top1": "exact raw five-class softmax argmax equals explicit soz_region",
    "mrr": "reciprocal of first-positive rank",
    "channel_auprc": {
        "averaging": "micro",
        "flattening": "event_by_available_channel",
        "unavailable_channels": "excluded",
    },
    "region_auprc": {
        "averaging": "micro",
        "flattening": "event_by_five_class_softmax",
    },
    "seed_aggregation": "unweighted arithmetic mean",
    "panel_aggregation": "event-pooled within locked panel, then unweighted seed mean",
    "patient_macro_aggregation": "compute per patient first, then equal patient mean",
}

# These values are part of the screening hypothesis, not tunable CLI knobs.
TKPR_K = 2
TKPR_WEIGHT = 0.25
RANK_MARGIN_WEIGHT = 0.25
RANK_MARGIN = 0.20
SET_BOTTLENECK = 16
SET_RESIDUAL_INIT = 0.0
EXPECTED_SET_ADDED_CLASSIFIER_PARAMETERS = 2707
BASE_TOKENIZER_EPOCHS = 16
BASE_CLASSIFIER_EPOCHS = 90
ARM_CLASSIFIER_EPOCHS = 12
EMPIRICAL_BASE_ONE_PLUS_ONE_SECONDS = 23.6
EMPIRICAL_FROZEN_ARM_ONE_EPOCH_SECONDS = 8.0
CONSERVATIVE_TOTAL_GPU_HOURS_UPPER = 2.0

CONFIG_SPECS: "OrderedDict[str, dict[str, Any]]" = OrderedDict(
    (
        (
            "tfm_soz",
            {
                "role": "locked TFM-SOZ control",
                "stage": "clean_base",
                "training_objective": "multitask",
                "channel_top1_margin_weight": RANK_MARGIN_WEIGHT,
                "channel_tkpr_weight": 0.0,
                "use_region_channel_set_head": False,
                "region_channel_set_mapping_policy": "canonical",
                "initialization": "fresh",
                "freeze_base_channel_head_only": False,
            },
        ),
        (
            "head_only",
            {
                "role": "rank-aware Head-only control",
                "stage": "matched_frozen_arm",
                "training_objective": "top1_joint",
                "channel_top1_margin_weight": RANK_MARGIN_WEIGHT,
                "channel_tkpr_weight": 0.0,
                "use_region_channel_set_head": False,
                "region_channel_set_mapping_policy": "canonical",
                "initialization": "same_seed_tfm_soz",
                "freeze_base_channel_head_only": True,
            },
        ),
        (
            "tkpr_only",
            {
                "role": "loss-only ablation",
                "stage": "matched_frozen_arm",
                "training_objective": "top1_joint",
                "channel_top1_margin_weight": RANK_MARGIN_WEIGHT,
                "channel_tkpr_weight": TKPR_WEIGHT,
                "use_region_channel_set_head": False,
                "region_channel_set_mapping_policy": "canonical",
                "initialization": "same_seed_tfm_soz",
                "freeze_base_channel_head_only": True,
            },
        ),
        (
            "set_only",
            {
                "role": "core-module-only ablation",
                "stage": "matched_frozen_arm",
                "training_objective": "top1_joint",
                "channel_top1_margin_weight": RANK_MARGIN_WEIGHT,
                "channel_tkpr_weight": 0.0,
                "use_region_channel_set_head": True,
                "region_channel_set_mapping_policy": "canonical",
                "initialization": "same_seed_tfm_soz",
                "freeze_base_channel_head_only": True,
            },
        ),
        (
            "full",
            {
                "role": "region-channel set residual plus TKPR candidate",
                "stage": "matched_frozen_arm",
                "training_objective": "top1_joint",
                "channel_top1_margin_weight": RANK_MARGIN_WEIGHT,
                "channel_tkpr_weight": TKPR_WEIGHT,
                "use_region_channel_set_head": True,
                "region_channel_set_mapping_policy": "canonical",
                "initialization": "same_seed_tfm_soz",
                "freeze_base_channel_head_only": True,
            },
        ),
        (
            "cyclic_permuted",
            {
                "role": "parameter-matched wrong-anatomy control",
                "stage": "matched_frozen_arm",
                "training_objective": "top1_joint",
                "channel_top1_margin_weight": RANK_MARGIN_WEIGHT,
                "channel_tkpr_weight": TKPR_WEIGHT,
                "use_region_channel_set_head": True,
                "region_channel_set_mapping_policy": "cyclic_permuted",
                "initialization": "same_seed_tfm_soz",
                "freeze_base_channel_head_only": True,
            },
        ),
    )
)

REQUIRED_TRAINER_FLAGS = (
    "--train-patients",
    "--val-patients",
    "--test-fraction",
    "--skip-test-eval",
    "--channel-tkpr-weight",
    "--channel-tkpr-k",
    "--use-region-channel-set-head",
    "--region-channel-set-bottleneck",
    "--region-channel-set-mapping-policy",
    "--region-channel-set-residual-init",
    "--use-region-attention-pooling",
    "--use-region-embedding-head",
    "--init-run-dir",
    "--freeze-base-channel-head-only",
)

REQUIRED_RECEIPT_ARTIFACTS = (
    "val_predictions.csv",
    "metrics.json",
    "run_config.json",
    "split_summary.json",
    "classifier_history.csv",
    "tfm_tokenizer_segments.pth",
    "tfm_soz_segment_classifier.pth",
)
OPTIONAL_RECEIPT_ARTIFACTS = ("train_predictions.csv",)


class ContractError(RuntimeError):
    """Raised when a locked screening invariant is not satisfied."""


def resolve_under_repo(path: Path | str, repo_root: Path = REPO_ROOT) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def relative_to_repo(path: Path, repo_root: Path = REPO_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def require_no_symlink_chain(path: Path, *, anchor: Path) -> None:
    """Reject symlinks and lexical traversal from ``anchor`` through ``path``."""

    anchor_absolute = anchor.absolute()
    path_absolute = path.absolute()
    try:
        relative = path_absolute.relative_to(anchor_absolute)
    except ValueError as exc:
        raise ContractError(f"Path is outside its required anchor: {path}") from exc
    if ".." in relative.parts:
        raise ContractError(f"Path contains forbidden parent traversal: {path}")
    if anchor_absolute.is_symlink():
        raise ContractError(f"Required path anchor is a symlink: {anchor_absolute}")
    current = anchor_absolute
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"Symlink is forbidden in strict screening paths: {current}")


def require_no_symlinks_in_run(run_dir: Path) -> None:
    require_no_symlink_chain(run_dir, anchor=REPO_ROOT)
    if not run_dir.exists():
        return
    for path in run_dir.rglob("*"):
        if path.is_symlink():
            raise ContractError(f"Symlink is forbidden inside a screening run: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_jsonable(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"Required JSON does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"Expected a JSON object: {path}")
    return value


def require_equal(label: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise ContractError(f"{label} mismatch: observed={observed!r}, expected={expected!r}")


def require_true(label: str, observed: object) -> None:
    if observed is not True:
        raise ContractError(f"{label} must be true, observed={observed!r}")


def pooled_validation_patients() -> tuple[str, ...]:
    return tuple(patient for patients in EXPECTED_PANELS.values() for patient in patients)


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    data = protocol.get("data_contract", {})
    splits = protocol.get("development_splits", {})
    random_seeds = protocol.get("random_seeds", {})
    gate = protocol.get("validation_promotion_gate", {})
    safety = protocol.get("execution_safety", {})
    if not all(isinstance(item, Mapping) for item in (data, splits, random_seeds, gate, safety)):
        raise ContractError("LOCKED_PROTOCOL is missing required mapping sections")

    require_equal("manifest basename", data.get("manifest_basename"), EXPECTED_MANIFEST_BASENAME)
    require_equal("manifest SHA256", data.get("manifest_sha256"), EXPECTED_MANIFEST_SHA256)
    require_equal("manifest event count", data.get("expected_events"), 119)
    require_equal("manifest patient count", data.get("expected_patients"), 43)
    require_equal("channel ground truth", data.get("channel_ground_truth"), "soz_bipolar")
    require_equal("region ground truth", data.get("region_ground_truth"), "soz_region")
    require_true(
        "independent explicit region ground truth",
        data.get("region_ground_truth_is_independent_of_channel_ground_truth"),
    )
    forbidden = set(data.get("forbidden_model_input_columns", []))
    for column in ("onset_channels", "soz_bipolar", "soz_region"):
        if column not in forbidden:
            raise ContractError(f"Forbidden input column is absent from the lock: {column}")

    locked_panels = splits.get("panels", {})
    require_equal(
        "validation panels",
        {name: list(locked_panels.get(name, [])) for name in EXPECTED_PANELS},
        {name: list(patients) for name, patients in EXPECTED_PANELS.items()},
    )
    require_equal("validation patient count", splits.get("validation_patient_count"), 15)
    require_equal("fixed train patients", splits.get("remaining_train_patients"), list(EXPECTED_TRAIN_PATIENTS))
    require_equal("fixed train patient count", splits.get("remaining_train_patient_count"), 28)
    require_true("fixed patient roles", splits.get("patient_roles_are_fixed_during_screening"))
    require_true("validation patients excluded from training", splits.get("validation_patients_never_enter_training_during_screening"))
    require_true("test-blind screening", splits.get("test_blind_screening"))
    require_true("skip test evaluation", splits.get("skip_test_eval"))
    require_true("no test artifacts", splits.get("test_artifacts_must_not_be_constructed_during_screening"))

    require_equal("model seeds", random_seeds.get("model_seeds"), list(EXPECTED_SEEDS))
    require_equal("panel seed", random_seeds.get("panel_seed"), EXPECTED_PANEL_SEED)
    require_equal("promotion controls", gate.get("controls"), ["TFM-SOZ", "rank-aware Head-only"])
    noninferiority = gate.get("noninferiority", {})
    meaningful = gate.get("meaningful_improvement", {})
    stability = gate.get("secondary_stability", {})
    directions = gate.get("directional_consistency", {})
    budget = gate.get("parameter_budget", {})
    ablation = gate.get("ablation_independence", {})
    hierarchy_consistency = gate.get("hierarchy_consistency", {})
    require_true("all promotion conditions required", gate.get("all_conditions_required"))
    require_equal("noninferiority bound", noninferiority.get("max_absolute_drop"), 0.02)
    require_equal("meaningful gain", meaningful.get("min_absolute_gain"), 0.02)
    require_true(
        "same primary endpoint meaningful gain across controls",
        meaningful.get("same_primary_endpoint_must_meet_gain_against_each_control"),
    )
    require_equal("MRR stability bound", stability.get("max_mean_mrr_drop"), 0.01)
    require_equal("AUPRC stability bound", stability.get("max_mean_auprc_drop"), 0.01)
    require_equal("minimum nonnegative panels", directions.get("minimum_nonnegative_validation_panels"), 2)
    require_equal("minimum nonnegative seeds", directions.get("minimum_nonnegative_model_seeds"), 2)
    require_equal("worst panel floor", directions.get("worst_panel_primary_endpoint_drop_floor"), -0.05)
    require_equal(
        "ablation primary gain",
        ablation.get("core_module_min_gain_on_one_primary_endpoint"),
        0.01,
    )
    require_equal(
        "ablation other-endpoint drop",
        ablation.get("core_module_max_drop_on_other_primary_endpoint"),
        0.01,
    )
    require_equal(
        "hierarchy consistency gain vs rank-aware Head-only",
        hierarchy_consistency.get("minimum_gain_vs_rank_aware_head_only"),
        0.02,
    )
    require_equal(
        "hierarchy consistency gain vs TFM-SOZ",
        hierarchy_consistency.get("minimum_gain_vs_tfm_soz"),
        0.0,
    )
    require_equal("parameter budget", budget.get("maximum_added_trainable_fraction_of_locked_base_classifier"), 0.1)
    require_true("runner skip-test safety", safety.get("screening_runner_must_set_skip_test_eval"))
    require_equal("architecture iteration cap", protocol.get("search_limits", {}).get("maximum_architecture_iterations"), 2)
    require_equal("metric estimands", protocol.get("metric_estimands"), EXPECTED_METRIC_ESTIMANDS)


def read_index_patients(index_path: Path) -> tuple[set[str], int]:
    with index_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    patients = {str(row.get("base_patient_id", "")).strip() for row in rows}
    patients.discard("")
    return patients, len(rows)


def preprocessed_sample_hash_lock(preprocessed_dir: Path, index_path: Path) -> dict[str, Any]:
    """Hash all indexed NPZ samples with their locked relative paths."""

    with index_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    path_hashes: dict[str, str] = {}
    root = preprocessed_dir.resolve()
    for row in rows:
        relative_text = str(row.get("npz_path", "")).strip().replace("\\", "/")
        relative = PurePosixPath(relative_text)
        if not relative_text or relative.is_absolute() or ".." in relative.parts:
            raise ContractError(f"Unsafe indexed NPZ path: {relative_text!r}")
        if relative_text in path_hashes:
            raise ContractError(f"Duplicate indexed NPZ path: {relative_text}")
        lexical_sample_path = preprocessed_dir / Path(*relative.parts)
        require_no_symlink_chain(lexical_sample_path, anchor=preprocessed_dir)
        sample_path = lexical_sample_path.resolve()
        try:
            sample_path.relative_to(root)
        except ValueError as exc:
            raise ContractError(f"Indexed NPZ escapes the preprocessed directory: {relative_text}") from exc
        if not sample_path.is_file():
            raise ContractError(f"Indexed NPZ is missing: {sample_path}")
        path_hashes[relative_text] = sha256_file(sample_path)
    ordered = {path: path_hashes[path] for path in sorted(path_hashes)}
    return {
        "count": len(ordered),
        "aggregate_sha256": sha256_jsonable(ordered),
    }


def protocol_lineage_artifact_hash_lock(
    protocol: Mapping[str, Any],
    protocol_path: Path,
) -> dict[str, str]:
    """Verify every lineage artifact declared by the locked protocol on disk."""

    lineage = protocol.get("lineage", {})
    if not isinstance(lineage, Mapping):
        raise ContractError("LOCKED_PROTOCOL lineage must be a mapping")
    declared = lineage.get("artifact_hashes", {})
    if not isinstance(declared, Mapping):
        raise ContractError("LOCKED_PROTOCOL lineage.artifact_hashes must be a mapping")
    require_equal(
        "protocol lineage artifact names",
        set(declared),
        set(EXPECTED_PROTOCOL_LINEAGE_ARTIFACTS),
    )
    protocol_dir = protocol_path.parent
    observed: dict[str, str] = {}
    for name in EXPECTED_PROTOCOL_LINEAGE_ARTIFACTS:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise ContractError(f"Unsafe protocol lineage artifact path: {name!r}")
        artifact_path = protocol_dir / name
        require_no_symlink_chain(artifact_path, anchor=protocol_dir)
        if not artifact_path.is_file():
            raise ContractError(f"Required protocol lineage artifact does not exist: {artifact_path}")
        digest = sha256_file(artifact_path)
        require_equal(f"protocol lineage artifact SHA256 for {name}", digest, declared.get(name))
        observed[name] = digest
    return observed


def validate_locked_inputs(
    *,
    repo_root: Path,
    protocol_path: Path,
    preprocessed_dir: Path,
    trainer_path: Path,
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    validate_protocol(protocol)
    lineage_artifact_hashes = protocol_lineage_artifact_hash_lock(protocol, protocol_path)

    manifest_path = repo_root / EXPECTED_MANIFEST_BASENAME
    index_path = preprocessed_dir / "index.csv"
    summary_path = preprocessed_dir / "preprocess_summary.json"
    source_paths = (
        trainer_path,
        resolve_under_repo(MODEL, repo_root),
        resolve_under_repo(CONSTANTS, repo_root),
        resolve_under_repo(DATASET, repo_root),
        Path(__file__).resolve(),
        resolve_under_repo(GATE_SCRIPT, repo_root),
    )
    for path in (manifest_path, index_path, summary_path, *source_paths):
        if not path.is_file():
            raise ContractError(f"Required locked input does not exist: {path}")

    require_equal("manifest file basename", manifest_path.name, EXPECTED_MANIFEST_BASENAME)
    require_equal("manifest file SHA256", sha256_file(manifest_path), EXPECTED_MANIFEST_SHA256)
    require_equal("preprocessed index SHA256", sha256_file(index_path), EXPECTED_INDEX_SHA256)
    require_equal("preprocess summary SHA256", sha256_file(summary_path), EXPECTED_SUMMARY_SHA256)

    summary = load_json(summary_path)
    require_equal("summary manifest basename", Path(str(summary.get("manifest", ""))).name, EXPECTED_MANIFEST_BASENAME)
    require_equal("summary manifest SHA256", summary.get("manifest_sha256"), EXPECTED_MANIFEST_SHA256)
    require_equal("summary channel label source", summary.get("channel_label_source"), "soz_bipolar")
    require_equal("summary region label source", summary.get("region_label_source"), "soz_region")
    require_equal("summary onset_channels policy", summary.get("onset_channels"), "ignored")
    require_equal("summary rows total", summary.get("rows_total"), 119)
    require_equal("summary rows written", summary.get("rows_written"), 119)
    require_equal("summary rows failed", summary.get("rows_failed"), 0)
    require_equal("summary 15-second window", summary.get("total_sec"), 15.0)

    index_patients, index_rows = read_index_patients(index_path)
    sample_lock = preprocessed_sample_hash_lock(preprocessed_dir, index_path)
    expected_patients = set(EXPECTED_TRAIN_PATIENTS) | set(pooled_validation_patients())
    require_equal("index row count", index_rows, 119)
    require_equal("index patient roles", index_patients, expected_patients)
    require_equal("index patient count", len(index_patients), 43)
    require_equal("indexed NPZ sample count", sample_lock["count"], 119)

    trainer_text = trainer_path.read_text(encoding="utf-8")
    missing_flags = [flag for flag in REQUIRED_TRAINER_FLAGS if flag not in trainer_text]
    source_hashes = {
        relative_to_repo(path, repo_root): sha256_file(path)
        for path in source_paths
    }
    protocol_sha256 = sha256_file(protocol_path)
    code_lock_sha256 = sha256_jsonable(source_hashes)
    execution_lock_sha256 = sha256_jsonable(
        {
            "protocol_sha256": protocol_sha256,
            "source_hashes": source_hashes,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "index_sha256": EXPECTED_INDEX_SHA256,
            "preprocess_summary_sha256": EXPECTED_SUMMARY_SHA256,
            "preprocessed_samples": sample_lock,
            "protocol_lineage_artifact_hashes": lineage_artifact_hashes,
        }
    )
    return {
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "index_sha256": EXPECTED_INDEX_SHA256,
        "preprocess_summary_sha256": EXPECTED_SUMMARY_SHA256,
        "preprocessed_samples": sample_lock,
        "protocol_lineage_artifact_hashes": lineage_artifact_hashes,
        "trainer_sha256": sha256_file(trainer_path),
        "source_hashes": source_hashes,
        "code_lock_sha256": code_lock_sha256,
        "execution_lock_sha256": execution_lock_sha256,
        "trainer_contract_ready": not missing_flags,
        "missing_trainer_flags": missing_flags,
        "index_rows": index_rows,
        "index_patients": len(index_patients),
    }


def config_cli(spec: Mapping[str, Any]) -> list[str]:
    args = [
        "--training-objective",
        str(spec["training_objective"]),
        "--channel-top1-margin-weight",
        str(spec["channel_top1_margin_weight"]),
        "--top1-margin",
        str(RANK_MARGIN),
        "--channel-tkpr-weight",
        str(spec["channel_tkpr_weight"]),
        "--channel-tkpr-k",
        str(TKPR_K),
        "--region-channel-set-bottleneck",
        str(SET_BOTTLENECK),
        "--region-channel-set-mapping-policy",
        str(spec["region_channel_set_mapping_policy"]),
        "--region-channel-set-residual-init",
        str(SET_RESIDUAL_INIT),
    ]
    if bool(spec["use_region_channel_set_head"]):
        args.append("--use-region-channel-set-head")
    return args


def build_command(
    *,
    trainer_path: Path,
    preprocessed_dir: Path,
    run_dir: Path,
    base_run_dir: Path,
    config_name: str,
    seed: int,
    spec: Mapping[str, Any],
    tokenizer_epochs: int,
    classifier_epochs: int,
    batch_size: int,
    num_workers: int,
    device: str,
    repo_root: Path,
) -> list[str]:
    train_csv = ",".join(EXPECTED_TRAIN_PATIENTS)
    val_csv = ",".join(pooled_validation_patients())
    command = [
        "rtk",
        "python3",
        "-u",
        relative_to_repo(trainer_path, repo_root),
        "--preprocessed-dir",
        relative_to_repo(preprocessed_dir, repo_root),
        "--output-dir",
        relative_to_repo(run_dir, repo_root),
        "--seed",
        str(seed),
        "--split-seed",
        str(EXPECTED_PANEL_SEED),
        "--train-patients",
        train_csv,
        "--val-patients",
        val_csv,
        "--val-fraction",
        "0",
        "--test-fraction",
        "0",
        "--skip-test-eval",
        "--tokenizer-epochs",
        str(tokenizer_epochs),
        "--classifier-epochs",
        str(classifier_epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--device",
        str(device),
        "--quiet-summary",
        "--use-region-attention-pooling",
        "--use-region-embedding-head",
    ]
    if config_name != "tfm_soz":
        command.extend(
            (
                "--init-run-dir",
                relative_to_repo(base_run_dir, repo_root),
                "--freeze-base-channel-head-only",
            )
        )
    command.extend(config_cli(spec))
    return command


def gate_policy(protocol: Mapping[str, Any]) -> dict[str, Any]:
    locked = protocol["validation_promotion_gate"]
    consistency = locked["hierarchy_consistency"]
    return {
        "all_conditions_required": True,
        "candidate": "full",
        "primary_controls": ["tfm_soz", "head_only"],
        "primary_endpoints": ["channel_top1", "region_top1"],
        "noninferiority_max_absolute_drop": locked["noninferiority"]["max_absolute_drop"],
        "meaningful_min_absolute_gain": locked["meaningful_improvement"]["min_absolute_gain"],
        "meaningful_gain_required_against_each_primary_control": True,
        "same_primary_endpoint_must_meet_gain_against_each_control": locked["meaningful_improvement"][
            "same_primary_endpoint_must_meet_gain_against_each_control"
        ],
        "secondary_endpoints": ["channel_mrr", "region_mrr", "channel_auprc", "region_auprc"],
        "secondary_max_mean_drop": {
            "mrr": locked["secondary_stability"]["max_mean_mrr_drop"],
            "auprc": locked["secondary_stability"]["max_mean_auprc_drop"],
        },
        "minimum_nonnegative_panels_per_primary_control_and_endpoint": locked["directional_consistency"][
            "minimum_nonnegative_validation_panels"
        ],
        "minimum_nonnegative_seeds_per_primary_control_and_endpoint": locked["directional_consistency"][
            "minimum_nonnegative_model_seeds"
        ],
        "worst_panel_primary_delta_floor": locked["directional_consistency"][
            "worst_panel_primary_endpoint_drop_floor"
        ],
        "ablation_controls": ["head_only", "tkpr_only", "set_only", "cyclic_permuted"],
        "full_vs_each_ablation_min_gain_on_one_primary": locked["ablation_independence"][
            "core_module_min_gain_on_one_primary_endpoint"
        ],
        "full_vs_each_ablation_max_drop_on_other_primary": locked["ablation_independence"][
            "core_module_max_drop_on_other_primary_endpoint"
        ],
        "hierarchy_consistency": {
            "min_gain_vs_head_only": consistency["minimum_gain_vs_rank_aware_head_only"],
            "min_gain_vs_tfm_soz": consistency["minimum_gain_vs_tfm_soz"],
        },
        "maximum_added_trainable_fraction": locked["parameter_budget"][
            "maximum_added_trainable_fraction_of_locked_base_classifier"
        ],
        "test_artifact_or_test_metric_tolerance": 0,
        "metric_estimands": protocol["metric_estimands"],
    }


def build_plan(
    *,
    mode: str,
    repo_root: Path,
    protocol_path: Path,
    preprocessed_dir: Path,
    screening_root: Path,
    trainer_path: Path,
    audit: Mapping[str, Any],
    tokenizer_epochs: int,
    classifier_epochs: int,
    arm_classifier_epochs: int,
    batch_size: int,
    num_workers: int,
    device: str,
    conservative_total_gpu_hours_upper: float,
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    is_smoke = mode == "smoke"
    selected_configs: Sequence[str] = ("tfm_soz", "full") if is_smoke else tuple(CONFIG_SPECS)
    selected_seeds: Sequence[int] = (EXPECTED_SEEDS[0],) if is_smoke else EXPECTED_SEEDS
    run_root = screening_root / ("smoke_runs" if is_smoke else "runs")
    entries: list[dict[str, Any]] = []
    # Dependency order is part of the protocol: finish a clean base for one
    # seed before constructing any matched arm that consumes its checkpoints.
    for seed in selected_seeds:
        base_run_dir = run_root / "tfm_soz" / f"seed_{seed}"
        for config_name in selected_configs:
            spec = CONFIG_SPECS[config_name]
            run_dir = run_root / config_name / f"seed_{seed}"
            if is_smoke:
                effective_tokenizer_epochs = 1 if config_name == "tfm_soz" else 0
                effective_classifier_epochs = 1
            else:
                effective_tokenizer_epochs = tokenizer_epochs if config_name == "tfm_soz" else 0
                effective_classifier_epochs = (
                    classifier_epochs if config_name == "tfm_soz" else arm_classifier_epochs
                )
            command = build_command(
                trainer_path=trainer_path,
                preprocessed_dir=preprocessed_dir,
                run_dir=run_dir,
                base_run_dir=base_run_dir,
                config_name=config_name,
                seed=seed,
                spec=spec,
                tokenizer_epochs=effective_tokenizer_epochs,
                classifier_epochs=effective_classifier_epochs,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                repo_root=repo_root,
            )
            entries.append(
                {
                    "config": config_name,
                    "seed": seed,
                    "role": spec["role"],
                    "run_dir": relative_to_repo(run_dir, repo_root),
                    "depends_on": None
                    if config_name == "tfm_soz"
                    else relative_to_repo(base_run_dir, repo_root),
                    "resolved_spec": dict(spec),
                    "tokenizer_epochs": effective_tokenizer_epochs,
                    "classifier_epochs": effective_classifier_epochs,
                    "command": command,
                    "command_shell": shlex.join(command),
                    "command_sha256": sha256_jsonable(command),
                    "code_lock_sha256": audit["code_lock_sha256"],
                    "execution_lock_sha256": audit["execution_lock_sha256"],
                }
            )

    base_count = sum(entry["config"] == "tfm_soz" for entry in entries)
    arm_count = len(entries) - base_count
    if is_smoke:
        projected_seconds = EMPIRICAL_BASE_ONE_PLUS_ONE_SECONDS + EMPIRICAL_FROZEN_ARM_ONE_EPOCH_SECONDS
    else:
        base_scale = (tokenizer_epochs + classifier_epochs) / 2.0
        projected_seconds = (
            base_count * EMPIRICAL_BASE_ONE_PLUS_ONE_SECONDS * base_scale
            + arm_count * EMPIRICAL_FROZEN_ARM_ONE_EPOCH_SECONDS * arm_classifier_epochs
        )
    gate_command = [
        "rtk",
        "python3",
        relative_to_repo(resolve_under_repo(GATE_SCRIPT, repo_root), repo_root),
        "--screening-root",
        relative_to_repo(screening_root, repo_root),
        "--protocol",
        relative_to_repo(protocol_path, repo_root),
        "--plan",
        relative_to_repo(screening_root / "SCREENING_PLAN.json", repo_root),
    ]
    return {
        "schema_version": "strict_hierarchical_set_screening_plan_v1",
        "non_executing_default": True,
        "mode": "smoke" if is_smoke else "screen",
        "execution_ready": bool(audit.get("trainer_contract_ready")),
        "blocked_reasons": [
            f"trainer missing required flags: {audit.get('missing_trainer_flags')}"
        ]
        if audit.get("missing_trainer_flags")
        else [],
        "protocol": {
            "path": relative_to_repo(protocol_path, repo_root),
            "sha256": audit["protocol_sha256"],
            "manifest_basename": EXPECTED_MANIFEST_BASENAME,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "preprocessed_dir": relative_to_repo(preprocessed_dir, repo_root),
            "index_sha256": EXPECTED_INDEX_SHA256,
            "preprocess_summary_sha256": EXPECTED_SUMMARY_SHA256,
            "channel_label_source": "soz_bipolar",
            "region_label_source": "soz_region",
            "onset_channels": "ignored",
            "test_fraction": 0.0,
            "skip_test_eval": True,
        },
        "split": {
            "checkpoint_selection": "one pooled validation checkpoint per config/seed",
            "panel_specific_checkpoint_selection": False,
            "train_only_patients": list(EXPECTED_TRAIN_PATIENTS),
            "train_only_patient_count": 28,
            "pooled_validation_patients": list(pooled_validation_patients()),
            "pooled_validation_patient_count": 15,
            "panels": {name: list(patients) for name, patients in EXPECTED_PANELS.items()},
            "expected_panel_events": EXPECTED_PANEL_EVENT_COUNTS,
            "panel_use": "gate stratification of the same pooled val_predictions.csv only",
            "test_patients": [],
            "excluded_patients": [],
        },
        "hypothesis": {
            "core_module": "one lightweight zero-initialized region-channel set residual",
            "objective": "K=2 Top-K pairwise channel ranking surrogate",
            "tkpr_k": TKPR_K,
            "tkpr_weight": TKPR_WEIGHT,
            "rank_aware_head_margin_weight": RANK_MARGIN_WEIGHT,
            "rank_margin": RANK_MARGIN,
            "set_bottleneck": SET_BOTTLENECK,
            "set_residual_init": SET_RESIDUAL_INIT,
            "expected_set_added_classifier_parameters": EXPECTED_SET_ADDED_CLASSIFIER_PARAMETERS,
            "architecture_iterations_used_by_this_plan": 1,
            "locked_maximum_architecture_iterations": protocol["search_limits"]["maximum_architecture_iterations"],
        },
        "configurations": {name: dict(spec) for name, spec in CONFIG_SPECS.items()},
        "seeds": list(EXPECTED_SEEDS),
        "training": {
            "clean_base_initialization_per_seed": True,
            "matched_arm_initialization": "same-seed tfm_soz checkpoints produced inside this screening root",
            "historical_or_external_checkpoint_initialization": False,
            "base_tokenizer_epochs": 1 if is_smoke else tokenizer_epochs,
            "base_classifier_epochs": 1 if is_smoke else classifier_epochs,
            "arm_tokenizer_epochs": 0,
            "arm_classifier_epochs": 1 if is_smoke else arm_classifier_epochs,
            "freeze_base_channel_head_only_for_all_arms": True,
            "region_attention_pooling_for_all_configs": True,
            "region_embedding_head_for_all_configs": True,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "device": device,
            "pooled_validation_checkpoint_selection_is_predeclared": True,
        },
        "promotion_gate": gate_policy(protocol),
        "statistics_after_gate": protocol.get("statistics", {}),
        "runtime_estimate": {
            "kind": "empirical linear projection plus predeclared conservative upper bound",
            "runs": len(entries),
            "clean_base_runs": base_count,
            "matched_frozen_arm_runs": arm_count,
            "measurements_seconds": {
                "clean_base_one_tokenizer_plus_one_classifier_epoch": EMPIRICAL_BASE_ONE_PLUS_ONE_SECONDS,
                "matched_frozen_arm_one_classifier_epoch": EMPIRICAL_FROZEN_ARM_ONE_EPOCH_SECONDS,
            },
            "projection_formula": "base_runs*23.6*((base_tokenizer_epochs+base_classifier_epochs)/2) + arm_runs*8*arm_classifier_epochs",
            "empirical_projected_total_seconds": projected_seconds,
            "empirical_projected_total_minutes": projected_seconds / 60.0,
            "empirical_projected_total_gpu_hours": projected_seconds / 3600.0,
            "conservative_total_gpu_hours_lower": 1.0 if not is_smoke else projected_seconds / 3600.0,
            "conservative_total_gpu_hours_upper": conservative_total_gpu_hours_upper
            if not is_smoke
            else projected_seconds / 3600.0,
            "parallelism_assumed": 1,
        },
        "commands": entries,
        "gate_command": gate_command,
        "gate_command_shell": shlex.join(gate_command),
        "safety": {
            "reads_existing_test_predictions_or_metrics": False,
            "test_dataset_constructed": False,
            "test_artifacts_allowed": False,
            "resume_policy": "skip only completed, re-audited runs; reject partial directories",
            "existing_43_patient_outputs": "quarantined exploratory and never read by this runner",
            "gate_exit_code_3": "locked scientific negative; stop this direction and do not auto-retry",
            "automatic_retry_after_gate_failure": False,
        },
        "input_audit": dict(audit),
    }


def json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n").encode("utf-8")


def write_new_or_identical(
    path: Path,
    payload: object,
    *,
    replace_unexecuted_plan: bool = False,
) -> None:
    encoded = json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != encoded:
            if not replace_unexecuted_plan or path.name not in {"SCREENING_PLAN.json", "SMOKE_PLAN.json"}:
                raise ContractError(f"Refusing to overwrite a different existing plan: {path}")
            try:
                prior = json.loads(existing.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContractError(f"Refusing to refresh an unreadable existing plan: {path}") from exc
            if prior.get("schema_version") != "strict_hierarchical_set_screening_plan_v1":
                raise ContractError(f"Refusing to refresh a plan owned by another schema: {path}")
            run_root = path.parent / ("smoke_runs" if path.name == "SMOKE_PLAN.json" else "runs")
            if run_root.exists() and any(item.is_file() for item in run_root.rglob("*")):
                raise ContractError(f"Refusing to refresh a plan after execution artifacts exist: {run_root}")
        else:
            return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def test_artifact_paths(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        return []
    found: list[Path] = []
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = [part.lower() for part in path.relative_to(run_dir).parts]
        if any("test" in part for part in relative_parts):
            found.append(path)
    return sorted(found)


def run_artifact_sha256(run_dir: Path) -> dict[str, str]:
    """Hash every immutable evidence artifact covered by a screening receipt."""

    require_no_symlinks_in_run(run_dir)
    missing = [name for name in REQUIRED_RECEIPT_ARTIFACTS if not (run_dir / name).is_file()]
    if missing:
        raise ContractError(f"Run is missing receipt-covered artifacts: {missing}")
    names = list(REQUIRED_RECEIPT_ARTIFACTS)
    names.extend(name for name in OPTIONAL_RECEIPT_ARTIFACTS if (run_dir / name).is_file())
    return {name: sha256_file(run_dir / name) for name in names}


def verify_receipt_artifact_sha256(run_dir: Path, receipt: Mapping[str, Any]) -> dict[str, str]:
    current = run_artifact_sha256(run_dir)
    recorded = receipt.get("artifact_sha256")
    if not isinstance(recorded, Mapping):
        raise ContractError("screening_receipt.json has no artifact_sha256 mapping")
    require_equal("receipt-covered artifact names", set(recorded), set(current))
    for name, digest in current.items():
        require_equal(f"receipt artifact SHA256 for {name}", recorded.get(name), digest)
    return current


def csv_patient_set(path: Path) -> tuple[set[str], int]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row.get("patient_id", "")).strip() for row in rows if str(row.get("patient_id", "")).strip()}, len(rows)


def verify_locked_run_execution_config(config: Mapping[str, Any]) -> None:
    """Bind RNG-critical execution settings to the locked run artifact."""

    require_equal("run batch size", config.get("batch_size"), 8)
    require_equal("run worker count", config.get("num_workers"), 0)
    require_equal("run device", config.get("device"), "cuda")


def verify_completed_run(
    run_dir: Path,
    *,
    entry: Mapping[str, Any],
    protocol_sha256: str,
    require_receipt: bool,
) -> dict[str, Any]:
    require_no_symlinks_in_run(run_dir)
    artifacts = test_artifact_paths(run_dir)
    if artifacts:
        raise ContractError(f"Validation-only run contains forbidden test artifacts: {artifacts}")
    required = tuple(run_dir / name for name in REQUIRED_RECEIPT_ARTIFACTS)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise ContractError(f"Run is incomplete: missing {missing}")

    # A gate/resume audit must authenticate every covered byte before parsing
    # any split, configuration, metric, or prediction evidence.
    receipt_path = run_dir / "screening_receipt.json"
    receipt: dict[str, Any] | None = None
    if require_receipt:
        receipt = load_json(receipt_path)
        artifact_sha256 = verify_receipt_artifact_sha256(run_dir, receipt)
    else:
        artifact_sha256 = run_artifact_sha256(run_dir)

    split = load_json(run_dir / "split_summary.json")
    require_equal("run train split", split.get("train"), sorted(EXPECTED_TRAIN_PATIENTS))
    require_equal("run pooled validation split", split.get("val"), sorted(pooled_validation_patients()))
    require_equal("run calibration split", split.get("calibration"), [])
    require_equal("run test split", split.get("test"), [])
    require_equal("run excluded split", split.get("excluded"), [])

    spec = entry["resolved_spec"]
    is_base = entry["config"] == "tfm_soz"
    config = load_json(run_dir / "run_config.json")
    verify_locked_run_execution_config(config)
    require_equal("run seed", config.get("seed"), entry["seed"])
    require_equal("run split seed", config.get("split_seed_used"), EXPECTED_PANEL_SEED)
    require_equal("run test fraction", float(config.get("test_fraction", -1)), 0.0)
    require_equal("run test patient", config.get("test_patient", ""), "")
    require_equal("run skip test", config.get("skip_test_eval"), True)
    require_equal("run tokenizer epochs", config.get("tokenizer_epochs"), entry["tokenizer_epochs"])
    require_equal("run classifier epochs", config.get("classifier_epochs"), entry["classifier_epochs"])
    require_equal("region attention pooling", config.get("use_region_attention_pooling"), True)
    require_equal("region embedding head", config.get("use_region_embedding_head"), True)
    require_equal("run train patients used", config.get("train_patients_used"), sorted(EXPECTED_TRAIN_PATIENTS))
    require_equal("run val patients used", config.get("val_patients_used"), sorted(pooled_validation_patients()))
    rng_contract = config.get("classifier_rng_contract", {})
    require_equal("classifier RNG reseed", rng_contract.get("seed"), entry["seed"])
    require_equal(
        "classifier RNG reseeded after construction/tokenizer training",
        rng_contract.get("reseeded_after_model_construction_and_tokenizer_training"),
        True,
    )
    if not isinstance(rng_contract.get("purpose"), str) or not rng_contract["purpose"].strip():
        raise ContractError("run_config classifier_rng_contract purpose is missing")
    evaluation = config.get("evaluation_contract", {})
    require_equal("test dataset construction", evaluation.get("test_dataset_constructed"), False)
    materialized = set(evaluation.get("materialized_prediction_splits", []))
    if "test" in materialized:
        raise ContractError("run_config says that test predictions were materialized")
    initialization = config.get("initialization", {})
    parameter_config = config.get("parameters", {}).get("classifier", {})
    classifier_total = int(parameter_config.get("total", 0))
    classifier_trainable = int(parameter_config.get("trainable", 0))
    if classifier_total <= 0 or classifier_trainable <= 0 or classifier_trainable > classifier_total:
        raise ContractError("Invalid classifier parameter audit in run_config.json")
    base_lineage: dict[str, Any] | None = None
    if is_base:
        require_equal("base tokenizer initialization", initialization.get("loaded_tokenizer"), False)
        require_equal("base classifier initialization", initialization.get("loaded_classifier"), False)
        require_equal("base freeze policy", initialization.get("freeze_base_channel_head_only"), False)
        require_equal("clean base fully trainable", classifier_trainable, classifier_total)
    else:
        require_equal("arm tokenizer initialization", initialization.get("loaded_tokenizer"), True)
        require_equal("arm classifier initialization", initialization.get("loaded_classifier"), True)
        require_equal("arm freeze policy", initialization.get("freeze_base_channel_head_only"), True)
        if not classifier_trainable < classifier_total:
            raise ContractError("Matched arm did not freeze the shared base classifier")
        require_equal(
            "frozen arm trainable parameter audit",
            initialization.get("channel_head_trainable_parameters"),
            classifier_trainable,
        )
        base_run_dir = resolve_under_repo(str(entry.get("depends_on", "")), REPO_ROOT)
        expected_init = str(entry.get("depends_on", ""))
        require_equal("same-seed base init path", config.get("init_run_dir"), expected_init)
        base_tokenizer = base_run_dir / "tfm_tokenizer_segments.pth"
        base_classifier = base_run_dir / "tfm_soz_segment_classifier.pth"
        if not base_tokenizer.is_file() or not base_classifier.is_file():
            raise ContractError(f"Matched arm base checkpoints are missing: {base_run_dir}")
        base_lineage = {
            "base_run_dir": expected_init,
            "base_tokenizer_sha256": sha256_file(base_tokenizer),
            "base_classifier_sha256": sha256_file(base_classifier),
        }
        require_equal(
            "frozen arm tokenizer checkpoint identity",
            sha256_file(run_dir / "tfm_tokenizer_segments.pth"),
            base_lineage["base_tokenizer_sha256"],
        )

    require_equal("training objective", config.get("training_objective"), spec["training_objective"])
    require_equal(
        "channel Top1 margin weight",
        float(config.get("channel_top1_margin_weight", -1)),
        float(spec["channel_top1_margin_weight"]),
    )
    require_equal("TKPR weight", float(config.get("channel_tkpr_weight", -1)), float(spec["channel_tkpr_weight"]))
    require_equal("TKPR K", int(config.get("channel_tkpr_k", -1)), TKPR_K)
    require_equal(
        "resolved arm freeze",
        config.get("resolved_model_options", {}).get("freeze_base_channel_head_only"),
        bool(spec["freeze_base_channel_head_only"]),
    )
    resolved_model = config.get("resolved_model_options", {})
    require_equal(
        "set head enabled",
        resolved_model.get("use_region_channel_set_head"),
        bool(spec["use_region_channel_set_head"]),
    )
    require_equal("set bottleneck", int(resolved_model.get("region_channel_set_bottleneck", -1)), SET_BOTTLENECK)
    require_equal(
        "set mapping policy",
        resolved_model.get("region_channel_set_mapping_policy"),
        spec["region_channel_set_mapping_policy"],
    )
    require_equal(
        "set residual initialization",
        float(resolved_model.get("region_channel_set_residual_init", float("nan"))),
        SET_RESIDUAL_INIT,
    )

    metrics = load_json(run_dir / "metrics.json")
    split_metrics = metrics.get("metrics", {})
    if not isinstance(split_metrics, Mapping) or "val" not in split_metrics:
        raise ContractError("metrics.json has no validation metrics")
    if "test" in split_metrics:
        raise ContractError("metrics.json contains a forbidden test split")
    observed_patients, observed_rows = csv_patient_set(run_dir / "val_predictions.csv")
    require_equal("val_predictions patients", observed_patients, set(pooled_validation_patients()))
    require_equal("val_predictions event count", observed_rows, sum(EXPECTED_PANEL_EVENT_COUNTS.values()))

    if require_receipt:
        if receipt is None:  # Defensive type narrowing; the branch above must set it.
            raise ContractError("Required screening receipt was not loaded")
        require_equal("receipt passed", receipt.get("passed"), True)
        require_equal("receipt protocol SHA256", receipt.get("protocol_sha256"), protocol_sha256)
        require_equal("receipt command SHA256", receipt.get("command_sha256"), entry["command_sha256"])
        require_equal("receipt code lock", receipt.get("code_lock_sha256"), entry["code_lock_sha256"])
        require_equal(
            "receipt execution lock",
            receipt.get("execution_lock_sha256"),
            entry["execution_lock_sha256"],
        )
        require_equal("receipt initialization lineage", receipt.get("initialization_lineage"), base_lineage)
        own_checkpoints = receipt.get("output_checkpoint_sha256", {})
        require_equal(
            "receipt output tokenizer SHA256",
            own_checkpoints.get("tfm_tokenizer_segments.pth"),
            sha256_file(run_dir / "tfm_tokenizer_segments.pth"),
        )
        require_equal(
            "receipt output classifier SHA256",
            own_checkpoints.get("tfm_soz_segment_classifier.pth"),
            sha256_file(run_dir / "tfm_soz_segment_classifier.pth"),
        )
    return {
        "passed": True,
        "config": entry["config"],
        "seed": entry["seed"],
        "protocol_sha256": protocol_sha256,
        "command_sha256": entry["command_sha256"],
        "code_lock_sha256": entry["code_lock_sha256"],
        "execution_lock_sha256": entry["execution_lock_sha256"],
        "initialization_lineage": base_lineage,
        "output_checkpoint_sha256": {
            "tfm_tokenizer_segments.pth": sha256_file(run_dir / "tfm_tokenizer_segments.pth"),
            "tfm_soz_segment_classifier.pth": sha256_file(run_dir / "tfm_soz_segment_classifier.pth"),
        },
        "artifact_sha256": artifact_sha256,
        "train_patients": 28,
        "validation_patients": 15,
        "validation_events": observed_rows,
        "test_dataset_constructed": False,
        "test_artifacts": [],
    }


def execute_plan(
    *,
    plan: Mapping[str, Any],
    repo_root: Path,
    resume: bool,
) -> None:
    if not plan.get("execution_ready"):
        raise ContractError(f"Execution is blocked: {plan.get('blocked_reasons')}")
    if shutil.which("rtk") is None:
        raise ContractError("rtk is required but was not found on PATH")
    protocol_sha256 = str(plan["protocol"]["sha256"])
    for entry in plan["commands"]:
        current_audit = validate_locked_inputs(
            repo_root=repo_root,
            protocol_path=resolve_under_repo(plan["protocol"]["path"], repo_root),
            preprocessed_dir=resolve_under_repo(plan["protocol"]["preprocessed_dir"], repo_root),
            trainer_path=resolve_under_repo(TRAINER, repo_root),
        )
        require_equal(
            "current execution lock",
            current_audit["execution_lock_sha256"],
            plan["input_audit"]["execution_lock_sha256"],
        )
        run_dir = resolve_under_repo(entry["run_dir"], repo_root)
        require_no_symlinks_in_run(run_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            if not resume:
                raise ContractError(f"Fresh output required; run directory already exists: {run_dir}")
            try:
                verify_completed_run(
                    run_dir,
                    entry=entry,
                    protocol_sha256=protocol_sha256,
                    require_receipt=True,
                )
            except ContractError as exc:
                raise ContractError(
                    f"Resume refuses to overwrite or continue a partial/invalid run directory {run_dir}: {exc}"
                ) from exc
            print(f"[resume] verified and skipped {entry['config']} seed={entry['seed']}", flush=True)
            continue
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[run] {entry['command_shell']}", flush=True)
        subprocess.run(entry["command"], cwd=repo_root, check=True)
        receipt = verify_completed_run(
            run_dir,
            entry=entry,
            protocol_sha256=protocol_sha256,
            require_receipt=False,
        )
        receipt_path = run_dir / "screening_receipt.json"
        write_new_or_identical(receipt_path, receipt)
        verify_completed_run(
            run_dir,
            entry=entry,
            protocol_sha256=protocol_sha256,
            require_receipt=True,
        )


def print_plan_summary(plan: Mapping[str, Any], plan_path: Path) -> None:
    estimate = plan["runtime_estimate"]
    print(
        json.dumps(
            {
                "plan_kind": plan["mode"],
                "plan": str(plan_path),
                "execution_ready": plan["execution_ready"],
                "runs": estimate["runs"],
                "empirical_projected_total_minutes": estimate["empirical_projected_total_minutes"],
                "empirical_projected_total_gpu_hours": estimate["empirical_projected_total_gpu_hours"],
                "conservative_total_gpu_hours_upper": estimate["conservative_total_gpu_hours_upper"],
                "split": "fixed train=28, pooled val=15, test=0",
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    for entry in plan["commands"]:
        print(entry["command_shell"], flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "smoke", "screen"), default="plan")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--preprocessed-dir", default=str(DEFAULT_PREPROCESSED))
    parser.add_argument("--screening-root", default=str(DEFAULT_SCREENING_ROOT))
    parser.add_argument("--trainer", default=str(TRAINER))
    parser.add_argument("--tokenizer-epochs", type=int, default=BASE_TOKENIZER_EPOCHS)
    parser.add_argument("--classifier-epochs", type=int, default=BASE_CLASSIFIER_EPOCHS)
    parser.add_argument("--arm-classifier-epochs", type=int, default=ARM_CLASSIFIER_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--conservative-total-gpu-hours-upper",
        type=float,
        default=CONSERVATIVE_TOTAL_GPU_HOURS_UPPER,
    )
    parser.add_argument(
        "--refresh-plan",
        action="store_true",
        help="Replace only this runner's unexecuted plan after an intentional code-lock refresh.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip only completed runs with a valid receipt; partial run directories remain fatal.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.tokenizer_epochs != BASE_TOKENIZER_EPOCHS
        or args.classifier_epochs != BASE_CLASSIFIER_EPOCHS
        or args.arm_classifier_epochs != ARM_CLASSIFIER_EPOCHS
    ):
        raise ContractError("The locked full schedule is base 16/90 and matched frozen arms 0/12")
    if (
        args.batch_size != 8
        or args.num_workers != 0
        or args.device != "cuda"
        or args.conservative_total_gpu_hours_upper != 2.0
    ):
        raise ContractError("The locked compute contract is batch_size=8, num_workers=0, device=cuda")

    repo_root = REPO_ROOT
    protocol_path = resolve_under_repo(args.protocol, repo_root)
    preprocessed_dir = resolve_under_repo(args.preprocessed_dir, repo_root)
    screening_root = resolve_under_repo(args.screening_root, repo_root)
    trainer_path = resolve_under_repo(args.trainer, repo_root)
    require_equal(
        "protocol path",
        protocol_path.resolve(),
        resolve_under_repo(DEFAULT_PROTOCOL, repo_root).resolve(),
    )
    require_equal(
        "preprocessed path",
        preprocessed_dir.resolve(),
        resolve_under_repo(DEFAULT_PREPROCESSED, repo_root).resolve(),
    )
    require_equal(
        "screening root",
        screening_root.resolve(),
        resolve_under_repo(DEFAULT_SCREENING_ROOT, repo_root).resolve(),
    )
    require_equal("trainer path", trainer_path.resolve(), resolve_under_repo(TRAINER, repo_root).resolve())
    require_no_symlink_chain(screening_root / "runs", anchor=repo_root)
    audit = validate_locked_inputs(
        repo_root=repo_root,
        protocol_path=protocol_path,
        preprocessed_dir=preprocessed_dir,
        trainer_path=trainer_path,
    )
    plan = build_plan(
        mode=args.mode,
        repo_root=repo_root,
        protocol_path=protocol_path,
        preprocessed_dir=preprocessed_dir,
        screening_root=screening_root,
        trainer_path=trainer_path,
        audit=audit,
        tokenizer_epochs=args.tokenizer_epochs,
        classifier_epochs=args.classifier_epochs,
        arm_classifier_epochs=args.arm_classifier_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        conservative_total_gpu_hours_upper=args.conservative_total_gpu_hours_upper,
    )
    plan_name = "SMOKE_PLAN.json" if args.mode == "smoke" else "SCREENING_PLAN.json"
    plan_path = screening_root / plan_name
    require_no_symlink_chain(plan_path, anchor=repo_root)
    write_new_or_identical(plan_path, plan, replace_unexecuted_plan=bool(args.refresh_plan))
    print_plan_summary(plan, plan_path)
    if args.mode == "plan":
        return 0

    execute_plan(plan=plan, repo_root=repo_root, resume=bool(args.resume))
    if args.mode == "screen":
        result = subprocess.run(plan["gate_command"], cwd=repo_root, check=False)
        if int(result.returncode) == 3:
            print(
                "LOCKED SCIENTIFIC NEGATIVE: promotion gate failed; no automatic retry or test evaluation is permitted.",
                file=sys.stderr,
            )
        return int(result.returncode)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"STRICT SCREEN REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
