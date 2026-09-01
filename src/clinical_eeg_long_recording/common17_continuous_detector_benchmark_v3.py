"""Strict validator for the frozen common-17 continuous-detector benchmark v3."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/common17_continuous_detector_benchmark_v3.json"

COMMON17 = (
    "FP1",
    "F3",
    "C3",
    "P3",
    "O1",
    "F7",
    "T7",
    "P7",
    "CZ",
    "FP2",
    "F4",
    "C4",
    "P4",
    "O2",
    "F8",
    "T8",
    "P8",
)

# This order is not cosmetic: it is the upstream ST18 longitudinal-bipolar
# order after deleting only FZ-CZ and CZ-PZ.  A scratch ST16 checkpoint is
# bound to these axis positions, so a permutation is a different provider.
CANONICAL_ST16_TYPED_UNITS = (
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
)
CANONICAL_ST16_PAIRS = tuple(
    tuple(unit.split("-", 1)) for unit in CANONICAL_ST16_TYPED_UNITS
)

EXPECTED_SPLITS = {
    "source_train": {
        "patients": 579,
        "recordings": 4664,
        "duration_seconds": 3276609,
        "analysis_identity_roster_sha256": "f692b78e317885bd558f61ebf3f988315986a5795aaed4ca3673504921b6eec9",
        "patient_roster_sha256": "25ebe0ec18e07e4720b8ceaa2468c6a7be05dc836decfecb5669c70667fe770b",
    },
    "source_dev": {
        "patients": 53,
        "recordings": 1821,
        "duration_seconds": 1566304,
        "analysis_identity_roster_sha256": "0f25a40838bd72616e50e4effb48f08d5ca4240f906334fd5ed0a67c5a6e9b99",
        "patient_roster_sha256": "b25ed2f3bc42b9ac23438f048f2f213c6354da1325c36a8258bfc20a58dbbb79",
    },
    "source_eval": {
        "patients": 43,
        "recordings": 864,
        "duration_seconds": 459481,
        "analysis_identity_roster_sha256": "4962082269c1bd25ec7028270b90503ff16f0255cd408a42199e149dca398487",
        "patient_roster_sha256": "c29d754dfa2497e7065751bbac1d5eeecc2a067aac6e932896ec7ed718577569",
    },
}


class Common17BenchmarkV3Error(ValueError):
    """Raised when the frozen protocol or an evidence binding is invalid."""


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Common17BenchmarkV3Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError) as exc:
        raise Common17BenchmarkV3Error(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Common17BenchmarkV3Error(f"top-level JSON must be an object: {path}")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def common17_benchmark_v3_self_sha256(config: Mapping[str, object]) -> str:
    payload = dict(config)
    payload.pop("receipt_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(value: object, context: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise Common17BenchmarkV3Error(f"{context} must be a non-empty project-relative path")
    candidate = (PROJECT_ROOT / value).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise Common17BenchmarkV3Error(f"{context} escapes project root") from exc
    return candidate


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise Common17BenchmarkV3Error(message)


def _expect_close(actual: object, expected: float, context: str, atol: float = 1e-12) -> None:
    _expect(isinstance(actual, (int, float)) and math.isfinite(float(actual)), f"{context} must be finite")
    _expect(abs(float(actual) - expected) <= atol, f"{context}: expected {expected}, got {actual}")


def _validate_file_binding(binding: Mapping[str, object], context: str) -> Path:
    path = _project_path(binding.get("path"), f"{context}.path")
    _expect(path.is_file(), f"{context}.path does not exist: {path}")
    expected = binding.get("file_sha256")
    _expect(isinstance(expected, str) and len(expected) == 64, f"{context}.file_sha256 invalid")
    actual = _file_sha256(path)
    _expect(actual == expected, f"{context} SHA-256 mismatch: {actual} != {expected}")
    return path


def _validate_channels(config: Mapping[str, object]) -> None:
    channel = config["channel_contract"]
    _expect(isinstance(channel, dict), "channel_contract must be an object")
    _expect(tuple(channel["canonical_axis_order"]) == COMMON17, "canonical common17 axis order changed")
    _expect("FZ" not in channel["canonical_axis_order"] and "PZ" not in channel["canonical_axis_order"], "FZ/PZ leaked into model axes")
    for key in ("FZ_or_PZ_model_axis_allowed", "zero_fill_allowed", "interpolation_allowed", "synthetic_midline_allowed"):
        _expect(channel[key] is False, f"channel_contract.{key} must remain false")
    _expect(channel["direct_observed_referential_axes_only"] is True, "common17 must use observed axes")
    views = channel["allowed_primary_provider_views"]
    _expect(isinstance(views, list) and {view["view_id"] for view in views} == {"C17-REF", "C17-LB16"}, "provider views changed")
    for view in views:
        if view["view_id"] == "C17-LB16":
            pairs = view.get("pairs")
            _expect(isinstance(pairs, list) and len(pairs) == 16, "C17-LB16 must contain 16 pairs")
            for pair in pairs:
                _expect(isinstance(pair, list) and len(pair) == 2, "each bipolar axis needs two endpoints")
                _expect(all(endpoint in COMMON17 for endpoint in pair), "derived montage endpoint is outside common17")
            _expect(
                tuple(tuple(pair) for pair in pairs) == CANONICAL_ST16_PAIRS,
                "C17-LB16 order differs from the upstream-derived canonical ST16 order",
            )


def _validate_population(config: Mapping[str, object]) -> None:
    population = config["population"]
    _expect(population["dataset"] == "TUSZ" and population["version"] == "2.0.3", "dataset/version changed")
    _expect(population["split_unit"] == "patient", "split unit must be patient")
    _expect(population["same_patient_across_splits_allowed"] is False, "patient overlap cannot be allowed")
    official = population["official_container_identity_counts"]
    _expect(official == {"source_train": 4667, "source_dev": 1832, "source_eval": 864, "total": 7363}, "official container counts changed")
    aliases = population["exact_physical_alias_policy"]
    _expect(aliases["alias_is_technical_failure"] is False, "physical aliases are not technical failures")
    _expect(aliases["alias_gets_independent_metric_weight"] is False, "physical aliases cannot get independent metric weight")
    _expect(aliases["total_aliases_removed"] == 14, "expected 14 exact physical aliases")
    rosters = population["common17_physical_analysis_rosters"]
    for split, expected in EXPECTED_SPLITS.items():
        row = rosters[split]
        for key, value in expected.items():
            _expect(row[key] == value, f"population.{split}.{key} changed")
    _expect(sum(rosters[s]["recordings"] for s in EXPECTED_SPLITS) == 7349, "physical total must be 7349")
    _expect(official["total"] - aliases["total_aliases_removed"] == 7349, "alias arithmetic mismatch")

    bindings = population["bindings"]
    audit_path = _validate_file_binding(bindings["canonical_physical_signal_audit"], "canonical audit")
    fold_path = _validate_file_binding(bindings["patient_disjoint_fold_plan"], "fold plan")
    manifest_path = _validate_file_binding(bindings["common17_detector_manifest"], "common17 manifest")
    audit = _load_json(audit_path)
    fold = _load_json(fold_path)
    manifest = _load_json(manifest_path)
    _expect(len(audit["outcomes"]) == 7363, "canonical audit identity count changed")
    _expect(manifest["patient_disjoint_train_dev"] is True and manifest["patient_overlap"] == [], "train/dev patient overlap")
    _expect(tuple(manifest["channel_contract"]["common17_channel_order"]) == COMMON17, "manifest channel order differs")
    _expect(manifest["channel_contract"]["FZ_or_PZ_read_into_model_tensor"] is False, "manifest reads FZ/PZ")
    for split, expected in EXPECTED_SPLITS.items():
        fold_row = fold["source_split_rosters"][split]
        _expect(fold_row["patient_count"] == expected["patients"], f"fold {split} patient count mismatch")
        _expect(fold_row["recording_count"] == expected["recordings"], f"fold {split} recording count mismatch")
        _expect(fold_row["duration_seconds_fraction"] == [expected["duration_seconds"], 1], f"fold {split} duration mismatch")
        _expect(fold_row["analysis_identity_roster_sha256"] == expected["analysis_identity_roster_sha256"], f"fold {split} identity hash mismatch")
        _expect(fold_row["patient_roster_sha256"] == expected["patient_roster_sha256"], f"fold {split} patient hash mismatch")
    for split in ("source_train", "source_dev"):
        summary = manifest["split_summaries"][split]
        _expect(summary["patient_count"] == EXPECTED_SPLITS[split]["patients"], f"manifest {split} patients mismatch")
        _expect(summary["recording_count"] == EXPECTED_SPLITS[split]["recordings"], f"manifest {split} recordings mismatch")


def _validate_firewall_and_inventory(config: Mapping[str, object]) -> None:
    firewall = config["split_and_leakage_firewall"]
    _expect(firewall["source_eval_reference_access_initially_authorized"] is False, "source-eval reference firewall opened")
    _expect(firewall["source_eval_dense_predictions_frozen_before_reference_join"] is True, "source-eval predictions must freeze first")
    _expect(firewall["source_eval_policy_and_checkpoint_frozen_before_reference_join"] is True, "source-eval policy/checkpoint must freeze first")
    forbidden = set(firewall["forbidden_in_model_input_or_inference"])
    for required in ("edf_annotations", "excel_or_spreadsheet_fields", "reference_seizure_intervals", "SOZ_labels", "clinical_text"):
        _expect(required in forbidden, f"missing forbidden input: {required}")
    inventory = config["prediction_first_inventory"]
    _expect(inventory["one_terminal_row_per_record_provider_seed_precision"] is True, "terminal inventory cross-product required")
    _expect(inventory["complete_inventory_required_for_primary_admission"] is True, "complete inventory gate required")
    _expect(inventory["force_positive_alarm_allowed"] is False, "forced positive alarms are forbidden")
    _expect(set(inventory["terminal_outcomes"]) == {"completed_with_candidates", "completed_zero_candidate", "partial_coverage", "technical_failure"}, "terminal outcome taxonomy changed")


def _validate_scoring_and_operating_points(config: Mapping[str, object]) -> None:
    scoring = config["scoring_tracks"]
    _expect(scoring["report_both_tracks_for_every_frozen_operating_point"] is True, "both metric tracks are mandatory")
    szcore = scoring["szcore_compatibility"]
    _expect(szcore["framework_doi"].lower() == "10.1111/epi.18113", "SzCORE DOI changed")
    _expect(szcore["framework_commit"] == "5161f4c3745d558b5466c10621b0a11cc7b3e266", "SzCORE commit changed")
    _expect(szcore["timescoring_commit"] == "426f8d2b77974641dc9db71884e0812b249ba93b", "timescoring commit changed")
    _expect(szcore["timescoring_version"] == "0.0.7", "timescoring version changed")
    _expect(szcore["default_event_rules"] == {
        "minimum_overlap": "any_positive_overlap",
        "preictal_tolerance_seconds": 30,
        "postictal_tolerance_seconds": 60,
        "merge_gap_less_than_seconds": 90,
        "split_event_longer_than_seconds": 300,
    }, "SzCORE defaults changed")
    _expect(szcore["may_be_interpreted_as_onset_accuracy"] is False, "SzCORE event score cannot become onset accuracy")
    strict = scoring["strict_event_onset"]
    _expect(strict["reference_interval_dilation_seconds"] == 0, "strict track cannot dilate references")
    _expect(strict["onset_hit_denominator"] == "all_reference_events_unmatched_are_misses", "onset denominator changed")
    _expect(strict["one_prediction_can_match_multiple_references"] is False, "one alarm cannot claim multiple references")
    _expect(strict["one_reference_can_match_multiple_predictions"] is False, "one reference cannot be claimed repeatedly")

    ops = config["operating_points"]
    alarm = ops["alarm"]
    _expect(alarm["op_id"] == "OP-ALARM" and alarm["may_select_accuracy_primary"] is True, "Alarm OP identity changed")
    _expect(alarm["absolute_admission_gates"] == {
        "strict_pooled_event_sensitivity_minimum": 0.9,
        "strict_patient_macro_event_sensitivity_minimum": 0.85,
        "strict_all_unmatched_alarms_per_24h_maximum": 12.0,
        "warm_end_to_end_RTF_maximum": 0.05,
        "technical_failure_count_maximum": 0,
        "partial_coverage_count_maximum": 0,
    }, "Alarm OP gates changed")
    navigation = ops["navigation"]
    _expect(navigation["op_id"] == "OP-NAVIGATION" and navigation["may_select_navigation_primary"] is True, "Navigation OP identity changed")
    _expect(navigation["navigation_candidate_is_confirmed_seizure"] is False, "navigation candidates cannot be called confirmed seizures")
    _expect(navigation["candidate_budgets_per_evaluable_hour"] == [1, 2, 4, 8, 16], "candidate budgets changed")
    _expect(navigation["queried_EEG_seconds_budgets_per_evaluable_hour"] == [60, 120, 300, 600], "query budgets changed")
    _expect(ops["blended_accuracy_efficiency_score_allowed"] is False, "blended score forbidden")


def _validate_hardware_and_statistics(config: Mapping[str, object]) -> None:
    stats = config["statistics"]
    _expect(stats["confidence_interval_unit"] == "patient", "bootstrap unit must be patient")
    _expect(stats["bootstrap_replicates"] == 2000, "bootstrap replicate count changed")
    _expect(stats["paired_model_comparison_uses_shared_patient_draws"] is True, "paired models need shared patient draws")
    hardware = config["fixed_hardware_efficiency_protocol"]
    _expect(hardware["GPU"]["uuid"] == "GPU-4e73be7a-8780-d0f5-7ec4-5ce8bdd072bc", "benchmark GPU changed")
    execution = hardware["execution"]
    _expect(execution["GPU_count"] == 1 and execution["recording_concurrency"] == 1, "fixed single-GPU concurrency changed")
    _expect(execution["background_GPU_compute_or_memory_interference_invalidates_run"] is True, "GPU interference gate required")
    _expect(execution["accuracy_and_efficiency_must_use_same_checkpoint_transform_decoder_and_precision"] is True, "accuracy/speed identity binding required")
    _expect(hardware["current_eventnet_RTF_is_v3_fixed_hardware_comparable"] is False, "legacy EventNet RTF cannot become v3-comparable")


def _validate_providers(config: Mapping[str, object]) -> None:
    providers = config["provider_plan"]
    _expect(isinstance(providers, list) and len(providers) == 5, "five provider rows required")
    ids = [provider["provider_id"] for provider in providers]
    _expect(len(ids) == len(set(ids)), "provider IDs must be unique")
    _expect([provider["priority"] for provider in providers] == [1, 2, 3, 4, 5], "provider priorities changed")
    _expect(providers[0]["provider_id"] == "st16_common17_cleanroom_v3", "ST16 must remain first accuracy challenger")
    allowed_views = {"C17-REF", "C17-LB16"}
    _expect(all(provider["provider_view"] in allowed_views for provider in providers), "provider view outside common17")
    eventnet = next(provider for provider in providers if provider["provider_id"] == "eventnet_en17_common17_shorttrain_control_v1")
    _expect(eventnet["role"] == "executed_failure_and_efficiency_control", "EventNet role changed")
    evidence = eventnet["local_evidence"]
    metrics_path = _project_path(evidence["metrics_path"], "EventNet metrics path")
    training_path = _project_path(evidence["training_receipt_path"], "EventNet training path")
    replay_path = _project_path(evidence["independent_metric_replay_path"], "EventNet metric replay path")
    for path, expected, context in (
        (metrics_path, evidence["metrics_file_sha256"], "EventNet metrics"),
        (training_path, evidence["training_receipt_file_sha256"], "EventNet training receipt"),
        (replay_path, evidence["independent_metric_replay_file_sha256"], "EventNet independent metric replay"),
    ):
        _expect(path.is_file(), f"{context} missing")
        _expect(_file_sha256(path) == expected, f"{context} SHA-256 mismatch")
    metrics = _load_json(metrics_path)
    training = _load_json(training_path)
    replay = _load_json(replay_path)
    _expect(replay["receipt_sha256"] == evidence["independent_metric_replay_receipt_sha256"], "EventNet metric replay receipt mismatch")
    _expect(replay["status"] == "pass_independent_decoder_matching_metric_replay", "EventNet metric replay did not pass")
    _expect(replay["replay_performed_inference"] is False, "EventNet metric replay unexpectedly ran inference")
    _expect(all(replay["exact_frozen_metric_comparisons"].values()), "EventNet replay differs from frozen metrics")
    best = metrics["best_source_dev_diagnostic_operating_point"]
    pooled = best["pooled"]
    _expect(best["center_threshold"] == evidence["diagnostic_threshold"], "EventNet diagnostic threshold mismatch")
    _expect(pooled["recording_count"] == 1821 and pooled["reference_event_count"] == 1074, "EventNet denominator mismatch")
    _expect_close(pooled["event_sensitivity"], evidence["strict_event_sensitivity"], "EventNet sensitivity")
    _expect_close(pooled["event_precision"], evidence["strict_event_precision"], "EventNet precision")
    _expect_close(pooled["event_f1"], evidence["strict_event_f1"], "EventNet F1")
    _expect_close(pooled["alarm_false_alarms_per_24h"], evidence["strict_all_unmatched_alarms_per_24h"], "EventNet FA/24h")
    _expect_close(pooled["onset_absolute_hit_rate"]["10s"]["rate"], evidence["reference_denominator_onset_hit_at_10s"], "EventNet Hit@10s")
    _expect_close(metrics["runtime"]["end_to_end_EEG_IO_resample_inference_decode_RTF"], evidence["end_to_end_RTF"], "EventNet RTF")
    _expect(training["requested_epochs"] == 3 and training["global_step"] == 1752, "EventNet short-training identity mismatch")
    _expect(training["FZ_or_PZ_model_axis_present"] is False, "EventNet checkpoint reads FZ/PZ")


def validate_common17_continuous_detector_benchmark_v3(
    config: Mapping[str, object], *, verify_receipt: bool = True
) -> dict[str, object]:
    """Validate the full v3 protocol and its local evidence bindings."""

    required_top = {
        "schema_version",
        "benchmark_id",
        "frozen_on",
        "status",
        "scope",
        "channel_contract",
        "population",
        "split_and_leakage_firewall",
        "prediction_first_inventory",
        "scoring_tracks",
        "operating_points",
        "statistics",
        "fixed_hardware_efficiency_protocol",
        "provider_plan",
        "selection_gate",
        "execution_order",
        "receipt_sha256",
    }
    _expect(set(config) == required_top, "v3 top-level schema changed")
    _expect(config["schema_version"] == "common17_continuous_long_eeg_detector_benchmark_v3", "schema version changed")
    _expect(config["benchmark_id"] == "COMMON17-CONTINUOUS-DETECTOR-BENCHMARK-V3-20260825", "benchmark ID changed")
    _expect(config["scope"]["accuracy_primary"] is None and config["scope"]["navigation_primary"] is None, "primary cannot be selected without evaluation")
    _expect(config["selection_gate"]["accuracy_primary_current_value"] is None, "accuracy primary gate must be null")
    _expect(config["selection_gate"]["navigation_primary_current_value"] is None, "navigation primary gate must be null")
    _validate_channels(config)
    _validate_population(config)
    _validate_firewall_and_inventory(config)
    _validate_scoring_and_operating_points(config)
    _validate_hardware_and_statistics(config)
    _validate_providers(config)
    expected_receipt = common17_benchmark_v3_self_sha256(config)
    if verify_receipt:
        _expect(config["receipt_sha256"] == expected_receipt, f"receipt mismatch: expected {expected_receipt}")
    return {
        "schema_version": "common17_continuous_detector_benchmark_v3_readiness_v1",
        "benchmark_id": config["benchmark_id"],
        "protocol_valid": True,
        "common17_channels": list(COMMON17),
        "physical_rosters": {split: EXPECTED_SPLITS[split]["recordings"] for split in EXPECTED_SPLITS},
        "accuracy_primary": None,
        "navigation_primary": None,
        "eventnet_status": "executed_failure_control_not_source_eval",
        "first_accuracy_challenger": "st16_common17_cleanroom_v3",
        "receipt_sha256": expected_receipt,
    }


def load_common17_continuous_detector_benchmark_v3(
    path: Path | str = DEFAULT_CONFIG_PATH, *, verify_receipt: bool = True
) -> tuple[dict[str, Any], dict[str, object]]:
    config = _load_json(Path(path))
    readiness = validate_common17_continuous_detector_benchmark_v3(config, verify_receipt=verify_receipt)
    return config, readiness
