#!/usr/bin/env python3
"""Content-bind and fail-closed audit the clinical EEG v4 architecture overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/clinical_eeg_top_tier_method_architecture_v4.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs/clinical_eeg_top_tier_method_architecture_v4_20260825/receipt.json"
)
EXPECTED_COMMON17 = {
    "FP1",
    "FP2",
    "F7",
    "F3",
    "F4",
    "F8",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "P7",
    "P3",
    "P4",
    "P8",
    "O1",
    "O2",
}
REQUIRED_FORBIDDEN = {
    "EDF_annotations",
    "Excel_onset_fields",
    "doctor_labels_or_text",
    "clinical_history",
    "video_or_behavior",
    "sleep_staging",
    "provocation",
    "ECG_EMG_EOG",
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_address(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["receipt_sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    return result


def _bound_file(binding: Mapping[str, object]) -> dict[str, str]:
    relative = Path(str(binding["path"]))
    path = (ROOT / relative).resolve(strict=True)
    expected = str(binding["sha256"])
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"content binding mismatch for {relative}: {observed} != {expected}")
    return {"path": str(relative), "sha256": observed}


def audit(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "clinical_eeg_top_tier_method_architecture_v4_overlay":
        raise ValueError("unexpected v4 architecture schema")

    inheritance = config["inheritance"]
    parent = _bound_file(
        {"path": inheritance["parent_path"], "sha256": inheritance["parent_sha256"]}
    )
    parent_markdown = _bound_file(
        {
            "path": inheritance["parent_markdown_path"],
            "sha256": inheritance["parent_markdown_sha256"],
        }
    )
    if inheritance["may_weaken_inference_firewall"] is not False:
        raise ValueError("v4 may not weaken the inference firewall")
    if inheritance["may_relabel_conditional_or_development_results_as_end_to_end_or_test"] is not False:
        raise ValueError("v4 may not relabel conditional/development results")

    scope = config["frozen_scope"]
    if set(scope["primary_sensor_set"]) != EXPECTED_COMMON17:
        raise ValueError("v4 common17 sensor set mismatch")
    if scope["FZ_or_PZ_signal_synthesis_interpolation_or_zero_fill"] is not False:
        raise ValueError("v4 permits synthetic FZ/PZ")
    if scope["ground_truth_only_midline_mapping"]["prediction_scores_remapped_to_CZ"]:
        raise ValueError("v4 remaps predictions to CZ")

    firewall = config["inference_firewall"]
    if not REQUIRED_FORBIDDEN <= set(firewall["forbidden"]):
        raise ValueError("v4 inference firewall is incomplete")
    if firewall["labels_allowed_for_training_or_postfreeze_evaluation_only"] is not True:
        raise ValueError("v4 label permission is not fail-closed")

    evidence = config["frozen_evidence_update"]
    retrain_report = _bound_file(evidence["common17_independent_retrain_report"])
    negative = evidence["adaptive_support_v1_negative_audit"]
    adaptive_receipt = _bound_file(
        {"path": negative["path"], "sha256": negative["file_sha256"]}
    )
    if negative["events"] != 259 or negative["technical_failures"] != 0:
        raise ValueError("adaptive-v1 denominator mismatch")
    if negative["dominant_geometry_fraction"] <= 0.9:
        raise ValueError("adaptive-v1 degeneracy is not represented")
    if negative["promotion_or_superiority_claim_allowed"] is not False:
        raise ValueError("adaptive-v1 was incorrectly promoted")

    support = config["anchor_decoupled_bilateral_support_v2"]
    background = support["background_bank"]
    search = support["candidate_search"]
    if background["minimum_nonoverlapping_consensus_blocks"] < 2:
        raise ValueError("v4 background may rely on one stable block")
    if background["stable_single_block_is_sufficient"] is not False:
        raise ValueError("v4 background single-block guard is open")
    if background["uses_TERM_reference_SOZ_or_clinical_text"] is not False:
        raise ValueError("v4 background selector can reach forbidden targets")
    if search["search_start_is_function_of_support_start"] is not False:
        raise ValueError("v4 search origin remains coupled to support start")
    if search["search_start_is_function_of_background_midpoint"] is not False:
        raise ValueError("v4 search origin remains coupled to background midpoint")
    if search["left_extension_reveals_earlier_cells_on_the_same_prefrozen_lattice"] is not True:
        raise ValueError("v4 left extension cannot recover earlier onset candidates")
    if "provisional" not in str(search["current_numeric_threshold_role"]):
        raise ValueError("v4 provisional changepoint thresholds are misrepresented")
    if "source_train_patient_separated" not in str(search["final_threshold_calibration"]):
        raise ValueError("v4 final threshold calibration is not patient-separated source-train")
    if search["source_dev_TERM_may_tune_thresholds"] is not False:
        raise ValueError("v4 allows source-dev TERM to tune support thresholds")

    endpoint = support["promotion_endpoint"]
    if endpoint["TERM_timing_alone_may_select_or_promote_support_policy"] is not False:
        raise ValueError("TERM timing can improperly promote a support policy")
    if endpoint["arm_agreement_alone_may_select_or_promote_support_policy"] is not False:
        raise ValueError("arm agreement can improperly promote a support policy")
    permission = support["temporal_evidence_permission"]
    if permission["detector_anchor_is_clinical_onset"] is not False:
        raise ValueError("v4 promotes a navigation anchor to clinical onset")
    if permission["pre_and_post_detector_anchor_cells_may_propose_EEG_change_candidate"] is not True:
        raise ValueError("v4 cannot recover an early detector anchor")
    if permission["postanchor_blocks_may_be_primary_onset_normalization_baseline"] is not False:
        raise ValueError("v4 allows post-anchor onset normalization")
    if permission["positive_onset_or_SOZ_rank_permission"] != "candidate_locked_onset_causal_prefix_only":
        raise ValueError("v4 positive onset-rank temporal permission drifted")
    if permission["late_spread_course_recovery_may_increase_positive_onset_or_SOZ_rank"] is not False:
        raise ValueError("v4 lets late evidence increase onset/SOZ rank")

    findings = config["event_clinical_findings_profile_v2"]
    if findings["legacy_event_findings_v2_replaced"] is not False:
        raise ValueError("v4 unexpectedly replaces the legacy findings wire")
    if not str(findings["status"]).endswith("production_materializer_pending"):
        raise ValueError("v4 findings execution status is not truthful")
    expected_capability_counts = {
        "total": 38,
        "replayable_measurement": 14,
        "research_proxy": 9,
        "unavailable_unqualified": 11,
        "forbidden_non_EEG": 4,
    }
    if findings["capability_registry_counts"] != expected_capability_counts:
        raise ValueError("v4 findings capability counts differ from the frozen registry audit")
    if findings["production_materializer_available"] is not False:
        raise ValueError("v4 claims a Findings production materializer that does not exist")
    if len(set(findings["required_categories"])) != 11:
        raise ValueError("v4 clinical findings profile must have 11 unique categories")
    if findings["Qwen_may_create_or_upgrade_measurements"] is not False:
        raise ValueError("Qwen can improperly create or upgrade native measurements")
    findings_artifacts: dict[str, dict[str, str]] = {}
    for role, relative_text in findings["artifacts"].items():
        relative = Path(str(relative_text))
        artifact_path = (ROOT / relative).resolve(strict=True)
        findings_artifacts[str(role)] = {
            "path": str(relative),
            "sha256": sha256_file(artifact_path),
        }

    claim_gate = config["end_to_end_claim_gate"]
    if any(claim_gate.values()):
        raise ValueError("v4 end-to-end claim gate opened before evidence exists")

    return content_address(
        {
            "schema_version": "clinical_eeg_top_tier_method_architecture_v4_audit_receipt",
            "status": "pass_content_bound_fail_closed_architecture_overlay",
            "config": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256_file(config_path)},
            "bindings": {
                "parent_config": parent,
                "parent_markdown": parent_markdown,
                "common17_independent_retrain_report": retrain_report,
                "adaptive_v1_negative_audit": adaptive_receipt,
                "clinical_findings_profile_v2": findings_artifacts,
            },
            "verified": {
                "common17_and_GT_only_midline_mapping": True,
                "EEG_only_inference_firewall": True,
                "adaptive_v1_preserved_as_negative_result": True,
                "background_consensus_and_target_firewall": True,
                "anchor_relative_search_decoupled_from_support_and_background": True,
                "patient_level_SOZ_promotion_endpoint_not_TERM_timing": True,
                "clinical_findings_profile_additive_and_Qwen_fail_closed": True,
                "end_to_end_claim_gate_closed": True,
            },
            "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    receipt = audit(config_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(receipt) + b"\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
