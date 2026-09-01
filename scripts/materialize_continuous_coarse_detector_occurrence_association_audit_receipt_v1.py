#!/usr/bin/env python3
"""Materialize a content-addressed synthetic/mutation audit receipt.

The executable association audit itself lives in the focused pytest module.
This small post-processor accepts its JUnit XML, requires the named safety and
robustness cases to have executed without failure, binds the exact module/test
bytes, and emits a machine-readable claim-bounded receipt.  It does not read
EEG, detector predictions, labels, annotations or clinical material.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.continuous_coarse_detector_occurrence_association_v1 import (  # noqa: E402
    DETECTOR_OCCURRENCE_SCHEMA_VERSION,
    METHOD_ID,
    NATIVE_MEASUREMENT_SCHEMA_VERSION,
    SCHEMA_VERSION,
)


AUDIT_SCHEMA_VERSION = (
    "clinical_eeg_continuous_coarse_detector_occurrence_association_synthetic_mutation_audit_v1"
)
AUDIT_METHOD_ID = (
    "COMMON17-CONTINUOUS-COARSE-DETECTOR-ASSOCIATION-SYNTHETIC-AUDIT-V1"
)
MODULE_PATH = ROOT / (
    "src/clinical_eeg_long_recording/"
    "continuous_coarse_detector_occurrence_association_v1.py"
)
TEST_PATH = ROOT / (
    "tests/"
    "test_clinical_eeg_continuous_coarse_detector_occurrence_association_v1.py"
)
SENTINEL_MODULE_PATH = ROOT / (
    "src/clinical_eeg_long_recording/continuous_coarse_sentinel_cache_v1.py"
)
SENTINEL_TEST_PATH = ROOT / (
    "tests/test_clinical_eeg_continuous_coarse_sentinel_cache_v1.py"
)

REQUIRED_TESTS = {
    "test_public_materializer_has_only_three_target_blind_handoffs",
    "test_later_detector_compatible_candidate_can_beat_earliest_and_all_are_retained",
    "test_segmental_paths_are_single_bout_proposals_and_top_k_is_not_a_fact",
    "test_broad_equal_detector_context_retains_ambiguous_top_k",
    "test_record_qc_neighbor_and_budget_blockers_remain_typed_censors",
    "test_unmeasured_candidate_cannot_become_a_preferred_association",
    "test_late_suffix_mutation_changes_provenance_but_not_association_core",
    "test_point_anchor_jitter_within_frozen_uncertainty_does_not_change_core",
    "test_plain_content_tamper_and_unsafe_input_fields_fail_closed",
    "test_detector_posterior_summary_must_gap_free_cover_the_recording",
    "test_missing_measurement_candidate_and_cross_receipt_binding_fail_closed",
    "test_no_sentinel_candidate_is_explicit_and_deterministic",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _integer_attribute(root: ET.Element, name: str) -> int:
    raw = root.attrib.get(name)
    if raw is None:
        return sum(int(child.attrib.get(name, "0")) for child in root.findall("testsuite"))
    return int(raw)


def materialize(junit_xml: Path) -> dict[str, object]:
    source = junit_xml.resolve(strict=True)
    tree = ET.parse(source)
    root = tree.getroot()
    if root.tag not in {"testsuite", "testsuites"}:
        raise ValueError("association audit JUnit root is invalid")
    tests = _integer_attribute(root, "tests")
    failures = _integer_attribute(root, "failures")
    errors = _integer_attribute(root, "errors")
    skipped = _integer_attribute(root, "skipped")
    cases = root.findall(".//testcase")
    names = {str(row.attrib.get("name", "")).split("[")[0] for row in cases}
    missing = sorted(REQUIRED_TESTS - names)
    failed_case_names = sorted(
        str(row.attrib.get("name", ""))
        for row in cases
        if row.find("failure") is not None or row.find("error") is not None
    )
    if tests < 16 or failures or errors or skipped or missing or failed_case_names:
        raise RuntimeError(
            "association synthetic/mutation test receipt is not a complete pass"
        )
    receipt: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "method_id": AUDIT_METHOD_ID,
        "status": "pass_synthetic_interface_and_mutation_audit",
        "association_binding": {
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "detector_input_schema_version": DETECTOR_OCCURRENCE_SCHEMA_VERSION,
            "native_measurement_input_schema_version": NATIVE_MEASUREMENT_SCHEMA_VERSION,
        },
        "source_files": {
            "association_module": {
                "path": str(MODULE_PATH.relative_to(ROOT)),
                "sha256": _sha256(MODULE_PATH),
            },
            "association_tests": {
                "path": str(TEST_PATH.relative_to(ROOT)),
                "sha256": _sha256(TEST_PATH),
            },
            "sentinel_module": {
                "path": str(SENTINEL_MODULE_PATH.relative_to(ROOT)),
                "sha256": _sha256(SENTINEL_MODULE_PATH),
            },
            "sentinel_tests": {
                "path": str(SENTINEL_TEST_PATH.relative_to(ROOT)),
                "sha256": _sha256(SENTINEL_TEST_PATH),
            },
            "junit_xml": {
                "path": str(source.relative_to(ROOT)),
                "sha256": _sha256(source),
            },
        },
        "test_accounting": {
            "tests": tests,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "required_named_test_count": len(REQUIRED_TESTS),
            "required_named_tests_all_observed": not missing,
            "executed_case_names": sorted(
                str(row.attrib.get("name", "")) for row in cases
            ),
        },
        "audited_contracts": {
            "only_three_target_blind_handoffs": "pass",
            "all_sentinel_candidates_retained": "pass",
            "physically_earliest_candidate_not_unconditional_winner": "pass",
            "single_bout_S0_S1_S2_S3_rule_path_only": "pass",
            "top_K_and_explicit_ambiguity": "pass",
            "record_QC_neighbor_and_budget_typed_censors": "pass",
            "unmeasured_candidate_cannot_become_preferred": "pass",
            "detector_posterior_bins_gap_free_recording_clock_partition": "pass",
            "late_suffix_core_fingerprint_invariance": "pass",
            "point_anchor_jitter_core_fingerprint_invariance": "pass",
            "content_hash_lineage_permission_and_candidate_loss_fail_closed": "pass",
        },
        "permission_boundary": {
            "output_is_occurrence_association_proposal_only": True,
            "finding_or_clinical_term_authorized": False,
            "seizure_or_boundary_fact_authorized": False,
            "channel_region_laterality_or_SOZ_rank_authorized": False,
            "clinical_report_fact_authorized": False,
        },
        "claim_boundary": {
            "synthetic_inputs_only": True,
            "real_EEG_executed_here": False,
            "detector_or_boundary_efficacy_estimated": False,
            "finding_or_SOZ_efficacy_estimated": False,
            "clinical_validation_performed": False,
            "engineering_interface_and_fail_closed_behavior_established": True,
        },
        "package_integration": {
            "module_level___all___exports_present": True,
            "package___init___modified_by_this_audit": False,
            "v5_config_doc_or_architecture_audit_modified": False,
        },
    }
    receipt["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--junit-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize(args.junit_xml)
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite frozen audit receipt: {destination}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

