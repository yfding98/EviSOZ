#!/usr/bin/env python3
"""Audit the current strict VEPiSet main-result lineage.

This checks that the trained branches use the same patient-disjoint split, that
calibration/fusion prediction files remain row-aligned, and that each derived
artifact records the expected validation-only source directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_BASE_RUN = Path("outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_logitadj025_macroselect_noamp20")
DEFAULT_BASE_EXPORT = Path("outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_logitadj025_macroselect_noamp20_region_export")
DEFAULT_BASE_REGION = Path("outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_macroselect_regionfusion_macro_valacc85")
DEFAULT_BASE_SMOOTH = Path("outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_macroselect_regionfusion_smooth_macro_valacc85")
DEFAULT_REGIONCONTRAST_RUN = Path("outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_logitadj025_macroselect_regioncontrast_noamp20")
DEFAULT_REGIONCONTRAST_EXPORT = Path("outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_logitadj025_macroselect_regioncontrast_noamp20_region_export")
DEFAULT_REGIONCONTRAST_REGION = Path("outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_regioncontrast_regionfusion_balanced_valacc85")
DEFAULT_ENSEMBLE_RUN = Path("outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_ensemble_smooth_regioncontrast_macro_valacc85")
DEFAULT_MAIN_RUN = Path("outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_main_patientprior_conservative_macro_valacc87")


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def norm(path: Path | str) -> str:
    return str(Path(path))


def add_failure(failures: List[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def load_prediction_keys(path: Path) -> List[Tuple[str, str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {"patient_id", "path", "target"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return [
        (str(row["patient_id"]), str(row["path"]), str(row["target"]))
        for row in rows
    ]


def compare_prediction_alignment(
    left_dir: Path,
    right_dir: Path,
    splits: Sequence[str] = ("val", "test"),
) -> Dict[str, object]:
    by_split: Dict[str, object] = {}
    aligned = True
    for split in splits:
        left_path = left_dir / f"{split}_predictions.csv"
        right_path = right_dir / f"{split}_predictions.csv"
        left_keys = load_prediction_keys(left_path)
        right_keys = load_prediction_keys(right_path)
        same = left_keys == right_keys
        aligned = aligned and same
        first_mismatch = None
        if not same:
            for idx, (left_key, right_key) in enumerate(zip(left_keys, right_keys)):
                if left_key != right_key:
                    first_mismatch = {
                        "index": idx,
                        "left": left_key,
                        "right": right_key,
                    }
                    break
            if first_mismatch is None and len(left_keys) != len(right_keys):
                first_mismatch = {
                    "index": min(len(left_keys), len(right_keys)),
                    "left_rows": len(left_keys),
                    "right_rows": len(right_keys),
                }
        by_split[split] = {
            "left_rows": len(left_keys),
            "right_rows": len(right_keys),
            "aligned": same,
            "first_mismatch": first_mismatch,
        }
    return {
        "left_dir": str(left_dir),
        "right_dir": str(right_dir),
        "aligned": aligned,
        "splits": by_split,
    }


def extract_split_patients(split_summary: Mapping[str, object]) -> Dict[str, List[str]]:
    for key in ("train_split_meta", "val_split_meta", "test_split_meta"):
        value = split_summary.get(key)
        if isinstance(value, Mapping) and isinstance(value.get("patients"), Mapping):
            patients = value["patients"]
            return {
                "train": sorted(str(item) for item in patients.get("train", [])),
                "val": sorted(str(item) for item in patients.get("val", [])),
                "test": sorted(str(item) for item in patients.get("test", [])),
            }
    raise ValueError("Could not find split patient metadata")


def patient_overlap(split_patients: Mapping[str, Sequence[str]]) -> Dict[str, object]:
    train = set(split_patients.get("train", []))
    val = set(split_patients.get("val", []))
    test = set(split_patients.get("test", []))
    return {
        "train_val": sorted(train & val),
        "train_test": sorted(train & test),
        "val_test": sorted(val & test),
        "has_overlap": bool((train & val) or (train & test) or (val & test)),
    }


def check_same_split(base_run: Path, branch_run: Path) -> Dict[str, object]:
    base_summary = read_json(base_run / "split_summary.json")
    branch_summary = read_json(branch_run / "split_summary.json")
    base_patients = extract_split_patients(base_summary)
    branch_patients = extract_split_patients(branch_summary)
    split_matches = base_patients == branch_patients
    return {
        "base_run": str(base_run),
        "branch_run": str(branch_run),
        "split_matches": split_matches,
        "base_overlap": patient_overlap(base_patients),
        "branch_overlap": patient_overlap(branch_patients),
        "patient_counts": {
            split: {
                "base": len(base_patients.get(split, [])),
                "branch": len(branch_patients.get(split, [])),
            }
            for split in ("train", "val", "test")
        },
    }


def path_field_matches(json_path: Path, field: str, expected: Path) -> Dict[str, object]:
    payload = read_json(json_path)
    observed = str(payload.get(field, ""))
    return {
        "json_path": str(json_path),
        "field": field,
        "expected": str(expected),
        "observed": observed,
        "matches": norm(observed) == norm(expected),
    }


def ensemble_sources_match(json_path: Path, expected_a: Path, expected_b: Path) -> Dict[str, object]:
    payload = read_json(json_path)
    observed_a = str(payload.get("run_dir_a", ""))
    observed_b = str(payload.get("run_dir_b", ""))
    return {
        "json_path": str(json_path),
        "expected_run_dir_a": str(expected_a),
        "observed_run_dir_a": observed_a,
        "run_dir_a_matches": norm(observed_a) == norm(expected_a),
        "expected_run_dir_b": str(expected_b),
        "observed_run_dir_b": observed_b,
        "run_dir_b_matches": norm(observed_b) == norm(expected_b),
    }


def collect_alignment_checks(pairs: Iterable[Tuple[str, Path, Path]]) -> List[Dict[str, object]]:
    checks = []
    for name, left, right in pairs:
        result = compare_prediction_alignment(left, right)
        result["name"] = name
        checks.append(result)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", default=str(DEFAULT_BASE_RUN))
    parser.add_argument("--base-export", default=str(DEFAULT_BASE_EXPORT))
    parser.add_argument("--base-region", default=str(DEFAULT_BASE_REGION))
    parser.add_argument("--base-smooth", default=str(DEFAULT_BASE_SMOOTH))
    parser.add_argument("--regioncontrast-run", default=str(DEFAULT_REGIONCONTRAST_RUN))
    parser.add_argument("--regioncontrast-export", default=str(DEFAULT_REGIONCONTRAST_EXPORT))
    parser.add_argument("--regioncontrast-region", default=str(DEFAULT_REGIONCONTRAST_REGION))
    parser.add_argument("--ensemble-run", default=str(DEFAULT_ENSEMBLE_RUN))
    parser.add_argument("--main-run", default=str(DEFAULT_MAIN_RUN))
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    base_run = Path(args.base_run)
    base_export = Path(args.base_export)
    base_region = Path(args.base_region)
    base_smooth = Path(args.base_smooth)
    regioncontrast_run = Path(args.regioncontrast_run)
    regioncontrast_export = Path(args.regioncontrast_export)
    regioncontrast_region = Path(args.regioncontrast_region)
    ensemble_run = Path(args.ensemble_run)
    main_run = Path(args.main_run)
    output_json = Path(args.output_json) if args.output_json else main_run / "strict_lineage_audit.json"

    failures: List[str] = []
    required_files = [
        base_run / "split_summary.json",
        base_run / "val_predictions.csv",
        base_run / "test_predictions.csv",
        base_export / "export_manifest.json",
        base_export / "val_predictions.csv",
        base_export / "test_predictions.csv",
        base_region / "region_fusion_metrics.json",
        base_smooth / "smoothed_metrics.json",
        regioncontrast_run / "split_summary.json",
        regioncontrast_run / "val_predictions.csv",
        regioncontrast_run / "test_predictions.csv",
        regioncontrast_export / "export_manifest.json",
        regioncontrast_region / "region_fusion_metrics.json",
        ensemble_run / "calibrated_metrics.json",
        main_run / "patient_prior_metrics.json",
        main_run / "val_predictions.csv",
        main_run / "test_predictions.csv",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    failures.extend(f"missing required artifact: {path}" for path in missing)
    if missing:
        audit = {
            "lineage_requirements_met": False,
            "failures": failures,
            "missing_artifacts": missing,
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"output_json": str(output_json), "lineage_requirements_met": False, "failures": failures}, indent=2))
        return 1

    split_check = check_same_split(base_run, regioncontrast_run)
    add_failure(failures, bool(split_check["split_matches"]), "base and region-contrast split summaries differ")
    add_failure(failures, not bool(split_check["base_overlap"].get("has_overlap", True)), "base split has patient overlap")
    add_failure(
        failures,
        not bool(split_check["branch_overlap"].get("has_overlap", True)),
        "region-contrast split has patient overlap",
    )

    source_checks = [
        path_field_matches(base_export / "export_manifest.json", "source_run_dir", base_run),
        path_field_matches(base_region / "region_fusion_metrics.json", "prediction_dir", base_export),
        path_field_matches(base_smooth / "smoothed_metrics.json", "run_dir", base_region),
        path_field_matches(regioncontrast_export / "export_manifest.json", "source_run_dir", regioncontrast_run),
        path_field_matches(regioncontrast_region / "region_fusion_metrics.json", "prediction_dir", regioncontrast_export),
        path_field_matches(main_run / "patient_prior_metrics.json", "run_dir", ensemble_run),
    ]
    ensemble_check = ensemble_sources_match(
        ensemble_run / "calibrated_metrics.json",
        base_smooth,
        regioncontrast_region,
    )
    for check in source_checks:
        add_failure(
            failures,
            bool(check["matches"]),
            f"{check['json_path']} field {check['field']} points to {check['observed']}, expected {check['expected']}",
        )
    add_failure(failures, bool(ensemble_check["run_dir_a_matches"]), "ensemble run_dir_a source mismatch")
    add_failure(failures, bool(ensemble_check["run_dir_b_matches"]), "ensemble run_dir_b source mismatch")

    alignment_checks = collect_alignment_checks([
        ("base_run_to_export", base_run, base_export),
        ("base_export_to_region_fusion", base_export, base_region),
        ("base_region_fusion_to_smoothing", base_region, base_smooth),
        ("regioncontrast_run_to_export", regioncontrast_run, regioncontrast_export),
        ("regioncontrast_export_to_region_fusion", regioncontrast_export, regioncontrast_region),
        ("base_smooth_to_ensemble", base_smooth, ensemble_run),
        ("regioncontrast_region_to_ensemble", regioncontrast_region, ensemble_run),
        ("ensemble_to_main_patient_prior", ensemble_run, main_run),
    ])
    for check in alignment_checks:
        add_failure(failures, bool(check["aligned"]), f"prediction rows are not aligned for {check['name']}")

    audit = {
        "lineage_requirements_met": not failures,
        "base_run": str(base_run),
        "main_run": str(main_run),
        "split_check": split_check,
        "source_checks": source_checks,
        "ensemble_source_check": ensemble_check,
        "alignment_checks": alignment_checks,
        "validation_only_lineage": [
            "checkpoint selected by validation metric in train_vepiset_ied_v2.py",
            "region fusion selected from validation predictions only",
            "patient/window smoothing selected from validation predictions only",
            "ensemble weights and Non-IED bias selected from validation predictions only",
            "conservative patient prior selected from validation predictions only",
            "test predictions are audited after choices are fixed",
        ],
        "failures": failures,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output_json": str(output_json),
        "lineage_requirements_met": audit["lineage_requirements_met"],
        "split_matches": split_check["split_matches"],
        "n_alignment_checks": len(alignment_checks),
        "failures": failures,
    }, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
