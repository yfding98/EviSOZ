#!/usr/bin/env python3
"""Rerun the QC preprocessing pipeline for files listed in error_log.csv."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.qc_report import canonical_config_hash, write_error_log  # noqa: E402
from src.preprocessing.run_preprocess_qc import load_config, load_label_index, run_batch  # noqa: E402


DEFAULT_ERROR_LOG = "outputs/preprocess_qc_all/qc/error_log.csv"
DEFAULT_OUTPUT_DIR = "outputs/preprocess_qc_all"
DEFAULT_CONFIG = "configs/preprocess_qc.yaml"


def read_failed_edf_paths(error_log: str | Path, max_files: int = 0) -> List[Path]:
    """Read unique failed EDF paths from a QC error log."""

    path = Path(error_log)
    if not path.exists():
        raise FileNotFoundError(path)
    out: List[Path] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = str(row.get("status") or "").strip().lower()
            if status and status != "failed":
                continue
            edf_path = str(row.get("edf_path") or "").strip()
            if not edf_path:
                continue
            key = edf_path.replace("\\", "/").lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(Path(edf_path))
            if max_files and len(out) >= int(max_files):
                break
    return out


def merge_label_indexes(label_files: Sequence[str | Path]) -> Dict[str, List[Dict[str, object]]]:
    """Load one or more label files into the index format used by QC."""

    merged: Dict[str, List[Dict[str, object]]] = {}
    for label_file in label_files:
        if not str(label_file).strip():
            continue
        index = load_label_index(label_file)
        for key, events in index.items():
            merged.setdefault(key, []).extend(events)
    return merged


def backup_error_log(error_log: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = error_log.with_name(f"{error_log.stem}.before_rerun_{timestamp}{error_log.suffix}")
    shutil.copy2(error_log, backup)
    return backup


def write_summary(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rerun run_preprocess_qc failures from qc/error_log.csv")
    parser.add_argument("--error_log", default=DEFAULT_ERROR_LOG, help="QC error log produced by run_preprocess_qc")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Same output root used by run_preprocess_qc")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Same preprocess_qc.yaml used by run_preprocess_qc")
    parser.add_argument(
        "--label_file",
        action="append",
        default=[],
        help="Optional label file; may be repeated for TUSZ and private labels",
    )
    parser.add_argument("--max_files", type=int, default=0, help="Optional retry limit for smoke tests")
    parser.add_argument("--no_figures", action="store_true", help="Skip artifact timeline figures during rerun")
    parser.add_argument("--no_overwrite", action="store_true", help="Do not force regeneration of matching outputs")
    parser.add_argument("--no_backup", action="store_true", help="Do not copy the original error_log.csv before rewriting it")
    parser.add_argument("--dry_run", action="store_true", help="Only print the files that would be retried")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    error_log = Path(args.error_log)
    output_dir = Path(args.output_dir)
    all_failed_files = read_failed_edf_paths(error_log)
    failed_files = all_failed_files[: int(args.max_files)] if int(args.max_files) > 0 else all_failed_files
    untried_files = all_failed_files[len(failed_files) :]

    if args.dry_run:
        print(f"would_retry={len(failed_files)}")
        for path in failed_files:
            print(path)
        return 0

    backup_path = ""
    if failed_files and not args.no_backup:
        backup_path = str(backup_error_log(error_log))

    config = load_config(args.config)
    label_index = merge_label_indexes(args.label_file)
    result = run_batch(
        failed_files,
        label_index=label_index,
        output_dir=output_dir,
        config=config,
        config_hash=canonical_config_hash(config),
        config_path=args.config,
        overwrite=not bool(args.no_overwrite),
        make_figures=not bool(args.no_figures) and bool(config.get("outputs", {}).get("make_figures", True)),
    )
    if untried_files:
        combined_errors = list(result.errors)
        combined_errors.extend(
            {
                "edf_path": str(path),
                "status": "failed",
                "error": "not_retried_due_to_max_files",
                "traceback": "",
            }
            for path in untried_files
        )
        write_error_log(output_dir, combined_errors)

    summary = {
        "error_log": str(error_log),
        "backup_error_log": backup_path,
        "output_dir": str(output_dir),
        "config": str(args.config),
        "label_files": list(args.label_file),
        "original_failed_files": len(all_failed_files),
        "retry_input_files": len(failed_files),
        "not_retried_files": len(untried_files),
        "processed": result.processed,
        "skipped": result.skipped,
        "remaining_failed": result.failed + len(untried_files),
        "remaining_error_log": str(output_dir / "qc" / "error_log.csv"),
    }
    write_summary(output_dir / "qc" / "rerun_failed_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["remaining_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
