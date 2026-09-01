#!/usr/bin/env python3
"""Read-only private-reference audit of v63 H/D partition stability."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_private_frozen_publication_v36 import (  # noqa: E402
    BOOTSTRAP_SEED,
    _event_rows,
    _read_csv,
    _summary,
)
from scripts.audit_raw200_shallow_baseline_v60 import (  # noqa: E402
    _attach_n2,
    _n2_summary,
    _paired_private_metric,
)


SCHEMA = "trustworthy_soz_v29_H_D_partition_stability_private_audit_v64"
DEFAULT_PARTITIONS = ROOT / "outputs/trustworthy_soz_v29_partition_stability_v63_20260816"
DEFAULT_PRIVATE_V29 = ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
DEFAULT_TARGET = ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/target_ledger.csv"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_v29_partition_stability_audit_v64_20260816"


def _range(rows: Sequence[Mapping[str, float]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in rows[0]:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "range": [float(values.min()), float(values.max())],
        }
    return result


def audit(
    *, partition_directory: Path, private_v29_directory: Path, target_path: Path
) -> dict[str, object]:
    manifest = json.loads(
        (partition_directory / "manifest.json").resolve(strict=True).read_text(
            encoding="utf-8"
        )
    )
    if manifest.get("status") != (
        "completed_public_OOF_private_reference_isolated_H_D_partition_stability"
    ):
        raise ValueError("formal v63 partition materialization is missing")
    if manifest.get("access_receipt", {}).get(
        "private_significant_or_spread_reference_loaded"
    ) is not False:
        raise ValueError("v63 materialization opened private reference")
    payload = load_file(
        str((partition_directory / "partition_predictions.safetensors").resolve(strict=True))
    )
    probabilities = payload["private.alternative_partition_probability"].float()
    if tuple(probabilities.shape) != (5, 88, 19):
        raise ValueError("v63 private partition prediction shape changed")
    private_manifest = json.loads(
        (private_v29_directory / "manifest.json").resolve(strict=True).read_text(
            encoding="utf-8"
        )
    )
    events = private_manifest["events"]
    target_rows = _read_csv(target_path)
    formal_probability = payload["private.formal_v29_probability"].float()
    formal_base, formal_flow = _event_rows(
        scores=formal_probability, events=events, target_rows=target_rows
    )
    formal_rows = _attach_n2(formal_base)
    formal_summary = _summary(formal_rows, seed=BOOTSTRAP_SEED)
    formal_n2 = _n2_summary(formal_rows, seed=BOOTSTRAP_SEED + 64)

    partition_results = []
    compact_rows: list[dict[str, float]] = []
    for partition in range(5):
        rows_base, flow = _event_rows(
            scores=probabilities[partition], events=events, target_rows=target_rows
        )
        if flow != formal_flow:
            raise RuntimeError("partition/formal private evaluation rosters differ")
        rows = _attach_n2(rows_base)
        summary = _summary(rows, seed=BOOTSTRAP_SEED + 64_000 + 100 * partition)
        n2 = _n2_summary(rows, seed=BOOTSTRAP_SEED + 64_500 + 100 * partition)
        compact = {
            "strict_event_micro": float(summary["event_micro"]["strict"]),
            "N2_event_micro": float(n2["event_micro"]),
            "N4_event_micro": float(summary["event_micro"]["relaxed"]),
            "laterality_event_micro": float(
                summary["event_micro"]["laterality_agreement"]
            ),
            "strict_patient_equal": float(
                summary["patient_equal_event_macro"]["strict"]
            ),
            "N2_patient_equal": float(n2["patient_equal"]),
            "N4_patient_equal": float(
                summary["patient_equal_event_macro"]["relaxed"]
            ),
            "contralateral_far_count": float(
                summary["endpoint_counts"]["contralateral_far"]
            ),
            "known_spread_top1_count": float(
                summary["endpoint_counts"]["known_spread_top1_all_enrolled"]
            ),
        }
        compact_rows.append(compact)
        partition_results.append(
            {
                "partition": partition,
                "summary": summary,
                "official_N2": n2,
                "compact": compact,
                "paired_formal_minus_partition_patient_equal": {
                    "strict": _paired_private_metric(
                        formal_rows,
                        rows,
                        proposed_key="strict",
                        comparator_key="strict",
                        seed=BOOTSTRAP_SEED + 65_000 + 10 * partition,
                    ),
                    "official_N2": _paired_private_metric(
                        formal_rows,
                        rows,
                        proposed_key="official_N2",
                        comparator_key="official_N2",
                        seed=BOOTSTRAP_SEED + 65_001 + 10 * partition,
                    ),
                    "official_N4": _paired_private_metric(
                        formal_rows,
                        rows,
                        proposed_key="relaxed",
                        comparator_key="relaxed",
                        seed=BOOTSTRAP_SEED + 65_002 + 10 * partition,
                    ),
                },
            }
        )
    formal_compact = {
        "strict_event_micro": float(formal_summary["event_micro"]["strict"]),
        "N2_event_micro": float(formal_n2["event_micro"]),
        "N4_event_micro": float(formal_summary["event_micro"]["relaxed"]),
        "laterality_event_micro": float(
            formal_summary["event_micro"]["laterality_agreement"]
        ),
        "strict_patient_equal": float(
            formal_summary["patient_equal_event_macro"]["strict"]
        ),
        "N2_patient_equal": float(formal_n2["patient_equal"]),
        "N4_patient_equal": float(
            formal_summary["patient_equal_event_macro"]["relaxed"]
        ),
        "contralateral_far_count": float(
            formal_summary["endpoint_counts"]["contralateral_far"]
        ),
        "known_spread_top1_count": float(
            formal_summary["endpoint_counts"]["known_spread_top1_all_enrolled"]
        ),
    }
    return {
        "schema_version": SCHEMA,
        "status": "completed_post_open_private_H_D_partition_stability_audit",
        "cohort_flow": formal_flow,
        "formal_v29": {
            "summary": formal_summary,
            "official_N2": formal_n2,
            "compact": formal_compact,
        },
        "alternative_partitions": partition_results,
        "alternative_partition_metric_distribution": _range(compact_rows),
        "access_receipt": {
            "private_reference_opened_only_after_all_five_partition_predictions": True,
            "private_used_for_partition_model_or_ensemble_selection": False,
            "formal_v29_changed_or_replaced": False,
        },
        "interpretation_boundary": {
            "private_is_fresh_or_external_validation": False,
            "best_partition_may_be_selected": False,
            "stability_proves_foundation_pretraining_clean": False,
            "allowed_claim": "post-open private performance sensitivity to H/D patient partition",
        },
    }


def publish(*, output: Path, result: Mapping[str, object]) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--partitions", type=Path, default=DEFAULT_PARTITIONS)
    parser.add_argument("--private-v29", type=Path, default=DEFAULT_PRIVATE_V29)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit(
        partition_directory=args.partitions,
        private_v29_directory=args.private_v29,
        target_path=args.target,
    )
    output = publish(output=args.output, result=result)
    print(json.dumps({"output": str(output), "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
