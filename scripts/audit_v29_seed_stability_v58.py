#!/usr/bin/env python3
"""Open private reference after v58 seed predictions were materialized."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_private_frozen_publication_v36 import (  # noqa: E402
    BOOTSTRAP_SEED,
    _event_rows,
    _paired,
    _read_csv,
    _summary,
)
from scripts.materialize_v29_seed_stability_v58 import SEED_NAMES  # noqa: E402
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import _evaluate  # noqa: E402


SCHEMA = "trustworthy_soz_v29_D_head_seed_stability_audit_v58"
DEFAULT_MATERIALIZATION = (
    ROOT / "outputs/trustworthy_soz_v29_seed_stability_v58_20260816"
)
DEFAULT_TARGET = (
    ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/target_ledger.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/trustworthy_soz_v29_seed_stability_audit_v58_20260816"
)


def _range(values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sd_population": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "range_width": float(array.max() - array.min()),
    }


def audit(
    *, materialization_directory: Path, target_path: Path
) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest_path = (materialization_directory / "manifest.json").resolve(strict=True)
    tensor_path = (
        materialization_directory / "seed_predictions.safetensors"
    ).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_public_OOF_and_target_blind_private_D_seed_stability":
        raise ValueError("formal target-blind v58 materialization is missing")
    if tuple(manifest.get("seed_names", ())) != SEED_NAMES:
        raise ValueError("v58 seed grid changed")
    if manifest.get("access_receipt", {}).get(
        "private_significant_or_spread_reference_loaded"
    ) is not False:
        raise ValueError("v58 materialization opened private reference")
    events = manifest["private"]["events"]
    if not isinstance(events, list) or len(events) != 88:
        raise ValueError("v58 private event roster changed")

    payload = load_file(str(tensor_path), device="cpu")
    public_targets = payload["public.targets"].float()
    public_mask = payload["public.target_mask"].bool()
    target_rows = _read_csv(target_path)
    formal_name = SEED_NAMES[0]

    public: dict[str, object] = {}
    private: dict[str, object] = {}
    private_rows: dict[str, list[dict[str, object]]] = {}
    table: list[dict[str, object]] = []
    cohort_flow = None
    for seed_index, name in enumerate(SEED_NAMES):
        public_probability = payload[f"public.{name}.ensemble_probability"].float()
        private_probability = payload[f"private.{name}.ensemble_probability"].float()
        public_metrics = _evaluate(
            torch.log(public_probability.clamp_min(1e-12)),
            public_targets,
            public_mask,
        )
        event_rows, flow = _event_rows(
            scores=private_probability,
            events=events,
            target_rows=target_rows,
        )
        if cohort_flow is None:
            cohort_flow = flow
        elif cohort_flow != flow:
            raise RuntimeError("seed family changed private evaluation roster")
        summary = _summary(event_rows, seed=BOOTSTRAP_SEED + 10_000 + seed_index)
        public[name] = {"metrics": public_metrics}
        private[name] = {"summary": summary}
        private_rows[name] = event_rows
        table.append(
            {
                "seed_family": name,
                "formal_v29": name == formal_name,
                "public_strict": public_metrics["top1"]["strict_accuracy"],
                "public_neighborhood4": public_metrics["top1"]["relaxed_accuracy"],
                "public_macro_ap": public_metrics["ranking"]["macro_average_precision"],
                "private_strict_event_micro": summary["event_micro"]["strict"],
                "private_strict_patient_equal": summary["patient_equal_event_macro"]["strict"],
                "private_neighborhood4_event_micro": summary["event_micro"]["relaxed"],
                "private_neighborhood4_patient_equal": summary[
                    "patient_equal_event_macro"
                ]["relaxed"],
                "private_laterality_event_micro": summary["event_micro"][
                    "laterality_agreement"
                ],
                "private_contralateral_far_count": summary["endpoint_counts"][
                    "contralateral_far"
                ],
            }
        )
    assert cohort_flow is not None

    for seed_index, name in enumerate(SEED_NAMES[1:], start=1):
        private[name]["paired_minus_formal_v29_on_51"] = _paired(
            private_rows[name],
            private_rows[formal_name],
            seed=BOOTSTRAP_SEED + 80_000 + seed_index,
        )

    public_strict = [float(row["public_strict"]) for row in table]
    public_n4 = [float(row["public_neighborhood4"]) for row in table]
    public_ap = [float(row["public_macro_ap"]) for row in table]
    private_strict = [float(row["private_strict_event_micro"]) for row in table]
    private_n4 = [float(row["private_neighborhood4_event_micro"]) for row in table]
    private_patient_strict = [float(row["private_strict_patient_equal"]) for row in table]
    private_patient_n4 = [float(row["private_neighborhood4_patient_equal"]) for row in table]

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_post_open_public_private_D_seed_stability_audit",
        "analysis_role": "frozen_v29_D_seed_audit_no_model_reselection",
        "cohort_flow": cohort_flow,
        "public": public,
        "private": private,
        "metric_ranges_across_five_seed_families": {
            "public_strict": _range(public_strict),
            "public_neighborhood4": _range(public_n4),
            "public_macro_ap": _range(public_ap),
            "private_strict_event_micro": _range(private_strict),
            "private_neighborhood4_event_micro": _range(private_n4),
            "private_strict_patient_equal": _range(private_patient_strict),
            "private_neighborhood4_patient_equal": _range(private_patient_n4),
        },
        "prediction_stability_before_private_reference": {
            "public": manifest["public"]["prediction_stability"],
            "private_all_88": manifest["private"]["prediction_stability"],
        },
        "source_files": {
            "target_blind_materialization_manifest": str(manifest_path.relative_to(ROOT)),
            "target_blind_materialization_tensor": str(tensor_path.relative_to(ROOT)),
            "opened_private_target_ledger": str(target_path.resolve().relative_to(ROOT)),
        },
        "access_receipt": {
            "all_seed_predictions_materialized_before_private_reference_evaluation": True,
            "private_reference_opened_for_post_open_descriptive_evaluation": True,
            "private_used_for_seed_model_ensemble_threshold_or_report_selection": False,
            "formal_v29_changed": False,
        },
        "interpretation_boundary": {
            "D_head_seed_only": True,
            "H_training_or_foundation_pretraining_stochasticity_audited": False,
            "best_seed_may_replace_or_join_formal_v29": False,
            "private_is_fresh_external_validation": False,
            "allowed_claim": (
                "the low-capacity D-head adaptation and fixed H/D ranking are "
                "stable over the five reported initialization seed families"
            ),
        },
    }
    return result, table


def publish(
    *, output: Path, result: Mapping[str, object], table: Sequence[Mapping[str, object]]
) -> Path:
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
        with (staging / "seed_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--materialization", type=Path, default=DEFAULT_MATERIALIZATION)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, table = audit(
        materialization_directory=args.materialization,
        target_path=args.target,
    )
    output = publish(output=args.output, result=result, table=table)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "private_reference_used_for_selection": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
