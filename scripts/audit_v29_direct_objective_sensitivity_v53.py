#!/usr/bin/env python3
"""Open private reference after v53 objective predictions are materialized."""

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

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_labram_v29_token_stress_v38 import _stability  # noqa: E402
from scripts.audit_private_frozen_publication_v36 import (  # noqa: E402
    BOOTSTRAP_SEED,
    _event_rows,
    _paired,
    _read_csv,
    _summary,
)
from scripts.materialize_v29_direct_objective_sensitivity_v53 import OBJECTIVES  # noqa: E402
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import (  # noqa: E402
    _evaluate,
    _paired_bootstrap,
)
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SCHEMA = "trustworthy_soz_v29_direct_objective_sensitivity_audit_v53"
DEFAULT_MATERIALIZATION = (
    ROOT / "outputs/trustworthy_soz_v29_direct_objective_sensitivity_v53_20260816"
)
DEFAULT_TARGET = (
    ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/target_ledger.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/trustworthy_soz_v29_direct_objective_sensitivity_audit_v53_20260816"
)


def audit(
    *, materialization_directory: Path, target_path: Path
) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest_path = (materialization_directory / "manifest.json").resolve(strict=True)
    tensor_path = (
        materialization_directory / "objective_predictions.safetensors"
    ).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != (
        "completed_public_oof_and_target_blind_private_objective_sensitivity"
    ):
        raise ValueError("formal target-blind objective materialization is missing")
    if tuple(manifest.get("objectives", ())) != OBJECTIVES:
        raise ValueError("objective grid changed")
    access = manifest.get("access_receipt", {})
    if access.get("private_significant_or_spread_reference_loaded") is not False:
        raise ValueError("objective materialization opened private reference")
    events = manifest["private"]["events"]
    if not isinstance(events, list) or len(events) != 88:
        raise ValueError("private objective roster changed")
    payload = load_file(str(tensor_path), device="cpu")
    targets = payload["public.targets"].float()
    mask = payload["public.target_mask"].bool()
    target_rows = _read_csv(target_path)

    public: dict[str, object] = {}
    private: dict[str, object] = {}
    private_rows: dict[str, list[dict[str, object]]] = {}
    table: list[dict[str, object]] = []
    current_public = payload[
        "public.set_mass_frozen_v28.ensemble_probability"
    ].float()
    current_private = payload[
        "private.set_mass_frozen_v28.ensemble_probability"
    ].float()
    private_mask = V11_CANDIDATE_MASK.unsqueeze(0).expand(88, -1)
    cohort_flow = None
    for objective_index, objective in enumerate(OBJECTIVES):
        public_probability = payload[f"public.{objective}.ensemble_probability"].float()
        private_probability = payload[f"private.{objective}.ensemble_probability"].float()
        public_metrics = _evaluate(
            torch.log(public_probability.clamp_min(1e-12)), targets, mask
        )
        event_rows, flow = _event_rows(
            scores=private_probability, events=events, target_rows=target_rows
        )
        if cohort_flow is None:
            cohort_flow = flow
        elif cohort_flow != flow:
            raise RuntimeError("objective changed private evaluation roster")
        summary = _summary(event_rows, seed=BOOTSTRAP_SEED + 1000 * objective_index)
        private_rows[objective] = event_rows
        public[objective] = {
            "metrics": public_metrics,
            "stability_vs_set_mass": _stability(
                current_public, public_probability, mask
            ),
            "paired_minus_set_mass": (
                None
                if objective == OBJECTIVES[0]
                else _paired_bootstrap(
                    torch.log(public_probability.clamp_min(1e-12)),
                    torch.log(current_public.clamp_min(1e-12)),
                    targets,
                    mask,
                )
            ),
        }
        private[objective] = {
            "summary": summary,
            "stability_vs_set_mass_all_88": _stability(
                current_private, private_probability, private_mask
            ),
        }
        table.append(
            {
                "objective": objective,
                "public_strict": public_metrics["top1"]["strict_accuracy"],
                "public_neighborhood4": public_metrics["top1"]["relaxed_accuracy"],
                "public_macro_ap": public_metrics["ranking"]["macro_average_precision"],
                "private_strict_event_micro": summary["event_micro"]["strict"],
                "private_strict_patient_equal": summary["patient_equal_event_macro"]["strict"],
                "private_neighborhood4_event_micro": summary["event_micro"]["relaxed"],
                "private_neighborhood4_patient_equal": summary[
                    "patient_equal_event_macro"
                ]["relaxed"],
                "private_contralateral_far": summary["endpoint_counts"][
                    "contralateral_far"
                ],
                "private_top1_retention_vs_set_mass": private[objective][
                    "stability_vs_set_mass_all_88"
                ]["top1_retention"],
                "private_top3_jaccard_vs_set_mass": private[objective][
                    "stability_vs_set_mass_all_88"
                ]["top3_jaccard"],
            }
        )
    assert cohort_flow is not None
    for objective in OBJECTIVES[1:]:
        private[objective]["paired_minus_set_mass_on_51"] = _paired(
            private_rows[objective],
            private_rows[OBJECTIVES[0]],
            seed=BOOTSTRAP_SEED + 90_000 + OBJECTIVES.index(objective),
        )

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_post_open_public_private_direct_objective_sensitivity",
        "analysis_role": "frozen_v29_D_branch_objective_audit_no_model_reselection",
        "cohort_flow": cohort_flow,
        "public": public,
        "private": private,
        "source_files": {
            "target_blind_materialization_manifest": str(manifest_path.relative_to(ROOT)),
            "target_blind_materialization_tensor": str(tensor_path.relative_to(ROOT)),
            "opened_private_target_ledger": str(target_path.resolve().relative_to(ROOT)),
        },
        "access_receipt": {
            "all_objective_predictions_materialized_before_private_reference_evaluation": True,
            "private_reference_opened_for_descriptive_evaluation": True,
            "private_used_for_objective_model_threshold_or_report_selection": False,
        },
        "interpretation_boundary": {
            "D_branch_only_not_full_H_objective_ablation": True,
            "set_mass_v29_primary_changed": False,
            "private_is_fresh_validation": False,
            "best_post_open_objective_may_replace_v29": False,
            "allowed_claim": (
                "the frozen v29 H/D ranking has the reported sensitivity to "
                "prespecified D-branch training objectives"
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
        with (staging / "objective_summary.csv").open(
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
        materialization_directory=args.materialization, target_path=args.target
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
