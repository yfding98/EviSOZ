#!/usr/bin/env python3
"""Evaluate the frozen private v29 raw acquisition robustness grid.

The private reference is opened only after all 88 target-blind predictions are
materialized.  Every prespecified condition is reported; no condition, model,
threshold, input-validity gate, or report wording is selected from these
post-open results.
"""

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
from scripts.audit_private_v29_raw_channel_time_interventions_v49 import (  # noqa: E402
    _cluster_mean_interval,
    _table_row,
)
from scripts.materialize_private_v29_raw_acquisition_robustness_v52 import (  # noqa: E402
    AMPLITUDE_SCALES,
    ANCHOR_SHIFTS_SEC,
    CANDIDATE_CHANNELS,
    CANDIDATE_INDICES,
    _number_id,
    _scale_id,
)
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SCHEMA = "trustworthy_soz_private_v29_raw_acquisition_robustness_audit_v52"
DEFAULT_MATERIALIZATION = (
    ROOT / "outputs/trustworthy_soz_private_v29_raw_acquisition_robustness_v52_20260816"
)
DEFAULT_TARGET = (
    ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/target_ledger.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/trustworthy_soz_private_v29_raw_acquisition_robustness_audit_v52_20260816"
)


def audit(
    *, materialization_directory: Path, target_path: Path
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    manifest_path = (materialization_directory / "manifest.json").resolve(strict=True)
    tensor_path = (
        materialization_directory / "raw_acquisition_robustness_predictions.safetensors"
    ).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_target_blind_private_raw_acquisition_robustness":
        raise ValueError("formal target-blind acquisition robustness grid is missing")
    access = manifest.get("access_receipt", {})
    if access.get("private_target_or_spread_ledger_loaded") is not False or access.get(
        "foundation_or_reasoner_training_performed"
    ) is not False:
        raise ValueError("raw acquisition target firewall failed")
    events = manifest.get("events")
    condition_ids = manifest.get("condition_ids")
    if not isinstance(events, list) or len(events) != 88 or not isinstance(
        condition_ids, list
    ):
        raise ValueError("raw acquisition robustness roster is malformed")
    condition_index = {str(name): index for index, name in enumerate(condition_ids)}
    payload = load_file(str(tensor_path), device="cpu")
    probability = payload["probability"].float()
    if tuple(probability.shape) != (88, len(condition_ids), 19):
        raise ValueError("raw acquisition probability tensor shape changed")
    identity = probability[:, condition_index["identity"]]
    private_mask = V11_CANDIDATE_MASK.unsqueeze(0).expand(88, -1)
    original_top1 = identity.masked_fill(~private_mask, -torch.inf).argmax(dim=1)

    target_rows = _read_csv(target_path)
    event_rows_by_condition: dict[str, list[dict[str, object]]] = {}
    summaries: dict[str, object] = {}
    condition_rows: list[dict[str, object]] = []
    identity_event_rows = None
    cohort_flow = None
    for name in condition_ids:
        current = probability[:, condition_index[name]]
        event_rows, flow = _event_rows(scores=current, events=events, target_rows=target_rows)
        if cohort_flow is None:
            cohort_flow = flow
        elif cohort_flow != flow:
            raise RuntimeError("robustness condition changed private evaluation roster")
        if name == "identity":
            identity_event_rows = event_rows
        event_rows_by_condition[name] = event_rows
        summary = _summary(event_rows, seed=BOOTSTRAP_SEED + 100 * condition_index[name])
        summaries[name] = summary
        stability = (
            {
                "top1_retention": 1.0,
                "top3_jaccard": 1.0,
                "mean_absolute_probability_shift": 0.0,
            }
            if name == "identity"
            else _stability(identity, current, private_mask)
        )
        condition_rows.append(
            _table_row(intervention=str(name), summary=summary, stability=stability)
        )
    assert cohort_flow is not None and identity_event_rows is not None

    fixed_conditions: dict[str, object] = {}
    for name in condition_ids:
        if name.startswith("drop_") or name == "identity":
            continue
        fixed_conditions[name] = {
            "summary": summaries[name],
            "stability_all_88": _stability(
                identity, probability[:, condition_index[name]], private_mask
            ),
            "paired_minus_identity_on_51": _paired(
                event_rows_by_condition[name],
                identity_event_rows,
                seed=BOOTSTRAP_SEED + 30_000 + condition_index[name],
            ),
        }

    drop_condition_indices = torch.tensor(
        [condition_index[f"drop_{channel}"] for channel in CANDIDATE_CHANNELS],
        dtype=torch.long,
    )
    drop_probability = probability.index_select(1, drop_condition_indices)
    drop_top1 = drop_probability.masked_fill(
        ~V11_CANDIDATE_MASK.view(1, 1, -1), -torch.inf
    ).argmax(dim=2)
    candidate_position = {int(channel): index for index, channel in enumerate(CANDIDATE_INDICES)}
    selected_drop_position = torch.tensor(
        [candidate_position[int(channel)] for channel in original_top1], dtype=torch.long
    )
    rows = torch.arange(88, dtype=torch.long)
    selected_drop_probability = drop_probability[rows, selected_drop_position]
    selected_event_rows, selected_flow = _event_rows(
        scores=selected_drop_probability, events=events, target_rows=target_rows
    )
    if selected_flow != cohort_flow:
        raise RuntimeError("dynamic original-Top1 dropout changed evaluation roster")
    selected_summary = _summary(selected_event_rows, seed=BOOTSTRAP_SEED + 80_000)

    identity_selected_score = identity[rows, original_top1]
    selected_score_after_drop = selected_drop_probability[rows, original_top1]
    selected_score_drop = identity_selected_score - selected_score_after_drop
    all_selected_scores_after_drop = drop_probability.gather(
        2, original_top1.view(-1, 1, 1).expand(-1, len(CANDIDATE_CHANNELS), 1)
    ).squeeze(2)
    all_score_drop = identity_selected_score.unsqueeze(1) - all_selected_scores_after_drop
    control_mask = torch.ones_like(all_score_drop, dtype=torch.bool)
    control_mask[rows, selected_drop_position] = False
    nonselected_mean_drop = all_score_drop[control_mask].reshape(88, -1).mean(dim=1)
    selected_retention = (drop_top1[rows, selected_drop_position] == original_top1).float()
    nonselected_retention = (
        (drop_top1 == original_top1.unsqueeze(1)).float()[control_mask].reshape(88, -1).mean(dim=1)
    )
    cluster_ids = [str(event["patient_id"]) for event in events]

    dropout_rows: list[dict[str, object]] = []
    for channel in CANDIDATE_CHANNELS:
        name = f"drop_{channel}"
        table = next(row for row in condition_rows if row["intervention"] == name)
        dropout_rows.append({"channel": channel, **table})

    exhaustive_dropout = {
        "dynamic_original_top1_dropout_summary": selected_summary,
        "dynamic_original_top1_dropout_stability_all_88": _stability(
            identity, selected_drop_probability, private_mask
        ),
        "dynamic_original_top1_dropout_minus_identity_on_51": _paired(
            selected_event_rows,
            identity_event_rows,
            seed=BOOTSTRAP_SEED + 81_000,
        ),
        "original_top1_probability_drop": _cluster_mean_interval(
            selected_score_drop, cluster_ids, seed=BOOTSTRAP_SEED + 82_000
        ),
        "nonselected_channel_mean_probability_drop": _cluster_mean_interval(
            nonselected_mean_drop, cluster_ids, seed=BOOTSTRAP_SEED + 83_000
        ),
        "selected_minus_nonselected_probability_drop": _cluster_mean_interval(
            selected_score_drop - nonselected_mean_drop,
            cluster_ids,
            seed=BOOTSTRAP_SEED + 84_000,
        ),
        "selected_minus_nonselected_top1_retention": _cluster_mean_interval(
            selected_retention - nonselected_retention,
            cluster_ids,
            seed=BOOTSTRAP_SEED + 85_000,
        ),
        "fraction_selected_drop_larger_than_all_17_nonselected": float(
            (
                selected_score_drop.unsqueeze(1)
                > all_score_drop.masked_fill(~control_mask, -torch.inf)
            )
            .all(dim=1)
            .float()
            .mean()
        ),
    }

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_private_post_open_raw_acquisition_robustness_audit",
        "analysis_role": "post_open_descriptive_private_frozen_model_robustness",
        "cohort_flow": cohort_flow,
        "identity_summary": summaries["identity"],
        "anchor_and_amplitude_conditions": fixed_conditions,
        "exhaustive_single_candidate_channel_dropout": exhaustive_dropout,
        "condition_summaries": summaries,
        "materialization_identity_replay_max_absolute_probability_difference": manifest[
            "identity_replay_max_absolute_probability_difference"
        ],
        "source_files": {
            "target_blind_materialization_manifest": str(manifest_path.relative_to(ROOT)),
            "target_blind_materialization_tensor": str(tensor_path.relative_to(ROOT)),
            "opened_private_target_ledger": str(target_path.resolve().relative_to(ROOT)),
        },
        "access_receipt": {
            "all_88_target_blind_conditions_materialized_before_reference_evaluation": True,
            "opened_private_target_loaded_for_descriptive_evaluation": True,
            "training_model_selection_threshold_input_gate_or_report_change": False,
        },
        "interpretation_boundary": {
            "private_is_fresh_validation": False,
            "post_open_results_may_define_a_clinical_robustness_threshold": False,
            "channel_replacement_may_be_out_of_distribution": True,
            "anchor_shift_is_not_clinical_onset_label_uncertainty": True,
            "amplitude_scaling_is_sensitivity_not_device_validation": True,
            "allowed_claim": (
                "the frozen v29 model has the reported post-open private ranking "
                "sensitivity under a prespecified raw acquisition perturbation grid"
            ),
        },
    }
    return result, condition_rows, dropout_rows


def publish(
    *,
    output: Path,
    result: Mapping[str, object],
    condition_rows: Sequence[Mapping[str, object]],
    dropout_rows: Sequence[Mapping[str, object]],
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
        for filename, rows in (
            ("condition_summary.csv", condition_rows),
            ("single_channel_dropout_summary.csv", dropout_rows),
        ):
            with (staging / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
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
    result, condition_rows, dropout_rows = audit(
        materialization_directory=args.materialization, target_path=args.target
    )
    output = publish(
        output=args.output,
        result=result,
        condition_rows=condition_rows,
        dropout_rows=dropout_rows,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "evaluable_events": result["identity_summary"]["event_count"],
                "training_performed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
