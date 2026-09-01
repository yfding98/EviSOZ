#!/usr/bin/env python3
"""Evaluate target-blind private v49 raw interventions after materialization.

This stage opens the historically available private significant/spread ledger
only after all 88 x 21 raw intervention predictions exist.  It reports frozen
performance, ranking stability and selected-channel effects relative to four
matched target-blind non-Top-1 channel controls.  It does not select a phase,
intervention, fusion weight, threshold or report wording.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
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

from scripts.audit_labram_v29_token_stress_v38 import _stability  # noqa: E402
from scripts.audit_private_frozen_publication_v36 import (  # noqa: E402
    BOOTSTRAP_SEED,
    _event_rows,
    _paired,
    _read_csv,
    _summary,
)
from scripts.materialize_private_v29_raw_channel_time_interventions_v49 import (  # noqa: E402
    CONTROL_COUNT,
    PHASES,
)
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SCHEMA = "trustworthy_soz_private_v29_raw_channel_time_audit_v49"
DEFAULT_INTERVENTION = (
    ROOT / "outputs/trustworthy_soz_private_v29_raw_channel_time_interventions_v49_20260816"
)
DEFAULT_TARGET = (
    ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/target_ledger.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/trustworthy_soz_private_v29_raw_channel_time_audit_v49_20260816"
)
BOOTSTRAP_REPLICATES = 10_000


def _cluster_mean_interval(
    values: torch.Tensor,
    cluster_ids: Sequence[str],
    *,
    seed: int,
) -> dict[str, object]:
    if values.ndim != 1 or len(values) != len(cluster_ids):
        raise ValueError("effect and cluster arrays differ")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values.tolist(), cluster_ids):
        grouped[str(cluster)].append(float(value))
    patient_means = np.asarray(
        [np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        len(patient_means),
        size=(BOOTSTRAP_REPLICATES, len(patient_means)),
    )
    bootstrap = patient_means[sampled].mean(axis=1)
    return {
        "event_micro_mean": float(values.mean()),
        "patient_equal_mean": float(patient_means.mean()),
        "patient_cluster_bootstrap_ci95": [
            float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
        ],
        "event_count": len(values),
        "patient_count": len(patient_means),
    }


def _table_row(
    *,
    intervention: str,
    summary: Mapping[str, object],
    stability: Mapping[str, float],
) -> dict[str, object]:
    return {
        "intervention": intervention,
        "evaluable_events": summary["event_count"],
        "patient_clusters": summary["patient_count"],
        "strict_event_micro": summary["event_micro"]["strict"],
        "strict_patient_equal": summary["patient_equal_event_macro"]["strict"],
        "neighborhood4_event_micro": summary["event_micro"]["relaxed"],
        "neighborhood4_patient_equal": summary["patient_equal_event_macro"][
            "relaxed"
        ],
        "laterality_patient_equal": summary["patient_equal_event_macro"][
            "laterality_agreement"
        ],
        "far_count": summary["endpoint_counts"]["far"],
        "contralateral_far_count": summary["endpoint_counts"][
            "contralateral_far"
        ],
        "known_spread_top1_count": summary["endpoint_counts"][
            "known_spread_top1_all_enrolled"
        ],
        **stability,
    }


def audit(
    *, intervention_directory: Path, target_path: Path
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    manifest_path = (intervention_directory / "manifest.json").resolve(strict=True)
    tensor_path = (
        intervention_directory / "raw_intervention_predictions.safetensors"
    ).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_target_blind_private_raw_intervention_materialization":
        raise ValueError("formal target-blind raw intervention materialization is missing")
    access = manifest.get("access_receipt", {})
    if access.get("private_target_or_spread_ledger_loaded") is not False or access.get(
        "foundation_or_reasoner_training_performed"
    ) is not False:
        raise ValueError("raw intervention target firewall failed")
    events = manifest.get("events")
    intervention_ids = manifest.get("intervention_ids")
    if not isinstance(events, list) or len(events) != 88 or not isinstance(
        intervention_ids, list
    ):
        raise ValueError("raw intervention roster is malformed")
    index = {str(name): position for position, name in enumerate(intervention_ids)}
    payload = load_file(str(tensor_path), device="cpu")
    probability = payload["probability"].float()
    raw_rms_change = payload["raw_rms_change"].float()
    original_top1 = payload["original_top1_channel"].long()
    if tuple(probability.shape) != (88, len(intervention_ids), 19) or tuple(
        raw_rms_change.shape
    ) != (88, len(intervention_ids)):
        raise ValueError("raw intervention tensor shape changed")
    identity = probability[:, index["identity"]]
    if float((identity.masked_fill(~V11_CANDIDATE_MASK, -torch.inf).argmax(dim=1) != original_top1).float().mean()) != 0.0:
        raise ValueError("stored original Top-1 differs from raw identity replay")

    target_rows = _read_csv(target_path)
    event_rows_by_intervention: dict[str, list[dict[str, object]]] = {}
    summaries: dict[str, object] = {}
    table_rows: list[dict[str, object]] = []
    private_mask = V11_CANDIDATE_MASK.unsqueeze(0).expand(88, -1)
    identity_event_rows = None
    cohort_flow = None
    for name in intervention_ids:
        current_probability = probability[:, index[name]]
        event_rows, flow = _event_rows(
            scores=current_probability,
            events=events,
            target_rows=target_rows,
        )
        if cohort_flow is None:
            cohort_flow = flow
        elif cohort_flow != flow:
            raise RuntimeError("raw intervention changed private evaluation roster")
        if name == "identity":
            identity_event_rows = event_rows
        event_rows_by_intervention[name] = event_rows
        summary = _summary(
            event_rows, seed=BOOTSTRAP_SEED + 100 * index[name]
        )
        summaries[name] = summary
        stability = (
            {
                "top1_retention": 1.0,
                "top3_jaccard": 1.0,
                "mean_absolute_probability_shift": 0.0,
            }
            if name == "identity"
            else _stability(identity, current_probability, private_mask)
        )
        table_rows.append(
            _table_row(intervention=name, summary=summary, stability=stability)
        )
    assert cohort_flow is not None and identity_event_rows is not None

    rows = torch.arange(88, dtype=torch.long)
    original_top1_probability = identity[rows, original_top1]
    all_effects = original_top1_probability.unsqueeze(1) - probability.gather(
        2, original_top1.view(-1, 1, 1).expand(-1, len(intervention_ids), 1)
    ).squeeze(2)
    cluster_ids = [str(event["patient_id"]) for event in events]
    phase_results: dict[str, object] = {}
    effect_rows: list[dict[str, object]] = []
    for phase_index, phase in enumerate(PHASES):
        selected_index = index[f"top1_{phase}_removed"]
        control_indices = torch.tensor(
            [index[f"control{control}_{phase}_removed"] for control in range(CONTROL_COUNT)],
            dtype=torch.long,
        )
        selected_effect = all_effects[:, selected_index]
        control_effects = all_effects.index_select(1, control_indices)
        control_mean_effect = control_effects.mean(dim=1)
        effect_contrast = selected_effect - control_mean_effect
        selected_rms = raw_rms_change[:, selected_index]
        control_rms = raw_rms_change.index_select(1, control_indices).mean(dim=1)
        selected_probability = probability[:, selected_index]
        selected_top1 = selected_probability.masked_fill(
            ~V11_CANDIDATE_MASK, -torch.inf
        ).argmax(dim=1)
        control_top1 = probability.index_select(1, control_indices).masked_fill(
            ~V11_CANDIDATE_MASK.view(1, 1, -1), -torch.inf
        ).argmax(dim=2)
        selected_retention = (selected_top1 == original_top1).float()
        control_retention = (
            control_top1 == original_top1.unsqueeze(1)
        ).float().mean(dim=1)
        paired = _paired(
            event_rows_by_intervention[f"top1_{phase}_removed"],
            identity_event_rows,
            seed=BOOTSTRAP_SEED + 50_000 + 1000 * phase_index,
        )
        phase_results[phase] = {
            "window_sec": manifest["phase_windows_sec"][phase],
            "selected_channel_original_top1_probability_drop": _cluster_mean_interval(
                selected_effect,
                cluster_ids,
                seed=BOOTSTRAP_SEED + 60_000 + phase_index,
            ),
            "matched_control_channel_probability_drop": _cluster_mean_interval(
                control_mean_effect,
                cluster_ids,
                seed=BOOTSTRAP_SEED + 61_000 + phase_index,
            ),
            "selected_minus_matched_control_probability_drop": _cluster_mean_interval(
                effect_contrast,
                cluster_ids,
                seed=BOOTSTRAP_SEED + 62_000 + phase_index,
            ),
            "selected_minus_control_top1_retention": _cluster_mean_interval(
                selected_retention - control_retention,
                cluster_ids,
                seed=BOOTSTRAP_SEED + 63_000 + phase_index,
            ),
            "selected_minus_control_raw_RMS_change": _cluster_mean_interval(
                selected_rms - control_rms,
                cluster_ids,
                seed=BOOTSTRAP_SEED + 64_000 + phase_index,
            ),
            "fraction_selected_effect_larger_than_all_four_controls": float(
                (selected_effect.unsqueeze(1) > control_effects)
                .all(dim=1)
                .float()
                .mean()
            ),
            "selected_intervention_stability_all_88": _stability(
                identity, selected_probability, private_mask
            ),
            "paired_selected_intervention_minus_identity_on_51": paired,
        }
        effect_rows.append(
            {
                "phase": phase,
                "window_sec": str(manifest["phase_windows_sec"][phase]),
                "selected_probability_drop_event_mean": float(selected_effect.mean()),
                "control_probability_drop_event_mean": float(control_mean_effect.mean()),
                "selected_minus_control_patient_equal": phase_results[phase][
                    "selected_minus_matched_control_probability_drop"
                ]["patient_equal_mean"],
                "selected_minus_control_ci_low": phase_results[phase][
                    "selected_minus_matched_control_probability_drop"
                ]["patient_cluster_bootstrap_ci95"][0],
                "selected_minus_control_ci_high": phase_results[phase][
                    "selected_minus_matched_control_probability_drop"
                ]["patient_cluster_bootstrap_ci95"][1],
                "selected_top1_retention": phase_results[phase][
                    "selected_intervention_stability_all_88"
                ]["top1_retention"],
                "selected_strict_event_micro": summaries[
                    f"top1_{phase}_removed"
                ]["event_micro"]["strict"],
                "selected_neighborhood4_event_micro": summaries[
                    f"top1_{phase}_removed"
                ]["event_micro"]["relaxed"],
            }
        )

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_private_post_open_raw_channel_time_reliance_audit",
        "analysis_role": "post_open_descriptive_private_raw_counterfactual_audit",
        "cohort_flow": cohort_flow,
        "identity_summary": summaries["identity"],
        "intervention_summaries": summaries,
        "phase_matched_selected_vs_control": phase_results,
        "materialization_identity_replay_max_absolute_probability_difference": manifest[
            "identity_replay_max_absolute_probability_difference"
        ],
        "source_files": {
            "target_blind_intervention_manifest": str(manifest_path.relative_to(ROOT)),
            "target_blind_intervention_tensor": str(tensor_path.relative_to(ROOT)),
            "opened_private_target_ledger": str(target_path.resolve().relative_to(ROOT)),
        },
        "access_receipt": {
            "all_88_target_blind_interventions_materialized_before_this_audit": True,
            "opened_private_target_loaded_for_descriptive_evaluation": True,
            "raw_EEG_loaded_in_this_evaluation_stage": False,
            "training_model_selection_threshold_or_report_change": False,
        },
        "interpretation_boundary": {
            "model_aligned_level": "preprocessed_raw_scalp_EEG_channel_time_counterfactual_reliance",
            "matched_control_channels_target_blind": True,
            "replacement_may_be_out_of_distribution": True,
            "phase_duration_differs_across_pre_early_late": True,
            "phase_effects_may_be_compared_across_phase_as_equal_dose": False,
            "clinical_onset_propagation_or_biological_causality_validated": False,
            "specific_report_waveform_interval_faithfulness_validated": False,
            "private_is_fresh_validation": False,
            "allowed_claim": (
                "the frozen v29 private ranking shows the reported raw scalp-EEG "
                "channel-time counterfactual reliance relative to matched controls"
            ),
        },
    }
    return result, table_rows, effect_rows


def publish(
    *,
    output: Path,
    result: Mapping[str, object],
    table_rows: Sequence[Mapping[str, object]],
    effect_rows: Sequence[Mapping[str, object]],
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
            ("intervention_summary.csv", table_rows),
            ("phase_effect_summary.csv", effect_rows),
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
    parser.add_argument("--intervention", type=Path, default=DEFAULT_INTERVENTION)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, table, effects = audit(
        intervention_directory=args.intervention, target_path=args.target
    )
    output = publish(
        output=args.output, result=result, table_rows=table, effect_rows=effects
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
