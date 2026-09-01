#!/usr/bin/env python3
"""Build the reproducible companion notebook for the SOZ-CrossFilter report."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "experiment_report_20260710"
NOTEBOOK = OUT / "soz_crossfilter_experiment_analysis.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3"}
    nb["cells"] = [
        md(
            """
            # SOZ-CrossFilter experiment audit

            ## tl;dr

            The current evidence supports the graph/TimeFilter path and set refiner as the two main contributors. The motif path is weak and dataset-sensitive. The package is **not yet top-conference ready** because the clean proposed model is not shown to outperform strong baselines under one locked protocol, uncertainty is not trained/calibrated, private data are small, and repeated test-set-guided iteration creates adaptive-overfitting risk.

            This notebook is a read-only, reproducible companion to the technical HTML report. It snapshots the files that exist when the notebook is executed; the v43 private `no_motif` run is still in progress and is never treated as final evidence.
            """
        ),
        md(
            """
            ## Context & Methods

            **Question.** Are the latest SOZ-CrossFilter architecture and experiments strong and trustworthy enough for a top-tier paper, and what should be improved first?

            **Unit of analysis.** Private results use held-out patients (LOPO folds); TUSZ results use seizures in the fixed eval split. Paired ablation uncertainty is estimated by deterministic patient bootstrap over the complete 43-fold v41 runs.

            ### Key Assumptions

            - `base_patient_id` is the correct leakage-control identity.
            - Complete v41 private ablations remain the current clean 43-fold evidence for graph/set-refiner effects.
            - v43 TUSZ is descriptive only because it is a single seed/split and its operating point differs from older baseline comparisons.
            - Confidence intervals quantify patient-fold sampling variation, not uncertainty from random seed, site shift, or adaptive model selection.
            """
        ),
        md("## Data\n\n### 1. Load frozen inputs and define audit helpers"),
        code(
            """
            from pathlib import Path
            import csv, json, math, sys
            import numpy as np
            import pandas as pd

            START = Path.cwd()
            ROOT = next((p for p in [START, *START.parents] if (p / "code" / "soz_crossfilter").is_dir()), None)
            if ROOT is None:
                raise RuntimeError(f"Repository root not found from {START}")
            sys.path.insert(0, str(ROOT))
            if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
                sys.modules.pop("code")

            PATHS = {
                "private_index": ROOT / "outputs/soz_crossfilter/private_rows119_segments_15s/index.csv",
                "private_preprocess": ROOT / "outputs/soz_crossfilter/private_rows119_segments_15s/preprocess_summary.json",
                "v41_root": ROOT / "outputs/soz_crossfilter/pure_v41_private_lopo43_baseline_full",
                "v43_private_root": ROOT / "outputs/soz_crossfilter/pure_v43_fusiononly_private_lopo43_core_ablation",
                "v43_tusz": ROOT / "outputs/soz_crossfilter/pure_v43_fusiononly_tusz_3epoch_core_ablation/ablation_summary.json",
                "baseline_comparison": ROOT / "outputs/soz_crossfilter/baseline_comparison_multiregion_full/baseline_comparison_summary.json",
            }

            def load_json(path):
                with Path(path).open("r", encoding="utf-8") as handle:
                    return json.load(handle)

            def load_lopo(root, variant):
                return load_json(Path(root) / variant / "private_lopo" / "lopo_summary.json")

            def fold_frame(summary, metrics):
                rows = []
                for fold in summary["folds"]:
                    if fold.get("status") != "ok":
                        continue
                    row = {"patient": fold["patient"]}
                    row.update({m: float(fold[m]) for m in metrics})
                    rows.append(row)
                return pd.DataFrame(rows).set_index("patient").sort_index()

            def bootstrap_mean(values, n_boot=20000, seed=20260710):
                values = np.asarray(values, dtype=float)
                rng = np.random.default_rng(seed)
                indices = rng.integers(0, len(values), size=(n_boot, len(values)))
                draws = values[indices].mean(axis=1)
                return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))

            print({name: str(path.relative_to(ROOT)) for name, path in PATHS.items()})
            """
        ),
        md("### 2. Check private-data grain, completeness, duplication, and label density"),
        code(
            """
            private = pd.read_csv(PATHS["private_index"], encoding="utf-8-sig")
            prep = load_json(PATHS["private_preprocess"])
            sample_files_exist = private["npz_path"].map(lambda p: (PATHS["private_index"].parent / p).is_file())
            exact_window_duplicates = private.duplicated(
                ["base_patient_id", "resolved_path", "absolute_window_start", "absolute_window_end"],
                keep=False,
            )
            patient_counts = private.groupby("base_patient_id").size()

            quality = pd.DataFrame(
                [
                    ("rows", len(private)),
                    ("patients", private["base_patient_id"].nunique()),
                    ("missing sample files", int((~sample_files_exist).sum())),
                    ("duplicate sample_id", int(private["sample_id"].duplicated().sum())),
                    ("duplicate patient/file/window rows", int(exact_window_duplicates.sum())),
                    ("median samples per patient", float(patient_counts.median())),
                    ("min / max samples per patient", f"{patient_counts.min()} / {patient_counts.max()}"),
                    ("mean positive channels", float(private["n_positive"].mean())),
                    ("mean positive regions (of 5)", float(private["n_positive_regions"].mean())),
                    ("rows with all 32 channels", int((private["n_available_channels"] == 32).sum())),
                ],
                columns=["check", "value"],
            )
            display(quality)
            print("preprocess:", {k: prep[k] for k in ["rows_written", "rows_failed", "target_sfreq", "pre_sec", "onset_sec", "post_sec", "label_rule"]})
            """
        ),
        md("### 3. Quantify how multi-positive labels inflate Top-k any-hit metrics"),
        code(
            """
            def random_any_hit(n, positives, k):
                positives = int(positives)
                n = int(n)
                k = min(int(k), n)
                if positives <= 0:
                    return 0.0
                if n - positives < k:
                    return 1.0
                return 1.0 - math.comb(n - positives, k) / math.comb(n, k)

            chance = pd.DataFrame(
                {
                    "metric": ["region Top-1", "region Top-3", "channel Top-1", "channel Top-5"],
                    "random_any_hit": [
                        np.mean([random_any_hit(5, m, 1) for m in private["n_positive_regions"]]),
                        np.mean([random_any_hit(5, m, 3) for m in private["n_positive_regions"]]),
                        np.mean([random_any_hit(n, m, 1) for n, m in zip(private["n_available_channels"], private["n_positive"])]),
                        np.mean([random_any_hit(n, m, 5) for n, m in zip(private["n_available_channels"], private["n_positive"])]),
                    ],
                }
            )
            chance["random_any_hit"] = chance["random_any_hit"].round(4)
            display(chance)
            """
        ),
        md("## Results\n\n### 4. Snapshot complete and in-progress v43 results"),
        code(
            """
            KEY_METRICS = {
                "region_compact_f1": "test_region_compact_f1",
                "region_top1": "test_region_top1",
                "region_auroc": "test_region_auroc",
                "region_auprc_lift": "test_region_auprc_lift",
                "channel_compact_f1": "test_channel_compact_f1",
                "channel_constrained_f1": "test_channel_region_constrained_compact_f1",
                "channel_top1": "test_channel_top1",
                "channel_top5": "test_channel_top5",
                "channel_auroc": "test_channel_auroc",
                "channel_auprc_lift": "test_channel_auprc_lift",
                "channel_avg_predicted": "test_channel_compact_avg_predicted",
            }

            v43_baseline = load_lopo(PATHS["v43_private_root"], "baseline")
            v43_nomotif = load_lopo(PATHS["v43_private_root"], "no_motif") if (PATHS["v43_private_root"] / "no_motif/private_lopo/lopo_summary.json").is_file() else None

            rows = []
            for variant, summary, status in [
                ("v43 baseline", v43_baseline, "complete"),
                ("v43 no_motif", v43_nomotif, "in progress"),
            ]:
                if summary is None:
                    continue
                agg = summary["aggregate"]
                row = {"variant": variant, "status": status, "folds": f"{agg['folds_ok']}/{43}"}
                for label, key in KEY_METRICS.items():
                    row[label] = agg.get(f"macro_{key}")
                rows.append(row)
            v43_private_snapshot = pd.DataFrame(rows)
            display(v43_private_snapshot.round(4))

            tusz_rows = load_json(PATHS["v43_tusz"])["rows"]
            tusz_snapshot = pd.DataFrame(tusz_rows)
            tusz_table = pd.DataFrame({"variant": tusz_snapshot["variant"]})
            for label, key in KEY_METRICS.items():
                tusz_table[label] = tusz_snapshot[key] if key in tusz_snapshot else np.nan
            display(tusz_table.round(4))
            """
        ),
        md("### 5. Bootstrap patient-level performance uncertainty for the complete v43 private baseline"),
        code(
            """
            ci_metrics = [
                "test_region_compact_f1", "test_region_top1", "test_region_auroc",
                "test_channel_compact_f1", "test_channel_top1", "test_channel_top5", "test_channel_auroc",
            ]
            v43_frame = fold_frame(v43_baseline, ci_metrics)
            ci_rows = []
            for metric in ci_metrics:
                mean, lo, hi = bootstrap_mean(v43_frame[metric].to_numpy())
                ci_rows.append({"metric": metric.removeprefix("test_"), "mean": mean, "ci95_low": lo, "ci95_high": hi, "n_patients": len(v43_frame)})
            v43_ci = pd.DataFrame(ci_rows)
            display(v43_ci.round(4))
            """
        ),
        md("### 6. Recompute paired v41 ablation effects with patient bootstrap"),
        code(
            """
            paired_metrics = [
                "test_region_compact_f1", "test_region_top1", "test_region_auroc",
                "test_channel_compact_f1", "test_channel_region_constrained_compact_f1",
                "test_channel_top1", "test_channel_top5", "test_channel_auroc", "test_channel_auprc_lift",
            ]
            v41 = {variant: load_lopo(PATHS["v41_root"], variant) for variant in ["baseline", "no_motif", "no_graph_filter", "no_set_refiner"]}
            v41_frames = {name: fold_frame(summary, paired_metrics) for name, summary in v41.items()}

            paired_rows = []
            for ablation in ["no_motif", "no_graph_filter", "no_set_refiner"]:
                common = v41_frames["baseline"].index.intersection(v41_frames[ablation].index)
                for metric in paired_metrics:
                    diff = v41_frames["baseline"].loc[common, metric] - v41_frames[ablation].loc[common, metric]
                    mean, lo, hi = bootstrap_mean(diff.to_numpy(), seed=20260710 + len(paired_rows))
                    paired_rows.append(
                        {
                            "ablation": ablation,
                            "metric": metric.removeprefix("test_"),
                            "delta_full_minus_ablation": mean,
                            "ci95_low": lo,
                            "ci95_high": hi,
                            "paired_wins": int((diff > 1e-12).sum()),
                            "ties": int((diff.abs() <= 1e-12).sum()),
                            "losses": int((diff < -1e-12).sum()),
                            "n": len(common),
                        }
                    )
            paired = pd.DataFrame(paired_rows)
            focus = paired[paired["metric"].isin(["channel_compact_f1", "channel_top1", "channel_top5", "channel_auroc", "channel_auprc_lift"])]
            display(focus.round(4))
            """
        ),
        md("### 7. Compare the ongoing v43 motif run only on overlapping patients"),
        code(
            """
            if v43_nomotif is not None:
                overlap_metrics = ["test_region_compact_f1", "test_region_auroc", "test_channel_compact_f1", "test_channel_top5", "test_channel_auroc", "test_channel_auprc_lift"]
                base_overlap = fold_frame(v43_baseline, overlap_metrics)
                motif_overlap = fold_frame(v43_nomotif, overlap_metrics)
                common = base_overlap.index.intersection(motif_overlap.index)
                overlap_rows = []
                for metric in overlap_metrics:
                    diff = base_overlap.loc[common, metric] - motif_overlap.loc[common, metric]
                    mean, lo, hi = bootstrap_mean(diff.to_numpy(), n_boot=10000, seed=20260801 + len(overlap_rows))
                    overlap_rows.append({"metric": metric.removeprefix("test_"), "delta_full_minus_no_motif": mean, "ci95_low": lo, "ci95_high": hi, "n_overlap": len(common)})
                display(pd.DataFrame(overlap_rows).round(4))
            else:
                print("v43 no_motif has no summary yet")
            """
        ),
        md("### 8. Audit cross-model comparability and protocol drift"),
        code(
            """
            baseline_rows = pd.DataFrame(load_json(PATHS["baseline_comparison"])["rows"])
            private_baselines = baseline_rows[baseline_rows["dataset"] == "private"].copy()
            comparison = private_baselines[[
                "model", "folds_ok", "macro_test_region_compact_f1", "macro_test_channel_compact_f1",
                "macro_test_region_top1", "macro_test_channel_top1", "macro_test_region_auroc", "macro_test_channel_auroc",
                "macro_test_channel_compact_avg_predicted",
            ]]
            display(comparison.round(4))

            example_config = load_json(PATHS["v43_private_root"] / "baseline/private_lopo/刘娟/run_config.json")["config"]
            protocol_drift = pd.DataFrame(
                [
                    ("proposed v43", example_config["epochs"], example_config["batch_size"], example_config["lr"], example_config["selection_metric"], example_config["policy_family"], example_config["channel_compact_max_labels"]),
                    ("baseline runner", 30, 8, 2e-4, "region_compact_f1", "top1_augmented", 5.0),
                ],
                columns=["route", "epochs", "batch_size", "lr", "selection_metric", "policy_family", "channel_max_labels"],
            )
            display(protocol_drift)
            """
        ),
        md("### 9. Measure architecture and experiment-search complexity"),
        code(
            """
            from code.soz_crossfilter.model import SOZCrossFilter

            model = SOZCrossFilter(
                emb_dim=96, depth=3, num_heads=4, patch_samples=200, codebook_size=128,
                use_motif=True, use_topology_prior=True, use_graph_filter=True,
                use_motif_prediction=True, use_compact_calibrator=False, use_set_refiner=True,
                motif_scale_init=-8.0, topology_mix_init=-1.5, graph_residual_init=-1.0,
            )
            experiment_dirs = [p for p in (ROOT / "outputs/soz_crossfilter").iterdir() if p.is_dir()]
            search_dirs = [p for p in experiment_dirs if p.name.startswith(("_smoke", "_probe", "_tune"))]
            architecture = pd.DataFrame(
                [
                    ("trainable parameters", sum(p.numel() for p in model.parameters() if p.requires_grad)),
                    ("PyTorch module objects", sum(1 for _ in model.modules())),
                    ("top-level experiment directories", len(experiment_dirs)),
                    ("smoke/probe/tune directories", len(search_dirs)),
                    ("direct motif residual at initialization", float(np.log1p(np.exp(-8.0)))),
                    ("topology mix at initialization", float(1 / (1 + np.exp(1.5)))),
                    ("graph residual gate at initialization", float(1 / (1 + np.exp(1.0)))),
                ],
                columns=["measure", "value"],
            )
            display(architecture)
            """
        ),
        md(
            """
            ## Takeaways

            1. **Strongest verified module evidence:** graph/TimeFilter and set refiner; their complete 43-patient paired effects are large on channel AUROC, AUPRC lift, compact F1, and Top-k.
            2. **Weak or unstable evidence:** motif and topology prior. Motif changes sign across datasets/metrics; topology effects are small.
            3. **Main validity risks:** only 119 private seizures from 43 patients, high multi-positive label density, one TUSZ seed/split, repeated use of test outcomes during architecture search, and non-comparable operating points across baselines.
            4. **Architecture risk:** the 3.7M-parameter model has many interacting gates and auxiliary losses, but several named modules are weakly identified by ablation. The Beta uncertainty head is not trained by `crossfilter_loss`, so uncertainty claims must be removed until calibrated.
            5. **Submission gate:** lock one protocol, rerun the clean model and all baselines under that protocol across multiple seeds, add patient-bootstrap/seed intervals and calibration, and obtain a truly external patient-level test before claiming top-tier readiness.
            """
        ),
    ]
    nbf.write(nb, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
