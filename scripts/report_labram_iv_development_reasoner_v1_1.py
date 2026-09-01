#!/usr/bin/env python3
"""Publish reporting-only channel metrics from saved v1.1 dev outputs.

This command performs no model forward.  The strict fit loader instantiates a
model object and loads its frozen state, while the strict diagnostic loader
loads the complete saved tensor bundle.  Metric computation uses the saved
patient logits together with split-scoped train/dev targets.  This is not a
checkpoint, threshold, calibration, or model-selection stage.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.development_reasoner_training_v1_1 import (  # noqa: E402
    FROZEN_SOURCE_DEV_TARGET_SCOPE_RECEIPT_SHA256,
    FROZEN_SOURCE_DEV_TARGET_TENSOR_FILE_SHA256,
    FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256,
    FROZEN_SOURCE_TRAIN_TARGET_TENSOR_FILE_SHA256,
    load_development_reasoner_dev_diagnostic_v1_1,
    load_development_reasoner_fit_v1_1,
)
from src.soz.development_target_scope_v1_1 import (  # noqa: E402
    load_development_target_scope_v1_1,
)
from src.soz.geometry import STANDARD_19  # noqa: E402
from src.soz.metrics import (  # noqa: E402
    DEEPSOZ_STANDARD19_NEIGHBORS,
    deepsoz_style_top1_metrics,
    patient_localization_metrics,
)


REPORTING_SCHEMA = "soz_labram_iv_channel_reporting_only_v1_1"
FINAL_FIT_MANIFEST_SHA256 = (
    "1cfa17830196d05fe41b39040af3b5be56aeb0d6681d9bab8460a59ca6616940"
)
FINAL_DIAGNOSTIC_MANIFEST_SHA256 = (
    "59e7ab52a4ec912f75e8cbb0ce12b0e5c66bdcc12001704f09be48c97789ef33"
)
FINAL_DIAGNOSTIC_TENSOR_SHA256 = (
    "82db58b08889eccb3afabbefe9aa6efd1de941583b368d3a39ed97b50060f421"
)
FINAL_DIAGNOSTIC_RECEIPT_SHA256 = (
    "8c8f1e09c14e1c687245a49cdab8c72aadcfda0eece2b9613bfc3d123e14351c"
)
FINAL_FIT_RECEIPT_SHA256 = (
    "86e1166d86e9a5ca0cbe535616a9808d8d3a96a759d84697740ab86121d256de"
)
TRAINING_LOADER_CORE_SHA256 = (
    "994df787a88f6369c2a088c93f6359b158bfe2829640b6836435d09a6bdde2c2"
)
SUPERSEDED_DIAGNOSTIC_MANIFEST_SHA256 = (
    "6cd49e68b39b1ccf68127440d952509dfe4bd68a3e47c158712bb558160c2c42"
)
UNSAFE_HISTORICAL_TEST_SOURCE_SHA256 = (
    "4013ee16539b078655927f346d566623483281461f002d459d1158fdec537a46"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _metric_delta(candidate: Mapping[str, object], baseline: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in (
        "macro_average_precision",
        "mean_reciprocal_rank",
        "brier",
        "nll",
    ):
        result[key] = float(candidate[key]) - float(baseline[key])
    for key in ("hit_at_k", "positive_recall_at_k"):
        result[key] = {
            str(k): float(candidate[key][k]) - float(baseline[key][k])
            for k in candidate[key]
        }
    return result


def _top1_delta(
    candidate: Mapping[str, object], baseline: Mapping[str, object]
) -> dict[str, object]:
    result = {
        key: float(candidate[key]) - float(baseline[key])
        for key in (
            "strict_accuracy",
            "relaxed_accuracy",
            "neighbor_only_accuracy_gain",
        )
    }
    candidate_spread = candidate["spread_top1_rate"]
    baseline_spread = baseline["spread_top1_rate"]
    result["spread_top1_rate"] = (
        None
        if candidate_spread is None or baseline_spread is None
        else float(candidate_spread) - float(baseline_spread)
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-artifact",
        type=Path,
        default=ROOT
        / "outputs/labram_iv_development_reasoner_fit_v1_1_final_20260810",
    )
    parser.add_argument(
        "--diagnostic-artifact",
        type=Path,
        default=ROOT
        / "outputs/labram_iv_development_reasoner_dev_diagnostic_v1_1_final_20260810",
    )
    parser.add_argument(
        "--target-scope-root",
        type=Path,
        default=ROOT / "outputs/development_target_scope_v1_1_final_20260810",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(os.path.abspath(args.output))
    if output.name in {"", ".", ".."} or os.path.lexists(output):
        raise FileExistsError(f"Reporting output already exists or is invalid: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("Reporting output parent must be a regular directory")
    input_paths = tuple(
        Path(os.path.abspath(value)).resolve(strict=True)
        for value in (
            args.fit_artifact,
            args.diagnostic_artifact,
            args.target_scope_root,
        )
    )
    resolved_output = output.resolve(strict=False)
    if any(
        resolved_output == source
        or resolved_output in source.parents
        or source in resolved_output.parents
        for source in input_paths
    ):
        raise ValueError("Reporting output/input path topology overlaps")

    fit = load_development_reasoner_fit_v1_1(
        args.fit_artifact,
        expected_manifest_sha256=FINAL_FIT_MANIFEST_SHA256,
    )
    diagnostic = load_development_reasoner_dev_diagnostic_v1_1(
        args.diagnostic_artifact,
        expected_manifest_sha256=FINAL_DIAGNOSTIC_MANIFEST_SHA256,
    )
    train_scope = load_development_target_scope_v1_1(
        args.target_scope_root / "train",
        expected_model_split="source_train",
        expected_receipt_file_sha256=(
            FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256
        ),
    )
    dev_scope = load_development_target_scope_v1_1(
        args.target_scope_root / "dev",
        expected_model_split="source_dev",
        expected_receipt_file_sha256=FROZEN_SOURCE_DEV_TARGET_SCOPE_RECEIPT_SHA256,
    )
    if (
        fit.fit_receipt_sha256 != FINAL_FIT_RECEIPT_SHA256
        or diagnostic.diagnostic_receipt_sha256
        != FINAL_DIAGNOSTIC_RECEIPT_SHA256
        or diagnostic.tensor_file_sha256 != FINAL_DIAGNOSTIC_TENSOR_SHA256
        or diagnostic.run.receipt.fit_manifest_sha256 != fit.manifest_sha256
        or diagnostic.run.receipt.fit_receipt_sha256 != fit.fit_receipt_sha256
        or diagnostic.run.receipt.target_scope_receipt_sha256
        != dev_scope.receipt_file_sha256
    ):
        raise ValueError("Reporting inputs do not share the frozen final lineage")

    train = train_scope.target_batch("source_train")
    dev = dev_scope.target_batch("source_dev", diagnostic.run.patient_ids)
    logits = diagnostic.run.tensors["patient_logits"]
    if logits.requires_grad or tuple(logits.shape) != (16, len(STANDARD_19)):
        raise ValueError("Saved diagnostic logits changed")

    positive = ((train.values == 1) & train.mask).sum(dim=0).float()
    observed = train.mask.sum(dim=0).float()
    prevalence = (positive + 0.5) / (observed + 1.0)
    baseline_logits = torch.logit(prevalence.clamp(1e-6, 1.0 - 1e-6)).repeat(
        len(dev.patient_ids), 1
    )

    candidate_ranking = asdict(
        patient_localization_metrics(
            logits, dev.values, dev.mask, k_values=(1, 3, 5)
        )
    )
    baseline_ranking = asdict(
        patient_localization_metrics(
            baseline_logits, dev.values, dev.mask, k_values=(1, 3, 5)
        )
    )
    candidate_top1 = asdict(
        deepsoz_style_top1_metrics(logits, dev.values, dev.mask)
    )
    baseline_top1 = asdict(
        deepsoz_style_top1_metrics(baseline_logits, dev.values, dev.mask)
    )
    # Public DeepSOZ provides no spread truth.  The metric implementation's
    # numeric placeholder is therefore normalized to JSON null for reporting.
    candidate_top1["spread_top1_rate"] = None
    baseline_top1["spread_top1_rate"] = None
    neighbor_graph = {
        channel: [STANDARD_19[index] for index in DEEPSOZ_STANDARD19_NEIGHBORS[row]]
        for row, channel in enumerate(STANDARD_19)
    }
    script_path = Path(__file__).resolve()
    metrics_path = ROOT / "src/soz/metrics.py"
    training_loader_path = ROOT / "src/soz/development_reasoner_training_v1_1.py"
    if _sha256(training_loader_path) != TRAINING_LOADER_CORE_SHA256:
        raise RuntimeError("Development reasoner training loader/core drifted")
    epoch_rows = fit.run.receipt.epochs
    payload = {
        "schema_version": REPORTING_SCHEMA,
        "purpose": (
            "post_hoc_descriptive_metrics_from_saved_dev_logits_"
            "with_strict_full_artifact_loading_and_no_model_forward"
        ),
        "lineage": {
            "fit_manifest_sha256": fit.manifest_sha256,
            "fit_receipt_sha256": fit.fit_receipt_sha256,
            "checkpoint_file_sha256": fit.checkpoint_file_sha256,
            "diagnostic_manifest_sha256": diagnostic.manifest_sha256,
            "diagnostic_receipt_sha256": diagnostic.diagnostic_receipt_sha256,
            "diagnostic_tensor_file_sha256": diagnostic.tensor_file_sha256,
            "source_train_target_scope_receipt_sha256": train_scope.receipt_file_sha256,
            "source_train_target_tensor_file_sha256": (
                FROZEN_SOURCE_TRAIN_TARGET_TENSOR_FILE_SHA256
            ),
            "source_dev_target_scope_receipt_sha256": dev_scope.receipt_file_sha256,
            "source_dev_target_tensor_file_sha256": (
                FROZEN_SOURCE_DEV_TARGET_TENSOR_FILE_SHA256
            ),
            "saved_patient_logits_sha256": (
                diagnostic.run.receipt.summary.patient_logits_sha256
            ),
            "report_script_sha256": _sha256(script_path),
            "metrics_implementation_sha256": _sha256(metrics_path),
            "training_loader_core_sha256": _sha256(training_loader_path),
        },
        "identities": {
            "patient_ids": list(diagnostic.run.patient_ids),
            "patient_roster_sha256": diagnostic.run.receipt.patient_roster_sha256,
            "channel_order": list(STANDARD_19),
            "channel_order_sha256": diagnostic.run.receipt.channel_order_sha256,
        },
        "reporting_boundary": {
            "model_forward_performed": False,
            "model_object_instantiated_by_strict_fit_loader": True,
            "diagnostic_bundle_fully_loaded": True,
            "candidate_scores_used_by_metrics": "saved_patient_logits",
            "optimizer_instantiated": False,
            "checkpoint_or_threshold_selected": False,
            "calibrator_fitted": False,
            "targets_used_for_reporting_metrics": True,
            "post_hoc_descriptive": True,
            "formal_promotion": False,
            "formal_reasoner_authorized": False,
            "source_eval_used": False,
            "private_used": False,
            "global_unique_dev_forward_claim_valid": False,
            "artifact_auditable_published_diagnostic_forwards": 2,
            "attested_conservative_lower_bound": 6,
            "historical_execution_ledger_complete": False,
            "contemporaneous_attestation_basis": {
                "unsafe_historical_test_source_sha256": (
                    UNSAFE_HISTORICAL_TEST_SOURCE_SHA256
                ),
                "attested_unsafe_test_suite_executions_before_rewrite": 2,
                "real_dev_diagnose_calls_per_unsafe_test_suite_execution": 2,
                "artifact_auditable_published_diagnostic_generations": 2,
                "published_diagnostic_manifest_sha256": [
                    SUPERSEDED_DIAGNOSTIC_MANIFEST_SHA256,
                    FINAL_DIAGNOSTIC_MANIFEST_SHA256,
                ],
                "lower_bound_calculation": "2*2+2=6",
                "limitation": (
                    "the two historical unsafe-test executions are a "
                    "contemporaneous attestation, not a repository-artifact "
                    "ledger; the execution ledger is incomplete, six is a "
                    "conservative lower bound, and the exact count may be higher"
                ),
            },
            "exact_global_real_dev_forward_count_auditable": False,
            "protocol_deviation": (
                "real-dev integration tests and a superseded diagnostic executed "
                "before the final reporting artifact; no result may be used to "
                "retune or select a checkpoint"
            ),
        },
        "metric_definitions": {
            "evaluation_unit": "patient_macro_over_16_source_dev_patients",
            "evaluable_channels": (
                "only channels with target_mask=true for that patient; masked labels "
                "are neither negatives nor denominator entries"
            ),
            "tie_policy": (
                "exact expectation under a uniform random permutation within every "
                "equal-score block; canonical channel order cannot break ties"
            ),
            "macro_average_precision_denominator": (
                "per patient, AP is normalized by that patient's number of observed "
                "positive channels; the reported value is the unweighted mean of 16 APs"
            ),
            "mean_reciprocal_rank_denominator": (
                "per patient, reciprocal rank of the first observed positive with exact "
                "tie expectation; the reported value is the unweighted mean over 16 patients"
            ),
            "hit_at_k": (
                "probability that at least one observed positive is in top-k under exact "
                "tie expectation, then unweighted patient mean"
            ),
            "positive_recall_at_k": (
                "expected observed positive channels in top-k divided by that patient's "
                "observed positive count, then unweighted patient mean"
            ),
            "brier": (
                "for each patient, mean (sigmoid(logit)-label)^2 over channels "
                "with target_mask=true, then unweighted mean over patients; lower "
                "is better; this proper score reflects both discrimination and "
                "calibration and is not a pure calibration test"
            ),
            "nll": (
                "for each patient, mean binary negative log-likelihood over channels "
                "with target_mask=true using the natural logarithm, then unweighted "
                "mean over patients; lower is better; this proper score reflects "
                "both discrimination and calibration and is not a pure calibration test"
            ),
            "deepsoz_one_hop": {
                "role": "sensitivity endpoint only, not a training target or biological equivalence",
                "positive_count_eligibility": "at most four observed positive electrodes",
                "adjacency_direction": "published true-electrode-indexed acceptance table",
                "adjacency": neighbor_graph,
                "adjacency_sha256": _canonical_sha256(neighbor_graph),
                "spread_labels_provided": False,
                "spread_top1_rate_interpretable": False,
            },
            "prevalence_baseline": (
                "per-channel train-only Jeffreys-smoothed prevalence: "
                "p_c=(positive_c+0.5)/(observed_c+1), repeated for all dev patients"
            ),
        },
        "fit_trace": {
            "epoch_0": asdict(epoch_rows[0]),
            "epoch_19": asdict(epoch_rows[-1]),
            "source_train_postfit_diagnostic": asdict(
                fit.run.receipt.source_train_postfit_diagnostic
            ),
            "source_dev_diagnostic": asdict(diagnostic.run.receipt.summary),
        },
        "results": {
            "candidate": {
                "patient_localization": candidate_ranking,
                "deepsoz_style_top1": candidate_top1,
            },
            "train_channel_prevalence_baseline": {
                "patient_localization": baseline_ranking,
                "deepsoz_style_top1": baseline_top1,
            },
            "candidate_minus_baseline": {
                "patient_localization": _metric_delta(
                    candidate_ranking, baseline_ranking
                ),
                "deepsoz_style_top1": _top1_delta(
                    candidate_top1, baseline_top1
                ),
            },
        },
        "concept_semantics": {
            "morphology_present": False,
            "ictal_evidence": (
                "TUSZ scalp-visible bipolar ictal involvement; not SOZ, origin, "
                "exact onset channel, or propagation truth"
            ),
            "temporal_evolution_evidence": (
                "signal-derived physical-channel temporal descriptors; no propagation "
                "labels, exact onset-channel labels, or validated direction"
            ),
        },
    }
    raw = _canonical_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", dir=output.parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, output)
        published = True
        directory_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if not published and temporary.exists():
            temporary.unlink()
    replay = output.read_bytes()
    if replay != raw or _canonical_bytes(json.loads(replay.decode("utf-8"))) != replay:
        raise RuntimeError("Reporting artifact failed canonical strict replay")
    print(
        json.dumps(
            {
                "status": "published_reporting_only_no_model_forward",
                "path": str(output),
                "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                "patient_count": len(dev.patient_ids),
                "post_hoc_descriptive": True,
                "formal_promotion": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
