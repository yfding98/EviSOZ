#!/usr/bin/env python3
"""Audit whether frozen v29 margin orders risk and transports to private EEG.

Thresholds are derived only from the public score distribution at fixed target
coverages and are applied unchanged to the private frozen scores.  No policy
is selected for deployment.  Because both cohorts were historically consumed,
the result is a descriptive negative/qualification audit, not calibration or
clinical risk control.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC = (
    ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
)
DEFAULT_PUBLIC_IDENTITY = (
    ROOT
    / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815/manifest.json"
)
DEFAULT_PRIVATE_REPORTS = (
    ROOT / "outputs/trustworthy_soz_v29_research_reports_v39_20260816/candidate_table.csv"
)
DEFAULT_PRIVATE_ERRORS = (
    ROOT
    / "outputs/trustworthy_soz_private_frozen_publication_v36_20260816/private_event_error_audit.csv"
)
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_v29_margin_transport_v42_20260816"


from src.soz.geometry import STANDARD_19  # noqa: E402
from src.soz.metrics import DEEPSOZ_STANDARD19_NEIGHBORS  # noqa: E402


COVERAGE_TARGETS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260842
LEFT = frozenset(("FP1", "F7", "F3", "T7", "C3", "P7", "P3", "O1"))
RIGHT = frozenset(("FP2", "F4", "F8", "T8", "C4", "P4", "P8", "O2"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _patient_cluster_summary(
    rows: Sequence[Mapping[str, object]],
    metric_names: Sequence[str],
) -> dict[str, object]:
    if not rows:
        return {
            "event_count": 0,
            "patient_count": 0,
            "metrics": {name: None for name in metric_names},
        }
    patient_ids = sorted({str(row["patient_id"]) for row in rows})
    patient_index = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    counts = np.zeros(len(patient_ids), dtype=np.float64)
    sums = {name: np.zeros(len(patient_ids), dtype=np.float64) for name in metric_names}
    for row in rows:
        index = patient_index[str(row["patient_id"])]
        counts[index] += 1.0
        for name in metric_names:
            sums[name][index] += float(row[name])
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    samples = generator.integers(
        0, len(patient_ids), size=(BOOTSTRAP_REPLICATES, len(patient_ids))
    )
    result = {}
    for name in metric_names:
        event_micro = float(sums[name].sum() / counts.sum())
        patient_means = sums[name] / counts
        patient_equal = float(patient_means.mean())
        sampled_sum = sums[name][samples].sum(axis=1)
        sampled_count = counts[samples].sum(axis=1)
        event_bootstrap = sampled_sum / sampled_count
        patient_bootstrap = patient_means[samples].mean(axis=1)
        result[name] = {
            "event_micro": event_micro,
            "patient_equal_event_macro": patient_equal,
            "cluster_event_micro_ci95": [
                float(np.quantile(event_bootstrap, 0.025)),
                float(np.quantile(event_bootstrap, 0.975)),
            ],
            "patient_equal_ci95": [
                float(np.quantile(patient_bootstrap, 0.025)),
                float(np.quantile(patient_bootstrap, 0.975)),
            ],
        }
    return {
        "event_count": len(rows),
        "patient_count": len(patient_ids),
        "metrics": result,
    }


def _accepted_minus_rejected_cluster_bootstrap(
    rows: Sequence[Mapping[str, object]],
    metric_names: Sequence[str],
) -> dict[str, object] | None:
    if not any(bool(row["accepted"]) for row in rows) or not any(
        not bool(row["accepted"]) for row in rows
    ):
        return None
    patient_ids = sorted({str(row["patient_id"]) for row in rows})
    patient_index = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    acc_count = np.zeros(len(patient_ids), dtype=np.float64)
    rej_count = np.zeros(len(patient_ids), dtype=np.float64)
    acc_sum = {name: np.zeros(len(patient_ids), dtype=np.float64) for name in metric_names}
    rej_sum = {name: np.zeros(len(patient_ids), dtype=np.float64) for name in metric_names}
    for row in rows:
        index = patient_index[str(row["patient_id"])]
        if bool(row["accepted"]):
            acc_count[index] += 1.0
            for name in metric_names:
                acc_sum[name][index] += float(row[name])
        else:
            rej_count[index] += 1.0
            for name in metric_names:
                rej_sum[name][index] += float(row[name])
    generator = np.random.default_rng(BOOTSTRAP_SEED + 1)
    samples = generator.integers(
        0, len(patient_ids), size=(BOOTSTRAP_REPLICATES, len(patient_ids))
    )
    result = {}
    for name in metric_names:
        estimate = float(
            acc_sum[name].sum() / acc_count.sum()
            - rej_sum[name].sum() / rej_count.sum()
        )
        sampled_acc_count = acc_count[samples].sum(axis=1)
        sampled_rej_count = rej_count[samples].sum(axis=1)
        valid = (sampled_acc_count > 0) & (sampled_rej_count > 0)
        bootstrap = (
            acc_sum[name][samples].sum(axis=1)[valid] / sampled_acc_count[valid]
            - rej_sum[name][samples].sum(axis=1)[valid] / sampled_rej_count[valid]
        )
        result[name] = {
            "accepted_minus_rejected_event_micro": estimate,
            "patient_cluster_ci95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "valid_bootstrap_replicates": int(valid.sum()),
        }
    return result


def _public_rows(
    directory: Path, identity_path: Path
) -> tuple[list[dict[str, object]], torch.Tensor]:
    tensor_path = (directory / "oof_predictions.safetensors").resolve(strict=True)
    payload = load_file(str(tensor_path), device="cpu")
    identity = json.loads(identity_path.resolve(strict=True).read_text(encoding="utf-8"))
    patient_ids = [str(value) for value in identity["patient_ids"]]
    probability = payload["oof.portable_equal_ensemble_probability"].float()
    targets = payload["targets"].float()
    mask = payload["target_mask"].bool()
    if len(patient_ids) != 102 or tuple(probability.shape) != (102, 19):
        raise ValueError("unexpected public v29 carrier")
    top2 = probability.masked_fill(~mask, -torch.inf).topk(2, dim=1)
    rows = []
    for index, patient_id in enumerate(patient_ids):
        predicted_index = int(top2.indices[index, 0])
        predicted_channel = STANDARD_19[predicted_index]
        positive_indices = set(
            torch.nonzero(
                (targets[index] == 1) & mask[index], as_tuple=False
            ).flatten().tolist()
        )
        acceptable = set(positive_indices)
        if len(positive_indices) <= 4:
            for positive_index in positive_indices:
                acceptable.update(DEEPSOZ_STANDARD19_NEIGHBORS[positive_index])
        positive_channels = {STANDARD_19[value] for value in positive_indices}
        contralateral = bool(
            (positive_channels <= LEFT and predicted_channel in RIGHT)
            or (positive_channels <= RIGHT and predicted_channel in LEFT)
        )
        rows.append(
            {
                "unit_id": patient_id,
                "patient_id": patient_id,
                "margin": float(top2.values[index, 0] - top2.values[index, 1]),
                "strict": float(predicted_index in positive_indices),
                "neighborhood4": float(predicted_index in acceptable),
                "far": float(predicted_index not in acceptable),
                "contralateral_far": float(contralateral and predicted_index not in acceptable),
            }
        )
    return rows, probability


def _private_rows(candidate_path: Path, error_path: Path) -> list[dict[str, object]]:
    candidates = {row["event_id"]: row for row in _csv_rows(candidate_path)}
    errors = _csv_rows(error_path)
    if len(candidates) != 88 or len(errors) != 51:
        raise ValueError("unexpected private margin/error carrier")
    rows = []
    for error in errors:
        candidate = candidates[error["unit_id"]]
        positive = ast.literal_eval(error["positive_channels"])
        spread = ast.literal_eval(error["known_spread_channels"])
        if not isinstance(positive, list) or not isinstance(spread, list):
            raise ValueError("invalid private channel list")
        rows.append(
            {
                "unit_id": error["unit_id"],
                "patient_id": error["patient_id"],
                "margin": float(candidate["top1_top2_margin"]),
                "strict": float(error["strict"]),
                "neighborhood4": 1.0 - float(error["far"]),
                "far": float(error["far"]),
                "contralateral_far": float(error["far_subtype"] == "contralateral_far"),
                "known_spread_top1": float(error["known_spread_top1"]),
            }
        )
    return rows


def run(
    public_directory: Path,
    public_identity_path: Path,
    private_candidate_path: Path,
    private_error_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    public_rows, _ = _public_rows(public_directory, public_identity_path)
    private_rows = _private_rows(private_candidate_path, private_error_path)
    sorted_public_margins = sorted(
        (float(row["margin"]), str(row["unit_id"])) for row in public_rows
    )
    policies = []
    table_rows: list[dict[str, object]] = []
    public_strict_sequence = []
    private_strict_sequence = []
    private_spread_sequence = []
    for target_coverage in COVERAGE_TARGETS:
        if target_coverage == 1.0:
            threshold = -math.inf
        else:
            retain_count = int(math.ceil(len(public_rows) * target_coverage))
            descending = sorted(sorted_public_margins, reverse=True)
            threshold = float(descending[retain_count - 1][0])
        public_policy_rows = [
            {**row, "accepted": bool(float(row["margin"]) >= threshold)}
            for row in public_rows
        ]
        private_policy_rows = [
            {**row, "accepted": bool(float(row["margin"]) >= threshold)}
            for row in private_rows
        ]
        public_accepted_rows = [row for row in public_policy_rows if row["accepted"]]
        private_accepted_rows = [row for row in private_policy_rows if row["accepted"]]
        public_summary = _patient_cluster_summary(
            public_accepted_rows,
            ("strict", "neighborhood4", "far", "contralateral_far"),
        )
        private_summary = _patient_cluster_summary(
            private_accepted_rows,
            (
                "strict",
                "neighborhood4",
                "far",
                "contralateral_far",
                "known_spread_top1",
            ),
        )
        public_difference = _accepted_minus_rejected_cluster_bootstrap(
            public_policy_rows, ("strict", "neighborhood4")
        )
        private_difference = _accepted_minus_rejected_cluster_bootstrap(
            private_policy_rows,
            ("strict", "neighborhood4", "known_spread_top1"),
        )
        public_coverage = len(public_accepted_rows) / len(public_rows)
        private_coverage = len(private_accepted_rows) / len(private_rows)
        policies.append(
            {
                "target_public_coverage": target_coverage,
                "public_score_margin_threshold": (
                    None if not math.isfinite(threshold) else threshold
                ),
                "public_coverage": public_coverage,
                "private_coverage": private_coverage,
                "public_accepted": public_summary,
                "private_accepted": private_summary,
                "public_accepted_minus_rejected": public_difference,
                "private_accepted_minus_rejected": private_difference,
            }
        )
        public_strict = public_summary["metrics"]["strict"]["event_micro"]
        private_strict = private_summary["metrics"]["strict"]["event_micro"]
        private_spread = private_summary["metrics"]["known_spread_top1"]["event_micro"]
        public_strict_sequence.append(public_strict)
        private_strict_sequence.append(private_strict)
        private_spread_sequence.append(private_spread)
        table_rows.extend(
            [
                {
                    "target_public_coverage": target_coverage,
                    "threshold": None if not math.isfinite(threshold) else threshold,
                    "cohort": "public_consumed_development",
                    "actual_coverage": public_coverage,
                    "event_or_patient_count": public_summary["event_count"],
                    "patient_count": public_summary["patient_count"],
                    "strict": public_strict,
                    "neighborhood4": public_summary["metrics"]["neighborhood4"][
                        "event_micro"
                    ],
                    "contralateral_far": public_summary["metrics"][
                        "contralateral_far"
                    ]["event_micro"],
                    "known_spread_top1": None,
                },
                {
                    "target_public_coverage": target_coverage,
                    "threshold": None if not math.isfinite(threshold) else threshold,
                    "cohort": "private_post_open_transport",
                    "actual_coverage": private_coverage,
                    "event_or_patient_count": private_summary["event_count"],
                    "patient_count": private_summary["patient_count"],
                    "strict": private_strict,
                    "neighborhood4": private_summary["metrics"]["neighborhood4"][
                        "event_micro"
                    ],
                    "contralateral_far": private_summary["metrics"][
                        "contralateral_far"
                    ]["event_micro"],
                    "known_spread_top1": private_spread,
                },
            ]
        )

    def _nondecreasing_as_coverage_falls(values: Sequence[float]) -> bool:
        return all(right + 1e-12 >= left for left, right in zip(values, values[1:]))

    result = {
        "schema_version": "trustworthy_soz_v29_margin_risk_transport_v42",
        "status": "NO_CLINICAL_RISK_QUALIFICATION",
        "analysis_role": "posthoc_descriptive_margin_ordering_and_transport_audit",
        "public_patient_count": len(public_rows),
        "private_event_count": len(private_rows),
        "private_patient_count": len({row["patient_id"] for row in private_rows}),
        "policies": policies,
        "qualification_checks": {
            "public_strict_monotone_as_coverage_falls": _nondecreasing_as_coverage_falls(
                public_strict_sequence
            ),
            "private_strict_monotone_as_coverage_falls": _nondecreasing_as_coverage_falls(
                private_strict_sequence
            ),
            "private_known_spread_risk_nonincreasing_as_coverage_falls": all(
                right <= left + 1e-12
                for left, right in zip(private_spread_sequence, private_spread_sequence[1:])
            ),
            "label_fresh_calibration_available": False,
            "private_margin_transport_qualified": False,
        },
        "headline": {
            "public_strict_at_full_coverage": public_strict_sequence[0],
            "public_strict_at_nominal_50pct_coverage": public_strict_sequence[-1],
            "private_strict_at_full_coverage": private_strict_sequence[0],
            "private_strict_at_public_nominal_50pct_threshold": private_strict_sequence[-1],
            "private_actual_coverage_at_public_nominal_50pct_threshold": policies[-1][
                "private_coverage"
            ],
            "private_known_spread_rate_full_coverage": private_spread_sequence[0],
            "private_known_spread_rate_at_public_nominal_50pct_threshold": private_spread_sequence[
                -1
            ],
        },
        "source_files": {
            "public_v29_manifest": str(
                (public_directory / "manifest.json").resolve(strict=True).relative_to(ROOT)
            ),
            "public_v29_manifest_sha256": _sha256(
                (public_directory / "manifest.json").resolve(strict=True)
            ),
            "public_v29_tensor": str(
                (public_directory / "oof_predictions.safetensors")
                .resolve(strict=True)
                .relative_to(ROOT)
            ),
            "public_v29_tensor_sha256": _sha256(
                (public_directory / "oof_predictions.safetensors").resolve(strict=True)
            ),
            "private_candidate_table": str(private_candidate_path.resolve(strict=True).relative_to(ROOT)),
            "private_candidate_table_sha256": _sha256(private_candidate_path.resolve(strict=True)),
            "private_error_audit": str(private_error_path.resolve(strict=True).relative_to(ROOT)),
            "private_error_audit_sha256": _sha256(private_error_path.resolve(strict=True)),
        },
        "access_receipt": {
            "model_training_or_policy_selection_performed": False,
            "public_scores_used_for_thresholds": True,
            "public_labels_used_for_thresholds": False,
            "private_scores_used_for_thresholds": False,
            "private_labels_loaded_for_posthoc_evaluation": True,
            "report_or_prediction_modified": False,
        },
        "interpretation_boundary": {
            "margin_is_calibrated_error_probability": False,
            "risk_guarantee": False,
            "fresh_calibration_or_confirmation": False,
            "public_and_private_historically_opened": True,
            "allowed_claim": (
                "v29 margin shows partial public risk ordering but does not qualify "
                "as a transported private clinical abstention rule"
            ),
        },
    }
    return result, table_rows


def publish(
    output: Path,
    result: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
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
        with (staging / "risk_coverage_table.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
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
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--public-identity", type=Path, default=DEFAULT_PUBLIC_IDENTITY)
    parser.add_argument("--private-candidates", type=Path, default=DEFAULT_PRIVATE_REPORTS)
    parser.add_argument("--private-errors", type=Path, default=DEFAULT_PRIVATE_ERRORS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, rows = run(
        args.public, args.public_identity, args.private_candidates, args.private_errors
    )
    output = publish(args.output, result, rows)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                **result["headline"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
