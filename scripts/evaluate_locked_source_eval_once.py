#!/usr/bin/env python3
"""Consume one source-eval release and evaluate sealed anchor/v9 predictions.

This command is intentionally fail-closed.  It first strictly replays the
target-free prediction artifact and its 185-event roster.  It then creates an
append-only release ledger with ``O_CREAT|O_EXCL``.  Only after that durable
consumption record exists may the verified DeepSOZ target-v2 loader run.

Never use this command for model selection, threshold tuning, or retries.  A
failed post-ledger attempt remains a consumed evaluation attempt.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.deepsoz_target_v2 import (  # noqa: E402
    TARGET_V2_POLICY_SHA256,
    load_verified_deepsoz_target_v2_artifact,
)
from src.soz.geometry import CHANNEL_INDEX, N_STANDARD_CHANNELS  # noqa: E402
from src.soz.locked_source_eval_predictions import (  # noqa: E402
    LOCKED_SOURCE_EVAL_PREDICTION_PROTOCOL,
    VerifiedLockedSourceEvalPredictions,
    load_locked_source_eval_predictions,
)
from src.soz.locked_source_eval_roster import (  # noqa: E402
    EXPECTED_SOURCE_EVAL_EVENT_COUNT,
    EXPECTED_SOURCE_EVAL_PATIENT_COUNT,
)
from src.soz.metrics import (  # noqa: E402
    DEEPSOZ_STANDARD19_NEIGHBORS,
    deepsoz_style_top1_metrics,
    patient_localization_metrics,
)


EVALUATION_SCHEMA = "soz_locked_source_eval_one_shot_evaluation_v1"
RELEASE_LEDGER_SCHEMA = "soz_source_eval_target_release_consumption_v1"
RESULT_FILENAME = "result.json"
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20_260_811
_METRIC_NAMES = (
    "strict_top1",
    "one_hop_relaxed_top1",
    "macro_average_precision",
    "mean_reciprocal_rank",
    "hit_at_3",
    "hit_at_5",
)
_SPATIAL_STATES = ("exact", "neighbor_only", "far")
_SHA256_HEX = frozenset("0123456789abcdef")
_MAX_RESULT_BYTES = 16 * 1024 * 1024


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation metadata is not canonical JSON data") from exc


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return text


def _sha256_argument(value: str) -> str:
    try:
        return _require_sha256(value, field="SHA256 argument")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _absolute_no_symlink(path: str | Path, *, field: str) -> Path:
    result = Path(os.path.abspath(path))
    for component in (result, *result.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot traverse symlinks")
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def consume_source_eval_release_ledger(
    ledger_path: str | Path,
    *,
    prediction_manifest_sha256: str,
    prediction_roster_artifact_sha256: str,
    target_v2_artifact_sha256: str,
    target_v2_receipt_sha256: str,
    target_v2_policy_sha256: str,
) -> tuple[Path, str]:
    """Durably consume the single label-release attempt with ``O_EXCL``.

    Once the file descriptor has been created, an exception deliberately does
    not remove the file.  A partial or failed write still blocks a retry and
    therefore fails closed rather than silently granting a second target read.
    """

    ledger = _absolute_no_symlink(ledger_path, field="source-eval release ledger")
    if ledger.name in {"", ".", ".."}:
        raise ValueError("source-eval release ledger requires a concrete file")
    if not ledger.parent.is_dir() or ledger.parent.is_symlink():
        raise ValueError("source-eval release ledger parent must be a regular directory")
    payload = {
        "schema_version": RELEASE_LEDGER_SCHEMA,
        "protocol": LOCKED_SOURCE_EVAL_PREDICTION_PROTOCOL,
        "state": "consumed_before_target_load",
        "attempt_consumed": True,
        "retry_authorized": False,
        "target_values_loaded_when_written": False,
        "ledger_created_before_target_loader": True,
        "prediction_manifest_sha256": _require_sha256(
            prediction_manifest_sha256, field="prediction_manifest_sha256"
        ),
        "prediction_roster_artifact_sha256": _require_sha256(
            prediction_roster_artifact_sha256,
            field="prediction_roster_artifact_sha256",
        ),
        "expected_target_v2_artifact_sha256": _require_sha256(
            target_v2_artifact_sha256, field="target_v2_artifact_sha256"
        ),
        "expected_target_v2_receipt_sha256": _require_sha256(
            target_v2_receipt_sha256, field="target_v2_receipt_sha256"
        ),
        "expected_target_v2_policy_sha256": _require_sha256(
            target_v2_policy_sha256, field="target_v2_policy_sha256"
        ),
        "event_count": EXPECTED_SOURCE_EVAL_EVENT_COUNT,
        "patient_count": EXPECTED_SOURCE_EVAL_PATIENT_COUNT,
    }
    raw = _canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(ledger, flags, 0o440)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("release-ledger write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(ledger.parent)
    return ledger, _bytes_sha256(raw)


def _model_metrics(
    logits: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
) -> dict[str, object]:
    top1 = asdict(deepsoz_style_top1_metrics(logits, values, mask))
    ranking = asdict(
        patient_localization_metrics(logits, values, mask, k_values=(3, 5))
    )
    return {
        "patient_count": int(logits.shape[0]),
        "strict_top1": float(top1["strict_accuracy"]),
        "one_hop_relaxed_top1": float(top1["relaxed_accuracy"]),
        "neighbor_only_accuracy_gain": float(top1["neighbor_only_accuracy_gain"]),
        "one_hop_eligible_patient_count": int(top1["n_neighbor_eligible_samples"]),
        "macro_average_precision": float(ranking["macro_average_precision"]),
        "mean_reciprocal_rank": float(ranking["mean_reciprocal_rank"]),
        "hit_at_3": float(ranking["hit_at_k"][3]),
        "hit_at_5": float(ranking["hit_at_k"][5]),
    }


def _per_patient_metrics(
    logits: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    rows = {name: [] for name in _METRIC_NAMES}
    for patient in range(EXPECTED_SOURCE_EVAL_PATIENT_COUNT):
        selected = slice(patient, patient + 1)
        metrics = _model_metrics(logits[selected], values[selected], mask[selected])
        for name in _METRIC_NAMES:
            rows[name].append(float(metrics[name]))
    return {
        name: torch.tensor(entries, dtype=torch.float64)
        for name, entries in rows.items()
    }


def _paired_patient_bootstrap(
    anchor_rows: Mapping[str, torch.Tensor],
    candidate_rows: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    if set(anchor_rows) != set(_METRIC_NAMES) or set(candidate_rows) != set(
        _METRIC_NAMES
    ):
        raise ValueError("paired bootstrap metric set changed")
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
    indices = torch.randint(
        0,
        EXPECTED_SOURCE_EVAL_PATIENT_COUNT,
        (BOOTSTRAP_REPLICATES, EXPECTED_SOURCE_EVAL_PATIENT_COUNT),
        generator=generator,
    )
    result: dict[str, object] = {}
    for name in _METRIC_NAMES:
        anchor = anchor_rows[name]
        candidate = candidate_rows[name]
        if tuple(anchor.shape) != (EXPECTED_SOURCE_EVAL_PATIENT_COUNT,) or tuple(
            candidate.shape
        ) != (EXPECTED_SOURCE_EVAL_PATIENT_COUNT,):
            raise ValueError("paired bootstrap requires one value per patient")
        delta = candidate - anchor
        draws = delta[indices].mean(dim=1)
        interval = torch.quantile(
            draws, torch.tensor([0.025, 0.975], dtype=torch.float64)
        )
        result[name] = {
            "observed_delta": float(delta.mean().item()),
            "ci95": [float(interval[0].item()), float(interval[1].item())],
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "cluster_unit": "patient",
        }
    return result


def _top1_indices(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~mask, -torch.inf)
    maximum = masked.max(dim=1).values
    ties = mask & (logits == maximum[:, None])
    if not bool((ties.sum(dim=1) == 1).all()):
        raise ValueError("locked evaluation requires unique patient Top-1 predictions")
    return ties.to(torch.int64).argmax(dim=1)


def _spatial_state(
    predicted: int, patient_values: torch.Tensor, patient_mask: torch.Tensor
) -> str:
    positive = (patient_values == 1) & patient_mask
    if bool(positive[predicted]):
        return "exact"
    if int(positive.sum().item()) <= 4:
        for index in torch.nonzero(positive, as_tuple=False).flatten().tolist():
            if predicted in DEEPSOZ_STANDARD19_NEIGHBORS[index]:
                return "neighbor_only"
    return "far"


def _transition_and_flip_outcomes(
    predictions: VerifiedLockedSourceEvalPredictions,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[dict[str, object], dict[str, object]]:
    anchor_top = _top1_indices(predictions.exact_anchor_logits, mask)
    candidate_top = _top1_indices(predictions.v9_logits, mask)
    transitions = {
        source: {destination: 0 for destination in _SPATIAL_STATES}
        for source in _SPATIAL_STATES
    }
    applied_transitions = {
        source: {destination: 0 for destination in _SPATIAL_STATES}
        for source in _SPATIAL_STATES
    }
    strict_rescue = strict_harm = strict_neutral = 0
    for patient in range(EXPECTED_SOURCE_EVAL_PATIENT_COUNT):
        source = _spatial_state(int(anchor_top[patient]), values[patient], mask[patient])
        destination = _spatial_state(
            int(candidate_top[patient]), values[patient], mask[patient]
        )
        transitions[source][destination] += 1
        if bool(predictions.flip_applied[patient]):
            applied_transitions[source][destination] += 1
            if source != "exact" and destination == "exact":
                strict_rescue += 1
            elif source == "exact" and destination != "exact":
                strict_harm += 1
            else:
                strict_neutral += 1
    transition = {
        "categories": list(_SPATIAL_STATES),
        "all_patients": transitions,
        "flipped_patients": applied_transitions,
        "anchor_far_error_count": sum(transitions["far"].values()),
        "v9_far_error_count": sum(
            transitions[source]["far"] for source in _SPATIAL_STATES
        ),
        "exact_to_nonexact_loss_count": (
            transitions["exact"]["neighbor_only"] + transitions["exact"]["far"]
        ),
    }
    flips = {
        "applied_count": int(predictions.flip_applied.sum().item()),
        "strict_rescue_count": strict_rescue,
        "strict_harm_count": strict_harm,
        "strict_neutral_count": strict_neutral,
    }
    if sum(flips[name] for name in (
        "strict_rescue_count", "strict_harm_count", "strict_neutral_count"
    )) != flips["applied_count"]:
        raise RuntimeError("flip outcome accounting is incomplete")
    return transition, flips


def evaluate_predictions_against_targets(
    predictions: VerifiedLockedSourceEvalPredictions,
    *,
    patient_ids: Sequence[str],
    values: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, object]:
    """Pure metric core used after release (and by synthetic unit tests)."""

    ids = tuple(str(value) for value in patient_ids)
    if ids != predictions.patient_ids:
        raise ValueError("target and sealed prediction patient rosters differ")
    expected_shape = (EXPECTED_SOURCE_EVAL_PATIENT_COUNT, N_STANDARD_CHANNELS)
    if tuple(values.shape) != expected_shape or tuple(mask.shape) != expected_shape:
        raise ValueError("released targets must have shape [21,19]")
    if values.dtype != torch.float32 or mask.dtype != torch.bool:
        raise TypeError("released targets must be float32 values and bool mask")
    if not torch.equal(mask.cpu(), predictions.deployment_mask):
        raise ValueError("released target mask differs from the preregistered PZ-only mask")
    if bool(mask[:, CHANNEL_INDEX["PZ"]].any()) or not bool(
        mask[:, [index for index in range(N_STANDARD_CHANNELS) if index != CHANNEL_INDEX["PZ"]]].all()
    ):
        raise ValueError("source-eval denominator must mask PZ and only PZ")
    observed = values[mask]
    if not torch.isfinite(observed).all() or not bool(
        ((observed == 0) | (observed == 1)).all()
    ):
        raise ValueError("released targets must be finite observed binary values")
    if bool((((values == 1) & mask).sum(dim=1) == 0).any()):
        raise ValueError("every released source-eval patient needs an observed positive")
    anchor_metrics = _model_metrics(predictions.exact_anchor_logits, values, mask)
    v9_metrics = _model_metrics(predictions.v9_logits, values, mask)
    anchor_rows = _per_patient_metrics(predictions.exact_anchor_logits, values, mask)
    v9_rows = _per_patient_metrics(predictions.v9_logits, values, mask)
    bootstrap = _paired_patient_bootstrap(anchor_rows, v9_rows)
    transition, flips = _transition_and_flip_outcomes(predictions, values, mask)
    delta = {
        name: float(v9_metrics[name]) - float(anchor_metrics[name])
        for name in _METRIC_NAMES
    }
    gate_checks = {
        "strict_top1_noninferior": delta["strict_top1"] >= -1e-12,
        "at_least_one_strict_patient_rescued_net": (
            flips["strict_rescue_count"] - flips["strict_harm_count"] >= 1
        ),
        "one_hop_relaxed_top1_noninferior": (
            delta["one_hop_relaxed_top1"] >= -1e-12
        ),
        "macro_average_precision_noninferior": (
            delta["macro_average_precision"] >= -1e-12
        ),
        "exact_to_neighbor_or_far_loss_zero": (
            transition["exact_to_nonexact_loss_count"] == 0
        ),
        "far_error_not_increased": (
            transition["v9_far_error_count"] <= transition["anchor_far_error_count"]
        ),
        "all_predictions_finite": bool(
            torch.isfinite(predictions.exact_anchor_logits).all()
            and torch.isfinite(predictions.v9_logits).all()
        ),
    }
    return {
        "models": {"exact_anchor": anchor_metrics, "v9": v9_metrics},
        "v9_minus_exact_anchor": delta,
        "transition": transition,
        "flip_outcomes": flips,
        "paired_patient_bootstrap": bootstrap,
        "preregistered_gate": {
            "checks": gate_checks,
            "v9_passed_all_checks": all(gate_checks.values()),
            "fallback_if_failed": "temporal_mil_exact",
        },
    }


def _atomic_publish_result(output_directory: str | Path, result: object) -> tuple[Path, str]:
    output = _absolute_no_symlink(output_directory, field="source-eval result output")
    if output.name in {"", ".", ".."} or os.path.lexists(output):
        raise FileExistsError(f"source-eval result output exists or is invalid: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("source-eval result parent must be a regular directory")
    raw = _canonical_json_bytes(result)
    if not 1 <= len(raw) <= _MAX_RESULT_BYTES:
        raise ValueError("source-eval result has an invalid size")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        path = temporary / RESULT_FILENAME
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(temporary)
        if os.path.lexists(output):
            raise FileExistsError(output)
        os.rename(temporary, output)
        published = True
        _fsync_directory(output.parent)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return output, _bytes_sha256(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--expected-prediction-manifest-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument("--locked-roster", type=Path, required=True)
    parser.add_argument(
        "--expected-locked-roster-artifact-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument(
        "--expected-signal-artifact-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument(
        "--expected-signal-receipt-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument("--target-v2-directory", type=Path, required=True)
    parser.add_argument("--deepsoz-source-csv", type=Path, required=True)
    parser.add_argument("--split-manifest-csv", type=Path, required=True)
    parser.add_argument(
        "--expected-target-v2-artifact-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument(
        "--expected-target-v2-summary-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument(
        "--expected-target-v2-readme-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument(
        "--expected-target-v2-receipt-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument(
        "--expected-target-v2-policy-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument(
        "--expected-deepsoz-source-input-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument(
        "--expected-split-input-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument("--release-ledger", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Phase 1: target-free validation only.  No target loader is called above
    # or during this strict prediction/roster replay.
    predictions = load_locked_source_eval_predictions(
        args.predictions,
        expected_manifest_sha256=args.expected_prediction_manifest_sha256,
        roster_bundle=args.locked_roster,
        expected_roster_artifact_sha256=(
            args.expected_locked_roster_artifact_sha256
        ),
        expected_signal_artifact_sha256=args.expected_signal_artifact_sha256,
        expected_signal_receipt_sha256=args.expected_signal_receipt_sha256,
    )
    output = _absolute_no_symlink(args.output_directory, field="source-eval result output")
    if os.path.lexists(output):
        raise FileExistsError(output)
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("source-eval result parent must exist before release")

    # Phase 2: irrevocably consume the attempt.  This line must remain before
    # the first verified target-v2 loader call.
    ledger_path, ledger_sha = consume_source_eval_release_ledger(
        args.release_ledger,
        prediction_manifest_sha256=predictions.manifest_sha256,
        prediction_roster_artifact_sha256=(
            predictions.manifest["locked_roster_artifact_sha256"]
        ),
        target_v2_artifact_sha256=args.expected_target_v2_artifact_sha256,
        target_v2_receipt_sha256=args.expected_target_v2_receipt_sha256,
        target_v2_policy_sha256=args.expected_target_v2_policy_sha256,
    )

    # Phase 3: the only target-value load in the command.
    target = load_verified_deepsoz_target_v2_artifact(
        args.target_v2_directory,
        args.deepsoz_source_csv,
        args.split_manifest_csv,
        expected_target_artifact_sha256=args.expected_target_v2_artifact_sha256,
        expected_summary_artifact_sha256=args.expected_target_v2_summary_sha256,
        expected_readme_artifact_sha256=args.expected_target_v2_readme_sha256,
        expected_source_input_sha256=args.expected_deepsoz_source_input_sha256,
        expected_split_input_sha256=args.expected_split_input_sha256,
    )
    if (
        target.receipt.receipt_sha256 != args.expected_target_v2_receipt_sha256
        or target.receipt.policy_sha256 != args.expected_target_v2_policy_sha256
        or target.receipt.policy_sha256 != TARGET_V2_POLICY_SHA256
        or target.receipt.split_input_sha256
        != args.expected_split_input_sha256
        or target.receipt.split_input_sha256
        != predictions.manifest["split_manifest_sha256"]
    ):
        raise ValueError("released target-v2 receipt, policy, or split lineage mismatch")

    references = target.registry.for_split("source_eval", eligible_only=True)
    target_ids = tuple(reference.patient_id for reference in references)
    if (
        len(references) != EXPECTED_SOURCE_EVAL_PATIENT_COUNT
        or target_ids != predictions.patient_ids
        or any(reference.official_split != "eval" for reference in references)
    ):
        raise ValueError("released target-v2 source_eval roster differs from predictions")
    batch = target.registry.target_batch(predictions.patient_ids, require_eligible=True)
    metrics = evaluate_predictions_against_targets(
        predictions,
        patient_ids=batch.patient_ids,
        values=batch.values.cpu(),
        mask=batch.mask.cpu(),
    )
    result = {
        "schema_version": EVALUATION_SCHEMA,
        "protocol": LOCKED_SOURCE_EVAL_PREDICTION_PROTOCOL,
        "evaluation_status": "one_shot_source_eval_consumed_and_completed",
        "label_informed_exploratory_evaluation": True,
        "confirmatory_external_validation": False,
        "model_selection_authorized": False,
        "threshold_tuning_authorized": False,
        "retry_authorized": False,
        "target_values_loaded": True,
        "target_values_loaded_only_after_release_ledger": True,
        "private_used": False,
        "event_count": EXPECTED_SOURCE_EVAL_EVENT_COUNT,
        "patient_count": EXPECTED_SOURCE_EVAL_PATIENT_COUNT,
        "pz_fixed_masked": True,
        "prediction_manifest_sha256": predictions.manifest_sha256,
        "prediction_patient_roster_sha256": predictions.manifest[
            "patient_roster_sha256"
        ],
        "release_ledger_path": str(ledger_path),
        "release_ledger_sha256": ledger_sha,
        "target_lineage": {
            "target_v2_artifact_sha256": target.receipt.target_artifact_sha256,
            "target_v2_summary_sha256": target.receipt.summary_artifact_sha256,
            "target_v2_readme_sha256": target.receipt.readme_artifact_sha256,
            "target_v2_receipt_sha256": target.receipt.receipt_sha256,
            "target_v2_policy_sha256": target.receipt.policy_sha256,
            "deepsoz_source_input_sha256": target.receipt.source_input_sha256,
            "split_input_sha256": target.receipt.split_input_sha256,
            "zero_semantics": (
                "dataset_complement_negative_not_biological_negative"
            ),
        },
        **metrics,
    }
    path, result_sha = _atomic_publish_result(output, result)
    print(
        json.dumps(
            {
                "status": "completed_one_shot_source_eval",
                "path": str(path),
                "result_sha256": result_sha,
                "release_ledger_sha256": ledger_sha,
                "v9_passed_all_checks": metrics["preregistered_gate"][
                    "v9_passed_all_checks"
                ],
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
