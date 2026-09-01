#!/usr/bin/env python3
"""Run target-blind frozen v16 five-fold inference on Siena evidence.

This command never opens the weak patient target ledger.  It writes event and
equal-event patient predictions, uncertainty indicators, deterministic spatial
views, and facts-locked reports before any weak-label evaluation is allowed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Mapping

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_private_labram_zero_adaptation_v18 import (  # noqa: E402
    ARM,
    _fold_probabilities,
    _target_blind_report,
    _uncertainty,
)
from src.soz.clinical_reporting import derive_spatial_report  # noqa: E402
from src.soz.geometry import STANDARD_19  # noqa: E402
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SCHEMA = "siena_frozen_external_target_blind_predictions_v1"
EVIDENCE_SCHEMA = "siena_target_blind_labram_evidence_v1"
DEFAULT_EVIDENCE = ROOT / "outputs/siena_target_blind_evidence_v1_20260815"
DEFAULT_STATES = (
    ROOT
    / "outputs/labram_identity_recovery_closed_replay_v16_20260812"
    / "outer_fold_states.safetensors"
)
DEFAULT_OUTPUT = ROOT / "outputs/siena_frozen_external_predictions_v1_20260815"


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _target_blind_evidence(
    evidence_dir: Path,
) -> tuple[dict[str, object], dict[str, torch.Tensor], list[Mapping[str, object]]]:
    manifest = _read_json(evidence_dir / "manifest.json")
    if manifest.get("schema_version") != EVIDENCE_SCHEMA:
        raise ValueError("Siena evidence is not the full target-blind artifact")
    access = manifest.get("access_receipt")
    if not isinstance(access, Mapping) or (
        access.get("weak_patient_target_ledger_opened") is not False
        or access.get("siena_weak_target_values_loaded") is not False
        or access.get("c18_soz_target_values_loaded") is not False
        or access.get("private_data_loaded") is not False
        or access.get("reasoner_training_performed") is not False
        or access.get("calibration_performed") is not False
        or access.get("model_or_threshold_selection_performed") is not False
    ):
        raise ValueError("Siena evidence target/private/fit firewall failed")
    tensors = load_file(str(evidence_dir / str(manifest["tensor_file"])))
    events_raw = manifest.get("events")
    if not isinstance(events_raw, list) or any(
        not isinstance(value, Mapping) for value in events_raw
    ):
        raise TypeError("Siena evidence event roster is invalid")
    events = list(events_raw)
    if (
        len(events) != tensors["h_event"].shape[0]
        or tensors["h_event"].shape[0] != tensors["fine_event"].shape[0]
    ):
        raise ValueError("Siena evidence event/tensor identities drifted")
    return manifest, tensors, events


def equal_event_patient_probabilities(
    event_fold_probability: torch.Tensor,
    event_patient_index: torch.Tensor,
    patient_count: int,
) -> torch.Tensor:
    if (
        event_fold_probability.ndim != 3
        or event_fold_probability.shape[1:] != (5, 19)
        or event_patient_index.shape != (event_fold_probability.shape[0],)
        or event_patient_index.dtype != torch.long
        or patient_count < 1
    ):
        raise ValueError("Siena patient aggregation input contract failed")
    if event_patient_index.numel() == 0 or int(event_patient_index.min()) < 0 or int(
        event_patient_index.max()
    ) >= patient_count:
        raise ValueError("Siena event-patient index is out of range")
    result = torch.zeros(
        patient_count,
        5,
        19,
        dtype=event_fold_probability.dtype,
        device=event_fold_probability.device,
    )
    counts = torch.zeros(
        patient_count,
        dtype=event_fold_probability.dtype,
        device=event_fold_probability.device,
    )
    result.index_add_(0, event_patient_index, event_fold_probability)
    counts.index_add_(
        0,
        event_patient_index,
        torch.ones_like(event_patient_index, dtype=event_fold_probability.dtype),
    )
    if bool((counts == 0).any()):
        raise ValueError("Siena patient aggregation contains an empty patient")
    result = result / counts[:, None, None]
    if not torch.isfinite(result).all() or not torch.allclose(
        result.sum(dim=2),
        torch.ones(patient_count, 5, dtype=result.dtype, device=result.device),
        atol=1e-6,
        rtol=0,
    ):
        raise RuntimeError("Siena patient probability contract failed")
    return result.contiguous()


def _spatial_summary(probability: torch.Tensor) -> dict[str, object]:
    report = derive_spatial_report(
        probability.detach().cpu(),
        V11_CANDIDATE_MASK.detach().cpu(),
        score_semantics="uncalibrated_localization_score",
    )
    return asdict(report)


def _event_report(
    event_id: str,
    patient_id: str,
    probability: torch.Tensor,
    uncertainty: Mapping[str, object],
    evidence: Mapping[str, torch.Tensor],
    index: int,
) -> dict[str, object]:
    report = _target_blind_report(
        event_id,
        patient_id,
        probability,
        uncertainty,
        evidence,
        index,
    )
    text = str(report["report_text_zh"])
    source_phrase = "私有EDF未记录原始共同参考"
    if text.count(source_phrase) != 1:
        raise RuntimeError("frozen target-blind report reference clause drifted")
    report["report_text_zh"] = text.replace(
        source_phrase, "Siena EDF未记录原始共同参考"
    )
    report["external_role"] = (
        "frozen_descriptive_external_signal_audit_not_c18_soz_validation"
    )
    report["automatic_decision_status"] = "abstained_uncalibrated_observability"
    report["spatial_view"] = _spatial_summary(probability)
    return report


def predict(evidence_dir: Path, state_path: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    evidence_manifest, evidence, events = _target_blind_evidence(evidence_dir)
    h_event = evidence["h_event"]
    fine_event = evidence["fine_event"]
    states = load_file(str(state_path.resolve(strict=True)))
    event_fold_probability = _fold_probabilities(h_event, fine_event, states)
    event_probability = event_fold_probability.mean(dim=1)
    event_uncertainty = _uncertainty(event_fold_probability)

    event_ids = [str(row["event_id"]) for row in events]
    event_patients = [str(row["patient_id"]) for row in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Siena evidence event IDs are duplicated")
    patient_ids = sorted(set(event_patients))
    patient_index = {patient: index for index, patient in enumerate(patient_ids)}
    event_patient_index = torch.tensor(
        [patient_index[patient] for patient in event_patients], dtype=torch.long
    )
    patient_fold_probability = equal_event_patient_probabilities(
        event_fold_probability, event_patient_index, len(patient_ids)
    )
    patient_probability = patient_fold_probability.mean(dim=1)
    patient_uncertainty = _uncertainty(patient_fold_probability)
    event_count_distribution = Counter(event_patients)

    output.mkdir(parents=True)
    save_file(
        {
            "event_fold_probability": event_fold_probability.contiguous(),
            "event_probability": event_probability.contiguous(),
            "patient_fold_probability": patient_fold_probability.contiguous(),
            "patient_probability": patient_probability.contiguous(),
            "event_patient_index": event_patient_index.contiguous(),
        },
        str(output / "predictions.safetensors"),
    )

    event_reports = [
        _event_report(
            event_ids[index],
            event_patients[index],
            event_probability[index],
            event_uncertainty[index],
            evidence,
            index,
        )
        for index in range(len(event_ids))
    ]
    with (output / "structured_reports.jsonl").open("w", encoding="utf-8") as stream:
        for report in event_reports:
            stream.write(json.dumps(report, ensure_ascii=False) + "\n")

    patient_summaries = []
    for index, patient_id in enumerate(patient_ids):
        probability = patient_probability[index]
        ranking = [
            STANDARD_19[channel]
            for channel in torch.argsort(probability, descending=True).tolist()
            if bool(V11_CANDIDATE_MASK[channel])
        ]
        patient_summaries.append(
            {
                "patient_id": patient_id,
                "eligible_event_count": event_count_distribution[patient_id],
                "top5_channels": ranking[:5],
                "uncertainty": patient_uncertainty[index],
                "spatial_view": _spatial_summary(probability),
                "automatic_decision_status": "abstained_uncalibrated_observability",
                "claim_boundary": (
                    "weak external scalp phenotype audit; not C18 SOZ gold, "
                    "cortical SOZ, EZ, or treatment target"
                ),
            }
        )
    with (output / "patient_summaries.jsonl").open("w", encoding="utf-8") as stream:
        for row in patient_summaries:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "model": {
            "backbone": "official_pretrained_LaBraM_Base",
            "foundation_trained_from_scratch": False,
            "inference_arm": ARM,
            "fold_ensemble": "equal_probability_mean_5",
            "patient_aggregation": "equal_event_mean_after_event_inference",
            "candidate_space": [
                channel
                for channel, allowed in zip(STANDARD_19, V11_CANDIDATE_MASK)
                if bool(allowed)
            ],
        },
        "evidence_manifest": str(evidence_dir / "manifest.json"),
        "source_time_supported_events": int(
            evidence_manifest["input_time_supported_event_count"]
        ),
        "signal_eligible_event_count": len(event_ids),
        "signal_eligible_patient_count": len(patient_ids),
        "event_ids": event_ids,
        "event_patient_ids": event_patients,
        "patient_ids": patient_ids,
        "event_count_distribution": dict(sorted(event_count_distribution.items())),
        "event_uncertainty": event_uncertainty,
        "patient_uncertainty": patient_uncertainty,
        "files": {
            "prediction_tensors": "predictions.safetensors",
            "event_reports": "structured_reports.jsonl",
            "patient_summaries": "patient_summaries.jsonl",
        },
        "access_receipt": {
            "target_blind_evidence_loaded": True,
            "weak_patient_target_ledger_opened": False,
            "siena_weak_target_values_loaded": False,
            "c18_soz_target_values_loaded": False,
            "private_data_loaded": False,
            "training_performed": False,
            "calibration_performed": False,
            "model_or_threshold_selection_performed": False,
            "all_automatic_decisions_abstained": True,
        },
    }
    (output / "prediction_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = predict(args.evidence, args.states, args.output)
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "signal_eligible_event_count": result[
                    "signal_eligible_event_count"
                ],
                "signal_eligible_patient_count": result[
                    "signal_eligible_patient_count"
                ],
                "all_automatic_decisions_abstained": result["access_receipt"][
                    "all_automatic_decisions_abstained"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
