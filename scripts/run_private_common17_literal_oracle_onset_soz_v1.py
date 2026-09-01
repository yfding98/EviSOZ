#!/usr/bin/env python3
"""Run the frozen public common17 SOZ heads at reference-derived private onsets.

This is a retrospective timing-oracle diagnostic arm.  The predictor opens a
sanitized onset-only roster whose times were derived upstream from an exact-ID
doctor-event/EDF-annotation match.  It therefore is not an EEG-only end-to-end
result and is never eligible for production report generation.  The predictor
does not open doctor SOZ targets, Excel workbooks, raw EDF annotations, or any
private calibration/model-selection artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_private_common17_literal_soz_v1 import (  # noqa: E402
    ARM,
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    COMMON17,
    DEFAULT_CHECKPOINT,
    DEFAULT_EEG_ROOT,
    DEFAULT_INVENTORY,
    DEFAULT_MODEL,
    DEFAULT_MODELING,
    EXPECTED_PATIENTS,
    EXPECTED_RECORDS,
    FOLD_COUNT,
    CausalEDFConfig,
    OfficialLaBraMCommon17FrozenPrefixEncoder,
    _canonical_sha256,
    _load_fold_model,
    _private_common17_event,
    _probability,
    _ranking,
    _read_json,
    _safe_edf,
    _sha256,
    common17_event_calls,
    extract_common17_phase_features,
)


SCHEMA = "private_common17_literal_oracle_onset_public_fold_ensemble_v1"
ORACLE_SCHEMA = "private_common17_oracle_onset_roster_v2"
STATUS = "completed_reference_timing_conditioned_private_common17_external_fold_ensemble"
DEFAULT_ORACLE_ROSTER = ROOT / "outputs/private_common17_oracle_onset_roster_v2_20260825"
DEFAULT_STRICT_OUTPUT = ROOT / "outputs/private_common17_literal_oracle_strict_v1_20260825"
DEFAULT_ALL_EXACT_OUTPUT = (
    ROOT / "outputs/private_common17_literal_oracle_all_exact_sensitivity_v1_20260825"
)
SCOPES = ("strict_primary", "all_exact_supported")
EXPECTED_SCOPE_EVENTS = {"strict_primary": 75, "all_exact_supported": 81}
ORACLE_EVENT_FIELDS = frozenset(
    {
        "oracle_event_id",
        "recording_id",
        "patient_pseudonym",
        "source_signal_sha256",
        "anchor_seconds",
        "anchor_time_semantics",
        "strict_single_marker_primary",
        "legacy_time_support_preeligible",
    }
)


def _validate_oracle_roster(root: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    manifest_path = root.resolve(strict=True) / "manifest.json"
    receipt_path = root.resolve(strict=True) / "receipt.json"
    manifest = _read_json(manifest_path)
    receipt = _read_json(receipt_path)
    if manifest.get("schema_version") != ORACLE_SCHEMA:
        raise ValueError("wrong sanitized oracle-onset roster schema")
    if manifest.get("status") != (
        "completed_frozen_private_exact_SZ_event_matched_onset_only_projection"
    ):
        raise ValueError("sanitized oracle-onset roster is not frozen and complete")
    expected_content = str(manifest.get("content_sha256", ""))
    unhashed = dict(manifest)
    unhashed.pop("content_sha256", None)
    if _canonical_sha256(unhashed) != expected_content:
        raise RuntimeError("sanitized oracle-onset content hash mismatch")
    if receipt.get("content_sha256") != expected_content:
        raise RuntimeError("sanitized oracle-onset receipt content hash mismatch")
    if receipt.get("manifest_sha256") != _sha256(manifest_path):
        raise RuntimeError("sanitized oracle-onset manifest hash mismatch")
    access = manifest.get("access_receipt", {})
    required_false = (
        "sibling_SOZ_target_ledger_opened",
        "doctor_label_release_opened",
        "raw_EDF_annotations_opened",
        "Excel_workbook_opened",
        "model_predictions_loaded",
        "training_calibration_or_model_selection_performed",
        "EEG_only_eligible",
    )
    if any(access.get(key) is not False for key in required_false):
        raise RuntimeError("sanitized oracle-onset roster violated its target/timing firewall")
    if (
        access.get("upstream_doctor_event_to_annotation_join_used") is not True
        or access.get("upstream_EDF_annotation_timing_used") is not True
    ):
        raise RuntimeError("sanitized oracle-onset roster omitted upstream timing provenance")
    events = manifest.get("events")
    if not isinstance(events, list) or len(events) != 81:
        raise ValueError("sanitized oracle-onset roster no longer has 81 exact-supported rows")
    for row in events:
        if not isinstance(row, dict) or set(row) != ORACLE_EVENT_FIELDS:
            raise RuntimeError("oracle-onset event projection is not an exact field allowlist")
        if row.get("anchor_time_semantics") != (
            "doctor_SZ_event_exact_ID_matched_annotation_derived_onset"
        ):
            raise RuntimeError("oracle-onset time semantics drifted")
        if row.get("legacy_time_support_preeligible") is not True:
            raise RuntimeError("oracle-onset legacy support flag drifted")
    return manifest, receipt, manifest_path, receipt_path


def _technical_exclusion(error: OSError | ValueError) -> tuple[str, str]:
    message = str(error)
    lowered = message.lower()
    if isinstance(error, OSError):
        category = "unreadable_edf"
    elif "warmup/pre/post support" in lowered:
        category = "insufficient_actual_window_support"
    elif "lacks unambiguous common17" in lowered:
        category = "common17_channel_qc"
    elif "raw qc failed" in lowered:
        category = "common17_signal_qc"
    else:
        category = "loader_value_error"
    return category, message


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    started = time.monotonic()
    inventory_path = args.inventory.resolve(strict=True)
    eeg_root = args.eeg_root.resolve(strict=True)
    model_root = args.model.resolve(strict=True)
    model_manifest_path = model_root / "manifest.json"
    model_tensor_path = model_root / "oof_predictions_and_states.safetensors"
    oracle_root = args.oracle_roster.resolve(strict=True)
    oracle, oracle_receipt, oracle_manifest_path, oracle_receipt_path = (
        _validate_oracle_roster(oracle_root)
    )
    inventory = _read_json(inventory_path)
    model_manifest = _read_json(model_manifest_path)
    records = inventory.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_RECORDS:
        raise ValueError("private inventory no longer contains 141 records")
    patient_ids = sorted({str(row["patient_pseudonym"]) for row in records})
    if len(patient_ids) != EXPECTED_PATIENTS:
        raise ValueError("private inventory no longer contains 45 patients")
    if tuple(model_manifest.get("common17_channels", ())) != tuple(COMMON17):
        raise ValueError("public model common17 order drifted")
    if model_manifest.get("access_receipt", {}).get("private_data_loaded") is not False:
        raise RuntimeError("public model artifact does not retain its private-data firewall")

    inventory_by_id = {str(row["recording_id"]): row for row in records}
    if len(inventory_by_id) != EXPECTED_RECORDS:
        raise ValueError("private inventory contains duplicate recording IDs")
    selected_source = [
        row
        for row in oracle["events"]
        if args.scope == "all_exact_supported" or row["strict_single_marker_primary"] is True
    ]
    if len(selected_source) != EXPECTED_SCOPE_EVENTS[args.scope]:
        raise RuntimeError("selected oracle-onset scope count drifted")
    selected_by_record: dict[str, list[Mapping[str, Any]]] = {}
    for row in selected_source:
        recording_id = str(row["recording_id"])
        current = inventory_by_id.get(recording_id)
        if current is None:
            raise ValueError("oracle-onset record is absent from current inventory")
        if (
            str(row["patient_pseudonym"]) != str(current["patient_pseudonym"])
            or str(row["source_signal_sha256"]) != str(current["source_signal_sha256"])
        ):
            raise RuntimeError("oracle-onset identity/hash does not match current inventory")
        selected_by_record.setdefault(recording_id, []).append(row)
    if any(len(rows) != 1 for rows in selected_by_record.values()):
        raise RuntimeError("current oracle scope unexpectedly repeats a recording")

    record_index_by_id = {
        str(row["recording_id"]): index for index, row in enumerate(records)
    }
    record_rows: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        recording_id = str(row["recording_id"])
        anchor_count = len(selected_by_record.get(recording_id, ()))
        record_rows.append(
            {
                "record_index": index,
                "recording_id": recording_id,
                "patient_pseudonym": str(row["patient_pseudonym"]),
                "source_signal_sha256": str(row["source_signal_sha256"]),
                "oracle_anchor_eligible": bool(anchor_count),
                "oracle_anchor_event_count": anchor_count,
                "successful_event_count": 0,
                "event_count": 0,
                "prediction_status": (
                    "pending_reference_timing_prediction"
                    if anchor_count
                    else "outside_oracle_onset_scope"
                ),
                "ranking": [],
            }
        )

    selected_events: list[dict[str, Any]] = []
    for source in selected_source:
        recording_id = str(source["recording_id"])
        selected_events.append(
            {
                "oracle_event_id": str(source["oracle_event_id"]),
                "event_index": None,
                "record_index": record_index_by_id[recording_id],
                "recording_id": recording_id,
                "patient_pseudonym": str(source["patient_pseudonym"]),
                "source_signal_sha256": str(source["source_signal_sha256"]),
                "anchor_seconds": float(source["anchor_seconds"]),
                "anchor_time_semantics": str(source["anchor_time_semantics"]),
                "strict_single_marker_primary": bool(source["strict_single_marker_primary"]),
                "legacy_time_support_preeligible": True,
                "technical_status": "pending_actual_reader_validation",
            }
        )
    selected_events.sort(key=lambda row: (int(row["record_index"]), str(row["oracle_event_id"])))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    encoder = OfficialLaBraMCommon17FrozenPrefixEncoder(
        modeling_path=args.modeling.resolve(strict=True),
        checkpoint_path=args.checkpoint.resolve(strict=True),
        expected_sha256=AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
    ).to(device).eval()
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("private common17 foundation exposes trainable parameters")
    encoder_dtype = next(encoder.parameters()).dtype
    config = CausalEDFConfig(
        reference_policy="unlabeled_uniform_common_reference_common17_transport_v1",
        apply_car19=False,
    )
    phase_rows: list[torch.Tensor] = []
    position_rows: list[torch.Tensor] = []
    successful_events: list[dict[str, Any]] = []
    verified_paths: set[Path] = set()
    for ordinal, event in enumerate(selected_events, start=1):
        source = inventory_by_id[event["recording_id"]]
        path = _safe_edf(eeg_root, source["edf_relative_path"])
        if path not in verified_paths:
            if not args.skip_source_hash_verification:
                if _sha256(path) != str(source["source_signal_sha256"]):
                    raise RuntimeError(f"private source hash mismatch for {event['recording_id']}")
            verified_paths.add(path)
        try:
            car17, binding, signal_receipt = _private_common17_event(
                path,
                float(event["anchor_seconds"]),
                config=config,
            )
        except (OSError, ValueError) as error:
            category, message = _technical_exclusion(error)
            event["technical_status"] = f"technical_exclusion_{category}"
            event["technical_exclusion"] = {
                "category": category,
                "exception_type": type(error).__name__,
                "message": message,
            }
            record_rows[int(event["record_index"])]["prediction_status"] = event[
                "technical_status"
            ]
        else:
            calls = common17_event_calls(car17).to(device=device, dtype=encoder_dtype)
            with torch.inference_mode():
                prefix = encoder.forward_with_record_binding(calls, binding)
            event_phase = extract_common17_phase_features(prefix.cpu().float())
            event_index = len(phase_rows)
            phase_rows.append(event_phase)
            position_rows.append(torch.tensor(binding.position_ids, dtype=torch.long))
            event["event_index"] = event_index
            event["technical_status"] = "completed_actual_reader_and_encoder"
            event["signal_receipt"] = signal_receipt
            successful_events.append(event)
            record = record_rows[int(event["record_index"])]
            record["successful_event_count"] += 1
            record["event_count"] += 1
        if ordinal % args.progress_every == 0 or ordinal == len(selected_events):
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "stage": "private_common17_oracle_phase",
                        "scope": args.scope,
                        "attempted": ordinal,
                        "total": len(selected_events),
                        "successful": len(successful_events),
                        "elapsed_seconds": elapsed,
                        "eta_seconds": elapsed / ordinal * (len(selected_events) - ordinal),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not successful_events:
        raise RuntimeError("all reference-timing events failed technical validation")
    phase = torch.stack(phase_rows).contiguous()
    position_ids = torch.stack(position_rows).contiguous()
    if tuple(phase.shape[1:]) != (17, 5, 200) or not torch.isfinite(phase).all():
        raise RuntimeError("private oracle common17 phase cache is invalid")

    event_record = torch.tensor(
        [int(row["record_index"]) for row in successful_events], dtype=torch.long
    )
    patient_index = {patient: index for index, patient in enumerate(patient_ids)}
    event_patient = torch.tensor(
        [patient_index[str(row["patient_pseudonym"])] for row in successful_events],
        dtype=torch.long,
    )
    predicted_records = sorted(set(event_record.tolist()))
    predicted_patients = sorted(set(event_patient.tolist()))
    record_local = {value: index for index, value in enumerate(predicted_records)}
    patient_local = {value: index for index, value in enumerate(predicted_patients)}
    event_record_local = torch.tensor(
        [record_local[int(value)] for value in event_record], dtype=torch.long
    )
    event_patient_local = torch.tensor(
        [patient_local[int(value)] for value in event_patient], dtype=torch.long
    )

    event_fold = torch.empty((len(successful_events), FOLD_COUNT, 17), dtype=torch.float32)
    record_fold = torch.zeros((EXPECTED_RECORDS, FOLD_COUNT, 17), dtype=torch.float32)
    patient_fold = torch.zeros((EXPECTED_PATIENTS, FOLD_COUNT, 17), dtype=torch.float32)
    for fold in range(FOLD_COUNT):
        model = _load_fold_model(model_tensor_path, fold=fold)
        event_fold[:, fold] = _probability(
            model,
            phase,
            torch.arange(len(successful_events), dtype=torch.long),
            len(successful_events),
        )
        fold_record_probability = _probability(
            model, phase, event_record_local, len(predicted_records)
        )
        fold_patient_probability = _probability(
            model, phase, event_patient_local, len(predicted_patients)
        )
        record_fold[torch.tensor(predicted_records), fold] = fold_record_probability
        patient_fold[torch.tensor(predicted_patients), fold] = fold_patient_probability
        print(
            json.dumps(
                {"stage": "public_fold_inference", "scope": args.scope, "fold": fold},
                sort_keys=True,
            ),
            flush=True,
        )
    event_probability = event_fold.mean(dim=1).contiguous()
    record_probability = record_fold.mean(dim=1).contiguous()
    patient_probability = patient_fold.mean(dim=1).contiguous()
    record_prediction_mask = torch.zeros(EXPECTED_RECORDS, dtype=torch.bool)
    record_prediction_mask[predicted_records] = True
    patient_prediction_mask = torch.zeros(EXPECTED_PATIENTS, dtype=torch.bool)
    patient_prediction_mask[predicted_patients] = True
    record_event_count = torch.bincount(event_record, minlength=EXPECTED_RECORDS).long()
    patient_event_count = torch.bincount(event_patient, minlength=EXPECTED_PATIENTS).long()
    if not torch.equal(record_event_count[record_prediction_mask], torch.ones(len(predicted_records), dtype=torch.long)):
        raise RuntimeError("current reference-onset scope unexpectedly has multi-event records")
    if not torch.allclose(
        record_fold[event_record, :, :], event_fold, atol=1e-7, rtol=0
    ):
        raise RuntimeError("single-event record and event fold probabilities diverged")

    for row in record_rows:
        index = int(row["record_index"])
        if bool(record_prediction_mask[index]):
            row["prediction_status"] = "completed_reference_timing_common17_prediction"
            row["ranking"] = _ranking(record_probability[index])

    patient_anchor_count = {patient: 0 for patient in patient_ids}
    for event in selected_events:
        patient_anchor_count[str(event["patient_pseudonym"])] += 1
    patient_rows: list[dict[str, Any]] = []
    for patient, index in patient_index.items():
        anchor_count = patient_anchor_count[patient]
        successful_count = int(patient_event_count[index])
        predicted = bool(patient_prediction_mask[index])
        if predicted:
            status = "completed_reference_timing_common17_prediction"
        elif anchor_count:
            status = "all_reference_timing_events_technical_exclusion"
        else:
            status = "outside_oracle_onset_scope"
        patient_rows.append(
            {
                "patient_index": index,
                "patient_pseudonym": patient,
                "oracle_anchor_eligible": bool(anchor_count),
                "oracle_anchor_event_count": anchor_count,
                "successful_event_count": successful_count,
                "event_count": successful_count,
                "prediction_status": status,
                "ranking": _ranking(patient_probability[index]) if predicted else [],
            }
        )

    tensors = {
        "phase_features": phase,
        "event_position_ids": position_ids,
        "event_record_index": event_record.contiguous(),
        "event_patient_index": event_patient.contiguous(),
        "event_fold_probability": event_fold.contiguous(),
        "event_probability": event_probability.contiguous(),
        "record_fold_probability": record_fold.contiguous(),
        "record_probability": record_probability.contiguous(),
        "record_prediction_mask": record_prediction_mask.contiguous(),
        "record_event_count": record_event_count.contiguous(),
        "patient_fold_probability": patient_fold.contiguous(),
        "patient_probability": patient_probability.contiguous(),
        "patient_prediction_mask": patient_prediction_mask.contiguous(),
        "patient_event_count": patient_event_count.contiguous(),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "interpretation": (
            "retrospective_reference_timing_conditioned_scalp_channel_ranking_"
            "not_EEG_only_not_cortical_SOZ_or_surgical_target"
        ),
        "execution_arm": ARM,
        "oracle_scope": args.scope,
        "common17_channels": list(COMMON17),
        "cohort": {
            "records": EXPECTED_RECORDS,
            "patients": EXPECTED_PATIENTS,
            "oracle_anchor_eligible_events": len(selected_events),
            "oracle_anchor_eligible_records": len(selected_by_record),
            "oracle_anchor_eligible_patients": sum(
                row["oracle_anchor_eligible"] for row in patient_rows
            ),
            "successful_events": len(successful_events),
            "records_with_prediction": int(record_prediction_mask.sum()),
            "patients_with_prediction": int(patient_prediction_mask.sum()),
            "technical_exclusion_events": len(selected_events) - len(successful_events),
            "technical_exclusion_records": len(selected_by_record) - int(record_prediction_mask.sum()),
        },
        "anchor_contract": {
            "source": "frozen_sanitized_onset_only_roster_v2",
            "scope": args.scope,
            "reference_timing_oracle": True,
            "clinician_verified_electrographic_onset": False,
            "candidate_semantics": (
                "doctor_SZ_event_exact_ID_matched_annotation_derived_onset"
            ),
            "relative_window_seconds": [-12.0, 48.0],
            "fixed_window": True,
            "upstream_doctor_event_to_EDF_annotation_join_used": True,
            "upstream_EDF_annotation_timing_used": True,
            "raw_EDF_annotations_read_during_this_inference": False,
            "production_EEG_only_eligible": False,
        },
        "signal_contract": {
            "channels": list(COMMON17),
            "excluded_channels": ["FZ", "PZ"],
            "source_reference": "unlabeled_uniform_common_reference_assumption_not_header_proven",
            "model_input_reference": "common_average_standard17",
            "preprocessing": asdict(config),
            "FZ_or_PZ_samples_read": False,
            "FZ_or_PZ_scores_produced": False,
            "zero_fill_interpolation_or_synthesis": False,
            "edf_annotations_read": False,
        },
        "model_contract": {
            "public_training_patients": 102,
            "public_training_events": 1_145,
            "fold_count": FOLD_COUNT,
            "external_prediction_ensemble": "equal_mean_of_five_fold_probabilities",
            "private_fold_assignment_performed": False,
            "private_training_calibration_threshold_or_model_selection": False,
            "foundation": "audited_official_LaBraM_Base_frozen_through_block9",
            "head": "common17_event_set_reasoner_6967_fit_time_parameters_per_fold",
        },
        "records": record_rows,
        "patients": patient_rows,
        "events": selected_events,
        "access_receipt": {
            "raw_private_EEG_loaded": True,
            "oracle_onset_release_loaded": True,
            "doctor_SOZ_targets_loaded": False,
            "private_target_values_loaded": False,
            "raw_EDF_annotations_read": False,
            "Excel_opened": False,
            "legacy_private_SOZ_ranking_used": False,
            "private_training_or_parameter_fitting_performed": False,
            "production_EEG_only_eligible": False,
            "retrospective_marker_oracle_diagnostic": True,
            "LLM_used": False,
        },
        "lineage": {
            "inventory": {"sha256": _sha256(inventory_path)},
            "oracle_onset_manifest": {
                "sha256": _sha256(oracle_manifest_path),
                "content_sha256": oracle["content_sha256"],
            },
            "oracle_onset_receipt": {"sha256": _sha256(oracle_receipt_path)},
            "public_model_manifest": {"sha256": _sha256(model_manifest_path)},
            "public_model_tensor": {"sha256": _sha256(model_tensor_path)},
            "labram_checkpoint": {"sha256": _sha256(args.checkpoint)},
            "labram_modeling": {"sha256": _sha256(args.modeling)},
        },
        "resource_usage": {
            "device": str(device),
            "elapsed_seconds": time.monotonic() - started,
        },
        "tensor_file": "predictions_and_phase.safetensors",
    }
    manifest["prediction_roster_sha256"] = _canonical_sha256(
        {
            "oracle_scope": args.scope,
            "records": record_rows,
            "patients": patient_rows,
            "events": [
                {
                    "oracle_event_id": row["oracle_event_id"],
                    "recording_id": row["recording_id"],
                    "anchor_seconds": row["anchor_seconds"],
                    "technical_status": row["technical_status"],
                }
                for row in selected_events
            ],
        }
    )
    return manifest, tensors


def publish(
    output: Path,
    manifest: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        tensor_path = staging / str(manifest["tensor_file"])
        save_file(dict(tensors), str(tensor_path))
        final = dict(manifest)
        final["tensor_sha256"] = _sha256(tensor_path)
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(final, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        receipt = {
            "schema_version": f"{SCHEMA}_receipt",
            "status": final["status"],
            "oracle_scope": final["oracle_scope"],
            "records": final["cohort"]["records"],
            "oracle_anchor_eligible_records": final["cohort"][
                "oracle_anchor_eligible_records"
            ],
            "records_with_prediction": final["cohort"]["records_with_prediction"],
            "patients": final["cohort"]["patients"],
            "events": final["cohort"]["oracle_anchor_eligible_events"],
            "doctor_SOZ_targets_loaded": False,
            "raw_EDF_annotations_read": False,
            "Excel_opened": False,
            "production_EEG_only_eligible": False,
            "retrospective_marker_oracle_diagnostic": True,
            "prediction_roster_sha256": final["prediction_roster_sha256"],
            "tensor_sha256": final["tensor_sha256"],
            "manifest_sha256": _sha256(manifest_path),
        }
        (staging / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
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
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--oracle-roster", type=Path, default=DEFAULT_ORACLE_ROSTER)
    parser.add_argument("--scope", choices=SCOPES, default="strict_primary")
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--modeling", type=Path, default=DEFAULT_MODELING)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--skip-source-hash-verification", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.progress_every < 1:
        raise ValueError("progress-every must be positive")
    if args.output is None:
        args.output = (
            DEFAULT_STRICT_OUTPUT
            if args.scope == "strict_primary"
            else DEFAULT_ALL_EXACT_OUTPUT
        )
    manifest, tensors = run(args)
    output = publish(args.output, manifest, tensors)
    print(
        json.dumps(
            {
                "output": str(output),
                "oracle_scope": manifest["oracle_scope"],
                "oracle_anchor_eligible_records": manifest["cohort"][
                    "oracle_anchor_eligible_records"
                ],
                "records_with_prediction": manifest["cohort"]["records_with_prediction"],
                "events": manifest["cohort"]["oracle_anchor_eligible_events"],
                "doctor_SOZ_targets_loaded": False,
                "production_EEG_only_eligible": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
