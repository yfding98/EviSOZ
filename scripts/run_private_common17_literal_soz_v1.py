#!/usr/bin/env python3
"""Run the frozen public common17 SOZ heads on private EEG-only anchors.

This producer is deliberately target blind.  It reads only the frozen private
record inventory, EEG-only detector selection manifests, physical EDF samples,
and the already-trained public fold states.  Doctor labels, Excel workbooks,
EDF annotations, clinical text, and legacy SOZ rankings are neither arguments
nor inputs.

The private EDF headers contain bare electrode names and do not prove their
reference.  The transport assumption is therefore recorded as an unlabeled
uniform common reference.  Only the 17 retained electrodes are read; FZ/PZ
samples and scores never enter this execution arm.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pyedflib
from safetensors import safe_open
from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sota_soz.train_common17_oracle_event_oof_v1 import (  # noqa: E402
    Common17EventSetReasoner,
)
from scripts.audit_common17_soz_raw_path_v1 import (  # noqa: E402
    COMMON17,
    _filter_resample_common17,
    _half_up_samples,
    _max_extreme_run_samples,
    _max_flatline_run_samples,
    _unit_scale,
    reference_to_car17,
)
from src.soz.data.edf import CausalEDFConfig, _rational_resampling  # noqa: E402
from src.soz.geometry import normalize_electrode_name  # noqa: E402
from src.soz.models.labram import (  # noqa: E402
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
)
from src.soz.models.labram_common17 import (  # noqa: E402
    COMMON17_PHASE_NAMES,
    OfficialLaBraMCommon17FrozenPrefixEncoder,
    bind_common17_labram_record_positions,
    common17_event_calls,
    extract_common17_phase_features,
)


SCHEMA = "private_common17_literal_public_fold_ensemble_v1"
ARM = "strict_car17_labram_literal_raw_FZ_PZ_OR_to_CZ"
DEFAULT_INVENTORY = ROOT / "outputs/private_long_recording_inventory_v1_full141_20260819.json"
DEFAULT_REPORT_ROOT = ROOT / "outputs/private_long_recording_reports_v2_3_full141_20260820"
DEFAULT_EEG_ROOT = Path("/mnt/hd1/dyf/dataset/EEG")
DEFAULT_MODEL = ROOT / "outputs/clinical_eeg_common17_user_requested_literal_midline_independent_retrain_v2_20260825"
DEFAULT_MODELING = Path("/mnt/hd1/dyf/workspace/LaBraM/modeling_finetune.py")
DEFAULT_CHECKPOINT = Path("/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth")
DEFAULT_OUTPUT = ROOT / "outputs/private_common17_literal_zero_shot_v1_20260825"
EXPECTED_RECORDS = 141
EXPECTED_PATIENTS = 45
EXPECTED_SELECTION_MANIFESTS = 125
EXPECTED_EVENT_RECORDS = 123
EXPECTED_EVENTS = 1_119
FOLD_COUNT = 5
LATENT_DIMENSION = 32


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_edf(root: Path, relative: object) -> Path:
    value = PurePosixPath(str(relative))
    if value.is_absolute() or ".." in value.parts or value.suffix.lower() != ".edf":
        raise ValueError("unsafe private EDF relative path")
    path = root.joinpath(*value.parts).resolve(strict=True)
    path.relative_to(root)
    return path


def _private_common17_event(
    path: Path,
    anchor_seconds: float,
    *,
    config: CausalEDFConfig,
) -> tuple[torch.Tensor, object, dict[str, Any]]:
    """Load one private event without reading FZ/PZ samples or annotations."""

    reader = pyedflib.EdfReader(str(path))
    try:
        labels = tuple(str(value).strip() for value in reader.getSignalLabels())
        candidates: dict[str, list[int]] = {channel: [] for channel in COMMON17}
        for index, label in enumerate(labels):
            canonical = normalize_electrode_name(label)
            if canonical in candidates:
                candidates[canonical].append(index)
        missing = [channel for channel, rows in candidates.items() if not rows]
        duplicates = [channel for channel, rows in candidates.items() if len(rows) > 1]
        if missing or duplicates:
            raise ValueError(
                f"private EDF lacks unambiguous common17; missing={missing}, duplicates={duplicates}"
            )
        indices = tuple(candidates[channel][0] for channel in COMMON17)
        raw_names = tuple(labels[index] for index in indices)
        binding = bind_common17_labram_record_positions(raw_names)

        sampling_rates = tuple(float(reader.getSampleFrequency(index)) for index in indices)
        if any(not math.isfinite(value) or value <= 0 for value in sampling_rates):
            raise ValueError("invalid common17 sampling frequency")
        source_sfreq = sampling_rates[0]
        if any(abs(value - source_sfreq) > 1e-9 for value in sampling_rates):
            raise ValueError("mixed common17 sampling frequencies")
        all_counts = reader.getNSamples()
        sample_counts = tuple(int(all_counts[index]) for index in indices)
        if any(value <= 0 for value in sample_counts) or len(set(sample_counts)) != 1:
            raise ValueError("mixed or invalid common17 sample counts")
        units = tuple(str(reader.getPhysicalDimension(index)).strip() for index in indices)
        scales = np.asarray([_unit_scale(value) for value in units], dtype=np.float64)

        up, down = _rational_resampling(source_sfreq, config.output_sfreq_hz)
        half_length = int(config.fir_half_length_per_rate) * max(up, down)
        latency_sec = half_length / (up * source_sfreq)
        onset_sample = _half_up_samples(anchor_seconds, source_sfreq)
        read_start = onset_sample - _half_up_samples(
            config.warmup_sec + config.pre_onset_sec, source_sfreq
        )
        read_stop = onset_sample + _half_up_samples(
            config.post_onset_sec + latency_sec + 1.0 / source_sfreq,
            source_sfreq,
        )
        if read_start < 0 or read_stop > sample_counts[0]:
            raise ValueError("private event lacks common17 warmup/pre/post support")
        n_read = read_stop - read_start
        raw = np.stack(
            [
                np.asarray(reader.readSignal(index, read_start, n_read), dtype=np.float64)
                for index in indices
            ]
        )
        if tuple(raw.shape) != (len(COMMON17), n_read):
            raise RuntimeError("unexpected private common17 payload shape")
        raw_volts = raw * scales[:, None]
        flatline_limit = _half_up_samples(config.flatline_run_sec, source_sfreq)
        clipping_limit = _half_up_samples(config.clipping_run_sec, source_sfreq)
        bad_flatline = [
            COMMON17[index]
            for index, channel in enumerate(raw_volts)
            if _max_flatline_run_samples(channel, config.qc_tolerance_volts)
            >= flatline_limit
        ]
        bad_clipping = [
            COMMON17[index]
            for index, channel in enumerate(raw_volts)
            if _max_extreme_run_samples(channel, config.qc_tolerance_volts)
            >= clipping_limit
        ]
        if bad_flatline or bad_clipping:
            raise ValueError(
                f"private common17 raw QC failed; flatline={bad_flatline}, clipping={bad_clipping}"
            )

        processed, actual_up, actual_down, taps, actual_latency = _filter_resample_common17(
            raw,
            source_sfreq_hz=source_sfreq,
            config=config,
        )
        if (actual_up, actual_down) != (up, down) or abs(actual_latency - latency_sec) > 1e-12:
            raise RuntimeError("private common17 resampling receipt drifted")
        reference = torch.from_numpy(processed).to(dtype=torch.float32)
        reference = reference * reference.new_tensor(scales).unsqueeze(1)
        onset_in_processed = (onset_sample - read_start) / source_sfreq + latency_sec
        output_onset = _half_up_samples(onset_in_processed, config.output_sfreq_hz)
        pre = _half_up_samples(config.pre_onset_sec, config.output_sfreq_hz)
        post = _half_up_samples(config.post_onset_sec, config.output_sfreq_hz)
        reference = reference[:, output_onset - pre : output_onset + post].contiguous()
        if tuple(reference.shape) != (17, 12_000):
            raise RuntimeError("private common17 event crop shape changed")
        car17 = reference_to_car17(reference)
        return car17, binding, {
            "source_sfreq_hz": source_sfreq,
            "output_sfreq_hz": float(config.output_sfreq_hz),
            "resample_up": up,
            "resample_down": down,
            "resample_fir_taps": taps,
            "resample_latency_sec": latency_sec,
            "selected_common17_only": True,
            "FZ_or_PZ_samples_read": False,
            "edf_annotations_read": False,
            "labram_position_names": list(binding.position_names),
            "labram_position_ids": list(binding.position_ids),
        }
    finally:
        reader.close()


def _load_fold_model(
    state_path: Path,
    *,
    fold: int,
) -> Common17EventSetReasoner:
    prefix = f"{ARM}.fold{fold}."
    with safe_open(str(state_path.resolve(strict=True)), framework="pt", device="cpu") as source:
        keys = [key for key in source.keys() if key.startswith(prefix)]
        state = {key[len(prefix) :]: source.get_tensor(key).float() for key in keys}
    expected = {
        "input_norm.weight",
        "input_norm.bias",
        "projection.weight",
        "projection.bias",
        "phase_logits",
        "fusion_norm.weight",
        "fusion_norm.bias",
        "event_attention.weight",
        "event_attention.bias",
        "channel_scorer.weight",
        "channel_scorer.bias",
        "prior_logits",
    }
    if set(state) != expected:
        raise RuntimeError(
            f"public fold state schema drifted: missing={sorted(expected - set(state))}, "
            f"extra={sorted(set(state) - expected)}"
        )
    model = Common17EventSetReasoner(
        input_dim=200,
        phase_count=5,
        latent_dim=LATENT_DIMENSION,
        prior_logits=state["prior_logits"],
    )
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def _probability(
    model: Common17EventSetReasoner,
    features: torch.Tensor,
    owners: torch.Tensor,
    owner_count: int,
) -> torch.Tensor:
    logits = model(features.float(), owners.long(), patient_count=owner_count).patient_logits
    value = torch.softmax(logits, dim=1).cpu().contiguous()
    if tuple(value.shape) != (owner_count, 17) or not torch.isfinite(value).all():
        raise RuntimeError("private common17 probability is invalid")
    if not torch.allclose(value.sum(dim=1), torch.ones(owner_count), atol=1e-6, rtol=0):
        raise RuntimeError("private common17 probability is not normalized")
    return value


def _ranking(probability: torch.Tensor) -> list[dict[str, object]]:
    order = torch.argsort(probability, descending=True, stable=True)
    return [
        {
            "rank": rank,
            "electrode": COMMON17[index],
            "score": float(probability[index]),
        }
        for rank, index in enumerate(order.tolist(), start=1)
    ]


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    started = time.monotonic()
    inventory_path = args.inventory.resolve(strict=True)
    report_root = args.report_root.resolve(strict=True)
    eeg_root = args.eeg_root.resolve(strict=True)
    model_root = args.model.resolve(strict=True)
    model_manifest_path = model_root / "manifest.json"
    model_tensor_path = model_root / "oof_predictions_and_states.safetensors"
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
    access = model_manifest.get("access_receipt", {})
    if access.get("private_data_loaded") is not False:
        raise ValueError("public model artifact does not retain its private-data firewall")

    record_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    selection_count = 0
    for record_index, source in enumerate(records):
        recording_id = str(source["recording_id"])
        patient = str(source["patient_pseudonym"])
        bundle_path = report_root / "records" / recording_id / "report" / "bundle.json"
        selected_events: list[dict[str, Any]] = []
        if bundle_path.is_file():
            selection_count += 1
            bundle = _read_json(bundle_path)
            detector = bundle.get("detection_manifest", {}).get("detector_receipt", {})
            if detector.get("annotations_used") is not False or detector.get("labels_used") is not False:
                raise RuntimeError(f"detector firewall failed for {recording_id}")
            if (
                str(bundle.get("recording_id")) != recording_id
                or str(bundle.get("patient_pseudonym")) != patient
                or str(bundle.get("source_signal_sha256")) != str(source["source_signal_sha256"])
            ):
                raise ValueError(f"bundle identity mismatch for {recording_id}")
            rows = bundle.get("events")
            if not isinstance(rows, list):
                raise TypeError("frozen EEG bundle events must be a list")
            if int(bundle.get("event_count", -1)) != len(rows):
                raise ValueError("frozen EEG bundle event count is inconsistent")
            for row in rows:
                event = {
                    "event_index": len(event_rows),
                    "record_index": record_index,
                    "recording_id": recording_id,
                    "patient_pseudonym": patient,
                    "source_signal_sha256": str(source["source_signal_sha256"]),
                    "eeg_event_id": str(row["eeg_event_id"]),
                    "candidate_anchor_offset_seconds": float(row["candidate_anchor_offset_seconds"]),
                    "candidate_id": str(row["candidate_id"]),
                }
                selected_events.append(event)
                event_rows.append(event)
            status = "pending_prediction" if selected_events else "zero_detector_events"
        else:
            status = "upstream_technical_unassessable"
        record_rows.append(
            {
                "record_index": record_index,
                "recording_id": recording_id,
                "patient_pseudonym": patient,
                "source_signal_sha256": str(source["source_signal_sha256"]),
                "event_count": len(selected_events),
                "prediction_status": status,
            }
        )
    event_record_count = len({row["record_index"] for row in event_rows})
    if (
        selection_count != EXPECTED_SELECTION_MANIFESTS
        or len(event_rows) != EXPECTED_EVENTS
        or event_record_count != EXPECTED_EVENT_RECORDS
    ):
        raise RuntimeError(
            "private EEG-only anchor roster drifted: "
            f"selection={selection_count}, events={len(event_rows)}, event_records={event_record_count}"
        )

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
    phase = torch.empty((len(event_rows), 17, 5, 200), dtype=torch.float32)
    position_ids = torch.empty((len(event_rows), 17), dtype=torch.long)
    verified_paths: set[Path] = set()
    inventory_by_id = {str(row["recording_id"]): row for row in records}
    for ordinal, event in enumerate(event_rows, start=1):
        source = inventory_by_id[event["recording_id"]]
        path = _safe_edf(eeg_root, source["edf_relative_path"])
        if path not in verified_paths:
            if not args.skip_source_hash_verification:
                observed = _sha256(path)
                if observed != str(source["source_signal_sha256"]):
                    raise RuntimeError(f"private source hash mismatch for {event['recording_id']}")
            verified_paths.add(path)
        car17, binding, receipt = _private_common17_event(
            path,
            event["candidate_anchor_offset_seconds"],
            config=config,
        )
        calls = common17_event_calls(car17).to(device=device, dtype=encoder_dtype)
        with torch.inference_mode():
            prefix = encoder.forward_with_record_binding(calls, binding)
        event_phase = extract_common17_phase_features(prefix.cpu().float())
        phase[ordinal - 1].copy_(event_phase)
        position_ids[ordinal - 1] = torch.tensor(binding.position_ids, dtype=torch.long)
        event["signal_receipt"] = receipt
        if ordinal % args.progress_every == 0 or ordinal == len(event_rows):
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "stage": "private_common17_phase",
                        "completed": ordinal,
                        "total": len(event_rows),
                        "elapsed_seconds": elapsed,
                        "eta_seconds": elapsed / ordinal * (len(event_rows) - ordinal),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not torch.isfinite(phase).all():
        raise RuntimeError("private common17 phase cache is incomplete")

    event_record = torch.tensor([int(row["record_index"]) for row in event_rows], dtype=torch.long)
    patient_index = {patient: index for index, patient in enumerate(patient_ids)}
    event_patient = torch.tensor(
        [patient_index[str(row["patient_pseudonym"])] for row in event_rows], dtype=torch.long
    )
    predicted_records = sorted(set(event_record.tolist()))
    predicted_patients = sorted(set(event_patient.tolist()))
    record_local = {value: index for index, value in enumerate(predicted_records)}
    patient_local = {value: index for index, value in enumerate(predicted_patients)}
    event_record_local = torch.tensor([record_local[int(value)] for value in event_record], dtype=torch.long)
    event_patient_local = torch.tensor([patient_local[int(value)] for value in event_patient], dtype=torch.long)

    event_fold = torch.empty((len(event_rows), FOLD_COUNT, 17), dtype=torch.float32)
    record_fold = torch.zeros((EXPECTED_RECORDS, FOLD_COUNT, 17), dtype=torch.float32)
    patient_fold = torch.zeros((EXPECTED_PATIENTS, FOLD_COUNT, 17), dtype=torch.float32)
    for fold in range(FOLD_COUNT):
        model = _load_fold_model(model_tensor_path, fold=fold)
        event_fold[:, fold] = _probability(
            model,
            phase,
            torch.arange(len(event_rows), dtype=torch.long),
            len(event_rows),
        )
        record_probability = _probability(
            model,
            phase,
            event_record_local,
            len(predicted_records),
        )
        patient_probability = _probability(
            model,
            phase,
            event_patient_local,
            len(predicted_patients),
        )
        record_fold[torch.tensor(predicted_records), fold] = record_probability
        patient_fold[torch.tensor(predicted_patients), fold] = patient_probability
        print(json.dumps({"stage": "public_fold_inference", "fold": fold}, sort_keys=True), flush=True)
    event_probability = event_fold.mean(dim=1).contiguous()
    record_probability = record_fold.mean(dim=1).contiguous()
    patient_probability = patient_fold.mean(dim=1).contiguous()
    record_prediction_mask = torch.zeros(EXPECTED_RECORDS, dtype=torch.bool)
    record_prediction_mask[predicted_records] = True
    patient_prediction_mask = torch.zeros(EXPECTED_PATIENTS, dtype=torch.bool)
    patient_prediction_mask[predicted_patients] = True
    record_event_count = torch.bincount(event_record, minlength=EXPECTED_RECORDS).long()
    patient_event_count = torch.bincount(event_patient, minlength=EXPECTED_PATIENTS).long()

    for row in record_rows:
        index = int(row["record_index"])
        if bool(record_prediction_mask[index]):
            row["prediction_status"] = "completed_target_blind_common17_prediction"
            row["ranking"] = _ranking(record_probability[index])
        else:
            row["ranking"] = []
    patient_rows = []
    for patient, index in patient_index.items():
        predicted = bool(patient_prediction_mask[index])
        patient_rows.append(
            {
                "patient_index": index,
                "patient_pseudonym": patient,
                "event_count": int(patient_event_count[index]),
                "prediction_status": (
                    "completed_target_blind_common17_prediction"
                    if predicted
                    else "no_EEG_only_detector_events"
                ),
                "ranking": _ranking(patient_probability[index]) if predicted else [],
            }
        )

    tensors = {
        "phase_features": phase.contiguous(),
        "event_position_ids": position_ids.contiguous(),
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
        "status": "completed_target_blind_private_common17_external_fold_ensemble",
        "interpretation": "research_scalp_visible_ictal_window_channel_ranking_not_cortical_SOZ_or_surgical_target",
        "execution_arm": ARM,
        "common17_channels": list(COMMON17),
        "cohort": {
            "records": EXPECTED_RECORDS,
            "patients": EXPECTED_PATIENTS,
            "upstream_EEG_assessable_records": selection_count,
            "records_with_detector_events_and_prediction": int(record_prediction_mask.sum()),
            "records_with_zero_detector_events": sum(
                row["prediction_status"] == "zero_detector_events" for row in record_rows
            ),
            "upstream_technical_unassessable_records": sum(
                row["prediction_status"] == "upstream_technical_unassessable"
                for row in record_rows
            ),
            "EEG_only_detector_events": len(event_rows),
            "patients_with_prediction": int(patient_prediction_mask.sum()),
        },
        "anchor_contract": {
            "source": "frozen_EEG_only_transition_review_preselector_v1_analysis_selection",
            "mature_seizure_detector": False,
            "oracle_onset": False,
            "candidate_semantics": "review_candidate_not_confirmed_seizure",
            "relative_window_seconds": [-12.0, 48.0],
            "fixed_window": True,
            "annotations_labels_or_doctor_onset_used": False,
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
        "events": event_rows,
        "access_receipt": {
            "raw_private_EEG_loaded": True,
            "frozen_EEG_only_detector_selection_loaded": True,
            "doctor_label_release_loaded": False,
            "private_target_values_loaded": False,
            "EDF_annotations_loaded": False,
            "Excel_workbook_or_doctor_text_loaded": False,
            "legacy_private_SOZ_ranking_used": False,
            "private_training_or_parameter_fitting_performed": False,
            "LLM_used": False,
        },
        "lineage": {
            "inventory": {"sha256": _sha256(inventory_path)},
            "report_coverage": {
                "sha256": _sha256(report_root / "coverage_manifest.json"),
                "used_fields": "records/*/report/bundle.json event_identity_anchor_and_detector_firewall_allowlist_only",
            },
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
            "records": [
                {
                    "recording_id": row["recording_id"],
                    "source_signal_sha256": row["source_signal_sha256"],
                    "event_count": row["event_count"],
                    "prediction_status": row["prediction_status"],
                    "ranking": row["ranking"],
                }
                for row in record_rows
            ],
            "patients": patient_rows,
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
            "records": final["cohort"]["records"],
            "records_with_prediction": final["cohort"]["records_with_detector_events_and_prediction"],
            "patients": final["cohort"]["patients"],
            "events": final["cohort"]["EEG_only_detector_events"],
            "targets_loaded": False,
            "FZ_or_PZ_samples_read": False,
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
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--modeling", type=Path, default=DEFAULT_MODELING)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--skip-source-hash-verification", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.progress_every < 1:
        raise ValueError("progress-every must be positive")
    manifest, tensors = run(args)
    output = publish(args.output, manifest, tensors)
    print(
        json.dumps(
            {
                "output": str(output),
                "records": manifest["cohort"]["records"],
                "records_with_prediction": manifest["cohort"]["records_with_detector_events_and_prediction"],
                "events": manifest["cohort"]["EEG_only_detector_events"],
                "private_targets_loaded": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
