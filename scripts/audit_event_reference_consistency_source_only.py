#!/usr/bin/env python3
"""Materialize a target-free C-CAR19/C-REF19 event-consistency audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
from typing import Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.soz.event_phenotype_producer import (  # noqa: E402
    EVENT_PHENOTYPE_PRODUCER_SCHEMA,
    EventPhenotypeProducerIdentity,
    produce_event_scalp_phenotype,
)
from src.soz.data.edf import (  # noqa: E402
    CausalEDFConfig,
    load_standard19_edf_event,
)
from src.soz.event_reference_consistency import (  # noqa: E402
    EVENT_REFERENCE_TEMPORAL_TOLERANCE_SEC,
    assess_event_reference_consistency,
)
from src.soz.fine_temporal_evidence import (  # noqa: E402
    FineTemporalEvidence,
    extract_fine_temporal_evidence,
)
from src.soz.preprocessing_arm_runtime import (  # noqa: E402
    CAUSAL_REFERENCE_PAIR_SCHEMA,
)


DEFAULT_SOURCE_RECEIPT = (
    ROOT / "outputs/tusz_ictal_master_manifest_v4_20260809_preflight/receipt.json"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT = (
    ROOT / "outputs/event_reference_consistency_source_only_n64_20260811.json"
)
OUTPUT_SCHEMA = "soz_event_reference_consistency_source_only_audit_v1"


def _load_object(path: Path) -> dict[str, object]:
    source = path.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError("Source receipt must be a canonical regular file")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Source receipt must contain one JSON object")
    return value


def _select_one_event_per_patient(
    events: Sequence[object], *, limit: int
) -> tuple[Mapping[str, object], ...]:
    rows = sorted(
        (value for value in events if isinstance(value, Mapping)),
        key=lambda row: (str(row.get("patient_id", "")), str(row.get("event_id", ""))),
    )
    selected: list[Mapping[str, object]] = []
    patients: set[str] = set()
    for row in rows:
        patient = str(row.get("patient_id", "")).strip()
        if not patient or patient in patients:
            continue
        patients.add(patient)
        selected.append(row)
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError("Insufficient unique source patients for requested audit")
    return tuple(selected)


def _safe_edf(root: Path, relative_value: object) -> Path:
    relative = PurePosixPath(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".edf":
        raise ValueError("Unsafe relative EDF path")
    source = root.joinpath(*relative.parts)
    for component in (source, *source.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("EDF path cannot traverse symlinks")
    resolved = source.resolve(strict=True)
    if resolved.relative_to(root).as_posix() != relative.as_posix():
        raise ValueError("EDF path escaped the pinned TUSZ root")
    return resolved


def _tensor_digest(evidence: FineTemporalEvidence) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("features", evidence.features),
        ("composite_trace", evidence.composite_trace),
        ("dominant_frequency_hz", evidence.dominant_frequency_hz),
        ("window_center_sec", evidence.window_center_sec),
        ("node_change_detected", evidence.node_change_detected),
        ("node_change_latency_sec", evidence.node_change_latency_sec),
        ("bipolar_change_detected", evidence.bipolar_change_detected),
        ("bipolar_change_latency_sec", evidence.bipolar_change_latency_sec),
    ):
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    target = path.absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, target)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def audit(
    *,
    source_receipt: Path,
    tusz_root: Path,
    output: Path,
    limit: int,
    progress_every: int,
) -> dict[str, object]:
    if isinstance(limit, bool) or int(limit) < 1:
        raise ValueError("limit must be a positive integer")
    source = _load_object(source_receipt)
    if source.get("schema_version") != "tusz_ictal_training_manifest_v4.0.0":
        raise ValueError("Expected the frozen TUSZ concept-source v4 receipt")
    raw_events = source.get("events")
    preprocess = source.get("preprocess_config")
    if not isinstance(raw_events, list) or not isinstance(preprocess, Mapping):
        raise TypeError("Source receipt lacks its event roster/preprocessing contract")
    car_config = CausalEDFConfig(**dict(preprocess))
    if not car_config.apply_car19:
        raise ValueError("Frozen source preprocessing is not C-CAR19")
    ref_config = replace(car_config, apply_car19=False)
    selected = _select_one_event_per_patient(raw_events, limit=int(limit))
    raw_root = tusz_root.resolve(strict=True)
    if not raw_root.is_dir() or raw_root.is_symlink():
        raise ValueError("TUSZ root must be a canonical directory")
    if output.absolute().exists() or output.absolute().is_symlink():
        raise FileExistsError(output.absolute())

    rows: list[dict[str, object]] = []
    state_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    started = time.monotonic()
    for position, row in enumerate(selected, start=1):
        patient = str(row.get("patient_id", "")).strip()
        event = str(row.get("event_id", "")).strip()
        onset = float(row.get("event_t0_sec"))
        edf_sha = str(row.get("edf_sha256", "")).strip()
        if (
            not patient
            or not event
            or not math.isfinite(onset)
            or len(edf_sha) != 64
        ):
            raise ValueError("Selected source event has invalid identity/timing")
        edf = _safe_edf(raw_root, row.get("relative_edf_path"))
        car_loaded = load_standard19_edf_event(edf, onset, config=car_config)
        ref_loaded = load_standard19_edf_event(edf, onset, config=ref_config)
        if (
            car_loaded.edf_receipt.edf_sha256 != edf_sha
            or ref_loaded.edf_receipt.edf_sha256 != edf_sha
        ):
            raise ValueError("Source manifest and paired EDF receipts disagree")
        car = car_loaded.window.data.detach().cpu().float().contiguous()
        ref = ref_loaded.window.data.detach().cpu().float().contiguous()
        if tuple(car.shape) != (19, 12_000) or tuple(ref.shape) != (19, 12_000):
            raise ValueError("Paired event windows must both have shape [19,12000]")
        car_evidence = extract_fine_temporal_evidence(
            car, sfreq_hz=car_loaded.window.sfreq_hz
        )
        ref_evidence = extract_fine_temporal_evidence(
            ref, sfreq_hz=ref_loaded.window.sfreq_hz
        )
        primary = produce_event_scalp_phenotype(
            car,
            car_evidence,
            identity=EventPhenotypeProducerIdentity(
                patient_pseudonym=patient,
                event_pseudonym=event,
                signal_artifact_sha256=edf_sha,
                evidence_artifact_sha256=_tensor_digest(car_evidence),
            ),
            event_anchor_coordinate_sec=onset,
            time_coordinate_semantics="recording_start_seconds",
            sfreq_hz=car_loaded.window.sfreq_hz,
            reference_arm_id="C-CAR19",
        )
        sensitivity = produce_event_scalp_phenotype(
            ref,
            ref_evidence,
            identity=EventPhenotypeProducerIdentity(
                patient_pseudonym=patient,
                event_pseudonym=event,
                signal_artifact_sha256=edf_sha,
                evidence_artifact_sha256=_tensor_digest(ref_evidence),
            ),
            event_anchor_coordinate_sec=onset,
            time_coordinate_semantics="recording_start_seconds",
            sfreq_hz=ref_loaded.window.sfreq_hz,
            reference_arm_id="C-REF19",
        )
        consistency = assess_event_reference_consistency(primary, sensitivity)
        state = consistency.receipt.montage_stability or "not_assessed"
        state_counts[state] = state_counts.get(state, 0) + 1
        for reason in consistency.receipt.reason_codes:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        rows.append(
            {
                "patient_id": patient,
                "event_id": event,
                "relative_edf_path": str(row.get("relative_edf_path")),
                "global_t0_sec": onset,
                "primary_status": primary.status,
                "sensitivity_status": sensitivity.status,
                "consistency_receipt": asdict(consistency.receipt),
                "consistency_receipt_sha256": consistency.receipt.receipt_sha256,
                "bound_primary_event": asdict(consistency.event),
            }
        )
        if progress_every > 0 and (
            position % progress_every == 0 or position == len(selected)
        ):
            print(
                json.dumps(
                    {
                        "completed": position,
                        "total": len(selected),
                        "elapsed_sec": round(time.monotonic() - started, 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    payload: dict[str, object] = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "target_free_source_only_event_reference_consistency_audit",
        "event_phenotype_producer_schema": EVENT_PHENOTYPE_PRODUCER_SCHEMA,
        "reference_pair_schema": CAUSAL_REFERENCE_PAIR_SCHEMA,
        "temporal_alignment_tolerance_sec": (
            EVENT_REFERENCE_TEMPORAL_TOLERANCE_SEC
        ),
        "scientific_boundary": {
            "measurement": "same_event_bipolar_phenotype_consistency",
            "independent_cortical_soz_replication": False,
            "common_reference_cancels_in_bipolar_derivations": True,
            "paired_preprocessing": (
                "same_frozen_config_and_crop_with_only_apply_car19_changed"
            ),
            "allowed_use": "report_fact_or_abstention_only",
            "soz_score_modification_allowed": False,
            "model_or_arm_selection_allowed": False,
        },
        "access_receipt": {
            "selected_patient_count": len(rows),
            "selected_event_count": len(rows),
            "selection": "lexical_first_event_per_unique_patient",
            "raw_public_tusz_eeg_loaded": True,
            "tusz_native_target_values_loaded": False,
            "deepsoz_target_values_loaded": False,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "localization_scores_loaded": False,
            "training_performed": False,
            "threshold_selection_performed": False,
        },
        "counts": {
            "consistency_state": dict(sorted(state_counts.items())),
            "unassessed_reason": dict(sorted(reason_counts.items())),
        },
        "events": rows,
        "elapsed_sec": time.monotonic() - started,
    }
    _atomic_write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", type=Path, default=DEFAULT_SOURCE_RECEIPT)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=8)
    args = parser.parse_args()
    result = audit(
        source_receipt=args.source_receipt,
        tusz_root=args.tusz_root,
        output=args.output,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {"output": str(args.output), "counts": result["counts"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
