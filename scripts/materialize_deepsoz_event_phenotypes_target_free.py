#!/usr/bin/env python3
"""Materialize target-free DeepSOZ event phenotypes from public TUSZ EEG.

The input boundary is deliberately narrow: one frozen DeepSOZ *signal*
preflight receipt and the local public TUSZ EDF tree.  DeepSOZ target values,
TUSZ channel targets, localization scores, and private data are neither
accepted nor loaded.  Every signal-eligible event is replayed under paired
C-CAR19/C-REF19 preprocessing and passed through the already frozen event
phenotype, reference-consistency, and later-visible-region producers.

This artifact is report/abstention evidence.  It cannot change an SOZ score,
select a model, or be interpreted as propagation or cortical-onset truth.
"""

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

from src.soz.clinical_reporting import (  # noqa: E402
    EventScalpPhenotypeAbstention,
    EventScalpPhenotypeEvidence,
)
from src.soz.data.edf import (  # noqa: E402
    CausalEDFConfig,
    load_standard19_edf_event,
)
from src.soz.event_phenotype_producer import (  # noqa: E402
    EVENT_PHENOTYPE_PRODUCER_SCHEMA,
    EVENT_PHENOTYPE_PRODUCER_SCHEMA_V2,
    EventPhenotypeProducerIdentity,
    EventPhenotypeProductionResult,
    produce_event_scalp_phenotype,
    produce_event_scalp_phenotype_v2,
)
from src.soz.event_reference_consistency import (  # noqa: E402
    EVENT_REFERENCE_TEMPORAL_TOLERANCE_SEC,
    assess_event_reference_consistency,
)
from src.soz.fine_temporal_evidence import (  # noqa: E402
    FineTemporalEvidence,
    extract_fine_temporal_evidence,
)
from src.soz.later_visible_region_producer import (  # noqa: E402
    LATER_VISIBLE_REGION_PRODUCER_SCHEMA,
    LATER_VISIBLE_REGION_RECEIPT_SCHEMA,
    build_later_visible_region_receipt,
    produce_later_visible_region,
)
from src.soz.preprocessing_arm_runtime import (  # noqa: E402
    CAUSAL_REFERENCE_PAIR_SCHEMA,
)


DEFAULT_SIGNAL_PREFLIGHT = (
    ROOT
    / "outputs/deepsoz_signal_preflight_v2_20260809_current"
    / "deepsoz_signal_preflight.json"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT = (
    ROOT / "outputs/deepsoz_event_phenotype_target_free_oof_v1_20260812.json"
)

OUTPUT_SCHEMA = "soz_deepsoz_event_phenotype_target_free_oof_v1"
OUTPUT_SCHEMA_V2 = "soz_deepsoz_event_phenotype_target_free_oof_v2"
OUTPUT_STATUS = (
    "completed_target_free_development_signal_application_not_evaluation"
)
SUPPORTED_PREFLIGHT_ARTIFACT_SCHEMAS = frozenset(
    {
        "soz_deepsoz_signal_preflight_artifact_v1",
        "soz_deepsoz_signal_preflight_artifact_v2",
    }
)
SUPPORTED_PREFLIGHT_RECEIPT_SCHEMAS = frozenset(
    {
        "soz_deepsoz_signal_preflight_v1",
        "soz_deepsoz_signal_preflight_v2",
    }
)
PREFLIGHT_SERIALIZATION = "canonical_json_utf8_newline_no_pickle"
SUPPORTED_PREPROCESS_SCHEMAS = frozenset(
    {
        "standard19_causal_edf_event_v1",
        "standard19_causal_edf_event_v2",
    }
)
EXPECTED_PREPROCESS_SCHEMA = "standard19_causal_edf_event_v2"
EXPECTED_EVENT_SHAPE = (19, 12_000)
EXPECTED_EVENT_DTYPE = "torch.float32"
_SHA256_CHARACTERS = frozenset("0123456789abcdef")

SLOT_NAMES = (
    "sustained_change_interval",
    "first_visible_derivations",
    "rhythm_state",
    "frequency_range_hz",
    "later_visible_delay",
    "later_visible_destination",
    "later_visible_region",
    "montage_stability",
    "artifact_assessment",
)


def _canonical_json_bytes(value: object, *, newline: bool) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value, newline=False)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _absolute_no_symlink(path: str | Path, *, field: str) -> Path:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot traverse symlinks")
    return absolute


def _strict_json_object(path: Path) -> tuple[dict[str, object], bytes]:
    source = _absolute_no_symlink(path, field="signal preflight artifact")
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = source.read_bytes()

    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field is forbidden: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"Non-finite JSON constant is forbidden: {value}")

    try:
        parsed = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Signal preflight is not strict UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise TypeError("Signal preflight must contain one JSON object")
    if _canonical_json_bytes(parsed, newline=True) != raw:
        raise ValueError("Signal preflight is not canonical JSON")
    return parsed, raw


def _load_preflight_contract(
    path: Path,
) -> tuple[dict[str, object], tuple[Mapping[str, object], ...], dict[str, object]]:
    artifact, raw = _strict_json_object(path)
    expected_keys = {"schema_version", "receipt", "receipt_sha256", "serialization"}
    if set(artifact) != expected_keys:
        raise ValueError("Signal preflight artifact violates its closed wrapper schema")
    if artifact.get("schema_version") not in SUPPORTED_PREFLIGHT_ARTIFACT_SCHEMAS:
        raise ValueError("Unsupported signal preflight artifact schema")
    if artifact.get("serialization") != PREFLIGHT_SERIALIZATION:
        raise ValueError("Unsupported signal preflight serialization")
    receipt = artifact.get("receipt")
    if not isinstance(receipt, dict):
        raise TypeError("Signal preflight receipt must be a JSON object")
    if receipt.get("schema_version") not in SUPPORTED_PREFLIGHT_RECEIPT_SCHEMAS:
        raise ValueError("Unsupported signal preflight receipt schema")
    receipt_sha = _require_sha256(
        artifact.get("receipt_sha256"), field="receipt_sha256"
    )
    if receipt_sha != _canonical_sha256(receipt):
        raise ValueError("Signal preflight receipt SHA mismatch")

    preprocess = receipt.get("preprocess_config")
    events = receipt.get("events")
    if not isinstance(preprocess, dict) or not isinstance(events, list):
        raise TypeError("Signal preflight lacks preprocess_config/events")
    preprocess_schema = receipt.get("preprocess_schema")
    if preprocess_schema not in SUPPORTED_PREPROCESS_SCHEMAS:
        raise ValueError("Signal preflight preprocessing schema is unsupported")
    preprocess_sha = _require_sha256(
        receipt.get("preprocess_config_sha256"), field="preprocess_config_sha256"
    )
    expected_preprocess_sha = _canonical_sha256(
        {
            "preprocess_schema": preprocess_schema,
            "config": preprocess,
        }
    )
    if preprocess_sha != expected_preprocess_sha:
        raise ValueError("Signal preflight preprocessing SHA mismatch")
    if receipt.get("eligible_event_count") != len(events):
        raise ValueError("Signal preflight eligible-event count disagrees with roster")

    rows: list[Mapping[str, object]] = []
    event_ids: list[str] = []
    patient_ids: set[str] = set()
    for ordinal, value in enumerate(events):
        if not isinstance(value, Mapping):
            raise TypeError(f"Signal preflight event {ordinal} must be an object")
        event_id = str(value.get("event_id", "")).strip()
        patient_id = str(value.get("patient_id", "")).strip()
        local_patient_id = str(value.get("local_patient_id", "")).strip()
        if not event_id or not patient_id or not local_patient_id:
            raise ValueError("Signal preflight event identity is incomplete")
        if value.get("preprocess_config_sha256") != preprocess_sha:
            raise ValueError("Event preprocessing SHA differs from preflight")
        if tuple(value.get("processed_window_shape", ())) != EXPECTED_EVENT_SHAPE:
            raise ValueError("Signal preflight event does not contain [19,12000]")
        if value.get("processed_window_dtype") != EXPECTED_EVENT_DTYPE:
            raise ValueError("Signal preflight event dtype is not torch.float32")
        _require_sha256(value.get("edf_sha256"), field="event.edf_sha256")
        _require_sha256(
            value.get("processed_window_sha256"),
            field="event.processed_window_sha256",
        )
        onset = value.get("global_t0_sec")
        if (
            isinstance(onset, bool)
            or not isinstance(onset, (int, float))
            or not math.isfinite(float(onset))
            or float(onset) < 0
        ):
            raise ValueError("Signal preflight event t0 must be finite and non-negative")
        event_index = value.get("global_event_index")
        if type(event_index) is not int or event_index < 0:
            raise ValueError("Signal preflight global_event_index is invalid")
        event_ids.append(event_id)
        patient_ids.add(patient_id)
        rows.append(value)
    if event_ids != sorted(event_ids) or len(set(event_ids)) != len(event_ids):
        raise ValueError("Signal preflight event roster must be sorted and unique")
    if receipt.get("eligible_patient_count") != len(patient_ids):
        raise ValueError("Signal preflight eligible-patient count disagrees with roster")
    declared_roster_sha = _require_sha256(
        receipt.get("eligible_event_roster_sha256"),
        field="eligible_event_roster_sha256",
    )
    if declared_roster_sha != _canonical_sha256(tuple(sorted(event_ids))):
        raise ValueError("Signal preflight eligible-event roster SHA mismatch")

    source_identity = {
        "artifact_schema_version": artifact["schema_version"],
        "receipt_schema_version": receipt["schema_version"],
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "receipt_sha256": receipt_sha,
        "eligible_event_roster_sha256": declared_roster_sha,
        "eligible_event_count": len(rows),
        "eligible_patient_count": len(patient_ids),
        "preprocess_schema": receipt["preprocess_schema"],
        "preprocess_config_sha256": preprocess_sha,
        "policy": str(receipt.get("policy", "")),
    }
    return dict(preprocess), tuple(rows), source_identity


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


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous()
    metadata = _canonical_json_bytes(
        {"dtype": str(values.dtype), "shape": list(values.shape)}, newline=False
    )
    raw = values.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _evidence_sha256(evidence: FineTemporalEvidence) -> str:
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
        metadata = _canonical_json_bytes(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            newline=False,
        )
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _serialize_production(
    result: EventPhenotypeProductionResult,
    *,
    arm_id: str,
    processed_window_sha256: str,
) -> dict[str, object]:
    return {
        "arm_id": arm_id,
        "processed_window_sha256": processed_window_sha256,
        "status": result.status,
        "reason_codes": list(result.reason_codes),
        "detected_bipolar_edge_count": result.detected_bipolar_edge_count,
        "phenotype": None if result.phenotype is None else asdict(result.phenotype),
        "abstention": None if result.abstention is None else asdict(result.abstention),
    }


def _slot_availability(
    event: EventScalpPhenotypeEvidence | EventScalpPhenotypeAbstention,
    *,
    later_region_available: bool,
) -> dict[str, bool]:
    if isinstance(event, EventScalpPhenotypeAbstention):
        return {name: False for name in SLOT_NAMES}
    if event.artifact_assessed is not None or event.artifact_types or (
        event.artifact_burden is not None
    ):
        raise ValueError("Frozen event producer cannot populate artifact facts")
    values = {
        "sustained_change_interval": True,
        "first_visible_derivations": bool(event.first_visible_derivations),
        "rhythm_state": event.rhythm_state is not None,
        "frequency_range_hz": event.frequency_range_hz is not None,
        "later_visible_delay": event.later_visible_delay_sec is not None,
        "later_visible_destination": bool(event.later_visible_derivations),
        "later_visible_region": bool(later_region_available),
        "montage_stability": event.montage_stability is not None,
        "artifact_assessment": False,
    }
    if set(values) != set(SLOT_NAMES):
        raise RuntimeError("Event fact availability schema drifted")
    return values


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    target = _absolute_no_symlink(path, field="output artifact")
    if os.path.lexists(target):
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json_bytes(payload, newline=True))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, target)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _increment(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def materialize(
    *,
    signal_preflight: Path,
    tusz_root: Path,
    output: Path,
    limit: int | None,
    progress_every: int,
    producer_version: str = "v1",
) -> dict[str, object]:
    """Replay the frozen target-free event evidence pipeline."""

    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    if type(progress_every) is not int or progress_every < 0:
        raise ValueError("progress_every must be a non-negative integer")
    if producer_version == "v1":
        producer_schema = EVENT_PHENOTYPE_PRODUCER_SCHEMA
        output_schema = OUTPUT_SCHEMA
        producer = produce_event_scalp_phenotype
        temporal_variation_semantics = "pooled_edge_and_time_v1_historical_replay"
    elif producer_version == "v2":
        producer_schema = EVENT_PHENOTYPE_PRODUCER_SCHEMA_V2
        output_schema = OUTPUT_SCHEMA_V2
        producer = produce_event_scalp_phenotype_v2
        temporal_variation_semantics = "edge_aggregated_then_offset_only_v2"
    else:
        raise ValueError("producer_version must be v1 or v2")
    target = _absolute_no_symlink(output, field="output artifact")
    if os.path.lexists(target):
        raise FileExistsError(target)

    preprocess, roster, source_identity = _load_preflight_contract(signal_preflight)
    config_car = CausalEDFConfig(**preprocess)
    if not config_car.apply_car19:
        raise ValueError("Signal preflight primary preprocessing is not C-CAR19")
    config_ref = replace(config_car, apply_car19=False)
    if limit is not None:
        if limit > len(roster):
            raise ValueError("limit exceeds the signal-eligible event roster")
        selected = roster[:limit]
    else:
        selected = roster

    raw_root = _absolute_no_symlink(tusz_root, field="TUSZ root")
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)

    events: list[dict[str, object]] = []
    primary_status: dict[str, int] = {}
    sensitivity_status: dict[str, int] = {}
    reference_state: dict[str, int] = {}
    later_region_state: dict[str, int] = {}
    slot_counts = {name: 0 for name in SLOT_NAMES}
    started = time.monotonic()

    for position, source_row in enumerate(selected, start=1):
        patient_id = str(source_row["patient_id"]).strip()
        local_patient_id = str(source_row["local_patient_id"]).strip()
        event_id = str(source_row["event_id"]).strip()
        relative_edf_path = str(source_row["relative_edf_path"])
        onset = float(source_row["global_t0_sec"])
        edf_sha = _require_sha256(
            source_row["edf_sha256"], field="event.edf_sha256"
        )
        expected_car_sha = _require_sha256(
            source_row["processed_window_sha256"],
            field="event.processed_window_sha256",
        )
        edf = _safe_edf(raw_root, relative_edf_path)
        loaded_car = load_standard19_edf_event(edf, onset, config=config_car)
        loaded_ref = load_standard19_edf_event(edf, onset, config=config_ref)
        if (
            loaded_car.edf_receipt.edf_sha256 != edf_sha
            or loaded_ref.edf_receipt.edf_sha256 != edf_sha
        ):
            raise ValueError("Preflight and paired EDF receipts disagree")
        car = loaded_car.window.data.detach().cpu().float().contiguous()
        ref = loaded_ref.window.data.detach().cpu().float().contiguous()
        if tuple(car.shape) != EXPECTED_EVENT_SHAPE or tuple(ref.shape) != (
            EXPECTED_EVENT_SHAPE
        ):
            raise ValueError("Paired event windows must both be [19,12000]")
        if not torch.isfinite(car).all() or not torch.isfinite(ref).all():
            raise ValueError("Paired event windows must be finite")
        car_sha = _tensor_sha256(car)
        ref_sha = _tensor_sha256(ref)
        if car_sha != expected_car_sha:
            raise ValueError("C-CAR19 replay differs from signal preflight")
        if abs(float(loaded_car.window.sfreq_hz) - 200.0) > 1e-9 or abs(
            float(loaded_ref.window.sfreq_hz) - 200.0
        ) > 1e-9:
            raise ValueError("Paired event windows must both be 200 Hz")

        car_evidence = extract_fine_temporal_evidence(car, sfreq_hz=200.0)
        ref_evidence = extract_fine_temporal_evidence(ref, sfreq_hz=200.0)
        primary = producer(
            car,
            car_evidence,
            identity=EventPhenotypeProducerIdentity(
                patient_pseudonym=patient_id,
                event_pseudonym=event_id,
                signal_artifact_sha256=edf_sha,
                evidence_artifact_sha256=_evidence_sha256(car_evidence),
                extractor_model_version=producer_schema,
            ),
            event_anchor_coordinate_sec=onset,
            time_coordinate_semantics="recording_start_seconds",
            sfreq_hz=200.0,
            reference_arm_id="C-CAR19",
        )
        sensitivity = producer(
            ref,
            ref_evidence,
            identity=EventPhenotypeProducerIdentity(
                patient_pseudonym=patient_id,
                event_pseudonym=event_id,
                signal_artifact_sha256=edf_sha,
                evidence_artifact_sha256=_evidence_sha256(ref_evidence),
                extractor_model_version=producer_schema,
            ),
            event_anchor_coordinate_sec=onset,
            time_coordinate_semantics="recording_start_seconds",
            sfreq_hz=200.0,
            reference_arm_id="C-REF19",
        )
        consistency = assess_event_reference_consistency(primary, sensitivity)
        bound_event = consistency.event

        later_receipt = None
        later_receipt_sha = None
        if isinstance(bound_event, EventScalpPhenotypeEvidence) and (
            bound_event.later_visible_derivations
        ):
            later_production = produce_later_visible_region(
                bound_event.later_visible_derivations
            )
            if later_production.status != "mapped":
                raise RuntimeError("Observed later-visible edges failed deterministic map")
            typed_later_receipt = build_later_visible_region_receipt(
                later_production, bound_event.receipt
            )
            later_receipt = asdict(typed_later_receipt)
            later_receipt_sha = typed_later_receipt.receipt_sha256
            later_region = {
                "status": "mapped",
                "reason_codes": [],
                "receipt": later_receipt,
                "receipt_sha256": later_receipt_sha,
            }
        else:
            reason = (
                "primary_event_phenotype_abstained"
                if isinstance(bound_event, EventScalpPhenotypeAbstention)
                else "no_observed_later_visible_derivations"
            )
            later_region = {
                "status": "unavailable",
                "reason_codes": [reason],
                "receipt": None,
                "receipt_sha256": None,
            }

        availability = _slot_availability(
            bound_event, later_region_available=later_receipt is not None
        )
        for name, available in availability.items():
            if available:
                slot_counts[name] += 1
        _increment(primary_status, primary.status)
        _increment(sensitivity_status, sensitivity.status)
        _increment(
            reference_state,
            consistency.receipt.montage_stability or "unassessed",
        )
        _increment(later_region_state, str(later_region["status"]))

        events.append(
            {
                "ordinal": position - 1,
                "patient_id": patient_id,
                "local_patient_id": local_patient_id,
                "event_id": event_id,
                "relative_edf_path": relative_edf_path,
                "global_t0_sec": onset,
                "global_stop_sec": float(source_row["global_stop_sec"]),
                "global_event_index": int(source_row["global_event_index"]),
                "official_split": str(source_row["official_split"]),
                "model_split": str(source_row["model_split"]),
                "edf_sha256": edf_sha,
                "event_record_sha256": _require_sha256(
                    source_row["event_record_sha256"],
                    field="event.event_record_sha256",
                ),
                "edf_receipt_sha256": _require_sha256(
                    source_row["edf_receipt_sha256"],
                    field="event.edf_receipt_sha256",
                ),
                "signal_receipt_sha256": _require_sha256(
                    source_row["signal_receipt_sha256"],
                    field="event.signal_receipt_sha256",
                ),
                "processed_window_sha256": expected_car_sha,
                "status": primary.status,
                "reason_codes": list(primary.reason_codes),
                "phenotype": (
                    asdict(bound_event)
                    if isinstance(bound_event, EventScalpPhenotypeEvidence)
                    else None
                ),
                "abstention": (
                    asdict(bound_event)
                    if isinstance(bound_event, EventScalpPhenotypeAbstention)
                    else None
                ),
                "primary_arm": _serialize_production(
                    primary,
                    arm_id="C-CAR19",
                    processed_window_sha256=car_sha,
                ),
                "sensitivity_arm": _serialize_production(
                    sensitivity,
                    arm_id="C-REF19",
                    processed_window_sha256=ref_sha,
                ),
                "event_reference_consistency_receipt": asdict(
                    consistency.receipt
                ),
                "event_reference_consistency_receipt_sha256": (
                    consistency.receipt.receipt_sha256
                ),
                "later_visible_region": later_region,
                "slot_availability": availability,
            }
        )
        if progress_every and (
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
        "schema_version": output_schema,
        "status": OUTPUT_STATUS,
        "producer_schema": producer_schema,
        "reference_pair_schema": CAUSAL_REFERENCE_PAIR_SCHEMA,
        "event_reference_temporal_tolerance_sec": (
            EVENT_REFERENCE_TEMPORAL_TOLERANCE_SEC
        ),
        "later_visible_region_producer_schema": (
            LATER_VISIBLE_REGION_PRODUCER_SCHEMA
        ),
        "later_visible_region_receipt_schema": LATER_VISIBLE_REGION_RECEIPT_SCHEMA,
        "scientific_boundary": {
            "measurement": "target_free_scalp_visible_event_phenotype",
            "allowed_use": "report_fact_availability_or_abstention_only",
            "soz_score_modification_allowed": False,
            "cortical_soz_claim_allowed": False,
            "propagation_truth_claim_allowed": False,
            "earliest_physical_electrode_claim_allowed": False,
            "temporal_variation_semantics": temporal_variation_semantics,
            "clinical_temporal_evolution_claim_allowed": False,
        },
        "source_preflight": source_identity,
        "access_receipt": {
            "input_event_selection": (
                "entire_sorted_signal_eligible_preflight_roster"
                if limit is None
                else "first_n_of_sorted_signal_eligible_preflight_roster_smoke"
            ),
            "limit": limit,
            "input_signal_eligible_event_count": len(roster),
            "materialized_event_count": len(events),
            "raw_public_tusz_eeg_loaded": True,
            "deepsoz_identity_roster_loaded": True,
            "deepsoz_preprocess_config_loaded": True,
            "deepsoz_target_values_loaded": False,
            "deepsoz_target_fields_accessed": False,
            "tusz_native_target_values_loaded": False,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "localization_scores_loaded": False,
            "training_performed": False,
            "model_selection_performed": False,
            "calibration_performed": False,
            "threshold_selection_performed": False,
            "producer_version": producer_version,
        },
        "counts": {
            "input_signal_eligible_patients": source_identity[
                "eligible_patient_count"
            ],
            "input_signal_eligible_events": len(roster),
            "materialized_patients": len({row["patient_id"] for row in events}),
            "materialized_events": len(events),
            "primary_status": dict(sorted(primary_status.items())),
            "sensitivity_status": dict(sorted(sensitivity_status.items())),
            "reference_state": dict(sorted(reference_state.items())),
            "later_visible_region_state": dict(sorted(later_region_state.items())),
            "slot_available": slot_counts,
        },
        "events": events,
        "elapsed_sec": time.monotonic() - started,
    }
    _atomic_write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signal-preflight", type=Path, default=DEFAULT_SIGNAL_PREFLIGHT
    )
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument(
        "--producer-version", choices=("v1", "v2"), default="v1"
    )
    args = parser.parse_args()
    result = materialize(
        signal_preflight=args.signal_preflight,
        tusz_root=args.tusz_root,
        output=args.output,
        limit=args.limit,
        progress_every=args.progress_every,
        producer_version=args.producer_version,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "schema_version": result["schema_version"],
                "counts": result["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
