#!/usr/bin/env python3
"""Audit the target-free event-phenotype producer on public TUSZ source EEG."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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

from src.soz.data.edf import CausalEDFConfig, load_standard19_edf_event  # noqa: E402
from src.soz.event_phenotype_producer import (  # noqa: E402
    EVENT_PHENOTYPE_PRODUCER_SCHEMA,
    EventPhenotypeProducerIdentity,
    produce_event_scalp_phenotype,
)
from src.soz.fine_temporal_evidence import (  # noqa: E402
    FineTemporalEvidence,
    extract_fine_temporal_evidence,
)


DEFAULT_SOURCE_RECEIPT = (
    ROOT
    / "outputs/tusz_ictal_master_manifest_v4_20260809_preflight/receipt.json"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT = ROOT / "outputs/event_phenotype_source_only_n64_20260811.json"
OUTPUT_SCHEMA = "soz_event_phenotype_source_only_audit_v1"


def _load_object(path: Path) -> dict[str, object]:
    source = path.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError("Source receipt must be a canonical regular file")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Source receipt must contain a JSON object")
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


def _summary(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "minimum": float(tensor.min().item()),
        "median": float(tensor.median().item()),
        "mean": float(tensor.mean().item()),
        "maximum": float(tensor.max().item()),
    }


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
        raise TypeError("Source receipt lacks events/preprocess configuration")
    config = CausalEDFConfig(**dict(preprocess))
    if not config.apply_car19:
        raise ValueError("Event phenotype audit requires frozen primary C-CAR19")
    selected = _select_one_event_per_patient(raw_events, limit=int(limit))
    raw_root = tusz_root.resolve(strict=True)
    if not raw_root.is_dir() or raw_root.is_symlink():
        raise ValueError("TUSZ root must be a canonical directory")
    target = output.absolute()
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)

    rows: list[dict[str, object]] = []
    started = time.monotonic()
    for position, row in enumerate(selected, start=1):
        patient = str(row.get("patient_id", "")).strip()
        event = str(row.get("event_id", "")).strip()
        onset = float(row.get("event_t0_sec"))
        if not patient or not event or not math.isfinite(onset):
            raise ValueError("Selected source event has invalid identity/timing")
        edf = _safe_edf(raw_root, row.get("relative_edf_path"))
        loaded = load_standard19_edf_event(edf, onset, config=config)
        evidence = extract_fine_temporal_evidence(
            loaded.window.data,
            sfreq_hz=loaded.window.sfreq_hz,
        )
        result = produce_event_scalp_phenotype(
            loaded.window.data,
            evidence,
            identity=EventPhenotypeProducerIdentity(
                patient_pseudonym=patient,
                event_pseudonym=event,
                signal_artifact_sha256=loaded.edf_receipt.edf_sha256,
                evidence_artifact_sha256=_tensor_digest(evidence),
            ),
            event_anchor_coordinate_sec=onset,
            time_coordinate_semantics="recording_start_seconds",
            sfreq_hz=loaded.window.sfreq_hz,
        )
        phenotype = None if result.phenotype is None else asdict(result.phenotype)
        rows.append(
            {
                "patient_id": patient,
                "event_id": event,
                "relative_edf_path": str(row.get("relative_edf_path")),
                "global_t0_sec": onset,
                "status": result.status,
                "reason_codes": list(result.reason_codes),
                "detected_bipolar_edge_count": result.detected_bipolar_edge_count,
                "phenotype": phenotype,
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

    reportable = [row for row in rows if row["status"] == "reportable"]
    rhythm_counts: dict[str, int] = {}
    first_edge_counts: dict[str, int] = {}
    onset_latencies: list[float] = []
    later_delays: list[float] = []
    for row in reportable:
        phenotype = row["phenotype"]
        if not isinstance(phenotype, Mapping):
            raise RuntimeError("Reportable row lost its phenotype")
        rhythm = phenotype.get("rhythm_state")
        rhythm_key = "none" if rhythm is None else str(rhythm)
        rhythm_counts[rhythm_key] = rhythm_counts.get(rhythm_key, 0) + 1
        for edge in phenotype.get("first_visible_derivations", ()):
            first_edge_counts[str(edge)] = first_edge_counts.get(str(edge), 0) + 1
        onset_latencies.append(
            float(phenotype["onset_start_sec"]) - float(row["global_t0_sec"])
        )
        delay = phenotype.get("later_visible_delay_sec")
        if delay is not None:
            later_delays.append(float(delay))

    payload: dict[str, object] = {
        "schema_version": OUTPUT_SCHEMA,
        "producer_schema": EVENT_PHENOTYPE_PRODUCER_SCHEMA,
        "status": "target_free_source_only_descriptive_audit",
        "access_receipt": {
            "selected_patient_count": len(rows),
            "selected_event_count": len(rows),
            "selection": "lexical_first_event_per_unique_patient",
            "tusz_native_target_values_loaded": False,
            "deepsoz_target_values_loaded": False,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "training_performed": False,
            "threshold_selection_performed": False,
        },
        "counts": {
            "reportable": len(reportable),
            "abstained": len(rows) - len(reportable),
            "rhythm_state": dict(sorted(rhythm_counts.items())),
            "later_visible": len(later_delays),
            "first_visible_derivations": dict(sorted(first_edge_counts.items())),
        },
        "descriptive": {
            "onset_latency_relative_to_global_t0_sec": _summary(onset_latencies),
            "later_visible_delay_sec": _summary(later_delays),
            "detected_bipolar_edge_count": _summary(
                [float(row["detected_bipolar_edge_count"]) for row in rows]
            ),
        },
        "events": rows,
        "elapsed_sec": time.monotonic() - started,
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
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
            {
                "output": str(args.output),
                "counts": result["counts"],
                "descriptive": result["descriptive"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
