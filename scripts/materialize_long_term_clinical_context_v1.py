#!/usr/bin/env python3
"""Materialize a PHI-free long-recording EDF/Excel source-context sidecar."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.long_term_clinical_context import (  # noqa: E402
    build_long_term_clinical_context,
)


DETECTOR_ALIGNED_FROZEN_EVENT_REGISTRY_SCHEMA = (
    "clinical_eeg_detector_aligned_frozen_event_registry_v1"
)
TRANSITION_DETECTION_MANIFEST_SCHEMA = "long_term_seizure_detection_manifest_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REGISTRY_ID_RE = re.compile(r"^LTFRZ-[0-9a-f]{24}$")
_REGISTRY_KEYS = {
    "schema_version",
    "registry_id",
    "recording_id",
    "patient_id",
    "source_signal_sha256",
    "recording_duration_seconds",
    "source_transition_manifest_id",
    "source_transition_manifest_sha256",
    "candidate_semantics",
    "selection_decision",
    "event_id_policy",
    "selected_event_count",
    "events",
}
_REGISTRY_EVENT_KEYS = {
    "eeg_event_id",
    "recording_id",
    "patient_id",
    "source_signal_sha256",
    "event_anchor_recording_seconds",
    "source_candidate_id",
    "source_start_offset_seconds",
    "source_stop_offset_seconds",
    "source_candidate_score",
    "source_candidate_semantics",
    "source_selection_decision",
}
_CONTEXT_EVENT_KEYS = {
    "eeg_event_id",
    "recording_id",
    "patient_id",
    "source_signal_sha256",
    "event_anchor_recording_seconds",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
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


def _strict_mapping(
    value: object,
    *,
    required: set[str],
    context: str,
    allow_extra: bool = False,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    keys = set(value)
    missing = required - keys
    extra = keys - required
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")
    if extra and not allow_extra:
        raise ValueError(f"{context} has unknown keys: {sorted(extra)}")
    return {str(key): item for key, item in value.items()}


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _finite_number(value: object, context: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{context} must be numeric")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    return number


def _stable_detector_event_id(recording_id: str, sequence_index: int) -> str:
    namespace, separator, recording_token = recording_id.partition("-")
    if not separator or not namespace or not recording_token:
        raise ValueError("recording_id cannot form a stable detector event ID")
    return f"{namespace}-E-{recording_token}-{sequence_index:03d}"


def validate_detector_aligned_frozen_event_registry(
    value: object,
) -> dict[str, Any]:
    """Validate the auditable detector-candidate to event-ID freeze receipt."""

    data = _strict_mapping(
        value, required=_REGISTRY_KEYS, context="detector-aligned event registry"
    )
    if data["schema_version"] != DETECTOR_ALIGNED_FROZEN_EVENT_REGISTRY_SCHEMA:
        raise ValueError("detector-aligned event registry schema drifted")
    registry_id = data["registry_id"]
    if not isinstance(registry_id, str) or _REGISTRY_ID_RE.fullmatch(registry_id) is None:
        raise ValueError("detector-aligned registry_id is invalid")
    recording_id = data["recording_id"]
    patient_id = data["patient_id"]
    if not isinstance(recording_id, str) or re.fullmatch(
        r"^(?:PRIV|SYNTH|DEID)-R[A-Z0-9._-]{1,55}$", recording_id
    ) is None:
        raise ValueError("registry recording_id is not de-identified")
    if not isinstance(patient_id, str) or re.fullmatch(
        r"^(?:PRIV|SYNTH|DEID)-P[A-Z0-9._-]{1,55}$", patient_id
    ) is None:
        raise ValueError("registry patient_id is not de-identified")
    signal_sha = _sha256(data["source_signal_sha256"], "registry signal SHA-256")
    duration = _finite_number(
        data["recording_duration_seconds"], "registry recording duration"
    )
    if duration <= 0:
        raise ValueError("registry recording duration must be positive")
    transition_id = data["source_transition_manifest_id"]
    if not isinstance(transition_id, str) or re.fullmatch(
        r"^LTDET-[0-9a-f]{24}$", transition_id
    ) is None:
        raise ValueError("source transition manifest ID is invalid")
    transition_sha = _sha256(
        data["source_transition_manifest_sha256"],
        "source transition manifest SHA-256",
    )
    if data["candidate_semantics"] != "review_candidate_not_confirmed_seizure":
        raise ValueError("detector candidate semantics were promoted")
    if data["selection_decision"] != "selected_for_event_analysis":
        raise ValueError("detector selection decision drifted")
    if (
        data["event_id_policy"]
        != "recording_time_order_from_selected_detector_candidates_v1"
    ):
        raise ValueError("detector event ID policy drifted")
    selected_count = data["selected_event_count"]
    if isinstance(selected_count, bool) or not isinstance(selected_count, int):
        raise TypeError("selected_event_count must be an integer")
    if selected_count < 1:
        raise ValueError("detector event registry must contain selected events")
    raw_events = data["events"]
    if not isinstance(raw_events, list):
        raise TypeError("detector event registry events must be a list")
    if len(raw_events) != selected_count:
        raise ValueError("selected_event_count does not match registry events")

    events: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    anchors: set[float] = set()
    for index, raw in enumerate(raw_events, start=1):
        event = _strict_mapping(
            raw,
            required=_REGISTRY_EVENT_KEYS,
            context=f"detector registry event {index}",
        )
        expected_event_id = _stable_detector_event_id(recording_id, index)
        if event["eeg_event_id"] != expected_event_id:
            raise ValueError("detector event ID is not stable recording-time order")
        if event["recording_id"] != recording_id or event["patient_id"] != patient_id:
            raise ValueError("detector registry event identity binding mismatch")
        if event["source_signal_sha256"] != signal_sha:
            raise ValueError("detector registry event signal binding mismatch")
        candidate_id = event["source_candidate_id"]
        if not isinstance(candidate_id, str) or re.fullmatch(
            r"^CAND-[0-9a-f]{24}$", candidate_id
        ) is None:
            raise ValueError("source detector candidate ID is invalid")
        if candidate_id in candidate_ids:
            raise ValueError("source detector candidate ID repeats")
        candidate_ids.add(candidate_id)
        start = _finite_number(
            event["source_start_offset_seconds"], "candidate start offset"
        )
        stop = _finite_number(
            event["source_stop_offset_seconds"], "candidate stop offset"
        )
        anchor = _finite_number(
            event["event_anchor_recording_seconds"], "candidate anchor offset"
        )
        score = _finite_number(event["source_candidate_score"], "candidate score")
        if not (0.0 <= start <= anchor <= stop <= duration):
            raise ValueError("detector candidate interval is outside the recording")
        if not 0.0 <= score <= 1.0:
            raise ValueError("detector candidate score is outside [0,1]")
        if anchor in anchors:
            raise ValueError("detector event anchors repeat")
        anchors.add(anchor)
        if event["source_candidate_semantics"] != data["candidate_semantics"]:
            raise ValueError("candidate/event semantics mismatch")
        if event["source_selection_decision"] != data["selection_decision"]:
            raise ValueError("candidate/event selection decision mismatch")
        events.append(
            {
                "eeg_event_id": expected_event_id,
                "recording_id": recording_id,
                "patient_id": patient_id,
                "source_signal_sha256": signal_sha,
                "event_anchor_recording_seconds": anchor,
                "source_candidate_id": candidate_id,
                "source_start_offset_seconds": start,
                "source_stop_offset_seconds": stop,
                "source_candidate_score": score,
                "source_candidate_semantics": data["candidate_semantics"],
                "source_selection_decision": data["selection_decision"],
            }
        )
    if [item["event_anchor_recording_seconds"] for item in events] != sorted(anchors):
        raise ValueError("detector registry events are not in recording-time order")
    normalized_without_id = {
        "schema_version": DETECTOR_ALIGNED_FROZEN_EVENT_REGISTRY_SCHEMA,
        "recording_id": recording_id,
        "patient_id": patient_id,
        "source_signal_sha256": signal_sha,
        "recording_duration_seconds": duration,
        "source_transition_manifest_id": transition_id,
        "source_transition_manifest_sha256": transition_sha,
        "candidate_semantics": data["candidate_semantics"],
        "selection_decision": data["selection_decision"],
        "event_id_policy": data["event_id_policy"],
        "selected_event_count": selected_count,
        "events": events,
    }
    expected_registry_id = "LTFRZ-" + _canonical_sha256(normalized_without_id)[:24]
    if registry_id != expected_registry_id:
        raise ValueError("registry_id does not bind detector event lineage")
    return {
        "schema_version": DETECTOR_ALIGNED_FROZEN_EVENT_REGISTRY_SCHEMA,
        "registry_id": registry_id,
        **normalized_without_id,
    }


def build_detector_aligned_frozen_event_registry(
    transition_manifest: object,
    *,
    source_transition_manifest_sha256: str,
    expected_selected_count: int | None = None,
) -> dict[str, Any]:
    """Freeze selected transition candidates into stable time-ordered events."""

    required_manifest_keys = {
        "schema_version",
        "manifest_id",
        "recording_id",
        "patient_pseudonym",
        "source_signal_sha256",
        "recording_duration_seconds",
        "candidate_semantics",
        "merge_candidates",
    }
    manifest = _strict_mapping(
        transition_manifest,
        required=required_manifest_keys,
        context="transition detection manifest",
        allow_extra=True,
    )
    if manifest["schema_version"] != TRANSITION_DETECTION_MANIFEST_SCHEMA:
        raise ValueError("transition detection manifest schema drifted")
    manifest_sha = _sha256(
        source_transition_manifest_sha256, "transition detection manifest SHA-256"
    )
    recording_id = manifest["recording_id"]
    patient_id = manifest["patient_pseudonym"]
    signal_sha = _sha256(manifest["source_signal_sha256"], "transition signal SHA-256")
    duration = _finite_number(
        manifest["recording_duration_seconds"], "transition recording duration"
    )
    candidates = manifest["merge_candidates"]
    if not isinstance(candidates, list):
        raise TypeError("transition merge_candidates must be a list")
    selected: list[dict[str, Any]] = []
    required_candidate_keys = {
        "candidate_id",
        "start_offset_seconds",
        "stop_offset_seconds",
        "anchor_offset_seconds",
        "score",
        "decision",
        "semantics",
    }
    for raw in candidates:
        candidate = _strict_mapping(
            raw,
            required=required_candidate_keys,
            context="transition merge candidate",
            allow_extra=True,
        )
        if candidate["decision"] != "selected_for_event_analysis":
            continue
        if candidate["semantics"] != manifest["candidate_semantics"]:
            raise ValueError("selected detector candidate semantics mismatch")
        selected.append(candidate)
    selected.sort(
        key=lambda item: (
            _finite_number(item["anchor_offset_seconds"], "candidate anchor"),
            str(item["candidate_id"]),
        )
    )
    if expected_selected_count is not None and len(selected) != expected_selected_count:
        raise ValueError("selected detector candidate count does not match expectation")
    if not selected:
        raise ValueError("transition manifest contains no selected candidates")
    events: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected, start=1):
        events.append(
            {
                "eeg_event_id": _stable_detector_event_id(str(recording_id), index),
                "recording_id": recording_id,
                "patient_id": patient_id,
                "source_signal_sha256": signal_sha,
                "event_anchor_recording_seconds": candidate["anchor_offset_seconds"],
                "source_candidate_id": candidate["candidate_id"],
                "source_start_offset_seconds": candidate["start_offset_seconds"],
                "source_stop_offset_seconds": candidate["stop_offset_seconds"],
                "source_candidate_score": candidate["score"],
                "source_candidate_semantics": candidate["semantics"],
                "source_selection_decision": candidate["decision"],
            }
        )
    without_id = {
        "schema_version": DETECTOR_ALIGNED_FROZEN_EVENT_REGISTRY_SCHEMA,
        "recording_id": recording_id,
        "patient_id": patient_id,
        "source_signal_sha256": signal_sha,
        "recording_duration_seconds": duration,
        "source_transition_manifest_id": manifest["manifest_id"],
        "source_transition_manifest_sha256": manifest_sha,
        "candidate_semantics": manifest["candidate_semantics"],
        "selection_decision": "selected_for_event_analysis",
        "event_id_policy": "recording_time_order_from_selected_detector_candidates_v1",
        "selected_event_count": len(events),
        "events": events,
    }
    registry = {
        "schema_version": DETECTOR_ALIGNED_FROZEN_EVENT_REGISTRY_SCHEMA,
        "registry_id": "LTFRZ-" + _canonical_sha256(without_id)[:24],
        **without_id,
    }
    return validate_detector_aligned_frozen_event_registry(registry)


def _frozen_event_inputs(value: object) -> list[dict[str, object]]:
    if isinstance(value, Mapping) and set(value) == {"events"}:
        events = value["events"]
        if not isinstance(events, list):
            raise TypeError("legacy frozen-events JSON events must be a list")
        return [dict(item) for item in events]
    registry = validate_detector_aligned_frozen_event_registry(value)
    return [
        {key: event[key] for key in _CONTEXT_EVENT_KEYS}
        for event in registry["events"]
    ]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _normalized_relative(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("relative EDF path must be a non-empty string")
    text = value.strip().replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("relative EDF path is unsafe")
    if path.suffix.lower() != ".edf":
        raise ValueError("relative EDF path must identify an EDF")
    return path.as_posix()


def _resolve_source(root: Path, relative_text: str) -> Path:
    candidate = root
    for part in PurePosixPath(relative_text).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("EDF path must not traverse a symlink")
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(root)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("EDF source must be a regular file")
    return resolved


def _resolve_annotation_csv(path: Path) -> Path:
    """Resolve one regular CSV without accepting a directory or symlink.

    ``Path("")`` becomes ``Path(".")``.  Checking the resolved object before
    opening it gives that common CLI mistake a deterministic fail-closed error
    rather than an ``IsADirectoryError`` halfway through materialization.
    """

    if path.is_symlink():
        raise ValueError("EDF annotations CSV must not be a symlink")
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("EDF annotations CSV must be a regular file")
    if resolved.suffix.lower() != ".csv":
        raise ValueError("EDF annotations source must be a CSV")
    return resolved


def _zero_duration(value: object) -> bool:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and math.isclose(number, 0.0, abs_tol=1e-9)


def _annotation_rows_for_recording(
    annotations_path: Path,
    *,
    relative_edf: str,
) -> list[dict[str, object]]:
    """Read matching point annotations and safely skip bad row paths.

    Real aggregate exports can contain separator/summary rows with an empty
    ``edf_path`` as well as stale absolute or parent-traversing paths.  Those
    rows cannot bind to the selected recording and are skipped before any
    description is released to the context builder.  A missing CSV column is
    different: it makes the complete source uninterpretable and therefore
    fails closed instead of silently producing an empty sidecar.
    """

    required_columns = {
        "edf_path",
        "duration_ann_sec",
        "onset_sec",
        "description",
    }
    rows: list[dict[str, object]] = []
    with annotations_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        if not required_columns.issubset(fieldnames):
            raise ValueError("EDF annotations CSV is missing required columns")
        for source_row, row in enumerate(reader, start=2):
            try:
                row_relative_edf = _normalized_relative(row.get("edf_path", ""))
            except (TypeError, ValueError):
                # Invalid source paths cannot match a safely resolved recording.
                continue
            if row_relative_edf != relative_edf:
                continue
            if not _zero_duration(row.get("duration_ann_sec")):
                continue
            try:
                offset = float(str(row.get("onset_sec", "")).strip())
            except (TypeError, ValueError):
                continue
            if not math.isfinite(offset):
                continue
            rows.append(
                {
                    "source_row": source_row,
                    "source_row_sha256": _canonical_sha256(row),
                    "recording_offset_seconds": offset,
                    "description": str(row.get("description", "") or ""),
                }
            )
    return rows


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    target = path.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--edf-annotations", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--excel-review",
        type=Path,
        default=None,
        help="Optional PHI-free typed Excel binding JSON; never raw cells.",
    )
    parser.add_argument("--frozen-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def materialize_context(
    *,
    binding: Mapping[str, object],
    annotations_path: Path,
    dataset_root: Path,
    frozen_events: Sequence[Mapping[str, object]],
    output: Path,
    workbook_sha256s: Sequence[str] = (),
    excel_review_bindings: Sequence[Mapping[str, object]] = (),
) -> dict[str, Any]:
    """Materialize one context from in-memory de-identified binding records."""

    expected_binding_keys = {
        "recording_id",
        "patient_id",
        "source_signal_sha256",
        "recording_duration_seconds",
        "relative_edf_path",
    }
    if set(binding) != expected_binding_keys:
        raise ValueError("recording binding has missing or unknown keys")
    relative_edf = _normalized_relative(binding["relative_edf_path"])
    resolved_dataset_root = dataset_root.resolve(strict=True)
    if not resolved_dataset_root.is_dir():
        raise ValueError("dataset root must be a directory")
    source_edf = _resolve_source(resolved_dataset_root, relative_edf)
    if _sha256_file(source_edf) != binding["source_signal_sha256"]:
        raise ValueError("recording binding source_signal_sha256 does not match EDF")

    resolved_annotations_path = _resolve_annotation_csv(annotations_path)
    rows = _annotation_rows_for_recording(
        resolved_annotations_path, relative_edf=relative_edf
    )
    workbook_hashes: Sequence[str] = ()
    if not isinstance(workbook_sha256s, Sequence) or isinstance(
        workbook_sha256s, (str, bytes)
    ):
        raise TypeError("workbook_sha256s must be a sequence")
    workbook_hashes = workbook_sha256s
    if not isinstance(excel_review_bindings, Sequence) or isinstance(
        excel_review_bindings, (str, bytes)
    ):
        raise TypeError("excel_review_bindings must be a sequence")

    context = build_long_term_clinical_context(
        recording_id=binding["recording_id"],
        patient_id=binding["patient_id"],
        source_signal_sha256=binding["source_signal_sha256"],
        recording_duration_seconds=binding["recording_duration_seconds"],
        edf_annotations_sha256=_sha256_file(resolved_annotations_path),
        annotation_rows=rows,
        workbook_sha256s=workbook_hashes,
        excel_review_bindings=excel_review_bindings,
        frozen_events=frozen_events,
    )
    _atomic_json(output, context)
    return context


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    binding = _json(args.binding)
    frozen = _json(args.frozen_events)
    frozen_event_inputs = _frozen_event_inputs(frozen)
    workbook_hashes: Sequence[str] = ()
    excel_bindings: Sequence[Mapping[str, object]] = ()
    if args.excel_review is not None:
        review = _json(args.excel_review)
        if set(review) != {"workbook_sha256s", "excel_review_bindings"}:
            raise ValueError("Excel review JSON has missing or unknown keys")
        workbook_hashes = review["workbook_sha256s"]
        excel_bindings = review["excel_review_bindings"]

    context = materialize_context(
        binding=binding,
        annotations_path=args.edf_annotations,
        dataset_root=args.dataset_root,
        frozen_events=frozen_event_inputs,
        output=args.output,
        workbook_sha256s=workbook_hashes,
        excel_review_bindings=excel_bindings,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "recording_id": context["recording_id"],
                "mapped_annotation_items": len(context["annotations"]),
                "excel_onset_observations": len(context["excel_onset_observations"]),
                "frozen_event_count": len(context["event_associations"]),
                "raw_text_or_path_released": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
