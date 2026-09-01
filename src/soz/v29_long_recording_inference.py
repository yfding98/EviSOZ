"""Frozen v29 research rankings for arbitrary long-recording candidates.

The historical private v29 script was intentionally frozen to one 88-event
roster.  That roster is an evaluation artifact, not a model input contract.
This module reuses the unchanged five-fold H/direct-token inference functions
while accepting only candidates selected by a recording-level detector and
their immutable 60-second segment receipts.

No target, annotation, spreadsheet, diagnosis, or legacy event roster is an
input.  Outputs are research-only scalp-electrode rankings.  They are not a
seizure detector, cortical SOZ, epileptogenic zone, treatment target, SOTA
claim, calibrated error probability, or clinical conclusion.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping, Sequence

import torch
from safetensors.torch import load_file

from scripts.audit_trustworthy_soz_candidate_v21 import (
    _fold_h_only_probability,
)
from scripts.materialize_private_labram_evidence_v18 import _split_calls
from scripts.predict_private_labram_portable_equal_v29 import _direct_probability
from scripts.run_labram_rank1_direct_token_oof_v28 import (
    extract_rank1_phase_features,
)
from src.clinical_eeg_long_recording.schema import (
    BOUNDARY_POLICY,
    CANDIDATE_SEMANTICS,
    FIXED_EVENT_WINDOW_SECONDS,
    FIXED_SEGMENT_DURATION_SECONDS,
    MINIMUM_ANALYZABLE_ANCHOR_SECONDS,
    SOZ_INTERPRETATION_STATUS,
    canonical_payload_sha256,
    validate_long_term_event_segment_receipt,
    validate_long_term_seizure_detection_manifest,
)
from src.clinical_eeg_long_recording.analysis_selection import (
    ANALYSIS_REJECTION_ID_PREFIX,
    ANALYSIS_REJECTION_SCHEMA_VERSION,
    ANALYSIS_SELECTION_ID_PREFIX,
    ANALYSIS_SELECTION_SCHEMA_VERSION,
    QC_FAILED_CHECKS,
    QC_STAGES,
    bind_long_term_eeg_analysis_selection,
    validate_analysis_rejection_receipt,
    validate_long_term_eeg_analysis_selection,
)
from src.soz.data.edf import (
    CausalEDFConfig,
    EDFEventEligibilityError,
    load_standard19_edf_event,
)
from src.soz.geometry import STANDARD_19
from src.soz.models.labram import (
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    bind_labram_record_positions,
)
from src.soz.models.labram_peft import OfficialLaBraMFrozenPrefixEncoder
from src.soz.v11_reasoner import (
    V11_CANDIDATE_MASK,
    extract_block9_phase_contrasts,
)


INPUT_SCHEMA_VERSION = "soz_v29_detector_candidate_input_v1"
FILTERED_INPUT_SCHEMA_VERSION = "soz_v29_filtered_detector_candidate_input_v2"
PRE_RANKING_WINDOW_RECEIPT_SCHEMA_VERSION = (
    "long_term_pre_ranking_event_window_receipt_v1"
)
EVENT_ID_ASSIGNMENT_SCHEMA_VERSION = "long_term_candidate_event_id_assignment_v1"
DETECTOR_ALIGNED_EVENT_REGISTRY_SCHEMA_VERSION = (
    "clinical_eeg_detector_aligned_frozen_event_registry_v1"
)
OUTPUT_SCHEMA_VERSION = "soz_v29_detector_candidate_rankings_v1"
FILTERED_OUTPUT_SCHEMA_VERSION = "soz_v29_filtered_detector_candidate_rankings_v2"
OUTPUT_STATUS = "completed_frozen_v29_research_candidate_ranking"
METHOD_ID = "v29_equal_H_D_probability_ensemble"
INFERENCE_POLICY = "frozen_five_fold_equal_probability_research_reuse_v1"
PROCESSED_WINDOW_HASH_POLICY = "deepsoz_signal_tensor_sha256_v1"
TENSOR_FILE = "v29_candidate_rankings.safetensors"
N_FOLDS = 5

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELING = Path("/mnt/hd1/dyf/workspace/LaBraM/modeling_finetune.py")
DEFAULT_CHECKPOINT = Path(
    "/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth"
)
DEFAULT_DIRECT_STATES = (
    ROOT
    / "outputs/labram_rank1_direct_token_oof_v28_20260815/"
    "model_and_oof.safetensors"
)
DEFAULT_H_STATES = (
    ROOT
    / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815/"
    "outer_fold_states.safetensors"
)
DEFAULT_PUBLIC_FREEZE_MANIFEST = (
    ROOT
    / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815/"
    "manifest.json"
)

EXPECTED_DIRECT_STATES_SHA256 = (
    "0b5ffaf0ed504c36e01a0be28676f7797a703a6ef57079dc743362984a4351a9"
)
EXPECTED_H_STATES_SHA256 = (
    "18b69f5e2fc718d2668b3a727f9a3f7bf0da33a613896d939559260ad3009b98"
)
EXPECTED_PUBLIC_FREEZE_MANIFEST_SHA256 = (
    "2db07cbc1319eac8a90c3f5cdf45ebfaf784ee383eb65ae33e6de7e110ea7906"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TIME_TOLERANCE = 1e-6


def _strict_object(
    value: object,
    *,
    required: Sequence[str],
    context: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    expected = set(required)
    missing = expected.difference(value)
    extra = set(value).difference(expected)
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{context} has unknown keys: {sorted(extra)}")
    return deepcopy(value)


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be an ASCII identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"resource must not be a symbolic link: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"resource must be a regular file: {path}")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _verified_file(path: str | Path, expected_sha256: str, context: str) -> Path:
    candidate = Path(path)
    resolved = candidate.resolve(strict=True)
    expected = _sha256(expected_sha256, f"{context}.expected_sha256")
    actual = _file_sha256(resolved)
    if actual != expected:
        raise ValueError(f"{context} SHA-256 does not match the frozen resource")
    return resolved


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"JSON contains duplicate key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"JSON contains invalid constant {value!r}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=invalid_constant,
    )
    if type(payload) is not dict:
        raise TypeError("frozen public manifest must be an object")
    return payload


def _tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor):
        raise TypeError("tensor receipt requires a torch.Tensor")
    tensor = value.detach().cpu().contiguous()
    metadata = _canonical_json_bytes(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    )
    raw = tensor.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _receipt_payload(value: object, context: str) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        payload = asdict(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError(f"{context} must be a dataclass or mapping")
    _canonical_json_bytes(payload)
    return payload


def _validate_float_tensor(
    value: object,
    *,
    shape_tail: tuple[int, ...],
    context: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{context} must be a floating-point tensor")
    if value.ndim != len(shape_tail) + 1 or tuple(value.shape[1:]) != shape_tail:
        raise ValueError(f"{context} must have shape [E,{','.join(map(str, shape_tail))}]")
    if value.shape[0] < 1:
        raise ValueError(f"{context} must contain at least one candidate event")
    if value.requires_grad or not torch.isfinite(value).all():
        raise ValueError(f"{context} must be detached and finite")
    return value.detach().cpu().float().contiguous()


def _validate_state_masks(
    direct_states: Mapping[str, torch.Tensor],
    h_states: Mapping[str, torch.Tensor],
) -> None:
    for fold in range(N_FOLDS):
        direct_name = f"outer_state.fold{fold}.candidate_mask"
        h_name = f"outer{fold}.frozen_labram_only.candidate_mask"
        for states, name in ((direct_states, direct_name), (h_states, h_name)):
            mask = states.get(name)
            if not isinstance(mask, torch.Tensor) or not torch.equal(
                mask.detach().cpu().bool(), V11_CANDIDATE_MASK
            ):
                raise ValueError(f"frozen v29 candidate mask drifted: {name}")


def infer_v29_probabilities(
    h_features: torch.Tensor,
    phase_features: torch.Tensor,
    *,
    direct_states: Mapping[str, torch.Tensor],
    h_states: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply the unchanged five-fold v29 ensemble to any positive event count."""

    h = _validate_float_tensor(
        h_features, shape_tail=(19, 600), context="v29 H features"
    )
    phase = _validate_float_tensor(
        phase_features,
        shape_tail=(19, 5, 200),
        context="v29 direct-token phase features",
    )
    if len(h) != len(phase):
        raise ValueError("v29 H/direct-token event counts differ")
    if not isinstance(direct_states, Mapping) or not isinstance(h_states, Mapping):
        raise TypeError("v29 frozen states must be mappings")
    _validate_state_masks(direct_states, h_states)
    direct_fold = torch.stack(
        [_direct_probability(phase, direct_states, fold) for fold in range(N_FOLDS)],
        dim=1,
    ).float().contiguous()
    h_fold = torch.stack(
        [_fold_h_only_probability(h, h_states, fold) for fold in range(N_FOLDS)],
        dim=1,
    ).float().contiguous()
    equal_fold = (0.5 * direct_fold + 0.5 * h_fold).contiguous()
    probability = equal_fold.mean(dim=1).contiguous()
    expected_fold_shape = (len(h), N_FOLDS, 19)
    for name, value in (
        ("rank1_direct_fold_probability", direct_fold),
        ("h_only_fold_probability", h_fold),
        ("portable_equal_fold_probability", equal_fold),
    ):
        if tuple(value.shape) != expected_fold_shape or not torch.isfinite(value).all():
            raise RuntimeError(f"v29 {name} contract failed")
        if not torch.allclose(
            value.sum(dim=2), torch.ones((len(h), N_FOLDS)), atol=1e-6, rtol=0
        ):
            raise RuntimeError(f"v29 {name} rows do not sum to one")
    if tuple(probability.shape) != (len(h), 19) or not torch.isfinite(
        probability
    ).all():
        raise RuntimeError("v29 ensemble probability contract failed")
    if not torch.allclose(
        probability.sum(dim=1), torch.ones(len(h)), atol=1e-6, rtol=0
    ):
        raise RuntimeError("v29 ensemble probability rows do not sum to one")
    if not torch.equal(
        probability[:, ~V11_CANDIDATE_MASK], torch.zeros((len(h), 1))
    ):
        raise RuntimeError("v29 excluded PZ carrier acquired non-zero probability")
    return {
        "portable_equal_probability": probability,
        "portable_equal_fold_probability": equal_fold,
        "rank1_direct_fold_probability": direct_fold,
        "h_only_fold_probability": h_fold,
        "candidate_mask": V11_CANDIDATE_MASK.clone(),
    }


def _same_time(left: object, right: object) -> bool:
    return abs(_finite(left, "time value") - _finite(right, "time value")) <= (
        _TIME_TOLERANCE
    )


def validate_v29_event_id_assignment(payload: object) -> dict[str, Any]:
    """Validate the target-free candidate-to-report-event identity mapping."""

    data = _strict_object(
        payload,
        required=("schema_version", "recording_id", "assignments"),
        context="v29 event ID assignment",
    )
    if data["schema_version"] != EVENT_ID_ASSIGNMENT_SCHEMA_VERSION:
        raise ValueError("v29 event ID assignment schema mismatch")
    recording_id = _identifier(data["recording_id"], "assignment.recording_id")
    raw_assignments = data["assignments"]
    if not isinstance(raw_assignments, list):
        raise TypeError("assignment.assignments must be an array")
    assignments: list[dict[str, str]] = []
    candidate_ids: set[str] = set()
    event_ids: set[str] = set()
    for index, raw in enumerate(raw_assignments):
        item = _strict_object(
            raw,
            required=("candidate_id", "eeg_event_id"),
            context=f"assignment.assignments[{index}]",
        )
        candidate_id = _identifier(
            item["candidate_id"], f"assignment.assignments[{index}].candidate_id"
        )
        event_id = _identifier(
            item["eeg_event_id"], f"assignment.assignments[{index}].eeg_event_id"
        )
        if candidate_id in candidate_ids or event_id in event_ids:
            raise ValueError("event ID assignment repeats a candidate or EEG event ID")
        candidate_ids.add(candidate_id)
        event_ids.add(event_id)
        assignments.append({"candidate_id": candidate_id, "eeg_event_id": event_id})
    assignments.sort(key=lambda item: (item["candidate_id"], item["eeg_event_id"]))
    return {
        "schema_version": EVENT_ID_ASSIGNMENT_SCHEMA_VERSION,
        "recording_id": recording_id,
        "assignments": assignments,
    }


def resolve_v29_event_id_assignment(
    payload: object,
    detection_manifest: object,
) -> dict[str, Any]:
    """Accept either the small assignment schema or the frozen registry.

    The registry adapter verifies its complete detector lineage before reducing
    it to the two target-free identity fields consumed by the ranker.
    """

    manifest = validate_long_term_seizure_detection_manifest(detection_manifest)
    if type(payload) is not dict:
        raise TypeError("v29 event identity input must be an object")
    if payload.get("schema_version") == EVENT_ID_ASSIGNMENT_SCHEMA_VERSION:
        assignment = validate_v29_event_id_assignment(payload)
        if assignment["recording_id"] != manifest["recording_id"]:
            raise ValueError(
                "event ID assignment recording_id differs from detector manifest"
            )
        return assignment
    data = _strict_object(
        payload,
        required=(
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
        ),
        context="detector-aligned frozen event registry",
    )
    if data["schema_version"] != DETECTOR_ALIGNED_EVENT_REGISTRY_SCHEMA_VERSION:
        raise ValueError("unsupported v29 event identity input schema")
    registry_id = _identifier(data["registry_id"], "registry.registry_id")
    recording_id = _identifier(data["recording_id"], "registry.recording_id")
    patient = _identifier(data["patient_id"], "registry.patient_id")
    source_hash = _sha256(data["source_signal_sha256"], "registry source SHA-256")
    duration = _finite(data["recording_duration_seconds"], "registry duration")
    transition_id = _identifier(
        data["source_transition_manifest_id"], "registry transition manifest ID"
    )
    _sha256(
        data["source_transition_manifest_sha256"],
        "registry transition manifest SHA-256",
    )
    if (
        recording_id != manifest["recording_id"]
        or patient != manifest["patient_pseudonym"]
        or source_hash != manifest["source_signal_sha256"]
        or not _same_time(duration, manifest["recording_duration_seconds"])
        or transition_id != manifest["manifest_id"]
    ):
        raise ValueError("detector event registry lineage differs from manifest")
    if (
        data["candidate_semantics"] != CANDIDATE_SEMANTICS
        or data["selection_decision"] != "selected_for_event_analysis"
        or data["event_id_policy"]
        != "recording_time_order_from_selected_detector_candidates_v1"
    ):
        raise ValueError("detector event registry selection policy drifted")
    selected = {
        candidate["candidate_id"]: candidate
        for candidate in manifest["merge_candidates"]
        if candidate["decision_available"] is True
        and candidate["decision"] == "selected_for_event_analysis"
    }
    count = data["selected_event_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count != len(selected):
        raise ValueError("detector event registry selected_event_count drifted")
    raw_events = data["events"]
    if not isinstance(raw_events, list) or len(raw_events) != count:
        raise ValueError("detector event registry events do not match selected count")
    registry_event_keys = (
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
    )
    assignments: list[dict[str, str]] = []
    seen_candidates: set[str] = set()
    seen_events: set[str] = set()
    previous_anchor = -math.inf
    for index, raw in enumerate(raw_events):
        event = _strict_object(
            raw,
            required=registry_event_keys,
            context=f"detector event registry.events[{index}]",
        )
        candidate_id = _identifier(
            event["source_candidate_id"], "registry event source_candidate_id"
        )
        event_id = _identifier(event["eeg_event_id"], "registry event eeg_event_id")
        candidate = selected.get(candidate_id)
        if candidate is None:
            raise ValueError("detector event registry references an unselected candidate")
        anchor = _finite(event["event_anchor_recording_seconds"], "registry event anchor")
        start = _finite(event["source_start_offset_seconds"], "registry event start")
        stop = _finite(event["source_stop_offset_seconds"], "registry event stop")
        score = _finite(event["source_candidate_score"], "registry event score")
        if (
            event["recording_id"] != recording_id
            or event["patient_id"] != patient
            or event["source_signal_sha256"] != source_hash
            or event["source_candidate_semantics"] != CANDIDATE_SEMANTICS
            or event["source_selection_decision"] != "selected_for_event_analysis"
            or not _same_time(anchor, candidate["anchor_offset_seconds"])
            or not _same_time(start, candidate["start_offset_seconds"])
            or not _same_time(stop, candidate["stop_offset_seconds"])
            or not _same_time(score, candidate["score"])
        ):
            raise ValueError("detector event registry candidate binding drifted")
        if candidate_id in seen_candidates or event_id in seen_events:
            raise ValueError("detector event registry repeats candidate/event identity")
        if anchor < previous_anchor - _TIME_TOLERANCE:
            raise ValueError("detector event registry is not in recording-time order")
        previous_anchor = anchor
        seen_candidates.add(candidate_id)
        seen_events.add(event_id)
        assignments.append({"candidate_id": candidate_id, "eeg_event_id": event_id})
    if seen_candidates != set(selected):
        raise ValueError("detector event registry does not cover all selected candidates")
    normalized_without_id = deepcopy(data)
    normalized_without_id.pop("registry_id")
    if registry_id != f"LTFRZ-{_canonical_sha256(normalized_without_id)[:24]}":
        raise ValueError("detector event registry ID does not bind its content")
    return {
        "schema_version": EVENT_ID_ASSIGNMENT_SCHEMA_VERSION,
        "recording_id": recording_id,
        "assignments": sorted(
            assignments,
            key=lambda item: (item["candidate_id"], item["eeg_event_id"]),
        ),
    }


def validate_v29_pre_ranking_window_receipt(payload: object) -> dict[str, Any]:
    """Validate a signal-only window receipt created before any report/ranking."""

    data = _strict_object(
        payload,
        required=(
            "schema_version",
            "window_receipt_id",
            "recording_id",
            "patient_pseudonym",
            "source_signal_sha256",
            "recording_duration_seconds",
            "candidate_id",
            "eeg_event_id",
            "candidate_anchor_offset_seconds",
            "requested_window_seconds",
            "window_start_offset_seconds",
            "window_stop_offset_seconds",
            "warmup_seconds_available",
            "post_anchor_seconds_available",
            "boundary_policy",
            "processed_window_hash_policy",
            "processed_window_sha256",
            "preprocessing_receipt_sha256",
            "content_boundary",
        ),
        context="v29 pre-ranking event window receipt",
    )
    if data["schema_version"] != PRE_RANKING_WINDOW_RECEIPT_SCHEMA_VERSION:
        raise ValueError("pre-ranking event window receipt schema mismatch")
    receipt_id = _identifier(data["window_receipt_id"], "window.window_receipt_id")
    recording_id = _identifier(data["recording_id"], "window.recording_id")
    patient = _identifier(data["patient_pseudonym"], "window.patient_pseudonym")
    source_hash = _sha256(data["source_signal_sha256"], "window.source_signal_sha256")
    duration = _finite(data["recording_duration_seconds"], "window recording duration")
    if duration <= 0:
        raise ValueError("window recording duration must be positive")
    candidate_id = _identifier(data["candidate_id"], "window.candidate_id")
    event_id = _identifier(data["eeg_event_id"], "window.eeg_event_id")
    anchor = _finite(data["candidate_anchor_offset_seconds"], "window candidate anchor")
    requested = data["requested_window_seconds"]
    if (
        not isinstance(requested, list)
        or len(requested) != 2
        or not all(
            _same_time(actual, expected)
            for actual, expected in zip(requested, FIXED_EVENT_WINDOW_SECONDS)
        )
    ):
        raise ValueError("pre-ranking window must request fixed [-12,+48] seconds")
    start = _finite(data["window_start_offset_seconds"], "window start")
    stop = _finite(data["window_stop_offset_seconds"], "window stop")
    warmup = _finite(data["warmup_seconds_available"], "window warmup")
    post = _finite(data["post_anchor_seconds_available"], "window post-anchor context")
    if not _same_time(start, anchor + FIXED_EVENT_WINDOW_SECONDS[0]):
        raise ValueError("pre-ranking window start must equal anchor minus 12 seconds")
    if not _same_time(stop, anchor + FIXED_EVENT_WINDOW_SECONDS[1]):
        raise ValueError("pre-ranking window stop must equal anchor plus 48 seconds")
    if not _same_time(stop - start, FIXED_SEGMENT_DURATION_SECONDS):
        raise ValueError("pre-ranking event window must contain exactly 60 seconds")
    if not _same_time(warmup, anchor) or not _same_time(post, duration - anchor):
        raise ValueError("pre-ranking window context must use the full-recording clock")
    if (
        anchor < MINIMUM_ANALYZABLE_ANCHOR_SECONDS - _TIME_TOLERANCE
        or start < -_TIME_TOLERANCE
        or stop > duration + _TIME_TOLERANCE
        or post < FIXED_EVENT_WINDOW_SECONDS[1] - _TIME_TOLERANCE
    ):
        raise ValueError("pre-ranking candidate lacks causal warmup or full fixed window")
    if data["boundary_policy"] != BOUNDARY_POLICY:
        raise ValueError("pre-ranking window boundary policy drifted")
    if data["processed_window_hash_policy"] != PROCESSED_WINDOW_HASH_POLICY:
        raise ValueError("pre-ranking processed-window hash policy drifted")
    processed_hash = _sha256(
        data["processed_window_sha256"], "window.processed_window_sha256"
    )
    preprocessing_hash = _sha256(
        data["preprocessing_receipt_sha256"],
        "window.preprocessing_receipt_sha256",
    )
    content = _strict_object(
        data["content_boundary"],
        required=(
            "clinical_report_payload_included",
            "waveform_figure_included",
            "research_soz_ranking_included",
        ),
        context="window.content_boundary",
    )
    if any(value is not False for value in content.values()):
        raise ValueError("pre-ranking window receipt contains downstream report content")
    result = {
        "schema_version": PRE_RANKING_WINDOW_RECEIPT_SCHEMA_VERSION,
        "window_receipt_id": receipt_id,
        "recording_id": recording_id,
        "patient_pseudonym": patient,
        "source_signal_sha256": source_hash,
        "recording_duration_seconds": duration,
        "candidate_id": candidate_id,
        "eeg_event_id": event_id,
        "candidate_anchor_offset_seconds": anchor,
        "requested_window_seconds": [
            float(FIXED_EVENT_WINDOW_SECONDS[0]),
            float(FIXED_EVENT_WINDOW_SECONDS[1]),
        ],
        "window_start_offset_seconds": start,
        "window_stop_offset_seconds": stop,
        "warmup_seconds_available": warmup,
        "post_anchor_seconds_available": post,
        "boundary_policy": BOUNDARY_POLICY,
        "processed_window_hash_policy": PROCESSED_WINDOW_HASH_POLICY,
        "processed_window_sha256": processed_hash,
        "preprocessing_receipt_sha256": preprocessing_hash,
        "content_boundary": {
            "clinical_report_payload_included": False,
            "waveform_figure_included": False,
            "research_soz_ranking_included": False,
        },
    }
    digest_source = deepcopy(result)
    digest_source["window_receipt_id"] = "CONTENT-ADDRESS-PENDING"
    expected_receipt_id = f"WIN-V29-{canonical_payload_sha256(digest_source)[:20]}"
    if receipt_id != expected_receipt_id:
        raise ValueError("pre-ranking window receipt ID does not bind its content")
    return result


def validate_v29_candidate_batch(payload: object) -> dict[str, Any]:
    """Strictly validate the detector/window-only input to frozen v29."""

    if type(payload) is not dict:
        raise TypeError("v29 detector candidate batch must be an object")
    schema_raw = payload.get("schema_version")
    filtered = schema_raw == FILTERED_INPUT_SCHEMA_VERSION
    if schema_raw not in (INPUT_SCHEMA_VERSION, FILTERED_INPUT_SCHEMA_VERSION):
        raise ValueError("v29 detector candidate batch schema mismatch")
    required = [
        "schema_version",
        "recording_id",
        "patient_pseudonym",
        "source_signal_sha256",
        "recording_duration_seconds",
        "detection_manifest_sha256",
        "candidate_semantics",
        "processed_window_hash_policy",
        "event_count",
        "events",
        "access_receipt",
    ]
    if filtered:
        required.append("analysis_selection_sha256")
    data = _strict_object(
        payload,
        required=tuple(required),
        context="v29 detector candidate batch",
    )
    recording_id = _identifier(data["recording_id"], "batch.recording_id")
    patient = _identifier(data["patient_pseudonym"], "batch.patient_pseudonym")
    source_hash = _sha256(data["source_signal_sha256"], "batch.source_signal_sha256")
    duration = _finite(data["recording_duration_seconds"], "batch recording duration")
    if duration <= 0:
        raise ValueError("batch recording duration must be positive")
    detection_hash = _sha256(
        data["detection_manifest_sha256"], "batch.detection_manifest_sha256"
    )
    selection_hash = (
        _sha256(
            data["analysis_selection_sha256"],
            "batch.analysis_selection_sha256",
        )
        if filtered
        else None
    )
    if data["candidate_semantics"] != CANDIDATE_SEMANTICS:
        raise ValueError("v29 candidate batch promotes detector candidates")
    if data["processed_window_hash_policy"] != PROCESSED_WINDOW_HASH_POLICY:
        raise ValueError("v29 candidate batch processed-window hash policy drifted")
    event_count = data["event_count"]
    if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count < 0:
        raise ValueError("v29 candidate batch event_count must be a non-negative integer")
    raw_events = data["events"]
    if not isinstance(raw_events, list) or len(raw_events) != event_count:
        raise ValueError("v29 candidate batch events do not match event_count")
    events: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    event_ids: set[str] = set()
    for index, raw in enumerate(raw_events):
        event = _strict_object(
            raw,
            required=(
                "candidate_id",
                "eeg_event_id",
                "candidate_anchor_offset_seconds",
                "window_start_offset_seconds",
                "window_stop_offset_seconds",
                "processed_window_sha256",
                "preprocessing_receipt_sha256",
                "pre_ranking_window_receipt_sha256",
                "pre_ranking_window_receipt",
            ),
            context=f"batch.events[{index}]",
        )
        window = validate_v29_pre_ranking_window_receipt(
            event["pre_ranking_window_receipt"]
        )
        candidate_id = _identifier(event["candidate_id"], "batch event candidate_id")
        event_id = _identifier(event["eeg_event_id"], "batch event eeg_event_id")
        anchor = _finite(event["candidate_anchor_offset_seconds"], "batch event anchor")
        start = _finite(event["window_start_offset_seconds"], "batch event start")
        stop = _finite(event["window_stop_offset_seconds"], "batch event stop")
        processed_hash = _sha256(
            event["processed_window_sha256"], "batch event processed_window_sha256"
        )
        preprocessing_hash = _sha256(
            event["preprocessing_receipt_sha256"],
            "batch event preprocessing_receipt_sha256",
        )
        receipt_hash = _sha256(
            event["pre_ranking_window_receipt_sha256"],
            "batch event pre_ranking_window_receipt_sha256",
        )
        expected_identity = {
            "recording_id": recording_id,
            "patient_pseudonym": patient,
            "source_signal_sha256": source_hash,
            "recording_duration_seconds": duration,
            "candidate_id": candidate_id,
            "eeg_event_id": event_id,
            "candidate_anchor_offset_seconds": anchor,
            "window_start_offset_seconds": start,
            "window_stop_offset_seconds": stop,
            "processed_window_sha256": processed_hash,
            "preprocessing_receipt_sha256": preprocessing_hash,
        }
        if any(not _same_time(window[key], value) for key, value in expected_identity.items() if isinstance(value, float)):
            raise ValueError("candidate batch time values differ from window receipt")
        if any(
            window[key] != value
            for key, value in expected_identity.items()
            if not isinstance(value, float)
        ):
            raise ValueError("candidate batch identity/hash differs from window receipt")
        if canonical_payload_sha256(window) != receipt_hash:
            raise ValueError("candidate batch window receipt SHA-256 mismatch")
        if candidate_id in candidate_ids or event_id in event_ids:
            raise ValueError("v29 candidate batch repeats candidate or EEG event identity")
        candidate_ids.add(candidate_id)
        event_ids.add(event_id)
        events.append(
            {
                "candidate_id": candidate_id,
                "eeg_event_id": event_id,
                "candidate_anchor_offset_seconds": anchor,
                "window_start_offset_seconds": start,
                "window_stop_offset_seconds": stop,
                "processed_window_sha256": processed_hash,
                "preprocessing_receipt_sha256": preprocessing_hash,
                "pre_ranking_window_receipt_sha256": receipt_hash,
                "pre_ranking_window_receipt": window,
            }
        )
    if events != sorted(
        events,
        key=lambda event: (
            event["candidate_anchor_offset_seconds"],
            event["eeg_event_id"],
        ),
    ):
        raise ValueError("v29 candidate batch events are not in recording-time order")
    access = _strict_object(
        data["access_receipt"],
        required=(
            "detector_candidates_loaded",
            "pre_ranking_window_receipts_loaded",
            "final_segment_receipts_loaded",
            "clinical_report_payloads_loaded",
            "waveform_figures_loaded",
            "legacy_88_event_roster_loaded",
            "event_targets_loaded",
            "edf_annotation_values_loaded",
            "excel_physician_observations_loaded",
            "input_research_soz_rankings_used",
        ),
        context="batch.access_receipt",
    )
    required_true = (
        "detector_candidates_loaded",
        "pre_ranking_window_receipts_loaded",
    )
    if any(access[key] is not True for key in required_true) or any(
        access[key] is not False for key in set(access).difference(required_true)
    ):
        raise ValueError("v29 candidate batch access boundary failed")
    result = {
        "schema_version": (
            FILTERED_INPUT_SCHEMA_VERSION if filtered else INPUT_SCHEMA_VERSION
        ),
        "recording_id": recording_id,
        "patient_pseudonym": patient,
        "source_signal_sha256": source_hash,
        "recording_duration_seconds": duration,
        "detection_manifest_sha256": detection_hash,
        "candidate_semantics": CANDIDATE_SEMANTICS,
        "processed_window_hash_policy": PROCESSED_WINDOW_HASH_POLICY,
        "event_count": event_count,
        "events": events,
        "access_receipt": access,
    }
    if selection_hash is not None:
        result["analysis_selection_sha256"] = selection_hash
    return result


def canonicalize_v29_candidate_batch(
    detection_manifest: object,
    pre_ranking_window_receipts: Sequence[object],
    *,
    analysis_selection: object | None = None,
) -> dict[str, Any]:
    """Bind detector candidates to signal-only windows, never final segments."""

    manifest = validate_long_term_seizure_detection_manifest(detection_manifest)
    if isinstance(pre_ranking_window_receipts, (str, bytes)) or not isinstance(
        pre_ranking_window_receipts, Sequence
    ):
        raise TypeError("pre_ranking_window_receipts must be an array")
    windows = [
        validate_v29_pre_ranking_window_receipt(item)
        for item in pre_ranking_window_receipts
    ]
    selected = {
        candidate["candidate_id"]: candidate
        for candidate in manifest["merge_candidates"]
        if candidate["decision_available"] is True
        and candidate["decision"] == "selected_for_event_analysis"
    }
    selection = None
    if analysis_selection is not None:
        selection = bind_long_term_eeg_analysis_selection(
            analysis_selection,
            manifest,
        )
        analyzable_ids = {
            item["candidate_id"]
            for item in selection["events"]
            if item["analysis_disposition"] == "analyzable"
        }
        selected = {
            candidate_id: candidate
            for candidate_id, candidate in selected.items()
            if candidate_id in analyzable_ids
        }
    by_candidate: dict[str, dict[str, Any]] = {}
    event_ids: set[str] = set()
    for window in windows:
        candidate_id = window["candidate_id"]
        if candidate_id in by_candidate:
            raise ValueError("pre-ranking windows repeat a detector candidate")
        if window["eeg_event_id"] in event_ids:
            raise ValueError("pre-ranking windows repeat an EEG event ID")
        event_ids.add(window["eeg_event_id"])
        candidate = selected.get(candidate_id)
        if candidate is None:
            raise ValueError("pre-ranking window does not bind a selected candidate")
        for key in ("recording_id", "patient_pseudonym", "source_signal_sha256"):
            if window[key] != manifest[key]:
                raise ValueError(f"window {key} does not match detector manifest")
        if not _same_time(
            window["recording_duration_seconds"],
            manifest["recording_duration_seconds"],
        ):
            raise ValueError("window duration does not match detector manifest")
        if not _same_time(
            window["candidate_anchor_offset_seconds"],
            candidate["anchor_offset_seconds"],
        ):
            raise ValueError("window anchor does not match detector candidate")
        by_candidate[candidate_id] = window
    if set(by_candidate) != set(selected):
        raise ValueError(
            "pre-ranking windows must exactly cover selected detector candidates"
        )

    events: list[dict[str, Any]] = []
    for candidate_id in selected:
        window = by_candidate[candidate_id]
        events.append(
            {
                "candidate_id": candidate_id,
                "eeg_event_id": window["eeg_event_id"],
                "candidate_anchor_offset_seconds": window[
                    "candidate_anchor_offset_seconds"
                ],
                "window_start_offset_seconds": window["window_start_offset_seconds"],
                "window_stop_offset_seconds": window["window_stop_offset_seconds"],
                "processed_window_sha256": window["processed_window_sha256"],
                "preprocessing_receipt_sha256": window[
                    "preprocessing_receipt_sha256"
                ],
                "pre_ranking_window_receipt_sha256": canonical_payload_sha256(window),
                "pre_ranking_window_receipt": window,
            }
        )
    events.sort(
        key=lambda event: (
            event["candidate_anchor_offset_seconds"],
            event["eeg_event_id"],
        )
    )
    payload = {
            "schema_version": (
                FILTERED_INPUT_SCHEMA_VERSION
                if selection is not None
                else INPUT_SCHEMA_VERSION
            ),
            "recording_id": manifest["recording_id"],
            "patient_pseudonym": manifest["patient_pseudonym"],
            "source_signal_sha256": manifest["source_signal_sha256"],
            "recording_duration_seconds": manifest["recording_duration_seconds"],
            "detection_manifest_sha256": canonical_payload_sha256(manifest),
            "candidate_semantics": CANDIDATE_SEMANTICS,
            "processed_window_hash_policy": PROCESSED_WINDOW_HASH_POLICY,
            "event_count": len(events),
            "events": events,
            "access_receipt": {
                "detector_candidates_loaded": True,
                "pre_ranking_window_receipts_loaded": True,
                "final_segment_receipts_loaded": False,
                "clinical_report_payloads_loaded": False,
                "waveform_figures_loaded": False,
                "legacy_88_event_roster_loaded": False,
                "event_targets_loaded": False,
                "edf_annotation_values_loaded": False,
                "excel_physician_observations_loaded": False,
                "input_research_soz_rankings_used": False,
            },
        }
    if selection is not None:
        payload["analysis_selection_sha256"] = canonical_payload_sha256(
            selection
        )
        selection_by_candidate = {
            item["candidate_id"]: item for item in selection["events"]
        }
        for event in events:
            selected_event = selection_by_candidate[event["candidate_id"]]
            if (
                selected_event["eeg_event_id"] != event["eeg_event_id"]
                or selected_event["pre_ranking_window_receipt_sha256"]
                != event["pre_ranking_window_receipt_sha256"]
                or selected_event["processed_window_sha256"]
                != event["processed_window_sha256"]
                or selected_event["preprocessing_receipt_sha256"]
                != event["preprocessing_receipt_sha256"]
            ):
                raise ValueError(
                    "filtered v29 candidate window differs from analysis selection"
                )
    return validate_v29_candidate_batch(payload)


@dataclass(frozen=True)
class ExtractedV29CandidateFeatures:
    """Compact target-free carriers and per-event replay receipts."""

    h_features: torch.Tensor
    phase_features: torch.Tensor
    event_receipts: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        h = _validate_float_tensor(
            self.h_features,
            shape_tail=(19, 600),
            context="extracted v29 H features",
        )
        phase = _validate_float_tensor(
            self.phase_features,
            shape_tail=(19, 5, 200),
            context="extracted v29 phase features",
        )
        if len(h) != len(phase) or len(h) != len(self.event_receipts):
            raise ValueError("extracted v29 feature/event counts differ")
        if not isinstance(self.event_receipts, tuple):
            raise TypeError("event_receipts must be a tuple")
        object.__setattr__(self, "h_features", h)
        object.__setattr__(self, "phase_features", phase)
        object.__setattr__(
            self,
            "event_receipts",
            tuple(deepcopy(receipt) for receipt in self.event_receipts),
        )


def processed_window_sha256(value: torch.Tensor) -> str:
    """Freeze the one canonical hash for a float32 standard-19 60s window."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("processed window must be a torch.Tensor")
    tensor = value.detach().cpu().contiguous()
    if tensor.dtype != torch.float32 or tuple(tensor.shape) != (19, 12_000):
        raise ValueError("processed window must be float32 [19,12000]")
    if not torch.isfinite(tensor).all():
        raise ValueError("processed window must be finite")
    return _tensor_sha256(tensor)


# Compatibility for the historical internal spelling used by focused tests.
_processed_window_sha256 = processed_window_sha256


def preprocessing_receipt_sha256(
    edf_receipt: object,
    signal_receipt: object,
) -> str:
    """Hash the exact preprocessing receipts shared by ranking and figures."""

    return _canonical_sha256(
        {
            "edf_receipt": _receipt_payload(edf_receipt, "EDF load receipt"),
            "signal_receipt": _receipt_payload(
                signal_receipt, "signal processing receipt"
            ),
            "processed_window_hash_policy": PROCESSED_WINDOW_HASH_POLICY,
        }
    )


def materialize_v29_pre_ranking_window_receipts(
    *,
    recording_path: str | Path,
    detection_manifest: object,
    event_id_assignment: object,
    event_loader: Callable[..., object] = load_standard19_edf_event,
) -> list[dict[str, Any]]:
    """Materialize signal-only receipts directly from one long EDF.

    This stage deliberately contains no report payload, waveform figure, SOZ
    ranking, annotation text, spreadsheet observation, or target value.
    """

    manifest = validate_long_term_seizure_detection_manifest(detection_manifest)
    assignment = resolve_v29_event_id_assignment(event_id_assignment, manifest)
    selected = {
        candidate["candidate_id"]: candidate
        for candidate in manifest["merge_candidates"]
        if candidate["decision_available"] is True
        and candidate["decision"] == "selected_for_event_analysis"
    }
    event_ids = {
        item["candidate_id"]: item["eeg_event_id"]
        for item in assignment["assignments"]
    }
    if set(event_ids) != set(selected):
        raise ValueError(
            "event ID assignment must exactly cover selected detector candidates"
        )
    source = Path(recording_path)
    if source.is_symlink():
        raise ValueError("long-recording EDF must not be a symbolic link")
    resolved_source = source.resolve(strict=True)
    if not resolved_source.is_file():
        raise ValueError("long-recording EDF must be a regular file")
    source_hash = _file_sha256(resolved_source)
    if source_hash != manifest["source_signal_sha256"]:
        raise ValueError("recording source SHA-256 does not match detector manifest")
    config = CausalEDFConfig(reference_policy="unlabeled_common_car19")
    windows: list[dict[str, Any]] = []
    ordered = sorted(
        selected.values(),
        key=lambda item: (item["anchor_offset_seconds"], event_ids[item["candidate_id"]]),
    )
    for candidate in ordered:
        candidate_id = candidate["candidate_id"]
        anchor = float(candidate["anchor_offset_seconds"])
        loaded = event_loader(
            resolved_source,
            anchor,
            config=config,
            use_edf_gap_annotations_for_signal_qc=False,
        )
        edf_receipt = getattr(loaded, "edf_receipt", None)
        signal_receipt = getattr(loaded, "signal_receipt", None)
        window = getattr(loaded, "window", None)
        data = getattr(window, "data", None)
        if not isinstance(data, torch.Tensor):
            raise TypeError("EDF loader must return a tensor-backed event window")
        signal = data.detach().cpu().to(torch.float32).contiguous()
        processed_hash = processed_window_sha256(signal)
        edf_payload = _receipt_payload(edf_receipt, "EDF load receipt")
        _receipt_payload(signal_receipt, "signal processing receipt")
        if edf_payload.get("edf_sha256") != source_hash:
            raise ValueError("EDF loader receipt does not match recording source")
        if not _same_time(edf_payload.get("requested_onset_sec"), anchor):
            raise ValueError("EDF loader receipt anchor does not match detector candidate")
        payload: dict[str, Any] = {
            "schema_version": PRE_RANKING_WINDOW_RECEIPT_SCHEMA_VERSION,
            "window_receipt_id": "CONTENT-ADDRESS-PENDING",
            "recording_id": manifest["recording_id"],
            "patient_pseudonym": manifest["patient_pseudonym"],
            "source_signal_sha256": source_hash,
            "recording_duration_seconds": manifest["recording_duration_seconds"],
            "candidate_id": candidate_id,
            "eeg_event_id": event_ids[candidate_id],
            "candidate_anchor_offset_seconds": anchor,
            "requested_window_seconds": [
                float(FIXED_EVENT_WINDOW_SECONDS[0]),
                float(FIXED_EVENT_WINDOW_SECONDS[1]),
            ],
            "window_start_offset_seconds": anchor + FIXED_EVENT_WINDOW_SECONDS[0],
            "window_stop_offset_seconds": anchor + FIXED_EVENT_WINDOW_SECONDS[1],
            "warmup_seconds_available": anchor,
            "post_anchor_seconds_available": float(
                manifest["recording_duration_seconds"]
            )
            - anchor,
            "boundary_policy": BOUNDARY_POLICY,
            "processed_window_hash_policy": PROCESSED_WINDOW_HASH_POLICY,
            "processed_window_sha256": processed_hash,
            "preprocessing_receipt_sha256": preprocessing_receipt_sha256(
                edf_receipt, signal_receipt
            ),
            "content_boundary": {
                "clinical_report_payload_included": False,
                "waveform_figure_included": False,
                "research_soz_ranking_included": False,
            },
        }
        payload["window_receipt_id"] = (
            f"WIN-V29-{canonical_payload_sha256(payload)[:20]}"
        )
        windows.append(validate_v29_pre_ranking_window_receipt(payload))
    return windows


def _selection_qc_details(error: EDFEventEligibilityError) -> dict[str, Any]:
    """Project only closed signal-QC fields; never persist exception prose."""

    if error.code != "signal_qc":
        return {
            "qc_stage": "not_available_for_eligibility_code",
            "failed_checks": [],
            "flatline_channels": [],
            "clipping_channels": [],
            "flatline_run_threshold_seconds": None,
            "clipping_run_threshold_seconds": None,
            "qc_tolerance_volts": None,
            "edf_gap_annotations_used": False,
        }
    raw = error.details if isinstance(error.details, Mapping) else {}
    if raw.get("edf_gap_annotations_used") is True:
        raise ValueError(
            "filtered analysis selection must not use EDF annotations for QC"
        )
    stage = raw.get("qc_stage")
    if stage not in QC_STAGES or stage == "not_available_for_eligibility_code":
        stage = "post_preprocessing_physical_contract"
    failed_raw = raw.get("failed_checks")
    failed = (
        [item for item in failed_raw if item in QC_FAILED_CHECKS]
        if isinstance(failed_raw, list)
        else []
    )
    if not failed:
        failed = ["downstream_physical_signal_contract"]

    def channels(key: str) -> list[str]:
        value = raw.get(key)
        if not isinstance(value, list):
            return []
        return [
            item
            for item in value
            if isinstance(item, str) and item in STANDARD_19
        ]

    def nonnegative(key: str) -> float | None:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None

    return {
        "qc_stage": stage,
        "failed_checks": list(dict.fromkeys(failed)),
        "flatline_channels": list(dict.fromkeys(channels("flatline_channels"))),
        "clipping_channels": list(dict.fromkeys(channels("clipping_channels"))),
        "flatline_run_threshold_seconds": nonnegative(
            "flatline_run_threshold_seconds"
        ),
        "clipping_run_threshold_seconds": nonnegative(
            "clipping_run_threshold_seconds"
        ),
        "qc_tolerance_volts": nonnegative("qc_tolerance_volts"),
        "edf_gap_annotations_used": False,
    }


def materialize_v29_filtered_analysis_selection(
    *,
    recording_path: str | Path,
    detection_manifest: object,
    event_id_assignment: object,
    event_loader: Callable[..., object] = load_standard19_edf_event,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Partition every selected detector candidate by signal eligibility.

    Expected :class:`EDFEventEligibilityError` instances become first-class
    rejection receipts.  Configuration, filesystem, reader and invariant
    failures still propagate as technical failures.
    """

    manifest = validate_long_term_seizure_detection_manifest(detection_manifest)
    assignment = resolve_v29_event_id_assignment(event_id_assignment, manifest)
    selected = {
        candidate["candidate_id"]: candidate
        for candidate in manifest["merge_candidates"]
        if candidate["decision_available"] is True
        and candidate["decision"] == "selected_for_event_analysis"
    }
    event_ids = {
        item["candidate_id"]: item["eeg_event_id"]
        for item in assignment["assignments"]
    }
    if set(event_ids) != set(selected):
        raise ValueError(
            "event ID assignment must exactly cover selected detector candidates"
        )
    source = Path(recording_path)
    if source.is_symlink():
        raise ValueError("long-recording EDF must not be a symbolic link")
    resolved_source = source.resolve(strict=True)
    if not resolved_source.is_file():
        raise ValueError("long-recording EDF must be a regular file")
    source_hash = _file_sha256(resolved_source)
    if source_hash != manifest["source_signal_sha256"]:
        raise ValueError("recording source SHA-256 does not match detector manifest")
    config = CausalEDFConfig(reference_policy="unlabeled_common_car19")
    ordered = sorted(
        selected.values(),
        key=lambda item: (
            item["anchor_offset_seconds"],
            event_ids[item["candidate_id"]],
        ),
    )
    windows: list[dict[str, Any]] = []
    selection_events: list[dict[str, Any]] = []
    for candidate in ordered:
        candidate_id = str(candidate["candidate_id"])
        event_id = str(event_ids[candidate_id])
        anchor = float(candidate["anchor_offset_seconds"])
        try:
            loaded = event_loader(
                resolved_source,
                anchor,
                config=config,
                use_edf_gap_annotations_for_signal_qc=False,
            )
        except EDFEventEligibilityError as error:
            rejection: dict[str, Any] = {
                "schema_version": ANALYSIS_REJECTION_SCHEMA_VERSION,
                "rejection_receipt_id": "CONTENT-ADDRESS-PENDING",
                "candidate_id": candidate_id,
                "eeg_event_id": event_id,
                "candidate_anchor_offset_seconds": anchor,
                "eligibility_code": error.code,
                "reason_code": "edf_event_eligibility_" + error.code,
                "signal_qc_details": _selection_qc_details(error),
                "scope_receipt": {
                    "eeg_signal_or_physical_metadata_used": True,
                    "edf_annotations_used": False,
                    "excel_used": False,
                    "clinical_context_used": False,
                    "labels_or_ground_truth_used": False,
                },
                "claim_boundary": {
                    "rejection_is_not_no_seizure": True,
                    "candidate_is_confirmed_seizure": False,
                    "candidate_is_confirmed_nonseizure": False,
                    "soz_conclusion_generated": False,
                },
            }
            rejection["rejection_receipt_id"] = (
                ANALYSIS_REJECTION_ID_PREFIX
                + canonical_payload_sha256(rejection)[:20]
            )
            rejection = validate_analysis_rejection_receipt(rejection)
            selection_events.append(
                {
                    "candidate_id": candidate_id,
                    "eeg_event_id": event_id,
                    "candidate_anchor_offset_seconds": anchor,
                    "analysis_disposition": "rejected_signal_eligibility",
                    "pre_ranking_window_receipt_sha256": None,
                    "processed_window_sha256": None,
                    "preprocessing_receipt_sha256": None,
                    "rejection_receipt": rejection,
                }
            )
            continue

        edf_receipt = getattr(loaded, "edf_receipt", None)
        signal_receipt = getattr(loaded, "signal_receipt", None)
        window = getattr(loaded, "window", None)
        data = getattr(window, "data", None)
        if not isinstance(data, torch.Tensor):
            raise TypeError("EDF loader must return a tensor-backed event window")
        signal = data.detach().cpu().to(torch.float32).contiguous()
        processed_hash = processed_window_sha256(signal)
        edf_payload = _receipt_payload(edf_receipt, "EDF load receipt")
        _receipt_payload(signal_receipt, "signal processing receipt")
        if edf_payload.get("edf_sha256") != source_hash:
            raise ValueError("EDF loader receipt does not match recording source")
        if not _same_time(edf_payload.get("requested_onset_sec"), anchor):
            raise ValueError(
                "EDF loader receipt anchor does not match detector candidate"
            )
        preprocessing_hash = preprocessing_receipt_sha256(
            edf_receipt, signal_receipt
        )
        payload: dict[str, Any] = {
            "schema_version": PRE_RANKING_WINDOW_RECEIPT_SCHEMA_VERSION,
            "window_receipt_id": "CONTENT-ADDRESS-PENDING",
            "recording_id": manifest["recording_id"],
            "patient_pseudonym": manifest["patient_pseudonym"],
            "source_signal_sha256": source_hash,
            "recording_duration_seconds": manifest["recording_duration_seconds"],
            "candidate_id": candidate_id,
            "eeg_event_id": event_id,
            "candidate_anchor_offset_seconds": anchor,
            "requested_window_seconds": [
                float(FIXED_EVENT_WINDOW_SECONDS[0]),
                float(FIXED_EVENT_WINDOW_SECONDS[1]),
            ],
            "window_start_offset_seconds": anchor
            + FIXED_EVENT_WINDOW_SECONDS[0],
            "window_stop_offset_seconds": anchor
            + FIXED_EVENT_WINDOW_SECONDS[1],
            "warmup_seconds_available": anchor,
            "post_anchor_seconds_available": float(
                manifest["recording_duration_seconds"]
            )
            - anchor,
            "boundary_policy": BOUNDARY_POLICY,
            "processed_window_hash_policy": PROCESSED_WINDOW_HASH_POLICY,
            "processed_window_sha256": processed_hash,
            "preprocessing_receipt_sha256": preprocessing_hash,
            "content_boundary": {
                "clinical_report_payload_included": False,
                "waveform_figure_included": False,
                "research_soz_ranking_included": False,
            },
        }
        payload["window_receipt_id"] = (
            f"WIN-V29-{canonical_payload_sha256(payload)[:20]}"
        )
        window_receipt = validate_v29_pre_ranking_window_receipt(payload)
        windows.append(window_receipt)
        selection_events.append(
            {
                "candidate_id": candidate_id,
                "eeg_event_id": event_id,
                "candidate_anchor_offset_seconds": anchor,
                "analysis_disposition": "analyzable",
                "pre_ranking_window_receipt_sha256": canonical_payload_sha256(
                    window_receipt
                ),
                "processed_window_sha256": processed_hash,
                "preprocessing_receipt_sha256": preprocessing_hash,
                "rejection_receipt": None,
            }
        )

    analyzable_count = len(windows)
    selection: dict[str, Any] = {
        "schema_version": ANALYSIS_SELECTION_SCHEMA_VERSION,
        "selection_id": "CONTENT-ADDRESS-PENDING",
        "recording_id": manifest["recording_id"],
        "patient_pseudonym": manifest["patient_pseudonym"],
        "source_signal_sha256": manifest["source_signal_sha256"],
        "recording_duration_seconds": manifest["recording_duration_seconds"],
        "detection_manifest_sha256": canonical_payload_sha256(manifest),
        "event_id_assignment_sha256": canonical_payload_sha256(assignment),
        "candidate_semantics": CANDIDATE_SEMANTICS,
        "detector_selected_count": len(selection_events),
        "analyzable_count": analyzable_count,
        "rejected_count": len(selection_events) - analyzable_count,
        "events": selection_events,
        "scope_receipt": {
            "physical_edf_signal_or_metadata_used": True,
            "edf_annotations_used": False,
            "excel_used": False,
            "clinical_context_used": False,
            "labels_or_ground_truth_used": False,
            "detector_decisions_modified": False,
            "rejected_candidates_silently_dropped": False,
        },
    }
    selection["selection_id"] = (
        ANALYSIS_SELECTION_ID_PREFIX
        + canonical_payload_sha256(selection)[:20]
    )
    selection = bind_long_term_eeg_analysis_selection(
        selection,
        manifest,
        event_id_assignment=assignment,
    )
    return selection, windows


def extract_v29_candidate_features(
    *,
    recording_path: str | Path,
    candidate_batch: Mapping[str, Any],
    pre_ranking_window_receipts: Sequence[object],
    encoder: OfficialLaBraMFrozenPrefixEncoder | torch.nn.Module,
    device: torch.device,
    event_loader: Callable[..., object] = load_standard19_edf_event,
) -> ExtractedV29CandidateFeatures:
    """Replay each signal-only event window through the frozen block-9 prefix.

    ``event_loader`` and ``encoder`` are injectable for deterministic unit
    tests.  Production callers use the audited physical-EDF loader and
    :class:`OfficialLaBraMFrozenPrefixEncoder`.
    """

    batch = validate_v29_candidate_batch(candidate_batch)
    events = batch["events"]
    if not events:
        raise ValueError("v29 feature extraction requires at least one candidate")
    if isinstance(pre_ranking_window_receipts, (str, bytes)) or not isinstance(
        pre_ranking_window_receipts, Sequence
    ):
        raise TypeError("pre_ranking_window_receipts must be an array")
    canonical_windows = [
        validate_v29_pre_ranking_window_receipt(item)
        for item in pre_ranking_window_receipts
    ]
    by_candidate = {window["candidate_id"]: window for window in canonical_windows}
    if len(by_candidate) != len(canonical_windows):
        raise ValueError("feature extraction windows repeat a detector candidate")
    if set(by_candidate) != {event["candidate_id"] for event in events}:
        raise ValueError("feature extraction windows differ from candidate batch")
    for event in events:
        window = by_candidate[event["candidate_id"]]
        if (
            canonical_payload_sha256(window)
            != event["pre_ranking_window_receipt_sha256"]
        ):
            raise ValueError("feature extraction window receipt differs from candidate batch")
    if not isinstance(device, torch.device):
        raise TypeError("device must be torch.device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not isinstance(encoder, torch.nn.Module):
        raise TypeError("encoder must be a torch module")
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("v29 foundation prefix encoder must remain frozen")

    source = Path(recording_path)
    if source.is_symlink():
        raise ValueError("long-recording EDF must not be a symbolic link")
    resolved_source = source.resolve(strict=True)
    source_hash = _file_sha256(resolved_source)
    if source_hash != batch["source_signal_sha256"]:
        raise ValueError("recording source SHA-256 does not match candidate batch")
    config = CausalEDFConfig(reference_policy="unlabeled_common_car19")
    preprocessing_policy_sha256 = _canonical_sha256(
        {
            "loader": "standard19_causal_edf_event_v2",
            "config": asdict(config),
            "edf_gap_annotations_used_for_signal_qc": False,
            "split_calls": "19x12000_to_15x19x4x200",
        }
    )
    h_rows: list[torch.Tensor] = []
    phase_rows: list[torch.Tensor] = []
    receipts: list[dict[str, Any]] = []
    encoder.eval()
    for event_index, event in enumerate(events):
        window_receipt = by_candidate[event["candidate_id"]]
        anchor = float(event["candidate_anchor_offset_seconds"])
        loaded = event_loader(
            resolved_source,
            anchor,
            config=config,
            use_edf_gap_annotations_for_signal_qc=False,
        )
        edf_receipt = getattr(loaded, "edf_receipt", None)
        signal_receipt = getattr(loaded, "signal_receipt", None)
        window = getattr(loaded, "window", None)
        data = getattr(window, "data", None)
        if not isinstance(data, torch.Tensor):
            raise TypeError("EDF loader must return a tensor-backed event window")
        signal = data.detach().cpu().to(torch.float32).contiguous()
        if tuple(signal.shape) != (19, 12_000) or not torch.isfinite(signal).all():
            raise ValueError("replayed v29 event must be finite float32 [19,12000]")
        edf_payload = _receipt_payload(edf_receipt, "EDF load receipt")
        signal_payload = _receipt_payload(signal_receipt, "signal receipt")
        if edf_payload.get("edf_sha256") != source_hash:
            raise ValueError("EDF loader receipt does not match recording source")
        if abs(float(edf_payload.get("requested_onset_sec")) - anchor) > _TIME_TOLERANCE:
            raise ValueError("EDF loader receipt anchor does not match candidate")
        processed_hash = processed_window_sha256(signal)
        if processed_hash != window_receipt["processed_window_sha256"]:
            raise ValueError("replayed processed window does not match window receipt")
        preprocessing_hash = preprocessing_receipt_sha256(
            edf_receipt, signal_receipt
        )
        if preprocessing_hash != window_receipt["preprocessing_receipt_sha256"]:
            raise ValueError("replayed preprocessing receipt differs from window receipt")
        raw_names = edf_payload.get("raw_channel_names")
        semantic_channels = edf_payload.get("semantic_channels")
        if not isinstance(raw_names, (list, tuple)) or not isinstance(
            semantic_channels, (list, tuple)
        ):
            raise ValueError("EDF receipt lacks record-position binding fields")
        binding = bind_labram_record_positions(
            tuple(raw_names), semantic_channels=tuple(semantic_channels)
        )
        calls = _split_calls(signal).to(device)
        with torch.inference_mode():
            prefix = encoder.forward_with_record_binding(calls, binding)
        prefix = prefix.detach().cpu().float().contiguous()
        if tuple(prefix.shape) != (15, 77, 200) or not torch.isfinite(prefix).all():
            raise RuntimeError("frozen LaBraM block-9 prefix contract failed")
        event_prefix = prefix.unsqueeze(0)
        h = extract_block9_phase_contrasts(event_prefix)[0].float().contiguous()
        phase = extract_rank1_phase_features(event_prefix)[0].float().contiguous()
        if tuple(h.shape) != (19, 600) or tuple(phase.shape) != (19, 5, 200):
            raise RuntimeError("v29 compact feature shape drifted")
        h_rows.append(h)
        phase_rows.append(phase)
        receipts.append(
            {
                "event_index": event_index,
                "candidate_id": event["candidate_id"],
                "eeg_event_id": event["eeg_event_id"],
                "candidate_anchor_offset_seconds": anchor,
                "source_signal_sha256": source_hash,
                "pre_ranking_window_receipt_sha256": event[
                    "pre_ranking_window_receipt_sha256"
                ],
                "processed_window_sha256": processed_hash,
                "window_preprocessing_receipt_sha256": preprocessing_hash,
                "v29_preprocessing_policy_sha256": preprocessing_policy_sha256,
                "v29_edf_receipt_sha256": _canonical_sha256(edf_payload),
                "v29_signal_receipt_sha256": _canonical_sha256(signal_payload),
                "block9_prefix_sha256": _tensor_sha256(prefix),
                "h_features_sha256": _tensor_sha256(h),
                "phase_features_sha256": _tensor_sha256(phase),
                "target_values_loaded": False,
                "annotation_values_used_for_features": False,
                "gap_annotations_used_for_signal_qc_only": False,
            }
        )
    return ExtractedV29CandidateFeatures(
        h_features=torch.stack(h_rows).float().contiguous(),
        phase_features=torch.stack(phase_rows).float().contiguous(),
        event_receipts=tuple(receipts),
    )


@dataclass(frozen=True)
class FrozenV29Runtime:
    encoder: OfficialLaBraMFrozenPrefixEncoder
    direct_states: Mapping[str, torch.Tensor]
    h_states: Mapping[str, torch.Tensor]
    model_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.encoder, OfficialLaBraMFrozenPrefixEncoder):
            raise TypeError("production v29 runtime requires the official prefix encoder")
        if any(parameter.requires_grad for parameter in self.encoder.parameters()):
            raise ValueError("production v29 encoder must remain frozen")
        if not isinstance(self.direct_states, Mapping) or not isinstance(
            self.h_states, Mapping
        ):
            raise TypeError("production v29 states must be mappings")
        _validate_state_masks(self.direct_states, self.h_states)
        if not isinstance(self.model_receipt, Mapping):
            raise TypeError("production v29 model receipt must be a mapping")


def _validate_public_freeze_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != (
        "soz_labram_portable_equal_ensemble_public_oof_v29"
    ):
        raise ValueError("v29 public freeze manifest schema mismatch")
    ensemble = payload.get("ensemble")
    if not isinstance(ensemble, Mapping) or dict(ensemble) != {
        "h_only_weight": 0.5,
        "rank1_direct_token_weight": 0.5,
        "combination_space": "candidate_masked_probability",
        "weight_trainable": False,
        "confidence_gate": False,
        "fine_feature_family_used": False,
        "fold_count": 5,
    }:
        raise ValueError("v29 public frozen ensemble contract drifted")
    access = payload.get("access_receipt")
    if not isinstance(access, Mapping) or any(
        access.get(key) is not False
        for key in (
            "private_eeg_loaded",
            "private_target_values_loaded",
            "private_prediction_or_metric_loaded",
            "training_performed",
            "ensemble_parameter_fitted",
        )
    ):
        raise ValueError("v29 public freeze was not target/private independent")
    if payload.get("go") is not True or payload.get("decision") != (
        "AUTHORIZE_ONE_TARGET_BLIND_PRIVATE_RUN"
    ):
        raise ValueError("historical v29 public freeze did not pass its original gate")


def load_frozen_v29_runtime(
    *,
    modeling_path: str | Path = DEFAULT_MODELING,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    direct_states_path: str | Path = DEFAULT_DIRECT_STATES,
    h_states_path: str | Path = DEFAULT_H_STATES,
    public_freeze_manifest_path: str | Path = DEFAULT_PUBLIC_FREEZE_MANIFEST,
    device: torch.device,
) -> FrozenV29Runtime:
    """Load only the exact local frozen v29 resources and audited foundation."""

    if not isinstance(device, torch.device):
        raise TypeError("device must be torch.device")
    modeling = _verified_file(
        modeling_path, AUDITED_LABRAM_MODELING_SHA256, "LaBraM modeling source"
    )
    checkpoint = _verified_file(
        checkpoint_path, AUDITED_LABRAM_BASE_SHA256, "LaBraM checkpoint"
    )
    direct_path = _verified_file(
        direct_states_path, EXPECTED_DIRECT_STATES_SHA256, "v29 direct states"
    )
    h_path = _verified_file(h_states_path, EXPECTED_H_STATES_SHA256, "v29 H states")
    public_path = _verified_file(
        public_freeze_manifest_path,
        EXPECTED_PUBLIC_FREEZE_MANIFEST_SHA256,
        "v29 public freeze manifest",
    )
    public_manifest = _strict_json(public_path)
    _validate_public_freeze_manifest(public_manifest)
    direct_states = load_file(str(direct_path), device="cpu")
    h_states = load_file(str(h_path), device="cpu")
    _validate_state_masks(direct_states, h_states)
    # A one-event zero carrier validates all required state keys/shapes through
    # the unchanged inference functions without consulting any target.
    infer_v29_probabilities(
        torch.zeros((1, 19, 600), dtype=torch.float32),
        torch.zeros((1, 19, 5, 200), dtype=torch.float32),
        direct_states=direct_states,
        h_states=h_states,
    )
    encoder = OfficialLaBraMFrozenPrefixEncoder(
        modeling_path=modeling,
        checkpoint_path=checkpoint,
        expected_sha256=AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
    ).to(device).eval()
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("frozen v29 foundation exposes trainable parameters")
    module_hash = _file_sha256(Path(__file__).resolve(strict=True))
    receipt: dict[str, Any] = {
        "method_id": METHOD_ID,
        "inference_policy": INFERENCE_POLICY,
        "foundation_checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
        "foundation_modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
        "direct_states_sha256": EXPECTED_DIRECT_STATES_SHA256,
        "h_states_sha256": EXPECTED_H_STATES_SHA256,
        "public_freeze_manifest_sha256": EXPECTED_PUBLIC_FREEZE_MANIFEST_SHA256,
        "inference_module_sha256": module_hash,
        "fold_count": N_FOLDS,
        "h_only_weight": 0.5,
        "rank1_direct_token_weight": 0.5,
        "candidate_channels": [
            channel
            for channel, available in zip(STANDARD_19, V11_CANDIDATE_MASK.tolist())
            if available
        ],
        "excluded_channels": ["PZ"],
        "historical_one_run_authorization_reused_as_new_validation": False,
        "post_open_research_reuse": True,
        "training_or_calibration_performed": False,
    }
    receipt["model_sha256"] = _canonical_sha256(receipt)
    return FrozenV29Runtime(
        encoder=encoder,
        direct_states=direct_states,
        h_states=h_states,
        model_receipt=receipt,
    )


def _ranking_from_probability(probability: torch.Tensor) -> list[dict[str, Any]]:
    if tuple(probability.shape) != (19,) or not probability.is_floating_point():
        raise ValueError("one v29 ranking row must be floating point [19]")
    row = probability.detach().cpu().float().contiguous()
    if not torch.isfinite(row).all() or not torch.allclose(
        row.sum(), torch.tensor(1.0), atol=1e-6, rtol=0
    ):
        raise ValueError("one v29 ranking row must be finite and sum to one")
    if float(row[STANDARD_19.index("PZ")]) != 0.0:
        raise ValueError("PZ must remain excluded from v29 ranking")
    candidate_indices = [
        index for index, available in enumerate(V11_CANDIDATE_MASK.tolist()) if available
    ]
    candidate_indices.sort(key=lambda index: (-float(row[index]), index))
    return [
        {
            "rank": rank,
            "electrode": STANDARD_19[channel_index],
            "score": float(row[channel_index]),
        }
        for rank, channel_index in enumerate(candidate_indices, start=1)
    ]


def assemble_v29_candidate_rankings(
    *,
    candidate_batch: Mapping[str, Any],
    features: ExtractedV29CandidateFeatures,
    probabilities: Mapping[str, torch.Tensor],
    model_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Build portable research receipts and auditable probability tensors."""

    batch = validate_v29_candidate_batch(candidate_batch)
    events = batch["events"]
    if not events:
        raise ValueError("v29 ranking assembly requires at least one event")
    if not isinstance(features, ExtractedV29CandidateFeatures):
        raise TypeError("features must be ExtractedV29CandidateFeatures")
    if len(features.event_receipts) != len(events):
        raise ValueError("feature receipts do not match candidate count")
    if not isinstance(probabilities, Mapping):
        raise TypeError("probabilities must be a mapping")
    required_tensors = (
        "portable_equal_probability",
        "portable_equal_fold_probability",
        "rank1_direct_fold_probability",
        "h_only_fold_probability",
        "candidate_mask",
    )
    if set(probabilities) != set(required_tensors):
        raise ValueError("v29 probability tensor keys drifted")
    tensor_payload = {
        name: value.detach().cpu().contiguous()
        for name, value in probabilities.items()
    }
    event_count = len(events)
    if tuple(tensor_payload["portable_equal_probability"].shape) != (
        event_count,
        19,
    ):
        raise ValueError("v29 probability event count does not match candidates")
    for name in (
        "portable_equal_fold_probability",
        "rank1_direct_fold_probability",
        "h_only_fold_probability",
    ):
        if tuple(tensor_payload[name].shape) != (event_count, N_FOLDS, 19):
            raise ValueError(f"v29 {name} shape does not match candidates")
    if not torch.equal(
        tensor_payload["candidate_mask"].bool(), V11_CANDIDATE_MASK
    ):
        raise ValueError("v29 output candidate mask drifted")
    if not isinstance(model_receipt, Mapping):
        raise TypeError("model_receipt must be a mapping")
    model = deepcopy(dict(model_receipt))
    model_sha = _sha256(model.get("model_sha256"), "model_receipt.model_sha256")
    if model.get("method_id") != METHOD_ID or model.get("fold_count") != N_FOLDS:
        raise ValueError("model receipt does not describe frozen v29")
    if model.get("training_or_calibration_performed") is not False:
        raise ValueError("v29 candidate inference must not train or calibrate")

    output_events: list[dict[str, Any]] = []
    for event_index, (event, feature_receipt) in enumerate(
        zip(events, features.event_receipts)
    ):
        if (
            feature_receipt.get("event_index") != event_index
            or feature_receipt.get("candidate_id") != event["candidate_id"]
            or feature_receipt.get("eeg_event_id") != event["eeg_event_id"]
            or feature_receipt.get("source_signal_sha256")
            != batch["source_signal_sha256"]
            or feature_receipt.get("pre_ranking_window_receipt_sha256")
            != event["pre_ranking_window_receipt_sha256"]
            or feature_receipt.get("processed_window_sha256")
            != event["processed_window_sha256"]
            or feature_receipt.get("window_preprocessing_receipt_sha256")
            != event["preprocessing_receipt_sha256"]
            or feature_receipt.get("target_values_loaded") is not False
            or feature_receipt.get("annotation_values_used_for_features") is not False
        ):
            raise ValueError("feature receipt order/identity differs from candidate batch")
        row = tensor_payload["portable_equal_probability"][event_index]
        row_hash = _tensor_sha256(row)
        ranking = _ranking_from_probability(row)
        receipt_identity = {
            "candidate_id": event["candidate_id"],
            "eeg_event_id": event["eeg_event_id"],
            "processed_window_sha256": event["processed_window_sha256"],
            "probability_sha256": row_hash,
            "model_sha256": model_sha,
        }
        research_receipt = {
            "receipt_id": f"SOZRANK-V29-{_canonical_sha256(receipt_identity)[:20]}",
            "method_id": METHOD_ID,
            "model_sha256": model_sha,
            "input_processed_window_sha256": event["processed_window_sha256"],
            "interpretation_status": SOZ_INTERPRETATION_STATUS,
            "ranked_electrodes": ranking,
            "used_in_clinical_facts": False,
            "used_in_impression": False,
            "sent_to_llm": False,
        }
        output_events.append(
            {
                "event_index": event_index,
                "candidate_id": event["candidate_id"],
                "eeg_event_id": event["eeg_event_id"],
                "candidate_anchor_offset_seconds": event[
                    "candidate_anchor_offset_seconds"
                ],
                "pre_ranking_window_receipt_sha256": event[
                    "pre_ranking_window_receipt_sha256"
                ],
                "pre_ranking_window_receipt": deepcopy(
                    event["pre_ranking_window_receipt"]
                ),
                "processed_window_sha256": event["processed_window_sha256"],
                "feature_receipt": deepcopy(feature_receipt),
                "portable_equal_probability_sha256": row_hash,
                "portable_equal_fold_probability_sha256": _tensor_sha256(
                    tensor_payload["portable_equal_fold_probability"][event_index]
                ),
                "rank1_direct_fold_probability_sha256": _tensor_sha256(
                    tensor_payload["rank1_direct_fold_probability"][event_index]
                ),
                "h_only_fold_probability_sha256": _tensor_sha256(
                    tensor_payload["h_only_fold_probability"][event_index]
                ),
                "research_soz_ranking_receipt": research_receipt,
            }
        )

    tensor_receipt = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
        for name, value in tensor_payload.items()
    }
    filtered = batch["schema_version"] == FILTERED_INPUT_SCHEMA_VERSION
    manifest: dict[str, Any] = {
        "schema_version": (
            FILTERED_OUTPUT_SCHEMA_VERSION if filtered else OUTPUT_SCHEMA_VERSION
        ),
        "status": OUTPUT_STATUS,
        "recording_id": batch["recording_id"],
        "patient_pseudonym": batch["patient_pseudonym"],
        "source_signal_sha256": batch["source_signal_sha256"],
        "recording_duration_seconds": batch["recording_duration_seconds"],
        "candidate_batch_sha256": canonical_payload_sha256(batch),
        "candidate_semantics": CANDIDATE_SEMANTICS,
        "processed_window_hash_policy": PROCESSED_WINDOW_HASH_POLICY,
        "method_id": METHOD_ID,
        "model_receipt": model,
        "event_count": event_count,
        "events": output_events,
        "tensor_file": TENSOR_FILE,
        "tensor_receipt": tensor_receipt,
        "access_receipt": {
            "long_recording_eeg_loaded": True,
            "detector_manifest_loaded": True,
            "pre_ranking_window_receipts_loaded": True,
            "final_segment_receipts_loaded": False,
            "legacy_88_event_roster_loaded": False,
            "private_target_ledger_path_argument_exposed": False,
            "private_target_values_loaded": False,
            "edf_annotation_text_used_for_ranking": False,
            "edf_gap_annotations_used_for_signal_qc_only": False,
            "excel_physician_observations_loaded": False,
            "training_calibration_or_model_selection_performed": False,
            "llm_used_for_prediction_or_ranking": False,
        },
        "stage_scope": {
            "research_rankings_generated": True,
            "final_segment_receipts_consumed": False,
            "segment_processed_windows_generated_by_ranking_stage": False,
            "segment_preprocessing_receipts_generated_by_ranking_stage": False,
            "waveform_tensors_or_figures_generated": False,
            "waveform_figure_bindings_generated": False,
            "target_blind_sustained_change_descriptors_generated": False,
            "clinical_report_facts_generated": False,
            "complete_clinical_segments_generated": False,
            "ranking_can_finalize_unranked_segment_draft": True,
        },
        "claim_boundary": {
            "research_scalp_electrode_ranking_only": True,
            "candidate_is_confirmed_seizure": False,
            "output_is_cortical_soz": False,
            "output_is_epileptogenic_zone": False,
            "output_is_treatment_target": False,
            "output_is_clinical_conclusion": False,
            "current_sota_claim": False,
            "fresh_external_validation": False,
            "scores_are_calibrated_error_probabilities": False,
            "physician_review_required": True,
        },
    }
    if filtered:
        manifest["analysis_selection_sha256"] = batch[
            "analysis_selection_sha256"
        ]
    _canonical_json_bytes(manifest)
    return manifest, tensor_payload


def validate_v29_candidate_ranking_manifest(payload: object) -> dict[str, Any]:
    """Strictly validate the portable manifest (tensor bytes are separate)."""

    if type(payload) is not dict:
        raise TypeError("v29 candidate ranking manifest must be an object")
    schema_raw = payload.get("schema_version")
    filtered = schema_raw == FILTERED_OUTPUT_SCHEMA_VERSION
    if schema_raw not in (OUTPUT_SCHEMA_VERSION, FILTERED_OUTPUT_SCHEMA_VERSION):
        raise ValueError("v29 candidate ranking manifest version/status mismatch")
    required = [
        "schema_version",
        "status",
        "recording_id",
        "patient_pseudonym",
        "source_signal_sha256",
        "recording_duration_seconds",
        "candidate_batch_sha256",
        "candidate_semantics",
        "processed_window_hash_policy",
        "method_id",
        "model_receipt",
        "event_count",
        "events",
        "tensor_file",
        "tensor_receipt",
        "access_receipt",
        "stage_scope",
        "claim_boundary",
    ]
    if filtered:
        required.append("analysis_selection_sha256")
    data = _strict_object(
        payload,
        required=tuple(required),
        context="v29 candidate ranking manifest",
    )
    if data["status"] != OUTPUT_STATUS:
        raise ValueError("v29 candidate ranking manifest version/status mismatch")
    _identifier(data["recording_id"], "manifest.recording_id")
    _identifier(data["patient_pseudonym"], "manifest.patient_pseudonym")
    _sha256(data["source_signal_sha256"], "manifest.source_signal_sha256")
    duration = _finite(
        data["recording_duration_seconds"], "manifest.recording_duration_seconds"
    )
    if duration <= 0:
        raise ValueError("manifest recording duration must be positive")
    _sha256(data["candidate_batch_sha256"], "manifest.candidate_batch_sha256")
    if filtered:
        _sha256(
            data["analysis_selection_sha256"],
            "manifest.analysis_selection_sha256",
        )
    if data["candidate_semantics"] != CANDIDATE_SEMANTICS:
        raise ValueError("manifest promotes detector candidates to confirmed seizures")
    if data["processed_window_hash_policy"] != PROCESSED_WINDOW_HASH_POLICY:
        raise ValueError("manifest processed-window hash policy drifted")
    if data["method_id"] != METHOD_ID or data["tensor_file"] != TENSOR_FILE:
        raise ValueError("manifest v29 method/tensor filename mismatch")
    event_count = data["event_count"]
    if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count < 1:
        raise ValueError("manifest event_count must be a positive integer")
    if not isinstance(data["events"], list) or len(data["events"]) != event_count:
        raise ValueError("manifest events do not match event_count")
    event_ids: set[str] = set()
    candidate_ids: set[str] = set()
    for index, raw in enumerate(data["events"]):
        event = _strict_object(
            raw,
            required=(
                "event_index",
                "candidate_id",
                "eeg_event_id",
                "candidate_anchor_offset_seconds",
                "pre_ranking_window_receipt_sha256",
                "pre_ranking_window_receipt",
                "processed_window_sha256",
                "feature_receipt",
                "portable_equal_probability_sha256",
                "portable_equal_fold_probability_sha256",
                "rank1_direct_fold_probability_sha256",
                "h_only_fold_probability_sha256",
                "research_soz_ranking_receipt",
            ),
            context=f"manifest.events[{index}]",
        )
        if event["event_index"] != index:
            raise ValueError("manifest event indices must be contiguous")
        candidate_id = _identifier(event["candidate_id"], "event.candidate_id")
        event_id = _identifier(event["eeg_event_id"], "event.eeg_event_id")
        if candidate_id in candidate_ids or event_id in event_ids:
            raise ValueError("manifest repeats candidate or EEG event identity")
        candidate_ids.add(candidate_id)
        event_ids.add(event_id)
        anchor = _finite(
            event["candidate_anchor_offset_seconds"], "event candidate anchor"
        )
        if anchor < 0 or anchor > duration:
            raise ValueError("manifest event anchor is outside the recording")
        for key in (
            "pre_ranking_window_receipt_sha256",
            "processed_window_sha256",
            "portable_equal_probability_sha256",
            "portable_equal_fold_probability_sha256",
            "rank1_direct_fold_probability_sha256",
            "h_only_fold_probability_sha256",
        ):
            _sha256(event[key], f"event.{key}")
        window = validate_v29_pre_ranking_window_receipt(
            event["pre_ranking_window_receipt"]
        )
        if (
            canonical_payload_sha256(window)
            != event["pre_ranking_window_receipt_sha256"]
            or window["recording_id"] != data["recording_id"]
            or window["patient_pseudonym"] != data["patient_pseudonym"]
            or window["source_signal_sha256"] != data["source_signal_sha256"]
            or not _same_time(
                window["recording_duration_seconds"],
                data["recording_duration_seconds"],
            )
            or window["candidate_id"] != candidate_id
            or window["eeg_event_id"] != event_id
            or not _same_time(
                window["candidate_anchor_offset_seconds"], anchor
            )
            or window["processed_window_sha256"]
            != event["processed_window_sha256"]
        ):
            raise ValueError("ranking manifest window receipt binding failed")
        receipt = event["research_soz_ranking_receipt"]
        if not isinstance(receipt, Mapping):
            raise TypeError("research ranking receipt must be an object")
        if (
            receipt.get("method_id") != METHOD_ID
            or receipt.get("interpretation_status") != SOZ_INTERPRETATION_STATUS
            or receipt.get("input_processed_window_sha256")
            != event["processed_window_sha256"]
            or receipt.get("used_in_clinical_facts") is not False
            or receipt.get("used_in_impression") is not False
            or receipt.get("sent_to_llm") is not False
        ):
            raise ValueError("research ranking receipt crossed its claim boundary")
        ranking = receipt.get("ranked_electrodes")
        if not isinstance(ranking, list) or len(ranking) != int(
            V11_CANDIDATE_MASK.sum()
        ):
            raise ValueError("research ranking must enumerate the 18 v29 candidates")
        previous = math.inf
        electrodes: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if not isinstance(item, Mapping) or item.get("rank") != rank:
                raise ValueError("research ranking ranks must be contiguous")
            electrode = item.get("electrode")
            score = _finite(item.get("score"), "research ranking score")
            if electrode not in STANDARD_19 or electrode == "PZ" or electrode in electrodes:
                raise ValueError("research ranking electrode set is invalid")
            if score < 0 or score > previous + 1e-12:
                raise ValueError("research ranking scores must be non-increasing")
            electrodes.add(str(electrode))
            previous = score
    access = data["access_receipt"]
    claims = data["claim_boundary"]
    if not isinstance(access, Mapping) or any(
        access.get(key) is not False
        for key in (
            "legacy_88_event_roster_loaded",
            "final_segment_receipts_loaded",
            "private_target_ledger_path_argument_exposed",
            "private_target_values_loaded",
            "edf_annotation_text_used_for_ranking",
            "excel_physician_observations_loaded",
            "training_calibration_or_model_selection_performed",
            "llm_used_for_prediction_or_ranking",
        )
    ):
        raise ValueError("v29 manifest access boundary failed")
    if (
        access.get("long_recording_eeg_loaded") is not True
        or access.get("detector_manifest_loaded") is not True
        or access.get("pre_ranking_window_receipts_loaded") is not True
    ):
        raise ValueError("v29 manifest required input access receipt is missing")
    stage = _strict_object(
        data["stage_scope"],
        required=(
            "research_rankings_generated",
            "final_segment_receipts_consumed",
            "segment_processed_windows_generated_by_ranking_stage",
            "segment_preprocessing_receipts_generated_by_ranking_stage",
            "waveform_tensors_or_figures_generated",
            "waveform_figure_bindings_generated",
            "target_blind_sustained_change_descriptors_generated",
            "clinical_report_facts_generated",
            "complete_clinical_segments_generated",
            "ranking_can_finalize_unranked_segment_draft",
        ),
        context="manifest.stage_scope",
    )
    true_stage_fields = (
        "research_rankings_generated",
        "ranking_can_finalize_unranked_segment_draft",
    )
    if any(stage[key] is not True for key in true_stage_fields) or any(
        stage[key] is not False
        for key in set(stage).difference(true_stage_fields)
    ):
        raise ValueError("v29 manifest stage scope overstates ranking output")
    if not isinstance(claims, Mapping) or any(
        claims.get(key) is not False
        for key in (
            "candidate_is_confirmed_seizure",
            "output_is_cortical_soz",
            "output_is_epileptogenic_zone",
            "output_is_treatment_target",
            "output_is_clinical_conclusion",
            "current_sota_claim",
            "fresh_external_validation",
            "scores_are_calibrated_error_probabilities",
        )
    ):
        raise ValueError("v29 manifest claim boundary failed")
    if (
        claims.get("research_scalp_electrode_ranking_only") is not True
        or claims.get("physician_review_required") is not True
    ):
        raise ValueError("v29 manifest research/physician qualification is missing")
    _canonical_json_bytes(data)
    return data


def run_frozen_v29_candidate_batch(
    *,
    recording_path: str | Path,
    detection_manifest: object,
    event_id_assignment: object,
    device: torch.device,
    modeling_path: str | Path = DEFAULT_MODELING,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    direct_states_path: str | Path = DEFAULT_DIRECT_STATES,
    h_states_path: str | Path = DEFAULT_H_STATES,
    public_freeze_manifest_path: str | Path = DEFAULT_PUBLIC_FREEZE_MANIFEST,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Run frozen v29 on detector-selected events from one long EDF."""

    pre_ranking_windows = materialize_v29_pre_ranking_window_receipts(
        recording_path=recording_path,
        detection_manifest=detection_manifest,
        event_id_assignment=event_id_assignment,
    )
    candidate_batch = canonicalize_v29_candidate_batch(
        detection_manifest, pre_ranking_windows
    )
    if candidate_batch["event_count"] < 1:
        raise ValueError("v29 ranking requires at least one selected detector candidate")
    runtime = load_frozen_v29_runtime(
        modeling_path=modeling_path,
        checkpoint_path=checkpoint_path,
        direct_states_path=direct_states_path,
        h_states_path=h_states_path,
        public_freeze_manifest_path=public_freeze_manifest_path,
        device=device,
    )
    features = extract_v29_candidate_features(
        recording_path=recording_path,
        candidate_batch=candidate_batch,
        pre_ranking_window_receipts=pre_ranking_windows,
        encoder=runtime.encoder,
        device=device,
    )
    probabilities = infer_v29_probabilities(
        features.h_features,
        features.phase_features,
        direct_states=runtime.direct_states,
        h_states=runtime.h_states,
    )
    manifest, tensors = assemble_v29_candidate_rankings(
        candidate_batch=candidate_batch,
        features=features,
        probabilities=probabilities,
        model_receipt=runtime.model_receipt,
    )
    return validate_v29_candidate_ranking_manifest(manifest), tensors


def run_filtered_frozen_v29_candidate_batch(
    *,
    recording_path: str | Path,
    detection_manifest: object,
    event_id_assignment: object,
    device: torch.device,
    modeling_path: str | Path = DEFAULT_MODELING,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    direct_states_path: str | Path = DEFAULT_DIRECT_STATES,
    h_states_path: str | Path = DEFAULT_H_STATES,
    public_freeze_manifest_path: str | Path = DEFAULT_PUBLIC_FREEZE_MANIFEST,
    event_loader: Callable[..., object] = load_standard19_edf_event,
) -> tuple[
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, torch.Tensor],
]:
    """Run v29 only for signal-eligible candidates and preserve all rejects.

    When no selected candidate is analyzable, the model is deliberately not
    loaded and no placeholder probability tensor is created.  The returned
    selection still proves exact detector-candidate coverage and is sufficient
    for an evidence-insufficient recording report.
    """

    selection, pre_ranking_windows = materialize_v29_filtered_analysis_selection(
        recording_path=recording_path,
        detection_manifest=detection_manifest,
        event_id_assignment=event_id_assignment,
        event_loader=event_loader,
    )
    if selection["analyzable_count"] == 0:
        return selection, None, {}
    candidate_batch = canonicalize_v29_candidate_batch(
        detection_manifest,
        pre_ranking_windows,
        analysis_selection=selection,
    )
    runtime = load_frozen_v29_runtime(
        modeling_path=modeling_path,
        checkpoint_path=checkpoint_path,
        direct_states_path=direct_states_path,
        h_states_path=h_states_path,
        public_freeze_manifest_path=public_freeze_manifest_path,
        device=device,
    )
    features = extract_v29_candidate_features(
        recording_path=recording_path,
        candidate_batch=candidate_batch,
        pre_ranking_window_receipts=pre_ranking_windows,
        encoder=runtime.encoder,
        device=device,
        event_loader=event_loader,
    )
    probabilities = infer_v29_probabilities(
        features.h_features,
        features.phase_features,
        direct_states=runtime.direct_states,
        h_states=runtime.h_states,
    )
    manifest, tensors = assemble_v29_candidate_rankings(
        candidate_batch=candidate_batch,
        features=features,
        probabilities=probabilities,
        model_receipt=runtime.model_receipt,
    )
    validated = validate_v29_candidate_ranking_manifest(manifest)
    if validated.get("analysis_selection_sha256") != canonical_payload_sha256(
        selection
    ):
        raise RuntimeError("filtered v29 ranking lost its analysis-selection binding")
    return selection, validated, tensors


def finalize_v29_ranked_segment_drafts(
    segment_drafts: Sequence[object],
    ranking_manifest: object,
) -> list[dict[str, Any]]:
    """Inject rankings into unranked drafts, then validate final segments.

    Drafts must omit both ``segment_receipt_id`` and
    ``research_soz_ranking_receipt``.  This ordering prevents callers from
    fabricating a placeholder ranking merely to satisfy the final schema.
    """

    manifest = validate_v29_candidate_ranking_manifest(ranking_manifest)
    if isinstance(segment_drafts, (str, bytes)) or not isinstance(
        segment_drafts, Sequence
    ):
        raise TypeError("segment_drafts must be an array")
    ranking_events = {event["candidate_id"]: event for event in manifest["events"]}
    draft_keys = (
        "schema_version",
        "recording_id",
        "patient_pseudonym",
        "source_signal_sha256",
        "recording_duration_seconds",
        "candidate_id",
        "eeg_event_id",
        "candidate_anchor_offset_seconds",
        "requested_window_seconds",
        "segment_start_offset_seconds",
        "segment_stop_offset_seconds",
        "warmup_seconds_available",
        "post_anchor_seconds_available",
        "boundary_policy",
        "processed_window_sha256",
        "preprocessing_receipt_sha256",
        "event_report_payload",
        "waveform_attachment",
    )
    result: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    for index, raw in enumerate(segment_drafts):
        draft = _strict_object(
            raw,
            required=draft_keys,
            context=f"unranked segment draft[{index}]",
        )
        candidate_id = _identifier(draft["candidate_id"], "draft.candidate_id")
        if candidate_id in seen_candidates:
            raise ValueError("unranked segment drafts repeat a detector candidate")
        seen_candidates.add(candidate_id)
        ranking_event = ranking_events.get(candidate_id)
        if ranking_event is None:
            raise ValueError("ranking manifest does not cover a segment draft")
        window = ranking_event["pre_ranking_window_receipt"]
        exact_bindings = {
            "recording_id": manifest["recording_id"],
            "patient_pseudonym": manifest["patient_pseudonym"],
            "source_signal_sha256": manifest["source_signal_sha256"],
            "candidate_id": ranking_event["candidate_id"],
            "eeg_event_id": ranking_event["eeg_event_id"],
            "processed_window_sha256": ranking_event["processed_window_sha256"],
            "preprocessing_receipt_sha256": window[
                "preprocessing_receipt_sha256"
            ],
        }
        if any(draft[key] != value for key, value in exact_bindings.items()):
            raise ValueError("unranked segment draft identity/hash binding failed")
        time_bindings = {
            "recording_duration_seconds": manifest["recording_duration_seconds"],
            "candidate_anchor_offset_seconds": ranking_event[
                "candidate_anchor_offset_seconds"
            ],
            "segment_start_offset_seconds": window["window_start_offset_seconds"],
            "segment_stop_offset_seconds": window["window_stop_offset_seconds"],
            "warmup_seconds_available": window["warmup_seconds_available"],
            "post_anchor_seconds_available": window["post_anchor_seconds_available"],
        }
        if any(not _same_time(draft[key], value) for key, value in time_bindings.items()):
            raise ValueError("unranked segment draft time binding failed")
        updated = deepcopy(draft)
        updated["research_soz_ranking_receipt"] = deepcopy(
            ranking_event["research_soz_ranking_receipt"]
        )
        updated["segment_receipt_id"] = "CONTENT-ADDRESS-PENDING"
        updated["segment_receipt_id"] = (
            f"SEG-V29-{canonical_payload_sha256(updated)[:20]}"
        )
        result.append(validate_long_term_event_segment_receipt(updated))
    if set(ranking_events) != seen_candidates:
        raise ValueError("segment drafts do not exactly cover ranking manifest")
    result.sort(
        key=lambda segment: (
            segment["candidate_anchor_offset_seconds"],
            segment["eeg_event_id"],
        )
    )
    return result


__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_DIRECT_STATES",
    "DEFAULT_H_STATES",
    "DEFAULT_MODELING",
    "DEFAULT_PUBLIC_FREEZE_MANIFEST",
    "DETECTOR_ALIGNED_EVENT_REGISTRY_SCHEMA_VERSION",
    "EVENT_ID_ASSIGNMENT_SCHEMA_VERSION",
    "EXPECTED_DIRECT_STATES_SHA256",
    "EXPECTED_H_STATES_SHA256",
    "EXPECTED_PUBLIC_FREEZE_MANIFEST_SHA256",
    "ExtractedV29CandidateFeatures",
    "FrozenV29Runtime",
    "INFERENCE_POLICY",
    "FILTERED_INPUT_SCHEMA_VERSION",
    "FILTERED_OUTPUT_SCHEMA_VERSION",
    "INPUT_SCHEMA_VERSION",
    "METHOD_ID",
    "OUTPUT_SCHEMA_VERSION",
    "OUTPUT_STATUS",
    "PRE_RANKING_WINDOW_RECEIPT_SCHEMA_VERSION",
    "PROCESSED_WINDOW_HASH_POLICY",
    "TENSOR_FILE",
    "assemble_v29_candidate_rankings",
    "canonicalize_v29_candidate_batch",
    "extract_v29_candidate_features",
    "finalize_v29_ranked_segment_drafts",
    "infer_v29_probabilities",
    "load_frozen_v29_runtime",
    "materialize_v29_pre_ranking_window_receipts",
    "materialize_v29_filtered_analysis_selection",
    "preprocessing_receipt_sha256",
    "processed_window_sha256",
    "resolve_v29_event_id_assignment",
    "run_frozen_v29_candidate_batch",
    "run_filtered_frozen_v29_candidate_batch",
    "validate_v29_candidate_batch",
    "validate_v29_candidate_ranking_manifest",
    "validate_v29_event_id_assignment",
    "validate_v29_pre_ranking_window_receipt",
]
