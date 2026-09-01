#!/usr/bin/env python3
"""Run target-free paired-reference MRSC for the identity-v16 OOF roster.

This runner consumes the target-excluding v16 C-CAR19 bridge, the paired
identity-v12 C-REF19 evidence cache, target-free C-CAR19 representation
caches, and frozen v16 outer-fold states.  C-CAR19 scores are copied without
change; C-REF19 is sensitivity evidence only.  No ranking is released because
the selective threshold is undefined, so every patient fails closed.

No SOZ target, private data, optimizer, calibration, or model-selection port
exists in this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from scripts.run_labram_mrsc_target_free_oof import (  # noqa: E402
    BRIDGE_TENSOR_KEYS_READ,
    OUTER_FOLDS,
    REF_CACHE_TENSOR_KEYS_READ,
    _distribution,
    _read_exact_tensors,
    _read_outer_states,
    _require_finite_float,
    _require_long,
    _required_outer_state_keys,
    _rowwise_normalized_jsd,
    _stable_topk,
    _topk_jaccard,
    assess_target_free_roster,
    compute_foldwise_reference_scores,
)
from src.soz.data.identity_v12_cache_extension import (  # noqa: E402
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    tensor_bitwise_equal,
    tensor_sha256,
)
from src.soz.data.public_development_union_identity_v12 import (  # noqa: E402
    EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_MANIFEST_SHA256,
    EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_PAYLOAD_SHA256,
    load_public_development_union_identity_v12,
)
from src.soz.fine_temporal_evidence import FINE_TEMPORAL_FEATURE_NAMES  # noqa: E402
from src.soz.mrsc import (  # noqa: E402
    MRSC_CANDIDATE_CHANNELS,
    MRSC_NONCONFORMITY_SEMANTICS,
    MRSC_REPORT_FACT_FIELDS,
    MRSC_SCHEMA,
    MRSC_USE_POLICY,
)
from src.soz.v11_reasoner import (  # noqa: E402
    V11_CANDIDATE_INDICES,
    extract_block9_phase_contrasts,
)


DEFAULT_ANCHOR = ROOT / "outputs/labram_identity_v16_anchor_target_excluding_20260812"
DEFAULT_REF = ROOT / "outputs/labram_mrsc_ref19_identity_v12_20260812"
DEFAULT_CAR_PREFIX = ROOT / "outputs/public_development_labram_prefix_identity_v12_20260812"
DEFAULT_CAR_FINE = ROOT / "outputs/public_development_fine_evidence_identity_v12_20260812"
DEFAULT_UNION = ROOT / "outputs/public_development_union_identity_v12_20260812"
DEFAULT_OUTER_STATES = (
    ROOT
    / "outputs/labram_identity_recovery_closed_replay_v16_20260812"
    / "outer_fold_states.safetensors"
)
DEFAULT_OUTPUT = ROOT / "outputs/labram_mrsc_target_free_identity_v16_20260812"

SCHEMA = "soz_labram_mrsc_target_free_identity_v16_descriptive_v1"
STATUS = "completed_target_free_identity_v16_mrsc_all_rankings_abstain"
MODEL_LINEAGE = "identity_v16_frozen_outer_fold_full_labram_plus_fine"
ANCHOR_SCHEMA = "soz_labram_identity_v16_target_excluding_anchor_bridge_v1"
ANCHOR_STATUS = "completed_target_excluding_identity_v16_anchor_bridge"
REF_SCHEMA = "soz_labram_mrsc_ref19_identity_v12_target_free_cache_v1"
REF_STATUS = "completed_target_free_ref19_identity_v12_evidence_cache"
CAR_PREFIX_SCHEMA = "soz_public_development_labram_prefix_identity_v12"
CAR_FINE_SCHEMA = "soz_public_development_fine_evidence_identity_v12"

EXPECTED_ANCHOR_MANIFEST_SHA256 = (
    "d858ce31cabadc169f44e54c1307ab4bc370a349f72d113d0ca6d58fee2f7c86"
)
EXPECTED_ANCHOR_TENSOR_SHA256 = (
    "9e142643047a575d048ae9eadea22eb27bdb3c5239a1f3f68730ea596ef7a174"
)
EXPECTED_REF_MANIFEST_SHA256 = (
    "ce557c456b5003b26a6a12716e71958a171337bbd3a662ec931e5c232d089c8b"
)
EXPECTED_REF_TENSOR_SHA256 = (
    "416ec430f2c9f5d8585f304a7bf5e88710b41c5542cb91aeb1ba474c337bb416"
)
EXPECTED_CAR_PREFIX_MANIFEST_SHA256 = (
    "defb6e608051e2767b49d8a566b6d0f5ea768e0f22d5a1fb46b28929df2fbe64"
)
EXPECTED_CAR_PREFIX_TENSOR_SHA256 = (
    "727382c1d072b6b4a59a7bdec3f6ff8c7e771179cd3eb1fe6c2550840b58583d"
)
EXPECTED_CAR_FINE_MANIFEST_SHA256 = (
    "6368cd6ea7ec30217bb69f3a742e1ee697dcae109125948a58b4fc927c2a8839"
)
EXPECTED_CAR_FINE_TENSOR_SHA256 = (
    "8f5dc0ab75eeeeeffda1a70650218d6e02f35f20a316481b1bb4028ca851809a"
)
EXPECTED_OUTER_STATES_SHA256 = (
    "18b69f5e2fc718d2668b3a727f9a3f7bf0da33a613896d939559260ad3009b98"
)

PATIENT_COUNT = 102
EVENT_COUNT = 1145
UNION_EVENT_COUNT = 1149
BLOCKED_PATIENT = "258"
REPLAY_TOLERANCE = 1e-6
OUTPUT_TENSOR_FILENAME = "mrsc_target_free.safetensors"
CAR_PREFIX_TENSOR_KEYS = frozenset({"prefix_tokens"})
CAR_FINE_TENSOR_KEYS = frozenset(
    {
        "bipolar_change_detected",
        "bipolar_change_latency_sec",
        "composite_trace",
        "dominant_frequency_hz",
        "features",
        "node_change_detected",
        "node_change_latency_sec",
        "window_center_sec",
    }
)


def _strict_json(path: Path, *, expected_sha256: str, name: str) -> dict[str, object]:
    source = path.resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{name} must be a canonical regular file")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{name} SHA256 mismatch")

    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate field {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{name} contains non-finite constant {value}")

    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict) or canonical_bytes(payload, newline=True) != raw:
        raise ValueError(f"{name} is not canonical JSON")
    return payload


def _require_target_free_access(payload: Mapping[str, object], *, name: str) -> None:
    access = payload.get("access_receipt")
    if not isinstance(access, Mapping):
        raise TypeError(f"{name} lacks an access receipt")
    fields = (
        "deepsoz_target_values_loaded",
        "target_tensor_values_loaded",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "training_performed",
        "foundation_training_performed",
        "reasoner_training_performed",
        "calibration_performed",
        "model_or_threshold_selection_performed",
        "threshold_or_model_selection_performed",
    )
    for field in fields:
        if field in access and access.get(field) is not False:
            raise ValueError(f"{name} violates target/private/fit boundary: {field}")


def _load_anchor(
    directory: Path,
) -> tuple[dict[str, object], dict[str, torch.Tensor], Path]:
    root = directory.resolve(strict=True)
    if root.is_symlink() or not root.is_dir() or tuple(
        sorted(path.name for path in root.iterdir())
    ) != ("anchor_scores.safetensors", "manifest.json"):
        raise ValueError("identity-v16 anchor violates its closed directory schema")
    manifest = _strict_json(
        root / "manifest.json",
        expected_sha256=EXPECTED_ANCHOR_MANIFEST_SHA256,
        name="identity-v16 anchor manifest",
    )
    if manifest.get("schema_version") != ANCHOR_SCHEMA or (
        manifest.get("status") != ANCHOR_STATUS
    ):
        raise ValueError("identity-v16 anchor schema/status changed")
    _require_target_free_access(manifest, name="identity-v16 anchor")
    tensor_path = root / "anchor_scores.safetensors"
    if file_sha256(tensor_path) != EXPECTED_ANCHOR_TENSOR_SHA256 or (
        manifest.get("tensor_file_sha256") != EXPECTED_ANCHOR_TENSOR_SHA256
    ):
        raise ValueError("identity-v16 anchor tensor changed")
    payload = _read_exact_tensors(
        tensor_path,
        keys=BRIDGE_TENSOR_KEYS_READ,
        name="identity-v16 target-excluding anchor tensor",
    )
    return manifest, payload, tensor_path


def _load_ref(
    directory: Path,
    *,
    expected_manifest_sha256: str,
    expected_tensor_sha256: str,
) -> tuple[dict[str, object], dict[str, torch.Tensor], Path]:
    root = directory.resolve(strict=True)
    if root.is_symlink() or not root.is_dir() or tuple(
        sorted(path.name for path in root.iterdir())
    ) != ("manifest.json", "ref19_evidence.safetensors"):
        raise ValueError("identity-v12 REF cache violates its closed directory schema")
    manifest = _strict_json(
        root / "manifest.json",
        expected_sha256=expected_manifest_sha256,
        name="identity-v12 REF manifest",
    )
    if (
        manifest.get("schema_version") != REF_SCHEMA
        or manifest.get("status") != REF_STATUS
        or manifest.get("full_scope") is not True
        or manifest.get("smoke_only") is not False
    ):
        raise ValueError("identity-v12 REF cache is not the complete formal cache")
    _require_target_free_access(manifest, name="identity-v12 REF cache")
    tensor_path = root / "ref19_evidence.safetensors"
    if file_sha256(tensor_path) != expected_tensor_sha256 or (
        manifest.get("tensor_file_sha256") != expected_tensor_sha256
    ):
        raise ValueError("identity-v12 REF tensor changed")
    payload = _read_exact_tensors(
        tensor_path,
        keys=REF_CACHE_TENSOR_KEYS_READ,
        name="identity-v12 target-free REF tensor",
    )
    return manifest, payload, tensor_path


def _load_car_cache(
    directory: Path,
    *,
    expected_schema: str,
    expected_manifest_sha256: str,
    expected_tensor_sha256: str,
    tensor_key: str,
    expected_tensor_keys: frozenset[str],
    tail_shape: tuple[int, ...],
    union_event_ids: tuple[str, ...],
) -> torch.Tensor:
    root = directory.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("identity-v12 CAR cache directory must be canonical")
    manifest = _strict_json(
        root / "manifest.json",
        expected_sha256=expected_manifest_sha256,
        name=f"identity-v12 CAR {tensor_key} manifest",
    )
    if (
        manifest.get("schema_version") != expected_schema
        or manifest.get("full_scope") is not True
        or manifest.get("smoke_only") is not False
        or manifest.get("event_count") != UNION_EVENT_COUNT
        or tuple(str(value) for value in manifest.get("event_ids", ()))
        != union_event_ids
    ):
        raise ValueError(f"identity-v12 CAR {tensor_key} manifest changed")
    _require_target_free_access(manifest, name=f"identity-v12 CAR {tensor_key}")
    tensor_path = root / str(manifest.get("tensor_file"))
    if tensor_path.is_symlink() or file_sha256(tensor_path) != expected_tensor_sha256 or (
        manifest.get("tensor_file_sha256") != expected_tensor_sha256
    ):
        raise ValueError(f"identity-v12 CAR {tensor_key} tensor changed")
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        available = frozenset(handle.keys())
        if available != expected_tensor_keys or tensor_key not in available:
            raise ValueError(f"identity-v12 CAR {tensor_key} vocabulary changed")
        if any(
            token in key.lower()
            for key in available
            for token in ("target", "label", "private")
        ):
            raise ValueError(f"identity-v12 CAR {tensor_key} exposes a forbidden field")
        value = handle.get_tensor(tensor_key).detach().cpu().contiguous()
    if (
        value.dtype != torch.float32
        or tuple(value.shape) != (UNION_EVENT_COUNT, *tail_shape)
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"identity-v12 CAR {tensor_key} tensor is invalid")
    return value


def _descriptive_results(
    tensors: Mapping[str, torch.Tensor],
    *,
    patients: int,
    events: int,
    review_vocabulary: Sequence[str],
    abstention_vocabulary: Sequence[str],
) -> dict[str, object]:
    patient_agreement = tensors["mrsc_top1_reference_agreement"]
    event_agreement = tensors["event_top1_reference_agreement"]
    return {
        "patient_reference_agreement": {
            "top1_agreement_count": int(patient_agreement.sum()),
            "patient_count": patients,
            "top1_agreement_rate": float(patient_agreement.double().mean()),
            "top3_jaccard": _distribution(tensors["mrsc_top3_reference_jaccard"]),
            "final_score_normalized_jsd": _distribution(
                tensors["mrsc_final_score_reference_disagreement"]
            ),
        },
        "event_reference_agreement": {
            "top1_agreement_count": int(event_agreement.sum()),
            "event_count": events,
            "top1_agreement_rate": float(event_agreement.double().mean()),
            "top3_jaccard": _distribution(tensors["event_top3_reference_jaccard"]),
            "final_score_normalized_jsd": _distribution(
                tensors["event_final_score_reference_disagreement"]
            ),
        },
        "mrsc_uncertainty": {
            "ranking_ambiguity": _distribution(tensors["mrsc_ranking_ambiguity"]),
            "within_patient_event_dispersion": _distribution(
                tensors["mrsc_event_dispersion"],
                valid_mask=tensors["mrsc_event_dispersion_estimable"],
            ),
            "event_dispersion_not_estimable_count": int(
                (~tensors["mrsc_event_dispersion_estimable"]).sum()
            ),
            "final_score_reference_disagreement": _distribution(
                tensors["mrsc_final_score_reference_disagreement"]
            ),
            "structural_quality_uncertainty": _distribution(
                tensors["mrsc_signal_quality_uncertainty"]
            ),
            "report_fact_unavailability": _distribution(
                tensors["mrsc_report_fact_unavailability"]
            ),
            "raw_uncalibrated_nonconformity": _distribution(
                tensors["mrsc_raw_nonconformity"]
            ),
        },
        "reason_code_counts": {
            "review": {
                code: int(tensors["mrsc_review_reason_flags"][:, index].sum())
                for index, code in enumerate(review_vocabulary)
            },
            "abstention": {
                code: int(tensors["mrsc_abstention_reason_flags"][:, index].sum())
                for index, code in enumerate(abstention_vocabulary)
            },
        },
        "no_soz_correctness_stratification": True,
        "no_fold_stratification": True,
        "quantiles_are_descriptive_not_operating_thresholds": True,
    }


def materialize_target_free_identity_v16_mrsc(
    *,
    anchor_directory: Path,
    ref_directory: Path,
    car_prefix_directory: Path,
    car_fine_directory: Path,
    union_directory: Path,
    outer_fold_states_path: Path,
    output_directory: Path,
    expected_ref_manifest_sha256: str = EXPECTED_REF_MANIFEST_SHA256,
    expected_ref_tensor_sha256: str = EXPECTED_REF_TENSOR_SHA256,
) -> dict[str, object]:
    union = load_public_development_union_identity_v12(
        union_directory,
        expected_manifest_sha256=(
            EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_MANIFEST_SHA256
        ),
        expected_payload_sha256=(
            EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_PAYLOAD_SHA256
        ),
    )
    anchor, anchor_payload, _ = _load_anchor(anchor_directory)
    ref, ref_payload, _ = _load_ref(
        ref_directory,
        expected_manifest_sha256=expected_ref_manifest_sha256,
        expected_tensor_sha256=expected_ref_tensor_sha256,
    )
    patient_ids = tuple(str(value) for value in anchor.get("patient_ids", ()))
    event_ids = tuple(str(value) for value in anchor.get("event_ids", ()))
    if (
        len(patient_ids) != PATIENT_COUNT
        or len(event_ids) != EVENT_COUNT
        or tuple(str(value) for value in ref.get("patient_ids", ())) != patient_ids
        or tuple(str(value) for value in ref.get("event_ids", ())) != event_ids
        or tuple(str(value) for value in anchor.get("candidate_channels", ()))
        != MRSC_CANDIDATE_CHANNELS
    ):
        raise ValueError("identity-v16 C-CAR19/C-REF19 rosters differ")
    blocked = anchor.get("report_join_blocks")
    if (
        not isinstance(blocked, list)
        or len(blocked) != 4
        or any(
            row.get("patient_id") != BLOCKED_PATIENT
            or row.get("reason") != "mrsc_anchor_identity_not_available"
            for row in blocked
            if isinstance(row, Mapping)
        )
        or any(not isinstance(row, Mapping) for row in blocked)
    ):
        raise ValueError("patient 258 report-join block roster changed")

    candidate_indices = _require_long(
        anchor_payload["candidate_indices"],
        name="candidate_indices",
        shape=(18,),
    )
    if not torch.equal(
        candidate_indices, torch.tensor(V11_CANDIDATE_INDICES, dtype=torch.long)
    ):
        raise ValueError("fixed-C18 candidate order changed")
    car_patient_scores = _require_finite_float(
        anchor_payload["car_patient_scores"],
        name="car_patient_scores",
        shape=(PATIENT_COUNT, 18),
    )
    car_event_scores = _require_finite_float(
        anchor_payload["car_event_scores"],
        name="car_event_scores",
        shape=(EVENT_COUNT, 18),
    )
    event_patient_index = _require_long(
        anchor_payload["event_patient_index"],
        name="event_patient_index",
        shape=(EVENT_COUNT,),
    )
    patient_event_counts = _require_long(
        anchor_payload["patient_event_counts"],
        name="patient_event_counts",
        shape=(PATIENT_COUNT,),
    )
    patient_folds = _require_long(
        anchor_payload["patient_folds"],
        name="patient_folds",
        shape=(PATIENT_COUNT,),
    )
    ref_epi = _require_long(
        ref_payload["event_patient_index"],
        name="ref_event_patient_index",
        shape=(EVENT_COUNT,),
    )
    if not torch.equal(event_patient_index, ref_epi) or not torch.equal(
        torch.bincount(event_patient_index, minlength=PATIENT_COUNT),
        patient_event_counts,
    ):
        raise ValueError("identity-v16 C-CAR19/C-REF19 event routing differs")

    union_event_ids = tuple(event.event_id for event in union.events)
    union_position = {event_id: index for index, event_id in enumerate(union_event_ids)}
    if any(event_id not in union_position for event_id in event_ids):
        raise ValueError("identity-v16 anchor events are absent from identity-v12 union")
    selected = torch.tensor([union_position[event_id] for event_id in event_ids], dtype=torch.long)
    car_prefix_all = _load_car_cache(
        car_prefix_directory,
        expected_schema=CAR_PREFIX_SCHEMA,
        expected_manifest_sha256=EXPECTED_CAR_PREFIX_MANIFEST_SHA256,
        expected_tensor_sha256=EXPECTED_CAR_PREFIX_TENSOR_SHA256,
        tensor_key="prefix_tokens",
        expected_tensor_keys=CAR_PREFIX_TENSOR_KEYS,
        tail_shape=(15, 77, 200),
        union_event_ids=union_event_ids,
    )
    car_fine_all = _load_car_cache(
        car_fine_directory,
        expected_schema=CAR_FINE_SCHEMA,
        expected_manifest_sha256=EXPECTED_CAR_FINE_MANIFEST_SHA256,
        expected_tensor_sha256=EXPECTED_CAR_FINE_TENSOR_SHA256,
        tensor_key="features",
        expected_tensor_keys=CAR_FINE_TENSOR_KEYS,
        tail_shape=(19, 20),
        union_event_ids=union_event_ids,
    )
    car_h = extract_block9_phase_contrasts(car_prefix_all.index_select(0, selected))
    car_fine = car_fine_all.index_select(0, selected).contiguous()
    del car_prefix_all, car_fine_all

    outer_path = outer_fold_states_path.resolve(strict=True)
    if outer_path.is_symlink() or file_sha256(outer_path) != EXPECTED_OUTER_STATES_SHA256:
        raise ValueError("identity-v16 outer-fold states changed")
    outer_states = _read_outer_states(outer_path)
    car_replay = compute_foldwise_reference_scores(
        car_h,
        car_fine,
        event_patient_index,
        patient_folds,
        patient_event_counts,
        outer_states,
    )
    car_patient_replay = car_replay.patient_scores_19.index_select(
        1, candidate_indices
    ).contiguous()
    car_event_replay = car_replay.event_scores_19.index_select(
        1, candidate_indices
    ).contiguous()
    patient_difference = float((car_patient_replay - car_patient_scores).abs().max())
    event_difference = float((car_event_replay - car_event_scores).abs().max())
    patient_top1_replay_count = int(
        (car_patient_replay.argmax(dim=1) == car_patient_scores.argmax(dim=1)).sum()
    )
    event_top1_replay_count = int(
        (car_event_replay.argmax(dim=1) == car_event_scores.argmax(dim=1)).sum()
    )
    if (
        patient_difference > REPLAY_TOLERANCE
        or event_difference > REPLAY_TOLERANCE
        or patient_top1_replay_count != PATIENT_COUNT
        or event_top1_replay_count != EVENT_COUNT
    ):
        raise RuntimeError(
            "identity-v16 CAR replay gate failed: "
            f"patient_max_abs={patient_difference}, event_max_abs={event_difference}"
        )
    del car_h, car_fine, car_patient_replay, car_event_replay

    ref_prefix = _require_finite_float(
        ref_payload["ref_prefix_tokens"],
        name="ref_prefix_tokens",
        shape=(EVENT_COUNT, 15, 77, 200),
    )
    ref_fine = _require_finite_float(
        ref_payload["ref_fine_features"],
        name="ref_fine_features",
        shape=(EVENT_COUNT, 19, 20),
    )
    if tuple(str(value) for value in ref.get("fine_feature_names", ())) != (
        FINE_TEMPORAL_FEATURE_NAMES
    ):
        raise ValueError("identity-v12 REF fine feature vocabulary changed")
    ref_h = extract_block9_phase_contrasts(ref_prefix)
    del ref_prefix, ref_payload
    foldwise = compute_foldwise_reference_scores(
        ref_h,
        ref_fine,
        event_patient_index,
        patient_folds,
        patient_event_counts,
        outer_states,
    )
    if foldwise.fold_held_patient_counts != car_replay.fold_held_patient_counts or (
        foldwise.fold_held_event_counts != car_replay.fold_held_event_counts
    ):
        raise RuntimeError("identity-v16 CAR/REF fold routing receipts differ")
    ref_patient_scores = foldwise.patient_scores_19.index_select(
        1, candidate_indices
    ).contiguous()
    ref_event_scores = foldwise.event_scores_19.index_select(
        1, candidate_indices
    ).contiguous()

    structural_quality_valid = torch.ones((EVENT_COUNT, 18), dtype=torch.bool)
    report_facts_unavailable = {field: False for field in MRSC_REPORT_FACT_FIELDS}
    roster = assess_target_free_roster(
        car_patient_scores,
        ref_patient_scores,
        car_event_scores,
        event_patient_index,
        structural_quality_valid,
        report_facts_unavailable,
    )
    output_tensors = dict(roster.tensors)
    output_tensors.update(
        {
            "candidate_indices": candidate_indices,
            "car_event_scores_preserved": car_event_scores.clone(),
            "ref_event_scores": ref_event_scores,
            "event_patient_index": event_patient_index,
            "patient_event_counts": patient_event_counts,
            "event_structural_quality_valid_mask": structural_quality_valid,
            "report_fact_available_mask": torch.tensor(
                [report_facts_unavailable[field] for field in MRSC_REPORT_FACT_FIELDS],
                dtype=torch.bool,
            ),
        }
    )
    if not tensor_bitwise_equal(
        output_tensors["car_patient_scores_preserved"], car_patient_scores
    ) or not tensor_bitwise_equal(
        output_tensors["car_event_scores_preserved"], car_event_scores
    ):
        raise RuntimeError("identity-v16 C-CAR19 bitwise score parity failed")
    if int(output_tensors["mrsc_abstain"].sum()) != PATIENT_COUNT:
        raise RuntimeError("undefined-threshold MRSC must abstain for all 102 patients")

    car_event_top1 = _stable_topk(car_event_scores, 1).flatten()
    ref_event_top1 = _stable_topk(ref_event_scores, 1).flatten()
    event_top1_agreement = car_event_top1 == ref_event_top1
    event_jsd = _rowwise_normalized_jsd(car_event_scores, ref_event_scores)
    event_top3 = _topk_jaccard(car_event_scores, ref_event_scores, k=3)
    output_tensors.update(
        {
            "event_car_top1_index": car_event_top1,
            "event_ref_top1_index": ref_event_top1,
            "event_top1_reference_agreement": event_top1_agreement,
            "event_top3_reference_jaccard": event_top3,
            "event_final_score_reference_disagreement": event_jsd,
        }
    )
    descriptive = _descriptive_results(
        output_tensors,
        patients=PATIENT_COUNT,
        events=EVENT_COUNT,
        review_vocabulary=roster.review_reason_vocabulary,
        abstention_vocabulary=roster.abstention_reason_vocabulary,
    )

    target = Path(os.path.abspath(output_directory))
    if os.path.lexists(target):
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        tensor_path = staging / OUTPUT_TENSOR_FILENAME
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in output_tensors.items()},
            str(tensor_path),
        )
        manifest: dict[str, object] = {
            "schema_version": SCHEMA,
            "status": STATUS,
            "model_lineage": MODEL_LINEAGE,
            "primary_reference": "C-CAR19_preserved",
            "sensitivity_reference": "C-REF19_same_event_same_frozen_fold_model",
            "patient_count": PATIENT_COUNT,
            "event_count": EVENT_COUNT,
            "patient_ids": list(patient_ids),
            "event_ids": list(event_ids),
            "candidate_channels": list(MRSC_CANDIDATE_CHANNELS),
            "report_join_blocks": blocked,
            "tensor_file": tensor_path.name,
            "tensor_file_sha256": file_sha256(tensor_path),
            "tensor_specs": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in sorted(output_tensors.items())
            },
            "tensor_integrity": {
                "car_patient_scores_tensor_sha256": tensor_sha256(car_patient_scores),
                "car_event_scores_tensor_sha256": tensor_sha256(car_event_scores),
                "output_car_patient_scores_tensor_sha256": tensor_sha256(
                    output_tensors["car_patient_scores_preserved"]
                ),
                "output_car_event_scores_tensor_sha256": tensor_sha256(
                    output_tensors["car_event_scores_preserved"]
                ),
            },
            "fold_checkpoint_routing": {
                "outer_folds": list(OUTER_FOLDS),
                "held_patient_counts": list(foldwise.fold_held_patient_counts),
                "held_event_counts": list(foldwise.fold_held_event_counts),
                "used_only_to_select_historical_oof_state": True,
                "fold_not_passed_to_reasoner_or_mrsc": True,
            },
            "score_parity": {
                "outer_state_car_replay_maximum_allowed_absolute_difference": (
                    REPLAY_TOLERANCE
                ),
                "outer_state_car_patient_replay_maximum_absolute_difference": (
                    patient_difference
                ),
                "outer_state_car_event_replay_maximum_absolute_difference": (
                    event_difference
                ),
                "outer_state_car_patient_top1_replay_count": (
                    patient_top1_replay_count
                ),
                "outer_state_car_event_top1_replay_count": event_top1_replay_count,
                "outer_state_car_replay_gate_passed": True,
                "car_patient_bitwise_equal_before_after_mrsc": True,
                "car_event_bitwise_equal_before_after_mrsc": True,
                "maximum_absolute_car_score_change": 0.0,
                "score_or_ranking_change_performed": False,
            },
            "mrsc_contract": {
                "core_schema": MRSC_SCHEMA,
                "nonconformity_semantics": MRSC_NONCONFORMITY_SEMANTICS,
                "use_policy": MRSC_USE_POLICY,
                "selective_threshold_defined": False,
                "all_patients_fail_closed": True,
                "report_fact_fields": list(MRSC_REPORT_FACT_FIELDS),
                "report_fact_available_mask": report_facts_unavailable,
                "review_reason_vocabulary": list(roster.review_reason_vocabulary),
                "abstention_reason_vocabulary": list(
                    roster.abstention_reason_vocabulary
                ),
            },
            "quality_contract": {
                "mask_semantics": "finite_complete_structural_carrier_only",
                "artifact_quality_not_materialized": True,
                "quality_port_cannot_increase_or_rerank_car_scores": True,
            },
            "descriptive_results": descriptive,
            "lineage": {
                "anchor_manifest_sha256": EXPECTED_ANCHOR_MANIFEST_SHA256,
                "anchor_tensor_sha256": EXPECTED_ANCHOR_TENSOR_SHA256,
                "ref_manifest_sha256": expected_ref_manifest_sha256,
                "ref_tensor_sha256": expected_ref_tensor_sha256,
                "car_prefix_manifest_sha256": EXPECTED_CAR_PREFIX_MANIFEST_SHA256,
                "car_prefix_tensor_sha256": EXPECTED_CAR_PREFIX_TENSOR_SHA256,
                "car_fine_manifest_sha256": EXPECTED_CAR_FINE_MANIFEST_SHA256,
                "car_fine_tensor_sha256": EXPECTED_CAR_FINE_TENSOR_SHA256,
                "outer_fold_states_sha256": EXPECTED_OUTER_STATES_SHA256,
                "union_manifest_sha256": union.manifest_sha256,
            },
            "access_receipt": {
                "target_excluding_anchor_only": True,
                "target_free_ref19_cache_only": True,
                "target_free_car_prefix_and_fine_cache_only": True,
                "historical_mixed_prediction_container_opened": False,
                "outer_fold_state_checkpoint_opened": True,
                "source_tensor_keys_read": {
                    "anchor": list(BRIDGE_TENSOR_KEYS_READ),
                    "ref19_cache": list(REF_CACHE_TENSOR_KEYS_READ),
                    "car_replay_caches": ["prefix_tokens", "features"],
                    "outer_states": list(_required_outer_state_keys()),
                },
                "deepsoz_target_values_loaded": False,
                "target_tensor_values_loaded": False,
                "private_eeg_loaded": False,
                "private_target_values_loaded": False,
                "training_performed": False,
                "optimizer_parameters": 0,
                "model_selection_performed": False,
                "calibration_performed": False,
                "threshold_selection_or_calibration_performed": False,
                "soz_outcome_metrics_computed": False,
                "label_based_subgrouping_performed": False,
                "patient_identity_passed_to_mrsc": False,
                "fold_id_passed_to_mrsc": False,
            },
            "claim_boundary": {
                "developmental_oof_reference_sensitivity_only": True,
                "not_external_validation": True,
                "not_a_new_soz_localizer": True,
                "does_not_improve_or_change_car_ranking": True,
                "all_rankings_abstain_until_independent_threshold_calibration": True,
                "reference_disagreement_is_not_error_probability": True,
                "patient_258_four_events_remain_blocked": True,
                "private_validation_allowed_from_this_artifact": False,
            },
        }
        (staging / "manifest.json").write_bytes(canonical_bytes(manifest, newline=True))
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--anchor-directory", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--ref-directory", type=Path, default=DEFAULT_REF)
    parser.add_argument("--car-prefix-directory", type=Path, default=DEFAULT_CAR_PREFIX)
    parser.add_argument("--car-fine-directory", type=Path, default=DEFAULT_CAR_FINE)
    parser.add_argument("--union-directory", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--outer-fold-states", type=Path, default=DEFAULT_OUTER_STATES)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-ref-manifest-sha256", default=EXPECTED_REF_MANIFEST_SHA256
    )
    parser.add_argument(
        "--expected-ref-tensor-sha256", default=EXPECTED_REF_TENSOR_SHA256
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    args = parse_args(argv)
    manifest = materialize_target_free_identity_v16_mrsc(
        anchor_directory=args.anchor_directory,
        ref_directory=args.ref_directory,
        car_prefix_directory=args.car_prefix_directory,
        car_fine_directory=args.car_fine_directory,
        union_directory=args.union_directory,
        outer_fold_states_path=args.outer_fold_states,
        output_directory=args.output_directory,
        expected_ref_manifest_sha256=args.expected_ref_manifest_sha256,
        expected_ref_tensor_sha256=args.expected_ref_tensor_sha256,
    )
    patient = manifest["descriptive_results"]["patient_reference_agreement"]
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(args.output_directory),
                "patient_count": manifest["patient_count"],
                "event_count": manifest["event_count"],
                "blocked_event_count": len(manifest["report_join_blocks"]),
                "patient_top1_reference_agreement_rate": patient[
                    "top1_agreement_rate"
                ],
                "car_score_change": 0.0,
                "all_rankings_abstain": True,
                "target_values_loaded": False,
                "private_used": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
