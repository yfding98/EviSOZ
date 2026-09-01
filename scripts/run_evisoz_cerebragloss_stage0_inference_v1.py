#!/usr/bin/env python3
"""Run a strictly bounded, development-only CerebraGloss inference job.

This adapter is deliberately separate from the typed teacher-cache importer.
It is the only repository entry point that executes the external CerebraGloss
checkpoint.  It reads only already materialized EviSOZ dual-montage caches,
never opens an EDF or physician report, and emits an auditable *input
envelope*.  The envelope can subsequently be passed to
``materialize_evisoz_teacher_candidates_v1.py``.

The checkpoint was trained for the CerebraGloss 19-channel, 200 Hz, 10-second
view.  EviSOZ supplies the exact CAR19 node view and cuts its frozen
``[-12, 48]``-second context at the canonical ``[-2, 8]`` onset window.  The
T3/T4/T5/T6 names expected by CerebraGloss are explicit aliases of the
canonical T7/T8/P7/P8 nodes; this mapping is recorded in the model manifest
and is never used to change EviSOZ node identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.artifact_ref import (  # noqa: E402
    build_json_artifact_ref,
    build_raw_artifact_ref,
    canonical_json_bytes,
    canonical_json_sha256,
)
from src.evisoz.data.stage0_dual_montage_cache import (  # noqa: E402
    open_stage0_dual_montage_cache_from_disk,
)
from src.soz.geometry import STANDARD_19  # noqa: E402


CG_CLASSES = (
    "sharp",
    "spike",
    "spsw",
    "spindle",
    "Kcomplex",
    "eyem",
    "eyer+",
    "eyer-",
    "hfnoise",
)
CG_CHANNEL_ORDER = (
    "FP1",
    "FP2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "FZ",
    "CZ",
    "PZ",
)
# EviSOZ's canonical Standard19 keeps modern names; CerebraGloss was trained
# with the older TUH naming convention.  Keep this as a named, reversible
# transport map rather than silently renaming the Evidence JSON units.
CG_TO_CANONICAL = {
    "FP1": "FP1",
    "FP2": "FP2",
    "F3": "F3",
    "F4": "F4",
    "C3": "C3",
    "C4": "C4",
    "P3": "P3",
    "P4": "P4",
    "O1": "O1",
    "O2": "O2",
    "F7": "F7",
    "F8": "F8",
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
    "FZ": "FZ",
    "CZ": "CZ",
    "PZ": "PZ",
}
CG_INDEX_FROM_CANONICAL = tuple(STANDARD_19.index(CG_TO_CANONICAL[name]) for name in CG_CHANNEL_ORDER)
MODEL_SCHEMA_VERSION = "evisoz_cerebragloss_teacher_model_manifest_v1"
INPUT_SCHEMA_VERSION = "evisoz_cerebragloss_stage0_inference_input_v1"
WINDOW_SECONDS = [-2.0, 8.0]
SAMPLING_RATE_HZ = 200
SEQ_LEN = 2000
ANCHORS = [[90 / SEQ_LEN, 300 / SEQ_LEN], [1900 / SEQ_LEN]]
MODEL_ANCHORS = [2, None, None, 1]
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5


def _json(path: Path, *, require_canonical: bool = True) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"not a regular JSON file: {path}")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if type(value) is not dict:
        raise ValueError(f"JSON object required: {path}")
    if require_canonical and raw != canonical_json_bytes(value):
        raise ValueError(f"JSON is not canonical: {path}")
    return value


def _artifact_ref_for_json(value: dict[str, Any], *, kind: str, schema: str) -> dict[str, Any]:
    return build_json_artifact_ref(value, artifact_kind=kind, payload_schema_version=schema)


def _artifact_ref_for_file(path: Path, *, kind: str, schema: str, media_type: str) -> dict[str, Any]:
    raw = path.read_bytes()
    return build_raw_artifact_ref(
        raw,
        artifact_kind=kind,
        media_type=media_type,
        payload_schema_version=schema,
    )


def _load_external_model(checkpoint: Path):
    external_root = checkpoint.parent.parent
    if str(external_root) not in sys.path:
        sys.path.insert(0, str(external_root))
    from CerebraGlossYOLO.model import CerebraGlossYOLO
    from CerebraGlossYOLO.utils import batch_nms, decode_predictions_fpn

    model = CerebraGlossYOLO(
        num_classes=len(CG_CLASSES),
        num_anchors_per_level=MODEL_ANCHORS,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, decode_predictions_fpn, batch_nms


def _inference_one(model, decode_predictions_fpn, batch_nms, tensor: torch.Tensor) -> list[dict[str, Any]]:
    if tuple(tensor.shape) != (19, 12000) or tensor.dtype is not torch.float32:
        raise ValueError("v29 reference tensor geometry drifted")
    # The source context is [-12,48] seconds and the canonical onset core is
    # the frozen [2000:4000] sample slice, i.e. [-2,8] seconds.
    core = tensor[:, 2000:4000].clone()
    # Reorder only for the external checkpoint's historical TUH channel order.
    x = core[list(CG_INDEX_FROM_CANONICAL)]
    x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
    with torch.no_grad():
        predictions = model(
            x.unsqueeze(0),
            torch.zeros((1, 19), dtype=torch.bool),
            torch.arange(19, dtype=torch.long).view(1, 19),
        )
    batch_indices, channel_indices, b_x, b_w, scores, labels = decode_predictions_fpn(
        predictions, ANCHORS, conf_threshold=CONFIDENCE_THRESHOLD
    )
    if len(scores) == 0:
        return []
    detections = batch_nms(
        batch_indices,
        channel_indices,
        b_x,
        b_w,
        scores,
        labels,
        NMS_IOU_THRESHOLD,
        1,
    )[0]
    rows: list[dict[str, Any]] = []
    for box, score, label in zip(
        detections["boxes"], detections["scores"], detections["labels"]
    ):
        # ``box`` is x_min, channel, x_max, channel+1 in normalized time.
        start = max(0.0, min(1.0, float(box[0]))) * 10.0
        end = max(0.0, min(1.0, float(box[2]))) * 10.0
        if not start < end:
            continue
        channel_index = int(box[1])
        if not 0 <= channel_index < len(CG_CHANNEL_ORDER):
            continue
        external_name = CG_CHANNEL_ORDER[channel_index]
        canonical_name = CG_TO_CANONICAL[external_name]
        rows.append(
            {
                "concept": f"{str(CG_CLASSES[int(label)]).lower()}_like",
                "support_kind": "node_interval",
                "support_view": "car19_context",
                "support_units": [canonical_name],
                # Convert from the 10-second input origin to onset-relative
                # seconds.  This is a candidate interval, not a clinical fact.
                "support_interval_seconds": [
                    round(WINDOW_SECONDS[0] + start, 6),
                    round(WINDOW_SECONDS[0] + end, 6),
                ],
                "confidence": round(float(score), 8),
                "probability_semantics": "cerebragloss_detector_score_uncalibrated",
                "authority": "offline_teacher",
                "status": "candidate_only",
                "calibration_state": "uncalibrated",
                "permitted_uses": ["soft_auxiliary"],
                "prohibited_uses": [
                    "clinical_label",
                    "measured_fact",
                    "node_localization_supervision",
                    "endpoint_expansion_from_edge",
                ],
            }
        )
    # Importer/contract computes candidate IDs from exactly this pending form.
    for row in rows:
        row["candidate_id"] = "CONTENT-ADDRESS-PENDING"
        row["candidate_id"] = "EVISOZ-TEACHER-CAND-" + canonical_json_sha256(row)[:24]
    rows.sort(key=lambda row: row["candidate_id"])
    return rows


def _model_manifest(checkpoint: Path) -> dict[str, Any]:
    checkpoint_ref = _artifact_ref_for_file(
        checkpoint,
        kind="teacher_model_checkpoint",
        schema="cerebragloss_yolo_checkpoint_v1",
        media_type="application/octet-stream",
    )
    body: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "teacher_id": "cerebragloss",
        "checkpoint_ref": checkpoint_ref,
        "checkpoint_path": str(checkpoint),
        "input_contract": {
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "window_seconds": WINDOW_SECONDS,
            "channel_order": list(CG_CHANNEL_ORDER),
            "channel_mask_supported": True,
            "normalization": "per_channel_zscore",
            "transport_from_evisoz_standard19": {
                "canonical_order": list(STANDARD_19),
                "external_to_canonical": dict(CG_TO_CANONICAL),
                "external_indices_from_canonical": list(CG_INDEX_FROM_CANONICAL),
            },
        },
        "classes": list(CG_CLASSES),
        "decode_contract": {
            "anchors": ANCHORS,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
        },
        "deployment_policy": {
            "runtime_teacher_required": False,
            "output_role": "candidate_only_uncalibrated_soft_auxiliary",
            "locked_test_allowed": False,
        },
    }
    body["manifest_sha256"] = "0" * 64
    body["manifest_sha256"] = canonical_json_sha256(body)
    return body


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--dual-root", type=Path, required=True)
    parser.add_argument("--split-roster", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_events < 1 or args.max_events > 2:
        raise ValueError("Stage-0 CerebraGloss inference is limited to 1-2 events")
    dual_root = args.dual_root.resolve(strict=True)
    split_path = args.split_roster.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("checkpoint must be a regular file")
    # The frozen split-roster publisher predates the canonical-JSON file
    # convention; its parsed object is still content-addressed below.
    split = _json(split_path, require_canonical=False)
    roster_ref = _artifact_ref_for_json(split, kind="split_roster", schema="evisoz_split_roster_v1")
    dual_manifest = _json(dual_root / "manifest.json")
    dev_events = [
        event for event in dual_manifest["events"] if event["evisoz_role"] == "development_cv"
    ]
    dev_events.sort(key=lambda event: event["event_id"])
    selected = dev_events[: args.max_events]
    if len(selected) != args.max_events:
        raise ValueError("dual montage manifest has fewer development events than requested")
    model, decode_predictions_fpn, batch_nms = _load_external_model(checkpoint)
    model_manifest = _model_manifest(checkpoint)
    model_ref = _artifact_ref_for_json(
        model_manifest,
        kind="teacher_model_manifest",
        schema=MODEL_SCHEMA_VERSION,
    )
    events: list[dict[str, Any]] = []
    for event in selected:
        cache_root = dual_root / event["relative_cache_path"]
        opened = open_stage0_dual_montage_cache_from_disk(cache_root)
        identity = _json(cache_root / "sidecars" / "event_identity.json")
        identity_ref = _artifact_ref_for_json(
            identity, kind="event_identity", schema="evisoz_event_identity_v1"
        )
        materialization = _json(cache_root / "audit" / "materialization_receipt.json")
        dual_ref = _artifact_ref_for_json(
            materialization,
            kind="dual_montage_cache_materialization_receipt",
            schema="evisoz_dual_montage_cache_materialization_receipt_v1",
        )
        candidates = _inference_one(
            model,
            decode_predictions_fpn,
            batch_nms,
            opened.checkout_v29_reference(),
        )
        events.append(
            {
                "event_id": event["event_id"],
                "linkage_group_id": event["linkage_group_id"],
                "outer_holdout_fold": event["outer_holdout_fold"],
                "event_identity_ref": identity_ref,
                "source_dual_montage_cache_ref": dual_ref,
                "input_view": "car19_context",
                "input_sampling_rate_hz": SAMPLING_RATE_HZ,
                "input_window_seconds": list(WINDOW_SECONDS),
                "candidate_rows": candidates,
            }
        )
    envelope = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "teacher_id": "cerebragloss",
        "source_split_roster_ref": roster_ref,
        "teacher_model_ref": model_ref,
        "teacher_model_manifest": model_manifest,
        "events": events,
        "scope": {
            "max_events": args.max_events,
            "selected_event_count": len(events),
            "development_only": True,
            "locked_test_included": False,
            "training_or_calibration_executed": False,
            "physician_report_read": False,
            "evisoz_role_of_outputs": "candidate_only_uncalibrated_soft_auxiliary",
        },
    }
    envelope["receipt_sha256"] = "0" * 64
    envelope["receipt_sha256"] = canonical_json_sha256(envelope)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "teacher_model_manifest.json").write_bytes(canonical_json_bytes(model_manifest))
    (output / "inference_input.json").write_bytes(canonical_json_bytes(envelope))
    # Keep the importer boundary intentionally narrow: the importer accepts
    # only the four fields needed to build typed caches.  The richer envelope
    # remains available as an audit artifact but cannot accidentally become a
    # new cache schema through permissive field forwarding.
    importer_input = {
        "teacher_id": envelope["teacher_id"],
        "source_split_roster_ref": envelope["source_split_roster_ref"],
        "teacher_model_ref": envelope["teacher_model_ref"],
        "events": envelope["events"],
    }
    (output / "teacher_import_input.json").write_bytes(canonical_json_bytes(importer_input))
    print(
        json.dumps(
            {
                "status": "completed_development_only_uncalibrated_candidate_input",
                "teacher_id": "cerebragloss",
                "event_count": len(events),
                "candidate_count": sum(len(event["candidate_rows"]) for event in events),
                "output": str(output),
                "receipt_sha256": envelope["receipt_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
