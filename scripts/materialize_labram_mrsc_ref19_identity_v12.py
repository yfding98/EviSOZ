#!/usr/bin/env python3
"""Append recovered C-REF19 evidence for the identity-v16 MRSC roster.

The legacy target-free reference cache already contains the 984 complete-case
events used by v11.1.  This producer preserves those tensors bit-for-bit and
computes only the 161 identity-recovered events with the same raw EDF,
causal filter/resample/crop, and frozen official LaBraM block-9 encoder.

The sensitivity montage is C-REF19.  It is not a second ground truth and may
only be used for reference-sensitivity uncertainty/abstention.  This command
does not open SOZ targets, train, calibrate, select a model, or access private
data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
from typing import Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.soz.data.edf import CausalEDFConfig, load_standard19_edf_event  # noqa: E402
from src.soz.data.identity_v12_cache_extension import (  # noqa: E402
    append_event_tensor_exact,
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    load_identity_v12_extension_contract,
    tensor_bitwise_equal,
    tensor_sha256,
)
from src.soz.fine_temporal_evidence import (  # noqa: E402
    FINE_TEMPORAL_FEATURE_NAMES,
    extract_fine_temporal_evidence,
)
from src.soz.frozen_h_crosswalk import _signal_tensor_sha256  # noqa: E402
from src.soz.models.labram import (  # noqa: E402
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    bind_labram_record_positions,
)
from src.soz.models.labram_peft import OfficialLaBraMFrozenPrefixEncoder  # noqa: E402
from src.soz.preprocessing_arm_runtime import (  # noqa: E402
    CAUSAL_REFERENCE_PAIR_ROLE,
    CAUSAL_REFERENCE_PAIR_SCHEMA,
    CAUSAL_REFERENCE_SENSITIVITY_ARM_ID,
)


DEFAULT_UNION = ROOT / "outputs/public_development_union_identity_v12_20260812"
DEFAULT_SIGNAL = ROOT / "outputs/deepsoz_signal_preflight_identity_v3_20260812"
DEFAULT_ANCHOR = ROOT / "outputs/labram_identity_v16_anchor_target_excluding_20260812"
DEFAULT_LEGACY_REF = ROOT / "outputs/labram_mrsc_ref19_cache_20260812"
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_MODELING = Path("/mnt/hd1/dyf/workspace/LaBraM/modeling_finetune.py")
DEFAULT_CHECKPOINT = Path("/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth")
DEFAULT_OUTPUT = ROOT / "outputs/labram_mrsc_ref19_identity_v12_20260812"

SCHEMA = "soz_labram_mrsc_ref19_identity_v12_target_free_cache_v1"
SMOKE_SCHEMA = "soz_labram_mrsc_ref19_identity_v12_target_free_cache_smoke_v1"
ANCHOR_SCHEMA = "soz_labram_identity_v16_target_excluding_anchor_bridge_v1"
ANCHOR_STATUS = "completed_target_excluding_identity_v16_anchor_bridge"
LEGACY_SCHEMA = "soz_labram_mrsc_ref19_target_free_cache_v1"
LEGACY_STATUS = "completed_target_free_ref19_evidence_cache"

EXPECTED_UNION_MANIFEST_SHA256 = (
    "645c55541c37dfc204fdd48c21e0a3c81fe7201f76b862556d1c4dc3bfa4d429"
)
EXPECTED_SIGNAL_ARTIFACT_SHA256 = (
    "2a6bb8a7be20993949e7250b10c83d11fe027ff1afc0fa0919124f7fa371ef8e"
)
EXPECTED_ANCHOR_MANIFEST_SHA256 = (
    "d858ce31cabadc169f44e54c1307ab4bc370a349f72d113d0ca6d58fee2f7c86"
)
EXPECTED_ANCHOR_TENSOR_SHA256 = (
    "9e142643047a575d048ae9eadea22eb27bdb3c5239a1f3f68730ea596ef7a174"
)
EXPECTED_LEGACY_MANIFEST_SHA256 = (
    "7e6c1fe29d90e3b8b312257980729083ef24572327bc27cb4dc30e376b6444dc"
)
EXPECTED_LEGACY_TENSOR_SHA256 = (
    "159910662594418c9a1d971ee9352a009a19e5dbc50070c68ab2ecfa2f0f75a4"
)

LEGACY_PATIENT_COUNT = 101
LEGACY_EVENT_COUNT = 984
APPENDED_EVENT_COUNT = 161
PRIMARY_PATIENT_COUNT = 102
PRIMARY_EVENT_COUNT = 1145
PREFIX_EVENT_SHAPE = (15, 77, 200)
FINE_EVENT_SHAPE = (19, len(FINE_TEMPORAL_FEATURE_NAMES))
LEGACY_TENSOR_KEYS = frozenset(
    {"event_patient_index", "ref_fine_features", "ref_prefix_tokens"}
)
ANCHOR_TENSOR_KEYS = frozenset(
    {
        "candidate_indices",
        "car_event_scores",
        "car_patient_scores",
        "event_patient_index",
        "patient_event_counts",
        "patient_folds",
    }
)
OUTPUT_TENSOR_KEYS = LEGACY_TENSOR_KEYS


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


def _safe_edf(root: Path, relative_value: object) -> Path:
    relative = PurePosixPath(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".edf":
        raise ValueError("identity-v12 event has an unsafe EDF path")
    source = root.joinpath(*relative.parts)
    for component in (source, *source.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("identity-v12 EDF path cannot traverse symlinks")
    resolved = source.resolve(strict=True)
    if resolved.relative_to(root).as_posix() != relative.as_posix():
        raise ValueError("identity-v12 EDF path escaped the pinned TUSZ root")
    return resolved


def _event_calls(window: torch.Tensor) -> torch.Tensor:
    if tuple(window.shape) != (19, 12_000) or window.dtype != torch.float32:
        raise ValueError("C-REF19 event must be float32 [19,12000]")
    calls = window.reshape(19, 15, 4, 200).permute(1, 0, 2, 3).contiguous()
    restored = calls.permute(1, 0, 2, 3).reshape(19, 12_000).contiguous()
    if not tensor_bitwise_equal(restored, window):
        raise RuntimeError("LaBraM C-REF19 calls do not bitwise reassemble the event")
    return calls


def _validate_float_tensor(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    result = value.detach().cpu().contiguous()
    if (
        result.dtype != torch.float32
        or tuple(result.shape) != shape
        or not torch.isfinite(result).all()
    ):
        raise ValueError(f"{name} must be finite float32 {list(shape)}")
    return result


@dataclass(frozen=True)
class RefIdentityInputs:
    contract: object
    anchor_manifest: Mapping[str, object]
    legacy_manifest: Mapping[str, object]
    anchor_tensor_path: Path
    legacy_tensor_path: Path
    patient_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    anchor_event_patient_index: torch.Tensor
    selected_append: tuple[Mapping[str, object], ...]
    full_scope: bool


def _prepare_inputs(
    *,
    union_directory: Path,
    signal_directory: Path,
    anchor_directory: Path,
    legacy_ref_directory: Path,
    append_limit: int | None,
) -> RefIdentityInputs:
    contract = load_identity_v12_extension_contract(
        union_directory,
        signal_directory,
        expected_union_manifest_sha256=EXPECTED_UNION_MANIFEST_SHA256,
        expected_signal_artifact_sha256=EXPECTED_SIGNAL_ARTIFACT_SHA256,
    )
    anchor_root = anchor_directory.resolve(strict=True)
    legacy_root = legacy_ref_directory.resolve(strict=True)
    for name, root in (("anchor", anchor_root), ("legacy REF", legacy_root)):
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"{name} directory must be canonical")
    if tuple(sorted(path.name for path in anchor_root.iterdir())) != (
        "anchor_scores.safetensors",
        "manifest.json",
    ):
        raise ValueError("identity-v16 anchor violates its closed file schema")
    if tuple(sorted(path.name for path in legacy_root.iterdir())) != (
        "manifest.json",
        "ref19_evidence.safetensors",
    ):
        raise ValueError("legacy REF cache violates its closed file schema")

    anchor = _strict_json(
        anchor_root / "manifest.json",
        expected_sha256=EXPECTED_ANCHOR_MANIFEST_SHA256,
        name="identity-v16 target-excluding anchor manifest",
    )
    legacy = _strict_json(
        legacy_root / "manifest.json",
        expected_sha256=EXPECTED_LEGACY_MANIFEST_SHA256,
        name="legacy target-free REF manifest",
    )
    if anchor.get("schema_version") != ANCHOR_SCHEMA or (
        anchor.get("status") != ANCHOR_STATUS
    ):
        raise ValueError("identity-v16 anchor schema/status changed")
    if legacy.get("schema_version") != LEGACY_SCHEMA or (
        legacy.get("status") != LEGACY_STATUS
    ):
        raise ValueError("legacy REF schema/status changed")
    if (
        anchor.get("patient_count") != PRIMARY_PATIENT_COUNT
        or anchor.get("event_count") != PRIMARY_EVENT_COUNT
        or legacy.get("patient_count") != LEGACY_PATIENT_COUNT
        or legacy.get("event_count") != LEGACY_EVENT_COUNT
    ):
        raise ValueError("REF extension cohort counts changed")
    anchor_access = anchor.get("access_receipt")
    legacy_access = legacy.get("access_receipt")
    if not isinstance(anchor_access, Mapping) or not isinstance(legacy_access, Mapping):
        raise TypeError("REF extension inputs lack access receipts")
    for name, access in (("anchor", anchor_access), ("legacy REF", legacy_access)):
        for field in (
            "target_tensor_values_loaded",
            "private_eeg_loaded",
            "private_target_values_loaded",
            "training_performed",
        ):
            if field in access and access.get(field) is not False:
                raise ValueError(f"{name} violates target/private boundary: {field}")
    for field in (
        "deepsoz_target_values_loaded",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "training_performed",
        "model_or_threshold_selection_performed",
    ):
        if legacy_access.get(field) is not False:
            raise ValueError(f"legacy REF cache is not target-free: {field}")

    patient_ids = tuple(str(value) for value in anchor.get("patient_ids", ()))
    event_ids = tuple(str(value) for value in anchor.get("event_ids", ()))
    legacy_patient_ids = tuple(str(value) for value in legacy.get("patient_ids", ()))
    legacy_event_ids = tuple(str(value) for value in legacy.get("event_ids", ()))
    expected_primary_events = tuple(
        event for event in contract.events if str(event["patient_id"]) != "258"
    )
    expected_primary_ids = tuple(str(event["event_id"]) for event in expected_primary_events)
    if (
        len(patient_ids) != PRIMARY_PATIENT_COUNT
        or len(event_ids) != PRIMARY_EVENT_COUNT
        or event_ids != expected_primary_ids
        or legacy_patient_ids != patient_ids[:LEGACY_PATIENT_COUNT]
        or legacy_event_ids != event_ids[:LEGACY_EVENT_COUNT]
    ):
        raise ValueError("anchor/legacy/identity-v12 REF rosters differ")
    recovered = tuple(contract.appended_events)
    if tuple(str(event["event_id"]) for event in recovered) != event_ids[LEGACY_EVENT_COUNT:]:
        raise ValueError("identity-v12 recovered events are not the anchor append")
    if append_limit is None:
        selected_append = recovered
        full_scope = True
    else:
        if isinstance(append_limit, bool) or not 1 <= int(append_limit) < len(recovered):
            raise ValueError("append_limit must be a smoke prefix in [1,160]")
        selected_append = recovered[: int(append_limit)]
        full_scope = False

    anchor_tensor_path = anchor_root / str(anchor.get("tensor_file"))
    legacy_tensor_path = legacy_root / str(legacy.get("tensor_file"))
    if (
        anchor_tensor_path.name != "anchor_scores.safetensors"
        or anchor_tensor_path.is_symlink()
        or file_sha256(anchor_tensor_path) != EXPECTED_ANCHOR_TENSOR_SHA256
        or anchor.get("tensor_file_sha256") != EXPECTED_ANCHOR_TENSOR_SHA256
    ):
        raise ValueError("identity-v16 target-excluding anchor tensor changed")
    if (
        legacy_tensor_path.name != "ref19_evidence.safetensors"
        or legacy_tensor_path.is_symlink()
        or file_sha256(legacy_tensor_path) != EXPECTED_LEGACY_TENSOR_SHA256
    ):
        raise ValueError("legacy target-free REF tensor changed")
    with safe_open(str(anchor_tensor_path), framework="pt", device="cpu") as handle:
        if frozenset(handle.keys()) != ANCHOR_TENSOR_KEYS:
            raise ValueError("identity-v16 anchor tensor vocabulary changed")
        anchor_epi = handle.get_tensor("event_patient_index").detach().cpu().contiguous()
    if anchor_epi.dtype != torch.long or tuple(anchor_epi.shape) != (PRIMARY_EVENT_COUNT,):
        raise ValueError("identity-v16 anchor event routing changed")

    return RefIdentityInputs(
        contract=contract,
        anchor_manifest=anchor,
        legacy_manifest=legacy,
        anchor_tensor_path=anchor_tensor_path,
        legacy_tensor_path=legacy_tensor_path,
        patient_ids=patient_ids,
        event_ids=event_ids,
        anchor_event_patient_index=anchor_epi,
        selected_append=selected_append,
        full_scope=full_scope,
    )


def materialize(
    *,
    union_directory: Path,
    signal_directory: Path,
    anchor_directory: Path,
    legacy_ref_directory: Path,
    tusz_root: Path,
    modeling_path: Path,
    checkpoint_path: Path,
    output_directory: Path,
    device: torch.device,
    append_limit: int | None,
    progress_every: int,
) -> tuple[Path, Mapping[str, object]]:
    if device.type not in {"cpu", "cuda"} or device.index is not None:
        raise ValueError("device must be cpu or cuda without an explicit index")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if type(progress_every) is not int or progress_every < 1:
        raise ValueError("progress_every must be a positive integer")
    inputs = _prepare_inputs(
        union_directory=union_directory,
        signal_directory=signal_directory,
        anchor_directory=anchor_directory,
        legacy_ref_directory=legacy_ref_directory,
        append_limit=append_limit,
    )
    contract = inputs.contract
    config_payload = contract.signal_bundle.receipt.get("preprocess_config")
    if not isinstance(config_payload, Mapping):
        raise TypeError("identity-v3 signal receipt lacks preprocessing config")
    car_config = CausalEDFConfig(**dict(config_payload))
    if not car_config.apply_car19:
        raise ValueError("identity-v3 primary preprocessing is not C-CAR19")
    ref_config = replace(car_config, apply_car19=False)
    legacy_config = inputs.legacy_manifest.get("preprocess_config")
    if not isinstance(legacy_config, Mapping) or dict(legacy_config) != asdict(ref_config):
        raise ValueError("legacy and recovered C-REF19 preprocessing contracts differ")

    raw_root = Path(os.path.abspath(tusz_root)).resolve(strict=True)
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    target = Path(os.path.abspath(output_directory))
    if os.path.lexists(target):
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    for source in (
        Path(union_directory).resolve(),
        Path(signal_directory).resolve(),
        Path(anchor_directory).resolve(),
        Path(legacy_ref_directory).resolve(),
        raw_root,
        Path(modeling_path).resolve(strict=True),
        Path(checkpoint_path).resolve(strict=True),
    ):
        if target == source or target in source.parents or source in target.parents:
            raise ValueError("REF extension output overlaps an input")

    with safe_open(str(inputs.legacy_tensor_path), framework="pt", device="cpu") as handle:
        if frozenset(handle.keys()) != LEGACY_TENSOR_KEYS:
            raise ValueError("legacy REF tensor vocabulary changed")
        legacy_prefix = _validate_float_tensor(
            handle.get_tensor("ref_prefix_tokens"),
            name="legacy ref_prefix_tokens",
            shape=(LEGACY_EVENT_COUNT, *PREFIX_EVENT_SHAPE),
        )
        legacy_fine = _validate_float_tensor(
            handle.get_tensor("ref_fine_features"),
            name="legacy ref_fine_features",
            shape=(LEGACY_EVENT_COUNT, *FINE_EVENT_SHAPE),
        )
        legacy_epi = handle.get_tensor("event_patient_index").detach().cpu().contiguous()
    if legacy_epi.dtype != torch.long or tuple(legacy_epi.shape) != (LEGACY_EVENT_COUNT,):
        raise ValueError("legacy REF event routing changed")
    if not torch.equal(
        legacy_epi, inputs.anchor_event_patient_index[:LEGACY_EVENT_COUNT]
    ):
        raise ValueError("legacy REF routing differs from identity-v16 anchor")

    encoder = OfficialLaBraMFrozenPrefixEncoder(
        modeling_path=modeling_path,
        checkpoint_path=checkpoint_path,
        expected_sha256=AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
    ).to(device).eval()
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("frozen LaBraM reference encoder exposes trainable weights")
    encoder_dtype = next(encoder.parameters()).dtype

    new_prefixes: list[torch.Tensor] = []
    new_fine: list[torch.Tensor] = []
    new_rows: list[dict[str, object]] = []
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for position, event in enumerate(inputs.selected_append, start=1):
        path = _safe_edf(raw_root, event["relative_edf_path"])
        loaded = load_standard19_edf_event(
            path,
            float(event["global_t0_sec"]),
            config=ref_config,
        )
        ref = loaded.window.data.detach().cpu().float().contiguous()
        if tuple(ref.shape) != (19, 12_000) or not torch.isfinite(ref).all():
            raise ValueError("C-REF19 preprocessing returned an invalid event")
        if loaded.edf_receipt.edf_sha256 != event["edf_sha256"] or (
            canonical_sha256(asdict(loaded.edf_receipt)) != event["edf_receipt_sha256"]
        ):
            raise ValueError(f"C-REF19 EDF lineage changed: {event['event_id']}")
        car_replay = (ref - ref.mean(dim=0, keepdim=True)).contiguous()
        car_replay_sha = _signal_tensor_sha256(car_replay)
        if car_replay_sha != event["processed_window_sha256"]:
            raise ValueError(f"C-REF19/C-CAR19 algebraic pairing failed: {event['event_id']}")
        binding = bind_labram_record_positions(
            loaded.edf_receipt.raw_channel_names,
            semantic_channels=loaded.edf_receipt.semantic_channels,
        )
        with torch.inference_mode():
            prefix = encoder.forward_with_record_binding(
                _event_calls(ref).to(device=device, dtype=encoder_dtype),
                binding,
            )
        prefix = _validate_float_tensor(
            prefix.to(device="cpu", dtype=torch.float32),
            name="appended ref_prefix_tokens",
            shape=PREFIX_EVENT_SHAPE,
        )
        fine = _validate_float_tensor(
            extract_fine_temporal_evidence(ref, sfreq_hz=200.0).features,
            name="appended ref_fine_features",
            shape=FINE_EVENT_SHAPE,
        )
        new_prefixes.append(prefix)
        new_fine.append(fine)
        new_rows.append(
            {
                "ordinal": LEGACY_EVENT_COUNT + position - 1,
                "event_id": str(event["event_id"]),
                "patient_id": str(event["patient_id"]),
                "outer_fold": int(event["outer_fold"]),
                "relative_edf_path": str(event["relative_edf_path"]),
                "global_t0_sec": float(event["global_t0_sec"]),
                "edf_sha256": str(event["edf_sha256"]),
                "car_replay_tensor_sha256": car_replay_sha,
                "ref_waveform_tensor_sha256": tensor_sha256(ref),
                "ref_prefix_tensor_sha256": tensor_sha256(prefix),
                "ref_fine_tensor_sha256": tensor_sha256(fine),
                "position_names": list(binding.position_names),
                "position_ids": list(binding.position_ids),
            }
        )
        if position % progress_every == 0 or position == len(inputs.selected_append):
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "new_event": position,
                        "new_total": len(inputs.selected_append),
                        "elapsed_sec": round(elapsed, 2),
                        "seconds_per_new_event": round(elapsed / position, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    appended_prefix = torch.stack(new_prefixes).contiguous()
    appended_fine = torch.stack(new_fine).contiguous()
    combined_prefix = append_event_tensor_exact(
        legacy_prefix,
        appended_prefix,
        expected_legacy_count=LEGACY_EVENT_COUNT,
    )
    combined_fine = append_event_tensor_exact(
        legacy_fine,
        appended_fine,
        expected_legacy_count=LEGACY_EVENT_COUNT,
    )
    output_event_count = LEGACY_EVENT_COUNT + len(inputs.selected_append)
    event_ids = inputs.event_ids[:output_event_count]
    event_patient_index = inputs.anchor_event_patient_index[:output_event_count].contiguous()
    if (
        tuple(combined_prefix.shape) != (output_event_count, *PREFIX_EVENT_SHAPE)
        or tuple(combined_fine.shape) != (output_event_count, *FINE_EVENT_SHAPE)
        or not tensor_bitwise_equal(
            combined_prefix[:LEGACY_EVENT_COUNT], legacy_prefix
        )
        or not tensor_bitwise_equal(combined_fine[:LEGACY_EVENT_COUNT], legacy_fine)
        or not torch.equal(event_patient_index[:LEGACY_EVENT_COUNT], legacy_epi)
    ):
        raise RuntimeError("append-only C-REF19 tensor receipt failed")
    if inputs.full_scope and (
        output_event_count != PRIMARY_EVENT_COUNT
        or tuple(event_ids) != inputs.event_ids
        or event_patient_index.max().item() != PRIMARY_PATIENT_COUNT - 1
        or torch.bincount(event_patient_index, minlength=PRIMARY_PATIENT_COUNT).min().item()
        < 1
    ):
        raise RuntimeError("full C-REF19 identity-v12 patient/event bags do not close")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        tensor_path = staging / "ref19_evidence.safetensors"
        output_tensors = {
            "event_patient_index": event_patient_index,
            "ref_fine_features": combined_fine,
            "ref_prefix_tokens": combined_prefix,
        }
        save_file(output_tensors, str(tensor_path))
        elapsed = time.monotonic() - started
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        manifest: dict[str, object] = {
            "schema_version": SCHEMA if inputs.full_scope else SMOKE_SCHEMA,
            "status": "completed_target_free_ref19_identity_v12_evidence_cache",
            "purpose": "paired_reference_sensitivity_for_identity_v16_mrsc_only",
            "full_scope": inputs.full_scope,
            "smoke_only": not inputs.full_scope,
            "reference_pair": {
                "schema_version": CAUSAL_REFERENCE_PAIR_SCHEMA,
                "role": CAUSAL_REFERENCE_PAIR_ROLE,
                "primary": "C-CAR19",
                "sensitivity": CAUSAL_REFERENCE_SENSITIVITY_ARM_ID,
                "shared_filter_resample_crop_contract": True,
                "algebraic_car_replay_verified_for_all_new_events": True,
            },
            "patient_count": PRIMARY_PATIENT_COUNT,
            "event_count": output_event_count,
            "legacy_reused_event_count": LEGACY_EVENT_COUNT,
            "newly_computed_event_count": len(inputs.selected_append),
            "patient_ids": list(inputs.patient_ids),
            "event_ids": list(event_ids),
            "event_order_sha256": canonical_sha256(list(event_ids)),
            "new_events": new_rows,
            "fine_feature_names": list(FINE_TEMPORAL_FEATURE_NAMES),
            "preprocess_config": asdict(ref_config),
            "foundation_backbone": "official_pretrained_LaBraM_Base_not_replaced",
            "foundation_checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
            "foundation_modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
            "foundation_trainable_parameters": 0,
            "tensor_file": tensor_path.name,
            "tensor_file_sha256": file_sha256(tensor_path),
            "tensor_specs": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in sorted(output_tensors.items())
            },
            "tensor_integrity": {
                "ref_prefix_tensor_sha256": tensor_sha256(combined_prefix),
                "ref_fine_tensor_sha256": tensor_sha256(combined_fine),
                "event_patient_index_sha256": tensor_sha256(event_patient_index),
                "legacy_prefix_bitwise_exact": True,
                "legacy_fine_bitwise_exact": True,
                "legacy_event_routing_exact": True,
                "append_only": True,
            },
            "lineage": {
                "public_union_manifest_sha256": contract.union_manifest_sha256,
                "signal_identity_recovery_artifact_sha256": (
                    contract.signal_bundle.artifact_sha256
                ),
                "signal_identity_recovery_receipt_sha256": (
                    contract.signal_bundle.receipt_sha256
                ),
                "signal_preprocess_config_sha256": (
                    contract.signal_bundle.receipt["preprocess_config_sha256"]
                ),
                "anchor_manifest_sha256": EXPECTED_ANCHOR_MANIFEST_SHA256,
                "anchor_tensor_sha256": EXPECTED_ANCHOR_TENSOR_SHA256,
                "legacy_ref_manifest_sha256": EXPECTED_LEGACY_MANIFEST_SHA256,
                "legacy_ref_tensor_sha256": EXPECTED_LEGACY_TENSOR_SHA256,
                "tusz_root": str(raw_root),
            },
            "materialization": {
                "device": str(device),
                "elapsed_sec": elapsed,
                "seconds_per_new_event": elapsed / len(inputs.selected_append),
                "peak_cuda_memory_bytes": int(peak),
                "complete_patient_bags": inputs.full_scope,
            },
            "access_receipt": {
                "legacy_target_free_ref_cache_loaded": True,
                "target_excluding_anchor_loaded": True,
                "raw_public_eeg_loaded": True,
                "raw_public_event_count": len(inputs.selected_append),
                "deepsoz_target_values_loaded": False,
                "tusz_channel_annotation_values_loaded": False,
                "historical_mixed_prediction_container_opened": False,
                "private_eeg_loaded": False,
                "private_target_values_loaded": False,
                "foundation_training_performed": False,
                "reasoner_training_performed": False,
                "calibration_performed": False,
                "model_or_threshold_selection_performed": False,
            },
            "claim_boundary": {
                "reference_evidence_is_sensitivity_not_second_ground_truth": True,
                "fine_change_is_not_soz_onset_or_propagation_truth": True,
                "cache_contains_no_soz_targets": True,
                "cache_does_not_change_car19_anchor_scores": True,
                "not_external_validation": True,
            },
        }
        (staging / "manifest.json").write_bytes(canonical_bytes(manifest, newline=True))
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--union-directory", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--signal-directory", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--anchor-directory", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--legacy-ref-directory", type=Path, default=DEFAULT_LEGACY_REF)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--modeling-path", type=Path, default=DEFAULT_MODELING)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--append-limit", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path, manifest = materialize(
        union_directory=args.union_directory,
        signal_directory=args.signal_directory,
        anchor_directory=args.anchor_directory,
        legacy_ref_directory=args.legacy_ref_directory,
        tusz_root=args.tusz_root,
        modeling_path=args.modeling_path,
        checkpoint_path=args.checkpoint_path,
        output_directory=args.output_directory,
        device=torch.device(args.device),
        append_limit=args.append_limit,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "status": "LABRAM_REF19_IDENTITY_V12_EXTENDED",
                "path": str(path),
                "manifest_sha256": file_sha256(path / "manifest.json"),
                "tensor_sha256": manifest["tensor_file_sha256"],
                "event_count": manifest["event_count"],
                "newly_computed_event_count": manifest[
                    "newly_computed_event_count"
                ],
                "full_scope": manifest["full_scope"],
                "target_values_loaded": False,
                "private_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
