#!/usr/bin/env python3
"""Materialize the source-train-only frozen LaBraM block-10 prefix cache.

The script opens the physically isolated 582-event source-train I/V
capability and reloads the historical Frozen-H crosswalk through its
source-only entry point.  It reads only the corresponding raw TUSZ EEG,
never opens a target artifact, and performs no optimization.  ``--limit`` is
an implementation smoke path with a schema that can never be accepted as the
formal full cache.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_tusz_ictal_token_cache import (  # noqa: E402
    _load_target_free_signal,
    load_formal_token_corpus,
)
from src.soz.data.tusz_training import (  # noqa: E402
    TUSZIctalEventRecord,
    load_tusz_ictal_training_manifest,
)
from src.soz.frozen_h_crosswalk import (  # noqa: E402
    _signal_tensor_sha256,
    load_frozen_h_source_train_crosswalk_from_source_only,
)
from src.soz.ictal_native_eval import (  # noqa: E402
    load_bound_deepsoz_signal_preflight_artifact,
)
from src.soz.ictal_recovery_evidence_v1_2 import (  # noqa: E402
    load_target_free_ictal_oof_protocol,
)
from src.soz.labram_peft_prefix_cache import (  # noqa: E402
    EXPECTED_SOURCE_TRAIN_EVENT_COUNT,
    EXPECTED_SOURCE_TRAIN_EVENT_ORDER_SHA256,
    EXPECTED_SOURCE_TRAIN_PATIENT_COUNT,
    LABRAM_PEFT_PREFIX_EVENT_SHAPE,
    publish_labram_peft_prefix_cache,
)
from src.soz.models.labram import (  # noqa: E402
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    bind_labram_record_positions,
)
from src.soz.models.labram_peft import (  # noqa: E402
    OfficialLaBraMFrozenPrefixEncoder,
    OfficialLaBraMMinimalPEFTSuffix,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    load_preprocessing_selection_capability,
)
from src.soz.source_train_iv_capability import (  # noqa: E402
    load_source_train_iv_capability,
)


SOURCE_TRAIN_CAPABILITY_PATH = (
    ROOT / "outputs/labram_iv_source_train_only_capability_v1_20260811"
)
SOURCE_TRAIN_CAPABILITY_MANIFEST_SHA256 = (
    "ccd238b17e1da0aa24f2542a314c770900eeed71cbc31282a4acb76dcf957821"
)
SIGNAL_PATH = ROOT / "outputs/deepsoz_signal_preflight_v2_20260809_current"
OOF_PROTOCOL_PATH = ROOT / "outputs/ictal_concept_oof_protocol_v2_20260808"
MASTER_MANIFEST_PATH = (
    ROOT / "outputs/tusz_ictal_master_manifest_v4_1_20260809_current_preflight"
)
PREPROCESSING_PATH = (
    ROOT / "outputs/preprocessing_parity_formal_v2_1_20260809/selection-capability"
)
TOKEN_CORPUS_PATH = (
    ROOT / "outputs/tusz_ictal_token_corpus_formal_v4_20260809/master"
)
CROSSWALK_PATH = (
    ROOT / "outputs/labram_frozen_h_source_train_crosswalk_v1_20260810"
)
FORMAL_OUTPUT_PATH = ROOT / "outputs/labram_peft_prefix_cache_v8_20260811"

SIGNAL_ARTIFACT_SHA256 = (
    "a2fdf45dd122e39ec6e73b3a3edafa1264669875fd2d8cd2b9cb7e8313d1ee66"
)
SIGNAL_RECEIPT_SHA256 = (
    "10128ad30d2163838222d0b4a27d9889a767276a9b697812e3cf568a3d9fd446"
)
OOF_PROTOCOL_ARTIFACT_SHA256 = (
    "cd1893031873b81053678316ed36145c1ba572d33ae332d221bc0907e1e0bca0"
)
OOF_PROTOCOL_RECEIPT_SHA256 = (
    "a1668bfaa9b3489851251924d618e2c107503455183bf54e0b44ae1613ed4803"
)
MASTER_BUNDLE_SHA256 = (
    "73e821d08805c3a7e8ae75011dd98fe10c388d7291c74881286438e91cacc35f"
)
MASTER_SOURCE_SHA256 = (
    "d5329b9231ecea7aaae6e126f5cd7a17a51f21b950025b32369592379acf8cb8"
)
PREPROCESSING_ARTIFACT_SHA256 = (
    "b4aa73bff2800f12186085976a5655db6882a38232d775d11234efa387171485"
)
PREPROCESSING_PROTOCOL_SHA256 = (
    "9a75dd2f3293d4d944380c0d82dcfca6a95e332f3b999e32e52b15d89622a196"
)
TOKEN_INDEX_SHA256 = (
    "a7d672e3228cdc71fafb46e910033f6a5302a9e2e0a5f5716f7f4c8292ecfc26"
)
CROSSWALK_MANIFEST_SHA256 = (
    "f5a0b40e7d9ecc48ffb2f10a76128da4e110b791db47ac09ace54495bd2d797b"
)
CROSSWALK_RECEIPT_SHA256 = (
    "4eec735065d93f761c1e17753977fe1f0e633d1fdbb6c6888f0af4eb78f6bbee"
)

DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_MODELING_PATH = Path(
    "/mnt/hd1/dyf/workspace/LaBraM/modeling_finetune.py"
)
DEFAULT_CHECKPOINT_PATH = Path(
    "/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth"
)


def split_source_event_into_labram_calls(eeg: torch.Tensor) -> torch.Tensor:
    """Return an exact non-overlapping ``[15,19,4,200]`` view/copy."""

    if not isinstance(eeg, torch.Tensor) or tuple(eeg.shape) != (19, 12_000):
        raise ValueError("Source-train EEG must have shape [19,12000]")
    if eeg.dtype != torch.float32 or eeg.requires_grad:
        raise TypeError("Source-train EEG must be detached float32")
    if not torch.isfinite(eeg).all().item():
        raise ValueError("Source-train EEG must be finite")
    calls = (
        eeg.contiguous()
        .reshape(19, 15, 4, 200)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    reassembled = calls.permute(1, 0, 2, 3).contiguous().reshape(19, 12_000)
    if not torch.equal(reassembled, eeg):
        raise RuntimeError("Four-second LaBraM calls do not exactly reassemble EEG")
    return calls


def _absolute_directory(path: str | Path, *, field: str) -> Path:
    result = Path(os.path.abspath(path))
    for component in (result, *result.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot traverse symlinks")
    if not result.is_dir():
        raise FileNotFoundError(result)
    return result


def _guard_output_topology(target: Path, inputs: Sequence[Path]) -> None:
    output = Path(os.path.abspath(target))
    for source_value in inputs:
        source = Path(os.path.abspath(source_value))
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"Prefix output overlaps immutable input {source}")


def _load_source_only_inputs():
    """Load the closed bridge without ever opening the shared v1.1 capability."""

    capability = load_source_train_iv_capability(
        SOURCE_TRAIN_CAPABILITY_PATH,
        expected_manifest_sha256=SOURCE_TRAIN_CAPABILITY_MANIFEST_SHA256,
    )
    signal = load_bound_deepsoz_signal_preflight_artifact(
        SIGNAL_PATH,
        expected_artifact_sha256=SIGNAL_ARTIFACT_SHA256,
        expected_receipt_sha256=SIGNAL_RECEIPT_SHA256,
    )
    protocol = load_target_free_ictal_oof_protocol(
        OOF_PROTOCOL_PATH,
        expected_artifact_sha256=OOF_PROTOCOL_ARTIFACT_SHA256,
        expected_protocol_receipt_sha256=OOF_PROTOCOL_RECEIPT_SHA256,
    )
    master = load_tusz_ictal_training_manifest(
        MASTER_MANIFEST_PATH,
        expected_bundle_manifest_sha256=MASTER_BUNDLE_SHA256,
        expected_source_manifest_sha256=MASTER_SOURCE_SHA256,
    )
    preprocessing = load_preprocessing_selection_capability(
        PREPROCESSING_PATH,
        expected_artifact_sha256=PREPROCESSING_ARTIFACT_SHA256,
        expected_protocol_receipt_sha256=PREPROCESSING_PROTOCOL_SHA256,
    )
    token_corpus = load_formal_token_corpus(
        TOKEN_CORPUS_PATH,
        expected_index_sha256=TOKEN_INDEX_SHA256,
        preprocessing_selection=preprocessing,
    )
    crosswalk = load_frozen_h_source_train_crosswalk_from_source_only(
        CROSSWALK_PATH,
        capability=capability,
        signal=signal,
        protocol=protocol,
        master_manifest=master,
        token_corpus=token_corpus,
        expected_manifest_sha256=CROSSWALK_MANIFEST_SHA256,
        expected_receipt_sha256=CROSSWALK_RECEIPT_SHA256,
    )
    checks = {
        "capability manifest": capability.manifest_sha256
        == SOURCE_TRAIN_CAPABILITY_MANIFEST_SHA256,
        "event count": len(crosswalk.events)
        == EXPECTED_SOURCE_TRAIN_EVENT_COUNT,
        "patient count": len(crosswalk.patient_ids)
        == EXPECTED_SOURCE_TRAIN_PATIENT_COUNT,
        "crosswalk order": crosswalk.receipt["event_order_sha256"]
        == EXPECTED_SOURCE_TRAIN_EVENT_ORDER_SHA256,
        "capability order": capability.receipt.event_order_sha256
        == EXPECTED_SOURCE_TRAIN_EVENT_ORDER_SHA256,
        "same events": tuple(crosswalk.event_ids) == tuple(capability.event_ids),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Source-train-only prefix inputs changed: {failed}")
    return capability, signal, protocol, master, token_corpus, crosswalk


def _master_by_event_id(master) -> dict[str, TUSZIctalEventRecord]:
    result: dict[str, TUSZIctalEventRecord] = {}
    for event in master:
        if event.event_id in result:
            raise ValueError("TUSZ master event IDs are not unique")
        result[event.event_id] = event
    return result


def _verify_master_row(
    row: Mapping[str, object], event: TUSZIctalEventRecord
) -> None:
    checks = {
        "event ID": event.event_id == row["token_event_id"],
        "public patient": event.patient_id == row["public_patient_id"],
        "relative EDF": event.relative_edf_path == row["relative_edf_path"],
        "global index": event.event_index == row["global_event_index"],
        "global t0": float(event.event_t0_sec) == float(row["global_t0_sec"]),
        "global stop": float(event.event_stop_sec)
        == float(row["global_stop_sec"]),
        "seizure type": event.seizure_type == row["seizure_type"],
        "event record": event.event_record_sha256
        == row["token_event_record_sha256"],
        "EDF": event.edf_sha256 == row["edf_sha256"],
        "channel annotation": event.channel_annotation_sha256
        == row["channel_annotation_sha256"],
        "global annotation": event.global_annotation_sha256
        == row["global_annotation_sha256"],
        "annotation pair": event.annotation_pair_sha256
        == row["annotation_pair_sha256"],
        "preprocess": event.signal_preflight_receipt_sha256
        == row["tusz_signal_preflight_receipt_sha256"],
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            f"Prefix master replay failed for {row['evidence_event_id']}: {failed}"
        )


def _event_cache_row(row: Mapping[str, object], *, ordinal: int) -> dict[str, object]:
    return {
        "ordinal": ordinal,
        "evidence_event_id": row["evidence_event_id"],
        "patient_id": row["target_patient_id"],
        "public_patient_id": row["public_patient_id"],
        "oof_fold": row["oof_fold"],
        "token_event_id": row["token_event_id"],
        "relative_edf_path": row["relative_edf_path"],
        "global_event_index": row["global_event_index"],
        "global_t0_sec": row["global_t0_sec"],
        "global_stop_sec": row["global_stop_sec"],
        "seizure_type": row["seizure_type"],
        "event_record_sha256": row["token_event_record_sha256"],
        "edf_sha256": row["edf_sha256"],
        "channel_annotation_sha256": row["channel_annotation_sha256"],
        "global_annotation_sha256": row["global_annotation_sha256"],
        "annotation_pair_sha256": row["annotation_pair_sha256"],
        "processed_window_sha256": row["processed_window_sha256"],
        "processed_window_shape": row["processed_window_shape"],
        "processed_window_dtype": row["processed_window_dtype"],
        "tusz_signal_preflight_receipt_sha256": row[
            "tusz_signal_preflight_receipt_sha256"
        ],
        "raw_replay_sha256": row["raw_replay_sha256"],
        "labram_position_binding_policy": row[
            "labram_position_binding_policy"
        ],
        "labram_position_names": row["labram_position_names"],
        "labram_position_ids": row["labram_position_ids"],
        "chunk_reassembly_exact": True,
    }


def materialize_labram_peft_prefix_cache(
    *,
    tusz_root: str | Path,
    modeling_path: str | Path,
    checkpoint_path: str | Path,
    output_directory: str | Path,
    device: str | torch.device,
    limit: int | None,
    progress_every: int = 10,
):
    raw_root = _absolute_directory(tusz_root, field="TUSZ root")
    target = Path(os.path.abspath(output_directory))
    _guard_output_topology(
        target,
        (
            raw_root,
            SOURCE_TRAIN_CAPABILITY_PATH,
            SIGNAL_PATH,
            OOF_PROTOCOL_PATH,
            MASTER_MANIFEST_PATH,
            PREPROCESSING_PATH,
            TOKEN_CORPUS_PATH,
            CROSSWALK_PATH,
        ),
    )
    execution_device = torch.device(device)
    if execution_device.type not in {"cpu", "cuda"} or execution_device.index is not None:
        raise ValueError("device must be cpu or cuda without an explicit index")
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if isinstance(progress_every, bool) or progress_every < 1:
        raise ValueError("progress_every must be positive")

    capability, signal, protocol, master, token_corpus, crosswalk = (
        _load_source_only_inputs()
    )
    full_scope = limit is None
    if full_scope:
        selected_count = EXPECTED_SOURCE_TRAIN_EVENT_COUNT
    else:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit < EXPECTED_SOURCE_TRAIN_EVENT_COUNT:
            raise ValueError("--limit smoke must be an integer in [1,581]")
        selected_count = limit
    rows = tuple(dict(row) for row in crosswalk.receipt["events"][:selected_count])
    master_by_id = _master_by_event_id(master)

    prefix = OfficialLaBraMFrozenPrefixEncoder(
        modeling_path=modeling_path,
        checkpoint_path=checkpoint_path,
        expected_sha256=AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
    ).to(execution_device)
    prefix.eval()
    if any(parameter.requires_grad for parameter in prefix.parameters()):
        raise RuntimeError("Frozen LaBraM prefix contains trainable parameters")
    foundation_feature_receipt = prefix.receipt

    tokens = torch.empty(
        (selected_count, *LABRAM_PEFT_PREFIX_EVENT_SHAPE), dtype=torch.float32
    )
    event_rows: list[dict[str, object]] = []
    if execution_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for ordinal, row in enumerate(rows):
        event = master_by_id.get(str(row["token_event_id"]))
        if event is None:
            raise ValueError("Crosswalk event no longer exists in TUSZ master")
        _verify_master_row(row, event)
        eeg = _load_target_free_signal(
            event,
            manifest=master,
            edf_root=raw_root,
            reader_factory=None,
            foundation_receipt=prefix.receipt,
        )
        checks = {
            "processed SHA": _signal_tensor_sha256(eeg)
            == row["processed_window_sha256"],
            "processed shape": list(eeg.shape) == row["processed_window_shape"],
            "processed dtype": str(eeg.dtype) == row["processed_window_dtype"],
            "EDF": event.edf_sha256 == row["edf_sha256"],
            "preprocess": event.signal_preflight_receipt_sha256
            == row["tusz_signal_preflight_receipt_sha256"],
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(
                f"Raw prefix replay failed for {row['evidence_event_id']}: {failed}"
            )
        binding = bind_labram_record_positions(row["labram_position_names"])
        if (
            list(binding.position_names) != row["labram_position_names"]
            or list(binding.position_ids) != row["labram_position_ids"]
            or binding.policy != row["labram_position_binding_policy"]
        ):
            raise ValueError("LaBraM record position binding changed")
        calls = split_source_event_into_labram_calls(eeg)
        with torch.inference_mode():
            encoded = prefix.forward_with_record_binding(
                calls.to(execution_device), binding
            ).detach().cpu().float().contiguous()
        if tuple(encoded.shape) != LABRAM_PEFT_PREFIX_EVENT_SHAPE or not torch.isfinite(encoded).all().item():
            raise RuntimeError("Frozen LaBraM prefix returned invalid activations")
        tokens[ordinal].copy_(encoded)
        event_rows.append(_event_cache_row(row, ordinal=ordinal))
        del encoded, calls, eeg
        if execution_device.type == "cuda":
            torch.cuda.empty_cache()
        if (ordinal + 1) % progress_every == 0 or ordinal + 1 == selected_count:
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "stage": "labram_peft_prefix_materialization",
                        "completed": ordinal + 1,
                        "total": selected_count,
                        "elapsed_sec": elapsed,
                        "seconds_per_event": elapsed / (ordinal + 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    # One operator-level control ties the cached prefix and zero-initialized
    # differentiable suffix to the independently published official full path.
    suffix = OfficialLaBraMMinimalPEFTSuffix(
        modeling_path=modeling_path,
        checkpoint_path=checkpoint_path,
        expected_sha256=AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
    ).to(execution_device)
    suffix.eval()
    with torch.inference_mode():
        zero_suffix = (
            suffix(tokens[0].to(execution_device))
            .detach()
            .cpu()
            .float()
            .contiguous()
        )
    official = (
        crosswalk.load_event_tokens(0)
        .permute(1, 0, 2, 3)
        .contiguous()
        .float()
    )
    if tuple(zero_suffix.shape) != (15, 19, 4, 200) or zero_suffix.shape != official.shape:
        raise RuntimeError("Zero-adapter equivalence control shape changed")
    equivalence_error = float((zero_suffix - official).abs().max().item())
    if equivalence_error > 1e-6:
        raise RuntimeError(
            "Zero-adapter prefix/suffix differs from official full forward: "
            f"{equivalence_error}"
        )
    del suffix, zero_suffix, official, prefix
    if execution_device.type == "cuda":
        torch.cuda.empty_cache()

    elapsed = time.monotonic() - started
    lineage = {
        "source_train_iv_manifest_sha256": capability.manifest_sha256,
        "source_train_iv_receipt_sha256": capability.receipt.receipt_sha256,
        "source_train_iv_event_order_sha256": capability.receipt.event_order_sha256,
        "crosswalk_manifest_sha256": crosswalk.manifest_sha256,
        "crosswalk_receipt_sha256": crosswalk.receipt_sha256,
        "crosswalk_parent_capability_manifest_sha256": crosswalk.receipt[
            "source_train_capability_manifest_sha256"
        ],
        "crosswalk_event_order_sha256": crosswalk.receipt["event_order_sha256"],
        "signal_preflight_artifact_sha256": signal.artifact_sha256,
        "signal_preflight_receipt_sha256": signal.receipt_sha256,
        "oof_protocol_artifact_sha256": protocol.artifact_sha256,
        "oof_protocol_receipt_sha256": protocol.receipt_sha256,
        "master_manifest_bundle_sha256": crosswalk.receipt[
            "master_manifest_bundle_sha256"
        ],
        "master_manifest_source_sha256": master.manifest_sha256,
        "formal_token_corpus_index_sha256": token_corpus.index_sha256,
        "formal_token_corpus_tensor_roster_sha256": token_corpus.tensor_roster_sha256,
        "preprocessing_selection_artifact_sha256": token_corpus.preprocessing_selection_artifact_sha256,
        "preprocessing_protocol_receipt_sha256": token_corpus.preprocessing_protocol_receipt_sha256,
    }
    return publish_labram_peft_prefix_cache(
        target,
        tokens=tokens,
        event_rows=event_rows,
        lineage=lineage,
        foundation_feature_receipt=foundation_feature_receipt,
        full_scope=full_scope,
        materialization_device=str(execution_device),
        elapsed_sec=elapsed,
        peak_cuda_memory_bytes=(
            int(torch.cuda.max_memory_allocated())
            if execution_device.type == "cuda"
            else None
        ),
        zero_adapter_official_equivalence_max_abs_error=equivalence_error,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument(
        "--labram-modeling-path", type=Path, default=DEFAULT_MODELING_PATH
    )
    parser.add_argument(
        "--labram-checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preflight_only:
        capability, _, _, _, _, crosswalk = _load_source_only_inputs()
        print(
            json.dumps(
                {
                    "stage": "source_train_only_preflight",
                    "capability_manifest_sha256": capability.manifest_sha256,
                    "crosswalk_manifest_sha256": crosswalk.manifest_sha256,
                    "event_count": len(crosswalk.events),
                    "patient_count": len(crosswalk.patient_ids),
                    "target_values_loaded": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.output_directory is None:
        output = (
            FORMAL_OUTPUT_PATH
            if args.limit is None
            else ROOT
            / f"outputs/labram_peft_prefix_cache_v8_smoke_limit{args.limit}_20260811"
        )
    else:
        output = args.output_directory
    artifact = materialize_labram_peft_prefix_cache(
        tusz_root=args.tusz_root,
        modeling_path=args.labram_modeling_path,
        checkpoint_path=args.labram_checkpoint_path,
        output_directory=output,
        device=args.device,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "path": str(artifact.path),
                "manifest_sha256": artifact.manifest_sha256,
                "schema_version": artifact.manifest["schema_version"],
                "full_scope": artifact.full_scope,
                "event_count": artifact.event_count,
                "patient_count": len(artifact.patient_ids),
                "tensor_shape": list(artifact.tokens.shape),
                "zero_adapter_official_equivalence_max_abs_error": artifact.manifest[
                    "zero_adapter_official_equivalence_max_abs_error"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
