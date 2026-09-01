#!/usr/bin/env python3
"""Run the frozen one-shot official-dev transport test for native TUSZ I.

The command rebuilds both formal-v5 heads and verifies their historical final
states before it opens any new official-dev channel annotation.  Official
eval, DeepSOZ SOZ values, and private data are never loaded.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Mapping, Sequence


_REQUIRED_CUBLAS_WORKSPACE = ":4096:8"
observed_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if observed_workspace is None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _REQUIRED_CUBLAS_WORKSPACE
elif observed_workspace != _REQUIRED_CUBLAS_WORKSPACE:
    raise RuntimeError("Fresh-I transport requires CUBLAS_WORKSPACE_CONFIG=':4096:8'")

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_tusz_eeg_only_s1_labram_prefix import (  # noqa: E402
    FULL_SCHEMA as PREFIX_SCHEMA,
    TENSOR_NAME as PREFIX_TENSOR_NAME,
)
from scripts.materialize_tusz_ictal_token_cache import (  # noqa: E402
    load_formal_token_corpus,
)
from scripts.run_ictal_v5_dev import (  # noqa: E402
    _collect_native_targets,
    _load_split,
    _train_head,
)
from src.soz.cached_concept_training import (  # noqa: E402
    IctalTokenBagDataset,
    IctalTokenPatientBag,
)
from src.soz.concept_metrics import patient_macro_ictal_metrics  # noqa: E402
from src.soz.concept_run import IctalTrainingConfig  # noqa: E402
from src.soz.concept_token_io import load_labram_concept_tokens  # noqa: E402
from src.soz.data.tusz import (  # noqa: E402
    BIN_STATE_EXPLICIT_BACKGROUND,
    BIN_STATE_EXPLICIT_ICTAL,
    load_tusz_ictal_involvement_target,
)
from src.soz.data.tusz_training import load_tusz_ictal_training_manifest  # noqa: E402
from src.soz.ictal_native_fresh_transport import (  # noqa: E402
    decide_native_fresh_transport,
    paired_patient_improvements,
    patient_bootstrap_interval,
)
from src.soz.ictal_fit_only_consumer_v13 import (  # noqa: E402
    load_fit_only_target_artifact_v13,
)
from src.soz.ictal_v5 import prevalence_baseline_metrics, v5_shortcut_logits  # noqa: E402
from src.soz.models.concept_heads import (  # noqa: E402
    IctalInvolvementHead,
    TemporalResidualIctalInvolvementHead,
)
from src.soz.models.labram_static_suffix import (  # noqa: E402
    OfficialLaBraMStaticAdapterSuffix,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    load_preprocessing_selection_capability,
)


DEFAULT_PROTOCOL = ROOT / "configs/tusz_ictal_native_fresh_transport_v1.json"
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_MODELING = Path("/mnt/hd1/dyf/workspace/LaBraM/modeling_finetune.py")
DEFAULT_CHECKPOINT = Path("/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth")
DEFAULT_OUTPUT = ROOT / "outputs/tusz_ictal_native_fresh_transport_v1_20260815"
PROTOCOL_SCHEMA = "tusz_ictal_native_fresh_transport_protocol_v1"
PROTOCOL_STATUS = "frozen_before_official_dev_channel_target_open"
RESULT_SCHEMA = "tusz_ictal_native_fresh_transport_result_v1"
_EVENT_INDEX_RE = re.compile(r"__ev(?P<index>[0-9]{4})$")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(value: object) -> Path:
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Protocol repository paths must be safe relative paths")
    resolved = ROOT.joinpath(*relative.parts).resolve(strict=True)
    resolved.relative_to(ROOT.resolve(strict=True))
    return resolved


def _safe_output(value: Path) -> Path:
    target = value.absolute()
    if target.name in {"", ".", ".."} or not target.parent.is_dir():
        raise ValueError("Fresh-I output requires an existing parent directory")
    if os.path.lexists(target):
        raise FileExistsError(target)
    return target


def _string_tuple(value: object, *, field: str, count: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array")
    result = tuple(str(item).strip() for item in value)
    if len(result) != count or len(set(result)) != count or any(not item for item in result):
        raise ValueError(f"{field} must contain exactly {count} unique IDs")
    return result


def _validate_protocol(
    protocol_path: Path,
) -> tuple[dict[str, object], tuple[str, ...], tuple[str, ...]]:
    protocol = _read_json(protocol_path)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA or (
        protocol.get("status") != PROTOCOL_STATUS
    ):
        raise ValueError("Fresh-I protocol is not the pre-target frozen version")
    if protocol.get("target_semantics") != (
        "tusz_bipolar_edge_time_ictal_involvement_not_soz"
    ):
        raise ValueError("Fresh-I protocol target semantics changed")
    dev = protocol.get("one_shot_official_dev")
    sealed = protocol.get("sealed_official_eval")
    if not isinstance(dev, Mapping) or not isinstance(sealed, Mapping):
        raise TypeError("Fresh-I protocol lacks dev/eval cohort contracts")
    dev_ids = _string_tuple(dev.get("patient_ids"), field="official-dev roster", count=17)
    eval_ids = _string_tuple(
        sealed.get("patient_ids"), field="sealed official-eval roster", count=8
    )
    if set(dev_ids) & set(eval_ids):
        raise ValueError("Fresh-I dev and sealed eval rosters overlap")
    if (
        dev.get("official_split") != "dev"
        or dev.get("patient_count") != 17
        or dev.get("event_count") != 221
        or dev.get("post_target_exclusion_allowed") is not False
        or sealed.get("status") != "remain_unopened_for_this_protocol"
        or sealed.get("patient_count") != 8
        or sealed.get("event_count") != 109
    ):
        raise ValueError("Fresh-I frozen cohort counts/policy changed")
    forbidden = protocol.get("forbidden")
    if not isinstance(forbidden, list) or not {
        "open_official_eval_native_channel_annotations",
        "read_deepsoz_soz_values",
        "read_private_eeg_or_targets",
        "exclude_a_dev_patient_after_target_open",
        "treat_unannotated_cells_as_background",
    } <= set(str(value) for value in forbidden):
        raise ValueError("Fresh-I access firewall is incomplete")
    return protocol, dev_ids, eval_ids


def _validate_prefix_manifest(
    protocol: Mapping[str, object],
    dev_ids: tuple[str, ...],
    eval_ids: tuple[str, ...],
) -> tuple[Path, dict[str, object], list[dict[str, object]]]:
    carrier = protocol.get("transport_carrier")
    if not isinstance(carrier, Mapping):
        raise TypeError("Fresh-I protocol lacks transport carrier")
    root = _repo_path(carrier["prefix_bundle"])
    manifest = _read_json(root / "manifest.json")
    if (
        manifest.get("schema_version") != PREFIX_SCHEMA
        or manifest.get("status")
        != "target_blind_frozen_labram_block9_s1_prefix_ready"
        or manifest.get("full_scope") is not True
        or manifest.get("patient_count") != 25
        or manifest.get("event_count") != 330
        or manifest.get("prefix_tensor_sha256")
        != carrier.get("expected_prefix_tensor_sha256")
        or manifest.get("tensor_file_sha256")
        != carrier.get("expected_prefix_file_sha256")
        or manifest.get("foundation_checkpoint_sha256")
        != carrier.get("foundation_checkpoint_sha256")
        or manifest.get("foundation_modeling_sha256")
        != carrier.get("foundation_modeling_sha256")
        or manifest.get("zero_adapter_official_equivalence_verified") is not True
        or manifest.get("zero_adapter_official_equivalence_max_abs_error") != 0
    ):
        raise ValueError("Fresh-I target-blind prefix contract changed")
    access = manifest.get("access_receipt")
    if not isinstance(access, Mapping) or any(
        access.get(field) is not False
        for field in (
            "completed_s1_labels_loaded",
            "deepsoz_targets_loaded",
            "tusz_channel_time_targets_loaded",
            "private_eeg_loaded",
            "private_targets_loaded",
            "training_performed",
        )
    ):
        raise ValueError("Fresh-I prefix was not target blind")
    tensor_path = root / str(manifest.get("tensor_file", ""))
    if (
        not tensor_path.is_file()
        or tensor_path.is_symlink()
        or _sha256_file(tensor_path) != manifest.get("tensor_file_sha256")
    ):
        raise ValueError("Fresh-I prefix safetensors file changed")
    raw_events = manifest.get("events")
    if not isinstance(raw_events, list):
        raise TypeError("Fresh-I prefix manifest lacks event rows")
    events = [dict(row) for row in raw_events if isinstance(row, Mapping)]
    if len(events) != 330:
        raise ValueError("Fresh-I prefix event rows changed")
    by_split: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    for row in events:
        relative = PurePosixPath(str(row.get("relative_edf_path", "")))
        split = relative.parts[0] if relative.parts else ""
        patient = str(row.get("patient_id", ""))
        by_split[split].add(patient)
        split_counts[split] += 1
    if (
        by_split != {"dev": set(dev_ids), "eval": set(eval_ids)}
        or split_counts != Counter({"dev": 221, "eval": 109})
    ):
        raise ValueError("Fresh-I prefix no longer matches the frozen 17/8 roster")
    return root, manifest, events


def _validate_backbone_files(
    protocol: Mapping[str, object],
    *,
    modeling: Path,
    checkpoint: Path,
) -> tuple[Path, Path]:
    carrier = protocol.get("transport_carrier")
    if not isinstance(carrier, Mapping):
        raise TypeError("Fresh-I protocol lacks transport carrier")
    model_path = modeling.resolve(strict=True)
    checkpoint_path = checkpoint.resolve(strict=True)
    if (
        model_path.is_symlink()
        or checkpoint_path.is_symlink()
        or _sha256_file(model_path) != carrier.get("foundation_modeling_sha256")
        or _sha256_file(checkpoint_path)
        != carrier.get("foundation_checkpoint_sha256")
    ):
        raise ValueError("Fresh-I frozen LaBraM files changed")
    return model_path, checkpoint_path


def _load_formal_lineage(
    protocol: Mapping[str, object],
):
    fit_contract = protocol.get("fit_contract")
    if not isinstance(fit_contract, Mapping):
        raise TypeError("Fresh-I protocol lacks fit contract")
    preprocessing = load_preprocessing_selection_capability(
        _repo_path(fit_contract["preprocessing_selection_bundle"]),
        expected_artifact_sha256=str(
            fit_contract["expected_preprocessing_selection_artifact_sha256"]
        ),
        expected_protocol_receipt_sha256=str(
            fit_contract["expected_preprocessing_protocol_receipt_sha256"]
        ),
    )
    manifest = load_tusz_ictal_training_manifest(
        _repo_path(fit_contract["training_manifest_bundle"])
    )
    corpus = load_formal_token_corpus(
        _repo_path(fit_contract["training_token_corpus"]),
        expected_index_sha256=str(
            fit_contract["expected_training_token_corpus_index_sha256"]
        ),
        preprocessing_selection=preprocessing,
    )
    dev, gate = _load_split(_repo_path(fit_contract["v5_split"]), manifest.patient_ids)
    fit = tuple(sorted(set(manifest.patient_ids) - set(dev) - set(gate)))
    if len(manifest.patient_ids) != 129 or len(fit) != 105:
        raise ValueError("Fresh-I formal-v5 replay expected 129 master/105 fit patients")
    historical = _read_json(_repo_path(fit_contract["formal_v5_result"]))
    if (
        historical.get("schema_version") != "soz_ictal_formal_v5_i_dev_run_v1"
        or tuple(historical.get("fit_patient_ids", ())) != fit
        or tuple(historical.get("i_dev_patient_ids", ())) != dev
        or tuple(historical.get("i_gate_patient_ids_excluded_unopened", ())) != gate
        or historical.get("training_config") != asdict(IctalTrainingConfig())
    ):
        raise ValueError("Fresh-I historical formal-v5 replay contract changed")
    return manifest, corpus, fit, dev, historical


def _build_formal_dataset_from_fit_cache(
    protocol: Mapping[str, object],
    *,
    manifest,
    corpus,
    fit: tuple[str, ...],
    formal_dev: tuple[str, ...],
) -> tuple[IctalTokenBagDataset, object, dict[str, object]]:
    """Build exact formal-v5 bags without reopening EDF or I-gate targets."""

    contract = protocol.get("fit_contract")
    if not isinstance(contract, Mapping):
        raise TypeError("Fresh-I protocol lacks fit contract")
    artifact = load_fit_only_target_artifact_v13(
        _repo_path(contract["fit_plus_i_dev_target_artifact"]),
        expected_manifest_sha256=str(
            contract["expected_fit_plus_i_dev_target_manifest_sha256"]
        ),
        expected_receipt_sha256=str(
            contract["expected_fit_plus_i_dev_target_receipt_sha256"]
        ),
    )
    # ``manifest_sha256`` canonicalizes the full manifest (including thousands
    # of omission rows).  It is immutable here, so compute it once instead of
    # once per event on every epoch.
    master_manifest_sha256 = manifest.manifest_sha256
    allowed = tuple(sorted(set(fit) | set(formal_dev)))
    allowed_set = frozenset(allowed)
    gate = tuple(
        sorted(
            set(manifest.patient_ids)
            - set(fit)
            - set(formal_dev)
        )
    )
    artifact_patients = tuple(str(value) for value in artifact.manifest["fit_patient_ids"])
    if (
        artifact.manifest.get("selection") != "final"
        or artifact_patients != allowed
        or len(artifact_patients) != int(
            contract["fit_plus_i_dev_target_patient_count"]
        )
        or len(artifact.events) != int(contract["fit_plus_i_dev_target_event_count"])
        or contract.get("i_gate_target_bytes_in_artifact") is not False
        or set(gate) & set(artifact_patients)
        or tuple(artifact.manifest["i_gate_patient_ids_excluded_unopened"]) != gate
        or artifact.manifest["i_gate_target_values_materialized"] is not False
        or artifact.manifest["i_gate_outcomes_opened"] is not False
        or artifact.snapshot.training_manifest_sha256 != master_manifest_sha256
        or artifact.snapshot.training_corpus_index_sha256 != corpus.index_sha256
    ):
        raise ValueError("Fresh-I cached target artifact is not exact fit+I-dev only")

    row_by_event = {
        row.event_id: (index, row)
        for index, row in enumerate(artifact.events)
    }
    expected_events = tuple(
        event
        for patient in allowed
        for event in manifest.events_for_patient(patient)
    )
    if len(expected_events) != len(artifact.events) or {
        event.event_id for event in expected_events
    } != set(row_by_event):
        raise ValueError("Fresh-I cached target event roster changed")
    for event in expected_events:
        _, row = row_by_event[event.event_id]
        if (
            row.patient_id != event.patient_id
            or row.event_record_sha256 != event.event_record_sha256
            or row.preprocess_receipt_sha256
            != event.signal_preflight_receipt_sha256
            or row.target_sha256 != event.target_sha256
            or row.target_mask_sha256 != event.target_mask_sha256
        ):
            raise ValueError("Fresh-I cached target/event lineage changed")

    binding_by_event = {
        binding.event_id: binding for binding in corpus.events
    }
    first_event = expected_events[0]
    first_binding = binding_by_event[first_event.event_id]
    first_token = load_labram_concept_tokens(
        first_binding.bundle_path,
        expected_manifest_sha256=first_binding.bundle_manifest_sha256,
    )
    foundation_receipt = first_token.foundation_feature_receipt_sha256
    foundation_checkpoint = first_token.foundation_checkpoint_sha256
    del first_token

    def materialize_patient(patient_id: str) -> IctalTokenPatientBag:
        if patient_id not in allowed_set:
            raise KeyError(f"Patient outside fresh-I formal cache: {patient_id}")
        patient_events = manifest.events_for_patient(patient_id)
        token_events = []
        targets = []
        masks = []
        for event in patient_events:
            binding = binding_by_event[event.event_id]
            token = load_labram_concept_tokens(
                binding.bundle_path,
                expected_manifest_sha256=binding.bundle_manifest_sha256,
            )
            if (
                token.event_id != event.event_id
                or token.source_concept_manifest_sha256 != master_manifest_sha256
                or token.event_record_sha256 != event.event_record_sha256
                or token.preprocess_receipt_sha256
                != event.signal_preflight_receipt_sha256
                or token.foundation_feature_receipt_sha256 != foundation_receipt
                or token.foundation_checkpoint_sha256 != foundation_checkpoint
            ):
                raise ValueError("Fresh-I cached token/event lineage changed")
            row_index, row = row_by_event[event.event_id]
            if row.patient_id != patient_id:
                raise ValueError("Fresh-I cached target patient routing changed")
            token_events.append(token)
            targets.append(artifact.snapshot.training_targets[row_index].clone())
            masks.append(artifact.snapshot.training_target_mask[row_index].clone())
        event_ids = tuple(event.event_id for event in patient_events)
        return IctalTokenPatientBag(
            patient_id=patient_id,
            event_ids=event_ids,
            expected_event_ids=event_ids,
            training_manifest_sha256=master_manifest_sha256,
            expected_event_record_sha256s=tuple(
                event.event_record_sha256 for event in patient_events
            ),
            token_events=tuple(token_events),
            targets=torch.stack(targets).float().contiguous(),
            target_mask=torch.stack(masks).bool().contiguous(),
        )

    # The formal replay uses the same 117 immutable patient bags for 40 epochs.
    # Preloading avoids re-reading 1,307 token bundles while preserving the
    # exact dataset order, tensors, optimizer steps, and gradient calculation.
    preloaded_bags = {
        patient_id: materialize_patient(patient_id) for patient_id in allowed
    }

    def load_patient(patient_id: str) -> IctalTokenPatientBag:
        try:
            return preloaded_bags[patient_id]
        except KeyError as exc:
            raise KeyError(
                f"Patient outside fresh-I formal cache: {patient_id}"
            ) from exc

    dataset = IctalTokenBagDataset(
        allowed,
        load_patient,
        training_manifest_sha256=master_manifest_sha256,
        token_source_manifest_sha256=master_manifest_sha256,
        foundation_feature_receipt_sha256=foundation_receipt,
        formal_token_corpus_verified=True,
        formal_token_corpus_index_sha256=corpus.index_sha256,
        formal_token_corpus_training_bundle_manifest_sha256=(
            corpus.training_bundle_manifest_sha256
        ),
        formal_token_corpus_event_roster_sha256=corpus.event_roster_sha256,
        formal_token_corpus_patient_roster_sha256=corpus.patient_roster_sha256,
        formal_token_corpus_tensor_roster_sha256=corpus.tensor_roster_sha256,
    )
    preload_receipt = {
        "formal_token_bags_preloaded_read_only": True,
        "patient_count": len(preloaded_bags),
        "event_count": sum(len(bag.event_ids) for bag in preloaded_bags.values()),
    }
    return dataset, artifact, preload_receipt


def _safe_tusz_edf(root: Path, relative_value: object, *, expected_split: str) -> Path:
    relative = PurePosixPath(str(relative_value))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 5
        or relative.parts[0] != expected_split
        or relative.suffix != ".edf"
    ):
        raise ValueError("Fresh-I event has an unsafe or wrong-split EDF path")
    source = root.joinpath(*relative.parts).resolve(strict=True)
    if source.relative_to(root).as_posix() != relative.as_posix():
        raise ValueError("Fresh-I EDF path escaped the pinned TUSZ root")
    return source


def _open_dev_targets_once(
    events: Sequence[Mapping[str, object]],
    dev_ids: tuple[str, ...],
    tusz_root: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, object]], Counter[str]]:
    """First function in the command permitted to open new channel targets."""

    selected = [row for row in events if str(row.get("patient_id")) in set(dev_ids)]
    if len(selected) != 221 or any(
        PurePosixPath(str(row.get("relative_edf_path", ""))).parts[0] != "dev"
        for row in selected
    ):
        raise ValueError("Fresh-I target-open roster is not exactly official-dev")
    patient_lookup = {patient: index for index, patient in enumerate(dev_ids)}
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    patient_indices: list[int] = []
    support: dict[str, Counter[str]] = {patient: Counter() for patient in dev_ids}
    state_counts: Counter[str] = Counter()
    receipts: list[dict[str, object]] = []
    for ordinal, row in enumerate(selected):
        patient = str(row["patient_id"])
        event_id = str(row["event_id"])
        match = _EVENT_INDEX_RE.search(event_id)
        if match is None:
            raise ValueError(f"Fresh-I event ID lacks frozen global index: {event_id}")
        edf = _safe_tusz_edf(
            tusz_root,
            row["relative_edf_path"],
            expected_split="dev",
        )
        target = load_tusz_ictal_involvement_target(
            edf.with_suffix(".csv"),
            edf.with_suffix(".csv_bi"),
            event_index=int(match.group("index")),
            source_path=edf,
        )
        if abs(target.event_t0_sec - float(row["global_event_t0_sec"])) > 1e-6:
            raise ValueError(f"Fresh-I event anchor changed: {event_id}")
        positive = target.source_positive_count
        negative = target.source_explicit_negative_count
        observed = positive + negative
        support[patient]["event_count"] += 1
        support[patient]["observed_labels"] += observed
        support[patient]["positive_labels"] += positive
        support[patient]["explicit_negative_labels"] += negative
        state_counts.update(state for edge in target.bin_states for state in edge)
        targets.append(target.targets)
        masks.append(target.source_target_mask)
        patient_indices.append(patient_lookup[patient])
        receipts.append(
            {
                "ordinal": ordinal,
                "event_id": event_id,
                "patient_id": patient,
                "relative_edf_path": str(row["relative_edf_path"]),
                "global_event_index": int(match.group("index")),
                "observed_labels": observed,
                "positive_labels": positive,
                "explicit_negative_labels": negative,
                "channel_annotation_sha256": target.receipt.channel_annotation_sha256,
                "global_annotation_sha256": target.receipt.global_annotation_sha256,
            }
        )
    support_rows = []
    for index, patient in enumerate(dev_ids):
        row = support[patient]
        if row["event_count"] < 1 or row["observed_labels"] < 1:
            raise ValueError("Fresh-I frozen patient lost all evaluable events/cells")
        support_rows.append(
            {
                "patient_index": index,
                "patient_id": patient,
                **dict(row),
                "has_both_classes": bool(
                    row["positive_labels"] and row["explicit_negative_labels"]
                ),
                "prevalence": row["positive_labels"] / row["observed_labels"],
            }
        )
    return (
        torch.stack(targets).float().contiguous(),
        torch.stack(masks).bool().contiguous(),
        torch.tensor(patient_indices, dtype=torch.long),
        support_rows,
        state_counts,
    )


def _chunks(count: int, size: int) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.arange(start, min(start + size, count), dtype=torch.long)
        for start in range(0, count, size)
    )


def _transport_logits(
    *,
    prefix: torch.Tensor,
    independent: IctalInvolvementHead,
    temporal: TemporalResidualIctalInvolvementHead,
    suffix: OfficialLaBraMStaticAdapterSuffix,
    device: torch.device,
    microbatch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    independent.eval()
    temporal.eval()
    suffix.eval()
    independent_rows: list[torch.Tensor] = []
    temporal_rows: list[torch.Tensor] = []
    with torch.inference_mode():
        for indices in _chunks(prefix.shape[0], microbatch):
            batch = prefix.index_select(0, indices).to(device)
            event_count = int(batch.shape[0])
            final = suffix(batch.reshape(event_count * 15, 77, 200))
            tokens = (
                final.reshape(event_count, 15, 19, 4, 200)
                .permute(0, 2, 1, 3, 4)
                .reshape(event_count, 19, 60, 200)
                .contiguous()
            )
            if tuple(tokens.shape[1:]) != (19, 60, 200):
                raise RuntimeError("Fresh-I final LaBraM token shape drifted")
            independent_rows.append(independent(tokens).cpu())
            temporal_rows.append(temporal(tokens).cpu())
    return torch.cat(independent_rows), torch.cat(temporal_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--modeling", type=Path, default=DEFAULT_MODELING)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--event-microbatch", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.event_microbatch < 1:
        raise ValueError("--event-microbatch must be positive")
    protocol_path = args.protocol.resolve(strict=True)
    protocol, dev_ids, eval_ids = _validate_protocol(protocol_path)
    prefix_root, prefix_manifest, prefix_events = _validate_prefix_manifest(
        protocol, dev_ids, eval_ids
    )
    modeling_path, checkpoint_path = _validate_backbone_files(
        protocol,
        modeling=args.modeling,
        checkpoint=args.checkpoint,
    )
    manifest, corpus, fit, formal_dev, historical = _load_formal_lineage(protocol)
    if set(dev_ids) & set(manifest.patient_ids) or set(eval_ids) & set(
        manifest.patient_ids
    ):
        raise ValueError("Fresh-I extension overlaps the formal-v5 129-patient master")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight_passed": True,
                    "new_channel_targets_opened": False,
                    "formal_master_patient_count": len(manifest.patient_ids),
                    "formal_fit_patient_count": len(fit),
                    "one_shot_official_dev_patient_count": len(dev_ids),
                    "one_shot_official_dev_event_count": 221,
                    "sealed_official_eval_patient_count": len(eval_ids),
                    "sealed_official_eval_event_count": 109,
                },
                sort_keys=True,
            )
        )
        return 0

    target = _safe_output(args.output_directory)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    tusz_root = args.tusz_root.resolve(strict=True)
    dataset, fit_target_artifact, preload_receipt = _build_formal_dataset_from_fit_cache(
        protocol,
        manifest=manifest,
        corpus=corpus,
        fit=fit,
        formal_dev=formal_dev,
    )
    config = IctalTrainingConfig()
    fit_dataset = dataset.subset(fit)
    independent, independent_run = _train_head(
        name="independent_second_v4_comparator",
        factory=IctalInvolvementHead,
        fit_dataset=fit_dataset,
        evaluation_dataset=dataset,
        evaluation_patient_ids=formal_dev,
        config=config,
        device=device,
    )
    temporal, temporal_run = _train_head(
        name="temporal_residual_k5",
        factory=TemporalResidualIctalInvolvementHead,
        fit_dataset=fit_dataset,
        evaluation_dataset=dataset,
        evaluation_patient_ids=formal_dev,
        config=config,
        device=device,
    )
    candidate = protocol.get("candidate")
    comparator = protocol.get("matched_comparator")
    if not isinstance(candidate, Mapping) or not isinstance(comparator, Mapping):
        raise TypeError("Fresh-I protocol lacks candidate contracts")
    state_checks = {
        "independent_matches_protocol": independent_run["final_state_sha256"]
        == comparator.get("expected_rebuilt_final_state_sha256"),
        "independent_matches_historical": independent_run["final_state_sha256"]
        == historical["independent_run"]["final_state_sha256"],
        "temporal_matches_protocol": temporal_run["final_state_sha256"]
        == candidate.get("expected_rebuilt_final_state_sha256"),
        "temporal_matches_historical": temporal_run["final_state_sha256"]
        == historical["temporal_run"]["final_state_sha256"],
    }
    if not all(state_checks.values()):
        raise RuntimeError(
            "Fresh-I head replay changed; official-dev channel targets remain unopened"
        )

    training_targets, training_mask, _ = _collect_native_targets(dataset, fit)

    # No official-dev channel annotation has been opened above this line.
    targets, target_mask, patient_index, support_rows, state_counts = (
        _open_dev_targets_once(prefix_events, dev_ids, tusz_root)
    )
    both_class_count = sum(bool(row["has_both_classes"]) for row in support_rows)

    tensor_path = prefix_root / str(prefix_manifest["tensor_file"])
    loaded = load_file(str(tensor_path), device="cpu")
    if set(loaded) != {PREFIX_TENSOR_NAME}:
        raise ValueError("Fresh-I prefix tensor vocabulary changed")
    full_prefix = loaded[PREFIX_TENSOR_NAME].float().contiguous()
    selected_indices = torch.tensor(
        [
            index
            for index, row in enumerate(prefix_events)
            if str(row.get("patient_id")) in set(dev_ids)
        ],
        dtype=torch.long,
    )
    prefix = full_prefix.index_select(0, selected_indices).contiguous()
    del full_prefix, loaded
    if tuple(prefix.shape) != (221, 15, 77, 200):
        raise ValueError("Fresh-I official-dev prefix shape changed")

    suffix = OfficialLaBraMStaticAdapterSuffix(
        modeling_path=modeling_path,
        checkpoint_path=checkpoint_path,
    ).to(device)
    independent_logits, temporal_logits = _transport_logits(
        prefix=prefix,
        independent=independent,
        temporal=temporal,
        suffix=suffix,
        device=device,
        microbatch=args.event_microbatch,
    )
    del suffix, prefix
    if device.type == "cuda":
        torch.cuda.empty_cache()

    independent_metrics = patient_macro_ictal_metrics(
        independent_logits, targets, target_mask, patient_index
    )
    temporal_metrics = patient_macro_ictal_metrics(
        temporal_logits, targets, target_mask, patient_index
    )
    time_logits = v5_shortcut_logits(
        control="time_only",
        training_targets=training_targets,
        training_mask=training_mask,
        evaluation_targets=targets,
        evaluation_mask=target_mask,
    )
    mask_logits = v5_shortcut_logits(
        control="mask_only",
        training_targets=training_targets,
        training_mask=training_mask,
        evaluation_targets=targets,
        evaluation_mask=target_mask,
    )
    time_metrics = patient_macro_ictal_metrics(
        time_logits, targets, target_mask, patient_index
    )
    mask_metrics = patient_macro_ictal_metrics(
        mask_logits, targets, target_mask, patient_index
    )
    prevalence_metrics = prevalence_baseline_metrics(
        training_targets=training_targets,
        training_mask=training_mask,
        evaluation_targets=targets,
        evaluation_mask=target_mask,
        evaluation_patient_ids=patient_index,
    )
    paired = paired_patient_improvements(
        independent_logits,
        temporal_logits,
        targets,
        target_mask,
        patient_index,
    )
    bootstrap = protocol.get("metrics")
    if not isinstance(bootstrap, Mapping):
        raise TypeError("Fresh-I protocol lacks metric contract")
    bce_interval = patient_bootstrap_interval(
        [float(row["bce_improvement"]) for row in paired],
        replicates=int(bootstrap["paired_bootstrap_replicates"]),
        seed=int(bootstrap["paired_bootstrap_seed"]),
    )
    brier_interval = patient_bootstrap_interval(
        [float(row["brier_improvement"]) for row in paired],
        replicates=int(bootstrap["paired_bootstrap_replicates"]),
        seed=int(bootstrap["paired_bootstrap_seed"]) + 1,
    )
    thresholds = protocol.get("qualification_thresholds")
    if not isinstance(thresholds, Mapping):
        raise TypeError("Fresh-I protocol lacks qualification thresholds")
    decision = decide_native_fresh_transport(
        independent=independent_metrics,
        temporal=temporal_metrics,
        time_only=time_metrics,
        mask_only=mask_metrics,
        prevalence=prevalence_metrics,
        paired_bce_interval=bce_interval,
        thresholds=thresholds,
    )
    if both_class_count != temporal_metrics.n_discrimination_patients:
        raise RuntimeError("Fresh-I support and discrimination denominators disagree")

    paired_with_ids = []
    for row in paired:
        index = int(row["patient_index"])
        paired_with_ids.append(dict(row) | {"patient_id": dev_ids[index]})
    event_rows = [
        {
            "event_id": str(row["event_id"]),
            "patient_id": str(row["patient_id"]),
            "relative_edf_path": str(row["relative_edf_path"]),
        }
        for row in prefix_events
        if str(row.get("patient_id")) in set(dev_ids)
    ]
    payload = {
        "schema_version": RESULT_SCHEMA,
        "status": "one_shot_official_dev_native_I_transport_complete",
        "protocol_path": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "target_semantics": "tusz_bipolar_edge_time_ictal_involvement_not_soz",
        "current_soz_reasoner_I_authorized": False,
        "formal_replay": {
            "fit_patient_count": len(fit),
            "i_dev_patient_count": len(formal_dev),
            "state_checks": state_checks,
            "target_cache": {
                "path": str(fit_target_artifact.path),
                "manifest_sha256": fit_target_artifact.manifest_sha256,
                "receipt_sha256": fit_target_artifact.receipt_sha256,
                "patient_count": len(fit_target_artifact.manifest["fit_patient_ids"]),
                "event_count": len(fit_target_artifact.events),
                "i_gate_target_values_materialized": False,
            },
            "token_preload": preload_receipt,
            "independent_run": independent_run,
            "temporal_run": temporal_run,
        },
        "transport_cohort": {
            "official_split": "dev",
            "patient_count": len(dev_ids),
            "event_count": len(event_rows),
            "patient_ids": list(dev_ids),
            "events": event_rows,
            "support_rows": support_rows,
            "both_class_patient_count": both_class_count,
            "bin_state_counts": dict(sorted(state_counts.items())),
        },
        "paired_patient_effects": paired_with_ids,
        "paired_bootstrap": {
            "bce_improvement": bce_interval,
            "brier_improvement": brier_interval,
        },
        "decision": decision,
        "access_receipt": {
            "formal_fit_plus_i_dev_cached_native_targets_loaded": True,
            "formal_token_bags_preloaded_read_only": True,
            "formal_i_gate_target_values_loaded": False,
            "official_dev_channel_time_targets_opened_after_state_match": True,
            "official_eval_signal_prefix_loaded_from_preexisting_target_blind_bundle": True,
            "official_eval_channel_time_targets_opened": False,
            "deepsoz_identity_roster_used_for_prior_exclusion_only": True,
            "deepsoz_soz_values_loaded": False,
            "private_eeg_loaded": False,
            "private_targets_loaded": False,
            "official_dev_used_for_training_or_calibration": False,
            "missing_cells_imputed_as_negative": False,
        },
        "interpretation_limits": [
            "native_I_is_ictal_involvement_not_soz",
            "pass_does_not_reinstate_I_in_the_current_soz_reasoner",
            "result_does_not_measure_or_improve_private_or_deepsoz_top1",
            "official_eval_native_targets_remain_unopened",
        ],
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        save_file(
            {
                "independent_logits": independent_logits.contiguous(),
                "temporal_logits": temporal_logits.contiguous(),
                "time_only_logits": time_logits.contiguous(),
                "mask_only_logits": mask_logits.contiguous(),
                "targets": targets.contiguous(),
                "target_mask": target_mask.contiguous(),
                "patient_index": patient_index.contiguous(),
            },
            str(staging / "evaluation.safetensors"),
        )
        (staging / "result.json").write_bytes(_canonical_bytes(payload))
        os.rename(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    print(
        json.dumps(
            {
                "output": str(target),
                "passed": decision["passed"],
                "qualification": decision["qualification"],
                "current_soz_reasoner_I_authorized": False,
                "both_class_patient_count": both_class_count,
                "checks": decision["checks"],
                "official_eval_channel_time_targets_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
