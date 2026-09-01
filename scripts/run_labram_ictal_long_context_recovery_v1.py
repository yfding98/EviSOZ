#!/usr/bin/env python3
"""Run the single development-only LaBraM k31 ictal recovery candidate.

The command reuses the frozen formal-v4 LaBraM token corpus and the already
opened I-dev patients.  I-gate patients are excluded from fitting and never
passed to an evaluation loader.  A passing result is development
qualification only and cannot authorize an evidence cache or SOZ reasoner.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Sequence


_REQUIRED_CUBLAS_WORKSPACE = ":4096:8"
_observed_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _observed_workspace is None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _REQUIRED_CUBLAS_WORKSPACE
elif _observed_workspace != _REQUIRED_CUBLAS_WORKSPACE:
    raise RuntimeError("LaBraM recovery requires CUBLAS_WORKSPACE_CONFIG=':4096:8'")

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_tusz_ictal_token_cache import (  # noqa: E402
    load_formal_token_corpus,
)
from scripts.run_ictal_v5_dev import (  # noqa: E402
    _collect_native_targets,
    _load_split,
    _train_head,
)
from src.soz.cached_concept_training import IctalTokenBagDataset  # noqa: E402
from src.soz.concept_metrics import (  # noqa: E402
    IctalConceptMetrics,
    patient_macro_ictal_metrics,
)
from src.soz.concept_run import IctalTrainingConfig  # noqa: E402
from src.soz.data.tusz_training import (  # noqa: E402
    load_tusz_ictal_training_manifest,
)
from src.soz.ictal_recovery import (  # noqa: E402
    LABRAM_LONG_CONTEXT_HEAD,
    decide_labram_long_context_development,
)
from src.soz.ictal_v5 import (  # noqa: E402
    prevalence_baseline_metrics,
    v5_shortcut_logits,
)
from src.soz.models.concept_heads import (  # noqa: E402
    IctalInvolvementHead,
    LongContextTemporalResidualIctalInvolvementHead,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    load_preprocessing_selection_capability,
)
from src.soz.tusz_token_dataset import (  # noqa: E402
    build_tusz_ictal_token_bag_dataset,
)


def _sha(value: str) -> str:
    normalized = str(value).strip()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256")
    return normalized


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest-bundle", type=Path, required=True)
    parser.add_argument("--training-token-corpus", type=Path, required=True)
    parser.add_argument(
        "--expected-training-token-corpus-index-sha256", type=_sha, required=True
    )
    parser.add_argument("--preprocessing-selection-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-preprocessing-selection-artifact-sha256", type=_sha, required=True
    )
    parser.add_argument(
        "--expected-preprocessing-protocol-receipt-sha256", type=_sha, required=True
    )
    parser.add_argument("--v5-split", type=Path, required=True)
    parser.add_argument("--formal-v5-result", type=Path, required=True)
    parser.add_argument("--edf-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _safe_new_output(value: Path) -> Path:
    target = Path(os.path.abspath(value))
    if target.name in {"", ".", ".."} or not target.parent.is_dir():
        raise ValueError("Recovery output requires a concrete path with an existing parent")
    if os.path.lexists(target):
        raise FileExistsError(f"Recovery output already exists: {target}")
    return target


def _load_formal_v5_result(
    path: Path,
    *,
    expected_dev: tuple[str, ...],
    expected_gate: tuple[str, ...],
) -> tuple[dict[str, object], str]:
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "soz_ictal_formal_v5_i_dev_run_v1":
        raise ValueError("Recovery must bind the formal-v5 I-dev result")
    if payload.get("decision", {}).get("passed") is not False:
        raise ValueError("Recovery requires the preserved negative formal-v5 result")
    if tuple(sorted(payload.get("i_dev_patient_ids", ()))) != expected_dev:
        raise ValueError("Formal-v5 I-dev roster differs from the frozen split")
    if tuple(sorted(payload.get("i_gate_patient_ids_excluded_unopened", ()))) != expected_gate:
        raise ValueError("Formal-v5 I-gate roster differs from the frozen split")
    if (
        payload.get("i_gate_outcomes_opened") is not False
        or payload.get("deepsoz_soz_labels_used") is not False
        or payload.get("private_labels_used") is not False
        or payload.get("missing_tusz_cells_imputed_as_negative") is not False
    ):
        raise ValueError("Formal-v5 result violates recovery data isolation")
    return payload, _file_sha256(resolved)


def _save_state(path: Path, model: torch.nn.Module) -> None:
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(state, str(path))


def _memoized_subset(
    dataset: IctalTokenBagDataset,
    patient_ids: tuple[str, ...],
) -> IctalTokenBagDataset:
    """Load an allowed subset once so repeated epochs do not reread 1.4 GB."""

    source = dataset.subset(patient_ids)
    bags = {bag.patient_id: bag for bag in source.iter_epoch()}
    if tuple(sorted(bags)) != source.patient_ids:
        raise RuntimeError("Memoized ictal subset changed its patient roster")
    return IctalTokenBagDataset(
        source.patient_ids,
        bags.__getitem__,
        training_manifest_sha256=source.training_manifest_sha256,
        token_source_manifest_sha256=source.token_source_manifest_sha256,
        foundation_feature_receipt_sha256=(
            source.foundation_feature_receipt_sha256
        ),
        formal_token_corpus_verified=source.formal_token_corpus_verified,
        formal_token_corpus_index_sha256=(
            source.formal_token_corpus_index_sha256
        ),
        formal_token_corpus_training_bundle_manifest_sha256=(
            source.formal_token_corpus_training_bundle_manifest_sha256
        ),
        formal_token_corpus_event_roster_sha256=(
            source.formal_token_corpus_event_roster_sha256
        ),
        formal_token_corpus_patient_roster_sha256=(
            source.formal_token_corpus_patient_roster_sha256
        ),
        formal_token_corpus_tensor_roster_sha256=(
            source.formal_token_corpus_tensor_roster_sha256
        ),
        training_authorized=source.training_authorized,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = _safe_new_output(args.output_directory)
    device = torch.device(args.device)
    if not args.preflight_only and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")

    preprocessing = load_preprocessing_selection_capability(
        args.preprocessing_selection_bundle,
        expected_artifact_sha256=(
            args.expected_preprocessing_selection_artifact_sha256
        ),
        expected_protocol_receipt_sha256=(
            args.expected_preprocessing_protocol_receipt_sha256
        ),
    )
    manifest = load_tusz_ictal_training_manifest(args.training_manifest_bundle)
    corpus = load_formal_token_corpus(
        args.training_token_corpus,
        expected_index_sha256=args.expected_training_token_corpus_index_sha256,
        preprocessing_selection=preprocessing,
    )
    dataset = build_tusz_ictal_token_bag_dataset(manifest, args.edf_root, corpus)
    dev, gate = _load_split(args.v5_split, dataset.patient_ids)
    fit = tuple(sorted(set(dataset.patient_ids) - set(dev) - set(gate)))
    if len(dataset.patient_ids) != 129 or len(fit) != 105:
        raise ValueError("Recovery expected 129 master and 105 fit patients")
    formal_v5, formal_v5_sha256 = _load_formal_v5_result(
        args.formal_v5_result,
        expected_dev=dev,
        expected_gate=gate,
    )

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight_passed": True,
                    "training_started": False,
                    "candidate": LABRAM_LONG_CONTEXT_HEAD,
                    "master_patient_count": len(dataset.patient_ids),
                    "fit_patient_count": len(fit),
                    "i_dev_patient_count": len(dev),
                    "i_gate_patient_count": len(gate),
                    "i_gate_outcomes_opened": False,
                    "formal_v5_result_sha256": formal_v5_sha256,
                },
                sort_keys=True,
            )
        )
        return 0

    fit_dataset = _memoized_subset(dataset, fit)
    dev_dataset = _memoized_subset(dataset, dev)
    config = IctalTrainingConfig()
    independent_head, independent_run = _train_head(
        name="independent_second_reproduction",
        factory=IctalInvolvementHead,
        fit_dataset=fit_dataset,
        evaluation_dataset=dev_dataset,
        evaluation_patient_ids=dev,
        config=config,
        device=device,
    )
    if independent_run != formal_v5["independent_run"]:
        raise RuntimeError(
            "Independent comparator did not exactly reproduce formal-v5; "
            "refusing to interpret the recovery candidate"
        )
    long_context_head, long_context_run = _train_head(
        name=LABRAM_LONG_CONTEXT_HEAD,
        factory=LongContextTemporalResidualIctalInvolvementHead,
        fit_dataset=fit_dataset,
        evaluation_dataset=dev_dataset,
        evaluation_patient_ids=dev,
        config=config,
        device=device,
    )

    training_targets, training_mask, _ = _collect_native_targets(fit_dataset, fit)
    dev_targets, dev_mask, dev_patient_ids = _collect_native_targets(dev_dataset, dev)
    time_metrics = patient_macro_ictal_metrics(
        v5_shortcut_logits(
            control="time_only",
            training_targets=training_targets,
            training_mask=training_mask,
            evaluation_targets=dev_targets,
            evaluation_mask=dev_mask,
        ),
        dev_targets,
        dev_mask,
        dev_patient_ids,
    )
    mask_metrics = patient_macro_ictal_metrics(
        v5_shortcut_logits(
            control="mask_only",
            training_targets=training_targets,
            training_mask=training_mask,
            evaluation_targets=dev_targets,
            evaluation_mask=dev_mask,
        ),
        dev_targets,
        dev_mask,
        dev_patient_ids,
    )
    prevalence_metrics = prevalence_baseline_metrics(
        training_targets=training_targets,
        training_mask=training_mask,
        evaluation_targets=dev_targets,
        evaluation_mask=dev_mask,
        evaluation_patient_ids=dev_patient_ids,
    )
    decision = decide_labram_long_context_development(
        independent_metrics=IctalConceptMetrics(**independent_run["metrics"]),
        long_context_metrics=IctalConceptMetrics(**long_context_run["metrics"]),
        time_only_metrics=time_metrics,
        mask_only_metrics=mask_metrics,
        prevalence_metrics=prevalence_metrics,
    )
    payload = {
        "schema_version": "soz_labram_ictal_long_context_recovery_run_v1",
        "candidate": LABRAM_LONG_CONTEXT_HEAD,
        "context_seconds": (
            LongContextTemporalResidualIctalInvolvementHead.context_seconds
        ),
        "context_direction": "symmetric_retrospective_not_causal_onset",
        "target_semantics": "tusz_bipolar_edge_time_involvement_not_soz",
        "development_only": True,
        "formal_promotion": False,
        "formal_v5_negative_preserved": True,
        "formal_v5_result_sha256": formal_v5_sha256,
        "deepsoz_soz_labels_used": False,
        "private_labels_used": False,
        "missing_tusz_cells_imputed_as_negative": False,
        "i_gate_outcomes_opened": False,
        "checkpoint_authorized_for_evidence_or_reasoner": False,
        "master_patient_count": len(dataset.patient_ids),
        "fit_patient_ids": list(fit),
        "i_dev_patient_ids": list(dev),
        "i_gate_patient_ids_excluded_unopened": list(gate),
        "training_config": asdict(config),
        "parameter_counts": {
            "independent": sum(parameter.numel() for parameter in independent_head.parameters()),
            "long_context": sum(parameter.numel() for parameter in long_context_head.parameters()),
        },
        "independent_run": independent_run,
        "long_context_run": long_context_run,
        "decision": decision,
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        _save_state(staging / "independent_head.safetensors", independent_head)
        _save_state(staging / "long_context_head.safetensors", long_context_head)
        (staging / "development_result.json").write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
            encoding="utf-8",
        )
        os.rename(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    print(
        json.dumps(
            {
                "path": str(target),
                "candidate": LABRAM_LONG_CONTEXT_HEAD,
                "development_qualified": decision["development_qualified"],
                "formal_promotion": False,
                "checks": decision["checks"],
                "i_gate_outcomes_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
