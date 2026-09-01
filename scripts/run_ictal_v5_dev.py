#!/usr/bin/env python3
"""Run the one allowed formal-v5 I-dev head comparison.

The command trains the unchanged independent-second head and the single
residual depthwise-temporal candidate from fresh initialization.  It opens
only I-dev outcomes.  I-gate patients are excluded from fitting and are never
passed to an evaluation loader in this command.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Callable, Sequence


_REQUIRED_CUBLAS_WORKSPACE = ":4096:8"
observed_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if observed_workspace is None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _REQUIRED_CUBLAS_WORKSPACE
elif observed_workspace != _REQUIRED_CUBLAS_WORKSPACE:
    raise RuntimeError("Ictal v5 requires CUBLAS_WORKSPACE_CONFIG=':4096:8'")

import torch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_tusz_ictal_token_cache import (  # noqa: E402
    load_formal_token_corpus,
)
from src.soz.cached_concept_training import (  # noqa: E402
    IctalTokenBagDataset,
    evaluate_cached_ictal_patients,
    train_cached_ictal_epoch,
)
from src.soz.concept_metrics import patient_macro_ictal_metrics  # noqa: E402
from src.soz.concept_run import (  # noqa: E402
    IctalTrainingConfig,
    ictal_determinism_runtime,
    ictal_head_state_sha256,
    validate_ictal_cuda_environment,
)
from src.soz.data.tusz_training import (  # noqa: E402
    load_tusz_ictal_training_manifest,
)
from src.soz.ictal_v5 import (  # noqa: E402
    decide_v5_i_dev,
    prevalence_baseline_metrics,
    v5_shortcut_logits,
)
from src.soz.models.concept_heads import (  # noqa: E402
    IctalInvolvementHead,
    TemporalResidualIctalInvolvementHead,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    load_preprocessing_selection_capability,
)
from src.soz.tusz_token_dataset import (  # noqa: E402
    build_tusz_ictal_token_bag_dataset,
)


def _sha(value: str) -> str:
    normalized = str(value).strip()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256")
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest-bundle", type=Path, required=True)
    parser.add_argument("--training-token-corpus", type=Path, required=True)
    parser.add_argument("--expected-training-token-corpus-index-sha256", type=_sha, required=True)
    parser.add_argument("--preprocessing-selection-bundle", type=Path, required=True)
    parser.add_argument("--expected-preprocessing-selection-artifact-sha256", type=_sha, required=True)
    parser.add_argument("--expected-preprocessing-protocol-receipt-sha256", type=_sha, required=True)
    parser.add_argument("--v5-split", type=Path, required=True)
    parser.add_argument("--edf-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _safe_new_output(value: Path) -> Path:
    target = Path(os.path.abspath(value))
    if target.name in {"", ".", ".."} or not target.parent.is_dir():
        raise ValueError("V5 I-dev output requires an existing parent directory")
    if os.path.lexists(target):
        raise FileExistsError(f"V5 I-dev output already exists: {target}")
    return target


def _load_split(path: Path, master_patients: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "soz_ictal_formal_v5_auxiliary_split_v1":
        raise ValueError("Unsupported ictal v5 split")
    if (
        payload.get("deepsoz_soz_labels_used") is not False
        or payload.get("private_labels_used") is not False
        or payload.get("missing_tusz_cells_imputed_as_negative") is not False
    ):
        raise ValueError("V5 split violates source-native label isolation")
    dev = tuple(sorted(str(value).strip() for value in payload["i_dev_patient_ids"]))
    gate = tuple(sorted(str(value).strip() for value in payload["i_gate_patient_ids"]))
    if len(dev) != 12 or len(gate) != 12 or set(dev) & set(gate):
        raise ValueError("V5 split must contain disjoint 12/12 patient groups")
    if not (set(dev) | set(gate)) <= set(master_patients):
        raise ValueError("V5 split patients are absent from the master corpus")
    return dev, gate


def _collect_native_targets(
    dataset: IctalTokenBagDataset,
    patient_ids: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = []
    masks = []
    patient_index = []
    for index, bag in enumerate(dataset.iter_subset(patient_ids)):
        targets.append(bag.targets.detach().cpu().to(torch.float32))
        masks.append(bag.target_mask.detach().cpu().to(torch.bool))
        patient_index.append(
            torch.full((len(bag.event_ids),), index, dtype=torch.long)
        )
    return (
        torch.cat(targets, dim=0).contiguous(),
        torch.cat(masks, dim=0).contiguous(),
        torch.cat(patient_index, dim=0).contiguous(),
    )


def _train_head(
    *,
    name: str,
    factory: Callable[[], IctalInvolvementHead],
    fit_dataset: IctalTokenBagDataset,
    evaluation_dataset: IctalTokenBagDataset,
    evaluation_patient_ids: tuple[str, ...],
    config: IctalTrainingConfig,
    device: torch.device,
) -> tuple[IctalInvolvementHead, dict[str, object]]:
    cuda_devices = []
    if device.type == "cuda":
        validate_ictal_cuda_environment()
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with ictal_determinism_runtime(config, execution_device_type=device.type):
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(config.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(config.seed)
            head = factory().to(device)
            initial_state = ictal_head_state_sha256(head)
            optimizer = torch.optim.AdamW(
                head.parameters(),
                lr=float(config.learning_rate),
                weight_decay=float(config.weight_decay),
            )
            epoch_rows = []
            for epoch in range(config.fixed_epochs):
                order = list(fit_dataset.patient_ids)
                random.Random(config.seed + epoch).shuffle(order)
                output = train_cached_ictal_epoch(
                    head,
                    fit_dataset,
                    optimizer,
                    patient_order=tuple(order),
                    max_grad_norm=config.max_grad_norm,
                    event_microbatch_size=config.event_microbatch_size,
                )
                epoch_rows.append(asdict(output))
                print(
                    json.dumps(
                        {
                            "stage": "i_dev_training",
                            "head": name,
                            "epoch": epoch + 1,
                            "epochs": config.fixed_epochs,
                            "mean_patient_loss": output.mean_patient_loss,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            final_state = ictal_head_state_sha256(head)
            evaluation_epoch, metrics = evaluate_cached_ictal_patients(
                head,
                evaluation_dataset,
                evaluation_patient_ids,
                event_microbatch_size=config.event_microbatch_size,
            )
    return head, {
        "initial_state_sha256": initial_state,
        "final_state_sha256": final_state,
        "epoch_rows": epoch_rows,
        "evaluation_epoch": asdict(evaluation_epoch),
        "metrics": asdict(metrics),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = _safe_new_output(args.output_directory)
    device = torch.device(args.device)
    if (
        not args.preflight_only
        and device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA was requested but unavailable")
    preprocessing = load_preprocessing_selection_capability(
        args.preprocessing_selection_bundle,
        expected_artifact_sha256=args.expected_preprocessing_selection_artifact_sha256,
        expected_protocol_receipt_sha256=args.expected_preprocessing_protocol_receipt_sha256,
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
        raise ValueError("V5 I-dev expected 129 master and 105 fit patients")
    fit_dataset = dataset.subset(fit)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight_passed": True,
                    "training_started": False,
                    "master_patient_count": len(dataset.patient_ids),
                    "fit_patient_count": len(fit),
                    "i_dev_patient_count": len(dev),
                    "i_gate_patient_count": len(gate),
                    "i_gate_outcomes_opened": False,
                },
                sort_keys=True,
            )
        )
        return 0
    config = IctalTrainingConfig()
    independent_head, independent_run = _train_head(
        name="independent_second_v4_comparator",
        factory=IctalInvolvementHead,
        fit_dataset=fit_dataset,
        evaluation_dataset=dataset,
        evaluation_patient_ids=dev,
        config=config,
        device=device,
    )
    del independent_head
    if device.type == "cuda":
        torch.cuda.empty_cache()
    temporal_head, temporal_run = _train_head(
        name="temporal_residual_k5",
        factory=TemporalResidualIctalInvolvementHead,
        fit_dataset=fit_dataset,
        evaluation_dataset=dataset,
        evaluation_patient_ids=dev,
        config=config,
        device=device,
    )
    del temporal_head
    if device.type == "cuda":
        torch.cuda.empty_cache()

    training_targets, training_mask, _ = _collect_native_targets(dataset, fit)
    dev_targets, dev_mask, dev_patient_ids = _collect_native_targets(dataset, dev)
    time_logits = v5_shortcut_logits(
        control="time_only",
        training_targets=training_targets,
        training_mask=training_mask,
        evaluation_targets=dev_targets,
        evaluation_mask=dev_mask,
    )
    mask_logits = v5_shortcut_logits(
        control="mask_only",
        training_targets=training_targets,
        training_mask=training_mask,
        evaluation_targets=dev_targets,
        evaluation_mask=dev_mask,
    )
    time_metrics = patient_macro_ictal_metrics(
        time_logits, dev_targets, dev_mask, dev_patient_ids
    )
    mask_metrics = patient_macro_ictal_metrics(
        mask_logits, dev_targets, dev_mask, dev_patient_ids
    )
    prevalence_metrics = prevalence_baseline_metrics(
        training_targets=training_targets,
        training_mask=training_mask,
        evaluation_targets=dev_targets,
        evaluation_mask=dev_mask,
        evaluation_patient_ids=dev_patient_ids,
    )
    from src.soz.concept_metrics import IctalConceptMetrics

    independent_metrics = IctalConceptMetrics(**independent_run["metrics"])
    temporal_metrics = IctalConceptMetrics(**temporal_run["metrics"])
    decision = decide_v5_i_dev(
        independent_metrics=independent_metrics,
        temporal_metrics=temporal_metrics,
        time_only_metrics=time_metrics,
        mask_only_metrics=mask_metrics,
        prevalence_metrics=prevalence_metrics,
    )
    payload = {
        "schema_version": "soz_ictal_formal_v5_i_dev_run_v1",
        "target_semantics": "tusz_bipolar_edge_time_involvement_not_soz",
        "deepsoz_soz_labels_used": False,
        "private_labels_used": False,
        "missing_tusz_cells_imputed_as_negative": False,
        "i_gate_outcomes_opened": False,
        "master_patient_count": len(dataset.patient_ids),
        "fit_patient_ids": list(fit),
        "i_dev_patient_ids": list(dev),
        "i_gate_patient_ids_excluded_unopened": list(gate),
        "training_config": asdict(config),
        "independent_run": independent_run,
        "temporal_run": temporal_run,
        "decision": decision,
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / "i_dev_result.json").write_text(
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
                "passed": decision["passed"],
                "selected_head": decision["selected_head"],
                "checks": decision["checks"],
                "i_gate_outcomes_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
