#!/usr/bin/env python3
"""Generate sealed source-eval exact-anchor and v9 predictions without targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.aggregation import aggregate_patient_logits  # noqa: E402
from src.soz.anchor_constrained_endpoint_reranker import (  # noqa: E402
    AnchorConstrainedEndpointReranker,
    apply_fixed_selective_endpoint_rerank,
    propose_anchor_adjacent_endpoint,
)
from src.soz.anchor_endpoint_features import (  # noqa: E402
    FoldEndpointFeatureState,
    H_FEATURE_SLICE,
    V_FEATURE_SLICE,
    transform_endpoint_features,
)
from src.soz.development_reasoner import (  # noqa: E402
    DevelopmentIVEvidenceBatch,
    physical_node_to_edge_mask,
)
from src.soz.geometry import CHANNEL_INDEX, N_STANDARD_CHANNELS  # noqa: E402
from src.soz.labram_source_eval_prefix import (  # noqa: E402
    load_labram_source_eval_prefix,
)
from src.soz.locked_source_eval_ictal import (  # noqa: E402
    load_locked_source_eval_ictal_artifact,
)
from src.soz.locked_source_eval_predictions import (  # noqa: E402
    publish_locked_source_eval_predictions,
)
from src.soz.locked_source_eval_roster import (  # noqa: E402
    load_locked_source_eval_roster,
)
from src.soz.locked_source_eval_vaq import (  # noqa: E402
    load_locked_source_eval_vaq,
)
from src.soz.temporal_mil_recovery import (  # noqa: E402
    TemporalMILEvidenceReasoner,
)


SCHEMA_VERSION = "soz_locked_labram_v9_source_eval_inference_v1"
PROTOCOL_PATH = (
    ROOT
    / "research/02_method/labram_locked_source_eval_protocol_v10_20260811_zh.md"
)
ROSTER_PATH = ROOT / "outputs/locked_source_eval_roster_v1_20260811"
PREFIX_PATH = ROOT / "outputs/labram_source_eval_prefix_v1_20260811"
ICTAL_PATH = ROOT / "outputs/locked_source_eval_ictal_v1_20260811"
VAQ_PATH = ROOT / "outputs/locked_source_eval_vaq_v1_20260811"
EXACT_CHECKPOINT = (
    ROOT
    / "outputs/labram_temporal_mil_exact_full_source_train_refit_v1_20260811/"
    "final_checkpoint.safetensors"
)
V9_CHECKPOINT = (
    ROOT
    / "outputs/labram_anchor_constrained_endpoint_reranker_oof_v9_final_20260811/"
    "final_checkpoint.safetensors"
)
DEFAULT_OUTPUT = ROOT / "outputs/labram_locked_source_eval_predictions_v10_20260811"

ROSTER_ARTIFACT_SHA256 = (
    "261387a221f3fbbf8a30e47c80e4f7074541075c838ebe891d14a1fb3b621122"
)
SIGNAL_ARTIFACT_SHA256 = (
    "a2fdf45dd122e39ec6e73b3a3edafa1264669875fd2d8cd2b9cb7e8313d1ee66"
)
SIGNAL_RECEIPT_SHA256 = (
    "10128ad30d2163838222d0b4a27d9889a767276a9b697812e3cf568a3d9fd446"
)
PREFIX_MANIFEST_SHA256 = (
    "4885cf6cdfa26811457f3debe2820f269f1cac43d8b269da40ff19865bf624b8"
)
ICTAL_MANIFEST_SHA256 = (
    "634121db0c2c89f312d7d6a17e21b88236bb644084f0369cd93cadda0c6da32d"
)
VAQ_MANIFEST_SHA256 = (
    "c186b171b632120005a64dd55aeeeacf4d69617875cdcbaa15e6199e8d0cb8e5"
)
EXACT_CHECKPOINT_SHA256 = (
    "f44375a2aa509643409f1dfc81cdd5ceb96526c9ffe5cfbba8616681d5e981d6"
)
V9_CHECKPOINT_SHA256 = (
    "b55fd223cc8e235232eaa24ecfc1dfd863f8974222aef6e96cf7ed21c7795cfc"
)

_EXACT_STATE_KEYS = frozenset(
    {
        "channel_prior_logits",
        "evolution_scorer.0.bias",
        "evolution_scorer.0.weight",
        "evolution_scorer.2.weight",
        "ictal_feature_logits",
        "incidence",
        "raw_attention_scale",
        "raw_evolution_gain",
        "raw_ictal_gain",
        "temporal_bias",
    }
)
_V9_STATE_KEYS = frozenset(
    {
        "endpoint_utility.weight",
        "h_center",
        "h_components",
        "feature_mean",
        "feature_scale",
    }
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_state(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_keys: frozenset[str],
) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for locked inference") from exc
    source = path.resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise ValueError("locked checkpoint must be a regular file")
    if _file_sha256(source) != expected_file_sha256:
        raise ValueError(f"locked checkpoint file changed: {source}")
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in load_file(str(source), device="cpu").items()
    }
    if set(state) != set(expected_keys):
        raise ValueError("locked checkpoint tensor keys changed")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("locked checkpoint contains non-finite values")
    return state


def _build_evidence(ictal, vaq_events, vaq_tensors) -> DevelopmentIVEvidenceBatch:
    rows = vaq_events.get("events")
    if not isinstance(rows, list):
        raise ValueError("locked V/AQ event roster is invalid")
    vaq_ids = tuple(str(row["event_id"]) for row in rows)
    vaq_patients = tuple(str(row["patient_id"]) for row in rows)
    ictal_rows = tuple(ictal.events["events"])
    if vaq_ids != ictal.event_ids or vaq_patients != tuple(
        str(row["patient_id"]) for row in ictal_rows
    ):
        raise ValueError("locked I and V/AQ event identity/order differ")
    evolution_mask = vaq_tensors["evolution_mask"].to(torch.bool).contiguous()
    physical_edge = physical_node_to_edge_mask(evolution_mask)
    pooled_mask = ictal.pooled_availability_mask.to(torch.bool).contiguous()
    evidence = DevelopmentIVEvidenceBatch(
        evolution=vaq_tensors["evolution_scaled"].to(torch.float32).contiguous(),
        ictal=ictal.pooled_scores.to(torch.float32).contiguous(),
        evolution_mask=evolution_mask,
        ictal_mask=(pooled_mask & physical_edge).contiguous(),
        phase_mask=vaq_tensors["ictal_phase_mask"].to(torch.bool).contiguous(),
        reliability=vaq_tensors["reliability"].to(torch.float32).contiguous(),
        event_abstain=vaq_tensors["event_abstain"].to(torch.bool).contiguous(),
    )
    return evidence


def _event_patient_index(
    event_patient_ids: Sequence[str], patient_ids: Sequence[str]
) -> torch.Tensor:
    index = {patient_id: ordinal for ordinal, patient_id in enumerate(patient_ids)}
    if len(index) != len(patient_ids):
        raise ValueError("locked patient roster contains duplicates")
    try:
        result = torch.tensor(
            [index[str(patient_id)] for patient_id in event_patient_ids],
            dtype=torch.long,
        )
    except KeyError as exc:
        raise ValueError("an event lies outside the locked patient roster") from exc
    if int(result.min()) != 0 or int(result.max()) != len(patient_ids) - 1:
        raise ValueError("event-to-patient carrier is incomplete")
    return result


def _load_exact_model(state: Mapping[str, torch.Tensor]) -> TemporalMILEvidenceReasoner:
    model = TemporalMILEvidenceReasoner(torch.zeros(N_STANDARD_CHANNELS))
    model.load_state_dict(dict(state), strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


def _load_v9(
    state: Mapping[str, torch.Tensor],
) -> tuple[AnchorConstrainedEndpointReranker, FoldEndpointFeatureState]:
    model = AnchorConstrainedEndpointReranker()
    model.load_state_dict(
        {"endpoint_utility.weight": state["endpoint_utility.weight"]}, strict=True
    )
    model.eval()
    model.requires_grad_(False)
    feature_state = FoldEndpointFeatureState(
        h_center=state["h_center"],
        h_components=state["h_components"],
        feature_mean=state["feature_mean"],
        feature_scale=state["feature_scale"],
    )
    return model, feature_state


def _proposal_with_frozen_patient_gap(
    model: AnchorConstrainedEndpointReranker,
    features: torch.Tensor,
    anchor_scores: torch.Tensor,
    deployment_mask: torch.Tensor,
):
    weight = model.endpoint_utility.weight.detach().squeeze(0)
    h = (features[..., H_FEATURE_SLICE] * weight[H_FEATURE_SLICE]).sum(dim=-1)
    v = (features[..., V_FEATURE_SLICE] * weight[V_FEATURE_SLICE]).sum(dim=-1)
    placeholder_gap = torch.full(
        (anchor_scores.shape[0],), 2.0, dtype=anchor_scores.dtype
    )
    initial = propose_anchor_adjacent_endpoint(
        model,
        features,
        anchor_scores,
        deployment_mask,
        h,
        v,
        placeholder_gap,
    )
    masked = anchor_scores.masked_fill(~deployment_mask, 0.0)
    count = deployment_mask.sum(dim=1).clamp_min(1).to(anchor_scores.dtype)
    mean = masked.sum(dim=1) / count
    variance = (
        ((anchor_scores - mean.unsqueeze(1)).square() * deployment_mask).sum(dim=1)
        / count
    )
    scale = variance.clamp_min(1e-8).sqrt()
    gap = placeholder_gap.clone()
    available = initial.candidate_available
    if bool(available.any()):
        rows = torch.nonzero(available, as_tuple=False).flatten()
        anchor = initial.anchor_index.index_select(0, rows)
        candidate = initial.candidate_index.index_select(0, rows)
        gap[rows] = (
            anchor_scores[rows, anchor] - anchor_scores[rows, candidate]
        ) / scale.index_select(0, rows)
    proposal = propose_anchor_adjacent_endpoint(
        model,
        features,
        anchor_scores,
        deployment_mask,
        h,
        v,
        gap,
    )
    if not torch.equal(initial.anchor_index, proposal.anchor_index) or not torch.equal(
        initial.candidate_index, proposal.candidate_index
    ):
        raise RuntimeError("frozen gap calculation changed an endpoint proposal")
    return proposal, gap


def _load_inputs():
    roster = load_locked_source_eval_roster(
        ROSTER_PATH,
        expected_artifact_sha256=ROSTER_ARTIFACT_SHA256,
        expected_signal_artifact_sha256=SIGNAL_ARTIFACT_SHA256,
        expected_signal_receipt_sha256=SIGNAL_RECEIPT_SHA256,
    )
    prefix = load_labram_source_eval_prefix(
        PREFIX_PATH,
        expected_manifest_sha256=PREFIX_MANIFEST_SHA256,
        require_full_scope=True,
    )
    ictal = load_locked_source_eval_ictal_artifact(
        ICTAL_PATH,
        expected_manifest_sha256=ICTAL_MANIFEST_SHA256,
        expected_roster_artifact_sha256=ROSTER_ARTIFACT_SHA256,
    )
    vaq_manifest, vaq_events, vaq_tensors = load_locked_source_eval_vaq(
        VAQ_PATH,
        expected_manifest_sha256=VAQ_MANIFEST_SHA256,
    )
    event_ids = roster.event_ids
    patient_ids = roster.patient_ids
    vaq_rows = tuple(vaq_events["events"])
    checks = {
        "prefix event order": prefix.event_ids == event_ids,
        "ictal event order": ictal.event_ids == event_ids,
        "vaq event order": tuple(str(row["event_id"]) for row in vaq_rows)
        == event_ids,
        "prefix patient roster": prefix.patient_ids == patient_ids,
        "ictal patient roster": ictal.patient_ids == patient_ids,
        "vaq patient roster": tuple(
            sorted({str(row["patient_id"]) for row in vaq_rows})
        )
        == patient_ids,
        "vaq event-order receipt": vaq_manifest["event_order_sha256"]
        == roster.receipt["event_order_sha256"],
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"locked feature roster join failed: {failed}")
    evidence = _build_evidence(ictal, vaq_events, vaq_tensors)
    event_patients = tuple(str(row["patient_id"]) for row in vaq_rows)
    patient_index = _event_patient_index(event_patients, patient_ids)
    exact_state = _load_state(
        EXACT_CHECKPOINT,
        expected_file_sha256=EXACT_CHECKPOINT_SHA256,
        expected_keys=_EXACT_STATE_KEYS,
    )
    v9_state = _load_state(
        V9_CHECKPOINT,
        expected_file_sha256=V9_CHECKPOINT_SHA256,
        expected_keys=_V9_STATE_KEYS,
    )
    return roster, prefix, evidence, patient_index, exact_state, v9_state


def _run_locked_inference(
    prefix_tokens: torch.Tensor,
    evidence: DevelopmentIVEvidenceBatch,
    event_patient_index: torch.Tensor,
    patient_count: int,
    exact_state: Mapping[str, torch.Tensor],
    v9_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    exact = _load_exact_model(exact_state)
    with torch.inference_mode():
        output = exact(evidence)
        anchor = aggregate_patient_logits(
            output.event_logits, event_patient_index
        ).logits.float().contiguous()
    if tuple(anchor.shape) != (patient_count, N_STANDARD_CHANNELS):
        raise RuntimeError("exact anchor did not return [21,19]")
    deployment_mask = torch.ones_like(anchor, dtype=torch.bool)
    deployment_mask[:, CHANNEL_INDEX["PZ"]] = False
    masked = anchor.masked_fill(~deployment_mask, -torch.inf)
    top_values = masked.max(dim=1).values
    if not bool(((deployment_mask & (anchor == top_values[:, None])).sum(dim=1) == 1).all()):
        raise RuntimeError("exact anchor contains a deployable Top-1 tie")
    anchor_index = masked.argmax(dim=1).to(torch.int64)

    reranker, feature_state = _load_v9(v9_state)
    features = transform_endpoint_features(
        prefix_tokens,
        evidence,
        event_patient_index,
        patient_count,
        feature_state,
    )
    proposal, gap = _proposal_with_frozen_patient_gap(
        reranker, features, anchor, deployment_mask
    )
    reranked = apply_fixed_selective_endpoint_rerank(
        anchor, deployment_mask, proposal
    )
    return {
        "exact_anchor_logits": anchor,
        "v9_logits": reranked.scores.float().contiguous(),
        "deployment_mask": deployment_mask,
        "flip_applied": reranked.applied,
        "anchor_index": anchor_index,
        "candidate_index": proposal.candidate_index.to(torch.int64),
        "pair_margin": proposal.candidate_minus_anchor_logit.float().contiguous(),
        "gap_z": gap.float().contiguous(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roster, prefix, evidence, patient_index, exact_state, v9_state = _load_inputs()
    preflight = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_locked_target_free_source_eval_inference",
        "patient_count": len(roster.patient_ids),
        "event_count": len(roster.event_ids),
        "event_order_sha256": roster.receipt["event_order_sha256"],
        "source_eval_target_values_loaded": False,
        "deepsoz_target_values_loaded": False,
        "tusz_channel_target_values_loaded": False,
        "private_used": False,
        "foundation_trainable_parameter_count": 0,
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True), flush=True)
        return 0
    tensors = _run_locked_inference(
        prefix.tokens,
        evidence,
        patient_index,
        len(roster.patient_ids),
        exact_state,
        v9_state,
    )
    runner_path = Path(__file__).resolve(strict=True)
    lineage = {
        "protocol_document_sha256": _file_sha256(PROTOCOL_PATH.resolve(strict=True)),
        "exact_anchor_checkpoint_sha256": EXACT_CHECKPOINT_SHA256,
        "v9_reranker_checkpoint_sha256": V9_CHECKPOINT_SHA256,
        "source_eval_prefix_manifest_sha256": PREFIX_MANIFEST_SHA256,
        "source_eval_ictal_manifest_sha256": ICTAL_MANIFEST_SHA256,
        "source_eval_vaq_manifest_sha256": VAQ_MANIFEST_SHA256,
        "producer_source_sha256": _file_sha256(runner_path),
    }
    artifact = publish_locked_source_eval_predictions(
        args.output_directory,
        roster=roster,
        tensors=tensors,
        lineage=lineage,
    )
    result = {
        **preflight,
        "status": "published_locked_target_free_source_eval_predictions",
        "path": str(artifact.path),
        "manifest_sha256": artifact.manifest_sha256,
        "flip_count": int(artifact.flip_applied.sum().item()),
        "exact_anchor_top1_indices": (
            artifact.exact_anchor_logits.masked_fill(
                ~artifact.deployment_mask, -torch.inf
            )
            .argmax(dim=1)
            .tolist()
        ),
        "v9_top1_indices": (
            artifact.v9_logits.masked_fill(~artifact.deployment_mask, -torch.inf)
            .argmax(dim=1)
            .tolist()
        ),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
