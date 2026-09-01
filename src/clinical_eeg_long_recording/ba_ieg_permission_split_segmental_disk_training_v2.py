"""Additive disk/composite training bridge for the BA-IEG v2 candidate.

This module leaves the frozen v1 disk format, trainer, model and configuration
untouched.  It joins one registered v1 disk batch to prediction-first A1
stable-origin/support lineage, runs the stable-clock single-bout v2 model,
orders a final-left-closure receipt before the earliest-prefix/K3 gate, and
connects the signal-only v2 typed head.

Three objectives retain disjoint gradient authority:

* ``L_segmental`` -> causal/offline segmental core only;
* ``L_typed_boundary_MIL`` -> typed-signal projection and boundary adapter;
* ``L_patient_positive_set`` -> detached identity/rank adapter only.

The real A1 post-freeze training authority is not yet available.  Therefore
the executable route below requires an explicitly non-promotable synthetic
software-fixture authority and fails closed otherwise.  It proves a software
training seam, not a trained model, G0/G1/G2 admission, SOZ accuracy or a
clinical claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence

import torch
from torch import nn

from .ba_ieg_a1_complete_patient_positive_set_bridge_v2 import (
    BAIEGA1CompletePatientCappedLogMeanExpV2,
    BAIEGA1CompletePatientRecordRosterV2,
    BAIEGA1SyntheticTrainingAuthorityV2,
    require_real_a1_postfreeze_training_authority_v2,
)
from .ba_ieg_capped_log_mean_exp_event_bag_v1 import (
    BAIEGCappedLogMeanExpEventBag,
    BAIEGRecordEventBagManifest,
)
from .ba_ieg_complete_patient_positive_set_bridge_v1 import (
    BAIEGBoundDeepSOZPositiveSetV1,
    BAIEGCompletePatientAggregationOutputV1,
    BAIEGCompletePatientPositiveSetLossOutputV1,
    BAIEGPhysicalRecordEvidenceBatchV1,
    complete_patient_positive_set_mass_loss_v1,
)
from .ba_ieg_earliest_prefix_k3_gate_v1 import (
    BAIEGEarliestPrefixK3DevelopmentPolicyV1,
    BAIEGEarliestPrefixK3GateResultV1,
    build_ba_ieg_earliest_prefix_k3_gate_v1,
)
from .ba_ieg_g0_a1_acquisition_support_lineage_v1 import (
    validate_ba_ieg_g0_a1_acquisition_support_lineage_v1,
)
from .ba_ieg_g0_a1_candidate_roster_v1 import (
    validate_ba_ieg_g0_a1_prediction_roster_v1,
)
from .ba_ieg_g0_support_relative_shortcut_surface_v1 import (
    BA_IEG_G0_SUPPORT_RELATIVE_TIME_FEATURE_NAMES_V1,
    BAIEGG0StableOriginRegistryV1,
    BAIEGG0SupportRelativeTimeSurfaceV1,
    _clean_intervals,
    _gap_overlap,
    _signed_opportunity_displacement,
    build_ba_ieg_g0_stable_origin_registry_v1,
    build_ba_ieg_g0_support_relative_time_surface_v1,
)
from .ba_ieg_permission_split_segmental_disk_training_v1 import (
    BAIEGSegmentalDiskBatchV1,
    BAIEGSegmentalDiskDatasetV1,
)
from .ba_ieg_permission_split_segmental_state_model_v2 import (
    BAIEGPermissionSplitSegmentalStateModelV2,
    BAIEGPermissionSplitSegmentalStateOutputV2,
)
from .ba_ieg_permission_split_segmental_supervision_v1 import (
    BAIEGPermissionSplitSegmentalLossOutputV1,
    permission_split_segmental_training_loss_v1,
    project_ba_ieg_segmental_target_to_frozen_lattice_v1,
)
from .ba_ieg_shallow_causal_typed_unit_head_v1 import (
    BAIEGShallowCausalTypedUnitHeadOutput,
)
from .ba_ieg_shallow_causal_typed_unit_head_v2 import (
    BAIEGShallowCausalTypedUnitOnsetHeadV2,
)
from .ba_ieg_shallow_causal_typed_unit_supervision_v2 import (
    BAIEGShallowCausalTypedUnitMILBoundaryLossOutputV2,
    build_ba_ieg_shallow_causal_typed_unit_mil_target_bundle_v2,
    shallow_causal_typed_unit_mil_boundary_loss_v2,
)


BA_IEG_V2_DISK_COMPOSITE_ID: Final[str] = (
    "ba_ieg_a1_stable_clock_single_bout_k3_three_loss_disk_composite_v2"
)
BA_IEG_V2_FINAL_LEFT_CLOSURE_SCHEMA: Final[str] = (
    "ba_ieg_a1_final_left_support_closure_before_k3_v2"
)
BA_IEG_V2_SOURCE_DEV_STABLE_ORIGIN_SCHEMA: Final[str] = (
    "ba_ieg_source_dev_calibration_only_stable_origin_registry_v2"
)

_SHA256_ALPHABET = frozenset("0123456789abcdef")
_TOLERANCE = 1e-8


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value).difference(_SHA256_ALPHABET)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    return value


def _canonical_intervals(value: torch.Tensor, mask: torch.Tensor) -> tuple[tuple[float, float], ...]:
    rows = [
        (float(row[0]), float(row[1]))
        for row in value.detach().cpu()[mask.detach().cpu()]
    ]
    return tuple(rows)


def _same_intervals(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> bool:
    return len(left) == len(right) and all(
        abs(float(a[0]) - float(b[0])) <= _TOLERANCE
        and abs(float(a[1]) - float(b[1])) <= _TOLERANCE
        for a, b in zip(left, right)
    )


def _detach_dataclass_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_dataclass_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_dataclass_tree(item) for item in value]
    if is_dataclass(value):
        payload = {
            item.name: _detach_dataclass_tree(getattr(value, item.name))
            for item in fields(value)
            if item.init
        }
        return type(value)(**payload)
    return value


@dataclass(frozen=True)
class BAIEGV2FinalLeftClosureReceipt:
    source_input_batch_sha256: str
    source_context_receipt_sha256: str
    source_stable_origin_registry_receipt_sha256: str
    source_acquisition_support_lineage_receipt_sha256: str
    event_ids: tuple[str, ...]
    final_left_edge_recording_seconds: tuple[float, ...]
    final_support_union_sha256s: tuple[str, ...]
    schema_version: str = BA_IEG_V2_FINAL_LEFT_CLOSURE_SCHEMA
    closure_precedes_earliest_prefix_k3: bool = True
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_input_batch_sha256",
            "source_context_receipt_sha256",
            "source_stable_origin_registry_receipt_sha256",
            "source_acquisition_support_lineage_receipt_sha256",
        ):
            _sha256(getattr(self, name), name)
        events = tuple(_identifier(value, "final-left event ID") for value in self.event_ids)
        if not events or len(set(events)) != len(events):
            raise ValueError("final-left closure needs unique events")
        if len(self.final_left_edge_recording_seconds) != len(events) or len(
            self.final_support_union_sha256s
        ) != len(events):
            raise ValueError("final-left closure rows do not align")
        if not all(math.isfinite(float(value)) for value in self.final_left_edge_recording_seconds):
            raise ValueError("final-left closure edge is not finite")
        for digest in self.final_support_union_sha256s:
            _sha256(digest, "final support union")
        if self.schema_version != BA_IEG_V2_FINAL_LEFT_CLOSURE_SCHEMA or self.closure_precedes_earliest_prefix_k3 is not True:
            raise ValueError("final-left closure schema/order drifted")
        object.__setattr__(self, "event_ids", events)
        object.__setattr__(
            self,
            "receipt_sha256",
            _canonical_sha256(
                {
                    "schema_version": self.schema_version,
                    "source_input_batch_sha256": self.source_input_batch_sha256,
                    "source_context_receipt_sha256": self.source_context_receipt_sha256,
                    "source_stable_origin_registry_receipt_sha256": self.source_stable_origin_registry_receipt_sha256,
                    "source_acquisition_support_lineage_receipt_sha256": self.source_acquisition_support_lineage_receipt_sha256,
                    "event_ids": list(events),
                    "final_left_edge_recording_seconds": list(self.final_left_edge_recording_seconds),
                    "final_support_union_sha256s": list(self.final_support_union_sha256s),
                    "closure_precedes_earliest_prefix_k3": True,
                    "later_right_support_may_not_revise_primary_lock": True,
                }
            ),
        )


@dataclass(frozen=True)
class BAIEGV2SourceTrainBatchLineage:
    stable_origin_registry: BAIEGG0StableOriginRegistryV1
    time_surface: BAIEGG0SupportRelativeTimeSurfaceV1
    final_left_closure: BAIEGV2FinalLeftClosureReceipt
    prediction_roster_receipt_sha256: str
    acquisition_support_lineage_receipt_sha256: str
    synthetic_training_authority_receipt_sha256: str
    event_acquisition_chain_tip_sha256s: tuple[str, ...]
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self.stable_origin_registry.verify_integrity()
        self.time_surface.verify_integrity()
        for name in (
            "prediction_roster_receipt_sha256",
            "acquisition_support_lineage_receipt_sha256",
            "synthetic_training_authority_receipt_sha256",
        ):
            _sha256(getattr(self, name), name)
        for value in self.event_acquisition_chain_tip_sha256s:
            _sha256(value, "event acquisition-chain tip")
        if (
            self.time_surface.source_stable_origin_registry_receipt_sha256
            != self.stable_origin_registry.receipt_sha256
            or self.final_left_closure.source_stable_origin_registry_receipt_sha256
            != self.stable_origin_registry.receipt_sha256
        ):
            raise ValueError("v2 batch lineage stable-origin binding drifted")
        object.__setattr__(
            self,
            "receipt_sha256",
            _canonical_sha256(
                {
                    "schema": "ba_ieg_v2_source_train_batch_lineage_v1",
                    "stable_origin_registry_receipt_sha256": self.stable_origin_registry.receipt_sha256,
                    "time_surface_receipt_sha256": self.time_surface.receipt_sha256,
                    "final_left_closure_receipt_sha256": self.final_left_closure.receipt_sha256,
                    "prediction_roster_receipt_sha256": self.prediction_roster_receipt_sha256,
                    "acquisition_support_lineage_receipt_sha256": self.acquisition_support_lineage_receipt_sha256,
                    "synthetic_training_authority_receipt_sha256": self.synthetic_training_authority_receipt_sha256,
                    "event_acquisition_chain_tip_sha256s": list(self.event_acquisition_chain_tip_sha256s),
                    "real_training_authorized": False,
                }
            ),
        )


def build_ba_ieg_v2_source_train_batch_lineage(
    batch: BAIEGSegmentalDiskBatchV1,
    *,
    prediction_roster: Mapping[str, Any] | None,
    candidate_ids_by_event: Sequence[str] | None,
    acquisition_support_lineage: Mapping[str, Any] | None,
    synthetic_training_authority: BAIEGA1SyntheticTrainingAuthorityV2 | None,
) -> BAIEGV2SourceTrainBatchLineage:
    """Bind one disk batch to A1 stable origin, support and final-left order."""

    if not isinstance(batch, BAIEGSegmentalDiskBatchV1):
        raise TypeError("v2 lineage requires a registered disk batch")
    if batch.optimization_role != "optimize" or batch.event_batch.model_split != "source_train":
        raise ValueError("v2 train lineage is source_train optimize-only")
    if prediction_roster is None:
        raise ValueError("v2 disk training requires an A1 prediction-roster receipt")
    if candidate_ids_by_event is None:
        raise ValueError("v2 disk training requires stable-origin candidate bindings")
    if acquisition_support_lineage is None:
        raise ValueError("v2 disk training requires A1 acquisition/support lineage")
    if synthetic_training_authority is None:
        require_real_a1_postfreeze_training_authority_v2(None)
    if not isinstance(synthetic_training_authority, BAIEGA1SyntheticTrainingAuthorityV2):
        raise TypeError("v2 software route requires typed synthetic authority")
    roster = validate_ba_ieg_g0_a1_prediction_roster_v1(dict(prediction_roster))
    lineage = validate_ba_ieg_g0_a1_acquisition_support_lineage_v1(
        dict(acquisition_support_lineage)
    )
    if (
        batch.target_independent_candidate_roster_receipt_sha256
        != roster["receipt_sha256"]
        or lineage["prediction_roster_receipt_sha256"] != roster["receipt_sha256"]
        or synthetic_training_authority.prediction_roster_receipt_sha256
        != roster["receipt_sha256"]
        or synthetic_training_authority.acquisition_support_lineage_receipt_sha256
        != lineage["receipt_sha256"]
        or synthetic_training_authority.target_independent_candidate_roster_receipt_sha256
        != batch.target_independent_candidate_roster_receipt_sha256
    ):
        raise ValueError("v2 disk batch crosses A1 candidate/support/training freeze")
    context = batch.build_context()
    stable = build_ba_ieg_g0_stable_origin_registry_v1(
        batch.event_batch,
        prediction_roster=roster,
        candidate_ids_by_event=tuple(candidate_ids_by_event),
    )
    if (
        lineage["stable_origin_registry_receipt_sha256"] != stable.receipt_sha256
        or synthetic_training_authority.stable_origin_registry_receipt_sha256
        != stable.receipt_sha256
    ):
        raise ValueError("v2 disk training stable-origin receipt is missing or crossed")
    event_by_id = {row["event_id"]: row for row in lineage["events"]}
    selected = []
    for index, event_id in enumerate(batch.event_batch.event_ids):
        row = event_by_id.get(event_id)
        if row is None:
            raise ValueError("v2 disk event is absent from A1 support lineage")
        if (
            row["patient_uid"] != batch.event_batch.patient_uids[index]
            or row["recording_id"] != batch.event_batch.recording_ids[index]
            or row["candidate_id"] != candidate_ids_by_event[index]
            or row["stable_origin_registry_receipt_sha256"] != stable.receipt_sha256
            or row["support_binding_status"] != "verified_reference_free_execution_chain"
            or row["acquisition_chain_tip_sha256"]
            != batch.adaptive_acquisition_receipt_sha256s[index]
        ):
            raise ValueError("v2 disk event A1 identity/support receipt drifted")
        observed = _canonical_intervals(
            context.observed_support_intervals_seconds[index],
            context.observed_support_mask[index],
        )
        if not _same_intervals(observed, row["final_support_union_recording_seconds"]):
            raise ValueError("v2 disk context differs from A1 final physical support")
        selected.append(row)
    surface = build_ba_ieg_g0_support_relative_time_surface_v1(
        batch.event_batch, context, stable
    )
    closure = BAIEGV2FinalLeftClosureReceipt(
        source_input_batch_sha256=batch.event_batch.input_batch_sha256,
        source_context_receipt_sha256=context.receipt_sha256,
        source_stable_origin_registry_receipt_sha256=stable.receipt_sha256,
        source_acquisition_support_lineage_receipt_sha256=lineage["receipt_sha256"],
        event_ids=batch.event_batch.event_ids,
        final_left_edge_recording_seconds=tuple(
            float(row["final_support_union_recording_seconds"][0][0])
            for row in selected
        ),
        final_support_union_sha256s=tuple(
            row["final_support_union_sha256"] for row in selected
        ),
    )
    return BAIEGV2SourceTrainBatchLineage(
        stable_origin_registry=stable,
        time_surface=surface,
        final_left_closure=closure,
        prediction_roster_receipt_sha256=roster["receipt_sha256"],
        acquisition_support_lineage_receipt_sha256=lineage["receipt_sha256"],
        synthetic_training_authority_receipt_sha256=(
            synthetic_training_authority.receipt_sha256
        ),
        event_acquisition_chain_tip_sha256s=tuple(
            row["acquisition_chain_tip_sha256"] for row in selected
        ),
    )


@dataclass(frozen=True)
class BAIEGSourceDevStableOriginRegistryV2:
    """Independent source-dev, calibration-only stable-origin registry."""

    source_input_batch_sha256: str
    event_ids: tuple[str, ...]
    recording_ids: tuple[str, ...]
    patient_uids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    candidate_receipt_sha256s: tuple[str, ...]
    acquisition_support_receipt_sha256s: tuple[str, ...]
    source_dev_prediction_roster_receipt_sha256: str
    provider_prediction_receipt_sha256: str
    decoder_policy_receipt_sha256: str
    checkpoint_patient_exclusion_receipt_sha256: str
    stable_origin_recording_seconds_output_only: torch.Tensor
    final_support_unions_recording_seconds: tuple[tuple[tuple[float, float], ...], ...]
    schema_version: str = BA_IEG_V2_SOURCE_DEV_STABLE_ORIGIN_SCHEMA
    model_split: str = "source_dev"
    calibration_only: bool = True
    gradient_updates_authorized: bool = False
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _sha256(self.source_input_batch_sha256, "source-dev input batch")
        for name in (
            "source_dev_prediction_roster_receipt_sha256",
            "provider_prediction_receipt_sha256",
            "decoder_policy_receipt_sha256",
            "checkpoint_patient_exclusion_receipt_sha256",
        ):
            _sha256(getattr(self, name), name)
        size = len(self.event_ids)
        aligned = (
            self.recording_ids,
            self.patient_uids,
            self.candidate_ids,
            self.candidate_receipt_sha256s,
            self.acquisition_support_receipt_sha256s,
            self.final_support_unions_recording_seconds,
        )
        if size < 1 or any(len(value) != size for value in aligned):
            raise ValueError("source-dev stable-origin rows do not align")
        if len(set(self.event_ids)) != size or len(set(self.candidate_ids)) != size:
            raise ValueError("source-dev stable-origin events/candidates must be unique")
        for digest in self.candidate_receipt_sha256s:
            _sha256(digest, "source-dev candidate receipt")
        for digest in self.acquisition_support_receipt_sha256s:
            _sha256(digest, "source-dev acquisition/support receipt")
        origins = self.stable_origin_recording_seconds_output_only.detach().clone().to(torch.float64)
        if tuple(origins.shape) != (size,) or not torch.isfinite(origins).all():
            raise ValueError("source-dev stable-origin tensor is invalid")
        if (
            self.schema_version != BA_IEG_V2_SOURCE_DEV_STABLE_ORIGIN_SCHEMA
            or self.model_split != "source_dev"
            or self.calibration_only is not True
            or self.gradient_updates_authorized is not False
        ):
            raise ValueError("source-dev stable-origin permission drifted")
        object.__setattr__(self, "stable_origin_recording_seconds_output_only", origins)
        tensor_hash = hashlib.sha256(origins.cpu().numpy().tobytes()).hexdigest()
        object.__setattr__(
            self,
            "receipt_sha256",
            self._compute_receipt(tensor_hash=tensor_hash),
        )

    def _compute_receipt(self, *, tensor_hash: str | None = None) -> str:
        if tensor_hash is None:
            tensor_hash = hashlib.sha256(
                self.stable_origin_recording_seconds_output_only.detach()
                .cpu()
                .contiguous()
                .numpy()
                .tobytes()
            ).hexdigest()
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "source_input_batch_sha256": self.source_input_batch_sha256,
                "event_ids": list(self.event_ids),
                "recording_ids": list(self.recording_ids),
                "patient_uids": list(self.patient_uids),
                "candidate_ids": list(self.candidate_ids),
                "candidate_receipt_sha256s": list(self.candidate_receipt_sha256s),
                "acquisition_support_receipt_sha256s": list(
                    self.acquisition_support_receipt_sha256s
                ),
                "source_dev_prediction_roster_receipt_sha256": self.source_dev_prediction_roster_receipt_sha256,
                "provider_prediction_receipt_sha256": self.provider_prediction_receipt_sha256,
                "decoder_policy_receipt_sha256": self.decoder_policy_receipt_sha256,
                "checkpoint_patient_exclusion_receipt_sha256": self.checkpoint_patient_exclusion_receipt_sha256,
                "stable_origin_tensor_sha256": tensor_hash,
                "final_support_unions_recording_seconds": self.final_support_unions_recording_seconds,
                "model_split": "source_dev",
                "calibration_only": True,
                "gradient_updates_authorized": False,
            }
        )

    def verify_integrity(self) -> None:
        if self.receipt_sha256 != self._compute_receipt():
            raise ValueError("source-dev stable-origin registry changed after registration")


def build_ba_ieg_source_dev_stable_origin_registry_v2(
    batch: BAIEGSegmentalDiskBatchV1,
    *,
    candidate_rows: Sequence[Mapping[str, Any]] | None,
    source_dev_prediction_roster_receipt_sha256: str | None,
    provider_prediction_receipt_sha256: str | None,
    decoder_policy_receipt_sha256: str | None,
    checkpoint_patient_exclusion_receipt_sha256: str | None,
) -> BAIEGSourceDevStableOriginRegistryV2:
    """Build the independent no-gradient source-dev candidate clock registry."""

    if not isinstance(batch, BAIEGSegmentalDiskBatchV1):
        raise TypeError("source-dev stable origins require a disk batch")
    if batch.optimization_role != "calibrate" or batch.event_batch.model_split != "source_dev":
        raise ValueError("source-dev stable origins are calibration-only")
    required_receipts = {
        "source_dev_prediction_roster_receipt_sha256": source_dev_prediction_roster_receipt_sha256,
        "provider_prediction_receipt_sha256": provider_prediction_receipt_sha256,
        "decoder_policy_receipt_sha256": decoder_policy_receipt_sha256,
        "checkpoint_patient_exclusion_receipt_sha256": checkpoint_patient_exclusion_receipt_sha256,
    }
    for name, value in required_receipts.items():
        if value is None:
            raise ValueError(f"source-dev calibration requires {name}")
        _sha256(value, name)
    if candidate_rows is None:
        raise ValueError("source-dev calibration requires candidate/stable-origin rows")
    rows = tuple(dict(value) for value in candidate_rows)
    fields_expected = {
        "event_id",
        "recording_id",
        "patient_uid",
        "candidate_id",
        "anchor_recording_seconds",
        "candidate_receipt_sha256",
        "acquisition_support_receipt_sha256",
        "final_support_union_recording_seconds",
    }
    if len(rows) != len(batch.event_batch.event_ids):
        raise ValueError("source-dev candidate rows must cover every event")
    context = batch.build_context()
    origins: list[float] = []
    candidate_ids: list[str] = []
    candidate_receipts: list[str] = []
    acquisition_receipts: list[str] = []
    support_unions: list[tuple[tuple[float, float], ...]] = []
    for index, row in enumerate(rows):
        if set(row) != fields_expected:
            raise ValueError("source-dev candidate row fields drifted")
        expected_identity = (
            batch.event_batch.event_ids[index],
            batch.event_batch.recording_ids[index],
            batch.event_batch.patient_uids[index],
        )
        if (row["event_id"], row["recording_id"], row["patient_uid"]) != expected_identity:
            raise ValueError("source-dev candidate identity/order drifted")
        candidate_ids.append(_identifier(row["candidate_id"], "source-dev candidate ID"))
        candidate_receipts.append(_sha256(row["candidate_receipt_sha256"], "source-dev candidate receipt"))
        acquisition_receipts.append(_sha256(row["acquisition_support_receipt_sha256"], "source-dev acquisition/support receipt"))
        anchor = float(row["anchor_recording_seconds"])
        if not math.isfinite(anchor):
            raise ValueError("source-dev stable origin must be finite")
        observed = _canonical_intervals(
            context.observed_support_intervals_seconds[index],
            context.observed_support_mask[index],
        )
        supplied = tuple(
            (float(value[0]), float(value[1]))
            for value in row["final_support_union_recording_seconds"]
        )
        if not _same_intervals(observed, supplied):
            raise ValueError("source-dev support row differs from disk context")
        if not any(start - _TOLERANCE <= anchor <= stop + _TOLERANCE for start, stop in observed):
            raise ValueError("source-dev stable origin lies outside observed support")
        if batch.adaptive_acquisition_receipt_sha256s[index] != acquisition_receipts[-1]:
            raise ValueError("source-dev disk acquisition receipt drifted")
        origins.append(anchor)
        support_unions.append(observed)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("source-dev stable-origin registry repeats a candidate")
    return BAIEGSourceDevStableOriginRegistryV2(
        source_input_batch_sha256=batch.event_batch.input_batch_sha256,
        event_ids=batch.event_batch.event_ids,
        recording_ids=batch.event_batch.recording_ids,
        patient_uids=batch.event_batch.patient_uids,
        candidate_ids=tuple(candidate_ids),
        candidate_receipt_sha256s=tuple(candidate_receipts),
        acquisition_support_receipt_sha256s=tuple(acquisition_receipts),
        source_dev_prediction_roster_receipt_sha256=source_dev_prediction_roster_receipt_sha256,
        provider_prediction_receipt_sha256=provider_prediction_receipt_sha256,
        decoder_policy_receipt_sha256=decoder_policy_receipt_sha256,
        checkpoint_patient_exclusion_receipt_sha256=checkpoint_patient_exclusion_receipt_sha256,
        stable_origin_recording_seconds_output_only=torch.tensor(
            origins,
            dtype=torch.float64,
            device=batch.event_batch.token_time_bounds_seconds.device,
        ),
        final_support_unions_recording_seconds=tuple(support_unions),
    )


def build_ba_ieg_source_dev_time_surface_v2(
    batch: BAIEGSegmentalDiskBatchV1,
    registry: BAIEGSourceDevStableOriginRegistryV2,
) -> BAIEGG0SupportRelativeTimeSurfaceV1:
    """Numerically project the dev registry without relabelling it source-train."""

    if not isinstance(registry, BAIEGSourceDevStableOriginRegistryV2):
        raise TypeError("source-dev time surface requires its typed registry")
    registry.verify_integrity()
    if batch.event_batch.input_batch_sha256 != registry.source_input_batch_sha256:
        raise ValueError("source-dev registry crosses disk batch identity")
    context = batch.build_context()
    eeg_batch = batch.event_batch
    active = eeg_batch.token_row_mask & eeg_batch.token_signal_mask
    batch_size, token_count = active.shape
    device = eeg_batch.token_time_bounds_seconds.device
    absolute = eeg_batch.token_time_bounds_seconds.detach().to(torch.float64)
    relative = torch.zeros((batch_size, token_count, 2), dtype=torch.float64, device=device)
    learned = torch.zeros(
        (batch_size, token_count, len(BA_IEG_G0_SUPPORT_RELATIVE_TIME_FEATURE_NAMES_V1)),
        dtype=torch.float64,
        device=device,
    )
    support_relative = torch.zeros_like(context.observed_support_intervals_seconds, dtype=torch.float64, device=device)
    gap_relative = torch.zeros_like(context.quality_gap_intervals_seconds, dtype=torch.float64, device=device)
    origins = registry.stable_origin_recording_seconds_output_only.to(device=device)
    for batch_index in range(batch_size):
        support_tensor = context.observed_support_intervals_seconds[
            batch_index, context.observed_support_mask[batch_index]
        ].detach().to(torch.float64)
        gap_tensor = context.quality_gap_intervals_seconds[
            batch_index, context.quality_gap_mask[batch_index]
        ].detach().to(torch.float64)
        origin = float(origins[batch_index])
        support_start = float(support_tensor[:, 0].min())
        support_stop = float(support_tensor[:, 1].max())
        support_relative[batch_index, context.observed_support_mask[batch_index]] = support_tensor - origin
        if bool(context.quality_gap_mask[batch_index].any()):
            gap_relative[batch_index, context.quality_gap_mask[batch_index]] = gap_tensor - origin
        support_rows = [(float(row[0]), float(row[1])) for row in support_tensor.cpu()]
        gap_rows = [(float(row[0]), float(row[1])) for row in gap_tensor.cpu()]
        clean = _clean_intervals(support_rows, gap_rows)
        for token_index in torch.nonzero(active[batch_index], as_tuple=False).flatten().tolist():
            start = float(absolute[batch_index, token_index, 0])
            stop = float(absolute[batch_index, token_index, 1])
            midpoint = 0.5 * (start + stop)
            duration = stop - start
            relative_start, relative_stop = start - origin, stop - origin
            relative_midpoint = midpoint - origin
            relative[batch_index, token_index] = torch.tensor(
                (relative_start, relative_stop), dtype=torch.float64, device=device
            )
            overlap = _gap_overlap(start, stop, gap_rows)
            learned[batch_index, token_index] = torch.tensor(
                (
                    math.asinh(relative_start / 60.0),
                    math.asinh(relative_stop / 60.0),
                    math.asinh(relative_midpoint / 60.0),
                    math.log1p(duration),
                    math.asinh(_signed_opportunity_displacement(start, origin, clean) / 60.0),
                    math.asinh(_signed_opportunity_displacement(stop, origin, clean) / 60.0),
                    math.asinh(_signed_opportunity_displacement(midpoint, origin, clean) / 60.0),
                    min(1.0, overlap / duration),
                    float(overlap > 1e-12),
                    float(abs(start - support_start) <= 1e-9),
                    float(abs(stop - support_stop) <= 1e-9),
                ),
                dtype=torch.float64,
                device=device,
            )
    return BAIEGG0SupportRelativeTimeSurfaceV1(
        source_input_batch_sha256=eeg_batch.input_batch_sha256,
        source_context_receipt_sha256=context.receipt_sha256,
        source_stable_origin_registry_receipt_sha256=registry.receipt_sha256,
        event_ids=eeg_batch.event_ids,
        stable_origin_recording_seconds_output_only=origins,
        absolute_token_bounds_recording_seconds_output_only=torch.where(
            active.unsqueeze(-1), absolute, torch.zeros_like(absolute)
        ),
        support_relative_token_bounds_seconds=relative,
        learned_time_features=learned,
        token_active_mask=active,
        support_relative_observed_intervals_seconds=support_relative,
        observed_support_mask=context.observed_support_mask,
        support_relative_quality_gap_intervals_seconds=gap_relative,
        quality_gap_mask=context.quality_gap_mask,
        left_censor_reason_codes=context.left_censor_reason_codes,
        right_censor_reason_codes=context.right_censor_reason_codes,
    )


@dataclass(frozen=True)
class BAIEGV2SourceDevCalibrationForward:
    context: Any
    registry_receipt_sha256: str
    time_surface_receipt_sha256: str
    final_left_closure: BAIEGV2FinalLeftClosureReceipt
    segmental: BAIEGPermissionSplitSegmentalStateOutputV2
    k3_gate: BAIEGEarliestPrefixK3GateResultV1
    typed_unit: BAIEGShallowCausalTypedUnitHeadOutput
    gradients_retained: bool = False


@dataclass(frozen=True)
class BAIEGV2CompositeForward:
    context: Any
    lineage: BAIEGV2SourceTrainBatchLineage
    segmental: BAIEGPermissionSplitSegmentalStateOutputV2
    final_left_closure: BAIEGV2FinalLeftClosureReceipt
    k3_gate: BAIEGEarliestPrefixK3GateResultV1
    typed_unit: BAIEGShallowCausalTypedUnitHeadOutput
    patient_aggregation: BAIEGCompletePatientAggregationOutputV1


@dataclass(frozen=True)
class BAIEGV2ThreeLossOutput:
    total_loss: torch.Tensor
    segmental: BAIEGPermissionSplitSegmentalLossOutputV1
    typed_boundary: BAIEGShallowCausalTypedUnitMILBoundaryLossOutputV2
    patient_positive_set: BAIEGCompletePatientPositiveSetLossOutputV1
    loss_weights: tuple[float, float, float]


@dataclass(frozen=True)
class BAIEGV2GradientAuthorityAudit:
    nonzero_parameter_names_by_loss: Mapping[str, tuple[str, ...]]
    parameter_group_by_name: Mapping[str, str]
    receipt_sha256: str


class BAIEGV2DiskCompositeModel(nn.Module):
    """Trainable v2 segmental + K3 typed + complete-patient composition."""

    implementation_id: Final[str] = BA_IEG_V2_DISK_COMPOSITE_ID

    def __init__(
        self,
        *,
        segmental_model: BAIEGPermissionSplitSegmentalStateModelV2,
        typed_head: BAIEGShallowCausalTypedUnitOnsetHeadV2,
        k3_policy: BAIEGEarliestPrefixK3DevelopmentPolicyV1,
        allow_synthetic_legacy_event_statuses: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(segmental_model, BAIEGPermissionSplitSegmentalStateModelV2):
            raise TypeError("v2 composite requires the stable-clock segmental model")
        if not isinstance(typed_head, BAIEGShallowCausalTypedUnitOnsetHeadV2):
            raise TypeError("v2 composite requires the signal-only typed head")
        if not isinstance(k3_policy, BAIEGEarliestPrefixK3DevelopmentPolicyV1):
            raise TypeError("v2 composite requires a registered K3 policy")
        if segmental_model.hidden_dim != typed_head.hidden_dim:
            raise ValueError("v2 segmental/typed hidden dimensions disagree")
        if type(allow_synthetic_legacy_event_statuses) is not bool:
            raise TypeError("synthetic event-status permission must be boolean")
        self.segmental_model = segmental_model
        self.typed_head = typed_head
        self.k3_policy = k3_policy
        self.event_bag = BAIEGCappedLogMeanExpEventBag(
            allow_legacy_component_test_only=allow_synthetic_legacy_event_statuses
        )
        self.patient_bag = BAIEGA1CompletePatientCappedLogMeanExpV2()
        self.allow_synthetic_legacy_event_statuses = (
            allow_synthetic_legacy_event_statuses
        )

    def parameter_group_by_name(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for name, _ in self.named_parameters():
            if name.startswith("segmental_model.typed_signal_"):
                group = "typed_signal_projection"
            elif name.startswith("segmental_model."):
                group = "segmental_causal_offline_core"
            elif name.startswith(("typed_head.boundary_fusion", "typed_head.boundary_head", "typed_head.no_boundary_logit_by_kind")):
                group = "typed_boundary_adapter"
            elif name.startswith(("typed_head.identity_fusion", "typed_head.cell_rank_head")):
                group = "identity_rank_adapter"
            else:
                raise RuntimeError(f"unregistered v2 trainable parameter: {name}")
            result[name] = group
        return result

    def forward(
        self,
        batch: BAIEGSegmentalDiskBatchV1,
        lineage: BAIEGV2SourceTrainBatchLineage,
        *,
        event_manifest: BAIEGRecordEventBagManifest,
        patient_roster: BAIEGA1CompletePatientRecordRosterV2,
    ) -> BAIEGV2CompositeForward:
        if not isinstance(batch, BAIEGSegmentalDiskBatchV1) or not isinstance(
            lineage, BAIEGV2SourceTrainBatchLineage
        ):
            raise TypeError("v2 composite forward requires disk batch and A1 lineage")
        context = batch.build_context()
        if (
            lineage.time_surface.source_input_batch_sha256
            != batch.event_batch.input_batch_sha256
            or lineage.time_surface.source_context_receipt_sha256 != context.receipt_sha256
            or lineage.final_left_closure.event_ids != batch.event_batch.event_ids
            or lineage.final_left_closure.source_context_receipt_sha256
            != context.receipt_sha256
            or patient_roster.synthetic_training_authority_receipt_sha256
            != lineage.synthetic_training_authority_receipt_sha256
        ):
            raise ValueError("v2 composite batch/lineage/left-closure/roster crossed")
        segmental = self.segmental_model(
            batch.event_batch, context, lineage.time_surface
        )
        # Ordering is deliberate: the content-bound final-left receipt is
        # verified before the earliest-prefix/K3 decision is even constructed.
        if not lineage.final_left_closure.closure_precedes_earliest_prefix_k3:
            raise ValueError("K3 gate cannot precede final-left closure")
        gate = build_ba_ieg_earliest_prefix_k3_gate_v1(
            segmental.causal_typed_unit_trace, self.k3_policy
        )
        typed = self.typed_head(gate)
        record = self.event_bag(typed, event_manifest)
        evidence = BAIEGPhysicalRecordEvidenceBatchV1.from_event_bag_output(record)
        patient = self.patient_bag(evidence, patient_roster)
        return BAIEGV2CompositeForward(
            context=context,
            lineage=lineage,
            segmental=segmental,
            final_left_closure=lineage.final_left_closure,
            k3_gate=gate,
            typed_unit=typed,
            patient_aggregation=patient,
        )

    def calibration_forward_source_dev(
        self,
        batch: BAIEGSegmentalDiskBatchV1,
        registry: BAIEGSourceDevStableOriginRegistryV2,
    ) -> BAIEGV2SourceDevCalibrationForward:
        """No-gradient source-dev route; it cannot compute or update losses."""

        if not isinstance(batch, BAIEGSegmentalDiskBatchV1):
            raise TypeError("v2 source-dev calibration requires a disk batch")
        if batch.optimization_role != "calibrate" or batch.event_batch.model_split != "source_dev":
            raise ValueError("v2 calibration forward is source-dev-only")
        registry.verify_integrity()
        if registry.source_input_batch_sha256 != batch.event_batch.input_batch_sha256:
            raise ValueError("v2 calibration registry crosses disk batch")
        context = batch.build_context()
        surface = build_ba_ieg_source_dev_time_surface_v2(batch, registry)
        acquisition_roster_receipt = _canonical_sha256(
            {
                "schema": "ba_ieg_source_dev_final_support_receipt_roster_v2",
                "registry_receipt_sha256": registry.receipt_sha256,
                "acquisition_support_receipt_sha256s": list(
                    registry.acquisition_support_receipt_sha256s
                ),
                "final_support_unions_recording_seconds": (
                    registry.final_support_unions_recording_seconds
                ),
                "calibration_only": True,
            }
        )
        support_hashes = tuple(
            _canonical_sha256(
                {
                    "recording_id": recording_id,
                    "support_union_recording_seconds": support,
                    "acquisition_support_receipt_sha256": acquisition_receipt,
                }
            )
            for recording_id, support, acquisition_receipt in zip(
                registry.recording_ids,
                registry.final_support_unions_recording_seconds,
                registry.acquisition_support_receipt_sha256s,
            )
        )
        closure = BAIEGV2FinalLeftClosureReceipt(
            source_input_batch_sha256=batch.event_batch.input_batch_sha256,
            source_context_receipt_sha256=context.receipt_sha256,
            source_stable_origin_registry_receipt_sha256=registry.receipt_sha256,
            source_acquisition_support_lineage_receipt_sha256=(
                acquisition_roster_receipt
            ),
            event_ids=batch.event_batch.event_ids,
            final_left_edge_recording_seconds=tuple(
                support[0][0]
                for support in registry.final_support_unions_recording_seconds
            ),
            final_support_union_sha256s=support_hashes,
        )
        self.eval()
        with torch.no_grad():
            segmental = self.segmental_model(batch.event_batch, context, surface)
            gate = build_ba_ieg_earliest_prefix_k3_gate_v1(
                segmental.causal_typed_unit_trace, self.k3_policy
            )
            typed = self.typed_head(gate)
        segmental = _detach_dataclass_tree(segmental)
        gate = _detach_dataclass_tree(gate)
        typed = _detach_dataclass_tree(typed)
        retained = any(
            tensor.requires_grad
            for output in (segmental.primary, typed)
            for tensor in output.__dict__.values()
            if isinstance(tensor, torch.Tensor)
        )
        if retained:
            raise RuntimeError("source-dev calibration retained a gradient")
        return BAIEGV2SourceDevCalibrationForward(
            context=context,
            registry_receipt_sha256=registry.receipt_sha256,
            time_surface_receipt_sha256=surface.receipt_sha256,
            final_left_closure=closure,
            segmental=segmental,
            k3_gate=gate,
            typed_unit=typed,
            gradients_retained=False,
        )


class BAIEGPermissionSplitSegmentalCompositeTrainerV2:
    """Three-loss optimizer with explicit per-parameter authority replay."""

    def __init__(
        self,
        model: BAIEGV2DiskCompositeModel,
        optimizer: torch.optim.Optimizer,
        *,
        training_dataset: BAIEGSegmentalDiskDatasetV1,
        training_authority: BAIEGA1SyntheticTrainingAuthorityV2,
        scheduler: Any | None = None,
        maximum_gradient_norm: float | None = None,
        loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        if not isinstance(model, BAIEGV2DiskCompositeModel):
            raise TypeError("v2 trainer requires the registered composite model")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("v2 trainer optimizer is invalid")
        if not isinstance(training_dataset, BAIEGSegmentalDiskDatasetV1) or training_dataset.purpose != "optimize":
            raise TypeError("v2 trainer requires a source-train disk dataset")
        if not isinstance(training_authority, BAIEGA1SyntheticTrainingAuthorityV2):
            raise TypeError(
                "v2 trainer requires typed A1 software authority; real training "
                "remains fail-closed"
            )
        if (
            training_authority.target_independent_candidate_roster_receipt_sha256
            != training_dataset.candidate_roster_receipt_sha256
        ):
            raise ValueError("v2 trainer authority crosses the disk candidate roster")
        weights = tuple(float(value) for value in loss_weights)
        if len(weights) != 3 or any(not math.isfinite(value) or value <= 0 for value in weights):
            raise ValueError("v2 three-loss weights must be positive finite values")
        if maximum_gradient_norm is not None and (
            not math.isfinite(float(maximum_gradient_norm))
            or float(maximum_gradient_norm) <= 0
        ):
            raise ValueError("v2 maximum gradient norm must be positive")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.maximum_gradient_norm = (
            None if maximum_gradient_norm is None else float(maximum_gradient_norm)
        )
        self.loss_weights = weights
        self.manifest_id = training_dataset.manifest_id
        self.manifest_file_sha256 = training_dataset.manifest_file_sha256
        self.candidate_roster_receipt_sha256 = (
            training_dataset.candidate_roster_receipt_sha256
        )
        self.prediction_roster_receipt_sha256 = (
            training_authority.prediction_roster_receipt_sha256
        )
        self.acquisition_support_lineage_receipt_sha256 = (
            training_authority.acquisition_support_lineage_receipt_sha256
        )
        self.stable_origin_registry_receipt_sha256 = (
            training_authority.stable_origin_registry_receipt_sha256
        )
        self.training_authority_receipt_sha256 = training_authority.receipt_sha256
        self.epoch = 0
        self.optimizer_step = 0
        self.batches_consumed_in_epoch = 0

    def _losses(
        self,
        batch: BAIEGSegmentalDiskBatchV1,
        forward: BAIEGV2CompositeForward,
        positive_sets: Sequence[BAIEGBoundDeepSOZPositiveSetV1],
    ) -> BAIEGV2ThreeLossOutput:
        target_bundle = batch.build_target_bundle(forward.context)
        segmental = permission_split_segmental_training_loss_v1(
            batch.event_batch,
            forward.context,
            forward.segmental.primary,
            target_bundle,
        )
        projections = tuple(
            project_ba_ieg_segmental_target_to_frozen_lattice_v1(
                target,
                forward.segmental.primary,
                forward.context,
                index,
            )
            for index, target in enumerate(batch.targets)
        )
        typed_targets = build_ba_ieg_shallow_causal_typed_unit_mil_target_bundle_v2(
            forward.k3_gate.gated_trace, batch.targets, projections
        )
        typed = shallow_causal_typed_unit_mil_boundary_loss_v2(
            forward.typed_unit, typed_targets
        )
        patient = complete_patient_positive_set_mass_loss_v1(
            forward.patient_aggregation, positive_sets
        )
        if not patient.optimizer_step_allowed:
            raise ValueError("v2 patient positive-set loss has no evaluable complete patient")
        total = (
            self.loss_weights[0] * segmental.total_loss
            + self.loss_weights[1] * typed.total_loss
            + self.loss_weights[2] * patient.total_loss
        )
        return BAIEGV2ThreeLossOutput(
            total_loss=total,
            segmental=segmental,
            typed_boundary=typed,
            patient_positive_set=patient,
            loss_weights=self.loss_weights,
        )

    def audit_gradient_authority(
        self, losses: BAIEGV2ThreeLossOutput
    ) -> BAIEGV2GradientAuthorityAudit:
        named = tuple(self.model.named_parameters())
        names = tuple(name for name, _ in named)
        parameters = tuple(parameter for _, parameter in named)
        groups = self.model.parameter_group_by_name()
        allowed = {
            "segmental": {"segmental_causal_offline_core"},
            "typed_boundary": {"typed_signal_projection", "typed_boundary_adapter"},
            "patient_positive_set": {"identity_rank_adapter"},
        }
        tensors = {
            "segmental": losses.segmental.total_loss,
            "typed_boundary": losses.typed_boundary.total_loss,
            "patient_positive_set": losses.patient_positive_set.total_loss,
        }
        observed: dict[str, tuple[str, ...]] = {}
        for loss_name, loss in tensors.items():
            gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            nonzero = tuple(
                name
                for name, gradient in zip(names, gradients)
                if gradient is not None and bool(gradient.detach().abs().sum() > 0)
            )
            illegal = [name for name in nonzero if groups[name] not in allowed[loss_name]]
            if illegal:
                raise RuntimeError(
                    f"{loss_name} crossed gradient authority: {illegal}"
                )
            seen_groups = {groups[name] for name in nonzero}
            if not allowed[loss_name].issubset(seen_groups):
                raise RuntimeError(
                    f"{loss_name} did not reach every authorized parameter group"
                )
            observed[loss_name] = nonzero
        receipt = _canonical_sha256(
            {
                "schema": "ba_ieg_v2_three_loss_per_parameter_gradient_authority_v1",
                "nonzero_parameter_names_by_loss": observed,
                "parameter_group_by_name": groups,
                "allowed_groups_by_loss": {
                    name: sorted(value) for name, value in allowed.items()
                },
                "global_hidden_detached_for_typed_boundary": True,
                "identity_input_and_locked_global_gate_detached": True,
            }
        )
        return BAIEGV2GradientAuthorityAudit(
            nonzero_parameter_names_by_loss=observed,
            parameter_group_by_name=groups,
            receipt_sha256=receipt,
        )

    def optimize_batch(
        self,
        batch: BAIEGSegmentalDiskBatchV1,
        lineage: BAIEGV2SourceTrainBatchLineage,
        *,
        event_manifest: BAIEGRecordEventBagManifest,
        patient_roster: BAIEGA1CompletePatientRecordRosterV2,
        positive_sets: Sequence[BAIEGBoundDeepSOZPositiveSetV1],
        audit_gradients: bool = True,
    ) -> tuple[BAIEGV2ThreeLossOutput, BAIEGV2GradientAuthorityAudit | None]:
        if (
            batch.optimization_role != "optimize"
            or batch.event_batch.model_split != "source_train"
        ):
            raise ValueError(
                "v2 gradient updates are source_train optimize-only; use the "
                "independent source-dev calibration registry"
            )
        if (
            batch.manifest_id != self.manifest_id
            or batch.manifest_file_sha256 != self.manifest_file_sha256
            or batch.target_independent_candidate_roster_receipt_sha256
            != self.candidate_roster_receipt_sha256
            or lineage.prediction_roster_receipt_sha256
            != self.prediction_roster_receipt_sha256
            or lineage.acquisition_support_lineage_receipt_sha256
            != self.acquisition_support_lineage_receipt_sha256
            or lineage.stable_origin_registry.receipt_sha256
            != self.stable_origin_registry_receipt_sha256
            or lineage.synthetic_training_authority_receipt_sha256
            != self.training_authority_receipt_sha256
        ):
            raise ValueError(
                "v2 optimizer batch crosses frozen disk/A1 lineage authority"
            )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        forward = self.model(
            batch,
            lineage,
            event_manifest=event_manifest,
            patient_roster=patient_roster,
        )
        losses = self._losses(batch, forward, positive_sets)
        audit = self.audit_gradient_authority(losses) if audit_gradients else None
        losses.total_loss.backward()
        if self.maximum_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.maximum_gradient_norm
            )
        self.optimizer.step()
        self.optimizer_step += 1
        self.batches_consumed_in_epoch += 1
        return losses, audit

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise TypeError("v2 trainer epoch must be a non-negative integer")
        self.epoch = epoch
        self.batches_consumed_in_epoch = 0

    def step_scheduler(self) -> None:
        if self.scheduler is None:
            raise ValueError("v2 trainer scheduler is not configured")
        self.scheduler.step()


__all__ = [
    "BA_IEG_V2_DISK_COMPOSITE_ID",
    "BA_IEG_V2_FINAL_LEFT_CLOSURE_SCHEMA",
    "BA_IEG_V2_SOURCE_DEV_STABLE_ORIGIN_SCHEMA",
    "BAIEGPermissionSplitSegmentalCompositeTrainerV2",
    "BAIEGSourceDevStableOriginRegistryV2",
    "BAIEGV2CompositeForward",
    "BAIEGV2DiskCompositeModel",
    "BAIEGV2FinalLeftClosureReceipt",
    "BAIEGV2GradientAuthorityAudit",
    "BAIEGV2SourceTrainBatchLineage",
    "BAIEGV2SourceDevCalibrationForward",
    "BAIEGV2ThreeLossOutput",
    "build_ba_ieg_source_dev_stable_origin_registry_v2",
    "build_ba_ieg_source_dev_time_surface_v2",
    "build_ba_ieg_v2_source_train_batch_lineage",
]
