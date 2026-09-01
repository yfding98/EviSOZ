"""Host-replayed P0 event projection with three-way content binding.

Projection v1 remains the target-detachment boundary.  This additive v2 wraps
that unchanged contract and additionally binds the complete, host-replayed
raw-sample dependency sidecar.  The model input remains target-free; neither
the deterministic supervision sidecar nor the dependency sidecar is a model
forward input.

Embedded hashes are not authority.  The public validator requires the host
canonical receipt and the complete trusted view registry, then replays every
raw-dependency row from those roots.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Final, Mapping

import torch

from .ba_ieg_event_model_input_projection_v1 import (
    BAIEGContentBoundDeterministicTargetSidecarV1,
    BAIEGEventModelInputProjectionV1,
    project_ba_ieg_event_model_input_v1,
)
from .ba_ieg_p0_raw_sample_dependency_v1 import (
    _validate_embedded_content as _validate_raw_dependency_embedded_content,
    materialize_ba_ieg_p0_raw_sample_dependency_sidecar_v1,
    validate_ba_ieg_p0_raw_sample_dependency_sidecar_v1,
)
from .ba_ieg_training_contract import (
    BA_IEG_TOKEN_SCALES,
    BAIEGEventTokens,
    BAIEGP0MaterializationResult,
)


BA_IEG_EVENT_MODEL_INPUT_PROJECTION_SCHEMA_VERSION_V2: Final[
    str
] = "ba_ieg_event_model_input_projection_v2"
BA_IEG_EVENT_MODEL_INPUT_PROJECTION_METHOD_ID_V2: Final[
    str
] = "ba_ieg_target_free_event_target_and_host_raw_dependency_projection_v2"

_SHA256_CHARACTERS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_PROJECTION_SCOPE_RECEIPT_V2: Final[dict[str, bool]] = {
    "model_input_target_free": True,
    "deterministic_target_sidecar_bound": True,
    "deterministic_target_available_to_model_forward": False,
    "raw_sample_dependency_sidecar_bound": True,
    "raw_sample_dependency_available_to_model_forward": False,
    "host_canonical_receipt_required_for_validation": True,
    "complete_trusted_view_registry_required_for_validation": True,
    "embedded_hash_treated_as_host_authority": False,
    "edf_annotation_used": False,
    "spreadsheet_used": False,
    "clinical_text_used": False,
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in _SHA256_CHARACTERS for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _validate_raw_event_cross_binding(
    event: BAIEGEventTokens,
    raw_sidecar: Mapping[str, Any],
    *,
    source_p0_materialization_receipt_sha256: str,
) -> None:
    source = raw_sidecar["source_binding"]
    expected_source = {
        "event_id": event.event_id,
        "recording_id": event.recording_id,
        "patient_uid": event.patient_uid,
        "model_split": event.model_split,
        "event_model_input_receipt_sha256": event.input_receipt_sha256,
        "source_p0_materialization_receipt_sha256": (
            source_p0_materialization_receipt_sha256
        ),
        "canonical_receipt_sha256": event.canonical_receipt_sha256,
        "trusted_view_receipt_sha256s": list(event.view_receipt_sha256s),
        "token_count": int(event.token_values.shape[0]),
    }
    for name, value in expected_source.items():
        if source.get(name) != value:
            raise ValueError(
                f"raw-dependency sidecar {name} drifted from the event input"
            )

    dependencies = raw_sidecar["dependencies"]
    if len(dependencies) != int(event.token_values.shape[0]):
        raise ValueError("raw-dependency sidecar lost an event token")
    for token_index, dependency in enumerate(dependencies):
        unit_index = int(event.token_unit_index[token_index])
        view_index = int(event.token_view_index[token_index])
        role = event.view_effective_temporal_roles[view_index]
        source_binding = dependency["source_binding"]
        coordinate = dependency["token_coordinate"]
        temporal = dependency["temporal_contract"]
        reference = dependency["reference_lineage"]
        output_support = dependency["output_support"]
        expected_binding = {
            "event_model_input_receipt_sha256": event.input_receipt_sha256,
            "source_p0_materialization_receipt_sha256": (
                source_p0_materialization_receipt_sha256
            ),
            "canonical_receipt_sha256": event.canonical_receipt_sha256,
            "view_receipt_sha256": event.view_receipt_sha256s[view_index],
            "view_transform_spec_sha256": (
                event.view_transform_sha256s[view_index]
            ),
        }
        for name, value in expected_binding.items():
            if source_binding.get(name) != value:
                raise ValueError(
                    f"raw dependency {token_index} {name} was rebound"
                )
        expected_coordinate = {
            "event_id": event.event_id,
            "recording_id": event.recording_id,
            "view_index": view_index,
            "view_id": event.view_ids[view_index],
            "unit_index": unit_index,
            "unit_id": event.unit_ids[unit_index],
            "unit_source_id": event.unit_source_ids[unit_index],
            "scale_index": int(event.token_scale_index[token_index]),
            "scale": BA_IEG_TOKEN_SCALES[
                int(event.token_scale_index[token_index])
            ],
            "signal_eligible": bool(event.token_signal_mask[token_index]),
        }
        for name, value in expected_coordinate.items():
            if coordinate.get(name) != value:
                raise ValueError(
                    f"raw dependency {token_index} token coordinate was rebound"
                )
        expected_temporal = {
            "effective_temporal_role": role,
            "dependency_policy": event.view_dependency_policies[view_index],
            "future_sample_access": bool(
                event.view_future_sample_access[view_index]
            ),
            "clinical_onset_evidence_authorized": bool(
                event.view_onset_evidence_authorized[view_index]
            ),
            "token_onset_evidence_eligible": bool(
                event.token_onset_evidence_mask[token_index]
            ),
            "temporal_evidence_sha256": (
                event.view_temporal_evidence_sha256s[view_index]
            ),
        }
        for name, value in expected_temporal.items():
            if temporal.get(name) != value:
                raise ValueError(
                    f"raw dependency {token_index} temporal contract drifted"
                )
        if role == "morphology_native" and any(
            (
                bool(temporal["future_sample_access"]),
                bool(temporal["clinical_onset_evidence_authorized"]),
                bool(temporal["token_onset_evidence_eligible"]),
            )
        ):
            raise ValueError("native morphology acquired onset/future authority")
        if output_support.get("event_token_recording_interval_seconds") != [
            float(item)
            for item in event.token_time_bounds_seconds[token_index].tolist()
        ]:
            raise ValueError("raw dependency token time was rebound")
        if reference.get("physical_input_unit_ids") != list(
            event.physical_electrode_ids
        ):
            raise ValueError("raw dependency physical basis drifted")
        supplied_row = torch.tensor(
            reference.get("exact_signed_reference_row"), dtype=torch.float32
        )
        if not torch.equal(
            supplied_row,
            event.unit_reference_matrix[unit_index].detach().cpu().to(torch.float32),
        ):
            raise ValueError("raw dependency signed reference row was rebound")
        closure = dependency["raw_support"].get(
            "raw_dependency_closure_proven"
        )
        if type(closure) is not bool:
            raise TypeError("raw dependency closure flag must be boolean")


@dataclass(frozen=True)
class BAIEGEventModelInputProjectionV2:
    """Target-free event, target sidecar and raw dependency sidecar."""

    model_input_event: BAIEGEventTokens
    deterministic_target_sidecar: BAIEGContentBoundDeterministicTargetSidecarV1
    raw_sample_dependency_sidecar: Mapping[str, Any] = field(
        repr=False, compare=False
    )
    source_p0_materialization_receipt_sha256: str
    schema_version: str = BA_IEG_EVENT_MODEL_INPUT_PROJECTION_SCHEMA_VERSION_V2
    method_id: str = BA_IEG_EVENT_MODEL_INPUT_PROJECTION_METHOD_ID_V2
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != (
            BA_IEG_EVENT_MODEL_INPUT_PROJECTION_SCHEMA_VERSION_V2
        ):
            raise ValueError("BA-IEG event projection v2 schema drifted")
        if self.method_id != BA_IEG_EVENT_MODEL_INPUT_PROJECTION_METHOD_ID_V2:
            raise ValueError("BA-IEG event projection v2 method drifted")
        _sha256(
            self.source_p0_materialization_receipt_sha256,
            "source_p0_materialization_receipt_sha256",
        )
        if not isinstance(self.model_input_event, BAIEGEventTokens):
            raise TypeError("projection v2 requires a BA-IEG event input")
        if not isinstance(
            self.deterministic_target_sidecar,
            BAIEGContentBoundDeterministicTargetSidecarV1,
        ):
            raise TypeError("projection v2 requires a deterministic target sidecar")
        # Reuse the unchanged v1 target-detachment invariant.
        BAIEGEventModelInputProjectionV1(
            model_input_event=self.model_input_event,
            deterministic_target_sidecar=self.deterministic_target_sidecar,
            source_p0_materialization_receipt_sha256=(
                self.source_p0_materialization_receipt_sha256
            ),
        )
        raw = _validate_raw_dependency_embedded_content(
            deepcopy(dict(self.raw_sample_dependency_sidecar))
        )
        _validate_raw_event_cross_binding(
            self.model_input_event,
            raw,
            source_p0_materialization_receipt_sha256=(
                self.source_p0_materialization_receipt_sha256
            ),
        )
        object.__setattr__(self, "raw_sample_dependency_sidecar", raw)
        object.__setattr__(self, "receipt_sha256", self._compute_sha256())

    @property
    def scope_receipt(self) -> dict[str, bool]:
        return dict(_PROJECTION_SCOPE_RECEIPT_V2)

    @property
    def raw_sample_dependency_sidecar_sha256(self) -> str:
        return str(self.raw_sample_dependency_sidecar["sidecar_sha256"])

    @property
    def raw_dependency_roster_sha256(self) -> str:
        return str(self.raw_sample_dependency_sidecar["dependency_roster_sha256"])

    def _compute_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "method_id": self.method_id,
                "source_p0_materialization_receipt_sha256": (
                    self.source_p0_materialization_receipt_sha256
                ),
                "model_input_event_receipt_sha256": (
                    self.model_input_event.input_receipt_sha256
                ),
                "deterministic_target_sidecar_receipt_sha256": (
                    self.deterministic_target_sidecar.receipt_sha256
                ),
                "deterministic_target_receipt_sha256": (
                    self.deterministic_target_sidecar.target_receipt_sha256
                ),
                "raw_sample_dependency_sidecar_id": (
                    self.raw_sample_dependency_sidecar["sidecar_id"]
                ),
                "raw_sample_dependency_sidecar_sha256": (
                    self.raw_sample_dependency_sidecar_sha256
                ),
                "raw_dependency_roster_sha256": (
                    self.raw_dependency_roster_sha256
                ),
                "scope_receipt": _PROJECTION_SCOPE_RECEIPT_V2,
            }
        )

    def verify_integrity(self) -> None:
        BAIEGEventModelInputProjectionV1(
            model_input_event=self.model_input_event,
            deterministic_target_sidecar=self.deterministic_target_sidecar,
            source_p0_materialization_receipt_sha256=(
                self.source_p0_materialization_receipt_sha256
            ),
        )
        raw = _validate_raw_dependency_embedded_content(
            self.raw_sample_dependency_sidecar
        )
        _validate_raw_event_cross_binding(
            self.model_input_event,
            raw,
            source_p0_materialization_receipt_sha256=(
                self.source_p0_materialization_receipt_sha256
            ),
        )
        if self.receipt_sha256 != self._compute_sha256():
            raise ValueError("BA-IEG event projection v2 content changed")


def validate_ba_ieg_event_model_input_projection_v2(
    projection: BAIEGEventModelInputProjectionV2,
    *,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
) -> BAIEGEventModelInputProjectionV2:
    """Validate all three projection branches and replay raw support at host."""

    if not isinstance(projection, BAIEGEventModelInputProjectionV2):
        raise TypeError("projection must be BAIEGEventModelInputProjectionV2")
    projection.verify_integrity()
    replayed = validate_ba_ieg_p0_raw_sample_dependency_sidecar_v1(
        projection.raw_sample_dependency_sidecar,
        event_tokens=projection.model_input_event,
        source_p0_materialization_receipt_sha256=(
            projection.source_p0_materialization_receipt_sha256
        ),
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )
    if replayed != projection.raw_sample_dependency_sidecar:
        raise ValueError("projection raw dependencies did not replay exactly")
    return projection


def replay_ba_ieg_event_model_input_projection_v2(
    projection: BAIEGEventModelInputProjectionV2,
    *,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
) -> BAIEGEventModelInputProjectionV2:
    """Explicit host-replay alias for registry boundaries."""

    return validate_ba_ieg_event_model_input_projection_v2(
        projection,
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )


def project_ba_ieg_event_model_input_v2(
    materialization: BAIEGP0MaterializationResult,
    *,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
) -> BAIEGEventModelInputProjectionV2:
    """Detach targets through v1, then bind host-replayed raw dependencies."""

    legacy = project_ba_ieg_event_model_input_v1(materialization)
    raw_sidecar = materialize_ba_ieg_p0_raw_sample_dependency_sidecar_v1(
        legacy.model_input_event,
        source_p0_materialization_receipt_sha256=(
            legacy.source_p0_materialization_receipt_sha256
        ),
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )
    projection = BAIEGEventModelInputProjectionV2(
        model_input_event=legacy.model_input_event,
        deterministic_target_sidecar=legacy.deterministic_target_sidecar,
        raw_sample_dependency_sidecar=raw_sidecar,
        source_p0_materialization_receipt_sha256=(
            legacy.source_p0_materialization_receipt_sha256
        ),
    )
    return validate_ba_ieg_event_model_input_projection_v2(
        projection,
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )


__all__ = [
    "BA_IEG_EVENT_MODEL_INPUT_PROJECTION_METHOD_ID_V2",
    "BA_IEG_EVENT_MODEL_INPUT_PROJECTION_SCHEMA_VERSION_V2",
    "BAIEGEventModelInputProjectionV2",
    "project_ba_ieg_event_model_input_v2",
    "replay_ba_ieg_event_model_input_projection_v2",
    "validate_ba_ieg_event_model_input_projection_v2",
]
