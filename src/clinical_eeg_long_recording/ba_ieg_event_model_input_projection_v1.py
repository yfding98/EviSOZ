"""Detach P0 deterministic supervision from BA-IEG model inputs.

The P0 materializer intentionally emits one self-auditing object with the
signal-derived dense measurement targets attached.  Segmental disk inputs,
however, are target-free artifacts.  This module is the explicit boundary
between those two contracts: it preserves the registered event-input receipt
bit-for-bit and moves deterministic supervision into a separately
content-bound sidecar.

The projection accepts only a validated P0 materialization result.  It cannot
read EDF annotations, spreadsheets, clinical text, private doctor labels, or
public interval targets.  The detached sidecar is supervision-only and is not
accepted by the model-input disk exporter or model forward path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from typing import Final

import torch

from .ba_ieg_dense_measurement_sidecar import (
    BA_IEG_DENSE_MEASUREMENT_SIDECAR_SCHEMA_VERSION,
)
from .ba_ieg_training_contract import (
    BA_IEG_ALLOWED_SPLITS,
    BAIEGDeterministicTargets,
    BAIEGEventTokens,
    BAIEGP0MaterializationResult,
)


BA_IEG_DETERMINISTIC_TARGET_SIDECAR_SCHEMA_VERSION: Final[
    str
] = "ba_ieg_content_bound_deterministic_target_sidecar_v1"
BA_IEG_EVENT_MODEL_INPUT_PROJECTION_SCHEMA_VERSION: Final[
    str
] = "ba_ieg_event_model_input_projection_v1"
_SHA256_CHARACTERS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_TARGET_SCOPE_RECEIPT: Final[dict[str, bool]] = {
    "eeg_signal_derived_measurement_supervision_only": True,
    "available_to_model_forward": False,
    "public_interval_target_used": False,
    "edf_annotation_used": False,
    "spreadsheet_used": False,
    "private_doctor_label_used": False,
    "clinical_text_used": False,
}
_PROJECTION_SCOPE_RECEIPT: Final[dict[str, bool]] = {
    "input_receipt_preserved": True,
    "deterministic_target_detached": True,
    "target_available_to_model_forward": False,
    "source_split_preserved": True,
    "disk_input_target_free": True,
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


def _identifier(value: object, name: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    return text


def _sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in _SHA256_CHARACTERS for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


@dataclass(frozen=True)
class BAIEGContentBoundDeterministicTargetSidecarV1:
    """Signal-derived supervision bound to one immutable model input.

    ``target_receipt_sha256`` is incorporated transitively through
    :attr:`receipt_sha256`; changing any target value, mask, coordinate, policy,
    or source binding therefore changes or invalidates this sidecar.  The
    source event input receipt is stored independently so the target cannot be
    rebound to another event that happens to share its tensor dimensions.
    """

    event_id: str
    recording_id: str
    patient_uid: str
    model_split: str
    source_event_input_receipt_sha256: str
    source_p0_materialization_receipt_sha256: str
    dense_measurement_sidecar_receipt_sha256: str
    dense_measurement_source_binding_sha256: str
    feature_scope_sha256: str
    targets: BAIEGDeterministicTargets = field(repr=False, compare=False)
    schema_version: str = BA_IEG_DETERMINISTIC_TARGET_SIDECAR_SCHEMA_VERSION
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("event_id", "recording_id", "patient_uid"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.model_split not in BA_IEG_ALLOWED_SPLITS:
            raise ValueError("deterministic target sidecar model_split is unsupported")
        if self.schema_version != BA_IEG_DETERMINISTIC_TARGET_SIDECAR_SCHEMA_VERSION:
            raise ValueError("deterministic target sidecar schema drifted")
        for name in (
            "source_event_input_receipt_sha256",
            "source_p0_materialization_receipt_sha256",
            "dense_measurement_sidecar_receipt_sha256",
            "dense_measurement_source_binding_sha256",
            "feature_scope_sha256",
        ):
            _sha256(getattr(self, name), name)
        if not isinstance(self.targets, BAIEGDeterministicTargets):
            raise TypeError("deterministic target sidecar requires registered targets")
        self.targets.verify_integrity()
        if (
            self.targets.source_binding_sha256
            != self.dense_measurement_source_binding_sha256
        ):
            raise ValueError("deterministic targets changed source binding")
        object.__setattr__(self, "receipt_sha256", self._compute_sha256())

    @property
    def target_receipt_sha256(self) -> str:
        return self.targets.receipt_sha256

    @property
    def scope_receipt(self) -> dict[str, bool]:
        return dict(_TARGET_SCOPE_RECEIPT)

    def _compute_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "identity": {
                    "event_id": self.event_id,
                    "recording_id": self.recording_id,
                    "patient_uid": self.patient_uid,
                    "model_split": self.model_split,
                },
                "binding": {
                    "source_event_input_receipt_sha256": (
                        self.source_event_input_receipt_sha256
                    ),
                    "source_p0_materialization_receipt_sha256": (
                        self.source_p0_materialization_receipt_sha256
                    ),
                    "dense_measurement_sidecar_receipt_sha256": (
                        self.dense_measurement_sidecar_receipt_sha256
                    ),
                    "dense_measurement_source_binding_sha256": (
                        self.dense_measurement_source_binding_sha256
                    ),
                    "feature_scope_sha256": self.feature_scope_sha256,
                    "target_receipt_sha256": self.targets.receipt_sha256,
                },
                "scope_receipt": _TARGET_SCOPE_RECEIPT,
            }
        )

    def verify_integrity(self) -> None:
        self.targets.verify_integrity()
        if (
            self.targets.source_binding_sha256
            != self.dense_measurement_source_binding_sha256
        ):
            raise ValueError("deterministic target sidecar source binding drifted")
        if self.receipt_sha256 != self._compute_sha256():
            raise ValueError("deterministic target sidecar content changed")


@dataclass(frozen=True)
class BAIEGEventModelInputProjectionV1:
    """One target-free event plus its separately bound supervision sidecar."""

    model_input_event: BAIEGEventTokens
    deterministic_target_sidecar: BAIEGContentBoundDeterministicTargetSidecarV1
    source_p0_materialization_receipt_sha256: str
    schema_version: str = BA_IEG_EVENT_MODEL_INPUT_PROJECTION_SCHEMA_VERSION
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model_input_event, BAIEGEventTokens):
            raise TypeError("projection requires a registered BA-IEG event input")
        if not isinstance(
            self.deterministic_target_sidecar,
            BAIEGContentBoundDeterministicTargetSidecarV1,
        ):
            raise TypeError("projection requires a registered target sidecar")
        if self.schema_version != BA_IEG_EVENT_MODEL_INPUT_PROJECTION_SCHEMA_VERSION:
            raise ValueError("BA-IEG event model-input projection schema drifted")
        _sha256(
            self.source_p0_materialization_receipt_sha256,
            "source_p0_materialization_receipt_sha256",
        )
        event = self.model_input_event
        sidecar = self.deterministic_target_sidecar
        event.verify_integrity()
        sidecar.verify_integrity()
        if event.deterministic_targets is not None:
            raise ValueError(
                "projected model input still contains deterministic targets"
            )
        if (
            sidecar.event_id != event.event_id
            or sidecar.recording_id != event.recording_id
            or sidecar.patient_uid != event.patient_uid
            or sidecar.model_split != event.model_split
            or sidecar.source_event_input_receipt_sha256 != event.input_receipt_sha256
            or sidecar.source_p0_materialization_receipt_sha256
            != self.source_p0_materialization_receipt_sha256
            or sidecar.feature_scope_sha256 != event.feature_scope.sha256
        ):
            raise ValueError(
                "projected input and deterministic sidecar binding drifted"
            )
        targets = sidecar.targets
        if (
            int(targets.row_unit_index.max()) >= len(event.unit_ids)
            or int(targets.row_view_index.max()) >= len(event.view_ids)
            or not torch.equal(
                targets.row_view_index,
                event.unit_view_index[targets.row_unit_index],
            )
        ):
            raise ValueError("deterministic sidecar coordinates do not bind to input")
        target_times = targets.row_time_bounds_seconds
        interval_start, interval_stop = event.analysis_interval_seconds
        if (
            torch.any(target_times[:, 0] < interval_start - 1e-6)
            or torch.any(target_times[:, 1] > interval_stop + 1e-6)
            or not math.isfinite(interval_start)
            or not math.isfinite(interval_stop)
        ):
            raise ValueError("deterministic sidecar time support exceeds input event")
        object.__setattr__(self, "receipt_sha256", self._compute_sha256())

    @property
    def scope_receipt(self) -> dict[str, bool]:
        return dict(_PROJECTION_SCOPE_RECEIPT)

    def _compute_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "source_p0_materialization_receipt_sha256": (
                    self.source_p0_materialization_receipt_sha256
                ),
                "model_input_event_receipt_sha256": (
                    self.model_input_event.input_receipt_sha256
                ),
                "deterministic_target_sidecar_receipt_sha256": (
                    self.deterministic_target_sidecar.receipt_sha256
                ),
                "scope_receipt": _PROJECTION_SCOPE_RECEIPT,
            }
        )

    def verify_integrity(self) -> None:
        self.model_input_event.verify_integrity()
        self.deterministic_target_sidecar.verify_integrity()
        if self.model_input_event.deterministic_targets is not None:
            raise ValueError("projected model input acquired deterministic targets")
        if self.receipt_sha256 != self._compute_sha256():
            raise ValueError("BA-IEG event model-input projection content changed")


def project_ba_ieg_event_model_input_v1(
    materialization: BAIEGP0MaterializationResult,
) -> BAIEGEventModelInputProjectionV1:
    """Split one successful P0 result into model input and supervision.

    The returned ``model_input_event`` has exactly the same
    ``input_receipt_sha256`` as the materialized event because deterministic
    supervision is deliberately outside that hash domain.  Failed P0 results
    and legacy target-less results fail closed instead of manufacturing an
    empty supervision object.
    """

    if not isinstance(materialization, BAIEGP0MaterializationResult):
        raise TypeError("P0 model-input projection requires a materialization result")
    source_event = materialization.event_tokens
    receipt = materialization.receipt
    if receipt["status"] != "materialized" or source_event is None:
        raise ValueError("only a successful P0 materialization can be projected")
    source_event.verify_integrity()
    targets = source_event.deterministic_targets
    if targets is None:
        raise ValueError("P0 materialization has no deterministic target sidecar")
    token_receipt = receipt["tokens"]
    if (
        token_receipt.get("deterministic_targets_attached") is not True
        or token_receipt.get("deterministic_target_source")
        != BA_IEG_DENSE_MEASUREMENT_SIDECAR_SCHEMA_VERSION
        or token_receipt.get("dense_measurement_sidecar_receipt_sha256")
        != receipt["lineage"].get("dense_measurement_sidecar_receipt_sha256")
        or token_receipt.get("deterministic_target_receipt_sha256")
        != targets.receipt_sha256
        or token_receipt.get("dense_measurement_source_binding_sha256")
        != targets.source_binding_sha256
        or receipt["lineage"].get("dense_measurement_source_binding_sha256")
        != targets.source_binding_sha256
    ):
        raise ValueError("P0 receipt does not bind the attached deterministic targets")

    model_input_event = replace(source_event, deterministic_targets=None)
    if model_input_event.input_receipt_sha256 != source_event.input_receipt_sha256:
        raise RuntimeError("target detachment changed the registered model input")
    sidecar = BAIEGContentBoundDeterministicTargetSidecarV1(
        event_id=source_event.event_id,
        recording_id=source_event.recording_id,
        patient_uid=source_event.patient_uid,
        model_split=source_event.model_split,
        source_event_input_receipt_sha256=source_event.input_receipt_sha256,
        source_p0_materialization_receipt_sha256=receipt["receipt_sha256"],
        dense_measurement_sidecar_receipt_sha256=token_receipt[
            "dense_measurement_sidecar_receipt_sha256"
        ],
        dense_measurement_source_binding_sha256=targets.source_binding_sha256,
        feature_scope_sha256=source_event.feature_scope.sha256,
        targets=targets,
    )
    return BAIEGEventModelInputProjectionV1(
        model_input_event=model_input_event,
        deterministic_target_sidecar=sidecar,
        source_p0_materialization_receipt_sha256=receipt["receipt_sha256"],
    )


__all__ = [
    "BA_IEG_DETERMINISTIC_TARGET_SIDECAR_SCHEMA_VERSION",
    "BA_IEG_EVENT_MODEL_INPUT_PROJECTION_SCHEMA_VERSION",
    "BAIEGContentBoundDeterministicTargetSidecarV1",
    "BAIEGEventModelInputProjectionV1",
    "project_ba_ieg_event_model_input_v1",
]
