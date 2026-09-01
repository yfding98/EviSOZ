"""Target-free C-CAR19/C-REF19 disagreement receipts for reporting.

This module is deliberately downstream of the frozen LaBraM reference audit.
It consumes only the audit's same-event representation statistics and never
loads a localization target, trains a model, or selects a preprocessing arm.
The resulting scalar is a normalized representation distance.  It is not a
SOZ probability, a localization-performance estimate, or an arm-selection
criterion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

from .preprocessing_arm_runtime import (
    CAUSAL_REFERENCE_PAIR_ROLE,
    CAUSAL_REFERENCE_PAIR_SCHEMA,
    CAUSAL_REFERENCE_SENSITIVITY_ARM_ID,
)


REFERENCE_AUDIT_SCHEMA = "soz_labram_reference_robustness_source_only_v1"
REFERENCE_AUDIT_STATUS = "target_free_reference_robustness_audit_only"
REFERENCE_DISAGREEMENT_RECEIPT_SCHEMA = "soz_reference_disagreement_receipt_v1"
REFERENCE_DISAGREEMENT_METRIC_ID = (
    "labram_block9_standard19_channel_mean_normalized_cosine_distance_"
    "patient_equal_event_mean_v1"
)
REFERENCE_DISAGREEMENT_SCOPE = (
    "frozen_labram_block9_node_representation_not_localizer_output"
)
REFERENCE_DISAGREEMENT_USE_POLICY = (
    "reference_robustness_and_abstention_only_not_model_or_arm_selection"
)
PRIMARY_REFERENCE_ARM_ID = "C-CAR19"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_TRUE_ACCESS_FIELDS = (
    "deepsoz_target_values_loaded",
    "tusz_native_target_values_loaded",
    "private_target_values_loaded",
    "private_eeg_loaded",
    "training_performed",
    "model_selection_performed",
)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ReferenceDisagreementReceipt:
    """Auditable patient-level reference-sensitivity measurement.

    Each event contributes ``(1 - cosine_mean) / 2`` over its 19 frozen
    LaBraM block-9 physical-node representations; ``montage_disagreement`` is
    the equal-event patient mean.  The transformation maps cosine similarity
    from ``[-1, 1]`` to a distance in ``[0, 1]`` without implying calibration
    or clinical probability.  Its event roster must exactly equal the patient
    ranking's aggregation roster before it can enter a report.
    """

    patient_pseudonym: str
    aggregation_event_ids: tuple[str, ...]
    event_signal_artifact_sha256s: tuple[tuple[str, str], ...]
    source_audit_artifact_sha256: str
    labram_checkpoint_sha256: str
    labram_modeling_sha256: str
    reference_pair_schema_version: str
    reference_pair_role: str
    primary_arm_id: str
    sensitivity_arm_id: str
    channel_cosine_means: tuple[float, ...]
    montage_disagreement: float
    shared_filter_resample_crop: bool
    target_values_loaded: bool
    private_data_loaded: bool
    training_performed: bool
    model_selection_performed: bool
    metric_id: str = REFERENCE_DISAGREEMENT_METRIC_ID
    measurement_scope: str = REFERENCE_DISAGREEMENT_SCOPE
    use_policy: str = REFERENCE_DISAGREEMENT_USE_POLICY
    schema_version: str = REFERENCE_DISAGREEMENT_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _require_text(self.patient_pseudonym, name="patient_pseudonym")
        if (
            not isinstance(self.aggregation_event_ids, tuple)
            or not self.aggregation_event_ids
            or len(set(self.aggregation_event_ids)) != len(self.aggregation_event_ids)
        ):
            raise ValueError("aggregation_event_ids must be non-empty and unique")
        for value in self.aggregation_event_ids:
            _require_text(value, name="aggregation_event_id")
        if (
            not isinstance(self.event_signal_artifact_sha256s, tuple)
            or len(self.event_signal_artifact_sha256s)
            != len(self.aggregation_event_ids)
        ):
            raise ValueError("event signal receipts must match aggregation events")
        signal_event_ids: list[str] = []
        for row in self.event_signal_artifact_sha256s:
            if not isinstance(row, tuple) or len(row) != 2:
                raise ValueError("event signal receipt must be an (event, sha256) tuple")
            event_id, signal_sha = row
            signal_event_ids.append(_require_text(event_id, name="signal event_id"))
            _require_sha256(signal_sha, name="signal_artifact_sha256")
        if tuple(signal_event_ids) != self.aggregation_event_ids:
            raise ValueError("event signal receipts must preserve aggregation order")
        for name in (
            "source_audit_artifact_sha256",
            "labram_checkpoint_sha256",
            "labram_modeling_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if self.reference_pair_schema_version != CAUSAL_REFERENCE_PAIR_SCHEMA:
            raise ValueError("Reference disagreement uses another pair schema")
        if self.reference_pair_role != CAUSAL_REFERENCE_PAIR_ROLE:
            raise ValueError("Reference disagreement uses another pair role")
        if self.primary_arm_id != PRIMARY_REFERENCE_ARM_ID:
            raise ValueError("Reference disagreement primary arm must be C-CAR19")
        if self.sensitivity_arm_id != CAUSAL_REFERENCE_SENSITIVITY_ARM_ID:
            raise ValueError("Reference disagreement sensitivity arm must be C-REF19")
        if (
            not isinstance(self.channel_cosine_means, tuple)
            or len(self.channel_cosine_means) != len(self.aggregation_event_ids)
        ):
            raise ValueError("channel cosine means must match aggregation events")
        cosines = tuple(
            _require_finite(value, name="channel_cosine_mean")
            for value in self.channel_cosine_means
        )
        disagreement = _require_finite(
            self.montage_disagreement, name="montage_disagreement"
        )
        if any(value < -1 or value > 1 for value in cosines):
            raise ValueError("channel_cosine_mean must lie in [-1,1]")
        if disagreement < 0 or disagreement > 1:
            raise ValueError("montage_disagreement must lie in [0,1]")
        expected = math.fsum((1.0 - value) / 2.0 for value in cosines) / len(
            cosines
        )
        if not math.isclose(disagreement, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("montage_disagreement does not replay from cosine mean")
        for name in (
            "shared_filter_resample_crop",
            "target_values_loaded",
            "private_data_loaded",
            "training_performed",
            "model_selection_performed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if not self.shared_filter_resample_crop:
            raise ValueError("Reference disagreement requires shared preprocessing")
        if any(
            (
                self.target_values_loaded,
                self.private_data_loaded,
                self.training_performed,
                self.model_selection_performed,
            )
        ):
            raise ValueError(
                "Reference disagreement must remain target-free and selection-free"
            )
        if self.metric_id != REFERENCE_DISAGREEMENT_METRIC_ID:
            raise ValueError("Unsupported reference-disagreement metric")
        if self.measurement_scope != REFERENCE_DISAGREEMENT_SCOPE:
            raise ValueError("Unsupported reference-disagreement scope")
        if self.use_policy != REFERENCE_DISAGREEMENT_USE_POLICY:
            raise ValueError("Unsupported reference-disagreement use policy")
        if self.schema_version != REFERENCE_DISAGREEMENT_RECEIPT_SCHEMA:
            raise ValueError("Unsupported reference-disagreement receipt schema")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))

    def signal_artifact_sha256_for_event(self, event_id: str) -> str:
        """Return the signal digest for one exactly aggregated event."""

        requested = _require_text(event_id, name="event_id")
        matches = tuple(
            signal_sha
            for bound_event, signal_sha in self.event_signal_artifact_sha256s
            if bound_event == requested
        )
        if len(matches) != 1:
            raise ValueError("Event is absent from reference-disagreement aggregation")
        return matches[0]


def build_reference_disagreement_receipt(
    audit_payload: Mapping[str, object],
    *,
    source_audit_artifact_sha256: str,
    patient_pseudonym: str,
    aggregation_event_ids: Sequence[str],
) -> ReferenceDisagreementReceipt:
    """Bind one patient's exact aggregation roster from a source-only audit.

    Selection is by exact patient/event pairs.  The producer fails closed if
    the audit declares any target/private access, training, model selection,
    different preprocessing, or anything other than one row per event.
    """

    payload = _require_mapping(audit_payload, name="audit_payload")
    if payload.get("schema_version") != REFERENCE_AUDIT_SCHEMA:
        raise ValueError("Unsupported LaBraM reference-audit schema")
    if payload.get("status") != REFERENCE_AUDIT_STATUS:
        raise ValueError("Reference audit is not a completed audit-only artifact")
    source_sha = _require_sha256(
        source_audit_artifact_sha256, name="source_audit_artifact_sha256"
    )
    patient = _require_text(patient_pseudonym, name="patient_pseudonym")
    if isinstance(aggregation_event_ids, (str, bytes)):
        raise TypeError("aggregation_event_ids must be a sequence of event IDs")
    event_ids = tuple(
        _require_text(value, name="aggregation_event_id")
        for value in aggregation_event_ids
    )
    if not event_ids or len(set(event_ids)) != len(event_ids):
        raise ValueError("aggregation_event_ids must be non-empty and unique")

    pair = _require_mapping(payload.get("reference_pair"), name="reference_pair")
    expected_pair = {
        "schema_version": CAUSAL_REFERENCE_PAIR_SCHEMA,
        "role": CAUSAL_REFERENCE_PAIR_ROLE,
        "primary": PRIMARY_REFERENCE_ARM_ID,
        "sensitivity": CAUSAL_REFERENCE_SENSITIVITY_ARM_ID,
        "shared_filter_resample_crop": True,
    }
    for name, expected in expected_pair.items():
        if pair.get(name) != expected:
            raise ValueError(f"Reference audit pair contract mismatch: {name}")

    access = _require_mapping(payload.get("access_receipt"), name="access_receipt")
    for name in _FORBIDDEN_TRUE_ACCESS_FIELDS:
        value = access.get(name)
        if type(value) is not bool:
            raise TypeError(f"access_receipt.{name} must be bool")
        if value:
            raise ValueError(
                f"Reference audit is not target/private/selection free: {name}"
            )

    labram = _require_mapping(payload.get("labram_receipt"), name="labram_receipt")
    checkpoint_sha = _require_sha256(
        labram.get("checkpoint_sha256"), name="labram_checkpoint_sha256"
    )
    modeling_sha = _require_sha256(
        labram.get("modeling_sha256"), name="labram_modeling_sha256"
    )
    semantic_channels = labram.get("semantic_channels")
    if (
        not isinstance(semantic_channels, Sequence)
        or isinstance(semantic_channels, (str, bytes))
        or len(semantic_channels) != 19
        or len(set(str(value) for value in semantic_channels)) != 19
    ):
        raise ValueError("Reference audit must bind 19 unique semantic channels")

    rows = payload.get("events")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("Reference audit events must be a sequence")
    bound_rows: list[Mapping[str, object]] = []
    for event_id in event_ids:
        matches = tuple(
            row
            for row in rows
            if isinstance(row, Mapping)
            and row.get("patient_id") == patient
            and row.get("event_id") == event_id
        )
        if len(matches) != 1:
            raise ValueError(
                "Reference audit must contain exactly one row per aggregation event"
            )
        bound_rows.append(matches[0])

    signal_receipts: list[tuple[str, str]] = []
    cosine_means: list[float] = []
    for event_id, row in zip(event_ids, bound_rows):
        edf = _require_mapping(
            row.get("edf_preprocess_receipt"), name="edf_preprocess_receipt"
        )
        signal_sha = _require_sha256(edf.get("edf_sha256"), name="edf_sha256")
        metrics = _require_mapping(row.get("metrics"), name="event.metrics")
        cosine_summary = _require_mapping(
            metrics.get("block9_channel_cosine"), name="block9_channel_cosine"
        )
        cosine_mean = _require_finite(
            cosine_summary.get("mean"), name="block9_channel_cosine.mean"
        )
        if cosine_mean < -1 or cosine_mean > 1:
            raise ValueError("block9 channel cosine mean must lie in [-1,1]")
        car_replay_error = _require_finite(
            metrics.get("car_replay_max_abs_volts"),
            name="car_replay_max_abs_volts",
        )
        bipolar_error = _require_finite(
            metrics.get("bipolar_reference_invariance_max_abs_volts"),
            name="bipolar_reference_invariance_max_abs_volts",
        )
        if car_replay_error < 0 or car_replay_error > 5e-12:
            raise ValueError("Reference audit fails CAR algebraic replay tolerance")
        if bipolar_error < 0 or bipolar_error > 1e-8:
            raise ValueError("Reference audit fails bipolar-invariance tolerance")
        signal_receipts.append((event_id, signal_sha))
        cosine_means.append(cosine_mean)

    disagreement = math.fsum(
        (1.0 - value) / 2.0 for value in cosine_means
    ) / len(cosine_means)
    return ReferenceDisagreementReceipt(
        patient_pseudonym=patient,
        aggregation_event_ids=event_ids,
        event_signal_artifact_sha256s=tuple(signal_receipts),
        source_audit_artifact_sha256=source_sha,
        labram_checkpoint_sha256=checkpoint_sha,
        labram_modeling_sha256=modeling_sha,
        reference_pair_schema_version=CAUSAL_REFERENCE_PAIR_SCHEMA,
        reference_pair_role=CAUSAL_REFERENCE_PAIR_ROLE,
        primary_arm_id=PRIMARY_REFERENCE_ARM_ID,
        sensitivity_arm_id=CAUSAL_REFERENCE_SENSITIVITY_ARM_ID,
        channel_cosine_means=tuple(cosine_means),
        montage_disagreement=disagreement,
        shared_filter_resample_crop=True,
        target_values_loaded=False,
        private_data_loaded=False,
        training_performed=False,
        model_selection_performed=False,
    )


def load_reference_disagreement_receipt(
    audit_path: str | Path,
    *,
    patient_pseudonym: str,
    aggregation_event_ids: Sequence[str],
) -> ReferenceDisagreementReceipt:
    """Load an aggregation receipt without reading EEG or target artifacts."""

    source = Path(audit_path).resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError("Reference audit must be a canonical regular file")
    raw = source.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise TypeError("Reference audit JSON must contain one object")
    return build_reference_disagreement_receipt(
        payload,
        source_audit_artifact_sha256=hashlib.sha256(raw).hexdigest(),
        patient_pseudonym=patient_pseudonym,
        aggregation_event_ids=aggregation_event_ids,
    )


__all__ = [
    "PRIMARY_REFERENCE_ARM_ID",
    "REFERENCE_AUDIT_SCHEMA",
    "REFERENCE_AUDIT_STATUS",
    "REFERENCE_DISAGREEMENT_METRIC_ID",
    "REFERENCE_DISAGREEMENT_RECEIPT_SCHEMA",
    "REFERENCE_DISAGREEMENT_SCOPE",
    "REFERENCE_DISAGREEMENT_USE_POLICY",
    "ReferenceDisagreementReceipt",
    "build_reference_disagreement_receipt",
    "load_reference_disagreement_receipt",
]
