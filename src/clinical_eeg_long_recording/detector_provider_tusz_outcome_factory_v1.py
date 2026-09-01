"""Real EEG-only TUSZ outcome factory for the frozen detector variants.

The compact pre-reference inventory deliberately accepts a very narrow
``(source_record, variant_id)`` callback.  This module is the production
implementation of that callback for TUSZ EDF files.  It opens one canonical
EEG record, replays its physical signal identity against the target-free
canonical audit/projection anchor, and evaluates the four frozen provider
variants in record-major order::

    EventNet EN19 -> EventNet EN17 -> SeizureTransformer ST18 -> ST16

The same in-memory canonical record is reused for all four calls and released
after ST16.  No seizure interval, EDF annotation, spreadsheet, doctor text,
clinical history, behaviour/video, sleep label, or auxiliary channel has an
input slot.  Unsupported channel/clock/length/transform conditions remain the
typed exclusions produced by the provider registries; source-integrity or
authority mismatches fail closed.

This factory establishes real transform eligibility only.  It does not train
a detector, open a reference phase, issue a checkpoint, estimate performance,
or authorize a clinical claim.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np

from src.soz.geometry import STANDARD_19

from .canonical_edf_materialization import (
    CanonicalEEGRecord,
    load_canonical_edf_record,
)
from .detector_provider_pre_reference_inventory_v1 import (
    PROVIDER_VARIANTS_V1,
    TargetFreeProviderSourceRecordV1,
    _compact_outcome,
)
from .detector_signal_lineage_authority_v1 import (
    CanonicalPolicyAuditTrustAnchor,
    ValidatedDetectorSignalLineageAuthority,
    authorize_detector_policy_lineage_from_canonical_audit,
    authorize_detector_signal_lineage_from_canonical_record,
    load_canonical_policy_audit_trust_anchor,
    require_validated_detector_signal_lineage_authority,
)
from . import eventnet_cleanroom_registry_v1 as _eventnet
from . import seizuretransformer_cleanroom_registry_v1 as _st
from .tusz_canonical_physical_signal_audit_v1 import (
    validate_tusz_canonical_physical_duplicate_audit_v1,
)
from .tusz_detector_cleanroom_fold_plan_v1 import (
    validate_tusz_detector_cleanroom_fold_plan_v1,
)


SCHEMA_VERSION: Final[str] = "clinical_eeg_tusz_target_free_provider_outcome_factory_v1"
METHOD_ID: Final[
    str
] = "one_canonical_EDF_session_four_frozen_provider_outcomes_EEG_only_v1"
SMOKE_SCHEMA_VERSION: Final[
    str
] = "clinical_eeg_tusz_target_free_provider_record_smoke_v1"
SMOKE_METHOD_ID: Final[
    str
] = "one_real_source_train_record_four_variant_live_transform_replay_v1"
_PENDING: Final[str] = "CONTENT-ADDRESS-PENDING"
_SHA256_CHARS: Final[frozenset[str]] = frozenset("0123456789abcdef")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    if result.get("receipt_sha256") != _PENDING:
        raise ValueError("content-addressed receipt must begin pending")
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def _require_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value).difference(_SHA256_CHARS)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def detector_provider_tusz_outcome_factory_source_sha256_v1() -> str:
    digest = hashlib.sha256()
    with Path(__file__).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_source_path(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or "\\" in relative_path
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != "train"
        or relative.suffix.lower() != ".edf"
    ):
        raise PermissionError("provider outcome factory accepts source-train EDF only")
    unresolved = root / relative
    if unresolved.is_symlink():
        raise ValueError("provider source EDF must not be a symbolic link")
    source = unresolved.resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError as error:
        raise PermissionError("provider source path escapes the TUSZ root") from error
    if not source.is_file() or source.suffix.lower() != ".edf":
        raise ValueError("provider source is not a regular EDF")
    return source


def _validate_source_record(
    value: object,
) -> TargetFreeProviderSourceRecordV1:
    if not isinstance(value, TargetFreeProviderSourceRecordV1):
        raise TypeError("factory requires TargetFreeProviderSourceRecordV1")
    if (
        not value.analysis_identity_id
        or value.analysis_identity_id != value.analysis_identity_id.strip()
        or not value.source_edf_relative_path
        or value.source_edf_relative_path != value.source_edf_relative_path.strip()
    ):
        raise ValueError("target-free source record identifiers are invalid")
    fraction = value.recording_duration_seconds_fraction
    if (
        type(fraction) is not tuple
        or len(fraction) != 2
        or type(fraction[0]) is not int
        or type(fraction[1]) is not int
        or fraction[0] <= 0
        or fraction[1] <= 0
    ):
        raise ValueError("target-free source duration fraction is invalid")
    return value


@dataclass(slots=True)
class _LiveTUSZProviderRecordSessionV1:
    source_record: TargetFreeProviderSourceRecordV1
    canonical_record: CanonicalEEGRecord
    referential_volts: np.ndarray
    provider_authority: ValidatedDetectorSignalLineageAuthority
    identity_authority: ValidatedDetectorSignalLineageAuthority


class TUSZTargetFreeProviderOutcomeFactoryV1:
    """Stateful, strict record-major callback for the four provider variants."""

    __slots__ = (
        "_tusz_root",
        "_anchor",
        "_eventnet_registry",
        "_st_registry",
        "_st_runtime_receipt",
        "_session",
        "_next_variant_index",
        "_source_record_session_open_count",
        "_source_record_session_release_count",
        "_completed_four_variant_record_count",
        "_aborted_record_session_count",
        "_variant_outcome_count",
        "_closed",
    )

    def __init__(
        self,
        *,
        tusz_root: str | Path,
        canonical_audit_bytes: bytes,
        physical_projection_bytes: bytes,
        eventnet_registry: Mapping[str, Any],
        seizuretransformer_registry: Mapping[str, Any],
    ) -> None:
        root_input = Path(tusz_root)
        if root_input.is_symlink():
            raise ValueError("TUSZ root must not be a symbolic link")
        root = root_input.resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(root)
        self._tusz_root = root
        self._anchor = load_canonical_policy_audit_trust_anchor(
            audit_bytes=canonical_audit_bytes,
            projection_bytes=physical_projection_bytes,
        )
        self._eventnet_registry = _eventnet._require_canonical_eventnet_registry(
            eventnet_registry
        )
        self._st_registry = _st._require_canonical_seizuretransformer_registry(
            seizuretransformer_registry
        )
        # Fail once at factory construction if the numerical runtime drifted.
        # Otherwise ST's narrow per-record transform catch would incorrectly
        # turn a global environment failure into thousands of typed records.
        self._st_runtime_receipt = _st.validate_runtime_environment(
            self._st_registry, training=False
        )
        self._session: _LiveTUSZProviderRecordSessionV1 | None = None
        self._next_variant_index = 0
        self._source_record_session_open_count = 0
        self._source_record_session_release_count = 0
        self._completed_four_variant_record_count = 0
        self._aborted_record_session_count = 0
        self._variant_outcome_count = 0
        self._closed = False

    @property
    def eventnet_registry(self) -> dict[str, Any]:
        return deepcopy(self._eventnet_registry)

    @property
    def seizuretransformer_registry(self) -> dict[str, Any]:
        return deepcopy(self._st_registry)

    def _open_session(
        self, source_record: TargetFreeProviderSourceRecordV1
    ) -> _LiveTUSZProviderRecordSessionV1:
        source = _safe_source_path(
            self._tusz_root, source_record.source_edf_relative_path
        )
        canonical_record = load_canonical_edf_record(source)
        provider_authority = authorize_detector_signal_lineage_from_canonical_record(
            canonical_record
        )
        identity_authority = authorize_detector_policy_lineage_from_canonical_audit(
            self._anchor,
            analysis_identity_id=source_record.analysis_identity_id,
        )
        provider = require_validated_detector_signal_lineage_authority(
            provider_authority
        )
        identity = require_validated_detector_signal_lineage_authority(
            identity_authority
        )
        provider_signal = provider["canonical_physical_signal"]
        identity_signal = identity["canonical_physical_signal"]
        for field in (
            "source_header_receipt_sha256",
            "source_signal_sha256",
            "source_tensor_sha256",
        ):
            if provider_signal[field] != identity_signal[field]:
                raise PermissionError(
                    "source EDF payload disagrees with canonical audit identity"
                )
        clock = provider["common_sampling_clock_authority"]
        actual_duration = Fraction(
            int(clock["sample_count"]) * int(clock["sampling_rate_fraction_hz"][1]),
            int(clock["sampling_rate_fraction_hz"][0]),
        )
        declared_duration = Fraction(*source_record.recording_duration_seconds_fraction)
        if actual_duration != declared_duration:
            raise PermissionError(
                "source EDF clock disagrees with frozen source-record duration"
            )
        referential_volts = np.asarray(
            canonical_record.observed_signal_volts.detach().cpu().numpy()
        )
        if (
            referential_volts.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or referential_volts.ndim != 2
            or not referential_volts.flags.c_contiguous
        ):
            raise ValueError("canonical provider carrier is not contiguous volts")
        self._source_record_session_open_count += 1
        return _LiveTUSZProviderRecordSessionV1(
            source_record=source_record,
            canonical_record=canonical_record,
            referential_volts=referential_volts,
            provider_authority=provider_authority,
            identity_authority=identity_authority,
        )

    def _release_session(self, *, completed: bool) -> None:
        if self._session is None:
            return
        self._session = None
        self._next_variant_index = 0
        self._source_record_session_release_count += 1
        if completed:
            self._completed_four_variant_record_count += 1
        else:
            self._aborted_record_session_count += 1

    def __call__(
        self, source_record: TargetFreeProviderSourceRecordV1, variant_id: str
    ) -> object:
        if self._closed:
            raise RuntimeError("TUSZ provider outcome factory is closed")
        record = _validate_source_record(source_record)
        if variant_id not in PROVIDER_VARIANTS_V1:
            raise ValueError("variant is outside the frozen four-variant roster")
        expected_variant = PROVIDER_VARIANTS_V1[self._next_variant_index]
        if variant_id != expected_variant:
            self._release_session(completed=False)
            raise RuntimeError(
                "provider outcome calls must follow the frozen record-major order"
            )
        if self._session is None:
            if self._next_variant_index != 0:
                raise AssertionError("factory variant cursor has no live session")
            try:
                self._session = self._open_session(record)
            except Exception:
                self._next_variant_index = 0
                raise
        elif self._session.source_record != record:
            self._release_session(completed=False)
            raise RuntimeError(
                "source record changed before all four variant outcomes completed"
            )

        session = self._session
        assert session is not None
        try:
            if variant_id in {_eventnet.EN19_VARIANT_ID, _eventnet.EN17_VARIANT_ID}:
                outcome = _eventnet.materialize_eventnet_pre_reference_eligibility(
                    session.referential_volts,
                    variant_id=variant_id,
                    signal_lineage_authority=session.provider_authority,
                    record_identity_authority=session.identity_authority,
                    registry=self._eventnet_registry,
                )
            else:
                outcome = _st.materialize_seizuretransformer_pre_reference_eligibility(
                    session.referential_volts,
                    variant_id=variant_id,
                    signal_lineage_authority=session.provider_authority,
                    record_identity_authority=session.identity_authority,
                    registry=self._st_registry,
                )
        except Exception:
            self._release_session(completed=False)
            raise
        self._variant_outcome_count += 1
        self._next_variant_index += 1
        if self._next_variant_index == len(PROVIDER_VARIANTS_V1):
            self._release_session(completed=True)
        return outcome

    def assert_idle(self) -> None:
        if self._session is not None or self._next_variant_index != 0:
            raise RuntimeError("provider factory has an incomplete four-variant record")

    def lifecycle_receipt(self) -> dict[str, Any]:
        return _content_address(
            {
                "schema_version": SCHEMA_VERSION,
                "method_id": METHOD_ID,
                "implementation_source_sha256": (
                    detector_provider_tusz_outcome_factory_source_sha256_v1()
                ),
                "frozen_variant_order": list(PROVIDER_VARIANTS_V1),
                "st_numeric_runtime_receipt": deepcopy(self._st_runtime_receipt),
                "source_record_session_open_count": (
                    self._source_record_session_open_count
                ),
                "source_record_session_release_count": (
                    self._source_record_session_release_count
                ),
                "completed_four_variant_record_count": (
                    self._completed_four_variant_record_count
                ),
                "aborted_record_session_count": self._aborted_record_session_count,
                "variant_outcome_count": self._variant_outcome_count,
                "live_source_record_session_count": int(self._session is not None),
                "maximum_simultaneous_source_record_sessions": 1,
                "same_canonical_EEG_session_reused_across_four_variants": True,
                "full_record_provider_arrays_serialized_by_factory": False,
                "scope_receipt": {
                    "EEG_samples_used": True,
                    "allowlisted_acquisition_clock_and_reference_used": True,
                    "canonical_physical_audit_projection_control_plane_used": True,
                    "seizure_target_or_reference_interval_used": False,
                    "EDF_annotation_used": False,
                    "spreadsheet_or_doctor_text_used": False,
                    "clinical_history_or_behaviour_used": False,
                    "sleep_activation_ECG_EMG_EOG_used": False,
                    "patient_or_subject_identity_used_as_model_feature": False,
                    "source_dev_or_eval_used": False,
                    "repository_bound_factory_implementation_used": True,
                    "OS_file_capability_sandbox_enforced": False,
                    "forbidden_file_open_runtime_trace_complete": False,
                },
                "claim_boundary": {
                    "real_provider_transform_eligibility_may_be_established": True,
                    "source_code_has_forbidden_input_API": False,
                    "absence_of_forbidden_file_opens_proven_by_OS_trace": False,
                    "detector_training_or_checkpoint_established": False,
                    "detection_accuracy_or_efficiency_established": False,
                    "Findings_or_SOZ_performance_established": False,
                    "clinical_use_authorized": False,
                },
                "receipt_sha256": _PENDING,
            }
        )

    def close(self) -> None:
        if not self._closed:
            self._release_session(completed=False)
            self._closed = True

    def __enter__(self) -> "TUSZTargetFreeProviderOutcomeFactoryV1":
        if self._closed:
            raise RuntimeError("TUSZ provider outcome factory is closed")
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def select_source_train_provider_smoke_record_v1(
    *,
    fold_plan: Mapping[str, Any],
    canonical_audit: Mapping[str, Any],
    target_duration_seconds: int = 121,
    support_profile: str = "complete19",
    require_eventnet_training_tile: bool = True,
) -> TargetFreeProviderSourceRecordV1:
    """Select one deterministic, target-free source-train smoke record."""

    if (
        isinstance(target_duration_seconds, bool)
        or not isinstance(target_duration_seconds, int)
        or target_duration_seconds <= 0
    ):
        raise ValueError("smoke target duration must be a positive integer")
    if support_profile not in {"complete19", "lateral17"}:
        raise ValueError("smoke support profile must be complete19 or lateral17")
    plan = validate_tusz_detector_cleanroom_fold_plan_v1(fold_plan)
    audit = validate_tusz_canonical_physical_duplicate_audit_v1(canonical_audit)
    physical_by_identity = {
        str(row["analysis_identity_id"]): row["physical_signal"]
        for row in audit["outcomes"]
        if row["terminal_status"] == "success"
    }
    candidates: list[tuple[Fraction, str, Mapping[str, Any]]] = []
    for row in plan["source_record_duration_rows"]:
        if row["model_split"] != "source_train":
            continue
        identity = str(row["analysis_identity_id"])
        physical = physical_by_identity.get(identity)
        if not isinstance(physical, Mapping):
            raise PermissionError("fold-plan source identity is absent from audit")
        observed = physical.get("observed_channel_ids")
        if not isinstance(observed, list):
            raise ValueError("canonical audit observed-channel roster is malformed")
        expected_observed = (
            set(STANDARD_19)
            if support_profile == "complete19"
            else set(STANDARD_19).difference({"FZ", "PZ"})
        )
        if set(observed) != expected_observed:
            continue
        duration = Fraction(*row["recording_duration_seconds_fraction"])
        minimum_eventnet_duration = Fraction(
            _eventnet.MODEL_INPUT_SAMPLES, _eventnet.TARGET_FS_HZ
        )
        if require_eventnet_training_tile and duration < minimum_eventnet_duration:
            continue
        candidates.append(
            (
                abs(duration - target_duration_seconds),
                identity,
                row,
            )
        )
    if not candidates:
        raise ValueError("no target-free source-train record satisfies smoke support")
    _distance, _identity, selected = min(candidates, key=lambda item: item[:2])
    fraction = selected["recording_duration_seconds_fraction"]
    return TargetFreeProviderSourceRecordV1(
        analysis_identity_id=str(selected["analysis_identity_id"]),
        source_edf_relative_path=str(selected["local_edf_path"]),
        recording_duration_seconds_fraction=(int(fraction[0]), int(fraction[1])),
    )


def materialize_tusz_target_free_provider_record_smoke_v1(
    *,
    factory: TUSZTargetFreeProviderOutcomeFactoryV1,
    source_record: TargetFreeProviderSourceRecordV1,
) -> dict[str, Any]:
    """Replay and compact one real record's four live provider outcomes."""

    if not isinstance(factory, TUSZTargetFreeProviderOutcomeFactoryV1):
        raise TypeError("record smoke requires the real TUSZ outcome factory")
    record = _validate_source_record(source_record)
    rows: list[dict[str, Any]] = []
    eventnet_registry = factory.eventnet_registry
    st_registry = factory.seizuretransformer_registry
    for variant_id in PROVIDER_VARIANTS_V1:
        outcome = factory(record, variant_id)
        compact = _compact_outcome(
            record,
            variant_id=variant_id,
            outcome=outcome,
            eventnet_registry=eventnet_registry,
            seizuretransformer_registry=st_registry,
        )
        rows.append(
            _content_address(
                {
                    "provider_family": compact["provider_family"],
                    "variant_id": compact["variant_id"],
                    "compact_outcome_receipt_sha256": compact["receipt_sha256"],
                    "eligibility_receipt": compact["eligibility_receipt"],
                    "transform_receipt": compact["transform_receipt"],
                    "full_record_array_or_tensor_retained": False,
                    "source_path_or_patient_identifier_emitted": False,
                    "receipt_sha256": _PENDING,
                }
            )
        )
        del outcome
    factory.assert_idle()
    lifecycle = factory.lifecycle_receipt()
    if (
        lifecycle["completed_four_variant_record_count"] != 1
        or lifecycle["variant_outcome_count"] != len(PROVIDER_VARIANTS_V1)
        or lifecycle["source_record_session_open_count"] != 1
        or lifecycle["source_record_session_release_count"] != 1
        or lifecycle["live_source_record_session_count"] != 0
    ):
        raise RuntimeError("one-record smoke did not reuse and release one EDF session")
    status_by_variant = {
        row["variant_id"]: row["eligibility_receipt"]["status"] for row in rows
    }
    result = _content_address(
        {
            "schema_version": SMOKE_SCHEMA_VERSION,
            "method_id": SMOKE_METHOD_ID,
            "factory_lifecycle_receipt": lifecycle,
            "source_record": {
                "analysis_identity_id": record.analysis_identity_id,
                "recording_duration_seconds_fraction": list(
                    record.recording_duration_seconds_fraction
                ),
                "source_path_or_patient_identifier_emitted": False,
            },
            "variant_order": list(PROVIDER_VARIANTS_V1),
            "compact_outcomes": rows,
            "status_by_variant": status_by_variant,
            "all_four_outcomes_accounted_for": True,
            "one_EDF_load_session_reused_for_all_four_variants": True,
            "eligible_transform_payloads_validated_while_live": True,
            "full_record_transform_arrays_serialized": False,
            "scope_receipt": {
                "public_TUSZ_source_train_EEG_only": True,
                "source_dev_or_eval_EEG_used": False,
                "reference_sidecar_or_seizure_interval_used": False,
                "EDF_annotation_used": False,
                "spreadsheet_doctor_text_history_or_behaviour_used": False,
                "auxiliary_non_EEG_channel_used": False,
                "performance_or_clinical_claim_authorized": False,
            },
            "receipt_sha256": _PENDING,
        }
    )
    validate_tusz_target_free_provider_record_smoke_v1(result)
    return result


def validate_tusz_target_free_provider_record_smoke_v1(
    value: object,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("TUSZ provider record smoke must be an object")
    required = {
        "schema_version",
        "method_id",
        "factory_lifecycle_receipt",
        "source_record",
        "variant_order",
        "compact_outcomes",
        "status_by_variant",
        "all_four_outcomes_accounted_for",
        "one_EDF_load_session_reused_for_all_four_variants",
        "eligible_transform_payloads_validated_while_live",
        "full_record_transform_arrays_serialized",
        "scope_receipt",
        "receipt_sha256",
    }
    if set(value) != required:
        raise ValueError("TUSZ provider record smoke fields drifted")
    data = deepcopy(value)
    supplied = _require_sha256(data.get("receipt_sha256"), "smoke receipt")
    data["receipt_sha256"] = _PENDING
    if supplied != _canonical_sha256(data):
        raise ValueError("TUSZ provider record smoke is not content-addressed")
    data["receipt_sha256"] = supplied
    if (
        data.get("schema_version") != SMOKE_SCHEMA_VERSION
        or data.get("method_id") != SMOKE_METHOD_ID
        or data.get("variant_order") != list(PROVIDER_VARIANTS_V1)
        or data.get("all_four_outcomes_accounted_for") is not True
        or data.get("one_EDF_load_session_reused_for_all_four_variants") is not True
        or data.get("eligible_transform_payloads_validated_while_live") is not True
        or data.get("full_record_transform_arrays_serialized") is not False
    ):
        raise ValueError("TUSZ provider record smoke semantics drifted")
    rows = data.get("compact_outcomes")
    if (
        not isinstance(rows, list)
        or [row.get("variant_id") for row in rows] != list(PROVIDER_VARIANTS_V1)
        or any(
            row.get("full_record_array_or_tensor_retained") is not False for row in rows
        )
    ):
        raise ValueError("TUSZ provider smoke variant Cartesian drifted")
    expected_row_fields = {
        "provider_family",
        "variant_id",
        "compact_outcome_receipt_sha256",
        "eligibility_receipt",
        "transform_receipt",
        "full_record_array_or_tensor_retained",
        "source_path_or_patient_identifier_emitted",
        "receipt_sha256",
    }
    source = data.get("source_record")
    if type(source) is not dict or set(source) != {
        "analysis_identity_id",
        "recording_duration_seconds_fraction",
        "source_path_or_patient_identifier_emitted",
    }:
        raise ValueError("TUSZ provider smoke source summary drifted")
    identity = source["analysis_identity_id"]
    if (
        not isinstance(identity, str)
        or not identity
        or source["source_path_or_patient_identifier_emitted"] is not False
        or type(source["recording_duration_seconds_fraction"]) is not list
        or len(source["recording_duration_seconds_fraction"]) != 2
        or any(
            type(item) is not int or item <= 0
            for item in source["recording_duration_seconds_fraction"]
        )
    ):
        raise PermissionError("TUSZ provider smoke source summary is invalid")
    status_by_variant: dict[str, str] = {}
    for row in rows:
        if type(row) is not dict or set(row) != expected_row_fields:
            raise ValueError("TUSZ provider smoke compact row fields drifted")
        row_pending = deepcopy(row)
        row_supplied = _require_sha256(
            row_pending.get("receipt_sha256"), "smoke compact row"
        )
        row_pending["receipt_sha256"] = _PENDING
        if row_supplied != _canonical_sha256(row_pending):
            raise ValueError("TUSZ provider smoke compact row is not addressed")
        variant_id = row["variant_id"]
        expected_family = (
            "eventnet"
            if variant_id in {_eventnet.EN19_VARIANT_ID, _eventnet.EN17_VARIANT_ID}
            else "seizuretransformer"
        )
        if (
            row["provider_family"] != expected_family
            or row["source_path_or_patient_identifier_emitted"] is not False
        ):
            raise PermissionError("TUSZ provider smoke row scope drifted")
        _require_sha256(
            row["compact_outcome_receipt_sha256"], "compact outcome receipt"
        )
        eligibility = row["eligibility_receipt"]
        if type(eligibility) is not dict:
            raise ValueError("TUSZ provider smoke eligibility is malformed")
        eligibility_pending = deepcopy(eligibility)
        eligibility_supplied = _require_sha256(
            eligibility_pending.get("receipt_sha256"), "eligibility receipt"
        )
        eligibility_pending["receipt_sha256"] = _PENDING
        if (
            eligibility_supplied != _canonical_sha256(eligibility_pending)
            or eligibility.get("analysis_identity_id") != identity
            or eligibility.get("variant_id") != variant_id
            or eligibility.get("status") not in {"eligible", "typed_exclusion"}
            or eligibility.get(
                "phase_reference_event_annotation_or_clinical_input_consumed"
            )
            is not False
        ):
            raise PermissionError("TUSZ provider smoke eligibility semantics drifted")
        transform = row["transform_receipt"]
        if eligibility["status"] == "eligible":
            if type(transform) is not dict:
                raise ValueError("eligible smoke outcome lacks transform receipt")
            transform_pending = deepcopy(transform)
            transform_supplied = _require_sha256(
                transform_pending.get("receipt_sha256"), "transform receipt"
            )
            transform_pending["receipt_sha256"] = _PENDING
            if transform_supplied != _canonical_sha256(
                transform_pending
            ) or transform_supplied != eligibility.get("transform_receipt_sha256"):
                raise ValueError("smoke transform receipt does not replay")
        elif (
            transform is not None
            or eligibility.get("transform_receipt_sha256") is not None
        ):
            raise ValueError("typed-exclusion smoke outcome retained a transform")
        status_by_variant[variant_id] = eligibility["status"]
    if data.get("status_by_variant") != status_by_variant:
        raise ValueError("TUSZ provider smoke status summary drifted")
    lifecycle = data.get("factory_lifecycle_receipt")
    if type(lifecycle) is not dict:
        raise ValueError("TUSZ provider smoke lifecycle receipt is malformed")
    lifecycle_pending = deepcopy(lifecycle)
    lifecycle_supplied = _require_sha256(
        lifecycle_pending.get("receipt_sha256"), "factory lifecycle receipt"
    )
    lifecycle_pending["receipt_sha256"] = _PENDING
    if (
        lifecycle_supplied != _canonical_sha256(lifecycle_pending)
        or lifecycle.get("schema_version") != SCHEMA_VERSION
        or lifecycle.get("method_id") != METHOD_ID
        or lifecycle.get("implementation_source_sha256")
        != detector_provider_tusz_outcome_factory_source_sha256_v1()
        or lifecycle.get("frozen_variant_order") != list(PROVIDER_VARIANTS_V1)
        or type(lifecycle.get("st_numeric_runtime_receipt")) is not dict
        or lifecycle.get("source_record_session_open_count") != 1
        or lifecycle.get("source_record_session_release_count") != 1
        or lifecycle.get("completed_four_variant_record_count") != 1
        or lifecycle.get("aborted_record_session_count") != 0
        or lifecycle.get("variant_outcome_count") != len(PROVIDER_VARIANTS_V1)
        or lifecycle.get("live_source_record_session_count") != 0
    ):
        raise ValueError("TUSZ provider smoke lifecycle semantics drifted")
    runtime = lifecycle["st_numeric_runtime_receipt"]
    runtime_pending = deepcopy(runtime)
    runtime_supplied = _require_sha256(
        runtime_pending.get("receipt_sha256"), "ST runtime receipt"
    )
    runtime_pending["receipt_sha256"] = _PENDING
    if (
        runtime_supplied != _canonical_sha256(runtime_pending)
        or runtime.get("schema_version") != "st_cleanroom_numeric_runtime_receipt_v1"
        or runtime.get("training_runtime_checked") is not False
        or runtime.get("GPU_initialized_or_queried") is not False
    ):
        raise ValueError("TUSZ provider smoke ST runtime receipt drifted")
    scope = data.get("scope_receipt")
    if (
        not isinstance(scope, Mapping)
        or scope.get("public_TUSZ_source_train_EEG_only") is not True
        or any(
            scope.get(field) is not False
            for field in (
                "source_dev_or_eval_EEG_used",
                "reference_sidecar_or_seizure_interval_used",
                "EDF_annotation_used",
                "spreadsheet_doctor_text_history_or_behaviour_used",
                "auxiliary_non_EEG_channel_used",
                "performance_or_clinical_claim_authorized",
            )
        )
    ):
        raise PermissionError("TUSZ provider smoke EEG-only firewall drifted")
    return data


__all__ = [
    "METHOD_ID",
    "SCHEMA_VERSION",
    "SMOKE_METHOD_ID",
    "SMOKE_SCHEMA_VERSION",
    "TUSZTargetFreeProviderOutcomeFactoryV1",
    "detector_provider_tusz_outcome_factory_source_sha256_v1",
    "materialize_tusz_target_free_provider_record_smoke_v1",
    "select_source_train_provider_smoke_record_v1",
    "validate_tusz_target_free_provider_record_smoke_v1",
]
