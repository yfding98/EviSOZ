"""Compact target-free provider eligibility inventory for detector training.

This module closes the full-record-array memory boundary between the
source-train EEG pass and this module's phase/roster intersection.  It
deliberately executes the four provider technical outcomes over the *complete*
source-train denominator before this module accepts a fold phase for roster
issuance.  Eligible provider transforms are validated while their arrays are
live, but only compact receipts are retained or serialized.  The separate
selection-fit reference opener does not yet consume this authority, so global
"first reference byte after complete inventory" ordering is not enforced by
the current repository API.

This is not a whole-corpus fully streaming or resumable executor.  Compact
receipt rows and the final JSONL bytes still scale with the source corpus, and
there is no record-level resume checkpoint.  The outcome factory is called in
record-major order.  Its production implementation must reuse one opened
source-record session across the four consecutive variant calls and release
that session after the fourth call; otherwise a naive implementation may read
the same EDF four times.  That caller-owned lifecycle is documented in the
receipt but is not enforceable by this Python callback boundary.

The serialized bundle is evidence, not authority.  A formal consumer receives
an opaque process-local authority only either directly from the streaming
materializer or after every source record/variant outcome has been replayed by
an outcome factory in the admitting process.  Provider-specific fold rosters
are then derived by intersecting this full source-train Cartesian inventory
with an existing opaque provider fold-phase authority.

No phase, seizure interval, reference sidecar, EDF annotation, spreadsheet,
doctor text, clinical history, or non-EEG auxiliary channel has an input slot
in the outcome-factory call.  However, the callback receives an EDF-relative
locator from which a sidecar name is mechanically derivable, and this module
does not sandbox arbitrary callback filesystem access.  A formal production
factory therefore still requires a repository-bound implementation and/or an
OS-level EDF-only mount capability before a no-reference-byte-open claim is
authorized.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Callable, Final, Mapping, Sequence

from . import eventnet_cleanroom_registry_v1 as _eventnet
from . import seizuretransformer_cleanroom_registry_v1 as _st
from .tusz_detector_cleanroom_fold_plan_v1 import (
    validate_tusz_detector_cleanroom_fold_plan_v1,
)


INVENTORY_SCHEMA_VERSION: Final[
    str
] = "clinical_eeg_detector_provider_pre_reference_inventory_v1"
INVENTORY_AUTHORITY_SCHEMA_VERSION: Final[
    str
] = "clinical_eeg_detector_provider_pre_reference_inventory_authority_v1"
COMPACT_OUTCOME_SCHEMA_VERSION: Final[
    str
] = "clinical_eeg_detector_provider_compact_pre_reference_outcome_v1"
MATERIALIZER_ID: Final[
    str
] = "CLINICAL-EEG-DETECTOR-PROVIDER-PRE-REFERENCE-INVENTORY-V1-20260824"
OUTCOMES_FILE_NAME: Final[str] = "outcomes.jsonl"
MANIFEST_FILE_NAME: Final[str] = "manifest.json"
PROVIDER_VARIANTS_V1: Final[tuple[str, ...]] = (
    _eventnet.EN19_VARIANT_ID,
    _eventnet.EN17_VARIANT_ID,
    _st.ST18_VARIANT_ID,
    _st.ST16_VARIANT_ID,
)

_CONTENT_PENDING: Final[str] = "CONTENT-ADDRESS-PENDING"
_SHA256_CHARS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_MAX_INVENTORY_FILE_BYTES: Final[int] = 1024 * 1024 * 1024
_INVENTORY_AUTHORITY_SEAL = object()
_ELIGIBILITY_RECEIPT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "registry_sha256",
        "variant_id",
        "analysis_identity_id",
        "provider_signal_lineage_authority_sha256",
        "record_identity_authority_sha256",
        "canonical_source_tensor_sha256",
        "support_route_policy_sha256",
        "support_route_receipt_sha256",
        "support_profile_id",
        "technical_eligibility_policy_sha256",
        "source_sampling_rate_fraction_hz",
        "source_sample_count",
        "provider_target_sample_count",
        "fully_observed_training_tile_count",
        "status",
        "reason_codes",
        "transform_receipt_sha256",
        "phase_reference_event_annotation_or_clinical_input_consumed",
        "must_be_frozen_before_corresponding_reference_phase_open",
        "raw_caller_status_or_reason_code_accepted",
        "receipt_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class TargetFreeProviderSourceRecordV1:
    """The complete callback surface for one target-free source record."""

    analysis_identity_id: str
    source_edf_relative_path: str
    recording_duration_seconds_fraction: tuple[int, int]


ProviderOutcomeFactoryV1 = Callable[[TargetFreeProviderSourceRecordV1, str], object]


class AuthorizedDetectorProviderPreReferenceInventoryV1:
    """Opaque compact authority; it never stores a provider signal array."""

    __slots__ = (
        "__root",
        "__manifest_json",
        "__outcomes_jsonl",
        "__authority_receipt_json",
        "__issuer_seal",
    )

    def __init__(
        self,
        *,
        root: Path,
        manifest: Mapping[str, Any],
        outcomes_jsonl: bytes,
        authority_receipt: Mapping[str, Any],
        _issuer_seal: object,
    ) -> None:
        if _issuer_seal is not _INVENTORY_AUTHORITY_SEAL:
            raise PermissionError("compact inventory has no valid issuer seal")
        self.__root = str(Path(root).resolve())
        self.__manifest_json = _canonical_json_bytes(manifest).decode("utf-8")
        self.__outcomes_jsonl = bytes(outcomes_jsonl)
        self.__authority_receipt_json = _canonical_json_bytes(authority_receipt).decode(
            "utf-8"
        )
        self.__issuer_seal = _issuer_seal

    @property
    def root(self) -> Path:
        return Path(self.__root)

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self.__manifest_json)

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self.__authority_receipt_json)

    def _outcomes_bytes(self) -> bytes:
        return bytes(self.__outcomes_jsonl)

    def _has_valid_issuer_seal(self) -> bool:
        return self.__issuer_seal is _INVENTORY_AUTHORITY_SEAL


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


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def detector_provider_pre_reference_inventory_source_sha256_v1() -> str:
    digest = hashlib.sha256()
    with Path(__file__).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value).difference(_SHA256_CHARS)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _identifier(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{context} must be a normalized non-empty string")
    return value


def _content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    if result.get("receipt_sha256") != _CONTENT_PENDING:
        raise ValueError("content-addressed object must begin pending")
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def _validate_content_address(
    value: object, *, required: set[str], context: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != required:
        raise ValueError(f"{context} fields drifted")
    result = deepcopy(value)
    supplied = _require_sha256(result["receipt_sha256"], f"{context} receipt")
    result["receipt_sha256"] = _CONTENT_PENDING
    if supplied != _canonical_sha256(result):
        raise ValueError(f"{context} is not content-addressed")
    result["receipt_sha256"] = supplied
    return result


def _provider_family(variant_id: str) -> str:
    if variant_id in {_eventnet.EN19_VARIANT_ID, _eventnet.EN17_VARIANT_ID}:
        return "eventnet"
    if variant_id in {_st.ST18_VARIANT_ID, _st.ST16_VARIANT_ID}:
        return "seizuretransformer"
    raise ValueError("provider variant is outside the frozen four-variant roster")


def _canonical_registries(
    *,
    eventnet_registry: Mapping[str, Any],
    seizuretransformer_registry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        _eventnet._require_canonical_eventnet_registry(eventnet_registry),
        _st._require_canonical_seizuretransformer_registry(seizuretransformer_registry),
    )


def _source_train_records(
    fold_plan: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[TargetFreeProviderSourceRecordV1, ...]]:
    plan = validate_tusz_detector_cleanroom_fold_plan_v1(fold_plan)
    rows = [
        row
        for row in plan["source_record_duration_rows"]
        if row["model_split"] == "source_train"
    ]
    records: list[TargetFreeProviderSourceRecordV1] = []
    for row in sorted(
        rows,
        key=lambda item: (
            str(item["analysis_identity_id"]),
            str(item["local_edf_path"]),
        ),
    ):
        identity = _identifier(row["analysis_identity_id"], "analysis identity")
        relative_path = _identifier(row["local_edf_path"], "source EDF path")
        parts = Path(relative_path).parts
        if (
            Path(relative_path).is_absolute()
            or "\\" in relative_path
            or ".." in parts
            or not parts
            or parts[0] != "train"
            or Path(relative_path).suffix.lower() != ".edf"
        ):
            raise PermissionError("pre-reference inventory is source-train EDF only")
        fraction = row["recording_duration_seconds_fraction"]
        if (
            type(fraction) is not list
            or len(fraction) != 2
            or type(fraction[0]) is not int
            or type(fraction[1]) is not int
            or fraction[0] <= 0
            or fraction[1] <= 0
        ):
            raise ValueError("source recording duration fraction is invalid")
        records.append(
            TargetFreeProviderSourceRecordV1(
                analysis_identity_id=identity,
                source_edf_relative_path=relative_path,
                recording_duration_seconds_fraction=(fraction[0], fraction[1]),
            )
        )
    if not records:
        raise ValueError("fold plan has no source-train provider denominator")
    identities = [record.analysis_identity_id for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("source-train inventory repeats an analysis identity")
    expected = plan["source_split_rosters"]["source_train"]
    if expected["recording_count"] != len(records) or expected[
        "analysis_identity_roster_sha256"
    ] != _canonical_sha256(sorted(identities)):
        raise ValueError("source-train inventory denominator drifted")
    return plan, tuple(records)


def _source_record_payload(record: TargetFreeProviderSourceRecordV1) -> dict[str, Any]:
    return {
        "analysis_identity_id": record.analysis_identity_id,
        "source_edf_relative_path": record.source_edf_relative_path,
        "recording_duration_seconds_fraction": list(
            record.recording_duration_seconds_fraction
        ),
    }


def _validate_transform_scope(receipt: Mapping[str, Any], *, family: str) -> None:
    scope = receipt.get("scope_receipt")
    if not isinstance(scope, Mapping):
        raise PermissionError("eligible transform lacks an EEG-only scope receipt")
    if (
        scope.get("EEG_samples_used") is not True
        or scope.get("acquisition_clock_used") is not True
        or scope.get("EEG_electrical_reference_provenance_used_as_model_feature")
        is not False
    ):
        raise PermissionError("eligible transform EEG input scope drifted")
    for key in (
        "seizure_target_or_reference_label_used",
        "EDF_annotation_used",
        "spreadsheet_or_doctor_text_used",
        "clinical_history_used",
        "auxiliary_non_EEG_channel_used",
    ):
        if scope.get(key) is not False:
            raise PermissionError(
                f"forbidden pre-reference transform input opened: {key}"
            )
    if (
        family == "eventnet"
        and scope.get("EEG_only_QC_used_as_admission_control_plane") is not True
    ):
        raise PermissionError("EventNet transform lacks its EEG-derived QC binding")


def _compact_outcome(
    record: TargetFreeProviderSourceRecordV1,
    *,
    variant_id: str,
    outcome: object,
    eventnet_registry: Mapping[str, Any],
    seizuretransformer_registry: Mapping[str, Any],
) -> dict[str, Any]:
    family = _provider_family(variant_id)
    if family == "eventnet":
        transform, eligibility = _eventnet._require_eventnet_pre_reference_eligibility(
            outcome
        )
        expected_registry_sha256 = eventnet_registry["registry_sha256"]
        expected_policy_sha256 = _eventnet._canonical_sha256(
            _eventnet._eventnet_pre_reference_technical_policy(variant_id)
        )
        expected_support_sha256 = _eventnet.detector_channel_support_policy_receipt()[
            "policy_sha256"
        ]
    else:
        transform, eligibility = _st._require_st_pre_reference_eligibility(outcome)
        expected_registry_sha256 = seizuretransformer_registry["registry_sha256"]
        expected_policy_sha256 = _st._canonical_sha256(
            _st._st_pre_reference_technical_policy(variant_id)
        )
        expected_support_sha256 = _st.detector_channel_support_policy_receipt()[
            "policy_sha256"
        ]

    if (
        eligibility["analysis_identity_id"] != record.analysis_identity_id
        or eligibility["variant_id"] != variant_id
        or eligibility["registry_sha256"] != expected_registry_sha256
        or eligibility["support_route_policy_sha256"] != expected_support_sha256
        or eligibility["technical_eligibility_policy_sha256"] != expected_policy_sha256
        or eligibility["phase_reference_event_annotation_or_clinical_input_consumed"]
        is not False
        or eligibility["must_be_frozen_before_corresponding_reference_phase_open"]
        is not True
        or eligibility["raw_caller_status_or_reason_code_accepted"] is not False
    ):
        raise PermissionError(
            "pre-reference outcome crosses source, variant or firewall"
        )
    for field in (
        "provider_signal_lineage_authority_sha256",
        "record_identity_authority_sha256",
        "canonical_source_tensor_sha256",
        "support_route_receipt_sha256",
        "receipt_sha256",
    ):
        _require_sha256(eligibility[field], f"eligibility {field}")
    rate = eligibility["source_sampling_rate_fraction_hz"]
    for field in (
        "source_sample_count",
        "provider_target_sample_count",
        "fully_observed_training_tile_count",
    ):
        if type(eligibility[field]) is not int or eligibility[field] < 0:
            raise ValueError(f"eligibility {field} is invalid")
    if (
        type(rate) is not list
        or len(rate) != 2
        or type(rate[0]) is not int
        or type(rate[1]) is not int
        or rate[0] <= 0
        or rate[1] <= 0
        or type(eligibility["reason_codes"]) is not list
        or any(
            not isinstance(reason, str) or not reason
            for reason in eligibility["reason_codes"]
        )
        or len(set(eligibility["reason_codes"])) != len(eligibility["reason_codes"])
    ):
        raise ValueError("pre-reference clock or reason-code ledger is invalid")

    transform_receipt: dict[str, Any] | None
    if eligibility["status"] == "eligible":
        if (
            transform is None
            or eligibility["fully_observed_training_tile_count"] < 1
            or eligibility["provider_target_sample_count"] < 1
        ):
            raise ValueError("eligible outcome lacks a usable provider transform")
        transform_receipt = deepcopy(transform.receipt)
        _require_sha256(transform_receipt.get("receipt_sha256"), "transform receipt")
        if (
            transform_receipt["receipt_sha256"]
            != eligibility["transform_receipt_sha256"]
        ):
            raise ValueError("eligibility and transform receipts differ")
        _validate_transform_scope(transform_receipt, family=family)
    elif eligibility["status"] == "typed_exclusion":
        if transform is not None or eligibility["transform_receipt_sha256"] is not None:
            raise ValueError("typed exclusion retained a provider transform")
        transform_receipt = None
    else:
        raise ValueError("pre-reference outcome has an unsupported status")

    return _content_address(
        {
            "schema_version": COMPACT_OUTCOME_SCHEMA_VERSION,
            "provider_family": family,
            "variant_id": variant_id,
            "source_record": _source_record_payload(record),
            "eligibility_receipt": deepcopy(eligibility),
            "transform_receipt": transform_receipt,
            "full_record_array_or_tensor_retained": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )


def _materialize_compact_rows(
    records: Sequence[TargetFreeProviderSourceRecordV1],
    *,
    outcome_factory: ProviderOutcomeFactoryV1,
    eventnet_registry: Mapping[str, Any],
    seizuretransformer_registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not callable(outcome_factory):
        raise TypeError("outcome_factory must be callable")
    rows: list[dict[str, Any]] = []
    for record in records:
        for variant_id in PROVIDER_VARIANTS_V1:
            outcome = outcome_factory(record, variant_id)
            row = _compact_outcome(
                record,
                variant_id=variant_id,
                outcome=outcome,
                eventnet_registry=eventnet_registry,
                seizuretransformer_registry=seizuretransformer_registry,
            )
            rows.append(row)
            # The compact row contains receipts only.  Do not retain the opaque
            # outcome or its potentially very large full-record transform.
            del outcome
    return rows


def _outcomes_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)


def _registry_bindings(
    eventnet_registry: Mapping[str, Any],
    seizuretransformer_registry: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "eventnet": {
            "registry_id": eventnet_registry["registry_id"],
            "registry_sha256": eventnet_registry["registry_sha256"],
            "implementation_code_sha256": eventnet_registry["implementation"][
                "code_sha256"
            ],
            "variant_ids": [
                _eventnet.EN19_VARIANT_ID,
                _eventnet.EN17_VARIANT_ID,
            ],
        },
        "seizuretransformer": {
            "registry_id": seizuretransformer_registry["registry_id"],
            "registry_sha256": seizuretransformer_registry["registry_sha256"],
            "implementation_code_sha256": seizuretransformer_registry["implementation"][
                "code_sha256"
            ],
            "variant_ids": [_st.ST18_VARIANT_ID, _st.ST16_VARIANT_ID],
        },
    }


def _build_manifest(
    *,
    plan: Mapping[str, Any],
    records: Sequence[TargetFreeProviderSourceRecordV1],
    rows: Sequence[Mapping[str, Any]],
    outcomes_bytes: bytes,
    eventnet_registry: Mapping[str, Any],
    seizuretransformer_registry: Mapping[str, Any],
) -> dict[str, Any]:
    status_counts: dict[str, dict[str, int]] = {}
    for variant_id in PROVIDER_VARIANTS_V1:
        variant_rows = [row for row in rows if row["variant_id"] == variant_id]
        status_counts[variant_id] = {
            "eligible": sum(
                row["eligibility_receipt"]["status"] == "eligible"
                for row in variant_rows
            ),
            "typed_exclusion": sum(
                row["eligibility_receipt"]["status"] == "typed_exclusion"
                for row in variant_rows
            ),
        }
    source_payloads = [_source_record_payload(record) for record in records]
    return _content_address(
        {
            "schema_version": INVENTORY_SCHEMA_VERSION,
            "materializer_id": MATERIALIZER_ID,
            "materializer_source_sha256": (
                detector_provider_pre_reference_inventory_source_sha256_v1()
            ),
            "fold_plan_id": plan["plan_id"],
            "fold_plan_receipt_sha256": plan["receipt_sha256"],
            "source_record_duration_binding_sha256": plan["source_binding"][
                "source_record_duration_binding_sha256"
            ],
            "source_train_record_count": len(records),
            "source_train_analysis_identity_roster_sha256": _canonical_sha256(
                sorted(record.analysis_identity_id for record in records)
            ),
            "source_train_record_spec_roster_sha256": _canonical_sha256(
                source_payloads
            ),
            "provider_registry_bindings": _registry_bindings(
                eventnet_registry, seizuretransformer_registry
            ),
            "variant_ids": list(PROVIDER_VARIANTS_V1),
            "cartesian_outcome_count": len(rows),
            "expected_cartesian_outcome_count": len(records)
            * len(PROVIDER_VARIANTS_V1),
            "compact_outcome_receipt_roster_sha256": _canonical_sha256(
                [row["receipt_sha256"] for row in rows]
            ),
            "outcome_status_counts_by_variant": status_counts,
            "outcomes_path": OUTCOMES_FILE_NAME,
            "outcomes_size_bytes": len(outcomes_bytes),
            "outcomes_file_sha256": _bytes_sha256(outcomes_bytes),
            "complete_source_train_by_four_variant_cartesian": True,
            "eligible_full_record_transform_arrays_serialized": False,
            "memory_and_resume_receipt": {
                "EEG_array_memory_bounded_per_outcome": True,
                "EEG_array_memory_bound_scope": (
                    "materializer_owned_outcome_references_only"
                ),
                "outcome_factory_external_array_retention_runtime_enforced": False,
                "compact_receipt_memory_scales_with_corpus": True,
                "record_level_resume_implemented": False,
                "whole_corpus_fully_streaming_claim_authorized": False,
                "large_real_fold_scalability_fully_admitted": False,
            },
            "callback_contract": {
                "record_argument_type": "TargetFreeProviderSourceRecordV1",
                "record_argument_fields": [
                    "analysis_identity_id",
                    "source_edf_relative_path",
                    "recording_duration_seconds_fraction",
                ],
                "variant_argument_only": True,
                "fold_phase_reference_event_annotation_or_clinical_argument": False,
                "call_order": "record_major_then_frozen_variant_order",
                "frozen_variant_order": list(PROVIDER_VARIANTS_V1),
                "production_factory_must_reuse_same_record_session": True,
                "production_factory_must_release_session_after_fourth_variant": True,
                "record_session_lifecycle_runtime_enforced_by_this_API": False,
                "naive_factory_may_read_each_EDF_four_times": True,
                "source_EDF_locator_exposed": True,
                "source_EDF_locator_contains_split_patient_session_components": True,
                "reference_sidecar_name_mechanically_derivable_from_locator": True,
                "outcome_factory_implementation_source_bound_by_this_API": False,
                "outcome_factory_filesystem_read_allowlist_runtime_enforced": False,
            },
            "sequencing_receipt": {
                "inventory_callback_accepts_phase_or_reference_authority": False,
                "complete_cartesian_required_before_inventory_authority_issue": True,
                "provider_roster_requires_complete_inventory_authority": True,
                "selection_fit_reference_opener_requires_inventory_authority": False,
                "global_first_reference_open_after_inventory_runtime_enforced": False,
            },
            "scope_receipt": {
                "public_TUSZ_source_train_EEG_only": True,
                "source_dev_or_eval_EEG_used": False,
                "reference_sidecar_or_seizure_interval_used": False,
                "reference_nonuse_claim_scope": (
                    "callback_arguments_and_admitted_opaque_provider_outcomes_only"
                ),
                "outcome_factory_incidental_filesystem_access_runtime_audited": False,
                "no_reference_byte_open_claim_authorized": False,
                "EDF_annotation_used": False,
                "spreadsheet_doctor_text_history_or_behaviour_used": False,
                "auxiliary_non_EEG_channel_used": False,
                "phase_or_fold_membership_used_for_technical_eligibility": False,
                "performance_or_clinical_claim_authorized": False,
            },
            "receipt_sha256": _CONTENT_PENDING,
        }
    )


def _parse_jsonl(payload: bytes) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("compact outcome JSONL is empty or unterminated")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(payload.splitlines()):
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"compact outcome row {index} is invalid JSON") from exc
        if _canonical_json_bytes(value) != raw:
            raise ValueError("compact outcome JSONL is not canonical")
        if type(value) is not dict:
            raise ValueError("compact outcome row must be an object")
        rows.append(value)
    return rows


def _validate_transform_receipt_evidence(
    transform: object,
    *,
    family: str,
    variant_id: str,
    expected_registry_sha256: str,
    eligibility: Mapping[str, Any],
) -> dict[str, Any] | None:
    if eligibility["status"] == "typed_exclusion":
        if transform is not None or eligibility["transform_receipt_sha256"] is not None:
            raise ValueError("compact typed exclusion retained transform evidence")
        return None
    if type(transform) is not dict:
        raise ValueError("compact eligible row lacks its transform receipt")
    result = deepcopy(transform)
    supplied = _require_sha256(result.get("receipt_sha256"), "transform receipt")
    pending = deepcopy(result)
    pending["receipt_sha256"] = _CONTENT_PENDING
    if supplied != _canonical_sha256(pending):
        raise ValueError("compact transform receipt does not replay")
    if supplied != eligibility["transform_receipt_sha256"]:
        raise ValueError("compact transform/eligibility receipts differ")
    expected_schema = (
        "eventnet_cleanroom_full_record_transform_receipt_v1"
        if family == "eventnet"
        else "seizuretransformer_full_record_transform_receipt_v1"
    )
    if (
        result.get("schema_version") != expected_schema
        or result.get("variant_id") != variant_id
        or result.get("registry_sha256") != expected_registry_sha256
        or result.get("detector_signal_lineage_authority_sha256")
        != eligibility["provider_signal_lineage_authority_sha256"]
        or result.get("canonical_source_tensor_sha256")
        != eligibility["canonical_source_tensor_sha256"]
    ):
        raise ValueError("compact transform lineage evidence drifted")
    _validate_transform_scope(result, family=family)
    output = result.get("output")
    if (
        not isinstance(output, Mapping)
        or output.get("sample_count") != eligibility["provider_target_sample_count"]
        or not isinstance(output.get("payload_receipt"), Mapping)
    ):
        raise ValueError("compact transform output evidence drifted")
    _require_sha256(
        output["payload_receipt"].get("payload_sha256"),
        "transform output payload",
    )
    return result


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    records: Sequence[TargetFreeProviderSourceRecordV1],
    eventnet_registry: Mapping[str, Any],
    seizuretransformer_registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected_keys = [
        (record.analysis_identity_id, variant_id)
        for record in records
        for variant_id in PROVIDER_VARIANTS_V1
    ]
    if len(rows) != len(expected_keys):
        raise PermissionError("compact provider Cartesian denominator is incomplete")
    required = {
        "schema_version",
        "provider_family",
        "variant_id",
        "source_record",
        "eligibility_receipt",
        "transform_receipt",
        "full_record_array_or_tensor_retained",
        "receipt_sha256",
    }
    result: list[dict[str, Any]] = []
    observed_keys: list[tuple[str, str]] = []
    record_by_identity = {record.analysis_identity_id: record for record in records}
    for raw in rows:
        row = _validate_content_address(
            raw, required=required, context="compact pre-reference outcome"
        )
        if (
            row["schema_version"] != COMPACT_OUTCOME_SCHEMA_VERSION
            or row["full_record_array_or_tensor_retained"] is not False
            or row["variant_id"] not in PROVIDER_VARIANTS_V1
            or row["provider_family"] != _provider_family(row["variant_id"])
        ):
            raise ValueError("compact outcome method or memory semantics drifted")
        source = row["source_record"]
        if type(source) is not dict or set(source) != {
            "analysis_identity_id",
            "source_edf_relative_path",
            "recording_duration_seconds_fraction",
        }:
            raise ValueError("compact source-record fields drifted")
        identity = source["analysis_identity_id"]
        record = record_by_identity.get(identity)
        if record is None or source != _source_record_payload(record):
            raise PermissionError("compact outcome lies outside source-train")
        variant_id = row["variant_id"]
        family = row["provider_family"]
        eligibility = row["eligibility_receipt"]
        if family == "eventnet":
            expected_registry = eventnet_registry["registry_sha256"]
            expected_technical = _eventnet._canonical_sha256(
                _eventnet._eventnet_pre_reference_technical_policy(variant_id)
            )
            expected_support = _eventnet.detector_channel_support_policy_receipt()[
                "policy_sha256"
            ]
        else:
            expected_registry = seizuretransformer_registry["registry_sha256"]
            expected_technical = _st._canonical_sha256(
                _st._st_pre_reference_technical_policy(variant_id)
            )
            expected_support = _st.detector_channel_support_policy_receipt()[
                "policy_sha256"
            ]
        if type(eligibility) is not dict or set(eligibility) != set(
            _ELIGIBILITY_RECEIPT_FIELDS
        ):
            raise ValueError("compact eligibility receipt is malformed")
        supplied = _require_sha256(
            eligibility.get("receipt_sha256"), "eligibility receipt"
        )
        pending = deepcopy(eligibility)
        pending["receipt_sha256"] = _CONTENT_PENDING
        if supplied != _canonical_sha256(pending):
            raise ValueError("compact eligibility receipt does not replay")
        expected_schema = (
            "eventnet_pre_reference_record_eligibility_outcome_v1"
            if family == "eventnet"
            else "st_pre_reference_record_eligibility_outcome_v1"
        )
        accepted_support_profiles = (
            {"complete19"}
            if variant_id in {_eventnet.EN19_VARIANT_ID, _st.ST18_VARIANT_ID}
            else {"complete19", "lateral17"}
        )
        rate = eligibility.get("source_sampling_rate_fraction_hz")
        counts = [
            eligibility.get("source_sample_count"),
            eligibility.get("provider_target_sample_count"),
            eligibility.get("fully_observed_training_tile_count"),
        ]
        reasons = eligibility.get("reason_codes")
        if (
            eligibility.get("schema_version") != expected_schema
            or eligibility.get("analysis_identity_id") != identity
            or eligibility.get("variant_id") != variant_id
            or eligibility.get("registry_sha256") != expected_registry
            or eligibility.get("support_route_policy_sha256") != expected_support
            or eligibility.get("technical_eligibility_policy_sha256")
            != expected_technical
            or eligibility.get(
                "phase_reference_event_annotation_or_clinical_input_consumed"
            )
            is not False
            or eligibility.get(
                "must_be_frozen_before_corresponding_reference_phase_open"
            )
            is not True
            or eligibility.get("raw_caller_status_or_reason_code_accepted") is not False
            or eligibility.get("status") not in {"eligible", "typed_exclusion"}
            or eligibility.get("support_profile_id") not in accepted_support_profiles
            or type(rate) is not list
            or len(rate) != 2
            or type(rate[0]) is not int
            or type(rate[1]) is not int
            or rate[0] <= 0
            or rate[1] <= 0
            or math.gcd(rate[0], rate[1]) != 1
            or any(type(count) is not int or count < 0 for count in counts)
            or type(reasons) is not list
            or any(
                not isinstance(reason, str) or not reason or reason != reason.strip()
                for reason in reasons
            )
            or len(reasons) != len(set(reasons))
            or (eligibility["status"] == "eligible") is not (reasons == [])
            or (eligibility["status"] == "typed_exclusion") is not bool(reasons)
        ):
            raise PermissionError("compact eligibility semantics or firewall drifted")
        for field in (
            "provider_signal_lineage_authority_sha256",
            "record_identity_authority_sha256",
            "canonical_source_tensor_sha256",
            "support_route_receipt_sha256",
        ):
            _require_sha256(eligibility[field], f"compact eligibility {field}")
        if eligibility["status"] == "eligible":
            _require_sha256(
                eligibility["transform_receipt_sha256"],
                "compact eligibility transform receipt",
            )
            if (
                eligibility["provider_target_sample_count"] < 1
                or eligibility["fully_observed_training_tile_count"] < 1
            ):
                raise ValueError("compact eligible outcome lacks a usable clock")
        elif eligibility["transform_receipt_sha256"] is not None:
            raise ValueError("compact typed exclusion binds a transform receipt")
        _validate_transform_receipt_evidence(
            row["transform_receipt"],
            family=family,
            variant_id=variant_id,
            expected_registry_sha256=expected_registry,
            eligibility=eligibility,
        )
        observed_keys.append((identity, variant_id))
        result.append(row)
    if observed_keys != expected_keys or len(set(observed_keys)) != len(observed_keys):
        raise PermissionError(
            "compact inventory is not the canonical source-train by variant Cartesian"
        )
    return result


def _validate_manifest(
    manifest: object,
    *,
    plan: Mapping[str, Any],
    records: Sequence[TargetFreeProviderSourceRecordV1],
    rows: Sequence[Mapping[str, Any]],
    outcomes_bytes: bytes,
    eventnet_registry: Mapping[str, Any],
    seizuretransformer_registry: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _build_manifest(
        plan=plan,
        records=records,
        rows=rows,
        outcomes_bytes=outcomes_bytes,
        eventnet_registry=eventnet_registry,
        seizuretransformer_registry=seizuretransformer_registry,
    )
    if manifest != expected:
        raise PermissionError("compact inventory manifest differs from frozen sources")
    return expected


def _secure_read(root: Path, name: str) -> bytes:
    if name not in {MANIFEST_FILE_NAME, OUTCOMES_FILE_NAME}:
        raise ValueError("compact inventory file is outside the allowlist")
    path = root / name
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_INVENTORY_FILE_BYTES
        ):
            raise ValueError("compact inventory artifact is not a safe regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError("compact inventory artifact was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("compact inventory artifact grew during replay")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("compact inventory artifact changed during replay")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _require_strict_bundle_inventory(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("compact inventory root must remain a real directory")
    entries = list(root.iterdir())
    inventory = {path.name for path in entries}
    if (
        len(entries) != 2
        or inventory != {MANIFEST_FILE_NAME, OUTCOMES_FILE_NAME}
        or any(path.is_symlink() or not path.is_file() for path in entries)
    ):
        raise ValueError("compact inventory file inventory drifted")


def _read_bundle(
    directory: str | Path,
    *,
    plan: Mapping[str, Any],
    records: Sequence[TargetFreeProviderSourceRecordV1],
    eventnet_registry: Mapping[str, Any],
    seizuretransformer_registry: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], bytes, bytes]:
    root_input = Path(directory)
    if root_input.is_symlink():
        raise ValueError("compact inventory root must not be a symlink")
    root = root_input.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("compact inventory root must be a real directory")
    _require_strict_bundle_inventory(root)
    manifest_bytes = _secure_read(root, MANIFEST_FILE_NAME)
    outcomes_bytes = _secure_read(root, OUTCOMES_FILE_NAME)
    _require_strict_bundle_inventory(root)
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("compact inventory manifest is invalid JSON") from exc
    if manifest_bytes != _canonical_json_bytes(manifest):
        raise ValueError("compact inventory manifest is not canonical")
    rows = _validate_rows(
        _parse_jsonl(outcomes_bytes),
        records=records,
        eventnet_registry=eventnet_registry,
        seizuretransformer_registry=seizuretransformer_registry,
    )
    manifest = _validate_manifest(
        manifest,
        plan=plan,
        records=records,
        rows=rows,
        outcomes_bytes=outcomes_bytes,
        eventnet_registry=eventnet_registry,
        seizuretransformer_registry=seizuretransformer_registry,
    )
    return root, manifest, rows, manifest_bytes, outcomes_bytes


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("compact inventory write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bundle(
    output_directory: str | Path,
    *,
    manifest: Mapping[str, Any],
    outcomes_bytes: bytes,
) -> Path:
    destination = Path(output_directory).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite compact inventory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        _atomic_write(temporary / OUTCOMES_FILE_NAME, outcomes_bytes)
        _atomic_write(temporary / MANIFEST_FILE_NAME, _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def _authority_receipt(
    *, manifest: Mapping[str, Any], manifest_bytes: bytes, outcomes_bytes: bytes
) -> dict[str, Any]:
    return _content_address(
        {
            "schema_version": INVENTORY_AUTHORITY_SCHEMA_VERSION,
            "materializer_id": MATERIALIZER_ID,
            "materializer_source_sha256": (
                detector_provider_pre_reference_inventory_source_sha256_v1()
            ),
            "manifest_receipt_sha256": manifest["receipt_sha256"],
            "manifest_file_sha256": _bytes_sha256(manifest_bytes),
            "outcomes_file_sha256": _bytes_sha256(outcomes_bytes),
            "source_train_record_count": manifest["source_train_record_count"],
            "cartesian_outcome_count": manifest["cartesian_outcome_count"],
            "actual_process_sealed_outcome_objects_replayed": True,
            "every_eligible_full_record_transform_payload_validated_while_live": True,
            "full_record_arrays_retained_by_authority": False,
            "EEG_array_memory_bounded_per_outcome": True,
            "EEG_array_memory_bound_scope": (
                "materializer_owned_outcome_references_only"
            ),
            "outcome_factory_external_array_retention_runtime_enforced": False,
            "compact_receipt_memory_scales_with_corpus": True,
            "record_level_resume_implemented": False,
            "whole_corpus_fully_streaming_claim_authorized": False,
            "large_real_fold_scalability_fully_admitted": False,
            "outcome_factory_call_order_record_major": True,
            "outcome_factory_record_session_lifecycle_runtime_enforced": False,
            "target_free_callback_argument_surface_enforced": True,
            "outcome_factory_filesystem_read_allowlist_runtime_enforced": False,
            "no_reference_byte_open_before_inventory_claim_authorized": False,
            "provider_roster_requires_complete_inventory_authority": True,
            "selection_fit_reference_opener_requires_inventory_authority": False,
            "global_first_reference_open_after_inventory_runtime_enforced": False,
            "serialized_manifest_or_JSONL_alone_is_authority": False,
            "reference_or_clinical_input_slot_exposed": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )


def _issue_authority(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_bytes: bytes,
    outcomes_bytes: bytes,
) -> AuthorizedDetectorProviderPreReferenceInventoryV1:
    return AuthorizedDetectorProviderPreReferenceInventoryV1(
        root=root,
        manifest=manifest,
        outcomes_jsonl=outcomes_bytes,
        authority_receipt=_authority_receipt(
            manifest=manifest,
            manifest_bytes=manifest_bytes,
            outcomes_bytes=outcomes_bytes,
        ),
        _issuer_seal=_INVENTORY_AUTHORITY_SEAL,
    )


def materialize_detector_provider_pre_reference_inventory_v1(
    output_directory: str | Path,
    *,
    fold_plan: Mapping[str, Any],
    eventnet_registry: Mapping[str, Any],
    seizuretransformer_registry: Mapping[str, Any],
    outcome_factory: ProviderOutcomeFactoryV1,
) -> AuthorizedDetectorProviderPreReferenceInventoryV1:
    """Evaluate source-train x four outcomes, atomically seal, and admit.

    Full-record transform arrays are live only for one callback outcome at a
    time from this materializer's perspective.  Compact rows and JSONL bytes
    remain corpus-scaled in memory, and interrupted runs cannot resume.  The
    factory must reuse a same-record source session across each consecutive
    four-variant group and release it after the fourth variant.
    """

    eventnet, seizuretransformer = _canonical_registries(
        eventnet_registry=eventnet_registry,
        seizuretransformer_registry=seizuretransformer_registry,
    )
    plan, records = _source_train_records(fold_plan)
    rows = _materialize_compact_rows(
        records,
        outcome_factory=outcome_factory,
        eventnet_registry=eventnet,
        seizuretransformer_registry=seizuretransformer,
    )
    outcomes_bytes = _outcomes_jsonl(rows)
    manifest = _build_manifest(
        plan=plan,
        records=records,
        rows=rows,
        outcomes_bytes=outcomes_bytes,
        eventnet_registry=eventnet,
        seizuretransformer_registry=seizuretransformer,
    )
    destination = _write_bundle(
        output_directory, manifest=manifest, outcomes_bytes=outcomes_bytes
    )
    (
        root,
        replayed_manifest,
        replayed_rows,
        manifest_bytes,
        replayed_outcomes,
    ) = _read_bundle(
        destination,
        plan=plan,
        records=records,
        eventnet_registry=eventnet,
        seizuretransformer_registry=seizuretransformer,
    )
    if replayed_rows != rows or replayed_outcomes != outcomes_bytes:
        raise PermissionError("published compact inventory differs from live outcomes")
    return _issue_authority(
        root,
        manifest=replayed_manifest,
        manifest_bytes=manifest_bytes,
        outcomes_bytes=replayed_outcomes,
    )


def authorize_detector_provider_pre_reference_inventory_v1(
    directory: str | Path,
    *,
    fold_plan: Mapping[str, Any],
    eventnet_registry: Mapping[str, Any],
    seizuretransformer_registry: Mapping[str, Any],
    outcome_factory: ProviderOutcomeFactoryV1,
) -> AuthorizedDetectorProviderPreReferenceInventoryV1:
    """Recompute every live outcome before admitting serialized evidence."""

    eventnet, seizuretransformer = _canonical_registries(
        eventnet_registry=eventnet_registry,
        seizuretransformer_registry=seizuretransformer_registry,
    )
    plan, records = _source_train_records(fold_plan)
    expected_rows = _materialize_compact_rows(
        records,
        outcome_factory=outcome_factory,
        eventnet_registry=eventnet,
        seizuretransformer_registry=seizuretransformer,
    )
    root, manifest, rows, manifest_bytes, outcomes_bytes = _read_bundle(
        directory,
        plan=plan,
        records=records,
        eventnet_registry=eventnet,
        seizuretransformer_registry=seizuretransformer,
    )
    if rows != expected_rows:
        raise PermissionError(
            "serialized compact inventory differs from actual outcome replay"
        )
    return _issue_authority(
        root,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        outcomes_bytes=outcomes_bytes,
    )


def validate_detector_provider_pre_reference_inventory_evidence_v1(
    directory: str | Path,
    *,
    fold_plan: Mapping[str, Any],
    eventnet_registry: Mapping[str, Any],
    seizuretransformer_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate serialized evidence only; no formal authority is issued."""

    eventnet, seizuretransformer = _canonical_registries(
        eventnet_registry=eventnet_registry,
        seizuretransformer_registry=seizuretransformer_registry,
    )
    plan, records = _source_train_records(fold_plan)
    _root, manifest, _rows, _manifest_bytes, _outcomes_bytes = _read_bundle(
        directory,
        plan=plan,
        records=records,
        eventnet_registry=eventnet,
        seizuretransformer_registry=seizuretransformer,
    )
    return deepcopy(manifest)


def _require_inventory(
    value: object,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if (
        not isinstance(value, AuthorizedDetectorProviderPreReferenceInventoryV1)
        or not value._has_valid_issuer_seal()
    ):
        raise TypeError(
            "provider roster requires an opaque actual-outcome-replayed compact inventory"
        )
    manifest = value.manifest
    outcomes_bytes = value._outcomes_bytes()
    rows = _parse_jsonl(outcomes_bytes)
    receipt = _validate_content_address(
        value.receipt,
        required={
            "schema_version",
            "materializer_id",
            "materializer_source_sha256",
            "manifest_receipt_sha256",
            "manifest_file_sha256",
            "outcomes_file_sha256",
            "source_train_record_count",
            "cartesian_outcome_count",
            "actual_process_sealed_outcome_objects_replayed",
            "every_eligible_full_record_transform_payload_validated_while_live",
            "full_record_arrays_retained_by_authority",
            "EEG_array_memory_bounded_per_outcome",
            "EEG_array_memory_bound_scope",
            "outcome_factory_external_array_retention_runtime_enforced",
            "compact_receipt_memory_scales_with_corpus",
            "record_level_resume_implemented",
            "whole_corpus_fully_streaming_claim_authorized",
            "large_real_fold_scalability_fully_admitted",
            "outcome_factory_call_order_record_major",
            "outcome_factory_record_session_lifecycle_runtime_enforced",
            "target_free_callback_argument_surface_enforced",
            "outcome_factory_filesystem_read_allowlist_runtime_enforced",
            "no_reference_byte_open_before_inventory_claim_authorized",
            "provider_roster_requires_complete_inventory_authority",
            "selection_fit_reference_opener_requires_inventory_authority",
            "global_first_reference_open_after_inventory_runtime_enforced",
            "serialized_manifest_or_JSONL_alone_is_authority",
            "reference_or_clinical_input_slot_exposed",
            "receipt_sha256",
        },
        context="compact inventory authority",
    )
    manifest_bytes = _canonical_json_bytes(manifest)
    if (
        receipt["schema_version"] != INVENTORY_AUTHORITY_SCHEMA_VERSION
        or receipt["materializer_id"] != MATERIALIZER_ID
        or receipt["materializer_source_sha256"]
        != detector_provider_pre_reference_inventory_source_sha256_v1()
        or receipt["manifest_receipt_sha256"] != manifest.get("receipt_sha256")
        or receipt["manifest_file_sha256"] != _bytes_sha256(manifest_bytes)
        or receipt["outcomes_file_sha256"] != _bytes_sha256(outcomes_bytes)
        or receipt["source_train_record_count"]
        != manifest.get("source_train_record_count")
        or receipt["cartesian_outcome_count"] != manifest.get("cartesian_outcome_count")
        or receipt["actual_process_sealed_outcome_objects_replayed"] is not True
        or receipt["every_eligible_full_record_transform_payload_validated_while_live"]
        is not True
        or receipt["full_record_arrays_retained_by_authority"] is not False
        or receipt["EEG_array_memory_bounded_per_outcome"] is not True
        or receipt["EEG_array_memory_bound_scope"]
        != "materializer_owned_outcome_references_only"
        or receipt["outcome_factory_external_array_retention_runtime_enforced"]
        is not False
        or receipt["compact_receipt_memory_scales_with_corpus"] is not True
        or receipt["record_level_resume_implemented"] is not False
        or receipt["whole_corpus_fully_streaming_claim_authorized"] is not False
        or receipt["large_real_fold_scalability_fully_admitted"] is not False
        or receipt["outcome_factory_call_order_record_major"] is not True
        or receipt["outcome_factory_record_session_lifecycle_runtime_enforced"]
        is not False
        or receipt["target_free_callback_argument_surface_enforced"] is not True
        or receipt["outcome_factory_filesystem_read_allowlist_runtime_enforced"]
        is not False
        or receipt["no_reference_byte_open_before_inventory_claim_authorized"]
        is not False
        or receipt["provider_roster_requires_complete_inventory_authority"] is not True
        or receipt["selection_fit_reference_opener_requires_inventory_authority"]
        is not False
        or receipt["global_first_reference_open_after_inventory_runtime_enforced"]
        is not False
        or receipt["serialized_manifest_or_JSONL_alone_is_authority"] is not False
        or receipt["reference_or_clinical_input_slot_exposed"] is not False
    ):
        raise ValueError("compact inventory opaque authority semantics drifted")
    _require_strict_bundle_inventory(value.root)
    current_manifest = _secure_read(value.root, MANIFEST_FILE_NAME)
    current_outcomes = _secure_read(value.root, OUTCOMES_FILE_NAME)
    _require_strict_bundle_inventory(value.root)
    if current_manifest != manifest_bytes or current_outcomes != outcomes_bytes:
        raise PermissionError(
            "compact inventory disk bytes changed after authority issue"
        )
    return manifest, rows, receipt


def require_authorized_detector_provider_pre_reference_inventory_v1(
    value: object,
) -> AuthorizedDetectorProviderPreReferenceInventoryV1:
    """Reject raw JSON and return only a replayed opaque authority."""

    _require_inventory(value)
    assert isinstance(value, AuthorizedDetectorProviderPreReferenceInventoryV1)
    return value


def _phase_plan_binding_matches(
    phase: Mapping[str, Any], manifest: Mapping[str, Any]
) -> bool:
    binding = phase.get("fold_plan_binding")
    return bool(
        isinstance(binding, Mapping)
        and binding.get("plan_receipt_sha256") == manifest["fold_plan_receipt_sha256"]
        and binding.get("source_train_analysis_identity_roster_sha256")
        == manifest["source_train_analysis_identity_roster_sha256"]
    )


def _inventory_rows_for_phase_variant(
    *,
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    phase: Mapping[str, Any],
    variant_id: str,
) -> dict[str, dict[str, Any]]:
    if not _phase_plan_binding_matches(phase, manifest):
        raise PermissionError("provider phase and compact source-train plan differ")
    expected = {str(row["analysis_identity_id"]) for row in phase["records"]}
    by_identity: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["variant_id"] != variant_id:
            continue
        identity = str(row["source_record"]["analysis_identity_id"])
        if identity in by_identity:
            raise ValueError("compact variant inventory repeats an identity")
        by_identity[identity] = deepcopy(dict(row))
    missing = sorted(expected.difference(by_identity))
    if missing:
        raise PermissionError(
            f"phase identities are absent from compact source-train inventory: {missing}"
        )
    return {identity: by_identity[identity] for identity in expected}


def authorize_eventnet_variant_training_roster_from_compact_inventory_v1(
    phase_authority: _eventnet.AuthorizedEventNetFoldPhase,
    inventory_authority: AuthorizedDetectorProviderPreReferenceInventoryV1,
    *,
    variant_id: str,
    registry: Mapping[str, Any],
) -> _eventnet.AuthorizedEventNetVariantTrainingRoster:
    """Issue the existing EventNet roster type without retaining transforms."""

    eventnet_registry = _eventnet._require_canonical_eventnet_registry(registry)
    (
        phase,
        patient_by_identity,
        phase_receipt,
    ) = _eventnet._require_authorized_eventnet_fold_phase(phase_authority)
    _eventnet._variant_profile(variant_id)
    manifest, rows, _inventory_receipt = _require_inventory(inventory_authority)
    if manifest["provider_registry_bindings"]["eventnet"]["registry_sha256"] != (
        eventnet_registry["registry_sha256"]
    ):
        raise ValueError("compact inventory binds another EventNet registry")
    outcomes = _inventory_rows_for_phase_variant(
        manifest=manifest, rows=rows, phase=phase, variant_id=variant_id
    )
    expected = {str(row["analysis_identity_id"]) for row in phase["records"]}
    support_policy_sha256 = _eventnet.detector_channel_support_policy_receipt()[
        "policy_sha256"
    ]
    technical_policy_sha256 = _eventnet._canonical_sha256(
        _eventnet._eventnet_pre_reference_technical_policy(variant_id)
    )
    eligible_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    for identity in sorted(expected):
        outcome = outcomes[identity]["eligibility_receipt"]
        if (
            outcome["registry_sha256"] != eventnet_registry["registry_sha256"]
            or outcome["support_route_policy_sha256"] != support_policy_sha256
            or outcome["technical_eligibility_policy_sha256"] != technical_policy_sha256
        ):
            raise ValueError("compact EventNet outcome policy binding drifted")
        common = {
            "analysis_identity_id": identity,
            "fold_owned_patient_key": patient_by_identity[identity],
            "pre_reference_eligibility_receipt_sha256": outcome["receipt_sha256"],
            "support_route_receipt_sha256": outcome["support_route_receipt_sha256"],
            "technical_eligibility_receipt_sha256": outcome["receipt_sha256"],
        }
        if outcome["status"] == "eligible":
            eligible_rows.append(
                {
                    **common,
                    "provider_signal_lineage_authority_sha256": outcome[
                        "provider_signal_lineage_authority_sha256"
                    ],
                    "record_identity_authority_sha256": outcome[
                        "record_identity_authority_sha256"
                    ],
                    "transform_receipt_sha256": outcome["transform_receipt_sha256"],
                }
            )
        else:
            exclusion_rows.append(
                {
                    **common,
                    "terminal_status": "technical_or_support_exclusion",
                    "reason_codes": outcome["reason_codes"],
                    "retained_in_full_prediction_first_benchmark_denominator": True,
                }
            )
    roster = {"eligible_records": eligible_rows, "typed_exclusions": exclusion_rows}
    receipt = _eventnet._content_address(
        {
            "schema_version": (
                "eventnet_target_blind_variant_training_roster_authority_v1"
            ),
            "registry_sha256": eventnet_registry["registry_sha256"],
            "variant_id": variant_id,
            "outer_fold": phase_receipt["outer_fold"],
            "phase": phase_receipt["phase"],
            "detector_fold_phase_receipt_sha256": phase_receipt[
                "detector_fold_phase_receipt_sha256"
            ],
            "support_route_policy_sha256": support_policy_sha256,
            "pre_reference_technical_eligibility_policy_sha256": technical_policy_sha256,
            "phase_record_count": len(expected),
            "eligible_record_count": len(eligible_rows),
            "eligible_patient_count": len(
                {row["fold_owned_patient_key"] for row in eligible_rows}
            ),
            "excluded_record_count": len(exclusion_rows),
            "eligible_analysis_identity_roster_sha256": _eventnet._canonical_sha256(
                sorted(row["analysis_identity_id"] for row in eligible_rows)
            ),
            "typed_exclusion_ledger_sha256": _eventnet._canonical_sha256(
                exclusion_rows
            ),
            "all_phase_records_accounted_for": True,
            "prediction_first_denominator_preserved": True,
            "phase_reference_events_used_for_route_or_eligibility": False,
            "caller_owned_subset_accepted": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    result = _eventnet.AuthorizedEventNetVariantTrainingRoster(
        _roster_json=_eventnet._canonical_json_bytes(roster).decode("utf-8"),
        _receipt_json=_eventnet._canonical_json_bytes(receipt).decode("utf-8"),
        _validation_seal=_eventnet._VARIANT_TRAINING_ROSTER_AUTHORITY_SEAL,
    )
    _eventnet._require_authorized_eventnet_variant_training_roster(result)
    return result


def authorize_seizuretransformer_variant_training_roster_from_compact_inventory_v1(
    phase_authority: _st.AuthorizedSeizureTransformerFoldPhase,
    inventory_authority: AuthorizedDetectorProviderPreReferenceInventoryV1,
    *,
    variant_id: str,
    registry: Mapping[str, Any],
) -> _st.AuthorizedSeizureTransformerVariantTrainingRoster:
    """Issue the existing ST roster type without retaining transforms."""

    st_registry = _st._require_canonical_seizuretransformer_registry(registry)
    (
        phase,
        patient_by_identity,
        phase_receipt,
    ) = _st._require_authorized_seizuretransformer_fold_phase(phase_authority)
    _st._variant_profile(variant_id)
    manifest, rows, _inventory_receipt = _require_inventory(inventory_authority)
    if (
        manifest["provider_registry_bindings"]["seizuretransformer"]["registry_sha256"]
        != st_registry["registry_sha256"]
    ):
        raise ValueError("compact inventory binds another SeizureTransformer registry")
    outcomes = _inventory_rows_for_phase_variant(
        manifest=manifest, rows=rows, phase=phase, variant_id=variant_id
    )
    expected = {str(row["analysis_identity_id"]) for row in phase["records"]}
    support_policy_sha256 = _st.detector_channel_support_policy_receipt()[
        "policy_sha256"
    ]
    technical_policy_sha256 = _st._canonical_sha256(
        _st._st_pre_reference_technical_policy(variant_id)
    )
    eligible_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    for identity in sorted(expected):
        outcome = outcomes[identity]["eligibility_receipt"]
        if (
            outcome["registry_sha256"] != st_registry["registry_sha256"]
            or outcome["support_route_policy_sha256"] != support_policy_sha256
            or outcome["technical_eligibility_policy_sha256"] != technical_policy_sha256
        ):
            raise ValueError(
                "compact SeizureTransformer outcome policy binding drifted"
            )
        common = {
            "analysis_identity_id": identity,
            "fold_owned_patient_key": patient_by_identity[identity],
            "pre_reference_eligibility_receipt_sha256": outcome["receipt_sha256"],
            "support_route_receipt_sha256": outcome["support_route_receipt_sha256"],
            "technical_eligibility_receipt_sha256": outcome["receipt_sha256"],
        }
        if outcome["status"] == "eligible":
            eligible_rows.append(
                {
                    **common,
                    "provider_signal_lineage_authority_sha256": outcome[
                        "provider_signal_lineage_authority_sha256"
                    ],
                    "record_identity_authority_sha256": outcome[
                        "record_identity_authority_sha256"
                    ],
                    "transform_receipt_sha256": outcome["transform_receipt_sha256"],
                    "provider_target_sample_count": outcome[
                        "provider_target_sample_count"
                    ],
                }
            )
        else:
            exclusion_rows.append(
                {
                    **common,
                    "terminal_status": "technical_or_support_exclusion",
                    "reason_codes": outcome["reason_codes"],
                    "retained_in_full_prediction_first_benchmark_denominator": True,
                }
            )
    roster = {"eligible_records": eligible_rows, "typed_exclusions": exclusion_rows}
    receipt = _st._content_address(
        {
            "schema_version": "st_target_blind_variant_training_roster_authority_v1",
            "registry_sha256": st_registry["registry_sha256"],
            "variant_id": variant_id,
            "outer_fold": phase_receipt["outer_fold"],
            "phase": phase_receipt["phase"],
            "detector_fold_phase_receipt_sha256": phase_receipt[
                "detector_fold_phase_receipt_sha256"
            ],
            "support_route_policy_sha256": support_policy_sha256,
            "pre_reference_technical_eligibility_policy_sha256": technical_policy_sha256,
            "phase_record_count": len(expected),
            "eligible_record_count": len(eligible_rows),
            "eligible_patient_count": len(
                {row["fold_owned_patient_key"] for row in eligible_rows}
            ),
            "excluded_record_count": len(exclusion_rows),
            "eligible_analysis_identity_roster_sha256": _st._canonical_sha256(
                sorted(row["analysis_identity_id"] for row in eligible_rows)
            ),
            "typed_exclusion_ledger_sha256": _st._canonical_sha256(exclusion_rows),
            "all_phase_records_accounted_for": True,
            "prediction_first_denominator_preserved": True,
            "phase_reference_events_used_for_route_or_eligibility": False,
            "caller_owned_subset_accepted": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    result = _st.AuthorizedSeizureTransformerVariantTrainingRoster(
        _roster_json=_st._canonical_json_bytes(roster).decode("utf-8"),
        _receipt_json=_st._canonical_json_bytes(receipt).decode("utf-8"),
        _validation_seal=_st._VARIANT_TRAINING_ROSTER_SEAL,
    )
    _st._require_authorized_st_variant_training_roster(result)
    return result


def authorize_provider_variant_training_roster_from_compact_inventory_v1(
    phase_authority: object,
    inventory_authority: AuthorizedDetectorProviderPreReferenceInventoryV1,
    *,
    variant_id: str,
    registry: Mapping[str, Any],
) -> object:
    """Dispatch to the existing provider-specific opaque roster type."""

    if _provider_family(variant_id) == "eventnet":
        return authorize_eventnet_variant_training_roster_from_compact_inventory_v1(
            phase_authority,  # type: ignore[arg-type]
            inventory_authority,
            variant_id=variant_id,
            registry=registry,
        )
    return (
        authorize_seizuretransformer_variant_training_roster_from_compact_inventory_v1(
            phase_authority,  # type: ignore[arg-type]
            inventory_authority,
            variant_id=variant_id,
            registry=registry,
        )
    )


__all__ = [
    "AuthorizedDetectorProviderPreReferenceInventoryV1",
    "COMPACT_OUTCOME_SCHEMA_VERSION",
    "INVENTORY_AUTHORITY_SCHEMA_VERSION",
    "INVENTORY_SCHEMA_VERSION",
    "MATERIALIZER_ID",
    "PROVIDER_VARIANTS_V1",
    "ProviderOutcomeFactoryV1",
    "TargetFreeProviderSourceRecordV1",
    "authorize_detector_provider_pre_reference_inventory_v1",
    "authorize_eventnet_variant_training_roster_from_compact_inventory_v1",
    "authorize_provider_variant_training_roster_from_compact_inventory_v1",
    "authorize_seizuretransformer_variant_training_roster_from_compact_inventory_v1",
    "detector_provider_pre_reference_inventory_source_sha256_v1",
    "materialize_detector_provider_pre_reference_inventory_v1",
    "require_authorized_detector_provider_pre_reference_inventory_v1",
    "validate_detector_provider_pre_reference_inventory_evidence_v1",
]
