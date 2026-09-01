"""Formal target-free complete-candidate roster for complete-ITA projection.

The roster is frozen from a complete observed-native-EEG typed-unit inventory
and a reference-free occurrence inventory.  It has no model-logit, top-k,
target, annotation, spreadsheet, or clinical-text input surface.  Actual JSON
bytes are replayed before a process-local opaque authority is issued.

The authority fixes one spatial resolution for the complete recording:
physical electrode when that observed inventory is non-empty, otherwise whole
bipolar lead when that inventory is non-empty, otherwise unresolved.  The
choice is therefore invariant to event logits and rank masks.  Qualification
receipts are attached only after the candidate roster receipt is frozen and
cannot change candidate membership or record resolution.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final, Mapping, Sequence

from src.soz.geometry import STANDARD_19

from .ba_ieg_complete_ita_multievent_aggregation_v1 import (
    BA_IEG_PROCESSING_STATUSES_V1,
    BAIEGCompleteITAOccurrenceEntryV1,
    BAIEGCompleteITARecordRosterManifestV1,
    BAIEGReferenceFreeOccurrenceDedupReceiptV1,
)
from .ba_ieg_target_free_event_qualification_v1 import (
    BAIEGTargetFreeEventQualificationReceiptV1,
)


BA_IEG_COMPLETE_ITA_NATIVE_CANDIDATE_INVENTORY_SCHEMA_V1: Final[str] = (
    "ba_ieg_complete_ita_native_candidate_inventory_artifact_v1"
)
BA_IEG_COMPLETE_ITA_CANDIDATE_ROSTER_AUTHORITY_SCHEMA_V1: Final[str] = (
    "ba_ieg_complete_ita_candidate_roster_authority_v1"
)
BA_IEG_COMPLETE_ITA_CANDIDATE_ROSTER_MATERIALIZER_ID_V1: Final[str] = (
    "ba_ieg_complete_ita_candidate_roster_actual_bytes_materializer_v1"
)

_PHYSICAL_KIND: Final[str] = "physical_electrode"
_LEAD_KIND: Final[str] = "whole_bipolar_lead"
_UNRESOLVED_KIND: Final[str] = "unresolved"
_SHA_RE = re.compile(r"^[a-f0-9]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_AUTHORITY_SEAL = object()

_FIREWALL: Final[dict[str, bool]] = {
    "observed_native_EEG_typed_unit_inventory_used": True,
    "reference_free_candidate_occurrence_inventory_used": True,
    "model_logits_used": False,
    "positive_rank_mask_used": False,
    "top1_or_topk_used": False,
    "detector_score_used_for_membership_or_resolution": False,
    "seizure_or_SOZ_target_used": False,
    "EDF_annotation_used": False,
    "spreadsheet_or_Excel_used": False,
    "doctor_label_report_or_clinical_text_used": False,
    "clinical_history_video_or_behavior_used": False,
    "LLM_used": False,
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _self_hash(value: Mapping[str, Any], field_name: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field_name, None)
    return _sha(body)


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a canonical identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _exact(value: object, fields: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{context} fields drifted")
    return deepcopy(dict(value))


def _finite_interval(value: object, context: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise TypeError(f"{context} must be a two-value interval")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{context} must be numeric")
        number = float(item)
        if not (-float("inf") < number < float("inf")):
            raise ValueError(f"{context} must be finite")
        result.append(number)
    if result[0] < 0.0 or result[1] < result[0]:
        raise ValueError(f"{context} is negative or reversed")
    return result


def _normalize_unit(value: object) -> dict[str, Any]:
    row = _exact(
        value,
        {
            "unit_key",
            "unit_kind",
            "electrode_id",
            "whole_bipolar_lead",
            "reference_family_id",
            "inventory_state",
        },
        "native typed unit",
    )
    kind = row["unit_kind"]
    standard_index = {item: index for index, item in enumerate(STANDARD_19)}
    if kind == _PHYSICAL_KIND:
        electrode = row["electrode_id"]
        if (
            electrode not in STANDARD_19
            or row["whole_bipolar_lead"] is not None
            or row["unit_key"] != f"physical_electrode:{electrode}"
            or row["reference_family_id"] != "referential"
        ):
            raise ValueError("physical-electrode native inventory ontology drifted")
    elif kind == _LEAD_KIND:
        endpoints = row["whole_bipolar_lead"]
        if (
            row["electrode_id"] is not None
            or type(endpoints) is not list
            or len(endpoints) != 2
            or endpoints[0] not in STANDARD_19
            or endpoints[1] not in STANDARD_19
            or standard_index[endpoints[0]] >= standard_index[endpoints[1]]
            or row["unit_key"]
            != f"whole_bipolar_lead:{endpoints[0]}--{endpoints[1]}"
            or row["reference_family_id"] != "tcp_bipolar"
        ):
            raise ValueError("whole-bipolar-lead native inventory ontology drifted")
    else:
        raise ValueError("native inventory contains an unsupported resolution kind")
    if row["inventory_state"] != "observed_native_EEG_opportunity":
        raise PermissionError("candidate roster contains a non-observed native unit")
    return row


def _normalize_occurrence(value: object) -> dict[str, Any]:
    row = _exact(
        value,
        {
            "occurrence_id",
            "fragment_event_ids",
            "fragment_source_event_receipt_sha256s",
            "canonical_event_id",
            "reference_free_temporal_envelope_seconds",
            "processing_status",
            "processing_receipt_sha256",
        },
        "candidate occurrence",
    )
    occurrence_id = _identifier(row["occurrence_id"], "occurrence ID")
    event_ids = row["fragment_event_ids"]
    event_receipts = row["fragment_source_event_receipt_sha256s"]
    if (
        type(event_ids) is not list
        or type(event_receipts) is not list
        or not event_ids
        or len(event_ids) != len(event_receipts)
    ):
        raise ValueError("fragment IDs and receipts must be non-empty and aligned")
    aligned = sorted(
        (
            _identifier(event_id, "fragment event ID"),
            _sha256(receipt, "fragment source receipt"),
        )
        for event_id, receipt in zip(event_ids, event_receipts)
    )
    if len({item[0] for item in aligned}) != len(aligned):
        raise ValueError("one occurrence repeats a detector fragment")
    canonical = _identifier(row["canonical_event_id"], "canonical event ID")
    if canonical not in {item[0] for item in aligned}:
        raise ValueError("canonical event is outside its occurrence fragments")
    status = row["processing_status"]
    if status not in BA_IEG_PROCESSING_STATUSES_V1:
        raise ValueError("occurrence processing status is unsupported")
    return {
        "occurrence_id": occurrence_id,
        "fragment_event_ids": [item[0] for item in aligned],
        "fragment_source_event_receipt_sha256s": [item[1] for item in aligned],
        "canonical_event_id": canonical,
        "reference_free_temporal_envelope_seconds": _finite_interval(
            row["reference_free_temporal_envelope_seconds"],
            "reference-free temporal envelope",
        ),
        "processing_status": status,
        "processing_receipt_sha256": _sha256(
            row["processing_receipt_sha256"], "processing receipt"
        ),
    }


def _normalize_record(value: object) -> dict[str, Any]:
    row = _exact(
        value,
        {
            "recording_id",
            "source_signal_sha256",
            "native_inventory_receipt_sha256",
            "native_typed_unit_inventory",
            "occurrences",
        },
        "candidate inventory record",
    )
    units_value = row["native_typed_unit_inventory"]
    occurrences_value = row["occurrences"]
    if type(units_value) is not list or type(occurrences_value) is not list:
        raise TypeError("native units and occurrences must be arrays")
    units = sorted((_normalize_unit(item) for item in units_value), key=lambda item: item["unit_key"])
    if len({item["unit_key"] for item in units}) != len(units):
        raise ValueError("record native inventory repeats a typed unit")
    occurrences = sorted(
        (_normalize_occurrence(item) for item in occurrences_value),
        key=lambda item: (
            item["reference_free_temporal_envelope_seconds"],
            item["occurrence_id"],
        ),
    )
    if len({item["occurrence_id"] for item in occurrences}) != len(occurrences):
        raise ValueError("record repeats an occurrence")
    fragment_ids = [
        event_id
        for occurrence in occurrences
        for event_id in occurrence["fragment_event_ids"]
    ]
    if len(set(fragment_ids)) != len(fragment_ids):
        raise ValueError("one detector fragment enters multiple occurrences")
    return {
        "recording_id": _identifier(row["recording_id"], "recording ID"),
        "source_signal_sha256": _sha256(row["source_signal_sha256"], "source signal"),
        "native_inventory_receipt_sha256": _sha256(
            row["native_inventory_receipt_sha256"], "native inventory receipt"
        ),
        "native_typed_unit_inventory": units,
        "occurrences": occurrences,
    }


def _normalize_records(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise TypeError("complete candidate inventory records must be non-empty")
    records = sorted((_normalize_record(item) for item in value), key=lambda item: item["recording_id"])
    if len({item["recording_id"] for item in records}) != len(records):
        raise ValueError("complete candidate inventory repeats a recording")
    occurrence_ids: set[str] = set()
    fragment_ids: set[str] = set()
    for record in records:
        for occurrence in record["occurrences"]:
            if occurrence["occurrence_id"] in occurrence_ids:
                raise ValueError("occurrence ID is not globally unique")
            occurrence_ids.add(occurrence["occurrence_id"])
            overlap = fragment_ids.intersection(occurrence["fragment_event_ids"])
            if overlap:
                raise ValueError("fragment ID is not globally unique")
            fragment_ids.update(occurrence["fragment_event_ids"])
    return records


def build_ba_ieg_complete_ita_native_candidate_inventory_artifact_v1(
    *,
    inventory_id: str,
    source_inventory_receipt_sha256: str,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the target-free input artifact; no model evidence is accepted."""

    body: dict[str, Any] = {
        "schema_version": BA_IEG_COMPLETE_ITA_NATIVE_CANDIDATE_INVENTORY_SCHEMA_V1,
        "inventory_id": _identifier(inventory_id, "inventory ID"),
        "source_inventory_receipt_sha256": _sha256(
            source_inventory_receipt_sha256, "source inventory receipt"
        ),
        "records": _normalize_records(records),
        "input_firewall": deepcopy(_FIREWALL),
        "content_sha256": "",
    }
    body["content_sha256"] = _self_hash(body, "content_sha256")
    return body


def validate_ba_ieg_complete_ita_native_candidate_inventory_artifact_v1(
    value: object,
) -> dict[str, Any]:
    row = _exact(
        value,
        {
            "schema_version",
            "inventory_id",
            "source_inventory_receipt_sha256",
            "records",
            "input_firewall",
            "content_sha256",
        },
        "complete-ITA native candidate inventory",
    )
    if row["schema_version"] != BA_IEG_COMPLETE_ITA_NATIVE_CANDIDATE_INVENTORY_SCHEMA_V1:
        raise ValueError("complete candidate inventory schema drifted")
    _identifier(row["inventory_id"], "inventory ID")
    _sha256(row["source_inventory_receipt_sha256"], "source inventory receipt")
    if row["input_firewall"] != _FIREWALL:
        raise PermissionError("complete candidate inventory firewall drifted")
    observed_hash = _sha256(row["content_sha256"], "inventory content receipt")
    if observed_hash != _self_hash(row, "content_sha256"):
        raise ValueError("complete candidate inventory content receipt is stale")
    row["records"] = _normalize_records(row["records"])
    return row


def _read_bound_inventory(
    binding_value: object, *, artifact_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _exact(
        binding_value,
        {"relative_path", "file_bytes", "file_sha256", "content_sha256"},
        "candidate inventory artifact binding",
    )
    relative = PurePosixPath(binding["relative_path"])
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in {"", "."}:
        raise ValueError("candidate inventory artifact path is unsafe")
    path = Path(artifact_root).resolve(strict=True) / Path(*relative.parts)
    path = path.resolve(strict=True)
    try:
        path.relative_to(Path(artifact_root).resolve(strict=True))
    except ValueError as error:
        raise ValueError("candidate inventory artifact escapes its root") from error
    file_hash, file_bytes = _file_sha256(path)
    if (
        binding["file_bytes"] != file_bytes
        or _sha256(binding["file_sha256"], "candidate inventory file hash") != file_hash
    ):
        raise PermissionError("candidate inventory artifact bytes changed after binding")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("candidate inventory artifact is not strict JSON") from error
    inventory = validate_ba_ieg_complete_ita_native_candidate_inventory_artifact_v1(payload)
    if binding["content_sha256"] != inventory["content_sha256"]:
        raise PermissionError("candidate inventory semantic receipt changed after binding")
    return inventory, {
        "relative_path": relative.as_posix(),
        "file_bytes": file_bytes,
        "file_sha256": file_hash,
        "content_sha256": inventory["content_sha256"],
    }


def _record_resolution(units: Sequence[Mapping[str, Any]]) -> str:
    kinds = {item["unit_kind"] for item in units}
    if _PHYSICAL_KIND in kinds:
        return _PHYSICAL_KIND
    if _LEAD_KIND in kinds:
        return _LEAD_KIND
    return _UNRESOLVED_KIND


def _candidate_semantics(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return only immutable membership/resolution facts for roster hashing.

    Processing state and its receipt belong to the downstream aggregation
    manifest.  They may determine EEG evaluability, but cannot retrospectively
    redefine which native units, recordings, or detector occurrences were in
    the complete target-free candidate roster.
    """

    result: list[dict[str, Any]] = []
    for record in records:
        result.append(
            {
                "recording_id": record["recording_id"],
                "source_signal_sha256": record["source_signal_sha256"],
                "native_inventory_receipt_sha256": record[
                    "native_inventory_receipt_sha256"
                ],
                "native_typed_unit_inventory": deepcopy(
                    record["native_typed_unit_inventory"]
                ),
                "occurrences": [
                    {
                        "occurrence_id": occurrence["occurrence_id"],
                        "fragment_event_ids": list(
                            occurrence["fragment_event_ids"]
                        ),
                        "fragment_source_event_receipt_sha256s": list(
                            occurrence[
                                "fragment_source_event_receipt_sha256s"
                            ]
                        ),
                        "canonical_event_id": occurrence[
                            "canonical_event_id"
                        ],
                        "reference_free_temporal_envelope_seconds": list(
                            occurrence[
                                "reference_free_temporal_envelope_seconds"
                            ]
                        ),
                    }
                    for occurrence in record["occurrences"]
                ],
            }
        )
    return result


@dataclass(frozen=True)
class ValidatedBAIEGCompleteITACandidateRosterAuthorityV1:
    """Process-local authority issued only after actual inventory-byte replay."""

    _receipt_json: str = field(repr=False)
    _manifest: BAIEGCompleteITARecordRosterManifestV1 = field(repr=False)
    _seal: object = field(repr=False, compare=False)

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)

    @property
    def manifest(self) -> BAIEGCompleteITARecordRosterManifestV1:
        return deepcopy(self._manifest)


def require_validated_ba_ieg_complete_ita_candidate_roster_authority_v1(
    value: object,
) -> ValidatedBAIEGCompleteITACandidateRosterAuthorityV1:
    if (
        not isinstance(value, ValidatedBAIEGCompleteITACandidateRosterAuthorityV1)
        or value._seal is not _AUTHORITY_SEAL
    ):
        raise TypeError(
            "formal complete-ITA projection requires an opaque actual-byte-replayed candidate roster authority"
        )
    return value


def materialize_ba_ieg_complete_ita_candidate_roster_v1(
    *,
    inventory_artifact_binding: Mapping[str, Any],
    artifact_root: Path,
    qualification_receipt_by_occurrence_id: Mapping[
        str, BAIEGTargetFreeEventQualificationReceiptV1
    ],
) -> ValidatedBAIEGCompleteITACandidateRosterAuthorityV1:
    """Replay the complete inventory, freeze roster, and construct ITA manifest."""

    inventory, binding = _read_bound_inventory(
        inventory_artifact_binding, artifact_root=artifact_root
    )
    records = inventory["records"]
    semantic_body = {
        "schema_version": "ba_ieg_complete_ita_candidate_roster_semantics_v1",
        "source_inventory_receipt_sha256": inventory[
            "source_inventory_receipt_sha256"
        ],
        "records": _candidate_semantics(records),
        "input_firewall": deepcopy(_FIREWALL),
        "membership_and_resolution_source": (
            "complete_target_free_observed_native_EEG_inventory_only"
        ),
    }
    candidate_roster_receipt = _sha(semantic_body)
    expected_qualification_ids = {
        occurrence["occurrence_id"]
        for record in records
        for occurrence in record["occurrences"]
        if occurrence["processing_status"] in {"complete", "partial"}
    }
    if (
        not isinstance(qualification_receipt_by_occurrence_id, Mapping)
        or set(qualification_receipt_by_occurrence_id) != expected_qualification_ids
    ):
        raise PermissionError(
            "qualification receipt roster differs from all EEG-evaluable occurrences"
        )
    entries: list[BAIEGCompleteITAOccurrenceEntryV1] = []
    authority_records: list[dict[str, Any]] = []
    for record in records:
        record_entries: list[BAIEGCompleteITAOccurrenceEntryV1] = []
        for occurrence in record["occurrences"]:
            receipt = BAIEGReferenceFreeOccurrenceDedupReceiptV1(
                recording_id=record["recording_id"],
                occurrence_id=occurrence["occurrence_id"],
                fragment_event_ids=tuple(occurrence["fragment_event_ids"]),
                fragment_source_event_receipt_sha256s=tuple(
                    occurrence["fragment_source_event_receipt_sha256s"]
                ),
                canonical_event_id=occurrence["canonical_event_id"],
                reference_free_temporal_envelope_seconds=tuple(
                    occurrence["reference_free_temporal_envelope_seconds"]
                ),
                complete_candidate_roster_receipt_sha256=(
                    candidate_roster_receipt
                ),
            )
            qualification = qualification_receipt_by_occurrence_id.get(
                occurrence["occurrence_id"]
            )
            entry = BAIEGCompleteITAOccurrenceEntryV1(
                dedup_receipt=receipt,
                processing_status=occurrence["processing_status"],
                processing_receipt_sha256=occurrence[
                    "processing_receipt_sha256"
                ],
                qualification_receipt=qualification,
            )
            entries.append(entry)
            record_entries.append(entry)
        units = record["native_typed_unit_inventory"]
        resolution = _record_resolution(units)
        authority_records.append(
            {
                "recording_id": record["recording_id"],
                "source_signal_sha256": record["source_signal_sha256"],
                "native_inventory_receipt_sha256": record[
                    "native_inventory_receipt_sha256"
                ],
                "resolution_kind": resolution,
                "physical_electrode_unit_keys": [
                    item["unit_key"]
                    for item in units
                    if item["unit_kind"] == _PHYSICAL_KIND
                ],
                "whole_bipolar_lead_unit_keys": [
                    item["unit_key"]
                    for item in units
                    if item["unit_kind"] == _LEAD_KIND
                ],
                "occurrence_ids": [
                    item.dedup_receipt.occurrence_id
                    for item in sorted(
                        record_entries,
                        key=lambda entry: entry.dedup_receipt.occurrence_id,
                    )
                ],
                "eeg_evaluable_occurrence_ids": [
                    item.dedup_receipt.occurrence_id
                    for item in sorted(
                        record_entries,
                        key=lambda entry: entry.dedup_receipt.occurrence_id,
                    )
                    if item.eeg_evaluable
                ],
                "candidate_fragment_count": sum(
                    len(item.dedup_receipt.fragment_event_ids)
                    for item in record_entries
                ),
            }
        )
    manifest = BAIEGCompleteITARecordRosterManifestV1(
        recording_ids=tuple(record["recording_id"] for record in records),
        candidate_fragment_counts=tuple(
            row["candidate_fragment_count"] for row in authority_records
        ),
        occurrence_entries=tuple(entries),
        complete_candidate_roster_receipt_sha256=candidate_roster_receipt,
    )
    receipt: dict[str, Any] = {
        "schema_version": BA_IEG_COMPLETE_ITA_CANDIDATE_ROSTER_AUTHORITY_SCHEMA_V1,
        "materializer_id": BA_IEG_COMPLETE_ITA_CANDIDATE_ROSTER_MATERIALIZER_ID_V1,
        "inventory_artifact_binding": binding,
        "source_inventory_receipt_sha256": inventory[
            "source_inventory_receipt_sha256"
        ],
        "candidate_roster_receipt_sha256": candidate_roster_receipt,
        "manifest_sha256": manifest.manifest_sha256,
        "records": authority_records,
        "input_firewall": deepcopy(_FIREWALL),
        "authorization": {
            "formal_kind_separated_projection_authorized": True,
            "positive_rank_or_clinical_claim_authorized": False,
            "report_text_authorized": False,
        },
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")
    return ValidatedBAIEGCompleteITACandidateRosterAuthorityV1(
        _receipt_json=_canonical_bytes(receipt).decode("utf-8"),
        _manifest=manifest,
        _seal=_AUTHORITY_SEAL,
    )


__all__ = [
    "BA_IEG_COMPLETE_ITA_NATIVE_CANDIDATE_INVENTORY_SCHEMA_V1",
    "BA_IEG_COMPLETE_ITA_CANDIDATE_ROSTER_AUTHORITY_SCHEMA_V1",
    "BA_IEG_COMPLETE_ITA_CANDIDATE_ROSTER_MATERIALIZER_ID_V1",
    "ValidatedBAIEGCompleteITACandidateRosterAuthorityV1",
    "build_ba_ieg_complete_ita_native_candidate_inventory_artifact_v1",
    "validate_ba_ieg_complete_ita_native_candidate_inventory_artifact_v1",
    "materialize_ba_ieg_complete_ita_candidate_roster_v1",
    "require_validated_ba_ieg_complete_ita_candidate_roster_authority_v1",
]
