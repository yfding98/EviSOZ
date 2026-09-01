"""Aggregate, identity-free replay of the strict EDF header parser on TUSZ.

This audit opens only the exact byte ranges authorized by
``eeg_acquisition_header_allowlist_v1``.  Source-relative paths are consumed
solely to locate the already-canonicalized EDF and are never emitted.  EEG
samples, annotation payloads, patient/recording text and reference sidecars
are not opened.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Final, Mapping

from .eeg_acquisition_header_allowlist_v1 import (
    acquisition_header_parser_source_sha256_v1,
    build_eeg_acquisition_header_allowlist_policy_v1,
    parse_eeg_acquisition_header_v1,
)


SCHEMA_VERSION: Final[str] = "clinical_eeg_tusz_acquisition_header_full_replay_v1"
METHOD_ID: Final[str] = "strict_identity_free_header_replay_all_successful_source_outcomes_v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_tusz_acquisition_headers_v1(
    *, canonical_audit_path: str | Path, tusz_root: str | Path
) -> dict[str, Any]:
    audit_path = Path(canonical_audit_path).resolve(strict=True)
    root = Path(tusz_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    outcomes = audit.get("outcomes")
    if not isinstance(outcomes, list):
        raise ValueError("canonical audit outcomes are missing")

    accepted = 0
    eeg_signal_count_profile: Counter[str] = Counter()
    physical_units: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    forbidden_byte_reads: Counter[str] = Counter()
    policy_hashes: set[str] = set()
    parser_hashes: set[str] = set()
    external_binding_hashes: list[str] = []

    for outcome in outcomes:
        if not isinstance(outcome, Mapping) or outcome.get("failure") is not None:
            continue
        relative = Path(str(outcome.get("local_edf_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("canonical EDF path escapes TUSZ root")
        source = (root / relative).resolve(strict=True)
        try:
            source.relative_to(root)
        except ValueError as error:
            raise ValueError("canonical EDF path escapes TUSZ root") from error
        physical = outcome.get("physical_signal")
        if not isinstance(physical, Mapping):
            raise ValueError("canonical outcome lacks physical-signal binding")
        binding = physical.get("canonical_source_signal_sha256")
        if not isinstance(binding, str):
            raise ValueError("canonical source-signal binding is absent")
        receipt = parse_eeg_acquisition_header_v1(
            source,
            external_source_binding_sha256=binding,
        )
        accepted += 1
        external_binding_hashes.append(binding)
        policy_hashes.add(receipt["policy_receipt_sha256"])
        parser_hashes.add(receipt["parser_source_sha256"])
        eeg_signal_count_profile[str(receipt["eeg_signal_count"])] += 1
        for channel in receipt["channels"]:
            physical_units[str(channel["physical_dimension"])] += 1
        excluded["annotation"] += receipt["excluded_annotation_channel_count"]
        excluded["auxiliary"] += receipt["excluded_auxiliary_channel_count"]
        excluded["registered_non_target"] += receipt[
            "excluded_non_target_signal_count"
        ]
        for key, value in receipt["scope_receipt"].items():
            if key.endswith("_bytes_read"):
                forbidden_byte_reads[key] += int(value)

    if accepted == 0:
        raise ValueError("canonical audit has no successful source outcome")
    policy = build_eeg_acquisition_header_allowlist_policy_v1()
    if policy_hashes != {policy["policy_receipt_sha256"]}:
        raise ValueError("header policy binding varied during full replay")
    parser_source_hash = acquisition_header_parser_source_sha256_v1()
    if parser_hashes != {parser_source_hash}:
        raise ValueError("header parser source binding varied during full replay")
    if any(forbidden_byte_reads.values()):
        raise PermissionError("full header replay opened a forbidden byte range")

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "canonical_audit_file_sha256": _file_sha256(audit_path),
        "parser_source_sha256": parser_source_hash,
        "policy_receipt_sha256": policy["policy_receipt_sha256"],
        "successful_source_outcome_count": accepted,
        "accepted_source_header_count": accepted,
        "rejected_source_header_count": 0,
        "eeg_signal_count_profile": dict(sorted(eeg_signal_count_profile.items())),
        "retained_scalp_EEG_channel_count": sum(physical_units.values()),
        "canonical_physical_dimension_channel_counts": dict(
            sorted(physical_units.items())
        ),
        "excluded_signal_row_counts": dict(sorted(excluded.items())),
        "external_source_binding_roster_sha256": _sha256(
            sorted(external_binding_hashes)
        ),
        "aggregate_forbidden_byte_read_counts": dict(
            sorted(forbidden_byte_reads.items())
        ),
        "EDF_sample_payload_opened": False,
        "EDF_annotation_payload_opened": False,
        "reference_sidecar_or_spreadsheet_opened": False,
        "path_patient_or_recording_identity_emitted": False,
        "scientific_interpretation": {
            "all_successful_source_EDF_headers_accepted": True,
            "optional_EEG_prefix_alone_authorizes_channel": False,
            "registered_auxiliary_or_non_target_rows_are_EEG_evidence": False,
            "performance_or_clinical_claim_authorized": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    receipt["receipt_sha256"] = _sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    return receipt


def validate_tusz_acquisition_header_full_replay_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("full header replay receipt must be an object")
    receipt = deepcopy(dict(value))
    observed = receipt.get("receipt_sha256")
    expected = _sha256(
        {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    )
    if observed != expected:
        raise ValueError("full header replay receipt hash mismatch")
    if receipt.get("schema_version") != SCHEMA_VERSION or receipt.get(
        "method_id"
    ) != METHOD_ID:
        raise ValueError("full header replay schema or method drifted")
    if receipt.get("successful_source_outcome_count") != receipt.get(
        "accepted_source_header_count"
    ) or receipt.get("rejected_source_header_count") != 0:
        raise ValueError("full header replay denominator does not close")
    if receipt.get("retained_scalp_EEG_channel_count") != sum(
        receipt.get("canonical_physical_dimension_channel_counts", {}).values()
    ):
        raise ValueError("full header replay EEG channel denominator does not close")
    forbidden = receipt.get("aggregate_forbidden_byte_read_counts")
    if not isinstance(forbidden, Mapping) or any(forbidden.values()):
        raise PermissionError("full header replay opened forbidden bytes")
    for field in (
        "EDF_sample_payload_opened",
        "EDF_annotation_payload_opened",
        "reference_sidecar_or_spreadsheet_opened",
        "path_patient_or_recording_identity_emitted",
    ):
        if receipt.get(field) is not False:
            raise PermissionError(f"full header replay source firewall opened: {field}")
    return receipt


__all__ = [
    "METHOD_ID",
    "SCHEMA_VERSION",
    "audit_tusz_acquisition_headers_v1",
    "validate_tusz_acquisition_header_full_replay_v1",
]
