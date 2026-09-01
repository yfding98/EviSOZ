"""A0-only oracle navigation and canonical identity contracts for BA-IEG.

This module deliberately does *not* implement a seizure detector.  A0 uses a
public TUSZ seizure interval solely to choose an EEG analysis support and is
therefore an upper-bound experiment conditional on oracle navigation.  Its
receipts have a schema and authority surface that are intentionally
incompatible with adaptive-search/detector receipts.

The identity bridge is similarly narrow.  It proves the exact join

``TUSZREC -> EDF container SHA-256 -> canonical EEGREC``

without exposing EDF annotations, spreadsheet fields, clinical text, seizure
type, or localization-channel targets to model materialization.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, Mapping

from .ba_ieg_a0_oracle_navigation_candidate_roster_v1 import (
    BA_IEG_NAVIGATION_ARM_A0,
    validate_ba_ieg_a0_oracle_navigation_candidate_roster_v1,
)
from .canonical_edf_materialization import (
    CanonicalEDFViewBundle,
    validate_canonical_edf_materialization,
)
from .deepsoz_tusz_identity_binding_v1 import (
    validate_deepsoz_tusz_source_train_identity_binding_v1,
)


BA_IEG_A0_CANONICAL_IDENTITY_BINDING_SCHEMA_V1: Final[str] = (
    "ba_ieg_a0_tusz_container_canonical_identity_binding_v1"
)
BA_IEG_A0_NAVIGATION_WINDOW_SCHEMA_V1: Final[str] = (
    "ba_ieg_a0_oracle_navigation_initial_watchdog_support_v1"
)
BA_IEG_A0_EVALUATION_SEMANTICS_V1: Final[str] = (
    "conditional_on_seizure_interval_upper_bound"
)
BA_IEG_A0_SUPPORT_POLICY_ID_V1: Final[str] = (
    "initial_onset_centered_bootstrap_watchdog_support_v1"
)

_SHA256_CHARACTERS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_IDENTITY_SCOPE: Final[dict[str, bool]] = {
    "source_edf_container_bytes_hashed": True,
    "canonical_physical_eeg_signal_receipt_bound": True,
    "edf_annotations_opened": False,
    "edf_clinical_header_identity_used": False,
    "spreadsheet_opened": False,
    "clinical_text_opened": False,
    "localization_channel_target_opened": False,
    "identity_binding_available_to_model_forward": False,
}
_NAVIGATION_SCOPE: Final[dict[str, Any]] = {
    "evaluation_semantics": BA_IEG_A0_EVALUATION_SEMANTICS_V1,
    "navigation_arm": BA_IEG_NAVIGATION_ARM_A0,
    "public_tusz_seizure_interval_used_for_navigation": True,
    "oracle_interval_used": True,
    "detector_output_used": False,
    "detector_receipt_claimed": False,
    "detector_frozen_claim_authorized": False,
    "localization_channel_target_used": False,
    "seizure_type_used": False,
    "oracle_interval_available_to_model_forward": False,
    "initial_support_only": True,
    "fixed_watchdog_is_final_analysis_window": False,
    "final_support_requires_iterative_rule_adaptive_acquisition": True,
    "iterative_rule_adaptive_acquisition_status": "not_materialized",
    "edf_annotations_opened": False,
    "spreadsheet_opened": False,
    "clinical_text_opened": False,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _identifier(value: object, name: str, *, prefix: str | None = None) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 2048
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be a valid non-empty trimmed identifier")
    if prefix is not None and not value.startswith(prefix):
        raise ValueError(f"{name} must start with {prefix}")
    return value


def _file_sha256(path: Path) -> tuple[int, str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("source EDF must be a non-symlink regular file")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    if size < 1:
        raise ValueError("source EDF container cannot be empty")
    return size, digest.hexdigest()


def _finalize(
    body: Mapping[str, Any], *, id_field: str, id_prefix: str
) -> dict[str, Any]:
    result = deepcopy(dict(body))
    result[id_field] = "CONTENT-ADDRESS-PENDING"
    result["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    result[id_field] = id_prefix + _canonical_sha256(result)[:24]
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def _validate_content_address(
    data: Mapping[str, Any], *, id_field: str, id_prefix: str
) -> None:
    _sha256(data["receipt_sha256"], "receipt_sha256")
    digest_source = deepcopy(dict(data))
    digest_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("A0 receipt hash does not bind its content")
    id_source = deepcopy(dict(data))
    id_source[id_field] = "CONTENT-ADDRESS-PENDING"
    id_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data[id_field] != id_prefix + _canonical_sha256(id_source)[:24]:
        raise ValueError("A0 receipt ID does not bind its content")


def _record_lookup(
    roster: Mapping[str, Any], model_recording_id: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in roster["records"]
        if row["model_recording_id"] == model_recording_id
    ]
    if len(matches) != 1:
        raise ValueError("model recording ID is not unique in the A0 roster")
    return matches[0]


def _identity_record_lookup(
    identity: Mapping[str, Any], model_recording_id: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in identity["records"]
        if row["detector_recording_id"] == model_recording_id
    ]
    if len(matches) != 1:
        raise ValueError("model recording ID is not unique in the identity binding")
    return matches[0]


def build_ba_ieg_a0_canonical_identity_binding_v1(
    *,
    candidate_roster: Mapping[str, Any],
    source_identity_binding: Mapping[str, Any],
    model_recording_id: str,
    source_edf_path: str | Path,
    canonical_bundle: CanonicalEDFViewBundle,
) -> dict[str, Any]:
    """Bind one frozen A0 TUSZ record to its independently loaded EEGREC."""

    validate_ba_ieg_a0_oracle_navigation_candidate_roster_v1(candidate_roster)
    roster = deepcopy(dict(candidate_roster))
    validate_deepsoz_tusz_source_train_identity_binding_v1(
        source_identity_binding
    )
    identity = deepcopy(dict(source_identity_binding))
    model_id = _identifier(
        model_recording_id, "model_recording_id", prefix="TUSZREC-"
    )
    if roster["identity_binding_sha256"] != identity["receipt_sha256"]:
        raise ValueError("A0 roster and source identity binding disagree")
    roster_row = _record_lookup(roster, model_id)
    identity_row = _identity_record_lookup(identity, model_id)
    expected_pairs = {
        "source_recording_id": (
            roster_row["source_recording_id"],
            identity_row["tusz_recording_id"],
        ),
        "patient_uid": (roster_row["patient_uid"], identity_row["patient_uid"]),
        "source_container_sha256": (
            roster_row["source_container_sha256"],
            identity_row["source_container_sha256"],
        ),
        "exact_container_equivalence_id": (
            roster_row["exact_container_equivalence_id"],
            identity_row["exact_container_equivalence_id"],
        ),
        "recording_duration_fraction": (
            roster_row["recording_duration_fraction"],
            identity_row["recording_duration_fraction"],
        ),
    }
    for name, pair in expected_pairs.items():
        if pair[0] != pair[1]:
            raise ValueError(f"A0 roster/identity {name} drifted")

    if not isinstance(canonical_bundle, CanonicalEDFViewBundle):
        raise TypeError("canonical_bundle must be CanonicalEDFViewBundle")
    materialization = validate_canonical_edf_materialization(canonical_bundle)
    canonical = canonical_bundle.canonical_record.canonical_receipt
    source_header = canonical_bundle.canonical_record.source_header_receipt
    source_path = Path(source_edf_path)
    container_bytes, container_sha256 = _file_sha256(source_path)
    if (
        container_sha256 != roster_row["source_container_sha256"]
        or container_bytes != identity_row["container_bytes"]
    ):
        raise ValueError("source EDF bytes do not match the frozen TUSZ container")
    duration_fraction = roster_row["recording_duration_fraction"]
    expected_duration = float(duration_fraction[0]) / float(duration_fraction[1])
    if abs(float(canonical["recording_duration_seconds"]) - expected_duration) > 1e-6:
        raise ValueError("canonical EEG duration disagrees with the frozen record")

    body = {
        "schema_version": BA_IEG_A0_CANONICAL_IDENTITY_BINDING_SCHEMA_V1,
        "binding_id": "CONTENT-ADDRESS-PENDING",
        "a0_candidate_roster_receipt_sha256": roster["receipt_sha256"],
        "a0_oracle_navigation_receipt_sha256": roster[
            "oracle_navigation_receipt_sha256"
        ],
        "source_identity_binding_receipt_sha256": identity["receipt_sha256"],
        "source_identity_row_sha256": _canonical_sha256(identity_row),
        "record_roster_receipt_sha256": roster_row[
            "record_roster_receipt_sha256"
        ],
        "patient_uid": roster_row["patient_uid"],
        "model_recording_id": model_id,
        "source_recording_id": roster_row["source_recording_id"],
        "source_container": {
            "sha256": container_sha256,
            "size_bytes": container_bytes,
            "exact_container_equivalence_id": roster_row[
                "exact_container_equivalence_id"
            ],
        },
        "canonical_signal": {
            "recording_id": canonical["recording_id"],
            "canonical_signal_id": canonical["canonical_signal_id"],
            "source_signal_sha256": canonical["source_signal_sha256"],
            "canonical_receipt_sha256": canonical["receipt_sha256"],
            "canonical_materialization_receipt_sha256": materialization[
                "receipt_sha256"
            ],
            "source_header_receipt_sha256": source_header["receipt_sha256"],
            "recording_duration_seconds": float(
                canonical["recording_duration_seconds"]
            ),
            "observed_channel_ids": list(
                canonical_bundle.canonical_record.observed_channel_ids
            ),
        },
        "scope_receipt": deepcopy(_IDENTITY_SCOPE),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    return validate_ba_ieg_a0_canonical_identity_binding_v1(
        _finalize(body, id_field="binding_id", id_prefix="BAIEG-A0-ID-")
    )


def validate_ba_ieg_a0_canonical_identity_binding_v1(
    payload: object,
    *,
    candidate_roster: Mapping[str, Any] | None = None,
    source_identity_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "binding_id",
        "a0_candidate_roster_receipt_sha256",
        "a0_oracle_navigation_receipt_sha256",
        "source_identity_binding_receipt_sha256",
        "source_identity_row_sha256",
        "record_roster_receipt_sha256",
        "patient_uid",
        "model_recording_id",
        "source_recording_id",
        "source_container",
        "canonical_signal",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("A0 canonical identity binding has missing/unknown fields")
    data = deepcopy(payload)
    if (
        data["schema_version"]
        != BA_IEG_A0_CANONICAL_IDENTITY_BINDING_SCHEMA_V1
        or data["scope_receipt"] != _IDENTITY_SCOPE
    ):
        raise ValueError("A0 canonical identity binding contract drifted")
    _identifier(data["binding_id"], "binding_id", prefix="BAIEG-A0-ID-")
    _identifier(data["patient_uid"], "patient_uid")
    _identifier(
        data["model_recording_id"], "model_recording_id", prefix="TUSZREC-"
    )
    _identifier(data["source_recording_id"], "source_recording_id")
    for name in (
        "a0_candidate_roster_receipt_sha256",
        "a0_oracle_navigation_receipt_sha256",
        "source_identity_binding_receipt_sha256",
        "source_identity_row_sha256",
        "record_roster_receipt_sha256",
    ):
        _sha256(data[name], name)
    container = data["source_container"]
    if type(container) is not dict or set(container) != {
        "sha256",
        "size_bytes",
        "exact_container_equivalence_id",
    }:
        raise ValueError("A0 source-container binding drifted")
    _sha256(container["sha256"], "source_container.sha256")
    if (
        isinstance(container["size_bytes"], bool)
        or not isinstance(container["size_bytes"], int)
        or container["size_bytes"] < 1
    ):
        raise ValueError("source container size must be positive")
    if container["exact_container_equivalence_id"] != (
        "TUSZ-EDF-CONTAINER-" + container["sha256"]
    ):
        raise ValueError("exact container equivalence ID drifted")
    canonical = data["canonical_signal"]
    if type(canonical) is not dict or set(canonical) != {
        "recording_id",
        "canonical_signal_id",
        "source_signal_sha256",
        "canonical_receipt_sha256",
        "canonical_materialization_receipt_sha256",
        "source_header_receipt_sha256",
        "recording_duration_seconds",
        "observed_channel_ids",
    }:
        raise ValueError("A0 canonical-signal binding drifted")
    _identifier(canonical["recording_id"], "canonical recording_id", prefix="EEGREC-")
    _identifier(canonical["canonical_signal_id"], "canonical_signal_id")
    for name in (
        "source_signal_sha256",
        "canonical_receipt_sha256",
        "canonical_materialization_receipt_sha256",
        "source_header_receipt_sha256",
    ):
        _sha256(canonical[name], f"canonical_signal.{name}")
    duration = canonical["recording_duration_seconds"]
    if not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or float(duration) <= 0:
        raise ValueError("canonical recording duration must be positive")
    if (
        not isinstance(canonical["observed_channel_ids"], list)
        or not canonical["observed_channel_ids"]
        or len(canonical["observed_channel_ids"])
        != len(set(canonical["observed_channel_ids"]))
    ):
        raise ValueError("canonical observed-channel roster is invalid")
    _validate_content_address(
        data, id_field="binding_id", id_prefix="BAIEG-A0-ID-"
    )

    if candidate_roster is not None:
        validate_ba_ieg_a0_oracle_navigation_candidate_roster_v1(
            candidate_roster
        )
        roster = deepcopy(dict(candidate_roster))
        row = _record_lookup(roster, data["model_recording_id"])
        expected = {
            "a0_candidate_roster_receipt_sha256": roster["receipt_sha256"],
            "a0_oracle_navigation_receipt_sha256": roster[
                "oracle_navigation_receipt_sha256"
            ],
            "record_roster_receipt_sha256": row["record_roster_receipt_sha256"],
            "patient_uid": row["patient_uid"],
            "source_recording_id": row["source_recording_id"],
        }
        for name, value in expected.items():
            if data[name] != value:
                raise ValueError(f"A0 identity {name} disagrees with roster")
        if data["source_container"]["sha256"] != row["source_container_sha256"]:
            raise ValueError("A0 identity container disagrees with roster")
    if source_identity_binding is not None:
        validate_deepsoz_tusz_source_train_identity_binding_v1(
            source_identity_binding
        )
        identity = deepcopy(dict(source_identity_binding))
        row = _identity_record_lookup(identity, data["model_recording_id"])
        if (
            data["source_identity_binding_receipt_sha256"]
            != identity["receipt_sha256"]
            or data["source_identity_row_sha256"] != _canonical_sha256(row)
            or data["source_container"]["sha256"]
            != row["source_container_sha256"]
        ):
            raise ValueError("A0 identity disagrees with source identity authority")
    return data


@dataclass(frozen=True)
class BAIEGA0NavigationSupportPolicyV1:
    """Initial physical-time watchdog support around the A0 oracle anchor.

    This is only the bootstrap/smoke support used to start signal analysis. It
    is not the final variable-length acquisition proposed by the BA-IEG method.
    A final analysis window must be produced by a separately versioned,
    iterative rule-adaptive acquisition and compared against fixed/progressive
    controls.
    """

    pre_anchor_seconds: float = 12.0
    post_anchor_seconds: float = 48.0
    policy_id: str = BA_IEG_A0_SUPPORT_POLICY_ID_V1

    def __post_init__(self) -> None:
        if self.policy_id != BA_IEG_A0_SUPPORT_POLICY_ID_V1:
            raise ValueError("A0 support policy ID drifted")
        for name in ("pre_anchor_seconds", "post_anchor_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ba_ieg_a0_navigation_support_policy_v1",
            "policy_id": self.policy_id,
            "pre_anchor_seconds": float(self.pre_anchor_seconds),
            "post_anchor_seconds": float(self.post_anchor_seconds),
            "anchor_source": "public_tusz_seizure_interval_start",
            "interval_clipped_to_recording_support": True,
            "support_role": "initial_bootstrap_watchdog_only",
            "initial_support_only": True,
            "fixed_watchdog_is_final_analysis_window": False,
            "final_support_requires_iterative_rule_adaptive_acquisition": True,
            "iterative_rule_adaptive_acquisition_status": "not_materialized",
            "event_offset_used_to_expand_model_input_support": False,
            "localization_target_used_to_choose_support": False,
        }


def build_ba_ieg_a0_navigation_window_v1(
    *,
    candidate_roster: Mapping[str, Any],
    canonical_identity_binding: Mapping[str, Any],
    model_event_id: str,
    policy: BAIEGA0NavigationSupportPolicyV1 = BAIEGA0NavigationSupportPolicyV1(),
) -> dict[str, Any]:
    """Build an oracle-navigation window that cannot be mistaken for detection."""

    validate_ba_ieg_a0_oracle_navigation_candidate_roster_v1(candidate_roster)
    roster = deepcopy(dict(candidate_roster))
    binding = validate_ba_ieg_a0_canonical_identity_binding_v1(
        canonical_identity_binding, candidate_roster=roster
    )
    if not isinstance(policy, BAIEGA0NavigationSupportPolicyV1):
        raise TypeError("policy must be BAIEGA0NavigationSupportPolicyV1")
    event_id = _identifier(model_event_id, "model_event_id", prefix="BAIEG-A0EVT-")
    matches = [row for row in roster["events"] if row["model_event_id"] == event_id]
    if len(matches) != 1:
        raise ValueError("A0 event ID is not unique in the candidate roster")
    event = matches[0]
    if (
        event["model_recording_id"] != binding["model_recording_id"]
        or event["patient_uid"] != binding["patient_uid"]
        or event["source_recording_id"] != binding["source_recording_id"]
    ):
        raise ValueError("A0 event and canonical identity binding disagree")
    seizure_start, seizure_stop = map(float, event["seizure_interval_seconds"])
    duration = float(binding["canonical_signal"]["recording_duration_seconds"])
    anchor = seizure_start
    analysis_start = max(0.0, anchor - float(policy.pre_anchor_seconds))
    analysis_stop = min(duration, anchor + float(policy.post_anchor_seconds))
    if not 0.0 <= analysis_start <= anchor < analysis_stop <= duration + 1e-6:
        raise ValueError("A0 policy produced an invalid analysis interval")
    body = {
        "schema_version": BA_IEG_A0_NAVIGATION_WINDOW_SCHEMA_V1,
        "window_id": "CONTENT-ADDRESS-PENDING",
        "navigation_arm": BA_IEG_NAVIGATION_ARM_A0,
        "evaluation_semantics": BA_IEG_A0_EVALUATION_SEMANTICS_V1,
        "a0_candidate_roster_receipt_sha256": roster["receipt_sha256"],
        "a0_oracle_navigation_receipt_sha256": roster[
            "oracle_navigation_receipt_sha256"
        ],
        "canonical_identity_binding_receipt_sha256": binding["receipt_sha256"],
        "canonical_signal_binding": deepcopy(binding["canonical_signal"]),
        "event_identity": {
            "event_id": event_id,
            "event_receipt_sha256": event["event_receipt_sha256"],
            "recording_id": binding["canonical_signal"]["recording_id"],
            "model_recording_id": binding["model_recording_id"],
            "patient_uid": binding["patient_uid"],
            "model_split": "source_train",
        },
        "timing": {
            "recording_duration_seconds": duration,
            "public_seizure_interval_seconds": [seizure_start, seizure_stop],
            "navigation_anchor_seconds": anchor,
            "analysis_interval_recording_seconds": [
                analysis_start,
                analysis_stop,
            ],
            "baseline_context_recording_seconds": (
                [analysis_start, anchor] if anchor > analysis_start else None
            ),
            "analysis_support_clipped_left": analysis_start == 0.0
            and anchor < float(policy.pre_anchor_seconds),
            "analysis_support_clipped_right": analysis_stop == duration
            and anchor + float(policy.post_anchor_seconds) > duration,
            "oracle_offset_inside_analysis_support": seizure_stop <= analysis_stop,
        },
        "support_policy": policy.to_dict(),
        "support_policy_sha256": policy.sha256,
        "scope_receipt": deepcopy(_NAVIGATION_SCOPE),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    return validate_ba_ieg_a0_navigation_window_v1(
        _finalize(body, id_field="window_id", id_prefix="BAIEG-A0-WIN-"),
        candidate_roster=roster,
        canonical_identity_binding=binding,
    )


def validate_ba_ieg_a0_navigation_window_v1(
    payload: object,
    *,
    candidate_roster: Mapping[str, Any] | None = None,
    canonical_identity_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "window_id",
        "navigation_arm",
        "evaluation_semantics",
        "a0_candidate_roster_receipt_sha256",
        "a0_oracle_navigation_receipt_sha256",
        "canonical_identity_binding_receipt_sha256",
        "canonical_signal_binding",
        "event_identity",
        "timing",
        "support_policy",
        "support_policy_sha256",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("A0 navigation window has missing/unknown fields")
    data = deepcopy(payload)
    if (
        data["schema_version"] != BA_IEG_A0_NAVIGATION_WINDOW_SCHEMA_V1
        or data["navigation_arm"] != BA_IEG_NAVIGATION_ARM_A0
        or data["evaluation_semantics"] != BA_IEG_A0_EVALUATION_SEMANTICS_V1
        or data["scope_receipt"] != _NAVIGATION_SCOPE
    ):
        raise ValueError("A0 navigation authority drifted")
    _identifier(data["window_id"], "window_id", prefix="BAIEG-A0-WIN-")
    for name in (
        "a0_candidate_roster_receipt_sha256",
        "a0_oracle_navigation_receipt_sha256",
        "canonical_identity_binding_receipt_sha256",
        "support_policy_sha256",
    ):
        _sha256(data[name], name)
    canonical_signal = data["canonical_signal_binding"]
    if type(canonical_signal) is not dict or set(canonical_signal) != {
        "recording_id",
        "canonical_signal_id",
        "source_signal_sha256",
        "canonical_receipt_sha256",
        "canonical_materialization_receipt_sha256",
        "source_header_receipt_sha256",
        "recording_duration_seconds",
        "observed_channel_ids",
    }:
        raise ValueError("A0 navigation canonical-signal binding drifted")
    _identifier(
        canonical_signal["recording_id"],
        "canonical_signal_binding.recording_id",
        prefix="EEGREC-",
    )
    _identifier(
        canonical_signal["canonical_signal_id"],
        "canonical_signal_binding.canonical_signal_id",
    )
    for name in (
        "source_signal_sha256",
        "canonical_receipt_sha256",
        "canonical_materialization_receipt_sha256",
        "source_header_receipt_sha256",
    ):
        _sha256(canonical_signal[name], f"canonical_signal_binding.{name}")
    if (
        not isinstance(canonical_signal["recording_duration_seconds"], (int, float))
        or not math.isfinite(float(canonical_signal["recording_duration_seconds"]))
        or float(canonical_signal["recording_duration_seconds"]) <= 0
        or not isinstance(canonical_signal["observed_channel_ids"], list)
        or not canonical_signal["observed_channel_ids"]
    ):
        raise ValueError("A0 navigation canonical-signal support is invalid")
    identity = data["event_identity"]
    if type(identity) is not dict or set(identity) != {
        "event_id",
        "event_receipt_sha256",
        "recording_id",
        "model_recording_id",
        "patient_uid",
        "model_split",
    }:
        raise ValueError("A0 navigation event identity drifted")
    _identifier(identity["event_id"], "event_id", prefix="BAIEG-A0EVT-")
    _sha256(identity["event_receipt_sha256"], "event_receipt_sha256")
    _identifier(identity["recording_id"], "recording_id", prefix="EEGREC-")
    _identifier(identity["model_recording_id"], "model_recording_id", prefix="TUSZREC-")
    _identifier(identity["patient_uid"], "patient_uid")
    if identity["model_split"] != "source_train":
        raise ValueError("A0 navigation is source-train-only")
    if identity["recording_id"] != canonical_signal["recording_id"]:
        raise ValueError("A0 event/canonical recording identity drifted")
    timing = data["timing"]
    if type(timing) is not dict or set(timing) != {
        "recording_duration_seconds",
        "public_seizure_interval_seconds",
        "navigation_anchor_seconds",
        "analysis_interval_recording_seconds",
        "baseline_context_recording_seconds",
        "analysis_support_clipped_left",
        "analysis_support_clipped_right",
        "oracle_offset_inside_analysis_support",
    }:
        raise ValueError("A0 navigation timing drifted")
    duration = float(timing["recording_duration_seconds"])
    seizure = tuple(map(float, timing["public_seizure_interval_seconds"]))
    analysis = tuple(map(float, timing["analysis_interval_recording_seconds"]))
    anchor = float(timing["navigation_anchor_seconds"])
    if (
        not math.isfinite(duration)
        or duration <= 0
        or len(seizure) != 2
        or len(analysis) != 2
        or not 0 <= seizure[0] < seizure[1] <= duration + 1e-6
        or not 0 <= analysis[0] <= anchor < analysis[1] <= duration + 1e-6
        or anchor != seizure[0]
        or duration != float(canonical_signal["recording_duration_seconds"])
    ):
        raise ValueError("A0 navigation physical timing is invalid")
    baseline = timing["baseline_context_recording_seconds"]
    if baseline is not None and list(map(float, baseline)) != [analysis[0], anchor]:
        raise ValueError("A0 baseline support drifted")
    for name in (
        "analysis_support_clipped_left",
        "analysis_support_clipped_right",
        "oracle_offset_inside_analysis_support",
    ):
        if type(timing[name]) is not bool:
            raise TypeError(f"{name} must be boolean")
    policy = data["support_policy"]
    expected_policy = BAIEGA0NavigationSupportPolicyV1(
        pre_anchor_seconds=float(policy.get("pre_anchor_seconds")),
        post_anchor_seconds=float(policy.get("post_anchor_seconds")),
        policy_id=str(policy.get("policy_id")),
    )
    if policy != expected_policy.to_dict() or data["support_policy_sha256"] != expected_policy.sha256:
        raise ValueError("A0 navigation support policy did not replay")
    expected_start = max(0.0, anchor - expected_policy.pre_anchor_seconds)
    expected_stop = min(duration, anchor + expected_policy.post_anchor_seconds)
    expected_timing = {
        "recording_duration_seconds": duration,
        "public_seizure_interval_seconds": list(seizure),
        "navigation_anchor_seconds": anchor,
        "analysis_interval_recording_seconds": [expected_start, expected_stop],
        "baseline_context_recording_seconds": (
            [expected_start, anchor] if anchor > expected_start else None
        ),
        "analysis_support_clipped_left": expected_start == 0.0
        and anchor < expected_policy.pre_anchor_seconds,
        "analysis_support_clipped_right": expected_stop == duration
        and anchor + expected_policy.post_anchor_seconds > duration,
        "oracle_offset_inside_analysis_support": seizure[1] <= expected_stop,
    }
    if timing != expected_timing:
        raise ValueError("A0 navigation timing did not replay from policy")
    _validate_content_address(
        data, id_field="window_id", id_prefix="BAIEG-A0-WIN-"
    )

    if candidate_roster is not None:
        validate_ba_ieg_a0_oracle_navigation_candidate_roster_v1(
            candidate_roster
        )
        roster = deepcopy(dict(candidate_roster))
        matches = [
            row
            for row in roster["events"]
            if row["model_event_id"] == identity["event_id"]
        ]
        if len(matches) != 1:
            raise ValueError("A0 navigation event is absent from roster")
        event = matches[0]
        if (
            data["a0_candidate_roster_receipt_sha256"] != roster["receipt_sha256"]
            or data["a0_oracle_navigation_receipt_sha256"]
            != roster["oracle_navigation_receipt_sha256"]
            or identity["event_receipt_sha256"] != event["event_receipt_sha256"]
            or identity["model_recording_id"] != event["model_recording_id"]
            or identity["patient_uid"] != event["patient_uid"]
            or timing["public_seizure_interval_seconds"]
            != event["seizure_interval_seconds"]
        ):
            raise ValueError("A0 navigation window disagrees with candidate roster")
    if canonical_identity_binding is not None:
        binding = validate_ba_ieg_a0_canonical_identity_binding_v1(
            canonical_identity_binding,
            candidate_roster=candidate_roster,
        )
        if (
            data["canonical_identity_binding_receipt_sha256"]
            != binding["receipt_sha256"]
            or identity["recording_id"]
            != binding["canonical_signal"]["recording_id"]
            or identity["model_recording_id"] != binding["model_recording_id"]
            or identity["patient_uid"] != binding["patient_uid"]
            or canonical_signal != binding["canonical_signal"]
            or duration
            != float(binding["canonical_signal"]["recording_duration_seconds"])
        ):
            raise ValueError("A0 navigation window disagrees with canonical identity")
    return data


__all__ = [
    "BA_IEG_A0_CANONICAL_IDENTITY_BINDING_SCHEMA_V1",
    "BA_IEG_A0_EVALUATION_SEMANTICS_V1",
    "BA_IEG_A0_NAVIGATION_WINDOW_SCHEMA_V1",
    "BA_IEG_A0_SUPPORT_POLICY_ID_V1",
    "BAIEGA0NavigationSupportPolicyV1",
    "build_ba_ieg_a0_canonical_identity_binding_v1",
    "build_ba_ieg_a0_navigation_window_v1",
    "validate_ba_ieg_a0_canonical_identity_binding_v1",
    "validate_ba_ieg_a0_navigation_window_v1",
]
