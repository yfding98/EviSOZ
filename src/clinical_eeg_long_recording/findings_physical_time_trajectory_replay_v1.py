"""Exact-clock, multi-query stabilization authority for native Findings atoms.

The v1 native atom wire stores convenient floating-point seconds and a
snapshot-local ``query_transition_state``.  Neither is trusted here.  This
admission layer requires, for every query:

* actual byte replay of the complete producer-to-atom wire JSON;
* actual byte replay of an exact-sample lineage sidecar;
* exact proposal, atom, and wire-row inventories derived from those bytes;
* a content-addressed canonical clock with rational sample rate;
* integer half-open measurement, change, and raw-dependency spans; and
* a multi-query state-machine replay using a fitted, cross-fold threshold.

All containment decisions are integer comparisons.  In particular, a caller
cannot enlarge a tolerance by self-reporting a tiny sample rate.  A legacy
float-only atom without the exact sidecar is rejected before it can become
positive support.  Stabilization is never copied from an atom snapshot.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final

from .findings_native_measurement_atom_wire_adapter_v1 import (
    FINDINGS_NATIVE_MEASUREMENT_ATOM_WIRE_SCHEMA_VERSION,
)
from .findings_onset_threshold_registry_v1 import (
    FINDINGS_CANONICAL_EXACT_CLOCK_SCHEMA_VERSION,
    FINDINGS_ONSET_THRESHOLD_ADMITTED_REGISTRY_SCHEMA_VERSION,
    FINDINGS_ONSET_THRESHOLD_PREREGISTRY_SCHEMA_VERSION,
    ValidatedFindingsOnsetThresholdRegistry,
    _validate_exact_clock,
    _validate_stratum,
    validate_findings_onset_threshold_admitted_registry_v1,
    validate_findings_onset_threshold_preregistry_v1,
    require_validated_findings_onset_threshold_registry_v1,
)
from .detector_signal_lineage_authority_v1 import (
    ValidatedDetectorSignalLineageAuthority,
)


FINDINGS_EXACT_SAMPLE_LINEAGE_SIDECAR_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_findings_exact_sample_lineage_sidecar_v1"
)
FINDINGS_PHYSICAL_TIME_TRAJECTORY_REPLAY_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_findings_physical_time_trajectory_replay_v1"
)
FINDINGS_PHYSICAL_TIME_TRAJECTORY_METHOD_ID: Final[str] = (
    "EXACT-CLOCK-MULTI-QUERY-NATIVE-FINDINGS-STABILIZATION-V1"
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHA_RE = re.compile(r"^[a-f0-9]{64}$")
_QUERY_STATES = frozenset(
    {
        "not_evaluable",
        "first_observed",
        "updated_unstable",
        "stabilized",
        "changed_after_stabilization",
        "invalidated",
    }
)

_SOURCE_FIREWALL: Final[dict[str, bool]] = {
    "EEG_samples_used_upstream": True,
    "allowlisted_acquisition_metadata_used": True,
    "EEG_derived_QC_used": True,
    "electrical_reference_provenance_used_for_transform_only": True,
    "electrical_reference_values_used_as_model_features": False,
    "EEG_QC_values_used_as_model_features": False,
    "EDF_annotations_used": False,
    "spreadsheet_or_Excel_used": False,
    "doctor_labels_or_reports_used": False,
    "SOZ_or_channel_GT_used": False,
    "clinical_history_used": False,
    "video_or_behavior_used": False,
    "sleep_staging_used": False,
    "provocation_or_activation_used": False,
    "ECG_EMG_EOG_used": False,
    "LLM_used": False,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _sha(body)


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a canonical identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _fraction(value: object, context: str, *, positive: bool = False) -> Fraction:
    if (
        type(value) is not list
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
        or value[1] <= 0
    ):
        raise ValueError(f"{context} must be a reduced fraction")
    result = Fraction(value[0], value[1])
    if [result.numerator, result.denominator] != value:
        raise ValueError(f"{context} must be reduced")
    if (positive and result <= 0) or (not positive and result < 0):
        raise ValueError(f"{context} has invalid sign")
    return result


def _fraction_json(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _ceil(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def _floor(value: Fraction) -> int:
    return value.numerator // value.denominator


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _exact_fields(value: object, fields: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    row = deepcopy(dict(value))
    if set(row) != fields:
        raise ValueError(
            f"{context} fields drifted; missing={sorted(fields-set(row))}, "
            f"extra={sorted(set(row)-fields)}"
        )
    return row


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _safe_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise TypeError("artifact relative path must be text")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise PermissionError("artifact path must be safe and relative")
    resolved_root = root.resolve(strict=True)
    path = resolved_root.joinpath(*pure.parts).resolve(strict=True)
    if path != resolved_root and resolved_root not in path.parents:
        raise PermissionError("artifact path escaped its root")
    return path


def _read_json_binding(value: object, *, root: Path, context: str) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _exact_fields(
        value,
        {"relative_path", "file_bytes", "file_sha256", "content_sha256"},
        f"{context} binding",
    )
    path = _safe_path(root, binding["relative_path"])
    observed_hash, observed_size = _file_sha256(path)
    if observed_hash != _sha256(binding["file_sha256"], f"{context} file hash"):
        raise PermissionError(f"{context} file bytes changed")
    if observed_size != _integer(binding["file_bytes"], f"{context} file size", minimum=1):
        raise PermissionError(f"{context} byte count changed")
    try:
        payload = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not JSON") from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"{context} JSON must be an object")
    return deepcopy(dict(payload)), binding


def _verify_self_hash(row: Mapping[str, Any], field: str, context: str) -> str:
    observed = _sha256(row.get(field), f"{context} {field}")
    if observed != _self_hash(row, field):
        raise ValueError(f"{context} self hash does not replay")
    return observed


def _wire_inventory(wire: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "recording_id",
        "occurrence_id",
        "query_index",
        "source_proposal_receipts",
        "wire_rows",
        "measurement_atoms",
        "receipt_sha256",
    }
    if not required.issubset(wire):
        raise ValueError("wire artifact lacks its complete inventory fields")
    if wire["schema_version"] != FINDINGS_NATIVE_MEASUREMENT_ATOM_WIRE_SCHEMA_VERSION:
        raise ValueError("wire artifact schema drifted")
    _verify_self_hash(wire, "receipt_sha256", "wire artifact")
    proposals = []
    proposal_ids = set()
    for item in wire["source_proposal_receipts"]:
        if not isinstance(item, Mapping):
            raise TypeError("wire proposal must be an object")
        proposal_id = _identifier(item.get("proposal_id"), "wire proposal ID")
        if proposal_id in proposal_ids:
            raise ValueError("duplicate wire proposal")
        proposal_ids.add(proposal_id)
        receipt = _verify_self_hash(item, "receipt_sha256", "wire proposal")
        proposals.append({"proposal_id": proposal_id, "receipt_sha256": receipt})
    atoms = []
    atom_ids = set()
    for item in wire["measurement_atoms"]:
        if not isinstance(item, Mapping):
            raise TypeError("wire atom must be an object")
        atom_id = _identifier(item.get("measurement_atom_id"), "wire atom ID")
        if atom_id in atom_ids:
            raise ValueError("duplicate wire atom")
        atom_ids.add(atom_id)
        content = _verify_self_hash(item, "measurement_content_sha256", "wire atom")
        source_proposals = item.get("source_proposal_ids")
        if (
            not isinstance(source_proposals, Sequence)
            or isinstance(source_proposals, (str, bytes))
            or not source_proposals
            or not set(source_proposals).issubset(proposal_ids)
        ):
            raise ValueError("wire atom proposal lineage is incomplete")
        atoms.append(
            {
                "measurement_atom_id": atom_id,
                "measurement_content_sha256": content,
            }
        )
    wire_rows = []
    row_ids = set()
    atom_row_count: dict[str, int] = defaultdict(int)
    for item in wire["wire_rows"]:
        if not isinstance(item, Mapping):
            raise TypeError("wire row must be an object")
        wire_row_id = _identifier(item.get("wire_row_id"), "wire row ID")
        if wire_row_id in row_ids:
            raise ValueError("duplicate wire row")
        row_ids.add(wire_row_id)
        row_hash = _verify_self_hash(item, "wire_row_sha256", "wire row")
        atom_id = item.get("measurement_atom_id")
        if atom_id is not None:
            if atom_id not in atom_ids:
                raise ValueError("wire row references an absent atom")
            atom_row_count[atom_id] += 1
        wire_rows.append({"wire_row_id": wire_row_id, "wire_row_sha256": row_hash})
    if any(atom_row_count[atom_id] == 0 for atom_id in atom_ids):
        raise ValueError("wire atom is absent from the complete wire-row inventory")
    return {
        "source_proposals": sorted(proposals, key=lambda item: item["proposal_id"]),
        "measurement_atoms": sorted(atoms, key=lambda item: item["measurement_atom_id"]),
        "wire_rows": sorted(wire_rows, key=lambda item: item["wire_row_id"]),
        "inventory_sha256": _sha(
            {
                "source_proposals": sorted(proposals, key=lambda item: item["proposal_id"]),
                "measurement_atoms": sorted(atoms, key=lambda item: item["measurement_atom_id"]),
                "wire_rows": sorted(wire_rows, key=lambda item: item["wire_row_id"]),
            }
        ),
    }


def _validate_index_span(value: object, context: str) -> list[int]:
    if type(value) is not list or len(value) != 2:
        raise ValueError(f"{context} must be an integer half-open pair")
    start = _integer(value[0], f"{context} start")
    stop = _integer(value[1], f"{context} stop", minimum=1)
    if stop <= start:
        raise ValueError(f"{context} must be non-empty")
    return [start, stop]


def _validate_lineage_row(
    value: object,
    *,
    atom: Mapping[str, Any],
    clock: Mapping[str, Any],
    query_prefix: Sequence[int],
    stratum_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    fields = {
        "measurement_atom_id",
        "measurement_content_sha256",
        "native_track_id",
        "native_track_receipt_sha256",
        "producer_track_ordinal",
        "producer_track_semantics_receipt_sha256",
        "operator_stratum_id",
        "measurement_half_open_sample_span",
        "change_half_open_sample_span",
        "raw_dependency_half_open_sample_spans",
        "raw_dependency_sha256s",
        "canonical_exact_clock_receipt_sha256",
        "exact_sample_projection_receipt_sha256",
        "producer_view_sample_rate_hz_fraction",
        "producer_view_measurement_half_open_sample_span",
        "producer_view_to_canonical_source_projection_receipt_sha256",
        "row_receipt_sha256",
    }
    row = _exact_fields(value, fields, "exact atom lineage row")
    if row["measurement_atom_id"] != atom["measurement_atom_id"]:
        raise ValueError("exact lineage atom ID crossed")
    if row["measurement_content_sha256"] != atom["measurement_content_sha256"]:
        raise PermissionError("exact lineage atom content binding changed")
    row["native_track_id"] = _identifier(row["native_track_id"], "native track ID")
    native_track_receipt = _sha256(
        row["native_track_receipt_sha256"], "native track receipt"
    )
    # The stable-track authority must at least be content-bound to the native
    # producer that emitted this atom.  An unrelated free hash is not enough.
    if native_track_receipt != atom.get("producer_receipt_sha256"):
        raise PermissionError("native track receipt is not the atom producer receipt")
    stratum_id = _identifier(row["operator_stratum_id"], "operator stratum ID")
    if stratum_id not in stratum_by_id:
        raise PermissionError("exact lineage opened an unregistered operator stratum")
    stratum = stratum_by_id[stratum_id]
    if (
        atom.get("operator_id") != stratum["operator_id"]
        or atom.get("operator_version") != stratum["operator_version"]
        or atom.get("reference_family") != stratum["reference_family"]
        or atom.get("physical_unit") != stratum["physical_unit"]
        or atom.get("effect_size_and_unit", {}).get("measurement_name_id")
        != stratum["measurement_name_id"]
        or atom.get("operator_parameter_receipt_sha256")
        != stratum["operator_parameter_receipt_sha256"]
    ):
        raise PermissionError("atom semantics differ from its signed operator/scale/reference stratum")
    producer_track_ordinal = _integer(
        row["producer_track_ordinal"], "producer track ordinal"
    )
    typed_unit = atom.get("typed_unit")
    if (
        not isinstance(typed_unit, Mapping)
        or set(typed_unit) != {"unit_type", "unit_id", "unit_key"}
        or typed_unit.get("unit_type") not in {"electrode", "lead"}
        or typed_unit.get("unit_key")
        != f"{typed_unit.get('unit_type')}:{typed_unit.get('unit_id')}"
    ):
        raise ValueError("atom lacks a canonical typed-unit spatial identity")
    for field in ("unit_id", "unit_key"):
        _identifier(typed_unit[field], f"atom typed unit {field}")
    canonical_source_channels = atom.get("canonical_source_channels")
    if (
        not isinstance(canonical_source_channels, Sequence)
        or isinstance(canonical_source_channels, (str, bytes))
        or not canonical_source_channels
        or len(canonical_source_channels) != len(set(canonical_source_channels))
    ):
        raise ValueError("atom lacks canonical source-channel spatial identity")
    for channel in canonical_source_channels:
        _identifier(channel, "atom canonical source channel")
    if typed_unit["unit_type"] == "electrode" and list(canonical_source_channels) != [
        typed_unit["unit_id"]
    ]:
        raise PermissionError("electrode track changed its direct source channel")
    if typed_unit["unit_type"] == "lead" and (
        typed_unit["unit_id"].split("-") != list(canonical_source_channels)
        or len(canonical_source_channels) != 2
    ):
        raise PermissionError("directed lead track changed endpoint identity")
    for field in (
        "recording_id",
        "occurrence_id",
        "source_slot_id",
        "measurement_domain",
        "namespace",
    ):
        _identifier(atom.get(field), f"atom stable track {field}")
    if atom.get("whole_bipolar_lead_identity_preserved") is not True:
        raise PermissionError("atom does not preserve whole-bipolar lead identity")
    if atom.get("bipolar_endpoint_attribution_authorized") is not False:
        raise PermissionError("atom opened bipolar endpoint attribution")
    track_semantics = {
        "domain": "native_findings_stable_typed_unit_operator_track_v1",
        "recording_id": atom.get("recording_id"),
        "occurrence_id": atom.get("occurrence_id"),
        "source_slot_id": atom.get("source_slot_id"),
        "measurement_domain": atom.get("measurement_domain"),
        "namespace": atom.get("namespace"),
        "typed_unit": deepcopy(dict(typed_unit)),
        "canonical_source_channels": list(canonical_source_channels),
        "spatial_region_semantics": (
            "scalp_typed_unit_and_directed_source_channels_not_anatomical_region"
        ),
        "reference_family": atom.get("reference_family"),
        "operator_id": atom.get("operator_id"),
        "operator_version": atom.get("operator_version"),
        "operator_parameter_receipt_sha256": atom.get(
            "operator_parameter_receipt_sha256"
        ),
        "measurement_name_id": atom.get("effect_size_and_unit", {}).get(
            "measurement_name_id"
        ),
        "physical_unit": atom.get("physical_unit"),
        "effect_sign_policy": stratum["effect_sign_policy"],
        "scale_id": stratum["scale_id"],
        "whole_bipolar_lead_identity_preserved": atom.get(
            "whole_bipolar_lead_identity_preserved"
        ),
        "bipolar_endpoint_attribution_authorized": atom.get(
            "bipolar_endpoint_attribution_authorized"
        ),
        "producer_track_ordinal": producer_track_ordinal,
    }
    track_semantics_receipt = _sha256(
        row["producer_track_semantics_receipt_sha256"],
        "producer track semantics receipt",
    )
    if track_semantics_receipt != _sha(track_semantics):
        raise PermissionError("producer track typed-unit/operator semantics do not replay")
    expected_native_track_id = "NATIVE-TRACK-" + track_semantics_receipt[:24]
    if row["native_track_id"] != expected_native_track_id:
        raise PermissionError(
            "native track ID is not bound to stable typed-unit/operator semantics"
        )
    measurement = _validate_index_span(
        row["measurement_half_open_sample_span"], "measurement sample span"
    )
    change = _validate_index_span(row["change_half_open_sample_span"], "change sample span")
    prefix = _validate_index_span(list(query_prefix), "locked causal prefix")
    if not prefix[0] <= measurement[0] < measurement[1] <= prefix[1]:
        raise PermissionError("measurement sample span leaves locked causal prefix")
    if not measurement[0] <= change[0] < change[1] <= measurement[1]:
        raise PermissionError("change sample span leaves measurement span")
    spans_value = row["raw_dependency_half_open_sample_spans"]
    if not isinstance(spans_value, Sequence) or isinstance(spans_value, (str, bytes)) or not spans_value:
        raise ValueError("raw dependency sample spans must be non-empty")
    spans = [_validate_index_span(item, "raw dependency sample span") for item in spans_value]
    if spans != sorted(spans) or any(
        spans[index][0] < spans[index - 1][1] for index in range(1, len(spans))
    ):
        raise ValueError("raw dependency spans must be sorted and non-overlapping")
    if any(not prefix[0] <= span[0] < span[1] <= prefix[1] for span in spans):
        raise PermissionError("raw dependency sample span leaves locked causal prefix")
    if not any(span[0] <= change[0] and change[1] <= span[1] for span in spans):
        raise PermissionError("raw dependency inventory does not contain the change span")
    raw_hashes = row["raw_dependency_sha256s"]
    if not isinstance(raw_hashes, Sequence) or isinstance(raw_hashes, (str, bytes)):
        raise TypeError("raw dependency hashes must be an array")
    normalized_hashes = [_sha256(item, "raw dependency hash") for item in raw_hashes]
    if not normalized_hashes or normalized_hashes != sorted(set(normalized_hashes)):
        raise ValueError("raw dependency hashes must be non-empty, unique, sorted")
    if normalized_hashes != atom.get("raw_dependency_sha256s"):
        raise PermissionError("exact lineage raw dependency content differs from atom")
    if row["canonical_exact_clock_receipt_sha256"] != clock["receipt_sha256"]:
        raise PermissionError("exact lineage self-reported a different clock")
    producer_view_fs = _fraction(
        row["producer_view_sample_rate_hz_fraction"],
        "producer view sample rate",
        positive=True,
    )
    producer_view_span = _validate_index_span(
        row["producer_view_measurement_half_open_sample_span"],
        "producer view measurement sample span",
    )
    projection_material = {
        "domain": "producer_view_to_canonical_source_sample_projection_v1",
        "producer_receipt_sha256": atom.get("producer_receipt_sha256"),
        "canonical_exact_clock_receipt_sha256": clock["receipt_sha256"],
        "producer_view_sample_rate_hz_fraction": _fraction_json(producer_view_fs),
        "producer_view_measurement_half_open_sample_span": producer_view_span,
        "canonical_source_measurement_half_open_sample_span": measurement,
        "canonical_source_change_half_open_sample_span": change,
        "canonical_source_raw_dependency_half_open_sample_spans": spans,
    }
    projection_receipt = _sha256(
        row["producer_view_to_canonical_source_projection_receipt_sha256"],
        "producer-view to canonical-source projection receipt",
    )
    if projection_receipt != _sha(projection_material):
        raise PermissionError(
            "producer-view to canonical-source sample projection does not replay"
        )
    exact_projection_material = {
        "domain": "exact_atom_sample_projection_v1",
        "measurement_atom_content_sha256": atom["measurement_content_sha256"],
        "producer_view_projection_receipt_sha256": projection_receipt,
        "raw_dependency_sha256s": normalized_hashes,
    }
    exact_projection_receipt = _sha256(
        row["exact_sample_projection_receipt_sha256"],
        "exact sample projection receipt",
    )
    if exact_projection_receipt != _sha(exact_projection_material):
        raise PermissionError("exact atom sample projection does not replay")
    fs = _fraction(clock["sample_rate_hz_fraction"], "sample rate", positive=True)
    nyquist = fs / 2
    effective_high = _fraction(
        stratum["effective_bandwidth_hz_fraction"][1], "effective bandwidth high", positive=True
    )
    if effective_high > nyquist:
        raise PermissionError("effective bandwidth exceeds canonical clock Nyquist")
    atom_effective = atom.get("effective_bandwidth_hz")
    atom_required = atom.get("required_bandwidth_hz")
    if (
        not isinstance(atom_effective, Sequence)
        or isinstance(atom_effective, (str, bytes))
        or len(atom_effective) != 2
        or not isinstance(atom_required, Sequence)
        or isinstance(atom_required, (str, bytes))
        or len(atom_required) != 2
    ):
        raise ValueError("atom bandwidth fields are absent")
    atom_effective_fraction = [
        Fraction(str(_finite(item, "atom effective bandwidth")))
        for item in atom_effective
    ]
    atom_required_fraction = [
        Fraction(str(_finite(item, "atom required bandwidth")))
        for item in atom_required
    ]
    if (
        atom_effective_fraction[0] < 0
        or atom_effective_fraction[1] <= atom_effective_fraction[0]
        or atom_required_fraction[0] < atom_effective_fraction[0]
        or atom_required_fraction[1] > atom_effective_fraction[1]
        or atom_effective_fraction[1] > nyquist
    ):
        raise PermissionError("atom bandwidth fails exact-clock Nyquist/containment")
    stratum_effective = [
        _fraction(item, "stratum effective bandwidth")
        for item in stratum["effective_bandwidth_hz_fraction"]
    ]
    stratum_required = [
        _fraction(item, "stratum required bandwidth")
        for item in stratum["required_bandwidth_hz_fraction"]
    ]
    if atom_effective_fraction != stratum_effective or atom_required_fraction != stratum_required:
        raise PermissionError("atom bandwidth differs from its registered operator stratum")

    # Float seconds are non-authoritative, but they must agree with the exact
    # sidecar projection.  This blocks a sidecar from hiding a future float
    # dependency inside an earlier integer prefix while still making all gate
    # decisions from the integer clock.
    def projected_span(seconds: object, label: str) -> list[int]:
        if (
            not isinstance(seconds, Sequence)
            or isinstance(seconds, (str, bytes))
            or len(seconds) != 2
        ):
            raise ValueError(f"atom {label} seconds interval is absent")
        start_seconds = Fraction(str(_finite(seconds[0], f"atom {label} start")))
        stop_seconds = Fraction(str(_finite(seconds[1], f"atom {label} stop")))
        if stop_seconds <= start_seconds:
            raise ValueError(f"atom {label} seconds interval is empty")
        return [_floor(start_seconds * fs), _ceil(stop_seconds * fs)]

    if projected_span(atom.get("recording_relative_half_open_interval_s"), "measurement") != measurement:
        raise PermissionError("atom measurement seconds do not replay to exact sample indices")
    if projected_span(atom.get("change_interval_s"), "change") != change:
        raise PermissionError("atom change seconds do not replay to exact sample indices")
    atom_raw_seconds = atom.get("raw_dependency_interval_union_s")
    if not isinstance(atom_raw_seconds, Sequence) or isinstance(atom_raw_seconds, (str, bytes)):
        raise ValueError("atom raw dependency seconds inventory is absent")
    projected_raw = [projected_span(item, "raw dependency") for item in atom_raw_seconds]
    if projected_raw != spans:
        raise PermissionError("atom raw dependency seconds do not replay to exact sample indices")
    row["measurement_half_open_sample_span"] = measurement
    row["change_half_open_sample_span"] = change
    row["raw_dependency_half_open_sample_spans"] = spans
    row["raw_dependency_sha256s"] = normalized_hashes
    _verify_self_hash(row, "row_receipt_sha256", "exact atom lineage row")
    return row


def _validate_sidecar(
    value: object,
    *,
    wire: Mapping[str, Any],
    wire_binding: Mapping[str, Any],
    inventory: Mapping[str, Any],
    canonical_clock: Mapping[str, Any],
    stratum_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "recording_id",
        "occurrence_id",
        "query_index",
        "query_stop_sample_index",
        "locked_causal_prefix_half_open_sample_span",
        "canonical_exact_clock_receipt_sha256",
        "detector_signal_lineage_authority_receipt_sha256",
        "wire_file_sha256",
        "wire_receipt_sha256",
        "source_proposal_inventory",
        "measurement_atom_inventory",
        "wire_row_inventory",
        "wire_inventory_sha256",
        "exact_atom_sample_lineage_rows",
        "source_firewall",
        "content_sha256",
    }
    row = _exact_fields(value, fields, "exact sample lineage sidecar")
    if row["schema_version"] != FINDINGS_EXACT_SAMPLE_LINEAGE_SIDECAR_SCHEMA_VERSION:
        raise ValueError("exact sample lineage sidecar schema drifted")
    for field in ("recording_id", "occurrence_id"):
        if row[field] != wire[field]:
            raise ValueError(f"sidecar {field} crossed wire artifact")
    if row["query_index"] != wire["query_index"]:
        raise ValueError("sidecar query index crossed wire artifact")
    query_stop = _integer(row["query_stop_sample_index"], "query stop", minimum=1)
    prefix = _validate_index_span(
        row["locked_causal_prefix_half_open_sample_span"], "locked causal prefix"
    )
    if prefix[1] != query_stop or query_stop > canonical_clock["total_sample_count"]:
        raise PermissionError("query stop and locked causal-prefix stop differ")
    if row["canonical_exact_clock_receipt_sha256"] != canonical_clock["receipt_sha256"]:
        raise PermissionError("sidecar crossed canonical exact clock")
    if row["detector_signal_lineage_authority_receipt_sha256"] != canonical_clock[
        "detector_signal_lineage_authority_receipt_sha256"
    ]:
        raise PermissionError("sidecar crossed opaque signal-lineage authority")
    if row["wire_file_sha256"] != wire_binding["file_sha256"]:
        raise PermissionError("sidecar wire-byte binding changed")
    if row["wire_receipt_sha256"] != wire["receipt_sha256"]:
        raise PermissionError("sidecar wire receipt binding changed")
    for field, expected in (
        ("source_proposal_inventory", inventory["source_proposals"]),
        ("measurement_atom_inventory", inventory["measurement_atoms"]),
        ("wire_row_inventory", inventory["wire_rows"]),
    ):
        if row[field] != expected:
            raise PermissionError(f"sidecar {field} is not the complete byte-derived inventory")
    if row["wire_inventory_sha256"] != inventory["inventory_sha256"]:
        raise PermissionError("sidecar wire inventory digest changed")
    if not isinstance(row["source_firewall"], Mapping) or dict(row["source_firewall"]) != _SOURCE_FIREWALL:
        raise PermissionError("exact sidecar source firewall drifted")
    atoms_by_id = {item["measurement_atom_id"]: item for item in wire["measurement_atoms"]}
    lineages = row["exact_atom_sample_lineage_rows"]
    if not isinstance(lineages, Sequence) or isinstance(lineages, (str, bytes)):
        raise TypeError("exact atom lineage rows must be an array")
    normalized = []
    seen = set()
    for lineage_value in lineages:
        if not isinstance(lineage_value, Mapping):
            raise TypeError("exact atom lineage row must be an object")
        atom_id = lineage_value.get("measurement_atom_id")
        if atom_id in seen or atom_id not in atoms_by_id:
            raise ValueError("exact atom lineage inventory has duplicate or unknown atom")
        seen.add(atom_id)
        normalized.append(
            _validate_lineage_row(
                lineage_value,
                atom=atoms_by_id[atom_id],
                clock=canonical_clock,
                query_prefix=prefix,
                stratum_by_id=stratum_by_id,
            )
        )
    # This is an exact set equality against the IDs derived from artifact
    # bytes, not a tautological comparison against the input row count.
    if seen != set(atoms_by_id):
        raise PermissionError("exact sample lineage omitted a byte-derived atom")
    row["exact_atom_sample_lineage_rows"] = sorted(
        normalized, key=lambda item: item["measurement_atom_id"]
    )
    _verify_self_hash(row, "content_sha256", "exact sample lineage sidecar")
    return row


def _selected_gates(
    *,
    admitted_registry: Mapping[str, Any],
    outer_fold_id: int,
) -> dict[str, Mapping[str, Any]]:
    fold_rows = [
        item for item in admitted_registry["fold_receipts"] if item["outer_fold_id"] == outer_fold_id
    ]
    if len(fold_rows) != 1:
        raise PermissionError("trajectory replay lacks exactly one selected outer-fold gate")
    return {
        item["stratum_id"]: item for item in fold_rows[0]["selected_operator_strata"]
    }


def _effect_score(value: float, sign: str) -> float:
    if sign == "increase":
        return value
    if sign == "decrease":
        return -value
    return abs(value)


def _trajectory_key(lineage: Mapping[str, Any]) -> str:
    # The semantics receipt is deterministically rebuilt from the atom's
    # typed unit, directed source channels, operator/measurement identity,
    # scale/reference and producer track ordinal.  A caller cannot preserve a
    # trajectory merely by reusing a free-form native_track_id.
    return "TRACK-" + str(lineage["producer_track_semantics_receipt_sha256"])[:24]


def replay_findings_physical_time_trajectory_v1(
    *,
    preregistry: Mapping[str, Any],
    threshold_registry: Mapping[str, Any] | ValidatedFindingsOnsetThresholdRegistry,
    outer_fold_id: int | None,
    canonical_exact_clock: Mapping[str, Any],
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
    recording_duration_seconds_fraction: Sequence[int],
    recording_id: str,
    occurrence_id: str,
    query_artifacts: Sequence[Mapping[str, Any]],
    artifact_root: Path,
) -> dict[str, Any]:
    """Replay exact physical-time trajectories across complete query snapshots."""

    protocol = validate_findings_onset_threshold_preregistry_v1(preregistry)
    recording_id = _identifier(recording_id, "recording_id")
    occurrence_id = _identifier(occurrence_id, "occurrence_id")
    duration = _fraction(
        list(recording_duration_seconds_fraction), "recording duration", positive=True
    )
    clock = _validate_exact_clock(
        canonical_exact_clock,
        duration=duration,
        recording_id=recording_id,
        signal_lineage_authority=signal_lineage_authority,
    )
    if (
        isinstance(threshold_registry, Mapping)
        and threshold_registry.get("schema_version")
        == FINDINGS_ONSET_THRESHOLD_PREREGISTRY_SCHEMA_VERSION
    ):
        unadmitted = validate_findings_onset_threshold_preregistry_v1(threshold_registry)
        if unadmitted["registry_receipt_sha256"] != protocol["registry_receipt_sha256"]:
            raise PermissionError("trajectory threshold preregistry crossed")
        registry_admitted = False
        selected_gates: dict[str, Mapping[str, Any]] = {}
        strata: list[dict[str, Any]] = []
        threshold_registry_receipt = unadmitted["registry_receipt_sha256"]
    elif isinstance(threshold_registry, ValidatedFindingsOnsetThresholdRegistry):
        admitted = require_validated_findings_onset_threshold_registry_v1(
            threshold_registry, preregistry=protocol
        )
        if outer_fold_id is None:
            raise PermissionError("admitted trajectory replay requires a frozen outer-fold gate")
        outer_fold_id = _integer(outer_fold_id, "outer fold ID")
        registry_admitted = True
        selected_gates = _selected_gates(
            admitted_registry=admitted, outer_fold_id=outer_fold_id
        )
        strata = admitted["operator_strata"]
        threshold_registry_receipt = admitted["registry_receipt_sha256"]
    elif (
        isinstance(threshold_registry, Mapping)
        and threshold_registry.get("schema_version")
        == FINDINGS_ONSET_THRESHOLD_ADMITTED_REGISTRY_SCHEMA_VERSION
    ):
        raise TypeError(
            "raw admitted threshold mappings are not authority; replay the real "
            "materializer and pass its opaque result"
        )
    else:
        raise ValueError("trajectory threshold registry schema is unsupported")

    if not isinstance(query_artifacts, Sequence) or isinstance(query_artifacts, (str, bytes)):
        raise TypeError("query artifacts must be an array")
    if not query_artifacts:
        raise ValueError("multi-query trajectory replay requires at least one query artifact")
    query_rows = []
    observed_strata: dict[str, dict[str, Any]] = {item["stratum_id"]: item for item in strata}
    for query_value in query_artifacts:
        query = _exact_fields(
            query_value,
            {"wire_artifact", "exact_sample_lineage_sidecar"},
            "trajectory query artifact",
        )
        wire, wire_binding = _read_json_binding(
            query["wire_artifact"], root=artifact_root, context="wire artifact"
        )
        if wire_binding["content_sha256"] != wire.get("receipt_sha256"):
            raise PermissionError("wire artifact binding semantic receipt changed")
        inventory = _wire_inventory(wire)
        if wire["recording_id"] != recording_id or wire["occurrence_id"] != occurrence_id:
            raise ValueError("wire artifact crossed trajectory recording/occurrence")
        sidecar, sidecar_binding = _read_json_binding(
            query["exact_sample_lineage_sidecar"],
            root=artifact_root,
            context="exact sample lineage sidecar",
        )
        if sidecar_binding["content_sha256"] != sidecar.get("content_sha256"):
            raise PermissionError("exact sidecar binding semantic content changed")
        # An unadmitted registry has no fitted stratum roster.  The sidecar's
        # own exact strata are intentionally not accepted as authority; thus
        # legacy/current atoms remain structurally unqualified and positive
        # output is impossible.  For useful structural replay callers may
        # provide admitted strata only.
        if not registry_admitted:
            if sidecar.get("exact_atom_sample_lineage_rows"):
                raise PermissionError(
                    "float-only/current atoms cannot enter trajectory replay before a real admitted stratum registry"
                )
            stratum_by_id: dict[str, Mapping[str, Any]] = {}
        else:
            stratum_by_id = observed_strata
        validated_sidecar = _validate_sidecar(
            sidecar,
            wire=wire,
            wire_binding=wire_binding,
            inventory=inventory,
            canonical_clock=clock,
            stratum_by_id=stratum_by_id,
        )
        query_rows.append(
            {
                "query_index": wire["query_index"],
                "query_stop_sample_index": validated_sidecar["query_stop_sample_index"],
                "wire": wire,
                "wire_binding": wire_binding,
                "inventory": inventory,
                "sidecar": validated_sidecar,
                "sidecar_binding": sidecar_binding,
            }
        )
    query_rows.sort(key=lambda item: item["query_index"])
    indices = [item["query_index"] for item in query_rows]
    if indices != sorted(set(indices)):
        raise ValueError("trajectory query indices must be unique and sorted")
    stops = [item["query_stop_sample_index"] for item in query_rows]
    if stops != sorted(stops) or len(stops) != len(set(stops)):
        raise ValueError("trajectory physical query stops must increase strictly")

    fs = _fraction(clock["sample_rate_hz_fraction"], "sample rate", positive=True)
    revision_tolerance_samples = _ceil(
        _fraction(
            protocol["trajectory_protocol"][
                "change_interval_revision_tolerance_seconds_fraction"
            ],
            "change revision tolerance",
        )
        * fs
    )
    histories: dict[str, dict[str, Any]] = {}
    audit_rows = []
    for query in query_rows:
        wire_atoms = {
            item["measurement_atom_id"]: item for item in query["wire"]["measurement_atoms"]
        }
        current_tracks = {}
        for lineage in query["sidecar"]["exact_atom_sample_lineage_rows"]:
            atom = wire_atoms[lineage["measurement_atom_id"]]
            track_key = _trajectory_key(lineage)
            if track_key in current_tracks:
                raise ValueError("one query contains duplicate native trajectory keys")
            current_tracks[track_key] = (lineage, atom)
        # Complete query inventory makes both pre- and post-stabilization
        # disappearance observable, including zero-atom producer snapshots.
        # A missing pre-stable track must break the consecutive persistence
        # run; otherwise two non-contiguous sightings could falsely stabilize.
        for track_key, history in histories.items():
            if history["latched_changed"] or track_key in current_tracks:
                continue
            if history["stabilized"]:
                history["latched_changed"] = True
                audit_rows.append(
                    {
                        "query_index": query["query_index"],
                        "query_stop_sample_index": query["query_stop_sample_index"],
                        "trajectory_key": track_key,
                        "measurement_atom_id": None,
                        "operator_stratum_id": history["operator_stratum_id"],
                        "atom_self_reported_query_transition_state": None,
                        "computed_query_transition_state": "changed_after_stabilization",
                        "effect_threshold_state": "not_evaluable",
                        "minimum_persistence_state": "not_evaluable",
                        "positive_onset_support_eligible": False,
                        "reason_codes": ["atom_absent_after_stabilization_in_complete_inventory"],
                    }
                )
            elif history["run"]:
                history["run"] = []
                history["last_query_stop"] = query["query_stop_sample_index"]
                audit_rows.append(
                    {
                        "query_index": query["query_index"],
                        "query_stop_sample_index": query[
                            "query_stop_sample_index"
                        ],
                        "trajectory_key": track_key,
                        "measurement_atom_id": None,
                        "operator_stratum_id": history["operator_stratum_id"],
                        "atom_self_reported_query_transition_state": None,
                        "computed_query_transition_state": "invalidated",
                        "effect_threshold_state": "not_evaluable",
                        "minimum_persistence_state": "continuity_reset",
                        "positive_onset_support_eligible": False,
                        "reason_codes": [
                            "atom_absent_before_stabilization_continuity_reset"
                        ],
                    }
                )
        for track_key, (lineage, atom) in sorted(current_tracks.items()):
            stratum_id = lineage["operator_stratum_id"]
            gate = selected_gates[stratum_id]
            sign = gate["effect_sign_policy"]
            effect = _finite(atom["effect_size_and_unit"]["value"], "atom effect")
            threshold = _finite(gate["effect_threshold_value"], "selected effect threshold")
            passes = _effect_score(effect, sign) >= threshold
            persistence_required = _fraction(
                gate["minimum_persistence_seconds_fraction"], "selected persistence", positive=True
            )
            maximum_gap = _fraction(
                gate["maximum_query_gap_seconds_fraction"], "selected query gap", positive=True
            )
            minimum_queries = _integer(
                gate["minimum_stable_query_count"], "selected minimum stable queries", minimum=2
            )
            history = histories.setdefault(
                track_key,
                {
                    "operator_stratum_id": stratum_id,
                    "stabilized": False,
                    "latched_changed": False,
                    "run": [],
                    "stable_change_span": None,
                    "last_query_stop": None,
                },
            )
            reasons = []
            if history["latched_changed"]:
                state = "changed_after_stabilization"
                reasons.append("changed_after_stabilization_is_latched")
            else:
                gap_exceeded = (
                    history["last_query_stop"] is not None
                    and Fraction(
                        query["query_stop_sample_index"] - history["last_query_stop"], 1
                    )
                    > maximum_gap * fs
                )
                signature_changed = False
                if history["stable_change_span"] is not None:
                    previous = history["stable_change_span"]
                    current = lineage["change_half_open_sample_span"]
                    signature_changed = (
                        abs(current[0] - previous[0]) > revision_tolerance_samples
                        or abs(current[1] - previous[1]) > revision_tolerance_samples
                    )
                if history["stabilized"] and (gap_exceeded or not passes or signature_changed):
                    history["latched_changed"] = True
                    state = "changed_after_stabilization"
                    if gap_exceeded:
                        reasons.append("query_gap_exceeded_after_stabilization")
                    if not passes:
                        reasons.append("threshold_failed_after_stabilization")
                    if signature_changed:
                        reasons.append("change_interval_revised_after_stabilization")
                elif history["stabilized"]:
                    state = "stabilized"
                elif not passes:
                    history["run"] = []
                    state = "invalidated"
                    reasons.append("effect_threshold_not_passed")
                else:
                    if gap_exceeded:
                        history["run"] = []
                        reasons.append("pre_stabilization_query_gap_started_new_run")
                    history["run"].append(
                        {
                            "query_stop": query["query_stop_sample_index"],
                            "measurement_span": lineage["measurement_half_open_sample_span"],
                        }
                    )
                    elapsed_samples = (
                        history["run"][-1]["measurement_span"][1]
                        - history["run"][0]["measurement_span"][0]
                    )
                    enough_time = Fraction(elapsed_samples, 1) >= persistence_required * fs
                    enough_queries = len(history["run"]) >= minimum_queries
                    if enough_time and enough_queries:
                        history["stabilized"] = True
                        history["stable_change_span"] = deepcopy(
                            lineage["change_half_open_sample_span"]
                        )
                        state = "stabilized"
                    elif len(history["run"]) == 1:
                        state = "first_observed"
                    else:
                        state = "updated_unstable"
            history["last_query_stop"] = query["query_stop_sample_index"]
            positive = state == "stabilized" and not history["latched_changed"]
            audit_rows.append(
                {
                    "query_index": query["query_index"],
                    "query_stop_sample_index": query["query_stop_sample_index"],
                    "trajectory_key": track_key,
                    "measurement_atom_id": atom["measurement_atom_id"],
                    "operator_stratum_id": stratum_id,
                    "atom_self_reported_query_transition_state": atom.get(
                        "query_transition_state"
                    ),
                    "computed_query_transition_state": state,
                    "effect_threshold_state": "pass" if passes else "not_passed",
                    "minimum_persistence_state": (
                        "pass" if state == "stabilized" else "not_yet_passed"
                    ),
                    "positive_onset_support_eligible": positive,
                    "reason_codes": reasons,
                }
            )

    if any(row["computed_query_transition_state"] not in _QUERY_STATES for row in audit_rows):
        raise AssertionError("unregistered trajectory state emitted")
    final_states = []
    for track_key, history in sorted(histories.items()):
        state = (
            "changed_after_stabilization"
            if history["latched_changed"]
            else "stabilized"
            if history["stabilized"]
            else "updated_unstable"
            if history["run"]
            else "invalidated"
        )
        final_states.append(
            {
                "trajectory_key": track_key,
                "operator_stratum_id": history["operator_stratum_id"],
                "final_state": state,
                "positive_onset_support_eligible": state == "stabilized",
                "positive_rank_contribution_authorized": False,
            }
        )
    body: dict[str, Any] = {
        "schema_version": FINDINGS_PHYSICAL_TIME_TRAJECTORY_REPLAY_SCHEMA_VERSION,
        "method_id": FINDINGS_PHYSICAL_TIME_TRAJECTORY_METHOD_ID,
        "recording_id": recording_id,
        "occurrence_id": occurrence_id,
        "outer_fold_id": outer_fold_id,
        "canonical_exact_clock_receipt_sha256": clock["receipt_sha256"],
        "preregistry_receipt_sha256": protocol["registry_receipt_sha256"],
        "threshold_registry_receipt_sha256": threshold_registry_receipt,
        "registry_admitted": registry_admitted,
        "query_artifact_bindings": [
            {
                "query_index": item["query_index"],
                "query_stop_sample_index": item["query_stop_sample_index"],
                "wire_artifact": deepcopy(item["wire_binding"]),
                "wire_inventory_sha256": item["inventory"]["inventory_sha256"],
                "exact_sample_lineage_sidecar": deepcopy(item["sidecar_binding"]),
            }
            for item in query_rows
        ],
        "trajectory_audit_rows": audit_rows,
        "final_trajectory_states": final_states,
        "denominators": {
            "query_artifact_count": len(query_rows),
            "byte_derived_source_proposal_count": sum(
                len(item["inventory"]["source_proposals"]) for item in query_rows
            ),
            "byte_derived_measurement_atom_count": sum(
                len(item["inventory"]["measurement_atoms"]) for item in query_rows
            ),
            "byte_derived_wire_row_count": sum(
                len(item["inventory"]["wire_rows"]) for item in query_rows
            ),
            "trajectory_count": len(histories),
            "stabilized_trajectory_count": sum(
                item["final_state"] == "stabilized" for item in final_states
            ),
            "changed_after_stabilization_count": sum(
                item["final_state"] == "changed_after_stabilization"
                for item in final_states
            ),
            "pre_stabilization_continuity_reset_count": sum(
                "atom_absent_before_stabilization_continuity_reset"
                in item["reason_codes"]
                for item in audit_rows
            ),
        },
        "source_firewall": deepcopy(_SOURCE_FIREWALL),
        "authorization": {
            "positive_onset_support_authorized_for_final_stabilized_rows": (
                registry_admitted
            ),
            "positive_rank_contribution_authorized": False,
            "clinical_term_authorized": False,
            "SOZ_EZ_or_surgical_target_claim_authorized": False,
            "report_text_authorized": False,
            "atom_self_reported_sample_rate_or_stability_trusted": False,
        },
        "receipt_sha256": "",
    }
    body["receipt_sha256"] = _self_hash(body, "receipt_sha256")
    return body


__all__ = [
    "FINDINGS_EXACT_SAMPLE_LINEAGE_SIDECAR_SCHEMA_VERSION",
    "FINDINGS_PHYSICAL_TIME_TRAJECTORY_METHOD_ID",
    "FINDINGS_PHYSICAL_TIME_TRAJECTORY_REPLAY_SCHEMA_VERSION",
    "replay_findings_physical_time_trajectory_v1",
]
