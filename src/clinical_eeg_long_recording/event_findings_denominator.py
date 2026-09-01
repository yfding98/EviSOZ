"""Independent, host-trusted opportunity denominator for event Findings.

The existing :mod:`event_findings_atom_roster` is a useful source-accounting
shadow, but it reads the same ``event_eeg_findings_v3`` payload whose output it
accounts for.  It therefore cannot prove that a producer did not omit a term,
unit, interval, or evaluation opportunity.  This module materializes that
denominator *before* a Findings producer runs, using only host-supplied EEG
scope, canonical-unit, quality/capability, and technical-failure receipts.

Version 1 deliberately closes the registered 28 core atoms and 12 child
rosters.  It is an item-scope structural denominator, not yet the stronger
term-query denominator needed to qualify clinical surface terms.  In
particular, ``absence_authorized`` in a cell means only that the corresponding
roster four-state slot may be recorded as ``absent_with_opportunity``.  It
never authorizes a clinical negative, cortical SOZ/EZ language, or report
promotion by itself.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .event_findings_atom_roster import (
    load_event_findings_atom_roster_policy,
    validate_event_findings_atom_roster_receipt,
)


EVENT_FINDINGS_DENOMINATOR_POLICY_SCHEMA_VERSION = (
    "clinical_eeg_event_findings_denominator_policy_v1"
)
EVENT_FINDINGS_DENOMINATOR_RECEIPT_SCHEMA_VERSION = (
    "clinical_eeg_event_findings_denominator_receipt_v1"
)
EVENT_FINDINGS_DENOMINATOR_SOURCE_INVENTORY_SCHEMA_VERSION = (
    "clinical_eeg_event_findings_denominator_source_inventory_v1"
)
EVENT_FINDINGS_DENOMINATOR_METHOD_ID = (
    "EEG-ONLY-INDEPENDENT-EVENT-FINDINGS-DENOMINATOR-V1"
)
EVENT_FINDINGS_DENOMINATOR_POLICY_ID = (
    "CLINICAL-EEG-EVENT-FINDINGS-DENOMINATOR-POLICY-V1"
)
DEFAULT_EVENT_FINDINGS_DENOMINATOR_POLICY_SHA256 = (
    "5a39af624f70c18e9383d8bef0d25ea315143c5587607f693b6a6b553f58094f"
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENT_FINDINGS_DENOMINATOR_POLICY_PATH = (
    _REPOSITORY_ROOT
    / "configs"
    / "clinical_eeg_event_findings_denominator_policy_v1.json"
)
EVENT_FINDINGS_DENOMINATOR_POLICY_SCHEMA_PATH = (
    _REPOSITORY_ROOT
    / "schemas"
    / "clinical_eeg_event_findings_denominator_policy_v1.schema.json"
)
EVENT_FINDINGS_DENOMINATOR_RECEIPT_SCHEMA_PATH = (
    _REPOSITORY_ROOT
    / "schemas"
    / "clinical_eeg_event_findings_denominator_receipt_v1.schema.json"
)

_EXPECTED_SCOPE_IDS = {
    "event_analysis_window",
    "background_context",
    "candidate_emergence_interval",
    "event_course_interval",
    "post_event_context",
    "quality_evaluable_interval",
}
_EVENT_UNIT = {
    "unit_type": "event",
    "unit_id": "GLOBAL",
    "unit_key": "event:GLOBAL",
}
_ALLOWED_PHYSICAL_UNIT_TYPES = {"electrode", "lead"}
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")

_SOURCE_FIREWALL_KEYS = {
    "event_findings_payload_used",
    "findings_candidates_used",
    "edf_annotations_used",
    "spreadsheet_used",
    "doctor_labels_used",
    "clinical_text_used",
    "patient_metadata_used",
    "video_used",
    "sleep_staging_used",
    "provocation_used",
    "ecg_emg_eog_used",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _self_hash(value: Mapping[str, object], field: str) -> str:
    candidate = deepcopy(dict(value))
    candidate.pop(field, None)
    return _canonical_sha256(candidate)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if type(value) is not dict:
        raise TypeError(f"{path} must contain a JSON object")
    return value


def event_findings_denominator_enumerator_code_sha256() -> str:
    """Hash the exact Python source implementing denominator enumeration."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _schema_errors(value: object, schema_path: Path) -> list[str]:
    validator = Draft202012Validator(_read_json(schema_path))
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
    rendered: list[str] = []
    for error in errors[:12]:
        path = "/" + "/".join(str(part) for part in error.path)
        rendered.append(f"{path}: {error.message}")
    if len(errors) > 12:
        rendered.append(f"... {len(errors) - 12} more error(s)")
    return rendered


def _require_id(value: object, context: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a canonical ID")
    return value


def _require_sha256(value: object, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _sorted_unique_ids(values: object, context: str) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{context} must be an ID array")
    result = sorted(_require_id(value, context) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{context} must not contain duplicate IDs")
    return result


def canonicalize_physical_interval_union(
    intervals: Sequence[object],
    *,
    tolerance_seconds: float = 1e-9,
) -> list[dict[str, float]]:
    """Return a sorted, gap-preserving half-open physical-time union.

    Overlapping and touching intervals are merged.  Genuine gaps are retained;
    consequently ``[0, 5) + [7, 10)`` has two components and eight evaluable
    seconds, rather than the ten seconds implied by a convex hull.
    """

    if isinstance(intervals, (str, bytes)) or not isinstance(intervals, Sequence):
        raise TypeError("physical interval union must be an array")
    if (
        isinstance(tolerance_seconds, bool)
        or not isinstance(tolerance_seconds, (int, float))
        or not math.isfinite(float(tolerance_seconds))
        or float(tolerance_seconds) < 0.0
    ):
        raise ValueError("tolerance_seconds must be a finite non-negative number")
    tolerance = float(tolerance_seconds)

    parsed: list[tuple[float, float]] = []
    for index, raw in enumerate(intervals):
        if isinstance(raw, Mapping):
            if set(raw) != {"start", "stop"}:
                raise ValueError(
                    f"interval[{index}] must contain exactly start and stop"
                )
            start_raw, stop_raw = raw["start"], raw["stop"]
        elif (
            isinstance(raw, Sequence)
            and not isinstance(raw, (str, bytes))
            and len(raw) == 2
        ):
            start_raw, stop_raw = raw
        else:
            raise TypeError(f"interval[{index}] must be an object or pair")
        if isinstance(start_raw, bool) or isinstance(stop_raw, bool):
            raise TypeError(f"interval[{index}] bounds must be numeric")
        try:
            start = float(start_raw)
            stop = float(stop_raw)
        except (TypeError, ValueError) as error:
            raise TypeError(f"interval[{index}] bounds must be numeric") from error
        if not math.isfinite(start) or not math.isfinite(stop):
            raise ValueError(f"interval[{index}] bounds must be finite")
        if start < 0.0:
            raise ValueError(
                f"interval[{index}] starts before the recording-relative origin"
            )
        if stop <= start:
            raise ValueError(
                f"interval[{index}] must have positive half-open duration"
            )
        # JSON distinguishes the textual encodings of -0.0 and 0.0 even
        # though they denote the same recording origin.  Normalize the signed
        # zero before sorting/hashing so interval identity is unique.
        if start == 0.0:
            start = 0.0
        parsed.append((start, stop))

    parsed.sort()
    merged: list[list[float]] = []
    for start, stop in parsed:
        if not merged or start > merged[-1][1] + tolerance:
            merged.append([start, stop])
            continue
        merged[-1][1] = max(merged[-1][1], stop)
    return [
        {"start": float(start), "stop": float(stop)}
        for start, stop in merged
    ]


def physical_interval_union_seconds(
    intervals: Sequence[Mapping[str, object]],
    *,
    tolerance_seconds: float = 1e-9,
) -> float:
    canonical = canonicalize_physical_interval_union(
        list(intervals), tolerance_seconds=tolerance_seconds
    )
    return float(
        math.fsum(row["stop"] - row["start"] for row in canonical)
    )


def _union_is_subset(
    subset: Sequence[Mapping[str, object]],
    superset: Sequence[Mapping[str, object]],
    tolerance: float,
) -> bool:
    outer = canonicalize_physical_interval_union(
        list(superset), tolerance_seconds=tolerance
    )
    inner = canonicalize_physical_interval_union(
        list(subset), tolerance_seconds=tolerance
    )
    outer_index = 0
    for segment in inner:
        while (
            outer_index < len(outer)
            and outer[outer_index]["stop"] < segment["start"] - tolerance
        ):
            outer_index += 1
        if outer_index == len(outer):
            return False
        support = outer[outer_index]
        if (
            support["start"] > segment["start"] + tolerance
            or support["stop"] < segment["stop"] - tolerance
        ):
            return False
    return True


def _unions_equal(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
    tolerance: float,
) -> bool:
    first = canonicalize_physical_interval_union(
        list(left), tolerance_seconds=tolerance
    )
    second = canonicalize_physical_interval_union(
        list(right), tolerance_seconds=tolerance
    )
    if len(first) != len(second):
        return False
    return all(
        abs(a["start"] - b["start"]) <= tolerance
        and abs(a["stop"] - b["stop"]) <= tolerance
        for a, b in zip(first, second)
    )


def _typed_unit(value: object, context: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    if set(value) != {"unit_type", "unit_id", "unit_key"}:
        raise ValueError(
            f"{context} must contain exactly unit_type, unit_id, and unit_key"
        )
    unit_type = _require_id(value["unit_type"], f"{context}.unit_type")
    unit_id = _require_id(value["unit_id"], f"{context}.unit_id")
    unit_key = _require_id(value["unit_key"], f"{context}.unit_key")
    if unit_type not in _ALLOWED_PHYSICAL_UNIT_TYPES | {"event"}:
        raise ValueError(f"{context} has an unsupported unit_type")
    if unit_key != f"{unit_type}:{unit_id}":
        raise ValueError(f"{context}.unit_key is not canonical for its type and ID")
    if unit_type == "event" and (unit_id != "GLOBAL" or unit_key != "event:GLOBAL"):
        raise ValueError("the event-global unit must be event:GLOBAL")
    if unit_type != "event" and unit_key == "event:GLOBAL":
        raise ValueError("physical units cannot use the event-global sentinel")
    return {
        "unit_type": unit_type,
        "unit_id": unit_id,
        "unit_key": unit_key,
    }


def _roster_specs(
    roster_policy: Mapping[str, object],
) -> dict[tuple[str, str], Mapping[str, object]]:
    result: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in roster_policy["core_atom_specs"]:  # type: ignore[index]
        result[("core_atom", str(row["atom_id"]))] = row
    for row in roster_policy["child_roster_specs"]:  # type: ignore[index]
        result[("child_roster", str(row["child_roster_id"]))] = row
    return result


def validate_event_findings_denominator_policy(
    value: object,
    *,
    trusted_policy_sha256: str | None = None,
    atom_roster_policy: Mapping[str, object] | None = None,
    trusted_atom_roster_policy_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate host trust and exact 28+12 roster/scope coverage."""

    if type(value) is not dict:
        raise TypeError("event Findings denominator policy must be an object")
    candidate = deepcopy(value)
    errors = _schema_errors(
        candidate, EVENT_FINDINGS_DENOMINATOR_POLICY_SCHEMA_PATH
    )
    if errors:
        raise ValueError(
            "event Findings denominator policy schema validation failed: "
            + "; ".join(errors)
        )
    expected_hash = _self_hash(candidate, "policy_sha256")
    if candidate["policy_sha256"] != expected_hash:
        raise ValueError("event Findings denominator policy_sha256 mismatch")
    if trusted_policy_sha256 is not None and expected_hash != trusted_policy_sha256:
        raise ValueError("event Findings denominator policy is not host trusted")
    if candidate["policy_id"] != EVENT_FINDINGS_DENOMINATOR_POLICY_ID:
        raise ValueError("event Findings denominator policy_id mismatch")

    roster = (
        load_event_findings_atom_roster_policy(
            trusted_policy_sha256=trusted_atom_roster_policy_sha256
        )
        if atom_roster_policy is None
        else deepcopy(dict(atom_roster_policy))
    )
    if atom_roster_policy is not None:
        # Reuse the public loader's validator through its checked-in trust
        # anchor unless a different anchor was explicitly supplied.
        from .event_findings_atom_roster import (
            validate_event_findings_atom_roster_policy,
        )

        roster = validate_event_findings_atom_roster_policy(
            roster,
            trusted_policy_sha256=(
                trusted_atom_roster_policy_sha256
                if trusted_atom_roster_policy_sha256 is not None
                else str(
                    load_event_findings_atom_roster_policy()["policy_sha256"]
                )
            ),
        )
    if candidate["atom_roster_policy_sha256"] != roster["policy_sha256"]:
        raise ValueError(
            "event Findings denominator policy is not bound to the trusted atom roster"
        )

    declared_scopes = list(candidate["scope_ids"])
    if set(declared_scopes) != _EXPECTED_SCOPE_IDS or len(declared_scopes) != len(
        _EXPECTED_SCOPE_IDS
    ):
        raise ValueError("denominator policy scope_ids do not match the frozen v1 set")
    specs = _roster_specs(roster)
    item_rows = list(candidate["item_scopes"])
    materialized_keys = [
        (str(row["roster_item_kind"]), str(row["roster_item_id"]))
        for row in item_rows
    ]
    if len(materialized_keys) != len(set(materialized_keys)):
        raise ValueError("denominator policy has duplicate roster item mappings")
    if set(materialized_keys) != set(specs):
        missing = sorted(set(specs) - set(materialized_keys))
        extra = sorted(set(materialized_keys) - set(specs))
        raise ValueError(
            "denominator policy does not exactly cover the trusted atom roster; "
            f"missing={missing}, extra={extra}"
        )
    used_scopes: set[str] = set()
    for row in item_rows:
        key = (str(row["roster_item_kind"]), str(row["roster_item_id"]))
        scope_id = str(row["scope_id"])
        if scope_id not in _EXPECTED_SCOPE_IDS:
            raise ValueError(f"{key}: denominator scope is not registered")
        used_scopes.add(scope_id)
        expected_granularity = (
            "unit"
            if "unit_mandatory" in specs[key]["structural_scopes"]
            else "event"
        )
        if row["granularity"] != expected_granularity:
            raise ValueError(
                f"{key}: denominator granularity disagrees with the atom roster"
            )
    if used_scopes != _EXPECTED_SCOPE_IDS:
        raise ValueError("every frozen denominator scope must be used")

    if set(candidate["source_firewall"]) != _SOURCE_FIREWALL_KEYS:
        raise ValueError("denominator policy source firewall is incomplete")
    if any(candidate["source_firewall"].values()):
        raise ValueError("denominator policy source firewall must exclude non-EEG inputs")
    return candidate


def load_event_findings_denominator_policy(
    path: str | Path = DEFAULT_EVENT_FINDINGS_DENOMINATOR_POLICY_PATH,
    *,
    trusted_policy_sha256: str | None = None,
    atom_roster_policy: Mapping[str, object] | None = None,
    trusted_atom_roster_policy_sha256: str | None = None,
) -> dict[str, Any]:
    resolved = Path(path)
    if trusted_policy_sha256 is None:
        if resolved.resolve() != DEFAULT_EVENT_FINDINGS_DENOMINATOR_POLICY_PATH.resolve():
            raise ValueError("a non-default denominator policy requires a host trust anchor")
        trusted_policy_sha256 = DEFAULT_EVENT_FINDINGS_DENOMINATOR_POLICY_SHA256
    return validate_event_findings_denominator_policy(
        _read_json(resolved),
        trusted_policy_sha256=trusted_policy_sha256,
        atom_roster_policy=atom_roster_policy,
        trusted_atom_roster_policy_sha256=trusted_atom_roster_policy_sha256,
    )


def _policy(
    value: Mapping[str, object] | None,
    trusted_policy_sha256: str | None,
) -> dict[str, Any]:
    if value is None:
        return load_event_findings_denominator_policy(
            trusted_policy_sha256=trusted_policy_sha256
        )
    if trusted_policy_sha256 is None:
        trusted_policy_sha256 = str(
            load_event_findings_denominator_policy()["policy_sha256"]
        )
    return validate_event_findings_denominator_policy(
        dict(value), trusted_policy_sha256=trusted_policy_sha256
    )


def enumerate_event_findings_denominator_cells(
    typed_expected_units: Sequence[Mapping[str, object]],
    *,
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
) -> list[dict[str, object]]:
    """Independently enumerate every registered item × typed-unit key."""

    denominator_policy = _policy(policy, trusted_policy_sha256)
    units = sorted(
        (_typed_unit(row, "typed_expected_units") for row in typed_expected_units),
        key=lambda row: row["unit_key"],
    )
    if not units:
        raise ValueError("typed expected-unit inventory must be non-empty")
    if any(row["unit_type"] == "event" for row in units):
        raise ValueError("typed expected-unit inventory must not contain event:GLOBAL")
    unit_keys = [row["unit_key"] for row in units]
    if len(unit_keys) != len(set(unit_keys)):
        raise ValueError("typed expected-unit inventory has duplicate canonical keys")

    result: list[dict[str, object]] = []
    for item in denominator_policy["item_scopes"]:
        targets = units if item["granularity"] == "unit" else [_EVENT_UNIT]
        for unit in targets:
            item_kind = str(item["roster_item_kind"])
            item_id = str(item["roster_item_id"])
            unit_key = str(unit["unit_key"])
            result.append(
                {
                    "cell_key": f"{item_kind}:{item_id}:{unit_key}",
                    "roster_item_kind": item_kind,
                    "roster_item_id": item_id,
                    "scope_id": str(item["scope_id"]),
                    "unit": deepcopy(unit),
                }
            )
    return sorted(result, key=lambda row: str(row["cell_key"]))


def _scope_keys(expected_cells: Sequence[Mapping[str, object]]) -> set[tuple[str, str]]:
    return {
        (str(row["scope_id"]), str(row["unit"]["unit_key"]))  # type: ignore[index]
        for row in expected_cells
    }


def _cell_keys(expected_cells: Sequence[Mapping[str, object]]) -> set[str]:
    return {str(row["cell_key"]) for row in expected_cells}


def _normalize_scope_row(
    value: object,
    *,
    tolerance: float,
    context: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    expected = {
        "scope_id",
        "unit_key",
        "required_interval_union",
        "evaluable_interval_union",
        "source_receipt_ids",
    }
    if set(value) != expected:
        raise ValueError(f"{context} fields do not match the v1 contract")
    required = canonicalize_physical_interval_union(
        value["required_interval_union"], tolerance_seconds=tolerance  # type: ignore[arg-type]
    )
    evaluable = canonicalize_physical_interval_union(
        value["evaluable_interval_union"], tolerance_seconds=tolerance  # type: ignore[arg-type]
    )
    if not _union_is_subset(evaluable, required, tolerance):
        raise ValueError(f"{context} evaluable union is outside its required scope")
    source_ids = _sorted_unique_ids(value["source_receipt_ids"], f"{context}.source_receipt_ids")
    if not source_ids:
        raise ValueError(f"{context} requires at least one trusted source receipt")
    return {
        "scope_id": _require_id(value["scope_id"], f"{context}.scope_id"),
        "unit_key": _require_id(value["unit_key"], f"{context}.unit_key"),
        "required_interval_union": required,
        "evaluable_interval_union": evaluable,
        "source_receipt_ids": source_ids,
    }


def _normalize_cell_qualification(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    expected = {
        "cell_key",
        "capability_receipt_ids",
        "sensitivity_receipt_ids",
        "technical_failure_receipt_ids",
    }
    if set(value) != expected:
        raise ValueError(f"{context} fields do not match the v1 contract")
    return {
        "cell_key": _require_id(value["cell_key"], f"{context}.cell_key"),
        "capability_receipt_ids": _sorted_unique_ids(
            value["capability_receipt_ids"], f"{context}.capability_receipt_ids"
        ),
        "sensitivity_receipt_ids": _sorted_unique_ids(
            value["sensitivity_receipt_ids"], f"{context}.sensitivity_receipt_ids"
        ),
        "technical_failure_receipt_ids": _sorted_unique_ids(
            value["technical_failure_receipt_ids"],
            f"{context}.technical_failure_receipt_ids",
        ),
    }


def _normalize_replay_sources(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("replay_sources must be an object")
    expected = {
        "enumerator_code_sha256",
        "event_scope_receipt_id",
        "event_scope_receipt_sha256",
        "unit_inventory_receipt_id",
        "unit_inventory_receipt_sha256",
        "technical_failure_ledger_receipt_id",
        "technical_failure_ledger_receipt_sha256",
    }
    if set(value) != expected:
        raise ValueError("replay_sources fields do not match the v1 contract")
    failure_id = value["technical_failure_ledger_receipt_id"]
    failure_hash = value["technical_failure_ledger_receipt_sha256"]
    if (failure_id is None) != (failure_hash is None):
        raise ValueError("technical-failure ledger ID and hash must be jointly nullable")
    return {
        "enumerator_code_sha256": _require_sha256(
            value["enumerator_code_sha256"], "enumerator_code_sha256"
        ),
        "event_scope_receipt_id": _require_id(
            value["event_scope_receipt_id"], "event_scope_receipt_id"
        ),
        "event_scope_receipt_sha256": _require_sha256(
            value["event_scope_receipt_sha256"], "event_scope_receipt_sha256"
        ),
        "unit_inventory_receipt_id": _require_id(
            value["unit_inventory_receipt_id"], "unit_inventory_receipt_id"
        ),
        "unit_inventory_receipt_sha256": _require_sha256(
            value["unit_inventory_receipt_sha256"],
            "unit_inventory_receipt_sha256",
        ),
        "technical_failure_ledger_receipt_id": (
            None
            if failure_id is None
            else _require_id(failure_id, "technical_failure_ledger_receipt_id")
        ),
        "technical_failure_ledger_receipt_sha256": (
            None
            if failure_hash is None
            else _require_sha256(
                failure_hash, "technical_failure_ledger_receipt_sha256"
            )
        ),
    }


def build_event_findings_denominator_source_inventory(
    *,
    record_id: str,
    event_id: str,
    canonical_signal_sha256: str,
    typed_expected_units: Sequence[Mapping[str, object]],
    scope_availability: Sequence[Mapping[str, object]],
    cell_qualifications: Sequence[Mapping[str, object]],
    replay_sources: Mapping[str, object],
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
    source_firewall: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build a canonical host-side source inventory.

    This helper is intended for a trusted pipeline stage.  The resulting hash
    still has to be supplied back as an external trust anchor when a
    denominator is materialized or validated.
    """

    denominator_policy = _policy(policy, trusted_policy_sha256)
    tolerance = float(
        denominator_policy["interval_union_policy"][  # type: ignore[index]
            "comparison_tolerance_seconds"
        ]
    )
    units = sorted(
        (_typed_unit(row, "typed_expected_units") for row in typed_expected_units),
        key=lambda row: row["unit_key"],
    )
    expected_cells = enumerate_event_findings_denominator_cells(
        units,
        policy=denominator_policy,
        trusted_policy_sha256=str(denominator_policy["policy_sha256"]),
    )
    normalized_scopes = sorted(
        (
            _normalize_scope_row(
                row, tolerance=tolerance, context="scope_availability"
            )
            for row in scope_availability
        ),
        key=lambda row: (str(row["scope_id"]), str(row["unit_key"])),
    )
    normalized_qualifications = sorted(
        (
            _normalize_cell_qualification(row, "cell_qualifications")
            for row in cell_qualifications
        ),
        key=lambda row: str(row["cell_key"]),
    )
    actual_scope_keys = [
        (str(row["scope_id"]), str(row["unit_key"]))
        for row in normalized_scopes
    ]
    if len(actual_scope_keys) != len(set(actual_scope_keys)):
        raise ValueError("scope_availability has duplicate scope/unit rows")
    if set(actual_scope_keys) != _scope_keys(expected_cells):
        raise ValueError("scope_availability does not exactly cover expected scope/unit keys")
    actual_cell_keys = [str(row["cell_key"]) for row in normalized_qualifications]
    if len(actual_cell_keys) != len(set(actual_cell_keys)):
        raise ValueError("cell_qualifications has duplicate cell rows")
    if set(actual_cell_keys) != _cell_keys(expected_cells):
        raise ValueError("cell_qualifications does not exactly cover expected cells")

    firewall = (
        {key: False for key in sorted(_SOURCE_FIREWALL_KEYS)}
        if source_firewall is None
        else deepcopy(dict(source_firewall))
    )
    if set(firewall) != _SOURCE_FIREWALL_KEYS or any(
        value is not False for value in firewall.values()
    ):
        raise ValueError("source inventory firewall must explicitly exclude all non-EEG inputs")

    seed = {
        "record_id": _require_id(record_id, "record_id"),
        "event_id": _require_id(event_id, "event_id"),
        "canonical_signal_sha256": _require_sha256(
            canonical_signal_sha256, "canonical_signal_sha256"
        ),
        "policy_sha256": denominator_policy["policy_sha256"],
        "typed_expected_units": units,
        "scope_availability": normalized_scopes,
        "cell_qualifications": normalized_qualifications,
        "replay_sources": _normalize_replay_sources(replay_sources),
        "source_firewall": firewall,
    }
    inventory: dict[str, Any] = {
        "schema_version": EVENT_FINDINGS_DENOMINATOR_SOURCE_INVENTORY_SCHEMA_VERSION,
        "source_inventory_id": f"EEGDENOMSRC-{_canonical_sha256(seed)[:24]}",
        **seed,
        "source_inventory_sha256": "0" * 64,
    }
    inventory["source_inventory_sha256"] = _self_hash(
        inventory, "source_inventory_sha256"
    )
    return validate_event_findings_denominator_source_inventory(
        inventory,
        trusted_source_inventory_sha256=str(inventory["source_inventory_sha256"]),
        policy=denominator_policy,
        trusted_policy_sha256=str(denominator_policy["policy_sha256"]),
    )


def validate_event_findings_denominator_source_inventory(
    value: object,
    *,
    trusted_source_inventory_sha256: str,
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("event Findings denominator source inventory must be an object")
    candidate = deepcopy(value)
    allowed = {
        "schema_version",
        "source_inventory_id",
        "record_id",
        "event_id",
        "canonical_signal_sha256",
        "policy_sha256",
        "typed_expected_units",
        "scope_availability",
        "cell_qualifications",
        "replay_sources",
        "source_firewall",
        "source_inventory_sha256",
    }
    if set(candidate) != allowed:
        raise ValueError("source inventory fields do not match the v1 contract")
    if candidate["schema_version"] != EVENT_FINDINGS_DENOMINATOR_SOURCE_INVENTORY_SCHEMA_VERSION:
        raise ValueError("source inventory schema_version mismatch")
    _require_id(candidate["source_inventory_id"], "source_inventory_id")
    _require_id(candidate["record_id"], "record_id")
    _require_id(candidate["event_id"], "event_id")
    _require_sha256(candidate["canonical_signal_sha256"], "canonical_signal_sha256")
    expected_hash = _self_hash(candidate, "source_inventory_sha256")
    if candidate["source_inventory_sha256"] != expected_hash:
        raise ValueError("source inventory SHA-256 mismatch")
    if expected_hash != _require_sha256(
        trusted_source_inventory_sha256, "trusted_source_inventory_sha256"
    ):
        raise ValueError("source inventory is not host trusted")

    denominator_policy = _policy(policy, trusted_policy_sha256)
    if candidate["policy_sha256"] != denominator_policy["policy_sha256"]:
        raise ValueError("source inventory is bound to another denominator policy")
    units = [
        _typed_unit(row, "typed_expected_units")
        for row in candidate["typed_expected_units"]
    ]
    if units != sorted(units, key=lambda row: row["unit_key"]):
        raise ValueError("typed expected units must be canonically sorted")
    if not units or any(row["unit_type"] == "event" for row in units):
        raise ValueError("source inventory requires non-event physical units")
    if len({row["unit_key"] for row in units}) != len(units):
        raise ValueError("source inventory typed unit keys must be unique")
    expected_cells = enumerate_event_findings_denominator_cells(
        units,
        policy=denominator_policy,
        trusted_policy_sha256=str(denominator_policy["policy_sha256"]),
    )
    tolerance = float(
        denominator_policy["interval_union_policy"][  # type: ignore[index]
            "comparison_tolerance_seconds"
        ]
    )
    scopes = [
        _normalize_scope_row(row, tolerance=tolerance, context="scope_availability")
        for row in candidate["scope_availability"]
    ]
    if candidate["scope_availability"] != scopes:
        raise ValueError(
            "source inventory interval unions/receipt IDs are not canonical"
        )
    if scopes != sorted(
        scopes, key=lambda row: (str(row["scope_id"]), str(row["unit_key"]))
    ):
        raise ValueError("scope availability rows must be canonically sorted")
    scope_keys = [(str(row["scope_id"]), str(row["unit_key"])) for row in scopes]
    if len(scope_keys) != len(set(scope_keys)) or set(scope_keys) != _scope_keys(
        expected_cells
    ):
        raise ValueError("source inventory scope rows do not exactly close the denominator")

    qualifications = [
        _normalize_cell_qualification(row, "cell_qualifications")
        for row in candidate["cell_qualifications"]
    ]
    if candidate["cell_qualifications"] != qualifications:
        raise ValueError("source inventory cell qualifications are not canonical")
    if qualifications != sorted(qualifications, key=lambda row: str(row["cell_key"])):
        raise ValueError("cell qualification rows must be canonically sorted")
    qualification_keys = [str(row["cell_key"]) for row in qualifications]
    if len(qualification_keys) != len(set(qualification_keys)) or set(
        qualification_keys
    ) != _cell_keys(expected_cells):
        raise ValueError("source inventory cell rows do not exactly close the denominator")
    replay_sources = _normalize_replay_sources(candidate["replay_sources"])
    if candidate["replay_sources"] != replay_sources:
        raise ValueError("source inventory replay sources are not canonical")
    if replay_sources["enumerator_code_sha256"] != (
        event_findings_denominator_enumerator_code_sha256()
    ):
        raise ValueError("source inventory enumerator code hash is not exact")
    failure_rows = [
        row for row in qualifications if row["technical_failure_receipt_ids"]
    ]
    failure_ledger_id = replay_sources["technical_failure_ledger_receipt_id"]
    if failure_rows and failure_ledger_id is None:
        raise ValueError(
            "cell technical failures require a host-trusted failure ledger"
        )
    if failure_rows and any(
        row["technical_failure_receipt_ids"] != [failure_ledger_id]
        for row in failure_rows
    ):
        raise ValueError(
            "v1 failure cells must bind exactly to the trusted failure-ledger receipt"
        )
    event_scope_receipt_id = str(replay_sources["event_scope_receipt_id"])
    if any(
        event_scope_receipt_id not in row["source_receipt_ids"] for row in scopes
    ):
        raise ValueError(
            "every scope row must bind the replay event-scope receipt ID"
        )
    firewall = candidate["source_firewall"]
    if not isinstance(firewall, Mapping) or set(firewall) != _SOURCE_FIREWALL_KEYS:
        raise ValueError("source inventory firewall is incomplete")
    if any(value is not False for value in firewall.values()):
        raise ValueError("source inventory used a forbidden non-EEG source")
    identifier_seed = {
        "record_id": candidate["record_id"],
        "event_id": candidate["event_id"],
        "canonical_signal_sha256": candidate["canonical_signal_sha256"],
        "policy_sha256": candidate["policy_sha256"],
        "typed_expected_units": candidate["typed_expected_units"],
        "scope_availability": candidate["scope_availability"],
        "cell_qualifications": candidate["cell_qualifications"],
        "replay_sources": candidate["replay_sources"],
        "source_firewall": candidate["source_firewall"],
    }
    if candidate["source_inventory_id"] != (
        f"EEGDENOMSRC-{_canonical_sha256(identifier_seed)[:24]}"
    ):
        raise ValueError("source inventory ID is not content addressed")
    return candidate


def _roster_spec_by_item() -> dict[tuple[str, str], Mapping[str, object]]:
    return _roster_specs(load_event_findings_atom_roster_policy())


def materialize_event_findings_denominator_receipt(
    source_inventory: object,
    *,
    trusted_source_inventory_sha256: str,
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
) -> dict[str, Any]:
    """Materialize a candidate-blind, deterministic denominator receipt."""

    denominator_policy = _policy(policy, trusted_policy_sha256)
    source = validate_event_findings_denominator_source_inventory(
        source_inventory,
        trusted_source_inventory_sha256=trusted_source_inventory_sha256,
        policy=denominator_policy,
        trusted_policy_sha256=str(denominator_policy["policy_sha256"]),
    )
    units = list(source["typed_expected_units"])
    expected_cells = enumerate_event_findings_denominator_cells(
        units,
        policy=denominator_policy,
        trusted_policy_sha256=str(denominator_policy["policy_sha256"]),
    )
    scope_by_key = {
        (str(row["scope_id"]), str(row["unit_key"])): row
        for row in source["scope_availability"]
    }
    qualification_by_key = {
        str(row["cell_key"]): row for row in source["cell_qualifications"]
    }
    roster_specs = _roster_spec_by_item()
    tolerance = float(
        denominator_policy["interval_union_policy"][  # type: ignore[index]
            "comparison_tolerance_seconds"
        ]
    )

    cells: list[dict[str, Any]] = []
    for expected in expected_cells:
        scope = scope_by_key[
            (str(expected["scope_id"]), str(expected["unit"]["unit_key"]))  # type: ignore[index]
        ]
        qualification = qualification_by_key[str(expected["cell_key"])]
        required_union = deepcopy(scope["required_interval_union"])
        available_union = deepcopy(scope["evaluable_interval_union"])
        required_seconds = physical_interval_union_seconds(
            required_union, tolerance_seconds=tolerance
        )
        available_seconds = physical_interval_union_seconds(
            available_union, tolerance_seconds=tolerance
        )
        coverage = (
            0.0
            if required_seconds <= tolerance
            else min(1.0, available_seconds / required_seconds)
        )
        capability_ids = list(qualification["capability_receipt_ids"])
        sensitivity_ids = list(qualification["sensitivity_receipt_ids"])
        failure_ids = list(qualification["technical_failure_receipt_ids"])
        reasons: list[str] = []
        if failure_ids:
            opportunity_status = "not_evaluable"
            reasons.append("trusted_technical_failure")
        elif not capability_ids:
            opportunity_status = "not_evaluable"
            reasons.append("capability_receipt_missing")
        elif required_seconds <= tolerance:
            opportunity_status = "not_evaluable"
            reasons.append("required_scope_empty_or_censored")
        elif available_seconds <= tolerance:
            opportunity_status = "not_evaluable"
            reasons.append("no_evaluable_signal_support")
        elif _unions_equal(available_union, required_union, tolerance):
            opportunity_status = "sufficient"
            reasons.append("complete_gap_preserving_scope_coverage")
        else:
            opportunity_status = "limited"
            reasons.append("partial_gap_preserving_scope_coverage")

        # ``evaluable_interval_union`` is the final opportunity after all
        # capability/failure gates, rather than raw signal availability.  A
        # not-evaluable cell therefore retains its required scope and lineage
        # but has an empty final opportunity union.
        if opportunity_status == "not_evaluable":
            evaluable_union: list[dict[str, float]] = []
            evaluable_seconds = 0.0
            coverage = 0.0
        else:
            evaluable_union = available_union
            evaluable_seconds = available_seconds

        item_key = (
            str(expected["roster_item_kind"]),
            str(expected["roster_item_id"]),
        )
        roster_spec = roster_specs[item_key]
        structural_absence_authorized = bool(
            roster_spec["absence_requires_complete_opportunity"]
            and opportunity_status == "sufficient"
            and (
                not roster_spec["absence_requires_sensitivity_receipt"]
                or sensitivity_ids
            )
        )
        if (
            opportunity_status == "sufficient"
            and roster_spec["absence_requires_sensitivity_receipt"]
            and not sensitivity_ids
        ):
            reasons.append("matching_sensitivity_receipt_missing")
        if structural_absence_authorized:
            reasons.append("roster_absence_opportunity_authorized_only")
        else:
            reasons.append("roster_absence_not_authorized")

        cell: dict[str, Any] = {
            "cell_key": str(expected["cell_key"]),
            "roster_item_kind": str(expected["roster_item_kind"]),
            "roster_item_id": str(expected["roster_item_id"]),
            "scope_id": str(expected["scope_id"]),
            "unit": deepcopy(expected["unit"]),
            "required_interval_union": required_union,
            "evaluable_interval_union": evaluable_union,
            "required_seconds": required_seconds,
            "evaluable_seconds": evaluable_seconds,
            "coverage_fraction": coverage,
            "opportunity_status": opportunity_status,
            "processing_disposition": (
                "technical_failure" if failure_ids else "completed"
            ),
            "capability_receipt_ids": capability_ids,
            "sensitivity_receipt_ids": sensitivity_ids,
            "source_receipt_ids": list(scope["source_receipt_ids"]),
            "technical_failure_receipt_ids": failure_ids,
            "absence_authorized": structural_absence_authorized,
            "reason_codes": sorted(set(reasons)),
            "cell_sha256": "0" * 64,
        }
        cell["cell_sha256"] = _self_hash(cell, "cell_sha256")
        cells.append(cell)

    cells.sort(key=lambda row: row["cell_key"])
    expected_keys = [str(row["cell_key"]) for row in expected_cells]
    materialized_keys = [str(row["cell_key"]) for row in cells]
    expected_item_keys = sorted(
        f"{row['roster_item_kind']}:{row['roster_item_id']}"
        for row in denominator_policy["item_scopes"]
    )
    materialized_item_keys = sorted(
        {
            f"{row['roster_item_kind']}:{row['roster_item_id']}"
            for row in cells
        }
    )
    event_global_count = sum(
        row["unit"]["unit_type"] == "event" for row in cells
    )
    typed_unit_count = len(cells) - event_global_count
    summary = {
        "expected_roster_item_count": len(expected_item_keys),
        "materialized_roster_item_count": len(materialized_item_keys),
        "expected_cell_count": len(expected_keys),
        "materialized_cell_count": len(materialized_keys),
        "expected_event_global_cell_count": event_global_count,
        "materialized_event_global_cell_count": event_global_count,
        "expected_typed_unit_cell_count": typed_unit_count,
        "materialized_typed_unit_cell_count": typed_unit_count,
        "expected_unit_count": len(units),
        "sufficient_count": sum(
            row["opportunity_status"] == "sufficient" for row in cells
        ),
        "limited_count": sum(
            row["opportunity_status"] == "limited" for row in cells
        ),
        "not_evaluable_count": sum(
            row["opportunity_status"] == "not_evaluable" for row in cells
        ),
        "technical_failure_count": sum(
            bool(row["technical_failure_receipt_ids"]) for row in cells
        ),
        "absence_authorized_count": sum(
            bool(row["absence_authorized"]) for row in cells
        ),
        "expected_roster_item_keys_sha256": _canonical_sha256(
            expected_item_keys
        ),
        "materialized_roster_item_keys_sha256": _canonical_sha256(
            materialized_item_keys
        ),
        "expected_cell_keys_sha256": _canonical_sha256(expected_keys),
        "materialized_cell_keys_sha256": _canonical_sha256(materialized_keys),
        "all_expected_roster_items_materialized": (
            expected_item_keys == materialized_item_keys
        ),
        "all_expected_cells_materialized": expected_keys == materialized_keys,
        "gap_preserving": True,
        "double_counting_allowed": False,
        "structural_completeness_only": True,
        "item_scope_denominator_only": True,
        "term_query_denominator_materialized": False,
        "clinical_term_absence_authorized": False,
        "clinical_correctness_claimed": False,
        "full_report_factuality_claimed": False,
    }
    source_hash = str(source["source_inventory_sha256"])
    replay_sources = dict(source["replay_sources"])
    replay_binding = {
        "enumerator_code_sha256": replay_sources["enumerator_code_sha256"],
        "trusted_input_manifest_sha256": source_hash,
        "event_scope_receipt_id": replay_sources["event_scope_receipt_id"],
        "event_scope_receipt_sha256": replay_sources[
            "event_scope_receipt_sha256"
        ],
        "unit_inventory_receipt_id": replay_sources[
            "unit_inventory_receipt_id"
        ],
        "unit_inventory_receipt_sha256": replay_sources[
            "unit_inventory_receipt_sha256"
        ],
        "technical_failure_ledger_receipt_id": replay_sources[
            "technical_failure_ledger_receipt_id"
        ],
        "technical_failure_ledger_receipt_sha256": replay_sources[
            "technical_failure_ledger_receipt_sha256"
        ],
        "host_trust_required": True,
        "exact_replay_required": True,
        "event_findings_payload_excluded": True,
    }
    receipt_seed = {
        "method_id": EVENT_FINDINGS_DENOMINATOR_METHOD_ID,
        "record_id": source["record_id"],
        "event_id": source["event_id"],
        "canonical_signal_sha256": source["canonical_signal_sha256"],
        "policy_sha256": denominator_policy["policy_sha256"],
        "source_inventory_sha256": source_hash,
        "replay_binding": replay_binding,
    }
    receipt: dict[str, Any] = {
        "schema_version": EVENT_FINDINGS_DENOMINATOR_RECEIPT_SCHEMA_VERSION,
        "receipt_id": f"EEGDENOM-{_canonical_sha256(receipt_seed)[:24]}",
        "method_id": EVENT_FINDINGS_DENOMINATOR_METHOD_ID,
        "event_id": str(source["event_id"]),
        "record_id": str(source["record_id"]),
        "canonical_signal_sha256": str(source["canonical_signal_sha256"]),
        "policy_id": str(denominator_policy["policy_id"]),
        "policy_sha256": str(denominator_policy["policy_sha256"]),
        "atom_roster_policy_sha256": str(
            denominator_policy["atom_roster_policy_sha256"]
        ),
        "source_inventory_id": str(source["source_inventory_id"]),
        "source_inventory_sha256": source_hash,
        "expected_unit_inventory_sha256": _canonical_sha256(units),
        "typed_expected_units": deepcopy(units),
        "replay_binding": replay_binding,
        "source_firewall": deepcopy(source["source_firewall"]),
        "cells": cells,
        "summary": summary,
        "receipt_sha256": "0" * 64,
    }
    receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")
    errors = _schema_errors(
        receipt, EVENT_FINDINGS_DENOMINATOR_RECEIPT_SCHEMA_PATH
    )
    if errors:
        raise ValueError(
            "materialized event Findings denominator receipt is invalid: "
            + "; ".join(errors)
        )
    return receipt


def validate_event_findings_denominator_receipt(
    value: object,
    *,
    source_inventory: object,
    trusted_source_inventory_sha256: str,
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate self-hashes and then exact-replay from host-trusted inputs."""

    if type(value) is not dict:
        raise TypeError("event Findings denominator receipt must be an object")
    candidate = deepcopy(value)
    errors = _schema_errors(
        candidate, EVENT_FINDINGS_DENOMINATOR_RECEIPT_SCHEMA_PATH
    )
    if errors:
        raise ValueError(
            "event Findings denominator receipt schema validation failed: "
            + "; ".join(errors)
        )
    if candidate["receipt_sha256"] != _self_hash(candidate, "receipt_sha256"):
        raise ValueError("event Findings denominator receipt_sha256 mismatch")
    for cell in candidate["cells"]:
        if cell["cell_sha256"] != _self_hash(cell, "cell_sha256"):
            raise ValueError(
                f"event Findings denominator cell hash mismatch: {cell['cell_key']}"
            )
    expected = materialize_event_findings_denominator_receipt(
        source_inventory,
        trusted_source_inventory_sha256=trusted_source_inventory_sha256,
        policy=policy,
        trusted_policy_sha256=trusted_policy_sha256,
    )
    if candidate != expected:
        raise ValueError(
            "event Findings denominator receipt does not match exact host-side replay"
        )
    return candidate


def validate_atom_roster_against_independent_denominator(
    denominator_receipt: object,
    atom_roster_receipt: object,
    *,
    source_inventory: object,
    trusted_source_inventory_sha256: str,
    event_findings_v3: object,
    denominator_policy: Mapping[str, object] | None = None,
    trusted_denominator_policy_sha256: str | None = None,
    atom_roster_unit_key_by_id: Mapping[str, str] | None = None,
    atom_roster_validation_kwargs: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Join the source-accounting roster to the independent denominator.

    The join authorizes only the roster-level ``absent_with_opportunity``
    state.  It does not qualify a report term.  Ambiguous untyped legacy unit
    IDs fail closed; callers may supply a trusted explicit ID→typed-key map.
    """

    denominator = validate_event_findings_denominator_receipt(
        denominator_receipt,
        source_inventory=source_inventory,
        trusted_source_inventory_sha256=trusted_source_inventory_sha256,
        policy=denominator_policy,
        trusted_policy_sha256=trusted_denominator_policy_sha256,
    )
    roster_kwargs = dict(atom_roster_validation_kwargs or {})
    roster = validate_event_findings_atom_roster_receipt(
        atom_roster_receipt,
        event_findings_v3=event_findings_v3,
        **roster_kwargs,
    )
    for field in ("event_id", "record_id", "canonical_signal_sha256"):
        if denominator[field] != roster[field]:
            raise ValueError(f"denominator/atom-roster {field} identity mismatch")
    if denominator["atom_roster_policy_sha256"] != roster["roster_policy_sha256"]:
        raise ValueError("denominator and atom roster use different roster policies")

    typed_units = list(denominator["typed_expected_units"])
    by_unit_id: dict[str, list[str]] = {}
    typed_by_key = {
        str(unit["unit_key"]): unit for unit in typed_units
    }
    for unit in typed_units:
        by_unit_id.setdefault(str(unit["unit_id"]), []).append(str(unit["unit_key"]))
    supplied_map = {
        str(key): str(value)
        for key, value in dict(atom_roster_unit_key_by_id or {}).items()
    }
    roster_unit_ids = [str(value) for value in roster["expected_unit_ids"]]
    if supplied_map:
        if set(supplied_map) != set(roster_unit_ids):
            raise ValueError("trusted atom-roster unit mapping is not exact")
        if set(supplied_map.values()) != {
            str(unit["unit_key"]) for unit in typed_units
        }:
            raise ValueError("trusted atom-roster unit mapping does not close typed units")
        if any(
            mapped_key not in typed_by_key
            or str(typed_by_key[mapped_key]["unit_id"]) != roster_unit_id
            for roster_unit_id, mapped_key in supplied_map.items()
        ):
            raise ValueError(
                "v1 atom-roster join forbids aliasing or permuting canonical unit IDs"
            )
    else:
        for unit_id in roster_unit_ids:
            candidates = by_unit_id.get(unit_id, [])
            if len(candidates) != 1:
                raise ValueError(
                    f"legacy atom-roster unit {unit_id} has no unique typed mapping"
                )
            supplied_map[unit_id] = candidates[0]
        if set(supplied_map.values()) != {
            str(unit["unit_key"]) for unit in typed_units
        }:
            raise ValueError("atom roster and denominator typed-unit inventories differ")

    cells = {str(row["cell_key"]): row for row in denominator["cells"]}
    item_policy = {
        (str(row["roster_item_kind"]), str(row["roster_item_id"])): row
        for row in _policy(
            denominator_policy, trusted_denominator_policy_sha256
        )["item_scopes"]
    }
    roster_specs = _roster_spec_by_item()
    violations: list[str] = []
    authorized_count = 0

    def check_row(
        row: Mapping[str, object], item_kind: str, item_id_key: str, status_key: str
    ) -> None:
        nonlocal authorized_count
        item_id = str(row[item_id_key])
        item = item_policy[(item_kind, item_id)]
        unit_keys = (
            sorted(supplied_map.values())
            if item["granularity"] == "unit"
            else ["event:GLOBAL"]
        )
        expected_cells = [
            cells[f"{item_kind}:{item_id}:{unit_key}"] for unit_key in unit_keys
        ]
        denominator_failures = [
            bool(cell["technical_failure_receipt_ids"])
            for cell in expected_cells
        ]
        roster_item_failed = row["processing_disposition"] == "technical_failure"
        if roster_item_failed != all(denominator_failures):
            violations.append(
                f"{item_id}: atom-roster and denominator technical-failure lineage differ"
            )
        roster_sensitivity_ids = set(
            str(value) for value in row["sensitivity_receipt_ids"]
        )

        def cell_authorizes_roster_absence(cell: Mapping[str, object]) -> bool:
            if not cell["absence_authorized"]:
                return False
            spec = roster_specs[(item_kind, item_id)]
            if not spec["absence_requires_sensitivity_receipt"]:
                return True
            return bool(
                roster_sensitivity_ids.intersection(
                    str(value) for value in cell["sensitivity_receipt_ids"]
                )
            )

        if row[status_key] == "absent_with_opportunity":
            if row["processing_disposition"] != "completed":
                violations.append(f"{item_id}: technical failure cannot create absence")
            unauthorized = [
                str(cell["cell_key"])
                for cell in expected_cells
                if not cell_authorizes_roster_absence(cell)
            ]
            if unauthorized:
                violations.append(
                    f"{item_id}: independent denominator does not authorize absence "
                    + ",".join(unauthorized)
                )
            else:
                authorized_count += 1
        for unit_row in row["unit_dispositions"]:
            unit_id = str(unit_row["unit_id"])
            if unit_id not in supplied_map:
                violations.append(f"{item_id}: unmapped unit {unit_id}")
                continue
            cell_key = f"{item_kind}:{item_id}:{supplied_map[unit_id]}"
            cell = cells[cell_key]
            if (
                unit_row["processing_disposition"] == "technical_failure"
            ) != bool(cell["technical_failure_receipt_ids"]):
                violations.append(
                    f"{item_id}/{unit_id}: technical-failure lineage differs"
                )
            if unit_row["status"] != "absent_with_opportunity":
                continue
            if (
                unit_row["processing_disposition"] != "completed"
                or not cell_authorizes_roster_absence(cell)
            ):
                violations.append(
                    f"{item_id}/{unit_id}: independent denominator does not authorize absence"
                )
            else:
                authorized_count += 1

    for row in roster["core_entries"]:
        check_row(row, "core_atom", "atom_id", "status")
    for row in roster["child_rosters"]:
        check_row(row, "child_roster", "child_roster_id", "finding_status")
    if violations:
        raise ValueError("; ".join(violations))

    result: dict[str, Any] = {
        "schema_version": "clinical_eeg_event_findings_denominator_roster_join_v1",
        "event_id": denominator["event_id"],
        "record_id": denominator["record_id"],
        "canonical_signal_sha256": denominator["canonical_signal_sha256"],
        "denominator_receipt_id": denominator["receipt_id"],
        "denominator_receipt_sha256": denominator["receipt_sha256"],
        "atom_roster_receipt_id": roster["receipt_id"],
        "atom_roster_receipt_sha256": roster["receipt_sha256"],
        "structural_absence_authorized_count": authorized_count,
        "all_roster_absences_independently_authorized": True,
        "clinical_term_promotion_authorized": False,
        "join_sha256": "0" * 64,
    }
    result["join_sha256"] = _self_hash(result, "join_sha256")
    return result


__all__ = [
    "DEFAULT_EVENT_FINDINGS_DENOMINATOR_POLICY_PATH",
    "EVENT_FINDINGS_DENOMINATOR_METHOD_ID",
    "EVENT_FINDINGS_DENOMINATOR_POLICY_ID",
    "DEFAULT_EVENT_FINDINGS_DENOMINATOR_POLICY_SHA256",
    "EVENT_FINDINGS_DENOMINATOR_POLICY_SCHEMA_PATH",
    "EVENT_FINDINGS_DENOMINATOR_POLICY_SCHEMA_VERSION",
    "EVENT_FINDINGS_DENOMINATOR_RECEIPT_SCHEMA_PATH",
    "EVENT_FINDINGS_DENOMINATOR_RECEIPT_SCHEMA_VERSION",
    "EVENT_FINDINGS_DENOMINATOR_SOURCE_INVENTORY_SCHEMA_VERSION",
    "build_event_findings_denominator_source_inventory",
    "canonicalize_physical_interval_union",
    "enumerate_event_findings_denominator_cells",
    "event_findings_denominator_enumerator_code_sha256",
    "load_event_findings_denominator_policy",
    "materialize_event_findings_denominator_receipt",
    "physical_interval_union_seconds",
    "validate_atom_roster_against_independent_denominator",
    "validate_event_findings_denominator_policy",
    "validate_event_findings_denominator_receipt",
    "validate_event_findings_denominator_source_inventory",
]
