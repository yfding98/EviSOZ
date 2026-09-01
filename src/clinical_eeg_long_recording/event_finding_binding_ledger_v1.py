"""Strict additive primary/secondary binding ledger for event EEG Findings.

This module is a public/synthetic-only shadow gate.  It does not change the
frozen ``event_eeg_findings_v3`` wire, the atom-roster implementation, the
private report route, or any Qwen path.  Instead it joins already materialized
EEG-only evidence to an independently enumerated term-query denominator.

The central invariant is intentionally simple: every source Finding receives
exactly one *primary disposition* and every independent query cell receives
exactly one query binding.  Status and assertion level are copied from the
same primary Finding.  Secondary links are typed provenance only; they can
never upgrade an assertion, add a vote, alter a denominator, or authorize a
surface claim.

Version 1 is fail closed.  Clinical correctness, cortical SOZ/EZ validity and
production connectivity are never claimed by this structural receipt.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator

from .event_findings_atom_roster import (
    validate_event_findings_atom_roster_receipt,
)
from .event_findings_denominator import (
    canonicalize_physical_interval_union,
    validate_atom_roster_against_independent_denominator,
)
from .event_findings_v3_validation import (
    validate_event_eeg_findings_v3_payload,
)


EVENT_FINDING_BINDING_POLICY_SCHEMA_VERSION = (
    "clinical_eeg_event_finding_binding_policy_v1"
)
EVENT_FINDING_BINDING_LEDGER_SCHEMA_VERSION = (
    "clinical_eeg_event_finding_binding_ledger_v1"
)
EVENT_FINDING_BINDING_METHOD_ID = (
    "EEG-ONLY-EVENT-FINDING-PRIMARY-SECONDARY-BINDING-V1"
)
EVENT_FINDING_BINDING_POLICY_ID = (
    "CLINICAL-EEG-EVENT-FINDING-BINDING-POLICY-V1"
)
EVENT_FINDING_PHYSICAL_INSTANCE_INVENTORY_SCHEMA_VERSION = (
    "clinical_eeg_event_finding_physical_instance_inventory_v1"
)
EVENT_FINDING_PRODUCER_BINDING_INVENTORY_SCHEMA_VERSION = (
    "clinical_eeg_event_finding_producer_binding_inventory_v1"
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENT_FINDING_BINDING_POLICY_PATH = (
    _ROOT / "configs" / "clinical_eeg_event_finding_binding_policy_v1.json"
)
EVENT_FINDING_BINDING_POLICY_SCHEMA_PATH = (
    _ROOT / "schemas" / "clinical_eeg_event_finding_binding_policy_v1.schema.json"
)
EVENT_FINDING_BINDING_LEDGER_SCHEMA_PATH = (
    _ROOT / "schemas" / "clinical_eeg_event_finding_binding_ledger_v1.schema.json"
)

DEFAULT_EVENT_FINDING_BINDING_POLICY_SHA256 = (
    "02fd2097d4c67ae80dfc5278f2d868d041a54c5dc3d97b14fe3eace3c9c0cf2e"
)

_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
)
_SHA_CHARS = frozenset("0123456789abcdef")
_FINDING_STATUSES = {
    "present",
    "absent_with_opportunity",
    "uncertain",
    "not_evaluable",
}
_ASSERTION_LEVELS = {
    "measured",
    "model_candidate",
    "report_eligible_automated",
}
_SECONDARY_RELATIONS = {
    "alternate_reference_corroboration",
    "same_instance_component",
    "derived_projection",
    "later_context",
    "competing_interpretation",
    "counterevidence",
}
_SOURCE_FIREWALL = {
    "eeg_signal_only": True,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_used": False,
    "sleep_staging_used": False,
    "provocation_used": False,
    "ecg_emg_eog_used": False,
}
_V3_VALIDATION_KWARGS = {
    "trusted_producer_receipts",
    "trusted_calibration_receipts",
    "trusted_capability_qualification_receipts",
    "trusted_sensitivity_receipts",
    "trusted_term_decision_receipts",
    "trusted_registry_bindings",
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
    source = deepcopy(dict(value))
    source.pop(field, None)
    return _canonical_sha256(source)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _schema_errors(value: object, path: Path) -> list[str]:
    schema = _read_json(path)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: list(item.path),
    )
    rendered: list[str] = []
    for error in errors[:16]:
        where = "/" + "/".join(str(item) for item in error.path)
        rendered.append(f"{where}: {error.message}")
    if len(errors) > 16:
        rendered.append(f"... {len(errors) - 16} more error(s)")
    return rendered


def _id(value: object, context: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 256
        or value[0] not in _ID_CHARS - frozenset("._:-")
        or any(character not in _ID_CHARS for character in value)
    ):
        raise ValueError(f"{context} must be a canonical ID")
    return value


def _sha(value: object, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA_CHARS for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _unique(values: Iterable[str], context: str) -> set[str]:
    rows = list(values)
    if len(rows) != len(set(rows)):
        raise ValueError(f"{context} contains duplicate IDs")
    return set(rows)


def _typed_unit(value: object, context: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    if set(value) != {"unit_type", "unit_id", "unit_key"}:
        raise ValueError(
            f"{context} must contain exactly unit_type, unit_id, unit_key"
        )
    unit_type = _id(value["unit_type"], f"{context}.unit_type")
    unit_id = _id(value["unit_id"], f"{context}.unit_id")
    unit_key = _id(value["unit_key"], f"{context}.unit_key")
    if unit_type not in {"event", "lead", "electrode"}:
        raise ValueError(f"{context}.unit_type is unsupported")
    if unit_key != f"{unit_type}:{unit_id}":
        raise ValueError(f"{context}.unit_key is not type-safe")
    if unit_type == "event" and unit_key != "event:GLOBAL":
        raise ValueError("event-global unit must be event:GLOBAL")
    if unit_type != "event" and unit_key == "event:GLOBAL":
        raise ValueError("physical unit cannot use event:GLOBAL")
    return {
        "unit_type": unit_type,
        "unit_id": unit_id,
        "unit_key": unit_key,
    }


def _typed_unit_inventory(value: object, context: str) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be an array")
    rows = [_typed_unit(item, f"{context}[{index}]") for index, item in enumerate(value)]
    keys = [row["unit_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{context} contains duplicate typed units")
    return sorted(rows, key=lambda row: row["unit_key"])


def _interval_union(
    value: object,
    *,
    context: str,
    tolerance: float,
) -> list[dict[str, float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be an interval array")
    return canonicalize_physical_interval_union(
        list(value), tolerance_seconds=tolerance
    )


def _union_subset(
    subset: Sequence[Mapping[str, object]],
    superset: Sequence[Mapping[str, object]],
    *,
    tolerance: float,
) -> bool:
    inner = canonicalize_physical_interval_union(
        list(subset), tolerance_seconds=tolerance
    )
    outer = canonicalize_physical_interval_union(
        list(superset), tolerance_seconds=tolerance
    )
    outer_index = 0
    for segment in inner:
        while (
            outer_index < len(outer)
            and outer[outer_index]["stop"] < segment["start"] - tolerance
        ):
            outer_index += 1
        if outer_index >= len(outer):
            return False
        carrier = outer[outer_index]
        if (
            carrier["start"] > segment["start"] + tolerance
            or carrier["stop"] < segment["stop"] - tolerance
        ):
            return False
    return True


def _union_equal(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
    *,
    tolerance: float,
) -> bool:
    first = canonicalize_physical_interval_union(
        list(left), tolerance_seconds=tolerance
    )
    second = canonicalize_physical_interval_union(
        list(right), tolerance_seconds=tolerance
    )
    return len(first) == len(second) and all(
        abs(a["start"] - b["start"]) <= tolerance
        and abs(a["stop"] - b["stop"]) <= tolerance
        for a, b in zip(first, second)
    )


def _term_ref(value: object, context: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    for key in ("term_id", "ontology_id", "operational_rule_id"):
        if key not in value:
            raise ValueError(f"{context}.{key} is required")
    return {
        "term_id": _id(value["term_id"], f"{context}.term_id"),
        "ontology_id": _id(value["ontology_id"], f"{context}.ontology_id"),
        "operational_rule_id": _id(
            value["operational_rule_id"],
            f"{context}.operational_rule_id",
        ),
    }


def _validated_v3(
    value: object,
    validation_kwargs: Mapping[str, object] | None,
) -> dict[str, Any]:
    kwargs = dict(validation_kwargs or {})
    unexpected = sorted(set(kwargs) - _V3_VALIDATION_KWARGS)
    if unexpected:
        raise ValueError(
            "unsupported v3 validation kwargs: " + ", ".join(unexpected)
        )
    return validate_event_eeg_findings_v3_payload(value, **kwargs)


def validate_event_finding_binding_policy(
    value: object,
    *,
    trusted_policy_sha256: str | None = None,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("event Finding binding policy must be an object")
    candidate = deepcopy(value)
    errors = _schema_errors(candidate, EVENT_FINDING_BINDING_POLICY_SCHEMA_PATH)
    if errors:
        raise ValueError(
            "event Finding binding policy schema validation failed: "
            + "; ".join(errors)
        )
    expected = _self_hash(candidate, "policy_sha256")
    if candidate["policy_sha256"] != expected:
        raise ValueError("event Finding binding policy_sha256 mismatch")
    anchor = trusted_policy_sha256 or DEFAULT_EVENT_FINDING_BINDING_POLICY_SHA256
    if expected != anchor:
        raise ValueError("event Finding binding policy is not host trusted")
    if candidate["policy_id"] != EVENT_FINDING_BINDING_POLICY_ID:
        raise ValueError("event Finding binding policy_id mismatch")
    if set(candidate["secondary_relation_types"]) != _SECONDARY_RELATIONS:
        raise ValueError("secondary relation vocabulary drifted")
    return candidate


def load_event_finding_binding_policy(
    path: str | Path = DEFAULT_EVENT_FINDING_BINDING_POLICY_PATH,
    *,
    trusted_policy_sha256: str | None = None,
) -> dict[str, Any]:
    return validate_event_finding_binding_policy(
        _read_json(Path(path)),
        trusted_policy_sha256=trusted_policy_sha256,
    )


def _policy(
    value: Mapping[str, object] | None,
    trusted_policy_sha256: str | None,
) -> dict[str, Any]:
    if value is None:
        return load_event_finding_binding_policy(
            trusted_policy_sha256=trusted_policy_sha256
        )
    return validate_event_finding_binding_policy(
        dict(value), trusted_policy_sha256=trusted_policy_sha256
    )


def _normalize_query_cell(
    raw: object,
    *,
    tolerance: float,
    term_manifest_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError("term-query denominator cell must be an object")
    row = dict(raw)
    key_name = "query_cell_key" if "query_cell_key" in row else "cell_key"
    cell_key = _id(row.get(key_name), "query cell key")
    query_id = _id(row.get("term_query_id"), f"{cell_key}.term_query_id")
    term_value: object
    if "term" in row:
        term_value = row["term"]
    else:
        term_value = {
            "term_id": row.get("term_id"),
            # The public v2 denominator intentionally carries a frozen term
            # manifest and query rule, not a payload-local ontology object.
            # Preserve that distinction in the ledger's compact term ref.
            "ontology_id": row.get("ontology_id", term_manifest_id),
            "operational_rule_id": row.get(
                "operational_rule_id", row.get("term_query_id")
            ),
        }
    term = _term_ref(term_value, f"{cell_key}.term")
    family = _id(row.get("family"), f"{cell_key}.family")
    unit = _typed_unit(row.get("unit"), f"{cell_key}.unit")
    granularity = str(
        row.get(
            "granularity",
            "event" if unit["unit_type"] == "event" else "unit",
        )
    )
    if granularity not in {"event", "unit"}:
        raise ValueError(f"{cell_key}.granularity is unsupported")
    if (granularity == "event") != (unit["unit_type"] == "event"):
        raise ValueError(f"{cell_key} granularity/unit mismatch")
    required = _interval_union(
        row.get("required_interval_union", []),
        context=f"{cell_key}.required_interval_union",
        tolerance=tolerance,
    )
    evaluable = _interval_union(
        row.get("evaluable_interval_union", []),
        context=f"{cell_key}.evaluable_interval_union",
        tolerance=tolerance,
    )
    if not _union_subset(evaluable, required, tolerance=tolerance):
        raise ValueError(f"{cell_key} evaluable union leaves required union")
    status = str(row.get("opportunity_status"))
    if status not in {"sufficient", "limited", "not_evaluable"}:
        raise ValueError(f"{cell_key}.opportunity_status is unsupported")
    negative_opportunity_eligible = row.get("negative_opportunity_eligible")
    if type(negative_opportunity_eligible) is not bool:
        raise TypeError(
            f"{cell_key}.negative_opportunity_eligible must be boolean"
        )
    failure_ids = sorted(
        _id(value, f"{cell_key}.technical_failure_receipt_ids")
        for value in row.get("technical_failure_receipt_ids", [])
    )
    scope = row.get("qualification_scope")
    if not isinstance(scope, Mapping):
        scope = {
            key: row.get(key)
            for key in (
                "view_profile_id",
                "reference_profile_id",
                "bandwidth_profile_id",
            )
            if row.get(key) is not None
        }
    query_cell_sha = row.get("query_cell_sha256", row.get("cell_sha256"))
    _sha(query_cell_sha, f"{cell_key}.query_cell_sha256")
    return {
        "query_cell_key": cell_key,
        "query_cell_sha256": str(query_cell_sha),
        "term_query_id": query_id,
        "term": term,
        "family": family,
        "claim_kind": str(row.get("claim_kind", "legacy_unspecified")),
        "temporal_context": str(
            row.get("temporal_context", "legacy_unspecified")
        ),
        "intrinsic_evidence_role": str(
            row.get("intrinsic_evidence_role", "legacy_unspecified")
        ),
        "unit": unit,
        "granularity": granularity,
        "scope_id": _id(row.get("scope_id"), f"{cell_key}.scope_id"),
        "view_profile_id": row.get("view_profile_id"),
        "reference_profile_id": row.get("reference_profile_id"),
        "bandwidth_profile_id": row.get("bandwidth_profile_id"),
        "required_interval_union": required,
        "evaluable_interval_union": evaluable,
        "opportunity_status": status,
        "processing_disposition": str(
            row.get("processing_disposition", "completed")
        ),
        "implementation_status": str(
            row.get("implementation_status", "legacy_unspecified")
        ),
        "negative_opportunity_eligible": bool(
            negative_opportunity_eligible
        ),
        "technical_failure_receipt_ids": failure_ids,
        "positive_capability_receipt_ids": sorted(
            _id(value, f"{cell_key}.positive_capability_receipt_ids")
            for value in row.get(
                "positive_capability_receipt_ids",
                row.get("capability_receipt_ids", []),
            )
        ),
        "negative_sensitivity_receipt_ids": sorted(
            _id(value, f"{cell_key}.negative_sensitivity_receipt_ids")
            for value in row.get(
                "negative_sensitivity_receipt_ids",
                row.get("sensitivity_receipt_ids", []),
            )
        ),
        "primary_roster_item_kind": row.get("primary_roster_item_kind"),
        "primary_roster_item_id": row.get("primary_roster_item_id"),
        "primary_v1_cell_key": row.get("primary_v1_cell_key"),
        "qualification_scope": deepcopy(dict(scope)),
    }


def _fallback_validate_term_query_denominator(
    value: object,
    *,
    trusted_receipt_sha256: str,
    tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    """Strict temporary Mapping contract for the parallel v2 denominator.

    When the dedicated module is present, the public validator is invoked
    first by :func:`_validate_term_query_denominator`.  This fallback still
    requires a host-pinned receipt hash, content self-hash, exact typed-unit
    inventory and unique query cells; it never trusts a Findings payload to
    enumerate its own denominator.
    """

    if type(value) is not dict:
        raise TypeError("term-query denominator receipt must be an object")
    candidate = deepcopy(value)
    receipt_sha = _sha(
        candidate.get("receipt_sha256"),
        "term-query denominator receipt_sha256",
    )
    if receipt_sha != trusted_receipt_sha256:
        raise ValueError("term-query denominator receipt is not host trusted")
    if receipt_sha != _self_hash(candidate, "receipt_sha256"):
        raise ValueError("term-query denominator receipt_sha256 mismatch")
    units = _typed_unit_inventory(
        candidate.get("typed_expected_units"),
        "term-query denominator typed_expected_units",
    )
    expected_inventory_sha = _canonical_sha256(units)
    if candidate.get("typed_unit_inventory_sha256", candidate.get("expected_unit_inventory_sha256")) != expected_inventory_sha:
        raise ValueError("term-query denominator typed-unit inventory hash drifted")
    raw_cells = candidate.get("query_cells", candidate.get("cells"))
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError("term-query denominator must contain query cells")
    cells = [
        _normalize_query_cell(
            row,
            tolerance=tolerance,
            term_manifest_id=str(candidate.get("term_manifest_id", "LEGACY")),
        )
        for row in raw_cells
    ]
    keys = [row["query_cell_key"] for row in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("term-query denominator contains duplicate query cells")
    return candidate, sorted(cells, key=lambda row: row["query_cell_key"]), units


def _term_query_validator() -> Callable[..., object] | None:
    try:
        from . import event_findings_term_query_denominator_v2 as module
    except ImportError:
        return None
    for name in (
        "validate_event_findings_term_query_denominator_receipt_v2",
        "validate_event_findings_term_query_denominator_receipt",
        "validate_event_findings_term_query_denominator_v2_receipt",
        "validate_term_query_denominator_receipt",
    ):
        value = getattr(module, name, None)
        if callable(value):
            return value
    return None


def _validate_term_query_denominator(
    value: object,
    *,
    trusted_receipt_sha256: str,
    validation_kwargs: Mapping[str, object] | None,
    tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    validator = _term_query_validator()
    if validator is not None:
        kwargs = dict(validation_kwargs or {})
        signature = inspect.signature(validator)
        if "trusted_receipt_sha256" in signature.parameters:
            kwargs.setdefault("trusted_receipt_sha256", trusted_receipt_sha256)
        elif "trusted_term_query_denominator_receipt_sha256" in signature.parameters:
            kwargs.setdefault(
                "trusted_term_query_denominator_receipt_sha256",
                trusted_receipt_sha256,
            )
        validated = validator(value, **kwargs)
        # Normalize the public receipt after its module has performed exact
        # host replay.  A second pinned-hash check prevents a wrong receipt
        # from being normalized through an otherwise valid API call.
        if not isinstance(validated, Mapping):
            raise TypeError("term-query denominator validator returned a non-object")
        candidate = deepcopy(dict(validated))
        if candidate.get("receipt_sha256") != trusted_receipt_sha256:
            raise ValueError("validated term-query denominator is not the pinned receipt")
        units = _typed_unit_inventory(
            candidate.get("typed_expected_units"),
            "term-query denominator typed_expected_units",
        )
        raw_cells = candidate.get("query_cells", candidate.get("cells"))
        if not isinstance(raw_cells, list) or not raw_cells:
            raise ValueError("validated term-query denominator has no cells")
        cells = [
            _normalize_query_cell(
                row,
                tolerance=tolerance,
                term_manifest_id=str(
                    candidate.get("term_manifest_id", "LEGACY")
                ),
            )
            for row in raw_cells
        ]
        keys = [row["query_cell_key"] for row in cells]
        if len(keys) != len(set(keys)):
            raise ValueError("validated term-query denominator has duplicate cells")
        return candidate, sorted(cells, key=lambda row: row["query_cell_key"]), units
    if validation_kwargs:
        raise ValueError(
            "term-query denominator validation kwargs were supplied but its public validator is unavailable"
        )
    return _fallback_validate_term_query_denominator(
        value,
        trusted_receipt_sha256=trusted_receipt_sha256,
        tolerance=tolerance,
    )


def _finding_sha256(finding: Mapping[str, object]) -> str:
    return _canonical_sha256(dict(finding))


def _finding_interval_union(
    finding: Mapping[str, object], *, tolerance: float
) -> list[dict[str, float]]:
    interval = finding.get("time_interval")
    if not isinstance(interval, Mapping):
        return []
    return _interval_union(
        [
            {
                "start": interval["start"],
                "stop": interval["stop"],
            }
        ],
        context=f"finding {finding.get('evidence_id')} interval",
        tolerance=tolerance,
    )


def _finding_typed_support(finding: Mapping[str, object]) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []
    for index, raw in enumerate(finding.get("spatial_support", [])):
        if not isinstance(raw, Mapping):
            raise TypeError("Finding spatial support must contain objects")
        unit_type = str(raw.get("unit_type"))
        if unit_type not in {"lead", "electrode"}:
            continue
        unit_id = _id(raw.get("id"), f"spatial_support[{index}].id")
        units.append(
            {
                "unit_type": unit_type,
                "unit_id": unit_id,
                "unit_key": f"{unit_type}:{unit_id}",
            }
        )
    keys = [row["unit_key"] for row in units]
    if len(keys) != len(set(keys)):
        raise ValueError("Finding spatial support has duplicate typed units")
    return sorted(units, key=lambda row: row["unit_key"])


def _waveform_map(payload: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {
        str(row["waveform_evidence_id"]): row
        for row in payload["waveform_evidence"]  # type: ignore[index]
    }


def _finding_semantic_fingerprint(
    payload: Mapping[str, object], finding: Mapping[str, object]
) -> str:
    """Return an ID-invariant semantic Finding fingerprint.

    Local evidence, measurement, waveform and dependency IDs are deliberately
    excluded.  Their signal-bound content is retained so changing only an ID
    cannot create another vote, while a genuinely different reference/view or
    physical support remains distinguishable and may be represented as a
    typed secondary relation.
    """

    waveforms = _waveform_map(payload)
    measurements: list[dict[str, Any]] = []
    for raw in finding["measurements"]:  # type: ignore[index]
        measurement = dict(raw)
        binding = dict(measurement["source_binding"])
        dependency = binding.get("raw_sample_dependency")
        measurements.append(
            {
                "name_id": measurement["name_id"],
                "value": measurement["value"],
                "unit_id": measurement["unit_id"],
                "baseline_delta": measurement["baseline_delta"],
                "view_role": binding["view_role"],
                "source_view_id": binding["source_view_id"],
                "source_unit_ids": sorted(binding["source_unit_ids"]),
                "recording_interval": binding["recording_interval"],
                "effective_bandwidth_hz": binding["effective_bandwidth_hz"],
                "reference_type": binding["reference_type"],
                "evidence_family": binding["evidence_family"],
                "quality_mask_sha256": binding["quality_mask_sha256"],
                "raw_dependency_sha256": (
                    None
                    if not isinstance(dependency, Mapping)
                    else dependency["dependency_sha256"]
                ),
            }
        )
    waveform_rows: list[dict[str, Any]] = []
    for waveform_id in finding["waveform_evidence_ids"]:  # type: ignore[index]
        waveform = waveforms[str(waveform_id)]
        dependency = waveform.get("raw_sample_dependency")
        waveform_rows.append(
            {
                "interval": waveform["interval"],
                "unit_ids": sorted(waveform["unit_ids"]),
                "source_view_id": waveform["source_view_id"],
                "view_role": waveform["view_role"],
                "processed_view_sha256": waveform["processed_view_sha256"],
                "quality_mask_sha256": waveform["quality_mask_sha256"],
                "raw_dependency_sha256": (
                    None
                    if not isinstance(dependency, Mapping)
                    else dependency["dependency_sha256"]
                ),
            }
        )
    spatial = []
    for raw in finding["spatial_support"]:  # type: ignore[index]
        row = dict(raw)
        spatial.append(
            {
                "unit_type": row["unit_type"],
                "id": row["id"],
                "mapping_status": row["mapping_status"],
                "observation_status": row["observation_status"],
                "evidence_eligible": row["evidence_eligible"],
                "field_observation": row["field_observation"],
            }
        )
    body = {
        "family": finding["family"],
        "term": finding["term"],
        "assertion_level": finding["assertion_level"],
        "status": finding["status"],
        "intrinsic_evidence_role": finding["intrinsic_evidence_role"],
        "signal_temporal_context": finding["signal_temporal_context"],
        "time_interval": finding["time_interval"],
        "spatial_support": sorted(
            spatial,
            key=lambda row: (str(row["unit_type"]), str(row["id"])),
        ),
        "measurements": sorted(
            measurements,
            key=lambda row: (
                str(row["name_id"]),
                _canonical_json(row),
            ),
        ),
        "waveforms": sorted(waveform_rows, key=_canonical_json),
    }
    return _canonical_sha256(
        {"binding_domain": "event-finding-semantic-fingerprint-v1", **body}
    )


def _pattern_semantic_fingerprint(
    payload: Mapping[str, object], pattern: Mapping[str, object]
) -> str:
    finding_map = {
        str(row["evidence_id"]): row
        for row in payload["findings"]  # type: ignore[index]
    }
    required = sorted(
        _finding_semantic_fingerprint(payload, finding_map[str(value)])
        for value in pattern["required_atom_ids"]  # type: ignore[index]
    )
    counter = sorted(
        _finding_semantic_fingerprint(payload, finding_map[str(value)])
        for value in pattern["counterevidence_ids"]  # type: ignore[index]
    )
    return _canonical_sha256(
        {
            "binding_domain": "event-pattern-semantic-candidate-v1",
            "term": pattern["term"],
            "assertion_level": pattern["assertion_level"],
            "status": pattern["status"],
            "source_domain_scope": pattern["source_domain_scope"],
            "required_atom_semantics": required,
            "counterevidence_semantics": counter,
        }
    )


def build_event_finding_physical_instance_inventory(
    event_findings_v3: object,
    *,
    typed_expected_units: Sequence[Mapping[str, object]],
    physical_instances: Sequence[Mapping[str, object]],
    finding_instance_keys: Mapping[str, str | None],
    pattern_candidate_instance_keys: Mapping[str, str] | None = None,
    findings_validation_kwargs: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build a host-pinnable physical-instance inventory.

    The caller supplies candidate-blind physical instances.  Source-local
    ``pattern_instance_id`` values are never used as physical identity.
    """

    source = _validated_v3(event_findings_v3, findings_validation_kwargs)
    units = _typed_unit_inventory(
        list(typed_expected_units), "physical inventory typed_expected_units"
    )
    allowed_unit_keys = {row["unit_key"] for row in units}
    instances: list[dict[str, Any]] = []
    for index, raw in enumerate(physical_instances):
        if not isinstance(raw, Mapping):
            raise TypeError(f"physical_instances[{index}] must be an object")
        key = _id(raw.get("physical_instance_key"), f"physical_instances[{index}].key")
        interval_union = _interval_union(
            raw.get("interval_union", []),
            context=f"physical_instances[{index}].interval_union",
            tolerance=1e-9,
        )
        if not interval_union:
            raise ValueError("physical instances require non-empty physical support")
        instance_units = _typed_unit_inventory(
            raw.get("typed_units", []),
            f"physical_instances[{index}].typed_units",
        )
        if any(row["unit_key"] not in allowed_unit_keys for row in instance_units):
            raise ValueError("physical instance leaves the trusted typed-unit inventory")
        dependencies = sorted(
            _id(value, f"physical_instances[{index}].raw_sample_dependency_ids")
            for value in raw.get("raw_sample_dependency_ids", [])
        )
        source_receipts = sorted(
            _id(value, f"physical_instances[{index}].source_receipt_ids")
            for value in raw.get("source_receipt_ids", [])
        )
        body: dict[str, Any] = {
            "physical_instance_key": key,
            "interval_union": interval_union,
            "typed_units": instance_units,
            "raw_sample_dependency_ids": dependencies,
            "source_receipt_ids": source_receipts,
            "instance_sha256": "0" * 64,
        }
        body["instance_sha256"] = _self_hash(body, "instance_sha256")
        instances.append(body)
    instances.sort(key=lambda row: row["physical_instance_key"])
    instance_keys = _unique(
        (str(row["physical_instance_key"]) for row in instances),
        "physical instance inventory",
    )

    finding_map = {
        str(row["evidence_id"]): row for row in source["findings"]
    }
    supplied_finding_ids = set(str(value) for value in finding_instance_keys)
    if supplied_finding_ids != set(finding_map):
        raise ValueError("physical inventory must assign every Finding exactly once")
    assignments: list[dict[str, Any]] = []
    for finding_id in sorted(finding_map):
        finding = finding_map[finding_id]
        instance_key = finding_instance_keys[finding_id]
        if instance_key is not None:
            instance_key = _id(instance_key, f"finding {finding_id} physical key")
            if instance_key not in instance_keys:
                raise ValueError(f"finding {finding_id} references an unknown physical instance")
            disposition = "assigned"
            reasons: list[str] = []
        else:
            disposition = "not_applicable"
            reasons = ["no_positive_physical_occurrence"]
        row = {
            "finding_id": finding_id,
            "finding_sha256": _finding_sha256(finding),
            "physical_instance_key": instance_key,
            "disposition": disposition,
            "reason_codes": reasons,
        }
        assignments.append(row)

    pattern_map = {
        str(row["pattern_candidate_id"]): row
        for row in source["pattern_candidates"]
    }
    supplied_pattern_map = dict(pattern_candidate_instance_keys or {})
    if set(supplied_pattern_map) != set(pattern_map):
        raise ValueError(
            "physical inventory must assign every pattern candidate exactly once"
        )
    pattern_assignments: list[dict[str, Any]] = []
    for candidate_id in sorted(pattern_map):
        physical_key = _id(
            supplied_pattern_map[candidate_id],
            f"pattern candidate {candidate_id} physical key",
        )
        if physical_key not in instance_keys:
            raise ValueError("pattern candidate references an unknown physical instance")
        pattern = pattern_map[candidate_id]
        pattern_assignments.append(
            {
                "pattern_candidate_id": candidate_id,
                "pattern_candidate_sha256": _canonical_sha256(pattern),
                "source_pattern_instance_alias": str(pattern["pattern_instance_id"]),
                "physical_instance_key": physical_key,
                "semantic_candidate_sha256": _pattern_semantic_fingerprint(
                    source, pattern
                ),
            }
        )

    identity = {
        "event_id": source["event_id"],
        "record_id": source["provenance"]["record_id"],
        "canonical_signal_sha256": source["provenance"][
            "canonical_signal_sha256"
        ],
        "typed_unit_inventory_sha256": _canonical_sha256(units),
        "findings_payload_sha256": _canonical_sha256(source),
    }
    seed = {
        **identity,
        "instances": [row["instance_sha256"] for row in instances],
        "assignments": assignments,
        "pattern_assignments": pattern_assignments,
    }
    inventory: dict[str, Any] = {
        "schema_version": EVENT_FINDING_PHYSICAL_INSTANCE_INVENTORY_SCHEMA_VERSION,
        "inventory_id": f"PHYINST-{_canonical_sha256(seed)[:24]}",
        **identity,
        "typed_expected_units": units,
        "physical_instances": instances,
        "finding_assignments": assignments,
        "pattern_candidate_assignments": pattern_assignments,
        "source_firewall": deepcopy(_SOURCE_FIREWALL),
        "inventory_sha256": "0" * 64,
    }
    inventory["inventory_sha256"] = _self_hash(inventory, "inventory_sha256")
    return validate_event_finding_physical_instance_inventory(
        inventory,
        event_findings_v3=source,
        trusted_inventory_sha256=str(inventory["inventory_sha256"]),
        findings_validation_kwargs=findings_validation_kwargs,
    )


def validate_event_finding_physical_instance_inventory(
    value: object,
    *,
    event_findings_v3: object,
    trusted_inventory_sha256: str,
    findings_validation_kwargs: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("physical-instance inventory must be an object")
    candidate = deepcopy(value)
    source = _validated_v3(event_findings_v3, findings_validation_kwargs)
    inventory_sha = _sha(candidate.get("inventory_sha256"), "physical inventory hash")
    if inventory_sha != trusted_inventory_sha256:
        raise ValueError("physical-instance inventory is not host trusted")
    if inventory_sha != _self_hash(candidate, "inventory_sha256"):
        raise ValueError("physical-instance inventory hash mismatch")
    expected_identity = {
        "event_id": source["event_id"],
        "record_id": source["provenance"]["record_id"],
        "canonical_signal_sha256": source["provenance"]["canonical_signal_sha256"],
        "findings_payload_sha256": _canonical_sha256(source),
    }
    for key, expected in expected_identity.items():
        if candidate.get(key) != expected:
            raise ValueError(f"physical-instance inventory {key} mismatch")
    if candidate.get("source_firewall") != _SOURCE_FIREWALL:
        raise ValueError("physical-instance inventory EEG-only firewall drifted")
    units = _typed_unit_inventory(
        candidate.get("typed_expected_units"),
        "physical inventory typed_expected_units",
    )
    if candidate.get("typed_unit_inventory_sha256") != _canonical_sha256(units):
        raise ValueError("physical-instance typed-unit inventory hash mismatch")
    unit_keys = {row["unit_key"] for row in units}

    instances = candidate.get("physical_instances")
    if not isinstance(instances, list):
        raise TypeError("physical_instances must be an array")
    instance_map: dict[str, Mapping[str, object]] = {}
    for index, raw in enumerate(instances):
        if not isinstance(raw, Mapping):
            raise TypeError(f"physical_instances[{index}] must be an object")
        key = _id(raw.get("physical_instance_key"), "physical instance key")
        if key in instance_map:
            raise ValueError("physical-instance inventory has duplicate keys")
        if raw.get("instance_sha256") != _self_hash(raw, "instance_sha256"):
            raise ValueError(f"physical instance {key} hash mismatch")
        interval_union = _interval_union(
            raw.get("interval_union", []),
            context=f"physical instance {key} interval_union",
            tolerance=1e-9,
        )
        if interval_union != raw.get("interval_union") or not interval_union:
            raise ValueError(f"physical instance {key} union is not canonical")
        typed = _typed_unit_inventory(
            raw.get("typed_units", []), f"physical instance {key} units"
        )
        if typed != raw.get("typed_units") or any(
            row["unit_key"] not in unit_keys for row in typed
        ):
            raise ValueError(f"physical instance {key} unit inventory drifted")
        instance_map[key] = raw

    finding_map = {
        str(row["evidence_id"]): row for row in source["findings"]
    }
    assignments = candidate.get("finding_assignments")
    if not isinstance(assignments, list):
        raise TypeError("finding_assignments must be an array")
    assignment_ids = _unique(
        (str(row["finding_id"]) for row in assignments),
        "physical finding assignments",
    )
    if assignment_ids != set(finding_map):
        raise ValueError("physical inventory Finding coverage is not exact")
    for assignment in assignments:
        finding_id = str(assignment["finding_id"])
        finding = finding_map[finding_id]
        if assignment["finding_sha256"] != _finding_sha256(finding):
            raise ValueError(f"physical assignment {finding_id} source hash drifted")
        physical_key = assignment["physical_instance_key"]
        status = str(finding["status"])
        if physical_key is None:
            if status in {"present", "uncertain"} and finding["time_interval"] is not None:
                raise ValueError(
                    f"positive/candidate Finding {finding_id} lacks a physical instance"
                )
            if assignment["disposition"] != "not_applicable":
                raise ValueError("null physical assignment must be not_applicable")
            continue
        physical_key = str(physical_key)
        if physical_key not in instance_map or assignment["disposition"] != "assigned":
            raise ValueError(f"physical assignment {finding_id} is invalid")
        instance = instance_map[physical_key]
        finding_union = _finding_interval_union(finding, tolerance=1e-9)
        if finding_union and not _union_subset(
            finding_union,
            instance["interval_union"],  # type: ignore[arg-type]
            tolerance=1e-9,
        ):
            raise ValueError(f"Finding {finding_id} leaves its physical instance")
        finding_units = {
            row["unit_key"] for row in _finding_typed_support(finding)
        }
        instance_units = {
            str(row["unit_key"]) for row in instance["typed_units"]  # type: ignore[index]
        }
        if not finding_units.issubset(instance_units):
            raise ValueError(f"Finding {finding_id} leaves physical-instance units")
        instance_dependencies = set(
            str(value) for value in instance["raw_sample_dependency_ids"]  # type: ignore[index]
        )
        if not set(str(value) for value in finding["raw_sample_dependency_ids"]).issubset(
            instance_dependencies
        ):
            raise ValueError(f"Finding {finding_id} leaves physical raw support")

    pattern_map = {
        str(row["pattern_candidate_id"]): row
        for row in source["pattern_candidates"]
    }
    pattern_assignments = candidate.get("pattern_candidate_assignments")
    if not isinstance(pattern_assignments, list):
        raise TypeError("pattern_candidate_assignments must be an array")
    pattern_ids = _unique(
        (str(row["pattern_candidate_id"]) for row in pattern_assignments),
        "physical pattern assignments",
    )
    if pattern_ids != set(pattern_map):
        raise ValueError("physical inventory pattern coverage is not exact")
    semantic_keys: set[tuple[str, str]] = set()
    for assignment in pattern_assignments:
        candidate_id = str(assignment["pattern_candidate_id"])
        pattern = pattern_map[candidate_id]
        physical_key = str(assignment["physical_instance_key"])
        if physical_key not in instance_map:
            raise ValueError("pattern assignment uses an unknown physical instance")
        expected_semantic = _pattern_semantic_fingerprint(source, pattern)
        if (
            assignment["pattern_candidate_sha256"] != _canonical_sha256(pattern)
            or assignment["semantic_candidate_sha256"] != expected_semantic
            or assignment["source_pattern_instance_alias"]
            != pattern["pattern_instance_id"]
        ):
            raise ValueError("pattern physical assignment source binding drifted")
        semantic_key = (physical_key, expected_semantic)
        if semantic_key in semantic_keys:
            raise ValueError(
                "duplicate semantic pattern candidate cannot create another physical vote"
            )
        semantic_keys.add(semantic_key)
    return candidate


def _sorted_ids(value: object, context: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be an array")
    rows = [_id(item, context) for item in value]
    if len(rows) != len(set(rows)):
        raise ValueError(f"{context} contains duplicate IDs")
    return sorted(rows)


def _nullable_id(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _id(value, context)


def _closed_bandwidth(
    value: object, context: str
) -> list[float] | None:
    if value is None:
        return None
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise TypeError(f"{context} must be null or [low, high]")
    low = _finite(value[0], f"{context}[0]")
    high = _finite(value[1], f"{context}[1]")
    if low < 0.0 or high <= low:
        raise ValueError(f"{context} must be an increasing nonnegative band")
    return [low, high]


_DOMAIN_SCOPE_KEYS = {
    "producer_id",
    "target_domain_id",
    "term_id",
    "operational_rule_id",
    "family",
    "unit",
    "granularity",
    "view_profile_id",
    "reference_profile_id",
    "bandwidth_profile_id",
    "view_roles",
    "reference_types",
    "sample_rate_hz",
    "effective_bandwidth_hz",
    "usable_fraction",
    "opportunity_policy_sha256",
}


def _domain_scope(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    if set(value) != _DOMAIN_SCOPE_KEYS:
        missing = sorted(_DOMAIN_SCOPE_KEYS - set(value))
        extra = sorted(set(value) - _DOMAIN_SCOPE_KEYS)
        raise ValueError(
            f"{context} keys drifted; missing={missing}, extra={extra}"
        )
    unit = _typed_unit(value["unit"], f"{context}.unit")
    granularity = _id(value["granularity"], f"{context}.granularity")
    if granularity not in {"event", "unit"}:
        raise ValueError(f"{context}.granularity is unsupported")
    if (granularity == "event") != (unit["unit_type"] == "event"):
        raise ValueError(f"{context} has a granularity/unit mismatch")
    sample_rate = _finite(value["sample_rate_hz"], f"{context}.sample_rate_hz")
    usable = _finite(value["usable_fraction"], f"{context}.usable_fraction")
    if sample_rate <= 0.0 or not 0.0 <= usable <= 1.0:
        raise ValueError(f"{context} has an invalid operating point")
    return {
        "producer_id": _id(value["producer_id"], f"{context}.producer_id"),
        "target_domain_id": _id(
            value["target_domain_id"], f"{context}.target_domain_id"
        ),
        "term_id": _id(value["term_id"], f"{context}.term_id"),
        "operational_rule_id": _id(
            value["operational_rule_id"],
            f"{context}.operational_rule_id",
        ),
        "family": _id(value["family"], f"{context}.family"),
        "unit": unit,
        "granularity": granularity,
        "view_profile_id": _id(
            value["view_profile_id"], f"{context}.view_profile_id"
        ),
        "reference_profile_id": _id(
            value["reference_profile_id"],
            f"{context}.reference_profile_id",
        ),
        "bandwidth_profile_id": _id(
            value["bandwidth_profile_id"],
            f"{context}.bandwidth_profile_id",
        ),
        "view_roles": _sorted_ids(value["view_roles"], f"{context}.view_roles"),
        "reference_types": _sorted_ids(
            value["reference_types"], f"{context}.reference_types"
        ),
        "sample_rate_hz": sample_rate,
        "effective_bandwidth_hz": _closed_bandwidth(
            value["effective_bandwidth_hz"],
            f"{context}.effective_bandwidth_hz",
        ),
        "usable_fraction": usable,
        "opportunity_policy_sha256": _sha(
            value["opportunity_policy_sha256"],
            f"{context}.opportunity_policy_sha256",
        ),
    }


def _receipt_map(
    source: Mapping[str, object], key: str, id_key: str = "receipt_id"
) -> dict[str, Mapping[str, object]]:
    rows = source.get(key, [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError(f"{key} must be an array")
    result = {str(row[id_key]): row for row in rows}  # type: ignore[index]
    if len(result) != len(rows):
        raise ValueError(f"{key} contains duplicate receipt IDs")
    return result


def _finding_actual_domain(
    source: Mapping[str, object], finding: Mapping[str, object]
) -> dict[str, Any]:
    waveform_map = _waveform_map(source)
    view_roles: set[str] = set()
    reference_types: set[str] = set()
    for waveform_id in finding.get("waveform_evidence_ids", []):
        waveform = waveform_map[str(waveform_id)]
        view_roles.add(str(waveform["view_role"]))
    for measurement in finding.get("measurements", []):
        binding = measurement["source_binding"]  # type: ignore[index]
        view_roles.add(str(binding["view_role"]))
        reference_types.add(str(binding["reference_type"]))
    if finding.get("spatial_support") or finding.get("waveform_evidence_ids"):
        reference_types.add(str(source["montage"]["analysis_reference"]))  # type: ignore[index]
    opportunities = {
        str(row["evaluation_opportunity_id"]): row
        for row in source["evaluation_opportunities"]  # type: ignore[index]
    }
    opportunity = opportunities[str(finding["evaluation_opportunity_id"])]
    return {
        "view_roles": sorted(view_roles),
        "reference_types": sorted(reference_types),
        "sample_rate_hz": float(
            source["coordinates"]["model_sample_rate_hz"]  # type: ignore[index]
        ),
        "effective_bandwidth_hz": (
            None
            if opportunity["effective_bandwidth_hz"] is None
            else [float(value) for value in opportunity["effective_bandwidth_hz"]]
        ),
        "usable_fraction": float(opportunity["usable_fraction"]),
    }


def _onset_future_free_causal(
    source: Mapping[str, object], finding: Mapping[str, object]
) -> bool:
    if not (
        finding.get("status") == "present"
        and finding.get("intrinsic_evidence_role") == "onset_eligible"
    ):
        return True
    dependencies: dict[str, Mapping[str, object]] = {}
    for waveform in source["waveform_evidence"]:  # type: ignore[index]
        dependency = waveform.get("raw_sample_dependency")
        if isinstance(dependency, Mapping):
            dependencies[str(dependency["dependency_id"])] = dependency
    for candidate in source["findings"]:  # type: ignore[index]
        for measurement in candidate["measurements"]:
            dependency = measurement["source_binding"].get(
                "raw_sample_dependency"
            )
            if isinstance(dependency, Mapping):
                dependencies[str(dependency["dependency_id"])] = dependency
    ids = [str(value) for value in finding.get("raw_sample_dependency_ids", [])]
    if not ids or any(value not in dependencies for value in ids):
        return False
    for dependency_id in ids:
        dependency = dependencies[dependency_id]
        if (
            dependency.get("future_sample_access") is not False
            or dependency.get("view_role") != "onset_causal"
            or dependency.get("onset_evidence_authorized") is not True
            or dependency.get("onset_support_eligible") is not True
        ):
            return False
    return True


def _secondary_links(value: object, context: str) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be an array")
    result: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise TypeError(f"{context}[{index}] must be an object")
        if set(raw) != {"query_cell_key", "relation_type"}:
            raise ValueError(f"{context}[{index}] keys drifted")
        key = _id(raw["query_cell_key"], f"{context}[{index}].query_cell_key")
        relation = str(raw["relation_type"])
        if relation not in _SECONDARY_RELATIONS:
            raise ValueError(f"{context}[{index}] relation is unsupported")
        pair = (key, relation)
        if pair in seen:
            raise ValueError(f"{context} contains a duplicate link")
        seen.add(pair)
        result.append(
            {
                "query_cell_key": key,
                "relation_type": relation,
                "vote_count": 0,
                "may_upgrade_status": False,
                "may_upgrade_assertion": False,
            }
        )
    return sorted(
        result,
        key=lambda row: (str(row["query_cell_key"]), str(row["relation_type"])),
    )


def build_event_finding_producer_binding_inventory(
    event_findings_v3: object,
    *,
    typed_expected_units: Sequence[Mapping[str, object]],
    producer_bindings: Sequence[Mapping[str, object]],
    findings_validation_kwargs: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build an exact-cover, host-pinnable producer/binding inventory."""

    source = _validated_v3(event_findings_v3, findings_validation_kwargs)
    units = _typed_unit_inventory(
        list(typed_expected_units), "producer inventory typed_expected_units"
    )
    finding_map = {str(row["evidence_id"]): row for row in source["findings"]}
    raw_map: dict[str, Mapping[str, object]] = {}
    for index, raw in enumerate(producer_bindings):
        if not isinstance(raw, Mapping):
            raise TypeError(f"producer_bindings[{index}] must be an object")
        finding_id = _id(raw.get("finding_id"), f"producer_bindings[{index}].finding_id")
        if finding_id in raw_map:
            raise ValueError("producer inventory contains multiple bindings for a Finding")
        raw_map[finding_id] = raw
    if set(raw_map) != set(finding_map):
        raise ValueError("producer inventory must bind every Finding exactly once")

    rows: list[dict[str, Any]] = []
    for finding_id in sorted(finding_map):
        raw = raw_map[finding_id]
        finding = finding_map[finding_id]
        scope = _domain_scope(raw.get("domain_scope"), f"binding {finding_id}.domain_scope")
        processing = str(raw.get("processing_disposition"))
        if processing not in {"completed", "technical_failure"}:
            raise ValueError(f"binding {finding_id} has an unsupported disposition")
        failure_ids = _sorted_ids(
            raw.get("technical_failure_receipt_ids", []),
            f"binding {finding_id}.technical_failure_receipt_ids",
        )
        if (processing == "technical_failure") != bool(failure_ids):
            raise ValueError("technical failure requires explicit failure receipts")
        primary = _nullable_id(
            raw.get("primary_query_cell_key"),
            f"binding {finding_id}.primary_query_cell_key",
        )
        physical = _nullable_id(
            raw.get("physical_instance_key"),
            f"binding {finding_id}.physical_instance_key",
        )
        producer_receipt_id = _nullable_id(
            raw.get("producer_receipt_id"),
            f"binding {finding_id}.producer_receipt_id",
        )
        body: dict[str, Any] = {
            "producer_binding_id": "PENDING",
            "finding_id": finding_id,
            "finding_sha256": _finding_sha256(finding),
            "semantic_fingerprint_sha256": _finding_semantic_fingerprint(
                source, finding
            ),
            "producer_id": scope["producer_id"],
            "producer_receipt_id": producer_receipt_id,
            "processing_disposition": processing,
            "primary_query_cell_key": primary,
            "secondary_links": _secondary_links(
                raw.get("secondary_links", []),
                f"binding {finding_id}.secondary_links",
            ),
            "physical_instance_key": physical,
            "domain_scope": scope,
            "finding_capability_receipt_id": finding["capability_receipt_id"],
            "finding_sensitivity_receipt_id": finding["sensitivity_receipt_id"],
            "finding_term_decision_receipt_id": finding[
                "term_decision_receipt_id"
            ],
            "query_capability_receipt_ids": _sorted_ids(
                raw.get("query_capability_receipt_ids", []),
                f"binding {finding_id}.query_capability_receipt_ids",
            ),
            "query_sensitivity_receipt_ids": _sorted_ids(
                raw.get("query_sensitivity_receipt_ids", []),
                f"binding {finding_id}.query_sensitivity_receipt_ids",
            ),
            "technical_failure_receipt_ids": failure_ids,
            "onset_future_free_causal": _onset_future_free_causal(
                source, finding
            ),
            "reason_codes": _sorted_ids(
                raw.get("reason_codes", []),
                f"binding {finding_id}.reason_codes",
            ),
            "binding_sha256": "0" * 64,
        }
        binding_seed = deepcopy(body)
        binding_seed.pop("producer_binding_id")
        binding_seed.pop("binding_sha256")
        body["producer_binding_id"] = (
            f"PRODBIND-{_canonical_sha256(binding_seed)[:24]}"
        )
        body["binding_sha256"] = _self_hash(body, "binding_sha256")
        rows.append(body)

    identity = {
        "event_id": source["event_id"],
        "record_id": source["provenance"]["record_id"],
        "canonical_signal_sha256": source["provenance"][
            "canonical_signal_sha256"
        ],
        "findings_payload_sha256": _canonical_sha256(source),
        "typed_unit_inventory_sha256": _canonical_sha256(units),
    }
    seed = {**identity, "binding_sha256s": [row["binding_sha256"] for row in rows]}
    inventory: dict[str, Any] = {
        "schema_version": EVENT_FINDING_PRODUCER_BINDING_INVENTORY_SCHEMA_VERSION,
        "inventory_id": f"PRODINV-{_canonical_sha256(seed)[:24]}",
        **identity,
        "typed_expected_units": units,
        "bindings": rows,
        "source_firewall": deepcopy(_SOURCE_FIREWALL),
        "inventory_sha256": "0" * 64,
    }
    inventory["inventory_sha256"] = _self_hash(inventory, "inventory_sha256")
    return validate_event_finding_producer_binding_inventory(
        inventory,
        event_findings_v3=source,
        trusted_inventory_sha256=str(inventory["inventory_sha256"]),
        findings_validation_kwargs=findings_validation_kwargs,
    )


def validate_event_finding_producer_binding_inventory(
    value: object,
    *,
    event_findings_v3: object,
    trusted_inventory_sha256: str,
    findings_validation_kwargs: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("producer-binding inventory must be an object")
    candidate = deepcopy(value)
    source = _validated_v3(event_findings_v3, findings_validation_kwargs)
    inventory_sha = _sha(
        candidate.get("inventory_sha256"), "producer inventory hash"
    )
    if inventory_sha != trusted_inventory_sha256:
        raise ValueError("producer-binding inventory is not host trusted")
    if inventory_sha != _self_hash(candidate, "inventory_sha256"):
        raise ValueError("producer-binding inventory hash mismatch")
    expected_top_keys = {
        "schema_version",
        "inventory_id",
        "event_id",
        "record_id",
        "canonical_signal_sha256",
        "findings_payload_sha256",
        "typed_unit_inventory_sha256",
        "typed_expected_units",
        "bindings",
        "source_firewall",
        "inventory_sha256",
    }
    if set(candidate) != expected_top_keys:
        raise ValueError("producer-binding inventory top-level keys drifted")
    if candidate["schema_version"] != EVENT_FINDING_PRODUCER_BINDING_INVENTORY_SCHEMA_VERSION:
        raise ValueError("producer-binding inventory schema version mismatch")
    expected_identity = {
        "event_id": source["event_id"],
        "record_id": source["provenance"]["record_id"],
        "canonical_signal_sha256": source["provenance"][
            "canonical_signal_sha256"
        ],
        "findings_payload_sha256": _canonical_sha256(source),
    }
    for key, expected in expected_identity.items():
        if candidate.get(key) != expected:
            raise ValueError(f"producer-binding inventory {key} mismatch")
    if candidate.get("source_firewall") != _SOURCE_FIREWALL:
        raise ValueError("producer-binding inventory EEG-only firewall drifted")
    units = _typed_unit_inventory(
        candidate.get("typed_expected_units"),
        "producer inventory typed_expected_units",
    )
    if candidate.get("typed_unit_inventory_sha256") != _canonical_sha256(units):
        raise ValueError("producer-binding typed-unit inventory hash mismatch")
    unit_keys = {row["unit_key"] for row in units} | {"event:GLOBAL"}

    finding_map = {str(row["evidence_id"]): row for row in source["findings"]}
    capabilities = _receipt_map(source, "capability_qualification_receipts")
    sensitivities = _receipt_map(source, "sensitivity_receipts")
    decisions = _receipt_map(source, "term_decision_receipts")
    producers = _receipt_map(source, "producer_receipts")
    rows = candidate.get("bindings")
    if not isinstance(rows, list):
        raise TypeError("producer inventory bindings must be an array")
    finding_ids = _unique(
        (str(row["finding_id"]) for row in rows),
        "producer inventory Finding bindings",
    )
    if finding_ids != set(finding_map):
        raise ValueError("producer inventory Finding coverage is not exact")
    binding_ids = _unique(
        (str(row["producer_binding_id"]) for row in rows),
        "producer inventory binding IDs",
    )
    if len(binding_ids) != len(rows):
        raise ValueError("producer inventory binding IDs are not unique")
    expected_row_keys = {
        "producer_binding_id",
        "finding_id",
        "finding_sha256",
        "semantic_fingerprint_sha256",
        "producer_id",
        "producer_receipt_id",
        "processing_disposition",
        "primary_query_cell_key",
        "secondary_links",
        "physical_instance_key",
        "domain_scope",
        "finding_capability_receipt_id",
        "finding_sensitivity_receipt_id",
        "finding_term_decision_receipt_id",
        "query_capability_receipt_ids",
        "query_sensitivity_receipt_ids",
        "technical_failure_receipt_ids",
        "onset_future_free_causal",
        "reason_codes",
        "binding_sha256",
    }
    for row in rows:
        if set(row) != expected_row_keys:
            raise ValueError("producer binding row keys drifted")
        finding_id = _id(row["finding_id"], "producer binding finding_id")
        finding = finding_map[finding_id]
        if (
            row["finding_sha256"] != _finding_sha256(finding)
            or row["semantic_fingerprint_sha256"]
            != _finding_semantic_fingerprint(source, finding)
            or row["binding_sha256"] != _self_hash(row, "binding_sha256")
        ):
            raise ValueError(f"producer binding {finding_id} source/hash drifted")
        scope = _domain_scope(row["domain_scope"], f"binding {finding_id}.domain_scope")
        if row["domain_scope"] != scope:
            raise ValueError(f"producer binding {finding_id} scope is not canonical")
        term = _term_ref(finding["term"], f"finding {finding_id}.term")
        if (
            scope["producer_id"] != row["producer_id"]
            or scope["term_id"] != term["term_id"]
            or scope["operational_rule_id"] != term["operational_rule_id"]
            or scope["family"] != finding["family"]
            or scope["unit"]["unit_key"] not in unit_keys
        ):
            raise ValueError(f"producer binding {finding_id} domain drifted")
        actual = _finding_actual_domain(source, finding)
        if any(scope[key] != actual[key] for key in actual):
            raise ValueError(
                f"producer binding {finding_id} launders its actual operating scope"
            )
        for key, source_key in (
            ("finding_capability_receipt_id", "capability_receipt_id"),
            ("finding_sensitivity_receipt_id", "sensitivity_receipt_id"),
            ("finding_term_decision_receipt_id", "term_decision_receipt_id"),
        ):
            if row[key] != finding[source_key]:
                raise ValueError(f"producer binding {finding_id} receipt binding drifted")
        producer_id = str(row["producer_id"])
        cap_id = row["finding_capability_receipt_id"]
        sens_id = row["finding_sensitivity_receipt_id"]
        decision_id = row["finding_term_decision_receipt_id"]
        for receipt_id, registry, label in (
            (cap_id, capabilities, "capability"),
            (sens_id, sensitivities, "sensitivity"),
            (decision_id, decisions, "term decision"),
        ):
            if receipt_id is not None:
                if str(receipt_id) not in registry:
                    raise ValueError(f"producer binding {finding_id} has unknown {label}")
                if registry[str(receipt_id)]["producer_id"] != producer_id:
                    raise ValueError(
                        f"producer binding {finding_id} has cross-producer {label}"
                    )
        producer_receipt_id = row["producer_receipt_id"]
        if producer_receipt_id is not None:
            if str(producer_receipt_id) not in producers:
                raise ValueError("producer binding references an unknown producer receipt")
            if producers[str(producer_receipt_id)]["producer_id"] != producer_id:
                raise ValueError("producer receipt ID launders another producer")
        processing = str(row["processing_disposition"])
        failure_ids = _sorted_ids(
            row["technical_failure_receipt_ids"],
            f"binding {finding_id}.technical_failure_receipt_ids",
        )
        if processing not in {"completed", "technical_failure"} or (
            (processing == "technical_failure") != bool(failure_ids)
        ):
            raise ValueError("producer processing/failure lineage is inconsistent")
        _nullable_id(row["primary_query_cell_key"], "primary query cell")
        _nullable_id(row["physical_instance_key"], "physical instance key")
        if row["secondary_links"] != _secondary_links(
            [
                {
                    "query_cell_key": link["query_cell_key"],
                    "relation_type": link["relation_type"],
                }
                for link in row["secondary_links"]
            ],
            f"binding {finding_id}.secondary_links",
        ):
            raise ValueError("secondary links are not canonical zero-vote links")
        _sorted_ids(
            row["query_capability_receipt_ids"],
            f"binding {finding_id}.query_capability_receipt_ids",
        )
        _sorted_ids(
            row["query_sensitivity_receipt_ids"],
            f"binding {finding_id}.query_sensitivity_receipt_ids",
        )
        _sorted_ids(row["reason_codes"], f"binding {finding_id}.reason_codes")
        if row["onset_future_free_causal"] is not _onset_future_free_causal(
            source, finding
        ):
            raise ValueError("producer binding causal-dependency result drifted")

    expected_seed = {
        **expected_identity,
        "typed_unit_inventory_sha256": _canonical_sha256(units),
        "binding_sha256s": [row["binding_sha256"] for row in rows],
    }
    expected_id = f"PRODINV-{_canonical_sha256(expected_seed)[:24]}"
    if candidate["inventory_id"] != expected_id:
        raise ValueError("producer-binding inventory ID mismatch")
    return candidate


def _manifest_aliases(
    validation_kwargs: Mapping[str, object] | None,
) -> dict[str, str]:
    manifest: Mapping[str, object] | None = None
    supplied = dict(validation_kwargs or {}).get("term_manifest")
    if isinstance(supplied, Mapping):
        manifest = supplied
    if manifest is None:
        try:
            from .event_findings_term_query_denominator_v2 import (
                load_clinical_eeg_finding_term_manifest_v2,
            )

            manifest = load_clinical_eeg_finding_term_manifest_v2()
        except (ImportError, OSError, TypeError, ValueError):
            manifest = None
    result: dict[str, str] = {}
    if manifest is None:
        return result
    for raw in manifest.get("terms", []):
        if not isinstance(raw, Mapping):
            continue
        canonical = str(raw.get("term_id"))
        result[canonical] = canonical
        for alias in raw.get("legacy_aliases", []):
            result[str(alias)] = canonical
    return result


def _find_receipt_object(
    roots: Mapping[str, object] | None, receipt_id: str
) -> Mapping[str, object] | None:
    """Find one unique typed receipt inside host-side validation inputs."""

    matches: list[Mapping[str, object]] = []

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            if value.get("receipt_id") == receipt_id:
                matches.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                visit(child)

    visit(roots or {})
    unique = {_canonical_json(dict(row)): row for row in matches}
    if len(unique) != 1:
        return None
    return next(iter(unique.values()))


def _band_covers(
    available: Sequence[float] | None,
    required: Sequence[float] | None,
    *,
    tolerance: float,
) -> bool:
    if required is None:
        return True
    if available is None:
        return False
    return (
        float(available[0]) <= float(required[0]) + tolerance
        and float(available[1]) >= float(required[1]) - tolerance
    )


def _term_receipt_same_domain(
    receipt: Mapping[str, object] | None,
    *,
    scope: Mapping[str, object],
    cell: Mapping[str, object],
) -> bool:
    if receipt is None:
        return False
    exact = {
        "producer_id": scope["producer_id"],
        "target_domain_id": scope["target_domain_id"],
        "term_query_id": cell["term_query_id"],
        "term_id": cell["term"]["term_id"],  # type: ignore[index]
        "query_cell_key": cell["query_cell_key"],
        "family": cell["family"],
        "scope_id": cell["scope_id"],
        "view_profile_id": cell["view_profile_id"],
        "reference_profile_id": cell["reference_profile_id"],
        "bandwidth_profile_id": cell["bandwidth_profile_id"],
    }
    for key, expected in exact.items():
        if key in receipt and receipt[key] != expected:
            return False
    unit_key = scope["unit"]["unit_key"]  # type: ignore[index]
    for key in ("unit_key", "typed_unit_key"):
        if key in receipt and receipt[key] != unit_key:
            return False
    receipt_unit = receipt.get("unit")
    if isinstance(receipt_unit, Mapping) and receipt_unit.get("unit_key") != unit_key:
        return False
    # A receipt that exposes none of the identity-bearing fields cannot bridge
    # producer identity; exact denominator replay alone is not sufficient.
    return any(key in receipt for key in exact)


def _capability_same_domain(
    *,
    source: Mapping[str, object],
    finding: Mapping[str, object],
    producer_row: Mapping[str, object],
    cell: Mapping[str, object],
    policy: Mapping[str, object],
    term_validation_kwargs: Mapping[str, object] | None,
    tolerance: float,
) -> bool:
    capability_id = finding.get("capability_receipt_id")
    query_ids = list(cell["positive_capability_receipt_ids"])
    if capability_id is None or not query_ids:
        return False
    if list(producer_row["query_capability_receipt_ids"]) != query_ids:
        return False
    capabilities = _receipt_map(source, "capability_qualification_receipts")
    capability = capabilities.get(str(capability_id))
    if capability is None:
        return False
    scope = producer_row["domain_scope"]
    thresholds = policy["same_domain_policy"]
    operating = capability["operating_scope"]
    if (
        capability["producer_id"] != scope["producer_id"]
        or capability["target_domain_id"] != scope["target_domain_id"]
        or finding["family"] not in capability["qualified_families"]
        or finding["term"]["term_id"] not in capability["qualified_term_ids"]
        or float(capability["precision_lower_bound"])
        < float(thresholds["minimum_capability_precision_lower_bound"])
        or float(capability["coverage"])
        < float(thresholds["minimum_qualification_coverage"])
        or not set(scope["view_roles"]).issubset(operating["view_roles"])
        or not set(scope["reference_types"]).issubset(
            operating["montage_references"]
        )
        or float(scope["sample_rate_hz"])
        + tolerance
        < float(operating["minimum_sample_rate_hz"])
        or not _band_covers(
            scope["effective_bandwidth_hz"],
            operating["required_bandwidth_hz"],
            tolerance=tolerance,
        )
        or float(scope["usable_fraction"]) + tolerance
        < float(operating["minimum_usable_fraction"])
    ):
        return False
    return all(
        _term_receipt_same_domain(
            _find_receipt_object(term_validation_kwargs, receipt_id),
            scope=scope,
            cell=cell,
        )
        for receipt_id in query_ids
    )


def _sensitivity_same_domain(
    *,
    source: Mapping[str, object],
    finding: Mapping[str, object],
    producer_row: Mapping[str, object],
    cell: Mapping[str, object],
    policy: Mapping[str, object],
    term_validation_kwargs: Mapping[str, object] | None,
) -> bool:
    sensitivity_id = finding.get("sensitivity_receipt_id")
    query_ids = list(cell["negative_sensitivity_receipt_ids"])
    if sensitivity_id is None or not query_ids:
        return False
    if list(producer_row["query_sensitivity_receipt_ids"]) != query_ids:
        return False
    sensitivity = _receipt_map(source, "sensitivity_receipts").get(
        str(sensitivity_id)
    )
    if sensitivity is None:
        return False
    scope = producer_row["domain_scope"]
    thresholds = policy["same_domain_policy"]
    if (
        sensitivity["producer_id"] != scope["producer_id"]
        or sensitivity["target_domain_id"] != scope["target_domain_id"]
        or sensitivity["qualified_family"] != finding["family"]
        or sensitivity["qualified_term_id"] != finding["term"]["term_id"]
        or sensitivity["opportunity_policy_sha256"]
        != scope["opportunity_policy_sha256"]
        or float(sensitivity["sensitivity_lower_bound"])
        < float(thresholds["minimum_sensitivity_lower_bound"])
        or float(sensitivity["coverage"])
        < float(thresholds["minimum_qualification_coverage"])
    ):
        return False
    return all(
        _term_receipt_same_domain(
            _find_receipt_object(term_validation_kwargs, receipt_id),
            scope=scope,
            cell=cell,
        )
        for receipt_id in query_ids
    )


def _explicit_term_decision(
    source: Mapping[str, object],
    finding: Mapping[str, object],
    producer_row: Mapping[str, object],
) -> bool:
    decision_id = finding.get("term_decision_receipt_id")
    if decision_id is None:
        return False
    decision = _receipt_map(source, "term_decision_receipts").get(
        str(decision_id)
    )
    if decision is None:
        return False
    evidence_id = str(finding["evidence_id"])
    criterion_evidence = {
        str(value)
        for criterion in decision["criterion_results"]
        for value in criterion["evidence_ids"]
    }
    return bool(
        decision["event_id"] == source["event_id"]
        and decision["producer_id"] == producer_row["producer_id"]
        and decision["term_id"] == finding["term"]["term_id"]
        and decision["asserted_status"] == finding["status"]
        and decision["decision"] == "qualified"
        and evidence_id in criterion_evidence
    )


def _opportunity_checks(
    source: Mapping[str, object],
    finding: Mapping[str, object],
    cell: Mapping[str, object],
    *,
    tolerance: float,
) -> tuple[bool, bool, bool]:
    opportunities = {
        str(row["evaluation_opportunity_id"]): row
        for row in source["evaluation_opportunities"]  # type: ignore[index]
    }
    opportunity = opportunities[str(finding["evaluation_opportunity_id"])]
    finding_union = _finding_interval_union(finding, tolerance=tolerance)
    opportunity_union = (
        []
        if opportunity["interval"] is None
        else _interval_union(
            [
                {
                    "start": opportunity["interval"]["start"],
                    "stop": opportunity["interval"]["stop"],
                }
            ],
            context="Finding self opportunity",
            tolerance=tolerance,
        )
    )
    present_interval_subset = bool(
        finding["status"] != "present"
        or (
            finding_union
            and _union_subset(
                finding_union,
                cell["evaluable_interval_union"],
                tolerance=tolerance,
            )
        )
    )
    finding_units = {
        row["unit_key"] for row in _finding_typed_support(finding)
    }
    opportunity_units = {
        str(value) for value in opportunity["spatial_unit_keys"]
    }
    unit = cell["unit"]
    cell_unit_ok = (
        unit["unit_type"] == "event"
        or (
            unit["unit_key"] in finding_units
            if finding["status"] in {"present", "uncertain"}
            else unit["unit_key"] in opportunity_units
        )
    )
    self_opportunity_subset = bool(
        (not finding_union or _union_subset(
            finding_union, opportunity_union, tolerance=tolerance
        ))
        and finding_units.issubset(opportunity_units)
        and (
            unit["unit_type"] == "event"
            or unit["unit_key"] in opportunity_units
        )
    )
    required = cell["required_interval_union"]
    evaluable = cell["evaluable_interval_union"]
    absence_complete = bool(
        finding["status"] != "absent_with_opportunity"
        or (
            opportunity["status"] == "sufficient"
            and cell["opportunity_status"] == "sufficient"
            and cell["negative_opportunity_eligible"]
            and _union_equal(required, evaluable, tolerance=tolerance)
            and _union_subset(required, opportunity_union, tolerance=tolerance)
            and (
                unit["unit_type"] == "event"
                or unit["unit_key"] in opportunity_units
            )
        )
    )
    return present_interval_subset, self_opportunity_subset and cell_unit_ok, absence_complete


def _validate_item_denominator_join(
    *,
    source: Mapping[str, object],
    atom_roster_receipt: object,
    item_denominator_receipt: object,
    supplied_join: object,
    validation_kwargs: Mapping[str, object],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    kwargs = dict(validation_kwargs)
    if "source_inventory" not in kwargs or "trusted_source_inventory_sha256" not in kwargs:
        raise ValueError(
            "item denominator validation requires a host-pinned source inventory"
        )
    expected_join = validate_atom_roster_against_independent_denominator(
        item_denominator_receipt,
        atom_roster_receipt,
        event_findings_v3=source,
        **kwargs,
    )
    if supplied_join != expected_join:
        raise ValueError("item denominator join does not match exact host replay")
    roster_kwargs = dict(kwargs.get("atom_roster_validation_kwargs") or {})
    roster = validate_event_findings_atom_roster_receipt(
        atom_roster_receipt,
        event_findings_v3=source,
        **roster_kwargs,
    )
    denominator = item_denominator_receipt
    if not isinstance(denominator, Mapping):
        raise TypeError("item denominator receipt must be an object")
    return deepcopy(dict(expected_join)), roster, deepcopy(dict(denominator))


def materialize_event_finding_binding_ledger_v1(
    event_findings_v3: object,
    *,
    atom_roster_receipt: object,
    item_denominator_receipt: object,
    item_denominator_join: object,
    item_denominator_validation_kwargs: Mapping[str, object],
    term_query_denominator_receipt: object,
    trusted_term_query_denominator_receipt_sha256: str,
    term_query_denominator_validation_kwargs: Mapping[str, object] | None,
    physical_instance_inventory: object,
    trusted_physical_instance_inventory_sha256: str,
    producer_binding_inventory: object,
    trusted_producer_binding_inventory_sha256: str,
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
    findings_validation_kwargs: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Materialize the exact-cover shadow binding ledger."""

    binding_policy = _policy(policy, trusted_policy_sha256)
    tolerance = float(
        binding_policy["interval_policy"]["comparison_tolerance_seconds"]
    )
    source = _validated_v3(event_findings_v3, findings_validation_kwargs)
    join, roster, item_denominator = _validate_item_denominator_join(
        source=source,
        atom_roster_receipt=atom_roster_receipt,
        item_denominator_receipt=item_denominator_receipt,
        supplied_join=item_denominator_join,
        validation_kwargs=item_denominator_validation_kwargs,
    )
    term_receipt, query_cells, term_units = _validate_term_query_denominator(
        term_query_denominator_receipt,
        trusted_receipt_sha256=trusted_term_query_denominator_receipt_sha256,
        validation_kwargs=term_query_denominator_validation_kwargs,
        tolerance=tolerance,
    )
    item_units = _typed_unit_inventory(
        item_denominator.get("typed_expected_units"),
        "item denominator typed_expected_units",
    )
    if item_units != term_units:
        raise ValueError("item and term denominators use different typed units")
    physical = validate_event_finding_physical_instance_inventory(
        physical_instance_inventory,
        event_findings_v3=source,
        trusted_inventory_sha256=trusted_physical_instance_inventory_sha256,
        findings_validation_kwargs=findings_validation_kwargs,
    )
    producer = validate_event_finding_producer_binding_inventory(
        producer_binding_inventory,
        event_findings_v3=source,
        trusted_inventory_sha256=trusted_producer_binding_inventory_sha256,
        findings_validation_kwargs=findings_validation_kwargs,
    )
    expected_unit_hash = _canonical_sha256(item_units)
    for label, inventory in (("physical", physical), ("producer", producer)):
        if (
            inventory["typed_expected_units"] != item_units
            or inventory["typed_unit_inventory_sha256"] != expected_unit_hash
        ):
            raise ValueError(f"{label} inventory typed units drifted")

    identity = {
        "event_id": source["event_id"],
        "record_id": source["provenance"]["record_id"],
        "canonical_signal_sha256": source["provenance"][
            "canonical_signal_sha256"
        ],
    }
    for label, value in (
        ("item join", join),
        ("term denominator", term_receipt),
        ("physical inventory", physical),
        ("producer inventory", producer),
    ):
        for key, expected in identity.items():
            if value.get(key) != expected:
                raise ValueError(f"{label} {key} identity mismatch")
    findings_hash = _canonical_sha256(source)
    if any(
        value.get("findings_payload_sha256") != findings_hash
        for value in (roster, physical, producer)
    ):
        raise ValueError("binding inputs do not share the exact Findings payload")

    finding_map = {str(row["evidence_id"]): row for row in source["findings"]}
    producer_rows = {
        str(row["finding_id"]): row for row in producer["bindings"]
    }
    physical_assignments = {
        str(row["finding_id"]): row
        for row in physical["finding_assignments"]
    }
    physical_keys = {
        str(row["physical_instance_key"])
        for row in physical["physical_instances"]
    }
    cell_map = {str(row["query_cell_key"]): row for row in query_cells}
    aliases = _manifest_aliases(term_query_denominator_validation_kwargs)
    semantic_groups: dict[str, list[str]] = {}
    for finding_id, row in producer_rows.items():
        semantic_groups.setdefault(
            str(row["semantic_fingerprint_sha256"]), []
        ).append(finding_id)
    duplicate_ids = {
        finding_id
        for values in semantic_groups.values()
        if len(values) > 1
        for finding_id in values
    }
    primary_by_cell: dict[str, list[str]] = {}
    for finding_id, row in producer_rows.items():
        primary = row["primary_query_cell_key"]
        if primary is not None:
            primary_by_cell.setdefault(str(primary), []).append(finding_id)

    finding_checks: dict[str, dict[str, bool]] = {}
    primary_dispositions: list[dict[str, Any]] = []
    for finding_id in sorted(finding_map):
        finding = finding_map[finding_id]
        producer_row = producer_rows[finding_id]
        primary_key = producer_row["primary_query_cell_key"]
        cell = cell_map.get(str(primary_key)) if primary_key is not None else None
        term = _term_ref(finding["term"], f"finding {finding_id}.term")
        canonical_term = aliases.get(term["term_id"], term["term_id"])
        scope = producer_row["domain_scope"]
        term_exact = bool(
            cell is not None
            and canonical_term == cell["term"]["term_id"]
            and finding["family"] == cell["family"]
            and scope["term_id"] == term["term_id"]
            and scope["operational_rule_id"] == term["operational_rule_id"]
        )
        typed_unit_exact = bool(
            cell is not None
            and scope["unit"] == cell["unit"]
            and scope["granularity"] == cell["granularity"]
        )
        profile_exact = bool(
            cell is not None
            and scope["view_profile_id"] == cell["view_profile_id"]
            and scope["reference_profile_id"] == cell["reference_profile_id"]
            and scope["bandwidth_profile_id"] == cell["bandwidth_profile_id"]
        )
        if cell is None:
            present_subset = self_opportunity = absence_complete = False
        else:
            present_subset, self_opportunity, absence_complete = _opportunity_checks(
                source, finding, cell, tolerance=tolerance
            )
        physical_key = producer_row["physical_instance_key"]
        physical_exact = bool(
            physical_assignments[finding_id]["physical_instance_key"]
            == physical_key
            and (physical_key is None or str(physical_key) in physical_keys)
        )
        producer_completed = (
            producer_row["processing_disposition"] == "completed"
        )
        capability_same = bool(
            cell is not None
            and profile_exact
            and _capability_same_domain(
                source=source,
                finding=finding,
                producer_row=producer_row,
                cell=cell,
                policy=binding_policy,
                term_validation_kwargs=term_query_denominator_validation_kwargs,
                tolerance=tolerance,
            )
        )
        sensitivity_same = bool(
            cell is not None
            and profile_exact
            and _sensitivity_same_domain(
                source=source,
                finding=finding,
                producer_row=producer_row,
                cell=cell,
                policy=binding_policy,
                term_validation_kwargs=term_query_denominator_validation_kwargs,
            )
        )
        explicit_decision = _explicit_term_decision(
            source, finding, producer_row
        )
        onset_causal = bool(producer_row["onset_future_free_causal"])
        single_primary = bool(
            primary_key is None
            or len(primary_by_cell.get(str(primary_key), [])) == 1
        )
        checks = {
            "term_exact": term_exact,
            "typed_unit_exact": typed_unit_exact,
            "present_interval_subset": present_subset,
            "self_opportunity_subset": self_opportunity,
            "absence_complete_scope": absence_complete,
            "single_primary": single_primary,
            "duplicate_free": finding_id not in duplicate_ids,
            "physical_instance_exact": physical_exact,
            "producer_completed": producer_completed,
            "capability_same_domain": capability_same,
            "sensitivity_same_domain": sensitivity_same,
            "explicit_term_decision": explicit_decision,
            "onset_future_free_causal": onset_causal,
            "item_denominator_join_valid": True,
            "term_denominator_valid": True,
        }
        finding_checks[finding_id] = checks
        reasons: list[str] = []
        if finding_id in duplicate_ids:
            disposition = "withheld_duplicate"
            reasons.append("semantic_duplicate")
        elif not producer_completed:
            disposition = "withheld_producer_failure"
            reasons.append("producer_technical_failure")
        elif primary_key is None or cell is None:
            disposition = (
                "not_evaluable"
                if finding["status"] == "not_evaluable"
                else "withheld_no_matching_query"
            )
            reasons.append("no_registered_primary_query")
        elif not single_primary:
            disposition = "withheld_multiple_primary"
            reasons.append("multiple_findings_claim_same_primary_query")
        elif not physical_exact:
            disposition = "withheld_physical_instance_mismatch"
            reasons.append("physical_instance_binding_mismatch")
        elif not term_exact or not typed_unit_exact or not profile_exact:
            disposition = "withheld_domain_mismatch"
            reasons.append("term_unit_or_profile_domain_mismatch")
        elif not present_subset or not self_opportunity or not absence_complete:
            disposition = "withheld_opportunity_mismatch"
            reasons.append("finding_opportunity_scope_mismatch")
        else:
            disposition = "bound_primary"
        secondary = deepcopy(producer_row["secondary_links"])
        row: dict[str, Any] = {
            "finding_id": finding_id,
            "finding_sha256": producer_row["finding_sha256"],
            "semantic_fingerprint_sha256": producer_row[
                "semantic_fingerprint_sha256"
            ],
            "finding_status": finding["status"],
            "assertion_level": finding["assertion_level"],
            "producer_binding_id": producer_row["producer_binding_id"],
            "producer_receipt_id": producer_row["producer_receipt_id"],
            "processing_disposition": producer_row["processing_disposition"],
            "physical_instance_key": physical_key,
            "primary_query_cell_key": primary_key,
            "disposition": disposition,
            "primary_vote_count": 1 if disposition == "bound_primary" else 0,
            "secondary_links": secondary,
            "reason_codes": sorted(set(reasons)),
            "binding_sha256": "0" * 64,
        }
        row["binding_sha256"] = _self_hash(row, "binding_sha256")
        primary_dispositions.append(row)

    disposition_map = {
        str(row["finding_id"]): row for row in primary_dispositions
    }
    query_bindings: list[dict[str, Any]] = []
    for cell_key in sorted(cell_map):
        cell = cell_map[cell_key]
        primary_ids = primary_by_cell.get(cell_key, [])
        primary_id = primary_ids[0] if len(primary_ids) == 1 else None
        primary_finding = finding_map.get(primary_id) if primary_id else None
        primary_disposition = disposition_map.get(primary_id) if primary_id else None
        checks = (
            finding_checks[primary_id]
            if primary_id is not None
            else {
                "term_exact": False,
                "typed_unit_exact": False,
                "present_interval_subset": False,
                "self_opportunity_subset": False,
                "absence_complete_scope": False,
                "single_primary": len(primary_ids) <= 1,
                "duplicate_free": True,
                "physical_instance_exact": False,
                "producer_completed": False,
                "capability_same_domain": False,
                "sensitivity_same_domain": False,
                "explicit_term_decision": False,
                "onset_future_free_causal": False,
                "item_denominator_join_valid": True,
                "term_denominator_valid": True,
            }
        )
        reasons: list[str] = []
        technical_failure = bool(
            cell["technical_failure_receipt_ids"]
            or cell["processing_disposition"] == "technical_failure"
            or (
                primary_id is not None
                and producer_rows[primary_id]["processing_disposition"]
                == "technical_failure"
            )
        )
        if technical_failure:
            promotion = "technical_failure"
            reasons.append("technical_failure_is_not_a_negative")
        elif len(primary_ids) > 1:
            promotion = "withheld_conflict"
            reasons.append("multiple_primary_findings")
        elif primary_id is None:
            promotion = (
                "not_evaluable"
                if cell["opportunity_status"] == "not_evaluable"
                else "withheld_missing_primary"
            )
            reasons.append("no_primary_finding")
        elif primary_disposition["disposition"] == "withheld_duplicate":
            promotion = "withheld_duplicate"
            reasons.append("semantic_duplicate")
        elif primary_disposition["disposition"] != "bound_primary":
            mapping = {
                "withheld_opportunity_mismatch": "withheld_opportunity_mismatch",
                "withheld_domain_mismatch": "withheld_domain_mismatch",
                "withheld_physical_instance_mismatch": "withheld_physical_instance_mismatch",
                "withheld_producer_failure": "technical_failure",
                "not_evaluable": "not_evaluable",
            }
            promotion = mapping.get(
                str(primary_disposition["disposition"]),
                "withheld_missing_primary",
            )
            reasons.extend(primary_disposition["reason_codes"])
        else:
            status = str(primary_finding["status"])
            report_eligible = (
                primary_finding["assertion_level"]
                == "report_eligible_automated"
            )
            base_required = all(
                checks[key]
                for key in (
                    "term_exact",
                    "typed_unit_exact",
                    "self_opportunity_subset",
                    "single_primary",
                    "duplicate_free",
                    "physical_instance_exact",
                    "producer_completed",
                    "explicit_term_decision",
                    "onset_future_free_causal",
                    "item_denominator_join_valid",
                    "term_denominator_valid",
                )
            )
            if status == "present":
                if (
                    report_eligible
                    and base_required
                    and checks["present_interval_subset"]
                    and checks["capability_same_domain"]
                ):
                    promotion = "promoted_present"
                else:
                    promotion = "candidate_present"
                    reasons.append("present_not_fully_promotion_qualified")
            elif status == "absent_with_opportunity":
                if (
                    report_eligible
                    and base_required
                    and checks["absence_complete_scope"]
                    and checks["capability_same_domain"]
                    and checks["sensitivity_same_domain"]
                    and cell["negative_opportunity_eligible"]
                ):
                    promotion = "promoted_absent"
                elif not checks["absence_complete_scope"]:
                    promotion = "withheld_opportunity_mismatch"
                    reasons.append("absence_scope_not_complete")
                else:
                    promotion = "withheld_domain_mismatch"
                    reasons.append("absence_not_fully_promotion_qualified")
            elif status == "uncertain":
                promotion = "candidate_uncertain"
            else:
                promotion = "not_evaluable"
        secondary_ids = sorted(
            finding_id
            for finding_id, row in producer_rows.items()
            if any(
                link["query_cell_key"] == cell_key
                for link in row["secondary_links"]
            )
        )
        physical_key = (
            None
            if primary_id is None
            else producer_rows[primary_id]["physical_instance_key"]
        )
        row = {
            "query_cell_key": cell_key,
            "query_cell_sha256": cell["query_cell_sha256"],
            "term_query_id": cell["term_query_id"],
            "term": cell["term"],
            "family": cell["family"],
            "unit": cell["unit"],
            "required_interval_union": cell["required_interval_union"],
            "evaluable_interval_union": cell["evaluable_interval_union"],
            "opportunity_status": cell["opportunity_status"],
            "negative_opportunity_eligible": cell[
                "negative_opportunity_eligible"
            ],
            "technical_failure": technical_failure,
            "primary_finding_id": primary_id,
            "primary_finding_sha256": (
                None
                if primary_finding is None
                else _finding_sha256(primary_finding)
            ),
            "primary_status": (
                None if primary_finding is None else primary_finding["status"]
            ),
            "primary_assertion_level": (
                None
                if primary_finding is None
                else primary_finding["assertion_level"]
            ),
            "physical_instance_key": physical_key,
            "secondary_finding_ids": secondary_ids,
            "checks": checks,
            "promotion_disposition": promotion,
            "promotion_vote_count": 1
            if promotion in {"promoted_present", "promoted_absent"}
            else 0,
            "reason_codes": sorted(set(str(value) for value in reasons)),
            "binding_sha256": "0" * 64,
        }
        row["binding_sha256"] = _self_hash(row, "binding_sha256")
        query_bindings.append(row)

    summary = {
        "expected_finding_count": len(finding_map),
        "primary_disposition_count": len(primary_dispositions),
        "expected_query_cell_count": len(query_cells),
        "query_cell_binding_count": len(query_bindings),
        "unique_physical_instance_count": len(physical_keys),
        "bound_primary_count": sum(
            row["disposition"] == "bound_primary"
            for row in primary_dispositions
        ),
        "secondary_link_count": sum(
            len(row["secondary_links"]) for row in primary_dispositions
        ),
        "duplicate_finding_count": len(duplicate_ids),
        "multiple_primary_query_count": sum(
            len(values) > 1 for values in primary_by_cell.values()
        ),
        "technical_failure_count": sum(
            row["technical_failure"] for row in query_bindings
        ),
        "promoted_present_count": sum(
            row["promotion_disposition"] == "promoted_present"
            for row in query_bindings
        ),
        "promoted_absent_count": sum(
            row["promotion_disposition"] == "promoted_absent"
            for row in query_bindings
        ),
        "withheld_count": sum(
            str(row["promotion_disposition"]).startswith("withheld_")
            for row in query_bindings
        ),
        "all_findings_exactly_once_as_primary_disposition": True,
        "all_query_cells_exactly_once": True,
        "secondary_vote_count": 0,
        "status_assertion_same_primary_only": True,
        "item_denominator_join_validated": True,
        "term_denominator_validated": True,
        "public_synthetic_shadow_only": True,
        "production_route_connected": False,
        "clinical_correctness_claimed": False,
    }
    term_receipt_id = term_receipt.get("receipt_id")
    if term_receipt_id is None:
        raise ValueError("term-query denominator receipt_id is required")
    ledger_seed = {
        **identity,
        "policy_sha256": binding_policy["policy_sha256"],
        "findings_payload_sha256": findings_hash,
        "primary_bindings": [row["binding_sha256"] for row in primary_dispositions],
        "query_bindings": [row["binding_sha256"] for row in query_bindings],
    }
    ledger: dict[str, Any] = {
        "schema_version": EVENT_FINDING_BINDING_LEDGER_SCHEMA_VERSION,
        "ledger_id": f"FINDBIND-{_canonical_sha256(ledger_seed)[:24]}",
        "method_id": EVENT_FINDING_BINDING_METHOD_ID,
        **identity,
        "policy_id": binding_policy["policy_id"],
        "policy_sha256": binding_policy["policy_sha256"],
        "findings_schema_version": source["schema_version"],
        "findings_payload_sha256": findings_hash,
        "atom_roster_receipt_id": roster["receipt_id"],
        "atom_roster_receipt_sha256": roster["receipt_sha256"],
        "item_denominator_receipt_id": item_denominator["receipt_id"],
        "item_denominator_receipt_sha256": item_denominator["receipt_sha256"],
        "item_denominator_join_sha256": join["join_sha256"],
        "term_query_denominator_receipt_id": term_receipt_id,
        "term_query_denominator_receipt_sha256": term_receipt["receipt_sha256"],
        "typed_unit_inventory_sha256": expected_unit_hash,
        "physical_instance_inventory_id": physical["inventory_id"],
        "physical_instance_inventory_sha256": physical["inventory_sha256"],
        "producer_binding_inventory_id": producer["inventory_id"],
        "producer_binding_inventory_sha256": producer["inventory_sha256"],
        "source_firewall": deepcopy(_SOURCE_FIREWALL),
        "primary_dispositions": primary_dispositions,
        "query_cell_bindings": query_bindings,
        "summary": summary,
        "ledger_sha256": "0" * 64,
    }
    ledger["ledger_sha256"] = _self_hash(ledger, "ledger_sha256")
    errors = _schema_errors(ledger, EVENT_FINDING_BINDING_LEDGER_SCHEMA_PATH)
    if errors:
        raise ValueError(
            "materialized event Finding binding ledger is invalid: "
            + "; ".join(errors)
        )
    return ledger


def validate_event_finding_binding_ledger_v1(
    value: object,
    *,
    event_findings_v3: object,
    atom_roster_receipt: object,
    item_denominator_receipt: object,
    item_denominator_join: object,
    item_denominator_validation_kwargs: Mapping[str, object],
    term_query_denominator_receipt: object,
    trusted_term_query_denominator_receipt_sha256: str,
    term_query_denominator_validation_kwargs: Mapping[str, object] | None,
    physical_instance_inventory: object,
    trusted_physical_instance_inventory_sha256: str,
    producer_binding_inventory: object,
    trusted_producer_binding_inventory_sha256: str,
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
    findings_validation_kwargs: Mapping[str, object] | None = None,
    trusted_ledger_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate self-hashes, then exact-replay from every host trust root."""

    if type(value) is not dict:
        raise TypeError("event Finding binding ledger must be an object")
    candidate = deepcopy(value)
    errors = _schema_errors(candidate, EVENT_FINDING_BINDING_LEDGER_SCHEMA_PATH)
    if errors:
        raise ValueError(
            "event Finding binding ledger schema validation failed: "
            + "; ".join(errors)
        )
    if candidate["ledger_sha256"] != _self_hash(candidate, "ledger_sha256"):
        raise ValueError("event Finding binding ledger hash mismatch")
    if trusted_ledger_sha256 is not None and candidate["ledger_sha256"] != _sha(
        trusted_ledger_sha256, "trusted_ledger_sha256"
    ):
        raise ValueError("event Finding binding ledger is not host trusted")
    for collection in ("primary_dispositions", "query_cell_bindings"):
        for row in candidate[collection]:
            if row["binding_sha256"] != _self_hash(row, "binding_sha256"):
                raise ValueError(f"{collection} contains a row hash mismatch")
    expected = materialize_event_finding_binding_ledger_v1(
        event_findings_v3,
        atom_roster_receipt=atom_roster_receipt,
        item_denominator_receipt=item_denominator_receipt,
        item_denominator_join=item_denominator_join,
        item_denominator_validation_kwargs=item_denominator_validation_kwargs,
        term_query_denominator_receipt=term_query_denominator_receipt,
        trusted_term_query_denominator_receipt_sha256=(
            trusted_term_query_denominator_receipt_sha256
        ),
        term_query_denominator_validation_kwargs=(
            term_query_denominator_validation_kwargs
        ),
        physical_instance_inventory=physical_instance_inventory,
        trusted_physical_instance_inventory_sha256=(
            trusted_physical_instance_inventory_sha256
        ),
        producer_binding_inventory=producer_binding_inventory,
        trusted_producer_binding_inventory_sha256=(
            trusted_producer_binding_inventory_sha256
        ),
        policy=policy,
        trusted_policy_sha256=trusted_policy_sha256,
        findings_validation_kwargs=findings_validation_kwargs,
    )
    if candidate != expected:
        raise ValueError(
            "event Finding binding ledger does not match exact host replay"
        )
    return candidate


# Descriptive aliases keep direct-module use ergonomic without changing the
# package-level public surface in this strictly additive shadow milestone.
materialize_event_finding_binding_ledger = (
    materialize_event_finding_binding_ledger_v1
)
validate_event_finding_binding_ledger = validate_event_finding_binding_ledger_v1


__all__ = [
    "DEFAULT_EVENT_FINDING_BINDING_POLICY_PATH",
    "DEFAULT_EVENT_FINDING_BINDING_POLICY_SHA256",
    "EVENT_FINDING_BINDING_LEDGER_SCHEMA_PATH",
    "EVENT_FINDING_BINDING_LEDGER_SCHEMA_VERSION",
    "EVENT_FINDING_BINDING_METHOD_ID",
    "EVENT_FINDING_BINDING_POLICY_ID",
    "EVENT_FINDING_BINDING_POLICY_SCHEMA_PATH",
    "EVENT_FINDING_BINDING_POLICY_SCHEMA_VERSION",
    "EVENT_FINDING_PHYSICAL_INSTANCE_INVENTORY_SCHEMA_VERSION",
    "EVENT_FINDING_PRODUCER_BINDING_INVENTORY_SCHEMA_VERSION",
    "build_event_finding_physical_instance_inventory",
    "build_event_finding_producer_binding_inventory",
    "load_event_finding_binding_policy",
    "materialize_event_finding_binding_ledger",
    "materialize_event_finding_binding_ledger_v1",
    "validate_event_finding_binding_ledger",
    "validate_event_finding_binding_ledger_v1",
    "validate_event_finding_binding_policy",
    "validate_event_finding_physical_instance_inventory",
    "validate_event_finding_producer_binding_inventory",
]
