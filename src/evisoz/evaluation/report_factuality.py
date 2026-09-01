"""Facts-first evaluation for EviSOZ structured reports.

The evaluator consumes a parsed report plan/output and a *trusted structured
evidence view*.  It does not run an LLM, tokenize raw EEG, or infer truth from
free text.  This makes it suitable for the future Qwen output validator and
for the required report controls (side swap, onset/spread swap, unsupported
channel, and certainty promotion).
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping, Sequence

from src.evisoz.data.artifact_ref import canonical_json_sha256


REPORT_CLAIM_TYPES = (
    "onset",
    "spread",
    "morphology",
    "laterality",
    "region",
    "quality",
    "localizability",
    "limitation",
)
_CERTAINTY_RANK = {
    "not_assessable": 0,
    "uncertain": 1,
    "possible": 1,
    "probable": 2,
    "supported": 3,
    "definite": 4,
}
_FORBIDDEN_SURFACE_PATTERNS = (
    re.compile(r"(?:确认|证实|确定).{0,12}(?:皮层)?SOZ", re.IGNORECASE),
    re.compile(r"(?:皮层SOZ|致痫区|手术靶点)(?:位于|为|确定)", re.IGNORECASE),
    re.compile(r"(?:建议|应当|需要).{0,12}(?:用药|停药|加药|手术|治疗)"),
    re.compile(r"(?:相位反转点|IED).{0,12}(?:就是|即为|证明)"),
)


def _string_set(value: object, *, name: str) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(f"{name} must be a string collection")
    result = {item for item in value if isinstance(item, str) and item}
    if len(result) != len(value):
        raise ValueError(f"{name} must contain only unique non-empty strings")
    return result


def _validate_evidence(evidence: Mapping[str, object]) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a mapping")
    allowed = {
        "evidence_ids",
        "onset_channels",
        "spread_channels",
        "onset_order",
        "spread_order",
        "morphologies",
        "laterality",
        "regions",
        "certainty",
        "report_scope",
    }
    unknown = set(evidence).difference(allowed)
    if unknown:
        raise ValueError(f"evidence has unknown fields: {sorted(unknown)}")
    result = {
        "evidence_ids": _string_set(evidence.get("evidence_ids", []), name="evidence_ids"),
        "onset_channels": _string_set(evidence.get("onset_channels", []), name="onset_channels"),
        "spread_channels": _string_set(evidence.get("spread_channels", []), name="spread_channels"),
        "onset_order": list(evidence.get("onset_order", [])),
        "spread_order": list(evidence.get("spread_order", [])),
        "morphologies": _string_set(evidence.get("morphologies", []), name="morphologies"),
        "regions": _string_set(evidence.get("regions", []), name="regions"),
        "laterality": evidence.get("laterality"),
        "certainty": evidence.get("certainty", "uncertain"),
        "report_scope": evidence.get("report_scope", "full_soz"),
    }
    for name in ("onset_order", "spread_order"):
        order = result[name]
        if not isinstance(order, list) or any(not isinstance(item, str) or not item for item in order):
            raise ValueError(f"{name} must be a non-empty-string list")
        if len(order) != len(set(order)):
            raise ValueError(f"{name} must not contain duplicates")
    if result["laterality"] is not None and result["laterality"] not in {"left", "right", "midline", "bilateral", "unknown"}:
        raise ValueError("evidence laterality is invalid")
    if result["certainty"] not in _CERTAINTY_RANK:
        raise ValueError("evidence certainty is invalid")
    if not isinstance(result["report_scope"], str) or not result["report_scope"]:
        raise ValueError("evidence report_scope is invalid")
    return result


def _validate_report(report: Mapping[str, object]) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    claims = report.get("claims")
    if not isinstance(claims, list):
        raise ValueError("report.claims must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(claims):
        if not isinstance(raw, Mapping):
            raise ValueError(f"report claim {index} must be an object")
        claim_type = raw.get("claim_type")
        if claim_type not in REPORT_CLAIM_TYPES:
            raise ValueError(f"report claim {index} has unsupported claim_type")
        claim_id = raw.get("claim_id", f"claim-{index}")
        if not isinstance(claim_id, str) or not claim_id or claim_id in seen:
            raise ValueError("report claim IDs must be unique non-empty strings")
        seen.add(claim_id)
        units = raw.get("units", [])
        if claim_type in {"onset", "spread", "morphology", "region"}:
            units_set = _string_set(units, name=f"report.claims[{index}].units")
        else:
            if units not in ([], None) and not isinstance(units, (list, tuple, set)):
                raise ValueError(f"report claim {index} units are malformed")
            units_set = _string_set(units or [], name=f"report.claims[{index}].units")
        evidence_ids = _string_set(raw.get("evidence_ids", []), name=f"report.claims[{index}].evidence_ids")
        certainty = raw.get("certainty", "uncertain")
        if certainty not in _CERTAINTY_RANK:
            raise ValueError(f"report claim {index} certainty is invalid")
        order = raw.get("order", [])
        if not isinstance(order, list) or any(not isinstance(item, str) or not item for item in order):
            raise ValueError(f"report claim {index} order is malformed")
        text = raw.get("text", "")
        if not isinstance(text, str):
            raise ValueError(f"report claim {index} text is not a string")
        normalized.append({
            "claim_id": claim_id,
            "claim_type": claim_type,
            "units": units_set,
            "evidence_ids": evidence_ids,
            "certainty": certainty,
            "order": order,
            "text": text,
        })
    surface = report.get("text", "")
    if not isinstance(surface, str):
        raise ValueError("report.text must be a string")
    return {"claims": normalized, "text": surface}


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_evisoz_report_factuality(
    report: Mapping[str, object],
    evidence: Mapping[str, object],
) -> dict[str, Any]:
    """Evaluate parsed report claims against a trusted evidence view.

    ``evidence`` is intentionally a compact, caller-built structured view;
    callers must construct it from released direct/candidate claims according
    to the report scope.  The evaluator never treats a missing evidence field
    as a negative finding.
    """

    parsed_report = _validate_report(report)
    trusted = _validate_evidence(evidence)
    claims = parsed_report["claims"]
    channel_claims = [claim for claim in claims if claim["claim_type"] in {"onset", "spread"}]
    reported_channels = set().union(*(claim["units"] for claim in channel_claims)) if channel_claims else set()
    trusted_channels = trusted["onset_channels"] | trusted["spread_channels"]
    supported_channels = reported_channels & trusted_channels
    morphology_claims = [claim for claim in claims if claim["claim_type"] == "morphology"]
    reported_morphologies = set().union(*(claim["units"] for claim in morphology_claims)) if morphology_claims else set()
    supported_morphologies = reported_morphologies & trusted["morphologies"]

    onset_claim = next((claim for claim in claims if claim["claim_type"] == "onset"), None)
    spread_claim = next((claim for claim in claims if claim["claim_type"] == "spread"), None)
    onset_units = onset_claim["units"] if onset_claim else set()
    spread_units = spread_claim["units"] if spread_claim else set()
    reversal = False
    if onset_claim and spread_claim:
        # A claim is reversed when onset is populated with trusted spread
        # units and spread with trusted onset units.  This remains safe for
        # disjoint regions and does not guess an order from lexical sorting.
        onset_is_spread = bool(onset_units & trusted["spread_channels"]) and not bool(onset_units & trusted["onset_channels"])
        spread_is_onset = bool(spread_units & trusted["onset_channels"]) and not bool(spread_units & trusted["spread_channels"])
        reversal = onset_is_spread and spread_is_onset

    laterality_claim = next((claim for claim in claims if claim["claim_type"] == "laterality"), None)
    reported_laterality = next(iter(laterality_claim["units"]), None) if laterality_claim and laterality_claim["units"] else None
    laterality_correct = reported_laterality is not None and trusted["laterality"] is not None and reported_laterality == trusted["laterality"]
    region_claim = next((claim for claim in claims if claim["claim_type"] == "region"), None)
    reported_regions = region_claim["units"] if region_claim else set()
    region_overlap = reported_regions & trusted["regions"]
    max_report_certainty = max((_CERTAINTY_RANK[claim["certainty"]] for claim in claims), default=0)
    uncertainty_preserved = max_report_certainty <= _CERTAINTY_RANK[trusted["certainty"]]
    unsupported_claims = [
        claim for claim in claims
        if not claim["evidence_ids"] or not claim["evidence_ids"].issubset(trusted["evidence_ids"])
    ]
    surface = parsed_report["text"] + "\n" + "\n".join(claim["text"] for claim in claims)
    boundary_violations = [pattern.pattern for pattern in _FORBIDDEN_SURFACE_PATTERNS if pattern.search(surface)]
    metrics = {
        "channel_entity_precision": _ratio(len(supported_channels), len(reported_channels)),
        "channel_entity_recall": _ratio(len(supported_channels), len(trusted_channels)),
        "unsupported_channel_rate": 1.0 - _ratio(len(supported_channels), len(reported_channels)) if reported_channels else 0.0,
        "morphology_correctness": _ratio(len(supported_morphologies), len(reported_morphologies)),
        "unsupported_morphology_rate": 1.0 - _ratio(len(supported_morphologies), len(reported_morphologies)) if reported_morphologies else 0.0,
        "laterality_correct": float(laterality_correct),
        "region_correctness": _ratio(len(region_overlap), len(reported_regions)),
        "propagation_correctness": float(not reversal),
        "onset_spread_reversal_rate": float(reversal),
        "uncertainty_preservation": float(uncertainty_preserved),
        "unsupported_claim_rate": _ratio(len(unsupported_claims), len(claims)),
        "clinical_boundary_compliance": float(not boundary_violations),
    }
    body: dict[str, Any] = {
        "schema_version": "evisoz_report_factuality_evaluation_v1",
        "status": "structured_report_factuality_evaluation",
        "metrics": metrics,
        "counts": {
            "claim_count": len(claims),
            "unsupported_claim_count": len(unsupported_claims),
            "reported_channel_count": len(reported_channels),
            "trusted_channel_count": len(trusted_channels),
            "boundary_violation_count": len(boundary_violations),
        },
        "diagnostics": {
            "unsupported_claim_ids": [claim["claim_id"] for claim in unsupported_claims],
            "boundary_violations": boundary_violations,
            "report_scope": trusted["report_scope"],
        },
        "policy": {
            "direct_or_bound_structured_evidence_only": True,
            "teacher_candidates_not_promoted": True,
            "tcp22_edges_expanded_to_nodes": False,
            "physician_report_text_used": False,
        },
        "receipt_sha256": "0" * 64,
    }
    body["receipt_sha256"] = canonical_json_sha256(body)
    return body


def validate_report_factuality_evaluation(value: object) -> dict[str, Any]:
    """Validate and replay a content-addressed factuality evaluation."""

    if type(value) is not dict:
        raise ValueError("report factuality evaluation must be an object")
    required = {"schema_version", "status", "metrics", "counts", "diagnostics", "policy", "receipt_sha256"}
    if set(value) != required:
        raise ValueError("report factuality evaluation fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != "evisoz_report_factuality_evaluation_v1" or data["status"] != "structured_report_factuality_evaluation":
        raise ValueError("report factuality evaluation identity drifted")
    metrics = data["metrics"]
    if type(metrics) is not dict or not metrics or any(
        not isinstance(item, (int, float)) or not 0 <= float(item) <= 1
        for item in metrics.values()
    ):
        raise ValueError("report factuality metrics must be finite rates in [0,1]")
    counts = data["counts"]
    if type(counts) is not dict or any(type(item) is not int or item < 0 for item in counts.values()):
        raise ValueError("report factuality counts are invalid")
    diagnostics = data["diagnostics"]
    if type(diagnostics) is not dict or set(diagnostics) != {"unsupported_claim_ids", "boundary_violations", "report_scope"}:
        raise ValueError("report factuality diagnostics drifted")
    if not isinstance(diagnostics["unsupported_claim_ids"], list) or not isinstance(diagnostics["boundary_violations"], list) or not isinstance(diagnostics["report_scope"], str):
        raise ValueError("report factuality diagnostics are malformed")
    policy = data["policy"]
    if policy != {
        "direct_or_bound_structured_evidence_only": True,
        "teacher_candidates_not_promoted": True,
        "tcp22_edges_expanded_to_nodes": False,
        "physician_report_text_used": False,
    }:
        raise ValueError("report factuality policy drifted")
    receipt = data["receipt_sha256"]
    if not isinstance(receipt, str) or len(receipt) != 64 or receipt != canonical_json_sha256(_replace_receipt_placeholder(data)):
        raise ValueError("report factuality receipt hash drifted")
    return data


def _replace_receipt_placeholder(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = "0" * 64
    return body


__all__ = [
    "REPORT_CLAIM_TYPES",
    "evaluate_evisoz_report_factuality",
    "validate_report_factuality_evaluation",
]
