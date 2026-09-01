"""Executable S1-C/A5 sealing and one-shot confirmation contracts.

This module deliberately separates target-free prediction materialization from
reference opening.  It is not a training entry point.  The public/private
development cohorts are rejected by a namespace-aware lineage firewall, and
synthetic rehearsals are permanently distinguishable from real label-fresh
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest

from .geometry import CHANNEL_INDEX, STANDARD_19
from .metrics import DEEPSOZ_STANDARD19_NEIGHBORS
from .risk_controlled_candidate_sets import CandidatePolicy, calibrate_candidate_policy


C18: tuple[str, ...] = tuple(channel for channel in STANDARD_19 if channel != "PZ")
C18_INDEX: dict[str, int] = {channel: index for index, channel in enumerate(C18)}
REFERENCE_STATES: frozenset[str] = frozenset(
    {
        "candidate_positive",
        "reviewed_not_candidate",
        "unknown_not_reviewed",
        "unavailable_signal_or_reference",
    }
)
EVIDENCE_CLASSES: frozenset[str] = frozenset(
    {"synthetic_rehearsal", "real_label_fresh_confirmation"}
)
COHORT_ROLES: frozenset[str] = frozenset({"S1_C", "A5"})
SHA256_HEX_LENGTH = 64


class ConfirmationContractError(ValueError):
    """Raised when a confirmation artifact violates the frozen contract."""


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write a new immutable-style artifact and refuse silent overwrite."""

    if path.exists():
        raise ConfirmationContractError(f"Refusing to overwrite sealed artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise ConfirmationContractError(f"Stale temporary artifact exists: {temporary}")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ConfirmationContractError(f"Expected a JSON object: {path}")
    return value


def load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ConfirmationContractError(
                    f"Expected a JSON object at {path}:{line_number}"
                )
            rows.append(value)
    if not rows:
        raise ConfirmationContractError(f"No rows found: {path}")
    return rows


def _require_exact_keys(
    value: Mapping[str, Any], *, required: set[str], optional: set[str], name: str
) -> None:
    keys = set(value)
    missing = required - keys
    unexpected = keys - required - optional
    if missing:
        raise ConfirmationContractError(f"{name} lacks fields: {sorted(missing)}")
    if unexpected:
        raise ConfirmationContractError(
            f"{name} contains forbidden/unexpected fields: {sorted(unexpected)}"
        )


def _require_nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfirmationContractError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise ConfirmationContractError(f"{name} may not contain a NUL byte")
    return value.strip()


def _require_sha256(value: object, *, name: str) -> str:
    text = _require_nonempty_string(value, name=name).lower()
    if len(text) != SHA256_HEX_LENGTH or any(ch not in "0123456789abcdef" for ch in text):
        raise ConfirmationContractError(f"{name} must be a lowercase SHA-256 hex digest")
    return text


def _parse_time(value: object, *, name: str) -> datetime:
    text = _require_nonempty_string(value, name=name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConfirmationContractError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ConfirmationContractError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def lineage_sha256(source_namespace: str, source_patient_id: str) -> str:
    namespace = _require_nonempty_string(source_namespace, name="source_namespace")
    patient_id = _require_nonempty_string(source_patient_id, name="source_patient_id")
    payload = b"trustworthy-soz-lineage-v1\x00" + namespace.encode("utf-8") + b"\x00" + patient_id.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_path(value: object, path: Sequence[str], *, name: str) -> object:
    current = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise ConfirmationContractError(f"{name} JSON path is missing: {list(path)}")
        current = current[key]
    return current


@dataclass(frozen=True)
class LineageFirewall:
    config_sha256: str
    forbidden_hashes: frozenset[str]
    source_counts: Mapping[str, int]


def load_lineage_firewall(*, workspace: Path, config_path: Path) -> LineageFirewall:
    config = load_json_object(config_path)
    if config.get("schema_version") != "trustworthy_soz_label_fresh_lineage_firewall_v1":
        raise ConfirmationContractError("Unexpected lineage firewall schema")
    sources = config.get("forbidden_cohorts")
    if not isinstance(sources, list) or not sources:
        raise ConfirmationContractError("Lineage firewall must contain forbidden cohorts")

    forbidden: set[str] = set()
    counts: dict[str, int] = {}
    for index, raw in enumerate(sources):
        if not isinstance(raw, Mapping):
            raise ConfirmationContractError("Every forbidden cohort must be an object")
        _require_exact_keys(
            raw,
            required={
                "cohort_id",
                "source_namespace",
                "path",
                "sha256",
                "format",
                "expected_unique_patient_count",
            },
            optional={"json_path", "id_field"},
            name=f"forbidden_cohorts[{index}]",
        )
        cohort_id = _require_nonempty_string(raw["cohort_id"], name="cohort_id")
        namespace = _require_nonempty_string(raw["source_namespace"], name="source_namespace")
        source_path = (workspace / _require_nonempty_string(raw["path"], name="path")).resolve()
        if not source_path.is_file():
            raise ConfirmationContractError(f"Forbidden-cohort roster is missing: {source_path}")
        expected_sha = _require_sha256(raw["sha256"], name="forbidden cohort sha256")
        if file_sha256(source_path) != expected_sha:
            raise ConfirmationContractError(f"Forbidden-cohort roster hash changed: {cohort_id}")

        source_format = raw["format"]
        ids: list[str]
        if source_format == "json_list":
            json_path = raw.get("json_path")
            if not isinstance(json_path, list) or not all(isinstance(item, str) for item in json_path):
                raise ConfirmationContractError("json_list source requires a string json_path")
            source_value = _json_path(load_json_object(source_path), json_path, name=cohort_id)
            if not isinstance(source_value, list):
                raise ConfirmationContractError(f"Forbidden roster is not a list: {cohort_id}")
            ids = [_require_nonempty_string(item, name=f"{cohort_id} patient id") for item in source_value]
        elif source_format == "jsonl_field":
            field = _require_nonempty_string(raw.get("id_field"), name="id_field")
            ids = [
                _require_nonempty_string(row.get(field), name=f"{cohort_id}.{field}")
                for row in load_jsonl_objects(source_path)
            ]
        else:
            raise ConfirmationContractError(f"Unsupported forbidden roster format: {source_format}")

        unique_ids = sorted(set(ids))
        expected_count = int(raw["expected_unique_patient_count"])
        if len(unique_ids) != expected_count:
            raise ConfirmationContractError(
                f"Forbidden roster count changed for {cohort_id}: {len(unique_ids)} != {expected_count}"
            )
        hashes = {lineage_sha256(namespace, patient_id) for patient_id in unique_ids}
        if forbidden.intersection(hashes):
            raise ConfirmationContractError("Forbidden cohort rosters overlap after namespacing")
        forbidden.update(hashes)
        counts[cohort_id] = len(hashes)
    return LineageFirewall(
        config_sha256=file_sha256(config_path),
        forbidden_hashes=frozenset(forbidden),
        source_counts=counts,
    )


def validate_policy_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("schema_version") != "trustworthy_soz_s1c_policy_contract_v1":
        raise ConfirmationContractError("Unexpected S1-C policy contract schema")
    evidence_class = contract.get("evidence_class")
    if evidence_class not in EVIDENCE_CLASSES:
        raise ConfirmationContractError("Invalid policy-contract evidence_class")
    if contract.get("cohort_role") != "S1_C":
        raise ConfirmationContractError("Policy contract must be bound to S1_C")
    if tuple(contract.get("candidate_space", ())) != C18:
        raise ConfirmationContractError("Policy contract candidate space must be canonical C18")
    if contract.get("policy_family") != "abstain_if_margin_below_tau_else_top_k":
        raise ConfirmationContractError("Unexpected candidate-policy family")
    locked_at = _parse_time(contract.get("locked_at"), name="policy contract locked_at")

    raw_grid = contract.get("policy_grid")
    if not isinstance(raw_grid, list) or not raw_grid:
        raise ConfirmationContractError("Policy contract requires a finite non-empty policy_grid")
    policies: list[CandidatePolicy] = []
    for row in raw_grid:
        if not isinstance(row, Mapping) or set(row) != {"tau", "k"}:
            raise ConfirmationContractError("Each policy_grid row must contain exactly tau and k")
        policy = CandidatePolicy(float(row["tau"]), int(row["k"]))
        if policy.k > len(C18):
            raise ConfirmationContractError("Policy k exceeds C18")
        policies.append(policy)
    if len(set(policies)) != len(policies):
        raise ConfirmationContractError("Policy grid contains duplicates")

    required_risks = {
        "strict_miss_all",
        "neighborhood4_miss_all",
        "contralateral_far_top1",
        "spread_top1",
        "candidate_burden",
    }
    risks = contract.get("risk_limits")
    if not isinstance(risks, Mapping) or set(risks) != required_risks:
        raise ConfirmationContractError("Policy contract risk_limits are incomplete")
    risk_limits = {name: float(value) for name, value in risks.items()}
    if any(not 0.0 <= value <= 1.0 for value in risk_limits.values()):
        raise ConfirmationContractError("Risk limits must lie in [0,1]")
    coverage_floor = float(contract.get("coverage_floor"))
    reference_floor = float(contract.get("minimum_primary_evaluable_fraction"))
    familywise_alpha = float(contract.get("familywise_alpha"))
    if not 0.0 <= coverage_floor <= 1.0 or not 0.0 <= reference_floor <= 1.0:
        raise ConfirmationContractError("Coverage floors must lie in [0,1]")
    if not 0.0 < familywise_alpha < 1.0:
        raise ConfirmationContractError("familywise_alpha must lie in (0,1)")
    if contract.get("policy_tie_break") != "highest_coverage_then_lowest_k_then_highest_tau":
        raise ConfirmationContractError("Policy tie-break does not match the implementation")
    if contract.get("score_tie_break") != "canonical_C18_order_label_blind":
        raise ConfirmationContractError("Score tie-break must remain label-blind")

    signoff = contract.get("clinical_and_method_signoff")
    if not isinstance(signoff, list) or len(signoff) < 2:
        raise ConfirmationContractError("Policy contract requires clinical and method signoff")
    roles = set()
    for row in signoff:
        if not isinstance(row, Mapping):
            raise ConfirmationContractError("Signoff row must be an object")
        signer = _require_nonempty_string(row.get("signer_id"), name="signer_id")
        role = _require_nonempty_string(row.get("role"), name="signoff role")
        signed_at = _parse_time(row.get("signed_at"), name=f"signoff {signer} signed_at")
        if signed_at > locked_at:
            raise ConfirmationContractError("Every signoff must precede or equal locked_at")
        roles.add(role)
    if not {"clinical_neurophysiologist", "methodologist"}.issubset(roles):
        raise ConfirmationContractError("Signoff must include clinical_neurophysiologist and methodologist")

    return {
        "evidence_class": evidence_class,
        "locked_at": locked_at,
        "policies": tuple(policies),
        "risk_limits": risk_limits,
        "coverage_floor": coverage_floor,
        "minimum_primary_evaluable_fraction": reference_floor,
        "familywise_alpha": familywise_alpha,
    }


def _validate_score_map(value: object, *, name: str) -> list[float]:
    if not isinstance(value, Mapping) or set(value) != set(C18):
        raise ConfirmationContractError(f"{name} must map exactly the canonical C18 channels")
    scores = [float(value[channel]) for channel in C18]
    if not all(math.isfinite(score) for score in scores):
        raise ConfirmationContractError(f"{name} must contain finite scores")
    return scores


def _rank(scores: Sequence[float]) -> list[int]:
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable").tolist()


def _policy_decision(scores: Sequence[float], *, tau: float, k: int) -> tuple[str, list[str], float]:
    order = _rank(scores)
    margin = float(scores[order[0]] - scores[order[1]])
    if margin < tau:
        return "abstain", [], margin
    return "display_candidate_set", [C18[index] for index in order[:k]], margin


def _selected_policy_from_receipt(receipt: Mapping[str, Any]) -> tuple[float, int]:
    validate_s1c_receipt(receipt, require_real=False)
    if receipt.get("status") != "QUALIFIED":
        raise ConfirmationContractError("A5 sealing requires a QUALIFIED S1-C policy")
    selected = receipt.get("calibration_result", {}).get("selected_policy")
    if not isinstance(selected, Mapping):
        raise ConfirmationContractError("S1-C receipt lacks selected_policy")
    return float(selected["tau"]), int(selected["k"])


def seal_target_free_predictions(
    *,
    rows: Sequence[Mapping[str, Any]],
    cohort_role: str,
    evidence_class: str,
    firewall: LineageFirewall,
    policy_contract: Mapping[str, Any],
    policy_contract_sha256: str,
    model_artifact_sha256: str,
    preprocessing_artifact_sha256: str,
    comparator_artifact_sha256: str | None,
    sealed_at: str,
    s1c_receipt: Mapping[str, Any] | None = None,
    additional_forbidden_lineages: Iterable[str] = (),
) -> dict[str, Any]:
    """Sanitize and seal prediction rows without accepting any target fields."""

    if cohort_role not in COHORT_ROLES:
        raise ConfirmationContractError("cohort_role must be S1_C or A5")
    if evidence_class not in EVIDENCE_CLASSES:
        raise ConfirmationContractError("Invalid evidence_class")
    parsed_policy = validate_policy_contract(policy_contract)
    if parsed_policy["evidence_class"] != evidence_class:
        raise ConfirmationContractError("Policy and prediction evidence_class disagree")
    _require_sha256(policy_contract_sha256, name="policy_contract_sha256")
    _require_sha256(model_artifact_sha256, name="model_artifact_sha256")
    _require_sha256(preprocessing_artifact_sha256, name="preprocessing_artifact_sha256")
    if comparator_artifact_sha256 is not None:
        _require_sha256(comparator_artifact_sha256, name="comparator_artifact_sha256")
    sealed_time = _parse_time(sealed_at, name="sealed_at")
    if sealed_time < parsed_policy["locked_at"]:
        raise ConfirmationContractError("Predictions cannot be sealed before policy lock")
    if not rows:
        raise ConfirmationContractError("Prediction seal requires at least one patient")

    forbidden = set(firewall.forbidden_hashes)
    for digest in additional_forbidden_lineages:
        forbidden.add(_require_sha256(digest, name="additional forbidden lineage"))
    if cohort_role == "A5":
        if s1c_receipt is None:
            raise ConfirmationContractError("A5 sealing requires the S1-C calibration receipt")
        validate_s1c_receipt(s1c_receipt, require_real=evidence_class == "real_label_fresh_confirmation")
        if s1c_receipt.get("evidence_class") != evidence_class:
            raise ConfirmationContractError("S1-C and A5 evidence classes disagree")
        forbidden.update(_receipt_lineages(s1c_receipt))
        selected_tau, selected_k = _selected_policy_from_receipt(s1c_receipt)
    else:
        if s1c_receipt is not None:
            raise ConfirmationContractError("S1_C prediction seal cannot consume an S1-C result")
        selected_tau = selected_k = None

    sanitized: list[dict[str, Any]] = []
    seen_confirmation_ids: set[str] = set()
    seen_lineages: set[str] = set()
    namespaces: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ConfirmationContractError("Prediction rows must be objects")
        common_required = {
            "schema_version",
            "confirmation_id",
            "source_namespace",
            "source_patient_id",
            "scores",
            "input_receipt_sha256",
        }
        if cohort_role == "A5":
            required = common_required | {
                "comparator_scores",
                "decision",
                "candidate_channels",
                "report_sha256",
            }
            optional: set[str] = set()
        else:
            required = common_required
            optional = {"comparator_scores"}
        _require_exact_keys(row, required=required, optional=optional, name=f"prediction row {index}")
        if row["schema_version"] != "trustworthy_soz_target_free_prediction_row_v1":
            raise ConfirmationContractError("Unexpected target-free prediction row schema")
        confirmation_id = _require_nonempty_string(row["confirmation_id"], name="confirmation_id")
        namespace = _require_nonempty_string(row["source_namespace"], name="source_namespace")
        patient_id = _require_nonempty_string(row["source_patient_id"], name="source_patient_id")
        lineage = lineage_sha256(namespace, patient_id)
        if confirmation_id in seen_confirmation_ids or lineage in seen_lineages:
            raise ConfirmationContractError("Prediction seal requires one row per unique patient")
        if lineage in forbidden:
            raise ConfirmationContractError(
                f"Lineage firewall rejected consumed or prior-stage patient: {confirmation_id}"
            )
        seen_confirmation_ids.add(confirmation_id)
        seen_lineages.add(lineage)
        namespaces.add(namespace)
        scores = _validate_score_map(row["scores"], name="scores")
        comparator_scores = None
        if row.get("comparator_scores") is not None:
            comparator_scores = _validate_score_map(row["comparator_scores"], name="comparator_scores")
        input_receipt = _require_sha256(row["input_receipt_sha256"], name="input_receipt_sha256")

        sealed_row: dict[str, Any] = {
            "confirmation_id": confirmation_id,
            "lineage_sha256": lineage,
            "scores": scores,
            "comparator_scores": comparator_scores,
            "input_receipt_sha256": input_receipt,
        }
        if cohort_role == "A5":
            assert selected_tau is not None and selected_k is not None
            expected_decision, expected_candidates, margin = _policy_decision(
                scores, tau=selected_tau, k=selected_k
            )
            candidates = row["candidate_channels"]
            if not isinstance(candidates, list) or not all(channel in C18 for channel in candidates):
                raise ConfirmationContractError("candidate_channels must be a C18 list")
            if row["decision"] != expected_decision or candidates != expected_candidates:
                raise ConfirmationContractError(
                    "A5 candidate action does not match the sealed S1-C policy"
                )
            sealed_row.update(
                {
                    "decision": expected_decision,
                    "candidate_channels": expected_candidates,
                    "margin": margin,
                    "report_sha256": _require_sha256(row["report_sha256"], name="report_sha256"),
                }
            )
        sanitized.append(sealed_row)

    sanitized.sort(key=lambda row: row["confirmation_id"])
    sealed_payload: dict[str, Any] = {
        "schema_version": "trustworthy_soz_target_blind_prediction_payload_v1",
        "cohort_role": cohort_role,
        "evidence_class": evidence_class,
        "sealed_at": sealed_at,
        "candidate_space": list(C18),
        "patient_count": len(sanitized),
        "model_artifact_sha256": model_artifact_sha256,
        "preprocessing_artifact_sha256": preprocessing_artifact_sha256,
        "comparator_artifact_sha256": comparator_artifact_sha256,
        "policy_contract_sha256": policy_contract_sha256,
        "s1c_receipt_payload_sha256": (
            None if s1c_receipt is None else s1c_receipt["receipt_payload_sha256"]
        ),
        "rows": sanitized,
        "access_receipt": {
            "target_fields_accepted": False,
            "reference_loaded": False,
            "training_performed": False,
            "candidate_action_derived_from_sealed_policy": cohort_role == "A5",
            "llm_changed_localization": False,
        },
        "lineage_firewall": {
            "config_sha256": firewall.config_sha256,
            "forbidden_source_counts": dict(firewall.source_counts),
            "additional_forbidden_count": len(forbidden) - len(firewall.forbidden_hashes),
            "checked_patient_count": len(sanitized),
            "overlap_count": 0,
            "source_namespaces": sorted(namespaces),
        },
    }
    return {
        "schema_version": "trustworthy_soz_target_blind_prediction_seal_v1",
        "status": "SEALED_TARGET_BLIND_PREDICTIONS",
        "sealed_payload": sealed_payload,
        "seal_payload_sha256": canonical_sha256(sealed_payload),
    }


def validate_prediction_seal(
    seal: Mapping[str, Any], *, expected_role: str | None = None, require_real: bool = False
) -> dict[str, Any]:
    if seal.get("schema_version") != "trustworthy_soz_target_blind_prediction_seal_v1":
        raise ConfirmationContractError("Unexpected prediction seal schema")
    if seal.get("status") != "SEALED_TARGET_BLIND_PREDICTIONS":
        raise ConfirmationContractError("Prediction seal is not finalized")
    payload = seal.get("sealed_payload")
    if not isinstance(payload, Mapping):
        raise ConfirmationContractError("Prediction seal lacks sealed_payload")
    expected_digest = _require_sha256(seal.get("seal_payload_sha256"), name="seal_payload_sha256")
    if canonical_sha256(payload) != expected_digest:
        raise ConfirmationContractError("Prediction seal payload hash mismatch")
    role = payload.get("cohort_role")
    if role not in COHORT_ROLES or (expected_role is not None and role != expected_role):
        raise ConfirmationContractError("Prediction seal cohort role mismatch")
    evidence_class = payload.get("evidence_class")
    if evidence_class not in EVIDENCE_CLASSES:
        raise ConfirmationContractError("Prediction seal evidence class is invalid")
    if require_real and evidence_class != "real_label_fresh_confirmation":
        raise ConfirmationContractError("Synthetic rehearsal cannot satisfy a real confirmation gate")
    if tuple(payload.get("candidate_space", ())) != C18:
        raise ConfirmationContractError("Prediction seal candidate space changed")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != payload.get("patient_count") or not rows:
        raise ConfirmationContractError("Prediction seal patient count mismatch")
    ids = [row.get("confirmation_id") for row in rows if isinstance(row, Mapping)]
    lineages = [row.get("lineage_sha256") for row in rows if isinstance(row, Mapping)]
    if len(ids) != len(rows) or len(set(ids)) != len(ids) or len(set(lineages)) != len(lineages):
        raise ConfirmationContractError("Prediction seal patient identities are not unique")
    if payload.get("access_receipt", {}).get("target_fields_accepted") is not False:
        raise ConfirmationContractError("Prediction seal did not preserve target blindness")
    if payload.get("lineage_firewall", {}).get("overlap_count") != 0:
        raise ConfirmationContractError("Prediction seal contains forbidden lineage overlap")
    return dict(payload)


def _validate_reference_row(row: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    _require_exact_keys(
        row,
        required={
            "confirmation_id",
            "lineage_sha256",
            "c18_states",
            "pz_state",
            "outside_primary_positive_labels",
            "spread_review",
            "reference_source",
            "reviewer_count",
            "adjudication_status",
            "reference_locked_at",
        },
        optional=set(),
        name=name,
    )
    confirmation_id = _require_nonempty_string(row["confirmation_id"], name="confirmation_id")
    lineage = _require_sha256(row["lineage_sha256"], name="lineage_sha256")
    states = row["c18_states"]
    if not isinstance(states, Mapping) or set(states) != set(C18):
        raise ConfirmationContractError("c18_states must cover exactly C18")
    if any(state not in REFERENCE_STATES for state in states.values()):
        raise ConfirmationContractError("c18_states contains an invalid four-state value")
    if row["pz_state"] not in REFERENCE_STATES:
        raise ConfirmationContractError("pz_state is invalid")
    outside = row["outside_primary_positive_labels"]
    if not isinstance(outside, list) or not all(isinstance(item, str) and item.strip() for item in outside):
        raise ConfirmationContractError("outside_primary_positive_labels must be a string list")
    spread_review = row["spread_review"]
    if not isinstance(spread_review, Mapping) or set(spread_review) != {"state", "electrodes"}:
        raise ConfirmationContractError("spread_review must contain state and electrodes")
    if spread_review["state"] not in {"reviewed", "not_reviewed", "unavailable"}:
        raise ConfirmationContractError("spread_review state is invalid")
    spread = spread_review["electrodes"]
    if not isinstance(spread, list) or len(set(spread)) != len(spread) or any(ch not in C18 for ch in spread):
        raise ConfirmationContractError("spread electrodes must be a unique C18 list")
    if spread_review["state"] != "reviewed" and spread:
        raise ConfirmationContractError("Unreviewed/unavailable spread must not contain electrodes")
    positive = {channel for channel, state in states.items() if state == "candidate_positive"}
    if positive.intersection(spread):
        raise ConfirmationContractError("An electrode cannot be both SOZ-reference positive and spread")
    reviewer_count = int(row["reviewer_count"])
    if reviewer_count < 1:
        raise ConfirmationContractError("reviewer_count must be positive")
    adjudication = row["adjudication_status"]
    if adjudication not in {"single_review", "independent_agreement", "adjudicated_disagreement"}:
        raise ConfirmationContractError("adjudication_status is invalid")
    _parse_time(row["reference_locked_at"], name="reference_locked_at")
    full_review = all(
        state in {"candidate_positive", "reviewed_not_candidate"} for state in states.values()
    )
    primary_evaluable = full_review and bool(positive)
    return {
        "confirmation_id": confirmation_id,
        "lineage_sha256": lineage,
        "states": dict(states),
        "strict_positive": positive,
        "spread": set(spread),
        "spread_known": spread_review["state"] == "reviewed",
        "primary_evaluable": primary_evaluable,
        "reference_source": _require_nonempty_string(row["reference_source"], name="reference_source"),
        "reviewer_count": reviewer_count,
        "adjudication_status": adjudication,
        "pz_state": row["pz_state"],
        "outside_primary_positive_labels": list(outside),
    }


def validate_reference_bundle(
    bundle: Mapping[str, Any], *, prediction_seal: Mapping[str, Any]
) -> dict[str, Any]:
    if bundle.get("schema_version") != "trustworthy_soz_four_state_reference_bundle_v1":
        raise ConfirmationContractError("Unexpected four-state reference schema")
    payload = bundle.get("reference_payload")
    if not isinstance(payload, Mapping):
        raise ConfirmationContractError("Reference bundle lacks reference_payload")
    if payload.get("schema_version") != "trustworthy_soz_four_state_reference_payload_v1":
        raise ConfirmationContractError("Unexpected four-state reference payload schema")
    if payload.get("cohort_role") not in COHORT_ROLES:
        raise ConfirmationContractError("Reference cohort role is invalid")
    if payload.get("evidence_class") not in EVIDENCE_CLASSES:
        raise ConfirmationContractError("Reference evidence class is invalid")
    digest = _require_sha256(bundle.get("reference_payload_sha256"), name="reference_payload_sha256")
    if canonical_sha256(payload) != digest:
        raise ConfirmationContractError("Reference payload hash mismatch")
    seal_payload = validate_prediction_seal(prediction_seal, expected_role=payload.get("cohort_role"))
    if payload.get("evidence_class") != seal_payload.get("evidence_class"):
        raise ConfirmationContractError("Reference and prediction evidence classes disagree")
    if payload.get("prediction_seal_payload_sha256") != prediction_seal.get("seal_payload_sha256"):
        raise ConfirmationContractError("Reference did not bind the prediction seal")
    opened_at = _parse_time(payload.get("opened_at"), name="reference opened_at")
    sealed_at = _parse_time(seal_payload.get("sealed_at"), name="prediction sealed_at")
    if opened_at <= sealed_at:
        raise ConfirmationContractError("Reference must open after prediction sealing")
    custody = payload.get("custody_attestation")
    required_attestation = {
        "independent_custodian",
        "labels_hidden_until_prediction_seal",
        "opened_once",
        "model_team_could_not_modify_seal_after_opening",
    }
    if not isinstance(custody, Mapping) or set(custody) != required_attestation:
        raise ConfirmationContractError("Reference custody attestation is incomplete")
    if not all(custody[field] is True for field in required_attestation):
        raise ConfirmationContractError("Every reference custody attestation must be true")
    _require_nonempty_string(payload.get("custodian_id"), name="custodian_id")

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ConfirmationContractError("Reference bundle requires patient rows")
    validated = [_validate_reference_row(row, name=f"reference row {index}") for index, row in enumerate(rows)]
    reference_by_id = {row["confirmation_id"]: row for row in validated}
    if len(reference_by_id) != len(validated):
        raise ConfirmationContractError("Reference confirmation IDs are not unique")
    seal_by_id = {row["confirmation_id"]: row for row in seal_payload["rows"]}
    if set(reference_by_id) != set(seal_by_id):
        raise ConfirmationContractError("Reference and prediction patient rosters differ")
    for confirmation_id, reference in reference_by_id.items():
        if reference["lineage_sha256"] != seal_by_id[confirmation_id]["lineage_sha256"]:
            raise ConfirmationContractError("Reference and prediction lineage hashes differ")
    return {
        "payload": dict(payload),
        "rows": validated,
        "by_id": reference_by_id,
        "seal_payload": seal_payload,
    }


def _reference_arrays(
    *, prediction_rows: Sequence[Mapping[str, Any]], reference_by_id: Mapping[str, Mapping[str, Any]]
) -> tuple[list[Mapping[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    evaluable_rows = [row for row in prediction_rows if reference_by_id[row["confirmation_id"]]["primary_evaluable"]]
    if not evaluable_rows:
        raise ConfirmationContractError("No fully reviewed C18 patient has an in-head positive")
    scores = np.asarray([row["scores"] for row in evaluable_rows], dtype=np.float64)
    strict = np.zeros_like(scores, dtype=bool)
    relaxed = np.zeros_like(scores, dtype=bool)
    contralateral = np.zeros_like(scores, dtype=bool)
    spread = np.zeros_like(scores, dtype=bool)
    spread_known = np.zeros(len(evaluable_rows), dtype=bool)

    left_channels = {"FP1", "F7", "F3", "T7", "C3", "P7", "P3", "O1"}
    right_channels = {"FP2", "F4", "F8", "C4", "T8", "P4", "P8", "O2"}
    for row_index, prediction in enumerate(evaluable_rows):
        reference = reference_by_id[prediction["confirmation_id"]]
        positives = set(reference["strict_positive"])
        spread_set = set(reference["spread"])
        for channel in positives:
            strict[row_index, C18_INDEX[channel]] = True
        acceptable = set(positives)
        if len(positives) <= 4:
            for channel in positives:
                standard_index = CHANNEL_INDEX[channel]
                acceptable.update(
                    STANDARD_19[index]
                    for index in DEEPSOZ_STANDARD19_NEIGHBORS[standard_index]
                    if STANDARD_19[index] in C18_INDEX
                )
        acceptable.difference_update(spread_set)
        for channel in acceptable:
            relaxed[row_index, C18_INDEX[channel]] = True
        for channel in spread_set:
            spread[row_index, C18_INDEX[channel]] = True
        spread_known[row_index] = bool(reference["spread_known"])

        if positives and positives.issubset(left_channels):
            for channel in right_channels - acceptable:
                contralateral[row_index, C18_INDEX[channel]] = True
        elif positives and positives.issubset(right_channels):
            for channel in left_channels - acceptable:
                contralateral[row_index, C18_INDEX[channel]] = True
    return evaluable_rows, scores, strict, relaxed, contralateral, spread, spread_known


def calibrate_s1c_from_sealed_predictions(
    *,
    prediction_seal: Mapping[str, Any],
    reference_bundle: Mapping[str, Any],
    policy_contract: Mapping[str, Any],
    policy_contract_sha256: str,
    calibrated_at: str,
) -> dict[str, Any]:
    seal_payload = validate_prediction_seal(prediction_seal, expected_role="S1_C")
    parsed_policy = validate_policy_contract(policy_contract)
    if seal_payload["policy_contract_sha256"] != policy_contract_sha256:
        raise ConfirmationContractError("S1-C seal and policy contract hash disagree")
    if file_or_canonical_contract_sha256(policy_contract) != policy_contract_sha256:
        raise ConfirmationContractError("Provided policy contract digest is incorrect")
    validated_reference = validate_reference_bundle(reference_bundle, prediction_seal=prediction_seal)
    calibrated_time = _parse_time(calibrated_at, name="calibrated_at")
    opened_time = _parse_time(validated_reference["payload"]["opened_at"], name="opened_at")
    if calibrated_time < opened_time:
        raise ConfirmationContractError("Calibration result cannot precede reference opening")
    prediction_rows = seal_payload["rows"]
    arrays = _reference_arrays(
        prediction_rows=prediction_rows, reference_by_id=validated_reference["by_id"]
    )
    evaluable_rows, scores, strict, relaxed, contralateral, spread, spread_known = arrays
    evaluable_fraction = len(evaluable_rows) / len(prediction_rows)
    if evaluable_fraction < parsed_policy["minimum_primary_evaluable_fraction"]:
        calibration_result: dict[str, Any] = {
            "schema_version": "trustworthy_soz_risk_controlled_candidate_policy_v1",
            "status": "NO_QUALIFIED_POLICY",
            "action": "fail_closed_insufficient_reference_coverage",
            "selected_policy": None,
            "patient_count": len(evaluable_rows),
            "policy_rows": [],
        }
    else:
        calibration_result = calibrate_candidate_policy(
            scores=scores,
            strict_positive=strict,
            relaxed_acceptable=relaxed,
            contralateral_far=contralateral,
            spread=spread,
            spread_known=spread_known,
            policies=parsed_policy["policies"],
            risk_limits=parsed_policy["risk_limits"],
            coverage_floor=parsed_policy["coverage_floor"],
            familywise_alpha=parsed_policy["familywise_alpha"],
        )
    status = calibration_result["status"]
    receipt_payload: dict[str, Any] = {
        "schema_version": "trustworthy_soz_s1c_calibration_receipt_payload_v1",
        "status": status,
        "evidence_class": seal_payload["evidence_class"],
        "calibrated_at": calibrated_at,
        "prediction_seal_payload_sha256": prediction_seal["seal_payload_sha256"],
        "reference_payload_sha256": reference_bundle["reference_payload_sha256"],
        "policy_contract_sha256": policy_contract_sha256,
        "enrolled_patient_count": len(prediction_rows),
        "primary_evaluable_patient_count": len(evaluable_rows),
        "primary_evaluable_fraction": evaluable_fraction,
        "lineage_sha256": [row["lineage_sha256"] for row in prediction_rows],
        "calibration_result": calibration_result,
        "access_receipt": {
            "model_weights_updated": False,
            "features_updated": False,
            "report_semantics_updated": False,
            "policy_selected_only_from_prespecified_finite_family": True,
            "synthetic_receipt_can_satisfy_readiness": False,
        },
    }
    return {
        "schema_version": "trustworthy_soz_s1c_calibration_receipt_v1",
        "status": status,
        "evidence_class": seal_payload["evidence_class"],
        "receipt_payload": receipt_payload,
        "receipt_payload_sha256": canonical_sha256(receipt_payload),
        "calibration_result": calibration_result,
    }


def file_or_canonical_contract_sha256(contract: Mapping[str, Any]) -> str:
    """Canonical digest used when a contract object, rather than bytes, is supplied."""

    return canonical_sha256(contract)


def _receipt_lineages(receipt: Mapping[str, Any]) -> set[str]:
    payload = receipt.get("receipt_payload")
    if not isinstance(payload, Mapping):
        raise ConfirmationContractError("S1-C receipt lacks receipt_payload")
    values = payload.get("lineage_sha256")
    if not isinstance(values, list):
        raise ConfirmationContractError("S1-C receipt lacks lineage roster")
    return {_require_sha256(value, name="S1-C lineage") for value in values}


def validate_s1c_receipt(receipt: Mapping[str, Any], *, require_real: bool) -> dict[str, Any]:
    if receipt.get("schema_version") != "trustworthy_soz_s1c_calibration_receipt_v1":
        raise ConfirmationContractError("Unexpected S1-C receipt schema")
    payload = receipt.get("receipt_payload")
    if not isinstance(payload, Mapping):
        raise ConfirmationContractError("S1-C receipt lacks receipt_payload")
    digest = _require_sha256(receipt.get("receipt_payload_sha256"), name="receipt_payload_sha256")
    if canonical_sha256(payload) != digest:
        raise ConfirmationContractError("S1-C receipt payload hash mismatch")
    if receipt.get("status") != payload.get("status"):
        raise ConfirmationContractError("S1-C receipt status mismatch")
    evidence_class = receipt.get("evidence_class")
    if evidence_class != payload.get("evidence_class") or evidence_class not in EVIDENCE_CLASSES:
        raise ConfirmationContractError("S1-C receipt evidence class mismatch")
    if require_real and evidence_class != "real_label_fresh_confirmation":
        raise ConfirmationContractError("Synthetic S1-C receipt cannot close a real readiness gate")
    lineages = _receipt_lineages(receipt)
    if len(lineages) != payload.get("enrolled_patient_count"):
        raise ConfirmationContractError("S1-C lineage roster count mismatch")
    result = receipt.get("calibration_result")
    if not isinstance(result, Mapping) or result != payload.get("calibration_result"):
        raise ConfirmationContractError("S1-C calibration result mismatch")
    if result.get("status") != receipt.get("status"):
        raise ConfirmationContractError("S1-C nested status mismatch")
    access = payload.get("access_receipt", {})
    if access.get("model_weights_updated") is not False or access.get("features_updated") is not False:
        raise ConfirmationContractError("S1-C receipt does not prove frozen-model calibration")
    return dict(payload)


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _binary_metric(values: Sequence[bool]) -> dict[str, Any]:
    successes = int(sum(values))
    total = len(values)
    return {
        "successes": successes,
        "total": total,
        "estimate": None if total == 0 else successes / total,
        "wilson_95_ci": _wilson(successes, total),
    }


def open_a5_confirmation(
    *,
    prediction_seal: Mapping[str, Any],
    reference_bundle: Mapping[str, Any],
    s1c_receipt: Mapping[str, Any],
    opened_analysis_at: str,
) -> dict[str, Any]:
    seal_payload = validate_prediction_seal(prediction_seal, expected_role="A5")
    validate_s1c_receipt(
        s1c_receipt,
        require_real=seal_payload["evidence_class"] == "real_label_fresh_confirmation",
    )
    if seal_payload.get("s1c_receipt_payload_sha256") != s1c_receipt.get("receipt_payload_sha256"):
        raise ConfirmationContractError("A5 seal is not bound to the supplied S1-C receipt")
    validated_reference = validate_reference_bundle(reference_bundle, prediction_seal=prediction_seal)
    analysis_time = _parse_time(opened_analysis_at, name="opened_analysis_at")
    reference_time = _parse_time(validated_reference["payload"]["opened_at"], name="opened_at")
    if analysis_time < reference_time:
        raise ConfirmationContractError("A5 analysis cannot precede reference opening")

    prediction_rows = seal_payload["rows"]
    reference_by_id = validated_reference["by_id"]
    enrolled_rows: list[dict[str, Any]] = []
    evaluable_rows: list[dict[str, Any]] = []
    for prediction in prediction_rows:
        reference = reference_by_id[prediction["confirmation_id"]]
        enrollment_row = {
            "confirmation_id": prediction["confirmation_id"],
            "primary_evaluable": bool(reference["primary_evaluable"]),
            "decision": prediction["decision"],
        }
        enrolled_rows.append(enrollment_row)
        if not reference["primary_evaluable"]:
            continue
        scores = list(map(float, prediction["scores"]))
        comparator_scores = prediction.get("comparator_scores")
        if comparator_scores is None:
            raise ConfirmationContractError("A5 comparison requires sealed comparator scores")
        order = _rank(scores)
        comparator_order = _rank(comparator_scores)
        top1 = C18[order[0]]
        comparator_top1 = C18[comparator_order[0]]
        strict_positive = set(reference["strict_positive"])
        spread = set(reference["spread"])
        acceptable = set(strict_positive)
        if len(strict_positive) <= 4:
            for channel in strict_positive:
                standard_index = CHANNEL_INDEX[channel]
                acceptable.update(
                    STANDARD_19[index]
                    for index in DEEPSOZ_STANDARD19_NEIGHBORS[standard_index]
                    if STANDARD_19[index] in C18_INDEX
                )
        acceptable.difference_update(spread)
        strict_hit = top1 in strict_positive
        relaxed_hit = top1 in acceptable
        comparator_strict = comparator_top1 in strict_positive
        comparator_relaxed = comparator_top1 in acceptable
        positive_ranks = [rank + 1 for rank, index in enumerate(order) if C18[index] in strict_positive]
        candidate_set = set(prediction["candidate_channels"])
        display = prediction["decision"] == "display_candidate_set"
        left = {"FP1", "F7", "F3", "T7", "C3", "P7", "P3", "O1"}
        right = {"FP2", "F4", "F8", "C4", "T8", "P4", "P8", "O2"}
        contralateral = (
            strict_positive.issubset(left) and top1 in right and not relaxed_hit
        ) or (
            strict_positive.issubset(right) and top1 in left and not relaxed_hit
        )
        evaluable_rows.append(
            {
                **enrollment_row,
                "top1": top1,
                "comparator_top1": comparator_top1,
                "strict_hit": strict_hit,
                "neighborhood4_hit": relaxed_hit,
                "comparator_strict_hit": comparator_strict,
                "comparator_neighborhood4_hit": comparator_relaxed,
                "far_top1": not relaxed_hit,
                "contralateral_far_top1": bool(contralateral),
                "spread_top1": top1 in spread if reference["spread_known"] else None,
                "candidate_miss_all_strict": (
                    candidate_set.isdisjoint(strict_positive) if display else None
                ),
                "candidate_miss_all_neighborhood4": (
                    candidate_set.isdisjoint(acceptable) if display else None
                ),
                "candidate_count": len(candidate_set),
                "hit_at_3": min(positive_ranks) <= 3,
                "hit_at_5": min(positive_ranks) <= 5,
                "reciprocal_rank": 1.0 / min(positive_ranks),
            }
        )

    model_strict = [row["strict_hit"] for row in evaluable_rows]
    comparator_strict = [row["comparator_strict_hit"] for row in evaluable_rows]
    model_relaxed = [row["neighborhood4_hit"] for row in evaluable_rows]
    comparator_relaxed = [row["comparator_neighborhood4_hit"] for row in evaluable_rows]
    display_rows = [row for row in evaluable_rows if row["decision"] == "display_candidate_set"]
    spread_rows = [row for row in evaluable_rows if row["spread_top1"] is not None]
    b = sum(model and not comparator for model, comparator in zip(model_strict, comparator_strict))
    c = sum(comparator and not model for model, comparator in zip(model_strict, comparator_strict))
    discordant = b + c
    mcnemar_p = 1.0 if discordant == 0 else float(binomtest(min(b, c), discordant, 0.5).pvalue)
    metrics = {
        "enrollment": {
            "enrolled_patient_count": len(enrolled_rows),
            "primary_evaluable_patient_count": len(evaluable_rows),
            "primary_evaluable_fraction": len(evaluable_rows) / len(enrolled_rows),
        },
        "full_coverage": {
            "strict_top1": _binary_metric(model_strict),
            "neighborhood4_top1": _binary_metric(model_relaxed),
            "far_top1": _binary_metric([row["far_top1"] for row in evaluable_rows]),
            "contralateral_far_top1": _binary_metric(
                [row["contralateral_far_top1"] for row in evaluable_rows]
            ),
            "hit_at_3": _binary_metric([row["hit_at_3"] for row in evaluable_rows]),
            "hit_at_5": _binary_metric([row["hit_at_5"] for row in evaluable_rows]),
            "mean_reciprocal_rank": float(np.mean([row["reciprocal_rank"] for row in evaluable_rows])),
            "spread_top1_on_reviewed_denominator": _binary_metric(
                [bool(row["spread_top1"]) for row in spread_rows]
            ),
        },
        "selected_policy": {
            "display_coverage": len(display_rows) / len(evaluable_rows),
            "display_count": len(display_rows),
            "abstain_count": len(evaluable_rows) - len(display_rows),
            "strict_miss_all": _binary_metric(
                [bool(row["candidate_miss_all_strict"]) for row in display_rows]
            ),
            "neighborhood4_miss_all": _binary_metric(
                [bool(row["candidate_miss_all_neighborhood4"]) for row in display_rows]
            ),
            "candidate_count_mean": (
                None if not display_rows else float(np.mean([row["candidate_count"] for row in display_rows]))
            ),
        },
        "paired_deepsoz_comparator": {
            "model_strict_top1": _binary_metric(model_strict),
            "comparator_strict_top1": _binary_metric(comparator_strict),
            "model_neighborhood4_top1": _binary_metric(model_relaxed),
            "comparator_neighborhood4_top1": _binary_metric(comparator_relaxed),
            "strict_difference": float(np.mean(model_strict) - np.mean(comparator_strict)),
            "discordant_model_only_correct": int(b),
            "discordant_comparator_only_correct": int(c),
            "exact_mcnemar_two_sided_p": mcnemar_p,
        },
    }
    result_payload: dict[str, Any] = {
        "schema_version": "trustworthy_soz_a5_opened_confirmation_payload_v1",
        "status": "COMPLETED_ONE_SHOT_CONFIRMATION",
        "evidence_class": seal_payload["evidence_class"],
        "opened_analysis_at": opened_analysis_at,
        "prediction_seal_payload_sha256": prediction_seal["seal_payload_sha256"],
        "reference_payload_sha256": reference_bundle["reference_payload_sha256"],
        "s1c_receipt_payload_sha256": s1c_receipt["receipt_payload_sha256"],
        "primary_endpoint": "patient_equal_strict_positive_set_top1",
        "metrics": metrics,
        "patient_rows": evaluable_rows,
        "non_evaluable_rows": [row for row in enrolled_rows if not row["primary_evaluable"]],
        "access_receipt": {
            "one_shot_prespecified_analysis_only": True,
            "model_or_policy_reselected": False,
            "reports_rewritten_after_reference_open": False,
            "unknown_treated_as_negative": False,
            "spread_treated_as_soz_success": False,
            "synthetic_result_can_satisfy_readiness": False,
        },
    }
    return {
        "schema_version": "trustworthy_soz_a5_opened_confirmation_result_v1",
        "status": "COMPLETED_ONE_SHOT_CONFIRMATION",
        "evidence_class": seal_payload["evidence_class"],
        "result_payload": result_payload,
        "result_payload_sha256": canonical_sha256(result_payload),
    }


def validate_a5_result(
    result: Mapping[str, Any],
    *,
    prediction_seal: Mapping[str, Any],
    s1c_receipt: Mapping[str, Any],
    require_real: bool,
) -> dict[str, Any]:
    if result.get("schema_version") != "trustworthy_soz_a5_opened_confirmation_result_v1":
        raise ConfirmationContractError("Unexpected A5 result schema")
    if result.get("status") != "COMPLETED_ONE_SHOT_CONFIRMATION":
        raise ConfirmationContractError("A5 result is incomplete")
    payload = result.get("result_payload")
    if not isinstance(payload, Mapping):
        raise ConfirmationContractError("A5 result lacks result_payload")
    digest = _require_sha256(result.get("result_payload_sha256"), name="result_payload_sha256")
    if canonical_sha256(payload) != digest:
        raise ConfirmationContractError("A5 result payload hash mismatch")
    seal_payload = validate_prediction_seal(
        prediction_seal, expected_role="A5", require_real=require_real
    )
    validate_s1c_receipt(s1c_receipt, require_real=require_real)
    if payload.get("evidence_class") != seal_payload.get("evidence_class"):
        raise ConfirmationContractError("A5 result evidence class mismatch")
    if require_real and payload.get("evidence_class") != "real_label_fresh_confirmation":
        raise ConfirmationContractError("Synthetic A5 result cannot close readiness")
    if payload.get("prediction_seal_payload_sha256") != prediction_seal.get("seal_payload_sha256"):
        raise ConfirmationContractError("A5 result is not bound to prediction seal")
    if payload.get("s1c_receipt_payload_sha256") != s1c_receipt.get("receipt_payload_sha256"):
        raise ConfirmationContractError("A5 result is not bound to S1-C receipt")
    if payload.get("primary_endpoint") != "patient_equal_strict_positive_set_top1":
        raise ConfirmationContractError("A5 primary endpoint changed")
    access = payload.get("access_receipt", {})
    if access.get("model_or_policy_reselected") is not False:
        raise ConfirmationContractError("A5 result indicates post-open reselection")
    if access.get("reports_rewritten_after_reference_open") is not False:
        raise ConfirmationContractError("A5 result indicates post-open report rewriting")
    return dict(payload)


__all__ = [
    "C18",
    "COHORT_ROLES",
    "EVIDENCE_CLASSES",
    "REFERENCE_STATES",
    "ConfirmationContractError",
    "LineageFirewall",
    "calibrate_s1c_from_sealed_predictions",
    "canonical_json_bytes",
    "canonical_sha256",
    "file_sha256",
    "lineage_sha256",
    "load_json_object",
    "load_jsonl_objects",
    "load_lineage_firewall",
    "open_a5_confirmation",
    "seal_target_free_predictions",
    "utc_now_iso",
    "validate_a5_result",
    "validate_policy_contract",
    "validate_prediction_seal",
    "validate_reference_bundle",
    "validate_s1c_receipt",
    "write_new_canonical_json",
]
