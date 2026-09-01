#!/usr/bin/env python3
"""Validate the decoupled EEG external knowledge bundle.

The validator intentionally uses only the Python standard library.  It checks
referential integrity, the non-patient-fact boundary, source hashes and a small
set of high-risk clinical invariants.  It does not claim clinical correctness;
cards remain blocked until their source locators and expert reviews are filled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_ROOT = ROOT / "knowledge" / "eeg"

SOURCE_REQUIRED_FIELDS = frozenset(
    {
        "id",
        "collection",
        "title",
        "citation",
        "year",
        "source_type",
        "url",
        "license_note",
        "concepts",
        "summary_zh",
        "application_rules",
        "limitations",
    }
)
CARD_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "card_id",
        "concept_id",
        "claim_type",
        "preferred_terms",
        "aliases",
        "namespace",
        "semantic_layer",
        "statement_zh",
        "applicability",
        "logic",
        "allowed_inferences",
        "forbidden_inferences",
        "report_language",
        "source_refs",
        "review",
    }
)
ACTIVE_COUPLING_TOKENS = (
    "LaBraM",
    "DeepSOZ",
    "Qwen",
    "oracle_onset",
    "private_clinical",
    "multimodal_model",
    "L_total",
    "lambda_",
)
PATIENT_SIDE_LITERAL_RE = re.compile(r"(?:左|右)(?:额|颞|顶|枕|中央|半球)")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"expected JSON object: {path}:{line_number}")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _strings(item)


def _require_string_list(value: Any, context: str, *, minimum: int = 0) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{context} must be a string list with at least {minimum} items")
    return value


def _validate_sources(
    passage_path: Path,
    registry: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[str]]:
    rows = _read_jsonl(passage_path)
    if len(rows) != registry.get("entry_count"):
        raise ValueError("source registry entry_count does not match source passages")
    if _sha256(passage_path) != registry.get("content_sha256"):
        raise ValueError("source passage SHA-256 does not match source registry")
    profiles = registry.get("collection_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("source collection profiles are missing")
    supported_collections = {
        str(item.get("id")) for item in profiles if isinstance(item, Mapping)
    }
    ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if set(row) != SOURCE_REQUIRED_FIELDS:
            raise ValueError(f"source row {index} fields drifted: {sorted(set(row))}")
        source_id = str(row["id"])
        if not source_id or source_id in ids:
            raise ValueError(f"duplicate or empty source ID: {source_id!r}")
        if row["collection"] not in supported_collections:
            raise ValueError(f"unregistered source collection: {row['collection']}")
        if not str(row["url"]).startswith(("https://", "http://")):
            raise ValueError(f"source URL is not auditable: {source_id}")
        _require_string_list(row["concepts"], f"{source_id}.concepts", minimum=3)
        _require_string_list(
            row["application_rules"], f"{source_id}.application_rules", minimum=2
        )
        _require_string_list(row["limitations"], f"{source_id}.limitations", minimum=1)
        ids.add(source_id)
    return rows, ids


def _validate_cards(
    card_path: Path,
    source_ids: set[str],
) -> tuple[list[dict[str, Any]], int]:
    cards = _read_jsonl(card_path)
    if not cards:
        raise ValueError("knowledge card bundle is empty")
    card_ids: set[str] = set()
    pending_locators = 0
    for index, card in enumerate(cards, start=1):
        if set(card) != CARD_REQUIRED_FIELDS:
            raise ValueError(f"card row {index} fields drifted: {sorted(set(card))}")
        if card.get("schema_version") != "eeg_knowledge_card_v2":
            raise ValueError(f"unsupported card schema at row {index}")
        card_id = str(card.get("card_id", ""))
        if not card_id or card_id in card_ids:
            raise ValueError(f"duplicate or empty card ID: {card_id!r}")
        _require_string_list(card.get("allowed_inferences"), f"{card_id}.allowed", minimum=1)
        _require_string_list(
            card.get("forbidden_inferences"), f"{card_id}.forbidden", minimum=1
        )
        logic = card.get("logic")
        if not isinstance(logic, dict) or set(logic) != {
            "required",
            "supporting",
            "exclusions",
            "differentials",
        }:
            raise ValueError(f"{card_id}.logic fields drifted")
        refs = card.get("source_refs")
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"{card_id} has no source references")
        review = card.get("review")
        if not isinstance(review, Mapping):
            raise ValueError(f"{card_id} review metadata is missing")
        for ref in refs:
            if not isinstance(ref, Mapping) or ref.get("source_id") not in source_ids:
                raise ValueError(f"{card_id} references an unknown source")
            if ref.get("locator") is None:
                pending_locators += 1
                if review.get("status") == "clinically_reviewed":
                    raise ValueError(f"reviewed card lacks a precise source locator: {card_id}")
        body = " ".join(_strings(card))
        if PATIENT_SIDE_LITERAL_RE.search(body):
            raise ValueError(f"generic card hard-codes a patient laterality/region: {card_id}")
        card_ids.add(card_id)
    return cards, pending_locators


def _validate_active_decoupling(root: Path, manifest: Mapping[str, Any]) -> None:
    scan_directories = (
        "ontology",
        "terminology",
        "annotation",
        "profiles",
        "cards",
        "reasoning",
        "reporting",
        "schemas",
    )
    for directory in scan_directories:
        for path in (root / directory).rglob("*"):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            hits = [token for token in ACTIVE_COUPLING_TOKENS if token in text]
            if hits:
                raise ValueError(f"active knowledge contains project coupling {hits}: {path}")
    if manifest.get("non_patient_fact_invariant") is not True:
        raise ValueError("knowledge manifest must forbid patient-fact creation")
    if manifest.get("clinical_deployment_allowed") is not False:
        raise ValueError("draft knowledge bundle cannot allow clinical deployment")


def _validate_high_risk_invariants(root: Path) -> None:
    layers = _read_json(root / "ontology" / "semantic_layers.json")
    if layers.get("upgrade_order") != ["OBS", "PAT", "LOC", "CLIN"]:
        raise ValueError("semantic upgrade order drifted")
    states = _read_json(root / "annotation" / "epistemic_states.json")
    state_ids = [item.get("id") for item in states.get("canonical_states", [])]
    if state_ids != ["present", "absent", "not_recorded", "not_assessable", "uncertain"]:
        raise ValueError("canonical epistemic states drifted")
    unknown_alias = states.get("ingestion_aliases", {}).get("unknown", {})
    if (
        "no_assertion" not in unknown_alias.get("allowed_targets", [])
        or unknown_alias.get("unresolved_action")
        != "emit_no_canonical_assertion_and_queue_for_review"
    ):
        raise ValueError("unresolved legacy unknown must fail closed without an assertion")
    critical = _read_json(root / "profiles" / "critical_care_acns_2021.json")
    evolution = critical.get("definite_evolution", {})
    if (
        evolution.get("minimum_sequential_changes") != 2
        or evolution.get("changes_must_be_within_same_category") is not True
        or evolution.get("frequency_change_same_direction_min_hz_each") != 0.5
        or evolution.get("minimum_cycles_per_frequency_morphology_or_location") != 3
        or evolution.get("unchanged_interval_must_be_less_than_minutes") != 5
        or evolution.get("unchanged_interval_five_or_more_minutes_excludes_evolution")
        is not True
        or evolution.get("amplitude_change_alone_is_sufficient") is not False
    ):
        raise ValueError("ACNS definite-evolution safeguards drifted")
    electrographic_seizure = critical.get("electrographic_seizure", {})
    if (
        electrographic_seizure.get("path_a")
        != "epileptiform_discharges_average_frequency_gt_2_5_hz_for_at_least_10_seconds"
        or electrographic_seizure.get("path_b")
        != "any_pattern_with_definite_evolution_for_at_least_10_seconds"
    ):
        raise ValueError("ACNS electrographic-seizure duration safeguards drifted")
    electrographic_status = critical.get("electrographic_status_epilepticus", {})
    if (
        electrographic_status.get("path_a")
        != "electrographic_seizure_continuous_for_at_least_10_minutes"
        or electrographic_status.get("path_b")
        != "electrographic_seizure_burden_at_least_20_percent_of_any_60_minute_period"
    ):
        raise ValueError("ACNS electrographic-status safeguards drifted")
    iic = critical.get("iic", {})
    if (
        iic.get("GRDA_is_included") is not False
        or iic.get("is_diagnosis") is not False
        or iic.get(
            "pattern_must_not_already_qualify_as_electrographic_seizure_or_electrographic_status_epilepticus"
        )
        is not True
    ):
        raise ValueError("ACNS IIC safeguards drifted")
    inference = _read_json(root / "reasoning" / "inference_rules.json")
    if inference.get("clinical_objects_are_not_a_linear_upgrade_chain") is not True:
        raise ValueError("clinical evidence objects were collapsed into a linear upgrade chain")
    transitions = inference.get("transitions", [])
    transition_pairs = {
        (item.get("from"), item.get("to"))
        for item in transitions
        if isinstance(item, Mapping)
    }
    if (
        ("CLIN.noninvasive_soz_hypothesis", "CLIN.invasive_eeg_onset_zone")
        in transition_pairs
        or ("INVASIVE_RECORDING_EVIDENCE", "CLIN.invasive_eeg_onset_zone")
        not in transition_pairs
        or (
            "CLIN.preoperative_epileptogenic_zone_hypothesis",
            "CLIN.outcome_anchored_ez_assessment",
        )
        not in transition_pairs
    ):
        raise ValueError("invasive/EZ evidence relationships drifted")
    forbidden = set(inference.get("forbidden_jumps", []))
    required_forbidden = {
        "phase_reversal_to_cortical_source",
        "IED_to_SOZ_or_EZ",
        "earliest_scalp_visible_change_to_biological_onset",
        "knowledge_source_to_patient_fact",
    }
    if not required_forbidden.issubset(forbidden):
        raise ValueError("high-risk forbidden inference is missing")
    claim_policy = _read_json(root / "reporting" / "claim_policy.json")
    claim_level_ids = {
        item.get("id")
        for item in claim_policy.get("claim_levels", [])
        if isinstance(item, Mapping)
    }
    required_clinical_claims = {
        "invasive_eeg_onset_zone",
        "preoperative_epileptogenic_zone_hypothesis",
        "outcome_anchored_ez_assessment",
    }
    if not required_clinical_claims.issubset(claim_level_ids) or (
        "invasive_soz_or_epileptogenic_zone" in claim_level_ids
    ):
        raise ValueError("invasive SOZ and EZ report claims are not separated")
    clinical_zones = _read_json(root / "ontology" / "clinical_zones.json")
    clinical_object_ids = {
        item.get("id")
        for item in clinical_zones.get("clinical_evidence_objects", [])
        if isinstance(item, Mapping)
    }
    if not {
        "multidisciplinary_noninvasive_soz_hypothesis",
        "invasive_eeg_onset_zone_observation",
        "preoperative_epileptogenic_zone_hypothesis",
        "outcome_anchored_ez_assessment",
    }.issubset(clinical_object_ids):
        raise ValueError("typed clinical SOZ/EZ evidence objects drifted")


def validate_knowledge_system(root: Path = DEFAULT_KNOWLEDGE_ROOT) -> dict[str, Any]:
    root = root.resolve(strict=True)
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema_version") != "eeg_external_knowledge_manifest_v2":
        raise ValueError("unsupported EEG knowledge manifest")
    entrypoints = manifest.get("active_entrypoints")
    schemas = manifest.get("schemas")
    if not isinstance(entrypoints, Mapping) or not isinstance(schemas, Mapping):
        raise ValueError("knowledge manifest entrypoints/schemas are missing")
    expected_paths = {str(relative) for relative in [*entrypoints.values(), *schemas.values()]}
    content_hashes = manifest.get("content_sha256")
    if not isinstance(content_hashes, Mapping) or set(content_hashes) != expected_paths:
        raise ValueError("knowledge manifest content hash map does not cover every active target")
    if _canonical_sha256(content_hashes) != manifest.get("active_bundle_sha256"):
        raise ValueError("knowledge manifest active bundle hash drifted")
    for relative in expected_paths:
        path = root / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"knowledge manifest target does not exist: {path}")
        if _sha256(path) != content_hashes[relative]:
            raise ValueError(f"knowledge manifest content hash drifted: {relative}")
        if path.suffix == ".json":
            _read_json(path)

    registry = _read_json(root / str(entrypoints["source_registry"]))
    source_path = root / str(entrypoints["source_passages"])
    rows, source_ids = _validate_sources(source_path, registry)
    cards, pending_locators = _validate_cards(
        root / str(entrypoints["knowledge_cards"]), source_ids
    )
    _validate_active_decoupling(root, manifest)
    _validate_high_risk_invariants(root)

    reviewed_cards = sum(
        card["review"]["status"] == "clinically_reviewed" for card in cards
    )
    return {
        "status": "valid_research_bundle_clinical_release_blocked",
        "knowledge_version": manifest["knowledge_version"],
        "source_passage_count": len(rows),
        "knowledge_card_count": len(cards),
        "clinically_reviewed_card_count": reviewed_cards,
        "pending_source_locator_count": pending_locators,
        "source_passage_sha256": _sha256(source_path),
        "knowledge_card_sha256": _sha256(root / str(entrypoints["knowledge_cards"])),
        "active_bundle_sha256": manifest["active_bundle_sha256"],
        "patient_fact_creation_allowed": False,
        "clinical_deployment_allowed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-root", type=Path, default=DEFAULT_KNOWLEDGE_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = validate_knowledge_system(args.knowledge_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
