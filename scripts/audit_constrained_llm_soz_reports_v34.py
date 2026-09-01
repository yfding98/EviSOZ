#!/usr/bin/env python3
"""Independently revalidate a completed constrained-language report bundle."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.constrained_llm_reporting import (
    OUTPUT_SCHEMA,
    build_fact_inventory,
    deterministic_fallback_payload,
    load_reporting_knowledge,
    select_reporting_knowledge,
    validate_llm_payload,
)


SOURCE = ROOT / "outputs/trustworthy_soz_clinical_reports_v32_20260816"
LANGUAGE = ROOT / "outputs/constrained_llm_soz_reports_v34_qwen36_20260816"
KNOWLEDGE = ROOT / "knowledge/eeg/knowledge_base.jsonl"
POLICY = ROOT / "configs/constrained_llm_reporting_v1.json"
OUTPUT = ROOT / "outputs/constrained_llm_soz_reports_v34_qwen36_audit_20260816"
LANGUAGE_MANIFEST_SCHEMA = "trustworthy_soz_constrained_llm_manifest_v1"
AUDIT_SCHEMA = "trustworthy_soz_constrained_llm_audit_v1"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.resolve(strict=True).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, object]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def audit(args: argparse.Namespace) -> dict[str, object]:
    source = args.source.resolve(strict=True)
    language = args.language.resolve(strict=True)
    knowledge, policy = load_reporting_knowledge(args.knowledge, args.policy)
    language_manifest = _json(language / "manifest.json")
    if language_manifest.get("schema_version") != LANGUAGE_MANIFEST_SCHEMA:
        raise ValueError("language manifest schema drifted")
    if language_manifest.get("knowledge_base_sha256") != knowledge.base.sha256:
        raise ValueError("knowledge base hash drifted")
    if language_manifest.get("policy_sha256") != knowledge.policy_sha256:
        raise ValueError("policy hash drifted")
    required = policy.get("required_source_ids")
    if not isinstance(required, list):
        raise TypeError("policy required source IDs are missing")
    max_sources = int(policy["max_knowledge_sources"])

    counts: Counter[str] = Counter()
    audited_ids: list[str] = []
    for scope, filename in (
        ("public_patient", "public_patient_reports.jsonl"),
        ("private_event", "private_event_reports.jsonl"),
    ):
        source_rows = _jsonl(source / filename)
        language_rows = _jsonl(language / filename)
        source_by_id = {str(row["unit_id"]): row for row in source_rows}
        language_by_id = {str(row["unit_id"]): row for row in language_rows}
        if len(source_by_id) != len(source_rows) or len(language_by_id) != len(language_rows):
            raise ValueError(f"duplicate unit identity in {scope}")
        if set(source_by_id) != set(language_by_id):
            raise ValueError(f"source/language identity roster mismatch in {scope}")
        for unit_id, report in source_by_id.items():
            record = language_by_id[unit_id]
            if record.get("schema_version") != OUTPUT_SCHEMA:
                raise ValueError(f"language record schema drifted: {unit_id}")
            if record.get("patient_id") != report.get("patient_id"):
                raise ValueError(f"patient identity changed: {unit_id}")
            if record.get("cohort") != report.get("cohort"):
                raise ValueError(f"cohort changed: {unit_id}")
            if record.get("source_report_schema") != report.get("schema_version"):
                raise ValueError(f"source report schema changed: {unit_id}")
            if record.get("source_report_sha256") != _canonical_sha256(report):
                raise ValueError(f"source report hash changed: {unit_id}")
            if record.get("localization") != report.get("localization"):
                raise ValueError(f"localization changed: {unit_id}")

            facts = build_fact_inventory(report)
            if record.get("fact_inventory") != list(facts):
                raise ValueError(f"fact inventory changed: {unit_id}")
            passages = select_reporting_knowledge(
                knowledge,
                report,
                max_sources=max_sources,
                required_source_ids=[str(value) for value in required],
            )
            receipt = record.get("knowledge_receipt")
            if not isinstance(receipt, dict):
                raise TypeError(f"knowledge receipt missing: {unit_id}")
            expected_ids = [passage.id for passage in passages]
            expected_citations = {passage.id: passage.citation for passage in passages}
            if receipt.get("knowledge_base_sha256") != knowledge.base.sha256:
                raise ValueError(f"record knowledge hash changed: {unit_id}")
            if receipt.get("policy_sha256") != knowledge.policy_sha256:
                raise ValueError(f"record policy hash changed: {unit_id}")
            if receipt.get("source_ids") != expected_ids or receipt.get("citations") != expected_citations:
                raise ValueError(f"knowledge selection changed: {unit_id}")

            published = record.get("published_narrative")
            if not isinstance(published, dict):
                raise TypeError(f"published narrative missing: {unit_id}")
            validate_llm_payload(published, report, facts, passages)
            generation = record.get("generation")
            if not isinstance(generation, dict):
                raise TypeError(f"generation receipt missing: {unit_id}")
            generator = str(generation.get("generator", ""))
            if generator not in {"qwen3.6_constrained_language_only", "deterministic_fallback"}:
                raise ValueError(f"unsupported generator: {unit_id}")
            if generation.get("llm_candidate_retained_for_audit") is not False:
                raise ValueError(f"raw LLM draft retention is forbidden: {unit_id}")
            if generation.get("llm_candidate_payload") is not None:
                raise ValueError(f"raw LLM draft leaked into release: {unit_id}")
            if generator == "qwen3.6_constrained_language_only":
                if generation.get("fallback_reason") is not None:
                    raise ValueError(f"successful Qwen record has fallback reason: {unit_id}")
                candidate_hash = generation.get("llm_candidate_sha256")
                if not isinstance(candidate_hash, str) or len(candidate_hash) != 64:
                    raise ValueError(f"successful Qwen record has no replay hash: {unit_id}")
                fallback_reference = deterministic_fallback_payload(report, facts, passages)
                qwen_sections = published["sections"]
                fallback_sections = fallback_reference["sections"]
                if not isinstance(qwen_sections, list) or not isinstance(fallback_sections, list):
                    raise TypeError(f"section comparison failed: {unit_id}")
                section_equal = [
                    qwen.get("text_zh") == fallback.get("text_zh")
                    for qwen, fallback in zip(qwen_sections, fallback_sections, strict=True)
                    if isinstance(qwen, dict) and isinstance(fallback, dict)
                ]
                counts["qwen_sections_total"] += len(section_equal)
                counts["qwen_sections_exactly_equal_deterministic"] += sum(section_equal)
                if section_equal and all(section_equal):
                    counts["qwen_reports_all_sections_exactly_equal_deterministic"] += 1
                notes = published["knowledge_notes"]
                if not isinstance(notes, list):
                    raise TypeError(f"knowledge note comparison failed: {unit_id}")
                source_summaries = {passage.summary_zh for passage in passages}
                counts["qwen_knowledge_notes_total"] += len(notes)
                counts["qwen_knowledge_notes_exactly_equal_source_summary"] += sum(
                    isinstance(note, dict) and note.get("text_zh") in source_summaries
                    for note in notes
                )
            else:
                expected_fallback = deterministic_fallback_payload(report, facts, passages)
                if published != expected_fallback:
                    raise ValueError(f"fallback body is not deterministic: {unit_id}")

            access = record.get("access_receipt")
            if not isinstance(access, dict):
                raise TypeError(f"access receipt missing: {unit_id}")
            for field in (
                "raw_eeg_loaded",
                "soz_gold_labels_loaded",
                "evaluation_rows_loaded",
                "model_scores_or_localization_changed",
                "patient_facts_added",
            ):
                if access.get(field) is not False:
                    raise ValueError(f"access contract failed ({field}): {unit_id}")
            counts[generator] += 1
            counts[f"{scope}_reports"] += 1
            audited_ids.append(f"{scope}/{unit_id}")

    artifacts = {
        "language_manifest": language / "manifest.json",
        "public_language_reports": language / "public_patient_reports.jsonl",
        "private_language_reports": language / "private_event_reports.jsonl",
        "policy": args.policy,
        "knowledge_base": args.knowledge,
        "validator": ROOT / "src/soz/constrained_llm_reporting.py",
        "materializer": ROOT / "scripts/materialize_constrained_llm_soz_reports_v34.py",
        "renderer": ROOT / "scripts/render_trustworthy_soz_reports_v23.py",
        "qwen_model_config": ROOT / "models/Qwen3.6-35B-A3B-GPTQ-Int4/config.json",
        "qwen_generation_config": ROOT
        / "models/Qwen3.6-35B-A3B-GPTQ-Int4/generation_config.json",
        "qwen_weight_index": ROOT
        / "models/Qwen3.6-35B-A3B-GPTQ-Int4/model.safetensors.index.json",
        "qwen_tokenizer": ROOT / "models/Qwen3.6-35B-A3B-GPTQ-Int4/tokenizer.json",
    }
    manifest: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed_independent_revalidation",
        "counts": dict(sorted(counts.items())),
        "audited_unit_count": len(audited_ids),
        "audited_identity_sha256": hashlib.sha256(
            "\n".join(sorted(audited_ids)).encode("utf-8")
        ).hexdigest(),
        "artifact_sha256": {name: _sha256(path) for name, path in artifacts.items()},
        "checks": {
            "source_report_hash_exact": True,
            "localization_exact": True,
            "fact_inventory_exact": True,
            "knowledge_selection_exact": True,
            "published_payload_revalidated": True,
            "raw_llm_candidate_absent": True,
            "raw_eeg_gold_and_evaluation_absent": True,
        },
    }

    target = args.output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--language", type=Path, default=LANGUAGE)
    parser.add_argument("--knowledge", type=Path, default=KNOWLEDGE)
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit(args)
    print(json.dumps({"output": str(args.output), **result["counts"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
