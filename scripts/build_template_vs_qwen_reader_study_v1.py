#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SECTION_ORDER = (
    "case_scope",
    "waveform_review",
    "localization_reference",
    "uncertainty_and_boundary",
)
SECTION_HEADINGS = {
    "case_scope": "分析范围",
    "waveform_review": "波形复核要点",
    "localization_reference": "定位参考",
    "uncertainty_and_boundary": "证据边界",
}
SECTION_FACT_TYPES = {
    "case_scope": ("analysis_scope",),
    "waveform_review": ("waveform_observation",),
    "localization_reference": ("localization_result", "reference_opinion"),
    "uncertainty_and_boundary": ("evidence_applicability", "clinical_boundary"),
}
DEFAULT_QUOTAS = {
    "public_display_candidate": 14,
    "public_localization_abstain": 6,
    "private_display_candidate": 14,
    "private_localization_abstain": 6,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object in {path}")
                rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cohort_name(row: Mapping[str, Any]) -> str:
    cohort = str(row["cohort"])
    if cohort.startswith("public_"):
        return "public"
    if cohort.startswith("private_"):
        return "private"
    raise ValueError(f"Unknown cohort: {cohort}")


def _stratum(row: Mapping[str, Any]) -> str:
    localization = row.get("localization")
    if not isinstance(localization, Mapping):
        raise ValueError("Missing localization")
    return f"{_cohort_name(row)}_{localization['action']}"


def _load_knowledge(path: Path) -> dict[str, str]:
    knowledge: dict[str, str] = {}
    for row in _read_jsonl(path):
        knowledge[str(row["id"])] = str(row["summary_zh"])
    return knowledge


def _template_payload(
    row: Mapping[str, Any],
    *,
    knowledge: Mapping[str, str],
) -> dict[str, Any]:
    facts = row.get("fact_inventory")
    qwen = row.get("published_narrative")
    localization = row.get("localization")
    if not isinstance(facts, Sequence) or not isinstance(qwen, Mapping) or not isinstance(localization, Mapping):
        raise ValueError("Malformed constrained report")
    by_type = {str(item["fact_type"]): item for item in facts if isinstance(item, Mapping)}
    sections: list[dict[str, Any]] = []
    for section_id in SECTION_ORDER:
        selected = [by_type[name] for name in SECTION_FACT_TYPES[section_id]]
        sections.append(
            {
                "section_id": section_id,
                "heading_zh": SECTION_HEADINGS[section_id],
                "text_zh": "。".join(str(item["text_zh"]).rstrip("。") for item in selected) + "。",
                "fact_ids": [str(item["fact_id"]) for item in selected],
            }
        )

    qwen_notes = qwen.get("knowledge_notes", [])
    notes: list[dict[str, Any]] = []
    for note in qwen_notes:
        if not isinstance(note, Mapping):
            raise ValueError("Malformed Qwen knowledge note")
        source_ids = [str(item) for item in note.get("source_ids", [])]
        missing = [source_id for source_id in source_ids if source_id not in knowledge]
        if missing:
            raise ValueError(f"Unknown knowledge source IDs: {missing}")
        notes.append(
            {
                "text_zh": " ".join(knowledge[source_id].rstrip() for source_id in source_ids),
                "source_ids": source_ids,
            }
        )
    return {
        "sections": sections,
        "knowledge_notes": notes,
        "localization_action": str(localization["action"]),
        "candidate_channels": [str(item["channel"]) for item in localization.get("displayed_candidates", [])],
        "top1_region_zh": localization.get("top1_region_projection_zh"),
    }


def _reader_text(payload: Mapping[str, Any]) -> str:
    parts = [f"{section['heading_zh']}\n{section['text_zh']}" for section in payload["sections"]]
    if payload.get("knowledge_notes"):
        notes = "\n".join(f"- {note['text_zh']}" for note in payload["knowledge_notes"])
        parts.append("一般医学知识说明\n" + notes)
    return "\n\n".join(parts)


def _validate_pair(template: Mapping[str, Any], qwen: Mapping[str, Any]) -> None:
    for field in ("localization_action", "candidate_channels", "top1_region_zh"):
        if template.get(field) != qwen.get(field):
            raise ValueError(f"Locked localization differs between variants: {field}")
    template_fact_ids = [tuple(section["fact_ids"]) for section in template["sections"]]
    qwen_fact_ids = [tuple(section["fact_ids"]) for section in qwen["sections"]]
    if template_fact_ids != qwen_fact_ids:
        raise ValueError("Template and Qwen sections do not bind the same patient facts")
    template_sources = [tuple(note["source_ids"]) for note in template["knowledge_notes"]]
    qwen_sources = [tuple(note["source_ids"]) for note in qwen["knowledge_notes"]]
    if template_sources != qwen_sources:
        raise ValueError("Template and Qwen variants do not use the same knowledge sources")


def build_pack(
    *,
    qwen_dir: Path,
    knowledge_path: Path,
    output_dir: Path,
    seed: int,
    quotas: Mapping[str, int] = DEFAULT_QUOTAS,
) -> dict[str, Any]:
    public_path = qwen_dir / "public_patient_reports.jsonl"
    private_path = qwen_dir / "private_event_reports.jsonl"
    source_rows = _read_jsonl(public_path) + _read_jsonl(private_path)
    eligible = [
        row
        for row in source_rows
        if row.get("generation", {}).get("generator") == "qwen3.6_constrained_language_only"
        and not row.get("generation", {}).get("fallback_reason")
    ]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        buckets[_stratum(row)].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for stratum, quota in quotas.items():
        rows = sorted(buckets.get(stratum, []), key=lambda row: (str(row["unit_id"]), str(row["patient_id"])))
        if len(rows) < quota:
            raise ValueError(f"Insufficient eligible reports for {stratum}: {len(rows)} < {quota}")
        selected.extend(rng.sample(rows, quota))
    rng.shuffle(selected)

    knowledge = _load_knowledge(knowledge_path)
    cards: list[dict[str, Any]] = []
    allocation: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    variant_a_template_count = 0
    for index, row in enumerate(selected, start=1):
        case_id = f"LLM-R2-{index:03d}"
        qwen = row["published_narrative"]
        template = _template_payload(row, knowledge=knowledge)
        _validate_pair(template, qwen)
        template_first = rng.random() < 0.5
        if template_first:
            variant_a_template_count += 1
        variant_a = template if template_first else qwen
        variant_b = qwen if template_first else template
        cards.append(
            {
                "schema_version": "trustworthy_soz_template_qwen_reader_card_v1",
                "case_id": case_id,
                "stratum": _stratum(row),
                "locked_fact_inventory": row["fact_inventory"],
                "variant_a_text_zh": _reader_text(variant_a),
                "variant_b_text_zh": _reader_text(variant_b),
                "variant_a_fact_ids": [section["fact_ids"] for section in variant_a["sections"]],
                "variant_b_fact_ids": [section["fact_ids"] for section in variant_b["sections"]],
                "knowledge_source_ids": [note["source_ids"] for note in qwen["knowledge_notes"]],
                "candidate_action": qwen["localization_action"],
                "candidate_channels": qwen["candidate_channels"],
                "top1_region_zh": qwen["top1_region_zh"],
            }
        )
        allocation.append(
            {
                "case_id": case_id,
                "source_cohort": row["cohort"],
                "source_unit_id": row["unit_id"],
                "source_patient_id": row["patient_id"],
                "variant_a": "deterministic_template" if template_first else "constrained_qwen3.6",
                "variant_b": "constrained_qwen3.6" if template_first else "deterministic_template",
            }
        )
        annotations.append(
            {
                "schema_version": "trustworthy_soz_template_qwen_reader_annotation_v1",
                "case_id": case_id,
                "reviewer_id": None,
                "review_status": "unreviewed",
                "variant_a_review_time_sec": None,
                "variant_b_review_time_sec": None,
                "variant_a_professionalism_1_to_5": None,
                "variant_b_professionalism_1_to_5": None,
                "variant_a_clarity_1_to_5": None,
                "variant_b_clarity_1_to_5": None,
                "variant_a_clinical_concision_1_to_5": None,
                "variant_b_clinical_concision_1_to_5": None,
                "variant_a_unsupported_patient_fact": None,
                "variant_b_unsupported_patient_fact": None,
                "variant_a_dangerous_overstatement": None,
                "variant_b_dangerous_overstatement": None,
                "variant_a_major_edit_required": None,
                "variant_b_major_edit_required": None,
                "knowledge_citations_assessable": None,
                "variant_a_knowledge_supported": None,
                "variant_b_knowledge_supported": None,
                "preferred_variant": None,
                "preference_reason": "",
                "review_completed_at": None,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    cards_path = output_dir / "blinded_report_pairs.jsonl"
    allocation_path = output_dir / "data_manager_allocation_key.jsonl"
    reader_a_path = output_dir / "reader_a_annotations.jsonl"
    reader_b_path = output_dir / "reader_b_annotations.jsonl"
    _write_jsonl(cards_path, cards)
    _write_jsonl(allocation_path, allocation)
    _write_jsonl(reader_a_path, ({**row, "reviewer_id": "reader_a"} for row in annotations))
    _write_jsonl(reader_b_path, ({**row, "reviewer_id": "reader_b"} for row in annotations))

    manifest = {
        "schema_version": "trustworthy_soz_template_qwen_reader_pack_v1",
        "status": "empty_target_blind_two_reader_language_comparison_pack_ready",
        "sampling_seed": seed,
        "counts": {
            "cases": len(cards),
            "readers": 2,
            "strata": dict(sorted((key, sum(card["stratum"] == key for card in cards)) for key in quotas)),
            "variant_a_template": variant_a_template_count,
            "variant_a_qwen": len(cards) - variant_a_template_count,
        },
        "comparison_contract": {
            "same_locked_patient_facts": True,
            "same_localization": True,
            "same_knowledge_source_ids": True,
            "generator_identity_blinded": True,
            "preference_is_not_soz_accuracy": True,
            "reader_edits_do_not_return_to_training": True,
        },
        "access_receipt": {
            "raw_eeg_loaded": False,
            "soz_gold_loaded": False,
            "prediction_correctness_loaded": False,
            "private_target_loaded": False,
            "localization_changed": False,
            "patient_facts_changed": False,
        },
        "source": {
            "qwen_manifest": str((qwen_dir / "manifest.json").resolve()),
            "qwen_manifest_sha256": _sha256(qwen_dir / "manifest.json"),
            "knowledge_base": str(knowledge_path.resolve()),
            "knowledge_base_sha256": _sha256(knowledge_path),
        },
        "files": {
            "blinded_report_pairs": cards_path.name,
            "data_manager_allocation_key": allocation_path.name,
            "reader_a_annotations": reader_a_path.name,
            "reader_b_annotations": reader_b_path.name,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a blinded deterministic-template versus constrained-Qwen reader pack")
    parser.add_argument(
        "--qwen-dir",
        type=Path,
        default=Path("outputs/constrained_llm_soz_reports_v34_qwen36_20260816"),
    )
    parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=Path("knowledge/eeg/knowledge_base.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/trustworthy_soz_template_vs_qwen_reader_study_v1_20260816"),
    )
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    qwen_dir = args.qwen_dir if args.qwen_dir.is_absolute() else root / args.qwen_dir
    knowledge_path = args.knowledge_base if args.knowledge_base.is_absolute() else root / args.knowledge_base
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_pack(
        qwen_dir=qwen_dir,
        knowledge_path=knowledge_path,
        output_dir=output_dir,
        seed=args.seed,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    print(f"output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
