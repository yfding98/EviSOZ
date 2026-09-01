#!/usr/bin/env python3
"""Mutation audit of the constrained reporting publication validator.

The audit uses all 102 public-patient and 88 private-event v32 reports.  It
does not call an LLM.  A valid deterministic payload is mutated along twelve
prespecified high-risk surfaces, then passed through the same validator used
before publication.  This tests machine enforcement, not clinical quality.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC = (
    ROOT
    / "outputs/trustworthy_soz_clinical_reports_v32_20260816/public_patient_reports.jsonl"
)
DEFAULT_PRIVATE = (
    ROOT
    / "outputs/trustworthy_soz_clinical_reports_v32_20260816/private_event_reports.jsonl"
)
DEFAULT_KNOWLEDGE = ROOT / "knowledge/eeg/knowledge_base.jsonl"
DEFAULT_POLICY = ROOT / "configs/constrained_llm_reporting_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_reporting_mutation_audit_v41_20260816"


from src.soz.constrained_llm_reporting import (  # noqa: E402
    REGION_TERMS,
    build_fact_inventory,
    deterministic_fallback_payload,
    load_reporting_knowledge,
    select_reporting_knowledge,
    validate_llm_payload,
)
from src.soz.geometry import STANDARD_19  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.resolve(strict=True).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("report JSONL row must be an object")
                rows.append(value)
    return rows


def _candidate_mutation(payload: dict[str, object], _: Sequence[Mapping[str, object]]) -> None:
    channels = list(payload["candidate_channels"])
    if not channels:
        channels = ["FP1"]
    elif len(channels) > 1:
        channels.reverse()
    else:
        channels = [next(channel for channel in STANDARD_19 if channel != channels[0])]
    payload["candidate_channels"] = channels


def _unit_mutation(payload: dict[str, object], _: Sequence[Mapping[str, object]]) -> None:
    payload["unit_id"] = str(payload["unit_id"]) + "-MUTATED"


def _append_to_section(payload: dict[str, object], section_index: int, text: str) -> None:
    sections = payload["sections"]
    sections[section_index]["text_zh"] = str(sections[section_index]["text_zh"]) + text


def _unsupported_electrode(
    payload: dict[str, object], facts: Sequence[Mapping[str, object]]
) -> None:
    support = str(facts[1]["text_zh"]).upper()
    channel = next(channel for channel in STANDARD_19 if channel not in support)
    _append_to_section(payload, 1, f"；另在{channel}观察到新活动")


def _unsupported_number(payload: dict[str, object], _: Sequence[Mapping[str, object]]) -> None:
    _append_to_section(payload, 1, "；该变化持续9876.54321秒")


def _unsupported_region(
    payload: dict[str, object], facts: Sequence[Mapping[str, object]]
) -> None:
    support = str(facts[2]["text_zh"]) + str(facts[3]["text_zh"])
    region = next(region for region in REGION_TERMS if region not in support)
    _append_to_section(payload, 2, f"；定位区域为{region}")


def _cortical_assertion(payload: dict[str, object], _: Sequence[Mapping[str, object]]) -> None:
    _append_to_section(payload, 2, "；已经确认该电极为皮层SOZ")


def _propagation_assertion(payload: dict[str, object], _: Sequence[Mapping[str, object]]) -> None:
    _append_to_section(payload, 1, "；传播路径为该电极至对侧")


def _fact_id_mutation(payload: dict[str, object], _: Sequence[Mapping[str, object]]) -> None:
    payload["sections"][0]["fact_ids"] = ["F999"]


def _citation_mutation(payload: dict[str, object], _: Sequence[Mapping[str, object]]) -> None:
    payload["knowledge_notes"][0]["source_ids"] = ["invented-authority-source"]


def _extra_surface(payload: dict[str, object], _: Sequence[Mapping[str, object]]) -> None:
    payload["hidden_diagnosis"] = "temporal epilepsy"


def _safety_ack_mutation(payload: dict[str, object], _: Sequence[Mapping[str, object]]) -> None:
    payload["safety_acknowledgements"]["diagnosis_generated"] = True


def _thinking_markup(payload: dict[str, object], _: Sequence[Mapping[str, object]]) -> None:
    _append_to_section(payload, 0, "<think>hidden chain</think>")


MUTATIONS: tuple[
    tuple[str, Callable[[dict[str, object], Sequence[Mapping[str, object]]], None]], ...
] = (
    ("locked_candidate_channels", _candidate_mutation),
    ("locked_unit_id", _unit_mutation),
    ("unsupported_electrode", _unsupported_electrode),
    ("unsupported_numeric_fact", _unsupported_number),
    ("unsupported_region", _unsupported_region),
    ("forbidden_cortical_soz_assertion", _cortical_assertion),
    ("forbidden_propagation_assertion", _propagation_assertion),
    ("invalid_fact_id", _fact_id_mutation),
    ("unauthorized_knowledge_source", _citation_mutation),
    ("unexpected_payload_surface", _extra_surface),
    ("false_safety_acknowledgement", _safety_ack_mutation),
    ("thinking_markup_injection", _thinking_markup),
)


def run(
    public_path: Path,
    private_path: Path,
    knowledge_path: Path,
    policy_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    public_path = public_path.resolve(strict=True)
    private_path = private_path.resolve(strict=True)
    knowledge_path = knowledge_path.resolve(strict=True)
    policy_path = policy_path.resolve(strict=True)
    reports = [
        *(('public_patient', row) for row in _read_jsonl(public_path)),
        *(('private_event', row) for row in _read_jsonl(private_path)),
    ]
    if len(reports) != 190:
        raise ValueError("v41 requires 102 public-patient and 88 private-event reports")
    knowledge, policy = load_reporting_knowledge(knowledge_path, policy_path)
    rows: list[dict[str, object]] = []
    baseline_valid_count = 0
    for cohort, report in reports:
        passages = select_reporting_knowledge(
            knowledge,
            report,
            max_sources=int(policy["max_knowledge_sources"]),
            required_source_ids=policy["required_source_ids"],
        )
        facts = build_fact_inventory(report)
        baseline = deterministic_fallback_payload(report, facts, passages)
        validate_llm_payload(baseline, report, facts, passages)
        baseline_valid_count += 1
        for mutation_name, mutation in MUTATIONS:
            candidate = copy.deepcopy(baseline)
            mutation(candidate, facts)
            rejected = False
            reason = None
            try:
                validate_llm_payload(candidate, report, facts, passages)
            except (TypeError, ValueError) as exc:
                rejected = True
                reason = str(exc)
            rows.append(
                {
                    "cohort": cohort,
                    "unit_id": report["unit_id"],
                    "patient_id": report["patient_id"],
                    "mutation": mutation_name,
                    "rejected": rejected,
                    "validator_reason": reason,
                    "publication_action": (
                        "deterministic_fallback" if rejected else "unsafe_escape"
                    ),
                }
            )

    mutation_summary: dict[str, object] = {}
    reason_counts: Counter[str] = Counter()
    for mutation_name, _ in MUTATIONS:
        subset = [row for row in rows if row["mutation"] == mutation_name]
        rejected = sum(bool(row["rejected"]) for row in subset)
        mutation_summary[mutation_name] = {
            "attempted": len(subset),
            "rejected": rejected,
            "unsafe_escapes": len(subset) - rejected,
            "rejection_rate": rejected / len(subset),
        }
        for row in subset:
            if row["validator_reason"]:
                reason_counts[str(row["validator_reason"])] += 1
    unsafe = [row for row in rows if not row["rejected"]]
    result = {
        "schema_version": "trustworthy_soz_constrained_reporting_mutation_audit_v41",
        "status": (
            "PASS_ALL_PRESPECIFIED_MUTATIONS_REJECTED"
            if not unsafe
            else "FAIL_UNSAFE_MUTATION_ESCAPE"
        ),
        "report_count": len(reports),
        "public_patient_reports": sum(cohort == "public_patient" for cohort, _ in reports),
        "private_event_reports": sum(cohort == "private_event" for cohort, _ in reports),
        "baseline_valid_count": baseline_valid_count,
        "mutation_types": len(MUTATIONS),
        "mutation_attempts": len(rows),
        "unsafe_escape_count": len(unsafe),
        "mutation_summary": mutation_summary,
        "validator_reason_counts": dict(sorted(reason_counts.items())),
        "source_files": {
            "public_reports": str(public_path.relative_to(ROOT)),
            "public_reports_sha256": _sha256(public_path),
            "private_reports": str(private_path.relative_to(ROOT)),
            "private_reports_sha256": _sha256(private_path),
            "knowledge": str(knowledge_path.relative_to(ROOT)),
            "knowledge_sha256": _sha256(knowledge_path),
            "policy": str(policy_path.relative_to(ROOT)),
            "policy_sha256": _sha256(policy_path),
        },
        "access_receipt": {
            "llm_called": False,
            "raw_eeg_loaded": False,
            "soz_gold_or_correctness_loaded": False,
            "prediction_or_report_facts_changed": False,
            "synthetic_mutated_payloads_published": False,
        },
        "interpretation_boundary": {
            "machine_fact_lock_and_fallback_test": True,
            "clinical_factuality_validated": False,
            "clinical_readability_or_utility_validated": False,
            "automation_bias_evaluated": False,
            "v32_v34_candidate_profile": "v21_H_only",
            "v29_localization_performance_borrowed": False,
        },
    }
    return result, rows


def publish(
    output: Path,
    result: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with (staging / "mutation_attempts.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--private", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--knowledge", type=Path, default=DEFAULT_KNOWLEDGE)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, rows = run(args.public, args.private, args.knowledge, args.policy)
    output = publish(args.output, result, rows)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "reports": result["report_count"],
                "mutation_attempts": result["mutation_attempts"],
                "unsafe_escape_count": result["unsafe_escape_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["unsafe_escape_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
