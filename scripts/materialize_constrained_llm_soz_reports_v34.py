#!/usr/bin/env python3
"""Create optional Qwen3.6 language-only narratives for facts-locked SOZ reports."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
    build_augmented_record,
    build_fact_inventory,
    build_llm_request,
    call_local_qwen_chat,
    load_reporting_knowledge,
    select_reporting_knowledge,
)


DEFAULT_REPORTS = ROOT / "outputs/trustworthy_soz_clinical_reports_v32_20260816"
DEFAULT_KNOWLEDGE = ROOT / "knowledge/eeg/knowledge_base.jsonl"
DEFAULT_POLICY = ROOT / "configs/constrained_llm_reporting_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/constrained_llm_soz_reports_v34_qwen36_20260816"
MANIFEST_SCHEMA = "trustworthy_soz_constrained_llm_manifest_v1"


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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def materialize(args: argparse.Namespace) -> dict[str, object]:
    knowledge, policy = load_reporting_knowledge(args.knowledge, args.policy)
    generation = policy.get("generation")
    if not isinstance(generation, dict):
        raise TypeError("policy generation contract is missing")
    required = policy.get("required_source_ids")
    if not isinstance(required, list):
        raise TypeError("policy required_source_ids is missing")
    max_sources = int(policy.get("max_knowledge_sources", 0))
    selected_ids = set(args.unit_id)
    rows: list[tuple[str, dict[str, object]]] = []
    for scope, filename in (
        ("public_patient", "public_patient_reports.jsonl"),
        ("private_event", "private_event_reports.jsonl"),
    ):
        for report in _jsonl(args.reports / filename):
            if selected_ids and str(report.get("unit_id")) not in selected_ids:
                continue
            rows.append((scope, report))
    if args.max_reports is not None:
        rows = rows[: args.max_reports]
    if not rows:
        raise ValueError("no reports selected")
    missing = selected_ids.difference(str(report["unit_id"]) for _, report in rows)
    if missing:
        raise ValueError(f"requested report IDs not found: {sorted(missing)}")

    if args.workers < 1 or args.workers > 16:
        raise ValueError("workers must be between 1 and 16")

    def process(item: tuple[str, dict[str, object]]) -> tuple[str, dict[str, object]]:
        scope, report = item
        facts = build_fact_inventory(report)
        passages = select_reporting_knowledge(
            knowledge,
            report,
            max_sources=max_sources,
            required_source_ids=[str(value) for value in required],
        )
        candidate = None
        metadata = None
        error = None
        if not args.dry_run:
            system_prompt, user_prompt = build_llm_request(report, facts, passages, knowledge)
            try:
                candidate, metadata = call_local_qwen_chat(
                    base_url=args.base_url,
                    model=str(policy["served_model_name"]),
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=int(generation["max_tokens"]),
                    temperature=float(generation["temperature"]),
                    enable_thinking=bool(generation["enable_thinking"]),
                    timeout_seconds=float(generation["timeout_seconds"]),
                    retries=int(generation["retries"]),
                )
            except Exception as exc:  # fail closed into the deterministic report
                error = f"{type(exc).__name__}: {exc}"
        record = build_augmented_record(
            report=report,
            facts=facts,
            passages=passages,
            knowledge=knowledge,
            candidate_payload=candidate,
            model_metadata=metadata,
            generation_error=error,
        )
        return scope, record

    if args.workers == 1:
        processed = [process(item) for item in rows]
    else:
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="soz-qwen") as pool:
            processed = list(pool.map(process, rows))

    output_rows: dict[str, list[dict[str, object]]] = {
        "public_patient": [],
        "private_event": [],
    }
    counts: Counter[str] = Counter()
    for scope, record in processed:
        output_rows[scope].append(record)
        generator = str(record["generation"]["generator"])
        counts[generator] += 1
        counts[f"{scope}_reports"] += 1

    target = args.output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        for scope, scope_rows in output_rows.items():
            if scope_rows:
                _write_jsonl(staging / f"{scope}_reports.jsonl", scope_rows)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "completed_optional_constrained_language_layer",
            "record_schema": OUTPUT_SCHEMA,
            "counts": dict(sorted(counts.items())),
            "source_reports": str(args.reports),
            "knowledge_base": str(args.knowledge),
            "knowledge_base_sha256": knowledge.base.sha256,
            "policy": str(args.policy),
            "policy_sha256": knowledge.policy_sha256,
            "model_release": policy["model_release"],
            "served_model_name": policy["served_model_name"],
            "generation_contract": {
                "enable_thinking": bool(generation["enable_thinking"]),
                "temperature": float(generation["temperature"]),
                "response_format": "strict_json_schema",
            },
            "dry_run": bool(args.dry_run),
            "request_workers": int(args.workers),
            "access_receipt": {
                "raw_eeg_loaded": False,
                "soz_gold_labels_loaded": False,
                "evaluation_rows_loaded": False,
                "localization_changed": False,
                "llm_role": policy["role"],
            },
        }
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
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--knowledge", type=Path, default=DEFAULT_KNOWLEDGE)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--unit-id", action="append", default=[])
    parser.add_argument("--max-reports", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = materialize(args)
    print(json.dumps({"output": str(args.output), **result["counts"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
