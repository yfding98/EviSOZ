#!/usr/bin/env python3
"""Materialize one trustworthy-AI unit as a strict EEG-only clinical draft.

This entry point intentionally bypasses the legacy v32/v34 clinical prose.
It adapts only target-blind typed EEG observations and receipt-bearing waveform
metadata into ``clinical_eeg_report_v1`` before invoking the common local-Qwen
or deterministic report pipeline.  Localization rankings, clinical context,
sleep EEG, activation procedures, and unqualified morphology/rhythm/artifact
fields never enter the clinical ledger or the language-model request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_report.pipeline import materialize_clinical_eeg_report  # noqa: E402
from src.soz.trustworthy_clinical_eeg_adapter import (  # noqa: E402
    adapt_trustworthy_clinical_eeg,
)
from src.soz.private_clinical_eeg_annotations import (  # noqa: E402
    select_private_annotation_event,
)


SCHEMA_VERSION = "trustworthy_clinical_eeg_report_materialization_v1"
SOURCE_REPORT_SCHEMA = "trustworthy_soz_qualified_report_v24"
SOURCE_WAVEFORM_SCHEMA = "trustworthy_soz_processed_waveform_figures_v32"
DEFAULT_SOURCE = ROOT / "outputs/trustworthy_soz_qualified_reports_v24_1_20260815"
DEFAULT_PUBLIC_TYPED_SOURCE = ROOT / "outputs/target_free_oof_reports_v3_recovered_20260813.json"
DEFAULT_POLICY = ROOT / "configs/clinical_eeg_report_v1.json"
DEFAULT_STYLE = ROOT / "configs/clinical_eeg_report_style_zh_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.resolve(strict=True).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _safe_source_png(root: Path, raw: object) -> Path:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError("waveform figure_file must be a POSIX relative path")
    relative = PurePosixPath(raw)
    if (
        relative.is_absolute()
        or relative.suffix != ".png"
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("waveform figure_file is unsafe")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("waveform figure path must not traverse a symlink")
    source = candidate.resolve(strict=True)
    source.relative_to(root)
    if source.is_symlink() or not source.is_file():
        raise ValueError("waveform figure must be a regular non-symlink file")
    return source


def _select_exact(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    value: str,
    context: str,
) -> dict[str, Any]:
    matches = [dict(row) for row in rows if row.get(key) == value]
    if len(matches) != 1:
        raise ValueError(f"{context} requires exactly one {key}={value!r}")
    return matches[0]


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source.resolve(strict=True)
    waveform_root = args.waveforms.resolve(strict=True)
    report_filename = (
        "private_event_reports.jsonl"
        if args.scope == "private_event"
        else "public_patient_reports.jsonl"
    )
    source_path = source_root / report_filename
    qualified_report = _select_exact(
        _jsonl(source_path),
        key="unit_id",
        value=args.unit_id,
        context="qualified report source",
    )
    if qualified_report.get("schema_version") != SOURCE_REPORT_SCHEMA:
        raise ValueError("qualified report schema drifted")

    source_waveform_manifest_path = waveform_root / "manifest.json"
    source_waveform_manifest = _json(source_waveform_manifest_path)
    if source_waveform_manifest.get("schema_version") != SOURCE_WAVEFORM_SCHEMA:
        raise ValueError("source waveform manifest schema drifted")
    binding = source_waveform_manifest.get("evidence_binding")
    if not isinstance(binding, Mapping) or any(
        binding.get(key) is not True
        for key in (
            "source_signal_sha256_recorded_per_figure",
            "preprocessing_receipt_sha256_recorded_per_figure",
            "processed_window_sha256_recorded_per_figure",
            "figure_sha256_recorded_per_figure",
            "channel_order_recorded_per_figure",
            "event_anchor_offset_recorded_per_figure",
        )
    ):
        raise ValueError("source waveform manifest lacks the required hash receipts")
    entries = source_waveform_manifest.get("entries")
    if not isinstance(entries, list):
        raise TypeError("source waveform manifest has no entries")
    waveform_entry = _select_exact(
        [
            row
            for row in entries
            if isinstance(row, Mapping) and row.get("scope") == args.scope
        ],
        key="unit_id",
        value=args.unit_id,
        context="waveform source",
    )
    source_png = _safe_source_png(waveform_root, waveform_entry.get("figure_file"))
    if _sha256(source_png) != waveform_entry.get("figure_sha256"):
        raise ValueError("source waveform PNG hash does not match its manifest")

    public_typed_source = None
    public_source_path = None
    if args.scope == "public_patient":
        public_source_path = args.public_typed_source.resolve(strict=True)
        public_typed_source = _json(public_source_path)
    private_annotation_event = None
    private_annotation_ledger_path = None
    private_annotation_ledger_sha256 = None
    if args.private_annotation_ledger is not None:
        if args.scope != "private_event":
            raise ValueError("private annotation ledger is valid only for private_event")
        private_annotation_ledger_path = args.private_annotation_ledger.resolve(
            strict=True
        )
        private_annotation_ledger_sha256 = _sha256(
            private_annotation_ledger_path
        )
        private_annotation_event = select_private_annotation_event(
            _json(private_annotation_ledger_path),
            event_id=str(qualified_report["unit_id"]),
            patient_id=str(qualified_report["patient_id"]),
            source_signal_sha256=str(waveform_entry["source_signal_sha256"]),
        )
    report_payload, report_waveform_manifest = adapt_trustworthy_clinical_eeg(
        qualified_report,
        waveform_entry,
        public_typed_source=public_typed_source,
        private_annotation_event=private_annotation_event,
    )

    target = args.output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        source_bundle = staging / "source"
        source_bundle.mkdir()
        facts_path = source_bundle / "clinical_eeg_facts.json"
        _write_json(facts_path, report_payload)
        bundled_png = source_bundle / "waveform.png"
        shutil.copy2(source_png, bundled_png)
        if _sha256(bundled_png) != waveform_entry["figure_sha256"]:
            raise ValueError("bundled waveform PNG failed its declared SHA256")
        report_waveform_manifest["attachments"][0]["figure_file"] = "waveform.png"
        report_waveform_manifest_path = source_bundle / "waveform_manifest.json"
        _write_json(report_waveform_manifest_path, report_waveform_manifest)
        private_annotation_event_path = None
        if private_annotation_event is not None:
            private_annotation_event_path = (
                source_bundle / "private_annotation_event.json"
            )
            _write_json(private_annotation_event_path, private_annotation_event)

        source_receipt: dict[str, Any] = {
            "qualified_report_file": str(source_path),
            "qualified_report_file_sha256": _sha256(source_path),
            "qualified_report_row_sha256": _canonical_sha256(qualified_report),
            "source_waveform_manifest": str(source_waveform_manifest_path),
            "source_waveform_manifest_sha256": _sha256(
                source_waveform_manifest_path
            ),
            "source_waveform_entry_sha256": _canonical_sha256(waveform_entry),
            "source_waveform_png_sha256": _sha256(source_png),
            "public_typed_source": (
                str(public_source_path) if public_source_path is not None else None
            ),
            "public_typed_source_sha256": (
                _sha256(public_source_path)
                if public_source_path is not None
                else None
            ),
            "adapter": "src.soz.trustworthy_clinical_eeg_adapter",
            "localization_copied_to_clinical_ledger": False,
            "clinical_sleep_or_activation_content_copied": False,
            "llm_selected_or_cropped_waveform": False,
            "private_annotation_ledger": (
                str(private_annotation_ledger_path)
                if private_annotation_ledger_path is not None
                else None
            ),
            "private_annotation_ledger_sha256": private_annotation_ledger_sha256,
            "private_annotation_event_sha256": (
                _sha256(private_annotation_event_path)
                if private_annotation_event_path is not None
                else None
            ),
            "raw_annotation_text_or_path_copied": False,
            "excel_pending_review_copied_to_report": False,
            "source_annotation_content_sent_to_llm": False,
            "source_annotation_promoted_to_onset_spread_or_duration": False,
        }
        _write_json(source_bundle / "source_receipt.json", source_receipt)

        materialize_clinical_eeg_report(
            input_path=facts_path,
            output_dir=staging / "report",
            policy_path=args.policy,
            style_path=args.style,
            base_url=args.base_url,
            dry_run=args.dry_run,
            waveform_manifest_path=report_waveform_manifest_path,
        )
        report_manifest_path = staging / "report" / "manifest.json"
        report_manifest = _json(report_manifest_path)
        report_manifest["input"] = "../source/clinical_eeg_facts.json"
        report_manifest["waveform_evidence"]["source_manifest"] = (
            "../source/waveform_manifest.json"
        )
        if private_annotation_event_path is not None:
            annotation_receipt = report_manifest.get(
                "source_annotation_evidence"
            )
            if not isinstance(annotation_receipt, dict):
                raise ValueError(
                    "report manifest is missing source annotation evidence receipt"
                )
            annotation_receipt["source_event"] = (
                "../source/private_annotation_event.json"
            )
            annotation_receipt["source_event_sha256"] = _sha256(
                private_annotation_event_path
            )
        _write_json(report_manifest_path, report_manifest)

        artifact_hashes: dict[str, str] = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                artifact_hashes[path.relative_to(staging).as_posix()] = _sha256(path)
        root_manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed_unsigned_trustworthy_ai_eeg_only_draft",
            "scope": args.scope,
            "unit_id": args.unit_id,
            "report_id": report_payload["report_id"],
            "clinical_report": "report/report.html",
            "docx_report": "report/report.docx",
            "facts": "source/clinical_eeg_facts.json",
            "waveform_manifest": "source/waveform_manifest.json",
            "source_receipt": "source/source_receipt.json",
            "generator": report_manifest["generator"],
            "dry_run": bool(args.dry_run),
            "artifacts": artifact_hashes,
            "claim_boundary": {
                "unsigned_ai_draft": True,
                "clinical_export_allowed": False,
                "physician_review_required": True,
                "waveform_is_independent_diagnosis": False,
                "neutral_change_is_seizure_onset_or_propagation": False,
                "soz_ez_or_treatment_target_generated": False,
                "source_annotation_point_marker_promoted_to_onset": False,
                "source_annotation_duration_inferred": False,
                "unreviewed_excel_annotation_generated": False,
                "raw_annotation_text_sent_to_llm": False,
            },
        }
        if private_annotation_event_path is not None:
            root_manifest["private_annotation_event"] = (
                "source/private_annotation_event.json"
            )
        _write_json(staging / "manifest.json", root_manifest)
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return root_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--waveforms", type=Path, required=True)
    parser.add_argument("--scope", choices=("private_event", "public_patient"), required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--public-typed-source", type=Path, default=DEFAULT_PUBLIC_TYPED_SOURCE)
    parser.add_argument(
        "--private-annotation-ledger",
        type=Path,
        default=None,
        help=(
            "Optional de-identified private_clinical_eeg_annotation_ledger_v1; "
            "raw EDF/Excel annotation sources are never read by this report process"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = materialize(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "generator": result["generator"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
