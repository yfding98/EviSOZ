"""End-to-end orchestration for the isolated clinical EEG report pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from .evidence import load_waveform_manifest
from .generation import (
    AUDIT_ONLY_FACT_TYPES,
    build_llm_request,
    build_pipeline_record,
    call_local_qwen_chat,
    eeg_only_generation_report_view,
    load_policy,
)
from .render import render_docx, render_html
from .schema import validate_report_payload
from .style import load_style_profile


MANIFEST_SCHEMA = "clinical_eeg_report_materialization_manifest_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def materialize_clinical_eeg_report(
    *,
    input_path: Path,
    output_dir: Path,
    policy_path: Path,
    style_path: Path,
    base_url: str = "http://127.0.0.1:8000/v1",
    dry_run: bool = False,
    waveform_manifest_path: Path | None = None,
) -> dict[str, Any]:
    policy = load_policy(policy_path)
    style = load_style_profile(style_path)
    if str(policy.get("style_profile")) != str(style_path):
        # Permit equivalent absolute/relative paths while rejecting accidental
        # use of an unrelated style policy.
        configured = (policy_path.resolve().parents[1] / str(policy["style_profile"])).resolve()
        if configured != style_path.resolve():
            raise ValueError("policy and style profile paths do not match")
    input_report = validate_report_payload(_read_object(input_path))
    audit_only_fact_count = sum(
        fact.fact_type in AUDIT_ONLY_FACT_TYPES for fact in input_report.facts
    )
    # The input hash in the outer manifest still attests to the complete legacy
    # input.  Every clinical-content consumer below receives only this reduced
    # signal view, so annotation values cannot affect prompts, waveform
    # bindings, narrative hashes, HTML or DOCX.
    report = validate_report_payload(
        eeg_only_generation_report_view(input_report)
    )
    waveform_manifest = None
    waveform_manifest_sha256: str | None = None
    if waveform_manifest_path is not None:
        resolved_waveform_manifest = waveform_manifest_path.resolve(strict=True)
        before_sha256 = _sha256(resolved_waveform_manifest)
        waveform_manifest = load_waveform_manifest(
            resolved_waveform_manifest,
            report,
        )
        after_sha256 = _sha256(resolved_waveform_manifest)
        if before_sha256 != after_sha256:
            raise ValueError("waveform manifest changed while it was being validated")
        waveform_manifest_sha256 = after_sha256
    event_order = {
        event_id: index for index, event_id in enumerate(report.eeg_event_ids)
    }
    waveform_attachments = (
        tuple(
            sorted(
                waveform_manifest.attachments,
                key=lambda item: (event_order[item.eeg_event_id], item.evidence_id),
            )
        )
        if waveform_manifest is not None
        else ()
    )
    candidate = None
    model_metadata: Mapping[str, Any] | None = None
    generation_error: str | None = "dry_run_llm_not_called" if dry_run else None
    if not dry_run:
        generation = policy["generation"]
        try:
            system_prompt, user_prompt, schema = build_llm_request(report, style)
            candidate, model_metadata = call_local_qwen_chat(
                base_url=base_url,
                model=str(policy["served_model_name"]),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=schema,
                max_tokens=int(generation["max_tokens"]),
                temperature=float(generation["temperature"]),
                enable_thinking=bool(generation["enable_thinking"]),
                timeout_seconds=float(generation["timeout_seconds"]),
                retries=int(generation["retries"]),
            )
        except Exception as exc:  # Fail closed into deterministic narration.
            generation_error = f"{type(exc).__name__}: {exc}"
    record = build_pipeline_record(
        report=report,
        style=style,
        policy=policy,
        candidate_payload=candidate,
        model_metadata=model_metadata,
        generation_error=generation_error,
    )
    scope_receipt = dict(record["scope_receipt"])
    scope_receipt.update(
        {
            "audit_only_input_fact_count": audit_only_fact_count,
            "audit_only_input_fact_values_retained": False,
            "audit_only_input_fact_ids_retained": False,
            "audit_only_facts_excluded_before_waveform_binding": True,
            "audit_only_facts_excluded_before_llm_request": True,
            "audit_only_facts_excluded_before_narrative_and_rendering": True,
        }
    )
    record["scope_receipt"] = scope_receipt
    if waveform_manifest is not None:
        scope_receipt = dict(record["scope_receipt"])
        scope_receipt.update(
            {
                "current_record_evidence_binding_verified": True,
                "evidence_binding_limitation": None,
                "waveform_interpretation_status": "unsigned_eeg_evidence_attachment",
                "waveform_selection_policy": waveform_manifest.selection_policy,
                "waveform_attachment_count": len(waveform_attachments),
            }
        )
        record["scope_receipt"] = scope_receipt
        record["waveform_evidence"] = {
            "schema_version": waveform_manifest.schema_version,
            "selection_policy": waveform_manifest.selection_policy,
            "source_manifest_sha256": waveform_manifest_sha256,
            "attachments": [
                {
                    "evidence_id": item.evidence_id,
                    "fact_ids": list(item.fact_ids),
                    "eeg_event_id": item.eeg_event_id,
                    "figure_sha256": item.figure_sha256,
                    "source_signal_sha256": item.source_signal_sha256,
                    "preprocessing_receipt_sha256": item.preprocessing_receipt_sha256,
                    "processed_window_sha256": item.processed_window_sha256,
                }
                for item in waveform_attachments
            ],
            "images_or_paths_sent_to_llm": False,
        }

    target = output_dir.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        waveform_hrefs: dict[str, str] = {}
        portable_waveform_manifest: dict[str, Any] | None = None
        if waveform_manifest is not None:
            portable_attachments: list[dict[str, Any]] = []
            for index, attachment in enumerate(waveform_attachments, start=1):
                relative = Path("waveforms") / f"eeg_waveform_{index:02d}.png"
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(attachment.source_path, destination)
                if _sha256(destination) != attachment.figure_sha256:
                    raise ValueError("copied waveform PNG failed its declared SHA256")
                waveform_hrefs[attachment.evidence_id] = relative.as_posix()
                item = attachment.to_dict()
                item["figure_file"] = relative.as_posix()
                portable_attachments.append(item)
            portable_waveform_manifest = {
                "schema_version": waveform_manifest.schema_version,
                "report_id": waveform_manifest.report_id,
                "patient_pseudonym": waveform_manifest.patient_pseudonym,
                "selection_policy": waveform_manifest.selection_policy,
                "attachments": portable_attachments,
            }
            (staging / "waveform_manifest.json").write_text(
                json.dumps(
                    portable_waveform_manifest,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        record_path = staging / "record.json"
        record_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        html_path = staging / "report.html"
        html_path.write_text(
            render_html(
                report,
                record,
                style,
                waveform_attachments=waveform_attachments,
                waveform_hrefs=waveform_hrefs or None,
            ),
            encoding="utf-8",
        )
        docx_path = staging / "report.docx"
        render_docx(
            docx_path,
            report,
            record,
            style,
            waveform_attachments=waveform_attachments,
        )
        artifacts = {
            "record.json": _sha256(record_path),
            "report.html": _sha256(html_path),
            "report.docx": _sha256(docx_path),
        }
        if portable_waveform_manifest is not None:
            artifacts["waveform_manifest.json"] = _sha256(
                staging / "waveform_manifest.json"
            )
            for href in waveform_hrefs.values():
                artifacts[href] = _sha256(staging / href)
        privacy_receipt = {
            "patient_identity_sent_to_llm": False,
            "signature_sent_to_llm": False,
            "raw_eeg_loaded_by_narrator": False,
            "non_eeg_context_sent_to_llm": False,
            "sleep_eeg_sent_to_llm": False,
            "activation_experiment_sent_to_llm": False,
            "event_occurrence_sent_to_llm": False,
            "unsupported_sections_omitted_from_report": True,
            "source_annotation_timing_sent_to_llm": False,
            "source_annotation_free_text_sent_to_llm": False,
            "source_annotation_paths_sent_to_llm": False,
            "source_annotation_timing_rendered": False,
            "source_annotation_values_retained_in_output": False,
            "source_annotation_fact_present_in_legacy_input": bool(
                audit_only_fact_count
            ),
        }
        if waveform_manifest is not None:
            privacy_receipt.update(
                {
                    "waveform_images_sent_to_llm": False,
                    "waveform_paths_sent_to_llm": False,
                }
            )
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "completed_unsigned_ai_draft",
            "pipeline_release": policy["pipeline_release"],
            "input": str(input_path),
            "input_sha256": _sha256(input_path.resolve(strict=True)),
            "policy": str(policy_path),
            "policy_sha256": _sha256(policy_path.resolve(strict=True)),
            "style_profile": str(style_path),
            "style_sha256": style.sha256,
            "generator": record["generation"]["generator"],
            "dry_run": bool(dry_run),
            "artifacts": artifacts,
            "release": record["release"],
            "scope_receipt": record["scope_receipt"],
            "privacy_receipt": privacy_receipt,
        }
        if waveform_manifest is not None:
            manifest["waveform_evidence"] = {
                "schema_version": waveform_manifest.schema_version,
                "selection_policy": waveform_manifest.selection_policy,
                "source_manifest": str(waveform_manifest_path),
                "source_manifest_sha256": waveform_manifest_sha256,
                "attachment_count": len(waveform_attachments),
                "evidence_ids": [item.evidence_id for item in waveform_attachments],
                "event_ids": [item.eeg_event_id for item in waveform_attachments],
                "binding_verified": True,
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


__all__ = ["MANIFEST_SCHEMA", "materialize_clinical_eeg_report"]
