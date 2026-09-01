"""Atomic materialization of one long-recording trustworthy EEG draft."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from src.clinical_eeg_report.generation import (
    build_llm_request,
    build_pipeline_record,
    call_local_qwen_chat,
    load_policy,
)
from src.clinical_eeg_report.schema import validate_report_payload
from src.clinical_eeg_report.style import load_style_profile
from .aggregation import (
    aggregate_long_term_event_segments,
    validate_trustworthy_long_term_clinical_eeg_bundle,
)
from .render import (
    _fact_locked_event_language,
    render_long_term_docx,
    render_long_term_html,
)
from .report_outcome import classify_recording_eeg_outcome
from .schema import validate_long_term_event_segment_receipt


MATERIALIZATION_SCHEMA = "trustworthy_long_term_clinical_eeg_materialization_v1"
FILTERED_MATERIALIZATION_SCHEMA = (
    "trustworthy_long_term_clinical_eeg_materialization_v2_signal_eligibility_partition"
)
LANGUAGE_LAYER_SCHEMA = "long_term_event_language_layer_v1"
LANGUAGE_REQUEST_AUDIT_SCHEMA = "long_term_event_language_request_audit_v1"

_UNAUTHORIZED_CURRENT_LONG_RECORDING_CLINICAL_TERM_RE = re.compile(
    r"(?:\b(?:spikes?|sharp(?:\s+waves?)?|IEDs?|ESz|LVFA|electrodecrement|"
    r"electrographic\s+seizures?|ictal\s+(?:onset|evolution|spread|termination)|"
    r"diffuse|generalized|bilateral(?:ly)?\s+synchronous|SOZ)\b|"
    r"棘波|尖波|癫痫样放电|电图发作|脑电发作|低电压快活动|"
    r"(?:电压|电极)递减|(?:发作期|临床)(?:脑电)?演变|"
    r"发作(?:起始|传播|扩散|终止)|病理性\s*[δθ]|局灶(?:性)?慢化|弥漫(?:性)?慢化|"
    r"(?:弥漫(?:性)?|广泛性|双侧同步)(?:分布|起始|发作)|"
    r"皮层\s*SOZ|致痫区|致痫灶|手术靶点)",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _distinctive_strings(value: object) -> set[str]:
    """Collect opaque values that would prove a side-channel prompt leak.

    Short clinical codes (for example ``right`` or ``T7``) can legitimately
    occur in both an EEG fact and a separately transcribed observation.  They
    therefore cannot establish provenance.  Opaque IDs, receipts, hashes and
    relative figure names can, and are checked byte-for-byte below.
    """

    result: set[str] = set()

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)
        elif isinstance(item, str) and len(item) >= 8:
            result.add(item)

    visit(value)
    return result


def _assert_prompt_firewall(
    *,
    system_prompt: str,
    user_prompt: str,
    event: Mapping[str, Any],
) -> dict[str, bool]:
    """Fail closed if any opaque non-narrative value reaches a prompt."""

    prompt = system_prompt + "\n" + user_prompt
    waveform = event["waveform_attachment"]
    ranking = event["research_soz_ranking_receipt"]
    event_envelope = {
        "eeg_event_id": event["eeg_event_id"],
        "candidate_id": event["candidate_id"],
    }
    groups: tuple[tuple[str, set[str]], ...] = (
        ("event_envelope", _distinctive_strings(event_envelope)),
        ("waveform", _distinctive_strings(waveform)),
        ("research_ranking", _distinctive_strings(ranking)),
    )
    for group_name, tokens in groups:
        leaked = sorted(token for token in tokens if token in prompt)
        if leaked:
            # Do not echo the value: an exception may be persisted in an audit
            # record by a caller.  The category is sufficient for diagnosis.
            raise ValueError(f"language prompt firewall rejected {group_name} values")
    return {
        "event_envelope_values_absent": True,
        "source_context_values_absent": True,
        "edf_annotation_values_absent": True,
        "excel_observation_values_absent": True,
        "waveform_values_absent": True,
        "research_ranking_values_absent": True,
    }


def _request_audit(
    *,
    system_prompt: str,
    user_prompt: str,
    schema: Mapping[str, Any],
    firewall: Mapping[str, bool],
    outcome: str,
) -> dict[str, Any]:
    if outcome not in {"candidate_received", "request_failed"}:
        raise ValueError("language request audit outcome is unsupported")
    return {
        "schema_version": LANGUAGE_REQUEST_AUDIT_SCHEMA,
        "request_constructed": True,
        "system_prompt_sha256": _text_sha256(system_prompt),
        "user_prompt_sha256": _text_sha256(user_prompt),
        "output_schema_sha256": _canonical_sha256(schema),
        "prompt_or_schema_content_persisted": False,
        "request_outcome": outcome,
        "firewall": dict(firewall),
    }


def _assert_current_long_recording_language_semantics(candidate: object) -> None:
    """Reject clinical promotion terms from the current neutral event path.

    The generic report schema intentionally remains backwards compatible with
    historical typed facts.  This additional gate is local to
    ``clinical_eeg_long_recording_v1`` whose current producer emits only a
    quantitative sustained-change candidate.  A rejected Qwen response is
    replaced by the deterministic narrative and never blocks publication.
    """

    def visit(value: object, *, text_field: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                visit(nested, text_field=isinstance(key, str) and key.endswith("text_zh"))
        elif isinstance(value, list):
            for nested in value:
                visit(nested, text_field=text_field)
        elif text_field and isinstance(value, str):
            if _UNAUTHORIZED_CURRENT_LONG_RECORDING_CLINICAL_TERM_RE.search(value):
                raise ValueError(
                    "Qwen promoted a neutral long-recording EEG observation"
                )

    visit(candidate)


def _safe_source_png(root: Path, figure_file: object) -> Path:
    if not isinstance(figure_file, str) or not figure_file or "\\" in figure_file:
        raise ValueError("waveform figure_file must be a POSIX relative path")
    relative = PurePosixPath(figure_file)
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
            raise ValueError("waveform path must not traverse a symlink")
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(root)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("waveform must be a regular non-symlink PNG")
    return resolved


def _language_layer(
    *,
    bundle: Mapping[str, Any],
    policy_path: Path,
    style_path: Path,
    base_url: str,
    use_qwen: bool,
) -> dict[str, Any]:
    """Generate validated event-language records with bounded render authority.

    Only an exact, validation-clean ``qwen3.6_facts_locked_draft`` may lend its
    three EEG event wording fields to the renderer.  The renderer revalidates
    that record against the current event payload and never delegates event
    identity, coordinates, facts, impression, context or attachments.
    """

    policy = load_policy(policy_path)
    style = load_style_profile(style_path)
    configured_style = str(policy.get("style_profile"))
    if configured_style != str(style_path):
        configured = (
            policy_path.resolve().parents[1] / configured_style
        ).resolve()
        if configured != style_path.resolve():
            raise ValueError("language policy and style profile do not match")
    records: list[dict[str, Any]] = []
    for event in bundle["events"]:
        report = validate_report_payload(event["event_report_payload"])
        candidate = None
        model_metadata: Mapping[str, Any] | None = None
        error: str | None = "qwen_disabled_deterministic_fallback"
        request_audit: dict[str, Any] | None = None
        if use_qwen:
            generation = policy["generation"]
            system_prompt, user_prompt, schema = build_llm_request(report, style)
            system_prompt += (
                "当前长程流水线的自动信号 producer 只形成双极导联级量化持续变化。"
                "不得把它命名为棘波、尖波、IED、电图发作、LVFA、电压递减、"
                "发作期演变、起始、传播、弥漫/广泛或双侧同步分布，也不得给出 SOZ。"
            )
            firewall = _assert_prompt_firewall(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                event=event,
            )
            try:
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
                request_audit = _request_audit(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=schema,
                    firewall=firewall,
                    outcome="candidate_received",
                )
                try:
                    _assert_current_long_recording_language_semantics(candidate)
                except ValueError:
                    candidate = None
                    error = "qwen_candidate_failed_long_recording_semantic_gate"
                else:
                    error = None
            except Exception:  # validated deterministic fallback
                # Never persist endpoint/OS exception prose.  A stable error
                # code makes repeated service-failure materializations byte
                # reproducible and cannot accidentally capture source values.
                error = "qwen_request_failed"
                request_audit = _request_audit(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=schema,
                    firewall=firewall,
                    outcome="request_failed",
                )
        records.append(
            {
                "eeg_event_id": event["eeg_event_id"],
                "recording_event_start_offset_seconds": event[
                    "recording_event_start_offset_seconds"
                ],
                "language_record": build_pipeline_record(
                    report=report,
                    style=style,
                    policy=policy,
                    candidate_payload=candidate,
                    model_metadata=model_metadata,
                    generation_error=error,
                ),
                "request_audit": request_audit,
            }
        )
    layer = {
        "schema_version": LANGUAGE_LAYER_SCHEMA,
        "role": "validated_audit_with_bounded_eeg_event_wording_projection",
        "served_model_name": str(policy["served_model_name"]),
        "qwen_requested": use_qwen,
        "event_records": records,
        "scope_receipt": {},
    }
    projection_eligible_count = len(
        _fact_locked_event_language(bundle, layer)
    )
    layer["scope_receipt"] = {
            "clinical_eeg_fact_ledgers_sent": use_qwen and bool(records),
            "source_context_sent": False,
            "edf_annotation_sent": False,
            "excel_observation_sent": False,
            "waveform_image_or_path_sent": False,
            "research_soz_ranking_sent": False,
            "may_change_event_count": False,
            "may_change_event_coordinates": False,
            "may_change_recording_impression": False,
            "used_by_deterministic_renderer": projection_eligible_count > 0,
            "bounded_event_wording_projection_eligible_count": projection_eligible_count,
            "projection_generator_must_equal": "qwen3.6_facts_locked_draft",
            "projection_excludes_findings_and_impression": True,
            "prompt_or_schema_content_persisted": False,
            "request_audit_hashes_only": True,
            "prompt_firewall_fail_closed": True,
    }
    return layer


def materialize_long_term_clinical_eeg_report(
    *,
    detection_manifest_path: Path,
    segment_receipt_paths: Sequence[Path],
    waveform_root: Path,
    output_dir: Path,
    bundle_id: str,
    policy_path: Path | None = None,
    style_path: Path | None = None,
    source_context_path: Path | None = None,
    analysis_selection_path: Path | None = None,
    base_url: str = "http://127.0.0.1:8000/v1",
    use_qwen: bool = False,
) -> dict[str, Any]:
    """Validate, aggregate and atomically publish one recording report."""

    if source_context_path is not None:
        raise ValueError(
            "EEG-only report generation rejects EDF annotation, spreadsheet, "
            "or physician-GT context; use the post-freeze evaluation pipeline"
        )
    if isinstance(segment_receipt_paths, (str, bytes)) or not isinstance(
        segment_receipt_paths, Sequence
    ):
        raise TypeError("segment_receipt_paths must be a sequence")
    detection_path = detection_manifest_path.resolve(strict=True)
    analysis_selection_source = (
        analysis_selection_path.resolve(strict=True)
        if analysis_selection_path is not None
        else None
    )
    analysis_selection = (
        _json_object(analysis_selection_source)
        if analysis_selection_source is not None
        else None
    )
    segments: list[dict[str, Any]] = []
    segment_sources: list[dict[str, str]] = []
    for raw_path in segment_receipt_paths:
        path = raw_path.resolve(strict=True)
        segments.append(validate_long_term_event_segment_receipt(_json_object(path)))
        segment_sources.append({"sha256": _sha256(path)})
    bundle = aggregate_long_term_event_segments(
        _json_object(detection_path),
        segments,
        bundle_id,
        analysis_selection=analysis_selection,
    )
    diagnostic_outcome = classify_recording_eeg_outcome(bundle)
    if (policy_path is None) != (style_path is None):
        raise ValueError("policy_path and style_path must be supplied together")
    language = None
    if policy_path is not None and style_path is not None:
        language = _language_layer(
            bundle=bundle,
            policy_path=policy_path.resolve(strict=True),
            style_path=style_path.resolve(strict=True),
            base_url=base_url,
            use_qwen=use_qwen,
        )
    elif use_qwen:
        raise ValueError("use_qwen requires a policy and style profile")

    source_waveform_root = waveform_root.resolve(strict=True)
    if not source_waveform_root.is_dir():
        raise ValueError("waveform_root must be a directory")
    source_images: list[Path] = []
    for event in bundle["events"]:
        attachment = event["waveform_attachment"]
        source = _safe_source_png(source_waveform_root, attachment["figure_file"])
        if _sha256(source) != attachment["figure_sha256"]:
            raise ValueError("waveform PNG does not match its declared SHA-256")
        source_images.append(source)

    target = output_dir.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        portable = deepcopy(bundle)
        hrefs: dict[str, str] = {}
        docx_paths: dict[str, Path] = {}
        for index, (event, source) in enumerate(
            zip(portable["events"], source_images), start=1
        ):
            relative = Path("waveforms") / f"eeg_waveform_{index:02d}.png"
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if _sha256(destination) != event["waveform_attachment"]["figure_sha256"]:
                raise ValueError("copied waveform failed its declared SHA-256")
            event_id = str(event["eeg_event_id"])
            event["waveform_attachment"]["figure_file"] = relative.as_posix()
            hrefs[event_id] = relative.as_posix()
            docx_paths[event_id] = destination
        portable = validate_trustworthy_long_term_clinical_eeg_bundle(portable)

        bundle_path = staging / "bundle.json"
        _write_json(bundle_path, portable)
        detection_output = staging / "detection_manifest.json"
        _write_json(detection_output, portable["detection_manifest"])
        selection_output = None
        if analysis_selection is not None:
            selection_output = staging / "analysis_selection_manifest.json"
            _write_json(selection_output, portable["analysis_selection"])
        segments_output = staging / "event_segment_receipts.json"
        _write_json(segments_output, segments)
        language_output = None
        if language is not None:
            language_output = staging / "language_records.json"
            _write_json(language_output, language)
        html_path = staging / "report.html"
        html_path.write_text(
            render_long_term_html(
                portable,
                waveform_hrefs=hrefs,
                language_layer=language,
            ),
            encoding="utf-8",
        )
        docx_path = staging / "report.docx"
        render_long_term_docx(
            docx_path,
            portable,
            waveform_paths=docx_paths,
            language_layer=language,
        )

        artifacts = {
            "bundle.json": _sha256(bundle_path),
            "detection_manifest.json": _sha256(detection_output),
            "event_segment_receipts.json": _sha256(segments_output),
            "report.html": _sha256(html_path),
            "report.docx": _sha256(docx_path),
        }
        if selection_output is not None:
            artifacts["analysis_selection_manifest.json"] = _sha256(
                selection_output
            )
        if language_output is not None:
            artifacts["language_records.json"] = _sha256(language_output)
        for event in portable["events"]:
            relative = str(event["waveform_attachment"]["figure_file"])
            artifacts[relative] = _sha256(staging / relative)

        language_records = language["event_records"] if language is not None else []
        # Count only event wording that survives the final fact-qualification
        # renderer gate.  A syntactically valid Qwen record with no authorized
        # clinical-facing fact block is a deterministic presentation fallback.
        qwen_validated_count = len(
            _fact_locked_event_language(portable, language)
            if language is not None
            else {}
        )
        deterministic_fallback_count = len(language_records) - qwen_validated_count
        manifest = {
            "schema_version": (
                FILTERED_MATERIALIZATION_SCHEMA
                if analysis_selection is not None
                else MATERIALIZATION_SCHEMA
            ),
            "status": "completed_unsigned_ai_draft",
            "diagnostic_status": diagnostic_outcome["report_status"],
            "diagnostic_outcome": diagnostic_outcome,
            "recording_id": portable["recording_id"],
            "patient_pseudonym": portable["patient_pseudonym"],
            "bundle_id": portable["bundle_id"],
            "event_count": portable["event_count"],
            "source_receipts": {
                "detection_manifest_sha256": _sha256(detection_path),
                "segment_receipts": segment_sources,
                "context_sha256": None,
            },
            "scope_receipt": {
                "entire_record_detection_manifest_validated": True,
                "three_timebase_closure_verified": True,
                "eeg_signal_only_generation": True,
                "eeg_facts_and_automatic_impression_signal_only": True,
                "external_edf_annotations_loaded": False,
                "excel_observations_loaded": False,
                "source_context_joined_post_freeze": False,
                "source_context_displayed_as_separate_attributed_section": False,
                "source_context_sent_to_qwen": False,
                "context_changed_event_count": False,
                "context_changed_eeg_facts_or_impression": False,
                "research_soz_used_in_clinical_facts_or_llm": False,
                "sleep_activation_ecg_emg_or_demographics_generated": False,
                "all_waveforms_hash_verified": True,
                "physician_signed": False,
            },
            "language_service_receipt": {
                "configured": language is not None,
                "qwen_requested": bool(
                    language is not None and language.get("qwen_requested") is True
                ),
                "event_count": len(language_records),
                "validated_qwen_wording_count": qwen_validated_count,
                "deterministic_fallback_count": deterministic_fallback_count,
                "language_failure_blocks_report_publication": False,
            },
            "artifacts": artifacts,
        }
        if analysis_selection is not None:
            assert analysis_selection_source is not None
            manifest.update(
                {
                    "detector_selected_candidate_count": portable[
                        "detector_selected_candidate_count"
                    ],
                    "analysis_analyzable_candidate_count": portable[
                        "analysis_analyzable_candidate_count"
                    ],
                    "analysis_rejected_candidate_count": portable[
                        "analysis_rejected_candidate_count"
                    ],
                }
            )
            manifest["source_receipts"]["analysis_selection_sha256"] = _sha256(
                analysis_selection_source
            )
            manifest["scope_receipt"].update(
                {
                    "signal_eligibility_partition_validated": True,
                    "detector_selected_candidates_exactly_partitioned": True,
                    "rejected_candidate_is_not_no_seizure": True,
                }
            )
        manifest_path = staging / "manifest.json"
        _write_json(manifest_path, manifest)
        for path in staging.rglob("*"):
            os.chmod(path, 0o700 if path.is_dir() else 0o600)
        os.chmod(staging, 0o700)
        os.replace(staging, target)
        os.chmod(target, 0o700)
        published = True
        return deepcopy(manifest)
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "LANGUAGE_LAYER_SCHEMA",
    "LANGUAGE_REQUEST_AUDIT_SCHEMA",
    "MATERIALIZATION_SCHEMA",
    "FILTERED_MATERIALIZATION_SCHEMA",
    "materialize_long_term_clinical_eeg_report",
]
