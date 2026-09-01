"""Materialize trustworthy event segments from one detector-scanned long EEG.

This stage sits strictly after detector selection and frozen v29 research
ranking.  It replays every selected ``[-12,+48]`` second window with the exact
v29 loader/hash policy, renders one de-identified waveform, and creates a
strict EEG-only fact ledger containing:

* immutable 60-second preprocessing metadata; and
* one *uncertain* detector candidate-support occurrence; and
* optional abstention-capable, signal-qualified quantitative change facts.

The detector support interval is never described as a confirmed seizure
onset, termination, or duration.  No annotation, spreadsheet, clinical
history, diagnosis, impression, confirmed onset, clinical spread or postictal
claim is admitted to the fact ledger.  Quantitative derivation/frequency/
rhythm values require the frozen signal-finding gates and remain uncertain
algorithm candidates.  Their content-bound receipt and exact waveform
evidence binding are persisted independently.  Research electrode rankings
are injected only after the unranked drafts have been built, through the v29
finalizer.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping

import torch

from src.clinical_eeg_report import validate_report_payload
from src.soz.data.edf import CausalEDFConfig, load_standard19_edf_event
from src.soz.geometry import STANDARD_19
from .schema import (
    BOUNDARY_POLICY,
    CANDIDATE_SEMANTICS,
    EVENT_SEGMENT_RECEIPT_SCHEMA_VERSION,
    FIXED_EVENT_ANCHOR_OFFSET_SECONDS,
    FIXED_EVENT_WINDOW_SECONDS,
    FIXED_SEGMENT_DURATION_SECONDS,
    WAVEFORM_SELECTION_POLICY,
    canonical_payload_sha256,
    validate_long_term_event_segment_receipt,
    validate_long_term_seizure_detection_manifest,
)
from .analysis_selection import bind_long_term_eeg_analysis_selection
from .signal_findings import (
    DEFAULT_SIGNAL_FINDING_POLICY,
    MORPHOLOGY_PRODUCER_PROMOTION_STATUS,
    SIGNAL_FINDINGS_PRODUCER_ID,
    SIGNAL_FINDINGS_RECEIPT_SCHEMA,
    SPATIAL_SPREAD_PRODUCER_PROMOTION_STATUS,
    SignalFindingResult,
    extract_signal_qualified_event_findings,
)


MATERIALIZATION_SCHEMA_VERSION = "long_term_event_segment_materialization_v1"
MATERIALIZATION_STATUS = "completed_unsigned_research_event_segment_receipts"
CANDIDATE_SUPPORT_SEMANTICS = (
    "detector_candidate_support_not_confirmed_seizure_boundaries"
)
SIGNAL_FINDING_BINDING_SCHEMA_VERSION = (
    "long_term_signal_finding_binding_receipt_v1"
)
SIGNAL_FINDING_ALLOWED_FACT_TYPES = frozenset(
    {
        "algorithmic_sustained_eeg_change",
    }
)
_CURRENT_SIGNAL_FINDING_VALUE_KEYS = frozenset(
    {
        "start_offset_seconds",
        "end_offset_seconds",
        "derivations",
        "frequency_hz",
        "frequency_band",
        "amplitude_uv",
        "rhythmicity",
        "quantitative_trajectory",
        "later_derivation_changes",
        "candidate_return_to_baseline_offset_seconds",
        "qualification",
        "text_zh",
    }
)
_CURRENT_SIGNAL_FINDING_REQUIRED_VALUE_KEYS = frozenset(
    {
        "start_offset_seconds",
        "end_offset_seconds",
        "derivations",
        "frequency_hz",
        "frequency_band",
        "amplitude_uv",
        "rhythmicity",
        "qualification",
        "text_zh",
    }
)
_CURRENT_SIGNAL_FINDING_FORBIDDEN_CLINICAL_TEXT_RE = re.compile(
    r"(?:\b(?:spikes?|sharp(?:\s+waves?)?|IEDs?|ESz|LVFA|electrodecrement|"
    r"ictal\s+(?:onset|evolution|spread|termination)|SOZ)\b|"
    r"棘波|尖波|癫痫样放电|电图发作|脑电发作|低电压快活动|"
    r"(?:电压|电极)递减|发作期演变|临床演变|发作(?:起始|传播|扩散|终止)|"
    r"(?:弥漫(?:性)?|广泛性|双侧同步)(?:分布|起始|发作)|"
    r"皮层\s*SOZ|致痫区|致痫灶|手术靶点)",
    re.IGNORECASE,
)
FIXED_PREPROCESSING_CONFIG = CausalEDFConfig(
    reference_policy="unlabeled_common_car19",
)

_TIME_TOLERANCE = 1e-6
EventLoader = Callable[..., object]
WaveformRenderer = Callable[..., object]
SignalFindingExtractor = Callable[..., SignalFindingResult]


def _same_time(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        left_number = float(left)
        right_number = float(right)
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        math.isfinite(left_number)
        and math.isfinite(right_number)
        and abs(left_number - right_number) <= _TIME_TOLERANCE
    )


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("hash source must be a regular non-symlinked file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_recording_source(path: str | Path) -> Path:
    source = Path(path)
    if source.is_symlink():
        raise ValueError("recording EDF must not be a symbolic link")
    resolved = source.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("recording EDF must be a regular file")
    if resolved.suffix.lower() != ".edf":
        raise ValueError("recording source must be an EDF file")
    return resolved


def _receipt_mapping(value: object, context: str) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        payload = asdict(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError(f"{context} must be a dataclass or mapping")
    # Prove that the receipt can be canonically persisted without NaN/Inf.
    json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return payload


def _checked_signal(loaded: object) -> torch.Tensor:
    window = getattr(loaded, "window", None)
    data = getattr(window, "data", None)
    if not isinstance(data, torch.Tensor):
        raise TypeError("EDF loader must return a tensor-backed event window")
    signal = data.detach().cpu().to(torch.float32).contiguous()
    if tuple(signal.shape) != (19, 12_000) or not torch.isfinite(signal).all():
        raise ValueError("replayed event must be finite float32 [19,12000]")
    return signal


def _support_interval(
    candidate: Mapping[str, Any],
    *,
    anchor: float,
) -> tuple[float, float]:
    start = float(candidate["start_offset_seconds"]) - anchor
    stop = float(candidate["stop_offset_seconds"]) - anchor
    if (
        start < FIXED_EVENT_WINDOW_SECONDS[0] - _TIME_TOLERANCE
        or stop > FIXED_EVENT_WINDOW_SECONDS[1] + _TIME_TOLERANCE
        or stop <= start
    ):
        raise ValueError(
            "detector candidate support interval is not wholly contained in "
            "the fixed event window"
        )
    return start, stop


def _fact(
    *,
    fact_id: str,
    section: str,
    fact_type: str,
    state: str,
    value: Mapping[str, Any],
    source_id: str,
    method: str,
    evidence_id: str,
    eeg_event_id: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "fact_id": fact_id,
        "section": section,
        "fact_type": fact_type,
        "state": state,
        "value": dict(value),
        "provenance": {
            "source_type": "signal_algorithm",
            "source_id": source_id,
            "method": method,
        },
        "verification": {"status": "algorithm_candidate"},
        "evidence_ids": [evidence_id],
    }
    if eeg_event_id is not None:
        result["eeg_event_id"] = eeg_event_id
    return result


def _minimal_event_report(
    *,
    recording_id: str,
    patient_pseudonym: str,
    candidate_id: str,
    eeg_event_id: str,
    event_number: int,
    processed_window_hash: str,
    preprocessing_receipt_hash: str,
    figure_hash: str,
    support_relative_to_anchor: tuple[float, float],
) -> tuple[dict[str, Any], str, str]:
    identity = {
        "recording_id": recording_id,
        "patient_pseudonym": patient_pseudonym,
        "candidate_id": candidate_id,
        "eeg_event_id": eeg_event_id,
        "processed_window_sha256": processed_window_hash,
        "preprocessing_receipt_sha256": preprocessing_receipt_hash,
        "figure_sha256": figure_hash,
        "candidate_support_interval_relative_to_anchor": list(
            support_relative_to_anchor
        ),
    }
    digest = canonical_payload_sha256(identity)
    suffix = digest[:16].upper()
    metadata_evidence_id = f"EEG-PROC-{digest[:20]}"
    waveform_evidence_id = f"EEG-WAVE-{digest[:20]}"
    metadata_source_id = f"EEG-PROC-ALG-{digest[:20]}"
    detector_source_id = f"EEG-DET-ALG-{digest[:20]}"
    occurrence_id = f"F-OCC-{suffix}"
    local_start = (
        support_relative_to_anchor[0] + FIXED_EVENT_ANCHOR_OFFSET_SECONDS
    )
    support_duration = (
        support_relative_to_anchor[1] - support_relative_to_anchor[0]
    )
    facts = [
        _fact(
            fact_id=f"F-DUR-{suffix}",
            section="metadata",
            fact_type="recording_duration",
            state="present",
            value={"duration_seconds": FIXED_SEGMENT_DURATION_SECONDS},
            source_id=metadata_source_id,
            method="固定窗头皮脑电预处理绑定",
            evidence_id=metadata_evidence_id,
        ),
        _fact(
            fact_id=f"F-SETUP-{suffix}",
            section="metadata",
            fact_type="electrode_setup",
            state="present",
            value={
                "system": "international_10_20",
                "electrodes": list(STANDARD_19),
                "montages": ["common_average"],
                "reference": "average",
            },
            source_id=metadata_source_id,
            method="固定窗头皮脑电预处理绑定",
            evidence_id=metadata_evidence_id,
        ),
        _fact(
            fact_id=f"F-ACQ-{suffix}",
            section="metadata",
            fact_type="acquisition_settings",
            state="present",
            value={
                "sampling_rate_hz": 200.0,
                "low_cut_hz": 0.5,
                "high_cut_hz": 45.0,
            },
            source_id=metadata_source_id,
            method="固定窗头皮脑电预处理绑定",
            evidence_id=metadata_evidence_id,
        ),
        _fact(
            fact_id=occurrence_id,
            section="ictal",
            fact_type="electrographic_event_occurrence",
            state="uncertain",
            value={
                "event_number": event_number,
                "start_offset_seconds": local_start,
                "duration_seconds": support_duration,
                "event_class": "uncertain_electrographic_pattern",
                "time_coordinate": "segment_start_seconds",
                "text_zh": (
                    "该区间仅为检测器待复核候选支持范围，不表示已确认的"
                    "脑电发作起始或终止。"
                ),
            },
            source_id=detector_source_id,
            method="全记录头皮脑电检测器候选支持区间投影",
            evidence_id=waveform_evidence_id,
            eeg_event_id=eeg_event_id,
        ),
    ]
    report = validate_report_payload(
        {
            "schema_version": "clinical_eeg_report_v1",
            "report_id": f"CER-LONG-{digest[:24]}",
            "patient_pseudonym": patient_pseudonym,
            "facts": facts,
            "eeg_event_ids": [eeg_event_id],
            "impression_fact_ids": [],
        }
    ).to_dict()
    return report, occurrence_id, waveform_evidence_id


def _validated_signal_finding_result(
    value: object,
    *,
    support_relative_to_anchor: tuple[float, float],
) -> SignalFindingResult:
    """Fail closed on extractor drift before a finding reaches the ledger."""

    if not isinstance(value, SignalFindingResult):
        raise TypeError("signal finding extractor must return SignalFindingResult")
    if value.status not in {"qualified", "abstained"}:
        raise ValueError("signal finding result status is unsupported")
    if not isinstance(value.facts, tuple):
        raise TypeError("signal finding result facts must be an immutable tuple")
    if not isinstance(value.receipt, Mapping):
        raise TypeError("signal finding result receipt must be an object")
    receipt = deepcopy(dict(value.receipt))
    base_keys = {
        "schema_version",
        "producer_id",
        "policy_sha256",
        "status",
        "abstention_reason",
        "input_scope",
        "analysis_scope",
        "quality",
        "gates",
        "thresholds",
        "emitted_fact_types",
        "source_receipt",
    }
    qualified_keys = {
        "qualified_interval_seconds_in_segment",
        "qualified_interval_seconds_relative_to_evidence_anchor",
        "qualified_derivations",
        "frequency_band",
        "rhythmicity",
    }
    expected_keys = base_keys | (qualified_keys if value.status == "qualified" else set())
    if set(receipt) != expected_keys:
        raise ValueError("signal finding receipt has missing or unknown fields")
    if (
        receipt["schema_version"] != SIGNAL_FINDINGS_RECEIPT_SCHEMA
        or receipt["producer_id"] != SIGNAL_FINDINGS_PRODUCER_ID
        or receipt["policy_sha256"] != DEFAULT_SIGNAL_FINDING_POLICY.sha256
        or receipt["status"] != value.status
        or receipt["input_scope"]
        != "processed_standard19_fixed_event_window_only"
    ):
        raise ValueError("signal finding receipt identity or input scope drifted")
    if receipt["thresholds"] != DEFAULT_SIGNAL_FINDING_POLICY.to_dict():
        raise ValueError("signal finding receipt threshold policy drifted")

    source_scope = receipt["source_receipt"]
    if source_scope != {
        "raw_eeg_used": True,
        "edf_annotations_used": False,
        "excel_used": False,
        "clinical_data_used": False,
        "research_ranking_used": False,
    }:
        raise ValueError("signal finding receipt violates the EEG-only source scope")
    gates = receipt["gates"]
    if not isinstance(gates, Mapping) or set(gates) != {
        "sustained_change_gate_passed",
        "reproducibility_gate_passed",
        "morphology_producer_promotion_status",
        "spatial_spread_producer_promotion_status",
    }:
        raise ValueError("signal finding gate receipt is malformed")
    if (
        gates["morphology_producer_promotion_status"]
        != MORPHOLOGY_PRODUCER_PROMOTION_STATUS
        or gates["spatial_spread_producer_promotion_status"]
        != SPATIAL_SPREAD_PRODUCER_PROMOTION_STATUS
    ):
        raise ValueError("signal finding producer promotion boundary drifted")
    gate_passed = value.status == "qualified"
    if (
        gates["sustained_change_gate_passed"] is not gate_passed
        or gates["reproducibility_gate_passed"] is not gate_passed
    ):
        raise ValueError("signal finding status and qualification gates disagree")

    quality = receipt["quality"]
    if not isinstance(quality, Mapping) or set(quality) != {
        "bad_channels",
        "usable_bipolar_derivation_count",
        "candidate_window_count",
        "artifact_gate_passed",
    }:
        raise ValueError("signal finding quality receipt is malformed")
    if (
        not isinstance(quality["bad_channels"], list)
        or not isinstance(quality["usable_bipolar_derivation_count"], int)
        or not isinstance(quality["candidate_window_count"], int)
        or type(quality["artifact_gate_passed"]) is not bool
    ):
        raise TypeError("signal finding quality receipt has invalid value types")

    scope = receipt["analysis_scope"]
    if not isinstance(scope, Mapping) or set(scope) != {
        "timebase",
        "segment_duration_seconds",
        "fixed_candidate_anchor_offset_seconds",
        "evidence_anchor_offset_seconds",
        "evidence_anchor_source",
        "candidate_support_interval_relative_to_fixed_anchor",
        "evidence_interval_seconds_relative_to_anchor",
        "effective_interval_seconds_in_segment",
        "default_full_carrier_used",
        "interval_combination_policy",
    }:
        raise ValueError("signal finding analysis scope is malformed")
    expected_local = [
        support_relative_to_anchor[0] + FIXED_EVENT_ANCHOR_OFFSET_SECONDS,
        support_relative_to_anchor[1] + FIXED_EVENT_ANCHOR_OFFSET_SECONDS,
    ]
    expected_relative = list(support_relative_to_anchor)
    if (
        scope["timebase"] != "processed_segment_start_seconds"
        or not _same_time(scope["segment_duration_seconds"], FIXED_SEGMENT_DURATION_SECONDS)
        or not _same_time(
            scope["fixed_candidate_anchor_offset_seconds"],
            FIXED_EVENT_ANCHOR_OFFSET_SECONDS,
        )
        or not _same_time(
            scope["evidence_anchor_offset_seconds"],
            FIXED_EVENT_ANCHOR_OFFSET_SECONDS,
        )
        or scope["evidence_anchor_source"] != "explicit"
        or scope["candidate_support_interval_relative_to_fixed_anchor"]
        != expected_relative
        or scope["evidence_interval_seconds_relative_to_anchor"]
        != expected_relative
        or scope["effective_interval_seconds_in_segment"] != expected_local
        or scope["default_full_carrier_used"] is not False
        or scope["interval_combination_policy"] != "intersection"
    ):
        raise ValueError("signal finding receipt does not bind the requested support")

    emitted_types: list[str] = []
    facts: list[dict[str, Any]] = []
    for index, raw in enumerate(value.facts):
        if not isinstance(raw, Mapping) or set(raw) != {
            "fact_type",
            "value",
            "method",
        }:
            raise ValueError(f"signal finding fact[{index}] is not a strict candidate")
        fact_type = raw["fact_type"]
        if fact_type not in SIGNAL_FINDING_ALLOWED_FACT_TYPES:
            raise ValueError("signal finding extractor emitted a forbidden fact type")
        if not isinstance(raw["value"], Mapping):
            raise TypeError("signal finding fact value must be an object")
        finding_value = dict(raw["value"])
        if not _CURRENT_SIGNAL_FINDING_REQUIRED_VALUE_KEYS.issubset(finding_value):
            raise ValueError(
                "current signal finding is missing a required neutral value field"
            )
        unknown_value_keys = set(finding_value).difference(
            _CURRENT_SIGNAL_FINDING_VALUE_KEYS
        )
        if unknown_value_keys:
            raise ValueError(
                "current signal finding emitted a forbidden or unknown value field"
            )
        expected_qualification = {
            "producer_id": SIGNAL_FINDINGS_PRODUCER_ID,
            "policy_sha256": DEFAULT_SIGNAL_FINDING_POLICY.sha256,
            "artifact_gate_passed": True,
            "sustained_change_gate_passed": True,
            "reproducibility_gate_passed": True,
            "source_signal_only": True,
            "external_context_used": False,
            "research_ranking_used": False,
            "morphology_terms_qualified": False,
            "spatial_spread_terms_qualified": False,
        }
        if finding_value.get("qualification") != expected_qualification:
            raise ValueError(
                "current signal finding qualification differs from its frozen receipt"
            )
        text_zh = finding_value.get("text_zh")
        if not isinstance(text_zh, str) or not text_zh.strip():
            raise TypeError("current signal finding text_zh must be non-empty")
        if _CURRENT_SIGNAL_FINDING_FORBIDDEN_CLINICAL_TEXT_RE.search(text_zh):
            raise ValueError(
                "current signal finding text promotes a neutral quantitative change"
            )
        if not isinstance(raw["method"], str) or not raw["method"].strip():
            raise TypeError("signal finding fact method must be non-empty")
        emitted_types.append(str(fact_type))
        facts.append(deepcopy(dict(raw)))
    if receipt["emitted_fact_types"] != emitted_types:
        raise ValueError("signal finding receipt fact types do not match its facts")
    if value.status == "qualified":
        if value.abstention_reason is not None or receipt["abstention_reason"] is not None:
            raise ValueError("qualified signal findings cannot carry an abstention")
        if emitted_types.count("algorithmic_sustained_eeg_change") != 1:
            raise ValueError("qualified signal findings require one sustained change")
        interval = receipt["qualified_interval_seconds_in_segment"]
        if not isinstance(interval, list) or len(interval) != 2:
            raise TypeError("qualified signal interval must be a pair")
        if (
            float(interval[0]) < expected_local[0] - _TIME_TOLERANCE
            or float(interval[1]) > expected_local[1] + _TIME_TOLERANCE
            or float(interval[1]) <= float(interval[0])
        ):
            raise ValueError("qualified signal interval leaves the bound support")
    else:
        if (
            not isinstance(value.abstention_reason, str)
            or receipt["abstention_reason"] != value.abstention_reason
            or facts
        ):
            raise ValueError("abstained signal findings must retain a reason and no facts")
    # A canonical serialization check rejects NaN/Inf hidden in nested receipts.
    canonical_payload_sha256(receipt)
    return SignalFindingResult(
        status=value.status,
        abstention_reason=value.abstention_reason,
        facts=tuple(facts),
        receipt=receipt,
    )


def _append_signal_facts(
    report_payload: Mapping[str, Any],
    *,
    result: SignalFindingResult,
    candidate_id: str,
    eeg_event_id: str,
    waveform_evidence_id: str,
    processed_window_hash: str,
) -> tuple[dict[str, Any], list[str]]:
    report = deepcopy(dict(report_payload))
    fact_ids: list[str] = []
    source_digest = canonical_payload_sha256(
        {
            "producer_id": SIGNAL_FINDINGS_PRODUCER_ID,
            "policy_sha256": DEFAULT_SIGNAL_FINDING_POLICY.sha256,
            "candidate_id": candidate_id,
            "eeg_event_id": eeg_event_id,
            "processed_window_sha256": processed_window_hash,
        }
    )
    source_id = f"EEG-SIGNAL-ALG-{source_digest[:20]}"
    for index, finding in enumerate(result.facts, start=1):
        fact_digest = canonical_payload_sha256(
            {
                "source_digest": source_digest,
                "sequence_index": index,
                "finding": finding,
            }
        )
        fact_id = f"F-SIG-{fact_digest[:20].upper()}"
        fact_ids.append(fact_id)
        report["facts"].append(
            _fact(
                fact_id=fact_id,
                section="ictal",
                fact_type=str(finding["fact_type"]),
                state="uncertain",
                value=finding["value"],
                source_id=source_id,
                method=str(finding["method"]),
                evidence_id=waveform_evidence_id,
                eeg_event_id=eeg_event_id,
            )
        )
    return validate_report_payload(report).to_dict(), fact_ids


def _signal_finding_binding_receipt(
    *,
    recording_id: str,
    candidate_id: str,
    eeg_event_id: str,
    processed_window_hash: str,
    preprocessing_receipt_hash: str,
    figure_hash: str,
    waveform_evidence_id: str,
    result: SignalFindingResult,
    fact_ids: list[str],
) -> dict[str, Any]:
    body = {
        "schema_version": SIGNAL_FINDING_BINDING_SCHEMA_VERSION,
        "recording_id": recording_id,
        "candidate_id": candidate_id,
        "eeg_event_id": eeg_event_id,
        "processed_window_sha256": processed_window_hash,
        "preprocessing_receipt_sha256": preprocessing_receipt_hash,
        "waveform_figure_sha256": figure_hash,
        "waveform_evidence_id": waveform_evidence_id,
        "status": result.status,
        "abstention_reason": result.abstention_reason,
        "signal_finding_receipt": deepcopy(dict(result.receipt)),
        "emitted_fact_ids": list(fact_ids),
        "emitted_fact_types": [str(item["fact_type"]) for item in result.facts],
        "adaptive_search_receipt_status": (
            "not_bound_requires_upstream_refined_v29_replay"
        ),
        "scope_receipt": {
            "processed_eeg_signal_only": True,
            "edf_annotations_used": False,
            "excel_used": False,
            "clinical_context_used": False,
            "research_ranking_used": False,
            "confirmed_onset_or_spread_generated": False,
            "diagnosis_or_impression_generated": False,
        },
    }
    binding_id = f"SIG-BIND-{canonical_payload_sha256(body)[:20]}"
    return {"binding_receipt_id": binding_id, **body}


def render_detector_candidate_waveform_png(
    eeg: torch.Tensor,
    output: Path,
    *,
    event_id: str,
    candidate_support_interval_relative_to_anchor: tuple[float, float],
) -> None:
    """Render a presentation-only standard-19 waveform without onset claims."""

    signal = eeg.detach().cpu().to(torch.float32).contiguous()
    if tuple(signal.shape) != (19, 12_000) or not torch.isfinite(signal).all():
        raise ValueError("waveform renderer requires finite float32 [19,12000]")
    start, stop = candidate_support_interval_relative_to_anchor
    if (
        start < FIXED_EVENT_WINDOW_SECONDS[0]
        or stop > FIXED_EVENT_WINDOW_SECONDS[1]
        or stop <= start
    ):
        raise ValueError("waveform candidate support interval is invalid")

    # Keep plotting imports lazy so schema-only consumers and unit tests do not
    # acquire a Matplotlib/font dependency.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/clinical-eeg-long-matplotlib")
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import numpy as np  # noqa: PLC0415
    from matplotlib import pyplot as plt  # noqa: PLC0415

    output.parent.mkdir(parents=True, exist_ok=True)
    values_uv = signal.numpy().astype(np.float64, copy=False) * 1e6
    robust = float(np.quantile(np.abs(values_uv), 0.99))
    spacing_uv = max(40.0, min(1000.0, 2.0 * robust))
    time_seconds = np.arange(12_000, dtype=np.float64) / 200.0 - 12.0
    baselines = np.arange(18, -1, -1, dtype=np.float64)

    figure, axis = plt.subplots(figsize=(18, 11))
    for channel_index, baseline in enumerate(baselines):
        axis.plot(
            time_seconds,
            baseline + values_uv[channel_index] / spacing_uv,
            color="#172033",
            linewidth=0.42,
            alpha=0.9,
            rasterized=True,
        )
    axis.axvspan(
        start,
        stop,
        color="#f4b942",
        alpha=0.24,
        label="Detector candidate support (not confirmed boundaries)",
    )
    axis.axvline(
        0.0,
        color="#b42318",
        linestyle="--",
        linewidth=1.2,
        label="Detector candidate anchor",
    )
    axis.set_xlim(*FIXED_EVENT_WINDOW_SECONDS)
    axis.set_ylim(-1.0, 19.0)
    axis.set_yticks(baselines)
    axis.set_yticklabels(STANDARD_19, fontsize=8)
    axis.set_xlabel("Seconds relative to detector candidate anchor")
    axis.set_ylabel("Standard-19 common-average channels")
    axis.set_title(f"Processed scalp EEG candidate window · {event_id}")
    axis.grid(axis="x", color="#d8deea", linewidth=0.5, alpha=0.75)
    axis.legend(loc="upper right", fontsize=8, framealpha=0.9)
    figure.text(
        0.5,
        0.012,
        (
            "0.5–45 Hz, 200 Hz, standard-19 CAR. Shading is detector support "
            "only; it is not a confirmed seizure onset, offset, or diagnosis."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    figure.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
        metadata={
            "Title": f"Processed scalp EEG candidate window {event_id}",
            "Description": CANDIDATE_SUPPORT_SEMANTICS,
        },
    )
    plt.close(figure)


def _bind_inputs(
    detection_manifest: object,
    event_registry: object,
    ranking_manifest: object,
    analysis_selection: object | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[tuple[dict[str, Any], dict[str, Any]]],
]:
    # Imported lazily so the v29 module can itself import the package's schema
    # without re-entering this post-ranking materializer during package init.
    from src.soz.v29_long_recording_inference import (
        resolve_v29_event_id_assignment,
        validate_v29_candidate_ranking_manifest,
    )

    detection = validate_long_term_seizure_detection_manifest(detection_manifest)
    ranking = validate_v29_candidate_ranking_manifest(ranking_manifest)
    assignment = resolve_v29_event_id_assignment(event_registry, detection)
    selection = (
        bind_long_term_eeg_analysis_selection(
            analysis_selection,
            detection,
            event_id_assignment=assignment,
        )
        if analysis_selection is not None
        else None
    )

    for key in ("recording_id", "patient_pseudonym", "source_signal_sha256"):
        if ranking[key] != detection[key]:
            raise ValueError(f"v29 ranking {key} differs from detection manifest")
    if not _same_time(
        ranking["recording_duration_seconds"],
        detection["recording_duration_seconds"],
    ):
        raise ValueError("v29 ranking duration differs from detection manifest")
    if ranking["candidate_semantics"] != CANDIDATE_SEMANTICS:
        raise ValueError("v29 ranking promotes detector candidate semantics")

    selected = {
        item["candidate_id"]: item
        for item in detection["merge_candidates"]
        if item["decision_available"] is True
        and item["decision"] == "selected_for_event_analysis"
    }
    assignment_by_candidate = {
        item["candidate_id"]: item["eeg_event_id"]
        for item in assignment["assignments"]
    }
    ranking_by_candidate = {
        item["candidate_id"]: item for item in ranking["events"]
    }
    if selection is None:
        if not (
            set(selected)
            == set(assignment_by_candidate)
            == set(ranking_by_candidate)
        ):
            raise ValueError(
                "detection, event registry and v29 ranking do not cover the same candidates"
            )
    else:
        analyzable = {
            item["candidate_id"]: item
            for item in selection["events"]
            if item["analysis_disposition"] == "analyzable"
        }
        if set(assignment_by_candidate) != set(selected):
            raise ValueError("event registry does not cover detector selection")
        if set(ranking_by_candidate) != set(analyzable):
            raise ValueError("v29 ranking does not exactly cover analyzable candidates")
        if ranking.get("analysis_selection_sha256") != canonical_payload_sha256(
            selection
        ):
            raise ValueError("v29 ranking is not bound to analysis selection")
        for candidate_id, ranking_event in ranking_by_candidate.items():
            selected_event = analyzable[candidate_id]
            if (
                selected_event["eeg_event_id"] != ranking_event["eeg_event_id"]
                or selected_event["pre_ranking_window_receipt_sha256"]
                != ranking_event["pre_ranking_window_receipt_sha256"]
                or selected_event["processed_window_sha256"]
                != ranking_event["processed_window_sha256"]
            ):
                raise ValueError(
                    "v29 ranking event differs from analyzable selection receipt"
                )

    ordered: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for ranking_event in ranking["events"]:
        candidate_id = ranking_event["candidate_id"]
        candidate = selected[candidate_id]
        if assignment_by_candidate[candidate_id] != ranking_event["eeg_event_id"]:
            raise ValueError("event registry identity differs from v29 ranking")
        if not _same_time(
            candidate["anchor_offset_seconds"],
            ranking_event["candidate_anchor_offset_seconds"],
        ):
            raise ValueError("detector candidate anchor differs from v29 ranking")
        _support_interval(
            candidate,
            anchor=float(ranking_event["candidate_anchor_offset_seconds"]),
        )
        ordered.append((candidate, ranking_event))
    if ordered != sorted(
        ordered,
        key=lambda pair: (
            pair[1]["candidate_anchor_offset_seconds"],
            pair[1]["eeg_event_id"],
        ),
    ):
        raise ValueError("v29 ranking events are not in recording-time order")
    return detection, ranking, ordered


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def materialize_long_term_event_segments(
    *,
    recording_path: str | Path,
    detection_manifest: object,
    event_registry: object,
    ranking_manifest: object,
    analysis_selection: object | None = None,
    output_dir: str | Path,
    event_loader: EventLoader = load_standard19_edf_event,
    waveform_renderer: WaveformRenderer = render_detector_candidate_waveform_png,
    signal_finding_extractor: SignalFindingExtractor = (
        extract_signal_qualified_event_findings
    ),
) -> dict[str, Any]:
    """Atomically publish final event receipts and their replayed waveforms."""

    # These helpers are post-ranking consumers.  Keeping the dependency local
    # avoids an eager package-level cycle with v29_long_recording_inference.
    from src.soz.v29_long_recording_inference import (
        finalize_v29_ranked_segment_drafts,
        preprocessing_receipt_sha256,
        processed_window_sha256,
    )

    detection, ranking, ordered = _bind_inputs(
        detection_manifest,
        event_registry,
        ranking_manifest,
        analysis_selection,
    )
    source = _regular_recording_source(recording_path)
    source_hash = _file_sha256(source)
    if source_hash != detection["source_signal_sha256"]:
        raise ValueError("recording EDF SHA-256 differs from detection manifest")
    if (
        not callable(event_loader)
        or not callable(waveform_renderer)
        or not callable(signal_finding_extractor)
    ):
        raise TypeError(
            "event_loader, waveform_renderer and signal_finding_extractor "
            "must be callable"
        )

    target = Path(output_dir).resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        drafts: list[dict[str, Any]] = []
        event_rows: list[dict[str, Any]] = []
        signal_binding_rows: list[dict[str, Any]] = []
        signal_bindings: list[dict[str, Any]] = []
        for event_number, (candidate, ranking_event) in enumerate(ordered, start=1):
            event_id = str(ranking_event["eeg_event_id"])
            candidate_id = str(ranking_event["candidate_id"])
            anchor = float(ranking_event["candidate_anchor_offset_seconds"])
            window_receipt = ranking_event["pre_ranking_window_receipt"]
            loaded = event_loader(
                source,
                anchor,
                config=FIXED_PREPROCESSING_CONFIG,
                use_edf_gap_annotations_for_signal_qc=False,
            )
            signal = _checked_signal(loaded)
            replay_processed_hash = processed_window_sha256(signal)
            if replay_processed_hash != ranking_event["processed_window_sha256"]:
                raise ValueError(
                    "replayed processed window differs from v29 ranking receipt"
                )
            edf_receipt = getattr(loaded, "edf_receipt", None)
            signal_receipt = getattr(loaded, "signal_receipt", None)
            edf_payload = _receipt_mapping(edf_receipt, "EDF load receipt")
            if edf_payload.get("edf_sha256") != source_hash:
                raise ValueError("EDF replay receipt differs from recording source")
            if not _same_time(edf_payload.get("requested_onset_sec"), anchor):
                raise ValueError("EDF replay receipt differs from candidate anchor")
            replay_preprocess_hash = preprocessing_receipt_sha256(
                edf_receipt, signal_receipt
            )
            if replay_preprocess_hash != window_receipt[
                "preprocessing_receipt_sha256"
            ]:
                raise ValueError(
                    "replayed preprocessing receipt differs from v29 window receipt"
                )

            support = _support_interval(candidate, anchor=anchor)
            figure_relative = Path("waveforms") / f"eeg_waveform_{event_number:02d}.png"
            figure_path = staging / figure_relative
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            waveform_renderer(
                signal.clone(),
                figure_path,
                event_id=event_id,
                candidate_support_interval_relative_to_anchor=support,
            )
            if not figure_path.is_file() or figure_path.is_symlink():
                raise ValueError("waveform renderer did not create a regular PNG")
            if figure_path.suffix.lower() != ".png":
                raise ValueError("waveform renderer output is not a PNG path")
            with figure_path.open("rb") as stream:
                if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                    raise ValueError("waveform renderer output lacks the PNG signature")
            # The renderer receives a clone.  Rechecking the bound tensor also
            # guards accidental mutation by future call-site refactors.
            if processed_window_sha256(signal) != replay_processed_hash:
                raise ValueError("waveform renderer mutated the processed window")
            figure_hash = _file_sha256(figure_path)

            report, occurrence_id, waveform_evidence_id = _minimal_event_report(
                recording_id=detection["recording_id"],
                patient_pseudonym=detection["patient_pseudonym"],
                candidate_id=candidate_id,
                eeg_event_id=event_id,
                event_number=event_number,
                processed_window_hash=replay_processed_hash,
                preprocessing_receipt_hash=replay_preprocess_hash,
                figure_hash=figure_hash,
                support_relative_to_anchor=support,
            )
            signal_result = _validated_signal_finding_result(
                signal_finding_extractor(
                    signal.clone(),
                    candidate_support_interval_relative_to_anchor=support,
                    evidence_interval_seconds_relative_to_anchor=support,
                    evidence_anchor_offset_seconds=(
                        FIXED_EVENT_ANCHOR_OFFSET_SECONDS
                    ),
                ),
                support_relative_to_anchor=support,
            )
            if processed_window_sha256(signal) != replay_processed_hash:
                raise ValueError("signal finding extractor mutated the processed window")
            report, signal_fact_ids = _append_signal_facts(
                report,
                result=signal_result,
                candidate_id=candidate_id,
                eeg_event_id=event_id,
                waveform_evidence_id=waveform_evidence_id,
                processed_window_hash=replay_processed_hash,
            )
            signal_binding = _signal_finding_binding_receipt(
                recording_id=detection["recording_id"],
                candidate_id=candidate_id,
                eeg_event_id=event_id,
                processed_window_hash=replay_processed_hash,
                preprocessing_receipt_hash=replay_preprocess_hash,
                figure_hash=figure_hash,
                waveform_evidence_id=waveform_evidence_id,
                result=signal_result,
                fact_ids=signal_fact_ids,
            )
            signal_binding_relative = (
                Path("signal_finding_receipts")
                / f"event_signal_findings_{event_number:02d}.json"
            )
            signal_binding_path = staging / signal_binding_relative
            _write_json(signal_binding_path, signal_binding)
            signal_binding_hash = _file_sha256(signal_binding_path)
            signal_binding_rows.append(
                {
                    "candidate_id": candidate_id,
                    "eeg_event_id": event_id,
                    "signal_finding_binding_file": (
                        signal_binding_relative.as_posix()
                    ),
                    "signal_finding_binding_sha256": signal_binding_hash,
                    "status": signal_result.status,
                    "emitted_fact_count": len(signal_fact_ids),
                }
            )
            signal_bindings.append(signal_binding)
            digest = canonical_payload_sha256(
                {
                    "candidate_id": candidate_id,
                    "eeg_event_id": event_id,
                    "processed_window_sha256": replay_processed_hash,
                    "figure_sha256": figure_hash,
                }
            )
            waveform = {
                "attachment_id": f"WAVE-LONG-{digest[:20]}",
                "evidence_id": waveform_evidence_id,
                "fact_ids": [occurrence_id, *signal_fact_ids],
                "eeg_event_id": event_id,
                "figure_file": figure_relative.as_posix(),
                "figure_sha256": figure_hash,
                "source_signal_sha256": source_hash,
                "preprocessing_receipt_sha256": replay_preprocess_hash,
                "processed_window_sha256": replay_processed_hash,
                "event_window_seconds": list(FIXED_EVENT_WINDOW_SECONDS),
                "event_anchor_offset_seconds": FIXED_EVENT_ANCHOR_OFFSET_SECONDS,
                "evidence_interval_seconds_relative_to_anchor": list(support),
                "selection_policy": WAVEFORM_SELECTION_POLICY,
                "sent_to_llm": False,
            }
            draft = {
                "schema_version": EVENT_SEGMENT_RECEIPT_SCHEMA_VERSION,
                "recording_id": detection["recording_id"],
                "patient_pseudonym": detection["patient_pseudonym"],
                "source_signal_sha256": source_hash,
                "recording_duration_seconds": detection[
                    "recording_duration_seconds"
                ],
                "candidate_id": candidate_id,
                "eeg_event_id": event_id,
                "candidate_anchor_offset_seconds": anchor,
                "requested_window_seconds": list(FIXED_EVENT_WINDOW_SECONDS),
                "segment_start_offset_seconds": window_receipt[
                    "window_start_offset_seconds"
                ],
                "segment_stop_offset_seconds": window_receipt[
                    "window_stop_offset_seconds"
                ],
                "warmup_seconds_available": window_receipt[
                    "warmup_seconds_available"
                ],
                "post_anchor_seconds_available": window_receipt[
                    "post_anchor_seconds_available"
                ],
                "boundary_policy": BOUNDARY_POLICY,
                "processed_window_sha256": replay_processed_hash,
                "preprocessing_receipt_sha256": replay_preprocess_hash,
                "event_report_payload": report,
                "waveform_attachment": waveform,
            }
            drafts.append(draft)
            event_rows.append(
                {
                    "event_number": event_number,
                    "candidate_id": candidate_id,
                    "eeg_event_id": event_id,
                    "candidate_anchor_offset_seconds": anchor,
                    "candidate_support_interval_relative_to_anchor": list(support),
                    "candidate_support_semantics": CANDIDATE_SUPPORT_SEMANTICS,
                    "processed_window_sha256": replay_processed_hash,
                    "preprocessing_receipt_sha256": replay_preprocess_hash,
                    "figure_file": figure_relative.as_posix(),
                    "figure_sha256": figure_hash,
                    "signal_finding_status": signal_result.status,
                    "signal_finding_binding_file": (
                        signal_binding_relative.as_posix()
                    ),
                    "signal_finding_binding_sha256": signal_binding_hash,
                    "signal_finding_fact_ids": list(signal_fact_ids),
                }
            )

        final_segments = finalize_v29_ranked_segment_drafts(drafts, ranking)
        if len(final_segments) != len(event_rows):
            raise RuntimeError("v29 finalizer changed the event count")
        final_segments = [
            validate_long_term_event_segment_receipt(item)
            for item in final_segments
        ]
        for segment, binding, event_row in zip(
            final_segments,
            signal_bindings,
            event_rows,
        ):
            waveform = segment["waveform_attachment"]
            if (
                binding["candidate_id"] != segment["candidate_id"]
                or binding["eeg_event_id"] != segment["eeg_event_id"]
                or binding["processed_window_sha256"]
                != segment["processed_window_sha256"]
                or binding["preprocessing_receipt_sha256"]
                != segment["preprocessing_receipt_sha256"]
                or binding["waveform_figure_sha256"] != waveform["figure_sha256"]
                or binding["waveform_evidence_id"] != waveform["evidence_id"]
                or binding["emitted_fact_ids"]
                != event_row["signal_finding_fact_ids"]
                or not set(binding["emitted_fact_ids"]).issubset(
                    waveform["fact_ids"]
                )
            ):
                raise RuntimeError(
                    "final v29 segment drifted from its signal finding binding"
                )

        receipt_rows: list[dict[str, Any]] = []
        for index, segment in enumerate(final_segments, start=1):
            relative = Path("segment_receipts") / f"event_segment_{index:02d}.json"
            destination = staging / relative
            _write_json(destination, segment)
            receipt_rows.append(
                {
                    "candidate_id": segment["candidate_id"],
                    "eeg_event_id": segment["eeg_event_id"],
                    "segment_receipt_file": relative.as_posix(),
                    "segment_receipt_sha256": _file_sha256(destination),
                }
            )
        collection_relative = Path("event_segment_receipts.json")
        _write_json(staging / collection_relative, final_segments)

        artifacts: dict[str, str] = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                artifacts[path.relative_to(staging).as_posix()] = _file_sha256(path)
        materialization = {
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "status": MATERIALIZATION_STATUS,
            "recording_id": detection["recording_id"],
            "patient_pseudonym": detection["patient_pseudonym"],
            "source_signal_sha256": source_hash,
            "recording_duration_seconds": detection[
                "recording_duration_seconds"
            ],
            "event_count": len(final_segments),
            "waveform_root": ".",
            "event_segment_receipt_collection": collection_relative.as_posix(),
            "segment_receipts": receipt_rows,
            "signal_finding_receipts": signal_binding_rows,
            "events": event_rows,
            "source_receipts": {
                "detection_manifest_payload_sha256": canonical_payload_sha256(
                    detection
                ),
                "event_registry_payload_sha256": canonical_payload_sha256(
                    event_registry
                ),
                "v29_ranking_manifest_payload_sha256": canonical_payload_sha256(
                    ranking
                ),
                "v29_model_sha256": ranking["model_receipt"]["model_sha256"],
                "signal_finding_policy_sha256": (
                    DEFAULT_SIGNAL_FINDING_POLICY.sha256
                ),
            },
            "scope_receipt": {
                "physical_edf_replayed": True,
                "processed_window_hash_verified_per_event": True,
                "preprocessing_receipt_hash_verified_per_event": True,
                "waveform_generated_per_event": True,
                "clinical_facts_limited_to_60s_metadata_and_uncertain_candidate_support": False,
                "clinical_facts_limited_to_metadata_candidate_support_and_qualified_signal_findings": True,
                "candidate_support_promoted_to_confirmed_seizure_boundaries": False,
                "qualified_signal_finding_receipt_per_event": True,
                "signal_finding_facts_bound_to_waveform_evidence": True,
                "derivation_frequency_rhythm_or_amplitude_requires_signal_qualification": True,
                "morphology_spread_postictal_or_diagnosis_generated": False,
                "adaptive_search_receipt_bound": False,
                "adaptive_refined_anchor_used_without_v29_replay": False,
                "annotation_or_excel_loaded": False,
                "raw_edf_path_persisted": False,
                "research_soz_used_in_clinical_facts": False,
                "research_soz_sent_to_llm": False,
                "unranked_segment_drafts_persisted": False,
                "rankings_injected_only_after_facts_and_waveforms": True,
                "physician_signed": False,
            },
            "claim_boundary": {
                "unsigned_research_ai_draft": True,
                "candidate_is_confirmed_seizure": False,
                "candidate_support_is_confirmed_onset_or_termination": False,
                "research_ranking_is_cortical_soz_or_treatment_target": False,
                "clinical_export_allowed_without_physician_review": False,
            },
            "artifacts": artifacts,
        }
        _write_json(staging / "manifest.json", materialization)
        for path in staging.rglob("*"):
            os.chmod(path, 0o700 if path.is_dir() else 0o600)
        os.chmod(staging, 0o700)
        os.replace(staging, target)
        os.chmod(target, 0o700)
        published = True
        return deepcopy(materialization)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "CANDIDATE_SUPPORT_SEMANTICS",
    "FIXED_PREPROCESSING_CONFIG",
    "MATERIALIZATION_SCHEMA_VERSION",
    "MATERIALIZATION_STATUS",
    "SIGNAL_FINDING_ALLOWED_FACT_TYPES",
    "SIGNAL_FINDING_BINDING_SCHEMA_VERSION",
    "materialize_long_term_event_segments",
    "render_detector_candidate_waveform_png",
]
