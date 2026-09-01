"""Clinician-facing presentation separated from the machine audit receipt.

The existing grounded renderer deliberately carries full provenance, model
identity, event identifiers, hashes, and raw reason codes.  Those fields are
valuable for audit but inappropriate in a short clinical paragraph.  This
module consumes the same facts-locked object, renders a concise clinical view,
and retains the original report unchanged as a separate machine-audit object.
It never invokes an LLM and never creates a fact that is absent from the typed
input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Final

from .clinical_reporting import (
    ClinicalReportFactsV2,
    EventScalpPhenotypeAbstention,
    EventScalpPhenotypeEvidence,
    GroundedChineseDiagnosticReport,
    _ARTIFACT_ZH,
    _MONTAGE_STABILITY_ZH,
    _frequency_phrase,
    _hypothesis_location_zh,
    render_grounded_chinese_diagnostic_report,
)


CLINICAL_PRESENTATION_SCHEMA: Final[str] = "soz-clinical-presentation.v1"
CLINICAL_PRESENTATION_POLICY: Final[str] = (
    "facts_locked_deterministic_clinical_text_with_separate_machine_audit"
)

_FORBIDDEN_UNSUPPORTED_CLAIMS: Final[tuple[str, ...]] = (
    "首先出现",
    "最早可见",
    "最早物理电极",
    "传播至",
    "传播到",
    "受累/范围扩展",
    "持续演变的节律性活动",
    "皮层SOZ可疑位于",
    "皮层SOZ位于",
    "轻度肌电",
)
_TECHNICAL_TEXT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"sha256", re.IGNORECASE),
    re.compile(r"checkpoint", re.IGNORECASE),
    re.compile(r"(?:patient|event)[-_][a-z0-9]", re.IGNORECASE),
    re.compile(r"原因代码"),
)


def _join(phrases: list[str]) -> str:
    return "。".join(value.rstrip("。") for value in phrases if value) + "。"


def _event_phrase(
    event: EventScalpPhenotypeEvidence,
    *,
    clinical_event_anchor_sec: float | None,
) -> str:
    receipt = event.receipt
    start = float(event.onset_start_sec)
    end = float(event.onset_end_sec)
    if clinical_event_anchor_sec is not None:
        if receipt.time_coordinate_semantics != "recording_start_seconds":
            raise ValueError(
                "A clinical event anchor is valid only for recording-start coordinates"
            )
        if (
            isinstance(clinical_event_anchor_sec, bool)
            or not isinstance(clinical_event_anchor_sec, (int, float))
            or not math.isfinite(float(clinical_event_anchor_sec))
        ):
            raise ValueError("clinical_event_anchor_sec must be finite")
        start -= float(clinical_event_anchor_sec)
        end -= float(clinical_event_anchor_sec)
        origin = "相对临床事件锚点"
    elif receipt.time_coordinate_semantics == "recording_start_seconds":
        origin = "自记录起"
    elif receipt.time_coordinate_semantics == "event_window_start_seconds":
        origin = "自事件窗起"
    else:  # protected again at presentation boundary
        raise ValueError("Unsupported event time-coordinate semantics")
    edges = "及".join(
        value.replace("-", "–") for value in event.first_visible_derivations
    )
    parts = [
        f"事件级头皮证据：{origin}{start:.1f}–{end:.1f}秒的算法证据支持窗内，"
        f"按固定检测规则标记的首批持续变化候选双极导联为{edges}"
    ]
    if event.rhythm_state is not None:
        # Producer v1 pooled frequency variation across both time and edges.
        # It therefore supports a spectral-rhythm feature, but not a clinical
        # temporal-evolution assertion in the clinician-facing layer.
        parts.append("局部频谱门控支持节律性特征")
    frequency = _frequency_phrase(event.frequency_range_hz)
    if frequency:
        parts.append("局部" + frequency)
    return "，".join(parts)


def _later_phrase(event: EventScalpPhenotypeEvidence) -> str:
    if not event.later_visible_derivations and event.later_visible_region_zh is None:
        return ""
    destination = event.later_visible_region_zh or "、".join(
        value.replace("-", "–") for value in event.later_visible_derivations
    )
    timing = (
        f"约{event.later_visible_delay_sec:.1f}秒后"
        if event.later_visible_delay_sec is not None
        else "随后"
    )
    return (
        f"{timing}在{destination}检测到后续持续变化候选；"
        "该先后次序仅描述头皮可见信号，不是传播路径、SOZ起始顺序或"
        "皮层传播速度真值"
    )


def _montage_phrase(event: EventScalpPhenotypeEvidence) -> str:
    if event.montage_stability in (None, "not_assessed"):
        return ""
    return (
        _MONTAGE_STABILITY_ZH[event.montage_stability]
        + "；共同参考在双极差分中相消，该结果仅为预处理/提取一致性检查，"
        "不是独立定位复现"
    )


def _artifact_phrase(event: EventScalpPhenotypeEvidence) -> str:
    if event.artifact_assessed is not True:
        return "当前未形成经验证的伪迹类型及严重度结论"
    if not event.artifact_types:
        return "预定义伪迹评估未提示主导伪迹"
    names = "、".join(_ARTIFACT_ZH[value] for value in event.artifact_types)
    if event.artifact_burden is None:
        return f"预定义伪迹评估记录到{names}伪迹"
    return f"预定义伪迹评估记录到{names}伪迹，连续负担指标为{event.artifact_burden:.2f}"


def _ranking_phrase(facts: ClinicalReportFactsV2) -> str:
    ranking = facts.patient_ranking
    spatial = ranking.spatial_report
    channels = "/".join(spatial.top_channels)
    tie = "并列首位候选" if len(spatial.top_channels) > 1 else "首位候选"
    prefix = f"患者级结果综合{ranking.aggregation_event_count}次发作"
    location = _hypothesis_location_zh(spatial)
    if ranking.uncertainty.abstain:
        return (
            f"{prefix}；C18 SOZ-reference排序尚未达到自动采信条件，"
            f"当前{tie}为{channels}，其确定性头皮区域投影为{location}，"
            "仅供医生复核"
        )
    return (
        f"{prefix}；C18 SOZ-reference排序{tie}为{channels}，"
        f"确定性头皮空间投影为{location}"
    )


def _uncertainty_phrase(facts: ClinicalReportFactsV2) -> str:
    uncertainty = facts.patient_ranking.uncertainty
    messages: list[str] = []
    if uncertainty.abstain:
        if "selective_threshold_undefined" in uncertainty.abstention_reason_codes:
            messages.append("尚无独立校准的自动采信阈值")
        else:
            messages.append("模型不确定性触发人工复核")
    if uncertainty.signal_quality_uncertainty is not None:
        messages.append("信号质量可靠性指标未经独立临床校准")
    if uncertainty.epistemic_uncertainty is not None:
        messages.append("模型分歧指标未经独立临床校准")
    return "；".join(messages)


def _reference_phrase(facts: ClinicalReportFactsV2) -> str:
    receipt = facts.final_score_reference_disagreement_receipt
    if receipt is None:
        return ""
    state = "一致" if receipt.top1_reference_agreement else "不一致"
    return (
        f"同一冻结定位器在C-CAR19与C-REF19下的患者级首位候选{state}；"
        "该结果仅描述参考敏感性，不代表定位正确性"
    )


@dataclass(frozen=True)
class ClinicalDiagnosticPresentation:
    clinical_text: str
    report_status: str
    machine_audit: dict[str, object]
    clinical_event_anchor_sec: float | None
    facts_locked: bool = True
    llm_used: bool = False
    presentation_policy: str = CLINICAL_PRESENTATION_POLICY
    schema_version: str = CLINICAL_PRESENTATION_SCHEMA

    def __post_init__(self) -> None:
        if not self.clinical_text.strip():
            raise ValueError("clinical_text must be non-empty")
        if any(value in self.clinical_text for value in _FORBIDDEN_UNSUPPORTED_CLAIMS):
            raise ValueError("clinical_text contains an unsupported clinical claim")
        if any(pattern.search(self.clinical_text) for pattern in _TECHNICAL_TEXT_PATTERNS):
            raise ValueError("clinical_text leaks machine-audit details")
        if type(self.facts_locked) is not bool or not self.facts_locked:
            raise ValueError("clinical presentation must remain facts locked")
        if type(self.llm_used) is not bool or self.llm_used:
            raise ValueError("clinical presentation must be deterministic")
        if self.presentation_policy != CLINICAL_PRESENTATION_POLICY:
            raise ValueError("unsupported presentation policy")
        if self.schema_version != CLINICAL_PRESENTATION_SCHEMA:
            raise ValueError("unsupported presentation schema")


def render_clinical_diagnostic_presentation(
    facts: ClinicalReportFactsV2,
    *,
    clinical_event_anchor_sec: float | None = None,
) -> ClinicalDiagnosticPresentation:
    """Render a concise clinical paragraph and retain a separate full audit."""

    if not isinstance(facts, ClinicalReportFactsV2):
        raise TypeError("facts must be ClinicalReportFactsV2")
    grounded: GroundedChineseDiagnosticReport = (
        render_grounded_chinese_diagnostic_report(facts)
    )
    event = facts.event_phenotype
    if isinstance(event, EventScalpPhenotypeAbstention):
        event_phrase = "事件级头皮证据未形成可报告的持续变化候选"
        later = montage = ""
        artifact = "当前未形成经验证的伪迹类型及严重度结论"
    else:
        event_phrase = _event_phrase(
            event, clinical_event_anchor_sec=clinical_event_anchor_sec
        )
        later = _later_phrase(event)
        montage = _montage_phrase(event)
        artifact = _artifact_phrase(event)
    limitation = (
        "该结果为头皮电极临床参考候选，不等同侵入式皮层SOZ、致痫区或手术靶点，"
        "需由医生结合症状学、影像和侵入式检查复核"
    )
    clinical_text = _join(
        [
            event_phrase,
            later,
            montage,
            artifact,
            _ranking_phrase(facts),
            _uncertainty_phrase(facts),
            _reference_phrase(facts),
            limitation,
        ]
    )
    return ClinicalDiagnosticPresentation(
        clinical_text=clinical_text,
        report_status=grounded.report_status,
        machine_audit=asdict(grounded),
        clinical_event_anchor_sec=(
            None
            if clinical_event_anchor_sec is None
            else float(clinical_event_anchor_sec)
        ),
    )


__all__ = [
    "CLINICAL_PRESENTATION_POLICY",
    "CLINICAL_PRESENTATION_SCHEMA",
    "ClinicalDiagnosticPresentation",
    "render_clinical_diagnostic_presentation",
]
