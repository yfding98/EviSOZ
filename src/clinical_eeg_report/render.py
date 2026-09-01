"""Deterministic HTML and dependency-free DOCX rendering.

The LLM supplies only validated text cells.  All headings, metadata, event
counts, table structure, review state and signature fields are generated here
from the fact ledger and fixed style profile. Unsupported clinical, activation
and sleep sections are omitted entirely.
"""

from __future__ import annotations

import hashlib
from html import escape as html_escape
import json
from pathlib import Path
from pathlib import PurePosixPath
import struct
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape as xml_escape
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .generation import (
    FINDING_SECTION_ORDER,
    IMPRESSION_SECTION_ORDER,
    PIPELINE_RECORD_SCHEMA,
    eeg_only_generation_report_view,
    validate_narrative_payload,
)
from .evidence import ValidatedWaveformAttachment
from .style import ClinicalEEGStyleProfile


_METADATA_LABELS = {
    "recording_modality": "监测类型",
    "recording_duration": "监测时间",
    "electrode_setup": "电极及导联",
    "acquisition_settings": "采集参数",
    "recording_quality": "脑电记录质量",
}
_IMPRESSION_LABELS = {
    "overall": "总体印象",
    "interictal": "一、发作间期",
    "ictal": "二、发作期",
    "limitations": "证据边界",
}
def _report_dict(report: Any) -> dict[str, Any]:
    value = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    if not isinstance(value, dict) or value.get("schema_version") != "clinical_eeg_report_v1":
        raise ValueError("renderer requires a validated clinical_eeg_report_v1")
    # The public renderer repeats the same quarantine as the generation
    # boundary.  Direct rendering therefore cannot bypass the pipeline and
    # expose legacy EDF annotation timing facts in the report body.
    value = eeg_only_generation_report_view(value)
    if set(value) != {
        "schema_version",
        "report_id",
        "patient_pseudonym",
        "facts",
        "eeg_event_ids",
        "impression_fact_ids",
    }:
        raise ValueError("renderer refuses non-EEG or legacy report fields")
    return value


def _validate_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != PIPELINE_RECORD_SCHEMA:
        raise ValueError("renderer requires a clinical EEG pipeline record")
    release = record.get("release")
    if not isinstance(release, Mapping) or release.get("status") != "ai_draft":
        raise ValueError("v1 renderer only emits an unsigned AI draft")
    if release.get("clinical_export_allowed") is not False:
        raise ValueError("unsigned report cannot be marked for clinical export")


def _value_text(value: Any) -> str:
    if value is None:
        return "未提供"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float, str)):
        return str(value)
    if isinstance(value, list):
        return "、".join(_value_text(item) for item in value)
    if isinstance(value, Mapping):
        for key in ("display_value_zh", "text_zh", "description_zh", "summary_zh"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
        return "；".join(
            f"{_METADATA_LABELS.get(str(key), str(key))}：{_value_text(item)}"
            for key, item in value.items()
            if key not in {"text_zh", "description_zh", "summary_zh"}
        )
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _metadata_rows(report: Mapping[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("病例代号", str(report["patient_pseudonym"])),
        ("报告代号", str(report["report_id"])),
    ]
    for fact in report["facts"]:
        if not isinstance(fact, Mapping) or fact.get("section") != "metadata":
            continue
        fact_type = str(fact.get("fact_type", "metadata"))
        value = fact.get("value")
        if fact_type == "recording_modality" and isinstance(value, Mapping):
            modality = {
                "routine_eeg": "常规脑电",
                "ambulatory_eeg": "动态脑电",
                "video_eeg": "视频脑电",
                "long_term_video_eeg": "长程视频脑电",
                "long_term_eeg": "长程脑电",
                "long_term_scalp_eeg": "长程头皮脑电",
                "eeg_segment": "脑电片段",
            }.get(str(value.get("modality")), str(value.get("modality")))
            rows.append((_METADATA_LABELS[fact_type], modality))
        elif fact_type == "recording_duration" and isinstance(value, Mapping):
            seconds = value.get("duration_seconds")
            if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
                duration = float(seconds)
                if duration >= 3600:
                    display = f"{duration / 3600.0:g}小时"
                elif duration >= 60:
                    display = f"{duration / 60.0:g}分钟"
                else:
                    display = f"{duration:g}秒"
            else:
                display = _value_text(value)
            rows.append((_METADATA_LABELS[fact_type], display))
        elif fact_type == "electrode_setup" and isinstance(value, Mapping):
            system = {
                "international_10_20": "国际10–20系统",
                "international_10_10": "国际10–10系统",
                "custom": "自定义系统",
            }.get(str(value.get("system")), str(value.get("system")))
            electrode_count = len(value.get("electrodes", [])) if isinstance(value.get("electrodes"), list) else 0
            montage_labels = {
                "longitudinal_bipolar": "纵向双极",
                "transverse_bipolar": "横向双极",
                "common_average": "共平均",
                "average": "平均参考",
                "referential": "参考导联",
            }
            raw_montages = value.get("montages", [])
            montages = "、".join(
                montage_labels.get(str(item), str(item))
                for item in raw_montages
            ) if isinstance(raw_montages, list) else _value_text(raw_montages)
            raw_reference = str(value.get("reference", "未提供"))
            reference = montage_labels.get(raw_reference, raw_reference)
            rows.append((_METADATA_LABELS[fact_type], f"{system}，{electrode_count}导；{montages}；参考={reference}"))
        elif fact_type == "acquisition_settings" and isinstance(value, Mapping):
            display = (
                f"采样率{value.get('sampling_rate_hz')} Hz；"
                f"滤波{value.get('low_cut_hz')}–{value.get('high_cut_hz')} Hz"
            )
            if value.get("notch_hz") is not None:
                display += f"；陷波{value.get('notch_hz')} Hz"
            rows.append((_METADATA_LABELS[fact_type], display))
        else:
            rows.append((_METADATA_LABELS.get(fact_type, fact_type), _value_text(value)))
    return rows


def _event_time_labels(report: Mapping[str, Any]) -> dict[str, str]:
    event_ids = [str(item) for item in report.get("eeg_event_ids", [])]
    labels: dict[str, str] = {event_id: f"脑电事件{index}" for index, event_id in enumerate(event_ids, start=1)}
    seen: set[str] = set()
    for fact in report["facts"]:
        if not isinstance(fact, Mapping) or fact.get("fact_type") != "electrographic_event_occurrence":
            continue
        event_id = fact.get("eeg_event_id")
        value = fact.get("value")
        if not isinstance(event_id, str) or not isinstance(value, Mapping):
            continue
        if event_id not in labels or event_id in seen:
            raise ValueError("EEG event occurrence does not match the declared event order")
        offset = value.get("start_offset_seconds")
        duration = value.get("duration_seconds")
        event_class = value.get("event_class")
        time_coordinate = value.get("time_coordinate", "recording_start_seconds")
        if not isinstance(offset, (int, float)) or isinstance(offset, bool):
            raise ValueError("EEG event occurrence has no numeric start offset")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            raise ValueError("EEG event occurrence has no numeric duration")
        total = float(offset)
        hours = int(total // 3600)
        minutes = int((total % 3600) // 60)
        seconds = total % 60
        seconds_text = f"{seconds:06.3f}".rstrip("0").rstrip(".")
        if seconds < 10 and not seconds_text.startswith("0"):
            seconds_text = "0" + seconds_text
        class_text = {
            "electrographic_seizure": "脑电发作",
            "electrographic_event": "脑电事件",
            "uncertain_electrographic_pattern": "意义不确定脑电图形",
        }.get(str(event_class), str(event_class))
        coordinate_text = {
            "recording_start_seconds": "自记录起",
            "segment_start_seconds": "自片段起",
        }.get(str(time_coordinate))
        if coordinate_text is None:
            raise ValueError("EEG event occurrence has an unsupported time coordinate")
        labels[event_id] = (
            f"{class_text}\n{coordinate_text}{hours:02d}:{minutes:02d}:{seconds_text}\n"
            f"持续{float(duration):g}秒"
        )
        seen.add(event_id)
    if seen != set(event_ids):
        raise ValueError("each EEG event requires one occurrence fact")
    return labels


def _pairs(rows: Sequence[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    paired: list[list[tuple[str, str]]] = []
    for index in range(0, len(rows), 2):
        group = list(rows[index : index + 2])
        if len(group) == 1:
            group.append(("", ""))
        paired.append(group)
    return paired


def _ordered_waveform_attachments(
    report: Mapping[str, Any],
    attachments: Sequence[ValidatedWaveformAttachment],
) -> tuple[ValidatedWaveformAttachment, ...]:
    if isinstance(attachments, (str, bytes)) or not isinstance(attachments, Sequence):
        raise TypeError("waveform_attachments must be a sequence")
    result = tuple(attachments)
    if not all(isinstance(item, ValidatedWaveformAttachment) for item in result):
        raise TypeError("renderer accepts only validated waveform attachments")
    evidence_ids = [item.evidence_id for item in result]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("waveform attachments contain duplicate evidence IDs")
    event_ids = [str(item) for item in report.get("eeg_event_ids", [])]
    event_order = {event_id: index for index, event_id in enumerate(event_ids)}
    if any(item.eeg_event_id not in event_order for item in result):
        raise ValueError("waveform attachment references an unknown EEG event")
    return tuple(
        sorted(result, key=lambda item: (event_order[item.eeg_event_id], item.evidence_id))
    )


def _safe_waveform_href(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError("waveform image href must be a non-empty trimmed string")
    if "\\" in value or ":" in value or value.startswith("/"):
        raise ValueError("waveform image href must be a safe relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.suffix != ".png"
    ):
        raise ValueError("waveform image href must be a canonical relative PNG path")
    return value


def _verified_attachment_png(
    attachment: ValidatedWaveformAttachment,
) -> tuple[bytes, int, int]:
    """Recheck the PNG immediately before embedding to close TOCTOU gaps."""

    path = attachment.source_path
    if path.is_symlink() or not path.is_file():
        raise ValueError("validated waveform source is no longer a regular file")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != attachment.figure_sha256:
        raise ValueError("waveform PNG changed after evidence validation")
    if len(payload) < 24 or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("waveform attachment is not a PNG")
    width, height = struct.unpack(">II", payload[16:24])
    if width <= 0 or height <= 0:
        raise ValueError("waveform PNG has invalid dimensions")
    return payload, width, height


def _waveform_audit_text(
    attachment: ValidatedWaveformAttachment,
    report: Mapping[str, Any],
) -> str:
    event_number = list(map(str, report.get("eeg_event_ids", []))).index(
        attachment.eeg_event_id
    ) + 1
    window_start, window_stop = attachment.event_window_seconds
    low_cut, high_cut = attachment.filter_hz
    representative = "；该图为患者级报告的代表性事件" if attachment.representative_event else ""
    return (
        f"脑电事件{event_number}（事件ID：{attachment.eeg_event_id}；"
        f"证据ID：{attachment.evidence_id}）。"
        f"显示窗为事件标记{window_start:g}至{window_stop:g}秒，"
        f"事件标记位于片段第{attachment.event_anchor_offset_seconds:g}秒；"
        f"{len(attachment.channel_order)}导，采样率{attachment.sampling_rate_hz:g} Hz，"
        f"滤波{low_cut:g}–{high_cut:g} Hz，共平均参考"
        f"{representative}。图像SHA256：{attachment.figure_sha256}。"
    )


def _waveform_section_html(
    report: Mapping[str, Any],
    attachments: Sequence[ValidatedWaveformAttachment],
    hrefs: Mapping[str, str] | None,
) -> str:
    ordered = _ordered_waveform_attachments(report, attachments)
    if not ordered:
        if hrefs:
            raise ValueError("waveform hrefs were supplied without validated attachments")
        return '<p class="placeholder">本报告未附可验证的脑电波形附件。</p>'
    if not isinstance(hrefs, Mapping) or set(hrefs) != {
        item.evidence_id for item in ordered
    }:
        raise ValueError("waveform hrefs must exactly match validated evidence IDs")
    figures: list[str] = []
    for attachment in ordered:
        _verified_attachment_png(attachment)
        href = _safe_waveform_href(hrefs[attachment.evidence_id])
        caption = attachment.caption_zh + "\n" + _waveform_audit_text(attachment, report)
        figures.append(
            '<figure class="waveform">'
            f'<img src="{html_escape(href, quote=True)}" '
            f'alt="{html_escape(attachment.caption_zh, quote=True)}">'
            f'<figcaption>{html_escape(caption)}</figcaption></figure>'
        )
    return "".join(figures)


def render_html(
    report: Any,
    record: Mapping[str, Any],
    style: ClinicalEEGStyleProfile,
    *,
    waveform_attachments: Sequence[ValidatedWaveformAttachment] = (),
    waveform_hrefs: Mapping[str, str] | None = None,
) -> str:
    source = _report_dict(report)
    _validate_record(record)
    raw_narrative = record.get("narrative")
    if not isinstance(raw_narrative, Mapping):
        raise TypeError("pipeline narrative is missing")
    narrative = validate_narrative_payload(raw_narrative, source)
    headings = style.section_headings_zh
    metadata = _metadata_rows(source)
    metadata_html = "".join(
        "<tr>"
        + "".join(
            f'<th scope="row">{html_escape(label)}</th><td>{html_escape(value)}</td>'
            for label, value in group
        )
        + "</tr>"
        for group in _pairs(metadata)
    )
    finding_by_id = {
        str(item["section_id"]): item
        for item in narrative.get("findings", [])
        if isinstance(item, Mapping)
    }
    finding_parts: list[str] = []
    for section_id in FINDING_SECTION_ORDER:
        if not finding_by_id[section_id].get("fact_ids"):
            continue
        finding_parts.append(
            f'<section><h3>{html_escape(str(headings[section_id]))}</h3>'
            f'<p>{html_escape(str(finding_by_id[section_id]["text_zh"]))}</p></section>'
        )
    finding_html = "".join(finding_parts)
    aliases = {
        f"EV{index}": event_id
        for index, event_id in enumerate(source.get("eeg_event_ids", []), start=1)
    }
    time_labels = _event_time_labels(source)
    event_rows = "".join(
        "<tr>"
        f'<td>{html_escape(time_labels[str(aliases[str(event["eeg_event_alias"])])])}</td>'
        f'<td>{html_escape(str(event["onset_text_zh"])) if event.get("onset_fact_ids") else ""}</td>'
        f'<td>{html_escape(str(event["evolution_spread_text_zh"])) if event.get("evolution_spread_fact_ids") else ""}</td>'
        f'<td>{html_escape(str(event["termination_postictal_text_zh"])) if event.get("termination_postictal_fact_ids") else ""}</td>'
        "</tr>"
        for event in narrative.get("events", [])
        if isinstance(event, Mapping)
    )
    columns = [html_escape(str(item)) for item in style.payload["event_table_columns_zh"]]
    impression_html = "".join(
        f'<section><h3>{html_escape(_IMPRESSION_LABELS[str(item["section_id"])])}</h3>'
        f'<p>{html_escape(str(item["text_zh"]))}</p></section>'
        for item in narrative.get("impression", [])
        if isinstance(item, Mapping) and item.get("fact_ids")
    )
    waveform_html = _waveform_section_html(
        source,
        waveform_attachments,
        waveform_hrefs,
    )
    generator = html_escape(str(record.get("generation", {}).get("generator", "unknown")))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>头皮脑电报告（AI草稿）</title>
<style>
@page {{ size: A4; margin: 16mm; }}
body {{ margin: 0 auto; max-width: 900px; padding: 28px; color: #20242b; font-family: "Noto Serif CJK SC","SimSun",serif; line-height: 1.65; }}
h1 {{ text-align: center; font-size: 26px; margin: 0; }}
.draft {{ text-align: center; color: #9c2f2f; font-weight: 700; margin: 4px 0 20px; }}
h2 {{ border-top: 3px solid #30353d; padding-top: 10px; font-size: 21px; }}
h3 {{ font-size: 17px; margin-bottom: 2px; }}
p {{ margin: 2px 0 12px; white-space: pre-wrap; }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0 18px; table-layout: fixed; }}
th,td {{ border: 1px solid #454b54; padding: 7px 8px; vertical-align: top; overflow-wrap: anywhere; }}
.metadata th {{ width: 13%; background: #f1f3f5; }}
.metadata td {{ width: 37%; }}
.placeholder p {{ color: #66707b; font-style: italic; }}
.events th {{ background: #e8edf2; text-align: center; }}
.events th:nth-child(1) {{ width: 18%; }}
.events th:nth-child(2) {{ width: 27%; }}
.events th:nth-child(3) {{ width: 30%; }}
.events th:nth-child(4) {{ width: 25%; }}
.events td:first-child {{ white-space: pre-line; }}
.waveform {{ margin: 16px 0 24px; page-break-inside: avoid; }}
.waveform img {{ width: 100%; height: auto; border: 1px solid #bbc2ca; display: block; }}
.waveform figcaption {{ color: #535d68; font-size: 13px; white-space: pre-wrap; overflow-wrap: anywhere; margin-top: 6px; }}
.review {{ margin-top: 36px; border: 1px solid #9ba3ad; padding: 12px; background: #fff8e8; }}
.audit {{ color: #5f6873; font-size: 13px; }}
</style>
</head>
<body>
<h1>头皮脑电报告</h1>
<div class="draft">AI 草稿 · 未经脑电医师签署 · 不得直接用于诊疗</div>
<table class="metadata"><tbody>{metadata_html}</tbody></table>
<h2>{html_escape(str(headings["findings"]))}</h2>
{finding_html}
<h2>{html_escape(str(headings["eeg_events"]))}</h2>
<p>共记录 {len(source.get("eeg_event_ids", []))} 个结构化脑电事件。</p>
<table class="events"><thead><tr>{''.join(f'<th>{column}</th>' for column in columns)}</tr></thead><tbody>{event_rows}</tbody></table>
<h2>相关 EEG 波形证据</h2>
{waveform_html}
<p class="audit">波形图仅为当前 EEG 事实的证据附件，不构成独立诊断；图中标记或着色区间仅用于定位待复核的信号变化。</p>
<h2>{html_escape(str(headings["impression"]))}</h2>
{impression_html}
<div class="review"><strong>审核状态：</strong>AI 草稿<br><strong>审核医师：</strong>________________<br><strong>签署日期：</strong>________________</div>
<p class="audit">生成器：{generator}。版式、脑电事件数、身份代号与签名字段均未由 LLM 控制。</p>
</body>
</html>
"""


def _w_run(text: str, *, bold: bool = False, size: int = 21) -> str:
    properties = (
        "<w:rPr>"
        '<w:rFonts w:ascii="SimSun" w:eastAsia="宋体" w:hAnsi="SimSun"/>'
        + ("<w:b/>" if bold else "")
        + f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
        "</w:rPr>"
    )
    return f"<w:r>{properties}<w:t xml:space=\"preserve\">{xml_escape(text)}</w:t></w:r>"


def _w_paragraph(text: str, *, bold: bool = False, size: int = 21, align: str | None = None) -> str:
    paragraph_properties = f'<w:pPr><w:jc w:val="{align}"/></w:pPr>' if align else ""
    lines = text.splitlines() or [""]
    runs = "<w:br/>".join(_w_run(line, bold=bold, size=size) for line in lines)
    return f"<w:p>{paragraph_properties}{runs}</w:p>"


def _w_image_paragraph(
    *,
    relationship_id: str,
    drawing_id: int,
    name: str,
    description: str,
    width_px: int,
    height_px: int,
) -> str:
    width_emu = 5_850_000
    height_emu = max(1, int(round(width_emu * height_px / width_px)))
    safe_name = xml_escape(name, {'"': "&quot;"})
    safe_description = xml_escape(description, {'"': "&quot;"})
    return (
        '<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
        '<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{width_emu}" cy="{height_emu}"/>'
        '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{drawing_id}" name="{safe_name}" '
        f'descr="{safe_description}"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/>'
        '</wp:cNvGraphicFramePr><a:graphic>'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic><pic:nvPicPr>'
        f'<pic:cNvPr id="{drawing_id}" name="{safe_name}"/>'
        '<pic:cNvPicPr/></pic:nvPicPr><pic:blipFill>'
        f'<a:blip r:embed="{relationship_id}"/>'
        '<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        '<pic:spPr><a:xfrm><a:off x="0" y="0"/>'
        f'<a:ext cx="{width_emu}" cy="{height_emu}"/>'
        '</a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        '</pic:spPr></pic:pic></a:graphicData></a:graphic>'
        '</wp:inline></w:drawing></w:r></w:p>'
    )


def _w_cell(text: str, *, bold: bool = False, width: int | None = None, shade: str | None = None) -> str:
    properties = "<w:tcPr>"
    if width is not None:
        properties += f'<w:tcW w:w="{width}" w:type="dxa"/>'
    if shade is not None:
        properties += f'<w:shd w:val="clear" w:color="auto" w:fill="{shade}"/>'
    properties += "</w:tcPr>"
    return f"<w:tc>{properties}{_w_paragraph(text, bold=bold, size=20)}</w:tc>"


def _w_table(
    rows: Sequence[Sequence[tuple[str, bool, int | None, str | None]]],
    *,
    header_rows: int = 0,
) -> str:
    if header_rows < 0 or header_rows > len(rows):
        raise ValueError("header_rows is outside the table")
    body_parts: list[str] = []
    for index, row in enumerate(rows):
        row_properties = "<w:trPr><w:tblHeader/></w:trPr>" if index < header_rows else ""
        cells = "".join(
            _w_cell(text, bold=bold, width=width, shade=shade)
            for text, bold, width, shade in row
        )
        body_parts.append(f"<w:tr>{row_properties}{cells}</w:tr>")
    body = "".join(body_parts)
    return (
        "<w:tbl><w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"6\" w:color=\"555555\"/>"
        "<w:left w:val=\"single\" w:sz=\"6\" w:color=\"555555\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"6\" w:color=\"555555\"/>"
        "<w:right w:val=\"single\" w:sz=\"6\" w:color=\"555555\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:color=\"777777\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:color=\"777777\"/>"
        "</w:tblBorders></w:tblPr>"
        + body
        + "</w:tbl>"
    )


def _document_xml(
    report: Mapping[str, Any],
    record: Mapping[str, Any],
    style: ClinicalEEGStyleProfile,
    waveform_attachments: Sequence[ValidatedWaveformAttachment] = (),
) -> str:
    raw_narrative = record.get("narrative")
    if not isinstance(raw_narrative, Mapping):
        raise TypeError("pipeline narrative is missing")
    narrative = validate_narrative_payload(raw_narrative, report)
    headings = style.section_headings_zh
    parts = [
        _w_paragraph("头皮脑电报告", bold=True, size=32, align="center"),
        _w_paragraph("AI草稿 · 未经脑电医师签署 · 不得直接用于诊疗", bold=True, size=20, align="center"),
    ]
    metadata_rows = []
    for group in _pairs(_metadata_rows(report)):
        row = []
        for label, value in group:
            row.extend(((label, True, 1500, "E9EDF2"), (value, False, 3100, None)))
        metadata_rows.append(row)
    parts.append(_w_table(metadata_rows))
    parts.append(_w_paragraph(str(headings["findings"]), bold=True, size=28))
    finding_by_id = {str(item["section_id"]): item for item in narrative["findings"]}
    for section_id in FINDING_SECTION_ORDER:
        if not finding_by_id[section_id].get("fact_ids"):
            continue
        parts.append(_w_paragraph(str(headings[section_id]) + "：", bold=True, size=22))
        parts.append(_w_paragraph(str(finding_by_id[section_id]["text_zh"]), size=21))
    parts.append(_w_paragraph(str(headings["eeg_events"]), bold=True, size=28))
    parts.append(_w_paragraph(f"共记录 {len(report.get('eeg_event_ids', []))} 个结构化脑电事件。", size=21))
    event_rows: list[list[tuple[str, bool, int | None, str | None]]] = [
        [
            (str(label), True, width, "DCE6F1")
            for label, width in zip(style.payload["event_table_columns_zh"], (1600, 2500, 2800, 2300), strict=True)
        ]
    ]
    aliases = {
        f"EV{index}": event_id
        for index, event_id in enumerate(report.get("eeg_event_ids", []), start=1)
    }
    labels = _event_time_labels(report)
    for event in narrative["events"]:
        event_rows.append(
            [
                (labels[str(aliases[str(event["eeg_event_alias"])])], False, 1600, None),
                (
                    str(event["onset_text_zh"])
                    if event.get("onset_fact_ids")
                    else "",
                    False,
                    2500,
                    None,
                ),
                (
                    str(event["evolution_spread_text_zh"])
                    if event.get("evolution_spread_fact_ids")
                    else "",
                    False,
                    2800,
                    None,
                ),
                (
                    str(event["termination_postictal_text_zh"])
                    if event.get("termination_postictal_fact_ids")
                    else "",
                    False,
                    2300,
                    None,
                ),
            ]
        )
    parts.append(_w_table(event_rows, header_rows=1))
    parts.append(_w_paragraph("相关 EEG 波形证据", bold=True, size=28))
    ordered_attachments = _ordered_waveform_attachments(report, waveform_attachments)
    if not ordered_attachments:
        parts.append(_w_paragraph("本报告未附可验证的脑电波形附件。", size=21))
    for index, attachment in enumerate(ordered_attachments, start=1):
        _, width_px, height_px = _verified_attachment_png(attachment)
        parts.append(_w_paragraph(attachment.caption_zh, bold=True, size=21))
        parts.append(
            _w_image_paragraph(
                relationship_id=f"rId{index + 1}",
                drawing_id=index,
                name=f"EEG waveform {index}",
                description=attachment.caption_zh,
                width_px=width_px,
                height_px=height_px,
            )
        )
        parts.append(_w_paragraph(_waveform_audit_text(attachment, report), size=18))
    parts.append(
        _w_paragraph(
            "波形图仅为当前 EEG 事实的证据附件，不构成独立诊断；"
            "图中标记或着色区间仅用于定位待复核的信号变化。",
            size=18,
        )
    )
    parts.append(_w_paragraph(str(headings["impression"]), bold=True, size=28))
    for item in narrative["impression"]:
        if not item.get("fact_ids"):
            continue
        section_id = str(item["section_id"])
        parts.append(_w_paragraph(_IMPRESSION_LABELS[section_id] + "：", bold=True, size=22))
        parts.append(_w_paragraph(str(item["text_zh"]), size=21))
    parts.extend(
        [
            _w_paragraph("审核状态：AI草稿", bold=True, size=21),
            _w_paragraph("审核医师：________________", size=21),
            _w_paragraph("签署日期：________________", size=21),
        ]
    )
    body = "".join(parts)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f"<w:body>{body}"
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="900" w:right="900" w:bottom="900" w:left="900" w:header="400" w:footer="400" w:gutter="0"/>'
        "</w:sectPr></w:body></w:document>"
    )


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/><w:qFormat/>
    <w:rPr><w:rFonts w:ascii="SimSun" w:eastAsia="宋体" w:hAnsi="SimSun"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>
  </w:style>
</w:styles>"""


def _zip_write(archive: ZipFile, name: str, text: str) -> None:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, text.encode("utf-8"))


def _zip_write_bytes(archive: ZipFile, name: str, payload: bytes) -> None:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, payload)


def render_docx(
    path: Path,
    report: Any,
    record: Mapping[str, Any],
    style: ClinicalEEGStyleProfile,
    *,
    waveform_attachments: Sequence[ValidatedWaveformAttachment] = (),
) -> None:
    source = _report_dict(report)
    _validate_record(record)
    ordered_attachments = _ordered_waveform_attachments(source, waveform_attachments)
    media = [
        (attachment, _verified_attachment_png(attachment)[0])
        for attachment in ordered_attachments
    ]
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        _zip_write(
            archive,
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>""",
        )
        _zip_write(
            archive,
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>""",
        )
        _zip_write(
            archive,
            "word/document.xml",
            _document_xml(source, record, style, ordered_attachments),
        )
        _zip_write(archive, "word/styles.xml", _styles_xml())
        image_relationships = "".join(
            '<Relationship '
            f'Id="rId{index + 1}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="media/eeg_waveform_{index:02d}.png"/>'
            for index, _ in enumerate(media, start=1)
        )
        _zip_write(
            archive,
            "word/_rels/document.xml.rels",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  {image_relationships}
</Relationships>""",
        )
        for index, (_, payload) in enumerate(media, start=1):
            _zip_write_bytes(
                archive,
                f"word/media/eeg_waveform_{index:02d}.png",
                payload,
            )


__all__ = ["render_docx", "render_html"]
