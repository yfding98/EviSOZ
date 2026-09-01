#!/usr/bin/env python3
"""Serve an interactive EEG viewer for the 607 DeepSOZ-to-TUSZ mappings."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import mne
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_deepsoz_llm_tusz_viewer import (  # noqa: E402
    TCP_LEADS,
    bipolar_data,
    local_event_onsets,
    split_tokens,
)

DEFAULT_CATALOG = ROOT / "outputs/deepsoz_llm_tusz_all_607_20260801/mapped_records.csv"
DEFAULT_LLM = (
    ROOT
    / "outputs/deepsoz_607_llm_qwen36_full_v3_20260801/indexes/all_records.jsonl"
)

LLM_COMPLETED_STATUSES = {
    "llm_candidate_ready",
    "llm_abstained",
    "llm_context_exhausted",
}


def clean(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def json_value(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_llm_index(path: Path) -> list[dict]:
    """Read either the legacy CSV index or the final sharded-run JSONL index."""

    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(payload)
        return rows
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return [
        {key: json_value(value) for key, value in row.to_dict().items()}
        for _, row in frame.iterrows()
    ]


class ViewerData:
    def __init__(self, catalog_path: Path, llm_path: Path, max_points: int):
        self.catalog_path = catalog_path.resolve()
        self.catalog = pd.read_csv(self.catalog_path, encoding="utf-8-sig")
        self.catalog = self.catalog.reset_index(drop=True)
        self.max_points = max_points
        self.llm_by_source: dict[str, list[dict]] = defaultdict(list)
        if llm_path.is_file():
            for source_row in load_llm_index(llm_path):
                if clean(source_row.get("dataset")).lower() != "tusz":
                    continue
                source_text = clean(source_row.get("source_file"))
                if not source_text:
                    continue
                source = str(Path(source_text).resolve())
                item = dict(source_row)
                record_path = Path(clean(source_row.get("record_path")))
                if record_path.is_file():
                    try:
                        record = json.loads(record_path.read_text(encoding="utf-8"))
                        item["llm_detail"] = record.get("llm_result") or {}
                    except (OSError, json.JSONDecodeError):
                        item["llm_detail"] = {}
                else:
                    item["llm_detail"] = {}
                self.llm_by_source[source].append(item)

    def catalog_payload(self):
        result = []
        for index, row in self.catalog.iterrows():
            llm_items = self.llm_by_source.get(
                str(Path(clean(row.get("local_edf"))).resolve()), []
            )
            success_count = sum(
                clean(item.get("record_status")) in LLM_COMPLETED_STATUSES
                for item in llm_items
            )
            failure_count = sum(
                clean(item.get("record_status")) == "processing_failed_closed"
                for item in llm_items
            )
            result.append({
                "index": int(index),
                "deepsoz_patient": clean(row.get("deepsoz_patient")),
                "deepsoz_record": clean(row.get("deepsoz_record")),
                "deepsoz_soz": clean(row.get("deepsoz_soz_electrodes")),
                "deepsoz_hemi": clean(row.get("deepsoz_hemi")),
                "deepsoz_comment": clean(row.get("deepsoz_comment")),
                "local_patient": clean(row.get("local_patient")),
                "local_edf": clean(row.get("local_edf")),
                "event_count": int(row.get("local_event_count", 0)),
                "llm_count": len(llm_items),
                "llm_success_count": success_count,
                "llm_failure_count": failure_count,
            })
        return result

    def record_payload(self, index: int):
        if index < 0 or index >= len(self.catalog):
            raise IndexError(index)
        row = self.catalog.iloc[index]
        csv_bi = Path(clean(row["local_csv_bi"]))
        events = local_event_onsets(csv_bi)
        source = str(Path(clean(row["local_edf"])).resolve())
        llm_items = []
        for item in self.llm_by_source.get(source, []):
            detail = item.get("llm_detail") or {}
            llm_items.append({
                "event_id": clean(item.get("event_id")),
                "record_status": clean(item.get("record_status")),
                "processing_error": clean(item.get("processing_error")),
                "t0_s": json_value(item.get("t0_s")),
                "soz_channels": split_tokens(item.get("soz_channels")),
                "soz_regions": split_tokens(item.get("soz_regions")),
                "confidence": json_value(
                    item.get("llm_confidence")
                    if item.get("llm_confidence") is not None
                    else detail.get("overall_confidence")
                ),
                "professional_report_zh": detail.get("professional_report_zh") or {},
                "key_local_interpretations": detail.get("key_local_interpretations") or [],
                "limitations_zh": detail.get("limitations_zh") or "",
                "evidence_summary_zh": detail.get("evidence_summary_zh") or "",
            })
        return {
            "index": index,
            "deepsoz": {
                "patient": clean(row.get("deepsoz_patient")),
                "record": clean(row.get("deepsoz_record")),
                "soz_electrodes": split_tokens(row.get("deepsoz_soz_electrodes")),
                "hemisphere": clean(row.get("deepsoz_hemi")),
                "region": clean(row.get("deepsoz_region")),
                "comment": clean(row.get("deepsoz_comment")),
            },
            "local": {
                "patient": clean(row.get("local_patient")),
                "edf": source,
                "sfreq_hz": json_value(row.get("edf_sfreq_hz")),
                "duration_s": json_value(row.get("edf_duration_s")),
                "n_channels": json_value(row.get("edf_n_channels")),
                "mapping_error_s": json_value(row.get("max_time_error_s")),
                "events": events,
            },
            "llm": llm_items,
        }

    def waveform_payload(self, index: int, event_index: int, pre_s: float, post_s: float):
        record = self.record_payload(index)
        events = record["local"]["events"]
        if event_index < 0 or event_index >= len(events):
            raise IndexError(event_index)
        event = events[event_index]
        edf = Path(record["local"]["edf"])
        raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
        llm_for_event = []
        event_id_suffix = f"__ev{event_index:04d}"
        for item in record["llm"]:
            if item["event_id"].endswith(event_id_suffix):
                llm_for_event.append(item)
        llm_leads = {
            lead for item in llm_for_event for lead in item.get("soz_channels", [])
        }
        center = float(event["start_s"])
        if llm_for_event and llm_for_event[0].get("t0_s") is not None:
            center = float(llm_for_event[0]["t0_s"])
        start = max(0.0, min(center, float(event["start_s"])) - pre_s)
        stop = min(float(raw.times[-1]), max(center, float(event["start_s"])) + post_s)
        data, names, times = bipolar_data(raw, start, stop, llm_leads)
        del raw
        step = max(1, int(math.ceil(len(times) / self.max_points)))
        times = times[::step]
        data = data[:, ::step]
        deep_electrodes = set(record["deepsoz"]["soz_electrodes"])
        local_leads = set(event["earliest_leads"])
        channels = []
        for channel_index, lead in enumerate(names):
            values = data[channel_index]
            scale = float(np.nanpercentile(np.abs(values), 95))
            if not math.isfinite(scale) or scale <= 0:
                scale = 1.0
            normalized = np.clip(values / scale, -4.0, 4.0)
            endpoints = set(lead.split("-"))
            channels.append({
                "lead": lead,
                "values": np.round(normalized, 4).tolist(),
                "scale_uv": scale * 1e6,
                "deepsoz_endpoint": bool(endpoints & deep_electrodes),
                "local_onset": lead in local_leads,
                "llm_soz": lead in llm_leads,
            })
        return {
            "record_index": index,
            "event_index": event_index,
            "event_start_s": event["start_s"],
            "event_stop_s": event["stop_s"],
            "llm_t0_s": llm_for_event[0].get("t0_s") if llm_for_event else None,
            "window_start_s": start,
            "window_stop_s": stop,
            "times": np.round(times, 4).tolist(),
            "channels": channels,
        }


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>DeepSOZ × LLM EEG Viewer</title>
<style>
:root{--bg:#eef2f6;--panel:#fff;--ink:#17202a;--muted:#607080;--blue:#1976d2;--red:#d32f2f;--green:#18864b;--purple:#7b2cbf;--orange:#ef6c00}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,"Noto Sans CJK SC",sans-serif;background:var(--bg);color:var(--ink)}
.app{display:grid;grid-template-columns:360px minmax(0,1fr);height:100vh}.side{background:#172b3a;color:white;display:flex;flex-direction:column;min-height:0}.brand{padding:18px}.brand h1{font-size:19px;margin:0 0 7px}.brand p{font-size:12px;margin:0;color:#b9c8d3}.search{padding:0 14px 12px}.search input{width:100%;padding:10px;border:0;border-radius:6px}.records{overflow:auto;flex:1}.record{padding:11px 14px;border-top:1px solid #ffffff18;cursor:pointer}.record:hover,.record.active{background:#25465d}.record b{font-size:13px}.record small{display:block;color:#b9c8d3;margin-top:3px}.main{overflow:auto;padding:20px}.empty{background:white;padding:30px;border-radius:10px}.top,.panel{background:var(--panel);border-radius:10px;padding:16px;margin-bottom:15px;box-shadow:0 2px 8px #0001}.top h2{margin:0 0 8px}.meta{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.meta div{background:#f6f8fa;padding:9px;border-radius:6px;word-break:break-all}.label{font-size:11px;color:var(--muted);display:block}.chips{display:flex;gap:6px;flex-wrap:wrap}.chip{border-radius:12px;padding:3px 8px;background:#e4edf5;font-size:12px}.chip.deep{background:#dcecff;color:#064f91}.chip.local{background:#dff5e7;color:#075f2c}.chip.llm{background:#fde1e1;color:#8c1515}.events button{margin:4px;padding:7px 11px;border:1px solid #b7c3cc;background:white;border-radius:5px;cursor:pointer}.events button.active{background:#263f53;color:white}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;margin:8px 0}.dot{width:12px;height:3px;display:inline-block;vertical-align:middle;margin-right:4px}.canvasWrap{overflow:auto;border:1px solid #ccd5dc;background:white}canvas{display:block}.llmMissing{background:#fff3cd;padding:12px;border-radius:6px;color:#604b00}.reportGrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.reportGrid div{padding:10px;background:#f6f8fa;border-radius:6px}.interpretation{border-left:3px solid var(--red);padding:8px 12px;margin:8px 0;background:#fff8f8}.status{padding:9px;color:var(--muted)}@media(max-width:900px){.app{grid-template-columns:1fr;height:auto}.side{height:45vh}.meta,.reportGrid{grid-template-columns:1fr}.main{padding:10px}}
</style></head><body><div class="app"><aside class="side"><div class="brand"><h1>DeepSOZ × LLM EEG Viewer</h1><p id="count">载入 607 条映射记录…</p></div><div class="search"><input id="search" placeholder="搜索患者、记录、电极、备注"></div><div id="records" class="records"></div></aside><main class="main"><div id="content" class="empty">从左侧选择一条记录。</div></main></div>
<script>
const S={catalog:[],filtered:[],selected:null,record:null,event:0};const $=x=>document.getElementById(x);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function chip(x,c=''){return `<span class="chip ${c}">${esc(x)}</span>`}function renderList(){const box=$('records');box.innerHTML=S.filtered.map(r=>`<div class="record ${r.index===S.selected?'active':''}" data-i="${r.index}"><b>${esc(r.deepsoz_record)}</b><small>${esc(r.local_patient)} · SOZ ${esc(r.deepsoz_soz||'—')} · ${r.event_count} events · LLM ${r.llm_success_count||0}${r.llm_failure_count?` / 失败 ${r.llm_failure_count}`:''}</small></div>`).join('');$('count').textContent=`显示 ${S.filtered.length} / ${S.catalog.length} 条唯一映射`;box.querySelectorAll('.record').forEach(x=>x.onclick=()=>selectRecord(+x.dataset.i));}
function filter(){const q=$('search').value.trim().toLowerCase();S.filtered=S.catalog.filter(r=>!q||JSON.stringify(r).toLowerCase().includes(q));renderList()}
async function selectRecord(i){S.selected=i;renderList();$('content').innerHTML='<div class="status">读取记录、标签和事件…</div>';const r=await (await fetch(`/api/record?index=${i}`)).json();S.record=r;S.event=0;renderRecord();if(r.local.events.length)loadWave(0)}
function renderRecord(){const r=S.record,d=r.deepsoz,l=r.local;const llm=r.llm;let llmHtml='<div class="llmMissing">该 EDF 没有大模型分析结果；这是缺失结果，不是阴性预测，也不会生成推测性解读。</div>';if(llm.length){llmHtml=llm.map(x=>{if(x.record_status==='processing_failed_closed'){return `<div class="llmMissing"><b>${esc(x.event_id)} · LLM 严格失败关闭</b><br>${esc(x.processing_error||'未记录具体失败原因')}</div>`}const p=x.professional_report_zh||{};return `<div><h3>${esc(x.event_id)} ${chip(x.record_status||'LLM已完成','llm')} ${chip((x.soz_channels||[]).join(', ')||'未给出SOZ','llm')}</h3><div class="reportGrid"><div><span class="label">电图描述</span>${esc(p.electrographic_description||'—')}</div><div><span class="label">电图印象</span>${esc(p.electrographic_impression||'—')}</div><div><span class="label">SOZ依据</span>${esc(p.soz_localization_rationale||'—')}</div><div><span class="label">伪迹鉴别/限制</span>${esc((p.artifact_differential||'')+' '+(p.clinical_limitations||''))}</div></div>${(x.key_local_interpretations||[]).map(k=>`<div class="interpretation"><b>${esc(k.finding_type||'波形发现')} · ${esc((k.channels||[]).join(', '))}</b><br>${esc(k.description_zh||'')}<br><small>${esc(k.quantitative_support_zh||'')}</small></div>`).join('')}</div>`}).join('')}
$('content').className='';$('content').innerHTML=`<section class="top"><h2>${esc(d.record)} → ${esc(l.patient)}</h2><div class="meta"><div><span class="label">本地EDF</span>${esc(l.edf)}</div><div><span class="label">映射最大时间误差</span>${Number(l.mapping_error_s).toFixed(4)} s</div><div><span class="label">EDF</span>${l.sfreq_hz} Hz · ${Number(l.duration_s).toFixed(1)} s · ${l.n_channels} ch</div><div><span class="label">DeepSOZ侧别/备注</span>${esc(d.hemisphere)} · ${esc(d.comment)}</div></div></section><section class="panel"><h3>DeepSOZ 标注</h3><div class="chips">${d.soz_electrodes.map(x=>chip(x,'deep')).join('')}</div><p>${esc(d.comment)}；该标签为 DeepSOZ manifest 的患者/记录级单极 SOZ。</p></section><section class="panel"><h3>本地 TUSZ 发作事件</h3><div class="events">${l.events.map((e,j)=>`<button id="ev${j}" onclick="loadWave(${j})">ev${String(j).padStart(4,'0')} · ${Number(e.start_s).toFixed(3)}–${Number(e.stop_s).toFixed(3)}s</button>`).join('')}</div><div id="localLabels"></div></section><section class="panel"><h3>EEG 波形</h3><div class="legend"><span><i class="dot" style="background:var(--blue)"></i>DeepSOZ电极端点</span><span><i class="dot" style="background:var(--green)"></i>本地首发导联</span><span><i class="dot" style="background:var(--red)"></i>大模型SOZ</span><span><i class="dot" style="background:var(--purple)"></i>标签重叠</span><span><i class="dot" style="background:var(--orange)"></i>大模型t0</span></div><div id="waveStatus" class="status">选择事件加载波形。</div><div class="canvasWrap"><canvas id="wave"></canvas></div></section><section class="panel"><h3>大模型波形解读</h3>${llmHtml}</section>`}
async function loadWave(j){S.event=j;document.querySelectorAll('.events button').forEach((b,k)=>b.classList.toggle('active',k===j));const e=S.record.local.events[j];$('localLabels').innerHTML=`<p><b>本地逐通道首发导联：</b> ${(e.earliest_leads||[]).map(x=>chip(x,'local')).join(' ')||'—'}；全局起点 ${Number(e.start_s).toFixed(4)}s</p>`;$('waveStatus').textContent='从 EDF 读取并派生双极导联…';const w=await (await fetch(`/api/waveform?index=${S.selected}&event=${j}`)).json();drawWave(w);$('waveStatus').textContent=`窗口 ${w.window_start_s.toFixed(3)}–${w.window_stop_s.toFixed(3)}s；${w.channels.length} 条导联；波形按各导联95百分位幅度归一化。`}
function drawWave(w){const c=$('wave'),wrap=c.parentElement,W=Math.max(1100,wrap.clientWidth-2),rowH=27,H=45+w.channels.length*rowH,ratio=window.devicePixelRatio||1;c.style.width=W+'px';c.style.height=H+'px';c.width=W*ratio;c.height=H*ratio;const x=c.getContext('2d');x.scale(ratio,ratio);x.fillStyle='#fff';x.fillRect(0,0,W,H);const left=150,right=20,top=25,pw=W-left-right,t=w.times;if(!t.length)return;const tx=v=>left+(v-t[0])/(t[t.length-1]-t[0])*pw;for(let k=0;k<=6;k++){const xx=left+k*pw/6;x.strokeStyle='#e3e8ec';x.beginPath();x.moveTo(xx,0);x.lineTo(xx,H);x.stroke();x.fillStyle='#607080';x.fillText((t[0]+k*(t[t.length-1]-t[0])/6).toFixed(1)+'s',xx-10,H-5)}function vline(v,color,dash=[]){if(v==null||v<t[0]||v>t[t.length-1])return;x.strokeStyle=color;x.setLineDash(dash);x.lineWidth=1.5;x.beginPath();x.moveTo(tx(v),0);x.lineTo(tx(v),H-18);x.stroke();x.setLineDash([])}vline(w.event_start_s,'#18864b',[5,4]);vline(w.llm_t0_s,'#ef6c00',[7,4]);w.channels.forEach((ch,i)=>{const y=top+i*rowH;let color='#333';const n=[ch.deepsoz_endpoint,ch.local_onset,ch.llm_soz].filter(Boolean).length;if(n>1)color='#7b2cbf';else if(ch.llm_soz)color='#d32f2f';else if(ch.local_onset)color='#18864b';else if(ch.deepsoz_endpoint)color='#1976d2';x.fillStyle=color;x.font='11px sans-serif';let tags=[];if(ch.deepsoz_endpoint)tags.push('D');if(ch.local_onset)tags.push('T');if(ch.llm_soz)tags.push('L');x.fillText(ch.lead+(tags.length?' ['+tags.join('')+']':''),5,y+4);x.strokeStyle=color;x.lineWidth=n?1.3:.65;x.beginPath();ch.values.forEach((v,k)=>{const xx=left+k/(ch.values.length-1)*pw,yy=y-v*4;if(k===0)x.moveTo(xx,yy);else x.lineTo(xx,yy)});x.stroke()})}
$('search').oninput=filter;fetch('/api/catalog').then(r=>r.json()).then(x=>{S.catalog=x;S.filtered=x;renderList()}).catch(e=>$('content').textContent='载入失败：'+e);
</script></body></html>'''


def make_handler(data: ViewerData):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, payload, status=200):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    body = HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                elif parsed.path == "/api/catalog":
                    self.send_json(data.catalog_payload())
                elif parsed.path == "/api/record":
                    self.send_json(data.record_payload(int(query["index"][0])))
                elif parsed.path == "/api/waveform":
                    self.send_json(data.waveform_payload(
                        int(query["index"][0]), int(query.get("event", [0])[0]),
                        float(query.get("pre", [10])[0]), float(query.get("post", [20])[0]),
                    ))
                else:
                    self.send_json({"error": "not found"}, 404)
            except Exception as exc:
                self.send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def log_message(self, fmt, *args):
            print(f"{self.client_address[0]} - {fmt % args}")

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--llm-success-index", type=Path, default=DEFAULT_LLM)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-points", type=int, default=1600)
    args = parser.parse_args()
    if not args.catalog.is_file():
        raise FileNotFoundError(f"catalog not found: {args.catalog}")
    data = ViewerData(args.catalog, args.llm_success_index, args.max_points)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(data))
    print(f"Loaded {len(data.catalog)} uniquely mapped records")
    print(f"Viewer: http://{args.host}:{args.port}/")
    print("Press Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
