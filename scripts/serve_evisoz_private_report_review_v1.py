#!/usr/bin/env python3
"""Serve a controlled, human-reviewable private EEG report workbench.

The service hashes and reads the raw DOCX files in memory, then returns only
the conservative de-identified clinical interval already defined by the
project policy.  Review submissions contain no raw text, patient names,
filenames, or source paths.  It is loopback-only by default; a LAN bind needs
an environment token (``EVISOZ_REVIEW_TOKEN``) unless explicitly overridden.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import socket
import tempfile
from threading import RLock
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.private_physician_reports import _docx_text_and_audit  # noqa: E402
from src.evisoz.forge.private_report_deidentification import (  # noqa: E402
    _docx_paragraphs,
    _safe_candidate_text,
    _source_patient_names,
)


DEFAULT_REPORT_ROOT = Path("/mnt/hd1/dyf/dataset/EEG_Reports/Reports")
# Private report inventory, exclusion and patient-name authority remain in the
# controlled parent artifact store.  The clean worktree deliberately does not
# contain these files, so defaults must not point at migration-era repository
# paths.  Raw DOCX bytes are still read only in memory by ``_build_dataset``.
DEFAULT_SOURCE_MANIFEST = Path(
    "/mnt/hd1/dyf/workspace/laptop/EEG_Seizure/outputs/soz_pre/private_edf_soz_manifest.csv"
)
DEFAULT_BUNDLE_ROOT = Path(
    "/mnt/hd1/dyf/workspace/laptop/EEG_Seizure/outputs/private_public_mapping_split_deid_v1_20260901_r4"
)
DEFAULT_OUTPUT = ROOT / "outputs/evisoz_private_report_manual_review_service_v1_20260901/reviews.json"
_REPORT_ID_RE = re.compile(r"^EVISOZ-PRPT-[0-9a-f]{24}$")
_ALLOWED_STATUS = {"pending", "pass", "reject"}
_ALLOWED_IDENTIFIER_REVIEW = {"not_started", "clear", "issues_found"}
_MAX_BODY = 64 * 1024


def _json(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"expected regular JSON file: {path}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_reviews(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": "evisoz_private_report_manual_review_receipts_v1", "entries": {}}
    value = _json(path)
    if value.get("schema_version") != "evisoz_private_report_manual_review_receipts_v1" or not isinstance(value.get("entries"), dict):
        raise ValueError("review receipt store has an unsupported schema")
    return value


def _build_dataset(*, report_root: Path, bundle_root: Path, source_manifest: Path, exclusion_path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    inventory = _json(bundle_root / "private_reports/inventory.json")
    candidates = _json(bundle_root / "private_reports/deidentified_candidates/manifest.json")
    excluded: set[str] = set()
    if exclusion_path is not None and exclusion_path.is_file() and not exclusion_path.is_symlink():
        exclusion = _json(exclusion_path)
        excluded = {str(row["report_id"]) for row in exclusion.get("entries", [])}
    # The source names/stems exist only for this in-memory extraction call.
    # Read the authority names only for this in-memory redaction call.
    source_names, _ = _source_patient_names(source_manifest.resolve(strict=True))
    source_paths: dict[str, Path] = {}
    for path in sorted(report_root.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.suffix.casefold() == ".docx" and not path.is_symlink():
            digest = _sha256(path)
            if digest in source_paths:
                raise ValueError("duplicate DOCX source bytes in controlled report root")
            source_paths[digest] = path
    candidate_by_report = {str(row["report_id"]): row for row in candidates.get("candidates", [])}
    reports: dict[str, dict[str, Any]] = {}
    for row in inventory.get("reports", []):
        report_id = str(row["report_id"])
        if not _REPORT_ID_RE.fullmatch(report_id) or report_id in excluded:
            continue
        if row.get("association", {}).get("status") != "linked_high_confidence":
            continue
        digest = str(row["document_ref"]["content_hash"]["sha256"])
        source = source_paths.get(digest)
        candidate = candidate_by_report.get(report_id)
        if source is None or candidate is None:
            continue
        raw = source.read_bytes()
        paragraphs = _docx_paragraphs(raw)
        # The conservative extractor removes demographics, signatures, known
        # names, dates, contacts and long IDs before anything is returned.
        text, extraction, scan = _safe_candidate_text(
            paragraphs,
            patient_names=source_names,
            source_stems=[item.stem for item in source_paths.values()],
        )
        reports[report_id] = {
            "report_id": report_id,
            "document_sha256": digest,
            "candidate_id": candidate["candidate_id"],
            "association_status": row["association"]["status"],
            "linkage_group_id": row["association"].get("linkage_group_id"),
            "split_assignment": deepcopy(row["association"].get("split_assignment")),
            "source_parse": _docx_text_and_audit(raw)[1],
            "extraction": extraction,
            "automated_phi_scan": scan,
            "deidentified_clinical_text": text,
        }
    facts = {
        "inventory_report_count": len(inventory.get("reports", [])),
        "excluded_report_count": len(excluded),
        "reviewable_report_count": len(reports),
        "raw_report_text_persisted": False,
        "raw_report_bytes_copied": False,
        "excluded_report_ids_not_served": sorted(excluded),
    }
    return reports, facts


HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EviSOZ 私有报告人工审核</title><style>
body{font:14px/1.5 system-ui,-apple-system,"Microsoft YaHei",sans-serif;margin:0;background:#f5f7fb;color:#182236}header{background:#172554;color:white;padding:20px 28px}main{max-width:1200px;margin:20px auto;padding:0 16px;display:grid;grid-template-columns:290px 1fr;gap:16px}.card{background:#fff;border:1px solid #dce3ee;border-radius:12px;padding:16px;box-shadow:0 4px 18px #22345412}.list{max-height:75vh;overflow:auto}.item{display:block;border:1px solid #dce3ee;border-radius:8px;padding:9px;margin:7px 0;cursor:pointer;background:#fff}.item.active{border-color:#3857d6;background:#eef2ff}.mono{font:12px ui-monospace,monospace;word-break:break-all}.muted{color:#64748b;font-size:12px}.text{white-space:pre-wrap;background:#f8fafc;border:1px solid #dce3ee;border-radius:8px;padding:13px;max-height:310px;overflow:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}label{display:flex;flex-direction:column;gap:4px;font-weight:600;margin:8px 0}input,select,textarea{font:inherit;padding:7px;border:1px solid #cbd5e1;border-radius:7px}button{border:0;border-radius:7px;padding:9px 14px;background:#3857d6;color:white;font-weight:700;cursor:pointer}button.secondary{background:#e8edf7;color:#263657}.checks{display:flex;gap:14px;flex-wrap:wrap}.checks label{display:flex;flex-direction:row;align-items:center;font-weight:500}.notice{border-left:4px solid #a15c00;background:#fff9ed;padding:10px;margin-bottom:12px}.hidden{display:none}@media(max-width:800px){main{grid-template-columns:1fr}}
</style></head><body><header><h1>EviSOZ 私有医生报告人工审核</h1><div class="muted" style="color:#c7d2fe">只显示从受控 DOCX 提取并自动去标识的临床段落；排除报告不会出现在列表中。</div></header><main><section class="card"><h2>可审核报告</h2><div id="facts" class="muted"></div><div id="list" class="list"></div></section><section class="card"><div id="empty">请选择左侧报告。</div><div id="detail" class="hidden"><div class="notice">人工审核必须由机构审查人完成。自动扫描通过不等于人工放行；审核结果只保存为本地草案回执。</div><h2 id="title"></h2><div id="meta" class="muted"></div><h3>去标识临床信息</h3><div id="clinical" class="text"></div><h3>自动检查与截取范围</h3><div id="audit" class="muted"></div><form id="form"><div class="grid"><label>人工审核状态<select id="status"><option value="pending">待审</option><option value="pass">通过</option><option value="reject">退回</option></select></label><label>间接标识符检查<select id="identifier"><option value="not_started">未开始</option><option value="clear">未发现</option><option value="issues_found">发现问题</option></select></label><label>审查人姓名<input id="reviewer" required></label><label>审查人角色<input id="role" required placeholder="privacy reviewer / data controller"></label></div><label class="checks"><span><input type="checkbox" id="scope"> 临床范围仅为 EEG 临床事实，未包含身份字段</span><span><input type="checkbox" id="dev"> 允许 development Qwen 文本训练</span><span><input type="checkbox" id="eval"> 允许 locked language evaluation</span></label><label>审核备注<textarea id="notes" rows="3" placeholder="记录发现的间接标识符、截取边界和 release lane 判断"></textarea><div style="margin-top:12px"><button>保存审核草案</button> <button type="button" class="secondary" id="reload">重新加载</button></div></form><div id="result" class="muted"></div></div></section></main><script>
let reports={},selected=null,reviews={};const $=id=>document.getElementById(id);const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(url,opts){let r=await fetch(url,opts);let d=await r.json();if(!r.ok)throw Error(d.error||r.status);return d}
function renderList(){let keys=Object.keys(reports).sort();$('list').innerHTML=keys.map(id=>{let r=reviews[id]||{};return `<div class="item ${selected===id?'active':''}" data-id="${esc(id)}"><div class="mono">${esc(id)}</div><div class="muted">${esc(r.status||'pending')} · ${esc(reports[id].split_assignment?.evisoz_role||'')}</div></div>`}).join('');document.querySelectorAll('.item').forEach(x=>x.onclick=()=>{selected=x.dataset.id;renderList();renderDetail()});$('facts').textContent=`可审核 ${keys.length} 条；排除/不服务 ${window.FACTS.excluded_report_count} 条`}
function renderDetail(){if(!selected){$('empty').classList.remove('hidden');$('detail').classList.add('hidden');return}let r=reports[selected],v=reviews[selected]||{};$('empty').classList.add('hidden');$('detail').classList.remove('hidden');$('title').textContent=selected;$('meta').textContent=`candidate ${r.candidate_id} · split ${r.split_assignment?.evisoz_role||''} · fold ${r.split_assignment?.outer_holdout_fold??'—'} · document SHA ${r.document_sha256}`;$('clinical').textContent=r.deidentified_clinical_text;$('audit').textContent=`route=${r.extraction.route}; paragraph ${r.extraction.selected_start_paragraph_index}–${r.extraction.selected_stop_paragraph_index_exclusive}; automated scan=${r.automated_phi_scan.automated_scan_status}; source parse=${r.source_parse.parse_status}`;$('status').value=v.status||'pending';$('identifier').value=v.indirect_identifier_review||'not_started';$('reviewer').value=v.reviewer_name||'';$('role').value=v.reviewer_role||'';$('scope').checked=!!v.clinical_scope_confirmed;$('dev').checked=!!v.development_qwen_training_release;$('eval').checked=!!v.locked_language_evaluation_release;$('notes').value=v.notes||'';$('result').textContent=''}
async function load(){let d=await api('/api/reports');reports=d.reports;reviews=(await api('/api/reviews')).entries;window.FACTS=d.facts;renderList();renderDetail()}
$('form').onsubmit=async e=>{e.preventDefault();if(!selected)return;let payload={report_id:selected,status:$('status').value,indirect_identifier_review:$('identifier').value,clinical_scope_confirmed:$('scope').checked,development_qwen_training_release:$('dev').checked,locked_language_evaluation_release:$('eval').checked,reviewer_name:$('reviewer').value.trim(),reviewer_role:$('role').value.trim(),notes:$('notes').value};try{let d=await api('/api/reviews',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});reviews=d.entries;renderList();renderDetail();$('result').textContent='已保存本地审核草案（未签发机构 release receipt）'}catch(err){$('result').textContent='保存失败：'+err.message}};$('reload').onclick=load;load().catch(e=>$('facts').textContent='服务读取失败：'+e.message);
</script></body></html>"""


class ReviewState:
    def __init__(self, reports: dict[str, dict[str, Any]], facts: dict[str, Any], output: Path, token: str | None) -> None:
        self.reports = reports
        self.facts = facts
        self.output = output
        self.token = token
        self.lock = RLock()
        self.reviews = _load_reviews(output)

    def public_reviews(self) -> dict[str, Any]:
        with self.lock:
            return deepcopy(self.reviews)

    def save_review(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        report_id = payload.get("report_id")
        if not isinstance(report_id, str) or report_id not in self.reports:
            raise ValueError("report_id is not reviewable (it may be explicitly excluded)")
        status = payload.get("status")
        identifier = payload.get("indirect_identifier_review")
        if status not in _ALLOWED_STATUS or identifier not in _ALLOWED_IDENTIFIER_REVIEW:
            raise ValueError("invalid review status or indirect-identifier state")
        reviewer = str(payload.get("reviewer_name", "")).strip()
        role = str(payload.get("reviewer_role", "")).strip()
        if status == "pass" and (not reviewer or not role or identifier != "clear" or payload.get("clinical_scope_confirmed") is not True):
            raise ValueError("a passed review requires reviewer, clear identifier check and clinical scope confirmation")
        split_role = self.reports[report_id]["split_assignment"]["evisoz_role"]
        dev = bool(payload.get("development_qwen_training_release"))
        evaluation = bool(payload.get("locked_language_evaluation_release"))
        if dev and split_role != "development_cv":
            raise ValueError("locked-test text cannot enter development training")
        if evaluation and split_role != "locked_test":
            raise ValueError("development text cannot enter locked evaluation")
        if status != "pass" and (dev or evaluation):
            raise ValueError("release lanes require a passed manual review")
        entry = {
            "report_id": report_id,
            "candidate_id": self.reports[report_id]["candidate_id"],
            "status": status,
            "indirect_identifier_review": identifier,
            "clinical_scope_confirmed": bool(payload.get("clinical_scope_confirmed")),
            "development_qwen_training_release": dev,
            "locked_language_evaluation_release": evaluation,
            "reviewer_name": reviewer,
            "reviewer_role": role,
            "notes": str(payload.get("notes", ""))[:4000],
            "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "institutional_release_receipt_issued": False,
        }
        with self.lock:
            self.reviews.setdefault("entries", {})[report_id] = entry
            _atomic_write_json(self.output, self.reviews)
            return deepcopy(self.reviews)


def _make_handler(state: ReviewState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EviSOZReview/1"

        def _authorized(self) -> bool:
            return state.token is None or self.headers.get("X-EviSOZ-Review-Token") == state.token

        def _json_response(self, payload: Mapping[str, object], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if not self._authorized():
                self._json_response({"error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
                return
            if self.path == "/" or self.path == "/index.html":
                body = HTML.encode("utf-8")
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body); return
            if self.path == "/api/health":
                self._json_response({"status": "ok", "reviewable_report_count": len(state.reports), "excluded_report_count": state.facts["excluded_report_count"]}); return
            if self.path == "/api/reports":
                self._json_response({"reports": state.reports, "facts": state.facts}); return
            if self.path == "/api/reviews":
                self._json_response(state.public_reviews()); return
            self._json_response({"error": "not_found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                self._json_response({"error": "unauthorized"}, HTTPStatus.UNAUTHORIZED); return
            if self.path != "/api/reviews":
                self._json_response({"error": "not_found"}, HTTPStatus.NOT_FOUND); return
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > _MAX_BODY:
                self._json_response({"error": "invalid_body_size"}, HTTPStatus.BAD_REQUEST); return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if type(payload) is not dict:
                    raise ValueError("JSON body must be an object")
                result = state.save_review(payload)
            except Exception as exc:  # noqa: BLE001
                self._json_response({"error": str(exc)}, HTTPStatus.BAD_REQUEST); return
            self._json_response(result)

        def log_message(self, format: str, *args: object) -> None:
            # Do not log URLs containing report IDs or review payloads.
            sys.stderr.write("EviSOZ review request\n")

    return Handler


def _lan_addresses(port: int) -> list[str]:
    result: set[str] = set()
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = item[4][0]
            if not address.startswith("127."):
                result.add(f"http://{address}:{port}/")
    except OSError:
        pass
    return sorted(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--exclusion", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--allow-unauthenticated-lan", action="store_true")
    args = parser.parse_args(argv)
    token = os.environ.get("EVISOZ_REVIEW_TOKEN") or None
    if not _loopback(args.host) and token is None and not args.allow_unauthenticated_lan:
        raise SystemExit("Refusing unauthenticated LAN bind; set EVISOZ_REVIEW_TOKEN or pass --allow-unauthenticated-lan")
    reports, facts = _build_dataset(report_root=args.report_root.resolve(strict=True), bundle_root=args.bundle_root.resolve(strict=True), source_manifest=args.source_manifest, exclusion_path=args.exclusion.resolve() if args.exclusion else args.bundle_root / "private_reports/exclusion_manifest.json")
    output = args.output.resolve()
    state = ReviewState(reports, facts, output, token)
    server = ThreadingHTTPServer((args.host, args.port), _make_handler(state))
    bound_host, bound_port = server.server_address[:2]
    print(f"EviSOZ review service ready: http://{bound_host}:{bound_port}/", flush=True)
    print(f"Reviewable reports: {len(reports)}; explicitly excluded and hidden: {facts['excluded_report_count']}", flush=True)
    print("Authentication: HTTP header X-EviSOZ-Review-Token " + ("enabled" if token else "disabled (loopback only)"), flush=True)
    if bound_host == "0.0.0.0":
        print("LAN URLs: " + (", ".join(_lan_addresses(bound_port)) or f"http://<内网IP>:{bound_port}/"), flush=True)
    print("Press Ctrl-C to stop.", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
