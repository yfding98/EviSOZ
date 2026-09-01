#!/usr/bin/env python3
"""Build the 607-record/1566-event TUSZ manifest and audit LLM coverage."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=ROOT / "outputs/deepsoz_llm_tusz_all_607_20260801/mapped_records.csv")
    parser.add_argument("--master-manifest", type=Path, default=ROOT / "outputs/soz_pre/tusz_region_vote_v1_manifest.csv")
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/deepsoz_607_llm_batch_20260801")
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog, encoding="utf-8-sig")
    master = pd.read_csv(args.master_manifest, encoding="utf-8-sig")
    edf_root_marker = "/edf/"
    relative = {
        str(path).split(edf_root_marker, 1)[1]
        for path in catalog.local_edf.astype(str)
        if edf_root_marker in str(path)
    }
    selected = master[master.edf_path.astype(str).isin(relative)].copy()
    selected = selected.sort_values(["split", "patient_id", "edf_path", "event_index"])
    if len(selected) != int(catalog.local_event_count.sum()):
        raise RuntimeError(
            f"event total mismatch: manifest={len(selected)} catalog={int(catalog.local_event_count.sum())}"
        )

    target_ids = set(selected.event_id.astype(str))
    feature_history: dict[str, list[dict]] = defaultdict(list)
    llm_history: dict[str, list[dict]] = defaultdict(list)
    for path in args.outputs_root.glob("*/records/*.json"):
        if path.name.endswith(".core_checkpoint.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        event_id = str(payload.get("event_id", ""))
        if event_id not in target_ids:
            continue
        item = {
            "path": str(path.resolve()),
            "run": path.parents[1].name,
            "record_status": payload.get("record_status"),
            "annotation_kind": payload.get("annotation_kind"),
            "processing_error": payload.get("processing_error"),
            "failed_stages": payload.get("failed_stages"),
        }
        if "llm_result" in payload or payload.get("record_status") is not None:
            llm_history[event_id].append(item)
        elif str(payload.get("annotation_kind", "")).startswith("autolabel_"):
            item["auto_soz_available"] = bool(payload.get("auto_soz_channels"))
            item["quality_flags"] = payload.get("quality_flags") or []
            feature_history[event_id].append(item)

    audit_rows = []
    for _, row in selected.iterrows():
        event_id = str(row.event_id)
        histories = llm_history.get(event_id, [])
        successes = [x for x in histories if x.get("record_status") == "llm_candidate_ready"]
        failures = [x for x in histories if x.get("record_status") == "processing_failed_closed"]
        packet_only = [x for x in histories if x.get("record_status") == "packet_only_llm_not_called"]
        if successes:
            status = "llm_success"
        elif failures:
            status = "llm_failed"
        elif packet_only:
            status = "packet_only_llm_not_called"
        else:
            status = "never_run_llm"
        feature = feature_history.get(event_id, [])
        audit_rows.append({
            "event_id": event_id,
            "patient_id": row.patient_id,
            "edf_path": row.edf_path,
            "event_index": row.event_index,
            "sz_start": row.sz_start,
            "sz_end": row.sz_end,
            "llm_coverage_status": status,
            "llm_success_count": len(successes),
            "llm_failure_count": len(failures),
            "llm_packet_only_count": len(packet_only),
            "llm_failure_reasons": " | ".join(
                str(x.get("processing_error") or x.get("failed_stages") or "unknown") for x in failures
            ),
            "feature_preanalysis_run": bool(feature),
            "feature_auto_soz_available": any(x.get("auto_soz_available") for x in feature),
            "feature_history_json": json.dumps(feature, ensure_ascii=False),
            "llm_history_json": json.dumps(histories, ensure_ascii=False),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "tusz_deepsoz_607_events_manifest.csv", index=False, encoding="utf-8-sig")
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.output_dir / "llm_coverage_before_run.csv", index=False, encoding="utf-8-sig")
    summary = {
        "mapped_recordings": len(catalog),
        "target_events": len(selected),
        "coverage_counts": audit.llm_coverage_status.value_counts().to_dict(),
        "feature_preanalysis_events": int(audit.feature_preanalysis_run.sum()),
        "feature_preanalysis_with_auto_soz": int(audit.feature_auto_soz_available.sum()),
        "feature_preanalysis_without_auto_soz": int((audit.feature_preanalysis_run & ~audit.feature_auto_soz_available).sum()),
        "important_semantics": "feature preanalysis is not an LLM interpretation",
        "filtered_manifest": str((args.output_dir / "tusz_deepsoz_607_events_manifest.csv").resolve()),
    }
    (args.output_dir / "coverage_summary_before_run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
