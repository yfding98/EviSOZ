#!/usr/bin/env python3
"""Freeze pseudonymous private signal and target ledgers before v18 inference.

This builder does not train or run a model.  It projects the historically
created private crosswalk into two physically separate files: a target-free
signal roster for feature materialization and a positive-only clinical
reference ledger opened only by the evaluation stage.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.soz_pre.build_private_edf_soz_manifest import (  # noqa: E402
    _load_flat_doctor_summary,
)
from code.soz_pre.utils import normalize_patient_name  # noqa: E402
from src.soz.geometry import STANDARD_19, normalize_electrode_name  # noqa: E402
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SCHEMA = "soz_private_zero_adaptation_bundle_v18"
DEFAULT_SOURCE = ROOT / "outputs/soz_pre/private_edf_soz_manifest.csv"
DEFAULT_EEG_ROOT = Path("/mnt/hd1/dyf/dataset/EEG")
DEFAULT_FLAT_SUMMARY = DEFAULT_EEG_ROOT / "发作起始通道汇总.csv"
DEFAULT_OUTPUT = ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814"
PROTOCOL = (
    ROOT
    / "research/02_method/"
    "labram_deepsoz_public_to_private_zero_adaptation_protocol_v18_20260814_zh.md"
)
SIGNAL_FIELDS = (
    "event_id",
    "patient_id",
    "source_row",
    "relative_edf_path",
    "global_event_t0_sec",
    "duration_sec",
    "time_source",
    "quality_flags",
    "time_support_preeligible",
    "primary_anchor_preeligible",
    "expanded_anchor_preeligible",
)
TARGET_FIELDS = (
    "event_id",
    "patient_id",
    "candidate_positive_electrodes",
    "standard19_positive_electrodes",
    "known_spread_electrodes",
    "outside_head_positive_electrodes",
    "diffuse_spread_present",
    "duplicate_conflict",
    "significant_spread_overlap",
    "has_c18_positive",
    "positive_set_exhaustiveness",
    "primary_reference_preeligible",
    "expanded_reference_preeligible",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(
    path: Path, fields: Iterable[str], rows: Iterable[Mapping[str, object]]
) -> None:
    names = tuple(fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in names})


def _json_list(values: Iterable[str]) -> str:
    return json.dumps(sorted(set(values)), ensure_ascii=True, separators=(",", ":"))


def _tokens(value: object) -> tuple[str, ...]:
    text = str(value or "").replace(",", ";")
    result = []
    for raw in text.split(";"):
        token = raw.strip()
        if not token:
            continue
        normalized = normalize_electrode_name(token)
        if normalized:
            result.append(normalized)
    return tuple(result)


def _event_sz_id(value: object) -> str:
    matches = re.findall(r"(?i)SZ\s*[-_ ]?(\d+)", str(value))
    if not matches:
        return ""
    return f"SZ{int(matches[-1])}"


def _conflicting_flat_keys(path: Path) -> set[tuple[str, str]]:
    groups: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for event in _load_flat_doctor_summary(path):
        groups[(event.patient_key, event.sz_id)].append(
            (event.onset_text, event.raw_significant, event.raw_spread)
        )
    return {
        key
        for key, values in groups.items()
        if len(values) > 1 and len(set(values)) > 1
    }


def build(
    source: Path,
    eeg_root: Path,
    flat_summary: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    if not source.is_file() or not flat_summary.is_file() or not eeg_root.is_dir():
        raise FileNotFoundError("private source, summary, or EEG root is missing")
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    rows = _read_csv(source)
    if not rows:
        raise ValueError("private source manifest is empty")
    patients = sorted({str(row["base_patient_id"]).strip() for row in rows})
    if "" in patients:
        raise ValueError("private source contains an empty patient identity")
    patient_ids = {name: f"PRIV-P{index + 1:03d}" for index, name in enumerate(patients)}
    conflicts = _conflicting_flat_keys(flat_summary)
    candidate_channels = {
        channel
        for channel, allowed in zip(STANDARD_19, V11_CANDIDATE_MASK.tolist())
        if allowed
    }
    standard_channels = set(STANDARD_19)
    signal_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    per_patient_event_count: Counter[str] = Counter()

    for ordinal, row in enumerate(rows, start=1):
        raw_patient = str(row["base_patient_id"]).strip()
        patient_id = patient_ids[raw_patient]
        per_patient_event_count[patient_id] += 1
        event_id = f"PRIV-E{ordinal:04d}"
        relative = Path(str(row["edf_path"]).replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".edf":
            raise ValueError(f"unsafe private EDF path at source row {ordinal}")
        flags = {value for value in str(row.get("quality_flags", "")).split(";") if value}
        try:
            t0 = float(row["sz_start"])
            duration = float(row["duration_sec"])
        except (TypeError, ValueError):
            t0 = duration = math.nan
        time_support = bool(
            math.isfinite(t0)
            and math.isfinite(duration)
            and t0 >= 42.0
            and duration >= t0 + 48.0
        )
        exact = str(row.get("time_source", "")) == "exact_sz_marker"
        no_multi = "multi_sz_markers_in_file" not in flags
        primary_anchor = time_support and exact and no_multi
        expanded_anchor = time_support

        significant_native = _tokens(row.get("doctor_significant_electrodes", ""))
        spread_native = _tokens(row.get("doctor_spread_electrodes", ""))
        standard_positive = set(significant_native) & standard_channels
        candidate_positive = standard_positive & candidate_channels
        spread = set(spread_native) & standard_channels
        outside = {
            value
            for value in significant_native
            if value not in standard_channels and value not in {"DIFFUSE", "NONE", "无"}
        }
        diffuse = "DIFFUSE" in spread_native
        # Any standard-19 significant/spread conflict invalidates the event,
        # including PZ outside the current C18 prediction head.
        overlap = bool(standard_positive & spread)
        key = (normalize_patient_name(raw_patient), _event_sz_id(row.get("event_id", "")))
        duplicate_conflict = key in conflicts
        reference_base = bool(candidate_positive) and not overlap and not duplicate_conflict
        primary_reference = reference_base and primary_anchor
        expanded_reference = reference_base and expanded_anchor

        signal_rows.append(
            {
                "event_id": event_id,
                "patient_id": patient_id,
                "source_row": ordinal,
                "relative_edf_path": relative.as_posix(),
                "global_event_t0_sec": "" if not math.isfinite(t0) else f"{t0:.9f}",
                "duration_sec": "" if not math.isfinite(duration) else f"{duration:.9f}",
                "time_source": str(row.get("time_source", "")),
                "quality_flags": ";".join(sorted(flags)),
                "time_support_preeligible": int(time_support),
                "primary_anchor_preeligible": int(primary_anchor),
                "expanded_anchor_preeligible": int(expanded_anchor),
            }
        )
        target_rows.append(
            {
                "event_id": event_id,
                "patient_id": patient_id,
                "candidate_positive_electrodes": _json_list(candidate_positive),
                "standard19_positive_electrodes": _json_list(standard_positive),
                "known_spread_electrodes": _json_list(spread),
                "outside_head_positive_electrodes": _json_list(outside),
                "diffuse_spread_present": int(diffuse),
                "duplicate_conflict": int(duplicate_conflict),
                "significant_spread_overlap": int(overlap),
                "has_c18_positive": int(bool(candidate_positive)),
                "positive_set_exhaustiveness": "positive_only_unknown_complement",
                "primary_reference_preeligible": int(primary_reference),
                "expanded_reference_preeligible": int(expanded_reference),
            }
        )
        counts["time_support_preeligible"] += int(time_support)
        counts["primary_anchor_preeligible"] += int(primary_anchor)
        counts["has_c18_positive"] += int(bool(candidate_positive))
        counts["duplicate_conflict"] += int(duplicate_conflict)
        counts["significant_spread_overlap"] += int(overlap)
        counts["primary_reference_preeligible"] += int(primary_reference)
        counts["expanded_reference_preeligible"] += int(expanded_reference)

    if len({row["event_id"] for row in signal_rows}) != len(rows):
        raise RuntimeError("private event pseudonyms are not unique")
    if {row["event_id"] for row in signal_rows} != {
        row["event_id"] for row in target_rows
    }:
        raise RuntimeError("private signal and target ledgers do not align")

    output.mkdir(parents=True)
    _write_csv(output / "signal_roster.csv", SIGNAL_FIELDS, signal_rows)
    _write_csv(output / "target_ledger.csv", TARGET_FIELDS, target_rows)
    manifest: dict[str, object] = {
        "schema_version": SCHEMA,
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "source_manifest": str(source),
        "eeg_root": str(eeg_root),
        "patient_count": len(patients),
        "event_count": len(rows),
        "conflicting_flat_summary_key_count": len(conflicts),
        "counts": dict(sorted(counts.items())),
        "patient_event_count_distribution": {
            "minimum": min(per_patient_event_count.values()),
            "maximum": max(per_patient_event_count.values()),
        },
        "files": {
            "target_free_signal_roster": "signal_roster.csv",
            "positive_only_target_ledger": "target_ledger.csv",
        },
        "frozen_policy": {
            "candidate_space": [
                channel for channel in STANDARD_19 if channel in candidate_channels
            ],
            "primary_anchor": "exact_sz_marker_and_no_multi_sz_markers",
            "expanded_anchor": "any_finite_complete_time_support",
            "spread_is_soz_positive": False,
            "unlisted_channel_is_negative": False,
            "duplicate_conflict_action": "exclude",
            "significant_spread_overlap_action": "exclude_event",
            "outside_head_mapping": "none",
        },
        "access_receipt": {
            "private_target_values_loaded_for_ledger_projection": True,
            "private_eeg_loaded": False,
            "model_predictions_loaded": False,
            "training_performed": False,
            "llm_annotation_performed": False,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--flat-summary", type=Path, default=DEFAULT_FLAT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.source, args.eeg_root, args.flat_summary, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
         
