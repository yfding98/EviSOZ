#!/usr/bin/env python3
"""Map the DeepSOZ TUH manifest to local TUSZ and export event rows.

DeepSOZ was built from an older TUH/TUSZ release whose numeric patient and
recording names differ from the anonymized names in local TUSZ v2.0.3.  This
script matches a recording only when its complete seizure start/end sequence
has exactly one local candidate within ``--time-tolerance-sec``.

DeepSOZ supplies one set of monopolar SOZ electrodes per recording.  For the
private-manifest-compatible output, that set is repeated on every seizure
event in the matched EDF and converted to the canonical 32 bipolar leads.  A
bipolar lead is positive when either endpoint is a DeepSOZ-positive monopolar
electrode.  These are record-level labels repeated per event, not event-level
clinical ground truth.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import mne
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.deepsoz.build_tusz_v203_manifest import (  # noqa: E402
    infer_onset_details,
    parse_annotation_csv,
)
from code.tfm_soz.constants import (  # noqa: E402
    FULL_TCP_CHANNELS,
    FULL_TCP_COLUMNS,
    FULL_TCP_PAIRS,
    region_endpoint_vote_ranking,
)


DEFAULT_DEEPSOZ_MANIFEST = (
    ROOT
    / "outputs/deepsoz_llm_tusz_all_607_20260801/source/TUH_manifest_final.csv"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT_DIR = ROOT / "outputs/deepsoz_tusz_adapted_manifest_20260803"

# Preserve a stable, clinically familiar order.  The duplicate source columns
# pz and pz.1 are intentionally merged into one PZ electrode.
MONOPOLAR_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("FP1", ("fp1",)),
    ("F7", ("f7",)),
    ("T3", ("t3",)),
    ("T5", ("t5",)),
    ("O1", ("o1",)),
    ("F3", ("f3",)),
    ("C3", ("c3",)),
    ("P3", ("p3",)),
    ("FZ", ("fz",)),
    ("CZ", ("cz",)),
    ("PZ", ("pz", "pz.1")),
    ("FP2", ("fp2",)),
    ("F8", ("f8",)),
    ("T4", ("t4",)),
    ("T6", ("t6",)),
    ("O2", ("o2",)),
    ("F4", ("f4",)),
    ("C4", ("c4",)),
    ("P4", ("p4",)),
    ("OZ", ("oz",)),
    ("A1", ("a1",)),
    ("A2", ("a2",)),
)

BASE_COLUMNS: tuple[str, ...] = (
    "source",
    "patient_id",
    "edf_path",
    "split",
    "duration",
    "sz_start",
    "sz_end",
    "sz_duration",
    "n_seizure_events",
    "seizure_type",
    "hemisphere",
    "onset_channels",
    "soz_bipolar",
)
OUTPUT_COLUMNS: tuple[str, ...] = BASE_COLUMNS + FULL_TCP_COLUMNS + ("soz_region",)


def parse_number_list(value: Any) -> list[float]:
    """Parse DeepSOZ's string representation of a numeric list."""

    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    try:
        values = [float(item) for item in parsed]
    except (TypeError, ValueError):
        return []
    return values if all(math.isfinite(item) for item in values) else []


def annotation_rows(path: Path) -> list[dict[str, str]]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return list(csv.DictReader(lines))


def seizure_intervals(path: Path) -> list[tuple[float, float]]:
    """Read global TERM seizure intervals from a TUSZ ``.csv_bi`` file."""

    intervals: list[tuple[float, float]] = []
    for row in annotation_rows(path):
        if row.get("channel", "").strip().upper() != "TERM":
            continue
        if row.get("label", "").strip().lower() != "seiz":
            continue
        start = float(row["start_time"])
        stop = float(row["stop_time"])
        if math.isfinite(start) and math.isfinite(stop) and stop > start:
            intervals.append((start, stop))
    return intervals


def max_interval_error(
    deep_starts: Sequence[float],
    deep_ends: Sequence[float],
    local_intervals: Sequence[tuple[float, float]],
) -> float:
    if (
        not local_intervals
        or len(deep_starts) != len(deep_ends)
        or len(deep_starts) != len(local_intervals)
    ):
        return math.inf
    errors = [
        abs(float(deep_start) - float(local_start))
        for deep_start, (local_start, _) in zip(deep_starts, local_intervals)
    ]
    errors.extend(
        abs(float(deep_end) - float(local_end))
        for deep_end, (_, local_end) in zip(deep_ends, local_intervals)
    )
    return max(errors)


def scan_local_tusz(
    tusz_root: Path,
) -> dict[int, list[tuple[Path, list[tuple[float, float]]]]]:
    """Index local seizure recordings by number of TERM events."""

    by_count: dict[int, list[tuple[Path, list[tuple[float, float]]]]] = defaultdict(list)
    for path in sorted(tusz_root.rglob("*.csv_bi")):
        try:
            intervals = seizure_intervals(path)
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if intervals:
            by_count[len(intervals)].append((path, intervals))
    return dict(by_count)


def _local_patient(path: Path, tusz_root: Path) -> str:
    relative = path.relative_to(tusz_root)
    return relative.parts[1] if len(relative.parts) > 1 else ""


def map_recordings(
    deepsoz: pd.DataFrame,
    local_by_count: Mapping[int, Sequence[tuple[Path, list[tuple[float, float]]]]],
    tusz_root: Path,
    tolerance_sec: float,
) -> pd.DataFrame:
    """Map DeepSOZ rows using unique complete seizure-timeline agreement."""

    rows: list[dict[str, Any]] = []
    for deep_index, row in deepsoz.iterrows():
        starts = parse_number_list(row.get("sz_starts"))
        ends = parse_number_list(row.get("sz_ends"))
        candidates: list[tuple[Path, float]] = []
        if starts and len(starts) == len(ends):
            for path, intervals in local_by_count.get(len(starts), []):
                error = max_interval_error(starts, ends, intervals)
                if error <= tolerance_sec:
                    candidates.append((path, error))

        status = "unmapped"
        selected: Path | None = None
        selected_error: float | None = None
        if len(candidates) == 1:
            selected, selected_error = candidates[0]
            status = "unique"
        elif len(candidates) > 1:
            status = "ambiguous"

        candidate_paths = [str(path) for path, _ in candidates]
        candidate_errors = [float(error) for _, error in candidates]
        rows.append(
            {
                "deepsoz_row": int(deep_index),
                "deepsoz_patient": str(row.get("pt_id", "")).strip(),
                "deepsoz_record": str(row.get("fn", "")).strip(),
                "local_patient": _local_patient(selected, tusz_root) if selected else "",
                "local_csv_bi": str(selected) if selected else "",
                "local_edf": str(selected.with_suffix(".edf")) if selected else "",
                "max_time_error_s": selected_error,
                "candidate_count": len(candidates),
                "mapping_status": status,
                "candidate_local_csv_bi": ";".join(candidate_paths),
                "candidate_max_errors_s": ";".join(f"{value:.9g}" for value in candidate_errors),
            }
        )
    return pd.DataFrame(rows)


def _is_positive(value: Any) -> bool:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return bool(pd.notna(numeric) and float(numeric) > 0.0)


def deepsoz_monopolar_soz(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract positive DeepSOZ electrodes, merging ``pz`` and ``pz.1``."""

    result: list[str] = []
    for electrode, source_columns in MONOPOLAR_SOURCES:
        if any(_is_positive(row.get(column)) for column in source_columns):
            result.append(electrode)
    return tuple(result)


def monopolar_to_bipolar(
    electrodes: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Convert a monopolar set using endpoint incidence in canonical order.

    Returns ``(positive_bipolar_leads, unmapped_monopolar_electrodes)``.
    """

    normalized = {str(item).strip().upper() for item in electrodes if str(item).strip()}
    endpoint_set = {endpoint for pair in FULL_TCP_PAIRS for endpoint in pair}
    bipolar = tuple(
        channel
        for channel, pair in zip(FULL_TCP_CHANNELS, FULL_TCP_PAIRS)
        if normalized.intersection(pair)
    )
    unmapped = tuple(sorted(normalized - endpoint_set))
    return bipolar, unmapped


def _edf_duration_seconds(edf_path: Path) -> float:
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    return float((raw.n_times - 1) / float(raw.info["sfreq"]))


def _split_and_relative_path(edf_path: Path, tusz_root: Path) -> tuple[str, str]:
    relative = edf_path.relative_to(tusz_root)
    split = relative.parts[0] if relative.parts else ""
    return split, relative.as_posix()


def build_event_manifest(
    deepsoz: pd.DataFrame,
    mapping: pd.DataFrame,
    tusz_root: Path,
    onset_tolerance_sec: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build private-format event rows and record-level conversion audit."""

    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    events_without_local_type = 0
    event_type_counts: Counter[str] = Counter()
    bipolar_count_distribution: Counter[int] = Counter()
    unmapped_electrode_counts: Counter[str] = Counter()
    monopolar_electrode_counts: Counter[str] = Counter()
    unique_map = mapping.loc[mapping["mapping_status"].eq("unique")].copy()

    for map_row in unique_map.sort_values("deepsoz_row").to_dict("records"):
        deep_index = int(map_row["deepsoz_row"])
        deep_row = deepsoz.loc[deep_index]
        edf_path = Path(str(map_row["local_edf"]))
        csv_bi_path = Path(str(map_row["local_csv_bi"]))
        csv_path = edf_path.with_suffix(".csv")
        if not edf_path.is_file():
            raise FileNotFoundError(f"Mapped EDF is missing: {edf_path}")
        if not csv_path.is_file() or not csv_bi_path.is_file():
            raise FileNotFoundError(f"Mapped annotation is missing for: {edf_path}")

        events = seizure_intervals(csv_bi_path)
        channel_rows = parse_annotation_csv(csv_path)
        duration = _edf_duration_seconds(edf_path)
        split, relative_edf = _split_and_relative_path(edf_path, tusz_root)
        monopolar = deepsoz_monopolar_soz(deep_row)
        bipolar, unmapped_monopolar = monopolar_to_bipolar(monopolar)
        active_bipolar = set(bipolar)
        hemisphere = str(deep_row.get("hemi", "")).strip()
        soz_bipolar_text = ",".join(bipolar)
        soz_region = (
            region_endpoint_vote_ranking(soz_bipolar_text)[0]
            if bipolar
            else ""
        )

        monopolar_electrode_counts.update(monopolar)
        unmapped_electrode_counts.update(unmapped_monopolar)
        bipolar_count_distribution[len(bipolar)] += 1
        audit_rows.append(
            {
                "deepsoz_row": deep_index,
                "deepsoz_patient": str(deep_row.get("pt_id", "")).strip(),
                "deepsoz_record": str(deep_row.get("fn", "")).strip(),
                "local_patient": str(map_row["local_patient"]),
                "local_edf": relative_edf,
                "monopolar_soz": ";".join(monopolar),
                "mapped_bipolar_soz": soz_bipolar_text,
                "n_monopolar_soz": len(monopolar),
                "n_mapped_bipolar_soz": len(bipolar),
                "unmapped_monopolar_electrodes": ";".join(unmapped_monopolar),
                "mapping_rule": "bipolar_positive_if_either_endpoint_is_deepsoz_positive",
                "label_scope": "record_level_repeated_per_event",
            }
        )

        label_values = {
            column: int(channel in active_bipolar)
            for channel, column in zip(FULL_TCP_CHANNELS, FULL_TCP_COLUMNS)
        }
        for event_start, event_stop in events:
            onset = infer_onset_details(
                channel_rows,
                event_start=event_start,
                event_stop=event_stop,
                tolerance_sec=onset_tolerance_sec,
            )
            seizure_type = str(onset["seizure_type"]).strip()
            if seizure_type:
                event_type_counts[seizure_type] += 1
            else:
                events_without_local_type += 1
            output_rows.append(
                {
                    "source": "tusz",
                    "patient_id": str(map_row["local_patient"]),
                    "edf_path": relative_edf,
                    "split": split,
                    "duration": round(duration, 9),
                    "sz_start": float(event_start),
                    "sz_end": float(event_stop),
                    "sz_duration": round(float(event_stop - event_start), 9),
                    "n_seizure_events": len(events),
                    "seizure_type": seizure_type,
                    "hemisphere": hemisphere,
                    "onset_channels": ";".join(monopolar),
                    "soz_bipolar": soz_bipolar_text,
                    **label_values,
                    "soz_region": soz_region,
                }
            )

    manifest = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    audit = pd.DataFrame(audit_rows)
    build_stats = {
        "mapped_records": int(len(unique_map)),
        "unique_mapped_edf": int(unique_map["local_edf"].nunique()),
        "output_event_rows": int(len(manifest)),
        "events_without_local_seizure_type": int(events_without_local_type),
        "local_seizure_type_counts": dict(sorted(event_type_counts.items())),
        "monopolar_electrode_positive_record_counts": dict(
            sorted(monopolar_electrode_counts.items())
        ),
        "mapped_bipolar_count_per_record_distribution": {
            str(key): value for key, value in sorted(bipolar_count_distribution.items())
        },
        "unmapped_monopolar_electrode_record_counts": dict(
            sorted(unmapped_electrode_counts.items())
        ),
    }
    return manifest, audit, build_stats


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(
    output_dir: Path,
    deepsoz_path: Path,
    mapping: pd.DataFrame,
    manifest: pd.DataFrame,
    audit: pd.DataFrame,
    summary: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_copy = source_dir / deepsoz_path.name
    if deepsoz_path.resolve() != source_copy.resolve():
        shutil.copy2(deepsoz_path, source_copy)
    mapping.to_csv(output_dir / "mapping.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(
        output_dir / "deepsoz_tusz_annotation_top1.csv",
        index=False,
        encoding="utf-8-sig",
        columns=OUTPUT_COLUMNS,
    )
    audit.to_csv(
        output_dir / "electrode_to_bipolar_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map DeepSOZ to local TUSZ and export private-format bipolar SOZ labels"
    )
    parser.add_argument("--deepsoz-manifest", type=Path, default=DEFAULT_DEEPSOZ_MANIFEST)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--time-tolerance-sec", type=float, default=0.25)
    parser.add_argument(
        "--onset-tolerance-sec",
        type=float,
        default=1.0,
        help="Only used to infer the local event seizure_type; it does not alter DeepSOZ SOZ labels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deepsoz_path = args.deepsoz_manifest.expanduser().resolve()
    tusz_root = args.tusz_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not deepsoz_path.is_file():
        raise FileNotFoundError(f"DeepSOZ manifest not found: {deepsoz_path}")
    if not tusz_root.is_dir():
        raise NotADirectoryError(f"TUSZ EDF root not found: {tusz_root}")
    if not math.isfinite(args.time_tolerance_sec) or args.time_tolerance_sec < 0:
        raise ValueError("--time-tolerance-sec must be finite and non-negative")
    if not math.isfinite(args.onset_tolerance_sec) or args.onset_tolerance_sec < 0:
        raise ValueError("--onset-tolerance-sec must be finite and non-negative")

    deepsoz = pd.read_csv(deepsoz_path)
    local_by_count = scan_local_tusz(tusz_root)
    mapping = map_recordings(
        deepsoz,
        local_by_count,
        tusz_root=tusz_root,
        tolerance_sec=float(args.time_tolerance_sec),
    )
    manifest, audit, build_stats = build_event_manifest(
        deepsoz,
        mapping,
        tusz_root=tusz_root,
        onset_tolerance_sec=float(args.onset_tolerance_sec),
    )
    mapping_status_counts = {
        str(key): int(value)
        for key, value in mapping["mapping_status"].value_counts().items()
    }
    summary = {
        "schema_version": "deepsoz_to_local_tusz_v203_private46_endpoint_incidence_v1",
        "deepsoz_manifest": str(deepsoz_path),
        "deepsoz_manifest_sha256": _sha256(deepsoz_path),
        "deepsoz_rows": int(len(deepsoz)),
        "tusz_root": str(tusz_root),
        "time_mapping_rule": "unique complete TERM seizure start/end sequence within tolerance",
        "time_tolerance_sec": float(args.time_tolerance_sec),
        "mapping_status_counts": mapping_status_counts,
        "label_source": "DeepSOZ TUH_manifest_final.csv monopolar electrode columns",
        "label_scope": "record_level_repeated_per_event",
        "label_scope_warning": (
            "DeepSOZ SOZ electrodes are recording-level labels repeated on each local TERM "
            "seizure event; they are not event-level clinical ground truth."
        ),
        "monopolar_to_bipolar_rule": (
            "a canonical bipolar lead is positive when either endpoint is a positive "
            "DeepSOZ monopolar SOZ electrode"
        ),
        "pz_source_columns_merged": ["pz", "pz.1"],
        "output_columns": list(OUTPUT_COLUMNS),
        "output_column_count": len(OUTPUT_COLUMNS),
        **build_stats,
    }
    write_outputs(output_dir, deepsoz_path, mapping, manifest, audit, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Mapping: {output_dir / 'mapping.csv'}")
    print(f"Manifest: {output_dir / 'deepsoz_tusz_annotation_top1.csv'}")
    print(f"Audit: {output_dir / 'electrode_to_bipolar_audit.csv'}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
