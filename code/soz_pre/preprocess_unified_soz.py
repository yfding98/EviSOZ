#!/usr/bin/env python3
"""Preprocess canonical SOZ manifest rows into model-ready NPZ sequences."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import mne
import numpy as np

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from soz_pre.constants import (  # noqa: E402
    CHANNEL_LABEL_MASK_COLUMNS,
    CHANNEL_PROP_COLUMNS,
    CHANNEL_TO_REGIONS,
    EXTRA_INPUT_ELECTRODES,
    HEMISPHERE_CLASSES,
    HEMISPHERE_INDEX,
    REGION_LABEL_COLUMNS,
    REGION_MASK_COLUMNS,
    REGION_PROP_COLUMNS,
    REGION_NAMES,
    TCP_CHANNELS,
    TCP_COLUMNS,
    TCP_PAIRS,
)
from soz_pre.utils import clean_cell, normalize_electrode_name, parse_float, read_csv_rows, write_csv_rows  # noqa: E402


DEFAULT_MANIFEST = "outputs/soz_pre/unified_region_soz_manifest.csv"
DEFAULT_OUTPUT = "outputs/soz_pre/preprocessed"
DEFAULT_TUSZ_ROOT = "/mnt/hd1/dyf/dataset/TUSZ"
DEFAULT_PRIVATE_ROOT = "/mnt/hd1/dyf/dataset/EEG"
DEFAULT_TUSZ_PROCESSED_SUBDIR = "processed_fif"
DEFAULT_TUSZ_PROCESSED_SUFFIX = "_fnsz_1-45Hz_200Hz_ica_raw.fif"
PROCESSED_TUSZ_PATH_FIELDS = (
    "processed_path",
    "processed_fif_path",
    "preprocessed_path",
    "clean_lite_path",
    "raw_filtered_path",
)


def safe_name(value: object) -> str:
    text = clean_cell(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def source_key(value: object) -> str:
    text = clean_cell(value).lower()
    if text in {"public", "tuh", "tusz_v203"}:
        return "tusz"
    if text in {"private", "eeg"}:
        return "private"
    return text or "unknown"


def parse_filter_values(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = list(value)
    return {clean_cell(item).lower() for item in items if clean_cell(item)}


def filter_rows_by_seizure_type(rows: Sequence[Dict[str, str]], seizure_types: object) -> List[Dict[str, str]]:
    wanted = parse_filter_values(seizure_types)
    if not wanted:
        return list(rows)
    return [row for row in rows if clean_cell(row.get("seizure_type")).lower() in wanted]


def path_match_keys(value: object) -> set[str]:
    """Build stable path keys for matching manifests to QC failure logs."""

    text = clean_cell(value).replace("\\", "/")
    if not text:
        return set()
    keys = {text.lower()}
    path = Path(text)
    try:
        if path.is_absolute():
            keys.add(str(path.expanduser().resolve(strict=False)).replace("\\", "/").lower())
    except Exception:
        pass
    parts = tuple(part for part in path.parts if part not in ("/", ""))
    for anchor in ("TUSZ", "EEG", "edf"):
        for idx, part in enumerate(parts):
            if part.lower() != anchor.lower():
                continue
            suffix = "/".join(parts[idx:])
            if "/" in suffix:
                keys.add(suffix.lower())
            tail = "/".join(parts[idx + 1 :])
            if "/" in tail:
                keys.add(tail.lower())
    return keys


def row_path_match_keys(row: Mapping[str, object], resolved_path: Path, candidates: Sequence[str]) -> set[str]:
    keys = set()
    keys.update(path_match_keys(resolved_path))
    keys.update(path_match_keys(row.get("edf_path")))
    for candidate in candidates:
        keys.update(path_match_keys(candidate))
    return keys


def resolve_failed_qc_log(args, qc_root: Optional[Path]) -> Optional[Path]:
    if bool(getattr(args, "ignore_failed_qc_log", False)):
        return None
    explicit = clean_cell(getattr(args, "failed_qc_log", ""))
    if explicit:
        return Path(explicit)
    if qc_root is None:
        return None
    default = qc_root / "qc" / "error_log.csv"
    return default if default.is_file() else None


def load_failed_qc_path_keys(path: Path) -> Tuple[set[str], List[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    keys: set[str] = set()
    paths: List[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = clean_cell(row.get("status")).lower()
            if status and status != "failed":
                continue
            edf_path = clean_cell(row.get("edf_path"))
            if not edf_path:
                continue
            paths.append(edf_path)
            keys.update(path_match_keys(edf_path))
    return keys, paths


def _candidate_roots(root: Path, source: str) -> List[Path]:
    roots = [root]
    if source == "tusz":
        roots.extend([
            root / "v2.0.3" / "edf",
            root / "edf",
            root / "TUSZ" / "v2.0.3" / "edf",
        ])
    return list(dict.fromkeys(roots))


def _is_fif_text(value: object) -> bool:
    return clean_cell(value).lower().replace("\\", "/").endswith((".fif", ".fif.gz"))


def _row_path_values(row: Mapping[str, object], source: str, prefer_processed_tusz: bool) -> List[Tuple[str, str]]:
    fields: List[str] = []
    if source == "tusz" and prefer_processed_tusz:
        fields.extend(PROCESSED_TUSZ_PATH_FIELDS)
    fields.append("edf_path")
    if source == "tusz":
        fields.append("original_edf_path")
        if not prefer_processed_tusz:
            fields.extend(PROCESSED_TUSZ_PATH_FIELDS)

    out: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for field in fields:
        text = clean_cell(row.get(field)).replace("\\", "/")
        if not text or text.lower() in seen:
            continue
        out.append((field, text))
        seen.add(text.lower())
    return out


def _relative_tusz_edf_path(raw: Path) -> Path:
    if not raw.is_absolute():
        return raw
    parts = tuple(part for part in raw.parts if part not in ("/", ""))
    for idx, part in enumerate(parts):
        if part.lower() == "edf" and idx + 1 < len(parts):
            return Path(*parts[idx + 1 :])
    return Path(raw.name)


def _processed_tusz_relpath_from_edf(raw_text: str, processed_subdir: str, processed_suffix: str) -> Optional[Path]:
    raw = Path(raw_text.replace("\\", "/"))
    if raw.name.lower().endswith((".fif", ".fif.gz")):
        return None
    if raw.suffix.lower() != ".edf":
        return None
    rel = _relative_tusz_edf_path(raw)
    return Path(processed_subdir) / rel.with_name(f"{rel.stem}{processed_suffix}")


def _roots_for_candidate(
    *,
    source: str,
    field: str,
    raw_text: str,
    tusz_root: Path,
    private_root: Path,
    tusz_processed_root: Optional[Path],
) -> List[Path]:
    if source != "tusz":
        return _candidate_roots(private_root, source)
    roots: List[Path] = []
    processed_like = field in PROCESSED_TUSZ_PATH_FIELDS or _is_fif_text(raw_text)
    if tusz_processed_root is not None and processed_like:
        roots.extend(_candidate_roots(tusz_processed_root, source))
    roots.extend(_candidate_roots(tusz_root, source))
    if tusz_processed_root is not None and not processed_like:
        roots.extend(_candidate_roots(tusz_processed_root, source))
    return list(dict.fromkeys(roots))


def candidate_edf_paths(
    row: Mapping[str, object],
    tusz_root: Path,
    private_root: Path,
    tusz_processed_root: Optional[Path] = None,
    prefer_processed_tusz: bool = True,
    tusz_processed_subdir: str = DEFAULT_TUSZ_PROCESSED_SUBDIR,
    tusz_processed_suffix: str = DEFAULT_TUSZ_PROCESSED_SUFFIX,
) -> List[Path]:
    source = source_key(row.get("source"))
    candidates: List[Path] = []

    def add(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)
        if source != "tusz" and path.suffix.lower() == ".set":
            edf_path = path.with_suffix(".edf")
            if edf_path not in candidates:
                candidates.append(edf_path)

    raw_values = _row_path_values(row, source, prefer_processed_tusz=prefer_processed_tusz)
    if source == "tusz" and tusz_processed_root is not None and prefer_processed_tusz:
        for _, raw_text in raw_values:
            processed_rel = _processed_tusz_relpath_from_edf(
                raw_text,
                processed_subdir=tusz_processed_subdir,
                processed_suffix=tusz_processed_suffix,
            )
            if processed_rel is None:
                continue
            if processed_rel.is_absolute():
                add(processed_rel)
            else:
                add(tusz_processed_root / processed_rel)

    for field, raw_text in raw_values:
        raw = Path(raw_text)
        if raw.is_absolute():
            add(raw)
            continue
        for base in _roots_for_candidate(
            source=source,
            field=field,
            raw_text=raw_text,
            tusz_root=tusz_root,
            private_root=private_root,
            tusz_processed_root=tusz_processed_root,
        ):
            add(base / raw)

    if not candidates:
        raw = Path(clean_cell(row.get("edf_path")).replace("\\", "/"))
        root = tusz_root if source == "tusz" else private_root
        candidates = [base / raw for base in _candidate_roots(root, source)]
    return candidates


def resolve_edf_path(
    row: Mapping[str, object],
    tusz_root: Path,
    private_root: Path,
    tusz_processed_root: Optional[Path] = None,
    prefer_processed_tusz: bool = True,
    tusz_processed_subdir: str = DEFAULT_TUSZ_PROCESSED_SUBDIR,
    tusz_processed_suffix: str = DEFAULT_TUSZ_PROCESSED_SUFFIX,
) -> Path:
    candidates = candidate_edf_paths(
        row,
        tusz_root=tusz_root,
        private_root=private_root,
        tusz_processed_root=tusz_processed_root,
        prefer_processed_tusz=prefer_processed_tusz,
        tusz_processed_subdir=tusz_processed_subdir,
        tusz_processed_suffix=tusz_processed_suffix,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def resolve_existing_edf_path(
    row: Mapping[str, object],
    tusz_root: Path,
    private_root: Path,
    tusz_processed_root: Optional[Path] = None,
    prefer_processed_tusz: bool = True,
    tusz_processed_subdir: str = DEFAULT_TUSZ_PROCESSED_SUBDIR,
    tusz_processed_suffix: str = DEFAULT_TUSZ_PROCESSED_SUFFIX,
) -> Tuple[Path, List[str]]:
    candidates = candidate_edf_paths(
        row,
        tusz_root=tusz_root,
        private_root=private_root,
        tusz_processed_root=tusz_processed_root,
        prefer_processed_tusz=prefer_processed_tusz,
        tusz_processed_subdir=tusz_processed_subdir,
        tusz_processed_suffix=tusz_processed_suffix,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate, [str(item) for item in candidates]
    return candidates[0], [str(item) for item in candidates]


def build_raw_lookup(raw) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for name in raw.ch_names:
        normalized = normalize_electrode_name(name)
        lookup.setdefault(normalized, name)
        lookup.setdefault(clean_cell(name).upper().replace("_", "-"), name)
    return lookup


def _direct_bipolar_candidates(a: str, b: str) -> List[str]:
    return [
        f"{a}-{b}", f"{a}_{b}", f"{a}{b}",
        f"EEG {a}-{b}", f"EEG {a}_{b}",
    ]


def read_raw_any(path: Path):
    if path.name.lower().endswith((".fif", ".fif.gz")):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="This filename .* does not conform to MNE naming conventions.*")
                raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
            try:
                raw.set_meas_date(None)
            except Exception:
                raw.info["meas_date"] = None
            return raw
        except ValueError as exc:
            if "Could not find measurement info" in str(exc):
                raise ValueError(
                    f"Invalid or incomplete MNE Raw FIF: {path}. "
                    "Regenerate the corresponding QC raw_filtered/clean_lite file with --overwrite, "
                    "and make sure preprocess_unified_soz is not running while QC is still writing."
                ) from exc
            raise
    suffix = path.suffix.lower()
    last_error: Exception | None = None
    for encoding in ("utf-8", "latin-1"):
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False, encoding=encoding)
            try:
                raw.set_meas_date(None)
            except Exception:
                raw.info["meas_date"] = None
            raw.set_annotations(mne.Annotations([], [], []))
            return raw
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read EDF {path}: {last_error}")


def read_bipolar_and_extra(
    edf_path: Path,
    sfreq: float,
    low_freq: float,
    high_freq: float,
    include_sph: bool,
    skip_filter: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    raw = read_raw_any(edf_path)
    raw.load_data()
    if not skip_filter:
        low = float(low_freq) if float(low_freq) > 0 else None
        high = float(high_freq) if float(high_freq) > 0 else None
        if high is not None:
            high = min(high, float(raw.info["sfreq"]) / 2.0 - 1.0)
        if high is None or high > (low or 0.0):
            raw.filter(low, high, verbose=False)
    if float(sfreq) > 0 and abs(float(raw.info["sfreq"]) - float(sfreq)) > 1e-3:
        raw.resample(float(sfreq), verbose=False)
    actual_sfreq = float(raw.info["sfreq"])
    data_by_name = {name: raw.get_data(picks=[name])[0].astype(np.float32) for name in raw.ch_names}
    lookup = build_raw_lookup(raw)
    n_samples = raw.n_times

    channels: List[np.ndarray] = []
    masks: List[float] = []
    sources: List[str] = []
    for anode, cathode in TCP_PAIRS:
        direct_name = None
        for candidate in _direct_bipolar_candidates(anode, cathode):
            direct_name = lookup.get(candidate.upper().replace("_", "-"))
            if direct_name is not None:
                break
        if direct_name is not None:
            channels.append(data_by_name[direct_name].astype(np.float32))
            masks.append(1.0)
            sources.append(direct_name)
            continue
        anode_name = lookup.get(normalize_electrode_name(anode))
        cathode_name = lookup.get(normalize_electrode_name(cathode))
        if anode_name is None or cathode_name is None:
            channels.append(np.zeros(n_samples, dtype=np.float32))
            masks.append(0.0)
            missing = ",".join(item for item, name in ((anode, anode_name), (cathode, cathode_name)) if name is None)
            sources.append(f"missing:{missing}")
            continue
        channels.append((data_by_name[anode_name] - data_by_name[cathode_name]).astype(np.float32))
        masks.append(1.0)
        sources.append(f"{anode_name}-{cathode_name}")

    if include_sph:
        for electrode in EXTRA_INPUT_ELECTRODES:
            raw_name = lookup.get(electrode)
            if raw_name is None:
                channels.append(np.zeros(n_samples, dtype=np.float32))
                masks.append(0.0)
                sources.append(f"missing:{electrode}")
            else:
                channels.append(data_by_name[raw_name].astype(np.float32))
                masks.append(1.0)
                sources.append(raw_name)

    return np.stack(channels).astype(np.float32), np.asarray(masks, dtype=np.float32), sources, actual_sfreq


def normalize_sequence(seq: np.ndarray, mask: np.ndarray, mode: str, baseline_samples: int) -> np.ndarray:
    mode = str(mode).lower()
    out = seq.astype(np.float32, copy=True)
    if mode == "none":
        out[mask <= 0.5] = 0.0
        return out
    ref = out[:, :max(1, min(int(baseline_samples), out.shape[1]))] if mode.startswith("baseline") else out
    if mode in {"zscore", "baseline_zscore"}:
        center = ref.mean(axis=1, keepdims=True)
        scale = ref.std(axis=1, keepdims=True) + 1e-6
    elif mode in {"robust", "baseline_robust"}:
        center = np.median(ref, axis=1, keepdims=True)
        scale = np.percentile(np.abs(ref - center), 95, axis=1, keepdims=True) + 1e-6
    else:
        raise ValueError("normalize must be none, zscore, robust, baseline_zscore, or baseline_robust")
    out = ((out - center) / scale).astype(np.float32)
    out[mask <= 0.5] = 0.0
    return out


def extract_windows(
    signals: np.ndarray,
    channel_mask: np.ndarray,
    start_sec: float,
    sfreq: float,
    sequence_sec: float,
    window_sec: float,
    normalize: str,
    baseline_sec: float,
) -> np.ndarray:
    sequence_samples = int(round(float(sequence_sec) * float(sfreq)))
    window_samples = int(round(float(window_sec) * float(sfreq)))
    n_windows = int(round(float(sequence_sec) / float(window_sec)))
    start_sample = int(round(float(start_sec) * float(sfreq)))
    seq = np.zeros((signals.shape[0], sequence_samples), dtype=np.float32)
    src_start = max(0, start_sample)
    src_stop = min(signals.shape[1], start_sample + sequence_samples)
    if src_stop > src_start:
        dst_start = src_start - start_sample
        seq[:, dst_start:dst_start + (src_stop - src_start)] = signals[:, src_start:src_stop]
    seq = normalize_sequence(seq, mask=channel_mask, mode=normalize, baseline_samples=int(round(float(baseline_sec) * float(sfreq))))
    windows = []
    for idx in range(n_windows):
        left = idx * window_samples
        windows.append(seq[:, left:left + window_samples])
    return np.stack(windows, axis=0).astype(np.float32)


def seizure_window_labels(
    sequence_start_sec: float,
    seizure_start_sec: float,
    seizure_end_sec: float,
    sequence_sec: float,
    window_sec: float,
    role: str,
) -> np.ndarray:
    n_windows = int(round(float(sequence_sec) / float(window_sec)))
    role_text = str(role or "").lower()
    if role_text.startswith("background") or "negative" in role_text:
        return np.zeros(n_windows, dtype=np.float32)
    labels = np.zeros(n_windows, dtype=np.float32)
    if not np.isfinite(seizure_start_sec) or not np.isfinite(seizure_end_sec):
        return labels
    for idx in range(n_windows):
        center = float(sequence_start_sec) + (idx + 0.5) * float(window_sec)
        labels[idx] = float(center >= seizure_start_sec and center < seizure_end_sec)
    return labels


def _parse_bool(value: object) -> bool:
    text = clean_cell(value).lower()
    return text in {"1", "true", "yes", "y", "是"}


def _input_channel_sources(channel: str) -> Tuple[str, ...]:
    channel = clean_cell(channel).upper().replace("_", "-")
    if channel in TCP_CHANNELS:
        return tuple(TCP_PAIRS[TCP_CHANNELS.index(channel)])
    return (normalize_electrode_name(channel),)


def find_window_qc_path(row: Mapping[str, object], edf_path: Path, qc_root: Optional[Path]) -> Optional[Path]:
    if qc_root is None:
        return None
    explicit = clean_cell(row.get("window_qc_path") or row.get("qc_window_path"))
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = qc_root / path
        if path.is_file():
            return path

    file_stem = safe_name(Path(clean_cell(row.get("edf_path")) or edf_path.name).stem or edf_path.stem)
    subject_candidates = [
        clean_cell(row.get("base_patient_id")),
        clean_cell(row.get("patient_id")),
        clean_cell(row.get("patient_id")).rsplit("_", 1)[0],
        edf_path.parent.name,
    ]
    roots = [qc_root / "qc" / "window_qc", qc_root / "window_qc", qc_root]
    for root in roots:
        for subject in subject_candidates:
            if not subject:
                continue
            path = root / safe_name(subject) / f"{file_stem}_window_qc.csv"
            if path.is_file():
                return path
        path = root / f"{file_stem}_window_qc.csv"
        if path.is_file():
            return path
    return None


def load_window_qc_scores(path: Path) -> Dict[str, List[Dict[str, object]]]:
    by_channel: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            channel = normalize_electrode_name(row.get("channel"))
            if not channel:
                continue
            start = parse_float(row.get("window_start"))
            end = parse_float(row.get("window_end"))
            score = parse_float(row.get("artifact_score"), 0.0)
            if not np.isfinite(start) or not np.isfinite(end):
                continue
            by_channel[channel].append({
                "start": float(start),
                "end": float(end),
                "score": float(np.clip(score, 0.0, 1.0)),
                "needs_review": _parse_bool(row.get("needs_review")),
                "is_core_onset_window": _parse_bool(row.get("is_core_onset_window")),
                "action": clean_cell(row.get("action")),
                "artifact_type": clean_cell(row.get("artifact_type")),
            })
    for rows in by_channel.values():
        rows.sort(key=lambda item: float(item["start"]))
    return by_channel


def _max_qc_score_for_interval(rows: Sequence[Dict[str, object]], start: float, end: float) -> Tuple[float, float]:
    best = 0.0
    seen = 0.0
    for item in rows:
        item_start = float(item["start"])
        item_end = float(item["end"])
        if item_end <= start:
            continue
        if item_start >= end:
            break
        best = max(best, float(item["score"]))
        seen = 1.0
    return best, seen


def artifact_windows_for_sample(
    qc_by_channel: Optional[Dict[str, List[Dict[str, object]]]],
    *,
    input_channels: Sequence[str],
    sequence_start_sec: float,
    sequence_sec: float,
    window_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_windows = int(round(float(sequence_sec) / float(window_sec)))
    artifact = np.zeros((n_windows, len(input_channels)), dtype=np.float32)
    mask = np.zeros_like(artifact, dtype=np.float32)
    if not qc_by_channel:
        return artifact, mask

    for win_idx in range(n_windows):
        start = float(sequence_start_sec) + win_idx * float(window_sec)
        end = start + float(window_sec)
        for ch_idx, channel in enumerate(input_channels):
            best = 0.0
            seen = 0.0
            for source_channel in _input_channel_sources(channel):
                score, source_seen = _max_qc_score_for_interval(
                    qc_by_channel.get(normalize_electrode_name(source_channel), []),
                    start,
                    end,
                )
                best = max(best, score)
                seen = max(seen, source_seen)
            artifact[win_idx, ch_idx] = best
            mask[win_idx, ch_idx] = seen
    return artifact, mask


def vector(row: Dict[str, str], columns: Sequence[str], default: float = 0.0) -> np.ndarray:
    return np.asarray([parse_float(row.get(col), default) for col in columns], dtype=np.float32)


def derive_regions_from_channel_labels(label: np.ndarray) -> np.ndarray:
    region_label = np.zeros(len(REGION_NAMES), dtype=np.float32)
    for channel_idx, value in enumerate(label):
        if float(value) <= 0.5:
            continue
        channel = TCP_CHANNELS[channel_idx]
        for region in CHANNEL_TO_REGIONS.get(channel, ()):
            region_label[REGION_NAMES.index(region)] = 1.0
    return region_label


def derive_hemisphere_label(row: Dict[str, str]) -> Tuple[int, float]:
    raw_label = clean_cell(row.get("hemisphere_label"))
    try:
        return int(float(raw_label)), 1.0
    except (TypeError, ValueError):
        pass
    hemi = clean_cell(row.get("hemisphere")).upper()
    if hemi in HEMISPHERE_INDEX:
        return int(HEMISPHERE_INDEX[hemi]), 1.0
    return -1, 0.0


def sample_specs_for_row(
    row: Dict[str, str],
    roles: Sequence[str],
    sequence_sec: float,
    background_gap_sec: float,
    onset_pre_sec: float,
) -> List[Tuple[str, float, float]]:
    start = parse_float(row.get("sz_start"))
    end = parse_float(row.get("sz_end"))
    duration = parse_float(row.get("duration_sec"))
    if not np.isfinite(duration):
        duration = max(end + background_gap_sec + sequence_sec, 0.0) if np.isfinite(end) else 0.0
    specs: List[Tuple[str, float, float]] = []
    if any(str(role).lower() in {"segment", "row_segment", "exact"} for role in roles):
        if np.isfinite(start):
            role = clean_cell(row.get("sample_role")) or "segment"
            specs.append((role, max(0.0, start), 1.0))
        return specs
    if np.isfinite(start):
        if "onset" in roles:
            specs.append(("onset", max(0.0, start - float(onset_pre_sec)), 1.0))
        if "early_ictal" in roles:
            specs.append(("early_ictal", max(0.0, start + 4.0), 0.5))
        if "propagation" in roles and np.isfinite(end) and end > start + 8.0:
            specs.append(("propagation", max(0.0, min(start + 8.0, end - sequence_sec)), 0.0))
        if "background" in roles:
            pre_start = start - float(background_gap_sec) - float(sequence_sec)
            if pre_start >= 0:
                specs.append(("background_pre", pre_start, 0.0))
            post_start = end + float(background_gap_sec) if np.isfinite(end) else float("nan")
            if np.isfinite(post_start) and post_start + sequence_sec <= duration:
                specs.append(("background_post", post_start, 0.0))
    return specs


def save_sample(
    output_dir: Path,
    split: str,
    sample_id: str,
    payload: Dict[str, object],
) -> str:
    rel_path = f"{split}/{sample_id}.npz"
    path = output_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return rel_path


def preprocess(args) -> Dict[str, object]:
    mne.set_log_level("ERROR")
    manifest = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv_rows(manifest)
    input_row_count = len(rows)
    if args.source_filter not in ("all", "both"):
        wanted = {item.strip().lower() for item in args.source_filter.split(",") if item.strip()}
        rows = [row for row in rows if clean_cell(row.get("source")).lower() in wanted]
    if args.splits:
        wanted_splits = {item.strip().lower() for item in args.splits.split(",") if item.strip()}
        rows = [row for row in rows if clean_cell(row.get("split")).lower() in wanted_splits]
    rows = filter_rows_by_seizure_type(rows, args.seizure_types)
    filtered_row_count = len(rows)
    if int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]

    roles = tuple(item.strip() for item in args.roles.split(",") if item.strip())
    tusz_root = Path(args.tusz_root)
    private_root = Path(args.private_root)
    tusz_processed_root = Path(args.tusz_processed_root) if clean_cell(args.tusz_processed_root) else None
    qc_root = Path(args.qc_root) if clean_cell(args.qc_root) else None
    failed_qc_log = resolve_failed_qc_log(args, qc_root)
    failed_qc_keys: set[str] = set()
    failed_qc_paths: List[str] = []
    if failed_qc_log is not None:
        failed_qc_keys, failed_qc_paths = load_failed_qc_path_keys(failed_qc_log)
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    path_lookup: Dict[str, Path] = {}
    path_candidates_lookup: Dict[str, List[str]] = {}
    path_match_lookup: Dict[str, set[str]] = {}
    for idx, row in enumerate(rows):
        row["_row_index"] = str(idx)
        edf_path, candidates = resolve_existing_edf_path(
            row,
            tusz_root=tusz_root,
            private_root=private_root,
            tusz_processed_root=tusz_processed_root,
            prefer_processed_tusz=bool(args.prefer_processed_tusz),
            tusz_processed_subdir=args.tusz_processed_subdir,
            tusz_processed_suffix=args.tusz_processed_suffix,
        )
        key = str(edf_path)
        grouped[key].append(row)
        path_lookup[key] = edf_path
        path_candidates_lookup[key] = candidates
        path_match_lookup.setdefault(key, set()).update(row_path_match_keys(row, edf_path, candidates))

    index_rows: List[Dict[str, object]] = []
    stats = Counter()
    failed_files: List[Dict[str, str]] = []
    skipped_failed_qc_files: List[Dict[str, str]] = []
    qc_cache: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for key, file_rows in grouped.items():
        edf_path = path_lookup[key]
        if failed_qc_keys and path_match_lookup.get(key, set()).intersection(failed_qc_keys):
            stats["skipped_failed_qc_files"] += 1
            stats["skipped_failed_qc_events"] += len(file_rows)
            if len(skipped_failed_qc_files) < 20:
                skipped_failed_qc_files.append({
                    "path": str(edf_path),
                    "error": "listed_in_failed_qc_log",
                    "failed_qc_log": str(failed_qc_log),
                })
            continue
        if not edf_path.is_file():
            stats["missing_files"] += 1
            stats["missing_events"] += len(file_rows)
            if len(failed_files) < 20:
                failed_files.append({
                    "path": str(edf_path),
                    "error": "missing",
                    "candidates": ";".join(path_candidates_lookup.get(key, [])),
                })
            continue
        try:
            signals, signal_mask, source_channels, actual_sfreq = read_bipolar_and_extra(
                edf_path,
                sfreq=args.sfreq,
                low_freq=args.low_freq,
                high_freq=args.high_freq,
                include_sph=bool(args.include_sph),
                skip_filter=bool(args.skip_filter),
            )
            duration_sec = signals.shape[1] / float(actual_sfreq)
            input_channels = list(TCP_CHANNELS) + (list(EXTRA_INPUT_ELECTRODES) if args.include_sph else [])
            for row in file_rows:
                split = clean_cell(row.get("split")) or "train"
                source = clean_cell(row.get("source")) or "unknown"
                if split == "private":
                    split = "private"
                sz_start = parse_float(row.get("sz_start"))
                sz_end = parse_float(row.get("sz_end"))
                label = vector(row, TCP_COLUMNS)
                label_mask = vector(row, CHANNEL_LABEL_MASK_COLUMNS, default=1.0) * signal_mask[: len(TCP_CHANNELS)]
                prop = vector(row, CHANNEL_PROP_COLUMNS)
                region_label = vector(row, REGION_LABEL_COLUMNS)
                region_mask = vector(row, REGION_MASK_COLUMNS)
                region_prop = vector(row, REGION_PROP_COLUMNS)
                has_channel_label = bool(label.sum() > 0.5)
                if has_channel_label and region_label.sum() <= 0.0:
                    region_label = derive_regions_from_channel_labels(label)
                if has_channel_label and region_mask.sum() <= 0.0:
                    region_mask = np.ones_like(region_label, dtype=np.float32)
                spatial_weight = parse_float(row.get("spatial_loss_weight"), 1.0)
                label_conf = parse_float(row.get("label_confidence"), 1.0)
                sample_specs = sample_specs_for_row(
                    row,
                    roles=roles,
                    sequence_sec=args.sequence_sec,
                    background_gap_sec=args.background_gap_sec,
                    onset_pre_sec=args.onset_pre_sec,
                )
                if not sample_specs:
                    stats["events_without_samples"] += 1
                    continue
                for local_idx, (role, seq_start, role_weight) in enumerate(sample_specs):
                    x = extract_windows(
                        signals,
                        signal_mask,
                        start_sec=seq_start,
                        sfreq=actual_sfreq,
                        sequence_sec=args.sequence_sec,
                        window_sec=args.window_sec,
                        normalize=args.normalize,
                        baseline_sec=args.baseline_sec,
                    )
                    qc_path = find_window_qc_path(row, edf_path, qc_root)
                    qc_by_channel = None
                    if qc_path is not None:
                        qc_key = str(qc_path)
                        if qc_key not in qc_cache:
                            qc_cache[qc_key] = load_window_qc_scores(qc_path)
                        qc_by_channel = qc_cache[qc_key]
                        stats["samples_with_qc"] += 1
                    else:
                        stats["samples_without_qc"] += 1
                    artifact_score, artifact_mask = artifact_windows_for_sample(
                        qc_by_channel,
                        input_channels=input_channels,
                        sequence_start_sec=seq_start,
                        sequence_sec=args.sequence_sec,
                        window_sec=args.window_sec,
                    )
                    seizure_y = seizure_window_labels(
                        sequence_start_sec=seq_start,
                        seizure_start_sec=sz_start,
                        seizure_end_sec=sz_end,
                        sequence_sec=args.sequence_sec,
                        window_sec=args.window_sec,
                        role=role,
                    )
                    channel_label_mask = label_mask.copy()
                    region_label_mask = region_mask.copy()
                    propagation_mask = np.zeros_like(region_prop)
                    sample_weight = float(spatial_weight) * float(label_conf) * float(role_weight)
                    if role.startswith("background"):
                        channel_label_mask[:] = 0.0
                        region_label_mask[:] = 0.0
                        propagation_mask[:] = 0.0
                    elif role == "propagation":
                        channel_label_mask[:] = 0.0
                        region_label_mask[:] = 0.0
                        propagation_mask[:] = 1.0
                        sample_weight = max(float(label_conf), 0.5)
                    else:
                        propagation_mask[:] = (region_prop > 0.5).astype(np.float32)
                    hemi_label_int, hemi_mask = derive_hemisphere_label(row)
                    sample_id = (
                        f"{source}_{safe_name(row.get('base_patient_id') or row.get('patient_id'))}_"
                        f"{safe_name(row.get('event_id') or row.get('_row_index'))}_{role}_{local_idx:02d}"
                    )
                    effective_sample_weight = float(sample_weight)
                    if args.artifact_weight_mode == "downweight" and artifact_mask.any():
                        mean_artifact_for_weight = float(artifact_score[artifact_mask > 0].mean())
                        effective_sample_weight = max(
                            0.0,
                            sample_weight * (1.0 - float(args.artifact_penalty_alpha) * mean_artifact_for_weight),
                        )

                    payload = {
                        "x": x.astype(np.float32),
                        "input_channel_mask": signal_mask.astype(np.float32),
                        "artifact_score": artifact_score.astype(np.float32),
                        "artifact_mask": artifact_mask.astype(np.float32),
                        "channel_labels": label.astype(np.float32),
                        "channel_label_mask": channel_label_mask.astype(np.float32),
                        "region_labels": region_label.astype(np.float32),
                        "region_label_mask": region_label_mask.astype(np.float32),
                        "propagation_region_labels": region_prop.astype(np.float32),
                        "propagation_region_mask": propagation_mask.astype(np.float32),
                        "seizure_y": seizure_y.astype(np.float32),
                        "seizure_mask": np.ones_like(seizure_y, dtype=np.float32),
                        "hemisphere_label": np.asarray(hemi_label_int, dtype=np.int64),
                        "hemisphere_mask": np.asarray(hemi_mask, dtype=np.float32),
                        "sample_weight": np.asarray(effective_sample_weight, dtype=np.float32),
                    }
                    rel_path = save_sample(output_dir, split, sample_id, payload)
                    artifact_known = artifact_mask > 0
                    artifact_mean = float(artifact_score[artifact_known].mean()) if artifact_known.any() else float("nan")
                    artifact_max = float(artifact_score[artifact_known].max()) if artifact_known.any() else float("nan")
                    index_row = {
                        "npz_path": rel_path,
                        "split": split,
                        "source": source,
                        "patient_id": clean_cell(row.get("patient_id")),
                        "base_patient_id": clean_cell(row.get("base_patient_id")) or clean_cell(row.get("patient_id")),
                        "edf_path": clean_cell(row.get("edf_path")),
                        "resolved_eeg_path": str(edf_path),
                        "event_id": clean_cell(row.get("event_id")),
                        "sample_id": sample_id,
                        "sample_role": role,
                        "sample_weight": effective_sample_weight,
                        "sequence_start_sec": float(seq_start),
                        "sequence_end_sec": float(seq_start + float(args.sequence_sec)),
                        "sz_start": sz_start,
                        "sz_end": sz_end,
                        "seizure_type": clean_cell(row.get("seizure_type")),
                        "hemisphere": clean_cell(row.get("hemisphere")),
                        "review_status": clean_cell(row.get("review_status")),
                        "quality_flags": clean_cell(row.get("quality_flags")),
                        "actual_sfreq": actual_sfreq,
                        "duration_sec": duration_sec,
                        "n_input_channels": int(signals.shape[0]),
                        "n_label_channels": len(TCP_CHANNELS),
                        "artifact_score_mean": artifact_mean,
                        "artifact_score_max": artifact_max,
                        "artifact_qc_path": str(qc_path) if qc_path is not None else "",
                    }
                    index_rows.append(index_row)
                    stats["samples_saved"] += 1
                    stats[f"role_{role}"] += 1
                    stats[f"source_{source}"] += 1
            stats["files_ok"] += 1
        except Exception as exc:
            stats["files_failed"] += 1
            stats["events_failed"] += len(file_rows)
            if len(failed_files) < 20:
                failed_files.append({"path": str(edf_path), "error": str(exc)})
            traceback.print_exc()

    index_fields = [
        "npz_path", "split", "source", "patient_id", "base_patient_id", "edf_path",
        "resolved_eeg_path", "event_id", "sample_id", "sample_role", "sample_weight", "sequence_start_sec",
        "sequence_end_sec", "sz_start", "sz_end", "seizure_type", "hemisphere",
        "review_status", "quality_flags", "actual_sfreq", "duration_sec",
        "n_input_channels", "n_label_channels", "artifact_score_mean",
        "artifact_score_max", "artifact_qc_path",
    ]
    write_csv_rows(output_dir / "index.csv", index_rows, index_fields)
    for split in sorted({clean_cell(row.get("split")) for row in index_rows if clean_cell(row.get("split"))}):
        write_csv_rows(output_dir / f"index_{split}.csv", [row for row in index_rows if row.get("split") == split], index_fields)

    summary = {
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "tusz_root": str(tusz_root),
        "tusz_processed_root": str(tusz_processed_root) if tusz_processed_root is not None else "",
        "private_root": str(private_root),
        "qc_root": str(qc_root) if qc_root is not None else "",
        "failed_qc_log": str(failed_qc_log) if failed_qc_log is not None else "",
        "failed_qc_log_entries": len(failed_qc_paths),
        "args": vars(args),
        "input_rows": input_row_count,
        "rows_after_filters": filtered_row_count,
        "stats": dict(stats),
        "failed_files_examples": failed_files,
        "skipped_failed_qc_examples": skipped_failed_qc_files,
        "input_channels": list(TCP_CHANNELS) + (list(EXTRA_INPUT_ELECTRODES) if args.include_sph else []),
        "label_channels": list(TCP_CHANNELS),
        "n_samples": len(index_rows),
    }
    (output_dir / "preprocess_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess unified SOZ manifest into NPZ sequences")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--tusz_root", default=DEFAULT_TUSZ_ROOT)
    parser.add_argument(
        "--tusz_processed_root",
        default="",
        help="Optional root containing preprocessed TUSZ FIF files; searched before --tusz_root when available.",
    )
    parser.add_argument(
        "--no_prefer_processed_tusz",
        dest="prefer_processed_tusz",
        action="store_false",
        help="Do not prefer processed_path/FIF candidates for TUSZ rows.",
    )
    parser.set_defaults(prefer_processed_tusz=True)
    parser.add_argument("--tusz_processed_subdir", default=DEFAULT_TUSZ_PROCESSED_SUBDIR)
    parser.add_argument("--tusz_processed_suffix", default=DEFAULT_TUSZ_PROCESSED_SUFFIX)
    parser.add_argument("--private_root", default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--qc_root", default="", help="Optional root containing qc/window_qc reports from src.preprocessing")
    parser.add_argument("--failed_qc_log", default="", help="Optional run_preprocess_qc error_log.csv; defaults to <qc_root>/qc/error_log.csv when present")
    parser.add_argument("--ignore_failed_qc_log", action="store_true", help="Do not skip files listed in the QC failure log")
    parser.add_argument("--source_filter", default="all", help="all, tusz, private, or comma-separated")
    parser.add_argument("--splits", default="", help="Optional comma-separated split filter")
    parser.add_argument("--seizure_types", default="", help="Optional comma-separated seizure type filter, e.g. fnsz")
    parser.add_argument(
        "--roles",
        default="onset",
        help="Comma-separated: onset,early_ictal,propagation,background,segment. segment uses each row's exact sz_start.",
    )
    parser.add_argument("--sfreq", type=float, default=200.0)
    parser.add_argument("--low_freq", type=float, default=1.0)
    parser.add_argument("--high_freq", type=float, default=50.0)
    parser.add_argument("--skip_filter", action="store_true")
    parser.add_argument("--sequence_sec", type=float, default=10.0)
    parser.add_argument("--window_sec", type=float, default=1.0)
    parser.add_argument("--onset_pre_sec", type=float, default=2.0)
    parser.add_argument("--baseline_sec", type=float, default=2.0)
    parser.add_argument("--background_gap_sec", type=float, default=300.0)
    parser.add_argument("--normalize", choices=["none", "zscore", "robust", "baseline_zscore", "baseline_robust"], default="baseline_robust")
    parser.add_argument("--include_sph", action="store_true", help="Append SPHL/SPHR raw electrode features when present")
    parser.add_argument("--artifact_weight_mode", choices=["none", "downweight"], default="none", help="Keep ArtifactScore as metadata or downweight sample_weight by mean artifact")
    parser.add_argument("--artifact_penalty_alpha", type=float, default=0.5, help="Weight penalty multiplier used only when --artifact_weight_mode downweight")
    parser.add_argument("--max_rows", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    preprocess(parse_args())
