#!/usr/bin/env python3
"""Shared parsing, normalization, and CSV helpers for ``code.soz_pre``."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from soz_pre.constants import (
        HEMISPHERE_CLASSES,
        SPH_TO_NEIGHBOR_ELECTRODES,
        TCP_CHANNELS,
        TCP_COLUMNS,
        TCP_INDEX,
    )
except ImportError:  # pragma: no cover - package import fallback
    from code.soz_pre.constants import (
        HEMISPHERE_CLASSES,
        SPH_TO_NEIGHBOR_ELECTRODES,
        TCP_CHANNELS,
        TCP_COLUMNS,
        TCP_INDEX,
    )


NO_LABEL_VALUES = {"", "NAN", "NA", "NONE", "NULL", "无", "未见", "-"}
DIFFUSE_TOKENS = {"弥漫性", "弥漫", "广泛", "全导", "全脑"}


def clean_cell(value: object) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.upper() in NO_LABEL_VALUES else text


def parse_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def normalize_patient_name(value: object) -> str:
    text = clean_cell(value)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[_-]?SZ\d.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[-_]?NO$", "", text, flags=re.IGNORECASE)
    return text


def base_patient_id(value: object) -> str:
    text = clean_cell(value)
    parts = text.rsplit("_", 1)
    if len(parts) == 2 and parts[1].upper().startswith("SZ"):
        return parts[0]
    return normalize_patient_name(text)


def normalize_electrode_name(name: object) -> str:
    text = clean_cell(name).upper()
    text = text.replace(" ", "").replace("_", "-")
    for prefix in ("EEG-", "EEG"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    for suffix in ("-REF", "-LE", "-AR", "-AVG", "-A1", "-A2"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    text = text.strip("-")
    aliases = {
        "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
        "M1": "A1", "M2": "A2",
        "SP1": "SPHL", "SPH1": "SPHL", "SP-L": "SPHL",
        "SPH-L": "SPHL", "SPHL": "SPHL", "SP-L": "SPHL",
        "SP2": "SPHR", "SPH2": "SPHR", "SP-R": "SPHR",
        "SPH-R": "SPHR", "SPHR": "SPHR", "SP-R": "SPHR",
        "FPZ": "FPZ", "OZ": "OZ",
    }
    return aliases.get(text, text)


def split_label_tokens(value: object) -> List[str]:
    text = clean_cell(value)
    if not text:
        return []
    text = (
        text.replace("，", ",")
        .replace("；", ",")
        .replace("、", ",")
        .replace(";", ",")
        .replace("→", ",")
        .replace("/", ",")
        .replace("著", "")
    )
    return [token.strip() for token in re.split(r"[,\s]+", text) if token.strip()]


def parse_electrodes(value: object, *, keep_diffuse: bool = True) -> List[str]:
    text = clean_cell(value)
    if not text:
        return []
    out: List[str] = []
    if keep_diffuse and any(marker in text for marker in DIFFUSE_TOKENS):
        out.append("DIFFUSE")
    for token in split_label_tokens(text):
        if token in DIFFUSE_TOKENS or token.upper() in NO_LABEL_VALUES:
            continue
        if not re.fullmatch(
            r"(?i)(FP[12Z]|FZ|F[3478]|CZ|C[3456]|PZ|P[34]|O[12Z]|"
            r"T[345678]|A[12]|M[12]|SPH?[-_]?[LR]|SPH?[12]|SP[-_]?[LR]|SP[12])",
            token,
        ):
            continue
        normalized = normalize_electrode_name(token)
        if normalized and normalized not in out:
            out.append(normalized)
    return out


def expand_sph_electrodes(electrodes: Iterable[str]) -> List[str]:
    out: List[str] = []
    for electrode in electrodes or []:
        normalized = normalize_electrode_name(electrode)
        if normalized == "DIFFUSE":
            if normalized not in out:
                out.append(normalized)
            continue
        for item in SPH_TO_NEIGHBOR_ELECTRODES.get(normalized, (normalized,)):
            if item not in out:
                out.append(item)
    return out


def canonical_bipolar_token(token: object) -> Optional[str]:
    text = clean_cell(token).upper()
    if not text:
        return None
    text = text.replace("–", "-").replace("—", "-").replace("_", "-")
    text = re.sub(r"\s+", "", text)
    direct_lookup = {ch.upper(): ch for ch in TCP_CHANNELS}
    if text in direct_lookup:
        return direct_lookup[text]
    parts = text.split("-")
    if len(parts) != 2:
        return None
    a = normalize_electrode_name(parts[0])
    b = normalize_electrode_name(parts[1])
    canonical = f"{a}-{b}"
    if canonical in TCP_INDEX:
        return canonical
    reverse = f"{b}-{a}"
    if reverse in TCP_INDEX:
        return reverse
    return None


def parse_bipolar_list(value: object) -> List[str]:
    out: List[str] = []
    for token in split_label_tokens(value):
        channel = canonical_bipolar_token(token)
        if channel and channel not in out:
            out.append(channel)
    return out


def parse_sz_ids_from_stem(value: object) -> List[str]:
    stem = Path(clean_cell(value)).stem.upper().replace("-", "_")
    stem = re.sub(r"_\d{4}$", "", stem)
    match = re.search(r"SZ(\d+)(?:_(\d+))?", stem)
    if not match:
        return []
    first = int(match.group(1))
    second = match.group(2)
    if second is None:
        return [f"SZ{first}"]
    last = int(second)
    if last >= first:
        return [f"SZ{i}" for i in range(first, last + 1)]
    return [f"SZ{first}", f"SZ{last}"]


def normalize_sz_id(value: object) -> str:
    text = clean_cell(value).upper().replace("-", "_")
    match = re.search(r"SZ\s*(\d+)", text)
    return f"SZ{int(match.group(1))}" if match else text


def normalize_hemisphere(value: object) -> str:
    text = clean_cell(value).upper()
    if text in HEMISPHERE_CLASSES:
        return text
    if "双" in text or "BIL" in text:
        return "B"
    if "左" in text or text.startswith("L"):
        return "L"
    if "右" in text or text.startswith("R"):
        return "R"
    if "中线" in text or text.startswith("M"):
        return "M"
    return ""


def infer_regions_from_text(text: object, hemisphere: object = "") -> List[str]:
    raw = clean_cell(text)
    hemi = normalize_hemisphere(hemisphere)
    regions: List[str] = []

    def add(region: str) -> None:
        if region not in regions:
            regions.append(region)

    has_left = "左" in raw or hemi == "L"
    has_right = "右" in raw or hemi == "R"
    has_temporal = "颞" in raw or "侧裂" in raw
    has_frontal = "额" in raw or "前头" in raw
    has_central = "中央" in raw or "中线" in raw or "旁中线" in raw
    has_parietal = "顶" in raw or has_central
    if has_temporal:
        if has_left:
            add("left_temporal")
        if has_right:
            add("right_temporal")
    if has_frontal:
        if has_left:
            add("left_frontal")
        if has_right:
            add("right_frontal")
    if has_parietal:
        add("central_parietal")
    if not regions and hemi == "L":
        add("left_temporal")
    if not regions and hemi == "R":
        add("right_temporal")
    return regions


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames and key not in extra_fields:
                extra_fields.append(key)
    ordered = list(fieldnames) + extra_fields
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ordered})


def semicolon(items: Iterable[object]) -> str:
    return ";".join(clean_cell(item) for item in items if clean_cell(item))


def comma(items: Iterable[object]) -> str:
    return ",".join(clean_cell(item) for item in items if clean_cell(item))


def vector_from_row(row: Dict[str, object], columns: Sequence[str], default: float = 0.0) -> np.ndarray:
    return np.asarray([parse_float(row.get(col), default=default) for col in columns], dtype=np.float32)


def dataframe_from_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, header=None)
