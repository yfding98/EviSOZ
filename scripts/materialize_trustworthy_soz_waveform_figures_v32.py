#!/usr/bin/env python3
"""Render target-blind processed EEG waveforms for trustworthy SOZ reports.

The figures use the same standard-19, 0.5--45 Hz, 200 Hz, CAR19 event window
as the localization pipeline.  No SOZ target or evaluation row is opened.
Private reports receive their exact event waveform.  A public patient report
receives the first deterministically ordered reportable event that can be
replayed; this is explicitly marked as a representative event because the
patient-level score aggregates all eligible seizures.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurosoz-v32-matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.edf import CausalEDFConfig, load_standard19_edf_event  # noqa: E402
from src.soz.geometry import STANDARD_19  # noqa: E402


SCHEMA = "trustworthy_soz_processed_waveform_figures_v32"
DEFAULT_REPORTS = ROOT / "outputs/trustworthy_soz_qualified_reports_v24_1_20260815"
DEFAULT_PRIVATE_BUNDLE = ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814"
DEFAULT_PRIVATE_EVIDENCE = ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814"
DEFAULT_PRIVATE_DESCRIPTORS = ROOT / "outputs/private_event_descriptors_target_blind_v24_20260815"
DEFAULT_PUBLIC_SOURCE = ROOT / "outputs/target_free_oof_reports_v3_recovered_20260813.json"
DEFAULT_PUBLIC_PREFLIGHT = ROOT / "outputs/deepsoz_signal_preflight_identity_v3_20260812/deepsoz_signal_preflight_identity_v3.json"
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_processed_waveforms_v32_20260816"
CJK_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if CJK_FONT_PATH.is_file():
    font_manager.fontManager.addfont(str(CJK_FONT_PATH))
    _CJK_FONT_FAMILY = font_manager.FontProperties(fname=str(CJK_FONT_PATH)).get_name()
else:  # Fail visibly on hosts without the expected font rather than emit blank labels.
    raise FileNotFoundError(f"Chinese report font is unavailable: {CJK_FONT_PATH}")
plt.rcParams["font.family"] = _CJK_FONT_FAMILY
plt.rcParams["font.sans-serif"] = [_CJK_FONT_FAMILY]
plt.rcParams["axes.unicode_minus"] = False


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _processed_window_sha256(eeg: torch.Tensor) -> str:
    if tuple(eeg.shape) != (19, 12_000) or eeg.dtype != torch.float32:
        raise ValueError("processed EEG must be float32 [19,12000]")
    array = eeg.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    digest = hashlib.sha256()
    digest.update(b"float32-le:[19,12000]:")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _signal_binding(loaded: object, figure: Path) -> dict[str, object]:
    """Return immutable source/preprocessing/window/figure bindings.

    ``load_standard19_edf_event`` already verifies the EDF and returns
    dataclass receipts.  Hashing those receipts and the canonical float32
    tensor prevents a later report from silently swapping either the source
    recording, preprocessing policy, processed window, or rendered PNG.
    """

    edf_receipt = getattr(loaded, "edf_receipt", None)
    signal_receipt = getattr(loaded, "signal_receipt", None)
    window = getattr(loaded, "window", None)
    if edf_receipt is None or signal_receipt is None or window is None:
        raise TypeError("loaded EEG event lacks receipt-bearing signal data")
    receipt_payload = {
        "edf_receipt": asdict(edf_receipt),
        "signal_receipt": asdict(signal_receipt),
    }
    return {
        "source_signal_sha256": str(edf_receipt.edf_sha256),
        "preprocessing_receipt_sha256": _canonical_sha256(receipt_payload),
        "processed_window_sha256": _processed_window_sha256(window.data),
        "figure_sha256": _sha256(figure),
    }


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.resolve(strict=True).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _csv(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def _safe_relative_edf(root: Path, value: object) -> Path:
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".edf":
        raise ValueError(f"unsafe EDF path: {value!r}")
    resolved = root.joinpath(*relative.parts).resolve(strict=True)
    resolved.relative_to(root)
    return resolved


def _interval(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError("evidence interval must be null or a two-value list")
    start, stop = float(value[0]), float(value[1])
    if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
        raise ValueError("evidence interval is invalid")
    return start, stop


def _nice_spacing_uv(signal_uv: np.ndarray) -> float:
    robust = float(np.quantile(np.abs(signal_uv), 0.99))
    robust = min(max(robust, 40.0), 1000.0)
    magnitude = 10.0 ** math.floor(math.log10(robust))
    normalized = robust / magnitude
    nice = 1.0 if normalized <= 1.0 else (2.0 if normalized <= 2.0 else (5.0 if normalized <= 5.0 else 10.0))
    return 2.0 * nice * magnitude


def _focus_limits(interval: tuple[float, float] | None) -> tuple[float, float]:
    if interval is None:
        return -4.0, 8.0
    start = max(-12.0, interval[0] - 4.0)
    stop = min(48.0, interval[1] + 6.0)
    if stop - start < 12.0:
        stop = min(48.0, start + 12.0)
        start = max(-12.0, stop - 12.0)
    return start, stop


def render_waveform_png(
    eeg: torch.Tensor,
    output: Path,
    *,
    unit_id: str,
    event_id: str,
    evidence_interval: tuple[float, float] | None,
    representative_event: bool,
) -> dict[str, float]:
    if tuple(eeg.shape) != (19, 12_000) or eeg.dtype != torch.float32:
        raise ValueError("processed EEG must be float32 [19,12000]")
    if not torch.isfinite(eeg).all():
        raise ValueError("processed EEG contains non-finite values")
    signal_uv = eeg.detach().cpu().numpy().astype(np.float64, copy=False) * 1e6
    spacing_uv = _nice_spacing_uv(signal_uv)
    time_sec = np.arange(12_000, dtype=np.float64) / 200.0 - 12.0

    figure, axes = plt.subplots(1, 2, figsize=(22, 12), sharey=True)
    focus = _focus_limits(evidence_interval)
    panels = ((-12.0, 48.0, "60秒处理后波形总览"), (*focus, "算法变化区间局部放大"))
    baselines = np.arange(18, -1, -1, dtype=np.float64)
    for panel_index, (axis, panel) in enumerate(zip(axes, panels)):
        left, right, title = panel
        keep = (time_sec >= left) & (time_sec <= right)
        for channel_index, baseline in enumerate(baselines):
            axis.plot(
                time_sec[keep],
                baseline + signal_uv[channel_index, keep] / spacing_uv,
                color="#172033",
                linewidth=0.38 if panel_index == 0 else 0.55,
                alpha=0.9,
                rasterized=True,
            )
        axis.axvline(0.0, color="#b42318", linestyle="--", linewidth=1.2, label="事件标记")
        if evidence_interval is not None:
            axis.axvspan(
                evidence_interval[0], evidence_interval[1], color="#f4b942", alpha=0.22,
                label="算法变化区间",
            )
        axis.set_xlim(left, right)
        axis.set_ylim(-1.0, 19.0)
        axis.set_yticks(baselines)
        axis.set_yticklabels(STANDARD_19, fontsize=8)
        axis.grid(axis="x", color="#d8deea", linewidth=0.5, alpha=0.75)
        axis.set_xlabel("相对事件标记时间（秒）")
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[0].set_ylabel("标准19导（共平均参考）")
    calibration_x = 45.5
    calibration_height = 100.0 / spacing_uv
    axes[0].plot([calibration_x, calibration_x], [0.1, 0.1 + calibration_height], color="#2457d6", linewidth=2.2)
    axes[0].text(calibration_x - 0.3, 0.2 + calibration_height, "100 µV", color="#2457d6", fontsize=8, ha="right")
    scope = "患者级报告的代表性事件" if representative_event else "当前报告事件"
    figure.suptitle(
        f"处理后头皮脑电 · {unit_id} · {scope} {event_id}",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.012,
        "0.5–45 Hz，200 Hz，标准19导共平均参考。红色虚线为事件标记；黄色区域为算法检测的持续变化区间，不能解释为医生确认的发作起始或SOZ起始。",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#5b6475",
    )
    figure.tight_layout(rect=(0.02, 0.035, 1.0, 0.965))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120, facecolor="white")
    plt.close(figure)
    return {"display_spacing_uv": spacing_uv, "calibration_bar_uv": 100.0}


def _private_descriptor_map(directory: Path) -> dict[str, dict[str, object]]:
    manifest = _json(directory / "manifest.json")
    rows = _jsonl(directory / str(manifest["descriptor_file"]))
    return {str(row["event_id"]): row for row in rows}


def _public_event_map(preflight: Mapping[str, object]) -> dict[str, dict[str, object]]:
    receipt = preflight.get("receipt")
    if not isinstance(receipt, Mapping) or not isinstance(receipt.get("events"), list):
        raise TypeError("public preflight lacks event rows")
    result: dict[str, dict[str, object]] = {}
    for raw in receipt["events"]:
        if not isinstance(raw, dict):
            raise TypeError("public preflight event is not an object")
        event_id = str(raw.get("event_id", ""))
        if event_id and event_id not in result:
            result[event_id] = raw
    return result


def materialize(args: argparse.Namespace) -> dict[str, object]:
    target = args.output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    private_bundle = _json(args.private_bundle / "manifest.json")
    private_root = Path(str(private_bundle["eeg_root"])).resolve(strict=True)
    private_signal_rows = {row["event_id"]: row for row in _csv(args.private_bundle / "signal_roster.csv")}
    private_events = _json(args.private_evidence / "manifest.json").get("events")
    if not isinstance(private_events, list):
        raise TypeError("private evidence has no event roster")
    private_ids = [str(row["event_id"]) for row in private_events if isinstance(row, Mapping)]
    if args.private_event_id:
        requested_private = set(args.private_event_id)
        missing_private = requested_private.difference(private_ids)
        if missing_private:
            raise ValueError(f"requested private events are unavailable: {sorted(missing_private)}")
        private_ids = [event_id for event_id in private_ids if event_id in requested_private]
    if args.scope == "public_patient":
        private_ids = []
    descriptors = _private_descriptor_map(args.private_descriptors)

    public_reports = _jsonl(args.reports / "public_patient_reports.jsonl")
    public_patient_ids = {str(row["patient_id"]) for row in public_reports}
    if args.public_patient_id:
        requested_public = set(args.public_patient_id)
        missing_public = requested_public.difference(public_patient_ids)
        if missing_public:
            raise ValueError(f"requested public patients are unavailable: {sorted(missing_public)}")
        public_patient_ids.intersection_update(requested_public)
    if args.scope == "private_event":
        public_patient_ids.clear()
    public_source = _json(args.public_source)
    public_records = public_source.get("records")
    if not isinstance(public_records, list):
        raise TypeError("public typed-fact source has no records")
    records_by_patient: dict[str, list[dict[str, object]]] = {}
    for raw in public_records:
        if isinstance(raw, dict) and raw.get("report") is not None:
            records_by_patient.setdefault(str(raw["patient_id"]), []).append(raw)
    for rows in records_by_patient.values():
        rows.sort(key=lambda row: str(row["event_id"]))
    public_events = _public_event_map(_json(args.public_preflight))
    tusz_root = args.tusz_root.resolve(strict=True)

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    entries: list[dict[str, object]] = []
    unavailable: list[dict[str, str]] = []
    published = False
    try:
        private_config = CausalEDFConfig(reference_policy="unlabeled_common_car19")
        for ordinal, event_id in enumerate(private_ids, start=1):
            row = private_signal_rows.get(event_id)
            descriptor = descriptors.get(event_id)
            if row is None or descriptor is None:
                raise ValueError(f"private waveform source is incomplete: {event_id}")
            source = _safe_relative_edf(private_root, row["relative_edf_path"])
            loaded = load_standard19_edf_event(source, float(row["global_event_t0_sec"]), config=private_config)
            interval = _interval(descriptor["algorithmic_sustained_change"].get("support_interval_sec_relative_to_clinical_event_anchor"))
            relative = Path("private_event") / f"{event_id}.png"
            display = render_waveform_png(
                loaded.window.data,
                staging / relative,
                unit_id=event_id,
                event_id=event_id,
                evidence_interval=interval,
                representative_event=False,
            )
            binding = _signal_binding(loaded, staging / relative)
            entries.append({
                "scope": "private_event",
                "unit_id": event_id,
                "patient_id": row["patient_id"],
                "event_id": event_id,
                "figure_file": relative.as_posix(),
                "representative_event": False,
                "event_window_sec": [-12.0, 48.0],
                "sampling_rate_hz": 200.0,
                "filter_hz": [0.5, 45.0],
                "reference": "common_average_standard19",
                "channel_order": list(STANDARD_19),
                "event_anchor_offset_seconds": 12.0,
                "evidence_interval_sec": list(interval) if interval else None,
                **binding,
                **display,
            })
            if ordinal % 10 == 0 or ordinal == len(private_ids):
                print(f"private-waveform {ordinal}/{len(private_ids)}", flush=True)

        public_config = CausalEDFConfig(reference_policy="primary_ref")
        ordered_patients = sorted(public_patient_ids, key=lambda value: (len(value), value))
        for ordinal, patient_id in enumerate(ordered_patients, start=1):
            chosen: tuple[dict[str, object], dict[str, object], object, tuple[float, float] | None] | None = None
            reasons: list[str] = []
            for source_record in records_by_patient.get(patient_id, []):
                event_id = str(source_record["event_id"])
                event = public_events.get(event_id)
                if event is None:
                    reasons.append(f"{event_id}:not_in_preflight")
                    continue
                try:
                    source = _safe_relative_edf(tusz_root, event["relative_edf_path"])
                    loaded = load_standard19_edf_event(source, float(event["global_t0_sec"]), config=public_config)
                    phenotype = source_record.get("typed_facts", {}).get("event_phenotype", {})
                    interval = None
                    if isinstance(phenotype, Mapping):
                        start = phenotype.get("onset_start_sec")
                        stop = phenotype.get("onset_end_sec")
                        if isinstance(start, (int, float)) and isinstance(stop, (int, float)):
                            interval = _interval([
                                float(start) - float(event["global_t0_sec"]),
                                float(stop) - float(event["global_t0_sec"]),
                            ])
                    chosen = (source_record, event, loaded, interval)
                    break
                except (OSError, ValueError) as exc:
                    reasons.append(f"{event_id}:{type(exc).__name__}")
            if chosen is None:
                unavailable.append({
                    "scope": "public_patient",
                    "unit_id": patient_id,
                    "reason": "no_reportable_replay_event",
                    "detail": ";".join(reasons[:5]),
                })
            else:
                source_record, _, loaded, interval = chosen
                event_id = str(source_record["event_id"])
                relative = Path("public_patient") / f"{patient_id}.png"
                display = render_waveform_png(
                    loaded.window.data,
                    staging / relative,
                    unit_id=patient_id,
                    event_id=event_id,
                    evidence_interval=interval,
                    representative_event=True,
                )
                binding = _signal_binding(loaded, staging / relative)
                entries.append({
                    "scope": "public_patient",
                    "unit_id": patient_id,
                    "patient_id": patient_id,
                    "event_id": event_id,
                    "figure_file": relative.as_posix(),
                    "representative_event": True,
                    "selection_policy": "lexicographically_first_reportable_target_blind_event",
                    "event_window_sec": [-12.0, 48.0],
                    "sampling_rate_hz": 200.0,
                    "filter_hz": [0.5, 45.0],
                    "reference": "common_average_standard19",
                    "channel_order": list(STANDARD_19),
                    "event_anchor_offset_seconds": 12.0,
                    "evidence_interval_sec": list(interval) if interval else None,
                    **binding,
                    **display,
                })
            if ordinal % 10 == 0 or ordinal == len(ordered_patients):
                print(f"public-waveform {ordinal}/{len(ordered_patients)} available={sum(e['scope'] == 'public_patient' for e in entries)}", flush=True)

        manifest: dict[str, object] = {
            "schema_version": SCHEMA,
            "status": "completed_target_blind_processed_waveform_render",
            "entry_count": len(entries),
            "counts": {
                "private_event_figures": sum(entry["scope"] == "private_event" for entry in entries),
                "public_patient_figures": sum(entry["scope"] == "public_patient" for entry in entries),
                "unavailable": len(unavailable),
            },
            "entries": entries,
            "unavailable": unavailable,
            "access_receipt": {
                "raw_eeg_loaded_for_frozen_preprocessing_replay": True,
                "private_soz_targets_loaded": False,
                "deepsoz_targets_loaded": False,
                "evaluation_rows_loaded": False,
                "model_scores_or_rankings_used_in_figure": False,
                "training_calibration_or_model_selection_performed": False,
                "llm_used": False,
            },
            "evidence_binding": {
                "source_signal_sha256_recorded_per_figure": True,
                "preprocessing_receipt_sha256_recorded_per_figure": True,
                "processed_window_sha256_recorded_per_figure": True,
                "figure_sha256_recorded_per_figure": True,
                "channel_order_recorded_per_figure": True,
                "event_anchor_offset_recorded_per_figure": True,
                "report_fact_evidence_ids_bound": False,
            },
            "claim_boundary": {
                "yellow_interval_is_clinician_confirmed_onset": False,
                "waveform_is_independent_clinical_interpretation": False,
                "public_representative_event_explains_entire_patient_score": False,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--private-bundle", type=Path, default=DEFAULT_PRIVATE_BUNDLE)
    parser.add_argument("--private-evidence", type=Path, default=DEFAULT_PRIVATE_EVIDENCE)
    parser.add_argument("--private-descriptors", type=Path, default=DEFAULT_PRIVATE_DESCRIPTORS)
    parser.add_argument("--public-source", type=Path, default=DEFAULT_PUBLIC_SOURCE)
    parser.add_argument("--public-preflight", type=Path, default=DEFAULT_PUBLIC_PREFLIGHT)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scope",
        choices=("all", "private_event", "public_patient"),
        default="all",
        help="Optionally materialize only one report scope.",
    )
    parser.add_argument(
        "--private-event-id",
        action="append",
        help="Restrict private rendering to an exact event ID (repeatable).",
    )
    parser.add_argument(
        "--public-patient-id",
        action="append",
        help="Restrict public rendering to an exact patient ID (repeatable).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = materialize(args)
    print(json.dumps({"output": str(args.output), **result["counts"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
