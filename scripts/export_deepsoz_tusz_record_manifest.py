#!/usr/bin/env python3
"""Export the audited DeepSOZ--TUSZ recording manifest to the repository root.

The output has one row per DeepSOZ recording, but its localization target is
explicitly patient-level.  Consumers must therefore split and weight by
``deepsoz_patient_id`` instead of treating recordings as independent cases.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "outputs/deepsoz_tusz_adapted_manifest_20260803/source/TUH_manifest_final.csv"
)
DEFAULT_CROSSWALK = (
    ROOT
    / "outputs/deepsoz_tusz_patient_splits_identity_v2_20260812/record_crosswalk.csv"
)
DEFAULT_TARGETS = (
    ROOT
    / "outputs/deepsoz_target_v2_identity_recovery_20260812/patient_targets_v2.csv"
)
DEFAULT_EVENTS = (
    ROOT
    / "outputs/deepsoz_tusz_patient_splits_identity_v2_20260812/event_inputs.csv"
)
DEFAULT_SPLITS = (
    ROOT
    / "outputs/deepsoz_tusz_patient_splits_identity_v2_20260812/split_manifest.csv"
)
DEFAULT_BENCHMARK = (
    ROOT / "outputs/labram_identity_recovery_closed_replay_v16_20260812/manifest.json"
)
DEFAULT_OUTPUT = ROOT / "deepsoz_tusz_652_record_manifest.csv"
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")

C18 = (
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "P7",
    "P3",
    "P4",
    "P8",
    "O1",
    "O2",
)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path.resolve(strict=True), dtype=str, keep_default_na=False)


def _normal_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            numeric = float(text)
        except ValueError:
            return text
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    return text


def _require_columns(frame: pd.DataFrame, names: Iterable[str], label: str) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _benchmark_roster(path: Path) -> dict[str, int]:
    payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    public = payload.get("public", payload)
    patient_ids = [_normal_id(value) for value in public.get("patient_ids", [])]
    patient_folds = public.get("patient_folds", [])
    if len(patient_ids) != 102 or len(patient_folds) != 102:
        raise ValueError("Expected the frozen 102-patient Raw200/CPBF benchmark roster")
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("CPBF benchmark roster contains duplicate patients")
    return {patient: int(fold) for patient, fold in zip(patient_ids, patient_folds)}


def build_manifest(
    *,
    source_path: Path,
    crosswalk_path: Path,
    target_path: Path,
    event_path: Path,
    split_path: Path,
    benchmark_path: Path,
    tusz_root: Path,
) -> pd.DataFrame:
    source = _read_csv(source_path).reset_index(drop=True)
    crosswalk = _read_csv(crosswalk_path)
    targets = _read_csv(target_path)
    events = _read_csv(event_path)
    splits = _read_csv(split_path)

    _require_columns(
        source,
        ("pt_id", "fn", "nsz", "sz_starts", "sz_ends", "onset_zone", "hemi", "region", "Comments"),
        "DeepSOZ source",
    )
    _require_columns(
        crosswalk,
        (
            "deepsoz_row",
            "deepsoz_patient_id",
            "deepsoz_record",
            "mapping_status",
            "local_patient_id",
            "source_official_split",
            "local_edf_path",
            "local_csv_path",
            "local_csv_bi_path",
            "sfreq_hz",
            "edf_duration_sec",
            "full19_available",
        ),
        "identity-v2 crosswalk",
    )
    target_columns = [
        item
        for channel in C18
        for item in (
            f"benchmark_value_{channel}",
            f"benchmark_mask_{channel}",
            f"benchmark_state_{channel}",
        )
    ]
    _require_columns(
        targets,
        (
            "deepsoz_patient_id",
            "eligible_for_localization",
            "exclusion_reason",
            "source_record_count",
            "zero_semantics",
            "pz_policy",
            "benchmark_state_PZ",
            *target_columns,
        ),
        "DeepSOZ target-v2",
    )
    _require_columns(
        events,
        (
            "deepsoz_row",
            "event_id",
            "signal_input_eligible",
            "warmup_signal_input_eligible",
            "fnsz_signal_input_eligible",
            "fnsz_warmup_signal_input_eligible",
        ),
        "target-free TUSZ event inputs",
    )
    _require_columns(
        splits,
        ("deepsoz_patient_id", "model_split", "concept_oof_fold"),
        "patient split manifest",
    )

    if (len(source), len(crosswalk), len(targets), len(events), len(splits)) != (
        652,
        652,
        124,
        1812,
        124,
    ):
        raise ValueError(
            "Frozen DeepSOZ roster changed; expected 652 records, 124 patients, "
            "and 1,812 local events"
        )

    source["deepsoz_row"] = range(len(source))
    source["deepsoz_patient_id_source"] = source["pt_id"].map(_normal_id)
    crosswalk["deepsoz_row"] = pd.to_numeric(
        crosswalk["deepsoz_row"], errors="raise"
    ).astype(int)
    crosswalk["deepsoz_patient_id"] = crosswalk["deepsoz_patient_id"].map(_normal_id)
    targets["deepsoz_patient_id"] = targets["deepsoz_patient_id"].map(_normal_id)
    splits["deepsoz_patient_id"] = splits["deepsoz_patient_id"].map(_normal_id)
    events["deepsoz_row"] = pd.to_numeric(events["deepsoz_row"], errors="raise").astype(int)

    expected_rows = set(range(652))
    if set(crosswalk["deepsoz_row"]) != expected_rows or crosswalk["deepsoz_row"].duplicated().any():
        raise ValueError("Identity-v2 crosswalk does not cover every source row once")
    if set(crosswalk["mapping_status"]) != {"unique"}:
        raise ValueError("Identity-v2 crosswalk is not fully unique")
    if crosswalk["local_edf_path"].duplicated().any():
        raise ValueError("Identity-v2 crosswalk reuses a local EDF")
    if targets["deepsoz_patient_id"].duplicated().any() or splits["deepsoz_patient_id"].duplicated().any():
        raise ValueError("Patient target/split inputs must have one row per patient")

    joined = crosswalk.merge(
        source,
        on="deepsoz_row",
        how="left",
        validate="one_to_one",
        suffixes=("", "_source"),
    )
    if not (
        joined["deepsoz_patient_id"] == joined["deepsoz_patient_id_source"]
    ).all():
        raise ValueError("DeepSOZ source and identity crosswalk disagree on patient IDs")
    if not (joined["deepsoz_record"] == joined["fn"]).all():
        raise ValueError("DeepSOZ source and identity crosswalk disagree on record IDs")

    count_columns = (
        "signal_input_eligible",
        "warmup_signal_input_eligible",
        "fnsz_signal_input_eligible",
        "fnsz_warmup_signal_input_eligible",
    )
    for column in count_columns:
        events[column] = pd.to_numeric(events[column], errors="raise").astype(int)
    event_counts = events.groupby("deepsoz_row", sort=False).agg(
        local_event_count=("event_id", "size"),
        signal_input_event_count=("signal_input_eligible", "sum"),
        warmup_signal_input_event_count=("warmup_signal_input_eligible", "sum"),
        fnsz_signal_input_event_count=("fnsz_signal_input_eligible", "sum"),
        fnsz_warmup_signal_input_event_count=("fnsz_warmup_signal_input_eligible", "sum"),
    )
    joined = joined.merge(
        event_counts,
        on="deepsoz_row",
        how="left",
        validate="one_to_one",
    )
    event_count_columns = (
        "local_event_count",
        "signal_input_event_count",
        "warmup_signal_input_event_count",
        "fnsz_signal_input_event_count",
        "fnsz_warmup_signal_input_event_count",
    )
    joined[list(event_count_columns)] = joined[list(event_count_columns)].fillna(0).astype(int)

    joined = joined.merge(
        targets,
        on="deepsoz_patient_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_target"),
    ).merge(
        splits[["deepsoz_patient_id", "model_split", "concept_oof_fold"]],
        on="deepsoz_patient_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_split"),
    )
    if joined["eligible_for_localization"].eq("").all():
        raise ValueError("Patient targets failed to join")

    benchmark_folds = _benchmark_roster(benchmark_path)
    target_patients = set(targets["deepsoz_patient_id"])
    if not set(benchmark_folds).issubset(target_patients):
        raise ValueError("CPBF benchmark roster is not a subset of DeepSOZ targets")

    output = pd.DataFrame()
    output["manifest_schema"] = ["deepsoz-tusz-record-manifest-v1"] * len(joined)
    output["source_dataset"] = ["DeepSOZ-TUSZ-overlay"] * len(joined)
    output["deepsoz_row"] = joined["deepsoz_row"].astype(int)
    output["deepsoz_patient_id"] = joined["deepsoz_patient_id"]
    output["local_patient_id"] = joined["local_patient_id"]
    output["deepsoz_record"] = joined["deepsoz_record"]
    output["official_split"] = joined["source_official_split"]
    output["model_split"] = joined["model_split"]
    output["concept_oof_fold"] = joined["concept_oof_fold"]
    output["cpbf_benchmark_patient"] = joined["deepsoz_patient_id"].isin(benchmark_folds).astype(int)
    output["cpbf_outer_fold"] = joined["deepsoz_patient_id"].map(benchmark_folds).fillna("")
    output["mapping_status"] = joined["mapping_status"]
    output["mapping_policy"] = [
        "identity_v2_from_607_timeline_anchors_then_record_identity"
    ] * len(joined)
    output["local_edf_path"] = joined["local_edf_path"]
    output["local_edf_abs_path"] = joined["local_edf_path"].map(
        lambda value: str((tusz_root / value).resolve())
    )
    output["local_csv_path"] = joined["local_csv_path"]
    output["local_csv_bi_path"] = joined["local_csv_bi_path"]
    output["sampling_rate_hz"] = joined["sfreq_hz"]
    output["duration_sec"] = joined["edf_duration_sec"]
    output["full19_available"] = joined["full19_available"]
    output["n_seizure_events_source"] = joined["nsz"]
    output["local_event_count"] = joined["local_event_count"]
    output["signal_input_event_count"] = joined["signal_input_event_count"]
    output["warmup_signal_input_event_count"] = joined["warmup_signal_input_event_count"]
    output["fnsz_signal_input_event_count"] = joined["fnsz_signal_input_event_count"]
    output["fnsz_warmup_signal_input_event_count"] = joined[
        "fnsz_warmup_signal_input_event_count"
    ]
    output["seizure_starts_sec"] = joined["sz_starts"]
    output["seizure_ends_sec"] = joined["sz_ends"]
    output["source_onset_zone"] = joined["onset_zone"]
    output["source_hemisphere"] = joined["hemi"]
    output["source_region"] = joined["region"]
    output["source_comments"] = joined["Comments"]
    output["patient_target_eligible"] = joined["eligible_for_localization"]
    output["patient_exclusion_reason"] = joined["exclusion_reason"]
    output["target_granularity"] = ["patient_repeated_for_record_indexing_only"] * len(joined)
    output["patient_group_split_required"] = [1] * len(joined)
    output["patient_source_record_count"] = joined["source_record_count"]
    output["patient_equal_record_weight"] = joined["source_record_count"].map(
        lambda value: 1.0 / int(value)
    )
    eligible = pd.to_numeric(joined["eligible_for_localization"], errors="raise").astype(int)
    output["target_and_signal_event_count"] = joined["signal_input_event_count"] * eligible
    output["target_and_warmup_signal_event_count"] = (
        joined["warmup_signal_input_event_count"] * eligible
    )
    output["training_loss_contract"] = ["patient_equal_positive_set_mass_nll"] * len(joined)
    output["zero_semantics"] = joined["zero_semantics"]
    output["biological_negative_available"] = [0] * len(joined)
    output["pz_policy"] = joined["pz_policy"]
    output["benchmark_state_PZ"] = joined["benchmark_state_PZ"]
    output["positive_electrodes_c18"] = joined.apply(
        lambda row: ";".join(
            channel
            for channel in C18
            if row[f"benchmark_state_{channel}"] == "explicit_1"
        ),
        axis=1,
    )
    for channel in C18:
        output[f"benchmark_value_{channel}"] = joined[f"benchmark_value_{channel}"]
        output[f"benchmark_mask_{channel}"] = joined[f"benchmark_mask_{channel}"]
        output[f"benchmark_state_{channel}"] = joined[f"benchmark_state_{channel}"]

    output = output.sort_values("deepsoz_row", kind="stable").reset_index(drop=True)
    if len(output) != 652 or output["deepsoz_patient_id"].nunique() != 124:
        raise RuntimeError("Exported DeepSOZ manifest failed its final roster check")
    if int(output["cpbf_benchmark_patient"].groupby(output["deepsoz_patient_id"]).max().sum()) != 102:
        raise RuntimeError("Exported CPBF benchmark patient flag is inconsistent")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frame = build_manifest(
        source_path=args.source,
        crosswalk_path=args.crosswalk,
        target_path=args.targets,
        event_path=args.events,
        split_path=args.splits,
        benchmark_path=args.benchmark,
        tusz_root=args.tusz_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig", lineterminator="\n")
    summary = {
        "output": str(args.output.resolve()),
        "records": int(len(frame)),
        "patients": int(frame["deepsoz_patient_id"].nunique()),
        "record_splits": frame["official_split"].value_counts().sort_index().to_dict(),
        "cpbf_benchmark_patients": int(
            frame.groupby("deepsoz_patient_id")["cpbf_benchmark_patient"].max().sum()
        ),
        "cpbf_benchmark_records": int(frame["cpbf_benchmark_patient"].sum()),
        "full19_records": int(pd.to_numeric(frame["full19_available"]).sum()),
        "local_events": int(frame["local_event_count"].sum()),
        "target_and_signal_events": int(frame["target_and_signal_event_count"].sum()),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
