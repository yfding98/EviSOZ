#!/usr/bin/env python3
"""Freeze the support-only I-dev/I-gate split for formal ictal v5."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.tusz import load_tusz_ictal_involvement_target  # noqa: E402
from src.soz.data.tusz_training import (  # noqa: E402
    load_tusz_ictal_training_manifest,
    parse_tusz_official_train_path,
)
from src.soz.ictal_v5 import (  # noqa: E402
    IctalV5PatientSupport,
    freeze_v5_auxiliary_split,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest-bundle", type=Path, required=True)
    parser.add_argument("--deepsoz-split-csv", type=Path, required=True)
    parser.add_argument("--edf-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def _safe_new_output(value: Path) -> Path:
    target = Path(os.path.abspath(value))
    if target.name in {"", ".", ".."} or not target.parent.is_dir():
        raise ValueError("V5 split output requires an existing parent directory")
    if os.path.lexists(target):
        raise FileExistsError(f"V5 split output already exists: {target}")
    return target


def _source_train_local_patient_ids(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source", "local_patient_id", "model_split", "cohort_status"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError("DeepSOZ split CSV lacks required patient fields")
        patients = {
            str(row["local_patient_id"]).strip()
            for row in reader
            if str(row["source"]).strip() == "deepsoz_tusz_overlay"
            and str(row["model_split"]).strip() == "source_train"
            and str(row["cohort_status"]).strip() == "included_positive_only"
        }
    ordered = tuple(sorted(patients))
    if len(ordered) != 69:
        raise ValueError(
            f"Expected 69 pre-signal DeepSOZ source-train patients, observed {len(ordered)}"
        )
    return ordered


def _support_rows(manifest, edf_root: Path) -> tuple[IctalV5PatientSupport, ...]:
    state: dict[str, dict[str, int]] = {
        patient: {
            "event_count": 0,
            "observed_labels": 0,
            "positive_labels": 0,
            "explicit_negative_labels": 0,
        }
        for patient in manifest.patient_ids
    }
    for event in manifest.events:
        source = parse_tusz_official_train_path(edf_root, event.relative_edf_path)
        if source.patient_id != event.patient_id:
            raise ValueError("V5 support replay changed the TUSZ patient identity")
        target = load_tusz_ictal_involvement_target(
            source.channel_annotation_path,
            source.global_annotation_path,
            event_index=event.event_index,
            source_path=source.edf_path,
        )
        mask = target.source_target_mask
        values = target.targets[mask]
        observed = int(mask.sum().item())
        positive = int(values.sum().item())
        if observed != event.observed_label_count:
            raise ValueError(f"V5 support replay changed event {event.event_id}")
        row = state[event.patient_id]
        row["event_count"] += 1
        row["observed_labels"] += observed
        row["positive_labels"] += positive
        row["explicit_negative_labels"] += observed - positive
    return tuple(
        IctalV5PatientSupport(patient_id=patient, **state[patient])
        for patient in sorted(state)
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = _safe_new_output(args.output_directory)
    manifest = load_tusz_ictal_training_manifest(args.training_manifest_bundle)
    source_train_before_signal = _source_train_local_patient_ids(
        args.deepsoz_split_csv
    )
    source_train = tuple(
        sorted(set(source_train_before_signal) & set(manifest.patient_ids))
    )
    signal_unavailable = tuple(
        sorted(set(source_train_before_signal) - set(source_train))
    )
    if len(source_train) != 65 or len(signal_unavailable) != 4:
        raise ValueError(
            "V5 source-train signal intersection changed from 65 retained/4 unavailable"
        )
    payload = freeze_v5_auxiliary_split(
        _support_rows(manifest, args.edf_root),
        source_train_target_patient_ids=source_train,
    )
    payload["source_train_pre_signal_patient_count"] = len(
        source_train_before_signal
    )
    payload["source_train_signal_unavailable_patient_ids"] = list(
        signal_unavailable
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        split_path = staging / "split.json"
        split_path.write_text(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        os.rename(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    print(
        json.dumps(
            {
                "path": str(target),
                "i_dev_patient_count": len(payload["i_dev_patient_ids"]),
                "i_gate_patient_count": len(payload["i_gate_patient_ids"]),
                "balance": payload["balance"],
                "deepsoz_soz_labels_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
