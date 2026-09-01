#!/usr/bin/env python3
"""Release adjudicated TUSZ EEG-only S1 targets after the cohort gate passes.

The publisher is deliberately label-only.  It never opens EEG, LaBraM
features, model predictions, DeepSOZ targets, TUSZ involvement targets, or
private data.  ``unknown`` and ``unavailable`` electrodes remain masked;
known spread electrodes are serialized separately and never become SOZ
positives or training negatives.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_tusz_eeg_only_s1_label_release import (  # noqa: E402
    audit_cohort,
)
from scripts.build_tusz_eeg_only_s1_reader_pack import (  # noqa: E402
    ADJUDICATION_SCHEMA,
    COHORT_SIZES,
    PACK_SCHEMA,
    validate_completed_adjudication,
    validate_completed_annotation,
)
from src.soz.geometry import CHANNEL_INDEX, STANDARD_19  # noqa: E402


DEFAULT_READER_PACK = ROOT / "outputs/tusz_eeg_only_s1_reader_pack_v1_20260813"
DEFAULT_OUTPUT = ROOT / "outputs/tusz_eeg_only_s1_development_targets_v1_20260813"
SCHEMA_VERSION = "tusz_eeg_only_s1_adjudicated_patient_targets_v1"
STATUS = "released_cohort_specific_adjudicated_targets"
TENSOR_FILE = "targets.safetensors"


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"Expected JSONL object: {path}")
        case_id = str(value.get("case_id", "")).strip()
        if not case_id or case_id in result:
            raise ValueError(f"Empty or duplicated case ID: {path}")
        result[case_id] = value
    return result


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _patient_rows(root: Path, cohort: str) -> list[dict[str, str]]:
    with (root / "patient_linkage.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = [
            dict(row)
            for row in csv.DictReader(stream)
            if row.get("cohort") == cohort
        ]
    if len(rows) != COHORT_SIZES[cohort]:
        raise ValueError("S1 patient linkage cohort size changed")
    case_ids = [row.get("case_id", "") for row in rows]
    patient_ids = [row.get("patient_pseudonym", "") for row in rows]
    if (
        "" in case_ids
        or "" in patient_ids
        or len(set(case_ids)) != len(rows)
        or len(set(patient_ids)) != len(rows)
    ):
        raise ValueError("S1 patient linkage identity is empty or duplicated")
    return rows


def _target_row(
    row: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states = row.get("electrode_states")
    if not isinstance(states, Mapping) or set(states) != set(STANDARD_19):
        raise ValueError("Adjudication lacks the complete standard-19 state map")
    target = torch.zeros(len(STANDARD_19), dtype=torch.float32)
    mask = torch.zeros(len(STANDARD_19), dtype=torch.bool)
    spread = torch.zeros(len(STANDARD_19), dtype=torch.float32)
    spread_mask = torch.zeros(len(STANDARD_19), dtype=torch.bool)
    for channel in STANDARD_19:
        index = CHANNEL_INDEX[channel]
        state = states[channel]
        if state == "candidate_positive":
            target[index] = 1.0
            mask[index] = True
        elif state == "reviewed_not_candidate":
            mask[index] = True
        elif state not in {"unknown", "unavailable"}:
            raise ValueError(f"Invalid adjudicated electrode state: {state}")
    spread_values = row.get("known_spread_electrodes")
    if not isinstance(spread_values, list):
        raise TypeError("known_spread_electrodes must be an array")
    for channel in spread_values:
        if channel not in CHANNEL_INDEX:
            raise ValueError("Spread electrode is outside standard-19")
        spread[CHANNEL_INDEX[str(channel)]] = 1.0
        spread_mask[CHANNEL_INDEX[str(channel)]] = True
    if bool(((target == 1) & (spread == 1)).any()):
        raise ValueError("A channel cannot be both S1 candidate and known spread")
    # Even if a reader marked a known-spread electrode as reviewed rather
    # than unknown, it is not a valid SOZ-reference complement.  Keep it out
    # of the optimization mask and retain it only in the separate spread
    # carrier used for evaluation/reporting.
    mask[spread_mask] = False
    if not bool(((target == 1) & mask).any()):
        raise ValueError("An available target must contain an observed positive")
    return target, mask, spread, spread_mask


def materialize_targets(
    *,
    reader_pack: Path,
    cohort: str,
    output_directory: Path,
) -> tuple[Path, Mapping[str, object]]:
    """Validate and publish one cohort's adjudicated target carrier."""

    if cohort not in COHORT_SIZES:
        raise ValueError(f"Unknown S1 cohort: {cohort}")
    root = reader_pack.resolve(strict=True)
    pack = _read_json(root / "manifest.json")
    if pack.get("schema_version") != PACK_SCHEMA:
        raise ValueError("Unexpected S1 reader-pack schema")
    release = audit_cohort(root, cohort)
    if release.get("ready") is not True or release.get("status") != (
        "ready_for_cohort_specific_target_release"
    ):
        raise RuntimeError(
            "S1 cohort is not ready for target release: "
            f"{release.get('valid_completed_patient_count')}/"
            f"{release.get('expected_patient_count')} completed"
        )

    patients = _patient_rows(root, cohort)
    roster = _read_json(root / "event_case_roster.json")
    reader_a = _read_jsonl(root / f"reader_a_{cohort}.jsonl")
    reader_b = _read_jsonl(root / f"reader_b_{cohort}.jsonl")
    adjudication = _read_jsonl(root / f"adjudication_{cohort}.jsonl")
    expected_cases = {row["case_id"] for row in patients}
    if any(set(value) != expected_cases for value in (reader_a, reader_b, adjudication)):
        raise ValueError("S1 target-release roster changed")

    target_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    spread_rows: list[torch.Tensor] = []
    spread_mask_rows: list[torch.Tensor] = []
    released: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    patient_by_case = {row["case_id"]: row for row in patients}
    for case_id in sorted(expected_cases):
        left = reader_a[case_id]
        right = reader_b[case_id]
        final = adjudication[case_id]
        if final.get("schema_version") != ADJUDICATION_SCHEMA:
            raise ValueError("S1 adjudication schema changed")
        event_ids = set(str(value) for value in roster[case_id])
        validate_completed_annotation(left, event_ids)
        validate_completed_annotation(right, event_ids)
        validate_completed_adjudication(final, left, right)
        patient = patient_by_case[case_id]
        if final.get("target_availability") != "available":
            excluded.append(
                {
                    "case_id": case_id,
                    "patient_id": patient["patient_pseudonym"],
                    "target_availability": str(final.get("target_availability")),
                }
            )
            continue
        target, mask, spread, spread_mask = _target_row(final)
        target_rows.append(target)
        mask_rows.append(mask)
        spread_rows.append(spread)
        spread_mask_rows.append(spread_mask)
        released.append(
            {
                "row_index": len(released),
                "case_id": case_id,
                "patient_id": patient["patient_pseudonym"],
                "available_event_count": int(patient["available_event_count"]),
                "positive_electrodes": [
                    channel
                    for channel in STANDARD_19
                    if bool(target[CHANNEL_INDEX[channel]])
                ],
                "known_spread_electrodes": [
                    channel
                    for channel in STANDARD_19
                    if bool(spread[CHANNEL_INDEX[channel]])
                ],
                "observed_electrode_count": int(mask.sum()),
                "unknown_or_unavailable_electrode_count": int((~mask).sum()),
                "set_exhaustive": final.get("set_exhaustive"),
                "label_confidence": float(final["label_confidence"]),
                "patient_event_consistency": final.get(
                    "patient_event_consistency"
                ),
                "evidence_bases": list(final.get("evidence_bases", ())),
            }
        )
    if not released:
        raise RuntimeError("Released S1 cohort contains no available supervised target")

    tensors = {
        "targets": torch.stack(target_rows).contiguous(),
        "target_mask": torch.stack(mask_rows).contiguous(),
        "spread_targets": torch.stack(spread_rows).contiguous(),
        "spread_mask": torch.stack(spread_mask_rows).contiguous(),
    }
    target = output_directory.absolute()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target == root or target in root.parents or root in target.parents:
        raise ValueError("S1 target output overlaps the immutable reader pack")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    published = False
    try:
        save_file(tensors, str(staging / TENSOR_FILE))
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "cohort": cohort,
            "target_semantics": pack.get("target_semantics"),
            "candidate_channels": list(STANDARD_19),
            "expected_patient_count": len(patients),
            "completed_patient_count": int(
                release["valid_completed_patient_count"]
            ),
            "available_supervised_patient_count": len(released),
            "excluded_indeterminate_or_unavailable_patient_count": len(excluded),
            "patients": released,
            "excluded_patients": excluded,
            "tensor_file": TENSOR_FILE,
            "tensor_shapes": {
                name: list(value.shape) for name, value in tensors.items()
            },
            "tensor_dtypes": {
                name: str(value.dtype) for name, value in tensors.items()
            },
            "electrode_state_policy": {
                "candidate_positive": "target=1;mask=1",
                "reviewed_not_candidate": "target=0;mask=1;reference_complement_only",
                "unknown": "target=0;mask=0",
                "unavailable": "target=0;mask=0",
                "known_spread": "separate_tensor_only;target_mask=0;never_soz_positive_or_training_negative",
            },
            "access_receipt": {
                "reader_a_labels_loaded": True,
                "reader_b_labels_loaded": True,
                "later_third_reader_adjudication_loaded": True,
                "raw_eeg_loaded": False,
                "labram_features_loaded": False,
                "model_predictions_loaded": False,
                "deepsoz_targets_loaded": False,
                "tusz_involvement_targets_loaded": False,
                "private_eeg_loaded": False,
                "private_targets_loaded": False,
                "free_text_rationale_exported": False,
                "training_performed": False,
            },
        }
        (staging / "manifest.json").write_bytes(_canonical_bytes(manifest))
        os.replace(staging, target)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    return target, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--reader-pack", type=Path, default=DEFAULT_READER_PACK)
    parser.add_argument("--cohort", choices=tuple(COHORT_SIZES), default="s1_development")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path, manifest = materialize_targets(
        reader_pack=args.reader_pack,
        cohort=args.cohort,
        output_directory=args.output_directory,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "cohort": manifest["cohort"],
                "available_supervised_patient_count": manifest[
                    "available_supervised_patient_count"
                ],
                "path": str(path),
                "training_performed": False,
                "private_loaded": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
