#!/usr/bin/env python3
"""Read-only population, mapping, and integer-count audit for CAR17 SOZ OOF.

This audit does not retrain or mutate any frozen artifact.  It independently
recomputes the patient-level endpoints, verifies the raw EDF header support of
the oracle-event cohort, and closes the 107-to-102 target/population boundary.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import pyedflib
from safetensors import safe_open
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sota_soz.train_common17_oracle_event_oof_v1 import (  # noqa: E402
    induced_common17_neighbors,
    map_targets_to_common17,
)
from scripts.audit_common17_soz_raw_path_v1 import (  # noqa: E402
    COMMON17,
    load_common17_edf_event,
)
from src.soz.data.edf import CausalEDFConfig  # noqa: E402
from src.soz.geometry import STANDARD_19, normalize_electrode_name  # noqa: E402


SCHEMA = "clinical_eeg_common17_car17_oof_statistics_audit_v1"
DEFAULT_OOF = ROOT / "outputs/clinical_eeg_common17_oracle_event_oof_r3r2_20260824"
DEFAULT_PHASE = ROOT / "outputs/clinical_eeg_common17_car17_labram_phase_v1_20260824"
DEFAULT_TARGET = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_ROSTER = ROOT / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815/manifest.json"
DEFAULT_TARGET_CSV = ROOT / "outputs/deepsoz_target_v2_identity_recovery_20260812/patient_targets_v2.csv"
DEFAULT_EVENT_INPUTS = ROOT / "outputs/deepsoz_tusz_patient_splits_identity_v2_20260812/event_inputs.csv"
DEFAULT_AMENDMENT = ROOT / "outputs/labram_iv_signal_evidence_eligibility_amendment_v1_1_20260810/amendment.json"
DEFAULT_SIGNAL = ROOT / "outputs/deepsoz_signal_preflight_identity_v3_20260812/deepsoz_signal_preflight_identity_v3.json"
DEFAULT_OFFICIAL_AUDIT = ROOT / "outputs/deepsoz_official_local_oof_full.audit.json"
DEFAULT_TUSZ = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT = ROOT / "outputs/clinical_eeg_common17_car17_oof_statistics_audit_v1_20260824/receipt.json"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load(path: Path, keys: Iterable[str]) -> dict[str, torch.Tensor]:
    requested = tuple(keys)
    with safe_open(str(path.resolve(strict=True)), framework="pt", device="cpu") as source:
        missing = sorted(set(requested) - set(source.keys()))
        if missing:
            raise KeyError(f"Missing tensors in {path}: {missing}")
        return {key: source.get_tensor(key) for key in requested}


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if not 0 <= successes <= total or total < 1:
        raise ValueError("Invalid binomial count")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "successes": successes,
        "total": total,
        "rate": p,
        "wilson_95_ci": [center - half, center + half],
    }


def _acceptable(
    positive: torch.Tensor,
    graph: Sequence[Sequence[int]],
    *,
    enable_neighbors: bool,
) -> torch.Tensor:
    result = positive.clone()
    if enable_neighbors:
        for index in torch.nonzero(positive, as_tuple=False).flatten().tolist():
            if graph[index]:
                result[list(graph[index])] = True
    return result


def _metric_counts(
    probability: torch.Tensor,
    targets: torch.Tensor,
    pre_count: torch.Tensor,
    graph: Sequence[Sequence[int]],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if tuple(probability.shape) != tuple(targets.shape) or probability.shape[1] != 17:
        raise ValueError("Metric inputs must be aligned [P,17]")
    order = torch.argsort(probability, dim=1, descending=True, stable=True)
    top = order[:, 0]
    top_values = probability.gather(1, top[:, None]).squeeze(1)
    tied_top_count = (probability == top_values[:, None]).sum(dim=1)
    if bool((tied_top_count != 1).any()):
        raise RuntimeError("Integer-count audit requires a unique Top-1 for every patient")
    ranked = targets.gather(1, order)
    exact = ranked[:, 0].bool()
    hit3 = (ranked[:, :3].sum(dim=1) > 0)
    hit5 = (ranked[:, :5].sum(dim=1) > 0)
    first = ranked.argmax(dim=1)
    n2_rows = []
    n4_rows = []
    for row in range(len(probability)):
        positive = targets[row] == 1
        n2_rows.append(
            _acceptable(positive, graph, enable_neighbors=int(pre_count[row]) <= 2)[top[row]]
        )
        n4_rows.append(
            _acceptable(positive, graph, enable_neighbors=int(pre_count[row]) <= 4)[top[row]]
        )
    n2 = torch.stack(n2_rows).bool()
    n4 = torch.stack(n4_rows).bool()
    total = len(probability)
    result = {
        "exact_top1": _wilson(int(exact.sum()), total),
        "accuracy": _wilson(int(exact.sum()), total),
        "hit_at_3": _wilson(int(hit3.sum()), total),
        "hit_at_5": _wilson(int(hit5.sum()), total),
        "mrr": float((1.0 / (first.float() + 1.0)).mean()),
        "deepsoz_N2": {
            **_wilson(int(n2.sum()), total),
            "eligible_for_neighbor_expansion": int((pre_count <= 2).sum()),
        },
        "deepsoz_N4": {
            **_wilson(int(n4.sum()), total),
            "eligible_for_neighbor_expansion": int((pre_count <= 4).sum()),
        },
    }
    return result, {
        "top": top,
        "exact": exact,
        "hit3": hit3,
        "hit5": hit5,
        "n2": n2,
        "n4": n4,
    }


def _raw_header_support(
    phase_manifest: Mapping[str, Any], tusz_root: Path
) -> dict[str, Any]:
    rows = phase_manifest.get("scope", {}).get("event_roster")
    if not isinstance(rows, list) or len(rows) != 1_145:
        raise ValueError("Strict phase event roster is not the frozen 1,145-event cohort")
    root = tusz_root.resolve(strict=True)
    cache: dict[str, tuple[bool, bool, tuple[str, ...]]] = {}
    missing_events = 0
    missing_patients: set[str] = set()
    for row in rows:
        relative = str(row["relative_edf_path"])
        if relative not in cache:
            path = (root / relative).resolve(strict=True)
            path.relative_to(root)
            reader = pyedflib.EdfReader(str(path))
            try:
                canonical = tuple(
                    channel
                    for channel in (normalize_electrode_name(label) for label in reader.getSignalLabels())
                    if channel is not None
                )
            finally:
                reader.close()
            channel_set = set(canonical)
            if not set(COMMON17) <= channel_set:
                raise RuntimeError(f"Raw EDF lacks common17 support: {relative}")
            cache[relative] = ("FZ" in channel_set, "PZ" in channel_set, canonical)
        fz, pz, _ = cache[relative]
        if not (fz and pz):
            missing_events += 1
            missing_patients.add(str(row["patient_id"]))
    missing_records = sum(not (fz and pz) for fz, pz, _ in cache.values())
    return {
        "events": len(rows),
        "patients": len({str(row["patient_id"]) for row in rows}),
        "unique_edf_records": len(cache),
        "all_common17_present": True,
        "raw_FZ_or_PZ_missing_events": missing_events,
        "raw_FZ_or_PZ_missing_unique_edf_records": missing_records,
        "raw_FZ_or_PZ_missing_patients": len(missing_patients),
        "edf_sample_payload_read_by_this_header_audit": False,
        "edf_annotation_payload_read": False,
    }


def _population_boundary(
    *,
    patient_ids: Sequence[str],
    target_rows: Sequence[Mapping[str, str]],
    event_rows: Sequence[Mapping[str, str]],
    amendment: Mapping[str, Any],
    signal: Mapping[str, Any],
    tusz_root: Path,
) -> dict[str, Any]:
    roster = set(patient_ids)
    eligible = {
        str(row["deepsoz_patient_id"])
        for row in target_rows
        if row["eligible_for_localization"] == "1"
    }
    common17_complete = {
        str(row["deepsoz_patient_id"])
        for row in target_rows
        if row["eligible_for_localization"] == "1"
        and all(row[f"benchmark_mask_{channel}"] == "1" for channel in COMMON17)
    }
    if len(target_rows) != 124 or len(eligible) != 107 or len(common17_complete) != 106:
        raise RuntimeError("DeepSOZ target universe counts drifted")
    if not roster <= common17_complete or len(roster) != 102:
        raise RuntimeError("Frozen 102-patient roster is not fully common17-labelled")
    excluded = sorted(eligible - roster, key=int)
    if excluded != ["258", "906", "10088", "11321", "13407"]:
        raise RuntimeError("107-to-102 excluded patient set drifted")

    amendment_rows = {
        str(row["patient_id"]): row
        for row in amendment["receipt"]["excluded_source_train_patients"]
    }
    events_by_patient: dict[str, list[Mapping[str, str]]] = {}
    for row in event_rows:
        events_by_patient.setdefault(str(row["deepsoz_patient_id"]), []).append(row)
    excluded_rows = []
    for patient in excluded:
        target = next(row for row in target_rows if str(row["deepsoz_patient_id"]) == patient)
        if patient == "258":
            missing = [channel for channel in COMMON17 if target[f"benchmark_mask_{channel}"] != "1"]
            if missing != ["O2"]:
                raise RuntimeError("Patient 258 partial-label reason drifted")
            excluded_rows.append(
                {
                    "patient_id": patient,
                    "reason": "partial_common17_target_not_complete_label_exact_denominator",
                    "missing_target_channels": missing,
                    "not_caused_by_FZ_or_PZ_signal_removal": True,
                }
            )
            continue
        row = amendment_rows.get(patient)
        if row is None:
            raise RuntimeError(f"Missing signed signal-exclusion row for patient {patient}")
        codes = list(row["exclusion_codes"])
        excluded_rows.append(
            {
                "patient_id": patient,
                "reason": codes[0] if len(set(codes)) == 1 else "mixed_signal_exclusion",
                "event_count": len(events_by_patient.get(patient, [])),
                "exclusion_codes": codes,
                "not_caused_by_FZ_or_PZ_signal_removal": True,
            }
        )

    # The sole otherwise in-bounds/full19 excluded patient remains invalid on
    # retained channels, so common17 does not rescue it.
    candidate = [
        row
        for row in events_by_patient["10088"]
        if row["causal_warmup_30s_available"] == "1"
        and row["full_minus12_plus48_in_bounds"] == "1"
    ]
    if len(candidate) != 1:
        raise RuntimeError("Patient 10088 candidate event count drifted")
    config = CausalEDFConfig(**dict(signal["receipt"]["preprocess_config"]))
    config = replace(config, apply_car19=False)
    source = (tusz_root.resolve(strict=True) / candidate[0]["local_edf_path"]).resolve(strict=True)
    source.relative_to(tusz_root.resolve(strict=True))
    failure = None
    try:
        load_common17_edf_event(source, float(candidate[0]["t0_sec"]), config=config)
    except ValueError as error:
        failure = str(error)
    if failure is None or "clipping=['FP2', 'F7']" not in failure:
        raise RuntimeError("Patient 10088 common17 retained-channel QC result drifted")
    for row in excluded_rows:
        if row["patient_id"] == "10088":
            row["common17_direct_replay"] = {
                "status": "failed_retained_channel_raw_QC",
                "retained_clipping_channels": ["FP2", "F7"],
                "FZ_or_PZ_in_failure": False,
            }

    return {
        "original_target_patients": len(target_rows),
        "localization_eligible_target_patients": len(eligible),
        "complete_common17_target_patients": len(common17_complete),
        "final_complete_label_signal_cohort_patients": len(roster),
        "eligible_to_final_difference": len(excluded),
        "excluded_patients": excluded_rows,
        "label_complete_patient_excluded_only_for_FZ_or_PZ_input_absence": 0,
        "common17_expansion_required_for_current_oracle_event_roster": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    oof_manifest_path = args.oof / "manifest.json"
    oof_tensor_path = args.oof / "oof_predictions_and_states.safetensors"
    phase_manifest_path = args.phase / "manifest.json"
    target_tensor_path = args.target / "oof_predictions.safetensors"
    oof_manifest = _json(oof_manifest_path)
    phase_manifest = _json(phase_manifest_path)
    roster_manifest = _json(args.roster)
    patient_ids = [str(value) for value in roster_manifest["patient_ids"]]
    if len(patient_ids) != 102 or len(set(patient_ids)) != 102:
        raise RuntimeError("Patient roster is not 102 unique IDs")
    payload = _load(
        oof_tensor_path,
        (
            "targets",
            "target_mask",
            "pre_mapping_positive_count",
            "patient_folds",
            "common17_standard19_indices",
            "oof_probability.strict_car17_labram",
        ),
    )
    target19 = _load(target_tensor_path, ("targets", "target_mask", "patient_folds"))
    indices = payload["common17_standard19_indices"].long()
    expected_indices = torch.tensor(
        [STANDARD_19.index(channel) for channel in COMMON17], dtype=torch.long
    )
    if not torch.equal(indices, expected_indices):
        raise RuntimeError("Saved common17 index carrier drifted")
    mapped, mapped_mask, pre_count, mapping_stats = map_targets_to_common17(
        target19["targets"].float(), target19["target_mask"].bool(), indices
    )
    for expected, observed, name in (
        (mapped, payload["targets"].float(), "targets"),
        (mapped_mask, payload["target_mask"].bool(), "target_mask"),
        (pre_count, payload["pre_mapping_positive_count"].long(), "pre_count"),
        (target19["patient_folds"].long(), payload["patient_folds"].long(), "folds"),
    ):
        if not torch.equal(expected, observed):
            raise RuntimeError(f"OOF/target carrier mismatch: {name}")
    probability = payload["oof_probability.strict_car17_labram"].float()
    graph = induced_common17_neighbors(indices)
    metrics, rows = _metric_counts(probability, mapped, pre_count, graph)

    manifest_metrics = oof_manifest["metrics"]["strict_car17_labram"]
    comparisons = {
        "exact_top1": manifest_metrics["exact_top1_accuracy"],
        "accuracy": manifest_metrics["accuracy"],
        "hit_at_3": manifest_metrics["hit_at_3"],
        "hit_at_5": manifest_metrics["hit_at_5"],
        "mrr": manifest_metrics["mrr"],
        "deepsoz_N2": manifest_metrics["deepsoz_N2"]["relaxed_top1"],
        "deepsoz_N4": manifest_metrics["deepsoz_N4"]["relaxed_top1"],
    }
    recomputed = {
        "exact_top1": metrics["exact_top1"]["rate"],
        "accuracy": metrics["accuracy"]["rate"],
        "hit_at_3": metrics["hit_at_3"]["rate"],
        "hit_at_5": metrics["hit_at_5"]["rate"],
        "mrr": metrics["mrr"],
        "deepsoz_N2": metrics["deepsoz_N2"]["rate"],
        "deepsoz_N4": metrics["deepsoz_N4"]["rate"],
    }
    if any(abs(float(comparisons[key]) - float(recomputed[key])) > 1e-7 for key in comparisons):
        raise RuntimeError("Recomputed aggregate metrics do not match the frozen manifest")

    fold_rows = []
    folds = payload["patient_folds"].long()
    for fold in range(5):
        held = torch.nonzero(folds == fold, as_tuple=False).flatten()
        fold_metrics, _ = _metric_counts(
            probability.index_select(0, held),
            mapped.index_select(0, held),
            pre_count.index_select(0, held),
            graph,
        )
        fold_rows.append({"fold": fold, "metrics": fold_metrics})

    observed19 = (target19["targets"].float() == 1) & target19["target_mask"].bool()
    before = observed19.index_select(1, indices).float()
    _, before_rows = _metric_counts(probability, before, pre_count, graph)
    affected19 = observed19[:, STANDARD_19.index("FZ")] | observed19[:, STANDARD_19.index("PZ")]
    affected_indices = torch.nonzero(affected19, as_tuple=False).flatten().tolist()
    affected_rows = []
    for row in affected_indices:
        top = int(rows["top"][row])
        affected_rows.append(
            {
                "patient_id": patient_ids[row],
                "original_midline_positive_channels": [
                    channel
                    for channel in ("FZ", "PZ")
                    if bool(observed19[row, STANDARD_19.index(channel)])
                ],
                "preexisting_CZ_positive": bool(observed19[row, STANDARD_19.index("CZ")]),
                "predicted_top1_common17_channel": COMMON17[top],
                "before_mapping": {
                    "exact": bool(before_rows["exact"][row]),
                    "N2": bool(before_rows["n2"][row]),
                    "N4": bool(before_rows["n4"][row]),
                },
                "after_mapping": {
                    "exact": bool(rows["exact"][row]),
                    "N2": bool(rows["n2"][row]),
                    "N4": bool(rows["n4"][row]),
                },
            }
        )

    official_audit = _json(args.official_audit)
    raw_support = _raw_header_support(phase_manifest, args.tusz_root)
    population = _population_boundary(
        patient_ids=patient_ids,
        target_rows=_csv(args.target_csv),
        event_rows=_csv(args.event_inputs),
        amendment=_json(args.amendment),
        signal=_json(args.signal),
        tusz_root=args.tusz_root,
    )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "pass_read_only_independent_recomputation",
        "primary_arm": "strict_car17_labram",
        "evaluation_unit": "patient_held_out_OOF_patient",
        "oracle_event_not_end_to_end_detection": True,
        "metrics": metrics,
        "folds": fold_rows,
        "accuracy_alias_audit": {
            "accuracy_equals_exact_top1_accuracy": metrics["accuracy"] == metrics["exact_top1"],
            "frozen_values_equal": float(manifest_metrics["accuracy"])
            == float(manifest_metrics["exact_top1_accuracy"]),
            "interpretation": "accuracy is an alias of exact Top-1 positive-set membership, not an independent endpoint",
        },
        "target_mapping": {
            "policy": "observed FZ/PZ hard positives OR into CZ, then delete FZ/PZ",
            "statistics": mapping_stats,
            "affected_patient_count": len(affected_rows),
            "affected_patients": affected_rows,
            "aggregate_success_counts_before_after": {
                name: {
                    "before": int(before_rows[key].sum()),
                    "after": int(rows[key].sum()),
                    "delta": int(rows[key].sum()) - int(before_rows[key].sum()),
                    "denominator": len(probability),
                }
                for name, key in (("exact_top1", "exact"), ("deepsoz_N2", "n2"), ("deepsoz_N4", "n4"))
            },
            "mapping_before_positive_count_used_for_N2_N4_gate": True,
        },
        "population_boundary": population,
        "raw_oracle_event_edf_support": raw_support,
        "old_official_compatibility_replay_boundary": {
            "zero_filled_patient_count": official_audit["zero_filled_patient_count"],
            "zero_filled_patient_ids": sorted(
                official_audit["zero_filled_channels_by_patient"], key=int
            ),
            "same_as_current_oracle_event_raw_edf_roster": False,
            "current_oracle_event_raw_FZ_or_PZ_missing_patients": raw_support[
                "raw_FZ_or_PZ_missing_patients"
            ],
            "detector_canonical_long_record_missing_FZ_PZ_record_count": 249,
            "interpretation": "the five-patient zero-fill path and 249 long-record detector rows are separate populations from the 1,145-event SOZ oracle roster",
        },
        "lineage": {
            "oof_manifest": {"path": str(oof_manifest_path), "sha256": _sha256(oof_manifest_path)},
            "oof_tensor": {"path": str(oof_tensor_path), "sha256": _sha256(oof_tensor_path)},
            "phase_manifest": {"path": str(phase_manifest_path), "sha256": _sha256(phase_manifest_path)},
            "target_tensor": {"path": str(target_tensor_path), "sha256": _sha256(target_tensor_path)},
            "patient_roster": {"path": str(args.roster), "sha256": _sha256(args.roster)},
            "target_csv": {"path": str(args.target_csv), "sha256": _sha256(args.target_csv)},
            "event_inputs": {"path": str(args.event_inputs), "sha256": _sha256(args.event_inputs)},
            "amendment": {"path": str(args.amendment), "sha256": _sha256(args.amendment)},
            "official_audit": {"path": str(args.official_audit), "sha256": _sha256(args.official_audit)},
            "audit_script": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        },
        "access_receipt": {
            "private_data_loaded": False,
            "edf_annotations_loaded": False,
            "excel_or_doctor_text_loaded": False,
            "retraining_performed": False,
            "frozen_artifacts_modified": False,
            "raw_EEG_loaded_only_for_patient_10088_retained_channel_QC_replay": True,
        },
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--phase", type=Path, default=DEFAULT_PHASE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET_CSV)
    parser.add_argument("--event-inputs", type=Path, default=DEFAULT_EVENT_INPUTS)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--signal", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--official-audit", type=Path, default=DEFAULT_OFFICIAL_AUDIT)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = run(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output), "receipt_sha256": receipt["receipt_sha256"]}))


if __name__ == "__main__":
    main()
