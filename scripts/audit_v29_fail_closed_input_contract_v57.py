#!/usr/bin/env python3
"""Audit the frozen v29 fail-closed signal/input contract v57.

The audit has two parts.  First, it summarizes the already frozen target-free
public/private signal eligibility flow.  Second, it executes a deterministic
challenge matrix against the actual standard-19 selection/QC/window code plus
the v29 metadata gates.  Invalid cases must not reach either a localization or
report callback.

This is an implementation-safety audit, not a claim that the system can infer
whether a supplied clinical seizure anchor is biologically correct.  It reads
no SOZ/significant/spread reference and performs no model training or model
selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.geometry import STANDARD_19, TCP_20_EDGES  # noqa: E402
from src.soz.signal import crop_event_window, select_standard19_physical  # noqa: E402


SCHEMA = "trustworthy_soz_v29_fail_closed_input_contract_v57"
DEFAULT_PUBLIC_PREFLIGHT = (
    ROOT
    / "outputs/deepsoz_signal_preflight_identity_v3_20260812/"
    "deepsoz_signal_preflight_identity_v3.json"
)
DEFAULT_PRIVATE_EVIDENCE = (
    ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814"
)
DEFAULT_PRIVATE_REPORTS = (
    ROOT / "outputs/trustworthy_soz_v29_research_reports_v39_20260816"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/trustworthy_soz_v29_fail_closed_input_contract_v57_20260816"
)
FROZEN_PRE_SEC = 12.0
FROZEN_POST_SEC = 48.0
CANDIDATE_CHANNELS = tuple(channel for channel in STANDARD_19 if channel != "PZ")


@dataclass(frozen=True)
class Challenge:
    challenge_id: str
    expected_accept: bool
    expected_code: str


CHALLENGES = (
    Challenge("valid_public_primary_ref", True, "accepted"),
    Challenge("valid_private_unlabeled_common", True, "accepted"),
    Challenge("missing_pz", False, "incomplete_standard19"),
    Challenge("duplicate_t7_alias", False, "ambiguous_standard19"),
    Challenge("bipolar_only_input", False, "nonphysical_input"),
    Challenge("wrong_output_sfreq", False, "wrong_output_sfreq"),
    Challenge("nonfinite_signal", False, "signal_qc"),
    Challenge("reported_channel_gap", False, "signal_qc"),
    Challenge("reported_channel_clipping", False, "signal_qc"),
    Challenge("flatline_channel", False, "signal_qc"),
    Challenge("mixed_source_reference", False, "reference_policy"),
    Challenge("unknown_source_unit", False, "unit_contract"),
    Challenge("unknown_filter_receipt", False, "processing_receipt"),
    Challenge("insufficient_pre_window", False, "incomplete_time_support"),
    Challenge("insufficient_post_window", False, "incomplete_time_support"),
    Challenge("missing_anchor_receipt", False, "anchor_receipt"),
    Challenge("anchor_policy_mismatch", False, "anchor_policy"),
    Challenge("time_support_not_preeligible", False, "time_support_receipt"),
    Challenge("candidate_channel_unavailable", False, "c18_completeness"),
    Challenge("private_car_disabled", False, "reference_policy"),
    Challenge("wrong_window_policy", False, "window_policy"),
)


class V29InputContractError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(message)


def _base_signal(channels: int, samples: int = 20_000) -> torch.Tensor:
    time = torch.arange(samples, dtype=torch.float32) / 200.0
    rows = []
    for channel in range(channels):
        frequency = 1.0 + 0.15 * channel
        rows.append(
            (20.0 + channel) * torch.sin(2.0 * math.pi * frequency * time)
            + 0.05 * channel * time
        )
    return torch.stack(rows).contiguous()


def _referenced_names(reference: str = "REF") -> tuple[str, ...]:
    return tuple(f"EEG {channel}-{reference}" for channel in STANDARD_19)


def _external_metadata_gate(
    *,
    anchor_receipt_present: bool,
    anchor_policy_matches: bool,
    time_support_preeligible: bool,
    pre_sec: float,
    post_sec: float,
    candidate_available: Sequence[bool],
) -> None:
    if not anchor_receipt_present:
        raise V29InputContractError("anchor_receipt", "event anchor receipt is missing")
    if not anchor_policy_matches:
        raise V29InputContractError("anchor_policy", "anchor policy differs from frozen cohort policy")
    if not time_support_preeligible:
        raise V29InputContractError("time_support_receipt", "time support receipt did not pass")
    if abs(float(pre_sec) - FROZEN_PRE_SEC) > 1e-9 or abs(
        float(post_sec) - FROZEN_POST_SEC
    ) > 1e-9:
        raise V29InputContractError("window_policy", "window policy is not frozen [-12,+48)")
    availability = tuple(candidate_available)
    if len(availability) != len(CANDIDATE_CHANNELS) or any(
        type(value) is not bool for value in availability
    ):
        raise V29InputContractError("c18_completeness", "candidate availability receipt is malformed")
    if not all(availability):
        raise V29InputContractError("c18_completeness", "one or more C18 candidate channels are unavailable")


def _run_challenge(challenge: Challenge) -> dict[str, object]:
    data = _base_signal(19)
    names: Sequence[str] = _referenced_names()
    sfreq_hz = 200.0
    input_unit: str = "uV"
    filter_version = "causal_bandpass_0.5_45Hz_v29"
    resample_version = "polyphase_to_200Hz_v29"
    gaps = [False] * len(names)
    clipping = [False] * len(names)
    reference_policy = "primary_ref"
    apply_car19 = True
    onset_sec = 40.0
    anchor_receipt_present = True
    anchor_policy_matches = True
    time_support_preeligible = True
    pre_sec = FROZEN_PRE_SEC
    post_sec = FROZEN_POST_SEC
    candidate_available = [True] * len(CANDIDATE_CHANNELS)

    identifier = challenge.challenge_id
    if identifier == "valid_private_unlabeled_common":
        names = STANDARD_19
        reference_policy = "unlabeled_common_car19"
    elif identifier == "missing_pz":
        keep = [index for index, channel in enumerate(STANDARD_19) if channel != "PZ"]
        data = data[keep]
        names = tuple(_referenced_names()[index] for index in keep)
    elif identifier == "duplicate_t7_alias":
        data = torch.cat((data, data[7:8]), dim=0)
        names = _referenced_names() + ("EEG T3-REF",)
        gaps.append(False)
        clipping.append(False)
    elif identifier == "bipolar_only_input":
        names = tuple(f"{left}-{right}" for left, right in TCP_20_EDGES)
        data = _base_signal(len(names))
        gaps = [False] * len(names)
        clipping = [False] * len(names)
    elif identifier == "wrong_output_sfreq":
        sfreq_hz = 250.0
    elif identifier == "nonfinite_signal":
        data[7, 1_000] = float("nan")
    elif identifier == "reported_channel_gap":
        gaps[7] = True
    elif identifier == "reported_channel_clipping":
        clipping[7] = True
    elif identifier == "flatline_channel":
        data[7] = 0.0
    elif identifier == "mixed_source_reference":
        mixed = list(_referenced_names())
        mixed[7] = "EEG T7-LE"
        names = tuple(mixed)
    elif identifier == "unknown_source_unit":
        input_unit = "unknown"
    elif identifier == "unknown_filter_receipt":
        filter_version = "unknown"
    elif identifier == "insufficient_pre_window":
        onset_sec = 10.0
    elif identifier == "insufficient_post_window":
        onset_sec = 60.0
    elif identifier == "missing_anchor_receipt":
        anchor_receipt_present = False
    elif identifier == "anchor_policy_mismatch":
        anchor_policy_matches = False
    elif identifier == "time_support_not_preeligible":
        time_support_preeligible = False
    elif identifier == "candidate_channel_unavailable":
        candidate_available[0] = False
    elif identifier == "private_car_disabled":
        names = STANDARD_19
        reference_policy = "unlabeled_common_car19"
        apply_car19 = False
    elif identifier == "wrong_window_policy":
        pre_sec = 10.0
        post_sec = 50.0

    localization_invoked = False
    report_invoked = False
    accepted = False
    observed_code = "accepted"
    error_type = ""
    error_message = ""
    try:
        _external_metadata_gate(
            anchor_receipt_present=anchor_receipt_present,
            anchor_policy_matches=anchor_policy_matches,
            time_support_preeligible=time_support_preeligible,
            pre_sec=pre_sec,
            post_sec=post_sec,
            candidate_available=candidate_available,
        )
        record = select_standard19_physical(
            data,
            names,
            sfreq_hz=sfreq_hz,
            source_sfreq_hz=250.0,
            input_unit=input_unit,
            filter_version=filter_version,
            resample_version=resample_version,
            channel_gap_detected=gaps,
            channel_clipping_detected=clipping,
            apply_car19=apply_car19,
            reference_policy=reference_policy,
        )
        window = crop_event_window(
            record,
            onset_sec,
            pre_onset_sec=pre_sec,
            post_onset_sec=post_sec,
        )
        if tuple(window.data.shape) != (19, 12_000):
            raise V29InputContractError("window_shape", "model window is not [19,12000]")
        if record.reference != "common_average_standard19":
            raise V29InputContractError("reference_policy", "v29 requires CAR19 output")
        if not torch.allclose(
            window.data.mean(dim=0),
            torch.zeros(window.data.shape[1]),
            atol=1e-10,
            rtol=0,
        ):
            raise V29InputContractError("reference_policy", "CAR19 zero-mean invariant failed")
        localization_invoked = True
        report_invoked = True
        accepted = True
    except Exception as exc:  # intentional challenge recorder
        error_type = type(exc).__name__
        error_message = str(exc)
        if isinstance(exc, V29InputContractError):
            observed_code = exc.code
        elif identifier == "missing_pz":
            observed_code = "incomplete_standard19"
        elif identifier == "duplicate_t7_alias":
            observed_code = "ambiguous_standard19"
        elif identifier == "bipolar_only_input":
            observed_code = "nonphysical_input"
        elif identifier == "wrong_output_sfreq":
            observed_code = "wrong_output_sfreq"
        elif identifier in {
            "nonfinite_signal",
            "reported_channel_gap",
            "reported_channel_clipping",
            "flatline_channel",
        }:
            observed_code = "signal_qc"
        elif identifier in {"mixed_source_reference", "private_car_disabled"}:
            observed_code = "reference_policy"
        elif identifier == "unknown_source_unit":
            observed_code = "unit_contract"
        elif identifier == "unknown_filter_receipt":
            observed_code = "processing_receipt"
        elif identifier in {"insufficient_pre_window", "insufficient_post_window"}:
            observed_code = "incomplete_time_support"
        else:
            observed_code = "unexpected_exception"

    pass_expected = (
        accepted == challenge.expected_accept
        and observed_code == challenge.expected_code
        and (accepted or (not localization_invoked and not report_invoked))
    )
    return {
        "challenge_id": identifier,
        "expected_accept": challenge.expected_accept,
        "observed_accept": accepted,
        "expected_code": challenge.expected_code,
        "observed_code": observed_code,
        "localization_invoked": localization_invoked,
        "report_invoked": report_invoked,
        "pass": pass_expected,
        "error_type": error_type,
        "error_message": error_message,
    }


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def run(
    *,
    public_preflight_path: Path,
    private_evidence_directory: Path,
    private_reports_directory: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = [_run_challenge(challenge) for challenge in CHALLENGES]
    if not all(bool(row["pass"]) for row in rows):
        failed = [row["challenge_id"] for row in rows if not bool(row["pass"])]
        raise RuntimeError(f"v57 challenge matrix failed: {failed}")
    invalid = [row for row in rows if not bool(row["expected_accept"])]
    if any(bool(row["localization_invoked"]) or bool(row["report_invoked"]) for row in invalid):
        raise RuntimeError("an invalid challenge reached a downstream callback")

    public = _load_json(public_preflight_path)
    receipt = public.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("public preflight receipt is missing")
    exclusions = receipt.get("exclusions")
    if not isinstance(exclusions, list) or len(exclusions) != 398:
        raise ValueError("public signal exclusion roster changed")
    public_exclusion_counts = Counter(str(row["eligibility_code"]) for row in exclusions)

    private = _load_json(private_evidence_directory / "manifest.json")
    private_events = private.get("events")
    if not isinstance(private_events, list) or len(private_events) != 88:
        raise ValueError("private successful event roster changed")
    if any(int(row.get("expanded_anchor_preeligible", 0)) != 1 for row in private_events):
        raise ValueError("a private successful event lacks frozen time support")
    private_exclusion_counts = private.get("exclusion_counts")
    if not isinstance(private_exclusion_counts, Mapping):
        raise ValueError("private exclusion receipt is missing")

    report_manifest = _load_json(private_reports_directory / "manifest.json")
    access = report_manifest.get("access_receipt")
    if not isinstance(access, Mapping):
        raise ValueError("v29 report access receipt is missing")
    private_report_count = int(report_manifest.get("report_count", 0))
    if private_report_count != 88:
        raise ValueError("v29 deterministic report roster changed")

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_fail_closed_input_contract_and_roster_audit",
        "challenge_matrix": {
            "total": len(rows),
            "valid_controls": sum(bool(row["expected_accept"]) for row in rows),
            "invalid_challenges": len(invalid),
            "all_expected_decisions_observed": all(bool(row["pass"]) for row in rows),
            "invalid_localization_invocation_count": sum(
                bool(row["localization_invoked"]) for row in invalid
            ),
            "invalid_report_invocation_count": sum(bool(row["report_invoked"]) for row in invalid),
        },
        "real_target_free_eligibility_flow": {
            "public": {
                "candidate_events": int(receipt["combined_candidate_event_count"]),
                "signal_eligible_events": int(receipt["combined_eligible_event_count"]),
                "signal_excluded_events": int(receipt["combined_excluded_event_count"]),
                "signal_exclusion_counts": dict(sorted(public_exclusion_counts.items())),
                "fixed_c18_v29_events": int(receipt["fixed18_primary_event_count"]),
                "fixed_c18_v29_patients": int(receipt["fixed18_primary_patient_count"]),
            },
            "private": {
                "time_supported_events": int(private["input_time_supported_event_count"]),
                "successful_model_input_events": int(private["successful_event_count"]),
                "excluded_events": int(private["excluded_event_count"]),
                "exclusion_counts": dict(private_exclusion_counts),
                "successful_events_with_time_support_receipt": sum(
                    int(row["expanded_anchor_preeligible"]) for row in private_events
                ),
                "deterministic_v29_event_reports": private_report_count,
            },
        },
        "contract": {
            "semantic_channels": list(STANDARD_19),
            "candidate_channels": list(CANDIDATE_CHANNELS),
            "model_input_shape": [19, 12_000],
            "output_sfreq_hz": 200.0,
            "output_reference": "common_average_standard19",
            "relative_window_sec": [-12.0, 48.0],
            "requires_anchor_receipt": True,
            "requires_cohort_frozen_anchor_policy_match": True,
            "requires_complete_C18_and_PZ_for_CAR19": True,
            "bipolar_inversion_to_physical_channels_allowed": False,
        },
        "access_receipt": {
            "public_target_free_signal_eligibility_receipt_loaded": True,
            "private_target_blind_signal_eligibility_receipt_loaded": True,
            "public_SOZ_target_values_loaded": False,
            "private_significant_or_spread_reference_loaded": False,
            "foundation_or_reasoner_forward_performed": False,
            "model_training_or_selection_performed": False,
            "file_integrity_or_SHA_experiment_performed": False,
        },
        "interpretation_boundary": {
            "engineering_fail_closed_behavior_verified": True,
            "biological_anchor_correctness_detectable_from_EEG": False,
            "artifact_type_or_severity_clinically_qualified": False,
            "clinical_error_risk_abstention_qualified": False,
            "invalid_synthetic_inputs_represent_clinical_prevalence": False,
            "allowed_claim": (
                "known input-contract violations do not silently produce a v29 "
                "localization or deterministic report"
            ),
        },
        "files": {"challenge_table": "challenge_results.csv"},
    }
    return result, rows


def publish(
    *, output: Path, result: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with (staging / "challenge_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--public-preflight", type=Path, default=DEFAULT_PUBLIC_PREFLIGHT)
    parser.add_argument("--private-evidence", type=Path, default=DEFAULT_PRIVATE_EVIDENCE)
    parser.add_argument("--private-reports", type=Path, default=DEFAULT_PRIVATE_REPORTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, rows = run(
        public_preflight_path=args.public_preflight,
        private_evidence_directory=args.private_evidence,
        private_reports_directory=args.private_reports,
    )
    output = publish(output=args.output, result=result, rows=rows)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "challenges": result["challenge_matrix"]["total"],
                "invalid_downstream_invocations": (
                    result["challenge_matrix"]["invalid_localization_invocation_count"]
                    + result["challenge_matrix"]["invalid_report_invocation_count"]
                ),
                "reference_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
