#!/usr/bin/env python3
"""Broker only the identity rosters needed by the BUNDL training process.

This one-purpose materializer may read the historical V5 split, but exports
only three public-patient identity rosters and their receipts.  The training
runner must consume the resulting closed-schema sidecar and has no path or
argument through which it can open the richer source split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/tusz_ictal_formal_v5_auxiliary_split_20260810/split.json"
SOURCE_SHA256 = "cdcf56c1d1931ad22d18a0d91b4c5506edd8adb726c1f043cb2c701294a8bc21"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/labram_bundl_identity_firewall_v1_20260812/identity_firewall.json"
)
SCHEMA = "soz_labram_bundl_identity_firewall_v1"


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _roster_sha256(values: tuple[str, ...]) -> str:
    return hashlib.sha256(_canonical_json_bytes(list(values))).hexdigest()


def _identity_roster(
    payload: Mapping[str, object], field: str, expected_count: int
) -> tuple[str, ...]:
    raw = payload.get(field)
    if not isinstance(raw, list):
        raise TypeError(f"{field} must be a list")
    values = tuple(sorted(str(value).strip() for value in raw))
    if len(values) != expected_count or len(set(values)) != expected_count:
        raise ValueError(f"{field} must contain {expected_count} unique identities")
    if any(not value for value in values):
        raise ValueError(f"{field} contains an empty identity")
    return values


def materialize(source: Path, output: Path) -> dict[str, object]:
    source = source.resolve(strict=True)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Historical identity source SHA mismatch")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Historical identity source must be an object")
    if payload.get("schema_version") != "soz_ictal_formal_v5_auxiliary_split_v1":
        raise ValueError("Historical identity source schema changed")
    if (
        payload.get("deepsoz_soz_labels_used") is not False
        or payload.get("private_labels_used") is not False
        or payload.get("missing_tusz_cells_imputed_as_negative") is not False
    ):
        raise ValueError("Historical identity source violates public source isolation")

    deepsoz = _identity_roster(payload, "source_train_target_patient_ids", 65)
    i_dev = _identity_roster(payload, "i_dev_patient_ids", 12)
    i_gate = _identity_roster(payload, "i_gate_patient_ids", 12)
    rosters = (set(deepsoz), set(i_dev), set(i_gate))
    if any(
        left & right
        for index, left in enumerate(rosters)
        for right in rosters[index + 1 :]
    ):
        raise ValueError("Identity firewall rosters overlap")

    sidecar = {
        "schema_version": SCHEMA,
        "serialization": "canonical_json_utf8_no_pickle",
        "source_split_schema_version": payload["schema_version"],
        "source_split_sha256": SOURCE_SHA256,
        "deepsoz_master_overlap_public_patient_ids": list(deepsoz),
        "deepsoz_master_overlap_roster_sha256": _roster_sha256(deepsoz),
        "i_dev_public_patient_ids": list(i_dev),
        "i_dev_roster_sha256": _roster_sha256(i_dev),
        "i_gate_public_patient_ids": list(i_gate),
        "i_gate_roster_sha256": _roster_sha256(i_gate),
        "rosters_pairwise_disjoint": True,
        "deepsoz_soz_target_values_exported": False,
        "private_values_exported": False,
        "label_counts_prevalence_balance_exported": False,
    }
    target = Path(os.path.abspath(output))
    if target.name in {"", ".", ".."}:
        raise ValueError("Identity sidecar requires a concrete output filename")
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_bytes(_canonical_json_bytes(sidecar))
    return sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sidecar = materialize(args.source, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": hashlib.sha256(_canonical_json_bytes(sidecar)).hexdigest(),
                "deepsoz_identity_count": len(
                    sidecar["deepsoz_master_overlap_public_patient_ids"]
                ),
                "i_dev_identity_count": len(sidecar["i_dev_public_patient_ids"]),
                "i_gate_identity_count": len(sidecar["i_gate_public_patient_ids"]),
                "label_counts_prevalence_balance_exported": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
