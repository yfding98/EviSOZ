#!/usr/bin/env python3
"""Custodian-side sealing of an already opened four-state reference payload."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.soz.label_fresh_confirmation import (
    canonical_sha256,
    load_json_object,
    validate_reference_bundle,
    write_new_canonical_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--prediction-seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]

    def absolute(path: Path) -> Path:
        return path if path.is_absolute() else workspace / path

    payload = load_json_object(absolute(args.payload))
    seal = load_json_object(absolute(args.prediction_seal))
    bundle = {
        "schema_version": "trustworthy_soz_four_state_reference_bundle_v1",
        "reference_payload": payload,
        "reference_payload_sha256": canonical_sha256(payload),
    }
    validate_reference_bundle(bundle, prediction_seal=seal)
    output = absolute(args.output)
    write_new_canonical_json(output, bundle)
    print("status=SEALED_FOUR_STATE_REFERENCE")
    print(f"reference_payload_sha256={bundle['reference_payload_sha256']}")
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
