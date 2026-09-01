#!/usr/bin/env python3
"""Materialize a static SeizureTransformer third-party artifact receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.seizuretransformer_third_party_artifact_audit import (
    SEIZURETRANSFORMER_DEFAULT_ARTIFACT_PATH,
    audit_seizuretransformer_third_party_artifact,
    build_seizuretransformer_third_party_activation_gate,
    compare_artifact_header_to_pinned_public_source,
    seizuretransformer_native_preprocessing_contract,
)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-artifact",
        type=Path,
        default=SEIZURETRANSFORMER_DEFAULT_ARTIFACT_PATH,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-finite-payload-check",
        action="store_true",
        help="Diagnostic only; this forces the audit to remain unverified.",
    )
    args = parser.parse_args()

    audit = audit_seizuretransformer_third_party_artifact(
        args.model_artifact,
        verify_finite_payload=not args.skip_finite_payload_check,
    )
    compatibility = compare_artifact_header_to_pinned_public_source(
        args.model_artifact,
        public_source_root=ROOT / "third_party" / "SeizureTransformer",
        verify_cpu_state_dict_load=True,
    )
    gate = build_seizuretransformer_third_party_activation_gate(
        artifact_audit=audit,
    )
    bundle = {
        "schema_version": "seizuretransformer_accuracy_challenger_preflight_v1",
        "artifact_audit": audit,
        "public_source_compatibility": compatibility,
        "native_preprocessing_contract": (
            seizuretransformer_native_preprocessing_contract()
        ),
        "activation_gate": gate,
    }
    if args.output is not None:
        _write_json_atomic(args.output, bundle)
    print(json.dumps(bundle, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if audit.get("static_artifact_verified") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
