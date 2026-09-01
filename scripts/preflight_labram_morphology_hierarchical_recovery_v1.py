#!/usr/bin/env python3
"""Freeze and audit the one LaBraM-only morphology recovery candidate.

The command is deliberately CPU-only and performs no optimization. It reads
only the already materialized source-train TUEV plan, contextualized C-CAR19
tokens, native CE6 targets, masks, and overlap-component weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.soz.morphology_recovery import (  # noqa: E402
    MORPHOLOGY_RECOVERY_PROTOCOL_SHA256,
    audit_morphology_recovery_source,
    morphology_recovery_protocol_payload,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--parity-directory",
        type=Path,
        default=Path("outputs/preprocessing_parity_formal_v1_20260809"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "outputs/labram_morphology_hierarchical_recovery_preflight_v1_20260810"
        ),
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    args = _parser().parse_args()
    parity = args.parity_directory.resolve(strict=True)
    output = args.output_directory.resolve()
    paths = {
        "run_plan": parity / "run-plan.json",
        "tokens": parity / "arrays" / "tuev_tokens_C-CAR19.npy",
        "labels": parity / "arrays" / "tuev_labels.npy",
        "mask": parity / "arrays" / "tuev_mask.npy",
        "weights": parity / "arrays" / "tuev_weights.npy",
    }
    for path in paths.values():
        path.resolve(strict=True)
    run_plan = json.loads(paths["run_plan"].read_text(encoding="utf-8"))
    if not isinstance(run_plan, dict):
        raise TypeError("The preprocessing parity run plan must be a JSON object")
    receipt = audit_morphology_recovery_source(
        run_plan=run_plan,
        tokens=np.load(paths["tokens"], mmap_mode="r"),
        labels=np.load(paths["labels"], mmap_mode="r"),
        source_target_mask=np.load(paths["mask"], mmap_mode="r"),
        overlap_component_weights=np.load(paths["weights"], mmap_mode="r"),
    )
    protocol = morphology_recovery_protocol_payload()
    if receipt.protocol_sha256 != MORPHOLOGY_RECOVERY_PROTOCOL_SHA256:
        raise RuntimeError("Recovery preflight and frozen protocol disagree")
    _atomic_json(output / "protocol.json", protocol)
    payload = {
        **receipt.canonical_payload,
        "receipt_sha256": receipt.receipt_sha256,
        "source_files": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in sorted(paths.items())
        },
        "protocol_file": "protocol.json",
        "protocol_file_sha256": _sha256(output / "protocol.json"),
        "official_tuev_eval_opened_for_candidate": False,
        "gpu_used": False,
        "optimization_performed": False,
    }
    _atomic_json(output / "preflight.json", payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "items": receipt.item_count,
                "groups": receipt.group_count,
                "fold_group_counts": receipt.fold_group_counts,
                "observed_cells": receipt.observed_cell_count,
                "unknown_cells": receipt.unknown_cell_count,
                "formal_promotion": receipt.formal_promotion,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

