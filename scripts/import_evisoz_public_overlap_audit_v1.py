#!/usr/bin/env python3
"""Import a dataset-authoritative public overlap audit receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.artifact_ref import canonical_json_bytes  # noqa: E402
from src.evisoz.data.public_overlap_audit import (  # noqa: E402
    validate_public_overlap_audit_receipt,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source_path = args.input.resolve(strict=True)
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError("public overlap input must be a regular file")
    receipt = validate_public_overlap_audit_receipt(
        json.loads(source_path.read_text(encoding="utf-8"))
    )
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(receipt) + b"\n")
    print(json.dumps({
        "status": receipt["status"],
        "audit_id": receipt["audit_id"],
        "missing_closure_codes": receipt["missing_closure_codes"],
        "training_authorized": receipt["permissions"]["training_authorized"],
        "output": str(output),
        "receipt_sha256": receipt["receipt_sha256"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
