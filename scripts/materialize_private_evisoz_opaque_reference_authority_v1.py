#!/usr/bin/env python3
"""Freeze the private EviSOZ protocol-authorized opaque reference route."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.opaque_reference_authority import (  # noqa: E402
    build_private_opaque_reference_authority,
)


DEFAULT_AUDIT = ROOT / "outputs/evisoz_stage0_private_reference_audit_v1_20260831/audit.json"
DEFAULT_EVIDENCE = ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814/manifest.json"
DEFAULT_V29 = ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815/manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/evisoz_stage0_private_opaque_reference_authority_v1_20260831"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError("authority input must be a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--evidence-manifest", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--v29-manifest", type=Path, default=DEFAULT_V29)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    authority = build_private_opaque_reference_authority(
        _json(args.audit),
        _json(args.evidence_manifest),
        _json(args.v29_manifest),
        evidence_file_sha256=_sha256(args.evidence_manifest.resolve(strict=True)),
        v29_file_sha256=_sha256(args.v29_manifest.resolve(strict=True)),
    )
    args.output.mkdir(parents=True)
    (args.output / "authority.json").write_text(
        json.dumps(authority, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": authority["status"],
                "route_id": authority["route_id"],
                "authorized_source_edf_count": authority["source_inventory_binding"][
                    "authorized_source_edf_count"
                ],
                "receipt_sha256": authority["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
