#!/usr/bin/env python3
"""Build a de-identified read-only web bundle from release-audited EEG reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_report_viewer import build_release_bundle  # noqa: E402


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_root(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("artifact root must be NAME=PATH")
    return name, Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--release-audit", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        action="append",
        type=_artifact_root,
        default=[],
        metavar="NAME=PATH",
        help=(
            "Report source root. Use default=PATH for direct coverage; for a "
            "combined overlay pass primary=PATH, recovery=PATH and, when used, "
            "remediation=PATH. May be repeated."
        ),
    )
    parser.add_argument("--full-root", type=Path, default=None)
    parser.add_argument("--primary-root", type=Path, default=None)
    parser.add_argument("--recovery-root", type=Path, default=None)
    parser.add_argument("--remediation-root", type=Path, default=None)
    parser.add_argument(
        "--doctor-label-bundle",
        type=Path,
        default=None,
        help=(
            "Optional private_postfreeze_doctor_label_release_bundle_v1 JSON. "
            "This frozen, PHI-free sidecar is opened only after all selected "
            "reports have been verified."
        ),
    )
    parser.add_argument(
        "--research-soz-sidecar-root",
        type=Path,
        default=None,
        help=(
            "Optional immutable private_long_recording_research_soz_sidecar_"
            "batch_v1_1 directory. Only validated EEG-only Top-k and "
            "descriptive evidence fields are projected into an independent "
            "viewer panel; the frozen qualified impression is unchanged."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots: dict[str, Path] = {}
    for name, path in args.artifact_root:
        if name in roots:
            raise SystemExit(f"duplicate --artifact-root name: {name}")
        roots[name] = path
    for name, path in (
        ("full", args.full_root),
        ("primary", args.primary_root),
        ("recovery", args.recovery_root),
        ("remediation", args.remediation_root),
    ):
        if path is None:
            continue
        if name in roots:
            raise SystemExit(f"duplicate artifact root name: {name}")
        roots[name] = path
    if not roots:
        raise SystemExit(
            "supply --full-root, explicit combined roots, or --artifact-root NAME=PATH"
        )
    result = build_release_bundle(
        coverage_manifest_path=args.coverage,
        release_audit_path=args.release_audit,
        artifact_roots=roots,
        doctor_label_bundle_path=args.doctor_label_bundle,
        research_soz_sidecar_root=args.research_soz_sidecar_root,
        output_root=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bundle_id": result["bundle_id"],
                "release_bundle_sha256": _sha256(
                    args.output / "release_bundle.json"
                ),
                **result["counts"],
                "raw_excel_text_included": False,
                "edf_annotations_included": False,
                "source_paths_included": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
