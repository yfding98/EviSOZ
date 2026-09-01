#!/usr/bin/env python3
"""Create a candidate-only bundle with explicitly excluded reports removed.

This operates on already de-identified candidate files.  It never reads or
copies raw DOCX/EDF data.  The source candidate bundle remains untouched as an
audit record; the output is the only candidate root intended for downstream
preprocessing/release workflows.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.artifact_ref import canonical_json_sha256  # noqa: E402
from src.evisoz.data.private_report_exclusion import validate_private_report_exclusion  # noqa: E402
from src.evisoz.forge.private_report_deidentification import validate_private_report_deidentification_candidates  # noqa: E402


_HASH_PLACEHOLDER = "0" * 64
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_BUNDLE_PREFIX = "EVISOZ-DEIDSET-"


def _json(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"expected regular JSON file: {path}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"expected JSON object: {path}")
    return value


def _readdress(body: dict[str, Any]) -> dict[str, Any]:
    source = deepcopy(body)
    source["receipt_sha256"] = _HASH_PLACEHOLDER
    source["bundle_id"] = _PENDING_ID
    body["bundle_id"] = _BUNDLE_PREFIX + canonical_json_sha256(source)[:24]
    source = deepcopy(body)
    source["receipt_sha256"] = _HASH_PLACEHOLDER
    body["receipt_sha256"] = canonical_json_sha256(source)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exclusion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source_root = args.source_root.resolve(strict=True)
    source_manifest = _json(source_root / "manifest.json")
    exclusion = _json(args.exclusion)
    excluded = validate_private_report_exclusion(exclusion)["entries"]
    excluded_ids = {str(row["report_id"]) for row in excluded}
    candidates = validate_private_report_deidentification_candidates(
        source_manifest, output_root=source_root
    )
    selected = [row for row in candidates["candidates"] if row["report_id"] not in excluded_ids]
    if len(selected) + len(excluded_ids) != len(candidates["candidates"]):
        raise ValueError("exclusion roster contains a report not present in candidate bundle")
    if not selected:
        raise ValueError("active candidate bundle would be empty")
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for row in selected:
        relative = Path(*str(row["relative_text_path"]).split("/"))
        source = source_root / relative
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    body = deepcopy(candidates)
    body["candidates"] = sorted(selected, key=lambda row: str(row["candidate_id"]))
    role_counts = Counter(
        (row["association"]["split_assignment"] or {}).get("evisoz_role", "unresolved")
        for row in selected
    )
    route_counts = Counter(str(row["extraction"]["route"]) for row in selected)
    body["counts"] = {
        "candidate_count": len(selected),
        "automated_phi_scan_pass_count": len(selected),
        "split_role_candidate_counts": dict(sorted(role_counts.items())),
        "extraction_route_counts": dict(sorted(route_counts.items())),
        "manual_review_pass_count": 0,
        "development_qwen_training_release_count": 0,
        "locked_language_evaluation_release_count": 0,
    }
    body = _readdress(body)
    (output / "manifest.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_private_report_deidentification_candidates(body, output_root=output)
    (output / "README.md").write_text(
        "# Active private report candidate bundle\n\n"
        f"This bundle contains {len(selected)} de-identified candidates. "
        f"{len(excluded_ids)} unresolved reports are excluded by the separate operational quarantine receipt; "
        "they have no candidate file, linkage, split, preprocessing or release path.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "active_candidate_bundle_materialized", "output": str(output), "candidate_count": len(selected), "excluded_count": len(excluded_ids), "bundle_id": body["bundle_id"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
