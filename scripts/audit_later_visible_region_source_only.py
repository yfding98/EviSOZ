#!/usr/bin/env python3
"""Audit deterministic later-visible edge-to-region mapping on public facts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.soz.later_visible_region_producer import (  # noqa: E402
    LATER_VISIBLE_REGION_PRODUCER_SCHEMA,
    produce_later_visible_region,
)


DEFAULT_SOURCE = ROOT / "outputs/event_phenotype_source_only_n64_20260811.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs/later_visible_region_source_only_n64_20260812.json"
)
OUTPUT_SCHEMA = "soz_later_visible_region_source_only_audit_v1"


def _load_source(path: Path) -> dict[str, object]:
    source = path.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError("Source phenotype artifact must be a canonical regular file")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Source phenotype artifact must contain a JSON object")
    if value.get("schema_version") != "soz_event_phenotype_source_only_audit_v1":
        raise ValueError("Expected the frozen source-only phenotype audit")
    if value.get("status") != "target_free_source_only_descriptive_audit":
        raise ValueError("Source artifact is not marked target-free/source-only")
    access = value.get("access_receipt")
    if not isinstance(access, Mapping):
        raise TypeError("Source artifact lacks an access receipt")
    forbidden_true = (
        "tusz_native_target_values_loaded",
        "deepsoz_target_values_loaded",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "training_performed",
        "threshold_selection_performed",
    )
    if any(access.get(name) is not False for name in forbidden_true):
        raise ValueError("Source artifact does not satisfy target-free access policy")
    return value


def audit(*, source: Path, output: Path) -> dict[str, object]:
    payload = _load_source(source)
    events = payload.get("events")
    if not isinstance(events, list):
        raise TypeError("Source phenotype artifact lacks event rows")
    target = output.absolute()
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)

    rows: list[dict[str, object]] = []
    region_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    mapped = 0
    cross_region = 0
    cross_laterality = 0
    for source_row in events:
        if not isinstance(source_row, Mapping):
            raise TypeError("Source event row must be an object")
        patient = str(source_row.get("patient_id", "")).strip()
        event = str(source_row.get("event_id", "")).strip()
        if not patient or not event:
            raise ValueError("Source event row lacks pseudonymous identity")
        phenotype = source_row.get("phenotype")
        if phenotype is None:
            later_derivations: tuple[str, ...] = ()
        elif isinstance(phenotype, Mapping):
            if phenotype.get("later_visible_region_zh") is not None:
                raise ValueError("Source event already contains an unbound region fact")
            raw_derivations = phenotype.get("later_visible_derivations")
            if not isinstance(raw_derivations, list) or any(
                not isinstance(value, str) for value in raw_derivations
            ):
                raise TypeError("Source later-visible derivations must be strings")
            later_derivations = tuple(raw_derivations)
        else:
            raise TypeError("Source phenotype must be an object or null")

        result = produce_later_visible_region(later_derivations)
        if result.status == "mapped":
            mapped += 1
            assert result.later_visible_region_zh is not None
            region_counts[result.later_visible_region_zh] = (
                region_counts.get(result.later_visible_region_zh, 0) + 1
            )
        else:
            for reason in result.reason_codes:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        cross_region += int(result.contains_cross_region_edge)
        cross_laterality += int(result.contains_cross_laterality_edge)
        rows.append(
            {
                "patient_id": patient,
                "event_id": event,
                "source_event_status": source_row.get("status"),
                "mapping": asdict(result),
            }
        )

    result_payload: dict[str, object] = {
        "schema_version": OUTPUT_SCHEMA,
        "producer_schema": LATER_VISIBLE_REGION_PRODUCER_SCHEMA,
        "status": "target_free_source_only_later_visible_region_audit",
        "access_receipt": {
            "source_event_count": len(rows),
            "source_fields_read": [
                "patient_id",
                "event_id",
                "status",
                "phenotype.later_visible_derivations",
                "phenotype.later_visible_region_zh_null_guard",
            ],
            "raw_eeg_loaded": False,
            "tusz_native_target_values_loaded": False,
            "deepsoz_target_values_loaded": False,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "propagation_labels_loaded": False,
            "localization_scores_loaded": False,
            "training_performed": False,
            "threshold_selection_performed": False,
        },
        "counts": {
            "events": len(rows),
            "mapped": mapped,
            "abstained": len(rows) - mapped,
            "contains_cross_region_edge": cross_region,
            "contains_cross_laterality_edge": cross_laterality,
            "regions": dict(sorted(region_counts.items())),
            "reason_codes": dict(sorted(reason_counts.items())),
        },
        "events": rows,
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                result_payload,
                stream,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return result_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(source=args.source, output=args.output)
    print(json.dumps({"output": str(args.output), "counts": result["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
