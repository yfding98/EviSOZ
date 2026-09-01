#!/usr/bin/env python3
"""Download the latest public OpenNeuro snapshot from its official API.

The downloader is intentionally small and dependency-free.  It enumerates the
official object listing, rejects unsafe keys, resumes partial files, validates
the advertised object sizes, and writes a machine-readable receipt.  It does
not infer or transform labels.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
from typing import Iterable, Sequence
from urllib.request import Request, urlopen


GRAPHQL_URL = "https://openneuro.org/crn/graphql"
RECEIPT_NAME = "openneuro_download_receipt.json"
RAW_SIGNAL_SUFFIXES = frozenset({".eeg", ".edf", ".bdf", ".set", ".fdt"})


@dataclass(frozen=True)
class ObjectRow:
    filename: str
    size: int
    url: str

    def relative_path(self) -> Path:
        relative = PurePosixPath(self.filename)
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe OpenNeuro object path: {self.filename!r}")
        return Path(*relative.parts)


def _dataset_id(value: str) -> str:
    text = str(value).strip()
    if len(text) != 8 or not text.startswith("ds") or not text[2:].isdigit():
        raise argparse.ArgumentTypeError("dataset id must look like ds003029")
    return text


def _list_latest_snapshot(
    dataset_id: str, *, timeout_sec: float
) -> tuple[str, int, tuple[ObjectRow, ...]]:
    query = (
        "query { dataset(id: \""
        + dataset_id
        + "\") { latestSnapshot { tag size files(recursive: true) "
        "{ filename size urls directory } } } }"
    )
    request = Request(
        GRAPHQL_URL,
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout_sec) as response:
        payload = json.load(response)
    if payload.get("errors"):
        raise RuntimeError(f"OpenNeuro GraphQL error: {payload['errors']}")
    try:
        snapshot = payload["data"]["dataset"]["latestSnapshot"]
        tag = str(snapshot["tag"])
        snapshot_size_value = snapshot["size"]
        files = snapshot["files"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("OpenNeuro latest-snapshot response is incomplete") from exc
    rows: list[ObjectRow] = []
    for item in files:
        if bool(item.get("directory")):
            continue
        urls = item.get("urls")
        if not isinstance(urls, list) or not urls or not isinstance(urls[0], str):
            raise RuntimeError(f"OpenNeuro file lacks a download URL: {item!r}")
        row = ObjectRow(
            filename=str(item["filename"]), size=int(item["size"]), url=urls[0]
        )
        row.relative_path()
        rows.append(row)
    ordered = tuple(sorted(rows, key=lambda row: row.filename))
    if not ordered or len({row.filename for row in ordered}) != len(ordered):
        raise RuntimeError("OpenNeuro object listing is empty or contains duplicates")
    # Newly published snapshots can temporarily expose a null aggregate size
    # even though every object already has an authoritative size.  Keep the
    # downloader usable in that state and make the receipt reproducible from
    # the pinned object listing.
    snapshot_size = (
        int(snapshot_size_value)
        if snapshot_size_value is not None
        else sum(row.size for row in ordered)
    )
    return tag, snapshot_size, ordered


def _select_objects(
    rows: Iterable[ObjectRow], *, metadata_only: bool
) -> tuple[ObjectRow, ...]:
    selected = []
    for row in rows:
        relative = row.relative_path()
        if metadata_only and relative.suffix.lower() in RAW_SIGNAL_SUFFIXES:
            continue
        selected.append(row)
    return tuple(selected)


def _download_one(
    row: ObjectRow,
    *,
    destination: Path,
    timeout_sec: float,
) -> tuple[str, str]:
    relative = row.relative_path()
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == row.size:
        return row.filename, "already_complete"
    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > row.size:
        partial.unlink()
        offset = 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = Request(row.url, headers=headers)
    with urlopen(request, timeout=timeout_sec) as response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            partial.unlink(missing_ok=True)
            offset = 0
            request = Request(row.url)
            with urlopen(request, timeout=timeout_sec) as restarted:
                with partial.open("wb") as stream:
                    shutil.copyfileobj(restarted, stream, length=1024 * 1024)
        else:
            with partial.open("ab" if offset else "wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
    actual = partial.stat().st_size
    if actual != row.size:
        raise IOError(
            f"Size mismatch for {row.filename}: expected {row.size}, got {actual}"
        )
    os.replace(partial, target)
    return row.filename, "downloaded"


def download(
    *,
    dataset_id: str,
    destination: Path,
    metadata_only: bool,
    workers: int,
    timeout_sec: float,
    reserve_gib: float,
) -> dict[str, object]:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("Destination cannot be a symlink")
    snapshot_tag, snapshot_size, listed = _list_latest_snapshot(
        dataset_id, timeout_sec=timeout_sec
    )
    selected = _select_objects(
        listed, metadata_only=metadata_only
    )
    required = sum(row.size for row in selected)
    existing = sum(
        row.size
        for row in selected
        if (destination / row.relative_path()).is_file()
        and (destination / row.relative_path()).stat().st_size == row.size
    )
    remaining = required - existing
    free = shutil.disk_usage(destination).free
    reserve = int(float(reserve_gib) * 1024**3)
    if remaining + reserve > free:
        raise OSError(
            f"Insufficient free space: remaining={remaining}, free={free}, reserve={reserve}"
        )

    started = time.time()
    counts = {"downloaded": 0, "already_complete": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _download_one,
                row,
                destination=destination,
                timeout_sec=timeout_sec,
            ): row
            for row in selected
        }
        completed_bytes = 0
        for position, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            _, state = future.result()
            counts[state] += 1
            completed_bytes += row.size
            if position % 25 == 0 or position == len(selected):
                print(
                    json.dumps(
                        {
                            "completed_objects": position,
                            "total_objects": len(selected),
                            "completed_bytes": completed_bytes,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    receipt = {
        "schema_version": "openneuro_latest_snapshot_download_v2",
        "dataset_id": dataset_id,
        "snapshot_tag": snapshot_tag,
        "snapshot_size_bytes": snapshot_size,
        "official_graphql_endpoint": GRAPHQL_URL,
        "destination": str(destination),
        "metadata_only": metadata_only,
        "listed_object_count": len(listed),
        "selected_object_count": len(selected),
        "selected_size_bytes": required,
        "downloaded_object_count": counts["downloaded"],
        "already_complete_object_count": counts["already_complete"],
        "all_selected_sizes_verified": True,
        "elapsed_sec": time.time() - started,
    }
    receipt_path = destination / RECEIPT_NAME
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, receipt_path)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("dataset_id", type=_dataset_id)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--reserve-gib", type=float, default=20.0)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be in [1,16]")
    if args.timeout_sec <= 0 or args.reserve_gib < 0:
        parser.error("timeout must be positive and reserve must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = download(
        dataset_id=args.dataset_id,
        destination=args.destination,
        metadata_only=args.metadata_only,
        workers=args.workers,
        timeout_sec=args.timeout_sec,
        reserve_gib=args.reserve_gib,
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
