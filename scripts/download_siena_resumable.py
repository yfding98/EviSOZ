#!/usr/bin/env python3
"""Resume the official Siena EDF files without trusting file existence alone."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import time

from scripts.audit_siena_external_weak_region_cohort import _edf_storage_contract


DEFAULT_ROOT = Path("/mnt/hd1/dyf/dataset/SienaScalpEEG_v1.0.0")
DEFAULT_BASE_URL = "https://physionet-open.s3.amazonaws.com/siena-scalp-eeg/1.0.0/"


def _safe_record(value: str) -> str:
    relative = PurePosixPath(value.strip())
    if (
        not value.strip()
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.suffix.lower() != ".edf"
    ):
        raise ValueError(f"unsafe Siena RECORDS entry: {value!r}")
    return str(relative)


def _is_complete(path: Path) -> bool:
    try:
        _edf_storage_contract(path)
    except (FileNotFoundError, ValueError):
        return False
    return True


def _download_one(
    *,
    rtk: str,
    root: Path,
    base_urls: tuple[str, ...],
    relative: str,
    connection_seconds: int,
    max_attempts: int,
) -> tuple[str, int, int]:
    destination = root.joinpath(*PurePosixPath(relative).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _is_complete(destination):
        return relative, 0, int(destination.stat().st_size)
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        base_url = base_urls[(attempt - 1) % len(base_urls)]
        url = base_url.rstrip("/") + "/" + relative
        command = [
            rtk,
            "curl",
            "--continue-at",
            "-",
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "10",
            "--max-time",
            str(connection_seconds),
            "--output",
            str(destination),
            url,
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=connection_seconds + 30,
            )
            last_error = result.stderr.strip()
        except subprocess.TimeoutExpired:
            last_error = "local subprocess timeout"
        if _is_complete(destination):
            size = int(destination.stat().st_size)
            print(f"COMPLETE {relative} attempts={attempt} bytes={size}", flush=True)
            return relative, attempt, size
        size = int(destination.stat().st_size) if destination.exists() else 0
        print(
            f"RESUME {relative} attempt={attempt} bytes={size} "
            f"last_error={last_error[:160]!r}",
            flush=True,
        )
        time.sleep(0.25)
    raise RuntimeError(
        f"Siena download did not complete after {max_attempts} attempts: {relative}; "
        f"last_error={last_error!r}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--base-url",
        dest="base_urls",
        action="append",
        help="repeat to rotate equivalent official S3 endpoints across attempts",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--connection-seconds", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=500)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 32:
        raise ValueError("workers must be in [1, 32]")
    if args.connection_seconds < 5 or args.connection_seconds > 300:
        raise ValueError("connection-seconds must be in [5, 300]")
    root = args.root.resolve(strict=True)
    records_path = root / "RECORDS"
    records = [
        _safe_record(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 41 or len(records) != len(set(records)):
        raise ValueError("expected 41 unique Siena EDF entries in RECORDS")
    rtk = shutil.which("rtk")
    if rtk is None:
        raise RuntimeError("rtk is required for download commands")
    base_urls = tuple(args.base_urls or [DEFAULT_BASE_URL])
    if any(not value.startswith("https://") for value in base_urls):
        raise ValueError("every base URL must use HTTPS")
    pending = [relative for relative in records if not _is_complete(root / relative)]
    print(
        f"Siena EDF acquisition: complete={len(records) - len(pending)} "
        f"pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _download_one,
                rtk=rtk,
                root=root,
                base_urls=base_urls,
                relative=relative,
                connection_seconds=int(args.connection_seconds),
                max_attempts=int(args.max_attempts),
            )
            for relative in pending
        ]
        for future in as_completed(futures):
            future.result()
    incomplete = [relative for relative in records if not _is_complete(root / relative)]
    if incomplete:
        raise RuntimeError(f"Siena EDF storage audit still fails for: {incomplete}")
    total_bytes = sum(int((root / relative).stat().st_size) for relative in records)
    print(f"Siena EDF acquisition complete: records=41 bytes={total_bytes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
