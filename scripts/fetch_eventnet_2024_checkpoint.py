#!/usr/bin/env python3
"""Fetch or verify the exact EventNet 2024 release checkpoint.

The URL is commit-pinned and the file is accepted only at the adapter's
allowlisted size and SHA-256.  This script never deserializes the artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.eventnet_full_record_adapter import (  # noqa: E402
    EVENTNET_CHECKPOINT_SHA256,
    EVENTNET_CHECKPOINT_SIZE_BYTES,
    EVENTNET_UPSTREAM_COMMIT,
)


DEFAULT_DESTINATION = ROOT / "models/eventnet_2024_official/model.pth"
PINNED_URL = (
    "https://raw.githubusercontent.com/esl-epfl/eventnet_2024/"
    f"{EVENTNET_UPSTREAM_COMMIT}/eventnet/src/eventnet/model.pth"
)


def _verify(path: Path) -> tuple[int, str]:
    if path.is_symlink():
        raise ValueError("EventNet checkpoint destination must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("EventNet checkpoint destination is not a regular file")
    payload = resolved.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != EVENTNET_CHECKPOINT_SIZE_BYTES:
        raise ValueError("EventNet checkpoint size does not match the allowlist")
    if digest != EVENTNET_CHECKPOINT_SHA256:
        raise ValueError("EventNet checkpoint SHA-256 does not match the allowlist")
    return len(payload), digest


def _download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        PINNED_URL,
        headers={"User-Agent": "clinical-eeg-eventnet-artifact-fetch/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read(EVENTNET_CHECKPOINT_SIZE_BYTES + 1)
    if len(payload) != EVENTNET_CHECKPOINT_SIZE_BYTES:
        raise ValueError("downloaded EventNet checkpoint has an unexpected size")
    if hashlib.sha256(payload).hexdigest() != EVENTNET_CHECKPOINT_SHA256:
        raise ValueError("downloaded EventNet checkpoint failed SHA-256 verification")
    with tempfile.NamedTemporaryFile(
        "wb", dir=destination.parent, delete=False
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    destination = arguments.destination
    if not destination.is_absolute():
        destination = (ROOT / destination).resolve()
    if not destination.exists():
        if arguments.verify_only:
            raise FileNotFoundError(destination)
        _download(destination)
    size, digest = _verify(destination)
    print(
        f"verified EventNet checkpoint: path={destination} size={size} sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
