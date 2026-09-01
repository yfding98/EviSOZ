#!/usr/bin/env python3
"""Thin entry point for corrected post-freeze LAN18 source-dev scoring."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from src.lookaroundnet_native18.cli import main


if __name__ == "__main__":
    main(["score-corrected-source-dev", *sys.argv[1:]])
