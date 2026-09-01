#!/usr/bin/env python3
"""CLI entry point for bounded full-record SOZ detailed-review selection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.auto_annotate.select_full_record_review_queue import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
