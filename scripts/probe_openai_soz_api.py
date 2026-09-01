#!/usr/bin/env python3
"""CLI entry point for the no-EEG OpenAI API preflight."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.auto_annotate.openai_api_probe import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
