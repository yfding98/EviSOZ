#!/usr/bin/env python3
"""Compatibility entry point for private clinical EEG annotation ledger v1.

The canonical CLI contract lives in
``materialize_private_clinical_eeg_annotations_v1.py``.  This wrapper remains
for callers of the earlier filename while sharing exactly the same source and
event-selection defaults.
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_private_clinical_eeg_annotations_v1 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
