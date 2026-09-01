"""Canonical named-axis transform for the common17 SeizureTransformer arm.

Axis position is model semantics.  The canonical ST16 order is obtained from
the upstream ST18 longitudinal-bipolar order by deleting only the two midline
derivations, never by slicing a model tensor.  This small primitive requires
explicit electrode names so a caller cannot silently reinterpret positions.
"""

from __future__ import annotations

from typing import Final, Sequence

import numpy as np


COMMON17_REFERENTIAL_AXIS_ORDER: Final[tuple[str, ...]] = (
    "FP1", "F3", "C3", "P3", "O1", "F7", "T7", "P7", "CZ",
    "FP2", "F4", "C4", "P4", "O2", "F8", "T8", "P8",
)
UPSTREAM_ST18_TYPED_UNITS: Final[tuple[str, ...]] = (
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FZ-CZ", "CZ-PZ",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
)
CANONICAL_ST16_TYPED_UNITS: Final[tuple[str, ...]] = tuple(
    unit for unit in UPSTREAM_ST18_TYPED_UNITS if unit not in {"FZ-CZ", "CZ-PZ"}
)
CANONICAL_ST16_PAIRS: Final[tuple[tuple[str, str], ...]] = tuple(
    tuple(unit.split("-", 1)) for unit in CANONICAL_ST16_TYPED_UNITS
)  # type: ignore[assignment]


def derive_st16_lb16_by_name(
    referential: object,
    *,
    electrode_order: Sequence[str],
) -> np.ndarray:
    """Return first-minus-second LB16 in the exact canonical axis order."""

    value = np.asarray(referential)
    order = tuple(str(item).strip().upper() for item in electrode_order)
    if value.ndim != 2 or value.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("common17 referential carrier must be float32/float64 [axis,sample]")
    if value.shape[0] != len(order) or value.shape[1] < 1:
        raise ValueError("referential carrier and explicit electrode order differ")
    if len(order) != len(set(order)):
        raise ValueError("explicit electrode order repeats an electrode")
    if set(order) != set(COMMON17_REFERENTIAL_AXIS_ORDER):
        missing = sorted(set(COMMON17_REFERENTIAL_AXIS_ORDER).difference(order))
        extra = sorted(set(order).difference(COMMON17_REFERENTIAL_AXIS_ORDER))
        raise ValueError(f"explicit carrier is not exact common17; missing={missing}, extra={extra}")
    if not bool(np.isfinite(value).all()):
        raise ValueError("common17 referential carrier contains nonfinite samples")
    index = {electrode: position for position, electrode in enumerate(order)}
    result = np.stack(
        [
            value[index[left]].astype(np.float64, copy=False)
            - value[index[right]].astype(np.float64, copy=False)
            for left, right in CANONICAL_ST16_PAIRS
        ],
        axis=0,
    )
    return np.ascontiguousarray(result, dtype=np.float64)


__all__ = [
    "CANONICAL_ST16_PAIRS",
    "CANONICAL_ST16_TYPED_UNITS",
    "COMMON17_REFERENTIAL_AXIS_ORDER",
    "UPSTREAM_ST18_TYPED_UNITS",
    "derive_st16_lb16_by_name",
]
