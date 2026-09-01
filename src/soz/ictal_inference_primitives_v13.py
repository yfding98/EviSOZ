"""Target-neutral primitives for the sealed v13 LaBraM inference path.

This module deliberately has no dependency on training orchestration, target
adapters, DeepSOZ identity/crosswalk code, recovery bundles, or evaluation.
It contains only canonical public-patient roster handling and the byte-exact
ictal-head state hash required to verify an already-trained inference head.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from typing import Sequence

import numpy as np

from .models.concept_heads import IctalInvolvementHead


ICTAL_HEAD_STATE_HASH_SCHEMA = "soz_ictal_head_state_hash_v1"

_PATIENT_RE = re.compile(r"[a-z0-9]{8}")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Value is not canonical JSON data") from exc


def canonical_patient_roster(
    values: Sequence[object], *, field: str = "patient_roster"
) -> tuple[str, ...]:
    """Return a closed, sorted roster of canonical eight-character IDs."""

    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a patient sequence")
    roster = tuple(str(value).strip() for value in values)
    if (
        not roster
        or roster != tuple(sorted(roster))
        or len(set(roster)) != len(roster)
        or any(not _PATIENT_RE.fullmatch(value) for value in roster)
    ):
        raise ValueError(f"{field} must be canonical, sorted, and unique")
    return roster


def patient_roster_sha256(values: Sequence[object]) -> str:
    """Hash one canonical public-patient roster without target normalization."""

    return hashlib.sha256(
        _canonical_json_bytes(canonical_patient_roster(values))
    ).hexdigest()


def ictal_head_state_sha256(head: IctalInvolvementHead) -> str:
    """Hash a complete ictal head using the frozen v1 canonical framing.

    The implementation is byte-for-byte compatible with the historical v1
    training receipt, while keeping training and target modules outside the
    inference import graph.
    """

    if not isinstance(head, IctalInvolvementHead):
        raise TypeError("head must be IctalInvolvementHead")
    digest = hashlib.sha256()
    digest.update(ICTAL_HEAD_STATE_HASH_SCHEMA.encode("ascii") + b"\0")
    for name, tensor in sorted(head.state_dict().items()):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"Ictal head state contains non-finite values: {name}")
        native_big_endian = array.dtype.byteorder == ">" or (
            array.dtype.byteorder == "=" and sys.byteorder == "big"
        )
        if native_big_endian:
            array = array.byteswap().view(array.dtype.newbyteorder("<"))
        metadata = json.dumps(
            {
                "dtype": array.dtype.newbyteorder("<").str,
                "name": name,
                "shape": list(array.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        raw = array.tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


__all__ = (
    "ICTAL_HEAD_STATE_HASH_SCHEMA",
    "canonical_patient_roster",
    "ictal_head_state_sha256",
    "patient_roster_sha256",
)
