"""Frozen node/edge identities for the EviSOZ dual-montage contract.

Standard19 electrodes are physical node identities.  TCP22 rows are signed
bipolar derivations and remain edge identities.  The registry deliberately
contains no edge-to-endpoint label expansion rule.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping, Sequence

from src.soz.data.tuev import TUEV_OFFICIAL_TCP22
from src.soz.geometry import STANDARD_19, normalize_electrode_name

from .artifact_ref import canonical_json_sha256


CHANNEL_REGISTRY_SCHEMA_VERSION = "evisoz_channel_registry_v1"
TCP22_ORDER_SOURCE = "code/soz_pre/constants.py::TCP_PAIRS"
EDGE_FORMULA = "positive_minus_negative"

_LEGACY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "T7": ("T3", "T7"),
    "T8": ("T4", "T8"),
    "P7": ("T5", "P7"),
    "P8": ("T6", "P8"),
}
_AUXILIARY_ELECTRODES = ("A1", "A2", "SP1", "SP2", "SPHL", "SPHR")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_PREFIX = "EVISOZ-CHANNELS-"
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_HASH_PLACEHOLDER = "0" * 64
_TOP_KEYS = {
    "schema_version",
    "registry_id",
    "normalization_policy",
    "node_units",
    "auxiliary_electrodes",
    "tcp22_derivations",
    "tcp22_order_binding",
    "registry_sha256",
}


def _aliases_for(channel: str) -> list[str]:
    return list(_LEGACY_ALIASES.get(channel, (channel,)))


def _normalized_endpoint(value: str) -> str:
    return normalize_electrode_name(value)


def _tcp22_order_payload() -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "positive_original": positive,
            "negative_original": negative,
        }
        for index, (positive, negative) in enumerate(TUEV_OFFICIAL_TCP22)
    ]


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    source = deepcopy(dict(value))
    source["registry_id"] = _PENDING_ID
    source["registry_sha256"] = _HASH_PLACEHOLDER
    return source


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    source = deepcopy(dict(value))
    source["registry_sha256"] = _HASH_PLACEHOLDER
    return source


def build_default_channel_registry() -> dict[str, Any]:
    """Build the frozen Standard19-node/TCP22-edge registry."""

    node_units = [
        {
            "unit_id": f"NODE:{channel}",
            "channel_type": "electrode_node",
            "normalized_name": channel,
            "identity_aliases": _aliases_for(channel),
            "standard19_index": index,
            "v29_candidate": channel != "PZ",
        }
        for index, channel in enumerate(STANDARD_19)
    ]
    auxiliary = [
        {
            "unit_id": f"AUX:{channel}",
            "channel_type": "auxiliary_electrode",
            "normalized_name": channel,
            "identity_aliases": [channel],
            "standard19_index": None,
            "v29_candidate": False,
        }
        for channel in _AUXILIARY_ELECTRODES
    ]
    derivations = []
    for index, (positive, negative) in enumerate(TUEV_OFFICIAL_TCP22):
        positive_normalized = _normalized_endpoint(positive)
        negative_normalized = _normalized_endpoint(negative)
        original_name = f"{positive}-{negative}"
        normalized_name = f"{positive_normalized}-{negative_normalized}"
        derivations.append(
            {
                "unit_id": f"EDGE:TCP22:{index:02d}:{normalized_name}",
                "source_index": index,
                "channel_original": original_name,
                "channel_normalized": normalized_name,
                "channel_type": "bipolar_derivation_edge",
                "positive_electrode": {
                    "original": positive,
                    "normalized": positive_normalized,
                },
                "negative_electrode": {
                    "original": negative,
                    "normalized": negative_normalized,
                },
                "formula": EDGE_FORMULA,
                "node_projection_authorized": False,
            }
        )

    order_payload = _tcp22_order_payload()
    body: dict[str, Any] = {
        "schema_version": CHANNEL_REGISTRY_SCHEMA_VERSION,
        "registry_id": _PENDING_ID,
        "normalization_policy": {
            "policy_id": "identity_alias_only_no_spatial_substitution_v1",
            "legacy_aliases": {
                "T3": "T7",
                "T4": "T8",
                "T5": "P7",
                "T6": "P8",
            },
            "sp1_sp2_preserved_without_forced_mapping": True,
            "edge_and_node_coordinates_separate": True,
        },
        "node_units": node_units,
        "auxiliary_electrodes": auxiliary,
        "tcp22_derivations": derivations,
        "tcp22_order_binding": {
            "source": TCP22_ORDER_SOURCE,
            "ordered_pairs_sha256": canonical_json_sha256(order_payload),
            "edge_count": 22,
            "orientation": EDGE_FORMULA,
        },
        "registry_sha256": _HASH_PLACEHOLDER,
    }
    body["registry_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["registry_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_channel_registry(body)


def _require_closed(value: object, expected: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        unknown = sorted(set(value).difference(expected))
        raise ValueError(f"{context} fields drifted; missing={missing}, unknown={unknown}")
    return deepcopy(value)


def _unique_strings(value: object, context: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{context} must be an array")
    rows = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item or item != item.strip():
            raise TypeError(f"{context}[{index}] must be a non-empty trimmed string")
        rows.append(item)
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f"{context} must be non-empty and unique")
    return rows


def _validate_electrode_unit(
    value: object,
    *,
    expected_name: str,
    expected_index: int | None,
    auxiliary: bool,
    context: str,
) -> dict[str, Any]:
    data = _require_closed(
        value,
        {
            "unit_id",
            "channel_type",
            "normalized_name",
            "identity_aliases",
            "standard19_index",
            "v29_candidate",
        },
        context,
    )
    expected_type = "auxiliary_electrode" if auxiliary else "electrode_node"
    expected_id = ("AUX:" if auxiliary else "NODE:") + expected_name
    if data["unit_id"] != expected_id or data["channel_type"] != expected_type:
        raise ValueError(f"{context} identity or type drifted")
    if data["normalized_name"] != expected_name:
        raise ValueError(f"{context} normalized identity drifted")
    aliases = _unique_strings(data["identity_aliases"], f"{context}.identity_aliases")
    if set(aliases) != set(_aliases_for(expected_name)):
        raise ValueError(f"{context} aliases drifted")
    if data["standard19_index"] != expected_index:
        raise ValueError(f"{context} Standard19 index drifted")
    expected_candidate = not auxiliary and expected_name != "PZ"
    if data["v29_candidate"] is not expected_candidate:
        raise ValueError(f"{context} v29 candidate flag drifted")
    return data


def validate_channel_registry(value: object) -> dict[str, Any]:
    """Fail closed on any node/edge, alias, order or orientation drift."""

    data = _require_closed(value, _TOP_KEYS, "channel registry")
    if data["schema_version"] != CHANNEL_REGISTRY_SCHEMA_VERSION:
        raise ValueError("channel registry schema_version drifted")
    policy = _require_closed(
        data["normalization_policy"],
        {
            "policy_id",
            "legacy_aliases",
            "sp1_sp2_preserved_without_forced_mapping",
            "edge_and_node_coordinates_separate",
        },
        "normalization_policy",
    )
    if policy != {
        "policy_id": "identity_alias_only_no_spatial_substitution_v1",
        "legacy_aliases": {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"},
        "sp1_sp2_preserved_without_forced_mapping": True,
        "edge_and_node_coordinates_separate": True,
    }:
        raise ValueError("normalization policy drifted")

    nodes = data["node_units"]
    if not isinstance(nodes, list) or len(nodes) != len(STANDARD_19):
        raise ValueError("channel registry must contain exactly Standard19 nodes")
    for index, channel in enumerate(STANDARD_19):
        _validate_electrode_unit(
            nodes[index],
            expected_name=channel,
            expected_index=index,
            auxiliary=False,
            context=f"node_units[{index}]",
        )

    auxiliary = data["auxiliary_electrodes"]
    if not isinstance(auxiliary, list) or len(auxiliary) != len(_AUXILIARY_ELECTRODES):
        raise ValueError("auxiliary electrode inventory drifted")
    for index, channel in enumerate(_AUXILIARY_ELECTRODES):
        _validate_electrode_unit(
            auxiliary[index],
            expected_name=channel,
            expected_index=None,
            auxiliary=True,
            context=f"auxiliary_electrodes[{index}]",
        )

    derivations = data["tcp22_derivations"]
    if not isinstance(derivations, list) or len(derivations) != 22:
        raise ValueError("TCP22 derivation inventory must contain exactly 22 edges")
    edge_ids: list[str] = []
    for index, (positive, negative) in enumerate(TUEV_OFFICIAL_TCP22):
        row = _require_closed(
            derivations[index],
            {
                "unit_id",
                "source_index",
                "channel_original",
                "channel_normalized",
                "channel_type",
                "positive_electrode",
                "negative_electrode",
                "formula",
                "node_projection_authorized",
            },
            f"tcp22_derivations[{index}]",
        )
        pos_norm = _normalized_endpoint(positive)
        neg_norm = _normalized_endpoint(negative)
        expected_name = f"{pos_norm}-{neg_norm}"
        expected_id = f"EDGE:TCP22:{index:02d}:{expected_name}"
        if row["unit_id"] != expected_id or row["source_index"] != index:
            raise ValueError(f"TCP22 edge {index} identity/order drifted")
        if row["channel_original"] != f"{positive}-{negative}":
            raise ValueError(f"TCP22 edge {index} original name drifted")
        if row["channel_normalized"] != expected_name:
            raise ValueError(f"TCP22 edge {index} normalized name drifted")
        if row["channel_type"] != "bipolar_derivation_edge":
            raise ValueError(f"TCP22 edge {index} was converted to a node")
        if row["positive_electrode"] != {"original": positive, "normalized": pos_norm}:
            raise ValueError(f"TCP22 edge {index} positive endpoint drifted")
        if row["negative_electrode"] != {"original": negative, "normalized": neg_norm}:
            raise ValueError(f"TCP22 edge {index} negative endpoint drifted")
        if row["formula"] != EDGE_FORMULA or row["node_projection_authorized"] is not False:
            raise ValueError(f"TCP22 edge {index} orientation/projection policy drifted")
        edge_ids.append(row["unit_id"])
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("TCP22 edge IDs are not unique")

    binding = _require_closed(
        data["tcp22_order_binding"],
        {"source", "ordered_pairs_sha256", "edge_count", "orientation"},
        "tcp22_order_binding",
    )
    expected_order_sha = canonical_json_sha256(_tcp22_order_payload())
    if binding != {
        "source": TCP22_ORDER_SOURCE,
        "ordered_pairs_sha256": expected_order_sha,
        "edge_count": 22,
        "orientation": EDGE_FORMULA,
    }:
        raise ValueError("TCP22 source-order binding drifted")

    expected_id = _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["registry_id"] != expected_id:
        raise ValueError("channel registry_id does not bind its content")
    if not isinstance(data["registry_sha256"], str) or _SHA256_RE.fullmatch(data["registry_sha256"]) is None:
        raise ValueError("channel registry_sha256 is invalid")
    if data["registry_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("channel registry_sha256 does not bind its content")
    return data


def normalize_record_electrode(raw_name: object) -> dict[str, str]:
    """Preserve one record's original name while applying identity aliases."""

    original = str(raw_name).strip().upper()
    if not original:
        raise ValueError("record electrode name must be non-empty")
    normalized = normalize_electrode_name(original)
    return {
        "channel_original": original,
        "channel_normalized": normalized,
        "channel_type": (
            "electrode" if normalized in STANDARD_19 else "extra_electrode"
        ),
    }


__all__ = [
    "CHANNEL_REGISTRY_SCHEMA_VERSION",
    "TCP22_ORDER_SOURCE",
    "EDGE_FORMULA",
    "build_default_channel_registry",
    "validate_channel_registry",
    "normalize_record_electrode",
]
