"""Convert released EviSOZ fields into typed EEG loss targets.

The converter is intentionally conservative.  It only emits targets for
fields whose release explicitly enables the corresponding loss port, never
turns a teacher/derived candidate into a node label, and never maps TCP22
edges to Standard19 endpoints.  Missing, unknown, or text-only fields are
omitted rather than imputed.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from src.soz.geometry import STANDARD_19


SLOT_VOCABULARIES: dict[str, tuple[str, ...]] = {
    "quality": ("clean", "artifact_obscured", "unknown"),
    "morphology": (
        "attenuation", "LVFA", "rhythmic_theta", "rhythmic_delta",
        "rhythmic_alpha_beta", "spike_or_sharp_like", "evolving_rhythm",
        "artifact", "uncertain",
    ),
    "evolution": ("none", "frequency", "amplitude", "spatial"),
    "localizability": ("clear_focal", "probable_focal", "nonlocalizing"),
}


def _categorical_target(
    field: Mapping[str, object],
    *,
    slot: str,
) -> tuple[Tensor, Tensor]:
    payload = field.get("value_payload")
    if type(payload) is not dict or set(payload) != {"value", "certainty"}:
        raise ValueError(f"enabled {slot} target must be categorical")
    value = payload["value"]
    if not isinstance(value, str) or value not in SLOT_VOCABULARIES[slot]:
        raise ValueError(f"enabled {slot} target value is outside the frozen vocabulary")
    certainty = payload["certainty"]
    if certainty not in {"high", "medium", "low", "unknown"}:
        raise ValueError(f"enabled {slot} target certainty is invalid")
    # Low/unknown certainty is not silently promoted to supervision.  The
    # release may retain it for audit/reporting, but it must explicitly be
    # upgraded before opening a typed loss port.
    if certainty in {"low", "unknown"}:
        return torch.tensor(0, dtype=torch.long), torch.tensor(False, dtype=torch.bool)
    return (
        torch.tensor(SLOT_VOCABULARIES[slot].index(value), dtype=torch.long),
        torch.tensor(True, dtype=torch.bool),
    )


def _node_target(
    field: Mapping[str, object],
    *,
    observed_node_mask: Sequence[bool] | None,
) -> tuple[Tensor, Tensor]:
    payload = field.get("value_payload")
    if type(payload) is not dict or set(payload) != {"values", "semantics"}:
        raise ValueError("enabled node localization target must be a node-set payload")
    values = payload["values"]
    semantics = payload["semantics"]
    if not isinstance(values, list) or not values:
        raise ValueError("enabled node localization target has no positive nodes")
    if semantics not in {"exhaustive", "incomplete_positive", "unknown"}:
        raise ValueError("node localization target semantics are invalid")
    if any(type(item) is not str or item not in STANDARD_19 for item in values):
        raise ValueError("node localization target contains a non-Standard19 node")
    if len(values) != len(set(values)):
        raise ValueError("node localization target nodes must be unique")
    target = torch.zeros(len(STANDARD_19), dtype=torch.float32)
    for node in values:
        target[STANDARD_19.index(node)] = 1.0
    if semantics == "unknown":
        mask = torch.zeros(len(STANDARD_19), dtype=torch.bool)
    elif semantics == "incomplete_positive":
        # Only confirmed positives are evaluable; unspecified channels are
        # not negatives.
        mask = target.bool()
    else:
        mask = torch.ones(len(STANDARD_19), dtype=torch.bool)
    if observed_node_mask is not None:
        if len(observed_node_mask) != len(STANDARD_19):
            raise ValueError("observed_node_mask must contain Standard19 entries")
        mask &= torch.tensor(list(observed_node_mask), dtype=torch.bool)
    if not mask.any():
        raise ValueError("node localization target has no observed evaluable nodes")
    return target, mask


def build_typed_loss_targets(
    field_release: Mapping[str, object],
    *,
    observed_node_mask: Sequence[bool] | None = None,
) -> dict[str, dict[str, Tensor]]:
    """Build targets for the explicitly enabled ports in one field release.

    The returned tensors are batch-free (`[ ]` for categorical slots and
    `[19]` for node localization); a trainer adds the batch dimension after
    assembling records.  At most one released field may supervise a slot.
    """

    if not isinstance(field_release, Mapping):
        raise TypeError("field_release must be a mapping")
    fields = field_release.get("fields")
    if not isinstance(fields, list):
        raise ValueError("field_release.fields must be a list")
    outputs: dict[str, dict[str, Tensor]] = {}
    for field in fields:
        if type(field) is not dict:
            raise ValueError("field release entries must be objects")
        if field.get("state") != "provided":
            continue
        permissions = field.get("loss_permissions")
        if type(permissions) is not dict:
            raise ValueError("field loss_permissions are missing")
        if permissions.get("report_text_loss"):
            # Report text is a Qwen-side port and is intentionally not
            # returned to the EEG objective builder.
            continue
        if permissions.get("node_localization_loss"):
            if field.get("semantic_role") != "node_label" or field.get("authority") not in {"physician", "dataset_direct"}:
                raise ValueError("node localization loss requires a direct node-label field")
            if "node_localization" in outputs:
                raise ValueError("multiple node localization fields are ambiguous")
            target, mask = _node_target(field, observed_node_mask=observed_node_mask)
            outputs["node_localization"] = {"target": target, "mask": mask}
        if permissions.get("typed_slot_loss"):
            role = field.get("semantic_role")
            slot = {
                "quality": "quality",
                "morphology": "morphology",
                "evolution": "evolution",
                "localizability": "localizability",
            }.get(str(role))
            if slot is None:
                # A channel-set spread field cannot be coerced to a TCP22 edge
                # class and a region/laterality field has no decoder head.
                raise ValueError("typed slot loss field has no compatible decoder slot")
            if slot in outputs:
                raise ValueError(f"multiple typed fields supervise {slot}")
            target, mask = _categorical_target(field, slot=slot)
            outputs[slot] = {"target": target, "mask": mask}
    return outputs


def batch_typed_loss_targets(
    target_rows: Sequence[Mapping[str, Mapping[str, Tensor]]],
    *,
    device: torch.device | str,
) -> dict[str, dict[str, Tensor]]:
    """Stack sparse per-record targets without inventing missing labels.

    A slot is returned only if every row supplies a valid target.  This keeps
    the current `compute_typed_evidence_losses` contract simple and ensures a
    batch cannot turn absent fields into implicit negatives.
    """

    if not target_rows:
        raise ValueError("target_rows must be non-empty")
    result: dict[str, dict[str, Tensor]] = {}
    keys = sorted(set().union(*(row.keys() for row in target_rows)))
    for key in keys:
        if not all(key in row for row in target_rows):
            continue
        items = [row[key] for row in target_rows]
        if any(set(item) != {"target", "mask"} for item in items):
            raise ValueError(f"target row {key} fields drifted")
        target = torch.stack([item["target"] for item in items]).to(device)
        mask = torch.stack([item["mask"] for item in items]).to(device)
        result[key] = {"target": target, "mask": mask}
    return result


__all__ = [
    "SLOT_VOCABULARIES",
    "build_typed_loss_targets",
    "batch_typed_loss_targets",
]
