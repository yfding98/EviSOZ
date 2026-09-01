#!/usr/bin/env python3
"""Map heterogeneous doctor/TUSZ labels into canonical channel and region labels."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    from tfm_soz.constants import (
        REGION_NAMES as VOTE_REGION_NAMES,
        parse_soz_bipolar as parse_vote_soz_bipolar,
        region_endpoint_vote_ranking,
        region_endpoint_votes_from_soz_bipolar,
    )
except ImportError:  # pragma: no cover - repository package import fallback
    from code.tfm_soz.constants import (
        REGION_NAMES as VOTE_REGION_NAMES,
        parse_soz_bipolar as parse_vote_soz_bipolar,
        region_endpoint_vote_ranking,
        region_endpoint_votes_from_soz_bipolar,
    )

try:
    from soz_pre.constants import (
        CHANNEL_INPUT_MASK_COLUMNS,
        CHANNEL_LABEL_MASK_COLUMNS,
        CHANNEL_PROP_COLUMNS,
        CHANNEL_TO_REGIONS,
        REGION_LABEL_COLUMNS,
        REGION_MASK_COLUMNS,
        REGION_NAMES,
        REGION_PROP_COLUMNS,
        SPH_TO_REGION,
        TCP_CHANNELS,
        TCP_COLUMNS,
        TCP_PAIRS,
    )
    from soz_pre.utils import (
        clean_cell,
        expand_sph_electrodes,
        infer_regions_from_text,
        normalize_electrode_name,
        parse_bipolar_list,
    )
except ImportError:  # pragma: no cover - package import fallback
    from code.soz_pre.constants import (
        CHANNEL_INPUT_MASK_COLUMNS,
        CHANNEL_LABEL_MASK_COLUMNS,
        CHANNEL_PROP_COLUMNS,
        CHANNEL_TO_REGIONS,
        REGION_LABEL_COLUMNS,
        REGION_MASK_COLUMNS,
        REGION_NAMES,
        REGION_PROP_COLUMNS,
        SPH_TO_REGION,
        TCP_CHANNELS,
        TCP_COLUMNS,
        TCP_PAIRS,
    )
    from code.soz_pre.utils import (
        clean_cell,
        expand_sph_electrodes,
        infer_regions_from_text,
        normalize_electrode_name,
        parse_bipolar_list,
    )


def empty_channel_vector(fill: float = 0.0) -> np.ndarray:
    return np.full(len(TCP_CHANNELS), float(fill), dtype=np.float32)


def empty_region_vector(fill: float = 0.0) -> np.ndarray:
    return np.full(len(REGION_NAMES), float(fill), dtype=np.float32)


def channels_to_regions(channels: Iterable[str]) -> List[str]:
    regions: List[str] = []
    for channel in channels or []:
        for region in CHANNEL_TO_REGIONS.get(str(channel).upper().replace("_", "-"), tuple()):
            if region not in regions:
                regions.append(region)
    return regions


def electrodes_to_regions(
    electrodes: Iterable[str],
    *,
    onset_text: object = "",
    hemisphere: object = "",
) -> List[str]:
    regions: List[str] = []
    for electrode in electrodes or []:
        name = normalize_electrode_name(electrode)
        if name in ("", "DIFFUSE"):
            continue
        sph_region = SPH_TO_REGION.get(name)
        if sph_region and sph_region not in regions:
            regions.append(sph_region)
        expanded = expand_sph_electrodes([name])
        for channel, pair in zip(TCP_CHANNELS, TCP_PAIRS):
            if set(pair) & set(expanded):
                for region in CHANNEL_TO_REGIONS.get(channel, tuple()):
                    if region not in regions:
                        regions.append(region)
    if not regions:
        for region in infer_regions_from_text(onset_text, hemisphere):
            if region not in regions:
                regions.append(region)
    return regions


def _region_vectors(
    soz_regions: Iterable[str],
    propagation_regions: Iterable[str],
    mask_value: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = empty_region_vector()
    masks = empty_region_vector(mask_value)
    propagation = empty_region_vector()
    for region in soz_regions or []:
        if region in REGION_NAMES:
            labels[REGION_NAMES.index(region)] = 1.0
    for region in propagation_regions or []:
        if region in REGION_NAMES:
            propagation[REGION_NAMES.index(region)] = 1.0
    return labels, masks, propagation


def _is_uncertain_onset_text(value: object) -> bool:
    text = clean_cell(value)
    if not text:
        return False
    markers = (
        "\u8d77\u59cb\u4e0d\u6e05",  # 起始不清
        "\u53ef\u7591",  # 可疑
        "?",
        "\uff1f",  # ？
        "unclear",
        "uncertain",
    )
    return any(marker in text.lower() for marker in markers)


def _set_region_labels(
    labels: np.ndarray,
    masks: np.ndarray,
    regions: Iterable[str],
    weight: float,
) -> None:
    for region in regions or []:
        if region in REGION_NAMES:
            idx = REGION_NAMES.index(region)
            labels[idx] = 1.0
            masks[idx] = max(float(masks[idx]), float(weight))


def map_private_doctor_labels(
    significant_electrodes: Iterable[str],
    spread_electrodes: Iterable[str],
    *,
    onset_text: object = "",
    hemisphere: object = "",
    confidence: float = 1.0,
) -> Dict[str, object]:
    """Map doctor significant/spread electrodes into canonical vectors.

    Clear significant electrodes become strong SOZ positives. For uncertain or
    diffuse descriptions, significant/spread electrodes become weaker SOZ
    positives through fractional label masks. If no channel can be mapped, the
    event becomes a low-confidence global SOZ label instead of providing no
    spatial supervision.
    """

    significant = [normalize_electrode_name(item) for item in significant_electrodes or []]
    spread = [normalize_electrode_name(item) for item in spread_electrodes or []]
    diffuse_spread = "DIFFUSE" in spread
    uncertain_onset = _is_uncertain_onset_text(onset_text)
    soft_context = bool(diffuse_spread or uncertain_onset)
    sig_expanded = set(expand_sph_electrodes([item for item in significant if item != "DIFFUSE"]))
    spread_expanded = set(expand_sph_electrodes([item for item in spread if item != "DIFFUSE"]))
    has_significant = bool(sig_expanded)
    has_spread_channels = bool(spread_expanded)

    strong_both_weight = 1.0
    strong_single_weight = 0.75
    uncertain_significant_weight = 0.60
    uncertain_single_significant_weight = 0.55
    spread_weak_soz_weight = 0.35
    global_weak_soz_weight = 0.15

    labels = empty_channel_vector()
    label_masks = empty_channel_vector(0.0 if (soft_context or not has_significant) else 1.0)
    propagation = empty_channel_vector()
    confidences = empty_channel_vector(0.0)
    soz_bipolar: List[str] = []
    propagation_bipolar: List[str] = []
    quality_flags: List[str] = []

    if not has_significant:
        quality_flags.append("no_significant_electrodes")

    if diffuse_spread:
        quality_flags.append("diffuse_spread")
    if uncertain_onset:
        quality_flags.append("uncertain_onset_soft_soz")
    if soft_context:
        quality_flags.append("soft_private_label_policy")

    for idx, (channel, (anode, cathode)) in enumerate(zip(TCP_CHANNELS, TCP_PAIRS)):
        endpoints = {anode, cathode}
        sig_hits = endpoints & sig_expanded
        spread_hits = endpoints & spread_expanded
        spread_only = spread_hits - sig_hits

        if sig_hits:
            labels[idx] = 1.0
            if soft_context:
                weight = uncertain_significant_weight if len(sig_hits) == 2 else uncertain_single_significant_weight
            else:
                weight = strong_both_weight if len(sig_hits) == 2 else strong_single_weight
            label_masks[idx] = max(float(label_masks[idx]), float(weight))
            confidences[idx] = float(label_masks[idx])
            if channel not in soz_bipolar:
                soz_bipolar.append(channel)
            if spread_only:
                propagation[idx] = 1.0
                if channel not in propagation_bipolar:
                    propagation_bipolar.append(channel)
        elif spread_hits:
            propagation[idx] = 1.0
            if channel not in propagation_bipolar:
                propagation_bipolar.append(channel)
            if soft_context or not has_significant:
                labels[idx] = 1.0
                label_masks[idx] = max(float(label_masks[idx]), spread_weak_soz_weight)
                confidences[idx] = float(label_masks[idx])
                if channel not in soz_bipolar:
                    soz_bipolar.append(channel)
                quality_flags.append("spread_used_as_weak_soz")
            else:
                label_masks[idx] = 0.0
        elif diffuse_spread:
            label_masks[idx] = 0.0
            confidences[idx] = 0.0

    if not has_significant and not has_spread_channels:
        labels[:] = 1.0
        label_masks[:] = global_weak_soz_weight
        confidences[:] = global_weak_soz_weight
        soz_bipolar = list(TCP_CHANNELS)
        quality_flags.append("global_weak_soz_no_mappable_channels")
        if diffuse_spread:
            propagation[:] = 1.0
            propagation_bipolar = list(TCP_CHANNELS)

    soz_regions = (
        electrodes_to_regions(significant, onset_text=onset_text, hemisphere=hemisphere)
        if has_significant
        else []
    )
    spread_regions = electrodes_to_regions(spread, hemisphere=hemisphere) if has_spread_channels else []
    if not has_significant and not has_spread_channels:
        soz_regions = list(REGION_NAMES)
    elif not soz_regions and not has_spread_channels:
        soz_regions = infer_regions_from_text(onset_text, hemisphere)
    if diffuse_spread and not spread_regions:
        spread_regions = list(REGION_NAMES)

    if soft_context or not has_significant:
        region_labels = empty_region_vector()
        region_masks = empty_region_vector()
        region_prop = empty_region_vector()
        if not has_significant and not has_spread_channels:
            region_labels[:] = 1.0
            region_masks[:] = global_weak_soz_weight
            if diffuse_spread:
                region_prop[:] = 1.0
        else:
            significant_region_weight = uncertain_significant_weight if soft_context else spread_weak_soz_weight
            _set_region_labels(region_labels, region_masks, soz_regions, significant_region_weight)
            _set_region_labels(region_labels, region_masks, spread_regions, spread_weak_soz_weight)
            for region in spread_regions or []:
                if region in REGION_NAMES:
                    region_prop[REGION_NAMES.index(region)] = 1.0
    else:
        region_mask_value = 1.0 if soz_regions else 0.0
        region_labels, region_masks, region_prop = _region_vectors(soz_regions, spread_regions, region_mask_value)

    max_mask = float(max(float(label_masks.max()), float(region_masks.max())))
    label_region_names = [REGION_NAMES[idx] for idx, value in enumerate(region_labels.tolist()) if value > 0.5]
    propagation_region_names = [REGION_NAMES[idx] for idx, value in enumerate(region_prop.tolist()) if value > 0.5]

    return {
        "channel_labels": labels,
        "channel_label_masks": label_masks,
        "channel_confidences": confidences,
        "channel_propagation": propagation,
        "region_labels": region_labels,
        "region_masks": region_masks,
        "region_propagation": region_prop,
        "soz_bipolar": soz_bipolar,
        "propagation_bipolar": propagation_bipolar,
        "regions": label_region_names,
        "propagation_regions": propagation_region_names,
        "quality_flags": quality_flags,
        "label_confidence": 1.0 if max_mask > 0.0 else 0.0,
        "spatial_loss_weight": 1.0 if max_mask > 0.0 else 0.0,
        "max_label_mask": max_mask,
    }


def map_tusz_row(row: Dict[str, object], *, spatial_weight: float) -> Dict[str, object]:
    if tuple(REGION_NAMES) != tuple(VOTE_REGION_NAMES):
        raise ValueError("TUSZ endpoint-vote region order disagrees with canonical SOZ region order")

    labels = empty_channel_vector()
    label_masks = empty_channel_vector(1.0 if float(spatial_weight) > 0 else 0.0)
    input_masks = empty_channel_vector(1.0)
    explicit_soz = parse_vote_soz_bipolar(row.get("soz_bipolar", ""))
    explicit_set = set(explicit_soz)
    column_positive_set = set()
    for idx, col in enumerate(TCP_COLUMNS):
        column_positive = clean_cell(row.get(col)) not in ("", "0", "0.0") and float(row.get(col, 0) or 0) > 0
        if column_positive:
            column_positive_set.add(TCP_CHANNELS[idx])
        labels[idx] = 1.0 if (TCP_CHANNELS[idx] in explicit_set if explicit_soz else column_positive) else 0.0
        mask_value = row.get(f"mask_{col}", row.get(f"input_mask_{col}", 1.0))
        try:
            input_masks[idx] = 1.0 if float(mask_value) > 0 else 0.0
        except (TypeError, ValueError):
            input_masks[idx] = 1.0
    if explicit_soz and explicit_set != column_positive_set:
        raise ValueError("TUSZ soz_bipolar disagrees with its binary TCP channel-label columns")
    label_masks *= input_masks
    channel_list = [channel for channel, value in zip(TCP_CHANNELS, labels.tolist()) if value > 0.5]
    votes = region_endpoint_votes_from_soz_bipolar(",".join(channel_list)) if channel_list else [0] * len(REGION_NAMES)
    ranking = region_endpoint_vote_ranking(",".join(channel_list)) if channel_list else []
    max_vote = max(votes) if votes else 0
    regions = [region for region, vote in zip(REGION_NAMES, votes) if max_vote > 0 and vote == max_vote]
    region_labels, region_masks, region_prop = _region_vectors(
        regions,
        [],
        1.0 if float(spatial_weight) > 0 else 0.0,
    )
    return {
        "channel_labels": labels,
        "channel_label_masks": label_masks,
        "channel_input_masks": input_masks,
        "channel_propagation": empty_channel_vector(),
        "region_labels": region_labels,
        "region_masks": region_masks,
        "region_propagation": region_prop,
        "soz_bipolar": channel_list,
        "regions": regions,
        "soz_region": ranking[0] if ranking else "",
        "soz_region_ranking": ranking,
        "soz_region_votes": dict(zip(REGION_NAMES, votes)),
        "soz_region_top1_tied_regions": regions,
        "soz_region_source": "tusz_earliest_onset_plus_1s_endpoint_vote",
        "propagation_regions": [],
    }


def vectors_to_manifest_fields(mapped: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    channel_labels = np.asarray(mapped.get("channel_labels", empty_channel_vector()), dtype=np.float32)
    label_masks = np.asarray(mapped.get("channel_label_masks", empty_channel_vector(1.0)), dtype=np.float32)
    propagation = np.asarray(mapped.get("channel_propagation", empty_channel_vector()), dtype=np.float32)
    input_masks = np.asarray(mapped.get("channel_input_masks", empty_channel_vector(1.0)), dtype=np.float32)
    region_labels = np.asarray(mapped.get("region_labels", empty_region_vector()), dtype=np.float32)
    region_masks = np.asarray(mapped.get("region_masks", empty_region_vector()), dtype=np.float32)
    region_prop = np.asarray(mapped.get("region_propagation", empty_region_vector()), dtype=np.float32)

    for col, value in zip(TCP_COLUMNS, channel_labels.tolist()):
        out[col] = int(value > 0.5)
    for col, value in zip(CHANNEL_LABEL_MASK_COLUMNS, label_masks.tolist()):
        out[col] = float(value)
    for col, value in zip(CHANNEL_PROP_COLUMNS, propagation.tolist()):
        out[col] = int(value > 0.5)
    for col, value in zip(CHANNEL_INPUT_MASK_COLUMNS, input_masks.tolist()):
        out[col] = float(value)
    for col, value in zip(REGION_LABEL_COLUMNS, region_labels.tolist()):
        out[col] = int(value > 0.5)
    for col, value in zip(REGION_MASK_COLUMNS, region_masks.tolist()):
        out[col] = float(value)
    for col, value in zip(REGION_PROP_COLUMNS, region_prop.tolist()):
        out[col] = int(value > 0.5)
    return out


def labels_from_soz_bipolar(value: object) -> np.ndarray:
    labels = empty_channel_vector()
    for channel in parse_bipolar_list(value):
        labels[TCP_CHANNELS.index(channel)] = 1.0
    return labels
