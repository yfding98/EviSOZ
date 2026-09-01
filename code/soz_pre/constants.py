#!/usr/bin/env python3
"""Canonical constants for the heterogeneous SOZ pretraining pipeline.

The canonical 22-channel order follows the TUSZ TCP montage order used by
``code.data_preprocess.config`` and the TUSZ DeepSOZ preprocessing scripts:
left temporal, right temporal, central, left parasagittal, right parasagittal.
All readers/writers in ``code.soz_pre`` align labels by column name, not by
source array position.
"""

from __future__ import annotations

from typing import Dict, Tuple


TCP_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("A1", "T3"), ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"),
    ("C4", "T4"), ("T4", "A2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
)

TCP_CHANNELS: Tuple[str, ...] = tuple(f"{a}-{b}" for a, b in TCP_PAIRS)
TCP_COLUMNS: Tuple[str, ...] = tuple(ch.replace("-", "_") for ch in TCP_CHANNELS)
TCP_INDEX: Dict[str, int] = {ch: idx for idx, ch in enumerate(TCP_CHANNELS)}
TCP_COLUMN_INDEX: Dict[str, int] = {col: idx for idx, col in enumerate(TCP_COLUMNS)}

EXTRA_INPUT_ELECTRODES: Tuple[str, ...] = ("SPHL", "SPHR")
INPUT_CHANNELS_WITH_SPH: Tuple[str, ...] = TCP_CHANNELS + EXTRA_INPUT_ELECTRODES

REGION_NAMES: Tuple[str, ...] = (
    "left_frontal",
    "right_frontal",
    "left_temporal",
    "right_temporal",
    "central_parietal",
)

REGION_TO_CHANNELS: Dict[str, Tuple[str, ...]] = {
    "left_frontal": ("FP1-F7", "FP1-F3"),
    "right_frontal": ("FP2-F8", "FP2-F4"),
    "left_temporal": ("F7-T3", "T3-T5", "T5-O1", "A1-T3", "T3-C3"),
    "right_temporal": ("F8-T4", "T4-T6", "T6-O2", "C4-T4", "T4-A2"),
    "central_parietal": (
        "C3-CZ", "CZ-C4", "F3-C3", "C3-P3", "P3-O1",
        "F4-C4", "C4-P4", "P4-O2",
    ),
}

CHANNEL_TO_REGIONS: Dict[str, Tuple[str, ...]] = {}
for _region, _channels in REGION_TO_CHANNELS.items():
    for _channel in _channels:
        CHANNEL_TO_REGIONS.setdefault(_channel, tuple())
        CHANNEL_TO_REGIONS[_channel] = CHANNEL_TO_REGIONS[_channel] + (_region,)

SPH_TO_REGION = {
    "SPHL": "left_temporal",
    "SPHR": "right_temporal",
    "SP1": "left_temporal",
    "SP2": "right_temporal",
}

SPH_TO_NEIGHBOR_ELECTRODES = {
    "SPHL": ("A1", "F7", "T3"),
    "SPHR": ("A2", "F8", "T4"),
}

HEMISPHERE_CLASSES: Tuple[str, ...] = ("L", "R", "B", "M")
HEMISPHERE_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(HEMISPHERE_CLASSES)}

FOCAL_SEIZURE_TYPES = {"fnsz", "cpsz", "spsz"}
SPATIAL_SEIZURE_TYPE_WEIGHT = {
    "fnsz": 1.0,
    "cpsz": 0.8,
    "spsz": 0.8,
    "gnsz": 0.0,
    "absz": 0.0,
    "tcsz": 0.0,
    "tnsz": 0.0,
    "mysz": 0.0,
    "nesz": 0.0,
    "bckg": 0.0,
}

MANIFEST_BASE_FIELDS: Tuple[str, ...] = (
    "source",
    "split",
    "patient_id",
    "base_patient_id",
    "edf_path",
    "event_id",
    "event_index",
    "duration_sec",
    "t_event_marker",
    "t_eeg_onset",
    "t_end",
    "sz_start",
    "sz_end",
    "sz_duration",
    "seizure_type",
    "hemisphere",
    "hemisphere_label",
    "label_source",
    "label_type",
    "label_confidence",
    "spatial_loss_weight",
    "raw_label_text",
    "doctor_significant_electrodes",
    "doctor_spread_electrodes",
    "onset_channels",
    "soz_bipolar",
    "candidate_seizure_types",
    "mixed_channel_seizure_types",
    "earliest_onset_channels",
    "plus1_added_channels",
    "n_earliest_onset_channels",
    "n_plus1_added_channels",
    "earliest_channel_onset_sec",
    "onset_candidate_limit_sec",
    "onset_tolerance_sec",
    "soz_region",
    "soz_region_source",
    "soz_region_ranking",
    "soz_region_top1_tied_regions",
    "soz_region_top1_tie_size",
    "soz_region_top1_margin",
    "soz_region_votes_left_frontal",
    "soz_region_votes_right_frontal",
    "soz_region_votes_left_temporal",
    "soz_region_votes_right_temporal",
    "soz_region_votes_central_parietal",
    "regions",
    "propagation_regions",
    "review_status",
    "quality_flags",
    "source_file",
)

REGION_LABEL_COLUMNS: Tuple[str, ...] = tuple(f"region_{name}" for name in REGION_NAMES)
REGION_MASK_COLUMNS: Tuple[str, ...] = tuple(f"region_mask_{name}" for name in REGION_NAMES)
REGION_PROP_COLUMNS: Tuple[str, ...] = tuple(f"propagation_region_{name}" for name in REGION_NAMES)
CHANNEL_LABEL_MASK_COLUMNS: Tuple[str, ...] = tuple(f"label_mask_{col}" for col in TCP_COLUMNS)
CHANNEL_PROP_COLUMNS: Tuple[str, ...] = tuple(f"propagation_{col}" for col in TCP_COLUMNS)
CHANNEL_INPUT_MASK_COLUMNS: Tuple[str, ...] = tuple(f"input_mask_{col}" for col in TCP_COLUMNS)

CANONICAL_MANIFEST_FIELDS: Tuple[str, ...] = (
    MANIFEST_BASE_FIELDS
    + TCP_COLUMNS
    + CHANNEL_LABEL_MASK_COLUMNS
    + CHANNEL_PROP_COLUMNS
    + CHANNEL_INPUT_MASK_COLUMNS
    + REGION_LABEL_COLUMNS
    + REGION_MASK_COLUMNS
    + REGION_PROP_COLUMNS
)
