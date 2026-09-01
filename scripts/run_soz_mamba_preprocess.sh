#!/usr/bin/env bash
set -euo pipefail

# Build long onset-aligned event sequences for SOZ-Mamba.
#
# Usage:
#   bash scripts/run_soz_mamba_preprocess.sh all
#   bash scripts/run_soz_mamba_preprocess.sh tusz
#   bash scripts/run_soz_mamba_preprocess.sh private
#
# The default `all` mode creates one mixed directory that can be used by
# scripts/run_soz_mamba_training.sh through PREPROCESSED_DIR=...

MODE="${1:-all}"

MANIFEST="${MANIFEST:-outputs/soz_pre/unified_region_soz_manifest_ica_fnsz.csv}"
TUSZ_ROOT="${TUSZ_ROOT:-/mnt/hd1/dyf/dataset/TUSZ}"
TUSZ_PROCESSED_ROOT="${TUSZ_PROCESSED_ROOT:-outputs/tusz_fnsz_soz_preprocessed}"
PRIVATE_ROOT="${PRIVATE_ROOT:-outputs/soz_pre/private_ica_preprocessed}"

SEQUENCE_SEC="${SEQUENCE_SEC:-128}"
WINDOW_SEC="${WINDOW_SEC:-1}"
PRE_CONTEXT_SEC="${PRE_CONTEXT_SEC:-5}"
BASELINE_SEC="${BASELINE_SEC:-5}"
LONG_STRIDE_SEC="${LONG_STRIDE_SEC:-64}"
MAX_EXTRA_CROPS="${MAX_EXTRA_CROPS:-1}"
LONG_CROP_SPATIAL_WEIGHT="${LONG_CROP_SPATIAL_WEIGHT:-0.0}"
NORMALIZE="${NORMALIZE:-baseline_robust}"
SEIZURE_TYPES="${SEIZURE_TYPES:-fnsz}"
MAX_ROWS="${MAX_ROWS:-0}"
INCLUDE_SPH="${INCLUDE_SPH:-1}"
SKIP_FILTER="${SKIP_FILTER:-1}"

sequence_tag="${SEQUENCE_SEC%.*}s"

case "${MODE}" in
  all)
    SOURCE_FILTER="${SOURCE_FILTER:-all}"
    SPLITS="${SPLITS:-}"
    DEFAULT_OUTPUT_DIR="outputs/soz_mamba/event_sequences_${sequence_tag}"
    ;;
  tusz)
    SOURCE_FILTER="${SOURCE_FILTER:-tusz}"
    SPLITS="${SPLITS:-}"
    DEFAULT_OUTPUT_DIR="outputs/soz_mamba/event_sequences_${sequence_tag}_tusz"
    ;;
  private)
    SOURCE_FILTER="${SOURCE_FILTER:-private}"
    SPLITS="${SPLITS:-private}"
    DEFAULT_OUTPUT_DIR="outputs/soz_mamba/event_sequences_${sequence_tag}_private"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected one of: all, tusz, private" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

args=(
  --manifest "${MANIFEST}"
  --output_dir "${OUTPUT_DIR}"
  --tusz_root "${TUSZ_ROOT}"
  --tusz_processed_root "${TUSZ_PROCESSED_ROOT}"
  --private_root "${PRIVATE_ROOT}"
  --source_filter "${SOURCE_FILTER}"
  --splits "${SPLITS}"
  --seizure_types "${SEIZURE_TYPES}"
  --sequence_sec "${SEQUENCE_SEC}"
  --window_sec "${WINDOW_SEC}"
  --pre_context_sec "${PRE_CONTEXT_SEC}"
  --baseline_sec "${BASELINE_SEC}"
  --long_stride_sec "${LONG_STRIDE_SEC}"
  --max_extra_crops "${MAX_EXTRA_CROPS}"
  --long_crop_spatial_weight "${LONG_CROP_SPATIAL_WEIGHT}"
  --normalize "${NORMALIZE}"
  --max_rows "${MAX_ROWS}"
)

if [[ "${INCLUDE_SPH}" == "1" ]]; then
  args+=(--include_sph)
fi

if [[ "${SKIP_FILTER}" == "1" ]]; then
  args+=(--skip_filter)
fi

python3 code/soz_mamba/preprocess_event_sequences.py "${args[@]}"
