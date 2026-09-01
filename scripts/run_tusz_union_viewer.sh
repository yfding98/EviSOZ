#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TUSZ_ROOT="${TUSZ_ROOT:-/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf}"
MANIFEST="${MANIFEST:-outputs/tusz_viewer/tusz_v203_viewer_manifest.csv}"
ANNOTATION_DIR="${ANNOTATION_DIR:-outputs/tusz_viewer/annotations}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ANNOTATION_MANIFEST="${ANNOTATION_MANIFEST:-${ANNOTATION_DIR}/tusz_viewer_annotated_${STAMP}.csv}"
SPLITS="${SPLITS:-train,dev,eval}"
ONSET_TOLERANCE_SEC="${ONSET_TOLERANCE_SEC:-1.0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8766}"
RAW_VIEW="${RAW_VIEW:-0}"
DATASET="${DATASET:-tusz}"
LIMIT="${LIMIT:-}"
REBUILD_MANIFEST="${REBUILD_MANIFEST:-0}"

if [[ "$RAW_VIEW" == "1" ]]; then
  SFREQ="${SFREQ:-0}"
  FILTER_LOW="${FILTER_LOW:-0}"
  FILTER_HIGH="${FILTER_HIGH:-0}"
else
  SFREQ="${SFREQ:-200}"
  FILTER_LOW="${FILTER_LOW:-1.0}"
  FILTER_HIGH="${FILTER_HIGH:-50.0}"
fi

mkdir -p "$(dirname "$MANIFEST")" "$ANNOTATION_DIR"

if [[ "$REBUILD_MANIFEST" == "1" || ! -s "$MANIFEST" ]]; then
  python3 code/deepsoz/build_tusz_v203_manifest.py \
    --tusz_root "$TUSZ_ROOT" \
    --output "$MANIFEST" \
    --splits "$SPLITS" \
    --onset_tolerance_sec "$ONSET_TOLERANCE_SEC"
fi

cmd=(
  python3 code/data_preprocess/realtime_eeg_union_viewer.py
  --manifest "$MANIFEST"
  --dataset "$DATASET"
  --tusz_root "$TUSZ_ROOT"
  --annotation_manifest "$ANNOTATION_MANIFEST"
  --host "$HOST"
  --port "$PORT"
  --sfreq "$SFREQ"
  --filter_low "$FILTER_LOW"
  --filter_high "$FILTER_HIGH"
)

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi
if [[ "${VERBOSE:-0}" == "1" ]]; then
  cmd+=(--verbose)
fi

printf 'TUSZ root: %s\n' "$TUSZ_ROOT"
printf 'Viewer manifest: %s\n' "$MANIFEST"
printf 'Annotation copy: %s\n' "$ANNOTATION_MANIFEST"
printf 'Signal mode: %s (sfreq=%s, filter_low=%s, filter_high=%s)\n' \
  "$([[ "$RAW_VIEW" == "1" ]] && printf raw || printf processed)" \
  "$SFREQ" "$FILTER_LOW" "$FILTER_HIGH"
printf 'Open: http://%s:%s/\n' "$HOST" "$PORT"

exec "${cmd[@]}"
