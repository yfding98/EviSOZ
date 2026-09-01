#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROLE="${1:-}"
REVIEWER_ID="${2:-}"
COHORT="${3:-s1_development}"

if [[ "$ROLE" != "reader_a" && "$ROLE" != "reader_b" && "$ROLE" != "adjudicator" ]]; then
  echo "Usage: $0 {reader_a|reader_b|adjudicator} REVIEWER_ID [s1_development|s1_calibration|s1_locked]" >&2
  exit 2
fi
if [[ -z "$REVIEWER_ID" ]]; then
  echo "REVIEWER_ID is required" >&2
  exit 2
fi

case "$ROLE" in
  reader_a) PORT=8771 ;;
  reader_b) PORT=8772 ;;
  adjudicator) PORT=8773 ;;
esac

cd "$PROJECT_ROOT"
exec python3 scripts/serve_tusz_eeg_only_s1_expert_reader.py \
  --role "$ROLE" \
  --reviewer-id "$REVIEWER_ID" \
  --cohort "$COHORT" \
  --host 127.0.0.1 \
  --port "$PORT"
