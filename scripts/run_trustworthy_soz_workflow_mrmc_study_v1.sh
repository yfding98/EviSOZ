#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:?usage: $0 reader_a|reader_b|reader_c REVIEWER_ID}"
REVIEWER_ID="${2:?usage: $0 reader_a|reader_b|reader_c REVIEWER_ID}"

case "$ROLE" in
  reader_a) PORT=8791 ;;
  reader_b) PORT=8792 ;;
  reader_c) PORT=8793 ;;
  *) echo "invalid role: $ROLE" >&2; exit 2 ;;
esac

exec env PYTHONPATH=. python3 scripts/serve_trustworthy_soz_workflow_mrmc_study_v1.py \
  --role "$ROLE" \
  --reviewer-id "$REVIEWER_ID" \
  --host 127.0.0.1 \
  --port "$PORT"
