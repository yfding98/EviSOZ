#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROLE="${1:-}"
REVIEWER_ID="${2:-}"

if [[ "$ROLE" != "reader_a" && "$ROLE" != "reader_b" ]]; then
  echo "Usage: $0 {reader_a|reader_b} REVIEWER_ID" >&2
  exit 2
fi
if [[ -z "$REVIEWER_ID" ]]; then
  echo "REVIEWER_ID is required" >&2
  exit 2
fi

case "$ROLE" in
  reader_a) PORT=8781 ;;
  reader_b) PORT=8782 ;;
esac

cd "$PROJECT_ROOT"
exec python3 scripts/serve_trustworthy_soz_report_reader_study_v1.py \
  --role "$ROLE" \
  --reviewer-id "$REVIEWER_ID" \
  --host 127.0.0.1 \
  --port "$PORT"
