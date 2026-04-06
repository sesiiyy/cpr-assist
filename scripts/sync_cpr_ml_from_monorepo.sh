#!/usr/bin/env bash
# Copy CPR ML sources from a full monorepo checkout into cpr-assist/cpr_ml/.
#
#   ./scripts/sync_cpr_ml_from_monorepo.sh /path/to/cpr
#
# Requires: rsync (Linux/macOS or Git Bash on Windows).

set -euo pipefail

MONO="${1:-}"
if [[ -z "${MONO}" || ! -f "${MONO}/src/config.py" ]]; then
  echo "Usage: $0 /path/to/cpr   (directory must contain src/config.py)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${SCRIPT_DIR}/../cpr_ml"

RSYNC=(rsync -a --delete --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.mypy_cache' --exclude '*.pyc')

"${RSYNC[@]}" "${MONO}/src/" "${DEST}/src/"
"${RSYNC[@]}" "${MONO}/configs/" "${DEST}/configs/"
"${RSYNC[@]}" "${MONO}/api/cpr_api/" "${DEST}/api/cpr_api/"

mkdir -p "${DEST}/experiments/cpr_s0_image_classifier/runs/s0_image_model"
mkdir -p "${DEST}/experiments/ct_depth_tabular/frozen"
"${RSYNC[@]}" "${MONO}/experiments/cpr_s0_image_classifier/runs/s0_image_model/" "${DEST}/experiments/cpr_s0_image_classifier/runs/s0_image_model/"
"${RSYNC[@]}" "${MONO}/experiments/ct_depth_tabular/frozen/" "${DEST}/experiments/ct_depth_tabular/frozen/"

echo "Synced into ${DEST}"
