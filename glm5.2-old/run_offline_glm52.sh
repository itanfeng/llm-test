#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/offline_glm52.log}"
mkdir -p "$(dirname -- "$LOG_FILE")"

echo "Log file: $LOG_FILE"
python3 offline_glm52.py "$@" 2>&1 | tee "$LOG_FILE"
