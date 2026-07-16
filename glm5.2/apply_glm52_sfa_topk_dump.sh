#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${VLLM_ASCEND_REPO:-/Users/tanfeng/code/vllm-ascend}"
PATCH_FILE="$SCRIPT_DIR/glm52_sfa_topk_dump.patch"
ACTION="${1:-apply}"

[[ -d "$REPO/.git" ]] || { echo "ERROR: not a git repository: $REPO" >&2; exit 1; }

case "$ACTION" in
    apply)
        git -C "$REPO" apply --check "$PATCH_FILE"
        git -C "$REPO" apply "$PATCH_FILE"
        echo "Applied: $PATCH_FILE"
        ;;
    check)
        git -C "$REPO" apply --check "$PATCH_FILE"
        echo "Patch can be applied cleanly."
        ;;
    reverse)
        git -C "$REPO" apply --reverse --check "$PATCH_FILE"
        git -C "$REPO" apply --reverse "$PATCH_FILE"
        echo "Reversed: $PATCH_FILE"
        ;;
    *)
        echo "Usage: $0 [apply|check|reverse]" >&2
        exit 2
        ;;
esac
