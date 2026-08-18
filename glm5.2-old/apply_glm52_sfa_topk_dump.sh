#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${VLLM_ASCEND_REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)/vllm-ascend}"
PATCH_FILE="$SCRIPT_DIR/glm52_sfa_topk_dump.patch"
ACTION="${1:-apply}"

[[ -d "$REPO/.git" ]] || { echo "ERROR: not a git repository: $REPO" >&2; exit 1; }

case "$ACTION" in
    apply)
        if git -C "$REPO" apply --check "$PATCH_FILE" 2>/dev/null; then
            git -C "$REPO" apply "$PATCH_FILE"
            echo "Applied: $PATCH_FILE"
        elif git -C "$REPO" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
            echo "Patch is already applied; no changes made."
        else
            echo "ERROR: patch cannot be applied or cleanly reversed; inspect local changes in $REPO" >&2
            exit 1
        fi
        ;;
    check)
        if git -C "$REPO" apply --check "$PATCH_FILE" 2>/dev/null; then
            echo "Patch is not applied and can be applied cleanly."
        elif git -C "$REPO" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
            echo "Patch is already applied."
        else
            echo "ERROR: patch state is inconsistent with $PATCH_FILE" >&2
            exit 1
        fi
        ;;
    reverse)
        if git -C "$REPO" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
            git -C "$REPO" apply --reverse "$PATCH_FILE"
            echo "Reversed: $PATCH_FILE"
        elif git -C "$REPO" apply --check "$PATCH_FILE" 2>/dev/null; then
            echo "Patch is not applied; no changes made."
        else
            echo "ERROR: patch cannot be cleanly reversed; inspect local changes in $REPO" >&2
            exit 1
        fi
        ;;
    *)
        echo "Usage: $0 [apply|check|reverse]" >&2
        exit 2
        ;;
esac
