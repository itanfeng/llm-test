#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

API_URL="${API_URL:-http://127.0.0.1:8077/v1/chat/completions}"
SERVED_MODEL="${SERVED_MODEL:-glm-52}"
PROMPT="${PROMPT:-Explain what a Top-K attention indexer does in two sentences.}"
MAX_TOKENS="${MAX_TOKENS:-32}"
TOPK_DIR="${GLM52_TOPK_DIR:-data}"
RESPONSE_FILE="${RESPONSE_FILE:-/tmp/glm52_topk_response.json}"
LOCK_FILE="${LOCK_FILE:-/tmp/glm52_topk_sender.lock}"

command -v curl >/dev/null
command -v jq >/dev/null
mkdir -p "$TOPK_DIR"

exec 9>"$LOCK_FILE"
if command -v flock >/dev/null 2>&1; then
    flock -n 9 || { echo "ERROR: another dump sender is active" >&2; exit 1; }
fi

BEFORE_COUNT="$(find "$TOPK_DIR" -maxdepth 1 -type f -name '*.pt' | wc -l | tr -d ' ')"

REQUEST_FILE="$(mktemp /tmp/glm52_topk_request.XXXXXX.json)"
trap 'rm -f "$REQUEST_FILE"' EXIT INT TERM
jq -n \
    --arg model "$SERVED_MODEL" \
    --arg prompt "$PROMPT" \
    --argjson max_tokens "$MAX_TOKENS" \
    '{model:$model,messages:[{role:"user",content:$prompt}],max_tokens:$max_tokens,temperature:0}' \
    > "$REQUEST_FILE"

HTTP_CODE="$(curl --silent --show-error --output "$RESPONSE_FILE" --write-out '%{http_code}' \
    --header 'Content-Type: application/json' --data-binary "@$REQUEST_FILE" "$API_URL")"

[[ "$HTTP_CODE" =~ ^2[0-9][0-9]$ ]] || { cat "$RESPONSE_FILE" >&2; exit 1; }
jq -e '.choices | type == "array"' "$RESPONSE_FILE" >/dev/null

AFTER_COUNT="$(find "$TOPK_DIR" -maxdepth 1 -type f -name '*.pt' | wc -l | tr -d ' ')"
NEW_DUMP_COUNT=$((AFTER_COUNT - BEFORE_COUNT))
echo "HTTP status: $HTTP_CODE"
echo "New dumps:   $NEW_DUMP_COUNT"
echo "Total dumps: $AFTER_COUNT"
echo "Response:    $RESPONSE_FILE"
(( NEW_DUMP_COUNT > 0 )) || { echo "ERROR: request succeeded but no new tensors were dumped" >&2; exit 1; }
