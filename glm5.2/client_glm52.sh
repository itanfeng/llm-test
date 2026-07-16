#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

URL="${URL:-http://127.0.0.1:8077/v1/chat/completions}"
MODEL="${MODEL:-glm-52}"
PROMPTS_FILE="${PROMPTS_FILE:-${SCRIPT_DIR}/swe_prompts.jsonl}"
RESP_FILE="${RESP_FILE:-${SCRIPT_DIR}/online_glm52_responses.jsonl}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/online_glm52_client.log}"
TOPK_DIR="${GLM52_TOPK_DIR:-${SCRIPT_DIR}/pt}"

START_LINE="${START_LINE:-1}"
COUNT="${COUNT:-0}"
DECODE_STEPS="${DECODE_STEPS:-64}"
MAX_TOKENS="${MAX_TOKENS:-$((DECODE_STEPS + 1))}"
TEMPERATURE="${TEMPERATURE:-0}"
IGNORE_EOS="${IGNORE_EOS:-1}"
CLEAR_TOPK_DUMPS="${CLEAR_TOPK_DUMPS:-0}"
REQUIRE_TOPK_DUMP="${REQUIRE_TOPK_DUMP:-0}"

command -v curl >/dev/null
command -v jq >/dev/null
[[ -f "$PROMPTS_FILE" ]] || { echo "ERROR: not found: $PROMPTS_FILE" >&2; exit 1; }
[[ "$START_LINE" =~ ^[1-9][0-9]*$ && "$COUNT" =~ ^[0-9]+$ ]] || {
    echo "ERROR: invalid START_LINE or COUNT" >&2; exit 1;
}

mkdir -p "$TOPK_DIR" "$(dirname "$RESP_FILE")" "$(dirname "$LOG_FILE")"
exec > >(tee "$LOG_FILE") 2>&1

echo "Log file: $LOG_FILE"
[[ "$CLEAR_TOPK_DUMPS" == "0" ]] || find "$TOPK_DIR" -maxdepth 1 -type f \
    \( -name '*.pt' -o -name '*.tmp' \) -delete
before_count="$(find "$TOPK_DIR" -maxdepth 1 -type f -name '*.pt' | wc -l | tr -d ' ')"
: > "$RESP_FILE"

request_file="$(mktemp "${SCRIPT_DIR}/.request.XXXXXX.json")"
response_file="$(mktemp "${SCRIPT_DIR}/.response.XXXXXX.json")"
trap 'rm -f "$request_file" "$response_file"' EXIT

sent=0
while IFS= read -r record; do
    line_number=$((START_LINE + sent))
    jq -n \
        --arg model "$MODEL" --argjson source "$record" \
        --argjson max_tokens "$MAX_TOKENS" --argjson temperature "$TEMPERATURE" \
        --argjson ignore_eos "$IGNORE_EOS" \
        '{model:$model, messages:[{role:"user", content:$source.prompt}],
          max_tokens:$max_tokens, temperature:$temperature, top_p:1.0,
          seed:1024, ignore_eos:($ignore_eos == 1)}' > "$request_file"

    http_code="$(curl -sS -o "$response_file" -w '%{http_code}' \
        -H 'Content-Type: application/json' --data-binary "@$request_file" "$URL")"
    [[ "$http_code" =~ ^2 ]] && jq -e '.error == null and (.choices | length > 0)' \
        "$response_file" >/dev/null || {
        echo "ERROR: line $line_number failed (HTTP $http_code)" >&2
        jq . "$response_file" >&2 || true
        exit 1
    }

    jq -cn --argjson line "$line_number" --argjson source "$record" \
        --slurpfile response "$response_file" \
        '{line:$line, instance_id:($source.instance_id // ""),
          repo:($source.repo // ""), response:$response[0]}' >> "$RESP_FILE"
    sent=$((sent + 1))
    echo "[$sent] line=$line_number HTTP=$http_code"
done < <(awk -v start="$START_LINE" -v count="$COUNT" \
    'NR >= start && (count == 0 || NR < start + count)' "$PROMPTS_FILE")

after_count="$(find "$TOPK_DIR" -maxdepth 1 -type f -name '*.pt' | wc -l | tr -d ' ')"
new_count=$((after_count - before_count))
echo "Requests: $sent; new dumps: $new_count; responses: $RESP_FILE"
[[ "$REQUIRE_TOPK_DUMP" == "0" || "$new_count" -gt 0 ]] || {
    echo "ERROR: no Top-K tensor was dumped" >&2; exit 1;
}
