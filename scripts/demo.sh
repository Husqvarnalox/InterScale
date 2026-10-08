#!/usr/bin/env bash
# Smoke-test a running InferScale server: health, models, one completion, one stream.
set -euo pipefail
URL="${1:-http://localhost:8000}"

curl -fsS "$URL/health" && echo
curl -fsS "$URL/v1/models" && echo

curl -fsS "$URL/v1/completions" -H 'content-type: application/json' \
  -d '{"prompt": "The KV cache stores", "max_tokens": 24, "temperature": 0}' && echo

curl -fsSN "$URL/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 16, "stream": true}'
