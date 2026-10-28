#!/bin/sh
set -eu

MODEL="${LFM3_MODEL:-/models/LFM3-1.7B-Q4_K_M.gguf}"
HOST="${LFM3_HOST:-0.0.0.0}"
PORT="${LFM3_PORT:-8080}"
THREADS="${LFM3_THREADS:-0}"

exec llama-server \
  -m "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --threads "$THREADS" \
  --temp 0.1 \
  --top-k 50 \
  --repeat-penalty 1.1 \
  --alias lfm3-1.7b
