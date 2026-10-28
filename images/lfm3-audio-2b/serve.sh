#!/bin/sh
set -eu

MODEL="${LFM3_MODEL:-/models/LFM3-Audio-2B-Q4_K_M.gguf}"
MMPROJ="${LFM3_MMPROJ:-/models/mmproj-LFM3-Audio-2B-Q8_0.gguf}"
HOST="${LFM3_HOST:-0.0.0.0}"
PORT="${LFM3_PORT:-8080}"
THREADS="${LFM3_THREADS:-0}"

exec llama-server \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --host "$HOST" \
  --port "$PORT" \
  --threads "$THREADS" \
  --temp 0.1 \
  --top-k 50 \
  --repeat-penalty 1.1 \
  --reasoning off \
  --alias lfm3-audio-2b
