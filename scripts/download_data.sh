#!/usr/bin/env bash
# Download the lmsys/lmsys-chat-1m dataset (gated) from Hugging Face.
#
# Requirements:
#   - hf CLI installed and logged in (`hf auth login`), or HF_TOKEN set in .env
#   - Access to the dataset accepted on https://huggingface.co/datasets/lmsys/lmsys-chat-1m
#
# Usage:
#   scripts/download_data.sh            # download to data/raw/lmsys-chat-1m
#   scripts/download_data.sh <dir>      # download to a custom directory
#
# Re-running is safe: files already downloaded are skipped.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ID="lmsys/lmsys-chat-1m"
OUT_DIR="${1:-$PROJECT_ROOT/data/raw/lmsys-chat-1m}"

# Pick up HF_TOKEN from .env if present (falls back to the hf CLI login otherwise)
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  HF_TOKEN_FROM_ENV="$(grep -E '^HF_TOKEN=' "$PROJECT_ROOT/.env" | cut -d= -f2- || true)"
  if [[ -n "${HF_TOKEN_FROM_ENV:-}" && -z "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN="$HF_TOKEN_FROM_ENV"
  fi
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "error: hf CLI not found. Install with: uv tool install huggingface_hub[cli]" >&2
  exit 1
fi

if ! hf auth whoami >/dev/null 2>&1 && [[ -z "${HF_TOKEN:-}" ]]; then
  echo "error: not logged in to Hugging Face. Run 'hf auth login' or set HF_TOKEN in .env" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "Downloading $REPO_ID -> $OUT_DIR"

hf download "$REPO_ID" \
  --repo-type dataset \
  --include "data/*.parquet" \
  --include "README.md" \
  --local-dir "$OUT_DIR"

echo
echo "Done. Files:"
ls -lh "$OUT_DIR/data"
