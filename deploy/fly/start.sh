#!/bin/bash
set -e

VINDEX_DIR="${VINDEX_PATH:-/data/vindex}"
HF_REPO="${HF_REPO:-chrishayuk/gemma-4-26b-a4b-it-vindex-expert-server}"

# Verify the vindex is complete (index.json + at least one layer file)
LAYER_COUNT=$(ls "$VINDEX_DIR/layers/"*.weights 2>/dev/null | wc -l)
if [ ! -f "$VINDEX_DIR/index.json" ] || [ "$LAYER_COUNT" -lt 30 ]; then
  echo "Vindex incomplete ($LAYER_COUNT/30 layers) — re-downloading..."
  rm -rf "$VINDEX_DIR"
  mkdir -p "$VINDEX_DIR"
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 - <<PYEOF
import os, sys
from huggingface_hub import snapshot_download

repo_id = os.environ.get("HF_REPO", "chrishayuk/gemma-4-26b-a4b-it-vindex-expert-server")
token   = os.environ.get("HF_TOKEN") or None
dest    = os.environ.get("VINDEX_PATH", "/data/vindex")

print(f"Downloading {repo_id} → {dest}", flush=True)
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=dest,
    token=token,
    ignore_patterns=["*.md", ".gitattributes"],
)
print("Download complete.", flush=True)
PYEOF
  echo "Vindex ready at $VINDEX_DIR"
fi

echo "Starting larql-server from $VINDEX_DIR"
echo "  EXPERTS: ${EXPERTS:-all}"
echo "  LAYERS:  ${LAYERS:-all}"

EXTRA_ARGS=""
[ -n "$EXPERTS" ] && EXTRA_ARGS="$EXTRA_ARGS --experts $EXPERTS"
[ -n "$LAYERS"  ] && EXTRA_ARGS="$EXTRA_ARGS --layers $LAYERS"

exec larql-server "$VINDEX_DIR" --port "${PORT:-8080}" --host 0.0.0.0 $EXTRA_ARGS
