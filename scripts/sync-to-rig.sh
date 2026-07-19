#!/usr/bin/env bash
# Full-tree sync of this RLX checkout to a remote Linux rig.
# Partial syncs (e.g. only rlx-cpu) cause IR/OpKind mismatches with older rlx-ir.
# Set RLX_RIG_HOST=user@host (and optionally RLX_RIG_DEST=path) before running.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${RLX_RIG_HOST:?set RLX_RIG_HOST=user@host}"
DEST="${RLX_RIG_DEST:-rlx}"
rsync -az --delete \
  --exclude target \
  --exclude .git \
  --exclude '*.o' \
  --exclude .DS_Store \
  "$ROOT/" "$HOST:$DEST/"
echo "synced $ROOT -> $HOST:$DEST"
