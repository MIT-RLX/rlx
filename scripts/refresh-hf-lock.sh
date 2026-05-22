#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# Resolve every `main` entry in tests/hf-repo-lock.tsv to a concrete
# SHA by querying the HF API. Plan #62.
#
# Usage:
#   scripts/refresh-hf-lock.sh           # resolve `main` → real SHAs
#   scripts/refresh-hf-lock.sh --dry-run # show what would change
#
# Requires: curl, jq, tab-aware awk.

set -eu

LOCKFILE="$(dirname "$0")/../tests/hf-repo-lock.tsv"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

if [ ! -f "$LOCKFILE" ]; then
  echo "missing $LOCKFILE" >&2
  exit 2
fi

today="$(date -u +%Y-%m-%d)"
tmp="$(mktemp)"

while IFS=$'\t' read -r repo rev date note; do
  # Pass through comments and blank lines.
  case "$repo" in
    \#*|"") echo "$repo	$rev	$date	$note" >> "$tmp"; continue ;;
  esac

  if [ "$rev" != "main" ]; then
    echo "$repo	$rev	$date	$note" >> "$tmp"
    continue
  fi

  # Query HF for the SHA of `main`.
  url="https://huggingface.co/api/models/$repo/revision/main"
  sha="$(curl -sf "$url" | jq -r '.sha // empty' 2>/dev/null || true)"
  if [ -z "$sha" ]; then
    echo "[hf-lock] WARN: failed to resolve $repo (network? auth?)" >&2
    echo "$repo	$rev	$date	$note" >> "$tmp"
    continue
  fi

  echo "[hf-lock] $repo: main → ${sha:0:12}"
  echo "$repo	$sha	$today	$note" >> "$tmp"
done < "$LOCKFILE"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "--- proposed lockfile ---"
  cat "$tmp"
  rm "$tmp"
else
  mv "$tmp" "$LOCKFILE"
  echo "[hf-lock] updated $LOCKFILE"
fi
