#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Pull the signed GPU firmware an external-GPU bring-up hands to the card.
#
# The blobs are the vendors' own — the AMD PSP and the NVIDIA GSP bootloader
# verify them against keys fused into the silicon. rlx neither produces nor
# re-signs them; it downloads them from linux-firmware at a pinned commit and
# checks each against the sha256 recorded in the manifest. A mismatch is a hard
# failure: it means the upstream file changed under a pin that claims it did
# not, which is exactly the case where continuing would be wrong.
#
# Usage:
#   scripts/pull_gpu_firmware.sh                 # everything in the manifest
#   scripts/pull_gpu_firmware.sh amd             # one vendor
#   scripts/pull_gpu_firmware.sh gc_11_0 psp_13_0  # specific IP families
#   scripts/pull_gpu_firmware.sh --list          # show what is available
#   scripts/pull_gpu_firmware.sh --check         # verify the cache, download nothing
#
# Destination: $RLX_FW_DIR, else $XDG_CACHE_HOME/rlx/firmware, else
# ~/.cache/rlx/firmware. Already-present files that verify are left alone, so
# re-running is cheap and safe.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
manifest="${RLX_FW_MANIFEST:-$script_dir/../crates/backends/rlx-egpu/firmware/manifest.tsv}"
fw_dir="${RLX_FW_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/rlx/firmware}"
repo_raw="https://gitlab.com/kernel-firmware/linux-firmware/-/raw"

[ -r "$manifest" ] || { echo "manifest not readable: $manifest" >&2; exit 1; }

commit=$(sed -n 's/^# commit[[:space:]]*//p' "$manifest" | head -1)
[ -n "$commit" ] || { echo "manifest has no pinned commit" >&2; exit 1; }

# sha256 across the platforms this repo builds on.
if command -v sha256sum >/dev/null 2>&1; then
  sha256_of() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
  sha256_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  echo "need sha256sum or shasum" >&2; exit 1
fi
command -v curl >/dev/null 2>&1 || { echo "need curl" >&2; exit 1; }

mode=pull
filters=()
for arg in "$@"; do
  case "$arg" in
    --list)  mode=list ;;
    --check) mode=check ;;
    -h|--help) sed -n '6,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option: $arg" >&2; exit 1 ;;
    *) filters+=("$arg") ;;
  esac
done

# An entry is selected when no filter was given, or when a filter matches its
# vendor or its IP family.
selected() {
  [ ${#filters[@]} -eq 0 ] && return 0
  local vendor=$1 family=$2 f
  for f in "${filters[@]}"; do
    [ "$f" = "$vendor" ] && return 0
    [ "$f" = "$family" ] && return 0
  done
  return 1
}

if [ "$mode" = list ]; then
  printf '%-8s %-10s %s\n' VENDOR FAMILY FILES
  awk -F'\t' '!/^#/ && NF>=5 { count[$1"\t"$2]++ } END { for (k in count) print k"\t"count[k] }' "$manifest" \
    | sort | while IFS=$'\t' read -r vendor family count; do
        printf '%-8s %-10s %s\n' "$vendor" "$family" "$count"
      done
  exit 0
fi

total=0 have=0 fetched=0 failed=0
mkdir -p "$fw_dir"

while IFS=$'\t' read -r vendor family path name sha256; do
  case "$vendor" in ''|\#*) continue ;; esac
  [ -n "${sha256:-}" ] || continue
  selected "$vendor" "$family" || continue
  total=$((total + 1))

  dest="$fw_dir/$path/$name"
  if [ -f "$dest" ] && [ "$(sha256_of "$dest")" = "$sha256" ]; then
    have=$((have + 1))
    continue
  fi

  if [ "$mode" = check ]; then
    if [ -f "$dest" ]; then
      echo "CORRUPT $path/$name" >&2
    else
      echo "MISSING $path/$name"
    fi
    failed=$((failed + 1))
    continue
  fi

  mkdir -p "$(dirname "$dest")"
  tmp="$dest.partial.$$"
  url="$repo_raw/$commit/$path/$name"
  if ! curl -fsSL --retry 3 --retry-delay 1 -o "$tmp" "$url"; then
    rm -f "$tmp"
    echo "FAILED  $path/$name (download)" >&2
    failed=$((failed + 1))
    continue
  fi

  got=$(sha256_of "$tmp")
  if [ "$got" != "$sha256" ]; then
    rm -f "$tmp"
    # Never leave an unverified blob where the loader would find it: this one
    # gets handed to a PSP that will reject it, and a silent bad file turns a
    # clear checksum error into an opaque firmware-load hang.
    echo "FAILED  $path/$name (sha256 $got, expected $sha256)" >&2
    failed=$((failed + 1))
    continue
  fi

  mv -f "$tmp" "$dest"
  fetched=$((fetched + 1))
  echo "ok      $path/$name"
done < "$manifest"

echo
echo "firmware dir: $fw_dir"
echo "selected $total  verified-present $have  downloaded $fetched  failed $failed"
[ "$failed" -eq 0 ]
