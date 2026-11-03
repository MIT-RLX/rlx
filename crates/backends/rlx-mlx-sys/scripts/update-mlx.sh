#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
# Move the vendored MLX submodule (vendor/mlx) to a new upstream ref and
# re-apply the tracked patches in ../patches.
#
# usage: update-mlx.sh [REF] [--build]
#   REF      upstream tag, branch or commit (default: newest vX.Y.Z release tag)
#            `main` moves to the tip of upstream main.
#   --build  run `cargo build -p rlx-mlx-sys` afterwards (CMake + metallib).
#
# The submodule pointer change is left unstaged in the parent repo; review it
# with `git diff --submodule` and commit it yourself.
set -euo pipefail

CRATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLX="$CRATE/vendor/mlx"
PATCHES=("$CRATE"/patches/*.patch)
REF=""
BUILD=0
for a in "$@"; do
    case "$a" in
        --build) BUILD=1 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) REF="$a" ;;
    esac
done

if [[ ! -e "$MLX/.git" ]]; then
    echo "initializing submodule vendor/mlx"
    git -C "$CRATE" submodule update --init --depth 1 vendor/mlx
fi
g() { git -C "$MLX" "$@"; }

old="$(g rev-parse --short HEAD)"
old_desc="$(g describe --tags --exact-match 2>/dev/null || echo "$old")"

# Take the tracked patches back out (newest first) so the tree is clean upstream.
for ((i = ${#PATCHES[@]} - 1; i >= 0; i--)); do
    p="${PATCHES[$i]}"
    if g apply --check --reverse "$p" 2>/dev/null; then
        g apply --reverse "$p"
    fi
done
if [[ -n "$(g status --porcelain --untracked-files=no)" ]]; then
    echo "error: vendor/mlx has local changes beyond patches/*.patch:" >&2
    g status --short --untracked-files=no >&2
    echo "stash or drop them, then re-run." >&2
    # Put the patches back so the build state is unchanged.
    for p in "${PATCHES[@]}"; do g apply --check "$p" 2>/dev/null && g apply "$p"; done
    exit 1
fi

if [[ -z "$REF" ]]; then
    REF="$(g ls-remote --tags --refs origin 'v*' \
        | awk -F/ '{print $3}' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' \
        | sort -V | tail -1)"
    [[ -n "$REF" ]] || { echo "error: no release tags found on origin" >&2; exit 1; }
fi

echo "fetching $REF"
if g ls-remote --exit-code --tags origin "refs/tags/$REF" >/dev/null 2>&1; then
    g fetch -q --depth 1 origin tag "$REF"
    target="refs/tags/$REF"
else
    g fetch -q --depth 1 origin "$REF"
    target="FETCH_HEAD"
fi
g checkout -q --detach "$target"
new="$(g rev-parse --short HEAD)"

failed=()
for p in "${PATCHES[@]}"; do
    if g apply --check "$p" 2>/dev/null; then
        g apply "$p"
        echo "  applied $(basename "$p")"
    else
        failed+=("$p")
        echo "  FAILED  $(basename "$p")"
    fi
done

echo "vendor/mlx: $old_desc -> $REF ($new)"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo >&2
    echo "Patches that no longer apply must be re-based (see patches/README.md):" >&2
    for p in "${failed[@]}"; do
        echo "  git -C $MLX apply --3way $p" >&2
    done
    echo "resolve, re-export the patch with \`git diff -- <its files>\`, then re-run." >&2
    exit 1
fi

if [[ "$BUILD" == 1 ]]; then
    cargo build -p rlx-mlx-sys --manifest-path "$CRATE/Cargo.toml"
fi
echo "review with: git -C $CRATE diff --submodule vendor/mlx"
