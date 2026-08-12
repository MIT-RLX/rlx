#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Leak gate for the Metal backend.
#
# Refcount bugs are invisible to the test suite: an over-retained Objective-C
# object still produces correct numbers, so a green suite says nothing about
# whether the backend frees what it allocates. This backend hand-rolls its Metal
# bindings (`src/mtl.rs`) and messages the Objective-C runtime directly in ~160
# more places, and Cocoa's ownership rule — `new`/`alloc`/`copy` return +1,
# everything else is autoreleased — is enforced by nothing but review. It has
# already been got wrong once: eight call sites handed a +1 `MPSGraphTensorData`
# to a collection that retains it and dropped their own reference, so every one
# bottomed out at refcount 1. `leaks` found it; the suite never would.
#
# Runs each binary under `leaks --atExit`, which exits non-zero when the process
# still holds unreachable allocations at exit.
#
# Usage: crates/backends/rlx-metal/scripts/leak-check.sh [test-name ...]
set -euo pipefail

if [[ "$(uname -s)" != Darwin ]]; then
    echo "leak-check: macOS only (needs /usr/bin/leaks) — skipping"
    exit 0
fi
if ! command -v leaks >/dev/null 2>&1; then
    echo "leak-check: /usr/bin/leaks not found — skipping"
    exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

# Default set: broad op coverage, the MPSGraph bind path (where the known leak
# lived), and a fused path that allocates per dispatch.
TESTS=("$@")
if [[ ${#TESTS[@]} -eq 0 ]]; then
    TESTS=(full_coverage_parity metal_swiglu_full_parity metal_mlp_head_parity)
fi

echo "leak-check: building test binaries…"
cargo test --release -p rlx-metal --no-run -q 2>/dev/null

status=0
for t in "${TESTS[@]}"; do
    # Cargo suffixes each test binary with a hash; take the newest match so a
    # stale build from an earlier revision can't be silently measured instead.
    bin="$(ls -t target/release/deps 2>/dev/null | grep -E "^${t}-[0-9a-f]+$" | head -1 || true)"
    if [[ -z "$bin" ]]; then
        echo "leak-check: FAIL ${t} — no built binary found"
        status=1
        continue
    fi

    out="$(MallocStackLogging=1 leaks --atExit -- "target/release/deps/${bin}" 2>&1 || true)"
    summary="$(printf '%s\n' "$out" | grep -E 'leaks for [0-9]+ total leaked bytes' | tail -1 || true)"

    if printf '%s\n' "$out" | grep -q '0 leaks for 0 total leaked bytes'; then
        echo "leak-check: ok   ${t} — ${summary#*: }"
    else
        echo "leak-check: FAIL ${t} — ${summary:-no summary line; see below}"
        # The ROOT LEAK stacks are what localise it; keep them in the log.
        printf '%s\n' "$out" | sed -n '/ROOT LEAK/,/^$/p' | head -40
        status=1
    fi
done

if [[ $status -ne 0 ]]; then
    echo
    echo "leak-check: leaks detected. Cocoa ownership: selectors starting new/alloc/copy"
    echo "return +1 and you must release; everything else is autoreleased and you must not."
fi
exit $status
