#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# MPSGraph executable-compile crash soak.
#
# `-[MPSGraph compileWithDevice:…]` has been observed faulting inside
# MetalPerformanceShadersGraph itself at roughly 2% per executable
# initialization (see docs/apple-feedback-mpsgraph-crash.md). `mps_graph.rs`
# mitigates it by passing a real MPSGraphCompilationDescriptor with
# `waitForCompilationCompletion = YES`, keeping the work on the calling thread.
#
# That mitigation was never measured, and at a 2% rate a short run proves
# nothing: P(zero crashes in 20 runs | p=0.02) is about 0.67, so "20 clean" is
# the *expected* outcome even if nothing was fixed. This runs N separate
# processes — separate because a fault takes the process with it — and reports
# the crash count with a one-sided 95% bound.
#
# Usage: crates/backends/rlx-metal/scripts/mpsgraph-soak.sh [N] [test-name]
set -uo pipefail

N="${1:-200}"
TEST="${2:-metal_reve_block_mpsgraph_parity}"

if [[ "$(uname -s)" != Darwin ]]; then
    echo "mpsgraph-soak: macOS only — skipping"
    exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

echo "mpsgraph-soak: building ${TEST}…"
cargo test --release -p rlx-metal --test "$TEST" --no-run -q 2>/dev/null

bin="$(ls -t target/release/deps 2>/dev/null | grep -E "^${TEST}-[0-9a-f]+$" | head -1 || true)"
if [[ -z "$bin" ]]; then
    echo "mpsgraph-soak: no built binary for ${TEST}"
    exit 1
fi

echo "mpsgraph-soak: ${N} runs of ${bin}"
crashes=0
failures=0
for ((i = 1; i <= N; i++)); do
    if ! out="$("target/release/deps/${bin}" 2>&1)"; then
        code=$?
        # >128 means killed by a signal: SIGSEGV/SIGABRT is the crash we care
        # about. A plain non-zero exit is an ordinary test failure, which is a
        # different problem and should not inflate the crash rate.
        if [[ $code -gt 128 ]]; then
            crashes=$((crashes + 1))
            echo "  run ${i}: CRASH (signal $((code - 128)))"
        else
            failures=$((failures + 1))
            echo "  run ${i}: test failure (exit ${code})"
        fi
    fi
    if (( i % 50 == 0 )); then echo "  … ${i}/${N} (${crashes} crashes)"; fi
done

echo
echo "mpsgraph-soak: ${crashes} crash(es) and ${failures} test failure(s) in ${N} runs"
if [[ $crashes -eq 0 ]]; then
    # One-sided 95% upper bound with zero events: 3/N (the "rule of three").
    python3 - "$N" <<'PY'
import sys
n = int(sys.argv[1])
print(f"  zero crashes in {n} runs → 95% upper bound on the rate ≈ {300.0/n:.2f}%")
print("  (rule of three; needs n≈150 to bound a 2% rate below itself)")
PY
else
    python3 - "$crashes" "$N" <<'PY'
import sys
c, n = int(sys.argv[1]), int(sys.argv[2])
print(f"  observed rate {100.0*c/n:.2f}% — mitigation is not sufficient")
PY
    exit 1
fi
