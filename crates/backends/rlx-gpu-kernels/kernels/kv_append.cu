// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// In-place KV append (`Op::KvAppend`) on the f32-uniform arena.
//
// Writes the new token's row (`src`, `outer * inner` elems) into the cache
// buffer at sequence index `pos`:
//
//   dst[(o * seq_cap + pos) * inner + i] = src[o * inner + i]
//
// `dst` ALIASES input 0 (the cache) — the shared memory planner maps
// `Op::KvAppend`'s output onto its `cache` input at offset 0
// (`rlx-compile/src/memory.rs`), so this kernel mutates the cache in place and
// every other row is left untouched. That is the whole point: the resident-KV
// decode path grows the cache on-device instead of re-uploading the padded
// cache from the host every token.
//
// `inner` / `pos` / `seq_cap` are a FIXED row write — deliberately NOT scaled by
// any active extent, matching rlx-metal's `Thunk::KvAppend`. The new K/V always
// lands at row `pos` (the bucket end) and the attention mask covers the padded
// gap.
//
// One thread per copied element; `outer * inner` is small (a single token's row),
// so a flat 1-D grid is the right shape.

extern "C" __global__ void kv_append(
    float* arena,
    unsigned int src_off,
    unsigned int dst_off,
    unsigned int outer,
    unsigned int seq_cap,
    unsigned int pos,
    unsigned int inner
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer * inner;
    if (gid >= total) { return; }
    unsigned int o = gid / inner;
    unsigned int i = gid - o * inner;
    arena[dst_off + (o * seq_cap + pos) * inner + i] = arena[src_off + gid];
}
