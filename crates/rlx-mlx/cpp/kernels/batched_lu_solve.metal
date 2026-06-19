// SPDX-License-Identifier: GPL-3.0-only
// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

// ─── Batched dense LU + solve, f32, one threadgroup per system ─────
//
// Workload shape that motivates this kernel:
//   A: [B, n, n]  — B independent dense systems
//   b: [B, n]     — per-system RHS
//   x: [B, n]     — per-system solution
// with B in the thousands and n in 10..NMAX — the eda-mna Monte-Carlo
// case after the per-draw MNA system is batched along axis 0.
//
// One MLX threadgroup handles one system. Each thread owns one row.
// Capacity bound: A_local + b_local must fit in threadgroup memory
// (32 KB on M-series). At f32 that's n² + n ≤ 8192 floats ⇒ n ≤ 90.
//
// **Algorithm:** Doolittle LU **with partial pivoting**, then forward
// + back substitution. Per-step pivot: tid 0 scans column k for the
// row with maximum |A[i,k]|, swaps that row with row k, then all
// threads cooperate on the rank-1 elimination. The pivot search +
// swap is serial on tid 0 — for n ≤ NMAX this is O(n) work per step
// against O(n) threads doing real elimination work in parallel after
// the barrier, so the serial bit is bounded and not the bottleneck.
// (For larger n a SIMD-reduction argmax would matter; out of scope
// for this kernel.)
//
// MLX dispatch contract (set on the host side):
//   grid       = (B, 1, 1)
//   threadgroup = (n, 1, 1)
//
// MLX-binding contract:
//   • Source is the **body** of the kernel function. MLX wraps it in
//     `[[kernel]] void custom_kernel_<name>(...)` with parameters
//     auto-generated from input/output names + dtypes.
//   • Inputs `A`, `b` are bound as `const device float*` (row-major
//     contiguous; ensure_row_contiguous=true on the host).
//   • Output `x` is bound as `device float*`.
//   • Because the source mentions `A_shape`, MLX also auto-injects
//     `const constant int* A_shape` — we read n and B from there.
//   • Built-ins `thread_position_in_threadgroup` and
//     `threadgroup_position_in_grid` are auto-detected and added with
//     the right `[[…]]` annotations.
//   • `NMAX` is supplied via the `header` arg as a #define so the
//     threadgroup-memory sizes are fixed per-dispatch-shape.

#ifndef NMAX
#define NMAX 64
#endif

// ── Workspace ────────────────────────────────────────────────────
threadgroup float Aloc[NMAX * NMAX];
threadgroup float bloc[NMAX];
// Single-slot threadgroup scratch for the per-step pivot row index.
// tid 0 writes; all threads read after the barrier.
threadgroup uint pivot_row[1];

// Read shape from MLX-injected A_shape: A is [B, n, n] row-major.
const uint B = (uint)A_shape[0];
const uint n = (uint)A_shape[1];

const uint gid = threadgroup_position_in_grid.x;
const uint tid = thread_position_in_threadgroup.x;

if (gid >= B) return;
if (tid >= n) return;

// ── Stage 1: cooperative load A[gid], b[gid] → threadgroup ─────
// Index directly off the parameter pointers — small inputs may land
// in `constant` address space (MLX uses constant for size < 8) while
// large inputs land in `device`. Keeping access inline lets the same
// kernel compile for both layouts without auto/decltype gymnastics.
for (uint j = 0; j < n; ++j) {
    Aloc[tid * NMAX + j] = A[gid * (n * n) + tid * n + j];
}
bloc[tid] = b[gid * n + tid];
threadgroup_barrier(mem_flags::mem_threadgroup);

// ── Stage 2: Doolittle LU in place, with partial pivoting ────────
// For k = 0..n-1:
//   • tid 0 finds row p in [k, n) with max |A[p, k]|, swaps rows k↔p
//     in both Aloc and bloc.
//   • Barrier so all threads see the post-swap matrix.
//   • Threads with tid > k do the rank-1 elimination on their row.
//   • Barrier so the next k sees a consistent matrix.
for (uint k = 0; k < n; ++k) {
    if (tid == 0) {
        uint  best     = k;
        float best_abs = fabs(Aloc[k * NMAX + k]);
        for (uint i = k + 1; i < n; ++i) {
            float v = fabs(Aloc[i * NMAX + k]);
            if (v > best_abs) { best = i; best_abs = v; }
        }
        pivot_row[0] = best;
        if (best != k) {
            for (uint j = 0; j < n; ++j) {
                float tmp           = Aloc[k    * NMAX + j];
                Aloc[k    * NMAX + j] = Aloc[best * NMAX + j];
                Aloc[best * NMAX + j] = tmp;
            }
            float tmp_b = bloc[k];
            bloc[k]    = bloc[best];
            bloc[best] = tmp_b;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float pivot = Aloc[k * NMAX + k];
    if (tid > k && tid < n) {
        float factor = Aloc[tid * NMAX + k] / pivot;
        Aloc[tid * NMAX + k] = factor;
        for (uint j = k + 1; j < n; ++j) {
            Aloc[tid * NMAX + j] -= factor * Aloc[k * NMAX + j];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ── Stage 3: forward solve L·y = b; L is unit-lower of Aloc ──────
// Sequential on i. For n ≤ NMAX a serial loop on tid==0 avoids
// barrier proliferation. Per-i inner dot could be SIMD-reduced if
// profiling shows this dominates.
if (tid == 0) {
    for (uint i = 0; i < n; ++i) {
        float sum = bloc[i];
        for (uint j = 0; j < i; ++j) {
            sum -= Aloc[i * NMAX + j] * bloc[j];
        }
        bloc[i] = sum;  // y, stored in place of b
    }

    // ── Stage 4: back-solve U·x = y; U is upper of Aloc inc. diag ─
    for (int i = (int)n - 1; i >= 0; --i) {
        float sum = bloc[i];
        for (uint j = (uint)i + 1; j < n; ++j) {
            sum -= Aloc[i * NMAX + j] * bloc[j];
        }
        bloc[i] = sum / Aloc[i * NMAX + i];  // x[i]
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ── Stage 5: write x out ─────────────────────────────────────────
x[gid * n + tid] = bloc[tid];
