// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Custom MSL compute kernels for element-wise + fused operations.
//!
//! Each kernel is a Metal compute pipeline. Compiled once at startup
//! from inline MSL source, dispatched via command encoder at runtime.
//!
//! Mirrors rlx-cpu/src/kernels.rs but for GPU.

use crate::device::metal_device;
use metal::{Buffer, ComputePipelineState, Library, MTLResourceOptions};
use std::sync::OnceLock;

/// Inline MSL source for all kernels — compiled once at startup.
pub const RLX_KERNELS_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Scalar activation math — one `inline float rlx_<name>_scalar(float x)` per
// activation, generated from the shared rlxsl manifest. Every activation kernel
// below (scalar f32/f16 + tuned vec4) calls these, so each activation's math
// (the A&S erf polynomial, softplus trick, …) has a single on-device definition.
// @@RLX_SCALAR_ACT_FNS@@

// Rust-`powf`-matching scalar pow (signed for negative base + integer exponent;
// bare MSL `pow` NaNs on any negative base). Generated from rlxsl so the tuned
// broadcast kernels below get the correct semantics at zero perf cost (inlined).
// @@RLX_POW_SCALAR_FN@@

// Naive sgemm: one thread per output element, one dot product each.
// C[m,n] = A[m,k] @ B[k,n]. Good baseline; tiled version below for speed.
kernel void sgemm(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& K      [[buffer(4)]],
    constant uint& N      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;
    float sum = 0.0;
    for (uint k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// F16 activations × F32 weights → F32, f32 accumulate. Mirror of `sgemm_f16w`
// for the A operand. Training's mixed-precision backward produces grad matmuls
// `A_f16^T @ grad_f32` (activation cast to f16, upstream grad in f32); routing
// those through the plain f32 `sgemm` would reinterpret the 2-byte f16 A bytes
// as f32 and corrupt the result. This reads A as half and accumulates in f32,
// so K-long reductions (e.g. K=B*T=1024) never overflow f16's ±65504 range.
kernel void sgemm_f16a(
    device const half* A  [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& K      [[buffer(4)]],
    constant uint& N      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;
    float sum = 0.0;
    for (uint k = 0; k < K; ++k) {
        sum += float(A[row * K + k]) * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// F32 activations × F16 weights → F32 (Zonos Metal Linear). Avoids casting
// the full weight matrix to F32 on every decode step.
kernel void sgemm_f16w(
    device const float* A [[buffer(0)]],
    device const half* B  [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& K      [[buffer(4)]],
    constant uint& N      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;
    float sum = 0.0;
    for (uint k = 0; k < K; ++k) {
        sum += A[row * K + k] * float(B[k * N + col]);
    }
    C[row * N + col] = sum;
}

// Small-M F16-weight GEMM (CFG decode: M=2). One thread per output column;
// all M rows share each B load — padded simdgroup wastes 6/8 A rows when M=2.
kernel void sgemm_f16w_small_m(
    device const float* A [[buffer(0)]],
    device const half* B  [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& K      [[buffer(4)]],
    constant uint& N      [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint col = gid;
    if (col >= N || M == 0u || M > 4u) return;

    // Hot path: M==2 (Zonos CFG) — no per-k row branches.
    if (M == 2u) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        uint k = 0;
        for (; k + 3u < K; k += 4u) {
            float4 bv = float4(
                float(B[(k + 0u) * N + col]),
                float(B[(k + 1u) * N + col]),
                float(B[(k + 2u) * N + col]),
                float(B[(k + 3u) * N + col]));
            float4 a0 = *(device const float4*)(A + k);
            float4 a1 = *(device const float4*)(A + K + k);
            acc0 += a0.x * bv.x + a0.y * bv.y + a0.z * bv.z + a0.w * bv.w;
            acc1 += a1.x * bv.x + a1.y * bv.y + a1.z * bv.z + a1.w * bv.w;
        }
        for (; k < K; ++k) {
            float b = float(B[k * N + col]);
            acc0 += A[k] * b;
            acc1 += A[K + k] * b;
        }
        C[col] = acc0;
        C[N + col] = acc1;
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    uint k = 0;
    for (; k + 3u < K; k += 4u) {
        float b0 = float(B[(k + 0u) * N + col]);
        float b1 = float(B[(k + 1u) * N + col]);
        float b2 = float(B[(k + 2u) * N + col]);
        float b3 = float(B[(k + 3u) * N + col]);
        float4 bv = float4(b0, b1, b2, b3);
        {
            float4 a = *(device const float4*)(A + k);
            acc0 += a.x * bv.x + a.y * bv.y + a.z * bv.z + a.w * bv.w;
        }
        if (M > 1u) {
            float4 a = *(device const float4*)(A + K + k);
            acc1 += a.x * bv.x + a.y * bv.y + a.z * bv.z + a.w * bv.w;
        }
        if (M > 2u) {
            float4 a = *(device const float4*)(A + 2u * K + k);
            acc2 += a.x * bv.x + a.y * bv.y + a.z * bv.z + a.w * bv.w;
        }
        if (M > 3u) {
            float4 a = *(device const float4*)(A + 3u * K + k);
            acc3 += a.x * bv.x + a.y * bv.y + a.z * bv.z + a.w * bv.w;
        }
    }
    for (; k < K; ++k) {
        float b = float(B[k * N + col]);
        acc0 += A[k] * b;
        if (M > 1u) acc1 += A[K + k] * b;
        if (M > 2u) acc2 += A[2u * K + k] * b;
        if (M > 3u) acc3 += A[3u * K + k] * b;
    }
    C[col] = acc0;
    if (M > 1u) C[N + col] = acc1;
    if (M > 2u) C[2u * N + col] = acc2;
    if (M > 3u) C[3u * N + col] = acc3;
}

// M=1 F16-weight GEMV with K-splitting. `sgemm_f16w_small_m` launches one
// thread per output column — for the decode projections (N ~1-3k) that is only
// ~1-3k threads, too few to saturate memory bandwidth (~37 GB/s of ~273 peak).
// Here KSPLIT threads cooperate per column: each strides k by KSPLIT and sums a
// K/KSPLIT slice, then the partials reduce through threadgroup memory. The
// threadgroup is (32 cols × KSPLIT), and a simdgroup is the 32 columns for one
// `ks`, so `B[k*N + col .. col+31]` stays fully coalesced (KSPLIT× the threads
// at the same DRAM traffic). A: [1,K] f32, B: [K,N] f16, C: [1,N] f32.
kernel void gemv_f16w_splitk(
    device const float* A [[buffer(0)]],
    device const half*  B [[buffer(1)]],
    device float*       C [[buffer(2)]],
    constant uint& K      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    constexpr uint KSPLIT = 32u;
    threadgroup float2 partial[32][32];
    // Two columns per thread via half2 loads: 32 threads × 4 B = 128-byte
    // coalesced transactions (full Apple cache line) vs 64 B for scalar half.
    // Requires N even & col-pair aligned — the dispatch only routes here for
    // even N (all transformer projection widths); odd N falls to scalar.
    uint col = tgid.x * 64u + tid.x * 2u;
    uint ks = tid.y;
    float2 acc = float2(0.0f);
    if (col < N) {
        device const half2* Bp = (device const half2*)(B + col);
        for (uint k = ks; k < K; k += KSPLIT) {
            float a = A[k];
            half2 b = Bp[k * (N / 2u)];
            acc.x += a * float(b.x);
            acc.y += a * float(b.y);
        }
    }
    partial[tid.x][ks] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks == 0u && col < N) {
        float2 s = float2(0.0f);
        for (uint j = 0u; j < KSPLIT; ++j) s += partial[tid.x][j];
        C[col] = s.x;
        if (col + 1u < N) {
            C[col + 1u] = s.y;
        }
    }
}

// K-partitioned F16 GEMV — CEILING PROBE for small-N decode projections.
// gemv_f16w_splitk launches only N/64 threadgroups (o_proj/down_proj n=1024 →
// 16 tgs), under-occupying the GPU. This variant adds a Kparts z-axis so each
// (col-block, k-partition) tg attends a K-slice and atomic-adds its partial
// into a pre-zeroed C — raising the tg count to (N/64)*Kparts. NOTE: float
// atomic-add is order-nondeterministic, so this is a measurement probe; a
// production version must reduce partials deterministically (scratch + fixed
// order) to stay token-identical.
kernel void gemv_f16w_kpart(
    device const float* A [[buffer(0)]],
    device const half*  B [[buffer(1)]],
    device atomic_float* C [[buffer(2)]],
    constant uint& K      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& Kparts [[buffer(5)]],
    uint3 tid  [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    constexpr uint KSPLIT = 32u;
    threadgroup float2 partial[32][32];
    uint col = tgid.x * 64u + tid.x * 2u;
    uint ks = tid.y;
    uint kchunk = (K + Kparts - 1u) / Kparts;
    uint kstart = tgid.z * kchunk;
    uint kend = min(kstart + kchunk, K);
    float2 acc = float2(0.0f);
    if (col < N) {
        device const half2* Bp = (device const half2*)(B + col);
        for (uint k = kstart + ks; k < kend; k += KSPLIT) {
            float a = A[k];
            half2 b = Bp[k * (N / 2u)];
            acc.x += a * float(b.x);
            acc.y += a * float(b.y);
        }
    }
    partial[tid.x][ks] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks == 0u && col < N) {
        float2 s = float2(0.0f);
        for (uint j = 0u; j < KSPLIT; ++j) s += partial[tid.x][j];
        atomic_fetch_add_explicit(&C[col], s.x, memory_order_relaxed);
        if (col + 1u < N) atomic_fetch_add_explicit(&C[col + 1u], s.y, memory_order_relaxed);
    }
}

// Zero N floats of an atomic_float region (pre-zero for gemv_f16w_kpart).
kernel void gemv_zero_f32(
    device atomic_float* C [[buffer(0)]],
    constant uint& N      [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < N) atomic_store_explicit(&C[gid], 0.0f, memory_order_relaxed);
}

// Padded simdgroup with F16 B (same staging as sgemm_simd_padded).
kernel void sgemm_simd_padded_f16w(
    device const float* A [[buffer(0)]],
    device const half* B  [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& K      [[buffer(4)]],
    constant uint& N      [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint slid  [[thread_index_in_simdgroup]]
) {
    uint row_base = tgid.y * 8;
    uint col_base = tgid.x * 8;
    threadgroup float A_pad[64];
    threadgroup float B_pad[64];
    simdgroup_float8x8 a, b, c;
    c = simdgroup_float8x8(0.0f);
    for (uint k = 0; k < K; k += 8) {
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 32 + slid;
            uint ar = idx / 8;
            uint ac = idx % 8;
            uint src_row = row_base + ar;
            uint src_col = k + ac;
            float v = (src_row < M && src_col < K) ? A[src_row * K + src_col] : 0.0f;
            A_pad[idx] = v;
        }
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 32 + slid;
            uint br = idx / 8;
            uint bc = idx % 8;
            uint src_row = k + br;
            uint src_col = col_base + bc;
            float v = (src_row < K && src_col < N) ? float(B[src_row * N + src_col]) : 0.0f;
            B_pad[idx] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(a, A_pad, 8);
        simdgroup_load(b, B_pad, 8);
        simdgroup_multiply_accumulate(c, a, b, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Write back with bounds (same as sgemm_simd_padded).
    threadgroup float C_pad[64];
    simdgroup_store(c, C_pad, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 2; ++i) {
        uint idx = i * 32 + slid;
        uint cr = idx / 8;
        uint cc = idx % 8;
        uint out_row = row_base + cr;
        uint out_col = col_base + cc;
        if (out_row < M && out_col < N) {
            C[out_row * N + out_col] = C_pad[idx];
        }
    }
}

// 64×64 F16-weight GEMM tile (8 simdgroups) — the prefill fix.
// sgemm_simd_padded_f16w computes one 8×8 output tile per threadgroup (1 MMA
// per K-step, A re-loaded per column block) → ~650 GFLOPS, 4–10× slower than
// MPS per the mps_re_bench RE. This computes a 64×64 tile per threadgroup:
// 8 simdgroups × 8 column-block accumulators, with half→float A/B staged
// cooperatively by 256 threads into threadgroup memory (bounds-zeroed, so any
// M/N/K works) — each staged A/B slab feeds 64 MMAs/K-step, the register-reuse
// MPS exploits. F32 accumulate (more accurate than MPS's f16). Grid
// (ceil(N/64), ceil(M/64)); dispatch with 256 threads/threadgroup.
kernel void sgemm_wide8x64_f16w(
    device const float* A [[buffer(0)]],
    device const half*  B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& K      [[buffer(4)]],
    constant uint& N      [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint slid  [[thread_index_in_simdgroup]]
) {
    constexpr uint TR = 64u, TC = 64u, NACC = 8u;
    uint row0 = tgid.y * TR;
    uint col0 = tgid.x * TC;
    uint tix = sgid * 32u + slid; // 0..255
    threadgroup float As[64u * 8u]; // 64 rows(M) × 8 (K)
    threadgroup float Bs[8u * 64u]; // 8 (K) × 64 cols(N)
    threadgroup float Cs[64u * 64u]; // 64×64 output tile
    simdgroup_float8x8 acc[NACC];
    for (uint j = 0; j < NACC; ++j) acc[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint kk = 0; kk < K; kk += 8u) {
        // Stage A [64×8] and B [8×64], 512 elems each, 256 threads → 2 each.
        for (uint i = 0; i < 2u; ++i) {
            uint idx = i * 256u + tix;
            uint ar = idx / 8u, ac = idx % 8u;
            uint sr = row0 + ar, sc = kk + ac;
            As[idx] = (sr < M && sc < K) ? A[sr * K + sc] : 0.0f;
            uint br = idx / 64u, bc = idx % 64u;
            uint tr = kk + br, tc = col0 + bc;
            Bs[idx] = (tr < K && tc < N) ? float(B[tr * N + tc]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 a, b;
        simdgroup_load(a, As + sgid * 8u * 8u, 8); // this simdgroup's 8 rows
        for (uint j = 0; j < NACC; ++j) {
            simdgroup_load(b, Bs + j * 8u, 64);
            simdgroup_multiply_accumulate(acc[j], a, b, acc[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = 0; j < NACC; ++j) simdgroup_store(acc[j], Cs + (sgid * 8u) * 64u + j * 8u, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Guarded write Cs → C: 4096 elems / 256 threads = 16 each.
    for (uint i = 0; i < 16u; ++i) {
        uint idx = i * 256u + tix;
        uint cr = idx / 64u, cc = idx % 64u;
        uint orow = row0 + cr, ocol = col0 + cc;
        if (orow < M && ocol < N) C[orow * N + ocol] = Cs[idx];
    }
}

// ── Half-precision (f16) variants ──────────────────────────────────────
// Apple Silicon supports simdgroup_half8x8 — same tensor unit pipeline
// but 2× peak FLOPs and ½ memory bandwidth vs simdgroup_float8x8.

// Tiled half-precision matmul: 32x32 output per TG, 16 simdgroups cooperate.
// Inputs A, B and output C all in f16; bias also f16 if provided.
kernel void hgemm_simd_4x4(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C       [[buffer(2)]],
    constant uint& M     [[buffer(3)]],
    constant uint& K     [[buffer(4)]],
    constant uint& N     [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint slid  [[thread_index_in_simdgroup]]
) {
    uint sg_row = sgid / 4;
    uint sg_col = sgid % 4;
    uint tg_row_base = tgid.y * 32;
    uint tg_col_base = tgid.x * 32;

    threadgroup half A_tg[32 * 32];
    threadgroup half B_tg[32 * 32];

    simdgroup_half8x8 a, b;
    simdgroup_half8x8 c = simdgroup_half8x8(0.0h);

    for (uint kk = 0; kk < K; kk += 32) {
        uint linear = sgid * 32 + slid;
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 512 + linear;
            uint ar = idx / 32, ac = idx % 32;
            A_tg[idx] = A[(tg_row_base + ar) * K + (kk + ac)];
        }
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 512 + linear;
            uint br = idx / 32, bc = idx % 32;
            B_tg[idx] = B[(kk + br) * N + (tg_col_base + bc)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_inner = 0; k_inner < 32; k_inner += 8) {
            simdgroup_load(a, &A_tg[sg_row * 8 * 32 + k_inner], 32);
            simdgroup_load(b, &B_tg[k_inner * 32 + sg_col * 8], 32);
            simdgroup_multiply_accumulate(c, a, b, c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint out_row = tg_row_base + sg_row * 8;
    uint out_col = tg_col_base + sg_col * 8;
    simdgroup_store(c, &C[out_row * N + out_col], N);
}

// Half-precision matmul + bias + activation fused.
kernel void hgemm_simd_4x4_bias(
    device const half* A     [[buffer(0)]],
    device const half* B     [[buffer(1)]],
    device const half* bias  [[buffer(2)]],
    device half* C           [[buffer(3)]],
    constant uint& M         [[buffer(4)]],
    constant uint& K         [[buffer(5)]],
    constant uint& N         [[buffer(6)]],
    constant uint& act_kind  [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint slid  [[thread_index_in_simdgroup]]
) {
    uint sg_row = sgid / 4;
    uint sg_col = sgid % 4;
    uint tg_row_base = tgid.y * 32;
    uint tg_col_base = tgid.x * 32;

    threadgroup half A_tg[32 * 32];
    threadgroup half B_tg[32 * 32];

    simdgroup_half8x8 a, b;
    simdgroup_half8x8 c = simdgroup_half8x8(0.0h);

    for (uint kk = 0; kk < K; kk += 32) {
        uint linear = sgid * 32 + slid;
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 512 + linear;
            uint ar = idx / 32, ac = idx % 32;
            A_tg[idx] = A[(tg_row_base + ar) * K + (kk + ac)];
        }
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 512 + linear;
            uint br = idx / 32, bc = idx % 32;
            B_tg[idx] = B[(kk + br) * N + (tg_col_base + bc)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_inner = 0; k_inner < 32; k_inner += 8) {
            simdgroup_load(a, &A_tg[sg_row * 8 * 32 + k_inner], 32);
            simdgroup_load(b, &B_tg[k_inner * 32 + sg_col * 8], 32);
            simdgroup_multiply_accumulate(c, a, b, c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup half tile[16 * 64];
    simdgroup_store(c, &tile[sgid * 64], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint out_row_base = tg_row_base + sg_row * 8;
    uint out_col_base = tg_col_base + sg_col * 8;
    for (uint i = 0; i < 2; ++i) {
        uint idx = i * 32 + slid;
        uint r = idx / 8;
        uint cc = idx % 8;
        // Promote to fp32 for activation math (more accurate)
        float v = float(tile[sgid * 64 + idx]) + float(bias[out_col_base + cc]);
        if (act_kind == 1) {
            v = rlx_gelu_scalar(v);
        } else if (act_kind == 2) {
            v = rlx_silu_scalar(v);
        }
        C[(out_row_base + r) * N + (out_col_base + cc)] = half(v);
    }
}

// ── Half-precision element-wise + reduction kernels ─────────────────

kernel void bias_add_h(
    device half* data       [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    constant uint& m        [[buffer(2)]],
    constant uint& n        [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y, col = gid.x;
    if (row >= m || col >= n) return;
    data[row * n + col] += bias[col];
}

// Core scalar-activation f16 in-place kernels — generated once from the shared
// rlxsl manifest (was hand-written MSL re-inlining the A&S erf polynomial with
// drift: unclamped erf arg + ties-away `round`). Injected by msl_source().
// @@RLX_ACT_INPLACE_H@@

// f16 input, f32 reduction, f16 output (mixed precision LayerNorm)
kernel void layer_norm_h(
    device const half* input [[buffer(0)]],
    device const half* gamma [[buffer(1)]],
    device const half* beta  [[buffer(2)]],
    device half* output      [[buffer(3)]],
    constant uint& h         [[buffer(4)]],
    constant float& eps      [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float partial_sum[256];
    threadgroup float partial_sumsq[256];

    float local_sum = 0.0f, local_sumsq = 0.0f;
    for (uint i = tid; i < h; i += tsize) {
        float v = float(input[row * h + i]);
        local_sum += v;
        local_sumsq += v * v;
    }
    partial_sum[tid] = local_sum;
    partial_sumsq[tid] = local_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
            partial_sumsq[tid] += partial_sumsq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = partial_sum[0] / float(h);
    float var = fmax(0.0f, partial_sumsq[0] / float(h) - mean * mean);
    float inv_std = rsqrt(var + eps);

    for (uint i = tid; i < h; i += tsize) {
        float v = float(input[row * h + i]);
        output[row * h + i] = half((v - mean) * inv_std * float(gamma[i]) + float(beta[i]));
    }
}

kernel void fused_residual_ln_h(
    device const half* x      [[buffer(0)]],
    device const half* res    [[buffer(1)]],
    device const half* gamma  [[buffer(2)]],
    device const half* beta   [[buffer(3)]],
    device half* out          [[buffer(4)]],
    constant uint& h          [[buffer(5)]],
    constant float& eps       [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float partial_sum[256];
    threadgroup float partial_sumsq[256];

    float local_sum = 0.0f, local_sumsq = 0.0f;
    for (uint i = tid; i < h; i += tsize) {
        float v = float(x[row * h + i]) + float(res[row * h + i]);
        local_sum += v;
        local_sumsq += v * v;
    }
    partial_sum[tid] = local_sum;
    partial_sumsq[tid] = local_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
            partial_sumsq[tid] += partial_sumsq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = partial_sum[0] / float(h);
    float var = fmax(0.0f, partial_sumsq[0] / float(h) - mean * mean);
    float inv_std = rsqrt(var + eps);

    for (uint i = tid; i < h; i += tsize) {
        float v = float(x[row * h + i]) + float(res[row * h + i]);
        out[row * h + i] = half((v - mean) * inv_std * float(gamma[i]) + float(beta[i]));
    }
}

kernel void fused_residual_rms_norm_h(
    device const char* arena  [[buffer(0)]],
    constant ulong& x_off     [[buffer(1)]],
    constant ulong& res_off   [[buffer(2)]],
    constant ulong& g_off     [[buffer(3)]],
    constant ulong& b_off     [[buffer(4)]],
    constant ulong& out_off   [[buffer(5)]],
    constant uint& h          [[buffer(6)]],
    constant float& eps       [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    device const half* x = (device const half*)(arena + x_off);
    device const half* res = (device const half*)(arena + res_off);
    device const half* gamma = (device const half*)(arena + g_off);
    device const half* beta = (device const half*)(arena + b_off);
    device half* out = (device half*)(arena + out_off);
    threadgroup float partial_sumsq[256];
    float local_sumsq = 0.0f;
    for (uint i = tid; i < h; i += tsize) {
        float v = float(x[row * h + i]) + float(res[row * h + i]);
        local_sumsq += v * v;
    }
    partial_sumsq[tid] = local_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sumsq[tid] += partial_sumsq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(partial_sumsq[0] / float(h) + eps);
    for (uint i = tid; i < h; i += tsize) {
        float v = float(x[row * h + i]) + float(res[row * h + i]);
        out[row * h + i] = half(v * inv_rms * float(gamma[i]) + float(beta[i]));
    }
}

kernel void elem_add_h(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c       [[buffer(2)]],
    constant uint& len   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    c[gid] = a[gid] + b[gid];
}

kernel void elem_mul_h(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c       [[buffer(2)]],
    constant uint& len   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    c[gid] = a[gid] * b[gid];
}

kernel void elem_sub_h(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c       [[buffer(2)]],
    constant uint& len   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    c[gid] = a[gid] - b[gid];
}

kernel void elem_div_h(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c       [[buffer(2)]],
    constant uint& len   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    c[gid] = a[gid] / b[gid];
}

kernel void elem_max_h(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c       [[buffer(2)]],
    constant uint& len   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) { if (gid >= len) return; c[gid] = max(a[gid], b[gid]); }

kernel void elem_min_h(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c       [[buffer(2)]],
    constant uint& len   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) { if (gid >= len) return; c[gid] = min(a[gid], b[gid]); }

kernel void elem_pow_h(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c       [[buffer(2)]],
    constant uint& len   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    c[gid] = half(pow(float(a[gid]), float(b[gid])));
}

kernel void gather_axis0_h(
    device const half* table [[buffer(0)]],
    device const half* idx   [[buffer(1)]],
    device half* out         [[buffer(2)]],
    constant uint& num_idx   [[buffer(3)]],
    constant uint& trailing  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y, j = gid.x;
    if (i >= num_idx || j >= trailing) return;
    uint row = uint(float(idx[i]));
    out[i * trailing + j] = table[row * trailing + j];
}

kernel void narrow_lastax_h(
    device const char* arena_src [[buffer(0)]],
    device char* arena_dst       [[buffer(1)]],
    constant uint& outer     [[buffer(2)]],
    constant uint& src_axis  [[buffer(3)]],
    constant uint& start     [[buffer(4)]],
    constant uint& len       [[buffer(5)]],
    constant ulong& src_byte_off [[buffer(6)]],
    constant ulong& dst_byte_off [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    device const half* src = (device const half*)(arena_src + src_byte_off);
    device half* dst       = (device half*)(arena_dst + dst_byte_off);
    uint i = gid.y, j = gid.x;
    if (i >= outer || j >= len) return;
    dst[i * len + j] = src[i * src_axis + start + j];
}

kernel void sdpa_h(
    device const half* Q    [[buffer(0)]],
    device const half* K    [[buffer(1)]],
    device const half* V    [[buffer(2)]],
    device const half* M    [[buffer(3)]],
    device half* OUT        [[buffer(4)]],
    constant uint& batch      [[buffer(5)]],
    constant uint& seq        [[buffer(6)]],
    constant uint& heads      [[buffer(7)]],
    constant uint& head_dim   [[buffer(8)]],
    constant uint& seq_stride [[buffer(9)]],
    constant uint& mask_kind  [[buffer(10)]],
    constant uint& seq_k      [[buffer(11)]],  // unused; mirrors sdpa signature
    constant uint& k_stride   [[buffer(12)]],  // unused; mirrors sdpa signature
    constant uint& bhsd       [[buffer(13)]],  // unused; mirrors sdpa signature
    constant uint& window     [[buffer(14)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tsize  [[threads_per_threadgroup]]
) {
    (void)seq_k; (void)k_stride; (void)bhsd;  // accepted to share encode_sdpa layout
    threadgroup float scores[64 * 64];
    threadgroup float row_max;
    threadgroup float row_sum;

    uint bi = tgid_x / heads;
    uint hi = tgid_x % heads;
    if (bi >= batch) return;

    uint hs = heads * head_dim;
    float scale = rsqrt(float(head_dim));
    uint per_batch_stride = seq_stride * hs;

    uint total = seq * seq;
    for (uint idx = tid; idx < total; idx += tsize) {
        uint qi = idx / seq;
        uint ki = idx % seq;
        float dot = 0.0f;
        uint q_base = bi * per_batch_stride + qi * hs + hi * head_dim;
        uint k_base = bi * per_batch_stride + ki * hs + hi * head_dim;
        for (uint d = 0; d < head_dim; ++d) {
            dot += float(Q[q_base + d]) * float(K[k_base + d]);
        }
        float s = dot * scale;
        if (mask_kind == 1u) {
            if (ki > qi) s = -1e9f;
        } else if (mask_kind == 2u) {
            if (float(M[bi * seq_stride + ki]) < 0.5f) s = -1e9f;
        } else if (mask_kind == 4u) {
            // Lq == Lk here (sdpa_h is prefill-only), so abs_q == qi.
            uint lo = qi > window ? qi - window : 0u;
            if (ki < lo || ki > qi) s = -1e9f;
        }
        scores[qi * seq + ki] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint qi = 0; qi < seq; ++qi) {
        if (tid == 0) {
            float mx = -1e30f;
            for (uint ki = 0; ki < seq; ++ki) {
                mx = max(mx, scores[qi * seq + ki]);
            }
            row_max = mx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float sum = 0.0f;
            for (uint ki = 0; ki < seq; ++ki) {
                float e = precise::exp(scores[qi * seq + ki] - row_max);
                scores[qi * seq + ki] = e;
                sum += e;
            }
            row_sum = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ki = tid; ki < seq; ki += tsize) {
            scores[qi * seq + ki] /= row_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint out_total = seq * head_dim;
    for (uint idx = tid; idx < out_total; idx += tsize) {
        uint qi = idx / head_dim;
        uint d = idx % head_dim;
        float acc = 0.0f;
        for (uint ki = 0; ki < seq; ++ki) {
            uint v_base = bi * per_batch_stride + ki * hs + hi * head_dim;
            acc += scores[qi * seq + ki] * float(V[v_base + d]);
        }
        uint o_base = bi * per_batch_stride + qi * hs + hi * head_dim;
        OUT[o_base + d] = half(acc);
    }
}

kernel void rope_h(
    device const half* x   [[buffer(0)]],
    device const half* cos [[buffer(1)]],
    device const half* sin [[buffer(2)]],
    device half* out       [[buffer(3)]],
    constant uint& batch          [[buffer(4)]],
    constant uint& seq            [[buffer(5)]],
    constant uint& hidden         [[buffer(6)]],
    constant uint& head_dim       [[buffer(7)]],
    constant uint& src_row_stride [[buffer(8)]],
    constant uint& seq_stride     [[buffer(9)]],
    constant uint& n_rot          [[buffer(10)]],
    constant uint& cos_per_token  [[buffer(11)]],
    constant uint& interleaved    [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint half_dh = head_dim / 2;
    uint rot_half = n_rot / 2;
    if (gid.x >= head_dim) return;

    uint bs = gid.z;
    uint bi = bs / seq;
    uint si = bs % seq;
    if (bi >= batch || si >= seq) return;

    uint nh = hidden / head_dim;
    uint hi = gid.y;
    if (hi >= nh) return;

    // Per-seq-position table by default; per global token for ragged decode.
    uint cos_row = (cos_per_token != 0u) ? bs : si;

    // PLAN L1 — `seq_stride` is the compile-time full extent for buffer
    // offsets; `seq` is the (possibly scaled) iteration bound.
    uint src_base = bi * seq_stride * src_row_stride + si * src_row_stride + hi * head_dim;
    uint dst_base = bi * seq_stride * hidden + si * hidden + hi * head_dim;
    uint d = gid.x;
    if (interleaved != 0u) {
        // GPT-J / llama.cpp-NORM: pairs are adjacent (2d, 2d+1). cos/sin
        // row index is the freq d (0..rot_half).
        if (d < rot_half) {
            uint a = 2u * d;
            uint b = 2u * d + 1u;
            float x1 = float(x[src_base + a]);
            float x2 = float(x[src_base + b]);
            float c = float(cos[cos_row * half_dh + d]);
            float s = float(sin[cos_row * half_dh + d]);
            out[dst_base + a] = half(x1 * c - x2 * s);
            out[dst_base + b] = half(x2 * c + x1 * s);
        } else if (d >= n_rot) {
            out[dst_base + d] = x[src_base + d];
        }
    } else if (d < rot_half) {
        float x1 = float(x[src_base + d]);
        float x2 = float(x[src_base + rot_half + d]);
        float c = float(cos[cos_row * half_dh + d]);
        float s = float(sin[cos_row * half_dh + d]);
        out[dst_base + d] = half(x1 * c - x2 * s);
        out[dst_base + rot_half + d] = half(x2 * c + x1 * s);
    } else if (d >= n_rot) {
        out[dst_base + d] = x[src_base + d];
    }
}

// Native f32 fused-attention core for `Op::FusedAttentionBlock` (no-bias
// path). Reads the PACKED QKV projection `[B,S,3*inner]` (per token
// `[Q(inner)|K(inner)|V(inner)]`, heads interleaved), applies optional NeoX
// RoPE to Q/K inline, runs softmax SDPA with the score matrix resident in
// threadgroup memory (one threadgroup per batch·head), and writes the
// attention output `[B,S,inner]`. Collapses narrow×3 + transpose×3 + rope×2
// + attention into a single dispatch; the QKV / out projections stay GEMMs.
// mask_kind: 0=None, 1=Causal, 2=Custom (binary [B,S], <0.5 ⇒ drop).
kernel void fused_attn_block(
    device const float* QKV  [[buffer(0)]],
    device const float* M    [[buffer(1)]],
    device const float* COS  [[buffer(2)]],
    device const float* SIN  [[buffer(3)]],
    device float* OUT        [[buffer(4)]],
    constant uint& batch     [[buffer(5)]],
    constant uint& seq       [[buffer(6)]],
    constant uint& heads     [[buffer(7)]],
    constant uint& head_dim  [[buffer(8)]],
    constant uint& mask_kind [[buffer(9)]],
    constant uint& scale_bits[[buffer(10)]],
    constant uint& has_rope  [[buffer(11)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float scores[64 * 64];   // seq ≤ 64 (gated in rlx-metal)
    uint inner = heads * head_dim;
    uint bi = tgid / heads;
    uint hi = tgid % heads;
    if (bi >= batch) return;
    float scale = as_type<float>(scale_bits);
    uint half_d = head_dim / 2;
    uint tok = 3u * inner;                // per-token stride in QKV

    uint total = seq * seq;
    for (uint idx = tid; idx < total; idx += tsize) {
        uint qi = idx / seq;
        uint ki = idx % seq;
        uint qb = (bi * seq + qi) * tok + hi * head_dim;            // Q
        uint kb = (bi * seq + ki) * tok + inner + hi * head_dim;    // K
        float dot = 0.0f;
        if (has_rope != 0u) {
            uint qc = qi * half_d;
            uint kc = ki * half_d;
            for (uint i = 0; i < half_d; ++i) {
                float q1 = QKV[qb + i],        q2 = QKV[qb + half_d + i];
                float k1 = QKV[kb + i],        k2 = QKV[kb + half_d + i];
                float cq = COS[qc + i], sq = SIN[qc + i];
                float ck = COS[kc + i], sk = SIN[kc + i];
                float qr1 = q1 * cq - q2 * sq, qr2 = q2 * cq + q1 * sq;
                float kr1 = k1 * ck - k2 * sk, kr2 = k2 * ck + k1 * sk;
                dot += qr1 * kr1 + qr2 * kr2;
            }
        } else {
            for (uint d = 0; d < head_dim; ++d) dot += QKV[qb + d] * QKV[kb + d];
        }
        float s = dot * scale;
        if (mask_kind == 1u) {
            if (ki > qi) s = -1e9f;
        } else if (mask_kind == 2u) {
            if (M[bi * seq + ki] < 0.5f) s = -1e9f;
        }
        scores[qi * seq + ki] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint qi = tid; qi < seq; qi += tsize) {
        float mx = -1e30f;
        for (uint ki = 0; ki < seq; ++ki) mx = max(mx, scores[qi * seq + ki]);
        float sum = 0.0f;
        for (uint ki = 0; ki < seq; ++ki) {
            float e = precise::exp(scores[qi * seq + ki] - mx);
            scores[qi * seq + ki] = e;
            sum += e;
        }
        float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (uint ki = 0; ki < seq; ++ki) scores[qi * seq + ki] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint otot = seq * head_dim;
    for (uint idx = tid; idx < otot; idx += tsize) {
        uint qi = idx / head_dim;
        uint d = idx % head_dim;
        float acc = 0.0f;
        for (uint ki = 0; ki < seq; ++ki) {
            uint vb = (bi * seq + ki) * tok + 2u * inner + hi * head_dim;
            acc += scores[qi * seq + ki] * QKV[vb + d];
        }
        OUT[(bi * seq + qi) * inner + hi * head_dim + d] = acc;
    }
}

// Cast f32 → f16 (used at I/O boundary)
kernel void cast_f32_to_f16(
    device const float* src [[buffer(0)]],
    device half* dst        [[buffer(1)]],
    constant uint& len      [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    dst[gid] = half(src[gid]);
}

// Cast f16 → f32 (used at I/O boundary)
kernel void cast_f16_to_f32(
    device const half* src [[buffer(0)]],
    device float* dst      [[buffer(1)]],
    constant uint& len     [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    dst[gid] = float(src[gid]);
}

// Plain f32 buffer copy — used for Reshape/Expand thunks when we want
// to stay on the shared compute encoder instead of switching to a blit
// encoder (encoder-switch overhead dominates for small ops).
kernel void copy_f32(
    device const char* arena [[buffer(0)]],
    constant ulong& src_byte_off [[buffer(1)]],
    constant ulong& dst_byte_off [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    device const float* src = (device const float*)(arena + src_byte_off);
    device float* dst = (device float*)(arena + dst_byte_off);
    dst[gid] = src[gid];
}

kernel void copy4(
    device const char* arena [[buffer(0)]],
    constant ulong& src_byte_off [[buffer(1)]],
    constant ulong& dst_byte_off [[buffer(2)]],
    constant uint& len4 [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len4) return;
    device const packed_float4* src = (device const packed_float4*)(arena + src_byte_off);
    device packed_float4* dst = (device packed_float4*)(arena + dst_byte_off);
    dst[gid] = src[gid];
}

// SIMD-group matrix sgemm: uses Apple Silicon's dedicated tensor units.
// One simdgroup computes an 8x8 output tile via simdgroup_multiply_accumulate.
// Threadgroup has 32 threads = 1 simdgroup, computing one 8x8 tile of C.
// For larger output, dispatch more threadgroups.
//
// All dimensions must be multiples of 8 for this kernel. Caller is responsible
// for routing non-multiple-of-8 cases to the scalar tiled fallback.
kernel void sgemm_simd(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& K      [[buffer(4)]],
    constant uint& N      [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]]
) {
    uint row_base = tgid.y * 8;
    uint col_base = tgid.x * 8;
    if (row_base >= M || col_base >= N) return;

    simdgroup_float8x8 a;
    simdgroup_float8x8 b;
    simdgroup_float8x8 c;
    c = simdgroup_float8x8(0.0f);

    for (uint k = 0; k < K; k += 8) {
        simdgroup_load(a, A + row_base * K + k, K);
        simdgroup_load(b, B + k * N + col_base, N);
        simdgroup_multiply_accumulate(c, a, b, c);
    }

    simdgroup_store(c, C + row_base * N + col_base, N);
}

// High-throughput simdgroup matmul: 32x32 output per threadgroup,
// 4x4 = 16 simdgroups cooperate through threadgroup memory.
// Each B element is reused 4× across rows of simdgroups; each A element 4× across cols.
// K loaded in 32-wide stripes into threadgroup memory.
//
// Requires M%32==K%32==N%32==0. Falls back to sgemm_simd for smaller dims.
kernel void sgemm_simd_4x4(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& K      [[buffer(4)]],
    constant uint& N      [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint slid  [[thread_index_in_simdgroup]]
) {
    // 4x4 simdgroup grid within threadgroup
    uint sg_row = sgid / 4;  // 0..3
    uint sg_col = sgid % 4;  // 0..3

    uint tg_row_base = tgid.y * 32;
    uint tg_col_base = tgid.x * 32;

    threadgroup float A_tg[32 * 32];  // 4 KB
    threadgroup float B_tg[32 * 32];  // 4 KB

    simdgroup_float8x8 a, b, c;
    c = simdgroup_float8x8(0.0f);

    for (uint kk = 0; kk < K; kk += 32) {
        // Cooperative load: 16 simdgroups × 32 threads = 512 threads
        // load 32×32 A tile and 32×32 B tile (1024 floats each = 4 elements per thread)
        uint linear = sgid * 32 + slid; // 0..511
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 512 + linear;
            uint ar = idx / 32;
            uint ac = idx % 32;
            A_tg[idx] = A[(tg_row_base + ar) * K + (kk + ac)];
        }
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 512 + linear;
            uint br = idx / 32;
            uint bc = idx % 32;
            B_tg[idx] = B[(kk + br) * N + (tg_col_base + bc)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4 inner-K steps of 8 each, accumulating into c
        for (uint k_inner = 0; k_inner < 32; k_inner += 8) {
            simdgroup_load(a, &A_tg[sg_row * 8 * 32 + k_inner], 32);
            simdgroup_load(b, &B_tg[k_inner * 32 + sg_col * 8], 32);
            simdgroup_multiply_accumulate(c, a, b, c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint out_row = tg_row_base + sg_row * 8;
    uint out_col = tg_col_base + sg_col * 8;
    simdgroup_store(c, &C[out_row * N + out_col], N);
}

// Register-blocked simdgroup GEMM family (plain + split-K) — generated per tile
// config by `sgemm_tile_variant!` / `sgemm_splitk_variant!` (see msl_source). Each
// config is TROWS×TCOLS output/threadgroup, TROWS/8 simdgroups × NACC 8×8-col
// accumulators. Measured: beats MPS on aligned shapes (plain on tall/short-K,
// split-K on fat-K). NO edge handling — `pick_sgemm` gates to 64/8 alignment.
// @@RLX_SGEMM_TILES@@

// Zero an f32 buffer region (for split-K accumulation init).
kernel void zero_f32(
    device float* C [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) C[i] = 0.0f;
}

// 32x32-tiled with bias + optional activation fused.
kernel void sgemm_simd_4x4_bias(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* C       [[buffer(3)]],
    constant uint& M      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    constant uint& N      [[buffer(6)]],
    constant uint& act_kind [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint slid  [[thread_index_in_simdgroup]]
) {
    uint sg_row = sgid / 4;
    uint sg_col = sgid % 4;
    uint tg_row_base = tgid.y * 32;
    uint tg_col_base = tgid.x * 32;

    threadgroup float A_tg[32 * 32];
    threadgroup float B_tg[32 * 32];

    simdgroup_float8x8 a, b, c;
    c = simdgroup_float8x8(0.0f);

    for (uint kk = 0; kk < K; kk += 32) {
        uint linear = sgid * 32 + slid;
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 512 + linear;
            uint ar = idx / 32;
            uint ac = idx % 32;
            A_tg[idx] = A[(tg_row_base + ar) * K + (kk + ac)];
        }
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 512 + linear;
            uint br = idx / 32;
            uint bc = idx % 32;
            B_tg[idx] = B[(kk + br) * N + (tg_col_base + bc)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_inner = 0; k_inner < 32; k_inner += 8) {
            simdgroup_load(a, &A_tg[sg_row * 8 * 32 + k_inner], 32);
            simdgroup_load(b, &B_tg[k_inner * 32 + sg_col * 8], 32);
            simdgroup_multiply_accumulate(c, a, b, c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Stage 8x8 output, apply bias + activation per element
    threadgroup float tile[16 * 64]; // 16 simdgroups × 64 elements each
    simdgroup_store(c, &tile[sgid * 64], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint out_row_base = tg_row_base + sg_row * 8;
    uint out_col_base = tg_col_base + sg_col * 8;
    for (uint i = 0; i < 2; ++i) {
        uint idx = i * 32 + slid;
        uint r = idx / 8;
        uint cc = idx % 8;
        float v = tile[sgid * 64 + idx] + bias[out_col_base + cc];
        if (act_kind == 1) {
            v = rlx_gelu_scalar(v);
        } else if (act_kind == 2) {
            v = rlx_silu_scalar(v);
        }
        C[(out_row_base + r) * N + (out_col_base + cc)] = v;
    }
}

// sgemm + bias (broadcast per column) fused into one kernel.
// Dispatched same as sgemm_simd: 1 threadgroup per 8x8 output tile.
kernel void sgemm_simd_bias(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* C       [[buffer(3)]],
    constant uint& M      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    constant uint& N      [[buffer(6)]],
    constant uint& act_kind [[buffer(7)]],  // 0=none, 1=gelu, 2=silu
    uint2 tgid [[threadgroup_position_in_grid]],
    uint slid  [[thread_index_in_simdgroup]]
) {
    uint row_base = tgid.y * 8;
    uint col_base = tgid.x * 8;
    if (row_base >= M || col_base >= N) return;

    simdgroup_float8x8 a, b, c;
    c = simdgroup_float8x8(0.0f);

    for (uint k = 0; k < K; k += 8) {
        simdgroup_load(a, A + row_base * K + k, K);
        simdgroup_load(b, B + k * N + col_base, N);
        simdgroup_multiply_accumulate(c, a, b, c);
    }

    // Stage tile in threadgroup memory, then apply bias + activation per element
    threadgroup float tile[64];
    simdgroup_store(c, tile, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 32 threads × 2 elements each cover the 8x8 tile
    for (uint i = 0; i < 2; ++i) {
        uint idx = i * 32 + slid;
        uint r = idx / 8;
        uint cc = idx % 8;
        float v = tile[idx] + bias[col_base + cc];
        if (act_kind == 1) {
            v = rlx_gelu_scalar(v);
        } else if (act_kind == 2) {
            v = rlx_silu_scalar(v);
        }
        C[(row_base + r) * N + (col_base + cc)] = v;
    }
}

// Padded variant: arbitrary M with bounds-checked stores + bias + optional act.
kernel void sgemm_simd_padded_bias(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* C       [[buffer(3)]],
    constant uint& M      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    constant uint& N      [[buffer(6)]],
    constant uint& act_kind [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint slid  [[thread_index_in_simdgroup]]
) {
    uint row_base = tgid.y * 8;
    uint col_base = tgid.x * 8;

    threadgroup float A_pad[64];
    threadgroup float B_pad[64];

    simdgroup_float8x8 a, b, c;
    c = simdgroup_float8x8(0.0f);

    for (uint k = 0; k < K; k += 8) {
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 32 + slid;
            uint ar = idx / 8, ac = idx % 8;
            uint sr = row_base + ar, sc = k + ac;
            A_pad[idx] = (sr < M && sc < K) ? A[sr * K + sc] : 0.0f;
        }
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 32 + slid;
            uint br = idx / 8, bc = idx % 8;
            uint sr = k + br, sc = col_base + bc;
            B_pad[idx] = (sr < K && sc < N) ? B[sr * N + sc] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_load(a, A_pad, 8);
        simdgroup_load(b, B_pad, 8);
        simdgroup_multiply_accumulate(c, a, b, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float C_pad[64];
    simdgroup_store(c, C_pad, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < 2; ++i) {
        uint idx = i * 32 + slid;
        uint r = idx / 8;
        uint cc = idx % 8;
        uint dst_row = row_base + r;
        uint dst_col = col_base + cc;
        if (dst_row < M && dst_col < N) {
            float v = C_pad[idx] + bias[dst_col];
            if (act_kind == 1) {
                v = rlx_gelu_scalar(v);
            } else if (act_kind == 2) {
                v = rlx_silu_scalar(v);
            }
            C[dst_row * N + dst_col] = v;
        }
    }
}

// Padded simdgroup sgemm: handles arbitrary M/K/N by zero-padding.
// Reads A row-by-row with bounds checks, computes 8x8 simdgroup tiles,
// writes back row-by-row with bounds checks. Slower than sgemm_simd for
// aligned dims but works for the common batch=1 case (m=6).
//
// Strategy: pre-stage A's relevant rows into threadgroup memory (zero-pad
// missing rows), then use simdgroup ops on the padded tile. Same for B's
// columns.
kernel void sgemm_simd_padded(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& K      [[buffer(4)]],
    constant uint& N      [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint slid  [[thread_index_in_simdgroup]]
) {
    uint row_base = tgid.y * 8;
    uint col_base = tgid.x * 8;

    // Per-tile staging in threadgroup memory: 8x8 A tile, 8x8 B tile.
    // 32 threads collaborate to stage; reuse the simdgroup_load API for
    // the multiply once data is in threadgroup or device memory.
    threadgroup float A_pad[64];
    threadgroup float B_pad[64];

    simdgroup_float8x8 a, b, c;
    c = simdgroup_float8x8(0.0f);

    for (uint k = 0; k < K; k += 8) {
        // Stage 8x8 A tile with bounds-checked loads (32 threads cover 64 elements: 2 each)
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 32 + slid;
            uint ar = idx / 8;
            uint ac = idx % 8;
            uint src_row = row_base + ar;
            uint src_col = k + ac;
            float v = (src_row < M && src_col < K) ? A[src_row * K + src_col] : 0.0f;
            A_pad[idx] = v;
        }
        // Stage 8x8 B tile
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 32 + slid;
            uint br = idx / 8;
            uint bc = idx % 8;
            uint src_row = k + br;
            uint src_col = col_base + bc;
            float v = (src_row < K && src_col < N) ? B[src_row * N + src_col] : 0.0f;
            B_pad[idx] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_load(a, A_pad, 8);
        simdgroup_load(b, B_pad, 8);
        simdgroup_multiply_accumulate(c, a, b, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Bounds-checked store of the 8x8 C tile (32 threads × 2 elements each)
    threadgroup float C_pad[64];
    simdgroup_store(c, C_pad, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 2; ++i) {
        uint idx = i * 32 + slid;
        uint cr = idx / 8;
        uint cc = idx % 8;
        uint dst_row = row_base + cr;
        uint dst_col = col_base + cc;
        if (dst_row < M && dst_col < N) {
            C[dst_row * N + dst_col] = C_pad[idx];
        }
    }
}

// sgemm_simd_padded with a fused full-tensor residual add in the epilogue:
// C = A·B + R (R same shape as C, [M,N]). Emitted for Op::FusedMatMulResidual
// so the transformer residual `add(skip, matmul_out)` folds into the matmul's
// store instead of a separate elementwise-add dispatch — one fewer launch on a
// launch-latency-bound decode. Body is identical to sgemm_simd_padded; only the
// store adds R[idx].
kernel void sgemm_simd_padded_residual(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& K      [[buffer(4)]],
    constant uint& N      [[buffer(5)]],
    device const float* R [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint slid  [[thread_index_in_simdgroup]]
) {
    uint row_base = tgid.y * 8;
    uint col_base = tgid.x * 8;

    threadgroup float A_pad[64];
    threadgroup float B_pad[64];

    simdgroup_float8x8 a, b, c;
    c = simdgroup_float8x8(0.0f);

    for (uint k = 0; k < K; k += 8) {
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 32 + slid;
            uint ar = idx / 8;
            uint ac = idx % 8;
            uint src_row = row_base + ar;
            uint src_col = k + ac;
            float v = (src_row < M && src_col < K) ? A[src_row * K + src_col] : 0.0f;
            A_pad[idx] = v;
        }
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 32 + slid;
            uint br = idx / 8;
            uint bc = idx % 8;
            uint src_row = k + br;
            uint src_col = col_base + bc;
            float v = (src_row < K && src_col < N) ? B[src_row * N + src_col] : 0.0f;
            B_pad[idx] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_load(a, A_pad, 8);
        simdgroup_load(b, B_pad, 8);
        simdgroup_multiply_accumulate(c, a, b, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float C_pad[64];
    simdgroup_store(c, C_pad, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 2; ++i) {
        uint idx = i * 32 + slid;
        uint cr = idx / 8;
        uint cc = idx % 8;
        uint dst_row = row_base + cr;
        uint dst_col = col_base + cc;
        if (dst_row < M && dst_col < N) {
            uint o = dst_row * N + dst_col;
            C[o] = C_pad[idx] + R[o];
        }
    }
}

// Tiled sgemm: TILExTILE output blocks, K loaded in TILE-wide stripes
// into threadgroup memory. Used for non-multiple-of-8 dimensions.
constant uint TILE = 16;

kernel void sgemm_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& K      [[buffer(4)]],
    constant uint& N      [[buffer(5)]],
    uint2 gid  [[thread_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup float Asub[16][16];
    threadgroup float Bsub[16][16];

    uint row = tgid.y * TILE + tid.y;
    uint col = tgid.x * TILE + tid.x;

    float sum = 0.0;
    uint num_tiles = (K + TILE - 1) / TILE;

    for (uint t = 0; t < num_tiles; ++t) {
        uint a_col = t * TILE + tid.x;
        uint b_row = t * TILE + tid.y;
        Asub[tid.y][tid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0;
        Bsub[tid.y][tid.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; ++k) {
            sum += Asub[tid.y][k] * Bsub[k][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// out = bias_add(data, bias, m, n)
kernel void bias_add(
    device float* data [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant uint& m [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= m || col >= n) return;
    data[row * n + col] += bias[col];
}

// in-place GELU using Abramowitz & Stegun erf approximation
// (matches CPU NEON kernel for parity)
// f32 gelu / gelu_approx in-place (arena form) — generated from rlxsl. The tuned
// vec4 `*_inplace4` variants below stay hand-written (peak-perf hot path).
// @@RLX_GELU_INPLACE_F32@@

kernel void gelu_approx_inplace4(
    device char* arena [[buffer(0)]],
    constant ulong& data_byte_off [[buffer(1)]],
    constant uint& len4 [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len4) return;
    device packed_float4* data = (device packed_float4*)(arena + data_byte_off);
    packed_float4 px = data[gid];
    packed_float4 out;
    for (uint c = 0; c < 4; ++c) {
        out[c] = rlx_gelu_approx_scalar(px[c]);
    }
    data[gid] = out;
}

kernel void gelu_approx_out4(
    device const char* arena [[buffer(0)]],
    constant ulong& src_off [[buffer(1)]],
    constant ulong& dst_off [[buffer(2)]],
    constant uint& len4 [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len4) return;
    device const packed_float4* src = (device const packed_float4*)(arena + src_off);
    device packed_float4* dst = (device packed_float4*)(arena + dst_off);
    packed_float4 px = src[gid];
    packed_float4 out;
    for (uint c = 0; c < 4; ++c) {
        out[c] = rlx_gelu_approx_scalar(px[c]);
    }
    dst[gid] = out;
}

kernel void gelu_inplace4(
    device char* arena [[buffer(0)]],
    constant ulong& data_byte_off [[buffer(1)]],
    constant uint& len4 [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len4) return;
    device packed_float4* data = (device packed_float4*)(arena + data_byte_off);
    packed_float4 px = data[gid];
    packed_float4 out;
    for (uint c = 0; c < 4; ++c) {
        out[c] = rlx_gelu_scalar(px[c]);
    }
    data[gid] = out;
}

kernel void silu_inplace4(
    device char* arena [[buffer(0)]],
    constant ulong& data_byte_off [[buffer(1)]],
    constant uint& len4 [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len4) return;
    device packed_float4* data = (device packed_float4*)(arena + data_byte_off);
    packed_float4 px = data[gid];
    packed_float4 out;
    for (uint c = 0; c < 4; ++c) {
        out[c] = rlx_silu_scalar(px[c]);
    }
    data[gid] = out;
}

// Out-of-place SiLU: dst = src * sigmoid(src) — skips a prior Copy.
kernel void silu_out4(
    device char* arena [[buffer(0)]],
    constant ulong& src_byte_off [[buffer(1)]],
    constant ulong& dst_byte_off [[buffer(2)]],
    constant uint& len4 [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len4) return;
    device const packed_float4* src = (device const packed_float4*)(arena + src_byte_off);
    device packed_float4* dst = (device packed_float4*)(arena + dst_byte_off);
    packed_float4 px = src[gid];
    packed_float4 out;
    for (uint c = 0; c < 4; ++c) {
        out[c] = rlx_silu_scalar(px[c]);
    }
    dst[gid] = out;
}

// rhs [cols] broadcast across rows: out[m, n] = lhs[m*cols+n] op rhs[n]
kernel void binary_broadcast_rhs_col_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* dst       [[buffer(2)]],
    constant uint& rows     [[buffer(3)]],
    constant uint& cols     [[buffer(4)]],
    constant uint& op       [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = gid.y;
    uint n = gid.x;
    if (m >= rows || n >= cols) return;
    float lv = lhs[m * cols + n];
    float rv = rhs[n];
    float out;
    switch (op) {
        case 0: out = lv + rv; break;
        case 1: out = lv - rv; break;
        case 2: out = lv * rv; break;
        case 3: out = lv / rv; break;
        case 4: out = max(lv, rv); break;
        case 5: out = min(lv, rv); break;
        default: out = rlx_pow_scalar(lv, rv); break;
    }
    dst[m * cols + n] = out;
}

kernel void binary_broadcast_rhs_col4(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* dst       [[buffer(2)]],
    constant uint& rows     [[buffer(3)]],
    constant uint& cols4    [[buffer(4)]],
    constant uint& op       [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = gid.y;
    uint n4 = gid.x;
    uint cols = cols4 * 4u;
    if (m >= rows || n4 >= cols4) return;
    device const packed_float4* lhs4 =
        (device const packed_float4*)(lhs + m * cols);
    device const packed_float4* rhs4 = (device const packed_float4*)(rhs);
    device packed_float4* dst4 = (device packed_float4*)(dst + m * cols);
    packed_float4 lv = lhs4[n4];
    packed_float4 rv = rhs4[n4];
    packed_float4 out;
    switch (op) {
        case 0: out = lv + rv; break;
        case 1: out = lv - rv; break;
        case 2: out = lv * rv; break;
        case 3: out = lv / rv; break;
        case 4: out = max(lv, rv); break;
        case 5: out = min(lv, rv); break;
        default: out = rlx_pow_scalar(lv, rv); break;
    }
    dst4[n4] = out;
}

// Dense lhs + rhs row vector broadcast on the last axis (e.g. `[rows, cols] op [rows, 1]`).
kernel void binary_broadcast_rhs_row_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* dst       [[buffer(2)]],
    constant uint& rows     [[buffer(3)]],
    constant uint& cols     [[buffer(4)]],
    constant uint& op       [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = gid.y;
    uint n = gid.x;
    if (m >= rows || n >= cols) return;
    float lv = lhs[m * cols + n];
    float rv = rhs[m];
    float out;
    switch (op) {
        case 0: out = lv + rv; break;
        case 1: out = lv - rv; break;
        case 2: out = lv * rv; break;
        case 3: out = lv / rv; break;
        case 4: out = max(lv, rv); break;
        case 5: out = min(lv, rv); break;
        default: out = rlx_pow_scalar(lv, rv); break;
    }
    dst[m * cols + n] = out;
}

kernel void binary_broadcast_rhs_row4(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* dst       [[buffer(2)]],
    constant uint& rows     [[buffer(3)]],
    constant uint& cols4    [[buffer(4)]],
    constant uint& op       [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = gid.y;
    uint n4 = gid.x;
    uint cols = cols4 * 4u;
    if (m >= rows || n4 >= cols4) return;
    device const packed_float4* lhs4 =
        (device const packed_float4*)(lhs + m * cols);
    float rv = rhs[m];
    packed_float4 rv4 = packed_float4(rv);
    device packed_float4* dst4 = (device packed_float4*)(dst + m * cols);
    packed_float4 lv = lhs4[n4];
    packed_float4 out;
    switch (op) {
        case 0: out = lv + rv4; break;
        case 1: out = lv - rv4; break;
        case 2: out = lv * rv4; break;
        case 3: out = lv / rv4; break;
        case 4: out = max(lv, rv4); break;
        case 5: out = min(lv, rv4); break;
        default: out = rlx_pow_scalar(lv, rv4); break;
    }
    dst4[n4] = out;
}

// Dense lhs + scalar rhs (all broadcast strides zero).
kernel void binary_broadcast_rhs_scalar_f32(
    device const char* arena [[buffer(0)]],
    constant ulong& lhs_off [[buffer(1)]],
    constant ulong& rhs_off [[buffer(2)]],
    constant ulong& dst_off [[buffer(3)]],
    constant uint& len      [[buffer(4)]],
    constant uint& op       [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    device const float* lhs = (device const float*)(arena + lhs_off);
    device const float* rhs = (device const float*)(arena + rhs_off);
    device float* dst = (device float*)(arena + dst_off);
    float lv = lhs[gid];
    float rv = rhs[0];
    float out;
    switch (op) {
        case 0: out = lv + rv; break;
        case 1: out = lv - rv; break;
        case 2: out = lv * rv; break;
        case 3: out = lv / rv; break;
        case 4: out = max(lv, rv); break;
        case 5: out = min(lv, rv); break;
        default: out = rlx_pow_scalar(lv, rv); break;
    }
    dst[gid] = out;
}

kernel void binary_broadcast_rhs_scalar4(
    device const char* arena [[buffer(0)]],
    constant ulong& lhs_off [[buffer(1)]],
    constant ulong& rhs_off [[buffer(2)]],
    constant ulong& dst_off [[buffer(3)]],
    constant uint& len4             [[buffer(4)]],
    constant uint& op               [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len4) return;
    device const packed_float4* lhs = (device const packed_float4*)(arena + lhs_off);
    device const float* rhs = (device const float*)(arena + rhs_off);
    device packed_float4* dst = (device packed_float4*)(arena + dst_off);
    float rv = rhs[0];
    packed_float4 rv4 = packed_float4(rv);
    packed_float4 lv = lhs[gid];
    packed_float4 out;
    switch (op) {
        case 0: out = lv + rv4; break;
        case 1: out = lv - rv4; break;
        case 2: out = lv * rv4; break;
        case 3: out = lv / rv4; break;
        case 4: out = max(lv, rv4); break;
        case 5: out = min(lv, rv4); break;
        default: out = rlx_pow_scalar(lv, rv4); break;
    }
    dst[gid] = out;
}

// Dense lhs + rhs broadcast on exactly one axis (e.g. `[B, T, H] op [B, 1, H]`).
kernel void binary_broadcast_1ax_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* dst       [[buffer(2)]],
    constant uint& rows     [[buffer(3)]],
    constant uint& cols     [[buffer(4)]],
    constant uint& mid      [[buffer(5)]],
    constant uint& op       [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    if (col >= cols || row >= rows) return;
    uint pre_i = row / mid;
    uint li = row * cols + col;
    uint ri = pre_i * cols + col;
    float lv = lhs[li];
    float rv = rhs[ri];
    float out;
    switch (op) {
        case 0: out = lv + rv; break;
        case 1: out = lv - rv; break;
        case 2: out = lv * rv; break;
        case 3: out = lv / rv; break;
        case 4: out = max(lv, rv); break;
        case 5: out = min(lv, rv); break;
        default: out = rlx_pow_scalar(lv, rv); break;
    }
    dst[li] = out;
}

kernel void binary_broadcast_1ax4(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* dst       [[buffer(2)]],
    constant uint& rows     [[buffer(3)]],
    constant uint& cols4    [[buffer(4)]],
    constant uint& mid      [[buffer(5)]],
    constant uint& op       [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col4 = gid.x;
    uint row = gid.y;
    uint cols = cols4 * 4u;
    if (col4 >= cols4 || row >= rows) return;
    uint pre_i = row / mid;
    device const packed_float4* lhs4 =
        (device const packed_float4*)(lhs + row * cols);
    device const float* rhs_row = rhs + pre_i * cols;
    device const packed_float4* rhs4 =
        (device const packed_float4*)(rhs_row);
    device packed_float4* dst4 =
        (device packed_float4*)(dst + row * cols);
    packed_float4 lv = lhs4[col4];
    packed_float4 rv = rhs4[col4];
    packed_float4 out;
    switch (op) {
        case 0: out = lv + rv; break;
        case 1: out = lv - rv; break;
        case 2: out = lv * rv; break;
        case 3: out = lv / rv; break;
        case 4: out = max(lv, rv); break;
        case 5: out = min(lv, rv); break;
        default: out = rlx_pow_scalar(lv, rv); break;
    }
    dst4[col4] = out;
}

// Binary-op math generated once from the shared rlxsl manifest (fixes the
// negative-base `pow` and 32-bit-bitwise drift the hand-written switch had).
// @@RLX_BINARY_FN@@
inline float fused_bin(float lv, float rv, uint op) { return rlx_binary_apply(op, lv, rv); }

// Same-size element-wise binary by opcode (Mod/bitwise: no dedicated elem_*).
kernel void elem_binop(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c        [[buffer(2)]],
    constant uint& n       [[buffer(3)]],
    constant uint& op      [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) { if (gid >= n) return; c[gid] = fused_bin(a[gid], b[gid], op); }

kernel void elem_binop_h(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c       [[buffer(2)]],
    constant uint& n     [[buffer(3)]],
    constant uint& op    [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) { if (gid >= n) return; c[gid] = half(fused_bin(float(a[gid]), float(b[gid]), op)); }

// Fused-region activation dispatch. Its mini opcode scheme (0=gelu 1=silu
// 2=relu 3=sigmoid 4=tanh) routes to the shared `rlx_<name>_scalar` functions
// generated from rlxsl — so the A&S erf polynomial (and its drift) is no longer
// re-inlined here. Metal inlines these, so the fused epilogue stays as fast.
inline float fused_act(float x, uint act) {
    switch (act) {
        case 0: return rlx_gelu_scalar(x);
        case 1: return rlx_silu_scalar(x);
        case 2: return rlx_relu_scalar(x);
        case 3: return rlx_sigmoid_scalar(x);
        case 4: return rlx_tanh_scalar(x);
        default: return x;
    }
}

kernel void fused_binary_activation_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* dst       [[buffer(2)]],
    constant uint& len      [[buffer(3)]],
    constant uint& bin_op   [[buffer(4)]],
    constant uint& act_op   [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    float v = fused_bin(lhs[gid], rhs[gid], bin_op);
    dst[gid] = fused_act(v, act_op);
}

kernel void fused_binary_activation4(
    device const packed_float4* lhs [[buffer(0)]],
    device const packed_float4* rhs [[buffer(1)]],
    device packed_float4* dst       [[buffer(2)]],
    constant uint& len4             [[buffer(3)]],
    constant uint& bin_op           [[buffer(4)]],
    constant uint& act_op           [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len4) return;
    packed_float4 lv = lhs[gid];
    packed_float4 rv = rhs[gid];
    packed_float4 out;
    for (uint c = 0; c < 4; ++c) {
        float v = fused_bin(lv[c], rv[c], bin_op);
        out[c] = fused_act(v, act_op);
    }
    dst[gid] = out;
}

kernel void fused_ternary_activation_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs0 [[buffer(1)]],
    device const float* rhs1 [[buffer(2)]],
    device float* dst       [[buffer(3)]],
    constant uint& len      [[buffer(4)]],
    constant uint& bin_op0  [[buffer(5)]],
    constant uint& bin_op1  [[buffer(6)]],
    constant uint& act_op   [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    float v = fused_bin(lhs[gid], rhs0[gid], bin_op0);
    v = fused_bin(v, rhs1[gid], bin_op1);
    dst[gid] = fused_act(v, act_op);
}

kernel void fused_ternary_activation4(
    device const packed_float4* lhs [[buffer(0)]],
    device const packed_float4* rhs0 [[buffer(1)]],
    device const packed_float4* rhs1 [[buffer(2)]],
    device packed_float4* dst       [[buffer(3)]],
    constant uint& len4             [[buffer(4)]],
    constant uint& bin_op0          [[buffer(5)]],
    constant uint& bin_op1          [[buffer(6)]],
    constant uint& act_op           [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len4) return;
    packed_float4 lv = lhs[gid];
    packed_float4 r0 = rhs0[gid];
    packed_float4 r1 = rhs1[gid];
    packed_float4 out;
    for (uint c = 0; c < 4; ++c) {
        float v = fused_bin(lv[c], r0[c], bin_op0);
        v = fused_bin(v, r1[c], bin_op1);
        out[c] = fused_act(v, act_op);
    }
    dst[gid] = out;
}

// Element-wise add: c = a + b (same length)
kernel void elem_add(
    device const char* arena [[buffer(0)]],
    constant ulong& a_off [[buffer(1)]],
    constant ulong& b_off [[buffer(2)]],
    constant ulong& c_off [[buffer(3)]],
    constant uint& len [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    device const float* a = (device const float*)(arena + a_off);
    device const float* b = (device const float*)(arena + b_off);
    device float* c = (device float*)(arena + c_off);
    c[gid] = a[gid] + b[gid];
}

kernel void elem_add4(
    device const char* arena [[buffer(0)]],
    constant ulong& a_off [[buffer(1)]],
    constant ulong& b_off [[buffer(2)]],
    constant ulong& c_off [[buffer(3)]],
    constant uint& len4 [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len4) return;
    device const packed_float4* a = (device const packed_float4*)(arena + a_off);
    device const packed_float4* b = (device const packed_float4*)(arena + b_off);
    device packed_float4* c = (device packed_float4*)(arena + c_off);
    c[gid] = a[gid] + b[gid];
}

kernel void elem_sub4(
    device const char* arena [[buffer(0)]],
    constant ulong& a_off [[buffer(1)]],
    constant ulong& b_off [[buffer(2)]],
    constant ulong& c_off [[buffer(3)]],
    constant uint& len4 [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len4) return;
    device const packed_float4* a = (device const packed_float4*)(arena + a_off);
    device const packed_float4* b = (device const packed_float4*)(arena + b_off);
    device packed_float4* c = (device packed_float4*)(arena + c_off);
    c[gid] = a[gid] - b[gid];
}

// Rank-2 broadcast without a per-element rank loop (fallback after vec fast paths).
kernel void binary_broadcast_rank2_f32(
    device const float* lhs       [[buffer(0)]],
    device const float* rhs       [[buffer(1)]],
    device float* dst             [[buffer(2)]],
    constant uint& len            [[buffer(3)]],
    constant uint& dim0           [[buffer(4)]],
    constant uint& dim1           [[buffer(5)]],
    constant uint& lhs_stride0    [[buffer(6)]],
    constant uint& lhs_stride1    [[buffer(7)]],
    constant uint& rhs_stride0    [[buffer(8)]],
    constant uint& rhs_stride1    [[buffer(9)]],
    constant uint& op             [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    uint j = gid % dim1;
    uint i = gid / dim1;
    uint li = i * lhs_stride0 + j * lhs_stride1;
    uint ri = i * rhs_stride0 + j * rhs_stride1;
    float lv = lhs[li];
    float rv = rhs[ri];
    float out;
    switch (op) {
        case 0: out = lv + rv; break;
        case 1: out = lv - rv; break;
        case 2: out = lv * rv; break;
        case 3: out = lv / rv; break;
        case 4: out = max(lv, rv); break;
        case 5: out = min(lv, rv); break;
        default: out = rlx_pow_scalar(lv, rv); break;
    }
    dst[gid] = out;
}

kernel void binary_broadcast_rank24(
    device const float* lhs       [[buffer(0)]],
    device const float* rhs       [[buffer(1)]],
    device float* dst             [[buffer(2)]],
    constant uint& len4           [[buffer(3)]],
    constant uint& dim0           [[buffer(4)]],
    constant uint& dim1           [[buffer(5)]],
    constant uint& lhs_stride0    [[buffer(6)]],
    constant uint& lhs_stride1    [[buffer(7)]],
    constant uint& rhs_stride0    [[buffer(8)]],
    constant uint& rhs_stride1    [[buffer(9)]],
    constant uint& op             [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len4) return;
    uint cols4 = dim1 / 4u;
    uint j4 = gid % cols4;
    uint i = gid / cols4;
    uint j = j4 * 4u;
    device const packed_float4* lhs4 =
        (device const packed_float4*)(lhs + i * lhs_stride0 + j * lhs_stride1);
    device const packed_float4* rhs4 =
        (device const packed_float4*)(rhs + i * rhs_stride0 + j * rhs_stride1);
    device packed_float4* dst4 = (device packed_float4*)(dst + i * dim1 + j);
    packed_float4 lv = *lhs4;
    packed_float4 rv = *rhs4;
    packed_float4 out;
    switch (op) {
        case 0: out = lv + rv; break;
        case 1: out = lv - rv; break;
        case 2: out = lv * rv; break;
        case 3: out = lv / rv; break;
        case 4: out = max(lv, rv); break;
        case 5: out = min(lv, rv); break;
        default: out = rlx_pow_scalar(lv, rv); break;
    }
    *dst4 = out;
}

// Shape-aware broadcast binary op. Each thread computes one output
// element by decomposing gid into coords against `out_dims` (row-major)
// and walking `lhs_strides`/`rhs_strides` (stride 0 ⇒ broadcast).
// Op encoding matches `rlx_ir::op::BinaryOp` discriminant order:
//   0=Add, 1=Sub, 2=Mul, 3=Div, 4=Max, 5=Min, 6=Pow. Rank capped at 8.
kernel void binary_broadcast_f32(
    device const float* lhs       [[buffer(0)]],
    device const float* rhs       [[buffer(1)]],
    device float* dst             [[buffer(2)]],
    constant uint& len            [[buffer(3)]],
    constant uint& rank           [[buffer(4)]],
    constant uint* out_dims       [[buffer(5)]],
    constant uint* lhs_strides    [[buffer(6)]],
    constant uint* rhs_strides    [[buffer(7)]],
    constant uint& op             [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    uint rem = gid;
    uint li = 0;
    uint ri = 0;
    // Walk from innermost dim to outermost (matches row-major decomposition).
    for (uint ax_rev = 0; ax_rev < rank; ++ax_rev) {
        uint ax = rank - 1 - ax_rev;
        uint sz = out_dims[ax];
        uint coord = rem % sz;
        rem /= sz;
        li += coord * lhs_strides[ax];
        ri += coord * rhs_strides[ax];
    }
    float lv = lhs[li];
    float rv = rhs[ri];
    float out;
    switch (op) {
        case 0: out = lv + rv; break;
        case 1: out = lv - rv; break;
        case 2: out = lv * rv; break;
        case 3: out = lv / rv; break;
        case 4: out = max(lv, rv); break;
        case 5: out = min(lv, rv); break;
        default: out = rlx_pow_scalar(lv, rv); break;
    }
    dst[gid] = out;
}

// Element-wise multiply: c = a * b
kernel void elem_mul(
    device const char* arena [[buffer(0)]],
    constant ulong& a_off [[buffer(1)]],
    constant ulong& b_off [[buffer(2)]],
    constant ulong& c_off [[buffer(3)]],
    constant uint& len [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    device const float* a = (device const float*)(arena + a_off);
    device const float* b = (device const float*)(arena + b_off);
    device float* c = (device float*)(arena + c_off);
    c[gid] = a[gid] * b[gid];
}

kernel void elem_mul4(
    device const char* arena [[buffer(0)]],
    constant ulong& a_off [[buffer(1)]],
    constant ulong& b_off [[buffer(2)]],
    constant ulong& c_off [[buffer(3)]],
    constant uint& len4 [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len4) return;
    device const packed_float4* a = (device const packed_float4*)(arena + a_off);
    device const packed_float4* b = (device const packed_float4*)(arena + b_off);
    device packed_float4* c = (device packed_float4*)(arena + c_off);
    c[gid] = a[gid] * b[gid];
}

kernel void elem_div4(
    device const char* arena [[buffer(0)]],
    constant ulong& a_off [[buffer(1)]],
    constant ulong& b_off [[buffer(2)]],
    constant ulong& c_off [[buffer(3)]],
    constant uint& len4 [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len4) return;
    device const packed_float4* a = (device const packed_float4*)(arena + a_off);
    device const packed_float4* b = (device const packed_float4*)(arena + b_off);
    device packed_float4* c = (device packed_float4*)(arena + c_off);
    c[gid] = a[gid] / b[gid];
}

// Element-wise subtract: c = a - b
kernel void elem_sub(
    device const char* arena [[buffer(0)]],
    constant ulong& a_off [[buffer(1)]],
    constant ulong& b_off [[buffer(2)]],
    constant ulong& c_off [[buffer(3)]],
    constant uint& len [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    device const float* a = (device const float*)(arena + a_off);
    device const float* b = (device const float*)(arena + b_off);
    device float* c = (device float*)(arena + c_off);
    c[gid] = a[gid] - b[gid];
}

// Element-wise divide: c = a / b
kernel void elem_div(
    device const char* arena [[buffer(0)]],
    constant ulong& a_off [[buffer(1)]],
    constant ulong& b_off [[buffer(2)]],
    constant ulong& c_off [[buffer(3)]],
    constant uint& len [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    device const float* a = (device const float*)(arena + a_off);
    device const float* b = (device const float*)(arena + b_off);
    device float* c = (device float*)(arena + c_off);
    c[gid] = a[gid] / b[gid];
}

kernel void elem_max(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c       [[buffer(2)]],
    constant uint& len    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) { if (gid >= len) return; c[gid] = max(a[gid], b[gid]); }

kernel void elem_min(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c       [[buffer(2)]],
    constant uint& len    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) { if (gid >= len) return; c[gid] = min(a[gid], b[gid]); }

kernel void elem_pow(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c       [[buffer(2)]],
    constant uint& len    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) { if (gid >= len) return; c[gid] = rlx_pow_scalar(a[gid], b[gid]); }

// Element-wise compare: writes 1.0 / 0.0 per element. `op_kind` selects:
//   0=Eq 1=Ne 2=Lt 3=Le 4=Gt 5=Ge
// One kernel for all six variants keeps the binary-shaped dispatch path
// uniform — the encoder picks op_kind at submit time.
// Compare-op math generated once from the shared rlxsl manifest.
// @@RLX_COMPARE_FN@@
kernel void elem_compare(
    device const float* a    [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    device float* c          [[buffer(2)]],
    constant uint& len       [[buffer(3)]],
    constant uint& op_kind   [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    c[gid] = rlx_compare_apply(op_kind, a[gid], b[gid]);
}

/// Compare with optional scalar broadcast (`flags`: bit0=lhs scalar, bit1=rhs).
kernel void elem_compare_bcast(
    device const float* a    [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    device float* c          [[buffer(2)]],
    constant uint& len       [[buffer(3)]],
    constant uint& op_kind   [[buffer(4)]],
    constant uint& flags     [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    float x = (flags & 1u) ? a[0] : a[gid];
    float y = (flags & 2u) ? b[0] : b[gid];
    c[gid] = rlx_compare_apply(op_kind, x, y);
}

// 2D convolution (naive direct, NCHW input). One thread per output
// element. Supports groups, dilation. Bias is a separate Op (matches the
// IR's two-input Conv shape). Two u32-arrays of dims pack into one
// constant buffer; an `aux` buffer carries the param triplets.
kernel void conv2d(
    device const float* src    [[buffer(0)]],
    device const float* wt     [[buffer(1)]],
    device float* dst          [[buffer(2)]],
    constant uint4& nch        [[buffer(3)]],   // [N, C_in, H, W]
    constant uint4& out_dims   [[buffer(4)]],   // [C_out, H_out, W_out, groups]
    constant uint4& kshape     [[buffer(5)]],   // [kh, kw, sh, sw]
    constant uint4& padd       [[buffer(6)]],   // [ph, pw, dh, dw]
    uint3 gid [[thread_position_in_grid]]
) {
    uint nco = gid.z;            // n * c_out + co
    uint ho = gid.y;
    uint wo = gid.x;
    uint c_out = out_dims.x;
    uint h_out = out_dims.y;
    uint w_out = out_dims.z;
    uint groups = out_dims.w;
    if (ho >= h_out || wo >= w_out || nco >= nch.x * c_out) return;
    uint n = nco / c_out;
    uint co = nco % c_out;
    uint c_in = nch.y;
    uint h = nch.z;
    uint w = nch.w;
    uint c_in_per_g = c_in / groups;
    uint c_out_per_g = c_out / groups;
    uint g = co / c_out_per_g;
    uint ci_start = g * c_in_per_g;
    uint kh = kshape.x; uint kw = kshape.y;
    uint sh = kshape.z; uint sw = kshape.w;
    uint ph = padd.x; uint pw = padd.y;
    uint dh = padd.z; uint dw = padd.w;

    float acc = 0.0f;
    for (uint ci_off = 0; ci_off < c_in_per_g; ++ci_off) {
        uint ci = ci_start + ci_off;
        uint in_chan = ((n * c_in) + ci) * h * w;
        uint wt_chan = ((co * c_in_per_g) + ci_off) * kh * kw;
        for (uint ki = 0; ki < kh; ++ki) {
            for (uint kj = 0; kj < kw; ++kj) {
                int hi = (int)(ho * sh + ki * dh) - (int)ph;
                int wi = (int)(wo * sw + kj * dw) - (int)pw;
                if (hi < 0 || wi < 0 || hi >= (int)h || wi >= (int)w) continue;
                acc += src[in_chan + (uint)hi * w + (uint)wi]
                     * wt[wt_chan + ki * kw + kj];
            }
        }
    }
    dst[((n * c_out) + co) * h_out * w_out + ho * w_out + wo] = acc;
}

// Depthwise causal 1-D conv on BSC layout `[B, W, C]` → `[B, out_seq, C]`.
// Weight is NCHW-packed `[C, 1, 1, K]` (same as `Op::Conv` depthwise). Optional
// fused SiLU. Replaces Transpose→Copy→Conv2D→Copy→Transpose(+Silu) for GDN.
kernel void depthwise_conv1d_bsc(
    device float* arena            [[buffer(0)]],
    constant ulong& src_off        [[buffer(1)]],
    constant ulong& wt_off         [[buffer(2)]],
    constant ulong& dst_off        [[buffer(3)]],
    constant uint4& dims           [[buffer(4)]], // batch, width, out_seq, channels
    constant uint2& k_silu         [[buffer(5)]], // k, silu!=0
    device const float* wt_buf     [[buffer(7)]],
    uint gid                       [[thread_position_in_grid]]
) {
    const uint batch = dims.x;
    const uint width = dims.y;
    const uint out_seq = dims.z;
    const uint channels = dims.w;
    const uint k = k_silu.x;
    const bool do_silu = k_silu.y != 0u;
    const uint total = batch * out_seq * channels;
    if (gid >= total || k == 0u) return;

    const uint c = gid % channels;
    const uint tmp = gid / channels;
    const uint t = tmp % out_seq;
    const uint b = tmp / out_seq;

    device const float* src = (device const float*)((device char*)arena + src_off);
    device float* dst = (device float*)((device char*)arena + dst_off);
    device const float* wt = (device const float*)((device const char*)wt_buf + wt_off);

    const uint src_base = b * width * channels + t * channels + c;
    const uint wt_base = c * k;
    float acc = 0.f;
    if (k == 4u) {
        acc += src[src_base + 0u * channels] * wt[wt_base + 0u];
        acc += src[src_base + 1u * channels] * wt[wt_base + 1u];
        acc += src[src_base + 2u * channels] * wt[wt_base + 2u];
        acc += src[src_base + 3u * channels] * wt[wt_base + 3u];
    } else {
        for (uint ki = 0u; ki < k; ++ki) {
            acc += src[src_base + ki * channels] * wt[wt_base + ki];
        }
    }
    if (do_silu) {
        acc = acc / (1.0f + exp(-acc));
    }
    dst[b * out_seq * channels + t * channels + c] = acc;
}

// 1-D conv (W_in = W_out = 1) — Voxtral codec layout `[N,C,T,1]`.
kernel void conv2d_w1(
    device const float* src [[buffer(0)]],
    device const float* wt [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant uint4& nch [[buffer(3)]],       // [N, C_in, H, 1]
    constant uint4& out_dims [[buffer(4)]],  // [C_out, H_out, 1, groups]
    constant uint4& kshape [[buffer(5)]],
    constant uint4& padd [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint nco = gid.z;
    uint ho = gid.y;
    uint h_out = out_dims.y;
    uint c_out = out_dims.x;
    uint groups = out_dims.w;
    if (ho >= h_out || nco >= nch.x * c_out) return;
    uint n = nco / c_out;
    uint co = nco % c_out;
    uint c_in = nch.y;
    uint h = nch.z;
    uint c_in_per_g = c_in / groups;
    uint c_out_per_g = c_out / groups;
    uint g = co / c_out_per_g;
    uint ci_start = g * c_in_per_g;
    uint kh = kshape.x;
    uint kw = kshape.y;
    uint sh = kshape.z;
    uint ph = padd.x;
    uint pw = padd.y;
    uint dh = padd.z;
    uint dw = padd.w;

    float acc = 0.0f;
    for (uint ci_off = 0; ci_off < c_in_per_g; ++ci_off) {
        uint ci = ci_start + ci_off;
        uint in_chan = ((n * c_in) + ci) * h;
        uint wt_chan = ((co * c_in_per_g) + ci_off) * kh * kw;
        for (uint ki = 0; ki < kh; ++ki) {
            for (uint kj = 0; kj < kw; ++kj) {
                int hi = (int)(ho * sh + ki * dh) - (int)ph;
                int wi = (int)(kj * dw) - (int)pw;
                if (hi < 0 || wi < 0 || hi >= (int)h || wi >= 1) continue;
                acc += src[in_chan + (uint)hi] * wt[wt_chan + ki * kw + kj];
            }
        }
    }
    dst[((n * c_out) + co) * h_out + ho] = acc;
}

// LayerNorm2d (candle / SAM semantics): normalize across channels at each
// spatial position. One thread per (batch, ho, wo). gamma/beta are [C].
kernel void layer_norm2d(
    device const float* src    [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta  [[buffer(2)]],
    device float* dst          [[buffer(3)]],
    constant uint4& nchw      [[buffer(4)]],   // [N, C, H, W]
    constant float& eps       [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint n = gid.z;
    uint ho = gid.y;
    uint wo = gid.x;
    uint batch = nchw.x;
    uint c = nchw.y;
    uint h = nchw.z;
    uint w = nchw.w;
    if (n >= batch || ho >= h || wo >= w) return;

    float mean = 0.0f;
    for (uint ch = 0; ch < c; ++ch) {
        mean += src[((n * c + ch) * h + ho) * w + wo];
    }
    mean /= (float)c;
    float var = 0.0f;
    for (uint ch = 0; ch < c; ++ch) {
        float d = src[((n * c + ch) * h + ho) * w + wo] - mean;
        var += d * d;
    }
    var /= (float)c;
    float inv = rsqrt(var + eps);
    for (uint ch = 0; ch < c; ++ch) {
        uint idx = ((n * c + ch) * h + ho) * w + wo;
        float v = (src[idx] - mean) * inv;
        dst[idx] = v * gamma[ch] + beta[ch];
    }
}

// Transposed 2D convolution (NCHW, PyTorch ConvTranspose2d, no bias).
// Weight layout [C_in, C_out/groups, kH, kW]. One thread per output
// element; accumulates in-register (no output zero pass).
kernel void conv_transpose2d(
    device const float* src    [[buffer(0)]],
    device const float* wt     [[buffer(1)]],
    device float* dst          [[buffer(2)]],
    constant uint4& nch        [[buffer(3)]],   // [N, C_in, H, W]
    constant uint4& out_dims   [[buffer(4)]],   // [C_out, H_out, W_out, groups]
    constant uint4& kshape     [[buffer(5)]],   // [kh, kw, sh, sw]
    constant uint4& padd       [[buffer(6)]],   // [ph, pw, dh, dw]
    uint3 gid [[thread_position_in_grid]]
) {
    uint nco = gid.z;
    uint ho = gid.y;
    uint wo = gid.x;
    uint c_out = out_dims.x;
    uint h_out = out_dims.y;
    uint w_out = out_dims.z;
    uint groups = out_dims.w;
    if (ho >= h_out || wo >= w_out || nco >= nch.x * c_out) return;
    uint n = nco / c_out;
    uint co = nco % c_out;
    uint c_in = nch.y;
    uint h = nch.z;
    uint w = nch.w;
    uint c_in_per_g = c_in / groups;
    uint c_out_per_g = c_out / groups;
    uint g = co / c_out_per_g;
    uint oc_off = co % c_out_per_g;
    uint kh = kshape.x; uint kw = kshape.y;
    uint sh = kshape.z; uint sw = kshape.w;
    uint ph = padd.x; uint pw = padd.y;
    uint dh = padd.z; uint dw = padd.w;

    float acc = 0.0f;
    for (uint ci_off = 0; ci_off < c_in_per_g; ++ci_off) {
        uint ci = g * c_in_per_g + ci_off;
        for (uint ky = 0; ky < kh; ++ky) {
            int t_h = (int)ho + (int)ph - (int)ky * (int)dh;
            if (t_h < 0 || t_h % (int)sh != 0) continue;
            int iy = t_h / (int)sh;
            if (iy < 0 || iy >= (int)h) continue;
            for (uint kx = 0; kx < kw; ++kx) {
                int t_w = (int)wo + (int)pw - (int)kx * (int)dw;
                if (t_w < 0 || t_w % (int)sw != 0) continue;
                int ix = t_w / (int)sw;
                if (ix < 0 || ix >= (int)w) continue;
                uint w_idx = ((ci * c_out_per_g + oc_off) * kh + ky) * kw + kx;
                float v = src[((n * c_in + ci) * h + (uint)iy) * w + (uint)ix];
                acc += v * wt[w_idx];
            }
        }
    }
    dst[((n * c_out) + co) * h_out * w_out + ho * w_out + wo] = acc;
}

// 3D NCDHW convolution. Weight: [C_out, C_in/groups, kD, kH, kW].
// One thread per output element; grid (w_out, h_out, n*c_out*d_out).
kernel void conv3d(
    device const float* src    [[buffer(0)]],
    device const float* wt     [[buffer(1)]],
    device float* dst          [[buffer(2)]],
    constant uint4& a          [[buffer(3)]],  // [N, C_in, D, H]
    constant uint4& b          [[buffer(4)]],  // [W, C_out, D_out, H_out]
    constant uint4& c          [[buffer(5)]],  // [W_out, kd, kh, kw]
    constant uint4& d          [[buffer(6)]],  // [sd, sh, sw, groups]
    constant uint4& e          [[buffer(7)]],  // [pd, ph, pw, dd]
    constant uint4& f          [[buffer(8)]],  // [dh, dw, 0, 0]
    uint3 gid [[thread_position_in_grid]]
) {
    uint n = a.x; uint c_in = a.y; uint din = a.z; uint hin = a.w;
    uint win = b.x; uint c_out = b.y; uint d_out = b.z; uint h_out = b.w;
    uint w_out = c.x; uint kd = c.y; uint kh = c.z; uint kw = c.w;
    uint sd = d.x; uint sh = d.y; uint sw = d.z; uint groups = d.w;
    uint pd = e.x; uint ph = e.y; uint pw = e.z; uint dd = e.w;
    uint dh = f.x; uint dw = f.y;

    uint wo = gid.x;
    uint ho = gid.y;
    uint ndco = gid.z;
    if (wo >= w_out || ho >= h_out || ndco >= n * c_out * d_out) return;
    uint d_o = ndco % d_out;
    uint nco = ndco / d_out;
    uint nn = nco / c_out;
    uint co = nco % c_out;

    uint c_in_per_g = c_in / groups;
    uint c_out_per_g = c_out / groups;
    uint g = co / c_out_per_g;
    uint ci_start = g * c_in_per_g;

    float acc = 0.0f;
    for (uint ci_off = 0; ci_off < c_in_per_g; ++ci_off) {
        uint ci = ci_start + ci_off;
        for (uint kz = 0; kz < kd; ++kz) {
            for (uint ky = 0; ky < kh; ++ky) {
                for (uint kx = 0; kx < kw; ++kx) {
                    int in_d = (int)(d_o * sd + kz * dd) - (int)pd;
                    int in_h = (int)(ho * sh + ky * dh) - (int)ph;
                    int in_w = (int)(wo * sw + kx * dw) - (int)pw;
                    if (in_d < 0 || in_h < 0 || in_w < 0
                        || in_d >= (int)din || in_h >= (int)hin || in_w >= (int)win) {
                        continue;
                    }
                    uint in_idx = (((nn * c_in + ci) * din + (uint)in_d) * hin + (uint)in_h) * win
                                  + (uint)in_w;
                    uint w_idx = (((co * c_in_per_g + ci_off) * kd + kz) * kh + ky) * kw + kx;
                    acc += src[in_idx] * wt[w_idx];
                }
            }
        }
    }
    dst[(((nn * c_out + co) * d_out + d_o) * h_out + ho) * w_out + wo] = acc;
}

// 3D NCDHW transposed conv (output-centric gather).
// Weight: [C_in, C_out/groups, kD, kH, kW] (PyTorch ConvTranspose3d).
kernel void conv_transpose3d(
    device const float* src    [[buffer(0)]],
    device const float* wt     [[buffer(1)]],
    device float* dst          [[buffer(2)]],
    constant uint4& a          [[buffer(3)]],  // [N, C_in, D, H]
    constant uint4& b          [[buffer(4)]],  // [W, C_out, D_out, H_out]
    constant uint4& c          [[buffer(5)]],  // [W_out, kd, kh, kw]
    constant uint4& d          [[buffer(6)]],  // [sd, sh, sw, groups]
    constant uint4& e          [[buffer(7)]],  // [pd, ph, pw, dd]
    constant uint4& f          [[buffer(8)]],  // [dh, dw, 0, 0]
    uint3 gid [[thread_position_in_grid]]
) {
    uint n = a.x; uint c_in = a.y; uint din = a.z; uint hin = a.w;
    uint win = b.x; uint c_out = b.y; uint d_out = b.z; uint h_out = b.w;
    uint w_out = c.x; uint kd = c.y; uint kh = c.z; uint kw = c.w;
    uint sd = d.x; uint sh = d.y; uint sw = d.z; uint groups = d.w;
    uint pd = e.x; uint ph = e.y; uint pw = e.z; uint dd = e.w;
    uint dh = f.x; uint dw = f.y;

    uint wo = gid.x;
    uint ho = gid.y;
    uint ndco = gid.z;
    if (wo >= w_out || ho >= h_out || ndco >= n * c_out * d_out) return;
    uint d_o = ndco % d_out;
    uint nco = ndco / d_out;
    uint nn = nco / c_out;
    uint co = nco % c_out;

    uint c_in_per_g = c_in / groups;
    uint c_out_per_g = c_out / groups;
    uint g = co / c_out_per_g;
    uint oc_off = co % c_out_per_g;
    uint ci_start = g * c_in_per_g;

    float acc = 0.0f;
    for (uint kz = 0; kz < kd; ++kz) {
        int num_d = (int)d_o + (int)pd - (int)(kz * dd);
        if (num_d < 0 || (num_d % (int)sd) != 0) continue;
        uint id = (uint)(num_d / (int)sd);
        if (id >= din) continue;
        for (uint ky = 0; ky < kh; ++ky) {
            int num_h = (int)ho + (int)ph - (int)(ky * dh);
            if (num_h < 0 || (num_h % (int)sh) != 0) continue;
            uint ih = (uint)(num_h / (int)sh);
            if (ih >= hin) continue;
            for (uint kx = 0; kx < kw; ++kx) {
                int num_w = (int)wo + (int)pw - (int)(kx * dw);
                if (num_w < 0 || (num_w % (int)sw) != 0) continue;
                uint iw = (uint)(num_w / (int)sw);
                if (iw >= win) continue;
                for (uint ci_off = 0; ci_off < c_in_per_g; ++ci_off) {
                    uint ci = ci_start + ci_off;
                    uint in_idx = (((nn * c_in + ci) * din + id) * hin + ih) * win + iw;
                    uint w_idx = (((ci * c_out_per_g + oc_off) * kd + kz) * kh + ky) * kw + kx;
                    acc += src[in_idx] * wt[w_idx];
                }
            }
        }
    }
    dst[(((nn * c_out + co) * d_out + d_o) * h_out + ho) * w_out + wo] = acc;
}

// NCHW group norm: normalize each (C/G)×H×W block. One threadgroup per
// (batch, group); 256-wide reduction then normalize.
kernel void group_norm(
    device const float* src    [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta  [[buffer(2)]],
    device float* dst          [[buffer(3)]],
    constant uint4& nchw      [[buffer(4)]],   // [N, C, H, W]
    constant uint& num_groups [[buffer(5)]],
    constant float& eps       [[buffer(6)]],
    uint ng [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    uint batch = nchw.x;
    uint c = nchw.y;
    uint h = nchw.z;
    uint w = nchw.w;
    if (ng >= batch * num_groups) return;
    uint n = ng / num_groups;
    uint g = ng % num_groups;
    uint cpg = c / num_groups;
    uint c0 = g * cpg;
    uint plane = h * w;
    uint count = cpg * plane;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (uint i = tid; i < count; i += tsize) {
        uint c_off = i / plane;
        uint s = i % plane;
        uint ch = c0 + c_off;
        float v = src[((n * c + ch) * plane) + s];
        local_sum += v;
        local_sumsq += v * v;
    }
    threadgroup float partial_sum[256];
    threadgroup float partial_sumsq[256];
    partial_sum[tid] = local_sum;
    partial_sumsq[tid] = local_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
            partial_sumsq[tid] += partial_sumsq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = partial_sum[0] / float(count);
    float var = fmax(0.0f, partial_sumsq[0] / float(count) - mean * mean);
    float inv = rsqrt(var + eps);

    for (uint i = tid; i < count; i += tsize) {
        uint c_off = i / plane;
        uint s = i % plane;
        uint ch = c0 + c_off;
        uint idx = ((n * c + ch) * plane) + s;
        float v = (src[idx] - mean) * inv;
        dst[idx] = v * gamma[ch] + beta[ch];
    }
}

// Nearest-neighbor 2× upsample on planar NCHW. One thread per output pixel.
kernel void resize_nearest_2x(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint4& nchw    [[buffer(2)]],   // [N, C, H, W] input
    uint3 gid [[thread_position_in_grid]]
) {
    uint wo = gid.x;
    uint ho = gid.y;
    uint nc = gid.z;
    uint n = nchw.x;
    uint c = nchw.y;
    uint h = nchw.z;
    uint w = nchw.w;
    uint h2 = h * 2u;
    uint w2 = w * 2u;
    if (nc >= n * c || ho >= h2 || wo >= w2) return;
    uint ni = nc / c;
    uint ci = nc % c;
    uint hi = ho / 2u;
    uint wi = wo / 2u;
    float v = src[((ni * c + ci) * h + hi) * w + wi];
    dst[((ni * c + ci) * h2 + ho) * w2 + wo] = v;
}

// 2D pooling. One thread per output element (n, c, ho, wo). Padding is
// implicit-zero; Mean divides by the full kernel area to match torch's
// `count_include_pad=True`. `kind`: 0=Mean (catch-all), 2=Max.
kernel void pool2d(
    device const float* src   [[buffer(0)]],
    device float* dst         [[buffer(1)]],
    constant uint4& nchw      [[buffer(2)]],   // [N, C, H, W]
    constant uint2& hw_out    [[buffer(3)]],   // [H_out, W_out]
    constant uint4& khsw      [[buffer(4)]],   // [kh, kw, sh, sw]
    constant uint2& pad       [[buffer(5)]],   // [ph, pw]
    constant uint& kind       [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint nc = gid.z;
    uint ho = gid.y;
    uint wo = gid.x;
    uint n_total = nchw.x;
    uint c_total = nchw.y;
    if (nc >= n_total * c_total || ho >= hw_out.x || wo >= hw_out.y) return;
    uint n = nc / c_total;
    uint c = nc % c_total;
    uint h = nchw.z;
    uint w = nchw.w;
    uint h_out = hw_out.x;
    uint w_out = hw_out.y;
    uint kh = khsw.x; uint kw = khsw.y;
    uint sh = khsw.z; uint sw = khsw.w;
    uint ph = pad.x; uint pw = pad.y;

    float acc = (kind == 2) ? -INFINITY : 0.0f;
    uint in_chan = ((n * c_total) + c) * h * w;
    for (uint ki = 0; ki < kh; ++ki) {
        for (uint kj = 0; kj < kw; ++kj) {
            int hi = (int)(ho * sh + ki) - (int)ph;
            int wi = (int)(wo * sw + kj) - (int)pw;
            if (hi < 0 || wi < 0 || hi >= (int)h || wi >= (int)w) continue;
            float v = src[in_chan + (uint)hi * w + (uint)wi];
            if (kind == 2) acc = max(acc, v); else acc += v;
        }
    }
    if (kind == 0 || kind == 1) acc /= (float)(kh * kw);  // Mean
    dst[((n * c_total) + c) * h_out * w_out + ho * w_out + wo] = acc;
}

// ──────────────────────────────────────────────────────────────────────
// Training backward kernels. All three are OUTPUT-PARALLEL: each thread owns
// exactly one output element and writes it once — no atomics, no pre-zeroing,
// no scratch buffers, and (critically) no GPU→CPU sync. They mirror the CPU
// reference (crates/rlx-cpu/src/{conv_bwd,training_bwd}.rs) bit-for-bit,
// including the max-pool strict-`>` first-in-scan arg-max tie-break.
// ──────────────────────────────────────────────────────────────────────

// Max-pool backward. One thread per INPUT element (n,c,ih,iw); it accumulates
// dy from every output window in which it is the arg-max. Handles overlapping
// windows and padding. Race-free: distinct threads write distinct dx.
kernel void maxpool2d_backward(
    device const float* x   [[buffer(0)]],
    device const float* dy  [[buffer(1)]],
    device float* dx        [[buffer(2)]],
    constant uint4& p0      [[buffer(3)]],  // [N, C, H, W]
    constant uint4& p1      [[buffer(4)]],  // [H_out, W_out, kh, kw]
    constant uint4& p2      [[buffer(5)]],  // [sh, sw, ph, pw]
    uint3 gid [[thread_position_in_grid]]
) {
    uint N = p0.x, C = p0.y, H = p0.z, W = p0.w;
    uint h_out = p1.x, w_out = p1.y, kh = p1.z, kw = p1.w;
    uint sh = p2.x, sw = p2.y, ph = p2.z, pw = p2.w;
    uint iw = gid.x, ih = gid.y, nc = gid.z;
    if (nc >= N * C || ih >= H || iw >= W) return;

    int p_h = (int)ih + (int)ph;
    int p_w = (int)iw + (int)pw;
    int oh_max = p_h / (int)sh;
    int ow_max = p_w / (int)sw;
    if (oh_max >= (int)h_out) oh_max = (int)h_out - 1;
    if (ow_max >= (int)w_out) ow_max = (int)w_out - 1;
    // floor((p - k)/s)+1, clamped: integer div is floor only for non-neg args.
    int oh_min = (p_h - (int)kh < 0) ? 0 : (p_h - (int)kh) / (int)sh + 1;
    int ow_min = (p_w - (int)kw < 0) ? 0 : (p_w - (int)kw) / (int)sw + 1;

    uint in_chan = nc * H * W;
    uint out_chan = nc * h_out * w_out;
    float acc = 0.0f;
    for (int oh = oh_min; oh <= oh_max; ++oh) {
        for (int ow = ow_min; ow <= ow_max; ++ow) {
            float best_v = -INFINITY;
            int best_h = -1, best_w = -1;
            for (uint ki = 0; ki < kh; ++ki) {
                int hh = oh * (int)sh + (int)ki - (int)ph;
                if (hh < 0 || hh >= (int)H) continue;
                for (uint kj = 0; kj < kw; ++kj) {
                    int ww = ow * (int)sw + (int)kj - (int)pw;
                    if (ww < 0 || ww >= (int)W) continue;
                    float v = x[in_chan + (uint)hh * W + (uint)ww];
                    if (v > best_v) { best_v = v; best_h = hh; best_w = ww; }
                }
            }
            if (best_h == (int)ih && best_w == (int)iw)
                acc += dy[out_chan + (uint)oh * w_out + (uint)ow];
        }
    }
    dx[in_chan + ih * W + iw] = acc;
}

// Conv2d backward-input (transposed-conv gather). One thread per dx element
// (n, ci, ih, iw); gathers from every (co,ki,kj) whose forward map lands here.
kernel void conv2d_backward_input(
    device const float* dy  [[buffer(0)]],
    device const float* wt  [[buffer(1)]],
    device float* dx        [[buffer(2)]],
    constant uint4& a       [[buffer(3)]],  // [N, C_in, H, W_in]
    constant uint4& b       [[buffer(4)]],  // [C_out, H_out, W_out, kh]
    constant uint4& cc      [[buffer(5)]],  // [kw, sh, sw, ph]
    constant uint4& d       [[buffer(6)]],  // [pw, dh, dw, groups]
    uint3 gid [[thread_position_in_grid]]
) {
    uint N=a.x, C_in=a.y, H=a.z, W_in=a.w;
    uint C_out=b.x, H_out=b.y, W_out=b.z, kh=b.w;
    uint kw=cc.x, sh=cc.y, sw=cc.z, ph=cc.w;
    uint pw=d.x, dh=d.y, dw=d.z, groups=d.w;

    uint iw = gid.x, ih = gid.y, nci = gid.z;
    if (nci >= N * C_in || ih >= H || iw >= W_in) return;
    uint n = nci / C_in;
    uint ci = nci % C_in;
    uint c_in_per_g = C_in / groups;
    uint c_out_per_g = C_out / groups;
    uint g = ci / c_in_per_g;
    uint ci_local = ci % c_in_per_g;

    float acc = 0.0f;
    for (uint ki = 0; ki < kh; ++ki) {
        int num_h = (int)ih + (int)ph - (int)(ki * dh);
        if (num_h < 0 || (num_h % (int)sh) != 0) continue;
        int ho = num_h / (int)sh;
        if (ho >= (int)H_out) continue;
        for (uint kj = 0; kj < kw; ++kj) {
            int num_w = (int)iw + (int)pw - (int)(kj * dw);
            if (num_w < 0 || (num_w % (int)sw) != 0) continue;
            int wo = num_w / (int)sw;
            if (wo >= (int)W_out) continue;
            for (uint col = 0; col < c_out_per_g; ++col) {
                uint co = g * c_out_per_g + col;
                uint w_idx = ((co * c_in_per_g + ci_local) * kh + ki) * kw + kj;
                uint dy_idx = ((n * C_out + co) * H_out + (uint)ho) * W_out + (uint)wo;
                acc += wt[w_idx] * dy[dy_idx];
            }
        }
    }
    dx[((n * C_in + ci) * H + ih) * W_in + iw] = acc;
}

// Conv2d backward-weight (direct, batch-reduced). One thread per dw element
// (co, ci_local, ki, kj); sums dy*x over (n, ho, wo).
kernel void conv2d_backward_weight(
    device const float* x   [[buffer(0)]],
    device const float* dy  [[buffer(1)]],
    device float* dw        [[buffer(2)]],
    constant uint4& a       [[buffer(3)]],  // [N, C_in, H, W]
    constant uint4& b       [[buffer(4)]],  // [C_out, H_out, W_out, kh]
    constant uint4& cc      [[buffer(5)]],  // [kw, sh, sw, ph]
    constant uint4& d       [[buffer(6)]],  // [pw, dh, dw_dil, groups]
    uint3 gid [[thread_position_in_grid]]
) {
    uint N=a.x, C_in=a.y, H=a.z, W=a.w;
    uint C_out=b.x, H_out=b.y, W_out=b.z, kh=b.w;
    uint kw=cc.x, sh=cc.y, sw=cc.z, ph=cc.w;
    uint pw=d.x, dh=d.y, dwd=d.z, groups=d.w;

    uint kj = gid.x, ki = gid.y, coci = gid.z;
    uint c_in_per_g = C_in / groups;
    uint c_out_per_g = C_out / groups;
    if (kj >= kw || ki >= kh || coci >= C_out * c_in_per_g) return;
    uint co = coci / c_in_per_g;
    uint ci_local = coci % c_in_per_g;
    uint g = co / c_out_per_g;
    uint ci = g * c_in_per_g + ci_local;

    float acc = 0.0f;
    for (uint n = 0; n < N; ++n) {
        for (uint ho = 0; ho < H_out; ++ho) {
            int hh = (int)(ho * sh + ki * dh) - (int)ph;
            if (hh < 0 || hh >= (int)H) continue;
            for (uint wo = 0; wo < W_out; ++wo) {
                int ww = (int)(wo * sw + kj * dwd) - (int)pw;
                if (ww < 0 || ww >= (int)W) continue;
                uint x_idx = ((n * C_in + ci) * H + (uint)hh) * W + (uint)ww;
                uint dy_idx = ((n * C_out + co) * H_out + ho) * W_out + wo;
                acc += x[x_idx] * dy[dy_idx];
            }
        }
    }
    dw[((co * c_in_per_g + ci_local) * kh + ki) * kw + kj] = acc;
}

// MaxPool3d backward (NCDHW). One thread per input element; accumulates dy
// from every window whose argmax (strict `>`) lands here.
kernel void maxpool3d_backward(
    device const float* x   [[buffer(0)]],
    device const float* dy  [[buffer(1)]],
    device float* dx        [[buffer(2)]],
    constant uint4& p0      [[buffer(3)]],  // [N, C, D, H]
    constant uint4& p1      [[buffer(4)]],  // [W, D_out, H_out, W_out]
    constant uint4& p2      [[buffer(5)]],  // [kd, kh, kw, sd]
    constant uint4& p3      [[buffer(6)]],  // [sh, sw, pd, ph]
    constant uint&  pw      [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint N = p0.x, C = p0.y, D = p0.z, H = p0.w;
    uint W = p1.x, d_out = p1.y, h_out = p1.z, w_out = p1.w;
    uint kd = p2.x, kh = p2.y, kw = p2.z, sd = p2.w;
    uint sh = p3.x, sw = p3.y, pd = p3.z, ph = p3.w;
    uint total = N * C * D * H * W;
    if (gid >= total) return;

    uint iw = gid % W;
    uint q1 = gid / W;
    uint ih = q1 % H;
    uint q2 = q1 / H;
    uint id = q2 % D;
    uint q3 = q2 / D;
    uint cc = q3 % C;
    uint nn = q3 / C;
    uint base_nc = (nn * C + cc) * D * H * W;

    int do_lo = (int)id + (int)pd - (int)kd + 1;
    do_lo = do_lo <= 0 ? 0 : (do_lo + (int)sd - 1) / (int)sd;
    int do_hi = ((int)id + (int)pd) / (int)sd;
    int ho_lo = (int)ih + (int)ph - (int)kh + 1;
    ho_lo = ho_lo <= 0 ? 0 : (ho_lo + (int)sh - 1) / (int)sh;
    int ho_hi = ((int)ih + (int)ph) / (int)sh;
    int wo_lo = (int)iw + (int)pw - (int)kw + 1;
    wo_lo = wo_lo <= 0 ? 0 : (wo_lo + (int)sw - 1) / (int)sw;
    int wo_hi = ((int)iw + (int)pw) / (int)sw;

    float acc = 0.0f;
    for (int do_ = do_lo; do_ <= do_hi && do_ < (int)d_out; ++do_) {
        int dstart = do_ * (int)sd - (int)pd;
        for (int ho = ho_lo; ho <= ho_hi && ho < (int)h_out; ++ho) {
            int hstart = ho * (int)sh - (int)ph;
            for (int wo = wo_lo; wo <= wo_hi && wo < (int)w_out; ++wo) {
                int wstart = wo * (int)sw - (int)pw;
                float best = -INFINITY;
                int best_idx = -1;
                for (uint kz = 0; kz < kd; ++kz) {
                    int irz = dstart + (int)kz;
                    if (irz < 0 || irz >= (int)D) continue;
                    for (uint i = 0; i < kh; ++i) {
                        int ir = hstart + (int)i;
                        if (ir < 0 || ir >= (int)H) continue;
                        for (uint j = 0; j < kw; ++j) {
                            int ic = wstart + (int)j;
                            if (ic < 0 || ic >= (int)W) continue;
                            uint id3 = base_nc + ((uint)irz * H + (uint)ir) * W + (uint)ic;
                            float v = x[id3];
                            if (v > best) { best = v; best_idx = (int)id3; }
                        }
                    }
                }
                if (best_idx == (int)gid)
                    acc += dy[((((nn * C + cc) * d_out + (uint)do_) * h_out + (uint)ho) * w_out + (uint)wo)];
            }
        }
    }
    dx[gid] = acc;
}

// Conv3d backward-input (NCDHW gather). Weight [C_out, C_in/groups, kD, kH, kW].
kernel void conv3d_backward_input(
    device const float* dy  [[buffer(0)]],
    device const float* wt  [[buffer(1)]],
    device float* dx        [[buffer(2)]],
    constant uint4& a       [[buffer(3)]],  // [N, C_in, D, H]
    constant uint4& b       [[buffer(4)]],  // [W, C_out, D_out, H_out]
    constant uint4& cc      [[buffer(5)]],  // [W_out, kd, kh, kw]
    constant uint4& d       [[buffer(6)]],  // [sd, sh, sw, pd]
    constant uint4& e       [[buffer(7)]],  // [ph, pw, dd, dh]
    constant uint2& f       [[buffer(8)]],  // [dw, groups]
    uint gid [[thread_position_in_grid]]
) {
    uint N=a.x, C_in=a.y, D=a.z, H=a.w;
    uint W=b.x, C_out=b.y, D_out=b.z, H_out=b.w;
    uint W_out=cc.x, kd=cc.y, kh=cc.z, kw=cc.w;
    uint sd=d.x, sh=d.y, sw=d.z, pd=d.w;
    uint ph=e.x, pw=e.y, dd=e.z, dh=e.w;
    uint dw=f.x, groups=f.y;

    uint total = N * C_in * D * H * W;
    if (gid >= total) return;
    uint iw = gid % W;
    uint q1 = gid / W;
    uint ih = q1 % H;
    uint q2 = q1 / H;
    uint id = q2 % D;
    uint q3 = q2 / D;
    uint ci = q3 % C_in;
    uint nn = q3 / C_in;

    uint c_in_per_g = C_in / groups;
    uint c_out_per_g = C_out / groups;
    uint g = ci / c_in_per_g;
    uint ci_off = ci - g * c_in_per_g;
    uint co_start = g * c_out_per_g;

    float acc = 0.0f;
    for (uint kz = 0; kz < kd; ++kz) {
        int num_d = (int)id + (int)pd - (int)(kz * dd);
        if (num_d < 0 || (num_d % (int)sd) != 0) continue;
        int do_ = num_d / (int)sd;
        if (do_ >= (int)D_out) continue;
        for (uint ki = 0; ki < kh; ++ki) {
            int num_h = (int)ih + (int)ph - (int)(ki * dh);
            if (num_h < 0 || (num_h % (int)sh) != 0) continue;
            int ho = num_h / (int)sh;
            if (ho >= (int)H_out) continue;
            for (uint kj = 0; kj < kw; ++kj) {
                int num_w = (int)iw + (int)pw - (int)(kj * dw);
                if (num_w < 0 || (num_w % (int)sw) != 0) continue;
                int wo = num_w / (int)sw;
                if (wo >= (int)W_out) continue;
                for (uint co_off = 0; co_off < c_out_per_g; ++co_off) {
                    uint co = co_start + co_off;
                    float dyv = dy[((((nn * C_out + co) * D_out + (uint)do_) * H_out + (uint)ho) * W_out + (uint)wo)];
                    float wv = wt[((((co * c_in_per_g + ci_off) * kd + kz) * kh + ki) * kw + kj)];
                    acc += dyv * wv;
                }
            }
        }
    }
    dx[gid] = acc;
}

// Conv3d backward-weight (NCDHW). One thread per dw element.
kernel void conv3d_backward_weight(
    device const float* x   [[buffer(0)]],
    device const float* dy  [[buffer(1)]],
    device float* dw        [[buffer(2)]],
    constant uint4& a       [[buffer(3)]],  // [N, C_in, D, H]
    constant uint4& b       [[buffer(4)]],  // [W, C_out, D_out, H_out]
    constant uint4& cc      [[buffer(5)]],  // [W_out, kd, kh, kw]
    constant uint4& d       [[buffer(6)]],  // [sd, sh, sw, pd]
    constant uint4& e       [[buffer(7)]],  // [ph, pw, dd, dh]
    constant uint2& f       [[buffer(8)]],  // [dw, groups]
    uint gid [[thread_position_in_grid]]
) {
    uint N=a.x, C_in=a.y, D=a.z, H=a.w;
    uint W=b.x, C_out=b.y, D_out=b.z, H_out=b.w;
    uint W_out=cc.x, kd=cc.y, kh=cc.z, kw=cc.w;
    uint sd=d.x, sh=d.y, sw=d.z, pd=d.w;
    uint ph=e.x, pw=e.y, dd=e.z, dh=e.w;
    uint dw_dil=f.x, groups=f.y;

    uint c_in_per_g = C_in / groups;
    uint c_out_per_g = C_out / groups;
    uint total = C_out * c_in_per_g * kd * kh * kw;
    if (gid >= total) return;

    uint kj = gid % kw;
    uint q1 = gid / kw;
    uint ki = q1 % kh;
    uint q2 = q1 / kh;
    uint kz = q2 % kd;
    uint q3 = q2 / kd;
    uint ci_off = q3 % c_in_per_g;
    uint co = q3 / c_in_per_g;
    uint g = co / c_out_per_g;
    uint ci = g * c_in_per_g + ci_off;

    float acc = 0.0f;
    for (uint nn = 0; nn < N; ++nn) {
        for (uint do_ = 0; do_ < D_out; ++do_) {
            int id = (int)(do_ * sd + kz * dd) - (int)pd;
            if (id < 0 || id >= (int)D) continue;
            for (uint ho = 0; ho < H_out; ++ho) {
                int ih = (int)(ho * sh + ki * dh) - (int)ph;
                if (ih < 0 || ih >= (int)H) continue;
                for (uint wo = 0; wo < W_out; ++wo) {
                    int iw = (int)(wo * sw + kj * dw_dil) - (int)pw;
                    if (iw < 0 || iw >= (int)W) continue;
                    float dyv = dy[((((nn * C_out + co) * D_out + do_) * H_out + ho) * W_out + wo)];
                    float xv = x[((((nn * C_in + ci) * D + (uint)id) * H + (uint)ih) * W + (uint)iw)];
                    acc += dyv * xv;
                }
            }
        }
    }
    dw[gid] = acc;
}

// Conv2d backward-weight, pass 1 (batch-parallel). One thread per
// (n, co, ci_local, ki, kj) writes a per-sample partial sum into `part`,
// laid out [N, C_out, c_in_per_g, kh, kw]. Threads scale with N, so small
// kernels (conv1: 288 dw elems) no longer starve the GPU.
kernel void conv2d_backward_weight_partial(
    device const float* x   [[buffer(0)]],
    device const float* dy  [[buffer(1)]],
    device float* part      [[buffer(2)]],
    constant uint4& a       [[buffer(3)]],  // [N, C_in, H, W]
    constant uint4& b       [[buffer(4)]],  // [C_out, H_out, W_out, kh]
    constant uint4& cc      [[buffer(5)]],  // [kw, sh, sw, ph]
    constant uint4& d       [[buffer(6)]],  // [pw, dh, dw_dil, groups]
    uint3 gid [[thread_position_in_grid]]
) {
    uint N=a.x, C_in=a.y, H=a.z, W=a.w;
    uint C_out=b.x, H_out=b.y, W_out=b.z, kh=b.w;
    uint kw=cc.x, sh=cc.y, sw=cc.z, ph=cc.w;
    uint pw=d.x, dh=d.y, dwd=d.z, groups=d.w;
    uint c_in_per_g = C_in / groups;
    uint c_out_per_g = C_out / groups;
    uint wsz = c_in_per_g * kh * kw;   // per-(n,co) weight slab

    uint j = gid.x, co = gid.y, n = gid.z;
    if (j >= wsz || co >= C_out || n >= N) return;
    uint kj = j % kw;
    uint ki = (j / kw) % kh;
    uint ci_local = j / (kw * kh);
    uint g = co / c_out_per_g;
    uint ci = g * c_in_per_g + ci_local;

    float acc = 0.0f;
    for (uint ho = 0; ho < H_out; ++ho) {
        int hh = (int)(ho * sh + ki * dh) - (int)ph;
        if (hh < 0 || hh >= (int)H) continue;
        for (uint wo = 0; wo < W_out; ++wo) {
            int ww = (int)(wo * sw + kj * dwd) - (int)pw;
            if (ww < 0 || ww >= (int)W) continue;
            uint x_idx = ((n * C_in + ci) * H + (uint)hh) * W + (uint)ww;
            uint dy_idx = ((n * C_out + co) * H_out + ho) * W_out + wo;
            acc += x[x_idx] * dy[dy_idx];
        }
    }
    part[(n * C_out + co) * wsz + j] = acc;
}

// Conv2d backward-weight, pass 2: sum the per-sample partials over the batch.
// One thread per dw element. `wslab` = C_out * c_in_per_g * kh * kw.
kernel void conv2d_backward_weight_reduce(
    device const float* part [[buffer(0)]],
    device float* dw         [[buffer(1)]],
    constant uint2& dims     [[buffer(2)]],   // [N, wslab]
    uint gid [[thread_position_in_grid]]
) {
    uint N = dims.x, wslab = dims.y;
    if (gid >= wslab) return;
    float acc = 0.0f;
    for (uint n = 0; n < N; ++n) acc += part[n * wslab + gid];
    dw[gid] = acc;
}

// Gather along an arbitrary axis. One thread per output element. Output
// is laid out as [outer, num_idx, trailing]; source as [outer, axis_dim, trailing].
kernel void gather_axis(
    device const float* table [[buffer(0)]],
    device const float* idx   [[buffer(1)]],
    device float* dst         [[buffer(2)]],
    constant uint& outer      [[buffer(3)]],
    constant uint& axis_dim   [[buffer(4)]],
    constant uint& num_idx    [[buffer(5)]],
    constant uint& trailing   [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint o = gid.z;
    uint k = gid.y;
    uint t = gid.x;
    if (o >= outer || k >= num_idx || t >= trailing) return;
    uint row = (uint)(idx[k]);
    dst[(o * num_idx + k) * trailing + t] =
        table[(o * axis_dim + row) * trailing + t];
}

// General N-D transpose. One thread per output element. The encoder packs
// out_dims and in_strides into a single u32 buffer of length 2*rank:
//   buffer = [out_dim_0, ..., out_dim_{r-1}, in_stride_0, ..., in_stride_{r-1}]
// Rank is bounded at 8 (sufficient for current models).
kernel void transpose_nd(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& rank     [[buffer(2)]],
    constant uint& total    [[buffer(3)]],
    constant uint* meta     [[buffer(4)]],   // [out_dims..., in_strides...]
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) return;
    uint src_idx = 0;
    uint remaining = gid;
    // Decompose flat output index into multi-dim coords (outer-to-inner)
    // using stride math, then accumulate the source index from in_strides.
    // Compute denominators on the fly to avoid a separate divisor table.
    uint stride_rem = total;
    for (uint d = 0; d < rank; ++d) {
        uint dim = meta[d];
        stride_rem /= dim;
        uint coord = remaining / stride_rem;
        remaining = remaining - coord * stride_rem;
        src_idx += coord * meta[rank + d];
    }
    dst[gid] = src[src_idx];
}

// F16 reindex — same index math as `transpose_nd`, half elements. Emitted for
// Expand/Transpose/repeat_kv on an F16-resident KV cache (`RLX_QWEN3_F16_KV`).
kernel void transpose_nd_h(
    device const half* src [[buffer(0)]],
    device half* dst       [[buffer(1)]],
    constant uint& rank    [[buffer(2)]],
    constant uint& total   [[buffer(3)]],
    constant uint* meta    [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) return;
    uint src_idx = 0;
    uint remaining = gid;
    uint stride_rem = total;
    for (uint d = 0; d < rank; ++d) {
        uint dim = meta[d];
        stride_rem /= dim;
        uint coord = remaining / stride_rem;
        remaining = remaining - coord * stride_rem;
        src_idx += coord * meta[rank + d];
    }
    dst[gid] = src[src_idx];
}

// Rank-2 swap (row-major [rows, cols] → [cols, rows]). Cheaper than transpose_nd
// for attention/layout reshapes — 2D thread grid, coalesced reads.
kernel void transpose_2d_f32(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& rows     [[buffer(2)]],
    constant uint& cols     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint r = gid.x;
    uint c = gid.y;
    if (r >= rows || c >= cols) return;
    dst[c * rows + r] = src[r * cols + c];
}

// Tiled transpose for large matrices (32x32 tile staged through threadgroup memory).
// Threadgroup size: (32, 8, 1). Each thread loads 4 rows of the tile.
kernel void transpose_2d_tiled_f32(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& rows     [[buffer(2)]],
    constant uint& cols     [[buffer(3)]],
    ushort2 tid [[thread_position_in_threadgroup]],
    ushort2 tgp [[threadgroup_position_in_grid]]
) {
    // Coalesced both ways: tid.x (the 32-wide fast lane) indexes the CONTIGUOUS
    // dimension on BOTH the global load (src col) and the store (dst row); the
    // transpose happens in the padded [32][33] threadgroup tile (+1 avoids bank
    // conflicts). The prior version indexed the ROW with tid.x on both sides, so
    // both global accesses were strided and the tile delivered ZERO coalescing —
    // profiling measured this rewrite at 1.1–2.7× (2.7× on the
    // common 192² weight-grad transpose). Same grid (tgp.x=row-block, tgp.y=col-block).
    threadgroup float tile[32][33];
    uint row_base = (uint)tgp.x * 32u;
    uint col_base = (uint)tgp.y * 32u;
    uint col = col_base + (uint)tid.x;
    for (uint i = 0; i < 32; i += 8) {
        uint row = row_base + (uint)tid.y + i;
        if (row < rows && col < cols) {
            tile[tid.y + i][tid.x] = src[row * cols + col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint drow = row_base + (uint)tid.x;
    for (uint i = 0; i < 32; i += 8) {
        uint dcol = col_base + (uint)tid.y + i;
        if (drow < rows && dcol < cols) {
            dst[dcol * rows + drow] = tile[tid.x][tid.y + i];
        }
    }
}

// Batched swap of the last two dims: src [batch, rows, cols] -> dst [batch, cols, rows]
kernel void transpose_last2_batched_f32(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& batch    [[buffer(2)]],
    constant uint& rows     [[buffer(3)]],
    constant uint& cols     [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint b = gid.z;
    uint r = gid.x;
    uint c = gid.y;
    if (b >= batch || r >= rows || c >= cols) return;
    uint src_base = b * rows * cols;
    uint dst_base = b * rows * cols;
    dst[dst_base + c * rows + r] = src[src_base + r * cols + c];
}

// Tiled batched last2 transpose. Dispatch threadgroups over (rows, cols, batch).
// Threadgroup size: (32, 8, 1).
kernel void transpose_last2_batched_tiled_f32(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& batch    [[buffer(2)]],
    constant uint& rows     [[buffer(3)]],
    constant uint& cols     [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgp [[threadgroup_position_in_grid]]
) {
    threadgroup float tile[32][33];
    uint b = tgp.z;
    if (b >= batch) return;
    uint r0 = tgp.x * 32u + tid.x;
    uint c0 = tgp.y * 32u + tid.y;
    uint base = b * rows * cols;
    for (uint i = 0; i < 32; i += 8) {
        uint c = c0 + i;
        if (r0 < rows && c < cols) {
            tile[tid.x][tid.y + i] = src[base + r0 * cols + c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rr0 = tgp.y * 32u + tid.x; // out row (col index)
    uint cc0 = tgp.x * 32u + tid.y; // out col (row index)
    for (uint i = 0; i < 32; i += 8) {
        uint rr = rr0;
        uint cc = cc0 + i;
        if (rr < cols && cc < rows) {
            dst[base + rr * rows + cc] = tile[tid.y + i][tid.x];
        }
    }
}

// `[B, A, C, D] → [B, C, A, D]` — swap axes 1 and 2 with trailing axis contiguous.
kernel void transpose_swap12_batched_trail_f32(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& batch    [[buffer(2)]],
    constant uint& rows     [[buffer(3)]],
    constant uint& cols     [[buffer(4)]],
    constant uint& trail    [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint bd = gid.z;
    uint b = bd / trail;
    uint d = bd % trail;
    uint r = gid.x;
    uint c = gid.y;
    if (b >= batch || r >= rows || c >= cols) return;
    uint block = rows * cols * trail;
    uint src_idx = b * block + c * rows * trail + r * trail + d;
    uint dst_idx = b * block + r * cols * trail + c * trail + d;
    dst[dst_idx] = src[src_idx];
}

kernel void transpose_swap12_batched_trail_tiled_f32(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& batch    [[buffer(2)]],
    constant uint& rows     [[buffer(3)]],
    constant uint& cols     [[buffer(4)]],
    constant uint& trail    [[buffer(5)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgp [[threadgroup_position_in_grid]]
) {
    threadgroup float tile[32][33];
    uint bd = tgp.z;
    uint b = bd / trail;
    uint d = bd % trail;
    if (b >= batch) return;
    uint block = rows * cols * trail;
    uint plane = b * block + d;
    uint r0 = tgp.x * 32u + tid.x;
    uint c0 = tgp.y * 32u + tid.y;
    for (uint i = 0; i < 32; i += 8) {
        uint c = c0 + i;
        if (r0 < rows && c < cols) {
            tile[tid.x][tid.y + i] =
                src[plane + c * rows * trail + r0 * trail];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rr0 = tgp.y * 32u + tid.x;
    uint cc0 = tgp.x * 32u + tid.y;
    for (uint i = 0; i < 32; i += 8) {
        uint rr = rr0;          // output column index (in `cols` range)
        uint cc = cc0 + i;      // output row index    (in `rows` range)
        if (rr < cols && cc < rows) {
            // Output layout [batch, rows, cols, trail]: element (row=cc, col=rr)
            // lives at cc*cols*trail + rr*trail (+ plane carries batch & trail).
            dst[plane + cc * cols * trail + rr * trail] = tile[tid.y + i][tid.x];
        }
    }
}

// Two-phase scatter-add: phase 0 zeros the output buffer, phase 1
// accumulates updates atomically. Atomic add is required because
// multiple updates may target the same destination row from different
// threads. `op_phase`: 0 = zero, 1 = accumulate.
//
// Each phase is a single dispatch: phase 0 runs over `out_total` threads,
// phase 1 over `num_updates * trailing` threads. The encoder fires both
// in sequence within one command buffer.
kernel void scatter_add_zero(
    device atomic_float* dst [[buffer(0)]],
    constant uint& out_total [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= out_total) return;
    atomic_store_explicit(&dst[gid], 0.0f, memory_order_relaxed);
}

kernel void scatter_add_accumulate(
    device const float* updates [[buffer(0)]],
    device const float* indices [[buffer(1)]],
    device atomic_float* dst    [[buffer(2)]],
    constant uint& trailing     [[buffer(3)]],
    constant uint& num_updates  [[buffer(4)]],
    constant uint& out_dim      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;     // which update
    uint j = gid.x;     // which trailing element
    if (i >= num_updates || j >= trailing) return;
    uint row = (uint)indices[i];
    if (row >= out_dim) return;            // OOB safety
    float v = updates[i * trailing + j];
    // Native hardware-arbitrated float atomic-add. Replaces the old f32
    // CAS-retry loop (as_type<uint> + compare_exchange_weak); profiling
    // measured 10.5× under codebook-grad contention (~36 threads/entry, where
    // the CAS spun). `atomic_float` compiles on metal3.1+ (all Apple-silicon
    // Metal targets); same bits + same unordered-fadd (non-)determinism as the
    // CAS.
    atomic_fetch_add_explicit(&dst[row * trailing + j], v, memory_order_relaxed);
}

// Indexed batched matmul (MoE GEMM). One thread per output element
// (i, j). Token i looks up its expert via expert_idx, then dot-products
// the row of `input` against the column of `weight[expert_idx[i]]`.
kernel void grouped_matmul(
    device const float* input      [[buffer(0)]],
    device const float* weight     [[buffer(1)]],
    device const float* expert_idx [[buffer(2)]],
    device float* dst              [[buffer(3)]],
    constant uint& m               [[buffer(4)]],
    constant uint& k_dim           [[buffer(5)]],
    constant uint& n               [[buffer(6)]],
    constant uint& num_experts     [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;
    if (i >= m || j >= n) return;
    uint e = (uint)(expert_idx[i]);
    if (e >= num_experts) return;          // OOB safety
    uint w_base = e * k_dim * n;
    uint in_base = i * k_dim;
    float acc = 0.0f;
    for (uint kk = 0; kk < k_dim; ++kk) {
        acc += input[in_base + kk] * weight[w_base + kk * n + j];
    }
    dst[i * n + j] = acc;
}

// Top-K indices along the last axis. One thread per output row. Repeated
// argmax with masking — O(k * axis_dim) per row; fine for small k (MoE
// typical k=2–8). Each thread maintains its own scratch space in private
// memory, no threadgroup coordination needed.
//
// Important: rlx writes float32-encoded indices; downstream Gather reads
// them via `(uint)idx[k]`. Cast on store mirrors that.
kernel void topk_lastax(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& axis_dim [[buffer(2)]],
    constant uint& k        [[buffer(3)]],
    uint o [[thread_position_in_grid]]
) {
    // Hard cap on axis_dim — guards the on-chip scratch. MoE expert
    // counts top out around 256 in practice; raise this if a real
    // workload needs more.
    const uint MAX_AXIS = 1024;
    if (axis_dim > MAX_AXIS) return;

    float scratch[MAX_AXIS];
    uint base = o * axis_dim;
    for (uint i = 0; i < axis_dim; ++i) scratch[i] = src[base + i];

    uint out_base = o * k;
    for (uint ki = 0; ki < k; ++ki) {
        float best_v = scratch[0];
        uint  best_i = 0;
        for (uint i = 1; i < axis_dim; ++i) {
            float v = scratch[i];
            if (v > best_v) { best_v = v; best_i = i; }
        }
        dst[out_base + ki] = (float)best_i;
        scratch[best_i] = -INFINITY;
    }
}

// Reduce over a contiguous axis range. Input layout [outer, reduced, inner];
// output [outer, inner]. One thread per output element walks `reduced`
// values with stride `inner`. `op_kind`: 0=Sum 1=Mean 2=Max 3=Min 4=Prod.
//
// Trade-off: a serial reduction loop per thread is slower than threadgroup
// reduction when `reduced` is large, but it generalises trivially to any
// axis range and avoids the per-row threadgroup setup cost. For the shapes
// we care about (Reduce::Sum on 60×768 is 22 µs CPU vs 135 µs Metal — the
// wait latency dominates either way), kernel choice barely moves the
// needle. Revisit if a launch-bound reduction shows up.
kernel void reduce_axes(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& reduced  [[buffer(2)]],
    constant uint& inner    [[buffer(3)]],
    constant uint& op_kind  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;            // inner axis index
    uint o = gid.y;            // outer axis index
    if (i >= inner) return;
    float acc;
    if      (op_kind == 2) acc = -INFINITY;
    else if (op_kind == 3) acc =  INFINITY;
    else if (op_kind == 4) acc =  1.0f;
    else                   acc =  0.0f;        // Sum / Mean

    uint base = o * reduced * inner + i;
    for (uint r = 0; r < reduced; ++r) {
        float v = src[base + r * inner];
        if      (op_kind == 0 || op_kind == 1) acc += v;
        else if (op_kind == 2) acc = max(acc, v);
        else if (op_kind == 3) acc = min(acc, v);
        else                   acc *= v;
    }
    if (op_kind == 1) acc /= float(reduced);
    dst[o * inner + i] = acc;
}

// SIMD-parallel Sum/Mean reduction: one 32-wide threadgroup (== 1 SIMD group)
// per output element; the 32 lanes split the `reduced` axis and combine with
// simd_sum. `reduce_axes` above dispatches ONE thread per output, each serially
// summing the whole reduced axis — for grad-sums (bias/beta: reduced = batch·seq
// ≈ 1024, few outputs) that is a serial reduction at low occupancy. Sum/Mean
// only (op_kind 0/1); max/min/prod stay on the scalar kernel.
kernel void reduce_axes_sum_simd(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& reduced  [[buffer(2)]],
    constant uint& inner    [[buffer(3)]],
    constant uint& op_kind  [[buffer(4)]],
    uint out_idx [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tsize   [[threads_per_threadgroup]]
) {
    uint i = out_idx % inner;
    uint o = out_idx / inner;
    uint base = o * reduced * inner + i;
    float local = 0.0f;
    for (uint r = tid; r < reduced; r += tsize) {
        local += src[base + r * inner];
    }
    float total = simd_sum(local);
    if (tid == 0) {
        dst[out_idx] = (op_kind == 1u) ? total / float(reduced) : total;
    }
}

// Ternary select: cond != 0 ? a : b. cond is treated as bool via != 0.
kernel void elem_where(
    device const float* cond [[buffer(0)]],
    device const float* a    [[buffer(1)]],
    device const float* b    [[buffer(2)]],
    device float* out        [[buffer(3)]],
    constant uint& len       [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    out[gid] = cond[gid] != 0.0f ? a[gid] : b[gid];
}

/// Where with optional scalar broadcast on any operand.
/// `flags`: bit0=cond scalar, bit1=a scalar, bit2=b scalar.
kernel void elem_where_bcast(
    device const float* cond [[buffer(0)]],
    device const float* a    [[buffer(1)]],
    device const float* b    [[buffer(2)]],
    device float* out        [[buffer(3)]],
    constant uint& len       [[buffer(4)]],
    constant uint& flags     [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    float c = (flags & 1u) ? cond[0] : cond[gid];
    float x = (flags & 2u) ? a[0] : a[gid];
    float y = (flags & 4u) ? b[0] : b[gid];
    out[gid] = c != 0.0f ? x : y;
}

// Single-rounded fused multiply-add: out = fma(a, b, c). MSL `fma` is a true
// fused op (one rounding) — required for compensated / error-free-transform math.
kernel void elem_fma(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* c [[buffer(2)]],
    device float* out     [[buffer(3)]],
    constant uint& len    [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    out[gid] = fma(a[gid], b[gid], c[gid]);
}

// Element-wise ReLU / activation backward. `op` matches CUDA
// `activation_op_id` / unary forward ids (0=relu … 16=atan).
// Formulas mirror rlx-cpu `activation_backward_kernel`.
kernel void relu_backward(
    device const float* x  [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* dx       [[buffer(2)]],
    constant uint& len     [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    float xv = x[gid];
    float dyv = dy[gid];
    dx[gid] = (xv > 0.0f) ? dyv : 0.0f;
}

// @@RLX_ACTIVATION_BACKWARD@@
kernel void activation_backward(
    device const float* x  [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* dx       [[buffer(2)]],
    constant uint& len     [[buffer(3)]],
    constant uint& op      [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    float xv = x[gid];
    float dyv = dy[gid];
    // Derivative dispatch generated from the rlxsl manifest (auto-differentiated
    // from the forward); definition substituted for the marker below at runtime.
    dx[gid] = rlx_activation_backward(op, xv, dyv);
}

// C64 Wirtinger surface on interleaved [re, im] f32 pairs (mirrors
// `complex_wirtinger.cu` / rlx-cpu `exec_complex_norm_sq{,_backward}_f32` /
// `exec_conjugate_c64`). Dispatched over the complex-element index `k ∈ [0, n)`.
// Buffer pointers are already advanced to each tensor's byte offset.
kernel void complex_norm_sq(
    device const float* in [[buffer(0)]],
    device float* out      [[buffer(1)]],
    constant uint& n       [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float re = in[2u * gid];
    float im = in[2u * gid + 1u];
    out[gid] = re * re + im * im;
}

kernel void complex_norm_sq_backward(
    device const float* z  [[buffer(0)]],
    device const float* g  [[buffer(1)]],
    device float* dz       [[buffer(2)]],
    constant uint& n       [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float re = z[2u * gid];
    float im = z[2u * gid + 1u];
    float gv = g[gid];
    dz[2u * gid]      = gv * re;
    dz[2u * gid + 1u] = gv * im;
}

kernel void conjugate_c64(
    device const float* in [[buffer(0)]],
    device float* out      [[buffer(1)]],
    constant uint& n       [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[2u * gid]      =  in[2u * gid];
    out[2u * gid + 1u] = -in[2u * gid + 1u];
}

// Ternary-pruned radix-2 butterfly stage (interleaved C64 [batch, n_fft, 2]).
// Mirrors CUDA `fft_butterfly_stage.cu` / CPU `execute_fft_butterfly_stage_f32`.
// One thread per (batch, butterfly); gate=0 copies the pair, else twiddle + optional rev.
struct FftButterflyStageParams {
    uint batch;
    uint n_fft;
    uint stage;
    uint n_half;
};

kernel void fft_butterfly_stage(
    device const float* state [[buffer(0)]],
    device float* out         [[buffer(1)]],
    device const float* gate  [[buffer(2)]],
    device const float* rev   [[buffer(3)]],
    device const float* tw_re [[buffer(4)]],
    device const float* tw_im [[buffer(5)]],
    constant FftButterflyStageParams& p [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint bf = gid.x;
    uint b = gid.y;
    if (b >= p.batch || bf >= p.n_half) return;

    uint stride = 1u << p.stage;
    uint row_elems = p.n_fft * 2u;
    device const float* inp = state + b * row_elems;
    device float* o = out + b * row_elems;

    uint group = bf / stride;
    uint k = bf % stride;
    uint i0 = group * 2u * stride + k;
    uint i1 = i0 + stride;

    if (gate[bf] == 0.0f) {
        o[i0 * 2u]      = inp[i0 * 2u];
        o[i0 * 2u + 1u] = inp[i0 * 2u + 1u];
        o[i1 * 2u]      = inp[i1 * 2u];
        o[i1 * 2u + 1u] = inp[i1 * 2u + 1u];
        return;
    }

    float w_re = tw_re[bf];
    float w_im = tw_im[bf];
    float in_a_re = inp[i0 * 2u];
    float in_a_im = inp[i0 * 2u + 1u];
    float in_b_re = inp[i1 * 2u];
    float in_b_im = inp[i1 * 2u + 1u];

    float b_re = in_b_re * w_re - in_b_im * w_im;
    float b_im = in_b_re * w_im + in_b_im * w_re;
    float top_re = in_a_re + b_re;
    float top_im = in_a_im + b_im;
    float bot_re = in_a_re - b_re;
    float bot_im = in_a_im - b_im;

    float oa_re, oa_im, ob_re, ob_im;
    if (rev[bf] >= 0.5f) {
        oa_re = bot_re; oa_im = bot_im;
        ob_re = top_re; ob_im = top_im;
    } else {
        oa_re = top_re; oa_im = top_im;
        ob_re = bot_re; ob_im = bot_im;
    }
    o[i0 * 2u]      = oa_re;
    o[i0 * 2u + 1u] = oa_im;
    o[i1 * 2u]      = ob_re;
    o[i1 * 2u + 1u] = ob_im;
}

// FakeQuantize forward: clamp(round(x / s), -q_max, q_max) * s.
// Matches `rlx_cpu::thunk::ops::quant::exec_fake_quantize` for Fixed and
// PerBatch (EMA stays on HostOp). Channel layout: c = (i / inner) % chan_dim.
// Rounding: Rust `f32::round` (half away from zero), not ties-to-even.
struct FakeQuantizeParams {
    uint n;
    uint chan_dim;
    uint inner;
    float q_max;
};

inline float apply_fq(float x, float s, float q_max) {
    float scaled = x / s;
    float rounded = sign(scaled) * floor(fabs(scaled) + 0.5f);
    float qv = clamp(rounded, -q_max, q_max);
    return qv * s;
}

inline uint fq_channel_of(uint i, uint chan_dim, uint inner) {
    if (chan_dim <= 1u) return 0u;
    return (i / inner) % chan_dim;
}

// One thread per element. Scale from `scale[c]` (Fixed).
kernel void fake_quantize_fixed(
    device const float* in [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant FakeQuantizeParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.n) return;
    uint c = fq_channel_of(gid, p.chan_dim, p.inner);
    float s = max(scale[c], 1e-12f);
    out[gid] = apply_fq(in[gid], s, p.q_max);
}

// One thread per channel. s = max(|x|) / q_max, then quantize that channel
// (PerBatch).
kernel void fake_quantize_perbatch(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant FakeQuantizeParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint c = gid;
    if (c >= p.chan_dim) return;

    float max_abs = 0.0f;
    uint stride = p.chan_dim * p.inner;
    uint outer = p.n / max(stride, 1u);
    for (uint o = 0u; o < outer; o++) {
        uint base = o * stride + c * p.inner;
        for (uint j = 0u; j < p.inner; j++) {
            max_abs = max(max_abs, fabs(in[base + j]));
        }
    }
    // axis=None: chan_dim=1, inner=n → outer=1, single scan of all elements.
    if (outer * stride != p.n) {
        for (uint i = 0u; i < p.n; i++) {
            if (fq_channel_of(i, p.chan_dim, p.inner) == c) {
                max_abs = max(max_abs, fabs(in[i]));
            }
        }
    }

    float s = max(max_abs / p.q_max, 1e-12f);

    for (uint o = 0u; o < outer; o++) {
        uint base = o * stride + c * p.inner;
        for (uint j = 0u; j < p.inner; j++) {
            uint idx = base + j;
            out[idx] = apply_fq(in[idx], s, p.q_max);
        }
    }
    if (outer * stride != p.n) {
        for (uint i = 0u; i < p.n; i++) {
            if (fq_channel_of(i, p.chan_dim, p.inner) == c) {
                out[i] = apply_fq(in[i], s, p.q_max);
            }
        }
    }
}

// In-place ReLU: data = max(0, data)
kernel void relu_inplace(
    device float* data [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    data[gid] = max(0.0f, data[gid]);
}

// In-place sigmoid: 1 / (1 + exp(-x))
kernel void sigmoid_inplace(
    device float* data [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    data[gid] = 1.0f / (1.0f + exp(-data[gid]));
}

// In-place tan
kernel void tan_inplace(
    device float* data [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    data[gid] = tan(data[gid]);
}

// In-place atan
kernel void atan_inplace(
    device float* data [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    data[gid] = atan(data[gid]);
}

// In-place sin
kernel void sin_inplace(
    device float* data [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    data[gid] = sin(data[gid]);
}

// In-place cos
kernel void cos_inplace(
    device float* data [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    data[gid] = cos(data[gid]);
}

// In-place tanh
kernel void tanh_inplace(
    device float* data [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    // Clamp to the tanh saturation range: Metal's fast-math `tanh` returns NaN for
    // large |x| (it evaluates via exp, which overflows) — e.g. VITS posterior
    // WaveNet pre-activations reach ~300. tanh(±15)≈±1 to f32 precision, matching
    // CPU's stable tanh; clamp also folds ±inf to ±15.
    data[gid] = tanh(clamp(data[gid], -15.0f, 15.0f));
}

// In-place exp / log / sqrt / rsqrt / neg / abs — one kernel each so the
// dispatch path stays uniform with the existing `*_inplace` family.
kernel void exp_inplace(
    device float* data [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) { if (gid >= len) return; data[gid] = exp(data[gid]); }

kernel void log_inplace(
    device float* data [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) { if (gid >= len) return; data[gid] = log(data[gid]); }

kernel void sqrt_inplace(
    device float* data [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) { if (gid >= len) return; data[gid] = sqrt(data[gid]); }

kernel void rsqrt_inplace(
    device float* data [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) { if (gid >= len) return; data[gid] = rsqrt(data[gid]); }

kernel void rec_inplace(
    device float* data [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) { if (gid >= len) return; data[gid] = 1.0f / data[gid]; }

kernel void neg_inplace(
    device float* data [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) { if (gid >= len) return; data[gid] = -data[gid]; }

kernel void abs_inplace(
    device float* data [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) { if (gid >= len) return; data[gid] = abs(data[gid]); }

kernel void round_inplace(
    device float* data [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) { if (gid >= len) return; data[gid] = round(data[gid]); }

// Standalone softmax along the last axis. One threadgroup per row,
// reduces max + exp-sum across the row, then normalizes. tg_size is
// the actual number of threads per group (passed via threads_per_threadgroup).
kernel void softmax_lastax(
    device float* data    [[buffer(0)]],
    constant uint& cols   [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float partial[256];
    uint base = row * cols;

    // Pass 1: find row max for numerical stability.
    float local_max = -INFINITY;
    for (uint i = tid; i < cols; i += tsize) {
        local_max = max(local_max, data[base + i]);
    }
    partial[tid] = (float)local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] = max(partial[tid], partial[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = partial[0];

    // Pass 2: exp(x - max) and sum.
    float local_sum = 0.0f;
    for (uint i = tid; i < cols; i += tsize) {
        float e = exp(data[base + i] - row_max);
        data[base + i] = e;
        local_sum += e;
    }
    partial[tid] = (float)local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / partial[0];

    // Pass 3: normalize.
    for (uint i = tid; i < cols; i += tsize) {
        data[base + i] *= inv_sum;
    }
}

// Causal-bounded row softmax for the attention-backward score recompute. Row
// r = slot*sq + qi softmaxes over only its causal band [0, qi]; cols ki>qi are
// masked sentinels (-1e4) whose exp underflows to EXACTLY 0, so they contribute
// nothing to the row max/sum — making this bit-identical to `softmax_lastax` for
// the band that the consumers (attn_bwd dv/ds) actually read. Cols>qi are left
// untouched (never read). ~2× less work per row for causal prefill.
kernel void softmax_lastax_causal(
    device float* data    [[buffer(0)]],
    constant uint& cols   [[buffer(1)]],
    constant uint& sq     [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float partial[256];
    uint qi = row % sq;
    uint active = min(qi + 1u, cols);   // causal band [0, qi]
    uint base = row * cols;

    float local_max = -INFINITY;
    for (uint i = tid; i < active; i += tsize) {
        local_max = max(local_max, data[base + i]);
    }
    partial[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] = max(partial[tid], partial[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = partial[0];

    float local_sum = 0.0f;
    for (uint i = tid; i < active; i += tsize) {
        float e = exp(data[base + i] - row_max);
        data[base + i] = e;
        local_sum += e;
    }
    partial[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / partial[0];

    for (uint i = tid; i < active; i += tsize) {
        data[base + i] *= inv_sum;
    }
}

// Fused dense / soft-label softmax cross-entropy along the last axis.
// One threadgroup per row computes, numerically stably,
//   loss[n] = logsumexp(logits[n]) - Σ_c targets[n,c]·logits[n,c]
// via three threadgroup reductions (row max, Σexp, Σtargets·logits).
// `cols` is the class count C; output is one scalar per row.
kernel void softmax_cross_entropy_dense(
    device const float* logits  [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* out           [[buffer(2)]],
    constant uint& cols         [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float partial[256];
    uint base = row * cols;

    // Pass 1: row max for numerical stability.
    float local_max = -INFINITY;
    for (uint i = tid; i < cols; i += tsize) {
        local_max = max(local_max, logits[base + i]);
    }
    partial[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] = max(partial[tid], partial[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = partial[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: Σ exp(x - max) and Σ targets·logits in one sweep.
    float local_sum = 0.0f;
    float local_dot = 0.0f;
    for (uint i = tid; i < cols; i += tsize) {
        float v = logits[base + i];
        local_sum += exp(v - row_max);
        local_dot += targets[base + i] * v;
    }
    partial[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_exp = partial[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    partial[tid] = local_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dot = partial[0];

    if (tid == 0) {
        out[row] = (row_max + log(sum_exp)) - dot;
    }
}

// Softmax cross-entropy with integer labels (forward). One threadgroup per row.
// loss[row] = logsumexp(logits[row]) - logits[row, label]. Replaces the
// softmax + one-hot(compare/where) + gather decomposition on Metal.
kernel void softmax_cross_entropy_with_logits(
    device const float* logits [[buffer(0)]],
    device const float* labels [[buffer(1)]],
    device float* out          [[buffer(2)]],
    constant uint& cols        [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float partial[256];
    uint base = row * cols;

    float local_max = -INFINITY;
    for (uint i = tid; i < cols; i += tsize) local_max = max(local_max, logits[base + i]);
    partial[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] = max(partial[tid], partial[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = partial[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint i = tid; i < cols; i += tsize) local_sum += exp(logits[base + i] - row_max);
    partial[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        uint label = (uint)labels[row];
        out[row] = (row_max + log(partial[0])) - logits[base + label];
    }
}

// Softmax cross-entropy backward (integer labels). One threadgroup per row.
// dlogits[row,k] = (softmax(logits[row])[k] - [k==label]) * d_loss[row].
kernel void softmax_cross_entropy_backward(
    device const float* logits [[buffer(0)]],
    device const float* labels [[buffer(1)]],
    device const float* d_loss [[buffer(2)]],
    device float* dlogits      [[buffer(3)]],
    constant uint& cols        [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float partial[256];
    uint base = row * cols;

    float local_max = -INFINITY;
    for (uint i = tid; i < cols; i += tsize) local_max = max(local_max, logits[base + i]);
    partial[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] = max(partial[tid], partial[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = partial[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint i = tid; i < cols; i += tsize) local_sum += exp(logits[base + i] - row_max);
    partial[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / partial[0];
    float scale = d_loss[row];
    uint label = (uint)labels[row];
    for (uint k = tid; k < cols; k += tsize) {
        float p = exp(logits[base + k] - row_max) * inv_sum;
        dlogits[base + k] = (p - (k == label ? 1.0f : 0.0f)) * scale;
    }
}

// Embedding lookup: out[i, .] = table[idx[i], .]
// table: [vocab, trailing], idx: [num_idx], out: [num_idx, trailing]
kernel void gather_axis0(
    device const float* table [[buffer(0)]],
    device const float* idx   [[buffer(1)]],
    device float* out         [[buffer(2)]],
    constant uint& num_idx    [[buffer(3)]],
    constant uint& trailing   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;
    if (i >= num_idx || j >= trailing) return;
    uint row = uint(idx[i]);
    out[i * trailing + j] = table[row * trailing + j];
}

// Narrow / slice along last axis. src is [outer, src_axis], dst is [outer, len].
// Each invocation copies one (outer, j) element.
kernel void narrow_lastax(
    device const char* arena_src [[buffer(0)]],
    device char* arena_dst       [[buffer(1)]],
    constant uint& outer    [[buffer(2)]],
    constant uint& src_axis [[buffer(3)]],
    constant uint& start    [[buffer(4)]],
    constant uint& len      [[buffer(5)]],
    constant ulong& src_byte_off [[buffer(6)]],
    constant ulong& dst_byte_off [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Task #50: > 4 GB activations need ulong byte offsets.
    device const float* src = (device const float*)(arena_src + src_byte_off);
    device float* dst       = (device float*)(arena_dst + dst_byte_off);
    uint i = gid.y;
    uint j = gid.x;
    if (i >= outer || j >= len) return;
    dst[i * len + j] = src[i * src_axis + start + j];
}

// Vectorized narrow for aligned shapes: src/dst treated as packed_float4.
// Requirements (enforced in encoder): start, src_axis, len are divisible by 4.
kernel void narrow_lastax4(
    device const char* arena_src    [[buffer(0)]],
    device char* arena_dst          [[buffer(1)]],
    constant uint& outer            [[buffer(2)]],
    constant uint& src_axis4        [[buffer(3)]],
    constant uint& start4           [[buffer(4)]],
    constant uint& len4             [[buffer(5)]],
    constant ulong& src_byte_off    [[buffer(6)]],
    constant ulong& dst_byte_off    [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    device const packed_float4* src = (device const packed_float4*)(arena_src + src_byte_off);
    device packed_float4* dst       = (device packed_float4*)(arena_dst + dst_byte_off);
    uint i = gid.y;
    uint j4 = gid.x;
    if (i >= outer || j4 >= len4) return;
    dst[i * len4 + j4] = src[i * src_axis4 + start4 + j4];
}

// `dst` widened to ulong (task #50: ≥4 GB models have activation byte
// offsets that exceed u32 — truncated wrap-around made K_rep narrows
// write to the wrong slot and SDPA saw all-zero K_rep).
struct NarrowSeg {
    ulong dst;
    uint start;
    uint len;
};

// Concat VJP: multiple last-axis slices from one source in one dispatch.
kernel void split_lastax(
    device const char* arena_src [[buffer(0)]],
    device char* arena    [[buffer(1)]],
    constant uint& outer    [[buffer(2)]],
    constant uint& src_axis [[buffer(3)]],
    constant uint& num_seg  [[buffer(4)]],
    constant NarrowSeg* segs [[buffer(5)]],
    constant ulong& src_byte_off [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    device const float* src = (device const float*)(arena_src + src_byte_off);
    uint s = gid.z;
    uint i = gid.y;
    uint j = gid.x;
    if (s >= num_seg) return;
    NarrowSeg seg = segs[s];
    if (i >= outer || j >= seg.len) return;
    device float* dst = (device float*)(arena + seg.dst);
    dst[i * seg.len + j] = src[i * src_axis + seg.start + j];
}

kernel void split_lastax4(
    device const char* arena_src    [[buffer(0)]],
    device char* arena            [[buffer(1)]],
    constant uint& outer            [[buffer(2)]],
    constant uint& src_axis4        [[buffer(3)]],
    constant uint& num_seg          [[buffer(4)]],
    constant NarrowSeg* segs        [[buffer(5)]],
    constant ulong& src_byte_off    [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    device const packed_float4* src = (device const packed_float4*)(arena_src + src_byte_off);
    uint s = gid.z;
    uint i = gid.y;
    uint j4 = gid.x;
    if (s >= num_seg) return;
    NarrowSeg seg = segs[s];
    uint len4 = seg.len / 4u;
    if (i >= outer || j4 >= len4) return;
    device packed_float4* dst = (device packed_float4*)(arena + seg.dst);
    uint start4 = seg.start / 4u;
    dst[i * len4 + j4] = src[i * src_axis4 + start4 + j4];
}

// Concat segment: copy one [outer, src_axis] tensor into [outer, dst_axis]
// at the column slice [dst_col .. dst_col + src_axis]. Multi-input concat
// = N dispatches of this kernel, one per source. Mirror of narrow_lastax.
kernel void concat_segment_lastax(
    device const char* arena_src [[buffer(0)]],
    device char* arena_dst       [[buffer(1)]],
    constant uint& outer    [[buffer(2)]],
    constant uint& src_axis [[buffer(3)]],
    constant uint& dst_axis [[buffer(4)]],
    constant uint& dst_col  [[buffer(5)]],
    constant ulong& src_byte_off [[buffer(6)]],
    constant ulong& dst_byte_off [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Task #50: large set_buffer offsets silently lose kernel writes on
    // M-series — bind arena base and apply byte offsets here.
    device const float* src = (device const float*)(arena_src + src_byte_off);
    device float* dst       = (device float*)(arena_dst + dst_byte_off);
    uint i = gid.y;
    uint j = gid.x;
    if (i >= outer || j >= src_axis) return;
    dst[i * dst_axis + dst_col + j] = src[i * src_axis + j];
}

kernel void concat_segment_lastax4(
    device const char* arena_src [[buffer(0)]],
    device char* arena_dst       [[buffer(1)]],
    constant uint& outer            [[buffer(2)]],
    constant uint& src_axis4        [[buffer(3)]],
    constant uint& dst_axis4        [[buffer(4)]],
    constant uint& dst_col4         [[buffer(5)]],
    constant ulong& src_byte_off [[buffer(6)]],
    constant ulong& dst_byte_off [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    device const packed_float4* src = (device const packed_float4*)(arena_src + src_byte_off);
    device packed_float4* dst       = (device packed_float4*)(arena_dst + dst_byte_off);
    uint i = gid.y;
    uint j4 = gid.x;
    if (i >= outer || j4 >= src_axis4) return;
    dst[i * dst_axis4 + dst_col4 + j4] = src[i * src_axis4 + j4];
}

// `src` widened to ulong (task #50: ≥4 GB models have activation byte
// offsets > u32 — truncated wrap-around made `repeat_kv` write the wrong
// slot and SDPA saw all-zero K_rep).
struct ConcatSeg {
    ulong src;
    uint dst_col;
    uint len;
};

kernel void concat_lastax_multi(
    device char* arena       [[buffer(0)]],
    constant ulong& dst_byte [[buffer(1)]],
    constant uint& outer     [[buffer(2)]],
    constant uint& dst_axis  [[buffer(3)]],
    constant uint& num_seg   [[buffer(4)]],
    constant ConcatSeg* segs [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint s = gid.z;
    uint i = gid.y;
    uint j = gid.x;
    if (s >= num_seg) return;
    ConcatSeg seg = segs[s];
    if (i >= outer || j >= seg.len) return;
    device const float* src = (device const float*)(arena + seg.src);
    device float* dst = (device float*)(arena + dst_byte);
    dst[i * dst_axis + seg.dst_col + j] = src[i * seg.len + j];
}

kernel void concat_lastax_multi4(
    device char* arena       [[buffer(0)]],
    constant ulong& dst_byte [[buffer(1)]],
    constant uint& outer     [[buffer(2)]],
    constant uint& dst_axis4 [[buffer(3)]],
    constant uint& num_seg   [[buffer(4)]],
    constant ConcatSeg* segs [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint s = gid.z;
    uint i = gid.y;
    uint j4 = gid.x;
    if (s >= num_seg) return;
    ConcatSeg seg = segs[s];
    uint len4 = seg.len / 4u;
    if (i >= outer || j4 >= len4) return;
    device const packed_float4* src =
        (device const packed_float4*)(arena + seg.src);
    device packed_float4* dst =
        (device packed_float4*)(arena + dst_byte);
    uint dst_col4 = seg.dst_col / 4u;
    dst[i * dst_axis4 + dst_col4 + j4] = src[i * len4 + j4];
}

kernel void concat_segment_lastax_h(
    device const char* arena_src [[buffer(0)]],
    device char* arena_dst       [[buffer(1)]],
    constant uint& outer    [[buffer(2)]],
    constant uint& src_axis [[buffer(3)]],
    constant uint& dst_axis [[buffer(4)]],
    constant uint& dst_col  [[buffer(5)]],
    constant ulong& src_byte_off [[buffer(6)]],
    constant ulong& dst_byte_off [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    device const half* src = (device const half*)(arena_src + src_byte_off);
    device half* dst       = (device half*)(arena_dst + dst_byte_off);
    uint i = gid.y;
    uint j = gid.x;
    if (i >= outer || j >= src_axis) return;
    dst[i * dst_axis + dst_col + j] = src[i * src_axis + j];
}

// Mid-axis concat (inner > 1): copy one segment src[outer][src_axis][inner]
// into dst[outer][dst_axis][inner] starting at axis offset `dst_col`. One 1D
// dispatch per segment, encoded into the live command buffer (NO commit/wait)
// — the host-copy fallback used to commit+wait per concat, serializing a
// decode step into ~100 tiny GPU submissions (the dominant cost on GGUF KV
// caches). `arena` bound at 0 + ulong byte offsets so it is correct on
// >4 GiB arenas (task #50). Generic: subsumes last-axis concat when inner==1.
// Mid-axis concat segment kernels, one per (input, output) precision pair —
// generated from CONCAT_MIDAXIS_SEG_TMPL by `concat_midaxis_variant!`. The
// converting variants (f32→f16 / f16→f32) let a concat with a mismatched-dtype
// input (e.g. an f32 `k_rope` into the f16 KV concat) convert on write instead
// of reading the source bytes at the wrong width (→ inf/NaN saturation).
// @@RLX_CONCAT_MIDAXIS@@

// Fused residual + LN: out = LN(x + residual + bias, gamma, beta)
// (bias is broadcast per row; pass empty/null offset for no-bias variant)
kernel void fused_residual_ln(
    device const float* x      [[buffer(0)]],
    device const float* res    [[buffer(1)]],
    device const float* gamma  [[buffer(2)]],
    device const float* beta   [[buffer(3)]],
    device float* out          [[buffer(4)]],
    constant uint& h           [[buffer(5)]],
    constant float& eps        [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float partial_sum[256];
    threadgroup float partial_sumsq[256];

    // Pass 1: compute (x + res) on the fly, accumulate sum/sumsq
    float local_sum = 0.0;
    float local_sumsq = 0.0;
    for (uint i = tid; i < h; i += tsize) {
        float v = x[row * h + i] + res[row * h + i];
        local_sum += v;
        local_sumsq += v * v;
    }
    partial_sum[tid] = local_sum;
    partial_sumsq[tid] = local_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
            partial_sumsq[tid] += partial_sumsq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = partial_sum[0] / float(h);
    float var = fmax(0.0f, partial_sumsq[0] / float(h) - mean * mean);
    float inv_std = rsqrt(var + eps);

    // Pass 2: write normalized output
    for (uint i = tid; i < h; i += tsize) {
        float v = x[row * h + i] + res[row * h + i];
        out[row * h + i] = (v - mean) * inv_std * gamma[i] + beta[i];
    }
}

// DiT adaLN-Zero: out = norm(x) * (1 + scale) + shift
// scale/shift broadcast over leading dims (typically [B,1,D] over [B,S,D]).
// `layer_norm != 0` → mean-subtract; else RMS only.
// Packed lead dims: [lead_rank, x_lead[8], mod_lead[8]] as 17 uints.
kernel void ada_layer_norm(
    device const float* x      [[buffer(0)]],
    device const float* scale  [[buffer(1)]],
    device const float* shift  [[buffer(2)]],
    device float* out          [[buffer(3)]],
    constant uint& h           [[buffer(4)]],
    constant float& eps        [[buffer(5)]],
    constant uint& layer_norm  [[buffer(6)]],
    constant uint* lead_pack   [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    uint lead_rank = lead_pack[0];
    // Decode `row` into a multi-index over x_lead; fold into mod base offset.
    uint rem = row;
    uint mod_base = 0;
    uint mod_stride = h;
    for (int j = int(lead_rank) - 1; j >= 0; --j) {
        uint xd = lead_pack[1 + j];
        if (xd == 0u) { xd = 1u; }
        uint xi = rem % xd;
        rem /= xd;
        uint md = lead_pack[9 + j];
        if (md == 0u) { md = 1u; }
        if (md != 1u) {
            mod_base += xi * mod_stride;
        }
        mod_stride *= md;
    }

    threadgroup float partial_sum[256];
    threadgroup float partial_sumsq[256];

    float local_sum = 0.0;
    float local_sumsq = 0.0;
    for (uint i = tid; i < h; i += tsize) {
        float v = x[row * h + i];
        local_sum += v;
        local_sumsq += v * v;
    }
    partial_sum[tid] = local_sum;
    partial_sumsq[tid] = local_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
            partial_sumsq[tid] += partial_sumsq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = 0.0;
    float inv;
    if (layer_norm != 0u) {
        mean = partial_sum[0] / float(h);
        float var = fmax(0.0f, partial_sumsq[0] / float(h) - mean * mean);
        inv = rsqrt(var + eps);
    } else {
        float ms = partial_sumsq[0] / float(h);
        inv = rsqrt(ms + eps);
    }

    for (uint i = tid; i < h; i += tsize) {
        float n = (x[row * h + i] - mean) * inv;
        out[row * h + i] = n * (1.0f + scale[mod_base + i]) + shift[mod_base + i];
    }
}

// f16 DiT adaLN — accumulate in f32, store half.
kernel void ada_layer_norm_h(
    device const half* x      [[buffer(0)]],
    device const half* scale  [[buffer(1)]],
    device const half* shift  [[buffer(2)]],
    device half* out          [[buffer(3)]],
    constant uint& h           [[buffer(4)]],
    constant float& eps        [[buffer(5)]],
    constant uint& layer_norm  [[buffer(6)]],
    constant uint* lead_pack   [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    uint lead_rank = lead_pack[0];
    uint rem = row;
    uint mod_base = 0;
    uint mod_stride = h;
    for (int j = int(lead_rank) - 1; j >= 0; --j) {
        uint xd = lead_pack[1 + j];
        if (xd == 0u) { xd = 1u; }
        uint xi = rem % xd;
        rem /= xd;
        uint md = lead_pack[9 + j];
        if (md == 0u) { md = 1u; }
        if (md != 1u) {
            mod_base += xi * mod_stride;
        }
        mod_stride *= md;
    }

    threadgroup float partial_sum[256];
    threadgroup float partial_sumsq[256];

    float local_sum = 0.0;
    float local_sumsq = 0.0;
    for (uint i = tid; i < h; i += tsize) {
        float v = float(x[row * h + i]);
        local_sum += v;
        local_sumsq += v * v;
    }
    partial_sum[tid] = local_sum;
    partial_sumsq[tid] = local_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
            partial_sumsq[tid] += partial_sumsq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = 0.0;
    float inv;
    if (layer_norm != 0u) {
        mean = partial_sum[0] / float(h);
        float var = fmax(0.0f, partial_sumsq[0] / float(h) - mean * mean);
        inv = rsqrt(var + eps);
    } else {
        float ms = partial_sumsq[0] / float(h);
        inv = rsqrt(ms + eps);
    }

    for (uint i = tid; i < h; i += tsize) {
        float n = (float(x[row * h + i]) - mean) * inv;
        out[row * h + i] = half(n * (1.0f + float(scale[mod_base + i])) + float(shift[mod_base + i]));
    }
}

// DiT gated residual: out = x + gate * y  (gate broadcasts like adaLN scale).
// lead_pack: [lead_rank, x_lead[8], gate_lead[8]].
kernel void gated_residual(
    device const float* x      [[buffer(0)]],
    device const float* y      [[buffer(1)]],
    device const float* gate   [[buffer(2)]],
    device float* out          [[buffer(3)]],
    constant uint& h           [[buffer(4)]],
    constant uint* lead_pack   [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint lead_rank = lead_pack[0];
    uint row = gid / h;
    uint col = gid % h;
    uint rem = row;
    uint gate_base = 0;
    uint gate_stride = h;
    for (int j = int(lead_rank) - 1; j >= 0; --j) {
        uint xd = lead_pack[1 + j];
        if (xd == 0u) { xd = 1u; }
        uint xi = rem % xd;
        rem /= xd;
        uint gd = lead_pack[9 + j];
        if (gd == 0u) { gd = 1u; }
        if (gd != 1u) {
            gate_base += xi * gate_stride;
        }
        gate_stride *= gd;
    }
    uint i = row * h + col;
    out[i] = x[i] + gate[gate_base + col] * y[i];
}

kernel void gated_residual_h(
    device const half* x      [[buffer(0)]],
    device const half* y      [[buffer(1)]],
    device const half* gate   [[buffer(2)]],
    device half* out          [[buffer(3)]],
    constant uint& h           [[buffer(4)]],
    constant uint* lead_pack   [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint lead_rank = lead_pack[0];
    uint row = gid / h;
    uint col = gid % h;
    uint rem = row;
    uint gate_base = 0;
    uint gate_stride = h;
    for (int j = int(lead_rank) - 1; j >= 0; --j) {
        uint xd = lead_pack[1 + j];
        if (xd == 0u) { xd = 1u; }
        uint xi = rem % xd;
        rem /= xd;
        uint gd = lead_pack[9 + j];
        if (gd == 0u) { gd = 1u; }
        if (gd != 1u) {
            gate_base += xi * gate_stride;
        }
        gate_stride *= gd;
    }
    uint i = row * h + col;
    out[i] = half(float(x[i]) + float(gate[gate_base + col]) * float(y[i]));
}

// Packed DiT gated residual backward: out = [dx ∥ dy ∥ dgate] (1-D).
// Threadgroups = mod_rows (unique gate rows); each TG owns one gate slice
// and loops `seq_per_mod` x-rows that share it (DiT [B,S,D] / [B,1,D]).
kernel void gated_residual_backward(
    device const float* y       [[buffer(0)]],
    device const float* gate    [[buffer(1)]],
    device const float* dy      [[buffer(2)]],
    device float* packed        [[buffer(3)]],
    constant uint& h            [[buffer(4)]],
    constant uint& seq_per_mod  [[buffer(5)]],
    constant uint& mod_rows     [[buffer(6)]],
    uint m [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    if (m >= mod_rows) return;
    uint nx = mod_rows * seq_per_mod * h;
    uint gate_base = m * h;
    device float* dx = packed;
    device float* dy_out = packed + nx;
    device float* dgate = packed + 2 * nx;

    for (uint i = tid; i < h; i += tsize) {
        float acc = 0.0f;
        for (uint s = 0; s < seq_per_mod; s++) {
            uint row = m * seq_per_mod + s;
            uint idx = row * h + i;
            float g = gate[gate_base + i];
            float d = dy[idx];
            dx[idx] = d;
            dy_out[idx] = d * g;
            acc += d * y[idx];
        }
        dgate[gate_base + i] = acc;
    }
}

kernel void gated_residual_backward_h(
    device const half* y       [[buffer(0)]],
    device const half* gate    [[buffer(1)]],
    device const half* dy      [[buffer(2)]],
    device half* packed        [[buffer(3)]],
    constant uint& h            [[buffer(4)]],
    constant uint& seq_per_mod  [[buffer(5)]],
    constant uint& mod_rows     [[buffer(6)]],
    uint m [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    if (m >= mod_rows) return;
    uint nx = mod_rows * seq_per_mod * h;
    uint gate_base = m * h;
    device half* dx = packed;
    device half* dy_out = packed + nx;
    device half* dgate = packed + 2 * nx;

    for (uint i = tid; i < h; i += tsize) {
        float acc = 0.0f;
        for (uint s = 0; s < seq_per_mod; s++) {
            uint row = m * seq_per_mod + s;
            uint idx = row * h + i;
            float g = float(gate[gate_base + i]);
            float d = float(dy[idx]);
            dx[idx] = half(d);
            dy_out[idx] = half(d * g);
            acc += d * float(y[idx]);
        }
        dgate[gate_base + i] = half(acc);
    }
}

// Packed AdaLayerNorm backward: out = [dx ∥ dscale ∥ dshift] (1-D).
// Same launch geometry as gated_residual_backward.
kernel void ada_layer_norm_backward(
    device const float* x       [[buffer(0)]],
    device const float* scale   [[buffer(1)]],
    device const float* dy      [[buffer(2)]],
    device float* packed        [[buffer(3)]],
    constant uint& h            [[buffer(4)]],
    constant float& eps         [[buffer(5)]],
    constant uint& layer_norm   [[buffer(6)]],
    constant uint& seq_per_mod  [[buffer(7)]],
    constant uint& mod_rows     [[buffer(8)]],
    uint m [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    if (m >= mod_rows) return;
    uint nx = mod_rows * seq_per_mod * h;
    uint mod_len = mod_rows * h;
    uint mod_base = m * h;
    device float* dx = packed;
    device float* dscale = packed + nx;
    device float* dshift = packed + nx + mod_len;

    threadgroup float partial_sum[256];
    threadgroup float partial_sumsq[256];

    // Zero modulation grads for this slice.
    for (uint i = tid; i < h; i += tsize) {
        dscale[mod_base + i] = 0.0f;
        dshift[mod_base + i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_device);

    float inv_h = 1.0f / float(h);
    for (uint s = 0; s < seq_per_mod; s++) {
        uint row = m * seq_per_mod + s;

        float local_sum = 0.0f;
        float local_sumsq = 0.0f;
        for (uint i = tid; i < h; i += tsize) {
            float v = x[row * h + i];
            local_sum += v;
            local_sumsq += v * v;
        }
        partial_sum[tid] = local_sum;
        partial_sumsq[tid] = local_sumsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tsize / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                partial_sum[tid] += partial_sum[tid + stride];
                partial_sumsq[tid] += partial_sumsq[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float mean = 0.0f;
        float inv;
        if (layer_norm != 0u) {
            mean = partial_sum[0] * inv_h;
            float var = fmax(0.0f, partial_sumsq[0] * inv_h - mean * mean);
            inv = rsqrt(var + eps);
        } else {
            inv = rsqrt(partial_sumsq[0] * inv_h + eps);
        }

        // First pass: accumulate dscale/dshift and reduction stats for dx.
        float local_sy = 0.0f;
        float local_sxh = 0.0f;
        for (uint i = tid; i < h; i += tsize) {
            float n = (x[row * h + i] - mean) * inv;
            float d = dy[row * h + i];
            float sc = scale[mod_base + i];
            float sy = d * (1.0f + sc);
            dscale[mod_base + i] += d * n;
            dshift[mod_base + i] += d;
            local_sy += sy;
            local_sxh += sy * n;
        }
        partial_sum[tid] = local_sy;
        partial_sumsq[tid] = local_sxh;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tsize / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                partial_sum[tid] += partial_sum[tid + stride];
                partial_sumsq[tid] += partial_sumsq[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float m_sy = partial_sum[0] * inv_h;
        float m_sxh = partial_sumsq[0] * inv_h;

        for (uint i = tid; i < h; i += tsize) {
            float n = (x[row * h + i] - mean) * inv;
            float d = dy[row * h + i];
            float sc = scale[mod_base + i];
            float sy = d * (1.0f + sc);
            if (layer_norm != 0u) {
                dx[row * h + i] = inv * (sy - m_sy - n * m_sxh);
            } else {
                float n_rms = x[row * h + i] * inv;
                dx[row * h + i] = inv * (sy - n_rms * m_sxh);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void ada_layer_norm_backward_h(
    device const half* x       [[buffer(0)]],
    device const half* scale   [[buffer(1)]],
    device const half* dy      [[buffer(2)]],
    device half* packed        [[buffer(3)]],
    constant uint& h            [[buffer(4)]],
    constant float& eps         [[buffer(5)]],
    constant uint& layer_norm   [[buffer(6)]],
    constant uint& seq_per_mod  [[buffer(7)]],
    constant uint& mod_rows     [[buffer(8)]],
    uint m [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    if (m >= mod_rows) return;
    uint nx = mod_rows * seq_per_mod * h;
    uint mod_len = mod_rows * h;
    uint mod_base = m * h;
    device half* dx = packed;
    device half* dscale = packed + nx;
    device half* dshift = packed + nx + mod_len;

    threadgroup float partial_sum[256];
    threadgroup float partial_sumsq[256];

    for (uint i = tid; i < h; i += tsize) {
        dscale[mod_base + i] = half(0.0h);
        dshift[mod_base + i] = half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_device);

    float inv_h = 1.0f / float(h);
    for (uint s = 0; s < seq_per_mod; s++) {
        uint row = m * seq_per_mod + s;

        float local_sum = 0.0f;
        float local_sumsq = 0.0f;
        for (uint i = tid; i < h; i += tsize) {
            float v = float(x[row * h + i]);
            local_sum += v;
            local_sumsq += v * v;
        }
        partial_sum[tid] = local_sum;
        partial_sumsq[tid] = local_sumsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tsize / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                partial_sum[tid] += partial_sum[tid + stride];
                partial_sumsq[tid] += partial_sumsq[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float mean = 0.0f;
        float inv;
        if (layer_norm != 0u) {
            mean = partial_sum[0] * inv_h;
            float var = fmax(0.0f, partial_sumsq[0] * inv_h - mean * mean);
            inv = rsqrt(var + eps);
        } else {
            inv = rsqrt(partial_sumsq[0] * inv_h + eps);
        }

        float local_sy = 0.0f;
        float local_sxh = 0.0f;
        for (uint i = tid; i < h; i += tsize) {
            float n = (float(x[row * h + i]) - mean) * inv;
            float d = float(dy[row * h + i]);
            float sc = float(scale[mod_base + i]);
            float sy = d * (1.0f + sc);
            dscale[mod_base + i] = half(float(dscale[mod_base + i]) + d * n);
            dshift[mod_base + i] = half(float(dshift[mod_base + i]) + d);
            local_sy += sy;
            local_sxh += sy * n;
        }
        partial_sum[tid] = local_sy;
        partial_sumsq[tid] = local_sxh;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tsize / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                partial_sum[tid] += partial_sum[tid + stride];
                partial_sumsq[tid] += partial_sumsq[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float m_sy = partial_sum[0] * inv_h;
        float m_sxh = partial_sumsq[0] * inv_h;

        for (uint i = tid; i < h; i += tsize) {
            float n = (float(x[row * h + i]) - mean) * inv;
            float d = float(dy[row * h + i]);
            float sc = float(scale[mod_base + i]);
            float sy = d * (1.0f + sc);
            if (layer_norm != 0u) {
                dx[row * h + i] = half(inv * (sy - m_sy - n * m_sxh));
            } else {
                float n_rms = float(x[row * h + i]) * inv;
                dx[row * h + i] = half(inv * (sy - n_rms * m_sxh));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Fused residual + RMSNorm: out = RmsNorm(x + residual, gamma, beta).
// Arena + byte offsets (task #50) — `set_buffer(..., large_offset)` silently
// drops writes on M-series for big Qwen3.5 / Bonsai arenas.
kernel void fused_residual_rms_norm(
    device const char* arena   [[buffer(0)]],
    constant ulong& x_off      [[buffer(1)]],
    constant ulong& res_off    [[buffer(2)]],
    constant ulong& g_off      [[buffer(3)]],
    constant ulong& b_off      [[buffer(4)]],
    constant ulong& out_off    [[buffer(5)]],
    constant uint& h           [[buffer(6)]],
    constant float& eps        [[buffer(7)]],
    constant ulong& sum_off    [[buffer(8)]],   // dual output: pre-norm sum (x+res), 0 = disabled
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    device const float* x = (device const float*)(arena + x_off);
    device const float* res = (device const float*)(arena + res_off);
    device const float* gamma = (device const float*)(arena + g_off);
    device const float* beta = (device const float*)(arena + b_off);
    device float* out = (device float*)(arena + out_off);
    device float* sum = (sum_off != 0) ? (device float*)(arena + sum_off) : (device float*)0;
    threadgroup float partial_sumsq[256];
    float local_sumsq = 0.0;
    for (uint i = tid; i < h; i += tsize) {
        float v = x[row * h + i] + res[row * h + i];
        local_sumsq += v * v;
        if (sum != 0) sum[row * h + i] = v;   // emit the residual sum for the skip stream
    }
    partial_sumsq[tid] = local_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sumsq[tid] += partial_sumsq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(partial_sumsq[0] / float(h) + eps);
    for (uint i = tid; i < h; i += tsize) {
        float v = x[row * h + i] + res[row * h + i];
        out[row * h + i] = v * inv_rms * gamma[i] + beta[i];
    }
}

// Packed SDPA byte offsets (task #50: lets the kernel reach activations
// that sit past 4 GB in the arena without needing `set_buffer(... ,large)`,
// which silently dropped kernel writes on M-series).
struct SdpaOffsets {
    ulong q;
    ulong k;
    ulong v;
    ulong m;
    ulong o;
    // GQA/MQA: K/V head count (equals `heads` for MHA). Packed here so we
    // don't need a 7th set_bytes slot past the Metal arg-table quirk.
    uint kv_heads;
    // V/output per-head width. Equals `head_dim` for symmetric SDPA; smaller
    // for asymmetric MLA (DeepSeek/Kimi qk=192, v=128). Packed into the high
    // 32 bits of the kv_heads slot on the host side. 0 ⇒ fall back to head_dim.
    uint v_head_dim;
};

// Q/K/V offset helpers — BSNH [B, L, H*D] vs BHSD [B, H, L, D].
static inline uint qkv_q_offset(
    uint bi, uint hi, uint qi,
    uint heads, uint seq_q, uint head_dim, uint q_stride, uint bhsd
) {
    if (bhsd != 0u) {
        return bi * heads * seq_q * head_dim + hi * seq_q * head_dim + qi * head_dim;
    }
    uint hs = heads * head_dim;
    return bi * q_stride * hs + qi * hs + hi * head_dim;
}
static inline uint qkv_kv_offset(
    uint bi, uint hi, uint ki,
    uint heads, uint kv_heads, uint seq_k, uint head_dim, uint k_stride, uint bhsd
) {
    // Map query head → shared KV head (GQA/MQA). MHA: kv_heads == heads.
    uint nkv = (kv_heads == 0u) ? heads : kv_heads;
    uint group = heads / nkv;
    uint hi_kv = (group > 1u) ? (hi / group) : hi;
    if (bhsd != 0u) {
        return bi * nkv * seq_k * head_dim + hi_kv * seq_k * head_dim + ki * head_dim;
    }
    uint hs = nkv * head_dim;
    return bi * k_stride * hs + ki * hs + hi_kv * head_dim;
}
// V read offset — like qkv_kv_offset but the per-head width is `v_head_dim`
// (asymmetric MLA). Reduces exactly to qkv_kv_offset when v_head_dim==head_dim.
static inline uint qkv_v_offset(
    uint bi, uint hi, uint ki,
    uint heads, uint kv_heads, uint seq_k, uint v_head_dim, uint k_stride, uint bhsd
) {
    uint nkv = (kv_heads == 0u) ? heads : kv_heads;
    uint group = heads / nkv;
    uint hi_kv = (group > 1u) ? (hi / group) : hi;
    if (bhsd != 0u) {
        return bi * nkv * seq_k * v_head_dim + hi_kv * seq_k * v_head_dim + ki * v_head_dim;
    }
    uint hs = nkv * v_head_dim;
    return bi * k_stride * hs + ki * hs + hi_kv * v_head_dim;
}
// Output write offset — the attention output row is `heads * v_head_dim` wide
// (asymmetric MLA). Reduces exactly to qkv_q_offset when v_head_dim==head_dim.
static inline uint qkv_out_offset(
    uint bi, uint hi, uint qi,
    uint heads, uint seq_q, uint v_head_dim, uint q_stride, uint bhsd
) {
    if (bhsd != 0u) {
        return bi * heads * seq_q * v_head_dim + hi * seq_q * v_head_dim + qi * v_head_dim;
    }
    uint hs = heads * v_head_dim;
    return bi * q_stride * hs + qi * hs + hi * v_head_dim;
}

// Multi-head SDPA: attention(Q, K, V, mask) → out
// Shapes: Q/out [batch, seq_q, heads*head_dim]; K/V [batch, seq_k, heads*head_dim]
// One threadgroup per (batch, head). Each TG computes [seq_q, seq_k] scores
// in threadgroup memory (seq_q * seq_k ≤ 64*64), applies softmax, then
// accumulates scores @ V.
kernel void sdpa(
    device const float* arena_q   [[buffer(0)]],
    device const float* arena_k   [[buffer(1)]],
    device const float* arena_v   [[buffer(2)]],
    device const float* arena_m   [[buffer(3)]],
    device float*       arena_o   [[buffer(4)]],
    constant uint& batch      [[buffer(5)]],
    constant uint& seq_q      [[buffer(6)]],
    constant uint& heads      [[buffer(7)]],
    constant uint& head_dim   [[buffer(8)]],
    constant uint& q_stride   [[buffer(9)]],
    constant uint& mask_kind  [[buffer(10)]],
    constant uint& seq_k      [[buffer(11)]],
    constant uint& k_stride   [[buffer(12)]],
    constant uint& bhsd       [[buffer(13)]],
    constant uint& window     [[buffer(14)]],
    constant float& score_scale  [[buffer(15)]],
    constant float& attn_softcap [[buffer(16)]],
    // Byte offsets relative to the arena buffer, packed as one struct.
    // For ≥4 GB models the activation byte offsets exceed u32 — and Metal
    // silently drops kernel writes when `set_buffer` is called with
    // `offset > 4 GB`. Binding all five buffers to `offset=0` and adding
    // the offsets here is the workaround proven by the dequant kernel
    // (works for offsets ≥ 14 GB). One inline-constant slot for all five
    // (task #50).
    constant SdpaOffsets& byte_offs [[buffer(17)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tsize  [[threads_per_threadgroup]]
) {
    ulong q_byte_off = byte_offs.q;
    ulong k_byte_off = byte_offs.k;
    ulong v_byte_off = byte_offs.v;
    ulong m_byte_off = byte_offs.m;
    ulong o_byte_off = byte_offs.o;
    device const float* Q = (device const float*)((device const char*)arena_q + q_byte_off);
    device const float* K = (device const float*)((device const char*)arena_k + k_byte_off);
    device const float* V = (device const float*)((device const char*)arena_v + v_byte_off);
    device const float* M = (device const float*)((device const char*)arena_m + m_byte_off);
    device float* OUT     = (device float*)((device char*)arena_o + o_byte_off);
    // mask_kind:
    //   0 = None           (no masking)
    //   1 = Causal         (mask ki > (seq_k - seq_q) + qi)
    //   2 = Custom         (column-wise binary mask buffer M; 0 = padded)
    //   4 = SlidingWindow  (visible range [abs_q - window, abs_q],
    //                       absolute positions so decode w/ cached K/V works)
    threadgroup float scores[64 * 64];   // up to seq_q * seq_k = 4096
    threadgroup float row_max;
    threadgroup float row_sum;

    // Linearized: tgid_x = bi * heads + hi
    uint bi = tgid_x / heads;
    uint hi = tgid_x % heads;
    if (bi >= batch) return;

    // `score_scale` is the host-provided multiplier (Gemma 4 sets 1.0
    // because Q is per-head RMS-normed before attention). Sentinel `0.0`
    // means "use the relaxed-precision default `1/sqrt(head_dim)`".
    float scale = (score_scale > 0.0f) ? score_scale : rsqrt(float(head_dim));
    // Gemma 2 carries an attention-logit softcap (50.0); Gemma 4 sets 0.
    float softcap_inv = (attn_softcap > 0.0f) ? (1.0f / attn_softcap) : 0.0f;
    uint q_offset = seq_k - seq_q;


    // 1. Compute scores[qi, ki] = scale * (Q[bi, qi, hi*dh:] · K[bi, ki, hi*dh:]) + mask.
    uint total = seq_q * seq_k;
    for (uint idx = tid; idx < total; idx += tsize) {
        uint qi = idx / seq_k;
        uint ki = idx % seq_k;
        float dot = 0.0;
        uint q_base = qkv_q_offset(bi, hi, qi, heads, seq_q, head_dim, q_stride, bhsd);
        uint k_base = qkv_kv_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, head_dim, k_stride, bhsd);
        for (uint d = 0; d < head_dim; ++d) {
            dot += Q[q_base + d] * K[k_base + d];
        }
        float s = dot * scale;
        if (softcap_inv > 0.0f) {
            s = precise::tanh(s * softcap_inv) * attn_softcap;
        }
        if (mask_kind == 1u) {
            if (ki > q_offset + qi) s = -1e9;
        } else if (mask_kind == 2u) {
            if (M[bi * k_stride + ki] < 0.5) s = -1e9;
        } else if (mask_kind == 4u) {
            uint abs_q = q_offset + qi;
            uint lo = abs_q > window ? abs_q - window : 0u;
            if (ki < lo || ki > abs_q) s = -1e9;
        }
        scores[qi * seq_k + ki] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Softmax row-by-row over scores[seq_q, seq_k]. `precise::exp`
    // matches CPU `f32::exp` to within 1 ULP; the default fast-math
    // `exp` accumulates several ULPs of error per token, which the
    // softcap + LM head amplify into visible logit drift.
    for (uint qi = 0; qi < seq_q; ++qi) {
        if (tid == 0) {
            float mx = -1e30;
            for (uint ki = 0; ki < seq_k; ++ki) {
                mx = max(mx, scores[qi * seq_k + ki]);
            }
            row_max = mx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float sum = 0.0;
            for (uint ki = 0; ki < seq_k; ++ki) {
                float e = precise::exp(scores[qi * seq_k + ki] - row_max);
                scores[qi * seq_k + ki] = e;
                sum += e;
            }
            row_sum = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ki = tid; ki < seq_k; ki += tsize) {
            scores[qi * seq_k + ki] /= row_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3. Output[qi, d] = sum_ki scores[qi, ki] * V[bi, ki, hi*vdh + d]
    // V is read and the output is written `v_head_dim`-wide (asymmetric MLA;
    // == head_dim for symmetric SDPA). Q/K scores above stay head_dim-wide.
    uint vdh = (byte_offs.v_head_dim == 0u) ? head_dim : byte_offs.v_head_dim;
    uint out_total = seq_q * vdh;
    for (uint idx = tid; idx < out_total; idx += tsize) {
        uint qi = idx / vdh;
        uint d = idx % vdh;
        float acc = 0.0;
        for (uint ki = 0; ki < seq_k; ++ki) {
            uint v_base = qkv_v_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, vdh, k_stride, bhsd);
            acc += scores[qi * seq_k + ki] * V[v_base + d];
        }
        uint o_base = qkv_out_offset(bi, hi, qi, heads, seq_q, vdh, q_stride, bhsd);
        OUT[o_base + d] = acc;
    }
}

// SIMD-group-parallel SDPA (seq<=64 prefill). Identical math to `sdpa`, but the
// per-row softmax reduction uses simd_max / simd_sum across the 32-wide
// threadgroup (== one Apple SIMD group) instead of a serial `tid==0` scan. This
// removes the single-thread softmax bottleneck and all per-row threadgroup
// barriers; the score-matmul and A·V steps are already thread-parallel and are
// kept verbatim. Dispatched with exactly 32 threads/threadgroup so the SIMD
// reductions cover the whole threadgroup.
kernel void sdpa_simd(
    device const float* arena_q   [[buffer(0)]],
    device const float* arena_k   [[buffer(1)]],
    device const float* arena_v   [[buffer(2)]],
    device const float* arena_m   [[buffer(3)]],
    device float*       arena_o   [[buffer(4)]],
    constant uint& batch      [[buffer(5)]],
    constant uint& seq_q      [[buffer(6)]],
    constant uint& heads      [[buffer(7)]],
    constant uint& head_dim   [[buffer(8)]],
    constant uint& q_stride   [[buffer(9)]],
    constant uint& mask_kind  [[buffer(10)]],
    constant uint& seq_k      [[buffer(11)]],
    constant uint& k_stride   [[buffer(12)]],
    constant uint& bhsd       [[buffer(13)]],
    constant uint& window     [[buffer(14)]],
    constant float& score_scale  [[buffer(15)]],
    constant float& attn_softcap [[buffer(16)]],
    constant SdpaOffsets& byte_offs [[buffer(17)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tsize  [[threads_per_threadgroup]]
) {
    device const float* Q = (device const float*)((device const char*)arena_q + byte_offs.q);
    device const float* K = (device const float*)((device const char*)arena_k + byte_offs.k);
    device const float* V = (device const float*)((device const char*)arena_v + byte_offs.v);
    device const float* M = (device const float*)((device const char*)arena_m + byte_offs.m);
    device float* OUT     = (device float*)((device char*)arena_o + byte_offs.o);
    threadgroup float scores[64 * 64];

    uint bi = tgid_x / heads;
    uint hi = tgid_x % heads;
    if (bi >= batch) return;

    float scale = (score_scale > 0.0f) ? score_scale : rsqrt(float(head_dim));
    float softcap_inv = (attn_softcap > 0.0f) ? (1.0f / attn_softcap) : 0.0f;
    uint q_offset = seq_k - seq_q;

    // 1. scores[qi,ki] = scale·(Q·K) + mask  (thread-parallel, verbatim from sdpa).
    uint total = seq_q * seq_k;
    for (uint idx = tid; idx < total; idx += tsize) {
        uint qi = idx / seq_k;
        uint ki = idx % seq_k;
        float dot = 0.0;
        uint q_base = qkv_q_offset(bi, hi, qi, heads, seq_q, head_dim, q_stride, bhsd);
        uint k_base = qkv_kv_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, head_dim, k_stride, bhsd);
        for (uint d = 0; d < head_dim; ++d) {
            dot += Q[q_base + d] * K[k_base + d];
        }
        float s = dot * scale;
        if (softcap_inv > 0.0f) {
            s = precise::tanh(s * softcap_inv) * attn_softcap;
        }
        if (mask_kind == 1u) {
            if (ki > q_offset + qi) s = -1e9;
        } else if (mask_kind == 2u) {
            if (M[bi * k_stride + ki] < 0.5) s = -1e9;
        } else if (mask_kind == 4u) {
            uint abs_q = q_offset + qi;
            uint lo = abs_q > window ? abs_q - window : 0u;
            if (ki < lo || ki > abs_q) s = -1e9;
        }
        scores[qi * seq_k + ki] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Softmax per row via SIMD-group reductions. Each thread owns the strided
    // columns ki = tid, tid+32, …; simd_max / simd_sum combine across the 32
    // lanes. Each thread only reads/writes its own columns, so no barrier is
    // needed between the phases (the SIMD ops synchronize the reduction).
    for (uint qi = 0; qi < seq_q; ++qi) {
        float lmax = -1e30;
        for (uint ki = tid; ki < seq_k; ki += tsize) {
            lmax = max(lmax, scores[qi * seq_k + ki]);
        }
        float mx = simd_max(lmax);
        float lsum = 0.0;
        for (uint ki = tid; ki < seq_k; ki += tsize) {
            float e = precise::exp(scores[qi * seq_k + ki] - mx);
            scores[qi * seq_k + ki] = e;
            lsum += e;
        }
        float inv = 1.0f / simd_sum(lsum);
        for (uint ki = tid; ki < seq_k; ki += tsize) {
            scores[qi * seq_k + ki] *= inv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Output[qi,d] = Σ_ki scores[qi,ki]·V[…]  (thread-parallel, verbatim).
    uint vdh = (byte_offs.v_head_dim == 0u) ? head_dim : byte_offs.v_head_dim;
    uint out_total = seq_q * vdh;
    for (uint idx = tid; idx < out_total; idx += tsize) {
        uint qi = idx / vdh;
        uint d = idx % vdh;
        float acc = 0.0;
        for (uint ki = 0; ki < seq_k; ++ki) {
            uint v_base = qkv_v_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, vdh, k_stride, bhsd);
            acc += scores[qi * seq_k + ki] * V[v_base + d];
        }
        uint o_base = qkv_out_offset(bi, hi, qi, heads, seq_q, vdh, q_stride, bhsd);
        OUT[o_base + d] = acc;
    }
}

// f16-scores variant of `sdpa_simd`: the threadgroup scores matrix is stored as
// `half` (8 KiB vs 16 KiB) so 2× as many threadgroups fit per core — the one real
// occupancy limiter the shader trace found. ALL arithmetic stays f32 (Q·K dot,
// softmax max/exp/sum, A·V accumulate); only the on-chip *storage* of scores /
// probabilities is half. The scores are O(1) after scaling and softmax is
// range-robust, so the accuracy cost is ~1 half-ULP in the probabilities.
kernel void sdpa_simd_h16(
    device const float* arena_q   [[buffer(0)]],
    device const float* arena_k   [[buffer(1)]],
    device const float* arena_v   [[buffer(2)]],
    device const float* arena_m   [[buffer(3)]],
    device float*       arena_o   [[buffer(4)]],
    constant uint& batch      [[buffer(5)]],
    constant uint& seq_q      [[buffer(6)]],
    constant uint& heads      [[buffer(7)]],
    constant uint& head_dim   [[buffer(8)]],
    constant uint& q_stride   [[buffer(9)]],
    constant uint& mask_kind  [[buffer(10)]],
    constant uint& seq_k      [[buffer(11)]],
    constant uint& k_stride   [[buffer(12)]],
    constant uint& bhsd       [[buffer(13)]],
    constant uint& window     [[buffer(14)]],
    constant float& score_scale  [[buffer(15)]],
    constant float& attn_softcap [[buffer(16)]],
    constant SdpaOffsets& byte_offs [[buffer(17)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tsize  [[threads_per_threadgroup]]
) {
    device const float* Q = (device const float*)((device const char*)arena_q + byte_offs.q);
    device const float* K = (device const float*)((device const char*)arena_k + byte_offs.k);
    device const float* V = (device const float*)((device const char*)arena_v + byte_offs.v);
    device const float* M = (device const float*)((device const char*)arena_m + byte_offs.m);
    device float* OUT     = (device float*)((device char*)arena_o + byte_offs.o);
    threadgroup half scores[64 * 64];   // 8 KiB (half of the f32 version)

    uint bi = tgid_x / heads;
    uint hi = tgid_x % heads;
    if (bi >= batch) return;

    float scale = (score_scale > 0.0f) ? score_scale : rsqrt(float(head_dim));
    float softcap_inv = (attn_softcap > 0.0f) ? (1.0f / attn_softcap) : 0.0f;
    uint q_offset = seq_k - seq_q;

    uint total = seq_q * seq_k;
    for (uint idx = tid; idx < total; idx += tsize) {
        uint qi = idx / seq_k, ki = idx % seq_k;
        float dot = 0.0;
        uint q_base = qkv_q_offset(bi, hi, qi, heads, seq_q, head_dim, q_stride, bhsd);
        uint k_base = qkv_kv_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, head_dim, k_stride, bhsd);
        // AIR/AGX opt: vectorized float4 loads (4× fewer load ops) + explicit
        // fma (1 fused instr, single-rounding ⇒ *more* accurate than mul+add).
        // float4 needs 16-byte-aligned bases; head_dim/hs are multiples of 4 for
        // real head dims, so the fast path applies. Scalar+fma fallback otherwise.
        if ((head_dim & 3u) == 0u) {
            device const float4* Q4 = (device const float4*)(Q + q_base);
            device const float4* K4 = (device const float4*)(K + k_base);
            for (uint d4 = 0; d4 < (head_dim >> 2); ++d4) {
                float4 qv = Q4[d4], kv = K4[d4];
                dot = fma(qv.x, kv.x, fma(qv.y, kv.y, fma(qv.z, kv.z, fma(qv.w, kv.w, dot))));
            }
        } else {
            for (uint d = 0; d < head_dim; ++d) {
                dot = fma(Q[q_base + d], K[k_base + d], dot);
            }
        }
        float s = dot * scale;
        if (softcap_inv > 0.0f) {
            s = precise::tanh(s * softcap_inv) * attn_softcap;
        }
        if (mask_kind == 1u) {
            if (ki > q_offset + qi) s = -65504.0f;
        } else if (mask_kind == 2u) {
            if (M[bi * k_stride + ki] < 0.5) s = -65504.0f;
        } else if (mask_kind == 4u) {
            uint abs_q = q_offset + qi;
            uint lo = abs_q > window ? abs_q - window : 0u;
            if (ki < lo || ki > abs_q) s = -65504.0f;
        }
        scores[qi * seq_k + ki] = half(s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint qi = 0; qi < seq_q; ++qi) {
        float lmax = -1e30;
        for (uint ki = tid; ki < seq_k; ki += tsize) {
            lmax = max(lmax, float(scores[qi * seq_k + ki]));
        }
        float mx = simd_max(lmax);
        float lsum = 0.0;
        for (uint ki = tid; ki < seq_k; ki += tsize) {
            float e = precise::exp(float(scores[qi * seq_k + ki]) - mx);
            scores[qi * seq_k + ki] = half(e);
            lsum += e;
        }
        float inv = 1.0f / simd_sum(lsum);
        for (uint ki = tid; ki < seq_k; ki += tsize) {
            scores[qi * seq_k + ki] = half(float(scores[qi * seq_k + ki]) * inv);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint vdh = (byte_offs.v_head_dim == 0u) ? head_dim : byte_offs.v_head_dim;
    uint out_total = seq_q * vdh;
    for (uint idx = tid; idx < out_total; idx += tsize) {
        uint qi = idx / vdh, d = idx % vdh;
        float acc = 0.0;
        for (uint ki = 0; ki < seq_k; ++ki) {
            uint v_base = qkv_v_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, vdh, k_stride, bhsd);
            acc += float(scores[qi * seq_k + ki]) * V[v_base + d];
        }
        OUT[qkv_out_offset(bi, hi, qi, heads, seq_q, vdh, q_stride, bhsd) + d] = acc;
    }
}

// Online-softmax SDPA (FlashAttention v1 inner-row form). Same algorithm
// as `wgpu/src/kernels/attention.wgsl` and `cpu/src/thunk.rs` Attention.
// One thread per (batch, head, q_row); each thread walks the K dimension
// exactly once, maintaining a running (m, l, O[D]) tuple — no scores
// matrix in threadgroup memory, so it scales to arbitrary seq length.
//
// The plain `sdpa` kernel above uses `threadgroup float scores[64*64]`;
// for vision (seq=257) that overflows. This kernel handles seq > 64.
//
// Mask layout (vision constant all-ones is `[batch, seq_stride]`):
// reads M[bi * seq_stride + ki] just like `sdpa`.
//
// MAX_HEAD_DIM = 128 covers BERT/Nomic/Vision (head_dim ≤ 128); larger
// head dims would need a per-thread spill buffer.
kernel void sdpa_long(
    device const float* arena_q   [[buffer(0)]],
    device const float* arena_k   [[buffer(1)]],
    device const float* arena_v   [[buffer(2)]],
    device const float* arena_m   [[buffer(3)]],
    device float*       arena_o   [[buffer(4)]],
    constant uint& batch       [[buffer(5)]],
    constant uint& seq_q       [[buffer(6)]],   // query length Lq
    constant uint& heads       [[buffer(7)]],
    constant uint& head_dim    [[buffer(8)]],
    constant uint& q_stride    [[buffer(9)]],   // per-batch Q row stride (= Lq for dense)
    constant uint& mask_kind   [[buffer(10)]],
    constant uint& seq_k       [[buffer(11)]],  // key/value length Lk
    constant uint& k_stride    [[buffer(12)]],  // per-batch K/V row stride (= Lk for dense)
    constant uint& bhsd        [[buffer(13)]],  // 1 = [B,H,S,D]
    constant uint& window      [[buffer(14)]],  // SlidingWindow lookback (0 otherwise)
    constant float& score_scale  [[buffer(15)]],
    constant float& attn_softcap [[buffer(16)]],
    // Task #50: > 4 GB activations need offsets in inline constants.
    constant SdpaOffsets& byte_offs [[buffer(17)]],
    uint tid_x [[thread_position_in_grid]]
) {
    device const float* Q = (device const float*)((device const char*)arena_q + byte_offs.q);
    device const float* K = (device const float*)((device const char*)arena_k + byte_offs.k);
    device const float* V = (device const float*)((device const char*)arena_v + byte_offs.v);
    device const float* M = (device const float*)((device const char*)arena_m + byte_offs.m);
    device float* OUT     = (device float*)((device char*)arena_o + byte_offs.o);
    // mask_kind:
    //   0 = None
    //   1 = Causal           (prefill — Lq == Lk required)
    //   2 = Custom            (binary key-padding mask M[B, Lk])
    //   3 = Bias              (additive per-head bias M[B, H, Lq, Lk])
    //   4 = SlidingWindow     (visible range [abs_q - window, abs_q])
    //
    // Gemma 4 12B's SWA layers use head_dim=256 and the FULL layers use
    // head_dim=512. The previous 128 cap silently overflowed q_reg/o_acc
    // and produced all-NaN logits when decode picked this kernel.
    constexpr uint MAX_HEAD_DIM = 512u;
    uint total = batch * heads * seq_q;
    if (tid_x >= total) return;

    uint qi = tid_x % seq_q;
    uint bh = tid_x / seq_q;
    uint hi = bh % heads;
    uint bi = bh / heads;

    // Precise scale; honour caller's `score_scale` (Gemma 4 = 1.0).
    float scale = (score_scale > 0.0f) ? score_scale : rsqrt(float(head_dim));
    float softcap_inv = (attn_softcap > 0.0f) ? (1.0f / attn_softcap) : 0.0f;

    // V/output per-head width (asymmetric MLA; == head_dim for symmetric).
    uint vdh = (byte_offs.v_head_dim == 0u) ? head_dim : byte_offs.v_head_dim;

    // Cache Q[qi, hi*dh : (hi+1)*dh] in registers — read seq_k times below.
    float q_reg[MAX_HEAD_DIM];
    uint q_base = qkv_q_offset(bi, hi, qi, heads, seq_q, head_dim, q_stride, bhsd);
    for (uint d = 0; d < head_dim; ++d) q_reg[d] = Q[q_base + d];

    // Bias base offset (only read when mask_kind == 3).
    uint bias_row_base = ((bi * heads + hi) * seq_q + qi) * seq_k;
    uint q_offset = seq_k - seq_q;

    // Online softmax accumulators. O is v_head_dim-wide.
    float m_acc = -1e30;
    float l_acc = 0.0;
    float o_acc[MAX_HEAD_DIM];
    for (uint d = 0; d < vdh; ++d) o_acc[d] = 0.0;

    for (uint ki = 0; ki < seq_k; ++ki) {
        // Causal early-exit: keys ki > q_offset+qi are fully masked and contribute
        // EXACTLY zero to the online softmax (m unchanged, e_cur = exp(-1e9-m) = 0,
        // l/o += 0). Since ki increases monotonically, once past the diagonal every
        // remaining key is masked — stop. Bit-identical; ~2× fewer iterations for
        // causal prefill. Only fires for mask_kind==1 (non-causal graphs — vision
        // seq=257, BERT, cross-attn — keep the full loop).
        if (mask_kind == 1u && ki > q_offset + qi) break;
        // Score: scale * (Q · K[ki]) + mask
        uint k_base = qkv_kv_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, head_dim, k_stride, bhsd);
        float dot = 0.0;
        for (uint d = 0; d < head_dim; ++d) dot += q_reg[d] * K[k_base + d];
        float s = dot * scale;
        if (softcap_inv > 0.0f) {
            s = precise::tanh(s * softcap_inv) * attn_softcap;
        }
        if (mask_kind == 1u) {
            if (ki > q_offset + qi) s = -1e9;
        } else if (mask_kind == 2u) {
            if (M[bi * k_stride + ki] < 0.5) s = -1e9;
        } else if (mask_kind == 3u) {
            s += M[bias_row_base + ki];
        } else if (mask_kind == 4u) {
            uint abs_q = q_offset + qi;
            uint lo = abs_q > window ? abs_q - window : 0u;
            if (ki < lo || ki > abs_q) s = -1e9;
        }

        // Online softmax update with precise exp.
        float m_new = max(m_acc, s);
        float e_old = precise::exp(m_acc - m_new);
        float e_cur = precise::exp(s - m_new);
        l_acc = e_old * l_acc + e_cur;
        uint v_base = qkv_v_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, vdh, k_stride, bhsd);
        for (uint d = 0; d < vdh; ++d) {
            o_acc[d] = e_old * o_acc[d] + e_cur * V[v_base + d];
        }
        m_acc = m_new;
    }

    // Normalize and emit (v_head_dim-wide output).
    float inv_l = 1.0 / l_acc;
    uint o_base = qkv_out_offset(bi, hi, qi, heads, seq_q, vdh, q_stride, bhsd);
    for (uint d = 0; d < vdh; ++d) {
        OUT[o_base + d] = o_acc[d] * inv_l;
    }
}

// ── Occupancy-isolation probe (benchmark only) ──────────────────────────────
// Byte-for-byte identical WORK to `sdpa_long` (same memory traffic, same
// instruction stream) but declares a 20 KB dummy threadgroup array — matching
// sdpa_fa2/attn_bwd_fused's tgMem footprint — so the compiler's
// staticThreadgroupMemoryLength (hence concurrent-threadgroups-per-core /
// occupancy) drops to their level while everything else is held constant. If
// this runs ~2-3× slower than `sdpa_long`, threadgroup-memory-limited occupancy
// is proven the cause of the flash-kernel slowdowns, independent of cache/roofline.
kernel void sdpa_long_occpad(
    device const float* arena_q   [[buffer(0)]],
    device const float* arena_k   [[buffer(1)]],
    device const float* arena_v   [[buffer(2)]],
    device const float* arena_m   [[buffer(3)]],
    device float*       arena_o   [[buffer(4)]],
    constant uint& batch       [[buffer(5)]],
    constant uint& seq_q       [[buffer(6)]],
    constant uint& heads       [[buffer(7)]],
    constant uint& head_dim    [[buffer(8)]],
    constant uint& q_stride    [[buffer(9)]],
    constant uint& mask_kind   [[buffer(10)]],
    constant uint& seq_k       [[buffer(11)]],
    constant uint& k_stride    [[buffer(12)]],
    constant uint& bhsd        [[buffer(13)]],
    constant uint& window      [[buffer(14)]],
    constant float& score_scale  [[buffer(15)]],
    constant float& attn_softcap [[buffer(16)]],
    constant SdpaOffsets& byte_offs [[buffer(17)]],
    uint tid_x [[thread_position_in_grid]],
    uint tlid  [[thread_index_in_threadgroup]]
) {
    // 20 KB of threadgroup memory — declared (so it counts against occupancy)
    // and lightly touched (so the compiler can't eliminate it), but it plays no
    // role in the math below.
    threadgroup float occ_pad[5120];
    device const float* Q = (device const float*)((device const char*)arena_q + byte_offs.q);
    device const float* K = (device const float*)((device const char*)arena_k + byte_offs.k);
    device const float* V = (device const float*)((device const char*)arena_v + byte_offs.v);
    device const float* M = (device const float*)((device const char*)arena_m + byte_offs.m);
    device float* OUT     = (device float*)((device char*)arena_o + byte_offs.o);
    constexpr uint MAX_HEAD_DIM = 512u;
    uint total = batch * heads * seq_q;
    if (tid_x >= total) return;
    uint qi = tid_x % seq_q;
    uint bh = tid_x / seq_q;
    uint hi = bh % heads;
    uint bi = bh / heads;
    float scale = (score_scale > 0.0f) ? score_scale : rsqrt(float(head_dim));
    float softcap_inv = (attn_softcap > 0.0f) ? (1.0f / attn_softcap) : 0.0f;
    uint vdh = (byte_offs.v_head_dim == 0u) ? head_dim : byte_offs.v_head_dim;
    occ_pad[tlid] = scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_reg[MAX_HEAD_DIM];
    uint q_base = qkv_q_offset(bi, hi, qi, heads, seq_q, head_dim, q_stride, bhsd);
    for (uint d = 0; d < head_dim; ++d) q_reg[d] = Q[q_base + d];
    uint bias_row_base = ((bi * heads + hi) * seq_q + qi) * seq_k;
    uint q_offset = seq_k - seq_q;
    float m_acc = -1e30;
    float l_acc = 0.0;
    float o_acc[MAX_HEAD_DIM];
    for (uint d = 0; d < vdh; ++d) o_acc[d] = 0.0;
    for (uint ki = 0; ki < seq_k; ++ki) {
        if (mask_kind == 1u && ki > q_offset + qi) break;
        uint k_base = qkv_kv_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, head_dim, k_stride, bhsd);
        float dot = 0.0;
        for (uint d = 0; d < head_dim; ++d) dot += q_reg[d] * K[k_base + d];
        float s = dot * scale;
        if (softcap_inv > 0.0f) s = precise::tanh(s * softcap_inv) * attn_softcap;
        if (mask_kind == 1u) { if (ki > q_offset + qi) s = -1e9; }
        else if (mask_kind == 2u) { if (M[bi * k_stride + ki] < 0.5) s = -1e9; }
        else if (mask_kind == 3u) { s += M[bias_row_base + ki]; }
        else if (mask_kind == 4u) { uint abs_q = q_offset + qi; uint lo = abs_q > window ? abs_q - window : 0u; if (ki < lo || ki > abs_q) s = -1e9; }
        float m_new = max(m_acc, s);
        float e_old = precise::exp(m_acc - m_new);
        float e_cur = precise::exp(s - m_new);
        l_acc = e_old * l_acc + e_cur;
        uint v_base = qkv_v_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, vdh, k_stride, bhsd);
        for (uint d = 0; d < vdh; ++d) o_acc[d] = e_old * o_acc[d] + e_cur * V[v_base + d];
        m_acc = m_new;
    }
    float inv_l = 1.0 / l_acc;
    uint o_base = qkv_out_offset(bi, hi, qi, heads, seq_q, vdh, q_stride, bhsd);
    // The dummy tgMem feeds a never-true branch so it can't be optimized away.
    float pad_guard = occ_pad[tlid];
    for (uint d = 0; d < vdh; ++d) {
        float o = o_acc[d] * inv_l;
        if (pad_guard > 1e38f) o += occ_pad[(tlid + d) % 5120];
        OUT[o_base + d] = o;
    }
}

// Decode-step SDPA fast path (Lq == 1, Lk arbitrary).
//
// One threadgroup per (batch, head); threads split the K axis and merge
// online-softmax states. The old 1-thread-per-head launch left Apple GPUs
// nearly idle on GQA decode (Zonos: batch=2 × 16 heads = 32 threads).
// @@RLX_SDPA_DECODE_M1@@

// Flash-attention tile kernel with optional additive bias mask.
//
// Targets the SAM3 detector decoder image cross-attention where the
// scalar `sdpa_long` is bandwidth-bound (each query thread re-reads K
// and V for all 5184 positions). This kernel processes Br=8 query
// rows per threadgroup with K, V, and bias tiles loaded cooperatively
// into threadgroup memory — each K/V/bias element is read once per
// row tile instead of once per query.
//
// Layout matches `sdpa_long`: Q/K/V are [B, Lq_or_Lk, heads*head_dim],
// bias is [B, H, Lq, Lk]. head_dim is dynamic but capped at 128 for
// the per-thread output accumulator.
kernel void sdpa_fa_f32(
    device const float* Q   [[buffer(0)]],
    device const float* K   [[buffer(1)]],
    device const float* V   [[buffer(2)]],
    device const float* M   [[buffer(3)]],
    device float* OUT       [[buffer(4)]],
    constant uint& batch       [[buffer(5)]],
    constant uint& seq_q       [[buffer(6)]],
    constant uint& heads       [[buffer(7)]],
    constant uint& head_dim    [[buffer(8)]],
    constant uint& q_stride    [[buffer(9)]],
    constant uint& mask_kind   [[buffer(10)]],
    constant uint& seq_k       [[buffer(11)]],
    constant uint& k_stride    [[buffer(12)]],
    constant uint& bhsd        [[buffer(13)]],
    constant uint& window      [[buffer(14)]],  // reserved; not yet wired in FA tile path
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]]
) {
    (void)window;  // SlidingWindow falls through to sdpa_long today
    // Tile sizes — tuned for SAM3 image CA (dh=16) but kernel is
    // generic. With Br=8, Bc=64, the per-TG threadgroup memory is
    // 8*128 (Q) + 64*128 (K) + 64*128 (V) + 8*64 (S/M) ≈ 71KB at
    // dh=128; well under the 32–64KB per-TG hard limit at dh=16
    // (where it's ~10KB).
    // Tile sizes — the threadgroup-memory cap on Apple7/8 (32KB) and
    // Apple9 (64KB) bounds `MAX_DH`. At MAX_DH=32 we use ~20KB,
    // leaving headroom for larger Bc later. dh up to 32 covers SAM
    // family models (dh=16) and DETR-style detectors. Larger dh
    // (LLM 64–128) falls back to scalar sdpa_long via the dispatch
    // guard in `encode_sdpa`.
    constexpr uint Br = 8u;
    constexpr uint Bc = 64u;
    constexpr uint MAX_DH = 32u;
    constexpr uint THREADS = 64u;

    threadgroup float Q_tg[Br * MAX_DH];     // 1 KB
    threadgroup float K_tg[Bc * MAX_DH];     // 8 KB
    threadgroup float V_tg[Bc * MAX_DH];     // 8 KB
    threadgroup float S_tg[Br * Bc];         // 2 KB

    // Per-row online softmax state.
    threadgroup float m_row[Br];
    threadgroup float l_row[Br];
    threadgroup float o_row[Br * MAX_DH];    // 1 KB

    uint q_tile = tgid.x;          // index over Lq / Br
    uint hi     = tgid.y;          // head
    uint bi     = tgid.z;          // batch
    uint q_start = q_tile * Br;

    float scale = rsqrt(float(head_dim));

    // ── Load Q tile cooperatively ────────────────────────────────────
    for (uint i = tid_in_tg; i < Br * head_dim; i += THREADS) {
        uint qi = i / head_dim;
        uint di = i % head_dim;
        uint pos = q_start + qi;
        Q_tg[qi * MAX_DH + di] = (pos < seq_q)
            ? Q[qkv_q_offset(bi, hi, pos, heads, seq_q, head_dim, q_stride, bhsd) + di]
            : 0.0f;
    }

    // Initialize per-row state.
    if (tid_in_tg < Br) {
        m_row[tid_in_tg] = -1e30f;
        l_row[tid_in_tg] = 0.0f;
    }
    for (uint i = tid_in_tg; i < Br * head_dim; i += THREADS) {
        o_row[(i / head_dim) * MAX_DH + (i % head_dim)] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Iterate K/V tiles ─────────────────────────────────────────────
    uint bias_row_base = (bi * heads + hi) * seq_q * seq_k;

    for (uint kt = 0; kt < seq_k; kt += Bc) {
        // Load K and V tiles (Bc * head_dim elements each).
        for (uint i = tid_in_tg; i < Bc * head_dim; i += THREADS) {
            uint ki = i / head_dim;
            uint di = i % head_dim;
            uint pos = kt + ki;
            uint kv_off = qkv_kv_offset(bi, hi, pos, heads, heads, seq_k, head_dim, k_stride, bhsd);
            bool in_range = pos < seq_k;
            K_tg[ki * MAX_DH + di] = in_range ? K[kv_off + di] : 0.0f;
            V_tg[ki * MAX_DH + di] = in_range ? V[kv_off + di] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute scores S[Br, Bc] = Q_tg @ K_tg^T, scaled, +bias, +pad-mask.
        // Each thread covers Br*Bc/THREADS = 8*64/64 = 8 cells.
        for (uint c = tid_in_tg; c < Br * Bc; c += THREADS) {
            uint qi = c / Bc;
            uint ki = c % Bc;
            uint pos = kt + ki;
            bool valid = (q_start + qi) < seq_q && pos < seq_k;
            float s = 0.0f;
            if (valid) {
                for (uint di = 0; di < head_dim; ++di) {
                    s += Q_tg[qi * MAX_DH + di] * K_tg[ki * MAX_DH + di];
                }
                s *= scale;
                if (mask_kind == 1u) {
                    uint q_offset = seq_k - seq_q;
                    if (pos > q_offset + q_start + qi) s = -1e9f;
                } else if (mask_kind == 2u) {
                    if (M[bi * k_stride + pos] < 0.5f) s = -1e9f;
                } else if (mask_kind == 3u) {
                    s += M[bias_row_base + (q_start + qi) * seq_k + pos];
                }
            } else {
                s = -1e9f;
            }
            S_tg[c] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax update — one thread per row (Br threads).
        if (tid_in_tg < Br) {
            uint qi = tid_in_tg;
            float m_new = m_row[qi];
            for (uint ki = 0; ki < Bc; ++ki) {
                m_new = max(m_new, S_tg[qi * Bc + ki]);
            }
            float e_old = exp(m_row[qi] - m_new);
            float l_new = e_old * l_row[qi];
            for (uint ki = 0; ki < Bc; ++ki) {
                float p = exp(S_tg[qi * Bc + ki] - m_new);
                S_tg[qi * Bc + ki] = p;
                l_new += p;
            }
            // O ← e_old * O + P @ V
            for (uint di = 0; di < head_dim; ++di) {
                float o = o_row[qi * MAX_DH + di] * e_old;
                for (uint ki = 0; ki < Bc; ++ki) {
                    o += S_tg[qi * Bc + ki] * V_tg[ki * MAX_DH + di];
                }
                o_row[qi * MAX_DH + di] = o;
            }
            m_row[qi] = m_new;
            l_row[qi] = l_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Normalize + emit ─────────────────────────────────────────────
    for (uint i = tid_in_tg; i < Br * head_dim; i += THREADS) {
        uint qi = i / head_dim;
        uint di = i % head_dim;
        uint pos = q_start + qi;
        if (pos < seq_q) {
            float o = o_row[qi * MAX_DH + di] / l_row[qi];
            OUT[qkv_q_offset(bi, hi, pos, heads, seq_q, head_dim, q_stride, bhsd) + di] = o;
        }
    }
}

// ── Variant 1: split-K SIMD prefill SDPA ────────────────────────────────────
// One SIMD group (32 threads) per (batch, head, query-row). The 32 lanes split
// the key axis (ki = lane, lane+32, …), each maintaining a partial online-softmax
// state (m,l,O) over its strided keys, then merge across lanes via simd_max /
// simd_sum. This parallelizes `sdpa_long`'s serial K walk 32-way while keeping
// the same O(D)-register / no-scores-matrix footprint (scales to any seq).
// head_dim, v_head_dim ≤ 128 (per-lane register accumulators). MUST be launched
// with exactly 32 threads/threadgroup (== one Apple SIMD group).
kernel void sdpa_splitk(
    device const float* arena_q   [[buffer(0)]],
    device const float* arena_k   [[buffer(1)]],
    device const float* arena_v   [[buffer(2)]],
    device const float* arena_m   [[buffer(3)]],
    device float*       arena_o   [[buffer(4)]],
    constant uint& batch       [[buffer(5)]],
    constant uint& seq_q       [[buffer(6)]],
    constant uint& heads       [[buffer(7)]],
    constant uint& head_dim    [[buffer(8)]],
    constant uint& q_stride    [[buffer(9)]],
    constant uint& mask_kind   [[buffer(10)]],
    constant uint& seq_k       [[buffer(11)]],
    constant uint& k_stride    [[buffer(12)]],
    constant uint& bhsd        [[buffer(13)]],
    constant uint& window      [[buffer(14)]],
    constant float& score_scale  [[buffer(15)]],
    constant float& attn_softcap [[buffer(16)]],
    constant SdpaOffsets& byte_offs [[buffer(17)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint lane   [[thread_index_in_threadgroup]]
) {
    device const float* Q = (device const float*)((device const char*)arena_q + byte_offs.q);
    device const float* K = (device const float*)((device const char*)arena_k + byte_offs.k);
    device const float* V = (device const float*)((device const char*)arena_v + byte_offs.v);
    device const float* M = (device const float*)((device const char*)arena_m + byte_offs.m);
    device float* OUT     = (device float*)((device char*)arena_o + byte_offs.o);

    constexpr uint MAX_DH = 128u;
    constexpr uint LANES = 32u;

    uint total_rows = batch * heads * seq_q;
    if (tgid_x >= total_rows) return;
    uint qi = tgid_x % seq_q;
    uint bh = tgid_x / seq_q;
    uint hi = bh % heads;
    uint bi = bh / heads;

    float scale = (score_scale > 0.0f) ? score_scale : rsqrt(float(head_dim));
    float softcap_inv = (attn_softcap > 0.0f) ? (1.0f / attn_softcap) : 0.0f;
    uint vdh = (byte_offs.v_head_dim == 0u) ? head_dim : byte_offs.v_head_dim;
    uint q_offset = seq_k - seq_q;

    // Q row into per-lane registers (small; head_dim ≤ 128).
    float q_reg[MAX_DH];
    uint q_base = qkv_q_offset(bi, hi, qi, heads, seq_q, head_dim, q_stride, bhsd);
    for (uint d = 0; d < head_dim; ++d) q_reg[d] = Q[q_base + d];

    // Per-lane online softmax over strided keys.
    float m_i = -1e30f;
    float l_i = 0.0f;
    float o_i[MAX_DH];
    for (uint d = 0; d < vdh; ++d) o_i[d] = 0.0f;
    uint bias_row_base = ((bi * heads + hi) * seq_q + qi) * seq_k;

    for (uint ki = lane; ki < seq_k; ki += LANES) {
        uint k_base = qkv_kv_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, head_dim, k_stride, bhsd);
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) dot = fma(q_reg[d], K[k_base + d], dot);
        float s = dot * scale;
        if (softcap_inv > 0.0f) s = precise::tanh(s * softcap_inv) * attn_softcap;
        bool masked = false;
        if (mask_kind == 1u) {
            if (ki > q_offset + qi) masked = true;
        } else if (mask_kind == 2u) {
            if (M[bi * k_stride + ki] < 0.5f) masked = true;
        } else if (mask_kind == 3u) {
            s += M[bias_row_base + ki];
        } else if (mask_kind == 4u) {
            uint abs_q = q_offset + qi;
            uint lo = abs_q > window ? abs_q - window : 0u;
            if (ki < lo || ki > abs_q) masked = true;
        }
        if (masked) continue;
        float m_new = max(m_i, s);
        float e_old = precise::exp(m_i - m_new);
        float e_cur = precise::exp(s - m_new);
        l_i = e_old * l_i + e_cur;
        uint v_base = qkv_v_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, vdh, k_stride, bhsd);
        for (uint d = 0; d < vdh; ++d) o_i[d] = e_old * o_i[d] + e_cur * V[v_base + d];
        m_i = m_new;
    }

    // Merge the 32 partial states across the SIMD group. Empty lanes keep
    // m_i = -1e30 ⇒ scale_i = exp(-inf) = 0, so they contribute nothing.
    float m_g = simd_max(m_i);
    float scale_i = precise::exp(m_i - m_g);
    float l_g = simd_sum(l_i * scale_i);
    float inv_l = (l_g > 0.0f) ? (1.0f / l_g) : 0.0f;
    uint o_base = qkv_out_offset(bi, hi, qi, heads, seq_q, vdh, q_stride, bhsd);
    for (uint d = 0; d < vdh; ++d) {
        float o_d = simd_sum(o_i[d] * scale_i);
        if (lane == 0u) OUT[o_base + d] = o_d * inv_l;
    }
}

// ── Variant 2: improved flash-attention tile (full thread utilization) ───────
// Rewrite of `sdpa_fa_f32`: Br=8 query rows per threadgroup with K/V loaded
// cooperatively into threadgroup memory (each K/V element read once per row-tile,
// not once per query), BUT — unlike sdpa_fa_f32, whose softmax + O update ran on
// only Br of the 64 threads — the score matmul AND the O=P·V update both run
// across all 64 threads. Only the tiny per-row softmax-stat reduction is Br-wide.
// Uses proper SdpaOffsets byte offsets (sdpa_fa_f32 ignored them). head_dim,
// v_head_dim ≤ 64 (MAX_DH); larger falls back to sdpa_long via the dispatch gate.
kernel void sdpa_fa2(
    device const float* arena_q   [[buffer(0)]],
    device const float* arena_k   [[buffer(1)]],
    device const float* arena_v   [[buffer(2)]],
    device const float* arena_m   [[buffer(3)]],
    device float*       arena_o   [[buffer(4)]],
    constant uint& batch       [[buffer(5)]],
    constant uint& seq_q       [[buffer(6)]],
    constant uint& heads       [[buffer(7)]],
    constant uint& head_dim    [[buffer(8)]],
    constant uint& q_stride    [[buffer(9)]],
    constant uint& mask_kind   [[buffer(10)]],
    constant uint& seq_k       [[buffer(11)]],
    constant uint& k_stride    [[buffer(12)]],
    constant uint& bhsd        [[buffer(13)]],
    constant uint& window      [[buffer(14)]],
    constant float& score_scale  [[buffer(15)]],
    constant float& attn_softcap [[buffer(16)]],
    constant SdpaOffsets& byte_offs [[buffer(17)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]]
) {
    device const float* Q = (device const float*)((device const char*)arena_q + byte_offs.q);
    device const float* K = (device const float*)((device const char*)arena_k + byte_offs.k);
    device const float* V = (device const float*)((device const char*)arena_v + byte_offs.v);
    device const float* M = (device const float*)((device const char*)arena_m + byte_offs.m);
    device float* OUT     = (device float*)((device char*)arena_o + byte_offs.o);

    constexpr uint Br = 8u;
    constexpr uint Bc = 32u;
    constexpr uint MAX_DH = 64u;
    constexpr uint THREADS = 64u;

    threadgroup float Q_tg[Br * MAX_DH];
    threadgroup float K_tg[Bc * MAX_DH];
    threadgroup float V_tg[Bc * MAX_DH];
    threadgroup float S_tg[Br * Bc];
    threadgroup float O_tg[Br * MAX_DH];
    threadgroup float m_row[Br];
    threadgroup float l_row[Br];
    threadgroup float eold_row[Br];

    uint q_tile = tgid.x;
    uint hi     = tgid.y;
    uint bi     = tgid.z;
    uint q_start = q_tile * Br;

    float scale = (score_scale > 0.0f) ? score_scale : rsqrt(float(head_dim));
    float softcap_inv = (attn_softcap > 0.0f) ? (1.0f / attn_softcap) : 0.0f;
    uint vdh = (byte_offs.v_head_dim == 0u) ? head_dim : byte_offs.v_head_dim;
    uint q_offset = seq_k - seq_q;

    // Load Q tile cooperatively.
    for (uint i = tid; i < Br * head_dim; i += THREADS) {
        uint qi = i / head_dim, di = i % head_dim;
        uint pos = q_start + qi;
        Q_tg[qi * MAX_DH + di] = (pos < seq_q)
            ? Q[qkv_q_offset(bi, hi, pos, heads, seq_q, head_dim, q_stride, bhsd) + di]
            : 0.0f;
    }
    if (tid < Br) { m_row[tid] = -1e30f; l_row[tid] = 0.0f; }
    for (uint i = tid; i < Br * MAX_DH; i += THREADS) O_tg[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kt = 0; kt < seq_k; kt += Bc) {
        // Load K and V tiles cooperatively.
        for (uint i = tid; i < Bc * head_dim; i += THREADS) {
            uint ki = i / head_dim, di = i % head_dim;
            uint pos = kt + ki;
            K_tg[ki * MAX_DH + di] = (pos < seq_k)
                ? K[qkv_kv_offset(bi, hi, pos, heads, byte_offs.kv_heads, seq_k, head_dim, k_stride, bhsd) + di]
                : 0.0f;
        }
        for (uint i = tid; i < Bc * vdh; i += THREADS) {
            uint ki = i / vdh, di = i % vdh;
            uint pos = kt + ki;
            V_tg[ki * MAX_DH + di] = (pos < seq_k)
                ? V[qkv_v_offset(bi, hi, pos, heads, byte_offs.kv_heads, seq_k, vdh, k_stride, bhsd) + di]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scores S[Br,Bc] = scale·(Q·Kᵀ) + mask — all 64 threads (Br·Bc/64 cells each).
        for (uint c = tid; c < Br * Bc; c += THREADS) {
            uint qi = c / Bc, ki = c % Bc;
            uint pos = kt + ki;
            bool valid = (q_start + qi) < seq_q && pos < seq_k;
            float s = -1e30f;
            if (valid) {
                float acc = 0.0f;
                for (uint di = 0; di < head_dim; ++di)
                    acc = fma(Q_tg[qi * MAX_DH + di], K_tg[ki * MAX_DH + di], acc);
                s = acc * scale;
                if (softcap_inv > 0.0f) s = precise::tanh(s * softcap_inv) * attn_softcap;
                uint abs_q = q_offset + q_start + qi;
                if (mask_kind == 1u) {
                    if (pos > abs_q) s = -1e30f;
                } else if (mask_kind == 2u) {
                    if (M[bi * k_stride + pos] < 0.5f) s = -1e30f;
                } else if (mask_kind == 3u) {
                    s += M[((bi * heads + hi) * seq_q + (q_start + qi)) * seq_k + pos];
                } else if (mask_kind == 4u) {
                    uint lo = abs_q > window ? abs_q - window : 0u;
                    if (pos < lo || pos > abs_q) s = -1e30f;
                }
            }
            S_tg[c] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-row online-softmax stats (Br threads): update m,l; rewrite S←P.
        if (tid < Br) {
            uint qi = tid;
            float m_old = m_row[qi];
            float m_new = m_old;
            for (uint ki = 0; ki < Bc; ++ki) m_new = max(m_new, S_tg[qi * Bc + ki]);
            float e_old = precise::exp(m_old - m_new);
            float l_new = e_old * l_row[qi];
            for (uint ki = 0; ki < Bc; ++ki) {
                float p = precise::exp(S_tg[qi * Bc + ki] - m_new);
                S_tg[qi * Bc + ki] = p;
                l_new += p;
            }
            m_row[qi] = m_new;
            l_row[qi] = l_new;
            eold_row[qi] = e_old;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O ← e_old·O + P·V — all 64 threads over (qi,di) = Br·vdh work items.
        for (uint c = tid; c < Br * vdh; c += THREADS) {
            uint qi = c / vdh, di = c % vdh;
            float o = O_tg[qi * MAX_DH + di] * eold_row[qi];
            for (uint ki = 0; ki < Bc; ++ki)
                o = fma(S_tg[qi * Bc + ki], V_tg[ki * MAX_DH + di], o);
            O_tg[qi * MAX_DH + di] = o;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize + emit.
    for (uint c = tid; c < Br * vdh; c += THREADS) {
        uint qi = c / vdh, di = c % vdh;
        uint pos = q_start + qi;
        if (pos < seq_q) {
            float inv = (l_row[qi] > 0.0f) ? (1.0f / l_row[qi]) : 0.0f;
            OUT[qkv_out_offset(bi, hi, pos, heads, seq_q, vdh, q_stride, bhsd) + di]
                = O_tg[qi * MAX_DH + di] * inv;
        }
    }
}

// ── Variant 3: simdgroup-matrix (float8x8 tensor-unit) flash attention ───────
// One SIMD group (32 threads) per (Br=8 query rows, head, batch). QK^T and P·V
// run on Apple's simdgroup_float8x8 matrix units (8×8×8 MMAs) instead of scalar
// dot loops; the online-softmax rescale of the O accumulator is kept in scalar
// threadgroup memory (simdgroup registers can't be row-scaled). Small Bc=8 blocks
// keep threadgroup memory ~5 KB (vs sdpa_fa2's ~21 KB) so occupancy stays high.
// Requires head_dim % 8 == 0 and head_dim, v_head_dim ≤ 64 (dispatch-gated).
kernel void sdpa_mma(
    device const float* arena_q   [[buffer(0)]],
    device const float* arena_k   [[buffer(1)]],
    device const float* arena_v   [[buffer(2)]],
    device const float* arena_m   [[buffer(3)]],
    device float*       arena_o   [[buffer(4)]],
    constant uint& batch       [[buffer(5)]],
    constant uint& seq_q       [[buffer(6)]],
    constant uint& heads       [[buffer(7)]],
    constant uint& head_dim    [[buffer(8)]],
    constant uint& q_stride    [[buffer(9)]],
    constant uint& mask_kind   [[buffer(10)]],
    constant uint& seq_k       [[buffer(11)]],
    constant uint& k_stride    [[buffer(12)]],
    constant uint& bhsd        [[buffer(13)]],
    constant uint& window      [[buffer(14)]],
    constant float& score_scale  [[buffer(15)]],
    constant float& attn_softcap [[buffer(16)]],
    constant SdpaOffsets& byte_offs [[buffer(17)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]]
) {
    device const float* Q = (device const float*)((device const char*)arena_q + byte_offs.q);
    device const float* K = (device const float*)((device const char*)arena_k + byte_offs.k);
    device const float* V = (device const float*)((device const char*)arena_v + byte_offs.v);
    device const float* M = (device const float*)((device const char*)arena_m + byte_offs.m);
    device float* OUT     = (device float*)((device char*)arena_o + byte_offs.o);

    constexpr uint Br = 8u;
    constexpr uint Bc = 8u;
    constexpr uint MAX_DH = 64u;
    constexpr uint THREADS = 32u;

    threadgroup float Q_tg[Br * MAX_DH];
    threadgroup float K_tg[Bc * MAX_DH];
    threadgroup float V_tg[Bc * MAX_DH];
    threadgroup float S_tg[Br * Bc];
    threadgroup float PV_tg[Br * MAX_DH];
    threadgroup float O_tg[Br * MAX_DH];
    threadgroup float m_row[Br];
    threadgroup float l_row[Br];
    threadgroup float eold_row[Br];

    uint q_tile = tgid.x, hi = tgid.y, bi = tgid.z;
    uint q_start = q_tile * Br;
    float scale = (score_scale > 0.0f) ? score_scale : rsqrt(float(head_dim));
    float softcap_inv = (attn_softcap > 0.0f) ? (1.0f / attn_softcap) : 0.0f;
    uint vdh = (byte_offs.v_head_dim == 0u) ? head_dim : byte_offs.v_head_dim;
    uint q_offset = seq_k - seq_q;

    // Load Q tile (Br × head_dim), zero-pad rows past seq_q.
    for (uint i = tid; i < Br * head_dim; i += THREADS) {
        uint qi = i / head_dim, di = i % head_dim;
        uint pos = q_start + qi;
        Q_tg[qi * MAX_DH + di] = (pos < seq_q)
            ? Q[qkv_q_offset(bi, hi, pos, heads, seq_q, head_dim, q_stride, bhsd) + di]
            : 0.0f;
    }
    if (tid < Br) { m_row[tid] = -1e30f; l_row[tid] = 0.0f; }
    for (uint i = tid; i < Br * MAX_DH; i += THREADS) O_tg[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint dh_chunks = head_dim / 8u;   // exact (dispatch-gated to head_dim%8==0)
    uint vd_chunks = vdh / 8u;

    for (uint kt = 0; kt < seq_k; kt += Bc) {
        // Load K, V blocks (Bc × head_dim / vdh), zero-pad past seq_k.
        for (uint i = tid; i < Bc * head_dim; i += THREADS) {
            uint ki = i / head_dim, di = i % head_dim;
            uint pos = kt + ki;
            K_tg[ki * MAX_DH + di] = (pos < seq_k)
                ? K[qkv_kv_offset(bi, hi, pos, heads, byte_offs.kv_heads, seq_k, head_dim, k_stride, bhsd) + di]
                : 0.0f;
        }
        for (uint i = tid; i < Bc * vdh; i += THREADS) {
            uint ki = i / vdh, di = i % vdh;
            uint pos = kt + ki;
            V_tg[ki * MAX_DH + di] = (pos < seq_k)
                ? V[qkv_v_offset(bi, hi, pos, heads, byte_offs.kv_heads, seq_k, vdh, k_stride, bhsd) + di]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S[Br,Bc] = Q @ Kᵀ via simdgroup MMA (accumulate over head_dim chunks).
        simdgroup_float8x8 sacc = simdgroup_float8x8(0.0f);
        for (uint c = 0; c < dh_chunks; ++c) {
            simdgroup_float8x8 qmat, kmat;
            simdgroup_load(qmat, &Q_tg[c * 8u], MAX_DH);
            simdgroup_load(kmat, &K_tg[c * 8u], MAX_DH, ulong2(0, 0), /*transpose=*/true);
            simdgroup_multiply_accumulate(sacc, qmat, kmat, sacc);
        }
        simdgroup_store(sacc, &S_tg[0], Bc);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scale + mask + online-softmax stats (Br threads); rewrite S ← P.
        if (tid < Br) {
            uint qi = tid;
            uint abs_q = q_offset + q_start + qi;
            bool row_valid = (q_start + qi) < seq_q;
            float m_old = m_row[qi];
            float m_new = m_old;
            for (uint ki = 0; ki < Bc; ++ki) {
                uint pos = kt + ki;
                float s = S_tg[qi * Bc + ki] * scale;
                if (softcap_inv > 0.0f) s = precise::tanh(s * softcap_inv) * attn_softcap;
                bool ok = row_valid && pos < seq_k;
                if (ok) {
                    if (mask_kind == 1u) { if (pos > abs_q) ok = false; }
                    else if (mask_kind == 2u) { if (M[bi * k_stride + pos] < 0.5f) ok = false; }
                    else if (mask_kind == 3u) { s += M[((bi * heads + hi) * seq_q + (q_start + qi)) * seq_k + pos]; }
                    else if (mask_kind == 4u) { uint lo = abs_q > window ? abs_q - window : 0u; if (pos < lo || pos > abs_q) ok = false; }
                }
                if (!ok) s = -1e30f;
                S_tg[qi * Bc + ki] = s;
                m_new = max(m_new, s);
            }
            float e_old = precise::exp(m_old - m_new);
            float l_new = e_old * l_row[qi];
            for (uint ki = 0; ki < Bc; ++ki) {
                float p = precise::exp(S_tg[qi * Bc + ki] - m_new);
                S_tg[qi * Bc + ki] = p;
                l_new += p;
            }
            m_row[qi] = m_new;
            l_row[qi] = l_new;
            eold_row[qi] = e_old;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // PV[Br,vdh] = P @ V via simdgroup MMA (P is 8×8; V in 8-col chunks).
        for (uint dc = 0; dc < vd_chunks; ++dc) {
            simdgroup_float8x8 pmat, vmat;
            simdgroup_load(pmat, &S_tg[0], Bc);
            simdgroup_load(vmat, &V_tg[dc * 8u], MAX_DH);
            simdgroup_float8x8 pvacc = simdgroup_float8x8(0.0f);
            simdgroup_multiply_accumulate(pvacc, pmat, vmat, pvacc);
            simdgroup_store(pvacc, &PV_tg[dc * 8u], MAX_DH);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O ← e_old·O + PV  (scalar rescale, all 32 threads).
        for (uint i = tid; i < Br * vdh; i += THREADS) {
            uint qi = i / vdh, di = i % vdh;
            O_tg[qi * MAX_DH + di] = eold_row[qi] * O_tg[qi * MAX_DH + di] + PV_tg[qi * MAX_DH + di];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize + emit.
    for (uint i = tid; i < Br * vdh; i += THREADS) {
        uint qi = i / vdh, di = i % vdh;
        uint pos = q_start + qi;
        if (pos < seq_q) {
            float inv = (l_row[qi] > 0.0f) ? (1.0f / l_row[qi]) : 0.0f;
            OUT[qkv_out_offset(bi, hi, pos, heads, seq_q, vdh, q_stride, bhsd) + di]
                = O_tg[qi * MAX_DH + di] * inv;
        }
    }
}

// RoPE: apply rotary position embeddings to one tensor (Q or K).
// x: [batch, seq, hidden], hidden = num_heads * head_dim
// cos/sin: [max_pos, head_dim/2]
// Out-of-place into out (or in-place via aliasing).
kernel void rope(
    device const float* x   [[buffer(0)]],
    device const float* cos [[buffer(1)]],
    device const float* sin [[buffer(2)]],
    device float* out       [[buffer(3)]],
    constant uint& batch          [[buffer(4)]],
    constant uint& seq            [[buffer(5)]],
    constant uint& hidden         [[buffer(6)]],
    constant uint& head_dim       [[buffer(7)]],
    constant uint& src_row_stride [[buffer(8)]],
    constant uint& seq_stride     [[buffer(9)]],
    constant uint& n_rot          [[buffer(10)]],
    constant uint& cos_per_token  [[buffer(11)]],
    constant uint& interleaved    [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // gid.x = dim index within head (0..head_dim)
    // gid.y = head index
    // gid.z = batch * seq + seq pos (linearized)
    uint half_dh = head_dim / 2;
    uint rot_half = n_rot / 2;
    if (gid.x >= head_dim) return;

    uint bs = gid.z;
    uint bi = bs / seq;
    uint si = bs % seq;
    if (bi >= batch || si >= seq) return;

    uint nh = hidden / head_dim;
    uint hi = gid.y;
    if (hi >= nh) return;

    // RoPE table row: per-seq-position by default; per global (batch·seq)
    // token for ragged batched decode, where each sequence sits at its own
    // absolute position.
    uint cos_row = (cos_per_token != 0u) ? bs : si;

    // PLAN L1 — `seq_stride` is the compile-time full extent for buffer
    // offsets; `seq` is the (possibly scaled) iteration bound. This
    // separation lets active-extent dispatch shrink the loop without
    // corrupting per-batch strides.
    uint src_base = bi * seq_stride * src_row_stride + si * src_row_stride + hi * head_dim;
    uint dst_base = bi * seq_stride * hidden + si * hidden + hi * head_dim;
    uint d = gid.x;
    if (interleaved != 0u) {
        // GPT-J / llama.cpp-NORM: rotated pairs are adjacent (2d, 2d+1);
        // cos/sin indexed by freq d. GGUF Llama weights need this flavor.
        if (d < rot_half) {
            uint a = 2u * d;
            uint b = 2u * d + 1u;
            float x1 = x[src_base + a];
            float x2 = x[src_base + b];
            float c = cos[cos_row * half_dh + d];
            float s = sin[cos_row * half_dh + d];
            out[dst_base + a] = x1 * c - x2 * s;
            out[dst_base + b] = x2 * c + x1 * s;
        } else if (d >= n_rot) {
            out[dst_base + d] = x[src_base + d];
        }
    } else if (d < rot_half) {
        float x1 = x[src_base + d];
        float x2 = x[src_base + rot_half + d];
        float c = cos[cos_row * half_dh + d];
        float s = sin[cos_row * half_dh + d];
        out[dst_base + d] = x1 * c - x2 * s;
        out[dst_base + rot_half + d] = x2 * c + x1 * s;
    } else if (d >= n_rot) {
        out[dst_base + d] = x[src_base + d];
    }
}

// ArgMax / ArgMin along the middle axis of [outer, reduced, inner], emitting
// the winning index (f32). One thread per (outer, inner) output element. Strict
// comparison with first-best tie-break — matches rlx-cpu execute_argreduce_f32.
kernel void argreduce(
    device const float* src [[buffer(0)]],
    device float* out       [[buffer(1)]],
    constant uint& outer    [[buffer(2)]],
    constant uint& reduced  [[buffer(3)]],
    constant uint& inner    [[buffer(4)]],
    constant uint& is_max   [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = outer * inner;
    if (gid >= total) return;
    uint o = gid / inner;
    uint i = gid % inner;
    uint base = o * reduced * inner + i;
    float best = src[base];
    uint best_idx = 0u;
    for (uint r = 1u; r < reduced; ++r) {
        float v = src[base + r * inner];
        bool better = (is_max != 0u) ? (v > best) : (v < best);
        if (better) { best = v; best_idx = r; }
    }
    out[o * inner + i] = float(best_idx);
}

// Cooperative last-axis ArgMax/ArgMin: one threadgroup reduces one `outer`
// row over the `reduced` axis (inner == 1 — the decode logits case, where the
// naive one-thread-per-output `argreduce` would loop a 128k-vocab row on a
// single GPU lane). Tie-break = lowest index wins, matching the strict `>`/`<`
// in rlx-cpu execute_argreduce_f32. Threadgroup size must be a power of two.
kernel void argreduce_lastaxis(
    device const float* src [[buffer(0)]],
    device float* out       [[buffer(1)]],
    constant uint& outer    [[buffer(2)]],
    constant uint& reduced  [[buffer(3)]],
    constant uint& is_max   [[buffer(4)]],
    uint tg       [[threadgroup_position_in_grid]],
    uint tid      [[thread_position_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]]
) {
    if (tg >= outer) return;
    device const float* row = src + (ulong)tg * (ulong)reduced;

    threadgroup float sval[256];
    threadgroup uint  sidx[256];

    float best = (is_max != 0u) ? -INFINITY : INFINITY;
    uint  bidx = 0u;
    // Strided scan: within a lane, strict comparison keeps the lowest index.
    for (uint r = tid; r < reduced; r += nthreads) {
        float v = row[r];
        bool better = (is_max != 0u) ? (v > best) : (v < best);
        if (better) { best = v; bidx = r; }
    }
    sval[tid] = best;
    sidx[tid] = bidx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = nthreads >> 1; s > 0u; s >>= 1) {
        if (tid < s) {
            float v  = sval[tid + s];
            uint  i  = sidx[tid + s];
            float cv = sval[tid];
            uint  ci = sidx[tid];
            bool better = (is_max != 0u) ? (v > cv) : (v < cv);
            // Equal value → keep the lower source index (CPU first-best).
            if (better || (v == cv && i < ci)) { sval[tid] = v; sidx[tid] = i; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) out[tg] = float(sidx[0]);
}

// Fused vector-quantization: one threadgroup per input row cooperatively finds
// the nearest codebook entry over the reduced K axis — no [N,K] materialization,
// no D2H copy (the host-callback custom-op path was ~100× slower on Metal).
// metric 0 = L2 (argmin ‖x−C‖²), 1 = cosine (argmax x·C/‖C‖). Output is the
// f32-encoded code index (ready for gather). Lowest index wins ties.
kernel void vq_assign(
    device const float* x   [[buffer(0)]],   // [N, D]
    device const float* cb  [[buffer(1)]],   // [K, D]
    device float* out       [[buffer(2)]],   // [N]
    constant uint& n        [[buffer(3)]],
    constant uint& d        [[buffer(4)]],
    constant uint& k        [[buffer(5)]],
    constant uint& metric   [[buffer(6)]],
    uint tg       [[threadgroup_position_in_grid]],
    uint tid      [[thread_position_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]]
) {
    if (tg >= n) return;
    device const float* xi = x + (ulong)tg * (ulong)d;
    threadgroup float sval[256];
    threadgroup uint  sidx[256];

    // float4 fast path when D is a multiple of 4 (the common case); the tail
    // loop handles any remainder. Vectorizing the dot closes most of the gap to
    // MPS's matrix units.
    const uint d4 = d >> 2;
    const uint drem = d & 3u;
    device const float4* xi4 = (device const float4*)xi;

    float best = (metric == 0u) ? INFINITY : -INFINITY;
    uint  bidx = 0u;
    for (uint j = tid; j < k; j += nthreads) {
        device const float* cj = cb + (ulong)j * (ulong)d;
        device const float4* cj4 = (device const float4*)cj;
        if (metric == 0u) {
            float4 acc = 0.0f;
            for (uint t = 0u; t < d4; ++t) { float4 df = xi4[t] - cj4[t]; acc += df * df; }
            float dist = acc.x + acc.y + acc.z + acc.w;
            for (uint t = d - drem; t < d; ++t) { float df = xi[t] - cj[t]; dist += df * df; }
            if (dist < best) { best = dist; bidx = j; }
        } else {
            float4 dacc = 0.0f, nacc = 0.0f;
            for (uint t = 0u; t < d4; ++t) { dacc += xi4[t] * cj4[t]; nacc += cj4[t] * cj4[t]; }
            float dot = dacc.x + dacc.y + dacc.z + dacc.w;
            float nc = nacc.x + nacc.y + nacc.z + nacc.w;
            for (uint t = d - drem; t < d; ++t) { dot += xi[t] * cj[t]; nc += cj[t] * cj[t]; }
            float sim = (nc > 0.0f) ? dot * rsqrt(nc) : 0.0f;
            if (sim > best) { best = sim; bidx = j; }
        }
    }
    sval[tid] = best;
    sidx[tid] = bidx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthreads >> 1; s > 0u; s >>= 1) {
        if (tid < s) {
            bool take = (metric == 0u) ? (sval[tid + s] < sval[tid]) : (sval[tid + s] > sval[tid]);
            if (take || (sval[tid + s] == sval[tid] && sidx[tid + s] < sidx[tid])) {
                sval[tid] = sval[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) out[tg] = float(sidx[0]);
}

// On-GPU logit sampling: temperature -> top-k -> softmax -> top-p -> Philox
// inverse-CDF draw. One threadgroup per batch row. Mirrors the CPU algorithm
// in rlx-cpu `sample_row` / `execute_sample_f32` exactly, including the
// Philox4x32 stream (rlx_ir::rng), so a fixed seed is bit-comparable.
//
// top-k / top-p cutoffs are found by parallel bisection on the value/prob
// range rather than a full sort: for distinct logits this selects the
// identical kept set (the cutoff lands strictly between the two order
// statistics that bracket it). The final inverse-CDF walk (thread 0) recomputes
// each token's filtered probability in original index order so the sequential
// float accumulation matches the CPU reference element-for-element.
//
// Threadgroup size must be a power of two (dispatched at 256).
kernel void sample_logits(
    device float* arena         [[buffer(0)]],
    constant ulong& logits_off  [[buffer(1)]],
    constant ulong& dst_off     [[buffer(2)]],
    constant uint& batch        [[buffer(3)]],
    constant uint& vocab        [[buffer(4)]],
    constant uint& top_k        [[buffer(5)]],
    constant float& top_p       [[buffer(6)]],
    constant float& temperature [[buffer(7)]],
    constant ulong& seed        [[buffer(8)]],
    uint tg       [[threadgroup_position_in_grid]],
    uint tid      [[thread_position_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]]
) {
    if (tg >= batch) return;
    device const float* logits =
        (device const float*)((device char*)arena + logits_off);
    device float* dst = (device float*)((device char*)arena + dst_off);
    device const float* row = logits + (ulong)tg * (ulong)vocab;

    const uint v = vocab;
    const float MIN_POS = 1.1754944e-38f;          // f32::MIN_POSITIVE
    const float temp = max(temperature, 1e-6f);
    const uint  kk   = min(top_k, v);
    const bool use_topk = (kk > 0u) && (kk < v);
    const bool use_topp = (top_p < 1.0f);

    if (v == 0u) { if (tid == 0u) dst[tg] = 0.0f; return; }

    // `red` is the reduction scratch; `bounds[0..1]` carries the bisection
    // lo/hi (an array, so per-element-init warnings don't fire). All cross-lane
    // values are read back from `red[0]` after the reduction barrier.
    threadgroup float red[256];
    threadgroup float bounds[2];

    // ── max(scaled) ────────────────────────────────────────────────
    float lmax = -INFINITY;
    for (uint i = tid; i < v; i += nthreads) lmax = max(lmax, row[i]);
    red[tid] = lmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthreads >> 1; s > 0u; s >>= 1) {
        if (tid < s) red[tid] = max(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_l = red[0] / temp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── min(scaled) (bisection lower bound) ────────────────────────
    float lmin = INFINITY;
    for (uint i = tid; i < v; i += nthreads) lmin = min(lmin, row[i]);
    red[tid] = lmin;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthreads >> 1; s > 0u; s >>= 1) {
        if (tid < s) red[tid] = min(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float min_l = red[0] / temp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── top-k cutoff = kk-th largest scaled value (bisection) ──────
    float cutoff = -INFINITY;
    if (use_topk) {
        if (tid == 0u) { bounds[0] = min_l; bounds[1] = max_l; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint it = 0u; it < 50u; ++it) {
            float mid = 0.5f * (bounds[0] + bounds[1]);
            float cnt = 0.0f;
            for (uint i = tid; i < v; i += nthreads)
                if (row[i] / temp >= mid) cnt += 1.0f;
            red[tid] = cnt;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = nthreads >> 1; s > 0u; s >>= 1) {
                if (tid < s) red[tid] += red[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0u) {
                if (red[0] >= float(kk)) bounds[0] = mid; else bounds[1] = mid;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        cutoff = bounds[0];
    }

    // ── softmax denom over the top-k set ───────────────────────────
    float s1 = 0.0f;
    for (uint i = tid; i < v; i += nthreads) {
        float sc = row[i] / temp;
        if (!use_topk || sc >= cutoff) s1 += exp(sc - max_l);
    }
    red[tid] = s1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthreads >> 1; s > 0u; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float sum1 = red[0];
    const float inv1 = 1.0f / max(sum1, MIN_POS);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── top-p prob cutoff (bisection over [0,1]) ───────────────────
    float pcut = 0.0f;
    if (use_topp) {
        if (tid == 0u) { bounds[0] = 0.0f; bounds[1] = 1.0f; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint it = 0u; it < 60u; ++it) {
            float mid = 0.5f * (bounds[0] + bounds[1]);
            float psum = 0.0f;
            for (uint i = tid; i < v; i += nthreads) {
                float sc = row[i] / temp;
                if (use_topk && sc < cutoff) continue;
                float p = exp(sc - max_l) * inv1;
                if (p >= mid) psum += p;
            }
            red[tid] = psum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = nthreads >> 1; s > 0u; s >>= 1) {
                if (tid < s) red[tid] += red[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0u) {
                if (red[0] >= top_p) bounds[0] = mid; else bounds[1] = mid;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        pcut = bounds[0];
    }

    // ── renorm denom over the top-p set ────────────────────────────
    float sum2 = 1.0f;
    if (use_topp) {
        float s2 = 0.0f;
        for (uint i = tid; i < v; i += nthreads) {
            float sc = row[i] / temp;
            if (use_topk && sc < cutoff) continue;
            float p = exp(sc - max_l) * inv1;
            if (p >= pcut) s2 += p;
        }
        red[tid] = s2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = nthreads >> 1; s > 0u; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        sum2 = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── thread 0: Philox draw + sequential inverse-CDF in index order ─
    if (tid == 0u) {
        ulong sd  = (seed == 0ul) ? 0xDEADBEEFul : seed;
        uint  k0  = (uint)(sd & 0xFFFFFFFFul);
        uint  k1  = (uint)(sd >> 32);
        uint  st0 = (uint)(tg / 4u), st1 = 0u, st2 = 0u, st3 = 0u;
        for (uint rr = 0u; rr < 10u; ++rr) {
            ulong p0 = (ulong)st0 * 0xD2561A75ul;
            ulong p1 = (ulong)st2 * 0xCD9E8D57ul;
            uint hi0 = (uint)(p0 >> 32), lo0 = (uint)(p0 & 0xFFFFFFFFul);
            uint hi1 = (uint)(p1 >> 32), lo1 = (uint)(p1 & 0xFFFFFFFFul);
            uint n0 = hi1 ^ st1 ^ k0;
            uint n1 = lo1;
            uint n2 = hi0 ^ st3 ^ k1;
            uint n3 = lo0;
            st0 = n0; st1 = n1; st2 = n2; st3 = n3;
            k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
        }
        uint lane = tg % 4u;
        uint bits = (lane == 0u ? st0 : (lane == 1u ? st1 : (lane == 2u ? st2 : st3))) >> 8;
        float r = (float)bits / 16777216.0f;

        float inv2 = use_topp ? (1.0f / max(sum2, MIN_POS)) : 1.0f;
        float acc = 0.0f;
        uint chosen = v - 1u;
        for (uint i = 0u; i < v; ++i) {
            float sc = row[i] / temp;
            float p;
            if (use_topk && sc < cutoff) {
                p = 0.0f;
            } else {
                p = exp(sc - max_l) * inv1;
                if (use_topp) { p = (p >= pcut) ? (p * inv2) : 0.0f; }
            }
            acc += p;
            if (r <= acc) { chosen = i; break; }
        }
        dst[tg] = float(chosen);
    }
}

// Block-quantized int8 weight matmul: out[m,n] = x[m,k] @ dequant(wq[k,n]).
// Per-(block-of-k, n) scale (+ optional zero-point). One thread per output
// element. Matches rlx-cpu dequant_matmul_int8.
kernel void dequant_matmul_int8(
    device const float* x      [[buffer(0)]],
    device const char*  wq     [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* zps    [[buffer(3)]],
    device float* out          [[buffer(4)]],
    constant uint& m           [[buffer(5)]],
    constant uint& k           [[buffer(6)]],
    constant uint& n           [[buffer(7)]],
    constant uint& block_size  [[buffer(8)]],
    constant uint& asym        [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= m * n) return;
    uint i = gid / n;
    uint j = gid % n;
    // Accumulate one block at a time and fold in that block's scale ONCE, not
    // per element (see rlx_cpu dequant_matmul_int8). Reassociation only —
    // fewer multiplies, tighter running-sum dynamic range, bit-consistent
    // with the CPU oracle.
    uint nblk = (k + block_size - 1u) / block_size;
    float acc = 0.0f;
    for (uint b = 0u; b < nblk; ++b) {
        float s = scales[b * n + j];
        float z = (asym != 0u) ? zps[b * n + j] : 0.0f;
        uint lo = b * block_size;
        uint hi = min(lo + block_size, k);
        float bacc = 0.0f;
        for (uint p = lo; p < hi; ++p) {
            float q = float(wq[p * n + j]);
            bacc += x[i * k + p] * (q - z);
        }
        acc += bacc * s;
    }
    out[i * n + j] = acc;
}

// Block-quantized int4 (two nibbles per byte) weight matmul. Low nibble first.
kernel void dequant_matmul_int4(
    device const float* x      [[buffer(0)]],
    device const uchar* wq     [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* zps    [[buffer(3)]],
    device float* out          [[buffer(4)]],
    constant uint& m           [[buffer(5)]],
    constant uint& k           [[buffer(6)]],
    constant uint& n           [[buffer(7)]],
    constant uint& block_size  [[buffer(8)]],
    constant uint& asym        [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= m * n) return;
    uint i = gid / n;
    uint j = gid % n;
    // Block-at-a-time accumulation, scale folded once per block (see int8).
    uint nblk = (k + block_size - 1u) / block_size;
    float acc = 0.0f;
    for (uint b = 0u; b < nblk; ++b) {
        float s = scales[b * n + j];
        float z = (asym != 0u) ? zps[b * n + j] : 0.0f;
        uint lo = b * block_size;
        uint hi = min(lo + block_size, k);
        float bacc = 0.0f;
        for (uint p = lo; p < hi; ++p) {
            uint idx = p * n + j;
            uchar byte = wq[idx >> 1];
            uint nib = ((idx & 1u) == 0u) ? (uint(byte) & 0x0Fu) : (uint(byte) >> 4);
            bacc += x[i * k + p] * (float(nib) - z);
        }
        acc += bacc * s;
    }
    out[i * n + j] = acc;
}

inline float dq_fp8_e4m3(uchar byte) {
    // Match rlx-mlx-io `dequant_scale_fp8_e4m3` (bit-exact OCP E4M3).
    uint sign = (uint(byte) >> 7) & 1u;
    int exp_v = int((uint(byte) >> 3) & 0x0Fu);
    uint mant = uint(byte) & 0x7u;
    if (exp_v == 0x0f && mant == 0x7u) {
        return NAN; // Match rlx-mlx-io host decode
    }
    if (exp_v == 0) {
        if (mant == 0u) {
            return (sign != 0u) ? -0.0f : 0.0f;
        }
        uint m = mant;
        int e = -6;
        while ((m & 0x8u) == 0u) {
            m <<= 1;
            e -= 1;
        }
        m &= 0x7u;
        uint bits = (sign << 31) | (uint(e + 127) << 23) | (m << 20);
        return as_type<float>(bits);
    }
    uint bits = (sign << 31) | (uint(exp_v - 7 + 127) << 23) | (mant << 20);
    return as_type<float>(bits);
}

inline float dq_fp8_e5m2(uchar byte) {
    uint sign = (uint(byte) >> 7) & 1u;
    uint exp_v = (uint(byte) >> 2) & 0x1Fu;
    uint mant = uint(byte) & 0x3u;
    float v;
    if (exp_v == 0u) {
        v = (mant == 0u) ? 0.0f : (float(mant) / 4.0f) * exp2(-14.0f);
    } else if (exp_v == 0x1Fu) {
        v = 0.0f;
    } else {
        v = (1.0f + float(mant) / 4.0f) * exp2(float(int(exp_v) - 15));
    }
    return (sign != 0u) ? -v : v;
}

constant float DQ_FP4_E2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

kernel void dequant_matmul_fp8(
    device const float* x      [[buffer(0)]],
    device const uchar* wq     [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device float* out          [[buffer(4)]],
    constant uint& m           [[buffer(5)]],
    constant uint& k           [[buffer(6)]],
    constant uint& n           [[buffer(7)]],
    constant uint& e5m2        [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= m * n) return;
    uint i = gid / n;
    uint j = gid % n;
    // The per-column scale is loop-invariant: apply it once to the finished dot
    // instead of every term. Reassociation only.
    float col_scale = scales[j];
    float acc = 0.0f;
    for (uint p = 0u; p < k; ++p) {
        uchar byte = wq[p * n + j];
        float w = (e5m2 != 0u) ? dq_fp8_e5m2(byte) : dq_fp8_e4m3(byte);
        acc += x[i * k + p] * w;
    }
    out[i * n + j] = acc * col_scale;
}

kernel void dequant_matmul_nvfp4(
    device const float* x      [[buffer(0)]],
    device const uchar* wq     [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device const float* gs_ptr [[buffer(3)]],
    device float* out          [[buffer(4)]],
    constant uint& m           [[buffer(5)]],
    constant uint& k           [[buffer(6)]],
    constant uint& n           [[buffer(7)]],
    constant uint& group_size  [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= m * n) return;
    uint i = gid / n;
    uint j = gid % n;
    // Fold the per-group E4M3 scale in once per group and the single global
    // scale in once at the end (see rlx_cpu dequant_matmul_nvfp4).
    float gs = gs_ptr[0];
    uint nblk = (k + group_size - 1u) / group_size;
    float acc = 0.0f;
    for (uint b = 0u; b < nblk; ++b) {
        float s = dq_fp8_e4m3(scales[b * n + j]);
        uint lo = b * group_size;
        uint hi = min(lo + group_size, k);
        float bacc = 0.0f;
        for (uint p = lo; p < hi; ++p) {
            uint idx = p * n + j;
            uint byte_idx = idx >> 1;
            uint nib = ((idx & 1u) == 0u) ? (uint(wq[byte_idx]) & 0x0Fu) : (uint(wq[byte_idx]) >> 4);
            bacc += x[i * k + p] * DQ_FP4_E2M1[nib];
        }
        acc += bacc * s;
    }
    out[i * n + j] = acc * gs;
}

// MxFp4x2 two-level residual E2M1 fused decode-matmul. wq = [plane0|plane1]
// (E2M1 nibbles packed 2/byte over the [k,n] grid); scales = [s0|s1] f32 per
// (block = k/group_size, n). out[i,j] = sum_p x[i,p]·(s0·LUT[q0] + s1·LUT[q1]).
// Same math + layout as rlx_cpu::dequant_matmul_mxfp4x2.
kernel void dequant_matmul_mxfp4x2(
    device const float* x      [[buffer(0)]],
    device const uchar* wq     [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device float* out          [[buffer(3)]],
    constant uint& m           [[buffer(4)]],
    constant uint& k           [[buffer(5)]],
    constant uint& n           [[buffer(6)]],
    constant uint& group_size  [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= m * n) return;
    uint i = gid / n;
    uint j = gid % n;
    uint total = k * n;
    uint plane = (total + 1u) >> 1;                     // bytes per nibble plane
    uint g = (group_size == 0u) ? 1u : group_size;
    uint nblk = (k + g - 1u) / g;
    // Accumulate each residual plane's dot per block, then fold each plane's
    // block scale in once (s0·Σq0 + s1·Σq1). Reassociation only.
    float acc = 0.0f;
    for (uint b = 0u; b < nblk; ++b) {
        float s0 = scales[b * n + j];
        float s1 = scales[nblk * n + b * n + j];
        uint lo = b * g;
        uint hi = min(lo + g, k);
        float a0 = 0.0f;
        float a1 = 0.0f;
        for (uint p = lo; p < hi; ++p) {
            uint idx = p * n + j;
            uint byte_idx = idx >> 1;
            uint sh = ((idx & 1u) == 0u) ? 0u : 4u;
            uint q0 = (uint(wq[byte_idx]) >> sh) & 0x0Fu;
            uint q1 = (uint(wq[plane + byte_idx]) >> sh) & 0x0Fu;
            float xv = x[i * k + p];
            a0 += xv * DQ_FP4_E2M1[q0];
            a1 += xv * DQ_FP4_E2M1[q1];
        }
        acc += a0 * s0 + a1 * s1;
    }
    out[i * n + j] = acc;
}

// MLX Linear packs: w [n,k] along K. kind 0=affine, 1=mxfp4, 2=mxfp8.
inline float mlx_e8m0(uint s) {
    if (s == 0u) return as_type<float>(uint(0x0040u) << 16);
    return as_type<float>(s << 23);
}
inline float mlx_group_scale(uint s, uint gs) {
    return (gs == 16u) ? dq_fp8_e4m3(uchar(s)) : mlx_e8m0(s);
}
inline uint mlx_pack_factor(uint bits) {
    if (bits == 2u || bits == 4u || bits == 8u) return 8u / bits;
    if (bits == 3u || bits == 5u) return 8u;
    if (bits == 6u) return 4u;
    return 1u;
}
inline uint mlx_bpp(uint bits) {
    if (bits == 2u || bits == 4u || bits == 8u) return 1u;
    if (bits == 3u || bits == 6u) return 3u;
    if (bits == 5u) return 5u;
    return 1u;
}

// Decode GEMV (m==1): one threadgroup per output column; threads split K.
kernel void dequant_matmul_mlx_gemv(
    device const float* x      [[buffer(0)]],
    device const uchar* wq     [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device float* out          [[buffer(4)]],
    constant uint& k           [[buffer(5)]],
    constant uint& n           [[buffer(6)]],
    constant uint& kind        [[buffer(7)]],
    constant uint& bits        [[buffer(8)]],
    constant uint& group_size  [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint j [[threadgroup_position_in_grid]],
    uint tpg [[threads_per_threadgroup]]
) {
    if (j >= n) return;
    uint gs = group_size;
    uint n_groups = k / gs;
    device const uchar* scale_u = (device const uchar*)scales;
    threadgroup float smem[256];
    float acc = 0.0f;
    for (uint p = tid; p < k; p += tpg) {
        uint g = p / gs;
        float w_dq;
        if (kind == 0u) {
            uint pf = mlx_pack_factor(bits);
            uint bpp = mlx_bpp(bits);
            uint packs_in_group = gs / pf;
            uint local = p % gs;
            uint row_base = j * n_groups * packs_in_group * bpp + g * packs_in_group * bpp;
            uint code = 0u;
            if (bits == 2u || bits == 4u || bits == 8u) {
                uint pack_idx = local / pf;
                uint in_pack = local % pf;
                uchar byte = wq[row_base + pack_idx];
                uint mask = (1u << bits) - 1u;
                code = (uint(byte) >> (in_pack * bits)) & mask;
            } else if (bits == 3u) {
                uint pack_idx = local / 8u;
                uint in_pack = local % 8u;
                uint bo = row_base + pack_idx * 3u;
                uchar b0 = wq[bo], b1 = wq[bo + 1], b2 = wq[bo + 2];
                uint codes[8] = {
                    uint(b0) & 0x7u,
                    (uint(b0) & 0x38u) >> 3,
                    ((uint(b0) & 0xc0u) >> 6) + ((uint(b1) & 0x1u) << 2),
                    (uint(b1) & 0xeu) >> 1,
                    (uint(b1) & 0x70u) >> 4,
                    ((uint(b1) & 0x80u) >> 7) + ((uint(b2) & 0x3u) << 1),
                    (uint(b2) & 0x1cu) >> 2,
                    (uint(b2) & 0xe0u) >> 5
                };
                code = codes[in_pack];
            } else if (bits == 5u) {
                uint pack_idx = local / 8u;
                uint in_pack = local % 8u;
                uint bo = row_base + pack_idx * 5u;
                uchar b0 = wq[bo], b1 = wq[bo+1], b2 = wq[bo+2], b3 = wq[bo+3], b4 = wq[bo+4];
                uint codes[8] = {
                    uint(b0) & 0x1fu,
                    ((uint(b0) & 0xe0u) >> 5) + ((uint(b1) & 0x3u) << 3),
                    (uint(b1) & 0x7cu) >> 2,
                    ((uint(b1) & 0x80u) >> 7) + ((uint(b2) & 0xfu) << 1),
                    ((uint(b2) & 0xf0u) >> 4) + ((uint(b3) & 0x1u) << 4),
                    (uint(b3) & 0x3eu) >> 1,
                    ((uint(b3) & 0xc0u) >> 6) + ((uint(b4) & 0x7u) << 2),
                    (uint(b4) & 0xf8u) >> 3
                };
                code = codes[in_pack];
            } else {
                uint pack_idx = local / 4u;
                uint in_pack = local % 4u;
                uint bo = row_base + pack_idx * 3u;
                uchar b0 = wq[bo], b1 = wq[bo+1], b2 = wq[bo+2];
                uint codes[4] = {
                    uint(b0) & 0x3fu,
                    ((uint(b0) >> 6) & 0x03u) + ((uint(b1) & 0x0fu) << 2),
                    ((uint(b1) >> 4) & 0x0fu) + ((uint(b2) & 0x03u) << 4),
                    (uint(b2) >> 2) & 0x3fu
                };
                code = codes[in_pack];
            }
            w_dq = scales[j * n_groups + g] * float(code) + biases[j * n_groups + g];
        } else if (kind == 1u) {
            uint bidx = j * (k / 2u) + (p / 2u);
            uchar byte = wq[bidx];
            uint nib = ((p & 1u) == 0u) ? (uint(byte) & 0x0Fu) : (uint(byte) >> 4);
            w_dq = DQ_FP4_E2M1[nib] * mlx_group_scale(uint(scale_u[j * n_groups + g]), gs);
        } else {
            w_dq = dq_fp8_e4m3(wq[j * k + p]) * mlx_group_scale(uint(scale_u[j * n_groups + g]), gs);
        }
        acc += x[p] * w_dq;
    }
    smem[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg >> 1u; s > 0u; s >>= 1u) {
        if (tid < s) smem[tid] += smem[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) out[j] = smem[0];
}

inline float mlx_bf16(ushort b) { return as_type<float>(uint(b) << 16); }

// Grouped decode GEMV (m==1) — the native MoE expert kernel. Identical dequant
// math to dequant_matmul_mlx_gemv, but the (single) row's expert e = e_idx[0]
// offsets into the stacked [E,n,k] weight slab (wq + e*slab_bytes) and the
// [E,n,n_groups] scales/biases (+ e*n*n_groups). Replaces the CPU host-delegate
// for the DeepSeek-V4 MoE — reads packed codes straight from unified memory and
// dequant+matmuls on-GPU. `scale_bf16`: scales/biases are BF16 (2B, the model's
// per-expert slab format — already e8m0-decoded) rather than f32/e8m0-uchar.
// One threadgroup per output column; threads split K.
kernel void grouped_dequant_matmul_mlx_gemv(
    device const float* x       [[buffer(0)]],
    device const uchar* wq      [[buffer(1)]],
    device const float* scales  [[buffer(2)]],
    device const float* biases  [[buffer(3)]],
    device float* out           [[buffer(4)]],
    device const float* e_idx   [[buffer(5)]],
    constant uint& k            [[buffer(6)]],
    constant uint& n            [[buffer(7)]],
    constant uint& kind         [[buffer(8)]],
    constant uint& bits         [[buffer(9)]],
    constant uint& group_size   [[buffer(10)]],
    constant uint& slab_bytes   [[buffer(11)]],
    constant uint& scale_bf16   [[buffer(12)]],
    uint tid [[thread_index_in_threadgroup]],
    uint j [[threadgroup_position_in_grid]],
    uint tpg [[threads_per_threadgroup]]
) {
    if (j >= n) return;
    uint gs = group_size;
    uint n_groups = k / gs;
    uint e = uint(e_idx[0] + 0.5f);              // this row's expert
    device const uchar* wqe = wq + e * slab_bytes;   // expert weight slab
    uint sc_off = e * n * n_groups;                  // expert scale/bias base
    device const uchar* scale_u = (device const uchar*)scales;
    device const ushort* scale_bf = (device const ushort*)scales;
    device const ushort* bias_bf = (device const ushort*)biases;
    threadgroup float smem[256];
    float acc = 0.0f;
    for (uint p = tid; p < k; p += tpg) {
        uint g = p / gs;
        float w_dq;
        if (kind == 0u) {
            uint pf = mlx_pack_factor(bits);
            uint bpp = mlx_bpp(bits);
            uint packs_in_group = gs / pf;
            uint local = p % gs;
            uint row_base = j * n_groups * packs_in_group * bpp + g * packs_in_group * bpp;
            uint code = 0u;
            if (bits == 2u || bits == 4u || bits == 8u) {
                uint pack_idx = local / pf;
                uint in_pack = local % pf;
                uchar byte = wqe[row_base + pack_idx];
                uint mask = (1u << bits) - 1u;
                code = (uint(byte) >> (in_pack * bits)) & mask;
            } else if (bits == 3u) {
                uint pack_idx = local / 8u;
                uint in_pack = local % 8u;
                uint bo = row_base + pack_idx * 3u;
                uchar b0 = wqe[bo], b1 = wqe[bo + 1], b2 = wqe[bo + 2];
                uint codes[8] = {
                    uint(b0) & 0x7u,
                    (uint(b0) & 0x38u) >> 3,
                    ((uint(b0) & 0xc0u) >> 6) + ((uint(b1) & 0x1u) << 2),
                    (uint(b1) & 0xeu) >> 1,
                    (uint(b1) & 0x70u) >> 4,
                    ((uint(b1) & 0x80u) >> 7) + ((uint(b2) & 0x3u) << 1),
                    (uint(b2) & 0x1cu) >> 2,
                    (uint(b2) & 0xe0u) >> 5
                };
                code = codes[in_pack];
            } else if (bits == 5u) {
                uint pack_idx = local / 8u;
                uint in_pack = local % 8u;
                uint bo = row_base + pack_idx * 5u;
                uchar b0 = wqe[bo], b1 = wqe[bo+1], b2 = wqe[bo+2], b3 = wqe[bo+3], b4 = wqe[bo+4];
                uint codes[8] = {
                    uint(b0) & 0x1fu,
                    ((uint(b0) & 0xe0u) >> 5) + ((uint(b1) & 0x3u) << 3),
                    (uint(b1) & 0x7cu) >> 2,
                    ((uint(b1) & 0x80u) >> 7) + ((uint(b2) & 0xfu) << 1),
                    ((uint(b2) & 0xf0u) >> 4) + ((uint(b3) & 0x1u) << 4),
                    (uint(b3) & 0x3eu) >> 1,
                    ((uint(b3) & 0xc0u) >> 6) + ((uint(b4) & 0x7u) << 2),
                    (uint(b4) & 0xf8u) >> 3
                };
                code = codes[in_pack];
            } else {
                uint pack_idx = local / 4u;
                uint in_pack = local % 4u;
                uint bo = row_base + pack_idx * 3u;
                uchar b0 = wqe[bo], b1 = wqe[bo+1], b2 = wqe[bo+2];
                uint codes[4] = {
                    uint(b0) & 0x3fu,
                    ((uint(b0) >> 6) & 0x03u) + ((uint(b1) & 0x0fu) << 2),
                    ((uint(b1) >> 4) & 0x0fu) + ((uint(b2) & 0x03u) << 4),
                    (uint(b2) >> 2) & 0x3fu
                };
                code = codes[in_pack];
            }
            uint si = sc_off + j * n_groups + g;
            float sv = scale_bf16 ? mlx_bf16(scale_bf[si]) : scales[si];
            float bv = scale_bf16 ? mlx_bf16(bias_bf[si]) : biases[si];
            w_dq = sv * float(code) + bv;
        } else if (kind == 1u) {
            uint bidx = j * (k / 2u) + (p / 2u);
            uchar byte = wqe[bidx];
            uint nib = ((p & 1u) == 0u) ? (uint(byte) & 0x0Fu) : (uint(byte) >> 4);
            uint si = sc_off + j * n_groups + g;
            float sv = scale_bf16 ? mlx_bf16(scale_bf[si]) : mlx_group_scale(uint(scale_u[si]), gs);
            w_dq = DQ_FP4_E2M1[nib] * sv;
        } else {
            uint si = sc_off + j * n_groups + g;
            float sv = scale_bf16 ? mlx_bf16(scale_bf[si]) : mlx_group_scale(uint(scale_u[si]), gs);
            w_dq = dq_fp8_e4m3(wqe[j * k + p]) * sv;
        }
        acc += x[p] * w_dq;
    }
    smem[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg >> 1u; s > 0u; s >>= 1u) {
        if (tid < s) smem[tid] += smem[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) out[j] = smem[0];
}

// Prefill: one threadgroup per (col, row_tile); threads split K and stage
// an X tile in threadgroup memory (TM=8 rows × threads_per_threadgroup).
// Threadgroup id is linearized: tg = col * n_row_tiles + row_tile.
kernel void dequant_matmul_mlx_gemm(
    device const float* x      [[buffer(0)]],
    device const uchar* wq     [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device float* out          [[buffer(4)]],
    constant uint& m           [[buffer(5)]],
    constant uint& k           [[buffer(6)]],
    constant uint& n           [[buffer(7)]],
    constant uint& kind        [[buffer(8)]],
    constant uint& bits        [[buffer(9)]],
    constant uint& group_size  [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]],
    uint tpg [[threads_per_threadgroup]]
) {
    const uint TM = 8u;
    uint n_row_tiles = (m + TM - 1u) / TM;
    uint col = tg / n_row_tiles;
    uint row0 = (tg - col * n_row_tiles) * TM;
    if (col >= n) return;
    uint gs = group_size;
    uint n_groups = k / gs;
    device const uchar* scale_u = (device const uchar*)scales;
    threadgroup float xs[8 * 256];
    threadgroup float smem[8 * 256];
    float acc[8];
    for (uint t = 0u; t < TM; ++t) acc[t] = 0.0f;
    for (uint p0 = 0u; p0 < k; p0 += tpg) {
        uint p = p0 + tid;
        for (uint t = 0u; t < TM; ++t) {
            uint row = row0 + t;
            float v = 0.0f;
            if (row < m && p < k) v = x[row * k + p];
            xs[t * tpg + tid] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (p < k) {
            uint g = p / gs;
            float w_dq;
            if (kind == 0u) {
                uint pf = mlx_pack_factor(bits);
                uint bpp = mlx_bpp(bits);
                uint packs_in_group = gs / pf;
                uint local = p % gs;
                uint row_base = col * n_groups * packs_in_group * bpp + g * packs_in_group * bpp;
                uint code = 0u;
                if (bits == 2u || bits == 4u || bits == 8u) {
                    uint pack_idx = local / pf;
                    uint in_pack = local % pf;
                    uchar byte = wq[row_base + pack_idx];
                    uint mask = (1u << bits) - 1u;
                    code = (uint(byte) >> (in_pack * bits)) & mask;
                } else if (bits == 3u) {
                    uint pack_idx = local / 8u;
                    uint in_pack = local % 8u;
                    uint bo = row_base + pack_idx * 3u;
                    uchar b0 = wq[bo], b1 = wq[bo + 1], b2 = wq[bo + 2];
                    uint codes[8] = {
                        uint(b0) & 0x7u, (uint(b0) & 0x38u) >> 3,
                        ((uint(b0) & 0xc0u) >> 6) + ((uint(b1) & 0x1u) << 2),
                        (uint(b1) & 0xeu) >> 1, (uint(b1) & 0x70u) >> 4,
                        ((uint(b1) & 0x80u) >> 7) + ((uint(b2) & 0x3u) << 1),
                        (uint(b2) & 0x1cu) >> 2, (uint(b2) & 0xe0u) >> 5
                    };
                    code = codes[in_pack];
                } else if (bits == 5u) {
                    uint pack_idx = local / 8u;
                    uint in_pack = local % 8u;
                    uint bo = row_base + pack_idx * 5u;
                    uchar b0 = wq[bo], b1 = wq[bo+1], b2 = wq[bo+2], b3 = wq[bo+3], b4 = wq[bo+4];
                    uint codes[8] = {
                        uint(b0) & 0x1fu,
                        ((uint(b0) & 0xe0u) >> 5) + ((uint(b1) & 0x3u) << 3),
                        (uint(b1) & 0x7cu) >> 2,
                        ((uint(b1) & 0x80u) >> 7) + ((uint(b2) & 0xfu) << 1),
                        ((uint(b2) & 0xf0u) >> 4) + ((uint(b3) & 0x1u) << 4),
                        (uint(b3) & 0x3eu) >> 1,
                        ((uint(b3) & 0xc0u) >> 6) + ((uint(b4) & 0x7u) << 2),
                        (uint(b4) & 0xf8u) >> 3
                    };
                    code = codes[in_pack];
                } else {
                    uint pack_idx = local / 4u;
                    uint in_pack = local % 4u;
                    uint bo = row_base + pack_idx * 3u;
                    uchar b0 = wq[bo], b1 = wq[bo+1], b2 = wq[bo+2];
                    uint codes[4] = {
                        uint(b0) & 0x3fu,
                        ((uint(b0) >> 6) & 0x03u) + ((uint(b1) & 0x0fu) << 2),
                        ((uint(b1) >> 4) & 0x0fu) + ((uint(b2) & 0x03u) << 4),
                        (uint(b2) >> 2) & 0x3fu
                    };
                    code = codes[in_pack];
                }
                w_dq = scales[col * n_groups + g] * float(code) + biases[col * n_groups + g];
            } else if (kind == 1u) {
                uint bidx = col * (k / 2u) + (p / 2u);
                uchar byte = wq[bidx];
                uint nib = ((p & 1u) == 0u) ? (uint(byte) & 0x0Fu) : (uint(byte) >> 4);
                w_dq = DQ_FP4_E2M1[nib] * mlx_group_scale(uint(scale_u[col * n_groups + g]), gs);
            } else {
                w_dq = dq_fp8_e4m3(wq[col * k + p]) * mlx_group_scale(uint(scale_u[col * n_groups + g]), gs);
            }
            for (uint t = 0u; t < TM; ++t) {
                acc[t] += xs[t * tpg + tid] * w_dq;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint t = 0u; t < TM; ++t) smem[t * tpg + tid] = acc[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg >> 1u; s > 0u; s >>= 1u) {
        if (tid < s) {
            for (uint t = 0u; t < TM; ++t) {
                smem[t * tpg + tid] += smem[t * tpg + tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        for (uint t = 0u; t < TM; ++t) {
            uint row = row0 + t;
            if (row < m) out[row * n + col] = smem[t * tpg];
        }
    }
}

// Dequant one weight element w[col, p] from expert slab `wqe` (+ scale base
// `sc_off`) for the grouped MoE GEMM — mirrors the GEMV dequant math (all bit
// widths + scale_bf16). `col` is the output/weight-row index.
inline float mlx_grouped_w(
    device const uchar* wqe,
    device const float* scales,
    device const float* biases,
    uint sc_off, uint col, uint p, uint g,
    uint n_groups, uint gs, uint kind, uint bits, uint scale_bf16, uint k
) {
    device const uchar* scale_u = (device const uchar*)scales;
    device const ushort* scale_bf = (device const ushort*)scales;
    device const ushort* bias_bf = (device const ushort*)biases;
    if (kind == 0u) {
        uint pf = mlx_pack_factor(bits);
        uint bpp = mlx_bpp(bits);
        uint packs_in_group = gs / pf;
        uint local = p % gs;
        uint row_base = col * n_groups * packs_in_group * bpp + g * packs_in_group * bpp;
        uint code = 0u;
        if (bits == 2u || bits == 4u || bits == 8u) {
            uint pack_idx = local / pf;
            uint in_pack = local % pf;
            uchar byte = wqe[row_base + pack_idx];
            uint mask = (1u << bits) - 1u;
            code = (uint(byte) >> (in_pack * bits)) & mask;
        } else if (bits == 3u) {
            uint pack_idx = local / 8u; uint in_pack = local % 8u;
            uint bo = row_base + pack_idx * 3u;
            uchar b0 = wqe[bo], b1 = wqe[bo+1], b2 = wqe[bo+2];
            uint codes[8] = {
                uint(b0) & 0x7u, (uint(b0) & 0x38u) >> 3,
                ((uint(b0) & 0xc0u) >> 6) + ((uint(b1) & 0x1u) << 2),
                (uint(b1) & 0xeu) >> 1, (uint(b1) & 0x70u) >> 4,
                ((uint(b1) & 0x80u) >> 7) + ((uint(b2) & 0x3u) << 1),
                (uint(b2) & 0x1cu) >> 2, (uint(b2) & 0xe0u) >> 5
            };
            code = codes[in_pack];
        } else if (bits == 5u) {
            uint pack_idx = local / 8u; uint in_pack = local % 8u;
            uint bo = row_base + pack_idx * 5u;
            uchar b0 = wqe[bo], b1 = wqe[bo+1], b2 = wqe[bo+2], b3 = wqe[bo+3], b4 = wqe[bo+4];
            uint codes[8] = {
                uint(b0) & 0x1fu,
                ((uint(b0) & 0xe0u) >> 5) + ((uint(b1) & 0x3u) << 3),
                (uint(b1) & 0x7cu) >> 2,
                ((uint(b1) & 0x80u) >> 7) + ((uint(b2) & 0xfu) << 1),
                ((uint(b2) & 0xf0u) >> 4) + ((uint(b3) & 0x1u) << 4),
                (uint(b3) & 0x3eu) >> 1,
                ((uint(b3) & 0xc0u) >> 6) + ((uint(b4) & 0x7u) << 2),
                (uint(b4) & 0xf8u) >> 3
            };
            code = codes[in_pack];
        } else {
            uint pack_idx = local / 4u; uint in_pack = local % 4u;
            uint bo = row_base + pack_idx * 3u;
            uchar b0 = wqe[bo], b1 = wqe[bo+1], b2 = wqe[bo+2];
            uint codes[4] = {
                uint(b0) & 0x3fu,
                ((uint(b0) >> 6) & 0x03u) + ((uint(b1) & 0x0fu) << 2),
                ((uint(b1) >> 4) & 0x0fu) + ((uint(b2) & 0x03u) << 4),
                (uint(b2) >> 2) & 0x3fu
            };
            code = codes[in_pack];
        }
        uint si = sc_off + col * n_groups + g;
        float sv = scale_bf16 ? mlx_bf16(scale_bf[si]) : scales[si];
        float bv = scale_bf16 ? mlx_bf16(bias_bf[si]) : biases[si];
        return sv * float(code) + bv;
    } else if (kind == 1u) {
        uint bidx = col * (k / 2u) + (p / 2u);
        uchar byte = wqe[bidx];
        uint nib = ((p & 1u) == 0u) ? (uint(byte) & 0x0Fu) : (uint(byte) >> 4);
        uint si = sc_off + col * n_groups + g;
        float sv = scale_bf16 ? mlx_bf16(scale_bf[si]) : mlx_group_scale(uint(scale_u[si]), gs);
        return DQ_FP4_E2M1[nib] * sv;
    } else {
        uint si = sc_off + col * n_groups + g;
        float sv = scale_bf16 ? mlx_bf16(scale_bf[si]) : mlx_group_scale(uint(scale_u[si]), gs);
        return dq_fp8_e4m3(wqe[col * k + p]) * sv;
    }
}

// Grouped prefill GEMM — the native MoE expert kernel for m>1. Each row's expert
// e = e_idx[row] offsets into the stacked slab; since rows in a TM-tile can route
// to different experts, w is dequanted per row (no column-share). One threadgroup
// per (col, row_tile); threads split K and stage an X tile in threadgroup memory.
kernel void grouped_dequant_matmul_mlx_gemm(
    device const float* x      [[buffer(0)]],
    device const uchar* wq     [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device float* out          [[buffer(4)]],
    device const float* e_idx  [[buffer(5)]],
    constant uint& m           [[buffer(6)]],
    constant uint& k           [[buffer(7)]],
    constant uint& n           [[buffer(8)]],
    constant uint& kind        [[buffer(9)]],
    constant uint& bits        [[buffer(10)]],
    constant uint& group_size  [[buffer(11)]],
    constant uint& slab_bytes  [[buffer(12)]],
    constant uint& scale_bf16  [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]],
    uint tpg [[threads_per_threadgroup]]
) {
    const uint TM = 8u;
    uint n_row_tiles = (m + TM - 1u) / TM;
    uint col = tg / n_row_tiles;
    uint row0 = (tg - col * n_row_tiles) * TM;
    if (col >= n) return;
    uint gs = group_size;
    uint n_groups = k / gs;
    threadgroup float xs[8 * 256];
    threadgroup float smem[8 * 256];
    float acc[8];
    for (uint t = 0u; t < TM; ++t) acc[t] = 0.0f;
    for (uint p0 = 0u; p0 < k; p0 += tpg) {
        uint p = p0 + tid;
        for (uint t = 0u; t < TM; ++t) {
            uint row = row0 + t;
            float v = 0.0f;
            if (row < m && p < k) v = x[row * k + p];
            xs[t * tpg + tid] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (p < k) {
            uint g = p / gs;
            for (uint t = 0u; t < TM; ++t) {
                uint row = row0 + t;
                if (row < m) {
                    uint e = uint(e_idx[row] + 0.5f);
                    device const uchar* wqe = wq + e * slab_bytes;
                    uint sc_off = e * n * n_groups;
                    float w_dq = mlx_grouped_w(wqe, scales, biases, sc_off, col, p, g, n_groups, gs, kind, bits, scale_bf16, k);
                    acc[t] += xs[t * tpg + tid] * w_dq;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint t = 0u; t < TM; ++t) smem[t * tpg + tid] = acc[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg >> 1u; s > 0u; s >>= 1u) {
        if (tid < s) {
            for (uint t = 0u; t < TM; ++t) {
                smem[t * tpg + tid] += smem[t * tpg + tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        for (uint t = 0u; t < TM; ++t) {
            uint row = row0 + t;
            if (row < m) out[row * n + col] = smem[t * tpg];
        }
    }
}

// in-place SiLU: x * sigmoid(x)
kernel void silu_inplace(
    device float* data [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    float x = data[gid];
    data[gid] = x / (1.0 + exp(-x));
}

// Fused SwiGLU: input is concat'd [outer, 2N] (per-row up || gate).
// Output: [outer, N] where out[r,i] = up[r,i] * silu(gate[r,i]).
// One thread per output element. Each thread reads exactly two source
// values from the same row (up + gate) and writes one — no inter-thread
// communication, no shared memory, no reductions.
//
// Grid: total output elements (outer * N). The thread maps to (row, col)
// via the n_half stride. Up and gate live at offsets [row*2N + col] and
// [row*2N + N + col] respectively.
kernel void fused_swiglu(
    device const float* x  [[buffer(0)]],   // [outer, 2*n_half]
    device float* out      [[buffer(1)]],   // [outer, n_half]
    constant uint& n_half  [[buffer(2)]],
    constant uint& total   [[buffer(3)]],   // outer * n_half
    constant uint& gate_first [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) return;
    uint row = gid / n_half;
    uint col = gid % n_half;
    uint base = row * (2u * n_half);
    float up;
    float gate;
    if (gate_first != 0u) {
        gate = x[base + col];
        up   = x[base + n_half + col];
    } else {
        up   = x[base + col];
        gate = x[base + n_half + col];
    }
    out[gid] = up * (gate / (1.0f + exp(-gate)));
}

// Half-precision variant: f16 in/out. Computation in f32 (silu's exp can
// underflow at half precision). Same dispatch as fused_swiglu.
kernel void fused_swiglu_h(
    device const half* x   [[buffer(0)]],
    device half* out       [[buffer(1)]],
    constant uint& n_half  [[buffer(2)]],
    constant uint& total   [[buffer(3)]],
    constant uint& gate_first [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) return;
    uint row = gid / n_half;
    uint col = gid % n_half;
    uint base = row * (2u * n_half);
    float up;
    float gate;
    if (gate_first != 0u) {
        gate = float(x[base + col]);
        up   = float(x[base + n_half + col]);
    } else {
        up   = float(x[base + col]);
        gate = float(x[base + n_half + col]);
    }
    out[gid] = half(up * (gate / (1.0f + exp(-gate))));
}

// SwiGLU + cast: f32 input, f16 output. Saves a separate cast pass when
// the next consumer wants half precision. Reserved for paths where the
// AutoMixedPrecision boundary lands right after SwiGLU.
kernel void fused_swiglu_cast_f32_to_f16(
    device const float* x  [[buffer(0)]],
    device half* out       [[buffer(1)]],
    constant uint& n_half  [[buffer(2)]],
    constant uint& total   [[buffer(3)]],
    constant uint& gate_first [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) return;
    uint row = gid / n_half;
    uint col = gid % n_half;
    uint base = row * (2u * n_half);
    float up;
    float gate;
    if (gate_first != 0u) {
        gate = x[base + col];
        up   = x[base + n_half + col];
    } else {
        up   = x[base + col];
        gate = x[base + n_half + col];
    }
    out[gid] = half(up * (gate / (1.0f + exp(-gate))));
}

// SwiGLU + cast: f16 input, f32 output. Symmetric to the above.
kernel void fused_swiglu_cast_f16_to_f32(
    device const half* x   [[buffer(0)]],
    device float* out      [[buffer(1)]],
    constant uint& n_half  [[buffer(2)]],
    constant uint& total   [[buffer(3)]],
    constant uint& gate_first [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) return;
    uint row = gid / n_half;
    uint col = gid % n_half;
    uint base = row * (2u * n_half);
    float up;
    float gate;
    if (gate_first != 0u) {
        gate = float(x[base + col]);
        up   = float(x[base + n_half + col]);
    } else {
        up   = float(x[base + col]);
        gate = float(x[base + n_half + col]);
    }
    out[gid] = up * (gate / (1.0f + exp(-gate)));
}

// LayerNorm: out = (x - mean) * inv_std * gamma + beta, per row
// One threadgroup per row; reductions via threadgroup memory.
kernel void layer_norm(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta  [[buffer(2)]],
    device float* output      [[buffer(3)]],
    constant uint& h          [[buffer(4)]],
    constant float& eps       [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float partial_sum[256];
    threadgroup float partial_dev[256];

    // TWO-PASS: var = mean((x−mean)²). The one-pass E[x²]−mean² catastrophically
    // cancels in f32 on rows with a large DC offset (pre-norm ViT/DINOv2) — this
    // matches the CPU oracle.
    // Pass 1: mean.
    float local_sum = 0.0;
    for (uint i = tid; i < h; i += tsize) {
        local_sum += input[row * h + i];
    }
    partial_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = partial_sum[0] / float(h);

    // Pass 2: mean of squared deviation (separate array → no extra barrier).
    float local_dev = 0.0;
    for (uint i = tid; i < h; i += tsize) {
        float d = input[row * h + i] - mean;
        local_dev += d * d;
    }
    partial_dev[tid] = local_dev;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_dev[tid] += partial_dev[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = fmax(0.0f, partial_dev[0] / float(h));
    float inv_std = rsqrt(var + eps);

    // Pass 3: normalize.
    for (uint i = tid; i < h; i += tsize) {
        float v = input[row * h + i];
        output[row * h + i] = (v - mean) * inv_std * gamma[i] + beta[i];
    }
}

// RMSNorm: out = (x / sqrt(mean(x^2) + eps)) * gamma + beta. No mean
// subtraction. Same dispatch shape as layer_norm (one threadgroup per row,
// power-of-2 reduction within the group).
kernel void rms_norm(
    device const char* arena [[buffer(0)]],
    constant ulong& in_off [[buffer(1)]],
    constant ulong& g_off [[buffer(2)]],
    constant ulong& b_off [[buffer(3)]],
    constant ulong& out_off [[buffer(4)]],
    constant uint& h [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    device const float* input = (device const float*)(arena + in_off);
    device const float* gamma = (device const float*)(arena + g_off);
    device const float* beta = (device const float*)(arena + b_off);
    device float* output = (device float*)(arena + out_off);
    threadgroup float partial[256];
    // Two-pass: mean(x²) = mean((x−mean)²) + mean². Subtract the row mean first
    // so the deviation sum stays well-conditioned under a large DC offset (the
    // canonical form CPU + all backends match).
    // Pass 1: row mean.
    float local_sum = 0.0f;
    for (uint i = tid; i < h; i += tsize) {
        local_sum += input[row * h + i];
    }
    partial[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = partial[0] / float(h);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Pass 2: mean of squared deviation.
    float local_sumsq = 0.0f;
    for (uint i = tid; i < h; i += tsize) {
        float d = input[row * h + i] - mean;
        local_sumsq += d * d;
    }
    partial[tid] = local_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(partial[0] / float(h) + mean * mean + eps);
    for (uint i = tid; i < h; i += tsize) {
        output[row * h + i] = input[row * h + i] * inv_rms * gamma[i] + beta[i];
    }
}

// GDN gated norm: out = rms_norm(x) * silu(z). Same dispatch as rms_norm.
kernel void rms_norm_mul_silu(
    device const char* arena [[buffer(0)]],
    constant ulong& in_off [[buffer(1)]],
    constant ulong& g_off [[buffer(2)]],
    constant ulong& b_off [[buffer(3)]],
    constant ulong& z_off [[buffer(4)]],
    constant ulong& out_off [[buffer(5)]],
    constant uint& h [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    device const float* input = (device const float*)(arena + in_off);
    device const float* gamma = (device const float*)(arena + g_off);
    device const float* beta = (device const float*)(arena + b_off);
    device const float* z = (device const float*)(arena + z_off);
    device float* output = (device float*)(arena + out_off);
    threadgroup float partial_sumsq[256];
    float local_sumsq = 0.0f;
    for (uint i = tid; i < h; i += tsize) {
        float v = input[row * h + i];
        local_sumsq += v * v;
    }
    partial_sumsq[tid] = local_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sumsq[tid] += partial_sumsq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(partial_sumsq[0] / float(h) + eps);
    for (uint i = tid; i < h; i += tsize) {
        float zv = z[row * h + i];
        float silu = zv / (1.0f + exp(-zv));
        output[row * h + i] =
            (input[row * h + i] * inv_rms * gamma[i] + beta[i]) * silu;
    }
}

// f16 RMSNorm: half I/O, float accumulation.
kernel void rms_norm_h(
    device const char* arena [[buffer(0)]],
    constant ulong& in_off [[buffer(1)]],
    constant ulong& g_off [[buffer(2)]],
    constant ulong& b_off [[buffer(3)]],
    constant ulong& out_off [[buffer(4)]],
    constant uint& h [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    device const half* input = (device const half*)(arena + in_off);
    device const half* gamma = (device const half*)(arena + g_off);
    device const half* beta = (device const half*)(arena + b_off);
    device half* output = (device half*)(arena + out_off);
    threadgroup float partial_sumsq[256];
    float local_sumsq = 0.0f;
    for (uint i = tid; i < h; i += tsize) {
        float v = float(input[row * h + i]);
        local_sumsq += v * v;
    }
    partial_sumsq[tid] = local_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sumsq[tid] += partial_sumsq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(partial_sumsq[0] / float(h) + eps);
    for (uint i = tid; i < h; i += tsize) {
        float v = float(input[row * h + i]);
        output[row * h + i] = half(v * inv_rms * float(gamma[i]) + float(beta[i]));
    }
}

// f16 standalone softmax along the last axis. Half I/O, float accumulation
// for max + exp-sum (matters: f16 sum overflows above ~65k summands and
// exp() loses precision for moderate negatives).
kernel void softmax_lastax_h(
    device half* data     [[buffer(0)]],
    constant uint& cols   [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float partial[256];
    uint base = row * cols;

    float local_max = -INFINITY;
    for (uint i = tid; i < cols; i += tsize) {
        local_max = max(local_max, float(data[base + i]));
    }
    partial[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] = max(partial[tid], partial[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = partial[0];

    float local_sum = 0.0f;
    for (uint i = tid; i < cols; i += tsize) {
        float e = exp(float(data[base + i]) - row_max);
        data[base + i] = half(e);
        local_sum += e;
    }
    partial[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / partial[0];

    for (uint i = tid; i < cols; i += tsize) {
        data[base + i] = half(float(data[base + i]) * inv_sum);
    }
}

// f16 multi-axis reduce. Same op_kind encoding as reduce_axes; accumulate
// in float so 1e-2 .. 1e+4 f16 values don't lose precision summing across
// the reduced axis.
kernel void reduce_axes_h(
    device const half* src  [[buffer(0)]],
    device half* dst        [[buffer(1)]],
    constant uint& reduced  [[buffer(2)]],
    constant uint& inner    [[buffer(3)]],
    constant uint& op_kind  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint o = gid.y;
    if (i >= inner) return;
    float acc;
    if      (op_kind == 2) acc = -INFINITY;
    else if (op_kind == 3) acc =  INFINITY;
    else if (op_kind == 4) acc =  1.0f;
    else                   acc =  0.0f;

    uint base = o * reduced * inner + i;
    for (uint r = 0; r < reduced; ++r) {
        float v = float(src[base + r * inner]);
        if      (op_kind == 0 || op_kind == 1) acc += v;
        else if (op_kind == 2) acc = max(acc, v);
        else if (op_kind == 3) acc = min(acc, v);
        else                   acc *= v;
    }
    if (op_kind == 1) acc /= float(reduced);
    dst[o * inner + i] = half(acc);
}

// PLAN L2 — interpreted N-ary element-wise chain kernel.
// One thread per output element. Walks the chain encoding (4 u32s
// per step: op_kind, op_sub, lhs_enc, rhs_enc) into a private
// scratch register array. Operand encoding: bit 31 = src kind
// (0=Input, 1=Step), bits 0..30 = index. Caps: 32 steps, 16 inputs.
inline uint region_input_row_resize2x_nchw(
    uint gid,
    uint out_n,
    uint out_c,
    uint out_h,
    uint out_w
) {
    uint plane = out_c * out_h * out_w;
    uint local = gid % plane;
    uint batch = gid / plane;
    uint w_pos = local % out_w;
    uint tmp = local / out_w;
    uint h_pos = tmp % out_h;
    uint c_pos = tmp / out_h;
    uint in_w = out_w / 2u;
    uint in_h = out_h / 2u;
    uint in_plane = out_c * in_h * in_w;
    return batch * in_plane + c_pos * in_h * in_w + (h_pos / 2u) * in_w + (w_pos / 2u);
}

inline uint region_resolve_row(
    uint gid,
    uint kind,
    uint idx,
    uint prologue_row0,
    uint has_prologue_row0,
    uint prologue_input,
    uint scalar_input_mask,
    device const uint* input_modulus
) {
    if (kind != 0u) { return 0u; }
    if (has_prologue_row0 != 0u && idx == prologue_input) {
        return prologue_row0;
    }
    if ((scalar_input_mask & (1u << idx)) != 0u) { return 0u; }
    if (input_modulus[idx] != 0u) { return gid % input_modulus[idx]; }
    return gid;
}

kernel void elementwise_region(
    device float* arena              [[buffer(0)]],
    constant uint& len               [[buffer(1)]],
    constant uint& num_inputs        [[buffer(2)]],
    constant uint& num_steps         [[buffer(3)]],
    constant uint& dst_off           [[buffer(4)]],
    device const uint* input_offs    [[buffer(5)]],   // 16 entries
    device const uint* chain         [[buffer(6)]],   // 128 entries (32 steps * 4)
    constant uint& scalar_input_mask [[buffer(7)]],
    device const uint* input_modulus [[buffer(8)]],   // 16 entries
    constant uint& prologue          [[buffer(9)]],
    constant uint& out_n             [[buffer(10)]],
    constant uint& out_c             [[buffer(11)]],
    constant uint& out_h             [[buffer(12)]],
    constant uint& out_w             [[buffer(13)]],
    constant uint& prologue_input    [[buffer(14)]],
    uint3 gpos [[thread_position_in_grid]]
) {
    uint gid;
    if (prologue == 1u) {
        uint nc = gpos.z;
        uint ho = gpos.y;
        uint wo = gpos.x;
        if (nc >= out_n * out_c || ho >= out_h || wo >= out_w) { return; }
        gid = nc * out_h * out_w + ho * out_w + wo;
    } else {
        gid = gpos.x;
        if (gid >= len) { return; }
    }
    uint prologue_row0 = 0u;
    uint has_prologue_row0 = 0u;
    if (prologue == 1u) {
        prologue_row0 = region_input_row_resize2x_nchw(gid, out_n, out_c, out_h, out_w);
        has_prologue_row0 = 1u;
    }
    float scratch[32];
    uint last_idx = 0;
    for (uint k = 0; k < num_steps; ++k) {
        uint base    = k * 4;
        uint op_kind = chain[base + 0];
        uint op_sub  = chain[base + 1];
        uint lhs_enc = chain[base + 2];
        uint rhs_enc = chain[base + 3];

        // resolve_operand inline. Scalar-broadcast inputs read element
        // 0 regardless of gid (fast path); trailing-shape broadcast
        // reads `gid % input_modulus[idx]`. `input_modulus[idx]==0`
        // means "no broadcast" and the kernel reads gid directly.
        float lhs;
        {
            uint kind = lhs_enc >> 31;
            uint idx  = lhs_enc & 0x7FFFFFFFu;
            uint row = region_resolve_row(
                gid, kind, idx, prologue_row0, has_prologue_row0, prologue_input,
                scalar_input_mask, input_modulus);
            lhs = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
        }
        float result;
        if (op_kind == 4u) {
            // Where (3-operand select). op_sub carries cond_enc; lhs_enc
            // / rhs_enc carry on_true / on_false. lhs already resolved
            // above is on_true; resolve cond from op_sub and on_false
            // from rhs_enc here.
            float cond;
            {
                uint kind = op_sub >> 31;
                uint idx  = op_sub & 0x7FFFFFFFu;
                uint row = region_resolve_row(
                    gid, kind, idx, prologue_row0, has_prologue_row0, prologue_input,
                    scalar_input_mask, input_modulus);
                cond = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
            }
            float on_false;
            {
                uint kind = rhs_enc >> 31;
                uint idx  = rhs_enc & 0x7FFFFFFFu;
                uint row = region_resolve_row(
                    gid, kind, idx, prologue_row0, has_prologue_row0, prologue_input,
                    scalar_input_mask, input_modulus);
                on_false = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
            }
            result = (cond != 0.0f) ? lhs : on_false;
        } else if (op_kind == 0u) {
            // Activation
            if      (op_sub == 3u) result = max(lhs, 0.0f);                // Relu
            else if (op_sub == 0u || op_sub == 1u) {
                float c = 0.7978845608f;
                float inner = clamp(c * (lhs + 0.044715f * lhs * lhs * lhs), -15.0f, 15.0f);
                result = 0.5f * lhs * (1.0f + tanh(inner));                // Gelu
            }
            else if (op_sub == 2u) result = lhs / (1.0f + exp(-lhs));      // Silu
            else if (op_sub == 4u) result = 1.0f / (1.0f + exp(-lhs));     // Sigmoid
            else if (op_sub == 5u) result = tanh(clamp(lhs, -15.0f, 15.0f));
            else if (op_sub == 6u) result = exp(lhs);
            else if (op_sub == 7u) result = log(lhs);
            else if (op_sub == 8u) result = sqrt(lhs);
            else if (op_sub == 9u) result = 1.0f / sqrt(lhs);
            else if (op_sub == 10u) result = -lhs;
            else if (op_sub == 11u) result = fabs(lhs);
            else if (op_sub == 12u) result = round(lhs);
            else if (op_sub == 13u) result = sin(lhs);
            else if (op_sub == 14u) result = cos(lhs);
            else if (op_sub == 15u) result = tan(lhs);
            else if (op_sub == 16u) result = atan(lhs);
            else                    result = lhs;
        } else if (op_kind == 1u) {
            // Cast at f32-arena layer is identity
            result = lhs;
        } else {
            float rhs;
            {
                uint kind = rhs_enc >> 31;
                uint idx  = rhs_enc & 0x7FFFFFFFu;
                uint row = region_resolve_row(
                    gid, kind, idx, prologue_row0, has_prologue_row0, prologue_input,
                    scalar_input_mask, input_modulus);
                rhs = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
            }
            if (op_kind == 2u) {
                if      (op_sub == 0u) result = lhs + rhs;
                else if (op_sub == 1u) result = lhs - rhs;
                else if (op_sub == 2u) result = lhs * rhs;
                else if (op_sub == 3u) result = lhs / rhs;
                else if (op_sub == 4u) result = max(lhs, rhs);
                else if (op_sub == 5u) result = min(lhs, rhs);
                else                   result = rlx_pow_scalar(lhs, rhs);
            } else {
                bool b;
                if      (op_sub == 0u) b = (lhs == rhs);
                else if (op_sub == 1u) b = (lhs != rhs);
                else if (op_sub == 2u) b = (lhs <  rhs);
                else if (op_sub == 3u) b = (lhs <= rhs);
                else if (op_sub == 4u) b = (lhs >  rhs);
                else                   b = (lhs >= rhs);
                result = b ? 1.0f : 0.0f;
            }
        }
        scratch[k] = result;
        last_idx = k;
    }
    arena[dst_off + gid] = scratch[last_idx];
}

inline uint batch_region_resolve_row(
    uint gid,
    uint kind,
    uint idx,
    uint scalar_input_mask,
    constant uint* input_modulus
) {
    if (kind != 0u) { return 0u; }
    if ((scalar_input_mask & (1u << idx)) != 0u) { return 0u; }
    if (input_modulus[idx] != 0u) { return gid % input_modulus[idx]; }
    return gid;
}

// FKL batch horizontal fusion: one dispatch, thread_position_in_grid.z = slice index.
// Requires prologue == 0 (no resize prologue on batch slices).
kernel void batch_elementwise_region(
    device float* arena              [[buffer(0)]],
    constant uint& slice_len         [[buffer(1)]],
    constant uint& num_batch         [[buffer(2)]],
    constant uint& num_steps         [[buffer(3)]],
    constant uint& base_dst_off      [[buffer(4)]],
    constant uint& slice_elems       [[buffer(5)]],
    constant uint* batch_input_offs  [[buffer(6)]],   // 64 entries
    constant uint* chain             [[buffer(7)]],   // 128 entries
    constant uint& scalar_input_mask [[buffer(8)]],
    constant uint* input_modulus     [[buffer(9)]],   // 16 entries
    uint3 gpos [[thread_position_in_grid]]
) {
    uint batch_idx = gpos.z;
    if (batch_idx >= num_batch) { return; }
    uint i = gpos.x;
    if (i >= slice_len) { return; }

    uint input_offs[16];
    for (uint k = 0; k < 16u; ++k) { input_offs[k] = 0u; }
    input_offs[0] = batch_input_offs[batch_idx];
    uint dst_off = base_dst_off + batch_idx * slice_elems;

    float scratch[32];
    uint last_idx = 0;
    for (uint k = 0; k < num_steps; ++k) {
        uint base    = k * 4;
        uint op_kind = chain[base + 0];
        uint op_sub  = chain[base + 1];
        uint lhs_enc = chain[base + 2];
        uint rhs_enc = chain[base + 3];

        float lhs;
        {
            uint kind = lhs_enc >> 31;
            uint idx  = lhs_enc & 0x7FFFFFFFu;
            uint row = batch_region_resolve_row(
                i, kind, idx, scalar_input_mask, input_modulus);
            lhs = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
        }
        float result;
        if (op_kind == 4u) {
            float cond;
            {
                uint kind = op_sub >> 31;
                uint idx  = op_sub & 0x7FFFFFFFu;
                uint row = batch_region_resolve_row(
                    i, kind, idx, scalar_input_mask, input_modulus);
                cond = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
            }
            float on_false;
            {
                uint kind = rhs_enc >> 31;
                uint idx  = rhs_enc & 0x7FFFFFFFu;
                uint row = batch_region_resolve_row(
                    i, kind, idx, scalar_input_mask, input_modulus);
                on_false = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
            }
            result = (cond != 0.0f) ? lhs : on_false;
        } else if (op_kind == 0u) {
            if      (op_sub == 3u) result = max(lhs, 0.0f);
            else if (op_sub == 0u || op_sub == 1u) {
                float c = 0.7978845608f;
                float inner = clamp(c * (lhs + 0.044715f * lhs * lhs * lhs), -15.0f, 15.0f);
                result = 0.5f * lhs * (1.0f + tanh(inner));
            }
            else if (op_sub == 2u) result = lhs / (1.0f + exp(-lhs));
            else if (op_sub == 4u) result = 1.0f / (1.0f + exp(-lhs));
            else if (op_sub == 5u) result = tanh(clamp(lhs, -15.0f, 15.0f));
            else if (op_sub == 6u) result = exp(lhs);
            else if (op_sub == 7u) result = log(lhs);
            else if (op_sub == 8u) result = sqrt(lhs);
            else if (op_sub == 9u) result = 1.0f / sqrt(lhs);
            else if (op_sub == 10u) result = -lhs;
            else if (op_sub == 11u) result = fabs(lhs);
            else if (op_sub == 12u) result = round(lhs);
            else if (op_sub == 13u) result = sin(lhs);
            else if (op_sub == 14u) result = cos(lhs);
            else if (op_sub == 15u) result = tan(lhs);
            else if (op_sub == 16u) result = atan(lhs);
            else                    result = lhs;
        } else if (op_kind == 1u) {
            result = lhs;
        } else {
            float rhs;
            {
                uint kind = rhs_enc >> 31;
                uint idx  = rhs_enc & 0x7FFFFFFFu;
                uint row = batch_region_resolve_row(
                    i, kind, idx, scalar_input_mask, input_modulus);
                rhs = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
            }
            if (op_kind == 2u) {
                if      (op_sub == 0u) result = lhs + rhs;
                else if (op_sub == 1u) result = lhs - rhs;
                else if (op_sub == 2u) result = lhs * rhs;
                else if (op_sub == 3u) result = lhs / rhs;
                else if (op_sub == 4u) result = max(lhs, rhs);
                else if (op_sub == 5u) result = min(lhs, rhs);
                else                   result = rlx_pow_scalar(lhs, rhs);
            } else {
                bool b;
                if      (op_sub == 0u) b = (lhs == rhs);
                else if (op_sub == 1u) b = (lhs != rhs);
                else if (op_sub == 2u) b = (lhs <  rhs);
                else if (op_sub == 3u) b = (lhs <= rhs);
                else if (op_sub == 4u) b = (lhs >  rhs);
                else                   b = (lhs >= rhs);
                result = b ? 1.0f : 0.0f;
            }
        }
        scratch[k] = result;
        last_idx = k;
    }
    arena[dst_off + i] = scratch[last_idx];
}

// ── Gated DeltaNet scan (f32) ───────────────────────────────────────
// One thread per (batch, head) — same dispatch model as `selective_scan`.
// Each thread owns the full n×n state in device scratch (exclusive) and
// scans seq serially. Matches CPU `execute_gated_delta_net_f32`.
// Offsets are ulong float indices: Fara-sized arenas place ephemeral GDN
// scratch past 16 GiB, which overflows uint indexing.
#define GDN_MAX_N 128u

kernel void gated_delta_net(
    device float* arena        [[buffer(0)]],
    constant ulong& q_off      [[buffer(1)]],
    constant ulong& k_off      [[buffer(2)]],
    constant ulong& v_off      [[buffer(3)]],
    constant ulong& g_off      [[buffer(4)]],
    constant ulong& beta_off   [[buffer(5)]],
    constant ulong& state_off  [[buffer(6)]],
    constant ulong& dst_off    [[buffer(7)]],
    constant uint4& dims       [[buffer(8)]], // batch, seq, heads, n
    constant uint& use_carry   [[buffer(9)]],
    constant uint& gate_per_channel [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint b = dims.x, s = dims.y, h = dims.z, n = dims.w;
    if (n > GDN_MAX_N || n == 0u || gid >= b * h) return;

    const uint bi = gid / h;
    const uint hi = gid % h;
    const float scale = rsqrt(float(n));
    const uint hs_n = h * n;
    const ulong s_base = state_off + (ulong)((bi * h + hi) * n * n);
    device float* s_mat = arena + s_base;

    if (use_carry == 0u) {
        for (uint i = 0u; i < n * n; ++i) {
            s_mat[i] = 0.0f;
        }
    }

    float sk[GDN_MAX_N];

    for (uint ti = 0u; ti < s; ++ti) {
        const uint qkv_step = bi * s * hs_n + ti * hs_n + hi * n;
        const uint gb_step = bi * s * h + ti * h + hi;
        device const float* q_ptr = arena + q_off + qkv_step;
        device const float* k_ptr = arena + k_off + qkv_step;
        device const float* v_ptr = arena + v_off + qkv_step;
        const float g_exp = (gate_per_channel == 0u) ? exp(arena[g_off + gb_step]) : 0.0f;
        const float beta_t = arena[beta_off + gb_step];

        if (gate_per_channel != 0u) {
            // per-channel decay: S[i,j] *= exp(g[i]), g is [b,s,h,n]
            for (uint i = 0u; i < n; ++i) {
                const float a = exp(arena[g_off + qkv_step + i]);
                for (uint jj = 0u; jj < n; ++jj) {
                    s_mat[i * n + jj] *= a;
                }
            }
        } else {
            for (uint i = 0u; i < n * n; ++i) {
                s_mat[i] *= g_exp;
            }
        }
        for (uint j = 0u; j < n; ++j) {
            float acc = 0.0f;
            for (uint i = 0u; i < n; ++i) {
                acc += s_mat[i * n + j] * k_ptr[i];
            }
            sk[j] = (v_ptr[j] - acc) * beta_t;
        }
        device float* out = arena + dst_off + qkv_step;
        for (uint j = 0u; j < n; ++j) {
            const float d = sk[j];
            float y = 0.0f;
            for (uint i = 0u; i < n; ++i) {
                s_mat[i * n + j] += k_ptr[i] * d;
                y += s_mat[i * n + j] * q_ptr[i];
            }
            out[j] = y * scale;
        }
    }
}
// Simdgroup-cooperative GDN for n == 128 (Bonsai / Qwen3.5). llama.cpp-style:
// one simdgroup owns one state column; 32 threads × 4 floats keep that column
// in registers and use simd_sum — no threadgroup barriers inside the scan.
// Grid: (n/NSG, heads, batch) threadgroups, threads (32, NSG, 1).
// Opt-in via RLX_METAL_GDN_SG=1; column-private kernel above is the default.
constant uint GDN_SG_N = 128u;
constant uint GDN_SG_NSG = 4u;

kernel void gated_delta_net_sg(
    device float* arena        [[buffer(0)]],
    constant ulong& q_off      [[buffer(1)]],
    constant ulong& k_off      [[buffer(2)]],
    constant ulong& v_off      [[buffer(3)]],
    constant ulong& g_off      [[buffer(4)]],
    constant ulong& beta_off   [[buffer(5)]],
    constant ulong& state_off  [[buffer(6)]],
    constant ulong& dst_off    [[buffer(7)]],
    constant uint4& dims       [[buffer(8)]], // batch, seq, heads, n
    constant uint& use_carry   [[buffer(9)]],
    constant uint& gate_per_channel [[buffer(10)]],
    uint3 tgpig                [[threadgroup_position_in_grid]],
    uint3 tpitg                [[thread_position_in_threadgroup]]
) {
    const uint b = dims.x, s = dims.y, h = dims.z, n = dims.w;
    if (n != GDN_SG_N) return;

    const uint tx = tpitg.x; // 0..31 within simdgroup
    const uint ty = tpitg.y; // simdgroup index in threadgroup
    const uint bi = tgpig.z;
    const uint hi = tgpig.y;
    const uint col = tgpig.x * GDN_SG_NSG + ty; // state column j
    if (bi >= b || hi >= h || col >= n || tx >= 32u) return;

    const float scale = rsqrt(float(n));
    const ulong s_base = state_off + (ulong)((bi * h + hi) * n * n);
    device float* s_mat = arena + s_base;

    // Register tile of column `col`: S[is][col] for is = tx*4 .. tx*4+3.
    float ls[GDN_SG_NSG];
    #pragma unroll
    for (uint j = 0u; j < GDN_SG_NSG; ++j) {
        const uint is = tx * GDN_SG_NSG + j;
        ls[j] = (use_carry != 0u) ? s_mat[is * n + col] : 0.0f;
    }

    const uint hs_n = h * n;
    for (uint ti = 0u; ti < s; ++ti) {
        const uint qkv_step = bi * s * hs_n + ti * hs_n + hi * n;
        const uint gb_step  = bi * s * h + ti * h + hi;
        device const float* q_ptr = arena + q_off + qkv_step;
        device const float* k_ptr = arena + k_off + qkv_step;
        device const float* v_ptr = arena + v_off + qkv_step;
        const float g_exp = (gate_per_channel == 0u) ? exp(arena[g_off + gb_step]) : 0.0f;
        const float beta_t = arena[beta_off + gb_step];

        float s_k = 0.0f;
        #pragma unroll
        for (uint j = 0u; j < GDN_SG_NSG; ++j) {
            const uint is = tx * GDN_SG_NSG + j;
            // per-channel decay uses the row/key channel `is`; per-head uses g_exp.
            const float a = (gate_per_channel != 0u) ? exp(arena[g_off + qkv_step + is]) : g_exp;
            ls[j] *= a;
            s_k += ls[j] * k_ptr[is];
        }
        s_k = simd_sum(s_k);

        const float d = (v_ptr[col] - s_k) * beta_t;

        float y = 0.0f;
        #pragma unroll
        for (uint j = 0u; j < GDN_SG_NSG; ++j) {
            const uint is = tx * GDN_SG_NSG + j;
            ls[j] += k_ptr[is] * d;
            y += ls[j] * q_ptr[is];
        }
        y = simd_sum(y);

        if (tx == 0u) {
            arena[dst_off + qkv_step + col] = y * scale;
        }
    }

    // Write column back (carry / next step).
    #pragma unroll
    for (uint j = 0u; j < GDN_SG_NSG; ++j) {
        const uint is = tx * GDN_SG_NSG + j;
        s_mat[is * n + col] = ls[j];
    }
}

// ── Selective scan (Mamba SSM, f32) ─────────────────────────────────
// One thread per (batch, channel); each thread owns a private state
// vector of size n (n ≤ SSM_MAX_N) and scans sequentially over seq.
// Inputs (float indices): x, delta [b,s,h]; a [h,n]; b, c [b,s,n].
// Output [b,s,h]. Matches `execute_selective_scan_f32` on CPU:
//   h[t] = exp(Δ·A)·h[t-1] + Δ·B·x;   y[t] = Σ_n C·h[t]
#define SSM_MAX_N 128u
kernel void selective_scan(
    device float* arena       [[buffer(0)]],
    constant uint& x_off      [[buffer(1)]],
    constant uint& delta_off  [[buffer(2)]],
    constant uint& a_off      [[buffer(3)]],
    constant uint& b_off      [[buffer(4)]],
    constant uint& c_off      [[buffer(5)]],
    constant uint& dst_off    [[buffer(6)]],
    constant uint4& dims      [[buffer(7)]], // batch, seq, hidden, n
    uint gid [[thread_position_in_grid]]
) {
    uint b = dims.x, s = dims.y, h = dims.z, n = dims.w;
    if (n > SSM_MAX_N || gid >= b * h) return;

    uint bi = gid / h;
    uint ci = gid % h;

    float state[SSM_MAX_N];
    for (uint i = 0; i < n; ++i) {
        state[i] = 0.0f;
    }

    // a[ci, :] is constant across the sequence for this channel.
    uint a_base = a_off + ci * n;

    for (uint si = 0; si < s; ++si) {
        uint bsh = bi * s * h + si * h + ci;   // x/delta/out element offset
        uint bsn = (bi * s + si) * n;          // b/c row base
        float d = arena[delta_off + bsh];
        float xv = arena[x_off + bsh];
        float acc = 0.0f;
        for (uint ni = 0; ni < n; ++ni) {
            float da = exp(d * arena[a_base + ni]);
            float st = da * state[ni] + d * arena[b_off + bsn + ni] * xv;
            state[ni] = st;
            acc += arena[c_off + bsn + ni] * st;
        }
        arena[dst_off + bsh] = acc;
    }
}

// LSTM (gate order i, f, g, o; single merged bias). Dispatched once per (layer,
// direction) by `encode_lstm`, which loops layers×dirs and ping-pongs
// intermediate layer outputs through an in-arena scratch region (x_off/dst_off
// are absolute arena word offsets). One threadgroup per batch item; thread `k`
// owns hidden unit `k`, keeps `c[k]` in a register, shares `h_prev` in
// threadgroup memory. `h0_off`/`c0_off` (0 → 0) seed hidden/cell state; `more` =
// (h0_off, out_width, dir_off, reverse). Single-layer/unidir/no-carry reduces to
// the plain kernel. Matches `execute_lstm_f32` on CPU. hidden ≤ LSTM_MAX_H.
#define LSTM_MAX_H 1024u
kernel void lstm(
    device float* arena      [[buffer(0)]],
    constant uint& x_off     [[buffer(1)]],
    constant uint& wih_off   [[buffer(2)]],
    constant uint& whh_off   [[buffer(3)]],
    constant uint& bias_off  [[buffer(4)]],
    constant uint& dst_off   [[buffer(5)]],
    constant uint4& dims     [[buffer(6)]], // batch, seq, in_l, hidden
    constant uint4& more     [[buffer(7)]], // h0_off, out_width, dir_off, reverse
    constant uint& c0_off    [[buffer(8)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint b = dims.x, s = dims.y, in_sz = dims.z, h = dims.w;
    if (h > LSTM_MAX_H || gid >= b || tid >= h) return;
    uint bi = gid;
    uint k = tid;
    uint h0_off = more.x, out_width = more.y, dir_off = more.z, reverse = more.w;

    threadgroup float h_sh[LSTM_MAX_H];
    h_sh[k] = (h0_off != 0u) ? arena[h0_off + bi * h + k] : 0.0f;
    float c_k = (c0_off != 0u) ? arena[c0_off + bi * h + k] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = 0; step < s; ++step) {
        uint t = (reverse != 0u) ? (s - 1u - step) : step;
        uint x_base = x_off + (bi * s + t) * in_sz;
        // Gate pre-activations for hidden unit k: rows i=k, f=h+k, g=2h+k, o=3h+k.
        float z[4];
        for (uint gate = 0u; gate < 4u; ++gate) {
            uint r = gate * h + k;
            float acc = arena[bias_off + r];
            uint wih_row = wih_off + r * in_sz;
            for (uint j = 0u; j < in_sz; ++j) {
                acc += arena[wih_row + j] * arena[x_base + j];
            }
            uint whh_row = whh_off + r * h;
            for (uint j = 0u; j < h; ++j) {
                acc += arena[whh_row + j] * h_sh[j];
            }
            z[gate] = acc;
        }
        float i_g = 1.0f / (1.0f + exp(-z[0]));
        float f_g = 1.0f / (1.0f + exp(-z[1]));
        float g_g = tanh(clamp(z[2], -15.0f, 15.0f));
        float o_g = 1.0f / (1.0f + exp(-z[3]));
        c_k = f_g * c_k + i_g * g_g;
        float h_k = o_g * tanh(c_k);
        // Finish reading h_prev across all threads before overwriting it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h_sh[k] = h_k;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        arena[dst_off + (bi * s + t) * out_width + dir_off + k] = h_k;
    }
}

// Single-layer unidirectional GRU (gate order r, z, n; linear_before_reset=1;
// separate b_ih/b_hh; h0 = 0). One threadgroup per batch item; thread `k` owns
// hidden unit `k`. Matches `execute_gru_f32` on CPU.
#define GRU_MAX_H 1024u
// GRU dispatched once per (layer, direction) — the caller (`encode_gru`) loops
// layers×dirs, ping-ponging intermediate layer outputs through an in-arena
// scratch region (x_off/dst_off are absolute arena word offsets, possibly into
// scratch). `dims.z` = in_l (input_size for layer 0, else dirs·hidden). `more`
// carries h0_off (0 → h0=0, else seed from arena[h0_off+bi·h+k]), out_width =
// dirs·hidden, dir_off = dir·hidden (this dir owns [dir_off,dir_off+h)), and
// reverse (walk the sequence backwards). Single-layer/unidir/no-carry reduces to
// the original kernel (h0_off=0, out_width=h, dir_off=0, reverse=0).
kernel void gru(
    device float* arena      [[buffer(0)]],
    constant uint& x_off     [[buffer(1)]],
    constant uint& wih_off   [[buffer(2)]],
    constant uint& whh_off   [[buffer(3)]],
    constant uint& bih_off   [[buffer(4)]],
    constant uint& bhh_off   [[buffer(5)]],
    constant uint& dst_off   [[buffer(6)]],
    constant uint4& dims     [[buffer(7)]], // batch, seq, in_l, hidden
    constant uint4& more     [[buffer(8)]], // h0_off, out_width, dir_off, reverse
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint b = dims.x, s = dims.y, in_sz = dims.z, h = dims.w;
    if (h > GRU_MAX_H || gid >= b || tid >= h) return;
    uint bi = gid, k = tid;
    uint h0_off = more.x, out_width = more.y, dir_off = more.z, reverse = more.w;

    threadgroup float h_sh[GRU_MAX_H];
    h_sh[k] = (h0_off != 0u) ? arena[h0_off + bi * h + k] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = 0; step < s; ++step) {
        uint t = (reverse != 0u) ? (s - 1u - step) : step;
        uint x_base = x_off + (bi * s + t) * in_sz;
        // Gate rows r=k, z=h+k, n=2h+k. Input and hidden parts kept separate
        // because the reset gate multiplies the hidden term after its bias.
        float xi[3], hi[3];
        for (uint g = 0u; g < 3u; ++g) {
            uint r = g * h + k;
            float ax = arena[bih_off + r];
            uint wih_row = wih_off + r * in_sz;
            for (uint j = 0u; j < in_sz; ++j) {
                ax += arena[wih_row + j] * arena[x_base + j];
            }
            float ah = arena[bhh_off + r];
            uint whh_row = whh_off + r * h;
            for (uint j = 0u; j < h; ++j) {
                ah += arena[whh_row + j] * h_sh[j];
            }
            xi[g] = ax;
            hi[g] = ah;
        }
        float rg = 1.0f / (1.0f + exp(-(xi[0] + hi[0])));
        float zg = 1.0f / (1.0f + exp(-(xi[1] + hi[1])));
        float ng = tanh(clamp(xi[2] + rg * hi[2], -15.0f, 15.0f));
        float h_k = (1.0f - zg) * ng + zg * h_sh[k];
        // Finish reading h_prev across all threads before overwriting.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h_sh[k] = h_k;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        arena[dst_off + (bi * s + t) * out_width + dir_off + k] = h_k;
    }
}

// Elman RNN (`relu_flag` ? relu : tanh). Dispatched once per (layer, direction)
// by `encode_rnn` — same in-arena scratch ping-pong and `dims.z`=in_l / `more`
// convention as `gru` above. Single-layer/unidir/no-carry reduces to the
// original kernel. Matches `execute_rnn_f32` on CPU.
#define RNN_MAX_H 1024u
kernel void rnn(
    device float* arena       [[buffer(0)]],
    constant uint& x_off      [[buffer(1)]],
    constant uint& wih_off    [[buffer(2)]],
    constant uint& whh_off    [[buffer(3)]],
    constant uint& bias_off   [[buffer(4)]],
    constant uint& dst_off    [[buffer(5)]],
    constant uint4& dims      [[buffer(6)]], // batch, seq, in_l, hidden
    constant uint& relu_flag  [[buffer(7)]],
    constant uint4& more      [[buffer(8)]], // h0_off, out_width, dir_off, reverse
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint b = dims.x, s = dims.y, in_sz = dims.z, h = dims.w;
    if (h > RNN_MAX_H || gid >= b || tid >= h) return;
    uint bi = gid, k = tid;
    uint h0_off = more.x, out_width = more.y, dir_off = more.z, reverse = more.w;

    threadgroup float h_sh[RNN_MAX_H];
    h_sh[k] = (h0_off != 0u) ? arena[h0_off + bi * h + k] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = 0; step < s; ++step) {
        uint t = (reverse != 0u) ? (s - 1u - step) : step;
        uint x_base = x_off + (bi * s + t) * in_sz;
        float acc = arena[bias_off + k];
        uint wih_row = wih_off + k * in_sz;
        for (uint j = 0u; j < in_sz; ++j) {
            acc += arena[wih_row + j] * arena[x_base + j];
        }
        uint whh_row = whh_off + k * h;
        for (uint j = 0u; j < h; ++j) {
            acc += arena[whh_row + j] * h_sh[j];
        }
        float h_k = relu_flag != 0u ? fmax(acc, 0.0f) : tanh(acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h_sh[k] = h_k;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        arena[dst_off + (bi * s + t) * out_width + dir_off + k] = h_k;
    }
}

// Mamba-2 / SSD scalar-decay scan. One thread per (batch, head, head_dim_pos);
// each owns a private N-state vector and scans the sequence. Matches
// `execute_mamba2_f32` on CPU. n ≤ MAMBA2_MAX_N.
#define MAMBA2_MAX_N 128u
kernel void mamba2(
    device float* arena      [[buffer(0)]],
    constant uint& x_off     [[buffer(1)]],
    constant uint& dt_off    [[buffer(2)]],
    constant uint& a_off     [[buffer(3)]],
    constant uint& b_off     [[buffer(4)]],
    constant uint& c_off     [[buffer(5)]],
    constant uint& dst_off   [[buffer(6)]],
    constant uint4& dims     [[buffer(7)]], // batch, seq, heads, (head_dim<<16 | state_size)
    uint gid [[thread_position_in_grid]]
) {
    uint bn = dims.x, s = dims.y, hh = dims.z;
    uint p = dims.w >> 16, n = dims.w & 0xffffu;
    if (n > MAMBA2_MAX_N || gid >= bn * hh * p) return;

    uint pi = gid % p;
    uint hi = (gid / p) % hh;
    uint bi = gid / (p * hh);

    float state[MAMBA2_MAX_N];
    for (uint i = 0u; i < n; ++i) {
        state[i] = 0.0f;
    }
    float ah = arena[a_off + hi];

    for (uint t = 0u; t < s; ++t) {
        uint bsh = (bi * s + t) * hh + hi;
        float dt_t = arena[dt_off + bsh];
        float da = exp(dt_t * ah);
        float dtx = dt_t * arena[x_off + bsh * p + pi];
        uint bc = bsh * n;
        float acc = 0.0f;
        for (uint ni = 0u; ni < n; ++ni) {
            float st = da * state[ni] + dtx * arena[b_off + bc + ni];
            state[ni] = st;
            acc += st * arena[c_off + bc + ni];
        }
        arena[dst_off + bsh * p + pi] = acc;
    }
}

// RMSNorm backward (wrt: 0=dx, 1=dgamma, 2=dbeta). One threadgroup per row.
kernel void rms_norm_bwd(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device const float* dy [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant uint& wrt [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    if (wrt != 0u) return;
    threadgroup float partial[256];
    float local_dot = 0.0f;
    for (uint i = tid; i < inner; i += tsize) {
        float xv = x[row * inner + i];
        float gv = gamma[i];
        float dyv = dy[row * inner + i];
        local_dot += dyv * gv * xv;
    }
    partial[tid] = local_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dot = partial[0] / float(inner);
    float local_ss = 0.0f;
    for (uint i = tid; i < inner; i += tsize) {
        float xv = x[row * inner + i];
        local_ss += xv * xv;
    }
    partial[tid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_r = rsqrt(partial[0] / float(inner) + eps);
    // Cross term is inv_r³ (= inv_r2·inv_r below), NOT inv_r⁴: outer ·inv_r already supplies
    // one factor, so the inner term uses inv_r2. The prior inv_r3-then-·inv_r was a stray 1/r.
    float inv_r2 = inv_r * inv_r;
    for (uint i = tid; i < inner; i += tsize) {
        float xv = x[row * inner + i];
        float gv = gamma[i];
        float dyv = dy[row * inner + i];
        float term = gv * dyv - xv * dot * inv_r2;
        out[row * inner + i] = term * inv_r;
    }
}

kernel void rms_norm_bwd_param(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device const float* dy [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant uint& wrt [[buffer(8)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0u) return;
    for (uint i = 0; i < inner; ++i) out[i] = 0.0f;
    for (uint row = 0; row < rows; ++row) {
        float sumsq = 0.0f;
        for (uint i = 0; i < inner; ++i) {
            float xv = x[row * inner + i];
            sumsq += xv * xv;
        }
        float inv_r = rsqrt(sumsq / float(inner) + eps);
        if (wrt == 1u) {
            for (uint i = 0; i < inner; ++i) {
                out[i] += dy[row * inner + i] * x[row * inner + i] * inv_r;
            }
        } else {
            for (uint i = 0; i < inner; ++i) {
                out[i] += dy[row * inner + i];
            }
        }
    }
}

// Per-row RMS inverse scale — scratch for parallel dgamma/dbeta.
kernel void rms_norm_bwd_inv_r_f32(
    device const float* x [[buffer(0)]],
    device float* inv_r [[buffer(1)]],
    constant uint& inner [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    float sumsq = 0.0f;
    for (uint i = 0; i < inner; ++i) {
        float xv = x[row * inner + i];
        sumsq += xv * xv;
    }
    inv_r[row] = rsqrt(sumsq / float(inner) + eps);
}

// Reduce dy·x·inv_r (gamma) or dy (beta) across rows — one thread per param.
kernel void rms_norm_bwd_param_reduce_f32(
    device const float* x [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device const float* inv_r [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    constant uint& wrt [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= inner) return;
    float acc = 0.0f;
    for (uint row = 0; row < rows; ++row) {
        if (wrt == 1u) {
            acc += dy[row * inner + i] * x[row * inner + i] * inv_r[row];
        } else {
            acc += dy[row * inner + i];
        }
    }
    out[i] = acc;
}

// LayerNorm backward (last-axis rows). Matches CPU `LayerNormBackward*`.
// One threadgroup per row; power-of-2 TG reduction.
kernel void layer_norm_bwd(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* dy [[buffer(2)]],
    device float* dx [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float partial_a[256];
    threadgroup float partial_b[256];
    float n_inv = 1.0f / float(inner);

    float local_sum = 0.0f;
    for (uint i = tid; i < inner; i += tsize) {
        local_sum += x[row * inner + i];
    }
    partial_a[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial_a[tid] += partial_a[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = partial_a[0] * n_inv;

    float local_var = 0.0f;
    for (uint i = tid; i < inner; i += tsize) {
        float d = x[row * inner + i] - mean;
        local_var += d * d;
    }
    partial_a[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial_a[tid] += partial_a[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(partial_a[0] * n_inv + eps);

    float local_sy = 0.0f;
    float local_sxh = 0.0f;
    for (uint i = tid; i < inner; i += tsize) {
        float xh = (x[row * inner + i] - mean) * inv_std;
        float sy = dy[row * inner + i] * gamma[i];
        local_sy += sy;
        local_sxh += sy * xh;
    }
    partial_a[tid] = local_sy;
    partial_b[tid] = local_sxh;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_a[tid] += partial_a[tid + stride];
            partial_b[tid] += partial_b[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m_sy = partial_a[0] * n_inv;
    float m_sxh = partial_b[0] * n_inv;

    for (uint i = tid; i < inner; i += tsize) {
        float xh = (x[row * inner + i] - mean) * inv_std;
        float sy = dy[row * inner + i] * gamma[i];
        dx[row * inner + i] = inv_std * (sy - m_sy - xh * m_sxh);
    }
}

// Serial dgamma: one thread walks all rows (small-row / no-scratch path).
kernel void layer_norm_bwd_gamma(
    device const float* x [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* dgamma [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0u) return;
    for (uint i = 0; i < inner; ++i) dgamma[i] = 0.0f;
    float n_inv = 1.0f / float(inner);
    for (uint row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (uint i = 0; i < inner; ++i) sum += x[row * inner + i];
        float mean = sum * n_inv;
        float var = 0.0f;
        for (uint i = 0; i < inner; ++i) {
            float d = x[row * inner + i] - mean;
            var += d * d;
        }
        float inv_std = rsqrt(var * n_inv + eps);
        for (uint i = 0; i < inner; ++i) {
            float xh = (x[row * inner + i] - mean) * inv_std;
            dgamma[i] += dy[row * inner + i] * xh;
        }
    }
}

// Per-row mean + inv_std into scratch[row*2 + {0,1}] for parallel dgamma.
kernel void layer_norm_bwd_stats_f32(
    device const float* x [[buffer(0)]],
    device float* stats [[buffer(1)]],
    constant uint& inner [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    float n_inv = 1.0f / float(inner);
    float sum = 0.0f;
    for (uint i = 0; i < inner; ++i) sum += x[row * inner + i];
    float mean = sum * n_inv;
    float var = 0.0f;
    for (uint i = 0; i < inner; ++i) {
        float d = x[row * inner + i] - mean;
        var += d * d;
    }
    stats[row * 2u + 0u] = mean;
    stats[row * 2u + 1u] = rsqrt(var * n_inv + eps);
}

// One thread per last-axis index: sum_r dy[r,i] * x_hat[r,i].
kernel void layer_norm_bwd_gamma_reduce_f32(
    device const float* x [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device const float* stats [[buffer(2)]],
    device float* dgamma [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= inner) return;
    float acc = 0.0f;
    for (uint row = 0; row < rows; ++row) {
        float mean = stats[row * 2u + 0u];
        float inv_std = stats[row * 2u + 1u];
        float xh = (x[row * inner + i] - mean) * inv_std;
        acc += dy[row * inner + i] * xh;
    }
    dgamma[i] = acc;
}

// SIMD-parallel variant: one 32-wide threadgroup (== 1 SIMD group) per column
// `i`; the 32 lanes split the `rows` reduction and combine with simd_sum. The
// scalar kernel above dispatches ONE thread per column, each serially summing
// all rows (batch·seq ≈ 1024) — low occupancy + a fully serial reduction. This
// adds 32× row parallelism (same win as `sdpa_simd`'s softmax).
kernel void layer_norm_bwd_gamma_reduce_simd(
    device const float* x [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device const float* stats [[buffer(2)]],
    device float* dgamma [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint i [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    if (i >= inner) return;
    float local = 0.0f;
    for (uint row = tid; row < rows; row += tsize) {
        float mean = stats[row * 2u + 0u];
        float inv_std = stats[row * 2u + 1u];
        float xh = (x[row * inner + i] - mean) * inv_std;
        local += dy[row * inner + i] * xh;
    }
    float total = simd_sum(local);
    if (tid == 0) {
        dgamma[i] = total;
    }
}

// GroupNorm (NCHW) backward — matches CPU `training_bwd::group_norm_backward_*`.
// One threadgroup per (batch, group); 256-wide reductions.
kernel void group_norm_bwd_input(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* dy [[buffer(2)]],
    device float* dx [[buffer(3)]],
    constant uint4& nchw [[buffer(4)]],
    constant uint& num_groups [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint ng [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    uint batch = nchw.x;
    uint c = nchw.y;
    uint h = nchw.z;
    uint w = nchw.w;
    if (ng >= batch * num_groups) return;
    uint n = ng / num_groups;
    uint g = ng % num_groups;
    uint cpg = c / num_groups;
    uint c0 = g * cpg;
    uint spatial = h * w;
    uint count = cpg * spatial;
    float n_inv = 1.0f / float(count);
    uint plane = c * spatial;
    uint b_base = n * plane;

    threadgroup float partial_a[256];
    threadgroup float partial_b[256];

    float local_sum = 0.0f;
    for (uint i = tid; i < count; i += tsize) {
        uint c_off = i / spatial;
        uint s = i % spatial;
        local_sum += x[b_base + (c0 + c_off) * spatial + s];
    }
    partial_a[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial_a[tid] += partial_a[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = partial_a[0] * n_inv;

    float local_var = 0.0f;
    for (uint i = tid; i < count; i += tsize) {
        uint c_off = i / spatial;
        uint s = i % spatial;
        float d = x[b_base + (c0 + c_off) * spatial + s] - mean;
        local_var += d * d;
    }
    partial_a[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial_a[tid] += partial_a[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(partial_a[0] * n_inv + eps);

    float local_sy = 0.0f;
    float local_sxh = 0.0f;
    for (uint i = tid; i < count; i += tsize) {
        uint c_off = i / spatial;
        uint s = i % spatial;
        uint gi = c0 + c_off;
        float xh = (x[b_base + gi * spatial + s] - mean) * inv_std;
        float sy = dy[b_base + gi * spatial + s] * gamma[gi];
        local_sy += sy;
        local_sxh += sy * xh;
    }
    partial_a[tid] = local_sy;
    partial_b[tid] = local_sxh;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_a[tid] += partial_a[tid + stride];
            partial_b[tid] += partial_b[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m_sy = partial_a[0] * n_inv;
    float m_sxh = partial_b[0] * n_inv;

    for (uint i = tid; i < count; i += tsize) {
        uint c_off = i / spatial;
        uint s = i % spatial;
        uint gi = c0 + c_off;
        float xh = (x[b_base + gi * spatial + s] - mean) * inv_std;
        float sy = dy[b_base + gi * spatial + s] * gamma[gi];
        dx[b_base + gi * spatial + s] = inv_std * (sy - m_sy - xh * m_sxh);
    }
}

kernel void group_norm_bwd_gamma(
    device const float* x [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* dgamma [[buffer(2)]],
    constant uint4& nchw [[buffer(3)]],
    constant uint& num_groups [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0u) return;
    uint n = nchw.x;
    uint c = nchw.y;
    uint h = nchw.z;
    uint w = nchw.w;
    uint spatial = h * w;
    uint plane = c * spatial;
    uint cpg = c / num_groups;
    float n_inv = 1.0f / float(cpg * spatial);
    for (uint ch = 0; ch < c; ++ch) dgamma[ch] = 0.0f;
    for (uint bn = 0; bn < n; ++bn) {
        uint b_base = bn * plane;
        for (uint g = 0; g < num_groups; ++g) {
            uint c0 = g * cpg;
            float mean = 0.0f;
            for (uint ci = 0; ci < cpg; ++ci) {
                uint base = b_base + (c0 + ci) * spatial;
                for (uint s = 0; s < spatial; ++s) mean += x[base + s];
            }
            mean *= n_inv;
            float var = 0.0f;
            for (uint ci = 0; ci < cpg; ++ci) {
                uint base = b_base + (c0 + ci) * spatial;
                for (uint s = 0; s < spatial; ++s) {
                    float d = x[base + s] - mean;
                    var += d * d;
                }
            }
            float inv_std = rsqrt(var * n_inv + eps);
            for (uint ci = 0; ci < cpg; ++ci) {
                uint gi = c0 + ci;
                uint x_base = b_base + gi * spatial;
                uint dy_base = b_base + gi * spatial;
                float acc = dgamma[gi];
                for (uint s = 0; s < spatial; ++s) {
                    float xh = (x[x_base + s] - mean) * inv_std;
                    acc += dy[dy_base + s] * xh;
                }
                dgamma[gi] = acc;
            }
        }
    }
}

kernel void group_norm_bwd_beta(
    device const float* dy [[buffer(0)]],
    device float* dbeta [[buffer(1)]],
    constant uint4& nchw [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0u) return;
    uint n = nchw.x;
    uint c = nchw.y;
    uint h = nchw.z;
    uint w = nchw.w;
    uint spatial = h * w;
    uint plane = c * spatial;
    for (uint ch = 0; ch < c; ++ch) dbeta[ch] = 0.0f;
    for (uint bn = 0; bn < n; ++bn) {
        uint b_base = bn * plane;
        for (uint ch = 0; ch < c; ++ch) {
            uint dy_base = b_base + ch * spatial;
            float acc = dbeta[ch];
            for (uint s = 0; s < spatial; ++s) acc += dy[dy_base + s];
            dbeta[ch] = acc;
        }
    }
}

kernel void rope_bwd(
    device const float* dy [[buffer(0)]],
    device const float* cos [[buffer(1)]],
    device const float* sin [[buffer(2)]],
    device float* dx [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& seq [[buffer(5)]],
    constant uint& hidden [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& n_rot [[buffer(8)]],
    constant uint& cos_len [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint hi = gid.y;
    uint bs = gid.z;
    if (d >= head_dim) return;
    uint nh = hidden / head_dim;
    if (hi >= nh) return;
    if (bs >= batch * seq) return;
    uint bi = bs / seq;
    uint si = bs % seq;
    uint rot_half = n_rot / 2u;
    uint half_dh = head_dim / 2u;
    uint tab_off = (si * half_dh) % max(cos_len, 1u);
    uint dy_base = bi * seq * hidden + si * hidden + hi * head_dim;
    uint dx_base = dy_base;
    if (d < rot_half) {
        float y1 = dy[dy_base + d];
        float y2 = dy[dy_base + rot_half + d];
        float c = cos[tab_off + d];
        float s = sin[tab_off + d];
        dx[dx_base + d] = y1 * c + y2 * s;
        dx[dx_base + rot_half + d] = -y1 * s + y2 * c;
    } else if (d >= n_rot) {
        dx[dx_base + d] = dy[dy_base + d];
    }
}

kernel void cumsum_fwd(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& inner [[buffer(2)]],
    constant uint& exclusive [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]]
) {
    float acc = 0.0f;
    for (uint i = 0; i < inner; ++i) {
        if (exclusive != 0u) {
            dst[row * inner + i] = acc;
            acc += src[row * inner + i];
        } else {
            acc += src[row * inner + i];
            dst[row * inner + i] = acc;
        }
    }
}

// Cumulative product / maximum along the last axis (one threadgroup per row).
// is_max=1 runs a running max (identity -inf) else a running product (identity 1).
kernel void cum_scan(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& inner [[buffer(2)]],
    constant uint& exclusive [[buffer(3)]],
    constant uint& is_max [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]]
) {
    float acc = (is_max != 0u) ? (-INFINITY) : 1.0f;
    for (uint i = 0; i < inner; ++i) {
        float v = src[row * inner + i];
        if (exclusive != 0u) {
            dst[row * inner + i] = acc;
            acc = (is_max != 0u) ? fmax(acc, v) : (acc * v);
        } else {
            acc = (is_max != 0u) ? fmax(acc, v) : (acc * v);
            dst[row * inner + i] = acc;
        }
    }
}

kernel void cumsum_bwd(
    device const float* dy [[buffer(0)]],
    device float* dx [[buffer(1)]],
    constant uint& inner [[buffer(2)]],
    constant uint& exclusive [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]]
) {
    float suffix = 0.0f;
    for (int i = int(inner) - 1; i >= 0; --i) {
        uint ui = uint(i);
        if (exclusive != 0u) {
            dx[row * inner + ui] = suffix;
            suffix += dy[row * inner + ui];
        } else {
            suffix += dy[row * inner + ui];
            dx[row * inner + ui] = suffix;
        }
    }
}

// Single im2col element for conv weight backward GEMM (B[k_idx, n_col]).
inline float conv_bwd_im2col_elem(
    device const float* x,
    uint k_idx,
    uint n_col,
    uint c_in,
    uint h,
    uint w_in,
    uint h_out,
    uint w_out,
    uint kh,
    uint kw,
    uint sh,
    uint sw,
    uint ph,
    uint pw,
    uint dh,
    uint dw_dil
) {
    uint ho = k_idx / w_out;
    uint wo = k_idx % w_out;
    uint rem = n_col;
    uint ci = rem / (kh * kw);
    rem = rem % (kh * kw);
    uint ki = rem / kw;
    uint kj = rem % kw;
    int hi = (int)(ho * sh + ki * dh) - (int)ph;
    int wi = (int)(wo * sw + kj * dw_dil) - (int)pw;
    if (hi < 0 || wi < 0 || hi >= (int)h || wi >= (int)w_in) {
        return 0.0f;
    }
    return x[(ci * h + (uint)hi) * w_in + (uint)wi];
}

// dw = dy @ im2col(x) — 8×8 simdgroup tiles, B generated on the fly (no scratch).
kernel void conv2d_bwd_weight_gemm(
    device const float* dy [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* dw [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint4& nchw [[buffer(6)]],
    constant uint4& out_dims [[buffer(7)]],
    constant uint4& kshape [[buffer(8)]],
    constant uint4& padd [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint slid [[thread_index_in_threadgroup]]
) {
    uint row_base = tgid.y * 8;
    uint col_base = tgid.x * 8;
    if (row_base >= M || col_base >= N) return;

    uint c_in = nchw.x;
    uint h = nchw.y;
    uint w_in = nchw.z;
    uint h_out = out_dims.y;
    uint w_out = out_dims.z;
    uint kh = kshape.x;
    uint kw = kshape.y;
    uint sh = kshape.z;
    uint sw = kshape.w;
    uint ph = padd.x;
    uint pw = padd.y;
    uint dh = padd.z;
    uint dw_dil = padd.w;

    threadgroup float B_tg[64];
    simdgroup_float8x8 a, b, c;
    c = simdgroup_float8x8(0.0f);

    for (uint k0 = 0; k0 < K; k0 += 8) {
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 32 + slid;
            if (idx < 64) {
                uint br = idx / 8;
                uint bc = idx % 8;
                uint k_idx = k0 + br;
                uint n_col = col_base + bc;
                B_tg[idx] = (k_idx < K && n_col < N)
                    ? conv_bwd_im2col_elem(
                          x, k_idx, n_col, c_in, h, w_in, h_out, w_out, kh, kw, sh, sw, ph, pw,
                          dh, dw_dil)
                    : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(a, dy + row_base * K + k0, K);
        simdgroup_load(b, B_tg, 8);
        simdgroup_multiply_accumulate(c, a, b, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c, dw + row_base * N + col_base, N);
}

// dw = dy @ im2col(x) — 32×32 threadgroup tiles (requires M,K,N % 32 == 0).
kernel void conv2d_bwd_weight_gemm_4x4(
    device const float* dy [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* dw [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint4& nchw [[buffer(6)]],
    constant uint4& out_dims [[buffer(7)]],
    constant uint4& kshape [[buffer(8)]],
    constant uint4& padd [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint slid [[thread_index_in_simdgroup]]
) {
    uint sg_row = sgid / 4;
    uint sg_col = sgid % 4;
    uint tg_row_base = tgid.y * 32;
    uint tg_col_base = tgid.x * 32;

    uint c_in = nchw.x;
    uint h = nchw.y;
    uint w_in = nchw.z;
    uint h_out = out_dims.y;
    uint w_out = out_dims.z;
    uint kh = kshape.x;
    uint kw = kshape.y;
    uint sh = kshape.z;
    uint sw = kshape.w;
    uint ph = padd.x;
    uint pw = padd.y;
    uint dh = padd.z;
    uint dw_dil = padd.w;

    threadgroup float A_tg[32 * 32];
    threadgroup float B_tg[32 * 32];
    simdgroup_float8x8 a, b, c;
    c = simdgroup_float8x8(0.0f);

    for (uint kk = 0; kk < K; kk += 32) {
        uint linear = sgid * 32 + slid;
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 512 + linear;
            uint ar = idx / 32;
            uint ac = idx % 32;
            A_tg[idx] = dy[(tg_row_base + ar) * K + (kk + ac)];
        }
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 512 + linear;
            uint br = idx / 32;
            uint bc = idx % 32;
            uint k_idx = kk + br;
            uint n_col = tg_col_base + bc;
            B_tg[idx] = conv_bwd_im2col_elem(
                x, k_idx, n_col, c_in, h, w_in, h_out, w_out, kh, kw, sh, sw, ph, pw, dh,
                dw_dil);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_inner = 0; k_inner < 32; k_inner += 8) {
            simdgroup_load(a, &A_tg[sg_row * 8 * 32 + k_inner], 32);
            simdgroup_load(b, &B_tg[k_inner * 32 + sg_col * 8], 32);
            simdgroup_multiply_accumulate(c, a, b, c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint out_row = tg_row_base + sg_row * 8;
    uint out_col = tg_col_base + sg_col * 8;
    simdgroup_store(c, &dw[out_row * N + out_col], N);
}

// Fast im2col when W_in = W_out = 1 (Voxtral codec `[1,C,T,1]` slices).
kernel void im2col_group_w1(
    device const float* x [[buffer(0)]],
    device float* col [[buffer(1)]],
    constant uint4& nchw [[buffer(2)]],     // [C_in/g, H, 1, unused]
    constant uint4& out_dims [[buffer(3)]], // [unused, H_out, 1, unused]
    constant uint4& kshape [[buffer(4)]],
    constant uint4& padd [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint h_out = out_dims.y;
    uint c_in = nchw.x;
    uint h = nchw.y;
    uint kh = kshape.x;
    uint kw = kshape.y;
    uint sh = kshape.z;
    uint ph = padd.x;
    uint dh = padd.z;
    uint n_dim = c_in * kh * kw;
    uint k_dim = h_out;
    uint idx = gid;
    if (idx >= n_dim * k_dim) return;
    uint row = idx / k_dim;
    uint ho = idx % k_dim;
    uint rem = row;
    uint ci = rem / (kh * kw);
    rem = rem % (kh * kw);
    uint ki = rem / kw;
    int hi = (int)(ho * sh + ki * dh) - (int)ph;
    col[ho * n_dim + row] = (hi < 0 || hi >= (int)h)
        ? 0.0f
        : x[ci * h + (uint)hi];
}

// im2col for one (batch, group) slice — layout matches `rlx_cpu::conv_bwd` /
// `[n_dim, k_dim]` row-major with `n_dim = C_in/g · kH · kW`, `k_dim = H_out · W_out`.
kernel void im2col_group(
    device const float* x [[buffer(0)]],
    device float* col [[buffer(1)]],
    constant uint4& nchw [[buffer(2)]],     // [C_in/g, H, W, unused] (group slice)
    constant uint4& out_dims [[buffer(3)]], // [unused, H_out, W_out, unused]
    constant uint4& kshape [[buffer(4)]],
    constant uint4& padd [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint h_out = out_dims.y;
    uint w_out = out_dims.z;
    uint c_in = nchw.x;
    uint h = nchw.y;
    uint w = nchw.z;
    uint kh = kshape.x;
    uint kw = kshape.y;
    uint sh = kshape.z;
    uint sw = kshape.w;
    uint ph = padd.x;
    uint pw = padd.y;
    uint dh = padd.z;
    uint dw_dil = padd.w;
    uint n_dim = c_in * kh * kw;
    uint k_dim = h_out * w_out;
    uint idx = gid.x;
    if (idx >= n_dim * k_dim) return;
    uint row = idx / k_dim;
    uint k_idx = idx % k_dim;
    uint ho = k_idx / w_out;
    uint wo = k_idx % w_out;
    uint rem = row;
    uint ci = rem / (kh * kw);
    rem = rem % (kh * kw);
    uint ki = rem / kw;
    uint kj = rem % kw;
    int hi = (int)(ho * sh + ki * dh) - (int)ph;
    int wi = (int)(wo * sw + kj * dw_dil) - (int)pw;
    col[k_idx * n_dim + row] = (hi < 0 || wi < 0 || hi >= (int)h || wi >= (int)w)
        ? 0.0f
        : x[(ci * h + (uint)hi) * w + (uint)wi];
}

// ── Attention backward (recompute scores + softmax) ─────────────────────

kernel void attn_bwd_scores_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& sq [[buffer(3)]],
    constant uint& sk [[buffer(4)]],
    constant uint& hs [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    constant uint& mask_kind [[buffer(8)]],
    constant uint& window [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint qi = gid.y;
    uint ki = gid.x;
    if (qi >= sq || ki >= sk) return;
    float dot = 0.0f;
    for (uint d = 0; d < head_dim; ++d) {
        dot += q[qi * hs + d] * k[ki * hs + d];
    }
    float s = dot * scale;
    if (mask_kind == 1u) {
        if (ki > qi) s = -1e4f;
    } else if (mask_kind == 3u) {
        uint lo = qi > window ? qi - window : 0u;
        if (ki < lo || ki > qi) s = -1e4f;
    }
    scores[qi * sk + ki] = s;
}

kernel void attn_bwd_dp_f32(
    device const float* dy [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device float* dp [[buffer(2)]],
    constant uint& sq [[buffer(3)]],
    constant uint& sk [[buffer(4)]],
    constant uint& hs [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint qi = gid.y;
    uint ki = gid.x;
    if (qi >= sq || ki >= sk) return;
    float acc = 0.0f;
    for (uint d = 0; d < head_dim; ++d) {
        acc += dy[qi * hs + d] * v[ki * hs + d];
    }
    dp[qi * sk + ki] = acc;
}

kernel void attn_bwd_ds_f32(
    device const float* scores [[buffer(0)]],
    device const float* dp [[buffer(1)]],
    device float* ds [[buffer(2)]],
    constant uint& sq [[buffer(3)]],
    constant uint& sk [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    if (row >= sq) return;
    float row_sum = 0.0f;
    for (uint ki = tid; ki < sk; ki += tsize) {
        row_sum += scores[row * sk + ki] * dp[row * sk + ki];
    }
    threadgroup float partial[256];
    partial[tid] = row_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum = partial[0];
    for (uint ki = tid; ki < sk; ki += tsize) {
        uint idx = row * sk + ki;
        float p = scores[idx];
        ds[idx] = p * (dp[idx] - sum) * scale;
    }
}

kernel void attn_bwd_dv_f32(
    device const float* scores [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& sq [[buffer(3)]],
    constant uint& sk [[buffer(4)]],
    constant uint& hs [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint ki = gid.y;
    uint d = gid.x;
    if (ki >= sk || d >= head_dim) return;
    float acc = 0.0f;
    for (uint qi = 0; qi < sq; ++qi) {
        acc += scores[qi * sk + ki] * dy[qi * hs + d];
    }
    out[ki * hs + d] = acc;
}

kernel void attn_bwd_dq_f32(
    device const float* ds [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& sq [[buffer(3)]],
    constant uint& sk [[buffer(4)]],
    constant uint& hs [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint qi = gid.y;
    uint d = gid.x;
    if (qi >= sq || d >= head_dim) return;
    float acc = 0.0f;
    for (uint ki = 0; ki < sk; ++ki) {
        acc += ds[qi * sk + ki] * k[ki * hs + d];
    }
    out[qi * hs + d] = acc;
}

kernel void attn_bwd_dk_f32(
    device const float* ds [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& sq [[buffer(3)]],
    constant uint& sk [[buffer(4)]],
    constant uint& hs [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint ki = gid.y;
    uint d = gid.x;
    if (ki >= sk || d >= head_dim) return;
    float acc = 0.0f;
    for (uint qi = 0; qi < sq; ++qi) {
        acc += ds[qi * sk + ki] * q[qi * hs + d];
    }
    out[ki * hs + d] = acc;
}

// ── Batched attention-backward kernels ────────────────────────────────────
// One 3D dispatch (grid.z = head-slot) processes MANY (batch,head) pairs in
// parallel instead of the per-(b,h) host loop, which serialized on the shared
// scratch (Metal hazard-tracks at buffer granularity). Per-slot pointers are
// computed in-kernel from `tile_base + slot`; scratch is laid out as
// `[tile][sq][sk]` (scores/dp/ds each a `tile*ss` region). `hs = heads*head_dim`.
kernel void attn_bwd_scores_batched_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& sq [[buffer(3)]],
    constant uint& sk [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    constant uint& mask_kind [[buffer(8)]],
    constant uint& window [[buffer(9)]],
    constant uint& tile_base [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint ki = gid.x, qi = gid.y, slot = gid.z;
    if (qi >= sq || ki >= sk) return;
    // Causal / sliding-window: masked pairs become 0 after softmax
    // (exp(-1e4)=0 in f32), so write -1e4 and SKIP the dot — ~half the work.
    bool masked = (mask_kind == 1u && ki > qi)
        || (mask_kind == 3u && (ki > qi || (qi > window && ki < qi - window)));
    if (masked) {
        scores[slot * sq * sk + qi * sk + ki] = -1e4f;
        return;
    }
    uint hs = heads * head_dim;
    uint gp = tile_base + slot;
    uint bi = gp / heads, hi = gp % heads;
    uint qbase = bi * sq * hs + hi * head_dim;
    uint kbase = bi * sk * hs + hi * head_dim;
    float dot = 0.0f;
    for (uint d = 0; d < head_dim; ++d) {
        dot += q[qbase + qi * hs + d] * k[kbase + ki * hs + d];
    }
    scores[slot * sq * sk + qi * sk + ki] = dot * scale;
}

kernel void attn_bwd_dp_batched_f32(
    device const float* dy [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device float* dp [[buffer(2)]],
    constant uint& sq [[buffer(3)]],
    constant uint& sk [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& tile_base [[buffer(7)]],
    constant uint& mask_kind [[buffer(8)]],
    constant uint& window [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint ki = gid.x, qi = gid.y, slot = gid.z;
    if (qi >= sq || ki >= sk) return;
    // Masked pairs contribute 0 to ds (scores=0); write 0, skip the dot.
    bool masked = (mask_kind == 1u && ki > qi)
        || (mask_kind == 3u && (ki > qi || (qi > window && ki < qi - window)));
    if (masked) {
        dp[slot * sq * sk + qi * sk + ki] = 0.0f;
        return;
    }
    uint hs = heads * head_dim;
    uint gp = tile_base + slot;
    uint bi = gp / heads, hi = gp % heads;
    uint qbase = bi * sq * hs + hi * head_dim;
    uint kbase = bi * sk * hs + hi * head_dim;
    float acc = 0.0f;
    for (uint d = 0; d < head_dim; ++d) {
        acc += dy[qbase + qi * hs + d] * v[kbase + ki * hs + d];
    }
    dp[slot * sq * sk + qi * sk + ki] = acc;
}

// Per-row softmax-jacobian; one threadgroup per (slot,row). `row = slot*sq + lr`.
kernel void attn_bwd_ds_batched_f32(
    device const float* scores [[buffer(0)]],
    device const float* dp [[buffer(1)]],
    device float* ds [[buffer(2)]],
    constant uint& sq [[buffer(3)]],
    constant uint& sk [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    constant uint& mask_kind [[buffer(6)]],
    constant uint& window [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    uint slot = row / sq;
    uint lr = row % sq;
    uint base = slot * sq * sk + lr * sk;
    // Only the causal/window band contributes (masked scores are 0); bound the
    // reduction AND the write to `[ki_lo, ki_hi]`. dq/dk are bounded the same
    // way, so they never read the un-written masked entries.
    uint ki_lo = 0u;
    uint ki_hi = sk - 1u;
    if (mask_kind == 1u) {
        ki_hi = min(lr, sk - 1u);
    } else if (mask_kind == 3u) {
        ki_hi = min(lr, sk - 1u);
        ki_lo = lr > window ? lr - window : 0u;
    }
    float row_sum = 0.0f;
    for (uint ki = ki_lo + tid; ki <= ki_hi; ki += tsize) {
        row_sum += scores[base + ki] * dp[base + ki];
    }
    threadgroup float partial[256];
    partial[tid] = row_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum = partial[0];
    for (uint ki = ki_lo + tid; ki <= ki_hi; ki += tsize) {
        uint idx = base + ki;
        ds[idx] = scores[idx] * (dp[idx] - sum) * scale;
    }
}

kernel void attn_bwd_dv_batched_f32(
    device const float* scores [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& sq [[buffer(3)]],
    constant uint& sk [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& tile_base [[buffer(7)]],
    constant uint& mask_kind [[buffer(8)]],
    constant uint& window [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x, ki = gid.y, slot = gid.z;
    if (ki >= sk || d >= head_dim) return;
    // scores[qi,ki] is 0 unless ki is in qi's causal/window band ⇒ qi ≥ ki
    // (and qi ≤ ki+window for sliding). Bound the qi accumulation to that band.
    uint qi_lo = 0u;
    uint qi_hi = sq - 1u;
    if (mask_kind == 1u) {
        qi_lo = ki;
    } else if (mask_kind == 3u) {
        qi_lo = ki;
        qi_hi = min(sq - 1u, ki + window);
    }
    uint hs = heads * head_dim;
    uint gp = tile_base + slot;
    uint bi = gp / heads, hi = gp % heads;
    uint qbase = bi * sq * hs + hi * head_dim;
    uint kbase = bi * sk * hs + hi * head_dim;
    float acc = 0.0f;
    for (uint qi = qi_lo; qi <= qi_hi; ++qi) {
        acc += scores[slot * sq * sk + qi * sk + ki] * dy[qbase + qi * hs + d];
    }
    out[kbase + ki * hs + d] = acc;
}

kernel void attn_bwd_dq_batched_f32(
    device const float* ds [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& sq [[buffer(3)]],
    constant uint& sk [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& tile_base [[buffer(7)]],
    constant uint& mask_kind [[buffer(8)]],
    constant uint& window [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x, qi = gid.y, slot = gid.z;
    if (qi >= sq || d >= head_dim) return;
    // ds[qi,ki] is 0 outside qi's causal/window band ⇒ ki ≤ qi (and ki ≥
    // qi-window for sliding). Bound the ki accumulation to that band.
    uint ki_lo = 0u;
    uint ki_hi = sk - 1u;
    if (mask_kind == 1u) {
        ki_hi = min(qi, sk - 1u);
    } else if (mask_kind == 3u) {
        ki_hi = min(qi, sk - 1u);
        ki_lo = qi > window ? qi - window : 0u;
    }
    uint hs = heads * head_dim;
    uint gp = tile_base + slot;
    uint bi = gp / heads, hi = gp % heads;
    uint qbase = bi * sq * hs + hi * head_dim;
    uint kbase = bi * sk * hs + hi * head_dim;
    float acc = 0.0f;
    for (uint ki = ki_lo; ki <= ki_hi; ++ki) {
        acc += ds[slot * sq * sk + qi * sk + ki] * k[kbase + ki * hs + d];
    }
    out[qbase + qi * hs + d] = acc;
}

kernel void attn_bwd_dk_batched_f32(
    device const float* ds [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& sq [[buffer(3)]],
    constant uint& sk [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& tile_base [[buffer(7)]],
    constant uint& mask_kind [[buffer(8)]],
    constant uint& window [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x, ki = gid.y, slot = gid.z;
    if (ki >= sk || d >= head_dim) return;
    // ds[qi,ki] is 0 unless ki is in qi's causal/window band ⇒ qi ≥ ki (and
    // qi ≤ ki+window for sliding). Bound the qi accumulation to that band.
    uint qi_lo = 0u;
    uint qi_hi = sq - 1u;
    if (mask_kind == 1u) {
        qi_lo = ki;
    } else if (mask_kind == 3u) {
        qi_lo = ki;
        qi_hi = min(sq - 1u, ki + window);
    }
    uint hs = heads * head_dim;
    uint gp = tile_base + slot;
    uint bi = gp / heads, hi = gp % heads;
    uint qbase = bi * sq * hs + hi * head_dim;
    uint kbase = bi * sk * hs + hi * head_dim;
    float acc = 0.0f;
    for (uint qi = qi_lo; qi <= qi_hi; ++qi) {
        acc += ds[slot * sq * sk + qi * sk + ki] * q[qbase + qi * hs + d];
    }
    out[kbase + ki * hs + d] = acc;
}

// ── Fused flash-attention backward (single kernel) ──────────────────────────
// One threadgroup per (query-tile of Br rows, head, batch). Recomputes S=QKᵀ·scale
// and the row softmax P entirely in THREADGROUP memory (never written to device —
// no scores/dp/ds S² scratch buffers), then:
//   dQ[qi] = Σ_ki dS[qi,ki]·K[ki]          — written directly (each query once)
//   dV[ki] += Σ_qi P[qi,ki]·dO[qi]          — atomic-add across query-tiles
//   dK[ki] += Σ_qi dS[qi,ki]·Q[qi]          — atomic-add across query-tiles
// where dS = P∘(dP - rowsum(P∘dP))·scale, dP = dO·Vᵀ. dK/dV MUST be pre-zeroed.
// Same math as the 6-pass attn_bwd_* kernels, fused. Causal (mask_kind==1) bounds
// ki to [0, qi_abs]. Requires sk ≤ MAX_SK and head_dim ≤ MAX_DH (threadgroup tiles).
// mask_kind ∈ {0,1} only (padding/bias fall back to the 6-pass path).
kernel void attn_bwd_fused_f32(
    device const float* q    [[buffer(0)]],
    device const float* k    [[buffer(1)]],
    device const float* v    [[buffer(2)]],
    device const float* dy   [[buffer(3)]],
    device float*        dq  [[buffer(4)]],
    device atomic_float* dk  [[buffer(5)]],
    device atomic_float* dv  [[buffer(6)]],
    constant uint&  sq        [[buffer(7)]],
    constant uint&  sk        [[buffer(8)]],
    constant uint&  heads     [[buffer(9)]],
    constant uint&  head_dim  [[buffer(10)]],
    constant float& scale     [[buffer(11)]],
    constant uint&  mask_kind [[buffer(12)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]]
) {
    constexpr uint Br = 8u;
    constexpr uint MAX_SK = 256u;
    constexpr uint MAX_DH = 64u;
    constexpr uint THREADS = 64u;

    threadgroup float Ptile[Br * MAX_SK];    // S, then softmax P
    threadgroup float dStile[Br * MAX_SK];   // dP, then dS
    threadgroup float Qtile[Br * MAX_DH];
    threadgroup float dOtile[Br * MAX_DH];
    threadgroup float Drow[Br];

    uint q_tile = tgid.x, hi = tgid.y, bi = tgid.z;
    uint qi0 = q_tile * Br;
    uint hs = heads * head_dim;
    uint qbase = bi * sq * hs + hi * head_dim;
    uint kbase = bi * sk * hs + hi * head_dim;
    bool causal = (mask_kind == 1u);

    // Load Q, dO tiles (Br × head_dim), zero-pad rows past sq.
    for (uint i = tid; i < Br * head_dim; i += THREADS) {
        uint qi = i / head_dim, d = i % head_dim;
        uint pos = qi0 + qi;
        Qtile[qi * MAX_DH + d]  = (pos < sq) ? q[qbase + pos * hs + d]  : 0.0f;
        dOtile[qi * MAX_DH + d] = (pos < sq) ? dy[qbase + pos * hs + d] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 1: S[qi,ki] = scale·(Q·K), causal-masked to -INF. Store in Ptile.
    for (uint c = tid; c < Br * sk; c += THREADS) {
        uint qi = c / sk, ki = c % sk;
        uint pos = qi0 + qi;
        float s = -INFINITY;
        if (pos < sq && !(causal && ki > pos)) {
            float acc = 0.0f;
            for (uint d = 0; d < head_dim; ++d)
                acc = fma(Qtile[qi * MAX_DH + d], k[kbase + ki * hs + d], acc);
            s = acc * scale;
        }
        Ptile[qi * MAX_SK + ki] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax each row over its causal band [0, khi]; Br threads.
    if (tid < Br) {
        uint qi = tid;
        uint pos = qi0 + qi;
        if (pos < sq) {
            uint khi = causal ? min(pos, sk - 1u) : (sk - 1u);
            float m = -INFINITY;
            for (uint ki = 0; ki <= khi; ++ki) m = max(m, Ptile[qi * MAX_SK + ki]);
            float l = 0.0f;
            for (uint ki = 0; ki <= khi; ++ki) {
                float e = exp(Ptile[qi * MAX_SK + ki] - m);
                Ptile[qi * MAX_SK + ki] = e;
                l += e;
            }
            float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
            for (uint ki = 0; ki <= khi; ++ki) Ptile[qi * MAX_SK + ki] *= inv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: dP[qi,ki] = dO[qi]·V[ki] over the band (else 0). Store in dStile.
    for (uint c = tid; c < Br * sk; c += THREADS) {
        uint qi = c / sk, ki = c % sk;
        uint pos = qi0 + qi;
        float acc = 0.0f;
        if (pos < sq && !(causal && ki > pos)) {
            for (uint d = 0; d < head_dim; ++d)
                acc = fma(dOtile[qi * MAX_DH + d], v[kbase + ki * hs + d], acc);
        }
        dStile[qi * MAX_SK + ki] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Drow[qi] = Σ_ki P·dP over the band; Br threads.
    if (tid < Br) {
        uint qi = tid;
        uint pos = qi0 + qi;
        float D = 0.0f;
        if (pos < sq) {
            uint khi = causal ? min(pos, sk - 1u) : (sk - 1u);
            for (uint ki = 0; ki <= khi; ++ki)
                D += Ptile[qi * MAX_SK + ki] * dStile[qi * MAX_SK + ki];
        }
        Drow[qi] = D;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dS[qi,ki] = P·(dP - D)·scale, in place in dStile (0 outside band).
    for (uint c = tid; c < Br * sk; c += THREADS) {
        uint qi = c / sk, ki = c % sk;
        uint pos = qi0 + qi;
        float ds = 0.0f;
        if (pos < sq && !(causal && ki > pos))
            ds = Ptile[qi * MAX_SK + ki] * (dStile[qi * MAX_SK + ki] - Drow[qi]) * scale;
        dStile[qi * MAX_SK + ki] = ds;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dQ[qi,d] = Σ_ki dS·K — direct write (each query row owned by this tile).
    for (uint c = tid; c < Br * head_dim; c += THREADS) {
        uint qi = c / head_dim, d = c % head_dim;
        uint pos = qi0 + qi;
        if (pos < sq) {
            uint khi = causal ? min(pos, sk - 1u) : (sk - 1u);
            float acc = 0.0f;
            for (uint ki = 0; ki <= khi; ++ki)
                acc = fma(dStile[qi * MAX_SK + ki], k[kbase + ki * hs + d], acc);
            dq[qbase + pos * hs + d] = acc;
        }
    }

    // dV[ki,d] += Σ_qi P·dO ; dK[ki,d] += Σ_qi dS·Q — atomic across query-tiles.
    for (uint c = tid; c < sk * head_dim; c += THREADS) {
        uint ki = c / head_dim, d = c % head_dim;
        float dvacc = 0.0f, dkacc = 0.0f;
        for (uint qi = 0; qi < Br; ++qi) {
            uint pos = qi0 + qi;
            if (pos < sq && !(causal && ki > pos)) {
                dvacc = fma(Ptile[qi * MAX_SK + ki], dOtile[qi * MAX_DH + d], dvacc);
                dkacc = fma(dStile[qi * MAX_SK + ki], Qtile[qi * MAX_DH + d], dkacc);
            }
        }
        if (dvacc != 0.0f)
            atomic_fetch_add_explicit(&dv[kbase + ki * hs + d], dvacc, memory_order_relaxed);
        if (dkacc != 0.0f)
            atomic_fetch_add_explicit(&dk[kbase + ki * hs + d], dkacc, memory_order_relaxed);
    }
}

kernel void gather_bwd_zero(
    device float* dst [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) dst[i] = 0.0f;
}

kernel void gather_bwd_acc(
    device const float* dy [[buffer(0)]],
    device const float* idx [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant uint& outer [[buffer(3)]],
    constant uint& axis_dim [[buffer(4)]],
    constant uint& num_idx [[buffer(5)]],
    constant uint& trailing [[buffer(6)]],
    uint o [[threadgroup_position_in_grid]]
) {
    if (o >= outer) return;
    for (uint k = 0; k < num_idx; ++k) {
        uint row = uint(idx[k]);
        if (row >= axis_dim) continue;
        for (uint j = 0; j < trailing; ++j) {
            float v = dy[(o * num_idx + k) * trailing + j];
            dst[(o * axis_dim + row) * trailing + j] += v;
        }
    }
}

// ── On-device Philox4x32-10 RNG ──────────────────────────────────────────
// Bit-matched to `rlx_ir::Philox4x32` / the shared `rng_philox.cu` (CUDA/ROCm):
// normal sample i reads Philox block i/2 lanes {0,1}|{2,3}; uniform i reads
// block i/4 lane i%4; `u32_to_unit=(bits>>8)/2^24`. The arena buffer is bound
// at the destination byte offset, so each entry writes `out[i]`.
static inline void rlx_philox_round(thread uint* s, uint k0, uint k1) {
    const ulong M0 = 0xD2561A75UL;
    const ulong M1 = 0xCD9E8D57UL;
    ulong p0 = (ulong)s[0] * M0;
    ulong p1 = (ulong)s[2] * M1;
    uint hi0 = (uint)(p0 >> 32); uint lo0 = (uint)p0;
    uint hi1 = (uint)(p1 >> 32); uint lo1 = (uint)p1;
    s[0] = hi1 ^ s[1] ^ k0;
    s[1] = lo1;
    s[2] = hi0 ^ s[3] ^ k1;
    s[3] = lo0;
}
static inline void rlx_philox_10(uint blk, uint seed_lo, uint seed_hi, thread uint* out) {
    // Counter for block `blk` (blk < 2^31 for u32 len → high words zero).
    uint s[4] = { blk, 0u, 0u, 0u };
    uint k0 = seed_lo, k1 = seed_hi;
    for (int i = 0; i < 10; ++i) {
        rlx_philox_round(s, k0, k1);
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    out[0] = s[0]; out[1] = s[1]; out[2] = s[2]; out[3] = s[3];
}
static inline float rlx_u32_to_unit(uint bits) {
    return (float)(bits >> 8) / (float)(1u << 24);
}
kernel void rng_normal_philox(
    device float* out [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    constant float& mean [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    constant uint& seed_lo [[buffer(4)]],
    constant uint& seed_hi [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= len) return;
    uint lane0 = (i & 1u) ? 2u : 0u;
    uint buf[4];
    rlx_philox_10(i / 2u, seed_lo, seed_hi, buf);
    float u1 = rlx_u32_to_unit(buf[lane0]);
    float u2 = rlx_u32_to_unit(buf[lane0 + 1u]);
    if (u1 < 1.17549435e-38f) u1 = 1.17549435e-38f; // f32::MIN_POSITIVE
    float r = sqrt(-2.0f * log(u1));
    float theta = 6.283185307179586f * u2;
    out[i] = mean + scale * (r * cos(theta));
}
kernel void rng_uniform_philox(
    device float* out [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    constant float& low [[buffer(2)]],
    constant float& high [[buffer(3)]],
    constant uint& seed_lo [[buffer(4)]],
    constant uint& seed_hi [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= len) return;
    uint buf[4];
    rlx_philox_10(i / 4u, seed_lo, seed_hi, buf);
    float u = rlx_u32_to_unit(buf[i & 3u]);
    out[i] = low + u * (high - low);
}
kernel void rng_fill_zero(
    device float* out [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= len) return;
    out[i] = 0.0f;
}
"#;

const RLX_KERNELS_MSL_DEQUANT: &str = include_str!("dequant_gguf.msl");
const RLX_KERNELS_MSL_FFT_GPU: &str = include_str!("fft_gpu.msl");
const RLX_KERNELS_MSL_SPLAT: &str = include_str!("splat.msl");
const RLX_KERNELS_MSL_SPLAT_CONIC: &str = include_str!("splat_conic_bin.msl");

// ── Register-blocked simdgroup GEMM tile configs (macro-generated) ──────────
// One MSL template per structure (plain / split-K), parameterized by tile:
//   {NAME}  kernel name          {NACC}  8×8-col accumulators / simdgroup
//   {TROWS} rows per threadgroup  {TCOLS} cols per threadgroup (= NACC*8)
// A config is TROWS×TCOLS output, (TROWS/8) simdgroups × NACC accumulators.
// `sgemm_tile_variant!(name, trows, nacc)` stamps a config; add a tile size by
// adding one call in `msl_source`. Keeps the kernel families DRY + tunable.
const SGEMM_TILE_TMPL: &str = r#"
kernel void {NAME}(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]]
) {
    uint row0 = tgid.y * {TROWS} + sgid * 8;
    uint col0 = tgid.x * {TCOLS};
    simdgroup_float8x8 acc[{NACC}];
    for (int j = 0; j < {NACC}; j++) acc[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0);
    for (uint kk = 0; kk < K; kk += 8) {
        simdgroup_float8x8 a, b[{NACC}];
        simdgroup_load(a, A + row0 * K + kk, K);
        for (int j = 0; j < {NACC}; j++) simdgroup_load(b[j], B + kk * N + col0 + j * 8, N);
        for (int j = 0; j < {NACC}; j++) simdgroup_multiply_accumulate(acc[j], a, b[j], acc[j]);
    }
    for (int j = 0; j < {NACC}; j++) simdgroup_store(acc[j], C + row0 * N + col0 + j * 8, N);
}
"#;

// Split-K variant: grid gains a Ksplits z-axis; each z-slice sums a K-chunk into
// its tile, then hardware-atomic-adds into a pre-zeroed C. For fat-K / small-MN.
const SGEMM_SPLITK_TMPL: &str = r#"
kernel void {NAME}(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device atomic_float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
    constant uint& Ksplits [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint slid  [[thread_index_in_simdgroup]]
) {
    uint row0 = tgid.y * {TROWS} + sgid * 8;
    uint col0 = tgid.x * {TCOLS};
    uint kchunk = K / Ksplits;
    uint kstart = tgid.z * kchunk;
    simdgroup_float8x8 acc[{NACC}];
    for (int j = 0; j < {NACC}; j++) acc[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0);
    for (uint kk = kstart; kk < kstart + kchunk; kk += 8) {
        simdgroup_float8x8 a, b[{NACC}];
        simdgroup_load(a, A + row0 * K + kk, K);
        for (int j = 0; j < {NACC}; j++) simdgroup_load(b[j], B + kk * N + col0 + j * 8, N);
        for (int j = 0; j < {NACC}; j++) simdgroup_multiply_accumulate(acc[j], a, b[j], acc[j]);
    }
    threadgroup float scratch[{TROWS} * {TCOLS}];
    for (int j = 0; j < {NACC}; j++) simdgroup_store(acc[j], scratch + (sgid * 8) * {TCOLS} + j * 8, {TCOLS});
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tile_elems = {TROWS} * {TCOLS};
    uint threads = ({TROWS} / 8) * 32;
    for (uint idx = sgid * 32 + slid; idx < tile_elems; idx += threads) {
        uint r = idx / {TCOLS}, c = idx % {TCOLS};
        atomic_fetch_add_explicit(&C[(tgid.y * {TROWS} + r) * N + col0 + c], scratch[idx],
                                  memory_order_relaxed);
    }
}
"#;

macro_rules! sgemm_tile_variant {
    ($name:literal, $trows:expr, $nacc:expr) => {
        SGEMM_TILE_TMPL
            .replace("{NAME}", $name)
            .replace("{NACC}", &($nacc).to_string())
            .replace("{TROWS}", &($trows).to_string())
            .replace("{TCOLS}", &(($nacc) * 8).to_string())
    };
}
macro_rules! sgemm_splitk_variant {
    ($name:literal, $trows:expr, $nacc:expr) => {
        SGEMM_SPLITK_TMPL
            .replace("{NAME}", $name)
            .replace("{NACC}", &($nacc).to_string())
            .replace("{TROWS}", &($trows).to_string())
            .replace("{TCOLS}", &(($nacc) * 8).to_string())
    };
}

/// Single source of truth for Metal's scalar activation kernels. Each entry
/// `(Variant, "name", "msl_expr")` generates an f32 + f16 in-place MSL kernel
/// (the expr sees `float x`) and its pipeline pair. Adding an activation is one
/// line here — no struct field, no dispatch arm, no separate kernel edit.
// The macro owns the (Activation → kernel-name) binding once; the scalar MATH
// comes from the shared `rlxsl` manifest (a single definition across all
// backends; the A&S erf polynomial now lives once in rlxsl rather than a
// hand-written `rlx_erf` helper — MSL has no erf builtin). Metal compiles MSL
// at runtime, so the source is assembled as a `String`; no build.rs needed.
macro_rules! scalar_activation_kernels {
    ( $( $variant:ident, $name:literal );+ $(;)? ) => {
        /// Build the standalone scalar-activation MSL kernels (f32 + f16) from
        /// the rlxsl manifest.
        fn scalar_act_msl() -> String {
            let mut src = String::new();
            $(
                {
                    let (stmts, expr) = rlxsl::emit_activation(
                        rlx_ir::op::Activation::$variant, rlxsl::Lang::Msl);
                    let body = stmts.join(" ");
                    src.push_str(&format!(
                        "kernel void {n}_sa(device float* d [[buffer(0)]], constant uint& n [[buffer(1)]], uint g [[thread_position_in_grid]]) {{ if (g>=n) return; float x = d[g]; {b} d[g] = {e}; }}\n",
                        n = $name, b = body, e = expr));
                    src.push_str(&format!(
                        "kernel void {n}_sa_h(device half* d [[buffer(0)]], constant uint& n [[buffer(1)]], uint g [[thread_position_in_grid]]) {{ if (g>=n) return; float x = float(d[g]); {b} d[g] = half({e}); }}\n",
                        n = $name, b = body, e = expr));
                }
            )+
            src
        }

        /// Build the (f32, f16) pipeline pair for each scalar activation.
        fn build_scalar_act_kernels(
            pipeline: &dyn Fn(&str) -> ComputePipelineState,
        ) -> std::collections::HashMap<rlx_ir::op::Activation, (ComputePipelineState, ComputePipelineState)>
        {
            let mut m = std::collections::HashMap::new();
            $(
                m.insert(
                    rlx_ir::op::Activation::$variant,
                    (pipeline(concat!($name, "_sa")), pipeline(concat!($name, "_sa_h"))),
                );
            )+
            m
        }
    };
}

scalar_activation_kernels! {
    Floor,       "floor";
    Ceil,        "ceil";
    Sign,        "sign";
    Softplus,    "softplus";
    Elu,         "elu";
    Erf,         "erf";
    HardSwish,   "hardswish";
    HardSigmoid, "hardsigmoid";
    Mish,        "mish";
    Softsign,    "softsign";
    LogSigmoid,  "logsigmoid";
}

/// MSL template for the split-K codebook weight-synthesis kernel (decode). `{S}`
/// = name suffix, `{T}` = x/dst element type (`float` or `half`); accumulation
/// is always f32 and the codebook stays f32 (only x/dst change precision).
const SYNTH_SPLITK_TMPL: &str = r#"
kernel void synth_matmul_codebook{S}(
    device float* arena          [[buffer(0)]],
    constant ulong& x_off        [[buffer(1)]],
    constant ulong& idx_off      [[buffer(2)]],
    constant ulong& cb_off       [[buffer(3)]],
    constant ulong& dst_off      [[buffer(4)]],
    constant uint& k_dim         [[buffer(5)]],
    constant uint& n_dim         [[buffer(6)]],
    constant uint& entry_dim     [[buffer(7)]],
    constant uint& num_entries   [[buffer(8)]],
    constant uint& m_dim         [[buffer(9)]],
    uint3 gid                    [[thread_position_in_grid]]
) {
    uint split = gid.x; uint j = gid.y; uint r = gid.z;
    if (j >= n_dim || r >= m_dim) return;
    (void)num_entries;
    uint nb = k_dim / entry_dim;
    device const {T}* xr    = (device const {T}*)((device const char*)arena + x_off) + (ulong)r * k_dim;
    device const uchar* idx = (device const uchar*)arena + idx_off + (ulong)j * nb;
    device const float* cb  = (device const float*)((device const char*)arena + cb_off);
    device {T}*         dst  = (device {T}*)((device char*)arena + dst_off);
    float acc = 0.0f;
    for (uint b = split; b < nb; b += 32u) {
        uint code = uint(idx[b]);
        device const float* c = cb + (ulong)code * entry_dim;
        uint base = b * entry_dim;
        for (uint t = 0u; t < entry_dim; ++t) acc += float(xr[base + t]) * c[t];
    }
    acc = simd_sum(acc);
    if (split == 0u) dst[(ulong)r * n_dim + j] = ({T})acc;
}
"#;

/// MSL template for the one-thread-per-output prefill kernel. NOTE: register-
/// tiling (reuse centroids across a row tile) was measured SLOWER here — the 4 KB
/// codebook is L1-resident so there's nothing to amortize, and tiling hurts x
/// coalescing + occupancy. A fused GPU kernel can't beat MPS for prefill anyway
/// (reconstruction is cache-cheap; MPS is a world-class tiled GEMM); the fast
/// prefill path is reconstruct→f32→MPS/AMX (see the CPU m>1 path). `{S}`=suffix,
/// `{T}`=x/dst element type.
const SYNTH_MM_TMPL: &str = r#"
kernel void synth_matmul_codebook_mm{S}(
    device float* arena          [[buffer(0)]],
    constant ulong& x_off        [[buffer(1)]],
    constant ulong& idx_off      [[buffer(2)]],
    constant ulong& cb_off       [[buffer(3)]],
    constant ulong& dst_off      [[buffer(4)]],
    constant uint& k_dim         [[buffer(5)]],
    constant uint& n_dim         [[buffer(6)]],
    constant uint& entry_dim     [[buffer(7)]],
    constant uint& num_entries   [[buffer(8)]],
    constant uint& m_dim         [[buffer(9)]],
    uint2 gid                    [[thread_position_in_grid]]
) {
    uint j = gid.x; uint r = gid.y;
    if (j >= n_dim || r >= m_dim) return;
    (void)num_entries;
    uint nb = k_dim / entry_dim;
    device const {T}* xr    = (device const {T}*)((device const char*)arena + x_off) + (ulong)r * k_dim;
    device const uchar* idx = (device const uchar*)arena + idx_off + (ulong)j * nb;
    device const float* cb  = (device const float*)((device const char*)arena + cb_off);
    device {T}*         dst  = (device {T}*)((device char*)arena + dst_off);
    float acc = 0.0f;
    for (uint b = 0u; b < nb; ++b) {
        uint code = uint(idx[b]);
        device const float* c = cb + (ulong)code * entry_dim;
        uint base = b * entry_dim;
        for (uint t = 0u; t < entry_dim; ++t) acc += float(xr[base + t]) * c[t];
    }
    dst[(ulong)r * n_dim + j] = ({T})acc;
}
"#;

/// Generate the codebook weight-synthesis MSL kernels for each
/// `(suffix => element-type)` pair from ONE template, so the f32 and f16
/// variants can't drift (mirrors `scalar_activation_kernels!`).
/// MSL kernel that reconstructs the dense f32 weight Wᵀ [n, k] (row-major,
/// contiguous) from u8 codebook indices — the weight-only half used by the m>8
/// prefill path (reconstruct → MPS sgemm, which beats any fused kernel). One
/// thread per (kb block, column j) writes entry_dim contiguous weights:
/// W[j, kb·entry_dim + t] = codebook[indices[j,kb], t]. Not dtype-templated
/// (MPS consumes f32); paired with `encode_mps_sgemm_bt` (B stored [n,k]).
const SYNTH_RECON_MSL: &str = r#"
kernel void synth_reconstruct(
    device float* arena          [[buffer(0)]],
    constant ulong& idx_off      [[buffer(1)]],
    constant ulong& cb_off       [[buffer(2)]],
    constant ulong& w_off        [[buffer(3)]],
    constant uint& k_dim         [[buffer(4)]],
    constant uint& n_dim         [[buffer(5)]],
    constant uint& entry_dim     [[buffer(6)]],
    uint2 gid                    [[thread_position_in_grid]]
) {
    uint nb = k_dim / entry_dim;
    uint kb = gid.x;
    uint j = gid.y;
    if (kb >= nb || j >= n_dim) return;
    device const uchar* idx = (device const uchar*)arena + idx_off + (ulong)j * nb;
    device const float* cb  = (device const float*)((device const char*)arena + cb_off);
    device float* w = (device float*)((device char*)arena + w_off)
                    + (ulong)j * k_dim + (ulong)kb * entry_dim;
    uint code = uint(idx[kb]);
    device const float* c = cb + (ulong)code * entry_dim;
    for (uint t = 0u; t < entry_dim; ++t) w[t] = c[t];
}
"#;

/// f16 reconstruct for the RLX_METAL_SYNTH_RECON_F16 prefill path: writes the dense
/// weight W[k,n] (row-major, stride n — the NON-transposed layout MPS `hgemm` reads)
/// as `half` from u8 indices + f32 codebook. Half the scratch bytes of the f32
/// reconstruct; paired with `encode_mps_hgemm` after casting x→f16. `W[b·d+t, j] =
/// codebook[indices[j,b], t]`.
const SYNTH_RECON_H_MSL: &str = r#"
kernel void synth_reconstruct_h(
    device float* arena          [[buffer(0)]],
    constant ulong& idx_off      [[buffer(1)]],
    constant ulong& cb_off       [[buffer(2)]],
    constant ulong& w_off        [[buffer(3)]],
    constant uint& k_dim         [[buffer(4)]],
    constant uint& n_dim         [[buffer(5)]],
    constant uint& entry_dim     [[buffer(6)]],
    uint2 gid                    [[thread_position_in_grid]]
) {
    uint nb = k_dim / entry_dim;
    uint kb = gid.x, j = gid.y;
    if (kb >= nb || j >= n_dim) return;
    device const uchar* idx = (device const uchar*)arena + idx_off + (ulong)j * nb;
    device const float* cb  = (device const float*)((device const char*)arena + cb_off);
    device half* w = (device half*)((device char*)arena + w_off);
    uint code = uint(idx[kb]);
    device const float* c = cb + (ulong)code * entry_dim;
    for (uint t = 0u; t < entry_dim; ++t) w[(ulong)(kb * entry_dim + t) * n_dim + j] = (half)c[t];
}
"#;

/// Threadgroup-tiled FUSED codebook matmul via `simdgroup_float8x8` MMAs. A 4×4
/// grid of 16 simdgroups (512 threads) computes a 32×32 output tile. Each K-step
/// stages the x tile (A) into threadgroup memory AND reconstructs the weight tile
/// (Bᵀ) from u8 indices + the L1-resident codebook directly into threadgroup memory
/// — the dense weight is NEVER materialized to DRAM. Bounds-checked on all three
/// dims (zero-fill loads, staged bounds-safe store), so it handles M/N/K not
/// multiples of 32. f32 only. Pairs with `encode_synth_matmul_tiled`.
///
/// MEASURED (M4 Pro, `synth_m_sweep_bench`): register blocking (this 64×64/2×2
/// version) cut large-M time ~15–20% vs the first 32×32/1×1 tiling (M=256: 1.5→1.27
/// ms) — but it STILL does NOT beat recon→MPS (M=256: 1.27 vs 0.71 ms; ~1.8× off
/// MPS's tuned GEMM). Two honest kernel iterations confirm the physics: at large M
/// you can't out-GEMM MPS by hand and the ceiling (dense-MPS) is only ~20% under
/// recon→MPS; at small/medium M the theoretical win is bigger (16× fewer weight
/// bytes) but a big register-blocked tile wastes rows + starves occupancy there — it
/// wants a different small-M split-K GEMM. Diminishing returns, so recon→MPS stays
/// the DEFAULT for m>8. Kept opt-in (`RLX_METAL_SYNTH_TILED`) for its non-speed wins:
/// zero scratch DRAM (recon→MPS needs a k·n·4 scratch — 16 MB at 2048²) and a single
/// capture-friendly pure-MSL dispatch (no MPS call). Next lever if ever needed:
/// double-buffer the K-panel (blocked here by the 32 KB threadgroup budget) +
/// a small-M split-K variant.
const SYNTH_TILED_MSL: &str = r#"
kernel void synth_matmul_codebook_tiled(
    device float* arena          [[buffer(0)]],
    constant ulong& x_off        [[buffer(1)]],
    constant ulong& idx_off      [[buffer(2)]],
    constant ulong& cb_off       [[buffer(3)]],
    constant ulong& dst_off      [[buffer(4)]],
    constant uint& k_dim         [[buffer(5)]],
    constant uint& n_dim         [[buffer(6)]],
    constant uint& entry_dim     [[buffer(7)]],
    constant uint& num_entries   [[buffer(8)]],
    constant uint& m_dim         [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint slid  [[thread_index_in_simdgroup]]
) {
    (void)num_entries;
    // 64×64 output tile per threadgroup, K-panel 32. 16 simdgroups (4×4); each owns
    // a 16×16 sub-tile = a 2×2 grid of 8×8 accumulators, so every A/B tile load feeds
    // FOUR MMAs (4× the arithmetic intensity of one 8×8 per load).
    uint sg_row = sgid / 4;            // 0..3
    uint sg_col = sgid % 4;            // 0..3
    uint tg_row_base = tgid.y * 64;    // output rows = m
    uint tg_col_base = tgid.x * 64;    // output cols = n
    uint nb = k_dim / entry_dim;

    threadgroup float A_tg[64 * 32];   // [m64][k32]
    threadgroup float B_tg[32 * 64];   // [k32][n64]

    simdgroup_float8x8 c00 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 c01 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 c10 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 c11 = simdgroup_float8x8(0.0f);

    device const float* xbase   = (device const float*)((device const char*)arena + x_off);
    device const uchar* idxbase = (device const uchar*)arena + idx_off;
    device const float* cb      = (device const float*)((device const char*)arena + cb_off);

    uint linear = sgid * 32 + slid;    // 0..511

    for (uint kk = 0; kk < k_dim; kk += 32) {
        // A tile (x): 64×32 = 2048 elems, 4 per thread.
        for (uint i = 0; i < 4; ++i) {
            uint idx = i * 512 + linear;
            uint ar = idx / 32, ac = idx % 32;
            uint r = tg_row_base + ar;
            uint kcol = kk + ac;
            A_tg[idx] = (r < m_dim && kcol < k_dim) ? xbase[(ulong)r * k_dim + kcol] : 0.0f;
        }
        // B tile (Wᵀ) reconstructed on-chip: B[k,j] = codebook[indices[j,k/D]*D + k%D].
        for (uint i = 0; i < 4; ++i) {
            uint idx = i * 512 + linear;
            uint br = idx / 64, bc = idx % 64;
            uint krow = kk + br;
            uint j = tg_col_base + bc;
            float w = 0.0f;
            if (j < n_dim && krow < k_dim) {
                uint blk = krow / entry_dim;
                uint t   = krow % entry_dim;
                uint code = uint(idxbase[(ulong)j * nb + blk]);
                w = cb[(ulong)code * entry_dim + t];
            }
            B_tg[idx] = w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kin = 0; kin < 32; kin += 8) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, &A_tg[(sg_row * 16 + 0) * 32 + kin], 32);
            simdgroup_load(a1, &A_tg[(sg_row * 16 + 8) * 32 + kin], 32);
            simdgroup_load(b0, &B_tg[kin * 64 + sg_col * 16 + 0], 64);
            simdgroup_load(b1, &B_tg[kin * 64 + sg_col * 16 + 8], 64);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Bounds-safe store of the four 8×8 blocks via a reused staging buffer.
    threadgroup float stage[16 * 64];
    device float* dst = (device float*)((device char*)arena + dst_off);
#define SYNTH_STORE_BLOCK(CC, DI, DJ)                                              \
    simdgroup_store(CC, &stage[sgid * 64], 8);                                     \
    threadgroup_barrier(mem_flags::mem_threadgroup);                               \
    {                                                                             \
        uint row_base = tg_row_base + sg_row * 16 + (DI) * 8;                      \
        uint col_base = tg_col_base + sg_col * 16 + (DJ) * 8;                      \
        for (uint e = slid; e < 64; e += 32) {                                    \
            uint rr = e / 8, cc2 = e % 8;                                         \
            uint r = row_base + rr, col = col_base + cc2;                          \
            if (r < m_dim && col < n_dim)                                          \
                dst[(ulong)r * n_dim + col] = stage[sgid * 64 + rr * 8 + cc2];     \
        }                                                                         \
    }                                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);
    SYNTH_STORE_BLOCK(c00, 0, 0)
    SYNTH_STORE_BLOCK(c01, 0, 1)
    SYNTH_STORE_BLOCK(c10, 1, 0)
    SYNTH_STORE_BLOCK(c11, 1, 1)
#undef SYNTH_STORE_BLOCK
}
"#;

/// f16 ("relaxed precision") variant of the tiled kernel: identical 64×64/2×2
/// structure, but the A/B threadgroup tiles are `half` and the MMAs use
/// `simdgroup_half8x8` inputs with an f32 accumulator — the fast path Apple's
/// matrix units + MPS/`matmul2d` actually take (`dv_f16_dv_f16_dv_f32`). External
/// I/O stays f32 (x/codebook cast to half on load; f32 accumulate → f32 dst), so it
/// drops into the same dispatch/bench harness. Half-width tiles also HALVE the
/// threadgroup-memory footprint (8 KB vs 16 KB), the headroom double-buffering needs.
/// Opt-in via `RLX_METAL_SYNTH_TILED` + `RLX_METAL_SYNTH_TILED_F16`.
///
/// MEASURED: correct (rel err ~1e-4 vs f32 CPU) but ~0 speedup over the f32 tiled
/// kernel (M=256: 1.24 vs 1.22 ms). That's the tell: this kernel is bound by the
/// SYNCHRONOUS-load + barrier stall (see the note on `simdgroup_async_copy`), NOT
/// matrix-ALU throughput — so switching to the faster f16 MMA path moves nothing.
/// It still loses to recon→MPS. The freed SRAM would let double-buffering hide some
/// of that stall (the one remaining lever short of the async-copy intrinsic).
const SYNTH_TILED_H_MSL: &str = r#"
kernel void synth_matmul_codebook_tiled_h(
    device float* arena          [[buffer(0)]],
    constant ulong& x_off        [[buffer(1)]],
    constant ulong& idx_off      [[buffer(2)]],
    constant ulong& cb_off       [[buffer(3)]],
    constant ulong& dst_off      [[buffer(4)]],
    constant uint& k_dim         [[buffer(5)]],
    constant uint& n_dim         [[buffer(6)]],
    constant uint& entry_dim     [[buffer(7)]],
    constant uint& num_entries   [[buffer(8)]],
    constant uint& m_dim         [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint slid  [[thread_index_in_simdgroup]]
) {
    (void)num_entries;
    uint sg_row = sgid / 4;
    uint sg_col = sgid % 4;
    uint tg_row_base = tgid.y * 64;
    uint tg_col_base = tgid.x * 64;
    uint nb = k_dim / entry_dim;

    threadgroup half A_tg[64 * 32];    // f16 tiles (half the SRAM of the f32 kernel)
    threadgroup half B_tg[32 * 64];

    simdgroup_float8x8 c00 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 c01 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 c10 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 c11 = simdgroup_float8x8(0.0f);

    device const float* xbase   = (device const float*)((device const char*)arena + x_off);
    device const uchar* idxbase = (device const uchar*)arena + idx_off;
    device const float* cb      = (device const float*)((device const char*)arena + cb_off);

    uint linear = sgid * 32 + slid;

    for (uint kk = 0; kk < k_dim; kk += 32) {
        for (uint i = 0; i < 4; ++i) {
            uint idx = i * 512 + linear;
            uint ar = idx / 32, ac = idx % 32;
            uint r = tg_row_base + ar;
            uint kcol = kk + ac;
            A_tg[idx] = (r < m_dim && kcol < k_dim) ? (half)xbase[(ulong)r * k_dim + kcol] : (half)0.0h;
        }
        for (uint i = 0; i < 4; ++i) {
            uint idx = i * 512 + linear;
            uint br = idx / 64, bc = idx % 64;
            uint krow = kk + br;
            uint j = tg_col_base + bc;
            half w = (half)0.0h;
            if (j < n_dim && krow < k_dim) {
                uint blk = krow / entry_dim;
                uint t   = krow % entry_dim;
                uint code = uint(idxbase[(ulong)j * nb + blk]);
                w = (half)cb[(ulong)code * entry_dim + t];
            }
            B_tg[idx] = w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kin = 0; kin < 32; kin += 8) {
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, &A_tg[(sg_row * 16 + 0) * 32 + kin], 32);
            simdgroup_load(a1, &A_tg[(sg_row * 16 + 8) * 32 + kin], 32);
            simdgroup_load(b0, &B_tg[kin * 64 + sg_col * 16 + 0], 64);
            simdgroup_load(b1, &B_tg[kin * 64 + sg_col * 16 + 8], 64);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float stage[16 * 64];
    device float* dst = (device float*)((device char*)arena + dst_off);
#define SYNTH_STORE_BLOCK_H(CC, DI, DJ)                                            \
    simdgroup_store(CC, &stage[sgid * 64], 8);                                     \
    threadgroup_barrier(mem_flags::mem_threadgroup);                               \
    {                                                                             \
        uint row_base = tg_row_base + sg_row * 16 + (DI) * 8;                      \
        uint col_base = tg_col_base + sg_col * 16 + (DJ) * 8;                      \
        for (uint e = slid; e < 64; e += 32) {                                    \
            uint rr = e / 8, cc2 = e % 8;                                         \
            uint r = row_base + rr, col = col_base + cc2;                          \
            if (r < m_dim && col < n_dim)                                          \
                dst[(ulong)r * n_dim + col] = stage[sgid * 64 + rr * 8 + cc2];     \
        }                                                                         \
    }                                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);
    SYNTH_STORE_BLOCK_H(c00, 0, 0)
    SYNTH_STORE_BLOCK_H(c01, 0, 1)
    SYNTH_STORE_BLOCK_H(c10, 1, 0)
    SYNTH_STORE_BLOCK_H(c11, 1, 1)
#undef SYNTH_STORE_BLOCK_H
}
"#;

/// Fused backward of `Op::SynthMatMul` (`SynthKind::Codebook`) — the two
/// gradients as single dispatches each, replacing the ~11-op decomposition.
/// Arena-buffer + byte-offset convention (matches `synth_reconstruct`); f32.
const SYNTH_BWD_MSL: &str = r#"
// dx[m,k] = upstream[m,n] · Ŵᵀ[n,k], where Ŵᵀ[j, kb·d+t] = codebook[idx[j·nb+kb]][t].
// One thread per output element (i,p); the codebook entry is reconstructed in the
// inner loop over n — no materialized weight, no separate gather.
kernel void synth_bwd_dx(
    device float* arena     [[buffer(0)]],
    constant ulong& up_off  [[buffer(1)]],   // upstream [m,n] f32
    constant ulong& idx_off [[buffer(2)]],   // indices  [n,nb] u8
    constant ulong& cb_off  [[buffer(3)]],   // codebook [ne,d] f32
    constant ulong& dst_off [[buffer(4)]],   // dx       [m,k] f32
    constant uint& m [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& k [[buffer(7)]],
    constant uint& d [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint p = gid.x;
    uint i = gid.y;
    if (i >= m || p >= k) return;
    device const float* up  = (device const float*)((device const char*)arena + up_off);
    device const uchar* idx = (device const uchar*)arena + idx_off;
    device const float* cb  = (device const float*)((device const char*)arena + cb_off);
    device float* dx        = (device float*)((device char*)arena + dst_off);
    uint nb = k / d;
    uint kb = p / d;
    uint t  = p - kb * d;
    float acc = 0.0f;
    for (uint j = 0; j < n; ++j) {
        uint code = uint(idx[j * nb + kb]);
        acc += up[i * n + j] * cb[code * d + t];
    }
    dx[i * k + p] = acc;
}

// d_codebook[ne,d]: each entry gets the summed weight-gradient of the blocks that
// index it. grad_W = upstreamᵀ·x; block bi=j·nb+kb contributes
// grad_W_block[bi][t] = Σ_i upstream[i,j]·x[i, kb·d+t] to codebook[idx[bi]][t].
// One thread per output (e,t) scans all blocks — no atomics, no pre-zero.
kernel void synth_bwd_codebook(
    device float* arena     [[buffer(0)]],
    constant ulong& up_off  [[buffer(1)]],   // upstream [m,n] f32
    constant ulong& idx_off [[buffer(2)]],   // indices  [n,nb] u8
    constant ulong& x_off   [[buffer(3)]],   // x        [m,k] f32
    constant ulong& dst_off [[buffer(4)]],   // d_codebook [ne,d] f32
    constant uint& m  [[buffer(5)]],
    constant uint& n  [[buffer(6)]],
    constant uint& k  [[buffer(7)]],
    constant uint& d  [[buffer(8)]],
    constant uint& ne [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint t = gid.x;
    uint e = gid.y;
    if (e >= ne || t >= d) return;
    device const float* up  = (device const float*)((device const char*)arena + up_off);
    device const uchar* idx = (device const uchar*)arena + idx_off;
    device const float* x   = (device const float*)((device const char*)arena + x_off);
    device float* dcb       = (device float*)((device char*)arena + dst_off);
    uint nb = k / d;
    float acc = 0.0f;
    for (uint j = 0; j < n; ++j) {
        for (uint kb = 0; kb < nb; ++kb) {
            if (uint(idx[j * nb + kb]) != e) continue;
            uint col = kb * d + t;
            float g = 0.0f;
            for (uint i = 0; i < m; ++i) {
                g += up[i * n + j] * x[i * k + col];
            }
            acc += g;
        }
    }
    dcb[e * d + t] = acc;
}
"#;

/// Register-blocked simdgroup GEMM with a bias + activation epilogue folded into
/// the store (one dispatch, no separate epilogue pass). 64×64 tile, 4×4
/// simdgroups × 2×2 accumulators — MPS-class arithmetic intensity (measured).
/// Arena-buffer + byte-offset convention; f32; m,n multiple of 64, k of 16.
/// `act`: 0 = none, 1 = ReLU. Beats MPS+separate-epilogue on small/aligned shapes
/// (2.14× on 192²) — the win on kernels MPS can't fuse. Opt-in via encode gate.
const RB_GEMM_BIAS_MSL: &str = r#"
kernel void gemm_rb_bias(
    device const float* A    [[buffer(0)]],   // A [M,K] f32 (arena)
    device const float* B    [[buffer(1)]],   // B [K,N] f32 (arena or weight buf)
    device const float* bias [[buffer(2)]],   // bias [N] f32
    device float* C          [[buffer(3)]],   // C [M,N] f32 (arena)
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& act [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint slid  [[thread_index_in_simdgroup]]
) {
    const uint KT = 16;
    uint sg_row = sgid / 4, sg_col = sgid % 4;
    uint rbase = tgid.y * 64, cbase = tgid.x * 64;
    threadgroup float A_tg[64 * 16];
    threadgroup float B_tg[16 * 64];
    threadgroup float C_tg[64 * 64];
    simdgroup_float8x8 c00 = simdgroup_float8x8(0.0f), c01 = c00, c10 = c00, c11 = c00;
    uint lin = sgid * 32 + slid;
    for (uint kk = 0; kk < K; kk += KT) {
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 512 + lin;
            uint ar = idx / KT, ac = idx % KT;
            A_tg[idx] = A[(rbase + ar) * K + (kk + ac)];
        }
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 512 + lin;
            uint br = idx / 64, bc = idx % 64;
            B_tg[br * 64 + bc] = B[(kk + br) * N + (cbase + bc)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ki = 0; ki < KT; ki += 8) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, &A_tg[(sg_row * 16 + 0) * KT + ki], KT);
            simdgroup_load(a1, &A_tg[(sg_row * 16 + 8) * KT + ki], KT);
            simdgroup_load(b0, &B_tg[ki * 64 + sg_col * 16 + 0], 64);
            simdgroup_load(b1, &B_tg[ki * 64 + sg_col * 16 + 8], 64);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint cr = sg_row * 16, cc = sg_col * 16;
    simdgroup_store(c00, &C_tg[(cr + 0) * 64 + cc + 0], 64);
    simdgroup_store(c01, &C_tg[(cr + 0) * 64 + cc + 8], 64);
    simdgroup_store(c10, &C_tg[(cr + 8) * 64 + cc + 0], 64);
    simdgroup_store(c11, &C_tg[(cr + 8) * 64 + cc + 8], 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 8; ++i) {
        uint idx = i * 512 + lin;
        uint r = idx / 64, cn = idx % 64;
        uint gc = cbase + cn;
        float v = C_tg[idx] + bias[gc];
        if (act == 1u) v = max(v, 0.0f);
        C[(rbase + r) * N + gc] = v;
    }
}
"#;

/// Fused reconstruct of the dense weight `W[k,n]` directly (coalesced write over
/// n) from u8 indices + codebook — one dispatch replacing cast+gather+reshape+
/// transpose. `W[kb·d+t, j] = codebook[idx[j·nb+kb], t]`, `nb=k/d`.
// Fused reconstruct of the codebook weight in the BACKWARD-friendly `w_bt[n,k]`
// layout (cast+gather+reshape in one dispatch, no transpose). The model emits
// `W[k,n] = Transpose(this)`, so the forward transpose is the only conversion and
// the backward `dx = dy·w_bt` reuses this buffer for free (AD cancels `Transposeᵀ`).
// Coalesced write over the fast `k` axis.
const SYNTH_RECON_NK_MSL: &str = r#"
kernel void synth_reconstruct_nk(
    device float* arena     [[buffer(0)]],
    constant ulong& idx_off [[buffer(1)]],
    constant ulong& cb_off  [[buffer(2)]],
    constant ulong& w_off   [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& d [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint j = gid.x;   // k (fast → coalesced write)
    uint i = gid.y;   // n
    if (j >= k || i >= n) return;
    uint nb = k / d;
    device const uchar* idx = (device const uchar*)arena + idx_off;
    device const float* cb  = (device const float*)((device const char*)arena + cb_off);
    device float* w         = (device float*)((device char*)arena + w_off);
    uint kb = j / d, t = j - kb * d;
    uint code = uint(idx[i * nb + kb]);
    w[i * k + j] = cb[code * d + t];
}
"#;

macro_rules! synth_matmul_kernels {
    ( $( $suffix:literal => $elem:literal );+ $(;)? ) => {
        fn synth_matmul_msl() -> String {
            let mut src = String::new();
            $(
                src.push_str(&SYNTH_SPLITK_TMPL.replace("{S}", $suffix).replace("{T}", $elem));
                src.push_str(&SYNTH_MM_TMPL.replace("{S}", $suffix).replace("{T}", $elem));
            )+
            src.push_str(SYNTH_RECON_MSL);
            src.push_str(SYNTH_RECON_H_MSL);
            src.push_str(SYNTH_TILED_MSL);
            src.push_str(SYNTH_TILED_H_MSL);
            src.push_str(SYNTH_BWD_MSL);
            src.push_str(RB_GEMM_BIAS_MSL);
            src.push_str(SYNTH_RECON_NK_MSL);
            src
        }
    };
}

synth_matmul_kernels! {
    ""   => "float";
    "_h" => "half";
}

/// The core scalar activations paired with their Metal kernel base name — the
/// single list driving the generated inline functions and the f16/f32 kernels.
const CORE_ACTS: &[(rlx_ir::op::Activation, &str)] = {
    use rlx_ir::op::Activation as A;
    &[
        (A::Gelu, "gelu"),
        (A::GeluApprox, "gelu_approx"),
        (A::Silu, "silu"),
        (A::Relu, "relu"),
        (A::Sigmoid, "sigmoid"),
        (A::Tanh, "tanh"),
        (A::Exp, "exp"),
        (A::Log, "log"),
        (A::Sqrt, "sqrt"),
        (A::Rsqrt, "rsqrt"),
        (A::Recip, "rec"),
        (A::Neg, "neg"),
        (A::Abs, "abs"),
        (A::Sin, "sin"),
        (A::Cos, "cos"),
        (A::Tan, "tan"),
        (A::Atan, "atan"),
        (A::Round, "round"),
    ]
};

/// One `inline float rlx_<name>_scalar(float x)` per core activation, generated
/// from the shared rlxsl manifest — the single on-device definition of each
/// activation's scalar math. The f16/f32/vec4 kernels below all call these, so
/// the A&S erf polynomial (etc.) is no longer re-inlined per kernel. Metal
/// inlines these, so routing through them costs no performance.
fn scalar_act_fns_msl() -> String {
    let mut src = String::from("// @generated from rlxsl — scalar activation device functions.\n");
    for (act, name) in CORE_ACTS {
        let (stmts, expr) = rlxsl::emit_activation(*act, rlxsl::Lang::Msl);
        let body = stmts.join(" ");
        src.push_str(&format!(
            "inline float rlx_{name}_scalar(float x) {{ {body} return {expr}; }}\n"
        ));
    }
    src
}

/// The core scalar-activation **f16** in-place kernels (`{name}_inplace_h`),
/// each just calling the shared `rlx_<name>_scalar`. Replaces the hand-written
/// MSL that re-inlined the A&S erf polynomial (and had drifted: unclamped erf
/// arg, ties-away `round`). Signature matches the dispatch: `buffer(0)=half*
/// data`, `buffer(1)=len`.
fn core_act_inplace_h_msl() -> String {
    let mut src =
        String::from("// @generated from rlxsl — core scalar-activation f16 in-place kernels.\n");
    for (_act, name) in CORE_ACTS {
        src.push_str(&format!(
            "kernel void {name}_inplace_h(device half* data [[buffer(0)]], constant uint& len [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{ if (gid >= len) return; data[gid] = half(rlx_{name}_scalar(float(data[gid]))); }}\n"
        ));
    }
    src
}

/// The f32 `gelu`/`gelu_approx` in-place kernels (char-arena form), each calling
/// the shared `rlx_<name>_scalar`. These were the f32 scalar `_inplace` kernels
/// that re-inlined non-trivial math; the trivial ones (relu/exp/…) stay
/// hand-written.
fn gelu_inplace_f32_msl() -> String {
    let mut src = String::from("// @generated from rlxsl — f32 gelu/gelu_approx in-place.\n");
    for name in ["gelu", "gelu_approx"] {
        src.push_str(&format!(
            "kernel void {name}_inplace(device char* arena [[buffer(0)]], constant ulong& data_byte_off [[buffer(1)]], constant uint& len [[buffer(2)]], uint gid [[thread_position_in_grid]]) {{ if (gid >= len) return; device float* data = (device float*)(arena + data_byte_off); data[gid] = rlx_{name}_scalar(data[gid]); }}\n"
        ));
    }
    src
}

/// `inline float rlx_pow_scalar(float a, float b)` — the Rust-`powf`-matching
/// pow generated from rlxsl, for the tuned broadcast kernels (whose bare MSL
/// `pow` NaN'd on a negative base). Metal inlines it → no perf cost.
fn pow_scalar_fn_msl() -> String {
    let (stmts, expr) = rlxsl::binary::emit_binary(rlx_ir::op::BinaryOp::Pow, rlxsl::Lang::Msl);
    format!(
        "// @generated from rlxsl — Rust-powf-matching scalar pow (+ a vec4 overload\n\
         // for the packed_float4 broadcast kernels).\n\
         inline float rlx_pow_scalar(float a, float b) {{ {} return {expr}; }}\n\
         inline packed_float4 rlx_pow_scalar(packed_float4 a, packed_float4 b) {{ \
         packed_float4 r; for (uint i = 0u; i < 4u; ++i) {{ r[i] = rlx_pow_scalar(a[i], b[i]); }} return r; }}\n",
        stmts.join(" ")
    )
}

/// f32 + f16-KV decode SDPA generated from ONE template (below): the K/V
/// buffers are `__KV__` (`float` or `half`), while Q, the online-softmax
/// accumulation, and the output stay f32 — so `sdpa_decode_m1_f16kv` reads a
/// half-sized KV cache with no accumulation-precision change. `float()` /
/// `float4()` are no-ops on the f32 variant.
const SDPA_DECODE_M1_TEMPLATE: &str = r####"kernel void __NAME__(
    device const float* arena_q   [[buffer(0)]],
    device const __KV__* arena_k   [[buffer(1)]],
    device const __KV__* arena_v   [[buffer(2)]],
    device const float* arena_m   [[buffer(3)]],
    device float*       arena_o   [[buffer(4)]],
    constant uint& batch       [[buffer(5)]],
    constant uint& heads       [[buffer(6)]],
    constant uint& head_dim    [[buffer(7)]],
    constant uint& q_stride    [[buffer(8)]],
    constant uint& mask_kind   [[buffer(9)]],
    constant uint& seq_k       [[buffer(10)]],
    constant uint& k_stride    [[buffer(11)]],
    constant uint& bhsd        [[buffer(12)]],
    constant uint& window      [[buffer(13)]],
    constant float& score_scale  [[buffer(14)]],
    constant float& attn_softcap [[buffer(15)]],
    constant SdpaOffsets& byte_offs [[buffer(16)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tsize [[threads_per_threadgroup]]
) {
    device const float* Q = (device const float*)((device const char*)arena_q + byte_offs.q);
    device const __KV__* K = (device const __KV__*)((device const char*)arena_k + byte_offs.k);
    device const __KV__* V = (device const __KV__*)((device const char*)arena_v + byte_offs.v);
    device const float* M = (device const float*)((device const char*)arena_m + byte_offs.m);
    device float* OUT     = (device float*)((device char*)arena_o + byte_offs.o);

    // TG=32 × dh≤128 → 16 KiB for partial O; fits Apple7/8 32 KiB tg limit.
    constexpr uint MAX_DH = 128u;
    constexpr uint TG = 32u;
    threadgroup float tg_m[TG];
    threadgroup float tg_l[TG];
    threadgroup float tg_o[TG * MAX_DH];

    uint bi = tgid / heads;
    uint hi = tgid % heads;
    if (bi >= batch) return;

    float scale = (score_scale > 0.0f) ? score_scale : rsqrt(float(head_dim));
    float softcap_inv = (attn_softcap > 0.0f) ? (1.0f / attn_softcap) : 0.0f;
    uint q_offset = seq_k - 1u;
    uint q_base = qkv_q_offset(bi, hi, 0u, heads, 1u, head_dim, q_stride, bhsd);
    // V/output per-head width (asymmetric MLA; == head_dim for symmetric).
    uint vdh = (byte_offs.v_head_dim == 0u) ? head_dim : byte_offs.v_head_dim;

    // Large head_dim OR v_head_dim: one thread walks all of K (rare for LLM
    // decode; Gemma-4, and MLA where qk=192 exceeds MAX_DH).
    if (head_dim > MAX_DH || vdh > MAX_DH || tsize < 2u) {
        if (tid != 0u) return;
        float q_reg[512];
        for (uint d = 0; d < head_dim; ++d) q_reg[d] = Q[q_base + d];
        float m_acc = -1e30f;
        float l_acc = 0.0f;
        float o_acc[512];
        for (uint d = 0; d < vdh; ++d) o_acc[d] = 0.0f;
        for (uint ki = 0; ki < seq_k; ++ki) {
            uint k_base = qkv_kv_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, head_dim, k_stride, bhsd);
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) dot += q_reg[d] * float(K[k_base + d]);
            float s = dot * scale;
            if (softcap_inv > 0.0f) s = precise::tanh(s * softcap_inv) * attn_softcap;
            if (mask_kind == 1u) {
                if (ki > q_offset) s = -1e9f;
            } else if (mask_kind == 2u) {
                if (M[bi * k_stride + ki] < 0.5f) s = -1e9f;
            } else if (mask_kind == 4u) {
                uint lo = q_offset > window ? q_offset - window : 0u;
                if (ki < lo || ki > q_offset) s = -1e9f;
            }
            // Masked positions contribute 0; skip them so a masked V slot holding
            // inf (uninitialized f16 KV padding) can't poison the accumulator via
            // 0*inf = NaN. Bit-identical for finite (f32) padding.
            if (s <= -1.0e9f) continue;
            float m_new = max(m_acc, s);
            float e_old = exp(m_acc - m_new);
            float e_cur = exp(s - m_new);
            l_acc = e_old * l_acc + e_cur;
            uint v_base = qkv_v_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, vdh, k_stride, bhsd);
            for (uint d = 0; d < vdh; ++d) {
                o_acc[d] = e_old * o_acc[d] + e_cur * float(V[v_base + d]);
            }
            m_acc = m_new;
        }
        float inv_l = 1.0f / l_acc;
        uint o_base = qkv_out_offset(bi, hi, 0u, heads, 1u, vdh, q_stride, bhsd);
        for (uint d = 0; d < vdh; ++d) OUT[o_base + d] = o_acc[d] * inv_l;
        return;
    }

    float q_reg[MAX_DH];
    for (uint d = tid; d < head_dim; d += tsize) tg_o[d] = Q[q_base + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = 0; d < head_dim; ++d) q_reg[d] = tg_o[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_acc = -1e30f;
    float l_acc = 0.0f;
    float o_acc[MAX_DH];
    for (uint d = 0; d < vdh; ++d) o_acc[d] = 0.0f;

    for (uint ki = tid; ki < seq_k; ki += tsize) {
        uint k_base = qkv_kv_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, head_dim, k_stride, bhsd);
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d += 4u) {
            if (d + 3u < head_dim) {
                float4 qv = float4(q_reg[d], q_reg[d + 1u], q_reg[d + 2u], q_reg[d + 3u]);
                float4 kv = float4(*(device const __KV__4*)(K + k_base + d));
                dot += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
            } else {
                for (uint dd = d; dd < head_dim; ++dd) dot += q_reg[dd] * float(K[k_base + dd]);
            }
        }
        float s = dot * scale;
        if (softcap_inv > 0.0f) s = precise::tanh(s * softcap_inv) * attn_softcap;
        if (mask_kind == 1u) {
            if (ki > q_offset) s = -1e9f;
        } else if (mask_kind == 2u) {
            if (M[bi * k_stride + ki] < 0.5f) s = -1e9f;
        } else if (mask_kind == 4u) {
            uint lo = q_offset > window ? q_offset - window : 0u;
            if (ki < lo || ki > q_offset) s = -1e9f;
        }
        // Skip masked positions (contribute 0) so an inf in a masked V slot
        // (uninitialized f16 KV padding) can't poison the accumulator via
        // 0*inf = NaN. Bit-identical for finite (f32) padding.
        if (s <= -1.0e9f) continue;
        float m_new = max(m_acc, s);
        float e_old = exp(m_acc - m_new);
        float e_cur = exp(s - m_new);
        l_acc = e_old * l_acc + e_cur;
        uint v_base = qkv_v_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, vdh, k_stride, bhsd);
        for (uint d = 0; d < vdh; ++d) {
            o_acc[d] = e_old * o_acc[d] + e_cur * float(V[v_base + d]);
        }
        m_acc = m_new;
    }

    // Idle lanes beyond TG keep identity merge state (already set if tid>=TG).
    if (tid < TG) {
        tg_m[tid] = m_acc;
        tg_l[tid] = l_acc;
        for (uint d = 0; d < vdh; ++d) tg_o[tid * MAX_DH + d] = o_acc[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = TG / 2u; stride > 0u; stride /= 2u) {
        if (tid < stride) {
            float m1 = tg_m[tid];
            float m2 = tg_m[tid + stride];
            float m_new = max(m1, m2);
            float e1 = exp(m1 - m_new);
            float e2 = exp(m2 - m_new);
            tg_l[tid] = e1 * tg_l[tid] + e2 * tg_l[tid + stride];
            tg_m[tid] = m_new;
            for (uint d = 0; d < vdh; ++d) {
                tg_o[tid * MAX_DH + d] =
                    e1 * tg_o[tid * MAX_DH + d] + e2 * tg_o[(tid + stride) * MAX_DH + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float inv_l = 1.0f / tg_l[0];
        uint o_base = qkv_out_offset(bi, hi, 0u, heads, 1u, vdh, q_stride, bhsd);
        for (uint d = 0; d < vdh; ++d) OUT[o_base + d] = tg_o[d] * inv_l;
    }
}"####;

macro_rules! sdpa_decode_m1_variant {
    ($name:expr, $kv:expr) => {
        SDPA_DECODE_M1_TEMPLATE
            .replace("__NAME__", $name)
            .replace("__KV__", $kv)
    };
}

// ── Flash-decoding (split-KV) for m=1 decode ────────────────────────────────
// The base `sdpa_decode_m1` launches exactly `batch*heads` threadgroups (one
// per head), each 32 threads walking ALL of K. At batch=1 that's ~16 tgs —
// far too few to fill the GPU, so decode attention is occupancy-starved (~37%
// of decode GPU time for trivial FLOPs). Flash-decoding adds a KV-partition
// axis: `batch*heads*P` threadgroups each attend one KV slice and write a
// partial online-softmax state {m, l, o[vdh]} to `scratch`; a tiny combine
// kernel then merges the P partials per head. Raises tg count 16→16*P.
//
// Scratch layout: float[(bi*heads+hi)*n_part + part][SLOT], SLOT = 2 + MAX_DH
// (m at +0, l at +1, un-normalized o at +2+d). Partial writes o WITHOUT the
// 1/l divide; combine does the cross-partition rescale then normalizes.
const SDPA_DECODE_M1_PARTIAL_TEMPLATE: &str = r####"kernel void __NAME__(
    device const float* arena_q   [[buffer(0)]],
    device const __KV__* arena_k   [[buffer(1)]],
    device const __KV__* arena_v   [[buffer(2)]],
    device const float* arena_m   [[buffer(3)]],
    device float*       arena_o   [[buffer(4)]],
    constant uint& batch       [[buffer(5)]],
    constant uint& heads       [[buffer(6)]],
    constant uint& head_dim    [[buffer(7)]],
    constant uint& q_stride    [[buffer(8)]],
    constant uint& mask_kind   [[buffer(9)]],
    constant uint& seq_k       [[buffer(10)]],
    constant uint& k_stride    [[buffer(11)]],
    constant uint& bhsd        [[buffer(12)]],
    constant uint& window      [[buffer(13)]],
    constant float& score_scale  [[buffer(14)]],
    constant float& attn_softcap [[buffer(15)]],
    constant SdpaOffsets& byte_offs [[buffer(16)]],
    device float* scratch      [[buffer(17)]],
    constant uint& n_part      [[buffer(18)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tsize [[threads_per_threadgroup]]
) {
    device const float* Q = (device const float*)((device const char*)arena_q + byte_offs.q);
    device const __KV__* K = (device const __KV__*)((device const char*)arena_k + byte_offs.k);
    device const __KV__* V = (device const __KV__*)((device const char*)arena_v + byte_offs.v);
    device const float* M = (device const float*)((device const char*)arena_m + byte_offs.m);

    constexpr uint MAX_DH = 128u;
    constexpr uint SLOT = 2u + MAX_DH;
    // No threadgroup memory: the threadgroup is exactly one simdgroup (32
    // threads), so the online-softmax reduction below uses simd_max/simd_sum
    // (warp-level) instead of a threadgroup-memory tree. This frees the old
    // 16 KB tg_o[32*128] scratch — the real occupancy limiter — so many more
    // decode-attention threadgroups stay resident per core, and it drops the
    // reduction barriers. Numerically identical to the tree merge.

    uint part = tgid % n_part;
    uint t    = tgid / n_part;
    uint hi   = t % heads;
    uint bi   = t / heads;
    if (bi >= batch) return;

    uint vdh = (byte_offs.v_head_dim == 0u) ? head_dim : byte_offs.v_head_dim;
    uint slot = (bi * heads + hi) * n_part + part;

    uint chunk = (seq_k + n_part - 1u) / n_part;
    uint start = part * chunk;
    uint end   = min(start + chunk, seq_k);
    if (start >= seq_k) {
        if (tid == 0u) {
            scratch[slot * SLOT + 0u] = -1e30f;
            scratch[slot * SLOT + 1u] = 0.0f;
            for (uint d = 0; d < vdh; ++d) scratch[slot * SLOT + 2u + d] = 0.0f;
        }
        return;
    }

    float scale = (score_scale > 0.0f) ? score_scale : rsqrt(float(head_dim));
    float softcap_inv = (attn_softcap > 0.0f) ? (1.0f / attn_softcap) : 0.0f;
    uint q_offset = seq_k - 1u;
    uint q_base = qkv_q_offset(bi, hi, 0u, heads, 1u, head_dim, q_stride, bhsd);

    // Each lane loads the full query row directly (≤128 floats, L1-cached — the
    // 32× redundancy is far cheaper than tg-memory staging + two barriers).
    float q_reg[MAX_DH];
    for (uint d = 0; d < head_dim; ++d) q_reg[d] = Q[q_base + d];

    float m_acc = -1e30f;
    float l_acc = 0.0f;
    float o_acc[MAX_DH];
    for (uint d = 0; d < vdh; ++d) o_acc[d] = 0.0f;

    for (uint ki = start + tid; ki < end; ki += tsize) {
        uint k_base = qkv_kv_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, head_dim, k_stride, bhsd);
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d += 4u) {
            if (d + 3u < head_dim) {
                float4 qv = float4(q_reg[d], q_reg[d + 1u], q_reg[d + 2u], q_reg[d + 3u]);
                float4 kv = float4(*(device const __KV__4*)(K + k_base + d));
                dot += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
            } else {
                for (uint dd = d; dd < head_dim; ++dd) dot += q_reg[dd] * float(K[k_base + dd]);
            }
        }
        float s = dot * scale;
        if (softcap_inv > 0.0f) s = precise::tanh(s * softcap_inv) * attn_softcap;
        if (mask_kind == 1u) {
            if (ki > q_offset) s = -1e9f;
        } else if (mask_kind == 2u) {
            if (M[bi * k_stride + ki] < 0.5f) s = -1e9f;
        } else if (mask_kind == 4u) {
            uint lo = q_offset > window ? q_offset - window : 0u;
            if (ki < lo || ki > q_offset) s = -1e9f;
        }
        if (s <= -1.0e9f) continue;
        float m_new = max(m_acc, s);
        float e_old = exp(m_acc - m_new);
        float e_cur = exp(s - m_new);
        l_acc = e_old * l_acc + e_cur;
        uint v_base = qkv_v_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, vdh, k_stride, bhsd);
        for (uint d = 0; d < vdh; ++d) o_acc[d] = e_old * o_acc[d] + e_cur * float(V[v_base + d]);
        m_acc = m_new;
    }

    // Warp-level online-softmax reduction across the 32 lanes: global max, then
    // rescale each lane by exp(m_lane - m_global) and sum l + each o dim.
    // Identical to the tree merge; no threadgroup memory, no barriers.
    float m_g = simd_max(m_acc);
    float resc = exp(m_acc - m_g);          // lanes with no keys (m_acc=-1e30) → 0
    float l_g = simd_sum(l_acc * resc);
    if (tid == 0u) {
        scratch[slot * SLOT + 0u] = m_g;
        scratch[slot * SLOT + 1u] = l_g;
    }
    for (uint d = 0; d < vdh; ++d) {
        float og = simd_sum(o_acc[d] * resc);
        if (tid == 0u) scratch[slot * SLOT + 2u + d] = og;
    }
}"####;

macro_rules! sdpa_decode_m1_partial_variant {
    ($name:expr, $kv:expr) => {
        SDPA_DECODE_M1_PARTIAL_TEMPLATE
            .replace("__NAME__", $name)
            .replace("__KV__", $kv)
    };
}

// ── Head-dim-split flash-decode partial (register-pressure fix) ─────────────
// The KV-split partial above has every lane hold the FULL q_reg[head_dim] +
// o_acc[vdh] (256 regs/thread at D=128) → occupancy is register-capped. This
// variant splits head_dim ACROSS the 32 lanes instead: each lane owns
// `head_dim/32` dims (4 at D=128), all lanes process each key in lockstep, a
// `simd_sum` forms the full q·k score (broadcast to every lane), and each lane
// accumulates only its o-slice — so q_reg[4]+o_acc[4] = 8 regs/thread and the
// final o needs NO cross-lane reduction (m/l are identical on every lane; each
// lane writes its disjoint o dims). K/V reads are coalesced (consecutive lanes →
// consecutive addresses). Requires head_dim % 32 == 0 and vdh % 32 == 0 (caller
// guards; else the KV-split kernel above is used). Port of the llama.cpp
// `flash_attn_ext_vec` thread mapping. `MAX_DPL` = 128/32.
const SDPA_DECODE_M1_PARTIAL_HD_TEMPLATE: &str = r####"kernel void __NAME__(
    device const float* arena_q   [[buffer(0)]],
    device const __KV__* arena_k   [[buffer(1)]],
    device const __KV__* arena_v   [[buffer(2)]],
    device const float* arena_m   [[buffer(3)]],
    device float*       arena_o   [[buffer(4)]],
    constant uint& batch       [[buffer(5)]],
    constant uint& heads       [[buffer(6)]],
    constant uint& head_dim    [[buffer(7)]],
    constant uint& q_stride    [[buffer(8)]],
    constant uint& mask_kind   [[buffer(9)]],
    constant uint& seq_k       [[buffer(10)]],
    constant uint& k_stride    [[buffer(11)]],
    constant uint& bhsd        [[buffer(12)]],
    constant uint& window      [[buffer(13)]],
    constant float& score_scale  [[buffer(14)]],
    constant float& attn_softcap [[buffer(15)]],
    constant SdpaOffsets& byte_offs [[buffer(16)]],
    device float* scratch      [[buffer(17)]],
    constant uint& n_part      [[buffer(18)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tsize [[threads_per_threadgroup]]
) {
    device const float* Q = (device const float*)((device const char*)arena_q + byte_offs.q);
    device const __KV__* K = (device const __KV__*)((device const char*)arena_k + byte_offs.k);
    device const __KV__* V = (device const __KV__*)((device const char*)arena_v + byte_offs.v);
    device const float* M = (device const float*)((device const char*)arena_m + byte_offs.m);

    constexpr uint MAX_DPL = 4u;
    constexpr uint SLOT = 2u + 128u;

    uint part = tgid % n_part;
    uint t    = tgid / n_part;
    uint hi   = t % heads;
    uint bi   = t / heads;
    if (bi >= batch) return;

    uint vdh = (byte_offs.v_head_dim == 0u) ? head_dim : byte_offs.v_head_dim;
    uint slot = (bi * heads + hi) * n_part + part;

    uint chunk = (seq_k + n_part - 1u) / n_part;
    uint start = part * chunk;
    uint end   = min(start + chunk, seq_k);
    if (start >= seq_k) {
        if (tid == 0u) {
            scratch[slot * SLOT + 0u] = -1e30f;
            scratch[slot * SLOT + 1u] = 0.0f;
            for (uint d = 0; d < vdh; ++d) scratch[slot * SLOT + 2u + d] = 0.0f;
        }
        return;
    }

    float scale = (score_scale > 0.0f) ? score_scale : rsqrt(float(head_dim));
    float softcap_inv = (attn_softcap > 0.0f) ? (1.0f / attn_softcap) : 0.0f;
    uint q_offset = seq_k - 1u;
    uint q_base = qkv_q_offset(bi, hi, 0u, heads, 1u, head_dim, q_stride, bhsd);

    // This lane owns dims [d0, d0+dpl) of q/k and [vd0, vd0+vdpl) of v/o.
    uint dpl  = head_dim / 32u;   // head_dim % 32 == 0 guaranteed by the caller
    uint d0   = tid * dpl;
    uint vdpl = vdh / 32u;
    uint vd0  = tid * vdpl;

    float q_reg[MAX_DPL];
    for (uint j = 0; j < dpl; ++j) q_reg[j] = Q[q_base + d0 + j];

    float o_acc[MAX_DPL];
    for (uint j = 0; j < vdpl; ++j) o_acc[j] = 0.0f;
    float m_acc = -1e30f;
    float l_acc = 0.0f;

    // All 32 lanes walk the SAME key range in lockstep; the score reduction is a
    // per-key simd_sum (barrier-free warp op), so m/l stay identical on every lane.
    for (uint ki = start; ki < end; ++ki) {
        uint k_base = qkv_kv_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, head_dim, k_stride, bhsd);
        float pd = 0.0f;
        for (uint j = 0; j < dpl; ++j) pd += q_reg[j] * float(K[k_base + d0 + j]);
        float s = simd_sum(pd) * scale;
        if (softcap_inv > 0.0f) s = precise::tanh(s * softcap_inv) * attn_softcap;
        if (mask_kind == 1u) {
            if (ki > q_offset) s = -1e9f;
        } else if (mask_kind == 2u) {
            if (M[bi * k_stride + ki] < 0.5f) s = -1e9f;
        } else if (mask_kind == 4u) {
            uint lo = q_offset > window ? q_offset - window : 0u;
            if (ki < lo || ki > q_offset) s = -1e9f;
        }
        if (s <= -1.0e9f) continue;
        float m_new = max(m_acc, s);
        float e_old = exp(m_acc - m_new);
        float e_cur = exp(s - m_new);
        l_acc = e_old * l_acc + e_cur;
        uint v_base = qkv_v_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, vdh, k_stride, bhsd);
        for (uint j = 0; j < vdpl; ++j) o_acc[j] = e_old * o_acc[j] + e_cur * float(V[v_base + vd0 + j]);
        m_acc = m_new;
    }

    if (tid == 0u) {
        scratch[slot * SLOT + 0u] = m_acc;
        scratch[slot * SLOT + 1u] = l_acc;
    }
    for (uint j = 0; j < vdpl; ++j) scratch[slot * SLOT + 2u + vd0 + j] = o_acc[j];
}"####;

macro_rules! sdpa_decode_m1_partial_hd_variant {
    ($name:expr, $kv:expr) => {
        SDPA_DECODE_M1_PARTIAL_HD_TEMPLATE
            .replace("__NAME__", $name)
            .replace("__KV__", $kv)
    };
}

// Combine the P partial online-softmax states per (batch, head) → final OUT.
// One threadgroup per (bi,hi); threads parallelize over vdh. m_g/l_g are cheap
// (loop over n_part ≤ ~16) so each thread recomputes them rather than sharing.
const SDPA_DECODE_M1_COMBINE: &str = r####"kernel void sdpa_decode_m1_combine(
    device const float* scratch [[buffer(0)]],
    device float*       arena_o [[buffer(1)]],
    constant uint& batch     [[buffer(2)]],
    constant uint& heads     [[buffer(3)]],
    constant uint& n_part    [[buffer(4)]],
    constant uint& head_dim  [[buffer(5)]],
    constant uint& q_stride  [[buffer(6)]],
    constant uint& bhsd      [[buffer(7)]],
    constant SdpaOffsets& byte_offs [[buffer(8)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tsize [[threads_per_threadgroup]]
) {
    constexpr uint MAX_DH = 128u;
    constexpr uint SLOT = 2u + MAX_DH;
    device float* OUT = (device float*)((device char*)arena_o + byte_offs.o);
    uint hi = tgid % heads;
    uint bi = tgid / heads;
    if (bi >= batch) return;
    uint vdh = (byte_offs.v_head_dim == 0u) ? head_dim : byte_offs.v_head_dim;
    uint base = (bi * heads + hi) * n_part;

    float m_g = -1e30f;
    for (uint p = 0; p < n_part; ++p) m_g = max(m_g, scratch[(base + p) * SLOT + 0u]);
    float l_g = 0.0f;
    for (uint p = 0; p < n_part; ++p) {
        float mp = scratch[(base + p) * SLOT + 0u];
        float lp = scratch[(base + p) * SLOT + 1u];
        l_g += exp(mp - m_g) * lp;
    }
    float inv_l = (l_g > 0.0f) ? (1.0f / l_g) : 0.0f;
    uint o_base = qkv_out_offset(bi, hi, 0u, heads, 1u, vdh, q_stride, bhsd);
    for (uint d = tid; d < vdh; d += tsize) {
        float acc = 0.0f;
        for (uint p = 0; p < n_part; ++p) {
            float mp = scratch[(base + p) * SLOT + 0u];
            float op = scratch[(base + p) * SLOT + 2u + d];
            acc += exp(mp - m_g) * op;
        }
        OUT[o_base + d] = acc * inv_l;
    }
}"####;

// ── W8A8 decode attention (int8 Q·K integer dot + int8 V) ───────────────────
// Quantize one KV row to int8 (contiguous kv-major) + a per-row f32 absmax
// scale, for the W8A8 flash-decode path. One thread per row;
// gid = (bi*kv_heads + hkv)*seq + ki. The arena K/V may be f32 or f16 (__KV__);
// the int8 output layout is always contiguous [row][dh] so the partial reads it
// independently of the arena's BSNH/BHSD layout. `src_off` is a BYTE offset.
const KV_QUANT_I8_TEMPLATE: &str = r####"kernel void __NAME__(
    device const void*  arena_raw [[buffer(0)]],
    device char*        i8out     [[buffer(1)]],
    device float*       scout     [[buffer(2)]],
    constant ulong& src_off       [[buffer(3)]],
    constant uint&  nrows         [[buffer(4)]],
    constant uint&  dh            [[buffer(5)]],
    constant uint&  bhsd          [[buffer(6)]],
    constant uint&  kv_heads      [[buffer(7)]],
    constant uint&  seq           [[buffer(8)]],
    constant uint&  kstride        [[buffer(9)]],
    constant uint&  blk           [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= nrows) return;
    uint ki  = gid % seq;
    uint bh  = gid / seq;
    uint hkv = bh % kv_heads;
    uint bi  = bh / kv_heads;
    device const __KV__* src = (device const __KV__*)((device const char*)arena_raw + src_off);
    uint aoff;
    if (bhsd != 0u) aoff = (bi * kv_heads + hkv) * seq * dh + ki * dh;
    else            aoff = bi * kstride * kv_heads * dh + ki * kv_heads * dh + hkv * dh;
    // blk=0: one absmax scale per `dh` row. blk=1: one scale per 32-element
    // sub-block (Q8_0-style) → ~4-8× lower rounding error at 4 scales/row.
    uint nb = (blk != 0u) ? (dh / 32u) : 1u;
    uint bs = (blk != 0u) ? 32u : dh;
    device char* dst = i8out + (ulong)gid * dh;
    for (uint b = 0; b < nb; ++b) {
        float amax = 0.0f;
        for (uint j = 0; j < bs; ++j) amax = max(amax, fabs(float(src[aoff + b * bs + j])));
        float s = amax * (1.0f / 127.0f);
        scout[(ulong)gid * nb + b] = s;
        float inv = (s > 1e-20f) ? (1.0f / s) : 0.0f;
        for (uint j = 0; j < bs; ++j)
            dst[b * bs + j] = char(clamp(round(float(src[aoff + b * bs + j]) * inv), -127.0f, 127.0f));
    }
}"####;

macro_rules! kv_quant_i8_variant {
    ($name:expr, $kv:expr) => {
        KV_QUANT_I8_TEMPLATE
            .replace("__NAME__", $name)
            .replace("__KV__", $kv)
    };
}

// W8A8 flash-decode partial: identical online-softmax / warp-reduction / scratch
// layout to `sdpa_decode_m1_partial` (so the SAME combine kernel merges it), but
// K/V come from the int8 scratch (contiguous kv-major) + per-row scales, and the
// query row is quantized to int8 in-kernel so the Q·K score is an INTEGER dot
// (int8×int8→int32 on the int ALU — ~1.6× the f32 dot on M-series). Score =
// int_dot * qscale * kscale * scale; V is dequantized per element (int8 * vscale).
// Gated by RLX_METAL_W8A8_ATTN; f32 KV only (arena K/V pre-quantized by kv_quant_i8).
const SDPA_DECODE_M1_PARTIAL_W8A8: &str = r####"kernel void sdpa_decode_m1_partial_w8a8(
    device const float* arena_q [[buffer(0)]],
    device const char*  i8k     [[buffer(1)]],
    device const char*  i8v     [[buffer(2)]],
    device const float* arena_m [[buffer(3)]],
    device const float* ksc     [[buffer(4)]],
    device const float* vsc     [[buffer(5)]],
    constant uint& batch        [[buffer(6)]],
    constant uint& heads        [[buffer(7)]],
    constant uint& head_dim     [[buffer(8)]],
    constant uint& q_stride     [[buffer(9)]],
    constant uint& mask_kind    [[buffer(10)]],
    constant uint& seq_k        [[buffer(11)]],
    constant uint& k_stride     [[buffer(12)]],
    constant uint& bhsd         [[buffer(13)]],
    constant uint& window       [[buffer(14)]],
    constant float& score_scale  [[buffer(15)]],
    constant float& attn_softcap [[buffer(16)]],
    constant SdpaOffsets& byte_offs [[buffer(17)]],
    device float* scratch       [[buffer(18)]],
    constant uint& packed       [[buffer(19)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tsize [[threads_per_threadgroup]]
) {
    // n_part + mode flags packed into one slot: Metal's arg-table aliases
    // individual set_bytes past index 16, so separate flag slots (20/21/22)
    // clobber each other. Unpack: n_part = low 16b; q_i8/v_i8/blk = bits 16/17/18.
    uint n_part = packed & 0xFFFFu;
    uint q_i8   = (packed >> 16) & 1u;
    uint v_i8   = (packed >> 17) & 1u;
    uint blk    = (packed >> 18) & 1u;
    uint k_i8   = (packed >> 19) & 1u;
    device const float* Q = (device const float*)((device const char*)arena_q + byte_offs.q);
    device const float* M = (device const float*)((device const char*)arena_m + byte_offs.m);
    // {k,v}_i8==0: read K/V straight from the f32 arena (exact) instead of the int8
    // scratch — isolates each operand's quant error. arena_q is the arena base.
    device const float* Vf = (device const float*)((device const char*)arena_q + byte_offs.v);
    device const float* Kf = (device const float*)((device const char*)arena_q + byte_offs.k);
    constexpr uint MAX_DH = 128u;
    constexpr uint SLOT = 2u + MAX_DH;
    uint part = tgid % n_part;
    uint t    = tgid / n_part;
    uint hi   = t % heads;
    uint bi   = t / heads;
    if (bi >= batch) return;
    uint vdh = (byte_offs.v_head_dim == 0u) ? head_dim : byte_offs.v_head_dim;
    uint slot = (bi * heads + hi) * n_part + part;
    uint chunk = (seq_k + n_part - 1u) / n_part;
    uint start = part * chunk;
    uint end   = min(start + chunk, seq_k);
    if (start >= seq_k) {
        if (tid == 0u) {
            scratch[slot * SLOT + 0u] = -1e30f;
            scratch[slot * SLOT + 1u] = 0.0f;
            for (uint d = 0; d < vdh; ++d) scratch[slot * SLOT + 2u + d] = 0.0f;
        }
        return;
    }
    float scale = (score_scale > 0.0f) ? score_scale : rsqrt(float(head_dim));
    float softcap_inv = (attn_softcap > 0.0f) ? (1.0f / attn_softcap) : 0.0f;
    uint q_offset = seq_k - 1u;
    uint q_base = qkv_q_offset(bi, hi, 0u, heads, 1u, head_dim, q_stride, bhsd);

    // Sub-block layout (blk=1 → 32-elem blocks, one scale each; blk=0 → whole row).
    uint nb = (blk != 0u) ? (head_dim / 32u) : 1u;
    uint bs = (blk != 0u) ? 32u : head_dim;
    uint vnb = (blk != 0u) ? (vdh / 32u) : 1u;

    // Load the query row; quantize to int8 per-block when q_i8 (else f32 Q is kept
    // and dotted against dequantized int8 K — isolates whether int8-Q matters).
    float q_reg[MAX_DH];
    for (uint d = 0; d < head_dim; ++d) q_reg[d] = Q[q_base + d];
    float qs_b[4];
    char qi[MAX_DH];
    if (q_i8 != 0u) {
        for (uint b = 0; b < nb; ++b) {
            float am = 0.0f;
            for (uint j = 0; j < bs; ++j) am = max(am, fabs(q_reg[b * bs + j]));
            qs_b[b] = am * (1.0f / 127.0f);
            float inv = (qs_b[b] > 1e-20f) ? (1.0f / qs_b[b]) : 0.0f;
            for (uint j = 0; j < bs; ++j)
                qi[b * bs + j] = char(clamp(round(q_reg[b * bs + j] * inv), -127.0f, 127.0f));
        }
    }

    // GQA kv head + contiguous int8 row base (matches kv_quant_i8's gid layout).
    uint nkv   = (byte_offs.kv_heads == 0u) ? heads : byte_offs.kv_heads;
    uint group = heads / nkv;
    uint hkv   = (group > 1u) ? (hi / group) : hi;
    uint rb    = (bi * nkv + hkv) * seq_k;

    float m_acc = -1e30f;
    float l_acc = 0.0f;
    float o_acc[MAX_DH];
    for (uint d = 0; d < vdh; ++d) o_acc[d] = 0.0f;

    for (uint ki = start + tid; ki < end; ki += tsize) {
        device const char* kr = i8k + (ulong)(rb + ki) * head_dim;
        ulong kscrow = (ulong)(rb + ki) * nb;
        uint k_base = (k_i8 != 0u)
            ? 0u
            : qkv_kv_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, head_dim, k_stride, bhsd);
        float s = 0.0f;
        // Per-block score. Fast path (int8 Q & int8 K): integer dot × block scales.
        // Otherwise an f32 accumulate so any exact operand (q_i8/k_i8==0) is used
        // directly — lets the all-exact case validate the kernel vs CPU (~1e-5).
        for (uint b = 0; b < nb; ++b) {
            if (q_i8 != 0u && k_i8 != 0u) {
                int idot = 0;
                for (uint j = 0; j < bs; ++j) idot += int(qi[b * bs + j]) * int(kr[b * bs + j]);
                s += float(idot) * qs_b[b] * ksc[kscrow + b];
            } else {
                float bd = 0.0f;
                for (uint j = 0; j < bs; ++j) {
                    uint d = b * bs + j;
                    float qv = (q_i8 != 0u) ? (float(qi[d]) * qs_b[b]) : q_reg[d];
                    float kv = (k_i8 != 0u) ? (float(kr[d]) * ksc[kscrow + b]) : Kf[k_base + d];
                    bd += qv * kv;
                }
                s += bd;
            }
        }
        s *= scale;
        if (softcap_inv > 0.0f) s = precise::tanh(s * softcap_inv) * attn_softcap;
        if (mask_kind == 1u) {
            if (ki > q_offset) s = -1e9f;
        } else if (mask_kind == 2u) {
            if (M[bi * k_stride + ki] < 0.5f) s = -1e9f;
        } else if (mask_kind == 4u) {
            uint lo = q_offset > window ? q_offset - window : 0u;
            if (ki < lo || ki > q_offset) s = -1e9f;
        }
        if (s <= -1.0e9f) continue;
        float m_new = max(m_acc, s);
        float e_old = exp(m_acc - m_new);
        float e_cur = exp(s - m_new);
        l_acc = e_old * l_acc + e_cur;
        // V: int8 (per-block or per-row scale) from scratch, or exact f32 from arena.
        device const char* vr = i8v + (ulong)(rb + ki) * vdh;
        ulong vscrow = (ulong)(rb + ki) * vnb;
        uint v_base = (v_i8 != 0u)
            ? 0u
            : qkv_v_offset(bi, hi, ki, heads, byte_offs.kv_heads, seq_k, vdh, k_stride, bhsd);
        for (uint d = 0; d < vdh; ++d) {
            float vval;
            if (v_i8 != 0u) {
                uint b = (blk != 0u) ? (d >> 5) : 0u;
                vval = float(vr[d]) * vsc[vscrow + b];
            } else {
                vval = Vf[v_base + d];
            }
            o_acc[d] = e_old * o_acc[d] + e_cur * vval;
        }
        m_acc = m_new;
    }

    float m_g  = simd_max(m_acc);
    float resc = exp(m_acc - m_g);
    float l_g  = simd_sum(l_acc * resc);
    if (tid == 0u) {
        scratch[slot * SLOT + 0u] = m_g;
        scratch[slot * SLOT + 1u] = l_g;
    }
    for (uint d = 0; d < vdh; ++d) {
        float og = simd_sum(o_acc[d] * resc);
        if (tid == 0u) scratch[slot * SLOT + 2u + d] = og;
    }
}"####;

// Flash-attention-2 PREFILL kernel (seq_q>1) supporting head_dim ≤ 128 + GQA.
// Fixes the O(seq²) DRAM bottleneck of the scalar `sdpa_long` at qwen3's
// head_dim=128 (which no existing tiled FA kernel handles: sdpa_fa_f32 caps at
// MAX_DH=32, sdpa_fa2 at 64). Each threadgroup owns Br=8 query rows of one
// (batch, head) and streams the K/V sequence in Bc=16 tiles staged to
// threadgroup memory — so each K/V element is read once per 8-query tile, not
// once per query (≈Br× less DRAM traffic). Online softmax, F32 accumulate.
// Uses the sdpa_long/FA arg layout (buffers 0-17, SdpaOffsets @17 → byte
// offsets + kv_heads for GQA + v_head_dim). tg mem @ Br=8,Bc=16,MAX_DH=128:
// Q 4KB + K 8KB + V 8KB + o 4KB + S 0.5KB ≈ 24.5KB < 32KB. Dispatch 128 threads,
// grid (ceil(seq_q/8), heads, batch).
// Shared skeleton for the flash-attention prefill kernels. The staging, online
// softmax, causal early-exit and emit are identical across variants; only the
// score (Q@Kᵀ) and P@V matmuls differ (scalar FMA vs simdgroup-matrix MMA), and
// the tile sizes (Br/Bc/MAX_DH) are parameters — so both kernels are generated
// from this one template via `sdpa_prefill_fa_variant!` instead of being
// hand-duplicated. Placeholders: __NAME__, __SG_DECL__ (extra sig params),
// __EXTRA_TG__ (extra threadgroup buffers), __BR__/__BC__/__MDH__ (tile dims),
// __SCORE_IMPL__ (fills masked/scaled S_tg), __PV_IMPL__ (updates o_row from P).
const SDPA_PREFILL_FA_SKELETON: &str = r####"kernel void __NAME__(
    device const float* arena_q   [[buffer(0)]],
    device const float* arena_k   [[buffer(1)]],
    device const float* arena_v   [[buffer(2)]],
    device const float* arena_m   [[buffer(3)]],
    device float*       arena_o   [[buffer(4)]],
    constant uint& batch       [[buffer(5)]],
    constant uint& seq_q       [[buffer(6)]],
    constant uint& heads       [[buffer(7)]],
    constant uint& head_dim    [[buffer(8)]],
    constant uint& q_stride    [[buffer(9)]],
    constant uint& mask_kind   [[buffer(10)]],
    constant uint& seq_k       [[buffer(11)]],
    constant uint& k_stride    [[buffer(12)]],
    constant uint& bhsd        [[buffer(13)]],
    constant uint& window      [[buffer(14)]],
    constant float& score_scale  [[buffer(15)]],
    constant float& attn_softcap [[buffer(16)]],
    constant SdpaOffsets& byte_offs [[buffer(17)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 tid3  [[thread_position_in_threadgroup]],
    uint3 tsz   [[threads_per_threadgroup]]__SG_DECL__
) {
    uint tid = tid3.x;
    uint tsize = tsz.x;
    device const float* Q = (device const float*)((device const char*)arena_q + byte_offs.q);
    device const float* K = (device const float*)((device const char*)arena_k + byte_offs.k);
    device const float* V = (device const float*)((device const char*)arena_v + byte_offs.v);
    device const float* M = (device const float*)((device const char*)arena_m + byte_offs.m);
    device float* OUT     = (device float*)((device char*)arena_o + byte_offs.o);

    constexpr uint Br = __BR__u;
    constexpr uint Bc = __BC__u;
    constexpr uint MAX_DH = __MDH__u;
    threadgroup float Q_tg[Br * MAX_DH];
    threadgroup float K_tg[Bc * MAX_DH];
    threadgroup float V_tg[Bc * MAX_DH];
    threadgroup float S_tg[Br * Bc];__EXTRA_TG__
    threadgroup float m_row[Br];
    threadgroup float l_row[Br];
    threadgroup float o_row[Br * MAX_DH];
    threadgroup float m_new_tg[Br];
    threadgroup float e_old_tg[Br];

    uint q_tile = tgid.x;
    uint hi     = tgid.y;
    uint bi     = tgid.z;
    if (bi >= batch) return;
    uint q_start = q_tile * Br;

    uint vdh = (byte_offs.v_head_dim == 0u) ? head_dim : byte_offs.v_head_dim;
    float scale = (score_scale > 0.0f) ? score_scale : rsqrt(float(head_dim));
    float softcap_inv = (attn_softcap > 0.0f) ? (1.0f / attn_softcap) : 0.0f;
    uint q_off = seq_k - seq_q;

    for (uint i = tid; i < Br * head_dim; i += tsize) {
        uint qi = i / head_dim, di = i % head_dim;
        uint pos = q_start + qi;
        Q_tg[qi * MAX_DH + di] = (pos < seq_q)
            ? Q[qkv_q_offset(bi, hi, pos, heads, seq_q, head_dim, q_stride, bhsd) + di]
            : 0.0f;
    }
    if (tid < Br) { m_row[tid] = -1e30f; l_row[tid] = 0.0f; }
    for (uint i = tid; i < Br * vdh; i += tsize) o_row[(i / vdh) * MAX_DH + (i % vdh)] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kt_end = seq_k;
    if (mask_kind == 1u) kt_end = min(seq_k, q_off + q_start + Br);

    for (uint kt = 0; kt < kt_end; kt += Bc) {
        for (uint i = tid; i < Bc * head_dim; i += tsize) {
            uint ki = i / head_dim, di = i % head_dim;
            uint pos = kt + ki;
            bool ok = pos < seq_k;
            uint koff = ok ? qkv_kv_offset(bi, hi, pos, heads, byte_offs.kv_heads, seq_k, head_dim, k_stride, bhsd) : 0u;
            K_tg[ki * MAX_DH + di] = ok ? K[koff + di] : 0.0f;
        }
        for (uint i = tid; i < Bc * vdh; i += tsize) {
            uint ki = i / vdh, di = i % vdh;
            uint pos = kt + ki;
            bool ok = pos < seq_k;
            uint voff = ok ? qkv_v_offset(bi, hi, pos, heads, byte_offs.kv_heads, seq_k, vdh, k_stride, bhsd) : 0u;
            V_tg[ki * MAX_DH + di] = ok ? V[voff + di] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

__SCORE_IMPL__

        // Online softmax: (A) running max, (B) P=exp(S-m), (C) row-sum l.
        if (tid < Br) {
            uint qi = tid;
            float m_prev = m_row[qi];
            float m_new = m_prev;
            for (uint ki = 0; ki < Bc; ++ki) m_new = max(m_new, S_tg[qi * Bc + ki]);
            m_new_tg[qi] = m_new;
            e_old_tg[qi] = exp(m_prev - m_new);
            m_row[qi] = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = tid; c < Br * Bc; c += tsize) {
            uint qi = c / Bc;
            S_tg[c] = exp(S_tg[c] - m_new_tg[qi]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < Br) {
            uint qi = tid;
            float lsum = 0.0f;
            for (uint ki = 0; ki < Bc; ++ki) lsum += S_tg[qi * Bc + ki];
            l_row[qi] = e_old_tg[qi] * l_row[qi] + lsum;
        }

__PV_IMPL__
    }

    for (uint i = tid; i < Br * vdh; i += tsize) {
        uint qi = i / vdh, di = i % vdh;
        uint pos = q_start + qi;
        if (pos < seq_q) {
            float l = l_row[qi];
            float o = (l > 0.0f) ? (o_row[qi * MAX_DH + di] / l) : 0.0f;
            OUT[qkv_out_offset(bi, hi, pos, heads, seq_q, vdh, q_stride, bhsd) + di] = o;
        }
    }
}"####;

// Scalar score: fused Q·K dot + scale + mask → S_tg.
const FA_SCORE_SCALAR: &str = r####"        for (uint c = tid; c < Br * Bc; c += tsize) {
            uint qi = c / Bc, ki = c % Bc;
            uint qpos = q_start + qi, kpos = kt + ki;
            float s;
            if (qpos < seq_q && kpos < seq_k) {
                float dot = 0.0f;
                for (uint di = 0; di < head_dim; ++di) dot += Q_tg[qi * MAX_DH + di] * K_tg[ki * MAX_DH + di];
                s = dot * scale;
                if (softcap_inv > 0.0f) s = precise::tanh(s * softcap_inv) * attn_softcap;
                if (mask_kind == 1u) { if (kpos > q_off + qpos) s = -1e9f; }
                else if (mask_kind == 2u) { if (M[bi * k_stride + kpos] < 0.5f) s = -1e9f; }
                else if (mask_kind == 4u) { uint hp = q_off + qpos; uint lo = hp > window ? hp - window : 0u; if (kpos < lo || kpos > hp) s = -1e9f; }
            } else { s = -1e9f; }
            S_tg[c] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);"####;

// MMA score: simdgroups 0..Bc/8 compute raw Q@Kᵀ via tensor units → S_tg, then
// scalar scale+mask. Bc/8 n-tiles; one simdgroup each.
const FA_SCORE_MMA: &str = r####"        if (sgid < (Bc / 8u)) {
            simdgroup_float8x8 sacc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint kd = 0; kd < head_dim; kd += 8u) {
                simdgroup_float8x8 qa, kb;
                simdgroup_load(qa, Q_tg + kd, MAX_DH);
                simdgroup_load(kb, K_tg + sgid * 8u * MAX_DH + kd, MAX_DH, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sacc, qa, kb, sacc);
            }
            simdgroup_store(sacc, S_tg + sgid * 8u, Bc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = tid; c < Br * Bc; c += tsize) {
            uint qi = c / Bc, ki = c % Bc;
            uint qpos = q_start + qi, kpos = kt + ki;
            float s = S_tg[c] * scale;
            if (softcap_inv > 0.0f) s = precise::tanh(s * softcap_inv) * attn_softcap;
            if (!(qpos < seq_q && kpos < seq_k)) { s = -1e9f; }
            else if (mask_kind == 1u) { if (kpos > q_off + qpos) s = -1e9f; }
            else if (mask_kind == 2u) { if (M[bi * k_stride + kpos] < 0.5f) s = -1e9f; }
            else if (mask_kind == 4u) { uint hp = q_off + qpos; uint lo = hp > window ? hp - window : 0u; if (kpos < lo || kpos > hp) s = -1e9f; }
            S_tg[c] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);"####;

// Scalar P@V: each thread accumulates its o_row dims (rescaled by e_old) over Bc.
const FA_PV_SCALAR: &str = r####"        for (uint i = tid; i < Br * vdh; i += tsize) {
            uint qi = i / vdh, di = i % vdh;
            float o = o_row[qi * MAX_DH + di] * e_old_tg[qi];
            for (uint ki = 0; ki < Bc; ++ki) o += S_tg[qi * Bc + ki] * V_tg[ki * MAX_DH + di];
            o_row[qi * MAX_DH + di] = o;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);"####;

// MMA P@V: 16 n-tiles (MAX_DH/8) striped across the 4 simdgroups → PV_tg, then
// scalar per-row rescale o = o·e_old + PV.
const FA_PV_MMA: &str = r####"        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint mm = 0; mm < (MAX_DH / 32u); ++mm) {
            uint n = (sgid + mm * 4u) * 8u;
            if (n < vdh) {
                simdgroup_float8x8 pacc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (uint kk = 0; kk < Bc; kk += 8u) {
                    simdgroup_float8x8 pa, vb;
                    simdgroup_load(pa, S_tg + kk, Bc);
                    simdgroup_load(vb, V_tg + kk * MAX_DH + n, MAX_DH);
                    simdgroup_multiply_accumulate(pacc, pa, vb, pacc);
                }
                simdgroup_store(pacc, PV_tg + n, MAX_DH);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < Br * vdh; i += tsize) {
            uint qi = i / vdh, di = i % vdh;
            o_row[qi * MAX_DH + di] = o_row[qi * MAX_DH + di] * e_old_tg[qi] + PV_tg[qi * MAX_DH + di];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);"####;

/// Generate one prefill flash-attention kernel from the shared skeleton with
/// the given name, extra signature/threadgroup decls, tile sizes, and score/PV
/// matmul implementations.
macro_rules! sdpa_prefill_fa_variant {
    ($name:expr, $sg_decl:expr, $extra_tg:expr, $br:expr, $bc:expr, $mdh:expr, $score:expr, $pv:expr) => {
        SDPA_PREFILL_FA_SKELETON
            .replace("__SCORE_IMPL__", $score)
            .replace("__PV_IMPL__", $pv)
            .replace("__NAME__", $name)
            .replace("__SG_DECL__", $sg_decl)
            .replace("__EXTRA_TG__", $extra_tg)
            .replace("__BR__", $br)
            .replace("__BC__", $bc)
            .replace("__MDH__", $mdh)
    };
}

/// Mid-axis concat segment, parameterized by input (`__IN__`) and output
/// (`__OUT__`) element type. The write casts `(__OUT__)(src)`, so a mismatched
/// pair converts precision (f32↔f16) instead of reinterpreting raw bytes.
const CONCAT_MIDAXIS_SEG_TMPL: &str = r####"kernel void __NAME__(
    device char* arena       [[buffer(0)]],
    constant ulong& dst_byte [[buffer(1)]],
    constant ulong& src_byte [[buffer(2)]],
    constant uint& outer     [[buffer(3)]],
    constant uint& dst_axis  [[buffer(4)]],
    constant uint& src_axis  [[buffer(5)]],
    constant uint& inner     [[buffer(6)]],
    constant uint& dst_col   [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = outer * src_axis * inner;
    if (gid >= total) return;
    uint ii = gid % inner;
    uint tmp = gid / inner;
    uint a = tmp % src_axis;
    uint o = tmp / src_axis;
    device const __IN__* src = (device const __IN__*)(arena + src_byte);
    device __OUT__* dst      = (device __OUT__*)(arena + dst_byte);
    dst[(o * dst_axis + dst_col + a) * inner + ii] = (__OUT__)(src[(o * src_axis + a) * inner + ii]);
}
"####;

/// Generate one concat-midaxis kernel `$name` reading `$in`, writing `$out`.
macro_rules! concat_midaxis_variant {
    ($name:expr, $in:expr, $out:expr) => {
        CONCAT_MIDAXIS_SEG_TMPL
            .replace("__NAME__", $name)
            .replace("__IN__", $in)
            .replace("__OUT__", $out)
    };
}

pub(crate) fn msl_source() -> String {
    // Substitute the generated dispatch/kernels at their markers. The backward
    // derivative dispatch must be defined before the `activation_backward` kernel
    // that calls it; the in-place activation kernels are standalone entry points.
    let core = RLX_KERNELS_MSL
        .replace("// @@RLX_SCALAR_ACT_FNS@@", &scalar_act_fns_msl())
        .replace("// @@RLX_POW_SCALAR_FN@@", &pow_scalar_fn_msl())
        .replace(
            "// @@RLX_ACTIVATION_BACKWARD@@",
            &rlxsl::msl_activation_backward_module(),
        )
        .replace("// @@RLX_ACT_INPLACE_H@@", &core_act_inplace_h_msl())
        .replace("// @@RLX_GELU_INPLACE_F32@@", &gelu_inplace_f32_msl())
        .replace("// @@RLX_BINARY_FN@@", &rlxsl::binary::msl_binary_module())
        .replace(
            "// @@RLX_COMPARE_FN@@",
            &rlxsl::compare::msl_compare_module(),
        )
        .replace(
            "// @@RLX_SDPA_DECODE_M1@@",
            &format!(
                "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
                sdpa_decode_m1_variant!("sdpa_decode_m1", "float"),
                sdpa_decode_m1_variant!("sdpa_decode_m1_f16kv", "half"),
                sdpa_decode_m1_partial_variant!("sdpa_decode_m1_partial", "float"),
                sdpa_decode_m1_partial_variant!("sdpa_decode_m1_partial_f16kv", "half"),
                sdpa_decode_m1_partial_hd_variant!("sdpa_decode_m1_partial_hd", "float"),
                sdpa_decode_m1_partial_hd_variant!("sdpa_decode_m1_partial_hd_f16kv", "half"),
                SDPA_DECODE_M1_COMBINE,
                sdpa_prefill_fa_variant!(
                    "sdpa_prefill_fa",
                    "",
                    "",
                    "8",
                    "16",
                    "128",
                    FA_SCORE_SCALAR,
                    FA_PV_SCALAR
                ),
                sdpa_prefill_fa_variant!(
                    "sdpa_prefill_fa_mma",
                    ",\n    uint sgid   [[simdgroup_index_in_threadgroup]]",
                    "\n    threadgroup float PV_tg[Br * MAX_DH];",
                    "8",
                    "16",
                    "128",
                    FA_SCORE_MMA,
                    FA_PV_MMA
                ),
                kv_quant_i8_variant!("kv_quant_i8", "float"),
                kv_quant_i8_variant!("kv_quant_i8_f16", "half"),
                SDPA_DECODE_M1_PARTIAL_W8A8,
            ),
        )
        .replace(
            "// @@RLX_CONCAT_MIDAXIS@@",
            &format!(
                "{}\n{}\n{}\n{}",
                concat_midaxis_variant!("concat_midaxis_seg", "float", "float"),
                concat_midaxis_variant!("concat_midaxis_seg_h", "half", "half"),
                concat_midaxis_variant!("concat_midaxis_seg_f32_to_f16", "float", "half"),
                concat_midaxis_variant!("concat_midaxis_seg_f16_to_f32", "half", "float"),
            ),
        )
        .replace(
            "// @@RLX_SGEMM_TILES@@",
            // 64×64 tile config (TROWS=64 → 8 simdgroups, NACC=8 → TCOLS=64).
            // Add a config (e.g. 64×128: trows=64, nacc=16) in one line + a pipeline.
            &format!(
                "{}\n{}",
                sgemm_tile_variant!("sgemm_simd64", 64, 8),
                sgemm_splitk_variant!("sgemm_simd64_splitk", 64, 8),
            ),
        );
    format!(
        "{core}\n{RLX_KERNELS_MSL_DEQUANT}\n{RLX_KERNELS_MSL_FFT_GPU}\n{RLX_KERNELS_MSL_SPLAT}\n{RLX_KERNELS_MSL_SPLAT_CONIC}\n{}\n{}",
        scalar_act_msl(),
        synth_matmul_msl()
    )
}

pub struct Kernels {
    pub library: Library,
    pub sgemm: ComputePipelineState,
    pub sgemm_f16a: ComputePipelineState,
    pub sgemm_f16w: ComputePipelineState,
    pub sgemm_f16w_small_m: ComputePipelineState,
    pub sgemm_simd: ComputePipelineState,
    pub sgemm_simd_bias: ComputePipelineState,
    pub sgemm_simd_4x4: ComputePipelineState,
    pub sgemm_simd64: ComputePipelineState,
    pub sgemm_simd64_splitk: ComputePipelineState,
    pub zero_f32: ComputePipelineState,
    pub sgemm_simd_4x4_bias: ComputePipelineState,
    pub hgemm_simd_4x4: ComputePipelineState,
    pub hgemm_simd_4x4_bias: ComputePipelineState,
    pub bias_add_h: ComputePipelineState,
    pub gelu_inplace_h: ComputePipelineState,
    pub gelu_approx_inplace_h: ComputePipelineState,
    pub silu_inplace_h: ComputePipelineState,
    pub relu_inplace_h: ComputePipelineState,
    pub sigmoid_inplace_h: ComputePipelineState,
    pub tanh_inplace_h: ComputePipelineState,
    pub exp_inplace_h: ComputePipelineState,
    pub log_inplace_h: ComputePipelineState,
    pub sqrt_inplace_h: ComputePipelineState,
    pub rsqrt_inplace_h: ComputePipelineState,
    pub rec_inplace_h: ComputePipelineState,
    pub neg_inplace_h: ComputePipelineState,
    pub abs_inplace_h: ComputePipelineState,
    pub sin_inplace_h: ComputePipelineState,
    pub cos_inplace_h: ComputePipelineState,
    pub tan_inplace_h: ComputePipelineState,
    pub atan_inplace_h: ComputePipelineState,
    pub round_inplace_h: ComputePipelineState,
    pub layer_norm_h: ComputePipelineState,
    pub fused_residual_ln_h: ComputePipelineState,
    pub fused_residual_rms_norm_h: ComputePipelineState,
    pub rms_norm_h: ComputePipelineState,
    pub softmax_lastax_h: ComputePipelineState,
    pub reduce_axes_h: ComputePipelineState,
    pub elem_add_h: ComputePipelineState,
    pub elem_mul_h: ComputePipelineState,
    pub elem_sub_h: ComputePipelineState,
    pub elem_div_h: ComputePipelineState,
    pub elem_max_h: ComputePipelineState,
    pub elem_min_h: ComputePipelineState,
    pub elem_pow_h: ComputePipelineState,
    pub gather_axis0_h: ComputePipelineState,
    pub narrow_lastax_h: ComputePipelineState,
    pub sdpa_h: ComputePipelineState,
    /// Native f32 fused-attention core for `Op::FusedAttentionBlock`.
    pub fused_attn_block: ComputePipelineState,
    pub rope_h: ComputePipelineState,
    pub cast_f32_to_f16: ComputePipelineState,
    pub cast_f16_to_f32: ComputePipelineState,
    pub copy_f32: ComputePipelineState,
    pub copy4: ComputePipelineState,
    pub sgemm_simd_padded: ComputePipelineState,
    pub sgemm_simd_padded_residual: ComputePipelineState,
    pub sgemm_simd_padded_f16w: ComputePipelineState,
    pub sgemm_wide8x64_f16w: ComputePipelineState,
    pub gemv_f16w_splitk: ComputePipelineState,
    pub gemv_f16w_kpart: ComputePipelineState,
    pub gemv_zero_f32: ComputePipelineState,
    pub sgemm_simd_padded_bias: ComputePipelineState,
    pub sgemm_tiled: ComputePipelineState,
    pub bias_add: ComputePipelineState,
    pub gelu_inplace: ComputePipelineState,
    pub gelu_inplace4: ComputePipelineState,
    pub gelu_approx_inplace: ComputePipelineState,
    pub gelu_approx_inplace4: ComputePipelineState,
    pub gelu_approx_out4: ComputePipelineState,
    pub silu_inplace: ComputePipelineState,
    pub silu_inplace4: ComputePipelineState,
    pub silu_out4: ComputePipelineState,
    pub binary_broadcast_rhs_col_f32: ComputePipelineState,
    pub binary_broadcast_rhs_col4: ComputePipelineState,
    pub binary_broadcast_rhs_row_f32: ComputePipelineState,
    pub binary_broadcast_rhs_row4: ComputePipelineState,
    pub binary_broadcast_rhs_scalar_f32: ComputePipelineState,
    pub binary_broadcast_rhs_scalar4: ComputePipelineState,
    pub binary_broadcast_1ax_f32: ComputePipelineState,
    pub binary_broadcast_1ax4: ComputePipelineState,
    pub fused_binary_activation_f32: ComputePipelineState,
    pub fused_binary_activation4: ComputePipelineState,
    pub fused_ternary_activation_f32: ComputePipelineState,
    pub fused_ternary_activation4: ComputePipelineState,
    pub layer_norm: ComputePipelineState,
    pub rms_norm: ComputePipelineState,
    pub rms_norm_mul_silu: ComputePipelineState,
    pub elem_add: ComputePipelineState,
    pub elem_add4: ComputePipelineState,
    pub elem_sub4: ComputePipelineState,
    pub binary_broadcast_f32: ComputePipelineState,
    pub binary_broadcast_rank2_f32: ComputePipelineState,
    pub binary_broadcast_rank24: ComputePipelineState,
    pub elem_mul: ComputePipelineState,
    pub elem_mul4: ComputePipelineState,
    pub elem_div4: ComputePipelineState,
    pub gather_axis0: ComputePipelineState,
    pub narrow_lastax: ComputePipelineState,
    pub narrow_lastax4: ComputePipelineState,
    pub split_lastax: ComputePipelineState,
    pub split_lastax4: ComputePipelineState,
    pub fused_residual_ln: ComputePipelineState,
    pub fused_residual_rms_norm: ComputePipelineState,
    pub ada_layer_norm: ComputePipelineState,
    pub ada_layer_norm_h: ComputePipelineState,
    pub gated_residual: ComputePipelineState,
    pub gated_residual_h: ComputePipelineState,
    pub ada_layer_norm_backward: ComputePipelineState,
    pub ada_layer_norm_backward_h: ComputePipelineState,
    pub gated_residual_backward: ComputePipelineState,
    pub gated_residual_backward_h: ComputePipelineState,
    pub sdpa: ComputePipelineState,
    pub sdpa_simd: ComputePipelineState,
    pub sdpa_simd_h16: ComputePipelineState,
    pub sdpa_long: ComputePipelineState,
    pub sdpa_long_occpad: ComputePipelineState,
    pub sdpa_decode_m1: ComputePipelineState,
    pub sdpa_decode_m1_f16kv: ComputePipelineState,
    pub sdpa_decode_m1_partial: ComputePipelineState,
    pub sdpa_decode_m1_partial_f16kv: ComputePipelineState,
    /// Head-dim-split flash-decode partials (`head_dim % 32 == 0`): split D
    /// across the 32 lanes → 8 regs/thread vs 256, no cross-lane o reduction,
    /// coalesced K/V. Off-switch RLX_METAL_SDPA_HDSPLIT=0.
    pub sdpa_decode_m1_partial_hd: ComputePipelineState,
    pub sdpa_decode_m1_partial_hd_f16kv: ComputePipelineState,
    pub sdpa_decode_m1_combine: ComputePipelineState,
    pub sdpa_decode_m1_partial_w8a8: ComputePipelineState,
    pub kv_quant_i8: ComputePipelineState,
    pub kv_quant_i8_f16: ComputePipelineState,
    pub sdpa_prefill_fa: ComputePipelineState,
    pub sdpa_prefill_fa_mma: ComputePipelineState,
    pub sdpa_fa_f32: ComputePipelineState,
    pub sdpa_splitk: ComputePipelineState,
    pub sdpa_fa2: ComputePipelineState,
    pub sdpa_mma: ComputePipelineState,
    pub argreduce: ComputePipelineState,
    /// Cooperative last-axis ArgMax/ArgMin (one threadgroup per row) — used
    /// for `inner == 1` (decode logits) instead of the serial `argreduce`.
    pub argreduce_lastaxis: ComputePipelineState,
    /// Fused nearest-codebook assignment (`Op::Custom("rlx.vq_assign")`).
    pub vq_assign: ComputePipelineState,
    /// On-GPU temperature/top-k/top-p/Philox logit sampler (one threadgroup
    /// per batch row). Replaces the unified-memory host fallback.
    pub sample_logits: ComputePipelineState,
    pub rng_normal_philox: ComputePipelineState,
    pub rng_uniform_philox: ComputePipelineState,
    pub rng_fill_zero: ComputePipelineState,
    pub dequant_matmul_int8: ComputePipelineState,
    pub dequant_matmul_int4: ComputePipelineState,
    pub dequant_matmul_fp8: ComputePipelineState,
    pub dequant_matmul_nvfp4: ComputePipelineState,
    pub dequant_matmul_mxfp4x2: ComputePipelineState,
    pub dequant_matmul_mlx_gemv: ComputePipelineState,
    pub dequant_matmul_mlx_gemm: ComputePipelineState,
    pub grouped_dequant_matmul_mlx_gemv: ComputePipelineState,
    pub grouped_dequant_matmul_mlx_gemm: ComputePipelineState,
    pub rope: ComputePipelineState,
    pub fused_swiglu: ComputePipelineState,
    pub fused_swiglu_h: ComputePipelineState,
    /// PLAN L2 — interpreted N-ary element-wise region kernel.
    pub elementwise_region: ComputePipelineState,
    /// FKL batch horizontal fusion (one launch; no prologue).
    pub batch_elementwise_region: ComputePipelineState,
    pub fused_swiglu_cast_f32_to_f16: ComputePipelineState,
    pub fused_swiglu_cast_f16_to_f32: ComputePipelineState,
    pub concat_segment_lastax: ComputePipelineState,
    pub concat_segment_lastax4: ComputePipelineState,
    pub concat_lastax_multi: ComputePipelineState,
    pub concat_lastax_multi4: ComputePipelineState,
    pub concat_segment_lastax_h: ComputePipelineState,
    pub concat_midaxis_seg: ComputePipelineState,
    pub concat_midaxis_seg_h: ComputePipelineState,
    pub concat_midaxis_seg_f32_to_f16: ComputePipelineState,
    pub concat_midaxis_seg_f16_to_f32: ComputePipelineState,
    pub elem_sub: ComputePipelineState,
    pub elem_div: ComputePipelineState,
    pub elem_max: ComputePipelineState,
    pub elem_min: ComputePipelineState,
    pub elem_pow: ComputePipelineState,
    pub elem_binop: ComputePipelineState,
    pub elem_binop_h: ComputePipelineState,
    pub elem_compare: ComputePipelineState,
    pub elem_compare_bcast: ComputePipelineState,
    pub elem_where: ComputePipelineState,
    pub elem_where_bcast: ComputePipelineState,
    pub elem_fma: ComputePipelineState,
    pub relu_backward: ComputePipelineState,
    pub activation_backward: ComputePipelineState,
    pub complex_norm_sq: ComputePipelineState,
    pub complex_norm_sq_backward: ComputePipelineState,
    pub conjugate_c64: ComputePipelineState,
    pub fft_butterfly_stage: ComputePipelineState,
    pub fake_quantize_fixed: ComputePipelineState,
    pub fake_quantize_perbatch: ComputePipelineState,
    pub reduce_axes: ComputePipelineState,
    pub reduce_axes_sum_simd: ComputePipelineState,
    /// Double-single (2× f32 ≈ f64) full-sum reduction — high-precision
    /// `Op::Reduce{Sum}` on Metal (opt-in via `RLX_METAL_DW_SUM`).
    pub dw_sum_arena: ComputePipelineState,
    pub topk_lastax: ComputePipelineState,
    pub grouped_matmul: ComputePipelineState,
    pub scatter_add_zero: ComputePipelineState,
    pub scatter_add_accumulate: ComputePipelineState,
    pub transpose_nd: ComputePipelineState,
    pub transpose_nd_h: ComputePipelineState,
    pub transpose_2d_f32: ComputePipelineState,
    pub transpose_2d_tiled_f32: ComputePipelineState,
    pub transpose_last2_batched_f32: ComputePipelineState,
    pub transpose_last2_batched_tiled_f32: ComputePipelineState,
    pub transpose_swap12_batched_trail_f32: ComputePipelineState,
    pub transpose_swap12_batched_trail_tiled_f32: ComputePipelineState,
    pub gather_axis: ComputePipelineState,
    pub pool2d: ComputePipelineState,
    pub maxpool2d_backward: ComputePipelineState,
    pub conv2d_backward_input: ComputePipelineState,
    pub conv2d_backward_weight: ComputePipelineState,
    pub conv2d_backward_weight_partial: ComputePipelineState,
    pub conv2d_backward_weight_reduce: ComputePipelineState,
    pub maxpool3d_backward: ComputePipelineState,
    pub conv3d_backward_input: ComputePipelineState,
    pub conv3d_backward_weight: ComputePipelineState,
    pub conv2d: ComputePipelineState,
    pub depthwise_conv1d_bsc: ComputePipelineState,
    pub conv2d_w1: ComputePipelineState,
    pub layer_norm2d: ComputePipelineState,
    pub group_norm: ComputePipelineState,
    pub resize_nearest_2x: ComputePipelineState,
    pub conv_transpose2d: ComputePipelineState,
    pub conv3d: ComputePipelineState,
    pub conv_transpose3d: ComputePipelineState,
    pub relu_inplace: ComputePipelineState,
    pub sigmoid_inplace: ComputePipelineState,
    pub tanh_inplace: ComputePipelineState,
    pub exp_inplace: ComputePipelineState,
    pub log_inplace: ComputePipelineState,
    pub sqrt_inplace: ComputePipelineState,
    pub rsqrt_inplace: ComputePipelineState,
    pub rec_inplace: ComputePipelineState,
    pub neg_inplace: ComputePipelineState,
    pub abs_inplace: ComputePipelineState,
    pub round_inplace: ComputePipelineState,
    pub sin_inplace: ComputePipelineState,
    pub cos_inplace: ComputePipelineState,
    pub tan_inplace: ComputePipelineState,
    pub atan_inplace: ComputePipelineState,
    pub softmax_lastax: ComputePipelineState,
    pub softmax_lastax_causal: ComputePipelineState,
    pub softmax_cross_entropy_dense: ComputePipelineState,
    pub softmax_cross_entropy_with_logits: ComputePipelineState,
    pub softmax_cross_entropy_backward: ComputePipelineState,
    pub fft_radix2_full_f32: ComputePipelineState,
    /// native-gpu-fft: single-kernel on-chip FFT for n in (1024, 4096].
    #[cfg(feature = "native-gpu-fft")]
    pub fft_radix2_full_big_f32: ComputePipelineState,
    /// native-gpu-fft: radix-4 in-place single-kernel FFT (pow-4 and 2·pow-4).
    #[cfg(feature = "native-gpu-fft")]
    pub fft_radix4_full_f32: ComputePipelineState,
    /// native-gpu-fft: radix-8 in-place single-kernel FFT for pow-8 sizes.
    #[cfg(feature = "native-gpu-fft")]
    pub fft_radix8_full_f32: ComputePipelineState,
    /// native-gpu-fft: radix-16 in-place single-kernel FFT for pow-16 sizes.
    #[cfg(feature = "native-gpu-fft")]
    pub fft_radix16_full_f32: ComputePipelineState,
    pub fft_bit_reverse_f32: ComputePipelineState,
    pub fft_inner_f32: ComputePipelineState,
    pub fft_outer_r4_f32: ComputePipelineState,
    pub fft_outer_r2_f32: ComputePipelineState,
    pub gated_delta_net: ComputePipelineState,
    pub gated_delta_net_sg: ComputePipelineState,
    pub selective_scan: ComputePipelineState,
    pub lstm: ComputePipelineState,
    pub gru: ComputePipelineState,
    pub rnn: ComputePipelineState,
    pub mamba2: ComputePipelineState,
    pub dequant_gguf: ComputePipelineState,
    /// Fused Q4_K_M GEMV — skips the f32 dequant scratch and produces
    /// `dst[n] = sum_k x[k] * dequant(w[n,k])` in a single pass. Used
    /// for `m == 1` (decode) GgufQ4K matmuls; m > 1 still goes through
    /// `dequant_gguf + encode_mps_sgemm_bt`.
    pub q4k_mv_f32: ComputePipelineState,
    /// Fused single-pass Q3_K decode GEMV (`m == 1`); mirrors `q4k_mv_f32`.
    pub q3k_mv_f32: ComputePipelineState,
    /// Fused single-pass Q6_K decode GEMV (`m == 1`) — the Q6_K LM head.
    pub q6k_mv_f32: ComputePipelineState,
    pub q4_0_mv_f32: ComputePipelineState,
    pub q4_1_mv_f32: ComputePipelineState,
    pub q8_0_mv_f32: ComputePipelineState,
    /// Fused Q1_0 (Bonsai-27B 1-bit) GEMV (decode) + GEMM (prefill): read the
    /// packed weight directly, skipping the dequant-to-f32 scratch + MPS sgemm
    /// path (whose shared scratch raced and zeroed large-n Q1_0 projections).
    pub q1_0_mv_f32: ComputePipelineState,
    /// Simdgroup-cooperative Q1_0 GEMV: 32 threads → 8 outputs via `simd_sum`
    /// (llama.cpp `kernel_mul_mv_q1_0_f32`). Used when `n_dim % 8 == 0`.
    pub q1_0_mv_f32_sg: ComputePipelineState,
    pub q1_0_dual_mv_f32_sg: ComputePipelineState,
    pub q1_0_mm_f32: ComputePipelineState,
    pub q2_0_mv_f32: ComputePipelineState,
    pub q2_0_mv_f32_sg: ComputePipelineState,
    pub q2_0_dual_mv_f32_sg: ComputePipelineState,
    pub q2_0_mm_f32: ComputePipelineState,
    pub iq4_nl_mv_f32: ComputePipelineState,
    pub iq2_xxs_mv_f32: ComputePipelineState,
    pub synth_matmul_codebook: ComputePipelineState,
    pub gemm_rb_bias: ComputePipelineState,
    pub synth_reconstruct_nk: ComputePipelineState,
    pub synth_bwd_dx: ComputePipelineState,
    pub synth_bwd_codebook: ComputePipelineState,
    pub synth_matmul_codebook_mm: ComputePipelineState,
    pub synth_matmul_codebook_h: ComputePipelineState,
    pub synth_matmul_codebook_mm_h: ComputePipelineState,
    pub synth_matmul_codebook_tiled: ComputePipelineState,
    pub synth_matmul_codebook_tiled_h: ComputePipelineState,
    pub synth_reconstruct: ComputePipelineState,
    pub synth_reconstruct_h: ComputePipelineState,
    pub spline_activation: ComputePipelineState,
    pub spline_activation_backward_x: ComputePipelineState,
    pub spline_activation_backward_coeff: ComputePipelineState,
    pub iq2_xs_mv_f32: ComputePipelineState,
    pub iq2_s_mv_f32: ComputePipelineState,
    pub iq3_xxs_mv_f32: ComputePipelineState,
    pub iq3_s_mv_f32: ComputePipelineState,
    /// Simdgroup-cooperative IQ3_XXS / IQ3_S GEMV — distributing the codebook
    /// LUT lookups across 32 lanes (the serial bottleneck in the 1-thread
    /// kernels) makes these ~2× Q3_K, second only to Q4_K.
    pub iq3_xxs_mv_f32_sg: ComputePipelineState,
    pub iq3_s_mv_f32_sg: ComputePipelineState,
    pub iq1_s_mv_f32: ComputePipelineState,
    pub iq1_m_mv_f32: ComputePipelineState,
    /// Simdgroup-cooperative Q4_K_M GEMV: 32 threads cooperate on 8
    /// output columns with `simd_sum`. Better x cache reuse than the
    /// single-thread-per-output `q4k_mv_f32`. Used when `n_dim % 8 == 0`.
    pub q4k_mv_f32_sg: ComputePipelineState,
    /// Simdgroup-cooperative Q6_K GEMV (32 threads reduce one output row via
    /// `simd_sum`; `Q6K_NSG` rows per threadgroup) and Q8_0 GEMV (32 threads →
    /// `Q8_0_NR0` rows). Replace the occupancy-starved one-thread-per-row
    /// `q6k_mv_f32` / `q8_0_mv_f32` on decode. Off: RLX_METAL_Q6K_SG_DISABLE /
    /// RLX_METAL_Q8_0_SG_DISABLE.
    pub q6k_mv_f32_sg: ComputePipelineState,
    pub q8_0_mv_f32_sg: ComputePipelineState,
    /// Simdgroup-cooperative Q4_0 / Q4_1 (32 threads → 4 rows) and Q3_K (one
    /// row per simdgroup) decode GEMVs — same simd_sum treatment as Q4_K/Q6_K,
    /// replacing the one-thread-per-row kernels. Off: RLX_METAL_Q40_SG_DISABLE /
    /// RLX_METAL_Q41_SG_DISABLE / RLX_METAL_Q3K_SG_DISABLE.
    pub q4_0_mv_f32_sg: ComputePipelineState,
    pub q4_1_mv_f32_sg: ComputePipelineState,
    pub q3k_mv_f32_sg: ComputePipelineState,
    /// Fused Q4_K / Q6_K GEMM (m > 1, prefill) — reads packed weight directly,
    /// dequants in-register and accumulates a row tile, replacing the
    /// `dequant_gguf` f32 scratch + MPS sgemm path for these two schemes.
    pub q4k_mm_f32: ComputePipelineState,
    pub q6k_mm_f32: ComputePipelineState,
    /// Fused decode-layer MLP GEMVs (m == 1). Q4_K / Q5_0 gate+up + silu/gelu;
    /// Q4_K / Q5_0 / Q6_K down + residual. Produced by `fuse_decode_mlp*`
    /// (off-switch `RLX_METAL_FUSE_DECODE=0`).
    pub q4k_swiglu_mv_f32: ComputePipelineState,
    pub q4k_gelu_mv_f32: ComputePipelineState,
    pub q5_0_swiglu_mv_f32: ComputePipelineState,
    pub q5_0_gelu_mv_f32: ComputePipelineState,
    pub q4k_mv_residual_f32: ComputePipelineState,
    pub q6k_mv_residual_f32: ComputePipelineState,
    pub q5_0_mv_residual_f32: ComputePipelineState,
    /// Fused Q1_0 decode MLP (Bonsai): gate+up+SwiGLU and down+residual.
    pub q1_0_swiglu_mv_f32: ComputePipelineState,
    pub q1_0_swiglu_mv_f32_sg: ComputePipelineState,
    pub q1_0_mv_residual_f32: ComputePipelineState,
    pub q1_0_mv_residual_f32_sg: ComputePipelineState,
    pub q2_0_swiglu_mv_f32: ComputePipelineState,
    pub q2_0_swiglu_mv_f32_sg: ComputePipelineState,
    pub q2_0_mv_residual_f32: ComputePipelineState,
    pub q2_0_mv_residual_f32_sg: ComputePipelineState,
    /// Device buffer holding the concatenated IQ grid LUTs. Built once
    /// at Kernels init from `rlx_gguf::iq_grids::*`. Layout — see
    /// `dequant_gguf.msl` `IQ_GRID_OFF_*` constants.
    iq_grid_lut: Buffer,
    pub rms_norm_bwd: ComputePipelineState,
    pub rms_norm_bwd_param: ComputePipelineState,
    pub rms_norm_bwd_inv_r_f32: ComputePipelineState,
    pub rms_norm_bwd_param_reduce_f32: ComputePipelineState,
    pub layer_norm_bwd: ComputePipelineState,
    pub layer_norm_bwd_gamma: ComputePipelineState,
    pub layer_norm_bwd_stats_f32: ComputePipelineState,
    pub layer_norm_bwd_gamma_reduce_f32: ComputePipelineState,
    pub layer_norm_bwd_gamma_reduce_simd: ComputePipelineState,
    pub group_norm_bwd_input: ComputePipelineState,
    pub group_norm_bwd_gamma: ComputePipelineState,
    pub group_norm_bwd_beta: ComputePipelineState,
    pub rope_bwd: ComputePipelineState,
    pub cumsum_fwd: ComputePipelineState,
    pub cum_scan: ComputePipelineState,
    pub cumsum_bwd: ComputePipelineState,
    pub im2col_group: ComputePipelineState,
    pub im2col_group_w1: ComputePipelineState,
    pub conv2d_bwd_weight_gemm: ComputePipelineState,
    pub conv2d_bwd_weight_gemm_4x4: ComputePipelineState,
    pub attn_bwd_scores_f32: ComputePipelineState,
    pub attn_bwd_dp_f32: ComputePipelineState,
    pub attn_bwd_ds_f32: ComputePipelineState,
    pub attn_bwd_dv_f32: ComputePipelineState,
    pub attn_bwd_dq_f32: ComputePipelineState,
    pub attn_bwd_dk_f32: ComputePipelineState,
    pub attn_bwd_scores_batched_f32: ComputePipelineState,
    pub attn_bwd_dp_batched_f32: ComputePipelineState,
    pub attn_bwd_ds_batched_f32: ComputePipelineState,
    pub attn_bwd_dv_batched_f32: ComputePipelineState,
    pub attn_bwd_dq_batched_f32: ComputePipelineState,
    pub attn_bwd_dk_batched_f32: ComputePipelineState,
    pub attn_bwd_fused_f32: ComputePipelineState,
    pub gather_bwd_zero: ComputePipelineState,
    pub gather_bwd_acc: ComputePipelineState,
    /// Native Gaussian splat tile raster (see `splat.msl`).
    pub gaussian_splat_rasterize: ComputePipelineState,
    /// Training linear radiance raster (no display gamma).
    pub gaussian_splat_rasterize_linear: ComputePipelineState,
    pub gaussian_splat_rasterize_linear_traced: ComputePipelineState,
    pub gaussian_splat_rasterize_backward_linear: ComputePipelineState,
    pub gaussian_splat_adam_step: ComputePipelineState,
    pub gaussian_splat_mse_loss_grad: ComputePipelineState,
    pub gaussian_splat_ssim_stats: ComputePipelineState,
    pub gaussian_splat_blended_loss_grad: ComputePipelineState,
    pub gaussian_splat_project_training: ComputePipelineState,
    pub gaussian_splat_geometry_backward: ComputePipelineState,
    pub gaussian_splat_scene_grad_projection: ComputePipelineState,
    pub gaussian_splat_splat_color_backward: ComputePipelineState,
    pub gaussian_splat_emit_tile_keys: ComputePipelineState,
    pub gaussian_splat_project_screen_ellipse: ComputePipelineState,
    pub gaussian_splat_emit_tile_keys_conic: ComputePipelineState,
    pub gaussian_splat_bin_histogram: ComputePipelineState,
    pub gaussian_splat_bin_copy_counts: ComputePipelineState,
    pub gaussian_splat_bin_prefix_sum: ComputePipelineState,
    pub gaussian_splat_bin_scatter: ComputePipelineState,
    pub gaussian_splat_build_tile_ranges: ComputePipelineState,
    pub gaussian_splat_pack_grads: ComputePipelineState,
    /// Macro-generated scalar activation kernels (Floor/Ceil/Sign/Softplus/Elu):
    /// `Activation → (f32_pipeline, f16_pipeline)`. See `scalar_activation_kernels!`.
    pub scalar_acts: std::collections::HashMap<
        rlx_ir::op::Activation,
        (ComputePipelineState, ComputePipelineState),
    >,
}

unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

/// Double-single (2× f32 ≈ f64) sum reduction over the f32 arena — the
/// high-precision path for `Op::Reduce{Sum}` on Metal (no native f64). Prepended
/// with the `rlxsl::dw` prelude and compiled with PRECISE math (fast-math would
/// break the error-free transforms). See [`Kernels::new`].
const DW_SUM_ARENA_MSL: &str = r#"
// Buffers are bound at the src/dst byte offsets (mirrors encode_reduce_axes),
// so index from 0. Writes the correctly-rounded scalar sum to out[0].
kernel void dw_sum_arena(device const float* x [[buffer(0)]],
                         device float* out      [[buffer(1)]],
                         constant uint& n       [[buffer(2)]],
                         uint tid      [[thread_position_in_threadgroup]],
                         uint nthreads [[threads_per_threadgroup]]) {
    threadgroup DwF32 shared[256];
    DwF32 acc = DwF32{0.0f, 0.0f};
    for (uint i = tid; i < n; i += nthreads) { acc = dw_add(acc, DwF32{x[i], 0.0f}); }
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthreads >> 1; s > 0u; s >>= 1) {
        if (tid < s) { shared[tid] = dw_add(shared[tid], shared[tid + s]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) { out[0] = shared[0].hi + shared[0].lo; }
}
"#;

impl Kernels {
    fn new() -> Self {
        let dev = metal_device().expect("Metal device required");
        if let Some(path) = rlx_ir::env::var("RLX_METAL_DUMP_MSL") {
            let s = msl_source();
            let _ = std::fs::write(&path, &s);
            eprintln!("[msl-dump] wrote {} bytes of MSL source to {path}", s.len());
        }
        let library = crate::pipeline_cache::load_or_compile_library(&dev.device, &msl_source());
        let pipeline = |name: &str| -> ComputePipelineState {
            let f = library.get_function(name, None).expect(name);
            dev.device
                .new_compute_pipeline_state_with_function(&f)
                .unwrap_or_else(|_| panic!("pipeline {name}"))
        };
        // Separate PRECISE-math library for the double-single reduction.
        let dw_sum_arena = {
            let src = format!(
                "#include <metal_stdlib>\nusing namespace metal;\n{}\n{DW_SUM_ARENA_MSL}",
                rlxsl::dw::double_single_prelude(rlxsl::Lang::Msl)
            );
            let opts = metal::CompileOptions::new();
            opts.set_fast_math_enabled(false);
            let lib = dev
                .device
                .new_library_with_source(&src, &opts)
                .expect("rlx-metal: compile dw_sum_arena library");
            let f = lib
                .get_function("dw_sum_arena", None)
                .expect("dw_sum_arena");
            dev.device
                .new_compute_pipeline_state_with_function(&f)
                .expect("rlx-metal: dw_sum_arena pipeline")
        };
        Self {
            dw_sum_arena,
            sgemm: pipeline("sgemm"),
            sgemm_f16a: pipeline("sgemm_f16a"),
            sgemm_f16w: pipeline("sgemm_f16w"),
            sgemm_f16w_small_m: pipeline("sgemm_f16w_small_m"),
            sgemm_simd: pipeline("sgemm_simd"),
            sgemm_simd_bias: pipeline("sgemm_simd_bias"),
            sgemm_simd_4x4: pipeline("sgemm_simd_4x4"),
            sgemm_simd64: pipeline("sgemm_simd64"),
            sgemm_simd64_splitk: pipeline("sgemm_simd64_splitk"),
            zero_f32: pipeline("zero_f32"),
            sgemm_simd_4x4_bias: pipeline("sgemm_simd_4x4_bias"),
            hgemm_simd_4x4: pipeline("hgemm_simd_4x4"),
            hgemm_simd_4x4_bias: pipeline("hgemm_simd_4x4_bias"),
            bias_add_h: pipeline("bias_add_h"),
            gelu_inplace_h: pipeline("gelu_inplace_h"),
            gelu_approx_inplace_h: pipeline("gelu_approx_inplace_h"),
            silu_inplace_h: pipeline("silu_inplace_h"),
            relu_inplace_h: pipeline("relu_inplace_h"),
            sigmoid_inplace_h: pipeline("sigmoid_inplace_h"),
            tanh_inplace_h: pipeline("tanh_inplace_h"),
            exp_inplace_h: pipeline("exp_inplace_h"),
            log_inplace_h: pipeline("log_inplace_h"),
            sqrt_inplace_h: pipeline("sqrt_inplace_h"),
            rsqrt_inplace_h: pipeline("rsqrt_inplace_h"),
            rec_inplace_h: pipeline("rec_inplace_h"),
            neg_inplace_h: pipeline("neg_inplace_h"),
            abs_inplace_h: pipeline("abs_inplace_h"),
            sin_inplace_h: pipeline("sin_inplace_h"),
            cos_inplace_h: pipeline("cos_inplace_h"),
            tan_inplace_h: pipeline("tan_inplace_h"),
            atan_inplace_h: pipeline("atan_inplace_h"),
            round_inplace_h: pipeline("round_inplace_h"),
            layer_norm_h: pipeline("layer_norm_h"),
            fused_residual_ln_h: pipeline("fused_residual_ln_h"),
            fused_residual_rms_norm_h: pipeline("fused_residual_rms_norm_h"),
            rms_norm_h: pipeline("rms_norm_h"),
            softmax_lastax_h: pipeline("softmax_lastax_h"),
            reduce_axes_h: pipeline("reduce_axes_h"),
            elem_add_h: pipeline("elem_add_h"),
            elem_mul_h: pipeline("elem_mul_h"),
            elem_sub_h: pipeline("elem_sub_h"),
            elem_div_h: pipeline("elem_div_h"),
            elem_max_h: pipeline("elem_max_h"),
            elem_min_h: pipeline("elem_min_h"),
            elem_pow_h: pipeline("elem_pow_h"),
            gather_axis0_h: pipeline("gather_axis0_h"),
            narrow_lastax_h: pipeline("narrow_lastax_h"),
            sdpa_h: pipeline("sdpa_h"),
            fused_attn_block: pipeline("fused_attn_block"),
            rope_h: pipeline("rope_h"),
            cast_f32_to_f16: pipeline("cast_f32_to_f16"),
            cast_f16_to_f32: pipeline("cast_f16_to_f32"),
            copy_f32: pipeline("copy_f32"),
            copy4: pipeline("copy4"),
            sgemm_simd_padded: pipeline("sgemm_simd_padded"),
            sgemm_simd_padded_residual: pipeline("sgemm_simd_padded_residual"),
            sgemm_simd_padded_f16w: pipeline("sgemm_simd_padded_f16w"),
            sgemm_wide8x64_f16w: pipeline("sgemm_wide8x64_f16w"),
            gemv_f16w_splitk: pipeline("gemv_f16w_splitk"),
            gemv_f16w_kpart: pipeline("gemv_f16w_kpart"),
            gemv_zero_f32: pipeline("gemv_zero_f32"),
            sgemm_simd_padded_bias: pipeline("sgemm_simd_padded_bias"),
            sgemm_tiled: pipeline("sgemm_tiled"),
            bias_add: pipeline("bias_add"),
            gelu_inplace: pipeline("gelu_inplace"),
            gelu_inplace4: pipeline("gelu_inplace4"),
            gelu_approx_inplace: pipeline("gelu_approx_inplace"),
            gelu_approx_inplace4: pipeline("gelu_approx_inplace4"),
            gelu_approx_out4: pipeline("gelu_approx_out4"),
            silu_inplace: pipeline("silu_inplace"),
            silu_inplace4: pipeline("silu_inplace4"),
            silu_out4: pipeline("silu_out4"),
            binary_broadcast_rhs_col_f32: pipeline("binary_broadcast_rhs_col_f32"),
            binary_broadcast_rhs_col4: pipeline("binary_broadcast_rhs_col4"),
            binary_broadcast_rhs_row_f32: pipeline("binary_broadcast_rhs_row_f32"),
            binary_broadcast_rhs_row4: pipeline("binary_broadcast_rhs_row4"),
            binary_broadcast_rhs_scalar_f32: pipeline("binary_broadcast_rhs_scalar_f32"),
            binary_broadcast_rhs_scalar4: pipeline("binary_broadcast_rhs_scalar4"),
            binary_broadcast_1ax_f32: pipeline("binary_broadcast_1ax_f32"),
            binary_broadcast_1ax4: pipeline("binary_broadcast_1ax4"),
            fused_binary_activation_f32: pipeline("fused_binary_activation_f32"),
            fused_binary_activation4: pipeline("fused_binary_activation4"),
            fused_ternary_activation_f32: pipeline("fused_ternary_activation_f32"),
            fused_ternary_activation4: pipeline("fused_ternary_activation4"),
            layer_norm: pipeline("layer_norm"),
            rms_norm: pipeline("rms_norm"),
            rms_norm_mul_silu: pipeline("rms_norm_mul_silu"),
            elem_add: pipeline("elem_add"),
            elem_add4: pipeline("elem_add4"),
            elem_sub4: pipeline("elem_sub4"),
            binary_broadcast_f32: pipeline("binary_broadcast_f32"),
            binary_broadcast_rank2_f32: pipeline("binary_broadcast_rank2_f32"),
            binary_broadcast_rank24: pipeline("binary_broadcast_rank24"),
            elem_mul: pipeline("elem_mul"),
            elem_mul4: pipeline("elem_mul4"),
            elem_div4: pipeline("elem_div4"),
            gather_axis0: pipeline("gather_axis0"),
            narrow_lastax: pipeline("narrow_lastax"),
            narrow_lastax4: pipeline("narrow_lastax4"),
            split_lastax: pipeline("split_lastax"),
            split_lastax4: pipeline("split_lastax4"),
            fused_residual_ln: pipeline("fused_residual_ln"),
            fused_residual_rms_norm: pipeline("fused_residual_rms_norm"),
            ada_layer_norm: pipeline("ada_layer_norm"),
            ada_layer_norm_h: pipeline("ada_layer_norm_h"),
            gated_residual: pipeline("gated_residual"),
            gated_residual_h: pipeline("gated_residual_h"),
            ada_layer_norm_backward: pipeline("ada_layer_norm_backward"),
            ada_layer_norm_backward_h: pipeline("ada_layer_norm_backward_h"),
            gated_residual_backward: pipeline("gated_residual_backward"),
            gated_residual_backward_h: pipeline("gated_residual_backward_h"),
            sdpa: pipeline("sdpa"),
            sdpa_simd: pipeline("sdpa_simd"),
            sdpa_simd_h16: pipeline("sdpa_simd_h16"),
            sdpa_long: pipeline("sdpa_long"),
            sdpa_long_occpad: pipeline("sdpa_long_occpad"),
            sdpa_decode_m1: pipeline("sdpa_decode_m1"),
            sdpa_decode_m1_f16kv: pipeline("sdpa_decode_m1_f16kv"),
            sdpa_decode_m1_partial: pipeline("sdpa_decode_m1_partial"),
            sdpa_decode_m1_partial_f16kv: pipeline("sdpa_decode_m1_partial_f16kv"),
            sdpa_decode_m1_partial_hd: pipeline("sdpa_decode_m1_partial_hd"),
            sdpa_decode_m1_partial_hd_f16kv: pipeline("sdpa_decode_m1_partial_hd_f16kv"),
            sdpa_decode_m1_combine: pipeline("sdpa_decode_m1_combine"),
            sdpa_decode_m1_partial_w8a8: pipeline("sdpa_decode_m1_partial_w8a8"),
            kv_quant_i8: pipeline("kv_quant_i8"),
            kv_quant_i8_f16: pipeline("kv_quant_i8_f16"),
            sdpa_prefill_fa: pipeline("sdpa_prefill_fa"),
            sdpa_prefill_fa_mma: pipeline("sdpa_prefill_fa_mma"),
            sdpa_fa_f32: pipeline("sdpa_fa_f32"),
            sdpa_splitk: pipeline("sdpa_splitk"),
            sdpa_fa2: pipeline("sdpa_fa2"),
            sdpa_mma: pipeline("sdpa_mma"),
            argreduce: pipeline("argreduce"),
            argreduce_lastaxis: pipeline("argreduce_lastaxis"),
            vq_assign: pipeline("vq_assign"),
            sample_logits: pipeline("sample_logits"),
            rng_normal_philox: pipeline("rng_normal_philox"),
            rng_uniform_philox: pipeline("rng_uniform_philox"),
            rng_fill_zero: pipeline("rng_fill_zero"),
            dequant_matmul_int8: pipeline("dequant_matmul_int8"),
            dequant_matmul_int4: pipeline("dequant_matmul_int4"),
            dequant_matmul_fp8: pipeline("dequant_matmul_fp8"),
            dequant_matmul_nvfp4: pipeline("dequant_matmul_nvfp4"),
            dequant_matmul_mxfp4x2: pipeline("dequant_matmul_mxfp4x2"),
            dequant_matmul_mlx_gemv: pipeline("dequant_matmul_mlx_gemv"),
            dequant_matmul_mlx_gemm: pipeline("dequant_matmul_mlx_gemm"),
            grouped_dequant_matmul_mlx_gemv: pipeline("grouped_dequant_matmul_mlx_gemv"),
            grouped_dequant_matmul_mlx_gemm: pipeline("grouped_dequant_matmul_mlx_gemm"),
            rope: pipeline("rope"),
            fused_swiglu: pipeline("fused_swiglu"),
            fused_swiglu_h: pipeline("fused_swiglu_h"),
            elementwise_region: pipeline("elementwise_region"),
            batch_elementwise_region: pipeline("batch_elementwise_region"),
            fused_swiglu_cast_f32_to_f16: pipeline("fused_swiglu_cast_f32_to_f16"),
            fused_swiglu_cast_f16_to_f32: pipeline("fused_swiglu_cast_f16_to_f32"),
            concat_segment_lastax: pipeline("concat_segment_lastax"),
            concat_segment_lastax4: pipeline("concat_segment_lastax4"),
            concat_lastax_multi: pipeline("concat_lastax_multi"),
            concat_lastax_multi4: pipeline("concat_lastax_multi4"),
            concat_segment_lastax_h: pipeline("concat_segment_lastax_h"),
            concat_midaxis_seg: pipeline("concat_midaxis_seg"),
            concat_midaxis_seg_h: pipeline("concat_midaxis_seg_h"),
            concat_midaxis_seg_f32_to_f16: pipeline("concat_midaxis_seg_f32_to_f16"),
            concat_midaxis_seg_f16_to_f32: pipeline("concat_midaxis_seg_f16_to_f32"),
            elem_sub: pipeline("elem_sub"),
            elem_div: pipeline("elem_div"),
            elem_max: pipeline("elem_max"),
            elem_min: pipeline("elem_min"),
            elem_pow: pipeline("elem_pow"),
            elem_binop: pipeline("elem_binop"),
            elem_binop_h: pipeline("elem_binop_h"),
            elem_compare: pipeline("elem_compare"),
            elem_compare_bcast: pipeline("elem_compare_bcast"),
            elem_where: pipeline("elem_where"),
            elem_where_bcast: pipeline("elem_where_bcast"),
            elem_fma: pipeline("elem_fma"),
            relu_backward: pipeline("relu_backward"),
            activation_backward: pipeline("activation_backward"),
            complex_norm_sq: pipeline("complex_norm_sq"),
            complex_norm_sq_backward: pipeline("complex_norm_sq_backward"),
            conjugate_c64: pipeline("conjugate_c64"),
            fft_butterfly_stage: pipeline("fft_butterfly_stage"),
            fake_quantize_fixed: pipeline("fake_quantize_fixed"),
            fake_quantize_perbatch: pipeline("fake_quantize_perbatch"),
            reduce_axes: pipeline("reduce_axes"),
            reduce_axes_sum_simd: pipeline("reduce_axes_sum_simd"),
            topk_lastax: pipeline("topk_lastax"),
            grouped_matmul: pipeline("grouped_matmul"),
            scatter_add_zero: pipeline("scatter_add_zero"),
            scatter_add_accumulate: pipeline("scatter_add_accumulate"),
            transpose_nd: pipeline("transpose_nd"),
            transpose_nd_h: pipeline("transpose_nd_h"),
            transpose_2d_f32: pipeline("transpose_2d_f32"),
            transpose_2d_tiled_f32: pipeline("transpose_2d_tiled_f32"),
            transpose_last2_batched_f32: pipeline("transpose_last2_batched_f32"),
            transpose_last2_batched_tiled_f32: pipeline("transpose_last2_batched_tiled_f32"),
            transpose_swap12_batched_trail_f32: pipeline("transpose_swap12_batched_trail_f32"),
            transpose_swap12_batched_trail_tiled_f32: pipeline(
                "transpose_swap12_batched_trail_tiled_f32",
            ),
            gather_axis: pipeline("gather_axis"),
            pool2d: pipeline("pool2d"),
            maxpool2d_backward: pipeline("maxpool2d_backward"),
            conv2d_backward_input: pipeline("conv2d_backward_input"),
            conv2d_backward_weight: pipeline("conv2d_backward_weight"),
            conv2d_backward_weight_partial: pipeline("conv2d_backward_weight_partial"),
            conv2d_backward_weight_reduce: pipeline("conv2d_backward_weight_reduce"),
            maxpool3d_backward: pipeline("maxpool3d_backward"),
            conv3d_backward_input: pipeline("conv3d_backward_input"),
            conv3d_backward_weight: pipeline("conv3d_backward_weight"),
            conv2d: pipeline("conv2d"),
            depthwise_conv1d_bsc: pipeline("depthwise_conv1d_bsc"),
            conv2d_w1: pipeline("conv2d_w1"),
            layer_norm2d: pipeline("layer_norm2d"),
            group_norm: pipeline("group_norm"),
            resize_nearest_2x: pipeline("resize_nearest_2x"),
            conv_transpose2d: pipeline("conv_transpose2d"),
            conv3d: pipeline("conv3d"),
            conv_transpose3d: pipeline("conv_transpose3d"),
            relu_inplace: pipeline("relu_inplace"),
            sigmoid_inplace: pipeline("sigmoid_inplace"),
            tanh_inplace: pipeline("tanh_inplace"),
            exp_inplace: pipeline("exp_inplace"),
            log_inplace: pipeline("log_inplace"),
            sqrt_inplace: pipeline("sqrt_inplace"),
            rsqrt_inplace: pipeline("rsqrt_inplace"),
            rec_inplace: pipeline("rec_inplace"),
            neg_inplace: pipeline("neg_inplace"),
            abs_inplace: pipeline("abs_inplace"),
            round_inplace: pipeline("round_inplace"),
            sin_inplace: pipeline("sin_inplace"),
            cos_inplace: pipeline("cos_inplace"),
            tan_inplace: pipeline("tan_inplace"),
            atan_inplace: pipeline("atan_inplace"),
            softmax_lastax: pipeline("softmax_lastax"),
            softmax_lastax_causal: pipeline("softmax_lastax_causal"),
            softmax_cross_entropy_dense: pipeline("softmax_cross_entropy_dense"),
            softmax_cross_entropy_with_logits: pipeline("softmax_cross_entropy_with_logits"),
            softmax_cross_entropy_backward: pipeline("softmax_cross_entropy_backward"),
            fft_radix2_full_f32: pipeline("fft_radix2_full_f32"),
            #[cfg(feature = "native-gpu-fft")]
            fft_radix2_full_big_f32: pipeline("fft_radix2_full_big_f32"),
            #[cfg(feature = "native-gpu-fft")]
            fft_radix4_full_f32: pipeline("fft_radix4_full_f32"),
            #[cfg(feature = "native-gpu-fft")]
            fft_radix8_full_f32: pipeline("fft_radix8_full_f32"),
            #[cfg(feature = "native-gpu-fft")]
            fft_radix16_full_f32: pipeline("fft_radix16_full_f32"),
            fft_bit_reverse_f32: pipeline("fft_bit_reverse_f32"),
            fft_inner_f32: pipeline("fft_inner_f32"),
            fft_outer_r4_f32: pipeline("fft_outer_r4_f32"),
            fft_outer_r2_f32: pipeline("fft_outer_r2_f32"),
            gated_delta_net: pipeline("gated_delta_net"),
            gated_delta_net_sg: pipeline("gated_delta_net_sg"),
            selective_scan: pipeline("selective_scan"),
            lstm: pipeline("lstm"),
            gru: pipeline("gru"),
            rnn: pipeline("rnn"),
            mamba2: pipeline("mamba2"),
            dequant_gguf: pipeline("dequant_gguf"),
            q4k_mv_f32: pipeline("q4k_mv_f32"),
            q3k_mv_f32: pipeline("q3k_mv_f32"),
            q6k_mv_f32: pipeline("q6k_mv_f32"),
            q6k_mv_f32_sg: pipeline("q6k_mv_f32_sg"),
            q8_0_mv_f32_sg: pipeline("q8_0_mv_f32_sg"),
            q4_0_mv_f32_sg: pipeline("q4_0_mv_f32_sg"),
            q4_1_mv_f32_sg: pipeline("q4_1_mv_f32_sg"),
            q3k_mv_f32_sg: pipeline("q3k_mv_f32_sg"),
            q1_0_mv_f32: pipeline("q1_0_mv_f32"),
            q1_0_mv_f32_sg: pipeline("q1_0_mv_f32_sg"),
            q1_0_dual_mv_f32_sg: pipeline("q1_0_dual_mv_f32_sg"),
            q1_0_mm_f32: pipeline("q1_0_mm_f32"),
            q2_0_mv_f32: pipeline("q2_0_mv_f32"),
            q2_0_mv_f32_sg: pipeline("q2_0_mv_f32_sg"),
            q2_0_dual_mv_f32_sg: pipeline("q2_0_dual_mv_f32_sg"),
            q2_0_mm_f32: pipeline("q2_0_mm_f32"),
            q4_0_mv_f32: pipeline("q4_0_mv_f32"),
            q4_1_mv_f32: pipeline("q4_1_mv_f32"),
            q8_0_mv_f32: pipeline("q8_0_mv_f32"),
            iq4_nl_mv_f32: pipeline("iq4_nl_mv_f32"),
            iq2_xxs_mv_f32: pipeline("iq2_xxs_mv_f32"),
            synth_matmul_codebook: pipeline("synth_matmul_codebook"),
            gemm_rb_bias: pipeline("gemm_rb_bias"),
            synth_reconstruct_nk: pipeline("synth_reconstruct_nk"),
            synth_bwd_dx: pipeline("synth_bwd_dx"),
            synth_bwd_codebook: pipeline("synth_bwd_codebook"),
            synth_matmul_codebook_mm: pipeline("synth_matmul_codebook_mm"),
            synth_matmul_codebook_h: pipeline("synth_matmul_codebook_h"),
            synth_matmul_codebook_mm_h: pipeline("synth_matmul_codebook_mm_h"),
            synth_matmul_codebook_tiled: pipeline("synth_matmul_codebook_tiled"),
            synth_matmul_codebook_tiled_h: pipeline("synth_matmul_codebook_tiled_h"),
            synth_reconstruct: pipeline("synth_reconstruct"),
            synth_reconstruct_h: pipeline("synth_reconstruct_h"),
            spline_activation: pipeline("spline_activation"),
            spline_activation_backward_x: pipeline("spline_activation_backward_x"),
            spline_activation_backward_coeff: pipeline("spline_activation_backward_coeff"),
            iq2_xs_mv_f32: pipeline("iq2_xs_mv_f32"),
            iq2_s_mv_f32: pipeline("iq2_s_mv_f32"),
            iq3_xxs_mv_f32: pipeline("iq3_xxs_mv_f32"),
            iq3_s_mv_f32: pipeline("iq3_s_mv_f32"),
            iq3_xxs_mv_f32_sg: pipeline("iq3_xxs_mv_f32_sg"),
            iq3_s_mv_f32_sg: pipeline("iq3_s_mv_f32_sg"),
            iq1_s_mv_f32: pipeline("iq1_s_mv_f32"),
            iq1_m_mv_f32: pipeline("iq1_m_mv_f32"),
            q4k_mv_f32_sg: pipeline("q4k_mv_f32_sg"),
            q4k_mm_f32: pipeline("q4k_mm_f32"),
            q6k_mm_f32: pipeline("q6k_mm_f32"),
            q4k_swiglu_mv_f32: pipeline("q4k_swiglu_mv_f32"),
            q4k_gelu_mv_f32: pipeline("q4k_gelu_mv_f32"),
            q5_0_swiglu_mv_f32: pipeline("q5_0_swiglu_mv_f32"),
            q5_0_gelu_mv_f32: pipeline("q5_0_gelu_mv_f32"),
            q4k_mv_residual_f32: pipeline("q4k_mv_residual_f32"),
            q6k_mv_residual_f32: pipeline("q6k_mv_residual_f32"),
            q5_0_mv_residual_f32: pipeline("q5_0_mv_residual_f32"),
            q1_0_swiglu_mv_f32: pipeline("q1_0_swiglu_mv_f32"),
            q1_0_swiglu_mv_f32_sg: pipeline("q1_0_swiglu_mv_f32_sg"),
            q1_0_mv_residual_f32: pipeline("q1_0_mv_residual_f32"),
            q1_0_mv_residual_f32_sg: pipeline("q1_0_mv_residual_f32_sg"),
            q2_0_swiglu_mv_f32: pipeline("q2_0_swiglu_mv_f32"),
            q2_0_swiglu_mv_f32_sg: pipeline("q2_0_swiglu_mv_f32_sg"),
            q2_0_mv_residual_f32: pipeline("q2_0_mv_residual_f32"),
            q2_0_mv_residual_f32_sg: pipeline("q2_0_mv_residual_f32_sg"),
            rms_norm_bwd: pipeline("rms_norm_bwd"),
            rms_norm_bwd_param: pipeline("rms_norm_bwd_param"),
            rms_norm_bwd_inv_r_f32: pipeline("rms_norm_bwd_inv_r_f32"),
            rms_norm_bwd_param_reduce_f32: pipeline("rms_norm_bwd_param_reduce_f32"),
            layer_norm_bwd: pipeline("layer_norm_bwd"),
            layer_norm_bwd_gamma: pipeline("layer_norm_bwd_gamma"),
            layer_norm_bwd_stats_f32: pipeline("layer_norm_bwd_stats_f32"),
            layer_norm_bwd_gamma_reduce_f32: pipeline("layer_norm_bwd_gamma_reduce_f32"),
            layer_norm_bwd_gamma_reduce_simd: pipeline("layer_norm_bwd_gamma_reduce_simd"),
            group_norm_bwd_input: pipeline("group_norm_bwd_input"),
            group_norm_bwd_gamma: pipeline("group_norm_bwd_gamma"),
            group_norm_bwd_beta: pipeline("group_norm_bwd_beta"),
            rope_bwd: pipeline("rope_bwd"),
            cumsum_fwd: pipeline("cumsum_fwd"),
            cum_scan: pipeline("cum_scan"),
            cumsum_bwd: pipeline("cumsum_bwd"),
            im2col_group: pipeline("im2col_group"),
            im2col_group_w1: pipeline("im2col_group_w1"),
            conv2d_bwd_weight_gemm: pipeline("conv2d_bwd_weight_gemm"),
            conv2d_bwd_weight_gemm_4x4: pipeline("conv2d_bwd_weight_gemm_4x4"),
            attn_bwd_scores_f32: pipeline("attn_bwd_scores_f32"),
            attn_bwd_dp_f32: pipeline("attn_bwd_dp_f32"),
            attn_bwd_ds_f32: pipeline("attn_bwd_ds_f32"),
            attn_bwd_dv_f32: pipeline("attn_bwd_dv_f32"),
            attn_bwd_dq_f32: pipeline("attn_bwd_dq_f32"),
            attn_bwd_dk_f32: pipeline("attn_bwd_dk_f32"),
            attn_bwd_scores_batched_f32: pipeline("attn_bwd_scores_batched_f32"),
            attn_bwd_dp_batched_f32: pipeline("attn_bwd_dp_batched_f32"),
            attn_bwd_ds_batched_f32: pipeline("attn_bwd_ds_batched_f32"),
            attn_bwd_dv_batched_f32: pipeline("attn_bwd_dv_batched_f32"),
            attn_bwd_dq_batched_f32: pipeline("attn_bwd_dq_batched_f32"),
            attn_bwd_dk_batched_f32: pipeline("attn_bwd_dk_batched_f32"),
            attn_bwd_fused_f32: pipeline("attn_bwd_fused_f32"),
            gather_bwd_zero: pipeline("gather_bwd_zero"),
            gather_bwd_acc: pipeline("gather_bwd_acc"),
            gaussian_splat_rasterize: pipeline("gaussian_splat_rasterize"),
            gaussian_splat_rasterize_linear: pipeline("gaussian_splat_rasterize_linear"),
            gaussian_splat_rasterize_linear_traced: pipeline(
                "gaussian_splat_rasterize_linear_traced",
            ),
            gaussian_splat_rasterize_backward_linear: pipeline(
                "gaussian_splat_rasterize_backward_linear",
            ),
            gaussian_splat_adam_step: pipeline("gaussian_splat_adam_step"),
            gaussian_splat_mse_loss_grad: pipeline("gaussian_splat_mse_loss_grad"),
            gaussian_splat_ssim_stats: pipeline("gaussian_splat_ssim_stats"),
            gaussian_splat_blended_loss_grad: pipeline("gaussian_splat_blended_loss_grad"),
            gaussian_splat_project_training: pipeline("gaussian_splat_project_training"),
            gaussian_splat_geometry_backward: pipeline("gaussian_splat_geometry_backward"),
            gaussian_splat_scene_grad_projection: pipeline("gaussian_splat_scene_grad_projection"),
            gaussian_splat_splat_color_backward: pipeline("gaussian_splat_splat_color_backward"),
            gaussian_splat_emit_tile_keys: pipeline("gaussian_splat_emit_tile_keys"),
            gaussian_splat_project_screen_ellipse: pipeline(
                "gaussian_splat_project_screen_ellipse",
            ),
            gaussian_splat_emit_tile_keys_conic: pipeline("gaussian_splat_emit_tile_keys_conic"),
            gaussian_splat_bin_histogram: pipeline("gaussian_splat_bin_histogram"),
            gaussian_splat_bin_copy_counts: pipeline("gaussian_splat_bin_copy_counts"),
            gaussian_splat_bin_prefix_sum: pipeline("gaussian_splat_bin_prefix_sum"),
            gaussian_splat_bin_scatter: pipeline("gaussian_splat_bin_scatter"),
            gaussian_splat_build_tile_ranges: pipeline("gaussian_splat_build_tile_ranges"),
            gaussian_splat_pack_grads: pipeline("gaussian_splat_pack_grads"),
            scalar_acts: build_scalar_act_kernels(&pipeline),
            iq_grid_lut: build_iq_grid_lut(
                &metal_device()
                    .expect("rlx-metal: no Metal device for IQ grid LUT staging")
                    .device,
            ),
            library,
        }
    }

    /// Device buffer holding the IQ grid LUTs (see field docs).
    pub fn iq_grid_buffer(&self) -> &Buffer {
        &self.iq_grid_lut
    }

    /// Dump compiled-pipeline occupancy characteristics for the training-hot
    /// kernels (`RLX_METAL_KERNEL_STATS=1`). Reports, per kernel:
    ///   maxTG   = maxTotalThreadsPerThreadgroup — the register-limited ceiling
    ///             on threads/threadgroup (low ⇒ high register pressure).
    ///   simdW   = threadExecutionWidth (32 on Apple).
    ///   tgMem   = staticThreadgroupMemoryLength (bytes) — more ⇒ fewer
    ///             concurrent threadgroups per core (occupancy).
    ///   dispatch= threads/threadgroup we actually launch → occupancy vs maxTG.
    pub fn dump_stats(&self) {
        let rows: &[(&str, &ComputePipelineState, u64)] = &[
            // (name, pipeline, threads/threadgroup we dispatch)
            ("sdpa (serial softmax)", &self.sdpa, 32),
            ("sdpa_simd (parallel)", &self.sdpa_simd, 32),
            ("sdpa_simd_h16 (f16 scores)", &self.sdpa_simd_h16, 32),
            ("sdpa_long (baseline fwd, seq>64)", &self.sdpa_long, 64),
            (
                "sdpa_long_occpad (same work +20KB tg)",
                &self.sdpa_long_occpad,
                64,
            ),
            ("sdpa_splitk (V1 split-K)", &self.sdpa_splitk, 32),
            ("sdpa_fa2 (V2 flash-tile)", &self.sdpa_fa2, 64),
            ("sdpa_mma (V3 simdgroup)", &self.sdpa_mma, 32),
            (
                "attn_bwd_dv_batched (6-pass bwd)",
                &self.attn_bwd_dv_batched_f32,
                64,
            ),
            (
                "attn_bwd_dq_batched (6-pass bwd)",
                &self.attn_bwd_dq_batched_f32,
                64,
            ),
            (
                "attn_bwd_fused (Level 2 fused bwd)",
                &self.attn_bwd_fused_f32,
                64,
            ),
            ("sgemm_simd_4x4", &self.sgemm_simd_4x4, 64),
            ("sgemm_simd", &self.sgemm_simd, 64),
            ("sgemm_tiled", &self.sgemm_tiled, 256),
            ("layer_norm (fwd)", &self.layer_norm, 128),
            ("layer_norm_bwd (dx)", &self.layer_norm_bwd, 128),
            (
                "layer_norm_bwd_gamma_reduce_f32 (serial)",
                &self.layer_norm_bwd_gamma_reduce_f32,
                128,
            ),
            (
                "layer_norm_bwd_gamma_reduce_simd",
                &self.layer_norm_bwd_gamma_reduce_simd,
                32,
            ),
            ("reduce_axes (serial)", &self.reduce_axes, 256),
            ("reduce_axes_sum_simd", &self.reduce_axes_sum_simd, 32),
            (
                "attn_bwd_scores_batched",
                &self.attn_bwd_scores_batched_f32,
                64,
            ),
            ("attn_bwd_ds_batched", &self.attn_bwd_ds_batched_f32, 256),
            ("gather_axis0 (embed fwd)", &self.gather_axis0, 64),
            (
                "scatter_add_accumulate (embed bwd)",
                &self.scatter_add_accumulate,
                64,
            ),
            ("softmax_lastax (lm head)", &self.softmax_lastax, 256),
            (
                "softmax_cross_entropy_dense",
                &self.softmax_cross_entropy_dense,
                256,
            ),
        ];
        eprintln!("[kernel-stats] compiled-pipeline occupancy (Apple SIMD width 32):");
        eprintln!(
            "  {:42}  {:>6} {:>6} {:>7}  {:>8}  {:>6}",
            "kernel", "maxTG", "simdW", "tgMem", "dispatch", "occ%"
        );
        for (name, p, disp) in rows {
            let maxtg = p.max_total_threads_per_threadgroup();
            let simdw = p.thread_execution_width();
            let tgmem = p.static_threadgroup_memory_length();
            let occ = if maxtg > 0 {
                (*disp as f64) / (maxtg as f64) * 100.0
            } else {
                0.0
            };
            eprintln!(
                "  {:42}  {:>6} {:>6} {:>6}B  {:>8} {:>6.1}",
                name, maxtg, simdw, tgmem, disp, occ
            );
        }
    }
}

/// Concatenate every IQ LUT into one shared Metal buffer in the layout
/// declared at the top of `dequant_gguf.msl`. Offsets must match the
/// `IQ_GRID_OFF_*` constants.
fn build_iq_grid_lut(device: &metal::DeviceRef) -> Buffer {
    use rlx_gguf::iq_grids::{
        IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
        KSIGNS_IQ2XS,
    };

    let mut bytes = Vec::with_capacity(33_944);
    // KMASK_IQ2XS (8) | KSIGNS_IQ2XS (128).
    bytes.extend_from_slice(&KMASK_IQ2XS);
    bytes.extend_from_slice(&KSIGNS_IQ2XS);
    // Each u64/u32 grid entry is stored little-endian — matches the
    // `to_le_bytes` cast used by the CPU kernel.
    for v in IQ2XXS_GRID.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in IQ2XS_GRID.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in IQ2S_GRID.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in IQ3XXS_GRID.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in IQ3S_GRID.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in IQ1S_GRID.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    device.new_buffer_with_data(
        bytes.as_ptr() as *const _,
        bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Get or compile the global kernel library.
pub fn kernels() -> &'static Kernels {
    static K: OnceLock<Kernels> = OnceLock::new();
    let k = K.get_or_init(Kernels::new);
    if rlx_ir::env::flag("RLX_METAL_KERNEL_STATS") {
        static ONCE: OnceLock<()> = OnceLock::new();
        ONCE.get_or_init(|| k.dump_stats());
    }
    k
}

/// Force MSL/metallib + pipeline state init (call once at process load).
pub fn prewarm() -> &'static Kernels {
    kernels()
}
