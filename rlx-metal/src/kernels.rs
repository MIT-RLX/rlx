// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Custom MSL compute kernels for element-wise + fused operations.
//!
//! Each kernel is a Metal compute pipeline. Compiled once at startup
//! from inline MSL source, dispatched via command encoder at runtime.
//!
//! Mirrors rlx-cpu/src/kernels.rs but for GPU.

use crate::device::metal_device;
use metal::{ComputePipelineState, Library};
use std::sync::OnceLock;

/// Inline MSL source for all kernels — compiled once at startup.
pub const RLX_KERNELS_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

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
            float arg = v * 0.7071067811865475f;
            float sign = arg >= 0.0f ? 1.0f : -1.0f;
            float xa = abs(arg);
            float t = 1.0f / (1.0f + 0.3275911f * xa);
            float y = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f
                    + t * (-1.453152027f + t * 1.061405429f))));
            float erf_val = sign * (1.0f - y * exp(-xa * xa));
            v = v * 0.5f * (1.0f + erf_val);
        } else if (act_kind == 2) {
            v = v / (1.0f + exp(-v));
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

kernel void gelu_inplace_h(
    device half* data  [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    // Promote to f32 for math (more accurate for f16 input)
    float x = float(data[gid]);
    float arg = x * 0.7071067811865475f;
    float sign = arg >= 0.0f ? 1.0f : -1.0f;
    float xa = abs(arg);
    float t = 1.0f / (1.0f + 0.3275911f * xa);
    float y = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f
            + t * (-1.453152027f + t * 1.061405429f))));
    float erf_val = sign * (1.0f - y * exp(-xa * xa));
    data[gid] = half(x * 0.5f * (1.0f + erf_val));
}

kernel void silu_inplace_h(
    device half* data  [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    float x = float(data[gid]);
    data[gid] = half(x / (1.0f + exp(-x)));
}

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
    float var = partial_sumsq[0] / float(h) - mean * mean;
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
    float var = partial_sumsq[0] / float(h) - mean * mean;
    float inv_std = rsqrt(var + eps);

    for (uint i = tid; i < h; i += tsize) {
        float v = float(x[row * h + i]) + float(res[row * h + i]);
        out[row * h + i] = half((v - mean) * inv_std * float(gamma[i]) + float(beta[i]));
    }
}

kernel void fused_residual_rms_norm_h(
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
    device const half* src   [[buffer(0)]],
    device half* dst         [[buffer(1)]],
    constant uint& outer     [[buffer(2)]],
    constant uint& src_axis  [[buffer(3)]],
    constant uint& start     [[buffer(4)]],
    constant uint& len       [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
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
    uint tgid_x [[threadgroup_position_in_grid]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tsize  [[threads_per_threadgroup]]
) {
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
                float e = exp(scores[qi * seq + ki] - row_max);
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

    // PLAN L1 — `seq_stride` is the compile-time full extent for buffer
    // offsets; `seq` is the (possibly scaled) iteration bound.
    uint src_base = bi * seq_stride * src_row_stride + si * src_row_stride + hi * head_dim;
    uint dst_base = bi * seq_stride * hidden + si * hidden + hi * head_dim;
    uint d = gid.x;
    if (d < rot_half) {
        float x1 = float(x[src_base + d]);
        float x2 = float(x[src_base + rot_half + d]);
        float c = float(cos[si * half_dh + d]);
        float s = float(sin[si * half_dh + d]);
        out[dst_base + d] = half(x1 * c - x2 * s);
        out[dst_base + rot_half + d] = half(x2 * c + x1 * s);
    } else if (d >= n_rot) {
        out[dst_base + d] = x[src_base + d];
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
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& len      [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
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
            float arg = v * 0.7071067811865475;
            float sign = arg >= 0.0 ? 1.0 : -1.0;
            float xa = abs(arg);
            float t = 1.0 / (1.0 + 0.3275911 * xa);
            float y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                    + t * (-1.453152027 + t * 1.061405429))));
            float erf_val = sign * (1.0 - y * exp(-xa * xa));
            v = v * 0.5 * (1.0 + erf_val);
        } else if (act_kind == 2) {
            v = v / (1.0 + exp(-v));
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
            // GELU (Abramowitz & Stegun erf approx)
            float arg = v * 0.7071067811865475;
            float sign = arg >= 0.0 ? 1.0 : -1.0;
            float xa = abs(arg);
            float t = 1.0 / (1.0 + 0.3275911 * xa);
            float y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                    + t * (-1.453152027 + t * 1.061405429))));
            float erf_val = sign * (1.0 - y * exp(-xa * xa));
            v = v * 0.5 * (1.0 + erf_val);
        } else if (act_kind == 2) {
            v = v / (1.0 + exp(-v));
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
                float arg = v * 0.7071067811865475;
                float sign = arg >= 0.0 ? 1.0 : -1.0;
                float xa = abs(arg);
                float t = 1.0 / (1.0 + 0.3275911 * xa);
                float y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                        + t * (-1.453152027 + t * 1.061405429))));
                float erf_val = sign * (1.0 - y * exp(-xa * xa));
                v = v * 0.5 * (1.0 + erf_val);
            } else if (act_kind == 2) {
                v = v / (1.0 + exp(-v));
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
kernel void gelu_inplace(
    device float* data [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    float x = data[gid];
    float arg = x * 0.7071067811865475;  // x / sqrt(2)
    float sign = arg >= 0.0 ? 1.0 : -1.0;
    float xa = abs(arg);
    float t = 1.0 / (1.0 + 0.3275911 * xa);
    float y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
            + t * (-1.453152027 + t * 1.061405429))));
    float erf_val = sign * (1.0 - y * exp(-xa * xa));
    data[gid] = x * 0.5 * (1.0 + erf_val);
}

// Element-wise add: c = a + b (same length)
kernel void elem_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c       [[buffer(2)]],
    constant uint& len    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    c[gid] = a[gid] + b[gid];
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
        default: out = pow(lv, rv); break;
    }
    dst[gid] = out;
}

// Element-wise multiply: c = a * b
kernel void elem_mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c       [[buffer(2)]],
    constant uint& len    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    c[gid] = a[gid] * b[gid];
}

// Element-wise subtract: c = a - b
kernel void elem_sub(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c       [[buffer(2)]],
    constant uint& len    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    c[gid] = a[gid] - b[gid];
}

// Element-wise divide: c = a / b
kernel void elem_div(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c       [[buffer(2)]],
    constant uint& len    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
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
) { if (gid >= len) return; c[gid] = pow(a[gid], b[gid]); }

// Element-wise compare: writes 1.0 / 0.0 per element. `op_kind` selects:
//   0=Eq 1=Ne 2=Lt 3=Le 4=Gt 5=Ge
// One kernel for all six variants keeps the binary-shaped dispatch path
// uniform — the encoder picks op_kind at submit time.
kernel void elem_compare(
    device const float* a    [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    device float* c          [[buffer(2)]],
    constant uint& len       [[buffer(3)]],
    constant uint& op_kind   [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    float x = a[gid], y = b[gid];
    bool r = false;
    if      (op_kind == 0) r = (x == y);
    else if (op_kind == 1) r = (x != y);
    else if (op_kind == 2) r = (x <  y);
    else if (op_kind == 3) r = (x <= y);
    else if (op_kind == 4) r = (x >  y);
    else                   r = (x >= y);
    c[gid] = r ? 1.0f : 0.0f;
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
    float var = partial_sumsq[0] / float(count) - mean * mean;
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

// Two-phase scatter-add: phase 0 zeros the output buffer, phase 1
// accumulates updates atomically. Atomic add is required because
// multiple updates may target the same destination row from different
// threads. `op_phase`: 0 = zero, 1 = accumulate.
//
// Each phase is a single dispatch: phase 0 runs over `out_total` threads,
// phase 1 over `num_updates * trailing` threads. The encoder fires both
// in sequence within one command buffer.
kernel void scatter_add_zero(
    device atomic_uint* dst [[buffer(0)]],   // bit-cast view of f32 buffer
    constant uint& out_total [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= out_total) return;
    atomic_store_explicit(&dst[gid], 0u, memory_order_relaxed);
}

kernel void scatter_add_accumulate(
    device const float* updates [[buffer(0)]],
    device const float* indices [[buffer(1)]],
    device atomic_uint* dst     [[buffer(2)]],   // f32 reinterpreted as u32 atomic
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
    // Compare-and-swap loop for atomic float-add. Metal lacks native
    // atomic_add for float; reinterpret as uint, CAS the float bits.
    uint dst_idx = row * trailing + j;
    uint old_bits = atomic_load_explicit(&dst[dst_idx], memory_order_relaxed);
    while (true) {
        float old_f = as_type<float>(old_bits);
        float new_f = old_f + v;
        uint new_bits = as_type<uint>(new_f);
        if (atomic_compare_exchange_weak_explicit(
                &dst[dst_idx], &old_bits, new_bits,
                memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
        // CAS failed → old_bits now holds the latest value; retry.
    }
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
    data[gid] = tanh(data[gid]);
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
    partial[tid] = local_max;
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
    partial[tid] = local_sum;
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
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& outer    [[buffer(2)]],
    constant uint& src_axis [[buffer(3)]],
    constant uint& start    [[buffer(4)]],
    constant uint& len      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;
    if (i >= outer || j >= len) return;
    dst[i * len + j] = src[i * src_axis + start + j];
}

// Concat segment: copy one [outer, src_axis] tensor into [outer, dst_axis]
// at the column slice [dst_col .. dst_col + src_axis]. Multi-input concat
// = N dispatches of this kernel, one per source. Mirror of narrow_lastax.
kernel void concat_segment_lastax(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& outer    [[buffer(2)]],
    constant uint& src_axis [[buffer(3)]],
    constant uint& dst_axis [[buffer(4)]],
    constant uint& dst_col  [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;
    if (i >= outer || j >= src_axis) return;
    dst[i * dst_axis + dst_col + j] = src[i * src_axis + j];
}

kernel void concat_segment_lastax_h(
    device const half* src [[buffer(0)]],
    device half* dst       [[buffer(1)]],
    constant uint& outer    [[buffer(2)]],
    constant uint& src_axis [[buffer(3)]],
    constant uint& dst_axis [[buffer(4)]],
    constant uint& dst_col  [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;
    if (i >= outer || j >= src_axis) return;
    dst[i * dst_axis + dst_col + j] = src[i * src_axis + j];
}

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
    float var = partial_sumsq[0] / float(h) - mean * mean;
    float inv_std = rsqrt(var + eps);

    // Pass 2: write normalized output
    for (uint i = tid; i < h; i += tsize) {
        float v = x[row * h + i] + res[row * h + i];
        out[row * h + i] = (v - mean) * inv_std * gamma[i] + beta[i];
    }
}

// Fused residual + RMSNorm: out = RmsNorm(x + residual, gamma, beta)
kernel void fused_residual_rms_norm(
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
    threadgroup float partial_sumsq[256];
    float local_sumsq = 0.0;
    for (uint i = tid; i < h; i += tsize) {
        float v = x[row * h + i] + res[row * h + i];
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
        float v = x[row * h + i] + res[row * h + i];
        out[row * h + i] = v * inv_rms * gamma[i] + beta[i];
    }
}

// Multi-head SDPA: attention(Q, K, V, mask) → out
// Shapes: Q/out [batch, seq_q, heads*head_dim]; K/V [batch, seq_k, heads*head_dim]
// One threadgroup per (batch, head). Each TG computes [seq_q, seq_k] scores
// in threadgroup memory (seq_q * seq_k ≤ 64*64), applies softmax, then
// accumulates scores @ V.
kernel void sdpa(
    device const float* Q   [[buffer(0)]],
    device const float* K   [[buffer(1)]],
    device const float* V   [[buffer(2)]],
    device const float* M   [[buffer(3)]],
    device float* OUT       [[buffer(4)]],
    constant uint& batch      [[buffer(5)]],
    constant uint& seq_q      [[buffer(6)]],
    constant uint& heads      [[buffer(7)]],
    constant uint& head_dim   [[buffer(8)]],
    constant uint& q_stride   [[buffer(9)]],
    constant uint& mask_kind  [[buffer(10)]],
    constant uint& seq_k      [[buffer(11)]],
    constant uint& k_stride   [[buffer(12)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tsize  [[threads_per_threadgroup]]
) {
    // mask_kind:
    //   0 = None       (no masking)
    //   1 = Causal     (mask ki > (seq_k - seq_q) + qi)
    //   2 = Custom     (column-wise binary mask buffer M; 0 = padded)
    threadgroup float scores[64 * 64];   // up to seq_q * seq_k = 4096
    threadgroup float row_max;
    threadgroup float row_sum;

    // Linearized: tgid_x = bi * heads + hi
    uint bi = tgid_x / heads;
    uint hi = tgid_x % heads;
    if (bi >= batch) return;

    uint hs = heads * head_dim;
    float scale = rsqrt(float(head_dim));
    uint q_per_batch = q_stride * hs;
    uint k_per_batch = k_stride * hs;

    // 1. Compute scores[qi, ki] = scale * (Q[bi, qi, hi*dh:] · K[bi, ki, hi*dh:]) + mask.
    uint total = seq_q * seq_k;
    for (uint idx = tid; idx < total; idx += tsize) {
        uint qi = idx / seq_k;
        uint ki = idx % seq_k;
        float dot = 0.0;
        uint q_base = bi * q_per_batch + qi * hs + hi * head_dim;
        uint k_base = bi * k_per_batch + ki * hs + hi * head_dim;
        for (uint d = 0; d < head_dim; ++d) {
            dot += Q[q_base + d] * K[k_base + d];
        }
        float s = dot * scale;
        if (mask_kind == 1u) {
            uint q_offset = seq_k - seq_q;
            if (ki > q_offset + qi) s = -1e9;
        } else if (mask_kind == 2u) {
            if (M[bi * k_stride + ki] < 0.5) s = -1e9;
        }
        scores[qi * seq_k + ki] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Softmax row-by-row over scores[seq_q, seq_k]
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
                float e = exp(scores[qi * seq_k + ki] - row_max);
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

    // 3. Output[qi, d] = sum_ki scores[qi, ki] * V[bi, ki, hi*dh + d]
    uint out_total = seq_q * head_dim;
    for (uint idx = tid; idx < out_total; idx += tsize) {
        uint qi = idx / head_dim;
        uint d = idx % head_dim;
        float acc = 0.0;
        for (uint ki = 0; ki < seq_k; ++ki) {
            uint v_base = bi * k_per_batch + ki * hs + hi * head_dim;
            acc += scores[qi * seq_k + ki] * V[v_base + d];
        }
        uint o_base = bi * q_per_batch + qi * hs + hi * head_dim;
        OUT[o_base + d] = acc;
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
    device const float* Q   [[buffer(0)]],
    device const float* K   [[buffer(1)]],
    device const float* V   [[buffer(2)]],
    device const float* M   [[buffer(3)]],
    device float* OUT       [[buffer(4)]],
    constant uint& batch       [[buffer(5)]],
    constant uint& seq_q       [[buffer(6)]],   // query length Lq
    constant uint& heads       [[buffer(7)]],
    constant uint& head_dim    [[buffer(8)]],
    constant uint& q_stride    [[buffer(9)]],   // per-batch Q row stride (= Lq for dense)
    constant uint& mask_kind   [[buffer(10)]],
    constant uint& seq_k       [[buffer(11)]],  // key/value length Lk
    constant uint& k_stride    [[buffer(12)]],  // per-batch K/V row stride (= Lk for dense)
    uint tid_x [[thread_position_in_grid]]
) {
    // mask_kind:
    //   0 = None
    //   1 = Causal           (prefill — Lq == Lk required)
    //   2 = Custom            (binary key-padding mask M[B, Lk])
    //   3 = Bias              (additive per-head bias M[B, H, Lq, Lk])
    constexpr uint MAX_HEAD_DIM = 128u;
    uint total = batch * heads * seq_q;
    if (tid_x >= total) return;

    uint qi = tid_x % seq_q;
    uint bh = tid_x / seq_q;
    uint hi = bh % heads;
    uint bi = bh / heads;

    uint hs = heads * head_dim;
    float scale = rsqrt(float(head_dim));
    uint q_per_batch = q_stride * hs;
    uint k_per_batch = k_stride * hs;

    // Cache Q[qi, hi*dh : (hi+1)*dh] in registers — read seq_k times below.
    float q_reg[MAX_HEAD_DIM];
    uint q_base = bi * q_per_batch + qi * hs + hi * head_dim;
    for (uint d = 0; d < head_dim; ++d) q_reg[d] = Q[q_base + d];

    // Bias base offset (only read when mask_kind == 3).
    uint bias_row_base = ((bi * heads + hi) * seq_q + qi) * seq_k;

    // Online softmax accumulators.
    float m_acc = -1e30;
    float l_acc = 0.0;
    float o_acc[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; ++d) o_acc[d] = 0.0;

    for (uint ki = 0; ki < seq_k; ++ki) {
        // Score: scale * (Q · K[ki]) + mask
        uint k_base = bi * k_per_batch + ki * hs + hi * head_dim;
        float dot = 0.0;
        for (uint d = 0; d < head_dim; ++d) dot += q_reg[d] * K[k_base + d];
        float s = dot * scale;
        if (mask_kind == 1u) {
            uint q_offset = seq_k - seq_q;
            if (ki > q_offset + qi) s = -1e9;
        } else if (mask_kind == 2u) {
            if (M[bi * k_stride + ki] < 0.5) s = -1e9;
        } else if (mask_kind == 3u) {
            s += M[bias_row_base + ki];
        }

        // Online softmax update.
        float m_new = max(m_acc, s);
        float e_old = exp(m_acc - m_new);
        float e_cur = exp(s - m_new);
        l_acc = e_old * l_acc + e_cur;
        uint v_base = bi * k_per_batch + ki * hs + hi * head_dim;
        for (uint d = 0; d < head_dim; ++d) {
            o_acc[d] = e_old * o_acc[d] + e_cur * V[v_base + d];
        }
        m_acc = m_new;
    }

    // Normalize and emit.
    float inv_l = 1.0 / l_acc;
    uint o_base = bi * q_per_batch + qi * hs + hi * head_dim;
    for (uint d = 0; d < head_dim; ++d) {
        OUT[o_base + d] = o_acc[d] * inv_l;
    }
}

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
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]]
) {
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

    uint hs = heads * head_dim;
    uint q_per_batch = q_stride * hs;
    uint k_per_batch = k_stride * hs;
    float scale = rsqrt(float(head_dim));

    // ── Load Q tile cooperatively ────────────────────────────────────
    for (uint i = tid_in_tg; i < Br * head_dim; i += THREADS) {
        uint qi = i / head_dim;
        uint di = i % head_dim;
        uint pos = q_start + qi;
        Q_tg[qi * MAX_DH + di] = (pos < seq_q)
            ? Q[bi * q_per_batch + pos * hs + hi * head_dim + di]
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
            uint kv_off = bi * k_per_batch + pos * hs + hi * head_dim + di;
            bool in_range = pos < seq_k;
            K_tg[ki * MAX_DH + di] = in_range ? K[kv_off] : 0.0f;
            V_tg[ki * MAX_DH + di] = in_range ? V[kv_off] : 0.0f;
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
            OUT[bi * q_per_batch + pos * hs + hi * head_dim + di] = o;
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

    // PLAN L1 — `seq_stride` is the compile-time full extent for buffer
    // offsets; `seq` is the (possibly scaled) iteration bound. This
    // separation lets active-extent dispatch shrink the loop without
    // corrupting per-batch strides.
    uint src_base = bi * seq_stride * src_row_stride + si * src_row_stride + hi * head_dim;
    uint dst_base = bi * seq_stride * hidden + si * hidden + hi * head_dim;
    uint d = gid.x;
    if (d < rot_half) {
        float x1 = x[src_base + d];
        float x2 = x[src_base + rot_half + d];
        float c = cos[si * half_dh + d];
        float s = sin[si * half_dh + d];
        out[dst_base + d] = x1 * c - x2 * s;
        out[dst_base + rot_half + d] = x2 * c + x1 * s;
    } else if (d >= n_rot) {
        out[dst_base + d] = x[src_base + d];
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
    threadgroup float partial_sumsq[256];

    // Pass 1: compute mean + variance via reduction
    float local_sum = 0.0;
    float local_sumsq = 0.0;
    for (uint i = tid; i < h; i += tsize) {
        float v = input[row * h + i];
        local_sum += v;
        local_sumsq += v * v;
    }
    partial_sum[tid] = local_sum;
    partial_sumsq[tid] = local_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction within threadgroup
    for (uint stride = tsize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
            partial_sumsq[tid] += partial_sumsq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = partial_sum[0] / float(h);
    float var = partial_sumsq[0] / float(h) - mean * mean;
    float inv_std = rsqrt(var + eps);

    // Pass 2: normalize
    for (uint i = tid; i < h; i += tsize) {
        float v = input[row * h + i];
        output[row * h + i] = (v - mean) * inv_std * gamma[i] + beta[i];
    }
}

// RMSNorm: out = (x / sqrt(mean(x^2) + eps)) * gamma + beta. No mean
// subtraction. Same dispatch shape as layer_norm (one threadgroup per row,
// power-of-2 reduction within the group).
kernel void rms_norm(
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
    threadgroup float partial_sumsq[256];
    float local_sumsq = 0.0;
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
        output[row * h + i] = input[row * h + i] * inv_rms * gamma[i] + beta[i];
    }
}

// f16 RMSNorm: half I/O, float accumulation.
kernel void rms_norm_h(
    device const half* input  [[buffer(0)]],
    device const half* gamma  [[buffer(1)]],
    device const half* beta   [[buffer(2)]],
    device half* output       [[buffer(3)]],
    constant uint& h          [[buffer(4)]],
    constant float& eps       [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
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
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
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
            uint row;
            if (kind != 0u) { row = 0u; /* unused; scratch path below */ }
            else if ((scalar_input_mask & (1u << idx)) != 0u) { row = 0u; }
            else if (input_modulus[idx] != 0u) { row = gid % input_modulus[idx]; }
            else { row = gid; }
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
                uint row;
                if (kind != 0u) { row = 0u; }
                else if ((scalar_input_mask & (1u << idx)) != 0u) { row = 0u; }
                else if (input_modulus[idx] != 0u) { row = gid % input_modulus[idx]; }
                else { row = gid; }
                cond = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
            }
            float on_false;
            {
                uint kind = rhs_enc >> 31;
                uint idx  = rhs_enc & 0x7FFFFFFFu;
                uint row;
                if (kind != 0u) { row = 0u; }
                else if ((scalar_input_mask & (1u << idx)) != 0u) { row = 0u; }
                else if (input_modulus[idx] != 0u) { row = gid % input_modulus[idx]; }
                else { row = gid; }
                on_false = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
            }
            result = (cond != 0.0f) ? lhs : on_false;
        } else if (op_kind == 0u) {
            // Activation
            if      (op_sub == 3u) result = max(lhs, 0.0f);                // Relu
            else if (op_sub == 0u || op_sub == 1u) {
                float c = 0.7978845608f;
                float inner = c * (lhs + 0.044715f * lhs * lhs * lhs);
                result = 0.5f * lhs * (1.0f + tanh(inner));                // Gelu
            }
            else if (op_sub == 2u) result = lhs / (1.0f + exp(-lhs));      // Silu
            else if (op_sub == 4u) result = 1.0f / (1.0f + exp(-lhs));     // Sigmoid
            else if (op_sub == 5u) result = tanh(lhs);
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
                uint row;
                if (kind != 0u) { row = 0u; }
                else if ((scalar_input_mask & (1u << idx)) != 0u) { row = 0u; }
                else if (input_modulus[idx] != 0u) { row = gid % input_modulus[idx]; }
                else { row = gid; }
                rhs = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
            }
            if (op_kind == 2u) {
                if      (op_sub == 0u) result = lhs + rhs;
                else if (op_sub == 1u) result = lhs - rhs;
                else if (op_sub == 2u) result = lhs * rhs;
                else if (op_sub == 3u) result = lhs / rhs;
                else if (op_sub == 4u) result = max(lhs, rhs);
                else if (op_sub == 5u) result = min(lhs, rhs);
                else                   result = pow(lhs, rhs);
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

// ── 1D FFT (radix-2 Cooley-Tukey, f32, in-place per-row) ─────────────
// One threadgroup per row of `outer` independent FFTs. Layout matches
// the CPU kernel exactly: each row is 2N f32 with first N real, then
// N imag along the contiguous axis. The host caps N at 2048 (TG memory
// budget = 16KB = 4096 floats); larger N falls back to the host path.
// Twiddle factors recomputed per butterfly via direct cos/sin — Apple
// GPUs have a fast trig unit, and the iterative recurrence used on CPU
// doesn't parallelize cleanly across butterflies in the same stage.
kernel void fft_radix2_f32(
    device float* arena         [[buffer(0)]],
    constant uint& src_off      [[buffer(1)]],
    constant uint& dst_off      [[buffer(2)]],
    constant uint& n            [[buffer(3)]],   // complex points per row
    constant uint& log2n        [[buffer(4)]],   // ceil_log2(n)
    constant uint& inverse      [[buffer(5)]],   // 0 = forward, 1 = inverse
    uint  row     [[threadgroup_position_in_grid]],
    uint  tid     [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]]
) {
    // Fixed-size TG memory: 2 * N_MAX floats (real + imag halves).
    // N_MAX = 2048 → 16KB. Apple supports up to 32KB per threadgroup
    // but we leave headroom for any future register-spill workspace.
    threadgroup float sre[2048];
    threadgroup float sim[2048];

    uint row_base = row * 2u * n;

    // Load with bit-reverse permutation so the in-place butterflies
    // produce naturally-ordered output. reverse_bits is a 32-bit
    // hardware op; shift right by (32 - log2n) to discard the high
    // bits we don't care about.
    uint k = tid;
    while (k < n) {
        uint rev = reverse_bits(k) >> (32u - log2n);
        sre[rev] = arena[src_off + row_base + k];
        sim[rev] = arena[src_off + row_base + n + k];
        k += tg_size;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sign = (inverse != 0u) ? 1.0f : -1.0f;
    float two_pi = 6.28318530717958647692f;

    // Cooley-Tukey butterflies: log2(n) stages, length doubles each
    // stage. Each thread iterates over (n/2)/tg_size butterflies per
    // stage. Twiddle theta is direct cos/sin per butterfly (cheap on
    // Apple GPUs; avoids per-stage recurrence state).
    for (uint len = 2u; len <= n; len <<= 1u) {
        uint h2 = len >> 1u;
        float theta_base = sign * two_pi / float(len);
        for (uint b = tid; b < n / 2u; b += tg_size) {
            uint group = b / h2;
            uint k_in  = b % h2;
            uint i_lo  = group * len + k_in;
            uint i_hi  = i_lo + h2;
            float theta = theta_base * float(k_in);
            float wre = cos(theta);
            float wim = sin(theta);
            float t_re = wre * sre[i_hi] - wim * sim[i_hi];
            float t_im = wre * sim[i_hi] + wim * sre[i_hi];
            float u_re = sre[i_lo];
            float u_im = sim[i_lo];
            sre[i_lo] = u_re + t_re;
            sim[i_lo] = u_im + t_im;
            sre[i_hi] = u_re - t_re;
            sim[i_hi] = u_im - t_im;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result to dst (may equal src — load already pulled into TG).
    k = tid;
    while (k < n) {
        arena[dst_off + row_base + k]     = sre[k];
        arena[dst_off + row_base + n + k] = sim[k];
        k += tg_size;
    }
}

// ── Gated DeltaNet scan (f32) ───────────────────────────────────────
// One threadgroup per (batch, head), `n` threads parallelize the state
// dimension (n ≤ 128). Matches `execute_gated_delta_net_f32` on CPU.
#define GDN_MAX_N 128u

kernel void gated_delta_net(
    device float* arena        [[buffer(0)]],
    constant uint& q_off       [[buffer(1)]],
    constant uint& k_off       [[buffer(2)]],
    constant uint& v_off       [[buffer(3)]],
    constant uint& g_off       [[buffer(4)]],
    constant uint& beta_off    [[buffer(5)]],
    constant uint& state_off   [[buffer(6)]],
    constant uint& dst_off     [[buffer(7)]],
    constant uint4& dims       [[buffer(8)]], // batch, seq, heads, n
    constant uint& use_carry   [[buffer(9)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint b = dims.x, s = dims.y, h = dims.z, n = dims.w;
    if (n > GDN_MAX_N || gid >= b * h || tid >= n) return;

    uint bi = gid / h;
    uint hi = gid % h;
    uint j = tid;
    float scale = rsqrt(float(n));

    uint s_base = state_off + (bi * h + hi) * n * n;
    device float* s_mat = arena + s_base;

    if (use_carry == 0u && tid == 0u) {
        for (uint i = 0; i < n * n; ++i) {
            s_mat[i] = 0.0f;
        }
    }
    threadgroup float sk_sh[GDN_MAX_N];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint hs_n = h * n;

    for (uint ti = 0; ti < s; ++ti) {
        uint qkv_step = bi * s * hs_n + ti * hs_n + hi * n;
        uint gb_step  = bi * s * h + ti * h + hi;

        uint q_row = q_off + qkv_step;
        uint k_row = k_off + qkv_step;
        uint v_row = v_off + qkv_step;
        float g_t = arena[g_off + gb_step];
        float beta_t = arena[beta_off + gb_step];
        float g_exp = exp(g_t);

        if (tid == 0u) {
            for (uint idx = 0; idx < n * n; ++idx) {
                s_mat[idx] *= g_exp;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float acc = 0.0f;
        for (uint i = 0; i < n; ++i) {
            acc += s_mat[i * n + j] * arena[k_row + i];
        }
        sk_sh[j] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        sk_sh[j] = (arena[v_row + j] - sk_sh[j]) * beta_t;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < n; ++i) {
            float ki = arena[k_row + i];
            s_mat[i * n + j] += ki * sk_sh[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint out_row = dst_off + qkv_step;
        acc = 0.0f;
        for (uint i = 0; i < n; ++i) {
            acc += s_mat[i * n + j] * arena[q_row + i];
        }
        arena[out_row + j] = acc * scale;
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
    float inv_r3 = inv_r * inv_r * inv_r;
    for (uint i = tid; i < inner; i += tsize) {
        float xv = x[row * inner + i];
        float gv = gamma[i];
        float dyv = dy[row * inner + i];
        float term = gv * dyv - xv * dot * inv_r3;
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
"#;

const RLX_KERNELS_MSL_DEQUANT: &str = include_str!("dequant_gguf.msl");
const RLX_KERNELS_MSL_SPLAT: &str = include_str!("splat.msl");
const RLX_KERNELS_MSL_SPLAT_CONIC: &str = include_str!("splat_conic_bin.msl");

fn msl_source() -> String {
    format!(
        "{RLX_KERNELS_MSL}\n{RLX_KERNELS_MSL_DEQUANT}\n{RLX_KERNELS_MSL_SPLAT}\n{RLX_KERNELS_MSL_SPLAT_CONIC}"
    )
}

pub struct Kernels {
    pub library: Library,
    pub sgemm: ComputePipelineState,
    pub sgemm_simd: ComputePipelineState,
    pub sgemm_simd_bias: ComputePipelineState,
    pub sgemm_simd_4x4: ComputePipelineState,
    pub sgemm_simd_4x4_bias: ComputePipelineState,
    pub hgemm_simd_4x4: ComputePipelineState,
    pub hgemm_simd_4x4_bias: ComputePipelineState,
    pub bias_add_h: ComputePipelineState,
    pub gelu_inplace_h: ComputePipelineState,
    pub silu_inplace_h: ComputePipelineState,
    pub layer_norm_h: ComputePipelineState,
    pub fused_residual_ln_h: ComputePipelineState,
    pub fused_residual_rms_norm_h: ComputePipelineState,
    pub rms_norm_h: ComputePipelineState,
    pub softmax_lastax_h: ComputePipelineState,
    pub reduce_axes_h: ComputePipelineState,
    pub elem_add_h: ComputePipelineState,
    pub elem_mul_h: ComputePipelineState,
    pub gather_axis0_h: ComputePipelineState,
    pub narrow_lastax_h: ComputePipelineState,
    pub sdpa_h: ComputePipelineState,
    pub rope_h: ComputePipelineState,
    pub cast_f32_to_f16: ComputePipelineState,
    pub cast_f16_to_f32: ComputePipelineState,
    pub copy_f32: ComputePipelineState,
    pub sgemm_simd_padded: ComputePipelineState,
    pub sgemm_simd_padded_bias: ComputePipelineState,
    pub sgemm_tiled: ComputePipelineState,
    pub bias_add: ComputePipelineState,
    pub gelu_inplace: ComputePipelineState,
    pub silu_inplace: ComputePipelineState,
    pub layer_norm: ComputePipelineState,
    pub rms_norm: ComputePipelineState,
    pub elem_add: ComputePipelineState,
    pub binary_broadcast_f32: ComputePipelineState,
    pub elem_mul: ComputePipelineState,
    pub gather_axis0: ComputePipelineState,
    pub narrow_lastax: ComputePipelineState,
    pub fused_residual_ln: ComputePipelineState,
    pub fused_residual_rms_norm: ComputePipelineState,
    pub sdpa: ComputePipelineState,
    pub sdpa_long: ComputePipelineState,
    pub sdpa_fa_f32: ComputePipelineState,
    pub rope: ComputePipelineState,
    pub fused_swiglu: ComputePipelineState,
    pub fused_swiglu_h: ComputePipelineState,
    /// PLAN L2 — interpreted N-ary element-wise region kernel.
    pub elementwise_region: ComputePipelineState,
    pub fused_swiglu_cast_f32_to_f16: ComputePipelineState,
    pub fused_swiglu_cast_f16_to_f32: ComputePipelineState,
    pub concat_segment_lastax: ComputePipelineState,
    pub concat_segment_lastax_h: ComputePipelineState,
    pub elem_sub: ComputePipelineState,
    pub elem_div: ComputePipelineState,
    pub elem_max: ComputePipelineState,
    pub elem_min: ComputePipelineState,
    pub elem_pow: ComputePipelineState,
    pub elem_compare: ComputePipelineState,
    pub elem_where: ComputePipelineState,
    pub reduce_axes: ComputePipelineState,
    pub topk_lastax: ComputePipelineState,
    pub grouped_matmul: ComputePipelineState,
    pub scatter_add_zero: ComputePipelineState,
    pub scatter_add_accumulate: ComputePipelineState,
    pub transpose_nd: ComputePipelineState,
    pub gather_axis: ComputePipelineState,
    pub pool2d: ComputePipelineState,
    pub conv2d: ComputePipelineState,
    pub layer_norm2d: ComputePipelineState,
    pub group_norm: ComputePipelineState,
    pub resize_nearest_2x: ComputePipelineState,
    pub conv_transpose2d: ComputePipelineState,
    pub relu_inplace: ComputePipelineState,
    pub sigmoid_inplace: ComputePipelineState,
    pub tanh_inplace: ComputePipelineState,
    pub exp_inplace: ComputePipelineState,
    pub log_inplace: ComputePipelineState,
    pub sqrt_inplace: ComputePipelineState,
    pub rsqrt_inplace: ComputePipelineState,
    pub neg_inplace: ComputePipelineState,
    pub abs_inplace: ComputePipelineState,
    pub sin_inplace: ComputePipelineState,
    pub cos_inplace: ComputePipelineState,
    pub tan_inplace: ComputePipelineState,
    pub atan_inplace: ComputePipelineState,
    pub softmax_lastax: ComputePipelineState,
    pub fft_radix2_f32: ComputePipelineState,
    pub gated_delta_net: ComputePipelineState,
    pub dequant_gguf: ComputePipelineState,
    pub rms_norm_bwd: ComputePipelineState,
    pub rms_norm_bwd_param: ComputePipelineState,
    pub rope_bwd: ComputePipelineState,
    pub cumsum_bwd: ComputePipelineState,
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
}

unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

impl Kernels {
    fn new() -> Self {
        let dev = metal_device().expect("Metal device required");
        let opts = metal::CompileOptions::new();
        let library = dev
            .device
            .new_library_with_source(&msl_source(), &opts)
            .expect("MSL compilation failed");
        let pipeline = |name: &str| -> ComputePipelineState {
            let f = library.get_function(name, None).expect(name);
            dev.device
                .new_compute_pipeline_state_with_function(&f)
                .unwrap_or_else(|_| panic!("pipeline {name}"))
        };
        Self {
            sgemm: pipeline("sgemm"),
            sgemm_simd: pipeline("sgemm_simd"),
            sgemm_simd_bias: pipeline("sgemm_simd_bias"),
            sgemm_simd_4x4: pipeline("sgemm_simd_4x4"),
            sgemm_simd_4x4_bias: pipeline("sgemm_simd_4x4_bias"),
            hgemm_simd_4x4: pipeline("hgemm_simd_4x4"),
            hgemm_simd_4x4_bias: pipeline("hgemm_simd_4x4_bias"),
            bias_add_h: pipeline("bias_add_h"),
            gelu_inplace_h: pipeline("gelu_inplace_h"),
            silu_inplace_h: pipeline("silu_inplace_h"),
            layer_norm_h: pipeline("layer_norm_h"),
            fused_residual_ln_h: pipeline("fused_residual_ln_h"),
            fused_residual_rms_norm_h: pipeline("fused_residual_rms_norm_h"),
            rms_norm_h: pipeline("rms_norm_h"),
            softmax_lastax_h: pipeline("softmax_lastax_h"),
            reduce_axes_h: pipeline("reduce_axes_h"),
            elem_add_h: pipeline("elem_add_h"),
            elem_mul_h: pipeline("elem_mul_h"),
            gather_axis0_h: pipeline("gather_axis0_h"),
            narrow_lastax_h: pipeline("narrow_lastax_h"),
            sdpa_h: pipeline("sdpa_h"),
            rope_h: pipeline("rope_h"),
            cast_f32_to_f16: pipeline("cast_f32_to_f16"),
            cast_f16_to_f32: pipeline("cast_f16_to_f32"),
            copy_f32: pipeline("copy_f32"),
            sgemm_simd_padded: pipeline("sgemm_simd_padded"),
            sgemm_simd_padded_bias: pipeline("sgemm_simd_padded_bias"),
            sgemm_tiled: pipeline("sgemm_tiled"),
            bias_add: pipeline("bias_add"),
            gelu_inplace: pipeline("gelu_inplace"),
            silu_inplace: pipeline("silu_inplace"),
            layer_norm: pipeline("layer_norm"),
            rms_norm: pipeline("rms_norm"),
            elem_add: pipeline("elem_add"),
            binary_broadcast_f32: pipeline("binary_broadcast_f32"),
            elem_mul: pipeline("elem_mul"),
            gather_axis0: pipeline("gather_axis0"),
            narrow_lastax: pipeline("narrow_lastax"),
            fused_residual_ln: pipeline("fused_residual_ln"),
            fused_residual_rms_norm: pipeline("fused_residual_rms_norm"),
            sdpa: pipeline("sdpa"),
            sdpa_long: pipeline("sdpa_long"),
            sdpa_fa_f32: pipeline("sdpa_fa_f32"),
            rope: pipeline("rope"),
            fused_swiglu: pipeline("fused_swiglu"),
            fused_swiglu_h: pipeline("fused_swiglu_h"),
            elementwise_region: pipeline("elementwise_region"),
            fused_swiglu_cast_f32_to_f16: pipeline("fused_swiglu_cast_f32_to_f16"),
            fused_swiglu_cast_f16_to_f32: pipeline("fused_swiglu_cast_f16_to_f32"),
            concat_segment_lastax: pipeline("concat_segment_lastax"),
            concat_segment_lastax_h: pipeline("concat_segment_lastax_h"),
            elem_sub: pipeline("elem_sub"),
            elem_div: pipeline("elem_div"),
            elem_max: pipeline("elem_max"),
            elem_min: pipeline("elem_min"),
            elem_pow: pipeline("elem_pow"),
            elem_compare: pipeline("elem_compare"),
            elem_where: pipeline("elem_where"),
            reduce_axes: pipeline("reduce_axes"),
            topk_lastax: pipeline("topk_lastax"),
            grouped_matmul: pipeline("grouped_matmul"),
            scatter_add_zero: pipeline("scatter_add_zero"),
            scatter_add_accumulate: pipeline("scatter_add_accumulate"),
            transpose_nd: pipeline("transpose_nd"),
            gather_axis: pipeline("gather_axis"),
            pool2d: pipeline("pool2d"),
            conv2d: pipeline("conv2d"),
            layer_norm2d: pipeline("layer_norm2d"),
            group_norm: pipeline("group_norm"),
            resize_nearest_2x: pipeline("resize_nearest_2x"),
            conv_transpose2d: pipeline("conv_transpose2d"),
            relu_inplace: pipeline("relu_inplace"),
            sigmoid_inplace: pipeline("sigmoid_inplace"),
            tanh_inplace: pipeline("tanh_inplace"),
            exp_inplace: pipeline("exp_inplace"),
            log_inplace: pipeline("log_inplace"),
            sqrt_inplace: pipeline("sqrt_inplace"),
            rsqrt_inplace: pipeline("rsqrt_inplace"),
            neg_inplace: pipeline("neg_inplace"),
            abs_inplace: pipeline("abs_inplace"),
            sin_inplace: pipeline("sin_inplace"),
            cos_inplace: pipeline("cos_inplace"),
            tan_inplace: pipeline("tan_inplace"),
            atan_inplace: pipeline("atan_inplace"),
            softmax_lastax: pipeline("softmax_lastax"),
            fft_radix2_f32: pipeline("fft_radix2_f32"),
            gated_delta_net: pipeline("gated_delta_net"),
            dequant_gguf: pipeline("dequant_gguf"),
            rms_norm_bwd: pipeline("rms_norm_bwd"),
            rms_norm_bwd_param: pipeline("rms_norm_bwd_param"),
            rope_bwd: pipeline("rope_bwd"),
            cumsum_bwd: pipeline("cumsum_bwd"),
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
            library,
        }
    }
}

/// Get or compile the global kernel library.
pub fn kernels() -> &'static Kernels {
    static K: OnceLock<Kernels> = OnceLock::new();
    K.get_or_init(Kernels::new)
}
