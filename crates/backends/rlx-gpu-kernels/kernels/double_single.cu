// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Op::DoubleSingle: f64 as interleaved (hi, lo) f32 pairs in the arena. The
// arithmetic is the `rlxsl::dw` CUDA prelude prepended by
// `rlx_gpu_kernels::double_single_cu`; indexing follows
// `rlx_ir::double_single::DfPlan`. kind 0 elementwise (op = `DfOp::code`),
// 1 matmul, 2 reduction (op 0 sum, 1 max). Offsets are f32 indices.

struct DfParams {
    unsigned int len, rank, op, kind;
    unsigned int m, n, k, a_batch;
    unsigned int b_batch, red_len, kept_rank, red_rank;
    unsigned int a_off, b_off, out_off, pad;
    unsigned int d0[8], s0[8], s1[8], d1[8], s2[8];
};

__device__ __forceinline__ unsigned int df_off(unsigned int i, unsigned int rank,
                                               const unsigned int* dims, const unsigned int* strides) {
    unsigned int off = 0;
    for (int d = (int)rank - 1; d >= 0; --d) {
        unsigned int n = dims[d];
        off += (i % n) * strides[d];
        i /= n;
    }
    return off;
}

__device__ __forceinline__ DwF32 df_ld(const float* base, unsigned int i) {
    return DwF32{base[2u * i], base[2u * i + 1u]};
}

extern "C" __global__ void double_single(float* arena, DfParams p) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= p.len) return;
    const float* a = arena + p.a_off;
    const float* b = arena + p.b_off;
    DwF32 r;
    if (p.kind == 0u) {
        DwF32 x = df_ld(a, df_off(gid, p.rank, p.d0, p.s0));
        DwF32 y = df_ld(b, df_off(gid, p.rank, p.d0, p.s1));
        switch (p.op) {
            case 0: r = dw_add(x, y); break;
            case 1: r = dw_sub(x, y); break;
            case 2: r = dw_mul(x, y); break;
            case 4: r = dw_sqrt(x); break;
            case 5: r = dw_exp(x); break;
            case 6: r = (y.hi > x.hi || (y.hi == x.hi && y.lo > x.lo)) ? y : x; break;
            default: r = dw_div(x, y); break;
        }
    } else if (p.kind == 1u) {
        unsigned int j = gid % p.n;
        unsigned int row = gid / p.n;
        unsigned int i = row % p.m;
        unsigned int bi = row / p.m;
        unsigned int a0 = bi * p.a_batch + i * p.k;
        unsigned int b0 = bi * p.b_batch + j;
        r = DwF32{0.0f, 0.0f};
        for (unsigned int kk = 0; kk < p.k; ++kk) {
            r = dw_add(r, dw_mul(df_ld(a, a0 + kk), df_ld(b, b0 + kk * p.n)));
        }
    } else {
        unsigned int base = df_off(gid, p.kept_rank, p.d0, p.s0);
        bool is_max = p.op == 1u;
        r = is_max ? df_ld(a, base) : DwF32{0.0f, 0.0f};
        for (unsigned int i = is_max ? 1u : 0u; i < p.red_len; ++i) {
            DwF32 x = df_ld(a, base + df_off(i, p.red_rank, p.d1, p.s2));
            if (is_max) {
                if (x.hi > r.hi || (x.hi == r.hi && x.lo > r.lo)) r = x;
            } else {
                r = dw_add(r, x);
            }
        }
    }
    float* out = arena + p.out_off;
    out[2u * gid] = r.hi;
    out[2u * gid + 1u] = r.lo;
}
