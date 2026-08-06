// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// FlashKDA-style CHUNKED-PARALLEL gated-delta-net (Kimi Delta Attention) forward.
// Prototype: one block per (batch, head), 128 threads = state column j. Processes
// CHUNK=16 tokens at a time with the WY / DPLR chunked form instead of the serial
// per-token scan in `gated_delta_net.cu`. Intra-chunk tensors live in shared
// memory; the n×n recurrent state stays in global (arena) scratch, touched once
// per chunk (≈16× less state traffic than the per-token native kernel).
//
// f32 throughout, natural-log gate (matches rlx `build_kda_chunked_scan` and the
// native `gated_delta_net` — NOT torch_ref's base-2 ex2). Requires n==128, and
// the Neumann product below is exact for CHUNK<=16 (strictly-lower L nilpotent).
// Opt-in via RLX_CUDA_KDA_CHUNK=1 (per-channel gate only).

#define KDA_N 128u
#define KDA_C 16u
#define KDA_THREADS 128u

// C[16x16] = A[16x16] @ B[16x16], cooperative over 128 threads (2 outputs each).
__device__ __forceinline__ void mm16(const float* A, const float* B, float* Cc,
                                      unsigned int tid) {
    for (unsigned int o = tid; o < KDA_C * KDA_C; o += KDA_THREADS) {
        unsigned int a = o >> 4, b = o & 15u;
        float s = 0.0f;
        #pragma unroll
        for (unsigned int k = 0u; k < KDA_C; ++k) s += A[a * KDA_C + k] * B[k * KDA_C + b];
        Cc[o] = s;
    }
}

extern "C" __global__ void kimi_delta_chunk(
    float* arena,
    unsigned long long q_off,
    unsigned long long k_off,
    unsigned long long v_off,
    unsigned long long g_off,
    unsigned long long beta_off,
    unsigned long long state_off,
    unsigned long long dst_off,
    unsigned int batch,
    unsigned int seq,
    unsigned int heads,
    unsigned int n,
    unsigned int use_carry,
    unsigned int gate_per_channel
) {
    if (n != KDA_N) return;                 // prototype: head_dim 128 only
    const unsigned int C = KDA_C;
    unsigned int gid = blockIdx.x;          // (batch, head)
    unsigned int j = threadIdx.x;           // state column / channel 0..127
    if (gid >= batch * heads || j >= n) return;
    unsigned int bi = gid / heads;
    unsigned int hi = gid % heads;
    float scale = rsqrtf((float)n);

    unsigned long long s_base = state_off + (unsigned long long)(bi * heads + hi) * n * n;
    float* H = arena + s_base;              // [n][n] global: H[i*n+j] = state[key i][val j]

    __shared__ float kd[KDA_C * KDA_N];     // k · exp(Gc)
    __shared__ float qd[KDA_C * KDA_N];     // q · exp(Gc) · scale
    __shared__ float ki[KDA_C * KDA_N];     // k · exp(-Gc)
    __shared__ float vv[KDA_C * KDA_N];     // v  (then reused as vcorr)
    __shared__ float U [KDA_C * KDA_N];     // INV · (beta·(v - kd·H))
    __shared__ float Lm[KDA_C * KDA_C];     // strictly-lower L · beta
    __shared__ float Iv[KDA_C * KDA_C];     // INV = (I+L)^-1
    __shared__ float Mq[KDA_C * KDA_C];     // tril(qd·ki^T)
    __shared__ float P [KDA_C * KDA_C];     // Neumann power buffer
    __shared__ float tt[KDA_C * KDA_C];     // Neumann temp
    __shared__ float egt[KDA_N];            // exp(Gtot) per channel
    __shared__ float bta[KDA_C];            // sigmoid beta per chunk row

    // Zero the recurrent state when not carrying (column-parallel).
    if (use_carry == 0u) {
        for (unsigned int i = 0u; i < n; ++i) H[i * n + j] = 0.0f;
    }
    __syncthreads();

    unsigned int hs_n = heads * n;
    unsigned int nch = (seq + C - 1u) / C;

    for (unsigned int c = 0u; c < nch; ++c) {
        // ── load chunk + per-column cumsum decay (thread j owns column j) ──
        float running = 0.0f;
        for (unsigned int r = 0u; r < C; ++r) {
            unsigned int t = c * C + r;
            float gval = 0.0f, kval = 0.0f, qval = 0.0f, vval = 0.0f;
            if (t < seq) {
                unsigned long long qkv = (unsigned long long)bi * seq * hs_n
                                       + (unsigned long long)t * hs_n + hi * n + j;
                gval = arena[g_off + qkv];   // per-channel gate [b,s,h,n]
                kval = arena[k_off + qkv];
                qval = arena[q_off + qkv];
                vval = arena[v_off + qkv];
            }
            running += gval;
            float e  = __expf(running);
            float ei = __expf(-running);
            kd[r * n + j] = kval * e;
            qd[r * n + j] = qval * e * scale;
            ki[r * n + j] = kval * ei;
            vv[r * n + j] = vval;
        }
        egt[j] = __expf(running);            // exp(Gtot) for channel j
        if (j < C) {
            unsigned int t = c * C + j;
            float bv = 0.0f;
            if (t < seq) bv = arena[beta_off + (unsigned long long)bi * seq * heads
                                              + (unsigned long long)t * heads + hi];
            bta[j] = bv;
        }
        __syncthreads();

        // ── L = tril(kd·ki^T, -1)·beta[row];  Mq = tril(qd·ki^T, 0) ──
        for (unsigned int o = j; o < C * C; o += KDA_THREADS) {
            unsigned int a = o / C, b = o % C;
            float sl = 0.0f, sm = 0.0f;
            for (unsigned int ch = 0u; ch < n; ++ch) {
                float kib = ki[b * n + ch];
                sl += kd[a * n + ch] * kib;
                sm += qd[a * n + ch] * kib;
            }
            Lm[o] = (a > b) ? sl * bta[a] : 0.0f;
            Mq[o] = (a >= b) ? sm : 0.0f;
        }
        __syncthreads();

        // ── INV = (I - L)(I + L^2)(I + L^4)(I + L^8) = (I + L)^-1  (exact, C<=16) ──
        for (unsigned int o = j; o < C * C; o += KDA_THREADS) {
            unsigned int a = o / C, b = o % C;
            Iv[o] = ((a == b) ? 1.0f : 0.0f) - Lm[o];
        }
        __syncthreads();
        mm16(Lm, Lm, P, j);                 // P = L^2
        __syncthreads();
        for (int step = 0; step < 3; ++step) {
            mm16(Iv, P, tt, j);             // tt = Iv · P
            __syncthreads();
            for (unsigned int o = j; o < C * C; o += KDA_THREADS) Iv[o] += tt[o];
            __syncthreads();
            if (step < 2) {
                mm16(P, P, tt, j);          // tt = P^2
                __syncthreads();
                for (unsigned int o = j; o < C * C; o += KDA_THREADS) P[o] = tt[o];
                __syncthreads();
            }
        }

        // ── vcorr = beta · (v - kd·H)   (overwrites vv) ──
        for (unsigned int a = 0u; a < C; ++a) {
            float s = 0.0f;
            for (unsigned int i = 0u; i < n; ++i) s += kd[a * n + i] * H[i * n + j];
            vv[a * n + j] = (vv[a * n + j] - s) * bta[a];
        }
        __syncthreads();

        // ── U = INV · vcorr ──
        for (unsigned int a = 0u; a < C; ++a) {
            float s = 0.0f;
            #pragma unroll
            for (unsigned int b = 0u; b < C; ++b) s += Iv[a * C + b] * vv[b * n + j];
            U[a * n + j] = s;
        }
        __syncthreads();

        // ── out = qd·H + Mq·U  (H still pre-update) ──
        for (unsigned int a = 0u; a < C; ++a) {
            unsigned int t = c * C + a;
            if (t >= seq) continue;
            float s1 = 0.0f;
            for (unsigned int i = 0u; i < n; ++i) s1 += qd[a * n + i] * H[i * n + j];
            float s2 = 0.0f;
            #pragma unroll
            for (unsigned int b = 0u; b < C; ++b) s2 += Mq[a * C + b] * U[b * n + j];
            unsigned long long out_off = dst_off + (unsigned long long)bi * seq * hs_n
                                       + (unsigned long long)t * hs_n + hi * n + j;
            arena[out_off] = s1 + s2;
        }
        __syncthreads();

        // ── H[i,j] = exp(Gtot[i]) · (H[i,j] + Σ_a ki[a,i]·U[a,j]) ──
        for (unsigned int i = 0u; i < n; ++i) {
            float s = 0.0f;
            #pragma unroll
            for (unsigned int a = 0u; a < C; ++a) s += ki[a * n + i] * U[a * n + j];
            H[i * n + j] = egt[i] * (H[i * n + j] + s);
        }
        __syncthreads();
    }
}
