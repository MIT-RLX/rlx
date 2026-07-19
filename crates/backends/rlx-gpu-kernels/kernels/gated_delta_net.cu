// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// GPL-3.0-only. See LICENSE.
//
// Gated-DeltaNet scan (f32). Mirrors Metal `gated_delta_net` / CPU
// `execute_gated_delta_net_f32`. One block per (batch, head); `n` threads
// parallelize the state column (n ≤ 128). Offsets are f32-word indices into
// the arena; 64-bit so packed 27B arenas (>4 GiB) don't wrap.

#define GDN_MAX_N 128u

extern "C" __global__ void gated_delta_net(
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
    unsigned int use_carry
) {
    unsigned int gid = blockIdx.x;   // (batch, head)
    unsigned int tid = threadIdx.x;  // column j in 0..n
    if (n > GDN_MAX_N || gid >= batch * heads || tid >= n) return;

    unsigned int bi = gid / heads;
    unsigned int hi = gid % heads;
    unsigned int j = tid;
    float scale = rsqrtf((float)n);

    unsigned long long s_base = state_off + (unsigned long long)(bi * heads + hi) * n * n;
    float* s_mat = arena + s_base;

    // Column-parallel zero (was serial on tid==0 → n² writes).
    if (use_carry == 0u) {
        for (unsigned int i = 0u; i < n; ++i) {
            s_mat[i * n + j] = 0.0f;
        }
    }
    __syncthreads();

    __shared__ float sk_sh[GDN_MAX_N];
    unsigned int hs_n = heads * n;

    for (unsigned int ti = 0u; ti < seq; ++ti) {
        unsigned int qkv_step = bi * seq * hs_n + ti * hs_n + hi * n;
        unsigned int gb_step = bi * seq * heads + ti * heads + hi;

        unsigned long long q_row = q_off + qkv_step;
        unsigned long long k_row = k_off + qkv_step;
        unsigned long long v_row = v_off + qkv_step;
        float g_t = arena[g_off + gb_step];
        float beta_t = arena[beta_off + gb_step];
        float g_exp = expf(g_t);

        // Column-parallel gate scale (was serial tid==0 over n²).
        for (unsigned int i = 0u; i < n; ++i) {
            s_mat[i * n + j] *= g_exp;
        }
        __syncthreads();

        float acc = 0.0f;
        for (unsigned int i = 0u; i < n; ++i) {
            acc += s_mat[i * n + j] * arena[k_row + i];
        }
        sk_sh[j] = acc;
        __syncthreads();

        sk_sh[j] = (arena[v_row + j] - sk_sh[j]) * beta_t;
        __syncthreads();

        for (unsigned int i = 0u; i < n; ++i) {
            float ki = arena[k_row + i];
            s_mat[i * n + j] += ki * sk_sh[j];
        }
        __syncthreads();

        unsigned long long out_row = dst_off + qkv_step;
        acc = 0.0f;
        for (unsigned int i = 0u; i < n; ++i) {
            acc += s_mat[i * n + j] * arena[q_row + i];
        }
        arena[out_row + j] = acc * scale;
    }
}
