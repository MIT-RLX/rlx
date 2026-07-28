// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// SAM2-style axial 2-D RoPE on `[batch, seq, num_heads * head_dim]`.
// Matches wgpu `axial_rope2d.wgsl` / `rlx_ir::ops::axial_rope2d`.
// Even lanes of each rotated pair write both outputs (avoids races).

extern "C" __global__ void axial_rope2d(
    float* arena,
    unsigned int batch,
    unsigned int seq,
    unsigned int hidden,
    unsigned int end_x,
    unsigned int end_y,
    unsigned int head_dim,
    unsigned int num_heads,
    unsigned int repeat_factor,
    float theta,
    unsigned int in_off,
    unsigned int out_off,
    unsigned int n_total
) {
    (void)num_heads;
    (void)end_y;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;

    unsigned int d = i % hidden;
    unsigned int q1 = i / hidden;
    unsigned int tok = q1 % seq;
    unsigned int bi = q1 / seq;

    unsigned int half = head_dim / 2u;
    unsigned int d_in_head = d % head_dim;
    unsigned int buf_idx = bi * seq * hidden + tok * hidden + d;
    unsigned int head_base = buf_idx - d_in_head;

    if ((d_in_head & 1u) != 0u) {
        return;
    }

    unsigned int repeat = repeat_factor < 1u ? 1u : repeat_factor;
    unsigned int pos = tok / repeat;
    float tx = (float)(pos % end_x);
    float ty = (float)(pos / end_x);

    if (d_in_head < half) {
        unsigned int c = d_in_head / 2u;
        float freq = 1.0f / powf(theta, (float)(4u * c) / (float)head_dim);
        float ang = tx * freq;
        float co = cosf(ang);
        float si = sinf(ang);
        unsigned int ix0 = head_base + 2u * c;
        unsigned int ix1 = ix0 + 1u;
        float x0 = arena[in_off + ix0];
        float x1 = arena[in_off + ix1];
        arena[out_off + ix0] = x0 * co - x1 * si;
        arena[out_off + ix1] = x0 * si + x1 * co;
    } else {
        unsigned int c = (d_in_head - half) / 2u;
        float freq = 1.0f / powf(theta, (float)(4u * c) / (float)head_dim);
        float ang = ty * freq;
        float co = cosf(ang);
        float si = sinf(ang);
        unsigned int ix0 = head_base + half + 2u * c;
        unsigned int ix1 = ix0 + 1u;
        float x0 = arena[in_off + ix0];
        float x1 = arena[in_off + ix1];
        arena[out_off + ix0] = x0 * co - x1 * si;
        arena[out_off + ix1] = x0 * si + x1 * co;
    }
}
