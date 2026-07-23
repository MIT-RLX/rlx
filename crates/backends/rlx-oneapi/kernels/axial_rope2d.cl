// SAM2-style axial 2-D RoPE on [batch, seq, num_heads * head_dim].
__kernel void axial_rope2d(__global float* arena,
                           uint batch, uint seq, uint hidden,
                           uint end_x, uint end_y,
                           uint head_dim, uint num_heads, uint repeat_factor,
                           float theta,
                           uint in_off, uint out_off, uint n_total) {
    uint i = get_global_id(0);
    if (i >= n_total) return;

    uint d = i % hidden;
    uint q1 = i / hidden;
    uint tok = q1 % seq;
    uint bi = q1 / seq;

    uint half = head_dim / 2u;
    uint d_in_head = d % head_dim;
    uint buf_idx = bi * seq * hidden + tok * hidden + d;
    uint head_base = buf_idx - d_in_head;

    if ((d_in_head & 1u) != 0u) {
        return;
    }

    uint repeat = max(repeat_factor, 1u);
    uint pos = tok / repeat;
    float tx = (float)(pos % end_x);
    float ty = (float)(pos / end_x);

    if (d_in_head < half) {
        uint c = d_in_head / 2u;
        float freq = 1.0f / pow(theta, (float)(4u * c) / (float)head_dim);
        float ang = tx * freq;
        float co = cos(ang);
        float si = sin(ang);
        uint ix0 = head_base + 2u * c;
        uint ix1 = ix0 + 1u;
        float x0 = arena[in_off + ix0];
        float x1 = arena[in_off + ix1];
        arena[out_off + ix0] = x0 * co - x1 * si;
        arena[out_off + ix1] = x0 * si + x1 * co;
    } else {
        uint c = (d_in_head - half) / 2u;
        float freq = 1.0f / pow(theta, (float)(4u * c) / (float)head_dim);
        float ang = ty * freq;
        float co = cos(ang);
        float si = sin(ang);
        uint ix0 = head_base + half + 2u * c;
        uint ix1 = ix0 + 1u;
        float x0 = arena[in_off + ix0];
        float x1 = arena[in_off + ix1];
        arena[out_off + ix0] = x0 * co - x1 * si;
        arena[out_off + ix1] = x0 * si + x1 * co;
    }
}
