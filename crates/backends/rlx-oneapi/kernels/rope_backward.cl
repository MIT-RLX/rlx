// NeoX RoPE backward: dx = rope(dy, cos, -sin) on rotated pairs.
// Mirrors wgpu `rope_backward.wgsl` / CUDA `rope_backward.cu`.

__kernel void rope_backward(__global float* arena,
                     uint batch, uint seq, uint hidden,
                     uint head_dim, uint n_rot,
                     uint dy_off, uint cos_off, uint sin_off,
                     uint dx_off, uint cos_len) {
    uint i = get_global_id(0);
    uint total = batch * seq * hidden;
    if (i >= total) return;

    uint nh = hidden / head_dim;
    uint d = i % head_dim;
    uint q1 = i / head_dim;
    uint hi = q1 % nh;
    uint q2 = q1 / nh;
    uint si = q2 % seq;
    uint bi = q2 / seq;
    uint half_dh = head_dim / 2u;
    uint rot_half = n_rot / 2u;
    uint tab_off = (si * half_dh) % (cos_len == 0u ? 1u : cos_len);

    uint dy_base = dy_off + bi * seq * hidden + si * hidden + hi * head_dim;
    uint dx_base = dx_off + bi * seq * hidden + si * hidden + hi * head_dim;

    if (d < rot_half) {
        float y1 = arena[dy_base + d];
        float y2 = arena[dy_base + rot_half + d];
        float c = arena[cos_off + tab_off + d];
        float s = arena[sin_off + tab_off + d];
        arena[dx_base + d] = y1 * c + y2 * s;
        arena[dx_base + rot_half + d] = -y1 * s + y2 * c;
    } else if (d >= n_rot) {
        arena[dx_base + d] = arena[dy_base + d];
    }
}
