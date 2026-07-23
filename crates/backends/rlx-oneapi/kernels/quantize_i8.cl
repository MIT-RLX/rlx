// INT8 Quantize matching rlx-cpu / CUDA `quantize_i8`.
// Channel: c = (i / inner) % chan_dim (chan_dim==1 → c=0).
// Affine: affine[2*c+0]=scale (f32 bits), affine[2*c+1]=zp as i32 bits.
// I8 codes written densely at byte offset `q_byte_off` into the arena.

float q_round_half_away(float x) {
    float sgn = (x > 0.0f) - (x < 0.0f);
    return sgn * floor(fabs(x) + 0.5f);
}

__kernel void quantize_i8(__global float* arena,
                     uint n, uint chan_dim, uint inner,
                     uint in_off, uint q_byte_off,
                     __global const uint* affine) {
    uint i = get_global_id(0);
    if (i >= n) return;
    uint c = (chan_dim <= 1u) ? 0u : ((i / inner) % chan_dim);
    float s = as_float(affine[2u * c]);
    int zp = (int)affine[2u * c + 1u];
    float scaled = arena[in_off + i] / s;
    int v = (int)q_round_half_away(scaled) + zp;
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    __global char* q = (__global char*)arena + q_byte_off;
    q[i] = (char)v;
}
