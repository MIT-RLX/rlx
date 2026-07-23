// INT8 Dequantize matching rlx-cpu / CUDA `dequantize_i8`.
// Channel: c = (i / inner) % chan_dim (chan_dim==1 → c=0).
// Affine: affine[2*c+0]=scale (f32 bits), affine[2*c+1]=zp as i32 bits.
// I8 codes read densely at byte offset `q_byte_off` into the arena.

__kernel void dequantize_i8(__global float* arena,
                     uint n, uint chan_dim, uint inner,
                     uint q_byte_off, uint out_off,
                     __global const uint* affine) {
    uint i = get_global_id(0);
    if (i >= n) return;
    uint c = (chan_dim <= 1u) ? 0u : ((i / inner) % chan_dim);
    float s = as_float(affine[2u * c]);
    int zp = (int)affine[2u * c + 1u];
    __global const char* q = (__global const char*)arena + q_byte_off;
    int qv = (int)q[i];
    arena[out_off + i] = (float)(qv - zp) * s;
}
