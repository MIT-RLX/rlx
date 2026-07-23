// Element-wise C64 conjugate: (re, -im).
__kernel void conjugate_c64(__global float* arena,
                            uint n, uint src_off, uint dst_off) {
    uint k = get_global_id(0);
    if (k >= n) return;
    arena[dst_off + 2u * k] = arena[src_off + 2u * k];
    arena[dst_off + 2u * k + 1u] = -arena[src_off + 2u * k + 1u];
}
