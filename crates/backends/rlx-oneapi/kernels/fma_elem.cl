// Single-rounded fused multiply-add: out[i] = fma(a[i], b[i], c[i]).
__kernel void fma_elem(__global float* arena,
                       uint n, uint a_off, uint b_off, uint c_off, uint out_off) {
    uint i = get_global_id(0);
    if (i >= n) return;
    arena[out_off + i] = fma(arena[a_off + i], arena[b_off + i], arena[c_off + i]);
}
