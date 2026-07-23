// Wirtinger VJP of ComplexNormSq: dz = g · z (g real, z C64).
__kernel void complex_norm_sq_backward(__global float* arena,
                                       uint n, uint z_off, uint g_off, uint dz_off) {
    uint k = get_global_id(0);
    if (k >= n) return;
    float re = arena[z_off + 2u * k];
    float im = arena[z_off + 2u * k + 1u];
    float gv = arena[g_off + k];
    arena[dz_off + 2u * k] = gv * re;
    arena[dz_off + 2u * k + 1u] = gv * im;
}
