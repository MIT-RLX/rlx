// RMSNorm backward w.r.t. input (last-axis rows). One work-item per row.
// y = x * inv_r * gamma + beta, inv_r = rsqrt(mean(x^2)+eps).
__kernel void rms_norm_bwd_input(__global float* arena,
                                 uint outer, uint inner,
                                 uint x_off, uint gamma_off, uint dy_off, uint out_off,
                                 float eps) {
    uint row = get_global_id(0);
    if (row >= outer || inner == 0u) return;
    uint x_base = x_off + row * inner;
    uint dy_base = dy_off + row * inner;
    uint out_base = out_off + row * inner;
    float n_inv = 1.0f / (float)inner;

    float dot = 0.0f;
    float sumsq = 0.0f;
    for (uint i = 0; i < inner; i++) {
        float xv = arena[x_base + i];
        float gv = arena[gamma_off + i];
        float dyv = arena[dy_base + i];
        dot += dyv * gv * xv;
        sumsq += xv * xv;
    }
    dot *= n_inv;
    float inv_r = rsqrt(sumsq * n_inv + eps);
    float inv_r2 = inv_r * inv_r;
    for (uint i = 0; i < inner; i++) {
        float xv = arena[x_base + i];
        float gv = arena[gamma_off + i];
        float dyv = arena[dy_base + i];
        float term = gv * dyv - xv * dot * inv_r2;
        arena[out_base + i] = term * inv_r;
    }
}
