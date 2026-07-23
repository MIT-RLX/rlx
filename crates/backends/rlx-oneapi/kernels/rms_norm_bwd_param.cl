// RMSNorm backward w.r.t. gamma (wrt=1) or beta (wrt=2). Single serial work-item.
__kernel void rms_norm_bwd_param(__global float* arena,
                                 uint outer, uint inner,
                                 uint x_off, uint dy_off, uint out_off,
                                 float eps, uint wrt) {
    if (get_global_id(0) != 0 || inner == 0u) return;
    float n_inv = 1.0f / (float)inner;
    for (uint i = 0; i < inner; i++) arena[out_off + i] = 0.0f;

    for (uint row = 0; row < outer; row++) {
        uint x_base = x_off + row * inner;
        uint dy_base = dy_off + row * inner;
        if (wrt == 2u) {
            for (uint i = 0; i < inner; i++)
                arena[out_off + i] += arena[dy_base + i];
            continue;
        }
        // wrt == 1: dgamma
        float sumsq = 0.0f;
        for (uint i = 0; i < inner; i++) {
            float xv = arena[x_base + i];
            sumsq += xv * xv;
        }
        float inv_r = rsqrt(sumsq * n_inv + eps);
        for (uint i = 0; i < inner; i++)
            arena[out_off + i] += arena[dy_base + i] * arena[x_base + i] * inv_r;
    }
}
