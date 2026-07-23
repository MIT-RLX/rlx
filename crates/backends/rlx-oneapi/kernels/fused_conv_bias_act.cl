// Fused 2D NCHW conv + bias[+ residual] + optional activation.
// Weight: [c_out, c_in/groups, kh, kw]. `act` matches OneAPI `act_id`
// (unary.cl / backend.rs); 0xFFFF = identity (skip). Residual is a full
// NCHW tensor added before the activation when `has_residual != 0`.

float fca_apply_act(float v, uint act) {
    if (act == 0xFFFFu) return v;
    switch (act) {
        case 0u: return 0.5f * v * (1.0f + erf(v * 0.70710678118f)); // Gelu
        case 1u: { // GeluApprox
            float c = 0.7978845608f;
            float t = tanh(c * (v + 0.044715f * v * v * v));
            return 0.5f * v * (1.0f + t);
        }
        case 2u: return v / (1.0f + exp(-v)); // Silu
        case 3u: return fmax(v, 0.0f); // Relu
        case 4u: return 1.0f / (1.0f + exp(-v)); // Sigmoid
        case 5u: return tanh(v); // Tanh
        case 6u: return exp(v);
        case 7u: return log(v);
        case 8u: return sqrt(v);
        case 9u: return rsqrt(v);
        case 10u: return -v;
        case 11u: return fabs(v);
        case 12u: return sin(v);
        case 13u: return cos(v);
        case 14u: return tan(v);
        case 15u: return atan(v);
        case 16u: return round(v);
        default: return v;
    }
}

__kernel void fused_conv_bias_act(__global float* arena,
                     uint n, uint c_in, uint c_out,
                     uint h, uint w,
                     uint h_out, uint w_out,
                     uint kh, uint kw,
                     uint sh, uint sw,
                     uint ph, uint pw,
                     uint dh, uint dw,
                     uint groups,
                     uint in_off, uint w_off, uint bias_off,
                     uint residual_off, uint out_off,
                     uint has_residual, uint act) {
    uint total = n * c_out * h_out * w_out;
    uint i = get_global_id(0);
    if (i >= total) return;
    uint wo = i % w_out;
    uint q1 = i / w_out;
    uint ho = q1 % h_out;
    uint q2 = q1 / h_out;
    uint co = q2 % c_out;
    uint nn = q2 / c_out;

    uint c_in_per_g = c_in / groups;
    uint c_out_per_g = c_out / groups;
    uint g = co / c_out_per_g;
    uint ci_start = g * c_in_per_g;

    float acc = 0.0f;
    for (uint ci_off = 0u; ci_off < c_in_per_g; ci_off++) {
        uint ci = ci_start + ci_off;
        for (uint ki = 0u; ki < kh; ki++) {
            for (uint kj = 0u; kj < kw; kj++) {
                int ih = (int)(ho * sh + ki * dh) - (int)ph;
                int iw = (int)(wo * sw + kj * dw) - (int)pw;
                if (ih < 0 || iw < 0 || ih >= (int)h || iw >= (int)w) continue;
                float xv = arena[in_off + ((nn * c_in + ci) * h + (uint)ih) * w + (uint)iw];
                float wv = arena[w_off + (((co * c_in_per_g + ci_off) * kh + ki) * kw + kj)];
                acc += xv * wv;
            }
        }
    }
    acc += arena[bias_off + co];
    if (has_residual != 0u)
        acc += arena[residual_off + i];
    arena[out_off + i] = fca_apply_act(acc, act);
}
