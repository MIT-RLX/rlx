// FakeQuantize PerBatch: one work-item per channel.
// s = max(|x|) / q_max, then clamp(round(x/s), -q_max, q_max) * s.
__kernel void fake_quantize_perbatch(__global float* arena,
                                     uint n, uint chan_dim, uint inner,
                                     float q_max,
                                     uint in_off, uint out_off) {
    uint c = get_global_id(0);
    if (c >= chan_dim) return;

    float max_abs = 0.0f;
    uint stride = chan_dim * inner;
    uint outer = (stride == 0u) ? 0u : (n / stride);
    for (uint o = 0u; o < outer; o++) {
        uint base = o * stride + c * inner;
        for (uint j = 0u; j < inner; j++) {
            max_abs = fmax(max_abs, fabs(arena[in_off + base + j]));
        }
    }
    if (outer * stride != n) {
        for (uint i = 0u; i < n; i++) {
            uint ch = (chan_dim <= 1u) ? 0u : ((i / inner) % chan_dim);
            if (ch == c) {
                max_abs = fmax(max_abs, fabs(arena[in_off + i]));
            }
        }
    }

    float s = fmax(max_abs / q_max, 1e-12f);

    for (uint o = 0u; o < outer; o++) {
        uint base = o * stride + c * inner;
        for (uint j = 0u; j < inner; j++) {
            uint idx = base + j;
            float scaled = arena[in_off + idx] / s;
            float sgn = (scaled > 0.0f) - (scaled < 0.0f);
            float rounded = sgn * floor(fabs(scaled) + 0.5f);
            float qv = clamp(rounded, -q_max, q_max);
            arena[out_off + idx] = qv * s;
        }
    }
    if (outer * stride != n) {
        for (uint i = 0u; i < n; i++) {
            uint ch = (chan_dim <= 1u) ? 0u : ((i / inner) % chan_dim);
            if (ch == c) {
                float scaled = arena[in_off + i] / s;
                float sgn = (scaled > 0.0f) - (scaled < 0.0f);
                float rounded = sgn * floor(fabs(scaled) + 0.5f);
                float qv = clamp(rounded, -q_max, q_max);
                arena[out_off + i] = qv * s;
            }
        }
    }
}
