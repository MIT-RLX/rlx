// FakeQuantize Fixed: clamp(round(x/s), -q_max, q_max) * s.
// One work-item per element. Rounding matches Rust f32::round (half away from 0).
__kernel void fake_quantize_fixed(__global float* arena,
                                  uint n, uint chan_dim, uint inner,
                                  float q_max,
                                  uint in_off, uint scale_off, uint out_off) {
    uint i = get_global_id(0);
    if (i >= n) return;
    uint c = (chan_dim <= 1u) ? 0u : ((i / inner) % chan_dim);
    float s = fmax(arena[scale_off + c], 1e-12f);
    float scaled = arena[in_off + i] / s;
    float sgn = (scaled > 0.0f) - (scaled < 0.0f);
    float rounded = sgn * floor(fabs(scaled) + 0.5f);
    float qv = clamp(rounded, -q_max, q_max);
    arena[out_off + i] = qv * s;
}
