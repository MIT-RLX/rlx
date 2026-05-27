// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// NCHW LayerNorm2d (SAM / candle semantics): normalize across channels
// at each spatial position. One thread per (batch, ho, wo).

extern "C" __global__ void layer_norm2d(
    float* arena,
    unsigned int src_off,
    unsigned int g_off,
    unsigned int b_off,
    unsigned int dst_off,
    unsigned int n,
    unsigned int c,
    unsigned int h,
    unsigned int w,
    unsigned int eps_bits
) {
    unsigned int total = n * h * w;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int wo = i % w;
    unsigned int ho = (i / w) % h;
    unsigned int bn = i / (h * w);
    unsigned int spatial = h * w;
    unsigned int pos = ho * w + wo;

    float mean = 0.0f;
    for (unsigned int ch = 0; ch < c; ++ch) {
        mean += arena[src_off + ((bn * c + ch) * spatial) + pos];
    }
    mean /= (float)c;

    float var = 0.0f;
    for (unsigned int ch = 0; ch < c; ++ch) {
        float d = arena[src_off + ((bn * c + ch) * spatial) + pos] - mean;
        var += d * d;
    }
    var /= (float)c;

    float eps = __int_as_float((int)eps_bits);
    float inv = rsqrtf(var + eps);

    for (unsigned int ch = 0; ch < c; ++ch) {
        unsigned int idx = ((bn * c + ch) * spatial) + pos;
        float v = (arena[src_off + idx] - mean) * inv;
        arena[dst_off + idx] = v * arena[g_off + ch] + arena[b_off + ch];
    }
}
