// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Nearest-neighbor 2× upsample on planar NCHW. One thread per output pixel.

extern "C" __global__ void resize_nearest_2x(
    float* arena,
    unsigned int src_off,
    unsigned int dst_off,
    unsigned int n,
    unsigned int c,
    unsigned int h,
    unsigned int w
) {
    unsigned int h2 = h * 2u;
    unsigned int w2 = w * 2u;
    unsigned int total = n * c * h2 * w2;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int wo = i % w2;
    unsigned int q1 = i / w2;
    unsigned int ho = q1 % h2;
    unsigned int q2 = q1 / h2;
    unsigned int co = q2 % c;
    unsigned int bn = q2 / c;
    unsigned int hi = ho / 2u;
    unsigned int wi = wo / 2u;
    float v = arena[src_off + (((bn * c + co) * h + hi) * w + wi)];
    arena[dst_off + i] = v;
}
