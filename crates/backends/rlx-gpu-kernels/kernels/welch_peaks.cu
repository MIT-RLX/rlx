// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// RLX — versatile ML compiler + runtime.
// Block-layout segment spectra [outer, 2*n_fft] → packed top-K peaks [batch, k*2].
// One thread per welch_batch row; mirrors rlx_ir::audio::welch_peaks_block_f32.

extern "C" __global__ void welch_peaks_gpu(
    float* arena,
    unsigned int spec_off,
    unsigned int dst_off,
    unsigned int welch_batch,
    unsigned int n_fft,
    unsigned int n_segments,
    unsigned int k,
    unsigned int n_bins
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= welch_batch) return;

    const float inv = 1.0f / (float)n_segments;
    const unsigned int row_len = n_fft * 2u;

    float psd[512];
    for (unsigned int b = 0; b < n_bins; ++b) {
        psd[b] = 0.0f;
    }

    for (unsigned int s = 0; s < n_segments; ++s) {
        unsigned int seg_row = row * n_segments + s;
        unsigned int base = spec_off + seg_row * row_len;
        float re0 = arena[base];
        float im0 = arena[base + n_fft];
        psd[0] += inv * (re0 * re0 + im0 * im0);
        for (unsigned int bin = 1; bin + 1u < n_bins; ++bin) {
            float re = arena[base + bin];
            float im = arena[base + n_fft + bin];
            psd[bin] += inv * 2.0f * (re * re + im * im);
        }
        if (n_bins > 1u) {
            unsigned int bin = n_bins - 1u;
            float re = arena[base + bin];
            float im = arena[base + n_fft + bin];
            psd[bin] += inv * (re * re + im * im);
        }
    }

    unsigned int out_base = dst_off + row * k * 2u;
    for (unsigned int step = 0; step < k; ++step) {
        float best_v = -3.4e38f;
        unsigned int best_i = 0u;
        for (unsigned int j = 0; j < n_bins; ++j) {
            bool taken = false;
            for (unsigned int p = 0; p < step; ++p) {
                if ((unsigned int)arena[out_base + p * 2u] == j) {
                    taken = true;
                    break;
                }
            }
            if (taken) continue;
            float v = psd[j];
            if (v > best_v || (v == best_v && j < best_i)) {
                best_v = v;
                best_i = j;
            }
        }
        arena[out_base + step * 2u] = (float)best_i;
        arena[out_base + step * 2u + 1u] = best_v;
    }
}
