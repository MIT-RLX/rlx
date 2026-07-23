// Welch PSD accumulate + top-K peaks (one work-item per welch_batch row).
// Matches rlx_ir::audio::welch_peaks_block_f32 / wgpu welch_peaks_gpu.wgsl.

#define NEG_INF (-3.4e38f)
#define MAX_BINS 512u

__kernel void welch_peaks(__global float* arena,
                     uint spec_off, uint dst_off,
                     uint welch_batch, uint n_fft,
                     uint n_segments, uint k, uint n_bins) {
    uint row = get_global_id(0);
    if (row >= welch_batch) return;

    float inv = 1.0f / (float)n_segments;
    uint row_len = n_fft * 2u;

    float psd[MAX_BINS];
    for (uint b = 0u; b < n_bins; b++) {
        psd[b] = 0.0f;
    }

    for (uint s = 0u; s < n_segments; s++) {
        uint seg_row = row * n_segments + s;
        uint base = spec_off + seg_row * row_len;
        float re0 = arena[base];
        float im0 = arena[base + n_fft];
        psd[0] = psd[0] + inv * (re0 * re0 + im0 * im0);
        for (uint bin = 1u; bin + 1u < n_bins; bin++) {
            float re = arena[base + bin];
            float im = arena[base + n_fft + bin];
            psd[bin] = psd[bin] + inv * 2.0f * (re * re + im * im);
        }
        if (n_bins > 1u) {
            uint bin = n_bins - 1u;
            float re = arena[base + bin];
            float im = arena[base + n_fft + bin];
            psd[bin] = psd[bin] + inv * (re * re + im * im);
        }
    }

    uint out_base = dst_off + row * k * 2u;
    for (uint step = 0u; step < k; step++) {
        float best_v = NEG_INF;
        uint best_i = 0u;
        for (uint j = 0u; j < n_bins; j++) {
            int taken = 0;
            for (uint p = 0u; p < step; p++) {
                if ((uint)arena[out_base + p * 2u] == j) {
                    taken = 1;
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
