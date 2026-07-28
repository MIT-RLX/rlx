// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// Welch PSD accumulate + top-K peaks from block-layout spectra.
// One thread per welch_batch row; matches rlx_ir::audio::welch_peaks_block_f32.

struct Params {
    spec_off: u32,
    dst_off: u32,
    welch_batch: u32,
    n_fft: u32,
    n_segments: u32,
    k: u32,
    n_bins: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

const NEG_INF: f32 = -3.4e38;
const MAX_BINS: u32 = 512u;

@compute @workgroup_size(64)
fn welch_peaks_gpu(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let row = gid.x + gid.y * nwg.x * 64u;
    if (row >= params.welch_batch) { return; }

    let n_fft = params.n_fft;
    let n_bins = params.n_bins;
    let inv = 1.0 / f32(params.n_segments);
    let row_len = n_fft * 2u;

    var psd: array<f32, MAX_BINS>;
    for (var b: u32 = 0u; b < n_bins; b = b + 1u) {
        psd[b] = 0.0;
    }

    for (var s: u32 = 0u; s < params.n_segments; s = s + 1u) {
        let seg_row = row * params.n_segments + s;
        let base = params.spec_off + seg_row * row_len;
        let re0 = arena[base];
        let im0 = arena[base + n_fft];
        psd[0] = psd[0] + inv * (re0 * re0 + im0 * im0);
        for (var bin: u32 = 1u; bin + 1u < n_bins; bin = bin + 1u) {
            let re = arena[base + bin];
            let im = arena[base + n_fft + bin];
            psd[bin] = psd[bin] + inv * 2.0 * (re * re + im * im);
        }
        if (n_bins > 1u) {
            let bin = n_bins - 1u;
            let re = arena[base + bin];
            let im = arena[base + n_fft + bin];
            psd[bin] = psd[bin] + inv * (re * re + im * im);
        }
    }

    let out_base = params.dst_off + row * params.k * 2u;
    for (var step: u32 = 0u; step < params.k; step = step + 1u) {
        var best_v: f32 = NEG_INF;
        var best_i: u32 = 0u;
        for (var j: u32 = 0u; j < n_bins; j = j + 1u) {
            var taken = false;
            for (var p: u32 = 0u; p < step; p = p + 1u) {
                if (u32(arena[out_base + p * 2u]) == j) {
                    taken = true;
                    break;
                }
            }
            if (taken) { continue; }
            let v = psd[j];
            if (v > best_v || (v == best_v && j < best_i)) {
                best_v = v;
                best_i = j;
            }
        }
        arena[out_base + step * 2u] = f32(best_i);
        arena[out_base + step * 2u + 1u] = best_v;
    }
}
