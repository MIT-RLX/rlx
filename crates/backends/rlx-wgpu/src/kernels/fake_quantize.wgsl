// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

// FakeQuantize forward: clamp(round(x / s), -q_max, q_max) * s.
// Matches `rlx_cpu::thunk::ops::quant::exec_fake_quantize` for Fixed and
// PerBatch (EMA stays on HostOp). Channel layout: c = (i / inner) % chan_dim.

struct Params {
    n: u32,        // total elements
    chan_dim: u32, // 1 when axis=None
    inner: u32,    // product of dims after channel axis (n when axis=None)
    q_max: f32,    // 127 / 7 / 1
    in_off: u32,
    scale_off: u32, // Fixed only; unused for PerBatch
    out_off: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

fn apply_fq(x: f32, s: f32, q_max: f32) -> f32 {
    // Match Rust `f32::round` (half away from zero), not WGSL `round` (ties to even).
    let scaled = x / s;
    let rounded = sign(scaled) * floor(abs(scaled) + 0.5);
    let qv = clamp(rounded, -q_max, q_max);
    return qv * s;
}

fn channel_of(i: u32) -> u32 {
    if (params.chan_dim <= 1u) {
        return 0u;
    }
    return (i / params.inner) % params.chan_dim;
}

// One thread per element. Scale comes from `scale_off[c]` (Fixed).
@compute @workgroup_size(64)
fn fake_quantize_fixed(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n) { return; }
    let c = channel_of(i);
    let s = max(arena[params.scale_off + c], 1e-12);
    let x = arena[params.in_off + i];
    arena[params.out_off + i] = apply_fq(x, s, params.q_max);
}

// One thread per channel. Computes s = max(|x|) / q_max, then quantizes
// every element belonging to that channel (PerBatch).
@compute @workgroup_size(64)
fn fake_quantize_perbatch(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let c = gid.x + gid.y * ngs.x * 64u;
    if (c >= params.chan_dim) { return; }

    var max_abs: f32 = 0.0;
    let stride = params.chan_dim * params.inner;
    let outer = params.n / max(stride, 1u);
    for (var o: u32 = 0u; o < outer; o = o + 1u) {
        let base = o * stride + c * params.inner;
        for (var j: u32 = 0u; j < params.inner; j = j + 1u) {
            let a = abs(arena[params.in_off + base + j]);
            max_abs = max(max_abs, a);
        }
    }
    // axis=None: chan_dim=1, inner=n → outer=1, single scan of all elements.
    // When n is not a multiple of stride (shouldn't happen), fall back to a
    // full-tensor scan for this channel.
    if (outer * stride != params.n) {
        for (var i: u32 = 0u; i < params.n; i = i + 1u) {
            if (channel_of(i) == c) {
                max_abs = max(max_abs, abs(arena[params.in_off + i]));
            }
        }
    }

    let s = max(max_abs / params.q_max, 1e-12);

    for (var o: u32 = 0u; o < outer; o = o + 1u) {
        let base = o * stride + c * params.inner;
        for (var j: u32 = 0u; j < params.inner; j = j + 1u) {
            let idx = base + j;
            let x = arena[params.in_off + idx];
            arena[params.out_off + idx] = apply_fq(x, s, params.q_max);
        }
    }
    if (outer * stride != params.n) {
        for (var i: u32 = 0u; i < params.n; i = i + 1u) {
            if (channel_of(i) == c) {
                let x = arena[params.in_off + i];
                arena[params.out_off + i] = apply_fq(x, s, params.q_max);
            }
        }
    }
}
