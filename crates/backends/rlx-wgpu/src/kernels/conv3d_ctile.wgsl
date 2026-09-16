// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// 3D NCDHW conv, four output channels per thread.
//
// `conv3d.wgsl` gives one thread one output element. That reloads the whole
// input patch once per output channel, and it recomputes the padding bounds
// test inside the input-channel loop — `c_in` times per tap for a result that
// depends on neither. This kernel fixes both:
//
//   * taps move outside the channel loop, so the bounds test and the input
//     spatial offset are computed once per tap rather than once per (tap, ci);
//   * four accumulators per thread, so each loaded input value feeds four
//     fused multiply-adds instead of one.
//
// WGSL has no cooperative-matrix type, so this is register tiling rather than
// the implicit-GEMM/MMA design the Metal backend uses. The arithmetic is
// identical to the scalar kernel's, accumulated in the same order.
//
// Ungrouped only. With groups > 1 the four channels of a tile can straddle a
// group boundary and then read *different* input channels, which is precisely
// the reuse this depends on; `lower` sends grouped convs to `conv3d`.
//
// The last tile of a c_out that is not a multiple of four reads a few weights
// past the end of the filter. Those land inside the arena (or are clamped —
// WGSL bounds-checks storage access either way), accumulate into a lane whose
// result is never stored, and cannot fault.

struct Params {
    n: u32, c_in: u32, c_out: u32,
    d: u32, h: u32, w: u32,
    d_out: u32, h_out: u32, w_out: u32,
    kd: u32, kh: u32, kw: u32,
    sd: u32, sh: u32, sw: u32,
    pd: u32, ph: u32, pw: u32,
    dd: u32, dh: u32, dw: u32,
    groups: u32,
    in_off: u32, w_off: u32, out_off: u32,
    bias_off: u32,   // per-output-channel bias; valid when has_bias != 0
    has_bias: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

// Four, measured. Eight accumulators doubles the reuse per loaded input and
// is worse: against the same scalar control in one process, four channels per
// thread is 4.18x and eight is 2.49x. Register pressure costs more occupancy
// than the extra reuse buys — the same trade that ruled out wider tiles on the
// Metal backend.
const CO_TILE: u32 = 4u;

@compute @workgroup_size(64)
fn conv3d_ctile(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let co_tiles = (params.c_out + CO_TILE - 1u) / CO_TILE;
    let total = params.n * co_tiles * params.d_out * params.h_out * params.w_out;
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= total) { return; }

    let wo = i % params.w_out;
    let q1 = i / params.w_out;
    let ho = q1 % params.h_out;
    let q2 = q1 / params.h_out;
    let do_ = q2 % params.d_out;
    let q3 = q2 / params.d_out;
    let cot = q3 % co_tiles;
    let nn = q3 / co_tiles;
    let co0 = cot * CO_TILE;

    let ktap = params.kd * params.kh * params.kw;
    let kstride = params.c_in * ktap;          // groups == 1
    let dhw = params.d * params.h * params.w;
    let in_base = params.in_off + nn * params.c_in * dhw;
    let w_base = params.w_off + co0 * kstride;

    var a0: f32 = 0.0;
    var a1: f32 = 0.0;
    var a2: f32 = 0.0;
    var a3: f32 = 0.0;

    for (var kd: u32 = 0u; kd < params.kd; kd = kd + 1u) {
      for (var kh: u32 = 0u; kh < params.kh; kh = kh + 1u) {
        for (var kw: u32 = 0u; kw < params.kw; kw = kw + 1u) {
            let in_d_s = i32(do_ * params.sd + kd * params.dd) - i32(params.pd);
            let in_h_s = i32(ho  * params.sh + kh * params.dh) - i32(params.ph);
            let in_w_s = i32(wo  * params.sw + kw * params.dw) - i32(params.pw);
            if (in_d_s < 0 || in_h_s < 0 || in_w_s < 0
                || in_d_s >= i32(params.d)
                || in_h_s >= i32(params.h)
                || in_w_s >= i32(params.w)) {
                continue;
            }
            let sp = (u32(in_d_s) * params.h + u32(in_h_s)) * params.w + u32(in_w_s);
            let tap = (kd * params.kh + kh) * params.kw + kw;
            var src = in_base + sp;
            var wt = w_base + tap;
            for (var ci: u32 = 0u; ci < params.c_in; ci = ci + 1u) {
                let v = arena[src];
                a0 = a0 + v * arena[wt];
                a1 = a1 + v * arena[wt + kstride];
                a2 = a2 + v * arena[wt + 2u * kstride];
                a3 = a3 + v * arena[wt + 3u * kstride];
                src = src + dhw;
                wt = wt + ktap;
            }
        }
      }
    }

    if (params.has_bias != 0u) {
        a0 = a0 + arena[params.bias_off + co0];
        a1 = a1 + arena[params.bias_off + co0 + 1u];
        a2 = a2 + arena[params.bias_off + co0 + 2u];
        a3 = a3 + arena[params.bias_off + co0 + 3u];
    }
    let out_dhw = params.d_out * params.h_out * params.w_out;
    let out_sp = params.out_off + (do_ * params.h_out + ho) * params.w_out + wo;
    let out_base = out_sp + nn * params.c_out * out_dhw + co0 * out_dhw;
    if (co0 + 0u < params.c_out) { arena[out_base] = a0; }
    if (co0 + 1u < params.c_out) { arena[out_base + out_dhw] = a1; }
    if (co0 + 2u < params.c_out) { arena[out_base + 2u * out_dhw] = a2; }
    if (co0 + 3u < params.c_out) { arena[out_base + 3u * out_dhw] = a3; }
}
