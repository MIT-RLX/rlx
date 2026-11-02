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
    act: u32,
    alpha: u32,       // LeakyReLU slope bits when act == 0xFFFF
    _split: u32, _half: u32,
    in_prelu: u32,    // nonzero: the input through a PReLU of slope `in_alpha`
    in_alpha: u32,    //   as it is loaded (`conv3d_k3_c8` only)
    res_max: u32,     // nonzero: the store's `max` with the output-shaped
    res_off: u32,     //   tensor at `res_off` (`conv3d_k3_c8` only)
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

// Half-packed activations (`unfuse::pack_half_activations`; the pipeline's
// overrides, `half_kernel`): the input and/or the output is two f16 to an
// arena word, element `e` of the tensor in word `off + e / 2`. Loads widen,
// stores round; the arithmetic between is f32 as ever.
override IN_HALF: bool = false;
override OUT_HALF: bool = false;

fn ld(i: u32) -> f32 {
    if (IN_HALF) {
        let e = i - params.in_off;
        return unpack2x16float(bitcast<u32>(arena[params.in_off + (e >> 1u)]))[e & 1u];
    }
    return arena[i];
}

// A half-packed input element `i`'s word and which half of it.
fn hword(i: u32) -> u32 { return params.in_off + ((i - params.in_off) >> 1u); }
fn hodd(i: u32) -> bool { return ((i - params.in_off) & 1u) != 0u; }
fn hpair(q: u32) -> vec2<f32> { return unpack2x16float(bitcast<u32>(arena[q])); }

// elements `o` (even within the tensor) and `o + 1` of the output
fn put2(o: u32, a: f32, b: f32) {
    let e = o - params.out_off;
    put(params.out_off + (e >> 1u), bitcast<f32>(pack2x16float(vec2<f32>(a, b))));
}

// `params.act`: 0 none, 0xFFFF LeakyReLU `max(v, alpha * v)`, else the
// activation's `opcode_relu_first + 1` — folded from what read this conv
// (`fold_conv_activation`). The same arithmetic as the kernels it replaces
// (the unary kernel's generated `rlx_activation_apply`, or a `*` and a `max`),
// so bit-identical.
fn conv_act(v: f32) -> f32 {
    if (params.act == 0u) { return v; }
    if (params.act == 0xFFFFu) { return max(v, v * bitcast<f32>(params.alpha)); }
    return rlx_activation_apply(params.act - 1u, v);
}

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
    if (co0 + 0u < params.c_out) { put(out_base, conv_act(a0)); }
    if (co0 + 1u < params.c_out) { put(out_base + out_dhw, conv_act(a1)); }
    if (co0 + 2u < params.c_out) { put(out_base + 2u * out_dhw, conv_act(a2)); }
    if (co0 + 3u < params.c_out) { put(out_base + 3u * out_dhw, conv_act(a3)); }
}

// Four output channels × two neighbouring W positions per thread: each
// weight load feeds two multiply-adds (positions) and each input load four
// (channels) — 0.75 loads per multiply-add against `conv3d_ctile`'s 1.25.
// Same taps-then-channels order per output, so the same sums.
@compute @workgroup_size(64)
fn conv3d_ctile_w2(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let co_tiles = (params.c_out + CO_TILE - 1u) / CO_TILE;
    let w_pairs = (params.w_out + 1u) / 2u;
    let total = params.n * co_tiles * params.d_out * params.h_out * w_pairs;
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= total) { return; }

    let wo = (i % w_pairs) * 2u;
    let q1 = i / w_pairs;
    let ho = q1 % params.h_out;
    let q2 = q1 / params.h_out;
    let do_ = q2 % params.d_out;
    let q3 = q2 / params.d_out;
    let cot = q3 % co_tiles;
    let nn = q3 / co_tiles;
    let co0 = cot * CO_TILE;
    let has_b = wo + 1u < params.w_out;

    let ktap = params.kd * params.kh * params.kw;
    let kstride = params.c_in * ktap;
    let dhw = params.d * params.h * params.w;
    let in_base = params.in_off + nn * params.c_in * dhw;
    let w_base = params.w_off + co0 * kstride;

    var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;
    var b0: f32 = 0.0; var b1: f32 = 0.0; var b2: f32 = 0.0; var b3: f32 = 0.0;

    for (var kd: u32 = 0u; kd < params.kd; kd = kd + 1u) {
      let in_d_s = i32(do_ * params.sd + kd * params.dd) - i32(params.pd);
      if (in_d_s < 0 || in_d_s >= i32(params.d)) { continue; }
      for (var kh: u32 = 0u; kh < params.kh; kh = kh + 1u) {
        let in_h_s = i32(ho * params.sh + kh * params.dh) - i32(params.ph);
        if (in_h_s < 0 || in_h_s >= i32(params.h)) { continue; }
        let row = (u32(in_d_s) * params.h + u32(in_h_s)) * params.w;
        for (var kw: u32 = 0u; kw < params.kw; kw = kw + 1u) {
            let wa = i32(wo * params.sw + kw * params.dw) - i32(params.pw);
            let wb = wa + i32(params.sw);
            let ok_a = wa >= 0 && wa < i32(params.w);
            let ok_b = has_b && wb >= 0 && wb < i32(params.w);
            if (!ok_a && !ok_b) { continue; }
            let tap = (kd * params.kh + kh) * params.kw + kw;
            // an out-of-range position reads its partner's voxel and adds 0
            let sa = in_base + row + u32(select(wb, wa, ok_a));
            let sb = in_base + row + u32(select(wa, wb, ok_b));
            let ma = select(0.0, 1.0, ok_a);
            let mb = select(0.0, 1.0, ok_b);
            var src_a = sa;
            var src_b = sb;
            var wt = w_base + tap;
            for (var ci: u32 = 0u; ci < params.c_in; ci = ci + 1u) {
                let va = arena[src_a] * ma;
                let vb = arena[src_b] * mb;
                let w0 = arena[wt];
                let w1 = arena[wt + kstride];
                let w2 = arena[wt + 2u * kstride];
                let w3 = arena[wt + 3u * kstride];
                a0 = a0 + va * w0; a1 = a1 + va * w1; a2 = a2 + va * w2; a3 = a3 + va * w3;
                b0 = b0 + vb * w0; b1 = b1 + vb * w1; b2 = b2 + vb * w2; b3 = b3 + vb * w3;
                src_a = src_a + dhw;
                src_b = src_b + dhw;
                wt = wt + ktap;
            }
        }
      }
    }

    if (params.has_bias != 0u) {
        let c0 = arena[params.bias_off + co0];
        let c1 = arena[params.bias_off + co0 + 1u];
        let c2 = arena[params.bias_off + co0 + 2u];
        let c3 = arena[params.bias_off + co0 + 3u];
        a0 = a0 + c0; a1 = a1 + c1; a2 = a2 + c2; a3 = a3 + c3;
        b0 = b0 + c0; b1 = b1 + c1; b2 = b2 + c2; b3 = b3 + c3;
    }
    let out_dhw = params.d_out * params.h_out * params.w_out;
    let out_sp = params.out_off + (do_ * params.h_out + ho) * params.w_out + wo;
    let o = out_sp + nn * params.c_out * out_dhw + co0 * out_dhw;
    if (co0 + 0u < params.c_out) { put(o, conv_act(a0)); if (has_b) { put(o + 1u, conv_act(b0)); } }
    if (co0 + 1u < params.c_out) { put(o + out_dhw, conv_act(a1)); if (has_b) { put(o + out_dhw + 1u, conv_act(b1)); } }
    if (co0 + 2u < params.c_out) { put(o + 2u * out_dhw, conv_act(a2)); if (has_b) { put(o + 2u * out_dhw + 1u, conv_act(b2)); } }
    if (co0 + 3u < params.c_out) { put(o + 3u * out_dhw, conv_act(a3)); if (has_b) { put(o + 3u * out_dhw + 1u, conv_act(b3)); } }
}

// Four output channels × four neighbouring W positions per thread, the four
// channels of a position in one vec4: each weight load feeds four
// multiply-adds and each input load four — 0.5 loads per multiply-add.
// Same taps-then-channels order per output as `conv3d_ctile`.
@compute @workgroup_size(64)
fn conv3d_ctile_w4(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let co_tiles = (params.c_out + CO_TILE - 1u) / CO_TILE;
    let w_quads = (params.w_out + 3u) / 4u;
    let total = params.n * co_tiles * params.d_out * params.h_out * w_quads;
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= total) { return; }

    let wo = (i % w_quads) * 4u;
    let q1 = i / w_quads;
    let ho = q1 % params.h_out;
    let q2 = q1 / params.h_out;
    let do_ = q2 % params.d_out;
    let q3 = q2 / params.d_out;
    let cot = q3 % co_tiles;
    let nn = q3 / co_tiles;
    let co0 = cot * CO_TILE;

    let ktap = params.kd * params.kh * params.kw;
    let kstride = params.c_in * ktap;
    let dhw = params.d * params.h * params.w;
    let in_base = params.in_off + nn * params.c_in * dhw;
    let w_base = params.w_off + co0 * kstride;
    let wmax = i32(params.w) - 1;

    var a = vec4<f32>(0.0); var b = vec4<f32>(0.0);
    var c = vec4<f32>(0.0); var e = vec4<f32>(0.0);

    for (var kd: u32 = 0u; kd < params.kd; kd = kd + 1u) {
      let in_d_s = i32(do_ * params.sd + kd * params.dd) - i32(params.pd);
      if (in_d_s < 0 || in_d_s >= i32(params.d)) { continue; }
      for (var kh: u32 = 0u; kh < params.kh; kh = kh + 1u) {
        let in_h_s = i32(ho * params.sh + kh * params.dh) - i32(params.ph);
        if (in_h_s < 0 || in_h_s >= i32(params.h)) { continue; }
        let row = in_base + (u32(in_d_s) * params.h + u32(in_h_s)) * params.w;
        for (var kw: u32 = 0u; kw < params.kw; kw = kw + 1u) {
            let w0 = i32(wo * params.sw + kw * params.dw) - i32(params.pw);
            let s = i32(params.sw);
            let wa = w0; let wb = w0 + s; let wc = w0 + 2 * s; let wd = w0 + 3 * s;
            // positions past w_out or taps outside the volume add 0 (read a
            // clamped, in-range voxel)
            let ma = select(0.0, 1.0, wa >= 0 && wa <= wmax);
            let mb = select(0.0, 1.0, wo + 1u < params.w_out && wb >= 0 && wb <= wmax);
            let mc = select(0.0, 1.0, wo + 2u < params.w_out && wc >= 0 && wc <= wmax);
            let md = select(0.0, 1.0, wo + 3u < params.w_out && wd >= 0 && wd <= wmax);
            if (ma + mb + mc + md == 0.0) { continue; }
            let tap = (kd * params.kh + kh) * params.kw + kw;
            var sa = row + u32(clamp(wa, 0, wmax));
            var sb = row + u32(clamp(wb, 0, wmax));
            var sc = row + u32(clamp(wc, 0, wmax));
            var sd = row + u32(clamp(wd, 0, wmax));
            var wt = w_base + tap;
            for (var ci: u32 = 0u; ci < params.c_in; ci = ci + 1u) {
                let wv = vec4<f32>(arena[wt], arena[wt + kstride],
                                   arena[wt + 2u * kstride], arena[wt + 3u * kstride]);
                a = a + (ld(sa) * ma) * wv;
                b = b + (ld(sb) * mb) * wv;
                c = c + (ld(sc) * mc) * wv;
                e = e + (ld(sd) * md) * wv;
                sa = sa + dhw; sb = sb + dhw; sc = sc + dhw; sd = sd + dhw;
                wt = wt + ktap;
            }
        }
      }
    }

    if (params.has_bias != 0u) {
        let bv = vec4<f32>(arena[params.bias_off + co0], arena[params.bias_off + co0 + 1u],
                           arena[params.bias_off + co0 + 2u], arena[params.bias_off + co0 + 3u]);
        a = a + bv; b = b + bv; c = c + bv; e = e + bv;
    }
    let out_dhw = params.d_out * params.h_out * params.w_out;
    let out_sp = params.out_off + (do_ * params.h_out + ho) * params.w_out + wo;
    let o0 = out_sp + nn * params.c_out * out_dhw + co0 * out_dhw;
    let nw = min(4u, params.w_out - wo);
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        if (co0 + k >= params.c_out) { break; }
        let o = o0 + k * out_dhw;
        put(o, conv_act(a[k]));
        if (nw > 1u) { put(o + 1u, conv_act(b[k])); }
        if (nw > 2u) { put(o + 2u, conv_act(c[k])); }
        if (nw > 3u) { put(o + 3u, conv_act(e[k])); }
    }
}

// Eight output channels × four neighbouring W positions per thread, two
// vec4s per position: each input load feeds eight multiply-adds — half the
// input traffic of `conv3d_ctile_w4`, whose four-channel tiles re-read every
// input voxel once per tile. Same taps-then-channels order per output, so
// the same sums. Measured on MindGrab (M4 Pro): 1.85 → 1.25 s of convs.
// Sixteen channels × two positions was no faster, weights staged in
// workgroup memory ten times slower, and a mask-free path for taps inside
// the volume made no difference.
@compute @workgroup_size(64)
fn conv3d_ctile_w4c8(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let co_tiles = (params.c_out + 7u) / 8u;
    let w_quads = (params.w_out + 3u) / 4u;
    let total = params.n * co_tiles * params.d_out * params.h_out * w_quads;
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= total) { return; }

    let wo = (i % w_quads) * 4u;
    let q1 = i / w_quads;
    let ho = q1 % params.h_out;
    let q2 = q1 / params.h_out;
    let do_ = q2 % params.d_out;
    let q3 = q2 / params.d_out;
    let cot = q3 % co_tiles;
    let nn = q3 / co_tiles;
    let co0 = cot * 8u;

    let ktap = params.kd * params.kh * params.kw;
    let kstride = params.c_in * ktap;
    let dhw = params.d * params.h * params.w;
    let in_base = params.in_off + nn * params.c_in * dhw;
    let w_base = params.w_off + co0 * kstride;
    let wmax = i32(params.w) - 1;

    var a = vec4<f32>(0.0); var b = vec4<f32>(0.0);
    var c = vec4<f32>(0.0); var e = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0); var b2 = vec4<f32>(0.0);
    var c2 = vec4<f32>(0.0); var e2 = vec4<f32>(0.0);

    for (var kd: u32 = 0u; kd < params.kd; kd = kd + 1u) {
      let in_d_s = i32(do_ * params.sd + kd * params.dd) - i32(params.pd);
      if (in_d_s < 0 || in_d_s >= i32(params.d)) { continue; }
      for (var kh: u32 = 0u; kh < params.kh; kh = kh + 1u) {
        let in_h_s = i32(ho * params.sh + kh * params.dh) - i32(params.ph);
        if (in_h_s < 0 || in_h_s >= i32(params.h)) { continue; }
        let row = in_base + (u32(in_d_s) * params.h + u32(in_h_s)) * params.w;
        for (var kw: u32 = 0u; kw < params.kw; kw = kw + 1u) {
            let w0 = i32(wo * params.sw + kw * params.dw) - i32(params.pw);
            let s = i32(params.sw);
            let wa = w0; let wb = w0 + s; let wc = w0 + 2 * s; let wd = w0 + 3 * s;
            let ma = select(0.0, 1.0, wa >= 0 && wa <= wmax);
            let mb = select(0.0, 1.0, wo + 1u < params.w_out && wb >= 0 && wb <= wmax);
            let mc = select(0.0, 1.0, wo + 2u < params.w_out && wc >= 0 && wc <= wmax);
            let md = select(0.0, 1.0, wo + 3u < params.w_out && wd >= 0 && wd <= wmax);
            if (ma + mb + mc + md == 0.0) { continue; }
            let tap = (kd * params.kh + kh) * params.kw + kw;
            var wt = w_base + tap;
            var sa = row + u32(clamp(wa, 0, wmax));
            var sb = row + u32(clamp(wb, 0, wmax));
            var sc = row + u32(clamp(wc, 0, wmax));
            var sd = row + u32(clamp(wd, 0, wmax));
            if (IN_HALF) {
                // halves: a channel is `dhw / 2` words on (`dhw` is even —
                // the producer's `w_out` is), and a position's half is the
                // same for every channel. Four neighbours inside the volume
                // are two words, or three when the first is an odd half.
                let hd = dhw >> 1u;
                let run = s == 1 && ma * mb * mc * md == 1.0;
                var qa = hword(sa); var qb = hword(sb); var qc = hword(sc); var qd = hword(sd);
                let oa = hodd(sa); let ob = hodd(sb); let oc = hodd(sc); let od = hodd(sd);
                for (var ci: u32 = 0u; ci < params.c_in; ci = ci + 1u) {
                    let wv = vec4<f32>(arena[wt], arena[wt + kstride],
                                       arena[wt + 2u * kstride], arena[wt + 3u * kstride]);
                    let wt4 = wt + 4u * kstride;
                    let wv2 = vec4<f32>(arena[wt4], arena[wt4 + kstride],
                                        arena[wt4 + 2u * kstride], arena[wt4 + 3u * kstride]);
                    var va: f32; var vb: f32; var vc: f32; var vd: f32;
                    if (run && !oa) {
                        let p0 = hpair(qa); let p1 = hpair(qa + 1u);
                        va = p0.x; vb = p0.y; vc = p1.x; vd = p1.y;
                    } else if (run) {
                        let p0 = hpair(qa); let p1 = hpair(qa + 1u); let p2 = hpair(qa + 2u);
                        va = p0.y; vb = p1.x; vc = p1.y; vd = p2.x;
                    } else {
                        let pa = hpair(qa); let pb = hpair(qb); let pc = hpair(qc); let pd = hpair(qd);
                        va = select(pa.x, pa.y, oa) * ma; vb = select(pb.x, pb.y, ob) * mb;
                        vc = select(pc.x, pc.y, oc) * mc; vd = select(pd.x, pd.y, od) * md;
                    }
                    a = a + va * wv; b = b + vb * wv; c = c + vc * wv; e = e + vd * wv;
                    a2 = a2 + va * wv2; b2 = b2 + vb * wv2; c2 = c2 + vc * wv2; e2 = e2 + vd * wv2;
                    qa = qa + hd; qb = qb + hd; qc = qc + hd; qd = qd + hd;
                    wt = wt + ktap;
                }
                continue;
            }
            for (var ci: u32 = 0u; ci < params.c_in; ci = ci + 1u) {
                let wv = vec4<f32>(arena[wt], arena[wt + kstride],
                                   arena[wt + 2u * kstride], arena[wt + 3u * kstride]);
                let wt4 = wt + 4u * kstride;
                let wv2 = vec4<f32>(arena[wt4], arena[wt4 + kstride],
                                    arena[wt4 + 2u * kstride], arena[wt4 + 3u * kstride]);
                let va = ld(sa) * ma; let vb = ld(sb) * mb;
                let vc = ld(sc) * mc; let vd = ld(sd) * md;
                a = a + va * wv; b = b + vb * wv; c = c + vc * wv; e = e + vd * wv;
                a2 = a2 + va * wv2; b2 = b2 + vb * wv2; c2 = c2 + vc * wv2; e2 = e2 + vd * wv2;
                sa = sa + dhw; sb = sb + dhw; sc = sc + dhw; sd = sd + dhw;
                wt = wt + ktap;
            }
        }
      }
    }

    let out_dhw = params.d_out * params.h_out * params.w_out;
    let out_sp = params.out_off + (do_ * params.h_out + ho) * params.w_out + wo;
    let o0 = out_sp + nn * params.c_out * out_dhw + co0 * out_dhw;
    let nw = min(4u, params.w_out - wo);
    if (OUT_HALF) {
        // halves in pairs: `w_out` is even (`pack_half_activations`), so a
        // thread's positions are two pairs or one, each pair in one word
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let co = co0 + k;
            if (co >= params.c_out) { break; }
            var bias = 0.0;
            if (params.has_bias != 0u) { bias = arena[params.bias_off + co]; }
            let o = o0 + k * out_dhw;
            let j = k & 3u;
            var v: vec4<f32>;
            if (k < 4u) { v = vec4<f32>(a[j], b[j], c[j], e[j]); } else { v = vec4<f32>(a2[j], b2[j], c2[j], e2[j]); }
            put2(o, conv_act(v.x + bias), conv_act(v.y + bias));
            if (nw > 2u) { put2(o + 2u, conv_act(v.z + bias), conv_act(v.w + bias)); }
        }
        return;
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let co = co0 + k;
        if (co >= params.c_out) { break; }
        var bias = 0.0;
        if (params.has_bias != 0u) { bias = arena[params.bias_off + co]; }
        let o = o0 + k * out_dhw;
        let j = k & 3u;
        if (k < 4u) {
            put(o, conv_act(a[j] + bias));
            if (nw > 1u) { put(o + 1u, conv_act(b[j] + bias)); }
            if (nw > 2u) { put(o + 2u, conv_act(c[j] + bias)); }
            if (nw > 3u) { put(o + 3u, conv_act(e[j] + bias)); }
        } else {
            put(o, conv_act(a2[j] + bias));
            if (nw > 1u) { put(o + 1u, conv_act(b2[j] + bias)); }
            if (nw > 2u) { put(o + 2u, conv_act(c2[j] + bias)); }
            if (nw > 3u) { put(o + 3u, conv_act(e2[j] + bias)); }
        }
    }
}

// A 2-D 3×3 conv — kernel [1, 3, 3], stride 1, padding [0, 1, 1], no
// dilation, ungrouped: each slice of the depth on its own (FastSurfer's
// views run their batch as the depth). Eight output channels by four W
// positions a thread, as `conv3d_ctile_w4c8`, but
//
//   * the weights of eight input channels at a time sit in workgroup memory,
//     a vec4 of output channels per (channel, tap): every thread of a
//     workgroup holds the same eight output channels (workgroups take the
//     channel tiles innermost), so each weight is read from the arena once a
//     workgroup rather than once a thread;
//   * a row's six inputs are read once and feed all three of its taps.
//
// 2.5-8× `conv3d_ctile_w4c8` on FastSurfer's layers (71 → 71 channels over
// 16 × 256² is 134 → 29 ms on an M4 Pro in Chrome). Each output still sums
// its channels in order, but channel-major then tap rather than tap-major,
// so sums differ from the other kernels' in the last bits.
var<workgroup> k3_w: array<vec4<f32>, 144>; // [8 channels][9 taps][2 vec4]

// an input value, through the folded PReLU when there is one
// (`unfuse::CONV3D_K3`): the unary kernel's `select(α·x, x, x ≥ 0)`
fn k3_in(i: u32) -> f32 {
    let x = ld(i);
    if (params.in_prelu != 0u) { return select(bitcast<f32>(params.in_alpha) * x, x, x >= 0.0); }
    return x;
}

@compute @workgroup_size(64)
fn conv3d_k3_c8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let co_tiles = (params.c_out + 7u) / 8u;
    let w_quads = (params.w_out + 3u) / 4u;
    let wid = wg.x + wg.y * ngs.x;
    let cot = wid % co_tiles;
    let i = (wid / co_tiles) * 64u + lid;
    // no early return: every thread takes part in the barriers
    let valid = i < params.n * params.d_out * params.h_out * w_quads;
    let wo = (i % w_quads) * 4u;
    let q1 = i / w_quads;
    let ho = q1 % params.h_out;
    let q2 = q1 / params.h_out;
    let do_ = q2 % params.d_out;
    let nn = q2 / params.d_out;
    let co0 = cot * 8u;
    let dhw = params.d * params.h * params.w;
    let kstride = params.c_in * 9u;
    let in_base = params.in_off + nn * params.c_in * dhw + do_ * params.h * params.w;
    let wmax = i32(params.w) - 1;
    // the columns wo - 1 .. wo + 4: clamped into the row, zero outside it
    let x = i32(wo);
    let m0 = select(0.0, 1.0, x >= 1);
    let m4 = select(0.0, 1.0, x + 3 <= wmax);
    let m5 = select(0.0, 1.0, x + 4 <= wmax);
    let m1 = select(0.0, 1.0, x <= wmax);
    let m2 = select(0.0, 1.0, x + 1 <= wmax);
    let m3 = select(0.0, 1.0, x + 2 <= wmax);
    let col0 = u32(clamp(x - 1, 0, wmax)); let col1 = u32(clamp(x, 0, wmax));
    let col2 = u32(clamp(x + 1, 0, wmax)); let col3 = u32(clamp(x + 2, 0, wmax));
    let col4 = u32(clamp(x + 3, 0, wmax)); let col5 = u32(clamp(x + 4, 0, wmax));

    // `pN_g`: position wo + N, output channels co0 + 4g .. + 3
    var p0_0 = vec4<f32>(0.0); var p0_1 = vec4<f32>(0.0);
    var p1_0 = vec4<f32>(0.0); var p1_1 = vec4<f32>(0.0);
    var p2_0 = vec4<f32>(0.0); var p2_1 = vec4<f32>(0.0);
    var p3_0 = vec4<f32>(0.0); var p3_1 = vec4<f32>(0.0);

    for (var c0: u32 = 0u; c0 < params.c_in; c0 = c0 + 8u) {
        workgroupBarrier();
        for (var j: u32 = lid; j < 144u; j = j + 64u) {
            let co = co0 + (j & 1u) * 4u;
            let ci = c0 + j / 18u;
            var v = vec4<f32>(0.0);
            if (ci < params.c_in) {
                let base = params.w_off + co * kstride + ci * 9u + (j >> 1u) % 9u;
                v = vec4<f32>(
                    select(0.0, arena[base], co < params.c_out),
                    select(0.0, arena[base + kstride], co + 1u < params.c_out),
                    select(0.0, arena[base + 2u * kstride], co + 2u < params.c_out),
                    select(0.0, arena[base + 3u * kstride], co + 3u < params.c_out));
            }
            k3_w[j] = v;
        }
        workgroupBarrier();
        // an `if`, not a `continue`: the next barrier stays in uniform
        // control flow (Tint rejects the pipeline otherwise; naga does not
        // check)
        if (valid) {
            let cn = min(8u, params.c_in - c0);
            for (var kh: u32 = 0u; kh < 3u; kh = kh + 1u) {
                let hs = i32(ho + kh) - 1;
                if (hs < 0 || hs >= i32(params.h)) { continue; }
                var row = in_base + c0 * dhw + u32(hs) * params.w;
                for (var ci: u32 = 0u; ci < cn; ci = ci + 1u) {
                    let v0 = k3_in(row + col0) * m0; let v1 = k3_in(row + col1) * m1;
                    let v2 = k3_in(row + col2) * m2; let v3 = k3_in(row + col3) * m3;
                    let v4 = k3_in(row + col4) * m4; let v5 = k3_in(row + col5) * m5;
                    let wb = (ci * 9u + kh * 3u) * 2u;
                    var w0 = k3_w[wb]; var w1 = k3_w[wb + 1u];
                    p0_0 = p0_0 + v0 * w0; p0_1 = p0_1 + v0 * w1;
                    p1_0 = p1_0 + v1 * w0; p1_1 = p1_1 + v1 * w1;
                    p2_0 = p2_0 + v2 * w0; p2_1 = p2_1 + v2 * w1;
                    p3_0 = p3_0 + v3 * w0; p3_1 = p3_1 + v3 * w1;
                    w0 = k3_w[wb + 2u]; w1 = k3_w[wb + 3u];
                    p0_0 = p0_0 + v1 * w0; p0_1 = p0_1 + v1 * w1;
                    p1_0 = p1_0 + v2 * w0; p1_1 = p1_1 + v2 * w1;
                    p2_0 = p2_0 + v3 * w0; p2_1 = p2_1 + v3 * w1;
                    p3_0 = p3_0 + v4 * w0; p3_1 = p3_1 + v4 * w1;
                    w0 = k3_w[wb + 4u]; w1 = k3_w[wb + 5u];
                    p0_0 = p0_0 + v2 * w0; p0_1 = p0_1 + v2 * w1;
                    p1_0 = p1_0 + v3 * w0; p1_1 = p1_1 + v3 * w1;
                    p2_0 = p2_0 + v4 * w0; p2_1 = p2_1 + v4 * w1;
                    p3_0 = p3_0 + v5 * w0; p3_1 = p3_1 + v5 * w1;
                    row = row + dhw;
                }
            }
        }
    }
    if (!valid) { return; }

    let out_dhw = params.d_out * params.h_out * params.w_out;
    let o0 = params.out_off + nn * params.c_out * out_dhw + co0 * out_dhw
        + (do_ * params.h_out + ho) * params.w_out + wo;
    let nw = min(4u, params.w_out - wo);
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let co = co0 + k;
        if (co >= params.c_out) { break; }
        var bias = 0.0;
        if (params.has_bias != 0u) { bias = arena[params.bias_off + co]; }
        let o = o0 + k * out_dhw;
        let j = k & 3u;
        var v: vec4<f32>;
        if (k < 4u) { v = vec4<f32>(p0_0[j], p1_0[j], p2_0[j], p3_0[j]); } else { v = vec4<f32>(p0_1[j], p1_1[j], p2_1[j], p3_1[j]); }
        v = v + bias;
        if (params.res_max != 0u) {
            // `unfuse::CONV3D_K3`'s folded `max`: the binary kernel's
            let r = params.res_off + (o - params.out_off);
            v = max(v, vec4<f32>(arena[r], arena[min(r + 1u, r + nw - 1u)],
                                 arena[min(r + 2u, r + nw - 1u)], arena[min(r + 3u, r + nw - 1u)]));
        }
        if (OUT_HALF) {
            // pairs, `w_out` even (`pack_half_activations`)
            put2(o, conv_act(v.x), conv_act(v.y));
            if (nw > 2u) { put2(o + 2u, conv_act(v.z), conv_act(v.w)); }
            continue;
        }
        put(o, conv_act(v.x));
        if (nw > 1u) { put(o + 1u, conv_act(v.y)); }
        if (nw > 2u) { put(o + 2u, conv_act(v.z)); }
        if (nw > 3u) { put(o + 3u, conv_act(v.w)); }
    }
}

// A 1×1×1, stride-1, unpadded, ungrouped conv and the argmax over its
// output channels (`params.act == 0xFFFE`, `unfuse::fold_conv_argmax`): one
// thread per voxel, the index of its largest channel (the first of equals,
// as the argmax kernel picks) written as f32 to an `[n, d, h, w]` output.
// The classes' logits — a full-size tensor of c_out channels, 104 for a
// brainchop aparc model — are never stored. Each channel is summed as
// `conv3d_ctile_w4` sums it (four channels in a vec4, the input value times
// the mask, then the bias), so the labels are the same.
@compute @workgroup_size(64)
fn conv3d_argmax(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let spatial = params.d_out * params.h_out * params.w_out;
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n * spatial) { return; }
    let nn = i / spatial;
    let sp = i % spatial;
    let dhw = params.d * params.h * params.w;
    let x0 = params.in_off + nn * params.c_in * dhw + sp;
    let kstride = params.c_in;
    var best: f32 = 0.0;
    var best_i: u32 = 0u;
    for (var co0: u32 = 0u; co0 < params.c_out; co0 = co0 + 4u) {
        var a = vec4<f32>(0.0);
        var src = x0;
        // a tile past c_out reads weights inside the arena (or clamped) into
        // lanes that are never compared
        var wt = params.w_off + co0 * kstride;
        for (var ci: u32 = 0u; ci < params.c_in; ci = ci + 1u) {
            let wv = vec4<f32>(arena[wt], arena[wt + kstride],
                               arena[wt + 2u * kstride], arena[wt + 3u * kstride]);
            a = a + (ld(src) * 1.0) * wv;
            src = src + dhw;
            wt = wt + 1u;
        }
        if (params.has_bias != 0u) {
            let b = params.bias_off + co0;
            a = a + vec4<f32>(arena[b], arena[b + 1u], arena[b + 2u], arena[b + 3u]);
        }
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let co = co0 + k;
            if (co < params.c_out && (co == 0u || a[k] > best)) {
                best = a[k];
                best_i = co;
            }
        }
    }
    put(params.out_off + nn * spatial + sp, f32(best_i));
}

// A 1×1×1 conv keeping only each voxel's `k` largest output channels
// (`params.act == 0xFFFC`, `k` the f32 in `params.alpha`; `unfuse::CONV3D_TOPK`):
// one thread a voxel, the channels summed four at a time as `conv3d_argmax`
// sums them, inserted into the best `k + 1` so far (strictly greater moves
// up: the earlier channel first among equals). Writes `[n, 2k + 1, spatial]`:
// the `k` values best first, the `k + 1`-th, then the `k` channels as f32.
const TOPK_SLOTS: u32 = 9u;

@compute @workgroup_size(64)
fn conv3d_topk(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let spatial = params.d_out * params.h_out * params.w_out;
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n * spatial) { return; }
    let nn = i / spatial;
    let sp = i % spatial;
    let k = min(u32(bitcast<f32>(params.alpha)), TOPK_SLOTS - 1u);
    let dhw = params.d * params.h * params.w;
    let x0 = params.in_off + nn * params.c_in * dhw + sp;
    let kstride = params.c_in;
    var tv: array<f32, 9>;
    var tc: array<u32, 9>;
    for (var j: u32 = 0u; j < TOPK_SLOTS; j = j + 1u) {
        tv[j] = -3.402823466e+38;
        tc[j] = 0u;
    }
    for (var co0: u32 = 0u; co0 < params.c_out; co0 = co0 + 4u) {
        var a = vec4<f32>(0.0);
        var src = x0;
        var wt = params.w_off + co0 * kstride;
        for (var ci: u32 = 0u; ci < params.c_in; ci = ci + 1u) {
            let wv = vec4<f32>(arena[wt], arena[wt + kstride],
                               arena[wt + 2u * kstride], arena[wt + 3u * kstride]);
            a = a + (ld(src) * 1.0) * wv;
            src = src + dhw;
            wt = wt + 1u;
        }
        if (params.has_bias != 0u) {
            let b = params.bias_off + co0;
            a = a + vec4<f32>(arena[b], arena[b + 1u], arena[b + 2u], arena[b + 3u]);
        }
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let co = co0 + q;
            let v = a[q];
            if (co >= params.c_out || !(v > tv[k])) { continue; }
            var j = k;
            loop {
                if (j == 0u || !(v > tv[j - 1u])) { break; }
                tv[j] = tv[j - 1u];
                tc[j] = tc[j - 1u];
                j = j - 1u;
            }
            tv[j] = v;
            tc[j] = co;
        }
    }
    let ob = params.out_off + nn * (2u * k + 1u) * spatial + sp;
    for (var j: u32 = 0u; j <= k; j = j + 1u) {
        put(ob + j * spatial, tv[j]);
    }
    for (var j: u32 = 0u; j < k; j = j + 1u) {
        put(ob + (k + 1u + j) * spatial, f32(tc[j]));
    }
}
