// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// RLX — fused GGUF K-quant GEMV (decode m=1) for `Op::DequantMatMul`.
//
// Computes one row of `y = x @ W^T` where `W` is a GGUF-packed weight `[n, k]`
// (row-major blocks) and `x` is `[1, k]`. One thread per output column `j`:
// it walks the `k/256` blocks of weight row `j`, dequantizes each block on the
// fly (NO f32 scratch), and accumulates `x[k] * w[j,k]` into `y[j]`.
//
// Binding scheme works around two wgpu limits that the whole-arena
// `as_entire_binding` path hits on multi-GiB models:
//   1. a storage binding may be at most `max_storage_buffer_binding_size`
//      (4 GiB) — so `x` and `weight` are bound as separate *windowed* slabs of
//      the arena, each well under 4 GiB;
//   2. a buffer bound STORAGE_READ_WRITE is exclusive in a dispatch (cannot
//      also be bound read-only) — so the arena is bound read-only twice (x +
//      weight) and `y` is written to a small *separate* output buffer that the
//      caller copies back into the arena.
//
// Dequant math mirrors `dequant_gguf.wgsl` (scheme 0 = Q4_K, scheme 2 = Q6_K)
// exactly; only the sink differs (accumulate vs store). Block element order is
// identity (`out_i` == k-offset within the block), so `x[k_base + out_i]`.

struct Params {
    k: u32,            // contraction dim (multiple of 256)
    n: u32,            // output columns (weight rows)
    scheme_id: u32,    // 0 = Q4_K, 2 = Q6_K
    x_f32_off: u32,    // f32 index of x[0] within the x binding
    w_byte_off: u32,   // byte offset of weight[0,0] within the weight binding
    out_f32_off: u32,  // f32 index of y[0] within the output binding
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read>        xarr: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;
@group(0) @binding(2) var<storage, read>        warr: array<u32>;
@group(0) @binding(3) var<storage, read_write>  outarr: array<f32>;

fn read_w(rel: u32) -> u32 {
    let abs = params.w_byte_off + rel;
    let word = abs / 4u;
    let shift = (abs % 4u) * 8u;
    return (warr[word] >> shift) & 0xffu;
}

fn read_w_i8(rel: u32) -> i32 {
    let b = read_w(rel);
    return select(i32(b), i32(b) - 256, b >= 128u);
}

fn dq_read_f16(rel: u32) -> f32 {
    let bits = read_w(rel) | (read_w(rel + 1u) << 8u);
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;
    var v: f32;
    if (exp == 0u) { v = f32(mant) / 1024.0 * exp2(-14.0); }
    else if (exp == 31u) { v = select(0.0, bitcast<f32>(0x7f800000u), mant == 0u); }
    else { v = (1.0 + f32(mant) / 1024.0) * exp2(f32(i32(exp) - 15)); }
    return select(v, -v, sign != 0u);
}

fn dq_get_scale_min_k4(scales_rel: u32, j: u32) -> vec2<u32> {
    if (j < 4u) {
        return vec2<u32>(read_w(scales_rel + j) & 63u, read_w(scales_rel + j + 4u) & 63u);
    }
    let sc = (read_w(scales_rel + j + 4u) & 0x0Fu) | (((read_w(scales_rel + j - 4u) >> 6u) & 3u) << 4u);
    let mn = (read_w(scales_rel + j + 4u) >> 4u) | (((read_w(scales_rel + j) >> 6u) & 3u) << 4u);
    return vec2<u32>(sc, mn);
}

@compute @workgroup_size(64)
fn dequant_gemv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= params.n) { return; }
    let nbk = params.k / 256u;       // 256-elem blocks per weight row
    let xb = params.x_f32_off;
    var acc = 0.0;

    if (params.scheme_id == 0u) {
        // Q4_K: 144 bytes / 256 elems.
        let blk = 2u + 2u + 12u + 256u / 2u;
        for (var b: u32 = 0u; b < nbk; b = b + 1u) {
            let off = (j * nbk + b) * blk;
            let d = dq_read_f16(off);
            let dmin = dq_read_f16(off + 2u);
            let scales_rel = off + 4u;
            let qs_rel = off + 16u;
            let ko = b * 256u;
            var is = 0u;
            var oi = 0u;
            for (var t: u32 = 0u; t < 8u; t = t + 2u) {
                let sm0 = dq_get_scale_min_k4(scales_rel, t);
                let sm1 = dq_get_scale_min_k4(scales_rel, t + 1u);
                let d0 = d * f32(sm0.x); let m0 = dmin * f32(sm0.y);
                let d1 = d * f32(sm1.x); let m1 = dmin * f32(sm1.y);
                for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                    let val = d0 * f32(read_w(qs_rel + is + l) & 0x0Fu) - m0;
                    acc = acc + xarr[xb + ko + oi] * val;
                    oi = oi + 1u;
                }
                for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                    let val = d1 * f32(read_w(qs_rel + is + l) >> 4u) - m1;
                    acc = acc + xarr[xb + ko + oi] * val;
                    oi = oi + 1u;
                }
                is = is + 32u;
            }
        }
    } else if (params.scheme_id == 24u) {
        // Q1_0 (prism-ml Bonsai-27B): 18 bytes / 128 elems. f16 scale `d` +
        // 16 sign bytes, LSB-first, bit1 -> +d, bit0 -> -d. Reads packed weight
        // directly (no f32 scratch). k is a multiple of 128.
        let nbk1 = params.k / 128u;
        let blk1 = 18u;
        for (var b: u32 = 0u; b < nbk1; b = b + 1u) {
            let off = (j * nbk1 + b) * blk1;
            let d = dq_read_f16(off);
            let nd = -d;
            let ko = b * 128u;
            for (var byte: u32 = 0u; byte < 16u; byte = byte + 1u) {
                let bits = read_w(off + 2u + byte);
                for (var bit: u32 = 0u; bit < 8u; bit = bit + 1u) {
                    let w = select(nd, d, ((bits >> bit) & 1u) != 0u);
                    acc = acc + xarr[xb + ko + byte * 8u + bit] * w;
                }
            }
        }
    } else if (params.scheme_id == 2u) {
        // Q6_K: 210 bytes / 256 elems.
        let ql_len = 256u / 2u;
        let qh_len = 256u / 4u;
        let sc_len = 256u / 16u;
        let blk = ql_len + qh_len + sc_len + 2u;
        for (var b: u32 = 0u; b < nbk; b = b + 1u) {
            let off = (j * nbk + b) * blk;
            let ql_rel = off;
            let qh_rel = off + ql_len;
            let sc_rel = off + ql_len + qh_len;
            let d = dq_read_f16(off + ql_len + qh_len + sc_len);
            let ko = b * 256u;
            for (var h: u32 = 0u; h < 2u; h = h + 1u) {
                let row_base = h * 128u;
                let ql_off = h * 64u;
                let qh_off_h = h * 32u;
                let sc_off = h * 8u;
                for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                    let is = l / 16u;
                    let qh_b = read_w(qh_rel + qh_off_h + l);
                    let q1 = f32(i32(((read_w(ql_rel + ql_off + l) & 0x0Fu) | (((qh_b >> 0u) & 3u) << 4u)) - 32));
                    let q2 = f32(i32(((read_w(ql_rel + ql_off + l + 32u) & 0x0Fu) | (((qh_b >> 2u) & 3u) << 4u)) - 32));
                    let q3 = f32(i32(((read_w(ql_rel + ql_off + l) >> 4u) | (((qh_b >> 4u) & 3u) << 4u)) - 32));
                    let q4 = f32(i32(((read_w(ql_rel + ql_off + l + 32u) >> 4u) | (((qh_b >> 6u) & 3u) << 4u)) - 32));
                    let v1 = d * f32(read_w_i8(sc_rel + sc_off + is)) * q1;
                    let v2 = d * f32(read_w_i8(sc_rel + sc_off + is + 2u)) * q2;
                    let v3 = d * f32(read_w_i8(sc_rel + sc_off + is + 4u)) * q3;
                    let v4 = d * f32(read_w_i8(sc_rel + sc_off + is + 6u)) * q4;
                    acc = acc + xarr[xb + ko + row_base + l] * v1;
                    acc = acc + xarr[xb + ko + row_base + l + 32u] * v2;
                    acc = acc + xarr[xb + ko + row_base + l + 64u] * v3;
                    acc = acc + xarr[xb + ko + row_base + l + 96u] * v4;
                }
            }
        }
    }

    outarr[params.out_f32_off + j] = acc;
}
