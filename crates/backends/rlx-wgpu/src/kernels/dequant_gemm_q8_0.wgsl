// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// RLX — tiled Q8_0 GEMM (m > 1) for `Op::DequantMatMul`: `Y[m,n] = X[m,k] · Wᵀ`
// with `W` GGUF Q8_0-packed `[n, k]` (34-byte blocks: f16 scale + 32 int8).
//
// 256 threads per 64×64 output tile, 4×4 outputs each (rows ty + 16i,
// columns tx + 16j). K advances 16 at a time (half a Q8_0 block): the X tile
// [64 rows × 16] and the dequantized W tile [64 columns × 16] are staged in
// workgroup memory (8 KB), so each weight is dequantized once per tile, not
// once per row as the per-row GEMV does. Bindings match `dequant_gemm_q1_0.wgsl`.

struct Params {
    m: u32,
    k: u32,
    n: u32,
    x_f32_off: u32,    // f32 index of X[0,0] within the x binding
    w_byte_off: u32,   // byte offset of W[0,0] within the weight binding
    out_f32_off: u32,  // f32 index of Y[0,0] within the output binding
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read>        xarr: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;
@group(0) @binding(2) var<storage, read>        warr: array<u32>;
@group(0) @binding(3) var<storage, read_write>  outarr: array<f32>;

const T: u32 = 64u;
const KS: u32 = 16u;
const BLK_BYTES: u32 = 34u;

var<workgroup> xs: array<f32, 1024>;  // [64 rows][16]
// W rows padded to 17: the inner loop's threads (one column each) would
// otherwise all start in the same bank.
const WS: u32 = 17u;
var<workgroup> ws: array<f32, 1088>;  // [64 cols][17]

fn read_w(rel: u32) -> u32 {
    let abs = params.w_byte_off + rel;
    return (warr[abs / 4u] >> ((abs % 4u) * 8u)) & 0xffu;
}

fn read_i8(rel: u32) -> f32 {
    let b = read_w(rel);
    return f32(i32(b) - select(0, 256, b >= 128u));
}

fn dq_read_f16(rel: u32) -> f32 {
    let bits = read_w(rel) | (read_w(rel + 1u) << 8u);
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;
    var v: f32;
    if (exp == 0u) { v = f32(mant) / 1024.0 * exp2(-14.0); }
    else if (exp == 31u) { v = bitcast<f32>(select(0u, 0x7f800000u, mant == 0u)); }
    else { v = (1.0 + f32(mant) / 1024.0) * exp2(f32(i32(exp) - 15)); }
    return select(v, -v, sign != 0u);
}

@compute @workgroup_size(16, 16)
fn dequant_gemm_q8_0(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let col0 = wg.x * T;
    let row0 = wg.y * T;
    let nblocks = params.k / 32u;

    // Staging roles: 4 threads per X row / W column, 4 k each.
    let lane = tid / 4u;
    let kq = (tid % 4u) * 4u;
    let row_ok = row0 + lane < params.m;
    let col_ok = col0 + lane < params.n;
    let xrow = params.x_f32_off + min(row0 + lane, params.m - 1u) * params.k;
    let wcol = min(col0 + lane, params.n - 1u) * nblocks * BLK_BYTES;

    // Four rows × four columns per thread as vec4s: indexed function-scope
    // arrays here were lowered to stack memory (0.04 TFLOPS).
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let steps = params.k / KS;
    for (var s: u32 = 0u; s < steps; s = s + 1u) {
        let k0 = s * KS;
        workgroupBarrier();
        let xo = xrow + k0 + kq;
        xs[lane * KS + kq]      = select(0.0, xarr[xo], row_ok);
        xs[lane * KS + kq + 1u] = select(0.0, xarr[xo + 1u], row_ok);
        xs[lane * KS + kq + 2u] = select(0.0, xarr[xo + 2u], row_ok);
        xs[lane * KS + kq + 3u] = select(0.0, xarr[xo + 3u], row_ok);
        let blk = wcol + (k0 / 32u) * BLK_BYTES;
        let d = select(0.0, dq_read_f16(blk), col_ok);
        let q0 = blk + 2u + (k0 % 32u) + kq;
        ws[lane * WS + kq]      = d * read_i8(q0);
        ws[lane * WS + kq + 1u] = d * read_i8(q0 + 1u);
        ws[lane * WS + kq + 2u] = d * read_i8(q0 + 2u);
        ws[lane * WS + kq + 3u] = d * read_i8(q0 + 3u);
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < KS; kk = kk + 1u) {
            let b = vec4<f32>(
                ws[tx * WS + kk],
                ws[(tx + 16u) * WS + kk],
                ws[(tx + 32u) * WS + kk],
                ws[(tx + 48u) * WS + kk],
            );
            acc0 = acc0 + xs[ty * KS + kk] * b;
            acc1 = acc1 + xs[(ty + 16u) * KS + kk] * b;
            acc2 = acc2 + xs[(ty + 32u) * KS + kk] * b;
            acc3 = acc3 + xs[(ty + 48u) * KS + kk] * b;
        }
    }

    let ob = params.out_f32_off;
    store_row(ob, row0 + ty, col0 + tx, acc0);
    store_row(ob, row0 + ty + 16u, col0 + tx, acc1);
    store_row(ob, row0 + ty + 32u, col0 + tx, acc2);
    store_row(ob, row0 + ty + 48u, col0 + tx, acc3);
}

// Row `r`, columns `c, c + 16, c + 32, c + 48`.
fn store_row(ob: u32, r: u32, c: u32, v: vec4<f32>) {
    if (r >= params.m) { return; }
    let base = ob + r * params.n;
    if (c < params.n) { outarr[base + c] = v.x; }
    if (c + 16u < params.n) { outarr[base + c + 16u] = v.y; }
    if (c + 32u < params.n) { outarr[base + c + 32u] = v.z; }
    if (c + 48u < params.n) { outarr[base + c + 48u] = v.w; }
}
