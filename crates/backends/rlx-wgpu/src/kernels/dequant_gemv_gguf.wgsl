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
// Dequant math mirrors `dequant_gguf.wgsl` (scheme 0 = Q4_K, 1 = Q5_K, 2 = Q6_K,
// 20 = Q8_0, 24 = Q1_0)
// exactly; only the sink differs (accumulate vs store). Block element order is
// identity (`out_i` == k-offset within the block), so `x[k_base + out_i]`.

struct Params {
    k: u32,            // contraction dim (multiple of 256)
    n: u32,            // output columns (weight rows)
    scheme_id: u32,    // 0 = Q4_K, 1 = Q5_K, 2 = Q6_K, 20 = Q8_0, 24 = Q1_0
    x_f32_off: u32,    // f32 index of x[0] within the x binding
    w_byte_off: u32,   // byte offset of weight[0,0] within the weight binding
    out_f32_off: u32,  // f32 index of y[0] within the output binding
    out_stride: u32,   // f32 stride between output rows (rows = dispatch y)
    // Activation rows in this dispatch, for the multi-row kernels
    // (`dequant_gemv_q4k16` / `_q6k16`: 8 rows per workgroup row); the
    // single-row kernels take one row per dispatch row and ignore it.
    m_rows: u32,
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
    else if (exp == 31u) { v = bitcast<f32>(select(0u, 0x7f800000u, mant == 0u)); }
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
    // One activation row per dispatch row: a prefill's m rows are one dispatch,
    // not m separately encoded passes.
    let row = gid.y;
    let nbk = params.k / 256u;       // 256-elem blocks per weight row
    let xb = params.x_f32_off + row * params.k;
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
    } else if (params.scheme_id == 1u) {
        // Q5_K: 176 bytes / 256 elems — Q4_K plus a 32-byte high-bit plane.
        let blk = 2u + 2u + 12u + 256u / 8u + 256u / 2u;
        for (var b: u32 = 0u; b < nbk; b = b + 1u) {
            let off = (j * nbk + b) * blk;
            let d = dq_read_f16(off);
            let dmin = dq_read_f16(off + 2u);
            let scales_rel = off + 4u;
            let qh_rel = off + 16u;
            let qs_rel = qh_rel + 32u;
            let ko = b * 256u;
            var is = 0u;
            var oi = 0u;
            var u1 = 1u;
            var u2 = 2u;
            for (var t: u32 = 0u; t < 8u; t = t + 2u) {
                let sm0 = dq_get_scale_min_k4(scales_rel, t);
                let sm1 = dq_get_scale_min_k4(scales_rel, t + 1u);
                let d0 = d * f32(sm0.x); let m0 = dmin * f32(sm0.y);
                let d1 = d * f32(sm1.x); let m1 = dmin * f32(sm1.y);
                for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                    let lo = read_w(qs_rel + is + l) & 0x0Fu;
                    let hi = select(0u, 16u, (read_w(qh_rel + l) & u1) != 0u);
                    acc = acc + xarr[xb + ko + oi] * (d0 * f32(lo + hi) - m0);
                    oi = oi + 1u;
                }
                for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                    let lo = read_w(qs_rel + is + l) >> 4u;
                    let hi = select(0u, 16u, (read_w(qh_rel + l) & u2) != 0u);
                    acc = acc + xarr[xb + ko + oi] * (d1 * f32(lo + hi) - m1);
                    oi = oi + 1u;
                }
                is = is + 32u;
                u1 = u1 << 2u;
                u2 = u2 << 2u;
            }
        }
    } else if (params.scheme_id == 20u) {
        // Q8_0: 34 bytes / 32 elems — f16 scale + 32 i8. k is a multiple of 32.
        let nb8 = params.k / 32u;
        for (var b: u32 = 0u; b < nb8; b = b + 1u) {
            let off = (j * nb8 + b) * 34u;
            let d = dq_read_f16(off);
            let ko = b * 32u;
            var s8 = 0.0;
            for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                s8 = s8 + xarr[xb + ko + l] * f32(read_w_i8(off + 2u + l));
            }
            acc = acc + d * s8;
        }
    } else if (params.scheme_id == 256u) {
        // bf16 kept packed (`QuantScheme::GgufBF16`, wgpu-local id 256): two
        // bytes per element, the upper half of an f32; row j is k elements.
        // A word-aligned row (k even, aligned base) reads two values per u32.
        let row = j * params.k * 2u;
        if ((params.w_byte_off + row) % 4u == 0u && params.k % 2u == 0u) {
            let w0 = (params.w_byte_off + row) / 4u;
            for (var p: u32 = 0u; p < params.k / 2u; p = p + 1u) {
                let word = warr[w0 + p];
                acc = acc + xarr[xb + 2u * p] * bitcast<f32>(word << 16u);
                acc = acc + xarr[xb + 2u * p + 1u] * bitcast<f32>(word & 0xffff0000u);
            }
        } else {
            for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
                let rel = row + kk * 2u;
                let bits = read_w(rel) | (read_w(rel + 1u) << 8u);
                acc = acc + xarr[xb + kk] * bitcast<f32>(bits << 16u);
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

    outarr[params.out_f32_off + row * params.out_stride + j] = acc;
}

// Four weight bytes starting at `rel`, little-endian, at any byte alignment.
fn read_w4(rel: u32) -> u32 {
    let abs = params.w_byte_off + rel;
    let word = abs / 4u;
    let shift = (abs % 4u) * 8u;
    if (shift == 0u) { return warr[word]; }
    return (warr[word] >> shift) | (warr[word + 1u] << (32u - shift));
}

// ── Row-group variant: 8 output columns per 256-thread workgroup ───────────
//
// `dequant_gemv` gives each output column to ONE thread, which walks the whole
// weight row serially; for decode-sized matrices (n ≈ 1024–4096) that is
// 16–64 workgroups for the whole GPU. Here 32 lanes share a column (`lane`)
// and a workgroup holds 8 columns (`slot`), so one set of reduction barriers
// serves 8 columns. Q4_K, Q6_K, Q8_0 and bf16; the rest keep `dequant_gemv`.
// The column group spans workgroup x and z (`n` can exceed the 65535 limit of
// one grid dimension — a 248320-row LM head does). Each lane reads its share of a
// block as whole 32-bit words where the layout allows (Q4_K: one word = 8
// weights) and decodes f16 scales with `unpack2x16float`.

var<workgroup> red8: array<f32, 256>;

// The f16 at byte `rel` (any alignment), as f32.
fn f16_at(rel: u32) -> f32 {
    return unpack2x16float(read_w4(rel) & 0xffffu).x;
}

// Byte `i` (0..11) of a Q4_K block's packed scales held as three words.
fn sc_byte(sc: vec3<u32>, i: u32) -> u32 {
    return (sc[i / 4u] >> ((i % 4u) * 8u)) & 0xffu;
}

// Q4_K (scale, min) of sub-blocks 2g and 2g+1 → (sc0, m0, sc1, m1): the same
// 6-bit unpacking as `dq_get_scale_min_k4`, from registers.
fn q4k_scale_mins(sc: vec3<u32>, g: u32) -> vec4<f32> {
    let j0 = 2u * g;
    let j1 = j0 + 1u;
    if (g < 2u) {
        return vec4<f32>(
            f32(sc_byte(sc, j0) & 63u), f32(sc_byte(sc, j0 + 4u) & 63u),
            f32(sc_byte(sc, j1) & 63u), f32(sc_byte(sc, j1 + 4u) & 63u),
        );
    }
    let s0 = (sc_byte(sc, j0 + 4u) & 0x0Fu) | ((sc_byte(sc, j0 - 4u) >> 6u) << 4u);
    let n0 = (sc_byte(sc, j0 + 4u) >> 4u) | ((sc_byte(sc, j0) >> 6u) << 4u);
    let s1 = (sc_byte(sc, j1 + 4u) & 0x0Fu) | ((sc_byte(sc, j1 - 4u) >> 6u) << 4u);
    let n1 = (sc_byte(sc, j1 + 4u) >> 4u) | ((sc_byte(sc, j1) >> 6u) << 4u);
    return vec4<f32>(f32(s0), f32(n0), f32(s1), f32(n1));
}

@compute @workgroup_size(256)
fn dequant_gemv_rows8(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    let lane = t % 32u;
    let slot = t / 32u;
    let j = (wid.z * nwg.x + wid.x) * 8u + slot;
    // Not uniform across the workgroup (8 columns share it): no early
    // return before the barriers, just no work.
    let col_on = j < params.n;
    let row = wid.y;
    let xb = params.x_f32_off + row * params.k;
    var acc = 0.0;

    if (col_on) {
        if (params.scheme_id == 0u) {
            // Q4_K, 144 bytes / 256: lane reads qs word `lane` (bytes 4·lane..),
            // i.e. group g = lane / 8 at byte offset l = 4·(lane % 8): four
            // low nibbles (sub-block 2g) and four high (2g + 1).
            let nbk = params.k / 256u;
            let g = lane / 8u;
            let l = (lane % 8u) * 4u;
            for (var b: u32 = 0u; b < nbk; b = b + 1u) {
                let off = (j * nbk + b) * 144u;
                let dm = unpack2x16float(read_w4(off));
                // The 12 scale bytes as three words; sub-blocks 2g and 2g+1.
                let sc = vec3<u32>(read_w4(off + 4u), read_w4(off + 8u), read_w4(off + 12u));
                let sm = q4k_scale_mins(sc, g);
                let d0 = dm.x * sm.x; let m0 = dm.y * sm.y;
                let d1 = dm.x * sm.z; let m1 = dm.y * sm.w;
                let q = read_w4(off + 16u + g * 32u + l);
                let xo = xb + b * 256u + g * 64u + l;
                for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                    let byte = (q >> (i * 8u)) & 0xffu;
                    acc = acc + xarr[xo + i] * (d0 * f32(byte & 0x0Fu) - m0);
                    acc = acc + xarr[xo + 32u + i] * (d1 * f32(byte >> 4u) - m1);
                }
            }
        } else if (params.scheme_id == 2u) {
            // Q6_K, 210 bytes / 256: lane is `l` of the serial kernel's inner
            // loop, for both halves h — eight elements.
            let nbk = params.k / 256u;
            let is = lane / 16u;
            for (var b: u32 = 0u; b < nbk; b = b + 1u) {
                let off = (j * nbk + b) * 210u;
                let d = f16_at(off + 208u);
                for (var h: u32 = 0u; h < 2u; h = h + 1u) {
                    let ql_rel = off + h * 64u;
                    let qh_b = read_w(off + 128u + h * 32u + lane);
                    let sc_rel = off + 192u + h * 8u;
                    let a = read_w(ql_rel + lane);
                    let c = read_w(ql_rel + lane + 32u);
                    let q1 = f32(i32((a & 0x0Fu) | ((qh_b & 3u) << 4u)) - 32);
                    let q2 = f32(i32((c & 0x0Fu) | (((qh_b >> 2u) & 3u) << 4u)) - 32);
                    let q3 = f32(i32((a >> 4u) | (((qh_b >> 4u) & 3u) << 4u)) - 32);
                    let q4 = f32(i32((c >> 4u) | (((qh_b >> 6u) & 3u) << 4u)) - 32);
                    let xo = xb + b * 256u + h * 128u + lane;
                    acc = acc + xarr[xo] * (d * f32(read_w_i8(sc_rel + is)) * q1);
                    acc = acc + xarr[xo + 32u] * (d * f32(read_w_i8(sc_rel + is + 2u)) * q2);
                    acc = acc + xarr[xo + 64u] * (d * f32(read_w_i8(sc_rel + is + 4u)) * q3);
                    acc = acc + xarr[xo + 96u] * (d * f32(read_w_i8(sc_rel + is + 6u)) * q4);
                }
            }
        } else if (params.scheme_id == 20u) {
            // Q8_0, 34 bytes / 32: lane takes element `lane` of every block.
            let nb8 = params.k / 32u;
            for (var b: u32 = 0u; b < nb8; b = b + 1u) {
                let off = (j * nb8 + b) * 34u;
                acc = acc + xarr[xb + b * 32u + lane] * (f16_at(off) * f32(read_w_i8(off + 2u + lane)));
            }
        } else if (params.scheme_id == 256u) {
            // bf16: row j is k elements of 2 bytes.
            let wrow = j * params.k * 2u;
            if ((params.w_byte_off + wrow) % 4u == 0u && params.k % 2u == 0u) {
                let w0 = (params.w_byte_off + wrow) / 4u;
                for (var p: u32 = lane; p < params.k / 2u; p = p + 32u) {
                    let word = warr[w0 + p];
                    acc = acc + xarr[xb + 2u * p] * bitcast<f32>(word << 16u);
                    acc = acc + xarr[xb + 2u * p + 1u] * bitcast<f32>(word & 0xffff0000u);
                }
            } else {
                for (var kk: u32 = lane; kk < params.k; kk = kk + 32u) {
                    let rel = wrow + kk * 2u;
                    let bits = read_w(rel) | (read_w(rel + 1u) << 8u);
                    acc = acc + xarr[xb + kk] * bitcast<f32>(bits << 16u);
                }
            }
        }
    }

    red8[t] = acc;
    workgroupBarrier();
    for (var s: u32 = 16u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            red8[t] = red8[t] + red8[t + s];
        }
        workgroupBarrier();
    }
    if (col_on && lane == 0u) {
        outarr[params.out_f32_off + row * params.out_stride + j] = red8[t];
    }
}

// ── Q4_K and Q6_K: 16 columns × up to 8 rows per 256-thread workgroup ────
//
// 16 lanes per column, each lane 16 weights of a block, decoded once and used
// against up to 8 activation rows (`R8`): a prefill's rows share every weight
// read instead of each re-reading the whole matrix (decode is the one-row
// case). Word-wide weight reads; the block scales come out of the inner sums
// (Q4_K: Σ x·(d·sc·q − dmin·m) = d·sc·Σ x·q − dmin·m·Σ x; Q6_K: one scale per
// sub-block applied to its partial dot), leaving each weight a nibble
// extract, a convert and one multiply-add per row.

const R8: u32 = 8u;
var<workgroup> red_rows: array<f32, 2048>;   // 8 rows × 256 threads

// Sum each row's 16 lane partials (lanes of a column are 16 consecutive
// threads) and write rows < nr of column j.
// `nr` is the same for the whole workgroup, so only live rows are reduced —
// a decode step's one row, not eight.
fn reduce_rows16(t: u32, lane: u32, acc: array<f32, 8>, nr: u32, row0: u32, j: u32, col_on: bool) {
    for (var r: u32 = 0u; r < nr; r = r + 1u) {
        red_rows[r * 256u + t] = acc[r];
    }
    workgroupBarrier();
    for (var s: u32 = 8u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            for (var r: u32 = 0u; r < nr; r = r + 1u) {
                red_rows[r * 256u + t] = red_rows[r * 256u + t] + red_rows[r * 256u + t + s];
            }
        }
        workgroupBarrier();
    }
    if (col_on && lane == 0u) {
        for (var r: u32 = 0u; r < nr; r = r + 1u) {
            outarr[params.out_f32_off + (row0 + r) * params.out_stride + j] = red_rows[r * 256u + t];
        }
    }
}

@compute @workgroup_size(256)
fn dequant_gemv_q4k16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    let lane = t % 16u;
    let slot = t / 16u;
    let j = (wid.z * nwg.x + wid.x) * 16u + slot;
    let col_on = j < params.n;
    let row0 = wid.y * R8;
    let nr = min(R8, max(params.m_rows, 1u) - row0);
    var acc: array<f32, 8>;

    if (col_on) {
        let nbk = params.k / 256u;
        // Group g (64 weights: 32 low nibbles, 32 high) and the lane's 8 bytes
        // within its 32.
        let g = lane / 4u;
        let l = (lane % 4u) * 8u;
        for (var b: u32 = 0u; b < nbk; b = b + 1u) {
            let off = (j * nbk + b) * 144u;
            let dm = unpack2x16float(read_w4(off));
            let sc = vec3<u32>(read_w4(off + 4u), read_w4(off + 8u), read_w4(off + 12u));
            let sm = q4k_scale_mins(sc, g);
            let q0 = read_w4(off + 16u + g * 32u + l);
            let q1 = read_w4(off + 16u + g * 32u + l + 4u);
            // The lane's 16 levels, decoded once for every row: low nibbles
            // (sub-block 2g) and high (2g + 1), as vec4s.
            let lo0 = vec4<f32>(unpack4xU8(q0 & 0x0F0F0F0Fu));
            let lo1 = vec4<f32>(unpack4xU8(q1 & 0x0F0F0F0Fu));
            let hi0 = vec4<f32>(unpack4xU8((q0 >> 4u) & 0x0F0F0F0Fu));
            let hi1 = vec4<f32>(unpack4xU8((q1 >> 4u) & 0x0F0F0F0Fu));
            let one = vec4<f32>(1.0);
            for (var r: u32 = 0u; r < nr; r = r + 1u) {
                let xo = params.x_f32_off + (row0 + r) * params.k + b * 256u + g * 64u + l;
                let x0 = vec4<f32>(xarr[xo], xarr[xo + 1u], xarr[xo + 2u], xarr[xo + 3u]);
                let x1 = vec4<f32>(xarr[xo + 4u], xarr[xo + 5u], xarr[xo + 6u], xarr[xo + 7u]);
                let y0 = vec4<f32>(xarr[xo + 32u], xarr[xo + 33u], xarr[xo + 34u], xarr[xo + 35u]);
                let y1 = vec4<f32>(xarr[xo + 36u], xarr[xo + 37u], xarr[xo + 38u], xarr[xo + 39u]);
                let dot_lo = dot(x0, lo0) + dot(x1, lo1);
                let dot_hi = dot(y0, hi0) + dot(y1, hi1);
                let sum_lo = dot(x0 + x1, one);
                let sum_hi = dot(y0 + y1, one);
                acc[r] = acc[r] + dm.x * (sm.x * dot_lo + sm.z * dot_hi)
                    - dm.y * (sm.y * sum_lo + sm.w * sum_hi);
            }
        }
    }
    reduce_rows16(t, lane, acc, nr, row0, j, col_on);
}

@compute @workgroup_size(256)
fn dequant_gemv_q6k16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    let lane = t % 16u;
    let slot = t / 16u;
    let j = (wid.z * nwg.x + wid.x) * 16u + slot;
    let col_on = j < params.n;
    let row0 = wid.y * R8;
    let nr = min(R8, max(params.m_rows, 1u) - row0);
    var acc: array<f32, 8>;

    if (col_on) {
        let nbk = params.k / 256u;
        // Lane = (half h, 4 consecutive positions l0..l0+3) — one word of low
        // bits at l, one at l + 32, one of high bits; the four positions share
        // a sub-block (l / 16), so each of four scales applies once.
        let h = lane / 8u;
        let l0 = (lane % 8u) * 4u;
        let is = l0 / 16u;
        for (var b: u32 = 0u; b < nbk; b = b + 1u) {
            let off = (j * nbk + b) * 210u;
            let ql_a = read_w4(off + h * 64u + l0);
            let ql_b = read_w4(off + h * 64u + l0 + 32u);
            let qh = read_w4(off + 128u + h * 32u + l0);
            // Scale bytes is, is+2, is+4, is+6 of this half's eight.
            let scw = vec2<u32>(read_w4(off + 192u + h * 8u), read_w4(off + 196u + h * 8u));
            let d = f16_at(off + 208u);
            var scs = vec4<f32>(0.0);
            for (var p: u32 = 0u; p < 4u; p = p + 1u) {
                let k = is + 2u * p;
                let byte = (scw[k / 4u] >> ((k % 4u) * 8u)) & 0xffu;
                scs[p] = d * f32(select(i32(byte), i32(byte) - 256, byte >= 128u));
            }
            var q: array<vec4<f32>, 4>;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let sh = i * 8u;
                let a = (ql_a >> sh) & 0xffu;
                let c = (ql_b >> sh) & 0xffu;
                let hb = (qh >> sh) & 0xffu;
                q[i] = vec4<f32>(
                    f32(i32((a & 0x0Fu) | ((hb & 3u) << 4u)) - 32),
                    f32(i32((c & 0x0Fu) | (((hb >> 2u) & 3u) << 4u)) - 32),
                    f32(i32((a >> 4u) | (((hb >> 4u) & 3u) << 4u)) - 32),
                    f32(i32((c >> 4u) | (((hb >> 6u) & 3u) << 4u)) - 32),
                );
            }
            for (var r: u32 = 0u; r < nr; r = r + 1u) {
                let xo = params.x_f32_off + (row0 + r) * params.k + b * 256u + h * 128u + l0;
                var part = vec4<f32>(0.0);
                for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                    part = part + q[i] * vec4<f32>(
                        xarr[xo + i],
                        xarr[xo + 32u + i],
                        xarr[xo + 64u + i],
                        xarr[xo + 96u + i],
                    );
                }
                acc[r] = acc[r] + dot(part, scs);
            }
        }
    }
    reduce_rows16(t, lane, acc, nr, row0, j, col_on);
}

// ── Q5_K: as `dequant_gemv_q4k16`, plus the high-bit plane ───────────────
//
// 176 bytes / 256: d, dmin, the Q4_K scales, 32 bytes of high bits (bit 2g of
// byte l for the low nibble of group g at position l, bit 2g+1 for the high
// one), then the 128 nibble bytes.

@compute @workgroup_size(256)
fn dequant_gemv_q5k16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    let lane = t % 16u;
    let slot = t / 16u;
    let j = (wid.z * nwg.x + wid.x) * 16u + slot;
    let col_on = j < params.n;
    let row0 = wid.y * R8;
    let nr = min(R8, max(params.m_rows, 1u) - row0);
    var acc: array<f32, 8>;

    if (col_on) {
        let nbk = params.k / 256u;
        let g = lane / 4u;
        let l = (lane % 4u) * 8u;
        for (var b: u32 = 0u; b < nbk; b = b + 1u) {
            let off = (j * nbk + b) * 176u;
            let dm = unpack2x16float(read_w4(off));
            let sc = vec3<u32>(read_w4(off + 4u), read_w4(off + 8u), read_w4(off + 12u));
            let sm = q4k_scale_mins(sc, g);
            let h0 = read_w4(off + 16u + l) >> (2u * g);
            let h1 = read_w4(off + 20u + l) >> (2u * g);
            let q0 = read_w4(off + 48u + g * 32u + l);
            let q1 = read_w4(off + 52u + g * 32u + l);
            var lo: array<f32, 8>;
            var hi: array<f32, 8>;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let sh = i * 8u;
                lo[i] = f32(((q0 >> sh) & 0x0Fu) | (((h0 >> sh) & 1u) << 4u));
                hi[i] = f32(((q0 >> (sh + 4u)) & 0x0Fu) | (((h0 >> (sh + 1u)) & 1u) << 4u));
                lo[i + 4u] = f32(((q1 >> sh) & 0x0Fu) | (((h1 >> sh) & 1u) << 4u));
                hi[i + 4u] = f32(((q1 >> (sh + 4u)) & 0x0Fu) | (((h1 >> (sh + 1u)) & 1u) << 4u));
            }
            for (var r: u32 = 0u; r < nr; r = r + 1u) {
                let xo = params.x_f32_off + (row0 + r) * params.k + b * 256u + g * 64u + l;
                var dot_lo = 0.0;
                var dot_hi = 0.0;
                var sum_lo = 0.0;
                var sum_hi = 0.0;
                for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                    let x = xarr[xo + i];
                    let y = xarr[xo + 32u + i];
                    dot_lo = dot_lo + x * lo[i];
                    dot_hi = dot_hi + y * hi[i];
                    sum_lo = sum_lo + x;
                    sum_hi = sum_hi + y;
                }
                acc[r] = acc[r] + dm.x * (sm.x * dot_lo + sm.z * dot_hi)
                    - dm.y * (sm.y * sum_lo + sm.w * sum_hi);
            }
        }
    }
    reduce_rows16(t, lane, acc, nr, row0, j, col_on);
}

// ── 32-weight blocks: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL and IQ4_XS ────
//
// Same 16 columns × 8 rows layout; a lane takes whole 32-weight (sub-)blocks
// of its column, decodes one to integer levels q and a pair (a, c) with
// w = a·q + c, and uses it for every row as a·Σx·q + c·Σx. IQ4_XS is a
// 256-weight block of eight such sub-blocks, each with its own 6-bit scale.

const IQ4NL_LEVELS = array<f32, 16>(
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
    1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0,
);

// Sub-block `s` of weight row `j` (`nsb` sub-blocks per row) into `q`;
// returns (a, c).
fn b32_levels(j: u32, s: u32, nsb: u32, q: ptr<function, array<f32, 32>>) -> vec2<f32> {
    let id = params.scheme_id;
    if (id == 20u) {
        // Q8_0: d, 32 × i8.
        let off = (j * nsb + s) * 34u;
        for (var w: u32 = 0u; w < 8u; w = w + 1u) {
            let word = read_w4(off + 2u + w * 4u);
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                (*q)[w * 4u + i] = f32(extractBits(i32(word), i * 8u, 8u));
            }
        }
        return vec2<f32>(f16_at(off), 0.0);
    }
    if (id == 25u) {
        // Q2_0: f16 d | 32 bytes of 2-bit codes per 128 weights, LSB-first,
        // code q -> (q - 1)·d = d·q - d. Sub-block s is bytes 8·(s % 4).. of
        // block s / 4.
        let off = (j * (nsb / 4u) + s / 4u) * 34u;
        let qs = off + 2u + (s % 4u) * 8u;
        for (var w: u32 = 0u; w < 2u; w = w + 1u) {
            let word = read_w4(qs + w * 4u);
            for (var i: u32 = 0u; i < 16u; i = i + 1u) {
                (*q)[w * 16u + i] = f32((word >> (i * 2u)) & 3u);
            }
        }
        let d = f16_at(off);
        return vec2<f32>(d, -d);
    }
    // The nibble formats: 16 bytes, low nibbles are weights 0..15, high
    // nibbles 16..31; `hb` holds a fifth bit per weight (Q5_0 / Q5_1).
    var off: u32;
    var qs: u32;
    var hb = 0u;
    var ac: vec2<f32>;
    var table = false;
    if (id == 19u) {
        off = (j * nsb + s) * 18u;
        qs = off + 2u;
        let d = f16_at(off);
        ac = vec2<f32>(d, -8.0 * d);
    } else if (id == 21u) {
        off = (j * nsb + s) * 20u;
        qs = off + 4u;
        ac = unpack2x16float(read_w4(off));
    } else if (id == 22u) {
        off = (j * nsb + s) * 22u;
        qs = off + 6u;
        hb = read_w4(off + 2u);
        let d = f16_at(off);
        ac = vec2<f32>(d, -16.0 * d);
    } else if (id == 23u) {
        off = (j * nsb + s) * 24u;
        qs = off + 8u;
        hb = read_w4(off + 4u);
        ac = unpack2x16float(read_w4(off));
    } else if (id == 6u) {
        off = (j * nsb + s) * 18u;
        qs = off + 2u;
        ac = vec2<f32>(f16_at(off), 0.0);
        table = true;
    } else {
        // IQ4_XS: d, scales_h (u16), scales_l (4 bytes), 128 bytes of nibbles.
        let ib = s % 8u;
        off = (j * (nsb / 8u) + s / 8u) * 136u;
        let head = read_w4(off);
        let lo = (read_w(off + 4u + ib / 2u) >> (4u * (ib % 2u))) & 0x0Fu;
        let hi = (head >> (16u + 2u * ib)) & 3u;
        qs = off + 8u + ib * 16u;
        ac = vec2<f32>(unpack2x16float(head).x * f32(i32(lo | (hi << 4u)) - 32), 0.0);
        table = true;
    }
    for (var w: u32 = 0u; w < 4u; w = w + 1u) {
        let word = read_w4(qs + w * 4u);
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let e = w * 4u + i;
            let lo = ((word >> (i * 8u)) & 0x0Fu) | (((hb >> e) & 1u) << 4u);
            let hi = ((word >> (i * 8u + 4u)) & 0x0Fu) | (((hb >> (e + 16u)) & 1u) << 4u);
            if (table) {
                (*q)[e] = IQ4NL_LEVELS[lo];
                (*q)[e + 16u] = IQ4NL_LEVELS[hi];
            } else {
                (*q)[e] = f32(lo);
                (*q)[e + 16u] = f32(hi);
            }
        }
    }
    return ac;
}

@compute @workgroup_size(256)
fn dequant_gemv_b32x16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    let lane = t % 16u;
    let slot = t / 16u;
    let j = (wid.z * nwg.x + wid.x) * 16u + slot;
    let col_on = j < params.n;
    let row0 = wid.y * R8;
    let nr = min(R8, max(params.m_rows, 1u) - row0);
    var acc: array<f32, 8>;

    if (col_on) {
        let nsb = params.k / 32u;
        var q: array<f32, 32>;
        for (var s: u32 = lane; s < nsb; s = s + 16u) {
            let ac = b32_levels(j, s, nsb, &q);
            for (var r: u32 = 0u; r < nr; r = r + 1u) {
                let xo = params.x_f32_off + (row0 + r) * params.k + s * 32u;
                var dq = 0.0;
                var sx = 0.0;
                for (var i: u32 = 0u; i < 32u; i = i + 1u) {
                    let x = xarr[xo + i];
                    dq = dq + x * q[i];
                    sx = sx + x;
                }
                acc[r] = acc[r] + ac.x * dq + ac.y * sx;
            }
        }
    }
    reduce_rows16(t, lane, acc, nr, row0, j, col_on);
}
