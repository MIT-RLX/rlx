// RLX — GGUF dequant WGSL (port of dequant_gguf.msl / dequant_gguf.cu).
// One thread per GGUF block. scheme_id 0..=21 matches Metal/CUDA (Q4_0=19, Q8_0=20, Q4_1=21).
// Backend matrix: docs/gguf-backend-paths.md

struct Params {
    w_byte_off: u32,
    dst_f32_off: u32,
    scheme_id: u32,
    num_blocks: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;
@group(0) @binding(2) var<storage, read>      iq_lut: array<u32>;

const kvalues_iq4nl_lut: array<i32, 16> = array<i32, 16>(
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
);
const FP4_E2M1: array<f32, 16> = array<f32, 16>(
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
);

const POW3: array<u32, 5> = array<u32, 5>(1u, 3u, 9u, 27u, 81u);

const IQ_GRID_OFF_KMASK: u32 = 0u;
const IQ_GRID_OFF_KSIGNS: u32 = 8u;
const IQ_GRID_OFF_IQ2XXS: u32 = 136u;
const IQ_GRID_OFF_IQ2XS: u32 = 2184u;
const IQ_GRID_OFF_IQ2S: u32 = 6280u;
const IQ_GRID_OFF_IQ3XXS: u32 = 14472u;
const IQ_GRID_OFF_IQ3S: u32 = 15496u;
const IQ_GRID_OFF_IQ1S: u32 = 17544u;

fn read_w(rel: u32) -> u32 {
    let abs = params.w_byte_off + rel;
    let word = abs / 4u;
    let shift = (abs % 4u) * 8u;
    return (bitcast<u32>(arena[word]) >> shift) & 0xffu;
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

fn lut_u8(off: u32) -> u32 {
    return (iq_lut[off / 4u] >> ((off % 4u) * 8u)) & 0xffu;
}

fn lut_i8(off: u32) -> i32 {
    let b = lut_u8(off);
    return select(i32(b), i32(b) - 256, b >= 128u);
}

fn lut_u16_w(rel: u32) -> u32 {
    return read_w(rel) | (read_w(rel + 1u) << 8u);
}

fn scale_byte(v: u32, i: u32) -> i32 {
    let b = (v >> (i * 8u)) & 0xffu;
    return select(i32(b), i32(b) - 256, b >= 128u);
}

fn mul_u8_pow3(byte: u32, pow3: u32) -> u32 {
    // TQ1_0 trit extraction uses u8 wrap (matches MSL `uchar` / CUDA `unsigned char`).
    return (byte * pow3) & 0xffu;
}

fn pack_scales_from_aux(aux0: u32, aux1: u32) -> array<i32, 8> {
    var out: array<i32, 8>;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) { out[i] = scale_byte(aux0, i); }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) { out[i + 4u] = scale_byte(aux1, i); }
    return out;
}

@compute @workgroup_size(256)
fn dequant_gguf(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= params.num_blocks) { return; }

    // ── 256-elem block schemes (K-quants, IQ4_XS, IQ2/3/1, TQ) ─────
    if (params.scheme_id != 6u && params.scheme_id != 10u && params.scheme_id != 11u
        && params.scheme_id != 19u && params.scheme_id != 20u
        && params.scheme_id != 21u && params.scheme_id != 22u && params.scheme_id != 23u) {
        let dst_base = params.dst_f32_off + gid * 256u;

        if (params.scheme_id == 3u) {
            let off = gid * (4u + 256u + (256u / 16u) * 2u);
            let d = bitcast<f32>(read_w(off) | (read_w(off + 1u) << 8u) | (read_w(off + 2u) << 16u) | (read_w(off + 3u) << 24u));
            let qs_rel = off + 4u;
            for (var i: u32 = 0u; i < 256u; i = i + 1u) {
                arena[dst_base + i] = d * f32(read_w_i8(qs_rel + i));
            }
            return;
        }

        if (params.scheme_id == 0u) {
            let blk = 2u + 2u + 12u + 256u / 2u;
            let off = gid * blk;
            let d = dq_read_f16(off);
            let dmin = dq_read_f16(off + 2u);
            let scales_rel = off + 4u;
            let qs_rel = off + 4u + 12u;
            var is = 0u;
            var out_i = 0u;
            for (var j: u32 = 0u; j < 8u; j = j + 2u) {
                let sm0 = dq_get_scale_min_k4(scales_rel, j);
                let sm1 = dq_get_scale_min_k4(scales_rel, j + 1u);
                let d0 = d * f32(sm0.x); let m0 = dmin * f32(sm0.y);
                let d1 = d * f32(sm1.x); let m1 = dmin * f32(sm1.y);
                for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                    arena[dst_base + out_i] = d0 * f32(read_w(qs_rel + is + l) & 0x0Fu) - m0;
                    out_i = out_i + 1u;
                }
                for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                    arena[dst_base + out_i] = d1 * f32(read_w(qs_rel + is + l) >> 4) - m1;
                    out_i = out_i + 1u;
                }
                is = is + 32u;
            }
            return;
        }

        if (params.scheme_id == 1u) {
            let blk = 2u + 2u + 12u + 256u / 8u + 256u / 2u;
            let off = gid * blk;
            let d = dq_read_f16(off);
            let dmin = dq_read_f16(off + 2u);
            let scales_rel = off + 4u;
            let qh_off = off + 4u + 12u;
            let qh_rel = qh_off;
            let qs_rel = qh_off + 256u / 8u;
            var is = 0u;
            var out_i = 0u;
            var u1 = 1u;
            var u2 = 2u;
            for (var j: u32 = 0u; j < 8u; j = j + 2u) {
                let sm0 = dq_get_scale_min_k4(scales_rel, j);
                let sm1 = dq_get_scale_min_k4(scales_rel, j + 1u);
                let d0 = d * f32(sm0.x); let m0 = dmin * f32(sm0.y);
                let d1 = d * f32(sm1.x); let m1 = dmin * f32(sm1.y);
                for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                    let lo = u32(read_w(qs_rel + is + l)) & 0x0Fu;
                    let hi = select(0u, 16u, (read_w(qh_rel + l) & u1) != 0u);
                    arena[dst_base + out_i] = d0 * f32(lo + hi) - m0;
                    out_i = out_i + 1u;
                }
                for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                    let lo = u32(read_w(qs_rel + is + l)) >> 4u;
                    let hi = select(0u, 16u, (read_w(qh_rel + l) & u2) != 0u);
                    arena[dst_base + out_i] = d1 * f32(lo + hi) - m1;
                    out_i = out_i + 1u;
                }
                is = is + 32u;
                u1 = u1 << 2u;
                u2 = u2 << 2u;
            }
            return;
        }

        if (params.scheme_id == 2u) {
            let ql_len = 256u / 2u;
            let qh_len = 256u / 4u;
            let sc_len = 256u / 16u;
            let blk = ql_len + qh_len + sc_len + 2u;
            let off = gid * blk;
            let ql_rel = off;
            let qh_rel = off + ql_len;
            let sc_rel = off + ql_len + qh_len;
            let d = dq_read_f16(off + ql_len + qh_len + sc_len);
            for (var h: u32 = 0u; h < 2u; h = h + 1u) {
                let row_base = h * 128u; let ql_off = h * 64u; let qh_off_h = h * 32u; let sc_off = h * 8u;
                for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                    let is = l / 16u;
                    let qh_b = read_w(qh_rel + qh_off_h + l);
                    let q1 = f32(i32(((read_w(ql_rel + ql_off + l) & 0x0Fu) | (((qh_b >> 0) & 3u) << 4u)) - 32));
                    let q2 = f32(i32(((read_w(ql_rel + ql_off + l + 32u) & 0x0Fu) | (((qh_b >> 2) & 3u) << 4u)) - 32));
                    let q3 = f32(i32(((read_w(ql_rel + ql_off + l) >> 4) | (((qh_b >> 4) & 3u) << 4u)) - 32));
                    let q4 = f32(i32(((read_w(ql_rel + ql_off + l + 32u) >> 4) | (((qh_b >> 6) & 3u) << 4u)) - 32));
                    arena[dst_base + row_base + l] = d * f32(read_w_i8(sc_rel + sc_off + is)) * q1;
                    arena[dst_base + row_base + l + 32u] = d * f32(read_w_i8(sc_rel + sc_off + is + 2u)) * q2;
                    arena[dst_base + row_base + l + 64u] = d * f32(read_w_i8(sc_rel + sc_off + is + 4u)) * q3;
                    arena[dst_base + row_base + l + 96u] = d * f32(read_w_i8(sc_rel + sc_off + is + 6u)) * q4;
                }
            }
            return;
        }

        if (params.scheme_id == 4u) {
            // Q2_K: read_w(scales_rel + 16) | read_w(qs_rel + 64) | d (f16) | dmin (f16).
            let blk = 2u + 2u + 256u / 16u + 256u / 4u;
            let off = gid * blk;
            let scales_off = off;
            let qs_off = off + 256u / 16u;
            let d_off = qs_off + 256u / 4u;
            let d = dq_read_f16(d_off);
            let min_v = dq_read_f16(d_off + 2u);
            var q_rel = qs_off;
            var is = 0u;
            var out_i = 0u;
            for (var sb: u32 = 0u; sb < 2u; sb = sb + 1u) {
                var shift = 0u;
                for (var t: u32 = 0u; t < 4u; t = t + 1u) {
                    var sc = read_w(scales_off + is); is = is + 1u;
                    let dl = d * f32(sc & 0xFu);
                    let ml = min_v * f32(sc >> 4);
                    for (var l: u32 = 0u; l < 16u; l = l + 1u) {
                        arena[dst_base + out_i] = dl * f32((read_w(q_rel + l) >> shift) & 3u) - ml;
                        out_i = out_i + 1u;
                    }
                    sc = read_w(scales_off + is); is = is + 1u;
                    let dl2 = d * f32(sc & 0xFu);
                    let ml2 = min_v * f32(sc >> 4);
                    for (var l: u32 = 0u; l < 16u; l = l + 1u) {
                        arena[dst_base + out_i] = dl2 * f32((read_w(q_rel + l + 16u) >> shift) & 3u) - ml2;
                        out_i = out_i + 1u;
                    }
                    shift = shift + 2u;
                }
                q_rel = q_rel + 32u;
            }
            return;
        }

        if (params.scheme_id == 5u) {
            // Q3_K: hmask[32] | read_w(qs_rel + 64) | read_w(scales_rel + 12) | d (f16).
            let KMASK1 = 0x03030303u;
            let KMASK2 = 0x0f0f0f0fu;
            let blk = 2u + 12u + 256u / 8u + 256u / 4u;
            let off = gid * blk;
            let hm_off = off;
            let qs_off = off + 256u / 8u;
            let scales_off = qs_off + 256u / 4u;
            let d_off = scales_off + 12u;
            let d_all = dq_read_f16(d_off);
            let hm_rel = hm_off;
            let q_rel = qs_off;
            var aux0 = u32(read_w(scales_off + 0u)) | (u32(read_w(scales_off + 1u)) << 8u)
                      | (u32(read_w(scales_off + 2u)) << 16u) | (u32(read_w(scales_off + 3u)) << 24u);
            var aux1 = u32(read_w(scales_off + 4u)) | (u32(read_w(scales_off + 5u)) << 8u)
                      | (u32(read_w(scales_off + 6u)) << 16u) | (u32(read_w(scales_off + 7u)) << 24u);
            let tmp = u32(read_w(scales_off + 8u)) | (u32(read_w(scales_off + 9u)) << 8u)
                      | (u32(read_w(scales_off + 10u)) << 16u) | (u32(read_w(scales_off + 11u)) << 24u);
            let aux2 = ((aux0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4u);
            aux0 = (aux0 & KMASK2) | (((tmp >> 0u) & KMASK1) << 4u);
            aux1 = (aux1 & KMASK2) | (((tmp >> 2u) & KMASK1) << 4u);
            let scales = pack_scales_from_aux(aux0, aux1);
            var is = 0u;
            var m: u32 = 1u;
            var out_i = 0u;
            var q_walk = q_rel;
            for (var sb: u32 = 0u; sb < 2u; sb = sb + 1u) {
                var shift = 0u;
                for (var t: u32 = 0u; t < 4u; t = t + 1u) {
                    let dl = d_all * f32(scales[is] - 32); is = is + 1u;
                    for (var l: u32 = 0u; l < 16u; l = l + 1u) {
                        let h = select(0, 4, (read_w(hm_rel + l) & m) == 0u);
                        arena[dst_base + out_i] = dl * f32(i32(((read_w(q_walk + l) >> shift) & 3u)) - h);
                        out_i = out_i + 1u;
                    }
                    let dl2 = d_all * f32(scales[is] - 32); is = is + 1u;
                    for (var l: u32 = 0u; l < 16u; l = l + 1u) {
                        let h = select(0, 4, (read_w(hm_rel + l + 16u) & m) == 0u);
                        arena[dst_base + out_i] = dl2 * f32(i32(((read_w(q_walk + l + 16u) >> shift) & 3u)) - h);
                        out_i = out_i + 1u;
                    }
                    shift = shift + 2u;
                    m = m << 1u;
                }
                q_walk = q_walk + 32u;
            }
            return;
        }

        if (params.scheme_id == 7u) {
            // IQ4_XS: 136 bytes / 256 elems.
            let blk = 2u + 2u + 256u / 64u + 256u / 2u;
            let off = gid * blk;
            let d = dq_read_f16(off);
            let scales_h = u32(read_w(off + 2u)) | (u32(read_w(off + 3u)) << 8u);
            let scales_l_rel = off + 4u;
            let qs_rel = off + 4u + 256u / 64u;
            var out_i = 0u;
            var qs_idx = 0u;
            for (var ib: u32 = 0u; ib < 256u / 32u; ib = ib + 1u) {
                let lo = (u32(read_w(scales_l_rel + ib / 2u)) >> (4u * (ib % 2u))) & 0xFu;
                let hi = (scales_h >> (2u * ib)) & 0x3u;
                let ls = i32(lo | (hi << 4u));
                let dl = d * f32(ls - 32);
                for (var j: u32 = 0u; j < 16u; j = j + 1u) {
                    let bx = read_w(qs_rel + qs_idx + j);
                    arena[dst_base + out_i + j] = dl * f32(kvalues_iq4nl_lut[bx & 0x0Fu]);
                    arena[dst_base + out_i + j + 16u] = dl * f32(kvalues_iq4nl_lut[bx >> 4]);
                }
                out_i = out_i + 32u;
                qs_idx = qs_idx + 16u;
            }
            return;
        }

        if (params.scheme_id == 8u) {
            // TQ1_0: read_w(qs_rel + 48) | read_w(qh_rel + 4) | d (f16). Trit unpacking via the
            // (q * pow3[n] * 3) >> 8 trick — see rlx_gguf::tq_dequant.
            // uses POW3 const
            let off = gid * 54u;
            let qs_rel = off;
            let qh_rel = off + 48u;
            let d = dq_read_f16(off + 52u);
            var y = 0u;
            // First chunk: 32 bytes packing 160 trits.
            for (var n: u32 = 0u; n < 5u; n = n + 1u) {
                for (var m: u32 = 0u; m < 32u; m = m + 1u) {
                    let q = mul_u8_pow3(read_w(qs_rel + m), POW3[n]);
                    let xi = i32((u32(q) * 3u) >> 8u);
                    arena[dst_base + y] = f32(xi - 1) * d;
                    y = y + 1u;
                }
            }
            // Second chunk: 16 bytes packing 80 trits.
            for (var n: u32 = 0u; n < 5u; n = n + 1u) {
                for (var m: u32 = 0u; m < 16u; m = m + 1u) {
                    let q = mul_u8_pow3(read_w(qs_rel + 32u + m), POW3[n]);
                    let xi = i32((u32(q) * 3u) >> 8u);
                    arena[dst_base + y] = f32(xi - 1) * d;
                    y = y + 1u;
                }
            }
            // qh: 4 bytes packing 16 trits.
            for (var n: u32 = 0u; n < 4u; n = n + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let q = mul_u8_pow3(read_w(qh_rel + j), POW3[n]);
                    let xi = i32((u32(q) * 3u) >> 8u);
                    arena[dst_base + y] = f32(xi - 1) * d;
                    y = y + 1u;
                }
            }
            return;
        }

        if (params.scheme_id == 9u) {
            // TQ2_0: read_w(qs_rel + 64) | d (f16). 4 trits per byte (low→high) but
            // emission order is (j+m, slot l) → element index out_y.
            let off = gid * 66u;
            let qs_rel = off;
            let d = dq_read_f16(off + 64u);
            var y = 0u;
            for (var j: u32 = 0u; j < 64u; j = j + 32u) {
                for (var l: u32 = 0u; l < 4u; l = l + 1u) {
                    for (var m: u32 = 0u; m < 32u; m = m + 1u) {
                        let q = i32((read_w(qs_rel + j + m) >> (l * 2u)) & 3u);
                    arena[dst_base + y] = f32(q - 1) * d;
                    y = y + 1u;
                    }
                }
            }
            return;
        }

        if (params.scheme_id == 12u) {
            // IQ2_XXS: f16 d | u16 read_w(qs_rel + 32) (= 64 bytes). 4 u16 per
            // 32-elem sub-block; aux32[0] = 4 grid indices (u8),
            // aux32[1] = 4 × 7-bit sign masks + 4-bit scale.
            let off = gid * 66u;
            let d = dq_read_f16(off);
            let qs_rel = off + 2u;
            var y = 0u;
            for (var ib32: u32 = 0u; ib32 < 8u; ib32 = ib32 + 1u) {
                let base = 8u * ib32;
                let aux32_0 = u32(read_w(qs_rel + base + 0u)) | (u32(read_w(qs_rel + base + 1u)) << 8u)
                             | (u32(read_w(qs_rel + base + 2u)) << 16u) | (u32(read_w(qs_rel + base + 3u)) << 24u);
                let aux32_1 = u32(read_w(qs_rel + base + 4u)) | (u32(read_w(qs_rel + base + 5u)) << 8u)
                             | (u32(read_w(qs_rel + base + 6u)) << 16u) | (u32(read_w(qs_rel + base + 7u)) << 24u);
                let a0 = u32(aux32_0 & 0xFFu);
                let a1 = u32((aux32_0 >> 8) & 0xFFu);
                let a2 = u32((aux32_0 >> 16) & 0xFFu);
                let a3 = u32((aux32_0 >> 24) & 0xFFu);
                let db = d * (0.5 + f32(aux32_1 >> 28)) * 0.25;
                for (var l: u32 = 0u; l < 4u; l = l + 1u) {
                    let gb = select(a3, select(a2, select(a1, a0, l == 0u), l == 1u), l == 2u);
                    let grid_off = IQ_GRID_OFF_IQ2XXS + gb * 8u;
                    let signs = lut_u8( IQ_GRID_OFF_KSIGNS + ((aux32_1 >> (7u * l)) & 127u));
                    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                        let gv = lut_i8( grid_off + j);
                        let mask = lut_u8( IQ_GRID_OFF_KMASK + j);
                        let s = select(1.0, -1.0, (signs & mask) != 0u);
                        arena[dst_base + y + j] = db * f32(gv) * s;
                    }
                    y = y + 8u;
                }
            }
            return;
        }

        if (params.scheme_id == 13u) {
            // IQ2_XS: f16 d | u16 read_w(qs_rel + 32) | u8 read_w(scales_rel + 8) = 74 bytes.
            let off = gid * 74u;
            let d = dq_read_f16(off);
            let qs_rel = off + 2u;
            let scales_rel = off + 2u + 64u;
            var y = 0u;
            for (var ib32: u32 = 0u; ib32 < 8u; ib32 = ib32 + 1u) {
                let db0 = d * (0.5 + f32(read_w(scales_rel + ib32) & 0xFu)) * 0.25;
                let db1 = d * (0.5 + f32(read_w(scales_rel + ib32) >> 4u)) * 0.25;
                for (var l: u32 = 0u; l < 4u; l = l + 1u) {
                    let pos = (4u * ib32 + l) * 2u;
                    let q = u32(read_w(qs_rel + pos)) | (u32(read_w(qs_rel + pos + 1u)) << 8u);
                    let grid_off = IQ_GRID_OFF_IQ2XS + (q & 511u) * 8u;
                    let signs = lut_u8( IQ_GRID_OFF_KSIGNS + (q >> 9));
                    let dl = select(db1, db0, l / 2u == 0u);
                    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                        let gv = lut_i8( grid_off + j);
                        let mask = lut_u8( IQ_GRID_OFF_KMASK + j);
                        let s = select(1.0, -1.0, (signs & mask) != 0u);
                        arena[dst_base + y + j] = dl * f32(gv) * s;
                    }
                    y = y + 8u;
                }
            }
            return;
        }

        if (params.scheme_id == 14u) {
            // IQ2_S: f16 d | read_w(qs_rel + 64) | read_w(qh_rel + 8) | read_w(scales_rel + 8) = 82 bytes.
            let off = gid * 82u;
            let d = dq_read_f16(off);
            let qs_rel = off + 2u;
            let qh_rel = off + 2u + 64u;
            let scales_rel = off + 2u + 64u + 8u;
            var y = 0u;
            var qs_idx = 0u;
            var signs_idx = 32u;
            for (var ib32: u32 = 0u; ib32 < 8u; ib32 = ib32 + 1u) {
                let db0 = d * (0.5 + f32(read_w(scales_rel + ib32) & 0xFu)) * 0.25;
                let db1 = d * (0.5 + f32(read_w(scales_rel + ib32) >> 4u)) * 0.25;
                for (var l: u32 = 0u; l < 4u; l = l + 1u) {
                    let dl = select(db1, db0, l / 2u == 0u);
                    let q = u32(read_w(qs_rel + qs_idx + l));
                    let qh_b = u32(read_w(qh_rel + ib32));
                    let idx = q | ((qh_b << (8u - 2u * l)) & 0x300u);
                    let grid_off = IQ_GRID_OFF_IQ2S + idx * 8u;
                    let sign = read_w(qs_rel + signs_idx + l);
                    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                        let gv = lut_i8( grid_off + j);
                        let mask = lut_u8( IQ_GRID_OFF_KMASK + j);
                        let s = select(1.0, -1.0, (sign & mask) != 0u);
                        arena[dst_base + y + j] = dl * f32(gv) * s;
                    }
                    y = y + 8u;
                }
                qs_idx = qs_idx + 4u; signs_idx = signs_idx + 4u;
            }
            return;
        }

        if (params.scheme_id == 15u) {
            // IQ3_XXS: f16 d | read_w(qs_rel + 64) | scales_and_signs[32] = 98 bytes.
            let off = gid * 98u;
            let d = dq_read_f16(off);
            let qs_rel = off + 2u;
            let sas_rel = off + 2u + 64u;
            var y = 0u;
            var qs_idx = 0u;
            for (var ib32: u32 = 0u; ib32 < 8u; ib32 = ib32 + 1u) {
                let aux32 = u32(read_w(sas_rel + 4u*ib32)) | (u32(read_w(sas_rel + 4u*ib32+1u)) << 8u)
                           | (u32(read_w(sas_rel + 4u*ib32+2u)) << 16u) | (u32(read_w(sas_rel + 4u*ib32+3u)) << 24u);
                let db = d * (0.5 + f32(aux32 >> 28)) * 0.5;
                for (var l: u32 = 0u; l < 4u; l = l + 1u) {
                    let signs = lut_u8( IQ_GRID_OFF_KSIGNS + ((aux32 >> (7u * l)) & 127u));
                    let g1_off = IQ_GRID_OFF_IQ3XXS + u32(read_w(qs_rel + qs_idx + 2u * l)) * 4u;
                    let g2_off = IQ_GRID_OFF_IQ3XXS + u32(read_w(qs_rel + qs_idx + 2u * l + 1u)) * 4u;
                    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                        let g1 = lut_i8( g1_off + j);
                        let g2 = lut_i8( g2_off + j);
                        let m0 = lut_u8( IQ_GRID_OFF_KMASK + j);
                        let m1 = lut_u8( IQ_GRID_OFF_KMASK + j + 4u);
                        let s0 = select(1.0, -1.0, (signs & m0) != 0u);
                        let s1 = select(1.0, -1.0, (signs & m1) != 0u);
                        arena[dst_base + y + j] = db * f32(g1) * s0;
                        arena[dst_base + y + j + 4u] = db * f32(g2) * s1;
                    }
                    y = y + 8u;
                }
                qs_idx = qs_idx + 8u;
            }
            return;
        }

        if (params.scheme_id == 16u) {
            // IQ3_S: f16 d | read_w(qs_rel + 64) | read_w(qh_rel + 8) | read_w(signs_rel + 32) | read_w(scales_rel + 4) = 110 bytes.
            let off = gid * 110u;
            let d = dq_read_f16(off);
            let qs_rel = off + 2u;
            let qh_rel = off + 2u + 64u;
            let signs_rel = off + 2u + 64u + 8u;
            let scales_rel = off + 2u + 64u + 8u + 32u;
            var y = 0u;
            var qs_walk = 0u;
            var signs_walk = 0u;
            var qh_walk = 0u;
            for (var ib32: u32 = 0u; ib32 < 8u; ib32 = ib32 + 2u) {
                let db1 = d * (1.0 + 2.0 * f32(read_w(scales_rel + ib32 / 2u) & 0xFu));
                let db2 = d * (1.0 + 2.0 * f32(read_w(scales_rel + ib32 / 2u) >> 4u));
                for (var half_iter: u32 = 0u; half_iter < 2u; half_iter = half_iter + 1u) {
                    let dl = select(db2, db1, half_iter == 0u);
                    let qh_byte = u32(read_w(qh_rel + qh_walk + half_iter));
                    for (var l: u32 = 0u; l < 4u; l = l + 1u) {
                        let idx1 = u32(read_w(qs_rel + qs_walk + 2u * l)) | (((qh_byte << (8u - 2u * l)) & 256u));
                        let idx2 = u32(read_w(qs_rel + qs_walk + 2u * l + 1u)) | (((qh_byte << (7u - 2u * l)) & 256u));
                        let g1_off = IQ_GRID_OFF_IQ3S + idx1 * 4u;
                        let g2_off = IQ_GRID_OFF_IQ3S + idx2 * 4u;
                        let sign = read_w(signs_rel + signs_walk + l);
                        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                            let g1 = lut_i8( g1_off + j);
                            let g2 = lut_i8( g2_off + j);
                            let m0 = lut_u8( IQ_GRID_OFF_KMASK + j);
                            let m1 = lut_u8( IQ_GRID_OFF_KMASK + j + 4u);
                            let s0 = select(1.0, -1.0, (sign & m0) != 0u);
                            let s1 = select(1.0, -1.0, (sign & m1) != 0u);
                            arena[dst_base + y + j] = dl * f32(g1) * s0;
                            arena[dst_base + y + j + 4u] = dl * f32(g2) * s1;
                        }
                        y = y + 8u;
                    }
                    qs_walk = qs_walk + 8u; signs_walk = signs_walk + 4u;
                }
                qh_walk = qh_walk + 2u;
            }
            return;
        }

        if (params.scheme_id == 17u) {
            // IQ1_S: f16 d | read_w(qs_rel + 32) | read_w(qh_rel + 16) (u16 × 8) = 50 bytes.
            let off = gid * 50u;
            let d = dq_read_f16(off);
            let qs_rel = off + 2u;
            let qh_bytes_rel = off + 2u + 32u;
            var y = 0u;
            var qs_idx = 0u;
            for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
                let qh = u32(read_w(qh_bytes_rel + 2u * ib)) | (u32(read_w(qh_bytes_rel + 2u * ib + 1u)) << 8u);
                let dl = d * (2.0 * f32((qh >> 12) & 7u) + 1.0);
                let delta = select(0.125, -0.125, (qh & 0x8000u) != 0u);
                for (var l: u32 = 0u; l < 4u; l = l + 1u) {
                    let idx = u32(read_w(qs_rel + qs_idx + l)) | (((qh >> (3u * l)) & 7u) << 8u);
                    let grid_off = IQ_GRID_OFF_IQ1S + idx * 8u;
                    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                        let gv = lut_i8( grid_off + j);
                        arena[dst_base + y + j] = dl * (f32(gv) + delta);
                    }
                    y = y + 8u;
                }
                qs_idx = qs_idx + 4u;
            }
            return;
        }

        if (params.scheme_id == 18u) {
            // IQ1_M: read_w(qs_rel + 32) | read_w(qh_rel + 16) | read_w(scales_rel + 8) = 56 bytes (no leading d!).
            let off = gid * 56u;
            let qs_rel = off;
            let qh_rel = off + 32u;
            let scales_b_rel = off + 48u;
            // Reconstruct f16 scale from 4 packed nibbles.
            let sc0 = lut_u16_w(scales_b_rel +  0u);
            let sc1 = lut_u16_w(scales_b_rel +  2u);
            let sc2 = lut_u16_w(scales_b_rel +  4u);
            let sc3 = lut_u16_w(scales_b_rel +  6u);
            let sc_u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0u)
                        | ((sc2 >> 4) & 0x0F00u) | (sc3 & 0xF000u);
            // Inline f16 decode.
            let sign = (sc_u16 >> 15) & 1u;
            let exp_v = (sc_u16 >> 10) & 0x1Fu;
            let mant = sc_u16 & 0x3FFu;
            var d: f32;
            if (exp_v == 0u) { d = f32(mant) / 1024.0 * exp2(-14.0); }
            else if (exp_v == 31u) { d = select(0.0, bitcast<f32>(0x7f800000u), mant == 0u); }
            else { d = (1.0 + f32(mant) / 1024.0) * exp2(f32(i32(exp_v) - 15)); }
            if (sign != 0u) { d = -d; }
            var y = 0u;
            var qs_walk = 0u;
            var qh_walk = 0u;
            for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
                let sc_word = select(
                    select(select(sc0, sc1, ib >= 2u), sc2, ib >= 4u),
                    sc3,
                    ib >= 6u,
                );
                let dl1 = d * (2.0 * f32((sc_word >> (6u * (ib % 2u))) & 0x7u) + 1.0);
                let dl2 = d * (2.0 * f32((sc_word >> (6u * (ib % 2u) + 3u)) & 0x7u) + 1.0);
                for (var l: u32 = 0u; l < 4u; l = l + 1u) {
                    var idx: u32;
                    var delta: f32;
                    var dl: f32;
                    if (l == 0u) {
                        idx = u32(read_w(qs_rel + qs_walk)) | ((u32(read_w(qh_rel + qh_walk)) << 8u) & 0x700u);
                        delta = select(0.125, -0.125, (read_w(qh_rel + qh_walk) & 0x08u) != 0u);
                        dl = dl1;
                    } else if (l == 1u) {
                        idx = u32(read_w(qs_rel + qs_walk + 1u)) | ((u32(read_w(qh_rel + qh_walk)) << 4u) & 0x700u);
                        delta = select(0.125, -0.125, (read_w(qh_rel + qh_walk) & 0x80u) != 0u);
                        dl = dl1;
                    } else if (l == 2u) {
                        idx = u32(read_w(qs_rel + qs_walk + 2u)) | ((u32(read_w(qh_rel + qh_walk + 1u)) << 8u) & 0x700u);
                        delta = select(0.125, -0.125, (read_w(qh_rel + qh_walk + 1u) & 0x08u) != 0u);
                        dl = dl2;
                    } else {
                        idx = u32(read_w(qs_rel + qs_walk + 3u)) | ((u32(read_w(qh_rel + qh_walk + 1u)) << 4u) & 0x700u);
                        delta = select(0.125, -0.125, (read_w(qh_rel + qh_walk + 1u) & 0x80u) != 0u);
                        dl = dl2;
                    }
                    let grid_off = IQ_GRID_OFF_IQ1S + idx * 8u;
                    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                        let gv = lut_i8(grid_off + j);
                        arena[dst_base + y + j] = dl * (f32(gv) + delta);
                    }
                    y = y + 8u;
                }
                qs_walk = qs_walk + 4u;
                qh_walk = qh_walk + 2u;
            }
            return;
        }
        return;
    }

    // ── 32-elem block schemes: IQ4_NL, Q4_0, Q8_0, MXFP4 ──────────
    if (params.scheme_id == 19u) {
        // Q4_0: f16 d | 16 packed nibbles (18 bytes / 32 elements).
        let off = gid * 18u;
        let d = dq_read_f16(off);
        let qs_rel = off + 2u;
        let dst_base = params.dst_f32_off + gid * 32u;
        for (var j: u32 = 0u; j < 32u / 2u; j = j + 1u) {
            let bx = read_w(qs_rel + j);
            arena[dst_base + j] = d * f32(i32(bx & 0x0Fu) - 8);
            arena[dst_base + j + 32u / 2u] = d * f32(i32(bx >> 4) - 8);
        }
        return;
    }

    if (params.scheme_id == 20u) {
        // Q8_0: f16 d | 32 i8 quants (34 bytes / 32 elements).
        let off = gid * 34u;
        let d = dq_read_f16(off);
        let qs_rel = off + 2u;
        let dst_base = params.dst_f32_off + gid * 32u;
        for (var j: u32 = 0u; j < 32u; j = j + 1u) {
            arena[dst_base + j] = d * f32(read_w_i8(qs_rel + j));
        }
        return;
    }

    if (params.scheme_id == 21u) {
        // Q4_1: f16 d | f16 min | 16 packed nibbles (20 bytes / 32 elements).
        let off = gid * 20u;
        let d = dq_read_f16(off);
        let m = dq_read_f16(off + 2u);
        let qs_rel = off + 4u;
        let dst_base = params.dst_f32_off + gid * 32u;
        for (var j: u32 = 0u; j < 32u / 2u; j = j + 1u) {
            let bx = read_w(qs_rel + j);
            arena[dst_base + j] = d * f32(bx & 0x0Fu) + m;
            arena[dst_base + j + 16u] = d * f32(bx >> 4) + m;
        }
        return;
    }

    if (params.scheme_id == 22u) {
        // Q5_0: f16 d | u32 qh | 16 packed nibbles (22 bytes / 32 elements).
        let off = gid * 22u;
        let d = dq_read_f16(off);
        let qh = u32(read_w(off + 2u)) | (u32(read_w(off + 3u)) << 8u)
            | (u32(read_w(off + 4u)) << 16u) | (u32(read_w(off + 5u)) << 24u);
        let qs_rel = off + 6u;
        let dst_base = params.dst_f32_off + gid * 32u;
        for (var j: u32 = 0u; j < 32u / 2u; j = j + 1u) {
            let bx = read_w(qs_rel + j);
            let xh0 = ((qh >> j) & 0x01u) << 4u;
            let xh1 = ((qh >> (j + 16u)) & 0x01u) << 4u;
            arena[dst_base + j] = d * f32(i32((bx & 0x0Fu) | xh0) - 16);
            arena[dst_base + j + 16u] = d * f32(i32((bx >> 4) | xh1) - 16);
        }
        return;
    }

    if (params.scheme_id == 23u) {
        // Q5_1: f16 d | f16 min | u32 qh | 16 packed nibbles (24 bytes / 32 elements).
        let off = gid * 24u;
        let d = dq_read_f16(off);
        let m = dq_read_f16(off + 2u);
        let qh = u32(read_w(off + 4u)) | (u32(read_w(off + 5u)) << 8u)
            | (u32(read_w(off + 6u)) << 16u) | (u32(read_w(off + 7u)) << 24u);
        let qs_rel = off + 8u;
        let dst_base = params.dst_f32_off + gid * 32u;
        for (var j: u32 = 0u; j < 32u / 2u; j = j + 1u) {
            let bx = read_w(qs_rel + j);
            let xh0 = ((qh >> j) & 0x01u) << 4u;
            let xh1 = ((qh >> (j + 16u)) & 0x01u) << 4u;
            arena[dst_base + j] = d * f32((bx & 0x0Fu) | xh0) + m;
            arena[dst_base + j + 16u] = d * f32((bx >> 4) | xh1) + m;
        }
        return;
    }

    if (params.scheme_id == 6u) {
        // IQ4_NL: 18 bytes / 32 elements.
        let off = gid * 18u;
        let d = dq_read_f16(off);
        let qs_rel = off + 2u;
        let dst_base = params.dst_f32_off + gid * 32u;
        for (var j: u32 = 0u; j < 32u / 2u; j = j + 1u) {
            let bx = read_w(qs_rel + j);
            arena[dst_base + j] = d * f32(kvalues_iq4nl_lut[bx & 0x0Fu]);
            arena[dst_base + j + 16u] = d * f32(kvalues_iq4nl_lut[bx >> 4]);
        }
        return;
    }

    if (params.scheme_id == 10u) {
        // MXFP4: E8M0 scale + 16 nibbles = 17 bytes / 32 elements.
        let off = gid * 17u;
        let e = read_w(off);
        var s: f32;
        if (e == 0xFFu) { s = 0.0; }
        else { s = exp2(f32(i32(u32(e)) - 127)); }
        let qs_rel = off + 1u;
        let dst_base = params.dst_f32_off + gid * 32u;
        for (var j: u32 = 0u; j < 32u / 2u; j = j + 1u) {
            let bx = read_w(qs_rel + j);
            arena[dst_base + 2u * j] = s * FP4_E2M1[bx & 0x0Fu];
            arena[dst_base + 2u * j + 1u] = s * FP4_E2M1[bx >> 4];
        }
        return;
    }

    // ── 16-elem block schemes: NVFP4 ──────────────────────────────
    if (params.scheme_id == 11u) {
        // NVFP4: E4M3 scale + 8 nibbles = 9 bytes / 16 elements.
        let off = gid * 9u;
        let e = read_w(off);
        // Inline E4M3 decode.
        var s: f32;
        let sign_b = (u32(e) >> 7) & 1u;
        let exp_v = (u32(e) >> 3) & 0x0Fu;
        let mant = u32(e) & 0x7u;
        if (exp_v == 0u) {
            s = select((f32(mant) / 8.0) * exp2(-6.0), 0.0, mant == 0u);
        } else if (exp_v == 0x0Fu && mant == 0x7u) {
            s = 0.0;
        } else {
            s = (1.0 + f32(mant) / 8.0) * exp2(f32(i32(exp_v) - 7));
        }
        if (sign_b != 0u) { s = -s; }
        let qs_rel = off + 1u;
        let dst_base = params.dst_f32_off + gid * 16u;
        for (var j: u32 = 0u; j < 16u / 2u; j = j + 1u) {
            let bx = read_w(qs_rel + j);
            arena[dst_base + 2u * j] = s * FP4_E2M1[bx & 0x0Fu];
            arena[dst_base + 2u * j + 1u] = s * FP4_E2M1[bx >> 4];
        }
        return;
    }
}
