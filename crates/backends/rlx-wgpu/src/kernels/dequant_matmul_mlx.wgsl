// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// GPL-3.0-only. MLX affine / mxfp4 / mxfp8 fused dequant-matmul.
// Weight layout [n, k] packed along K (row j → output column j).
// kind: 0=affine, 1=mxfp4, 2=mxfp8.

struct Params {
    m: u32,
    k: u32,
    n: u32,
    kind: u32,
    bits: u32,
    group_size: u32,
    x_byte_off: u32,
    w_byte_off: u32,
    scale_byte_off: u32,
    zp_byte_off: u32,
    out_byte_off: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

fn rd_byte(byte_off: u32) -> u32 {
    let word = byte_off / 4u;
    let shift = (byte_off % 4u) * 8u;
    return (bitcast<u32>(arena[word]) >> shift) & 0xffu;
}

const FP4_E2M1: array<f32, 16> = array<f32, 16>(
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
);

fn decode_e4m3(byte: u32) -> f32 {
    let sign = (byte >> 7u) & 1u;
    let exp = i32((byte >> 3u) & 0xfu);
    let mant = byte & 0x7u;
    if (exp == 0x0f && mant == 0x7u) {
        // Match host `dequant_scale_fp8_e4m3` — quiet NaN.
        return bitcast<f32>(0x7fc00000u);
    }
    if (exp == 0) {
        if (mant == 0u) {
            return select(0.0, -0.0, sign != 0u);
        }
        var m = mant;
        var e = -6;
        while ((m & 0x8u) == 0u) {
            m = m << 1u;
            e = e - 1;
        }
        m = m & 0x7u;
        let bits = (sign << 31u) | (u32(e + 127) << 23u) | (m << 20u);
        return bitcast<f32>(bits);
    }
    let bits = (sign << 31u) | (u32(exp - 7 + 127) << 23u) | (mant << 20u);
    return bitcast<f32>(bits);
}

fn decode_e8m0(s: u32) -> f32 {
    if (s == 0u) {
        return bitcast<f32>(0x0040u << 16u);
    }
    return bitcast<f32>(s << 23u);
}

fn group_scale(s: u32, gs: u32) -> f32 {
    return select(decode_e8m0(s), decode_e4m3(s), gs == 16u);
}

fn pack_factor(bits: u32) -> u32 {
    if (bits == 2u || bits == 4u || bits == 8u) { return 8u / bits; }
    if (bits == 3u || bits == 5u) { return 8u; }
    if (bits == 6u) { return 4u; }
    return 1u;
}

fn bytes_per_pack(bits: u32) -> u32 {
    if (bits == 2u || bits == 4u || bits == 8u) { return 1u; }
    if (bits == 3u || bits == 6u) { return 3u; }
    if (bits == 5u) { return 5u; }
    return 1u;
}

fn affine_code(bits: u32, gs: u32, n_groups: u32, j: u32, p: u32) -> f32 {
    let pf = pack_factor(bits);
    let bpp = bytes_per_pack(bits);
    let packs_in_group = gs / pf;
    let g = p / gs;
    let local = p % gs;
    let row_base = j * n_groups * packs_in_group * bpp + g * packs_in_group * bpp;
    var code: u32 = 0u;
    if (bits == 2u || bits == 4u || bits == 8u) {
        let pack_idx = local / pf;
        let in_pack = local % pf;
        let byte = rd_byte(params.w_byte_off + row_base + pack_idx);
        let mask = (1u << bits) - 1u;
        code = (byte >> (in_pack * bits)) & mask;
    } else if (bits == 3u) {
        let pack_idx = local / 8u;
        let in_pack = local % 8u;
        let bo = params.w_byte_off + row_base + pack_idx * 3u;
        let b0 = rd_byte(bo);
        let b1 = rd_byte(bo + 1u);
        let b2 = rd_byte(bo + 2u);
        let codes = array<u32, 8>(
            b0 & 0x7u,
            (b0 & 0x38u) >> 3u,
            ((b0 & 0xc0u) >> 6u) + ((b1 & 0x1u) << 2u),
            (b1 & 0xeu) >> 1u,
            (b1 & 0x70u) >> 4u,
            ((b1 & 0x80u) >> 7u) + ((b2 & 0x3u) << 1u),
            (b2 & 0x1cu) >> 2u,
            (b2 & 0xe0u) >> 5u,
        );
        code = codes[in_pack];
    } else if (bits == 5u) {
        let pack_idx = local / 8u;
        let in_pack = local % 8u;
        let bo = params.w_byte_off + row_base + pack_idx * 5u;
        let b0 = rd_byte(bo);
        let b1 = rd_byte(bo + 1u);
        let b2 = rd_byte(bo + 2u);
        let b3 = rd_byte(bo + 3u);
        let b4 = rd_byte(bo + 4u);
        let codes = array<u32, 8>(
            b0 & 0x1fu,
            ((b0 & 0xe0u) >> 5u) + ((b1 & 0x3u) << 3u),
            (b1 & 0x7cu) >> 2u,
            ((b1 & 0x80u) >> 7u) + ((b2 & 0xfu) << 1u),
            ((b2 & 0xf0u) >> 4u) + ((b3 & 0x1u) << 4u),
            (b3 & 0x3eu) >> 1u,
            ((b3 & 0xc0u) >> 6u) + ((b4 & 0x7u) << 2u),
            (b4 & 0xf8u) >> 3u,
        );
        code = codes[in_pack];
    } else {
        let pack_idx = local / 4u;
        let in_pack = local % 4u;
        let bo = params.w_byte_off + row_base + pack_idx * 3u;
        let b0 = rd_byte(bo);
        let b1 = rd_byte(bo + 1u);
        let b2 = rd_byte(bo + 2u);
        let codes = array<u32, 4>(
            b0 & 0x3fu,
            ((b0 >> 6u) & 0x03u) + ((b1 & 0x0fu) << 2u),
            ((b1 >> 4u) & 0x0fu) + ((b2 & 0x03u) << 4u),
            (b2 >> 2u) & 0x3fu,
        );
        code = codes[in_pack];
    }
    return f32(code);
}

// Prefill / decode: one workgroup per (col, row_tile); threads split K and
// stage an X tile in workgroup memory (TM=8 × 256).
const TM: u32 = 8u;

var<workgroup> xs: array<f32, 8 * 256>;
var<workgroup> smem: array<f32, 8 * 256>;

@compute @workgroup_size(256, 1, 1)
fn dequant_matmul_mlx(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tpg = 256u;
    let n_row_tiles = (params.m + TM - 1u) / TM;
    let tg = wid.x;
    let col = tg / n_row_tiles;
    let row0 = (tg - col * n_row_tiles) * TM;
    if (col >= params.n) { return; }

    let gs = params.group_size;
    let n_groups = params.k / gs;
    let x_off = params.x_byte_off / 4u;
    let out_off = params.out_byte_off / 4u;
    let scale_f = params.scale_byte_off / 4u;
    let zp_f = params.zp_byte_off / 4u;

    var acc: array<f32, 8>;
    for (var t: u32 = 0u; t < TM; t = t + 1u) { acc[t] = 0.0; }

    for (var p0: u32 = 0u; p0 < params.k; p0 = p0 + tpg) {
        let p = p0 + tid;
        for (var t: u32 = 0u; t < TM; t = t + 1u) {
            let row = row0 + t;
            var v = 0.0;
            if (row < params.m && p < params.k) {
                v = arena[x_off + row * params.k + p];
            }
            xs[t * tpg + tid] = v;
        }
        workgroupBarrier();

        if (p < params.k) {
            let g = p / gs;
            var w_dq: f32;
            if (params.kind == 0u) {
                let code = affine_code(params.bits, gs, n_groups, col, p);
                let s = arena[scale_f + col * n_groups + g];
                let b = arena[zp_f + col * n_groups + g];
                w_dq = s * code + b;
            } else if (params.kind == 1u) {
                let bidx = col * (params.k / 2u) + (p / 2u);
                let byte = rd_byte(params.w_byte_off + bidx);
                let nib = select(byte >> 4u, byte & 0x0fu, (p & 1u) == 0u);
                let sb = rd_byte(params.scale_byte_off + col * n_groups + g);
                w_dq = FP4_E2M1[nib] * group_scale(sb, gs);
            } else {
                let bidx = col * params.k + p;
                let wb = rd_byte(params.w_byte_off + bidx);
                let sb = rd_byte(params.scale_byte_off + col * n_groups + g);
                w_dq = decode_e4m3(wb) * group_scale(sb, gs);
            }
            for (var t: u32 = 0u; t < TM; t = t + 1u) {
                acc[t] = acc[t] + xs[t * tpg + tid] * w_dq;
            }
        }
        workgroupBarrier();
    }

    for (var t: u32 = 0u; t < TM; t = t + 1u) {
        smem[t * tpg + tid] = acc[t];
    }
    workgroupBarrier();
    var s = tpg >> 1u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) {
            for (var t: u32 = 0u; t < TM; t = t + 1u) {
                smem[t * tpg + tid] = smem[t * tpg + tid] + smem[t * tpg + tid + s];
            }
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (tid == 0u) {
        for (var t: u32 = 0u; t < TM; t = t + 1u) {
            let row = row0 + t;
            if (row < params.m) {
                arena[out_off + row * params.n + col] = smem[t * tpg];
            }
        }
    }
}
