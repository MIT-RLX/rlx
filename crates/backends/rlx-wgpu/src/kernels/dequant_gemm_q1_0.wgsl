// RLX — fused Q1_0 GEMM (prefill m>1) for `Op::DequantMatMul`.
//
// Mirrors Metal `q1_0_mm_f32`: TM=8 output rows per thread, one thread per
// (column, row-tile). Reads packed 1-bit weights directly (no f32 scratch).
// Binding scheme matches `dequant_gemv_gguf.wgsl` (windowed x/w + separate out).

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

const TM: u32 = 8u;
const BLK_BYTES: u32 = 18u;

fn read_w(rel: u32) -> u32 {
    let abs = params.w_byte_off + rel;
    let word = abs / 4u;
    let shift = (abs % 4u) * 8u;
    return (warr[word] >> shift) & 0xffu;
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

@compute @workgroup_size(64)
fn dequant_gemm_q1_0(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row0 = gid.y * TM;
    if (col >= params.n || row0 >= params.m) { return; }

    let nblocks = params.k / 128u;
    let xb = params.x_f32_off;
    let w_row = col * nblocks * BLK_BYTES;

    var acc: array<f32, 8>;
    for (var r: u32 = 0u; r < TM; r = r + 1u) { acc[r] = 0.0; }

    var p: u32 = 0u;
    for (var b: u32 = 0u; b < nblocks; b = b + 1u) {
        let off = w_row + b * BLK_BYTES;
        let d = dq_read_f16(off);
        let nd = -d;
        for (var byte: u32 = 0u; byte < 16u; byte = byte + 1u) {
            let bits = read_w(off + 2u + byte);
            var ww: array<f32, 8>;
            ww[0] = select(nd, d, (bits & 1u) != 0u);
            ww[1] = select(nd, d, (bits & 2u) != 0u);
            ww[2] = select(nd, d, (bits & 4u) != 0u);
            ww[3] = select(nd, d, (bits & 8u) != 0u);
            ww[4] = select(nd, d, (bits & 16u) != 0u);
            ww[5] = select(nd, d, (bits & 32u) != 0u);
            ww[6] = select(nd, d, (bits & 64u) != 0u);
            ww[7] = select(nd, d, (bits & 128u) != 0u);
            for (var r: u32 = 0u; r < TM; r = r + 1u) {
                let rr = row0 + r;
                let rc = min(rr, params.m - 1u);
                let xr = xb + rc * params.k + p;
                let s = xarr[xr] * ww[0] + xarr[xr + 1u] * ww[1]
                      + xarr[xr + 2u] * ww[2] + xarr[xr + 3u] * ww[3]
                      + xarr[xr + 4u] * ww[4] + xarr[xr + 5u] * ww[5]
                      + xarr[xr + 6u] * ww[6] + xarr[xr + 7u] * ww[7];
                if (rr < params.m) { acc[r] = acc[r] + s; }
            }
            p = p + 8u;
        }
    }

    let ob = params.out_f32_off;
    for (var r: u32 = 0u; r < TM; r = r + 1u) {
        let rr = row0 + r;
        if (rr < params.m) {
            outarr[ob + rr * params.n + col] = acc[r];
        }
    }
}
