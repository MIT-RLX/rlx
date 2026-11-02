// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// DiT adaLN-Zero: out = norm(x) * (1 + scale) + shift
//
// scale/shift broadcast over leading dims (typically [B,1,D] over [B,S,D]).
// `layer_norm != 0` → mean-subtract LayerNorm; else RMSNorm only.
// `lead_pack`: [lead_rank, x_lead[8], mod_lead[8]] as 20 uints in 5×vec4
// (uniform arrays must have stride multiple of 16).

struct Params {
    outer: u32,
    inner: u32,
    in_off: u32,
    scale_off: u32,
    shift_off: u32,
    out_off: u32,
    eps_bits: u32,
    layer_norm: u32,
    lead_pack: array<vec4<u32>, 5>,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

fn lead_at(i: u32) -> u32 {
    return params.lead_pack[i / 4u][i % 4u];
}

fn mod_base_for_row(row: u32, inner: u32) -> u32 {
    let lead_rank = lead_at(0u);
    var rem = row;
    var mod_base: u32 = 0u;
    var mod_stride: u32 = inner;
    var j: i32 = i32(lead_rank) - 1;
    loop {
        if (j < 0) { break; }
        var xd = lead_at(1u + u32(j));
        if (xd == 0u) { xd = 1u; }
        let xi = rem % xd;
        rem = rem / xd;
        var md = lead_at(9u + u32(j));
        if (md == 0u) { md = 1u; }
        if (md != 1u) {
            mod_base += xi * mod_stride;
        }
        mod_stride = mod_stride * md;
        j = j - 1;
    }
    return mod_base;
}

@compute @workgroup_size(64)
fn ada_layer_norm(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let row = gid.x + gid.y * ngs.x * 64u;
    if (row >= params.outer || params.inner == 0u) { return; }
    let in_base = params.in_off + row * params.inner;
    let out_base = params.out_off + row * params.inner;
    let mod_base = mod_base_for_row(row, params.inner);
    let n_inv = epi_div_rn(1.0, f32(params.inner));
    let eps = bitcast<f32>(params.eps_bits);
    let do_ln = params.layer_norm != 0u;

    // Kahan sums — F5 DiT chains 44 AdaLN ops per step; f32 reduction noise
    // compounds across the NFE ODE on wgpu.
    var sum_x: f32 = 0.0;
    var c_x: f32 = 0.0;
    var sum_x2: f32 = 0.0;
    var c_x2: f32 = 0.0;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        let v = arena[in_base + i];
        let yx = v - c_x;
        let tx = sum_x + yx;
        c_x = (tx - sum_x) - yx;
        sum_x = tx;
        let y2 = v * v - c_x2;
        let t2 = sum_x2 + y2;
        c_x2 = (t2 - sum_x2) - y2;
        sum_x2 = t2;
    }

    var mean: f32 = 0.0;
    var inv: f32;
    if (do_ln) {
        // STABLE TWO-PASS variance = mean((x − mean)²). The one-pass
        // E[x²] − (E[x])² identity catastrophically cancels in f32 under a
        // large DC offset (pre-norm transformer activations) and corrupts
        // the norm on wgpu only; two-pass matches CPU/Metal/MLX/CoreML.
        mean = sum_x * n_inv;
        var sum_sq: f32 = 0.0;
        var c_sq: f32 = 0.0;
        for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
            let d = arena[in_base + i] - mean;
            let y = d * d - c_sq;
            let t = sum_sq + y;
            c_sq = (t - sum_sq) - y;
            sum_sq = t;
        }
        inv = epi_rsqrt_rn(sum_sq * n_inv + eps);
    } else {
        inv = epi_rsqrt_rn(sum_x2 * n_inv + eps);
    }

    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        let n = (arena[in_base + i] - mean) * inv;
        let s = arena[params.scale_off + mod_base + i];
        let t = arena[params.shift_off + mod_base + i];
        arena[out_base + i] = n * (1.0 + s) + t;
    }
}

// `rlxsl::exact`'s correctly-rounded helpers, verbatim (`rlx_*` → `epi_*`): the
// builtin `/`, `inverseSqrt` and `exp` round by the compiler's math mode, so
// native wgpu (Apple fast math) and a browser (safe math) disagreed.
fn epi_div_rn(a: f32, b: f32) -> f32 {
    let k = abs(a / b) < 0x1p-100f;
    let a2 = select(a, a * 0x1p64f, k);
    let q = a2 / b;
    let ok = q != 0.0 && abs(q) <= 0x1.fffffep+127f;
    var c = select(q, q + fma(-b, q, a2) / b, ok);
    // The step rounds its correction before adding it, so near a midpoint
    // `c` depends on the first quotient. Move to the neighbour on the
    // quotient's side when the quotient lies past their midpoint (ties to
    // even): `r = a - b·c` is exact, and `r - b·u/2` has the sign of
    // `a - b·(c + n)/2`.
    if (ok && c != 0.0 && abs(c) < 0x1.fffffep+127f) {
        let r = fma(-b, c, a2);
        if (r != 0.0) {
            let up = (r > 0.0) == (b > 0.0);
            let cb = bitcast<u32>(c);
            let nb = select(cb - 1u, cb + 1u, up == (c > 0.0));
            let n = bitcast<f32>(nb);
            let t = fma(-b, (n - c) * 0.5, r);
            if ((t != 0.0 && ((t > 0.0) == (b > 0.0)) == up) || (t == 0.0 && (nb & 1u) == 0u)) {
                c = n;
            }
        }
    }
    return select(c, c * 0x1p-64f, k);
}

fn epi_sqrt_rn(x: f32) -> f32 {
    let bx = bitcast<u32>(x);
    if ((bx & 0x7fffffffu) == 0u) { return x; }
    if (bx >= 0x7f800000u) {
        // +inf stays; negatives and NaN are NaN. The NaN is built from `bx`:
        // a constant NaN (or inf) is a shader-creation error in Chrome.
        return select(bitcast<f32>((bx & 0u) | 0x7fc00000u), x, bx == 0x7f800000u);
    }
    var s = x;
    var back = 1.0;
    if (x < 0x1p-60f) {
        s = x * 0x1p100f;
        back = 0x1p-50f;
    } else if (x > 0x1p60f) {
        s = x * 0x1p-100f;
        back = 0x1p50f;
    }
    var y = sqrt(s);
    for (var i = 0u; i < 4u; i++) {
        let r = fma(-y, y, s);
        if (r < 0.0) {
            let lo = bitcast<f32>(bitcast<u32>(y) - 1u);
            let u = y - lo;
            if (fma(-lo, lo, s) + r + 0.5 * u * u < 0.0) { y = lo; } else { break; }
        } else {
            let hi = bitcast<f32>(bitcast<u32>(y) + 1u);
            let u = hi - y;
            if (fma(-hi, hi, s) + r + 0.5 * u * u > 0.0) { y = hi; } else { break; }
        }
    }
    return y * back;
}

fn epi_rsqrt_rn(x: f32) -> f32 {
    return epi_div_rn(1.0, epi_sqrt_rn(x));
}
