// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Fused (residual add + optional bias add) + RMSNorm.
//
//   y = rms_norm(x + residual + [bias], gamma, beta)
//
// One thread per outer row, sequential reduction over the inner dim.
//
// Inputs (offsets in f32 elements):
//   in_off:       [outer, inner]
//   residual_off: [outer, inner]
//   bias_off:     [inner]   (only read when has_bias != 0)
//   gamma_off:    [inner]
//   beta_off:     [inner]
// Output:
//   out_off:      [outer, inner]

struct Params {
    outer: u32,
    inner: u32,
    in_off: u32,
    residual_off: u32,
    bias_off: u32,
    gamma_off: u32,
    beta_off: u32,
    out_off: u32,
    eps_bits: u32,
    has_bias: u32,
    _p0: u32, _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn fused_residual_rms_norm(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let row = gid.x + gid.y * ngs.x * 64u;
    if (row >= params.outer || params.inner == 0u) { return; }
    let in_base  = params.in_off       + row * params.inner;
    let res_base = params.residual_off + row * params.inner;
    let out_base = params.out_off      + row * params.inner;
    let n_inv = epi_div_rn(1.0, f32(params.inner));
    let eps = bitcast<f32>(params.eps_bits);
    let with_bias = params.has_bias != 0u;

    // Pass 1a: fuse residual (+ bias) into the output slot and accumulate the mean.
    var sum_x: f32 = 0.0;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        var v = arena[in_base + i] + arena[res_base + i];
        if (with_bias) { v = v + arena[params.bias_off + i]; }
        arena[out_base + i] = v;
        sum_x = sum_x + v;
    }
    let mean = sum_x * n_inv;
    // Pass 1b: STABLE TWO-PASS mean(x²) = mean((x − mean)²) + mean² (see the RmsNorm
    // note in layernorm.wgsl) — a large per-row DC offset otherwise swamps the
    // one-pass f32 Σx² and drifts vs CPU's f64, corrupting the per-row scale on wgpu.
    var sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        let d = arena[out_base + i] - mean;
        sum_sq = sum_sq + d * d;
    }
    let inv_rms = epi_rsqrt_rn(sum_sq * n_inv + mean * mean + eps);

    // Pass 2: scale by gamma and shift by beta.
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        let g = arena[params.gamma_off + i];
        let b = arena[params.beta_off + i];
        arena[out_base + i] = arena[out_base + i] * inv_rms * g + b;
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
