// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// RMSNorm backward (row = last-axis slice). wrt: 0=dx, 1=dgamma, 2=dbeta.

struct Params {
    outer: u32,
    inner: u32,
    x_off: u32,
    gamma_off: u32,
    beta_off: u32,
    dy_off: u32,
    out_off: u32,
    eps_bits: u32,
    wrt: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(1)
fn rms_norm_bwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.wrt != 0u || params.inner == 0u) { return; }
    let row = gid.x;
    if (row >= params.outer) { return; }
    let x_base = params.x_off + row * params.inner;
    let dy_base = params.dy_off + row * params.inner;
    let out_base = params.out_off + row * params.inner;
    let n_inv = epi_div_rn(1.0, f32(params.inner));
    let eps = bitcast<f32>(params.eps_bits);

    var dot: f32 = 0.0;
    var sumsq: f32 = 0.0;
    for (var i: u32 = 0u; i < params.inner; i++) {
        let xv = arena[x_base + i];
        let gv = arena[params.gamma_off + i];
        let dyv = arena[dy_base + i];
        dot = dot + dyv * gv * xv;
        sumsq = sumsq + xv * xv;
    }
    dot = dot * n_inv;
    let inv_r = epi_rsqrt_rn(sumsq * n_inv + eps);
    // Cross term is inv_r³ (inv_r2 here, ·inv_r below), not inv_r⁴ — drop the stray 1/r.
    let inv_r2 = inv_r * inv_r;
    for (var i: u32 = 0u; i < params.inner; i++) {
        let xv = arena[x_base + i];
        let gv = arena[params.gamma_off + i];
        let dyv = arena[dy_base + i];
        let term = gv * dyv - xv * dot * inv_r2;
        arena[out_base + i] = term * inv_r;
    }
}

@compute @workgroup_size(1)
fn rms_norm_bwd_param(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u || params.inner == 0u) { return; }
    let eps = bitcast<f32>(params.eps_bits);
    let n_inv = epi_div_rn(1.0, f32(params.inner));
    for (var i: u32 = 0u; i < params.inner; i++) {
        arena[params.out_off + i] = 0.0;
    }
    for (var row: u32 = 0u; row < params.outer; row++) {
        let x_base = params.x_off + row * params.inner;
        let dy_base = params.dy_off + row * params.inner;
        var sumsq: f32 = 0.0;
        for (var i: u32 = 0u; i < params.inner; i++) {
            let xv = arena[x_base + i];
            sumsq = sumsq + xv * xv;
        }
        let inv_r = epi_rsqrt_rn(sumsq * n_inv + eps);
        if (params.wrt == 1u) {
            for (var i: u32 = 0u; i < params.inner; i++) {
                let xv = arena[x_base + i];
                let dyv = arena[dy_base + i];
                arena[params.out_off + i] = arena[params.out_off + i] + dyv * xv * inv_r;
            }
        } else {
            for (var i: u32 = 0u; i < params.inner; i++) {
                let dyv = arena[dy_base + i];
                arena[params.out_off + i] = arena[params.out_off + i] + dyv;
            }
        }
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
