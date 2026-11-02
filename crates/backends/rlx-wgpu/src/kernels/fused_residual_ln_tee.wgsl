// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// FusedResidualLN / FusedResidualRmsNorm with a "tee" output (`is_rms`): writes BOTH (h + residual + [bias])
// AND LN(h + residual + [bias]) to two separate arena slots. Used when
// the sum has multiple consumers downstream (the LN AND a later residual
// add) and the regular `fused_residual_ln` can't fire because of its
// single-consumer guard.
//
// Replaces the (Add → LN) sequence with one dispatch when Add is multi-
// consumer. Vision pre-norm transformers hit this 23× per forward
// (every residual feeds both the next LN and the *next-next* residual);
// w/o this kernel the wgpu schedule has 35 standalone Binary + 36
// standalone LayerNorm. With this kernel each (Add → LN) collapses to
// one FusedResidualLnTee.
//
// Memory layout:
//   sum_off:      [outer, inner]   ← (h + residual + [bias])
//   ln_out_off:   [outer, inner]   ← LN of the sum
// Other consumers of the sum read sum_off (= the eliminated Add's old
// arena slot, which the planner has already allocated for them).

struct Params {
    outer: u32,
    inner: u32,
    in_off: u32,
    residual_off: u32,
    bias_off: u32,
    gamma_off: u32,
    beta_off: u32,
    sum_off: u32,
    ln_out_off: u32,
    eps_bits: u32,
    has_bias: u32,
    is_rms: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

// One 64-thread workgroup per row (as `layernorm.wgsl`): a decode step has a
// single row, and one thread walking it three times serially was ~1 ms.
var<workgroup> scratch: array<f32, 64>;

fn tree_sum(tid: u32) {
    for (var stride: u32 = 32u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            scratch[tid] = scratch[tid] + scratch[tid + stride];
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(64)
fn fused_residual_ln_tee(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let row = wid.x + wid.y * ngs.x;
    if (row >= params.outer || params.inner == 0u) { return; }
    let tid = lid.x;
    let in_base  = params.in_off       + row * params.inner;
    let res_base = params.residual_off + row * params.inner;
    let sum_base = params.sum_off      + row * params.inner;
    let out_base = params.ln_out_off   + row * params.inner;
    let n_inv = epi_div_rn(1.0, f32(params.inner));
    let eps = bitcast<f32>(params.eps_bits);
    let with_bias = params.has_bias != 0u;

    // Pass 1: write the SUM to sum_base and accumulate the mean. Each thread
    // owns elements tid, tid+64, …, and re-reads only its own writes below.
    var partial: f32 = 0.0;
    for (var i: u32 = tid; i < params.inner; i = i + 64u) {
        var v = arena[in_base + i] + arena[res_base + i];
        if (with_bias) { v = v + arena[params.bias_off + i]; }
        arena[sum_base + i] = v;
        partial = partial + v;
    }
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let mean = scratch[0] * n_inv;
    workgroupBarrier(); // everyone has read scratch[0] before it is reused.

    // Pass 1b: STABLE TWO-PASS variance = mean((x − mean)²). The one-pass
    // E[x²] − (E[x])² identity catastrophically cancels in f32 under a large
    // DC offset (pre-norm transformer activations); two-pass matches
    // CPU/Metal/MLX/CoreML.
    partial = 0.0;
    for (var i: u32 = tid; i < params.inner; i = i + 64u) {
        let d = arena[sum_base + i] - mean;
        partial = partial + d * d;
    }
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let var_ = scratch[0] * n_inv;
    // RMS norm scales by 1/sqrt(mean(x^2)) and does NOT recentre:
    // mean(x^2) = mean((x-mean)^2) + mean^2, the same stable accumulation.
    let rms_only = params.is_rms != 0u;
    let inv_scale = select(
        epi_rsqrt_rn(var_ + eps),
        epi_rsqrt_rn(var_ + mean * mean + eps),
        rms_only,
    );

    // Pass 2: read sum_base, write the norm result to out_base. Two distinct
    // slots — the sum stays available for the other consumer.
    for (var i: u32 = tid; i < params.inner; i = i + 64u) {
        let g = arena[params.gamma_off + i];
        let b = arena[params.beta_off  + i];
        let v = arena[sum_base + i];
        let centred = select(v - mean, v, rms_only);
        arena[out_base + i] = centred * inv_scale * g + b;
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
