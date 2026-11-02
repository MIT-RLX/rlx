// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// LayerNorm and RmsNorm fused into one kernel via op flag. Both
// reduce along the last axis (feature dim).
//
//   LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
//   RmsNorm:   y = x / sqrt(mean(x^2) + eps) * gamma
//
// ONE WORKGROUP PER ROW (64 threads), shared-memory tree reduction over the
// feature dim — replaces the prior scalar one-thread-per-row loop. Both the
// mean and the deviation sum use the STABLE TWO-PASS form (see below); a tree
// sum is at least as accurate as the sequential f32 sum it replaces, so this
// keeps wgpu matching CPU/Metal/MLX/CoreML/CUDA.
//
// Inputs (offsets in f32 elements):
//   in_off:    [outer, inner]
//   gamma_off: [inner]
//   beta_off:  [inner]   (LayerNorm only; RmsNorm ignores)
// Output:
//   out_off:   [outer, inner]

struct Params {
    outer: u32,
    inner: u32,
    in_off: u32,
    out_off: u32,
    gamma_off: u32,
    beta_off: u32,
    eps_bits: u32,    // bitcast-encoded f32 eps
    op: u32,          // 0=LayerNorm, 1=RmsNorm
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

var<workgroup> scratch: array<f32, 64>;

// Sum-reduce `scratch[0..64]` into `scratch[0]`. Caller must barrier before.
fn tree_sum(tid: u32) {
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            scratch[tid] = scratch[tid] + scratch[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
}

@compute @workgroup_size(64)
fn norm(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let row = wid.x + wid.y * ngs.x;
    if (row >= params.outer || params.inner == 0u) { return; }
    let tid = lid.x;
    let in_base  = params.in_off  + row * params.inner;
    let out_base = params.out_off + row * params.inner;
    let n_inv = epi_div_rn(1.0, f32(params.inner));
    let eps = bitcast<f32>(params.eps_bits);

    // ── Pass 1: mean = Σx / n (tree). ──
    var partial: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= params.inner) { break; }
        partial = partial + arena[in_base + i];
        i = i + 64u;
    }
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let mean = scratch[0] * n_inv;
    workgroupBarrier(); // all threads read scratch[0] before Pass 2 overwrites it.

    // ── Pass 2: mean of squared deviation (tree). ──
    // Both LayerNorm var and RmsNorm mean(x²) use the STABLE TWO-PASS form:
    // subtracting `mean` first avoids the catastrophic f32 cancellation of the
    // one-pass `E[x²] − E[x]²` on rows with a large DC offset (pre-norm residual
    // streams: DINOv2/BEiT/ViT, input-projection bias, positional tables).
    partial = 0.0;
    i = tid;
    loop {
        if (i >= params.inner) { break; }
        let d = arena[in_base + i] - mean;
        partial = partial + d * d;
        i = i + 64u;
    }
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let dev_mean = scratch[0] * n_inv; // mean((x − mean)²)

    // ── Pass 3: normalize (each thread strides over its lanes). ──
    if (params.op == 0u) {
        let inv_std = epi_rsqrt_rn(dev_mean + eps);
        var j: u32 = tid;
        loop {
            if (j >= params.inner) { break; }
            let g = arena[params.gamma_off + j];
            let b = arena[params.beta_off + j];
            arena[out_base + j] = (arena[in_base + j] - mean) * inv_std * g + b;
            j = j + 64u;
        }
    } else {
        // mean(x²) = mean((x − mean)²) + mean². `+ beta` matches the CPU oracle
        // (RmsNorm carries beta; the lower now passes the real beta input).
        let inv_rms = epi_rsqrt_rn(dev_mean + mean * mean + eps);
        var j: u32 = tid;
        loop {
            if (j >= params.inner) { break; }
            let g = arena[params.gamma_off + j];
            let b = arena[params.beta_off + j];
            arena[out_base + j] = arena[in_base + j] * inv_rms * g + b;
            j = j + 64u;
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
