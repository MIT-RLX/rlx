// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// BatchNormInference and its three backwards, channels-last (`idx = row *
// channels + c`). Mirrors `rlx_cpu::kernels::batch_norm_inference{,_backward_*}`.
//
//   y   = γ · x̂ + β,   x̂ = (x − μ) / sqrt(σ² + ε)
//   dx  = dy · γ / sqrt(σ² + ε)          (μ, σ² are constants at inference)
//   dγ  = Σ_rows dy · x̂
//   dβ  = Σ_rows dy
//
// wgpu was the only f32-uniform arena backend without these — CUDA and ROCm
// share `batch_norm_inference.cu`, Vulkan has four `.comp`. Host-routing them
// costs a full arena readback per BN layer, which in a CNN is dozens per forward
// pass.
//
// **`1 / sqrt(v + eps)`, not `inverseSqrt(v + eps)`.** The CPU oracle computes a
// correctly-rounded `sqrt` then a correctly-rounded divide; `inverseSqrt` is an
// approximation and Metal's fast math approximates the divide too. `inv`
// multiplies every element of its channel, so that error is not local — a bare
// `1.0 / sqrt(x)` measured 1 ULP off the oracle and broke even the pure-multiply
// `bwd_input`. (The Vulkan twins still use `inversesqrt`; left alone here
// because changing their numerics without a parity gate to hold them would be
// the wrong trade.)
//
// What remains after that is a genuine difference, not a gap: Metal contracts
// the multiply-ADD in the forward and in `bwd_gamma` into an `fma`, one rounding
// where the CPU does two. `bwd_input` and `bwd_beta` have no multiply-add and
// come out bit-identical, which is what pins the diagnosis. See
// `rlx-runtime/tests/wgpu_batch_norm_parity.rs` for the per-kernel gates.

struct Params {
    /// Total elements (forward / bwd_input) — `count * channels`.
    n: u32,
    /// Rows (bwd_gamma / bwd_beta).
    count: u32,
    channels: u32,
    eps: f32,
    src_off: u32,
    gamma_off: u32,
    beta_off: u32,
    mean_off: u32,
    var_off: u32,
    dy_off: u32,
    dst_off: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>             params: Params;

// Correctly-rounded a/b — Metal's fast math lowers `/` to a reciprocal
// multiply, up to 1 ULP off. `fma` forms the residual exactly.
fn div_rn(a: f32, b: f32) -> f32 {
    let q = a / b;
    let r = fma(-b, q, a);
    return q + r / b;
}

// One Newton step on `sqrt`, then a correctly-rounded reciprocal.
//
// The oracle computes `1.0 / (var + eps).sqrt()` with an IEEE sqrt and an IEEE
// divide. Under Metal's default fast math neither is exact, and the error lands
// in `inv` — which multiplies every element of the channel, so a bare
// `1.0 / sqrt(x)` came out 1 ULP from the oracle on real data.
fn sqrt_rn(x: f32) -> f32 {
    let y = sqrt(x);
    if (y <= 0.0) { return y; }
    return 0.5 * (y + div_rn(x, y));
}

fn inv_std(c: u32) -> f32 {
    return div_rn(1.0, sqrt_rn(arena[params.var_off + c] + params.eps));
}

fn tid(gid: vec3<u32>, ngs: vec3<u32>) -> u32 {
    return gid.x + gid.y * ngs.x * 64u;
}

@compute @workgroup_size(64)
fn batch_norm_inference(@builtin(global_invocation_id) gid: vec3<u32>,
                        @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = tid(gid, ngs);
    if (i >= params.n || params.channels == 0u) { return; }
    let c = i % params.channels;
    let xhat = (arena[params.src_off + i] - arena[params.mean_off + c]) * inv_std(c);
    arena[params.dst_off + i] =
        arena[params.gamma_off + c] * xhat + arena[params.beta_off + c];
}

@compute @workgroup_size(64)
fn batch_norm_inference_bwd_input(@builtin(global_invocation_id) gid: vec3<u32>,
                                  @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = tid(gid, ngs);
    if (i >= params.n || params.channels == 0u) { return; }
    let c = i % params.channels;
    arena[params.dst_off + i] =
        arena[params.dy_off + i] * arena[params.gamma_off + c] * inv_std(c);
}

// One invocation per channel, accumulating down the rows in the same order the
// CPU loop does — f32 addition is not associative, so a tree reduction here
// would diverge from the oracle even though it is the faster shape.
@compute @workgroup_size(64)
fn batch_norm_inference_bwd_gamma(@builtin(global_invocation_id) gid: vec3<u32>,
                                  @builtin(num_workgroups) ngs: vec3<u32>) {
    let c = tid(gid, ngs);
    if (c >= params.channels) { return; }
    let inv = inv_std(c);
    let mean = arena[params.mean_off + c];
    var acc = 0.0;
    for (var row = 0u; row < params.count; row = row + 1u) {
        let idx = row * params.channels + c;
        let xhat = (arena[params.src_off + idx] - mean) * inv;
        acc = acc + arena[params.dy_off + idx] * xhat;
    }
    arena[params.dst_off + c] = acc;
}

@compute @workgroup_size(64)
fn batch_norm_inference_bwd_beta(@builtin(global_invocation_id) gid: vec3<u32>,
                                 @builtin(num_workgroups) ngs: vec3<u32>) {
    let c = tid(gid, ngs);
    if (c >= params.channels) { return; }
    var acc = 0.0;
    for (var row = 0u; row < params.count; row = row + 1u) {
        acc = acc + arena[params.dy_off + row * params.channels + c];
    }
    arena[params.dst_off + c] = acc;
}
