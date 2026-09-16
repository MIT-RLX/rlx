// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// GroupNorm on NCHW. Replaces the `GroupNormHost` staging step, which mirrored
// the WHOLE arena to the CPU and back for every norm — on a 35-norm MobileNet
// that is gigabytes of round-trip per forward.
//
//   y = (x - mean_g) / sqrt(var_g + eps) * gamma[c] + beta[c]
//
// where the statistics are taken over one group: `(C / num_groups) * H * W`
// elements, per batch item. `num_groups == C` is instance norm (each group is
// a single channel's H×W plane) and `num_groups == 1` is layer-norm-over-CHW;
// both fall out of the same indexing.
//
// ONE WORKGROUP PER (batch, group), 64 threads, shared-memory tree reduction —
// the same shape as `layernorm.wgsl`, including its STABLE TWO-PASS variance:
// subtracting the mean before squaring avoids the f32 cancellation that
// `E[x²] − E[x]²` suffers on feature maps with a large DC offset. That keeps
// wgpu matching the CPU oracle rather than merely being close.
//
// Note gamma/beta are indexed per CHANNEL, not per group, so the write-back
// pass recovers the channel from the element index.
//
// Offsets are in f32 elements.

struct Params {
    groups_total: u32,  // n * num_groups  (one workgroup each)
    group_elems: u32,   // (c / num_groups) * hw
    in_off: u32,
    out_off: u32,
    gamma_off: u32,
    beta_off: u32,
    eps_bits: u32,      // bitcast-encoded f32 eps
    num_groups: u32,
    c: u32,
    hw: u32,            // h * w
    _pad0: u32,
    _pad1: u32,
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
fn group_norm(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let grp = wid.x + wid.y * ngs.x;
    if (grp >= params.groups_total || params.group_elems == 0u) { return; }
    let tid = lid.x;

    let ni = grp / params.num_groups;          // batch index
    let gi = grp - ni * params.num_groups;     // group index within the batch
    let cpg = params.c / params.num_groups;    // channels per group
    let plane = params.c * params.hw;          // elements per batch item
    let span = cpg * params.hw;                // == params.group_elems
    let in_base = params.in_off + ni * plane + gi * span;
    let out_base = params.out_off + ni * plane + gi * span;
    let n_inv = 1.0 / f32(span);
    let eps = bitcast<f32>(params.eps_bits);

    // ── Pass 1: mean over the group. ──
    var partial: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= span) { break; }
        partial = partial + arena[in_base + i];
        i = i + 64u;
    }
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let mean = scratch[0] * n_inv;
    workgroupBarrier(); // every thread reads scratch[0] before pass 2 reuses it.

    // ── Pass 2: mean squared deviation (stable two-pass). ──
    partial = 0.0;
    i = tid;
    loop {
        if (i >= span) { break; }
        let d = arena[in_base + i] - mean;
        partial = partial + d * d;
        i = i + 64u;
    }
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let inv_std = inverseSqrt(scratch[0] * n_inv + eps);
    workgroupBarrier();

    // ── Pass 3: scale/shift per channel. ──
    var j: u32 = tid;
    loop {
        if (j >= span) { break; }
        let ch = gi * cpg + j / params.hw;
        let g = arena[params.gamma_off + ch];
        let b = arena[params.beta_off + ch];
        arena[out_base + j] = (arena[in_base + j] - mean) * inv_std * g + b;
        j = j + 64u;
    }
}
