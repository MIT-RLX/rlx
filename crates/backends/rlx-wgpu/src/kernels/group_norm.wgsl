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
// ONE WORKGROUP PER (batch, group), 256 threads, shared-memory tree reduction —
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
    act: u32,           // 0 none, else opcode_relu_first + 1 (`fold_norm_activation`)
    chunks: u32,        // > 1: the split kernels below, this many per group
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

// 256 threads, not 64: an instance norm over a volume (15 groups of 6M
// voxels) is a handful of workgroups streaming hundreds of MB each, and at
// 64 threads each was latency-bound — 270 ms a norm on MindGrab, 41% of its
// GPU time. WebGPU's default limit is 256 invocations per workgroup. The
// reductions also keep four loads in flight per thread: 270 ms → 49 ms →
// 27 ms a norm over the two changes.
const WG: u32 = 256u;
var<workgroup> scratch: array<f32, 256>;

// Half-packed activations (`unfuse::pack_half_activations`; the pipeline's
// overrides): input and output two f16 to an arena word. Loads widen, the
// write-back rounds pairs (a group starts on a word: `h·w` is even); the
// statistics stay f32.
override IN_HALF: bool = false;
override OUT_HALF: bool = false;

fn ld(i: u32) -> f32 {
    if (IN_HALF) {
        let e = i - params.in_off;
        return unpack2x16float(bitcast<u32>(arena[params.in_off + (e >> 1u)]))[e & 1u];
    }
    return arena[i];
}

// output elements `o` (even within the tensor) and `o + 1`
fn put2(o: u32, a: f32, b: f32) {
    let e = o - params.out_off;
    put(params.out_off + (e >> 1u), bitcast<f32>(pack2x16float(vec2<f32>(a, b))));
}

// The activation that read this norm, applied in its write-back with the
// unary kernel's generated `rlx_activation_apply`: bit-identical to the
// separate pass it replaces.
fn norm_act(v: f32) -> f32 {
    if (params.act == 0u) { return v; }
    return rlx_activation_apply(params.act - 1u, v);
}

// Sum-reduce `scratch[0..WG]` into `scratch[0]`. Caller must barrier before.
fn tree_sum(tid: u32) {
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            scratch[tid] = scratch[tid] + scratch[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
}

@compute @workgroup_size(256)
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
    // four independent loads in flight per thread (`p0..p3`), then the tail
    var p0: f32 = 0.0; var p1: f32 = 0.0; var p2: f32 = 0.0; var p3: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i + 3u * WG >= span) { break; }
        p0 = p0 + ld(in_base + i);
        p1 = p1 + ld(in_base + i + WG);
        p2 = p2 + ld(in_base + i + 2u * WG);
        p3 = p3 + ld(in_base + i + 3u * WG);
        i = i + 4u * WG;
    }
    loop {
        if (i >= span) { break; }
        p0 = p0 + ld(in_base + i);
        i = i + WG;
    }
    var partial: f32 = (p0 + p1) + (p2 + p3);
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let mean = scratch[0] * n_inv;
    workgroupBarrier(); // every thread reads scratch[0] before pass 2 reuses it.

    // ── Pass 2: mean squared deviation (stable two-pass). ──
    p0 = 0.0; p1 = 0.0; p2 = 0.0; p3 = 0.0;
    i = tid;
    loop {
        if (i + 3u * WG >= span) { break; }
        let d0 = ld(in_base + i) - mean;
        let d1 = ld(in_base + i + WG) - mean;
        let d2 = ld(in_base + i + 2u * WG) - mean;
        let d3 = ld(in_base + i + 3u * WG) - mean;
        p0 = p0 + d0 * d0; p1 = p1 + d1 * d1; p2 = p2 + d2 * d2; p3 = p3 + d3 * d3;
        i = i + 4u * WG;
    }
    loop {
        if (i >= span) { break; }
        let d = ld(in_base + i) - mean;
        p0 = p0 + d * d;
        i = i + WG;
    }
    partial = (p0 + p1) + (p2 + p3);
    scratch[tid] = partial;
    workgroupBarrier();
    tree_sum(tid);
    let inv_std = inverseSqrt(scratch[0] * n_inv + eps);
    workgroupBarrier();

    // ── Pass 3: scale/shift per channel. ──
    if (OUT_HALF) {
        // pairs: `hw` is even, so both halves of a word are one channel
        var j2: u32 = 2u * tid;
        loop {
            if (j2 >= span) { break; }
            let ch = gi * cpg + j2 / params.hw;
            let g = arena[params.gamma_off + ch];
            let b = arena[params.beta_off + ch];
            put2(out_base + j2,
                 norm_act((ld(in_base + j2) - mean) * inv_std * g + b),
                 norm_act((ld(in_base + j2 + 1u) - mean) * inv_std * g + b));
            j2 = j2 + 2u * WG;
        }
        return;
    }
    var j: u32 = tid;
    loop {
        if (j >= span) { break; }
        let ch = gi * cpg + j / params.hw;
        let g = arena[params.gamma_off + ch];
        let b = arena[params.beta_off + ch];
        put(out_base + j, norm_act((ld(in_base + j) - mean) * inv_std * g + b));
        j = j + WG;
    }
}

// ── Split over `chunks` workgroups per group ──
//
// One workgroup per group leaves an instance norm over a volume with a
// handful of workgroups (MindGrab: 15, each streaming 64 MB) — the GPU mostly
// idle. Instead, three dispatches:
//
//   group_norm_partial  (group, chunk): the chunk's mean and its sum of
//                       squared deviations from it, stored in the head of the
//                       group's OUTPUT (2 floats per chunk; the output is not
//                       written yet, and `lower` keeps the split off in-place
//                       norms);
//   group_norm_apply    (group, chunk): combines the partials into the
//                       group's mean and variance — exactly, `M2 = Σ M2_c +
//                       n_c (m_c − m)²`, no cancellation — and writes the
//                       chunk, except the head the partials live in;
//   group_norm_head     (group): the same statistics, then the head.
//
// Every workgroup combines the partials in the same order, so all of them
// use bit-identical statistics.

// A chunk's length; with a packed output, chunks start on a word.
fn chunk_per(span: u32) -> u32 {
    let per = (span + params.chunks - 1u) / params.chunks;
    if (OUT_HALF) { return (per + 1u) & ~1u; }
    return per;
}

// Where a group's partials live: the head of its output, in words.
fn stat_at(at: u32) -> u32 {
    if (OUT_HALF) { return params.out_off + (at >> 1u); }
    return params.out_off + at;
}

// Output elements the partials cover, written last by `group_norm_head`.
fn head_len() -> u32 {
    if (OUT_HALF) { return 4u * params.chunks; }
    return 2u * params.chunks;
}

fn chunk_len(c: u32, span: u32) -> u32 {
    let per = chunk_per(span);
    let lo = min(c * per, span);
    return min(lo + per, span) - lo;
}

fn partial_sum(base: u32, len: u32, tid: u32, shift: f32, sq: bool) -> f32 {
    var p0: f32 = 0.0; var p1: f32 = 0.0; var p2: f32 = 0.0; var p3: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i + 3u * WG >= len) { break; }
        var v0 = ld(base + i) - shift;
        var v1 = ld(base + i + WG) - shift;
        var v2 = ld(base + i + 2u * WG) - shift;
        var v3 = ld(base + i + 3u * WG) - shift;
        if (sq) { v0 = v0 * v0; v1 = v1 * v1; v2 = v2 * v2; v3 = v3 * v3; }
        p0 = p0 + v0; p1 = p1 + v1; p2 = p2 + v2; p3 = p3 + v3;
        i = i + 4u * WG;
    }
    loop {
        if (i >= len) { break; }
        var v = ld(base + i) - shift;
        if (sq) { v = v * v; }
        p0 = p0 + v;
        i = i + WG;
    }
    scratch[tid] = (p0 + p1) + (p2 + p3);
    workgroupBarrier();
    tree_sum(tid);
    let r = scratch[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(256)
fn group_norm_partial(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let lin = wid.x + wid.y * ngs.x;
    if (lin >= params.groups_total * params.chunks) { return; }
    let grp = lin / params.chunks;
    let c = lin - grp * params.chunks;
    let tid = lid.x;
    let ni = grp / params.num_groups;
    let gi = grp - ni * params.num_groups;
    let span = params.group_elems;
    let at = ni * params.c * params.hw + gi * span;
    let per = chunk_per(span);
    let len = chunk_len(c, span);
    let base = params.in_off + at + min(c * per, span);
    var m: f32 = 0.0;
    var m2: f32 = 0.0;
    if (len > 0u) {
        m = partial_sum(base, len, tid, 0.0, false) / f32(len);
        m2 = partial_sum(base, len, tid, m, true);
    }
    if (tid == 0u) {
        put(stat_at(at) + 2u * c, m);
        put(stat_at(at) + 2u * c + 1u, m2);
    }
}

// The group's (mean, inv_std) from its partials; all threads get them.
fn split_stats(at: u32, tid: u32, span: u32, eps: f32) -> vec2<f32> {
    var nm: f32 = 0.0;
    if (tid < params.chunks) {
        nm = f32(chunk_len(tid, span)) * got(stat_at(at) + 2u * tid);
    }
    scratch[tid] = nm;
    workgroupBarrier();
    tree_sum(tid);
    let mean = scratch[0] / f32(span);
    workgroupBarrier();
    var q: f32 = 0.0;
    if (tid < params.chunks) {
        let d = got(stat_at(at) + 2u * tid) - mean;
        q = got(stat_at(at) + 2u * tid + 1u) + f32(chunk_len(tid, span)) * d * d;
    }
    scratch[tid] = q;
    workgroupBarrier();
    tree_sum(tid);
    let inv_std = inverseSqrt(scratch[0] / f32(span) + eps);
    workgroupBarrier();
    return vec2<f32>(mean, inv_std);
}

fn split_write(at: u32, gi: u32, lo: u32, hi: u32, tid: u32, st: vec2<f32>) {
    let cpg = params.c / params.num_groups;
    if (OUT_HALF) {
        // pairs: `lo`, `hi` and `hw` are even
        var j2: u32 = lo + 2u * tid;
        loop {
            if (j2 >= hi) { break; }
            let ch = gi * cpg + j2 / params.hw;
            let g = arena[params.gamma_off + ch];
            let b = arena[params.beta_off + ch];
            put2(params.out_off + at + j2,
                 norm_act((ld(params.in_off + at + j2) - st.x) * st.y * g + b),
                 norm_act((ld(params.in_off + at + j2 + 1u) - st.x) * st.y * g + b));
            j2 = j2 + 2u * WG;
        }
        return;
    }
    var j: u32 = lo + tid;
    loop {
        if (j >= hi) { break; }
        let ch = gi * cpg + j / params.hw;
        let g = arena[params.gamma_off + ch];
        let b = arena[params.beta_off + ch];
        put(params.out_off + at + j, norm_act((ld(params.in_off + at + j) - st.x) * st.y * g + b));
        j = j + WG;
    }
}

@compute @workgroup_size(256)
fn group_norm_apply(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let lin = wid.x + wid.y * ngs.x;
    if (lin >= params.groups_total * params.chunks) { return; }
    let grp = lin / params.chunks;
    let c = lin - grp * params.chunks;
    let ni = grp / params.num_groups;
    let gi = grp - ni * params.num_groups;
    let span = params.group_elems;
    let at = ni * params.c * params.hw + gi * span;
    let st = split_stats(at, lid.x, span, bitcast<f32>(params.eps_bits));
    let per = chunk_per(span);
    let lo = max(min(c * per, span), head_len());
    let hi = min(c * per + per, span);
    split_write(at, gi, lo, hi, lid.x, st);
}

@compute @workgroup_size(256)
fn group_norm_head(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let grp = wid.x + wid.y * ngs.x;
    if (grp >= params.groups_total) { return; }
    let ni = grp / params.num_groups;
    let gi = grp - ni * params.num_groups;
    let span = params.group_elems;
    let at = ni * params.c * params.hw + gi * span;
    let st = split_stats(at, lid.x, span, bitcast<f32>(params.eps_bits));
    split_write(at, gi, 0u, min(head_len(), span), lid.x, st);
}
