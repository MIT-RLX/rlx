// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Conv3d backward-weight (NCDHW). One thread per dw element.

struct Params {
    n: u32,
    c_in: u32,
    d: u32,
    h: u32,
    w: u32,
    c_out: u32,
    d_out: u32,
    h_out: u32,
    w_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
    dd: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    x_off: u32,
    dy_off: u32,
    dw_off: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

var<workgroup> partial: array<f32, 256>;

// One workgroup per weight element: its 256 threads split the sum over
// batch × output positions (strided), then reduce in shared memory. A thread
// per weight element left the GPU idle for small conv weights (EEGNet's
// temporal filter: ~500 threads, each a ~90k-term serial sum).
@compute @workgroup_size(256, 1, 1)
fn conv3d_backward_weight(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let i = wid.x + wid.y * nwg.x;
    let c_in_per_g = params.c_in / params.groups;
    let c_out_per_g = params.c_out / params.groups;
    let total = params.c_out * c_in_per_g * params.kd * params.kh * params.kw;
    if (i >= total) {
        return; // uniform across the workgroup
    }
    let kj = i % params.kw;
    let q1 = i / params.kw;
    let ki = q1 % params.kh;
    let q2 = q1 / params.kh;
    let kz = q2 % params.kd;
    let q3 = q2 / params.kd;
    let ci_off = q3 % c_in_per_g;
    let co = q3 / c_in_per_g;
    let g = co / c_out_per_g;
    let ci = g * c_in_per_g + ci_off;

    let spatial = params.d_out * params.h_out * params.w_out;
    let m = params.n * spatial;
    // Thread `lid` visits t = lid, lid + 256, … — the decomposition of t into
    // (nn, do_, ho, wo) is advanced by the mixed-radix digits of 256 instead of
    // being re-divided every step: four integer divisions per MAC were the
    // kernel's cost. Same t sequence, so the same sums in the same order.
    let hw = params.h_out * params.w_out;
    let s_wo = 256u % params.w_out;
    let s_ho = (256u / params.w_out) % params.h_out;
    let s_do = (256u / hw) % params.d_out;
    let s_nn = 256u / spatial;
    var wo = lid % params.w_out;
    var ho = (lid / params.w_out) % params.h_out;
    var do_ = (lid / hw) % params.d_out;
    var nn = lid / spatial;
    let iz_base = i32(kz * params.dd) - i32(params.pd);
    let ih_base = i32(ki * params.dh) - i32(params.ph);
    let iw_base = i32(kj * params.dw) - i32(params.pw);
    var acc = 0.0;
    for (var t = lid; t < m; t = t + 256u) {
        let id = i32(do_ * params.sd) + iz_base;
        let ih = i32(ho * params.sh) + ih_base;
        let iw = i32(wo * params.sw) + iw_base;
        if (id >= 0 && id < i32(params.d) && ih >= 0 && ih < i32(params.h) && iw >= 0 && iw < i32(params.w)) {
            let dyv = arena[params.dy_off + ((((nn * params.c_out + co) * params.d_out + do_) * params.h_out + ho) * params.w_out + wo)];
            let xv = arena[params.x_off + ((((nn * params.c_in + ci) * params.d + u32(id)) * params.h + u32(ih)) * params.w + u32(iw))];
            acc = acc + dyv * xv;
        }
        wo = wo + s_wo;
        var carry = select(0u, 1u, wo >= params.w_out);
        wo = wo - carry * params.w_out;
        ho = ho + s_ho + carry;
        carry = select(0u, 1u, ho >= params.h_out);
        ho = ho - carry * params.h_out;
        do_ = do_ + s_do + carry;
        carry = select(0u, 1u, do_ >= params.d_out);
        do_ = do_ - carry * params.d_out;
        nn = nn + s_nn + carry;
    }
    partial[lid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            partial[lid] = partial[lid] + partial[lid + s];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        arena[params.dw_off + i] = partial[0];
    }
}
