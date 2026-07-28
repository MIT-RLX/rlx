// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// LayerNorm backward over the last axis ("row" = last-axis slice).
//
// Three entry points share the same Params layout:
//   * `layer_norm_bwd_input`         — dx (one workgroup per row).
//   * `layer_norm_bwd_gamma_partial` — partial dgamma per row chunk,
//     written to scratch[wg_idx, d].
//   * `layer_norm_bwd_gamma_reduce`  — sum scratch[*, d] → dgamma[d].
//
// Both gamma entry points + the input kernel recompute per-row
// mean/inv_std from x (the forward didn't save them), matching the
// CPU thunk in `rlx-cpu/src/thunk.rs::Thunk::LayerNormBackward*`.

struct Params {
    outer: u32,       // number of rows
    inner: u32,       // size of last axis (H)
    x_off: u32,
    gamma_off: u32,
    dy_off: u32,
    out_off: u32,     // dx or dgamma destination
    eps_bits: u32,
    scratch_off: u32, // word-offset (f32 index) into the tail scratch
                      // zone (gamma path only; ignored for input).
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

var<workgroup> scratch_wg: array<f32, 64>;

fn workgroup_sum(tid: u32, val: f32) -> f32 {
    scratch_wg[tid] = val;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            scratch_wg[tid] = scratch_wg[tid] + scratch_wg[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    return scratch_wg[0];
}

// dx kernel: one workgroup per row, 64 lanes cooperate.
@compute @workgroup_size(64)
fn layer_norm_bwd_input(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= params.outer || params.inner == 0u) { return; }
    let tid = lid.x;
    let h = params.inner;
    let x_base = params.x_off + row * h;
    let dy_base = params.dy_off + row * h;
    let out_base = params.out_off + row * h;
    let n_inv = 1.0 / f32(h);
    let eps = bitcast<f32>(params.eps_bits);

    var s: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= h) { break; }
        s = s + arena[x_base + i];
        i = i + 64u;
    }
    let mean = workgroup_sum(tid, s) * n_inv;

    var s2: f32 = 0.0;
    i = tid;
    loop {
        if (i >= h) { break; }
        let d = arena[x_base + i] - mean;
        s2 = s2 + d * d;
        i = i + 64u;
    }
    let inv_std = inverseSqrt(workgroup_sum(tid, s2) * n_inv + eps);

    var p_sy: f32 = 0.0;
    var p_sxh: f32 = 0.0;
    i = tid;
    loop {
        if (i >= h) { break; }
        let xv = arena[x_base + i];
        let dyv = arena[dy_base + i];
        let gv = arena[params.gamma_off + i];
        let xh = (xv - mean) * inv_std;
        let sy = dyv * gv;
        p_sy = p_sy + sy;
        p_sxh = p_sxh + sy * xh;
        i = i + 64u;
    }
    let m_sy = workgroup_sum(tid, p_sy) * n_inv;
    let m_sxh = workgroup_sum(tid, p_sxh) * n_inv;

    i = tid;
    loop {
        if (i >= h) { break; }
        let xv = arena[x_base + i];
        let dyv = arena[dy_base + i];
        let gv = arena[params.gamma_off + i];
        let xh = (xv - mean) * inv_std;
        let sy = dyv * gv;
        arena[out_base + i] = inv_std * (sy - m_sy - xh * m_sxh);
        i = i + 64u;
    }
}

// `dgamma` partial sums: one workgroup per chunk of ROWS_PER_WG rows.
// `gamma_partial[wg_idx, d] = sum over rows in chunk of dy[r,d] * x_hat[r,d]`.
//
// Each workgroup uses its 64 lanes for the per-row mean+var
// reductions; lane d (d < H) also accumulates its column's
// contribution in a private register across the row chunk, then
// writes the final per-chunk partial to `scratch[wg_idx, d]`.
//
// Caller dispatches `num_workgroups = ceil(rows / ROWS_PER_WG)` and
// allocates `scratch_off` for `num_workgroups * H` f32 entries.
const ROWS_PER_WG: u32 = 16u;

var<workgroup> row_mean_inv: vec2<f32>;

@compute @workgroup_size(64)
fn layer_norm_bwd_gamma_partial(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    if (params.inner == 0u) { return; }
    let tid = lid.x;
    let h = params.inner;
    let rows = params.outer;
    let n_inv = 1.0 / f32(h);
    let eps = bitcast<f32>(params.eps_bits);
    let wg = wid.x;
    let row_start = wg * ROWS_PER_WG;
    if (row_start >= rows) { return; }
    let row_end = min(row_start + ROWS_PER_WG, rows);

    // Each lane d (d < H) holds its accumulator across the chunk.
    var acc: f32 = 0.0;
    for (var r: u32 = row_start; r < row_end; r = r + 1u) {
        let x_base = params.x_off + r * h;
        let dy_base = params.dy_off + r * h;

        // Cooperative mean(x_r).
        var s: f32 = 0.0;
        var i: u32 = tid;
        loop {
            if (i >= h) { break; }
            s = s + arena[x_base + i];
            i = i + 64u;
        }
        let sum_x = workgroup_sum(tid, s);
        if (tid == 0u) { row_mean_inv.x = sum_x * n_inv; }
        workgroupBarrier();
        let mean = row_mean_inv.x;

        // Cooperative var(x_r).
        var s2: f32 = 0.0;
        i = tid;
        loop {
            if (i >= h) { break; }
            let dv = arena[x_base + i] - mean;
            s2 = s2 + dv * dv;
            i = i + 64u;
        }
        let var_sum = workgroup_sum(tid, s2);
        if (tid == 0u) { row_mean_inv.y = inverseSqrt(var_sum * n_inv + eps); }
        workgroupBarrier();
        let inv_std = row_mean_inv.y;

        // Lanes d < H accumulate dy[r,d] * x_hat[r,d].
        if (tid < h) {
            let xv = arena[x_base + tid];
            let dyv = arena[dy_base + tid];
            acc = acc + dyv * (xv - mean) * inv_std;
        }
    }
    // Write per-chunk partial.
    if (tid < h) {
        arena[params.scratch_off + wg * h + tid] = acc;
    }
}

// Final reduce: one workgroup, lanes d sum `scratch[*, d]` across all
// chunks into `dgamma[d]`. `outer` in this dispatch carries the partial
// chunk count (= the original `num_workgroups`).
@compute @workgroup_size(64)
fn layer_norm_bwd_gamma_reduce(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let h = params.inner;
    if (tid >= h || h == 0u) { return; }
    let num_chunks = params.outer;
    var acc: f32 = 0.0;
    for (var c: u32 = 0u; c < num_chunks; c = c + 1u) {
        acc = acc + arena[params.scratch_off + c * h + tid];
    }
    arena[params.out_off + tid] = acc;
}
