// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Scaled dot-product attention backward ([B, H, S, D] or strided layouts).
// `wrt`: 0 = dQ, 1 = dK, 2 = dV.

const MAX_HEAD_DIM: u32 = 128u;
const MAX_ATTN_SEQ: u32 = 512u;

struct Params {
    batch: u32,
    heads: u32,
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    dy_off: u32,
    out_off: u32,
    mask_off: u32,
    mask_kind: u32,
    scale_bits: u32,
    window: u32,
    wrt: u32,
    seq_q_stride: u32,
    seq_k_stride: u32,
    mask_batch_stride: u32,
    mask_head_stride: u32,
    _pad_mask_0: u32,
    _pad_mask_1: u32,
    _pad_mask_2: u32,
    q_batch_stride: u32, q_head_stride: u32, q_seq_stride: u32, _pad_q: u32,
    k_batch_stride: u32, k_head_stride: u32, k_seq_stride: u32, _pad_k: u32,
    v_batch_stride: u32, v_head_stride: u32, v_seq_stride: u32, _pad_v: u32,
    o_batch_stride: u32, o_head_stride: u32, o_seq_stride: u32, _pad_o: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

fn mask_score(dot: f32, qi: u32, ki: u32, b: u32, h: u32) -> f32 {
    var s = dot;
    if (params.mask_kind == 1u) {
        if (ki > qi) { s = -3.4e38; }
    } else if (params.mask_kind == 2u) {
        let m = params.mask_off
            + b * params.mask_batch_stride
            + h * params.mask_head_stride
            + qi * params.seq_q_stride
            + ki * params.seq_k_stride;
        if (arena[m] < 0.5) { s = -1e9; }
    } else if (params.mask_kind == 3u) {
        if (ki > qi) { s = -3.4e38; }
        else if (qi - ki > params.window) { s = -3.4e38; }
    } else if (params.mask_kind == 4u) {
        let m = params.mask_off
            + b * params.mask_batch_stride
            + h * params.mask_head_stride
            + qi * params.seq_q_stride
            + ki * params.seq_k_stride;
        s = s + arena[m];
    }
    return s;
}

fn softmax_row(scores: ptr<function, array<f32, MAX_ATTN_SEQ>>, seq_k: u32) {
    var m: f32 = -3.4e38;
    for (var s: u32 = 0u; s < seq_k; s = s + 1u) {
        m = max(m, (*scores)[s]);
    }
    var sum: f32 = 0.0;
    for (var s: u32 = 0u; s < seq_k; s = s + 1u) {
        let e = exp((*scores)[s] - m);
        (*scores)[s] = e;
        sum = sum + e;
    }
    let inv = 1.0 / max(sum, 1e-30);
    for (var s: u32 = 0u; s < seq_k; s = s + 1u) {
        (*scores)[s] = (*scores)[s] * inv;
    }
}

@compute @workgroup_size(64)
fn attention_bwd(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    if (params.head_dim > MAX_HEAD_DIM || params.seq_k > MAX_ATTN_SEQ || params.seq_q > MAX_ATTN_SEQ) {
        return;
    }
    let scale = bitcast<f32>(params.scale_bits);
    let row = gid.x + gid.y * ngs.x * 64u;
    let axis_len = select(params.seq_k, params.seq_q, params.wrt == 0u);
    let total = params.batch * params.heads * axis_len;
    if (row >= total) { return; }
    let axis_idx = row % axis_len;
    let q1 = row / axis_len;
    let h = q1 % params.heads;
    let b = q1 / params.heads;

    let q_bh = params.q_off + b * params.q_batch_stride + h * params.q_head_stride;
    let k_bh = params.k_off + b * params.k_batch_stride + h * params.k_head_stride;
    let v_bh = params.v_off + b * params.v_batch_stride + h * params.v_head_stride;
    let dy_bh = params.dy_off + b * params.q_batch_stride + h * params.q_head_stride;
    let o_bh = params.out_off + b * params.o_batch_stride + h * params.o_head_stride;

    var scores: array<f32, MAX_ATTN_SEQ>;
    var dp: array<f32, MAX_ATTN_SEQ>;
    let hd = params.head_dim;

    if (params.wrt == 0u) {
        let qi = axis_idx;
        let q_base = q_bh + qi * params.q_seq_stride;
        let dy_base = dy_bh + qi * params.q_seq_stride;
        let o_base = o_bh + qi * params.o_seq_stride;
        for (var ki: u32 = 0u; ki < params.seq_k; ki = ki + 1u) {
            let k_base = k_bh + ki * params.k_seq_stride;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < hd; d = d + 1u) {
                dot = dot + arena[q_base + d] * arena[k_base + d];
            }
            scores[ki] = mask_score(dot * scale, qi, ki, b, h);
        }
        softmax_row(&scores, params.seq_k);
        for (var ki: u32 = 0u; ki < params.seq_k; ki = ki + 1u) {
            let v_base = v_bh + ki * params.v_seq_stride;
            var acc: f32 = 0.0;
            for (var d: u32 = 0u; d < hd; d = d + 1u) {
                acc = acc + arena[dy_base + d] * arena[v_base + d];
            }
            dp[ki] = acc;
        }
        var row_sum: f32 = 0.0;
        for (var ki: u32 = 0u; ki < params.seq_k; ki = ki + 1u) {
            row_sum = row_sum + scores[ki] * dp[ki];
        }
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            var acc: f32 = 0.0;
            for (var ki: u32 = 0u; ki < params.seq_k; ki = ki + 1u) {
                let ds = scores[ki] * (dp[ki] - row_sum) * scale;
                acc = acc + ds * arena[k_bh + ki * params.k_seq_stride + d];
            }
            arena[o_base + d] = acc;
        }
    } else if (params.wrt == 2u) {
        let ki = axis_idx;
        let k_base = k_bh + ki * params.k_seq_stride;
        let v_base = v_bh + ki * params.v_seq_stride;
        let o_base = o_bh + ki * params.o_seq_stride;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            arena[o_base + d] = 0.0;
        }
        for (var qi: u32 = 0u; qi < params.seq_q; qi = qi + 1u) {
            let q_base = q_bh + qi * params.q_seq_stride;
            let dy_base = dy_bh + qi * params.q_seq_stride;
            for (var kj: u32 = 0u; kj < params.seq_k; kj = kj + 1u) {
                let kb = k_bh + kj * params.k_seq_stride;
                var dot: f32 = 0.0;
                for (var d: u32 = 0u; d < hd; d = d + 1u) {
                    dot = dot + arena[q_base + d] * arena[kb + d];
                }
                scores[kj] = mask_score(dot * scale, qi, kj, b, h);
            }
            softmax_row(&scores, params.seq_k);
            for (var d: u32 = 0u; d < hd; d = d + 1u) {
                arena[o_base + d] = arena[o_base + d] + scores[ki] * arena[dy_base + d];
            }
        }
    } else if (params.wrt == 1u) {
        let ki = axis_idx;
        let o_base = o_bh + ki * params.o_seq_stride;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            arena[o_base + d] = 0.0;
        }
        for (var qi: u32 = 0u; qi < params.seq_q; qi = qi + 1u) {
            let q_base = q_bh + qi * params.q_seq_stride;
            let dy_base = dy_bh + qi * params.q_seq_stride;
            for (var kj: u32 = 0u; kj < params.seq_k; kj = kj + 1u) {
                let kb = k_bh + kj * params.k_seq_stride;
                var dot: f32 = 0.0;
                for (var d: u32 = 0u; d < hd; d = d + 1u) {
                    dot = dot + arena[q_base + d] * arena[kb + d];
                }
                scores[kj] = mask_score(dot * scale, qi, kj, b, h);
            }
            softmax_row(&scores, params.seq_k);
            for (var kj: u32 = 0u; kj < params.seq_k; kj = kj + 1u) {
                let vb = v_bh + kj * params.v_seq_stride;
                var acc: f32 = 0.0;
                for (var d: u32 = 0u; d < hd; d = d + 1u) {
                    acc = acc + arena[dy_base + d] * arena[vb + d];
                }
                dp[kj] = acc;
            }
            var row_sum: f32 = 0.0;
            for (var kj: u32 = 0u; kj < params.seq_k; kj = kj + 1u) {
                row_sum = row_sum + scores[kj] * dp[kj];
            }
            let ds_ki = scores[ki] * (dp[ki] - row_sum) * scale;
            for (var d: u32 = 0u; d < hd; d = d + 1u) {
                arena[o_base + d] = arena[o_base + d] + ds_ki * arena[q_base + d];
            }
        }
    }
}
