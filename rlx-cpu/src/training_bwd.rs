// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Closed-form training backward kernels (RMSNorm, RoPE, GroupNorm, Cumsum, Gather).

/// RMSNorm backward for one row: `y = x * inv_rms * gamma + beta`.
pub fn rms_norm_backward_row(
    x: &[f32],
    gamma: &[f32],
    _beta: &[f32],
    dy: &[f32],
    dx: &mut [f32],
    dgamma: &mut [f32],
    dbeta: &mut [f32],
    eps: f32,
) {
    let h = x.len();
    debug_assert_eq!(h, gamma.len());
    let inv_h = 1.0 / h as f32;
    let mut sumsq = 0f32;
    for &v in x {
        sumsq += v * v;
    }
    let inv_r = (sumsq * inv_h + eps).sqrt().recip();
    let inv_r3 = inv_r * inv_r * inv_r;

    let mut dot = 0f32;
    for i in 0..h {
        dot += dy[i] * gamma[i] * x[i];
    }
    dot *= inv_h;

    for i in 0..h {
        let term = gamma[i] * dy[i] - x[i] * dot * inv_r3;
        dx[i] = term * inv_r;
        dgamma[i] += dy[i] * x[i] * inv_r;
        dbeta[i] += dy[i];
    }
}

/// Inclusive/exclusive cumsum backward along one row.
pub fn cumsum_backward_row(src_dy: &[f32], dst_dx: &mut [f32], exclusive: bool) {
    let l = src_dy.len();
    if exclusive {
        let mut suffix = 0f32;
        for i in (0..l).rev() {
            dst_dx[i] = suffix;
            suffix += src_dy[i];
        }
    } else {
        let mut suffix = 0f32;
        for i in (0..l).rev() {
            suffix += src_dy[i];
            dst_dx[i] = suffix;
        }
    }
}

/// NeoX RoPE backward: same as forward with negated sin table.
pub fn rope_backward_row(
    dy: &[f32],
    cos: &[f32],
    sin: &[f32],
    dx: &mut [f32],
    head_dim: usize,
    n_rot: usize,
) {
    let tab_half = head_dim / 2;
    let rot_half = n_rot / 2;
    debug_assert!(dy.len() >= head_dim && dx.len() >= head_dim);
    for i in 0..rot_half {
        let y1 = dy[i];
        let y2 = dy[rot_half + i];
        let cv = cos[i];
        let sv = sin[i];
        dx[i] = y1 * cv + y2 * sv;
        dx[rot_half + i] = -y1 * sv + y2 * cv;
    }
    dx[n_rot..head_dim].copy_from_slice(&dy[n_rot..head_dim]);
    let _ = tab_half;
}

/// GroupNorm (NCHW) backward w.r.t. input.
pub fn group_norm_backward_input_nchw(
    input: &[f32],
    gamma: &[f32],
    dy: &[f32],
    d_input: &mut [f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    num_groups: usize,
    eps: f32,
) {
    let spatial = h * w;
    let plane = channels * spatial;
    let cpg = channels / num_groups;
    let n = (cpg * spatial) as f32;
    let n_inv = 1.0 / n;
    for b in 0..batch {
        let b_in = b * plane;
        let b_dy = b * plane;
        let b_out = b * plane;
        for g in 0..num_groups {
            let c0 = g * cpg;
            let mut mean = 0f32;
            for c in 0..cpg {
                let base = b_in + (c0 + c) * spatial;
                for s in 0..spatial {
                    mean += input[base + s];
                }
            }
            mean *= n_inv;
            let mut var = 0f32;
            for c in 0..cpg {
                let base = b_in + (c0 + c) * spatial;
                for s in 0..spatial {
                    let d = input[base + s] - mean;
                    var += d * d;
                }
            }
            var *= n_inv;
            let inv_std = 1.0 / (var + eps).sqrt();
            let mut s_sy = 0f32;
            let mut s_sxh = 0f32;
            for c in 0..cpg {
                let gi = c0 + c;
                let gamm = gamma[gi];
                let base = b_in + gi * spatial;
                let dy_base = b_dy + gi * spatial;
                for s in 0..spatial {
                    let xh = (input[base + s] - mean) * inv_std;
                    let sy = dy[dy_base + s] * gamm;
                    s_sy += sy;
                    s_sxh += sy * xh;
                }
            }
            let m_sy = s_sy * n_inv;
            let m_sxh = s_sxh * n_inv;
            for c in 0..cpg {
                let gi = c0 + c;
                let gamm = gamma[gi];
                let base = b_in + gi * spatial;
                let dy_base = b_dy + gi * spatial;
                let out_base = b_out + gi * spatial;
                for s in 0..spatial {
                    let xh = (input[base + s] - mean) * inv_std;
                    let sy = dy[dy_base + s] * gamm;
                    d_input[out_base + s] = inv_std * (sy - m_sy - xh * m_sxh);
                }
            }
        }
    }
}

/// GroupNorm backward w.r.t. gamma (accumulates over batch and spatial).
pub fn group_norm_backward_gamma_nchw(
    input: &[f32],
    dy: &[f32],
    d_gamma: &mut [f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    num_groups: usize,
    eps: f32,
) {
    d_gamma.fill(0.0);
    let spatial = h * w;
    let plane = channels * spatial;
    let cpg = channels / num_groups;
    let n = (cpg * spatial) as f32;
    let n_inv = 1.0 / n;
    for b in 0..batch {
        let b_in = b * plane;
        let b_dy = b * plane;
        for g in 0..num_groups {
            let c0 = g * cpg;
            let mut mean = 0f32;
            for c in 0..cpg {
                let base = b_in + (c0 + c) * spatial;
                for s in 0..spatial {
                    mean += input[base + s];
                }
            }
            mean *= n_inv;
            let mut var = 0f32;
            for c in 0..cpg {
                let base = b_in + (c0 + c) * spatial;
                for s in 0..spatial {
                    let d = input[base + s] - mean;
                    var += d * d;
                }
            }
            var *= n_inv;
            let inv_std = 1.0 / (var + eps).sqrt();
            for c in 0..cpg {
                let gi = c0 + c;
                let base = b_in + gi * spatial;
                let dy_base = b_dy + gi * spatial;
                for s in 0..spatial {
                    let xh = (input[base + s] - mean) * inv_std;
                    d_gamma[gi] += dy[dy_base + s] * xh;
                }
            }
        }
    }
}

/// GroupNorm backward w.r.t. beta.
pub fn group_norm_backward_beta_nchw(
    dy: &[f32],
    d_beta: &mut [f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
) {
    d_beta.fill(0.0);
    let spatial = h * w;
    let plane = channels * spatial;
    for b in 0..batch {
        let b_dy = b * plane;
        for c in 0..channels {
            let dy_base = b_dy + c * spatial;
            for s in 0..spatial {
                d_beta[c] += dy[dy_base + s];
            }
        }
    }
}

/// Gather-axis backward: `d_table[o, idx[k], t] += dy[o, k, t]`.
pub fn gather_axis_backward(
    dy: &[f32],
    indices: &[f32],
    d_table: &mut [f32],
    outer: usize,
    axis_dim: usize,
    num_idx: usize,
    trailing: usize,
) {
    for o in 0..outer {
        let dy_outer = o * num_idx * trailing;
        let tab_outer = o * axis_dim * trailing;
        for k in 0..num_idx {
            let row = indices[k] as usize;
            debug_assert!(row < axis_dim);
            let dy_row = dy_outer + k * trailing;
            let tab_row = tab_outer + row * trailing;
            for j in 0..trailing {
                d_table[tab_row + j] += dy[dy_row + j];
            }
        }
    }
}
