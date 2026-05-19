// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Tier 1 autodiff lowering parity: each backward op runs on MLX and
//! its output matches a hand-computed reference. Mirrors the formulas
//! in `rlx-cpu/src/thunk.rs` so a regression in either side surfaces.

#![cfg(target_os = "macos")]

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_mlx::MlxExecutable;

fn close(got: &[f32], want: &[f32], tol: f32) -> bool {
    got.len() == want.len() && got.iter().zip(want).all(|(a, b)| (a - b).abs() <= tol)
}

/// Abramowitz & Stegun 7.1.26 erf approximation (max error ~1.5e-7).
/// Matches the polynomial used by `rlx-cpu`'s `erf_f32`; MLX itself
/// uses a true erf via `mc::erf`, but we tolerate the ~1e-7 gap.
fn erf_approx(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    s * y
}

fn run(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    let mut exe = MlxExecutable::compile(g);
    exe.run(inputs).into_iter().next().unwrap()
}

#[test]
fn relu_backward_matches_reference() {
    let mut g = Graph::new("relu_bwd");
    let x = g.input("x", Shape::new(&[6], DType::F32));
    let dy = g.input("dy", Shape::new(&[6], DType::F32));
    let dx = g.relu_backward(x, dy);
    g.set_outputs(vec![dx]);

    let xs = [-2.0, -0.0, 0.5, 1.0, -3.0, 4.0];
    let dys = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    let want: Vec<f32> = xs
        .iter()
        .zip(dys.iter())
        .map(|(&x, &dy)| if x > 0.0 { dy } else { 0.0 })
        .collect();
    let got = run(g, &[("x", &xs), ("dy", &dys)]);
    assert!(
        close(&got, &want, 1e-5),
        "ReluBackward: got {got:?} want {want:?}"
    );
}

fn activation_grad_ref(kind: Activation, x: f32, dy: f32) -> f32 {
    match kind {
        Activation::Relu => {
            if x > 0.0 {
                dy
            } else {
                0.0
            }
        }
        Activation::Sigmoid => {
            let s = 1.0 / (1.0 + (-x).exp());
            s * (1.0 - s) * dy
        }
        Activation::Tanh => {
            let t = x.tanh();
            (1.0 - t * t) * dy
        }
        Activation::Silu => {
            let s = 1.0 / (1.0 + (-x).exp());
            s * (1.0 + x * (1.0 - s)) * dy
        }
        Activation::Gelu => {
            const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
            const INV_SQRT_2PI: f32 = 0.398_942_3;
            let phi = 0.5 * (1.0 + erf_approx(x * INV_SQRT2));
            let pdf = INV_SQRT_2PI * (-(x * x) * 0.5).exp();
            (phi + x * pdf) * dy
        }
        Activation::GeluApprox => {
            const C: f32 = 0.797_884_6;
            const A: f32 = 0.044_715;
            let inner = C * (x + A * x * x * x);
            let t = inner.tanh();
            let dinner = C * (1.0 + 3.0 * A * x * x);
            let d = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner;
            d * dy
        }
        Activation::Exp => x.exp() * dy,
        Activation::Log => dy / x,
        Activation::Sqrt => {
            let s = x.sqrt();
            if s > 0.0 { 0.5 * dy / s } else { 0.0 }
        }
        Activation::Rsqrt => {
            let s = x.sqrt();
            if s > 0.0 { -0.5 * dy / (x * s) } else { 0.0 }
        }
        Activation::Neg => -dy,
        Activation::Abs => {
            let s = if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            };
            s * dy
        }
        Activation::Round => dy,
        Activation::Sin => x.cos() * dy,
        Activation::Cos => -x.sin() * dy,
    }
}

fn check_activation_backward(kind: Activation, xs: &[f32], dys: &[f32], tol: f32) {
    let mut g = Graph::new("act_bwd");
    let x = g.input("x", Shape::new(&[xs.len()], DType::F32));
    let dy = g.input("dy", Shape::new(&[dys.len()], DType::F32));
    let dx = g.activation_backward(kind, x, dy);
    g.set_outputs(vec![dx]);
    let want: Vec<f32> = xs
        .iter()
        .zip(dys.iter())
        .map(|(&x, &dy)| activation_grad_ref(kind, x, dy))
        .collect();
    let got = run(g, &[("x", xs), ("dy", dys)]);
    assert!(
        close(&got, &want, tol),
        "ActivationBackward({kind:?}): got {got:?} want {want:?}"
    );
}

#[test]
fn activation_backward_all_kinds() {
    // Mix of positive / negative / zero / mid-range. Avoid x≤0 for
    // Log/Sqrt/Rsqrt (undefined / clipped to 0 by both sides).
    let xs_pos: Vec<f32> = (1..=8).map(|i| 0.25 * i as f32).collect();
    let dys_pos: Vec<f32> = (1..=8).map(|i| 0.5 + 0.1 * i as f32).collect();
    let xs_any: Vec<f32> = vec![-2.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0];
    let dys_any: Vec<f32> = vec![1.0, -2.0, 3.0, 4.0, -5.0, 6.0, -7.0, 8.0];

    for k in [
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::Silu,
        Activation::Gelu,
        Activation::GeluApprox,
        Activation::Exp,
        Activation::Neg,
        Activation::Abs,
        Activation::Round,
        Activation::Relu,
    ] {
        check_activation_backward(k, &xs_any, &dys_any, 5e-5);
    }
    for k in [Activation::Log, Activation::Sqrt, Activation::Rsqrt] {
        check_activation_backward(k, &xs_pos, &dys_pos, 5e-5);
    }
}

#[test]
fn softmax_cross_entropy_with_logits_matches_reference() {
    let n = 3usize;
    let c = 4usize;
    let mut g = Graph::new("sce_fwd");
    let logits = g.input("logits", Shape::new(&[n, c], DType::F32));
    let labels = g.input("labels", Shape::new(&[n], DType::F32));
    let loss = g.softmax_cross_entropy_with_logits(logits, labels);
    g.set_outputs(vec![loss]);

    let lg: Vec<f32> = vec![2.0, 1.0, 0.1, -0.5, 0.0, 0.5, 0.7, 0.2, -1.0, 0.3, 2.5, 1.0];
    let lb: Vec<f32> = vec![0.0, 2.0, 2.0];

    let want: Vec<f32> = (0..n)
        .map(|i| {
            let row = &lg[i * c..(i + 1) * c];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|v| (v - m).exp()).sum();
            let lse = m + sum.ln();
            lse - row[lb[i] as usize]
        })
        .collect();

    let got = run(g, &[("logits", &lg), ("labels", &lb)]);
    assert!(
        close(&got, &want, 1e-5),
        "SCE forward: got {got:?} want {want:?}"
    );
}

#[test]
fn softmax_cross_entropy_backward_matches_reference() {
    let n = 3usize;
    let c = 4usize;
    let mut g = Graph::new("sce_bwd");
    let logits = g.input("logits", Shape::new(&[n, c], DType::F32));
    let labels = g.input("labels", Shape::new(&[n], DType::F32));
    let d_loss = g.input("d_loss", Shape::new(&[n], DType::F32));
    let dlogits = g.softmax_cross_entropy_backward(logits, labels, d_loss);
    g.set_outputs(vec![dlogits]);

    let lg: Vec<f32> = vec![2.0, 1.0, 0.1, -0.5, 0.0, 0.5, 0.7, 0.2, -1.0, 0.3, 2.5, 1.0];
    let lb: Vec<f32> = vec![0.0, 2.0, 2.0];
    let dl: Vec<f32> = vec![1.0, 0.5, 2.0];

    let mut want = vec![0f32; n * c];
    for i in 0..n {
        let row = &lg[i * c..(i + 1) * c];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|v| (v - m).exp()).sum();
        let inv = 1.0 / sum;
        let label = lb[i] as usize;
        for k in 0..c {
            let p = (row[k] - m).exp() * inv;
            let oh = if k == label { 1.0 } else { 0.0 };
            want[i * c + k] = (p - oh) * dl[i];
        }
    }

    let got = run(g, &[("logits", &lg), ("labels", &lb), ("d_loss", &dl)]);
    assert!(
        close(&got, &want, 1e-5),
        "SCE backward: got {got:?} want {want:?}"
    );
}

fn layernorm_grad_ref_input(
    xs: &[f32],
    gamma: &[f32],
    dys: &[f32],
    rows: usize,
    h: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; rows * h];
    let n_inv = 1.0 / h as f32;
    for r in 0..rows {
        let xr = &xs[r * h..(r + 1) * h];
        let dyr = &dys[r * h..(r + 1) * h];
        let mean: f32 = xr.iter().sum::<f32>() * n_inv;
        let var: f32 = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() * n_inv;
        let inv_std = 1.0 / (var + eps).sqrt();
        let mut s_sy = 0f32;
        let mut s_sxh = 0f32;
        for d in 0..h {
            let xh = (xr[d] - mean) * inv_std;
            let sy = dyr[d] * gamma[d];
            s_sy += sy;
            s_sxh += sy * xh;
        }
        let m_sy = s_sy * n_inv;
        let m_sxh = s_sxh * n_inv;
        for d in 0..h {
            let xh = (xr[d] - mean) * inv_std;
            let sy = dyr[d] * gamma[d];
            out[r * h + d] = inv_std * (sy - m_sy - xh * m_sxh);
        }
    }
    out
}

fn layernorm_grad_ref_gamma(xs: &[f32], dys: &[f32], rows: usize, h: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; h];
    let n_inv = 1.0 / h as f32;
    for r in 0..rows {
        let xr = &xs[r * h..(r + 1) * h];
        let dyr = &dys[r * h..(r + 1) * h];
        let mean: f32 = xr.iter().sum::<f32>() * n_inv;
        let var: f32 = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() * n_inv;
        let inv_std = 1.0 / (var + eps).sqrt();
        for d in 0..h {
            let xh = (xr[d] - mean) * inv_std;
            out[d] += dyr[d] * xh;
        }
    }
    out
}

#[test]
fn layer_norm_backward_input_matches_reference() {
    let rows = 3usize;
    let h = 5usize;
    let eps = 1e-5;
    let mut g = Graph::new("ln_bwd_in");
    let x = g.input("x", Shape::new(&[rows, h], DType::F32));
    let gamma = g.input("gamma", Shape::new(&[h], DType::F32));
    let dy = g.input("dy", Shape::new(&[rows, h], DType::F32));
    let dx = g.layer_norm_backward_input(x, gamma, dy, -1, eps);
    g.set_outputs(vec![dx]);

    let xs: Vec<f32> = (0..rows * h).map(|i| 0.1 * (i as f32 - 5.0)).collect();
    let gs: Vec<f32> = (0..h).map(|i| 0.5 + 0.2 * i as f32).collect();
    let dys: Vec<f32> = (0..rows * h).map(|i| 1.0 + 0.05 * i as f32).collect();

    let want = layernorm_grad_ref_input(&xs, &gs, &dys, rows, h, eps);
    let got = run(g, &[("x", &xs), ("gamma", &gs), ("dy", &dys)]);
    assert!(
        close(&got, &want, 5e-5),
        "LayerNormBackwardInput: got {got:?} want {want:?}"
    );
}

#[test]
fn layer_norm_backward_gamma_matches_reference() {
    let rows = 4usize;
    let h = 6usize;
    let eps = 1e-5;
    let mut g = Graph::new("ln_bwd_g");
    let x = g.input("x", Shape::new(&[rows, h], DType::F32));
    let dy = g.input("dy", Shape::new(&[rows, h], DType::F32));
    let dgamma = g.layer_norm_backward_gamma(x, dy, Shape::new(&[h], DType::F32), -1, eps);
    g.set_outputs(vec![dgamma]);

    let xs: Vec<f32> = (0..rows * h).map(|i| 0.05 * (i as f32 - 7.0)).collect();
    let dys: Vec<f32> = (0..rows * h).map(|i| 0.5 + 0.03 * i as f32).collect();

    let want = layernorm_grad_ref_gamma(&xs, &dys, rows, h, eps);
    let got = run(g, &[("x", &xs), ("dy", &dys)]);
    assert!(
        close(&got, &want, 5e-5),
        "LayerNormBackwardGamma: got {got:?} want {want:?}"
    );
}

#[test]
fn end_to_end_grad_relu_mlp_matches_reference() {
    // Small MLP with a relu nonlinearity and a softmax-CE loss.
    // Build the forward graph, run grad_with_loss, execute the
    // gradient graph on MLX, and check the param gradients against a
    // hand-computed reference.
    let n = 2usize;
    let in_d = 3usize;
    let hidden = 4usize;
    let c = 3usize;
    let mut fwd = Graph::new("mlp");
    let x = fwd.input("x", Shape::new(&[n, in_d], DType::F32));
    let labels = fwd.input("labels", Shape::new(&[n], DType::F32));
    let w1 = fwd.param("w1", Shape::new(&[in_d, hidden], DType::F32));
    let w2 = fwd.param("w2", Shape::new(&[hidden, c], DType::F32));
    let h1 = fwd.matmul(x, w1, Shape::new(&[n, hidden], DType::F32));
    let a1 = fwd.activation(Activation::Relu, h1, Shape::new(&[n, hidden], DType::F32));
    let logits = fwd.matmul(a1, w2, Shape::new(&[n, c], DType::F32));
    let losses = fwd.softmax_cross_entropy_with_logits(logits, labels);
    let loss = fwd.add_node(
        Op::Reduce {
            op: rlx_ir::op::ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![losses],
        Shape::new(&[], DType::F32),
    );
    fwd.set_outputs(vec![loss]);

    let bwd = rlx_opt::autodiff::grad_with_loss(&fwd, &[w1, w2]);

    let xs: Vec<f32> = vec![0.1, -0.2, 0.3, 0.4, 0.5, -0.6];
    let lb: Vec<f32> = vec![0.0, 2.0];
    let w1v: Vec<f32> = (0..in_d * hidden).map(|i| 0.1 + 0.05 * i as f32).collect();
    let w2v: Vec<f32> = (0..hidden * c).map(|i| -0.1 + 0.07 * i as f32).collect();
    let d_out: Vec<f32> = vec![1.0];

    let mut exe = MlxExecutable::compile(bwd);
    exe.set_param("w1", &w1v);
    exe.set_param("w2", &w2v);
    let outs = exe.run(&[("x", &xs), ("labels", &lb), ("d_output", &d_out)]);
    assert_eq!(outs.len(), 3, "expected [loss, dW1, dW2]");

    // Hand-compute reference: forward then standard MLP backward.
    let mut h1v = vec![0f32; n * hidden];
    for i in 0..n {
        for j in 0..hidden {
            let mut s = 0f32;
            for k in 0..in_d {
                s += xs[i * in_d + k] * w1v[k * hidden + j];
            }
            h1v[i * hidden + j] = s;
        }
    }
    let a1v: Vec<f32> = h1v.iter().map(|&v| v.max(0.0)).collect();
    let mut logitsv = vec![0f32; n * c];
    for i in 0..n {
        for j in 0..c {
            let mut s = 0f32;
            for k in 0..hidden {
                s += a1v[i * hidden + k] * w2v[k * c + j];
            }
            logitsv[i * c + j] = s;
        }
    }
    let mut loss_ref = 0f32;
    let mut dlogits = vec![0f32; n * c];
    for i in 0..n {
        let row = &logitsv[i * c..(i + 1) * c];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|v| (v - m).exp()).sum();
        let lse = m + sum.ln();
        let label = lb[i] as usize;
        loss_ref += lse - row[label];
        let inv = 1.0 / sum;
        for k in 0..c {
            let p = (row[k] - m).exp() * inv;
            let oh = if k == label { 1.0 } else { 0.0 };
            dlogits[i * c + k] = (p - oh) * d_out[0];
        }
    }
    // dW2 = a1ᵀ · dlogits.
    let mut dw2 = vec![0f32; hidden * c];
    for k in 0..hidden {
        for j in 0..c {
            let mut s = 0f32;
            for i in 0..n {
                s += a1v[i * hidden + k] * dlogits[i * c + j];
            }
            dw2[k * c + j] = s;
        }
    }
    // da1 = dlogits · w2ᵀ; dh1 = da1 ⊙ (h1>0); dW1 = xᵀ · dh1.
    let mut da1 = vec![0f32; n * hidden];
    for i in 0..n {
        for k in 0..hidden {
            let mut s = 0f32;
            for j in 0..c {
                s += dlogits[i * c + j] * w2v[k * c + j];
            }
            da1[i * hidden + k] = s;
        }
    }
    let dh1: Vec<f32> = da1
        .iter()
        .zip(h1v.iter())
        .map(|(&g, &h)| if h > 0.0 { g } else { 0.0 })
        .collect();
    let mut dw1 = vec![0f32; in_d * hidden];
    for k in 0..in_d {
        for j in 0..hidden {
            let mut s = 0f32;
            for i in 0..n {
                s += xs[i * in_d + k] * dh1[i * hidden + j];
            }
            dw1[k * hidden + j] = s;
        }
    }

    assert!(
        close(&outs[0], &[loss_ref], 1e-4),
        "loss: got {:?} want {loss_ref}",
        outs[0]
    );
    assert!(
        close(&outs[1], &dw1, 1e-4),
        "dW1: got {:?} want {dw1:?}",
        outs[1]
    );
    assert!(
        close(&outs[2], &dw2, 1e-4),
        "dW2: got {:?} want {dw2:?}",
        outs[2]
    );
}
