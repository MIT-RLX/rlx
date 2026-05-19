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

//! Tier 3 autodiff lowering parity: `MaxPool2dBackward` on MLX vs a
//! hand-written reference matching `rlx-cpu/src/thunk.rs`.
//!
//! Tiebreaking convention: first hit wins (strict `>`), matching both
//! the CPU thunk and MLX's `argmax` first-index-on-ties.

#![cfg(target_os = "macos")]

use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_mlx::MlxExecutable;

fn close(got: &[f32], want: &[f32], tol: f32) -> bool {
    got.len() == want.len() && got.iter().zip(want).all(|(a, b)| (a - b).abs() <= tol)
}

fn run(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    let mut exe = MlxExecutable::compile(g);
    exe.run(inputs).into_iter().next().unwrap()
}

#[allow(clippy::too_many_arguments)]
fn maxpool_forward_ref(
    x: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> (Vec<f32>, usize, usize) {
    let h_out = (h + 2 * ph - kh) / sh + 1;
    let w_out = (w + 2 * pw - kw) / sw + 1;
    let mut y = vec![0f32; n * c * h_out * w_out];
    for ni in 0..n {
        for ci in 0..c {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut best = f32::NEG_INFINITY;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let hi = ho * sh + ki;
                            let wi = wo * sw + kj;
                            if hi < ph || wi < pw {
                                continue;
                            }
                            let hi = hi - ph;
                            let wi = wi - pw;
                            if hi >= h || wi >= w {
                                continue;
                            }
                            let v = x[((ni * c) + ci) * h * w + hi * w + wi];
                            if v > best {
                                best = v;
                            }
                        }
                    }
                    y[((ni * c) + ci) * h_out * w_out + ho * w_out + wo] = best;
                }
            }
        }
    }
    (y, h_out, w_out)
}

#[allow(clippy::too_many_arguments)]
fn maxpool_backward_ref(
    x: &[f32],
    dy: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> Vec<f32> {
    let mut dx = vec![0f32; n * c * h * w];
    for ni in 0..n {
        for ci in 0..c {
            let in_chan = (ni * c + ci) * h * w;
            let out_chan = (ni * c + ci) * h_out * w_out;
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut best_v = f32::NEG_INFINITY;
                    let mut best_idx: Option<usize> = None;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let hi = ho * sh + ki;
                            let wi = wo * sw + kj;
                            if hi < ph || wi < pw {
                                continue;
                            }
                            let hi = hi - ph;
                            let wi = wi - pw;
                            if hi >= h || wi >= w {
                                continue;
                            }
                            let idx = in_chan + hi * w + wi;
                            let v = x[idx];
                            // First hit wins (strict `>`).
                            if v > best_v {
                                best_v = v;
                                best_idx = Some(idx);
                            }
                        }
                    }
                    if let Some(idx) = best_idx {
                        dx[idx] += dy[out_chan + ho * w_out + wo];
                    }
                }
            }
        }
    }
    dx
}

#[allow(clippy::too_many_arguments)]
fn check_maxpool_backward(
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) {
    // Use a non-monotonic input so argmax positions are interesting.
    let xs: Vec<f32> = (0..n * c * h * w)
        .map(|i| {
            let f = (i as f32) * 0.137 + (i as f32).sin() * 2.5;
            // Throw in a few duplicates so the tie-breaking path gets exercised.
            if i % 7 == 0 { 1.0 } else { f }
        })
        .collect();
    let (_y, h_out, w_out) = maxpool_forward_ref(&xs, n, c, h, w, kh, kw, sh, sw, ph, pw);
    let dys: Vec<f32> = (0..n * c * h_out * w_out)
        .map(|i| 0.13 * (i as f32) - 0.5)
        .collect();

    let mut g = Graph::new("pool_bwd");
    let x = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
    let dy = g.input("dy", Shape::new(&[n, c, h_out, w_out], DType::F32));
    let dx = g.maxpool2d_backward(x, dy, vec![kh, kw], vec![sh, sw], vec![ph, pw]);
    g.set_outputs(vec![dx]);

    let want = maxpool_backward_ref(&xs, &dys, n, c, h, w, h_out, w_out, kh, kw, sh, sw, ph, pw);
    let got = run(g, &[("x", &xs), ("dy", &dys)]);
    assert!(
        close(&got, &want, 1e-5),
        "MaxPool2dBackward @ shape (n={n}, c={c}, h={h}, w={w}, k={kh}x{kw}, \
         s={sh}x{sw}, p={ph}x{pw}): got {got:?} want {want:?}"
    );
}

#[test]
fn maxpool_backward_2x2_stride_2_no_padding() {
    check_maxpool_backward(2, 3, 4, 4, 2, 2, 2, 2, 0, 0);
}

#[test]
fn maxpool_backward_3x3_stride_1_no_padding() {
    check_maxpool_backward(1, 2, 5, 5, 3, 3, 1, 1, 0, 0);
}

#[test]
fn maxpool_backward_3x3_stride_2_padding_1() {
    check_maxpool_backward(2, 2, 6, 6, 3, 3, 2, 2, 1, 1);
}

#[test]
fn maxpool_backward_overlapping_2x2_stride_1() {
    // stride < kernel — multiple windows can pick the same input
    // position; scatter-add must accumulate (not overwrite).
    check_maxpool_backward(1, 1, 4, 4, 2, 2, 1, 1, 0, 0);
}

#[test]
fn maxpool_backward_kernel_1x1() {
    // Trivial 1x1 pool: backward should pass dy straight through
    // (each output position has exactly one input contributor).
    check_maxpool_backward(2, 3, 4, 4, 1, 1, 1, 1, 0, 0);
}

#[test]
fn maxpool_backward_padding_only_2x2() {
    check_maxpool_backward(1, 2, 3, 3, 2, 2, 1, 1, 1, 1);
}

#[test]
fn end_to_end_grad_through_maxpool_layer() {
    // Forward: x → maxpool(2x2, s=2) → mean → loss.
    // Gradient w.r.t. x should match a hand-computed reference.
    let n = 2usize;
    let c = 2usize;
    let h = 4usize;
    let w = 4usize;
    let kh = 2usize;
    let kw = 2usize;
    let sh = 2usize;
    let sw = 2usize;
    let h_out = h / sh;
    let w_out = w / sw;

    let mut fwd = Graph::new("pool_only");
    let x = fwd.input("x", Shape::new(&[n, c, h, w], DType::F32));
    let p = fwd.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![kh, kw],
            stride: vec![sh, sw],
            padding: vec![0, 0],
        },
        vec![x],
        Shape::new(&[n, c, h_out, w_out], DType::F32),
    );
    let loss = fwd.add_node(
        Op::Reduce {
            op: ReduceOp::Mean,
            axes: vec![0, 1, 2, 3],
            keep_dim: false,
        },
        vec![p],
        Shape::new(&[], DType::F32),
    );
    fwd.set_outputs(vec![loss]);

    // We want grad w.r.t. x. Wrap x as a "param" so grad_with_loss
    // can target it (the autodiff API takes `wrt: &[NodeId]`).
    // Actually, NodeIds for Inputs work too — grad_with_loss treats
    // any NodeId as a leaf to propagate gradient to.
    let bwd = rlx_opt::autodiff::grad_with_loss(&fwd, &[x]);

    let xs: Vec<f32> = (0..n * c * h * w)
        .map(|i| (i as f32) * 0.21 + ((i * i) as f32 * 0.013).sin())
        .collect();
    let d_out = vec![1.0f32];

    let mut exe = MlxExecutable::compile(bwd);
    let outs = exe.run(&[("x", &xs), ("d_output", &d_out)]);
    assert_eq!(outs.len(), 2, "expected [loss, dx]");

    // Reference: forward maxpool then mean-of-output.
    let (yv, _, _) = maxpool_forward_ref(&xs, n, c, h, w, kh, kw, sh, sw, 0, 0);
    let total = (n * c * h_out * w_out) as f32;
    let loss_ref: f32 = yv.iter().sum::<f32>() / total;
    // d_y = (1/total) · d_out for each output element
    let dy_ref: Vec<f32> = (0..yv.len()).map(|_| d_out[0] / total).collect();
    let want_dx =
        maxpool_backward_ref(&xs, &dy_ref, n, c, h, w, h_out, w_out, kh, kw, sh, sw, 0, 0);

    assert!(
        close(&outs[0], &[loss_ref], 1e-5),
        "loss: got {:?} want {loss_ref}",
        outs[0]
    );
    assert!(
        close(&outs[1], &want_dx, 1e-5),
        "dx: got {:?} want {want_dx:?}",
        outs[1]
    );
}
