// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The conv weight gradient is a cross-batch reduction. The parallel path
//! used to add per-chunk partial sums under a lock in finishing order, so the
//! f32 result could change from run to run. It now adds them in chunk order:
//! repeated runs are bit-identical, and the sum matches a host f64 reference.
//! EEGNet-like geometry (a temporal conv over 8 × 128 trials, batch 64).

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const B: usize = 64;
const C: usize = 8;
const T: usize = 128;
const O: usize = 16;
const K: usize = 32;
const P: usize = 16;

fn val(i: usize, a: usize, m: usize) -> f32 {
    (((i * a) % m) as f32 / m as f32) - 0.5
}

#[test]
fn the_conv_weight_gradient_is_deterministic_and_correct() {
    let f = DType::F32;
    let t_out = T + 2 * P - K + 1;
    let mut g = Graph::new("conv_dw");
    let x = g.input("x", Shape::new(&[B, 1, C, T], f));
    let w = g.input("w", Shape::new(&[O, 1, 1, K], f));
    let y = g.conv2d(x, w, [1, K], [1, 1], [0, P], [1, 1], 1); // [B,O,C,t_out]
    let cdat: Vec<f32> = (0..B * O * C * t_out).map(|i| val(i, 37, 101)).collect();
    let cb: Vec<u8> = cdat.iter().flat_map(|v| v.to_le_bytes()).collect();
    let c = g.add_node(
        Op::Constant { data: cb },
        vec![],
        Shape::new(&[B, O, C, t_out], f),
    );
    let cy = g.mul(c, y);
    let flat = g.reshape_(cy, vec![(B * O * C * t_out) as i64]);
    let loss = g.reduce(flat, ReduceOp::Sum, vec![0], false, Shape::new(&[1], f));
    g.set_outputs(vec![loss]);
    let bwd = rlx_autodiff::grad_with_loss(&g, &[w]);

    let xd: Vec<f32> = (0..B * C * T).map(|i| val(i, 13, 97)).collect();
    let wd: Vec<f32> = (0..O * K).map(|i| val(i, 7, 23)).collect();
    let mut s = Session::new(Device::Cpu).compile(bwd);
    let first = s
        .run(&[("x", &xd), ("w", &wd), ("d_output", &[1.0])])
        .pop()
        .unwrap();
    for run in 1..10 {
        let again = s
            .run(&[("x", &xd), ("w", &wd), ("d_output", &[1.0])])
            .pop()
            .unwrap();
        let same = first
            .iter()
            .zip(&again)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "run {run}: the weight gradient changed bits");
    }
    // dW[o,k] = Σ_b Σ_h Σ_t c[b,o,h,t] · x[b,h,t+k−P]
    let mut want = vec![0f64; O * K];
    for o in 0..O {
        for k in 0..K {
            let mut acc = 0f64;
            for b in 0..B {
                for h in 0..C {
                    for t in 0..t_out {
                        let ti = t as isize + k as isize - P as isize;
                        if ti >= 0 && (ti as usize) < T {
                            acc += cdat[((b * O + o) * C + h) * t_out + t] as f64
                                * xd[(b * C + h) * T + ti as usize] as f64;
                        }
                    }
                }
            }
            want[o * K + k] = acc;
        }
    }
    let scale = want.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let err = first
        .iter()
        .zip(&want)
        .map(|(&a, &b)| (a as f64 - b).abs())
        .fold(0.0, f64::max)
        / scale;
    assert!(err < 1e-5, "dW rel error {err}");
}
