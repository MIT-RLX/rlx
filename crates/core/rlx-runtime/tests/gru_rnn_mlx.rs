// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Op::Gru` / `Op::Rnn` (carry = false) native on-device on MLX, checked for
//! numerical parity against the CPU reference kernels (`execute_gru_f32` /
//! `execute_rnn_f32`). Native = unrolled MLX ops, no CPU host-eval.
#![cfg(all(feature = "cpu", feature = "mlx"))]
use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};

fn mk(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i.wrapping_mul(2654435761).wrapping_add(seed)) % 1000) as f32) / 500.0 - 1.0)
        .collect()
}

// `(w_ih, w_hh, bias)` element counts from the canonical sizing helper, so the
// test can't under-size a weight (batch/seq don't affect these three).
fn dims(
    _b: usize,
    _s: usize,
    inp: usize,
    h: usize,
    layers: usize,
    bidir: bool,
    gates: usize,
) -> (usize, usize, usize) {
    let ex = rlx_cpu::thunk::rnn_expected_lens(gates, 1, 1, inp, h, layers, bidir);
    (ex.w_ih, ex.w_hh, ex.bias)
}

fn maxd(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn run_gru(b: usize, s: usize, inp: usize, h: usize, layers: usize, bidir: bool) {
    if !is_available(Device::Mlx) {
        eprintln!("skip: no MLX device");
        return;
    }
    let f = DType::F32;
    let dirs = if bidir { 2 } else { 1 };
    let (wt, ht, bt) = dims(b, s, inp, h, layers, bidir, 3);
    let (xd, wih, whh, bih, bhh) = (
        mk(b * s * inp, 1),
        mk(wt, 2),
        mk(ht, 3),
        mk(bt, 4),
        mk(bt, 5),
    );
    let build = || {
        let mut g = Graph::new("gru");
        let x = g.input("x", Shape::new(&[b, s, inp], f));
        let a = g.input("w_ih", Shape::new(&[wt], f));
        let c = g.input("w_hh", Shape::new(&[ht], f));
        let d = g.input("b_ih", Shape::new(&[bt], f));
        let e = g.input("b_hh", Shape::new(&[bt], f));
        let y = g.add_node(
            Op::Gru {
                hidden_size: h,
                num_layers: layers,
                bidirectional: bidir,
                carry: false,
            },
            vec![x, a, c, d, e],
            Shape::new(&[b, s, dirs * h], f),
        );
        g.set_outputs(vec![y]);
        g
    };
    let feed: Vec<(&str, &[f32])> = vec![
        ("x", &xd),
        ("w_ih", &wih),
        ("w_hh", &whh),
        ("b_ih", &bih),
        ("b_hh", &bhh),
    ];
    let cpu = Session::new(Device::Cpu)
        .compile(build())
        .run(&feed)
        .pop()
        .unwrap();
    let mlx = Session::new(Device::Mlx)
        .compile(build())
        .run(&feed)
        .pop()
        .unwrap();
    let d = maxd(&cpu, &mlx);
    println!("GRU {b}x{s}x{inp} h{h} l{layers} bi{bidir}  max|delta|={d:.3e}");
    assert!(d < 1e-4, "GRU MLX vs CPU maxd={d}");
}

fn run_rnn(b: usize, s: usize, inp: usize, h: usize, layers: usize, bidir: bool, relu: bool) {
    if !is_available(Device::Mlx) {
        eprintln!("skip: no MLX device");
        return;
    }
    let f = DType::F32;
    let dirs = if bidir { 2 } else { 1 };
    let (wt, ht, bt) = dims(b, s, inp, h, layers, bidir, 1);
    let (xd, wih, whh, bias) = (mk(b * s * inp, 1), mk(wt, 2), mk(ht, 3), mk(bt, 4));
    let build = || {
        let mut g = Graph::new("rnn");
        let x = g.input("x", Shape::new(&[b, s, inp], f));
        let a = g.input("w_ih", Shape::new(&[wt], f));
        let c = g.input("w_hh", Shape::new(&[ht], f));
        let d = g.input("bias", Shape::new(&[bt], f));
        let y = g.add_node(
            Op::Rnn {
                hidden_size: h,
                num_layers: layers,
                bidirectional: bidir,
                carry: false,
                relu,
            },
            vec![x, a, c, d],
            Shape::new(&[b, s, dirs * h], f),
        );
        g.set_outputs(vec![y]);
        g
    };
    let feed: Vec<(&str, &[f32])> =
        vec![("x", &xd), ("w_ih", &wih), ("w_hh", &whh), ("bias", &bias)];
    let cpu = Session::new(Device::Cpu)
        .compile(build())
        .run(&feed)
        .pop()
        .unwrap();
    let mlx = Session::new(Device::Mlx)
        .compile(build())
        .run(&feed)
        .pop()
        .unwrap();
    let d = maxd(&cpu, &mlx);
    println!(
        "RNN({}) {b}x{s}x{inp} h{h} l{layers} bi{bidir}  max|delta|={d:.3e}",
        if relu { "relu" } else { "tanh" }
    );
    assert!(d < 1e-4, "RNN MLX vs CPU maxd={d}");
}

#[test]
fn gru_single() {
    run_gru(2, 5, 4, 4, 1, false);
}

#[test]
fn gru_multi_layer_bidirectional() {
    run_gru(2, 6, 5, 4, 2, true);
}

#[test]
fn rnn_tanh_bidirectional() {
    run_rnn(2, 5, 4, 4, 1, true, false);
}

#[test]
fn rnn_relu_multi_layer() {
    run_rnn(1, 6, 5, 4, 3, false, true);
}
