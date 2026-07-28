// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! fp16 compute for the native LSTM / GRU / RNN unrolls on MLX (Metal GPU).
//! `RLX_MLX_RNN_F16` opts the recurrence into fp16 (matmuls accumulate in f32);
//! inputs/outputs stay f32. Verified against the f32 CPU reference within fp16
//! tolerance. Needs `--features mlx`.
#![cfg(feature = "mlx")]

use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn mk(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i.wrapping_mul(2654435761).wrapping_add(seed)) % 1000) as f32) / 500.0 - 1.0)
        .collect()
}

fn maxd(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// CPU (f32 reference) vs MLX with the fp16 compute path enabled.
fn f16_vs_cpu(label: &str, build: impl Fn() -> Graph, feed: &[(&str, &[f32])]) {
    let cpu = Session::new(Device::Cpu)
        .compile(build())
        .run(feed)
        .remove(0);
    // Opt the MLX recurrence into fp16 for this process.
    unsafe { std::env::set_var("RLX_MLX_RNN_F16", "1") };
    let mlx = Session::new(Device::Mlx)
        .compile(build())
        .run(feed)
        .remove(0);
    let d = maxd(&cpu, &mlx);
    println!("{label} MLX-f16 vs CPU-f32: max|Δ|={d:.4}");
    // fp16 precision: comfortably tighter than 1e-2, but far looser than f32.
    assert!(d < 1e-2, "{label} fp16 vs f32 exceeds tolerance: {d}");
    assert!(
        d > 1e-6,
        "{label} delta {d} looks like f32 — fp16 path may be inactive"
    );
}

const CFG: (usize, usize, usize, usize, usize, bool) = (2, 5, 6, 8, 1, true);

#[test]
fn lstm_mlx_f16() {
    let (b, s, inp, h, l, bi) = CFG;
    let f = DType::F32;
    let ex = rlx_cpu::thunk::rnn_expected_lens(4, b, s, inp, h, l, bi);
    let d = if bi { 2 } else { 1 };
    let (xd, wih, whh, bias) = (mk(ex.x, 1), mk(ex.w_ih, 2), mk(ex.w_hh, 3), mk(ex.bias, 4));
    let build = || {
        let mut g = Graph::new("lstm");
        let x = g.input("x", Shape::new(&[b, s, inp], f));
        let a = g.input("w_ih", Shape::new(&[ex.w_ih], f));
        let c = g.input("w_hh", Shape::new(&[ex.w_hh], f));
        let e = g.input("bias", Shape::new(&[ex.bias], f));
        let y = g.add_node(
            Op::Lstm {
                hidden_size: h,
                num_layers: l,
                bidirectional: bi,
                carry: false,
            },
            vec![x, a, c, e],
            Shape::new(&[b, s, d * h], f),
        );
        g.set_outputs(vec![y]);
        g
    };
    f16_vs_cpu(
        "LSTM",
        build,
        &[("x", &xd), ("w_ih", &wih), ("w_hh", &whh), ("bias", &bias)],
    );
}

#[test]
fn gru_mlx_f16() {
    let (b, s, inp, h, l, bi) = CFG;
    let f = DType::F32;
    let ex = rlx_cpu::thunk::rnn_expected_lens(3, b, s, inp, h, l, bi);
    let d = if bi { 2 } else { 1 };
    let (xd, wih, whh, bih, bhh) = (
        mk(ex.x, 1),
        mk(ex.w_ih, 2),
        mk(ex.w_hh, 3),
        mk(ex.bias, 4),
        mk(ex.bias, 5),
    );
    let build = || {
        let mut g = Graph::new("gru");
        let x = g.input("x", Shape::new(&[b, s, inp], f));
        let a = g.input("w_ih", Shape::new(&[ex.w_ih], f));
        let c = g.input("w_hh", Shape::new(&[ex.w_hh], f));
        let di = g.input("b_ih", Shape::new(&[ex.bias], f));
        let dh = g.input("b_hh", Shape::new(&[ex.bias], f));
        let y = g.add_node(
            Op::Gru {
                hidden_size: h,
                num_layers: l,
                bidirectional: bi,
                carry: false,
            },
            vec![x, a, c, di, dh],
            Shape::new(&[b, s, d * h], f),
        );
        g.set_outputs(vec![y]);
        g
    };
    f16_vs_cpu(
        "GRU",
        build,
        &[
            ("x", &xd),
            ("w_ih", &wih),
            ("w_hh", &whh),
            ("b_ih", &bih),
            ("b_hh", &bhh),
        ],
    );
}

#[test]
fn rnn_mlx_f16() {
    let (b, s, inp, h, l, bi) = CFG;
    let f = DType::F32;
    let ex = rlx_cpu::thunk::rnn_expected_lens(1, b, s, inp, h, l, bi);
    let d = if bi { 2 } else { 1 };
    let (xd, wih, whh, bias) = (mk(ex.x, 1), mk(ex.w_ih, 2), mk(ex.w_hh, 3), mk(ex.bias, 4));
    let build = || {
        let mut g = Graph::new("rnn");
        let x = g.input("x", Shape::new(&[b, s, inp], f));
        let a = g.input("w_ih", Shape::new(&[ex.w_ih], f));
        let c = g.input("w_hh", Shape::new(&[ex.w_hh], f));
        let e = g.input("bias", Shape::new(&[ex.bias], f));
        let y = g.add_node(
            Op::Rnn {
                hidden_size: h,
                num_layers: l,
                bidirectional: bi,
                carry: false,
                relu: false,
            },
            vec![x, a, c, e],
            Shape::new(&[b, s, d * h], f),
        );
        g.set_outputs(vec![y]);
        g
    };
    f16_vs_cpu(
        "RNN",
        build,
        &[("x", &xd), ("w_ih", &wih), ("w_hh", &whh), ("bias", &bias)],
    );
}
