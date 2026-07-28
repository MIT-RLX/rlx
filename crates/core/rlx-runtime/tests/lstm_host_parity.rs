// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Op::Lstm` on MLX vs CPU. For `carry = false` (incl. bidirectional /
//! multi-layer) MLX runs the *native* on-device unroll (`native_lstm`), so
//! this is a numerical-parity check against the CPU `execute_lstm_f32`
//! reference (not bit-exact — different float accumulation order). Deltas
//! observed ≤ ~2e-6 even at the Kokoro H=256 shape.
#![cfg(all(feature = "cpu", feature = "mlx"))]
use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};

fn mk(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i.wrapping_mul(2654435761).wrapping_add(seed)) % 1000) as f32) / 500.0 - 1.0)
        .collect()
}

fn build(b: usize, s: usize, inp: usize, h: usize, bidirectional: bool) -> Graph {
    let f = DType::F32;
    let dirs = if bidirectional { 2 } else { 1 };
    let mut g = Graph::new("lstm_parity");
    let x = g.input("x", Shape::new(&[b, s, inp], f));
    let wih = g.input("w_ih", Shape::new(&[dirs * 4 * h * inp], f));
    let whh = g.input("w_hh", Shape::new(&[dirs * 4 * h * h], f));
    let bias = g.input("bias", Shape::new(&[dirs * 4 * h], f));
    let out = g.add_node(
        Op::Lstm {
            hidden_size: h,
            num_layers: 1,
            bidirectional,
            carry: false,
        },
        vec![x, wih, whh, bias],
        Shape::new(&[b, s, dirs * h], f),
    );
    g.set_outputs(vec![out]);
    g
}

fn run_parity(bidirectional: bool) {
    if !is_available(Device::Mlx) {
        eprintln!("skip: no MLX device");
        return;
    }
    let (b, s, inp, h) = (2usize, 9usize, 6usize, 6usize);
    let dirs = if bidirectional { 2 } else { 1 };
    let xd = mk(b * s * inp, 1);
    let wihd = mk(dirs * 4 * h * inp, 2);
    let whhd = mk(dirs * 4 * h * h, 3);
    let bd = mk(dirs * 4 * h, 4);
    let slots: [(&str, &[f32]); 4] = [("x", &xd), ("w_ih", &wihd), ("w_hh", &whhd), ("bias", &bd)];
    let run = |dev| {
        let mut c = Session::new(dev).compile(build(b, s, inp, h, bidirectional));
        c.run(&slots).pop().unwrap()
    };
    let cpu = run(Device::Cpu);
    let mlx = run(Device::Mlx);
    let maxd = cpu
        .iter()
        .zip(&mlx)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let label = if bidirectional { "BiLSTM" } else { "LSTM" };
    println!("{label} CPU-vs-MLX  max|delta|={maxd:.3e}");
    assert!(
        maxd < 1e-5,
        "MLX host-eval {label} must match CPU, got {maxd}"
    );
}

#[test]
fn lstm_mlx_matches_cpu() {
    run_parity(false);
}

#[test]
fn bilstm_mlx_matches_cpu() {
    run_parity(true);
}

/// Multi-layer bidirectional — exercises the native `wih_cursor` per-layer
/// weight offsets (layer 0 reads `input_size`, later layers read `D*hidden`).
#[test]
fn multilayer_bilstm_mlx_matches_cpu() {
    if !is_available(Device::Mlx) {
        eprintln!("skip: no MLX device");
        return;
    }
    let f = DType::F32;
    let (b, s, inp, h, layers) = (2usize, 7usize, 5usize, 4usize, 3usize);
    let dirs = 2usize;
    // Flat w_ih packs, per layer, `dirs` blocks of `[4h, in_l]` where
    // in_l = inp for layer 0, else dirs*h.
    let in_l = |l: usize| if l == 0 { inp } else { dirs * h };
    let wih_total: usize = (0..layers).map(|l| dirs * 4 * h * in_l(l)).sum();
    let whh_total = layers * dirs * 4 * h * h;
    let bias_total = layers * dirs * 4 * h;

    let xd = mk(b * s * inp, 21);
    let wihd = mk(wih_total, 22);
    let whhd = mk(whh_total, 23);
    let bd = mk(bias_total, 24);

    let mut g = Graph::new("lstm_ml");
    let x = g.input("x", Shape::new(&[b, s, inp], f));
    let wih = g.input("w_ih", Shape::new(&[wih_total], f));
    let whh = g.input("w_hh", Shape::new(&[whh_total], f));
    let bias = g.input("bias", Shape::new(&[bias_total], f));
    let out = g.add_node(
        Op::Lstm {
            hidden_size: h,
            num_layers: layers,
            bidirectional: true,
            carry: false,
        },
        vec![x, wih, whh, bias],
        Shape::new(&[b, s, dirs * h], f),
    );
    g.set_outputs(vec![out]);

    let slots: [(&str, &[f32]); 4] = [("x", &xd), ("w_ih", &wihd), ("w_hh", &whhd), ("bias", &bd)];
    let run = |dev| {
        let mut c = Session::new(dev).compile(g.clone());
        c.run(&slots).pop().unwrap()
    };
    let cpu = run(Device::Cpu);
    let mlx = run(Device::Mlx);
    let maxd = cpu
        .iter()
        .zip(&mlx)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("3-layer BiLSTM CPU-vs-MLX  max|delta|={maxd:.3e}");
    assert!(maxd < 1e-4, "multi-layer BiLSTM MLX vs CPU maxd={maxd}");
}

/// Kokoro StyleTTS2 encoder shape: H=256, bidirectional, B=1.
#[test]
fn bilstm_kokoro_shape_mlx_matches_cpu() {
    if !is_available(Device::Mlx) {
        eprintln!("skip: no MLX device");
        return;
    }
    let (b, s, inp, h) = (1usize, 17usize, 128usize, 256usize);
    let dirs = 2usize;
    let xd = mk(b * s * inp, 11);
    let wihd = mk(dirs * 4 * h * inp, 12);
    let whhd = mk(dirs * 4 * h * h, 13);
    let bd = mk(dirs * 4 * h, 14);
    let slots: [(&str, &[f32]); 4] = [("x", &xd), ("w_ih", &wihd), ("w_hh", &whhd), ("bias", &bd)];
    let run = |dev| {
        let mut c = Session::new(dev).compile(build(b, s, inp, h, true));
        c.run(&slots).pop().unwrap()
    };
    let cpu = run(Device::Cpu);
    let mlx = run(Device::Mlx);
    let maxd = cpu
        .iter()
        .zip(&mlx)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("Kokoro-shape BiLSTM CPU-vs-MLX  max|delta|={maxd:.3e}");
    assert!(maxd < 1e-5, "Kokoro-shape BiLSTM MLX vs CPU maxd={maxd}");
}
