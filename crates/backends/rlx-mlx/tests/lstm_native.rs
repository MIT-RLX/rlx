// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Proof that `Op::Lstm` runs *natively* on MLX (not via the CPU
//! host-eval fallback):
//!  - `carry = false` (incl. bidirectional) stays in `MlxMode::Compiled`
//!    with `compile_disabled_reason() == None` — i.e. the whole recurrence
//!    traces cleanly through `mlx::compile` and runs on the Metal device.
//!  - `carry = true` correctly *does* fall back (host `to_f32` write-back),
//!    so it reports a host-eval reason and drops to Lazy.

#![cfg(rlx_mlx_host)]

use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_mlx::{MlxExecutable, MlxMode};

fn mk(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i.wrapping_mul(2654435761).wrapping_add(seed)) % 1000) as f32) / 500.0 - 1.0)
        .collect()
}

/// Bidirectional single-layer LSTM, `carry = false`.
#[test]
fn bilstm_runs_natively_in_compiled_mode() {
    let f = DType::F32;
    let (b, s, inp, h) = (1usize, 6usize, 8usize, 8usize);
    let dirs = 2usize;

    let mut g = Graph::new("bilstm_native");
    let x = g.input("x", Shape::new(&[b, s, inp], f));
    // ONNX import preserves recurrent weights as matrices; native LSTM must
    // accept those as well as the historical packed rank-1 representation.
    let wih = g.input("w_ih", Shape::new(&[dirs * 4 * h, inp], f));
    let whh = g.input("w_hh", Shape::new(&[dirs * 4 * h, h], f));
    let bias = g.input("bias", Shape::new(&[dirs * 4 * h], f));
    let out = g.add_node(
        Op::Lstm {
            hidden_size: h,
            num_layers: 1,
            bidirectional: true,
            carry: false,
        },
        vec![x, wih, whh, bias],
        Shape::new(&[b, s, dirs * h], f),
    );
    g.set_outputs(vec![out]);

    let mut exe = MlxExecutable::compile_with_mode(g, MlxMode::Compiled);
    // Build the mlx::compile trace up front; a host-eval op would set the
    // fallback reason here.
    exe.warm_compile().expect("warm_compile");
    assert_eq!(
        exe.compile_disabled_reason(),
        None,
        "BiLSTM must trace natively through mlx::compile (no host-eval fallback), \
         got reason: {:?}",
        exe.compile_disabled_reason()
    );

    let xd = mk(b * s * inp, 1);
    let wihd = mk(dirs * 4 * h * inp, 2);
    let whhd = mk(dirs * 4 * h * h, 3);
    let bd = mk(dirs * 4 * h, 4);
    let out = exe
        .run(&[("x", &xd), ("w_ih", &wihd), ("w_hh", &whhd), ("bias", &bd)])
        .pop()
        .unwrap();
    assert_eq!(out.len(), b * s * dirs * h);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "native BiLSTM output must be finite"
    );
    // Still native after a real run.
    assert_eq!(exe.compile_disabled_reason(), None);
}

/// `carry = true` needs functional state write-back MLX can't express in
/// place, so it host-evals and drops out of Compiled mode.
#[test]
fn carry_lstm_falls_back_to_host_eval() {
    let f = DType::F32;
    let (b, s, inp, h) = (1usize, 6usize, 8usize, 8usize);

    let mut g = Graph::new("lstm_carry");
    let x = g.input("x", Shape::new(&[b, s, inp], f));
    let wih = g.input("w_ih", Shape::new(&[4 * h * inp], f));
    let whh = g.input("w_hh", Shape::new(&[4 * h * h], f));
    let bias = g.input("bias", Shape::new(&[4 * h], f));
    let h0 = g.input("h0", Shape::new(&[1, b, h], f));
    let c0 = g.input("c0", Shape::new(&[1, b, h], f));
    let out = g.add_node(
        Op::Lstm {
            hidden_size: h,
            num_layers: 1,
            bidirectional: false,
            carry: true,
        },
        vec![x, wih, whh, bias, h0, c0],
        Shape::new(&[b, s, h], f),
    );
    g.set_outputs(vec![out]);

    let mut exe = MlxExecutable::compile_with_mode(g, MlxMode::Compiled);
    exe.warm_compile().expect("warm_compile");
    assert!(
        exe.compile_disabled_reason().is_some(),
        "carry=true LSTM must fall back to host-eval (Lazy)"
    );
}
