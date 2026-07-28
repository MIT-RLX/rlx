// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multiple *distinct* compiled graphs coexisting in one process.
//!
//! MLX's `compile` keys its trace cache by a `fun_id` derived from the
//! address of the boxed `std::function` we hand it. Each `CompiledFn`
//! owns a distinct box, so distinct rlx graphs must get distinct
//! `fun_id`s and never replay each other's trace. This test builds two
//! structurally different graphs (different ops AND different output
//! shapes) and runs them interleaved, asserting each keeps producing
//! its own result.

#![cfg(rlx_mlx_host)]

use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::rf::const_f32;
use rlx_ir::{DType, Graph, Shape};
use rlx_mlx::{MlxExecutable, MlxMode};

fn s4() -> Shape {
    Shape::new(&[4], DType::F32)
}
fn s1() -> Shape {
    Shape::new(&[1], DType::F32)
}

fn close(got: &[f32], want: &[f32], tol: f32) -> (bool, f32) {
    let max_abs = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    (got.len() == want.len() && max_abs <= tol, max_abs)
}

/// Graph A: elementwise `relu(x * 2 + y)`, output shape [4].
fn graph_a() -> Graph {
    let mut g = Graph::new("multi_a");
    let x = g.input("x", s4());
    let y = g.input("y", s4());
    let two = const_f32(&mut g, 2.0, s1());
    let scaled = g.binary(BinaryOp::Mul, x, two, s4());
    let sum = g.binary(BinaryOp::Add, scaled, y, s4());
    let out = g.activation(Activation::Relu, sum, s4());
    g.set_outputs(vec![out]);
    g
}

fn eval_a(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter()
        .zip(y)
        .map(|(a, b)| (a * 2.0 + b).max(0.0))
        .collect()
}

/// Graph B: `sum(x - y)`, output shape [1] — deliberately a *different*
/// shape and op set so a trace collision with A is unmistakable.
fn graph_b() -> Graph {
    let mut g = Graph::new("multi_b");
    let x = g.input("x", s4());
    let y = g.input("y", s4());
    let diff = g.binary(BinaryOp::Sub, x, y, s4());
    let out = g.reduce(diff, ReduceOp::Sum, vec![0], true, s1());
    g.set_outputs(vec![out]);
    g
}

fn eval_b(x: &[f32], y: &[f32]) -> Vec<f32> {
    vec![x.iter().zip(y).map(|(a, b)| a - b).sum()]
}

#[test]
fn two_distinct_compiled_graphs_do_not_collide() {
    let xa = [1.0f32, -2.0, 3.0, -4.0];
    let ya = [0.5f32, 0.5, -10.0, 8.0];
    let xb = [10.0f32, 20.0, 30.0, 40.0];
    let yb = [1.0f32, 2.0, 3.0, 4.0];

    let mut a = MlxExecutable::compile_with_mode(graph_a(), MlxMode::Compiled);
    let mut b = MlxExecutable::compile_with_mode(graph_b(), MlxMode::Compiled);

    // Interleave: A, B, A, B — force both compiled traces to be live and
    // exercised alternately on the same thread.
    for round in 0..3 {
        let out_a = a.run(&[("x", &xa), ("y", &ya)]).into_iter().next().unwrap();
        let out_b = b.run(&[("x", &xb), ("y", &yb)]).into_iter().next().unwrap();

        let want_a = eval_a(&xa, &ya);
        let want_b = eval_b(&xb, &yb);

        let (ok_a, m_a) = close(&out_a, &want_a, 1e-4);
        let (ok_b, m_b) = close(&out_b, &want_b, 1e-4);

        assert!(
            ok_a,
            "round {round}: graph A wrong. want={want_a:?} got={out_a:?} max_abs={m_a:.3e}"
        );
        assert!(
            ok_b,
            "round {round}: graph B wrong (shape/values). want={want_b:?} got={out_b:?} max_abs={m_b:.3e}"
        );
    }
}

/// Compile → run → drop, repeatedly, with a *different* graph each time.
/// This is the MLX `fun_id`-reuse footgun: MLX keys its trace cache by the
/// heap address of the boxed callback, and a freed box's address is prime
/// for reuse by the next compile. If the freed graph's cache entry isn't
/// erased on drop, the next graph replays a stale trace. The C++ `compile`
/// path erases via the shared_ptr deleter — this proves it end-to-end.
#[test]
fn compile_drop_recompile_reuses_address_without_stale_trace() {
    let x = [1.0f32, 2.0, 3.0, 4.0];
    for i in 0..12u32 {
        // Distinct affine graph per iteration: relu(x*scale + bias).
        let scale = (i as f32) + 1.0;
        let bias = -(i as f32) * 3.0;
        let mut g = Graph::new(format!("affine_{i}"));
        let xn = g.input("x", s4());
        let sc = const_f32(&mut g, scale, s1());
        let bi = const_f32(&mut g, bias, s1());
        let scaled = g.binary(BinaryOp::Mul, xn, sc, s4());
        let shifted = g.binary(BinaryOp::Add, scaled, bi, s4());
        let out = g.activation(Activation::Relu, shifted, s4());
        g.set_outputs(vec![out]);

        // Build, run, then let `exec` drop at end of iteration so the next
        // compile can reclaim its address.
        let mut exec = MlxExecutable::compile_with_mode(g, MlxMode::Compiled);
        let got = exec.run(&[("x", &x)]).into_iter().next().unwrap();
        let want: Vec<f32> = x.iter().map(|v| (v * scale + bias).max(0.0)).collect();
        let (ok, m) = close(&got, &want, 1e-4);
        assert!(
            ok,
            "iter {i}: stale-trace suspected. scale={scale} bias={bias} want={want:?} got={got:?} max_abs={m:.3e}"
        );
    }
}

/// Build many distinct compiled graphs and keep them all live, then run
/// each — models the runtime `BucketedCompileCache` holding one compiled
/// MLX graph per sequence-length bucket.
#[test]
fn many_live_compiled_graphs_each_replay_their_own_trace() {
    let mut execs: Vec<(MlxExecutable, f32)> = Vec::new();
    // Each graph adds a distinct bias constant, so a collision surfaces as
    // the wrong bias in the output.
    for i in 0..8u32 {
        let bias = i as f32 * 100.0 + 7.0;
        let mut g = Graph::new(format!("bias_{i}"));
        let x = g.input("x", s4());
        let c = const_f32(&mut g, bias, s1());
        let out = g.binary(BinaryOp::Add, x, c, s4());
        g.set_outputs(vec![out]);
        execs.push((MlxExecutable::compile_with_mode(g, MlxMode::Compiled), bias));
    }

    let x = [1.0f32, 2.0, 3.0, 4.0];
    for (exec, bias) in execs.iter_mut() {
        let out = exec.run(&[("x", &x)]).into_iter().next().unwrap();
        let want: Vec<f32> = x.iter().map(|v| v + *bias).collect();
        let (ok, m) = close(&out, &want, 1e-4);
        assert!(ok, "bias {bias}: want={want:?} got={out:?} max_abs={m:.3e}");
    }
}
