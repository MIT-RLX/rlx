// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Transpose(operand) → MatMul` folds into GEMM trans-flags, and the folded
//! transpose must not reserve arena either.
//!
//! Matmul backward emits this shape once per weight — `dA = g·Bᵀ`, `dB = Aᵀ·g` —
//! so a training graph carries one transposed copy of every weight matrix. The
//! backend has folded the *compute* away for a while; what it had not done is
//! stop *reserving* the buffer, because the memory plan is built before the fold
//! is decided. Measured on a 512x512 weight: a 2.36 MB plan for a graph whose
//! only real tensors come to 1.31 MB.
//!
//! The two halves have to agree about which nodes vanish. They share one
//! predicate (`rlx_compile::memory::is_elidable_matmul_transpose`) precisely so
//! they cannot drift: a planner that elides a buffer the compiler still writes
//! is silent corruption, not a saving.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const M: usize = 96;
const K: usize = 64;
const N: usize = 32;

fn ramp(n: usize, seed: usize) -> Vec<f32> {
    // Non-dyadic, so f32 rounding actually happens and a comparison means
    // something.
    (0..n)
        .map(|i| ((((i * 37 + seed * 13) % 91) as f32) - 45.0) / 91.0)
        .collect()
}

/// `out = aᵀ · g`, from the definition.
fn reference(a: &[f32], g: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; K * N];
    for k in 0..K {
        for n in 0..N {
            let mut acc = 0f64;
            for m in 0..M {
                acc += a[m * K + k] as f64 * g[m * N + n] as f64;
            }
            out[k * N + n] = acc as f32;
        }
    }
    out
}

/// `MatMul(Transpose(a), g)`, optionally with a second reader of the transpose
/// that forces it to be materialized.
fn build(extra_reader: bool) -> Graph {
    let mut g = Graph::new("mm_fold");
    let a = g.param("a", Shape::new(&[M, K], DType::F32));
    let grad = g.input("g", Shape::new(&[M, N], DType::F32));
    let at = g.add_node(
        Op::Transpose { perm: vec![1, 0] },
        vec![a],
        Shape::new(&[K, M], DType::F32),
    );
    let y = g.add_node(Op::MatMul, vec![at, grad], Shape::new(&[K, N], DType::F32));
    let mut outs = vec![y];
    if extra_reader {
        outs.push(g.add_node(
            Op::Binary(BinaryOp::Add),
            vec![at, at],
            Shape::new(&[K, M], DType::F32),
        ));
    }
    g.set_outputs(outs);
    g
}

fn run(extra_reader: bool, a: &[f32], grad: &[f32]) -> Vec<f32> {
    let mut c = Session::new(Device::Cpu).compile(build(extra_reader));
    c.set_param("a", a);
    c.run(&[("g", grad)]).into_iter().next().unwrap()
}

#[test]
fn a_folded_operand_transpose_gives_the_same_answer() {
    let a = ramp(M * K, 1);
    let grad = ramp(M * N, 2);
    let want = reference(&a, &grad);

    let folded = run(false, &a, &grad);
    let materialized = run(true, &a, &grad);

    let err = |v: &[f32]| -> f32 {
        v.iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs() / w.abs().max(1e-3))
            .fold(0.0f32, f32::max)
    };
    assert!(err(&folded) < 1e-4, "folded rel err {}", err(&folded));
    assert!(
        err(&materialized) < 1e-4,
        "materialized rel err {}",
        err(&materialized)
    );
    // Both are valid f32 orderings; they need not be bit-identical, only both
    // correct against the f64 reference above.
}

/// The folded transpose must not be planned a buffer.
///
/// Asserted against the graph's real tensors rather than against the
/// materialized plan: the extra reader in that plan has an output buffer the
/// same size as the transpose, so a comparison between the two cannot tell
/// "transpose elided" from "extra output allocated". Only a direct floor can.
#[test]
fn a_folded_operand_transpose_reserves_no_arena() {
    let plan = |extra| rlx_opt::memory::plan_memory_native_in_order(&build(extra), 64).arena_size;

    let a_bytes = M * K * 4;
    let g_bytes = M * N * 4;
    let y_bytes = K * N * 4;
    // Everything the folded graph genuinely needs, plus room for alignment.
    let needed = a_bytes + g_bytes + y_bytes;
    let folded = plan(false);
    assert!(
        folded < needed + a_bytes / 2,
        "the folded graph reserves {folded} bytes where its real tensors come to \
         {needed}; a transposed copy of the {a_bytes}-byte weight is still being \
         planned"
    );
    // And the materialized graph must reserve more, or the planner is dropping
    // a buffer that is still read.
    assert!(
        plan(true) > folded,
        "materializing the transpose reserved no more than folding it"
    );
}

/// A transpose with a second reader must survive. If the planner drops it, that
/// reader silently gets an unallocated buffer.
#[test]
fn a_second_reader_keeps_the_transpose_alive() {
    let a = ramp(M * K, 3);
    let grad = ramp(M * N, 4);

    let mut c = Session::new(Device::Cpu).compile(build(true));
    c.set_param("a", &a);
    let outs = c.run(&[("g", grad.as_slice())]);
    assert_eq!(outs.len(), 2);

    let mut want = vec![0f32; K * M];
    for m in 0..M {
        for k in 0..K {
            want[k * M + m] = a[m * K + k] * 2.0;
        }
    }
    assert_eq!(
        outs[1], want,
        "the second reader got the wrong tensor — the transpose was elided out \
         from under it"
    );
}
