// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Regression test: `Op::GroupedMatMul` must reject an expert bank that is not
//! in `[E, K, N]` order instead of quietly computing on it.
//!
//! Every backend takes `N` off `weight.dim(2)` while the arena sizes the output
//! slot from the node's declared shape. A bank left in a checkpoint's `[E, N, K]`
//! (`[out, in]`) order has the right rank *and* the right element count, so
//! nothing used to reject it — the kernel then wrote `M·K` floats into an `M·N`
//! slot. With `K < N` that leaves the tail of every output row holding whatever
//! the arena last put there (an MoE model that looks non-causal: logits for
//! early tokens move when the prompt grows); with `K > N` it runs past the slot
//! and corrupts the neighbouring tensor.
//!
//! `rlx_ir::shape::grouped_matmul_dims` now checks K against the input and the
//! derived `[M, N]` against the declared output, and every lowering path calls
//! it.

use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Graph, HirGraphExt, Op, Shape};
use rlx_runtime::{Device, Session};

const M: usize = 4;
const K: usize = 3;
const N: usize = 5;
const E: usize = 2;

fn param(g: &mut Graph, name: &str, shape: &[usize]) -> rlx_ir::NodeId {
    g.add_node(
        Op::Param {
            name: name.to_string(),
        },
        vec![],
        Shape::new(shape, DType::F32),
    )
}

/// `[M,K] · bank[idx[r]] → [M,N]` with the bank shaped `bank_dims` and the
/// output declared `out_dims`.
fn build(bank_dims: &[usize], out_dims: &[usize]) -> Graph {
    let mut g = Graph::new("gmm");
    let x = g.add_node(
        Op::Input {
            name: "x".to_string(),
        },
        vec![],
        Shape::new(&[M, K], DType::F32),
    );
    let w = param(&mut g, "bank", bank_dims);
    let idx = g.add_node(
        Op::Input {
            name: "idx".to_string(),
        },
        vec![],
        Shape::new(&[M], DType::F32),
    );
    let out = g.add_node(
        Op::GroupedMatMul,
        vec![x, w, idx],
        Shape::new(out_dims, DType::F32),
    );
    g.set_outputs(vec![out]);
    g
}

fn run(g: Graph) {
    let mut compiled = Session::new(Device::Cpu).compile(g);
    compiled.set_param("bank", &[0.5f32; E * K * N]);
    let _ = compiled.run(&[("x", &[1.0f32; M * K][..]), ("idx", &[0.0f32; M][..])]);
}

#[test]
fn correct_bank_layout_runs() {
    run(build(&[E, K, N], &[M, N]));
}

/// The bank the checkpoint ships, un-transposed. Same rank, same element count,
/// wrong axis order.
#[test]
#[should_panic(expected = "K mismatch")]
fn untransposed_bank_is_rejected() {
    run(build(&[E, N, K], &[M, N]));
}

/// When `K == N` the two layouts are indistinguishable from the operands alone,
/// so the declared output is the only witness left. (This is the real-world
/// shape of the trap: a MoE whose `2·intermediate` happens to equal `hidden`.)
/// In a debug build the IR verifier reports it first — `infer_output_shape` now
/// covers `GroupedMatMul` — and in release the lowering check does.
#[test]
#[should_panic(expected = "shape mismatch")]
fn square_bank_with_a_wrong_output_is_rejected() {
    run(build(&[E, K, K], &[M, N]));
}

/// The builder helper derives the output shape, so it cannot be got wrong —
/// but it still refuses a bank whose K axis disagrees with the input.
#[test]
fn builder_helper_derives_the_output_shape() {
    let mut hir = HirModule::new("gmm");
    let mut b = HirMut::new(&mut hir);
    let x = b.input("x", Shape::new(&[M, K], DType::F32));
    let w = b.param("bank", Shape::new(&[E, K, N], DType::F32));
    let idx = b.input("idx", Shape::new(&[M], DType::F32));
    let out = b.grouped_matmul(x, w, idx);
    assert_eq!(b.shape(out).dims(), Shape::new(&[M, N], DType::F32).dims());
}
