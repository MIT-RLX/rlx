// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::ScatterAdd` along an arbitrary axis, with `f32` or `i64` indices.
//!
//! The oracle is an independent index computation written from the op's
//! definition (`out[.., idx[i], ..] += upd[.., i, ..]`), not the lowering — so
//! the transpose/scatter/transpose rewrite is being checked against the
//! semantics it is supposed to preserve rather than against itself.

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// Reference: accumulate `updates` into a zeroed `out_dims` along `axis`.
///
/// Indices are assumed in range — that is the op's contract, so the oracle does
/// not model an out-of-range policy the kernels do not implement either.
fn reference(
    upd_dims: &[usize],
    out_dims: &[usize],
    axis: usize,
    updates: &[f32],
    indices: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dims.iter().product()];
    let n_upd = upd_dims[axis];
    let out_len = out_dims[axis];
    let inner: usize = upd_dims[axis + 1..].iter().product();
    let outer: usize = upd_dims[..axis].iter().product();
    let out_inner: usize = out_dims[axis + 1..].iter().product();

    for o in 0..outer {
        for u in 0..n_upd {
            let dst = indices[u] as usize;
            assert!(dst < out_len, "test oracle: index {dst} out of range");
            for j in 0..inner {
                let src_i = (o * n_upd + u) * inner + j;
                let dst_i = (o * out_len + dst) * out_inner + j;
                out[dst_i] += updates[src_i];
            }
        }
    }
    out
}

fn run(
    upd_dims: &[usize],
    out_dims: &[usize],
    axis: usize,
    updates: &[f32],
    indices: &[f32],
    idx_dtype: DType,
) -> Vec<f32> {
    let mut g = Graph::new("scatter_axis");
    let u = g.input("u", Shape::new(upd_dims, DType::F32));
    let i = g.input("i", Shape::new(&[upd_dims[axis]], idx_dtype));
    let out = g.add_node(
        Op::ScatterAdd { axis },
        vec![u, i],
        Shape::new(out_dims, DType::F32),
    );
    g.set_outputs(vec![out]);
    Session::new(Device::Cpu)
        .compile(g)
        .run(&[("u", updates), ("i", indices)])
        .pop()
        .unwrap()
}

fn iota(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 + 1.0).collect()
}

/// Axis 0 must be unchanged by the generalization — this is the path every
/// existing MoE unpermute and embedding gradient already takes.
#[test]
fn axis_zero_is_unchanged() {
    let upd = [4usize, 3];
    let out = [6usize, 3];
    let u = iota(12);
    let idx = vec![0.0f32, 2.0, 2.0, 5.0];
    let got = run(&upd, &out, 0, &u, &idx, DType::F32);
    assert_eq!(got, reference(&upd, &out, 0, &u, &idx));
}

/// Scatter along the trailing axis of a 2-D tensor.
#[test]
fn axis_one_matches_reference() {
    let upd = [3usize, 4];
    let out = [3usize, 6];
    let u = iota(12);
    let idx = vec![1.0f32, 1.0, 4.0, 0.0];
    let got = run(&upd, &out, 1, &u, &idx, DType::F32);
    assert_eq!(got, reference(&upd, &out, 1, &u, &idx));
}

/// Every axis of a rank-3 tensor.
#[test]
fn every_axis_of_rank_three() {
    for axis in 0..3 {
        let mut upd = [2usize, 3, 4];
        let mut out = [2usize, 3, 4];
        upd[axis] = 3;
        out[axis] = 5;
        let n_upd: usize = upd.iter().product();
        let u = iota(n_upd);
        let idx = vec![4.0f32, 0.0, 4.0]; // duplicate on purpose
        let got = run(&upd, &out, axis, &u, &idx, DType::F32);
        let want = reference(&upd, &out, axis, &u, &idx);
        assert_eq!(got, want, "axis {axis}");
    }
}

/// Repeated indices must accumulate, not overwrite — the property that makes
/// this a *scatter-add*.
#[test]
fn duplicate_indices_accumulate() {
    let upd = [4usize, 2];
    let out = [3usize, 2];
    let u = vec![1.0f32; 8];
    let idx = vec![1.0f32, 1.0, 1.0, 1.0];
    let got = run(&upd, &out, 0, &u, &idx, DType::F32);
    assert_eq!(got, vec![0.0, 0.0, 4.0, 4.0, 0.0, 0.0]);
}

/// Run with the index tensor embedded as an `i64` constant.
///
/// A constant rather than an input because the `run` feed path is `&[f32]`;
/// handing it f32 bytes for an `I64`-typed input would reinterpret them as
/// garbage indices, which is a property of the test harness, not of the op.
fn run_i64_indices(
    upd_dims: &[usize],
    out_dims: &[usize],
    axis: usize,
    updates: &[f32],
    indices: &[i64],
) -> Vec<f32> {
    let mut g = Graph::new("scatter_axis_i64");
    let u = g.input("u", Shape::new(upd_dims, DType::F32));
    let data: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
    let i = g.add_node(
        Op::Constant { data },
        vec![],
        Shape::new(&[indices.len()], DType::I64),
    );
    let out = g.add_node(
        Op::ScatterAdd { axis },
        vec![u, i],
        Shape::new(out_dims, DType::F32),
    );
    g.set_outputs(vec![out]);
    Session::new(Device::Cpu)
        .compile(g)
        .run(&[("u", updates)])
        .pop()
        .unwrap()
}

/// `i64` indices must give the same answer as the `f32` encoding, on every axis.
#[test]
fn i64_indices_match_f32_indices() {
    for axis in 0..2 {
        let mut upd = [3usize, 4];
        let mut out = [3usize, 4];
        upd[axis] = 4;
        out[axis] = 7;
        let u = iota(upd.iter().product());
        let idx_f32 = vec![6.0f32, 0.0, 3.0, 3.0];
        let idx_i64 = vec![6i64, 0, 3, 3];

        let a = run(&upd, &out, axis, &u, &idx_f32, DType::F32);
        let b = run_i64_indices(&upd, &out, axis, &u, &idx_i64);
        assert_eq!(a, b, "axis {axis}: i64 and f32 index paths disagree");
        assert_eq!(a, reference(&upd, &out, axis, &u, &idx_f32));
    }
}

/// Scatter-add is the transpose of gather, so `gather(scatter(u, i), i)` must
/// return `u` whenever the indices are distinct and in range.
///
/// A structural identity: it holds for the right permutation and fails for any
/// axis confusion, without needing a reference implementation at all.
#[test]
fn scatter_then_gather_is_identity_for_distinct_indices() {
    let axis = 1usize;
    let upd = [2usize, 3];
    let out = [2usize, 5];
    let u = iota(6);
    let idx = vec![4.0f32, 1.0, 2.0]; // distinct, in range

    let scattered = run(&upd, &out, axis, &u, &idx, DType::F32);

    let mut g = Graph::new("gather_back");
    let s = g.input("s", Shape::new(&out, DType::F32));
    let i = g.input("i", Shape::new(&[3], DType::F32));
    let y = g.add_node(
        Op::Gather { axis },
        vec![s, i],
        Shape::new(&upd, DType::F32),
    );
    g.set_outputs(vec![y]);
    let back = Session::new(Device::Cpu)
        .compile(g)
        .run(&[("s", &scattered), ("i", &idx)])
        .pop()
        .unwrap();

    assert_eq!(back, u, "gather did not invert scatter along axis {axis}");
}

/// The immersed-boundary shape that motivated this: many grid cells
/// accumulating into a handful of per-object slots.
#[test]
fn many_to_few_reduction() {
    let cells = 64usize;
    let objects = 3usize;
    let u: Vec<f32> = (0..cells).map(|_| 1.0).collect();
    // Round-robin the cells across objects.
    let idx: Vec<f32> = (0..cells).map(|i| (i % objects) as f32).collect();
    let got = run(&[cells], &[objects], 0, &u, &idx, DType::F32);
    let want = reference(&[cells], &[objects], 0, &u, &idx);
    assert_eq!(got, want);
    assert_eq!(
        got.iter().sum::<f32>(),
        cells as f32,
        "mass must be conserved"
    );
}
