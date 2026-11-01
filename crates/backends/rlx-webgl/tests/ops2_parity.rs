// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Parity for Im2Col, ArgMax/ArgMin, runtime Gather, and RoPE vs RLX's CPU
// backend.

use rlx_ir::{DType, Graph, GraphExt, Shape};

fn f32s(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..n).map(f).collect()
}
fn sh(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

fn parity(g: Graph, inputs: &[(&str, &[f32])]) {
    let mut sess = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g.clone());
    let reference = sess.run(inputs);
    let plan = rlx_webgl::build_plan(&g).expect("plan");
    let got = rlx_webgl::run_cpu(&plan, inputs).expect("run_cpu");
    assert_eq!(reference.len(), got.len(), "output count");
    for (oi, (r, gv)) in reference.iter().zip(&got).enumerate() {
        assert_eq!(
            r.len(),
            gv.len(),
            "output {oi} length: {} vs {}",
            r.len(),
            gv.len()
        );
        for (i, (a, b)) in r.iter().zip(gv).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * (1.0 + a.abs()),
                "output {oi}[{i}]: reference={a} webgl={b}"
            );
        }
    }
}

#[test]
fn im2col() {
    let xs = f32s(2 * 4 * 4, |i| (i as f32 * 0.3).sin());
    let mut g = Graph::new("im2col");
    let x = g.input("x", sh(&[1, 2, 4, 4]));
    let y = g.im2col(x, [2, 2], [1, 1], [0, 0], [1, 1]);
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs)]);
}

#[test]
fn im2col_padded() {
    let xs = f32s(9, |i| i as f32 - 4.0);
    let mut g = Graph::new("im2col_p");
    let x = g.input("x", sh(&[1, 1, 3, 3]));
    let y = g.im2col(x, [3, 3], [1, 1], [1, 1], [1, 1]);
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs)]);
}

#[test]
fn argmax_argmin() {
    // distinct per-row extremes
    let xs = vec![1.0f32, 5.0, 2.0, 3.0, 9.0, 1.0, 1.0, 1.0];
    let mut g = Graph::new("argmax");
    let x = g.input("x", sh(&[2, 4]));
    let mx = g.argmax(x, 1, false, sh(&[2]));
    let mn = g.argmin(x, 1, false, sh(&[2]));
    g.set_outputs(vec![mx, mn]);
    parity(g, &[("x", &xs)]);
}

#[test]
fn gather_embedding() {
    // table [4,3], indices [2] along axis 0 → [2,3].
    let table = f32s(4 * 3, |i| i as f32);
    let idx = vec![2.0f32, 0.0];
    let mut g = Graph::new("gather");
    let t = g.input("t", sh(&[4, 3]));
    let i = g.input("i", sh(&[2]));
    let y = g.gather(t, i, 0, sh(&[2, 3]));
    g.set_outputs(vec![y]);
    parity(g, &[("t", &table), ("i", &idx)]);
}

#[test]
fn rope_full_and_partial() {
    // x [seq=2, head_dim=4], cos/sin [seq=2, head_dim/2=2].
    let xs = f32s(8, |i| i as f32 * 0.3 - 1.0);
    let cs = f32s(4, |i| (0.2 * i as f32).cos());
    let ss = f32s(4, |i| (0.2 * i as f32).sin());

    // Full rotation.
    let mut g = Graph::new("rope");
    let x = g.input("x", sh(&[2, 4]));
    let cos = g.input("c", sh(&[2, 2]));
    let sin = g.input("s", sh(&[2, 2]));
    let y = g.rope(x, cos, sin, 4);
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs), ("c", &cs), ("s", &ss)]);

    // Partial rotation (n_rot = 2).
    let mut g = Graph::new("rope_n");
    let x = g.input("x", sh(&[2, 4]));
    let cos = g.input("c", sh(&[2, 2]));
    let sin = g.input("s", sh(&[2, 2]));
    let y = g.rope_n(x, cos, sin, 4, 2);
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs), ("c", &cs), ("s", &ss)]);
}

use rlx_ir::op::ScatterNdReduction;

/// Add a node whose output shape the IR infers.
fn node(g: &mut Graph, op: rlx_ir::Op, ins: Vec<rlx_ir::NodeId>) -> rlx_ir::NodeId {
    let id = g.add_node(op, ins, sh(&[1]));
    let s = rlx_ir::infer_shape::infer_output_shape(g, g.node(id)).expect("infer");
    g.node_mut(id).shape = s;
    id
}

/// Index values `i ↦ ((i·7 + 3) mod m) − neg` (f32-encoded), so some are negative.
fn idx(n: usize, m: usize, neg: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + 3) % m) as f32 - neg as f32)
        .collect()
}

#[test]
fn gather_elements_with_negative_indices() {
    let xs = f32s(5 * 4, |i| (i as f32 * 0.31).sin());
    for axis in [0i32, 1, -1] {
        let mut g = Graph::new("gather_elements");
        let d = g.input("x", sh(&[5, 4]));
        let i = g.input("i", sh(&[3, 4]));
        let y = g.gather_elements(d, i, axis);
        g.set_outputs(vec![y]);
        let m = if axis == 0 { 5 } else { 4 };
        parity(g, &[("x", &xs), ("i", &idx(12, 2 * m - 1, m - 1))]);
    }
}

#[test]
fn gather_nd_tuples_and_batch_dims() {
    // k = 2 of 3 data axes, negative components included.
    let xs = f32s(4 * 3 * 5, |i| i as f32 * 0.5 - 7.0);
    let mut g = Graph::new("gather_nd");
    let d = g.input("x", sh(&[4, 3, 5]));
    let i = g.input("i", sh(&[2, 3, 2]));
    let y = g.gather_nd(d, i, 0, sh(&[2, 3, 5]));
    g.set_outputs(vec![y]);
    let iv = vec![
        0.0f32, 1.0, -1.0, 2.0, 2.0, -3.0, 3.0, 0.0, -4.0, 1.0, 1.0, 1.0,
    ];
    parity(g, &[("x", &xs), ("i", &iv)]);

    // batch_dims = 1: each batch row indexes its own [3, 5] slab.
    let mut g = Graph::new("gather_nd_b");
    let d = g.input("x", sh(&[4, 3, 5]));
    let i = g.input("i", sh(&[4, 2, 1]));
    let y = g.gather_nd(d, i, 1, sh(&[4, 2, 5]));
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs), ("i", &idx(8, 3, 0))]);
}

#[test]
fn scatter_nd_every_reduction() {
    let data = f32s(5 * 3, |i| 1.0 + i as f32 * 0.25);
    let upd = f32s(4 * 3, |i| 0.5 + (i as f32 * 0.9).cos());
    // Repeated rows (2 twice) so the reductions differ; one negative index.
    let iv = vec![2.0f32, 0.0, -3.0, 4.0];
    for red in [
        ScatterNdReduction::None,
        ScatterNdReduction::Add,
        ScatterNdReduction::Mul,
        ScatterNdReduction::Max,
        ScatterNdReduction::Min,
    ] {
        let mut g = Graph::new("scatter_nd");
        let d = g.input("x", sh(&[5, 3]));
        let i = g.input("i", sh(&[4, 1]));
        let u = g.input("u", sh(&[4, 3]));
        let y = g.scatter_nd(d, i, u, red);
        g.set_outputs(vec![y]);
        parity(g, &[("x", &data), ("i", &iv), ("u", &upd)]);
    }
}

#[test]
fn scatter_elements_axes_and_reductions() {
    let data = f32s(4 * 5, |i| (i as f32 * 0.4).sin());
    let upd = f32s(3 * 5, |i| 1.0 + i as f32 * 0.1);
    for (axis, red) in [
        (0i32, ScatterNdReduction::None),
        (0, ScatterNdReduction::Add),
        (-1, ScatterNdReduction::Max),
    ] {
        // Updates [3, 5] (axis 0) or [4, 3] (axis 1) into data [4, 5].
        let (ud, m) = if axis == 0 {
            ([3usize, 5usize], 4)
        } else {
            ([4, 3], 5)
        };
        let mut g = Graph::new("scatter_elements");
        let d = g.input("x", sh(&[4, 5]));
        let i = g.input("i", sh(&ud));
        let u = g.input("u", sh(&ud));
        let y = g.scatter_elements(d, i, u, axis, red);
        g.set_outputs(vec![y]);
        let n = ud[0] * ud[1];
        parity(g, &[("x", &data), ("i", &idx(n, m, 0)), ("u", &upd[..n])]);
    }
}

#[test]
fn scatter_add_axes() {
    // ScatterAdd inputs are [updates, indices]; repeated targets accumulate.
    let upd = f32s(2 * 4 * 3, |i| 0.2 * i as f32 - 1.0);
    for (axis, out) in [(0usize, [6usize, 4, 3]), (1, [2, 5, 3])] {
        let ud = match axis {
            0 => [4usize, 4, 3],
            _ => [2, 4, 3],
        };
        let n_upd = ud.iter().product::<usize>();
        let mut g = Graph::new("scatter_add");
        let u = g.input("u", sh(&ud));
        let i = g.input("i", sh(&[ud[axis]]));
        let y = g.add_node(rlx_ir::Op::ScatterAdd { axis }, vec![u, i], sh(&out));
        g.set_outputs(vec![y]);
        let u_vals = f32s(n_upd, |i| upd[i % upd.len()]);
        parity(g, &[("u", &u_vals), ("i", &idx(ud[axis], out[axis], 0))]);
    }
}

#[test]
fn sort_argsort_topk() {
    // Ties (repeated values) check the stable / smaller-index-first order.
    let xs = vec![
        3.0f32, -1.0, 2.0, 3.0, 0.5, -1.0, //
        0.0, 4.0, 4.0, -2.0, 1.0, 1.0, //
    ];
    for (axis, desc) in [(1usize, false), (1, true), (0, true)] {
        let mut g = Graph::new("sort");
        let x = g.input("x", sh(&[2, 6]));
        let s = g.add_node(
            rlx_ir::Op::Sort {
                axis,
                descending: desc,
            },
            vec![x],
            sh(&[2, 6]),
        );
        let a = g.add_node(
            rlx_ir::Op::ArgSort {
                axis,
                descending: desc,
            },
            vec![x],
            sh(&[2, 6]),
        );
        g.set_outputs(vec![s, a]);
        parity(g, &[("x", &xs)]);
    }
    let mut g = Graph::new("topk");
    let x = g.input("x", sh(&[2, 6]));
    let t = g.add_node(rlx_ir::Op::TopK { k: 3 }, vec![x], sh(&[2, 3]));
    g.set_outputs(vec![t]);
    parity(g, &[("x", &xs)]);
}

/// Strided / reversed `Slice` legalizes through an I64 index constant — the
/// constant must be decoded by its dtype, not as raw f32 bytes.
#[test]
fn slice_step_and_negative_step() {
    let xs = f32s(4 * 9, |i| i as f32);
    for (start, len, step) in [(1usize, 4usize, 2i64), (7, 3, -2), (8, 9, -1)] {
        let mut g = Graph::new("slice");
        let x = g.input("x", sh(&[4, 9]));
        let y = node(
            &mut g,
            rlx_ir::Op::Slice {
                axis: 1,
                start,
                len,
                step,
            },
            vec![x],
        );
        g.set_outputs(vec![y]);
        parity(g, &[("x", &xs)]);
    }
}

/// `ResizeNearest2x` legalizes through `LowerResizeNearest2x`; every pixel is
/// repeated 2×2 (it used to tile the whole image instead).
#[test]
fn resize_nearest_2x() {
    let xs = f32s(2 * 2 * 3 * 4, |i| i as f32);
    let mut g = Graph::new("resize");
    let x = g.input("x", sh(&[2, 2, 3, 4]));
    let y = g.add_node(rlx_ir::Op::ResizeNearest2x, vec![x], sh(&[2, 2, 6, 8]));
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs)]);
}
