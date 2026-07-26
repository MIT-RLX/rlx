// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Decomposition oracle parity: `Op::AxialRope2d` / `Op::FakeQuantize` run
//! natively on CPU must equal the same graph after `LowerAxialRope2d` /
//! `LowerFakeQuantize` decomposes them to primitives — proving the arms that
//! close the two "hard-fail where not native" gaps are bit-exact.

use rlx_fusion::pass::Pass;
use rlx_fusion::{LowerAxialRope2d, LowerFakeQuantize};
use rlx_ir::op::{ScaleMode, SteKind};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn run(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    Session::new(Device::Cpu)
        .compile(g)
        .run(inputs)
        .pop()
        .unwrap()
}

// ---- AxialRope2d ----------------------------------------------------------

const END_X: usize = 4;
const END_Y: usize = 3;
const HEAD_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const THETA: f32 = 10_000.0;
const REPEAT: usize = 1;

fn axial_graph() -> (Graph, usize) {
    let n_tokens = END_X * END_Y * REPEAT;
    let hs = NUM_HEADS * HEAD_DIM;
    let mut g = Graph::new("axial");
    let x = g.input("x", Shape::new(&[1, n_tokens, hs], DType::F32));
    let o = g.axial_rope2d(x, END_X, END_Y, HEAD_DIM, NUM_HEADS, THETA, REPEAT);
    g.set_outputs(vec![o]);
    (g, n_tokens * hs)
}

#[test]
fn axial_rope2d_decompose_matches_native() {
    let (native, n) = axial_graph();
    let x: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.37 + 0.1).sin() * 1.3)
        .collect();

    let ref_out = run(native, &[("x", &x)]);

    let (g2, _) = axial_graph();
    let decomposed = LowerAxialRope2d.run(g2);
    // The op must actually be gone (proves the pass fired).
    assert!(
        !decomposed
            .nodes()
            .iter()
            .any(|nd| matches!(nd.op, Op::AxialRope2d { .. })),
        "AxialRope2d survived the lowering pass"
    );
    let dec_out = run(decomposed, &[("x", &x)]);

    let err = max_abs(&ref_out, &dec_out);
    eprintln!("axial_rope2d decompose max_abs={err:.3e}");
    assert!(err < 1e-5, "axial decompose parity failed: {err:.3e}");
}

#[test]
fn axial_rope2d_decompose_batched() {
    // batch = 2 exercises the per-batch table tiling + pair offset.
    let n_tokens = END_X * END_Y * REPEAT;
    let hs = NUM_HEADS * HEAD_DIM;
    let build = || {
        let mut g = Graph::new("axial_b");
        let x = g.input("x", Shape::new(&[2, n_tokens, hs], DType::F32));
        let o = g.axial_rope2d(x, END_X, END_Y, HEAD_DIM, NUM_HEADS, THETA, REPEAT);
        g.set_outputs(vec![o]);
        g
    };
    let n = 2 * n_tokens * hs;
    let x: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.21 + 0.7).cos() * 0.9)
        .collect();
    let ref_out = run(build(), &[("x", &x)]);
    let dec_out = run(LowerAxialRope2d.run(build()), &[("x", &x)]);
    let err = max_abs(&ref_out, &dec_out);
    eprintln!("axial_rope2d batched decompose max_abs={err:.3e}");
    assert!(
        err < 1e-5,
        "axial batched decompose parity failed: {err:.3e}"
    );
}

// ---- FakeQuantize ---------------------------------------------------------

fn fq_graph(shape: &[usize], bits: u8, axis: Option<usize>) -> Graph {
    let mut g = Graph::new("fq");
    let x = g.input("x", Shape::new(shape, DType::F32));
    let o = g.add_node(
        Op::FakeQuantize {
            bits,
            axis,
            ste: SteKind::Identity,
            scale_mode: ScaleMode::PerBatch,
        },
        vec![x],
        Shape::new(shape, DType::F32),
    );
    g.set_outputs(vec![o]);
    g
}

fn check_fq(shape: &[usize], bits: u8, axis: Option<usize>) {
    let n: usize = shape.iter().product();
    let x: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.53 + 0.2).sin() * 2.0 - 0.3)
        .collect();

    let ref_out = run(fq_graph(shape, bits, axis), &[("x", &x)]);
    let decomposed = LowerFakeQuantize.run(fq_graph(shape, bits, axis));
    assert!(
        !decomposed
            .nodes()
            .iter()
            .any(|nd| matches!(nd.op, Op::FakeQuantize { .. })),
        "FakeQuantize survived the lowering pass"
    );
    let dec_out = run(decomposed, &[("x", &x)]);

    let err = max_abs(&ref_out, &dec_out);
    eprintln!("fake_quantize bits={bits} axis={axis:?} max_abs={err:.3e}");
    assert!(
        err < 1e-5,
        "fake_quant decompose parity failed (bits={bits}, axis={axis:?}): {err:.3e}"
    );
}

#[test]
fn fake_quantize_per_tensor_decompose() {
    check_fq(&[3, 5], 8, None);
    check_fq(&[4, 6], 4, None);
    check_fq(&[8], 2, None);
}

#[test]
fn fake_quantize_per_channel_decompose() {
    // axis = 1: per-channel scale on a [N, C, L] tensor.
    check_fq(&[2, 4, 3], 8, Some(1));
    check_fq(&[3, 5], 8, Some(0));
}
