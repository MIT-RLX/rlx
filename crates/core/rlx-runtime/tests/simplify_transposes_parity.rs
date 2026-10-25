// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `SimplifyTransposes` only removes data movement, so a graph computes the
//! same values — bit for bit — before and after it, on CPU.

mod common;

use rlx_compile::SimplifyTransposes;
use rlx_fusion::pass::Pass;
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Op, OpKind, Shape};
use rlx_runtime::{Device, Session};

const F: DType = DType::F32;

fn t(g: &mut Graph, x: NodeId, perm: &[usize]) -> NodeId {
    let dims = g.node(x).shape.dims().to_vec();
    let nd: Vec<_> = perm.iter().map(|&p| dims[p]).collect();
    g.add_node(
        Op::Transpose {
            perm: perm.to_vec(),
        },
        vec![x],
        Shape::from_dims(&nd, F),
    )
}

/// A channels-last island inside an NCHW graph, as an importer or an
/// NHWC-preferring lowering produces: x → NHWC → relu·2 + y → NCHW, twice,
/// with a constant operand transposed too.
fn island() -> Graph {
    let mut g = Graph::new("island");
    let x = g.input("x", Shape::new(&[1, 3, 4, 5], F));
    let y = g.input("y", Shape::new(&[1, 3, 4, 5], F));
    let mut h = x;
    for _ in 0..2 {
        let a = t(&mut g, h, &[0, 2, 3, 1]);
        let b = t(&mut g, y, &[0, 2, 3, 1]);
        let s = g.node(a).shape.clone();
        let r = g.add_node(Op::Activation(Activation::Relu), vec![a], s.clone());
        let two = g.add_node(
            Op::Constant {
                data: 2f32.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[], F),
        );
        let m = g.binary(BinaryOp::Mul, r, two, s.clone());
        let p = g.binary(BinaryOp::Add, m, b, s);
        h = t(&mut g, p, &[0, 3, 1, 2]);
    }
    let c_data: Vec<u8> = (0..60)
        .flat_map(|i| (i as f32 * 0.1).to_le_bytes())
        .collect();
    let c = g.add_node(
        Op::Constant { data: c_data },
        vec![],
        Shape::new(&[1, 4, 5, 3], F),
    );
    let ct = t(&mut g, c, &[0, 3, 1, 2]);
    let out = g.binary(BinaryOp::Sub, h, ct, Shape::new(&[1, 3, 4, 5], F));
    // A transpose chain that composes rather than cancels.
    let c1 = t(&mut g, out, &[0, 2, 1, 3]);
    let c2 = t(&mut g, c1, &[0, 1, 3, 2]);
    g.set_outputs(vec![out, c2]);
    g
}

fn run(g: Graph) -> Vec<Vec<f32>> {
    let x: Vec<f32> = (0..60).map(|i| (i as f32 * 0.37).sin()).collect();
    let y: Vec<f32> = (0..60).map(|i| (i as f32 * 0.11).cos()).collect();
    Session::new(Device::Cpu)
        .compile(g)
        .run(&[("x", &x), ("y", &y)])
}

#[test]
fn removes_the_island_transposes_and_keeps_the_values() {
    let g = island();
    let before = g
        .nodes()
        .iter()
        .filter(|n| n.op.kind() == OpKind::Transpose)
        .count();
    let simplified = SimplifyTransposes.run(g.clone());
    let after = simplified
        .nodes()
        .iter()
        .filter(|n| n.op.kind() == OpKind::Transpose)
        .count();
    assert_eq!(before, 9);
    assert_eq!(after, 1, "only the composed chain should remain");
    assert_eq!(run(g), run(simplified));
}

/// The compile option routes the pass through every backend's precompile
/// cleanup; the result matches the default compile on each registered device.
#[test]
fn compile_option_matches_default_on_every_device() {
    let _gpu = common::serialize_gpu();
    let x: Vec<f32> = (0..60).map(|i| (i as f32 * 0.37).sin()).collect();
    let y: Vec<f32> = (0..60).map(|i| (i as f32 * 0.11).cos()).collect();
    for dev in rlx_runtime::device_ext::available_devices() {
        let on = rlx_runtime::CompileOptions {
            simplify_transposes: true,
            ..Default::default()
        };
        let run = |opts: &rlx_runtime::CompileOptions| {
            Session::new(dev)
                .compile_with(island(), opts)
                .run(&[("x", &x), ("y", &y)])
        };
        let a = run(&rlx_runtime::CompileOptions::default());
        let b = run(&on);
        for (p, q) in a.iter().zip(&b) {
            let d = p
                .iter()
                .zip(q)
                .map(|(u, v)| (u - v).abs())
                .fold(0f32, f32::max);
            assert!(d <= 1e-6, "{dev:?}: option changed the result by {d}");
        }
    }
}
