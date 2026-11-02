// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx_wgpu.prelu` (`unfuse::PRELU`): one pass of the unary kernel, `x` where
//! `x ≥ 0`, else `α·x` — PyTorch's `prelu` with one slope, bit for bit, for a
//! slope below zero, between, and above one; and the element-wise PReLU
//! forms (`Graph::prelu`, ONNX's) folded into it.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

#[test]
fn prelu_is_x_or_alpha_x() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    let n = 70_001;
    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 1001) as f32 - 500.0) / 77.0)
        .chain([0.0, -0.0])
        .collect();
    for alpha in [-0.48f32, 0.2, 3.468] {
        let mut g = Graph::new("prelu");
        let s = Shape::new(&[x.len()], DType::F32);
        let xi = g.input("x", s.clone());
        let y = g.add_node(
            Op::Custom {
                name: rlx_wgpu::unfuse::PRELU.to_string(),
                num_inputs: 1,
                attrs: alpha.to_bits().to_le_bytes().to_vec(),
            },
            vec![xi],
            s,
        );
        g.set_outputs(vec![y]);
        let out = WgpuExecutable::compile(g).run(&[("x", &x)]).remove(0);
        for (i, (&o, &v)) in out.iter().zip(&x).enumerate() {
            let want = if v >= 0.0 { v } else { alpha * v };
            assert_eq!(o.to_bits(), want.to_bits(), "α {alpha} at {i}: {v} → {o}");
        }
    }
}

/// A PReLU built from element-wise ops — `Graph::prelu`'s
/// `max(x, 0) + a·min(x, 0)` or the ONNX importer's `relu(x) − a·relu(−x)` —
/// runs as one `PRELU` (`unfuse::fold_prelu`): a constant slope as the op's
/// attribute, a scalar or per-channel slope tensor as its second input. The
/// same values as the element-wise ops (`RLX_WGPU_NO_FOLD_PRELU=1`) and as
/// `x` or `a·x` directly; one whose interior has another reader stays.
#[test]
fn graph_prelu_runs_as_one_pass() {
    use rlx_ir::op::{Activation, BinaryOp};
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    let (n, c, l) = (2, 5, 301);
    let sh = |d: &[usize]| Shape::new(d, DType::F32);
    let xs = sh(&[n, c, l]);
    let x: Vec<f32> = (0..n * c * l)
        .map(|i| ((i * 37 % 1001) as f32 - 500.0) / 77.0)
        .collect();
    let slopes: Vec<f32> = vec![-0.48, 0.2, 3.468, 1.0, 0.0];
    let prelus = |g: &rlx_ir::Graph| {
        rlx_wgpu::unfuse::unfuse(g.clone())
            .nodes()
            .iter()
            .filter(|n| matches!(&n.op, Op::Custom { name, .. } if name == rlx_wgpu::unfuse::PRELU))
            .count()
    };
    // (form, the slope as given, a reader of the interior too)
    #[derive(Clone, Copy, Debug)]
    enum Form {
        Const,
        Scalar,
        Channel,
        Onnx,
        Shared,
    }
    for form in [
        Form::Const,
        Form::Scalar,
        Form::Channel,
        Form::Onnx,
        Form::Shared,
    ] {
        let build = || {
            let mut g = Graph::new("prelu_forms");
            let xi = g.input("x", xs.clone());
            let a = match form {
                Form::Channel => g.input("a", sh(&[1, c, 1])),
                _ => g.input("a", sh(&[1])),
            };
            let y = match form {
                Form::Const => g.prelu_scalar(xi, -0.48),
                Form::Scalar | Form::Channel => g.prelu(xi, a),
                Form::Onnx => {
                    let pos = g.activation(Activation::Relu, xi, xs.clone());
                    let ng = g.activation(Activation::Neg, xi, xs.clone());
                    let rn = g.activation(Activation::Relu, ng, xs.clone());
                    let p = g.binary(BinaryOp::Mul, rn, a, xs.clone());
                    g.binary(BinaryOp::Sub, pos, p, xs.clone())
                }
                Form::Shared => {
                    let y = g.prelu(xi, a);
                    // the canonical form's `max(x, 0)` read again
                    let pos = g.node(y).inputs[0];
                    g.binary(BinaryOp::Add, y, pos, xs.clone())
                }
            };
            g.set_outputs(vec![y]);
            g
        };
        let a: Vec<f32> = match form {
            Form::Channel => slopes.clone(),
            _ => vec![-0.48],
        };
        let ins: [(&str, &[f32]); 2] = [("x", &x), ("a", &a)];
        let want_prelus = usize::from(!matches!(form, Form::Shared));
        assert_eq!(prelus(&build()), want_prelus, "{form:?}: PRELU nodes");
        let folded = WgpuExecutable::compile(build()).run(&ins).remove(0);
        rlx_ir::env::set("RLX_WGPU_NO_FOLD_PRELU", "1");
        assert_eq!(prelus(&build()), 0, "{form:?}: no fold");
        let plain = WgpuExecutable::compile(build()).run(&ins).remove(0);
        rlx_ir::env::clear_overrides();
        assert_eq!(folded, plain, "{form:?}: folded = element-wise");
        for (i, (&o, &v)) in folded.iter().zip(&x).enumerate() {
            let s = a[if a.len() == 1 { 0 } else { (i / l) % c }];
            let p = if v >= 0.0 { v } else { s * v };
            let want = if matches!(form, Form::Shared) {
                p + v.max(0.0)
            } else {
                p
            };
            assert_eq!(o, want, "{form:?} at {i}: {v} → {o}");
        }
    }
}
