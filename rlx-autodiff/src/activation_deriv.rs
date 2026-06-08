// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Closed-form activation derivatives as primitive MIR (`f'(x)`).

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

pub fn scalar_const(value: f64, shape: &Shape, g: &mut Graph) -> NodeId {
    let bytes = match shape.dtype() {
        DType::F32 => (value as f32).to_le_bytes().to_vec(),
        DType::F64 => value.to_le_bytes().to_vec(),
        DType::F16 => half::f16::from_f32(value as f32).to_le_bytes().to_vec(),
        DType::BF16 => half::bf16::from_f32(value as f32).to_le_bytes().to_vec(),
        other => panic!("activation_deriv: unsupported dtype {other:?}"),
    };
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::scalar(shape.dtype()),
    )
}

/// Emit `f'(x)` for a unary activation, composed from primitives.
pub fn activation_deriv_wrt_x(
    g: &mut Graph,
    kind: Activation,
    x: NodeId,
    y: Option<NodeId>,
    shape: &Shape,
) -> NodeId {
    match kind {
        Activation::Neg => scalar_const(-1.0, shape, g),
        Activation::Exp => y.unwrap_or_else(|| g.activation(Activation::Exp, x, shape.clone())),
        Activation::Log => {
            let one = scalar_const(1.0, shape, g);
            g.binary(BinaryOp::Div, one, x, shape.clone())
        }
        Activation::Sqrt => {
            let half = scalar_const(0.5, shape, g);
            let y = y.unwrap_or_else(|| g.activation(Activation::Sqrt, x, shape.clone()));
            g.binary(BinaryOp::Div, half, y, shape.clone())
        }
        Activation::Rsqrt => {
            let y = y.unwrap_or_else(|| g.activation(Activation::Rsqrt, x, shape.clone()));
            let y2 = g.binary(BinaryOp::Mul, y, y, shape.clone());
            let y3 = g.binary(BinaryOp::Mul, y2, y, shape.clone());
            let neg_half = scalar_const(-0.5, shape, g);
            g.binary(BinaryOp::Mul, neg_half, y3, shape.clone())
        }
        Activation::Tanh => {
            let y = y.unwrap_or_else(|| g.activation(Activation::Tanh, x, shape.clone()));
            let y2 = g.binary(BinaryOp::Mul, y, y, shape.clone());
            let one = scalar_const(1.0, shape, g);
            g.binary(BinaryOp::Sub, one, y2, shape.clone())
        }
        Activation::Sigmoid => {
            let y = y.unwrap_or_else(|| g.activation(Activation::Sigmoid, x, shape.clone()));
            let one = scalar_const(1.0, shape, g);
            let om = g.binary(BinaryOp::Sub, one, y, shape.clone());
            g.binary(BinaryOp::Mul, y, om, shape.clone())
        }
        Activation::Relu => {
            // H(x) = relu(x)/x for x≠0 (0 at x=0). Differentiable for stacking
            // without `Compare`/`Cast` bool paths that break CPU execution.
            let rx = g.activation(Activation::Relu, x, shape.clone());
            g.binary(BinaryOp::Div, rx, x, shape.clone())
        }
        Activation::Sin => g.activation(Activation::Cos, x, shape.clone()),
        Activation::Cos => {
            let sx = g.activation(Activation::Sin, x, shape.clone());
            g.activation(Activation::Neg, sx, shape.clone())
        }
        Activation::Tan => {
            let y = y.unwrap_or_else(|| g.activation(Activation::Tan, x, shape.clone()));
            let y2 = g.binary(BinaryOp::Mul, y, y, shape.clone());
            let one = scalar_const(1.0, shape, g);
            g.binary(BinaryOp::Add, one, y2, shape.clone())
        }
        Activation::Atan => {
            let x2 = g.binary(BinaryOp::Mul, x, x, shape.clone());
            let one = scalar_const(1.0, shape, g);
            let denom = g.binary(BinaryOp::Add, one, x2, shape.clone());
            let one2 = scalar_const(1.0, shape, g);
            g.binary(BinaryOp::Div, one2, denom, shape.clone())
        }
        Activation::Abs => {
            let ax = g.activation(Activation::Abs, x, shape.clone());
            g.binary(BinaryOp::Div, x, ax, shape.clone())
        }
        Activation::Gelu => {
            let c = scalar_const(0.7978845608 * 0.5, shape, g);
            let x2 = g.binary(BinaryOp::Mul, x, x, shape.clone());
            let x3 = g.binary(BinaryOp::Mul, x, x2, shape.clone());
            let c_x3 = g.binary(BinaryOp::Mul, c, x3, shape.clone());
            let inner = g.binary(BinaryOp::Add, x, c_x3, shape.clone());
            let t = g.activation(Activation::Tanh, inner, shape.clone());
            let one = scalar_const(1.0, shape, g);
            let t2 = g.binary(BinaryOp::Mul, t, t, shape.clone());
            let sech2 = g.binary(BinaryOp::Sub, one, t2, shape.clone());
            let one_half = scalar_const(1.5, shape, g);
            let one_half_x2 = g.binary(BinaryOp::Mul, one_half, x2, shape.clone());
            let inner_deriv = g.binary(BinaryOp::Add, c, one_half_x2, shape.clone());
            g.binary(BinaryOp::Mul, sech2, inner_deriv, shape.clone())
        }
        Activation::GeluApprox | Activation::Silu => {
            let sig = g.activation(Activation::Sigmoid, x, shape.clone());
            let one = scalar_const(1.0, shape, g);
            let one_minus = g.binary(BinaryOp::Sub, one, sig, shape.clone());
            let sig_om = g.binary(BinaryOp::Mul, sig, one_minus, shape.clone());
            let x_sig_om = g.binary(BinaryOp::Mul, x, sig_om, shape.clone());
            g.binary(BinaryOp::Add, sig, x_sig_om, shape.clone())
        }
        Activation::Round => scalar_const(0.0, shape, g),
    }
}
