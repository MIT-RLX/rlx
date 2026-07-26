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
        Activation::Recip => {
            let y = y.unwrap_or_else(|| g.activation(Activation::Recip, x, shape.clone()));
            let y2 = g.binary(BinaryOp::Mul, y, y, shape.clone());
            g.activation(Activation::Neg, y2, shape.clone())
        }
        Activation::Abs => {
            let ax = g.activation(Activation::Abs, x, shape.clone());
            g.binary(BinaryOp::Div, x, ax, shape.clone())
        }
        // `GeluApprox` (tanh-approx GELU, the PyTorch/ViT default) shares this
        // derivative EXACTLY; `Gelu` (erf form) uses it as a ~1e-3 approximation.
        // NB `GeluApprox` was previously (wrongly) grouped with `Silu` below and
        // got SILU's derivative — same class of silent-wrong-gradient bug as the
        // old `Gelu`, caught by `tests/activation_backward_fd.rs`.
        Activation::Gelu | Activation::GeluApprox => {
            // Tanh-approximation of the GELU derivative (matches the exact erf
            // form to ~1e-3 — the same approximation `GeluApprox` uses):
            //   g(x)  = 0.5·x·(1 + tanh(u)),  u = c·(x + a·x³), c = √(2/π), a = 0.044715
            //   g'(x) = 0.5·(1 + tanh(u)) + 0.5·x·(1 − tanh²(u))·u',  u' = c·(1 + 3a·x²)
            // The previous formula was WRONG: it computed `(1−tanh²(u))·(c+1.5x²)`
            // ≈ d/dx tanh(u), dropping BOTH the `0.5·(1+tanh(u))` term and the
            // `0.5·x` factor — silently corrupting gelu gradients on every backend
            // that DECOMPOSES `ActivationBackward` (e.g. CUDA, which lacks a native
            // kernel) while CPU/MLX (native kernel) were correct. That mismatch made
            // conv models diverge on CUDA at LRs where CPU/MLX trained fine.
            let c = scalar_const(0.7978845608, shape, g); // √(2/π)
            let a = scalar_const(0.044715, shape, g);
            let half = scalar_const(0.5, shape, g);
            let x2 = g.binary(BinaryOp::Mul, x, x, shape.clone());
            let x3 = g.binary(BinaryOp::Mul, x, x2, shape.clone());
            let a_x3 = g.binary(BinaryOp::Mul, a, x3, shape.clone());
            let inner_arg = g.binary(BinaryOp::Add, x, a_x3, shape.clone());
            let u = g.binary(BinaryOp::Mul, c, inner_arg, shape.clone());
            let t = g.activation(Activation::Tanh, u, shape.clone());
            // term1 = 0.5·(1 + tanh(u))
            let one_a = scalar_const(1.0, shape, g);
            let one_plus_t = g.binary(BinaryOp::Add, one_a, t, shape.clone());
            let term1 = g.binary(BinaryOp::Mul, half, one_plus_t, shape.clone());
            // term2 = 0.5·x·(1 − tanh²(u))·u',  u' = c·(1 + 3a·x²)
            let t2 = g.binary(BinaryOp::Mul, t, t, shape.clone());
            let one_b = scalar_const(1.0, shape, g);
            let sech2 = g.binary(BinaryOp::Sub, one_b, t2, shape.clone());
            let three_a = scalar_const(3.0 * 0.044715, shape, g); // 0.134145
            let three_a_x2 = g.binary(BinaryOp::Mul, three_a, x2, shape.clone());
            let one_c = scalar_const(1.0, shape, g);
            let u_arg = g.binary(BinaryOp::Add, one_c, three_a_x2, shape.clone());
            let u_prime = g.binary(BinaryOp::Mul, c, u_arg, shape.clone());
            let half_x = g.binary(BinaryOp::Mul, half, x, shape.clone());
            let hx_sech2 = g.binary(BinaryOp::Mul, half_x, sech2, shape.clone());
            let term2 = g.binary(BinaryOp::Mul, hx_sech2, u_prime, shape.clone());
            g.binary(BinaryOp::Add, term1, term2, shape.clone())
        }
        Activation::Silu => {
            let sig = g.activation(Activation::Sigmoid, x, shape.clone());
            let one = scalar_const(1.0, shape, g);
            let one_minus = g.binary(BinaryOp::Sub, one, sig, shape.clone());
            let sig_om = g.binary(BinaryOp::Mul, sig, one_minus, shape.clone());
            let x_sig_om = g.binary(BinaryOp::Mul, x, sig_om, shape.clone());
            g.binary(BinaryOp::Add, sig, x_sig_om, shape.clone())
        }
        Activation::Round => scalar_const(0.0, shape, g),
        // Piecewise-constant: zero (sub)gradient.
        Activation::Floor | Activation::Ceil | Activation::Sign => scalar_const(0.0, shape, g),
        // softplus'(x) = sigmoid(x).
        Activation::Softplus => g.activation(Activation::Sigmoid, x, shape.clone()),
        // ELU'(x) = 1 (x>0) else eˣ  ==  min(eˣ, 1).
        Activation::Elu => {
            let ex = g.activation(Activation::Exp, x, shape.clone());
            let one = scalar_const(1.0, shape, g);
            g.add_node(Op::Binary(BinaryOp::Min), vec![ex, one], shape.clone())
        }
        // erf'(x) = (2/√π)·e^(−x²).
        Activation::Erf => {
            let x2 = g.binary(BinaryOp::Mul, x, x, shape.clone());
            let neg = g.activation(Activation::Neg, x2, shape.clone());
            let e = g.activation(Activation::Exp, neg, shape.clone());
            let c = scalar_const(std::f64::consts::FRAC_2_SQRT_PI, shape, g); // 2/√π
            g.binary(BinaryOp::Mul, e, c, shape.clone())
        }
        // softsign'(x) = 1/(1+|x|)².
        Activation::Softsign => {
            let ax = g.activation(Activation::Abs, x, shape.clone());
            let one = scalar_const(1.0, shape, g);
            let denom = g.binary(BinaryOp::Add, one, ax, shape.clone());
            let denom2 = g.binary(BinaryOp::Mul, denom, denom, shape.clone());
            let num = scalar_const(1.0, shape, g);
            g.binary(BinaryOp::Div, num, denom2, shape.clone())
        }
        // logsigmoid'(x) = σ(−x).
        Activation::LogSigmoid => {
            let nx = g.activation(Activation::Neg, x, shape.clone());
            g.activation(Activation::Sigmoid, nx, shape.clone())
        }
        // hardsigmoid'(x) = 1/6 on |x|<3 else 0 == relu(sign(3−|x|))/6 (no Compare).
        Activation::HardSigmoid => hard_sigmoid_deriv(g, x, shape),
        // hardswish'(x) = hardsigmoid(x) + x·hardsigmoid'(x).
        Activation::HardSwish => {
            let hs = g.activation(Activation::HardSigmoid, x, shape.clone());
            let hsp = hard_sigmoid_deriv(g, x, shape);
            let x_hsp = g.binary(BinaryOp::Mul, x, hsp, shape.clone());
            g.binary(BinaryOp::Add, hs, x_hsp, shape.clone())
        }
        // mish'(x) = tanh(sp) + x·(1−tanh²(sp))·σ(x),  sp = softplus(x).
        Activation::Mish => {
            let sp = g.activation(Activation::Softplus, x, shape.clone());
            let t = g.activation(Activation::Tanh, sp, shape.clone());
            let t2 = g.binary(BinaryOp::Mul, t, t, shape.clone());
            let one = scalar_const(1.0, shape, g);
            let sech2 = g.binary(BinaryOp::Sub, one, t2, shape.clone());
            let sig = g.activation(Activation::Sigmoid, x, shape.clone());
            let a = g.binary(BinaryOp::Mul, x, sech2, shape.clone());
            let b = g.binary(BinaryOp::Mul, a, sig, shape.clone());
            g.binary(BinaryOp::Add, t, b, shape.clone())
        }
    }
}

/// `hardsigmoid'(x) = 1/6` on `|x|<3` else 0, built as `relu(sign(3−|x|))/6`
/// to avoid `Compare`/`Where` bool paths (per the module convention).
fn hard_sigmoid_deriv(g: &mut Graph, x: NodeId, shape: &Shape) -> NodeId {
    let ax = g.activation(Activation::Abs, x, shape.clone());
    let three = scalar_const(3.0, shape, g);
    let d = g.binary(BinaryOp::Sub, three, ax, shape.clone());
    let s = g.activation(Activation::Sign, d, shape.clone());
    let ind = g.activation(Activation::Relu, s, shape.clone());
    let sixth = scalar_const(1.0 / 6.0, shape, g);
    g.binary(BinaryOp::Mul, ind, sixth, shape.clone())
}
