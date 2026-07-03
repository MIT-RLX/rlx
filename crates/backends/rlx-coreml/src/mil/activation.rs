// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// IR → CoreML ML Program (MIL) lowering. Pure data transformation: takes
// an RLX `Graph` plus baked parameter/constant data and produces a
// `proto::Model` ready to serialise into a `.mlpackage`. No FFI, so this
// builds and unit-tests on any host.

//! `activation` — extracted from the `mil` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use std::collections::HashMap;
use rlx_ir::op::{Activation, CmpOp, MaskKind, ReduceOp};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Dim, Graph, NodeId, Op, Shape};
use crate::proto;
use crate::{CoremlError, Result};
use super::helpers::simple_op_flex;
use super::helpers::*;

use super::*;

impl<'a> LowerCtx<'a> {
    /// Lower an activation, composing the ones MIL has no direct op for.
    pub(crate) fn lower_activation(&mut self, id: NodeId, act: Activation, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        // (mil_type, optional ("param", value)) for the simple unary cases.
        let direct: Option<(&str, Vec<(&str, proto::Argument)>)> = match act {
            Activation::Relu => Some(("relu", vec![])),
            Activation::Sigmoid => Some(("sigmoid", vec![])),
            Activation::Tanh => Some(("tanh", vec![])),
            Activation::Exp => Some(("exp", vec![])),
            // MIL `log` is `log(x + epsilon)` and — like `rsqrt` — requires the
            // param explicitly (CoreML rejects the model otherwise). Use CoreML's
            // own default 1e-45 so the result is unperturbed (negligible for x≥1,
            // e.g. the log-sum-exp in softmax-cross-entropy that surfaced this).
            Activation::Log => Some(("log", vec![("epsilon", bind_value(scalar_f32(1e-45)))])),
            Activation::Sqrt => Some(("sqrt", vec![])),
            Activation::Rsqrt => Some(("rsqrt", vec![("epsilon", bind_value(scalar_f32(1e-12)))])),
            Activation::Abs => Some(("abs", vec![])),
            Activation::Sin => Some(("sin", vec![])),
            Activation::Cos => Some(("cos", vec![])),
            Activation::Tan => Some(("tan", vec![])),
            Activation::Atan => Some(("atan", vec![])),
            Activation::Round => Some(("round", vec![])),
            Activation::Gelu => Some(("gelu", vec![("mode", bind_value(scalar_str("EXACT")))])),
            Activation::GeluApprox => Some((
                "gelu",
                vec![("mode", bind_value(scalar_str("TANH_APPROXIMATION")))],
            )),
            // Composed below.
            Activation::Silu | Activation::Neg => None,
        };

        if let Some((ty, mut params)) = direct {
            let mut binds = vec![("x", bind_name(&x))];
            binds.append(&mut params);
            let op = self.simple_op(ty, out_name, &node.shape, binds)?;
            self.push_named(id, out_name.to_string(), op);
            return Ok(());
        }

        match act {
            // silu(x) = x * sigmoid(x)
            Activation::Silu => {
                let sig = format!("{out_name}_sig");
                let sig_op =
                    self.simple_op("sigmoid", &sig, &node.shape, vec![("x", bind_name(&x))])?;
                self.operations.push(sig_op);
                let op = self.simple_op(
                    "mul",
                    out_name,
                    &node.shape,
                    vec![("x", bind_name(&x)), ("y", bind_name(&sig))],
                )?;
                self.push_named(id, out_name.to_string(), op);
            }
            // neg(x) = mul(x, -1)
            Activation::Neg => {
                let op = self.simple_op(
                    "mul",
                    out_name,
                    &node.shape,
                    vec![("x", bind_name(&x)), ("y", bind_value(scalar_f32(-1.0)))],
                )?;
                self.push_named(id, out_name.to_string(), op);
            }
            _ => unreachable!("handled above"),
        }
        Ok(())
    }

}
