// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// IR → CoreML ML Program (MIL) lowering. Pure data transformation: takes
// an RLX `Graph` plus baked parameter/constant data and produces a
// `proto::Model` ready to serialise into a `.mlpackage`. No FFI, so this
// builds and unit-tests on any host.

//! `loss` — extracted from the `mil` module for navigability (see `mod.rs`).

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
    /// Native softmax-cross-entropy-with-logits forward. Inputs `[logits [N,C],
    /// labels [N]]`, output per-row loss `[N] = logsumexp(logits) − logits[label]`.
    /// Mirrors `rlx_fusion::lower_softmax_cross_entropy_with_logits` but selects the
    /// label logit with one `one_hot`+`reduce_sum` instead of concatenating C columns.
    /// Pairs with the native backward so a full SCE training step stays off the O(C)
    /// decompose path on the ANE.
    #[cfg(feature = "training")]
    pub(crate) fn lower_softmax_cross_entropy_with_logits(
        &mut self,
        id: NodeId,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let logits_id = node.inputs[0];
        let labels_id = node.inputs[1];
        let logits_shape = self.graph.shape(logits_id).clone(); // [N, C]
        let out_shape = node.shape.clone(); // [N]
        let n = logits_shape.dim(0).unwrap_static();
        let c = logits_shape.dim(1).unwrap_static() as i32;
        let dt = logits_shape.dtype();
        let logits = self.val(logits_id);
        let kept = Shape::new(&[n, 1], dt); // [N,1]

        // lse = max + log(sum(exp(logits − max)))  (the numerically-stable logsumexp)
        let m = format!("{out_name}_max");
        self.emit(
            "reduce_max",
            &m,
            &kept,
            vec![
                ("x", bind_name(&logits)),
                ("axes", bind_value(vec_i32(&[-1]))),
                ("keep_dims", bind_value(scalar_bool(true))),
            ],
        )?;
        let shifted = format!("{out_name}_sh");
        self.emit(
            "sub",
            &shifted,
            &logits_shape,
            vec![("x", bind_name(&logits)), ("y", bind_name(&m))],
        )?;
        let exp_d = format!("{out_name}_exp");
        self.emit(
            "exp",
            &exp_d,
            &logits_shape,
            vec![("x", bind_name(&shifted))],
        )?;
        let sum_exp = format!("{out_name}_se");
        self.emit(
            "reduce_sum",
            &sum_exp,
            &out_shape,
            vec![
                ("x", bind_name(&exp_d)),
                ("axes", bind_value(vec_i32(&[-1]))),
                ("keep_dims", bind_value(scalar_bool(false))),
            ],
        )?;
        let log_sum = format!("{out_name}_ls");
        self.emit(
            "log",
            &log_sum,
            &out_shape,
            vec![
                ("x", bind_name(&sum_exp)),
                ("epsilon", bind_value(scalar_f32(1e-45))),
            ],
        )?;
        let m_flat = format!("{out_name}_mf");
        self.emit(
            "reshape",
            &m_flat,
            &out_shape,
            vec![
                ("x", bind_name(&m)),
                ("shape", bind_value(vec_i32(&[n as i32]))),
            ],
        )?;
        let lse = format!("{out_name}_lse");
        self.emit(
            "add",
            &lse,
            &out_shape,
            vec![("x", bind_name(&m_flat)), ("y", bind_name(&log_sum))],
        )?;

        // label_logit[n] = Σ_c logits[n,c]·onehot(labels)[n,c]
        let idx = format!("{out_name}_idx");
        let lshape = self.graph.shape(labels_id).clone().with_dtype(DType::I32);
        self.emit(
            "cast",
            &idx,
            &lshape,
            vec![
                ("x", bind_name(&self.val(labels_id))),
                ("dtype", bind_value(scalar_str("int32"))),
            ],
        )?;
        let oh = format!("{out_name}_oh");
        self.emit(
            "one_hot",
            &oh,
            &logits_shape,
            vec![
                ("indices", bind_name(&idx)),
                ("one_hot_vector_size", bind_value(scalar_i32(c))),
                ("axis", bind_value(scalar_i32(-1))),
                ("on_value", bind_value(scalar_f32(1.0))),
                ("off_value", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let masked = format!("{out_name}_msk");
        self.emit(
            "mul",
            &masked,
            &logits_shape,
            vec![("x", bind_name(&logits)), ("y", bind_name(&oh))],
        )?;
        let label_logit = format!("{out_name}_ll");
        self.emit(
            "reduce_sum",
            &label_logit,
            &out_shape,
            vec![
                ("x", bind_name(&masked)),
                ("axes", bind_value(vec_i32(&[-1]))),
                ("keep_dims", bind_value(scalar_bool(false))),
            ],
        )?;

        let op = self.simple_op(
            "sub",
            out_name,
            &out_shape,
            vec![("x", bind_name(&lse)), ("y", bind_name(&label_logit))],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }


    /// Native softmax-cross-entropy backward. Inputs `[logits [N,C], labels [N],
    /// d_loss [N]]`, output `dlogits [N,C] = (softmax(logits) − onehot(labels))·d_loss`.
    /// Mirrors `rlx_fusion::lower_softmax_cross_entropy_backward`, but emits MIL's
    /// single `one_hot` op instead of concatenating C class columns — the decompose
    /// path is O(C) graph nodes, which is unusable at LLM vocab sizes.
    #[cfg(feature = "training")]
    pub(crate) fn lower_softmax_cross_entropy_backward(&mut self, id: NodeId, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let logits_id = node.inputs[0];
        let labels_id = node.inputs[1];
        let d_loss_id = node.inputs[2];
        let full = node.shape.clone(); // [N, C]
        let n = full.dim(0).unwrap_static();
        let c = full.dim(1).unwrap_static() as i32;
        let logits = self.val(logits_id);

        // sm = softmax(logits, axis=-1)
        let sm = format!("{out_name}_sm");
        self.emit(
            "softmax",
            &sm,
            &full,
            vec![
                ("x", bind_name(&logits)),
                ("axis", bind_value(scalar_i32(-1))),
            ],
        )?;

        // onehot(labels): CoreML one_hot needs int indices; labels are f32-encoded.
        let idx = format!("{out_name}_idx");
        let lshape = self.graph.shape(labels_id).clone().with_dtype(DType::I32);
        self.emit(
            "cast",
            &idx,
            &lshape,
            vec![
                ("x", bind_name(&self.val(labels_id))),
                ("dtype", bind_value(scalar_str("int32"))),
            ],
        )?;
        let oh = format!("{out_name}_oh");
        self.emit(
            "one_hot",
            &oh,
            &full,
            vec![
                ("indices", bind_name(&idx)),
                ("one_hot_vector_size", bind_value(scalar_i32(c))),
                ("axis", bind_value(scalar_i32(-1))),
                ("on_value", bind_value(scalar_f32(1.0))),
                ("off_value", bind_value(scalar_f32(0.0))),
            ],
        )?;

        // diff = sm − onehot
        let diff = format!("{out_name}_diff");
        self.emit(
            "sub",
            &diff,
            &full,
            vec![("x", bind_name(&sm)), ("y", bind_name(&oh))],
        )?;

        // dlogits = diff · d_loss, with d_loss [N] reshaped to [N,1] to broadcast over C.
        let dl2 = format!("{out_name}_dl2");
        let dl2_shape = Shape::new(&[n, 1], full.dtype());
        self.emit(
            "reshape",
            &dl2,
            &dl2_shape,
            vec![
                ("x", bind_name(&self.val(d_loss_id))),
                ("shape", bind_value(vec_i32(&[n as i32, 1]))),
            ],
        )?;
        let op = self.simple_op(
            "mul",
            out_name,
            &full,
            vec![("x", bind_name(&diff)), ("y", bind_name(&dl2))],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

}
