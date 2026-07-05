// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// IR → CoreML ML Program (MIL) lowering. Pure data transformation: takes
// an RLX `Graph` plus baked parameter/constant data and produces a
// `proto::Model` ready to serialise into a `.mlpackage`. No FFI, so this
// builds and unit-tests on any host.

//! `reduce_index` — extracted from the `mil` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use super::helpers::simple_op_flex;
use super::helpers::*;
use crate::proto;
use crate::{CoremlError, Result};
use rlx_ir::op::{Activation, CmpOp, MaskKind, ReduceOp};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Dim, Graph, NodeId, Op, Shape};
use std::collections::HashMap;

use super::*;

impl<'a> LowerCtx<'a> {
    /// Top-k along the last axis. The IR result is the f32-encoded
    /// **indices** (descending by value, ties → smaller index), so we emit
    /// MIL `topk` (a two-output op: values + indices) and cast the int32
    /// indices to f32.
    pub(crate) fn lower_topk(&mut self, id: NodeId, k: usize, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone(); // [..., k], f32 (indices encoded)
        let x = self.val(node.inputs[0]);
        let axis = (shape.rank() - 1) as i32;

        let values = format!("{out_name}_vals");
        let indices = format!("{out_name}_idx_i32");
        let vals_ty = named_value_type(&values, &shape)?;
        let idx_ty = named_value_type(&indices, &shape.clone().with_dtype(DType::I32))?;

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), bind_name(&x));
        inputs.insert("k".to_string(), bind_value(scalar_i32(k as i32)));
        inputs.insert("axis".to_string(), bind_value(scalar_i32(axis)));
        inputs.insert("ascending".to_string(), bind_value(scalar_bool(false)));
        let mut attributes = HashMap::new();
        attributes.insert("name".to_string(), scalar_str(out_name));
        self.operations.push(proto::Operation {
            r#type: "topk".to_string(),
            inputs,
            outputs: vec![vals_ty, idx_ty],
            blocks: vec![],
            attributes,
        });

        // Cast int32 indices → f32 to match the IR's encoding.
        self.emit(
            "cast",
            out_name,
            &shape,
            vec![
                ("x", bind_name(&indices)),
                ("dtype", bind_value(scalar_str("fp32"))),
            ],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// `ArgMax` / `ArgMin` along one axis; indices are f32-encoded at the IR boundary.
    pub(crate) fn lower_argreduce(
        &mut self,
        id: NodeId,
        axis: usize,
        keep_dim: bool,
        is_max: bool,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let in_shape = self.graph.shape(node.inputs[0]).clone();
        let rank = in_shape.rank();
        let ax = axis as i32;
        let _ = rank;
        let mut x = self.val(node.inputs[0]);
        if !is_max {
            let neg = format!("{out_name}_neg");
            self.emit(
                "mul",
                &neg,
                &in_shape,
                vec![("x", bind_name(&x)), ("y", bind_value(scalar_f32(-1.0)))],
            )?;
            x = neg;
        }
        let idx_i32 = format!("{out_name}_idx_i32");
        let idx_shape = node.shape.clone().with_dtype(DType::I32);
        self.emit(
            "reduce_argmax",
            &idx_i32,
            &idx_shape,
            vec![
                ("x", bind_name(&x)),
                ("axis", bind_value(scalar_i32(ax))),
                ("keep_dims", bind_value(scalar_bool(keep_dim))),
            ],
        )?;
        self.emit(
            "cast",
            out_name,
            &node.shape,
            vec![
                ("x", bind_name(&idx_i32)),
                ("dtype", bind_value(scalar_str("fp32"))),
            ],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Batch-general flip along `axes` via per-axis `gather` with reversed indices.
    pub(crate) fn lower_reverse(
        &mut self,
        id: NodeId,
        axes: &[usize],
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let in_shape = self.graph.shape(node.inputs[0]).clone();
        if axes.is_empty() {
            let x = self.val(node.inputs[0]);
            self.names.insert(id.0, x);
            return Ok(());
        }
        let mut cur = self.val(node.inputs[0]);
        let shape = in_shape.clone();
        for &ax in axes {
            let d = dim_static(&shape, ax)?;
            let idx_f: Vec<f32> = (0..d).rev().map(|i| i as f32).collect();
            let idx_name = format!("{out_name}_rev_{ax}");
            self.operations.push(make_const(
                &mut self.blob,
                &idx_name,
                &Shape::new(&[d], DType::F32),
                &idx_f,
            )?);
            let idx_i32 = format!("{idx_name}_i32");
            self.emit(
                "cast",
                &idx_i32,
                &Shape::new(&[d], DType::I32),
                vec![
                    ("x", bind_name(&idx_name)),
                    ("dtype", bind_value(scalar_str("int32"))),
                ],
            )?;
            let next = format!("{out_name}_g{ax}");
            self.emit(
                "gather",
                &next,
                &shape,
                vec![
                    ("x", bind_name(&cur)),
                    ("indices", bind_name(&idx_i32)),
                    ("axis", bind_value(scalar_i32(ax as i32))),
                ],
            )?;
            cur = next;
        }
        self.names.insert(id.0, cur);
        Ok(())
    }
}
