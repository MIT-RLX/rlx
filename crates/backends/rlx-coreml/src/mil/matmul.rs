// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// IR → CoreML ML Program (MIL) lowering. Pure data transformation: takes
// an RLX `Graph` plus baked parameter/constant data and produces a
// `proto::Model` ready to serialise into a `.mlpackage`. No FFI, so this
// builds and unit-tests on any host.

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
    /// LoRA: `out = x·W + scale·(x·A)·B`. Inputs `[x, W, A, B]`.
    pub(crate) fn lower_lora_matmul(&mut self, id: NodeId, scale: f32, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let m = dim_static(&shape, 0)?;
        let n = dim_static(&shape, 1)?;
        let x = self.val(node.inputs[0]);
        let w = self.val(node.inputs[1]);
        let a = self.val(node.inputs[2]);
        let b = self.val(node.inputs[3]);
        let r = dim_static(&self.graph.shape(node.inputs[2]).clone(), 1)?;

        let xa = format!("{out_name}_xa");
        self.matmul(&xa, &x, &a, &Shape::new(&[m, r], DType::F32))?;
        let xab = format!("{out_name}_xab");
        self.matmul(&xab, &xa, &b, &shape)?;
        let scaled = format!("{out_name}_lora");
        self.emit(
            "mul",
            &scaled,
            &shape,
            vec![("x", bind_name(&xab)), ("y", bind_value(scalar_f32(scale)))],
        )?;
        let xw = format!("{out_name}_xw");
        self.matmul(&xw, &x, &w, &shape)?;
        self.emit(
            "add",
            out_name,
            &shape,
            vec![("x", bind_name(&xw)), ("y", bind_name(&scaled))],
        )?;
        let _ = n;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }


    /// MoE grouped matmul. Inputs `[input(M,K), weight(E,K,N),
    /// expert_idx(M)]`: each row picks its expert's weight slab.
    /// `out[m] = input[m] · weight[expert_idx[m]]`. Lowered as
    /// gather-then-batched-matmul.
    pub(crate) fn lower_grouped_matmul(&mut self, id: NodeId, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone(); // [M, N]
        let in_shape = self.graph.shape(node.inputs[0]).clone();
        let m = dim_static(&in_shape, in_shape.rank() - 2)?;
        let k = dim_static(&in_shape, in_shape.rank() - 1)?;
        let n = dim_static(&shape, shape.rank() - 1)?;

        let input = self.val(node.inputs[0]);
        let weight = self.val(node.inputs[1]);
        let eidx = self.val(node.inputs[2]);

        // expert_idx f32 → int32
        let eidx_i32 = format!("{out_name}_eidx");
        let eidx_shape = self
            .graph
            .shape(node.inputs[2])
            .clone()
            .with_dtype(DType::I32);
        self.emit(
            "cast",
            &eidx_i32,
            &eidx_shape,
            vec![
                ("x", bind_name(&eidx)),
                ("dtype", bind_value(scalar_str("int32"))),
            ],
        )?;
        // W_sel = gather(weight, eidx, axis=0) -> [M, K, N]
        let wsel = format!("{out_name}_wsel");
        self.emit(
            "gather",
            &wsel,
            &Shape::new(&[m, k, n], DType::F32),
            vec![
                ("x", bind_name(&weight)),
                ("indices", bind_name(&eidx_i32)),
                ("axis", bind_value(scalar_i32(0))),
            ],
        )?;
        // input [M,K] -> [M,1,K]
        let in3 = format!("{out_name}_in3");
        self.reshape_to(
            &input,
            &[m as i64, 1, k as i64],
            &Shape::new(&[m, 1, k], DType::F32),
            &in3,
        )?;
        // batched matmul [M,1,K] @ [M,K,N] -> [M,1,N]
        let mm = format!("{out_name}_mm");
        self.matmul(&mm, &in3, &wsel, &Shape::new(&[m, 1, n], DType::F32))?;
        // -> [M, N]
        self.reshape_to(&mm, &[m as i64, n as i64], &shape, out_name)?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

}
