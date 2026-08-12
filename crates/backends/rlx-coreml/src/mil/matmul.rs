// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// IR → CoreML ML Program (MIL) lowering. Pure data transformation: takes
// an RLX `Graph` plus baked parameter/constant data and produces a
// `proto::Model` ready to serialise into a `.mlpackage`. No FFI, so this
// builds and unit-tests on any host.

//! `matmul` — extracted from the `mil` module for navigability (see `mod.rs`).

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
    /// LoRA: `out = x·W + scale·(x·A)·B`. Inputs `[x, W, A, B]`.
    pub(crate) fn lower_lora_matmul(
        &mut self,
        id: NodeId,
        scale: f32,
        out_name: &str,
    ) -> Result<()> {
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
        // `n` is taken from the declared output; check the expert bank agrees
        // with it (and with K) rather than letting a `[E, N, K]` bank through.
        let w_shape = self.graph.shape(node.inputs[1]).clone();
        let gd = rlx_ir::shape::grouped_matmul_dims(&in_shape, &w_shape, Some(&shape))
            .map_err(|e| CoremlError::Unsupported(format!("node {id:?}: {e}")))?;
        let (m, k, n) = (gd.m, gd.k, gd.n);

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
                // Required by the iOS17+ MIL opset; the older-opset
                // post-pass in mod.rs strips it back out when unsupported.
                ("validate_indices", bind_value(scalar_bool(false))),
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

    /// `matmul(x, W) + bias` then optional activation.
    pub(crate) fn lower_fused_matmul_bias_act(
        &mut self,
        id: NodeId,
        activation: Option<Activation>,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let x = self.val(node.inputs[0]);
        let w = self.val(node.inputs[1]);
        let b = self.val(node.inputs[2]);
        let mm = format!("{out_name}_mm");
        self.matmul(&mm, &x, &w, &shape)?;
        let biased = format!("{out_name}_bias");
        self.emit(
            "add",
            &biased,
            &shape,
            vec![("x", bind_name(&mm)), ("y", bind_name(&b))],
        )?;
        self.emit_activation(&biased, activation, out_name, &shape)?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Split last axis in half: `up * silu(gate)`.
    pub(crate) fn lower_fused_swiglu(&mut self, id: NodeId, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let in_shape = self.graph.shape(node.inputs[0]).clone();
        let out_shape = node.shape.clone();
        let rank = in_shape.rank();
        let last = in_shape.dim(rank - 1).unwrap_static();
        if !last.is_multiple_of(2) {
            return Err(CoremlError::Unsupported(format!(
                "FusedSwiGLU: last dim {last} must be even"
            )));
        }
        let half = last / 2;
        // `gate_first` swaps the halves: default (false) is `[up | gate]`, true is
        // `[gate | up]` — what a builder emits when gate_proj precedes up_proj
        // (DeepSeek / Bailing MoE). Dropping it computes `gate · silu(up)`.
        let gate_first = matches!(
            node.op,
            rlx_ir::Op::FusedSwiGLU {
                gate_first: true,
                ..
            }
        );
        let (up_off, gate_off) = if gate_first { (half, 0) } else { (0, half) };
        let x = self.val(node.inputs[0]);
        let mut begin_up = vec![0i32; rank];
        begin_up[rank - 1] = up_off as i32;
        let mut size_up: Vec<i32> = in_shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i32)
            .collect();
        size_up[rank - 1] = half as i32;
        let mut begin_g = vec![0i32; rank];
        begin_g[rank - 1] = gate_off as i32;
        let size_g = size_up.clone();
        let up = format!("{out_name}_up");
        self.emit(
            "slice_by_size",
            &up,
            &out_shape,
            vec![
                ("x", bind_name(&x)),
                ("begin", bind_value(vec_i32(&begin_up))),
                ("size", bind_value(vec_i32(&size_up))),
            ],
        )?;
        let gate = format!("{out_name}_gate");
        self.emit(
            "slice_by_size",
            &gate,
            &out_shape,
            vec![
                ("x", bind_name(&x)),
                ("begin", bind_value(vec_i32(&begin_g))),
                ("size", bind_value(vec_i32(&size_g))),
            ],
        )?;
        let silu = format!("{out_name}_silu");
        self.emit_activation(&gate, Some(Activation::Silu), &silu, &out_shape)?;
        self.emit(
            "mul",
            out_name,
            &out_shape,
            vec![("x", bind_name(&up)), ("y", bind_name(&silu))],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }
}
