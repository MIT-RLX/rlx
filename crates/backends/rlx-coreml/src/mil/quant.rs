// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// IR → CoreML ML Program (MIL) lowering. Pure data transformation: takes
// an RLX `Graph` plus baked parameter/constant data and produces a
// `proto::Model` ready to serialise into a `.mlpackage`. No FFI, so this
// builds and unit-tests on any host.

//! `quant` — extracted from the `mil` module for navigability (see `mod.rs`).

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
    /// Fetch the GGUF/quantized bytes for the `Param` weight at `w_id`.
    pub(crate) fn quant_bytes(&self, w_id: NodeId) -> Result<&[u8]> {
        match &self.graph.node(w_id).op {
            Op::Param { name } => self
                .typed_params
                .get(name)
                .map(|(b, _)| b.as_slice())
                .ok_or_else(|| CoremlError::Runtime(format!("missing quantized param '{name}'"))),
            Op::Constant { data } => Ok(data.as_slice()),
            other => Err(CoremlError::Unsupported(format!(
                "dequant weight must be a Param/Constant, got {other:?}"
            ))),
        }
    }

    /// Bake on-device dequantized weights `[n,k]` as MIL constants + `mul`/`sub`.
    ///
    /// Supports Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4NL, Q4/5/8_K, Q2/3/6_K. Scale tensors are
    /// `[nb,1]` or `[nb,32]` depending on scheme (see `split_gguf_ondevice`).
    /// Documented in [docs/gguf-backend-paths.md](../../../docs/gguf-backend-paths.md).
    pub(crate) fn bake_ondevice_weight(
        &mut self,
        prefix: &str,
        scheme: QuantScheme,
        bytes: &[u8],
        n: usize,
        k: usize,
    ) -> Result<String> {
        const QK: usize = 32;
        let nb = (k * n) / QK;
        if nb * QK != k * n {
            return Err(CoremlError::Runtime(format!(
                "ondevice dequant: {n}x{k} not divisible by {QK}"
            )));
        }
        let (qs, scales, offsets) = split_gguf_ondevice(scheme, bytes, nb)?;
        let per_elem_scales = scales.len() == nb * QK;
        let sc_shape = if per_elem_scales {
            Shape::new(&[nb, QK], DType::F32)
        } else {
            Shape::new(&[nb, 1], DType::F32)
        };
        let q_name = format!("{prefix}_q");
        self.operations.push(make_const(
            &mut self.blob,
            &q_name,
            &Shape::new(&[nb, QK], DType::F32),
            &qs,
        )?);
        let sc_name = format!("{prefix}_sc");
        self.operations
            .push(make_const(&mut self.blob, &sc_name, &sc_shape, &scales)?);
        let mul_name = format!("{prefix}_mul");
        self.emit(
            "mul",
            &mul_name,
            &Shape::new(&[nb, QK], DType::F32),
            vec![("x", bind_name(&q_name)), ("y", bind_name(&sc_name))],
        )?;
        let dq = if offsets.iter().any(|&o| o != 0.0) {
            let per_elem_offsets = offsets.len() == nb * QK;
            let off_shape = if per_elem_offsets {
                Shape::new(&[nb, QK], DType::F32)
            } else {
                Shape::new(&[nb, 1], DType::F32)
            };
            let off_name = format!("{prefix}_off");
            self.operations
                .push(make_const(&mut self.blob, &off_name, &off_shape, &offsets)?);
            let sub_name = format!("{prefix}_dq");
            self.emit(
                "sub",
                &sub_name,
                &Shape::new(&[nb, QK], DType::F32),
                vec![("x", bind_name(&mul_name)), ("y", bind_name(&off_name))],
            )?;
            sub_name
        } else {
            mul_name
        };
        let wc = format!("{prefix}_w");
        self.reshape_to(
            &dq,
            &[n as i64, k as i64],
            &Shape::new(&[n, k], DType::F32),
            &wc,
        )?;
        Ok(wc)
    }

    /// On-device block dequant for supported GGUF schemes, then MIL matmul.
    pub(crate) fn lower_dequant_matmul_ondevice(
        &mut self,
        id: NodeId,
        scheme: QuantScheme,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let out_shape = node.shape.clone();
        let x_id = node.inputs[0];
        let w_id = node.inputs[1];
        let n = dim_static(&out_shape, out_shape.rank() - 1)?;
        let m = out_shape.num_elements().unwrap_or(0) / n.max(1);
        let k = self.graph.shape(x_id).num_elements().unwrap_or(0) / m.max(1);
        let bytes = self.quant_bytes(w_id)?.to_vec();
        let wc = self.bake_ondevice_weight(out_name, scheme, &bytes, n, k)?;
        let x = self.val(x_id);
        let op = self.simple_op(
            "matmul",
            out_name,
            &out_shape,
            vec![
                ("x", bind_name(&x)),
                ("y", bind_name(&wc)),
                ("transpose_x", bind_value(scalar_bool(false))),
                ("transpose_y", bind_value(scalar_bool(true))),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// `x @ dequant(W)ᵀ`. GGUF weights are stored `[N, K]` (B-transposed),
    /// so we host-dequantize to f32 `[N, K]`, bake it, and matmul with
    /// `transpose_y`. The dequant happens at finalize (weights present),
    /// trading the proto's on-device dequant for size — correct + simple.
    pub(crate) fn lower_dequant_matmul(
        &mut self,
        id: NodeId,
        scheme: QuantScheme,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let out_shape = node.shape.clone();
        let x_id = node.inputs[0];
        let w_id = node.inputs[1];
        let n = dim_static(&out_shape, out_shape.rank() - 1)?;
        let m = out_shape.num_elements().unwrap_or(0) / n.max(1);
        let k = self.graph.shape(x_id).num_elements().unwrap_or(0) / m.max(1);

        let wf = dequant_scheme(scheme, self.quant_bytes(w_id)?, k * n)?;
        let x = self.val(x_id);
        let wc = format!("{out_name}_w");
        self.operations.push(make_const(
            &mut self.blob,
            &wc,
            &Shape::new(&[n, k], DType::F32),
            &wf,
        )?);
        let op = self.simple_op(
            "matmul",
            out_name,
            &out_shape,
            vec![
                ("x", bind_name(&x)),
                ("y", bind_name(&wc)),
                ("transpose_x", bind_value(scalar_bool(false))),
                ("transpose_y", bind_value(scalar_bool(true))),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Dequantize packed MoE weights to a plain f32 const (no matmul).
    pub(crate) fn lower_dequant_moe_weights(
        &mut self,
        id: NodeId,
        scheme: QuantScheme,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let total = shape.num_elements().unwrap_or(0);
        let wf = dequant_scheme(scheme, self.quant_bytes(node.inputs[0])?, total)?;
        self.operations
            .push(make_const(&mut self.blob, out_name, &shape, &wf)?);
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// MoE grouped matmul with on-device Q8_0 / Q4_0 / IQ4NL / K-quant dequant.
    pub(crate) fn lower_dequant_grouped_matmul_ondevice(
        &mut self,
        id: NodeId,
        scheme: QuantScheme,
        out_name: &str,
    ) -> Result<()> {
        const QK: usize = 32;
        let node = self.graph.node(id);
        let out_shape = node.shape.clone();
        let in_shape = self.graph.shape(node.inputs[0]).clone();
        let m = dim_static(&in_shape, in_shape.rank() - 2)?;
        let k = dim_static(&in_shape, in_shape.rank() - 1)?;
        let n = dim_static(&out_shape, out_shape.rank() - 1)?;
        let bytes = self.quant_bytes(node.inputs[1])?;
        let block_elems = scheme.gguf_block_size() as usize;
        let block_bytes = scheme.gguf_block_bytes() as usize;
        let slab_bytes = (k * n) / block_elems.max(1) * block_bytes;
        let num_experts = bytes.len() / slab_bytes.max(1);
        let nb_per_expert = (k * n) / QK;
        if nb_per_expert * QK != k * n {
            return self.lower_dequant_grouped_matmul(id, scheme, out_name);
        }

        let mut all_qs = Vec::with_capacity(num_experts * nb_per_expert * QK);
        let mut all_sc = Vec::with_capacity(num_experts * nb_per_expert);
        let mut all_off = Vec::with_capacity(num_experts * nb_per_expert);
        for e in 0..num_experts {
            let slab = &bytes[e * slab_bytes..(e + 1) * slab_bytes];
            let (qs, sc, off) = split_gguf_ondevice(scheme, slab, nb_per_expert)?;
            all_qs.extend(qs);
            all_sc.extend(sc);
            all_off.extend(off);
        }
        let nb_total = num_experts * nb_per_expert;
        let q_name = format!("{out_name}_q");
        self.operations.push(make_const(
            &mut self.blob,
            &q_name,
            &Shape::new(&[nb_total, QK], DType::F32),
            &all_qs,
        )?);
        let sc_name = format!("{out_name}_sc");
        self.operations.push(make_const(
            &mut self.blob,
            &sc_name,
            &Shape::new(&[nb_total, 1], DType::F32),
            &all_sc,
        )?);
        let mul_name = format!("{out_name}_mul");
        self.emit(
            "mul",
            &mul_name,
            &Shape::new(&[nb_total, QK], DType::F32),
            vec![("x", bind_name(&q_name)), ("y", bind_name(&sc_name))],
        )?;
        let dq = if all_off.iter().any(|&o| o != 0.0) {
            let off_name = format!("{out_name}_off");
            self.operations.push(make_const(
                &mut self.blob,
                &off_name,
                &Shape::new(&[nb_total, 1], DType::F32),
                &all_off,
            )?);
            let sub_name = format!("{out_name}_dq");
            self.emit(
                "sub",
                &sub_name,
                &Shape::new(&[nb_total, QK], DType::F32),
                vec![("x", bind_name(&mul_name)), ("y", bind_name(&off_name))],
            )?;
            sub_name
        } else {
            mul_name
        };
        let weight = format!("{out_name}_wdq");
        self.reshape_to(
            &dq,
            &[num_experts as i64, n as i64, k as i64],
            &Shape::new(&[num_experts, n, k], DType::F32),
            &weight,
        )?;

        let input = self.val(node.inputs[0]);
        let eidx = self.val(node.inputs[2]);
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
        let wsel = format!("{out_name}_wsel");
        self.emit(
            "gather",
            &wsel,
            &Shape::new(&[m, n, k], DType::F32),
            vec![
                ("x", bind_name(&weight)),
                ("indices", bind_name(&eidx_i32)),
                ("axis", bind_value(scalar_i32(0))),
            ],
        )?;
        let in3 = format!("{out_name}_in3");
        self.reshape_to(
            &input,
            &[m as i64, 1, k as i64],
            &Shape::new(&[m, 1, k], DType::F32),
            &in3,
        )?;
        let mm = format!("{out_name}_mm");
        self.emit(
            "matmul",
            &mm,
            &Shape::new(&[m, 1, n], DType::F32),
            vec![
                ("x", bind_name(&in3)),
                ("y", bind_name(&wsel)),
                ("transpose_x", bind_value(scalar_bool(false))),
                ("transpose_y", bind_value(scalar_bool(true))),
            ],
        )?;
        self.reshape_to(&mm, &[m as i64, n as i64], &out_shape, out_name)?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// MoE grouped matmul with quantized expert weights. Dequantizes all
    /// `E` expert slabs (`[E, N, K]`), gathers per token, then batched
    /// matmul with `transpose_y`.
    pub(crate) fn lower_dequant_grouped_matmul(
        &mut self,
        id: NodeId,
        scheme: QuantScheme,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let out_shape = node.shape.clone();
        let in_shape = self.graph.shape(node.inputs[0]).clone();
        let m = dim_static(&in_shape, in_shape.rank() - 2)?;
        let k = dim_static(&in_shape, in_shape.rank() - 1)?;
        let n = dim_static(&out_shape, out_shape.rank() - 1)?;

        // Dequant every expert slab. Block-byte math gives the expert count.
        let bytes = self.quant_bytes(node.inputs[1])?;
        let block_elems = scheme.gguf_block_size() as usize;
        let block_bytes = scheme.gguf_block_bytes() as usize;
        let slab_bytes = (k * n) / block_elems.max(1) * block_bytes;
        let num_experts = bytes.len() / slab_bytes.max(1);
        let total = num_experts * n * k;
        let wf = dequant_scheme(scheme, bytes, total)?;

        let weight = format!("{out_name}_wdq");
        self.operations.push(make_const(
            &mut self.blob,
            &weight,
            &Shape::new(&[num_experts, n, k], DType::F32),
            &wf,
        )?);

        let input = self.val(node.inputs[0]);
        let eidx = self.val(node.inputs[2]);
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
        // gather expert slabs → [M, N, K]
        let wsel = format!("{out_name}_wsel");
        self.emit(
            "gather",
            &wsel,
            &Shape::new(&[m, n, k], DType::F32),
            vec![
                ("x", bind_name(&weight)),
                ("indices", bind_name(&eidx_i32)),
                ("axis", bind_value(scalar_i32(0))),
            ],
        )?;
        // input [M,K] → [M,1,K]; matmul([M,1,K],[M,N,K]ᵀ) → [M,1,N] → [M,N]
        let in3 = format!("{out_name}_in3");
        self.reshape_to(
            &input,
            &[m as i64, 1, k as i64],
            &Shape::new(&[m, 1, k], DType::F32),
            &in3,
        )?;
        let mm = format!("{out_name}_mm");
        self.emit(
            "matmul",
            &mm,
            &Shape::new(&[m, 1, n], DType::F32),
            vec![
                ("x", bind_name(&in3)),
                ("y", bind_name(&wsel)),
                ("transpose_x", bind_value(scalar_bool(false))),
                ("transpose_y", bind_value(scalar_bool(true))),
            ],
        )?;
        self.reshape_to(&mm, &[m as i64, n as i64], &out_shape, out_name)?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Bake an affine (scale / zero-point) parameter as a const that
    /// broadcasts against a rank-`rank` tensor: a scalar for per-tensor
    /// quant, or a `[1,…,C,…,1]` vector along `axis` for per-channel.
    pub(crate) fn bake_affine(
        &mut self,
        name: &str,
        values: &[f32],
        axis: Option<usize>,
        rank: usize,
    ) -> Result<()> {
        let op = match axis {
            Some(ax) if values.len() > 1 => {
                let mut dims = vec![1usize; rank];
                dims[ax] = values.len();
                make_const(&mut self.blob, name, &Shape::new(&dims, DType::F32), values)?
            }
            // Per-tensor: a rank-0 scalar.
            _ => make_const(
                &mut self.blob,
                name,
                &Shape::new(&[], DType::F32),
                &[values[0]],
            )?,
        };
        self.operations.push(op);
        Ok(())
    }

    /// Dequantize an int8 tensor: `out = (cast(q,f32) - zp) · scale`.
    pub(crate) fn lower_dequantize(
        &mut self,
        id: NodeId,
        axis: Option<usize>,
        scales: &[f32],
        zero_points: &[i32],
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone(); // f32 output
        let rank = shape.rank();
        let q = self.val(node.inputs[0]);

        // MIL ios16 has no int8 activations: quantized values flow as
        // integer-valued fp32 (from `Quantize`). Only an int32 producer
        // needs an explicit cast.
        let in_dt = self.graph.shape(node.inputs[0]).dtype();
        let qf = if in_dt == DType::I32 {
            let c = format!("{out_name}_qf");
            self.emit(
                "cast",
                &c,
                &shape,
                vec![
                    ("x", bind_name(&q)),
                    ("dtype", bind_value(scalar_str("fp32"))),
                ],
            )?;
            c
        } else {
            q
        };
        let zp: Vec<f32> = zero_points.iter().map(|&z| z as f32).collect();
        let zpc = format!("{out_name}_zp");
        self.bake_affine(&zpc, &zp, axis, rank)?;
        let sub = format!("{out_name}_sub");
        self.emit(
            "sub",
            &sub,
            &shape,
            vec![("x", bind_name(&qf)), ("y", bind_name(&zpc))],
        )?;
        let sc = format!("{out_name}_sc");
        self.bake_affine(&sc, scales, axis, rank)?;
        self.emit(
            "mul",
            out_name,
            &shape,
            vec![("x", bind_name(&sub)), ("y", bind_name(&sc))],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Quantize a f32 tensor to int8:
    /// `out = cast(clip(round(x/scale) + zp, -128, 127), int8)`.
    pub(crate) fn lower_quantize(
        &mut self,
        id: NodeId,
        axis: Option<usize>,
        scales: &[f32],
        zero_points: &[i32],
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone(); // int8 output
        let f32_shape = shape.clone().with_dtype(DType::F32);
        let rank = shape.rank();
        let x = self.val(node.inputs[0]);

        let inv: Vec<f32> = scales.iter().map(|&s| 1.0 / s).collect();
        let invc = format!("{out_name}_inv");
        self.bake_affine(&invc, &inv, axis, rank)?;
        let scaled = format!("{out_name}_xs");
        self.emit(
            "mul",
            &scaled,
            &f32_shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&invc))],
        )?;
        let rounded = format!("{out_name}_rnd");
        self.emit(
            "round",
            &rounded,
            &f32_shape,
            vec![("x", bind_name(&scaled))],
        )?;
        let zp: Vec<f32> = zero_points.iter().map(|&z| z as f32).collect();
        let zpc = format!("{out_name}_zp");
        self.bake_affine(&zpc, &zp, axis, rank)?;
        let shifted = format!("{out_name}_shift");
        self.emit(
            "add",
            &shifted,
            &f32_shape,
            vec![("x", bind_name(&rounded)), ("y", bind_name(&zpc))],
        )?;
        // MIL ios16 `cast` can't target int8, so the quantized value stays
        // as integer-valued fp32 (the IR's I8 type is satisfied logically;
        // `Dequantize` consumes the fp32 representation directly).
        self.emit(
            "clip",
            out_name,
            &f32_shape,
            vec![
                ("x", bind_name(&shifted)),
                ("alpha", bind_value(scalar_f32(-128.0))),
                ("beta", bind_value(scalar_f32(127.0))),
            ],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }
}
