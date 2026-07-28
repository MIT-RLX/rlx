// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
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

/// How Q1_0 weights are lowered for on-device CoreML dequant. All non-F32 modes
/// keep the weight COMPRESSED in weight.bin (no full f32 unfold).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Q1Mode {
    /// Legacy: expand qs to f32 constants (multi-GiB; 64M-elem capped).
    F32,
    /// 1-bit LUT palettization via `constexpr_lut_to_dense` (iOS18 opset) —
    /// packed UINT1 indices (~n·k/8 bytes, the true no-unfold size, ~3.4 GB for
    /// Bonsai-27B) + per-128-block LUT {−d,+d}. Validated on-device against the
    /// dequant reference (see `coreml_quant`). Preferred non-unfold path.
    Lut,
}

fn q1_ondevice_mode() -> Q1Mode {
    match std::env::var("RLX_COREML_Q1_MODE").as_deref() {
        // Legacy multi-GiB F32 unfold (disk OOM on 27B). Opt in only when
        // deliberately comparing against the old bake path.
        Ok("f32") | Ok("F32") => Q1Mode::F32,
        // Default + aliases: 1-bit LUT palettization (no F32 unfold).
        // "int8"/"affine" used to select the deprecated ios16 int8 constexpr
        // path; on iOS18 they map here (smaller, and affine no longer takes
        // `quantized_data`).
        Ok("lut") | Ok("int8") | Ok("affine") | Err(_) => Q1Mode::Lut,
        Ok(other) => {
            eprintln!(
                "[rlx-coreml] unknown RLX_COREML_Q1_MODE={other:?}; using lut \
                 (set f32 for the legacy unfold)"
            );
            Q1Mode::Lut
        }
    }
}

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
        let elems = n.saturating_mul(k);
        // Q1_0 defaults to Lut (1-bit palettization, no F32 unfold). Opt into
        // the legacy multi-GiB F32 bake with RLX_COREML_Q1_MODE=f32 — that path
        // alone keeps the 64M-elem guard (27B F32 qs fills the disk).
        if matches!(scheme, QuantScheme::GgufQ1_0) {
            let mode = self.opts.q1_mode.unwrap_or_else(q1_ondevice_mode);
            match mode {
                Q1Mode::Lut => return self.bake_q1_lut(prefix, bytes, n, k),
                Q1Mode::F32 => {}
            }
        }
        // qs is baked as F32 (±1 / nibbles / …). A single 27B Q1_0 projection
        // is ~n·k f32 ≈ multi-GiB and writing it into weight.bin OOMs the disk
        // (Bonsai-27B prefills produced ~94 GiB mlpackages). Refuse loudly.
        const MAX_ONDEVICE_WEIGHT_ELEMS: usize = 64 * 1024 * 1024; // 64M → 256 MiB qs
        if elems > MAX_ONDEVICE_WEIGHT_ELEMS {
            return Err(CoremlError::Unsupported(format!(
                "CoreML on-device dequant refuses {scheme:?} weight [{n}×{k}] \
                 ({elems} elems ≈ {:.1} GiB as F32 qs). For Q1_0 use the default \
                 Lut path (or RLX_COREML_Q1_MODE=lut); do not set MODE=f32 on \
                 27B-class models. Other schemes need Metal/MLX/CUDA/CPU.",
                (elems * 4) as f64 / (1u64 << 30) as f64
            )));
        }
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

    /// Non-unfolding Q1_0 via 1-bit LUT palettization (`constexpr_lut_to_dense`)
    /// — the true no-unfold path. The weight stays 1-bit packed in weight.bin
    /// (~n·k/8 bytes) and CoreML dequants on ANE. Q1_0's own sign bytes ARE the
    /// palette indices (bit1→+d bit0→−d, LSB-first), so the packed indices are
    /// copied straight out of the block; the per-128-block scale `d` becomes a
    /// per-group LUT `{−d, +d}`. Requires the UINT1 proto DataType (added to
    /// coreml.proto). Weight `[n,k]` row-major, blocks 128-along-k.
    fn bake_q1_lut(&mut self, prefix: &str, bytes: &[u8], n: usize, k: usize) -> Result<String> {
        const BLK: usize = 18; // Q1_0 block: 2 (f16 d) + 16 sign bytes
        const GS: usize = 128; // group / block size along k
        if !k.is_multiple_of(GS) {
            return Err(CoremlError::Runtime(format!(
                "Q1_0 lut: k={k} not a multiple of {GS}"
            )));
        }
        let kg = k / GS; // 128-blocks per row
        let n_blocks = n * kg;
        if bytes.len() < n_blocks * BLK {
            return Err(CoremlError::Runtime(format!(
                "Q1_0 lut: need {} bytes, got {}",
                n_blocks * BLK,
                bytes.len()
            )));
        }
        // Extract packed 1-bit indices (= the sign bytes, already LSB-first) and
        // a per-block LUT {−d, +d}, both in row-major (n, kg) block order.
        let mut indices = Vec::with_capacity(n * k / 8);
        let mut lut = Vec::with_capacity(n_blocks * 2);
        for b in 0..n_blocks {
            let base = b * BLK;
            let d = read_f16_le(&bytes[base..base + 2]);
            lut.push(-d);
            lut.push(d);
            indices.extend_from_slice(&bytes[base + 2..base + BLK]); // 16 sign bytes
        }
        let idx_name = format!("{prefix}_idx");
        self.operations.push(make_const_u1_packed(
            &mut self.blob,
            &idx_name,
            &[n, k],
            &indices,
        )?);
        // LUT [n, kg, 2, 1] f16: grouped palettization (group_size 128 along k,
        // 1 along n; 2 palette entries = 2^1; scalar vector).
        let lut_name = format!("{prefix}_lut");
        self.operations.push(make_const_float(
            &mut self.blob,
            &lut_name,
            &Shape::new(&[n, kg, 2, 1], DType::F16),
            &lut,
            DType::F16,
        )?);
        // `constexpr_lut_to_dense` requires lut.dtype == output.dtype (f16 here).
        let dq_name = format!("{prefix}_w");
        self.emit(
            "constexpr_lut_to_dense",
            &dq_name,
            &Shape::new(&[n, k], DType::F16),
            vec![
                ("indices", bind_name(&idx_name)),
                ("lut", bind_name(&lut_name)),
            ],
        )?;
        Ok(dq_name)
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
    ///
    /// Linear int schemes (`Int8Block` / `Int8BlockAsym` / `Int4Block`) carry
    /// separate scale (+ zp) side tensors; those are host-folded the same way.
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

        const MAX_HOST_BAKE_ELEMS: usize = 64 * 1024 * 1024;
        let elems = n.saturating_mul(k);
        if elems > MAX_HOST_BAKE_ELEMS {
            return Err(CoremlError::Unsupported(format!(
                "CoreML host-bake dequant refuses {scheme:?} weight [{n}×{k}] \
                 ({elems} elems ≈ {:.1} GiB F32). Use Metal/MLX/CUDA/CPU for \
                 packed 27B-class models.",
                (elems * 4) as f64 / (1u64 << 30) as f64
            )));
        }

        let wf = match scheme {
            QuantScheme::Int8Block { block_size } | QuantScheme::Int8BlockAsym { block_size } => {
                let asym = matches!(scheme, QuantScheme::Int8BlockAsym { .. });
                if node.inputs.len() < 3 + usize::from(asym) {
                    return Err(CoremlError::Runtime(format!(
                        "DequantMatMul {scheme:?} needs scale{} inputs",
                        if asym { "+zp" } else { "" }
                    )));
                }
                let w_bytes = self.quant_bytes(w_id)?;
                if w_bytes.len() < k * n {
                    return Err(CoremlError::Runtime(format!(
                        "Int8 weight bytes {} < k*n={}",
                        w_bytes.len(),
                        k * n
                    )));
                }
                let bs = block_size.max(1) as usize;
                let n_blocks = k.div_ceil(bs);
                let scales = self.f32_side_tensor(node.inputs[2], n_blocks * n, "scale")?;
                let zps = if asym {
                    self.f32_side_tensor(node.inputs[3], n_blocks * n, "zp")?
                } else {
                    Vec::new()
                };
                // Bake as `[n, k]` (B-transposed) to match the GGUF path.
                let mut wf = vec![0.0f32; n * k];
                for p in 0..k {
                    let block = p / bs;
                    for j in 0..n {
                        let q = w_bytes[p * n + j] as i8 as f32;
                        let s = scales[block * n + j];
                        let z = if asym { zps[block * n + j] } else { 0.0 };
                        wf[j * k + p] = (q - z) * s;
                    }
                }
                wf
            }
            QuantScheme::Int4Block { block_size } => {
                if node.inputs.len() < 3 {
                    return Err(CoremlError::Runtime(
                        "DequantMatMul Int4Block needs a scale input".into(),
                    ));
                }
                let w_bytes = self.quant_bytes(w_id)?;
                let packed = (k * n).div_ceil(2);
                if w_bytes.len() < packed {
                    return Err(CoremlError::Runtime(format!(
                        "Int4 weight bytes {} < packed={}",
                        w_bytes.len(),
                        packed
                    )));
                }
                let bs = block_size.max(1) as usize;
                let n_blocks = k.div_ceil(bs);
                let scales = self.f32_side_tensor(node.inputs[2], n_blocks * n, "scale")?;
                let mut wf = vec![0.0f32; n * k];
                for p in 0..k {
                    let block = p / bs;
                    for j in 0..n {
                        let idx = p * n + j;
                        let byte = w_bytes[idx / 2];
                        let nibble = if idx % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                        let s = scales[block * n + j];
                        wf[j * k + p] = (nibble as f32) * s;
                    }
                }
                wf
            }
            QuantScheme::MlxAffine { bits, group_size } => {
                if node.inputs.len() < 4 {
                    return Err(CoremlError::Runtime(
                        "DequantMatMul MlxAffine needs scale+bias inputs".into(),
                    ));
                }
                let gs = group_size as usize;
                let n_groups = k / gs.max(1);
                let w_bytes = self.quant_bytes(w_id)?;
                let scales = self.f32_side_tensor(node.inputs[2], n * n_groups, "scale")?;
                let biases = self.f32_side_tensor(node.inputs[3], n * n_groups, "bias")?;
                rlx_mlx_io::dequant_affine_f32(
                    w_bytes,
                    &scales,
                    &biases,
                    bits as u32,
                    group_size,
                    n,
                    n_groups,
                )
                .map_err(|e| CoremlError::Runtime(e.to_string()))?
            }
            QuantScheme::MlxMxfp4 { group_size } => {
                if node.inputs.len() < 3 {
                    return Err(CoremlError::Runtime(
                        "DequantMatMul MlxMxfp4 needs scale input".into(),
                    ));
                }
                let gs = group_size as usize;
                let n_groups = k / gs.max(1);
                let w_bytes = self.quant_bytes(w_id)?;
                let scales = self.u8_side_tensor(node.inputs[2], n * n_groups, "scale")?;
                rlx_mlx_io::dequant_mxfp4_f32(w_bytes, &scales, group_size, n, n_groups)
                    .map_err(|e| CoremlError::Runtime(e.to_string()))?
            }
            QuantScheme::MlxMxfp8 { group_size } => {
                if node.inputs.len() < 3 {
                    return Err(CoremlError::Runtime(
                        "DequantMatMul MlxMxfp8 needs scale input".into(),
                    ));
                }
                let gs = group_size as usize;
                let n_groups = k / gs.max(1);
                let w_bytes = self.quant_bytes(w_id)?;
                let scales = self.u8_side_tensor(node.inputs[2], n * n_groups, "scale")?;
                rlx_mlx_io::dequant_mxfp8_f32(w_bytes, &scales, group_size, n, n_groups)
                    .map_err(|e| CoremlError::Runtime(e.to_string()))?
            }
            _ => dequant_scheme(scheme, self.quant_bytes(w_id)?, k * n)?,
        };
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

    /// Load an f32 scale/zp side tensor from a Param or Constant node.
    fn f32_side_tensor(&self, id: NodeId, expect: usize, label: &str) -> Result<Vec<f32>> {
        let floats = match &self.graph.node(id).op {
            Op::Param { name } => {
                if let Some(v) = self.params.get(name) {
                    v.clone()
                } else if let Some((bytes, dt)) = self.typed_params.get(name) {
                    if *dt != DType::F32 {
                        return Err(CoremlError::Runtime(format!(
                            "{label} param '{name}' has dtype {dt:?}, expected F32"
                        )));
                    }
                    bytes_to_f32(bytes, &Shape::new(&[bytes.len() / 4], DType::F32))?
                } else {
                    return Err(CoremlError::Runtime(format!(
                        "missing {label} param '{name}'"
                    )));
                }
            }
            Op::Constant { data } => bytes_to_f32(data, &self.graph.node(id).shape)?,
            other => {
                return Err(CoremlError::Unsupported(format!(
                    "dequant {label} must be Param/Constant, got {other:?}"
                )));
            }
        };
        if floats.len() < expect {
            return Err(CoremlError::Runtime(format!(
                "dequant {label} len {} < expected {expect}",
                floats.len()
            )));
        }
        Ok(floats)
    }

    /// Raw u8 side tensor (MLX mxfp scales).
    fn u8_side_tensor(&self, id: NodeId, expect: usize, label: &str) -> Result<Vec<u8>> {
        let bytes = match &self.graph.node(id).op {
            Op::Param { name } => {
                if let Some((bytes, dt)) = self.typed_params.get(name) {
                    if *dt != DType::U8 && *dt != DType::I8 {
                        return Err(CoremlError::Runtime(format!(
                            "{label} param '{name}' has dtype {dt:?}, expected U8"
                        )));
                    }
                    bytes.clone()
                } else {
                    return Err(CoremlError::Runtime(format!(
                        "missing {label} param '{name}'"
                    )));
                }
            }
            Op::Constant { data } => data.clone(),
            other => {
                return Err(CoremlError::Unsupported(format!(
                    "dequant {label} must be Param/Constant, got {other:?}"
                )));
            }
        };
        if bytes.len() < expect {
            return Err(CoremlError::Runtime(format!(
                "dequant {label} len {} < expected {expect}",
                bytes.len()
            )));
        }
        Ok(bytes)
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

    /// FakeQuantize forward: `clamp(round(x/s), -qmax, qmax) * s`.
    /// Fixed/EMA use the scale input; PerBatch derives `s = max(|x|)/qmax`.
    pub(crate) fn lower_fake_quantize(
        &mut self,
        id: NodeId,
        bits: u8,
        axis: Option<usize>,
        scale_mode: rlx_ir::op::ScaleMode,
        out_name: &str,
    ) -> Result<()> {
        use rlx_ir::op::ScaleMode;
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let f32_shape = shape.clone().with_dtype(DType::F32);
        let rank = shape.rank();
        let x = self.val(node.inputs[0]);
        let q_max = match bits {
            8 => 127.0f32,
            4 => 7.0,
            2 => 1.0,
            n => {
                return Err(CoremlError::Unsupported(format!(
                    "FakeQuantize: unsupported bits {n}"
                )));
            }
        };
        let scale_name = match scale_mode {
            ScaleMode::Fixed | ScaleMode::EMA { .. } => {
                if node.inputs.len() < 2 {
                    return Err(CoremlError::Unsupported(
                        "FakeQuantize Fixed/EMA requires scale input".into(),
                    ));
                }
                self.val(node.inputs[1])
            }
            ScaleMode::PerBatch => {
                let ax = format!("{out_name}_abs");
                self.emit("abs", &ax, &f32_shape, vec![("x", bind_name(&x))])?;
                let reduce_axes: Vec<i32> = match axis {
                    None => (0..rank as i32).collect(),
                    Some(a) => (0..rank as i32).filter(|&i| i != a as i32).collect(),
                };
                let mut red_dims: Vec<usize> =
                    (0..rank).map(|i| shape.dim(i).unwrap_static()).collect();
                for &a in &reduce_axes {
                    red_dims[a as usize] = 1;
                }
                let red_shape = Shape::new(&red_dims, DType::F32);
                let red = format!("{out_name}_amax");
                self.emit(
                    "reduce_max",
                    &red,
                    &red_shape,
                    vec![
                        ("x", bind_name(&ax)),
                        ("axes", bind_value(vec_i32(&reduce_axes))),
                        ("keep_dims", bind_value(scalar_bool(true))),
                    ],
                )?;
                let s = format!("{out_name}_s");
                self.emit(
                    "mul",
                    &s,
                    &red_shape,
                    vec![
                        ("x", bind_name(&red)),
                        ("y", bind_value(scalar_f32(1.0 / q_max))),
                    ],
                )?;
                let s_eps = format!("{out_name}_se");
                self.emit(
                    "maximum",
                    &s_eps,
                    &red_shape,
                    vec![("x", bind_name(&s)), ("y", bind_value(scalar_f32(1e-8)))],
                )?;
                s_eps
            }
        };
        let scaled = format!("{out_name}_xs");
        self.emit(
            "real_div",
            &scaled,
            &f32_shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&scale_name))],
        )?;
        // Half away from zero (Rust `f32::round`), not MIL `round` (ties-to-even).
        // sign(s) * floor(|s| + 0.5)
        let abs_s = format!("{out_name}_ab");
        self.emit("abs", &abs_s, &f32_shape, vec![("x", bind_name(&scaled))])?;
        let shifted = format!("{out_name}_sh");
        self.emit(
            "add",
            &shifted,
            &f32_shape,
            vec![("x", bind_name(&abs_s)), ("y", bind_value(scalar_f32(0.5)))],
        )?;
        let floored = format!("{out_name}_fl");
        self.emit(
            "floor",
            &floored,
            &f32_shape,
            vec![("x", bind_name(&shifted))],
        )?;
        let sgn = format!("{out_name}_sg");
        self.emit("sign", &sgn, &f32_shape, vec![("x", bind_name(&scaled))])?;
        let rounded = format!("{out_name}_rnd");
        self.emit(
            "mul",
            &rounded,
            &f32_shape,
            vec![("x", bind_name(&floored)), ("y", bind_name(&sgn))],
        )?;
        let clipped = format!("{out_name}_clip");
        self.emit(
            "clip",
            &clipped,
            &f32_shape,
            vec![
                ("x", bind_name(&rounded)),
                ("alpha", bind_value(scalar_f32(-q_max))),
                ("beta", bind_value(scalar_f32(q_max))),
            ],
        )?;
        self.emit(
            "mul",
            out_name,
            &f32_shape,
            vec![("x", bind_name(&clipped)), ("y", bind_name(&scale_name))],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }
}
