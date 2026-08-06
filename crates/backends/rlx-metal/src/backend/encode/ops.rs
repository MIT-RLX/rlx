// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Kernel encode helpers extracted from `backend/mod.rs` for navigability.

#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use rlx_ir::{Graph, NodeId, Op};
use std::collections::{HashMap, HashSet};

use crate::device::metal_device;
use crate::kernels::kernels;

/// Largest `m·k·n` across every `Op::MatMul` and `Op::FusedMatMulBiasAct`
/// in the graph. Used by the MPSGraph adaptive-dispatch heuristic to
/// decide whether the per-call overhead is worth eating for this
/// workload.
pub(crate) fn max_matmul_flops_in(graph: &Graph) -> u64 {
    let mut best: u64 = 0;
    for node in graph.nodes() {
        let flops = match &node.op {
            Op::MatMul | Op::FusedMatMulBiasAct { .. } => {
                let out_shape = &node.shape;
                let n_dim = match out_shape.dim(out_shape.rank().saturating_sub(1)) {
                    d if d.is_static() => d.unwrap_static(),
                    _ => continue,
                };
                let out_total: usize = match out_shape.num_elements() {
                    Some(v) => v,
                    None => continue,
                };
                let m_dim = out_total / n_dim.max(1);
                let a_shape = &graph.node(node.inputs[0]).shape;
                let a_total: usize = match a_shape.num_elements() {
                    Some(v) => v,
                    None => continue,
                };
                let k_dim = a_total / m_dim.max(1);
                (m_dim as u64) * (k_dim as u64) * (n_dim as u64)
            }
            // Conv (forward + gradients) is the bulk of a CNN's compute but was
            // invisible here — so a conv-heavy graph looked "tiny" (matmul-only)
            // and the adaptive dispatch skipped its own MPSGraph plan, losing the
            // ~2.6× fusion win. Count conv as out_elems × per-output MACs
            // (C_in/g·kH·kW = weight elems / C_out); the gradients are the same
            // order, so the forward conv alone is enough to cross the threshold.
            Op::Conv { .. } | Op::Conv2dBackwardInput { .. } | Op::Conv2dBackwardWeight { .. } => {
                let out_total: usize = match node.shape.num_elements() {
                    Some(v) => v,
                    None => continue,
                };
                // weight is input[1] for Conv / BackwardInput, input shapes vary
                // for BackwardWeight (output IS the weight) — use the largest
                // input's per-element fan to stay an order-of-magnitude estimate.
                let w_id = *node.inputs.last().unwrap_or(&node.inputs[0]);
                let w_shape = &graph.node(w_id).shape;
                let w_total: usize = match w_shape.num_elements() {
                    Some(v) => v,
                    None => continue,
                };
                let c_out = match w_shape.dim(0) {
                    d if d.is_static() => d.unwrap_static().max(1),
                    _ => 1,
                };
                (out_total as u64) * (w_total as u64 / c_out as u64).max(1)
            }
            _ => continue,
        };
        if flops > best {
            best = flops;
        }
    }
    best
}

pub(crate) fn gguf_dequant_dims_for_param(
    graph: &Graph,
    param_id: NodeId,
) -> Option<(usize, usize, rlx_ir::quant::QuantScheme)> {
    for node in graph.nodes() {
        if let Op::DequantMatMul { scheme } = &node.op
            && node.inputs.get(1) == Some(&param_id)
        {
            let n = node
                .shape
                .dim(node.shape.rank().saturating_sub(1))
                .unwrap_static();
            let out_total = node.shape.num_elements()?;
            let m = out_total / n.max(1);
            let a_total = graph.node(node.inputs[0]).shape.num_elements()?;
            let k = a_total / m.max(1);
            return Some((k, n, *scheme));
        }
    }
    None
}

pub(crate) fn transpose_nk_to_kn_bytes(dequant: &[f32], n: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(k * n * 4);
    for p in 0..k {
        for j in 0..n {
            out.extend_from_slice(&dequant[j * k + p].to_le_bytes());
        }
    }
    out
}

// ── Host-side shape-aware broadcast (Apple Silicon unified memory) ──

/// Compute the in-buffer element count implied by a broadcast-stride
/// vector. A stride of 0 means "size 1" along that output axis (we
/// don't read past element 0 of that axis); a non-zero stride means
/// the axis size matches `out_dims[axis]`.
pub(crate) fn inferred_input_len(strides: &[u32], out_dims: &[u32]) -> usize {
    let mut acc: usize = 1;
    for d in 0..out_dims.len() {
        if strides[d] != 0 {
            acc *= out_dims[d] as usize;
        }
    }
    acc
}

/// Generic host-side binary broadcast. Walks the output index space,
/// decomposes into per-axis coords, and reads via the provided
/// broadcast strides (0 ⇒ replicate along that axis). Correctness-first
/// implementation — a proper MSL kernel would be a follow-on.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn binary_broadcast_host<T>(
    lhs: *const T,
    lhs_len: usize,
    rhs: *const T,
    rhs_len: usize,
    dst: *mut T,
    out_len: usize,
    rank: usize,
    out_dims: &[u32],
    lhs_strides: &[u32],
    rhs_strides: &[u32],
    op: rlx_ir::op::BinaryOp,
) where
    T: Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + PartialOrd,
{
    use rlx_ir::op::BinaryOp;
    let l = unsafe { std::slice::from_raw_parts(lhs, lhs_len) };
    let r = unsafe { std::slice::from_raw_parts(rhs, rhs_len) };
    let o = unsafe { std::slice::from_raw_parts_mut(dst, out_len) };
    for i in 0..out_len {
        // Decompose flat output index into per-axis coords.
        let mut rem = i;
        let mut li: usize = 0;
        let mut ri: usize = 0;
        for ax in (0..rank).rev() {
            let sz = out_dims[ax] as usize;
            let coord = rem % sz;
            rem /= sz;
            li += coord * lhs_strides[ax] as usize;
            ri += coord * rhs_strides[ax] as usize;
        }
        let lv = l[li];
        let rv = r[ri];
        o[i] = match op {
            BinaryOp::Add => lv + rv,
            BinaryOp::Sub => lv - rv,
            BinaryOp::Mul => lv * rv,
            BinaryOp::Div => lv / rv,
            BinaryOp::Max => {
                if lv >= rv {
                    lv
                } else {
                    rv
                }
            }
            BinaryOp::Min => {
                if lv <= rv {
                    lv
                } else {
                    rv
                }
            }
            BinaryOp::Pow
            | BinaryOp::Mod
            | BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor
            | BinaryOp::Shl
            | BinaryOp::Shr
            | BinaryOp::Atan2 => {
                // Not expressible at the generic `T` trait level here; these
                // ops run on the standalone binary kernel path instead.
                panic!("BinaryBroadcast {op:?} not implemented in host path");
            }
        };
    }
}

pub(crate) fn widen_input_bytes_to_f32(data: &[u8], dt: rlx_ir::DType) -> Vec<f32> {
    use rlx_ir::DType;
    match dt {
        DType::F32 => {
            let n = data.len() / 4;
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) }.to_vec()
        }
        DType::F16 => {
            let n = data.len() / 2;
            let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const half::f16, n) };
            s.iter().map(|h| h.to_f32()).collect()
        }
        DType::BF16 => {
            let n = data.len() / 2;
            let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, n) };
            s.iter().map(|h| h.to_f32()).collect()
        }
        // Integer/bool inputs widen to f32 — `widen_integer_activations_to_f32`
        // rewrites their arena slots to F32, so this matches the graph dtype.
        DType::I64 => data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::I32 => data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::U32 => data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::Bool => data.iter().map(|&b| b as f32).collect(),
        other => panic!(
            "rlx-metal widen_input_bytes_to_f32: dtype {other:?} unsupported \
             (use direct byte write for F64/U8/I8 dtypes)"
        ),
    }
}

pub(crate) fn encode_cast(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    len: u32,
    src_dt: crate::thunk::HalfFlag,
    dst_dt: crate::thunk::HalfFlag,
) {
    encode_cast_bufs(enc, k, buffer, src, buffer, dst, len, src_dt, dst_dt);
}

pub(crate) fn encode_cast_bufs(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    src_buf: &metal::Buffer,
    src: usize,
    dst_buf: &metal::Buffer,
    dst: usize,
    len: u32,
    src_dt: crate::thunk::HalfFlag,
    dst_dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match (src_dt, dst_dt) {
        (HalfFlag::F32, HalfFlag::F16) => &k.cast_f32_to_f16,
        (HalfFlag::F16, HalfFlag::F32) => &k.cast_f16_to_f32,
        (a, b) if a == b => {
            let n = match a {
                HalfFlag::F32 => len,
                HalfFlag::F16 => len.div_ceil(2),
            };
            let p = &k.copy_f32;
            enc.set_compute_pipeline_state(p);
            enc.set_buffer(0, Some(src_buf), src as u64);
            enc.set_buffer(1, Some(dst_buf), dst as u64);
            enc.set_bytes(2, 4, &n as *const u32 as *const _);
            let tg_w = p.thread_execution_width().min(n as u64);
            enc.dispatch_threads(
                metal::MTLSize {
                    width: n as u64,
                    height: 1,
                    depth: 1,
                },
                metal::MTLSize {
                    width: tg_w,
                    height: 1,
                    depth: 1,
                },
            );
            return;
        }
        _ => return,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(src_buf), src as u64);
    enc.set_buffer(1, Some(dst_buf), dst as u64);
    enc.set_bytes(2, 4, &len as *const u32 as *const _);
    let tg_w = pipeline.thread_execution_width().min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

// NOTE: the former hand-rolled `cast_host` table lived here. It covered only a
// dozen dtype pairs and `panic!`d on the rest (anything with I8/I16/U8, BF16,
// F64, C64, and several int/bool combos). It was replaced by a direct call to
// rlx-cpu's `exec_cast_generic` (see `backend/encode/mod.rs`, `Thunk::CastHost`),
// which converts ALL 12 dtypes correctly against the unified-memory arena.

pub(crate) fn encode_bias_add(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    data_buf: &metal::Buffer,
    data_off: usize,
    bias_buf: &metal::Buffer,
    bias_off: usize,
    m: u32,
    n: u32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F32 => &k.bias_add,
        HalfFlag::F16 => &k.bias_add_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(data_buf), data_off as u64);
    enc.set_buffer(1, Some(bias_buf), bias_off as u64);
    enc.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &m as *const u32 as *const _,
    );
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &n as *const u32 as *const _,
    );
    let grid = metal::MTLSize {
        width: n as u64,
        height: m as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 16.min(n as u64),
        height: 16.min(m as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_fused_binary_activation(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    lhs: usize,
    rhs: usize,
    dst: usize,
    len: u32,
    op: rlx_ir::op::BinaryOp,
    act: rlx_ir::op::Activation,
) {
    use rlx_ir::op::{Activation, BinaryOp};
    let bin_op: u32 = op.opcode();
    let act_op: u32 = match act {
        Activation::Gelu | Activation::GeluApprox => 0,
        Activation::Silu => 1,
        Activation::Relu => 2,
        Activation::Sigmoid => 3,
        Activation::Tanh => 4,
        _ => 255,
    };
    let use_vec4 = len.is_multiple_of(4) && len >= 4;
    if use_vec4 {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.fused_binary_activation4);
        enc.set_buffer(0, Some(buffer), lhs as u64);
        enc.set_buffer(1, Some(buffer), rhs as u64);
        enc.set_buffer(2, Some(buffer), dst as u64);
        enc.set_bytes(3, 4, &len4 as *const u32 as *const _);
        enc.set_bytes(4, 4, &bin_op as *const u32 as *const _);
        enc.set_bytes(5, 4, &act_op as *const u32 as *const _);
        let tg_w = k
            .fused_binary_activation4
            .thread_execution_width()
            .min(len4 as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: len4 as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
        return;
    }
    enc.set_compute_pipeline_state(&k.fused_binary_activation_f32);
    enc.set_buffer(0, Some(buffer), lhs as u64);
    enc.set_buffer(1, Some(buffer), rhs as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &len as *const u32 as *const _);
    enc.set_bytes(4, 4, &bin_op as *const u32 as *const _);
    enc.set_bytes(5, 4, &act_op as *const u32 as *const _);
    let tg_w = k
        .fused_binary_activation_f32
        .thread_execution_width()
        .min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_fused_ternary_activation(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    lhs: usize,
    rhs0: usize,
    rhs1: usize,
    dst: usize,
    len: u32,
    op0: rlx_ir::op::BinaryOp,
    op1: rlx_ir::op::BinaryOp,
    act: rlx_ir::op::Activation,
) {
    use rlx_ir::op::{Activation, BinaryOp};
    let bin_op0 = op0.opcode();
    let bin_op1 = op1.opcode();
    let act_op: u32 = match act {
        Activation::Gelu | Activation::GeluApprox => 0,
        Activation::Silu => 1,
        Activation::Relu => 2,
        Activation::Sigmoid => 3,
        Activation::Tanh => 4,
        _ => 255,
    };
    let use_vec4 = len.is_multiple_of(4) && len >= 4;
    if use_vec4 {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.fused_ternary_activation4);
        enc.set_buffer(0, Some(buffer), lhs as u64);
        enc.set_buffer(1, Some(buffer), rhs0 as u64);
        enc.set_buffer(2, Some(buffer), rhs1 as u64);
        enc.set_buffer(3, Some(buffer), dst as u64);
        enc.set_bytes(4, 4, &len4 as *const u32 as *const _);
        enc.set_bytes(5, 4, &bin_op0 as *const u32 as *const _);
        enc.set_bytes(6, 4, &bin_op1 as *const u32 as *const _);
        enc.set_bytes(7, 4, &act_op as *const u32 as *const _);
        let tg_w = k
            .fused_ternary_activation4
            .thread_execution_width()
            .min(len4 as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: len4 as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
        return;
    }
    enc.set_compute_pipeline_state(&k.fused_ternary_activation_f32);
    enc.set_buffer(0, Some(buffer), lhs as u64);
    enc.set_buffer(1, Some(buffer), rhs0 as u64);
    enc.set_buffer(2, Some(buffer), rhs1 as u64);
    enc.set_buffer(3, Some(buffer), dst as u64);
    enc.set_bytes(4, 4, &len as *const u32 as *const _);
    enc.set_bytes(5, 4, &bin_op0 as *const u32 as *const _);
    enc.set_bytes(6, 4, &bin_op1 as *const u32 as *const _);
    enc.set_bytes(7, 4, &act_op as *const u32 as *const _);
    let tg_w = k
        .fused_ternary_activation_f32
        .thread_execution_width()
        .min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_gelu_approx_out(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    len: u32,
) {
    debug_assert!(
        len.is_multiple_of(4) && len >= 4,
        "gelu_approx_out expects vec4 len"
    );
    let len4 = len / 4;
    enc.set_compute_pipeline_state(&k.gelu_approx_out4);
    enc.set_buffer(0, Some(buffer), 0);
    let src_u = src as u64;
    let dst_u = dst as u64;
    enc.set_bytes(1, 8, &src_u as *const u64 as *const _);
    enc.set_bytes(2, 8, &dst_u as *const u64 as *const _);
    enc.set_bytes(3, 4, &len4 as *const u32 as *const _);
    let tg_w = k.gelu_approx_out4.thread_execution_width().min(len4 as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len4 as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_activation(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    data_off: usize,
    len: u32,
    act: rlx_ir::op::Activation,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    use rlx_ir::op::Activation;
    if matches!(dt, HalfFlag::F32)
        && len.is_multiple_of(4)
        && len >= 4
        && matches!(act, Activation::Gelu)
    {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.gelu_inplace4);
        enc.set_buffer(0, Some(buffer), 0);
        let off = data_off as u64;
        enc.set_bytes(1, 8, &off as *const u64 as *const _);
        enc.set_bytes(2, 4, &len4 as *const u32 as *const _);
        let tg_w = k.gelu_inplace4.thread_execution_width().min(len4 as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: len4 as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
        return;
    }
    if matches!(dt, HalfFlag::F32)
        && len.is_multiple_of(4)
        && len >= 4
        && matches!(act, Activation::GeluApprox)
    {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.gelu_approx_inplace4);
        enc.set_buffer(0, Some(buffer), 0);
        let off = data_off as u64;
        enc.set_bytes(1, 8, &off as *const u64 as *const _);
        enc.set_bytes(2, 4, &len4 as *const u32 as *const _);
        let tg_w = k
            .gelu_approx_inplace4
            .thread_execution_width()
            .min(len4 as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: len4 as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
        return;
    }
    if matches!(dt, HalfFlag::F32)
        && len.is_multiple_of(4)
        && len >= 4
        && matches!(act, Activation::Silu)
    {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.silu_inplace4);
        enc.set_buffer(0, Some(buffer), 0);
        let off = data_off as u64;
        enc.set_bytes(1, 8, &off as *const u64 as *const _);
        enc.set_bytes(2, 4, &len4 as *const u32 as *const _);
        let tg_w = k.silu_inplace4.thread_execution_width().min(len4 as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: len4 as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
        return;
    }
    // Full f16 activation coverage — math promotes to f32 then stores half.
    let pipeline = match (dt, act) {
        (HalfFlag::F16, Activation::Gelu) => &k.gelu_inplace_h,
        (HalfFlag::F16, Activation::GeluApprox) => &k.gelu_approx_inplace_h,
        (HalfFlag::F16, Activation::Silu) => &k.silu_inplace_h,
        (HalfFlag::F16, Activation::Relu) => &k.relu_inplace_h,
        (HalfFlag::F16, Activation::Sigmoid) => &k.sigmoid_inplace_h,
        (HalfFlag::F16, Activation::Tanh) => &k.tanh_inplace_h,
        (HalfFlag::F16, Activation::Exp) => &k.exp_inplace_h,
        (HalfFlag::F16, Activation::Log) => &k.log_inplace_h,
        (HalfFlag::F16, Activation::Sqrt) => &k.sqrt_inplace_h,
        (HalfFlag::F16, Activation::Rsqrt) => &k.rsqrt_inplace_h,
        (HalfFlag::F16, Activation::Neg) => &k.neg_inplace_h,
        (HalfFlag::F16, Activation::Abs) => &k.abs_inplace_h,
        (HalfFlag::F16, Activation::Sin) => &k.sin_inplace_h,
        (HalfFlag::F16, Activation::Cos) => &k.cos_inplace_h,
        (HalfFlag::F16, Activation::Tan) => &k.tan_inplace_h,
        (HalfFlag::F16, Activation::Atan) => &k.atan_inplace_h,
        (HalfFlag::F16, Activation::Recip) => &k.rec_inplace_h,
        (HalfFlag::F16, Activation::Round) => &k.round_inplace_h,
        (_, Activation::Gelu) => &k.gelu_inplace,
        (_, Activation::GeluApprox) => &k.gelu_approx_inplace,
        (_, Activation::Silu) => &k.silu_inplace,
        (_, Activation::Relu) => &k.relu_inplace,
        (_, Activation::Sigmoid) => &k.sigmoid_inplace,
        (_, Activation::Tanh) => &k.tanh_inplace,
        (_, Activation::Exp) => &k.exp_inplace,
        (_, Activation::Log) => &k.log_inplace,
        (_, Activation::Sqrt) => &k.sqrt_inplace,
        (_, Activation::Rsqrt) => &k.rsqrt_inplace,
        (_, Activation::Neg) => &k.neg_inplace,
        (_, Activation::Abs) => &k.abs_inplace,
        (_, Activation::Sin) => &k.sin_inplace,
        (_, Activation::Cos) => &k.cos_inplace,
        (_, Activation::Tan) => &k.tan_inplace,
        (_, Activation::Atan) => &k.atan_inplace,
        (_, Activation::Recip) => &k.rec_inplace,
        (_, Activation::Round) => &k.round_inplace,
        // Macro-generated scalar activations: one (f32, f16) pipeline pair per
        // activation, keyed by the activation (see `scalar_activation_kernels!`).
        (
            _,
            Activation::Floor
            | Activation::Ceil
            | Activation::Sign
            | Activation::Softplus
            | Activation::Elu
            | Activation::Erf
            | Activation::HardSwish
            | Activation::HardSigmoid
            | Activation::Mish
            | Activation::Softsign
            | Activation::LogSigmoid,
        ) => {
            let (f32p, f16p) = &k.scalar_acts[&act];
            if matches!(dt, HalfFlag::F16) {
                f16p
            } else {
                f32p
            }
        }
    };
    enc.set_compute_pipeline_state(pipeline);
    if matches!(dt, HalfFlag::F32)
        && matches!(
            act,
            Activation::Gelu | Activation::GeluApprox | Activation::Silu
        )
    {
        // Task #50: arena base + byte offset for activations past 4 GB.
        enc.set_buffer(0, Some(buffer), 0);
        let off = data_off as u64;
        enc.set_bytes(1, 8, &off as *const u64 as *const _);
        enc.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &len as *const u32 as *const _,
        );
    } else {
        enc.set_buffer(0, Some(buffer), data_off as u64);
        enc.set_bytes(
            1,
            std::mem::size_of::<u32>() as u64,
            &len as *const u32 as *const _,
        );
    }
    let tg_size = pipeline.thread_execution_width().min(len as u64);
    let grid = metal::MTLSize {
        width: len as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: tg_size,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_activation_out(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src_off: usize,
    dst_off: usize,
    len: u32,
    act: rlx_ir::op::Activation,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    use rlx_ir::op::Activation;
    if matches!(dt, HalfFlag::F32)
        && len.is_multiple_of(4)
        && len >= 4
        && matches!(act, Activation::Silu)
    {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.silu_out4);
        enc.set_buffer(0, Some(buffer), 0);
        let src = src_off as u64;
        let dst = dst_off as u64;
        enc.set_bytes(1, 8, &src as *const u64 as *const _);
        enc.set_bytes(2, 8, &dst as *const u64 as *const _);
        enc.set_bytes(3, 4, &len4 as *const u32 as *const _);
        let tg_w = k.silu_out4.thread_execution_width().min(len4 as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: len4 as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
        return;
    }
    if matches!(act, Activation::GeluApprox) && matches!(dt, HalfFlag::F32) {
        encode_gelu_approx_out(enc, k, buffer, src_off, dst_off, len);
        return;
    }
    // Fallback: copy then in-place (still one schedule node; two dispatches).
    encode_copy(enc, k, buffer, src_off, dst_off, len, dt);
    encode_activation(enc, k, buffer, dst_off, len, act, dt);
}

pub(crate) fn encode_layer_norm(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    g: usize,
    b: usize,
    dst: usize,
    rows: u32,
    h: u32,
    eps: f32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F32 => &k.layer_norm,
        HalfFlag::F16 => &k.layer_norm_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), g as u64);
    enc.set_buffer(2, Some(buffer), b as u64);
    enc.set_buffer(3, Some(buffer), dst as u64);
    enc.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &h as *const u32 as *const _,
    );
    enc.set_bytes(
        5,
        std::mem::size_of::<f32>() as u64,
        &eps as *const f32 as *const _,
    );
    // One threadgroup per row; reduction requires power-of-2 threadgroup size.
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= h as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    let grid = metal::MTLSize {
        width: rows as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg);
}

/// Row-major dense strides for `out_dims`.
pub(crate) fn dense_row_major_strides(out_dims: &[u32], rank: usize) -> Vec<u32> {
    let mut dense = vec![1u32; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        dense[i] = dense[i + 1].saturating_mul(out_dims[i + 1].max(1));
    }
    dense
}

pub(crate) fn broadcast_strides_u32(in_dims: &[u32], out_dims: &[u32]) -> Vec<u32> {
    let r_out = out_dims.len();
    let r_in = in_dims.len();
    let pad = r_out.saturating_sub(r_in);
    let mut strides = vec![0u32; r_out];
    let mut acc: u32 = 1;
    for d in (0..r_out).rev() {
        let in_size = if d < pad { 1 } else { in_dims[d - pad].max(1) };
        if in_size == 1 {
            strides[d] = 0;
        } else {
            strides[d] = acc;
            acc = acc.saturating_mul(in_size);
        }
    }
    strides
}

pub(crate) fn is_scalar_broadcast(strides: &[u32], rank: usize) -> bool {
    rank == 0 || (strides.len() >= rank && strides[..rank].iter().all(|&s| s == 0))
}

pub(crate) fn is_row_vector_broadcast(strides: &[u32], rank: usize, out_dims: &[u32]) -> bool {
    if rank < 2 || strides.len() < rank || strides[rank - 1] != 0 {
        return false;
    }
    let mut in_dims = Vec::with_capacity(rank);
    for i in 0..rank - 1 {
        in_dims.push(out_dims[i]);
    }
    in_dims.push(1);
    let expected = broadcast_strides_u32(&in_dims, out_dims);
    strides[..rank] == expected[..rank]
}

/// `Some(rhs_is_scalar)` when one side is dense and the other is a scalar broadcast.
pub(crate) fn detect_scalar_broadcast(
    rank: u32,
    out_dims: &[u32],
    lhs_strides: &[u32],
    rhs_strides: &[u32],
) -> Option<bool> {
    let rank = rank as usize;
    if out_dims.len() < rank {
        return None;
    }
    let dense = dense_row_major_strides(out_dims, rank);
    if lhs_strides.len() >= rank
        && rhs_strides.len() >= rank
        && lhs_strides[..rank] == dense[..]
        && is_scalar_broadcast(rhs_strides, rank)
    {
        return Some(true);
    }
    if rhs_strides.len() >= rank
        && lhs_strides.len() >= rank
        && rhs_strides[..rank] == dense[..]
        && is_scalar_broadcast(lhs_strides, rank)
    {
        return Some(false);
    }
    None
}

/// `Some((rows, cols, rhs_is_broadcast))` when one operand is dense row-major and the other
/// is a last-axis vector broadcast over all leading dimensions (`stride 0` on outer axes).
pub(crate) fn detect_last_axis_col_broadcast(
    rank: u32,
    out_dims: &[u32],
    lhs_strides: &[u32],
    rhs_strides: &[u32],
) -> Option<(u32, u32, bool)> {
    let rank = rank as usize;
    if rank < 2 || out_dims.len() < rank {
        return None;
    }
    let cols = out_dims[rank - 1];
    if cols == 0 {
        return None;
    }
    let mut rows_u64 = 1u64;
    for &d in &out_dims[..rank - 1] {
        rows_u64 = rows_u64.saturating_mul(d.max(1) as u64);
    }
    if rows_u64 == 0 || rows_u64 > u32::MAX as u64 {
        return None;
    }
    let rows = rows_u64 as u32;

    let dense = dense_row_major_strides(out_dims, rank);

    let rhs_is_vec = |strides: &[u32]| -> bool {
        strides.len() >= rank
            && strides[rank - 1] == 1
            && strides[..rank - 1].iter().all(|&s| s == 0)
    };

    if lhs_strides.len() >= rank
        && rhs_strides.len() >= rank
        && lhs_strides[..rank] == dense[..]
        && rhs_is_vec(rhs_strides)
    {
        return Some((rows, cols, true));
    }
    if rhs_strides.len() >= rank
        && lhs_strides.len() >= rank
        && rhs_strides[..rank] == dense[..]
        && rhs_is_vec(lhs_strides)
    {
        return Some((rows, cols, false));
    }
    None
}

/// `Some((rows, cols, rhs_is_broadcast))` for `[…, cols] op […, 1]` row-vector broadcast.
pub(crate) fn detect_last_axis_row_broadcast(
    rank: u32,
    out_dims: &[u32],
    lhs_strides: &[u32],
    rhs_strides: &[u32],
) -> Option<(u32, u32, bool)> {
    let rank = rank as usize;
    if rank < 2 || out_dims.len() < rank {
        return None;
    }
    let cols = out_dims[rank - 1];
    if cols == 0 {
        return None;
    }
    let mut rows_u64 = 1u64;
    for &d in &out_dims[..rank - 1] {
        rows_u64 = rows_u64.saturating_mul(d.max(1) as u64);
    }
    if rows_u64 == 0 || rows_u64 > u32::MAX as u64 {
        return None;
    }
    let rows = rows_u64 as u32;
    let dense = dense_row_major_strides(out_dims, rank);

    if lhs_strides.len() >= rank
        && rhs_strides.len() >= rank
        && lhs_strides[..rank] == dense[..]
        && is_row_vector_broadcast(rhs_strides, rank, out_dims)
    {
        return Some((rows, cols, true));
    }
    if rhs_strides.len() >= rank
        && lhs_strides.len() >= rank
        && rhs_strides[..rank] == dense[..]
        && is_row_vector_broadcast(lhs_strides, rank, out_dims)
    {
        return Some((rows, cols, false));
    }
    None
}

/// Exactly one broadcast axis (e.g. `[B, T, H] op [B, 1, H]`).
pub(crate) fn detect_single_axis_broadcast(
    rank: u32,
    out_dims: &[u32],
    lhs_strides: &[u32],
    rhs_strides: &[u32],
) -> Option<(u32, u32, u32, bool)> {
    let rank = rank as usize;
    if rank < 2 || out_dims.len() < rank {
        return None;
    }
    let dense = dense_row_major_strides(out_dims, rank);

    let try_side = |strides: &[u32], other: &[u32]| -> Option<(u32, u32, u32)> {
        if strides.len() < rank || other.len() < rank || other[..rank] != dense[..rank] {
            return None;
        }
        let zero_axes: Vec<usize> = (0..rank).filter(|&i| strides[i] == 0).collect();
        if zero_axes.len() != 1 {
            return None;
        }
        let ba = zero_axes[0];
        let mut in_dims = out_dims[..rank].to_vec();
        in_dims[ba] = 1;
        let expected = broadcast_strides_u32(&in_dims, out_dims);
        if strides[..rank] != expected[..rank] {
            return None;
        }
        let mut pre_u64 = 1u64;
        for &d in &out_dims[..ba] {
            pre_u64 = pre_u64.saturating_mul(d.max(1) as u64);
        }
        let mid = out_dims[ba];
        let mut post_u64 = 1u64;
        for &d in &out_dims[ba + 1..] {
            post_u64 = post_u64.saturating_mul(d.max(1) as u64);
        }
        let rows_u64 = pre_u64.saturating_mul(mid.max(1) as u64);
        if rows_u64 == 0
            || post_u64 == 0
            || rows_u64 > u32::MAX as u64
            || post_u64 > u32::MAX as u64
        {
            return None;
        }
        Some((rows_u64 as u32, post_u64 as u32, mid))
    };

    if let Some((rows, cols, mid)) = try_side(rhs_strides, lhs_strides) {
        return Some((rows, cols, mid, true));
    }
    if let Some((rows, cols, mid)) = try_side(lhs_strides, rhs_strides) {
        return Some((rows, cols, mid, false));
    }
    None
}

pub(crate) fn encode_binary_broadcast_1ax(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    lhs: usize,
    rhs: usize,
    dst: usize,
    rows: u32,
    cols: u32,
    mid: u32,
    op: u32,
    rhs_is_broadcast: bool,
) {
    let (a, b) = if rhs_is_broadcast {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };
    let use_vec4 = cols.is_multiple_of(4) && cols >= 4;
    if use_vec4 {
        let cols4 = cols / 4;
        enc.set_compute_pipeline_state(&k.binary_broadcast_1ax4);
        enc.set_buffer(0, Some(buffer), a as u64);
        enc.set_buffer(1, Some(buffer), b as u64);
        enc.set_buffer(2, Some(buffer), dst as u64);
        enc.set_bytes(3, 4, &rows as *const u32 as *const _);
        enc.set_bytes(4, 4, &cols4 as *const u32 as *const _);
        enc.set_bytes(5, 4, &mid as *const u32 as *const _);
        enc.set_bytes(6, 4, &op as *const u32 as *const _);
        let grid = metal::MTLSize {
            width: cols4 as u64,
            height: rows as u64,
            depth: 1,
        };
        let tg = metal::MTLSize {
            width: 64.min(cols4 as u64),
            height: 4.min(rows as u64),
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        return;
    }
    enc.set_compute_pipeline_state(&k.binary_broadcast_1ax_f32);
    enc.set_buffer(0, Some(buffer), a as u64);
    enc.set_buffer(1, Some(buffer), b as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &rows as *const u32 as *const _);
    enc.set_bytes(4, 4, &cols as *const u32 as *const _);
    enc.set_bytes(5, 4, &mid as *const u32 as *const _);
    enc.set_bytes(6, 4, &op as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: cols as u64,
        height: rows as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 32.min(cols as u64),
        height: 8.min(rows as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_binary_broadcast_rhs_scalar(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    lhs: usize,
    rhs: usize,
    dst: usize,
    len: u32,
    op: u32,
    rhs_is_scalar: bool,
) {
    let (a, b) = if rhs_is_scalar {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };
    let use_vec4 = len.is_multiple_of(4) && len >= 4;
    if use_vec4 {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.binary_broadcast_rhs_scalar4);
        enc.set_buffer(0, Some(buffer), 0);
        let a_u = a as u64;
        let b_u = b as u64;
        let d_u = dst as u64;
        enc.set_bytes(1, 8, &a_u as *const u64 as *const _);
        enc.set_bytes(2, 8, &b_u as *const u64 as *const _);
        enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
        enc.set_bytes(4, 4, &len4 as *const u32 as *const _);
        enc.set_bytes(5, 4, &op as *const u32 as *const _);
        let tg_w = k
            .binary_broadcast_rhs_scalar4
            .thread_execution_width()
            .min(len4 as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: len4 as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
        return;
    }
    enc.set_compute_pipeline_state(&k.binary_broadcast_rhs_scalar_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let a_u = a as u64;
    let b_u = b as u64;
    let d_u = dst as u64;
    enc.set_bytes(1, 8, &a_u as *const u64 as *const _);
    enc.set_bytes(2, 8, &b_u as *const u64 as *const _);
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    enc.set_bytes(4, 4, &len as *const u32 as *const _);
    enc.set_bytes(5, 4, &op as *const u32 as *const _);
    let tg_w = k
        .binary_broadcast_rhs_scalar_f32
        .thread_execution_width()
        .min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_binary_broadcast_rhs_row(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    lhs: usize,
    rhs: usize,
    dst: usize,
    rows: u32,
    cols: u32,
    op: u32,
    rhs_is_broadcast: bool,
) {
    let (a, b) = if rhs_is_broadcast {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };
    let use_vec4 = cols.is_multiple_of(4) && cols >= 4;
    if use_vec4 {
        let cols4 = cols / 4;
        enc.set_compute_pipeline_state(&k.binary_broadcast_rhs_row4);
        enc.set_buffer(0, Some(buffer), a as u64);
        enc.set_buffer(1, Some(buffer), b as u64);
        enc.set_buffer(2, Some(buffer), dst as u64);
        enc.set_bytes(3, 4, &rows as *const u32 as *const _);
        enc.set_bytes(4, 4, &cols4 as *const u32 as *const _);
        enc.set_bytes(5, 4, &op as *const u32 as *const _);
        let grid = metal::MTLSize {
            width: cols4 as u64,
            height: rows as u64,
            depth: 1,
        };
        let tg = metal::MTLSize {
            width: 64.min(cols4 as u64),
            height: 4.min(rows as u64),
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        return;
    }
    enc.set_compute_pipeline_state(&k.binary_broadcast_rhs_row_f32);
    enc.set_buffer(0, Some(buffer), a as u64);
    enc.set_buffer(1, Some(buffer), b as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &rows as *const u32 as *const _);
    enc.set_bytes(4, 4, &cols as *const u32 as *const _);
    enc.set_bytes(5, 4, &op as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: cols as u64,
        height: rows as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 32.min(cols as u64),
        height: 8.min(rows as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_binary_broadcast_rhs_col(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    lhs: usize,
    rhs: usize,
    dst: usize,
    rows: u32,
    cols: u32,
    op: u32,
    rhs_is_broadcast: bool,
) {
    let (a, b) = if rhs_is_broadcast {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };
    let use_vec4 = cols.is_multiple_of(4) && cols >= 4;
    if use_vec4 {
        let cols4 = cols / 4;
        enc.set_compute_pipeline_state(&k.binary_broadcast_rhs_col4);
        enc.set_buffer(0, Some(buffer), a as u64);
        enc.set_buffer(1, Some(buffer), b as u64);
        enc.set_buffer(2, Some(buffer), dst as u64);
        enc.set_bytes(3, 4, &rows as *const u32 as *const _);
        enc.set_bytes(4, 4, &cols4 as *const u32 as *const _);
        enc.set_bytes(5, 4, &op as *const u32 as *const _);
        let grid = metal::MTLSize {
            width: cols4 as u64,
            height: rows as u64,
            depth: 1,
        };
        let tg = metal::MTLSize {
            width: 64.min(cols4 as u64),
            height: 4.min(rows as u64),
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        return;
    }
    enc.set_compute_pipeline_state(&k.binary_broadcast_rhs_col_f32);
    enc.set_buffer(0, Some(buffer), a as u64);
    enc.set_buffer(1, Some(buffer), b as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &rows as *const u32 as *const _);
    enc.set_bytes(4, 4, &cols as *const u32 as *const _);
    enc.set_bytes(5, 4, &op as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: cols as u64,
        height: rows as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 32.min(cols as u64),
        height: 8.min(rows as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_binary_broadcast_rank2(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    lhs: usize,
    rhs: usize,
    dst: usize,
    len: u32,
    dim0: u32,
    dim1: u32,
    lhs_stride0: u32,
    lhs_stride1: u32,
    rhs_stride0: u32,
    rhs_stride1: u32,
    op: u32,
) {
    let use_vec4 = len.is_multiple_of(4)
        && dim1.is_multiple_of(4)
        && len >= 4
        && lhs_stride1 == 1
        && rhs_stride1 == 1;
    if use_vec4 {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.binary_broadcast_rank24);
        enc.set_buffer(0, Some(buffer), lhs as u64);
        enc.set_buffer(1, Some(buffer), rhs as u64);
        enc.set_buffer(2, Some(buffer), dst as u64);
        enc.set_bytes(3, 4, &len4 as *const u32 as *const _);
        enc.set_bytes(4, 4, &dim0 as *const u32 as *const _);
        enc.set_bytes(5, 4, &dim1 as *const u32 as *const _);
        enc.set_bytes(6, 4, &lhs_stride0 as *const u32 as *const _);
        enc.set_bytes(7, 4, &lhs_stride1 as *const u32 as *const _);
        enc.set_bytes(8, 4, &rhs_stride0 as *const u32 as *const _);
        enc.set_bytes(9, 4, &rhs_stride1 as *const u32 as *const _);
        enc.set_bytes(10, 4, &op as *const u32 as *const _);
        let tg_w = k
            .binary_broadcast_rank24
            .thread_execution_width()
            .min(len4 as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: len4 as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
        return;
    }
    enc.set_compute_pipeline_state(&k.binary_broadcast_rank2_f32);
    enc.set_buffer(0, Some(buffer), lhs as u64);
    enc.set_buffer(1, Some(buffer), rhs as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &len as *const u32 as *const _);
    enc.set_bytes(4, 4, &dim0 as *const u32 as *const _);
    enc.set_bytes(5, 4, &dim1 as *const u32 as *const _);
    enc.set_bytes(6, 4, &lhs_stride0 as *const u32 as *const _);
    enc.set_bytes(7, 4, &lhs_stride1 as *const u32 as *const _);
    enc.set_bytes(8, 4, &rhs_stride0 as *const u32 as *const _);
    enc.set_bytes(9, 4, &rhs_stride1 as *const u32 as *const _);
    enc.set_bytes(10, 4, &op as *const u32 as *const _);
    let tg_w = k
        .binary_broadcast_rank2_f32
        .thread_execution_width()
        .min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_binary(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    lhs: usize,
    rhs: usize,
    dst: usize,
    len: u32,
    op: rlx_ir::op::BinaryOp,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    use rlx_ir::op::BinaryOp;
    // Only Add/Mul/Sub/Div have vec4 kernels (`elem_{add,mul,sub,div}4`). The
    // other region-fusable ops (Max/Min/Pow) and Mod/bitwise fall back to their
    // scalar kernels, so they must NOT take the vec4 path — otherwise
    // `dispatch_len = len/4` would run the scalar kernel over only the first
    // quarter of the elements and leave the rest at zero (silent, and only when
    // `len % 4 == 0`).
    let use_vec4 = matches!(dt, HalfFlag::F32)
        && len.is_multiple_of(4)
        && len >= 4
        && matches!(
            op,
            BinaryOp::Add | BinaryOp::Mul | BinaryOp::Sub | BinaryOp::Div
        );
    // Full f16 binary coverage (Add/Mul/Sub/Div/Max/Min/Pow).
    let pipeline = match (dt, op, use_vec4) {
        (HalfFlag::F16, BinaryOp::Add, _) => &k.elem_add_h,
        (HalfFlag::F16, BinaryOp::Mul, _) => &k.elem_mul_h,
        (HalfFlag::F16, BinaryOp::Sub, _) => &k.elem_sub_h,
        (HalfFlag::F16, BinaryOp::Div, _) => &k.elem_div_h,
        (HalfFlag::F16, BinaryOp::Max, _) => &k.elem_max_h,
        (HalfFlag::F16, BinaryOp::Min, _) => &k.elem_min_h,
        (HalfFlag::F16, BinaryOp::Pow, _) => &k.elem_pow_h,
        (_, BinaryOp::Add, true) => &k.elem_add4,
        (_, BinaryOp::Mul, true) => &k.elem_mul4,
        (_, BinaryOp::Sub, true) => &k.elem_sub4,
        (_, BinaryOp::Div, true) => &k.elem_div4,
        (_, BinaryOp::Add, false) => &k.elem_add,
        (_, BinaryOp::Mul, false) => &k.elem_mul,
        (_, BinaryOp::Sub, false) => &k.elem_sub,
        (_, BinaryOp::Div, false) => &k.elem_div,
        (_, BinaryOp::Max, _) => &k.elem_max,
        (_, BinaryOp::Min, _) => &k.elem_min,
        (_, BinaryOp::Pow, _) => &k.elem_pow,
        // Mod/bitwise: single opcode-driven `elem_binop` kernel (fused_bin).
        (
            HalfFlag::F16,
            BinaryOp::Mod
            | BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor
            | BinaryOp::Shl
            | BinaryOp::Shr
            | BinaryOp::Atan2,
            _,
        ) => &k.elem_binop_h,
        (
            _,
            BinaryOp::Mod
            | BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor
            | BinaryOp::Shl
            | BinaryOp::Shr
            | BinaryOp::Atan2,
            _,
        ) => &k.elem_binop,
    };
    let dispatch_len = if use_vec4 { len / 4 } else { len };
    enc.set_compute_pipeline_state(pipeline);
    let use_arena_off = matches!(dt, HalfFlag::F32)
        && matches!(
            op,
            BinaryOp::Add | BinaryOp::Mul | BinaryOp::Sub | BinaryOp::Div
        );
    if use_arena_off {
        // Task #50: arena base + byte offsets for tensors past 4 GB.
        enc.set_buffer(0, Some(buffer), 0);
        let lhs_u64 = lhs as u64;
        let rhs_u64 = rhs as u64;
        let dst_u64 = dst as u64;
        enc.set_bytes(1, 8, &lhs_u64 as *const u64 as *const _);
        enc.set_bytes(2, 8, &rhs_u64 as *const u64 as *const _);
        enc.set_bytes(3, 8, &dst_u64 as *const u64 as *const _);
        enc.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &dispatch_len as *const u32 as *const _,
        );
    } else {
        enc.set_buffer(0, Some(buffer), lhs as u64);
        enc.set_buffer(1, Some(buffer), rhs as u64);
        enc.set_buffer(2, Some(buffer), dst as u64);
        enc.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &dispatch_len as *const u32 as *const _,
        );
        // `elem_binop` also needs the opcode (Mod=7 … Shr=12).
        if !op.region_fusable() {
            let op_id: u32 = match op {
                BinaryOp::Mod => 7,
                BinaryOp::BitAnd => 8,
                BinaryOp::BitOr => 9,
                BinaryOp::BitXor => 10,
                BinaryOp::Shl => 11,
                BinaryOp::Shr => 12,
                BinaryOp::Atan2 => 13,
                _ => unreachable!(),
            };
            enc.set_bytes(4, 4, &op_id as *const u32 as *const _);
        }
    }
    let tg_w = pipeline.thread_execution_width().min(dispatch_len as u64);
    let grid = metal::MTLSize {
        width: dispatch_len as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_copy(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    len: u32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    if matches!(dt, HalfFlag::F32) && len.is_multiple_of(4) && len >= 4 {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.copy4);
        enc.set_buffer(0, Some(buffer), 0);
        let src_u64 = src as u64;
        let dst_u64 = dst as u64;
        enc.set_bytes(1, 8, &src_u64 as *const u64 as *const _);
        enc.set_bytes(2, 8, &dst_u64 as *const u64 as *const _);
        enc.set_bytes(3, 4, &len4 as *const u32 as *const _);
        let tg_w = k.copy4.thread_execution_width().min(len4 as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: len4 as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
        return;
    }
    // copy_f32 moves 4 bytes per dispatch slot. For f16, two f16 values
    // pack into one f32 slot, so we halve the dispatch count and reuse
    // the same kernel. Assumes even len (Nomic shapes always are).
    let dispatch_len = match dt {
        HalfFlag::F32 => len,
        HalfFlag::F16 => len.div_ceil(2),
    };
    enc.set_compute_pipeline_state(&k.copy_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let src_u64 = src as u64;
    let dst_u64 = dst as u64;
    enc.set_bytes(1, 8, &src_u64 as *const u64 as *const _);
    enc.set_bytes(2, 8, &dst_u64 as *const u64 as *const _);
    enc.set_bytes(3, 4, &dispatch_len as *const u32 as *const _);
    let tg_w = k.copy_f32.thread_execution_width().min(dispatch_len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: dispatch_len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_gather(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    table_buf: &metal::Buffer,
    table: usize,
    idx_buf: &metal::Buffer,
    idx: usize,
    dst_buf: &metal::Buffer,
    dst: usize,
    num_idx: u32,
    trailing: u32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F32 => &k.gather_axis0,
        HalfFlag::F16 => &k.gather_axis0_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(table_buf), table as u64);
    enc.set_buffer(1, Some(idx_buf), idx as u64);
    enc.set_buffer(2, Some(dst_buf), dst as u64);
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &num_idx as *const u32 as *const _,
    );
    enc.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &trailing as *const u32 as *const _,
    );
    let grid = metal::MTLSize {
        width: trailing as u64,
        height: num_idx as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 16.min(trailing as u64),
        height: 16.min(num_idx as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct NarrowSegGpu {
    // u64 for the same reason as ConcatSegGpu — task #50: ≥4 GB Q4 models
    // have activation byte offsets that exceed u32.
    pub(crate) dst: u64,
    pub(crate) start: u32,
    pub(crate) len: u32,
}

pub(crate) struct PendingNarrowBatch {
    pub(crate) src: usize,
    pub(crate) outer: u32,
    pub(crate) src_axis: u32,
    pub(crate) dt: crate::thunk::HalfFlag,
    pub(crate) segments: Vec<(usize, u32, u32)>,
}

pub(crate) const NARROW_BATCH_MAX: usize = 64;

pub(crate) fn metal_narrow_batch_enabled() -> bool {
    !rlx_ir::env::flag("RLX_METAL_NARROW_BATCH")
}

pub(crate) fn narrow_segments_partition(src_axis: u32, segments: &[(u32, u32)]) -> bool {
    let mut sorted = segments.to_vec();
    sorted.sort_by_key(|(s, _)| *s);
    let mut end = 0u32;
    for (start, len) in sorted {
        if start != end {
            return false;
        }
        end = end.saturating_add(len);
    }
    end == src_axis
}

pub(crate) fn flush_pending_narrow_batch(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    batch: &mut Option<PendingNarrowBatch>,
) {
    let Some(b) = batch.take() else {
        return;
    };
    if b.segments.is_empty() {
        return;
    }
    if b.segments.len() == 1 {
        let (dst, start, len) = b.segments[0];
        encode_narrow(
            enc, k, buffer, b.src, dst, b.outer, b.src_axis, start, len, b.dt,
        );
        return;
    }
    let meta: Vec<(u32, u32)> = b
        .segments
        .iter()
        .map(|(_, start, len)| (*start, *len))
        .collect();
    if narrow_segments_partition(b.src_axis, &meta) {
        encode_split_lastax(enc, k, buffer, &b);
    } else {
        for (dst, start, len) in b.segments {
            encode_narrow(
                enc, k, buffer, b.src, dst, b.outer, b.src_axis, start, len, b.dt,
            );
        }
    }
}

pub(crate) fn try_queue_narrow_batch(
    batch: &mut Option<PendingNarrowBatch>,
    src: usize,
    dst: usize,
    outer: u32,
    src_axis: u32,
    start: u32,
    len: u32,
    dt: crate::thunk::HalfFlag,
) -> bool {
    if !metal_narrow_batch_enabled() || outer == 0 {
        return false;
    }
    if !matches!(dt, crate::thunk::HalfFlag::F32) {
        return false;
    }
    match batch {
        None => {
            *batch = Some(PendingNarrowBatch {
                src,
                outer,
                src_axis,
                dt,
                segments: vec![(dst, start, len)],
            });
            true
        }
        Some(b) if b.src == src && b.outer == outer && b.src_axis == src_axis && b.dt == dt => {
            if b.segments.len() >= NARROW_BATCH_MAX {
                return false;
            }
            let mut meta: Vec<(u32, u32)> = b.segments.iter().map(|(_, s, l)| (*s, *l)).collect();
            meta.push((start, len));
            if !narrow_segments_partition(b.src_axis, &meta) {
                return false;
            }
            b.segments.push((dst, start, len));
            true
        }
        Some(_) => false,
    }
}

pub(crate) fn encode_split_lastax(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    batch: &PendingNarrowBatch,
) {
    use crate::thunk::HalfFlag;
    debug_assert!(batch.segments.len() >= 2);
    let segs: Vec<NarrowSegGpu> = batch
        .segments
        .iter()
        .map(|(dst, start, len)| NarrowSegGpu {
            dst: *dst as u64,
            start: *start,
            len: *len,
        })
        .collect();
    let num_seg = segs.len() as u32;
    let max_len = segs.iter().map(|s| s.len).max().unwrap_or(0);
    let use_vec4 = batch.src_axis.is_multiple_of(4)
        && segs
            .iter()
            .all(|s| (s.start % 4) == 0 && (s.len % 4) == 0 && s.len >= 4);
    if use_vec4 {
        let src_axis4 = batch.src_axis / 4;
        let max_len4 = max_len / 4;
        enc.set_compute_pipeline_state(&k.split_lastax4);
        // Bind to arena base + pass src byte offset (task #50).
        enc.set_buffer(0, Some(buffer), 0);
        enc.set_buffer(1, Some(buffer), 0);
        enc.set_bytes(2, 4, &batch.outer as *const u32 as *const _);
        enc.set_bytes(3, 4, &src_axis4 as *const u32 as *const _);
        enc.set_bytes(4, 4, &num_seg as *const u32 as *const _);
        enc.set_bytes(
            5,
            (segs.len() * std::mem::size_of::<NarrowSegGpu>()) as u64,
            segs.as_ptr() as *const _,
        );
        let src_u64 = batch.src as u64;
        enc.set_bytes(6, 8, &src_u64 as *const u64 as *const _);
        let grid = metal::MTLSize {
            width: max_len4 as u64,
            height: batch.outer as u64,
            depth: num_seg as u64,
        };
        // Task #50: cap total threads per threadgroup at 1024.
        let tg_depth = (1024u64 / (64 * 4)).min(num_seg as u64).max(1);
        let tg = metal::MTLSize {
            width: 64.min(max_len4 as u64),
            height: 4.min(batch.outer as u64),
            depth: tg_depth,
        };
        enc.dispatch_threads(grid, tg);
    } else {
        enc.set_compute_pipeline_state(&k.split_lastax);
        // Bind to arena base + pass src byte offset (task #50).
        enc.set_buffer(0, Some(buffer), 0);
        enc.set_buffer(1, Some(buffer), 0);
        enc.set_bytes(2, 4, &batch.outer as *const u32 as *const _);
        enc.set_bytes(3, 4, &batch.src_axis as *const u32 as *const _);
        enc.set_bytes(4, 4, &num_seg as *const u32 as *const _);
        enc.set_bytes(
            5,
            (segs.len() * std::mem::size_of::<NarrowSegGpu>()) as u64,
            segs.as_ptr() as *const _,
        );
        let src_u64 = batch.src as u64;
        enc.set_bytes(6, 8, &src_u64 as *const u64 as *const _);
        let grid = metal::MTLSize {
            width: max_len as u64,
            height: batch.outer as u64,
            depth: num_seg as u64,
        };
        let tg_depth = (1024u64 / (32 * 8)).min(num_seg as u64).max(1);
        let tg = metal::MTLSize {
            width: 32.min(max_len as u64),
            height: 8.min(batch.outer as u64),
            depth: tg_depth,
        };
        enc.dispatch_threads(grid, tg);
    }
    let _ = HalfFlag::F32;
}

pub(crate) fn encode_narrow(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    outer: u32,
    src_axis: u32,
    start: u32,
    len: u32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    match dt {
        HalfFlag::F32
            if start.is_multiple_of(4)
                && src_axis.is_multiple_of(4)
                && len.is_multiple_of(4)
                && len >= 4 =>
        {
            let src_axis4 = src_axis / 4;
            let start4 = start / 4;
            let len4 = len / 4;
            enc.set_compute_pipeline_state(&k.narrow_lastax4);
            // Task #50: bind to arena base + pass byte offsets as ulong.
            enc.set_buffer(0, Some(buffer), 0);
            enc.set_buffer(1, Some(buffer), 0);
            enc.set_bytes(2, 4, &outer as *const u32 as *const _);
            enc.set_bytes(3, 4, &src_axis4 as *const u32 as *const _);
            enc.set_bytes(4, 4, &start4 as *const u32 as *const _);
            enc.set_bytes(5, 4, &len4 as *const u32 as *const _);
            let src_u64 = src as u64;
            let dst_u64 = dst as u64;
            enc.set_bytes(6, 8, &src_u64 as *const u64 as *const _);
            enc.set_bytes(7, 8, &dst_u64 as *const u64 as *const _);
            let grid = metal::MTLSize {
                width: len4 as u64,
                height: outer as u64,
                depth: 1,
            };
            let tg = metal::MTLSize {
                width: 64.min(len4 as u64),
                height: 4.min(outer as u64),
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
        }
        HalfFlag::F32 => {
            enc.set_compute_pipeline_state(&k.narrow_lastax);
            enc.set_buffer(0, Some(buffer), 0);
            enc.set_buffer(1, Some(buffer), 0);
            enc.set_bytes(2, 4, &outer as *const u32 as *const _);
            enc.set_bytes(3, 4, &src_axis as *const u32 as *const _);
            enc.set_bytes(4, 4, &start as *const u32 as *const _);
            enc.set_bytes(5, 4, &len as *const u32 as *const _);
            let src_u64 = src as u64;
            let dst_u64 = dst as u64;
            enc.set_bytes(6, 8, &src_u64 as *const u64 as *const _);
            enc.set_bytes(7, 8, &dst_u64 as *const u64 as *const _);
            let grid = metal::MTLSize {
                width: len as u64,
                height: outer as u64,
                depth: 1,
            };
            let tg = metal::MTLSize {
                width: 64.min(len as u64),
                height: 8.min(outer as u64),
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
        }
        HalfFlag::F16 => {
            enc.set_compute_pipeline_state(&k.narrow_lastax_h);
            enc.set_buffer(0, Some(buffer), 0);
            enc.set_buffer(1, Some(buffer), 0);
            enc.set_bytes(2, 4, &outer as *const u32 as *const _);
            enc.set_bytes(3, 4, &src_axis as *const u32 as *const _);
            enc.set_bytes(4, 4, &start as *const u32 as *const _);
            enc.set_bytes(5, 4, &len as *const u32 as *const _);
            let src_u64 = src as u64;
            let dst_u64 = dst as u64;
            enc.set_bytes(6, 8, &src_u64 as *const u64 as *const _);
            enc.set_bytes(7, 8, &dst_u64 as *const u64 as *const _);
            let grid = metal::MTLSize {
                width: len as u64,
                height: outer as u64,
                depth: 1,
            };
            let tg = metal::MTLSize {
                width: 32.min(len as u64),
                height: 8.min(outer as u64),
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
        }
    }
}

pub(crate) fn encode_fused_residual_ln(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    res: usize,
    g: usize,
    b: usize,
    out: usize,
    rows: u32,
    h: u32,
    eps: f32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F32 => &k.fused_residual_ln,
        HalfFlag::F16 => &k.fused_residual_ln_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), res as u64);
    enc.set_buffer(2, Some(buffer), g as u64);
    enc.set_buffer(3, Some(buffer), b as u64);
    enc.set_buffer(4, Some(buffer), out as u64);
    enc.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &h as *const u32 as *const _,
    );
    enc.set_bytes(
        6,
        std::mem::size_of::<f32>() as u64,
        &eps as *const f32 as *const _,
    );
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= h as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    let tg = metal::MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    let tg_count = metal::MTLSize {
        width: rows as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(tg_count, tg);
}

pub(crate) fn encode_fused_residual_rms_norm(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    res: usize,
    g: usize,
    b: usize,
    out: usize,
    rows: u32,
    h: u32,
    eps: f32,
    dt: crate::thunk::HalfFlag,
    sum_out: usize,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F32 => &k.fused_residual_rms_norm,
        HalfFlag::F16 => &k.fused_residual_rms_norm_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    // Task #50: arena base + byte offsets (large set_buffer offsets drop writes).
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    let res_u = res as u64;
    let g_u = g as u64;
    let b_u = b as u64;
    let out_u = out as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    enc.set_bytes(2, 8, &res_u as *const u64 as *const _);
    enc.set_bytes(3, 8, &g_u as *const u64 as *const _);
    enc.set_bytes(4, 8, &b_u as *const u64 as *const _);
    enc.set_bytes(5, 8, &out_u as *const u64 as *const _);
    // Dual output (f32 kernel only): the pre-norm sum for the skip stream.
    if matches!(dt, HalfFlag::F32) {
        let sum_u = sum_out as u64;
        enc.set_bytes(8, 8, &sum_u as *const u64 as *const _);
    }
    enc.set_bytes(
        6,
        std::mem::size_of::<u32>() as u64,
        &h as *const u32 as *const _,
    );
    enc.set_bytes(
        7,
        std::mem::size_of::<f32>() as u64,
        &eps as *const f32 as *const _,
    );
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= h as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    let tg = metal::MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    let tg_count = metal::MTLSize {
        width: rows as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(tg_count, tg);
}

/// Choose the flash-decode partition count `P` for an m=1 decode attention:
/// enough `heads*P` threadgroups to fill the GPU, but ≥32 keys/partition so
/// each partition does real work. `RLX_METAL_SDPA_FLASH_P` overrides.
pub(crate) fn sdpa_flash_partitions(batch: u32, heads: u32, kv_seq: u32) -> u32 {
    sdpa_flash_partitions_tuned(batch, heads, kv_seq, 128, 32, 1)
}

/// Tile-aware partition chooser for m=1 flash decode. `tile_n` tunes
/// occupancy target (threadgroups in flight), `tile_k` tunes desired keys per
/// partition (granularity of KV slicing).
pub(crate) fn sdpa_flash_partitions_tuned(
    batch: u32,
    heads: u32,
    kv_seq: u32,
    tile_n: u32,
    tile_k: u32,
    pad_kv: u32,
) -> u32 {
    if let Some(p) = crate::runtime_config().sdpa_flash_partitions {
        return p.max(1);
    }
    let bh = (batch * heads).max(1);
    // ~128 threadgroups fills the M4 Pro; past that the combine + redundant-Q
    // overhead outweighs the occupancy gain (measured P-sweep @4k ctx: P=8→13.0,
    // P=16→12.3, P=32→11.6 tok/s at 16 heads).
    let target_tg = 128u32.max(tile_n.clamp(32, 1024));
    let by_occupancy = target_tg.div_ceil(bh).max(1);
    // ≥64 keys/partition — measured sweet spot for m=1 decode (M4 Pro): at ~256
    // ctx, P=4 (64 keys/part) beats P=8 (32/part → combine overhead ≈ base
    // kernel) and P=2; attention 6.4→3.75ms. Fewer, fatter partitions keep the
    // combine cheap while still filling the GPU; occupancy still caps P at long ctx.
    let keys_per_part = tile_k.clamp(64, 1024);
    let padded_kv = kv_seq.div_ceil(pad_kv.max(1)) * pad_kv.max(1);
    let by_keys = (padded_kv / keys_per_part).max(1);
    by_occupancy.min(by_keys).max(1).min(64)
}

/// Flash-decoding (split-KV) SDPA for m=1 decode: `batch*heads*n_part`
/// threadgroups each attend one KV slice and write a partial online-softmax
/// state to `scratch`; `sdpa_decode_m1_combine` then merges the partials into
/// OUT. Raises the decode-attention threadgroup count from `batch*heads` (≈16)
/// to `batch*heads*n_part`, fixing the occupancy starvation. Numerically equal
/// to `sdpa_decode_m1` (same online-softmax math, just partitioned).
///
/// `scratch` must hold ≥ `batch*heads*n_part*(2+128)` floats. Requires
/// head_dim ≤ 128 and v_head_dim ≤ 128 (per-thread register accumulators).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_sdpa_flash_decode(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    scratch: &metal::Buffer,
    n_part: u32,
    q: usize,
    k_off: usize,
    v: usize,
    mask: usize,
    out: usize,
    batch: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    v_head_dim: u32,
    seq_stride: u32,
    mask_kind: u32,
    window: u32,
    kv_seq: u32,
    kv_stride: u32,
    bhsd: u32,
    score_scale: f32,
    attn_logit_softcap: f32,
    kv_f16: bool,
) {
    let kv_heads = if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
        heads
    } else {
        kv_heads
    };
    let v_head_dim = if v_head_dim == 0 {
        head_dim
    } else {
        v_head_dim
    };
    let kv_v_pack: u64 = (kv_heads as u64) | ((v_head_dim as u64) << 32);
    let offs_pack: [u64; 6] = [
        q as u64,
        k_off as u64,
        v as u64,
        mask as u64,
        out as u64,
        kv_v_pack,
    ];
    let u4 = std::mem::size_of::<u32>() as u64;
    let f4 = std::mem::size_of::<f32>() as u64;

    // ── Pass 1: partials ────────────────────────────────────────────────
    // Head-dim-split variant (OPT-IN: RLX_METAL_SDPA_HDSPLIT=1) when D and vdh
    // are multiples of 32: 8 regs/thread vs 256, coalesced K/V, no cross-lane o
    // reduce. Token-identical, but MEASURED NEUTRAL on qwen3-0.6B decode — flash
    // already yields 64-128 tgs, so occupancy isn't register-limited here, and
    // it adds a per-key simd_sum. Kept for few-head / low-P models where the tg
    // count IS register-bound. Default = the proven KV-split kernel.
    let hd_split = head_dim.is_multiple_of(32)
        && v_head_dim.is_multiple_of(32)
        && rlx_ir::env::var("RLX_METAL_SDPA_HDSPLIT").as_deref() == Some("1");
    let partial = match (hd_split, kv_f16) {
        (true, true) => &k.sdpa_decode_m1_partial_hd_f16kv,
        (true, false) => &k.sdpa_decode_m1_partial_hd,
        (false, true) => &k.sdpa_decode_m1_partial_f16kv,
        (false, false) => &k.sdpa_decode_m1_partial,
    };
    enc.set_compute_pipeline_state(partial);
    for i in 0..5u64 {
        enc.set_buffer(i, Some(buffer), 0);
    }
    enc.set_bytes(5, u4, &batch as *const u32 as *const _);
    enc.set_bytes(6, u4, &heads as *const u32 as *const _);
    enc.set_bytes(7, u4, &head_dim as *const u32 as *const _);
    enc.set_bytes(8, u4, &seq_stride as *const u32 as *const _);
    enc.set_bytes(9, u4, &mask_kind as *const u32 as *const _);
    enc.set_bytes(10, u4, &kv_seq as *const u32 as *const _);
    enc.set_bytes(11, u4, &kv_stride as *const u32 as *const _);
    enc.set_bytes(12, u4, &bhsd as *const u32 as *const _);
    enc.set_bytes(13, u4, &window as *const u32 as *const _);
    enc.set_bytes(14, f4, &score_scale as *const f32 as *const _);
    enc.set_bytes(15, f4, &attn_logit_softcap as *const f32 as *const _);
    enc.set_bytes(
        16,
        (6 * std::mem::size_of::<u64>()) as u64,
        offs_pack.as_ptr() as *const _,
    );
    enc.set_buffer(17, Some(scratch), 0);
    enc.set_bytes(18, u4, &n_part as *const u32 as *const _);
    enc.dispatch_thread_groups(
        metal::MTLSize {
            width: (batch as u64) * (heads as u64) * (n_part as u64),
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    // ── Pass 2: combine (reads scratch written above; Serial-ordered) ────
    enc.set_compute_pipeline_state(&k.sdpa_decode_m1_combine);
    enc.set_buffer(0, Some(scratch), 0);
    enc.set_buffer(1, Some(buffer), 0);
    enc.set_bytes(2, u4, &batch as *const u32 as *const _);
    enc.set_bytes(3, u4, &heads as *const u32 as *const _);
    enc.set_bytes(4, u4, &n_part as *const u32 as *const _);
    enc.set_bytes(5, u4, &head_dim as *const u32 as *const _);
    enc.set_bytes(6, u4, &seq_stride as *const u32 as *const _);
    enc.set_bytes(7, u4, &bhsd as *const u32 as *const _);
    enc.set_bytes(
        8,
        (6 * std::mem::size_of::<u64>()) as u64,
        offs_pack.as_ptr() as *const _,
    );
    let combine_threads = (v_head_dim.max(1) as u64).min(128);
    enc.dispatch_thread_groups(
        metal::MTLSize {
            width: (batch as u64) * (heads as u64),
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: combine_threads,
            height: 1,
            depth: 1,
        },
    );
}

/// W8A8 flash-decode: quantize the KV cache to int8 (into `i8scratch`), then run
/// the int8 Q·K integer-dot partial + the SHARED f32 combine. Same math/shape as
/// `encode_sdpa_flash_decode` but the score dot is integer (int8×int8→int32).
/// `i8scratch` holds, in order (each 256-B aligned): int8 K, int8 V, f32 K-scales,
/// f32 V-scales. The Serial encoder guarantees quantize → partial → combine order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_sdpa_flash_decode_w8a8(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    scratch: &metal::Buffer,
    i8scratch: &metal::Buffer,
    n_part: u32,
    q: usize,
    k_off: usize,
    v: usize,
    mask: usize,
    out: usize,
    batch: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    v_head_dim: u32,
    seq_stride: u32,
    mask_kind: u32,
    window: u32,
    kv_seq: u32,
    kv_stride: u32,
    bhsd: u32,
    score_scale: f32,
    attn_logit_softcap: f32,
    kv_f16: bool,
) {
    let kv_heads = if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
        heads
    } else {
        kv_heads
    };
    let v_head_dim = if v_head_dim == 0 {
        head_dim
    } else {
        v_head_dim
    };
    let kv_v_pack: u64 = (kv_heads as u64) | ((v_head_dim as u64) << 32);
    let offs_pack: [u64; 6] = [
        q as u64,
        k_off as u64,
        v as u64,
        mask as u64,
        out as u64,
        kv_v_pack,
    ];
    let u4 = std::mem::size_of::<u32>() as u64;
    let f4 = std::mem::size_of::<f32>() as u64;

    // Scaling / V-source modes (env-gated probes). blk=1: per-32-block scales;
    // v_i8=0 (RLX_METAL_W8A8_VMODE=f32): exact f32 V from arena (isolates K error).
    let blk: u32 = if rlx_ir::env::flag("RLX_METAL_W8A8_BLOCK") {
        1
    } else {
        0
    };
    let v_i8: u32 = if rlx_ir::env::var("RLX_METAL_W8A8_VMODE").as_deref() == Some("f32") {
        0
    } else {
        1
    };
    let nbk = if blk != 0 { head_dim as u64 / 32 } else { 1 };

    // int8-scratch section offsets (each 256-B aligned for set_buffer). Scales are
    // nbk/nbv per row (per-32-block when blk=1, else 1 per row).
    let align256 = |x: u64| (x + 255) & !255u64;
    let nrows = (batch as u64) * (kv_heads as u64) * (kv_seq as u64);
    let i8k_off: u64 = 0;
    let i8v_off = align256(nrows * head_dim as u64);
    let ksc_off = align256(i8v_off + nrows * v_head_dim as u64);
    let vsc_off = align256(ksc_off + nrows * nbk * f4);

    // Incremental-quantize TIMING probe: RLX_METAL_W8A8_INCR=<tokens> dispatches
    // the quantize over only the last <tokens> rows instead of the whole cache —
    // GPU op timing is value-independent, so the tps is FAITHFUL to a real
    // persistent-int8-cache + incremental-append design (tokens are wrong, timing
    // is not). Answers "does incremental flip W8A8 positive" without the full
    // (RoPE-timing/prefill-seed/bucket-transition) cache-lifecycle integration.
    let incr_tokens: u64 = rlx_ir::env::var("RLX_METAL_W8A8_INCR")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let quant_rows = if incr_tokens > 0 {
        (incr_tokens * kv_heads as u64 * batch as u64).min(nrows)
    } else {
        nrows
    };

    // ── Pass 0: quantize K (and V unless v_i8==0) into the int8 scratch ──
    let quant = if kv_f16 {
        &k.kv_quant_i8_f16
    } else {
        &k.kv_quant_i8
    };
    let quant_kv = |src_off: usize, i8_off: u64, sc_off: u64, dh: u32| {
        enc.set_compute_pipeline_state(quant);
        enc.set_buffer(0, Some(buffer), 0);
        enc.set_buffer(1, Some(i8scratch), i8_off);
        enc.set_buffer(2, Some(i8scratch), sc_off);
        let src = src_off as u64;
        enc.set_bytes(
            3,
            std::mem::size_of::<u64>() as u64,
            &src as *const u64 as *const _,
        );
        let nrows32 = nrows as u32;
        enc.set_bytes(4, u4, &nrows32 as *const u32 as *const _);
        enc.set_bytes(5, u4, &dh as *const u32 as *const _);
        enc.set_bytes(6, u4, &bhsd as *const u32 as *const _);
        enc.set_bytes(7, u4, &kv_heads as *const u32 as *const _);
        enc.set_bytes(8, u4, &kv_seq as *const u32 as *const _);
        enc.set_bytes(9, u4, &kv_stride as *const u32 as *const _);
        enc.set_bytes(10, u4, &blk as *const u32 as *const _);
        let tg = 64u64;
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: quant_rows.div_ceil(tg),
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );
    };
    quant_kv(k_off, i8k_off, ksc_off, head_dim);
    if v_i8 != 0 {
        quant_kv(v, i8v_off, vsc_off, v_head_dim);
    }

    // ── Pass 1: W8A8 partials (int8 Q·K integer dot) ────────────────────
    enc.set_compute_pipeline_state(&k.sdpa_decode_m1_partial_w8a8);
    enc.set_buffer(0, Some(buffer), 0); // Q (arena)
    enc.set_buffer(1, Some(i8scratch), i8k_off);
    enc.set_buffer(2, Some(i8scratch), i8v_off);
    enc.set_buffer(3, Some(buffer), 0); // mask (arena)
    enc.set_buffer(4, Some(i8scratch), ksc_off);
    enc.set_buffer(5, Some(i8scratch), vsc_off);
    enc.set_bytes(6, u4, &batch as *const u32 as *const _);
    enc.set_bytes(7, u4, &heads as *const u32 as *const _);
    enc.set_bytes(8, u4, &head_dim as *const u32 as *const _);
    enc.set_bytes(9, u4, &seq_stride as *const u32 as *const _);
    enc.set_bytes(10, u4, &mask_kind as *const u32 as *const _);
    enc.set_bytes(11, u4, &kv_seq as *const u32 as *const _);
    enc.set_bytes(12, u4, &kv_stride as *const u32 as *const _);
    enc.set_bytes(13, u4, &bhsd as *const u32 as *const _);
    enc.set_bytes(14, u4, &window as *const u32 as *const _);
    enc.set_bytes(15, f4, &score_scale as *const f32 as *const _);
    enc.set_bytes(16, f4, &attn_logit_softcap as *const f32 as *const _);
    enc.set_bytes(
        17,
        (6 * std::mem::size_of::<u64>()) as u64,
        offs_pack.as_ptr() as *const _,
    );
    enc.set_buffer(18, Some(scratch), 0);
    // q_i8: 1 = int8 Q integer dot (W8A8); 0 (RLX_METAL_W8A8_QMODE=f32) = f32 Q ·
    // dequant int8 K. Packed with n_part + v_i8 + blk into one slot (buffer 19) —
    // Metal aliases separate set_bytes past index 16.
    let q_i8: u32 = if rlx_ir::env::var("RLX_METAL_W8A8_QMODE").as_deref() == Some("f32") {
        0
    } else {
        1
    };
    // k_i8=0 (RLX_METAL_W8A8_KMODE=f32): exact K from arena — diagnostic isolation.
    let k_i8: u32 = if rlx_ir::env::var("RLX_METAL_W8A8_KMODE").as_deref() == Some("f32") {
        0
    } else {
        1
    };
    let packed19: u32 =
        (n_part & 0xFFFF) | (q_i8 << 16) | (v_i8 << 17) | (blk << 18) | (k_i8 << 19);
    enc.set_bytes(19, u4, &packed19 as *const u32 as *const _);
    enc.dispatch_thread_groups(
        metal::MTLSize {
            width: (batch as u64) * (heads as u64) * (n_part as u64),
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    // ── Pass 2: combine (shared with the f32 flash path) ────────────────
    enc.set_compute_pipeline_state(&k.sdpa_decode_m1_combine);
    enc.set_buffer(0, Some(scratch), 0);
    enc.set_buffer(1, Some(buffer), 0);
    enc.set_bytes(2, u4, &batch as *const u32 as *const _);
    enc.set_bytes(3, u4, &heads as *const u32 as *const _);
    enc.set_bytes(4, u4, &n_part as *const u32 as *const _);
    enc.set_bytes(5, u4, &head_dim as *const u32 as *const _);
    enc.set_bytes(6, u4, &seq_stride as *const u32 as *const _);
    enc.set_bytes(7, u4, &bhsd as *const u32 as *const _);
    enc.set_bytes(
        8,
        (6 * std::mem::size_of::<u64>()) as u64,
        offs_pack.as_ptr() as *const _,
    );
    let combine_threads = (v_head_dim.max(1) as u64).min(128);
    enc.dispatch_thread_groups(
        metal::MTLSize {
            width: (batch as u64) * (heads as u64),
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: combine_threads,
            height: 1,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_sdpa(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    q: usize,
    k_off: usize,
    v: usize,
    mask: usize,
    out: usize,
    batch: u32,
    seq: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    v_head_dim: u32,
    dt: crate::thunk::HalfFlag,
    seq_stride: u32,
    mask_kind: u32,
    window: u32,
    kv_seq: u32,
    kv_stride: u32,
    bhsd: u32,
    score_scale: f32,
    attn_logit_softcap: f32,
    kv_f16: bool,
) {
    // The kernels read these as constants right after `window`. Sentinel
    // `0.0` keeps the existing default (`1/sqrt(head_dim)`, no softcap).
    // Honour caller's score_scale (Gemma 4 sets 1.0); pass 0.0 (sentinel)
    // so the kernel computes the default itself. This matches the
    // historical MSL behaviour where `score_scale` was nonexistent.
    let kernel_score_scale: f32 = score_scale;
    let kernel_softcap: f32 = attn_logit_softcap;
    let kv_heads = if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
        heads
    } else {
        kv_heads
    };
    // V/output per-head width (== head_dim for symmetric SDPA). Packed into the
    // high 32 bits of the SdpaOffsets kv_heads slot so no extra arg-table slot
    // is needed (the Metal arg-table aliases individual set_bytes past index 16).
    let v_head_dim = if v_head_dim == 0 {
        head_dim
    } else {
        v_head_dim
    };
    let kv_v_pack: u64 = (kv_heads as u64) | ((v_head_dim as u64) << 32);
    use crate::thunk::HalfFlag;
    // The two-pass `sdpa` / `sdpa_h` kernels store an [seq, seq] scores
    // matrix in threadgroup memory (`scores[64*64]`); they're correct
    // only for self-attention prefill where Lq == Lk and seq ≤ 64.
    // For longer sequences (e.g. NomicVision's seq=257
    // = 256 patches + 1 CLS) we route to `sdpa_long`, an online-softmax
    // FA-v1 variant that's O(D) memory per query row and scales to any
    // seq length. Also route decode steps (Lq=1, Lk=past+1) through
    // `sdpa_long` — the rectangular `sdpa` TG scores buffer is sized for
    // self-attention prefill; bucketed decode must use the online kernel.
    // F16 input/output isn't supported by sdpa_long yet —
    // that path falls through and would hit the seq-64 ceiling; today
    // no f16-tagged graph hits seq>64 in production.
    if matches!(dt, HalfFlag::F32) && (seq > 64 || kv_seq > 64 || kv_seq != seq) {
        // Pick between the scalar online-softmax (`sdpa_long`) and the
        // tile-based flash-attention (`sdpa_fa_f32`). FA amortizes K/V
        // reads across an 8-query tile via threadgroup memory, so it
        // wins over `sdpa_long` (~35% faster) when Lk dominates. It
        // still lags MPSGraph's batched matmul decomp for SAM3 image
        // CA (Lq=201, Lk=5184, dh=16) because MPSGraph uses
        // simdgroup_float8x8 internally; opt-in via `RLX_METAL_FA=1`
        // for benchmarking until the kernel is upgraded to use
        // simdgroup matrix primitives.
        // sdpa_fa_f32 is symmetric-only (no v_head_dim path); asymmetric MLA
        // (head_dim=192) can't reach it anyway (head_dim<=32 gate).
        let use_fa = kv_seq >= 256
            && head_dim <= 32
            && v_head_dim == head_dim
            && rlx_ir::env::flag("RLX_METAL_FA");
        let use_decode_m1 = seq == 1
            && kv_seq != seq
            && head_dim <= 512
            && rlx_ir::env::var("RLX_METAL_SDPA_DECODE_M1").as_deref() != Some("0");
        // New prefill parallelizations (opt-in A/B). Both share sdpa_long's arg
        // layout (SdpaOffsets @ buffer 17); only the grid/threadgroup shape and
        // pipeline differ. Gated to Lq>1 (prefill) so decode still uses decode_m1.
        let use_mma = seq > 1
            && head_dim <= 64
            && v_head_dim <= 64
            && head_dim.is_multiple_of(8)
            && v_head_dim % 8 == 0
            && rlx_ir::env::flag("RLX_METAL_SDPA_MMA");
        let use_fa2 = seq > 1
            && !use_mma
            && head_dim <= 64
            && v_head_dim <= 64
            && rlx_ir::env::flag("RLX_METAL_SDPA_FA2");
        let use_splitk = seq > 1
            && !use_fa2
            && !use_mma
            && head_dim <= 128
            && v_head_dim <= 128
            && rlx_ir::env::flag("RLX_METAL_SDPA_SPLITK");
        // Tiled flash-attention prefill (fixes O(seq²) sdpa_long at head_dim=128).
        // Default for multi-row F32 prefill; the K/V tile is staged to tg memory
        // (≈Br× less DRAM than sdpa_long). Handles causal / padding / sliding /
        // no-mask; additive-bias (mask_kind==3) still uses sdpa_long. Off with
        // RLX_METAL_PREFILL_FA=0.
        let use_prefill_fa = seq > 1
            && !use_mma
            && !use_fa2
            && !use_splitk
            && !use_fa
            && matches!(dt, HalfFlag::F32)
            && head_dim <= 128
            && v_head_dim <= 128
            && mask_kind != 3
            && rlx_ir::env::var("RLX_METAL_PREFILL_FA").as_deref() != Some("0");
        // simdgroup-MMA variant (score/PV matmuls on the Apple tensor units).
        // Opt-in (RLX_METAL_PREFILL_FA_MMA=1): measured a WASH vs the scalar FA
        // at head_dim=128 — the kernel is barrier/softmax-bound, not matmul-bound,
        // so the tensor units have nothing to bite on. Kept for other shapes/HW.
        let use_prefill_fa_mma = use_prefill_fa
            && head_dim.is_multiple_of(8)
            && v_head_dim.is_multiple_of(8)
            && rlx_ir::env::var("RLX_METAL_PREFILL_FA_MMA").as_deref() == Some("1");
        let pipeline = if use_mma {
            &k.sdpa_mma
        } else if use_fa2 {
            &k.sdpa_fa2
        } else if use_splitk {
            &k.sdpa_splitk
        } else if use_fa {
            if seq > 1 {
                crate::prefill_stats::record_sdpa_prefill_fa();
            }
            &k.sdpa_fa_f32
        } else if use_prefill_fa {
            crate::prefill_stats::record_sdpa_prefill_fa();
            if use_prefill_fa_mma {
                &k.sdpa_prefill_fa_mma
            } else {
                &k.sdpa_prefill_fa
            }
        } else if use_decode_m1 {
            // F16-resident KV cache: read K/V as half (f32 Q/accum/out).
            if kv_f16 {
                &k.sdpa_decode_m1_f16kv
            } else {
                &k.sdpa_decode_m1
            }
        } else if seq > 1 && rlx_ir::env::flag("RLX_METAL_SDPA_OCCPAD") {
            crate::prefill_stats::record_sdpa_long();
            // Occupancy probe: sdpa_long + 20 KB dummy tgMem (same work, lower occupancy).
            &k.sdpa_long_occpad
        } else {
            if seq > 1 {
                crate::prefill_stats::record_sdpa_long();
            }
            &k.sdpa_long
        };
        enc.set_compute_pipeline_state(pipeline);
        // Bind to arena base (offset 0) and pass byte offsets via inline
        // constants — large `set_buffer` offsets silently lose kernel writes
        // on M-series at offsets ≥ ~4 GB (task #50, same pattern as the
        // non-long `sdpa` path below).
        enc.set_buffer(0, Some(buffer), 0);
        enc.set_buffer(1, Some(buffer), 0);
        enc.set_buffer(2, Some(buffer), 0);
        enc.set_buffer(3, Some(buffer), 0);
        enc.set_buffer(4, Some(buffer), 0);
        enc.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &batch as *const u32 as *const _,
        );
        if use_decode_m1 {
            enc.set_bytes(
                6,
                std::mem::size_of::<u32>() as u64,
                &heads as *const u32 as *const _,
            );
            enc.set_bytes(
                7,
                std::mem::size_of::<u32>() as u64,
                &head_dim as *const u32 as *const _,
            );
            enc.set_bytes(
                8,
                std::mem::size_of::<u32>() as u64,
                &seq_stride as *const u32 as *const _,
            );
            enc.set_bytes(
                9,
                std::mem::size_of::<u32>() as u64,
                &mask_kind as *const u32 as *const _,
            );
            enc.set_bytes(
                10,
                std::mem::size_of::<u32>() as u64,
                &kv_seq as *const u32 as *const _,
            );
            enc.set_bytes(
                11,
                std::mem::size_of::<u32>() as u64,
                &kv_stride as *const u32 as *const _,
            );
            enc.set_bytes(
                12,
                std::mem::size_of::<u32>() as u64,
                &bhsd as *const u32 as *const _,
            );
            enc.set_bytes(
                13,
                std::mem::size_of::<u32>() as u64,
                &window as *const u32 as *const _,
            );
            enc.set_bytes(
                14,
                std::mem::size_of::<f32>() as u64,
                &kernel_score_scale as *const f32 as *const _,
            );
            enc.set_bytes(
                15,
                std::mem::size_of::<f32>() as u64,
                &kernel_softcap as *const f32 as *const _,
            );
            let long_offs_pack: [u64; 6] = [
                q as u64,
                k_off as u64,
                v as u64,
                mask as u64,
                out as u64,
                // slot 5: low 32b = kv_heads, high 32b = v_head_dim (SdpaOffsets)
                kv_v_pack,
            ];
            enc.set_bytes(
                16,
                (6 * std::mem::size_of::<u64>()) as u64,
                long_offs_pack.as_ptr() as *const _,
            );
            let n_tg = (batch as u64) * (heads as u64);
            // Split-K decode: 32 threads/head (see sdpa_decode_m1). Larger
            // head_dim falls back to tid==0 inside the kernel.
            let threads_per_tg: u64 = if head_dim <= 128 { 32 } else { 1 };
            let grid = metal::MTLSize {
                width: n_tg,
                height: 1,
                depth: 1,
            };
            let tg = metal::MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            };
            enc.dispatch_thread_groups(grid, tg);
            return;
        }
        // sdpa_long / FA path — also pass kv_heads at buffer 17.
        enc.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &seq as *const u32 as *const _,
        );
        enc.set_bytes(
            7,
            std::mem::size_of::<u32>() as u64,
            &heads as *const u32 as *const _,
        );
        enc.set_bytes(
            8,
            std::mem::size_of::<u32>() as u64,
            &head_dim as *const u32 as *const _,
        );
        enc.set_bytes(
            9,
            std::mem::size_of::<u32>() as u64,
            &seq_stride as *const u32 as *const _,
        );
        enc.set_bytes(
            10,
            std::mem::size_of::<u32>() as u64,
            &mask_kind as *const u32 as *const _,
        );
        enc.set_bytes(
            11,
            std::mem::size_of::<u32>() as u64,
            &kv_seq as *const u32 as *const _,
        );
        enc.set_bytes(
            12,
            std::mem::size_of::<u32>() as u64,
            &kv_stride as *const u32 as *const _,
        );
        enc.set_bytes(
            13,
            std::mem::size_of::<u32>() as u64,
            &bhsd as *const u32 as *const _,
        );
        enc.set_bytes(
            14,
            std::mem::size_of::<u32>() as u64,
            &window as *const u32 as *const _,
        );
        enc.set_bytes(
            15,
            std::mem::size_of::<f32>() as u64,
            &kernel_score_scale as *const f32 as *const _,
        );
        enc.set_bytes(
            16,
            std::mem::size_of::<f32>() as u64,
            &kernel_softcap as *const f32 as *const _,
        );
        let long_offs_pack: [u64; 6] = [
            q as u64,
            k_off as u64,
            v as u64,
            mask as u64,
            out as u64,
            kv_v_pack,
        ];
        enc.set_bytes(
            17,
            (6 * std::mem::size_of::<u64>()) as u64,
            long_offs_pack.as_ptr() as *const _,
        );
        // GQA for long/FA is in SdpaOffsets.kv_heads (FA still indexes as MHA
        // until that kernel grows a kv_heads path — decode_m1 is the hot path).
        if use_fa2 || use_mma {
            // Br=8 query rows per threadgroup. grid = (q_tiles, H, B).
            // fa2 = 64 threads (thread-parallel matmul); mma = 32 (one simdgroup).
            const BR: u32 = 8;
            let q_tiles = seq.div_ceil(BR);
            let grid = metal::MTLSize {
                width: q_tiles as u64,
                height: heads as u64,
                depth: batch as u64,
            };
            let tg = metal::MTLSize {
                width: if use_mma { 32 } else { 64 },
                height: 1,
                depth: 1,
            };
            enc.dispatch_thread_groups(grid, tg);
        } else if use_splitk {
            // One SIMD group (32 threads) per (batch, head, query-row).
            let n_rows = (batch as u64) * (heads as u64) * (seq as u64);
            let grid = metal::MTLSize {
                width: n_rows,
                height: 1,
                depth: 1,
            };
            let tg = metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            };
            enc.dispatch_thread_groups(grid, tg);
        } else if use_fa {
            const BR: u32 = 8;
            let q_tiles = seq.div_ceil(BR);
            let grid = metal::MTLSize {
                width: q_tiles as u64,
                height: heads as u64,
                depth: batch as u64,
            };
            let tg = metal::MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            };
            enc.dispatch_thread_groups(grid, tg);
        } else if use_prefill_fa {
            // Br=8 query rows per threadgroup; grid = (q_tiles, heads, batch).
            // Scalar variant: 128 threads to stage the K/V tiles. MMA variant:
            // 32 threads (one simdgroup owns the tensor-unit matmuls).
            const BR: u32 = 8;
            let q_tiles = seq.div_ceil(BR);
            let grid = metal::MTLSize {
                width: q_tiles as u64,
                height: heads as u64,
                depth: batch as u64,
            };
            let tg = metal::MTLSize {
                width: 128, // 4 simdgroups: staging/scalar use all; MMA distributes
                height: 1,
                depth: 1,
            };
            enc.dispatch_thread_groups(grid, tg);
        } else {
            let total = (batch as u64) * (heads as u64) * (seq as u64);
            let grid = metal::MTLSize {
                width: total,
                height: 1,
                depth: 1,
            };
            let tg = metal::MTLSize {
                width: 64.min(total).max(1),
                height: 1,
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
        }
        return;
    }
    // `sdpa_simd` = SIMD-group-parallel softmax variant of `sdpa` (same
    // dispatch: 32 threads/tg == 1 SIMD group). Opt-in for A/B while validated;
    // once proven it can become the default seq<=64 f32 path.
    let use_simd = matches!(dt, HalfFlag::F32) && rlx_ir::env::flag("RLX_METAL_SDPA_SIMD");
    // f16-scores variant: half the threadgroup memory (8 KiB) for higher
    // occupancy; f32 accumulation preserved. Requires seq<=64 (the [64,64] TG
    // scores buffer). Opt-in via RLX_METAL_SDPA_H16 (implies SIMD softmax).
    let use_h16 = matches!(dt, HalfFlag::F32)
        && rlx_ir::env::flag("RLX_METAL_SDPA_H16")
        && seq <= 64
        && kv_seq <= 64;
    let pipeline = match dt {
        HalfFlag::F32 if use_h16 => &k.sdpa_simd_h16,
        HalfFlag::F32 if use_simd => &k.sdpa_simd,
        HalfFlag::F32 => &k.sdpa,
        HalfFlag::F16 => &k.sdpa_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    // 12B Q4 GGUF activations sit at arena offsets > 4 GB. `set_buffer`'s
    // `offset` param is NSUInteger (64-bit) so the API takes it, but in
    // practice writes from kernels bound this way silently get dropped
    // (task #50 — sentinel `OUT[i]=7.0` doesn't reach the slot, even
    // though CPU writes at the same byte offset DO show up). Workaround:
    // bind to (buffer, 0) and pass byte offsets as ulong constants; the
    // kernel adds them itself. Q4K dequant uses this pattern and works
    // for offsets ≥ 14 GB.
    enc.set_buffer(0, Some(buffer), 0);
    enc.set_buffer(1, Some(buffer), 0);
    enc.set_buffer(2, Some(buffer), 0);
    enc.set_buffer(3, Some(buffer), 0);
    enc.set_buffer(4, Some(buffer), 0);
    let q_off = q as u64;
    let k_off_u = k_off as u64;
    let v_off = v as u64;
    let m_off = mask as u64;
    let o_off = out as u64;
    enc.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &batch as *const u32 as *const _,
    );
    enc.set_bytes(
        6,
        std::mem::size_of::<u32>() as u64,
        &seq as *const u32 as *const _,
    );
    enc.set_bytes(
        7,
        std::mem::size_of::<u32>() as u64,
        &heads as *const u32 as *const _,
    );
    enc.set_bytes(
        8,
        std::mem::size_of::<u32>() as u64,
        &head_dim as *const u32 as *const _,
    );
    enc.set_bytes(
        9,
        std::mem::size_of::<u32>() as u64,
        &seq_stride as *const u32 as *const _,
    );
    enc.set_bytes(
        10,
        std::mem::size_of::<u32>() as u64,
        &mask_kind as *const u32 as *const _,
    );
    enc.set_bytes(
        11,
        std::mem::size_of::<u32>() as u64,
        &kv_seq as *const u32 as *const _,
    );
    enc.set_bytes(
        12,
        std::mem::size_of::<u32>() as u64,
        &kv_stride as *const u32 as *const _,
    );
    enc.set_bytes(
        13,
        std::mem::size_of::<u32>() as u64,
        &bhsd as *const u32 as *const _,
    );
    enc.set_bytes(
        14,
        std::mem::size_of::<u32>() as u64,
        &window as *const u32 as *const _,
    );
    enc.set_bytes(
        15,
        std::mem::size_of::<f32>() as u64,
        &kernel_score_scale as *const f32 as *const _,
    );
    enc.set_bytes(
        16,
        std::mem::size_of::<f32>() as u64,
        &kernel_softcap as *const f32 as *const _,
    );
    // Pack 5 byte-offsets into one inline-constant buffer (5×u64 = 40 bytes).
    // Setting them individually at buffer indices 17..21 turned out to bind
    // the SAME value to all five slots — Metal's argument table seemed to
    // alias them past index 16. Packing into one struct sidesteps that
    // (task #50, post-u64 dequant fix).
    let offs_pack: [u64; 6] = [q_off, k_off_u, v_off, m_off, o_off, kv_v_pack];
    enc.set_bytes(
        17,
        (6 * std::mem::size_of::<u64>()) as u64,
        offs_pack.as_ptr() as *const _,
    );
    let tg_count = metal::MTLSize {
        width: (batch * heads) as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(tg_count, tg);
}

/// Native block-quantized (int8 / int4) weight matmul over the unified-memory
/// arena. `out[m,n] = x[m,k] @ dequant(wq)`, one GPU thread per output element.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_dequant_matmul(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    scale: usize,
    zp: usize,
    dst: usize,
    m: u32,
    k: u32,
    n: u32,
    block_size: u32,
    asym: u32,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), w_q as u64);
    enc.set_buffer(2, Some(buffer), scale as u64);
    enc.set_buffer(3, Some(buffer), zp as u64);
    enc.set_buffer(4, Some(buffer), dst as u64);
    let sz = std::mem::size_of::<u32>() as u64;
    enc.set_bytes(5, sz, &m as *const u32 as *const _);
    enc.set_bytes(6, sz, &k as *const u32 as *const _);
    enc.set_bytes(7, sz, &n as *const u32 as *const _);
    enc.set_bytes(8, sz, &block_size as *const u32 as *const _);
    enc.set_bytes(9, sz, &asym as *const u32 as *const _);
    let total = (m * n) as u64;
    let grid = metal::MTLSize {
        width: total,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: total.min(256),
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_dequant_matmul_fp8(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    scale: usize,
    dst: usize,
    m: u32,
    k: u32,
    n: u32,
    e5m2: u32,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), w_q as u64);
    enc.set_buffer(2, Some(buffer), scale as u64);
    enc.set_buffer(4, Some(buffer), dst as u64);
    let sz = std::mem::size_of::<u32>() as u64;
    enc.set_bytes(5, sz, &m as *const u32 as *const _);
    enc.set_bytes(6, sz, &k as *const u32 as *const _);
    enc.set_bytes(7, sz, &n as *const u32 as *const _);
    enc.set_bytes(8, sz, &e5m2 as *const u32 as *const _);
    let total = (m * n) as u64;
    let grid = metal::MTLSize {
        width: total,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: total.min(256),
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_dequant_matmul_nvfp4(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    scale: usize,
    global_scale: usize,
    dst: usize,
    m: u32,
    k: u32,
    n: u32,
) {
    use rlx_ir::NVFP4_GROUP_SIZE;
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), w_q as u64);
    enc.set_buffer(2, Some(buffer), scale as u64);
    enc.set_buffer(3, Some(buffer), global_scale as u64);
    enc.set_buffer(4, Some(buffer), dst as u64);
    let sz = std::mem::size_of::<u32>() as u64;
    enc.set_bytes(5, sz, &m as *const u32 as *const _);
    enc.set_bytes(6, sz, &k as *const u32 as *const _);
    enc.set_bytes(7, sz, &n as *const u32 as *const _);
    let gs = NVFP4_GROUP_SIZE as u32;
    enc.set_bytes(8, sz, &gs as *const u32 as *const _);
    let total = (m * n) as u64;
    let grid = metal::MTLSize {
        width: total,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: total.min(256),
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// MxFp4x2 two-level residual E2M1 fused decode-matmul (`dequant_matmul_mxfp4x2`
/// MSL kernel). `w_q`=[plane0|plane1] nibbles, `scale`=[s0|s1] f32.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_dequant_matmul_mxfp4x2(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    scale: usize,
    dst: usize,
    m: u32,
    k: u32,
    n: u32,
    group: u32,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), w_q as u64);
    enc.set_buffer(2, Some(buffer), scale as u64);
    enc.set_buffer(3, Some(buffer), dst as u64);
    let sz = std::mem::size_of::<u32>() as u64;
    enc.set_bytes(4, sz, &m as *const u32 as *const _);
    enc.set_bytes(5, sz, &k as *const u32 as *const _);
    enc.set_bytes(6, sz, &n as *const u32 as *const _);
    enc.set_bytes(7, sz, &group as *const u32 as *const _);
    let total = (m * n) as u64;
    let grid = metal::MTLSize {
        width: total,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: total.min(256),
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// Prefill GEMM (`m > 1`): one threadgroup per `(col, row_tile)`;
/// threads split K and stage an X tile in threadgroup memory.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_dequant_matmul_mlx_gemm(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    scale: usize,
    zp: usize,
    dst: usize,
    m: u32,
    k: u32,
    n: u32,
    kind: u32,
    bits: u32,
    group_size: u32,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), w_q as u64);
    enc.set_buffer(2, Some(buffer), scale as u64);
    enc.set_buffer(3, Some(buffer), zp as u64);
    enc.set_buffer(4, Some(buffer), dst as u64);
    let sz = std::mem::size_of::<u32>() as u64;
    enc.set_bytes(5, sz, &m as *const u32 as *const _);
    enc.set_bytes(6, sz, &k as *const u32 as *const _);
    enc.set_bytes(7, sz, &n as *const u32 as *const _);
    enc.set_bytes(8, sz, &kind as *const u32 as *const _);
    enc.set_bytes(9, sz, &bits as *const u32 as *const _);
    enc.set_bytes(10, sz, &group_size as *const u32 as *const _);
    let n_row_tiles = m.div_ceil(8);
    let total = (n * n_row_tiles) as u64;
    let grid = metal::MTLSize {
        width: total,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg);
}

/// Decode GEMV (`m == 1`): one threadgroup per output column.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_dequant_matmul_mlx_gemv(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    scale: usize,
    zp: usize,
    dst: usize,
    k: u32,
    n: u32,
    kind: u32,
    bits: u32,
    group_size: u32,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), w_q as u64);
    enc.set_buffer(2, Some(buffer), scale as u64);
    enc.set_buffer(3, Some(buffer), zp as u64);
    enc.set_buffer(4, Some(buffer), dst as u64);
    let sz = std::mem::size_of::<u32>() as u64;
    enc.set_bytes(5, sz, &k as *const u32 as *const _);
    enc.set_bytes(6, sz, &n as *const u32 as *const _);
    enc.set_bytes(7, sz, &kind as *const u32 as *const _);
    enc.set_bytes(8, sz, &bits as *const u32 as *const _);
    enc.set_bytes(9, sz, &group_size as *const u32 as *const _);
    let tg = metal::MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let grid = metal::MTLSize {
        width: n as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg);
}

/// Native MoE expert GEMV (m==1): like [`encode_dequant_matmul_mlx_gemv`] plus the
/// `e_idx` buffer + `slab_bytes` so the kernel offsets into the stacked expert slab.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_grouped_dequant_matmul_mlx_gemv(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    scale: usize,
    zp: usize,
    dst: usize,
    e_idx: usize,
    k: u32,
    n: u32,
    kind: u32,
    bits: u32,
    group_size: u32,
    slab_bytes: u32,
    scale_bf16: u32,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), w_q as u64);
    enc.set_buffer(2, Some(buffer), scale as u64);
    enc.set_buffer(3, Some(buffer), zp as u64);
    enc.set_buffer(4, Some(buffer), dst as u64);
    enc.set_buffer(5, Some(buffer), e_idx as u64);
    let sz = std::mem::size_of::<u32>() as u64;
    enc.set_bytes(6, sz, &k as *const u32 as *const _);
    enc.set_bytes(7, sz, &n as *const u32 as *const _);
    enc.set_bytes(8, sz, &kind as *const u32 as *const _);
    enc.set_bytes(9, sz, &bits as *const u32 as *const _);
    enc.set_bytes(10, sz, &group_size as *const u32 as *const _);
    enc.set_bytes(11, sz, &slab_bytes as *const u32 as *const _);
    enc.set_bytes(12, sz, &scale_bf16 as *const u32 as *const _);
    let tg = metal::MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let grid = metal::MTLSize {
        width: n as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg);
}

/// Native MoE expert prefill GEMM (m>1): per-row expert via `e_idx` + `slab_bytes`.
/// One threadgroup per (col, row_tile of 8).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_grouped_dequant_matmul_mlx_gemm(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    scale: usize,
    zp: usize,
    dst: usize,
    e_idx: usize,
    m: u32,
    k: u32,
    n: u32,
    kind: u32,
    bits: u32,
    group_size: u32,
    slab_bytes: u32,
    scale_bf16: u32,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), w_q as u64);
    enc.set_buffer(2, Some(buffer), scale as u64);
    enc.set_buffer(3, Some(buffer), zp as u64);
    enc.set_buffer(4, Some(buffer), dst as u64);
    enc.set_buffer(5, Some(buffer), e_idx as u64);
    let sz = std::mem::size_of::<u32>() as u64;
    enc.set_bytes(6, sz, &m as *const u32 as *const _);
    enc.set_bytes(7, sz, &k as *const u32 as *const _);
    enc.set_bytes(8, sz, &n as *const u32 as *const _);
    enc.set_bytes(9, sz, &kind as *const u32 as *const _);
    enc.set_bytes(10, sz, &bits as *const u32 as *const _);
    enc.set_bytes(11, sz, &group_size as *const u32 as *const _);
    enc.set_bytes(12, sz, &slab_bytes as *const u32 as *const _);
    enc.set_bytes(13, sz, &scale_bf16 as *const u32 as *const _);
    let tg = metal::MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let n_row_tiles = m.div_ceil(8);
    let grid = metal::MTLSize {
        width: (n * n_row_tiles) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg);
}

pub(crate) fn encode_rope(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    cos: usize,
    sin: usize,
    dst: usize,
    batch: u32,
    seq: u32,
    hidden: u32,
    head_dim: u32,
    n_rot: u32,
    dt: crate::thunk::HalfFlag,
    src_row_stride: u32,
    seq_stride: u32,
    cos_per_token: bool,
    interleaved: bool,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F32 => &k.rope,
        HalfFlag::F16 => &k.rope_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), cos as u64);
    enc.set_buffer(2, Some(buffer), sin as u64);
    enc.set_buffer(3, Some(buffer), dst as u64);
    enc.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &batch as *const u32 as *const _,
    );
    enc.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &seq as *const u32 as *const _,
    );
    enc.set_bytes(
        6,
        std::mem::size_of::<u32>() as u64,
        &hidden as *const u32 as *const _,
    );
    enc.set_bytes(
        7,
        std::mem::size_of::<u32>() as u64,
        &head_dim as *const u32 as *const _,
    );
    enc.set_bytes(
        8,
        std::mem::size_of::<u32>() as u64,
        &src_row_stride as *const u32 as *const _,
    );
    enc.set_bytes(
        9,
        std::mem::size_of::<u32>() as u64,
        &seq_stride as *const u32 as *const _,
    );
    enc.set_bytes(
        10,
        std::mem::size_of::<u32>() as u64,
        &n_rot as *const u32 as *const _,
    );
    let cos_per_token_u32: u32 = cos_per_token as u32;
    enc.set_bytes(
        11,
        std::mem::size_of::<u32>() as u64,
        &cos_per_token_u32 as *const u32 as *const _,
    );
    let interleaved_u32: u32 = interleaved as u32;
    enc.set_bytes(
        12,
        std::mem::size_of::<u32>() as u64,
        &interleaved_u32 as *const u32 as *const _,
    );
    let nh = hidden / head_dim;
    let grid = metal::MTLSize {
        width: head_dim as u64,
        height: nh as u64,
        depth: (batch * seq) as u64,
    };
    let tg = metal::MTLSize {
        width: head_dim.min(16) as u64,
        height: nh.min(8) as u64,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_rms_norm(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    g: usize,
    b: usize,
    dst: usize,
    rows: u32,
    h: u32,
    eps: f32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F32 => &k.rms_norm,
        HalfFlag::F16 => &k.rms_norm_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    // Task #50: arena base + byte offsets for activations past 4 GB.
    enc.set_buffer(0, Some(buffer), 0);
    let src_u64 = src as u64;
    let g_u64 = g as u64;
    let b_u64 = b as u64;
    let dst_u64 = dst as u64;
    enc.set_bytes(1, 8, &src_u64 as *const u64 as *const _);
    enc.set_bytes(2, 8, &g_u64 as *const u64 as *const _);
    enc.set_bytes(3, 8, &b_u64 as *const u64 as *const _);
    enc.set_bytes(4, 8, &dst_u64 as *const u64 as *const _);
    enc.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &h as *const u32 as *const _,
    );
    enc.set_bytes(
        6,
        std::mem::size_of::<f32>() as u64,
        &eps as *const f32 as *const _,
    );
    // One threadgroup per row; power-of-2 tg size for reduction (see encode_layer_norm).
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= h as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    let grid = metal::MTLSize {
        width: rows as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg);
}

pub(crate) fn encode_ada_layer_norm(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    scale: usize,
    shift: usize,
    out: usize,
    rows: u32,
    h: u32,
    eps: f32,
    layer_norm: bool,
    lead_pack: &[u32; 17],
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F16 => &k.ada_layer_norm_h,
        HalfFlag::F32 => &k.ada_layer_norm,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), scale as u64);
    enc.set_buffer(2, Some(buffer), shift as u64);
    enc.set_buffer(3, Some(buffer), out as u64);
    enc.set_bytes(4, 4, &h as *const u32 as *const _);
    enc.set_bytes(5, 4, &eps as *const f32 as *const _);
    let ln: u32 = u32::from(layer_norm);
    enc.set_bytes(6, 4, &ln as *const u32 as *const _);
    enc.set_bytes(
        7,
        (lead_pack.len() * std::mem::size_of::<u32>()) as u64,
        lead_pack.as_ptr() as *const _,
    );
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= h as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    enc.dispatch_thread_groups(
        metal::MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_gated_residual(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    y: usize,
    gate: usize,
    out: usize,
    rows: u32,
    h: u32,
    lead_pack: &[u32; 17],
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F16 => &k.gated_residual_h,
        HalfFlag::F32 => &k.gated_residual,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), y as u64);
    enc.set_buffer(2, Some(buffer), gate as u64);
    enc.set_buffer(3, Some(buffer), out as u64);
    enc.set_bytes(4, 4, &h as *const u32 as *const _);
    enc.set_bytes(
        5,
        (lead_pack.len() * std::mem::size_of::<u32>()) as u64,
        lead_pack.as_ptr() as *const _,
    );
    let n = rows.saturating_mul(h);
    let tg_w = pipeline.thread_execution_width().min(n as u64).max(1);
    enc.dispatch_threads(
        metal::MTLSize {
            width: n as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_ada_layer_norm_backward(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    scale: usize,
    dy: usize,
    out: usize,
    h: u32,
    eps: f32,
    layer_norm: bool,
    seq_per_mod: u32,
    mod_rows: u32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F16 => &k.ada_layer_norm_backward_h,
        HalfFlag::F32 => &k.ada_layer_norm_backward,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), scale as u64);
    enc.set_buffer(2, Some(buffer), dy as u64);
    enc.set_buffer(3, Some(buffer), out as u64);
    enc.set_bytes(4, 4, &h as *const u32 as *const _);
    enc.set_bytes(5, 4, &eps as *const f32 as *const _);
    let ln: u32 = u32::from(layer_norm);
    enc.set_bytes(6, 4, &ln as *const u32 as *const _);
    enc.set_bytes(7, 4, &seq_per_mod as *const u32 as *const _);
    enc.set_bytes(8, 4, &mod_rows as *const u32 as *const _);
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= h as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    enc.dispatch_thread_groups(
        metal::MTLSize {
            width: mod_rows as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_gated_residual_backward(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    y: usize,
    gate: usize,
    dy: usize,
    out: usize,
    h: u32,
    seq_per_mod: u32,
    mod_rows: u32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F16 => &k.gated_residual_backward_h,
        HalfFlag::F32 => &k.gated_residual_backward,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), y as u64);
    enc.set_buffer(1, Some(buffer), gate as u64);
    enc.set_buffer(2, Some(buffer), dy as u64);
    enc.set_buffer(3, Some(buffer), out as u64);
    enc.set_bytes(4, 4, &h as *const u32 as *const _);
    enc.set_bytes(5, 4, &seq_per_mod as *const u32 as *const _);
    enc.set_bytes(6, 4, &mod_rows as *const u32 as *const _);
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= h as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    enc.dispatch_thread_groups(
        metal::MTLSize {
            width: mod_rows as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_rms_norm_mul_silu(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    g: usize,
    b: usize,
    z: usize,
    dst: usize,
    rows: u32,
    h: u32,
    eps: f32,
) {
    enc.set_compute_pipeline_state(&k.rms_norm_mul_silu);
    enc.set_buffer(0, Some(buffer), 0);
    let src_u64 = src as u64;
    let g_u64 = g as u64;
    let b_u64 = b as u64;
    let z_u64 = z as u64;
    let dst_u64 = dst as u64;
    enc.set_bytes(1, 8, &src_u64 as *const u64 as *const _);
    enc.set_bytes(2, 8, &g_u64 as *const u64 as *const _);
    enc.set_bytes(3, 8, &b_u64 as *const u64 as *const _);
    enc.set_bytes(4, 8, &z_u64 as *const u64 as *const _);
    enc.set_bytes(5, 8, &dst_u64 as *const u64 as *const _);
    enc.set_bytes(
        6,
        std::mem::size_of::<u32>() as u64,
        &h as *const u32 as *const _,
    );
    enc.set_bytes(
        7,
        std::mem::size_of::<f32>() as u64,
        &eps as *const f32 as *const _,
    );
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= h as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    let grid = metal::MTLSize {
        width: rows as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_depthwise_conv1d_bsc(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    w_buffer: &metal::Buffer,
    src: usize,
    wt_raw: usize,
    dst: usize,
    batch: u32,
    width: u32,
    out_seq: u32,
    channels: u32,
    kw: u32,
    silu: bool,
) {
    enc.set_compute_pipeline_state(&k.depthwise_conv1d_bsc);
    enc.set_buffer(0, Some(buffer), 0);
    let src_u = src as u64;
    enc.set_bytes(1, 8, &src_u as *const u64 as *const _);
    let wt_u = wt_raw as u64;
    enc.set_bytes(2, 8, &wt_u as *const u64 as *const _);
    let dst_u = dst as u64;
    enc.set_bytes(3, 8, &dst_u as *const u64 as *const _);
    let dims: [u32; 4] = [batch, width, out_seq, channels];
    enc.set_bytes(4, 16, dims.as_ptr() as *const _);
    let k_silu: [u32; 2] = [kw, u32::from(silu)];
    enc.set_bytes(5, 8, k_silu.as_ptr() as *const _);
    enc.set_buffer(7, Some(w_buffer), 0);
    let total = batch as u64 * out_seq as u64 * channels as u64;
    let grid = metal::MTLSize {
        width: total.max(1),
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(total.max(1)),
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_rms_norm_bwd_input(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    gamma: usize,
    beta: usize,
    dy: usize,
    dx: usize,
    rows: u32,
    h: u32,
    eps: f32,
) {
    enc.set_compute_pipeline_state(&k.rms_norm_bwd);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), gamma as u64);
    enc.set_buffer(2, Some(buffer), beta as u64);
    enc.set_buffer(3, Some(buffer), dy as u64);
    enc.set_buffer(4, Some(buffer), dx as u64);
    enc.set_bytes(5, 4, &h as *const u32 as *const _);
    enc.set_bytes(6, 4, &eps as *const f32 as *const _);
    let wrt: u32 = 0;
    enc.set_bytes(7, 4, &wrt as *const u32 as *const _);
    let tg_w = 256u64.min(h as u64);
    // One threadgroup per row (see encode_softmax comment): dispatch_threads
    // with a row-packed grid is unreliable for reduction kernels on this
    // driver — use the uniform dispatch_thread_groups form instead.
    enc.dispatch_thread_groups(
        metal::MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_rms_norm_bwd_param(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    _gamma: usize,
    _beta: usize,
    dy: usize,
    out: usize,
    rows: u32,
    h: u32,
    eps: f32,
    wrt: u32,
    inv_r_scratch: usize,
) {
    let use_parallel = inv_r_scratch != 0 && rows > 1;
    if !use_parallel {
        enc.set_compute_pipeline_state(&k.rms_norm_bwd_param);
        enc.set_buffer(0, Some(buffer), x as u64);
        enc.set_buffer(1, Some(buffer), _gamma as u64);
        enc.set_buffer(2, Some(buffer), _beta as u64);
        enc.set_buffer(3, Some(buffer), dy as u64);
        enc.set_buffer(4, Some(buffer), out as u64);
        enc.set_bytes(5, 4, &rows as *const u32 as *const _);
        enc.set_bytes(6, 4, &h as *const u32 as *const _);
        enc.set_bytes(7, 4, &eps as *const f32 as *const _);
        enc.set_bytes(8, 4, &wrt as *const u32 as *const _);
        enc.dispatch_threads(
            metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        return;
    }

    if wrt == 1 {
        enc.set_compute_pipeline_state(&k.rms_norm_bwd_inv_r_f32);
        enc.set_buffer(0, Some(buffer), x as u64);
        enc.set_buffer(1, Some(buffer), inv_r_scratch as u64);
        enc.set_bytes(2, 4, &h as *const u32 as *const _);
        enc.set_bytes(3, 4, &eps as *const f32 as *const _);
        enc.dispatch_threads(
            metal::MTLSize {
                width: rows as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256.min(rows as u64).max(1),
                height: 1,
                depth: 1,
            },
        );
    }

    enc.set_compute_pipeline_state(&k.rms_norm_bwd_param_reduce_f32);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), dy as u64);
    enc.set_buffer(2, Some(buffer), inv_r_scratch as u64);
    enc.set_buffer(3, Some(buffer), out as u64);
    enc.set_bytes(4, 4, &rows as *const u32 as *const _);
    enc.set_bytes(5, 4, &h as *const u32 as *const _);
    enc.set_bytes(6, 4, &wrt as *const u32 as *const _);
    let tg_w = 256u64.min(h as u64).max(1);
    enc.dispatch_threads(
        metal::MTLSize {
            width: h as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

/// True when the native `fused_attn_block` MSL kernel can serve this
/// `Op::FusedAttentionBlock`: f32, no bias, rank-3, even head_dim, and a
/// `[seq,seq]` score matrix that fits the kernel's `threadgroup float[64*64]`
/// (`seq ≤ 64`). Everything else decomposes to the primitive chain.
pub(crate) fn fab_is_native(node: &rlx_ir::Node) -> bool {
    if let Op::FusedAttentionBlock {
        head_dim, has_bias, ..
    } = &node.op
    {
        let dims = node.shape.dims();
        dims.len() == 3
            && !*has_bias
            && node.shape.dtype() == rlx_ir::DType::F32
            && dims[1].unwrap_static() <= 64
            && *head_dim % 2 == 0
    } else {
        false
    }
}

/// Expand fused ops that Metal claims for coverage but cannot HostOp through
/// CPU (`Thunk::Nop` catch-all): `FusedConvBiasAct`, `PartitionedConv`,
/// `FusedTransformerLayer`. Metal-native fused kernels (SwiGLU / FMBA /
/// ResidualLN / FAB) are left intact.
pub(crate) fn lower_cpu_nop_fused_for_metal(g: Graph) -> Graph {
    let needs = g.nodes().iter().any(|n| {
        matches!(
            n.op,
            Op::FusedConvBiasAct { .. }
                | Op::PartitionedConv { .. }
                | Op::FusedTransformerLayer { .. }
        )
    });
    if !needs {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();
    let nodes: Vec<rlx_ir::Node> = g.nodes().to_vec();
    for node in &nodes {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = match &node.op {
            Op::FusedConvBiasAct { .. }
            | Op::PartitionedConv { .. }
            | Op::FusedTransformerLayer { .. } => {
                inline_unfused_fused_op(&mut out, &node.op, &new_inputs, &node.shape)
            }
            _ => out.add_node(node.op.clone(), new_inputs, node.shape.clone()),
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

fn inline_unfused_fused_op(
    out: &mut Graph,
    op: &Op,
    inputs: &[NodeId],
    shape: &rlx_ir::Shape,
) -> NodeId {
    let mut mini = Graph::new("mtl_unfuse");
    let mut mini_ins = Vec::with_capacity(inputs.len());
    for (i, &src) in inputs.iter().enumerate() {
        let sh = out.node(src).shape.clone();
        mini_ins.push(mini.append_node(
            Op::Input {
                name: format!("in{i}"),
            },
            vec![],
            sh,
            None,
        ));
    }
    let out_id = mini.append_node(op.clone(), mini_ins, shape.clone(), None);
    mini.set_outputs(vec![out_id]);
    let expanded = rlx_opt::unfuse_fused_for_autodiff(mini);
    let mut map: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();
    for n in expanded.nodes() {
        if let Op::Input { name } = &n.op {
            if let Some(rest) = name.strip_prefix("in") {
                if let Ok(i) = rest.parse::<usize>() {
                    map.insert(n.id, inputs[i]);
                    continue;
                }
            }
        }
        let mapped_ins: Vec<NodeId> = n.inputs.iter().map(|i| map[i]).collect();
        let id = out.add_node(n.op.clone(), mapped_ins, n.shape.clone());
        map.insert(n.id, id);
    }
    map[&expanded.outputs[0]]
}

/// Decompose non-native `Op::FusedAttentionBlock` nodes to the primitive chain
/// (via the shared `expand_attention_block`), leaving native-eligible blocks
/// intact for the `fused_attn_block` kernel. No rewrite when there is no FAB,
/// or when every FAB is already native.
pub(crate) fn lower_fab_for_metal(g: Graph) -> Graph {
    let has_fab = g
        .nodes()
        .iter()
        .any(|n| matches!(n.op, Op::FusedAttentionBlock { .. }));
    if !has_fab {
        return g;
    }
    let all_native = g
        .nodes()
        .iter()
        .all(|n| !matches!(n.op, Op::FusedAttentionBlock { .. }) || fab_is_native(n));
    if all_native {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();
    let nodes: Vec<rlx_ir::Node> = g.nodes().to_vec();
    for node in &nodes {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = if let Op::FusedAttentionBlock {
            num_heads,
            head_dim,
            has_bias,
            has_rope,
        } = &node.op
        {
            if fab_is_native(node) {
                out.add_node(node.op.clone(), new_inputs, node.shape.clone())
            } else {
                rlx_opt::unfuse::expand_attention_block(
                    &mut out,
                    &new_inputs,
                    *num_heads,
                    *head_dim,
                    *has_bias,
                    *has_rope,
                )
            }
        } else {
            out.add_node(node.op.clone(), new_inputs, node.shape.clone())
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

/// Split `FusedMatMulBiasAct { activation: Some(Gelu) }` into a
/// `FusedMatMulBiasAct { activation: None }` (matmul + bias) feeding a
/// standalone `Activation(Gelu)`.
///
/// Apple's MPSGraph **and** the fused `sgemm_simd_bias` epilogue mis-execute an
/// erf-GELU fused onto a matmul→bias: the result diverges by O(1) (repro:
/// `rlx-dinov3 tests/metal_isolate` case `up+b+gelu_erf` — CPU/MLX/wgpu are
/// bit-exact). A *standalone* `Activation(Gelu)` node is exact on Metal, and
/// `FusedMatMulBiasAct{None}` (matmul+bias) is exact — so decompose the erf case
/// into those two, keeping the matmul+bias fusion while routing the erf-GELU
/// through the correct path. tanh-GELU (`GeluApprox`) and SiLU epilogues apply
/// inline correctly and stay fused. Only rewrites when such a node exists.
pub(crate) fn split_erf_gelu_fmba_for_metal(g: Graph) -> Graph {
    use rlx_ir::op::Activation;
    let needs = g.nodes().iter().any(|n| {
        matches!(
            n.op,
            Op::FusedMatMulBiasAct {
                activation: Some(Activation::Gelu)
            }
        )
    });
    if !needs {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();
    let nodes: Vec<rlx_ir::Node> = g.nodes().to_vec();
    for node in &nodes {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = if matches!(
            node.op,
            Op::FusedMatMulBiasAct {
                activation: Some(Activation::Gelu)
            }
        ) {
            let mm_bias = out.add_node(
                Op::FusedMatMulBiasAct { activation: None },
                new_inputs,
                node.shape.clone(),
            );
            out.add_node(
                Op::Activation(Activation::Gelu),
                vec![mm_bias],
                node.shape.clone(),
            )
        } else {
            out.add_node(node.op.clone(), new_inputs, node.shape.clone())
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

#[cfg(test)]
mod split_erf_gelu_tests {
    use super::split_erf_gelu_fmba_for_metal;
    use rlx_ir::op::Activation;
    use rlx_ir::{DType, Graph, Op, Shape};

    fn fmba_graph(activation: Option<Activation>) -> Graph {
        let f = DType::F32;
        let mut g = Graph::new("t");
        let x = g.input("x", Shape::new(&[2, 4], f));
        let w = g.param("w", Shape::new(&[4, 4], f));
        let b = g.param("b", Shape::new(&[4], f));
        let fused = g.add_node(
            Op::FusedMatMulBiasAct { activation },
            vec![x, w, b],
            Shape::new(&[2, 4], f),
        );
        g.set_outputs(vec![fused]);
        g
    }

    #[test]
    fn erf_gelu_fmba_is_split_into_matmul_bias_plus_standalone_gelu() {
        let out = split_erf_gelu_fmba_for_metal(fmba_graph(Some(Activation::Gelu)));
        // The buggy fused erf-GELU node is gone …
        assert!(out.nodes().iter().all(|n| !matches!(
            n.op,
            Op::FusedMatMulBiasAct {
                activation: Some(Activation::Gelu)
            }
        )));
        // … replaced by a fused matmul+bias feeding a standalone GELU.
        assert!(
            out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::FusedMatMulBiasAct { activation: None }))
        );
        assert!(
            out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Activation(Activation::Gelu)))
        );
    }

    #[test]
    fn tanh_gelu_fmba_stays_fused() {
        // tanh-GELU applies inline correctly on Metal — leave it fused.
        let out = split_erf_gelu_fmba_for_metal(fmba_graph(Some(Activation::GeluApprox)));
        assert!(out.nodes().iter().any(|n| matches!(
            n.op,
            Op::FusedMatMulBiasAct {
                activation: Some(Activation::GeluApprox)
            }
        )));
        assert!(
            !out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Activation(_)))
        );
    }
}

/// Per native-FAB-node `(qkv, attn)` BYTE offsets *relative to the FAB scratch
/// base*, plus the total scratch size in bytes. `qkv = [B,S,3*inner]` and
/// `attn = [B,S,inner]` (both f32), each block 128-byte aligned.
pub(crate) fn fab_scratch_layout(graph: &Graph) -> (usize, Vec<(NodeId, usize, usize)>) {
    let mut rel: Vec<(NodeId, usize, usize)> = Vec::new();
    let mut cur: usize = 0;
    for node in graph.nodes() {
        if !fab_is_native(node) {
            continue;
        }
        if let Op::FusedAttentionBlock {
            num_heads,
            head_dim,
            ..
        } = &node.op
        {
            let dims = node.shape.dims();
            let b = dims[0].unwrap_static();
            let s = dims[1].unwrap_static();
            let inner = num_heads * head_dim;
            cur = (cur + 127) & !127;
            let qkv_off = cur;
            cur += b * s * 3 * inner * 4;
            cur = (cur + 127) & !127;
            let attn_off = cur;
            cur += b * s * inner * 4;
            rel.push((node.id, qkv_off, attn_off));
        }
    }
    (cur, rel)
}

pub(crate) fn rms_norm_bwd_scratch_bytes(graph: &Graph) -> usize {
    let mut max_bytes = 0usize;
    for node in graph.nodes() {
        match &node.op {
            Op::RmsNormBackwardGamma { .. } | Op::RmsNormBackwardBeta { .. } => {
                let x_shape = &graph.node(node.inputs[0]).shape;
                let h = x_shape.dim(x_shape.rank() - 1).unwrap_static();
                let rows = x_shape.num_elements().unwrap() / h;
                max_bytes = max_bytes.max(rows * std::mem::size_of::<f32>());
            }
            Op::LayerNormBackwardGamma { .. } => {
                let x_shape = &graph.node(node.inputs[0]).shape;
                let h = x_shape.dim(x_shape.rank() - 1).unwrap_static();
                let rows = x_shape.num_elements().unwrap() / h;
                // mean + inv_std per row
                max_bytes = max_bytes.max(rows * 2 * std::mem::size_of::<f32>());
            }
            _ => {}
        }
    }
    max_bytes
}

/// Scratch bytes for native multi-layer GRU / Elman RNN / LSTM: a ping-pong pair
/// of `[batch, seq, dirs·hidden]` f32 buffers holding intermediate layer outputs
/// (the LSTM cell state stays in registers, so no extra scratch). Only
/// `num_layers > 1` needs it (single-layer writes straight to `dst`; both
/// directions own disjoint output slices). Shared/reused across ops (sequential
/// execution) → the max over nodes.
pub(crate) fn rnn_gru_scratch_bytes(graph: &Graph) -> usize {
    let mut max_bytes = 0usize;
    for node in graph.nodes() {
        let (hidden, num_layers, bidirectional) = match &node.op {
            Op::Gru {
                hidden_size,
                num_layers,
                bidirectional,
                ..
            }
            | Op::Rnn {
                hidden_size,
                num_layers,
                bidirectional,
                ..
            }
            | Op::Lstm {
                hidden_size,
                num_layers,
                bidirectional,
                ..
            } => (*hidden_size, *num_layers, *bidirectional),
            _ => continue,
        };
        if num_layers <= 1 {
            continue;
        }
        let x_shape = &graph.node(node.inputs[0]).shape;
        let batch = x_shape.dim(0).unwrap_static();
        let seq = x_shape.dim(1).unwrap_static();
        let dirs = if bidirectional { 2 } else { 1 };
        let layer_elems = batch * seq * dirs * hidden;
        max_bytes = max_bytes.max(2 * layer_elems * std::mem::size_of::<f32>());
    }
    max_bytes
}

pub(crate) fn encode_layer_norm_bwd_input(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    gamma: usize,
    dy: usize,
    dx: usize,
    rows: u32,
    h: u32,
    eps: f32,
) {
    enc.set_compute_pipeline_state(&k.layer_norm_bwd);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), gamma as u64);
    enc.set_buffer(2, Some(buffer), dy as u64);
    enc.set_buffer(3, Some(buffer), dx as u64);
    enc.set_bytes(4, 4, &h as *const u32 as *const _);
    enc.set_bytes(5, 4, &eps as *const f32 as *const _);
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= h as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    enc.dispatch_thread_groups(
        metal::MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w.max(1),
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_layer_norm_bwd_gamma(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    dy: usize,
    dgamma: usize,
    rows: u32,
    h: u32,
    eps: f32,
    stats_scratch: usize,
) {
    let use_parallel = stats_scratch != 0 && rows > 1;
    if !use_parallel {
        enc.set_compute_pipeline_state(&k.layer_norm_bwd_gamma);
        enc.set_buffer(0, Some(buffer), x as u64);
        enc.set_buffer(1, Some(buffer), dy as u64);
        enc.set_buffer(2, Some(buffer), dgamma as u64);
        enc.set_bytes(3, 4, &rows as *const u32 as *const _);
        enc.set_bytes(4, 4, &h as *const u32 as *const _);
        enc.set_bytes(5, 4, &eps as *const f32 as *const _);
        enc.dispatch_threads(
            metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        return;
    }

    enc.set_compute_pipeline_state(&k.layer_norm_bwd_stats_f32);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), stats_scratch as u64);
    enc.set_bytes(2, 4, &h as *const u32 as *const _);
    enc.set_bytes(3, 4, &eps as *const f32 as *const _);
    enc.dispatch_threads(
        metal::MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 256.min(rows as u64).max(1),
            height: 1,
            depth: 1,
        },
    );

    // SIMD variant: 32-wide threadgroup per column parallelizes the row
    // reduction (the scalar kernel serializes it on one thread/column). More
    // threads/tg (two-level reduction) was measured neutral — the reduction is
    // memory-bound and small, so GPU fill isn't the limiter here.
    let use_simd = rlx_ir::env::flag("RLX_METAL_LN_GAMMA_SIMD");
    if use_simd {
        enc.set_compute_pipeline_state(&k.layer_norm_bwd_gamma_reduce_simd);
        enc.set_buffer(0, Some(buffer), x as u64);
        enc.set_buffer(1, Some(buffer), dy as u64);
        enc.set_buffer(2, Some(buffer), stats_scratch as u64);
        enc.set_buffer(3, Some(buffer), dgamma as u64);
        enc.set_bytes(4, 4, &rows as *const u32 as *const _);
        enc.set_bytes(5, 4, &h as *const u32 as *const _);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: h as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        return;
    }
    enc.set_compute_pipeline_state(&k.layer_norm_bwd_gamma_reduce_f32);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), dy as u64);
    enc.set_buffer(2, Some(buffer), stats_scratch as u64);
    enc.set_buffer(3, Some(buffer), dgamma as u64);
    enc.set_bytes(4, 4, &rows as *const u32 as *const _);
    enc.set_bytes(5, 4, &h as *const u32 as *const _);
    let tg_w = 256u64.min(h as u64).max(1);
    enc.dispatch_threads(
        metal::MTLSize {
            width: h as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_group_norm_bwd_input(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    gamma: usize,
    dy: usize,
    dx: usize,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    num_groups: u32,
    eps: f32,
) {
    let nchw: [u32; 4] = [n, c, h, w];
    enc.set_compute_pipeline_state(&k.group_norm_bwd_input);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), gamma as u64);
    enc.set_buffer(2, Some(buffer), dy as u64);
    enc.set_buffer(3, Some(buffer), dx as u64);
    enc.set_bytes(4, 16, nchw.as_ptr() as *const _);
    enc.set_bytes(5, 4, &num_groups as *const u32 as *const _);
    enc.set_bytes(6, 4, &eps as *const f32 as *const _);
    let groups = (n * num_groups) as u64;
    enc.dispatch_thread_groups(
        metal::MTLSize {
            width: groups.max(1),
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_group_norm_bwd_gamma(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    dy: usize,
    dgamma: usize,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    num_groups: u32,
    eps: f32,
) {
    let nchw: [u32; 4] = [n, c, h, w];
    enc.set_compute_pipeline_state(&k.group_norm_bwd_gamma);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), dy as u64);
    enc.set_buffer(2, Some(buffer), dgamma as u64);
    enc.set_bytes(3, 16, nchw.as_ptr() as *const _);
    enc.set_bytes(4, 4, &num_groups as *const u32 as *const _);
    enc.set_bytes(5, 4, &eps as *const f32 as *const _);
    enc.dispatch_threads(
        metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_group_norm_bwd_beta(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    dy: usize,
    dbeta: usize,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
) {
    let nchw: [u32; 4] = [n, c, h, w];
    enc.set_compute_pipeline_state(&k.group_norm_bwd_beta);
    enc.set_buffer(0, Some(buffer), dy as u64);
    enc.set_buffer(1, Some(buffer), dbeta as u64);
    enc.set_bytes(2, 16, nchw.as_ptr() as *const _);
    enc.dispatch_threads(
        metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_rope_bwd(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    dy: usize,
    cos: usize,
    sin: usize,
    dx: usize,
    batch: u32,
    seq: u32,
    hidden: u32,
    head_dim: u32,
    n_rot: u32,
    cos_len: u32,
) {
    enc.set_compute_pipeline_state(&k.rope_bwd);
    enc.set_buffer(0, Some(buffer), dy as u64);
    enc.set_buffer(1, Some(buffer), cos as u64);
    enc.set_buffer(2, Some(buffer), sin as u64);
    enc.set_buffer(3, Some(buffer), dx as u64);
    enc.set_bytes(4, 4, &batch as *const u32 as *const _);
    enc.set_bytes(5, 4, &seq as *const u32 as *const _);
    enc.set_bytes(6, 4, &hidden as *const u32 as *const _);
    enc.set_bytes(7, 4, &head_dim as *const u32 as *const _);
    enc.set_bytes(8, 4, &n_rot as *const u32 as *const _);
    enc.set_bytes(9, 4, &cos_len as *const u32 as *const _);
    let nh = hidden / head_dim.max(1);
    enc.dispatch_threads(
        metal::MTLSize {
            width: head_dim as u64,
            height: nh as u64,
            depth: (batch * seq) as u64,
        },
        metal::MTLSize {
            width: head_dim.min(16) as u64,
            height: nh.min(8) as u64,
            depth: 1,
        },
    );
}

pub(crate) fn encode_cumsum(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    rows: u32,
    cols: u32,
    exclusive: bool,
) {
    enc.set_compute_pipeline_state(&k.cumsum_fwd);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 4, &cols as *const u32 as *const _);
    let ex: u32 = if exclusive { 1 } else { 0 };
    enc.set_bytes(3, 4, &ex as *const u32 as *const _);
    enc.dispatch_threads(
        metal::MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_cum_scan(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    rows: u32,
    cols: u32,
    exclusive: bool,
    is_max: bool,
) {
    enc.set_compute_pipeline_state(&k.cum_scan);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 4, &cols as *const u32 as *const _);
    let ex: u32 = if exclusive { 1 } else { 0 };
    enc.set_bytes(3, 4, &ex as *const u32 as *const _);
    let mx: u32 = if is_max { 1 } else { 0 };
    enc.set_bytes(4, 4, &mx as *const u32 as *const _);
    enc.dispatch_threads(
        metal::MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_cumsum_bwd(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    dy: usize,
    dx: usize,
    rows: u32,
    cols: u32,
    exclusive: bool,
) {
    enc.set_compute_pipeline_state(&k.cumsum_bwd);
    enc.set_buffer(0, Some(buffer), dy as u64);
    enc.set_buffer(1, Some(buffer), dx as u64);
    enc.set_bytes(2, 4, &cols as *const u32 as *const _);
    let ex: u32 = if exclusive { 1 } else { 0 };
    enc.set_bytes(3, 4, &ex as *const u32 as *const _);
    enc.dispatch_threads(
        metal::MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_gather_bwd(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    dy: usize,
    indices: usize,
    dst: usize,
    outer: u32,
    axis_dim: u32,
    num_idx: u32,
    trailing: u32,
) {
    let n = outer * axis_dim * trailing;
    if n > 0 {
        enc.set_compute_pipeline_state(&k.gather_bwd_zero);
        enc.set_buffer(0, Some(buffer), dst as u64);
        enc.set_bytes(1, 4, &n as *const u32 as *const _);
        enc.dispatch_threads(
            metal::MTLSize {
                width: n as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }
    enc.set_compute_pipeline_state(&k.gather_bwd_acc);
    enc.set_buffer(0, Some(buffer), dy as u64);
    enc.set_buffer(1, Some(buffer), indices as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &outer as *const u32 as *const _);
    enc.set_bytes(4, 4, &axis_dim as *const u32 as *const _);
    enc.set_bytes(5, 4, &num_idx as *const u32 as *const _);
    enc.set_bytes(6, 4, &trailing as *const u32 as *const _);
    enc.dispatch_threads(
        metal::MTLSize {
            width: outer as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
}

/// Bytes of arena-tail scratch for the m>8 SynthMatMul recon→MPS prefill path:
/// one shared f32 [n,k] weight slab, sized to the largest such node (small-m
/// nodes use split-K and need no scratch).
pub(crate) fn synth_matmul_scratch_bytes(graph: &Graph) -> usize {
    let mut max = 0usize;
    for node in graph.nodes() {
        if let Op::SynthMatMul { .. } = &node.op {
            let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
            let total = node.shape.num_elements().unwrap();
            let m = total / n.max(1);
            if m <= 8 {
                continue;
            }
            let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
            let k = x_total / m.max(1);
            // f32 recon→MPS needs the [n,k] f32 weight scratch. The opt-in f16 path
            // (RLX_METAL_SYNTH_RECON_F16) instead needs three f16 buffers: W[k,n],
            // x[m,k], dst[m,n] (256-aligned for MPS). Reserve the larger so either
            // path fits without knowing the runtime flag at compile time.
            let a256 = |b: usize| (b + 255) & !255;
            let f32_need = k * n * 4;
            let f16_need = a256(k * n * 2) + a256(m * k * 2) + a256(m * n * 2);
            max = max.max(f32_need.max(f16_need));
        }
    }
    max
}

/// f16 reconstruct: writes the dense weight W[k,n] as `half` into `w_scratch` — the
/// weight half of the RLX_METAL_SYNTH_RECON_F16 prefill path (pair with a cast of x
/// to f16 and `encode_mps_hgemm`, then cast the f16 result back to f32).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_synth_reconstruct_h(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    indices: usize,
    codebook: usize,
    w_scratch: usize,
    k_dim: u32,
    n_dim: u32,
    entry_dim: u32,
) {
    enc.set_compute_pipeline_state(&k.synth_reconstruct_h);
    enc.set_buffer(0, Some(buffer), 0);
    let i_u = indices as u64;
    enc.set_bytes(1, 8, &i_u as *const u64 as *const _);
    let c_u = codebook as u64;
    enc.set_bytes(2, 8, &c_u as *const u64 as *const _);
    let w_u = w_scratch as u64;
    enc.set_bytes(3, 8, &w_u as *const u64 as *const _);
    enc.set_bytes(4, 4, &k_dim as *const u32 as *const _);
    enc.set_bytes(5, 4, &n_dim as *const u32 as *const _);
    enc.set_bytes(6, 4, &entry_dim as *const u32 as *const _);
    let nb = k_dim / entry_dim.max(1);
    enc.dispatch_threads(
        metal::MTLSize {
            width: nb as u64,
            height: n_dim as u64,
            depth: 1,
        },
        metal::MTLSize {
            width: 32u64.min(nb as u64).max(1),
            height: 8u64.min(n_dim as u64).max(1),
            depth: 1,
        },
    );
}

/// Encode a `cast_f32_to_f16` (or reverse) over `len` elements from `src` to `dst`
/// (both arena byte offsets). Used to move x into f16 and the result back to f32 for
/// the RLX_METAL_SYNTH_RECON_F16 path.
pub(crate) fn encode_arena_cast(
    enc: &metal::ComputeCommandEncoderRef,
    pipe: &metal::ComputePipelineState,
    buffer: &metal::Buffer,
    src_off: usize,
    dst_off: usize,
    len: u32,
) {
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(buffer), src_off as u64);
    enc.set_buffer(1, Some(buffer), dst_off as u64);
    enc.set_bytes(2, 4, &len as *const u32 as *const _);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 256u64.min(len as u64).max(1),
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn dequant_gguf_scratch_bytes(graph: &Graph) -> usize {
    let mut max = 0usize;
    for node in graph.nodes() {
        if let Op::DequantMatMul { scheme } = &node.op
            && scheme.is_gguf()
        {
            // Q1_0 runs through the fused q1_0 mv/mm kernels (read packed weights
            // directly), so it never uses the dequant→f32 scratch. Skipping it
            // here avoids reserving a huge f32 slab — e.g. the Bonsai-27B tied LM
            // head [248320,5120] would demand ~5 GiB of dead scratch per graph.
            // (Mirrors wgpu's `gemv_supports_scheme` scratch skip.) The off-switch
            // RLX_METAL_Q1_0_FUSED_DISABLE reverts to the scratch path.
            let direct_fused = match scheme {
                rlx_ir::QuantScheme::GgufQ1_0 => !rlx_ir::env::flag("RLX_METAL_Q1_0_FUSED_DISABLE"),
                rlx_ir::QuantScheme::GgufQ2_0 => !rlx_ir::env::flag("RLX_METAL_Q2_0_FUSED_DISABLE"),
                _ => false,
            };
            if direct_fused {
                continue;
            }
            let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
            let total = node.shape.num_elements().unwrap();
            let m = total / n.max(1);
            let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
            let k = x_total / m.max(1);
            max = max.max(k * n * std::mem::size_of::<f32>());
        }
        if let Op::DequantGroupedMatMul { .. } = &node.op {
            let in_shape = &graph.node(node.inputs[0]).shape;
            let m = in_shape.dim(in_shape.rank() - 2).unwrap_static();
            let k = in_shape.dim(in_shape.rank() - 1).unwrap_static();
            let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
            max = max.max(k * n * 4 + m * k * 4 + m * n * 4);
        }
    }
    max
}

/// Maps [`QuantScheme`] to the shared GPU `dequant_gguf` MSL kernel scheme id.
///
/// Delegates to [`rlx_gpu_host::gguf_scheme_id`] (same table as CUDA/ROCm/wgpu).
pub(crate) fn gguf_scheme_id(scheme: rlx_ir::quant::QuantScheme) -> u32 {
    rlx_gpu_host::gguf_scheme_id(scheme)
}

/// Returns `true` when this scheme has a native on-device dequant kernel
/// in the `dequant_gguf` MSL shader, `false` when callers should route
/// through the CPU dequant path (`rlx_gguf::dequant_*`) instead.
///
/// Fused GEMV (`q4k_mv_f32`, `q4_0_mv_f32`, `q8_0_mv_f32`) is separate — see
/// [docs/gguf-backend-paths.md](../../../docs/gguf-backend-paths.md).
pub fn has_metal_dequant_kernel(scheme: rlx_ir::quant::QuantScheme) -> bool {
    // Every GGUF scheme with a shared `gpu_dequant_scheme_id` has a Metal
    // branch in `dequant_gguf.msl` — keep this in sync via the IR table.
    scheme.gpu_dequant_scheme_id().is_some()
}

/// Simdgroup-cooperative Q4_K_M GEMV. 32 threads share x reads and
/// produce 8 output columns each via `simd_sum`. Constraint:
/// `n_dim % 8 == 0` (caller enforces). Adapted from llama.cpp's
/// `kernel_mul_mv_q4_K_f32_impl`.
pub(crate) fn encode_q4k_mv_f32_sg(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.q4k_mv_f32_sg);
    enc.set_buffer(0, Some(buffer), 0);
    // u64 byte offsets (task #50) — 12B Q4 activations sit past 4 GB.
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    // 4 simdgroups per threadgroup; each simdgroup handles Q4K_NR0 output rows.
    // MUST match `Q4K_NR0` in dequant_gguf.msl (2 → more threadgroups → the m=1
    // decode GEMV fills the GPU instead of starving it at 8% of peak).
    const NSG: u64 = 4;
    const Q4K_NR0: u64 = 2;
    let n_output_groups = (n_dim.div_ceil(Q4K_NR0 as usize)) as u64;
    let n_threadgroups = n_output_groups.div_ceil(NSG);
    let grid = metal::MTLSize {
        width: n_threadgroups * NSG * 32,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: NSG * 32,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// Generate a simdgroup-cooperative decode-GEMV encoder. Every `_sg` GEMV
/// kernel takes the identical 6 buffers (arena, x_off, w_off, dst_off, k_dim,
/// n_dim) and dispatches `NSG` simdgroups per threadgroup, each producing `NR0`
/// output rows via `simd_sum`; only the pipeline field + `(NSG, NR0)` differ.
/// `(NSG, NR0)` MUST match the kernel's constants in `dequant_gguf.msl`. One
/// invocation per scheme (Q6_K/Q3_K reduce one row per simdgroup → NR0 = 1).
macro_rules! sg_mv_encoder {
    ($fn_name:ident, $pipe:ident, $nsg:expr, $nr0:expr) => {
        pub(crate) fn $fn_name(
            enc: &metal::ComputeCommandEncoderRef,
            k: &crate::kernels::Kernels,
            buffer: &metal::Buffer,
            x: usize,
            w_q: usize,
            dst: usize,
            k_dim: usize,
            n_dim: usize,
        ) {
            enc.set_compute_pipeline_state(&k.$pipe);
            enc.set_buffer(0, Some(buffer), 0);
            let x_u = x as u64;
            enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
            let w_u = w_q as u64;
            enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
            let d_u = dst as u64;
            enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
            let k_u = k_dim as u32;
            enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
            let n_u = n_dim as u32;
            enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
            const NSG: u64 = $nsg;
            const NR0: u64 = $nr0;
            let n_output_groups = (n_dim.div_ceil(NR0 as usize)) as u64;
            let n_threadgroups = n_output_groups.div_ceil(NSG);
            let grid = metal::MTLSize {
                width: n_threadgroups * NSG * 32,
                height: 1,
                depth: 1,
            };
            let tg = metal::MTLSize {
                width: NSG * 32,
                height: 1,
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
        }
    };
}

// One simdgroup-cooperative GEMV encoder per quant scheme. `_sg` kernels adapted
// from llama.cpp's `mul_mv_q*_f32` (32-thread `simd_sum` reduction). NR0 = rows
// per simdgroup; Q6_K/Q3_K reduce a whole 256-wide super-block row per simdgroup
// so they emit one row each (NR0 = 1).
sg_mv_encoder!(encode_q8_0_mv_f32_sg, q8_0_mv_f32_sg, 2, 4);
sg_mv_encoder!(encode_q4_0_mv_f32_sg, q4_0_mv_f32_sg, 2, 4);
sg_mv_encoder!(encode_q4_1_mv_f32_sg, q4_1_mv_f32_sg, 2, 4);
sg_mv_encoder!(encode_q6k_mv_f32_sg, q6k_mv_f32_sg, 4, 1);
sg_mv_encoder!(encode_q3k_mv_f32_sg, q3k_mv_f32_sg, 4, 2);

/// Fused Q4_K / Q6_K GEMM (`m > 1`, prefill): `C[m,n] = A[m,k] @ dequant(w)^T`
/// straight from the packed weight — no f32 scratch, no MPS sgemm. Grid is
/// `(n columns) × ceil(m / TM)` row-tiles; threadgroup = up to 64 columns.
/// Caller must guarantee `k_dim % 256 == 0`. Used for `m > 1` GgufQ4K/Q6K.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qk_mm_f32(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let m_u = m_dim as u32;
    enc.set_bytes(4, 4, &m_u as *const u32 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(5, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(6, 4, &n_u as *const u32 as *const _);
    // TM must match Q4K_MM_TM / Q6K_MM_TM in dequant_gguf.msl.
    const TM: u64 = 8;
    let row_tiles = (m_dim as u64).div_ceil(TM);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: row_tiles,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: (n_dim as u64).min(64),
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// Fused Q4_K_M GEMV: `dst[n] = sum_k x[k] * dequant(w[n,k])` in one
/// pass, skipping the f32 dequant scratch the dequant + MPS sgemm path
/// would write. Caller must guarantee `k_dim % 256 == 0`. Decode-only
/// (`m == 1`) — m > 1 still falls through to the legacy GPU path.
/// Generate a fused single-pass decode-GEMV encoder for one K-quant / block
/// scheme. Every such kernel takes the identical 6 buffers (arena, x_off,
/// w_off, dst_off, k_dim, n_dim) and dispatches one thread per output row —
/// only the `Kernels` pipeline field differs. One invocation per quant/precision
/// (e.g. `fused_mv_encoder!(encode_q3k_mv_f32, q3k_mv_f32)`); add a matching MSL
/// kernel + `Kernels` field and the whole encoder is generated.
macro_rules! fused_mv_encoder {
    ($fn_name:ident, $pipe:ident) => {
        pub(crate) fn $fn_name(
            enc: &metal::ComputeCommandEncoderRef,
            k: &crate::kernels::Kernels,
            buffer: &metal::Buffer,
            x: usize,
            w_q: usize,
            dst: usize,
            k_dim: usize,
            n_dim: usize,
        ) {
            enc.set_compute_pipeline_state(&k.$pipe);
            enc.set_buffer(0, Some(buffer), 0);
            let x_u = x as u64;
            enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
            let w_u = w_q as u64;
            enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
            let d_u = dst as u64;
            enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
            let k_u = k_dim as u32;
            enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
            let n_u = n_dim as u32;
            enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
            let grid = metal::MTLSize {
                width: n_dim as u64,
                height: 1,
                depth: 1,
            };
            let tg = metal::MTLSize {
                width: 256.min(n_dim) as u64,
                height: 1,
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
        }
    };
}

// One line per quant/precision — MSL kernel + `Kernels` field is all else needed.
fused_mv_encoder!(encode_q3k_mv_f32, q3k_mv_f32); // Q3_K_S trunk bulk weights
fused_mv_encoder!(encode_q6k_mv_f32, q6k_mv_f32); // Q6_K LM head

pub(crate) fn encode_q4k_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.q4k_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// Fused Q1_0 GEMV: `dst[n] = sum_k x[k] * dequant(w[n,k])` reading the packed
/// 1-bit weight directly. Decode-only (`m == 1`); caller guarantees
/// `k_dim % 128 == 0`. Skips the dequant-scratch + MPS sgemm path (whose shared
/// scratch races and zeros large-n Q1_0 outputs in the full Bonsai-27B graph).
pub(crate) fn encode_q1_0_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    scheme: rlx_ir::QuantScheme,
    buffer: &metal::Buffer,
    w_buffer: &metal::Buffer,
    x: usize,
    w_raw: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    let pipeline = match scheme {
        rlx_ir::QuantScheme::GgufQ1_0 => &k.q1_0_mv_f32,
        rlx_ir::QuantScheme::GgufQ2_0 => &k.q2_0_mv_f32,
        _ => unreachable!(),
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_raw as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    // Weight may live in the external weight buffer (large params) or the arena
    // (small params) — the caller resolves the tag and passes the raw offset.
    enc.set_buffer(7, Some(w_buffer), 0);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// Simdgroup-cooperative Q1_0 GEMV (llama.cpp `kernel_mul_mv_q1_0_f32`).
/// 32 threads share x reads and produce 8 outputs via `simd_sum`.
/// Constraint: `n_dim % 8 == 0` (caller enforces).
pub(crate) fn encode_q1_0_mv_f32_sg_flags(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    scheme: rlx_ir::QuantScheme,
    buffer: &metal::Buffer,
    w_buffer: &metal::Buffer,
    x: usize,
    w_raw: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
    x_f16: bool,
    dst_f16: bool,
) {
    let pipeline = match scheme {
        rlx_ir::QuantScheme::GgufQ1_0 => &k.q1_0_mv_f32_sg,
        rlx_ir::QuantScheme::GgufQ2_0 => &k.q2_0_mv_f32_sg,
        _ => unreachable!(),
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_raw as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    let flags: u32 = u32::from(x_f16) | (u32::from(dst_f16) << 1);
    enc.set_bytes(6, 4, &flags as *const u32 as *const _);
    enc.set_buffer(7, Some(w_buffer), 0);
    // NSG=2 simdgroups share a threadgroup x tile (see dequant_gguf.msl).
    const NSG: u64 = 2;
    let n_output_groups = (n_dim.div_ceil(8)) as u64;
    let n_threadgroups = n_output_groups.div_ceil(NSG);
    let grid = metal::MTLSize {
        width: n_threadgroups * NSG * 32,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: NSG * 32,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// Shared-x dual Q1_0 GEMV (simdgroup): one `x` feed, two weight matrices.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_q1_0_dual_mv_f32_sg_flags(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    scheme: rlx_ir::QuantScheme,
    buffer: &metal::Buffer,
    w_buffer: &metal::Buffer,
    x: usize,
    w0_raw: usize,
    w1_raw: usize,
    dst0: usize,
    dst1: usize,
    k_dim: usize,
    n0: usize,
    n1: usize,
    x_f16: bool,
    dst_f16: bool,
) {
    let pipeline = match scheme {
        rlx_ir::QuantScheme::GgufQ1_0 => &k.q1_0_dual_mv_f32_sg,
        rlx_ir::QuantScheme::GgufQ2_0 => &k.q2_0_dual_mv_f32_sg,
        _ => unreachable!(),
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w0_u = w0_raw as u64;
    enc.set_bytes(2, 8, &w0_u as *const u64 as *const _);
    let w1_u = w1_raw as u64;
    enc.set_bytes(3, 8, &w1_u as *const u64 as *const _);
    let d0_u = dst0 as u64;
    enc.set_bytes(4, 8, &d0_u as *const u64 as *const _);
    let d1_u = dst1 as u64;
    enc.set_bytes(5, 8, &d1_u as *const u64 as *const _);
    let dims: [u32; 4] = [
        k_dim as u32,
        n0 as u32,
        n1 as u32,
        u32::from(x_f16) | (u32::from(dst_f16) << 1),
    ];
    enc.set_bytes(6, 16, dims.as_ptr() as *const _);
    enc.set_buffer(7, Some(w_buffer), 0);
    const NSG: u64 = 2;
    let n_max = n0.max(n1);
    let n_output_groups = (n_max.div_ceil(8)) as u64;
    let n_threadgroups = n_output_groups.div_ceil(NSG);
    let grid = metal::MTLSize {
        width: n_threadgroups * NSG * 32,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: NSG * 32,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// Fused Q1_0 GEMM (`m > 1`, prefill): reads the packed 1-bit weight directly,
/// accumulating a TM=8 row tile per thread — replaces the dequant-to-f32 scratch
/// + MPS sgemm path for Q1_0. Caller guarantees `k_dim % 128 == 0`.
pub(crate) fn encode_q1_0_mm_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    scheme: rlx_ir::QuantScheme,
    buffer: &metal::Buffer,
    w_buffer: &metal::Buffer,
    x: usize,
    w_raw: usize,
    dst: usize,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) {
    let pipeline = match scheme {
        rlx_ir::QuantScheme::GgufQ1_0 => &k.q1_0_mm_f32,
        rlx_ir::QuantScheme::GgufQ2_0 => &k.q2_0_mm_f32,
        _ => unreachable!(),
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_raw as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let m_u = m_dim as u32;
    enc.set_bytes(4, 4, &m_u as *const u32 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(5, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(6, 4, &n_u as *const u32 as *const _);
    enc.set_buffer(7, Some(w_buffer), 0);
    const TM: u64 = 8;
    let row_tiles = (m_dim as u64).div_ceil(TM);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: row_tiles,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: (n_dim as u64).min(64),
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// Fused decode MLP gate+up packed GEMV dispatch (`m == 1`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_fused_mlp_gate_up_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    buffer: &metal::Buffer,
    x: usize,
    gate_w: usize,
    up_w: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let g_u = gate_w as u64;
    enc.set_bytes(2, 8, &g_u as *const u64 as *const _);
    let u_u = up_w as u64;
    enc.set_bytes(3, 8, &u_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(4, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(5, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(6, 4, &n_u as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// Fused decode MLP gate+up packed GEMV with SwiGLU epilogue (`m == 1`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_fused_mlp_gate_up_swiglu(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    scheme: rlx_ir::quant::QuantScheme,
    x: usize,
    gate_w: usize,
    up_w: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    use rlx_ir::quant::QuantScheme;
    let pipeline = match scheme {
        QuantScheme::GgufQ4K => &k.q4k_swiglu_mv_f32,
        QuantScheme::GgufQ5_0 => &k.q5_0_swiglu_mv_f32,
        other => panic!("encode_fused_mlp_gate_up_swiglu: unsupported {other:?}"),
    };
    encode_fused_mlp_gate_up_mv_f32(enc, pipeline, buffer, x, gate_w, up_w, dst, k_dim, n_dim);
}

/// Fused Q1_0 gate+up+SwiGLU (`m == 1`). Weights in `w_buffer` (arena or external).
/// Uses simdgroup cooperative GEMV when `n_dim % 8 == 0` (Bonsai dims).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_q1_0_swiglu_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    scheme: rlx_ir::QuantScheme,
    buffer: &metal::Buffer,
    w_buffer: &metal::Buffer,
    x: usize,
    gate_raw: usize,
    up_raw: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
    x_f16: bool,
    dst_f16: bool,
) {
    let use_sg = n_dim.is_multiple_of(8)
        && match scheme {
            rlx_ir::QuantScheme::GgufQ1_0 => !rlx_ir::env::flag("RLX_METAL_Q1_0_SG_DISABLE"),
            rlx_ir::QuantScheme::GgufQ2_0 => !rlx_ir::env::flag("RLX_METAL_Q2_0_SG_DISABLE"),
            _ => false,
        };
    let pipeline = match (scheme, use_sg) {
        (rlx_ir::QuantScheme::GgufQ1_0, true) => &k.q1_0_swiglu_mv_f32_sg,
        (rlx_ir::QuantScheme::GgufQ1_0, false) => &k.q1_0_swiglu_mv_f32,
        (rlx_ir::QuantScheme::GgufQ2_0, true) => &k.q2_0_swiglu_mv_f32_sg,
        (rlx_ir::QuantScheme::GgufQ2_0, false) => &k.q2_0_swiglu_mv_f32,
        _ => unreachable!(),
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let g_u = gate_raw as u64;
    enc.set_bytes(2, 8, &g_u as *const u64 as *const _);
    let u_u = up_raw as u64;
    enc.set_bytes(3, 8, &u_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(4, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(5, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(6, 4, &n_u as *const u32 as *const _);
    enc.set_buffer(7, Some(w_buffer), 0);
    if use_sg {
        let flags: u32 = u32::from(x_f16) | (u32::from(dst_f16) << 1);
        enc.set_bytes(8, 4, &flags as *const u32 as *const _);
        const NSG: u64 = 2;
        let n_output_groups = (n_dim.div_ceil(8)) as u64;
        let n_threadgroups = n_output_groups.div_ceil(NSG);
        let grid = metal::MTLSize {
            width: n_threadgroups * NSG * 32,
            height: 1,
            depth: 1,
        };
        let tg = metal::MTLSize {
            width: NSG * 32,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
    } else {
        let grid = metal::MTLSize {
            width: n_dim as u64,
            height: 1,
            depth: 1,
        };
        let tg = metal::MTLSize {
            width: 256.min(n_dim) as u64,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
    }
}

/// Fused Q1_0 down GEMV + residual add (`m == 1`).
/// Uses simdgroup cooperative GEMV when `n_dim % 8 == 0`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_q1_0_mv_residual_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    scheme: rlx_ir::QuantScheme,
    buffer: &metal::Buffer,
    w_buffer: &metal::Buffer,
    x: usize,
    w_raw: usize,
    res: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
    x_f16: bool,
    dst_f16: bool,
    res_f16: bool,
) {
    let use_sg = n_dim.is_multiple_of(8)
        && match scheme {
            rlx_ir::QuantScheme::GgufQ1_0 => !rlx_ir::env::flag("RLX_METAL_Q1_0_SG_DISABLE"),
            rlx_ir::QuantScheme::GgufQ2_0 => !rlx_ir::env::flag("RLX_METAL_Q2_0_SG_DISABLE"),
            _ => false,
        };
    let pipeline = match (scheme, use_sg) {
        (rlx_ir::QuantScheme::GgufQ1_0, true) => &k.q1_0_mv_residual_f32_sg,
        (rlx_ir::QuantScheme::GgufQ1_0, false) => &k.q1_0_mv_residual_f32,
        (rlx_ir::QuantScheme::GgufQ2_0, true) => &k.q2_0_mv_residual_f32_sg,
        (rlx_ir::QuantScheme::GgufQ2_0, false) => &k.q2_0_mv_residual_f32,
        _ => unreachable!(),
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_raw as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let r_u = res as u64;
    enc.set_bytes(4, 8, &r_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(5, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(6, 4, &n_u as *const u32 as *const _);
    enc.set_buffer(7, Some(w_buffer), 0);
    if use_sg {
        let flags: u32 = u32::from(x_f16) | (u32::from(dst_f16) << 1) | (u32::from(res_f16) << 2);
        enc.set_bytes(8, 4, &flags as *const u32 as *const _);
        const NSG: u64 = 2;
        let n_output_groups = (n_dim.div_ceil(8)) as u64;
        let n_threadgroups = n_output_groups.div_ceil(NSG);
        let grid = metal::MTLSize {
            width: n_threadgroups * NSG * 32,
            height: 1,
            depth: 1,
        };
        let tg = metal::MTLSize {
            width: NSG * 32,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
    } else {
        let grid = metal::MTLSize {
            width: n_dim as u64,
            height: 1,
            depth: 1,
        };
        let tg = metal::MTLSize {
            width: 256.min(n_dim) as u64,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
    }
}

/// Fused decode MLP gate+up packed GEMV with GELU-approx epilogue (`m == 1`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_fused_mlp_gate_up_gelu(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    scheme: rlx_ir::quant::QuantScheme,
    x: usize,
    gate_w: usize,
    up_w: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    use rlx_ir::quant::QuantScheme;
    let pipeline = match scheme {
        QuantScheme::GgufQ4K => &k.q4k_gelu_mv_f32,
        QuantScheme::GgufQ5_0 => &k.q5_0_gelu_mv_f32,
        other => panic!("encode_fused_mlp_gate_up_gelu: unsupported {other:?}"),
    };
    encode_fused_mlp_gate_up_mv_f32(enc, pipeline, buffer, x, gate_w, up_w, dst, k_dim, n_dim);
}

/// Fused decode MLP down-projection GEMV + residual add (`m == 1`).
/// `dst[j] = res[j] + down(x)[j]`. `pipeline` selects Q4_K / Q5_0 / Q6_K.
/// one thread per output column. Caller guarantees `k_dim % 256 == 0`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_q4k_mv_residual_f32(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    res: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let r_u = res as u64;
    enc.set_bytes(4, 8, &r_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(5, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(6, 4, &n_u as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_q4_0_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.q4_0_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_q4_1_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.q4_1_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_q8_0_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.q8_0_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_iq4_nl_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.iq4_nl_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_iq2_xxs_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.iq2_xxs_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    enc.set_buffer(6, Some(k.iq_grid_buffer()), 0);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// Codebook weight-synthesis matmul. One thread per output column `gid ∈ [0,n)`;
/// the kernel loops over the `m` rows internally. `x`, `codebook`, `dst` are f32
/// arena tensors; `indices` is a packed U8 arena tensor (1 byte/elem).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_synth_matmul(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    indices: usize,
    codebook: usize,
    dst: usize,
    m: u32,
    k_dim: u32,
    n_dim: u32,
    entry_dim: u32,
    num_entries: u32,
    half: bool,
) {
    let split_k = m <= 8; // decode / small-M → split-K; else one-thread-per-output
    enc.set_compute_pipeline_state(match (split_k, half) {
        (true, false) => &k.synth_matmul_codebook,
        (true, true) => &k.synth_matmul_codebook_h,
        (false, false) => &k.synth_matmul_codebook_mm,
        (false, true) => &k.synth_matmul_codebook_mm_h,
    });
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let i_u = indices as u64;
    enc.set_bytes(2, 8, &i_u as *const u64 as *const _);
    let c_u = codebook as u64;
    enc.set_bytes(3, 8, &c_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(4, 8, &d_u as *const u64 as *const _);
    enc.set_bytes(5, 4, &k_dim as *const u32 as *const _);
    enc.set_bytes(6, 4, &n_dim as *const u32 as *const _);
    enc.set_bytes(7, 4, &entry_dim as *const u32 as *const _);
    enc.set_bytes(8, 4, &num_entries as *const u32 as *const _);
    enc.set_bytes(9, 4, &m as *const u32 as *const _);
    // Small M / decode: split-K — grid (32 lanes × n × m); each SIMD group's 32
    // lanes cooperate over the k-blocks + simd_sum, so the GPU isn't starved at
    // M=1. Large M: one thread per output element (n·m already saturates; split-K
    // there just adds reduction overhead).
    let (grid, tg) = if split_k {
        const KSPLIT: u64 = 32;
        (
            metal::MTLSize {
                width: KSPLIT,
                height: n_dim as u64,
                depth: m as u64,
            },
            metal::MTLSize {
                width: KSPLIT,
                height: 8u64.min(n_dim as u64).max(1),
                depth: 1,
            },
        )
    } else {
        // One thread per output element (fused prefill kernel; MPS wins here but
        // this is the correct all-shapes fallback).
        let tgh = 8u64.min(m as u64).max(1);
        let tgw = (256 / tgh).min(n_dim as u64).max(1);
        (
            metal::MTLSize {
                width: n_dim as u64,
                height: m as u64,
                depth: 1,
            },
            metal::MTLSize {
                width: tgw,
                height: tgh,
                depth: 1,
            },
        )
    };
    enc.dispatch_threads(grid, tg);
}

/// Threadgroup-tiled fused codebook matmul (`simdgroup_float8x8` MMAs). A 32×32
/// output tile per threadgroup (16 simdgroups × 32 threads = 512), reconstructing
/// the weight tile on-chip from u8 indices + the L1 codebook — no dense-weight DRAM
/// materialization, no MPS launch. Targets the medium-M regime where recon→MPS's
/// fixed reconstruct→scratch→MPS cost dominates. f32 only.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_synth_matmul_tiled(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    indices: usize,
    codebook: usize,
    dst: usize,
    m: u32,
    k_dim: u32,
    n_dim: u32,
    entry_dim: u32,
    num_entries: u32,
    f16: bool,
) {
    enc.set_compute_pipeline_state(if f16 {
        &k.synth_matmul_codebook_tiled_h
    } else {
        &k.synth_matmul_codebook_tiled
    });
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let i_u = indices as u64;
    enc.set_bytes(2, 8, &i_u as *const u64 as *const _);
    let c_u = codebook as u64;
    enc.set_bytes(3, 8, &c_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(4, 8, &d_u as *const u64 as *const _);
    enc.set_bytes(5, 4, &k_dim as *const u32 as *const _);
    enc.set_bytes(6, 4, &n_dim as *const u32 as *const _);
    enc.set_bytes(7, 4, &entry_dim as *const u32 as *const _);
    enc.set_bytes(8, 4, &num_entries as *const u32 as *const _);
    enc.set_bytes(9, 4, &m as *const u32 as *const _);
    // One 64×64 output tile per threadgroup; 512 threads = 16 simdgroups (4×4),
    // each computing a 16×16 sub-tile (2×2 grid of 8×8 accumulators).
    let tg_count = metal::MTLSize {
        width: n_dim.div_ceil(64) as u64,
        height: m.div_ceil(64) as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 32,
        height: 16,
        depth: 1,
    };
    enc.dispatch_thread_groups(tg_count, tg);
}

/// Reconstruct the dense f32 weight Wᵀ[n,k] from u8 indices + f32 codebook into
/// the arena scratch at `w_scratch` — the weight-only half of the m>8 recon→MPS
/// prefill path. Pair with `encode_mps_sgemm_bt(x, w_scratch, dst, m, k, n)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_synth_reconstruct(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    indices: usize,
    codebook: usize,
    w_scratch: usize,
    k_dim: u32,
    n_dim: u32,
    entry_dim: u32,
) {
    enc.set_compute_pipeline_state(&k.synth_reconstruct);
    enc.set_buffer(0, Some(buffer), 0);
    let i_u = indices as u64;
    enc.set_bytes(1, 8, &i_u as *const u64 as *const _);
    let c_u = codebook as u64;
    enc.set_bytes(2, 8, &c_u as *const u64 as *const _);
    let w_u = w_scratch as u64;
    enc.set_bytes(3, 8, &w_u as *const u64 as *const _);
    enc.set_bytes(4, 4, &k_dim as *const u32 as *const _);
    enc.set_bytes(5, 4, &n_dim as *const u32 as *const _);
    enc.set_bytes(6, 4, &entry_dim as *const u32 as *const _);
    let nb = k_dim / entry_dim.max(1);
    let grid = metal::MTLSize {
        width: nb as u64,
        height: n_dim as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 32u64.min(nb as u64).max(1),
        height: 8u64.min(n_dim as u64).max(1),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// KAN Gaussian-RBF spline activation. One thread per output element
/// `gid ∈ [0, rows·channels)`; channel `c = gid % channels`. `x`, `coeff`,
/// `dst` are f32 arena tensors.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_spline_activation(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    coeff: usize,
    dst: usize,
    total: u32,
    channels: u32,
    num_basis: u32,
    grid_min: f32,
    grid_max: f32,
) {
    enc.set_compute_pipeline_state(&k.spline_activation);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let c_u = coeff as u64;
    enc.set_bytes(2, 8, &c_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    enc.set_bytes(4, 4, &total as *const u32 as *const _);
    enc.set_bytes(5, 4, &channels as *const u32 as *const _);
    enc.set_bytes(6, 4, &num_basis as *const u32 as *const _);
    enc.set_bytes(7, 4, &grid_min as *const f32 as *const _);
    enc.set_bytes(8, 4, &grid_max as *const f32 as *const _);
    let grid = metal::MTLSize {
        width: total as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(total) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// Fused synth backward: `dx` (`[m,k] = upstream·Ŵᵀ`, reconstruct-in-loop) or
/// `d_codebook` (`[num_entries, entry_dim]`, scatter-free block scan). One
/// dispatch, no tiling — see MSL `synth_bwd_dx` / `synth_bwd_codebook`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_synth_bwd(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    indices: usize,
    codebook: usize,
    upstream: usize,
    dst: usize,
    m: u32,
    n: u32,
    k_dim: u32,
    entry_dim: u32,
    num_entries: u32,
    dx: bool,
) {
    let d = entry_dim;
    let up = upstream as u64;
    let idx = indices as u64;
    let ds = dst as u64;
    enc.set_buffer(0, Some(buffer), 0);
    enc.set_bytes(1, 8, &up as *const u64 as *const _);
    enc.set_bytes(2, 8, &idx as *const u64 as *const _);
    if dx {
        // buffer(3) = codebook; grid = (k, m).
        enc.set_compute_pipeline_state(&k.synth_bwd_dx);
        let cb = codebook as u64;
        enc.set_bytes(3, 8, &cb as *const u64 as *const _);
        enc.set_bytes(4, 8, &ds as *const u64 as *const _);
        enc.set_bytes(5, 4, &m as *const u32 as *const _);
        enc.set_bytes(6, 4, &n as *const u32 as *const _);
        enc.set_bytes(7, 4, &k_dim as *const u32 as *const _);
        enc.set_bytes(8, 4, &d as *const u32 as *const _);
        let grid = metal::MTLSize {
            width: k_dim as u64,
            height: m as u64,
            depth: 1,
        };
        let tg = metal::MTLSize {
            width: 16.min(k_dim as u64).max(1),
            height: 16,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
    } else {
        // buffer(3) = x; grid = (entry_dim, num_entries).
        enc.set_compute_pipeline_state(&k.synth_bwd_codebook);
        let xx = x as u64;
        enc.set_bytes(3, 8, &xx as *const u64 as *const _);
        enc.set_bytes(4, 8, &ds as *const u64 as *const _);
        enc.set_bytes(5, 4, &m as *const u32 as *const _);
        enc.set_bytes(6, 4, &n as *const u32 as *const _);
        enc.set_bytes(7, 4, &k_dim as *const u32 as *const _);
        enc.set_bytes(8, 4, &d as *const u32 as *const _);
        enc.set_bytes(9, 4, &num_entries as *const u32 as *const _);
        let grid = metal::MTLSize {
            width: d as u64,
            height: num_entries as u64,
            depth: 1,
        };
        let tg = metal::MTLSize {
            width: (d as u64).max(1),
            height: 64,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
    }
}

pub(crate) fn encode_iq2_xs_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.iq2_xs_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    enc.set_buffer(6, Some(k.iq_grid_buffer()), 0);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_iq3_xxs_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.iq3_xxs_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    enc.set_buffer(6, Some(k.iq_grid_buffer()), 0);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_iq2_s_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.iq2_s_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    enc.set_buffer(6, Some(k.iq_grid_buffer()), 0);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_iq3_s_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.iq3_s_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    enc.set_buffer(6, Some(k.iq_grid_buffer()), 0);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_iq1_s_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.iq1_s_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    enc.set_buffer(6, Some(k.iq_grid_buffer()), 0);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_iq1_m_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.iq1_m_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    enc.set_buffer(6, Some(k.iq_grid_buffer()), 0);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_dequant_gguf(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    w_q: usize,
    dst: usize,
    scheme: rlx_ir::quant::QuantScheme,
    k_dim: usize,
    n_dim: usize,
) {
    let block_elems = scheme.gguf_block_size() as usize;
    let total = k_dim * n_dim;
    let num_blocks = total / block_elems.max(1);
    let scheme_id = gguf_scheme_id(scheme);
    // 12B Q4 GGUF activations sit at arena offsets > 4 GB. u32 cast on a
    // ~14 GB byte offset silently truncates the high bits and the dequant
    // kernel reads garbage from a wrap-around pointer — producing Q4K output
    // with values up to 1.2e11 and sparse NaN (task #50). Pass offsets as u64.
    let dst_f32 = (dst / 4) as u64;
    let w_u = w_q as u64;
    enc.set_compute_pipeline_state(&k.dequant_gguf);
    enc.set_buffer(0, Some(buffer), 0);
    enc.set_bytes(1, 8, &w_u as *const u64 as *const _);
    enc.set_bytes(2, 8, &dst_f32 as *const u64 as *const _);
    enc.set_bytes(3, 4, &scheme_id as *const u32 as *const _);
    let nb = num_blocks as u32;
    enc.set_bytes(4, 4, &nb as *const u32 as *const _);
    // buffer(5): IQ grid LUTs (KMASK | KSIGNS | KGRID_IQ2XXS | ... |
    // KGRID_IQ1S). Schemes 0..=11 ignore it. See `crate::kernels::iq_grid_buffer`.
    let lut = k.iq_grid_buffer();
    enc.set_buffer(5, Some(lut), 0);
    let grid = metal::MTLSize {
        width: num_blocks as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(num_blocks) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// True when MoE grouped matmul can dispatch per-token fused kernels onto the
/// parent compute encoder (no host sort / pack / unpermute, no private cmd_buf).
pub(crate) fn dequant_grouped_can_encode_per_row(
    scheme: rlx_ir::quant::QuantScheme,
    k_dim: usize,
) -> bool {
    if rlx_ir::env::flag("RLX_METAL_GROUPED_GEMV_DISABLE") {
        return false;
    }
    match scheme {
        rlx_ir::quant::QuantScheme::GgufQ4_0 | rlx_ir::quant::QuantScheme::GgufQ8_0 => {
            k_dim.is_multiple_of(32)
        }
        rlx_ir::quant::QuantScheme::GgufQ4K => {
            k_dim.is_multiple_of(256) && !rlx_ir::env::flag("RLX_METAL_Q4K_FUSED_DISABLE")
        }
        rlx_ir::quant::QuantScheme::GgufQ6K => {
            k_dim.is_multiple_of(256) && !rlx_ir::env::flag("RLX_METAL_Q6K_GEMM_DISABLE")
        }
        _ => false,
    }
}

/// Per-token fused MoE GEMV/GEMM on an existing encoder. Writes `dst` in the
/// original token order so callers do not need host unpermute or a mid-pipeline
/// `wait_until_completed`. `expert_idx` must already be resident in `buffer`
/// (host-uploaded or previously synced).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_dequant_grouped_matmul_gguf_per_row(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    input: usize,
    w_q: usize,
    expert_idx: usize,
    dst: usize,
    m: usize,
    k_dim: usize,
    n: usize,
    num_experts: usize,
    scheme: rlx_ir::quant::QuantScheme,
) {
    debug_assert!(dequant_grouped_can_encode_per_row(scheme, k_dim));
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let slab_bytes = (k_dim * n) / block_elems * block_bytes;
    let use_q4k_sg = matches!(scheme, rlx_ir::quant::QuantScheme::GgufQ4K)
        && n.is_multiple_of(8)
        && !rlx_ir::env::flag("RLX_METAL_Q4K_SG_DISABLE");
    let base = buffer.contents() as *const u8;
    let idx_host = unsafe { std::slice::from_raw_parts(base.add(expert_idx) as *const f32, m) };
    for row in 0..m {
        let e = idx_host[row] as usize;
        debug_assert!(e < num_experts, "expert_idx[{row}]={e} >= {num_experts}");
        let x_off = input + row * k_dim * 4;
        let y_off = dst + row * n * 4;
        let w_off = w_q + e * slab_bytes;
        match scheme {
            rlx_ir::quant::QuantScheme::GgufQ8_0 => {
                encode_q8_0_mv_f32(enc, k, buffer, x_off, w_off, y_off, k_dim, n)
            }
            rlx_ir::quant::QuantScheme::GgufQ4_0 => {
                encode_q4_0_mv_f32(enc, k, buffer, x_off, w_off, y_off, k_dim, n)
            }
            rlx_ir::quant::QuantScheme::GgufQ4K if use_q4k_sg => {
                encode_q4k_mv_f32_sg(enc, k, buffer, x_off, w_off, y_off, k_dim, n)
            }
            rlx_ir::quant::QuantScheme::GgufQ4K => {
                encode_q4k_mv_f32(enc, k, buffer, x_off, w_off, y_off, k_dim, n)
            }
            rlx_ir::quant::QuantScheme::GgufQ6K => {
                encode_qk_mm_f32(enc, &k.q6k_mm_f32, buffer, x_off, w_off, y_off, 1, k_dim, n)
            }
            _ => unreachable!("per-row grouped scheme guard"),
        }
    }
}

pub(crate) fn encode_dequant_grouped_matmul_gguf(
    queue: &metal::CommandQueueRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    scratch_off: usize,
    input: usize,
    w_q: usize,
    expert_idx: usize,
    dst: usize,
    m: usize,
    k_dim: usize,
    n: usize,
    num_experts: usize,
    scheme: rlx_ir::quant::QuantScheme,
) {
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let slab_bytes = (k_dim * n) / block_elems * block_bytes;

    let base = buffer.contents() as *const u8;
    unsafe {
        let x_host = std::slice::from_raw_parts(base.add(input) as *const f32, m * k_dim);
        let idx_host = std::slice::from_raw_parts(base.add(expert_idx) as *const f32, m);
        let (packed_in, original_pos, offsets) =
            rlx_cpu::gguf_matmul::grouped_moe_sort_plan(x_host, idx_host, m, k_dim, num_experts);

        let dequant_off = scratch_off;
        let pack_in_off = scratch_off + k_dim * n * 4;
        let pack_out_off = scratch_off + (k_dim * n + m * k_dim) * 4;

        std::ptr::copy_nonoverlapping(
            packed_in.as_ptr(),
            base.add(pack_in_off) as *mut f32,
            packed_in.len(),
        );

        // Fast paths that skip full-slab f32 dequant + MPS sgemm:
        // - Singleton experts (typical MoE decode, m=top_k): fused GEMV for
        //   Q4_0 / Q8_0 / Q4_K, or Q6_K via the fused mm kernel at m=1.
        // - Multi-token expert groups (Q4_K / Q6_K): fused GEMM per expert.
        // Off-switch: RLX_METAL_GROUPED_GEMV_DISABLE=1 (falls back to MPS).
        let all_singleton = (0..num_experts).all(|e| {
            let c = offsets[e + 1] - offsets[e];
            c == 0 || c == 1
        });
        let fused_off = rlx_ir::env::flag("RLX_METAL_GROUPED_GEMV_DISABLE");
        let use_fused_q40_gemv = !fused_off
            && all_singleton
            && k_dim.is_multiple_of(32)
            && matches!(
                scheme,
                rlx_ir::quant::QuantScheme::GgufQ4_0 | rlx_ir::quant::QuantScheme::GgufQ8_0
            );
        let use_fused_q4k_gemv = !fused_off
            && all_singleton
            && k_dim.is_multiple_of(256)
            && matches!(scheme, rlx_ir::quant::QuantScheme::GgufQ4K)
            && !rlx_ir::env::flag("RLX_METAL_Q4K_FUSED_DISABLE");
        let use_fused_qk_mm = !fused_off
            && k_dim.is_multiple_of(256)
            && match scheme {
                rlx_ir::quant::QuantScheme::GgufQ4K => {
                    !rlx_ir::env::flag("RLX_METAL_Q4K_GEMM_DISABLE")
                }
                rlx_ir::quant::QuantScheme::GgufQ6K => {
                    !rlx_ir::env::flag("RLX_METAL_Q6K_GEMM_DISABLE")
                }
                _ => false,
            };
        // Prefer Q4_K GEMV when every expert is a singleton; otherwise GEMM
        // (also covers Q6_K singletons — there is no dedicated q6k_mv).
        let use_fused_gemv = use_fused_q40_gemv || use_fused_q4k_gemv;
        let use_fused_mm = !use_fused_gemv && use_fused_qk_mm;

        let cmd_buf = queue.new_command_buffer();
        if use_fused_gemv {
            // Independent per-expert writes — one Concurrent encoder is enough.
            let enc = cmd_buf
                .compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
            let use_q4k_sg = use_fused_q4k_gemv
                && n.is_multiple_of(8)
                && !rlx_ir::env::flag("RLX_METAL_Q4K_SG_DISABLE");
            for e in 0..num_experts {
                let count = offsets[e + 1] - offsets[e];
                if count == 0 {
                    continue;
                }
                let x_off = pack_in_off + offsets[e] * k_dim * 4;
                let y_off = pack_out_off + offsets[e] * n * 4;
                let w_off = w_q + e * slab_bytes;
                match scheme {
                    rlx_ir::quant::QuantScheme::GgufQ8_0 => {
                        encode_q8_0_mv_f32(enc, k, buffer, x_off, w_off, y_off, k_dim, n)
                    }
                    rlx_ir::quant::QuantScheme::GgufQ4_0 => {
                        encode_q4_0_mv_f32(enc, k, buffer, x_off, w_off, y_off, k_dim, n)
                    }
                    rlx_ir::quant::QuantScheme::GgufQ4K if use_q4k_sg => {
                        encode_q4k_mv_f32_sg(enc, k, buffer, x_off, w_off, y_off, k_dim, n)
                    }
                    rlx_ir::quant::QuantScheme::GgufQ4K => {
                        encode_q4k_mv_f32(enc, k, buffer, x_off, w_off, y_off, k_dim, n)
                    }
                    _ => unreachable!("fused grouped GEMV scheme guard"),
                }
            }
            enc.end_encoding();
        } else if use_fused_mm {
            let pipeline = match scheme {
                rlx_ir::quant::QuantScheme::GgufQ4K => &k.q4k_mm_f32,
                rlx_ir::quant::QuantScheme::GgufQ6K => &k.q6k_mm_f32,
                _ => unreachable!("fused grouped GEMM scheme guard"),
            };
            let enc = cmd_buf
                .compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
            for e in 0..num_experts {
                let count = offsets[e + 1] - offsets[e];
                if count == 0 {
                    continue;
                }
                let x_off = pack_in_off + offsets[e] * k_dim * 4;
                let y_off = pack_out_off + offsets[e] * n * 4;
                let w_off = w_q + e * slab_bytes;
                encode_qk_mm_f32(enc, pipeline, buffer, x_off, w_off, y_off, count, k_dim, n);
            }
            enc.end_encoding();
        } else {
            // Per expert: dequant the slab into `dequant_off` on a dedicated MSL
            // compute encoder, END that encoder, then MPS sgemm. The compute
            // encoder MUST be ended before the MPS call — MPS opens its own
            // encoder internally, and two live encoders on one command buffer is a
            // hard Metal abort (`A command encoder is already encoding...`).
            // Encoders execute serially in submission order, so expert e's sgemm
            // reads `dequant_off` before expert e+1's dequant overwrites it.
            for e in 0..num_experts {
                let count = offsets[e + 1] - offsets[e];
                if count == 0 {
                    continue;
                }
                let enc = cmd_buf
                    .compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial);
                encode_dequant_gguf(
                    enc,
                    k,
                    buffer,
                    w_q + e * slab_bytes,
                    dequant_off,
                    scheme,
                    k_dim,
                    n,
                );
                enc.end_encoding();
                let in_start = offsets[e];
                crate::mps_blas::encode_mps_sgemm_bt(
                    cmd_buf,
                    buffer,
                    pack_in_off + in_start * k_dim * 4,
                    dequant_off,
                    pack_out_off + in_start * n * 4,
                    count,
                    k_dim,
                    n,
                );
            }
        }
        // Sgemm/GEMV results must land before the host-side unpermute reads them.
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let pack_out_host = std::slice::from_raw_parts(base.add(pack_out_off) as *const f32, m * n);
        let mut out_host = vec![0f32; m * n];
        rlx_cpu::gguf_matmul::grouped_moe_unpermute_out(
            pack_out_host,
            &original_pos,
            &mut out_host,
            m,
            n,
        );
        std::ptr::copy_nonoverlapping(out_host.as_ptr(), base.add(dst) as *mut f32, out_host.len());
    }
}

pub(crate) fn gdn_ephemeral_state_bytes(graph: &Graph) -> usize {
    let mut max = 0usize;
    for node in graph.nodes() {
        if let Op::GatedDeltaNet {
            carry_state,
            state_size,
            ..
        } = &node.op
            && !*carry_state
        {
            let q_shape = &graph.node(node.inputs[0]).shape;
            let elems = q_shape.dim(0).unwrap_static()
                * q_shape.dim(2).unwrap_static()
                * state_size
                * state_size;
            max = max.max(elems * std::mem::size_of::<f32>());
        }
    }
    max
}

pub(crate) fn encode_gated_delta_net(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    q: usize,
    k_off: usize,
    v: usize,
    g: usize,
    beta: usize,
    state: usize,
    dst: usize,
    batch: u32,
    seq: u32,
    heads: u32,
    state_size: u32,
    use_carry: bool,
    gate_per_channel: bool,
) {
    // ulong float indices — Fara arenas put GDN scratch past 16 GiB.
    let f32_idx = |byte_off: usize| -> u64 { (byte_off / 4) as u64 };
    // Simdgroup GDN (n==128) is the default: the column-private kernel launches
    // only b*h threads (~48 for Qwen3.6-27B) and left ~99% of the GPU idle —
    // it was 70% of decode time. The simdgroup variant (one simdgroup per state
    // column) is ~1.5× faster on decode and ~6× on prefill, bit-identical.
    // Opt out with RLX_METAL_GDN_SG_DISABLE=1.
    let use_sg = state_size == 128 && !rlx_ir::env::flag("RLX_METAL_GDN_SG_DISABLE");
    if use_sg {
        enc.set_compute_pipeline_state(&k.gated_delta_net_sg);
    } else {
        enc.set_compute_pipeline_state(&k.gated_delta_net);
    }
    enc.set_buffer(0, Some(buffer), 0);
    let q_u = f32_idx(q);
    let k_u = f32_idx(k_off);
    let v_u = f32_idx(v);
    let g_u = f32_idx(g);
    let beta_u = f32_idx(beta);
    let state_u = f32_idx(state);
    let dst_u = f32_idx(dst);
    enc.set_bytes(1, 8, &q_u as *const u64 as *const _);
    enc.set_bytes(2, 8, &k_u as *const u64 as *const _);
    enc.set_bytes(3, 8, &v_u as *const u64 as *const _);
    enc.set_bytes(4, 8, &g_u as *const u64 as *const _);
    enc.set_bytes(5, 8, &beta_u as *const u64 as *const _);
    enc.set_bytes(6, 8, &state_u as *const u64 as *const _);
    enc.set_bytes(7, 8, &dst_u as *const u64 as *const _);
    let dims = [batch, seq, heads, state_size];
    enc.set_bytes(8, 16, dims.as_ptr() as *const _);
    let use_carry_u: u32 = if use_carry { 1 } else { 0 };
    enc.set_bytes(9, 4, &use_carry_u as *const u32 as *const _);
    let gpc_u: u32 = if gate_per_channel { 1 } else { 0 };
    enc.set_bytes(10, 4, &gpc_u as *const u32 as *const _);
    if use_sg {
        const NSG: u64 = 4;
        let grid = metal::MTLSize {
            width: (state_size as u64) / NSG,
            height: heads as u64,
            depth: batch as u64,
        };
        let tg = metal::MTLSize {
            width: 32,
            height: NSG,
            depth: 1,
        };
        enc.dispatch_thread_groups(grid, tg);
    } else {
        // One thread per (batch, head) — matches selective_scan.
        let threads = (batch * heads).max(1) as u64;
        let tg_w = k.gated_delta_net.thread_execution_width().min(threads);
        enc.dispatch_threads(
            metal::MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
    }
}

/// Native MSL selective scan (f32, `state_size <= SSM_MAX_N = 128`). One
/// thread per `(batch, channel)`; each owns a private state vector and
/// scans sequentially over the seq axis. Matches `execute_selective_scan_f32`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_selective_scan(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    delta: usize,
    a: usize,
    b: usize,
    c: usize,
    dst: usize,
    batch: u32,
    seq: u32,
    hidden: u32,
    state_size: u32,
) {
    let f32_idx = |byte_off: usize| -> u32 { (byte_off / 4) as u32 };
    let p = &k.selective_scan;
    enc.set_compute_pipeline_state(p);
    enc.set_buffer(0, Some(buffer), 0);
    let offs = [
        f32_idx(x),
        f32_idx(delta),
        f32_idx(a),
        f32_idx(b),
        f32_idx(c),
        f32_idx(dst),
    ];
    for (i, off) in offs.iter().enumerate() {
        enc.set_bytes((i + 1) as u64, 4, off as *const u32 as *const _);
    }
    let dims = [batch, seq, hidden, state_size];
    enc.set_bytes(7, 16, dims.as_ptr() as *const _);
    let threads = (batch * hidden) as u64;
    let tg_w = p.thread_execution_width().min(threads.max(1));
    enc.dispatch_threads(
        metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

/// Native MSL forward LSTM (f32, `hidden <= LSTM_MAX_H = 1024`). One
/// threadgroup per batch item, `hidden` threads each.
/// Native MSL LSTM (f32, any layers / dirs / carry, `hidden ≤ 1024`). Same
/// per-(layer, direction) loop / in-arena scratch as `encode_gru`; gate order
/// i,f,g,o with a single merged bias, `h0`+`c0` carry. Bit-for-bit mirror of
/// `execute_lstm_f32`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lstm(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    scratch: usize,
    x: usize,
    w_ih: usize,
    w_hh: usize,
    bias: usize,
    h0: usize,
    c0: usize,
    dst: usize,
    batch: u32,
    seq: u32,
    input_size: u32,
    hidden: u32,
    num_layers: u32,
    bidirectional: bool,
    carry: bool,
) {
    let (b, s, h) = (batch as usize, seq as usize, hidden as usize);
    let dirs = if bidirectional { 2 } else { 1 };
    let four_h = 4 * h;
    let out_width = dirs * h;
    let layer_elems = b * s * out_width;
    let scratch_w = scratch / 4;

    enc.set_compute_pipeline_state(&k.lstm);
    enc.set_buffer(0, Some(buffer), 0);
    let grid = metal::MTLSize {
        width: batch as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: hidden as u64,
        height: 1,
        depth: 1,
    };

    let mut in_l = input_size as usize;
    let mut wih_cursor = 0usize;
    let mut in_off_w = x / 4;

    for l in 0..num_layers as usize {
        let last = l + 1 == num_layers as usize;
        let out_off_w = if last {
            dst / 4
        } else {
            scratch_w + (l % 2) * layer_elems
        };
        let wih_block = four_h * in_l;

        for dir in 0..dirs {
            let ld = l * dirs + dir;
            let offs: [u32; 5] = [
                (in_off_w) as u32,
                (w_ih / 4 + wih_cursor + dir * wih_block) as u32,
                (w_hh / 4 + ld * four_h * h) as u32,
                (bias / 4 + ld * four_h) as u32,
                (out_off_w) as u32,
            ];
            for (i, o) in offs.iter().enumerate() {
                enc.set_bytes((i + 1) as u64, 4, o as *const u32 as *const _);
            }
            let dims = [batch, seq, in_l as u32, hidden];
            enc.set_bytes(6, 16, dims.as_ptr() as *const _);
            let h0_off = if carry { h0 / 4 + ld * b * h } else { 0 };
            let c0_off = if carry { c0 / 4 + ld * b * h } else { 0 };
            let more = [
                h0_off as u32,
                out_width as u32,
                (dir * h) as u32,
                u32::from(dir == 1),
            ];
            enc.set_bytes(7, 16, more.as_ptr() as *const _);
            let c0_u = c0_off as u32;
            enc.set_bytes(8, 4, &c0_u as *const u32 as *const _);
            enc.dispatch_thread_groups(grid, tg);
        }

        wih_cursor += dirs * wih_block;
        in_l = out_width;
        in_off_w = scratch_w + (l % 2) * layer_elems;
    }
}

/// Native MSL GRU (f32, any layers / dirs / carry, `hidden ≤ 1024`). Loops
/// (layer, direction) on one encoder — Metal's default serial dispatch orders
/// the launches — ping-ponging intermediate layer outputs through an in-arena
/// scratch region at `scratch` (byte offset; needed only when `num_layers > 1`).
/// `h0` is a byte offset when `carry`, else ignored. Bit-for-bit mirror of
/// `execute_gru_f32`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gru(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    scratch: usize,
    x: usize,
    w_ih: usize,
    w_hh: usize,
    b_ih: usize,
    b_hh: usize,
    h0: usize,
    dst: usize,
    batch: u32,
    seq: u32,
    input_size: u32,
    hidden: u32,
    num_layers: u32,
    bidirectional: bool,
    carry: bool,
) {
    let (b, s, h) = (batch as usize, seq as usize, hidden as usize);
    let dirs = if bidirectional { 2 } else { 1 };
    let three_h = 3 * h;
    let out_width = dirs * h;
    let layer_elems = b * s * out_width;
    let scratch_w = scratch / 4;

    enc.set_compute_pipeline_state(&k.gru);
    enc.set_buffer(0, Some(buffer), 0);
    let grid = metal::MTLSize {
        width: batch as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: hidden as u64,
        height: 1,
        depth: 1,
    };

    let mut in_l = input_size as usize;
    let mut wih_cursor = 0usize;
    let mut in_off_w = x / 4;

    for l in 0..num_layers as usize {
        let last = l + 1 == num_layers as usize;
        let out_off_w = if last {
            dst / 4
        } else {
            scratch_w + (l % 2) * layer_elems
        };
        let wih_block = three_h * in_l;

        for dir in 0..dirs {
            let ld = l * dirs + dir;
            let offs: [u32; 6] = [
                (in_off_w) as u32,
                (w_ih / 4 + wih_cursor + dir * wih_block) as u32,
                (w_hh / 4 + ld * three_h * h) as u32,
                (b_ih / 4 + ld * three_h) as u32,
                (b_hh / 4 + ld * three_h) as u32,
                (out_off_w) as u32,
            ];
            for (i, o) in offs.iter().enumerate() {
                enc.set_bytes((i + 1) as u64, 4, o as *const u32 as *const _);
            }
            let dims = [batch, seq, in_l as u32, hidden];
            enc.set_bytes(7, 16, dims.as_ptr() as *const _);
            let h0_off = if carry { h0 / 4 + ld * b * h } else { 0 };
            let more = [
                h0_off as u32,
                out_width as u32,
                (dir * h) as u32,
                u32::from(dir == 1),
            ];
            enc.set_bytes(8, 16, more.as_ptr() as *const _);
            enc.dispatch_thread_groups(grid, tg);
        }

        wih_cursor += dirs * wih_block;
        in_l = out_width;
        in_off_w = scratch_w + (l % 2) * layer_elems;
    }
}

/// Native MSL Elman RNN (f32, any layers / dirs / carry, `hidden ≤ 1024`). Same
/// per-(layer, direction) loop / in-arena scratch as `encode_gru`; `relu`
/// selects ReLU vs tanh. Bit-for-bit mirror of `execute_rnn_f32`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_rnn(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    scratch: usize,
    x: usize,
    w_ih: usize,
    w_hh: usize,
    bias: usize,
    h0: usize,
    dst: usize,
    batch: u32,
    seq: u32,
    input_size: u32,
    hidden: u32,
    num_layers: u32,
    bidirectional: bool,
    carry: bool,
    relu: bool,
) {
    let (b, s, h) = (batch as usize, seq as usize, hidden as usize);
    let dirs = if bidirectional { 2 } else { 1 };
    let out_width = dirs * h;
    let layer_elems = b * s * out_width;
    let scratch_w = scratch / 4;

    enc.set_compute_pipeline_state(&k.rnn);
    enc.set_buffer(0, Some(buffer), 0);
    let relu_u: u32 = if relu { 1 } else { 0 };
    let grid = metal::MTLSize {
        width: batch as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: hidden as u64,
        height: 1,
        depth: 1,
    };

    let mut in_l = input_size as usize;
    let mut wih_cursor = 0usize;
    let mut in_off_w = x / 4;

    for l in 0..num_layers as usize {
        let last = l + 1 == num_layers as usize;
        let out_off_w = if last {
            dst / 4
        } else {
            scratch_w + (l % 2) * layer_elems
        };
        let wih_block = h * in_l;

        for dir in 0..dirs {
            let ld = l * dirs + dir;
            let offs: [u32; 5] = [
                (in_off_w) as u32,
                (w_ih / 4 + wih_cursor + dir * wih_block) as u32,
                (w_hh / 4 + ld * h * h) as u32,
                (bias / 4 + ld * h) as u32,
                (out_off_w) as u32,
            ];
            for (i, o) in offs.iter().enumerate() {
                enc.set_bytes((i + 1) as u64, 4, o as *const u32 as *const _);
            }
            let dims = [batch, seq, in_l as u32, hidden];
            enc.set_bytes(6, 16, dims.as_ptr() as *const _);
            enc.set_bytes(7, 4, &relu_u as *const u32 as *const _);
            let h0_off = if carry { h0 / 4 + ld * b * h } else { 0 };
            let more = [
                h0_off as u32,
                out_width as u32,
                (dir * h) as u32,
                u32::from(dir == 1),
            ];
            enc.set_bytes(8, 16, more.as_ptr() as *const _);
            enc.dispatch_thread_groups(grid, tg);
        }

        wih_cursor += dirs * wih_block;
        in_l = out_width;
        in_off_w = scratch_w + (l % 2) * layer_elems;
    }
}

/// Native MSL Mamba-2 SSD scan (f32, `state_size ≤ 128`). One thread per
/// `(batch, head, head_dim_pos)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mamba2(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    dt: usize,
    a: usize,
    b: usize,
    c: usize,
    dst: usize,
    batch: u32,
    seq: u32,
    heads: u32,
    head_dim: u32,
    state_size: u32,
) {
    let f32_idx = |o: usize| -> u32 { (o / 4) as u32 };
    let p = &k.mamba2;
    enc.set_compute_pipeline_state(p);
    enc.set_buffer(0, Some(buffer), 0);
    let offs = [
        f32_idx(x),
        f32_idx(dt),
        f32_idx(a),
        f32_idx(b),
        f32_idx(c),
        f32_idx(dst),
    ];
    for (i, o) in offs.iter().enumerate() {
        enc.set_bytes((i + 1) as u64, 4, o as *const u32 as *const _);
    }
    // dims.w packs head_dim (high 16) | state_size (low 16).
    let packed = (head_dim << 16) | (state_size & 0xffff);
    let dims = [batch, seq, heads, packed];
    enc.set_bytes(7, 16, dims.as_ptr() as *const _);
    let threads = (batch * heads * head_dim) as u64;
    let tg_w = p.thread_execution_width().min(threads.max(1));
    enc.dispatch_threads(
        metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn conv_bwd_scratch_bytes(graph: &Graph) -> usize {
    let mut max = 0usize;
    for node in graph.nodes() {
        if let Op::Conv2dBackwardWeight {
            kernel_size,
            groups,
            ..
        } = &node.op
        {
            let x_shape = &graph.node(node.inputs[0]).shape;
            let dy_shape = &graph.node(node.inputs[1]).shape;
            if x_shape.rank() != 4 || dy_shape.rank() != 4 {
                continue;
            }
            let c_in = x_shape.dim(1).unwrap_static();
            let h_out = dy_shape.dim(2).unwrap_static();
            let w_out = dy_shape.dim(3).unwrap_static();
            let kh = kernel_size.first().copied().unwrap_or(1);
            let kw = kernel_size.get(1).copied().unwrap_or(1);
            let groups = (*groups).max(1);
            let c_in_per_g = c_in / groups;
            let n_dim = c_in_per_g * kh * kw;
            let k_dim = h_out * w_out;
            // n==1 im2col path needs n_dim*k_dim f32; the batch-parallel
            // two-pass path needs N*C_out*c_in_per_g*kh*kw f32 partials.
            // Size the shared scratch for whichever is larger.
            let n_batch = x_shape.dim(0).unwrap_static();
            let c_out = dy_shape.dim(1).unwrap_static();
            let two_pass = n_batch * c_out * c_in_per_g * kh * kw;
            let need = (n_dim * k_dim).max(two_pass);
            max = max.max(need * std::mem::size_of::<f32>());
        }
    }
    max
}

/// Implicit im2col+GEMM only when explicitly enabled — materialized im2col + MPS/simd
/// sgemm wins on Voxtral-scale conv weight backward (see bench-encoder).
pub(crate) fn conv_bwd_weight_use_implicit_gemm(m: usize, k: usize, n: usize) -> bool {
    if !rlx_ir::env::var("RLX_METAL_CONV_BWD_IMPLICIT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return false;
    }
    if !k.is_multiple_of(8) || n < 8 || m < 1 {
        return false;
    }
    !matches!(
        crate::cost::hw_model().pick_sgemm(m, k, n),
        crate::cost::SgemmVariant::Mps
            | crate::cost::SgemmVariant::Tiled
            | crate::cost::SgemmVariant::Naive
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_conv2d_bwd_weight_gemm(
    enc: &metal::ComputeCommandEncoderRef,
    kk: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    dy: usize,
    x: usize,
    dw: usize,
    m: usize,
    k: usize,
    n: usize,
    nchw: &[u32; 4],
    out_dims: &[u32; 4],
    kshape: &[u32; 4],
    padd: &[u32; 4],
) {
    let m_u = m as u32;
    let k_u = k as u32;
    let n_u = n as u32;
    enc.set_buffer(0, Some(buffer), dy as u64);
    enc.set_buffer(1, Some(buffer), x as u64);
    enc.set_buffer(2, Some(buffer), dw as u64);
    enc.set_bytes(3, 4, &m_u as *const _ as *const _);
    enc.set_bytes(4, 4, &k_u as *const _ as *const _);
    enc.set_bytes(5, 4, &n_u as *const _ as *const _);
    enc.set_bytes(6, 16, nchw.as_ptr() as *const _);
    enc.set_bytes(7, 16, out_dims.as_ptr() as *const _);
    enc.set_bytes(8, 16, kshape.as_ptr() as *const _);
    enc.set_bytes(9, 16, padd.as_ptr() as *const _);

    let aligned_32 = m.is_multiple_of(32) && k.is_multiple_of(32) && n.is_multiple_of(32);
    if aligned_32 && m >= 32 && n >= 32 {
        enc.set_compute_pipeline_state(&kk.conv2d_bwd_weight_gemm_4x4);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: n.div_ceil(32) as u64,
                height: m.div_ceil(32) as u64,
                depth: 1,
            },
            metal::MTLSize {
                width: 512,
                height: 1,
                depth: 1,
            },
        );
    } else {
        enc.set_compute_pipeline_state(&kk.conv2d_bwd_weight_gemm);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: n.div_ceil(8) as u64,
                height: m.div_ceil(8) as u64,
                depth: 1,
            },
            metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_im2col_group(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    col: usize,
    nchw: &[u32; 4],
    out_dims: &[u32; 4],
    kshape: &[u32; 4],
    padd: &[u32; 4],
    elems: u64,
) {
    let w1 = nchw[2] == 1 && out_dims[2] == 1;
    if w1 {
        enc.set_compute_pipeline_state(&k.im2col_group_w1);
    } else {
        enc.set_compute_pipeline_state(&k.im2col_group);
    }
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), col as u64);
    enc.set_bytes(2, 16, nchw.as_ptr() as *const _);
    enc.set_bytes(3, 16, out_dims.as_ptr() as *const _);
    enc.set_bytes(4, 16, kshape.as_ptr() as *const _);
    enc.set_bytes(5, 16, padd.as_ptr() as *const _);
    let tg_w = 512u64.min(elems).max(1);
    enc.dispatch_threads(
        metal::MTLSize {
            width: elems,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_conv2d(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    weight: usize,
    dst: usize,
    n: u32,
    c_in: u32,
    h: u32,
    w: u32,
    c_out: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    groups: u32,
) {
    let nch: [u32; 4] = [n, c_in, h, w];
    let out_dims: [u32; 4] = [c_out, h_out, w_out, groups];
    let kshape: [u32; 4] = [kh, kw, sh, sw];
    let padd: [u32; 4] = [ph, pw, dh, dw];
    let w1 = w == 1 && w_out == 1;
    if w1 {
        enc.set_compute_pipeline_state(&k.conv2d_w1);
    } else {
        enc.set_compute_pipeline_state(&k.conv2d);
    }
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), weight as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 16, nch.as_ptr() as *const _);
    enc.set_bytes(4, 16, out_dims.as_ptr() as *const _);
    enc.set_bytes(5, 16, kshape.as_ptr() as *const _);
    enc.set_bytes(6, 16, padd.as_ptr() as *const _);
    let grid = if w1 {
        metal::MTLSize {
            width: 1,
            height: h_out as u64,
            depth: (n * c_out) as u64,
        }
    } else {
        metal::MTLSize {
            width: w_out as u64,
            height: h_out as u64,
            depth: (n * c_out) as u64,
        }
    };
    let tg = if w1 {
        metal::MTLSize {
            width: 1,
            height: 8.min(h_out as u64),
            depth: 1,
        }
    } else {
        metal::MTLSize {
            width: 8.min(w_out as u64),
            height: 8.min(h_out as u64),
            depth: 1,
        }
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_group_norm(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    g: usize,
    b: usize,
    dst: usize,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    num_groups: u32,
    eps: f32,
) {
    let nchw: [u32; 4] = [n, c, h, w];
    enc.set_compute_pipeline_state(&k.group_norm);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), g as u64);
    enc.set_buffer(2, Some(buffer), b as u64);
    enc.set_buffer(3, Some(buffer), dst as u64);
    enc.set_bytes(4, 16, nchw.as_ptr() as *const _);
    enc.set_bytes(5, 4, &num_groups as *const u32 as *const _);
    enc.set_bytes(6, 4, &eps as *const f32 as *const _);
    let groups = (n * num_groups) as u64;
    let tg = metal::MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    // Dispatch one threadgroup per (batch, group) along grid *width* so
    // `threadgroup_position_in_grid` (scalar .x) indexes 0..batch*num_groups-1.
    let grid = metal::MTLSize {
        width: groups.max(1),
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg);
}

pub(crate) fn encode_resize_nearest_2x(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
) {
    let nchw: [u32; 4] = [n, c, h, w];
    let w2 = w * 2;
    let h2 = h * 2;
    enc.set_compute_pipeline_state(&k.resize_nearest_2x);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 16, nchw.as_ptr() as *const _);
    let grid = metal::MTLSize {
        width: w2 as u64,
        height: h2 as u64,
        depth: (n * c) as u64,
    };
    let tg = metal::MTLSize {
        width: 8.min(w2 as u64),
        height: 8.min(h2 as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_layer_norm2d(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    g: usize,
    b: usize,
    dst: usize,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    eps: f32,
) {
    let nchw: [u32; 4] = [n, c, h, w];
    enc.set_compute_pipeline_state(&k.layer_norm2d);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), g as u64);
    enc.set_buffer(2, Some(buffer), b as u64);
    enc.set_buffer(3, Some(buffer), dst as u64);
    enc.set_bytes(4, 16, nchw.as_ptr() as *const _);
    enc.set_bytes(5, 4, &eps as *const f32 as *const _);
    let grid = metal::MTLSize {
        width: w as u64,
        height: h as u64,
        depth: n as u64,
    };
    let tg = metal::MTLSize {
        width: 8.min(w as u64),
        height: 8.min(h as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_conv_transpose2d(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    weight: usize,
    dst: usize,
    n: u32,
    c_in: u32,
    h: u32,
    w: u32,
    c_out: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    groups: u32,
) {
    let nch: [u32; 4] = [n, c_in, h, w];
    let out_dims: [u32; 4] = [c_out, h_out, w_out, groups];
    let kshape: [u32; 4] = [kh, kw, sh, sw];
    let padd: [u32; 4] = [ph, pw, dh, dw];
    enc.set_compute_pipeline_state(&k.conv_transpose2d);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), weight as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 16, nch.as_ptr() as *const _);
    enc.set_bytes(4, 16, out_dims.as_ptr() as *const _);
    enc.set_bytes(5, 16, kshape.as_ptr() as *const _);
    enc.set_bytes(6, 16, padd.as_ptr() as *const _);
    let grid = metal::MTLSize {
        width: w_out as u64,
        height: h_out as u64,
        depth: (n * c_out) as u64,
    };
    let tg = metal::MTLSize {
        width: 8.min(w_out as u64),
        height: 8.min(h_out as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_conv3d(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    weight: usize,
    dst: usize,
    n: u32,
    c_in: u32,
    d: u32,
    h: u32,
    w: u32,
    c_out: u32,
    d_out: u32,
    h_out: u32,
    w_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
    dd: u32,
    dh: u32,
    dw: u32,
    groups: u32,
) {
    let a: [u32; 4] = [n, c_in, d, h];
    let b: [u32; 4] = [w, c_out, d_out, h_out];
    let c: [u32; 4] = [w_out, kd, kh, kw];
    let dparams: [u32; 4] = [sd, sh, sw, groups];
    let e: [u32; 4] = [pd, ph, pw, dd];
    let f: [u32; 4] = [dh, dw, 0, 0];
    enc.set_compute_pipeline_state(&k.conv3d);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), weight as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 16, a.as_ptr() as *const _);
    enc.set_bytes(4, 16, b.as_ptr() as *const _);
    enc.set_bytes(5, 16, c.as_ptr() as *const _);
    enc.set_bytes(6, 16, dparams.as_ptr() as *const _);
    enc.set_bytes(7, 16, e.as_ptr() as *const _);
    enc.set_bytes(8, 16, f.as_ptr() as *const _);
    // Grid: (w_out, h_out, n * c_out * d_out). Kernel decodes
    // d_o = z % d_out, then (n, c_out) from z / d_out.
    let grid = metal::MTLSize {
        width: w_out as u64,
        height: h_out as u64,
        depth: (n * c_out * d_out) as u64,
    };
    let tg = metal::MTLSize {
        width: 8.min(w_out as u64),
        height: 8.min(h_out as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_conv_transpose3d(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    weight: usize,
    dst: usize,
    n: u32,
    c_in: u32,
    d: u32,
    h: u32,
    w: u32,
    c_out: u32,
    d_out: u32,
    h_out: u32,
    w_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
    dd: u32,
    dh: u32,
    dw: u32,
    groups: u32,
) {
    let a: [u32; 4] = [n, c_in, d, h];
    let b: [u32; 4] = [w, c_out, d_out, h_out];
    let c: [u32; 4] = [w_out, kd, kh, kw];
    let dparams: [u32; 4] = [sd, sh, sw, groups];
    let e: [u32; 4] = [pd, ph, pw, dd];
    let f: [u32; 4] = [dh, dw, 0, 0];
    enc.set_compute_pipeline_state(&k.conv_transpose3d);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), weight as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 16, a.as_ptr() as *const _);
    enc.set_bytes(4, 16, b.as_ptr() as *const _);
    enc.set_bytes(5, 16, c.as_ptr() as *const _);
    enc.set_bytes(6, 16, dparams.as_ptr() as *const _);
    enc.set_bytes(7, 16, e.as_ptr() as *const _);
    enc.set_bytes(8, 16, f.as_ptr() as *const _);
    let grid = metal::MTLSize {
        width: w_out as u64,
        height: h_out as u64,
        depth: (n * c_out * d_out) as u64,
    };
    let tg = metal::MTLSize {
        width: 8.min(w_out as u64),
        height: 8.min(h_out as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_pool2d(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    kind: rlx_ir::op::ReduceOp,
) {
    use rlx_ir::op::ReduceOp;
    let kind_u: u32 = match kind {
        ReduceOp::Sum => 0,
        ReduceOp::Mean => 1,
        ReduceOp::Max => 2,
        ReduceOp::Min => 3,
        ReduceOp::Prod => 4,
    };
    let nchw: [u32; 4] = [n, c, h, w];
    let hw_out: [u32; 2] = [h_out, w_out];
    let khsw: [u32; 4] = [kh, kw, sh, sw];
    let pad: [u32; 2] = [ph, pw];
    enc.set_compute_pipeline_state(&k.pool2d);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 16, nchw.as_ptr() as *const _);
    enc.set_bytes(3, 8, hw_out.as_ptr() as *const _);
    enc.set_bytes(4, 16, khsw.as_ptr() as *const _);
    enc.set_bytes(5, 8, pad.as_ptr() as *const _);
    enc.set_bytes(6, 4, &kind_u as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: w_out as u64,
        height: h_out as u64,
        depth: (n * c) as u64,
    };
    let tg = metal::MTLSize {
        width: 8.min(w_out as u64),
        height: 8.min(h_out as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_maxpool2d_backward(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    dy: usize,
    dx: usize,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
) {
    let p0: [u32; 4] = [n, c, h, w];
    let p1: [u32; 4] = [h_out, w_out, kh, kw];
    let p2: [u32; 4] = [sh, sw, ph, pw];
    enc.set_compute_pipeline_state(&k.maxpool2d_backward);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), dy as u64);
    enc.set_buffer(2, Some(buffer), dx as u64);
    enc.set_bytes(3, 16, p0.as_ptr() as *const _);
    enc.set_bytes(4, 16, p1.as_ptr() as *const _);
    enc.set_bytes(5, 16, p2.as_ptr() as *const _);
    let grid = metal::MTLSize {
        width: w as u64,
        height: h as u64,
        depth: (n * c) as u64,
    };
    let tg = metal::MTLSize {
        width: 8.min(w as u64),
        height: 8.min(h as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_maxpool3d_backward(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    dy: usize,
    dx: usize,
    n: u32,
    c: u32,
    d: u32,
    h: u32,
    w: u32,
    d_out: u32,
    h_out: u32,
    w_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
) {
    let p0: [u32; 4] = [n, c, d, h];
    let p1: [u32; 4] = [w, d_out, h_out, w_out];
    let p2: [u32; 4] = [kd, kh, kw, sd];
    let p3: [u32; 4] = [sh, sw, pd, ph];
    enc.set_compute_pipeline_state(&k.maxpool3d_backward);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), dy as u64);
    enc.set_buffer(2, Some(buffer), dx as u64);
    enc.set_bytes(3, 16, p0.as_ptr() as *const _);
    enc.set_bytes(4, 16, p1.as_ptr() as *const _);
    enc.set_bytes(5, 16, p2.as_ptr() as *const _);
    enc.set_bytes(6, 16, p3.as_ptr() as *const _);
    enc.set_bytes(7, 4, (&pw as *const u32) as *const _);
    let total = (n * c * d * h * w) as u64;
    let grid = metal::MTLSize {
        width: total.max(1),
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(total.max(1)),
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_conv3d_backward_input(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    dy: usize,
    w: usize,
    dx: usize,
    n: u32,
    c_in: u32,
    d: u32,
    h: u32,
    w_in: u32,
    c_out: u32,
    d_out: u32,
    h_out: u32,
    w_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
    dd: u32,
    dh: u32,
    dw: u32,
    groups: u32,
) {
    let a: [u32; 4] = [n, c_in, d, h];
    let b: [u32; 4] = [w_in, c_out, d_out, h_out];
    let cc: [u32; 4] = [w_out, kd, kh, kw];
    let dd_p: [u32; 4] = [sd, sh, sw, pd];
    let e: [u32; 4] = [ph, pw, dd, dh];
    let f: [u32; 2] = [dw, groups];
    enc.set_compute_pipeline_state(&k.conv3d_backward_input);
    enc.set_buffer(0, Some(buffer), dy as u64);
    enc.set_buffer(1, Some(buffer), w as u64);
    enc.set_buffer(2, Some(buffer), dx as u64);
    enc.set_bytes(3, 16, a.as_ptr() as *const _);
    enc.set_bytes(4, 16, b.as_ptr() as *const _);
    enc.set_bytes(5, 16, cc.as_ptr() as *const _);
    enc.set_bytes(6, 16, dd_p.as_ptr() as *const _);
    enc.set_bytes(7, 16, e.as_ptr() as *const _);
    enc.set_bytes(8, 8, f.as_ptr() as *const _);
    let total = (n * c_in * d * h * w_in) as u64;
    let grid = metal::MTLSize {
        width: total.max(1),
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(total.max(1)),
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_conv3d_backward_weight(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    dy: usize,
    dw: usize,
    n: u32,
    c_in: u32,
    d: u32,
    h: u32,
    w: u32,
    c_out: u32,
    d_out: u32,
    h_out: u32,
    w_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
    dd: u32,
    dh: u32,
    dw_dil: u32,
    groups: u32,
) {
    let a: [u32; 4] = [n, c_in, d, h];
    let b: [u32; 4] = [w, c_out, d_out, h_out];
    let cc: [u32; 4] = [w_out, kd, kh, kw];
    let dd_p: [u32; 4] = [sd, sh, sw, pd];
    let e: [u32; 4] = [ph, pw, dd, dh];
    let f: [u32; 2] = [dw_dil, groups];
    enc.set_compute_pipeline_state(&k.conv3d_backward_weight);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), dy as u64);
    enc.set_buffer(2, Some(buffer), dw as u64);
    enc.set_bytes(3, 16, a.as_ptr() as *const _);
    enc.set_bytes(4, 16, b.as_ptr() as *const _);
    enc.set_bytes(5, 16, cc.as_ptr() as *const _);
    enc.set_bytes(6, 16, dd_p.as_ptr() as *const _);
    enc.set_bytes(7, 16, e.as_ptr() as *const _);
    enc.set_bytes(8, 8, f.as_ptr() as *const _);
    let c_in_per_g = c_in / groups.max(1);
    let total = (c_out * c_in_per_g * kd * kh * kw) as u64;
    let grid = metal::MTLSize {
        width: total.max(1),
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(total.max(1)),
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_conv2d_backward_input(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    dy: usize,
    w: usize,
    dx: usize,
    n: u32,
    c_in: u32,
    h: u32,
    w_in: u32,
    c_out: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    groups: u32,
) {
    let a: [u32; 4] = [n, c_in, h, w_in];
    let b: [u32; 4] = [c_out, h_out, w_out, kh];
    let cc: [u32; 4] = [kw, sh, sw, ph];
    let d: [u32; 4] = [pw, dh, dw, groups];
    enc.set_compute_pipeline_state(&k.conv2d_backward_input);
    enc.set_buffer(0, Some(buffer), dy as u64);
    enc.set_buffer(1, Some(buffer), w as u64);
    enc.set_buffer(2, Some(buffer), dx as u64);
    enc.set_bytes(3, 16, a.as_ptr() as *const _);
    enc.set_bytes(4, 16, b.as_ptr() as *const _);
    enc.set_bytes(5, 16, cc.as_ptr() as *const _);
    enc.set_bytes(6, 16, d.as_ptr() as *const _);
    let grid = metal::MTLSize {
        width: w_in as u64,
        height: h as u64,
        depth: (n * c_in) as u64,
    };
    let tg = metal::MTLSize {
        width: 8.min(w_in as u64),
        height: 8.min(h as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_conv2d_backward_weight(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    dy: usize,
    dw: usize,
    n: u32,
    c_in: u32,
    h: u32,
    w: u32,
    c_out: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw_dil: u32,
    groups: u32,
) {
    let a: [u32; 4] = [n, c_in, h, w];
    let b: [u32; 4] = [c_out, h_out, w_out, kh];
    let cc: [u32; 4] = [kw, sh, sw, ph];
    let d: [u32; 4] = [pw, dh, dw_dil, groups];
    let c_in_per_g = c_in / groups;
    enc.set_compute_pipeline_state(&k.conv2d_backward_weight);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), dy as u64);
    enc.set_buffer(2, Some(buffer), dw as u64);
    enc.set_bytes(3, 16, a.as_ptr() as *const _);
    enc.set_bytes(4, 16, b.as_ptr() as *const _);
    enc.set_bytes(5, 16, cc.as_ptr() as *const _);
    enc.set_bytes(6, 16, d.as_ptr() as *const _);
    let grid = metal::MTLSize {
        width: kw as u64,
        height: kh as u64,
        depth: (c_out * c_in_per_g) as u64,
    };
    let tg = metal::MTLSize {
        width: kw as u64,
        height: kh as u64,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

// Two-pass batch-parallel conv2d weight-grad. `part_off` is the conv-bwd
// scratch slot, sized (by conv_bwd_scratch_bytes) to hold N*C_out*c_in_per_g*
// kh*kw f32 partials. Pass 1 fills it (one thread per per-sample weight elem),
// pass 2 reduces over N. Both run in the same serial encoder, so pass 2 sees
// pass 1's writes with no explicit barrier.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_conv2d_backward_weight_2pass(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    part_off: usize,
    x: usize,
    dy: usize,
    dw: usize,
    n: u32,
    c_in: u32,
    h: u32,
    w: u32,
    c_out: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw_dil: u32,
    groups: u32,
) {
    let a: [u32; 4] = [n, c_in, h, w];
    let b: [u32; 4] = [c_out, h_out, w_out, kh];
    let cc: [u32; 4] = [kw, sh, sw, ph];
    let d: [u32; 4] = [pw, dh, dw_dil, groups];
    let c_in_per_g = c_in / groups;
    let wsz = c_in_per_g * kh * kw; // per-(n,co) slab
    let wslab = c_out * wsz; // per-sample slab

    // Pass 1: per-sample partials → scratch.
    enc.set_compute_pipeline_state(&k.conv2d_backward_weight_partial);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), dy as u64);
    enc.set_buffer(2, Some(buffer), part_off as u64);
    enc.set_bytes(3, 16, a.as_ptr() as *const _);
    enc.set_bytes(4, 16, b.as_ptr() as *const _);
    enc.set_bytes(5, 16, cc.as_ptr() as *const _);
    enc.set_bytes(6, 16, d.as_ptr() as *const _);
    let grid1 = metal::MTLSize {
        width: wsz as u64,
        height: c_out as u64,
        depth: n as u64,
    };
    let tgw = 8.min(wsz as u64).max(1);
    let tg1 = metal::MTLSize {
        width: tgw,
        height: 8.min(c_out as u64).max(1),
        depth: 1,
    };
    enc.dispatch_threads(grid1, tg1);

    // Pass 2: reduce partials over the batch → dw.
    let dims: [u32; 2] = [n, wslab];
    enc.set_compute_pipeline_state(&k.conv2d_backward_weight_reduce);
    enc.set_buffer(0, Some(buffer), part_off as u64);
    enc.set_buffer(1, Some(buffer), dw as u64);
    enc.set_bytes(2, 8, dims.as_ptr() as *const _);
    let grid2 = metal::MTLSize {
        width: wslab as u64,
        height: 1,
        depth: 1,
    };
    let tg2 = metal::MTLSize {
        width: 64.min(wslab as u64).max(1),
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid2, tg2);
}

pub(crate) fn encode_gather_axis(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    table: usize,
    idx: usize,
    dst: usize,
    outer: u32,
    axis_dim: u32,
    num_idx: u32,
    trailing: u32,
) {
    enc.set_compute_pipeline_state(&k.gather_axis);
    enc.set_buffer(0, Some(buffer), table as u64);
    enc.set_buffer(1, Some(buffer), idx as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &outer as *const u32 as *const _);
    enc.set_bytes(4, 4, &axis_dim as *const u32 as *const _);
    enc.set_bytes(5, 4, &num_idx as *const u32 as *const _);
    enc.set_bytes(6, 4, &trailing as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: trailing as u64,
        height: num_idx as u64,
        depth: outer as u64,
    };
    // Apple simdgroups are 32 lanes. The previous 8×8 threadgroup left
    // 75% of each simdgroup idle when the gather had a single row
    // (num_idx == 1, the embedding-lookup hot path). Pick the largest
    // axis as the threadgroup-x dimension and pack to 32 — keeps the
    // 2-D case fast for general gathers while making the embed lookup
    // ~4× more parallel per simdgroup.
    let tg_x = 32.min(trailing as u64).max(1);
    let tg_y = (32 / tg_x).clamp(1, num_idx as u64).max(1);
    let tg = metal::MTLSize {
        width: tg_x,
        height: tg_y,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// Swap of the last two axes with dense leading batch dims → `(batch, rows, cols)`.
pub(crate) fn detect_last2_batched_swap(
    out_dims: &[u32],
    in_strides: &[u32],
) -> Option<(u32, u32, u32)> {
    let rank = out_dims.len();
    if rank < 3 || in_strides.len() < rank {
        return None;
    }
    let rows = out_dims[rank - 1];
    let cols = out_dims[rank - 2];
    if in_strides[rank - 2] != 1 || in_strides[rank - 1] != cols {
        return None;
    }
    let mut tail = cols.saturating_mul(rows);
    if rank >= 3 && in_strides[rank - 3] != tail {
        return None;
    }
    for i in (0..rank.saturating_sub(3)).rev() {
        let expected = tail.saturating_mul(out_dims[i + 1].max(1));
        if in_strides[i] != expected {
            return None;
        }
        tail = expected;
    }
    let mut batch_u64 = 1u64;
    for &d in &out_dims[..rank - 2] {
        batch_u64 = batch_u64.saturating_mul(d.max(1) as u64);
    }
    if batch_u64 == 0 || batch_u64 > u32::MAX as u64 {
        return None;
    }
    Some((batch_u64 as u32, rows, cols))
}

/// `[B, A, C, D] → [B, C, A, D]` (perm `[0, 2, 1, 3]`).
pub(crate) fn detect_swap12_batched_trailing(
    out_dims: &[u32],
    in_strides: &[u32],
) -> Option<(u32, u32, u32, u32)> {
    if out_dims.len() != 4 || in_strides.len() != 4 {
        return None;
    }
    let batch = out_dims[0];
    let rows = out_dims[1];
    let cols = out_dims[2];
    let trail = out_dims[3];
    if batch == 0 || rows == 0 || cols == 0 || trail == 0 {
        return None;
    }
    if in_strides[3] != 1 {
        return None;
    }
    if in_strides[1] != trail {
        return None;
    }
    if in_strides[2] != rows.saturating_mul(trail) {
        return None;
    }
    if in_strides[0] != cols.saturating_mul(rows).saturating_mul(trail) {
        return None;
    }
    Some((batch, rows, cols, trail))
}

pub(crate) fn encode_transpose_swap12_batched_trailing(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    batch: u32,
    rows: u32,
    cols: u32,
    trail: u32,
) {
    let use_tiled = rows >= 32 && cols >= 32;
    let depth = (batch as u64).saturating_mul(trail as u64);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 4, &batch as *const u32 as *const _);
    enc.set_bytes(3, 4, &rows as *const u32 as *const _);
    enc.set_bytes(4, 4, &cols as *const u32 as *const _);
    enc.set_bytes(5, 4, &trail as *const u32 as *const _);
    if use_tiled {
        enc.set_compute_pipeline_state(&k.transpose_swap12_batched_trail_tiled_f32);
        let tg = metal::MTLSize {
            width: 32,
            height: 8,
            depth: 1,
        };
        let groups = metal::MTLSize {
            width: (rows as u64).div_ceil(32),
            height: (cols as u64).div_ceil(32),
            depth,
        };
        enc.dispatch_thread_groups(groups, tg);
        return;
    }
    enc.set_compute_pipeline_state(&k.transpose_swap12_batched_trail_f32);
    enc.dispatch_threads(
        metal::MTLSize {
            width: rows as u64,
            height: cols as u64,
            depth,
        },
        metal::MTLSize {
            width: 16.min(rows as u64),
            height: 16.min(cols as u64),
            depth: 1,
        },
    );
}

pub(crate) fn metal_host_slices_enabled() -> bool {
    matches!(
        std::env::var("RLX_METAL_HOST_SLICE").as_deref(),
        Ok("1") | Ok("true") | Ok("on")
    )
}

/// CPU unified-memory fallbacks for copy / elem / activation (debug only).
/// Default is GPU arena-base + u64 byte offsets (Task #50, >4 GiB arenas).
pub(crate) fn metal_host_fallback_enabled() -> bool {
    metal_host_slices_enabled() || rlx_ir::env::flag("RLX_METAL_HOST_FALLBACK")
}

/// Task #50: activations past 4 GiB need arena-base + u64 byte offsets.
pub(crate) const ARENA_LARGE_OFF: usize = 1usize << 32;

#[inline]
pub(crate) fn arena_off_large(off: usize) -> bool {
    use crate::thunk::{is_weight_off, raw_off};
    !is_weight_off(off) && raw_off(off) >= ARENA_LARGE_OFF
}

pub(crate) fn encode_transpose_2d(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    rows: u32,
    cols: u32,
) {
    let use_tiled = rows >= 32 && cols >= 32;
    enc.set_compute_pipeline_state(if use_tiled {
        &k.transpose_2d_tiled_f32
    } else {
        &k.transpose_2d_f32
    });
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 4, &rows as *const u32 as *const _);
    enc.set_bytes(3, 4, &cols as *const u32 as *const _);
    if use_tiled {
        // 32x32 tile, threadgroup (32,8).
        let tg = metal::MTLSize {
            width: 32,
            height: 8,
            depth: 1,
        };
        let groups = metal::MTLSize {
            width: (rows as u64).div_ceil(32),
            height: (cols as u64).div_ceil(32),
            depth: 1,
        };
        enc.dispatch_thread_groups(groups, tg);
    } else {
        let tg_w = k.transpose_2d_f32.thread_execution_width().min(cols as u64);
        let tg_h = (256 / tg_w.max(1)).min(rows as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: rows as u64,
                height: cols as u64,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: tg_h,
                depth: 1,
            },
        );
    }
}

pub(crate) fn encode_transpose_last2_batched(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    batch: u32,
    rows: u32,
    cols: u32,
) {
    let use_tiled = rows >= 32 && cols >= 32;
    enc.set_compute_pipeline_state(if use_tiled {
        &k.transpose_last2_batched_tiled_f32
    } else {
        &k.transpose_last2_batched_f32
    });
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 4, &batch as *const u32 as *const _);
    enc.set_bytes(3, 4, &rows as *const u32 as *const _);
    enc.set_bytes(4, 4, &cols as *const u32 as *const _);
    if use_tiled {
        let tg = metal::MTLSize {
            width: 32,
            height: 8,
            depth: 1,
        };
        let groups = metal::MTLSize {
            width: (rows as u64).div_ceil(32),
            height: (cols as u64).div_ceil(32),
            depth: batch as u64,
        };
        enc.dispatch_thread_groups(groups, tg);
    } else {
        let tg_w = k
            .transpose_last2_batched_f32
            .thread_execution_width()
            .min(rows as u64);
        let tg_h = (256 / tg_w.max(1)).min(cols as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: rows as u64,
                height: cols as u64,
                depth: batch as u64,
            },
            metal::MTLSize {
                width: tg_w,
                height: tg_h,
                depth: 1,
            },
        );
    }
}

pub(crate) fn encode_transpose(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    total: u32,
    out_dims: &[u32],
    in_strides: &[u32],
    half: bool,
) {
    let rank = out_dims.len() as u32;
    // Pack [out_dims..., in_strides...] into a single inline meta buffer.
    let mut meta: Vec<u32> = Vec::with_capacity(2 * out_dims.len());
    meta.extend_from_slice(out_dims);
    meta.extend_from_slice(in_strides);
    enc.set_compute_pipeline_state(if half {
        &k.transpose_nd_h
    } else {
        &k.transpose_nd
    });
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 4, &rank as *const u32 as *const _);
    enc.set_bytes(3, 4, &total as *const u32 as *const _);
    enc.set_bytes(4, (meta.len() * 4) as u64, meta.as_ptr() as *const _);
    let tg_w = k.transpose_nd.thread_execution_width().min(total as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: total as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_elementwise_region(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    len: u32,
    num_inputs: u32,
    num_steps: u32,
    dst: usize,
    input_offs: &[u32; 16],
    chain: &[u32; 128],
    scalar_input_mask: u32,
    input_modulus: &[u32; 16],
    prologue: u32,
    out_n: u32,
    out_c: u32,
    out_h: u32,
    out_w: u32,
    prologue_input: u32,
) {
    enc.set_compute_pipeline_state(&k.elementwise_region);
    enc.set_buffer(0, Some(buffer), 0);
    enc.set_bytes(
        1,
        std::mem::size_of::<u32>() as u64,
        &len as *const u32 as *const _,
    );
    enc.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &num_inputs as *const u32 as *const _,
    );
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &num_steps as *const u32 as *const _,
    );
    let dst_u32 = (dst / 4) as u32;
    enc.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &dst_u32 as *const u32 as *const _,
    );
    enc.set_bytes(
        5,
        (input_offs.len() * 4) as u64,
        input_offs.as_ptr() as *const _,
    );
    enc.set_bytes(6, (chain.len() * 4) as u64, chain.as_ptr() as *const _);
    enc.set_bytes(
        7,
        std::mem::size_of::<u32>() as u64,
        &scalar_input_mask as *const u32 as *const _,
    );
    enc.set_bytes(
        8,
        (input_modulus.len() * 4) as u64,
        input_modulus.as_ptr() as *const _,
    );
    enc.set_bytes(
        9,
        std::mem::size_of::<u32>() as u64,
        &prologue as *const u32 as *const _,
    );
    enc.set_bytes(
        10,
        std::mem::size_of::<u32>() as u64,
        &out_n as *const u32 as *const _,
    );
    enc.set_bytes(
        11,
        std::mem::size_of::<u32>() as u64,
        &out_c as *const u32 as *const _,
    );
    enc.set_bytes(
        12,
        std::mem::size_of::<u32>() as u64,
        &out_h as *const u32 as *const _,
    );
    enc.set_bytes(
        13,
        std::mem::size_of::<u32>() as u64,
        &out_w as *const u32 as *const _,
    );
    enc.set_bytes(
        14,
        std::mem::size_of::<u32>() as u64,
        &prologue_input as *const u32 as *const _,
    );
    if prologue != 0 && out_h > 0 && out_w > 0 {
        let grid = metal::MTLSize {
            width: out_w as u64,
            height: out_h as u64,
            depth: (out_n as u64) * (out_c as u64),
        };
        let tg = metal::MTLSize {
            width: 8.min(out_w as u64),
            height: 8.min(out_h as u64),
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
    } else {
        let tg_w = k
            .elementwise_region
            .thread_execution_width()
            .min(len as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: len as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
    }
}

pub(crate) fn encode_batch_elementwise_region(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    slice_len: u32,
    num_batch: u32,
    num_steps: u32,
    base_dst: usize,
    slice_elems: u32,
    batch_input_offs: &[u32; 64],
    chain: &[u32; 128],
    scalar_input_mask: u32,
    input_modulus: &[u32; 16],
) {
    enc.set_compute_pipeline_state(&k.batch_elementwise_region);
    enc.set_buffer(0, Some(buffer), 0);
    enc.set_bytes(
        1,
        std::mem::size_of::<u32>() as u64,
        &slice_len as *const u32 as *const _,
    );
    enc.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &num_batch as *const u32 as *const _,
    );
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &num_steps as *const u32 as *const _,
    );
    let base_dst_u32 = (base_dst / 4) as u32;
    enc.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &base_dst_u32 as *const u32 as *const _,
    );
    enc.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &slice_elems as *const u32 as *const _,
    );
    enc.set_bytes(
        6,
        (batch_input_offs.len() * 4) as u64,
        batch_input_offs.as_ptr() as *const _,
    );
    enc.set_bytes(7, (chain.len() * 4) as u64, chain.as_ptr() as *const _);
    enc.set_bytes(
        8,
        std::mem::size_of::<u32>() as u64,
        &scalar_input_mask as *const u32 as *const _,
    );
    enc.set_bytes(
        9,
        (input_modulus.len() * 4) as u64,
        input_modulus.as_ptr() as *const _,
    );
    let tg_w = k
        .batch_elementwise_region
        .thread_execution_width()
        .min(slice_len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: slice_len as u64,
            height: 1,
            depth: num_batch as u64,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_scatter_add(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    updates: usize,
    indices: usize,
    dst: usize,
    num_updates: u32,
    out_dim: u32,
    trailing: u32,
) {
    // Phase 0: zero the output buffer (out_dim * trailing u32 atomics).
    let out_total = out_dim * trailing;
    enc.set_compute_pipeline_state(&k.scatter_add_zero);
    enc.set_buffer(0, Some(buffer), dst as u64);
    enc.set_bytes(1, 4, &out_total as *const u32 as *const _);
    let tg_w0 = k
        .scatter_add_zero
        .thread_execution_width()
        .min(out_total as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: out_total as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w0,
            height: 1,
            depth: 1,
        },
    );

    // Phase 1: atomic accumulate.
    enc.set_compute_pipeline_state(&k.scatter_add_accumulate);
    enc.set_buffer(0, Some(buffer), updates as u64);
    enc.set_buffer(1, Some(buffer), indices as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &trailing as *const u32 as *const _);
    enc.set_bytes(4, 4, &num_updates as *const u32 as *const _);
    enc.set_bytes(5, 4, &out_dim as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: trailing as u64,
        height: num_updates as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 8.min(trailing as u64),
        height: 8.min(num_updates as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_grouped_matmul(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    input: usize,
    weight: usize,
    expert_idx: usize,
    dst: usize,
    m: u32,
    k_dim: u32,
    n: u32,
    num_experts: u32,
) {
    enc.set_compute_pipeline_state(&k.grouped_matmul);
    enc.set_buffer(0, Some(buffer), input as u64);
    enc.set_buffer(1, Some(buffer), weight as u64);
    enc.set_buffer(2, Some(buffer), expert_idx as u64);
    enc.set_buffer(3, Some(buffer), dst as u64);
    enc.set_bytes(4, 4, &m as *const u32 as *const _);
    enc.set_bytes(5, 4, &k_dim as *const u32 as *const _);
    enc.set_bytes(6, 4, &n as *const u32 as *const _);
    enc.set_bytes(7, 4, &num_experts as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: n as u64,
        height: m as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 8.min(n as u64),
        height: 8.min(m as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_topk(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    outer: u32,
    axis_dim: u32,
    k_val: u32,
) {
    enc.set_compute_pipeline_state(&k.topk_lastax);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 4, &axis_dim as *const u32 as *const _);
    enc.set_bytes(3, 4, &k_val as *const u32 as *const _);
    let tg_w = k.topk_lastax.thread_execution_width().min(outer as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: outer as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_reduce_axes(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    outer: u32,
    reduced: u32,
    inner: u32,
    op: rlx_ir::op::ReduceOp,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    use rlx_ir::op::ReduceOp;
    let op_kind: u32 = match op {
        ReduceOp::Sum => 0,
        ReduceOp::Mean => 1,
        ReduceOp::Max => 2,
        ReduceOp::Min => 3,
        ReduceOp::Prod => 4,
    };
    // SIMD variant for Sum/Mean when the reduced axis is long and there are few
    // outputs (the bias/beta grad-sum pattern): one 32-wide threadgroup per
    // output parallelizes the reduction the scalar kernel serializes.
    let use_simd = matches!(dt, HalfFlag::F32)
        && matches!(op, ReduceOp::Sum | ReduceOp::Mean)
        && reduced >= 64
        && (inner as u64 * outer as u64) <= reduced as u64
        && rlx_ir::env::flag("RLX_METAL_REDUCE_SIMD");
    if use_simd {
        enc.set_compute_pipeline_state(&k.reduce_axes_sum_simd);
        enc.set_buffer(0, Some(buffer), src as u64);
        enc.set_buffer(1, Some(buffer), dst as u64);
        enc.set_bytes(2, 4, &reduced as *const u32 as *const _);
        enc.set_bytes(3, 4, &inner as *const u32 as *const _);
        enc.set_bytes(4, 4, &op_kind as *const u32 as *const _);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: inner as u64 * outer as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        return;
    }
    let pipeline = match dt {
        HalfFlag::F32 => &k.reduce_axes,
        HalfFlag::F16 => &k.reduce_axes_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 4, &reduced as *const u32 as *const _);
    enc.set_bytes(3, 4, &inner as *const u32 as *const _);
    enc.set_bytes(4, 4, &op_kind as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: inner as u64,
        height: outer as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 16.min(inner as u64),
        height: 16.min(outer as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_compare(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    lhs: usize,
    rhs: usize,
    dst: usize,
    len: u32,
    op: rlx_ir::op::CmpOp,
    lhs_scalar: bool,
    rhs_scalar: bool,
) {
    use rlx_ir::op::CmpOp;
    let op_kind: u32 = match op {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Lt => 2,
        CmpOp::Le => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
    };
    let bcast = lhs_scalar || rhs_scalar;
    let pipeline = if bcast {
        &k.elem_compare_bcast
    } else {
        &k.elem_compare
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), lhs as u64);
    enc.set_buffer(1, Some(buffer), rhs as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &len as *const u32 as *const _);
    enc.set_bytes(4, 4, &op_kind as *const u32 as *const _);
    if bcast {
        let flags: u32 = u32::from(lhs_scalar) | (u32::from(rhs_scalar) << 1);
        enc.set_bytes(5, 4, &flags as *const u32 as *const _);
    }
    let tg_w = pipeline.thread_execution_width().min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_where(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    cond: usize,
    on_true: usize,
    on_false: usize,
    dst: usize,
    len: u32,
    cond_scalar: bool,
    true_scalar: bool,
    false_scalar: bool,
) {
    let bcast = cond_scalar || true_scalar || false_scalar;
    let pipeline = if bcast {
        &k.elem_where_bcast
    } else {
        &k.elem_where
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), cond as u64);
    enc.set_buffer(1, Some(buffer), on_true as u64);
    enc.set_buffer(2, Some(buffer), on_false as u64);
    enc.set_buffer(3, Some(buffer), dst as u64);
    enc.set_bytes(4, 4, &len as *const u32 as *const _);
    if bcast {
        let flags: u32 =
            u32::from(cond_scalar) | (u32::from(true_scalar) << 1) | (u32::from(false_scalar) << 2);
        enc.set_bytes(5, 4, &flags as *const u32 as *const _);
    }
    let tg_w = pipeline.thread_execution_width().min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_fma(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    a: usize,
    b: usize,
    c: usize,
    dst: usize,
    len: u32,
) {
    enc.set_compute_pipeline_state(&k.elem_fma);
    enc.set_buffer(0, Some(buffer), a as u64);
    enc.set_buffer(1, Some(buffer), b as u64);
    enc.set_buffer(2, Some(buffer), c as u64);
    enc.set_buffer(3, Some(buffer), dst as u64);
    enc.set_bytes(4, 4, &len as *const u32 as *const _);
    let tg_w = k.elem_fma.thread_execution_width().min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_relu_backward(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    dy: usize,
    dx: usize,
    len: u32,
) {
    enc.set_compute_pipeline_state(&k.relu_backward);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), dy as u64);
    enc.set_buffer(2, Some(buffer), dx as u64);
    enc.set_bytes(3, 4, &len as *const u32 as *const _);
    let tg_w = k.relu_backward.thread_execution_width().min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_activation_backward(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    dy: usize,
    dx: usize,
    len: u32,
    op: u32,
) {
    enc.set_compute_pipeline_state(&k.activation_backward);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), dy as u64);
    enc.set_buffer(2, Some(buffer), dx as u64);
    enc.set_bytes(3, 4, &len as *const u32 as *const _);
    enc.set_bytes(4, 4, &op as *const u32 as *const _);
    let tg_w = k
        .activation_backward
        .thread_execution_width()
        .min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_complex_norm_sq(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    len: u32,
) {
    enc.set_compute_pipeline_state(&k.complex_norm_sq);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 4, &len as *const u32 as *const _);
    let tg_w = k.complex_norm_sq.thread_execution_width().min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_complex_norm_sq_backward(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    z: usize,
    g: usize,
    dz: usize,
    len: u32,
) {
    enc.set_compute_pipeline_state(&k.complex_norm_sq_backward);
    enc.set_buffer(0, Some(buffer), z as u64);
    enc.set_buffer(1, Some(buffer), g as u64);
    enc.set_buffer(2, Some(buffer), dz as u64);
    enc.set_bytes(3, 4, &len as *const u32 as *const _);
    let tg_w = k
        .complex_norm_sq_backward
        .thread_execution_width()
        .min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn encode_conjugate_c64(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    len: u32,
) {
    enc.set_compute_pipeline_state(&k.conjugate_c64);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 4, &len as *const u32 as *const _);
    let tg_w = k.conjugate_c64.thread_execution_width().min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

#[repr(C)]
struct FftButterflyStageParams {
    batch: u32,
    n_fft: u32,
    stage: u32,
    n_half: u32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_fft_butterfly_stage(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    state: usize,
    out: usize,
    gate: usize,
    rev: usize,
    tw_re: usize,
    tw_im: usize,
    batch: u32,
    n_fft: u32,
    stage: u32,
) {
    let n_half = n_fft / 2;
    if batch == 0 || n_half == 0 {
        return;
    }
    let p = FftButterflyStageParams {
        batch,
        n_fft,
        stage,
        n_half,
    };
    enc.set_compute_pipeline_state(&k.fft_butterfly_stage);
    enc.set_buffer(0, Some(buffer), state as u64);
    enc.set_buffer(1, Some(buffer), out as u64);
    enc.set_buffer(2, Some(buffer), gate as u64);
    enc.set_buffer(3, Some(buffer), rev as u64);
    enc.set_buffer(4, Some(buffer), tw_re as u64);
    enc.set_buffer(5, Some(buffer), tw_im as u64);
    enc.set_bytes(
        6,
        std::mem::size_of::<FftButterflyStageParams>() as u64,
        &p as *const FftButterflyStageParams as *const _,
    );
    let tg_w = k
        .fft_butterfly_stage
        .thread_execution_width()
        .min(n_half as u64)
        .max(1);
    enc.dispatch_threads(
        metal::MTLSize {
            width: n_half as u64,
            height: batch as u64,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

#[repr(C)]
struct FakeQuantizeParams {
    n: u32,
    chan_dim: u32,
    inner: u32,
    q_max: f32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_fake_quantize_fixed(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    scale: usize,
    dst: usize,
    n: u32,
    chan_dim: u32,
    inner: u32,
    q_max: f32,
) {
    if n == 0 {
        return;
    }
    let p = FakeQuantizeParams {
        n,
        chan_dim,
        inner,
        q_max,
    };
    enc.set_compute_pipeline_state(&k.fake_quantize_fixed);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), scale as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(
        3,
        std::mem::size_of::<FakeQuantizeParams>() as u64,
        &p as *const FakeQuantizeParams as *const _,
    );
    let tg_w = k.fake_quantize_fixed.thread_execution_width().min(n as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: n as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_fake_quantize_perbatch(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    n: u32,
    chan_dim: u32,
    inner: u32,
    q_max: f32,
) {
    if n == 0 || chan_dim == 0 {
        return;
    }
    let p = FakeQuantizeParams {
        n,
        chan_dim,
        inner,
        q_max,
    };
    enc.set_compute_pipeline_state(&k.fake_quantize_perbatch);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(
        2,
        std::mem::size_of::<FakeQuantizeParams>() as u64,
        &p as *const FakeQuantizeParams as *const _,
    );
    let tg_w = k
        .fake_quantize_perbatch
        .thread_execution_width()
        .min(chan_dim as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: chan_dim as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

/// Standalone softmax: one threadgroup per row, in-place exp+normalize.
/// Threadgroup size must be a power of 2 and ≤256 (the kernel's reduction
/// buffer). Picks the largest pow2 ≤ cols, capped at 256.
pub(crate) fn encode_softmax(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    data: usize,
    rows: u32,
    cols: u32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F32 => &k.softmax_lastax,
        HalfFlag::F16 => &k.softmax_lastax_h,
    };
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= cols as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    if rlx_ir::env::flag("RLX_METAL_SOFTMAX_TRACE") {
        eprintln!("[softmax-trace] rows={rows} cols={cols} tg_w={tg_w} dt={dt:?}");
    }
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), data as u64);
    enc.set_bytes(
        1,
        std::mem::size_of::<u32>() as u64,
        &cols as *const u32 as *const _,
    );
    // One threadgroup per row via dispatch_thread_groups. `dispatch_threads`
    // with a row-packed grid (width = tg_w * rows) is what this used to do,
    // matching the "pack rows along width, threadgroup_position_in_grid.x is
    // the row index" trick used elsewhere (encode_layer_norm, etc.) — but for
    // this specific one-threadgroup-per-row reduction shape it intermittently
    // corrupted a handful of output rows (observed as scattered NaNs in
    // Softmax rows immediately following a clean, NaN-free MatMul input;
    // reproduced in F5-TTS DiT attention on Apple Silicon Metal, ~15-20% of
    // dispatches). `dispatch_thread_groups` with an explicit threadgroup
    // count is the uniform/reliable dispatch form and is what every other
    // one-threadgroup-per-row reduction kernel here already uses
    // (encode_layer_norm, encode_rms_norm, encode_fused_residual_ln) — only
    // encode_softmax and the softmax_cross_entropy_* encoders still used the
    // buggy dispatch_threads form; all are fixed to match.
    let groups = metal::MTLSize {
        width: rows as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(groups, tg);
}

/// Fused dense softmax cross-entropy: one threadgroup per row, three
/// threadgroup reductions (row max, Σexp, Σtargets·logits). `cols` is
/// the class count C. Threadgroup width is the largest pow2 ≤ cols,
/// capped at 256 (the kernel's reduction buffer). f32 only.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_softmax_cross_entropy_dense(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    logits: usize,
    targets: usize,
    dst: usize,
    rows: u32,
    cols: u32,
) {
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= cols as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    enc.set_compute_pipeline_state(&k.softmax_cross_entropy_dense);
    enc.set_buffer(0, Some(buffer), logits as u64);
    enc.set_buffer(1, Some(buffer), targets as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &cols as *const u32 as *const _,
    );
    // One threadgroup per row via dispatch_thread_groups (not
    // dispatch_threads with a row-packed grid — see encode_softmax).
    let groups = metal::MTLSize {
        width: rows as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(groups, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_softmax_cross_entropy_with_logits(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    logits: usize,
    labels: usize,
    dst: usize,
    rows: u32,
    cols: u32,
) {
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= cols as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    enc.set_compute_pipeline_state(&k.softmax_cross_entropy_with_logits);
    enc.set_buffer(0, Some(buffer), logits as u64);
    enc.set_buffer(1, Some(buffer), labels as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &cols as *const u32 as *const _);
    // One threadgroup per row via dispatch_thread_groups (see encode_softmax).
    let groups = metal::MTLSize {
        width: rows as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(groups, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_softmax_cross_entropy_backward(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    logits: usize,
    labels: usize,
    d_loss: usize,
    dlogits: usize,
    rows: u32,
    cols: u32,
) {
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= cols as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    enc.set_compute_pipeline_state(&k.softmax_cross_entropy_backward);
    enc.set_buffer(0, Some(buffer), logits as u64);
    enc.set_buffer(1, Some(buffer), labels as u64);
    enc.set_buffer(2, Some(buffer), d_loss as u64);
    enc.set_buffer(3, Some(buffer), dlogits as u64);
    enc.set_bytes(4, 4, &cols as *const u32 as *const _);
    // One threadgroup per row via dispatch_thread_groups (see encode_softmax).
    let groups = metal::MTLSize {
        width: rows as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(groups, tg);
}

pub(crate) fn metal_concat_multi_enabled() -> bool {
    !rlx_ir::env::flag("RLX_METAL_CONCAT_MULTI")
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ConcatSegGpu {
    // u64 because for ≥4 GB models the source byte offset exceeds u32 and
    // u32-truncation wrap-around made `repeat_kv` write to the wrong slot,
    // leaving K_rep / V_rep as zeros and SDPA output as zero (task #50).
    pub(crate) src: u64,
    pub(crate) dst_col: u32,
    pub(crate) len: u32,
}

/// Dispatch a concat-along-last-axis. Uses one multi-segment kernel when possible.
/// Mid-axis concat (inner > 1) encoded entirely into the live command buffer
/// — one 1D dispatch per input segment, NO commit/wait. Replaces the
/// per-concat `commit + wait_until_completed` host-copy fallback that
/// serialized a decode step into one GPU submission per concat (the dominant
/// Metal decode cost on KV caches). Offsets are element offsets within the
/// f32/f16 arena; the kernel takes ulong byte offsets so it is correct on
/// >4 GiB arenas (task #50). `dst`/`inputs` offsets are byte offsets into the
/// arena (as stored in the thunk).
pub(crate) fn encode_concat_midaxis(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    dst: usize,
    outer: u32,
    dst_axis: u32,
    inner: u32,
    dt: crate::thunk::HalfFlag,
    inputs: &[(usize, u32)],
    input_dts: &[crate::thunk::HalfFlag],
) {
    use crate::thunk::HalfFlag;
    // `dst`/`src_off` are byte offsets into the arena (the kernel adds them to
    // a char* base, then indexes elements), so no element-size scaling here.
    let inner_e = inner as u64;
    let mut axis_off: u32 = 0;
    for (seg_i, &(src_off, src_axis)) in inputs.iter().enumerate() {
        let total = outer as u64 * src_axis as u64 * inner_e;
        if total == 0 {
            axis_off += src_axis;
            continue;
        }
        // Per-segment pipeline: when a source dtype differs from the output
        // (an F32 segment into an F16 concat — e.g. an F32 `k_rope` after a pass
        // dropped its cast), CONVERT (read f32, write half) instead of reading
        // f32 bytes as half (→ inf/NaN saturation).
        let src_dt = input_dts.get(seg_i).copied().unwrap_or(dt);
        let pipeline = match (src_dt, dt) {
            (HalfFlag::F32, HalfFlag::F16) => &k.concat_midaxis_seg_f32_to_f16,
            (HalfFlag::F16, HalfFlag::F32) => &k.concat_midaxis_seg_f16_to_f32,
            (HalfFlag::F16, HalfFlag::F16) => &k.concat_midaxis_seg_h,
            (HalfFlag::F32, HalfFlag::F32) => &k.concat_midaxis_seg,
        };
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(buffer), 0);
        let dst_byte = dst as u64; // already a byte offset
        let src_byte = src_off as u64;
        enc.set_bytes(1, 8, &dst_byte as *const u64 as *const _);
        enc.set_bytes(2, 8, &src_byte as *const u64 as *const _);
        enc.set_bytes(3, 4, &outer as *const u32 as *const _);
        enc.set_bytes(4, 4, &dst_axis as *const u32 as *const _);
        enc.set_bytes(5, 4, &src_axis as *const u32 as *const _);
        enc.set_bytes(6, 4, &inner as *const u32 as *const _);
        enc.set_bytes(7, 4, &axis_off as *const u32 as *const _);
        let tg = 256u64.min(total);
        let grid = metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let tgs = metal::MTLSize {
            width: tg.max(1),
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tgs);
        axis_off += src_axis;
    }
}

pub(crate) fn encode_concat_lastax(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    dst: usize,
    outer: u32,
    dst_axis: u32,
    dt: crate::thunk::HalfFlag,
    inputs: &[(usize, u32)],
) {
    use crate::thunk::HalfFlag;
    // Historically `concat_lastax_multi{,4}` was reported to mis-copy beyond 8
    // segments on Apple GPUs and we fell through to the per-segment kernels for
    // GQA `repeat_kv` (16 head slices). Per-segment fallback uses `set_buffer`
    // at large arena offsets which silently drops writes on ≥4 GB models
    // (task #50). The multi kernel takes byte offsets from `ConcatSeg` (now
    // u64) and binds `arena` at offset 0 — works at any offset.
    if inputs.len() >= 2
        && inputs.len() <= NARROW_BATCH_MAX
        && matches!(dt, HalfFlag::F32)
        && metal_concat_multi_enabled()
    {
        let mut cum = 0u32;
        let segs: Vec<ConcatSegGpu> = inputs
            .iter()
            .map(|&(src_off, src_axis)| {
                let seg = ConcatSegGpu {
                    src: src_off as u64,
                    dst_col: cum,
                    len: src_axis,
                };
                cum += src_axis;
                seg
            })
            .collect();
        let num_seg = segs.len() as u32;
        let max_len = segs.iter().map(|s| s.len).max().unwrap_or(0);
        let use_vec4 = dst_axis.is_multiple_of(4)
            && cum == dst_axis
            && segs
                .iter()
                .all(|s| (s.dst_col % 4) == 0 && (s.len % 4) == 0 && s.len >= 4);
        if use_vec4 {
            let dst_axis4 = dst_axis / 4;
            let max_len4 = max_len / 4;
            enc.set_compute_pipeline_state(&k.concat_lastax_multi4);
            enc.set_buffer(0, Some(buffer), 0);
            let dst_u64 = dst as u64;
            enc.set_bytes(1, 8, &dst_u64 as *const u64 as *const _);
            enc.set_bytes(2, 4, &outer as *const u32 as *const _);
            enc.set_bytes(3, 4, &dst_axis4 as *const u32 as *const _);
            enc.set_bytes(4, 4, &num_seg as *const u32 as *const _);
            enc.set_bytes(
                5,
                (segs.len() * std::mem::size_of::<ConcatSegGpu>()) as u64,
                segs.as_ptr() as *const _,
            );
            let grid = metal::MTLSize {
                width: max_len4 as u64,
                height: outer as u64,
                depth: num_seg as u64,
            };
            // Task #50: total threads per threadgroup must be ≤ 1024 on
            // Apple Silicon. Width×height×depth previously was 64×4×num_seg —
            // exceeds the cap once num_seg≥5 (GQA repeat_kv concats 16 head
            // slices). Metal silently fails the dispatch when over the cap,
            // leaving the destination buffer zero, which manifested as the
            // long-standing K_rep / V_rep zero bug on Gemma 4 12B SWA layers.
            let tg_depth = (1024u64 / (64 * 4)).min(num_seg as u64).max(1);
            let tg = metal::MTLSize {
                width: 64.min(max_len4 as u64),
                height: 4.min(outer as u64),
                depth: tg_depth,
            };
            enc.dispatch_threads(grid, tg);
            return;
        }
        enc.set_compute_pipeline_state(&k.concat_lastax_multi);
        enc.set_buffer(0, Some(buffer), 0);
        let dst_u64 = dst as u64;
        enc.set_bytes(1, 8, &dst_u64 as *const u64 as *const _);
        enc.set_bytes(2, 4, &outer as *const u32 as *const _);
        enc.set_bytes(3, 4, &dst_axis as *const u32 as *const _);
        enc.set_bytes(4, 4, &num_seg as *const u32 as *const _);
        enc.set_bytes(
            5,
            (segs.len() * std::mem::size_of::<ConcatSegGpu>()) as u64,
            segs.as_ptr() as *const _,
        );
        let grid = metal::MTLSize {
            width: max_len as u64,
            height: outer as u64,
            depth: num_seg as u64,
        };
        // Task #50: cap total threads per threadgroup at 1024.
        let tg_depth = (1024u64 / (32 * 8)).min(num_seg as u64).max(1);
        let tg = metal::MTLSize {
            width: 32.min(max_len as u64),
            height: 8.min(outer as u64),
            depth: tg_depth,
        };
        enc.dispatch_threads(grid, tg);
        return;
    }

    let pipeline = match dt {
        HalfFlag::F32 => &k.concat_segment_lastax,
        HalfFlag::F16 => &k.concat_segment_lastax_h,
    };
    let mut cum: u32 = 0;
    for &(src_off, src_axis) in inputs {
        let use_vec4 = matches!(dt, HalfFlag::F32)
            && (src_axis % 4) == 0
            && dst_axis.is_multiple_of(4)
            && cum.is_multiple_of(4)
            && src_axis >= 4;
        if use_vec4 {
            let src_axis4 = src_axis / 4;
            let dst_axis4 = dst_axis / 4;
            let dst_col4 = cum / 4;
            enc.set_compute_pipeline_state(&k.concat_segment_lastax4);
            // Large set_buffer offsets silently drop kernel writes on
            // M-series at offsets ≥ ~4 GB (task #50). Bind to arena base
            // and pass byte offsets as ulong constants in buffers 6/7.
            enc.set_buffer(0, Some(buffer), 0);
            enc.set_buffer(1, Some(buffer), 0);
            enc.set_bytes(2, 4, &outer as *const u32 as *const _);
            enc.set_bytes(3, 4, &src_axis4 as *const u32 as *const _);
            enc.set_bytes(4, 4, &dst_axis4 as *const u32 as *const _);
            enc.set_bytes(5, 4, &dst_col4 as *const u32 as *const _);
            let src_off_u64 = src_off as u64;
            let dst_u64 = dst as u64;
            enc.set_bytes(6, 8, &src_off_u64 as *const u64 as *const _);
            enc.set_bytes(7, 8, &dst_u64 as *const u64 as *const _);
            let grid = metal::MTLSize {
                width: src_axis4 as u64,
                height: outer as u64,
                depth: 1,
            };
            let tg = metal::MTLSize {
                width: 64.min(src_axis4 as u64),
                height: 4.min(outer as u64),
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
        } else {
            enc.set_compute_pipeline_state(pipeline);
            // Task #50: same arena-base + ulong-offset workaround.
            enc.set_buffer(0, Some(buffer), 0);
            enc.set_buffer(1, Some(buffer), 0);
            enc.set_bytes(2, 4, &outer as *const u32 as *const _);
            enc.set_bytes(3, 4, &src_axis as *const u32 as *const _);
            enc.set_bytes(4, 4, &dst_axis as *const u32 as *const _);
            enc.set_bytes(5, 4, &cum as *const u32 as *const _);
            let src_off_u64 = src_off as u64;
            let dst_u64 = dst as u64;
            enc.set_bytes(6, 8, &src_off_u64 as *const u64 as *const _);
            enc.set_bytes(7, 8, &dst_u64 as *const u64 as *const _);
            let grid = metal::MTLSize {
                width: src_axis as u64,
                height: outer as u64,
                depth: 1,
            };
            let tg = metal::MTLSize {
                width: 16.min(src_axis as u64),
                height: 16.min(outer as u64),
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
        }
        cum += src_axis;
    }
}

/// Dispatch a FusedSwiGLU kernel. Picks the variant matching `(src_dt, dst_dt)`:
/// f32→f32, f16→f16, f32→f16 (cast), f16→f32 (cast).
pub(crate) fn encode_fused_swiglu(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    n_half: u32,
    total: u32,
    src_dt: crate::thunk::HalfFlag,
    dst_dt: crate::thunk::HalfFlag,
    gate_first: bool,
) {
    use crate::thunk::HalfFlag;
    let gate_first_u32 = u32::from(gate_first);
    let pipeline = match (src_dt, dst_dt) {
        (HalfFlag::F32, HalfFlag::F32) => &k.fused_swiglu,
        (HalfFlag::F16, HalfFlag::F16) => &k.fused_swiglu_h,
        (HalfFlag::F32, HalfFlag::F16) => &k.fused_swiglu_cast_f32_to_f16,
        (HalfFlag::F16, HalfFlag::F32) => &k.fused_swiglu_cast_f16_to_f32,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &n_half as *const u32 as *const _,
    );
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &total as *const u32 as *const _,
    );
    enc.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &gate_first_u32 as *const u32 as *const _,
    );
    let tg_w = pipeline.thread_execution_width().min(total as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: total as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}
