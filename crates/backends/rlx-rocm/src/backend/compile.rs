// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `compile` — extracted from the `backend` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::arena::{
    Arena, CastLower, HalfDtype, arena_lane_count, classify_cast, plan_f32_uniform,
};
use crate::device::{RocmContext, rocm_blas, rocm_blas_lt, rocm_context, rocm_dnn};
use crate::hip::{HipBuffer, HipDeviceptr};
use crate::hipblas::{
    HipblasComputeType, HipblasContext, HipblasDatatype, HipblasOperation, hipblas_gemm_default,
};
use crate::hipblaslt::HipblasLtContext;
use crate::host_staging::F32HostSlot;
use crate::miopen::MiopenContext;
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::{Graph, NodeId, Op};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use super::*;

/// Resolve `RLX_ROCM_EXEC` (`graph` | `ms<N>` | `stream`) over the caller's
/// default, so hipGraph capture / multi-stream are reachable without new API.
fn resolve_rocm_exec_mode(default: ExecMode) -> ExecMode {
    match rlx_ir::env::var("RLX_ROCM_EXEC") {
        Some(v) => {
            let v = v.trim().to_ascii_lowercase();
            if v == "graph" {
                ExecMode::Graph
            } else if v == "stream" {
                ExecMode::Stream
            } else if let Some(n) = v.strip_prefix("ms") {
                ExecMode::MultiStream(n.parse().unwrap_or(2).max(1))
            } else {
                default
            }
        }
        None => default,
    }
}

impl RocmExecutable {
    /// JIT compile, stream-mode execution. Default entry point.
    pub fn compile(graph: Graph) -> Self {
        Self::compile_with_rng(
            graph,
            CompileMode::Jit,
            ExecMode::Stream,
            rlx_ir::RngOptions::default(),
        )
    }

    pub fn compile_rng(graph: Graph, rng: rlx_ir::RngOptions) -> Self {
        Self::compile_with_rng(graph, CompileMode::Jit, ExecMode::Stream, rng)
    }

    /// Compile with explicit RNG policy (used by [`rlx-runtime`]).
    pub fn compile_with_rng(
        graph: Graph,
        compile_mode: CompileMode,
        exec_mode: ExecMode,
        rng: rlx_ir::RngOptions,
    ) -> Self {
        // `RLX_ROCM_EXEC` lets any caller opt into hipGraph capture (`graph`) or
        // multi-stream (`msN`) without new API — the default Session path hardcodes
        // Stream. Graph REPLAY collapses a decode forward's ~1400 per-op kernel
        // launches into a single graph launch: the big win for launch-overhead-bound
        // (memory-bound) inference. Capture-unsafe schedules fall back automatically.
        let exec_mode = resolve_rocm_exec_mode(exec_mode);
        let ctx = rocm_context().expect("rlx-rocm: no HIP runtime available");

        if compile_mode == CompileMode::Aot {
            crate::kernels::prewarm_all(&ctx);
        }

        // Decompose composed ops we don't yet have native kernels for
        // (FusedMatMulBiasAct, canonical DotGeneral) into primitives
        // before memory planning.
        let graph = crate::unfuse::unfuse(graph);

        let dequant_scratch = crate::gguf_gpu::dequant_gguf_scratch_bytes(&graph);
        let mut plan = plan_f32_uniform(&graph, 16);
        let dequant_scratch_off = if dequant_scratch > 0 {
            let aligned = plan.arena_size.div_ceil(16) * 16;
            plan.arena_size = aligned + dequant_scratch;
            aligned
        } else {
            0
        };
        let mut arena = Arena::from_plan(&ctx, &plan);
        for node in graph.nodes() {
            let slot_bytes = node
                .shape
                .size_bytes()
                .unwrap_or_else(|| node.shape.num_elements().unwrap_or(0) * 4);
            arena.set_actual_len(node.id, slot_bytes);
        }

        let mut input_offsets = HashMap::new();
        let mut param_offsets = HashMap::new();
        for node in graph.nodes() {
            match &node.op {
                Op::Input { name } => {
                    input_offsets.insert(name.clone(), node.id);
                }
                Op::Param { name } => {
                    param_offsets.insert(name.clone(), node.id);
                }
                _ => {}
            }
        }

        // Initialise Constants directly into the arena.
        let arena_ptr = arena.buffer.ptr;
        for node in graph.nodes() {
            if let Op::Constant { data } = &node.op
                && arena.has(node.id)
                && !data.is_empty()
            {
                let bytes_to_write = data.len().min(arena.len_of(node.id));
                let n_f32 = bytes_to_write / 4;
                let f32_view: &[f32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n_f32) };
                let off_f32 = arena.offset(node.id) / 4;
                upload_to_arena(&ctx, arena_ptr, off_f32, f32_view);
            }
        }

        let mut schedule: Vec<Step> = Vec::new();
        let mut meta_buffers: Vec<HipBuffer<u32>> = Vec::new();
        let mut packed_bshd_attn: HashMap<NodeId, (NodeId, u32)> = HashMap::new();
        if !rlx_ir::env::flag("RLX_ROCM_NO_PACKED_BSHD_ATTN") {
            for node in graph.nodes() {
                let Op::Attention { .. } = &node.op else {
                    continue;
                };
                if node.inputs.len() < 3 {
                    continue;
                }
                if let Some((parent, head_width, _)) = rlx_ir::detect_packed_bshd_qkv_attention(
                    &graph,
                    node.inputs[0],
                    node.inputs[1],
                    node.inputs[2],
                ) {
                    packed_bshd_attn.insert(node.id, (parent, head_width as u32));
                }
            }
        }
        for node in graph.nodes() {
            let elems = node.shape.num_elements().unwrap_or(0) as u32;
            match &node.op {
                Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => continue,
                Op::Reshape { .. } | Op::StopGradient => {
                    // No-op: arena planner aliased the slot. StopGradient is a
                    // pure forward identity (AD already consumed its semantics).
                }
                Op::Cast { to } => {
                    // Identity relabels are arena-aliased (no-op). float→int /
                    // →Bool casts got their own f32-sized slot and need a real
                    // conversion via the unary kernel (value stored as an f32
                    // lane on the f32-uniform arena).
                    let src_dtype = graph.node(node.inputs[0]).shape.dtype();
                    match classify_cast(src_dtype, *to) {
                        CastLower::Identity => {}
                        CastLower::Kernel(op) => {
                            schedule.push(Step::Unary {
                                n: elems,
                                in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                                out_off: (arena.offset(node.id) / 4) as u32,
                                op,
                            });
                        }
                        CastLower::Complex(mode) => {
                            // Simulated-complex lane move (real↔C64, real↔C128,
                            // C64↔C128) via the shared `complex_cast` kernel.
                            // `elems` is the complex-element count; the kernel
                            // re-pairs the interleaved f32 lanes.
                            schedule.push(Step::ComplexCast {
                                n: elems,
                                in_byte_off: arena.offset(node.inputs[0]) as u32,
                                out_byte_off: arena.offset(node.id) as u32,
                                mode,
                            });
                        }
                        CastLower::Reject => panic!(
                            "rlx-rocm: Cast {src_dtype:?} → {to:?} touches F64, which has \
                             no f32-lane storage in this arena — run it on CPU"
                        ),
                    }
                }
                Op::ScaledMatMul {
                    lhs_format,
                    rhs_format,
                    scale_layout,
                    has_bias,
                } => {
                    let out_dims = node.shape.dims();
                    let m = out_dims[0].unwrap_static() as u32;
                    let n = out_dims[1].unwrap_static() as u32;
                    let k = graph.node(node.inputs[0]).shape.dims()[1].unwrap_static() as u32;
                    let bias_byte = if *has_bias {
                        arena.offset(node.inputs[4]) as u32
                    } else {
                        0
                    };
                    let native = lhs_format.is_native_fp8()
                        && rhs_format.is_native_fp8()
                        && matches!(scale_layout, rlx_ir::ScaleLayout::PerTensor);
                    if native {
                        schedule.push(Step::ScaledMatMul {
                            m,
                            k,
                            n,
                            lhs_byte_off: arena.offset(node.inputs[0]) as u32,
                            rhs_byte_off: arena.offset(node.inputs[1]) as u32,
                            lhs_scale_byte_off: arena.offset(node.inputs[2]) as u32,
                            rhs_scale_byte_off: arena.offset(node.inputs[3]) as u32,
                            out_byte_off: arena.offset(node.id) as u32,
                            has_bias: u32::from(*has_bias),
                            bias_byte_off: bias_byte,
                            lhs_e5m2: u32::from(*lhs_format == rlx_ir::ScaledFormat::F8E5M2),
                            rhs_e5m2: u32::from(*rhs_format == rlx_ir::ScaledFormat::F8E5M2),
                        });
                    } else {
                        let (scale_mode, block) = scale_layout.mode_block();
                        schedule.push(Step::ScaledMatMulDecode {
                            m,
                            k,
                            n,
                            lhs_byte_off: arena.offset(node.inputs[0]) as u32,
                            rhs_byte_off: arena.offset(node.inputs[1]) as u32,
                            lhs_scale_byte_off: arena.offset(node.inputs[2]) as u32,
                            rhs_scale_byte_off: arena.offset(node.inputs[3]) as u32,
                            out_off_f32: (arena.offset(node.id) / 4) as u32,
                            lhs_fmt: lhs_format.kernel_id(),
                            rhs_fmt: rhs_format.kernel_id(),
                            scale_mode,
                            block,
                            has_bias: u32::from(*has_bias),
                            bias_off_f32: bias_byte / 4,
                        });
                    }
                }
                Op::ScaledGroupedMatMul {
                    lhs_format,
                    rhs_format,
                    scale_layout,
                    has_bias,
                } => {
                    // input [M,K], weight [E,N,K] (TN), expert_idx [M]; out [M,N].
                    // Memory-sane on-device decode-GEMM: only the routed expert's
                    // FP4 codes are read per token (no f32 weight materialization).
                    let in_dims = graph.node(node.inputs[0]).shape.dims();
                    let w_dims = graph.node(node.inputs[1]).shape.dims();
                    let m = in_dims[0].unwrap_static() as u32;
                    let k = in_dims[1].unwrap_static() as u32;
                    let ne = w_dims[0].unwrap_static() as u32;
                    let n = w_dims[w_dims.len() - 2].unwrap_static() as u32;
                    let bias_byte = if *has_bias {
                        arena.offset(node.inputs[5]) as u32
                    } else {
                        0
                    };
                    let (scale_mode, block) = scale_layout.mode_block();
                    schedule.push(Step::ScaledGroupedMatMulDecode {
                        m,
                        k,
                        n,
                        num_experts: ne,
                        input_byte_off: arena.offset(node.inputs[0]) as u32,
                        weight_byte_off: arena.offset(node.inputs[1]) as u32,
                        input_scale_byte_off: arena.offset(node.inputs[2]) as u32,
                        weight_scale_byte_off: arena.offset(node.inputs[3]) as u32,
                        idx_off_f32: (arena.offset(node.inputs[4]) / 4) as u32,
                        out_off_f32: (arena.offset(node.id) / 4) as u32,
                        bias_off_f32: bias_byte / 4,
                        lhs_fmt: lhs_format.kernel_id(),
                        rhs_fmt: rhs_format.kernel_id(),
                        scale_mode,
                        block,
                        has_bias: u32::from(*has_bias),
                    });
                }
                Op::ScaledQuantScale {
                    format,
                    scale_layout,
                } => {
                    let x_id = node.inputs[0];
                    if format.is_native_fp8()
                        && matches!(scale_layout, rlx_ir::ScaleLayout::PerTensor)
                    {
                        let n = graph.node(x_id).shape.num_elements().unwrap() as u32;
                        schedule.push(Step::ScaledQuantScale {
                            x_off_f32: (arena.offset(x_id) / 4) as u32,
                            scale_off_f32: (arena.offset(node.id) / 4) as u32,
                            n,
                            max_finite: format.max_finite(),
                        });
                    } else {
                        let xs = graph.node(x_id).shape.dims();
                        let cols = xs[xs.len() - 1].unwrap_static() as u32;
                        let rows =
                            graph.node(x_id).shape.num_elements().unwrap() as u32 / cols.max(1);
                        let (scale_mode, block) = scale_layout.mode_block();
                        schedule.push(Step::ScaledQuantScaleGeneral {
                            x_off_f32: (arena.offset(x_id) / 4) as u32,
                            scale_byte_off: arena.offset(node.id) as u32,
                            rows,
                            cols,
                            fmt: format.kernel_id(),
                            scale_mode,
                            block,
                        });
                    }
                }
                Op::ScaledQuantize {
                    format,
                    scale_layout,
                } => {
                    let x_id = node.inputs[0];
                    let scale_id = node.inputs[1];
                    if format.is_native_fp8()
                        && matches!(scale_layout, rlx_ir::ScaleLayout::PerTensor)
                    {
                        let n = graph.node(x_id).shape.num_elements().unwrap() as u32;
                        schedule.push(Step::ScaledQuantizeFp8 {
                            x_off_f32: (arena.offset(x_id) / 4) as u32,
                            scale_off_f32: (arena.offset(scale_id) / 4) as u32,
                            out_byte_off: arena.offset(node.id) as u32,
                            n,
                            e5m2: u32::from(*format == rlx_ir::ScaledFormat::F8E5M2),
                        });
                    } else {
                        let xs = graph.node(x_id).shape.dims();
                        let cols = xs[xs.len() - 1].unwrap_static() as u32;
                        let rows =
                            graph.node(x_id).shape.num_elements().unwrap() as u32 / cols.max(1);
                        let (scale_mode, block) = scale_layout.mode_block();
                        schedule.push(Step::ScaledQuantizeGeneral {
                            x_off_f32: (arena.offset(x_id) / 4) as u32,
                            scale_byte_off: arena.offset(scale_id) as u32,
                            out_byte_off: arena.offset(node.id) as u32,
                            rows,
                            cols,
                            fmt: format.kernel_id(),
                            scale_mode,
                            block,
                        });
                    }
                }
                Op::ScaledDequantize {
                    format,
                    scale_layout,
                } => {
                    // codes (U8, input 0) + scale (input 1) → f32; one general
                    // kernel covers all layouts. Shape follows the codes.
                    let codes_id = node.inputs[0];
                    let scale_id = node.inputs[1];
                    let xs = graph.node(codes_id).shape.dims();
                    let cols = xs[xs.len() - 1].unwrap_static() as u32;
                    let rows =
                        graph.node(codes_id).shape.num_elements().unwrap() as u32 / cols.max(1);
                    let (scale_mode, block) = scale_layout.mode_block();
                    schedule.push(Step::ScaledDequantizeGeneral {
                        codes_byte_off: arena.offset(codes_id) as u32,
                        scale_byte_off: arena.offset(scale_id) as u32,
                        out_off_f32: (arena.offset(node.id) / 4) as u32,
                        rows,
                        cols,
                        fmt: format.kernel_id(),
                        scale_mode,
                        block,
                    });
                }
                Op::MatMul => {
                    let (m, k, n, batch, a_bs, b_bs, c_bs, a_id, b_id) =
                        matmul_shape(&graph, node, "MatMul");
                    schedule.push(Step::Matmul {
                        m,
                        k,
                        n,
                        batch,
                        a_batch_stride: a_bs,
                        b_batch_stride: b_bs,
                        c_batch_stride: c_bs,
                        a_off_f32: (arena.offset(a_id) / 4) as u32,
                        b_off_f32: (arena.offset(b_id) / 4) as u32,
                        c_off_f32: (arena.offset(node.id) / 4) as u32,
                        has_bias: 0,
                        bias_off_f32: 0,
                        act_id: 0xFFFF,
                    });
                }
                Op::FusedMatMulBiasAct { activation } => {
                    let (m, k, n, batch, a_bs, b_bs, c_bs, a_id, b_id) =
                        matmul_shape(&graph, node, "FusedMatMulBiasAct");
                    let bias_id = node.inputs[2];
                    let act_id = match activation {
                        None => 0xFFFFu32,
                        Some(a) => activation_op_id(*a),
                    };
                    schedule.push(Step::Matmul {
                        m,
                        k,
                        n,
                        batch,
                        a_batch_stride: a_bs,
                        b_batch_stride: b_bs,
                        c_batch_stride: c_bs,
                        a_off_f32: (arena.offset(a_id) / 4) as u32,
                        b_off_f32: (arena.offset(b_id) / 4) as u32,
                        c_off_f32: (arena.offset(node.id) / 4) as u32,
                        has_bias: 1,
                        bias_off_f32: (arena.offset(bias_id) / 4) as u32,
                        act_id,
                    });
                }
                Op::FusedConvBiasAct {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    activation,
                    has_residual,
                } => {
                    // Only the 2-D conv is fused (matches `FuseConvBiasAct`).
                    // Same `Step::Conv2d` as a plain conv, plus bias + act (+
                    // optional residual). MIOpen has no fused path — runtime
                    // applies `conv_bias_act_epilogue` after the forward conv.
                    let in_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    let bias_id = node.inputs[2];
                    let in_dims = graph.node(in_id).shape.dims();
                    let w_dims = graph.node(w_id).shape.dims();
                    let out_dims = node.shape.dims();
                    let act_id = match activation {
                        None => 0xFFFFu32,
                        Some(a) => activation_op_id(*a),
                    };
                    let (has_res, res_off) = if *has_residual {
                        (1u32, (arena.offset(node.inputs[3]) / 4) as u32)
                    } else {
                        (0u32, 0u32)
                    };
                    schedule.push(Step::Conv2d {
                        n: in_dims[0].unwrap_static() as u32,
                        c_in: in_dims[1].unwrap_static() as u32,
                        c_out: w_dims[0].unwrap_static() as u32,
                        h: in_dims[2].unwrap_static() as u32,
                        w: in_dims[3].unwrap_static() as u32,
                        h_out: out_dims[2].unwrap_static() as u32,
                        w_out: out_dims[3].unwrap_static() as u32,
                        kh: kernel_size[0] as u32,
                        kw: kernel_size[1] as u32,
                        sh: stride[0] as u32,
                        sw: stride[1] as u32,
                        ph: padding[0] as u32,
                        pw: padding[1] as u32,
                        dh: dilation[0] as u32,
                        dw: dilation[1] as u32,
                        groups: *groups as u32,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        w_off: (arena.offset(w_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        has_bias: 1,
                        bias_off_f32: (arena.offset(bias_id) / 4) as u32,
                        act_id,
                        has_residual: has_res,
                        residual_off_f32: res_off,
                    });
                }
                Op::Binary(bop) => {
                    // Complex binary: C64 add/sub/mul/div reads BOTH `[re, im]`
                    // lanes per element, so it can't ride the scalar-per-thread
                    // Binary kernel — lower to the shared `binary_c64` dispatch.
                    // C128 arithmetic is out of scope (rlx-cpu has none either) →
                    // reject; broadcast rides the kernel's own `k % n_x` modulo
                    // (matching CPU), so reading `node.inputs` directly gives the
                    // right per-operand element counts (mirrors rlx-cuda).
                    if node.shape.dtype().is_complex() {
                        if node.shape.dtype() == rlx_ir::DType::C128 {
                            panic!(
                                "rlx-rocm Binary on C128: complex-f64 arithmetic is \
                                 unsupported (rlx-cpu has none either) — only C64 \
                                 add/sub/mul/div are wired"
                            );
                        }
                        let op_code = binary_op_id(*bop);
                        if op_code > 3 {
                            panic!(
                                "rlx-rocm C64 Binary: {bop:?} is undefined for complex \
                                 (only Add/Sub/Mul/Div); matches rlx-cpu rejection"
                            );
                        }
                        let a_id = node.inputs[0];
                        let b_id = node.inputs[1];
                        let n_a = graph.node(a_id).shape.num_elements().unwrap_or(0).max(1) as u32;
                        let n_b = graph.node(b_id).shape.num_elements().unwrap_or(0).max(1) as u32;
                        schedule.push(Step::BinaryC64 {
                            n: elems,
                            a_byte_off: arena.offset(a_id) as u32,
                            b_byte_off: arena.offset(b_id) as u32,
                            c_byte_off: arena.offset(node.id) as u32,
                            op: op_code,
                            n_a,
                            n_b,
                        });
                        continue;
                    }
                    schedule.push(Step::Binary {
                        n: elems,
                        a_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        b_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        c_off: (arena.offset(node.id) / 4) as u32,
                        op: binary_op_id(*bop),
                    });
                }
                Op::Activation(act) => {
                    schedule.push(Step::Unary {
                        n: elems,
                        in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        op: activation_op_id(*act),
                    });
                }
                Op::Compare(cop) => {
                    schedule.push(Step::Compare {
                        n: elems,
                        a_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        b_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        c_off: (arena.offset(node.id) / 4) as u32,
                        op: compare_op_id(*cop),
                    });
                }
                Op::Where => {
                    schedule.push(Step::Where {
                        n: elems,
                        cond_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        x_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        y_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::Fma => {
                    schedule.push(Step::Fma {
                        n: elems,
                        a_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        b_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        c_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::BatchElementwiseRegion {
                    chain,
                    num_batch_inputs,
                    scalar_input_mask,
                    input_modulus,
                    prologue,
                    prologue_input,
                } => {
                    let n = *num_batch_inputs as usize;
                    if n == 0 || chain.len() > 32 {
                        panic!(
                            "rlx-rocm BatchElementwiseRegion: num_batch_inputs={n} steps={}",
                            chain.len()
                        );
                    }
                    let slice_shape = rlx_ir::batch_region_slice_shape(&node.shape);
                    let slice_elems = rlx_ir::batch_region_slice_elems(&node.shape, n)
                        .expect("batch region static shape");
                    let base_dst_off = (arena.offset(node.id) / 4) as u32;
                    let use_single = rlx_ir::fk_batch_use_single_launch(n, *prologue);
                    if use_single {
                        let mut batch_input_offs = [0u32; 64];
                        for i in 0..n {
                            batch_input_offs[i] = (arena.offset(node.inputs[i]) / 4) as u32;
                        }
                        let input_offs_meta = [0u32; 16];
                        let meta_arr = rlx_ir::encode_elementwise_region_meta(
                            &input_offs_meta,
                            chain,
                            *prologue,
                            &slice_shape,
                            *prologue_input,
                        );
                        let meta = upload_meta(&ctx, &meta_arr);
                        let meta_idx = meta_buffers.len();
                        meta_buffers.push(meta);
                        let batch_vec: Vec<u32> = batch_input_offs[..n].to_vec();
                        let batch_dev = upload_meta(&ctx, &batch_vec);
                        let batch_offs_idx = meta_buffers.len();
                        meta_buffers.push(batch_dev);
                        schedule.push(Step::BatchElementwiseRegion {
                            slice_len: slice_elems,
                            num_batch: n as u32,
                            num_steps: chain.len() as u32,
                            base_dst_off,
                            slice_elems,
                            batch_input_offs,
                            batch_offs_idx,
                            meta_idx,
                            scalar_input_mask: *scalar_input_mask,
                            input_modulus: *input_modulus,
                        });
                    } else {
                        for i in 0..n {
                            let mut input_offs = [0u32; 16];
                            input_offs[0] = (arena.offset(node.inputs[i]) / 4) as u32;
                            let meta_arr = rlx_ir::encode_elementwise_region_meta(
                                &input_offs,
                                chain,
                                *prologue,
                                &slice_shape,
                                *prologue_input,
                            );
                            let meta = upload_meta(&ctx, &meta_arr);
                            let meta_idx = meta_buffers.len();
                            meta_buffers.push(meta);
                            let spatial =
                                matches!(*prologue, rlx_ir::RegionPrologue::ResizeNearest2x);
                            let grid = rlx_ir::PrologueLaunchGrid::from_output_shape(&slice_shape);
                            schedule.push(Step::ElementwiseRegion {
                                len: slice_elems,
                                num_inputs: 1,
                                num_steps: chain.len() as u32,
                                dst_off: rlx_ir::batch_region_slice_dst_off_f32(
                                    base_dst_off,
                                    slice_elems,
                                    i,
                                ),
                                input_offs,
                                scalar_input_mask: *scalar_input_mask,
                                input_modulus: *input_modulus,
                                meta_idx,
                                spatial_prologue: spatial,
                                prologue_w: grid.map(|g| g.width).unwrap_or(0),
                                prologue_h: grid.map(|g| g.height).unwrap_or(0),
                                prologue_nc: grid.map(|g| g.depth).unwrap_or(0),
                            });
                        }
                    }
                }
                Op::ElementwiseRegion {
                    chain,
                    num_inputs,
                    scalar_input_mask,
                    input_modulus,
                    prologue,
                    prologue_input,
                } => {
                    // PLAN L2 native lowering. Encode the chain into a
                    // 149-u32 metadata buffer (16 input offsets + 32 steps *
                    // 4 u32s + prologue tail) uploaded once at compile time;
                    // the kernel walks the chain interpretively in registers.
                    let n = *num_inputs as usize;
                    if n > 16 || chain.len() > 32 {
                        panic!(
                            "rlx-rocm ElementwiseRegion: chain too large \
                                (inputs={n}, steps={}). Caps: 16 / 32. \
                                Run UnfuseElementwiseRegions to fall back \
                                to atomic ops.",
                            chain.len()
                        );
                    }
                    let mut input_offs = [0u32; 16];
                    for (i, &id) in node.inputs.iter().enumerate() {
                        input_offs[i] = (arena.offset(id) / 4) as u32;
                    }
                    let meta_arr = rlx_ir::encode_elementwise_region_meta(
                        &input_offs,
                        chain,
                        *prologue,
                        &node.shape,
                        *prologue_input,
                    );
                    let meta = upload_meta(&ctx, &meta_arr);
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta);
                    let spatial = matches!(*prologue, rlx_ir::RegionPrologue::ResizeNearest2x);
                    let grid = rlx_ir::PrologueLaunchGrid::from_output_shape(&node.shape);
                    schedule.push(Step::ElementwiseRegion {
                        len: elems,
                        num_inputs: *num_inputs,
                        num_steps: chain.len() as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        input_offs,
                        scalar_input_mask: *scalar_input_mask,
                        input_modulus: *input_modulus,
                        meta_idx,
                        spatial_prologue: spatial,
                        prologue_w: grid.map(|g| g.width).unwrap_or(0),
                        prologue_h: grid.map(|g| g.height).unwrap_or(0),
                        prologue_nc: grid.map(|g| g.depth).unwrap_or(0),
                    });
                }
                Op::Reduce {
                    op,
                    axes,
                    keep_dim: _,
                } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let rank = in_dims.len();
                    // The reduce kernel collapses a CONTIGUOUS TRAILING block of
                    // `inner` elements per `outer` slice. A single last axis is the
                    // common case, but a contiguous *suffix* of axes (e.g. [1,2] on
                    // rank 3 — reduce the last two dims, as in global mean/pool or a
                    // LayerNorm variance) maps to the very same kernel with
                    // inner = ∏(trailing dims), outer = ∏(leading dims). Reduction is
                    // commutative over axes, so sort a copy before checking.
                    let mut sorted: Vec<usize> = axes.to_vec();
                    sorted.sort_unstable();
                    let is_trailing_suffix = !sorted.is_empty()
                        && *sorted.last().unwrap() == rank - 1
                        && sorted
                            .iter()
                            .enumerate()
                            .all(|(i, &a)| a == rank - sorted.len() + i);
                    if !is_trailing_suffix {
                        panic!(
                            "rlx-rocm Reduce: only a contiguous trailing axis block is \
                                supported (got axes={axes:?}, rank={rank})"
                        );
                    }
                    let split = rank - sorted.len();
                    let inner = in_dims[split..]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let outer = in_dims[..split]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    schedule.push(Step::Reduce {
                        outer,
                        inner,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        op: reduce_op_id(*op),
                    });
                }
                Op::Softmax { axis: _ } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let outer = in_dims[..in_dims.len() - 1]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    schedule.push(Step::Softmax {
                        outer,
                        inner,
                        stride: 1,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::ReluBackward => {
                    let x_id = node.inputs[0];
                    let dy_id = node.inputs[1];
                    schedule.push(Step::ReluBackward {
                        n: elems,
                        x_off: (arena.offset(x_id) / 4) as u32,
                        dy_off: (arena.offset(dy_id) / 4) as u32,
                        dx_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::ActivationBackward { kind } => {
                    let x_id = node.inputs[0];
                    let dy_id = node.inputs[1];
                    schedule.push(Step::ActivationBackward {
                        n: elems,
                        x_off: (arena.offset(x_id) / 4) as u32,
                        dy_off: (arena.offset(dy_id) / 4) as u32,
                        dx_off: (arena.offset(node.id) / 4) as u32,
                        op: activation_op_id(*kind),
                    });
                }
                Op::SoftmaxCrossEntropy => {
                    let logits_id = node.inputs[0];
                    let targets_id = node.inputs[1];
                    let logits_shape = graph.node(logits_id).shape.dims();
                    let inner = logits_shape.last().unwrap().unwrap_static() as u32;
                    let total: u32 = logits_shape
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product();
                    let outer = total / inner.max(1);
                    schedule.push(Step::SoftmaxCrossEntropy {
                        outer,
                        inner,
                        logits_off: (arena.offset(logits_id) / 4) as u32,
                        targets_off: (arena.offset(targets_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::SoftmaxCrossEntropyWithLogits => {
                    let logits_id = node.inputs[0];
                    let labels_id = node.inputs[1];
                    let logits_shape = graph.node(logits_id).shape.dims();
                    let inner = logits_shape.last().unwrap().unwrap_static() as u32;
                    let total: u32 = logits_shape
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product();
                    let outer = total / inner.max(1);
                    schedule.push(Step::SoftmaxCrossEntropyWithLogits {
                        outer,
                        inner,
                        logits_off: (arena.offset(logits_id) / 4) as u32,
                        labels_off: (arena.offset(labels_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::SoftmaxCrossEntropyBackward => {
                    let logits_id = node.inputs[0];
                    let labels_id = node.inputs[1];
                    let d_loss_id = node.inputs[2];
                    let logits_shape = graph.node(logits_id).shape.dims();
                    let inner = logits_shape.last().unwrap().unwrap_static() as u32;
                    let total: u32 = logits_shape
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product();
                    let outer = total / inner.max(1);
                    schedule.push(Step::SoftmaxCrossEntropyBackward {
                        outer,
                        inner,
                        logits_off: (arena.offset(logits_id) / 4) as u32,
                        labels_off: (arena.offset(labels_id) / 4) as u32,
                        d_loss_off: (arena.offset(d_loss_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::LayerNorm { axis: _, eps } | Op::RmsNorm { axis: _, eps } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    let is_layer = matches!(&node.op, Op::LayerNorm { .. });
                    let gamma_id = node.inputs[1];
                    // Both LayerNorm and RmsNorm carry beta (inputs[2]); the
                    // RmsNorm kernel branch now adds it (matches the CPU oracle).
                    let beta_id = if node.inputs.len() >= 3 {
                        node.inputs[2]
                    } else {
                        gamma_id
                    };
                    schedule.push(Step::LayerNorm {
                        outer,
                        inner,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        gamma_off: (arena.offset(gamma_id) / 4) as u32,
                        beta_off: (arena.offset(beta_id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        op: if is_layer { 0 } else { 1 },
                    });
                }
                Op::FusedResidualLN { has_bias, eps } => {
                    let x_id = node.inputs[0];
                    let r_id = node.inputs[1];
                    let (bias_id, g_id, b_id) = if *has_bias {
                        (node.inputs[2], node.inputs[3], node.inputs[4])
                    } else {
                        (x_id, node.inputs[2], node.inputs[3])
                    };
                    let in_dims = node.shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    schedule.push(Step::FusedResidualLn {
                        outer,
                        inner,
                        in_off: (arena.offset(x_id) / 4) as u32,
                        residual_off: (arena.offset(r_id) / 4) as u32,
                        bias_off: (arena.offset(bias_id) / 4) as u32,
                        gamma_off: (arena.offset(g_id) / 4) as u32,
                        beta_off: (arena.offset(b_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        has_bias: if *has_bias { 1 } else { 0 },
                    });
                }
                // Residual-add + RMSNorm in one native kernel (shared
                // `fused_residual_rms_norm.cu` via hipRTC). Mirrors the
                // `FusedResidualLN` arm above and rlx-cuda's native path.
                Op::FusedResidualRmsNorm { has_bias, eps } => {
                    let x_id = node.inputs[0];
                    let r_id = node.inputs[1];
                    let (bias_id, g_id, b_id) = if *has_bias {
                        (node.inputs[2], node.inputs[3], node.inputs[4])
                    } else {
                        (x_id, node.inputs[2], node.inputs[3])
                    };
                    let in_dims = node.shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    schedule.push(Step::FusedResidualRmsNorm {
                        outer,
                        inner,
                        in_off: (arena.offset(x_id) / 4) as u32,
                        residual_off: (arena.offset(r_id) / 4) as u32,
                        bias_off: (arena.offset(bias_id) / 4) as u32,
                        gamma_off: (arena.offset(g_id) / 4) as u32,
                        beta_off: (arena.offset(b_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        has_bias: if *has_bias { 1 } else { 0 },
                    });
                }
                Op::AdaLayerNorm { norm, eps } => {
                    let x_id = node.inputs[0];
                    let scale_id = node.inputs[1];
                    let shift_id = node.inputs[2];
                    let in_dims = node.shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    let x_dims: Vec<usize> = graph
                        .node(x_id)
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    let mod_dims: Vec<usize> = graph
                        .node(scale_id)
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    let lead_pack = rlx_ir::ada_modulation_lead_pack(&x_dims, &mod_dims);
                    let meta = upload_meta(&ctx, &lead_pack);
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta);
                    schedule.push(Step::AdaLayerNorm {
                        outer,
                        inner,
                        in_off: (arena.offset(x_id) / 4) as u32,
                        scale_off: (arena.offset(scale_id) / 4) as u32,
                        shift_off: (arena.offset(shift_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        layer_norm: u32::from(matches!(norm, rlx_ir::op::AdaNormKind::LayerNorm)),
                        meta_idx,
                    });
                }
                Op::GatedResidual => {
                    let x_id = node.inputs[0];
                    let y_id = node.inputs[1];
                    let gate_id = node.inputs[2];
                    let in_dims = node.shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                    let x_dims: Vec<usize> = graph
                        .node(x_id)
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    let gate_dims: Vec<usize> = graph
                        .node(gate_id)
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    let lead_pack = rlx_ir::ada_modulation_lead_pack(&x_dims, &gate_dims);
                    let meta = upload_meta(&ctx, &lead_pack);
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta);
                    schedule.push(Step::GatedResidual {
                        total,
                        inner,
                        x_off: (arena.offset(x_id) / 4) as u32,
                        y_off: (arena.offset(y_id) / 4) as u32,
                        gate_off: (arena.offset(gate_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        meta_idx,
                    });
                }
                Op::AdaLayerNormBackward { norm, eps } => {
                    let x_id = node.inputs[0];
                    let scale_id = node.inputs[1];
                    let shift_id = node.inputs[2];
                    let dy_id = node.inputs[3];
                    let _ = shift_id;
                    let in_dims = graph.node(x_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let x_dims: Vec<usize> = in_dims.iter().map(|d| d.unwrap_static()).collect();
                    let mod_dims: Vec<usize> = graph
                        .node(scale_id)
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    let (mod_rows, seq_per_mod) = rlx_ir::ada_modulation_launch(&x_dims, &mod_dims);
                    schedule.push(Step::AdaLayerNormBackward {
                        mod_rows,
                        seq_per_mod,
                        inner,
                        x_off: (arena.offset(x_id) / 4) as u32,
                        scale_off: (arena.offset(scale_id) / 4) as u32,
                        dy_off: (arena.offset(dy_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        layer_norm: u32::from(matches!(norm, rlx_ir::op::AdaNormKind::LayerNorm)),
                    });
                }
                Op::GatedResidualBackward => {
                    let x_id = node.inputs[0];
                    let y_id = node.inputs[1];
                    let gate_id = node.inputs[2];
                    let dy_id = node.inputs[3];
                    let _ = x_id;
                    let in_dims = graph.node(y_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let x_dims: Vec<usize> = graph
                        .node(x_id)
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    let gate_dims: Vec<usize> = graph
                        .node(gate_id)
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    let (mod_rows, seq_per_mod) =
                        rlx_ir::ada_modulation_launch(&x_dims, &gate_dims);
                    schedule.push(Step::GatedResidualBackward {
                        mod_rows,
                        seq_per_mod,
                        inner,
                        y_off: (arena.offset(y_id) / 4) as u32,
                        gate_off: (arena.offset(gate_id) / 4) as u32,
                        dy_off: (arena.offset(dy_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::Gather { axis } => {
                    let table_id = node.inputs[0];
                    let idx_id = node.inputs[1];
                    // Complex packs `lanes` contiguous f32 per element. Index
                    // values select ELEMENTS (indices stay unscaled, real-typed),
                    // but each gathered element is `lanes` contiguous f32 — so the
                    // per-element contiguous span (`dim`/`trailing`) and the total
                    // copy count scale by lanes. lanes=1 ⇒ strict no-op.
                    let lanes: u32 = match node.shape.dtype() {
                        rlx_ir::DType::C64 => 2,
                        rlx_ir::DType::C128 => 4,
                        _ => 1,
                    };
                    if *axis == 0 {
                        let table_shape = graph.node(table_id).shape.dims();
                        let idx_shape = graph.node(idx_id).shape.dims();
                        let vocab = table_shape[0].unwrap_static() as u32;
                        let dim: u32 = table_shape[1..]
                            .iter()
                            .map(|d| d.unwrap_static() as u32)
                            .product::<u32>()
                            .max(1)
                            * lanes;
                        let n_idx: u32 =
                            idx_shape.iter().map(|d| d.unwrap_static() as u32).product();
                        schedule.push(Step::Gather {
                            n_out: elems * lanes,
                            n_idx,
                            dim,
                            vocab,
                            in_off: (arena.offset(table_id) / 4) as u32,
                            idx_off: (arena.offset(idx_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                        });
                    } else {
                        let table_shape = graph.node(table_id).shape.dims();
                        let idx_shape = graph.node(idx_id).shape.dims();
                        let outer: u32 = table_shape[..*axis]
                            .iter()
                            .map(|d| d.unwrap_static() as u32)
                            .product::<u32>()
                            .max(1);
                        let trailing: u32 = table_shape[*axis + 1..]
                            .iter()
                            .map(|d| d.unwrap_static() as u32)
                            .product::<u32>()
                            .max(1)
                            * lanes;
                        let axis_dim = table_shape[*axis].unwrap_static() as u32;
                        let num_idx: u32 =
                            idx_shape.iter().map(|d| d.unwrap_static() as u32).product();
                        let total = outer * num_idx * trailing;
                        schedule.push(Step::GatherAxis {
                            total,
                            outer,
                            axis_dim,
                            num_idx,
                            trailing,
                            table_off: (arena.offset(table_id) / 4) as u32,
                            idx_off: (arena.offset(idx_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                        });
                    }
                }
                Op::Narrow { axis, start, len } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    // Complex packs `lanes` contiguous f32 per element. The lane
                    // axis is innermost, so `axis`/`start`/`len` stay element-
                    // indexed; only the per-copy contiguous `inner` count (dims
                    // after `axis`) scales by lanes so each thread moves a whole
                    // complex element. lanes=1 ⇒ strict no-op.
                    let lanes: u32 = match node.shape.dtype() {
                        rlx_ir::DType::C64 => 2,
                        rlx_ir::DType::C128 => 4,
                        _ => 1,
                    };
                    let outer: u32 = in_dims[..*axis]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let inner: u32 = in_dims[*axis + 1..]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1)
                        * lanes;
                    let axis_in = in_dims[*axis].unwrap_static() as u32;
                    schedule.push(Step::Narrow {
                        total: elems * lanes,
                        outer,
                        inner,
                        axis_in_size: axis_in,
                        axis_out_size: *len as u32,
                        start: *start as u32,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::Transpose { perm } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    // Complex packs `lanes` contiguous f32 per element (C64=2
                    // [re,im], C128=4 df64). The transpose kernel copies one f32
                    // per "element" via reindexed strides, so append an INNERMOST
                    // lane axis that maps to ITSELF (never permuted): each thread
                    // copies a whole complex element's lanes contiguously instead
                    // of shattering [re,im]. lanes=1 for real/int ⇒ strict no-op.
                    let lanes: u32 = match node.shape.dtype() {
                        rlx_ir::DType::C64 => 2,
                        rlx_ir::DType::C128 => 4,
                        _ => 1,
                    };
                    let mut perm: Vec<usize> = perm.to_vec();
                    let mut in_dims_u: Vec<u32> =
                        in_dims.iter().map(|d| d.unwrap_static() as u32).collect();
                    if lanes > 1 {
                        // The lane axis is input axis `in_dims_u.len()` (appended
                        // innermost); it maps to itself as the output's innermost.
                        perm.push(in_dims_u.len());
                        in_dims_u.push(lanes);
                    }
                    let rank = perm.len();
                    // Cumulative input strides (row-major, innermost = 1) over the
                    // extended rank — element strides now count in lane units.
                    let mut in_strides = vec![1u32; rank];
                    for i in (0..rank.saturating_sub(1)).rev() {
                        in_strides[i] = in_strides[i + 1] * in_dims_u[i + 1];
                    }
                    let out_dims_u: Vec<u32> = perm.iter().map(|&i| in_dims_u[i]).collect();
                    let strides_for_out: Vec<u32> = perm.iter().map(|&i| in_strides[i]).collect();
                    let mut meta_data: Vec<u32> = Vec::with_capacity(rank * 2);
                    meta_data.extend_from_slice(&out_dims_u);
                    meta_data.extend_from_slice(&strides_for_out);
                    let meta = upload_meta(&ctx, &meta_data);
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta);
                    schedule.push(Step::Transpose {
                        rank: rank as u32,
                        out_total: elems * lanes,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        meta_idx,
                    });
                }
                Op::Expand { target_shape } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let rank = target_shape.len();
                    if rank != in_shape.len() {
                        panic!(
                            "rlx-rocm Expand: rank mismatch (in={}, target={})",
                            in_shape.len(),
                            rank
                        );
                    }
                    let out_dims: Vec<u32> = target_shape.iter().map(|&d| d as u32).collect();
                    let in_dims: Vec<u32> =
                        in_shape.iter().map(|d| d.unwrap_static() as u32).collect();
                    // Complex tensors pack `lanes` contiguous f32 per element
                    // (C64=2 [re,im], C128=4 df64). The expand kernel copies one
                    // f32 per "element", so append an innermost lane axis (in==out,
                    // never a broadcast) — each thread copies a whole complex
                    // element's lanes contiguously instead of shattering [re,im].
                    // lanes=1 for real/int ⇒ strict no-op.
                    let lanes: u32 = match node.shape.dtype() {
                        rlx_ir::DType::C64 => 2,
                        rlx_ir::DType::C128 => 4,
                        _ => 1,
                    };
                    let (rank, out_dims, in_dims) = if lanes > 1 {
                        let mut od = out_dims;
                        let mut idm = in_dims;
                        od.push(lanes);
                        idm.push(lanes);
                        (rank + 1, od, idm)
                    } else {
                        (rank, out_dims, in_dims)
                    };
                    let mut in_strides_row = vec![1u32; rank];
                    for i in (0..rank.saturating_sub(1)).rev() {
                        in_strides_row[i] = in_strides_row[i + 1] * in_dims[i + 1];
                    }
                    let strides_for_out: Vec<u32> = (0..rank)
                        .map(|i| {
                            if in_dims[i] == 1 && out_dims[i] != 1 {
                                0
                            } else {
                                in_strides_row[i]
                            }
                        })
                        .collect();
                    let mut meta_data: Vec<u32> = Vec::with_capacity(rank * 2);
                    meta_data.extend_from_slice(&out_dims);
                    meta_data.extend_from_slice(&strides_for_out);
                    let meta = upload_meta(&ctx, &meta_data);
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta);
                    schedule.push(Step::Expand {
                        rank: rank as u32,
                        out_total: elems * lanes,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        meta_idx,
                    });
                }
                Op::Concat { axis } => {
                    // Caller convention: one Step::Concat per input, copying
                    // each input's slice into the output at the right axis offset.
                    // Complex packs `lanes` contiguous f32 per element; the lane
                    // axis is innermost, so `axis`/axis offsets stay element-
                    // indexed and only the per-copy contiguous `inner` count (and
                    // each input's total copy length) scale by lanes so whole
                    // complex elements move as a group. lanes=1 ⇒ strict no-op.
                    let lanes: u32 = match node.shape.dtype() {
                        rlx_ir::DType::C64 => 2,
                        rlx_ir::DType::C128 => 4,
                        _ => 1,
                    };
                    let mut start: u32 = 0;
                    let out_dims = node.shape.dims();
                    let outer: u32 = out_dims[..*axis]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let inner: u32 = out_dims[*axis + 1..]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1)
                        * lanes;
                    let axis_out_size = out_dims[*axis].unwrap_static() as u32;
                    for &in_id in &node.inputs {
                        let in_dims = graph.node(in_id).shape.dims();
                        let axis_in = in_dims[*axis].unwrap_static() as u32;
                        let total: u32 = in_dims
                            .iter()
                            .map(|d| d.unwrap_static() as u32)
                            .product::<u32>()
                            * lanes;
                        schedule.push(Step::Concat {
                            total,
                            outer,
                            inner,
                            axis_in_size: axis_in,
                            axis_out_size,
                            start,
                            in_off: (arena.offset(in_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                        });
                        start += axis_in;
                    }
                }
                Op::Attention {
                    num_heads,
                    head_dim,
                    v_head_dim,
                    mask_kind,
                    score_scale: _,
                    attn_logit_softcap,
                } => {
                    assert!(
                        v_head_dim.is_none_or(|v| v == *head_dim),
                        "rlx-rocm: asymmetric v_head_dim (MLA) not yet supported"
                    );
                    let q_id = node.inputs[0];
                    let k_id = node.inputs[1];
                    let v_id = node.inputs[2];
                    let q_shape = graph.node(q_id).shape.dims();
                    let k_shape = graph.node(k_id).shape.dims();
                    if q_shape.len() != 4 {
                        panic!("rlx-rocm Attention: unfuse should have promoted to rank-4");
                    }
                    let q_ir = graph.node(q_id).shape.clone();
                    let k_ir = graph.node(k_id).shape.clone();
                    let geom = rlx_ir::attention_geom(&q_ir, &k_ir, *num_heads, *head_dim);
                    let batch = geom.batch as u32;
                    let heads = geom.heads as u32;
                    let seq_q = geom.seq_q as u32;
                    let seq_k = geom.seq_k as u32;
                    let hd = *head_dim as u32;
                    let scale = 1.0_f32 / (hd as f32).sqrt();
                    // Gemma 2 attention logit soft-cap (0 = disabled). Applied
                    // pre-mask in the kernel; matches rlx-cpu executor.rs and rlx-cuda.
                    let softcap_bits = attn_logit_softcap.unwrap_or(0.0).to_bits();
                    let mask_shape = if matches!(mask_kind, MaskKind::Custom | MaskKind::Bias) {
                        Some(graph.node(node.inputs[3]).shape.dims())
                    } else {
                        None
                    };
                    let packed_parent = packed_bshd_attn.get(&node.id).copied();
                    let st = if let Some((_, head_width)) = packed_parent {
                        let (qb, qh, qs) =
                            rlx_ir::packed_bshd_qkv_strides(head_width as usize, hd, seq_q);
                        let (ob, oh, os) =
                            rlx_ir::strides_for_shape(node.shape.dims(), heads, hd, seq_q, false);
                        let (mb, mh, mq, mk) = mask_shape
                            .map(|m| rlx_ir::mask_strides_for_shape(m, heads, seq_q, seq_k))
                            .unwrap_or_else(|| rlx_ir::mask_strides_bhsd(heads, seq_q, seq_k));
                        rlx_ir::AttentionLaunchStrides {
                            q_batch: qb,
                            q_head: qh,
                            q_seq: qs,
                            k_batch: qb,
                            k_head: qh,
                            k_seq: qs,
                            v_batch: qb,
                            v_head: qh,
                            v_seq: qs,
                            o_batch: ob,
                            o_head: oh,
                            o_seq: os,
                            mask_batch: mb,
                            mask_head: mh,
                            mask_q: mq,
                            mask_k: mk,
                        }
                    } else {
                        rlx_ir::attention_launch_strides(
                            geom,
                            q_shape,
                            k_shape,
                            graph.node(v_id).shape.dims(),
                            node.shape.dims(),
                            mask_shape,
                        )
                    };
                    let (q_off, k_off, v_off) = if let Some((parent, head_width)) = packed_parent {
                        let p = (arena.offset(parent) / 4) as u32;
                        (
                            p,
                            p.saturating_add(head_width),
                            p.saturating_add(head_width * 2),
                        )
                    } else {
                        (
                            (arena.offset(q_id) / 4) as u32,
                            (arena.offset(k_id) / 4) as u32,
                            (arena.offset(v_id) / 4) as u32,
                        )
                    };
                    let (mask_kind_id, mask_off, window) = match mask_kind {
                        MaskKind::None => (0u32, 0u32, 0u32),
                        MaskKind::Causal => (1u32, 0u32, 0u32),
                        MaskKind::Custom => (2u32, (arena.offset(node.inputs[3]) / 4) as u32, 0u32),
                        MaskKind::SlidingWindow(w) => (3u32, 0u32, *w as u32),
                        MaskKind::Bias => (4u32, (arena.offset(node.inputs[3]) / 4) as u32, 0u32),
                    };
                    schedule.push(Step::Attention {
                        batch,
                        heads,
                        seq_q,
                        seq_k,
                        head_dim: hd,
                        q_off,
                        k_off,
                        v_off,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        mask_off,
                        mask_kind: mask_kind_id,
                        scale_bits: scale.to_bits(),
                        softcap_bits,
                        window,
                        seq_q_stride: st.mask_q,
                        seq_k_stride: st.mask_k,
                        mask_batch_stride: st.mask_batch,
                        mask_head_stride: st.mask_head,
                        q_batch_stride: st.q_batch,
                        q_head_stride: st.q_head,
                        q_seq_stride: st.q_seq,
                        k_batch_stride: st.k_batch,
                        k_head_stride: st.k_head,
                        k_seq_stride: st.k_seq,
                        v_batch_stride: st.v_batch,
                        v_head_stride: st.v_head,
                        v_seq_stride: st.v_seq,
                        o_batch_stride: st.o_batch,
                        o_head_stride: st.o_head,
                        o_seq_stride: st.o_seq,
                    });
                }
                Op::AttentionBackward {
                    num_heads: _,
                    head_dim,
                    mask_kind,
                    wrt,
                } => {
                    use rlx_ir::op::AttentionBwdWrt;
                    let q_id = node.inputs[0];
                    let k_id = node.inputs[1];
                    let v_id = node.inputs[2];
                    let dy_id = node.inputs[3];
                    let q_shape = graph.node(q_id).shape.dims();
                    let k_shape = graph.node(k_id).shape.dims();
                    if q_shape.len() != 4 {
                        panic!("rlx-rocm AttentionBackward: unfuse should have promoted to rank-4");
                    }
                    let batch = q_shape[0].unwrap_static() as u32;
                    let heads = q_shape[1].unwrap_static() as u32;
                    let seq_q = q_shape[2].unwrap_static() as u32;
                    let seq_k = k_shape[2].unwrap_static() as u32;
                    let hd = *head_dim as u32;
                    let scale = 1.0_f32 / (hd as f32).sqrt();
                    let (mask_kind_id, mask_off, window) = match mask_kind {
                        MaskKind::None => (0u32, 0u32, 0u32),
                        MaskKind::Causal => (1u32, 0u32, 0u32),
                        MaskKind::Custom => (2u32, (arena.offset(node.inputs[4]) / 4) as u32, 0u32),
                        MaskKind::SlidingWindow(w) => (3u32, 0u32, *w as u32),
                        MaskKind::Bias => (4u32, (arena.offset(node.inputs[4]) / 4) as u32, 0u32),
                    };
                    let wrt_id = match wrt {
                        AttentionBwdWrt::Query => 0u32,
                        AttentionBwdWrt::Key => 1u32,
                        AttentionBwdWrt::Value => 2u32,
                    };
                    schedule.push(Step::AttentionBackward {
                        batch,
                        heads,
                        seq_q,
                        seq_k,
                        head_dim: hd,
                        q_off: (arena.offset(q_id) / 4) as u32,
                        k_off: (arena.offset(k_id) / 4) as u32,
                        v_off: (arena.offset(v_id) / 4) as u32,
                        dy_off: (arena.offset(dy_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        mask_off,
                        mask_kind: mask_kind_id,
                        scale_bits: scale.to_bits(),
                        window,
                        wrt: wrt_id,
                    });
                }
                Op::Rope {
                    head_dim,
                    n_rot,
                    style,
                } => {
                    let x_id = node.inputs[0];
                    let cos_id = node.inputs[1];
                    let sin_id = node.inputs[2];
                    let x_shape = graph.node(x_id).shape.dims();
                    let last = x_shape.last().map(|d| d.unwrap_static()).unwrap_or(0);
                    if !last.is_multiple_of(*head_dim) {
                        panic!(
                            "rlx-rocm Rope: last_dim {} not multiple of head_dim {}",
                            last, head_dim
                        );
                    }
                    if head_dim % 2 != 0 {
                        panic!("rlx-rocm Rope: head_dim must be even");
                    }
                    let total: u32 = x_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let seq = x_shape[x_shape.len() - 2].unwrap_static() as u32;
                    let interleaved = match style {
                        rlx_ir::op::RopeStyle::NeoX => 0u32,
                        rlx_ir::op::RopeStyle::GptJ => 1u32,
                    };
                    schedule.push(Step::Rope {
                        n_total: total,
                        seq,
                        head_dim: *head_dim as u32,
                        half: (*head_dim / 2) as u32,
                        // Partial rotary: rotate only n_rot dims (Gemma); == half for
                        // full rope (qwen/llama). Mirrors rlx-cuda's Step::Rope.
                        rot_half: (*n_rot / 2) as u32,
                        in_off: (arena.offset(x_id) / 4) as u32,
                        cos_off: (arena.offset(cos_id) / 4) as u32,
                        sin_off: (arena.offset(sin_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        last_dim: last as u32,
                        interleaved,
                    });
                }
                Op::Cumsum { axis: _, exclusive } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let outer = in_dims[..in_dims.len() - 1]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    schedule.push(Step::Cumsum {
                        outer,
                        inner,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        exclusive: if *exclusive { 1 } else { 0 },
                    });
                }
                Op::CumProd { axis: _, exclusive } | Op::CumMax { axis: _, exclusive } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let outer = in_dims[..in_dims.len() - 1]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    schedule.push(Step::CumScan {
                        outer,
                        inner,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        exclusive: if *exclusive { 1 } else { 0 },
                        is_max: matches!(node.op, Op::CumMax { .. }) as u32,
                    });
                }
                Op::TopK { k } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let outer = in_dims[..in_dims.len() - 1]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    schedule.push(Step::TopK {
                        outer,
                        inner,
                        k: *k as u32,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::GroupedMatMul => {
                    let in_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    let idx_id = node.inputs[2];
                    let in_dims = graph.node(in_id).shape.dims();
                    let w_dims = graph.node(w_id).shape.dims();
                    let m = in_dims[0].unwrap_static() as u32;
                    let k = in_dims[1].unwrap_static() as u32;
                    let n = w_dims[2].unwrap_static() as u32;
                    let ne = w_dims[0].unwrap_static() as u32;
                    schedule.push(Step::GroupedMatmul {
                        m,
                        k,
                        n,
                        num_experts: ne,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        w_off: (arena.offset(w_id) / 4) as u32,
                        idx_off: (arena.offset(idx_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::ScatterAdd => {
                    let upd_id = node.inputs[0];
                    let idx_id = node.inputs[1];
                    let upd_dims = graph.node(upd_id).shape.dims();
                    let out_dims = node.shape.dims();
                    let num_updates = upd_dims[0].unwrap_static() as u32;
                    let trailing: u32 = upd_dims
                        .iter()
                        .skip(1)
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let out_dim = out_dims[0].unwrap_static() as u32;
                    let out_total = out_dim * trailing;
                    let out_off = (arena.offset(node.id) / 4) as u32;
                    schedule.push(Step::ScatterAddZero { out_off, out_total });
                    schedule.push(Step::ScatterAddAcc {
                        out_off,
                        upd_off: (arena.offset(upd_id) / 4) as u32,
                        idx_off: (arena.offset(idx_id) / 4) as u32,
                        num_updates,
                        trailing,
                        out_dim,
                    });
                }
                Op::DequantMatMul { scheme } => {
                    use rlx_ir::quant::QuantScheme;
                    let x_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    let out_total = node.shape.num_elements().unwrap_or(0) as u32;
                    let n = node.shape.dim(node.shape.rank() - 1).unwrap_static() as u32;
                    let m = out_total / n.max(1);
                    let x_total = graph.node(x_id).shape.num_elements().unwrap_or(0) as u32;
                    let k = x_total / m.max(1);
                    if scheme.is_gguf() {
                        schedule.push(Step::DequantMatmulGguf {
                            m,
                            k,
                            n,
                            scheme_id: crate::gguf_host::gguf_scheme_id(*scheme),
                            x_byte_off: arena.offset(x_id) as u32,
                            w_byte_off: arena.offset(w_id) as u32,
                            out_byte_off: arena.offset(node.id) as u32,
                        });
                    } else {
                        let (block_size, scheme_id) = match scheme {
                            QuantScheme::Int8Block { block_size } => (*block_size, 0u32),
                            QuantScheme::Int8BlockAsym { block_size } => (*block_size, 1u32),
                            QuantScheme::Int4Block { block_size } => (*block_size, 2u32),
                            QuantScheme::Fp8E4m3 => (1, 3u32),
                            QuantScheme::Fp8E5m2 => (1, 4u32),
                            QuantScheme::Nvfp4Block => (rlx_ir::NVFP4_GROUP_SIZE as u32, 5u32),
                            QuantScheme::MlxAffine { .. }
                            | QuantScheme::MlxMxfp4 { .. }
                            | QuantScheme::MlxMxfp8 { .. } => {
                                let scale_id = node.inputs[2];
                                let zp_id = node.inputs[3];
                                schedule.push(Step::DequantMatmulMlx {
                                    m,
                                    k,
                                    n,
                                    scheme: *scheme,
                                    x_byte_off: arena.offset(x_id) as u32,
                                    w_byte_off: arena.offset(w_id) as u32,
                                    scale_byte_off: arena.offset(scale_id) as u32,
                                    zp_byte_off: arena.offset(zp_id) as u32,
                                    out_byte_off: arena.offset(node.id) as u32,
                                });
                                continue;
                            }
                            QuantScheme::MxFp4x2Block { group_size } => {
                                // 3 inputs (x, w_q=[plane0|plane1], scale=[s0|s1]);
                                // decode to f32 scratch then hipBLAS sgemm.
                                let scale_id = node.inputs[2];
                                schedule.push(Step::DequantMatmulMxFp4x2 {
                                    m,
                                    k,
                                    n,
                                    group: *group_size,
                                    x_byte_off: arena.offset(x_id) as u32,
                                    w_byte_off: arena.offset(w_id) as u32,
                                    scale_byte_off: arena.offset(scale_id) as u32,
                                    out_byte_off: arena.offset(node.id) as u32,
                                });
                                continue;
                            }
                            other => panic!("rlx-rocm DequantMatMul: unsupported scheme {other:?}"),
                        };
                        let scale_id = node.inputs[2];
                        let zp_id = node.inputs[3];
                        schedule.push(Step::DequantMatmul {
                            m,
                            k,
                            n,
                            block_size,
                            scheme_id,
                            x_off: (arena.offset(x_id) / 4) as u32,
                            w_off: (arena.offset(w_id) / 4) as u32,
                            scale_off: (arena.offset(scale_id) / 4) as u32,
                            zp_off: (arena.offset(zp_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                        });
                    }
                }
                Op::DequantGroupedMatMul { scheme } => {
                    let in_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    let idx_id = node.inputs[2];
                    let in_dims = graph.node(in_id).shape.dims();
                    let out_dims = node.shape.dims();
                    let m = in_dims[0].unwrap_static() as u32;
                    let k = in_dims[1].unwrap_static() as u32;
                    let n = out_dims[out_dims.len() - 1].unwrap_static() as u32;
                    let block_elems = scheme.gguf_block_size() as usize;
                    let block_bytes = scheme.gguf_block_bytes() as usize;
                    let slab_bytes = (k as usize * n as usize) / block_elems * block_bytes;
                    let total_bytes = graph.node(w_id).shape.num_elements().unwrap();
                    let ne = (total_bytes / slab_bytes.max(1)) as u32;
                    schedule.push(Step::DequantGroupedMatmulGguf {
                        m,
                        k,
                        n,
                        num_experts: ne,
                        scheme_id: crate::gguf_host::gguf_scheme_id(*scheme),
                        x_byte_off: arena.offset(in_id) as u32,
                        w_byte_off: arena.offset(w_id) as u32,
                        idx_byte_off: arena.offset(idx_id) as u32,
                        out_byte_off: arena.offset(node.id) as u32,
                    });
                }
                Op::DequantGroupedMatMulMlx { scheme } => {
                    // 5 inputs: input, w_q, scales, biases/zp, expert_idx. Host-delegated
                    // (mirrors CUDA — no native grouped-MLX ROCm kernel yet).
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let out_dims = node.shape.dims();
                    let m = in_dims[in_dims.len() - 2].unwrap_static() as u32;
                    let k = in_dims[in_dims.len() - 1].unwrap_static() as u32;
                    let n = out_dims[out_dims.len() - 1].unwrap_static() as u32;
                    let scale_id = node.inputs[2];
                    let ne = graph.node(scale_id).shape.dims()[0].unwrap_static() as u32;
                    // MXFP4 experts → native on-device decode-GEMM (register nibble-decode,
                    // no host round-trip); affine → host-delegate. Scales are f32 in the arena.
                    if let rlx_ir::quant::QuantScheme::MlxMxfp4 { group_size } = scheme {
                        schedule.push(Step::DequantGroupedMatmulMlxNative {
                            m,
                            k,
                            n,
                            num_experts: ne,
                            group_size: *group_size,
                            x_byte_off: arena.offset(in_id) as u32,
                            w_byte_off: arena.offset(node.inputs[1]) as u32,
                            scale_byte_off: arena.offset(scale_id) as u32,
                            idx_byte_off: arena.offset(node.inputs[4]) as u32,
                            out_byte_off: arena.offset(node.id) as u32,
                        });
                    } else {
                        schedule.push(Step::DequantGroupedMatmulMlxHost {
                            m,
                            k,
                            n,
                            num_experts: ne,
                            scheme: *scheme,
                            x_byte_off: arena.offset(in_id) as u32,
                            w_byte_off: arena.offset(node.inputs[1]) as u32,
                            scale_byte_off: arena.offset(scale_id) as u32,
                            zp_byte_off: arena.offset(node.inputs[3]) as u32,
                            idx_byte_off: arena.offset(node.inputs[4]) as u32,
                            out_byte_off: arena.offset(node.id) as u32,
                            // BF16 scale params are WIDENED to f32 in this backend's main arena
                            // (mirrors rlx-cuda) — the host-delegate reads the f32 main buffer, so
                            // it must read f32, never bf16 (graph dtype BF16 → every-other value=0).
                            scale_bf16: false,
                        });
                    }
                }
                Op::SelectiveScan { state_size } => {
                    if *state_size > 256 {
                        panic!("rlx-rocm SelectiveScan: state_size {state_size} > 256 cap");
                    }
                    let x_id = node.inputs[0];
                    let dt_id = node.inputs[1];
                    let a_id = node.inputs[2];
                    let b_id = node.inputs[3];
                    let c_id = node.inputs[4];
                    let in_dims = graph.node(x_id).shape.dims();
                    schedule.push(Step::SelectiveScan {
                        batch: in_dims[0].unwrap_static() as u32,
                        seq: in_dims[1].unwrap_static() as u32,
                        hidden: in_dims[2].unwrap_static() as u32,
                        state_size: *state_size as u32,
                        x_off: (arena.offset(x_id) / 4) as u32,
                        delta_off: (arena.offset(dt_id) / 4) as u32,
                        a_off: (arena.offset(a_id) / 4) as u32,
                        b_off: (arena.offset(b_id) / 4) as u32,
                        c_off: (arena.offset(c_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::Fft { inverse, norm } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.clone();
                    let meta = rlx_ir::fft::fft_meta(&in_shape);
                    let dtype = in_shape.dtype();
                    let use_gpu = matches!(dtype, rlx_ir::DType::F32)
                        && meta.n_complex.is_power_of_two()
                        && meta.n_complex >= 2;
                    schedule.push(Step::Fft {
                        src_byte_off: arena.offset(in_id) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        outer: meta.outer as u32,
                        n_complex: meta.n_complex as u32,
                        inverse: *inverse,
                        norm_tag: norm.tag(),
                        dtype_tag: rocm_fft_dtype_tag(dtype),
                        use_gpu,
                    });
                }
                Op::LogMel => {
                    let spec_shape = graph.node(node.inputs[0]).shape.clone();
                    let filt_shape = graph.node(node.inputs[1]).shape.clone();
                    let meta = rlx_ir::audio::log_mel_meta(&spec_shape, &filt_shape)
                        .unwrap_or_else(|e| panic!("Op::LogMel: {e}"));
                    schedule.push(Step::LogMelHost {
                        spec_byte_off: arena.offset(node.inputs[0]) as u32,
                        filt_byte_off: arena.offset(node.inputs[1]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        outer: meta.outer as u32,
                        n_fft: meta.n_fft as u32,
                        n_bins: meta.n_bins as u32,
                        n_mels: meta.n_mels as u32,
                    });
                }
                Op::LogMelBackward => {
                    let spec_shape = graph.node(node.inputs[0]).shape.clone();
                    let filt_shape = graph.node(node.inputs[1]).shape.clone();
                    let meta = rlx_ir::audio::log_mel_meta(&spec_shape, &filt_shape)
                        .unwrap_or_else(|e| panic!("Op::LogMelBackward: {e}"));
                    schedule.push(Step::LogMelBackwardHost {
                        spec_byte_off: arena.offset(node.inputs[0]) as u32,
                        filt_byte_off: arena.offset(node.inputs[1]) as u32,
                        dy_byte_off: arena.offset(node.inputs[2]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        outer: meta.outer as u32,
                        n_fft: meta.n_fft as u32,
                        n_bins: meta.n_bins as u32,
                        n_mels: meta.n_mels as u32,
                    });
                }
                Op::WelchPeaks { k, n_segments } => {
                    let spec_shape = graph.node(node.inputs[0]).shape.clone();
                    let meta = rlx_ir::audio::welch_peaks_meta(&spec_shape, *k, *n_segments)
                        .unwrap_or_else(|e| panic!("Op::WelchPeaks: {e}"));
                    let use_gpu = rlx_ir::audio::welch_peaks_gpu_native_eligible(
                        &spec_shape,
                        *k,
                        *n_segments,
                    )
                    .unwrap_or(false);
                    if use_gpu {
                        schedule.push(Step::WelchPeaksGpu {
                            spec_off: (arena.offset(node.inputs[0]) / 4) as u32,
                            dst_off: (arena.offset(node.id) / 4) as u32,
                            welch_batch: meta.welch_batch as u32,
                            n_fft: meta.n_fft as u32,
                            n_segments: meta.n_segments as u32,
                            k: meta.k as u32,
                            n_bins: meta.n_bins as u32,
                        });
                    } else {
                        schedule.push(Step::WelchPeaksHost {
                            spec_byte_off: arena.offset(node.inputs[0]) as u32,
                            dst_byte_off: arena.offset(node.id) as u32,
                            welch_batch: meta.welch_batch as u32,
                            n_fft: meta.n_fft as u32,
                            n_segments: meta.n_segments as u32,
                            k: meta.k as u32,
                        });
                    }
                }
                Op::FftButterflyStage { stage, n_fft } => {
                    let state_shape = &graph.node(node.inputs[0]).shape;
                    assert_eq!(
                        state_shape.dtype(),
                        rlx_ir::DType::F32,
                        "rlx-rocm Op::FftButterflyStage requires F32 state"
                    );
                    schedule.push(Step::FftButterflyStage {
                        state_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        gate_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        rev_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        tw_re_off: (arena.offset(node.inputs[3]) / 4) as u32,
                        tw_im_off: (arena.offset(node.inputs[4]) / 4) as u32,
                        batch: state_shape.dim(0).unwrap_static() as u32,
                        n_fft: *n_fft,
                        stage: *stage,
                    });
                }
                Op::Im2Col {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    if kernel_size.len() != 2 || x_shape.rank() != 4 {
                        panic!("rlx-rocm Im2Col: 2D NCHW only");
                    }
                    let n = match x_shape.dim(0) {
                        rlx_ir::shape::Dim::Static(v) => v as u32,
                        _ => 0,
                    };
                    let c_in = x_shape.dim(1).unwrap_static() as u32;
                    let h = x_shape.dim(2).unwrap_static() as u32;
                    let w = x_shape.dim(3).unwrap_static() as u32;
                    let kh = kernel_size[0] as u32;
                    let kw = kernel_size[1] as u32;
                    let sh = stride.first().copied().unwrap_or(1) as u32;
                    let sw = stride.get(1).copied().unwrap_or(1) as u32;
                    let ph = padding.first().copied().unwrap_or(0) as u32;
                    let pw = padding.get(1).copied().unwrap_or(0) as u32;
                    let dh = dilation.first().copied().unwrap_or(1) as u32;
                    let dw_dil = dilation.get(1).copied().unwrap_or(1) as u32;
                    let h_out = rlx_ir::shape::conv2d_spatial_output(
                        h as usize,
                        kh as usize,
                        sh as usize,
                        ph as usize,
                        dh as usize,
                    ) as u32;
                    let w_out = rlx_ir::shape::conv2d_spatial_output(
                        w as usize,
                        kw as usize,
                        sw as usize,
                        pw as usize,
                        dw_dil as usize,
                    ) as u32;
                    schedule.push(Step::Im2ColHost {
                        x_byte_off: arena.offset(node.inputs[0]) as u32,
                        col_byte_off: arena.offset(node.id) as u32,
                        n,
                        c_in,
                        h,
                        w,
                        h_out,
                        w_out,
                        kh,
                        kw,
                        sh,
                        sw,
                        ph,
                        pw,
                        dh,
                        dw_dil,
                        use_gpu: im2col_use_gpu(n, exec_mode),
                    });
                }
                Op::Reverse { axes } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let rank = in_shape.rank();
                    let dims: Vec<u32> = (0..rank)
                        .map(|i| in_shape.dim(i).unwrap_static() as u32)
                        .collect();
                    let mut rev_mask = vec![false; rank];
                    for &a in axes {
                        if a < rank {
                            rev_mask[a] = true;
                        }
                    }
                    schedule.push(Step::ReverseHost {
                        src_byte_off: arena.offset(node.inputs[0]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        dims,
                        rev_mask,
                        elem_bytes: in_shape.dtype().size_bytes() as u32,
                    });
                }
                Op::ArgMax { axis, keep_dim: _ } | Op::ArgMin { axis, keep_dim: _ } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let rank = in_shape.rank();
                    let outer: usize = (0..*axis)
                        .map(|i| in_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let reduced = in_shape.dim(*axis).unwrap_static();
                    let inner: usize = (*axis + 1..rank)
                        .map(|i| in_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    schedule.push(Step::ArgReduceHost {
                        src_byte_off: arena.offset(node.inputs[0]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        outer: outer as u32,
                        reduced: reduced as u32,
                        inner: inner as u32,
                        is_max: matches!(node.op, Op::ArgMax { .. }),
                    });
                }
                Op::AxialRope2d {
                    end_x,
                    end_y,
                    head_dim,
                    num_heads,
                    theta,
                    repeat_factor,
                } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    schedule.push(Step::AxialRope2d {
                        in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        batch: in_shape.dim(0).unwrap_static() as u32,
                        seq: in_shape.dim(1).unwrap_static() as u32,
                        hidden: in_shape.dim(2).unwrap_static() as u32,
                        end_x: *end_x as u32,
                        end_y: *end_y as u32,
                        head_dim: *head_dim as u32,
                        num_heads: *num_heads as u32,
                        theta: *theta,
                        repeat_factor: *repeat_factor as u32,
                    });
                }
                Op::GatedDeltaNet {
                    state_size,
                    carry_state,
                    gate_per_channel,
                } => {
                    if *state_size > rlx_cpu::gdn::GDN_MAX_STATE {
                        panic!(
                            "rlx-rocm GatedDeltaNet: state_size {state_size} > {}",
                            rlx_cpu::gdn::GDN_MAX_STATE
                        );
                    }
                    let q_id = node.inputs[0];
                    let q_shape = &graph.node(q_id).shape;
                    let state_off = if *carry_state {
                        arena.offset(node.inputs[5])
                    } else {
                        0
                    };
                    schedule.push(Step::GatedDeltaNet {
                        q_byte_off: arena.offset(q_id) as u32,
                        k_byte_off: arena.offset(node.inputs[1]) as u32,
                        v_byte_off: arena.offset(node.inputs[2]) as u32,
                        g_byte_off: arena.offset(node.inputs[3]) as u32,
                        beta_byte_off: arena.offset(node.inputs[4]) as u32,
                        state_byte_off: state_off as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        batch: q_shape.dim(0).unwrap_static() as u32,
                        seq: q_shape.dim(1).unwrap_static() as u32,
                        heads: q_shape.dim(2).unwrap_static() as u32,
                        state_size: *state_size as u32,
                        use_carry: *carry_state,
                        gate_per_channel: *gate_per_channel,
                    });
                }
                Op::Lstm {
                    hidden_size,
                    num_layers,
                    bidirectional,
                    carry,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let (h0, c0) = if *carry {
                        (
                            arena.offset(node.inputs[4]) as u32,
                            arena.offset(node.inputs[5]) as u32,
                        )
                    } else {
                        (0u32, 0u32)
                    };
                    schedule.push(Step::Lstm {
                        x_byte_off: arena.offset(node.inputs[0]) as u32,
                        w_ih_byte_off: arena.offset(node.inputs[1]) as u32,
                        w_hh_byte_off: arena.offset(node.inputs[2]) as u32,
                        bias_byte_off: arena.offset(node.inputs[3]) as u32,
                        h0_byte_off: h0,
                        c0_byte_off: c0,
                        dst_byte_off: arena.offset(node.id) as u32,
                        batch: x_shape.dim(0).unwrap_static() as u32,
                        seq: x_shape.dim(1).unwrap_static() as u32,
                        input_size: x_shape.dim(2).unwrap_static() as u32,
                        hidden: *hidden_size as u32,
                        num_layers: *num_layers as u32,
                        bidirectional: *bidirectional,
                        carry: *carry,
                    });
                }
                Op::Gru {
                    hidden_size,
                    num_layers,
                    bidirectional,
                    carry,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let batch = x_shape.dim(0).unwrap_static() as u32;
                    let seq = x_shape.dim(1).unwrap_static() as u32;
                    let input_size = x_shape.dim(2).unwrap_static() as u32;
                    let hidden = *hidden_size as u32;
                    let force_host = rlx_ir::env::flag("RLX_ROCM_RNN_HOST_FALLBACK");
                    let native = !force_host
                        && crate::gru_gpu::native_gru_ok(
                            *num_layers,
                            *bidirectional,
                            *carry,
                            *hidden_size,
                        );
                    if native {
                        // Carry (h0) is native now: seed from inputs[5] when set.
                        let h0_byte_off = if *carry {
                            arena.offset(node.inputs[5]) as u32
                        } else {
                            0
                        };
                        schedule.push(Step::Gru {
                            x_byte_off: arena.offset(node.inputs[0]) as u32,
                            w_ih_byte_off: arena.offset(node.inputs[1]) as u32,
                            w_hh_byte_off: arena.offset(node.inputs[2]) as u32,
                            b_ih_byte_off: arena.offset(node.inputs[3]) as u32,
                            b_hh_byte_off: arena.offset(node.inputs[4]) as u32,
                            dst_byte_off: arena.offset(node.id) as u32,
                            batch,
                            seq,
                            input_size,
                            hidden,
                            num_layers: *num_layers as u32,
                            bidirectional: *bidirectional,
                            h0_byte_off,
                        });
                    } else {
                        let h0 = if *carry {
                            arena.offset(node.inputs[5]) as u32
                        } else {
                            0u32
                        };
                        schedule.push(Step::GruHost {
                            x_byte_off: arena.offset(node.inputs[0]) as u32,
                            w_ih_byte_off: arena.offset(node.inputs[1]) as u32,
                            w_hh_byte_off: arena.offset(node.inputs[2]) as u32,
                            b_ih_byte_off: arena.offset(node.inputs[3]) as u32,
                            b_hh_byte_off: arena.offset(node.inputs[4]) as u32,
                            h0_byte_off: h0,
                            dst_byte_off: arena.offset(node.id) as u32,
                            batch,
                            seq,
                            input_size,
                            hidden,
                            num_layers: *num_layers as u32,
                            bidirectional: *bidirectional,
                            carry: *carry,
                        });
                    }
                }
                Op::Rnn {
                    hidden_size,
                    num_layers,
                    bidirectional,
                    carry,
                    relu,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let batch = x_shape.dim(0).unwrap_static() as u32;
                    let seq = x_shape.dim(1).unwrap_static() as u32;
                    let input_size = x_shape.dim(2).unwrap_static() as u32;
                    let hidden = *hidden_size as u32;
                    let force_host = rlx_ir::env::flag("RLX_ROCM_RNN_HOST_FALLBACK");
                    let native = !force_host
                        && crate::rnn_gpu::native_rnn_ok(
                            *num_layers,
                            *bidirectional,
                            *carry,
                            *hidden_size,
                        );
                    if native {
                        let h0_byte_off = if *carry {
                            arena.offset(node.inputs[4]) as u32
                        } else {
                            0u32
                        };
                        schedule.push(Step::Rnn {
                            x_byte_off: arena.offset(node.inputs[0]) as u32,
                            w_ih_byte_off: arena.offset(node.inputs[1]) as u32,
                            w_hh_byte_off: arena.offset(node.inputs[2]) as u32,
                            bias_byte_off: arena.offset(node.inputs[3]) as u32,
                            dst_byte_off: arena.offset(node.id) as u32,
                            batch,
                            seq,
                            input_size,
                            hidden,
                            num_layers: *num_layers as u32,
                            bidirectional: *bidirectional,
                            h0_byte_off,
                            relu: *relu,
                        });
                    } else {
                        let h0 = if *carry {
                            arena.offset(node.inputs[4]) as u32
                        } else {
                            0u32
                        };
                        schedule.push(Step::RnnHost {
                            x_byte_off: arena.offset(node.inputs[0]) as u32,
                            w_ih_byte_off: arena.offset(node.inputs[1]) as u32,
                            w_hh_byte_off: arena.offset(node.inputs[2]) as u32,
                            bias_byte_off: arena.offset(node.inputs[3]) as u32,
                            h0_byte_off: h0,
                            dst_byte_off: arena.offset(node.id) as u32,
                            batch,
                            seq,
                            input_size,
                            hidden,
                            num_layers: *num_layers as u32,
                            bidirectional: *bidirectional,
                            carry: *carry,
                            relu: *relu,
                        });
                    }
                }
                Op::Mamba2 {
                    head_dim,
                    state_size,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let batch = x_shape.dim(0).unwrap_static() as u32;
                    let seq = x_shape.dim(1).unwrap_static() as u32;
                    let heads = x_shape.dim(2).unwrap_static() as u32;
                    let force_host = rlx_ir::env::flag("RLX_ROCM_SSM_HOST_FALLBACK");
                    let native = !force_host && crate::mamba2_gpu::native_mamba2_ok(*state_size);
                    let x_byte_off = arena.offset(node.inputs[0]) as u32;
                    let dt_byte_off = arena.offset(node.inputs[1]) as u32;
                    let a_byte_off = arena.offset(node.inputs[2]) as u32;
                    let b_byte_off = arena.offset(node.inputs[3]) as u32;
                    let c_byte_off = arena.offset(node.inputs[4]) as u32;
                    let dst_byte_off = arena.offset(node.id) as u32;
                    if native {
                        schedule.push(Step::Mamba2 {
                            x_byte_off,
                            dt_byte_off,
                            a_byte_off,
                            b_byte_off,
                            c_byte_off,
                            dst_byte_off,
                            batch,
                            seq,
                            heads,
                            head_dim: *head_dim as u32,
                            state_size: *state_size as u32,
                        });
                    } else {
                        schedule.push(Step::Mamba2Host {
                            x_byte_off,
                            dt_byte_off,
                            a_byte_off,
                            b_byte_off,
                            c_byte_off,
                            dst_byte_off,
                            batch,
                            seq,
                            heads,
                            head_dim: *head_dim as u32,
                            state_size: *state_size as u32,
                        });
                    }
                }
                Op::Scan { .. } => {
                    schedule.push(Step::ScanHost {
                        desc: rlx_cpu::rlx_scan_host_desc!(graph, node, |id| arena.offset(id)),
                    });
                }
                Op::ScanBackward { .. } | Op::ScanBackwardXs { .. } => {
                    schedule.push(Step::HostOp {
                        desc: rlx_cpu::rlx_host_op_desc!(graph, node, |id| arena.offset(id)),
                    });
                }
                Op::ScatterNd { .. }
                | Op::ScatterElements { .. }
                | Op::GatherNd { .. }
                | Op::GatherElements { .. } => {
                    // f32-uniform arena: I64 indices live as f32 slots (same as
                    // CUDA/wgpu). Force the f32→i64 reader for ScatterNd/Gather*.
                    schedule.push(Step::CpuIndexing {
                        thunk: rlx_cpu::rlx_indexing_thunk!(graph, node, |id| arena.offset(id))
                            .force_indices_f32(),
                    });
                }
                Op::Custom { name, attrs, .. } => match name.as_str() {
                    "llada2.group_limited_gate" => {
                        let sig_id = node.inputs[0];
                        let route_id = node.inputs[1];
                        let n_elems = graph.node(sig_id).shape.num_elements().unwrap() as u32;
                        let mut attr_buf = [0u8; 20];
                        let n = attrs.len().min(20);
                        attr_buf[..n].copy_from_slice(&attrs[..n]);
                        schedule.push(Step::Llada2GroupLimitedGate {
                            sig_off: (arena.offset(sig_id) / 4) as u32,
                            route_off: (arena.offset(route_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            n_elems,
                            attrs: attr_buf,
                        });
                    }
                    "umap.knn" => {
                        let pw_id = node.inputs[0];
                        let n = graph.node(pw_id).shape.dims()[0].unwrap_static() as u32;
                        let k = u32::from_le_bytes(attrs[..4].try_into().unwrap());
                        schedule.push(Step::UmapKnn {
                            pairwise_off: (arena.offset(pw_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            n,
                            k,
                        });
                    }
                    "gdino.ms_deform_attn" => {
                        let in_offs: Vec<(u32, u32)> = node
                            .inputs
                            .iter()
                            .map(|&id| {
                                let len = graph.node(id).shape.num_elements().unwrap() as u32;
                                ((arena.offset(id) / 4) as u32, len)
                            })
                            .collect();
                        let out_len = node.shape.num_elements().unwrap() as u32;
                        schedule.push(Step::MsDeformAttnHost {
                            in_offs,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            out_len,
                            attrs: attrs.clone(),
                        });
                    }
                    n if crate::collective_host::COLLECTIVE_OPS.contains(&n) => {
                        // Host-delegate collective (all_reduce / all_gather /
                        // reduce_scatter / f / g): stage off-GPU and run the
                        // registered rlx-cpu collective kernel. Offsets/lengths
                        // are f32 elements (arena convention).
                        let in_id = node.inputs[0];
                        let in_len = graph.node(in_id).shape.num_elements().unwrap() as u32;
                        let out_len = node.shape.num_elements().unwrap() as u32;
                        schedule.push(Step::CollectiveHost {
                            name: n.to_string(),
                            in_off: (arena.offset(in_id) / 4) as u32,
                            in_len,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            out_len,
                            attrs: attrs.clone(),
                        });
                    }
                    other
                        if crate::rocm_gpu_kernels::has_gpu_kernel(other)
                            && node.inputs.len() <= crate::rocm_gpu_kernels::MAX_INPUTS =>
                    {
                        // Raw-GPU custom op: hipRTC kernel launched against the
                        // arena (no host roundtrip). Offsets baked as f32-element
                        // offsets; guarded to ≤ MAX_INPUTS.
                        let in_offs: Vec<(u32, u32)> = node
                            .inputs
                            .iter()
                            .map(|&id| {
                                (
                                    (arena.offset(id) / 4) as u32,
                                    graph.node(id).shape.num_elements().unwrap_or(0) as u32,
                                )
                            })
                            .collect();
                        schedule.push(Step::RocmGpuKernel {
                            name: other.to_string(),
                            out_off: (arena.offset(node.id) / 4) as u32,
                            out_len: node.shape.num_elements().unwrap_or(0) as u32,
                            in_offs,
                        });
                    }
                    other => panic!("rlx-rocm: unsupported Op::Custom('{other}')"),
                },

                Op::GaussianSplatRender {
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    let elem_len = |id: NodeId| -> u32 {
                        graph.node(id).shape.num_elements().unwrap_or(0) as u32
                    };
                    schedule.push(Step::GaussianSplatRender {
                        positions_off: arena.offset(node.inputs[0]) as u32,
                        positions_len: elem_len(node.inputs[0]),
                        scales_off: arena.offset(node.inputs[1]) as u32,
                        scales_len: elem_len(node.inputs[1]),
                        rotations_off: arena.offset(node.inputs[2]) as u32,
                        rotations_len: elem_len(node.inputs[2]),
                        opacities_off: arena.offset(node.inputs[3]) as u32,
                        opacities_len: elem_len(node.inputs[3]),
                        colors_off: arena.offset(node.inputs[4]) as u32,
                        colors_len: elem_len(node.inputs[4]),
                        sh_coeffs_off: arena.offset(node.inputs[5]) as u32,
                        sh_coeffs_len: elem_len(node.inputs[5]),
                        meta_off: arena.offset(node.inputs[6]) as u32,
                        dst_off: arena.offset(node.id) as u32,
                        dst_len: node.shape.num_elements().unwrap_or(0) as u32,
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        radius_scale: *radius_scale,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                    });
                }

                Op::GaussianSplatRenderBackward {
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                    loss_grad_clip,
                    sh_band,
                    max_anisotropy,
                } => {
                    let elem_len = |id: NodeId| -> u32 {
                        graph.node(id).shape.num_elements().unwrap_or(0) as u32
                    };
                    schedule.push(Step::GaussianSplatRenderBackward {
                        positions_off: arena.offset(node.inputs[0]) as u32,
                        positions_len: elem_len(node.inputs[0]),
                        scales_off: arena.offset(node.inputs[1]) as u32,
                        scales_len: elem_len(node.inputs[1]),
                        rotations_off: arena.offset(node.inputs[2]) as u32,
                        rotations_len: elem_len(node.inputs[2]),
                        opacities_off: arena.offset(node.inputs[3]) as u32,
                        opacities_len: elem_len(node.inputs[3]),
                        colors_off: arena.offset(node.inputs[4]) as u32,
                        colors_len: elem_len(node.inputs[4]),
                        sh_coeffs_off: arena.offset(node.inputs[5]) as u32,
                        sh_coeffs_len: elem_len(node.inputs[5]),
                        meta_off: arena.offset(node.inputs[6]) as u32,
                        d_loss_off: arena.offset(node.inputs[7]) as u32,
                        d_loss_len: elem_len(node.inputs[7]),
                        packed_off: arena.offset(node.id) as u32,
                        packed_len: node.shape.num_elements().unwrap_or(0) as u32,
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        radius_scale: *radius_scale,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                        loss_grad_clip: *loss_grad_clip,
                        sh_band: *sh_band,
                        max_anisotropy: *max_anisotropy,
                    });
                }

                Op::GaussianSplatPrepare {
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    let elem_len = |id: NodeId| -> u32 {
                        graph.node(id).shape.num_elements().unwrap_or(0) as u32
                    };
                    schedule.push(Step::GaussianSplatPrepare {
                        positions_off: arena.offset(node.inputs[0]) as u32,
                        positions_len: elem_len(node.inputs[0]),
                        scales_off: arena.offset(node.inputs[1]) as u32,
                        scales_len: elem_len(node.inputs[1]),
                        rotations_off: arena.offset(node.inputs[2]) as u32,
                        rotations_len: elem_len(node.inputs[2]),
                        opacities_off: arena.offset(node.inputs[3]) as u32,
                        opacities_len: elem_len(node.inputs[3]),
                        colors_off: arena.offset(node.inputs[4]) as u32,
                        colors_len: elem_len(node.inputs[4]),
                        sh_coeffs_off: arena.offset(node.inputs[5]) as u32,
                        sh_coeffs_len: elem_len(node.inputs[5]),
                        meta_off: arena.offset(node.inputs[6]) as u32,
                        meta_len: elem_len(node.inputs[6]),
                        prep_off: arena.offset(node.id) as u32,
                        prep_len: node.shape.num_elements().unwrap_or(0) as u32,
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        radius_scale: *radius_scale,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                    });
                }

                Op::GaussianSplatRasterize {
                    width,
                    height,
                    tile_size,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    let elem_len = |id: NodeId| -> u32 {
                        graph.node(id).shape.num_elements().unwrap_or(0) as u32
                    };
                    let prep_id = node.inputs[0];
                    let count = match &graph.node(prep_id).op {
                        rlx_ir::Op::GaussianSplatPrepare { .. } => {
                            elem_len(graph.node(prep_id).inputs[0]) / 3
                        }
                        _ => 1,
                    };
                    schedule.push(Step::GaussianSplatRasterize {
                        prep_off: arena.offset(prep_id) as u32,
                        prep_len: elem_len(prep_id),
                        meta_off: arena.offset(node.inputs[1]) as u32,
                        meta_len: elem_len(node.inputs[1]),
                        dst_off: arena.offset(node.id) as u32,
                        dst_len: node.shape.num_elements().unwrap_or(0) as u32,
                        count,
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                    });
                }

                Op::LayerNorm2d { eps } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    schedule.push(Step::LayerNorm2d {
                        src_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        g_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        b_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                        eps_bits: eps.to_bits(),
                    });
                }
                Op::ConvTranspose2d {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    output_padding: _,
                    groups,
                } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let out_shape = &node.shape;
                    schedule.push(Step::ConvTranspose2d {
                        src_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        w_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c_in: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w_in: in_shape.dim(3).unwrap_static() as u32,
                        c_out: out_shape.dim(1).unwrap_static() as u32,
                        h_out: out_shape.dim(2).unwrap_static() as u32,
                        w_out: out_shape.dim(3).unwrap_static() as u32,
                        kh: kernel_size[0] as u32,
                        kw: kernel_size[1] as u32,
                        sh: stride.first().copied().unwrap_or(1) as u32,
                        sw: stride.get(1).copied().unwrap_or(1) as u32,
                        ph: padding.first().copied().unwrap_or(0) as u32,
                        pw: padding.get(1).copied().unwrap_or(0) as u32,
                        dh: dilation.first().copied().unwrap_or(1) as u32,
                        dw: dilation.get(1).copied().unwrap_or(1) as u32,
                        groups: *groups as u32,
                    });
                }
                Op::ConvTranspose3d {
                    stride,
                    padding,
                    dilation,
                    output_padding: _,
                    groups,
                } => {
                    let in_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    let in_dims = graph.node(in_id).shape.dims();
                    let w_dims = graph.node(w_id).shape.dims();
                    let out_dims = node.shape.dims();
                    schedule.push(Step::ConvTranspose3d {
                        n: in_dims[0].unwrap_static() as u32,
                        c_in: in_dims[1].unwrap_static() as u32,
                        c_out: out_dims[1].unwrap_static() as u32,
                        d: in_dims[2].unwrap_static() as u32,
                        h: in_dims[3].unwrap_static() as u32,
                        w: in_dims[4].unwrap_static() as u32,
                        d_out: out_dims[2].unwrap_static() as u32,
                        h_out: out_dims[3].unwrap_static() as u32,
                        w_out: out_dims[4].unwrap_static() as u32,
                        kd: w_dims[2].unwrap_static() as u32,
                        kh: w_dims[3].unwrap_static() as u32,
                        kw: w_dims[4].unwrap_static() as u32,
                        sd: stride[0] as u32,
                        sh: stride[1] as u32,
                        sw: stride[2] as u32,
                        pd: padding[0] as u32,
                        ph: padding[1] as u32,
                        pw: padding[2] as u32,
                        dd: dilation[0] as u32,
                        dh: dilation[1] as u32,
                        dw: dilation[2] as u32,
                        groups: (*groups).max(1) as u32,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        w_off: (arena.offset(w_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::FusedSwiGLU {
                    cast_to: _,
                    gate_first,
                } => {
                    let n_half = node.shape.dim(node.shape.rank() - 1).unwrap_static() as u32;
                    let total = node.shape.num_elements().unwrap() as u32;
                    schedule.push(Step::FusedSwiGLU {
                        in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        n_half,
                        total,
                        gate_first: if *gate_first { 1 } else { 0 },
                    });
                }
                Op::GroupNorm { num_groups, eps } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    schedule.push(Step::GroupNorm {
                        src_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        g_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        b_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                        num_groups: *num_groups as u32,
                        eps_bits: eps.to_bits(),
                    });
                }
                Op::BatchNormInference { eps } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let rank = in_shape.rank();
                    let channels = in_shape.dim(rank - 1).unwrap_static() as u32;
                    let total = in_shape.num_elements().unwrap_or(0) as u32;
                    schedule.push(Step::BatchNormInference {
                        src_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        g_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        b_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        mean_off: (arena.offset(node.inputs[3]) / 4) as u32,
                        var_off: (arena.offset(node.inputs[4]) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        count: total / channels.max(1),
                        channels,
                        eps_bits: eps.to_bits(),
                    });
                }
                Op::BatchNormInferenceBackwardInput { eps } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let rank = x_shape.rank();
                    let channels = x_shape.dim(rank - 1).unwrap_static() as u32;
                    let total = x_shape.num_elements().unwrap_or(0) as u32;
                    schedule.push(Step::BatchNormInferenceBackwardInput {
                        gamma_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        var_off: (arena.offset(node.inputs[3]) / 4) as u32,
                        dy_off: (arena.offset(node.inputs[4]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        count: total / channels.max(1),
                        channels,
                        eps_bits: eps.to_bits(),
                    });
                }
                Op::BatchNormInferenceBackwardGamma { eps } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let rank = x_shape.rank();
                    let channels = x_shape.dim(rank - 1).unwrap_static() as u32;
                    let total = x_shape.num_elements().unwrap_or(0) as u32;
                    schedule.push(Step::BatchNormInferenceBackwardGamma {
                        x_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        mean_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        var_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        dy_off: (arena.offset(node.inputs[3]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        count: total / channels.max(1),
                        channels,
                        eps_bits: eps.to_bits(),
                    });
                }
                Op::BatchNormInferenceBackwardBeta => {
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let rank = dy_shape.rank();
                    let channels = dy_shape.dim(rank - 1).unwrap_static() as u32;
                    let total = dy_shape.num_elements().unwrap_or(0) as u32;
                    schedule.push(Step::BatchNormInferenceBackwardBeta {
                        dy_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        count: total / channels.max(1),
                        channels,
                    });
                }
                Op::GroupNormBackwardInput { num_groups, eps }
                | Op::GroupNormBackwardGamma { num_groups, eps }
                | Op::GroupNormBackwardBeta { num_groups, eps } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let n = x_shape.dim(0).unwrap_static() as u32;
                    let c = x_shape.dim(1).unwrap_static() as u32;
                    let h = x_shape.dim(2).unwrap_static() as u32;
                    let w = x_shape.dim(3).unwrap_static() as u32;
                    let eps_bits = eps.to_bits();
                    let num_groups = *num_groups as u32;
                    match &node.op {
                        Op::GroupNormBackwardInput { .. } => {
                            schedule.push(Step::GroupNormBackwardInput {
                                x_off: (arena.offset(node.inputs[0]) / 4) as u32,
                                gamma_off: (arena.offset(node.inputs[1]) / 4) as u32,
                                dy_off: (arena.offset(node.inputs[3]) / 4) as u32,
                                out_off: (arena.offset(node.id) / 4) as u32,
                                n,
                                c,
                                h,
                                w,
                                num_groups,
                                eps_bits,
                            });
                        }
                        Op::GroupNormBackwardGamma { .. } => {
                            schedule.push(Step::GroupNormBackwardGamma {
                                x_off: (arena.offset(node.inputs[0]) / 4) as u32,
                                dy_off: (arena.offset(node.inputs[1]) / 4) as u32,
                                out_off: (arena.offset(node.id) / 4) as u32,
                                n,
                                c,
                                h,
                                w,
                                num_groups,
                                eps_bits,
                            });
                        }
                        Op::GroupNormBackwardBeta { .. } => {
                            schedule.push(Step::GroupNormBackwardBeta {
                                dy_off: (arena.offset(node.inputs[1]) / 4) as u32,
                                out_off: (arena.offset(node.id) / 4) as u32,
                                n,
                                c,
                                h,
                                w,
                            });
                        }
                        _ => unreachable!(),
                    }
                }
                Op::LayerNormBackwardInput { eps, .. } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let h = x_shape.dim(x_shape.rank() - 1).unwrap_static() as u32;
                    let rows = (x_shape.num_elements().unwrap() / h.max(1) as usize) as u32;
                    schedule.push(Step::LayerNormBackwardInput {
                        x_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        gamma_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        dy_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        rows,
                        h,
                        eps_bits: eps.to_bits(),
                    });
                }
                Op::LayerNormBackwardGamma { eps, .. } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let h = x_shape.dim(x_shape.rank() - 1).unwrap_static() as u32;
                    let rows = (x_shape.num_elements().unwrap() / h.max(1) as usize) as u32;
                    schedule.push(Step::LayerNormBackwardGamma {
                        x_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        dy_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        rows,
                        h,
                        eps_bits: eps.to_bits(),
                    });
                }
                Op::FakeQuantize {
                    bits,
                    axis,
                    ste: _,
                    scale_mode,
                } => {
                    use rlx_ir::op::ScaleMode;
                    let q_max = match *bits {
                        8 => 127.0f32,
                        4 => 7.0,
                        2 => 1.0,
                        n => panic!("rlx-rocm FakeQuantize: unsupported bits {n}"),
                    };
                    let (chan_dim, inner) = match *axis {
                        None => (1usize, node.shape.num_elements().unwrap_or(0).max(1)),
                        Some(d) => {
                            let chan_dim = node.shape.dim(d).unwrap_static();
                            let inner: usize = (d + 1..node.shape.rank())
                                .map(|i| node.shape.dim(i).unwrap_static())
                                .product::<usize>()
                                .max(1);
                            (chan_dim, inner)
                        }
                    };
                    let n = node.shape.num_elements().unwrap() as u32;
                    let chan_dim = chan_dim as u32;
                    let inner = inner as u32;
                    let q_max_bits = q_max.to_bits();
                    match scale_mode {
                        ScaleMode::Fixed => {
                            schedule.push(Step::FakeQuantizeFixed {
                                in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                                scale_off: (arena.offset(node.inputs[1]) / 4) as u32,
                                out_off: (arena.offset(node.id) / 4) as u32,
                                n,
                                chan_dim,
                                inner,
                                q_max_bits,
                            });
                        }
                        ScaleMode::PerBatch => {
                            schedule.push(Step::FakeQuantizePerBatch {
                                in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                                out_off: (arena.offset(node.id) / 4) as u32,
                                n,
                                chan_dim,
                                inner,
                                q_max_bits,
                            });
                        }
                        ScaleMode::EMA { decay } => {
                            schedule.push(Step::FakeQuantizeEma {
                                in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                                scale_off: (arena.offset(node.inputs[1]) / 4) as u32,
                                out_off: (arena.offset(node.id) / 4) as u32,
                                n,
                                chan_dim,
                                inner,
                                q_max_bits,
                                decay_bits: decay.to_bits(),
                            });
                        }
                    }
                }
                Op::FakeQuantizeLSQ { bits, axis } => {
                    let q_max = match *bits {
                        8 => 127.0f32,
                        4 => 7.0,
                        2 => 1.0,
                        n => panic!("rlx-rocm FakeQuantizeLSQ: unsupported bits {n}"),
                    };
                    let (chan_dim, inner) = match *axis {
                        None => (1usize, node.shape.num_elements().unwrap_or(0).max(1)),
                        Some(d) => {
                            let chan_dim = node.shape.dim(d).unwrap_static();
                            let inner: usize = (d + 1..node.shape.rank())
                                .map(|i| node.shape.dim(i).unwrap_static())
                                .product::<usize>()
                                .max(1);
                            (chan_dim, inner)
                        }
                    };
                    schedule.push(Step::FakeQuantizeFixed {
                        in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        scale_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        n: node.shape.num_elements().unwrap() as u32,
                        chan_dim: chan_dim as u32,
                        inner: inner as u32,
                        q_max_bits: q_max.to_bits(),
                    });
                }
                Op::FakeQuantizeLSQBackwardX { bits, axis } => {
                    let q_max = match *bits {
                        8 => 127.0f32,
                        4 => 7.0,
                        2 => 1.0,
                        n => panic!("rlx-rocm FakeQuantizeLSQBackwardX: unsupported bits {n}"),
                    };
                    let (chan_dim, inner) = match *axis {
                        None => (1usize, node.shape.num_elements().unwrap_or(0).max(1)),
                        Some(d) => {
                            let chan_dim = node.shape.dim(d).unwrap_static();
                            let inner: usize = (d + 1..node.shape.rank())
                                .map(|i| node.shape.dim(i).unwrap_static())
                                .product::<usize>()
                                .max(1);
                            (chan_dim, inner)
                        }
                    };
                    schedule.push(Step::FakeQuantizeLsqBwdX {
                        x_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        scale_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        dy_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        dx_off: (arena.offset(node.id) / 4) as u32,
                        n: node.shape.num_elements().unwrap() as u32,
                        chan_dim: chan_dim as u32,
                        inner: inner as u32,
                        q_max_bits: q_max.to_bits(),
                    });
                }
                Op::FakeQuantizeLSQBackwardScale { bits, axis } => {
                    let q_max = match *bits {
                        8 => 127.0f32,
                        4 => 7.0,
                        2 => 1.0,
                        n => panic!("rlx-rocm FakeQuantizeLSQBackwardScale: unsupported bits {n}"),
                    };
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let (chan_dim, inner) = match *axis {
                        None => (1usize, in_shape.num_elements().unwrap_or(0).max(1)),
                        Some(d) => {
                            let chan_dim = in_shape.dim(d).unwrap_static();
                            let inner: usize = (d + 1..in_shape.rank())
                                .map(|i| in_shape.dim(i).unwrap_static())
                                .product::<usize>()
                                .max(1);
                            (chan_dim, inner)
                        }
                    };
                    schedule.push(Step::FakeQuantizeLsqBwdScale {
                        x_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        scale_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        dy_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        dscale_off: (arena.offset(node.id) / 4) as u32,
                        n: in_shape.num_elements().unwrap() as u32,
                        chan_dim: chan_dim as u32,
                        inner: inner as u32,
                        q_max_bits: q_max.to_bits(),
                    });
                }
                Op::FakeQuantizeBackward { bits, axis, ste } => {
                    use rlx_ir::op::SteKind;
                    let q_max = match *bits {
                        8 => 127.0f32,
                        4 => 7.0,
                        2 => 1.0,
                        n => panic!("rlx-rocm FakeQuantizeBackward: unsupported bits {n}"),
                    };
                    let ste_kind = match ste {
                        SteKind::Identity => 0u32,
                        SteKind::ClippedIdentity => 1,
                        SteKind::Tanh => 2,
                        SteKind::HardTanh => 3,
                    };
                    let (chan_dim, inner) = match *axis {
                        None => (1usize, node.shape.num_elements().unwrap_or(0).max(1)),
                        Some(d) => {
                            let chan_dim = node.shape.dim(d).unwrap_static();
                            let inner: usize = (d + 1..node.shape.rank())
                                .map(|i| node.shape.dim(i).unwrap_static())
                                .product::<usize>()
                                .max(1);
                            (chan_dim, inner)
                        }
                    };
                    schedule.push(Step::FakeQuantizeBackward {
                        x_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        dy_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        dx_off: (arena.offset(node.id) / 4) as u32,
                        n: node.shape.num_elements().unwrap() as u32,
                        chan_dim: chan_dim as u32,
                        inner: inner as u32,
                        q_max_bits: q_max.to_bits(),
                        ste_kind,
                    });
                }
                Op::Quantize {
                    axis,
                    scales,
                    zero_points,
                } => {
                    let (chan_dim, inner) = match *axis {
                        None => (1usize, node.shape.num_elements().unwrap_or(0).max(1)),
                        Some(d) => {
                            let chan_dim = node.shape.dim(d).unwrap_static();
                            let inner: usize = (d + 1..node.shape.rank())
                                .map(|i| node.shape.dim(i).unwrap_static())
                                .product::<usize>()
                                .max(1);
                            (chan_dim, inner)
                        }
                    };
                    debug_assert_eq!(scales.len(), chan_dim);
                    debug_assert_eq!(zero_points.len(), chan_dim);
                    let mut affine = Vec::with_capacity(chan_dim * 2);
                    for c in 0..chan_dim {
                        affine.push(scales[c].to_bits());
                        affine.push(zero_points[c] as u32);
                    }
                    let meta = upload_meta(&ctx, &affine);
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta);
                    schedule.push(Step::QuantizeI8 {
                        in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        q_byte_off: arena.offset(node.id) as u32,
                        n: node.shape.num_elements().unwrap() as u32,
                        chan_dim: chan_dim as u32,
                        inner: inner as u32,
                        meta_idx,
                    });
                }
                Op::Dequantize {
                    axis,
                    scales,
                    zero_points,
                } => {
                    let (chan_dim, inner) = match *axis {
                        None => (1usize, node.shape.num_elements().unwrap_or(0).max(1)),
                        Some(d) => {
                            let chan_dim = node.shape.dim(d).unwrap_static();
                            let inner: usize = (d + 1..node.shape.rank())
                                .map(|i| node.shape.dim(i).unwrap_static())
                                .product::<usize>()
                                .max(1);
                            (chan_dim, inner)
                        }
                    };
                    debug_assert_eq!(scales.len(), chan_dim);
                    debug_assert_eq!(zero_points.len(), chan_dim);
                    let mut affine = Vec::with_capacity(chan_dim * 2);
                    for c in 0..chan_dim {
                        affine.push(scales[c].to_bits());
                        affine.push(zero_points[c] as u32);
                    }
                    let meta = upload_meta(&ctx, &affine);
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta);
                    schedule.push(Step::DequantizeI8 {
                        q_byte_off: arena.offset(node.inputs[0]) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        n: node.shape.num_elements().unwrap() as u32,
                        chan_dim: chan_dim as u32,
                        inner: inner as u32,
                        meta_idx,
                    });
                }
                Op::QMatMul {
                    x_zp,
                    w_zp,
                    out_zp,
                    mult,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let w_shape = &graph.node(node.inputs[1]).shape;
                    schedule.push(Step::QMatMul {
                        m: x_shape.dim(0).unwrap_static() as u32,
                        k: x_shape.dim(1).unwrap_static() as u32,
                        n: w_shape.dim(1).unwrap_static() as u32,
                        x_byte_off: arena.offset(node.inputs[0]) as u32,
                        w_byte_off: arena.offset(node.inputs[1]) as u32,
                        bias_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        out_byte_off: arena.offset(node.id) as u32,
                        x_zp: *x_zp,
                        w_zp: *w_zp,
                        out_zp: *out_zp,
                        mult_bits: mult.to_bits(),
                    });
                }
                Op::QConv2d {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    x_zp,
                    w_zp,
                    out_zp,
                    mult,
                } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let out_shape = &node.shape;
                    schedule.push(Step::QConv2d {
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c_in: in_shape.dim(1).unwrap_static() as u32,
                        c_out: out_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                        h_out: out_shape.dim(2).unwrap_static() as u32,
                        w_out: out_shape.dim(3).unwrap_static() as u32,
                        kh: kernel_size[0] as u32,
                        kw: kernel_size[1] as u32,
                        sh: stride.first().copied().unwrap_or(1) as u32,
                        sw: stride.get(1).copied().unwrap_or(1) as u32,
                        ph: padding.first().copied().unwrap_or(0) as u32,
                        pw: padding.get(1).copied().unwrap_or(0) as u32,
                        dh: dilation.first().copied().unwrap_or(1) as u32,
                        dw: dilation.get(1).copied().unwrap_or(1) as u32,
                        groups: *groups as u32,
                        x_byte_off: arena.offset(node.inputs[0]) as u32,
                        w_byte_off: arena.offset(node.inputs[1]) as u32,
                        bias_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        out_byte_off: arena.offset(node.id) as u32,
                        x_zp: *x_zp,
                        w_zp: *w_zp,
                        out_zp: *out_zp,
                        mult_bits: mult.to_bits(),
                    });
                }
                Op::ResizeNearest2x => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    schedule.push(Step::ResizeNearest2x {
                        src_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                    });
                }
                // Region-marking wraps a bare `Op::ResizeNearest2x` into a
                // single-step `TransformRegion` (same as Metal / rlx-cuda).
                // Unwrap it back to the native 2× nearest upsample step.
                Op::TransformRegion { steps, .. }
                    if steps.len() == 1
                        && matches!(
                            steps[0],
                            rlx_ir::op::TransformStep::ResizeNearest2x(
                                rlx_ir::op::ChainOperand::Input(0)
                            )
                        ) =>
                {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    schedule.push(Step::ResizeNearest2x {
                        src_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                    });
                }
                Op::Interpolate3d { size } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    schedule.push(Step::Interpolate3d {
                        src_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        d_in: in_shape.dim(2).unwrap_static() as u32,
                        h_in: in_shape.dim(3).unwrap_static() as u32,
                        w_in: in_shape.dim(4).unwrap_static() as u32,
                        d_out: size[0] as u32,
                        h_out: size[1] as u32,
                        w_out: size[2] as u32,
                    });
                }

                Op::Pool {
                    kind,
                    kernel_size,
                    stride,
                    padding,
                } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let out_dims = node.shape.dims();
                    let op_id = reduce_op_id(*kind);
                    let in_off = (arena.offset(in_id) / 4) as u32;
                    let out_off = (arena.offset(node.id) / 4) as u32;
                    match kernel_size.len() {
                        1 => schedule.push(Step::Pool1d {
                            n: in_dims[0].unwrap_static() as u32,
                            c: in_dims[1].unwrap_static() as u32,
                            l: in_dims[2].unwrap_static() as u32,
                            l_out: out_dims[2].unwrap_static() as u32,
                            kl: kernel_size[0] as u32,
                            sl: stride[0] as u32,
                            pl: padding[0] as u32,
                            op: op_id,
                            in_off,
                            out_off,
                        }),
                        2 => schedule.push(Step::Pool2d {
                            n: in_dims[0].unwrap_static() as u32,
                            c: in_dims[1].unwrap_static() as u32,
                            h: in_dims[2].unwrap_static() as u32,
                            w: in_dims[3].unwrap_static() as u32,
                            h_out: out_dims[2].unwrap_static() as u32,
                            w_out: out_dims[3].unwrap_static() as u32,
                            kh: kernel_size[0] as u32,
                            kw: kernel_size[1] as u32,
                            sh: stride[0] as u32,
                            sw: stride[1] as u32,
                            ph: padding[0] as u32,
                            pw: padding[1] as u32,
                            op: op_id,
                            in_off,
                            out_off,
                        }),
                        3 => schedule.push(Step::Pool3d {
                            n: in_dims[0].unwrap_static() as u32,
                            c: in_dims[1].unwrap_static() as u32,
                            d: in_dims[2].unwrap_static() as u32,
                            h: in_dims[3].unwrap_static() as u32,
                            w: in_dims[4].unwrap_static() as u32,
                            d_out: out_dims[2].unwrap_static() as u32,
                            h_out: out_dims[3].unwrap_static() as u32,
                            w_out: out_dims[4].unwrap_static() as u32,
                            kd: kernel_size[0] as u32,
                            kh: kernel_size[1] as u32,
                            kw: kernel_size[2] as u32,
                            sd: stride[0] as u32,
                            sh: stride[1] as u32,
                            sw: stride[2] as u32,
                            pd: padding[0] as u32,
                            ph: padding[1] as u32,
                            pw: padding[2] as u32,
                            op: op_id,
                            in_off,
                            out_off,
                        }),
                        other => panic!("rlx-rocm Pool: unsupported kernel rank {other}"),
                    }
                }
                Op::Conv {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    let in_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    let in_dims = graph.node(in_id).shape.dims();
                    let w_dims = graph.node(w_id).shape.dims();
                    let out_dims = node.shape.dims();
                    let in_off = (arena.offset(in_id) / 4) as u32;
                    let w_off = (arena.offset(w_id) / 4) as u32;
                    let out_off = (arena.offset(node.id) / 4) as u32;
                    match kernel_size.len() {
                        1 => schedule.push(Step::Conv1d {
                            n: in_dims[0].unwrap_static() as u32,
                            c_in: in_dims[1].unwrap_static() as u32,
                            c_out: w_dims[0].unwrap_static() as u32,
                            l: in_dims[2].unwrap_static() as u32,
                            l_out: out_dims[2].unwrap_static() as u32,
                            kl: kernel_size[0] as u32,
                            sl: stride[0] as u32,
                            pl: padding[0] as u32,
                            dl: dilation[0] as u32,
                            groups: *groups as u32,
                            in_off,
                            w_off,
                            out_off,
                        }),
                        2 => schedule.push(Step::Conv2d {
                            n: in_dims[0].unwrap_static() as u32,
                            c_in: in_dims[1].unwrap_static() as u32,
                            c_out: w_dims[0].unwrap_static() as u32,
                            h: in_dims[2].unwrap_static() as u32,
                            w: in_dims[3].unwrap_static() as u32,
                            h_out: out_dims[2].unwrap_static() as u32,
                            w_out: out_dims[3].unwrap_static() as u32,
                            kh: kernel_size[0] as u32,
                            kw: kernel_size[1] as u32,
                            sh: stride[0] as u32,
                            sw: stride[1] as u32,
                            ph: padding[0] as u32,
                            pw: padding[1] as u32,
                            dh: dilation[0] as u32,
                            dw: dilation[1] as u32,
                            groups: *groups as u32,
                            in_off,
                            w_off,
                            out_off,
                            has_bias: 0,
                            bias_off_f32: 0,
                            act_id: 0xFFFF,
                            has_residual: 0,
                            residual_off_f32: 0,
                        }),
                        3 => schedule.push(Step::Conv3d {
                            n: in_dims[0].unwrap_static() as u32,
                            c_in: in_dims[1].unwrap_static() as u32,
                            c_out: w_dims[0].unwrap_static() as u32,
                            d: in_dims[2].unwrap_static() as u32,
                            h: in_dims[3].unwrap_static() as u32,
                            w: in_dims[4].unwrap_static() as u32,
                            d_out: out_dims[2].unwrap_static() as u32,
                            h_out: out_dims[3].unwrap_static() as u32,
                            w_out: out_dims[4].unwrap_static() as u32,
                            kd: kernel_size[0] as u32,
                            kh: kernel_size[1] as u32,
                            kw: kernel_size[2] as u32,
                            sd: stride[0] as u32,
                            sh: stride[1] as u32,
                            sw: stride[2] as u32,
                            pd: padding[0] as u32,
                            ph: padding[1] as u32,
                            pw: padding[2] as u32,
                            dd: dilation[0] as u32,
                            dh: dilation[1] as u32,
                            dw: dilation[2] as u32,
                            groups: *groups as u32,
                            in_off,
                            w_off,
                            out_off,
                        }),
                        other => panic!("rlx-rocm Conv: unsupported kernel rank {other}"),
                    }
                }
                Op::Conv3d {
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    let in_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    let in_dims = graph.node(in_id).shape.dims();
                    let w_dims = graph.node(w_id).shape.dims();
                    let out_dims = node.shape.dims();
                    schedule.push(Step::Conv3d {
                        n: in_dims[0].unwrap_static() as u32,
                        c_in: in_dims[1].unwrap_static() as u32,
                        c_out: w_dims[0].unwrap_static() as u32,
                        d: in_dims[2].unwrap_static() as u32,
                        h: in_dims[3].unwrap_static() as u32,
                        w: in_dims[4].unwrap_static() as u32,
                        d_out: out_dims[2].unwrap_static() as u32,
                        h_out: out_dims[3].unwrap_static() as u32,
                        w_out: out_dims[4].unwrap_static() as u32,
                        kd: w_dims[2].unwrap_static() as u32,
                        kh: w_dims[3].unwrap_static() as u32,
                        kw: w_dims[4].unwrap_static() as u32,
                        sd: stride[0] as u32,
                        sh: stride[1] as u32,
                        sw: stride[2] as u32,
                        pd: padding[0] as u32,
                        ph: padding[1] as u32,
                        pw: padding[2] as u32,
                        dd: dilation[0] as u32,
                        dh: dilation[1] as u32,
                        dw: dilation[2] as u32,
                        groups: *groups as u32,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        w_off: (arena.offset(w_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::Sample {
                    top_k,
                    top_p,
                    temperature,
                    seed,
                } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let outer = in_dims[..in_dims.len() - 1]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let is_greedy = *top_k == 0
                        && (*top_p - 1.0).abs() < 1e-6
                        && (*temperature - 1.0).abs() < 1e-6;
                    if is_greedy {
                        schedule.push(Step::Argmax {
                            outer,
                            inner,
                            in_off: (arena.offset(in_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                        });
                    } else {
                        schedule.push(Step::Sample {
                            outer,
                            inner,
                            in_off: (arena.offset(in_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            top_k: *top_k as u32,
                            top_p_bits: top_p.to_bits(),
                            temp_bits: temperature.to_bits(),
                            seed_lo: *seed as u32,
                            seed_hi: (*seed >> 32) as u32,
                        });
                    }
                }
                Op::RngNormal {
                    mean,
                    scale,
                    key,
                    op_seed,
                } => {
                    let len = node.shape.num_elements().unwrap_or(0);
                    schedule.push(Step::RngNormal {
                        dst_byte_off: arena.offset(node.id) as u32,
                        len: len as u32,
                        mean: *mean,
                        scale: *scale,
                        key: *key,
                        op_seed: *op_seed,
                    });
                }
                Op::RngUniform {
                    low,
                    high,
                    key,
                    op_seed,
                } => {
                    let len = node.shape.num_elements().unwrap_or(0);
                    schedule.push(Step::RngUniform {
                        dst_byte_off: arena.offset(node.id) as u32,
                        len: len as u32,
                        low: *low,
                        high: *high,
                        key: *key,
                        op_seed: *op_seed,
                    });
                }
                Op::RmsNormBackwardInput { eps, .. }
                | Op::RmsNormBackwardGamma { eps, .. }
                | Op::RmsNormBackwardBeta { eps, .. } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let h = x_shape.dim(x_shape.rank() - 1).unwrap_static() as u32;
                    let rows = (x_shape.num_elements().unwrap() / h.max(1) as usize) as u32;
                    let eps_bits = eps.to_bits();
                    let off = |i: usize| arena.offset(node.inputs[i]) as u32;
                    let common = (off(0), off(1), off(2), off(3), rows, h, eps_bits);
                    match &node.op {
                        Op::RmsNormBackwardInput { .. } => {
                            schedule.push(Step::RmsNormBackwardInput {
                                x_byte_off: common.0,
                                gamma_byte_off: common.1,
                                beta_byte_off: common.2,
                                dy_byte_off: common.3,
                                dx_byte_off: arena.offset(node.id) as u32,
                                rows: common.4,
                                h: common.5,
                                eps_bits: common.6,
                            });
                        }
                        Op::RmsNormBackwardGamma { .. } => {
                            schedule.push(Step::RmsNormBackwardGamma {
                                x_byte_off: common.0,
                                gamma_byte_off: common.1,
                                beta_byte_off: common.2,
                                dy_byte_off: common.3,
                                dgamma_byte_off: arena.offset(node.id) as u32,
                                rows: common.4,
                                h: common.5,
                                eps_bits: common.6,
                            });
                        }
                        Op::RmsNormBackwardBeta { .. } => {
                            schedule.push(Step::RmsNormBackwardBeta {
                                x_byte_off: common.0,
                                gamma_byte_off: common.1,
                                beta_byte_off: common.2,
                                dy_byte_off: common.3,
                                dbeta_byte_off: arena.offset(node.id) as u32,
                                rows: common.4,
                                h: common.5,
                                eps_bits: common.6,
                            });
                        }
                        _ => unreachable!(),
                    }
                }
                Op::RopeBackward { head_dim, n_rot } => {
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let (batch, seq, hidden) = if dy_shape.rank() >= 3 {
                        (
                            dy_shape.dim(0).unwrap_static() as u32,
                            dy_shape.dim(1).unwrap_static() as u32,
                            dy_shape.dim(2).unwrap_static() as u32,
                        )
                    } else {
                        (
                            1,
                            dy_shape.dim(0).unwrap_static() as u32,
                            dy_shape.dim(1).unwrap_static() as u32,
                        )
                    };
                    let cos_len = graph.node(node.inputs[1]).shape.num_elements().unwrap() as u32;
                    schedule.push(Step::RopeBackward {
                        dy_byte_off: arena.offset(node.inputs[0]) as u32,
                        cos_byte_off: arena.offset(node.inputs[1]) as u32,
                        sin_byte_off: arena.offset(node.inputs[2]) as u32,
                        dx_byte_off: arena.offset(node.id) as u32,
                        batch,
                        seq,
                        hidden,
                        head_dim: *head_dim as u32,
                        n_rot: *n_rot as u32,
                        cos_len,
                    });
                }
                Op::CumsumBackward { exclusive, .. } => {
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let cols = dy_shape.dim(dy_shape.rank() - 1).unwrap_static() as u32;
                    let rows = (dy_shape.num_elements().unwrap() / cols.max(1) as usize) as u32;
                    schedule.push(Step::CumsumBackward {
                        dy_byte_off: arena.offset(node.inputs[0]) as u32,
                        dx_byte_off: arena.offset(node.id) as u32,
                        rows,
                        cols,
                        exclusive: *exclusive,
                    });
                }
                Op::GatherBackward { .. } => {
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let idx_shape = &graph.node(node.inputs[1]).shape;
                    let out_shape = &node.shape;
                    let rank = out_shape.rank();
                    let axis = match &node.op {
                        Op::GatherBackward { axis } => *axis,
                        _ => 0,
                    };
                    let axis_u = if axis < 0 {
                        (rank as i32 + axis) as usize
                    } else {
                        axis as usize
                    };
                    let outer: usize = (0..axis_u)
                        .map(|i| dy_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let num_idx = idx_shape.dim(axis_u).unwrap_static();
                    let trailing: usize = (axis_u + 1..dy_shape.rank())
                        .map(|i| dy_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let axis_dim = out_shape.dim(axis_u).unwrap_static();
                    schedule.push(Step::GatherBackward {
                        dy_byte_off: arena.offset(node.inputs[0]) as u32,
                        indices_byte_off: arena.offset(node.inputs[1]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        outer: outer as u32,
                        axis_dim: axis_dim as u32,
                        num_idx: num_idx as u32,
                        trailing: trailing as u32,
                    });
                }
                Op::Conv2dBackwardInput {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let out_shape = &node.shape;
                    if kernel_size.len() == 2 && dy_shape.rank() == 4 && out_shape.rank() == 4 {
                        schedule.push(Step::Conv2dBackwardInput {
                            dy_byte_off: arena.offset(node.inputs[0]) as u64,
                            w_byte_off: arena.offset(node.inputs[1]) as u64,
                            dx_byte_off: arena.offset(node.id) as u64,
                            n: out_shape.dim(0).unwrap_static() as u32,
                            c_in: out_shape.dim(1).unwrap_static() as u32,
                            h: out_shape.dim(2).unwrap_static() as u32,
                            w_in: out_shape.dim(3).unwrap_static() as u32,
                            c_out: dy_shape.dim(1).unwrap_static() as u32,
                            h_out: dy_shape.dim(2).unwrap_static() as u32,
                            w_out: dy_shape.dim(3).unwrap_static() as u32,
                            kh: kernel_size[0] as u32,
                            kw: kernel_size[1] as u32,
                            sh: stride.first().copied().unwrap_or(1) as u32,
                            sw: stride.get(1).copied().unwrap_or(1) as u32,
                            ph: padding.first().copied().unwrap_or(0) as u32,
                            pw: padding.get(1).copied().unwrap_or(0) as u32,
                            dh: dilation.first().copied().unwrap_or(1) as u32,
                            dw: dilation.get(1).copied().unwrap_or(1) as u32,
                            groups: *groups as u32,
                        });
                    } else {
                        panic!("rlx-rocm: Conv2dBackwardInput expects 2-D conv on NCHW tensors");
                    }
                }
                Op::Conv2dBackwardWeight {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let dy_shape = &graph.node(node.inputs[1]).shape;
                    if kernel_size.len() == 2 && x_shape.rank() == 4 && dy_shape.rank() == 4 {
                        schedule.push(Step::Conv2dBackwardWeight {
                            x_byte_off: arena.offset(node.inputs[0]) as u64,
                            dy_byte_off: arena.offset(node.inputs[1]) as u64,
                            dw_byte_off: arena.offset(node.id) as u64,
                            n: x_shape.dim(0).unwrap_static() as u32,
                            c_in: x_shape.dim(1).unwrap_static() as u32,
                            h: x_shape.dim(2).unwrap_static() as u32,
                            w: x_shape.dim(3).unwrap_static() as u32,
                            c_out: dy_shape.dim(1).unwrap_static() as u32,
                            h_out: dy_shape.dim(2).unwrap_static() as u32,
                            w_out: dy_shape.dim(3).unwrap_static() as u32,
                            kh: kernel_size[0] as u32,
                            kw: kernel_size[1] as u32,
                            sh: stride.first().copied().unwrap_or(1) as u32,
                            sw: stride.get(1).copied().unwrap_or(1) as u32,
                            ph: padding.first().copied().unwrap_or(0) as u32,
                            pw: padding.get(1).copied().unwrap_or(0) as u32,
                            dh: dilation.first().copied().unwrap_or(1) as u32,
                            dw_dil: dilation.get(1).copied().unwrap_or(1) as u32,
                            groups: *groups as u32,
                        });
                    } else {
                        panic!("rlx-rocm: Conv2dBackwardWeight expects 2-D conv on NCHW tensors");
                    }
                }
                Op::MaxPool2dBackward {
                    kernel_size,
                    stride,
                    padding,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let dy_shape = &graph.node(node.inputs[1]).shape;
                    if kernel_size.len() == 2 && x_shape.rank() == 4 && dy_shape.rank() == 4 {
                        schedule.push(Step::MaxPool2dBackward {
                            x_byte_off: arena.offset(node.inputs[0]) as u64,
                            dy_byte_off: arena.offset(node.inputs[1]) as u64,
                            dx_byte_off: arena.offset(node.id) as u64,
                            n: x_shape.dim(0).unwrap_static() as u32,
                            c: x_shape.dim(1).unwrap_static() as u32,
                            h: x_shape.dim(2).unwrap_static() as u32,
                            w: x_shape.dim(3).unwrap_static() as u32,
                            h_out: dy_shape.dim(2).unwrap_static() as u32,
                            w_out: dy_shape.dim(3).unwrap_static() as u32,
                            kh: kernel_size[0] as u32,
                            kw: kernel_size[1] as u32,
                            sh: stride.first().copied().unwrap_or(1) as u32,
                            sw: stride.get(1).copied().unwrap_or(1) as u32,
                            ph: padding.first().copied().unwrap_or(0) as u32,
                            pw: padding.get(1).copied().unwrap_or(0) as u32,
                        });
                    } else {
                        panic!("rlx-rocm: MaxPool2dBackward expects 2-D pool on NCHW tensors");
                    }
                }
                Op::MaxPool3dBackward {
                    kernel_size,
                    stride,
                    padding,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let dy_shape = &graph.node(node.inputs[1]).shape;
                    schedule.push(Step::MaxPool3dBackward {
                        x_byte_off: arena.offset(node.inputs[0]) as u64,
                        dy_byte_off: arena.offset(node.inputs[1]) as u64,
                        dx_byte_off: arena.offset(node.id) as u64,
                        n: x_shape.dim(0).unwrap_static() as u32,
                        c: x_shape.dim(1).unwrap_static() as u32,
                        d: x_shape.dim(2).unwrap_static() as u32,
                        h: x_shape.dim(3).unwrap_static() as u32,
                        w: x_shape.dim(4).unwrap_static() as u32,
                        d_out: dy_shape.dim(2).unwrap_static() as u32,
                        h_out: dy_shape.dim(3).unwrap_static() as u32,
                        w_out: dy_shape.dim(4).unwrap_static() as u32,
                        kd: kernel_size[0] as u32,
                        kh: kernel_size[1] as u32,
                        kw: kernel_size[2] as u32,
                        sd: stride[0] as u32,
                        sh: stride[1] as u32,
                        sw: stride[2] as u32,
                        pd: padding[0] as u32,
                        ph: padding[1] as u32,
                        pw: padding[2] as u32,
                    });
                }
                Op::Conv3dBackwardInput {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let out_shape = &node.shape;
                    schedule.push(Step::Conv3dBackwardInput {
                        dy_byte_off: arena.offset(node.inputs[0]) as u64,
                        w_byte_off: arena.offset(node.inputs[1]) as u64,
                        dx_byte_off: arena.offset(node.id) as u64,
                        n: out_shape.dim(0).unwrap_static() as u32,
                        c_in: out_shape.dim(1).unwrap_static() as u32,
                        d: out_shape.dim(2).unwrap_static() as u32,
                        h: out_shape.dim(3).unwrap_static() as u32,
                        w_in: out_shape.dim(4).unwrap_static() as u32,
                        c_out: dy_shape.dim(1).unwrap_static() as u32,
                        d_out: dy_shape.dim(2).unwrap_static() as u32,
                        h_out: dy_shape.dim(3).unwrap_static() as u32,
                        w_out: dy_shape.dim(4).unwrap_static() as u32,
                        kd: kernel_size[0] as u32,
                        kh: kernel_size[1] as u32,
                        kw: kernel_size[2] as u32,
                        sd: stride[0] as u32,
                        sh: stride[1] as u32,
                        sw: stride[2] as u32,
                        pd: padding[0] as u32,
                        ph: padding[1] as u32,
                        pw: padding[2] as u32,
                        dd: dilation[0] as u32,
                        dh: dilation[1] as u32,
                        dw: dilation[2] as u32,
                        groups: *groups as u32,
                    });
                }
                Op::Conv3dBackwardWeight {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let dy_shape = &graph.node(node.inputs[1]).shape;
                    schedule.push(Step::Conv3dBackwardWeight {
                        x_byte_off: arena.offset(node.inputs[0]) as u64,
                        dy_byte_off: arena.offset(node.inputs[1]) as u64,
                        dw_byte_off: arena.offset(node.id) as u64,
                        n: x_shape.dim(0).unwrap_static() as u32,
                        c_in: x_shape.dim(1).unwrap_static() as u32,
                        d: x_shape.dim(2).unwrap_static() as u32,
                        h: x_shape.dim(3).unwrap_static() as u32,
                        w: x_shape.dim(4).unwrap_static() as u32,
                        c_out: dy_shape.dim(1).unwrap_static() as u32,
                        d_out: dy_shape.dim(2).unwrap_static() as u32,
                        h_out: dy_shape.dim(3).unwrap_static() as u32,
                        w_out: dy_shape.dim(4).unwrap_static() as u32,
                        kd: kernel_size[0] as u32,
                        kh: kernel_size[1] as u32,
                        kw: kernel_size[2] as u32,
                        sd: stride[0] as u32,
                        sh: stride[1] as u32,
                        sw: stride[2] as u32,
                        pd: padding[0] as u32,
                        ph: padding[1] as u32,
                        pw: padding[2] as u32,
                        dd: dilation[0] as u32,
                        dh: dilation[1] as u32,
                        dw_dil: dilation[2] as u32,
                        groups: *groups as u32,
                    });
                }
                // Native batched symmetric eigendecomposition: `Op::Eigh` /
                // `Op::EighBatch` (n ≤ 32) run on-device via hipSOLVER
                // `SsyevjBatched` when libhipsolver is loadable. Larger `n` or
                // missing hipSOLVER falls through to the CPU host-fallback
                // below. Input `[n,n]` (Eigh, batch=1) or `[B,n,n]` (EighBatch);
                // output packed `[.. , n²+n]`.
                op @ (Op::Eigh | Op::EighBatch)
                    if {
                        let s = &graph.node(node.inputs[0]).shape;
                        let n = s.dim(s.rank().saturating_sub(1)).unwrap_static();
                        n <= crate::eigh_native::MAX_N && crate::eigh_native::is_available()
                    } =>
                {
                    let in_shape = graph.node(node.inputs[0]).shape.clone();
                    let n = in_shape
                        .dim(in_shape.rank().saturating_sub(1))
                        .unwrap_static();
                    let batch = if matches!(op, Op::EighBatch) {
                        in_shape.dim(0).unwrap_static()
                    } else {
                        1
                    };
                    schedule.push(Step::EighNative {
                        in_off: arena.offset(node.inputs[0]) / 4,
                        out_off: arena.offset(node.id) / 4,
                        n,
                        batch,
                    });
                }
                // Core Riemannian / SPD-manifold ops (F64, no ROCm kernel) run
                // on the CPU reference between GPU segments (D2H → CPU → H2D),
                // like `Op::Scan` / `Op::Fft`. Delegating through the CPU thunk
                // with each node's REAL declared dtype/shape handles the packed
                // `[2n²+n]` ReEig/LogEig forward output and the precomputed
                // backward layout for free — no shapes are hardcoded here.
                op if crate::spd::is_spd_host(op) => {
                    let inputs: Vec<(usize, rlx_ir::Shape)> = node
                        .inputs
                        .iter()
                        .map(|&id| (arena.offset(id) / 4, graph.node(id).shape.clone()))
                        .collect();
                    schedule.push(Step::SpdHost {
                        op: op.clone(),
                        out_off: arena.offset(node.id) / 4,
                        out_shape: node.shape.clone(),
                        inputs,
                    });
                }
                // C64 Wirtinger surface — native `complex_wirtinger.cu` (shared
                // with CUDA). Interleaved [re, im] pairs; `elems` is the
                // complex-element count (output of ComplexNormSq is real F32).
                Op::ComplexNormSq => {
                    let src = node.inputs[0];
                    if graph.node(src).shape.dtype() != rlx_ir::DType::C64 {
                        panic!(
                            "rlx-rocm ComplexNormSq: expected C64 input, got {:?}",
                            graph.node(src).shape.dtype()
                        );
                    }
                    schedule.push(Step::ComplexNormSq {
                        n: elems,
                        src_byte_off: arena.offset(src) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                    });
                }
                Op::ComplexNormSqBackward => {
                    let z = node.inputs[0];
                    let g = node.inputs[1];
                    if graph.node(z).shape.dtype() != rlx_ir::DType::C64 {
                        panic!(
                            "rlx-rocm ComplexNormSqBackward: expected C64 z, got {:?}",
                            graph.node(z).shape.dtype()
                        );
                    }
                    schedule.push(Step::ComplexNormSqBackward {
                        n: elems,
                        z_byte_off: arena.offset(z) as u32,
                        g_byte_off: arena.offset(g) as u32,
                        dz_byte_off: arena.offset(node.id) as u32,
                    });
                }
                Op::Conjugate => {
                    let src = node.inputs[0];
                    if graph.node(src).shape.dtype() != rlx_ir::DType::C64 {
                        panic!(
                            "rlx-rocm Conjugate: expected C64 input, got {:?}",
                            graph.node(src).shape.dtype()
                        );
                    }
                    schedule.push(Step::ConjugateC64 {
                        n: elems,
                        src_byte_off: arena.offset(src) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                    });
                }
                // Native F32 dense solve via hipSOLVER getrf+getrs /
                // hipBLAS getrfBatched+getrsBatched when libraries load.
                // Otherwise (or F64) fall through to HostOp.
                Op::DenseSolve
                    if node.shape.dtype() == rlx_ir::DType::F32
                        && crate::dense_solve_native::is_available() =>
                {
                    let a_shape = &graph.node(node.inputs[0]).shape;
                    let n = a_shape.dim(0).unwrap_static();
                    let b_elems = node.shape.num_elements().unwrap();
                    let nrhs = b_elems / n;
                    schedule.push(Step::DenseSolveNative {
                        a_off: arena.offset(node.inputs[0]) / 4,
                        b_off: arena.offset(node.inputs[1]) / 4,
                        x_off: arena.offset(node.id) / 4,
                        n,
                        nrhs,
                    });
                }
                Op::BatchedDenseSolve
                    if node.shape.dtype() == rlx_ir::DType::F32
                        && crate::device::rocm_blas().as_ref().is_some_and(|b| {
                            crate::hipblas::batched_lu_available(&b.lock().unwrap().runtime)
                        }) =>
                {
                    let a_shape = &graph.node(node.inputs[0]).shape;
                    let batch = a_shape.dim(0).unwrap_static();
                    let n = a_shape.dim(1).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    let nrhs = total / (batch * n);
                    schedule.push(Step::BatchedDenseSolveNative {
                        a_off: arena.offset(node.inputs[0]) / 4,
                        b_off: arena.offset(node.inputs[1]) / 4,
                        x_off: arena.offset(node.id) / 4,
                        batch,
                        n,
                        nrhs,
                    });
                }
                // Host-staged ops (D2H → CPU `eval_single_op_f32` → H2D), same
                // catch-all as wgpu `lower.rs`. DenseSolve / BatchedDenseSolve
                // fall here when native libs/dtypes are unavailable; CustomFn
                // runs the opaque body. `PartitionedConv` is expanded to
                // Fft/MatMul in `crate::unfuse` before this match.
                Op::DenseSolve
                | Op::BatchedDenseSolve
                | Op::Cholesky
                | Op::TriangularSolve { .. }
                | Op::Det
                | Op::LogDet
                | Op::Sort { .. }
                | Op::Svd { .. }
                | Op::Qr { .. }
                | Op::ArgSort { .. }
                | Op::CustomFn { .. } => {
                    schedule.push(Step::HostOp {
                        desc: rlx_cpu::rlx_host_op_desc!(graph, node, |id| arena.offset(id)),
                    });
                }
                other => panic!(
                    "rlx-rocm: op {other:?} not yet lowered. \
                     Open a follow-up PR if you hit this — every other op \
                     in the IR is wired."
                ),
            }
        }

        let schedule = fuse_elementwise_chains(schedule);
        let blas = rocm_blas();
        let blas_lt = rocm_blas_lt();
        let blas_lt_workspace = if blas_lt.is_some() {
            HipBuffer::<u8>::alloc_zeros(&ctx.runtime, HIPBLASLT_WORKSPACE_BYTES).ok()
        } else {
            None
        };
        let dnn = rocm_dnn();
        let dnn_workspace = if dnn.is_some() {
            HipBuffer::<u8>::alloc_zeros(&ctx.runtime, MIOPEN_WORKSPACE_BYTES).ok()
        } else {
            None
        };

        // Stream pool for MultiStream(n). Allocated up-front so the
        // scheduler doesn't pay creation cost per run().
        let mut streams: Vec<crate::hip::HipStream> = Vec::new();
        if let ExecMode::MultiStream(n) = exec_mode
            && n > 1
        {
            for _ in 0..n {
                let mut s: crate::hip::HipStream = std::ptr::null_mut();
                unsafe {
                    if (ctx.runtime.hip_stream_create)(&mut s).ok().is_ok() {
                        streams.push(s);
                    }
                }
            }
        }

        // Host staging is sized by LANE count, not element count: a complex
        // output occupies 2 (C64) / 4 (C128) f32 lanes per element, so
        // `num_elements` would truncate the readback to the real parts (the
        // "biggest gotcha"). `arena_lane_count` == `num_elements` for every
        // non-complex dtype, so this is a no-op for real tensors.
        let output_staging: Vec<F32HostSlot> = graph
            .outputs
            .iter()
            .map(|&id| {
                let lanes = arena_lane_count(&graph.node(id).shape);
                F32HostSlot::new(&ctx.runtime, lanes, pinned_io_enabled(exec_mode))
            })
            .collect();

        let mut input_staging = HashMap::new();
        if pinned_io_enabled(exec_mode) {
            for (name, &id) in &input_offsets {
                let lanes = arena_lane_count(&graph.node(id).shape);
                input_staging.insert(name.clone(), F32HostSlot::new(&ctx.runtime, lanes, true));
            }
        }

        let mut input_slot_names = Vec::new();
        let mut input_slots = Vec::new();
        for node in graph.nodes() {
            if let Op::Input { name } = &node.op {
                let off = if arena.has(node.id) {
                    arena.offset(node.id)
                } else {
                    0
                };
                // Lane count: a complex input feeds 2/4 f32 lanes per element, so
                // the upload bound must count lanes or the input is truncated.
                let len = arena_lane_count(&node.shape);
                input_slot_names.push(name.clone());
                input_slots.push((off, len));
            }
        }

        let mut host_total = 0usize;
        let mut output_slots = Vec::new();
        for &id in &graph.outputs {
            let n = arena_lane_count(&graph.node(id).shape);
            output_slots.push((host_total * 4, n));
            host_total += n;
        }
        let host_arena = vec![0.0f32; host_total];

        Self {
            ctx,
            blas,
            blas_lt,
            blas_lt_workspace,
            dnn,
            dnn_workspace,
            dequant_scratch_off,
            graph,
            arena,
            schedule,
            input_offsets,
            param_offsets,
            meta_buffers,
            exec_mode,
            half_act_scratch: None,
            captured_graph: None,
            streams,
            active_extent: None,
            output_staging,
            input_staging,
            gpu_handles: HashMap::new(),
            gpu_handle_feeds: HashMap::new(),
            kv_row_feeds: HashMap::new(),
            gpu_handle_resident: std::collections::HashSet::new(),
            pending_read_indices: None,
            input_slot_names,
            input_slots,
            output_slots,
            host_arena,
            rng: std::sync::Arc::new(std::sync::RwLock::new(rng)),
        }
    }

    pub fn compile_with(graph: Graph, compile_mode: CompileMode, exec_mode: ExecMode) -> Self {
        Self::compile_with_rng(
            graph,
            compile_mode,
            exec_mode,
            rlx_ir::RngOptions::default(),
        )
    }
}
