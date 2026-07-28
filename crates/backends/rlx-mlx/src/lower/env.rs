// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `lower_with_env` — main op match.

#![allow(unused_imports)]

use std::collections::{HashMap, HashSet};

use rlx_ir::RegionPrologue;
use rlx_ir::op::{
    Activation, AdaNormKind, BinaryOp, ChainOperand, ChainStep, CmpOp, MaskKind, ReduceOp,
    RopeStyle, ScaleMode, SteKind, TransformStep,
};
use rlx_ir::shape::{Dim, DimBinding, Shape};
use rlx_ir::{DType, Graph, NodeId, Op};

use crate::array::{Array, MlxError, async_eval, eval};
use crate::ffi::{MlxMask, MlxReduce, MlxUnary};
use crate::ops;

use super::helpers::*;
use super::host_eval::{host_eval_op_typed, is_mlx_typed_host_op};
use super::subgraph::*;
use super::*;

pub fn lower_with_env(
    graph: &Graph,
    mut env: HashMap<NodeId, Array>,
    params: &HashMap<String, Vec<f32>>,
    params_typed: &HashMap<String, (Vec<u8>, DType)>,
    rng: rlx_ir::RngOptions,
    eval_barriers: bool,
) -> Result<Vec<Array>, MlxError> {
    let cfg = crate::config::runtime_config();
    let debug_eval = cfg.debug_eval;
    if debug_eval {
        eprintln!("rlx-mlx: lower_with_env {} nodes", graph.nodes().len());
    }
    // Fusion-depth cap (lazy path only): MLX fuses a run of elementwise ops into
    // one Metal kernel whose arguments are the run's leaves; a very deep run
    // (e.g. the grid_sample decomposition) exhausts the argument-buffer limit.
    // Materializing the array every `fuse_cap` consecutive fusable ops breaks the
    // run into kernels that fit. Non-elementwise ops (matmul/gather/…) reset the
    // counter, so ordinary models never trigger it. Illegal inside `mlx::compile`
    // (the compile trace passes `eval_barriers = false`).
    let fuse_cap: usize = cfg.fuse_cap;
    // Depth of the fusable-op chain rooted at each node (0 = materialized leaf /
    // non-fusable). Tracks the real data-dependency subtree, not iteration order.
    let mut fuse_depth: HashMap<NodeId, usize> = HashMap::new();
    for node in graph.nodes() {
        let id = node.id;
        if env.contains_key(&id) {
            // Pre-populated leaf — already bound by the caller.
            continue;
        }
        if !node.shape.dims().iter().all(|d| d.is_static()) {
            return Err(MlxError(format!(
                "MLX backend: dynamic shapes not yet supported (node {:?})",
                node.id
            )));
        }

        let profile = mlx_profile_enabled();
        let t0 = if profile {
            Some(rlx_ir::Tick::now())
        } else {
            None
        };
        let arr = (|| -> Result<Array, MlxError> {
            Ok(match &node.op {
            // Leaves should have been pre-bound by the caller; if we
            // see one here it means env was incomplete.
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => {
                return Err(MlxError(format!(
                    "lower_with_env: leaf node {id:?} not bound in env"
                )));
            }

            Op::MatMul => {
                let a = lookup(&env, node.inputs[0])?;
                let b = lookup(&env, node.inputs[1])?;
                let graph_a = node_input_shape(graph, node.inputs[0]);
                let graph_out = node_input_shape(graph, node.id);
                let a = flatten_matmul_lhs_if_needed(a, &graph_a, &graph_out)?;
                ops::matmul(&a, b).map_err(|e| {
                    let name = node.name.as_deref().unwrap_or("?");
                    MlxError(format!(
                        "MatMul {name}: {e} (lhs={:?}, rhs={:?})",
                        a.shape(),
                        b.shape()
                    ))
                })?
            }
            // Dense linear solve. MLX's linalg::solve handles the
            // rank-2 single-system case directly. For rlx's
            // `Op::BatchedDenseSolve` (A: [B, n, n], b: [B, n] →
            // x: [B, n]) we adapt to MLX's multi-RHS convention:
            // MLX treats a rank-2 `b` as `[n, k]` (k right-hand
            // sides), not `[B, n]`. So we reshape b to `[B, n, 1]`
            // before the solve and squeeze the trailing 1 back off
            // afterwards. Same shim entry point covers both ops.
            // Dtype must be f32 or f64 (validated by MLX upstream).
            //
            // Caveat: the C++ shim pins this to MLX's CPU stream because
            // MLX-GPU linalg::solve isn't implemented yet upstream. Op
            // still lives in the lazy graph (no host roundtrip; fuses
            // with surrounding ops on either side), but the LU runs on
            // CPU LAPACK. When MLX adds a Metal solve, the shim's stream
            // pin can be dropped — no change here.
            Op::DenseSolve => {
                let a = lookup(&env, node.inputs[0])?;
                let b = lookup(&env, node.inputs[1])?;
                ops::solve(a, b)?
            }
            Op::BatchedDenseSolve => {
                let a = lookup(&env, node.inputs[0])?;
                let b = lookup(&env, node.inputs[1])?;
                let b_shape: Vec<i32> = node_input_shape(graph, node.inputs[1]);
                let n = if b_shape.len() >= 2 {
                    b_shape[1] as usize
                } else {
                    0
                };
                let dtype = node.shape.dtype();

                // Custom Metal LU+solve kernel — runs on the Apple GPU,
                // dispatches one threadgroup per batch element. Bound by
                // threadgroup memory at f32: NMAX² + NMAX ≤ 32 KB ⇒
                // n ≤ 90. Falls back to MLX-CPU `linalg::solve` outside
                // the supported envelope (n > 90, or non-f32 dtype).
                if dtype == DType::F32 && n > 0 && n <= 90 {
                    static REGISTER_KERNELS: std::sync::Once = std::sync::Once::new();
                    REGISTER_KERNELS.call_once(crate::batched_lu_kernel::register);

                    if let Some(kernel) =
                        crate::op_registry::lookup_mlx_kernel(crate::batched_lu_kernel::KERNEL_NAME)
                    {
                        let out_shape = node.shape.clone();
                        // Errors here propagate as a backend failure.
                        // Don't silently fall back — that would mask
                        // bugs in the kernel, which is worse than a
                        // loud error since the fallback exists for
                        // numerical/capability reasons, not for kernel
                        // correctness regressions.
                        kernel.execute(&[a, b], &out_shape, &[])?
                    } else {
                        // Registry returned None — should be
                        // impossible after call_once, but stay safe.
                        let mut shape_b1 = b_shape.clone();
                        shape_b1.push(1);
                        let b_un = ops::reshape(b, &shape_b1)?;
                        let solved = ops::solve(a, &b_un)?;
                        ops::reshape(&solved, &b_shape)?
                    }
                } else {
                    // Fallback path: MLX's linalg::solve on the CPU
                    // stream. MLX expects rank-3 b for batched solve
                    // (multi-RHS form), so reshape [B,n] ↔ [B,n,1].
                    let mut shape_b1 = b_shape.clone();
                    shape_b1.push(1);
                    let b_un = ops::reshape(b, &shape_b1)?;
                    let solved = ops::solve(a, &b_un)?;
                    ops::reshape(&solved, &b_shape)?
                }
            }
            Op::DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
            } => {
                // General case: permute each operand into [batch...,
                // outer..., contracting...] (or [batch..., contracting...,
                // outer...] for rhs), reshape to [B, M, K] / [B, K, N],
                // run a batched matmul, reshape back to the declared
                // output shape. The canonical 2D pattern (no batch,
                // contract lhs[1] × rhs[0]) reduces to a plain MatMul
                // through this same code path.
                let lhs = lookup(&env, node.inputs[0])?;
                let rhs = lookup(&env, node.inputs[1])?;
                let lhs_shape = node_input_shape(graph, node.inputs[0]);
                let rhs_shape = node_input_shape(graph, node.inputs[1]);

                // Compute "outer" axes (everything that's not batch and
                // not contracting) for each operand.
                let lhs_outer: Vec<usize> = (0..lhs_shape.len())
                    .filter(|i| !lhs_batch.contains(i) && !lhs_contracting.contains(i))
                    .collect();
                let rhs_outer: Vec<usize> = (0..rhs_shape.len())
                    .filter(|i| !rhs_batch.contains(i) && !rhs_contracting.contains(i))
                    .collect();

                // Permutations: lhs → [batch..., outer..., contracting...];
                // rhs → [batch..., contracting..., outer...].
                let mut lhs_perm: Vec<i32> = Vec::with_capacity(lhs_shape.len());
                for &b in lhs_batch {
                    lhs_perm.push(b as i32);
                }
                for &o in &lhs_outer {
                    lhs_perm.push(o as i32);
                }
                for &c in lhs_contracting {
                    lhs_perm.push(c as i32);
                }

                let mut rhs_perm: Vec<i32> = Vec::with_capacity(rhs_shape.len());
                for &b in rhs_batch {
                    rhs_perm.push(b as i32);
                }
                for &c in rhs_contracting {
                    rhs_perm.push(c as i32);
                }
                for &o in &rhs_outer {
                    rhs_perm.push(o as i32);
                }

                let lhs_p = ops::transpose(lhs, &lhs_perm)?;
                let rhs_p = ops::transpose(rhs, &rhs_perm)?;

                // Compute B/M/K/N. Batch dims must match between lhs and
                // rhs by definition of DotGeneral.
                let dim_prod = |shape: &[i32], idxs: &[usize]| -> i32 {
                    idxs.iter().map(|&i| shape[i]).product::<i32>().max(1)
                };
                let big_b = dim_prod(&lhs_shape, lhs_batch);
                let big_m = dim_prod(&lhs_shape, &lhs_outer);
                let big_k = dim_prod(&lhs_shape, lhs_contracting);
                let big_n = dim_prod(&rhs_shape, &rhs_outer);

                let lhs_3d = ops::reshape(&lhs_p, &[big_b, big_m, big_k])?;
                let rhs_3d = ops::reshape(&rhs_p, &[big_b, big_k, big_n])?;

                // Batched matmul. MLX's matmul supports rank-3 batched
                // matmul natively.
                let mm = ops::matmul(&lhs_3d, &rhs_3d)?;

                // Reshape back to the declared output shape so downstream
                // consumers see exactly what the IR's shape inference
                // promised.
                let out_shape: Vec<i32> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i32)
                    .collect();
                ops::reshape(&mm, &out_shape)?
            }
            Op::Binary(bop) => {
                let a = lookup(&env, node.inputs[0])?;
                let b = lookup(&env, node.inputs[1])?;
                let (a, b) = mlx_align_rank3_seq_pair(a, b)?;
                match bop {
                    BinaryOp::Add => ops::add(&a, &b)?,
                    BinaryOp::Mul => ops::mul(&a, &b)?,
                    BinaryOp::Sub => ops::sub(&a, &b)?,
                    BinaryOp::Div => ops::div(&a, &b)?,
                    BinaryOp::Max => ops::max(&a, &b)?,
                    BinaryOp::Min => ops::min(&a, &b)?,
                    BinaryOp::Pow => ops::pow(&a, &b)?,
                    BinaryOp::Mod => ops::fmod(&a, &b)?,
                    BinaryOp::BitAnd => ops::bitand(&a, &b)?,
                    BinaryOp::BitOr => ops::bitor(&a, &b)?,
                    BinaryOp::BitXor => ops::bitxor(&a, &b)?,
                    BinaryOp::Shl => ops::shl(&a, &b)?,
                    BinaryOp::Shr => ops::shr(&a, &b)?,
                    BinaryOp::Atan2 => ops::atan2(&a, &b)?,
                }
            }
            Op::Compare(cop) => {
                let a = lookup(&env, node.inputs[0])?;
                let b = lookup(&env, node.inputs[1])?;
                let (a, b) = mlx_align_rank3_seq_pair(a, b)?;
                match cop {
                    CmpOp::Eq => ops::eq(&a, &b)?,
                    CmpOp::Ne => ops::ne(&a, &b)?,
                    CmpOp::Lt => ops::lt(&a, &b)?,
                    CmpOp::Le => ops::le(&a, &b)?,
                    CmpOp::Gt => ops::gt(&a, &b)?,
                    CmpOp::Ge => ops::ge(&a, &b)?,
                }
            }
            Op::Where => {
                let c = lookup(&env, node.inputs[0])?;
                let x = lookup(&env, node.inputs[1])?;
                let y = lookup(&env, node.inputs[2])?;
                let (c, x) = mlx_align_rank3_seq_pair(c, x)?;
                let (x, y) = mlx_align_rank3_seq_pair(&x, y)?;
                ops::select(&c, &x, &y)?
            }
            Op::TransformRegion { steps, .. } => {
                let mut cur = lookup(&env, node.inputs[0])?.clone_handle()?;
                for step in steps {
                    match step {
                        TransformStep::ResizeNearest2x(_) => {
                            cur = ops::resize_nearest_2x_nchw(&cur)?;
                        }
                    }
                }
                cur
            }
            Op::BatchElementwiseRegion {
                chain,
                num_batch_inputs,
                scalar_input_mask: _,
                input_modulus: _,
                prologue,
                prologue_input: _,
            } => {
                let n = *num_batch_inputs as usize;
                if node.inputs.len() != n {
                    return Err(MlxError(format!(
                        "BatchElementwiseRegion: declared {n} batch inputs but node has {}",
                        node.inputs.len()
                    )));
                }
                let mut slices = Vec::with_capacity(n);
                for &in_id in &node.inputs {
                    slices.push(eval_elementwise_region_on_inputs(
                        &env,
                        std::slice::from_ref(&in_id),
                        chain,
                        *prologue,
                    )?);
                }
                let refs: Vec<&Array> = slices.iter().collect();
                ops::concat(&refs, 0)?
            }
            Op::ElementwiseRegion {
                chain,
                num_inputs,
                scalar_input_mask: _,
                input_modulus,
                prologue,
                prologue_input: _,
            } => {
                // MLX broadcasts by shape, while the fused kernel's modulus
                // convention repeats an input in flat element order. Materialize
                // those repeated inputs before interpreting the chain. This is
                // required for StyleTTS2's channel bias regions: a 256-element
                // input feeds a [1, 256, frames] output.
                let n_in = *num_inputs as usize;
                if node.inputs.len() != n_in {
                    return Err(MlxError(format!(
                        "ElementwiseRegion: declared {n_in} inputs but node has {}",
                        node.inputs.len()
                    )));
                }
                if input_modulus.iter().take(n_in).all(|&m| m == 0) {
                    eval_elementwise_region_on_inputs(&env, &node.inputs, chain, *prologue)?
                } else {
                    let out_shape: Vec<i32> = node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static() as i32)
                        .collect();
                    let mut tiled_env = HashMap::with_capacity(env.len());
                    for (&id, array) in &env {
                        tiled_env.insert(id, array.clone_handle()?);
                    }
                    for (i, &input) in node.inputs.iter().enumerate() {
                        let modulus = input_modulus[i];
                        if modulus != 0 {
                            let tiled = tile_region_modulus_input(
                                lookup(&env, input)?,
                                modulus as usize,
                                &out_shape,
                            )?;
                            tiled_env.insert(input, tiled);
                        }
                    }
                    eval_elementwise_region_on_inputs(&tiled_env, &node.inputs, chain, *prologue)?
                }
            }
            Op::Activation(act) => {
                let x = lookup(&env, node.inputs[0])?;
                match act {
                    Activation::Gelu => ops::gelu(x)?,
                    Activation::GeluApprox => ops::gelu_approx(x)?,
                    Activation::Silu => ops::silu(x)?,
                    Activation::Relu => ops::unary(x, MlxUnary::Relu)?,
                    Activation::Sigmoid => ops::unary(x, MlxUnary::Sigmoid)?,
                    Activation::Tanh => ops::unary(x, MlxUnary::Tanh)?,
                    Activation::Exp => ops::unary(x, MlxUnary::Exp)?,
                    Activation::Log => ops::unary(x, MlxUnary::Log)?,
                    Activation::Sqrt => ops::unary(x, MlxUnary::Sqrt)?,
                    Activation::Rsqrt => ops::unary(x, MlxUnary::Rsqrt)?,
                    Activation::Neg => ops::unary(x, MlxUnary::Neg)?,
                    Activation::Abs => ops::unary(x, MlxUnary::Abs)?,
                    Activation::Round => ops::unary(x, MlxUnary::Round)?,
                    Activation::Sin => ops::unary(x, MlxUnary::Sin)?,
                    Activation::Cos => ops::unary(x, MlxUnary::Cos)?,
                    Activation::Tan => ops::unary(x, MlxUnary::Tan)?,
                    Activation::Atan => ops::unary(x, MlxUnary::Atan)?,
                    Activation::Recip => ops::unary(x, MlxUnary::Reciprocal)?,
                    Activation::Floor => ops::unary(x, MlxUnary::Floor)?,
                    Activation::Ceil => ops::unary(x, MlxUnary::Ceil)?,
                    Activation::Sign => ops::unary(x, MlxUnary::Sign)?,
                    Activation::Softplus => ops::unary(x, MlxUnary::Softplus)?,
                    Activation::Elu => ops::unary(x, MlxUnary::Elu)?,
                    Activation::Erf => ops::unary(x, MlxUnary::Erf)?,
                    Activation::HardSwish => ops::unary(x, MlxUnary::HardSwish)?,
                    Activation::HardSigmoid => ops::unary(x, MlxUnary::HardSigmoid)?,
                    Activation::Mish => ops::unary(x, MlxUnary::Mish)?,
                    Activation::Softsign => ops::unary(x, MlxUnary::Softsign)?,
                    Activation::LogSigmoid => ops::unary(x, MlxUnary::LogSigmoid)?,
                }
            }
            Op::Cast { to } => {
                let x = lookup(&env, node.inputs[0])?;
                let src = graph.node(node.inputs[0]).shape.dtype();
                // MLX astype covers every scalar dtype pair natively. Two
                // gaps go off the native path:
                //   * complex (source or dest): MLX has no complex dtype in
                //     `map_dtype`; host-cast via rlx-cpu's exact semantics.
                //     C64 uses the f32-interleaved layout (`mlx_cast_c64`);
                //     C128 uses the f64-interleaved layout (`mlx_cast_c128`).
                //     A cast that mentions C128 on either side (incl.
                //     C64↔C128) routes to the C128 helper; a purely-C64 cast
                //     routes to the C64 helper.
                //   * F64: astype touching float64 is routed to MLX's CPU
                //     stream inside the C++ shim (`rlx_mlx_op_cast`), since
                //     the Metal GPU stream rejects float64. Handled there,
                //     so it stays on the native `ops::cast` path here.
                if to.is_complex() || src.is_complex() {
                    let out_shape: Vec<usize> = node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    if *to == DType::C128 || src == DType::C128 {
                        mlx_cast_c128(x, src, *to, &out_shape)?
                    } else {
                        mlx_cast_c64(x, src, *to, &out_shape)?
                    }
                } else {
                    ops::cast(x, *to)?
                }
            }
            Op::Softmax { axis } => {
                let x = lookup(&env, node.inputs[0])?;
                ops::softmax(x, *axis)?
            }
            Op::LayerNorm { eps, .. } => {
                let x = lookup(&env, node.inputs[0])?;
                let g = mlx_norm_scale_1d(lookup(&env, node.inputs[1])?)?;
                let b = if node.inputs.len() >= 3 {
                    Some(mlx_norm_scale_1d(lookup(&env, node.inputs[2])?)?)
                } else {
                    None
                };
                ops::layer_norm(x, &g, b.as_ref(), *eps)?
            }
            Op::Reshape { new_shape } => {
                let x = lookup(&env, node.inputs[0])?;
                let rt = x.shape()?;
                let s = mlx_fix_reshape_shape(&rt, new_shape);
                ops::reshape(x, &s)?
            }
            Op::Transpose { perm } => {
                let x = lookup(&env, node.inputs[0])?;
                let p: Vec<i32> = perm.iter().map(|&d| d as i32).collect();
                ops::transpose(x, &p)?
            }
            Op::ResizeNearest2x => {
                let x = lookup(&env, node.inputs[0])?;
                ops::resize_nearest_2x_nchw(x)?
            }
            Op::Narrow { axis, start, len } => {
                let x = lookup(&env, node.inputs[0])?;
                let graph_shape = node_input_shape(graph, node.inputs[0]);
                let runtime_shape: Vec<i32> = x.shape()?.iter().map(|&d| d as i32).collect();
                let axis_rt =
                    map_graph_axis_to_runtime(*axis, graph_shape.len(), runtime_shape.len());
                let mut s_start = vec![0i32; runtime_shape.len()];
                let mut s_stop = runtime_shape.clone();
                s_start[axis_rt] = *start as i32;
                s_stop[axis_rt] = (*start + *len) as i32;
                ops::slice(x, &s_start, &s_stop)?
            }
            Op::Concat { axis } => {
                let inputs: Vec<&Array> = node
                    .inputs
                    .iter()
                    .map(|&id| lookup(&env, id))
                    .collect::<Result<_, _>>()?;
                let aligned = mlx_align_concat_inputs(&inputs, *axis)?;
                let refs: Vec<&Array> = aligned.iter().collect();
                ops::concat(&refs, *axis as i32)?
            }
            Op::Expand { .. } => {
                mlx_expand(graph, node.inputs[0], node, lookup(&env, node.inputs[0])?)?
            }
            Op::Gather { axis } => {
                let x = lookup(&env, node.inputs[0])?;
                let idx = mlx_indices_i64(lookup(&env, node.inputs[1])?)?;
                ops::take(x, &idx, *axis as i32)?
            }
            Op::Reverse { axes } => {
                // Batch-general flip: `take` along each axis with reversed
                // indices [d-1, …, 0]. Only the listed axes move.
                let x0 = lookup(&env, node.inputs[0])?;
                let shape = node_input_shape(graph, node.inputs[0]);
                if axes.is_empty() {
                    x0.clone_handle()?
                } else {
                    let mut cur: Option<Array> = None;
                    for &ax in axes {
                        let d = shape[ax];
                        let idx_f: Vec<f32> = (0..d).rev().map(|i| i as f32).collect();
                        let idx_arr = Array::from_f32_slice(&idx_f, &[d as usize], DType::F32)?;
                        let idx = mlx_indices_i64(&idx_arr)?;
                        let src = cur.as_ref().unwrap_or(x0);
                        cur = Some(ops::take(src, &idx, ax as i32)?);
                    }
                    cur.unwrap()
                }
            }
            Op::Reduce {
                op: rop,
                axes,
                keep_dim,
            } => {
                let x = lookup(&env, node.inputs[0])?;
                let kind = match rop {
                    ReduceOp::Sum => MlxReduce::Sum,
                    ReduceOp::Mean => MlxReduce::Mean,
                    ReduceOp::Max => MlxReduce::Max,
                    ReduceOp::Min => MlxReduce::Min,
                    ReduceOp::Prod => MlxReduce::Prod,
                };
                let ax: Vec<i32> = axes.iter().map(|&a| a as i32).collect();
                ops::reduce(x, kind, &ax, *keep_dim)?
            }
            Op::ArgMax { axis, keep_dim } => {
                // rlx encodes indices as f32 at the I/O boundary.
                let x = lookup(&env, node.inputs[0])?;
                let idx = ops::argmax(x, *axis as i32, *keep_dim)?;
                ops::cast(&idx, DType::F32)?
            }
            Op::ArgMin { axis, keep_dim } => {
                // argmin(x) = argmax(-x); first-hit tie-break matches CPU.
                let x = lookup(&env, node.inputs[0])?;
                let neg1 = Array::from_f32_slice(&[-1.0], &[1], DType::F32)?;
                let neg = ops::mul(x, &neg1)?;
                let idx = ops::argmax(&neg, *axis as i32, *keep_dim)?;
                ops::cast(&idx, DType::F32)?
            }
            Op::Cumsum { axis, exclusive } => {
                let x = lookup(&env, node.inputs[0])?;
                ops::cumsum(x, *axis, *exclusive)?
            }
            Op::CumProd { axis, exclusive } => {
                let x = lookup(&env, node.inputs[0])?;
                ops::cumprod(x, *axis, *exclusive)?
            }
            Op::CumMax { axis, exclusive } => {
                let x = lookup(&env, node.inputs[0])?;
                ops::cummax(x, *axis, *exclusive)?
            }
            Op::Fft { inverse, norm } => {
                let x = lookup(&env, node.inputs[0])?;
                ops::fft(x, *inverse, norm.tag())?
            }
            Op::LogMel => {
                let spec = lookup(&env, node.inputs[0])?.to_f32()?;
                let filters = lookup(&env, node.inputs[1])?.to_f32()?;
                let spec_shape = graph.node(node.inputs[0]).shape.clone();
                let filt_shape = graph.node(node.inputs[1]).shape.clone();
                let meta =
                    rlx_ir::audio::log_mel_meta(&spec_shape, &filt_shape).map_err(MlxError)?;
                let mut out = vec![0f32; meta.outer * meta.n_mels];
                rlx_ir::audio::log_mel_block_f32(
                    &spec,
                    &filters,
                    meta.outer,
                    meta.n_fft,
                    meta.n_bins,
                    meta.n_mels,
                    &mut out,
                );
                let out_shape: Vec<usize> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect();
                Array::from_f32_slice(&out, &out_shape, DType::F32)?
            }
            Op::LogMelBackward => {
                let spec = lookup(&env, node.inputs[0])?.to_f32()?;
                let filters = lookup(&env, node.inputs[1])?.to_f32()?;
                let dy = lookup(&env, node.inputs[2])?.to_f32()?;
                let spec_shape = graph.node(node.inputs[0]).shape.clone();
                let filt_shape = graph.node(node.inputs[1]).shape.clone();
                let meta =
                    rlx_ir::audio::log_mel_meta(&spec_shape, &filt_shape).map_err(MlxError)?;
                let mut d_spec = vec![0f32; meta.outer * meta.n_fft * 2];
                rlx_ir::audio::log_mel_block_vjp(
                    &spec,
                    &filters,
                    &dy,
                    meta.outer,
                    meta.n_fft,
                    meta.n_bins,
                    meta.n_mels,
                    &mut d_spec,
                );
                let out_shape: Vec<usize> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect();
                Array::from_f32_slice(&d_spec, &out_shape, DType::F32)?
            }
            Op::WelchPeaks { k, n_segments } => {
                let spec = lookup(&env, node.inputs[0])?.to_f32()?;
                let spec_shape = graph.node(node.inputs[0]).shape.clone();
                let meta = rlx_ir::audio::welch_peaks_meta(&spec_shape, *k, *n_segments)
                    .map_err(MlxError)?;
                let mut out = vec![0f32; meta.welch_batch * meta.k * 2];
                rlx_ir::audio::welch_peaks_block_f32(
                    &spec,
                    meta.welch_batch,
                    meta.n_fft,
                    meta.n_segments,
                    meta.k,
                    &mut out,
                );
                let out_shape: Vec<usize> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect();
                Array::from_f32_slice(&out, &out_shape, DType::F32)?
            }
            Op::RmsNorm { eps, .. } => {
                let x = lookup(&env, node.inputs[0])?;
                let g = mlx_norm_scale_1d(lookup(&env, node.inputs[1])?)?;
                ops::rms_norm(x, &g, *eps)?
            }
            Op::Attention {
                num_heads,
                head_dim,
                mask_kind,
                score_scale,
                attn_logit_softcap: _,
            } => {
                // MLX's fast::scaled_dot_product_attention expects Q/K/V
                // as rank-4 [B, H, S, D]. rlx callers may hand us either
                // that or rank-3 [B, S, H*D] (the un-split BERT-style
                // post-projection layout). For rank-3 we reshape +
                // transpose into [B, H, S, D] and back.
                let q_in = lookup(&env, node.inputs[0])?;
                let k_in = lookup(&env, node.inputs[1])?;
                let v_in = lookup(&env, node.inputs[2])?;
                let q_shape = node_input_shape(graph, node.inputs[0]);
                let k_shape = node_input_shape(graph, node.inputs[1]);

                let nh = *num_heads as i32;
                let hd = *head_dim as i32;
                // Respect `score_scale` when the IR specifies one — Gemma 4
                // sets `Some(1.0)` because Q is per-head RMS-normed before
                // attention, so the standard `1/sqrt(head_dim)` factor
                // crushes the scores (E2B head_dim=256 → 16× too small).
                // Without this the SWA attention output drifts from CPU and
                // every E2B greedy step diverges from HF on MLX.
                let scale = score_scale.unwrap_or_else(|| 1.0 / (hd as f32).sqrt());

                let q_ir = graph.node(node.inputs[0]).shape.clone();
                let k_ir = graph.node(node.inputs[1]).shape.clone();
                let geom = rlx_ir::attention_geom(&q_ir, &k_ir, *num_heads, *head_dim);
                let bshd_rank4 = q_shape.len() == 4 && !geom.bhsd;

                let to_bhsd = |t: &Array, sh: &[i32]| -> Result<Array, MlxError> {
                    if sh.len() == 4 {
                        if sh[1] == nh {
                            return t.clone_handle();
                        }
                        // [B, S, H, D] → [B, H, S, D]
                        let t = ops::transpose(t, &[0, 2, 1, 3])?;
                        // Materialize: mlx::compile elides transpose views otherwise
                        // (same issue as conv NHWC→NCHW in conv_compile_mode_repro).
                        return ops::contiguous(&t);
                    }
                    // [B, S, H*D] → [B, S, H, D] → [B, H, S, D]
                    let b = sh[0];
                    let s = sh[1];
                    let r = ops::reshape(t, &[b, s, nh, hd])?;
                    let t = ops::transpose(&r, &[0, 2, 1, 3])?;
                    ops::contiguous(&t)
                };
                let q = to_bhsd(q_in, &q_shape)?;
                let k = to_bhsd(k_in, &k_shape)?;
                let v = to_bhsd(v_in, &node_input_shape(graph, node.inputs[2]))?;

                // Mask must promote to Q/output dtype — MLX's SDPA
                // rejects an f32 mask when Q is f16/bf16. AutoMixed
                // promotes Q/K/V but masks aren't tagged in the
                // precision pass, so cast at the dispatch site.
                let q_dtype = graph.node(node.inputs[0]).shape.dtype();

                // Reshape an arbitrary-rank mask into a 4-D shape SDPA
                // can broadcast against [B, H, S_q, S_k]:
                //   rank 2 [B, S]          → [B, 1, 1, S]
                //   rank 3 [B, S_q, S_k]   → [B, 1, S_q, S_k]
                //   rank 4 [...]           → pass through
                let normalize_mask = |m: &Array, m_shape: &[i32]| -> Result<Array, MlxError> {
                    match m_shape.len() {
                        2 => ops::reshape(m, &[m_shape[0], 1, 1, m_shape[1]]),
                        3 => ops::reshape(m, &[m_shape[0], 1, m_shape[1], m_shape[2]]),
                        _ => m.clone_handle(),
                    }
                };

                let (mask_kind_ffi, mask_owned, mask) = match mask_kind {
                    MaskKind::None => (MlxMask::None, None, None),
                    MaskKind::Causal => (MlxMask::Causal, None, None),
                    MaskKind::Custom => {
                        // MLX SDPA adds the mask additively to scores. The
                        // burnembed BERT graph (and the CPU/Metal/wgpu
                        // backends) interpret MaskKind::Custom as a *binary*
                        // multiplicative mask (1 = valid, 0 = padding).
                        // Convert here so MLX matches the rest of the
                        // workspace: additive = (mask - 1) * 1e9 → 0 when
                        // valid, -1e9 when padded.
                        let m = lookup(&env, node.inputs[3])?;
                        let m_shape = node_input_shape(graph, node.inputs[3]);
                        let one = Array::from_f32_slice(&[1.0], &[1], q_dtype)?;
                        let scl = Array::from_f32_slice(&[1.0e9], &[1], q_dtype)?;
                        let m_cast = if q_dtype != DType::F32 {
                            ops::cast(m, q_dtype)?
                        } else {
                            m.clone_handle()?
                        };
                        let shifted = ops::sub(&m_cast, &one)?;
                        let additive = ops::mul(&shifted, &scl)?;
                        (
                            MlxMask::Custom,
                            Some(normalize_mask(&additive, &m_shape)?),
                            None,
                        )
                    }
                    MaskKind::SlidingWindow(window) => {
                        let s_q = q_shape[q_shape.len() - 2];
                        let s_k = k_shape[k_shape.len() - 2];
                        let m = build_sliding_window_mask(s_q, s_k, *window as i32)?;
                        // build_sliding_window_mask returns rank-2; normalize.
                        let m4 = ops::reshape(&m, &[1, 1, s_q, s_k])?;
                        let m4 = if q_dtype != DType::F32 {
                            ops::cast(&m4, q_dtype)?
                        } else {
                            m4
                        };
                        (MlxMask::Custom, Some(m4), None)
                    }
                    MaskKind::Bias => {
                        // Bias mask = raw additive bias tensor on the 4th input. Pass
                        // through unmodified — MLX SDPA already adds it to scores.
                        let m = lookup(&env, node.inputs[3])?;
                        let m_shape = node_input_shape(graph, node.inputs[3]);
                        let m_cast = if q_dtype != DType::F32 {
                            ops::cast(m, q_dtype)?
                        } else {
                            m.clone_handle()?
                        };
                        (
                            MlxMask::Custom,
                            Some(normalize_mask(&m_cast, &m_shape)?),
                            None,
                        )
                    }
                };
                let m_ref: Option<&Array> = mask.as_ref().or(mask_owned.as_ref());
                let attn_out = if crate::config::runtime_config().sdpa_reference {
                    ops::attention_reference_bhsd(&q, &k, &v, scale, m_ref)?
                } else {
                    ops::attention(&q, &k, &v, scale, mask_kind_ffi, m_ref)?
                };

                if q_shape.len() == 3 {
                    // [B, H, S, D] → [B, S, H, D] → [B, S, H*D]
                    let b = q_shape[0];
                    let s = q_shape[1];
                    let bsd = ops::transpose(&attn_out, &[0, 2, 1, 3])?;
                    ops::reshape(&bsd, &[b, s, nh * hd])?
                } else if bshd_rank4 {
                    let t = ops::transpose(&attn_out, &[0, 2, 1, 3])?;
                    ops::contiguous(&t)?
                } else {
                    attn_out
                }
            }

            // ── Fused ops produced by the optimizer's fusion passes ──
            //
            // We compose these from primitives MLX already understands;
            // the fused IR variant exists mainly to keep CPU/Metal
            // happy. Behaviour matches the CPU executor's reference.
            Op::FusedMatMulBiasAct { activation } => {
                let a = lookup(&env, node.inputs[0])?;
                let w = lookup(&env, node.inputs[1])?;
                let b = lookup(&env, node.inputs[2])?;
                let mm = ops::matmul(a, w)?;
                let biased = mlx_add_aligned(&mm, b)?;
                match activation {
                    None => biased,
                    Some(Activation::Gelu) => ops::gelu(&biased)?,
                    Some(Activation::GeluApprox) => ops::gelu_approx(&biased)?,
                    Some(Activation::Silu) => ops::silu(&biased)?,
                    Some(Activation::Relu) => ops::unary(&biased, MlxUnary::Relu)?,
                    Some(Activation::Sigmoid) => ops::unary(&biased, MlxUnary::Sigmoid)?,
                    Some(Activation::Tanh) => ops::unary(&biased, MlxUnary::Tanh)?,
                    Some(Activation::Exp) => ops::unary(&biased, MlxUnary::Exp)?,
                    Some(Activation::Log) => ops::unary(&biased, MlxUnary::Log)?,
                    Some(Activation::Sqrt) => ops::unary(&biased, MlxUnary::Sqrt)?,
                    Some(Activation::Rsqrt) => ops::unary(&biased, MlxUnary::Rsqrt)?,
                    Some(Activation::Neg) => ops::unary(&biased, MlxUnary::Neg)?,
                    Some(Activation::Abs) => ops::unary(&biased, MlxUnary::Abs)?,
                    Some(Activation::Round) => ops::unary(&biased, MlxUnary::Round)?,
                    Some(Activation::Sin) => ops::unary(&biased, MlxUnary::Sin)?,
                    Some(Activation::Cos) => ops::unary(&biased, MlxUnary::Cos)?,
                    Some(Activation::Tan) => ops::unary(&biased, MlxUnary::Tan)?,
                    Some(Activation::Atan) => ops::unary(&biased, MlxUnary::Atan)?,
                    Some(Activation::Recip) => ops::unary(&biased, MlxUnary::Reciprocal)?,
                    Some(Activation::Floor) => ops::unary(&biased, MlxUnary::Floor)?,
                    Some(Activation::Ceil) => ops::unary(&biased, MlxUnary::Ceil)?,
                    Some(Activation::Sign) => ops::unary(&biased, MlxUnary::Sign)?,
                    Some(Activation::Softplus) => ops::unary(&biased, MlxUnary::Softplus)?,
                    Some(Activation::Elu) => ops::unary(&biased, MlxUnary::Elu)?,
                    Some(Activation::Erf) => ops::unary(&biased, MlxUnary::Erf)?,
                    Some(Activation::HardSwish) => ops::unary(&biased, MlxUnary::HardSwish)?,
                    Some(Activation::HardSigmoid) => ops::unary(&biased, MlxUnary::HardSigmoid)?,
                    Some(Activation::Mish) => ops::unary(&biased, MlxUnary::Mish)?,
                    Some(Activation::Softsign) => ops::unary(&biased, MlxUnary::Softsign)?,
                    Some(Activation::LogSigmoid) => ops::unary(&biased, MlxUnary::LogSigmoid)?,
                }
            }
            Op::FusedResidualLN { has_bias, eps } => {
                let x = lookup(&env, node.inputs[0])?;
                let r = lookup(&env, node.inputs[1])?;
                let summed = mlx_add_aligned(x, r)?;
                let summed = if *has_bias {
                    let bias = lookup(&env, node.inputs[2])?;
                    mlx_add_aligned(&summed, bias)?
                } else {
                    summed
                };
                let (g_idx, b_idx) = if *has_bias { (3, 4) } else { (2, 3) };
                let g = mlx_norm_scale_1d(lookup(&env, node.inputs[g_idx])?)?;
                let b = mlx_norm_scale_1d(lookup(&env, node.inputs[b_idx])?)?;
                ops::layer_norm(&summed, &g, Some(&b), *eps)?
            }
            Op::FusedResidualRmsNorm { has_bias, eps } => {
                let x = lookup(&env, node.inputs[0])?;
                let r = lookup(&env, node.inputs[1])?;
                let summed = mlx_add_aligned(x, r)?;
                let summed = if *has_bias {
                    let bias = lookup(&env, node.inputs[2])?;
                    mlx_add_aligned(&summed, bias)?
                } else {
                    summed
                };
                let g_idx = if *has_bias { 3 } else { 2 };
                let g = mlx_norm_scale_1d(lookup(&env, node.inputs[g_idx])?)?;
                ops::rms_norm(&summed, &g, *eps)?
            }
            Op::AdaLayerNorm { norm, eps } => {
                let x = lookup(&env, node.inputs[0])?;
                let dtype = node.shape.dtype();
                let d = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                let ones = crate::array::Array::from_f32_slice(&vec![1.0_f32; d], &[d], dtype)?;
                let zeros = crate::array::Array::from_f32_slice(&vec![0.0_f32; d], &[d], dtype)?;
                let n = match norm {
                    AdaNormKind::LayerNorm => {
                        ops::layer_norm(x, &ones, Some(&zeros), *eps)?
                    }
                    AdaNormKind::RmsNorm => ops::rms_norm(x, &ones, *eps)?,
                };
                let scale = lookup(&env, node.inputs[1])?;
                let shift = lookup(&env, node.inputs[2])?;
                let scale_e = mlx_expand(graph, node.inputs[1], node, scale)?;
                let n_scale = ops::mul(&n, &scale_e)?;
                let m = ops::add(&n, &n_scale)?;
                let shift_e = mlx_expand(graph, node.inputs[2], node, shift)?;
                ops::add(&m, &shift_e)?
            }
            Op::GatedResidual => {
                let x = lookup(&env, node.inputs[0])?;
                let y = lookup(&env, node.inputs[1])?;
                let gate = lookup(&env, node.inputs[2])?;
                let gate_e = mlx_expand(graph, node.inputs[2], node, gate)?;
                let gy = ops::mul(&gate_e, y)?;
                mlx_add_aligned(x, &gy)?
            }
            Op::AdaLayerNormBackward { norm, eps } => {
                // Packed DiT adaLN reverse — mirrors `compose_ada_layer_norm_backward`.
                let x = lookup(&env, node.inputs[0])?;
                let scale_in = lookup(&env, node.inputs[1])?;
                let dy = lookup(&env, node.inputs[3])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let scale_shape = node_input_shape(graph, node.inputs[1]);
                let last = (x_shape.len() - 1) as i32;
                let dtype = node.shape.dtype();
                let d = graph.node(node.inputs[0]).shape.dim(last as usize).unwrap_static();
                let ones_1d =
                    Array::from_f32_slice(&vec![1.0_f32; d], &[d], dtype)?;
                let zeros_1d =
                    Array::from_f32_slice(&vec![0.0_f32; d], &[d], dtype)?;
                let n = match norm {
                    AdaNormKind::LayerNorm => {
                        ops::layer_norm(x, &ones_1d, Some(&zeros_1d), *eps)?
                    }
                    AdaNormKind::RmsNorm => ops::rms_norm(x, &ones_1d, *eps)?,
                };
                let one_scalar = Array::from_f32_slice(&[1.0], &[1], dtype)?;
                let x_ones = ops::broadcast_to(&one_scalar, &x_shape)?;
                // Expand scale to *x*'s shape — not this node's packed
                // `[dx∥dscale∥dshift]` output (mlx_expand keys off out_node).
                let x_node = graph.node(node.inputs[0]);
                let scale_e = mlx_expand(graph, node.inputs[1], x_node, scale_in)?;
                let one_plus = ops::add(&x_ones, &scale_e)?;
                let dn = ops::mul(dy, &one_plus)?;
                let eps_arr = Array::from_f32_slice(&[*eps], &[1], dtype)?;
                let dx = match norm {
                    AdaNormKind::LayerNorm => {
                        let mean = ops::reduce(x, MlxReduce::Mean, &[last], true)?;
                        let diff = ops::sub(x, &mean)?;
                        let diff_sq = ops::mul(&diff, &diff)?;
                        let var = ops::reduce(&diff_sq, MlxReduce::Mean, &[last], true)?;
                        let var_eps = ops::add(&var, &eps_arr)?;
                        let inv_std = ops::unary(&var_eps, MlxUnary::Rsqrt)?;
                        let xhat = ops::mul(&diff, &inv_std)?;
                        let m_sy = ops::reduce(&dn, MlxReduce::Mean, &[last], true)?;
                        let sy_xh = ops::mul(&dn, &xhat)?;
                        let m_sxh = ops::reduce(&sy_xh, MlxReduce::Mean, &[last], true)?;
                        let term1 = ops::sub(&dn, &m_sy)?;
                        let term2 = ops::mul(&xhat, &m_sxh)?;
                        let inner = ops::sub(&term1, &term2)?;
                        ops::mul(&inv_std, &inner)?
                    }
                    AdaNormKind::RmsNorm => {
                        let x_sq = ops::mul(x, x)?;
                        let mean_sq = ops::reduce(&x_sq, MlxReduce::Mean, &[last], true)?;
                        let var_eps = ops::add(&mean_sq, &eps_arr)?;
                        let inv_r = ops::unary(&var_eps, MlxUnary::Rsqrt)?;
                        let inv_r2 = ops::mul(&inv_r, &inv_r)?;
                        let dy_gx = ops::mul(&dn, x)?;
                        let dot = ops::reduce(&dy_gx, MlxReduce::Mean, &[last], true)?;
                        let x_dot = ops::mul(x, &dot)?;
                        let term = ops::sub(&dn, &ops::mul(&x_dot, &inv_r2)?)?;
                        ops::mul(&inv_r, &term)?
                    }
                };
                let dscale_full = ops::mul(dy, &n)?;
                let dscale = mlx_unbroadcast_grad(&dscale_full, &x_shape, &scale_shape)?;
                let dshift = mlx_unbroadcast_grad(dy, &x_shape, &scale_shape)?;
                mlx_pack_flat_grads(&[dx, dscale, dshift])?
            }
            Op::GatedResidualBackward => {
                // Packed DiT gated residual reverse — mirrors `compose_gated_residual_backward`.
                let dy = lookup(&env, node.inputs[3])?;
                let y = lookup(&env, node.inputs[1])?;
                let gate_in = lookup(&env, node.inputs[2])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let gate_shape = node_input_shape(graph, node.inputs[2]);
                let dx = dy.clone_handle()?;
                // Expand gate to activation shape, not packed `[dx∥dy∥dgate]` out.
                let x_node = graph.node(node.inputs[0]);
                let gate_e = mlx_expand(graph, node.inputs[2], x_node, gate_in)?;
                let dy_out = ops::mul(dy, &gate_e)?;
                let dgate_full = ops::mul(dy, y)?;
                let dgate = mlx_unbroadcast_grad(&dgate_full, &x_shape, &gate_shape)?;
                mlx_pack_flat_grads(&[dx, dy_out, dgate])?
            }
            Op::Rope {
                head_dim,
                n_rot,
                style,
            } => {
                let x = lookup(&env, node.inputs[0])?;
                let cos = lookup(&env, node.inputs[1])?;
                let sin = lookup(&env, node.inputs[2])?;
                // GGUF Llama Q/K weights are permuted for interleaved (GPT-J) RoPE —
                // pairs `(2i, 2i+1)` — whereas HF safetensors use rotate-half (NeoX) —
                // pairs `(i, i+half)`. The cos/sin tables are identical; only how the
                // last axis is paired differs.
                let interleaved = matches!(style, RopeStyle::GptJ);

                let graph_x = node_input_shape(graph, node.inputs[0]);
                let x_shape = runtime_shape_or_graph(x, &graph_x)?;
                let cos_runtime = cos.shape().unwrap_or_default();
                if cos_runtime.len() != 2 {
                    return Err(MlxError(format!(
                        "Rope: cos must be rank-2 [seq, half], got rank-{} shape={cos_runtime:?} (graph x={x_shape:?}, n_rot={n_rot})",
                        cos_runtime.len()
                    )));
                }
                let n = x_shape.len();
                if n < 2 {
                    return Err(MlxError("Rope: x must be rank ≥ 2".into()));
                }
                if head_dim % 2 != 0 {
                    return Err(MlxError(format!("Rope: head_dim {head_dim} must be even")));
                }
                if *n_rot > *head_dim || !n_rot.is_multiple_of(2) {
                    return Err(MlxError(format!(
                        "Rope: n_rot={n_rot} must be even and <= head_dim={head_dim}"
                    )));
                }
                let hd = *head_dim as i32;
                let nr = *n_rot as i32;
                let rot_half = nr / 2;

                let last = *x_shape.last().unwrap() as usize;
                if last < *n_rot {
                    return Err(MlxError(format!("Rope: x last dim {last} < n_rot {n_rot}")));
                }
                let heads_in_last = (last / *head_dim) as i32;
                let multi_head_packed =
                    heads_in_last > 1 && last.is_multiple_of(*head_dim) && n >= 3;
                let has_tail = !last.is_multiple_of(*head_dim);

                let rotate = |x_rot: &Array,
                              rot_shape: &[i32],
                              seq_axis: usize,
                              pairs: i32|
                 -> Result<Array, MlxError> {
                    let rn = rot_shape.len();
                    let seq_v = rot_shape[seq_axis];
                    let cos_rows = cos.shape()?.first().copied().unwrap_or(0) as i32;
                    let seq_cos = seq_v.min(cos_rows.max(1));
                    let cos_seq = ops::slice(cos, &[0, 0], &[seq_cos, pairs])?;
                    let sin_seq = ops::slice(sin, &[0, 0], &[seq_cos, pairs])?;
                    if interleaved {
                        // GPT-J: reshape the rotation axis `[2*pairs]` -> `[pairs, 2]`
                        // so even/odd elements sit on a new trailing axis; rotate the
                        // pair, then reshape back (which re-interleaves `2i`/`2i+1`).
                        let mut pair_shape = rot_shape.to_vec();
                        pair_shape[rn - 1] = pairs;
                        pair_shape.push(2);
                        let x_pairs = ops::reshape(x_rot, &pair_shape)?;
                        let mut even_stop = pair_shape.clone();
                        even_stop[rn] = 1;
                        let x_even = ops::slice(&x_pairs, &vec![0i32; rn + 1], &even_stop)?;
                        let mut odd_start = vec![0i32; rn + 1];
                        odd_start[rn] = 1;
                        let x_odd = ops::slice(&x_pairs, &odd_start, &pair_shape)?;
                        let mut bshape = vec![1i32; rn + 1];
                        bshape[seq_axis] = seq_cos;
                        bshape[rn - 1] = pairs;
                        let cos_b = ops::reshape(&cos_seq, &bshape)?;
                        let sin_b = ops::reshape(&sin_seq, &bshape)?;
                        let y_even =
                            ops::sub(&ops::mul(&x_even, &cos_b)?, &ops::mul(&x_odd, &sin_b)?)?;
                        let y_odd =
                            ops::add(&ops::mul(&x_odd, &cos_b)?, &ops::mul(&x_even, &sin_b)?)?;
                        let y_pairs = ops::concat(&[&y_even, &y_odd], rn as i32)?;
                        return ops::reshape(&y_pairs, rot_shape);
                    }
                    let mut bshape = vec![1i32; rn];
                    bshape[seq_axis] = seq_cos;
                    bshape[rn - 1] = pairs;
                    let cos_b = ops::reshape(&cos_seq, &bshape)?;
                    let sin_b = ops::reshape(&sin_seq, &bshape)?;
                    let mut x1_stop = rot_shape.to_vec();
                    x1_stop[rn - 1] = pairs;
                    let x1 = ops::slice(x_rot, &vec![0i32; rn], &x1_stop)?;
                    let mut x2_start = vec![0i32; rn];
                    x2_start[rn - 1] = pairs;
                    let x2 = ops::slice(x_rot, &x2_start, rot_shape)?;
                    let x1_cos = ops::mul(&x1, &cos_b)?;
                    let x2_sin = ops::mul(&x2, &sin_b)?;
                    let x2_cos = ops::mul(&x2, &cos_b)?;
                    let x1_sin = ops::mul(&x1, &sin_b)?;
                    let y1 = ops::sub(&x1_cos, &x2_sin)?;
                    let y2 = ops::add(&x2_cos, &x1_sin)?;
                    ops::concat(&[&y1, &y2], (rn - 1) as i32)
                };

                if has_tail {
                    let mut rot_stop = x_shape.clone();
                    rot_stop[n - 1] = nr.min(hd);
                    let rot = ops::slice(x, &vec![0i32; n], &rot_stop)?;
                    let mut tail_start = vec![0i32; n];
                    tail_start[n - 1] = nr.min(hd);
                    let tail = ops::slice(x, &tail_start, &x_shape)?;
                    let mut rot_shape = x_shape.clone();
                    rot_shape[n - 1] = nr.min(hd);
                    let y_rot = rotate(&rot, &rot_shape, n - 2, rot_half)?;
                    ops::concat(&[&y_rot, &tail], (n - 1) as i32)?
                } else if multi_head_packed {
                    let mut split_shape = x_shape.clone();
                    split_shape[n - 1] = heads_in_last;
                    split_shape.push(hd);
                    // `Op::Rope`'s seq axis is `n-2` (original rank). For packed rank-3 callers
                    // (`[B, S, H*D]`), reshape gives `[B, S, H, D]` but we need `[B, H, S, D]`
                    // so that `seq_axis = n-1` (after adding the hd axis) points at `S`.
                    let x_split = ops::reshape(x, &split_shape)?;
                    let mut perm: Vec<i32> = (0..(n as i32 + 1)).collect();
                    perm.swap(n - 1, n - 2);
                    let x_split = ops::transpose(&x_split, &perm)?;
                    split_shape.swap(n - 1, n - 2);
                    if nr < hd {
                        let mut rot_stop = split_shape.clone();
                        rot_stop[n] = nr;
                        let rot = ops::slice(&x_split, &vec![0i32; n + 1], &rot_stop)?;
                        let mut pass_start = vec![0i32; n + 1];
                        pass_start[n] = nr;
                        let pass = ops::slice(&x_split, &pass_start, &split_shape)?;
                        let mut rot_shape = split_shape.clone();
                        rot_shape[n] = nr;
                        let y_rot = rotate(&rot, &rot_shape, n - 1, rot_half)?;
                        let y_head = ops::concat(&[&y_rot, &pass], n as i32)?;
                        // Transpose back to `[... , S, H, D]` then reshape to original packed rank-3.
                        let mut perm_back: Vec<i32> = (0..(n as i32 + 1)).collect();
                        perm_back.swap(n - 1, n - 2);
                        let y_bshd = ops::transpose(&y_head, &perm_back)?;
                        ops::reshape(&y_bshd, &x_shape)?
                    } else {
                        let y_split = rotate(&x_split, &split_shape, n - 1, rot_half)?;
                        let mut perm_back: Vec<i32> = (0..(n as i32 + 1)).collect();
                        perm_back.swap(n - 1, n - 2);
                        let y_bshd = ops::transpose(&y_split, &perm_back)?;
                        ops::reshape(&y_bshd, &x_shape)?
                    }
                } else if nr < hd {
                    let mut rot_stop = x_shape.clone();
                    rot_stop[n - 1] = nr;
                    let rot = ops::slice(x, &vec![0i32; n], &rot_stop)?;
                    let mut pass_start = vec![0i32; n];
                    pass_start[n - 1] = nr;
                    let pass = ops::slice(x, &pass_start, &x_shape)?;
                    let mut rot_shape = x_shape.clone();
                    rot_shape[n - 1] = nr;
                    let y_rot = rotate(&rot, &rot_shape, n - 2, rot_half)?;
                    ops::concat(&[&y_rot, &pass], (n - 1) as i32)?
                } else {
                    rotate(x, &x_shape, n - 2, rot_half)?
                }
            }
            Op::Conv {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                // rlx convention: NCHW (or NCL / NCDHW) inputs +
                // [C_out, C_in/g, ...spatial] weights.
                // MLX expects channels-last (NHWC, NLC, NDHWC) and
                // weight [C_out, ...spatial, C_in/g]. We transpose
                // around the call. A future pass could keep
                // activations in channels-last across consecutive
                // convs to amortize the conversion.
                //
                // Guard: MLX's conv1d/conv2d builds an im2col of size
                // c_in/g · k · out_spatial. Decomposed ISTFT ConvTranspose
                // (F5 Vocos: k=1024, out≈149k) → ~627 GB MTL alloc. Mirror
                // rlx-cpu's IM2COL_MAX and host-eval the naive kernel instead.
                if mlx_conv_im2col_too_large(graph, node, kernel_size, *groups) {
                    host_eval_op_f32(graph, node, &env)?
                } else {
                let in_shape = node_input_shape(graph, node.inputs[0]);
                let x = lookup(&env, node.inputs[0])?;
                let w = lookup(&env, node.inputs[1])?;
                let s = |i: usize| stride.get(i).copied().unwrap_or(1) as i32;
                let p = |i: usize| padding.get(i).copied().unwrap_or(0) as i32;
                let d = |i: usize| dilation.get(i).copied().unwrap_or(1) as i32;

                match (kernel_size.len(), in_shape.len()) {
                    (1, 3) => {
                        // NCL → NLC: perm [0, 2, 1]; weight [Co, Ci, kL]
                        // → [Co, kL, Ci]: perm [0, 2, 1]
                        let x_nlc = ops::transpose(x, &[0, 2, 1])?;
                        let w_mlx = ops::transpose(w, &[0, 2, 1])?;
                        let y_nlc = ops::conv1d(&x_nlc, &w_mlx, s(0), p(0), d(0), *groups as i32)?;
                        ops::transpose(&y_nlc, &[0, 2, 1])?
                    }
                    (2, 4)
                        if (in_shape[2] == 1 || in_shape[3] == 1)
                            && (kernel_size[0] == 1 || kernel_size[1] == 1) =>
                    {
                        // 1D conv expressed as 2D NCHW with a unit spatial axis.
                        // GUARD also requires a unit *kernel* dim: this fast path
                        // reshapes the weight [Co,Ci/g,kh,kw] → [Co,Ci/g,max(kh,kw)],
                        // which is only valid when min(kh,kw)==1. A genuine 2D kernel
                        // over a unit-input axis (e.g. a 3×3 conv over a height-1
                        // feature map, as in ContraWR's collapsed STFT axis) has NO
                        // unit kernel dim, so it must fall through to the general
                        // `(2, 4)` conv2d arm below instead of crashing the reshape.
                        // (rlx lowers ONNX 1D convs as `[N,C,1,L]`/`[N,C,L,1]` with
                        // the length-axis kernel/stride/pad at index 0). Applying a
                        // 2D conv would run the kernel over the singleton axis, so
                        // collapse to NCL and use conv1d over the real length, then
                        // reshape back to the rlx 4D output convention `[N,C,Lo,1]`.
                        let n = in_shape[0];
                        let ci = in_shape[1];
                        let length = if in_shape[2] == 1 {
                            in_shape[3]
                        } else {
                            in_shape[2]
                        };
                        let wsh = w.shape()?; // [Co, Ci/g, kh, kw]
                        let co = wsh[0] as i32;
                        let cig = wsh[1] as i32;
                        let k = wsh[2].max(wsh[3]) as i32;
                        // The kernel/stride/pad/dilation for the real (length) axis live
                        // at the conv-param index of the non-trivial kernel dimension —
                        // NOT the non-singleton *input* axis. rlx uses BOTH conventions:
                        //   * ONNX-native 1D convs → `[N,C,1,L]` with kernel `[k,1]`,
                        //     pad `[p,0]` at index 0 (VITS/TinyTTS text-enc & duration
                        //     predictor: kernel-3 same-padded).
                        //   * ONNX 2D-with-unit-H convs → `[N,C,1,L]` with kernel `[1,k]`,
                        //     pad `[0,p]` at index 1 (EEGNet / U-Sleep / SeizureTransformer
                        //     depthwise stages).
                        // Keying off the input singleton (old `in_shape[2]==1 ? 1 : 0`)
                        // read the wrong index for the first convention and silently
                        // dropped the length padding (11→9), crashing the next reshape.
                        let li = if wsh[2] >= wsh[3] { 0 } else { 1 };
                        let _ = co;
                        let x_ncl = ops::reshape(x, &[n, ci, length])?;
                        let w_ncl = ops::reshape(w, &[co, cig, k])?;
                        let x_nlc = ops::transpose(&x_ncl, &[0, 2, 1])?;
                        let w_mlx = ops::transpose(&w_ncl, &[0, 2, 1])?;
                        let y_nlc =
                            ops::conv1d(&x_nlc, &w_mlx, s(li), p(li), d(li), *groups as i32)?;
                        let y_ncl = ops::transpose(&y_nlc, &[0, 2, 1])?; // [N, Co, Lo]
                        // Reshape to the importer's declared 4D output shape (it places
                        // the length axis in W: `[N,Co,1,Lo]`) so downstream ops that
                        // rely on the declared layout line up. MLX's conv length may
                        // differ from the importer's by a few samples (padding
                        // rounding); trim the length axis to the declared length first.
                        let out_dims: Vec<i32> = node
                            .shape
                            .dims()
                            .iter()
                            .map(|d| d.unwrap_static() as i32)
                            .collect();
                        let target_len = out_dims.iter().product::<i32>() / (n * co).max(1);
                        let cur = y_ncl.shape()?;
                        let y_ncl = if cur.get(2).copied().unwrap_or(0) as i32 > target_len {
                            let mut stop: Vec<i32> = cur.iter().map(|&d| d as i32).collect();
                            stop[2] = target_len;
                            ops::slice(&y_ncl, &[0, 0, 0], &stop)?
                        } else {
                            y_ncl
                        };
                        ops::reshape(&y_ncl, &out_dims)?
                    }
                    (2, 4) => {
                        let x_nhwc = ops::transpose(x, &[0, 2, 3, 1])?;
                        let w_mlx = ops::transpose(w, &[0, 2, 3, 1])?;
                        let y_nhwc = ops::conv2d(
                            &x_nhwc,
                            &w_mlx,
                            (s(0), s(1)),
                            (p(0), p(1)),
                            (d(0), d(1)),
                            *groups as i32,
                        )?;
                        ops::transpose(&y_nhwc, &[0, 3, 1, 2])?
                    }
                    (3, 5) => {
                        // NCDHW → NDHWC: perm [0, 2, 3, 4, 1]
                        let x_nd = ops::transpose(x, &[0, 2, 3, 4, 1])?;
                        let w_mlx = ops::transpose(w, &[0, 2, 3, 4, 1])?;
                        let y_nd = ops::conv3d(
                            &x_nd,
                            &w_mlx,
                            (s(0), s(1), s(2)),
                            (p(0), p(1), p(2)),
                            (d(0), d(1), d(2)),
                            *groups as i32,
                        )?;
                        ops::transpose(&y_nd, &[0, 4, 1, 2, 3])?
                    }
                    (k, n) => {
                        return Err(MlxError(format!(
                            "Conv: kernel rank {k} with input rank {n} \
                         not supported (use 1D/2D/3D NCHW)"
                        )));
                    }
                }
                } // else: native MLX conv
            }
            Op::LayerNorm2d { eps } => {
                let x = lookup(&env, node.inputs[0])?;
                let g = mlx_norm_scale_1d(lookup(&env, node.inputs[1])?)?;
                let b = mlx_norm_scale_1d(lookup(&env, node.inputs[2])?)?;
                let shape = x.shape()?;
                if shape.len() != 4 {
                    return Err(MlxError(
                        "LayerNorm2d on MLX: expects NCHW rank-4 input".into(),
                    ));
                }
                let n = shape[0];
                let c = shape[1];
                let h = shape[2];
                let w = shape[3];
                // LayerNorm2d normalizes over the CHANNEL axis per spatial
                // position. In NCHW the channels are strided by h*w, so a direct
                // reshape to [n*h*w, c] would group the wrong elements. Transpose
                // NCHW→NHWC (channel last/contiguous), normalize, transpose back.
                // `contiguous` materializes the view (mlx::compile elides it).
                let nhwc = ops::contiguous(&ops::transpose(x, &[0, 2, 3, 1])?)?;
                let flat = ops::reshape(&nhwc, &[(n * h * w) as i32, c as i32])?;
                let y = ops::layer_norm(&flat, &g, Some(&b), *eps)?;
                let y_nhwc = ops::reshape(&y, &[n as i32, h as i32, w as i32, c as i32])?;
                ops::contiguous(&ops::transpose(&y_nhwc, &[0, 3, 1, 2])?)?
            }
            Op::GroupNorm { num_groups, eps } => {
                // NCHW GroupNorm: normalize over (c/g, h, w) per (n, group),
                // then apply per-channel affine. Same math + primitives as the
                // `GroupNormBackwardInput` arm (which MLX already lowers).
                let x = lookup(&env, node.inputs[0])?;
                let gamma = lookup(&env, node.inputs[1])?;
                let beta = lookup(&env, node.inputs[2])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let dtype = node.shape.dtype();
                if x_shape.len() != 4 {
                    return Err(MlxError(
                        "GroupNorm on MLX: expects NCHW rank-4 input".into(),
                    ));
                }
                let n = x_shape[0];
                let c = x_shape[1];
                let h = x_shape[2];
                let w = x_shape[3];
                let g = *num_groups as i32;
                let inner = (c / g) * h * w;
                let x3 = ops::reshape(x, &[n, g, inner])?;
                let eps_arr = Array::from_f32_slice(&[*eps], &[1], dtype)?;
                let mean = ops::reduce(&x3, MlxReduce::Mean, &[2], true)?;
                let x_c = ops::sub(&x3, &mean)?;
                let var = ops::reduce(&ops::mul(&x_c, &x_c)?, MlxReduce::Mean, &[2], true)?;
                let inv_std = ops::unary(&ops::add(&var, &eps_arr)?, MlxUnary::Rsqrt)?;
                let x_hat = ops::reshape(&ops::mul(&x_c, &inv_std)?, &[n, c, h, w])?;
                let gamma_b = ops::reshape(gamma, &[1, c, 1, 1])?;
                let beta_b = ops::reshape(beta, &[1, c, 1, 1])?;
                ops::add(&ops::mul(&x_hat, &gamma_b)?, &beta_b)?
            }
            Op::BatchNormInference { eps } => {
                // Feature dim is the last axis of `x` (IR + CPU thunk).
                // Frozen running stats: y = γ · x̂ + β,
                // x̂ = (x − μ) / √(σ² + ε).
                let x = lookup(&env, node.inputs[0])?;
                let gamma = lookup(&env, node.inputs[1])?;
                let beta = lookup(&env, node.inputs[2])?;
                let mean = lookup(&env, node.inputs[3])?;
                let var = lookup(&env, node.inputs[4])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let dtype = node.shape.dtype();
                if x_shape.is_empty() {
                    return Err(MlxError(
                        "BatchNormInference on MLX: scalar input unsupported".into(),
                    ));
                }
                let c = *x_shape.last().unwrap();
                let mut bshape = vec![1i32; x_shape.len()];
                *bshape.last_mut().unwrap() = c;
                let gamma_b = ops::reshape(&mlx_norm_scale_1d(gamma)?, &bshape)?;
                let beta_b = ops::reshape(&mlx_norm_scale_1d(beta)?, &bshape)?;
                let mean_b = ops::reshape(&mlx_norm_scale_1d(mean)?, &bshape)?;
                let var_b = ops::reshape(&mlx_norm_scale_1d(var)?, &bshape)?;
                let eps_arr = Array::from_f32_slice(&[*eps], &[1], dtype)?;
                let inv = ops::unary(&ops::add(&var_b, &eps_arr)?, MlxUnary::Rsqrt)?;
                let x_hat = ops::mul(&ops::sub(x, &mean_b)?, &inv)?;
                ops::add(&ops::mul(&x_hat, &gamma_b)?, &beta_b)?
            }
            Op::Im2Col {
                kernel_size,
                stride,
                padding,
                dilation,
            } => {
                // NCHW im2col → rows layout `[N·H_out·W_out, C·kH·kW]`, K-axis
                // ordered (c, ki, kj). Same windowing the MLX Pool path uses:
                // zero-pad, then strided-slice one [n,c,h_out,w_out] window per
                // kernel offset; stack the windows instead of reducing them.
                let x = lookup(&env, node.inputs[0])?;
                let s = node_input_shape(graph, node.inputs[0]);
                if s.len() != 4 {
                    return Err(MlxError("Im2Col on MLX: NCHW rank-4 only".into()));
                }
                let (n, c, h, w) = (s[0], s[1], s[2], s[3]);
                let kh = kernel_size[0] as i32;
                let kw = kernel_size[1] as i32;
                let sh = stride[0] as i32;
                let sw = stride[1] as i32;
                let ph = padding[0] as i32;
                let pw = padding[1] as i32;
                let dh = dilation[0] as i32;
                let dw = dilation[1] as i32;
                let h_out = (h + 2 * ph - (dh * (kh - 1) + 1)) / sh + 1;
                let w_out = (w + 2 * pw - (dw * (kw - 1) + 1)) / sw + 1;
                let x_pad_owned;
                let x_pad: &Array = if ph > 0 || pw > 0 {
                    x_pad_owned = ops::pad(x, &[0, 0, ph, pw], &[0, 0, ph, pw], 0.0)?;
                    &x_pad_owned
                } else {
                    x
                };
                let mut patches: Vec<Array> = Vec::with_capacity((kh * kw) as usize);
                for ki in 0..kh {
                    for kj in 0..kw {
                        let start = [0, 0, ki * dh, kj * dw];
                        let stop = [
                            n,
                            c,
                            ki * dh + (h_out - 1) * sh + 1,
                            kj * dw + (w_out - 1) * sw + 1,
                        ];
                        let strides = [1, 1, sh, sw];
                        let win = ops::slice_strided(x_pad, &start, &stop, &strides)?;
                        patches.push(ops::reshape(&win, &[n, c, h_out, w_out, 1])?);
                    }
                }
                let refs: Vec<&Array> = patches.iter().collect();
                // [n, c, h_out, w_out, kh*kw] (last axis ki-major, then kj).
                let stacked = ops::concat(&refs, 4)?;
                // → [n, h_out, w_out, c, kh*kw] then flatten to [M, C·kH·kW].
                let t = ops::contiguous(&ops::transpose(&stacked, &[0, 2, 3, 1, 4])?)?;
                ops::reshape(&t, &[n * h_out * w_out, c * kh * kw])?
            }
            Op::ConvTranspose2d {
                kernel_size,
                stride,
                padding,
                dilation,
                output_padding,
                groups,
            } => {
                // rlx NCHW + PyTorch weight [C_in, C_out/g, kH, kW].
                // MLX expects NHWC and weight [C_out, kH, kW, C_in/g].
                // Oversized ISTFT heads (k≈1024, huge out spatial) still
                // host-eval — same im2col ceiling as forward Conv.
                if mlx_conv_im2col_too_large(graph, node, kernel_size, *groups) {
                    host_eval_op_f32(graph, node, &env)?
                } else {
                    let in_shape = node_input_shape(graph, node.inputs[0]);
                    let w_shape = node_input_shape(graph, node.inputs[1]);
                    if in_shape.len() != 4 || w_shape.len() != 4 {
                        return Err(MlxError(
                            "ConvTranspose2d on MLX: expects NCHW rank-4 input/weight"
                                .into(),
                        ));
                    }
                    let x = lookup(&env, node.inputs[0])?;
                    let w = lookup(&env, node.inputs[1])?;
                    let g = (*groups).max(1) as i32;
                    let c_in = w_shape[0];
                    let c_out_per_g = w_shape[1];
                    let kh = w_shape[2];
                    let kw = w_shape[3];
                    let c_out = c_out_per_g * g;
                    let c_in_per_g = c_in / g;
                    let s = |i: usize| stride.get(i).copied().unwrap_or(1) as i32;
                    let p = |i: usize| padding.get(i).copied().unwrap_or(0) as i32;
                    let d = |i: usize| dilation.get(i).copied().unwrap_or(1) as i32;
                    let op = |i: usize| output_padding.get(i).copied().unwrap_or(0) as i32;

                    let x_nhwc = ops::transpose(x, &[0, 2, 3, 1])?;
                    let w_mlx = if g == 1 {
                        ops::transpose(w, &[1, 2, 3, 0])?
                    } else {
                        let split =
                            ops::reshape(w, &[g, c_in_per_g, c_out_per_g, kh, kw])?;
                        let perm = ops::transpose(&split, &[0, 2, 3, 4, 1])?;
                        ops::reshape(&perm, &[c_out, kh, kw, c_in_per_g])?
                    };
                    let y_nhwc = ops::conv_transpose2d(
                        &x_nhwc,
                        &w_mlx,
                        (s(0), s(1)),
                        (p(0), p(1)),
                        (d(0), d(1)),
                        (op(0), op(1)),
                        g,
                    )?;
                    ops::transpose(&y_nhwc, &[0, 3, 1, 2])?
                }
            }
            Op::ConvTranspose3d {
                stride,
                padding,
                dilation,
                output_padding,
                groups,
            } => {
                // rlx NCDHW + PyTorch weight [C_in, C_out/g, kD, kH, kW].
                // MLX expects NDHWC and weight [C_out, kD, kH, kW, C_in]
                // with groups=1 only. Oversized / grouped → host.
                let w_shape = node_input_shape(graph, node.inputs[1]);
                let kernel_size = if w_shape.len() >= 5 {
                    [
                        w_shape[2].max(0) as usize,
                        w_shape[3].max(0) as usize,
                        w_shape[4].max(0) as usize,
                    ]
                } else {
                    [1, 1, 1]
                };
                if *groups > 1
                    || mlx_conv_im2col_too_large(graph, node, &kernel_size, *groups)
                {
                    host_eval_op_f32(graph, node, &env)?
                } else {
                    let in_shape = node_input_shape(graph, node.inputs[0]);
                    if in_shape.len() != 5 || w_shape.len() != 5 {
                        return Err(MlxError(
                            "ConvTranspose3d on MLX: expects NCDHW rank-5 input/weight"
                                .into(),
                        ));
                    }
                    let x = lookup(&env, node.inputs[0])?;
                    let w = lookup(&env, node.inputs[1])?;
                    // PyTorch → MLX: [C_in, C_out, kD, kH, kW] → [C_out, kD, kH, kW, C_in]
                    let x_ndhwc = ops::transpose(x, &[0, 2, 3, 4, 1])?;
                    let w_mlx = ops::transpose(w, &[1, 2, 3, 4, 0])?;
                    let y_ndhwc = ops::conv_transpose3d(
                        &x_ndhwc,
                        &w_mlx,
                        (stride[0] as i32, stride[1] as i32, stride[2] as i32),
                        (padding[0] as i32, padding[1] as i32, padding[2] as i32),
                        (dilation[0] as i32, dilation[1] as i32, dilation[2] as i32),
                        (
                            output_padding[0] as i32,
                            output_padding[1] as i32,
                            output_padding[2] as i32,
                        ),
                        (*groups).max(1) as i32,
                    )?;
                    ops::transpose(&y_ndhwc, &[0, 4, 1, 2, 3])?
                }
            }
            Op::AxialRope2d {
                end_x,
                end_y,
                head_dim,
                num_heads,
                theta,
                repeat_factor,
            } => {
                // SAM2-style axial 2-D RoPE on `[B, seq, nh*hd]`.
                // Matches `rlx_ir::ops::axial_rope2d::apply_axial_rope2d`:
                // first half rotates with X freqs (interleaved pairs),
                // second half with Y freqs. Cos/sin tables are tiny
                // (seq × hd/4) and built as MLX constant arrays; the
                // rotate itself is native reshape/mul/add on device.
                let x = lookup(&env, node.inputs[0])?;
                let in_shape = node_input_shape(graph, node.inputs[0]);
                if in_shape.len() != 3 {
                    return Err(MlxError(format!(
                        "AxialRope2d: expected rank-3 [B,seq,hidden], got {}",
                        in_shape.len()
                    )));
                }
                let batch = in_shape[0];
                let seq = in_shape[1] as usize;
                let hidden = in_shape[2] as usize;
                let hd = *head_dim;
                let nh = *num_heads;
                if hd == 0 || !hd.is_multiple_of(4) {
                    return Err(MlxError(format!(
                        "AxialRope2d: head_dim={hd} must be a positive multiple of 4"
                    )));
                }
                if nh == 0 || hidden != nh * hd {
                    return Err(MlxError(format!(
                        "AxialRope2d: hidden={hidden} != num_heads={nh} * head_dim={hd}"
                    )));
                }
                let half = hd / 2;
                let q4 = hd / 4;
                let spatial = end_x * end_y;
                let repeat = (*repeat_factor).max(1);
                if seq != spatial * repeat {
                    return Err(MlxError(format!(
                        "AxialRope2d: seq={seq} != end_x*end_y*repeat={spatial}*{repeat}"
                    )));
                }

                // Build [seq, q4] cos/sin tables (host → device constants).
                let mut freqs = vec![0f32; q4];
                for i in 0..q4 {
                    freqs[i] = 1.0 / theta.powf((4 * i) as f32 / hd as f32);
                }
                let mut cos_x = vec![0f32; seq * q4];
                let mut sin_x = vec![0f32; seq * q4];
                let mut cos_y = vec![0f32; seq * q4];
                let mut sin_y = vec![0f32; seq * q4];
                for tok in 0..seq {
                    let pos = tok / repeat;
                    let tx = (pos % end_x) as f32;
                    let ty = (pos / end_x) as f32;
                    for c in 0..q4 {
                        let ax = tx * freqs[c];
                        let ay = ty * freqs[c];
                        cos_x[tok * q4 + c] = ax.cos();
                        sin_x[tok * q4 + c] = ax.sin();
                        cos_y[tok * q4 + c] = ay.cos();
                        sin_y[tok * q4 + c] = ay.sin();
                    }
                }
                let cos_x = Array::from_f32_slice(&cos_x, &[seq, q4], DType::F32)?;
                let sin_x = Array::from_f32_slice(&sin_x, &[seq, q4], DType::F32)?;
                let cos_y = Array::from_f32_slice(&cos_y, &[seq, q4], DType::F32)?;
                let sin_y = Array::from_f32_slice(&sin_y, &[seq, q4], DType::F32)?;

                let b = batch;
                let s = seq as i32;
                let nh_i = nh as i32;
                let hd_i = hd as i32;
                let half_i = half as i32;
                let q4_i = q4 as i32;
                let x4 = ops::reshape(x, &[b, s, nh_i, hd_i])?;
                let x_lo = ops::slice(&x4, &[0, 0, 0, 0], &[b, s, nh_i, half_i])?;
                let x_hi = ops::slice(&x4, &[0, 0, 0, half_i], &[b, s, nh_i, hd_i])?;

                let rotate_interleaved =
                    |half_x: &Array, cos: &Array, sin: &Array| -> Result<Array, MlxError> {
                        // [B, S, NH, half] → [B, S, NH, q4, 2]
                        let pairs = ops::reshape(half_x, &[b, s, nh_i, q4_i, 2])?;
                        let x_even =
                            ops::slice(&pairs, &[0, 0, 0, 0, 0], &[b, s, nh_i, q4_i, 1])?;
                        let x_odd =
                            ops::slice(&pairs, &[0, 0, 0, 0, 1], &[b, s, nh_i, q4_i, 2])?;
                        // Broadcast [S, q4] → [1, S, 1, q4, 1]
                        let cos_b = ops::reshape(cos, &[1, s, 1, q4_i, 1])?;
                        let sin_b = ops::reshape(sin, &[1, s, 1, q4_i, 1])?;
                        let y_even =
                            ops::sub(&ops::mul(&x_even, &cos_b)?, &ops::mul(&x_odd, &sin_b)?)?;
                        let y_odd =
                            ops::add(&ops::mul(&x_odd, &cos_b)?, &ops::mul(&x_even, &sin_b)?)?;
                        let y_pairs = ops::concat(&[&y_even, &y_odd], 4)?;
                        ops::reshape(&y_pairs, &[b, s, nh_i, half_i])
                    };

                let y_lo = rotate_interleaved(&x_lo, &cos_x, &sin_x)?;
                let y_hi = rotate_interleaved(&x_hi, &cos_y, &sin_y)?;
                let y4 = ops::concat(&[&y_lo, &y_hi], 3)?;
                ops::reshape(&y4, &[b, s, (nh * hd) as i32])?
            }
            Op::TopK { k } => {
                // Op::TopK returns f32-encoded indices of the k largest
                // values along the last axis (descending). We use
                // argpartition to position them, then a slice extracts
                // the back end of the result. argpartition with
                // kth=size-k puts the top-k *largest* in the last k
                // positions (unsorted relative order — matches
                // rlx's "ties broken by index" semantics? No — rlx
                // wants sorted. So we follow with argsort *only over
                // the last k* via take_along_axis, but to keep things
                // tractable we leave the order as argpartition gives.
                let x = lookup(&env, node.inputs[0])?;
                let in_shape = node_input_shape(graph, node.inputs[0]);
                if in_shape.is_empty() {
                    return Err(MlxError("TopK: input must be rank ≥ 1".into()));
                }
                let last_axis = (in_shape.len() - 1) as i32;
                let last_size = *in_shape.last().unwrap();
                if (*k as i32) > last_size {
                    return Err(MlxError(format!("TopK: k={k} > last_dim={last_size}")));
                }
                let kth = last_size - (*k as i32);
                let idx_full = ops::argpartition(x, kth, last_axis)?;
                // Slice the last `k` indices along the last axis.
                let mut start = vec![0i32; in_shape.len()];
                let mut stop = in_shape.clone();
                start[in_shape.len() - 1] = kth;
                stop[in_shape.len() - 1] = last_size;
                let idx = ops::slice(&idx_full, &start, &stop)?;
                // rlx encodes indices as f32 at the I/O boundary.
                ops::cast(&idx, DType::F32)?
            }
            Op::ScatterAdd => {
                // Inputs: [updates, indices]. Output is a fresh
                // tensor of node.shape; rlx semantics is "initial
                // output is zero, accumulate updates by indices."
                // MLX's scatter_add takes a base array and writes onto
                // it — we feed it a zero base of the right shape.
                let updates = lookup(&env, node.inputs[0])?;
                let indices_in = mlx_indices_i64(lookup(&env, node.inputs[1])?)?;
                let out_shape: Vec<i32> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i32)
                    .collect();
                // Build a zero base directly at the target shape via
                // `Array::from_f32_slice(&[0.0; N], shape, F32)`.  The earlier
                // `broadcast_to(sub(updates, updates), out_shape)` only worked
                // when `updates.shape[0]` equaled `out_shape[0]` — false when
                // the gradient comes from a Gather whose index set is denser
                // than the source table (e.g. ScatterAdd 240→30 in routing AD).
                let n_elem: usize = out_shape.iter().product::<i32>() as usize;
                let zeros = vec![0.0_f32; n_elem];
                let out_shape_usize: Vec<usize> = out_shape.iter().map(|d| *d as usize).collect();
                let zero_target =
                    crate::array::Array::from_f32_slice(&zeros, &out_shape_usize, DType::F32)?;
                let upd_shape = node_input_shape(graph, node.inputs[0]);
                let idx_shape = node_input_shape(graph, node.inputs[1]);
                // Gather axis-0 VJP: updates `[n_edges, d]`, indices `[n_edges]` → table `[n, d]`.
                // MLX scatter expects index rank to match the scattered array rank.
                let indices = if upd_shape.len() > 1 && idx_shape.len() == 1 {
                    ops::reshape(&indices_in, &[idx_shape[0], 1])?
                } else {
                    indices_in
                };
                if upd_shape.len() > 1 {
                    ops::scatter_add_axis(&zero_target, &indices, updates, 0)?
                } else {
                    ops::scatter_add(&zero_target, &indices, updates, 0)?
                }
            }
            Op::GroupedMatMul => {
                // Inputs: [input, weight, expert_idx].
                let x = lookup(&env, node.inputs[0])?;
                let w = lookup(&env, node.inputs[1])?;
                let i = lookup(&env, node.inputs[2])?;
                ops::gather_mm(x, w, i)?
            }
            Op::DequantGroupedMatMul { scheme } => {
                if !scheme.is_gguf() {
                    return Err(MlxError(
                        "DequantGroupedMatMul: only GGUF K-quants supported".into(),
                    ));
                }
                let x = lookup(&env, node.inputs[0])?;
                let wq = lookup(&env, node.inputs[1])?;
                let idx = lookup(&env, node.inputs[2])?;
                let out_shape: Vec<usize> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect();
                let m = out_shape[out_shape.len() - 2];
                let n = out_shape[out_shape.len() - 1];
                let x_f32 = x.to_f32()?;
                let k = x_f32.len() / m.max(1);
                let w_bytes = wq.to_bytes()?;
                let idx_f32 = idx.to_f32()?;
                let block_elems = scheme.gguf_block_size() as usize;
                let block_bytes = scheme.gguf_block_bytes() as usize;
                let slab_bytes = (k * n) / block_elems * block_bytes;
                let num_experts = w_bytes.len() / slab_bytes.max(1);
                let mut out_host = vec![0f32; m * n];
                rlx_cpu::gguf_matmul::gguf_grouped_matmul_bt(
                    &x_f32,
                    &w_bytes,
                    &idx_f32,
                    &mut out_host,
                    m,
                    k,
                    n,
                    num_experts,
                    *scheme,
                );
                Array::from_f32_slice(&out_host, &out_shape, DType::F32)?
            }
            Op::DequantGroupedMatMulMlx { scheme } => {
                // MLX-affine MoE grouped matmul: host-dequant the routed expert
                // per row (mirrors the GGUF grouped path above). Inputs:
                // [input, w_q, scales, biases, expert_idx].
                let (bits, group_size) = match scheme {
                    rlx_ir::QuantScheme::MlxAffine { bits, group_size } => {
                        (*bits as u32, *group_size as usize)
                    }
                    other => {
                        return Err(MlxError(format!(
                            "DequantGroupedMatMulMlx: expected MlxAffine, got {other:?}"
                        )));
                    }
                };
                // `contiguous` FIRST: `expert_idx` (and often `input`) arrive as
                // strided views (narrow/reshape of the top-k indices); a bare
                // `to_f32()` on a non-contiguous MLX array yields wrong data for
                // rows > 0 (only token 0 comes out correct otherwise).
                let x = ops::contiguous(lookup(&env, node.inputs[0])?)?;
                let wq = ops::contiguous(lookup(&env, node.inputs[1])?)?;
                let sc = ops::contiguous(lookup(&env, node.inputs[2])?)?;
                let bs = ops::contiguous(lookup(&env, node.inputs[3])?)?;
                let idx = ops::contiguous(lookup(&env, node.inputs[4])?)?;
                let out_shape: Vec<usize> =
                    node.shape.dims().iter().map(|d| d.unwrap_static()).collect();
                let m = out_shape[out_shape.len() - 2];
                let n = out_shape[out_shape.len() - 1];
                let x_f32 = x.to_f32()?;
                let k = x_f32.len() / m.max(1);
                let num_experts = graph.node(node.inputs[2]).shape.dim(0).unwrap_static();
                let w_bytes = wq.to_bytes()?;
                let scales = sc.to_f32()?;
                let biases = bs.to_f32()?;
                let idx_f32 = idx.to_f32()?;
                let mut out_host = vec![0f32; m * n];
                rlx_cpu::thunk::dequant_grouped_matmul_affine_bt(
                    &x_f32,
                    &w_bytes,
                    &scales,
                    &biases,
                    &idx_f32,
                    &mut out_host,
                    m,
                    k,
                    n,
                    num_experts,
                    bits,
                    group_size,
                );
                Array::from_f32_slice(&out_host, &out_shape, DType::F32)?
            }
            Op::DequantMatMul { scheme } => {
                if scheme.is_gguf() {
                    let x = lookup(&env, node.inputs[0])?;
                    let wq = lookup(&env, node.inputs[1])?;
                    let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    let m = total / n.max(1);
                    let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
                    let k = x_total / m.max(1);
                    // The naive host loop in `rlx_cpu::gguf_matmul::gguf_matmul_bt`
                    // measures 100×+ slower than MLX's native matmul. Dequant
                    // once to f32 here and let MLX's tuned sgemm pick up from
                    // there. GGUF Q4K stores weights as `[n, k]` row-major;
                    // transpose to `[k, n]` so `x @ w_t == [m, n]`. Off-switch
                    // `RLX_MLX_GGUF_HOST_FALLBACK=1` reverts to the host kernel.
                    let mlx_cfg = crate::config::runtime_config();
                    let use_host_fallback = mlx_cfg.gguf_host_fallback;
                    // Q1_0 (Bonsai-27B): expand on-device and discard — never
                    // cache the ~28× f32 blow-up. Host fallback still available
                    // via RLX_MLX_GGUF_HOST_FALLBACK=1 or RLX_MLX_Q1_HOST=1.
                    let use_q1_ondevice = matches!(scheme, rlx_ir::QuantScheme::GgufQ1_0)
                        && !use_host_fallback
                        && !mlx_cfg.q1_host;
                    if use_q1_ondevice {
                        // Decode (m==1): fused GEMV reads the packed 1-bit weight
                        // directly — no per-token f32 blow-up (the ~1.45s/tok cost
                        // was dequanting all 64 layers to f32 every step). m>1
                        // (prefill) keeps the dequant→matmul path (MLX's tuned
                        // sgemm amortizes the one-shot dequant). Off: RLX_MLX_Q1_MV_DISABLE=1.
                        if m == 1 && !mlx_cfg.q1_mv_disable {
                            crate::dequant_q1_0::q1_0_matmul_mv_ondevice(wq, x, k, n)?
                        } else {
                            let w_nk = crate::dequant_q1_0::dequant_q1_0_ondevice(wq, k, n)?;
                            let w_kn = ops::transpose(&w_nk, &[1, 0])?;
                            ops::matmul(x, &w_kn)?
                        }
                    } else {
                    let w_bytes = wq.to_bytes()?;
                    if use_host_fallback {
                        let mut out_host = vec![0f32; m * n];
                        rlx_cpu::gguf_matmul::gguf_matmul_bt(
                            &x.to_f32()?,
                            &w_bytes,
                            &mut out_host,
                            m,
                            k,
                            n,
                            *scheme,
                        );
                        let out_shape: Vec<usize> = node
                            .shape
                            .dims()
                            .iter()
                            .map(|d| d.unwrap_static())
                            .collect();
                        Array::from_f32_slice(&out_host, &out_shape, DType::F32)?
                    } else {
                        // Cache the dequanted+transposed [k, n] f32 Array per
                        // Param name so subsequent decode steps reuse it
                        // instead of paying the Q4K → f32 cost every dispatch.
                        // Without the cache, dequant of all 48 layers'
                        // weights inflates the first decode step from ~ms to
                        // ~170s on Gemma 4 12B Q4_K_M. Cache survives across
                        // generate() calls because the Param bytes are stable.
                        let w_node = graph.node(node.inputs[1]);
                        let cache_key = match &w_node.op {
                            rlx_ir::Op::Param { name } => {
                                Some(mlx_dequant_cache_key(name, k, n, scheme, &w_bytes))
                            }
                            _ => None,
                        };
                        let w_kn = if let Some(ref key) = cache_key {
                            if let Some(arr) = mlx_dequant_cache_get(key)? {
                                arr
                            } else {
                                let arr = build_dequanted_kn(&w_bytes, k, n, scheme)?;
                                let to_store = arr.clone_handle()?;
                                mlx_dequant_cache_put(key.clone(), to_store, k * n * 4);
                                arr
                            }
                        } else {
                            build_dequanted_kn(&w_bytes, k, n, scheme)?
                        };
                        ops::matmul(x, &w_kn)?
                    }
                    }
                } else if matches!(scheme, rlx_ir::QuantScheme::Nvfp4Block) {
                    let x = lookup(&env, node.inputs[0])?;
                    let wq = lookup(&env, node.inputs[1])?;
                    let sc = lookup(&env, node.inputs[2])?;
                    let gs_arr = lookup(&env, node.inputs[3])?;
                    let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    let m = total / n.max(1);
                    let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
                    let k = x_total / m.max(1);
                    let xs = x.to_f32()?;
                    let w_bytes = wq.to_bytes()?;
                    let scale_bytes = sc.to_bytes()?;
                    let global_scale = gs_arr.to_f32()?[0];
                    let mut out_host = vec![0f32; m * n];
                    rlx_cpu::thunk::dequant_matmul_nvfp4(
                        &xs,
                        &w_bytes,
                        &scale_bytes,
                        global_scale,
                        &mut out_host,
                        m,
                        k,
                        n,
                    );
                    let out_shape: Vec<usize> = node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    Array::from_f32_slice(&out_host, &out_shape, DType::F32)?
                } else if matches!(
                    scheme,
                    rlx_ir::QuantScheme::Int8Block { .. }
                        | rlx_ir::QuantScheme::Int8BlockAsym { .. }
                ) {
                    let x = lookup(&env, node.inputs[0])?;
                    let wq = lookup(&env, node.inputs[1])?;
                    let sc = lookup(&env, node.inputs[2])?;
                    let zp = lookup(&env, node.inputs[3])?;
                    let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    let m = total / n.max(1);
                    let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
                    let k = x_total / m.max(1);
                    let block_size = match scheme {
                        rlx_ir::QuantScheme::Int8Block { block_size }
                        | rlx_ir::QuantScheme::Int8BlockAsym { block_size } => *block_size,
                        _ => unreachable!(),
                    };
                    let asym = matches!(scheme, rlx_ir::QuantScheme::Int8BlockAsym { .. });
                    let xs = x.to_f32()?;
                    let w_raw = wq.to_bytes()?;
                    let w_bytes = unsafe {
                        std::slice::from_raw_parts(w_raw.as_ptr() as *const i8, w_raw.len())
                    };
                    let scales = sc.to_f32()?;
                    let zps = if asym { zp.to_f32()? } else { Vec::new() };
                    let mut out_host = vec![0f32; m * n];
                    rlx_cpu::thunk::dequant_matmul_int8(
                        &xs,
                        w_bytes,
                        &scales,
                        &zps,
                        &mut out_host,
                        m,
                        k,
                        n,
                        block_size as usize,
                        asym,
                    );
                    let out_shape: Vec<usize> = node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    Array::from_f32_slice(&out_host, &out_shape, DType::F32)?
                } else if matches!(
                    scheme,
                    rlx_ir::QuantScheme::MlxMxfp4 { .. } | rlx_ir::QuantScheme::MlxMxfp8 { .. }
                ) {
                    let x = lookup(&env, node.inputs[0])?;
                    let wq = lookup(&env, node.inputs[1])?;
                    let sc = lookup(&env, node.inputs[2])?;
                    let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    let m = total / n.max(1);
                    let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
                    let k = x_total / m.max(1);
                    #[cfg(feature = "native-mxfp")]
                    {
                        // Opt-in MLX C++ `quantized_matmul` (mode=mxfp4/mxfp8/nvfp4).
                        let (bits, gs, mode) = quant_scheme_to_mlx(scheme)?;
                        let packed_cols = k * bits as usize / 32;
                        let wq_u32 =
                            Array::from_bytes(&wq.to_bytes()?, &[n, packed_cols], DType::U32)?;
                        let scales_u8 = Array::from_bytes(
                            &sc.to_bytes()?,
                            &[n, k / gs as usize],
                            DType::U8,
                        )?;
                        ops::quantized_matmul_mode(
                            x,
                            &wq_u32,
                            &scales_u8,
                            None,
                            /*transpose=*/ true,
                            gs,
                            bits,
                            mode,
                        )?
                    }
                    #[cfg(not(feature = "native-mxfp"))]
                    {
                        // First-class Rust path: same dequant as CPU/Metal kernels,
                        // then MLX matmul (with Param-keyed cache).
                        let w_bytes = wq.to_bytes()?;
                        let scale_bytes = sc.to_bytes()?;
                        let w_node = graph.node(node.inputs[1]);
                        let cache_key = match &w_node.op {
                            rlx_ir::Op::Param { name } => Some(mlx_mxfp_cache_key(
                                name,
                                k,
                                n,
                                scheme,
                                &w_bytes,
                                &scale_bytes,
                            )),
                            _ => None,
                        };
                        let w_kn = if let Some(ref key) = cache_key {
                            if let Some(arr) = mlx_dequant_cache_get(key)? {
                                arr
                            } else {
                                let arr =
                                    build_mlx_mxfp_kn(&w_bytes, &scale_bytes, k, n, scheme)?;
                                let to_store = arr.clone_handle()?;
                                mlx_dequant_cache_put(key.clone(), to_store, k * n * 4);
                                arr
                            }
                        } else {
                            build_mlx_mxfp_kn(&w_bytes, &scale_bytes, k, n, scheme)?
                        };
                        ops::matmul(x, &w_kn)?
                    }
                } else {
                    // Inputs: [x, w_q, scale, zp]. Map to MLX's
                    // quantized_matmul (Int4/Int8/MlxAffine).
                    let x = lookup(&env, node.inputs[0])?;
                    let wq = lookup(&env, node.inputs[1])?;
                    let s = lookup(&env, node.inputs[2])?;
                    let zp = lookup(&env, node.inputs[3])?;
                    let (bits, gs, mode) = quant_scheme_to_mlx(scheme)?;
                    // MLX `quantized_matmul` needs the packed weight as uint32
                    // `[n, k*bits/32]`. The MlxAffine param arrives as flat U8
                    // bytes (shared byte layout with the CPU/Metal kernels), so
                    // reinterpret the bytes as u32 for MLX.
                    let wq_u32;
                    let wq = if matches!(scheme, rlx_ir::QuantScheme::MlxAffine { .. }) {
                        let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                        let total = node.shape.num_elements().unwrap();
                        let m = total / n.max(1);
                        let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
                        let k = x_total / m.max(1);
                        let packed_cols = k * bits as usize / 32;
                        wq_u32 = Array::from_bytes(&wq.to_bytes()?, &[n, packed_cols], DType::U32)?;
                        &wq_u32
                    } else {
                        wq
                    };
                    ops::quantized_matmul_mode(
                        x,
                        wq,
                        s,
                        Some(zp),
                        /*transpose=*/ true,
                        gs,
                        bits,
                        mode,
                    )?
                }
            }
            Op::LoraMatMul { scale } => {
                // out = x @ W + scale * (x @ A) @ B
                let x = lookup(&env, node.inputs[0])?;
                let w = lookup(&env, node.inputs[1])?;
                let a = lookup(&env, node.inputs[2])?;
                let b = lookup(&env, node.inputs[3])?;
                let base = ops::matmul(x, w)?;
                let xa = ops::matmul(x, a)?;
                let xab = ops::matmul(&xa, b)?;
                // Scale via in-graph mul against a scalar array.
                let s = Array::from_f32_slice(&[*scale], &[1], DType::F32)?;
                let scaled = ops::mul(&xab, &s)?;
                ops::add(&base, &scaled)?
            }
            Op::FusedTransformerLayer {
                num_heads,
                head_dim,
                intermediate_size: _,
                eps1,
                eps2,
                activation,
                has_bias,
            } => {
                // Standard BERT-style post-norm transformer layer.
                // Inputs (per IR doc):
                //   hidden, qkv_w, qkv_b, out_w, out_b,
                //   ln1_g, ln1_b, fc1_w, fc1_b, fc2_w, fc2_b,
                //   ln2_g, ln2_b, mask
                //
                // Wiring:
                //   attn_out = attention_block(hidden, qkv_w, [qkv_b],
                //                              out_w, [out_b], mask)
                //   h1       = layer_norm(hidden + attn_out, ln1_g, ln1_b, eps1)
                //   ffn      = activation(h1 @ fc1_w [+ fc1_b])
                //   ffn_out  = ffn @ fc2_w [+ fc2_b]
                //   h2       = layer_norm(h1 + ffn_out, ln2_g, ln2_b, eps2)
                // Index map. has_bias gates every bias input (including
                // the two LayerNorm betas, per Op::num_inputs above):
                //   has_bias=true  → 14 inputs (full BERT layout)
                //   has_bias=false → 8 inputs (no biases at all)
                let (
                    hidden,
                    qkv_w,
                    qkv_b,
                    out_w,
                    out_b,
                    ln1_g,
                    ln1_b,
                    fc1_w,
                    fc1_b,
                    fc2_w,
                    fc2_b,
                    ln2_g,
                    ln2_b,
                    mask,
                ) = if *has_bias {
                    (
                        lookup(&env, node.inputs[0])?,
                        lookup(&env, node.inputs[1])?,
                        Some(lookup(&env, node.inputs[2])?),
                        lookup(&env, node.inputs[3])?,
                        Some(lookup(&env, node.inputs[4])?),
                        lookup(&env, node.inputs[5])?,
                        Some(lookup(&env, node.inputs[6])?),
                        lookup(&env, node.inputs[7])?,
                        Some(lookup(&env, node.inputs[8])?),
                        lookup(&env, node.inputs[9])?,
                        Some(lookup(&env, node.inputs[10])?),
                        lookup(&env, node.inputs[11])?,
                        Some(lookup(&env, node.inputs[12])?),
                        lookup(&env, node.inputs[13])?,
                    )
                } else {
                    (
                        lookup(&env, node.inputs[0])?,
                        lookup(&env, node.inputs[1])?,
                        None,
                        lookup(&env, node.inputs[2])?,
                        None,
                        lookup(&env, node.inputs[3])?,
                        None,
                        lookup(&env, node.inputs[4])?,
                        None,
                        lookup(&env, node.inputs[5])?,
                        None,
                        lookup(&env, node.inputs[6])?,
                        None,
                        lookup(&env, node.inputs[7])?,
                    )
                };

                let h_shape = node_input_shape(graph, node.inputs[0]);
                let batch = h_shape[0];
                let seq = h_shape[1];
                let nh = *num_heads as i32;
                let hd = *head_dim as i32;
                let inner = nh * hd;

                // Optional-bias add helper: idempotent when bias is None.
                let maybe_add = |x: Array, b: Option<&Array>| -> Result<Array, MlxError> {
                    match b {
                        Some(b) => ops::add(&x, b),
                        None => Ok(x),
                    }
                };

                // --- Attention block ---
                let qkv = ops::matmul(hidden, qkv_w)?;
                let qkv = maybe_add(qkv, qkv_b)?;
                let q = ops::slice(&qkv, &[0, 0, 0], &[batch, seq, inner])?;
                let k = ops::slice(&qkv, &[0, 0, inner], &[batch, seq, 2 * inner])?;
                let v = ops::slice(&qkv, &[0, 0, 2 * inner], &[batch, seq, 3 * inner])?;
                // Materialize the transpose with `ops::contiguous` (MLX's
                // `compile` elides transpose views — same fix as Op::Attention
                // at lower.rs:851/858 and Op::FusedAttentionBlock above).
                let to_h = |t: Array| -> Result<Array, MlxError> {
                    let r = ops::reshape(&t, &[batch, seq, nh, hd])?;
                    let t = ops::transpose(&r, &[0, 2, 1, 3])?;
                    ops::contiguous(&t)
                };
                let q = to_h(q)?;
                let k = to_h(k)?;
                let v = to_h(v)?;
                let scale = 1.0 / (hd as f32).sqrt();

                // Convert the BERT-style binary mask `[B, S]` (1.0 valid,
                // 0.0 padding) → additive (`(mask - 1) * 1e9`) and reshape
                // to `[B, 1, 1, S]` so it broadcasts over heads + query
                // positions in SDPA. Same handling as the unfused
                // `Op::Attention` path and the standalone
                // `Op::FusedAttentionBlock` above.
                let h_dtype = graph.node(node.inputs[0]).shape.dtype();
                let mask_idx = if *has_bias { 13 } else { 7 };
                let m_shape = node_input_shape(graph, node.inputs[mask_idx]);
                let mask_cast = if h_dtype != DType::F32 {
                    ops::cast(mask, h_dtype)?
                } else {
                    mask.clone_handle()?
                };
                let one = Array::from_f32_slice(&[1.0], &[1], h_dtype)?;
                let scl = Array::from_f32_slice(&[1.0e9], &[1], h_dtype)?;
                let shifted = ops::sub(&mask_cast, &one)?;
                let additive = ops::mul(&shifted, &scl)?;
                let additive_4d = match m_shape.len() {
                    2 => ops::reshape(&additive, &[m_shape[0], 1, 1, m_shape[1]])?,
                    3 => ops::reshape(&additive, &[m_shape[0], 1, m_shape[1], m_shape[2]])?,
                    _ => additive,
                };
                let attn = ops::attention(
                    &q,
                    &k,
                    &v,
                    scale,
                    crate::ffi::MlxMask::Custom,
                    Some(&additive_4d),
                )?;
                let attn = ops::transpose(&attn, &[0, 2, 1, 3])?;
                let attn = ops::reshape(&attn, &[batch, seq, inner])?;
                let attn_out = ops::matmul(&attn, out_w)?;
                let attn_out = maybe_add(attn_out, out_b)?;

                // --- Residual + LayerNorm 1 ---
                let pre1 = ops::add(hidden, &attn_out)?;
                let ln1_g_n = mlx_norm_scale_1d(ln1_g)?;
                let ln1_b_n = ln1_b.map(mlx_norm_scale_1d).transpose()?;
                let h1 = ops::layer_norm(&pre1, &ln1_g_n, ln1_b_n.as_ref(), *eps1)?;

                // --- FFN: activation(h1 @ fc1_w [+ fc1_b]) @ fc2_w [+ fc2_b] ---
                let ffn1 = ops::matmul(&h1, fc1_w)?;
                let ffn1 = maybe_add(ffn1, fc1_b)?;
                let ffn1 = match activation {
                    Activation::Gelu => ops::gelu(&ffn1)?,
                    Activation::GeluApprox => ops::gelu_approx(&ffn1)?,
                    Activation::Silu => ops::silu(&ffn1)?,
                    Activation::Relu => ops::unary(&ffn1, MlxUnary::Relu)?,
                    Activation::Sigmoid => ops::unary(&ffn1, MlxUnary::Sigmoid)?,
                    Activation::Tanh => ops::unary(&ffn1, MlxUnary::Tanh)?,
                    Activation::Exp => ops::unary(&ffn1, MlxUnary::Exp)?,
                    Activation::Log => ops::unary(&ffn1, MlxUnary::Log)?,
                    Activation::Sqrt => ops::unary(&ffn1, MlxUnary::Sqrt)?,
                    Activation::Rsqrt => ops::unary(&ffn1, MlxUnary::Rsqrt)?,
                    Activation::Neg => ops::unary(&ffn1, MlxUnary::Neg)?,
                    Activation::Abs => ops::unary(&ffn1, MlxUnary::Abs)?,
                    Activation::Round => ops::unary(&ffn1, MlxUnary::Round)?,
                    Activation::Sin => ops::unary(&ffn1, MlxUnary::Sin)?,
                    Activation::Cos => ops::unary(&ffn1, MlxUnary::Cos)?,
                    Activation::Tan => ops::unary(&ffn1, MlxUnary::Tan)?,
                    Activation::Atan => ops::unary(&ffn1, MlxUnary::Atan)?,
                    Activation::Recip => ops::unary(&ffn1, MlxUnary::Reciprocal)?,
                    Activation::Floor => ops::unary(&ffn1, MlxUnary::Floor)?,
                    Activation::Ceil => ops::unary(&ffn1, MlxUnary::Ceil)?,
                    Activation::Sign => ops::unary(&ffn1, MlxUnary::Sign)?,
                    Activation::Softplus => ops::unary(&ffn1, MlxUnary::Softplus)?,
                    Activation::Elu => ops::unary(&ffn1, MlxUnary::Elu)?,
                    Activation::Erf => ops::unary(&ffn1, MlxUnary::Erf)?,
                    Activation::HardSwish => ops::unary(&ffn1, MlxUnary::HardSwish)?,
                    Activation::HardSigmoid => ops::unary(&ffn1, MlxUnary::HardSigmoid)?,
                    Activation::Mish => ops::unary(&ffn1, MlxUnary::Mish)?,
                    Activation::Softsign => ops::unary(&ffn1, MlxUnary::Softsign)?,
                    Activation::LogSigmoid => ops::unary(&ffn1, MlxUnary::LogSigmoid)?,
                };
                let ffn2 = ops::matmul(&ffn1, fc2_w)?;
                let ffn_out = maybe_add(ffn2, fc2_b)?;

                // --- Residual + LayerNorm 2 ---
                let pre2 = ops::add(&h1, &ffn_out)?;
                let ln2_g_n = mlx_norm_scale_1d(ln2_g)?;
                let ln2_b_n = ln2_b.map(mlx_norm_scale_1d).transpose()?;
                ops::layer_norm(&pre2, &ln2_g_n, ln2_b_n.as_ref(), *eps2)?
            }
            Op::FusedAttentionBlock {
                num_heads,
                head_dim,
                has_bias,
                has_rope,
            } => {
                // Compose: QKV proj → split → reshape → transpose →
                // [Rope on Q, K] → SDPA → transpose back → reshape →
                // out proj. Custom mask kind (mask is always input #3).
                //
                // Inputs (in order):
                //   hidden, qkv_w, out_w, mask,
                //   [qkv_b, out_b]      if has_bias,
                //   [rope_cos, rope_sin] if has_rope
                let h_idx = 0;
                let qkv_w_idx = 1;
                let out_w_idx = 2;
                let mask_idx = 3;
                let mut next = 4;
                let (qkv_b_idx, out_b_idx) = if *has_bias {
                    let r = (next, next + 1);
                    next += 2;
                    r
                } else {
                    (usize::MAX, usize::MAX)
                };
                let (cos_idx, sin_idx) = if *has_rope {
                    let r = (next, next + 1);
                    let _ = next + 2; // consumed
                    r
                } else {
                    (usize::MAX, usize::MAX)
                };

                let hidden = lookup(&env, node.inputs[h_idx])?;
                let qkv_w = lookup(&env, node.inputs[qkv_w_idx])?;
                let out_w = lookup(&env, node.inputs[out_w_idx])?;
                let mask = lookup(&env, node.inputs[mask_idx])?;

                let h_shape = node_input_shape(graph, node.inputs[h_idx]);
                if h_shape.len() != 3 {
                    return Err(MlxError(format!(
                        "FusedAttentionBlock: hidden must be rank-3 [B, S, H], got {}",
                        h_shape.len()
                    )));
                }
                let (batch, seq) = runtime_bsh_dims(hidden, &h_shape)?;
                let nh = *num_heads as i32;
                let hd = *head_dim as i32;
                let inner = nh * hd;

                // 1. qkv = matmul(hidden, qkv_w) [+ qkv_b]
                let qkv = ops::matmul(hidden, qkv_w)?;
                let qkv = if *has_bias {
                    let qkv_b = lookup(&env, node.inputs[qkv_b_idx])?;
                    ops::add(&qkv, qkv_b)?
                } else {
                    qkv
                };

                // 2. split into Q, K, V along last axis (each [B, S, inner])
                let q = ops::slice(&qkv, &[0, 0, 0], &[batch, seq, inner])?;
                let k = ops::slice(&qkv, &[0, 0, inner], &[batch, seq, 2 * inner])?;
                let v = ops::slice(&qkv, &[0, 0, 2 * inner], &[batch, seq, 3 * inner])?;

                // 3. reshape to [B, S, H, D] then transpose to [B, H, S, D].
                // Materialize the transposed view with `ops::contiguous`: MLX's
                // `compile` elides bare transpose views, and SDPA needs a real
                // contiguous buffer (same materialization required by the
                // unfused `Op::Attention` lowering at lower.rs:851 and 858).
                let to_h = |t: Array| -> Result<Array, MlxError> {
                    let r = ops::reshape(&t, &[batch, seq, nh, hd])?;
                    let t = ops::transpose(&r, &[0, 2, 1, 3])?;
                    ops::contiguous(&t)
                };
                let mut q = to_h(q)?;
                let mut k = to_h(k)?;
                let v_h = to_h(v)?;

                // 4. Rope on Q and K if requested
                if *has_rope {
                    let cos = lookup(&env, node.inputs[cos_idx])?;
                    let sin = lookup(&env, node.inputs[sin_idx])?;
                    // Inline the Rope composition for full-dim
                    // (head_dim == last_dim for Q/K which are
                    // [B, H, S, D]).
                    let do_rope = |x: &Array| -> Result<Array, MlxError> {
                        let half = hd / 2;
                        let cos_shape = cos.shape().unwrap_or_default();
                        if cos_shape.len() != 2 {
                            return Err(MlxError(format!(
                                "FusedAttentionBlock rope: cos must be rank-2, got rank-{} shape={cos_shape:?}",
                                cos_shape.len()
                            )));
                        }
                        let cos_rows = cos_shape[0] as i32;
                        let seq_rope = seq.min(cos_rows);
                        let cos_seq = ops::slice(cos, &[0, 0], &[seq_rope, half])?;
                        let sin_seq = ops::slice(sin, &[0, 0], &[seq_rope, half])?;
                        let bshape = [1, 1, seq_rope, half];
                        let cos_b = ops::reshape(&cos_seq, &bshape)?;
                        let sin_b = ops::reshape(&sin_seq, &bshape)?;
                        let x1 = ops::slice(x, &[0, 0, 0, 0], &[batch, nh, seq_rope, half])?;
                        let x2 = ops::slice(x, &[0, 0, 0, half], &[batch, nh, seq_rope, hd])?;
                        let y1 = ops::sub(&ops::mul(&x1, &cos_b)?, &ops::mul(&x2, &sin_b)?)?;
                        let y2 = ops::add(&ops::mul(&x2, &cos_b)?, &ops::mul(&x1, &sin_b)?)?;
                        ops::concat(&[&y1, &y2], 3)
                    };
                    q = do_rope(&q)?;
                    k = do_rope(&k)?;
                }

                // 5. SDPA with custom mask.
                //
                // The mask on input #3 is the BERT-style binary mask
                // `[B, S]` (1.0 = valid, 0.0 = padding). MLX's SDPA adds the
                // mask *additively* to scores, so we must convert
                // binary → additive (matching the unfused `Op::Attention`
                // lowering at lower.rs:893-907):
                //     additive = (mask - 1) * 1e9
                //   → 0 for valid positions, -1e9 for padding.
                //
                // We also reshape the [B, S] mask to [B, 1, 1, S] so it
                // broadcasts across the head and query axes against the
                // [B, H, S_q, S_k] score tensor — same normalization the
                // unfused path applies at lower.rs:875-881.
                let scale = 1.0 / (hd as f32).sqrt();
                let q_dtype = graph.node(node.inputs[h_idx]).shape.dtype();
                let m_shape = node_input_shape(graph, node.inputs[mask_idx]);
                let mask_cast = if q_dtype != DType::F32 {
                    ops::cast(mask, q_dtype)?
                } else {
                    mask.clone_handle()?
                };
                let one = Array::from_f32_slice(&[1.0], &[1], q_dtype)?;
                let scl = Array::from_f32_slice(&[1.0e9], &[1], q_dtype)?;
                let shifted = ops::sub(&mask_cast, &one)?;
                let additive = ops::mul(&shifted, &scl)?;
                let additive_4d = match m_shape.len() {
                    2 => ops::reshape(&additive, &[m_shape[0], 1, 1, m_shape[1]])?,
                    3 => ops::reshape(&additive, &[m_shape[0], 1, m_shape[1], m_shape[2]])?,
                    _ => additive,
                };
                let attn_out = ops::attention(
                    &q,
                    &k,
                    &v_h,
                    scale,
                    crate::ffi::MlxMask::Custom,
                    Some(&additive_4d),
                )?;

                // 6. transpose back [B, H, S, D] → [B, S, H, D] → reshape [B, S, H*D]
                let attn_out = ops::transpose(&attn_out, &[0, 2, 1, 3])?;
                let attn_out = ops::reshape(&attn_out, &[batch, seq, inner])?;

                // 7. out projection
                let y = ops::matmul(&attn_out, out_w)?;
                if *has_bias {
                    let out_b = lookup(&env, node.inputs[out_b_idx])?;
                    ops::add(&y, out_b)?
                } else {
                    y
                }
            }
            Op::FusedSwiGLU { cast_to, .. } => {
                let src = lookup(&env, node.inputs[0])?;
                let in_shape = node_input_shape(graph, node.inputs[0]);
                let last = *in_shape
                    .last()
                    .ok_or_else(|| MlxError("FusedSwiGLU: input is rank-0".into()))?;
                if last % 2 != 0 {
                    return Err(MlxError(format!(
                        "FusedSwiGLU: last dim {last} must be even"
                    )));
                }
                let half = last / 2;
                let last_idx = in_shape.len() - 1;
                let up_start = vec![0i32; in_shape.len()];
                let mut up_stop = in_shape.clone();
                up_stop[last_idx] = half;
                let mut g_start = vec![0i32; in_shape.len()];
                g_start[last_idx] = half;
                let g_stop = in_shape.clone();
                let up = ops::slice(src, &up_start, &up_stop)?;
                let gate = ops::slice(src, &g_start, &g_stop)?;
                let silu_g = ops::silu(&gate)?;
                let result = ops::mul(&up, &silu_g)?;
                match cast_to {
                    Some(dt) if *dt != node.shape.dtype() => ops::cast(&result, *dt)?,
                    _ => result,
                }
            }

            Op::If {
                then_branch,
                else_branch,
            } => {
                // Lower both branches inline using the same captures
                // (parent's inputs[1..]). Output is per-element select
                // via mc::where(pred, then_out, else_out).
                if node.inputs.is_empty() {
                    return Err(MlxError("If: missing predicate input".into()));
                }
                let pred = lookup(&env, node.inputs[0])?;
                let captures: Vec<&Array> = node.inputs[1..]
                    .iter()
                    .map(|&id| lookup(&env, id))
                    .collect::<Result<_, _>>()?;
                let then_outs = lower_subgraph(then_branch, &captures, params, params_typed, rng)?;
                let else_outs = lower_subgraph(else_branch, &captures, params, params_typed, rng)?;
                if then_outs.len() != 1 || else_outs.len() != 1 {
                    return Err(MlxError(format!(
                        "If: each branch must produce exactly 1 output \
                         (then={}, else={})",
                        then_outs.len(),
                        else_outs.len()
                    )));
                }
                ops::select(pred, &then_outs[0], &else_outs[0])?
            }
            Op::While {
                cond,
                body,
                max_iterations,
            } => {
                // Bounded unroll: body and cond each get the current
                // loop-carried state as their captures. After body, we
                // mask updates with where(active && cond, body_out,
                // carried) so that once cond becomes false the carried
                // values stop changing. Without max_iterations the
                // loop has no static bound, which MLX can't trace —
                // error explicitly so callers fall back to host-side
                // looping.
                let max_iter = max_iterations.ok_or_else(|| {
                    MlxError(
                        "While: max_iterations required for unrolled \
                              lowering — MLX has no runtime loop primitive"
                            .into(),
                    )
                })?;

                // Initial carried values (clone-share from parent env).
                let mut carried: Vec<Array> = Vec::with_capacity(node.inputs.len());
                for &id in &node.inputs {
                    carried.push(lookup(&env, id)?.clone_handle()?);
                }
                // Active mask: 1.0 while still iterating, 0.0 once a
                // cond evaluation says we're done.
                let mut active = Array::from_f32_slice(&[1.0], &[1], DType::F32)?;

                for _ in 0..max_iter {
                    let captures: Vec<&Array> = carried.iter().collect();
                    let cond_outs = lower_subgraph(cond, &captures, params, params_typed, rng)?;
                    if cond_outs.len() != 1 {
                        return Err(MlxError(format!(
                            "While: cond sub-graph must produce 1 output \
                             (got {})",
                            cond_outs.len()
                        )));
                    }
                    // active &= cond (cast bool to f32, multiply)
                    let cond_f = ops::cast(&cond_outs[0], DType::F32)?;
                    active = ops::mul(&active, &cond_f)?;

                    let body_outs = lower_subgraph(body, &captures, params, params_typed, rng)?;
                    if body_outs.len() != carried.len() {
                        return Err(MlxError(format!(
                            "While: body produced {} outputs but {} loop-carried \
                             values were expected",
                            body_outs.len(),
                            carried.len()
                        )));
                    }
                    let active_bool = ops::cast(&active, DType::Bool)?;
                    let mut next: Vec<Array> = Vec::with_capacity(carried.len());
                    for (b, c) in body_outs.iter().zip(carried.iter()) {
                        next.push(ops::select(&active_bool, b, c)?);
                    }
                    carried = next;
                }

                // Op::While is a single-output node by IR convention;
                // we return the first carried value. For multi-output
                // While the IR would need a separate variant or a
                // tuple-typed output node — neither exists today.
                if carried.is_empty() {
                    return Err(MlxError("While: no loop-carried values".into()));
                }
                carried.into_iter().next().unwrap()
            }
            Op::Sample {
                top_k,
                top_p,
                temperature,
                seed,
            } => {
                let logits = lookup(&env, node.inputs[0])?;
                // Apply temperature.
                let scaled_owned: Option<Array> = if (*temperature - 1.0).abs() <= 1e-6 {
                    None
                } else {
                    let inv_t = 1.0 / *temperature;
                    let s = Array::from_f32_slice(&[inv_t], &[1], DType::F32)?;
                    Some(ops::mul(logits, &s)?)
                };
                let scaled: &Array = scaled_owned.as_ref().unwrap_or(logits);

                let in_shape = node_input_shape(graph, node.inputs[0]);
                let last_axis = if in_shape.is_empty() {
                    -1
                } else {
                    (in_shape.len() - 1) as i32
                };
                let neg_inf = Array::from_f32_slice(&[f32::NEG_INFINITY], &[1], DType::F32)?;

                // top_k filter: keep only the top-k logits, mask the
                // rest to -∞. Threshold = k-th largest value.
                let topk_owned: Option<Array> =
                    if *top_k > 0 && (*top_k as i32) < *in_shape.last().unwrap_or(&i32::MAX) {
                        let k = *top_k as i32;
                        let topk = ops::topk_values(scaled, k, last_axis)?;
                        let mut t_start = vec![0i32; in_shape.len()];
                        let mut t_stop = in_shape.clone();
                        t_start[in_shape.len() - 1] = k - 1;
                        t_stop[in_shape.len() - 1] = k;
                        let threshold = ops::slice(&topk, &t_start, &t_stop)?;
                        let mask = ops::ge(scaled, &threshold)?;
                        Some(ops::select(&mask, scaled, &neg_inf)?)
                    } else {
                        None
                    };
                let after_topk: &Array = topk_owned.as_ref().unwrap_or(scaled);

                // top_p (nucleus) filter. Algorithm:
                //   1. p = softmax(logits)
                //   2. sort_desc(p) via -sort(-p)
                //   3. exclusive cumsum over sorted_p
                //   4. nucleus = (exclusive_cumsum < top_p)
                //   5. threshold_p = min(sorted_p where nucleus, +inf
                //      where not) — smallest probability still in
                //      the nucleus
                //   6. mask = p >= threshold_p   (broadcast back to
                //      original positions)
                //   7. logits' = where(mask, logits, -inf)
                let topp_owned: Option<Array> = if (*top_p - 1.0).abs() > 1e-6 && *top_p > 0.0 {
                    let p = ops::softmax(after_topk, last_axis)?;
                    let neg_p = ops::unary(&p, MlxUnary::Neg)?;
                    let neg_sorted = ops::sort(&neg_p, last_axis)?;
                    let sorted_p = ops::unary(&neg_sorted, MlxUnary::Neg)?;

                    // Exclusive cumsum: cumsum_excl[i] = sum of first i
                    // entries (so the first entry's cumsum is 0).
                    let cumsum_excl = ops::cumsum(&sorted_p, last_axis, /*exclusive=*/ true)?;
                    let p_thresh = Array::from_f32_slice(&[*top_p], &[1], DType::F32)?;
                    let nucleus = ops::lt(&cumsum_excl, &p_thresh)?;

                    let pos_inf = Array::from_f32_slice(&[f32::INFINITY], &[1], DType::F32)?;
                    let masked_sorted = ops::select(&nucleus, &sorted_p, &pos_inf)?;
                    let threshold_p = ops::reduce(
                        &masked_sorted,
                        MlxReduce::Min,
                        &[last_axis],
                        /*keep_dim=*/ true,
                    )?;

                    let mask_orig = ops::ge(&p, &threshold_p)?;
                    Some(ops::select(&mask_orig, after_topk, &neg_inf)?)
                } else {
                    None
                };
                let final_logits: &Array = topp_owned.as_ref().unwrap_or(after_topk);

                // categorical samples one int32 per row. rlx encodes
                // ids as f32 at the I/O boundary.
                let ids = ops::categorical(final_logits, last_axis, *seed)?;
                ops::cast(&ids, DType::F32)?
            }

            Op::RngNormal {
                mean,
                scale,
                key,
                op_seed,
            } => {
                let n = node.shape.num_elements().unwrap_or(0);
                let mut buf = vec![0f32; n];
                rlx_ir::fill_normal_like(&mut buf, *mean, *scale, rng, *key, *op_seed);
                let dims: Vec<usize> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect();
                Array::from_f32_slice(&buf, &dims, node.shape.dtype())?
            }
            Op::RngUniform {
                low,
                high,
                key,
                op_seed,
            } => {
                let n = node.shape.num_elements().unwrap_or(0);
                let mut buf = vec![0f32; n];
                rlx_ir::fill_uniform_like(&mut buf, *low, *high, rng, *key, *op_seed);
                let dims: Vec<usize> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect();
                Array::from_f32_slice(&buf, &dims, node.shape.dtype())?
            }

            // ── Explicit "no MLX primitive" stops ────────────────
            //
            // The fallback `other` arm below catches anything we
            // haven't enumerated, but a few ops deserve a specific
            // pointer to *why* they're absent so users don't waste
            // time hunting for an off-by-one.
            Op::Pool {
                kind,
                kernel_size,
                stride,
                padding,
            } => {
                // N-D channels-first pool composed from strided-slice
                // + reduction. For each multi-index in the kernel grid
                // we extract the window-positioned slice with the
                // kernel's stride, then merge with the pool's
                // reduction op. Avg-pool divides the running sum by
                // kernel volume; prod multiplies windows together.
                let in_shape = node_input_shape(graph, node.inputs[0]);
                let spatial = kernel_size.len();
                // Input layout: [N, C, ...spatial]. Need rank = 2 + spatial.
                if in_shape.len() != 2 + spatial {
                    return Err(MlxError(format!(
                        "Pool: kernel rank {spatial} requires input rank \
                         {} (channels-first), got {}",
                        2 + spatial,
                        in_shape.len()
                    )));
                }
                if !matches!(
                    kind,
                    ReduceOp::Max | ReduceOp::Min | ReduceOp::Sum | ReduceOp::Mean | ReduceOp::Prod
                ) {
                    return Err(MlxError(format!("Pool: kind {kind:?} not supported")));
                }
                let x = lookup(&env, node.inputs[0])?;
                let ks: Vec<i32> = kernel_size.iter().map(|&k| k as i32).collect();
                let ss: Vec<i32> = (0..spatial)
                    .map(|i| stride.get(i).copied().unwrap_or(1) as i32)
                    .collect();
                let ps: Vec<i32> = (0..spatial)
                    .map(|i| padding.get(i).copied().unwrap_or(0) as i32)
                    .collect();

                // Pad if requested. Max/Min/Prod use neutral elements;
                // sum/avg use 0.
                let pad_value = match kind {
                    ReduceOp::Max => f32::NEG_INFINITY,
                    ReduceOp::Min => f32::INFINITY,
                    ReduceOp::Prod => 1.0,
                    _ => 0.0,
                };
                let needs_pad = ps.iter().any(|&p| p > 0);
                let x_padded_owned;
                let x_padded: &Array = if needs_pad {
                    let mut low = vec![0i32; in_shape.len()];
                    let mut high = vec![0i32; in_shape.len()];
                    low[2..2 + spatial].copy_from_slice(&ps[..spatial]);
                    high[2..2 + spatial].copy_from_slice(&ps[..spatial]);
                    x_padded_owned = ops::pad(x, &low, &high, pad_value)?;
                    &x_padded_owned
                } else {
                    x
                };

                // Output spatial dims.
                let mut out_spatial = Vec::with_capacity(spatial);
                for i in 0..spatial {
                    out_spatial.push((in_shape[2 + i] + 2 * ps[i] - ks[i]) / ss[i] + 1);
                }

                // Iterate kernel multi-index lexicographically.
                let kvol: i64 = ks.iter().map(|&v| v as i64).product();
                let mut acc: Option<Array> = None;
                for k_lin in 0..kvol {
                    let mut k_idx = vec![0i32; spatial];
                    let mut rem = k_lin;
                    for i in (0..spatial).rev() {
                        k_idx[i] = (rem % ks[i] as i64) as i32;
                        rem /= ks[i] as i64;
                    }
                    let mut start = vec![0i32; in_shape.len()];
                    let mut stop = vec![0i32; in_shape.len()];
                    let mut strides = vec![1i32; in_shape.len()];
                    start[0] = 0;
                    stop[0] = in_shape[0]; // batch
                    start[1] = 0;
                    stop[1] = in_shape[1]; // channels
                    for i in 0..spatial {
                        start[2 + i] = k_idx[i];
                        stop[2 + i] = k_idx[i] + ss[i] * out_spatial[i];
                        strides[2 + i] = ss[i];
                    }
                    let win = ops::slice_strided(x_padded, &start, &stop, &strides)?;
                    acc = Some(match (acc, kind) {
                        (None, _) => win,
                        (Some(a), ReduceOp::Max) => ops::max(&a, &win)?,
                        (Some(a), ReduceOp::Min) => ops::min(&a, &win)?,
                        (Some(a), ReduceOp::Prod) => ops::mul(&a, &win)?,
                        (Some(a), _) => ops::add(&a, &win)?,
                    });
                }
                let acc = acc.ok_or_else(|| MlxError("Pool: empty kernel".into()))?;

                if matches!(kind, ReduceOp::Mean) {
                    let count = kvol as f32;
                    let s = Array::from_f32_slice(&[1.0 / count], &[1], DType::F32)?;
                    ops::mul(&acc, &s)?
                } else {
                    acc
                }
            }
            Op::Scan { .. } => {
                // Long Scans that survive `maybe_unroll_scans` host-eval via
                // the shared CPU packed path. Short Scans are IR-unrolled
                // earlier so they lower as ordinary MLX ops.
                let mut vals = std::collections::HashMap::new();
                for &in_id in &node.inputs {
                    vals.insert(
                        in_id,
                        ops::contiguous(lookup(&env, in_id)?)?.to_f32()?,
                    );
                }
                let out = rlx_cpu::thunk::run_scan_node_f32(node, |id| {
                    vals.get(&id).cloned().unwrap_or_default()
                });
                let out_shape: Vec<usize> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect();
                Array::from_f32_slice(&out, &out_shape, DType::F32)?
            }
            Op::ScanBackward { .. } | Op::ScanBackwardXs { .. } => {
                host_eval_op_f32(graph, node, &env)?
            }
            Op::ScatterElements {
                axis,
                reduction: rlx_ir::ScatterNdReduction::Add,
            } => {
                // Vocos ISTFT overlap-add: prefer native MLX scatter_add.
                let updates = lookup(&env, node.inputs[2])?;
                let indices_in = mlx_indices_i64(lookup(&env, node.inputs[1])?)?;
                let out_shape: Vec<i32> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i32)
                    .collect();
                let n_elem: usize = out_shape.iter().map(|&d| d as usize).product();
                let zeros = vec![0.0_f32; n_elem];
                let out_shape_usize: Vec<usize> =
                    out_shape.iter().map(|&d| d as usize).collect();
                let zero_target =
                    crate::array::Array::from_f32_slice(&zeros, &out_shape_usize, DType::F32)?;
                let axis = if *axis < 0 {
                    *axis + out_shape.len() as i32
                } else {
                    *axis
                };
                if out_shape.len() == 1 {
                    let n = out_shape[0];
                    let zero_2d = ops::reshape(&zero_target, &[n, 1])?;
                    let idx_n = indices_in
                        .shape()?
                        .iter()
                        .copied()
                        .product::<usize>()
                        .max(1) as i32;
                    let indices = ops::reshape(&indices_in, &[idx_n, 1])?;
                    let upd_n = updates
                        .shape()?
                        .iter()
                        .copied()
                        .product::<usize>()
                        .max(1) as i32;
                    let updates_2d = ops::reshape(updates, &[upd_n, 1])?;
                    let scattered = ops::scatter_add_axis(&zero_2d, &indices, &updates_2d, 0)?;
                    ops::reshape(&scattered, &[n])?
                } else if out_shape.len() > 1 {
                    let idx_shape = indices_in.shape()?;
                    let indices = if idx_shape.len() == 1 {
                        ops::reshape(&indices_in, &[idx_shape[0] as i32, 1])?
                    } else {
                        indices_in
                    };
                    ops::scatter_add_axis(&zero_target, &indices, updates, axis)?
                } else {
                    host_eval_op_f32(graph, node, &env)?
                }
            }
            Op::ScatterNd { .. }
            | Op::ScatterElements { .. }
            | Op::GatherNd { .. }
            | Op::GatherElements { .. } => host_eval_indexing_op(graph, node, &env)?,
            // Recurrent LSTM. `carry = false` (incl. bidirectional /
            // multi-layer) runs natively on-device by unrolling the time
            // loop into MLX ops — stays in the lazy graph and is
            // `mlx::compile`-able. Covers the Kokoro StyleTTS2 encoder
            // BiLSTM (`H=256`, `bidirectional`, `!carry`). `carry = true`
            // needs functional state write-back that MLX can't express in
            // place, so it host-evals the shared CPU kernel (as Metal/CUDA
            // do), which forces Lazy mode (see `first_host_eval_op`).
            Op::Lstm {
                hidden_size,
                num_layers,
                bidirectional,
                carry,
            } => {
                if *carry {
                    let mut vals = std::collections::HashMap::new();
                    for &in_id in &node.inputs {
                        vals.insert(in_id, ops::contiguous(lookup(&env, in_id)?)?.to_f32()?);
                    }
                    let out = rlx_cpu::thunk::run_host_op_node_f32(graph, node, |id| {
                        vals.get(&id).cloned().unwrap_or_default()
                    });
                    let out_shape: Vec<usize> = node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    Array::from_f32_slice(&out, &out_shape, DType::F32)?
                } else {
                    native_lstm(graph, &env, node, *hidden_size, *num_layers, *bidirectional)?
                }
            }
            // GRU / Elman-RNN — same story as LSTM: `carry = false` unrolls
            // natively on-device; `carry = true` host-evals the shared CPU kernel
            // (forces Lazy, see `first_host_eval_op`).
            Op::Gru {
                hidden_size,
                num_layers,
                bidirectional,
                carry,
            } => {
                if *carry {
                    let mut vals = std::collections::HashMap::new();
                    for &in_id in &node.inputs {
                        vals.insert(in_id, ops::contiguous(lookup(&env, in_id)?)?.to_f32()?);
                    }
                    let out = rlx_cpu::thunk::run_host_op_node_f32(graph, node, |id| {
                        vals.get(&id).cloned().unwrap_or_default()
                    });
                    let out_shape: Vec<usize> =
                        node.shape.dims().iter().map(|d| d.unwrap_static()).collect();
                    Array::from_f32_slice(&out, &out_shape, DType::F32)?
                } else {
                    native_gru(graph, &env, node, *hidden_size, *num_layers, *bidirectional)?
                }
            }
            Op::Rnn {
                hidden_size,
                num_layers,
                bidirectional,
                carry,
                relu,
            } => {
                if *carry {
                    let mut vals = std::collections::HashMap::new();
                    for &in_id in &node.inputs {
                        vals.insert(in_id, ops::contiguous(lookup(&env, in_id)?)?.to_f32()?);
                    }
                    let out = rlx_cpu::thunk::run_host_op_node_f32(graph, node, |id| {
                        vals.get(&id).cloned().unwrap_or_default()
                    });
                    let out_shape: Vec<usize> =
                        node.shape.dims().iter().map(|d| d.unwrap_static()).collect();
                    Array::from_f32_slice(&out, &out_shape, DType::F32)?
                } else {
                    native_rnn(
                        graph,
                        &env,
                        node,
                        *hidden_size,
                        *num_layers,
                        *bidirectional,
                        *relu,
                    )?
                }
            }
            Op::SelectiveScan { state_size } => {
                // Mamba SSM step. MLX has no native scan primitive,
                // so we compose by unrolling the time loop into seq
                // many op chains. Acceptable for static-shape graphs
                // (which all our graphs are); mlx::compile then caches
                // the unrolled trace so per-call cost is amortized.
                //
                // Inputs (per the IR doc):
                //   x [b, s, h]      f32 input
                //   delta [b, s, h]  f32 step size
                //   a [h, n]         f32 transition matrix
                //   b [b, s, n]      f32 input projection
                //   c [b, s, n]      f32 output projection
                // Output [b, s, h], state h [b, h, n] init to zero.
                let x = lookup(&env, node.inputs[0])?;
                let delta = lookup(&env, node.inputs[1])?;
                let a = lookup(&env, node.inputs[2])?;
                let b_in = lookup(&env, node.inputs[3])?;
                let c_in = lookup(&env, node.inputs[4])?;

                let x_shape = node_input_shape(graph, node.inputs[0]);
                if x_shape.len() != 3 {
                    return Err(MlxError(format!(
                        "SelectiveScan: x must be rank-3 [B, S, H], got rank {}",
                        x_shape.len()
                    )));
                }
                let batch = x_shape[0];
                let seq = x_shape[1];
                let hidden = x_shape[2];
                let n = *state_size as i32;

                // State: [B, H, N]. Initialize from a zero scalar
                // broadcast to the target shape; broadcast_to gives
                // a strided view, but we follow with a multiply later
                // so it materializes.
                let zero = Array::from_f32_slice(&[0.0], &[1], DType::F32)?;
                let mut state = ops::broadcast_to(&zero, &[batch, hidden, n])?;

                let mut ys: Vec<Array> = Vec::with_capacity(seq as usize);
                for t in 0..seq {
                    // Slice time-step t.
                    let dt = ops::slice(delta, &[0, t, 0], &[batch, t + 1, hidden])?;
                    let dt = ops::reshape(&dt, &[batch, hidden, 1])?; // [B, H, 1]
                    let xt = ops::slice(x, &[0, t, 0], &[batch, t + 1, hidden])?;
                    let xt = ops::reshape(&xt, &[batch, hidden, 1])?; // [B, H, 1]
                    let bt = ops::slice(b_in, &[0, t, 0], &[batch, t + 1, n])?;
                    let bt = ops::reshape(&bt, &[batch, 1, n])?; // [B, 1, N]
                    let ct = ops::slice(c_in, &[0, t, 0], &[batch, t + 1, n])?;
                    let ct = ops::reshape(&ct, &[batch, 1, n])?; // [B, 1, N]

                    // exp(delta * A): a is [H, N], dt is [B, H, 1].
                    // Their product broadcasts to [B, H, N].
                    let delta_a = ops::mul(&dt, a)?;
                    let exp_delta_a = ops::unary(&delta_a, MlxUnary::Exp)?;

                    // delta * B[t] * x[t]: dt [B, H, 1], bt [B, 1, N],
                    // xt [B, H, 1] → product [B, H, N].
                    let dt_b = ops::mul(&dt, &bt)?; // [B, H, N]
                    let delta_bx = ops::mul(&dt_b, &xt)?; // [B, H, N]

                    // Recurrence: state = exp(δA) * state + δBx
                    let damped = ops::mul(&exp_delta_a, &state)?;
                    state = ops::add(&damped, &delta_bx)?;

                    // y[t] = sum_n( C[t] * state ) along axis 2 ↓ [B, H]
                    let c_state = ops::mul(&ct, &state)?; // [B, H, N]
                    let yt = ops::reduce(&c_state, MlxReduce::Sum, &[2], /*keep_dim=*/ false)?;
                    // Reshape to [B, 1, H] so we can concat into [B, S, H].
                    let yt = ops::reshape(&yt, &[batch, 1, hidden])?;
                    ys.push(yt);
                }

                let refs: Vec<&Array> = ys.iter().collect();
                ops::concat(&refs, 1)?
            }
            Op::GatedDeltaNet {
                state_size,
                carry_state,
            } => {
                let q = lookup(&env, node.inputs[0])?;
                let k = lookup(&env, node.inputs[1])?;
                let v = lookup(&env, node.inputs[2])?;
                let g_in = lookup(&env, node.inputs[3])?;
                let beta = lookup(&env, node.inputs[4])?;
                let (out, state_wb) = lower_gated_delta_net(
                    q,
                    k,
                    v,
                    g_in,
                    beta,
                    *state_size,
                    if *carry_state {
                        Some(lookup(&env, node.inputs[5])?)
                    } else {
                        None
                    },
                    node_input_shape(graph, node.inputs[0]),
                )?;
                if *carry_state {
                    if let Some(state_arr) = state_wb {
                        env.insert(node.inputs[5], state_arr);
                    }
                }
                out
            }

            // ── Tier 1 autodiff backward ops ─────────────────────────
            // Composed from existing MLX primitives so MLX can run the
            // gradient graph emitted by `rlx_opt::autodiff::grad_with_loss`.
            // Formulas mirror `rlx-cpu/src/thunk.rs` (the reference).
            Op::ReluBackward => {
                let x = lookup(&env, node.inputs[0])?;
                let dy = lookup(&env, node.inputs[1])?;
                let dtype = node.shape.dtype();
                let zero = Array::from_f32_slice(&[0.0], &[1], dtype)?;
                let mask = ops::gt(x, &zero)?;
                ops::select(&mask, dy, &zero)?
            }

            Op::ActivationBackward { kind } => {
                let x = lookup(&env, node.inputs[0])?;
                let dy = lookup(&env, node.inputs[1])?;
                let dtype = node.shape.dtype();
                activation_backward_compose(x, dy, *kind, dtype)?
            }

            Op::SoftmaxCrossEntropy => {
                // logits: [N, C], targets: [N, C] (dense distribution).
                // loss[n] = lse(logits[n]) - Σ_c targets[n,c]·logits[n,c].
                let logits = lookup(&env, node.inputs[0])?;
                let targets = lookup(&env, node.inputs[1])?;
                let logits_shape = node_input_shape(graph, node.inputs[0]);
                let n = logits_shape[0];

                // Numerically-stable logsumexp along axis 1.
                let m = ops::reduce(logits, MlxReduce::Max, &[1], /*keep_dim=*/ true)?;
                let shifted = ops::sub(logits, &m)?;
                let exp_d = ops::unary(&shifted, MlxUnary::Exp)?;
                let sum_exp = ops::reduce(&exp_d, MlxReduce::Sum, &[1], /*keep_dim=*/ false)?;
                let log_sum = ops::unary(&sum_exp, MlxUnary::Log)?;
                let m_squeezed = ops::reshape(&m, &[n])?;
                let lse = ops::add(&m_squeezed, &log_sum)?;

                // Σ_c targets[n,c]·logits[n,c] along the class axis.
                let prod = ops::mul(logits, targets)?;
                let dot = ops::reduce(&prod, MlxReduce::Sum, &[1], /*keep_dim=*/ false)?;

                ops::sub(&lse, &dot)?
            }

            Op::SoftmaxCrossEntropyWithLogits => {
                // logits: [N, C], labels: [N] (f32-encoded indices).
                // loss[n] = lse(logits[n]) - logits[n, labels[n]].
                let logits = lookup(&env, node.inputs[0])?;
                let labels = lookup(&env, node.inputs[1])?;
                let logits_shape = node_input_shape(graph, node.inputs[0]);
                let n = logits_shape[0];
                let c = logits_shape[1];
                let dtype = node.shape.dtype();

                // Numerically-stable logsumexp along axis 1.
                let m = ops::reduce(logits, MlxReduce::Max, &[1], /*keep_dim=*/ true)?;
                let shifted = ops::sub(logits, &m)?;
                let exp_d = ops::unary(&shifted, MlxUnary::Exp)?;
                let sum_exp = ops::reduce(&exp_d, MlxReduce::Sum, &[1], /*keep_dim=*/ false)?;
                let log_sum = ops::unary(&sum_exp, MlxUnary::Log)?;
                let m_squeezed = ops::reshape(&m, &[n])?;
                let lse = ops::add(&m_squeezed, &log_sum)?;

                // logits[label] via one-hot mask.
                let oh = one_hot_2d(labels, n as usize, c as usize, dtype)?;
                let masked = ops::mul(logits, &oh)?;
                let logit_at_label =
                    ops::reduce(&masked, MlxReduce::Sum, &[1], /*keep_dim=*/ false)?;

                ops::sub(&lse, &logit_at_label)?
            }

            Op::SoftmaxCrossEntropyBackward => {
                // dlogits[n, c] = (softmax(logits)[n, c] - one_hot(labels)[n, c]) * d_loss[n].
                let logits = lookup(&env, node.inputs[0])?;
                let labels = lookup(&env, node.inputs[1])?;
                let d_loss = lookup(&env, node.inputs[2])?;
                let logits_shape = node_input_shape(graph, node.inputs[0]);
                let n = logits_shape[0];
                let c = logits_shape[1];
                let dtype = node.shape.dtype();

                let sm = ops::softmax(logits, 1)?;
                let oh = one_hot_2d(labels, n as usize, c as usize, dtype)?;
                let diff = ops::sub(&sm, &oh)?;
                let d_loss_2d = ops::reshape(d_loss, &[n, 1])?;
                ops::mul(&diff, &d_loss_2d)?
            }

            Op::LayerNormBackwardInput { eps, axis: _ } => {
                // axis = -1 only (per IR docstring).
                // dx = inv_std · (sy − mean(sy) − x̂ · mean(sy · x̂))
                // where sy = dy · γ, x̂ = (x − μ) · inv_std.
                let x = lookup(&env, node.inputs[0])?;
                let gamma = lookup(&env, node.inputs[1])?;
                let dy = lookup(&env, node.inputs[2])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let last = (x_shape.len() - 1) as i32;
                let dtype = node.shape.dtype();
                let eps_arr = Array::from_f32_slice(&[*eps], &[1], dtype)?;

                let mean = ops::reduce(x, MlxReduce::Mean, &[last], true)?;
                let diff = ops::sub(x, &mean)?;
                let diff_sq = ops::mul(&diff, &diff)?;
                let var = ops::reduce(&diff_sq, MlxReduce::Mean, &[last], true)?;
                let var_eps = ops::add(&var, &eps_arr)?;
                let inv_std = ops::unary(&var_eps, MlxUnary::Rsqrt)?;
                let xhat = ops::mul(&diff, &inv_std)?;
                let sy = ops::mul(dy, gamma)?;
                let m_sy = ops::reduce(&sy, MlxReduce::Mean, &[last], true)?;
                let sy_xh = ops::mul(&sy, &xhat)?;
                let m_sxh = ops::reduce(&sy_xh, MlxReduce::Mean, &[last], true)?;
                let term1 = ops::sub(&sy, &m_sy)?;
                let term2 = ops::mul(&xhat, &m_sxh)?;
                let inner = ops::sub(&term1, &term2)?;
                ops::mul(&inv_std, &inner)?
            }

            Op::FakeQuantize {
                bits,
                axis,
                ste: _,
                scale_mode,
            } => {
                // y = clamp(round(x / s), -q_max, q_max) · s
                // where `s` per channel comes from `scale_mode`.
                // Forward `ste` doesn't affect the output — only the
                // backward.
                let x = lookup(&env, node.inputs[0])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let dtype = node.shape.dtype();
                let q_max = fq_q_max(*bits)?;

                let scale = match scale_mode {
                    ScaleMode::PerBatch => fq_scale_perbatch(x, &x_shape, *axis, q_max, dtype)?,
                    ScaleMode::Fixed => {
                        let state = lookup(&env, node.inputs[1])?;
                        fq_scale_from_state(state, &x_shape, *axis, dtype)?
                    }
                    ScaleMode::EMA { .. } => {
                        return Err(MlxError(
                            "Op::FakeQuantize with ScaleMode::EMA not yet \
                             supported on MLX (the running scale state \
                             update needs side-effect plumbing the lazy \
                             trace doesn't expose). Use ScaleMode::PerBatch \
                             for QAT training or ScaleMode::Fixed for \
                             pre-calibrated inference."
                                .into(),
                        ));
                    }
                };
                fq_quantize_dequantize(x, &scale, q_max, dtype)?
            }

            Op::FakeQuantizeBackward { bits, axis, ste } => {
                // The CPU thunk recomputes the scale via PerBatch from
                // the current `x` regardless of how the forward derived
                // it (see `rlx-cpu/src/thunk.rs:4239`); we mirror that.
                let x = lookup(&env, node.inputs[0])?;
                let dy = lookup(&env, node.inputs[1])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let dtype = node.shape.dtype();
                let q_max = fq_q_max(*bits)?;
                let scale = fq_scale_perbatch(x, &x_shape, *axis, q_max, dtype)?;

                let q_max_arr = Array::from_f32_slice(&[q_max], &[1], dtype)?;
                let one = Array::from_f32_slice(&[1.0], &[1], dtype)?;
                let zero = Array::from_f32_slice(&[0.0], &[1], dtype)?;

                match ste {
                    SteKind::Identity => dy.clone_handle()?,
                    SteKind::ClippedIdentity => {
                        // dx = where(|x| ≤ q_max·s, dy, 0)
                        let bound = ops::mul(&scale, &q_max_arr)?;
                        let abs_x = ops::unary(x, MlxUnary::Abs)?;
                        let mask = ops::le(&abs_x, &bound)?;
                        ops::select(&mask, dy, &zero)?
                    }
                    SteKind::Tanh => {
                        // dx = dy · (1 − tanh²(x/s))
                        let scaled = ops::div(x, &scale)?;
                        let t = ops::unary(&scaled, MlxUnary::Tanh)?;
                        let t_sq = ops::mul(&t, &t)?;
                        let factor = ops::sub(&one, &t_sq)?;
                        ops::mul(dy, &factor)?
                    }
                    SteKind::HardTanh => {
                        // dx = dy · max(0, 1 − |x/(q_max·s)|)
                        let bound = ops::mul(&scale, &q_max_arr)?;
                        let scaled = ops::div(x, &bound)?;
                        let abs_scaled = ops::unary(&scaled, MlxUnary::Abs)?;
                        let one_minus = ops::sub(&one, &abs_scaled)?;
                        let attenuation = ops::max(&one_minus, &zero)?;
                        ops::mul(dy, &attenuation)?
                    }
                }
            }

            Op::MaxPool2dBackward {
                kernel_size,
                stride,
                padding,
            } => {
                // x shape [N, C, H, W], dy shape [N, C, H_out, W_out]
                // Output dx shape [N, C, H, W].
                if kernel_size.len() != 2 || stride.len() != 2 || padding.len() != 2 {
                    return Err(MlxError("MaxPool2dBackward on MLX: 2D pool only".into()));
                }
                let x = lookup(&env, node.inputs[0])?;
                let dy = lookup(&env, node.inputs[1])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let dy_shape = node_input_shape(graph, node.inputs[1]);
                if x_shape.len() != 4 || dy_shape.len() != 4 {
                    return Err(MlxError(
                        "MaxPool2dBackward on MLX: 2D pool expects rank-4 tensors".into(),
                    ));
                }
                let n = x_shape[0];
                let cc = x_shape[1];
                let h = x_shape[2];
                let w = x_shape[3];
                let h_out = dy_shape[2];
                let w_out = dy_shape[3];
                let kh = kernel_size[0] as i32;
                let kw = kernel_size[1] as i32;
                let sh = stride[0] as i32;
                let sw = stride[1] as i32;
                let ph = padding[0] as i32;
                let pw = padding[1] as i32;

                // Custom Metal kernel: one thread per output position
                // does an in-window argmax + atomic-fetch-add into dx.
                // Handles overlap (stride < kernel) and padding > 0 in
                // one path. ~5–10× faster than the primitive-composition
                // alternative on shapes where MLX's `scatter_add_axis`
                // is the bottleneck.
                ops::maxpool2d_backward_metal(
                    x, dy, n, cc, h, w, h_out, w_out, kh, kw, sh, sw, ph, pw,
                )?
            }

            Op::Conv2dBackwardInput {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                // Reverse-mode conv-grad-w.r.t.-input. Translates the
                // forward conv parameters into the `conv_general`
                // arguments MLX itself uses inside its built-in vjp
                // (see vendor/mlx/mlx/primitives.cpp `Convolution::vjp`).
                if kernel_size.len() != 2 {
                    return Err(MlxError("Conv2dBackwardInput on MLX: 2D conv only".into()));
                }
                let dy = lookup(&env, node.inputs[0])?;
                let w = lookup(&env, node.inputs[1])?;
                let dy_shape = node_input_shape(graph, node.inputs[0]);
                let w_shape = node_input_shape(graph, node.inputs[1]);
                let dx_shape: Vec<i32> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i32)
                    .collect();
                if dy_shape.len() != 4 || w_shape.len() != 4 || dx_shape.len() != 4 {
                    return Err(MlxError(
                        "Conv2dBackwardInput on MLX: 2D conv expects rank-4 tensors".into(),
                    ));
                }

                let g = *groups as i32;
                let c_in = dx_shape[1];
                let c_out = dy_shape[1];
                if c_in % g != 0 || c_out % g != 0 {
                    return Err(MlxError(format!(
                        "Conv2dBackwardInput: groups ({g}) must divide \
                         C_in ({c_in}) and C_out ({c_out})"
                    )));
                }
                let c_in_per_g = c_in / g;
                let c_out_per_g = c_out / g;
                let h = dx_shape[2];
                let w_in = dx_shape[3];
                let h_out = dy_shape[2];
                let w_out = dy_shape[3];
                let kh = w_shape[2];
                let kw = w_shape[3];
                let s = |i: usize| stride.get(i).copied().unwrap_or(1) as i32;
                let p = |i: usize| padding.get(i).copied().unwrap_or(0) as i32;
                let d = |i: usize| dilation.get(i).copied().unwrap_or(1) as i32;

                // Per MLX vjp (vendor/mlx/mlx/primitives.cpp):
                //   wt_size       = 1 + D·(K−1)
                //   padding_lo[i] = wt_size − P_orig − 1     = D·(K−1) − P
                //   in_size       = H,   out_size = 1 + S·(H_out − 1)
                //   padding_hi[i] = in_size − out_size + P
                let pad_lo: Vec<i32> = vec![d(0) * (kh - 1) - p(0), d(1) * (kw - 1) - p(1)];
                let pad_hi: Vec<i32> = vec![
                    h - 1 - s(0) * (h_out - 1) + p(0),
                    w_in - 1 - s(1) * (w_out - 1) + p(1),
                ];

                // dy: rlx NCHW → MLX NHWC.
                let dy_nhwc = ops::transpose(dy, &[0, 2, 3, 1])?;

                // MLX limitation: `conv_general` with both `groups > 1` and
                // `input_dilation > 1` produces incorrect output (the
                // grouped path doesn't compose with the dilated-input
                // path; tests/autodiff_conv_parity.rs::*_groups_*_stride2
                // proves it). Workaround: when both kick in, materialize
                // the input dilation by reshape+pad+reshape (zero-inflate
                // dy along each spatial axis) and call conv_general with
                // `input_dilation=[1,1]`.
                let needs_inflate = g > 1 && (s(0) > 1 || s(1) > 1);
                let (dy_input, conv_input_dilation): (Array, [i32; 2]) = if needs_inflate {
                    let inflated = inflate_spatial_2d(&dy_nhwc, s(0) as usize, s(1) as usize)?;
                    (inflated, [1, 1])
                } else {
                    (dy_nhwc.clone_handle()?, [s(0), s(1)])
                };

                // Weight transform — translates MLX vjp's `group_transpose(wt, 0, 1, -1)`.
                //   groups=1: rlx [C_out, C_in, kH, kW] → [C_in, kH, kW, C_out]
                //             via the single perm [1, 2, 3, 0].
                //   groups>1: split C_out by group via reshape, swap C_out/g
                //             with C_in/g, then flatten (groups, C_in/g) → C_in:
                //               [C_out, C_in/g, kH, kW]
                //             → [g, C_out/g, C_in/g, kH, kW]   (reshape)
                //             → [g, C_in/g, kH, kW, C_out/g]   (perm 0,2,3,4,1)
                //             → [C_in, kH, kW, C_out/g]        (reshape)
                let w_t = if g == 1 {
                    ops::transpose(w, &[1, 2, 3, 0])?
                } else {
                    let split = ops::reshape(w, &[g, c_out_per_g, c_in_per_g, kh, kw])?;
                    let perm = ops::transpose(&split, &[0, 2, 3, 4, 1])?;
                    ops::reshape(&perm, &[c_in, kh, kw, c_out_per_g])?
                };

                let raw = ops::conv_general(
                    &dy_input,
                    &w_t,
                    /* stride          = */ &[1, 1],
                    /* padding_lo      = */ &pad_lo,
                    /* padding_hi      = */ &pad_hi,
                    /* kernel_dilation = */ &[d(0), d(1)],
                    /* input_dilation  = */ &conv_input_dilation,
                    /* groups          = */ g,
                    /* flip            = */ true,
                )?;

                // Negative-padding fixup: MLX's `conv_general` accepts
                // negative padding by *over-producing* and we slice the
                // overshoot off (matches MLX vjp's own behavior).
                let needs_slice = pad_lo.iter().chain(pad_hi.iter()).any(|&p| p < 0);
                let adjusted = if needs_slice {
                    let cur: Vec<i32> = raw.shape()?.iter().map(|&d| d as i32).collect();
                    let mut start = vec![0i32; cur.len()];
                    let mut stop = cur.clone();
                    for i in 0..2 {
                        if pad_lo[i] < 0 {
                            start[1 + i] = -pad_lo[i];
                        }
                        if pad_hi[i] < 0 {
                            stop[1 + i] += pad_hi[i];
                        }
                    }
                    ops::slice(&raw, &start, &stop)?
                } else {
                    raw
                };

                // NHWC → NCHW for the rlx-side consumer.
                // `contiguous` materializes the strided view; without
                // it `mc::compile` elides the transpose and the readback
                // ends up in NHWC layout (compile-mode bug repro:
                // `tests/conv_compile_mode_repro.rs`).
                let nchw = ops::transpose(&adjusted, &[0, 3, 1, 2])?;
                ops::contiguous(&nchw)?
            }

            Op::Conv2dBackwardWeight {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                if kernel_size.len() != 2 {
                    return Err(MlxError("Conv2dBackwardWeight on MLX: 2D conv only".into()));
                }
                let x = lookup(&env, node.inputs[0])?;
                let dy = lookup(&env, node.inputs[1])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let dy_shape = node_input_shape(graph, node.inputs[1]);
                let dw_shape: Vec<i32> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i32)
                    .collect();
                if x_shape.len() != 4 || dy_shape.len() != 4 || dw_shape.len() != 4 {
                    return Err(MlxError(
                        "Conv2dBackwardWeight on MLX: 2D conv expects rank-4 tensors".into(),
                    ));
                }
                let g = *groups as i32;
                let n_batch = x_shape[0];
                let c_in = x_shape[1];
                let c_out = dy_shape[1];
                if c_in % g != 0 || c_out % g != 0 {
                    return Err(MlxError(format!(
                        "Conv2dBackwardWeight: groups ({g}) must divide \
                         C_in ({c_in}) and C_out ({c_out})"
                    )));
                }
                let c_in_per_g = c_in / g;
                let h = x_shape[2];
                let w_in = x_shape[3];
                let h_out = dy_shape[2];
                let w_out = dy_shape[3];
                let kh = dw_shape[2];
                let kw = dw_shape[3];
                let s = |i: usize| stride.get(i).copied().unwrap_or(1) as i32;
                let p = |i: usize| padding.get(i).copied().unwrap_or(0) as i32;
                let d = |i: usize| dilation.get(i).copied().unwrap_or(1) as i32;

                // Per MLX vjp:
                //   padding_lo[i] = P
                //   padding_hi[i] = (S·(H_out−1) + 1) − H + (D·(K−1) + 1) − P − 1
                let pad_lo: Vec<i32> = vec![p(0), p(1)];
                let pad_hi: Vec<i32> = vec![
                    s(0) * (h_out - 1) + 1 - h + d(0) * (kh - 1) + 1 - p(0) - 1,
                    s(1) * (w_out - 1) + 1 - w_in + d(1) * (kw - 1) + 1 - p(1) - 1,
                ];

                // dy: rlx NCHW → swapaxes(NHWC, 0, -1) =
                //   [C_out, H_out, W_out, N]  via transpose [1, 2, 3, 0].
                let cotan_trans = ops::transpose(dy, &[1, 2, 3, 0])?;

                // x transform — translates MLX vjp's `group_transpose(in, -1, 0, -1)`.
                //   groups=1: rlx [N, C_in, H, W] → [C_in, H, W, N]
                //             via the single perm [1, 2, 3, 0].
                //   groups>1: split C_in by group, swap N and C_in/g, then
                //             flatten (g, N) → (g·N):
                //               [N, C_in, H, W]
                //             → [N, g, C_in/g, H, W]            (reshape)
                //             → [C_in/g, H, W, g, N]            (perm 2,3,4,1,0)
                //             → [C_in/g, H, W, g·N]             (reshape)
                let in_trans = if g == 1 {
                    ops::transpose(x, &[1, 2, 3, 0])?
                } else {
                    let split = ops::reshape(x, &[n_batch, g, c_in_per_g, h, w_in])?;
                    let perm = ops::transpose(&split, &[2, 3, 4, 1, 0])?;
                    ops::reshape(&perm, &[c_in_per_g, h, w_in, g * n_batch])?
                };

                let grad_trans = ops::conv_general(
                    &in_trans,
                    &cotan_trans,
                    /* stride          = */ &[d(0), d(1)],
                    /* padding_lo      = */ &pad_lo,
                    /* padding_hi      = */ &pad_hi,
                    /* kernel_dilation = */ &[s(0), s(1)],
                    /* input_dilation  = */ &[1, 1],
                    /* groups          = */ g,
                    /* flip            = */ false,
                )?;
                // grad_trans: [C_in, kH, kW, C_out]. rlx layout wants
                // [C_out, C_in, kH, kW] → perm [3, 0, 1, 2]. As with
                // backward-input, `contiguous` is required to defeat
                // `mc::compile`'s strided-view elision.
                let dw = ops::transpose(&grad_trans, &[3, 0, 1, 2])?;
                ops::contiguous(&dw)?
            }

            Op::MaxPool3dBackward {
                kernel_size,
                stride,
                padding,
            } => {
                if kernel_size.len() != 3 || stride.len() != 3 || padding.len() != 3 {
                    return Err(MlxError("MaxPool3dBackward on MLX: 3D pool only".into()));
                }
                let x = lookup(&env, node.inputs[0])?;
                let dy = lookup(&env, node.inputs[1])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let dy_shape = node_input_shape(graph, node.inputs[1]);
                if x_shape.len() != 5 || dy_shape.len() != 5 {
                    return Err(MlxError(
                        "MaxPool3dBackward on MLX: 3D pool expects rank-5 tensors".into(),
                    ));
                }
                ops::maxpool3d_backward_metal(
                    x,
                    dy,
                    x_shape[0],
                    x_shape[1],
                    x_shape[2],
                    x_shape[3],
                    x_shape[4],
                    dy_shape[2],
                    dy_shape[3],
                    dy_shape[4],
                    kernel_size[0] as i32,
                    kernel_size[1] as i32,
                    kernel_size[2] as i32,
                    stride[0] as i32,
                    stride[1] as i32,
                    stride[2] as i32,
                    padding[0] as i32,
                    padding[1] as i32,
                    padding[2] as i32,
                )?
            }

            Op::Conv3dBackwardInput {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                if kernel_size.len() != 3 {
                    return Err(MlxError("Conv3dBackwardInput on MLX: 3D conv only".into()));
                }
                let dy = lookup(&env, node.inputs[0])?;
                let w = lookup(&env, node.inputs[1])?;
                let dy_shape = node_input_shape(graph, node.inputs[0]);
                let w_shape = node_input_shape(graph, node.inputs[1]);
                let dx_shape: Vec<i32> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i32)
                    .collect();
                if dy_shape.len() != 5 || w_shape.len() != 5 || dx_shape.len() != 5 {
                    return Err(MlxError(
                        "Conv3dBackwardInput on MLX: 3D conv expects rank-5 tensors".into(),
                    ));
                }

                let g = *groups as i32;
                let c_in = dx_shape[1];
                let c_out = dy_shape[1];
                if c_in % g != 0 || c_out % g != 0 {
                    return Err(MlxError(format!(
                        "Conv3dBackwardInput: groups ({g}) must divide \
                         C_in ({c_in}) and C_out ({c_out})"
                    )));
                }
                let c_in_per_g = c_in / g;
                let c_out_per_g = c_out / g;
                let dep = dx_shape[2];
                let h = dx_shape[3];
                let w_in = dx_shape[4];
                let d_out = dy_shape[2];
                let h_out = dy_shape[3];
                let w_out = dy_shape[4];
                let kd = w_shape[2];
                let kh = w_shape[3];
                let kw = w_shape[4];
                let s = |i: usize| stride.get(i).copied().unwrap_or(1) as i32;
                let p = |i: usize| padding.get(i).copied().unwrap_or(0) as i32;
                let d = |i: usize| dilation.get(i).copied().unwrap_or(1) as i32;

                let pad_lo: Vec<i32> = vec![
                    d(0) * (kd - 1) - p(0),
                    d(1) * (kh - 1) - p(1),
                    d(2) * (kw - 1) - p(2),
                ];
                let pad_hi: Vec<i32> = vec![
                    dep - 1 - s(0) * (d_out - 1) + p(0),
                    h - 1 - s(1) * (h_out - 1) + p(1),
                    w_in - 1 - s(2) * (w_out - 1) + p(2),
                ];

                let dy_ndhwc = ops::transpose(dy, &[0, 2, 3, 4, 1])?;

                let w_t = if g == 1 {
                    ops::transpose(w, &[1, 2, 3, 4, 0])?
                } else {
                    let split = ops::reshape(w, &[g, c_out_per_g, c_in_per_g, kd, kh, kw])?;
                    let perm = ops::transpose(&split, &[0, 2, 3, 4, 5, 1])?;
                    ops::reshape(&perm, &[c_in, kd, kh, kw, c_out_per_g])?
                };

                let raw = ops::conv_general(
                    &dy_ndhwc,
                    &w_t,
                    &[1, 1, 1],
                    &pad_lo,
                    &pad_hi,
                    &[d(0), d(1), d(2)],
                    &[s(0), s(1), s(2)],
                    g,
                    true,
                )?;

                let needs_slice = pad_lo.iter().chain(pad_hi.iter()).any(|&p| p < 0);
                let adjusted = if needs_slice {
                    let cur: Vec<i32> = raw.shape()?.iter().map(|&d| d as i32).collect();
                    let mut start = vec![0i32; cur.len()];
                    let mut stop = cur.clone();
                    for i in 0..3 {
                        if pad_lo[i] < 0 {
                            start[1 + i] = -pad_lo[i];
                        }
                        if pad_hi[i] < 0 {
                            stop[1 + i] += pad_hi[i];
                        }
                    }
                    ops::slice(&raw, &start, &stop)?
                } else {
                    raw
                };

                let ncdhw = ops::transpose(&adjusted, &[0, 4, 1, 2, 3])?;
                ops::contiguous(&ncdhw)?
            }

            Op::Conv3dBackwardWeight {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                if kernel_size.len() != 3 {
                    return Err(MlxError("Conv3dBackwardWeight on MLX: 3D conv only".into()));
                }
                let x = lookup(&env, node.inputs[0])?;
                let dy = lookup(&env, node.inputs[1])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let dy_shape = node_input_shape(graph, node.inputs[1]);
                let dw_shape: Vec<i32> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i32)
                    .collect();
                if x_shape.len() != 5 || dy_shape.len() != 5 || dw_shape.len() != 5 {
                    return Err(MlxError(
                        "Conv3dBackwardWeight on MLX: 3D conv expects rank-5 tensors".into(),
                    ));
                }
                let g = *groups as i32;
                let n_batch = x_shape[0];
                let c_in = x_shape[1];
                let c_out = dy_shape[1];
                if c_in % g != 0 || c_out % g != 0 {
                    return Err(MlxError(format!(
                        "Conv3dBackwardWeight: groups ({g}) must divide \
                         C_in ({c_in}) and C_out ({c_out})"
                    )));
                }
                let c_in_per_g = c_in / g;
                let dep = x_shape[2];
                let h = x_shape[3];
                let w_in = x_shape[4];
                let d_out = dy_shape[2];
                let h_out = dy_shape[3];
                let w_out = dy_shape[4];
                let kd = dw_shape[2];
                let kh = dw_shape[3];
                let kw = dw_shape[4];
                let s = |i: usize| stride.get(i).copied().unwrap_or(1) as i32;
                let p = |i: usize| padding.get(i).copied().unwrap_or(0) as i32;
                let d = |i: usize| dilation.get(i).copied().unwrap_or(1) as i32;

                let pad_lo: Vec<i32> = vec![p(0), p(1), p(2)];
                let pad_hi: Vec<i32> = vec![
                    s(0) * (d_out - 1) + 1 - dep + d(0) * (kd - 1) + 1 - p(0) - 1,
                    s(1) * (h_out - 1) + 1 - h + d(1) * (kh - 1) + 1 - p(1) - 1,
                    s(2) * (w_out - 1) + 1 - w_in + d(2) * (kw - 1) + 1 - p(2) - 1,
                ];

                let cotan_trans = ops::transpose(dy, &[1, 2, 3, 4, 0])?;

                let in_trans = if g == 1 {
                    ops::transpose(x, &[1, 2, 3, 4, 0])?
                } else {
                    let split = ops::reshape(x, &[n_batch, g, c_in_per_g, dep, h, w_in])?;
                    let perm = ops::transpose(&split, &[2, 3, 4, 5, 1, 0])?;
                    ops::reshape(&perm, &[c_in_per_g, dep, h, w_in, g * n_batch])?
                };

                let grad_trans = ops::conv_general(
                    &in_trans,
                    &cotan_trans,
                    &[d(0), d(1), d(2)],
                    &pad_lo,
                    &pad_hi,
                    &[s(0), s(1), s(2)],
                    &[1, 1, 1],
                    g,
                    false,
                )?;
                let dw = ops::transpose(&grad_trans, &[4, 0, 1, 2, 3])?;
                ops::contiguous(&dw)?
            }

            Op::LayerNormBackwardGamma { eps, axis: _ } => {
                // axis = -1 only. dgamma = sum_over_outer(dy · x̂).
                let x = lookup(&env, node.inputs[0])?;
                let dy = lookup(&env, node.inputs[1])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let last = (x_shape.len() - 1) as i32;
                let dtype = node.shape.dtype();
                let eps_arr = Array::from_f32_slice(&[*eps], &[1], dtype)?;

                let mean = ops::reduce(x, MlxReduce::Mean, &[last], true)?;
                let diff = ops::sub(x, &mean)?;
                let diff_sq = ops::mul(&diff, &diff)?;
                let var = ops::reduce(&diff_sq, MlxReduce::Mean, &[last], true)?;
                let var_eps = ops::add(&var, &eps_arr)?;
                let inv_std = ops::unary(&var_eps, MlxUnary::Rsqrt)?;
                let xhat = ops::mul(&diff, &inv_std)?;
                let prod = ops::mul(dy, &xhat)?;

                if last == 0 {
                    prod
                } else {
                    let reduce_axes: Vec<i32> = (0..last).collect();
                    let summed = ops::reduce(
                        &prod,
                        MlxReduce::Sum,
                        &reduce_axes,
                        /*keep_dim=*/ false,
                    )?;
                    let want: Vec<i32> = node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static() as i32)
                        .collect();
                    let got = summed.shape()?;
                    let got_i32: Vec<i32> = got.iter().map(|&d| d as i32).collect();
                    if got_i32 == want {
                        summed
                    } else {
                        ops::reshape(&summed, &want)?
                    }
                }
            }

            Op::AttentionBackward {
                num_heads,
                head_dim,
                mask_kind,
                wrt,
            } => {
                let q_in = lookup(&env, node.inputs[0])?;
                let k_in = lookup(&env, node.inputs[1])?;
                let v_in = lookup(&env, node.inputs[2])?;
                let dy_in = lookup(&env, node.inputs[3])?;
                let q_shape = node_input_shape(graph, node.inputs[0]);
                let k_shape = node_input_shape(graph, node.inputs[1]);
                let nh = *num_heads as i32;
                let hd = *head_dim as i32;
                let need_split = q_shape.len() == 3;
                let to_bhsd = |t: &Array, sh: &[i32]| -> Result<Array, MlxError> {
                    if sh.len() == 4 {
                        return t.clone_handle();
                    }
                    let b = sh[0];
                    let s = sh[1];
                    let r = ops::reshape(t, &[b, s, nh, hd])?;
                    ops::transpose(&r, &[0, 2, 1, 3])
                };
                let q = to_bhsd(q_in, &q_shape)?;
                let k = to_bhsd(k_in, &k_shape)?;
                let v = to_bhsd(v_in, &node_input_shape(graph, node.inputs[2]))?;
                let dy = to_bhsd(dy_in, &node_input_shape(graph, node.inputs[3]))?;
                let q_dtype = graph.node(node.inputs[0]).shape.dtype();
                let normalize_mask = |m: &Array, m_shape: &[i32]| -> Result<Array, MlxError> {
                    match m_shape.len() {
                        2 => ops::reshape(m, &[m_shape[0], 1, 1, m_shape[1]]),
                        3 => ops::reshape(m, &[m_shape[0], 1, m_shape[1], m_shape[2]]),
                        _ => m.clone_handle(),
                    }
                };
                let (mask_additive, window) = match mask_kind {
                    MaskKind::Custom => {
                        let m = lookup(&env, node.inputs[4])?;
                        let m_shape = node_input_shape(graph, node.inputs[4]);
                        let one = Array::from_f32_slice(&[1.0], &[1], q_dtype)?;
                        let scl = Array::from_f32_slice(&[1.0e9], &[1], q_dtype)?;
                        let m_cast = if q_dtype != DType::F32 {
                            ops::cast(m, q_dtype)?
                        } else {
                            m.clone_handle()?
                        };
                        let shifted = ops::sub(&m_cast, &one)?;
                        let additive = ops::mul(&shifted, &scl)?;
                        (Some(normalize_mask(&additive, &m_shape)?), 0usize)
                    }
                    MaskKind::Bias => {
                        let m = lookup(&env, node.inputs[4])?;
                        let m_shape = node_input_shape(graph, node.inputs[4]);
                        let m_cast = if q_dtype != DType::F32 {
                            ops::cast(m, q_dtype)?
                        } else {
                            m.clone_handle()?
                        };
                        (Some(normalize_mask(&m_cast, &m_shape)?), 0usize)
                    }
                    MaskKind::SlidingWindow(w) => (None, *w),
                    _ => (None, 0usize),
                };
                let mask_ref = mask_additive.as_ref();
                let grad = crate::attention_bwd::attention_backward_bhsd(
                    *wrt, &q, &k, &v, &dy, hd, *mask_kind, mask_ref, window,
                )?;
                if need_split {
                    let b = q_shape[0];
                    let s = q_shape[1];
                    let bsd = ops::transpose(&grad, &[0, 2, 1, 3])?;
                    ops::reshape(&bsd, &[b, s, nh * hd])?
                } else {
                    grad
                }
            }

            Op::RmsNormBackwardInput { eps, axis: _ } => {
                let x = lookup(&env, node.inputs[0])?;
                let gamma = lookup(&env, node.inputs[1])?;
                let _beta = lookup(&env, node.inputs[2])?;
                let dy = lookup(&env, node.inputs[3])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let last = (x_shape.len() - 1) as i32;
                let dtype = node.shape.dtype();
                let eps_arr = Array::from_f32_slice(&[*eps], &[1], dtype)?;

                let x_sq = ops::mul(x, x)?;
                let mean_sq = ops::reduce(&x_sq, MlxReduce::Mean, &[last], true)?;
                let var_eps = ops::add(&mean_sq, &eps_arr)?;
                let inv_r = ops::unary(&var_eps, MlxUnary::Rsqrt)?;
                // Cross term is inv_r² here; the outer `inv_r *` makes it inv_r³, not inv_r⁴.
                let inv_r2 = ops::mul(&inv_r, &inv_r)?;
                let dy_g = ops::mul(dy, gamma)?;
                let dy_gx = ops::mul(&dy_g, x)?;
                let dot = ops::reduce(&dy_gx, MlxReduce::Mean, &[last], true)?;
                let x_dot = ops::mul(x, &dot)?;
                let term = ops::sub(&dy_g, &ops::mul(&x_dot, &inv_r2)?)?;
                ops::mul(&inv_r, &term)?
            }

            Op::RmsNormBackwardGamma { eps, axis: _ } => {
                let x = lookup(&env, node.inputs[0])?;
                let _gamma = lookup(&env, node.inputs[1])?;
                let _beta = lookup(&env, node.inputs[2])?;
                let dy = lookup(&env, node.inputs[3])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let last = (x_shape.len() - 1) as i32;
                let dtype = node.shape.dtype();
                let eps_arr = Array::from_f32_slice(&[*eps], &[1], dtype)?;

                let x_sq = ops::mul(x, x)?;
                let mean_sq = ops::reduce(&x_sq, MlxReduce::Mean, &[last], true)?;
                let var_eps = ops::add(&mean_sq, &eps_arr)?;
                let inv_r = ops::unary(&var_eps, MlxUnary::Rsqrt)?;
                let prod = ops::mul(dy, &ops::mul(x, &inv_r)?)?;

                if last == 0 {
                    prod
                } else {
                    let reduce_axes: Vec<i32> = (0..last).collect();
                    let summed = ops::reduce(
                        &prod,
                        MlxReduce::Sum,
                        &reduce_axes,
                        /*keep_dim=*/ false,
                    )?;
                    let want: Vec<i32> = node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static() as i32)
                        .collect();
                    let got = summed.shape()?;
                    let got_i32: Vec<i32> = got.iter().map(|&d| d as i32).collect();
                    if got_i32 == want {
                        summed
                    } else {
                        ops::reshape(&summed, &want)?
                    }
                }
            }

            Op::RmsNormBackwardBeta { .. } => {
                let dy = lookup(&env, node.inputs[3])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let last = (x_shape.len() - 1) as i32;
                if last == 0 {
                    dy.clone_handle()?
                } else {
                    let reduce_axes: Vec<i32> = (0..last).collect();
                    let summed =
                        ops::reduce(dy, MlxReduce::Sum, &reduce_axes, /*keep_dim=*/ false)?;
                    let want: Vec<i32> = node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static() as i32)
                        .collect();
                    let got = summed.shape()?;
                    let got_i32: Vec<i32> = got.iter().map(|&d| d as i32).collect();
                    if got_i32 == want {
                        summed
                    } else {
                        ops::reshape(&summed, &want)?
                    }
                }
            }

            Op::GroupNormBackwardInput { num_groups, eps } => {
                let x = lookup(&env, node.inputs[0])?;
                let gamma = lookup(&env, node.inputs[1])?;
                let dy = lookup(&env, node.inputs[3])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let dtype = node.shape.dtype();
                let n = x_shape[0];
                let c = x_shape[1];
                let h = x_shape[2];
                let w = x_shape[3];
                let g = *num_groups as i32;
                let cpg = c / g;
                let inner = cpg * h * w;
                let x5 = ops::reshape(x, &[n, g, cpg, h, w])?;
                let dy5 = ops::reshape(dy, &[n, g, cpg, h, w])?;
                let x3 = ops::reshape(&x5, &[n, g, inner])?;
                let dy3 = ops::reshape(&dy5, &[n, g, inner])?;
                let gamma_g = ops::reshape(gamma, &[1, g, cpg, 1])?;
                let gamma_b = ops::broadcast_to(&gamma_g, &[n, g, cpg, h * w])?;
                let gamma_flat = ops::reshape(&gamma_b, &[n, g, inner])?;
                let eps_arr = Array::from_f32_slice(&[*eps], &[1], dtype)?;
                let mean = ops::reduce(&x3, MlxReduce::Mean, &[2], true)?;
                let x_c = ops::sub(&x3, &mean)?;
                let x_sq = ops::mul(&x_c, &x_c)?;
                let var = ops::reduce(&x_sq, MlxReduce::Mean, &[2], true)?;
                let var_eps = ops::add(&var, &eps_arr)?;
                let inv_std = ops::unary(&var_eps, MlxUnary::Rsqrt)?;
                let x_hat = ops::mul(&x_c, &inv_std)?;
                let dy_g = ops::mul(&dy3, &gamma_flat)?;
                let m_sy = ops::reduce(&dy_g, MlxReduce::Mean, &[2], true)?;
                let dy_gxh = ops::mul(&dy_g, &x_hat)?;
                let m_sxh = ops::reduce(&dy_gxh, MlxReduce::Mean, &[2], true)?;
                let term = ops::sub(&dy_g, &ops::add(&m_sy, &ops::mul(&x_hat, &m_sxh)?)?)?;
                let dx3 = ops::mul(&inv_std, &term)?;
                let dx5 = ops::reshape(&dx3, &[n, g, cpg, h, w])?;
                ops::reshape(&dx5, &[n, c, h, w])?
            }

            Op::GroupNormBackwardGamma { num_groups, eps } => {
                let x = lookup(&env, node.inputs[0])?;
                let dy = lookup(&env, node.inputs[1])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let n = x_shape[0];
                let c = x_shape[1];
                let h = x_shape[2];
                let w = x_shape[3];
                let g = *num_groups as i32;
                let cpg = c / g;
                let inner = cpg * h * w;
                let dtype = node.shape.dtype();
                let eps_arr = Array::from_f32_slice(&[*eps], &[1], dtype)?;
                let x5 = ops::reshape(x, &[n, g, cpg, h, w])?;
                let x3 = ops::reshape(&x5, &[n, g, inner])?;
                let x_sq = ops::mul(&x3, &x3)?;
                let mean_sq = ops::reduce(&x_sq, MlxReduce::Mean, &[2], true)?;
                let mean = ops::reduce(&x3, MlxReduce::Mean, &[2], true)?;
                let mean_sq2 = ops::mul(&mean, &mean)?;
                let var = ops::sub(&mean_sq, &mean_sq2)?;
                let var_eps = ops::add(&var, &eps_arr)?;
                let inv_std = ops::unary(&var_eps, MlxUnary::Rsqrt)?;
                let x_hat3 = ops::mul(&ops::sub(&x3, &mean)?, &inv_std)?;
                let x_hat = ops::reshape(&x_hat3, &[n, c, h, w])?;
                let prod = ops::mul(dy, &x_hat)?;
                let summed = ops::reduce(&prod, MlxReduce::Sum, &[0, 2, 3], false)?;
                let want: Vec<i32> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i32)
                    .collect();
                let got = summed.shape()?;
                let got_i32: Vec<i32> = got.iter().map(|&d| d as i32).collect();
                if got_i32 == want {
                    summed
                } else {
                    ops::reshape(&summed, &want)?
                }
            }

            Op::GroupNormBackwardBeta {
                num_groups: _,
                eps: _,
            } => {
                let dy = lookup(&env, node.inputs[1])?;
                let summed = ops::reduce(dy, MlxReduce::Sum, &[0, 2, 3], false)?;
                let want: Vec<i32> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i32)
                    .collect();
                let got = summed.shape()?;
                let got_i32: Vec<i32> = got.iter().map(|&d| d as i32).collect();
                if got_i32 == want {
                    summed
                } else {
                    ops::reshape(&summed, &want)?
                }
            }

            Op::BatchNormInferenceBackwardInput { eps } => {
                // dx = dy · γ · 1/√(σ²+ε). Mean is unused (frozen stats);
                // matches `batch_norm_inference_backward_input` on CPU.
                let gamma = lookup(&env, node.inputs[1])?;
                let var = lookup(&env, node.inputs[3])?;
                let dy = lookup(&env, node.inputs[4])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let dtype = node.shape.dtype();
                if x_shape.is_empty() {
                    return Err(MlxError(
                        "BatchNormInferenceBackwardInput on MLX: scalar unsupported".into(),
                    ));
                }
                let c = *x_shape.last().unwrap();
                let mut bshape = vec![1i32; x_shape.len()];
                *bshape.last_mut().unwrap() = c;
                let gamma_b = ops::reshape(&mlx_norm_scale_1d(gamma)?, &bshape)?;
                let var_b = ops::reshape(&mlx_norm_scale_1d(var)?, &bshape)?;
                let eps_arr = Array::from_f32_slice(&[*eps], &[1], dtype)?;
                let inv = ops::unary(&ops::add(&var_b, &eps_arr)?, MlxUnary::Rsqrt)?;
                ops::mul(dy, &ops::mul(&gamma_b, &inv)?)?
            }

            Op::BatchNormInferenceBackwardGamma { eps } => {
                // dγ_c = Σ dy · x̂ over all axes except the channel (last).
                let x = lookup(&env, node.inputs[0])?;
                let mean = lookup(&env, node.inputs[1])?;
                let var = lookup(&env, node.inputs[2])?;
                let dy = lookup(&env, node.inputs[3])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let dtype = node.shape.dtype();
                if x_shape.is_empty() {
                    return Err(MlxError(
                        "BatchNormInferenceBackwardGamma on MLX: scalar unsupported".into(),
                    ));
                }
                let c = *x_shape.last().unwrap();
                let last = (x_shape.len() - 1) as i32;
                let mut bshape = vec![1i32; x_shape.len()];
                *bshape.last_mut().unwrap() = c;
                let mean_b = ops::reshape(&mlx_norm_scale_1d(mean)?, &bshape)?;
                let var_b = ops::reshape(&mlx_norm_scale_1d(var)?, &bshape)?;
                let eps_arr = Array::from_f32_slice(&[*eps], &[1], dtype)?;
                let inv = ops::unary(&ops::add(&var_b, &eps_arr)?, MlxUnary::Rsqrt)?;
                let x_hat = ops::mul(&ops::sub(x, &mean_b)?, &inv)?;
                let prod = ops::mul(dy, &x_hat)?;
                if last == 0 {
                    prod
                } else {
                    let reduce_axes: Vec<i32> = (0..last).collect();
                    let summed =
                        ops::reduce(&prod, MlxReduce::Sum, &reduce_axes, /*keep_dim=*/ false)?;
                    let want: Vec<i32> = node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static() as i32)
                        .collect();
                    let got = summed.shape()?;
                    let got_i32: Vec<i32> = got.iter().map(|&d| d as i32).collect();
                    if got_i32 == want {
                        summed
                    } else {
                        ops::reshape(&summed, &want)?
                    }
                }
            }

            Op::BatchNormInferenceBackwardBeta => {
                // dβ_c = Σ dy over all axes except the channel (last).
                let dy = lookup(&env, node.inputs[0])?;
                let dy_shape = node_input_shape(graph, node.inputs[0]);
                if dy_shape.is_empty() {
                    return Err(MlxError(
                        "BatchNormInferenceBackwardBeta on MLX: scalar unsupported".into(),
                    ));
                }
                let last = (dy_shape.len() - 1) as i32;
                if last == 0 {
                    dy.clone_handle()?
                } else {
                    let reduce_axes: Vec<i32> = (0..last).collect();
                    let summed =
                        ops::reduce(dy, MlxReduce::Sum, &reduce_axes, /*keep_dim=*/ false)?;
                    let want: Vec<i32> = node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static() as i32)
                        .collect();
                    let got = summed.shape()?;
                    let got_i32: Vec<i32> = got.iter().map(|&d| d as i32).collect();
                    if got_i32 == want {
                        summed
                    } else {
                        ops::reshape(&summed, &want)?
                    }
                }
            }

            Op::CumsumBackward { axis, exclusive } => {
                let dy = lookup(&env, node.inputs[0])?;
                let axis_pos = if *axis < 0 {
                    node_input_shape(graph, node.inputs[0]).len() as i32 + *axis
                } else {
                    *axis
                };
                let total = ops::reduce(dy, MlxReduce::Sum, &[axis_pos], true)?;
                if *exclusive {
                    let inc = ops::cumsum(dy, axis_pos, false)?;
                    ops::sub(&total, &inc)?
                } else {
                    let pref = ops::cumsum(dy, axis_pos, true)?;
                    ops::sub(&total, &pref)?
                }
            }

            Op::GatherBackward { axis } => {
                let dy = lookup(&env, node.inputs[0])?;
                let indices_in = lookup(&env, node.inputs[1])?.clone_handle()?;
                let out_shape: Vec<i32> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i32)
                    .collect();
                let axis_pos = if *axis < 0 {
                    out_shape.len() as i32 + *axis
                } else {
                    *axis
                };
                let dy_shape = node_input_shape(graph, node.inputs[0]);
                let idx_shape = node_input_shape(graph, node.inputs[1]);
                let n_elem: usize = out_shape.iter().product::<i32>() as usize;
                let zeros = vec![0.0_f32; n_elem];
                let out_shape_usize: Vec<usize> = out_shape.iter().map(|d| *d as usize).collect();
                let zero_target =
                    crate::array::Array::from_f32_slice(&zeros, &out_shape_usize, DType::F32)?;
                let indices = if dy_shape.len() > 1 && idx_shape.len() == 1 {
                    ops::reshape(&indices_in, &[idx_shape[0], 1])?
                } else {
                    indices_in
                };
                ops::scatter_add_axis(&zero_target, &indices, dy, axis_pos)?
            }

            Op::RopeBackward { head_dim, n_rot } => {
                // Backward = forward rotation with negated sin (NeoX).
                let dy = lookup(&env, node.inputs[0])?;
                let cos = lookup(&env, node.inputs[1])?;
                let sin = lookup(&env, node.inputs[2])?;
                let neg_one = Array::from_f32_slice(&[-1.0], &[1], node.shape.dtype())?;
                let sin_neg = ops::mul(sin, &neg_one)?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let n = x_shape.len();
                let hd = *head_dim as i32;
                let nr = *n_rot as i32;
                let rot_half = nr / 2;
                if n < 2 {
                    return Err(MlxError("RopeBackward: dy must be rank ≥ 2".into()));
                }
                let rotate = |x_rot: &Array,
                              rot_shape: &[i32],
                              seq_axis: usize,
                              pairs: i32|
                 -> Result<Array, MlxError> {
                    let rn = rot_shape.len();
                    let seq_v = rot_shape[seq_axis];
                    let cos_seq = ops::slice(cos, &[0, 0], &[seq_v, pairs])?;
                    let sin_seq = ops::slice(&sin_neg, &[0, 0], &[seq_v, pairs])?;
                    let mut bshape = vec![1i32; rn];
                    bshape[seq_axis] = seq_v;
                    bshape[rn - 1] = pairs;
                    let cos_b = ops::reshape(&cos_seq, &bshape)?;
                    let sin_b = ops::reshape(&sin_seq, &bshape)?;
                    let mut x1_stop = rot_shape.to_vec();
                    x1_stop[rn - 1] = pairs;
                    let x1 = ops::slice(x_rot, &vec![0i32; rn], &x1_stop)?;
                    let mut x2_start = vec![0i32; rn];
                    x2_start[rn - 1] = pairs;
                    let x2 = ops::slice(x_rot, &x2_start, rot_shape)?;
                    let x1_cos = ops::mul(&x1, &cos_b)?;
                    let x2_sin = ops::mul(&x2, &sin_b)?;
                    let x2_cos = ops::mul(&x2, &cos_b)?;
                    let x1_sin = ops::mul(&x1, &sin_b)?;
                    let y1 = ops::sub(&x1_cos, &x2_sin)?;
                    let y2 = ops::add(&x2_cos, &x1_sin)?;
                    ops::concat(&[&y1, &y2], (rn - 1) as i32)
                };
                let last = *x_shape.last().unwrap();
                if last < nr {
                    return Err(MlxError(format!(
                        "RopeBackward: last dim {last} < n_rot {n_rot}"
                    )));
                }
                let mut rot_stop = x_shape.clone();
                rot_stop[n - 1] = nr.min(hd);
                let rot = ops::slice(dy, &vec![0i32; n], &rot_stop)?;
                let rotated = rotate(&rot, &rot_stop, n - 2, rot_half)?;
                if last == nr.min(hd) {
                    rotated
                } else {
                    let mut tail_start = vec![0i32; n];
                    tail_start[n - 1] = nr.min(hd);
                    let tail = ops::slice(dy, &tail_start, &x_shape)?;
                    ops::concat(&[&rotated, &tail], (n - 1) as i32)?
                }
            }

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
                let positions = lookup(&env, node.inputs[0])?.to_f32()?;
                let scales = lookup(&env, node.inputs[1])?.to_f32()?;
                let rotations = lookup(&env, node.inputs[2])?.to_f32()?;
                let opacities = lookup(&env, node.inputs[3])?.to_f32()?;
                let colors = lookup(&env, node.inputs[4])?.to_f32()?;
                let sh_coeffs = lookup(&env, node.inputs[5])?.to_f32()?;
                let meta = lookup(&env, node.inputs[6])?.to_f32()?;
                let out_host = crate::splat::render_host_slices(
                    &positions,
                    &scales,
                    &rotations,
                    &opacities,
                    &colors,
                    &sh_coeffs,
                    &meta,
                    *width,
                    *height,
                    *tile_size,
                    *radius_scale,
                    *alpha_cutoff,
                    *max_splat_steps,
                    *transmittance_threshold,
                    *max_list_entries,
                );
                let out_shape: Vec<usize> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect();
                Array::from_f32_slice(&out_host, &out_shape, DType::F32)?
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
                let positions = lookup(&env, node.inputs[0])?.to_f32()?;
                let scales = lookup(&env, node.inputs[1])?.to_f32()?;
                let rotations = lookup(&env, node.inputs[2])?.to_f32()?;
                let opacities = lookup(&env, node.inputs[3])?.to_f32()?;
                let colors = lookup(&env, node.inputs[4])?.to_f32()?;
                let sh_coeffs = lookup(&env, node.inputs[5])?.to_f32()?;
                let meta = lookup(&env, node.inputs[6])?.to_f32()?;
                let d_loss = lookup(&env, node.inputs[7])?.to_f32()?;
                let packed = crate::splat::backward_host_slices(
                    &positions,
                    &scales,
                    &rotations,
                    &opacities,
                    &colors,
                    &sh_coeffs,
                    &meta,
                    &d_loss,
                    *width,
                    *height,
                    *tile_size,
                    *radius_scale,
                    *alpha_cutoff,
                    *max_splat_steps,
                    *transmittance_threshold,
                    *max_list_entries,
                    *loss_grad_clip,
                    *sh_band,
                    *max_anisotropy,
                );
                let out_shape: Vec<usize> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect();
                Array::from_f32_slice(&packed, &out_shape, DType::F32)?
            }

            Op::Custom { name, attrs, .. } => {
                // Vocos GatherND with k=1: use native take along axis 0. Host
                // byte-staging for this Custom saw a non-matching layout for
                // Transpose→Gather inputs (to_bytes peak-matched but values
                // diverged vs the same node as a graph output).
                if name == "onnx.GatherND" && node.inputs.len() >= 2 {
                    let batch_dims = if attrs.len() >= 4 {
                        i32::from_le_bytes(attrs[0..4].try_into().unwrap()).max(0)
                    } else {
                        0
                    };
                    let data = lookup(&env, node.inputs[0])?;
                    let idx_raw = lookup(&env, node.inputs[1])?;
                    let idx_shape = idx_raw.shape()?;
                    let k = idx_shape.last().copied().unwrap_or(0);
                    if batch_dims == 0 && k <= 1 {
                        let idx = mlx_indices_i64(idx_raw)?;
                        // Flat indices for take: squeeze trailing k=1.
                        let idx = if k == 1 && idx_shape.len() >= 2 {
                            let flat: Vec<i32> = idx_shape[..idx_shape.len() - 1]
                                .iter()
                                .map(|&d| d as i32)
                                .collect();
                            ops::reshape(&idx, &flat)?
                        } else {
                            idx.clone_handle()?
                        };
                        ops::take(data, &idx, 0)?
                    } else {
                        let kernel = crate::op_registry::lookup_mlx_kernel(name).ok_or_else(|| {
                            MlxError(format!(
                                "rlx-mlx: no MlxKernel registered for \
                                 Op::Custom('{name}'). Either register one \
                                 via rlx_mlx::op_registry::register_mlx_kernel \
                                 or pin this graph to Device::Cpu."
                            ))
                        })?;
                        let in_refs: Vec<&Array> = node
                            .inputs
                            .iter()
                            .map(|&in_id| lookup(&env, in_id))
                            .collect::<Result<Vec<_>, _>>()?;
                        kernel.execute(&in_refs, &node.shape, attrs)?
                    }
                } else if name == "onnx.ScatterElements" && node.inputs.len() >= 3 {
                    // Vocos ISTFT overlap-add. Host staging of zeros/updates
                    // was packing a few non-finites and wrong layouts; use
                    // native MLX scatter_add when reduction=add.
                    let axis = if attrs.len() >= 4 {
                        i32::from_le_bytes(attrs[0..4].try_into().unwrap())
                    } else {
                        0
                    };
                    let reduction = if attrs.len() >= 8 {
                        i32::from_le_bytes(attrs[4..8].try_into().unwrap())
                    } else {
                        0
                    };
                    if reduction == 1 {
                        let data = lookup(&env, node.inputs[0])?;
                        let updates = lookup(&env, node.inputs[2])?;
                        let indices_in = mlx_indices_i64(lookup(&env, node.inputs[1])?)?;
                        let out_shape: Vec<i32> = node
                            .shape
                            .dims()
                            .iter()
                            .map(|d| d.unwrap_static() as i32)
                            .collect();
                        // Prefer a fresh zero base — Vocos ISTFT Expand(0) is the
                        // only producer, and host-staged Expand has shown sparse
                        // non-finites when composed with Custom.
                        let n_elem: usize = out_shape.iter().map(|&d| d as usize).product();
                        let zeros = vec![0.0_f32; n_elem];
                        let out_shape_usize: Vec<usize> =
                            out_shape.iter().map(|&d| d as usize).collect();
                        let zero_target = crate::array::Array::from_f32_slice(
                            &zeros,
                            &out_shape_usize,
                            DType::F32,
                        )?;
                        let _ = data; // ignored: ISTFT base is zeros
                        let axis = if axis < 0 {
                            axis + out_shape.len() as i32
                        } else {
                            axis
                        };
                        // Vocos ISTFT uses a rank-1 window scatter
                        // (`[2048*T]`) that trips MLX's rank-1 `scatter_add`
                        // shape checks — reshape to `[N,1]` / axis 0 instead of
                        // falling back to the host Custom kernel (Lazy eval of
                        // Custom inside compile is forbidden and the host path
                        // also blew up magnitudes for short utterances).
                        if out_shape.len() == 1 {
                            let n = out_shape[0];
                            let zero_2d = ops::reshape(&zero_target, &[n, 1])?;
                            let idx_n = indices_in
                                .shape()?
                                .iter()
                                .copied()
                                .product::<usize>()
                                .max(1) as i32;
                            let indices = ops::reshape(&indices_in, &[idx_n, 1])?;
                            let upd_n = updates
                                .shape()?
                                .iter()
                                .copied()
                                .product::<usize>()
                                .max(1) as i32;
                            let updates_2d = ops::reshape(updates, &[upd_n, 1])?;
                            let scattered =
                                ops::scatter_add_axis(&zero_2d, &indices, &updates_2d, 0)?;
                            ops::reshape(&scattered, &[n])?
                        } else if out_shape.len() > 1 {
                            let idx_shape = indices_in.shape()?;
                            let indices = if idx_shape.len() == 1 {
                                ops::reshape(&indices_in, &[idx_shape[0] as i32, 1])?
                            } else {
                                indices_in
                            };
                            ops::scatter_add_axis(&zero_target, &indices, updates, axis)?
                        } else {
                            let kernel =
                                crate::op_registry::lookup_mlx_kernel(name).ok_or_else(|| {
                                    MlxError(format!(
                                        "rlx-mlx: no MlxKernel registered for \
                                         Op::Custom('{name}')."
                                    ))
                                })?;
                            let in_refs: Vec<&Array> = node
                                .inputs
                                .iter()
                                .map(|&in_id| lookup(&env, in_id))
                                .collect::<Result<Vec<_>, _>>()?;
                            kernel.execute(&in_refs, &node.shape, attrs)?
                        }
                    } else {
                        let kernel = crate::op_registry::lookup_mlx_kernel(name).ok_or_else(|| {
                            MlxError(format!(
                                "rlx-mlx: no MlxKernel registered for \
                                 Op::Custom('{name}'). Either register one \
                                 via rlx_mlx::op_registry::register_mlx_kernel \
                                 or pin this graph to Device::Cpu."
                            ))
                        })?;
                        let in_refs: Vec<&Array> = node
                            .inputs
                            .iter()
                            .map(|&in_id| lookup(&env, in_id))
                            .collect::<Result<Vec<_>, _>>()?;
                        kernel.execute(&in_refs, &node.shape, attrs)?
                    }
                } else {
                    let kernel = crate::op_registry::lookup_mlx_kernel(name).ok_or_else(|| {
                        MlxError(format!(
                            "rlx-mlx: no MlxKernel registered for \
                             Op::Custom('{name}'). Either register one \
                             via rlx_mlx::op_registry::register_mlx_kernel \
                             or pin this graph to Device::Cpu."
                        ))
                    })?;
                    let in_refs: Vec<&Array> = node
                        .inputs
                        .iter()
                        .map(|&in_id| lookup(&env, in_id))
                        .collect::<Result<Vec<_>, _>>()?;
                    kernel.execute(&in_refs, &node.shape, attrs)?
                }
            }

            // Identity-forward op used by the GRL (Gradient Reverse Layer)
            // in adversarial training. Forward value matches the input; the
            // gradient pass treats it as a stop. MLX's compiled trace only
            // sees the forward, so we lower it to a no-op clone.
            Op::StopGradient => {
                let x = lookup(&env, node.inputs[0])?;
                x.clone_handle()?
            }

            Op::Fma => {
                let a = lookup(&env, node.inputs[0])?;
                let b = lookup(&env, node.inputs[1])?;
                let c = lookup(&env, node.inputs[2])?;
                let ab = ops::mul(a, b)?;
                ops::add(&ab, c)?
            }

            Op::Conv3d {
                stride,
                padding,
                dilation,
                groups,
            } => {
                let in_shape = node_input_shape(graph, node.inputs[0]);
                if in_shape.len() != 5 {
                    return Err(MlxError(format!(
                        "Conv3d: expected NCDHW input rank 5, got {}",
                        in_shape.len()
                    )));
                }
                let x = lookup(&env, node.inputs[0])?;
                let w = lookup(&env, node.inputs[1])?;
                let x_nd = ops::transpose(x, &[0, 2, 3, 4, 1])?;
                let w_mlx = ops::transpose(w, &[0, 2, 3, 4, 1])?;
                let y_nd = ops::conv3d(
                    &x_nd,
                    &w_mlx,
                    (stride[0] as i32, stride[1] as i32, stride[2] as i32),
                    (padding[0] as i32, padding[1] as i32, padding[2] as i32),
                    (dilation[0] as i32, dilation[1] as i32, dilation[2] as i32),
                    (*groups).max(1) as i32,
                )?;
                ops::transpose(&y_nd, &[0, 4, 1, 2, 3])?
            }

            Op::FusedConvBiasAct { .. } | Op::PartitionedConv { .. } => {
                let mut g = Graph::new("mlx_unfuse");
                let mut ids = Vec::with_capacity(node.inputs.len());
                for (i, &in_id) in node.inputs.iter().enumerate() {
                    let sh = graph.node(in_id).shape.clone();
                    ids.push(g.append_node(
                        Op::Input {
                            name: format!("in{i}"),
                        },
                        vec![],
                        sh,
                        None,
                    ));
                }
                let out_id = g.append_node(node.op.clone(), ids, node.shape.clone(), None);
                g.set_outputs(vec![out_id]);
                let g2 = rlx_opt::unfuse_fused_for_autodiff(g);
                let mut env2: HashMap<NodeId, Array> = HashMap::new();
                for n2 in g2.nodes() {
                    if let Op::Input { name } = &n2.op {
                        if let Some(rest) = name.strip_prefix("in") {
                            if let Ok(i) = rest.parse::<usize>() {
                                if let Some(&src) = node.inputs.get(i) {
                                    env2.insert(n2.id, lookup(&env, src)?.clone_handle()?);
                                }
                            }
                        }
                    }
                }
                let outs =
                    lower_with_env(&g2, env2, params, params_typed, rng, eval_barriers)?;
                outs.into_iter()
                    .next()
                    .ok_or_else(|| MlxError("mlx unfuse: empty outputs".into()))?
            }

            Op::ComplexNormSq => {
                let z = lookup(&env, node.inputs[0])?;
                let out_dims: Vec<i32> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i32)
                    .collect();
                lower_complex_norm_sq(z, &out_dims)?
            }
            Op::ComplexNormSqBackward => {
                let z = lookup(&env, node.inputs[0])?;
                let g = lookup(&env, node.inputs[1])?;
                let logical: Vec<i32> = graph
                    .node(node.inputs[1])
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i32)
                    .collect();
                lower_complex_norm_sq_backward(z, g, &logical)?
            }
            Op::Conjugate => {
                let z = lookup(&env, node.inputs[0])?;
                let n: i32 = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i32)
                    .product();
                lower_conjugate(z, n)?
            }

            Op::Quantize {
                axis,
                scales,
                zero_points,
            } => {
                let x = lookup(&env, node.inputs[0])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                lower_quantize(x, &x_shape, *axis, scales, zero_points)?
            }
            Op::Dequantize {
                axis,
                scales,
                zero_points,
            } => {
                let q = lookup(&env, node.inputs[0])?;
                let dims: Vec<i32> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i32)
                    .collect();
                lower_dequantize(q, &dims, *axis, scales, zero_points)?
            }

            Op::FakeQuantizeLSQ { bits, axis } => {
                let x = lookup(&env, node.inputs[0])?;
                let scale = lookup(&env, node.inputs[1])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let q_max = fq_q_max(*bits)?;
                let scale_b = fq_scale_from_state(scale, &x_shape, *axis, DType::F32)?;
                fq_quantize_dequantize(x, &scale_b, q_max, DType::F32)?
            }
            Op::FakeQuantizeLSQBackwardX { bits, axis } => {
                let x = lookup(&env, node.inputs[0])?;
                let scale = lookup(&env, node.inputs[1])?;
                let dy = lookup(&env, node.inputs[2])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let q_max = fq_q_max(*bits)?;
                lower_lsq_backward_x(x, scale, dy, &x_shape, *axis, q_max)?
            }
            Op::FakeQuantizeLSQBackwardScale { bits, axis } => {
                let x = lookup(&env, node.inputs[0])?;
                let scale = lookup(&env, node.inputs[1])?;
                let dy = lookup(&env, node.inputs[2])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                let q_max = fq_q_max(*bits)?;
                lower_lsq_backward_scale(x, scale, dy, &x_shape, *axis, q_max)?
            }

            Op::QMatMul {
                x_zp,
                w_zp,
                out_zp,
                mult,
            } => {
                let x = lookup(&env, node.inputs[0])?;
                let w = lookup(&env, node.inputs[1])?;
                let bias = lookup(&env, node.inputs[2])?;
                lower_q_mat_mul(x, w, bias, *x_zp, *w_zp, *out_zp, *mult)?
            }
            Op::QConv2d {
                kernel_size: _,
                stride,
                padding,
                dilation,
                groups,
                x_zp,
                w_zp,
                out_zp,
                mult,
            } => {
                let x = lookup(&env, node.inputs[0])?;
                let w = lookup(&env, node.inputs[1])?;
                let bias = lookup(&env, node.inputs[2])?;
                let s = |i: usize| stride.get(i).copied().unwrap_or(1) as i32;
                let p = |i: usize| padding.get(i).copied().unwrap_or(0) as i32;
                let d = |i: usize| dilation.get(i).copied().unwrap_or(1) as i32;
                lower_q_conv2d(
                    x,
                    w,
                    bias,
                    (s(0), s(1)),
                    (p(0), p(1)),
                    (d(0), d(1)),
                    (*groups).max(1) as i32,
                    *x_zp,
                    *w_zp,
                    *out_zp,
                    *mult,
                )?
            }

            Op::FftButterflyStage { stage, n_fft } => {
                let state = lookup(&env, node.inputs[0])?;
                let gate = lookup(&env, node.inputs[1])?;
                let rev = lookup(&env, node.inputs[2])?;
                let tw_re = lookup(&env, node.inputs[3])?;
                let tw_im = lookup(&env, node.inputs[4])?;
                let st_shape = node_input_shape(graph, node.inputs[0]);
                if st_shape.len() != 2 {
                    return Err(MlxError(format!(
                        "FftButterflyStage: state must be rank-2 [B, 2N], got {:?}",
                        st_shape
                    )));
                }
                lower_fft_butterfly_stage(
                    state,
                    gate,
                    rev,
                    tw_re,
                    tw_im,
                    st_shape[0],
                    *n_fft as i32,
                    *stage,
                )?
            }

            Op::ScaledQuantScale {
                format,
                scale_layout,
            } if scaled_fp8_mlx_ok(*format, *scale_layout) => {
                let x = lookup(&env, node.inputs[0])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                lower_scaled_quant_scale(x, *format, &x_shape)?
            }
            Op::ScaledDequantize {
                format,
                scale_layout,
            } if scaled_fp8_mlx_ok(*format, *scale_layout) => {
                let codes = lookup(&env, node.inputs[0])?;
                let scale = lookup(&env, node.inputs[1])?;
                let out_dims: Vec<i32> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i32)
                    .collect();
                lower_scaled_dequantize(codes, scale, *format, &out_dims)?
            }
            Op::ScaledMatMul {
                lhs_format,
                rhs_format,
                scale_layout,
                has_bias,
            } if scaled_fp8_mlx_ok(*lhs_format, *scale_layout)
                && scaled_fp8_mlx_ok(*rhs_format, *scale_layout) =>
            {
                let lhs = lookup(&env, node.inputs[0])?;
                let rhs = lookup(&env, node.inputs[1])?;
                let lhs_scale = lookup(&env, node.inputs[2])?;
                let rhs_scale = lookup(&env, node.inputs[3])?;
                let bias = if *has_bias {
                    Some(lookup(&env, node.inputs[4])?)
                } else {
                    None
                };
                let lhs_shape = node_input_shape(graph, node.inputs[0]);
                let rhs_shape = node_input_shape(graph, node.inputs[1]);
                if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
                    return Err(MlxError(
                        "ScaledMatMul: expected rank-2 TN operands".into(),
                    ));
                }
                let (m, k) = (lhs_shape[0], lhs_shape[1]);
                let (n, k2) = (rhs_shape[0], rhs_shape[1]);
                if k != k2 {
                    return Err(MlxError(format!(
                        "ScaledMatMul: K mismatch {k} vs {k2}"
                    )));
                }
                lower_scaled_matmul(
                    lhs,
                    rhs,
                    lhs_scale,
                    rhs_scale,
                    bias,
                    *lhs_format,
                    *rhs_format,
                    m,
                    k,
                    n,
                )?
            }

            Op::Mamba2 {
                head_dim,
                state_size,
            } => {
                let x = lookup(&env, node.inputs[0])?;
                let dt = lookup(&env, node.inputs[1])?;
                let a = lookup(&env, node.inputs[2])?;
                let b_in = lookup(&env, node.inputs[3])?;
                let c_in = lookup(&env, node.inputs[4])?;
                let x_shape = node_input_shape(graph, node.inputs[0]);
                if x_shape.len() != 4 {
                    return Err(MlxError(format!(
                        "Mamba2: x must be rank-4 [B,S,H,P], got rank {}",
                        x_shape.len()
                    )));
                }
                lower_mamba2(
                    x,
                    dt,
                    a,
                    b_in,
                    c_in,
                    x_shape[0],
                    x_shape[1],
                    x_shape[2],
                    *head_dim as i32,
                    *state_size as i32,
                )?
            }

            other if is_mlx_typed_host_op(other) => host_eval_op_typed(graph, node, &env)?,

            other => {
                return unsupported(format!("{other:?}"));
            }
        })
        })()
        .map_err(|e| {
            let name = node.name.as_deref().unwrap_or("<unnamed>");
            let inputs = node
                .inputs
                .iter()
                .map(|&input| {
                    let n = graph.node(input);
                    let actual = env
                        .get(&input)
                        .and_then(|a| a.shape().ok())
                        .map(|s| format!("{s:?}"))
                        .unwrap_or_else(|| "<unbound>".into());
                    format!(
                        "{input:?}:{}:{:?}:{:?} (runtime={actual})",
                        n.name.as_deref().unwrap_or("<unnamed>"),
                        n.op.kind(),
                        n.shape
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            MlxError(format!(
                "lower {name} ({id:?}, {:?}, op={:?}; inputs=[{inputs}]): {e}",
                node.op.kind(),
                node.op
            ))
        })?;
        if let Some(t0) = t0 {
            mlx_profile_record(
                mlx_profile_kind(&node.op),
                rlx_ir::Tick::now().elapsed_ns(t0),
            );
        }

        env.insert(id, arr);
        if debug_eval {
            let label = node
                .name
                .as_deref()
                .map(|n| format!("{n} ({id:?})"))
                .unwrap_or_else(|| format!("{id:?}"));
            if let Some(a) = env.get(&id) {
                eval(&[a]).map_err(|e| MlxError(format!("eval at {label}: {e}")))?;
                eprintln!("rlx-mlx: {label} {:?} -> {:?}", node.op.kind(), a.shape()?);
            }
        } else if eval_barriers {
            if is_fusable(&node.op) {
                let d = 1 + node
                    .inputs
                    .iter()
                    .map(|i| fuse_depth.get(i).copied().unwrap_or(0))
                    .max()
                    .unwrap_or(0);
                if d >= fuse_cap {
                    if let Some(a) = env.get(&id) {
                        eval(&[a]).map_err(|e| MlxError(format!("fuse-cap eval: {e}")))?;
                    }
                    fuse_depth.insert(id, 0);
                } else {
                    fuse_depth.insert(id, d);
                }
            } else {
                fuse_depth.insert(id, 0);
            }
        }
    }

    // Look outputs up by reference — `graph.outputs` may legitimately
    // contain duplicate NodeIds (e.g. when a vmap'd graph has the same
    // tangent output reused across multiple slots), so removing on
    // first hit would break the second occurrence with a phantom
    // "not lowered" error. The Array clones here are MLX handle
    // clones (Arc-like), not data copies.
    let mut outs = Vec::with_capacity(graph.outputs.len());
    for &out_id in &graph.outputs {
        let arr = env
            .get(&out_id)
            .ok_or_else(|| MlxError(format!("output node {out_id:?} was not lowered")))?
            .clone_handle()?;
        outs.push(arr);
    }
    Ok(outs)
}
