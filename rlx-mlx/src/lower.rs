// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Lower an `rlx_ir::Graph` into a chain of MLX `Array` handles.
//!
//! Strategy is "fresh graph per run": every call rebuilds the MLX
//! graph from scratch using current input/param data. Simpler than
//! holding a persistent graph + replaceable placeholders, and MLX's
//! own trace cache amortizes the per-build cost. A future pass can
//! switch to `mlx::compile`-style placeholder bindings if we need
//! to drop the per-run construction overhead.

use std::collections::HashMap;

use rlx_ir::op::{
    Activation, BinaryOp, ChainOperand, ChainStep, CmpOp, MaskKind, ReduceOp, ScaleMode, SteKind,
};
use rlx_ir::shape::{Dim, DimBinding, Shape};
use rlx_ir::{DType, Graph, NodeId, Op};

use crate::array::{Array, MlxError, async_eval, eval};
use crate::ffi::{MlxMask, MlxReduce, MlxUnary};
use crate::ops;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MlxMode {
    /// Eval after every op. Slower but useful for debugging — failures
    /// surface at the offending op rather than at the final eval.
    Eager,
    /// Build the full graph, eval all outputs in one shot. Default.
    /// Lets MLX's optimizer schedule the whole DAG.
    #[default]
    Lazy,
    /// Build the full graph and `async_eval` the outputs, but don't
    /// wait for completion. Used by `commit_no_wait` to amortize sync
    /// latency across pipelined runs.
    AsyncCommit,
    /// Compile the graph once via `mlx::compile` and replay the
    /// optimized trace on every subsequent `run()`. First call pays
    /// the trace cost; subsequent calls skip the per-run rebuild.
    Compiled,
}

/// What kind of host-side data each leaf node needs. Built once at
/// compile time; re-used at run time to materialize MLX leaves in the
/// same order across calls (essential for the mlx::compile path —
/// position determines which placeholder the compiled trace expects).
#[derive(Debug, Clone)]
pub enum LeafKey {
    Input(String),
    Param(String),
    Constant, // node id is implicit from leaf_order's NodeId
}

/// Walk `graph` in topo order and return the (NodeId, LeafKey) pairs
/// for every Input/Param/Constant node, in declaration order. Used by
/// the runtime's compile path to know which f32 buffers to bind to
/// which positional input of the compiled function.
pub fn leaf_order(graph: &Graph) -> Vec<(NodeId, LeafKey)> {
    let mut out = Vec::new();
    for node in graph.nodes() {
        match &node.op {
            Op::Input { name } => out.push((node.id, LeafKey::Input(name.clone()))),
            Op::Param { name } => out.push((node.id, LeafKey::Param(name.clone()))),
            Op::Constant { .. } => out.push((node.id, LeafKey::Constant)),
            _ => {}
        }
    }
    out
}

/// Build the leaf array for a single node. Prefers typed bytes if a
/// matching name appears in `inputs_typed` / `params_typed`; falls
/// back to the f32 host map. The typed path uses Array::from_bytes
/// for zero-widen F16/BF16 / I32 leaves.
pub fn build_leaf_for(
    graph: &Graph,
    id: NodeId,
    params: &HashMap<String, Vec<f32>>,
    inputs: &HashMap<String, Vec<f32>>,
    params_typed: &HashMap<String, (Vec<u8>, DType)>,
    inputs_typed: &HashMap<String, (Vec<u8>, DType)>,
) -> Result<Array, MlxError> {
    let node = graph.node(id);
    let shape: Vec<usize> = node
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let dtype = node.shape.dtype();
    match &node.op {
        Op::Input { name } => {
            if let Some((bytes, dt)) = inputs_typed.get(name) {
                if *dt != dtype {
                    return Err(MlxError(format!(
                        "typed input '{name}' dtype {dt:?} doesn't match graph's {dtype:?}"
                    )));
                }
                return Array::from_bytes(bytes, &shape, dtype);
            }
            let data = inputs
                .get(name)
                .ok_or_else(|| MlxError(format!("missing input '{name}'")))?;
            Array::from_f32_slice(data, &shape, dtype)
        }
        Op::Param { name } => {
            if let Some((bytes, dt)) = params_typed.get(name) {
                if *dt != dtype {
                    return Err(MlxError(format!(
                        "typed param '{name}' dtype {dt:?} doesn't match graph's {dtype:?}"
                    )));
                }
                return Array::from_bytes(bytes, &shape, dtype);
            }
            let data = params
                .get(name)
                .ok_or_else(|| MlxError(format!("missing param '{name}'")))?;
            Array::from_f32_slice(data, &shape, dtype)
        }
        Op::Constant { data } => {
            // Constants are little-endian raw bytes in the node's
            // dtype. Every dtype rlx-ir declares has a native MLX
            // counterpart; from_bytes handles the typed read directly.
            // F32 still goes through the iterator path because that
            // matches the prior behavior bit-for-bit.
            match dtype {
                DType::F32 => {
                    let n = data.len() / 4;
                    let mut buf = Vec::with_capacity(n);
                    for i in 0..n {
                        let bytes = [
                            data[i * 4],
                            data[i * 4 + 1],
                            data[i * 4 + 2],
                            data[i * 4 + 3],
                        ];
                        buf.push(f32::from_le_bytes(bytes));
                    }
                    Array::from_f32_slice(&buf, &shape, dtype)
                }
                _ => Array::from_bytes(data, &shape, dtype),
            }
        }
        other => Err(MlxError(format!("build_leaf called on non-leaf {other:?}"))),
    }
}

/// Lower a sub-graph (then/else branch of `Op::If`, or body/cond of
/// `Op::While`). Captures bind positionally: the i-th `Op::Input` in
/// the sub-graph (in topo order) is bound to `captures[i]`. Params
/// look up in the parent's `params` / `params_typed` by name. Every
/// leaf array gets a fresh `clone_handle` so the parent's ownership
/// is undisturbed.
pub fn lower_subgraph(
    sub: &Graph,
    captures: &[&Array],
    parent_params: &HashMap<String, Vec<f32>>,
    parent_params_typed: &HashMap<String, (Vec<u8>, DType)>,
) -> Result<Vec<Array>, MlxError> {
    let mut sub_env: HashMap<NodeId, Array> = HashMap::with_capacity(sub.nodes().len());

    let mut input_idx = 0;
    for node in sub.nodes() {
        match &node.op {
            Op::Input { name } => {
                if input_idx >= captures.len() {
                    return Err(MlxError(format!(
                        "sub-graph has more Op::Input nodes than parent supplied \
                         captures (input #{input_idx} = {name:?})"
                    )));
                }
                sub_env.insert(node.id, captures[input_idx].clone_handle()?);
                input_idx += 1;
            }
            Op::Param { name } => {
                if let Some((bytes, dt)) = parent_params_typed.get(name) {
                    let shape: Vec<usize> = node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    sub_env.insert(node.id, Array::from_bytes(bytes, &shape, *dt)?);
                } else if let Some(data) = parent_params.get(name) {
                    let shape: Vec<usize> = node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    let dtype = node.shape.dtype();
                    sub_env.insert(node.id, Array::from_f32_slice(data, &shape, dtype)?);
                } else {
                    return Err(MlxError(format!(
                        "sub-graph param '{name}' not found in parent's param maps"
                    )));
                }
            }
            Op::Constant { data } => {
                let shape: Vec<usize> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect();
                let dtype = node.shape.dtype();
                let leaf = match dtype {
                    DType::F32 => {
                        let n = data.len() / 4;
                        let mut buf = Vec::with_capacity(n);
                        for i in 0..n {
                            let bytes = [
                                data[i * 4],
                                data[i * 4 + 1],
                                data[i * 4 + 2],
                                data[i * 4 + 3],
                            ];
                            buf.push(f32::from_le_bytes(bytes));
                        }
                        Array::from_f32_slice(&buf, &shape, dtype)?
                    }
                    _ => Array::from_bytes(data, &shape, dtype)?,
                };
                sub_env.insert(node.id, leaf);
            }
            _ => {} // non-leaf: handled by lower_with_env
        }
    }

    if input_idx < captures.len() {
        // More captures than the sub-graph used. Not necessarily an
        // error — extra captures may have been provided "in case" —
        // but worth a debug-friendly note. For now silently allow.
    }

    lower_with_env(sub, sub_env, parent_params, parent_params_typed)
}

/// Walk `graph` with `env` already populated for every leaf node
/// (Input/Param/Constant). Internal nodes are dispatched to ops::* in
/// topological order; the resulting Array is inserted into `env`.
/// Returns the arrays for `graph.outputs`.
///
/// The eval semantics are the caller's responsibility — this function
/// only constructs the symbolic chain. `params` / `params_typed` are
/// the parent-scope param maps; they're needed only for ops that
/// recurse into sub-graphs (Op::If, Op::While) — sub-graph leaves
/// look them up by name. Pass empty maps for trace contexts that
/// don't see sub-graphs.
pub fn lower_with_env(
    graph: &Graph,
    mut env: HashMap<NodeId, Array>,
    params: &HashMap<String, Vec<f32>>,
    params_typed: &HashMap<String, (Vec<u8>, DType)>,
) -> Result<Vec<Array>, MlxError> {
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

        let arr = match &node.op {
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
                ops::matmul(a, b)?
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
                match bop {
                    BinaryOp::Add => ops::add(a, b)?,
                    BinaryOp::Mul => ops::mul(a, b)?,
                    BinaryOp::Sub => ops::sub(a, b)?,
                    BinaryOp::Div => ops::div(a, b)?,
                    BinaryOp::Max => ops::max(a, b)?,
                    BinaryOp::Min => ops::min(a, b)?,
                    BinaryOp::Pow => ops::pow(a, b)?,
                }
            }
            Op::Compare(cop) => {
                let a = lookup(&env, node.inputs[0])?;
                let b = lookup(&env, node.inputs[1])?;
                match cop {
                    CmpOp::Eq => ops::eq(a, b)?,
                    CmpOp::Ne => ops::ne(a, b)?,
                    CmpOp::Lt => ops::lt(a, b)?,
                    CmpOp::Le => ops::le(a, b)?,
                    CmpOp::Gt => ops::gt(a, b)?,
                    CmpOp::Ge => ops::ge(a, b)?,
                }
            }
            Op::Where => {
                let c = lookup(&env, node.inputs[0])?;
                let x = lookup(&env, node.inputs[1])?;
                let y = lookup(&env, node.inputs[2])?;
                ops::select(c, x, y)?
            }
            Op::ElementwiseRegion {
                chain,
                num_inputs,
                scalar_input_mask: _,
                input_modulus: _,
            } => {
                // PLAN L2: native MLX lowering. Compose `mlx::core::ops::*`
                // per ChainStep in declaration order; the resulting array
                // sub-graph stays inside MLX's lazy trace, so the optimizer
                // and `mlx::compile` get to fuse the whole chain into one
                // kernel — no decomposer round-trip, no extra Op nodes for
                // the executor to walk. Acts as the kernel-of-record for
                // L2 on MLX.
                //
                // `scalar_input_mask` is intentionally ignored here:
                // MLX's lazy eval natively broadcasts `[1]`-shape arrays
                // against larger ones in element-wise ops, so scalar-
                // broadcast inputs flow through the chain without any
                // explicit per-operand handling. The mask exists for the
                // kernel-launch backends (CPU/Metal/wgpu/CUDA/ROCm)
                // whose interpreted-chain kernels need the explicit hint
                // to swap their per-output indexing for element-0 reads.
                let n_in = *num_inputs as usize;
                if node.inputs.len() != n_in {
                    return Err(MlxError(format!(
                        "ElementwiseRegion: declared {n_in} inputs but node has {}",
                        node.inputs.len()
                    )));
                }
                let inputs: Vec<&Array> = node
                    .inputs
                    .iter()
                    .map(|&id| lookup(&env, id))
                    .collect::<Result<_, _>>()?;
                let mut steps: Vec<Array> = Vec::with_capacity(chain.len());
                fn resolve<'a>(
                    op: ChainOperand,
                    inputs: &'a [&Array],
                    steps: &'a [Array],
                ) -> Result<&'a Array, MlxError> {
                    match op {
                        ChainOperand::Input(i) => {
                            let i = i as usize;
                            inputs.get(i).copied().ok_or_else(|| {
                                MlxError(format!(
                                    "ElementwiseRegion: ChainOperand::Input({i}) \
                                 out of range (have {} inputs)",
                                    inputs.len()
                                ))
                            })
                        }
                        ChainOperand::Step(i) => {
                            let i = i as usize;
                            steps.get(i).ok_or_else(|| {
                                MlxError(format!(
                                    "ElementwiseRegion: ChainOperand::Step({i}) \
                                 references step not yet produced (have {} steps)",
                                    steps.len()
                                ))
                            })
                        }
                    }
                }
                for step in chain {
                    let arr = match step {
                        ChainStep::Activation(act, x_op) => {
                            let x = resolve(*x_op, &inputs, &steps)?;
                            match act {
                                Activation::Gelu | Activation::GeluApprox => ops::gelu(x)?,
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
                            }
                        }
                        ChainStep::Cast(to, x_op) => {
                            let x = resolve(*x_op, &inputs, &steps)?;
                            ops::cast(x, *to)?
                        }
                        ChainStep::Binary(bop, l_op, r_op) => {
                            let a = resolve(*l_op, &inputs, &steps)?;
                            let b = resolve(*r_op, &inputs, &steps)?;
                            match bop {
                                BinaryOp::Add => ops::add(a, b)?,
                                BinaryOp::Mul => ops::mul(a, b)?,
                                BinaryOp::Sub => ops::sub(a, b)?,
                                BinaryOp::Div => ops::div(a, b)?,
                                BinaryOp::Max => ops::max(a, b)?,
                                BinaryOp::Min => ops::min(a, b)?,
                                BinaryOp::Pow => ops::pow(a, b)?,
                            }
                        }
                        ChainStep::Compare(cop, l_op, r_op) => {
                            let a = resolve(*l_op, &inputs, &steps)?;
                            let b = resolve(*r_op, &inputs, &steps)?;
                            match cop {
                                CmpOp::Eq => ops::eq(a, b)?,
                                CmpOp::Ne => ops::ne(a, b)?,
                                CmpOp::Lt => ops::lt(a, b)?,
                                CmpOp::Le => ops::le(a, b)?,
                                CmpOp::Gt => ops::gt(a, b)?,
                                CmpOp::Ge => ops::ge(a, b)?,
                            }
                        }
                        ChainStep::Where(c_op, t_op, f_op) => {
                            let c = resolve(*c_op, &inputs, &steps)?;
                            let t = resolve(*t_op, &inputs, &steps)?;
                            let f = resolve(*f_op, &inputs, &steps)?;
                            ops::select(c, t, f)?
                        }
                    };
                    steps.push(arr);
                }
                steps.pop().ok_or_else(|| {
                    MlxError("ElementwiseRegion: empty chain has no output".into())
                })?
            }
            Op::Activation(act) => {
                let x = lookup(&env, node.inputs[0])?;
                match act {
                    Activation::Gelu | Activation::GeluApprox => ops::gelu(x)?,
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
                }
            }
            Op::Cast { to } => {
                let x = lookup(&env, node.inputs[0])?;
                ops::cast(x, *to)?
            }
            Op::Softmax { axis } => {
                let x = lookup(&env, node.inputs[0])?;
                ops::softmax(x, *axis)?
            }
            Op::LayerNorm { eps, .. } => {
                let x = lookup(&env, node.inputs[0])?;
                let g = lookup(&env, node.inputs[1])?;
                let b = if node.inputs.len() >= 3 {
                    Some(lookup(&env, node.inputs[2])?)
                } else {
                    None
                };
                ops::layer_norm(x, g, b, *eps)?
            }
            Op::Reshape { new_shape } => {
                let x = lookup(&env, node.inputs[0])?;
                let s: Vec<i32> = new_shape.iter().map(|&d| d as i32).collect();
                ops::reshape(x, &s)?
            }
            Op::Transpose { perm } => {
                let x = lookup(&env, node.inputs[0])?;
                let p: Vec<i32> = perm.iter().map(|&d| d as i32).collect();
                ops::transpose(x, &p)?
            }
            Op::Narrow { axis, start, len } => {
                let x = lookup(&env, node.inputs[0])?;
                let in_shape: Vec<i32> = node_input_shape(graph, node.inputs[0]);
                let mut s_start = vec![0i32; in_shape.len()];
                let mut s_stop = in_shape.clone();
                s_start[*axis] = *start as i32;
                s_stop[*axis] = (*start + *len) as i32;
                ops::slice(x, &s_start, &s_stop)?
            }
            Op::Concat { axis } => {
                let inputs: Vec<&Array> = node
                    .inputs
                    .iter()
                    .map(|&id| lookup(&env, id))
                    .collect::<Result<_, _>>()?;
                ops::concat(&inputs, *axis as i32)?
            }
            Op::Expand { target_shape } => {
                let x = lookup(&env, node.inputs[0])?;
                let s: Vec<i32> = target_shape.iter().map(|&d| d as i32).collect();
                ops::broadcast_to(x, &s)?
            }
            Op::Gather { axis } => {
                let x = lookup(&env, node.inputs[0])?;
                let idx = lookup(&env, node.inputs[1])?;
                ops::take(x, idx, *axis as i32)?
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
            Op::Cumsum { axis, exclusive } => {
                let x = lookup(&env, node.inputs[0])?;
                ops::cumsum(x, *axis, *exclusive)?
            }
            Op::RmsNorm { eps, .. } => {
                let x = lookup(&env, node.inputs[0])?;
                let g = lookup(&env, node.inputs[1])?;
                ops::rms_norm(x, g, *eps)?
            }
            Op::Attention {
                num_heads,
                head_dim,
                mask_kind,
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
                let scale = 1.0 / (hd as f32).sqrt();

                // Detect layout from rank.
                let need_split = q_shape.len() == 3;
                let to_bhsd = |t: &Array, sh: &[i32]| -> Result<Array, MlxError> {
                    if sh.len() == 4 {
                        return t.clone_handle();
                    }
                    // [B, S, H*D] → [B, S, H, D] → [B, H, S, D]
                    let b = sh[0];
                    let s = sh[1];
                    let r = ops::reshape(t, &[b, s, nh, hd])?;
                    ops::transpose(&r, &[0, 2, 1, 3])
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
                        (MlxMask::Custom, Some(normalize_mask(&m_cast, &m_shape)?), None)
                    }
                };
                let m_ref: Option<&Array> = mask.as_ref().or(mask_owned.as_ref());
                let attn_out = ops::attention(&q, &k, &v, scale, mask_kind_ffi, m_ref)?;

                if need_split {
                    // [B, H, S, D] → [B, S, H, D] → [B, S, H*D]
                    let b = q_shape[0];
                    let s = q_shape[1];
                    let bsd = ops::transpose(&attn_out, &[0, 2, 1, 3])?;
                    ops::reshape(&bsd, &[b, s, nh * hd])?
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
                let biased = ops::add(&mm, b)?;
                match activation {
                    None => biased,
                    Some(Activation::Gelu) | Some(Activation::GeluApprox) => ops::gelu(&biased)?,
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
                }
            }
            Op::FusedResidualLN { has_bias, eps } => {
                let x = lookup(&env, node.inputs[0])?;
                let r = lookup(&env, node.inputs[1])?;
                let summed = ops::add(x, r)?;
                let summed = if *has_bias {
                    let bias = lookup(&env, node.inputs[2])?;
                    ops::add(&summed, bias)?
                } else {
                    summed
                };
                let (g_idx, b_idx) = if *has_bias { (3, 4) } else { (2, 3) };
                let g = lookup(&env, node.inputs[g_idx])?;
                let b = lookup(&env, node.inputs[b_idx])?;
                ops::layer_norm(&summed, g, Some(b), *eps)?
            }
            Op::Rope { head_dim } => {
                // Standard transformer RoPE applied per (batch, seq,
                // head): for each token at sequence position `s`,
                // rotate every head's `head_dim` vector using
                // `cos[s, :]` / `sin[s, :]`. Same position for all
                // heads (matches candle's `rotary_emb.apply` and HF
                // transformers `apply_rotary_pos_emb`).
                //
                // Accepted layouts (axis containing `seq` shown bold):
                //   - `[B, **S**, H*D]` — rlx-models packed multi-head;
                //     reshape to `[B, S, H, D]`, rotate over axis 1.
                //   - `[B, H, **S**, D]` — candle-style; rotate over axis 2.
                //   - `[..., **S**, D]` last dim == head_dim — rotate over axis n-2.
                //   - `[..., **S**, head_dim+tail]` — partial-dim;
                //     rotate first `head_dim`, pass tail through.
                let x = lookup(&env, node.inputs[0])?;
                let cos = lookup(&env, node.inputs[1])?;
                let sin = lookup(&env, node.inputs[2])?;

                let x_shape = node_input_shape(graph, node.inputs[0]);
                let n = x_shape.len();
                if n < 2 {
                    return Err(MlxError("Rope: x must be rank ≥ 2".into()));
                }
                if head_dim % 2 != 0 {
                    return Err(MlxError(format!("Rope: head_dim {head_dim} must be even")));
                }
                let last = *x_shape.last().unwrap() as usize;
                if last < *head_dim {
                    return Err(MlxError(format!(
                        "Rope: x last dim {last} < head_dim {head_dim}"
                    )));
                }
                let hd = *head_dim as i32;
                let half = (head_dim / 2) as i32;

                let heads_in_last = (last / *head_dim) as i32;
                let multi_head_packed = heads_in_last > 1 && last % *head_dim == 0 && n >= 3;
                let has_tail = last % *head_dim != 0;

                // ── Helper: rotate `x_rot` of shape `rot_shape`,
                // broadcasting cos/sin across all axes except `seq_axis`
                // (which selects the cos/sin row). Returns same-shape.
                let rotate = |x_rot: &Array,
                              rot_shape: &[i32],
                              seq_axis: usize|
                 -> Result<Array, MlxError> {
                    let rn = rot_shape.len();
                    let seq_v = rot_shape[seq_axis];

                    let cos_seq = ops::slice(cos, &[0, 0], &[seq_v, half])?;
                    let sin_seq = ops::slice(sin, &[0, 0], &[seq_v, half])?;

                    // Broadcast cos/sin: [1, ..., 1, seq, 1, ..., 1, half]
                    // where `seq` is at position `seq_axis` and `half`
                    // is at the last axis.
                    let mut bshape = vec![1i32; rn];
                    bshape[seq_axis] = seq_v;
                    bshape[rn - 1] = half;
                    let cos_b = ops::reshape(&cos_seq, &bshape)?;
                    let sin_b = ops::reshape(&sin_seq, &bshape)?;

                    // Split last dim into halves.
                    let mut x1_stop = rot_shape.to_vec();
                    x1_stop[rn - 1] = half;
                    let x1 = ops::slice(x_rot, &vec![0i32; rn], &x1_stop)?;
                    let mut x2_start = vec![0i32; rn];
                    x2_start[rn - 1] = half;
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
                    // Rotate first head_dim; concat the unrotated tail.
                    let mut rot_stop = x_shape.clone();
                    rot_stop[n - 1] = hd;
                    let rot = ops::slice(x, &vec![0i32; n], &rot_stop)?;
                    let mut tail_start = vec![0i32; n];
                    tail_start[n - 1] = hd;
                    let tail = ops::slice(x, &tail_start, &x_shape)?;
                    let mut rot_shape = x_shape.clone();
                    rot_shape[n - 1] = hd;
                    let y_rot = rotate(&rot, &rot_shape, n - 2)?;
                    ops::concat(&[&y_rot, &tail], (n - 1) as i32)?
                } else if multi_head_packed {
                    // [B, S, H*D] → [B, S, H, D]. Seq is at axis n-2
                    // of the ORIGINAL shape == axis n-2 of the split
                    // shape too (because we only split the last axis).
                    let mut split_shape = x_shape.clone();
                    split_shape[n - 1] = heads_in_last;
                    split_shape.push(hd);
                    let x_split = ops::reshape(x, &split_shape)?;
                    // seq is at axis n-2 of x_shape == axis n-2 of split_shape
                    let y_split = rotate(&x_split, &split_shape, n - 2)?;
                    ops::reshape(&y_split, &x_shape)?
                } else {
                    // last == head_dim. seq at axis n-2.
                    rotate(x, &x_shape, n - 2)?
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
                let indices = lookup(&env, node.inputs[1])?;
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
                let zero_target = crate::array::Array::from_f32_slice(
                    &zeros, &out_shape_usize, DType::F32,
                )?;
                ops::scatter_add(&zero_target, indices, updates, 0)?
            }
            Op::GroupedMatMul => {
                // Inputs: [input, weight, expert_idx].
                let x = lookup(&env, node.inputs[0])?;
                let w = lookup(&env, node.inputs[1])?;
                let i = lookup(&env, node.inputs[2])?;
                ops::gather_mm(x, w, i)?
            }
            Op::DequantMatMul { scheme } => {
                // Inputs: [x, w_q, scale, zp]. Map to MLX's
                // quantized_matmul. The bit-width and group-size come
                // from the rlx QuantScheme.
                let x = lookup(&env, node.inputs[0])?;
                let wq = lookup(&env, node.inputs[1])?;
                let s = lookup(&env, node.inputs[2])?;
                let zp = lookup(&env, node.inputs[3])?;
                let (bits, gs) = quant_scheme_to_mlx(scheme)?;
                ops::quantized_matmul(x, wq, s, Some(zp), /*transpose=*/ true, gs, bits)?
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
                let to_h = |t: Array| -> Result<Array, MlxError> {
                    let r = ops::reshape(&t, &[batch, seq, nh, hd])?;
                    ops::transpose(&r, &[0, 2, 1, 3])
                };
                let q = to_h(q)?;
                let k = to_h(k)?;
                let v = to_h(v)?;
                let scale = 1.0 / (hd as f32).sqrt();
                // Promote mask to Q's dtype (AutoMixed casts Q/K/V
                // but not mask leaves — see Op::Attention site above).
                let h_dtype = graph.node(node.inputs[0]).shape.dtype();
                let mask_owned;
                let mask_ref: &Array = if h_dtype != DType::F32 {
                    mask_owned = ops::cast(mask, h_dtype)?;
                    &mask_owned
                } else {
                    mask
                };
                let attn = ops::attention(
                    &q,
                    &k,
                    &v,
                    scale,
                    crate::ffi::MlxMask::Custom,
                    Some(mask_ref),
                )?;
                let attn = ops::transpose(&attn, &[0, 2, 1, 3])?;
                let attn = ops::reshape(&attn, &[batch, seq, inner])?;
                let attn_out = ops::matmul(&attn, out_w)?;
                let attn_out = maybe_add(attn_out, out_b)?;

                // --- Residual + LayerNorm 1 ---
                let pre1 = ops::add(hidden, &attn_out)?;
                let h1 = ops::layer_norm(&pre1, ln1_g, ln1_b, *eps1)?;

                // --- FFN: activation(h1 @ fc1_w [+ fc1_b]) @ fc2_w [+ fc2_b] ---
                let ffn1 = ops::matmul(&h1, fc1_w)?;
                let ffn1 = maybe_add(ffn1, fc1_b)?;
                let ffn1 = match activation {
                    Activation::Gelu | Activation::GeluApprox => ops::gelu(&ffn1)?,
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
                };
                let ffn2 = ops::matmul(&ffn1, fc2_w)?;
                let ffn_out = maybe_add(ffn2, fc2_b)?;

                // --- Residual + LayerNorm 2 ---
                let pre2 = ops::add(&h1, &ffn_out)?;
                ops::layer_norm(&pre2, ln2_g, ln2_b, *eps2)?
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
                let batch = h_shape[0];
                let seq = h_shape[1];
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

                // 3. reshape to [B, S, H, D] then transpose to [B, H, S, D]
                let to_h = |t: Array| -> Result<Array, MlxError> {
                    let r = ops::reshape(&t, &[batch, seq, nh, hd])?;
                    ops::transpose(&r, &[0, 2, 1, 3])
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
                        let cos_seq = ops::slice(cos, &[0, 0], &[seq, half])?;
                        let sin_seq = ops::slice(sin, &[0, 0], &[seq, half])?;
                        let bshape = [1, 1, seq, half];
                        let cos_b = ops::reshape(&cos_seq, &bshape)?;
                        let sin_b = ops::reshape(&sin_seq, &bshape)?;
                        let x1 = ops::slice(x, &[0, 0, 0, 0], &[batch, nh, seq, half])?;
                        let x2 = ops::slice(x, &[0, 0, 0, half], &[batch, nh, seq, hd])?;
                        let y1 = ops::sub(&ops::mul(&x1, &cos_b)?, &ops::mul(&x2, &sin_b)?)?;
                        let y2 = ops::add(&ops::mul(&x2, &cos_b)?, &ops::mul(&x1, &sin_b)?)?;
                        ops::concat(&[&y1, &y2], 3)
                    };
                    q = do_rope(&q)?;
                    k = do_rope(&k)?;
                }

                // 5. SDPA with custom mask
                let scale = 1.0 / (hd as f32).sqrt();
                // Mask must promote to Q dtype (AutoMixed promotes
                // Q/K/V but not mask leaves).
                let q_dtype = graph.node(node.inputs[h_idx]).shape.dtype();
                let mask_owned;
                let mask_ref: &Array = if q_dtype != DType::F32 {
                    mask_owned = ops::cast(mask, q_dtype)?;
                    &mask_owned
                } else {
                    mask
                };
                let attn_out = ops::attention(
                    &q,
                    &k,
                    &v_h,
                    scale,
                    crate::ffi::MlxMask::Custom,
                    Some(mask_ref),
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
            Op::FusedSwiGLU { cast_to } => {
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
                let then_outs = lower_subgraph(then_branch, &captures, params, params_typed)?;
                let else_outs = lower_subgraph(else_branch, &captures, params, params_typed)?;
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
                    let cond_outs = lower_subgraph(cond, &captures, params, params_typed)?;
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

                    let body_outs = lower_subgraph(body, &captures, params, params_typed)?;
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
            Op::Scan {
                body,
                length,
                save_trajectory,
                num_xs,
                num_bcast,
                num_checkpoints: _,
            } => {
                // Generic loop-unrolled scan. MLX has no native scan
                // primitive, so we lower it the same way SelectiveScan
                // below does: walk t = 0..length, lower the body once
                // per iter with the previous step's carry as the first
                // capture, and (if save_trajectory) collect the
                // outputs into a stacked `[length, *carry]` tensor.
                //
                // Inputs layout (per Op::Scan IR doc):
                //   [init, bcast_0..bcast_{B-1}, x_t_0..x_t_{X-1}]
                // The body's Op::Inputs in declaration order are:
                //   [carry, bcast_0..bcast_{B-1}, x_at_t_0..x_at_t_{X-1}]
                //
                // For static `length`, the unrolled trace lives in
                // MLX's lazy graph and gets compiled once on first
                // dispatch — same amortization the SelectiveScan
                // path relies on.
                let init = lookup(&env, node.inputs[0])?;
                let bcasts: Vec<&Array> = (0..*num_bcast as usize)
                    .map(|i| lookup(&env, node.inputs[1 + i]).map(|a| a))
                    .collect::<Result<Vec<_>, _>>()?;
                let xs: Vec<&Array> = (0..*num_xs as usize)
                    .map(|i| lookup(&env, node.inputs[1 + *num_bcast as usize + i]))
                    .collect::<Result<Vec<_>, _>>()?;

                // Carry shape (used for both per-iter trial reshape
                // and the final stacked-trajectory shape).
                let carry_shape: Vec<i32> = init.shape()?.iter().map(|d| *d as i32).collect();
                let carry_rank = carry_shape.len();

                let mut carry: Array = init.clone_handle()?;
                let mut traj_slices: Vec<Array> = if *save_trajectory {
                    Vec::with_capacity(*length as usize)
                } else {
                    Vec::new()
                };

                for t in 0..(*length as i32) {
                    // Build per-iter captures: carry, bcasts, xs[t].
                    let mut captures: Vec<Array> = Vec::with_capacity(1 + bcasts.len() + xs.len());
                    captures.push(carry.clone_handle()?);
                    for b in &bcasts {
                        captures.push(b.clone_handle()?);
                    }
                    for x in &xs {
                        // x has shape [length, *per_step]. Slice axis-0
                        // row t and squeeze that axis to feed body.
                        let mut start = vec![t];
                        let mut stop = vec![t + 1];
                        let x_shape = x.shape()?;
                        for i in 1..x_shape.len() {
                            start.push(0);
                            stop.push(x_shape[i] as i32);
                        }
                        let row = ops::slice(x, &start, &stop)?;
                        let per_step_dims: Vec<i32> =
                            x_shape[1..].iter().map(|d| *d as i32).collect();
                        let row_squeezed = ops::reshape(&row, &per_step_dims)?;
                        captures.push(row_squeezed);
                    }
                    let capture_refs: Vec<&Array> = captures.iter().collect();
                    let body_outs = lower_subgraph(body, &capture_refs, &params, &params_typed)?;
                    if body_outs.is_empty() {
                        return Err(MlxError("Op::Scan: body produced no outputs".into()));
                    }
                    // First output is next carry.
                    carry = body_outs.into_iter().next().unwrap();

                    if *save_trajectory {
                        // Reshape to add a leading length-1 axis so we
                        // can concat into [length, *carry].
                        let mut row_shape: Vec<i32> = vec![1];
                        row_shape.extend_from_slice(&carry_shape);
                        traj_slices.push(ops::reshape(&carry, &row_shape)?);
                    }
                }

                if *save_trajectory {
                    let refs: Vec<&Array> = traj_slices.iter().collect();
                    ops::concat(&refs, 0)?
                } else {
                    let _ = carry_rank;
                    carry
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

            Op::Custom { name, attrs, .. } => {
                // Dispatch through the registered MlxKernel. Each
                // input is looked up as an MLX Array (already
                // computed by earlier iterations); the kernel
                // produces a fresh Array for this node, which feeds
                // any consumers downstream. The kernel is free to
                // compose existing MLX `Array` ops (staying in the
                // lazy graph for `mlx::compile`'s benefit) or to
                // call into `mlx::fast::metal_kernel` for raw MSL.
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

            other => {
                return unsupported(format!("{other:?}"));
            }
        };

        env.insert(id, arr);
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

/// Build the MLX graph and return the array handles for the graph's
/// declared outputs (in `graph.outputs` order).
///
/// Host-data variant: leaves are constructed from f32 input/param
/// buffers. The compile path uses [`lower_with_env`] directly with a
/// pre-built leaf binding instead.
pub fn lower_and_run(
    graph: &Graph,
    params: &HashMap<String, Vec<f32>>,
    inputs: &HashMap<String, Vec<f32>>,
    mode: MlxMode,
) -> Result<Vec<Array>, MlxError> {
    // PLAN L3: coarse Perfetto span around the whole MLX lower+eval
    // pass. MLX is lazy (graph build → eval); per-node spans would
    // measure build time, not GPU compute. One span per run() is the
    // honest cross-backend marker for an MLX execution.
    let _perf = rlx_ir::perfetto::TraceSpan::new("lower_and_run", "mlx");
    lower_and_run_typed(
        graph,
        params,
        &HashMap::new(),
        inputs,
        &HashMap::new(),
        mode,
    )
}

/// Same as `lower_and_run` but accepts parallel typed maps. When a
/// name appears in `params_typed` / `inputs_typed`, the typed bytes
/// are bound directly via `Array::from_bytes` (no f32 round-trip).
/// Existing f32 callers thread empty maps through `lower_and_run`.
///
/// Dynamic shapes (`Dim::Dynamic`) get resolved here too: we infer
/// symbol→size bindings from the actual data lengths of each Input,
/// rebuild the graph with bound shapes, and lower against the
/// concretized version. MLX's per-shape trace caching handles the
/// re-shape efficiency on subsequent calls.
pub fn lower_and_run_typed(
    graph: &Graph,
    params: &HashMap<String, Vec<f32>>,
    params_typed: &HashMap<String, (Vec<u8>, DType)>,
    inputs: &HashMap<String, Vec<f32>>,
    inputs_typed: &HashMap<String, (Vec<u8>, DType)>,
    mode: MlxMode,
) -> Result<Vec<Array>, MlxError> {
    lower_and_run_typed_with_extent(
        graph,
        params,
        params_typed,
        inputs,
        inputs_typed,
        mode,
        /*active_extent=*/ None,
    )
}

/// Variant of [`lower_and_run_typed`] honoring a PLAN L1 active-extent
/// hint (`Some((actual, upper))`). When set AND the graph passes
/// [`is_safe_for_active_extent`], every input leaf whose outer dim
/// equals `upper` is sliced along axis 0 to `actual` before
/// composition. MLX's lazy eval propagates the smaller shapes through
/// the rest of the trace, so most ops just produce smaller outputs
/// naturally — no per-op kernel scaling needed. Falls back to the full
/// extent when the hint is `None` or the graph contains an unsafe op.
pub fn lower_and_run_typed_with_extent(
    graph: &Graph,
    params: &HashMap<String, Vec<f32>>,
    params_typed: &HashMap<String, (Vec<u8>, DType)>,
    inputs: &HashMap<String, Vec<f32>>,
    inputs_typed: &HashMap<String, (Vec<u8>, DType)>,
    mode: MlxMode,
    active_extent: Option<(usize, usize)>,
) -> Result<Vec<Array>, MlxError> {
    // Resolve dynamic dims if any. The graph as-given may have
    // Dim::Dynamic entries in Input shapes (and propagated through
    // inferred internal shapes). We gather concrete bindings from the
    // supplied data and rebuild the graph with every shape bound.
    let resolved_owner;
    let graph: &Graph = if has_dynamic_dims(graph) {
        let binding = collect_bindings(graph, inputs, inputs_typed)?;
        resolved_owner = resolve_graph(graph, &binding);
        &resolved_owner
    } else {
        graph
    };

    let order = leaf_order(graph);
    let mut env: HashMap<NodeId, Array> = HashMap::with_capacity(graph.nodes().len());
    for (id, _key) in &order {
        env.insert(
            *id,
            build_leaf_for(graph, *id, params, inputs, params_typed, inputs_typed)?,
        );
    }

    // PLAN L1 active-extent: when hinted + safe, slice each Input leaf
    // along axis 0 from `upper` to `actual`. Only Input leaves get
    // sliced — Param/Constant tensors don't carry a batch dim that
    // matches the bucket axis. MLX's lazy graph propagates the smaller
    // shapes naturally through downstream element-wise / reduction-on-
    // inner / matmul ops.
    if let Some((actual, upper)) = active_extent
        && actual < upper
        && is_safe_for_active_extent(graph, upper)
    {
        for (id, _key) in &order {
            let node = graph.node(*id);
            if !matches!(node.op, Op::Input { .. }) {
                continue;
            }
            let dims = node.shape.dims();
            if dims.is_empty() {
                continue;
            }
            let outer = match dims[0] {
                Dim::Static(d) => d,
                _ => continue,
            };
            if outer != upper {
                continue;
            }
            let leaf = env.get(id).unwrap();
            let in_shape: Vec<usize> = dims.iter().map(|d| d.unwrap_static()).collect();
            let mut start = vec![0i32; in_shape.len()];
            let mut stop: Vec<i32> = in_shape.iter().map(|&d| d as i32).collect();
            start[0] = 0;
            stop[0] = actual as i32;
            let sliced = ops::slice(leaf, &start, &stop)?;
            env.insert(*id, sliced);
        }
    }

    // Eager mode wants per-op eval for debugging; the env-walker's
    // construction is pure (no eval), so we trigger it here against
    // outputs after lowering. For interleaved per-op eval we'd need
    // a separate walker variant — currently no caller asks for that.
    let outs = lower_with_env(graph, env, params, params_typed)?;

    let refs: Vec<&Array> = outs.iter().collect();
    match mode {
        MlxMode::Eager => {
            // Eval outputs one at a time. Functionally equivalent to
            // per-op eval since outputs are dependency roots; only
            // the failure-localization aspect is weaker.
            for o in &outs {
                eval(&[o])?;
            }
        }
        MlxMode::Lazy => {
            eval(&refs)?;
        }
        MlxMode::AsyncCommit => {
            async_eval(&refs)?;
        }
        MlxMode::Compiled => {
            // Compiled mode shouldn't reach this code path —
            // backend.rs dispatches to run_compiled before calling
            // here. If we did get here it means the host-data path
            // was used, so just eval normally (correct, just misses
            // the trace-cache benefit).
            eval(&refs)?;
        }
    }

    Ok(outs)
}

/// PLAN L1 — true when the graph is safe for active-extent dispatch
/// at the given `upper` extent. Conservative: rejects ops that either
/// (a) hardcode the outer dim in their parameters
/// (`Op::Reshape { new_shape }` / `Op::Expand { target_shape }` / etc.
/// when those shapes mention `upper`), (b) operate along axis 0
/// (`Op::Reduce` / `Op::Cumsum` / `Op::Concat` / `Op::Narrow` with
/// axis 0; `Op::Transpose` whose perm permutes axis 0), or (c) have
/// outer-dim semantics that can't be honored by simply slicing the
/// input (`Op::Gather` / `Op::ScatterAdd` / `Op::Sample` / `Op::TopK`
/// / `Op::SelectiveScan` / `Op::GroupedMatMul` / `Op::Pool` /
/// `Op::Conv` / `Op::FusedTransformerLayer` / sub-graph control flow).
pub fn is_safe_for_active_extent(graph: &Graph, upper: usize) -> bool {
    let upper_i64 = upper as i64;
    for node in graph.nodes() {
        match &node.op {
            // Leaves & element-wise ops: always safe (slicing inputs
            // produces correctly-sized intermediates via lazy eval).
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => {}
            Op::Activation(_)
            | Op::Cast { .. }
            | Op::Binary(_)
            | Op::Compare(_)
            | Op::Where
            | Op::ElementwiseRegion { .. } => {}
            // Per-row normalizations: operate on inner axes, batch is
            // pass-through. Safe.
            Op::Softmax { axis: _ } | Op::LayerNorm { .. } | Op::RmsNorm { .. } => {}
            // Rope / Attention / matmul: batch in outer dim, computation
            // on inner axes. Safe by construction.
            Op::Rope { .. }
            | Op::Attention { .. }
            | Op::MatMul
            | Op::DotGeneral { .. }
            | Op::FusedMatMulBiasAct { .. }
            | Op::FusedSwiGLU { .. }
            | Op::FusedResidualLN { .. }
            | Op::FusedAttentionBlock { .. } => {}
            // DequantMatMul / LoraMatMul follow MatMul's batch-outer
            // contract.
            Op::DequantMatMul { .. } | Op::LoraMatMul { .. } => {}
            // Real INT8 ops: not lowered on MLX yet — train/quantize
            // on CPU, run inference there. Reject so the dispatch
            // surfaces a clear error.
            Op::QMatMul { .. } | Op::QConv2d { .. } => return false,
            // Reduce / Cumsum: safe iff the operation doesn't touch
            // axis 0.
            Op::Reduce { axes, .. } => {
                if axes.contains(&0) {
                    return false;
                }
            }
            Op::Cumsum { axis, .. } => {
                if *axis == 0 {
                    return false;
                }
            }
            // Concat: safe iff axis != 0 (concatenating along the batch
            // axis would mix batches across the slice boundary).
            Op::Concat { axis } => {
                if *axis == 0 {
                    return false;
                }
            }
            // Narrow on axis 0 changes the bucket itself — unsafe.
            Op::Narrow { axis, .. } => {
                if *axis == 0 {
                    return false;
                }
            }
            // Transpose is safe iff perm[0] == 0 (axis 0 stays put;
            // inner axes can permute freely).
            Op::Transpose { perm } => {
                if perm.first().copied() != Some(0) {
                    return false;
                }
            }
            // Reshape / Expand: reject if their target shape mentions
            // `upper` — that hardcoded dim won't survive the slice.
            Op::Reshape { new_shape } => {
                if new_shape.contains(&upper_i64) {
                    return false;
                }
            }
            Op::Expand { target_shape } => {
                if target_shape.contains(&upper_i64) {
                    return false;
                }
            }
            // Gather operates on axis 0 of its lookup table; the
            // batch contract isn't compatible with bucket slicing.
            Op::Gather { .. } => return false,
            // Conservatively unsafe — these have batch-touching
            // semantics (or sub-graph leaves) that the slice trick
            // doesn't handle.
            Op::ScatterAdd
            | Op::Sample { .. }
            | Op::TopK { .. }
            | Op::SelectiveScan { .. }
            | Op::GroupedMatMul
            | Op::Pool { .. }
            | Op::Conv { .. }
            | Op::FusedTransformerLayer { .. }
            | Op::DenseSolve
            | Op::Custom { .. }
            | Op::If { .. }
            | Op::While { .. } => return false,
            // Quantization: not lowered on MLX yet — train/quantize on
            // CPU, run inference on the dequantized fp32/fp16 path.
            Op::Quantize { .. }
            | Op::Dequantize { .. }
            | Op::FakeQuantize { .. }
            | Op::FakeQuantizeBackward { .. }
            | Op::FakeQuantizeLSQ { .. }
            | Op::FakeQuantizeLSQBackwardX { .. }
            | Op::FakeQuantizeLSQBackwardScale { .. } => return false,
            // Backward / training ops: active-extent dispatch is an
            // inference-only batch-bucketing optimization, so the safe
            // default for any training-graph node is `false` regardless
            // of whether MLX can lower it. Tier 1 (Relu/Activation/SCE/
            // LayerNorm backward) DOES lower on MLX — see the Op match
            // in `lower_with_env` — it's just never relevant here.
            Op::ReluBackward
            | Op::ActivationBackward { .. }
            | Op::MaxPool2dBackward { .. }
            | Op::Conv2dBackwardInput { .. }
            | Op::Conv2dBackwardWeight { .. }
            | Op::SoftmaxCrossEntropyWithLogits
            | Op::SoftmaxCrossEntropyBackward
            | Op::LayerNormBackwardInput { .. }
            | Op::LayerNormBackwardGamma { .. } => return false,
            Op::Scan { .. }
            | Op::ScanBackward { .. }
            | Op::ScanBackwardXs { .. }
            | Op::BatchedDenseSolve => return false,
            // CustomFn is opaque to active-extent analysis — the body
            // graph may have arbitrary internal structure. Fall back
            // to full extent for graphs that contain them. (Op::Custom
            // is already rejected in the conservatively-unsafe arm.)
            Op::CustomFn { .. } => return false,
            // FFT not yet lowered to MLX — pin to Device::Cpu for now.
            Op::Fft { .. } => return false,
            // C64 ops are CPU-only today; pin to Device::Cpu.
            Op::ComplexNormSq | Op::ComplexNormSqBackward | Op::Conjugate => return false,
            // Stateful RNN op — conservatively pin to CPU.
            Op::GatedDeltaNet { .. } => return false,
        }
    }
    true
}

/// True if any node in the graph has a Dim::Dynamic entry. Cheap
/// scan; lets us skip the resolve step for fully-static graphs.
fn has_dynamic_dims(graph: &Graph) -> bool {
    graph
        .nodes()
        .iter()
        .any(|n| n.shape.dims().iter().any(|d| !d.is_static()))
}

/// Walk the graph, infer concrete sizes for each `Dim::Dynamic` symbol
/// from the supplied input data. Each Input with exactly one dynamic
/// dim contributes a binding (data_nelems / static_dim_product). The
/// inference is conservative: if a single input has multiple dynamic
/// dims it errors, since the data length is one number and we can't
/// distribute it across multiple unknowns. Multi-dynamic inputs would
/// need an externally-supplied DimBinding; out of scope today.
fn collect_bindings(
    graph: &Graph,
    inputs: &HashMap<String, Vec<f32>>,
    inputs_typed: &HashMap<String, (Vec<u8>, DType)>,
) -> Result<DimBinding, MlxError> {
    let mut binding = DimBinding::new();
    for node in graph.nodes() {
        if let Op::Input { name } = &node.op {
            // Element count from the supplied data (typed wins).
            let n_elems = if let Some((bytes, dt)) = inputs_typed.get(name) {
                let elem_size = dt.size_bytes();
                if elem_size == 0 || bytes.len() % elem_size != 0 {
                    return Err(MlxError(format!(
                        "Input '{name}': typed bytes len {} not aligned to dtype size",
                        bytes.len()
                    )));
                }
                bytes.len() / elem_size
            } else if let Some(data) = inputs.get(name) {
                data.len()
            } else {
                // No data yet — skip; the leaf-build step will error
                // with a clearer "missing input" diagnostic.
                continue;
            };

            // Walk the shape's dims, accumulating the static product
            // and identifying the (single allowed) dynamic position.
            let mut static_prod: usize = 1;
            let mut dynamic_sym: Option<u32> = None;
            for d in node.shape.dims().iter() {
                match d {
                    Dim::Static(n) => {
                        static_prod = static_prod.checked_mul(*n).ok_or_else(|| {
                            MlxError(format!("Input '{name}': static dim product overflow"))
                        })?;
                    }
                    Dim::Dynamic(sym) => {
                        if dynamic_sym.is_some() {
                            return Err(MlxError(format!(
                                "Input '{name}' has multiple dynamic dims; \
                                 explicit DimBinding required"
                            )));
                        }
                        dynamic_sym = Some(*sym);
                    }
                }
            }

            if let Some(sym) = dynamic_sym {
                if static_prod == 0 {
                    return Err(MlxError(format!(
                        "Input '{name}': can't infer dynamic dim against zero \
                         static product"
                    )));
                }
                if n_elems % static_prod != 0 {
                    return Err(MlxError(format!(
                        "Input '{name}': nelems {n_elems} not divisible by \
                         static dim product {static_prod}"
                    )));
                }
                let dim_size = n_elems / static_prod;
                if let Some(prev) = binding.get(sym) {
                    if prev != dim_size {
                        return Err(MlxError(format!(
                            "Dynamic dim ?{sym}: inconsistent values across \
                             inputs ({prev} vs {dim_size})"
                        )));
                    }
                } else {
                    binding.set(sym, dim_size);
                }
            }
        }
    }
    Ok(binding)
}

/// Rebuild the graph with every Shape bound against `binding`. Node
/// IDs are preserved because we re-add ops in the same order via the
/// public `Graph::add_node` API (which allocates IDs sequentially).
fn resolve_graph(graph: &Graph, binding: &DimBinding) -> Graph {
    let mut fresh = Graph::new(&graph.name);
    for node in graph.nodes() {
        let bound: Shape = node.shape.bind(binding);
        // add_node preserves declaration order → preserves NodeIds.
        fresh.add_node(node.op.clone(), node.inputs.clone(), bound);
    }
    fresh.set_outputs(graph.outputs.clone());
    fresh
}

/// Build an additive `[seq_q, seq_k]` SDPA mask for sliding-window
/// attention: 0 where (ki <= qi) AND (qi - ki <= window), -inf
/// elsewhere. Constructed host-side as f32 because MLX SDPA wants
/// the mask added to the pre-softmax scores.
fn build_sliding_window_mask(s_q: i32, s_k: i32, window: i32) -> Result<Array, MlxError> {
    let neg_inf = f32::NEG_INFINITY;
    let s_q = s_q as usize;
    let s_k = s_k as usize;
    let w = window as i64;
    let mut buf = vec![neg_inf; s_q * s_k];
    for qi in 0..s_q {
        for ki in 0..s_k {
            let q = qi as i64;
            let k = ki as i64;
            // Causal + bounded distance.
            if k <= q && (q - k) <= w {
                buf[qi * s_k + ki] = 0.0;
            }
        }
    }
    Array::from_f32_slice(&buf, &[s_q, s_k], DType::F32)
}

fn quant_scheme_to_mlx(scheme: &rlx_ir::QuantScheme) -> Result<(i32, i32), MlxError> {
    use rlx_ir::QuantScheme as Q;
    let bits = scheme.bits_per_element() as i32;
    let gs = match scheme {
        Q::Int8Block { block_size } => *block_size as i32,
        Q::Int8BlockAsym { block_size } => *block_size as i32,
        Q::Int4Block { block_size } => *block_size as i32,
        other => {
            return Err(MlxError(format!(
                "MLX quantized_matmul: unsupported scheme {other:?}"
            )));
        }
    };
    Ok((bits, gs))
}

fn node_input_shape(graph: &Graph, id: NodeId) -> Vec<i32> {
    graph
        .node(id)
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static() as i32)
        .collect()
}

fn lookup(env: &HashMap<NodeId, Array>, id: NodeId) -> Result<&Array, MlxError> {
    env.get(&id)
        .ok_or_else(|| MlxError(format!("node {id:?} referenced before being lowered")))
}

fn unsupported<T>(what: String) -> Result<T, MlxError> {
    Err(MlxError(format!("MLX backend: unsupported op {what}")))
}

/// Zero-inflate a 4-D NHWC array along the two spatial axes by factors
/// (`sh`, `sw`). Produces a new array of shape
/// `[N, (H − 1)·sh + 1, (W − 1)·sw + 1, C]`, with original values at
/// strided positions and zeros between them.
///
/// Workaround for an MLX `conv_general` limitation: when `groups > 1`
/// AND `input_dilation > 1`, the kernel produces incorrect output. We
/// materialize the input dilation explicitly (reshape → pad → reshape
/// per spatial axis) so the downstream `conv_general` can run with
/// `input_dilation=[1,1]`.
fn inflate_spatial_2d(a: &Array, sh: usize, sw: usize) -> Result<Array, MlxError> {
    if sh == 1 && sw == 1 {
        return a.clone_handle();
    }
    let shape = a.shape()?;
    if shape.len() != 4 {
        return Err(MlxError(format!(
            "inflate_spatial_2d: expected rank-4 NHWC, got rank {}",
            shape.len()
        )));
    }
    let n = shape[0] as i32;
    let h = shape[1] as i32;
    let w = shape[2] as i32;
    let c = shape[3] as i32;

    let mut cur = a.clone_handle()?;
    if sh > 1 {
        let sh_i = sh as i32;
        // [N, H, W, C] → [N, H, 1, W, C] → pad axis 2 by (0, sh-1) →
        // [N, H, sh, W, C] → reshape [N, H*sh, W, C] → slice trailing
        // (sh-1) frames so dim becomes (H-1)*sh + 1.
        let r1 = ops::reshape(&cur, &[n, h, 1, w, c])?;
        let padded = ops::pad(
            &r1,
            /*low =*/ &[0, 0, 0, 0, 0],
            /*high=*/ &[0, 0, sh_i - 1, 0, 0],
            /*pad_value=*/ 0.0,
        )?;
        let merged = ops::reshape(&padded, &[n, h * sh_i, w, c])?;
        let new_h = (h - 1) * sh_i + 1;
        cur = ops::slice(&merged, &[0, 0, 0, 0], &[n, new_h, w, c])?;
    }
    if sw > 1 {
        let sw_i = sw as i32;
        let cur_shape = cur.shape()?;
        let cur_h = cur_shape[1] as i32;
        let r1 = ops::reshape(&cur, &[n, cur_h, w, 1, c])?;
        let padded = ops::pad(
            &r1,
            /*low =*/ &[0, 0, 0, 0, 0],
            /*high=*/ &[0, 0, 0, sw_i - 1, 0],
            /*pad_value=*/ 0.0,
        )?;
        let merged = ops::reshape(&padded, &[n, cur_h, w * sw_i, c])?;
        let new_w = (w - 1) * sw_i + 1;
        cur = ops::slice(&merged, &[0, 0, 0, 0], &[n, cur_h, new_w, c])?;
    }
    Ok(cur)
}

/// Map `bits` ∈ {8, 4, 2} to its quantization range `q_max`.
fn fq_q_max(bits: u8) -> Result<f32, MlxError> {
    match bits {
        8 => Ok(127.0),
        4 => Ok(7.0),
        2 => Ok(1.0),
        n => Err(MlxError(format!("FakeQuantize: unsupported bits {n}"))),
    }
}

/// PerBatch-style scale: per-channel `max(|x|) / q_max`, floored at
/// `1e-12` so dividing by it never blows up. Returned shape is
/// broadcast-compatible against `x` (via `keep_dim=true` on the reduce).
fn fq_scale_perbatch(
    x: &Array,
    x_shape: &[i32],
    axis: Option<usize>,
    q_max: f32,
    dtype: DType,
) -> Result<Array, MlxError> {
    let abs_x = ops::unary(x, MlxUnary::Abs)?;
    let reduce_axes: Vec<i32> = match axis {
        None => (0..x_shape.len() as i32).collect(),
        Some(c) => (0..x_shape.len() as i32)
            .filter(|&i| i != c as i32)
            .collect(),
    };
    let max_abs = ops::reduce(
        &abs_x,
        MlxReduce::Max,
        &reduce_axes,
        /*keep_dim=*/ true,
    )?;
    let q_max_arr = Array::from_f32_slice(&[q_max], &[1], dtype)?;
    let scale_unclamped = ops::div(&max_abs, &q_max_arr)?;
    let eps = Array::from_f32_slice(&[1e-12], &[1], dtype)?;
    ops::max(&scale_unclamped, &eps)
}

/// Build a broadcast-shaped scale tensor from a 1-D `state` (shape `[C]`
/// for per-channel; `[1]` for per-tensor) so it broadcasts against `x`.
fn fq_scale_from_state(
    state: &Array,
    x_shape: &[i32],
    axis: Option<usize>,
    dtype: DType,
) -> Result<Array, MlxError> {
    let eps = Array::from_f32_slice(&[1e-12], &[1], dtype)?;
    let clamped = ops::max(state, &eps)?;
    match axis {
        None => Ok(clamped),
        Some(c) => {
            let state_dim = state.shape()?;
            let dim_c = state_dim.first().copied().unwrap_or(1) as i32;
            let mut bcast: Vec<i32> = vec![1; x_shape.len()];
            bcast[c] = dim_c;
            ops::reshape(&clamped, &bcast)
        }
    }
}

/// Shared quant + dequant tail of `Op::FakeQuantize`. Same formula
/// regardless of which `scale_mode` produced `scale`.
fn fq_quantize_dequantize(
    x: &Array,
    scale: &Array,
    q_max: f32,
    dtype: DType,
) -> Result<Array, MlxError> {
    let scaled = ops::div(x, scale)?;
    let rounded = ops::unary(&scaled, MlxUnary::Round)?;
    let neg_qmax = Array::from_f32_slice(&[-q_max], &[1], dtype)?;
    let pos_qmax = Array::from_f32_slice(&[q_max], &[1], dtype)?;
    let clamped = ops::max(&rounded, &neg_qmax)?;
    let clamped = ops::min(&clamped, &pos_qmax)?;
    ops::mul(&clamped, scale)
}

/// `[N, C]` one-hot encoding of f32-valued integer labels.
/// `oh[n, c] = 1.0` if `labels[n] == c` else `0.0`.
fn one_hot_2d(labels: &Array, n: usize, c: usize, dtype: DType) -> Result<Array, MlxError> {
    let arange_data: Vec<f32> = (0..c).map(|i| i as f32).collect();
    let arange = Array::from_f32_slice(&arange_data, &[c], dtype)?;
    let arange_2d = ops::reshape(&arange, &[1, c as i32])?;
    let labels_2d = ops::reshape(labels, &[n as i32, 1])?;
    let mask_bool = ops::eq(&labels_2d, &arange_2d)?;
    ops::cast(&mask_bool, dtype)
}

/// Closed-form derivative of every `Activation` kind. Mirrors
/// `rlx-cpu/src/thunk.rs::activation_backward_kernel`.
fn activation_backward_compose(
    x: &Array,
    dy: &Array,
    kind: Activation,
    dtype: DType,
) -> Result<Array, MlxError> {
    use Activation::*;
    match kind {
        Relu => {
            let zero = Array::from_f32_slice(&[0.0], &[1], dtype)?;
            let mask = ops::gt(x, &zero)?;
            ops::select(&mask, dy, &zero)
        }
        Sigmoid => {
            // dy · σ(x) · (1 − σ(x))
            let s = ops::unary(x, MlxUnary::Sigmoid)?;
            let one = Array::from_f32_slice(&[1.0], &[1], dtype)?;
            let one_minus_s = ops::sub(&one, &s)?;
            let s_compl = ops::mul(&s, &one_minus_s)?;
            ops::mul(dy, &s_compl)
        }
        Tanh => {
            // dy · (1 − tanh²(x))
            let t = ops::unary(x, MlxUnary::Tanh)?;
            let t_sq = ops::mul(&t, &t)?;
            let one = Array::from_f32_slice(&[1.0], &[1], dtype)?;
            let factor = ops::sub(&one, &t_sq)?;
            ops::mul(dy, &factor)
        }
        Silu => {
            // dy · σ(x) · (1 + x · (1 − σ(x)))
            let s = ops::unary(x, MlxUnary::Sigmoid)?;
            let one = Array::from_f32_slice(&[1.0], &[1], dtype)?;
            let one_minus_s = ops::sub(&one, &s)?;
            let x_times = ops::mul(x, &one_minus_s)?;
            let inner = ops::add(&one, &x_times)?;
            let factor = ops::mul(&s, &inner)?;
            ops::mul(dy, &factor)
        }
        Gelu => {
            // dy · (½(1 + erf(x/√2)) + x · φ(x)),  φ(x) = exp(−x²/2)/√(2π)
            const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
            const INV_SQRT_2PI: f32 = 0.398_942_3;
            let inv_sqrt2 = Array::from_f32_slice(&[INV_SQRT2], &[1], dtype)?;
            let inv_sqrt_2pi = Array::from_f32_slice(&[INV_SQRT_2PI], &[1], dtype)?;
            let half = Array::from_f32_slice(&[0.5], &[1], dtype)?;
            let one = Array::from_f32_slice(&[1.0], &[1], dtype)?;
            let neg_half = Array::from_f32_slice(&[-0.5], &[1], dtype)?;

            let x_sc = ops::mul(x, &inv_sqrt2)?;
            let erf_v = ops::unary(&x_sc, MlxUnary::Erf)?;
            let phi_inner = ops::add(&one, &erf_v)?;
            let phi = ops::mul(&half, &phi_inner)?;
            let x_sq = ops::mul(x, x)?;
            let arg = ops::mul(&x_sq, &neg_half)?;
            let pdf_e = ops::unary(&arg, MlxUnary::Exp)?;
            let pdf = ops::mul(&pdf_e, &inv_sqrt_2pi)?;
            let x_pdf = ops::mul(x, &pdf)?;
            let deriv = ops::add(&phi, &x_pdf)?;
            ops::mul(dy, &deriv)
        }
        GeluApprox => {
            // y = ½ x (1 + tanh(c (x + a x³))), c = √(2/π), a = 0.044715
            // dy/dx = ½(1+t) + ½ x (1−t²) · c (1 + 3 a x²)
            const C: f32 = 0.797_884_6;
            const A: f32 = 0.044_715;
            let half = Array::from_f32_slice(&[0.5], &[1], dtype)?;
            let one = Array::from_f32_slice(&[1.0], &[1], dtype)?;
            let c_arr = Array::from_f32_slice(&[C], &[1], dtype)?;
            let a_arr = Array::from_f32_slice(&[A], &[1], dtype)?;
            let three_a = Array::from_f32_slice(&[3.0 * A], &[1], dtype)?;

            let x_sq = ops::mul(x, x)?;
            let x_cu = ops::mul(&x_sq, x)?;
            let a_x_cu = ops::mul(&a_arr, &x_cu)?;
            let inner_sum = ops::add(x, &a_x_cu)?;
            let inner = ops::mul(&c_arr, &inner_sum)?;
            let t = ops::unary(&inner, MlxUnary::Tanh)?;
            let one_plus_t = ops::add(&one, &t)?;
            let term1 = ops::mul(&half, &one_plus_t)?;
            let t_sq = ops::mul(&t, &t)?;
            let one_minus_t_sq = ops::sub(&one, &t_sq)?;
            let three_a_x_sq = ops::mul(&three_a, &x_sq)?;
            let one_plus_3ax2 = ops::add(&one, &three_a_x_sq)?;
            let dinner = ops::mul(&c_arr, &one_plus_3ax2)?;
            let half_x = ops::mul(&half, x)?;
            let part2_a = ops::mul(&half_x, &one_minus_t_sq)?;
            let term2 = ops::mul(&part2_a, &dinner)?;
            let deriv = ops::add(&term1, &term2)?;
            ops::mul(dy, &deriv)
        }
        Exp => {
            let ex = ops::unary(x, MlxUnary::Exp)?;
            ops::mul(dy, &ex)
        }
        Log => ops::div(dy, x),
        Sqrt => {
            // 0.5 · dy / √x; zero where √x ≤ 0.
            let s = ops::unary(x, MlxUnary::Sqrt)?;
            let zero = Array::from_f32_slice(&[0.0], &[1], dtype)?;
            let half = Array::from_f32_slice(&[0.5], &[1], dtype)?;
            let mask = ops::gt(&s, &zero)?;
            let half_dy = ops::mul(&half, dy)?;
            let raw = ops::div(&half_dy, &s)?;
            ops::select(&mask, &raw, &zero)
        }
        Rsqrt => {
            // −0.5 · dy / (x · √x); zero where √x ≤ 0.
            let s = ops::unary(x, MlxUnary::Sqrt)?;
            let zero = Array::from_f32_slice(&[0.0], &[1], dtype)?;
            let neg_half = Array::from_f32_slice(&[-0.5], &[1], dtype)?;
            let mask = ops::gt(&s, &zero)?;
            let denom = ops::mul(x, &s)?;
            let neg_half_dy = ops::mul(&neg_half, dy)?;
            let raw = ops::div(&neg_half_dy, &denom)?;
            ops::select(&mask, &raw, &zero)
        }
        Neg => ops::unary(dy, MlxUnary::Neg),
        Abs => {
            // sign(x) · dy. CPU reference uses 0 at x=0 (not ±0).
            let zero = Array::from_f32_slice(&[0.0], &[1], dtype)?;
            let one = Array::from_f32_slice(&[1.0], &[1], dtype)?;
            let neg_one = Array::from_f32_slice(&[-1.0], &[1], dtype)?;
            let pos = ops::gt(x, &zero)?;
            let neg = ops::lt(x, &zero)?;
            let inner = ops::select(&neg, &neg_one, &zero)?;
            let sign = ops::select(&pos, &one, &inner)?;
            ops::mul(&sign, dy)
        }
        Round => {
            // STE: pretend Round was identity (zero-grad almost everywhere
            // means the optimizer can't learn through it without this).
            dy.clone_handle()
        }
        Sin => {
            // d/dx sin(x) = cos(x) · upstream.
            let c = ops::unary(x, MlxUnary::Cos)?;
            ops::mul(&c, dy)
        }
        Cos => {
            // d/dx cos(x) = −sin(x) · upstream.
            let s = ops::unary(x, MlxUnary::Sin)?;
            let neg_s = ops::unary(&s, MlxUnary::Neg)?;
            ops::mul(&neg_s, dy)
        }
        Tan => {
            // dy · (1 + tan²(x))
            let t = ops::unary(x, MlxUnary::Tan)?;
            let t2 = ops::mul(&t, &t)?;
            let one = Array::from_f32_slice(&[1.0], &[1], dtype)?;
            let sec2 = ops::add(&one, &t2)?;
            ops::mul(dy, &sec2)
        }
        Atan => {
            // dy · (1 / (1 + x²))
            let x2 = ops::mul(x, x)?;
            let one = Array::from_f32_slice(&[1.0], &[1], dtype)?;
            let denom = ops::add(&one, &x2)?;
            ops::div(dy, &denom)
        }
    }
}
