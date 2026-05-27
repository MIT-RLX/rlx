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

//! Lower an RLX IR graph to an MPSGraph executable.
//!
//! Walks every `Node` in topo order, builds the corresponding `MpsTensor`
//! via the bridge in `mps_graph`. Returns `None` if any op isn't yet
//! supported — caller falls back to the thunk path.
//!
//! Op coverage today (matches what the `mps_graph` bridge exposes):
//!   Input | Param | Constant | MatMul | FusedMatMulBiasAct |
//!   Activation(Gelu/Silu) | Binary(Add/Mul) | LayerNorm |
//!   FusedResidualLN | Reshape | Cast | Gather | Narrow | Conv2d
//!
//! Not yet supported (graph stays on thunks):
//!   Op::Attention, Op::Rope, Op::FusedAttentionBlock, Op::Softmax (rare)
//!
//! Once Attention is added, full BERT lowering is enabled.

use rlx_ir::op::{Activation, BinaryOp, ChainOperand, ChainStep};
use rlx_ir::{DType, Graph, NodeId, Op};
use std::collections::HashMap;

use crate::mps_graph::{MpsGraph, MpsGraphExecutable, MpsTensor, mps_graph_supported};

/// Compiled plan: an MpsGraph plus the bookkeeping to bind inputs/outputs
/// at runtime against our arena buffers. When `executable` is `Some`,
/// runs go through the precompiled binary (no per-call JIT analysis,
/// positional binding instead of dict-key lookup).
pub struct MpsGraphPlan {
    pub graph: MpsGraph,
    /// Ordered (name, MpsTensor, shape, dtype) for inputs (placeholders).
    pub inputs: Vec<(String, MpsTensor, Vec<usize>, u32)>,
    /// Ordered (name, MpsTensor, shape, dtype) for parameters (also placeholders).
    pub params: Vec<(String, MpsTensor, Vec<usize>, u32)>,
    /// Ordered (NodeId, MpsTensor, shape, dtype) for graph outputs.
    pub outputs: Vec<(NodeId, MpsTensor, Vec<usize>, u32)>,
    /// Precompiled binary — set after lowering by `try_lower`. None
    /// when running on a macOS that lacks
    /// `compileWithDevice:feeds:targetTensors:...` (very rare).
    pub executable: Option<MpsGraphExecutable>,
}

impl MpsGraphPlan {
    pub fn output_node_ids(&self) -> Vec<NodeId> {
        self.outputs.iter().map(|(id, _, _, _)| *id).collect()
    }
}

const F32_DT: u32 = 0x10000000 | 32;
const F16_DT: u32 = 0x10000000 | 16;
const I32_DT: u32 = 0x20000000 | 32;

fn dtype_to_mps(d: DType) -> Option<u32> {
    match d {
        DType::F32 => Some(F32_DT),
        DType::F16 => Some(F16_DT),
        DType::I32 => Some(I32_DT),
        _ => None,
    }
}

fn shape_dims(graph: &Graph, id: NodeId) -> Option<Vec<usize>> {
    let nd = graph.node(id);
    let mut out = Vec::with_capacity(nd.shape.rank());
    for i in 0..nd.shape.rank() {
        let d = nd.shape.dim(i);
        if !d.is_static() {
            return None;
        }
        out.push(d.unwrap_static());
    }
    Some(out)
}

/// Try to lower `graph` to an MPSGraph plan. Returns `None` if any op is
/// unsupported, dynamic-shaped, or has an unknown dtype.
pub fn try_lower(graph: &Graph) -> Option<MpsGraphPlan> {
    try_lower_with_constants(graph, None)
}

/// Same as [`try_lower`] but with an optional `params_as_constants`
/// map. When provided, every `Op::Param { name }` whose name appears
/// in the map is lowered as `constantWithData:shape:dataType:` —
/// baked into the compiled executable — instead of as a per-call
/// placeholder. Params not in the map keep the placeholder + feed
/// path. The MPSGraph optimizer can then specialize matmul kernels,
/// fold reshapes through constants, and skip the per-call NSArray
/// entry for those tensors entirely.
///
/// Used by `MetalExecutable::freeze_params_to_mps_constants` after
/// `set_param` has populated the arena, to re-lower the graph with
/// the now-available bytes turned into IR constants.
pub fn try_lower_with_constants(
    graph: &Graph,
    params_as_constants: Option<&HashMap<String, Vec<u8>>>,
) -> Option<MpsGraphPlan> {
    if !mps_graph_supported() {
        return None;
    }

    let mg = MpsGraph::new();
    let mut node_to_tensor: HashMap<NodeId, MpsTensor> = HashMap::new();
    let mut inputs = Vec::new();
    let mut params = Vec::new();

    let trace = rlx_ir::env::flag("RLX_MPSGRAPH_TRACE");
    for node in graph.nodes() {
        let dt = dtype_to_mps(node.shape.dtype())?;
        let dims = shape_dims(graph, node.id)?;
        let t = match &node.op {
            Op::Input { name } => {
                let t = mg.placeholder(&dims, dt, name);
                inputs.push((name.clone(), copy_tensor(&t), dims.clone(), dt));
                t
            }
            Op::Param { name } => {
                // If the caller provided bytes for this param, bake
                // them in as a graph constant instead of a placeholder.
                // The tensor is then *not* added to `params` (so it
                // won't appear in the executable's feed list).
                if let Some(bytes) = params_as_constants.and_then(|m| m.get(name)) {
                    mg.constant_from_bytes(bytes, &dims, dt)
                } else {
                    let t = mg.placeholder(&dims, dt, name);
                    params.push((name.clone(), copy_tensor(&t), dims.clone(), dt));
                    t
                }
            }
            Op::Constant { data } => {
                // Bake constant bytes into the graph at compile time.
                mg.constant_from_bytes(data, &dims, dt)
            }
            Op::MatMul => {
                let a = node_to_tensor.get(&node.inputs[0])?;
                let b = node_to_tensor.get(&node.inputs[1])?;
                mg.matmul(a, b)
            }
            Op::FusedMatMulBiasAct { activation } => {
                let a = node_to_tensor.get(&node.inputs[0])?;
                let w = node_to_tensor.get(&node.inputs[1])?;
                let bias = node_to_tensor.get(&node.inputs[2])?;
                let mm = mg.matmul(a, w);
                let withbias = mg.add(&mm, bias);
                match activation {
                    Some(Activation::Gelu) => mg.gelu(&withbias),
                    Some(Activation::Silu) => mg.silu(&withbias),
                    Some(Activation::Relu) | None => withbias,
                    _ => return None,
                }
            }
            Op::Activation(Activation::Gelu) => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.gelu(x)
            }
            Op::Activation(Activation::Silu) => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.silu(x)
            }
            Op::Binary(BinaryOp::Add) => {
                let a = node_to_tensor.get(&node.inputs[0])?;
                let b = node_to_tensor.get(&node.inputs[1])?;
                mg.add(a, b)
            }
            Op::Binary(BinaryOp::Mul) => {
                let a = node_to_tensor.get(&node.inputs[0])?;
                let b = node_to_tensor.get(&node.inputs[1])?;
                mg.mul(a, b)
            }
            Op::LayerNorm { axis, eps } => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                let g = node_to_tensor.get(&node.inputs[1])?;
                let b = node_to_tensor.get(&node.inputs[2])?;
                // CPU thunk treats axis as "last dim" regardless; MPSGraph
                // requires concrete positive indices. Normalize negative
                // axes (e.g. -1) to rank-relative positive form so the
                // mean/variance reductions hit the right dimension.
                let rank = node.shape.rank() as i32;
                let pos_axis = if *axis < 0 { rank + *axis } else { *axis };
                mg.layer_norm(x, g, b, &[pos_axis], *eps)
            }
            Op::RmsNorm { axis, eps } => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                let g = node_to_tensor.get(&node.inputs[1])?;
                let b = node_to_tensor.get(&node.inputs[2])?;
                let rank = node.shape.rank() as i32;
                let pos_axis = if *axis < 0 { rank + *axis } else { *axis };
                mg.rms_norm(x, g, b, &[pos_axis], *eps)
            }
            Op::FusedResidualLN { has_bias, eps } => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                let res = node_to_tensor.get(&node.inputs[1])?;
                // Layout depends on has_bias:
                //   has_bias=false → inputs = [x, res, gamma, beta]
                //   has_bias=true  → inputs = [x, res, bias, gamma, beta]
                let (bias_t, gamma, beta) = if *has_bias {
                    let bias = node_to_tensor.get(&node.inputs[2])?;
                    let gamma = node_to_tensor.get(&node.inputs[3])?;
                    let beta = node_to_tensor.get(&node.inputs[4])?;
                    (Some(bias), gamma, beta)
                } else {
                    let gamma = node_to_tensor.get(&node.inputs[2])?;
                    let beta = node_to_tensor.get(&node.inputs[3])?;
                    (None, gamma, beta)
                };
                // pre = x + res [+ bias]
                let pre = mg.add(x, res);
                let pre = match bias_t {
                    Some(b) => mg.add(&pre, b),
                    None => pre,
                };
                let last = (node.shape.rank() - 1) as i32;
                mg.layer_norm(&pre, gamma, beta, &[last], *eps)
            }
            Op::FusedResidualRmsNorm { has_bias, eps } => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                let res = node_to_tensor.get(&node.inputs[1])?;
                let (bias_t, gamma, beta) = if *has_bias {
                    let bias = node_to_tensor.get(&node.inputs[2])?;
                    let gamma = node_to_tensor.get(&node.inputs[3])?;
                    let beta = node_to_tensor.get(&node.inputs[4])?;
                    (Some(bias), gamma, beta)
                } else {
                    let gamma = node_to_tensor.get(&node.inputs[2])?;
                    let beta = node_to_tensor.get(&node.inputs[3])?;
                    (None, gamma, beta)
                };
                let pre = mg.add(x, res);
                let pre = match bias_t {
                    Some(b) => mg.add(&pre, b),
                    None => pre,
                };
                let last = (node.shape.rank() - 1) as i32;
                mg.rms_norm(&pre, gamma, beta, &[last], *eps)
            }
            Op::Reshape { .. } => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.reshape(x, &dims)
            }
            Op::Expand { .. } => {
                // For now treat Expand as a no-op pass-through (broadcast
                // semantics handled by MPSGraph's add/mul broadcasting).
                let x = node_to_tensor.get(&node.inputs[0])?;
                copy_tensor(x)
            }
            Op::Cast { to } => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                let to_dt = dtype_to_mps(*to)?;
                mg.cast(x, to_dt)
            }
            Op::Gather { axis } => {
                let table = node_to_tensor.get(&node.inputs[0])?;
                let idx = node_to_tensor.get(&node.inputs[1])?;
                // MPSGraph's gather requires int indices. RLX uses f32
                // for indices in many graphs (input_ids passed as f32);
                // cast to i32 here.
                let idx_dt = graph.node(node.inputs[1]).shape.dtype();
                let idx_i = if matches!(idx_dt, DType::I32 | DType::I64) {
                    copy_tensor(idx)
                } else {
                    mg.cast(idx, I32_DT)
                };
                mg.gather(table, &idx_i, *axis as u64)
            }
            Op::Narrow { axis, start, len } => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.slice(x, *axis as u64, *start as i64, *len as i64)
            }
            Op::FusedSwiGLU { cast_to, .. } => {
                // Input layout: last axis holds [gate || up] of width 2n.
                // SwiGLU = silu(gate) * up, optionally cast.
                let x = node_to_tensor.get(&node.inputs[0])?;
                let in_shape = shape_dims(graph, node.inputs[0])?;
                let rank = in_shape.len();
                let n = in_shape[rank - 1] / 2;
                let last = (rank - 1) as u64;
                let gate = mg.slice(x, last, 0, n as i64);
                let up = mg.slice(x, last, n as i64, n as i64);
                let g_silu = mg.silu(&gate);
                let mul = mg.mul(&g_silu, &up);
                match cast_to {
                    Some(dt) => {
                        let to = dtype_to_mps(*dt)?;
                        mg.cast(&mul, to)
                    }
                    None => mul,
                }
            }
            Op::ElementwiseRegion { chain, .. } => {
                // Replay the fused chain via the bridge primitives.
                // Limited to the kernels MPSGraph already exposes —
                // (silu, gelu, sigmoid, tanh, neg, abs) × (add, sub,
                // mul, div). Anything else bails so the graph keeps
                // 1:1 numerics with the thunk path.
                let inputs_t: Vec<&MpsTensor> = node
                    .inputs
                    .iter()
                    .filter_map(|id| node_to_tensor.get(id))
                    .collect();
                if inputs_t.len() != node.inputs.len() {
                    if trace {
                        eprintln!(
                            "[mpsgraph] bail elementwise_region inputs: node {}",
                            node.id
                        );
                    }
                    return None;
                }
                let mut steps: Vec<MpsTensor> = Vec::with_capacity(chain.len());
                let pick = |op: ChainOperand,
                            inputs_t: &[&MpsTensor],
                            steps: &[MpsTensor]|
                 -> Option<MpsTensor> {
                    match op {
                        ChainOperand::Input(i) => Some(copy_tensor(inputs_t.get(i as usize)?)),
                        ChainOperand::Step(i) => Some(copy_tensor(steps.get(i as usize)?)),
                    }
                };
                for step in chain {
                    let t = match step {
                        ChainStep::Activation(act, a) => {
                            let xt = pick(*a, &inputs_t, &steps)?;
                            match act {
                                Activation::Silu => mg.silu(&xt),
                                Activation::Gelu | Activation::GeluApprox => mg.gelu(&xt),
                                Activation::Sigmoid => mg.sigmoid(&xt),
                                Activation::Tanh => mg.tanh(&xt),
                                _ => {
                                    if trace {
                                        eprintln!(
                                            "[mpsgraph] bail chain activation: node {} act {:?}",
                                            node.id, act
                                        );
                                    }
                                    return None;
                                }
                            }
                        }
                        ChainStep::Binary(op, a, b) => {
                            let at = pick(*a, &inputs_t, &steps)?;
                            let bt = pick(*b, &inputs_t, &steps)?;
                            match op {
                                BinaryOp::Add => mg.add(&at, &bt),
                                BinaryOp::Mul => mg.mul(&at, &bt),
                                _ => {
                                    if trace {
                                        eprintln!(
                                            "[mpsgraph] bail chain binary: node {} op {:?}",
                                            node.id, op
                                        );
                                    }
                                    return None;
                                }
                            }
                        }
                        _ => {
                            if trace {
                                eprintln!(
                                    "[mpsgraph] bail chain step: node {} step {:?}",
                                    node.id, step
                                );
                            }
                            return None;
                        }
                    };
                    steps.push(t);
                }
                steps.pop()?
            }
            Op::Concat { axis } => {
                let mut refs: Vec<&MpsTensor> = Vec::with_capacity(node.inputs.len());
                for &in_id in &node.inputs {
                    refs.push(node_to_tensor.get(&in_id)?);
                }
                mg.concat(&refs, *axis as i32)
            }
            Op::Attention {
                num_heads,
                head_dim,
                mask_kind,
            } => {
                let q = node_to_tensor.get(&node.inputs[0])?;
                let k = node_to_tensor.get(&node.inputs[1])?;
                let v = node_to_tensor.get(&node.inputs[2])?;
                let q_shape = shape_dims(graph, node.inputs[0])?;
                if q_shape.len() != 3 {
                    if trace {
                        eprintln!(
                            "[mpsgraph] bail attention rank: node {} q_shape={:?}",
                            node.id, q_shape
                        );
                    }
                    return None;
                }
                let (b, s) = (q_shape[0], q_shape[1]);
                let k_shape = shape_dims(graph, node.inputs[1])?;
                let kv_seq = k_shape[1];
                match mask_kind {
                    rlx_ir::op::MaskKind::None => {
                        mg.attention_unmasked(q, k, v, b, s, kv_seq, *num_heads, *head_dim)
                    }
                    rlx_ir::op::MaskKind::Causal => {
                        mg.attention_causal(q, k, v, b, s, *num_heads, *head_dim)
                    }
                    rlx_ir::op::MaskKind::Custom => {
                        let mask = node_to_tensor.get(&node.inputs[3])?;
                        mg.attention(q, k, v, mask, b, s, *num_heads, *head_dim)
                    }
                    _ => {
                        if trace {
                            eprintln!(
                                "[mpsgraph] bail attention mask_kind: node {} kind {:?}",
                                node.id, mask_kind
                            );
                        }
                        return None;
                    }
                }
            }
            Op::Rope { head_dim, n_rot } => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                let cos_t = node_to_tensor.get(&node.inputs[1])?;
                let sin_t = node_to_tensor.get(&node.inputs[2])?;
                let x_shape = shape_dims(graph, node.inputs[0])?;
                if x_shape.len() != 3 {
                    return None;
                }
                let (b, s) = (x_shape[0], x_shape[1]);
                let nh = x_shape[2] / *head_dim;
                mg.rope(x, cos_t, sin_t, b, s, nh, *head_dim, *n_rot)
            }
            Op::DequantMatMul { scheme } => {
                if !scheme.is_gguf() {
                    return None;
                }
                let w_id = node.inputs[1];
                let Op::Param { name } = &graph.node(w_id).op else {
                    return None;
                };
                let w_bytes = params_as_constants.and_then(|m| m.get(name))?;
                let x_shape = shape_dims(graph, node.inputs[0])?;
                let out_shape = shape_dims(graph, node.id)?;
                let k = *x_shape.last()?;
                let n = *out_shape.last()?;
                if w_bytes.len() != k * n * 4 {
                    if trace {
                        eprintln!(
                            "[mpsgraph] bail dequant_matmul bytes: node {} len={} want {}",
                            node.id,
                            w_bytes.len(),
                            k * n * 4
                        );
                    }
                    return None;
                }
                let w = mg.constant_from_bytes(w_bytes, &[k, n], F32_DT);
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.matmul(x, &w)
            }
            // Unsupported ops — bail out so caller falls back to thunks.
            _ => {
                if rlx_ir::env::flag("RLX_MPSGRAPH_TRACE") {
                    eprintln!("[mpsgraph] unsupported: node {} op {:?}", node.id, node.op);
                }
                return None;
            }
        };
        node_to_tensor.insert(node.id, t);
    }

    // Outputs: collect from graph.outputs.
    let mut outputs = Vec::new();
    for &out_id in &graph.outputs {
        let t = node_to_tensor.remove(&out_id)?;
        let dims = shape_dims(graph, out_id)?;
        let dt = dtype_to_mps(graph.node(out_id).shape.dtype())?;
        outputs.push((out_id, t, dims, dt));
    }

    // Precompile the executable: per-call dispatch drops from "JIT
    // analyze + build feeds dict + lookup-by-NSObject" to "build
    // inputs/results NSArrays + run binary". Big win on small graphs
    // (B≤2, L≤8) where the JIT analyze is the floor.
    let feed_tensors_ordered: Vec<&MpsTensor> = inputs
        .iter()
        .map(|(_, t, _, _)| t)
        .chain(params.iter().map(|(_, t, _, _)| t))
        .collect();
    let feed_shapes_ordered: Vec<Vec<usize>> = inputs
        .iter()
        .map(|(_, _, s, _)| s.clone())
        .chain(params.iter().map(|(_, _, s, _)| s.clone()))
        .collect();
    let feed_dtypes_ordered: Vec<u32> = inputs
        .iter()
        .map(|(_, _, _, d)| *d)
        .chain(params.iter().map(|(_, _, _, d)| *d))
        .collect();
    let target_tensors_ordered: Vec<&MpsTensor> = outputs.iter().map(|(_, t, _, _)| t).collect();
    // Precompiled executable: per-call dispatch drops to a binary
    // ObjC call instead of JIT analysis + dict-key lookup. ~2× win on
    // small graphs (B=1, L=8 prefill). Opt out with
    // RLX_DISABLE_MPSGRAPH_EXECUTABLE=1.
    let executable = if rlx_ir::env::flag("RLX_DISABLE_MPSGRAPH_EXECUTABLE") {
        None
    } else {
        mg.compile_executable(
            &feed_tensors_ordered,
            &feed_shapes_ordered,
            &feed_dtypes_ordered,
            &target_tensors_ordered,
        )
    };

    Some(MpsGraphPlan {
        graph: mg,
        inputs,
        params,
        outputs,
        executable,
    })
}

/// MpsTensor is just an objc pointer; copy is safe (the graph owns the
/// real lifetime). The bridge defines MpsTensor as `pub` but no Copy
/// derive — we replicate a shallow copy here.
fn copy_tensor(t: &MpsTensor) -> MpsTensor {
    // SAFETY: MpsTensor wraps an objc pointer owned by the MPSGraph;
    // duplicating the pointer is fine as long as the graph outlives all
    // copies, which is true for our use (the plan owns the graph).
    unsafe { std::ptr::read(t) }
}
