// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower an RLX IR graph to an MPSGraph executable.
//!
//! Walks every `Node` in topo order, builds the corresponding `MpsTensor`
//! via the bridge in `mps_graph`. Returns `None` if any op isn't yet
//! supported — caller falls back to the thunk path.
//!
//! Op coverage today (matches what the `mps_graph` bridge exposes):
//!   Input | Param | Constant | MatMul | FusedMatMulBiasAct |
//!   Activation(Gelu/Silu/GeluApprox) | Binary(Add/Mul) | LayerNorm |
//!   FusedResidualLN | Reshape | Transpose (2D swap + arbitrary perm) |
//!   Cast | Gather | Narrow | Attention | Softmax | SoftmaxCrossEntropy |
//!   Reduce
//!
//! Conv2d (forward + data/weights gradients) and MaxPool2d (forward +
//! gradient) lower to native MPSGraph primitives (NCHW/OIHW), so a full CNN
//! training step — fwd, bwd, in-graph SGD — stays on one fused MPSGraph.
//!
//! Not yet supported (graph stays on thunks / hybrid):
//!   Op::Rope, Op::FusedAttentionBlock, avg-pool, most higher-order ops

use rlx_ir::op::{Activation, BinaryOp, ChainOperand, ChainStep, MaskKind, TransformStep};
use rlx_ir::{DType, Graph, Node, NodeId, Op, RegionPrologue};
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
// MPSDataTypeBFloat16 = MPSDataTypeAlternateEncodingBit(0x80000000)
// | MPSDataTypeFloatBit(0x10000000) | 16. Supported by MPSGraph on macOS 14+.
const BF16_DT: u32 = 0x90000000 | 16;
const I32_DT: u32 = 0x20000000 | 32;

fn dtype_to_mps(d: DType) -> Option<u32> {
    match d {
        DType::F32 => Some(F32_DT),
        DType::F16 => Some(F16_DT),
        DType::BF16 => Some(BF16_DT),
        DType::I32 => Some(I32_DT),
        _ => None,
    }
}

/// Mixed-precision wrapper for a 2-input compute op (`RLX_MPS_FP16`): cast both
/// inputs to fp16, run `op` in half precision (Apple GPUs have ~2× fp16
/// throughput), cast the result back to fp32. Leaves `node_to_tensor` in fp32 so
/// everything downstream — bias add, loss, the in-graph SGD update — stays
/// full-precision (compute-in-fp16, store-in-fp32; MPSGraph conv/matmul still
/// accumulate in fp32 internally). When `fp16` is false this is the identity op.
fn f16_compute2(
    mg: &MpsGraph,
    fp16: bool,
    a: &MpsTensor,
    b: &MpsTensor,
    op: impl FnOnce(&MpsTensor, &MpsTensor) -> MpsTensor,
) -> MpsTensor {
    if fp16 {
        let ah = mg.cast(a, F16_DT);
        let bh = mg.cast(b, F16_DT);
        let out = op(&ah, &bh);
        mg.cast(&out, F32_DT)
    } else {
        op(a, b)
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

/// MPSGraph rejects rank-0 tensor shapes; promote RLX scalars to `[1]`.
fn mps_shape_dims(graph: &Graph, id: NodeId) -> Option<Vec<usize>> {
    let mut dims = shape_dims(graph, id)?;
    if dims.is_empty() {
        dims.push(1);
    }
    Some(dims)
}

/// Try to lower `graph` to an MPSGraph plan. Returns `None` if any op is
/// unsupported, dynamic-shaped, or has an unknown dtype.
fn apply_region_prologue(
    mg: &MpsGraph,
    graph: &Graph,
    prologue: RegionPrologue,
    input0_id: NodeId,
    inputs_t: &mut [MpsTensor],
    out_dims: &[usize],
) -> bool {
    match prologue {
        RegionPrologue::ResizeNearest2x => {
            let in_dims = match mps_shape_dims(graph, input0_id) {
                Some(d) => d,
                None => return false,
            };
            if in_dims.len() != 4 || out_dims.len() != 4 {
                return false;
            }
            let h2 = match in_dims[2].checked_mul(2) {
                Some(h) => h,
                None => return false,
            };
            let w2 = match in_dims[3].checked_mul(2) {
                Some(w) => w,
                None => return false,
            };
            if out_dims[2] != h2 || out_dims[3] != w2 {
                return false;
            }
            inputs_t[0] = mg.resize_nearest_nchw(&inputs_t[0], h2, w2);
            true
        }
        RegionPrologue::None => true,
    }
}

fn eval_elementwise_region_chain(
    mg: &MpsGraph,
    chain: &[ChainStep],
    inputs_t: &[MpsTensor],
    trace: bool,
    node_id: NodeId,
) -> Option<MpsTensor> {
    let inputs_ref: Vec<&MpsTensor> = inputs_t.iter().collect();
    let mut steps: Vec<MpsTensor> = Vec::with_capacity(chain.len());
    let pick =
        |op: ChainOperand, inputs_t: &[&MpsTensor], steps: &[MpsTensor]| -> Option<MpsTensor> {
            match op {
                ChainOperand::Input(i) => Some(copy_tensor(inputs_t.get(i as usize)?)),
                ChainOperand::Step(i) => Some(copy_tensor(steps.get(i as usize)?)),
            }
        };
    for step in chain {
        let t = match step {
            ChainStep::Activation(act, a) => {
                let xt = pick(*a, &inputs_ref, &steps)?;
                match act {
                    Activation::Silu => mg.silu(&xt),
                    Activation::GeluApprox => mg.gelu_approx(&xt),
                    Activation::Gelu => mg.gelu(&xt),
                    Activation::Sigmoid => mg.sigmoid(&xt),
                    Activation::Tanh => mg.tanh(&xt),
                    Activation::Exp => mg.exp(&xt),
                    Activation::Log => mg.log(&xt),
                    Activation::Sqrt => mg.sqrt(&xt),
                    Activation::Rsqrt => mg.rsqrt(&xt),
                    Activation::Neg => mg.neg(&xt),
                    Activation::Abs => mg.abs(&xt),
                    Activation::Relu => mg.relu(&xt),
                    Activation::Sin => mg.sin(&xt),
                    Activation::Cos => mg.cos(&xt),
                    _ => {
                        if trace {
                            eprintln!(
                                "[mpsgraph] bail chain activation: node {} act {:?}",
                                node_id, act
                            );
                        }
                        return None;
                    }
                }
            }
            ChainStep::Binary(op, a, b) => {
                let at = pick(*a, &inputs_ref, &steps)?;
                let bt = pick(*b, &inputs_ref, &steps)?;
                match op {
                    BinaryOp::Add => mg.add(&at, &bt),
                    BinaryOp::Mul => mg.mul(&at, &bt),
                    BinaryOp::Sub => mg.sub(&at, &bt),
                    BinaryOp::Div => mg.div(&at, &bt),
                    _ => {
                        if trace {
                            eprintln!("[mpsgraph] bail chain binary: node {} op {:?}", node_id, op);
                        }
                        return None;
                    }
                }
            }
            ChainStep::Cast(dt, a) => {
                let at = pick(*a, &inputs_ref, &steps)?;
                let to = dtype_to_mps(*dt)?;
                mg.cast(&at, to)
            }
            ChainStep::Compare(cop, a, b) => {
                let at = pick(*a, &inputs_ref, &steps)?;
                let bt = pick(*b, &inputs_ref, &steps)?;
                match cop {
                    rlx_ir::op::CmpOp::Eq => mg.cmp_eq(&at, &bt),
                    rlx_ir::op::CmpOp::Ne => mg.cmp_ne(&at, &bt),
                    rlx_ir::op::CmpOp::Lt => mg.cmp_lt(&at, &bt),
                    rlx_ir::op::CmpOp::Le => mg.cmp_le(&at, &bt),
                    rlx_ir::op::CmpOp::Gt => mg.cmp_gt(&at, &bt),
                    rlx_ir::op::CmpOp::Ge => mg.cmp_ge(&at, &bt),
                }
            }
            _ => {
                if trace {
                    eprintln!(
                        "[mpsgraph] bail chain step: node {} step {:?}",
                        node_id, step
                    );
                }
                return None;
            }
        };
        steps.push(t);
    }
    steps.pop()
}

fn slice_out_dims_for_batch(_graph: &Graph, node: &Node) -> Option<Vec<usize>> {
    let rank = node.shape.rank();
    if rank == 0 {
        return None;
    }
    let mut dims: Vec<usize> = (0..rank)
        .map(|i| node.shape.dim(i).unwrap_static())
        .collect();
    if rank >= 1 {
        dims[0] = 1;
    }
    Some(dims)
}

pub fn try_lower(graph: &Graph) -> Option<MpsGraphPlan> {
    try_lower_with_constants(graph, None)
}

/// True when `graph` contains an **interior-axis reduction of a rank≥4 tensor** —
/// a `Reduce` whose reduced axis has a *non-reduced* axis after it (the reduction
/// squeezes a middle dimension while keeping a later one). MPSGraph's whole-graph
/// optimizer miscompiles this when the result reaches a graph output through
/// elementwise ops: the DeepSeek-V4 hyper-connection tail (`Σ_k comb·residual`, a
/// `[rows,hc,hc,d]` sum over axis 2) and the KV-compressor window pool read back
/// as ZEROS on `--attn-gpu` even though every input is correct (the same op
/// mid-graph, and the whole graph on the per-op thunk path, are exact). The thunk
/// path is always correct, so both the full plan ([`try_lower_with_constants`])
/// and the hybrid segmenter (`compile.rs`) refuse graphs with this pattern.
/// Trailing reductions (vision `[N,C,H,W]` over `[2,3]`) are NOT interior and stay
/// on MPSGraph. See the DSV4-Flash GA paged-decode work / `metal_hc_post_parity`.
///
/// NB an attempted `permute(axis→trailing) + trailing reduce` rewrite (2026-08-01)
/// did NOT fix it: narrowing this guard to "only non-rewritable interior reduces"
/// re-enabled the MPSGraph backbone at ~568 ms/tok (12L warm) but the output went
/// back to ALL ZEROS — the miscompile is not confined to the reduce alone. So the
/// guard stays BROAD (any interior rank≥4 reduce → thunks); the ~2.8× MPSGraph
/// "speedup" is a mirage (it's fast because it skips/miscomputes work). The correct
/// floor is the per-op thunk backbone (~764 ms/tok).
pub fn graph_has_mps_hostile_reduce(graph: &Graph) -> bool {
    graph.nodes().iter().any(|n| {
        let Op::Reduce { axes, .. } = &n.op else {
            return false;
        };
        let rank = graph.node(n.inputs[0]).shape.rank();
        if rank < 4 {
            return false;
        }
        let reduced: std::collections::HashSet<usize> = axes.iter().copied().collect();
        axes.iter()
            .any(|&a| (a + 1..rank).any(|b| !reduced.contains(&b)))
    })
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
    if graph
        .nodes()
        .iter()
        .any(|n| matches!(n.op, Op::Fft { .. } | Op::LogMel | Op::LogMelBackward))
    {
        return None;
    }
    if graph_has_mps_hostile_reduce(graph) {
        if rlx_ir::env::flag("RLX_MPSGRAPH_TRACE") {
            eprintln!(
                "[rlx-metal] mps: refusing plan (interior rank≥4 reduction — HC/KV-pool; \
                 MPSGraph miscompiles the output tail, using thunks)"
            );
        }
        return None;
    }

    let mg = MpsGraph::new();
    let mut node_to_tensor: HashMap<NodeId, MpsTensor> = HashMap::new();
    let mut inputs = Vec::new();
    let mut params = Vec::new();

    let trace = rlx_ir::env::flag("RLX_MPSGRAPH_TRACE");
    // Run conv/matmul (forward + gradients) in fp16, keeping storage/loss/SGD in
    // fp32. Apple GPUs do fp16 conv ~2× faster; MNIST tolerates the precision.
    let fp16 = rlx_ir::env::flag("RLX_MPS_FP16");
    for node in graph.nodes() {
        let dt = dtype_to_mps(node.shape.dtype())?;
        let dims = mps_shape_dims(graph, node.id)?;
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
            Op::ResizeNearest2x => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                let in_dims = mps_shape_dims(graph, node.inputs[0])?;
                if in_dims.len() != 4 {
                    return None;
                }
                let h2 = in_dims[2].checked_mul(2)?;
                let w2 = in_dims[3].checked_mul(2)?;
                if dims[2] != h2 || dims[3] != w2 {
                    return None;
                }
                mg.resize_nearest_nchw(x, h2, w2)
            }
            Op::MatMul => {
                let a = node_to_tensor.get(&node.inputs[0])?;
                let b = node_to_tensor.get(&node.inputs[1])?;
                // F16/BF16 weight × F32 activation (Zonos Metal F16; bf16-resident
                // LM head): cast B up to F32 inside MPSGraph so the arena/weight
                // buffer keeps half-sized params (bf16 = 2 bytes/elem). MPSGraph
                // reads the 16-bit weight from the buffer and fuses the cast into
                // the matmul (no persistent f32 copy of the weight). matmul itself
                // requires both operands to share a dtype, so we widen rather than
                // feed a mixed f32×bf16 pair.
                let b_dt = graph.node(node.inputs[1]).shape.dtype();
                let a_dt = graph.node(node.inputs[0]).shape.dtype();
                let b_f32;
                let b_use =
                    if matches!(b_dt, DType::F16 | DType::BF16) && matches!(a_dt, DType::F32) {
                        b_f32 = mg.cast(b, F32_DT);
                        &b_f32
                    } else {
                        b
                    };
                let a_f32;
                let a_use =
                    if matches!(a_dt, DType::F16 | DType::BF16) && matches!(b_dt, DType::F32) {
                        a_f32 = mg.cast(a, F32_DT);
                        &a_f32
                    } else {
                        a
                    };
                f16_compute2(&mg, fp16, a_use, b_use, |a, b| mg.matmul(a, b))
            }
            // ── Conv / pool (NCHW) and their gradients ──
            // MPSGraph has native primitives for all five; lowering them keeps
            // the whole CNN training step on one fused graph instead of falling
            // back to per-kernel thunks. `dims` is the node's output shape, which
            // is exactly the `outputShape` the conv-gradient ops require.
            Op::Conv {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                if stride.len() != 2 || padding.len() != 2 || dilation.len() != 2 {
                    return None;
                }
                let x = node_to_tensor.get(&node.inputs[0])?;
                let w = node_to_tensor.get(&node.inputs[1])?;
                let (s, p, d, g) = (
                    (stride[0], stride[1]),
                    (padding[0], padding[1]),
                    (dilation[0], dilation[1]),
                    *groups,
                );
                // rlx lowers ONNX 1D convs as `[N,C,1,L]` with kernel `[k,1]`, keeping
                // the length kernel/stride/pad at index 0. A literal 2D conv2d would run
                // the k-tap kernel over the singleton H axis and ignore the length. Since
                // `[N,C,1,L]` and `[N,C,L,1]` share row-major layout, relabel the length
                // onto H (reshape, no copy) so `conv2d_nchw` convolves it with the
                // index-0 params — matching rlx-cpu, the MLX 1D path, and onnxruntime.
                let in_dims = mps_shape_dims(graph, node.inputs[0])?;
                let one_d_w = in_dims.len() == 4
                    && in_dims[2] == 1
                    && in_dims[3] > 1
                    && kernel_size.len() == 2
                    && kernel_size[0] > 1
                    && kernel_size[1] == 1;
                if one_d_w {
                    let n = in_dims[0];
                    let c = in_dims[1];
                    let l = in_dims[3];
                    let x_nl1 = mg.reshape(x, &[n, c, l, 1]);
                    let y = f16_compute2(&mg, fp16, &x_nl1, w, |x, w| {
                        mg.conv2d_nchw(x, w, s, p, d, g)
                    });
                    mg.reshape(&y, &dims)
                } else {
                    f16_compute2(&mg, fp16, x, w, |x, w| mg.conv2d_nchw(x, w, s, p, d, g))
                }
            }
            Op::Pool {
                kind,
                kernel_size,
                stride,
                padding,
            } => {
                if !matches!(kind, rlx_ir::op::ReduceOp::Max) {
                    return None; // only max-pool has an MPSGraph gradient pairing here
                }
                if kernel_size.len() != 2 || stride.len() != 2 || padding.len() != 2 {
                    return None;
                }
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.maxpool2d(
                    x,
                    (kernel_size[0], kernel_size[1]),
                    (stride[0], stride[1]),
                    (padding[0], padding[1]),
                )
            }
            Op::Conv2dBackwardInput {
                kernel_size: _,
                stride,
                padding,
                dilation,
                groups,
            } => {
                if stride.len() != 2 || padding.len() != 2 || dilation.len() != 2 {
                    return None;
                }
                let dy = node_to_tensor.get(&node.inputs[0])?;
                let w = node_to_tensor.get(&node.inputs[1])?;
                let (s, p, d, g) = (
                    (stride[0], stride[1]),
                    (padding[0], padding[1]),
                    (dilation[0], dilation[1]),
                    *groups,
                );
                f16_compute2(&mg, fp16, dy, w, |dy, w| {
                    mg.conv2d_data_grad(dy, w, &dims, s, p, d, g)
                })
            }
            Op::Conv2dBackwardWeight {
                kernel_size: _,
                stride,
                padding,
                dilation,
                groups,
            } => {
                if stride.len() != 2 || padding.len() != 2 || dilation.len() != 2 {
                    return None;
                }
                // RLX inputs are [x, dy]; MPSGraph wants (grad=dy, source=x).
                let x = node_to_tensor.get(&node.inputs[0])?;
                let dy = node_to_tensor.get(&node.inputs[1])?;
                let (s, p, d, g) = (
                    (stride[0], stride[1]),
                    (padding[0], padding[1]),
                    (dilation[0], dilation[1]),
                    *groups,
                );
                f16_compute2(&mg, fp16, dy, x, |dy, x| {
                    mg.conv2d_weights_grad(dy, x, &dims, s, p, d, g)
                })
            }
            Op::MaxPool2dBackward {
                kernel_size,
                stride,
                padding,
            } => {
                if kernel_size.len() != 2 || stride.len() != 2 || padding.len() != 2 {
                    return None;
                }
                // RLX inputs are [x, dy]; MPSGraph wants (grad=dy, source=x).
                let x = node_to_tensor.get(&node.inputs[0])?;
                let dy = node_to_tensor.get(&node.inputs[1])?;
                mg.maxpool2d_grad(
                    dy,
                    x,
                    (kernel_size[0], kernel_size[1]),
                    (stride[0], stride[1]),
                    (padding[0], padding[1]),
                )
            }
            Op::FusedMatMulBiasAct { activation } => {
                let a = node_to_tensor.get(&node.inputs[0])?;
                let w = node_to_tensor.get(&node.inputs[1])?;
                let bias = node_to_tensor.get(&node.inputs[2])?;
                let mm = mg.matmul(a, w);
                let withbias = mg.add(&mm, bias);
                match activation {
                    Some(Activation::GeluApprox) => mg.gelu_approx(&withbias),
                    Some(Activation::Gelu) => mg.gelu(&withbias),
                    Some(Activation::Silu) => mg.silu(&withbias),
                    Some(Activation::Relu) | None => withbias,
                    _ => return None,
                }
            }
            Op::Activation(Activation::GeluApprox) => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.gelu_approx(x)
            }
            Op::Activation(Activation::Gelu) => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.gelu(x)
            }
            Op::Activation(Activation::Silu) => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.silu(x)
            }
            Op::Activation(Activation::Sigmoid) => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.sigmoid(x)
            }
            Op::Activation(Activation::Tanh) => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.tanh(x)
            }
            Op::Activation(Activation::Exp) => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.exp(x)
            }
            Op::Activation(Activation::Log) => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.log(x)
            }
            Op::Activation(Activation::Sqrt) => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.sqrt(x)
            }
            Op::Activation(Activation::Rsqrt) => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.rsqrt(x)
            }
            Op::Activation(Activation::Neg) => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.neg(x)
            }
            Op::Activation(Activation::Abs) => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.abs(x)
            }
            Op::Activation(Activation::Relu) => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.relu(x)
            }
            Op::Activation(Activation::Sin) => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.sin(x)
            }
            Op::Activation(Activation::Cos) => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                mg.cos(x)
            }
            Op::Binary(BinaryOp::Sub) => {
                let a = node_to_tensor.get(&node.inputs[0])?;
                let b = node_to_tensor.get(&node.inputs[1])?;
                mg.sub(a, b)
            }
            Op::Binary(BinaryOp::Div) => {
                let a = node_to_tensor.get(&node.inputs[0])?;
                let b = node_to_tensor.get(&node.inputs[1])?;
                mg.div(a, b)
            }
            Op::Reduce {
                op: rop,
                axes,
                keep_dim,
            } => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                let pos_axes: Vec<i32> = axes.iter().map(|&a| a as i32).collect();
                let reduced = match rop {
                    rlx_ir::op::ReduceOp::Sum => mg.reduce_sum(x, &pos_axes),
                    // For Mean we build via sum / N so MPSGraph's
                    // optimizer doesn't pattern-match against our
                    // reductions and substitute a fused normalize
                    // kernel that fails with "epsilon must be a scalar".
                    rlx_ir::op::ReduceOp::Mean => {
                        let sum = mg.reduce_sum(x, &pos_axes);
                        let n = {
                            let src_shape = shape_dims(graph, node.inputs[0])?;
                            let mut n: usize = 1;
                            for &ax in axes {
                                n *= src_shape[ax];
                            }
                            mg.constant_scalar(n as f32)
                        };
                        mg.div(&sum, &n)
                    }
                    rlx_ir::op::ReduceOp::Max => mg.reduce_max(x, &pos_axes),
                    rlx_ir::op::ReduceOp::Min => mg.reduce_min(x, &pos_axes),
                    rlx_ir::op::ReduceOp::Prod => mg.reduce_product(x, &pos_axes),
                };
                // MPSGraph always keeps dims after reduction. If the IR
                // op asked for them to be squeezed, reshape to the IR-
                // declared output shape (which has them dropped).
                if *keep_dim {
                    reduced
                } else {
                    mg.reshape(&reduced, &dims)
                }
            }
            Op::Softmax { axis } => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                let rank = node.shape.rank() as i32;
                let pos_axis = if *axis < 0 { rank + *axis } else { *axis };
                mg.softmax(x, pos_axis)
            }
            Op::SoftmaxCrossEntropy => {
                // Dense / soft-label cross-entropy: logits [N,C], targets
                // [N,C] → loss [N].
                //   loss = logsumexp(logits) - Σ_c targets[n,c]·logits[n,c]
                // MPSGraph keeps dims after reduction, so the intermediates
                // stay [N,1]; reshape the final [N,1] result to the IR's [N].
                // Apple's graph optimizer fuses the max/exp/sum/log chain.
                let logits = node_to_tensor.get(&node.inputs[0])?;
                let targets = node_to_tensor.get(&node.inputs[1])?;
                let m = mg.reduce_max(logits, &[1]);
                let shifted = mg.sub(logits, &m);
                let exp_d = mg.exp(&shifted);
                let sum_exp = mg.reduce_sum(&exp_d, &[1]);
                let log_sum = mg.log(&sum_exp);
                let lse = mg.add(&m, &log_sum);
                let prod = mg.mul(logits, targets);
                let dot = mg.reduce_sum(&prod, &[1]);
                let loss = mg.sub(&lse, &dot);
                mg.reshape(&loss, &dims)
            }
            Op::SoftmaxCrossEntropyWithLogits => {
                // Integer-label variant: logits [N,C], labels [N] → loss [N].
                // Build a one-hot from the labels, then reuse the dense formula.
                let logits = node_to_tensor.get(&node.inputs[0])?;
                let labels = node_to_tensor.get(&node.inputs[1])?;
                let c = mps_shape_dims(graph, node.inputs[0])?.get(1).copied()?;
                let onehot = mg.one_hot(labels, c as u64, 1);
                let m = mg.reduce_max(logits, &[1]);
                let shifted = mg.sub(logits, &m);
                let exp_d = mg.exp(&shifted);
                let sum_exp = mg.reduce_sum(&exp_d, &[1]);
                let log_sum = mg.log(&sum_exp);
                let lse = mg.add(&m, &log_sum);
                let prod = mg.mul(logits, &onehot);
                let dot = mg.reduce_sum(&prod, &[1]);
                let loss = mg.sub(&lse, &dot);
                mg.reshape(&loss, &dims)
            }
            Op::SoftmaxCrossEntropyBackward => {
                // dlogits[n,k] = (softmax(logits)[n,k] - onehot[n,k]) * d_loss[n].
                // logits/dlogits [N,C], labels [N], d_loss [N].
                let logits = node_to_tensor.get(&node.inputs[0])?;
                let labels = node_to_tensor.get(&node.inputs[1])?;
                let d_loss = node_to_tensor.get(&node.inputs[2])?;
                let n = *dims.first()?;
                let c = *dims.get(1)?;
                let sm = mg.softmax(logits, 1);
                let onehot = mg.one_hot(labels, c as u64, 1);
                let diff = mg.sub(&sm, &onehot);
                let dl = mg.reshape(d_loss, &[n, 1]); // broadcast over C
                let out = mg.mul(&diff, &dl);
                mg.reshape(&out, &dims)
            }
            Op::Transpose { perm } => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                if perm.len() == 2 {
                    mg.transpose(x, perm[0], perm[1])
                } else if perm.len() == 4
                    && perm[0] == 0
                    && perm[1] == 2
                    && perm[2] == 1
                    && perm[3] == 3
                {
                    // [B, seq, heads, dim] <-> [B, heads, seq, dim] (ViT / Brain-JEPA on Metal).
                    mg.transpose(x, 1, 2)
                } else {
                    let perm_i32: Vec<i32> = perm.iter().map(|&p| p as i32).collect();
                    mg.permute(x, &perm_i32)
                }
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
                // Resolve the reduction axis against the ACTUAL MPS rank, not
                // the IR rank: MPSGraph left-pads lower-rank RLX tensors (an
                // IR rank-2 `[seq, hidden]` hidden-state can arrive as MPS
                // `[1, seq, hidden]` after an upstream broadcast). Using the
                // IR rank made `axis=-1` resolve to MPS axis 1 (seq) instead
                // of the last (hidden) axis — RmsNorm then reduced over the
                // wrong dimension and degenerated Bonsai-27B's padded prefill
                // (cos≈-0.14 at max_seq 96; bit-exact once resolved on MPS
                // rank). Mirrors the Op::Narrow / Op::FusedSwiGLU fixes.
                let ir_rank = node.shape.rank() as i32;
                let mps_rank = x.mps_rank().map(|r| r as i32).unwrap_or(ir_rank);
                let pos_axis = if *axis < 0 {
                    mps_rank + *axis
                } else {
                    *axis + (mps_rank - ir_rank)
                };
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
                // Normalize over the MPS last axis (may be left-padded above
                // the IR rank — see the Op::RmsNorm note).
                let last = (pre.mps_rank().unwrap_or(node.shape.rank()) - 1) as i32;
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
                // Normalize over the MPS last axis (may be left-padded above
                // the IR rank — see the Op::RmsNorm note).
                let last = (pre.mps_rank().unwrap_or(node.shape.rank()) - 1) as i32;
                mg.rms_norm(&pre, gamma, beta, &[last], *eps)
            }
            Op::Reshape { .. } => {
                let x = node_to_tensor.get(&node.inputs[0])?;
                // IR Reshape sometimes embeds a broadcast (1-elem source,
                // multi-elem dst). MPSGraph's reshape is strict — same
                // element count required — so dispatch broadcast for the
                // expanding case and reshape otherwise.
                let src_n: usize = shape_dims(graph, node.inputs[0])
                    .map(|s| s.iter().product())
                    .unwrap_or(0);
                let dst_n: usize = dims.iter().product();
                if dst_n != src_n {
                    mg.broadcast_to(x, &dims)
                } else {
                    mg.reshape(x, &dims)
                }
            }
            Op::Expand { .. } => {
                // Broadcast to the IR-declared output shape.
                let x = node_to_tensor.get(&node.inputs[0])?;
                let src_n: usize = shape_dims(graph, node.inputs[0])
                    .map(|s| s.iter().product())
                    .unwrap_or(0);
                let dst_n: usize = dims.iter().product();
                if dst_n != src_n {
                    mg.broadcast_to(x, &dims)
                } else {
                    copy_tensor(x)
                }
            }
            // Identity-forward op used by the GRL (Gradient Reverse Layer)
            // in DAT training. The AD pass already handled the backward
            // semantics (zero gradient through the stop). At MPSGraph
            // lowering time we just pass the tensor through.
            Op::StopGradient => {
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
                // MPSGraph left-pads lower-rank RLX tensors with leading batch
                // dims (IR `[12,34]` → MPS `[1,12,34]`). The IR `axis` is
                // relative to the IR rank, so rebase it by the leading-dim
                // difference; a no-op when the ranks already match (e.g. 3-D
                // conv-split narrows). Fixes qwen35 GDN degeneration on Metal
                // where a 2-D channel-axis narrow landed on the seq axis.
                let mut ax = *axis as u64;
                // Use the IR rank (available even for dynamic/symbolic dims,
                // unlike `shape_dims` which returns None on any non-static dim).
                let ir_rank = graph.node(node.inputs[0]).shape.rank();
                if let Some(mr) = x.mps_rank()
                    && mr > ir_rank
                {
                    ax += (mr - ir_rank) as u64;
                }
                mg.slice(x, ax, *start as i64, *len as i64)
            }
            Op::FusedSwiGLU {
                cast_to,
                gate_first,
            } => {
                // SwiGLU = up * silu(gate). The concat layout depends on
                // `gate_first`: default (false) = [up || gate] (up @ low half,
                // gate @ high — the canonical builder order); true = [gate ||
                // up]. The earlier code hardcoded the gate-first slicing, so for
                // the canonical (up-first) layout it computed silu(up)*gate and
                // garbled every MPSGraph LM decoder (SwiGLU MLP) on Metal. Must
                // match the CPU executor (`up = in[i]; gate = in[n+i]`).
                let x = node_to_tensor.get(&node.inputs[0])?;
                let in_shape = shape_dims(graph, node.inputs[0])?;
                let rank = in_shape.len();
                let n = in_shape[rank - 1] / 2;
                // Slice the *MPS* last axis: MPSGraph left-pads lower-rank RLX
                // tensors with leading batch dims (IR `[12,64]` → MPS
                // `[1,12,64]`), so an IR-rank-based `last` would land on the
                // seq axis. Fixes qwen35 full-attn Q+gate split on Metal.
                let last = x.mps_rank().unwrap_or(rank).saturating_sub(1) as u64;
                let lo = mg.slice(x, last, 0, n as i64);
                let hi = mg.slice(x, last, n as i64, n as i64);
                let (gate, up) = if *gate_first { (lo, hi) } else { (hi, lo) };
                let g_silu = mg.silu(&gate);
                let mul = mg.mul(&up, &g_silu);
                match cast_to {
                    Some(dt) => {
                        let to = dtype_to_mps(*dt)?;
                        mg.cast(&mul, to)
                    }
                    None => mul,
                }
            }
            Op::TransformRegion { steps, .. } => {
                let mut cur_dims = mps_shape_dims(graph, node.inputs[0])?;
                let mut t = copy_tensor(node_to_tensor.get(&node.inputs[0])?);
                for step in steps {
                    match step {
                        TransformStep::ResizeNearest2x(_) => {
                            if cur_dims.len() != 4 {
                                return None;
                            }
                            let h2 = cur_dims[2].checked_mul(2)?;
                            let w2 = cur_dims[3].checked_mul(2)?;
                            t = mg.resize_nearest_nchw(&t, h2, w2);
                            cur_dims[2] = h2;
                            cur_dims[3] = w2;
                        }
                    }
                }
                if cur_dims != dims {
                    return None;
                }
                t
            }
            Op::BatchElementwiseRegion {
                chain,
                num_batch_inputs,
                prologue,
                ..
            } => {
                let n = *num_batch_inputs as usize;
                if n == 0 || node.inputs.len() != n {
                    return None;
                }
                let slice_dims = slice_out_dims_for_batch(graph, node)?;
                let mut slice_ts: Vec<MpsTensor> = Vec::with_capacity(n);
                for &in_id in &node.inputs {
                    let mut inputs_t = vec![copy_tensor(node_to_tensor.get(&in_id)?)];
                    if !apply_region_prologue(
                        &mg,
                        graph,
                        *prologue,
                        in_id,
                        &mut inputs_t,
                        &slice_dims,
                    ) {
                        return None;
                    }
                    let out = eval_elementwise_region_chain(&mg, chain, &inputs_t, trace, node.id)?;
                    slice_ts.push(out);
                }
                let refs: Vec<&MpsTensor> = slice_ts.iter().collect();
                mg.concat(&refs, 0)
            }
            Op::ElementwiseRegion {
                chain, prologue, ..
            } => {
                let mut inputs_t: Vec<MpsTensor> = Vec::with_capacity(node.inputs.len());
                for &in_id in &node.inputs {
                    inputs_t.push(copy_tensor(node_to_tensor.get(&in_id)?));
                }
                if !apply_region_prologue(
                    &mg,
                    graph,
                    *prologue,
                    node.inputs[0],
                    &mut inputs_t,
                    &dims,
                ) {
                    return None;
                }
                eval_elementwise_region_chain(&mg, chain, &inputs_t, trace, node.id)?
            }
            Op::Concat { axis } => {
                let mut refs: Vec<&MpsTensor> = Vec::with_capacity(node.inputs.len());
                for &in_id in &node.inputs {
                    refs.push(node_to_tensor.get(&in_id)?);
                }
                // MPSGraph left-pads lower-rank RLX tensors with leading batch
                // dims (IR `[R,D]` → MPS `[1,R,D]`). The IR `axis` is relative to
                // the IR rank, so rebase it by the leading-dim difference — same
                // as the `Op::Narrow` lowering. Without this a 2-D last-axis
                // concat (e.g. the dense-MLP `gate || up`) lands on the row axis,
                // giving `[1,2R,D]`; a following last-axis narrow then slices the
                // size-`D` axis at `start=D` and MPSGraph errors "start N does not
                // fit dimension size N". No-op when ranks already match.
                let mut ax = *axis as u64;
                let ir_rank = graph.node(node.inputs[0]).shape.rank();
                if let Some(mr) = refs[0].mps_rank()
                    && mr > ir_rank
                {
                    ax += (mr - ir_rank) as u64;
                }
                mg.concat(&refs, ax as i32)
            }
            Op::Attention {
                num_heads,
                head_dim,
                v_head_dim,
                mask_kind,
                score_scale,
                attn_logit_softcap: _,
            } => {
                // Asymmetric MLA (v_head_dim != head_dim): the MPSGraph SDPA
                // reshapes assume a square per-head width. Bail to the thunk
                // path, which handles v_head_dim natively.
                if v_head_dim.is_some_and(|v| v != *head_dim) {
                    if trace {
                        eprintln!(
                            "[mpsgraph] bail attention asymmetric v_head_dim: node {}",
                            node.id
                        );
                    }
                    return None;
                }
                // Apple's MPSGraph optimizer mishandles SDPA when Q/K/V are
                // slice-views of *computed* tensors (narrow of MatMul/RoPE).
                // Host-fed / leaf views are fine — bail so hybrid can run
                // attention as a thunk after materializing boundary buffers.
                if !attn_qkv_feeds_mps_safe(graph, node) {
                    if trace {
                        eprintln!(
                            "[mpsgraph] bail attention computed-slice Q/K/V: node {}",
                            node.id
                        );
                    }
                    return None;
                }
                let q = node_to_tensor.get(&node.inputs[0])?;
                let k = node_to_tensor.get(&node.inputs[1])?;
                let v = node_to_tensor.get(&node.inputs[2])?;
                let q_shape = shape_dims(graph, node.inputs[0])?;
                let k_shape = shape_dims(graph, node.inputs[1])?;
                // Direct 4D [B, H, S, D] or [B, S, H, D] → SDPA (macOS 14.4+).
                if q_shape.len() == 4 {
                    let scale = score_scale.unwrap_or(1.0 / (*head_dim as f32).sqrt());
                    lower_attention_4d(
                        &mg,
                        q,
                        k,
                        v,
                        node,
                        graph,
                        &node_to_tensor,
                        *num_heads,
                        *head_dim,
                        *mask_kind,
                        scale,
                        &q_shape,
                        &k_shape,
                        trace,
                    )?
                } else {
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
                    let kv_seq = k_shape[1];
                    let scale = score_scale.unwrap_or(1.0 / (*head_dim as f32).sqrt());
                    match mask_kind {
                        rlx_ir::op::MaskKind::None => mg.attention_unmasked(
                            q, k, v, b, s, kv_seq, *num_heads, *head_dim, scale,
                        ),
                        rlx_ir::op::MaskKind::Causal => {
                            if kv_seq == s {
                                mg.attention_causal(q, k, v, b, s, *num_heads, *head_dim, scale)
                            } else {
                                // Asymmetric causal (decode Lq=1, Lk>1) needs an
                                // absolute-position mask — use Custom / thunks.
                                if trace {
                                    eprintln!(
                                        "[mpsgraph] bail attention causal Lq!=Lk: node {} Lq={s} Lk={kv_seq}",
                                        node.id
                                    );
                                }
                                return None;
                            }
                        }
                        rlx_ir::op::MaskKind::Custom => {
                            let mask = node_to_tensor.get(&node.inputs[3])?;
                            mg.attention(q, k, v, mask, b, s, kv_seq, *num_heads, *head_dim, scale)
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
            }
            Op::Rope {
                head_dim, n_rot, ..
            } => {
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
        node_to_tensor.insert(node.id, t.with_shape(dims));
    }

    // Outputs: collect from graph.outputs.
    let mut outputs = Vec::new();
    for &out_id in &graph.outputs {
        let t = node_to_tensor.remove(&out_id)?;
        let dims = mps_shape_dims(graph, out_id)?;
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
    MpsTensor {
        obj: t.obj,
        shape: t.shape.clone(),
    }
}

/// Leaf Q/K/V (or views of leaves) are safe for whole-graph MPSGraph SDPA.
/// Slice-views of *computed* tensors trip Apple's optimizer (100% rel err).
fn attn_tensor_is_leaf_materialized(graph: &Graph, id: NodeId) -> bool {
    match &graph.node(id).op {
        Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => true,
        Op::Reshape { .. } | Op::Transpose { .. } | Op::Cast { .. } | Op::Narrow { .. } => graph
            .node(id)
            .inputs
            .first()
            .copied()
            .is_some_and(|i| attn_tensor_is_leaf_materialized(graph, i)),
        _ => false,
    }
}

fn attn_qkv_feeds_mps_safe(graph: &Graph, node: &Node) -> bool {
    node.inputs
        .iter()
        .take(3)
        .all(|&id| attn_tensor_is_leaf_materialized(graph, id))
}

fn causal_mask_bytes(seq_q: usize, seq_k: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(seq_q * seq_k * 4);
    for i in 0..seq_q {
        for j in 0..seq_k {
            let v: f32 = if j > i { -1.0e9 } else { 0.0 };
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    bytes
}

fn zero_mask_bytes(seq_q: usize, seq_k: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(seq_q * seq_k * 4);
    for _ in 0..(seq_q * seq_k) {
        bytes.extend_from_slice(&0f32.to_le_bytes());
    }
    bytes
}

/// 4-D attention: honor [`MaskKind`], support Lq≠Lk for Custom/None.
fn lower_attention_4d(
    mg: &MpsGraph,
    q: &MpsTensor,
    k: &MpsTensor,
    v: &MpsTensor,
    node: &Node,
    graph: &Graph,
    node_to_tensor: &HashMap<NodeId, MpsTensor>,
    num_heads: usize,
    head_dim: usize,
    mask_kind: MaskKind,
    scale: f32,
    q_shape: &[usize],
    k_shape: &[usize],
    trace: bool,
) -> Option<MpsTensor> {
    let _ = head_dim;
    // MPSGraph SDPA expects [B, H, S, D]. EEG-DINO / Zonos use [B, S, H, D].
    let is_bhsd = q_shape[1] == num_heads;
    let (batch, seq_q) = if is_bhsd {
        (q_shape[0], q_shape[2])
    } else {
        (q_shape[0], q_shape[1])
    };
    let seq_k = if is_bhsd { k_shape[2] } else { k_shape[1] };

    let (q4, k4, v4) = if is_bhsd {
        (copy_tensor(q), copy_tensor(k), copy_tensor(v))
    } else {
        (
            mg.transpose(q, 1, 2),
            mg.transpose(k, 1, 2),
            mg.transpose(v, 1, 2),
        )
    };

    let mask4 = match mask_kind {
        MaskKind::None => {
            let bytes = zero_mask_bytes(seq_q, seq_k);
            mg.constant_from_bytes(&bytes, &[1, 1, seq_q, seq_k], F32_DT)
        }
        MaskKind::Causal => {
            if seq_q != seq_k {
                if trace {
                    eprintln!(
                        "[mpsgraph] bail attention 4D causal Lq!=Lk: node {} Lq={seq_q} Lk={seq_k}",
                        node.id
                    );
                }
                return None;
            }
            let bytes = causal_mask_bytes(seq_q, seq_k);
            mg.constant_from_bytes(&bytes, &[1, 1, seq_q, seq_k], F32_DT)
        }
        MaskKind::Custom => {
            // Keep-mask `[B, Sk]` → additive `[B, 1, 1, Sk]` via (m-1)·1e9.
            let mask = node_to_tensor.get(&node.inputs[3])?;
            let mask_shape = shape_dims(graph, node.inputs[3])?;
            if mask_shape.len() != 2 || mask_shape[0] != batch || mask_shape[1] != seq_k {
                if trace {
                    eprintln!(
                        "[mpsgraph] bail attention 4D custom mask shape: node {} mask={:?} want [{batch}, {seq_k}]",
                        node.id, mask_shape
                    );
                }
                return None;
            }
            let mask_bc = mg.reshape(mask, &[batch, 1, 1, seq_k]);
            let neg_one = mg.constant_scalar(-1.0);
            let large = mg.constant_scalar(1.0e9);
            let mask_minus = mg.add(&mask_bc, &neg_one);
            mg.mul(&mask_minus, &large)
        }
        other => {
            if trace {
                eprintln!(
                    "[mpsgraph] bail attention 4D mask_kind: node {} kind {:?}",
                    node.id, other
                );
            }
            return None;
        }
    };

    // DEFAULT: numerically-exact hand-rolled SDPA (scores → scale → +mask →
    // softmax → @V), which matches the CPU `Op::Attention` thunk on every
    // shape. Apple's MPSGraph `scaledDotProductAttention` uses a fast-softmax
    // accumulation that DIVERGES for strided / RoPE'd / aliased q/k/v — cos
    // ~0.98 on EEG-DINO / MantisV2 / CBraMod — and it returns a (wrong) result
    // rather than an error, so the old `match … { Some => use it }` silently
    // accepted the divergence. Opt back into MPS SDPA (faster where it's known
    // safe) via `RLX_METAL_MPS_SDPA=1`.
    let use_mps = std::env::var("RLX_METAL_MPS_SDPA").as_deref() == Ok("1");
    let out4 = if use_mps {
        mg.scaled_dot_product_attention(&q4, &k4, &v4, &mask4, scale)
    } else {
        None
    }
    .unwrap_or_else(|| {
        let k4_t = mg.transpose(&k4, 2, 3);
        let scores = mg.matmul(&q4, &k4_t);
        let scale_t = mg.constant_scalar(scale);
        let scores = mg.mul(&scores, &scale_t);
        let scores = mg.add(&scores, &mask4);
        let weights = mg.softmax(&scores, 3);
        mg.matmul(&weights, &v4)
    });

    if is_bhsd {
        Some(out4)
    } else {
        Some(mg.transpose(&out4, 1, 2))
    }
}
