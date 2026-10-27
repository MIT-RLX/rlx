// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! IR-level "unfusion" pass for the wgpu backend.
//!
//! The shared decompose driver lives in `rlx-unfuse`; this module only
//! supplies wgpu's [`DecomposePolicy`] and re-exports `collapse_reshapes`.
//!
//! The wgpu backend lowers `FusedMatMulBiasAct` (matmul folds bias +
//! activation into its WGSL epilogue) and `FusedResidualLN`
//! (`fused_residual_ln.wgsl` does Add[+bias] + LayerNorm in one pass)
//! natively — so the decompose pass folds biased projections into
//! FusedMatMulBiasAct and residual+norm pairs into FusedResidualLN. Its
//! Attention kernel reads Q/K/V (and the mask) via per-axis strides, so it
//! accepts rank-3 `[B, S, H·D]` inputs directly and skips the
//! reshape/transpose to `[B, H, S, D]`.

use std::collections::HashMap;

use rlx_ir::{Graph, NodeId, Op, Shape};
use rlx_unfuse::DecomposePolicy;

/// wgpu's decompose policy: fold biased matmuls into `FusedMatMulBiasAct`,
/// fold residual+norm pairs into `FusedResidualLN`, and pass rank-3 Q/K/V/
/// mask straight to the stride-driven Attention kernel.
pub(crate) struct WgpuPolicy;

impl DecomposePolicy for WgpuPolicy {
    /// `Op::While` runs as a host loop (`Step::HostOp`).
    fn while_native(&self) -> bool {
        true
    }

    fn fold_matmul_bias_act(&self) -> bool {
        !rlx_ir::env::flag("RLX_WGPU_NO_FOLD_MMBA")
    }

    fn fold_residual_ln(&self) -> bool {
        !rlx_ir::env::flag("RLX_WGPU_NO_FOLD_RESLN")
    }

    fn attention_accepts_rank3(&self) -> bool {
        true
    }

    /// `binary_main.wgsl` reads each operand through `(i / rep) % len`, so a
    /// scalar or per-channel operand does not need materialising. Without this
    /// every broadcast became a full-size `Expand`: on a 192^3 SynthStrip, 52
    /// of them, 5.57 GiB of a 7.05 GiB arena, which pushed the arena past the
    /// 4 GiB binding limit into striping — and striped arenas are refused.
    fn binary_broadcast_native(&self) -> bool {
        true
    }

    // Keep `Op::FusedSwiGLU` fused — wgpu has a native kernel
    // (`fused_swiglu.wgsl`) instead of the Narrow+Silu+Mul decompose.
    fn swiglu_native(&self) -> bool {
        true
    }
}

pub fn unfuse(graph: Graph) -> Graph {
    let g = fold_conv_bias_into_conv3d(expand_unsupported_fusions(graph));
    rlx_unfuse::unfuse(g, &WgpuPolicy)
}

/// Rewrite a 3-D [`Op::FusedConvBiasAct`] into an [`Op::Conv3d`] that keeps the
/// bias as a third input.
///
/// The conv3d shaders add the bias in their store, so this costs nothing at
/// run time and saves two full-size tensors per convolution: the `Expand` of
/// the bias to the output shape, and the `Add` that consumed it. On SynthStrip
/// that is 27 of each — most of the reason a 192^3 activation arena did not
/// fit under the 4 GiB binding limit.
///
/// The extra input is invisible to everything except this backend's lowering:
/// the memory planner reads it as one more liveness edge, which is exactly
/// what it is.
/// Rewrite a rank-5 `FusedConvBiasAct` (no activation, no residual) into a
/// **3-input** `Op::Conv3d` so `conv3d.wgsl` can add the bias in its store.
///
/// This is a **backend-private** representation. `Op::Conv3d` is
/// `Arity::Exact(2)` in the IR contract ("bias via Add"), so the graph this
/// produces would not pass `rlx_ir::verify` — which is fine, because it is
/// created after verification and never leaves this crate.
///
/// The arity is deliberately *not* widened to `Range { min: 2, max: 3 }` in
/// `Op::arity`: CPU's `compile_conv3d` does not read a third operand, so a
/// bias-carrying `Conv3d` reaching any backend other than this one would have
/// its bias silently dropped. Keeping the global contract at 2 means only the
/// backend that implements the fused form can construct it.
fn fold_conv_bias_into_conv3d(g: Graph) -> Graph {
    let is_target = |n: &rlx_ir::Node| {
        matches!(
            &n.op,
            Op::FusedConvBiasAct {
                activation: None,
                has_residual: false,
                ..
            }
        ) && n.shape.rank() == 5
            && n.inputs.len() == 3
    };
    if !g.nodes().iter().any(is_target) {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        let ins: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let op = if is_target(node) {
            let Op::FusedConvBiasAct {
                stride,
                padding,
                dilation,
                groups,
                ..
            } = &node.op
            else {
                unreachable!()
            };
            let a3 = |v: &Vec<usize>| [v[0], v[1], v[2]];
            Op::Conv3d {
                stride: a3(stride),
                padding: a3(padding),
                dilation: a3(dilation),
                groups: *groups,
            }
        } else {
            node.op.clone()
        };
        let nid = out.add_node(op, ins, node.shape.clone());
        id_map.insert(node.id, nid);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

/// Expand any [`Op::FusedConvBiasAct`] that is not the 2-D case wgpu lowers.
///
/// `SUPPORTED_OPS` lists the op unconditionally and `compile::lower` asserts
/// `only 2D NCHW convs are fused`, so a 3-D graph aborts at compile time
/// rather than running. (CUDA and ROCm share the claim without the assert and
/// silently compute a 2-D convolution instead.) Sending other ranks back to
/// primitives lets them run on `conv3d.wgsl`.
fn expand_unsupported_fusions(g: Graph) -> Graph {
    // Rank-5 with no activation and no residual is lowered natively — the
    // conv3d shaders apply the bias in their store. Expanding it instead costs
    // a full-size `Expand` of the bias and a full-size `Add` per convolution:
    // 27 of each on SynthStrip, which is most of why a 192^3 arena did not fit.
    let unsupported = |n: &rlx_ir::Node| -> bool {
        match &n.op {
            Op::FusedConvBiasAct {
                activation: None,
                has_residual: false,
                ..
            } => n.shape.rank() != 5 || n.inputs.len() != 3,
            Op::FusedConvBiasAct { .. } => n.shape.rank() != 4,
            _ => false,
        }
    };
    if !g.nodes().iter().any(unsupported) {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = if unsupported(node) {
            inline_unfused(&mut out, &node.op, &new_inputs, &node.shape)
        } else {
            out.add_node(node.op.clone(), new_inputs, node.shape.clone())
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

/// Run one node through the shared decomposer and splice the result back in.
fn inline_unfused(out: &mut Graph, op: &Op, inputs: &[NodeId], shape: &Shape) -> NodeId {
    let mut mini = Graph::new("wgpu_unfuse_fcba");
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
    let mut map: HashMap<NodeId, NodeId> = HashMap::new();
    for n in expanded.nodes() {
        if let Op::Input { name } = &n.op
            && let Some(rest) = name.strip_prefix("in")
            && let Ok(i) = rest.parse::<usize>()
        {
            map.insert(n.id, inputs[i]);
            continue;
        }
        let mapped: Vec<NodeId> = n.inputs.iter().map(|id| map[id]).collect();
        let nid = out.add_node(n.op.clone(), mapped, n.shape.clone());
        map.insert(n.id, nid);
    }
    map[&expanded.outputs[0]]
}

pub fn collapse_reshapes(graph: Graph) -> Graph {
    rlx_unfuse::collapse_reshapes(graph)
}

/// Read a broadcast `Binary` operand from the tensor an `Expand` was fed,
/// instead of from the full-size copy the `Expand` materialises.
///
/// `binary_main.wgsl` already indexes each operand through `(i / rep) % len`
/// (see [`WgpuPolicy::binary_broadcast_native`]), but graphs arrive with the
/// broadcasts spelled out: a batch-norm writes `expand(gamma)` before its
/// `Mul`, and autodiff emits `expand(expand(g))` for every reduced gradient.
/// On EEGNet's training step that was 31 of 216 dispatches, each writing a
/// full activation-sized tensor that one `Binary` then read once.
///
/// An operand is rewired through a chain of `Expand`s only while the source
/// stays a [`rlx_unfuse::simple_broadcast`] of the `Binary`'s output — the
/// exact test the lowering uses to pick `(rep, len)` — so every rewired read
/// is the element the `Expand` would have stored. `Expand`s left without a
/// consumer are dropped; inputs and every other node are kept as they are.
pub fn fold_expand_into_binary(graph: Graph) -> Graph {
    let source = |node: &rlx_ir::Node, mut id: NodeId| -> NodeId {
        while let Op::Expand { .. } = graph.node(id).op {
            let src = graph.node(id).inputs[0];
            let s = &graph.node(src).shape;
            if s.dtype() != node.shape.dtype()
                || s.dims()
                    .iter()
                    .chain(node.shape.dims())
                    .any(|d| !d.is_static())
                || rlx_unfuse::simple_broadcast(&node.shape, s).is_none()
            {
                break;
            }
            id = src;
        }
        id
    };
    let mut rewired: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for node in graph.nodes() {
        if !matches!(node.op, Op::Binary(_)) || node.inputs.len() != 2 {
            continue;
        }
        let ins: Vec<NodeId> = node.inputs.iter().map(|&i| source(node, i)).collect();
        if ins != node.inputs {
            rewired.insert(node.id, ins);
        }
    }
    if rewired.is_empty() {
        return graph;
    }
    // Consumers after rewiring; an Expand nobody reads (and that is not an
    // output) is dead, which can in turn kill the Expand that fed it.
    let inputs_of = |n: &rlx_ir::Node| {
        rewired
            .get(&n.id)
            .cloned()
            .unwrap_or_else(|| n.inputs.clone())
    };
    let mut uses: HashMap<NodeId, usize> = HashMap::new();
    for n in graph.nodes() {
        for i in inputs_of(n) {
            *uses.entry(i).or_default() += 1;
        }
    }
    for &o in &graph.outputs {
        *uses.entry(o).or_default() += 1;
    }
    let mut dead: std::collections::HashSet<NodeId> = Default::default();
    for n in graph.nodes().iter().rev() {
        if matches!(n.op, Op::Expand { .. }) && uses.get(&n.id).copied().unwrap_or(0) == 0 {
            dead.insert(n.id);
            for i in &n.inputs {
                if let Some(u) = uses.get_mut(i) {
                    *u -= 1;
                }
            }
        }
    }
    let mut out = Graph::new(graph.name.clone());
    let mut map: HashMap<NodeId, NodeId> = HashMap::new();
    for n in graph.nodes() {
        if dead.contains(&n.id) {
            continue;
        }
        let ins: Vec<NodeId> = inputs_of(n).iter().map(|i| map[i]).collect();
        map.insert(n.id, out.add_node(n.op.clone(), ins, n.shape.clone()));
    }
    out.set_outputs(graph.outputs.iter().map(|o| map[o]).collect());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::BinaryOp;
    use rlx_ir::{DType, Shape};

    fn expand(g: &mut Graph, x: NodeId, dims: &[usize]) -> NodeId {
        g.add_node(
            Op::Expand {
                target_shape: dims.iter().map(|&d| d as i64).collect(),
            },
            vec![x],
            Shape::new(dims, DType::F32),
        )
    }

    fn expands(g: &Graph) -> usize {
        g.nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Expand { .. }))
            .count()
    }

    /// The batch-norm affine and autodiff's `expand(expand(g))` lose every
    /// `Expand`; a broadcast no `(rep, len)` describes keeps its own.
    #[test]
    fn simple_broadcast_expands_are_folded_and_dropped() {
        let f = DType::F32;
        let full = [4, 8, 3, 5];
        let mut g = Graph::new("fold");
        let x = g.input("x", Shape::new(&full, f));
        let gamma = g.input("gamma", Shape::new(&[1, 8, 1, 1], f));
        let r = g.input("r", Shape::new(&[4, 8, 1, 1], f));
        let m = g.input("m", Shape::new(&[4, 1, 3, 5], f));
        let ge = expand(&mut g, gamma, &full);
        let y = g.binary(BinaryOp::Mul, x, ge, Shape::new(&full, f));
        let r1 = expand(&mut g, r, &[4, 8, 1, 1]);
        let r2 = expand(&mut g, r1, &full);
        let y = g.binary(BinaryOp::Add, r2, y, Shape::new(&full, f));
        let me = expand(&mut g, m, &full);
        let y = g.binary(BinaryOp::Sub, y, me, Shape::new(&full, f));
        g.set_outputs(vec![y]);
        assert_eq!(expands(&g), 4);
        let folded = fold_expand_into_binary(g);
        assert_eq!(
            expands(&folded),
            1,
            "only the [4,1,3,5] broadcast stays expanded:\n{folded}"
        );
    }
}

/// `Op::Custom` name of a batched matmul whose operands are read through
/// strided views of their base tensors (see [`fold_strided_bmm`]).
pub const STRIDED_BMM: &str = "rlx_wgpu.strided_bmm";

/// Attributes of a [`STRIDED_BMM`] node: `C[z] = A[z]·B[z]` over
/// `batch × m × k × n`. Element (z, r, c) of each tensor lives at
/// `base + off + (z / b2)·s0 + (z % b2)·s1 + r·sr + c·sc` (f32 elements);
/// `a` / `b` / `c` hold `[off, s0, s1, sr, sc]`, C relative to the node's own
/// slot (whose shape may be a folded output layout, not `[batch, m, n]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StridedBmm {
    pub batch: u32,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub b2: u32,
    pub a: [u32; 5],
    pub b: [u32; 5],
    pub c: [u32; 5],
}

impl StridedBmm {
    fn encode(&self) -> Vec<u8> {
        [self.batch, self.m, self.k, self.n, self.b2]
            .iter()
            .chain(&self.a)
            .chain(&self.b)
            .chain(&self.c)
            .flat_map(|v| v.to_le_bytes())
            .collect()
    }

    pub fn decode(attrs: &[u8]) -> Self {
        let (words, rest) = attrs.as_chunks::<4>();
        assert!(rest.is_empty() && words.len() == 20, "strided_bmm attrs");
        let w: Vec<u32> = words.iter().map(|c| u32::from_le_bytes(*c)).collect();
        let five = |i: usize| [w[i], w[i + 1], w[i + 2], w[i + 3], w[i + 4]];
        Self {
            batch: w[0],
            m: w[1],
            k: w[2],
            n: w[3],
            b2: w[4],
            a: five(5),
            b: five(10),
            c: five(15),
        }
    }
}

/// `[z, m, n]` output of a folded matmul consumed ONLY by
/// `Reshape [z/b2, b2, m, n] → Transpose(perm) → Reshape(any)` (single-use, and
/// none but the last a graph output): the C strides that write it straight into that final
/// layout, and the final Reshape's id. Merged heads `[B,H,S,Dh] → [B,S,H·Dh]`.
fn output_fold(
    graph: &Graph,
    consumers: &HashMap<NodeId, Vec<NodeId>>,
    outputs: &std::collections::HashSet<NodeId>,
    mm: NodeId,
    b2: usize,
) -> Option<(NodeId, [u32; 5])> {
    let only = |id: NodeId| -> Option<NodeId> {
        match consumers.get(&id).map(|c| c.as_slice()) {
            Some([c]) if !outputs.contains(&id) => Some(*c),
            _ => None,
        }
    };
    let od = static_dims(&graph.node(mm).shape)?;
    let r1 = only(mm)?;
    let d4 = static_dims(&graph.node(r1).shape)?;
    if !matches!(graph.node(r1).op, Op::Reshape { .. })
        || d4.len() != 4
        || d4[1] != b2
        || d4[0] * d4[1] != od[0]
        || d4[2] != od[1]
        || d4[3] != od[2]
    {
        return None;
    }
    let t = only(r1)?;
    let Op::Transpose { perm } = &graph.node(t).op else {
        return None;
    };
    let r2 = only(t)?;
    // r2 may itself be a graph output: the folded matmul takes its place.
    if !matches!(graph.node(r2).op, Op::Reshape { .. }) {
        return None;
    }
    // Transposed tensor is contiguous: Y4[j] = X4[perm[j]], so X axis a has
    // the stride of the Y axis j with perm[j] == a.
    let y4: Vec<usize> = perm.iter().map(|&p| d4[p]).collect();
    let mut sy = [1usize; 4];
    for j in (0..3).rev() {
        sy[j] = sy[j + 1] * y4[j + 1];
    }
    let mut sx = [0usize; 4];
    for (j, &p) in perm.iter().enumerate() {
        sx[p] = sy[j];
    }
    let f = |x: usize| u32::try_from(x).ok();
    Some((r2, [0, f(sx[0])?, f(sx[1])?, f(sx[2])?, f(sx[3])?]))
}

/// A tensor seen as a strided window of `base`: element `idx` is at
/// `off + Σ idx[i]·strides[i]` within `base` (f32 elements).
#[derive(Debug, Clone)]
struct View {
    base: NodeId,
    off: usize,
    dims: Vec<usize>,
    strides: Vec<usize>,
}

fn static_dims(s: &Shape) -> Option<Vec<usize>> {
    s.dims()
        .iter()
        .map(|d| d.is_static().then(|| d.unwrap_static()))
        .collect()
}

fn contiguous(base: NodeId, dims: Vec<usize>) -> View {
    let mut strides = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    View {
        base,
        off: 0,
        dims,
        strides,
    }
}

/// Walk back through copy-only ops (`Narrow`, `Transpose`, a `Reshape` that
/// only SPLITS axes) to the tensor they copy from. Anything else is the base.
fn resolve_view(graph: &Graph, id: NodeId) -> Option<View> {
    let node = graph.node(id);
    let dims = static_dims(&node.shape)?;
    if node.shape.dtype() != rlx_ir::DType::F32 {
        return None;
    }
    let parent = |g: &Graph| resolve_view(g, node.inputs[0]);
    Some(match &node.op {
        Op::Narrow { axis, start, len } => {
            let mut v = parent(graph)?;
            v.off += start * v.strides[*axis];
            v.dims[*axis] = *len;
            v
        }
        Op::Transpose { perm } => {
            let v = parent(graph)?;
            View {
                base: v.base,
                off: v.off,
                dims: perm.iter().map(|&p| v.dims[p]).collect(),
                strides: perm.iter().map(|&p| v.strides[p]).collect(),
            }
        }
        Op::Reshape { .. } => {
            // Split-only: each parent axis becomes a run of new axes whose
            // product is that axis (size-1 axes are free).
            let v = parent(graph)?;
            let mut strides = Vec::with_capacity(dims.len());
            let mut j = 0usize;
            for (pd, ps) in v.dims.iter().zip(&v.strides) {
                let start = j;
                let mut prod = 1usize;
                while j < dims.len() && prod < *pd {
                    prod *= dims[j];
                    j += 1;
                }
                if prod != *pd {
                    return None;
                }
                // Strides inside the run, innermost = the parent stride.
                let mut s = *ps;
                let mut run: Vec<usize> = Vec::new();
                for &d in dims[start..j].iter().rev() {
                    run.push(s);
                    s *= d;
                }
                run.reverse();
                strides.extend(run);
            }
            while j < dims.len() && dims[j] == 1 {
                strides.push(1);
                j += 1;
            }
            if j != dims.len() {
                return None;
            }
            View {
                base: v.base,
                off: v.off,
                dims,
                strides,
            }
        }
        _ => contiguous(id, dims),
    })
}

/// One matmul operand `[z, r, c]` as `(base, [off, s0, s1, sr, sc])` for a
/// batch split `z = z1·b2 + z2`. `None` when the strides don't fit u32.
fn operand(graph: &Graph, id: NodeId, z: usize, b2: usize) -> Option<(NodeId, [u32; 5])> {
    let node = graph.node(id);
    // `[z1, z2, r, c]` merged into `[z, r, c]` by a Reshape: see through it.
    let v4 = match &node.op {
        Op::Reshape { .. } => resolve_view(graph, node.inputs[0]).filter(|v| v.dims.len() == 4),
        _ => None,
    };
    let (base, off, s0, s1, sr, sc) = match v4 {
        Some(v) if v.dims[0] * v.dims[1] == z && v.dims[1] == b2 => (
            v.base,
            v.off,
            v.strides[0],
            v.strides[1],
            v.strides[2],
            v.strides[3],
        ),
        _ => {
            let v = resolve_view(graph, id)?;
            if v.dims.len() != 3 || v.dims[0] != z {
                return None;
            }
            (
                v.base,
                v.off,
                v.strides[0] * b2,
                v.strides[0],
                v.strides[1],
                v.strides[2],
            )
        }
    };
    let f = |x: usize| u32::try_from(x).ok();
    Some((base, [f(off)?, f(s0)?, f(s1)?, f(sr)?, f(sc)?]))
}

/// The `b2` (inner batch) a `[z, ·, ·]` operand exposes through a merging
/// Reshape of a rank-4 view, if any.
fn inner_batch(graph: &Graph, id: NodeId) -> Option<usize> {
    let node = graph.node(id);
    match &node.op {
        Op::Reshape { .. } => resolve_view(graph, node.inputs[0])
            .filter(|v| v.dims.len() == 4)
            .map(|v| v.dims[1]),
        _ => None,
    }
}

/// Rewrite a batched f32 `MatMul` whose operands are copies of a strided
/// window of another tensor — multi-head attention's `narrow → reshape →
/// transpose → reshape` heads — into a [`STRIDED_BMM`] custom node that reads
/// the base tensors directly. The base tensors become the node's inputs, so
/// the memory planner keeps them alive until the matmul (reading through the
/// elided copies' parent would otherwise race the planner's slot reuse). The
/// copies left without a consumer are dropped.
///
/// Only when it removes at least one copy, and only for rank-3 `[z,m,k]·[z,k,n]`.
pub fn fold_strided_bmm(graph: Graph) -> Graph {
    let mut folded: HashMap<NodeId, (Vec<NodeId>, Op)> = HashMap::new();
    // folded matmul -> the final Reshape whose layout it now writes directly.
    let mut out_fold: HashMap<NodeId, NodeId> = HashMap::new();
    let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for n in graph.nodes() {
        for &i in &n.inputs {
            consumers.entry(i).or_default().push(n.id);
        }
    }
    let outputs: std::collections::HashSet<NodeId> = graph.outputs.iter().copied().collect();
    for node in graph.nodes() {
        if !matches!(node.op, Op::MatMul) || node.inputs.len() != 2 {
            continue;
        }
        if node.shape.dtype() != rlx_ir::DType::F32 {
            continue;
        }
        let (a, b) = (node.inputs[0], node.inputs[1]);
        let (Some(od), Some(ad), Some(bd)) = (
            static_dims(&node.shape),
            static_dims(&graph.node(a).shape),
            static_dims(&graph.node(b).shape),
        ) else {
            continue;
        };
        if od.len() != 3 || ad.len() != 3 || bd.len() != 3 || ad[0] != od[0] || bd[0] != od[0] {
            continue;
        }
        let z = od[0];
        let b2 = match (inner_batch(&graph, a), inner_batch(&graph, b)) {
            (Some(x), Some(y)) if x != y => continue,
            (Some(x), _) | (None, Some(x)) => x,
            (None, None) => 1,
        };
        if b2 == 0 || z % b2 != 0 {
            continue;
        }
        let (Some((ab, av)), Some((bb, bv))) =
            (operand(&graph, a, z, b2), operand(&graph, b, z, b2))
        else {
            continue;
        };
        if ab == a && bb == b {
            continue; // nothing to elide
        }
        let (m, n) = (od[1], od[2]);
        let contiguous_c = [0, (b2 * m * n) as u32, (m * n) as u32, n as u32, 1];
        let (c, out_as) = match output_fold(&graph, &consumers, &outputs, node.id, b2) {
            Some((r2, c)) => (c, Some(r2)),
            None => (contiguous_c, None),
        };
        let attrs = StridedBmm {
            batch: z as u32,
            m: m as u32,
            k: ad[2] as u32,
            n: n as u32,
            b2: b2 as u32,
            a: av,
            b: bv,
            c,
        }
        .encode();
        folded.insert(
            node.id,
            (
                vec![ab, bb],
                Op::Custom {
                    name: STRIDED_BMM.to_string(),
                    num_inputs: 2,
                    attrs,
                },
            ),
        );
        if let Some(r2) = out_as {
            out_fold.insert(node.id, r2);
        }
    }
    if folded.is_empty() {
        return graph;
    }
    let inputs_of = |n: &rlx_ir::Node| {
        folded
            .get(&n.id)
            .map(|f| f.0.clone())
            .unwrap_or_else(|| n.inputs.clone())
    };
    let mut uses: HashMap<NodeId, usize> = HashMap::new();
    for n in graph.nodes() {
        for i in inputs_of(n) {
            *uses.entry(i).or_default() += 1;
        }
    }
    for &o in &graph.outputs {
        *uses.entry(o).or_default() += 1;
    }
    let mut dead: std::collections::HashSet<NodeId> = Default::default();
    for n in graph.nodes().iter().rev() {
        let copy_op = matches!(
            n.op,
            Op::Narrow { .. } | Op::Transpose { .. } | Op::Reshape { .. }
        );
        if copy_op && uses.get(&n.id).copied().unwrap_or(0) == 0 {
            dead.insert(n.id);
            for i in &n.inputs {
                if let Some(u) = uses.get_mut(i) {
                    *u -= 1;
                }
            }
        }
    }
    // An output-folded matmul writes its final layout itself: the
    // Reshape → Transpose → Reshape after it go, and the last one's consumers
    // read the matmul.
    for (&mm, &r2) in &out_fold {
        let r1 = consumers[&mm][0];
        let t = consumers[&r1][0];
        dead.extend([r1, t, r2]);
    }
    let mut out = Graph::new(graph.name.clone());
    let mut map: HashMap<NodeId, NodeId> = HashMap::new();
    for n in graph.nodes() {
        if dead.contains(&n.id) {
            continue;
        }
        let ins: Vec<NodeId> = inputs_of(n).iter().map(|i| map[i]).collect();
        let op = folded
            .get(&n.id)
            .map(|f| f.1.clone())
            .unwrap_or_else(|| n.op.clone());
        let shape = match out_fold.get(&n.id) {
            Some(&r2) => graph.node(r2).shape.clone(),
            None => n.shape.clone(),
        };
        let id = out.add_node(op, ins, shape);
        map.insert(n.id, id);
        if let Some(&r2) = out_fold.get(&n.id) {
            map.insert(r2, id);
        }
    }
    out.set_outputs(graph.outputs.iter().map(|o| map[o]).collect());
    out
}
