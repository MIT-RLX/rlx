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
    unfuse_with(graph, true)
}

/// [`unfuse`], with the activation folds or without. Without, a rank-5
/// `FusedConvBiasAct` carrying an activation is expanded as well: `compile`
/// retries this way when a folded node would be replayed on the host (see
/// [`is_folded`]).
pub fn unfuse_with(graph: Graph, folds: bool) -> Graph {
    let g = if folds {
        fold_conv_max(fold_conv_input_prelu(fold_conv_argmax(
            fold_norm_activation(fold_conv_activation(fold_prelu(graph))),
        )))
    } else {
        graph
    };
    let g = fold_conv_bias_into_conv3d(expand_unsupported_fusions(g, folds));
    rlx_unfuse::unfuse(g, &WgpuPolicy)
}

/// Whether [`pack_half_activations`] runs (`RLX_WGPU_HALF_ACTIVATIONS=1`):
/// off under the switches that pick conv kernels without a packed store
/// (`RLX_WGPU_CONV3D_NO_C8`, `_W2`, `_NO_W2`, `_SCALAR`) or a norm without the
/// split (`RLX_WGPU_GROUP_NORM_ONE_WG` is fine — both read and write halves).
pub fn half_activations() -> bool {
    rlx_ir::env::flag("RLX_WGPU_HALF_ACTIVATIONS")
        && ![
            "RLX_WGPU_CONV3D_NO_C8",
            "RLX_WGPU_CONV3D_W2",
            "RLX_WGPU_CONV3D_NO_W2",
            "RLX_WGPU_CONV3D_SCALAR",
        ]
        .iter()
        .any(|k| rlx_ir::env::flag(k))
}

/// Store a conv or norm activation as f16, two to an arena word, when every
/// reader is a kernel that loads halves: a conv3d (`conv3d_ctile*`,
/// `conv3d_argmax`) reading it as its input, a group norm, or a `Reshape`
/// that is itself packed. The node's dtype becomes `F16`, which the hybrid
/// planner sizes at two bytes; the math stays f32 (each load widens, each
/// store rounds). Half the activation memory and traffic: a MeshNet's
/// largest tensors fit a smaller storage binding.
///
/// Packed producers: an ungrouped conv with more than four output channels
/// (`conv3d_ctile_w4c8`, which stores pairs — so an even `w_out`), not a
/// [`CONV3D_ARGMAX`]; a group norm over an even `h·w` (a group starts on a
/// word). A norm's input and output are packed together or not at all: an
/// in-place norm (input and output one slot) must read what it writes.
/// Never a graph output, and nothing past an `Input` or `Param`.
pub fn pack_half_activations(g: Graph) -> Graph {
    use std::collections::HashSet;
    let mut readers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for n in g.nodes() {
        for &i in &n.inputs {
            readers.entry(i).or_default().push(n.id);
        }
    }
    let f32_static = |n: &rlx_ir::Node| {
        n.shape.dtype() == rlx_ir::DType::F32 && n.shape.num_elements().is_some()
    };
    let conv_producer = |n: &rlx_ir::Node| {
        let Some((_, _, _, groups, act, _)) = conv3d_geometry(&n.op) else {
            return false;
        };
        n.shape.rank() == 5
            && groups <= 1
            && act != ARGMAX_ACT
            && n.shape.dim(1).unwrap_static() > 4
            && n.shape.dim(4).unwrap_static().is_multiple_of(2)
    };
    let norm = |n: &rlx_ir::Node| {
        group_norm_act(&n.op).is_some() && {
            let d = n.shape.dims();
            n.shape.rank() == 4 && (d[2].unwrap_static() * d[3].unwrap_static()).is_multiple_of(2)
        }
    };
    let reshape = |n: &rlx_ir::Node| matches!(n.op, Op::Reshape { .. });
    let mut packed: HashSet<NodeId> = g
        .nodes()
        .iter()
        .filter(|n| f32_static(n) && !g.outputs.contains(&n.id))
        .filter(|n| conv_producer(n) || norm(n) || reshape(n))
        .map(|n| n.id)
        .collect();
    // drop what breaks a rule until nothing does
    loop {
        let before = packed.len();
        let snapshot: Vec<NodeId> = packed.iter().copied().collect();
        for id in snapshot {
            let n = g.node(id);
            let reads_ok = readers.get(&id).is_some_and(|rs| {
                rs.iter().all(|&r| {
                    let rn = g.node(r);
                    let as_input = rn.inputs.first() == Some(&id) && !rn.inputs[1..].contains(&id);
                    as_input
                        && ((conv3d_geometry(&rn.op).is_some_and(|c| c.3 <= 1)
                            && rn.shape.rank() >= 4)
                            || packed.contains(&r)
                            || (group_norm_act(&rn.op).is_some() && packed.contains(&r)))
                })
            });
            let input_ok = if reshape(n) || norm(n) {
                packed.contains(&n.inputs[0])
            } else {
                true
            };
            if !reads_ok || !input_ok {
                packed.remove(&id);
            }
        }
        // a norm reading a packed tensor but not packed itself: unpack its input
        for n in g.nodes() {
            if group_norm_act(&n.op).is_some() && !packed.contains(&n.id) {
                packed.remove(&n.inputs[0]);
            }
        }
        if packed.len() == before {
            break;
        }
    }
    // a packed reshape needs a packed producer under it (checked above via
    // input_ok); a lone reshape chain over nothing packed is gone by now
    if packed.is_empty() {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        let ins: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let shape = if packed.contains(&node.id) {
            node.shape.clone().with_dtype(rlx_ir::DType::F16)
        } else {
            node.shape.clone()
        };
        id_map.insert(node.id, out.add_node(node.op.clone(), ins, shape));
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

thread_local! {
    /// Set while compiling a graph [`pack_half_activations`] packed: its
    /// `F16` activations are two halves to an arena word (the hybrid planner's
    /// two bytes), not the f32-widened `F16` of other graphs.
    static HALF_PACKED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Mark (or clear) the graph being compiled as half-packed.
pub fn set_half_packed(on: bool) {
    HALF_PACKED.with(|h| h.set(on));
}

/// Whether the graph being compiled is half-packed.
pub fn half_packed() -> bool {
    HALF_PACKED.with(|h| h.get())
}

/// A tensor [`pack_half_activations`] packed: two halves to an arena word.
pub fn is_half(node: &rlx_ir::Node) -> bool {
    half_packed() && node.shape.dtype() == rlx_ir::DType::F16
}

/// Run [`pack_half_activations`] when it is on and the graph is all f32 —
/// no `F16`/`BF16` tensor whose (widened) storage the packed reading would
/// misread — and record whether it packed anything ([`half_packed`]).
pub fn maybe_pack_half(g: Graph) -> Graph {
    set_half_packed(false);
    if rlx_ir::env::flag("RLX_WGPU_STRIPE_VERBOSE") {
        eprintln!("[rlx-wgpu] half activations: {}", half_activations());
    }
    if !half_activations()
        || g.nodes()
            .iter()
            .any(|n| matches!(n.shape.dtype(), rlx_ir::DType::F16 | rlx_ir::DType::BF16))
    {
        return g;
    }
    let g = pack_half_activations(g);
    let n = g
        .nodes()
        .iter()
        .filter(|n| n.shape.dtype() == rlx_ir::DType::F16)
        .count();
    if rlx_ir::env::flag("RLX_WGPU_STRIPE_VERBOSE") {
        eprintln!("[rlx-wgpu] half-packed {n} activations");
    }
    set_half_packed(n > 0);
    g
}

/// A node only this backend's GPU kernels run: a rank-5 `FusedConvBiasAct`
/// with an activation, a [`CONV3D_LEAKY`] or a [`GROUP_NORM_ACT`]. The host
/// replay (`Step::HostOp`, rlx-cpu's single-op evaluator) knows none of them
/// — it ran the first as a 2-D conv and the others not at all, and the labels
/// of a striped arena (a 1 GiB storage binding) came back wrong.
pub fn is_folded(node: &rlx_ir::Node) -> bool {
    match &node.op {
        Op::FusedConvBiasAct {
            activation: Some(_),
            ..
        } => node.shape.rank() == 5,
        Op::Custom { name, .. } => {
            name == CONV3D_LEAKY
                || name == GROUP_NORM_ACT
                || name == CONV3D_ARGMAX
                || name == CONV3D_TOPK
                || name == PRELU
                || name == CONV3D_K3
        }
        _ => false,
    }
}

/// `Op::Custom` name of a rank-5 `FusedConvBiasAct` (same inputs, same
/// output) whose store applies LeakyReLU, `max(v, α·v)`; see
/// [`fold_conv_activation`]. Attrs: stride, padding, dilation (3 each),
/// groups, α bits.
pub const CONV3D_LEAKY: &str = "rlx_wgpu.conv3d_leaky";

/// `(stride, padding, dilation, groups, act, α)` of a convolution the conv3d
/// shaders run: `Op::Conv3d`, a rank-5 `FusedConvBiasAct` (its activation as
/// `opcode_relu_first + 1`, `0` for none) or a [`CONV3D_LEAKY`]
/// (`act == LEAKY_ACT`).
#[allow(clippy::type_complexity)]
pub fn conv3d_geometry(op: &Op) -> Option<([usize; 3], [usize; 3], [usize; 3], usize, u32, f32)> {
    let a3 = |v: &[usize]| [v[0], v[1], v[2]];
    match op {
        Op::Conv3d {
            stride,
            padding,
            dilation,
            groups,
        } => Some((*stride, *padding, *dilation, *groups, 0, 0.0)),
        Op::FusedConvBiasAct {
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            activation,
            has_residual: false,
        } if kernel_size.len() == 3 => Some((
            a3(stride),
            a3(padding),
            a3(dilation),
            *groups,
            activation.map_or(0, |a| a.opcode_relu_first() + 1),
            0.0,
        )),
        Op::Custom { name, attrs, .. } if name == CONV3D_K3 => {
            let w: Vec<u32> = attrs
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            let u = |i: usize| w[i] as usize;
            Some((
                [u(0), u(1), u(2)],
                [u(3), u(4), u(5)],
                [u(6), u(7), u(8)],
                u(9),
                w[10],
                f32::from_bits(w[11]),
            ))
        }
        Op::Custom { name, attrs, .. }
            if name == CONV3D_LEAKY || name == CONV3D_ARGMAX || name == CONV3D_TOPK =>
        {
            let w: Vec<u32> = attrs
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            let u = |i: usize| w[i] as usize;
            Some((
                [u(0), u(1), u(2)],
                [u(3), u(4), u(5)],
                [u(6), u(7), u(8)],
                u(9),
                if name == CONV3D_ARGMAX {
                    ARGMAX_ACT
                } else if name == CONV3D_TOPK {
                    TOPK_ACT
                } else {
                    LEAKY_ACT
                },
                w.get(10).map_or(0.0, |&b| f32::from_bits(b)),
            ))
        }
        _ => None,
    }
}

/// `Conv3dParams::act` for LeakyReLU, slope in `Conv3dParams::alpha`.
pub const LEAKY_ACT: u32 = 0xFFFF;

/// `Conv3dParams::act` of a [`CONV3D_ARGMAX`]: the store writes the index of
/// the largest output channel, not the channels.
pub const ARGMAX_ACT: u32 = 0xFFFE;

/// `Op::Custom` name of a 1×1×1 conv3d (a rank-5 `FusedConvBiasAct`'s or a
/// `Conv3d`'s inputs) and the `ArgMax` over its channels that was its only
/// reader; the output is the argmax's. Attrs as [`CONV3D_LEAKY`]'s, without α.
/// See [`fold_conv_argmax`].
pub const CONV3D_ARGMAX: &str = "rlx_wgpu.conv3d_argmax";

/// `Op::Custom` name of a PReLU with one slope, `x` where `x ≥ 0`, else `α·x`
/// (attrs: `α` as f32 bits): one pass of the unary kernel
/// ([`PRELU_OP`]), where a graph of max, min, multiply and add took four —
/// FastSurfer has 44 a slice batch, a quarter of its GPU time.
pub const PRELU: &str = "rlx_wgpu.prelu";

/// The unary kernel's op id for [`PRELU`] (`α` in `_p0`).
pub const PRELU_OP: u32 = 0xFFFF;

/// The unary kernel's op id for a [`PRELU`] whose slope is a tensor (its
/// second input; attrs: elements a channel, channels): `_p0` the slope's
/// offset, `_p1`, `_p2` the attrs.
pub const PRELU_TENSOR_OP: u32 = 0xFFFE;

/// Run each PReLU [`rlx_ir::match_prelu`] finds — [`Graph::prelu`]'s
/// `max(x, 0) + a·min(x, 0)`, the ONNX importer's `relu(x) − a·relu(−x)` —
/// as one [`PRELU`], when nothing else reads its interior: a constant slope
/// as the op's attribute (which [`fold_conv_input_prelu`] can take into a
/// conv's loads), any other slope — a scalar, or one value per channel of a
/// learned weight — as its second input. Four or five full-size passes
/// become one, the same values (each output is `x` or `a·x` either way).
/// `RLX_WGPU_NO_FOLD_PRELU=1` keeps the element-wise ops.
fn fold_prelu(g: Graph) -> Graph {
    use std::collections::HashSet;
    if rlx_ir::env::flag("RLX_WGPU_NO_FOLD_PRELU") {
        return g;
    }
    let mut readers: HashMap<NodeId, usize> = HashMap::new();
    for n in g.nodes() {
        for &i in &n.inputs {
            *readers.entry(i).or_default() += 1;
        }
    }
    // root -> (x, slope as attrs, or slope node and its (inner, count))
    enum Slope {
        Const(f32),
        Tensor(NodeId, u32, u32),
    }
    let mut folds: HashMap<NodeId, (NodeId, Slope)> = HashMap::new();
    let mut gone: HashSet<NodeId> = HashSet::new();
    for n in g.nodes() {
        let Some(m) = rlx_ir::match_prelu(&g, n.id) else {
            continue;
        };
        let xs = &g.node(m.x).shape;
        if xs.dtype() != rlx_ir::DType::F32 || xs.num_elements().is_none() {
            continue;
        }
        // the interior read only inside the match: max/min (or relu and
        // neg) by one reader each, the min's/neg's input `x` aside
        let interior_ok = m
            .interior
            .iter()
            .all(|&i| readers.get(&i) == Some(&1) && !g.outputs.contains(&i) && !gone.contains(&i));
        if !interior_ok {
            continue;
        }
        let slope = match m.constant_slope(&g) {
            Some(a) => Slope::Const(a),
            None => {
                // one non-1 dim of the slope, right-aligned against `x`: a
                // channel axis; none: a scalar
                let sd: Vec<usize> = match g
                    .node(m.slope)
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.is_static().then(|| d.unwrap_static()))
                    .collect::<Option<Vec<_>>>()
                {
                    Some(v) => v,
                    None => continue,
                };
                let xd: Vec<usize> = xs.dims().iter().map(|d| d.unwrap_static()).collect();
                if sd.len() > xd.len() || g.node(m.slope).shape.dtype() != rlx_ir::DType::F32 {
                    continue;
                }
                let lead = xd.len() - sd.len();
                let wide: Vec<usize> = (0..sd.len()).filter(|&k| sd[k] != 1).collect();
                match wide.as_slice() {
                    [] => Slope::Tensor(m.slope, 1, 1),
                    [k] if sd[*k] == xd[lead + k] => {
                        let inner: usize = xd[lead + k + 1..].iter().product();
                        Slope::Tensor(m.slope, inner.max(1) as u32, sd[*k] as u32)
                    }
                    _ => continue,
                }
            }
        };
        gone.extend(m.interior.iter().copied());
        folds.insert(n.id, (m.x, slope));
    }
    if folds.is_empty() {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        if gone.contains(&node.id) {
            continue;
        }
        let (op, ins) = match folds.get(&node.id) {
            Some((x, Slope::Const(a))) => (
                Op::Custom {
                    name: PRELU.to_string(),
                    num_inputs: 1,
                    attrs: a.to_bits().to_le_bytes().to_vec(),
                },
                vec![id_map[x]],
            ),
            Some((x, Slope::Tensor(sl, inner, count))) => (
                Op::Custom {
                    name: PRELU.to_string(),
                    num_inputs: 2,
                    attrs: [*inner, *count]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                },
                vec![id_map[x], id_map[sl]],
            ),
            None => (
                node.op.clone(),
                node.inputs.iter().map(|i| id_map[i]).collect(),
            ),
        };
        id_map.insert(node.id, out.add_node(op, ins, node.shape.clone()));
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

/// `Conv3dParams::act` of a [`CONV3D_TOPK`]; `alpha` carries `k`.
pub const TOPK_ACT: u32 = 0xFFFC;

/// `Op::Custom` name of a 1×1×1, stride-1, unpadded, ungrouped conv whose
/// store keeps, per voxel, only its `k` largest output channels — a
/// classifier read by a vote over several networks, whose full scores
/// (FastSurfer's 79 classes, a slice batch at a time) are the largest
/// tensor and the largest read-back. Inputs as a biased conv (`x`, `w`, `b`);
/// attrs: stride, padding, dilation (3 each), groups, `k` as f32 bits.
/// Output `[n, 2k + 1, d, h, w]`: the `k` largest values, best first (the
/// earlier channel first among equals), the `k + 1`-th (a bound on all the
/// rest), then the `k` channels, f32-encoded. Built by the caller (no fold
/// makes it); `k` at most 8.
pub const CONV3D_TOPK: &str = "rlx_wgpu.conv3d_topk";

/// `Op::Custom` name of a 2-D 3×3 conv (`conv3d_k3_c8`'s shape) with the
/// element-wise op on either side of it folded in:
///
/// * flag bit 0: its input read through a PReLU of one slope, applied as each
///   value is loaded (`Conv3dParams::in_prelu`; [`fold_conv_input_prelu`]);
/// * flag bit 1: its output the elementwise `max` with a fourth input of its
///   shape, taken in the store (`Conv3dParams::res_max`; [`fold_conv_max`]).
///
/// The folded tensors are never written or read. Inputs `x` (the PReLU's
/// input), `w`, `b`, and the `max`'s other operand; attrs: stride, padding,
/// dilation (3 each), groups, the store's `act` and `α` (as
/// [`conv3d_geometry`] gives them), the input slope as f32 bits, the flags.
pub const CONV3D_K3: &str = "rlx_wgpu.conv3d_k3";

fn conv3d_k3_word(op: &Op, k: usize) -> Option<u32> {
    match op {
        Op::Custom { name, attrs, .. } if name == CONV3D_K3 => Some(u32::from_le_bytes(
            attrs[4 * k..4 * k + 4].try_into().unwrap(),
        )),
        _ => None,
    }
}

/// The input slope of a [`CONV3D_K3`] that has one.
pub fn conv3d_in_prelu(op: &Op) -> Option<f32> {
    let flags = conv3d_k3_word(op, 13)?;
    (flags & 1 != 0).then(|| f32::from_bits(conv3d_k3_word(op, 12).unwrap()))
}

/// Whether a [`CONV3D_K3`] takes the `max` with its fourth input.
pub fn conv3d_max_residual(op: &Op) -> bool {
    conv3d_k3_word(op, 13).is_some_and(|f| f & 2 != 0)
}

/// Off under the switches that take `conv3d_k3_c8` away (the folds need it).
fn k3_folds_off(own: &str) -> bool {
    [
        own,
        "RLX_WGPU_CONV3D_NO_K3",
        "RLX_WGPU_CONV3D_NO_C8",
        "RLX_WGPU_CONV3D_W2",
        "RLX_WGPU_CONV3D_NO_W2",
        "RLX_WGPU_CONV3D_SCALAR",
    ]
    .iter()
    .any(|k| rlx_ir::env::flag(k))
}

/// A biased rank-5 conv `conv3d_k3_c8` runs: kernel `[1, 3, 3]`, stride 1,
/// padding `[0, 1, 1]`, no dilation, ungrouped, past four output channels —
/// a `FusedConvBiasAct`, a [`CONV3D_LEAKY`] or a [`CONV3D_K3`].
fn k3_conv(g: &Graph, n: &rlx_ir::Node) -> bool {
    let kind = matches!(
        n.op,
        Op::FusedConvBiasAct {
            has_residual: false,
            ..
        }
    ) || matches!(&n.op, Op::Custom { name, .. } if name == CONV3D_LEAKY || name == CONV3D_K3);
    let Some((s, pad, d, groups, ..)) = conv3d_geometry(&n.op) else {
        return false;
    };
    if !kind || n.inputs.len() < 3 || n.shape.rank() != 5 {
        return false;
    }
    let w = g.node(n.inputs[1]).shape.dims();
    w.len() == 5
        && (2..5).all(|i| w[i].is_static())
        && (
            w[2].unwrap_static(),
            w[3].unwrap_static(),
            w[4].unwrap_static(),
        ) == (1, 3, 3)
        && s == [1, 1, 1]
        && pad == [0, 1, 1]
        && d == [1, 1, 1]
        && groups <= 1
        && n.shape.dim(1).unwrap_static() > 4
}

/// `node` as a [`CONV3D_K3`] with these inputs, its input slope and flags
/// (added to what it has, when it is one already)
fn as_k3(node: &rlx_ir::Node, in_slope: Option<f32>, add_flags: u32, n_in: usize) -> Op {
    let (s, p, d, groups, act, alpha) = conv3d_geometry(&node.op).unwrap();
    let flags = conv3d_k3_word(&node.op, 13).unwrap_or(0) | add_flags;
    let slope = in_slope
        .or_else(|| conv3d_in_prelu(&node.op))
        .unwrap_or(0.0);
    Op::Custom {
        name: CONV3D_K3.to_string(),
        num_inputs: n_in as u32,
        attrs: s
            .iter()
            .chain(&p)
            .chain(&d)
            .map(|&v| v as u32)
            .chain([groups as u32, act, alpha.to_bits(), slope.to_bits(), flags])
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    }
}

/// Fold a [`PRELU`] whose every reader is a conv `conv3d_k3_c8` runs, reading
/// it as its input, into those convs: [`CONV3D_K3`]s reading the PReLU's
/// input, the slope applied as each value is loaded — the same
/// `select(α·x, x, x ≥ 0)` as the unary kernel, so the same sums; a
/// full-size write and two reads per PReLU are gone (FastSurfer: 39 a
/// batch). `RLX_WGPU_NO_FOLD_CONV_PRELU=1` keeps them.
fn fold_conv_input_prelu(g: Graph) -> Graph {
    if k3_folds_off("RLX_WGPU_NO_FOLD_CONV_PRELU") {
        return g;
    }
    let mut readers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for n in g.nodes() {
        for &i in &n.inputs {
            readers.entry(i).or_default().push(n.id);
        }
    }
    // a reader that takes the fold: the PReLU its input and nothing else,
    // no input slope of its own yet
    let k3_reader = |r: NodeId, p: NodeId| {
        let n = g.node(r);
        k3_conv(&g, n)
            && n.inputs[0] == p
            && !n.inputs[1..].contains(&p)
            && conv3d_in_prelu(&n.op).is_none()
    };
    let mut slope: HashMap<NodeId, f32> = HashMap::new(); // conv -> input slope
    let mut gone: HashMap<NodeId, NodeId> = HashMap::new(); // prelu -> its input
    for n in g.nodes() {
        let Op::Custom { name, attrs, .. } = &n.op else {
            continue;
        };
        // the slope an attribute (a constant), not a tensor input
        if name != PRELU || n.inputs.len() != 1 || g.outputs.contains(&n.id) {
            continue;
        }
        let rs = readers.get(&n.id).map_or(&[][..], |v| v.as_slice());
        if rs.is_empty() || !rs.iter().all(|&r| k3_reader(r, n.id)) {
            continue;
        }
        let a = f32::from_le_bytes(attrs[..4].try_into().unwrap());
        for &r in rs {
            slope.insert(r, a);
        }
        gone.insert(n.id, n.inputs[0]);
    }
    if gone.is_empty() {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        if gone.contains_key(&node.id) {
            continue;
        }
        let ins: Vec<NodeId> = node
            .inputs
            .iter()
            .map(|i| id_map[gone.get(i).unwrap_or(i)])
            .collect();
        let op = match slope.get(&node.id) {
            Some(&a) => as_k3(node, Some(a), 1, ins.len()),
            None => node.op.clone(),
        };
        id_map.insert(node.id, out.add_node(op, ins, node.shape.clone()));
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

/// Fold an element-wise `max(conv, r)` — the conv's only reader, `r` of the
/// conv's shape — into a conv `conv3d_k3_c8` runs with no activation of its
/// own: a [`CONV3D_K3`] taking `r` as a fourth input and the `max` in its
/// store (FastSurfer's competitive dense blocks: 36 a batch). The same
/// values: the store's `max` is the binary kernel's. The fused node takes the
/// `max`'s place in the order (`r` may come after the conv).
/// `RLX_WGPU_NO_FOLD_CONV_MAX=1` keeps them.
fn fold_conv_max(g: Graph) -> Graph {
    use rlx_ir::op::BinaryOp;
    if k3_folds_off("RLX_WGPU_NO_FOLD_CONV_MAX") {
        return g;
    }
    let mut readers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for n in g.nodes() {
        for &i in &n.inputs {
            readers.entry(i).or_default().push(n.id);
        }
    }
    // max node -> (conv, the other operand)
    let mut folds: HashMap<NodeId, (NodeId, NodeId)> = HashMap::new();
    let mut convs: std::collections::HashSet<NodeId> = Default::default();
    for n in g.nodes() {
        let no_act = conv3d_geometry(&n.op).is_some_and(|c| c.4 == 0);
        if !k3_conv(&g, n)
            || !no_act
            || n.inputs.len() != 3
            || conv3d_max_residual(&n.op)
            || g.outputs.contains(&n.id)
            || n.shape.dtype() != rlx_ir::DType::F32
        {
            continue;
        }
        let [m] = readers.get(&n.id).map_or(&[][..], |v| v.as_slice()) else {
            continue;
        };
        let mn = g.node(*m);
        if !matches!(mn.op, Op::Binary(BinaryOp::Max)) || mn.shape != n.shape {
            continue;
        }
        let r = if mn.inputs[0] == n.id {
            mn.inputs[1]
        } else {
            mn.inputs[0]
        };
        if r == n.id || g.node(r).shape != n.shape {
            continue;
        }
        folds.insert(*m, (n.id, r));
        convs.insert(n.id);
    }
    if folds.is_empty() {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        if convs.contains(&node.id) {
            // emitted at its max
            continue;
        }
        let (op, ins) = match folds.get(&node.id) {
            Some(&(c, r)) => {
                let cn = g.node(c);
                let mut ins: Vec<NodeId> = cn.inputs.iter().map(|i| id_map[i]).collect();
                ins.push(id_map[&r]);
                (as_k3(cn, None, 2, 4), ins)
            }
            None => (
                node.op.clone(),
                node.inputs.iter().map(|i| id_map[i]).collect(),
            ),
        };
        let shape = node.shape.clone();
        let nid = out.add_node(op, ins, shape);
        id_map.insert(node.id, nid);
        if let Some(&(c, _)) = folds.get(&node.id) {
            id_map.insert(c, nid);
        }
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

/// A scalar `Op::Constant`'s value.
fn scalar_constant(g: &Graph, id: NodeId) -> Option<f32> {
    match &g.node(id).op {
        Op::Constant { data } if data.len() == 4 => {
            Some(f32::from_le_bytes(data[..4].try_into().unwrap()))
        }
        _ => None,
    }
}

/// Fold the activation reading a rank-5 `FusedConvBiasAct` (no activation,
/// no residual, not a graph output) into it:
///
/// * an [`Op::Activation`], the conv's only reader, becomes the conv's own
///   activation;
/// * LeakyReLU spelled `max(y, α·y)` (SynthStrip's: a `Mul` by a scalar
///   constant, the conv's only readers being that and the `Max`) becomes a
///   [`CONV3D_LEAKY`].
///
/// The conv3d shaders apply it in their store (`Conv3dParams::act`) with the
/// same arithmetic as the kernels it replaces — the generated
/// `rlx_activation_apply` of the unary kernel, or a `*` and a `max` — so the
/// result is bit-identical, and a full-size read and write per layer (two for
/// LeakyReLU) is gone: 17% of a 256^3 brainchop MeshNet's GPU time, a third
/// of MindGrab's.
fn fold_conv_activation(g: Graph) -> Graph {
    if rlx_ir::env::flag("RLX_WGPU_NO_FOLD_CONV_ACT") {
        return g;
    }
    let mut readers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for n in g.nodes() {
        for &i in &n.inputs {
            readers.entry(i).or_default().push(n.id);
        }
    }
    let readers_of = |id: NodeId| readers.get(&id).map_or(&[][..], |v| v.as_slice());
    let foldable = |id: NodeId| {
        let n = g.node(id);
        matches!(
            n.op,
            Op::FusedConvBiasAct {
                activation: None,
                has_residual: false,
                ..
            }
        ) && n.shape.rank() == 5
            && n.inputs.len() == 3
            && !g.outputs.contains(&id)
    };
    enum Fold {
        Act(rlx_ir::op::Activation),
        Leaky(f32),
    }
    // conv id -> (its activation, the node now produced by the conv), and
    // the nodes folded away -> the conv standing in for them
    let mut folds: HashMap<NodeId, Fold> = HashMap::new();
    let mut gone: HashMap<NodeId, NodeId> = HashMap::new();
    for n in g.nodes() {
        if !foldable(n.id) {
            continue;
        }
        match readers_of(n.id) {
            [r] => {
                let r = g.node(*r);
                if let Op::Activation(a) = r.op
                    && r.shape == n.shape
                {
                    folds.insert(n.id, Fold::Act(a));
                    gone.insert(r.id, n.id);
                }
            }
            [a, b] => {
                use rlx_ir::op::BinaryOp;
                let is = |id: NodeId, op: BinaryOp| {
                    matches!(g.node(id).op, Op::Binary(o) if o == op) && g.node(id).shape == n.shape
                };
                let (mul, max) = if is(*a, BinaryOp::Mul) {
                    (*a, *b)
                } else {
                    (*b, *a)
                };
                if !(is(mul, BinaryOp::Mul) && is(max, BinaryOp::Max)) {
                    continue;
                }
                let mi = &g.node(mul).inputs;
                let alpha = if mi[0] == n.id {
                    scalar_constant(&g, mi[1])
                } else {
                    scalar_constant(&g, mi[0])
                };
                let xi = &g.node(max).inputs;
                let pairs = (xi[0] == n.id && xi[1] == mul) || (xi[1] == n.id && xi[0] == mul);
                if let Some(alpha) = alpha
                    && pairs
                    && readers_of(mul) == [max]
                    && !g.outputs.contains(&mul)
                {
                    folds.insert(n.id, Fold::Leaky(alpha));
                    gone.insert(mul, n.id);
                    gone.insert(max, n.id);
                }
            }
            _ => {}
        }
    }
    if folds.is_empty() {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        if let Some(conv) = gone.get(&node.id) {
            // the conv now produces this node's value
            id_map.insert(node.id, id_map[conv]);
            continue;
        }
        let ins: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let op = match (folds.get(&node.id), &node.op) {
            (Some(Fold::Act(a)), Op::FusedConvBiasAct { .. }) => {
                let mut op = node.op.clone();
                if let Op::FusedConvBiasAct { activation, .. } = &mut op {
                    *activation = Some(*a);
                }
                op
            }
            (Some(Fold::Leaky(alpha)), op) => {
                let (s, p, d, groups, ..) = conv3d_geometry(op).unwrap();
                Op::Custom {
                    name: CONV3D_LEAKY.to_string(),
                    num_inputs: 3,
                    attrs: s
                        .iter()
                        .chain(&p)
                        .chain(&d)
                        .map(|&v| v as u32)
                        .chain([groups as u32, alpha.to_bits()])
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                }
            }
            _ => node.op.clone(),
        };
        let nid = out.add_node(op, ins, node.shape.clone());
        id_map.insert(node.id, nid);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

/// Fold an `ArgMax` over the channels of a 1×1×1 conv3d (stride 1, no
/// padding, ungrouped, no activation; a rank-5 `FusedConvBiasAct` or a
/// `Conv3d`) that is the conv's only reader into a [`CONV3D_ARGMAX`]: one
/// pass computes each voxel's logits and keeps the index of the largest. A
/// segmentation net's class logits are never stored — for a 104-class
/// brainchop model, a tensor of 104 channels over the volume, most of its
/// arena, written and read back once. `RLX_WGPU_NO_FOLD_CONV_ARGMAX` turns it
/// off.
fn fold_conv_argmax(g: Graph) -> Graph {
    if rlx_ir::env::flag("RLX_WGPU_NO_FOLD_CONV_ARGMAX") {
        return g;
    }
    let mut readers: HashMap<NodeId, usize> = HashMap::new();
    for n in g.nodes() {
        for &i in &n.inputs {
            *readers.entry(i).or_default() += 1;
        }
    }
    let pointwise = |id: NodeId| {
        let n = g.node(id);
        let w = &g.node(*n.inputs.get(1)?).shape;
        let plain = match &n.op {
            Op::Conv3d { .. } => n.inputs.len() == 2,
            Op::FusedConvBiasAct {
                activation: None,
                has_residual: false,
                ..
            } => n.inputs.len() == 3,
            _ => false,
        };
        let (s, p, _, groups, ..) = conv3d_geometry(&n.op)?;
        (plain
            && n.shape.rank() == 5
            && w.rank() == 5
            && (2..5).all(|a| w.dim(a).is_static() && w.dim(a).unwrap_static() == 1)
            && s == [1; 3]
            && p == [0; 3]
            && groups <= 1
            && readers.get(&id) == Some(&1)
            && !g.outputs.contains(&id))
        .then_some(())
    };
    let folds: HashMap<NodeId, NodeId> = g
        .nodes()
        .iter()
        .filter(|n| {
            matches!(n.op, Op::ArgMax { axis: 1, .. })
                && n.shape.rank() == 4
                && pointwise(n.inputs[0]).is_some()
        })
        .map(|n| (n.id, n.inputs[0]))
        .collect();
    if folds.is_empty() {
        return g;
    }
    let convs: std::collections::HashSet<NodeId> = folds.values().copied().collect();
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        if convs.contains(&node.id) {
            continue;
        }
        let nid = match folds.get(&node.id) {
            Some(&conv) => {
                let c = g.node(conv);
                let (s, p, d, groups, ..) = conv3d_geometry(&c.op).unwrap();
                let ins: Vec<NodeId> = c.inputs.iter().map(|i| id_map[i]).collect();
                out.add_node(
                    Op::Custom {
                        name: CONV3D_ARGMAX.to_string(),
                        num_inputs: ins.len() as u32,
                        attrs: s
                            .iter()
                            .chain(&p)
                            .chain(&d)
                            .map(|&v| v as u32)
                            .chain([groups.max(1) as u32])
                            .flat_map(|v| v.to_le_bytes())
                            .collect(),
                    },
                    ins,
                    node.shape.clone(),
                )
            }
            None => {
                let ins: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                out.add_node(node.op.clone(), ins, node.shape.clone())
            }
        };
        id_map.insert(node.id, nid);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

/// `Op::Custom` name of an `Op::GroupNorm` (same inputs, same output) that
/// applies an activation in its write-back; see [`fold_norm_activation`].
pub const GROUP_NORM_ACT: &str = "rlx_wgpu.group_norm_act";

/// `(num_groups, eps, activation)` of a group norm: an `Op::GroupNorm`, or a
/// [`GROUP_NORM_ACT`] node (attrs: `num_groups`, `eps` bits, opcode).
pub fn group_norm_act(op: &Op) -> Option<(usize, f32, Option<rlx_ir::op::Activation>)> {
    match op {
        Op::GroupNorm { num_groups, eps } => Some((*num_groups, *eps, None)),
        Op::Custom { name, attrs, .. } if name == GROUP_NORM_ACT => {
            let w = |i: usize| u32::from_le_bytes(attrs[4 * i..4 * i + 4].try_into().unwrap());
            let act = rlx_ir::op::Activation::ALL
                .into_iter()
                .find(|a| a.opcode_relu_first() == w(2))
                .expect("rlx-wgpu: GROUP_NORM_ACT carries a known activation");
            Some((w(0) as usize, f32::from_bits(w(1)), Some(act)))
        }
        _ => None,
    }
}

/// Fold an [`Op::Activation`] into the `Op::GroupNorm` it reads — directly
/// or through one `Reshape` (an instance norm over a volume runs on it
/// flattened) — when the norm and the reshape have no other reader and are
/// not graph outputs. The norm becomes a [`GROUP_NORM_ACT`] whose write-back
/// applies the activation, so a full-size read and write is gone per norm;
/// MindGrab has one after each of its 17 norms.
fn fold_norm_activation(g: Graph) -> Graph {
    if rlx_ir::env::flag("RLX_WGPU_NO_FOLD_NORM_ACT") {
        return g;
    }
    let mut readers: HashMap<NodeId, usize> = HashMap::new();
    for n in g.nodes() {
        for &i in &n.inputs {
            *readers.entry(i).or_default() += 1;
        }
    }
    let sole = |id: NodeId| readers.get(&id) == Some(&1) && !g.outputs.contains(&id);
    // norm id -> (activation, the node standing in for the activation)
    let mut folds: HashMap<NodeId, (rlx_ir::op::Activation, NodeId)> = HashMap::new();
    for n in g.nodes() {
        let Op::Activation(a) = n.op else { continue };
        let x = n.inputs[0];
        let (norm, stand_in) = match g.node(x).op {
            Op::Reshape { .. } => (g.node(x).inputs[0], x),
            _ => (x, x),
        };
        let Op::GroupNorm { .. } = g.node(norm).op else {
            continue;
        };
        if sole(x)
            && (norm == x || sole(norm))
            && n.shape.num_elements() == g.node(norm).shape.num_elements()
        {
            folds.insert(norm, (a, stand_in));
        }
    }
    if folds.is_empty() {
        return g;
    }
    let skip: HashMap<NodeId, NodeId> = g
        .nodes()
        .iter()
        .filter_map(|n| match n.op {
            Op::Activation(_) => {
                let x = n.inputs[0];
                let norm = match g.node(x).op {
                    Op::Reshape { .. } => g.node(x).inputs[0],
                    _ => x,
                };
                folds.get(&norm).filter(|f| f.1 == x).map(|_| (n.id, x))
            }
            _ => None,
        })
        .collect();
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        if let Some(&x) = skip.get(&node.id) {
            // the norm (or its reshape) now produces the activation's value
            id_map.insert(node.id, id_map[&x]);
            continue;
        }
        let ins: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let op = match (&node.op, folds.get(&node.id)) {
            (Op::GroupNorm { num_groups, eps }, Some(&(a, _))) => Op::Custom {
                name: GROUP_NORM_ACT.to_string(),
                num_inputs: 3,
                attrs: [*num_groups as u32, eps.to_bits(), a.opcode_relu_first()]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            },
            _ => node.op.clone(),
        };
        let nid = out.add_node(op, ins, node.shape.clone());
        id_map.insert(node.id, nid);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
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
fn expand_unsupported_fusions(g: Graph, folds: bool) -> Graph {
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
            // ... and so is rank 5 carrying `fold_conv_activation`'s activation.
            Op::FusedConvBiasAct {
                has_residual: false,
                ..
            } if n.shape.rank() == 5 => !folds || n.inputs.len() != 3,
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
