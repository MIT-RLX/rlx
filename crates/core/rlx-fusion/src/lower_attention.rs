// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Attention` → matmul / softmax / elementwise, for backends without an
//! attention kernel (WebGL2).
//!
//! Mirrors the CPU kernel step for step, so a lowered graph computes what the
//! fused op computes there:
//!
//! 1. `scores = q · kᵀ · scale` per head, `scale` defaulting to `head_dim^-½`;
//! 2. `MaskKind::Custom` — a `[batch, k_seq]` key mask: keys with value `< 0.5`
//!    get score `-1e9` (replaced, not added);
//! 3. `MaskKind::Bias` — a `[batch, heads, q_seq, k_seq]` tensor is added;
//! 4. `Causal` / `SlidingWindow(w)` — query `i` sits at absolute position
//!    `k_seq − q_seq + i` (a decode step's query is the newest row), and keys
//!    past it (or more than `w` before it) get `-1e9`;
//! 5. logit soft-cap `c · tanh(s / c)` when set — after masking, as there;
//! 6. softmax over keys, `· v`.
//!
//! Layouts: rank-3 `[B, S, H·D]`, rank-2 `[S, H·D]`, and rank-4 either
//! `[B, H, S, D]` or `[B, S, H, D]` — told apart, as the CPU kernel does, by
//! which axis equals `num_heads`. K/V may carry fewer heads (GQA); query head
//! `h` reads KV head `h / (heads / kv_heads)`. "Replace with −1e9" is computed
//! as `s·(1−drop) + drop·(−1e9)` with a 0/1 `drop`, which is exact for finite
//! scores and needs no boolean tensors.

use crate::pass::Pass;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::MaskKind;
use rlx_ir::*;
use std::collections::HashMap;

/// The CPU kernel's masked score (`RuntimeConfig::attn_mask_neg_inf`).
const MASKED: f32 = -1e9;
/// The CPU kernel's key-mask threshold (`mask_binary_threshold`).
const KEEP_THRESHOLD: f32 = 0.5;

/// `Op::Attention` → primitive ops (see the module docs).
pub struct LowerAttention;

impl Pass for LowerAttention {
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::Attention]
    }

    fn name(&self) -> &str {
        "lower_attention"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Attention { .. }))
        {
            return graph;
        }
        let mut out = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        for node in graph.nodes() {
            let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
            let new_id = match &node.op {
                Op::Attention {
                    num_heads,
                    head_dim,
                    v_head_dim,
                    mask_kind,
                    score_scale,
                    attn_logit_softcap,
                } => lower_one(
                    &mut out,
                    &inputs,
                    &node.shape,
                    Spec {
                        heads: *num_heads,
                        head_dim: *head_dim,
                        v_head_dim: v_head_dim.unwrap_or(*head_dim),
                        mask: *mask_kind,
                        scale: score_scale.unwrap_or((*head_dim as f32).powf(-0.5)),
                        softcap: attn_logit_softcap.filter(|c| *c > 0.0),
                    },
                ),
                _ => out.add_node(node.op.clone(), inputs, node.shape.clone()),
            };
            id_map.insert(node.id, new_id);
        }
        out.set_outputs(graph.outputs.iter().map(|o| id_map[o]).collect());
        out
    }
}

#[derive(Clone, Copy)]
struct Spec {
    heads: usize,
    head_dim: usize,
    v_head_dim: usize,
    mask: MaskKind,
    scale: f32,
    softcap: Option<f32>,
}

/// How a Q/K/V tensor lays out its axes.
#[derive(Clone, Copy, PartialEq)]
enum Layout {
    /// `[S, H·D]`
    Rank2,
    /// `[B, S, H·D]`
    Rank3,
    /// `[B, H, S, D]`
    Bhsd,
    /// `[B, S, H, D]`
    Bshd,
}

fn dims(g: &Graph, x: NodeId) -> Vec<usize> {
    g.shape(x)
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect()
}

fn layout_of(d: &[usize], heads: usize) -> Layout {
    match d.len() {
        2 => Layout::Rank2,
        3 => Layout::Rank3,
        _ if d[1] == heads => Layout::Bhsd,
        _ => Layout::Bshd,
    }
}

/// `(batch, seq)` of a tensor in `layout`.
fn batch_seq(d: &[usize], layout: Layout) -> (usize, usize) {
    match layout {
        Layout::Rank2 => (1, d[0]),
        Layout::Rank3 | Layout::Bshd => (d[0], d[1]),
        Layout::Bhsd => (d[0], d[2]),
    }
}

/// Bring `x` to `[B, H, S, D]` with `h` heads of width `dh`.
fn to_bhsd(
    g: &mut Graph,
    x: NodeId,
    layout: Layout,
    b: usize,
    s: usize,
    h: usize,
    dh: usize,
) -> NodeId {
    let bshd = vec![b as i64, s as i64, h as i64, dh as i64];
    match layout {
        Layout::Bhsd => x,
        Layout::Bshd => g.transpose_(x, vec![0, 2, 1, 3]),
        Layout::Rank2 | Layout::Rank3 => {
            let x = g.reshape_(x, bshd);
            g.transpose_(x, vec![0, 2, 1, 3])
        }
    }
}

/// Repeat KV heads so `[B, kv, S, D]` becomes `[B, kv·group, S, D]`, head `h`
/// reading KV head `h / group`.
fn repeat_heads(
    g: &mut Graph,
    x: NodeId,
    b: usize,
    kv: usize,
    group: usize,
    s: usize,
    dh: usize,
) -> NodeId {
    if group == 1 {
        return x;
    }
    let x = g.reshape_(x, vec![b as i64, kv as i64, 1, s as i64, dh as i64]);
    let target = vec![b as i64, kv as i64, group as i64, s as i64, dh as i64];
    let shape = Shape::new(&[b, kv, group, s, dh], g.shape(x).dtype());
    let x = g.add_node(
        Op::Expand {
            target_shape: target,
        },
        vec![x],
        shape,
    );
    g.reshape_(x, vec![b as i64, (kv * group) as i64, s as i64, dh as i64])
}

fn expand(g: &mut Graph, x: NodeId, target: &[usize]) -> NodeId {
    if dims(g, x) == target {
        return x;
    }
    let shape = Shape::new(target, g.shape(x).dtype());
    g.add_node(
        Op::Expand {
            target_shape: target.iter().map(|&d| d as i64).collect(),
        },
        vec![x],
        shape,
    )
}

fn scalar(g: &mut Graph, v: f32, like: NodeId) -> NodeId {
    let c = g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F32),
    );
    let target = dims(g, like);
    let c = g.reshape_(c, vec![1; target.len()]);
    expand(g, c, &target)
}

/// `s · (1 − drop) + drop · MASKED`, `drop` broadcast to `s`'s shape.
fn replace_masked(g: &mut Graph, s: NodeId, drop: NodeId) -> NodeId {
    let target = dims(g, s);
    let drop = expand(g, drop, &target);
    let one = scalar(g, 1.0, s);
    let keep = g.sub(one, drop);
    let kept = g.mul(s, keep);
    let neg = scalar(g, MASKED, s);
    let masked = g.mul(drop, neg);
    g.add(kept, masked)
}

/// Constant `[1, 1, q_seq, k_seq]` 0/1 drop pattern for a positional mask.
fn positional_drop(g: &mut Graph, q_seq: usize, k_seq: usize, mask: MaskKind) -> Option<NodeId> {
    let window = match mask {
        MaskKind::Causal => None,
        MaskKind::SlidingWindow(w) => Some(w),
        _ => return None,
    };
    let offset = k_seq.saturating_sub(q_seq);
    let mut data = Vec::with_capacity(q_seq * k_seq * 4);
    for qi in 0..q_seq {
        let abs_q = offset + qi;
        let lo = window.map_or(0, |w| abs_q.saturating_sub(w));
        for ki in 0..k_seq {
            let dropped = ki > abs_q || ki < lo;
            data.extend_from_slice(&(if dropped { 1f32 } else { 0f32 }).to_le_bytes());
        }
    }
    Some(g.add_node(
        Op::Constant { data },
        vec![],
        Shape::new(&[1, 1, q_seq, k_seq], DType::F32),
    ))
}

fn lower_one(g: &mut Graph, inputs: &[NodeId], out_shape: &Shape, spec: Spec) -> NodeId {
    let (q, k, v) = (inputs[0], inputs[1], inputs[2]);
    let (qd, kd) = (dims(g, q), dims(g, k));
    let layout = layout_of(&qd, spec.heads);
    let (b, q_seq) = batch_seq(&qd, layout);
    let (_, k_seq) = batch_seq(&kd, layout_of(&kd, spec.heads));
    let k_numel: usize = kd.iter().product();
    let kv_heads = (k_numel / (b.max(1) * k_seq.max(1) * spec.head_dim.max(1))).max(1);
    let group = (spec.heads / kv_heads).max(1);
    let kv_layout = match layout {
        // A rank-4 K/V with fewer heads keeps its own axis order.
        Layout::Bhsd | Layout::Bshd => layout_of(&kd, kv_heads),
        other => other,
    };

    let q4 = to_bhsd(g, q, layout, b, q_seq, spec.heads, spec.head_dim);
    let k4 = to_bhsd(g, k, kv_layout, b, k_seq, kv_heads, spec.head_dim);
    let v4 = to_bhsd(g, v, kv_layout, b, k_seq, kv_heads, spec.v_head_dim);
    let k4 = repeat_heads(g, k4, b, kv_heads, group, k_seq, spec.head_dim);
    let v4 = repeat_heads(g, v4, b, kv_heads, group, k_seq, spec.v_head_dim);

    let kt = g.transpose_(k4, vec![0, 1, 3, 2]);
    let mut s = g.mm(q4, kt); // [B, H, Sq, Sk]
    let scale = scalar(g, spec.scale, s);
    s = g.mul(s, scale);

    match spec.mask {
        MaskKind::Custom if inputs.len() > 3 => {
            // [B, Sk] key mask → drop where < threshold, as [B, 1, 1, Sk].
            let m = g.reshape_(inputs[3], vec![b as i64, 1, 1, k_seq as i64]);
            let thr = scalar(g, KEEP_THRESHOLD, m);
            let lt = g.lt(m, thr);
            let drop = g.cast(lt, DType::F32);
            s = replace_masked(g, s, drop);
        }
        MaskKind::Bias if inputs.len() > 3 => {
            let bias = g.reshape_(
                inputs[3],
                vec![b as i64, spec.heads as i64, q_seq as i64, k_seq as i64],
            );
            s = g.add(s, bias);
        }
        _ => {}
    }
    if let Some(drop) = positional_drop(g, q_seq, k_seq, spec.mask) {
        s = replace_masked(g, s, drop);
    }
    if let Some(c) = spec.softcap {
        let inv = scalar(g, 1.0 / c, s);
        let t = g.mul(s, inv);
        let t = g.tanh(t);
        let cap = scalar(g, c, t);
        s = g.mul(t, cap);
    }
    let p = g.sm(s, -1);
    let o = g.mm(p, v4); // [B, H, Sq, Dv]

    let o = match layout {
        Layout::Bhsd => o,
        _ => g.transpose_(o, vec![0, 2, 1, 3]), // [B, Sq, H, Dv]
    };
    let target: Vec<i64> = out_shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static() as i64)
        .collect();
    g.reshape_(o, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run `graph` on the CPU backend with named f32 inputs; first output.
    fn run_cpu(graph: &Graph, feeds: &[(&str, &[f32])]) -> Vec<f32> {
        let plan = rlx_opt::memory::plan_memory(graph);
        let mut arena = rlx_cpu::arena::Arena::from_plan(plan);
        let sched = rlx_cpu::thunk::compile_thunks(graph, &arena);
        for node in graph.nodes() {
            match &node.op {
                Op::Constant { data } if arena.has_buffer(node.id) && !data.is_empty() => {
                    let buf = arena.slice_mut(node.id);
                    for (dst, c) in buf.iter_mut().zip(data.as_chunks::<4>().0.iter()) {
                        *dst = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    }
                }
                Op::Input { name, .. } => {
                    let (_, v) = feeds.iter().find(|(n, _)| n == name).expect("feed");
                    arena.slice_mut(node.id)[..v.len()].copy_from_slice(v);
                }
                _ => {}
            }
        }
        rlx_cpu::thunk::execute_thunks(&sched, arena.raw_buf_mut());
        let out = graph.outputs[0];
        let n = graph.node(out).shape.num_elements().expect("static output");
        arena.slice_mut(out)[..n].to_vec()
    }

    fn data(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 1000) as f32 / 500.0 - 1.0)
            .collect()
    }

    /// The fused op on the CPU kernel vs the lowered graph on the same CPU
    /// backend. `q_dims` / `kv_dims` pick the layout.
    fn check(
        q_dims: &[usize],
        kv_dims: &[usize],
        h: usize,
        d: usize,
        mask: MaskKind,
        key_mask: Option<Vec<f32>>,
    ) {
        let qn: usize = q_dims.iter().product();
        let kn: usize = kv_dims.iter().product();
        let (q, k, v) = (data(qn, 1), data(kn, 2), data(kn, 3));
        let mut g = Graph::new("attn");
        let qi = g.input("q", Shape::new(q_dims, DType::F32));
        let ki = g.input("k", Shape::new(kv_dims, DType::F32));
        let vi = g.input("v", Shape::new(kv_dims, DType::F32));
        let mut ins = vec![qi, ki, vi];
        if let Some(m) = &key_mask {
            ins.push(g.input("m", Shape::new(&[1, m.len()], DType::F32)));
        }
        let a = g.add_node(
            Op::Attention {
                num_heads: h,
                head_dim: d,
                v_head_dim: None,
                mask_kind: mask,
                score_scale: None,
                attn_logit_softcap: None,
            },
            ins,
            Shape::new(q_dims, DType::F32),
        );
        g.set_outputs(vec![a]);
        let mut feeds: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v)];
        if let Some(m) = &key_mask {
            feeds.push(("m", m));
        }
        let fused = run_cpu(&g, &feeds);
        let lowered = LowerAttention.run(g);
        assert!(
            !lowered
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Attention { .. })),
            "attention left in the graph"
        );
        let got = run_cpu(&lowered, &feeds);
        let worst = got
            .iter()
            .zip(&fused)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst < 1e-5,
            "{q_dims:?} {mask:?}: lowered vs fused max |Δ| {worst}"
        );
    }

    #[test]
    fn lowered_matches_the_cpu_kernel() {
        check(&[1, 4, 16], &[1, 4, 16], 2, 8, MaskKind::None, None);
        check(&[1, 4, 16], &[1, 4, 16], 2, 8, MaskKind::Causal, None);
        // decode step against a longer cache, GQA 4 query heads / 2 kv heads
        check(&[1, 1, 32], &[1, 6, 16], 4, 8, MaskKind::Causal, None);
        check(
            &[1, 5, 8],
            &[1, 5, 4],
            2,
            4,
            MaskKind::SlidingWindow(2),
            None,
        );
        check(
            &[1, 1, 16],
            &[1, 6, 16],
            2,
            8,
            MaskKind::Custom,
            Some(vec![1., 1., 0., 1., 0., 0.]),
        );
        // rank-4 [B, H, S, D]
        check(&[1, 2, 3, 8], &[1, 2, 3, 8], 2, 8, MaskKind::Causal, None);
    }
}
