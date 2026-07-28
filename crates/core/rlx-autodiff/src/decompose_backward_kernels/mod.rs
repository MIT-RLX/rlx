// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//
// Primitive compositions for training `*Backward` ops (higher-order AD).

use rlx_ir::infer::GraphExt;
use rlx_ir::op::{CmpOp, MaskKind};
use rlx_ir::shape;
use rlx_ir::shape::Dim;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

mod attention;
mod conv_pool;
mod indexing;
mod loss;
mod norm;
mod quant;
mod rope;
mod scan;

pub use attention::*;
pub use conv_pool::*;
pub use indexing::*;
pub use loss::*;
pub use norm::*;
pub use quant::*;
pub use rope::*;
pub use scan::*;

/// Matches `RuntimeConfig::attn_mask_neg_inf` default on CPU.
const ATTN_MASK_NEG_INF: f32 = -1e9;

const MASK_BINARY_THRESHOLD: f32 = 0.5;

/// Max scan length for decomposed `ScanBackward` / `ScanBackwardXs` unrolls.
pub const SCAN_DECOMPOSE_MAX_LENGTH: u32 = 256;

use crate::activation_deriv::scalar_const;
use crate::autodiff::grad_with_loss;
use crate::compose::{broadcast_scalar, merge_subgraph};
use crate::prepare_ad::prepare_graph_for_ad;

fn axis_pos(axis: i32, rank: usize) -> usize {
    if axis < 0 {
        (rank as i32 + axis) as usize
    } else {
        axis as usize
    }
}

fn broadcast_eps(g: &mut Graph, eps: f32, like: &Shape) -> NodeId {
    let eps_s = scalar_const(eps as f64, &Shape::scalar(like.dtype()), g);
    broadcast_scalar(g, eps_s, like)
}

fn batch_reduce_axes(rank: usize, feature_axis: usize) -> Vec<usize> {
    (0..rank).filter(|&i| i != feature_axis).collect()
}

fn static_dim4(shape: &Shape) -> Option<[usize; 4]> {
    if shape.rank() != 4 {
        return None;
    }
    let mut out = [0usize; 4];
    for (i, d) in shape.dims().iter().enumerate() {
        out[i] = match d {
            Dim::Static(n) => *n,
            Dim::Dynamic(_) => return None,
        };
    }
    Some(out)
}

fn static_dim5(shape: &Shape) -> Option<[usize; 5]> {
    if shape.rank() != 5 {
        return None;
    }
    let mut out = [0usize; 5];
    for (i, d) in shape.dims().iter().enumerate() {
        out[i] = match d {
            Dim::Static(n) => *n,
            Dim::Dynamic(_) => return None,
        };
    }
    Some(out)
}

fn gather_flat_f32(g: &mut Graph, flat_x: NodeId, index: usize, dt: DType) -> NodeId {
    let idx = f32_tensor_const(vec![index as f32], Shape::new(&[1], dt), g);
    g.gather_(flat_x, idx, 0)
}

/// True when conv backward-input can decompose to forward `Op::Conv` with
/// dynamic batch (spatial + channel dims static; weights static).
pub fn conv_di_decompose_eligible(dy: &Shape, w: &Shape, dx: &Shape) -> bool {
    if dy.rank() != 4 || w.rank() != 4 || dx.rank() != 4 {
        return false;
    }
    if !w.is_static() {
        return false;
    }
    if dy.dim(0) != dx.dim(0) {
        return false;
    }
    (1..4).all(|axis| dy.dim(axis).is_static() && dx.dim(axis).is_static())
}

/// True when conv backward-weight can decompose via `Op::Im2Col` without
/// binding batch (spatial + channel dims static; batch may be dynamic).
pub fn conv_dw_im2col_eligible(x: &Shape, dy: &Shape, dw: &Shape) -> bool {
    if x.rank() != 4 || dy.rank() != 4 || dw.rank() != 4 {
        return false;
    }
    if !dw.is_static() {
        return false;
    }
    if x.dim(0) != dy.dim(0) {
        return false;
    }
    (1..4).all(|axis| x.dim(axis).is_static() && dy.dim(axis).is_static())
}

fn static_dim2(shape: &Shape) -> Option<[usize; 2]> {
    if shape.rank() != 2 {
        return None;
    }
    Some([shape.dim(0).unwrap_static(), shape.dim(1).unwrap_static()])
}

fn dim_product(shape: &Shape, start: usize, end: usize) -> usize {
    shape.dims()[start..end]
        .iter()
        .map(|d| d.unwrap_static())
        .product()
}

fn perm_move_axis_last(rank: usize, axis: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.remove(axis);
    perm.push(axis);
    perm
}

fn invert_perm(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

fn apply_perm_shape(shape: &Shape, perm: &[usize]) -> Shape {
    let dims: Vec<Dim> = perm.iter().map(|&i| shape.dim(i)).collect();
    Shape::from_dims(&dims, shape.dtype())
}

fn cumsum_backward_matrix(l: usize, exclusive: bool) -> Vec<f32> {
    let mut mat = vec![0f32; l * l];
    for j in 0..l {
        for k in 0..l {
            let inc = if exclusive { j > k } else { j >= k };
            if inc {
                mat[j * l + k] = 1.0;
            }
        }
    }
    mat
}

fn q_max_for_bits(bits: u8) -> f32 {
    match bits {
        8 => 127.0,
        4 => 7.0,
        2 => 1.0,
        n => panic!("compose_fake_quantize_backward: bad bits {n}"),
    }
}

fn scan_vjp_input_names(body_vjp: &Graph) -> (String, Vec<String>) {
    scan_primal_input_names(body_vjp, "d_output")
}

fn scan_primal_input_names(body: &Graph, skip: &str) -> (String, Vec<String>) {
    let mut names: Vec<(NodeId, String)> = body
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Input { name } if skip.is_empty() || name.as_str() != skip => {
                Some((n.id, name.clone()))
            }
            _ => None,
        })
        .collect();
    names.sort_by_key(|(id, _)| *id);
    let ordered: Vec<String> = names.into_iter().map(|(_, n)| n).collect();
    let carry = ordered.first().expect("scan body carry input").clone();
    let xs = ordered.into_iter().skip(1).collect();
    (carry, xs)
}

fn xs_step_shape(g: &Graph, xs_id: NodeId, dt: DType) -> Shape {
    let xs_shape = g.node(xs_id).shape.clone();
    let mut step_dims: Vec<usize> = xs_shape
        .dims()
        .iter()
        .skip(1)
        .map(|d| d.unwrap_static())
        .collect();
    if step_dims.is_empty() {
        step_dims.push(1);
    }
    Shape::new(&step_dims, dt)
}

fn checkpoint_t_for_k(k: usize, k_total: usize, n_steps: usize) -> usize {
    if k_total == n_steps {
        k
    } else {
        ((k + 1) * n_steps)
            .div_ceil(k_total)
            .saturating_sub(1)
            .min(n_steps - 1)
    }
}

fn carry_before_step(
    g: &mut Graph,
    init: NodeId,
    trajectory: NodeId,
    t: usize,
    step_shape: &Shape,
) -> NodeId {
    if t == 0 {
        init
    } else {
        narrow_step(g, trajectory, t - 1, step_shape)
    }
}

fn forward_step_carry(
    g: &mut Graph,
    forward_body: &Graph,
    carry_name: &str,
    xs_names: &[String],
    xs: &[NodeId],
    carry_in: NodeId,
    t: usize,
    dt: DType,
) -> NodeId {
    let mut bind = std::collections::HashMap::from([(carry_name.to_string(), carry_in)]);
    for (i, name) in xs_names.iter().enumerate() {
        let step_shape = xs_step_shape(g, xs[i], dt);
        bind.insert(name.clone(), narrow_step(g, xs[i], t, &step_shape));
    }
    let id_map = merge_subgraph(g, forward_body, &bind);
    id_map[&forward_body.outputs[0]]
}

fn run_scan_backward_steps(
    g: &mut Graph,
    init: NodeId,
    trajectory: NodeId,
    upstream: NodeId,
    xs: &[NodeId],
    body_vjp: &Graph,
    forward_body: Option<&Graph>,
    length: u32,
    save_trajectory: bool,
    num_checkpoints: u32,
    out_shape: &Shape,
    xs_out_idx: Option<usize>,
) -> (NodeId, Option<Vec<NodeId>>) {
    let l = length as usize;
    let k_total = if num_checkpoints == 0 || num_checkpoints == length {
        l
    } else {
        num_checkpoints as usize
    };
    let is_partial = num_checkpoints != 0 && num_checkpoints != length;
    if is_partial {
        assert!(
            forward_body.is_some(),
            "compose_scan_backward: forward_body required for partial checkpoints"
        );
    }

    let (carry_name, xs_names) = scan_vjp_input_names(body_vjp);
    assert_eq!(xs.len(), xs_names.len());
    let (fwd_carry_name, fwd_xs_names) = forward_body
        .map(|fb| scan_primal_input_names(fb, ""))
        .unwrap_or_default();

    let dt = out_shape.dtype();
    let zero_bytes = vec![0u8; out_shape.num_elements().expect("static carry") * dt.size_bytes()];
    let zero_carry = g.add_node(Op::Constant { data: zero_bytes }, vec![], out_shape.clone());

    let mut dcarry = zero_carry;
    let mut dx_steps = xs_out_idx.map(|_| vec![NodeId(0); l]);

    let mut process_t = |g: &mut Graph, t: usize, carry_t: NodeId| {
        let mut d_out = dcarry;
        if save_trajectory {
            let up_t = narrow_step(g, upstream, t, out_shape);
            d_out = g.add(d_out, up_t);
            reconcile_node_shape(g, d_out);
        }
        let mut bind = std::collections::HashMap::from([
            (carry_name.clone(), carry_t),
            ("d_output".to_string(), d_out),
        ]);
        for (i, name) in xs_names.iter().enumerate() {
            let step_shape = xs_step_shape(g, xs[i], dt);
            bind.insert(name.clone(), narrow_step(g, xs[i], t, &step_shape));
        }
        let id_map = merge_subgraph(g, body_vjp, &bind);
        dcarry = id_map[&body_vjp.outputs[0]];
        reconcile_node_shape(g, dcarry);
        if let (Some(idx), Some(steps)) = (xs_out_idx, dx_steps.as_mut()) {
            steps[t] = id_map[&body_vjp.outputs[idx]];
            reconcile_node_shape(g, steps[t]);
        }
    };

    if is_partial {
        let fb = forward_body.expect("forward_body");
        for seg_k in (0..k_total).rev() {
            let seg_end = checkpoint_t_for_k(seg_k, k_total, l);
            let seg_start = if seg_k == 0 {
                0
            } else {
                checkpoint_t_for_k(seg_k - 1, k_total, l) + 1
            };
            let anchor = if seg_k == 0 {
                init
            } else {
                narrow_step(g, trajectory, seg_k - 1, out_shape)
            };
            let mut carry_before = std::collections::HashMap::new();
            let mut carry = anchor;
            for t in seg_start..=seg_end {
                carry_before.insert(t, carry);
                if t < seg_end {
                    carry =
                        forward_step_carry(g, fb, &fwd_carry_name, &fwd_xs_names, xs, carry, t, dt);
                }
            }
            for t in (seg_start..=seg_end).rev() {
                process_t(g, t, carry_before[&t]);
            }
        }
    } else {
        for t in (0..l).rev() {
            let carry_t = carry_before_step(g, init, trajectory, t, out_shape);
            process_t(g, t, carry_t);
        }
    }
    (dcarry, dx_steps)
}

fn finalize_scan_backward_carry(g: &mut Graph, dcarry: NodeId, out_shape: &Shape) -> NodeId {
    reconcile_node_shape(g, dcarry);
    if g.node(dcarry).shape.dims() != out_shape.dims() {
        let dims: Vec<i64> = out_shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect();
        g.reshape_(dcarry, dims)
    } else {
        dcarry
    }
}

fn narrow_step(g: &mut Graph, x: NodeId, t: usize, step_shape: &Shape) -> NodeId {
    let narrowed = g.narrow_(x, 0, t, 1);
    let narrowed_shape = g.node(narrowed).shape.clone();
    if narrowed_shape.dims() == step_shape.dims() {
        narrowed
    } else {
        let dims: Vec<i64> = step_shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect();
        g.reshape_(narrowed, dims)
    }
}

fn reconcile_node_shape(g: &mut Graph, id: NodeId) {
    let node = g.node(id).clone();
    if let Some(inferred) = rlx_ir::infer_shape::infer_output_shape(g, &node) {
        if inferred.dims() != node.shape.dims() {
            g.node_mut(id).shape = inferred;
        }
    }
}

/// Max im2col rows × patch cols per matmul tile (raise for larger static convs).
// Im2col chunk-size guard for `compose_conv2d_backward_weight`. The original
// 16384 forces a chunked loop whenever `m * k` exceeds it, which on a
// modest-sized spatiotemporal CNN (e.g. `kernel (1, 25)`, batch 8, time 500)
// expands to ~50 chunks per backward op — each adding ~5 nodes — and slows
// graph compilation to multi-minute timescales. Bumped to 4 M (≈16 MiB per
// im2col tensor at f32) so a typical conv backward stays as a single
// im2col + matmul, dramatically reducing post-decompose graph size at the
// cost of larger transient buffers. Tunable per call site if the buffer
// blow-up ever becomes the bottleneck instead.
const IM2COL_MAX_MKL: usize = 4_194_304;

fn reshape_attn_rank4(
    g: &mut Graph,
    q: NodeId,
    k: NodeId,
    v: NodeId,
    dy: NodeId,
    num_heads: usize,
    head_dim: usize,
) -> (NodeId, NodeId, NodeId, NodeId, Shape) {
    let q_shape = g.node(q).shape.clone();
    if q_shape.rank() == 4 {
        return (q, k, v, dy, q_shape);
    }
    if q_shape.rank() == 3 {
        let b = q_shape.dim(0).unwrap_static();
        let s = q_shape.dim(1).unwrap_static();
        let e = q_shape.dim(2).unwrap_static();
        assert_eq!(
            e,
            num_heads * head_dim,
            "rank-3 attention: last dim must be num_heads * head_dim"
        );
        let r4 = Shape::new(&[b, num_heads, s, head_dim], q_shape.dtype());
        let q4 = g.reshape_(
            q,
            vec![b as i64, num_heads as i64, s as i64, head_dim as i64],
        );
        let k4 = g.reshape_(
            k,
            vec![b as i64, num_heads as i64, s as i64, head_dim as i64],
        );
        let v4 = g.reshape_(
            v,
            vec![b as i64, num_heads as i64, s as i64, head_dim as i64],
        );
        let dy4 = g.reshape_(
            dy,
            vec![b as i64, num_heads as i64, s as i64, head_dim as i64],
        );
        return (q4, k4, v4, dy4, r4);
    }
    panic!(
        "compose_attention_backward: rank-3/4 [B,H,S,D] or [B,S,H*D] only, got rank {}",
        q_shape.rank()
    );
}

fn build_im2col_rows(
    g: &mut Graph,
    flat_x: NodeId,
    zero: NodeId,
    n: usize,
    c_in: usize,
    h: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_d: usize,
    k: usize,
    m_start: usize,
    m_end: usize,
) -> NodeId {
    let mut rows: Vec<NodeId> = Vec::with_capacity(m_end - m_start);
    let mut flat = 0usize;
    'outer: for ni in 0..n {
        for ho in 0..h_out {
            for wo in 0..w_out {
                if flat >= m_end {
                    break 'outer;
                }
                if flat >= m_start {
                    let mut patch: Vec<NodeId> = Vec::with_capacity(k);
                    for ci in 0..c_in {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let hi = ho * sh + ki * dh;
                                let wi = wo * sw + kj * dw_d;
                                let val = if hi < ph || wi < pw || hi - ph >= h || wi - pw >= w_in {
                                    zero
                                } else {
                                    let idx = ((ni * c_in + ci) * h + (hi - ph)) * w_in + (wi - pw);
                                    gather_flat_f32(g, flat_x, idx, g.node(flat_x).shape.dtype())
                                };
                                patch.push(val);
                            }
                        }
                    }
                    rows.push(g.concat_(patch, 0));
                }
                flat += 1;
            }
        }
    }
    g.concat_(rows, 0)
}

#[allow(clippy::too_many_arguments)]
fn build_im2col3d_rows(
    g: &mut Graph,
    flat_x: NodeId,
    zero: NodeId,
    n: usize,
    c_in: usize,
    d: usize,
    h: usize,
    w_in: usize,
    d_out: usize,
    h_out: usize,
    w_out: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    sd: usize,
    sh: usize,
    sw: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    dd: usize,
    dh: usize,
    dw_d: usize,
    k: usize,
    m_start: usize,
    m_end: usize,
) -> NodeId {
    let mut rows: Vec<NodeId> = Vec::with_capacity(m_end - m_start);
    let mut flat = 0usize;
    'outer: for ni in 0..n {
        for do_ in 0..d_out {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    if flat >= m_end {
                        break 'outer;
                    }
                    if flat >= m_start {
                        let mut patch: Vec<NodeId> = Vec::with_capacity(k);
                        for ci in 0..c_in {
                            for kz in 0..kd {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let di = do_ * sd + kz * dd;
                                        let hi = ho * sh + ki * dh;
                                        let wi = wo * sw + kj * dw_d;
                                        let val = if di < pd
                                            || hi < ph
                                            || wi < pw
                                            || di - pd >= d
                                            || hi - ph >= h
                                            || wi - pw >= w_in
                                        {
                                            zero
                                        } else {
                                            let idx = (((ni * c_in + ci) * d + (di - pd)) * h
                                                + (hi - ph))
                                                * w_in
                                                + (wi - pw);
                                            gather_flat_f32(
                                                g,
                                                flat_x,
                                                idx,
                                                g.node(flat_x).shape.dtype(),
                                            )
                                        };
                                        patch.push(val);
                                    }
                                }
                            }
                        }
                        rows.push(g.concat_(patch, 0));
                    }
                    flat += 1;
                }
            }
        }
    }
    g.concat_(rows, 0)
}

fn compare_eq(g: &mut Graph, lhs: NodeId, rhs: NodeId) -> NodeId {
    let s = shape::compare_shape(&g.node(lhs).shape, &g.node(rhs).shape).expect("compare eq");
    g.add_node(Op::Compare(CmpOp::Eq), vec![lhs, rhs], s)
}

fn compare_ge(g: &mut Graph, lhs: NodeId, rhs: NodeId) -> NodeId {
    let s = shape::compare_shape(&g.node(lhs).shape, &g.node(rhs).shape).expect("compare ge");
    g.add_node(Op::Compare(CmpOp::Ge), vec![lhs, rhs], s)
}

fn compare_gt(g: &mut Graph, lhs: NodeId, rhs: NodeId) -> NodeId {
    let s = shape::compare_shape(&g.node(lhs).shape, &g.node(rhs).shape).expect("compare gt");
    g.add_node(Op::Compare(CmpOp::Gt), vec![lhs, rhs], s)
}

fn cast_f32(g: &mut Graph, x: NodeId) -> NodeId {
    let s = g.node(x).shape.clone().with_dtype(DType::F32);
    g.add_node(Op::Cast { to: DType::F32 }, vec![x], s)
}

fn argmax_window_flat(
    g: &mut Graph,
    flat_x: NodeId,
    _n: usize,
    c: usize,
    h: usize,
    w: usize,
    ni: usize,
    ci: usize,
    ho: usize,
    wo: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dt: DType,
) -> NodeId {
    let in_chan = (ni * c + ci) * h * w;
    let mut best_v: Option<NodeId> = None;
    let mut best_i: Option<NodeId> = None;
    for ki in 0..kh {
        for kj in 0..kw {
            let hi = ho * sh + ki;
            let wi = wo * sw + kj;
            if hi < ph || wi < pw {
                continue;
            }
            let hi = hi - ph;
            let wi = wi - pw;
            if hi >= h || wi >= w {
                continue;
            }
            let idx = in_chan + hi * w + wi;
            let val = gather_flat_f32(g, flat_x, idx, dt);
            let idx_n = f32_tensor_const(vec![idx as f32], Shape::new(&[1], dt), g);
            match (best_v, best_i) {
                (None, None) => {
                    best_v = Some(val);
                    best_i = Some(idx_n);
                }
                (Some(bv), Some(bi)) => {
                    let cond = compare_gt(g, val, bv);
                    best_v = Some(where_select(g, cond, val, bv));
                    best_i = Some(where_select(g, cond, idx_n, bi));
                }
                _ => unreachable!("maxpool argmax partial state"),
            }
        }
    }
    best_i.expect("maxpool window has no in-bounds positions")
}

/// Nearest-neighbour upsample of an NCHW tensor by integer factors (kh, kw):
/// each `[ho,wo]` element is repeated into a `kh×kw` block. Implemented with
/// reshape → expand (broadcast) → reshape, so it lowers on every backend.
fn nn_upsample_nchw(
    g: &mut Graph,
    t: NodeId,
    n: usize,
    c: usize,
    ho: usize,
    wo: usize,
    kh: usize,
    kw: usize,
    dt: rlx_ir::DType,
) -> NodeId {
    let r1 = g.reshape_(t, vec![n as i64, c as i64, ho as i64, 1, wo as i64, 1]);
    let ex = g.add_node(
        Op::Expand {
            target_shape: vec![
                n as i64, c as i64, ho as i64, kh as i64, wo as i64, kw as i64,
            ],
        },
        vec![r1],
        Shape::new(&[n, c, ho, kh, wo, kw], dt),
    );
    g.reshape_(
        ex,
        vec![n as i64, c as i64, (ho * kh) as i64, (wo * kw) as i64],
    )
}

/// Nearest-neighbour upsample of an NCDHW tensor by integer factors.
fn nn_upsample_ncdhw(
    g: &mut Graph,
    t: NodeId,
    n: usize,
    c: usize,
    d_out: usize,
    h_out: usize,
    w_out: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    dt: rlx_ir::DType,
) -> NodeId {
    let r1 = g.reshape_(
        t,
        vec![
            n as i64,
            c as i64,
            d_out as i64,
            1,
            h_out as i64,
            1,
            w_out as i64,
            1,
        ],
    );
    let ex = g.add_node(
        Op::Expand {
            target_shape: vec![
                n as i64,
                c as i64,
                d_out as i64,
                kd as i64,
                h_out as i64,
                kh as i64,
                w_out as i64,
                kw as i64,
            ],
        },
        vec![r1],
        Shape::new(&[n, c, d_out, kd, h_out, kh, w_out, kw], dt),
    );
    g.reshape_(
        ex,
        vec![
            n as i64,
            c as i64,
            (d_out * kd) as i64,
            (h_out * kh) as i64,
            (w_out * kw) as i64,
        ],
    )
}

fn f32_tensor_const(data: Vec<f32>, shape: Shape, g: &mut Graph) -> NodeId {
    let bytes: Vec<u8> = data.into_iter().flat_map(f32::to_le_bytes).collect();
    g.add_node(Op::Constant { data: bytes }, vec![], shape)
}

fn where_select(g: &mut Graph, cond: NodeId, on_true: NodeId, on_false: NodeId) -> NodeId {
    let s = shape::binary_shape(&g.node(on_true).shape, &g.node(on_false).shape).expect("where");
    g.add_node(Op::Where, vec![cond, on_true, on_false], s)
}

/// Additive 0 / `-inf` mask for [`MaskKind::Causal`] and [`MaskKind::SlidingWindow`],
/// matching `apply_synthetic_mask` in `rlx-cpu`.
fn synthetic_additive_mask_data(
    bh: usize,
    q_seq: usize,
    k_seq: usize,
    mask_kind: MaskKind,
) -> Vec<f32> {
    let mut buf = vec![0.0f32; bh * q_seq * k_seq];
    let q_offset = k_seq.saturating_sub(q_seq);
    match mask_kind {
        MaskKind::Causal => {
            for bh_i in 0..bh {
                for qi in 0..q_seq {
                    let abs_q = q_offset + qi;
                    for ki in (abs_q + 1)..k_seq {
                        buf[bh_i * q_seq * k_seq + qi * k_seq + ki] = ATTN_MASK_NEG_INF;
                    }
                }
            }
        }
        MaskKind::SlidingWindow(w) => {
            for bh_i in 0..bh {
                for qi in 0..q_seq {
                    let abs_q = q_offset + qi;
                    let lo = abs_q.saturating_sub(w);
                    for ki in 0..k_seq {
                        if ki < lo || ki > abs_q {
                            buf[bh_i * q_seq * k_seq + qi * k_seq + ki] = ATTN_MASK_NEG_INF;
                        }
                    }
                }
            }
        }
        MaskKind::None | MaskKind::Custom | MaskKind::Bias => {}
    }
    buf
}

fn broadcast_mask_to_scores(
    g: &mut Graph,
    mask: NodeId,
    mask_shape: &Shape,
    scores_shape: &Shape,
    b: usize,
    h: usize,
    bh: usize,
    q_seq: usize,
    k_seq: usize,
    mask_kind: MaskKind,
) -> NodeId {
    let dtype = scores_shape.dtype();
    match mask_kind {
        MaskKind::Bias => {
            let full = Shape::new(&[b, h, q_seq, k_seq], dtype);
            let node = if mask_shape == &full {
                mask
            } else {
                broadcast_scalar(g, mask, &full)
            };
            g.reshape(
                node,
                vec![bh as i64, q_seq as i64, k_seq as i64],
                scores_shape.clone(),
            )
        }
        MaskKind::Custom => match mask_shape.rank() {
            2 => {
                let mid = Shape::new(&[b, 1, 1, k_seq], dtype);
                let reshaped = g.reshape(mask, vec![b as i64, 1, 1, k_seq as i64], mid.clone());
                let full = Shape::new(&[b, h, q_seq, k_seq], dtype);
                let expanded = broadcast_scalar(g, reshaped, &full);
                g.reshape(
                    expanded,
                    vec![bh as i64, q_seq as i64, k_seq as i64],
                    scores_shape.clone(),
                )
            }
            4 => g.reshape(
                mask,
                vec![bh as i64, q_seq as i64, k_seq as i64],
                scores_shape.clone(),
            ),
            _ => broadcast_scalar(g, mask, scores_shape),
        },
        _ => panic!("broadcast_mask_to_scores: Custom/Bias only"),
    }
}

fn apply_attn_score_mask(
    g: &mut Graph,
    scaled: NodeId,
    scores_shape: &Shape,
    mask_kind: MaskKind,
    mask: Option<NodeId>,
    mask_shape: Option<&Shape>,
    bh: usize,
    b: usize,
    h: usize,
    q_seq: usize,
    k_seq: usize,
) -> NodeId {
    let dtype = scores_shape.dtype();
    let mut scores = scaled;

    if matches!(mask_kind, MaskKind::Custom) {
        let mask_id = mask.expect("Custom attention requires mask input");
        let mask_s = mask_shape.expect("Custom attention requires mask shape");
        let mask_b = broadcast_mask_to_scores(
            g,
            mask_id,
            mask_s,
            scores_shape,
            b,
            h,
            bh,
            q_seq,
            k_seq,
            mask_kind,
        );
        let thr = scalar_const(MASK_BINARY_THRESHOLD as f64, &Shape::scalar(dtype), g);
        let mask_b_shape = g.node(mask_b).shape.clone();
        let thr_b = broadcast_scalar(g, thr, &mask_b_shape);
        let valid = compare_ge(g, mask_b, thr_b);
        let neg = scalar_const(ATTN_MASK_NEG_INF as f64, &Shape::scalar(dtype), g);
        let neg_b = broadcast_scalar(g, neg, scores_shape);
        scores = where_select(g, valid, scores, neg_b);
    }

    if matches!(mask_kind, MaskKind::Bias) {
        let mask_id = mask.expect("Bias attention requires mask input");
        let mask_s = mask_shape.expect("Bias attention requires mask shape");
        let mask_b = broadcast_mask_to_scores(
            g,
            mask_id,
            mask_s,
            scores_shape,
            b,
            h,
            bh,
            q_seq,
            k_seq,
            mask_kind,
        );
        scores = g.add(scores, mask_b);
    }

    if matches!(mask_kind, MaskKind::Causal | MaskKind::SlidingWindow(_)) {
        let data = synthetic_additive_mask_data(bh, q_seq, k_seq, mask_kind);
        let additive = f32_tensor_const(data, scores_shape.clone(), g);
        scores = g.add(scores, additive);
    }

    scores
}

/// Rank-4 `[B, H, S, D]` SDPA via matmul + mask + softmax + matmul.
/// Used before reverse-mode so higher-order AD never sees `Op::Attention`.
pub fn expand_attention_forward_primitives(
    g: &mut Graph,
    q: NodeId,
    k: NodeId,
    v: NodeId,
    num_heads: usize,
    head_dim: usize,
    out_shape: &Shape,
    q_seq: usize,
    k_seq: usize,
    mask_kind: MaskKind,
    mask: Option<NodeId>,
    mask_shape: Option<&Shape>,
) -> NodeId {
    assert_eq!(
        out_shape.rank(),
        4,
        "expand_attention_forward_primitives: rank-4 [B,H,S,D] only"
    );
    let dtype = out_shape.dtype();
    let b = out_shape.dim(0).unwrap_static();
    let h = out_shape.dim(1).unwrap_static();
    let s = out_shape.dim(2).unwrap_static();
    let d = out_shape.dim(3).unwrap_static();
    assert_eq!(h, num_heads, "num_heads mismatch");
    assert_eq!(d, head_dim, "head_dim mismatch");
    let bh = b * h;

    let flat3 = Shape::new(&[bh, s, d], dtype);
    let q_flat = g.reshape(q, vec![bh as i64, s as i64, d as i64], flat3.clone());
    let k_flat = g.reshape(k, vec![bh as i64, s as i64, d as i64], flat3.clone());
    let v_flat = g.reshape(v, vec![bh as i64, s as i64, d as i64], flat3);

    let k_t_shape = Shape::new(&[bh, d, s], dtype);
    let k_t = g.add_node(
        Op::Transpose {
            perm: vec![0, 2, 1],
        },
        vec![k_flat],
        k_t_shape.clone(),
    );

    let scores_shape =
        shape::matmul_shape(&g.node(q_flat).shape, &k_t_shape).expect("attn scores shape");
    let scores = g.matmul(q_flat, k_t, scores_shape.clone());

    let scale = (head_dim as f32).sqrt().recip();
    let scale_n = scalar_const(scale as f64, &Shape::scalar(dtype), g);
    let scale_b = broadcast_scalar(g, scale_n, &scores_shape);
    let scaled = g.mul(scores, scale_b);
    let masked = apply_attn_score_mask(
        g,
        scaled,
        &scores_shape,
        mask_kind,
        mask,
        mask_shape,
        bh,
        b,
        h,
        q_seq,
        k_seq,
    );

    let weights = g.softmax(masked, -1, scores_shape.clone());

    let out_flat_shape =
        shape::matmul_shape(&g.node(weights).shape, &g.node(v_flat).shape).expect("attn out shape");
    let out_flat = g.matmul(weights, v_flat, out_flat_shape);

    g.reshape(
        out_flat,
        vec![b as i64, h as i64, s as i64, d as i64],
        out_shape.clone(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::DType;

    #[test]
    fn composed_ln_bwd_matches_kernel_smoke() {
        let rows = 2usize;
        let h = 4usize;
        let eps = 1e-5f32;
        let mut g = Graph::new("ln");
        let x = g.input("x", Shape::new(&[rows, h], DType::F32));
        let gamma = g.input("gamma", Shape::new(&[h], DType::F32));
        let dy = g.input("dy", Shape::new(&[rows, h], DType::F32));
        let dx_k = g.layer_norm_backward_input(x, gamma, dy, -1, eps);
        let dx_c = compose_layer_norm_backward_input(
            &mut g,
            x,
            gamma,
            dy,
            -1,
            eps,
            &Shape::new(&[rows, h], DType::F32),
        );
        assert_eq!(g.node(dx_k).shape, g.node(dx_c).shape);
    }
}
