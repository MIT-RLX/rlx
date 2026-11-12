// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Lower F64 to double-single** for backends without native f64.
//!
//! Every F64 tensor of shape `[…]` becomes an F32 tensor `[…, 2]` of
//! `(hi, lo)` pairs ([`rlx_ir::double_single`]). Arithmetic becomes
//! `Op::DoubleSingle`; data movement keeps its op with the pair axis carried
//! along; casts are spelled with existing ops (`f32 → df` pads a zero `lo`,
//! `df → f32` is `hi + lo`). F64 graph inputs, params and outputs keep their
//! names and change representation, which [`LoweredF64`] records so the
//! runtime can convert at the boundary.
//!
//! Ops with no df lowering yet are rejected rather than run wrong: the point
//! of the pass is that an F64 graph either computes in ~48-bit precision or
//! says it cannot.

use std::collections::HashMap;

use rlx_ir::double_single::DfOp;
use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

/// A graph after [`lower_f64`], plus what changed representation.
#[derive(Debug, Clone)]
pub struct LoweredF64 {
    pub graph: Graph,
    /// Inputs that were F64 and now take `[…, 2]` F32 pairs.
    pub df_inputs: Vec<String>,
    /// Params likewise.
    pub df_params: Vec<String>,
    /// Indices of graph outputs that are pair tensors standing for F64.
    pub df_outputs: Vec<usize>,
}

/// Does `g` compute anything in F64?
pub fn has_f64(g: &Graph) -> bool {
    g.nodes().iter().any(|n| n.shape.dtype() == DType::F64)
}

fn pair_shape(s: &Shape) -> Result<Shape, String> {
    let mut dims: Vec<usize> = Vec::with_capacity(s.rank() + 1);
    for d in s.dims() {
        if !d.is_static() {
            return Err("lower_f64: dynamic shapes are not supported yet".into());
        }
        dims.push(d.unwrap_static());
    }
    dims.push(2);
    Ok(Shape::new(&dims, DType::F32))
}

/// Rewrite `g`'s F64 computation into double-single form.
pub fn lower_f64(g: &Graph) -> Result<LoweredF64, String> {
    let mut out = Graph::new(&g.name);
    // old node → (new node, carries df pairs)
    let mut map: HashMap<NodeId, (NodeId, bool)> = HashMap::new();
    let (mut df_inputs, mut df_params) = (Vec::new(), Vec::new());
    for n in g.nodes() {
        let is_f64 = n.shape.dtype() == DType::F64;
        let ins: Vec<(NodeId, bool)> = n.inputs.iter().map(|i| map[i]).collect();
        let ids: Vec<NodeId> = ins.iter().map(|i| i.0).collect();
        let any_df = ins.iter().any(|i| i.1);
        let new = if !is_f64 && !any_df {
            (out.add_node(n.op.clone(), ids, n.shape.clone()), false)
        } else if !is_f64 {
            // A df value leaving F64: only a cast can do that.
            match &n.op {
                Op::Cast { to: DType::F32 } => (df_to_f32(&mut out, ids[0], &n.shape)?, false),
                // Indices out of a df tensor are plain f32.
                Op::ArgMax { axis, keep_dim } | Op::ArgMin { axis, keep_dim } if ins[0].1 => {
                    let is_max = matches!(n.op, Op::ArgMax { .. });
                    let src = &g.node(n.inputs[0]).shape;
                    let idx =
                        df_arg_reduce(&mut out, ids[0], src, *axis, *keep_dim, is_max, &n.shape)?;
                    (idx, false)
                }
                op => {
                    return Err(format!(
                        "lower_f64: {op} reads F64 into {:?}",
                        n.shape.dtype()
                    ));
                }
            }
        } else {
            // A narrower float operand of an F64 op (an f16 LM head, say) is
            // widened on entry: exact, and what native f64 kernels do.
            let (mut ins, mut ids) = (ins, ids);
            if !matches!(n.op, Op::Cast { .. } | Op::Input { .. } | Op::Param { .. })
                && ins.iter().any(|i| i.1)
            {
                for (j, &old) in n.inputs.iter().enumerate() {
                    let dt = g.node(old).shape.dtype();
                    let index = matches!(n.op, Op::Gather { .. }) && j == 1;
                    if ins[j].1 || index || !dt.is_float() {
                        continue;
                    }
                    let mut x = ids[j];
                    if dt != DType::F32 {
                        x = out.add_node(
                            Op::Cast { to: DType::F32 },
                            vec![x],
                            g.node(old).shape.clone().with_dtype(DType::F32),
                        );
                    }
                    let ps_in = pair_shape(&g.node(old).shape.clone().with_dtype(DType::F64))?;
                    ids[j] = f32_to_df(&mut out, x, &ps_in)?;
                    ins[j] = (ids[j], true);
                }
            }
            let any_df = ins.iter().any(|i| i.1);
            let ps = pair_shape(&n.shape)?;
            let id = match &n.op {
                Op::Input { name } => {
                    df_inputs.push(name.clone());
                    out.add_node(n.op.clone(), vec![], ps)
                }
                Op::Param { name } => {
                    df_params.push(name.clone());
                    out.add_node(n.op.clone(), vec![], ps)
                }
                Op::Constant { data } => {
                    let v: Vec<f64> = data
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|c| f64::from_le_bytes(*c))
                        .collect();
                    let pairs = rlx_ir::double_single::split_f64(&v);
                    let bytes = pairs.iter().flat_map(|x| x.to_le_bytes()).collect();
                    out.add_node(Op::Constant { data: bytes }, vec![], ps)
                }
                Op::Cast { .. } if !any_df => f32_to_df(&mut out, ids[0], &ps)?,
                Op::Cast { .. } => ids[0],
                Op::Binary(b) => {
                    let df = match b {
                        BinaryOp::Add => DfOp::Add,
                        BinaryOp::Sub => DfOp::Sub,
                        BinaryOp::Mul => DfOp::Mul,
                        BinaryOp::Div => DfOp::Div,
                        BinaryOp::Max => DfOp::Maximum,
                        other => return Err(format!("lower_f64: binary {other:?} on F64")),
                    };
                    require_df(&ins, &n.op)?;
                    out.add_node(Op::DoubleSingle(df), ids, ps)
                }
                Op::MatMul => {
                    require_df(&ins, &n.op)?;
                    out.add_node(Op::DoubleSingle(DfOp::MatMul), ids, ps)
                }
                Op::Reduce {
                    op: ReduceOp::Sum,
                    axes,
                    keep_dim,
                } => {
                    require_df(&ins, &n.op)?;
                    let df = DfOp::Sum {
                        axes: axes.clone(),
                        keep_dim: *keep_dim,
                    };
                    out.add_node(Op::DoubleSingle(df), ids, ps)
                }
                Op::Reduce {
                    op: ReduceOp::Max,
                    axes,
                    keep_dim,
                } => {
                    require_df(&ins, &n.op)?;
                    let df = DfOp::Max {
                        axes: axes.clone(),
                        keep_dim: *keep_dim,
                    };
                    out.add_node(Op::DoubleSingle(df), ids, ps)
                }
                Op::Activation(a @ (Activation::Exp | Activation::Sqrt)) => {
                    require_df(&ins, &n.op)?;
                    let df = match a {
                        Activation::Exp => DfOp::Exp,
                        _ => DfOp::Sqrt,
                    };
                    out.add_node(Op::DoubleSingle(df), ids, ps)
                }
                Op::Expand { target_shape } => {
                    let mut t = target_shape.clone();
                    t.push(2);
                    out.add_node(Op::Expand { target_shape: t }, ids, ps)
                }
                // Data movement: same op, pair axis appended / carried.
                Op::Reshape { new_shape } => {
                    let mut ns = new_shape.clone();
                    ns.push(2);
                    out.add_node(Op::Reshape { new_shape: ns }, ids, ps)
                }
                Op::Transpose { perm } => {
                    let mut p = perm.clone();
                    p.push(perm.len());
                    out.add_node(Op::Transpose { perm: p }, ids, ps)
                }
                Op::Narrow { .. } | Op::Concat { .. } | Op::KvAppend { .. } => {
                    require_df(&ins, &n.op)?;
                    out.add_node(n.op.clone(), ids, ps)
                }
                // Indices stay plain; only the table is df.
                Op::Gather { .. } if ins[0].1 => out.add_node(n.op.clone(), ids, ps),
                // Indices promoted to F64 (a whole graph under number format
                // f64): computed plain, then carried as df like any F64 value.
                Op::ArgMax { axis, keep_dim } | Op::ArgMin { axis, keep_dim } if ins[0].1 => {
                    let is_max = matches!(n.op, Op::ArgMax { .. });
                    let src = &g.node(n.inputs[0]).shape;
                    let plain = n.shape.clone().with_dtype(DType::F32);
                    let idx =
                        df_arg_reduce(&mut out, ids[0], src, *axis, *keep_dim, is_max, &plain)?;
                    f32_to_df(&mut out, idx, &ps)?
                }
                Op::ScatterAdd { axis: 0 } if ins[0].1 => {
                    let src = &g.node(n.inputs[0]).shape;
                    let idx_shape = &g.node(n.inputs[1]).shape;
                    // Indices widened to F64 are small integers: `hi + lo` is exact.
                    let (idx, idx_dt) = if ins[1].1 {
                        let plain = idx_shape.clone().with_dtype(DType::F32);
                        (df_to_f32(&mut out, ids[1], &plain)?, DType::F32)
                    } else {
                        (ids[1], idx_shape.dtype())
                    };
                    df_scatter_add(&mut out, ids[0], idx, idx_dt, src, &n.shape, &ps)?
                }
                op => return Err(format!("lower_f64: no double-single lowering for {op}")),
            };
            (id, true)
        };
        map.insert(n.id, new);
    }
    let mut df_outputs = Vec::new();
    let outs: Vec<NodeId> = g
        .outputs
        .iter()
        .enumerate()
        .map(|(i, o)| {
            let (id, df) = map[o];
            if df {
                df_outputs.push(i);
            }
            id
        })
        .collect();
    out.set_outputs(outs);
    Ok(LoweredF64 {
        graph: out,
        df_inputs,
        df_params,
        df_outputs,
    })
}

fn require_df(ins: &[(NodeId, bool)], op: &Op) -> Result<(), String> {
    match ins.iter().all(|i| i.1) {
        true => Ok(()),
        false => Err(format!("lower_f64: {op} mixes F64 with another dtype")),
    }
}

fn static_dims(s: &Shape) -> Vec<usize> {
    s.dims().iter().map(|d| d.unwrap_static()).collect()
}

fn f32_const(g: &mut Graph, v: f32) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F32),
    )
}

/// The hi or lo half (`which` = 0 / 1) of a df tensor, without the pair axis.
fn df_half(g: &mut Graph, x: NodeId, dims: &[usize], which: usize) -> NodeId {
    let mut col = dims.to_vec();
    col.push(1);
    let part = g.add_node(
        Op::Narrow {
            axis: dims.len(),
            start: which,
            len: 1,
        },
        vec![x],
        Shape::new(&col, DType::F32),
    );
    g.add_node(
        Op::Reshape {
            new_shape: dims.iter().map(|&d| d as i64).collect(),
        },
        vec![part],
        Shape::new(dims, DType::F32),
    )
}

/// ArgMin / ArgMax of a df tensor. A normalized pair orders like `(hi, lo)`
/// lexicographically, so the winner has the extreme `hi` and, among the
/// positions sharing it, the extreme `lo`: reduce `hi`, then take the arg of
/// `lo` with every other position pushed out of reach. Ties resolve to the
/// first index, as the f32 op does.
fn df_arg_reduce(
    g: &mut Graph,
    x: NodeId,
    src: &Shape,
    axis: usize,
    keep_dim: bool,
    is_max: bool,
    out: &Shape,
) -> Result<NodeId, String> {
    let dims = static_dims(src);
    let shape = Shape::new(&dims, DType::F32);
    let hi = df_half(g, x, &dims, 0);
    let lo = df_half(g, x, &dims, 1);
    let mut red = dims.clone();
    red[axis] = 1;
    let extreme = g.add_node(
        Op::Reduce {
            op: if is_max { ReduceOp::Max } else { ReduceOp::Min },
            axes: vec![axis],
            keep_dim: true,
        },
        vec![hi],
        Shape::new(&red, DType::F32),
    );
    let extreme = g.add_node(
        Op::Expand {
            target_shape: dims.iter().map(|&d| d as i64).collect(),
        },
        vec![extreme],
        shape.clone(),
    );
    let tie = g.add_node(
        Op::Compare(rlx_ir::op::CmpOp::Eq),
        vec![hi, extreme],
        Shape::new(&dims, DType::Bool),
    );
    let tie = g.add_node(Op::Cast { to: DType::F32 }, vec![tie], shape.clone());
    // lo ± (1 − tie)·1e30: |lo| ≤ ulp(hi)/2, far below the penalty.
    let one = f32_const(g, 1.0);
    let off = g.add_node(Op::Binary(BinaryOp::Sub), vec![one, tie], shape.clone());
    let big = f32_const(g, 1e30);
    let penalty = g.add_node(Op::Binary(BinaryOp::Mul), vec![off, big], shape.clone());
    let key = g.add_node(
        Op::Binary(if is_max { BinaryOp::Sub } else { BinaryOp::Add }),
        vec![lo, penalty],
        shape,
    );
    let op = if is_max {
        Op::ArgMax { axis, keep_dim }
    } else {
        Op::ArgMin { axis, keep_dim }
    };
    Ok(g.add_node(op, vec![key], out.clone().with_dtype(DType::F32)))
}

/// Axis-0 ScatterAdd of df updates `[n, ..]` into `[k, ..]`, as the df matmul
/// `onehotᵀ[k, n] · updates[n, ..]`: exact one-hot weights keep the sums in
/// double-single, where scattering `hi` and `lo` separately would round each
/// partial sum to f32.
fn df_scatter_add(
    g: &mut Graph,
    upd: NodeId,
    idx: NodeId,
    idx_dt: DType,
    src: &Shape,
    out: &Shape,
    pairs: &Shape,
) -> Result<NodeId, String> {
    let src_dims = static_dims(src);
    let out_dims = static_dims(out);
    let (n, k) = (src_dims[0], out_dims[0]);
    let rest: usize = src_dims[1..].iter().product();
    let idx = if idx_dt == DType::F32 {
        idx
    } else {
        g.add_node(
            Op::Cast { to: DType::F32 },
            vec![idx],
            Shape::new(&[n], DType::F32),
        )
    };
    let row = g.add_node(
        Op::Reshape {
            new_shape: vec![1, n as i64],
        },
        vec![idx],
        Shape::new(&[1, n], DType::F32),
    );
    let kn = Shape::new(&[k, n], DType::F32);
    let row = g.add_node(
        Op::Expand {
            target_shape: vec![k as i64, n as i64],
        },
        vec![row],
        kn.clone(),
    );
    let iota: Vec<u8> = (0..k).flat_map(|j| (j as f32).to_le_bytes()).collect();
    let col = g.add_node(
        Op::Constant { data: iota },
        vec![],
        Shape::new(&[k, 1], DType::F32),
    );
    let col = g.add_node(
        Op::Expand {
            target_shape: vec![k as i64, n as i64],
        },
        vec![col],
        kn.clone(),
    );
    let hit = g.add_node(
        Op::Compare(rlx_ir::op::CmpOp::Eq),
        vec![row, col],
        Shape::new(&[k, n], DType::Bool),
    );
    let onehot = g.add_node(Op::Cast { to: DType::F32 }, vec![hit], kn);
    let onehot = f32_to_df(g, onehot, &Shape::new(&[k, n, 2], DType::F32))?;
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![n as i64, rest as i64, 2],
        },
        vec![upd],
        Shape::new(&[n, rest, 2], DType::F32),
    );
    let sums = g.add_node(
        Op::DoubleSingle(DfOp::MatMul),
        vec![onehot, flat],
        Shape::new(&[k, rest, 2], DType::F32),
    );
    let pair_dims: Vec<i64> = static_dims(pairs).iter().map(|&d| d as i64).collect();
    Ok(g.add_node(
        Op::Reshape {
            new_shape: pair_dims,
        },
        vec![sums],
        pairs.clone(),
    ))
}

/// `x` (F32) as df: `(x, 0)`.
fn f32_to_df(g: &mut Graph, x: NodeId, pairs: &Shape) -> Result<NodeId, String> {
    let dims: Vec<usize> = pairs.dims().iter().map(|d| d.unwrap_static()).collect();
    let mut col = dims.clone();
    *col.last_mut().unwrap() = 1;
    let col_shape = Shape::new(&col, DType::F32);
    let hi = g.add_node(
        Op::Reshape {
            new_shape: col.iter().map(|&d| d as i64).collect(),
        },
        vec![x],
        col_shape.clone(),
    );
    let n: usize = col.iter().product();
    let zeros = g.add_node(
        Op::Constant {
            data: vec![0u8; n * 4],
        },
        vec![],
        col_shape,
    );
    Ok(g.add_node(
        Op::Concat {
            axis: dims.len() - 1,
        },
        vec![hi, zeros],
        pairs.clone(),
    ))
}

/// df `x` as F32: `hi + lo`, rounded once.
fn df_to_f32(g: &mut Graph, x: NodeId, out: &Shape) -> Result<NodeId, String> {
    let dims: Vec<usize> = out.dims().iter().map(|d| d.unwrap_static()).collect();
    let mut col = dims.clone();
    col.push(1);
    let col_shape = Shape::new(&col, DType::F32);
    let axis = dims.len();
    let hi = g.add_node(
        Op::Narrow {
            axis,
            start: 0,
            len: 1,
        },
        vec![x],
        col_shape.clone(),
    );
    let lo = g.add_node(
        Op::Narrow {
            axis,
            start: 1,
            len: 1,
        },
        vec![x],
        col_shape.clone(),
    );
    let sum = g.add_node(Op::Binary(BinaryOp::Add), vec![hi, lo], col_shape);
    Ok(g.add_node(
        Op::Reshape {
            new_shape: dims.iter().map(|&d| d as i64).collect(),
        },
        vec![sum],
        out.clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_graph_becomes_pairs() {
        let f = DType::F64;
        let mut g = Graph::new("t");
        let a = g.input("a", Shape::new(&[3, 4], f));
        let b = g.input("b", Shape::new(&[4, 5], f));
        let mm = g.matmul(a, b, Shape::new(&[3, 5], f));
        let s = g.reduce(mm, ReduceOp::Sum, vec![1], false, Shape::new(&[3], f));
        let c = g.add_node(
            Op::Cast { to: DType::F32 },
            vec![s],
            Shape::new(&[3], DType::F32),
        );
        g.set_outputs(vec![mm, c]);
        let l = lower_f64(&g).unwrap();
        assert!(!has_f64(&l.graph));
        assert_eq!(l.df_inputs, ["a", "b"]);
        assert_eq!(l.df_outputs, [0]);
        let o = l.graph.outputs[0];
        assert_eq!(l.graph.node(o).shape, Shape::new(&[3, 5, 2], DType::F32));
        assert_eq!(
            l.graph.node(l.graph.outputs[1]).shape,
            Shape::new(&[3], DType::F32)
        );
    }

    /// ArgMin over F64 and an axis-0 ScatterAdd of F64 rows lower to f32
    /// primitives and the df matmul: no F64 left, indices plain, sums pairs.
    #[test]
    fn argmin_and_scatter_add_lower_to_pairs() {
        let f = DType::F64;
        let mut g = Graph::new("t");
        let x = g.input("x", Shape::new(&[4, 3], f));
        let idx = g.input("idx", Shape::new(&[4], DType::F32));
        let am = g.add_node(
            Op::ArgMin {
                axis: 1,
                keep_dim: false,
            },
            vec![x],
            Shape::new(&[4], DType::F32),
        );
        let sa = g.add_node(
            Op::ScatterAdd { axis: 0 },
            vec![x, idx],
            Shape::new(&[2, 3], f),
        );
        g.set_outputs(vec![am, sa]);
        let l = lower_f64(&g).unwrap();
        assert!(!has_f64(&l.graph));
        assert_eq!(l.df_outputs, [1]);
        let o = &l.graph.outputs;
        assert_eq!(l.graph.node(o[0]).shape, Shape::new(&[4], DType::F32));
        assert_eq!(l.graph.node(o[1]).shape, Shape::new(&[2, 3, 2], DType::F32));
        assert!(
            l.graph
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::DoubleSingle(DfOp::MatMul)))
        );
    }

    #[test]
    fn unsupported_f64_op_is_an_error() {
        let f = DType::F64;
        let mut g = Graph::new("t");
        let a = g.input("a", Shape::new(&[3], f));
        let e = g.add_node(
            // Exact (erf) GELU has no double-single form.
            Op::Activation(rlx_ir::op::Activation::Gelu),
            vec![a],
            Shape::new(&[3], f),
        );
        g.set_outputs(vec![e]);
        assert!(
            lower_f64(&g)
                .unwrap_err()
                .contains("no double-single lowering")
        );
    }
}
