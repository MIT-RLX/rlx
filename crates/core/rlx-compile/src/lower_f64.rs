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
