// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Lower VAE-specific ops (`GroupNorm`, `ResizeNearest2x`) to primitives.

use crate::pass::Pass;
use rlx_ir::infer::GraphExt;
use rlx_ir::*;
use std::collections::HashMap;

fn scalar_const(g: &mut Graph, v: f32, dtype: DType) -> NodeId {
    let bytes = v.to_le_bytes().to_vec();
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[1], dtype),
    )
}

fn expand_to(g: &mut Graph, x: NodeId, target: &[i64]) -> NodeId {
    let dtype = g.shape(x).dtype();
    let shape = Shape::new(
        &target.iter().map(|&d| d as usize).collect::<Vec<_>>(),
        dtype,
    );
    g.add_node(
        Op::Expand {
            target_shape: target.to_vec(),
        },
        vec![x],
        shape,
    )
}

fn broadcast_scalar_like(g: &mut Graph, scalar: NodeId, like: NodeId) -> NodeId {
    let target: Vec<i64> = g
        .shape(like)
        .dims()
        .iter()
        .map(|d| d.unwrap_static() as i64)
        .collect();
    expand_to(g, scalar, &target)
}

/// `GroupNorm` on NCHW → reshape / reduce / elementwise ops.
pub struct LowerGroupNorm;

impl Pass for LowerGroupNorm {
    fn name(&self) -> &str {
        "lower_group_norm"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GroupNorm { .. }))
        {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = if let Op::GroupNorm { num_groups, eps } = &node.op {
                let x = id_map[&node.inputs[0]];
                let gamma = id_map[&node.inputs[1]];
                let beta = id_map[&node.inputs[2]];
                lower_group_norm(
                    &mut new_graph,
                    x,
                    gamma,
                    beta,
                    node.shape.clone(),
                    *num_groups,
                    *eps,
                )
            } else {
                let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                new_graph.add_node(node.op.clone(), inputs, node.shape.clone())
            };
            id_map.insert(node.id, new_id);
        }

        let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
        new_graph.set_outputs(new_outputs);
        new_graph
    }
}

fn lower_group_norm(
    g: &mut Graph,
    x: NodeId,
    gamma: NodeId,
    beta: NodeId,
    out_shape: Shape,
    num_groups: usize,
    eps: f32,
) -> NodeId {
    let dtype = out_shape.dtype();
    let dims: Vec<usize> = out_shape.dims().iter().map(|d| d.unwrap_static()).collect();
    let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let cpg = c / num_groups;

    let x5 = g.reshape_(
        x,
        vec![n as i64, num_groups as i64, cpg as i64, h as i64, w as i64],
    );
    // CUDA/MLX backends only support last-axis Reduce; fold (cpg, h, w) then mean once.
    let inner = (cpg * h * w) as i64;
    let x3 = g.reshape_(x5, vec![n as i64, num_groups as i64, inner]);
    let mean = g.mean(x3, vec![2], true);
    let x5_sq = g.mul(x5, x5);
    let x3_sq = g.reshape_(x5_sq, vec![n as i64, num_groups as i64, inner]);
    let sq_mean = g.mean(x3_sq, vec![2], true);
    let mean_sq = g.mul(mean, mean);
    let var = g.sub(sq_mean, mean_sq);
    let eps_c = scalar_const(g, eps, dtype);
    let eps_b = broadcast_scalar_like(g, eps_c, var);
    let var_eps = g.add(var, eps_b);
    let one = scalar_const(g, 1.0, dtype);
    let sqrt_var = g.sqrt(var_eps);
    let inv_std = g.div(one, sqrt_var);
    let mean_r = g.reshape_(mean, vec![n as i64, num_groups as i64, 1, 1, 1]);
    let mean5 = expand_to(
        g,
        mean_r,
        &[n as i64, num_groups as i64, cpg as i64, h as i64, w as i64],
    );
    let inv_std_r = g.reshape_(inv_std, vec![n as i64, num_groups as i64, 1, 1, 1]);
    let inv_std5 = expand_to(
        g,
        inv_std_r,
        &[n as i64, num_groups as i64, cpg as i64, h as i64, w as i64],
    );
    let x5_centered = g.sub(x5, mean5);
    let centered = g.mul(x5_centered, inv_std5);

    let gamma_r = g.reshape_(gamma, vec![1, num_groups as i64, cpg as i64, 1, 1]);
    let gamma5 = expand_to(
        g,
        gamma_r,
        &[n as i64, num_groups as i64, cpg as i64, h as i64, w as i64],
    );
    let beta_r = g.reshape_(beta, vec![1, num_groups as i64, cpg as i64, 1, 1]);
    let beta5 = expand_to(
        g,
        beta_r,
        &[n as i64, num_groups as i64, cpg as i64, h as i64, w as i64],
    );
    let scaled = g.mul(centered, gamma5);
    let normed5 = g.add(scaled, beta5);
    g.reshape_(normed5, vec![n as i64, c as i64, h as i64, w as i64])
}

/// Nearest 2× upsample on NCHW → concat along H and W.
pub struct LowerResizeNearest2x;

impl Pass for LowerResizeNearest2x {
    fn name(&self) -> &str {
        "lower_resize_nearest_2x"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::ResizeNearest2x))
        {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = if matches!(node.op, Op::ResizeNearest2x) {
                let x = id_map[&node.inputs[0]];
                let cat_h = new_graph.concat_(vec![x, x], 2);
                new_graph.concat_(vec![cat_h, cat_h], 3)
            } else {
                let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                new_graph.add_node(node.op.clone(), inputs, node.shape.clone())
            };
            id_map.insert(node.id, new_id);
        }

        let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
        new_graph.set_outputs(new_outputs);
        new_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn group_norm_lowers_to_primitives() {
        let f = DType::F32;
        let mut g = Graph::new("gn");
        let x = g.input("x", Shape::new(&[1, 4, 2, 2], f));
        let gamma = g.param("g", Shape::new(&[4], f));
        let beta = g.param("b", Shape::new(&[4], f));
        let out = g.add_node(
            Op::GroupNorm {
                num_groups: 2,
                eps: 1e-6,
            },
            vec![x, gamma, beta],
            Shape::new(&[1, 4, 2, 2], f),
        );
        g.set_outputs(vec![out]);

        let lowered = LowerGroupNorm.run(g);
        assert!(
            !lowered
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::GroupNorm { .. }))
        );
        assert!(
            lowered
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Reduce { .. }))
        );
    }

    #[test]
    fn resize_nearest_2x_lowers_to_concat() {
        let f = DType::F32;
        let mut g = Graph::new("up");
        let x = g.input("x", Shape::new(&[1, 2, 2, 2], f));
        let out = g.add_node(Op::ResizeNearest2x, vec![x], Shape::new(&[1, 2, 4, 4], f));
        g.set_outputs(vec![out]);

        let lowered = LowerResizeNearest2x.run(g);
        assert!(
            !lowered
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::ResizeNearest2x))
        );
        assert!(
            lowered
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Concat { .. }))
        );
    }
}
