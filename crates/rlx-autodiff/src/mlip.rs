// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! ML interatomic potential (MLIP) helpers — force + energy supervision via
//! embedded inner gradients.

use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{Graph, NodeId, Op, Shape};
use std::collections::HashMap;

use crate::autodiff::grad_with_loss;
use crate::compose::{find_input_by_name, merge_subgraph};
use crate::decompose_backward::{decompose_backward_ops, prepare_grad_graph_for_jvp};

/// Weights for force + energy MSE terms.
#[derive(Debug, Clone, Copy)]
pub struct ForceEnergyLossWeights {
    pub force: f64,
    pub energy: f64,
}

impl Default for ForceEnergyLossWeights {
    fn default() -> Self {
        Self {
            force: 1.0,
            energy: 1.0,
        }
    }
}

/// Decomposed, AD-ready gradient graph: `[energy, dE/d(positions…)]`.
pub fn grad_subgraph(forward: &Graph, wrt: &[NodeId]) -> Graph {
    let mut g = grad_with_loss(forward, wrt);
    g = decompose_backward_ops(g);
    crate::compose::internalize_d_output(&mut g);
    g
}

/// Prepared for an outer [`crate::autodiff_fwd::jvp`].
pub fn grad_subgraph_for_jvp(forward: &Graph, wrt: &[NodeId]) -> Graph {
    prepare_grad_graph_for_jvp(grad_with_loss(forward, wrt))
}

/// Build `w_f·MSE(−∇E, F_ref) + w_e·MSE(E, E_ref)`.
///
/// Adds fresh inputs `force_ref_name` and `energy_ref_name` on the returned graph.
pub fn build_force_energy_loss(
    energy_graph: &Graph,
    positions_name: &str,
    force_ref_name: &str,
    energy_ref_name: &str,
    weights: ForceEnergyLossWeights,
) -> Graph {
    let positions = find_input_by_name(energy_graph, positions_name)
        .unwrap_or_else(|| panic!("build_force_energy_loss: no input '{positions_name}'"));
    let grad_g = grad_subgraph(energy_graph, &[positions]);

    let mut loss_g = Graph::new(format!("{}_mlip_loss", energy_graph.name));
    let mut bind = HashMap::new();
    for node in energy_graph.nodes() {
        if let Op::Input { name } | Op::Param { name } = &node.op {
            bind.entry(name.clone())
                .or_insert_with(|| loss_g.add_node(node.op.clone(), vec![], node.shape.clone()));
        }
    }
    let sub_map = merge_subgraph(&mut loss_g, &grad_g, &bind);
    let energy = sub_map[&grad_g.outputs[0]];
    let grad_pos = sub_map[&grad_g.outputs[1]];
    let force_shape = loss_g.node(grad_pos).shape.clone();
    let dtype = force_shape.dtype();
    let scalar = Shape::scalar(dtype);
    let force_ref = loss_g.input(force_ref_name, force_shape.clone());
    let energy_ref = loss_g.input(energy_ref_name, scalar.clone());
    let zero = loss_g.add_node(
        crate::compose::constant_zero(&force_shape),
        vec![],
        force_shape.clone(),
    );
    let neg_grad = loss_g.binary(BinaryOp::Sub, zero, grad_pos, force_shape.clone());
    let f_diff = loss_g.binary(BinaryOp::Sub, neg_grad, force_ref, force_shape.clone());
    let f_sq = loss_g.binary(BinaryOp::Mul, f_diff, f_diff, force_shape.clone());
    let axes: Vec<usize> = (0..force_shape.rank()).collect();
    let f_mse = loss_g.reduce(f_sq, ReduceOp::Mean, axes, false, scalar.clone());
    let e_diff = loss_g.binary(BinaryOp::Sub, energy, energy_ref, scalar.clone());
    let e_sq = loss_g.binary(BinaryOp::Mul, e_diff, e_diff, scalar.clone());
    let wf = scalar_weight(&mut loss_g, weights.force, dtype);
    let we = scalar_weight(&mut loss_g, weights.energy, dtype);
    let wf_term = loss_g.binary(BinaryOp::Mul, wf, f_mse, scalar.clone());
    let we_term = loss_g.binary(BinaryOp::Mul, we, e_sq, scalar.clone());
    let loss = loss_g.binary(BinaryOp::Add, wf_term, we_term, scalar);
    loss_g.set_outputs(vec![loss]);
    loss_g
}

fn scalar_weight(g: &mut Graph, v: f64, dtype: rlx_ir::DType) -> NodeId {
    let bytes = match dtype {
        rlx_ir::DType::F64 => v.to_le_bytes().to_vec(),
        rlx_ir::DType::F32 => (v as f32).to_le_bytes().to_vec(),
        other => panic!("mlip weights: {other:?}"),
    };
    g.add_node(Op::Constant { data: bytes }, vec![], Shape::scalar(dtype))
}
