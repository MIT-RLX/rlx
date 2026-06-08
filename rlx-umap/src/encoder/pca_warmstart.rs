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

//! MSE graph to pre-train the encoder toward a PCA target.

use rlx_autodiff::grad_with_loss;
use rlx_compile::legalize_broadcast;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};

use super::mlp::{ModelSpec, ParamSlot, build_forward_graph};

#[derive(Clone)]
pub struct PcaWarmstartGraph {
    pub backward: Graph,
    pub params: Vec<ParamSlot>,
    pub d_output: NodeId,
}

pub fn build_pca_warmstart_graph(spec: &ModelSpec) -> PcaWarmstartGraph {
    let (mut g, embed, mut params) = build_forward_graph(spec);
    let f = DType::F32;
    let target = g.input("pca_target", Shape::new(&[spec.n, spec.output_dim], f));
    let diff = g.sub(embed, target);
    let sq = g.mul(diff, diff);
    let loss = g.mean(sq, vec![0, 1], false);
    g.set_outputs(vec![loss]);
    let loss_node = loss;

    let (g, remap) = legalize_broadcast::run_with_remap(g);
    params = params
        .into_iter()
        .map(|mut p| {
            p.param = remap[&p.param];
            p
        })
        .collect();
    let _loss = remap[&loss_node];
    let wrt: Vec<NodeId> = params.iter().map(|p| p.param).collect();
    let bwd = grad_with_loss(&g, &wrt);
    let d_output = bwd
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, rlx_ir::Op::Input { name } if name == "d_output"))
        .map(|n| n.id)
        .expect("d_output");
    let grad_ids: Vec<NodeId> = bwd.outputs[1..=params.len()].to_vec();
    let params: Vec<ParamSlot> = params
        .into_iter()
        .zip(grad_ids)
        .map(|(mut p, grad)| {
            p.grad = Some(grad);
            p
        })
        .collect();

    PcaWarmstartGraph {
        backward: bwd,
        params,
        d_output,
    }
}
