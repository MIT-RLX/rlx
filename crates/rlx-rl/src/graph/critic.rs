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
// RLX — twin critic graphs.

use rlx_compile::legalize_broadcast;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use crate::graph::actor::WeightStore;
use crate::graph::mlp::{ParamSlot, concat_features, init_mat, init_vec, mlp_trunk};
use crate::spec::RlSpec;

#[derive(Debug, Clone)]
pub struct CriticGraphBundle {
    pub forward: Graph,
    pub train: CriticTrainGraph,
    pub q_grad: CriticQGradGraph,
}

#[derive(Debug, Clone)]
pub struct CriticTrainGraph {
    pub graph: Graph,
    pub d_output: NodeId,
    pub loss: NodeId,
    pub params: Vec<ParamSlot>,
}

#[derive(Debug, Clone)]
pub struct CriticQGradGraph {
    pub graph: Graph,
    pub d_output: NodeId,
    pub action: NodeId,
    pub grad_action: NodeId,
}

pub fn build_critic_graphs(spec: &RlSpec) -> CriticGraphBundle {
    CriticGraphBundle {
        forward: legalize_broadcast::run(build_forward(spec)),
        train: build_train(spec),
        q_grad: build_q_grad(spec),
    }
}

pub struct CompiledTwinCritic {
    pub forward: rlx_runtime::CompiledGraph,
    pub train: rlx_runtime::CompiledGraph,
    pub q_grad: rlx_runtime::CompiledGraph,
    pub train_meta: CriticTrainGraph,
    pub q_grad_meta: CriticQGradGraph,
}

impl CompiledTwinCritic {
    pub fn compile(session: &rlx_runtime::Session, bundle: &CriticGraphBundle) -> Self {
        Self {
            forward: session.compile(bundle.forward.clone()),
            train: session.compile(bundle.train.graph.clone()),
            q_grad: session.compile(bundle.q_grad.graph.clone()),
            train_meta: bundle.train.clone(),
            q_grad_meta: bundle.q_grad.clone(),
        }
    }

    pub fn set_weights(&mut self, weights: &WeightStore) {
        for g in [&mut self.forward, &mut self.train, &mut self.q_grad] {
            weights.apply(g);
        }
    }

    pub fn q_values(&mut self, state: &[f32], action: &[f32]) -> (f32, f32) {
        let outs = self.forward.run(&[("state", state), ("action", action)]);
        let q1 = outs.first().map(|v| v[0]).unwrap_or(0.0);
        let q2 = outs.get(1).map(|v| v[0]).unwrap_or(0.0);
        (q1, q2)
    }

    pub fn action_grad(&mut self, state: &[f32], action: &[f32]) -> Vec<f32> {
        let outs = self.q_grad.run(&[
            ("state", state),
            ("action", action),
            ("d_output", &[1.0f32]),
        ]);
        outs.into_iter().nth(1).unwrap_or_default()
    }
}

pub fn init_critic_weights(spec: &RlSpec, seed: u64) -> WeightStore {
    let mut w = WeightStore::default();
    let mut s = seed.wrapping_add(0x000C_171C);
    let mut in_d = spec.critic_in_dim();
    for (li, &hd) in spec.hidden.iter().enumerate() {
        init_mat(&mut w, &format!("critic_w{li}"), in_d, hd, &mut s);
        init_vec(&mut w, &format!("critic_b{li}"), hd, &mut s);
        in_d = hd;
    }
    init_mat(&mut w, "critic_q1_w", in_d, 1, &mut s);
    init_vec(&mut w, "critic_q1_b", 1, &mut s);
    init_mat(&mut w, "critic_q2_w", in_d, 1, &mut s);
    init_vec(&mut w, "critic_q2_b", 1, &mut s);
    w
}

fn build_forward(spec: &RlSpec) -> Graph {
    let f = DType::F32;
    let b = spec.batch;
    let sd = spec.state_dim;
    let ad = spec.action_dim;
    let mut g = Graph::new("critic_forward");
    let mut params = Vec::new();

    let state = g.input("state", Shape::new(&[b, sd], f));
    let action = g.input("action", Shape::new(&[b, ad], f));
    let feats = concat_features(&mut g, vec![state, action]);
    let (h, in_d) = mlp_trunk(
        &mut g,
        feats,
        spec.critic_in_dim(),
        &spec.hidden,
        "critic",
        &mut params,
    );
    let q1 = q_head(&mut g, h, in_d, "critic_q1", b, &mut params);
    let q2 = q_head(&mut g, h, in_d, "critic_q2", b, &mut params);
    let _ = params;
    g.set_outputs(vec![q1, q2]);
    g
}

fn q_head(
    g: &mut Graph,
    h: NodeId,
    in_d: usize,
    prefix: &str,
    batch: usize,
    params: &mut Vec<ParamSlot>,
) -> NodeId {
    let f = DType::F32;
    let w_name = format!("{prefix}_w");
    let b_name = format!("{prefix}_b");
    let w = g.param(&w_name, Shape::new(&[in_d, 1], f));
    let b = g.param(&b_name, Shape::new(&[1], f));
    params.push(ParamSlot {
        name: w_name,
        shape: vec![in_d, 1],
        param: w,
        grad: None,
    });
    params.push(ParamSlot {
        name: b_name,
        shape: vec![1],
        param: b,
        grad: None,
    });
    let q = g.mm(h, w);
    let q = g.add(q, b);
    g.reshape_(q, vec![batch as i64, 1])
}

fn build_train(spec: &RlSpec) -> CriticTrainGraph {
    let f = DType::F32;
    let b = spec.batch;
    let sd = spec.state_dim;
    let ad = spec.action_dim;
    let mut g = Graph::new("critic_train");
    let mut params = Vec::new();

    let state = g.input("state", Shape::new(&[b, sd], f));
    let action = g.input("action", Shape::new(&[b, ad], f));
    let target = g.input("target", Shape::new(&[b, 1], f));

    let feats = concat_features(&mut g, vec![state, action]);
    let (h, in_d) = mlp_trunk(
        &mut g,
        feats,
        spec.critic_in_dim(),
        &spec.hidden,
        "critic",
        &mut params,
    );
    let q1 = q_head(&mut g, h, in_d, "critic_q1", b, &mut params);
    let err = g.sub(q1, target);
    let sq = g.mul(err, err);
    let loss = g.mean(sq, vec![0, 1], false);
    g.set_outputs(vec![loss]);
    finalize_critic_train(g, params, loss)
}

fn build_q_grad(spec: &RlSpec) -> CriticQGradGraph {
    let f = DType::F32;
    let b = spec.batch;
    let sd = spec.state_dim;
    let ad = spec.action_dim;
    let mut g = Graph::new("critic_q_grad");

    let state = g.input("state", Shape::new(&[b, sd], f));
    let action = g.input("action", Shape::new(&[b, ad], f));
    let feats = concat_features(&mut g, vec![state, action]);
    let mut params = Vec::new();
    let (h, in_d) = mlp_trunk(
        &mut g,
        feats,
        spec.critic_in_dim(),
        &spec.hidden,
        "critic",
        &mut params,
    );
    let q1 = q_head(&mut g, h, in_d, "critic_q1", b, &mut params);
    let loss = g.sum(q1, vec![0, 1], false);
    g.set_outputs(vec![loss]);
    let _ = params;

    let (g, remap) = legalize_broadcast::run_with_remap(g);
    let action = remap[&action];
    let bwd = rlx_autodiff::grad_with_loss(&g, &[action]);
    let d_output = bwd
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
        .map(|n| n.id)
        .expect("d_output");
    let grad_action = bwd.outputs[1];

    CriticQGradGraph {
        graph: bwd,
        d_output,
        action,
        grad_action,
    }
}

fn finalize_critic_train(g: Graph, params: Vec<ParamSlot>, loss_fwd: NodeId) -> CriticTrainGraph {
    let (g, remap) = legalize_broadcast::run_with_remap(g);
    let params: Vec<ParamSlot> = params
        .into_iter()
        .map(|mut p| {
            p.param = remap[&p.param];
            p
        })
        .collect();
    let loss_fwd = remap[&loss_fwd];
    let wrt: Vec<NodeId> = params.iter().map(|p| p.param).collect();
    let bwd = rlx_autodiff::grad_with_loss(&g, &wrt);
    let d_output = bwd
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
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
    CriticTrainGraph {
        graph: bwd,
        d_output,
        loss: loss_fwd,
        params,
    }
}
