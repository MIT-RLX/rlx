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
// RLX — flow-map actor graphs.

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_compile::legalize_broadcast;

use crate::graph::mlp::{
    concat_features, flow_map_jump, init_mat, init_vec, mlp_layers, mse_mean, ParamSlot,
};
use crate::spec::RlSpec;

/// Named parameter tensors shared across compiled graphs.
#[derive(Debug, Clone, Default)]
pub struct WeightStore(pub std::collections::HashMap<String, Vec<f32>>);

impl WeightStore {
    pub fn apply(&self, exec: &mut rlx_runtime::CompiledGraph) {
        for (name, data) in &self.0 {
            exec.set_param(name, data);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActorGraphBundle {
    pub forward: Graph,
    /// Average velocity \(u_{r,t}\) only (FMQ anchor / online targets).
    pub velocity: Graph,
    pub offline_train: ActorTrainGraph,
    pub online_train: ActorTrainGraph,
}

#[derive(Debug, Clone)]
pub struct ActorTrainGraph {
    pub graph: Graph,
    pub d_output: NodeId,
    pub loss: NodeId,
    pub params: Vec<ParamSlot>,
}

pub fn build_actor_graphs(spec: &RlSpec) -> ActorGraphBundle {
    ActorGraphBundle {
        forward: legalize_broadcast::run(build_forward(spec)),
        velocity: legalize_broadcast::run(build_velocity(spec)),
        offline_train: build_offline_train(spec),
        online_train: build_online_train(spec),
    }
}

pub struct CompiledFlowMapAgent {
    pub forward: rlx_runtime::CompiledGraph,
    pub velocity: rlx_runtime::CompiledGraph,
    pub offline: rlx_runtime::CompiledGraph,
    pub online: rlx_runtime::CompiledGraph,
    pub offline_meta: ActorTrainGraph,
    pub online_meta: ActorTrainGraph,
}

impl CompiledFlowMapAgent {
    pub fn compile(session: &rlx_runtime::Session, bundle: &ActorGraphBundle) -> Self {
        Self {
            forward: session.compile(bundle.forward.clone()),
            velocity: session.compile(bundle.velocity.clone()),
            offline: session.compile(bundle.offline_train.graph.clone()),
            online: session.compile(bundle.online_train.graph.clone()),
            offline_meta: bundle.offline_train.clone(),
            online_meta: bundle.online_train.clone(),
        }
    }

    pub fn set_weights(&mut self, weights: &WeightStore) {
        for g in [
            &mut self.forward,
            &mut self.velocity,
            &mut self.offline,
            &mut self.online,
        ] {
            weights.apply(g);
        }
    }

    /// Average velocity \(u_{r,t}(a_r|s)\) (Python `actor_bc_flow` output).
    pub fn velocity(&mut self, state: &[f32], a_r: &[f32], r: f32, t: f32) -> Vec<f32> {
        self.velocity
            .run(&[
                ("state", state),
                ("a_r", a_r),
                ("r", &[r]),
                ("t", &[t]),
            ])
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    pub fn one_step(&mut self, state: &[f32], a0: &[f32]) -> Vec<f32> {
        let r = [0.0f32];
        let t = [1.0f32];
        self.forward
            .run(&[("state", state), ("a_r", a0), ("r", &r), ("t", &t)])
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    pub fn jump(&mut self, state: &[f32], a_r: &[f32], r: f32, t: f32) -> Vec<f32> {
        let rv = [r];
        let tv = [t];
        self.forward
            .run(&[("state", state), ("a_r", a_r), ("r", &rv), ("t", &tv)])
            .into_iter()
            .next()
            .unwrap_or_default()
    }
}

pub fn init_actor_weights(spec: &RlSpec, seed: u64) -> WeightStore {
    let mut w = WeightStore::default();
    let mut s = seed;
    let mut in_d = spec.actor_in_dim();
    for (li, &hd) in spec.hidden.iter().enumerate() {
        init_mat(&mut w, &format!("actor_w{li}"), in_d, hd, &mut s);
        init_vec(&mut w, &format!("actor_b{li}"), hd, &mut s);
        in_d = hd;
    }
    init_mat(&mut w, "actor_w_out", in_d, spec.action_dim, &mut s);
    init_vec(&mut w, "actor_b_out", spec.action_dim, &mut s);
    w
}

fn build_forward(spec: &RlSpec) -> Graph {
    let f = DType::F32;
    let b = spec.batch;
    let sd = spec.state_dim;
    let ad = spec.action_dim;
    let mut g = Graph::new("flow_map_forward");
    let mut params = Vec::new();

    let state = g.input("state", Shape::new(&[b, sd], f));
    let a_r = g.input("a_r", Shape::new(&[b, ad], f));
    let r = g.input("r", Shape::new(&[b, 1], f));
    let t = g.input("t", Shape::new(&[b, 1], f));

    let feats = concat_features(&mut g, vec![state, a_r, r, t]);
    let u = mlp_layers(
        &mut g,
        feats,
        spec.actor_in_dim(),
        &spec.hidden,
        ad,
        "actor",
        &mut params,
    );
    let a_t = flow_map_jump(&mut g, a_r, u, r, t, b);
    let _ = params;
    g.set_outputs(vec![a_t]);
    g
}

fn build_velocity(spec: &RlSpec) -> Graph {
    let f = DType::F32;
    let b = spec.batch;
    let sd = spec.state_dim;
    let ad = spec.action_dim;
    let mut g = Graph::new("flow_map_velocity");
    let mut params = Vec::new();

    let state = g.input("state", Shape::new(&[b, sd], f));
    let a_r = g.input("a_r", Shape::new(&[b, ad], f));
    let r = g.input("r", Shape::new(&[b, 1], f));
    let t = g.input("t", Shape::new(&[b, 1], f));

    let feats = concat_features(&mut g, vec![state, a_r, r, t]);
    let u = mlp_layers(
        &mut g,
        feats,
        spec.actor_in_dim(),
        &spec.hidden,
        ad,
        "actor",
        &mut params,
    );
    let _ = params;
    g.set_outputs(vec![u]);
    g
}

fn build_offline_train(spec: &RlSpec) -> ActorTrainGraph {
    let f = DType::F32;
    let b = spec.batch;
    let sd = spec.state_dim;
    let ad = spec.action_dim;
    let mut g = Graph::new("flow_map_offline");
    let mut params = Vec::new();

    let state = g.input("state", Shape::new(&[b, sd], f));
    let a_r = g.input("a_r", Shape::new(&[b, ad], f));
    let r = g.input("r", Shape::new(&[b, 1], f));
    let t = g.input("t", Shape::new(&[b, 1], f));
    let target_u = g.input("target_u", Shape::new(&[b, ad], f));

    let feats = concat_features(&mut g, vec![state, a_r, r, t]);
    let u = mlp_layers(
        &mut g,
        feats,
        spec.actor_in_dim(),
        &spec.hidden,
        ad,
        "actor",
        &mut params,
    );
    let loss = mse_mean(&mut g, u, target_u);
    g.set_outputs(vec![loss]);
    finalize_train(g, params, loss)
}

fn build_online_train(spec: &RlSpec) -> ActorTrainGraph {
    let f = DType::F32;
    let b = spec.batch;
    let sd = spec.state_dim;
    let ad = spec.action_dim;
    let mut g = Graph::new("flow_map_online");
    let mut params = Vec::new();

    let state = g.input("state", Shape::new(&[b, sd], f));
    let a_r = g.input("a_r", Shape::new(&[b, ad], f));
    let target_u = g.input("target_u", Shape::new(&[b, ad], f));
    let r = g.input("r", Shape::new(&[b, 1], f));
    let t = g.input("t", Shape::new(&[b, 1], f));

    let feats = concat_features(&mut g, vec![state, a_r, r, t]);
    let u = mlp_layers(
        &mut g,
        feats,
        spec.actor_in_dim(),
        &spec.hidden,
        ad,
        "actor",
        &mut params,
    );
    let loss = mse_mean(&mut g, u, target_u);
    g.set_outputs(vec![loss]);
    finalize_train(g, params, loss)
}

fn finalize_train(g: Graph, params: Vec<ParamSlot>, loss_fwd: NodeId) -> ActorTrainGraph {
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
    ActorTrainGraph {
        graph: bwd,
        d_output,
        loss: loss_fwd,
        params,
    }
}
