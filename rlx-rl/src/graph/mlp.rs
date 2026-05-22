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
// RLX — shared MLP helpers.

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};

/// Trainable parameter (forward + backward grad node).
#[derive(Debug, Clone)]
pub struct ParamSlot {
    pub name: String,
    pub shape: Vec<usize>,
    pub param: NodeId,
    pub grad: Option<NodeId>,
}

impl ParamSlot {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Hidden MLP trunk (no output head). Returns `(hidden, out_dim)`.
pub fn mlp_trunk(
    g: &mut Graph,
    x: NodeId,
    in_dim: usize,
    hidden: &[usize],
    prefix: &str,
    params: &mut Vec<ParamSlot>,
) -> (NodeId, usize) {
    let f = DType::F32;
    let mut h = x;
    let mut in_d = in_dim;

    for (li, &hd) in hidden.iter().enumerate() {
        let w_name = format!("{prefix}_w{li}");
        let b_name = format!("{prefix}_b{li}");
        let w = g.param(&w_name, Shape::new(&[in_d, hd], f));
        let b = g.param(&b_name, Shape::new(&[hd], f));
        params.push(ParamSlot {
            name: w_name,
            shape: vec![in_d, hd],
            param: w,
            grad: None,
        });
        params.push(ParamSlot {
            name: b_name,
            shape: vec![hd],
            param: b,
            grad: None,
        });
        h = g.mm(h, w);
        h = g.add(h, b);
        h = g.tanh(h);
        in_d = hd;
    }
    (h, in_d)
}

/// Full MLP including linear head to `out_dim`.
pub fn mlp_layers(
    g: &mut Graph,
    x: NodeId,
    in_dim: usize,
    hidden: &[usize],
    out_dim: usize,
    prefix: &str,
    params: &mut Vec<ParamSlot>,
) -> NodeId {
    let f = DType::F32;
    let (h, in_d) = mlp_trunk(g, x, in_dim, hidden, prefix, params);
    let w_name = format!("{prefix}_w_out");
    let b_name = format!("{prefix}_b_out");
    let w = g.param(&w_name, Shape::new(&[in_d, out_dim], f));
    let b = g.param(&b_name, Shape::new(&[out_dim], f));
    params.push(ParamSlot {
        name: w_name,
        shape: vec![in_d, out_dim],
        param: w,
        grad: None,
    });
    params.push(ParamSlot {
        name: b_name,
        shape: vec![out_dim],
        param: b,
        grad: None,
    });
    let out = g.mm(h, w);
    g.add(out, b)
}

pub fn flow_map_jump(
    g: &mut Graph,
    a_r: NodeId,
    u: NodeId,
    r: NodeId,
    t: NodeId,
    batch: usize,
) -> NodeId {
    let dt = g.sub(t, r);
    let dt_col = g.reshape_(dt, vec![batch as i64, 1]);
    let scaled = g.mul(u, dt_col);
    g.add(a_r, scaled)
}

pub fn mse_mean(g: &mut Graph, pred: NodeId, target: NodeId) -> NodeId {
    let diff = g.sub(pred, target);
    let sq = g.mul(diff, diff);
    let rank = g.shape(sq).rank();
    let axes: Vec<usize> = (0..rank).collect();
    g.mean(sq, axes, false)
}

pub fn concat_features(g: &mut Graph, parts: Vec<NodeId>) -> NodeId {
    g.concat_(parts, 1)
}

pub fn init_mat(w: &mut crate::graph::actor::WeightStore, name: &str, rows: usize, cols: usize, seed: &mut u64) {
    let scale = (2.0 / (rows + cols) as f32).sqrt();
    let n = rows * cols;
    let mut v = vec![0.0f32; n];
    for x in &mut v {
        *seed = crate::buffer::rand_like(*seed);
        let u = (*seed >> 11) as f32 / (1u32 << 21) as f32;
        *seed = crate::buffer::rand_like(*seed);
        let n2 = (*seed >> 11) as f32 / (1u32 << 21) as f32;
        *x = (u * 2.0 * std::f32::consts::PI * n2).sin() * scale;
    }
    w.0.insert(name.to_string(), v);
}

pub fn init_vec(w: &mut crate::graph::actor::WeightStore, name: &str, n: usize, seed: &mut u64) {
    let mut v = vec![0.0f32; n];
    for x in &mut v {
        *seed = crate::buffer::rand_like(*seed);
        *x = 0.01 * ((*seed >> 11) as f32 / (1u32 << 21) as f32 - 0.5);
    }
    w.0.insert(name.to_string(), v);
}
