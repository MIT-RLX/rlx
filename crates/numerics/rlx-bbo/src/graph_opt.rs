// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Adam on compiled RLX loss graphs via reverse-mode AD (`grad_with_loss`).
//!
//! Replaces hand-rolled Adam loops in domain crates (RF passives, LNA match,
//! placement demos). Scalar FD [`adam_opt_nd`] remains for black-box objectives.

use std::collections::HashMap;

use rlx_ir::{
    Graph, NodeId, find_param_node as ir_find_param_node, find_param_nodes as ir_find_param_nodes,
};
use rlx_opt::rlx_autodiff::grad_with_loss;
use rlx_optim::{Adam, Optimizer};
use rlx_runtime::{CompiledGraph, Device, Session};
use serde::{Deserialize, Serialize};

/// Resolve a single `Op::Param` node by name.
pub fn find_param_node(g: &Graph, name: &str) -> Option<NodeId> {
    ir_find_param_node(g, name)
}

/// Resolve param nodes in the same order as `names`.
pub fn find_param_nodes(g: &Graph, names: &[&str]) -> Result<Vec<NodeId>, GraphOptError> {
    ir_find_param_nodes(g, names).map_err(GraphOptError::ParamNotFound)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphOptConfig {
    pub steps: u32,
    /// Base Adam learning rate (see [`relative_lr`]).
    pub lr: f32,
    /// When true, each optimized coordinate is scaled by `max(|x|, lr_floor)`
    /// before the Adam update — useful when params span orders of magnitude
    /// (e.g. Lg ≈ 17 nH vs gm ≈ 50 mS).
    pub relative_lr: bool,
    pub lr_floor: f32,
    pub beta1: f32,
    pub beta2: f32,
}

impl Default for GraphOptConfig {
    fn default() -> Self {
        Self {
            steps: 128,
            lr: 0.02,
            relative_lr: true,
            lr_floor: 1e-12,
            beta1: 0.9,
            beta2: 0.999,
        }
    }
}

impl GraphOptConfig {
    #[must_use]
    pub fn from_steps(steps: u32) -> Self {
        Self {
            steps,
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphOptResult {
    /// Final value for every param present in the spec (optimized + frozen).
    pub params: HashMap<String, f32>,
    pub final_loss: f32,
    pub history: Vec<f32>,
    pub final_grads: HashMap<String, f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GraphOptError {
    ParamNotFound(String),
    OptimizeEmpty,
    GradcheckMismatch { param: String, ad: f32, fd: f32 },
}

impl std::fmt::Display for GraphOptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParamNotFound(n) => write!(f, "param not found in graph: {n}"),
            Self::OptimizeEmpty => write!(f, "optimize list is empty"),
            Self::GradcheckMismatch { param, ad, fd } => {
                write!(f, "gradcheck mismatch at {param}: AD={ad:.6e} FD={fd:.6e}")
            }
        }
    }
}

impl std::error::Error for GraphOptError {}

/// Parameter bundle for [`adam_opt_graph`].
pub struct GraphOptSpec<'a> {
    /// Names optimized by Adam (must be `Op::Param` in `fwd`).
    pub optimize: &'a [&'a str],
    /// Initial values for **all** params referenced by the graph (optimized + frozen).
    pub values: HashMap<String, f32>,
    /// Per-coordinate bounds (only required for optimized names).
    pub bounds: HashMap<String, (f32, f32)>,
    /// Forward inputs (e.g. `("freq_hz", &[2.4e9])`). `d_output` is injected automatically.
    pub inputs: &'a [(&'a str, &'a [f32])],
}

/// Compile `grad_with_loss` on `fwd` and run Adam.
pub fn adam_opt_graph(
    fwd: &Graph,
    spec: &GraphOptSpec<'_>,
    cfg: &GraphOptConfig,
    device: Device,
) -> Result<GraphOptResult, GraphOptError> {
    if spec.optimize.is_empty() {
        return Err(GraphOptError::OptimizeEmpty);
    }

    let param_ids = find_param_nodes(fwd, spec.optimize)?;
    let bwd = grad_with_loss(fwd, &param_ids);
    let session = Session::new(device);
    let mut compiled = session.compile(bwd);

    let mut opt_values: Vec<f32> = spec
        .optimize
        .iter()
        .map(|n| {
            spec.values
                .get(*n)
                .copied()
                .ok_or_else(|| GraphOptError::ParamNotFound((*n).into()))
        })
        .collect::<Result<_, _>>()?;

    let mut opt = Adam::new(cfg.lr).with_betas(cfg.beta1, cfg.beta2);
    let mut history = Vec::with_capacity(cfg.steps as usize);
    let mut last_grads: HashMap<String, f32> = HashMap::new();
    let mut last_loss = f32::MAX;

    for _ in 0..cfg.steps {
        apply_all_params(&mut compiled, &spec.values, spec.optimize, &opt_values);

        let mut run_in: Vec<(&str, &[f32])> = spec.inputs.to_vec();
        run_in.push(("d_output", &[1.0]));
        let outs = compiled.run(&run_in);
        last_loss = outs[0][0];
        history.push(last_loss);

        let mut scaled_grads = Vec::with_capacity(opt_values.len());
        for (i, gout) in outs[1..].iter().enumerate() {
            let g = gout[0];
            let name = spec.optimize[i];
            last_grads.insert(name.to_string(), g);
            let scale = if cfg.relative_lr {
                opt_values[i].abs().max(cfg.lr_floor)
            } else {
                1.0
            };
            scaled_grads.push(g * scale);
        }

        opt.lr = cfg.lr;
        opt.step(
            "params",
            &[opt_values.len()],
            &mut opt_values,
            &scaled_grads,
        );
        opt.end_iteration();

        for (i, name) in spec.optimize.iter().enumerate() {
            if let Some(&(lo, hi)) = spec.bounds.get(*name) {
                opt_values[i] = opt_values[i].clamp(lo, hi);
            }
        }
    }

    let mut params = spec.values.clone();
    for (name, val) in spec.optimize.iter().zip(opt_values.iter()) {
        params.insert((*name).to_string(), *val);
    }

    Ok(GraphOptResult {
        params,
        final_loss: last_loss,
        history,
        final_grads: last_grads,
    })
}

pub(crate) fn apply_all_params(
    compiled: &mut CompiledGraph,
    all: &HashMap<String, f32>,
    optimize: &[&str],
    opt_values: &[f32],
) {
    for (name, val) in all {
        if !optimize.contains(&name.as_str()) {
            compiled.set_param(name, &[*val]);
        }
    }
    for (name, val) in optimize.iter().zip(opt_values.iter()) {
        compiled.set_param(name, &[*val]);
    }
}

#[cfg(test)]
mod tests {
    use rlx_ir::{DType, Graph, Op, Shape, op::BinaryOp};

    use super::*;

    fn quadratic_loss_graph() -> (Graph, &'static str) {
        let mut g = Graph::new("quad");
        let s = Shape::new(&[1], DType::F32);
        let x = g.param("x", s.clone());
        let target = g.add_node(
            Op::Constant {
                data: 2.0f32.to_le_bytes().to_vec(),
            },
            vec![],
            s.clone(),
        );
        let err = g.binary(BinaryOp::Sub, x, target, s.clone());
        let loss = g.binary(BinaryOp::Mul, err, err, s);
        g.set_outputs(vec![loss]);
        (g, "x")
    }

    #[test]
    fn parabolic_1d_converges() {
        let (fwd, pname) = quadratic_loss_graph();
        let values = HashMap::from([(pname.to_string(), 0.0f32)]);
        let bounds = HashMap::from([(pname.to_string(), (-10.0, 10.0))]);
        let spec = GraphOptSpec {
            optimize: &[pname],
            values,
            bounds,
            inputs: &[],
        };
        let cfg = GraphOptConfig {
            steps: 96,
            lr: 0.15,
            relative_lr: false,
            ..Default::default()
        };
        let r = adam_opt_graph(&fwd, &spec, &cfg, Device::Cpu).unwrap();
        assert!(
            r.final_loss < 0.01,
            "loss={} x={}",
            r.final_loss,
            r.params[pname]
        );
        assert!((r.params[pname] - 2.0).abs() < 0.08);
    }
}
