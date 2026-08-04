// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A compiled, runnable [`Model`] built from an `rlx!` / graph-builder [`Graph`]:
//! bind weights by name and run, without hand-writing the
//! `Session` → `set_param` loop → `run` dance at every call site.

use std::collections::HashSet;

use rlx_ir::{Graph, Op};
use rlx_runtime::{Device, Session};

/// A compiled graph plus its parameter roster — a runnable model.
///
/// Bind weights with [`set`](Self::set) / [`load`](Self::load) *by name* (for an
/// `rlx!` graph that name is the param's `@ "…"` key or, absent that, its
/// binding ident), then [`run`](Self::run). Weight sources that carry extra
/// tensors (a whole checkpoint) are fine — only the params this model declares
/// are bound, and [`unbound`](Self::unbound) reports any it still needs.
///
/// ```ignore
/// let mut m = Model::on(graph, Device::Cpu);
/// m.load(weights.iter().map(|(k, v)| (k.as_str(), v.as_slice())));
/// assert!(m.unbound().is_empty(), "missing weights: {:?}", m.unbound());
/// let out = m.run(&[("input_ids", &ids)]);
/// ```
pub struct Model {
    compiled: rlx_runtime::CompiledGraph,
    params: HashSet<String>,
    bound: HashSet<String>,
}

impl Model {
    /// Compile on the fastest backend this build can run the graph on.
    pub fn new(graph: Graph) -> Self {
        let device = rlx_runtime::fastest_device_for(&graph);
        Self::on(graph, device)
    }

    /// Compile on an explicit device.
    pub fn on(graph: Graph, device: Device) -> Self {
        let params = graph
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Param { name } => Some(name.clone()),
                _ => None,
            })
            .collect();
        Self {
            compiled: Session::new(device).compile(graph),
            params,
            bound: HashSet::new(),
        }
    }

    /// Every parameter name the model expects.
    pub fn param_names(&self) -> impl Iterator<Item = &str> {
        self.params.iter().map(String::as_str)
    }

    /// Parameters not yet bound (in arbitrary order) — check this is empty after
    /// [`load`](Self::load) to catch a missing / misnamed weight loudly.
    pub fn unbound(&self) -> Vec<&str> {
        self.params
            .difference(&self.bound)
            .map(String::as_str)
            .collect()
    }

    /// Bind one parameter by name. A name the model doesn't declare is ignored
    /// (so binding straight from a superset checkpoint is safe).
    pub fn set(&mut self, name: &str, data: &[f32]) -> &mut Self {
        if self.params.contains(name) {
            self.compiled.set_param(name, data);
            self.bound.insert(name.to_string());
        }
        self
    }

    /// Bind every parameter the model declares from a `name → data` source
    /// (a `HashMap`, a slice of pairs, …); extra entries are ignored.
    pub fn load<'a, I>(&mut self, weights: I) -> &mut Self
    where
        I: IntoIterator<Item = (&'a str, &'a [f32])>,
    {
        for (name, data) in weights {
            self.set(name, data);
        }
        self
    }

    /// Run with named inputs, returning one `Vec<f32>` per graph output.
    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.compiled.run(inputs)
    }
}
