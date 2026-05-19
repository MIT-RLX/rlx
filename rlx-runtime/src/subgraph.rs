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

//! Sub-graph execution helper.
//!
//! Op::If branches and Op::While body/cond are sub-graphs nested inside
//! a parent graph. To execute them, the backend recursively compiles
//! and runs the inner graph with bound inputs.
//!
//! Strategy: compile the sub-graph lazily on first encounter, cache the
//! `ExecutableGraph` for repeated invocations (loops). A future
//! optimization: hoist the compile to the parent's compile-time once
//! we have a stable IR for sub-graphs.

use crate::CompileOptions;
use crate::backend::{Backend, ExecutableGraph};
use rlx_ir::Graph;
use std::collections::HashMap;

/// Lazily-compiled sub-graph cache.
/// Keyed by sub-graph name (caller must ensure names are unique within
/// the parent graph). Backend-agnostic: stores boxed ExecutableGraphs.
pub struct SubgraphCache {
    cache: HashMap<String, Box<dyn ExecutableGraph>>,
    options: CompileOptions,
}

impl SubgraphCache {
    pub fn new(options: CompileOptions) -> Self {
        Self {
            cache: HashMap::new(),
            options,
        }
    }

    /// Compile a sub-graph if not cached, return mutable executable handle.
    pub fn get_or_compile<'a>(
        &'a mut self,
        backend: &dyn Backend,
        graph: &Graph,
    ) -> &'a mut Box<dyn ExecutableGraph> {
        let key = graph.name.clone();
        self.cache
            .entry(key)
            .or_insert_with(|| backend.compile(graph.clone(), &self.options))
    }

    /// Run a sub-graph with named inputs, returning its outputs.
    pub fn run(
        &mut self,
        backend: &dyn Backend,
        graph: &Graph,
        inputs: &[(&str, &[f32])],
    ) -> Vec<Vec<f32>> {
        let exe = self.get_or_compile(backend, graph);
        exe.run(inputs)
    }
}

/// Helper: evaluate an Op::If by running one of two sub-graphs.
pub fn run_if(
    cache: &mut SubgraphCache,
    backend: &dyn Backend,
    predicate: f32,
    then_branch: &Graph,
    else_branch: &Graph,
    inputs: &[(&str, &[f32])],
) -> Vec<Vec<f32>> {
    let chosen = if predicate != 0.0 {
        then_branch
    } else {
        else_branch
    };
    cache.run(backend, chosen, inputs)
}

/// Helper: evaluate an Op::While by repeatedly running cond + body.
/// `loop_carried` are the values flowing through iterations.
pub fn run_while(
    cache: &mut SubgraphCache,
    backend: &dyn Backend,
    cond: &Graph,
    body: &Graph,
    initial: Vec<Vec<f32>>,
    input_names: &[&str],
    max_iterations: Option<usize>,
) -> Vec<Vec<f32>> {
    let mut state = initial;
    let limit = max_iterations.unwrap_or(usize::MAX);
    for _ in 0..limit {
        // Build named-input slice for cond + body
        let bindings: Vec<(&str, &[f32])> = input_names
            .iter()
            .zip(state.iter())
            .map(|(n, v)| (*n, v.as_slice()))
            .collect();
        let cond_out = cache.run(backend, cond, &bindings);
        // Cond is a scalar bool: stop if it's zero / false
        if cond_out
            .first()
            .map(|v| v.first().copied().unwrap_or(0.0))
            .unwrap_or(0.0)
            == 0.0
        {
            break;
        }
        state = cache.run(backend, body, &bindings);
    }
    state
}
