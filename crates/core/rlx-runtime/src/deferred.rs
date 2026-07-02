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

//! Dynamic-shape support that works on EVERY backend.
//!
//! When [`crate::Session::compile`] is handed a graph with `Dim::Dynamic` dims, it
//! wraps it in a [`DeferredExecutable`] instead of compiling immediately. On each
//! `run`, the concrete shape is inferred from the input lengths, the graph is
//! specialized to a fully-static graph ([`bind_graph`] + [`sync_graph_shapes`]),
//! and compiled through the normal backend pipeline. The most recent
//! specialization is cached, so repeated runs at the same shape pay no recompile.
//!
//! Because the backend only ever sees a static graph, this gives CPU / Metal /
//! CUDA / ROCm — none of which handle dynamic dims in their kernel dispatch —
//! multiple-shape support for free, matching what wgpu and MLX do internally.

use crate::backend::{Backend, ExecutableGraph};
use rlx_driver::Device;
use rlx_ir::dynamic::{
    bind_graph, infer_bindings_from_f32_inputs, infer_bindings_from_inputs, same_binding,
    sync_graph_shapes,
};
use rlx_ir::{DimBinding, Graph, GraphModule};

/// A not-yet-specialized graph that recompiles per input shape on demand.
pub(crate) struct DeferredExecutable {
    /// The original graph, still carrying `Dim::Dynamic` symbols.
    graph: Graph,
    backend: Box<dyn Backend>,
    options: crate::CompileOptions,
    device: Device,
    /// Params replayed against every fresh specialization (set_param is usually
    /// called once, before the first run, so we must remember them).
    params: Vec<(String, Vec<f32>)>,
    typed_params: Vec<(String, Vec<u8>, rlx_ir::DType)>,
    rng: rlx_ir::RngOptions,
    /// The currently-active specialization and the binding it was built for.
    current: Option<(DimBinding, Box<dyn ExecutableGraph>)>,
}

impl DeferredExecutable {
    pub(crate) fn new(
        graph: Graph,
        backend: Box<dyn Backend>,
        options: crate::CompileOptions,
        device: Device,
    ) -> Self {
        Self {
            graph,
            backend,
            options,
            device,
            params: Vec::new(),
            typed_params: Vec::new(),
            rng: rlx_ir::RngOptions::default(),
            current: None,
        }
    }

    /// Ensure a specialization matching `binding` is compiled and active.
    fn ensure(&mut self, binding: DimBinding) {
        if let Some((prev, _)) = &self.current
            && same_binding(prev, &binding)
        {
            return;
        }
        let mut resolved = bind_graph(&self.graph, &binding);
        // Re-infer every node's shape now that the inputs are concrete, so ops
        // that cache shape attributes (Reshape/Concat/Expand/…) line up.
        sync_graph_shapes(&mut resolved);
        let module = GraphModule::from_graph(resolved);
        let mut inner = self
            .backend
            .compile_module(module, self.device, &self.options)
            .expect("deferred dynamic-shape compile: backend failed on specialized graph");
        inner.set_rng(self.rng);
        for (name, data) in &self.params {
            inner.set_param(name, data);
        }
        for (name, data, dtype) in &self.typed_params {
            inner.set_param_typed(name, data, *dtype);
        }
        if !self.params.is_empty() || !self.typed_params.is_empty() {
            inner.finalize_params();
        }
        self.current = Some((binding, inner));
    }

    fn ensure_from_f32(&mut self, inputs: &[(&str, &[f32])]) {
        let binding = infer_bindings_from_f32_inputs(&self.graph, inputs)
            .expect("deferred dynamic-shape compile: could not infer shape from f32 inputs");
        self.ensure(binding);
    }

    fn ensure_from_typed(&mut self, inputs: &[(&str, &[u8], rlx_ir::DType)]) {
        let lens: Vec<(&str, usize)> = inputs
            .iter()
            .map(|(n, b, dt)| (*n, b.len() / dt.size_bytes().max(1)))
            .collect();
        let binding = infer_bindings_from_inputs(&self.graph, &lens)
            .expect("deferred dynamic-shape compile: could not infer shape from typed inputs");
        self.ensure(binding);
    }

    fn active(&mut self) -> &mut Box<dyn ExecutableGraph> {
        &mut self
            .current
            .as_mut()
            .expect("deferred executable: run before a specialization was compiled")
            .1
    }
}

impl ExecutableGraph for DeferredExecutable {
    fn set_param(&mut self, name: &str, data: &[f32]) {
        match self.params.iter_mut().find(|(n, _)| n == name) {
            Some(slot) => slot.1 = data.to_vec(),
            None => self.params.push((name.to_string(), data.to_vec())),
        }
        if let Some((_, inner)) = &mut self.current {
            inner.set_param(name, data);
        }
    }

    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        match self.typed_params.iter_mut().find(|(n, _, _)| n == name) {
            Some(slot) => {
                slot.1 = data.to_vec();
                slot.2 = dtype;
            }
            None => self
                .typed_params
                .push((name.to_string(), data.to_vec(), dtype)),
        }
        if let Some((_, inner)) = &mut self.current {
            inner.set_param_typed(name, data, dtype);
        }
    }

    fn finalize_params(&mut self) {
        if let Some((_, inner)) = &mut self.current {
            inner.finalize_params();
        }
    }

    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.ensure_from_f32(inputs);
        self.active().run(inputs)
    }

    fn run_read_outputs(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        self.ensure_from_f32(inputs);
        self.active().run_read_outputs(inputs, read_indices)
    }

    fn run_raw(&mut self, inputs: &[(&str, &[f32])]) -> Vec<(*const f32, usize)> {
        self.ensure_from_f32(inputs);
        self.active().run_raw(inputs)
    }

    fn run_typed(
        &mut self,
        inputs: &[(&str, &[u8], rlx_ir::DType)],
    ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
        self.ensure_from_typed(inputs);
        self.active().run_typed(inputs)
    }

    fn set_rng(&mut self, rng: rlx_ir::RngOptions) {
        self.rng = rng;
        if let Some((_, inner)) = &mut self.current {
            inner.set_rng(rng);
        }
    }

    fn rng(&self) -> rlx_ir::RngOptions {
        self.rng
    }

    fn clone_box(&self) -> Box<dyn ExecutableGraph> {
        // A specialization could be cloned, but the unresolved graph + backend
        // cannot be re-created here. Deferred graphs aren't on any clone path
        // today; fail loudly rather than silently dropping the dynamic wrapper.
        panic!("DeferredExecutable (dynamic-shape graph) does not support clone_box");
    }
}
