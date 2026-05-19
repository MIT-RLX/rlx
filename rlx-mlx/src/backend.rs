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

//! `MlxExecutable` — the per-graph runtime that lower.rs feeds.
//!
//! We keep the compiled graph + a name→f32 map of params/inputs.
//! Every `run()` rebuilds the MLX-side graph fresh (see lower.rs for
//! why). Mode (Eager vs Lazy) is set at compile time via
//! `MlxExecutable::compile_with_mode`.

use std::collections::HashMap;

use rlx_ir::{DType, Graph, NodeId};

use crate::array::{Array, MlxError, synchronize};
use crate::compiled::CompiledFn;
use crate::lower::{self, LeafKey, MlxMode};

pub struct MlxExecutable {
    graph: Graph,
    mode: MlxMode,
    params: HashMap<String, Vec<f32>>,
    /// Persistent inputs (handles) — survive across run() calls and
    /// act as defaults when run() is called without an explicit input
    /// of the same name.
    handles: HashMap<String, Vec<f32>>,
    /// (byte_offset, num_elements) per output. Slots are ordered to
    /// match `graph.outputs`. Filled at compile time from output
    /// shapes; the offsets are stable across `run_slots` calls so the
    /// caller can `arena_ptr().add(offset)` once.
    output_slots: Vec<(usize, usize)>,
    /// Synthesized arena that backs `arena_ptr()` for the slot path.
    /// Outputs are copied into this buffer at the end of `run_slots`.
    arena: Vec<u8>,
    output_names: Vec<NodeId>,
    /// Names of inputs in the order `run_slots` expects them.
    /// Captured at compile time so we can dispatch positional inputs
    /// to the right name without a per-call lookup.
    input_names: Vec<String>,
    /// In-flight outputs from `commit_no_wait`. Held until
    /// `sync_pending` to keep the array refs alive across the async
    /// eval and let later code force their materialization on demand.
    pending: Vec<Array>,
    /// Lazily-built compiled function for `MlxMode::Compiled`. We
    /// can't construct it at compile_with_mode time because the
    /// graph would be moved into both the CompiledFn (for replay)
    /// and the executable's metadata fields. Built on first run().
    compiled: Option<CompiledFn>,
    /// Typed parameters keyed by name — stored separately from
    /// `params` so callers can mix the f32 set_param API with the
    /// typed set_param_typed API without conflicts.
    params_typed: HashMap<String, (Vec<u8>, DType)>,
    /// Typed inputs from `run_typed` calls (transient: filled per
    /// call, not persistent like handles). Kept on the executable
    /// just so the compiled-mode code path can read it.
    inputs_typed: HashMap<String, (Vec<u8>, DType)>,
    /// Output dtypes captured at compile time so `run_typed` can
    /// report the correct dtype for each output without a separate
    /// FFI call.
    output_dtypes: Vec<DType>,
    /// PLAN L1 active-extent hint (`Some((actual, upper))`). When set
    /// AND the graph is in `lower::is_safe_for_active_extent`'s safe
    /// set, lowering slices each input leaf along axis 0 from `upper`
    /// to `actual` before composition; MLX's lazy eval propagates the
    /// smaller shapes through the rest of the trace. Falls back to the
    /// full extent when unset OR the graph contains an unsafe op (e.g.
    /// `Reshape`/`Expand` with a hardcoded `upper` dim, axis-0
    /// `Reduce`/`Cumsum`/`Concat`/`Narrow`).
    active_extent: Option<(usize, usize)>,
}

impl MlxExecutable {
    pub fn compile(graph: Graph) -> Self {
        Self::compile_with_mode(graph, mode_from_env())
    }

    pub fn compile_with_mode(graph: Graph, mode: MlxMode) -> Self {
        let output_names = graph.outputs.clone();

        // Pre-resolve output slot layout. We pack outputs end-to-end
        // in the synthetic arena; offsets are bumped by element-size
        // (always 4 here — slot path is f32-typed since the trait
        // method returns Vec<u8>-as-f32 via arena_ptr).
        let mut output_slots: Vec<(usize, usize)> = Vec::new();
        let mut cursor = 0usize;
        for &out_id in &output_names {
            let shape = &graph.node(out_id).shape;
            let elems = shape.num_elements().unwrap_or(0);
            output_slots.push((cursor, elems));
            cursor += elems * 4; // f32 bytes
        }
        let arena = vec![0u8; cursor];

        // Capture input names in declaration order so run_slots can
        // map positional inputs to the right name without per-call
        // bookkeeping.
        let mut input_names = Vec::new();
        for node in graph.nodes() {
            if let rlx_ir::Op::Input { name } = &node.op {
                input_names.push(name.clone());
            }
        }

        // Capture output dtypes at compile time so run_typed can
        // report them without a per-call FFI roundtrip.
        let output_dtypes: Vec<DType> = output_names
            .iter()
            .map(|&id| graph.node(id).shape.dtype())
            .collect();

        Self {
            graph,
            mode,
            params: HashMap::new(),
            handles: HashMap::new(),
            output_slots,
            arena,
            output_names,
            input_names,
            pending: Vec::new(),
            compiled: None,
            params_typed: HashMap::new(),
            inputs_typed: HashMap::new(),
            output_dtypes,
            active_extent: None,
        }
    }

    /// PLAN L1 — hint the next `run` to compute only the first `actual`
    /// rows along the bucket (outermost) axis (out of `upper`, the
    /// compile extent). Honored when every Op in the graph passes
    /// `lower::is_safe_for_active_extent`; otherwise the lowering path
    /// silently falls back to the full extent. Pass `None` to clear.
    pub fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        self.active_extent = extent;
    }

    /// Eagerly build the compiled fn (otherwise it's lazy on first
    /// run). Useful when callers want to pay the trace cost up front.
    /// No-op for non-Compiled modes.
    pub fn warm_compile(&mut self) -> Result<(), MlxError> {
        if self.mode != MlxMode::Compiled || self.compiled.is_some() {
            return Ok(());
        }
        self.compiled = Some(CompiledFn::compile(self.graph.clone())?);
        Ok(())
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        self.params.insert(name.to_string(), data.to_vec());
        // Drop any typed override so subsequent runs see the f32 data.
        self.params_typed.remove(name);
    }

    /// Bind a parameter from raw bytes in the given dtype. No f32
    /// widen/narrow round-trip — the bytes feed straight into
    /// Array::from_bytes during lowering.
    pub fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: DType) {
        self.params_typed
            .insert(name.to_string(), (data.to_vec(), dtype));
        // Drop any f32 override so subsequent runs see the typed data.
        self.params.remove(name);
    }

    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        // Drain anything still in flight from a prior commit_no_wait
        // before we mutate input data — otherwise the async run might
        // observe partially-overwritten inputs.
        self.sync_pending();

        // Build a name→data map for inputs, applying handles first then
        // letting explicit inputs override.
        let mut input_map: HashMap<String, Vec<f32>> = self.handles.clone();
        for &(name, data) in inputs {
            input_map.insert(name.to_string(), data.to_vec());
        }

        let outs = if self.mode == MlxMode::Compiled {
            match self.run_compiled(&input_map) {
                Ok(outs) => outs,
                Err(e) => panic!("MLX compiled run failed: {e}"),
            }
        } else {
            // Thread typed param overrides so set_param_typed-bound
            // weights are honored on the f32 run() path too. PLAN L1
            // active-extent (when set + safe) is honored inside the
            // `_with_extent` variant by slicing input leaves.
            match lower::lower_and_run_typed_with_extent(
                &self.graph,
                &self.params,
                &self.params_typed,
                &input_map,
                &self.inputs_typed,
                self.mode,
                self.active_extent,
            ) {
                Ok(outs) => outs,
                Err(e) => panic!("MLX backend run failed: {e}"),
            }
        };

        let result: Vec<Vec<f32>> = outs
            .iter()
            .map(|a| a.to_f32().unwrap_or_default())
            .collect();
        // KV-cache pattern (matches the CPU backend): if a
        // persistent handle's name matches "out{i}" for an
        // output slot, sync the f32 data back so the next run
        // picks it up as the input of the same name.
        for (i, vals) in result.iter().enumerate() {
            let name = format!("out{i}");
            if self.handles.contains_key(&name) {
                self.handles.insert(name, vals.clone());
            }
        }
        result
    }

    /// Run with typed inputs and read outputs back as raw bytes in
    /// each output's native dtype. Combines with `set_param_typed`
    /// for a true zero-widen path through the backend.
    pub fn run_typed(&mut self, inputs: &[(&str, &[u8], DType)]) -> Vec<(Vec<u8>, DType)> {
        self.sync_pending();

        // Stash typed inputs so run_compiled / lower_and_run_typed
        // can read them. Cleared at the end so the executable doesn't
        // hold onto user buffers longer than needed.
        self.inputs_typed.clear();
        for (name, data, dt) in inputs {
            self.inputs_typed
                .insert(name.to_string(), (data.to_vec(), *dt));
        }

        let outs = if self.mode == MlxMode::Compiled {
            // Compiled-mode picks up self.inputs_typed via run_compiled's
            // call to build_leaf_for, which already threads the typed maps.
            // input_map (f32) stays empty for typed-only runs.
            match self.run_compiled(&HashMap::new()) {
                Ok(o) => o,
                Err(e) => panic!("MLX compiled run_typed failed: {e}"),
            }
        } else {
            match lower::lower_and_run_typed_with_extent(
                &self.graph,
                &self.params,
                &self.params_typed,
                &HashMap::new(),
                &self.inputs_typed,
                self.mode,
                self.active_extent,
            ) {
                Ok(o) => o,
                Err(e) => panic!("MLX run_typed failed: {e}"),
            }
        };

        self.inputs_typed.clear();

        // Read back as native bytes.
        outs.iter()
            .enumerate()
            .map(|(i, a)| {
                let bytes = a.to_bytes().unwrap_or_default();
                let dt = *self.output_dtypes.get(i).unwrap_or(&DType::F32);
                (bytes, dt)
            })
            .collect()
    }

    /// Compiled-mode dispatch. Builds leaf arrays from current host
    /// data in the order the compiled fn expects and invokes the
    /// compiled trace. Returns symbolic outputs — caller chooses
    /// `eval` (sync) or `async_eval` (no wait) before readback.
    fn run_compiled(
        &mut self,
        input_map: &HashMap<String, Vec<f32>>,
    ) -> Result<Vec<Array>, MlxError> {
        if self.compiled.is_none() {
            self.compiled = Some(CompiledFn::compile(self.graph.clone())?);
        }
        let compiled = self.compiled.as_ref().unwrap();
        let order = compiled.leaf_order();

        // Build leaves in the exact order the compiled fn expects.
        let mut leaves: Vec<Array> = Vec::with_capacity(order.len());
        for (id, key) in order {
            let leaf = match key {
                LeafKey::Input(_) | LeafKey::Param(_) | LeafKey::Constant => lower::build_leaf_for(
                    &self.graph,
                    *id,
                    &self.params,
                    input_map,
                    &self.params_typed,
                    &self.inputs_typed,
                )?,
            };
            leaves.push(leaf);
        }

        compiled.invoke(&leaves)
    }

    pub fn arena_ptr(&self) -> *const u8 {
        self.arena.as_ptr()
    }

    /// Fast positional path for users who know their inputs by index.
    /// Same lowering as `run()`, but skips name-based lookups and
    /// copies outputs into the synthetic arena so callers can read
    /// them via `arena_ptr().add(offset)` without per-output
    /// allocations.
    pub fn run_slots(&mut self, inputs: &[&[f32]]) -> &[(usize, usize)] {
        self.sync_pending();

        // Build a name→data map by zipping positional inputs against
        // the captured input_names. Anything beyond what was supplied
        // falls through to handles.
        let mut input_map: HashMap<String, Vec<f32>> = self.handles.clone();
        for (i, &data) in inputs.iter().enumerate() {
            if let Some(name) = self.input_names.get(i) {
                input_map.insert(name.clone(), data.to_vec());
            }
        }

        let lowered = if self.mode == MlxMode::Compiled {
            self.run_compiled(&input_map)
        } else {
            lower::lower_and_run_typed(
                &self.graph,
                &self.params,
                &self.params_typed,
                &input_map,
                &self.inputs_typed,
                self.mode,
            )
        };
        match lowered {
            Ok(outs) => {
                // Copy each output into its slot in the synthetic arena.
                for (i, arr) in outs.iter().enumerate() {
                    let (off, n) = self.output_slots[i];
                    let v = match arr.to_f32() {
                        Ok(v) => v,
                        Err(e) => panic!("MLX run_slots readback failed: {e}"),
                    };
                    let want_bytes = n * 4;
                    let end = off + want_bytes;
                    if end <= self.arena.len() && v.len() == n {
                        // SAFETY: we own self.arena, the destination is
                        // 4-byte aligned by construction (Vec<u8>'s
                        // start + 4-byte-stride offsets), and we've
                        // bounds-checked end. The source is a valid
                        // contiguous f32 slice.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                v.as_ptr() as *const u8,
                                self.arena.as_mut_ptr().add(off),
                                want_bytes,
                            );
                        }
                    }
                }
                &self.output_slots
            }
            Err(e) => panic!("MLX run_slots failed: {e}"),
        }
    }

    pub fn commit_no_wait(&mut self, inputs: &[(&str, &[f32])]) {
        // Drain any prior in-flight work so we don't accumulate.
        self.sync_pending();
        let mut input_map: HashMap<String, Vec<f32>> = self.handles.clone();
        for &(name, data) in inputs {
            input_map.insert(name.to_string(), data.to_vec());
        }

        if self.mode == MlxMode::Compiled {
            // Compiled-mode async: invoke the compiled fn (replays the
            // optimized trace), then async_eval its outputs without
            // waiting. sync_pending later drains.
            match self.run_compiled(&input_map) {
                Ok(outs) => {
                    let refs: Vec<&Array> = outs.iter().collect();
                    if let Err(e) = crate::array::async_eval(&refs) {
                        panic!("MLX compiled commit_no_wait async_eval failed: {e}");
                    }
                    self.pending = outs;
                }
                Err(e) => panic!("MLX compiled commit_no_wait failed: {e}"),
            }
            return;
        }

        match lower::lower_and_run_typed(
            &self.graph,
            &self.params,
            &self.params_typed,
            &input_map,
            &self.inputs_typed,
            MlxMode::AsyncCommit,
        ) {
            Ok(outs) => self.pending = outs,
            Err(e) => panic!("MLX commit_no_wait failed: {e}"),
        }
    }

    pub fn sync_pending(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        // Wait for the in-flight async eval to finish.
        if let Err(e) = synchronize() {
            panic!("MLX sync_pending failed: {e}");
        }
        self.pending.clear();
    }

    pub fn run_pipelined(&mut self, input_sets: &[Vec<(&str, &[f32])>]) -> Vec<Vec<Vec<f32>>> {
        input_sets
            .iter()
            .map(|inputs| {
                let refs: Vec<(&str, &[f32])> = inputs.iter().map(|(n, d)| (*n, *d)).collect();
                self.run(&refs)
            })
            .collect()
    }

    pub fn bind_handle(&mut self, name: &str, data: &[f32]) -> bool {
        self.handles.insert(name.to_string(), data.to_vec());
        true
    }

    pub fn read_handle(&self, name: &str) -> Option<Vec<f32>> {
        self.handles.get(name).cloned()
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }
    pub fn mode(&self) -> MlxMode {
        self.mode
    }
    pub fn output_ids(&self) -> &[NodeId] {
        &self.output_names
    }
}

/// Read `RLX_MLX_MODE=eager|lazy|compiled` (case-insensitive) and
/// pick a default. `compiled` enables persistent `mlx::compile` trace
/// caching; `eager` evals after every op (debug-friendly); `lazy`
/// (default) evals once per run.
fn mode_from_env() -> MlxMode {
    match std::env::var("RLX_MLX_MODE").ok().as_deref() {
        Some(s) if s.eq_ignore_ascii_case("eager") => MlxMode::Eager,
        Some(s) if s.eq_ignore_ascii_case("compiled") => MlxMode::Compiled,
        _ => MlxMode::Lazy,
    }
}
