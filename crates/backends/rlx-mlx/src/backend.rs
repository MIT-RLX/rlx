// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `MlxExecutable` — the per-graph runtime that lower.rs feeds.
//!
//! We keep the compiled graph + a name→f32 map of params/inputs.
//! Every `run()` rebuilds the MLX-side graph fresh (see lower.rs for
//! why). Mode (Eager vs Lazy) is set at compile time via
//! `MlxExecutable::compile_with_mode`.

use std::collections::HashMap;

use rlx_ir::{DType, Graph, NodeId, Op};

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
    /// GPU-resident inputs — reused across `run()` without host upload.
    gpu_handles: HashMap<String, Array>,
    /// Device-resident param arrays, materialized once and reused across
    /// `run()` (compiled path) so multi-GB weights aren't rebuilt from the host
    /// `params` map on every call. Invalidated per-name on
    /// `set_param`/`set_param_typed`/`copy_params_from`.
    gpu_params: HashMap<String, Array>,
    /// After each `run`, copy `outputs[idx]` into the named GPU handle.
    gpu_handle_feeds: HashMap<String, usize>,
    /// Row feeds for resident KV: [`feed_kv_row`] copies one output row into the handle.
    kv_row_feeds: HashMap<String, usize>,
    /// Last run outputs — used by [`feed_kv_row`] before readback.
    last_outputs: Vec<Array>,
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
    /// Set once `CompiledFn::compile` refuses the graph (host-eval op
    /// detected). Subsequent `run` calls take the Lazy path instead of
    /// re-attempting compile every step.
    compile_disabled: Option<String>,
    /// Runtime-mutable RNG policy for in-graph random ops.
    rng: std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
}

// RLX_MLX_PROFILE: when set, dump the per-op-kind wall-time breakdown once,
// when the (typically sole) executable is dropped at the end of the run.
impl Drop for MlxExecutable {
    fn drop(&mut self) {
        static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if std::env::var_os("RLX_MLX_PROFILE").is_some()
            && !REPORTED.swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            crate::lower::mlx_profile_report();
        }
    }
}

impl MlxExecutable {
    fn current_rng(&self) -> rlx_ir::RngOptions {
        *self.rng.read().expect("rng lock")
    }

    /// Override RNG policy for in-graph random ops without recompiling.
    pub fn set_rng(&mut self, rng: rlx_ir::RngOptions) {
        *self.rng.write().expect("rng lock") = rng;
    }

    /// Current RNG compile/execute policy.
    pub fn rng(&self) -> rlx_ir::RngOptions {
        self.current_rng()
    }
}

impl MlxExecutable {
    pub fn compile(graph: Graph) -> Self {
        Self::compile_with_mode(graph, mode_from_env())
    }

    pub fn compile_with_mode(graph: Graph, mode: MlxMode) -> Self {
        Self::compile_from_fused(graph, mode)
    }

    /// Compile a graph that already went through the fusion pipeline
    /// (e.g. from [`rlx_ir::LirModule`]). Does not re-run fusion passes.
    pub fn compile_from_fused(graph: Graph, mode: MlxMode) -> Self {
        Self::compile_from_fused_with_rng(graph, mode, rlx_ir::RngOptions::default())
    }

    pub fn compile_from_fused_with_rng(
        graph: Graph,
        mode: MlxMode,
        rng: rlx_ir::RngOptions,
    ) -> Self {
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
            gpu_handles: HashMap::new(),
            gpu_params: HashMap::new(),
            gpu_handle_feeds: HashMap::new(),
            kv_row_feeds: HashMap::new(),
            last_outputs: Vec::new(),
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
            compile_disabled: None,
            rng: std::sync::Arc::new(std::sync::RwLock::new(rng)),
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
        let _guard = crate::sync::runtime_guard();
        self.maybe_disable_compile_for_graph_size();
        if self.mode != MlxMode::Compiled
            || self.compiled.is_some()
            || self.compile_disabled.is_some()
        {
            return Ok(());
        }
        match CompiledFn::compile(self.graph.clone()) {
            Ok(c) => {
                self.compiled = Some(c);
                Ok(())
            }
            Err(e) => {
                self.note_compile_disabled(e.to_string());
                Ok(())
            }
        }
    }

    /// Skip `mlx::compile` for oversized graphs. First `invoke` runs
    /// `compile_simplify`, which can hang unboundedly on large TTS/vocoder
    /// IR (Kokoro decoder ~2400 nodes). Lazy finishes in seconds.
    fn maybe_disable_compile_for_graph_size(&mut self) {
        if self.mode != MlxMode::Compiled || self.compile_disabled.is_some() {
            return;
        }
        let Some(limit) = compile_max_nodes_from_env() else {
            return;
        };
        let n = self.graph.nodes().len();
        if n > limit {
            self.note_compile_disabled(format!(
                "graph has {n} nodes (limit {limit}); mlx::compile_simplify is unbounded on \
                 large TTS/vocoder graphs — set RLX_MLX_COMPILE_MAX_NODES=0 to force Compiled"
            ));
        }
    }

    /// True if this executable has fallen back from Compiled → Lazy
    /// because the graph contains a host-eval op.
    pub fn compile_disabled_reason(&self) -> Option<&str> {
        self.compile_disabled.as_deref()
    }

    fn note_compile_disabled(&mut self, reason: String) {
        // Warn at most once per distinct reason per process. Many models build
        // several executables that share the same host-eval op (e.g. one deform
        // module per encoder layer), so a per-executable warning floods the log
        // with identical lines. The fallback itself is correctness-neutral —
        // Lazy mode runs the host-eval op fine; it only forgoes compile-trace
        // caching, which mainly benefits repeated-decode loops, not one-shot
        // graphs. Set `RLX_MLX_WARN_LAZY=all` to restore per-executable warnings.
        if self.compile_disabled.is_none() {
            use std::collections::HashSet;
            use std::sync::{Mutex, OnceLock};
            static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
            let warn_all = std::env::var("RLX_MLX_WARN_LAZY")
                .map(|v| v.eq_ignore_ascii_case("all"))
                .unwrap_or(false);
            let first = {
                let mut seen = SEEN
                    .get_or_init(|| Mutex::new(HashSet::new()))
                    .lock()
                    .unwrap();
                seen.insert(reason.clone())
            };
            if warn_all || first {
                eprintln!(
                    "rlx-mlx: falling back to MlxMode::Lazy — compile mode unsupported for this \
                     graph: {reason}"
                );
            }
        }
        self.compile_disabled = Some(reason);
    }

    fn use_compiled(&self) -> bool {
        // `active_extent` is a per-call shape hint for Lazy lowering
        // (PLAN L1 bucket trimming); the compiled trace is built once
        // at a fixed shape, so honoring it requires Lazy.
        self.mode == MlxMode::Compiled
            && self.compile_disabled.is_none()
            && self.active_extent.is_none()
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        // Mutate the existing Vec in place when possible so the
        // backing buffer's address stays stable across calls. This
        // lets `build_leaf_for` construct a zero-copy MLX Array view
        // over the persistent buffer instead of copying through MLX's
        // float-iterator constructor on every step. Realloc only on
        // first set (or if the caller hands us a different length).
        match self.params.get_mut(name) {
            Some(existing) if existing.len() == data.len() => {
                existing.copy_from_slice(data);
            }
            _ => {
                self.params.insert(name.to_string(), data.to_vec());
            }
        }
        // Drop any typed override so subsequent runs see the f32 data.
        self.params_typed.remove(name);
        // Invalidate the device-resident cache so the new data is re-materialized.
        self.gpu_params.remove(name);
    }

    /// Bind a parameter from raw bytes in the given dtype. No f32
    /// widen/narrow round-trip — the bytes feed straight into
    /// Array::from_bytes during lowering.
    pub fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: DType) {
        self.params_typed
            .insert(name.to_string(), (data.to_vec(), dtype));
        // Drop any f32 override so subsequent runs see the typed data.
        self.params.remove(name);
        self.gpu_params.remove(name);
    }

    /// Copy named params from another executable (decode bucket weight sharing).
    pub fn copy_params_from(&mut self, other: &Self) -> bool {
        if self.params.len() != other.params.len()
            || self.params_typed.len() != other.params_typed.len()
        {
            return false;
        }
        for (name, data) in &other.params {
            let Some(dst) = self.params.get_mut(name) else {
                return false;
            };
            if dst.len() != data.len() {
                return false;
            }
            dst.copy_from_slice(data);
        }
        for (name, (data, dtype)) in &other.params_typed {
            let Some((dst, dt)) = self.params_typed.get_mut(name) else {
                return false;
            };
            if dst.len() != data.len() || *dt != *dtype {
                return false;
            }
            dst.copy_from_slice(data);
        }
        // Mutated in place → drop any device-resident cache for these params.
        self.gpu_params.clear();
        true
    }

    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.run_read_outputs(inputs, None)
            .unwrap_or_else(|e| panic!("MLX backend run failed: {e}"))
    }

    /// Run the graph and read back only the listed output indices (e.g. `[0]` for logits-only).
    /// GPU handle feeds still run for every output. Pass `None` to read all outputs.
    pub fn run_read_outputs(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Result<Vec<Vec<f32>>, MlxError> {
        match self.run_read_outputs_inner(inputs, read_indices) {
            Ok(o) => Ok(o),
            // A compiled kernel can fail lazily when its output is materialized —
            // e.g. an over-fused elementwise region (grid_sample) exhausts Metal's
            // argument buffers. Disable compile and retry in Lazy, which caps
            // fusion depth with eval barriers. Only retry once (compile now off).
            Err(e) if self.compile_disabled.is_none() => {
                self.note_compile_disabled(e.to_string());
                self.compiled = None;
                self.run_read_outputs_inner(inputs, read_indices)
            }
            Err(e) => Err(e),
        }
    }

    fn run_read_outputs_inner(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Result<Vec<Vec<f32>>, MlxError> {
        // Hold across `to_f32` as well: `run_arrays`' guard would otherwise
        // drop before the shim's eval-on-read, racing parallel test threads /
        // cargo-test binaries (SIGTRAP / SIGSEGV). Reentrant with the nested
        // guard inside `run_arrays`.
        let _guard = crate::sync::runtime_guard();
        let outs = self.run_arrays(inputs)?;
        let indices: Vec<usize> = match read_indices {
            None => (0..outs.len()).collect(),
            Some(ix) => ix.to_vec(),
        };
        let f32_outs: Vec<Vec<f32>> = indices
            .iter()
            .map(|&i| {
                outs.get(i)
                    .ok_or_else(|| MlxError(format!("output index {i} missing")))?
                    .to_f32()
            })
            .collect::<Result<Vec<_>, _>>()?;
        // NaN/Inf output-boundary scan (RLX_DEBUG_NANS). MLX execution is
        // delegated to its C++ runtime with no per-op host boundary, so we scan
        // the outputs; for internal localization replay on the CPU backend.
        let scanner = rlx_ir::numeric_check::DebugScanner::from_env("mlx");
        if scanner.enabled() {
            for (&i, buf) in indices.iter().zip(&f32_outs) {
                if let Some(&id) = self.graph.outputs.get(i) {
                    scanner.check(&self.graph, id, buf, &[]);
                }
            }
        }
        Ok(f32_outs)
    }

    /// Execute and return MLX output arrays (after GPU handle refresh).
    fn run_arrays(&mut self, inputs: &[(&str, &[f32])]) -> Result<Vec<Array>, MlxError> {
        let _guard = crate::sync::runtime_guard();
        self.sync_pending_inner();
        self.maybe_disable_compile_for_graph_size();
        let mut input_map: HashMap<String, Vec<f32>> = self.handles.clone();
        for &(name, data) in inputs {
            input_map.insert(name.to_string(), data.to_vec());
        }

        let outs = if self.use_compiled() {
            self.run_compiled(&input_map)?
        } else {
            lower::lower_and_run_typed_with_extent(
                &self.graph,
                &self.params,
                &self.params_typed,
                &input_map,
                &self.inputs_typed,
                if self.compile_disabled.is_some() {
                    MlxMode::Lazy
                } else {
                    self.mode
                },
                self.active_extent,
                Some(&self.gpu_handles),
                self.current_rng(),
            )?
        };

        self.refresh_gpu_handles_from_outputs(&outs)?;
        self.last_outputs = outs.iter().filter_map(|a| a.clone_handle().ok()).collect();
        Ok(outs)
    }

    /// Run with typed inputs and read outputs back as raw bytes in
    /// each output's native dtype. Combines with `set_param_typed`
    /// for a true zero-widen path through the backend.
    pub fn run_typed(&mut self, inputs: &[(&str, &[u8], DType)]) -> Vec<(Vec<u8>, DType)> {
        let _guard = crate::sync::runtime_guard();
        self.sync_pending_inner();
        self.maybe_disable_compile_for_graph_size();

        // Stash typed inputs so run_compiled / lower_and_run_typed
        // can read them. Cleared at the end so the executable doesn't
        // hold onto user buffers longer than needed.
        //
        // F5 (and other ONNX-f16 models) often feed F16 host bytes while
        // `prepare_f32_exec_graph` rewrote Input shapes to F32. Widen at
        // the boundary instead of requiring callers to match exactly.
        self.inputs_typed.clear();
        for (name, data, dt) in inputs {
            let graph_dt = self
                .graph
                .nodes()
                .iter()
                .find(|n| matches!(&n.op, Op::Input { name: nme } if nme == name))
                .map(|n| n.shape.dtype())
                .unwrap_or(*dt);
            let (bytes, stored_dt) = match (*dt, graph_dt) {
                (DType::F16, DType::F32) => {
                    let f: Vec<f32> = data
                        .chunks_exact(2)
                        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect();
                    (f.iter().flat_map(|v| v.to_le_bytes()).collect(), DType::F32)
                }
                (DType::BF16, DType::F32) => {
                    let f: Vec<f32> = data
                        .chunks_exact(2)
                        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect();
                    (f.iter().flat_map(|v| v.to_le_bytes()).collect(), DType::F32)
                }
                // Metal/CPU widen I64 activations to F32; MLX graphs after
                // prepare_f32 may still declare I64 inputs while some leaves
                // were rewritten — also accept I64/I32 host bytes for F32 slots.
                (DType::I64, DType::F32) => {
                    let f: Vec<f32> = data
                        .chunks_exact(8)
                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                        .collect();
                    (f.iter().flat_map(|v| v.to_le_bytes()).collect(), DType::F32)
                }
                (DType::I32, DType::F32) => {
                    let f: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                        .collect();
                    (f.iter().flat_map(|v| v.to_le_bytes()).collect(), DType::F32)
                }
                _ => (data.to_vec(), *dt),
            };
            self.inputs_typed
                .insert(name.to_string(), (bytes, stored_dt));
        }

        let outs = if self.use_compiled() {
            match self.run_compiled(&HashMap::new()) {
                Ok(o) => o,
                Err(e) => {
                    self.note_compile_disabled(e.to_string());
                    let lower_mode = MlxMode::Lazy;
                    lower::lower_and_run_typed_with_extent(
                        &self.graph,
                        &self.params,
                        &self.params_typed,
                        &HashMap::new(),
                        &self.inputs_typed,
                        lower_mode,
                        self.active_extent,
                        Some(&self.gpu_handles),
                        self.current_rng(),
                    )
                    .unwrap_or_else(|e2| panic!("MLX run_typed failed: {e2}"))
                }
            }
        } else {
            let lower_mode = if self.compile_disabled.is_some() {
                MlxMode::Lazy
            } else {
                self.mode
            };
            match lower::lower_and_run_typed_with_extent(
                &self.graph,
                &self.params,
                &self.params_typed,
                &HashMap::new(),
                &self.inputs_typed,
                lower_mode,
                self.active_extent,
                Some(&self.gpu_handles),
                self.current_rng(),
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
            match CompiledFn::compile(self.graph.clone()) {
                Ok(c) => self.compiled = Some(c),
                Err(e) => {
                    self.note_compile_disabled(e.to_string());
                    return lower::lower_and_run_typed_with_extent(
                        &self.graph,
                        &self.params,
                        &self.params_typed,
                        input_map,
                        &self.inputs_typed,
                        MlxMode::Lazy,
                        self.active_extent,
                        Some(&self.gpu_handles),
                        self.current_rng(),
                    );
                }
            }
        }
        // Clone the leaf order so the immutable borrow of `compiled` ends before
        // we mutate `self.gpu_params` below.
        let order: Vec<(NodeId, LeafKey)> = self.compiled.as_ref().unwrap().leaf_order().to_vec();

        // Build leaves in the exact order the compiled fn expects. Params are
        // materialized on device once and cached, so multi-GB weights aren't
        // rebuilt from the host `params` map on every run (that copy was the
        // dominant cost for weight-heavy models).
        let mut leaves: Vec<Array> = Vec::with_capacity(order.len());
        for (id, key) in &order {
            let leaf = match key {
                LeafKey::Param(name)
                    if !self.params_typed.contains_key(name) && self.params.contains_key(name) =>
                {
                    if let Some(cached) = self.gpu_params.get(name) {
                        cached.clone_handle()?
                    } else {
                        let a = lower::build_leaf_for(
                            &self.graph,
                            *id,
                            &self.params,
                            input_map,
                            &self.params_typed,
                            &self.inputs_typed,
                            Some(&self.gpu_handles),
                        )?;
                        crate::array::eval(&[&a])?; // force device materialization
                        self.gpu_params.insert(name.clone(), a.clone_handle()?);
                        a
                    }
                }
                LeafKey::Input(_) | LeafKey::Param(_) | LeafKey::Constant => lower::build_leaf_for(
                    &self.graph,
                    *id,
                    &self.params,
                    input_map,
                    &self.params_typed,
                    &self.inputs_typed,
                    Some(&self.gpu_handles),
                )?,
            };
            leaves.push(leaf);
        }

        // A compiled kernel can still fail at *invoke* time — e.g. a very deep
        // fused elementwise region (grid_sample decomposition) exhausts Metal's
        // argument buffers. Lazy lowering caps fusion depth, so fall back to it.
        let compiled = self.compiled.as_ref().unwrap();
        match compiled.invoke(&leaves) {
            Ok(o) => Ok(o),
            Err(e) => {
                self.note_compile_disabled(e.to_string());
                self.compiled = None;
                lower::lower_and_run_typed_with_extent(
                    &self.graph,
                    &self.params,
                    &self.params_typed,
                    input_map,
                    &self.inputs_typed,
                    MlxMode::Lazy,
                    self.active_extent,
                    Some(&self.gpu_handles),
                    self.current_rng(),
                )
            }
        }
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
        let _guard = crate::sync::runtime_guard();
        self.sync_pending_inner();
        self.maybe_disable_compile_for_graph_size();

        // Build a name→data map by zipping positional inputs against
        // the captured input_names. Anything beyond what was supplied
        // falls through to handles.
        let mut input_map: HashMap<String, Vec<f32>> = self.handles.clone();
        for (i, &data) in inputs.iter().enumerate() {
            if let Some(name) = self.input_names.get(i) {
                input_map.insert(name.clone(), data.to_vec());
            }
        }

        let lowered = if self.use_compiled() {
            self.run_compiled(&input_map)
        } else {
            lower::lower_and_run_typed(
                &self.graph,
                &self.params,
                &self.params_typed,
                &input_map,
                &self.inputs_typed,
                if self.compile_disabled.is_some() {
                    MlxMode::Lazy
                } else {
                    self.mode
                },
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
        let _guard = crate::sync::runtime_guard();
        // Drain any prior in-flight work so we don't accumulate.
        self.sync_pending_inner();
        let mut input_map: HashMap<String, Vec<f32>> = self.handles.clone();
        for &(name, data) in inputs {
            input_map.insert(name.to_string(), data.to_vec());
        }

        if self.use_compiled() {
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
        let _guard = crate::sync::runtime_guard();
        self.sync_pending_inner();
    }

    fn sync_pending_inner(&mut self) {
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

    /// Upload `data` once and keep the MLX array as a graph input across runs.
    pub fn bind_gpu_handle(&mut self, name: &str, data: &[f32]) -> Result<(), MlxError> {
        let shape = self.input_shape_for_name(name)?;
        let data = lower::broadcast_leaf_data(name, data, &shape)?;
        let arr = Array::from_f32_slice(&data, &shape, DType::F32)?;
        self.gpu_handles.insert(name.to_string(), arr);
        Ok(())
    }

    pub fn has_gpu_handle(&self, name: &str) -> bool {
        self.gpu_handles.contains_key(name)
    }

    /// After each [`run`] / [`run_feed_gpu`], copy `outputs[out_idx]` into `handle_name`.
    pub fn set_gpu_handle_feed(&mut self, handle_name: &str, output_index: usize) {
        self.gpu_handle_feeds
            .insert(handle_name.to_string(), output_index);
    }

    /// Read a GPU-resident handle back to host `f32` (eval + sync).
    pub fn read_gpu_handle(&self, name: &str) -> Result<Vec<f32>, MlxError> {
        let arr = self
            .gpu_handles
            .get(name)
            .ok_or_else(|| MlxError(format!("no gpu handle '{name}'")))?;
        arr.to_f32()
    }

    /// Run with host inputs, refresh GPU handle feeds, optional readback of `out_idx`.
    pub fn run_feed_gpu(
        &mut self,
        inputs: &[(&str, &[f32])],
        handle_name: &str,
        output_index: usize,
    ) -> Result<Vec<f32>, MlxError> {
        self.set_gpu_handle_feed(handle_name, output_index);
        let outs = self.run_internal(inputs, true)?;
        outs.into_iter()
            .nth(output_index)
            .ok_or_else(|| MlxError(format!("output index {output_index} missing")))
    }

    fn input_shape_for_name(&self, name: &str) -> Result<Vec<usize>, MlxError> {
        for node in self.graph.nodes() {
            if let rlx_ir::Op::Input { name: n } = &node.op {
                if n == name {
                    return Ok(node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect());
                }
            }
        }
        Err(MlxError(format!("input '{name}' not in graph")))
    }

    fn refresh_gpu_handles_from_outputs(&mut self, outs: &[Array]) -> Result<(), MlxError> {
        for (name, &idx) in &self.gpu_handle_feeds {
            if self.kv_row_feeds.contains_key(name) {
                continue;
            }
            let arr = outs
                .get(idx)
                .ok_or_else(|| MlxError(format!("gpu feed output {idx} missing")))?;
            self.gpu_handles.insert(name.clone(), arr.clone_handle()?);
        }
        Ok(())
    }

    /// Register a resident-KV row feed: [`feed_kv_row`] copies one output row into the handle.
    pub fn register_kv_row_feed(&mut self, handle_name: &str, output_index: usize) {
        self.kv_row_feeds
            .insert(handle_name.to_string(), output_index);
        self.gpu_handle_feeds.remove(handle_name);
    }

    /// Fold the new-token K/V row from the last run into resident GPU handles.
    pub fn feed_kv_row(
        &mut self,
        src_row: usize,
        dst_row: usize,
        row_elems: usize,
    ) -> Result<(), MlxError> {
        if self.kv_row_feeds.is_empty() || self.last_outputs.is_empty() {
            return Ok(());
        }
        let feeds: Vec<(String, usize)> = self
            .kv_row_feeds
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        for (name, out_idx) in feeds {
            let out_arr = self
                .last_outputs
                .get(out_idx)
                .ok_or_else(|| MlxError(format!("kv row feed output {out_idx} missing")))?;
            let mut handle = self
                .gpu_handles
                .get(&name)
                .ok_or_else(|| MlxError(format!("no gpu handle '{name}'")))?
                .clone_handle()?;
            let out_shape = out_arr.shape()?;
            let handle_shape = handle.shape()?;
            let out_data = out_arr.to_f32()?;
            let mut handle_data = handle.to_f32()?;
            let src_start = src_row * row_elems;
            let dst_start = dst_row * row_elems;
            let src_end = src_start + row_elems;
            let dst_end = dst_start + row_elems;
            if src_end > out_data.len() || dst_end > handle_data.len() {
                return Err(MlxError(format!(
                    "feed_kv_row {name}: src_row={src_row} dst_row={dst_row} row_elems={row_elems} \
                     out_len={} handle_len={}",
                    out_data.len(),
                    handle_data.len()
                )));
            }
            handle_data[dst_start..dst_end].copy_from_slice(&out_data[src_start..src_end]);
            handle = Array::from_f32_slice(&handle_data, &handle_shape, DType::F32)?;
            let _ = out_shape;
            self.gpu_handles.insert(name, handle);
        }
        Ok(())
    }

    /// Batch-major past `[B, seq_cap, row_elems]` ← new `[B, 1, row_elems]`.
    pub fn feed_kv_batch_major(
        &mut self,
        dst_row: usize,
        batch: usize,
        seq_cap: usize,
        row_elems: usize,
    ) -> Result<(), MlxError> {
        if batch <= 1 {
            return self.feed_kv_row(0, dst_row, row_elems);
        }
        if self.kv_row_feeds.is_empty() || self.last_outputs.is_empty() {
            return Ok(());
        }
        let feeds: Vec<(String, usize)> = self
            .kv_row_feeds
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        for (name, out_idx) in feeds {
            let out_arr = self
                .last_outputs
                .get(out_idx)
                .ok_or_else(|| MlxError(format!("kv row feed output {out_idx} missing")))?;
            let mut handle = self
                .gpu_handles
                .get(&name)
                .ok_or_else(|| MlxError(format!("no gpu handle '{name}'")))?
                .clone_handle()?;
            let handle_shape = handle.shape()?;
            let out_data = out_arr.to_f32()?;
            let mut handle_data = handle.to_f32()?;
            for b in 0..batch {
                let src_start = b * row_elems;
                let dst_start = b * seq_cap * row_elems + dst_row * row_elems;
                let src_end = src_start + row_elems;
                let dst_end = dst_start + row_elems;
                if src_end > out_data.len() || dst_end > handle_data.len() {
                    return Err(MlxError(format!(
                        "feed_kv_batch_major {name}: b={b} dst_row={dst_row} batch={batch} \
                         seq_cap={seq_cap} row_elems={row_elems} out_len={} handle_len={}",
                        out_data.len(),
                        handle_data.len()
                    )));
                }
                handle_data[dst_start..dst_end].copy_from_slice(&out_data[src_start..src_end]);
            }
            handle = Array::from_f32_slice(&handle_data, &handle_shape, DType::F32)?;
            self.gpu_handles.insert(name, handle);
        }
        Ok(())
    }

    fn run_internal(
        &mut self,
        inputs: &[(&str, &[f32])],
        readback_outputs: bool,
    ) -> Result<Vec<Vec<f32>>, MlxError> {
        let read = if readback_outputs {
            None
        } else {
            Some(&[][..])
        };
        self.run_read_outputs(inputs, read)
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

    /// Clone into an independent executable (recompiles from the stored graph).
    pub fn clone_for_cache(&self) -> Self {
        let gpu_snap: Vec<(String, Vec<f32>)> = self
            .gpu_handles
            .keys()
            .filter_map(|k| self.read_gpu_handle(k).ok().map(|v| (k.clone(), v)))
            .collect();
        let mut exe =
            Self::compile_from_fused_with_rng(self.graph.clone(), self.mode, self.current_rng());
        for (k, v) in &self.params {
            exe.set_param(k, v);
        }
        for (k, v) in &self.handles {
            let _ = exe.bind_handle(k, v);
        }
        for (k, (bytes, dtype)) in &self.params_typed {
            exe.set_param_typed(k, bytes, *dtype);
        }
        for (k, &idx) in &self.gpu_handle_feeds {
            exe.set_gpu_handle_feed(k, idx);
        }
        for (k, &idx) in &self.kv_row_feeds {
            exe.register_kv_row_feed(k, idx);
        }
        for (k, v) in gpu_snap {
            let _ = exe.bind_gpu_handle(&k, &v);
        }
        exe.set_active_extent(self.active_extent);
        exe
    }
}

/// Read `RLX_MLX_MODE=eager|lazy|compiled` (case-insensitive) and
/// pick a default. `compiled` (default) enables persistent
/// `mlx::compile` trace caching; `eager` evals after every op
/// (debug-friendly); `lazy` evals once per run without compile.
/// Compiled-mode graphs that contain host-eval ops (e.g. GGUF
/// DequantMatMul) auto-fall back to Lazy on first run with a
/// warning — see `MlxExecutable::compile_disabled_reason`.
/// Oversized graphs also fall back via [`compile_max_nodes_from_env`].
fn mode_from_env() -> MlxMode {
    crate::config::runtime_config().mode
}

/// `RLX_MLX_COMPILE_MAX_NODES`: max nodes for Compiled (default 1536).
/// `0` disables the limit (force Compiled even on huge graphs).
fn compile_max_nodes_from_env() -> Option<usize> {
    crate::config::runtime_config().compile_max_nodes
}
