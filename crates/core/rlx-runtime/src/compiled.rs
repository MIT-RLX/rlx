// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compiled graph — the hot-path execution object.

use crate::backend::ExecutableGraph;
use rlx_driver::Device;
use std::collections::HashSet;

/// Param-invariant "prepare" stage attached to a compiled graph.
///
/// Holds a separately-compiled weight-only graph (`prepare`) whose outputs are
/// the boundary tensors of the main graph. Run once (lazily, on the first
/// forward / `finalize_params`), its outputs are bound into the main graph via
/// [`CompiledGraph::bind_handle`] so weight-derived compute happens once instead
/// of every forward. See [`rlx_compile::split_param_invariant`].
struct Staging {
    prepare: Box<CompiledGraph>,
    prepared: bool,
    /// Boundary input names on the main graph, in `prepare`-output order.
    boundary: Vec<String>,
    prepare_params: HashSet<String>,
    main_params: HashSet<String>,
    /// True once the prepared outputs were injected via persistent `bind_handle`
    /// (CPU). When false (backends without persistent handles, e.g. CUDA), the
    /// prepared values are fed as ordinary inputs each forward from `values`.
    bound: bool,
    /// Prepared boundary values, kept only for the feed-each-forward fallback.
    values: Vec<Vec<f32>>,
}

/// A compiled graph ready for execution.
///
/// Created by [`crate::Session::compile`]. Holds the fused + memory-planned
/// graph and all pre-allocated execution state. Call
/// [`CompiledGraph::run`] repeatedly with different inputs — zero
/// allocation per call.
pub struct CompiledGraph {
    inner: Box<dyn ExecutableGraph>,
    device: Device,
    /// Optional param-invariant prepare stage (None = ordinary graph).
    staging: Option<Box<Staging>>,
}

impl Clone for CompiledGraph {
    /// Deep-clones the underlying executable via `ExecutableGraph::clone_box`.
    /// Backends that don't support cloning will panic at this point.
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone_box(),
            device: self.device,
            staging: self.staging.as_ref().map(|s| {
                Box::new(Staging {
                    prepare: s.prepare.clone(),
                    prepared: s.prepared,
                    boundary: s.boundary.clone(),
                    prepare_params: s.prepare_params.clone(),
                    main_params: s.main_params.clone(),
                    bound: s.bound,
                    values: s.values.clone(),
                })
            }),
        }
    }
}

impl CompiledGraph {
    pub(crate) fn new(inner: Box<dyn ExecutableGraph>, device: Device) -> Self {
        Self {
            inner,
            device,
            staging: None,
        }
    }

    /// Attach a param-invariant `prepare` stage to a compiled MAIN graph.
    /// `prepare` and `boundary` come from [`rlx_compile::split_param_invariant`].
    pub(crate) fn with_staging(
        mut self,
        prepare: CompiledGraph,
        boundary: Vec<String>,
        prepare_params: HashSet<String>,
        main_params: HashSet<String>,
    ) -> Self {
        self.staging = Some(Box::new(Staging {
            prepare: Box::new(prepare),
            prepared: false,
            boundary,
            prepare_params,
            main_params,
            bound: false,
            values: Vec::new(),
        }));
        self
    }

    /// Run the `prepare` stage once (lazily) and make its outputs available to
    /// the main graph — via persistent `bind_handle` when supported (CPU), else
    /// kept for feed-each-forward (CUDA). No-op after the first call.
    fn ensure_prepared(&mut self) {
        let Some(st) = self.staging.as_mut() else {
            return;
        };
        if st.prepared {
            return;
        }
        st.prepare.finalize_params();
        let outs = st.prepare.run(&[]);
        // Try persistent binding; it succeeds only when every boundary binds.
        let mut all_bound = !outs.is_empty();
        for (name, out) in st.boundary.iter().zip(outs.iter()) {
            if !self.inner.bind_handle(name, out) {
                all_bound = false;
            }
        }
        st.bound = all_bound;
        st.values = if all_bound { Vec::new() } else { outs };
        st.prepared = true;
    }

    /// Which device this graph runs on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Set a named parameter (model weight).
    /// Call once per parameter after compilation.
    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        if let Some(st) = self.staging.as_mut() {
            if st.prepare_params.contains(name) {
                st.prepare.set_param(name, data);
                st.prepared = false; // weight changed → re-run prepare next forward
            }
            if st.main_params.contains(name) {
                self.inner.set_param(name, data);
            }
        } else {
            self.inner.set_param(name, data);
        }
    }

    /// Execute the graph with named inputs.
    /// Returns one `Vec<f32>` per graph output (copies from arena).
    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.ensure_prepared();
        match self.staging.as_ref() {
            Some(st) if !st.bound => {
                let mut merged: Vec<(&str, &[f32])> =
                    Vec::with_capacity(inputs.len() + st.boundary.len());
                merged.extend_from_slice(inputs);
                for (name, vals) in st.boundary.iter().zip(st.values.iter()) {
                    merged.push((name.as_str(), vals.as_slice()));
                }
                self.inner.run(&merged)
            }
            _ => self.inner.run(inputs),
        }
    }

    /// Async WebGPU execution on wasm (non-blocking GPU→CPU readback).
    /// WebGL and CPU backends fall back to synchronous [`Self::run`].
    #[cfg(target_arch = "wasm32")]
    pub async fn run_async(&mut self, inputs: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>, String> {
        match self.device {
            Device::WebGpu | Device::Gpu => self.inner.wgpu_run_async(inputs).await,
            _ => Ok(self.run(inputs)),
        }
    }

    /// Run and read back only selected outputs (logits-only decode on MLX).
    pub fn run_read_outputs(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        self.ensure_prepared();
        match self.staging.as_ref() {
            Some(st) if !st.bound => {
                let mut merged: Vec<(&str, &[f32])> =
                    Vec::with_capacity(inputs.len() + st.boundary.len());
                merged.extend_from_slice(inputs);
                for (name, vals) in st.boundary.iter().zip(st.values.iter()) {
                    merged.push((name.as_str(), vals.as_slice()));
                }
                self.inner.run_read_outputs(&merged, read_indices)
            }
            _ => self.inner.run_read_outputs(inputs, read_indices),
        }
    }

    /// Read one row from a row-major output tensor after a forward pass.
    pub fn read_output_row(
        &self,
        out_idx: usize,
        row: usize,
        row_inner: usize,
    ) -> Option<Vec<f32>> {
        self.inner.read_output_row(out_idx, row, row_inner)
    }

    /// Execute and return raw pointers to output data (zero-copy).
    /// Data is valid until the next `run`/`run_raw` call.
    ///
    /// # Safety
    /// The returned pointers point into the arena. Do not use after
    /// the next call to run/run_raw (arena data will be overwritten).
    pub fn run_raw(&mut self, inputs: &[(&str, &[f32])]) -> Vec<(*const f32, usize)> {
        self.ensure_prepared();
        self.inner.run_raw(inputs)
    }

    /// Fastest execution: inputs by slot index (order matches graph input declaration).
    /// Returns output (offset, len) pairs. Read data via `arena_ptr().add(offset)`.
    /// Zero HashMap lookup, zero Vec allocation, zero name matching.
    pub fn run_slots(&mut self, inputs: &[&[f32]]) -> &[(usize, usize)] {
        self.ensure_prepared();
        self.inner.run_slots(inputs)
    }

    /// Arena pointer for reading output data after `run_slots`.
    pub fn arena_ptr(&self) -> *const u8 {
        self.inner.arena_ptr()
    }

    /// Bind a persistent buffer (KV-cache, optimizer state, etc.).
    /// Stays alive across `run()` calls; the backend uses it as the
    /// graph input with the matching name.
    /// Returns true if the backend supports persistent handles.
    pub fn bind_handle(&mut self, name: &str, data: &[f32]) -> bool {
        self.inner.bind_handle(name, data)
    }

    /// Read the current contents of a persistent buffer.
    pub fn read_handle(&self, name: &str) -> Option<Vec<f32>> {
        self.inner.read_handle(name)
    }

    /// GPU-resident MLX input (no-op on non-MLX backends).
    pub fn bind_gpu_handle(&mut self, name: &str, data: &[f32]) -> bool {
        self.inner.bind_gpu_handle(name, data)
    }

    pub fn has_gpu_handle(&self, name: &str) -> bool {
        self.inner.has_gpu_handle(name)
    }

    pub fn set_gpu_handle_feed(&mut self, handle_name: &str, output_index: usize) -> bool {
        self.inner.set_gpu_handle_feed(handle_name, output_index)
    }

    pub fn read_gpu_handle(&self, name: &str) -> Option<Vec<f32>> {
        self.inner.read_gpu_handle(name)
    }

    /// Read one row from a resident GPU input handle without full-tensor D2H.
    pub fn read_gpu_handle_row(
        &self,
        name: &str,
        row: usize,
        row_inner: usize,
    ) -> Option<Vec<f32>> {
        self.inner.read_gpu_handle_row(name, row, row_inner)
    }

    /// Register a targeted row feed for resident KV decode (graphs that emit the
    /// new token at the last bucket-padded output row). No-op (false) on
    /// backends without GPU-resident handle support. See [`Self::feed_kv_row`].
    pub fn register_kv_row_feed(&mut self, handle_name: &str, output_index: usize) -> bool {
        self.inner.register_kv_row_feed(handle_name, output_index)
    }

    /// Fold each registered row feed's new-token row (`src_row` of its output)
    /// into the resident handle slot at `dst_row` (`row_elems` = kv_dim),
    /// in-place on device. Returns false when unsupported.
    pub fn feed_kv_row(&mut self, src_row: usize, dst_row: usize, row_elems: usize) -> bool {
        self.inner.feed_kv_row(src_row, dst_row, row_elems)
    }

    /// Batch-major resident KV feed (`past[b, dst_row]` ← `new[b, 0]`).
    pub fn feed_kv_batch_major(
        &mut self,
        dst_row: usize,
        batch: usize,
        seq_cap: usize,
        row_elems: usize,
    ) -> bool {
        self.inner
            .feed_kv_batch_major(dst_row, batch, seq_cap, row_elems)
    }

    /// Mark a graph input as device-resident without host staging.
    pub fn prepare_resident_gpu_handle(&mut self, name: &str) -> bool {
        self.inner.prepare_resident_gpu_handle(name)
    }

    /// Upload bound (non-resident) GPU handle mirrors into the arena.
    pub fn stage_bound_gpu_handles_to_arena(&mut self) -> bool {
        self.inner.stage_bound_gpu_handles_to_arena();
        true
    }

    /// D2D copy resident KV rows `[from_row..to_row)` from another compiled graph.
    pub fn seed_resident_kv_prefix_from(
        &mut self,
        src: &CompiledGraph,
        prefix_tokens: usize,
        outgoing_upper: usize,
        kv_dim: usize,
        n_layers: usize,
    ) -> bool {
        if self.device != src.device {
            return false;
        }
        self.inner.seed_resident_kv_prefix_from(
            src.inner.as_ref(),
            prefix_tokens,
            outgoing_upper,
            kv_dim,
            n_layers,
        )
    }

    /// D2D copy resident KV rows `[from_row..to_row)` from another compiled graph.
    pub fn copy_resident_kv_rows_from(
        &mut self,
        src: &CompiledGraph,
        from_row: usize,
        to_row: usize,
        outgoing_upper: usize,
        kv_dim: usize,
        n_layers: usize,
    ) -> bool {
        if self.device != src.device {
            return false;
        }
        self.inner.copy_resident_kv_rows_from(
            src.inner.as_ref(),
            from_row,
            to_row,
            outgoing_upper,
            kv_dim,
            n_layers,
        )
    }

    /// Copy parameter buffers from `src` when layouts match (same device/backend).
    pub fn copy_params_from(&mut self, src: &CompiledGraph) -> bool {
        if self.device != src.device {
            return false;
        }
        self.inner.copy_params_from(src.inner.as_ref())
    }

    /// Share `src`'s packed weight buffer (retain, not copy) instead of uploading
    /// our own, when the weight layout matches EXACTLY — one GPU copy backs both.
    /// Returns false (caller must upload) on device mismatch or differing layout.
    pub fn share_params_from(&mut self, src: &CompiledGraph) -> bool {
        if self.device != src.device {
            return false;
        }
        self.inner.share_params_from(src.inner.as_ref())
    }

    /// Run, refresh GPU handle from output, return that output vector.
    pub fn run_feed_gpu_handle(
        &mut self,
        inputs: &[(&str, &[f32])],
        handle_name: &str,
        output_index: usize,
    ) -> Option<Vec<f32>> {
        self.inner
            .run_feed_gpu_handle(inputs, handle_name, output_index)
    }

    /// Hint subsequent `run` calls to process only the first `actual`
    /// rows along the bucket axis (out of `upper`, the compile extent).
    /// Backends that support per-kernel active-extent dispatch honor
    /// this; others ignore it. Pass `None` to clear.
    ///
    /// See `BucketedCompileCache::run_padded` for the canonical caller.
    pub fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        #[cfg(feature = "cpu")]
        if let Some((actual, _)) = extent {
            crate::onnx_active::set_active_token_count(Some(actual))
        }
        self.inner.set_active_extent(extent);
    }

    /// TIDE merged MoE placement (`mask[expert]` device-resident if any layer has it).
    pub fn set_moe_resident_experts(&mut self, mask: &[bool]) {
        self.inner.set_moe_resident_experts(mask);
    }

    /// Per MoE layer placement (forward order). Preferred on CPU over merged mask.
    pub fn set_moe_resident_experts_per_layer(&mut self, masks: &[&[bool]]) {
        self.inner.set_moe_resident_experts_per_layer(masks);
    }

    /// Capture MoE router TopK on next forward (CPU). Returns false if unsupported.
    pub fn enable_moe_topk_capture(&mut self, num_experts: usize) -> bool {
        self.inner.enable_moe_topk_capture(num_experts)
    }

    /// Per-layer expert indices from the last forward (MoE router TopK order).
    pub fn take_moe_topk_capture(&mut self) -> Option<Vec<Vec<u32>>> {
        self.inner.take_moe_topk_capture()
    }

    /// GroupedMatMul GPU/CPU token accounting from the last forward (CPU).
    pub fn take_moe_residency_stats(&mut self) -> Option<crate::MoeResidencyStats> {
        self.inner.take_moe_residency_stats()
    }

    // ── Pipelined / async execution (Phase C) ─────────────────────────

    /// Encode + commit a forward pass without waiting for the device.
    ///
    /// Outputs of intermediate calls are stomped — use `run_pipelined`
    /// when you need each call's outputs back. Pair with `sync_pending`
    /// to drain. CPU is synchronous, so this falls back to `run`.
    pub fn commit_no_wait(&mut self, inputs: &[(&str, &[f32])]) {
        self.inner.commit_no_wait(inputs);
    }

    /// Wait for every command queued by `commit_no_wait`. CPU is a no-op.
    pub fn sync_pending(&mut self) {
        self.inner.sync_pending();
    }

    /// Pipelined batch run. Issues one commit per input set, syncs once
    /// at the end. On Metal, each commit gets its own output snapshot
    /// (allocated + blit-copied), so subsequent commits stomping the
    /// shared arena don't corrupt earlier runs' outputs.
    /// Returns `out[run_idx][output_idx][element_idx]`.
    pub fn run_pipelined(&mut self, input_sets: &[Vec<(&str, &[f32])>]) -> Vec<Vec<Vec<f32>>> {
        self.inner.run_pipelined(input_sets)
    }

    /// Set a named parameter from raw bytes in the given dtype. The
    /// backend handles the widen-to-f32 (or zero-widen, when supported
    /// natively) on the way in. Lets callers feed F16/BF16 weights
    /// without a host-side cast.
    pub fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        if let Some(st) = self.staging.as_mut() {
            if st.prepare_params.contains(name) {
                st.prepare.set_param_typed(name, data, dtype);
                st.prepared = false;
            }
            if st.main_params.contains(name) {
                self.inner.set_param_typed(name, data, dtype);
            }
        } else {
            self.inner.set_param_typed(name, data, dtype);
        }
    }

    /// Finish param upload — warms backend caches when supported.
    pub fn finalize_params(&mut self) {
        self.ensure_prepared();
        self.inner.finalize_params();
    }

    /// Execute with typed inputs and return outputs in their declared
    /// graph dtype, byte-encoded. Mirrors the wgpu / MLX zero-widen
    /// semantics on f32-arena backends (CPU + Metal) by widening at
    /// the boundary.
    pub fn run_typed(
        &mut self,
        inputs: &[(&str, &[u8], rlx_ir::DType)],
    ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
        self.ensure_prepared();
        self.inner.run_typed(inputs)
    }

    /// Override RNG policy for in-graph random ops without recompiling.
    pub fn set_rng(&mut self, rng: rlx_ir::RngOptions) {
        self.inner.set_rng(rng);
    }

    /// Current RNG compile/execute policy.
    pub fn rng(&self) -> rlx_ir::RngOptions {
        self.inner.rng()
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    #[cfg(feature = "cpu")]
    fn end_to_end_session() {
        let mut g = Graph::new("matmul_bias_gelu");
        let x = g.input("x", Shape::new(&[2, 4], DType::F32));
        let w = g.param("w", Shape::new(&[4, 3], DType::F32));
        let b = g.param("b", Shape::new(&[3], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[2, 3], DType::F32));
        let add = g.binary(op::BinaryOp::Add, mm, b, Shape::new(&[2, 3], DType::F32));
        let out = g.activation(op::Activation::Gelu, add, Shape::new(&[2, 3], DType::F32));
        g.set_outputs(vec![out]);

        // Compile
        let session = Session::new(Device::Cpu);
        let mut compiled = session.compile(g);

        // Set weights
        // w = identity-ish [4, 3]: first 3 rows are I, last row is 0
        compiled.set_param(
            "w",
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        );
        compiled.set_param("b", &[0.5, -0.5, 0.0]);

        // Run
        let x_data = vec![
            1.0, 0.0, 0.0, 0.0, // row 0: [1,0,0,0] @ w = [1,0,0] + bias = [1.5,-0.5,0]
            0.0, 1.0, 0.0, 0.0, // row 1: [0,1,0,0] @ w = [0,1,0] + bias = [0.5, 0.5,0]
        ];
        let outputs = compiled.run(&[("x", &x_data)]);

        assert_eq!(outputs.len(), 1);
        let result = &outputs[0];
        assert_eq!(result.len(), 6); // [2, 3]

        // gelu(1.5) ≈ 1.399, gelu(-0.5) ≈ -0.154, gelu(0) = 0
        assert!(
            (result[0] - 1.399).abs() < 0.01,
            "gelu(1.5) = {}",
            result[0]
        );
        assert!(
            (result[1] - -0.154).abs() < 0.01,
            "gelu(-0.5) = {}",
            result[1]
        );
        assert!((result[2]).abs() < 0.01, "gelu(0) = {}", result[2]);

        // gelu(0.5) ≈ 0.346, gelu(0.5) ≈ 0.346, gelu(0) = 0
        assert!(
            (result[3] - 0.346).abs() < 0.01,
            "gelu(0.5) = {}",
            result[3]
        );
        assert!(
            (result[4] - 0.346).abs() < 0.01,
            "gelu(0.5) = {}",
            result[4]
        );

        // Run again with different input — zero allocation
        let x2 = vec![0.0f32; 8];
        let outputs2 = compiled.run(&[("x", &x2)]);
        // All zeros input → gelu(bias) for each output
        let r2 = &outputs2[0];
        assert!((r2[0] - 0.346).abs() < 0.01, "gelu(0.5) = {}", r2[0]); // gelu(0+0.5)
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn device_display() {
        use crate::device_ext::is_available;
        assert!(format!("{}", Device::Cpu).starts_with("CPU"));
        assert!(is_available(Device::Cpu));
        // Backend availability is feature-gated; only assert
        // unavailable when the corresponding feature is off.
        #[cfg(not(feature = "gpu"))]
        assert!(!is_available(Device::Gpu));
        #[cfg(not(feature = "cuda"))]
        assert!(!is_available(Device::Cuda));
        #[cfg(not(feature = "rocm"))]
        assert!(!is_available(Device::Rocm));
        #[cfg(not(feature = "tpu"))]
        assert!(!is_available(Device::Tpu));
    }
}
