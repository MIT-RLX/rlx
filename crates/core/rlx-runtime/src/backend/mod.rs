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

//! Backend trait — abstraction over CPU/GPU/CUDA execution.
//!
//! Each backend implements `Backend::compile(graph, &CompileOptions)` and
//! returns an `ExecutableGraph`. New compile knobs go in `CompileOptions`
//! rather than as new trait methods.

use crate::CompileOptions;
use rlx_ir::Graph;
use rlx_ir::hir::HirModule;
use rlx_ir::lir::LirModule;
use std::collections::HashMap;
use std::sync::Arc;

use crate::cpu_low_precision;

// ── Typed I/O helpers (shared across f32-arena backends) ────────────────

/// Widen (or, for F64, narrow) a typed byte buffer to `Vec<f32>`. Used by
/// `set_param_typed` / `run_typed` overrides on backends whose internal
/// arena is f32-uniform (Vulkan / wgpu / CUDA / ROCm / oneAPI / TPU) so
/// callers can hand in F16/BF16/F64 without doing the host-side cast
/// themselves. F64 is narrowed to f32 to match the f32-uniform arena slot
/// (SPD-manifold ops re-widen to f64 inside their CPU host-fallback, so
/// the eigendecomposition itself stays f64). Panics on dtypes the f32
/// arena can't carry.
#[allow(dead_code)]
pub(crate) fn widen_bytes_to_f32(data: &[u8], dtype: rlx_ir::DType) -> Vec<f32> {
    use rlx_ir::DType;
    // An empty typed input (e.g. an unused KV-cache slot passed as `Vec::new()`)
    // has a dangling, 1-byte-aligned pointer. `slice::from_raw_parts` requires an
    // aligned pointer even for length 0, so reinterpreting it as `*const f32`
    // trips the UB precondition check. Nothing to widen — return empty.
    if data.is_empty() {
        return Vec::new();
    }
    match dtype {
        DType::F32 => {
            let n = data.len() / 4;
            let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
            s.to_vec()
        }
        DType::F64 => {
            let n = data.len() / 8;
            let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, n) };
            s.iter().map(|&x| x as f32).collect()
        }
        DType::F16 => {
            let n = data.len() / 2;
            let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const half::f16, n) };
            s.iter().map(|h| h.to_f32()).collect()
        }
        DType::BF16 => {
            let n = data.len() / 2;
            let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, n) };
            s.iter().map(|h| h.to_f32()).collect()
        }
        // Integer I/O (token ids, durations, masks) widened to f32 for the
        // f32-uniform arena — same convention wgpu already uses. Values are
        // small integers (<2^24), so f32 is exact. This lets TTS/LM graphs
        // that feed I64/I32 inputs run on CUDA/Vulkan (VITS token ids etc.).
        DType::I64 => {
            let n = data.len() / 8;
            let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i64, n) };
            s.iter().map(|&x| x as f32).collect()
        }
        DType::I32 => {
            let n = data.len() / 4;
            let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, n) };
            s.iter().map(|&x| x as f32).collect()
        }
        DType::I8 => data.iter().map(|&b| b as i8 as f32).collect(),
        DType::U8 | DType::Bool => data.iter().map(|&b| b as f32).collect(),
        other => panic!(
            "widen_bytes_to_f32: dtype {other:?} unsupported on f32-arena backends \
             (only F32/F64/F16/BF16/I64/I32/I8/U8/Bool are accepted on the host I/O surface)"
        ),
    }
}

/// Narrow a `&[f32]` buffer down to the declared output dtype, returning
/// the corresponding little-endian byte stream. Mirrors the bytes a
/// backend that stores the native dtype would emit. Used by `run_typed`
/// to keep the byte-level output contract identical across backends.
#[allow(dead_code)]
pub(crate) fn narrow_f32_to_bytes(v: &[f32], dt: rlx_ir::DType) -> Vec<u8> {
    use rlx_ir::DType;
    match dt {
        DType::F32 => {
            let mut bytes = Vec::with_capacity(v.len() * 4);
            for &x in v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            bytes
        }
        DType::F16 => {
            let mut bytes = Vec::with_capacity(v.len() * 2);
            for &x in v {
                bytes.extend_from_slice(&half::f16::from_f32(x).to_le_bytes());
            }
            bytes
        }
        DType::BF16 => {
            let mut bytes = Vec::with_capacity(v.len() * 2);
            for &x in v {
                bytes.extend_from_slice(&half::bf16::from_f32(x).to_le_bytes());
            }
            bytes
        }
        DType::F64 => {
            let mut bytes = Vec::with_capacity(v.len() * 8);
            for &x in v {
                bytes.extend_from_slice(&(x as f64).to_le_bytes());
            }
            bytes
        }
        DType::I8 => v.iter().map(|&x| x as i8 as u8).collect(),
        DType::U8 => v.iter().map(|&x| x as u8).collect(),
        DType::I16 => {
            let mut bytes = Vec::with_capacity(v.len() * 2);
            for &x in v {
                bytes.extend_from_slice(&(x as i16).to_le_bytes());
            }
            bytes
        }
        DType::I32 => {
            let mut bytes = Vec::with_capacity(v.len() * 4);
            for &x in v {
                bytes.extend_from_slice(&(x as i32).to_le_bytes());
            }
            bytes
        }
        DType::U32 => {
            let mut bytes = Vec::with_capacity(v.len() * 4);
            for &x in v {
                bytes.extend_from_slice(&(x as u32).to_le_bytes());
            }
            bytes
        }
        DType::I64 => {
            let mut bytes = Vec::with_capacity(v.len() * 8);
            for &x in v {
                bytes.extend_from_slice(&(x as i64).to_le_bytes());
            }
            bytes
        }
        DType::Bool => v
            .iter()
            .map(|&x| if x != 0.0 { 1u8 } else { 0u8 })
            .collect(),
        DType::C64 => {
            // Complex narrow path: real part = the f32 value, imaginary
            // part = 0. Mirrors how the backend stores narrowed f32
            // operands when promoted to a complex op input.
            let mut bytes = Vec::with_capacity(v.len() * 8);
            for &x in v {
                bytes.extend_from_slice(&x.to_le_bytes());
                bytes.extend_from_slice(&0.0_f32.to_le_bytes());
            }
            bytes
        }
    }
}

#[cfg(test)]
mod f64_boundary_tests {
    use super::{narrow_f32_to_bytes, widen_bytes_to_f32};
    use rlx_ir::DType;

    /// Regression: `widen_bytes_to_f32` used to `panic!` on F64, so uploading
    /// an F64 param/input (e.g. SPDNet's learnable `G`) on an f32-uniform
    /// backend (Vulkan/wgpu/CUDA/ROCm/oneAPI/TPU) crashed. It must now narrow
    /// f64→f32 for the f32 arena slot.
    #[test]
    fn widen_f64_narrows_without_panic() {
        let vals = [1.5f64, -2.25, 3.0, 4.5];
        let bytes: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
        let out = widen_bytes_to_f32(&bytes, DType::F64);
        assert_eq!(out, vec![1.5f32, -2.25, 3.0, 4.5]);
    }

    /// F64 output round-trips: narrow f32→f64-bytes then widen back.
    #[test]
    fn f64_boundary_round_trips() {
        let v = [0.5f32, -1.0, 2.0];
        let bytes = narrow_f32_to_bytes(&v, DType::F64);
        assert_eq!(bytes.len(), v.len() * 8);
        assert_eq!(widen_bytes_to_f32(&bytes, DType::F64), v.to_vec());
    }
}

#[cfg(all(test, feature = "cpu"))]
mod capabilities_tests {
    use super::*;
    use rlx_ir::{DType, Graph, Shape};

    #[test]
    fn cpu_executable_declares_core_capabilities() {
        let mut g = Graph::new("caps");
        let x = g.input("x", Shape::new(&[2], DType::F32));
        g.set_outputs(vec![x]);
        let be = cpu_backend::CpuBackend;
        let exec = be.compile(g, &CompileOptions::default());
        let caps = exec.capabilities();
        assert!(caps.clone);
        assert!(caps.moe);
        assert!(caps.typed_io);
        assert!(caps.active_extent);
        assert!(!caps.gpu_handles);
        assert_eq!(
            caps.enabled_names(),
            vec!["clone", "moe", "typed_io", "active_extent"]
        );
    }
}

/// A compiled, ready-to-execute graph on a specific backend.
///
/// # Method groups
///
/// See [`crate::ExecutableCapabilities`] for the optional feature bits and a
/// table of which trait methods belong to which group. Core execution is
/// `set_param` / `run` (and friends); everything else defaults to unsupported.
pub trait ExecutableGraph: Send {
    /// Advisory optional-feature bitmask. Default: all clear. Override when
    /// you implement MoE / GPU handles / typed I/O / etc. Callers must still
    /// treat individual method return values as authoritative.
    fn capabilities(&self) -> crate::ExecutableCapabilities {
        crate::ExecutableCapabilities::NONE
    }

    /// Set a named parameter (weight) buffer.
    fn set_param(&mut self, name: &str, data: &[f32]);

    /// Called after all params are uploaded (`set_param` / `set_param_typed`).
    /// Backends may warm caches (e.g. Metal QMatMul weight dequant).
    fn finalize_params(&mut self) {}

    /// Deep-clone this executable into a fresh `Box`. Lets
    /// `CompiledGraph` implement `Clone` so callers (e.g. eda-mna's
    /// `SensitivityContext`) can spin up N independent executor
    /// copies for thread-parallel dispatch without paying the full
    /// graph-compile cost N times. Default implementation panics;
    /// backends that support cloning override.
    fn clone_box(&self) -> Box<dyn ExecutableGraph> {
        panic!("clone_box not implemented for this backend");
    }

    /// Execute the graph with named inputs. Returns output data (copies from arena).
    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>>;

    /// Like [`Self::run`] but only read back outputs at `read_indices`.
    /// GPU handle feeds still update for every output. Default: all outputs.
    fn run_read_outputs(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        match read_indices {
            None => self.run(inputs),
            Some(ix) => {
                // Backends without a native partial-read path still run the full
                // graph; only clone the requested outputs on the host.
                let all = self.run(inputs);
                ix.iter().filter_map(|&i| all.get(i).cloned()).collect()
            }
        }
    }

    /// Execute and return raw pointers to output data in arena (zero-copy).
    fn run_raw(&mut self, inputs: &[(&str, &[f32])]) -> Vec<(*const f32, usize)> {
        let vecs = self.run(inputs);
        vecs.iter().map(|v| (v.as_ptr(), v.len())).collect()
    }

    /// Fastest: inputs by slot index, returns output (offset, len) pairs.
    /// Read output from arena via `arena_ptr().add(offset)`.
    fn run_slots(&mut self, _inputs: &[&[f32]]) -> &[(usize, usize)] {
        &[] // default: not supported
    }

    /// Get the raw arena buffer pointer for reading outputs after run_slots.
    fn arena_ptr(&self) -> *const u8 {
        std::ptr::null()
    }

    /// Hint the executor that subsequent `run` calls should process
    /// only the first `actual` rows along the bucket axis (out of
    /// `upper`, the extent the graph was compiled at). Backends that
    /// support per-kernel active-extent dispatch honor this; others
    /// ignore it and process the full compiled extent.
    ///
    /// Pass `None` to clear the hint. The hint is sticky — set it
    /// before each `run` and clear it after, or maintain it across
    /// runs at your discretion.
    ///
    /// Even when honored, callers must not rely on the contents of the
    /// output past `actual` rows — that region may contain stale data
    /// from earlier runs (kernels skip it).
    ///
    /// Default: no-op. See `BucketedCompileCache::run_padded` for the
    /// canonical caller; backends opt in by overriding this method.
    fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        let _ = extent;
    }

    /// Override RNG policy for in-graph random ops without recompiling.
    fn set_rng(&mut self, rng: rlx_ir::RngOptions) {
        let _ = rng;
    }

    /// Current RNG policy (default when the backend does not override).
    fn rng(&self) -> rlx_ir::RngOptions {
        rlx_ir::RngOptions::default()
    }

    /// TIDE merged placement mask (union across MoE layers). CPU: stats + host path.
    fn set_moe_resident_experts(&mut self, _mask: &[bool]) {}

    /// Per MoE layer placement (`masks[layer][expert]`). Preferred over merged on CPU.
    fn set_moe_resident_experts_per_layer(&mut self, _masks: &[&[bool]]) {}

    /// Capture MoE router TopK indices on the next CPU forward (TIDE refresh).
    fn enable_moe_topk_capture(&mut self, _num_experts: usize) -> bool {
        false
    }

    /// Take captured per-layer expert indices (one vec per MoE TopK in order).
    fn take_moe_topk_capture(&mut self) -> Option<Vec<Vec<u32>>> {
        None
    }

    /// MoE GroupedMatMul residency accounting from the last forward (CPU).
    fn take_moe_residency_stats(&mut self) -> Option<crate::MoeResidencyStats> {
        None
    }

    /// Bind a persistent buffer handle (KV-cache, training state, etc.).
    /// The buffer lives across run() calls and is not in the arena.
    /// Returns true if the backend supports persistent handles.
    fn bind_handle(&mut self, _name: &str, _data: &[f32]) -> bool {
        false
    }

    /// Read a persistent buffer's current contents.
    fn read_handle(&self, _name: &str) -> Option<Vec<f32>> {
        None
    }

    /// GPU-resident input (MLX): upload once, reuse across runs.
    fn bind_gpu_handle(&mut self, _name: &str, _data: &[f32]) -> bool {
        false
    }

    fn has_gpu_handle(&self, _name: &str) -> bool {
        false
    }

    fn set_gpu_handle_feed(&mut self, _handle_name: &str, _output_index: usize) -> bool {
        false
    }

    fn read_gpu_handle(&self, _name: &str) -> Option<Vec<f32>> {
        None
    }

    /// Read one row from a resident GPU input handle without full-tensor D2H.
    fn read_gpu_handle_row(&self, _name: &str, _row: usize, _row_inner: usize) -> Option<Vec<f32>> {
        None
    }

    /// Register a targeted *row* feed for resident KV decode (graphs that emit
    /// the new token at the last bucket-padded output row). Returns false when
    /// the backend has no GPU-resident handle support. See [`feed_kv_row`].
    fn register_kv_row_feed(&mut self, _handle_name: &str, _output_index: usize) -> bool {
        false
    }

    /// Fold each registered row feed's new-token row (`src_row` of its output)
    /// into the resident handle slot at `dst_row` (`row_elems` = kv_dim),
    /// in-place on device. Call after a logits-only run. Returns false when
    /// unsupported (caller keeps the host KV path).
    fn feed_kv_row(&mut self, _src_row: usize, _dst_row: usize, _row_elems: usize) -> bool {
        false
    }

    /// Batch-major resident KV: `new` is `[B,1,row_elems]`, past is
    /// `[B,seq_cap,row_elems]`. Writes each batch's new row into
    /// `past[b, dst_row, :]`. Default falls back to contiguous `feed_kv_row`
    /// when `batch == 1`.
    fn feed_kv_batch_major(
        &mut self,
        dst_row: usize,
        batch: usize,
        seq_cap: usize,
        row_elems: usize,
    ) -> bool {
        if batch == 1 {
            return self.feed_kv_row(0, dst_row, row_elems);
        }
        let _ = (dst_row, seq_cap, row_elems);
        false
    }

    /// Mark a graph input as a device-resident handle with no host mirror.
    fn prepare_resident_gpu_handle(&mut self, _name: &str) -> bool {
        false
    }

    /// Upload bound (non-resident) GPU handle mirrors into the arena.
    fn stage_bound_gpu_handles_to_arena(&mut self) {}

    /// D2D seed of resident `past_k_*` / `past_v_*` from another executable's
    /// resident prefix (bucket rollover without host DRAM round-trip).
    fn seed_resident_kv_prefix_from(
        &mut self,
        _src: &dyn ExecutableGraph,
        _prefix_tokens: usize,
        _outgoing_upper: usize,
        _kv_dim: usize,
        _n_layers: usize,
    ) -> bool {
        false
    }

    /// D2D copy resident KV rows `[from_row..to_row)` from another executable.
    fn copy_resident_kv_rows_from(
        &mut self,
        _src: &dyn ExecutableGraph,
        _from_row: usize,
        _to_row: usize,
        _outgoing_upper: usize,
        _kv_dim: usize,
        _n_layers: usize,
    ) -> bool {
        false
    }

    /// Copy named parameter storage from another executable on the same backend.
    /// Used to avoid re-uploading packed U8 weights when compiling decode buckets.
    fn copy_params_from(&mut self, src: &dyn ExecutableGraph) -> bool {
        let _ = src;
        false
    }

    /// Share (retain, don't copy) the packed weight storage of another executable
    /// on the same backend when the weight layout matches EXACTLY — one GPU copy
    /// of the (large, read-only) weights backs both executables, instead of a
    /// per-executable duplicate. Returns false (caller must fall back to a full
    /// upload/copy) when unsupported or the layouts differ. `src` must already be
    /// fully populated. See `rlx_metal::MetalExecutable::share_weights_from`.
    fn share_params_from(&mut self, src: &dyn ExecutableGraph) -> bool {
        let _ = src;
        false
    }

    /// Downcast hook for [`Self::copy_params_from`]. Backends override when supported.
    fn executable_as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }

    /// Mutable downcast hook for [`Self::copy_params_from`].
    fn executable_as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }

    /// CUDA-only: mutable access for device KV seeding. Default `None`.
    #[cfg(feature = "cuda")]
    fn cuda_executable_for_kv_seed(&mut self) -> Option<&mut rlx_cuda::backend::CudaExecutable> {
        let _ = self;
        None
    }

    /// CUDA-only: immutable access for device KV seeding. Default `None`.
    #[cfg(feature = "cuda")]
    fn cuda_executable_for_kv_seed_ref(&self) -> Option<&rlx_cuda::backend::CudaExecutable> {
        None
    }

    /// Read one row from a row-major graph output after `run` / `run_read_outputs`.
    /// Metal reads a single row from the arena; default returns `None` (caller falls back).
    fn read_output_row(&self, _out_idx: usize, _row: usize, _row_inner: usize) -> Option<Vec<f32>> {
        None
    }

    /// Run and refresh a GPU handle from `output_index`; returns that output on host.
    fn run_feed_gpu_handle(
        &mut self,
        inputs: &[(&str, &[f32])],
        _handle_name: &str,
        _output_index: usize,
    ) -> Option<Vec<f32>> {
        let _ = inputs;
        None
    }

    // ── Pipelined / async execution (Phase C) ─────────────────────────
    //
    // These allow callers to amortize per-run sync latency on backends
    // where it matters (Metal: ~150 µs `wait_until_completed` per commit).
    // CPU has no such cost, so the default impls just call `run` serially.

    /// Encode + commit a forward pass without waiting for completion.
    ///
    /// Outputs of intermediate calls are stomped — use `run_pipelined` if
    /// you need outputs from each individual commit. Pair with
    /// `sync_pending` to drain.
    ///
    /// Default: synchronous fallback (calls `run`, discards output). CPU
    /// uses this default since BLAS is synchronous anyway.
    fn commit_no_wait(&mut self, inputs: &[(&str, &[f32])]) {
        let _ = self.run(inputs);
    }

    /// Wait for every command queued by `commit_no_wait`.
    /// Default: no-op (synchronous backends have nothing pending).
    fn sync_pending(&mut self) {}

    /// Issue a batch of forward passes pipelined, returning per-run outputs.
    ///
    /// The Metal impl encodes a per-commit blit so each in-flight run's
    /// outputs survive subsequent commits stomping the shared arena. The
    /// CPU default is just sequential `run`s — equally correct, no perf
    /// penalty (CPU has no GPU sync cost to amortize).
    ///
    /// Returns `out[run_idx][output_idx][element_idx]`.
    fn run_pipelined(&mut self, input_sets: &[Vec<(&str, &[f32])>]) -> Vec<Vec<Vec<f32>>> {
        input_sets.iter().map(|inputs| self.run(inputs)).collect()
    }

    // ── Typed (non-F32) host I/O ──────────────────────────────────
    //
    // `set_param` and `run` are F32 by contract. The typed entry
    // points let callers pass and receive raw bytes in any rlx-ir
    // dtype, avoiding the f32 widen/narrow round-trip that's
    // wasteful for F16/BF16 weights and activations.
    //
    // The default impls only handle F32 — any other dtype panics.
    // Backends that support typed I/O natively (e.g. MLX via
    // Array::from_bytes/to_bytes) override these.

    /// Set a named parameter from raw bytes in the given dtype.
    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        if dtype != rlx_ir::DType::F32 {
            panic!(
                "backend's default set_param_typed only handles F32; \
                    got {dtype:?}. Override on the backend for typed support."
            );
        }
        if !data.len().is_multiple_of(4) {
            panic!(
                "set_param_typed F32: data length {} not a multiple of 4",
                data.len()
            );
        }
        // SAFETY: F32 bytes are 4-aligned by source convention; we
        // only widen access (read &[f32] from owned &[u8]). Failure
        // mode if a caller hands us mis-aligned bytes is undefined,
        // hence the % 4 length check.
        let n = data.len() / 4;
        let f32_slice = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
        self.set_param(name, f32_slice);
    }

    /// Run with typed inputs and typed outputs. Returns
    /// `(bytes, dtype)` per output; the dtype is whatever the
    /// graph's output node was declared as.
    fn run_typed(
        &mut self,
        inputs: &[(&str, &[u8], rlx_ir::DType)],
    ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
        // Default impl: convert each typed input to f32 (F32-only),
        // run, then re-emit outputs as F32 bytes.
        let mut owned: Vec<(String, Vec<f32>)> = Vec::with_capacity(inputs.len());
        for (name, data, dt) in inputs {
            if *dt != rlx_ir::DType::F32 {
                panic!(
                    "backend's default run_typed only handles F32 inputs; \
                        got {dt:?} for input '{name}'"
                );
            }
            if data.len() % 4 != 0 {
                panic!(
                    "run_typed F32 input '{name}': len {} not multiple of 4",
                    data.len()
                );
            }
            let n = data.len() / 4;
            let v: Vec<f32> =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) }.to_vec();
            owned.push((name.to_string(), v));
        }
        let refs: Vec<(&str, &[f32])> = owned
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        let outs = self.run(&refs);
        outs.into_iter()
            .map(|v| {
                let bytes =
                    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
                        .to_vec();
                (bytes, rlx_ir::DType::F32)
            })
            .collect()
    }

    /// True when the lowered wgpu schedule contains host-executed steps
    /// (unsupported on browser WebGPU). Default: false.
    #[cfg(target_arch = "wasm32")]
    fn wgpu_requires_host(&self) -> bool {
        false
    }

    /// Async WebGPU execution for wasm (non-blocking readback). Default: error.
    ///
    /// Futures are **not** `Send` on wasm — WebGPU buffer mapping uses
    /// `Rc`/`RefCell` and browser callbacks that are single-threaded.
    #[cfg(target_arch = "wasm32")]
    fn wgpu_run_async<'a>(
        &'a mut self,
        _inputs: &'a [(&'a str, &'a [f32])],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>, String>> + 'a>>
    {
        Box::pin(async move { Err("wgpu_run_async not implemented for this backend".into()) })
    }
}

/// Backend implementation trait.
///
/// Single compile entry point. New compile-time knobs are added to
/// `CompileOptions`, not as new trait methods.
///
/// `Send + Sync` because backends are stateless factories — multiple
/// threads can call `compile` concurrently. The returned
/// `Box<dyn ExecutableGraph>` is `Send` (moveable to a worker thread)
/// but **not** `Sync` (`run`/`run_slots` take `&mut self`).
pub trait Backend: Send + Sync {
    /// Compile a graph for this backend with the given options.
    fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph>;

    /// Compile pre-optimized LIR (HIR → MIR → LIR pipeline output).
    /// Default re-enters [`Self::compile`] — backends should override
    /// when they can reuse the embedded buffer plan.
    fn compile_lir(&self, lir: LirModule, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        self.compile(lir.into_graph(), options)
    }

    /// HIR-first compile: lower blocks, run fusion pipeline, emit executable.
    fn compile_hir(
        &self,
        hir: HirModule,
        device: rlx_driver::Device,
        options: &CompileOptions,
    ) -> Result<Box<dyn ExecutableGraph>, rlx_ir::hir::LowerError> {
        let result = crate::stages::compile_hir_stages(device, hir, options)?;
        crate::stages::maybe_log_fusion(&result.fusion);
        Ok(self.compile_lir(result.lir, options))
    }

    /// [`GraphModule`] compile — unified HIR/MIR/LIR entry.
    fn compile_module(
        &self,
        module: rlx_ir::GraphModule,
        device: rlx_driver::Device,
        options: &CompileOptions,
    ) -> Result<Box<dyn ExecutableGraph>, rlx_ir::hir::LowerError> {
        let result = crate::stages::compile_module_stages(device, module, options)?;
        crate::stages::maybe_log_fusion(&result.fusion);
        Ok(self.compile_lir(result.lir, options))
    }

    /// PLAN L4: declare which `OpKind`s this backend can lower.
    /// Default: empty slice = "no claim made — accept everything"
    /// (preserves existing behavior; backends opt in by overriding).
    /// When non-empty, the `LegalizeForBackend` pass will refuse to
    /// compile a graph that contains an op outside this set, instead
    /// of silently falling through to slower / wrong dispatch.
    fn supported_ops(&self) -> &'static [rlx_ir::OpKind] {
        &[]
    }
}

/// Prepare a fused MIR graph from LIR for backend executable construction.
/// Skips the fusion pipeline — LIR must come from `compile_*_stages`.
#[allow(dead_code)]
fn prepare_fused_graph(
    graph: Graph,
    options: &CompileOptions,
    supported_ops: &[rlx_ir::OpKind],
    backend_name: &str,
) -> Graph {
    let (mut graph, report) = rlx_opt::prepare_graph_for_backend_with_report(
        graph,
        backend_name,
        supported_ops,
        options.kernel_dispatch,
    );
    rlx_opt::maybe_log_dispatch_report(&report);
    if !report.compile_ready {
        panic!(
            "{}\n{}",
            rlx_opt::format_legalize_error(backend_name, &report.still_unsupported),
            rlx_opt::format_dispatch_report(&report)
        );
    }
    // Backends that claim `OpKind::Scan` keep long Scans for host-fallback;
    // short ones IR-unroll so the body runs as ordinary device ops.
    if supported_ops.contains(&rlx_ir::OpKind::Scan) {
        graph = apply_scan_device_preference(graph, options);
    }
    graph = crate::precompile::post_fusion_cleanup(graph, options);
    if let Some(p) = options.policy.clone() {
        use rlx_opt::pass::Pass as _;
        graph = rlx_opt::AutoMixedPrecision::new(p).run(graph);
    }
    graph
}

/// Short-length + small-budget Scan unroll so GPU backends run the body on-device.
fn apply_scan_device_preference(graph: Graph, options: &CompileOptions) -> Graph {
    let g = rlx_cpu::rlx_maybe_unroll_scans!(graph, options.scan_unroll_max_length);
    rlx_opt::maybe_unroll_scans_budget(g, 4096)
}

#[allow(dead_code)]
fn declared_output_dtypes(
    manifest: &cpu_low_precision::IoDtypeManifest,
    exec_dtypes: Vec<rlx_ir::DType>,
) -> Vec<rlx_ir::DType> {
    exec_dtypes
        .into_iter()
        .enumerate()
        .map(|(i, exec)| manifest.output_dtype(i, exec))
        .collect()
}

// ── Convenience helpers preserved from older API ──────────────────────
//
// These let existing call sites keep working unchanged while the new
// trait is the canonical one. We provide free functions rather than
// trait methods so adding them doesn't grow the trait surface.

/// Compile at default options (F32, no policy).
pub fn compile(backend: &dyn Backend, graph: Graph) -> Box<dyn ExecutableGraph> {
    backend.compile(graph, &CompileOptions::default())
}

/// Compile HIR through the fusion-first pipeline.
pub fn compile_hir(
    backend: &dyn Backend,
    hir: HirModule,
    device: rlx_driver::Device,
    options: &CompileOptions,
) -> Result<Box<dyn ExecutableGraph>, rlx_ir::hir::LowerError> {
    backend.compile_hir(hir, device, options)
}

/// Compile a [`GraphModule`] through the fusion-first pipeline.
pub fn compile_module(
    backend: &dyn Backend,
    module: rlx_ir::GraphModule,
    device: rlx_driver::Device,
    options: &CompileOptions,
) -> Result<Box<dyn ExecutableGraph>, rlx_ir::hir::LowerError> {
    backend.compile_module(module, device, options)
}

/// Compile at a specific precision (default policy = none).
pub fn compile_with_precision(
    backend: &dyn Backend,
    graph: Graph,
    precision: crate::Precision,
) -> Box<dyn ExecutableGraph> {
    backend.compile(graph, &CompileOptions::new().precision(precision))
}

/// Helper retained for backward compatibility — applies the precision
/// rewrite at the runtime layer if backends don't override their
/// pipeline placement. Modern code: pass the policy via CompileOptions
/// and let the backend handle ordering.
fn _legacy_apply_policy(graph: Graph, policy: Option<rlx_opt::PrecisionPolicy>) -> Graph {
    use rlx_opt::pass::Pass as _;
    match policy {
        Some(p) => rlx_opt::AutoMixedPrecision::new(p).run(graph),
        None => graph,
    }
}

// ── CPU Backend ─────────────────────────────────────────────────────────

// Per-device Backend / ExecutableGraph wiring (feature-gated).

#[cfg(feature = "cpu")]
pub mod cpu_backend;

#[cfg(feature = "gpu")]
pub mod wgpu_backend;

#[cfg(feature = "opengl")]
pub mod webgl_backend;

#[cfg(feature = "vulkan")]
pub mod vulkan_backend;

#[cfg(feature = "oneapi")]
pub mod oneapi_backend;

#[cfg(all(feature = "mlx", rlx_mlx_host))]
pub mod mlx_backend;

/// Apple CoreML / Neural Engine (ANE) backend wiring.
///
/// Unlike the GPU backends, CoreML compiles a *static* model with weights
/// baked in, so we hand the raw graph (Param nodes intact) straight to
/// `rlx_coreml` and skip the fusion/LIR arena pipeline.
#[cfg(all(
    feature = "coreml",
    target_vendor = "apple",
    not(target_os = "watchos")
))]
pub mod coreml_backend;

#[cfg(all(feature = "metal", target_vendor = "apple", not(target_os = "watchos")))]
pub mod metal_backend;

#[cfg(feature = "cuda")]
pub mod cuda_backend;

#[cfg(feature = "rocm")]
pub mod rocm_backend;

#[cfg(feature = "tpu")]
pub mod tpu_backend;

/// QNN (Hexagon NPU) backend adapter — wraps the `rlx-qnn` FFI runtime
/// (`Device::Hexagon`). Lowers a general supported subgraph in-process via
/// `libQnnCpu.so` / `libQnnHtp.so`. Claims `FusedAttentionBlock` and
/// decomposes it before FFI lower (same pattern as CoreML / Vulkan).
#[cfg(feature = "qnn")]
pub mod qnn_backend;
