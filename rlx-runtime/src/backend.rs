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
use std::collections::HashMap;

// ── Typed I/O helpers (shared across f32-arena backends) ────────────────

/// Widen a typed byte buffer to `Vec<f32>`. Used by `set_param_typed` /
/// `run_typed` overrides on backends whose internal arena is f32-uniform
/// (CPU, Metal, wgpu) so callers can hand in F16/BF16 without doing the
/// host-side cast themselves. Panics on dtypes the f32 arena can't carry.
#[allow(dead_code)]
pub(crate) fn widen_bytes_to_f32(data: &[u8], dtype: rlx_ir::DType) -> Vec<f32> {
    use rlx_ir::DType;
    match dtype {
        DType::F32 => {
            let n = data.len() / 4;
            let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
            s.to_vec()
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
        other => panic!(
            "widen_bytes_to_f32: dtype {other:?} unsupported on f32-arena backends \
             (only F32/F16/BF16 are accepted on the host I/O surface)"
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

/// A compiled, ready-to-execute graph on a specific backend.
pub trait ExecutableGraph: Send {
    /// Set a named parameter (weight) buffer.
    fn set_param(&mut self, name: &str, data: &[f32]);

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

// ── Convenience helpers preserved from older API ──────────────────────
//
// These let existing call sites keep working unchanged while the new
// trait is the canonical one. We provide free functions rather than
// trait methods so adding them doesn't grow the trait surface.

/// Compile at default options (F32, no policy).
pub fn compile(backend: &dyn Backend, graph: Graph) -> Box<dyn ExecutableGraph> {
    backend.compile(graph, &CompileOptions::default())
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
    match policy {
        Some(p) => {
            use rlx_opt::pass::Pass;
            rlx_opt::AutoMixedPrecision::new(p).run(graph)
        }
        None => graph,
    }
}

// ── CPU Backend ─────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
pub mod cpu_backend {
    use super::*;
    use rlx_cpu::{arena::Arena, thunk};
    use rlx_ir::{DType, NodeId, Op};
    use rlx_opt::{fusion, memory, pass::Pass};

    // Arena typed read/write helpers live in `crate::arena` so every
    // backend (CPU, Metal, future CUDA/wgpu/WASM) shares one implementation.
    use rlx_driver::arena::{read_typed_to_f32, write_typed_from_f32};

    pub struct CpuBackend;

    /// PLAN L4: ops the CPU backend can lower today. Includes
    /// DotGeneral (lowered via `LowerDotGeneral` pass) and
    /// ElementwiseRegion (lowered natively per L2). Excludes
    /// FusedTransformerLayer / If / While — those have IR variants
    /// but no CPU lowering yet (see `compile_thunks` arm absence +
    /// `subgraph.rs` "If/While executor wiring is pending" note).
    const CPU_SUPPORTED_OPS: &[rlx_ir::OpKind] = {
        use rlx_ir::OpKind::*;
        &[
            Input,
            Param,
            Constant,
            Activation,
            Cast,
            Binary,
            Compare,
            Where,
            ElementwiseRegion,
            MatMul,
            DotGeneral,
            DenseSolve,
            BatchedDenseSolve,
            Scan,
            ScanBackward,
            ScanBackwardXs,
            LayerNorm,
            RmsNorm,
            Attention,
            Rope,
            Reshape,
            Transpose,
            Narrow,
            Concat,
            Expand,
            Gather,
            Reduce,
            Softmax,
            Cumsum,
            TopK,
            Sample,
            Conv,
            Pool,
            GroupedMatMul,
            ScatterAdd,
            LoraMatMul,
            DequantMatMul,
            SelectiveScan,
            GatedDeltaNet,
            FusedSwiGLU,
            FusedMatMulBiasAct,
            FusedResidualLN,
            FusedAttentionBlock,
            // Backward ops emitted by `rlx_opt::autodiff::grad_with_loss`.
            // Their thunks live in rlx-cpu/src/thunk.rs alongside the
            // forward kernels; without these entries the legalize step
            // below would reject any compiled gradient graph.
            ReluBackward,
            ActivationBackward,
            FakeQuantize,
            FakeQuantizeBackward,
            MaxPool2dBackward,
            Conv2dBackwardInput,
            Conv2dBackwardWeight,
            SoftmaxCrossEntropyWithLogits,
            SoftmaxCrossEntropyBackward,
            LayerNormBackwardInput,
            LayerNormBackwardGamma,
            // User-registered custom ops dispatched through
            // `rlx_cpu::op_registry`. Lowering panics with a clear
            // message if the named CPU kernel isn't registered.
            Custom,
            // User-defined sub-graph with optional override AD rules
            // (JAX-shaped custom_vjp / custom_jvp). Body is a regular
            // Graph compiled recursively in compile_thunks.
            CustomFn,
            // FFT primitive (1D last-axis, 2N real-block layout, f64
            // power-of-2 sizes). Other backends panic at lowering;
            // pin FFT-containing graphs to Device::Cpu for now.
            Fft,
            // C64 Wirtinger AD surface. ComplexNormSq is the canonical
            // real-valued loss for complex inputs; Conjugate is emitted
            // by the new Wirtinger VJP rules for BinaryOp::Mul/Div on
            // C64. Both have CPU thunks in rlx-cpu.
            ComplexNormSq,
            ComplexNormSqBackward,
            Conjugate,
        ]
    };

    impl Backend for CpuBackend {
        fn supported_ops(&self) -> &'static [rlx_ir::OpKind] {
            CPU_SUPPORTED_OPS
        }

        fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
            use rlx_opt::pass::Pass as _;
            // Lower Op::If / Op::While to primitives BEFORE legalize
            // so the supported-op check doesn't reject them — the CPU
            // backend has no native sub-graph executor; this rewrite
            // makes If/While invisible to the rest of the pipeline.
            // No-op when neither op is in the graph.
            let graph = rlx_opt::LowerControlFlow.run(graph);
            // PLAN L4: legalize against the backend's claimed op set
            // BEFORE running fusion (so the diagnostic points at the
            // user's IR, not at a fused-away node).
            if let Err(errors) = rlx_opt::legalize_for_backend(&graph, CPU_SUPPORTED_OPS) {
                panic!("{}", rlx_opt::format_legalize_error("cpu", &errors));
            }
            let policy = options.policy.clone();
            let _precision = options.precision;
            let cfg = rlx_cpu::config::RuntimeConfig::global();

            // Optional preliminary cleanup passes
            let graph = if options.dce {
                rlx_opt::DeadCodeElimination.run(graph)
            } else {
                graph
            };
            let graph = if options.constant_folding {
                rlx_opt::ConstantFolding.run(graph)
            } else {
                graph
            };

            // Run fusion passes.
            // FuseAttentionBlock runs first on the raw graph (before other fusions
            // rewrite the pattern it looks for). It auto-checks batch*seq and is
            // a no-op for large batches.
            let passes: Vec<&dyn Pass> = vec![
                // LowerDotGeneral runs FIRST so any DotGeneral becomes a
                // MatMul (or stays put when the pattern isn't canonical),
                // letting the rest of the fusion pipeline see only MatMuls.
                &rlx_opt::LowerDotGeneral,
                &fusion::FuseAttentionBlock,
                &fusion::FuseMatMulBiasAct,
                &fusion::FuseResidualLN,
                &fusion::FuseSharedInputMatMul,
                // FuseSwiGLU runs AFTER FuseSharedInputMatMul because it
                // matches the narrow×2 + silu + mul pattern that pass
                // produces.
                &fusion::FuseSwiGLU,
                // PLAN L2: collapse maximal element-wise chains AFTER the
                // big-pattern fusions so we only catch leftover ops the
                // pattern fusions didn't claim.
                &rlx_opt::MarkElementwiseRegions,
                // CPU backend doesn't yet have an ElementwiseRegion thunk;
                // run the inverse pass to break any regions back into
                // primitive Activation/Cast/Binary/Compare/Where ops the
                // thunk lowering does handle. Cheap no-op when no regions
                // were marked.
                &rlx_opt::UnfuseElementwiseRegions,
            ];
            let verbose = cfg.verbose >= 1;
            let fused = rlx_opt::run_passes(graph, &passes, verbose);

            // Apply precision policy AFTER fusion — Cast nodes don't disrupt
            // the now-flattened fused ops.
            let fused = match policy {
                Some(p) => rlx_opt::AutoMixedPrecision::new(p).run(fused),
                None => fused,
            };

            // Plan memory with configured alignment
            let plan = memory::plan_memory_aligned(&fused, cfg.arena_alignment);
            if cfg.verbose >= 1 {
                eprintln!(
                    "[rlx] arena: {} bytes, {} buffers, alignment: {}",
                    plan.arena_size,
                    plan.assignments.len(),
                    cfg.arena_alignment
                );
            }
            let mut arena = Arena::from_plan(plan);

            // Pre-compute name → NodeId maps + dtype map (so set_param /
            // run() can cast f32 ↔ F16/BF16 when AutoMixedPrecision has
            // rewritten a Param/Input node away from F32).
            let mut input_ids = HashMap::new();
            let mut param_ids = HashMap::new();
            let mut node_dtypes: HashMap<NodeId, DType> = HashMap::new();
            for node in fused.nodes() {
                node_dtypes.insert(node.id, node.shape.dtype());
                match &node.op {
                    Op::Input { name } => {
                        input_ids.insert(name.clone(), node.id);
                    }
                    Op::Param { name } => {
                        param_ids.insert(name.clone(), node.id);
                    }
                    _ => {}
                }
            }

            // Compile thunk schedule and strip Nop entries
            let schedule = thunk::compile_thunks(&fused, &arena);
            // Don't strip_nops — compiled_fns already filters them out

            // Pre-resolve input slots (name → byte offset, max_elems, dtype)
            let mut input_slots = Vec::new();
            for node in fused.nodes() {
                if let Op::Input { name } = &node.op {
                    let off = arena.byte_offset(node.id);
                    let len = node.shape.num_elements().unwrap_or(0);
                    input_slots.push((name.clone(), off, len, node.shape.dtype()));
                }
            }

            // Pre-resolve output (byte_offset, num_elements) — dtype is
            // already available via node_dtypes for typed reads.
            let output_slots: Vec<(usize, usize)> = fused
                .outputs
                .iter()
                .map(|&id| {
                    let off = arena.byte_offset(id);
                    let len = fused.node(id).shape.num_elements().unwrap_or(0);
                    (off, len)
                })
                .collect();

            // Initialize Constant nodes' arena slots with their literal data.
            // (Without this, constant buffers stay zero — silent miscompute.)
            // Dispatch on dtype: f64 constants must NOT be reinterpreted as
            // f32 (silent garbage). The byte width is whatever the dtype's
            // size says.
            for node in fused.nodes() {
                if let Op::Constant { data } = &node.op
                    && arena.has_buffer(node.id)
                    && !data.is_empty()
                {
                    match node.shape.dtype() {
                        DType::F64 => {
                            let off = arena.byte_offset(node.id);
                            let buf = arena.raw_buf_mut();
                            let n = buf.len().saturating_sub(off).min(data.len());
                            buf[off..off + n].copy_from_slice(&data[..n]);
                        }
                        // Default: f32-aliased arena. Existing F32 + half-
                        // precision rewrite paths land here.
                        _ => {
                            let buf = arena.slice_mut(node.id);
                            let n_floats = data.len() / 4;
                            let n = buf.len().min(n_floats);
                            for i in 0..n {
                                let bytes = [
                                    data[i * 4],
                                    data[i * 4 + 1],
                                    data[i * 4 + 2],
                                    data[i * 4 + 3],
                                ];
                                buf[i] = f32::from_le_bytes(bytes);
                            }
                        }
                    }
                }
            }

            Box::new(CpuExecutable {
                graph: fused,
                arena,
                params: HashMap::new(),
                input_ids,
                param_ids,
                node_dtypes,
                schedule,
                input_slots,
                output_slots,
                handles: HashMap::new(),
                active_extent: None,
            })
        }
    }

    #[derive(Clone)]
    struct CpuExecutable {
        graph: Graph,
        arena: Arena,
        params: HashMap<String, Vec<f32>>,
        input_ids: HashMap<String, NodeId>,
        param_ids: HashMap<String, NodeId>,
        /// Per-node arena dtype. Lets set_param/run cast f32 ↔ F16/BF16
        /// when AutoMixedPrecision has rewritten the graph.
        node_dtypes: HashMap<NodeId, DType>,
        schedule: thunk::ThunkSchedule,
        // Pre-resolved: ordered list of (input_name, arena_byte_offset, max_elems, dtype)
        input_slots: Vec<(String, usize, usize, DType)>,
        /// Output (byte_offset, num_elements). dtype is in node_dtypes.
        output_slots: Vec<(usize, usize)>,
        /// Persistent buffer handles (KV-cache, optimizer state, etc.).
        /// Lives outside the arena and survives across run() calls.
        /// On run(): if a handle's name matches a graph input, the
        /// handle's data is used as the input.
        handles: HashMap<String, Vec<f32>>,
        /// Active-extent hint (`Some((actual, upper))`) for L1 bucketed
        /// dispatch. When set AND every thunk in the schedule is in
        /// `Thunk::safe_for_active_extent`, the executor processes only
        /// `actual / upper` of each kernel's work. Otherwise (or when
        /// `None`) runs at the full compiled extent. See PLAN L1.
        active_extent: Option<(usize, usize)>,
    }

    unsafe impl Send for CpuExecutable {}

    impl CpuExecutable {
        /// Write a f32 input slice into the arena, casting to the node's dtype.
        fn write_input(&mut self, id: NodeId, data: &[f32]) {
            let dtype = self.node_dtypes.get(&id).copied().unwrap_or(DType::F32);
            let off = self.arena.byte_offset(id);
            let buf = self.arena.raw_buf_mut();
            let elem_size = dtype.size_bytes();
            let max_elems = (buf.len() - off) / elem_size;
            unsafe {
                write_typed_from_f32(buf.as_mut_ptr().add(off), dtype, data, max_elems);
            }
        }

        /// Read a node's arena bytes back as Vec<f32>, casting from its dtype.
        fn read_output(&self, id: NodeId) -> Vec<f32> {
            let dtype = self.node_dtypes.get(&id).copied().unwrap_or(DType::F32);
            let off = self.arena.byte_offset(id);
            let buf = self.arena.raw_buf();
            let n_elems = self.graph.node(id).shape.num_elements().unwrap_or(0);
            unsafe { read_typed_to_f32(buf.as_ptr().add(off), dtype, n_elems) }
        }
    }

    impl ExecutableGraph for CpuExecutable {
        fn clone_box(&self) -> Box<dyn ExecutableGraph> {
            Box::new(self.clone())
        }
        fn set_param(&mut self, name: &str, data: &[f32]) {
            // Write directly into the arena — zero per-call lookup for params.
            // Cast f32 → arena dtype when the param has been rewritten to F16/BF16.
            if let Some(&id) = self.param_ids.get(name)
                && self.arena.has_buffer(id)
            {
                let dtype = self.node_dtypes.get(&id).copied().unwrap_or(DType::F32);
                let off = self.arena.byte_offset(id);
                let buf = self.arena.raw_buf_mut();
                let elem_size = dtype.size_bytes();
                let max_elems = (buf.len() - off) / elem_size;
                unsafe {
                    write_typed_from_f32(buf.as_mut_ptr().add(off), dtype, data, max_elems);
                }
                return;
            }
            // Fallback: store in HashMap if no arena slot
            self.params.insert(name.to_string(), data.to_vec());
        }

        fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
            // 1. Apply persistent handles first — they act like default inputs.
            //    Explicit `inputs` passed to run() override matching handle names.
            let handle_names: Vec<String> = self.handles.keys().cloned().collect();
            for name in &handle_names {
                if let Some(&id) = self.input_ids.get(name)
                    && self.arena.has_buffer(id)
                {
                    let data = self.handles.get(name).cloned().unwrap_or_default();
                    self.write_input(id, &data);
                }
            }
            // 2. Explicit per-call inputs override handles.
            for &(name, data) in inputs {
                if let Some(&id) = self.input_ids.get(name)
                    && self.arena.has_buffer(id)
                {
                    self.write_input(id, data);
                }
            }

            // Active-extent fast-path (PLAN L1): if hinted AND every thunk
            // in the schedule supports it, run scaled. Otherwise fall back
            // to full-extent dispatch — preserves correctness when the
            // schedule contains a thunk that hasn't yet been wired in.
            let active_used = if let Some((actual, upper)) = self.active_extent {
                thunk::execute_thunks_active(
                    &self.schedule,
                    self.arena.raw_buf_mut(),
                    actual,
                    upper,
                )
            } else {
                false
            };
            if !active_used {
                // Execute via pre-compiled thunks (zero per-node dispatch overhead)
                thunk::execute_thunks(&self.schedule, self.arena.raw_buf_mut());
            }

            // 3. Sync any handle whose name matches a graph OUTPUT —
            //    KV-cache pattern: outputs flow back into the same-named
            //    handle for the next iteration.
            for (idx, &out_id) in self.graph.outputs.iter().enumerate() {
                let name = format!("out{idx}");
                if self.handles.contains_key(&name) {
                    let v = self.read_output(out_id);
                    self.handles.insert(name, v);
                }
            }

            self.graph
                .outputs
                .iter()
                .map(|&out_id| self.read_output(out_id))
                .collect()
        }

        fn run_raw(&mut self, inputs: &[(&str, &[f32])]) -> Vec<(*const f32, usize)> {
            // Copy inputs by name (HashMap lookup), casting to arena dtype.
            for &(name, data) in inputs {
                if let Some(&id) = self.input_ids.get(name)
                    && self.arena.has_buffer(id)
                {
                    self.write_input(id, data);
                }
            }
            thunk::execute_thunks(&self.schedule, self.arena.raw_buf_mut());
            // Note: pointers are raw arena bytes — for F16 outputs, callers
            // must read 2 bytes/elem, not 4. run() is the safe path for
            // mixed precision; run_raw() is only meaningful for F32.
            self.graph
                .outputs
                .iter()
                .map(|&out_id| {
                    let (ptr, len) = self.arena.raw_ptr(out_id);
                    (ptr as *const f32, len)
                })
                .collect()
        }

        /// Fastest path: inputs by index (matching input_slots order), zero-copy output.
        /// No HashMap, no name matching, no Vec allocation. Casts f32 input
        /// to F16/BF16 if the input slot's dtype was rewritten.
        fn run_slots(&mut self, inputs: &[&[f32]]) -> &[(usize, usize)] {
            let buf = self.arena.raw_buf_mut();
            for (i, &data) in inputs.iter().enumerate() {
                if i < self.input_slots.len() {
                    let (_, off, max_len, dtype) = &self.input_slots[i];
                    unsafe {
                        write_typed_from_f32(buf.as_mut_ptr().add(*off), *dtype, data, *max_len);
                    }
                }
            }
            thunk::execute_thunks(&self.schedule, self.arena.raw_buf_mut());
            &self.output_slots
        }

        fn arena_ptr(&self) -> *const u8 {
            self.arena.raw_buf_mut_ptr()
        }

        fn bind_handle(&mut self, name: &str, data: &[f32]) -> bool {
            // Persistent buffer: stored separately from arena, survives run().
            // If the name matches a graph input, run() will use this data
            // as the input. If the graph also writes back to this name (via
            // an output binding pattern), read_handle returns the latest.
            self.handles.insert(name.to_string(), data.to_vec());
            true
        }

        fn read_handle(&self, name: &str) -> Option<Vec<f32>> {
            self.handles.get(name).cloned()
        }

        fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
            self.active_extent = extent;
        }

        /// Typed param upload. F32 / F16 / BF16 go through the existing
        /// widen-to-f32 path (the CPU arena is historically f32 with
        /// optional half-precision rewrite). F64 (and any future
        /// non-widenable dtype) lands directly in the arena as bytes —
        /// the f32 path would lose precision.
        fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
            if dtype == DType::F64 {
                self.set_param_bytes(name, data, dtype);
                return;
            }
            // U8 / I8 raw byte tensors: opaque storage for the GGUF
            // K-quant `Op::DequantMatMul` path (weights stay packed
            // in the arena). One arena byte = one element.
            if matches!(dtype, DType::U8 | DType::I8) {
                self.set_param_bytes(name, data, dtype);
                return;
            }
            if dtype == DType::F32 {
                let n = data.len() / 4;
                let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
                self.set_param(name, s);
            } else {
                let f32_buf = super::widen_bytes_to_f32(data, dtype);
                self.set_param(name, &f32_buf);
            }
        }

        /// Typed run with mixed-dtype inputs/outputs.
        ///
        /// For each input: if its declared graph dtype matches the
        /// caller's bytes, we write directly into the arena (zero
        /// precision loss — F64 stays F64). For F32 with a half-precision
        /// arena rewrite, we widen as before. F16/BF16 callers go
        /// through the existing widen path.
        ///
        /// Outputs are read straight from the arena in the graph node's
        /// declared dtype — F64 outputs come back as 8 bytes/element,
        /// F32 as 4, etc.
        fn run_typed(
            &mut self,
            inputs: &[(&str, &[u8], rlx_ir::DType)],
        ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
            // Decide: are *all* inputs F64? If so, use the direct-byte
            // path for everything and skip the f32 widening machinery
            // entirely. Mixed dtype graphs (F32 + F64) take the
            // per-input dispatch route below.
            let all_f64 = !inputs.is_empty() && inputs.iter().all(|(_, _, dt)| *dt == DType::F64);

            if all_f64 {
                for (name, data, _) in inputs {
                    if let Some(&id) = self.input_ids.get(*name) {
                        if !self.arena.has_buffer(id) {
                            continue;
                        }
                        let off = self.arena.byte_offset(id);
                        let buf = self.arena.raw_buf_mut();
                        let n = data.len();
                        debug_assert!(
                            off + n <= buf.len(),
                            "run_typed: input '{name}' overflows arena slot"
                        );
                        buf[off..off + n].copy_from_slice(data);
                    }
                }
                thunk::execute_thunks(&self.schedule, self.arena.raw_buf_mut());
            } else {
                // Mixed-dtype path: dtypes that survive untouched
                // through the f32-aliased arena (F64, I32, I64, U32)
                // go in as bytes; F32 and the half-precision family
                // route through widen-to-f32 + run.
                let mut f32_owned: Vec<(String, Vec<f32>)> = Vec::new();
                for (name, data, dt) in inputs {
                    let direct = matches!(*dt, DType::F64 | DType::I32 | DType::I64 | DType::U32,);
                    if direct {
                        if let Some(&id) = self.input_ids.get(*name) {
                            if !self.arena.has_buffer(id) {
                                continue;
                            }
                            let off = self.arena.byte_offset(id);
                            let buf = self.arena.raw_buf_mut();
                            buf[off..off + data.len()].copy_from_slice(data);
                        }
                    } else {
                        let v = super::widen_bytes_to_f32(data, *dt);
                        f32_owned.push((name.to_string(), v));
                    }
                }
                let refs: Vec<(&str, &[f32])> = f32_owned
                    .iter()
                    .map(|(n, d)| (n.as_str(), d.as_slice()))
                    .collect();
                let _ = self.run(&refs);
            }

            // Read each output's bytes from the arena in its declared dtype.
            self.graph
                .outputs
                .iter()
                .map(|&id| {
                    let dtype = self.graph.node(id).shape.dtype();
                    let n_elems = self.graph.node(id).shape.num_elements().unwrap_or(0);
                    let n_bytes = n_elems * dtype.size_bytes();
                    let off = self.arena.byte_offset(id);
                    let bytes = self.arena.raw_buf()[off..off + n_bytes].to_vec();
                    (bytes, dtype)
                })
                .collect()
        }
    }

    impl CpuExecutable {
        /// Direct-byte param upload — copies caller's bytes into the
        /// arena slot for the named param without any dtype conversion.
        /// Used by `set_param_typed` for dtypes that f32-widening would
        /// corrupt (F64). Caller is responsible for matching the param's
        /// declared graph dtype.
        fn set_param_bytes(&mut self, name: &str, data: &[u8], _dtype: rlx_ir::DType) {
            if let Some(&id) = self.param_ids.get(name)
                && self.arena.has_buffer(id)
            {
                let off = self.arena.byte_offset(id);
                let buf = self.arena.raw_buf_mut();
                debug_assert!(
                    off + data.len() <= buf.len(),
                    "set_param_bytes: '{name}' would overflow arena slot"
                );
                buf[off..off + data.len()].copy_from_slice(data);
            }
        }
    }
}

// ── Metal Backend ───────────────────────────────────────────────────────

// ── wgpu Backend ────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
pub mod wgpu_backend {
    use super::*;
    use rlx_ir::OpKind;
    use rlx_wgpu::backend::WgpuExecutable;

    pub struct WgpuBackend;

    /// PLAN L4: ops the wgpu backend can lower today. The fused
    /// macro-kernels (FAB, FTL, FusedSwiGLU) get decomposed by
    /// `crate::unfuse::unfuse` upstream — they're listed here too so
    /// graphs that already contain them legalize cleanly. Conv1d/3d
    /// and Pool1d/3d are deferred (Conv2d only).
    const WGPU_SUPPORTED_OPS: &[OpKind] = &[
        OpKind::Input,
        OpKind::Param,
        OpKind::Constant,
        OpKind::Activation,
        OpKind::Cast,
        OpKind::Binary,
        OpKind::Compare,
        OpKind::Where,
        OpKind::ElementwiseRegion,
        OpKind::MatMul,
        OpKind::DotGeneral,
        OpKind::LayerNorm,
        OpKind::RmsNorm,
        OpKind::Attention,
        OpKind::Rope,
        OpKind::Reshape,
        OpKind::Transpose,
        OpKind::Narrow,
        OpKind::Concat,
        OpKind::Expand,
        OpKind::Gather,
        OpKind::Reduce,
        OpKind::Softmax,
        OpKind::Cumsum,
        OpKind::TopK,
        OpKind::Sample,
        OpKind::Conv,
        OpKind::Pool,
        OpKind::GroupedMatMul,
        OpKind::ScatterAdd,
        OpKind::SelectiveScan,
        OpKind::DequantMatMul,
        OpKind::FusedMatMulBiasAct,
        OpKind::FusedResidualLN,
        OpKind::FusedSwiGLU,
        OpKind::FusedAttentionBlock,
        OpKind::FusedTransformerLayer,
        // Native FFT (WGSL radix-2): f32 only, power-of-2 N ≤ 1024.
        // Anything outside that envelope panics at lowering with a
        // "pin to Device::Cpu" hint. No host fallback — WGPU has no
        // unified memory, so silent CPU round-trip would be a hidden
        // performance cliff.
        OpKind::Fft,
        // LoRA, If, While: not yet wired in wgpu — fail loudly.
    ];

    impl Backend for WgpuBackend {
        fn supported_ops(&self) -> &'static [OpKind] {
            WGPU_SUPPORTED_OPS
        }

        fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
            // PLAN L4: legalize against the backend's claimed op set
            // BEFORE running fusion passes (so the diagnostic points
            // at the user's IR, not at a fused-away node).
            if let Err(errors) = rlx_opt::legalize_for_backend(&graph, WGPU_SUPPORTED_OPS) {
                panic!("{}", rlx_opt::format_legalize_error("wgpu", &errors));
            }
            use rlx_opt::fusion;
            use rlx_opt::pass::Pass as _;
            // Cleanup passes upstream of wgpu's pipeline.
            let graph = if options.dce {
                rlx_opt::DeadCodeElimination.run(graph)
            } else {
                graph
            };
            let graph = if options.constant_folding {
                rlx_opt::ConstantFolding.run(graph)
            } else {
                graph
            };
            // ORDER MATTERS: targeted-pattern fusions run BEFORE the
            // catch-all `MarkElementwiseRegions`. Otherwise the region
            // pass swallows the Add / Activation nodes into chains and
            // FuseMatMulBiasAct / FuseResidualLN fail to match the
            // narrower patterns they look for. (Metal pipeline at line
            // ~377 already orders these correctly; wgpu was inverted
            // and silently shipped 13 unfused LayerNorms per BERT
            // forward where 12 should have been FusedResidualLN.)
            let graph = fusion::FuseMatMulBiasAct.run(graph);
            let graph = fusion::FuseResidualLN.run(graph);
            // Then collapse the leftover element-wise ops the targeted
            // fusions didn't claim.
            let graph = rlx_opt::MarkElementwiseRegions.run(graph);
            let graph = match options.policy.clone() {
                Some(p) => rlx_opt::AutoMixedPrecision::new(p).run(graph),
                None => graph,
            };
            Box::new(WgpuExecutableWrapper {
                inner: WgpuExecutable::compile(graph),
            })
        }
    }

    struct WgpuExecutableWrapper {
        inner: WgpuExecutable,
    }

    unsafe impl Send for WgpuExecutableWrapper {}

    impl ExecutableGraph for WgpuExecutableWrapper {
        fn set_param(&mut self, name: &str, data: &[f32]) {
            self.inner.set_param(name, data);
        }
        fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
            self.inner.run(inputs)
        }
        fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
            self.inner.set_active_extent(extent);
        }

        /// Typed param upload: widens F16/BF16 to F32 at the host boundary,
        /// since the wgpu arena is f32-uniform.
        fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
            match dtype {
                rlx_ir::DType::F32 => {
                    let n = data.len() / 4;
                    let f32_slice =
                        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
                    self.inner.set_param(name, f32_slice);
                }
                rlx_ir::DType::F16 => {
                    let n = data.len() / 2;
                    let f16_slice =
                        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const half::f16, n) };
                    let f32: Vec<f32> = f16_slice.iter().map(|h| h.to_f32()).collect();
                    self.inner.set_param(name, &f32);
                }
                rlx_ir::DType::BF16 => {
                    let n = data.len() / 2;
                    let bf16_slice = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, n)
                    };
                    let f32: Vec<f32> = bf16_slice.iter().map(|h| h.to_f32()).collect();
                    self.inner.set_param(name, &f32);
                }
                other => panic!(
                    "rlx-wgpu set_param_typed: dtype {other:?} unsupported \
                                 (F32, F16, BF16 only — wgpu arena is f32-uniform)"
                ),
            }
        }

        /// Typed run: widen each typed input to F32, run, then narrow each
        /// output back to its declared dtype.
        fn run_typed(
            &mut self,
            inputs: &[(&str, &[u8], rlx_ir::DType)],
        ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
            let mut owned: Vec<(String, Vec<f32>)> = Vec::with_capacity(inputs.len());
            for (name, data, dt) in inputs {
                let v: Vec<f32> = match *dt {
                    rlx_ir::DType::F32 => {
                        let n = data.len() / 4;
                        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) }
                            .to_vec()
                    }
                    rlx_ir::DType::F16 => {
                        let n = data.len() / 2;
                        let s = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const half::f16, n)
                        };
                        s.iter().map(|h| h.to_f32()).collect()
                    }
                    rlx_ir::DType::BF16 => {
                        let n = data.len() / 2;
                        let s = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, n)
                        };
                        s.iter().map(|h| h.to_f32()).collect()
                    }
                    other => {
                        panic!("rlx-wgpu run_typed: input '{name}' dtype {other:?} unsupported")
                    }
                };
                owned.push((name.to_string(), v));
            }
            let refs: Vec<(&str, &[f32])> = owned
                .iter()
                .map(|(n, d)| (n.as_str(), d.as_slice()))
                .collect();
            let dtypes = self.inner.output_dtypes();
            let outs = self.inner.run(&refs);
            outs.into_iter()
                .zip(
                    dtypes
                        .into_iter()
                        .chain(std::iter::repeat(rlx_ir::DType::F32)),
                )
                .map(|(v, dt)| (narrow_to_dtype(&v, dt), dt))
                .collect()
        }
    }

    /// Cast every element of a wgpu f32 output buffer down to the
    /// declared output dtype, returning the corresponding byte stream.
    /// The arena keeps every value as f32; declared output dtypes
    /// (Bool, I8, I32, F16, ...) require an exit-time narrowing to be
    /// byte-identical with backends that store the native dtype.
    fn narrow_to_dtype(v: &[f32], dt: rlx_ir::DType) -> Vec<u8> {
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
            // C64 (complex f32 pair) — the wgpu backend's f32 arena
            // doesn't synthesize complex outputs today; this branch
            // only fires if a graph somehow asks for a C64 output and
            // the backend lowered it as 2N real floats. We pass the
            // raw f32 stream straight through; downstream code that
            // wants complex semantics is responsible for re-pairing.
            DType::C64 => {
                let mut bytes = Vec::with_capacity(v.len() * 4);
                for &x in v {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                bytes
            }
        }
    }
}

// ── MLX Backend ─────────────────────────────────────────────────────────

#[cfg(all(feature = "mlx", target_os = "macos"))]
pub mod mlx_backend {
    use super::*;
    use rlx_mlx::MlxExecutable;

    pub struct MlxBackend;

    /// PLAN L4: ops the MLX backend can lower today. MLX has the
    /// widest IR coverage of any GPU backend — handles everything
    /// including If/While via topo unrolling, and lowers
    /// ElementwiseRegion natively via the per-step composition in
    /// rlx-mlx/src/lower.rs (PLAN L2).
    const MLX_SUPPORTED_OPS: &[rlx_ir::OpKind] = {
        use rlx_ir::OpKind::*;
        &[
            Input,
            Param,
            Constant,
            Activation,
            Cast,
            Binary,
            Compare,
            Where,
            ElementwiseRegion,
            MatMul,
            DotGeneral,
            LayerNorm,
            RmsNorm,
            Attention,
            Rope,
            Reshape,
            Transpose,
            Narrow,
            Concat,
            Expand,
            Gather,
            Reduce,
            Softmax,
            Cumsum,
            TopK,
            Sample,
            Conv,
            Pool,
            GroupedMatMul,
            ScatterAdd,
            LoraMatMul,
            DequantMatMul,
            SelectiveScan,
            FusedSwiGLU,
            FusedMatMulBiasAct,
            FusedResidualLN,
            FusedAttentionBlock,
            FusedTransformerLayer,
            If,
            While,
            // Loop-unrolled scan (Op::Scan body is statically unrolled
            // `length` times into MLX ops; mirror of Op::While's
            // bounded-unroll lowering). ScanBackward is the AD
            // companion — handled the same way.
            Scan,
            ScanBackward,
            ScanBackwardXs,
            // Tier 1 autodiff backward ops — lowered as primitive
            // compositions in `rlx-mlx/src/lower.rs`.
            ReluBackward,
            ActivationBackward,
            SoftmaxCrossEntropyWithLogits,
            SoftmaxCrossEntropyBackward,
            LayerNormBackwardInput,
            LayerNormBackwardGamma,
            // Tier 2 — conv backward via `mc::conv_general` with the
            // same parameter-mapping MLX uses inside its built-in vjp.
            // Currently groups=1 only; grouped conv backward will
            // surface as a clear error from `lower.rs`.
            Conv2dBackwardInput,
            Conv2dBackwardWeight,
            // Tier 3 — max-pool backward via slice-strided argmax over
            // pool windows + a per-kernel-slot scatter-add, matching
            // the CPU thunk's "first-hit-wins" tiebreaking.
            MaxPool2dBackward,
            // QAT — `FakeQuantize` (PerBatch + Fixed scale modes;
            // EMA returns a clear error from `lower.rs`) and the
            // `FakeQuantizeBackward` family covering all 4 STE
            // variants. Closes the last gap vs `CPU_SUPPORTED_OPS`.
            FakeQuantize,
            FakeQuantizeBackward,
            // User-registered custom ops dispatched through
            // `rlx_mlx::op_registry`. Lowering looks up the
            // registered `MlxKernel` and calls its `execute` method
            // to produce the lazy MLX `Array` for this node.
            Custom,
            // Op::Fft on MLX: NOT supported. Host-fallback was tried
            // and rejected — MLX's compile callback forbids `eval`,
            // and `Array::to_bytes` requires eval, so we can't
            // materialize/transform/rematerialize inside the lower
            // pass. Pin FFT subgraphs to Device::Cpu (or Device::Metal,
            // which has a working unified-memory host-fallback). Real
            // MLX support needs a native `mlx::fft::fft` FFI shim;
            // tracked in PLAN.md.
        ]
    };

    impl Backend for MlxBackend {
        fn supported_ops(&self) -> &'static [rlx_ir::OpKind] {
            MLX_SUPPORTED_OPS
        }

        fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
            // PLAN L4: legalize against MLX's claimed op set.
            if let Err(errors) = rlx_opt::legalize_for_backend(&graph, MLX_SUPPORTED_OPS) {
                panic!("{}", rlx_opt::format_legalize_error("mlx", &errors));
            }
            use rlx_opt::fusion;
            use rlx_opt::pass::Pass as _;
            // Optional cleanup passes upstream of MLX's pipeline.
            let graph = if options.dce {
                rlx_opt::DeadCodeElimination.run(graph)
            } else {
                graph
            };
            let graph = if options.constant_folding {
                rlx_opt::ConstantFolding.run(graph)
            } else {
                graph
            };

            // Fusion: enable the passes whose output ops we lower in
            // rlx-mlx/src/lower.rs. FuseAttentionBlock is intentionally
            // skipped — its output op (FusedAttnBlock-style monolithic
            // kernel) doesn't have an MLX equivalent yet, and the
            // unfused path through fast::scaled_dot_product_attention
            // already gets us most of the benefit.
            let passes: Vec<&dyn rlx_opt::pass::Pass> = vec![
                // PLAN L2: collapse maximal element-wise chains into
                // Op::ElementwiseRegion — MLX lowers them natively in
                // lower.rs by composing per-step ops::* into the lazy
                // trace. mlx::compile then folds the whole region into
                // a single fused kernel.
                &rlx_opt::MarkElementwiseRegions,
                &rlx_opt::LowerDotGeneral,
                &fusion::FuseMatMulBiasAct,
                &fusion::FuseResidualLN,
                &fusion::FuseSharedInputMatMul,
                &fusion::FuseSwiGLU,
            ];
            let graph = rlx_opt::run_passes(graph, &passes, /*verbose=*/ false);

            // Apply precision policy AFTER fusion — Cast nodes don't
            // disrupt the now-flattened fused ops.
            let graph = match options.policy.clone() {
                Some(p) => rlx_opt::AutoMixedPrecision::new(p).run(graph),
                None => graph,
            };
            // Default to `MlxMode::Compiled` + eager warm-compile so the
            // expensive trace + JIT cost is paid once at `Session::compile`
            // time rather than on every `run()`. Without this, MLX
            // re-lowers and re-traces the full graph on every forward,
            // making it ~10× slower than CPU for SAM ViT-B. Set
            // `RLX_MLX_MODE=lazy`/`eager` to opt out.
            let mode = match std::env::var("RLX_MLX_MODE").ok().as_deref() {
                Some(s) if s.eq_ignore_ascii_case("eager") => rlx_mlx::lower::MlxMode::Eager,
                Some(s) if s.eq_ignore_ascii_case("lazy") => rlx_mlx::lower::MlxMode::Lazy,
                _ => rlx_mlx::lower::MlxMode::Compiled,
            };
            let mut exe = MlxExecutable::compile_with_mode(graph, mode);
            if mode == rlx_mlx::lower::MlxMode::Compiled {
                if let Err(e) = exe.warm_compile() {
                    eprintln!(
                        "[rlx-runtime] MLX warm_compile failed ({e}); first run will pay the trace cost"
                    );
                }
            }
            Box::new(MlxExecutableWrapper { inner: exe })
        }
    }

    struct MlxExecutableWrapper {
        inner: MlxExecutable,
    }

    unsafe impl Send for MlxExecutableWrapper {}

    impl ExecutableGraph for MlxExecutableWrapper {
        fn set_param(&mut self, name: &str, data: &[f32]) {
            self.inner.set_param(name, data);
        }
        fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
            self.inner.run(inputs)
        }
        fn run_slots(&mut self, inputs: &[&[f32]]) -> &[(usize, usize)] {
            self.inner.run_slots(inputs)
        }
        fn arena_ptr(&self) -> *const u8 {
            self.inner.arena_ptr()
        }
        fn commit_no_wait(&mut self, inputs: &[(&str, &[f32])]) {
            self.inner.commit_no_wait(inputs);
        }
        fn sync_pending(&mut self) {
            self.inner.sync_pending();
        }
        fn run_pipelined(&mut self, input_sets: &[Vec<(&str, &[f32])>]) -> Vec<Vec<Vec<f32>>> {
            self.inner.run_pipelined(input_sets)
        }
        fn bind_handle(&mut self, name: &str, data: &[f32]) -> bool {
            self.inner.bind_handle(name, data)
        }
        fn read_handle(&self, name: &str) -> Option<Vec<f32>> {
            self.inner.read_handle(name)
        }
        fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
            self.inner.set_param_typed(name, data, dtype);
        }
        fn run_typed(
            &mut self,
            inputs: &[(&str, &[u8], rlx_ir::DType)],
        ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
            self.inner.run_typed(inputs)
        }
        fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
            self.inner.set_active_extent(extent);
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal_backend {
    use super::*;
    use rlx_metal::backend::MetalExecutable;

    pub struct MetalBackend;

    /// PLAN L4: ops the Metal backend can lower today. Includes
    /// DotGeneral (LowerDotGeneral pass) and ElementwiseRegion
    /// (decomposed by UnfuseElementwiseRegions). Excludes Cumsum,
    /// SelectiveScan, LoraMatMul, DequantMatMul, Sample,
    /// FusedAttentionBlock, FusedTransformerLayer, If, While —
    /// not yet wired in `rlx-metal/src/thunk.rs`'s compile_thunks.
    const METAL_SUPPORTED_OPS: &[rlx_ir::OpKind] = {
        use rlx_ir::OpKind::*;
        &[
            Input,
            Param,
            Constant,
            Activation,
            Cast,
            Binary,
            Compare,
            Where,
            ElementwiseRegion,
            MatMul,
            DotGeneral,
            LayerNorm,
            RmsNorm,
            Attention,
            Rope,
            Reshape,
            Transpose,
            Narrow,
            Concat,
            Expand,
            Gather,
            Reduce,
            Softmax,
            TopK,
            Conv,
            Pool,
            GroupedMatMul,
            ScatterAdd,
            FusedSwiGLU,
            FusedMatMulBiasAct,
            FusedResidualLN,
            // User-registered custom ops dispatched through
            // `rlx_metal::op_registry`. Lowering panics with a clear
            // message if the named MetalKernel isn't registered;
            // executor inserts a sync point + runs the host kernel
            // against the unified-memory arena.
            Custom,
            // Op::Fft is supported via the same host-fallback pattern
            // as Custom: sync the GPU, run rlx-cpu's FFT against the
            // unified-memory arena, restart cmd_buf. A native Metal
            // compute kernel will replace this when a workload makes
            // the sync the bottleneck.
            Fft,
        ]
    };

    impl Backend for MetalBackend {
        fn supported_ops(&self) -> &'static [rlx_ir::OpKind] {
            METAL_SUPPORTED_OPS
        }

        fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
            use rlx_opt::pass::Pass as _;
            // Same If/While → primitive rewrite as the CPU pipeline
            // (Metal also has no native sub-graph executor wired
            // through its thunk schedule).
            let graph = rlx_opt::LowerControlFlow.run(graph);
            // PLAN L4: legalize against Metal's claimed op set.
            if let Err(errors) = rlx_opt::legalize_for_backend(&graph, METAL_SUPPORTED_OPS) {
                panic!("{}", rlx_opt::format_legalize_error("metal", &errors));
            }
            // Optional cleanup passes upstream of Metal's pipeline
            let graph = if options.dce {
                rlx_opt::DeadCodeElimination.run(graph)
            } else {
                graph
            };
            let graph = if options.constant_folding {
                rlx_opt::ConstantFolding.run(graph)
            } else {
                graph
            };

            // Hand the policy to MetalExecutable so the rewrite runs AFTER
            // its internal fusion passes (avoids breaking pattern matchers).
            Box::new(MetalExecutableWrapper {
                inner: MetalExecutable::compile_with_policy(graph, options.policy.clone()),
            })
        }
    }

    struct MetalExecutableWrapper {
        inner: MetalExecutable,
    }

    unsafe impl Send for MetalExecutableWrapper {}

    impl ExecutableGraph for MetalExecutableWrapper {
        fn set_param(&mut self, name: &str, data: &[f32]) {
            self.inner.set_param(name, data);
        }
        fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
            self.inner.run(inputs)
        }
        fn run_slots(&mut self, inputs: &[&[f32]]) -> &[(usize, usize)] {
            self.inner.run_slots(inputs)
        }
        fn arena_ptr(&self) -> *const u8 {
            self.inner.arena_ptr()
        }
        fn commit_no_wait(&mut self, inputs: &[(&str, &[f32])]) {
            self.inner.commit_no_wait(inputs);
        }
        fn sync_pending(&mut self) {
            self.inner.sync_pending();
        }
        fn run_pipelined(&mut self, input_sets: &[Vec<(&str, &[f32])>]) -> Vec<Vec<Vec<f32>>> {
            self.inner.run_pipelined(input_sets)
        }
        fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
            self.inner.set_active_extent(extent);
        }

        /// Typed param upload — accepts F16/BF16 host bytes by widening
        /// to F32 first, then routing through `set_param`. The Metal
        /// arena's `write_from_f32` honors per-node F16 storage when
        /// AutoMixedPrecision rewrote the param.
        fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
            if dtype == rlx_ir::DType::F32 {
                let n = data.len() / 4;
                let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
                self.inner.set_param(name, s);
            } else {
                let f32_buf = super::widen_bytes_to_f32(data, dtype);
                self.inner.set_param(name, &f32_buf);
            }
        }

        /// Typed run. Inputs widen to F32 (existing path; F64 host
        /// inputs through `run_typed` is a separate Metal extension).
        /// Outputs: F64 outputs go through the byte-direct
        /// `output_bytes_per_node` path (no precision loss in the
        /// f32 round-trip); other dtypes keep the f32-narrow path
        /// for backward compatibility with existing AutoMixedPrecision
        /// rewrites.
        fn run_typed(
            &mut self,
            inputs: &[(&str, &[u8], rlx_ir::DType)],
        ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
            let mut owned: Vec<(String, Vec<f32>)> = Vec::with_capacity(inputs.len());
            for (name, data, dt) in inputs {
                let v = super::widen_bytes_to_f32(data, *dt);
                owned.push((name.to_string(), v));
            }
            let refs: Vec<(&str, &[f32])> = owned
                .iter()
                .map(|(n, d)| (n.as_str(), d.as_slice()))
                .collect();
            let dtypes = self.inner.output_dtypes();
            let f32_outs = self.inner.run(&refs);
            let byte_outs = self.inner.output_bytes_per_node();
            f32_outs
                .into_iter()
                .zip(byte_outs.into_iter())
                .zip(
                    dtypes
                        .into_iter()
                        .chain(std::iter::repeat(rlx_ir::DType::F32)),
                )
                .map(|((f32_v, byte_v), dt)| match dt {
                    rlx_ir::DType::F64 => (byte_v, dt),
                    _ => (super::narrow_f32_to_bytes(&f32_v, dt), dt),
                })
                .collect()
        }
    }
}

// ── CUDA Backend ────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub mod cuda_backend {
    use super::*;
    use rlx_cuda::backend::CudaExecutable;

    pub struct CudaBackend;

    /// PLAN L4: ops the CUDA backend can lower today. Excludes
    /// FusedSwiGLU, LoraMatMul, FusedAttentionBlock,
    /// FusedTransformerLayer (no kernel) + If, While (no executor
    /// wiring). DotGeneral via LowerDotGeneral; ElementwiseRegion
    /// lowered natively by an NVRTC interpreted-chain kernel.
    const CUDA_SUPPORTED_OPS: &[rlx_ir::OpKind] = {
        use rlx_ir::OpKind::*;
        &[
            Input,
            Param,
            Constant,
            Activation,
            Cast,
            Binary,
            Compare,
            Where,
            ElementwiseRegion,
            MatMul,
            DotGeneral,
            LayerNorm,
            RmsNorm,
            Attention,
            Rope,
            Reshape,
            Transpose,
            Narrow,
            Concat,
            Expand,
            Gather,
            Reduce,
            Softmax,
            Cumsum,
            TopK,
            Sample,
            Conv,
            Pool,
            GroupedMatMul,
            ScatterAdd,
            DequantMatMul,
            SelectiveScan,
            FusedMatMulBiasAct,
            FusedResidualLN,
        ]
    };

    impl Backend for CudaBackend {
        fn supported_ops(&self) -> &'static [rlx_ir::OpKind] {
            CUDA_SUPPORTED_OPS
        }

        fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
            // PLAN L4: legalize against CUDA's claimed op set.
            if let Err(errors) = rlx_opt::legalize_for_backend(&graph, CUDA_SUPPORTED_OPS) {
                panic!("{}", rlx_opt::format_legalize_error("cuda", &errors));
            }
            use rlx_opt::pass::Pass as _;
            // Match the cleanup passes the wgpu backend runs. CUDA's
            // op coverage is a strict subset of wgpu's today (matmul +
            // element-wise + leaves), so fusion passes that target ops
            // we haven't lowered yet would just leak through to the
            // executable's "op not yet lowered" panic. Skip them here
            // and let users run the unfusion-style rewrite within
            // rlx-cuda once the kernel set catches up.
            let graph = if options.dce {
                rlx_opt::DeadCodeElimination.run(graph)
            } else {
                graph
            };
            let graph = if options.constant_folding {
                rlx_opt::ConstantFolding.run(graph)
            } else {
                graph
            };
            // PLAN L2: collapse maximal element-wise chains into
            // Op::ElementwiseRegion — CUDA lowers them natively via
            // an NVRTC interpreted-chain kernel
            // (`kernels/elementwise_region.cu`). Safe no-op when no
            // chains are eligible.
            let graph = rlx_opt::MarkElementwiseRegions.run(graph);
            let graph = match options.policy.clone() {
                Some(p) => rlx_opt::AutoMixedPrecision::new(p).run(graph),
                None => graph,
            };
            Box::new(CudaExecutableWrapper {
                inner: CudaExecutable::compile(graph),
            })
        }
    }

    struct CudaExecutableWrapper {
        inner: CudaExecutable,
    }

    // CudaExecutable owns CudaContext + CudaSlice handles; cudarc claims
    // they're Send (CudaContext is Arc-wrapped, CudaSlice is logically
    // a device pointer + length). The Backend trait requires Send for
    // the executable; we honor that here.
    unsafe impl Send for CudaExecutableWrapper {}

    impl ExecutableGraph for CudaExecutableWrapper {
        fn set_param(&mut self, name: &str, data: &[f32]) {
            self.inner.set_param(name, data);
        }
        fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
            self.inner.run(inputs)
        }
        fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
            self.inner.set_active_extent(extent);
        }

        /// Typed param upload — widens F16/BF16 host bytes to f32
        /// before routing through `set_param`. CUDA's arena is
        /// f32-uniform; the half-precision matmul tier opts in via
        /// the separate `set_param_half` API.
        fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
            if dtype == rlx_ir::DType::F32 {
                let n = data.len() / 4;
                let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
                self.inner.set_param(name, s);
            } else {
                let f32_buf = super::widen_bytes_to_f32(data, dtype);
                self.inner.set_param(name, &f32_buf);
            }
        }

        /// Typed run — widen each typed input to F32, run, then narrow
        /// each output back to its declared graph dtype.
        fn run_typed(
            &mut self,
            inputs: &[(&str, &[u8], rlx_ir::DType)],
        ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
            let mut owned: Vec<(String, Vec<f32>)> = Vec::with_capacity(inputs.len());
            for (name, data, dt) in inputs {
                let v = super::widen_bytes_to_f32(data, *dt);
                owned.push((name.to_string(), v));
            }
            let refs: Vec<(&str, &[f32])> = owned
                .iter()
                .map(|(n, d)| (n.as_str(), d.as_slice()))
                .collect();
            let dtypes = self.inner.output_dtypes();
            let outs = self.inner.run(&refs);
            outs.into_iter()
                .zip(
                    dtypes
                        .into_iter()
                        .chain(std::iter::repeat(rlx_ir::DType::F32)),
                )
                .map(|(v, dt)| (super::narrow_f32_to_bytes(&v, dt), dt))
                .collect()
        }
    }
}

// ── ROCm Backend ────────────────────────────────────────────────────────

#[cfg(feature = "rocm")]
pub mod rocm_backend {
    use super::*;
    use rlx_rocm::backend::RocmExecutable;

    pub struct RocmBackend;

    /// PLAN L4: ROCm is the sister crate of CUDA; identical Step
    /// enum + dispatch shape → identical claimed op set.
    const ROCM_SUPPORTED_OPS: &[rlx_ir::OpKind] = {
        use rlx_ir::OpKind::*;
        &[
            Input,
            Param,
            Constant,
            Activation,
            Cast,
            Binary,
            Compare,
            Where,
            ElementwiseRegion,
            MatMul,
            DotGeneral,
            LayerNorm,
            RmsNorm,
            Attention,
            Rope,
            Reshape,
            Transpose,
            Narrow,
            Concat,
            Expand,
            Gather,
            Reduce,
            Softmax,
            Cumsum,
            TopK,
            Sample,
            Conv,
            Pool,
            GroupedMatMul,
            ScatterAdd,
            DequantMatMul,
            SelectiveScan,
            FusedMatMulBiasAct,
            FusedResidualLN,
        ]
    };

    impl Backend for RocmBackend {
        fn supported_ops(&self) -> &'static [rlx_ir::OpKind] {
            ROCM_SUPPORTED_OPS
        }

        fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
            // PLAN L4: legalize against ROCm's claimed op set.
            if let Err(errors) = rlx_opt::legalize_for_backend(&graph, ROCM_SUPPORTED_OPS) {
                panic!("{}", rlx_opt::format_legalize_error("rocm", &errors));
            }
            use rlx_opt::pass::Pass as _;
            // Same upstream cleanup as the CUDA backend — ROCm's op
            // coverage matches rlx-cuda's, so the same passes apply.
            let graph = if options.dce {
                rlx_opt::DeadCodeElimination.run(graph)
            } else {
                graph
            };
            let graph = if options.constant_folding {
                rlx_opt::ConstantFolding.run(graph)
            } else {
                graph
            };
            // PLAN L2: collapse maximal element-wise chains into
            // Op::ElementwiseRegion — ROCm lowers them natively via
            // a hipRTC interpreted-chain kernel (`elementwise_region.cu`,
            // shared with rlx-cuda via include_str!). Safe no-op when
            // no chains are eligible.
            let graph = rlx_opt::MarkElementwiseRegions.run(graph);
            let graph = match options.policy.clone() {
                Some(p) => rlx_opt::AutoMixedPrecision::new(p).run(graph),
                None => graph,
            };
            Box::new(RocmExecutableWrapper {
                inner: RocmExecutable::compile(graph),
            })
        }
    }

    struct RocmExecutableWrapper {
        inner: RocmExecutable,
    }

    // Same Send-claim shape as CudaExecutableWrapper. RocmExecutable
    // owns Arc<RocmContext> + HipBuffer handles; the HipRuntime bundle
    // is internally thread-safe per AMD's documentation.
    unsafe impl Send for RocmExecutableWrapper {}

    impl ExecutableGraph for RocmExecutableWrapper {
        fn set_param(&mut self, name: &str, data: &[f32]) {
            self.inner.set_param(name, data);
        }
        fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
            self.inner.run(inputs)
        }
        fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
            self.inner.set_active_extent(extent);
        }

        /// Typed param upload — widens F16/BF16 host bytes to f32
        /// before routing through `set_param`. ROCm's arena is
        /// f32-uniform; the half-precision matmul tier opts in via
        /// the separate `set_param_half` API.
        fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
            if dtype == rlx_ir::DType::F32 {
                let n = data.len() / 4;
                let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
                self.inner.set_param(name, s);
            } else {
                let f32_buf = super::widen_bytes_to_f32(data, dtype);
                self.inner.set_param(name, &f32_buf);
            }
        }

        /// Typed run — widen each typed input to F32, run, then narrow
        /// each output back to its declared graph dtype.
        fn run_typed(
            &mut self,
            inputs: &[(&str, &[u8], rlx_ir::DType)],
        ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
            let mut owned: Vec<(String, Vec<f32>)> = Vec::with_capacity(inputs.len());
            for (name, data, dt) in inputs {
                let v = super::widen_bytes_to_f32(data, *dt);
                owned.push((name.to_string(), v));
            }
            let refs: Vec<(&str, &[f32])> = owned
                .iter()
                .map(|(n, d)| (n.as_str(), d.as_slice()))
                .collect();
            let dtypes = self.inner.output_dtypes();
            let outs = self.inner.run(&refs);
            outs.into_iter()
                .zip(
                    dtypes
                        .into_iter()
                        .chain(std::iter::repeat(rlx_ir::DType::F32)),
                )
                .map(|(v, dt)| (super::narrow_f32_to_bytes(&v, dt), dt))
                .collect()
        }
    }
}

// ── TPU Backend ─────────────────────────────────────────────────────────

#[cfg(feature = "tpu")]
pub mod tpu_backend {
    use super::*;
    use rlx_tpu::TpuExecutable;

    pub struct TpuBackend;

    /// Ops the TPU backend lowers to HLO. Full inference parity with
    /// rlx-cuda / rlx-rocm. Composite ops (FusedSwiGLU /
    /// FusedAttentionBlock / FusedTransformerLayer / LoraMatMul / If /
    /// While) are unfused inside `rlx_tpu::unfuse::unfuse` ahead of
    /// HLO emission, so they don't appear here.
    const TPU_SUPPORTED_OPS: &[rlx_ir::OpKind] = {
        use rlx_ir::OpKind::*;
        &[
            Input,
            Param,
            Constant,
            Activation,
            Cast,
            Binary,
            Compare,
            Where,
            ElementwiseRegion,
            MatMul,
            DotGeneral,
            LayerNorm,
            RmsNorm,
            Attention,
            Rope,
            Reshape,
            Transpose,
            Narrow,
            Concat,
            Expand,
            Gather,
            Reduce,
            Softmax,
            Cumsum,
            TopK,
            Sample,
            Conv,
            Pool,
            GroupedMatMul,
            ScatterAdd,
            DequantMatMul,
            SelectiveScan,
            // Real-INT8 path + fake-quant.
            QMatMul,
            QConv2d,
            Quantize,
            Dequantize,
            FusedMatMulBiasAct,
            FusedResidualLN,
        ]
    };

    impl Backend for TpuBackend {
        fn supported_ops(&self) -> &'static [rlx_ir::OpKind] {
            TPU_SUPPORTED_OPS
        }

        fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
            if let Err(errors) = rlx_opt::legalize_for_backend(&graph, TPU_SUPPORTED_OPS) {
                panic!("{}", rlx_opt::format_legalize_error("tpu", &errors));
            }
            // The TPU's IR-side pass pipeline (DCE, ConstFold,
            // FuseResidualLN, FuseMatMulBiasAct, LegalizeBroadcast,
            // MarkElementwiseRegions) lives inside
            // `TpuExecutable::compile` so the same passes run whether
            // a caller goes through Session or invokes the executable
            // directly. We only do backend-cross-cutting work here:
            // legalization (must precede the pipeline so we panic
            // early on unsupported ops) and AutoMixedPrecision.
            //
            // Default policy on TPU is `AutoMixedBf16`: BF16 is the
            // native compute dtype on TPU silicon and recent GPUs,
            // and XLA's CPU plugin handles it natively too. Callers
            // can opt out by passing an explicit `PrecisionPolicy`
            // (e.g. `AlwaysF32` for accuracy debugging or
            // `AlwaysF16` to match a CUDA workload's choice).
            use rlx_opt::pass::Pass as _;
            let policy = options
                .policy
                .clone()
                .unwrap_or(rlx_opt::PrecisionPolicy::AutoMixedBf16);
            let graph = rlx_opt::AutoMixedPrecision::new(policy).run(graph);
            let _ = options.dce;
            let _ = options.constant_folding;
            Box::new(TpuExecutableWrapper {
                inner: TpuExecutable::compile(graph),
            })
        }
    }

    struct TpuExecutableWrapper {
        inner: TpuExecutable,
    }

    // PJRT clients + buffers are documented as thread-safe per the
    // upstream C API. Same Send-claim shape as CudaExecutableWrapper /
    // RocmExecutableWrapper.
    unsafe impl Send for TpuExecutableWrapper {}

    impl ExecutableGraph for TpuExecutableWrapper {
        fn set_param(&mut self, name: &str, data: &[f32]) {
            self.inner.set_param(name, data);
        }
        fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
            self.inner.run(inputs)
        }

        /// Typed param upload — widens F16/BF16/etc. host bytes to
        /// f32 today. Once the HLO emitter speaks bf16 natively
        /// (which TPUs prefer over f16), the typed path will hand
        /// the original bytes straight through `Buffer_FromHostBuffer`.
        fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
            if dtype == rlx_ir::DType::F32 {
                let n = data.len() / 4;
                let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
                self.inner.set_param(name, s);
            } else {
                let f32_buf = super::widen_bytes_to_f32(data, dtype);
                self.inner.set_param(name, &f32_buf);
            }
        }

        fn run_typed(
            &mut self,
            inputs: &[(&str, &[u8], rlx_ir::DType)],
        ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
            let mut owned: Vec<(String, Vec<f32>)> = Vec::with_capacity(inputs.len());
            for (name, data, dt) in inputs {
                let v = super::widen_bytes_to_f32(data, *dt);
                owned.push((name.to_string(), v));
            }
            let refs: Vec<(&str, &[f32])> = owned
                .iter()
                .map(|(n, d)| (n.as_str(), d.as_slice()))
                .collect();
            let dtypes = self.inner.output_dtypes();
            let outs = self.inner.run(&refs);
            outs.into_iter()
                .zip(
                    dtypes
                        .into_iter()
                        .chain(std::iter::repeat(rlx_ir::DType::F32)),
                )
                .map(|(v, dt)| (super::narrow_f32_to_bytes(&v, dt), dt))
                .collect()
        }
    }
}
