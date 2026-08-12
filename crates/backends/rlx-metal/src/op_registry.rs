// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-backend (Metal) kernel registry for `Op::Custom`.
//!
//! Companion to [`rlx_ir::op_registry`] (IR-level: shape inference +
//! autodiff) and `rlx_cpu::op_registry` (CPU execution). This module
//! is the **API surface** downstream packages register Metal-side
//! custom kernels against.
//!
//! ## Status: end-to-end dispatch wired
//!
//! All three pieces are in place:
//!   - ✅ `Custom` is whitelisted in `SUPPORTED_OPS`.
//!   - ✅ `Thunk::CustomOp` variant + lowering arm in
//!     `rlx-metal/src/thunk.rs::ThunkSchedule::compile`.
//!   - ✅ Executor arm in `backend.rs::encode_commit` flushes the
//!     active MSL encoder, commits + waits the current cmd_buf,
//!     runs `MetalKernel::execute` against the unified-memory arena,
//!     then rebinds cmd_buf to a fresh one for subsequent thunks.
//!
//! The crucial enabler was making the lazy compute encoder
//! `enc: Option<ComputeCommandEncoder>` (owned, refcount-bumped via
//! `to_owned()`) instead of `Option<&ComputeCommandEncoderRef>`
//! (borrowed). The owned form decouples the encoder's lifetime from
//! cmd_buf's, so `enc.take()` fully releases the borrow and cmd_buf
//! is freely reassignable mid-function. See
//! `rlx-runtime/tests/metal_sparse_ops.rs` for the end-to-end test
//! (sparse-LU + sparse-matvec from `rlx-sparse`, running on
//! `Device::Metal`, results bit-exact against `Device::Cpu`).
//!
//! ## Performance characterization
//!
//! Each `Op::Custom` is one Metal queue trip
//! (`wait_until_completed` ≈ 150 µs typical) plus the host
//! kernel's compute time. `Buffer::contents()` is host-accessible
//! at zero cost on Apple Silicon (unified memory), so there's no
//! GPU↔host data copy — only the synchronization point.
//!
//! For ops that compose many GPU dispatches into a single host
//! kernel call (Sparse-LU, FFT, eigensolve), the sync overhead
//! amortizes well. For fine-grained per-element ops, prefer
//! lowering through MSL kernels directly.
//!
//! ## Why a per-backend trait at all?
//!
//! Per-backend kernel registries match how rlx already segregates
//! backend-flavored types (no `MTLBuffer` types reach `rlx-ir`; no
//! Accelerate types reach `rlx-mlx`; etc). The trait identity
//! `MetalKernel` says "this kernel runs on Metal" — distinct from
//! `CpuKernel` ("this kernel runs on CPU") — even when the v1
//! signature happens to look similar.
//!
//! ## v1 trait signature: raw bytes
//!
//! The v1 `execute` method takes inputs/output as raw bytes already
//! copied to host. This is a deliberately-conservative signature:
//!
//!   - **Honest about cost**: a host roundtrip on Metal is slow
//!     (PCIe-equivalent cost over Metal's unified memory bus). Users
//!     who want true GPU performance will subclass to a future
//!     `MetalGpuKernel` trait that exposes `MTLCommandBuffer` /
//!     `MTLBuffer` directly.
//!   - **Compatible with the CpuKernel they probably already wrote**:
//!     a downstream `SparseLuMetal` impl can delegate to the existing
//!     `SparseLuCpu` until a real Metal kernel ships.
//!   - **Zero metal-rs in the trait surface**: keeps rlx-metal's
//!     dependency on `metal-rs` an implementation detail.

#![cfg(all(target_vendor = "apple", not(target_os = "watchos")))]

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use rlx_ir::Shape;

/// Trait a Metal-side kernel implements for one custom op. Registered
/// under the same `name` used in `Op::Custom` and `OpExtension::name`.
///
/// **v1 contract**: receive contiguous host-side bytes per input
/// (already copied off the GPU) and a contiguous host-side mutable
/// byte slice for the output (will be copied back to the GPU). This
/// matches the CPU kernel pattern; performance-critical custom ops
/// will graduate to a future trait that exposes raw MTLBuffer +
/// MTLCommandBuffer once the dispatch path is wired.
pub trait MetalKernel: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;

    fn execute(
        &self,
        inputs: &[(&[u8], &Shape)],
        output: (&mut [u8], &Shape),
        attrs: &[u8],
    ) -> Result<(), String>;
}

pub struct MetalKernelRegistry {
    kernels: RwLock<HashMap<String, Arc<dyn MetalKernel>>>,
}

impl MetalKernelRegistry {
    pub fn new() -> Self {
        Self {
            kernels: RwLock::new(HashMap::new()),
        }
    }

    pub fn register(&self, k: Arc<dyn MetalKernel>) {
        let name = k.name().to_string();
        let mut g = self.kernels.write().unwrap();
        if g.contains_key(&name) {
            eprintln!(
                "rlx-metal: MetalKernel '{name}' was already registered — \
                 replacing the previous entry"
            );
        }
        g.insert(name, k);
    }

    pub fn lookup(&self, name: &str) -> Option<Arc<dyn MetalKernel>> {
        self.kernels.read().unwrap().get(name).cloned()
    }
}

impl Default for MetalKernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub fn global_metal_kernels() -> &'static MetalKernelRegistry {
    static R: OnceLock<MetalKernelRegistry> = OnceLock::new();
    R.get_or_init(MetalKernelRegistry::new)
}

pub fn register_metal_kernel(k: Arc<dyn MetalKernel>) {
    global_metal_kernels().register(k);
}

/// Register rlx-metal's own built-in custom-op kernels exactly once. Run before
/// every custom-op lookup so consumers get them on Metal automatically — no
/// explicit `register()` call or extra cargo feature required (these kernels are
/// host-delegates over unified memory, see each module).
fn ensure_builtins_registered() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        crate::ms_deform_attn::register();
        // Host-delegate collective ops (all_reduce / all_gather / reduce_scatter
        // + Megatron f/g). Pure host/transport ops with no device kernel; they
        // stage to host and delegate to the registered rlx-collectives CPU
        // kernel. See `crate::collective`.
        crate::collective::register();
        // Register rlx-cpu's ONNX reference kernels (ScatterND / NonZero /
        // GatherND / Einsum / Mod / …) so the generic host-delegate below can
        // find them even in a Metal-only run (the CPU thunk-compile path that
        // normally registers them never executes). Idempotent.
        rlx_cpu::onnx_ref::register_onnx_reference_kernels();
        // NB: `llada2_gate` is registered by its consumer (rlx-llada2); do not
        // auto-register it here or it double-registers.
    });
}

/// Generic host-delegate for any `onnx.*` (or other) custom op that has a
/// registered rlx-cpu reference kernel but no native Metal kernel. Stages the
/// operands to host (zero-copy on Apple's unified memory) and runs the CPU
/// reference, dtype-aware. This lets every ONNX-imported graph run on Metal —
/// the heavy compute lowers to MSL, the handful of reference-only indexing ops
/// (Einsum, Mod, ScatterND, …) fall back to the host.
#[derive(Debug)]
struct OnnxHostDelegate {
    name: String,
}

impl MetalKernel for OnnxHostDelegate {
    fn name(&self) -> &str {
        &self.name
    }
    fn execute(
        &self,
        inputs: &[(&[u8], &Shape)],
        output: (&mut [u8], &Shape),
        attrs: &[u8],
    ) -> Result<(), String> {
        if std::env::var("RLX_DBG_CUSTOM").is_ok() {
            eprintln!(
                "[custom] {} in={:?} out={:?}",
                self.name,
                inputs
                    .iter()
                    .map(|(_, s)| (s.dtype(), s.dims().to_vec()))
                    .collect::<Vec<_>>(),
                (output.1.dtype(), output.1.dims().to_vec()),
            );
        }
        rlx_cpu::op_registry::run_custom_op_host(&self.name, inputs, output, attrs)
    }
}

pub fn lookup_metal_kernel(name: &str) -> Option<Arc<dyn MetalKernel>> {
    ensure_builtins_registered();
    if let Some(k) = global_metal_kernels().lookup(name) {
        return Some(k);
    }
    // No native Metal kernel — fall back to the rlx-cpu reference over unified
    // memory if one is registered under this name.
    if rlx_cpu::op_registry::lookup_cpu_kernel(name).is_some() {
        return Some(Arc::new(OnnxHostDelegate {
            name: name.to_string(),
        }));
    }
    None
}

// ── Raw-GPU custom kernels (v2: no host roundtrip) ──────────────────────────

/// Dispatch context handed to a [`MetalGpuKernel`]: the live serial compute
/// encoder on the current command buffer, the unified-memory arena buffer, and
/// per-operand byte offsets + shapes + attribute bytes.
///
/// Bind an operand's sub-buffer with
/// `d.encoder.set_buffer(index, Some(d.arena), off as u64)` (the `off` values
/// are byte offsets into `d.arena`, matching metal-rs's `set_buffer` offset arg).
pub struct MetalGpuDispatch<'a> {
    /// Live compute encoder — encode your dispatch here. Do **not** end it,
    /// commit, or wait; the executor owns command-buffer lifetime.
    pub encoder: &'a crate::mtl::ComputeCommandEncoderRef,
    /// The `StorageModeShared` arena buffer holding every operand.
    pub arena: &'a crate::mtl::BufferRef,
    /// Per-input `(byte offset into arena, element count, shape)`.
    pub inputs: &'a [(usize, u32, Shape)],
    /// Output `(byte offset, element count, shape)`.
    pub output: &'a (usize, u32, Shape),
    /// Per-instance attribute bytes from `Op::Custom.attrs`.
    pub attrs: &'a [u8],
}

/// A raw-GPU Metal custom kernel: dispatches a real MSL kernel onto the active
/// compute encoder with **no host roundtrip and no queue sync** — contrast
/// [`MetalKernel`], which copies operands to host, runs CPU-style, and forces a
/// commit + `wait_until_completed`. Use this for fine-grained ops where the host
/// roundtrip would dominate; for coarse ops (Sparse-LU, FFT) the host-delegate
/// [`MetalKernel`] is fine, and for ops expressible as primitives prefer
/// `OpExtension::lower`.
///
/// Register under the same `name` used in `Op::Custom` / `OpExtension::name`; a
/// registered GPU kernel takes **precedence** over a host-delegate `MetalKernel`
/// of the same name. Compile a `ComputePipelineState` once (e.g. behind a
/// `OnceLock`) via [`crate::metal_device`] +
/// [`crate::pipeline_cache::load_or_compile_library`], then bind + dispatch in
/// `encode`.
pub trait MetalGpuKernel: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;

    /// Encode this op's dispatch into `d.encoder`. Return `Err` to abort the run.
    fn encode(&self, d: &MetalGpuDispatch) -> Result<(), String>;
}

struct MetalGpuKernelRegistry {
    kernels: RwLock<HashMap<String, Arc<dyn MetalGpuKernel>>>,
}

impl MetalGpuKernelRegistry {
    fn new() -> Self {
        Self {
            kernels: RwLock::new(HashMap::new()),
        }
    }

    fn register(&self, k: Arc<dyn MetalGpuKernel>) {
        let name = k.name().to_string();
        let mut g = self.kernels.write().unwrap();
        if g.contains_key(&name) {
            eprintln!(
                "rlx-metal: MetalGpuKernel '{name}' was already registered — \
                 replacing the previous entry"
            );
        }
        g.insert(name, k);
    }

    fn lookup(&self, name: &str) -> Option<Arc<dyn MetalGpuKernel>> {
        self.kernels.read().unwrap().get(name).cloned()
    }
}

fn global_metal_gpu_kernels() -> &'static MetalGpuKernelRegistry {
    static R: OnceLock<MetalGpuKernelRegistry> = OnceLock::new();
    R.get_or_init(MetalGpuKernelRegistry::new)
}

/// Register a raw-GPU Metal custom kernel. Takes precedence over a host-delegate
/// [`MetalKernel`] registered under the same name.
pub fn register_metal_gpu_kernel(k: Arc<dyn MetalGpuKernel>) {
    global_metal_gpu_kernels().register(k);
}

/// Look up a registered raw-GPU Metal custom kernel by name.
pub fn lookup_metal_gpu_kernel(name: &str) -> Option<Arc<dyn MetalGpuKernel>> {
    global_metal_gpu_kernels().lookup(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::DType;

    #[derive(Debug)]
    struct StubKernel;
    impl MetalKernel for StubKernel {
        fn name(&self) -> &str {
            "stub.metal"
        }
        fn execute(
            &self,
            _inputs: &[(&[u8], &Shape)],
            _output: (&mut [u8], &Shape),
            _attrs: &[u8],
        ) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn register_and_lookup_round_trips() {
        let reg = MetalKernelRegistry::new();
        reg.register(Arc::new(StubKernel));
        let k = reg
            .lookup("stub.metal")
            .expect("registered kernel must be findable");
        assert_eq!(k.name(), "stub.metal");
    }

    #[test]
    fn execute_signature_compiles_and_runs() {
        let k: Arc<dyn MetalKernel> = Arc::new(StubKernel);
        let in_shape = Shape::new(&[4], DType::F32);
        let out_shape = Shape::new(&[4], DType::F32);
        let in_bytes = vec![0u8; 16];
        let mut out_bytes = vec![0u8; 16];
        k.execute(&[(&in_bytes, &in_shape)], (&mut out_bytes, &out_shape), &[])
            .expect("stub kernel must succeed");
    }
}
