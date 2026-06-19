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
//!   - ✅ `Custom` is whitelisted in `METAL_SUPPORTED_OPS`.
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

#![cfg(target_os = "macos")]

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

pub fn lookup_metal_kernel(name: &str) -> Option<Arc<dyn MetalKernel>> {
    global_metal_kernels().lookup(name)
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
