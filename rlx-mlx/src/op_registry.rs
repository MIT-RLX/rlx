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

//! Per-backend (MLX) kernel registry for `Op::Custom`.
//!
//! Companion to [`rlx_ir::op_registry`] (IR-level: shape inference +
//! autodiff) and `rlx_cpu::op_registry` (CPU execution). This
//! module is the **API surface** downstream packages register
//! MLX-side custom kernels against.
//!
//! ## Status: end-to-end dispatch wired
//!
//! All three pieces are in place:
//!   - ✅ `Custom` is whitelisted in `MLX_SUPPORTED_OPS`.
//!   - ✅ Lowering arm in `rlx-mlx/src/lower.rs::lower_with_env`
//!     resolves the registered `MlxKernel` by name and calls its
//!     `execute` method with the input `Array` refs (mapped from
//!     IR `NodeId`s via the lowering env).
//!   - ✅ The kernel's returned `Array` becomes the env entry for
//!     this `Op::Custom` node, so consumers downstream see it as
//!     just another lazy operand.
//!
//! ## Performance characterization
//!
//! Each `Op::Custom` triggers an `Array::to_bytes` on each input
//! (forces evaluation up to that point) plus an `Array::from_bytes`
//! for the output. The eval is ~µs of overhead plus the kernel
//! body. For sparse-LU / CG / matvec sized at PDE workloads, the
//! kernel body dominates.
//!
//! See `rlx-runtime/tests/mlx_sparse_ops.rs` for the end-to-end
//! test (sparse-LU + sparse-matvec from `rlx-sparse`, running on
//! `Device::Mlx`, results bit-exact against `Device::Cpu`).
//!
//! ## v1 trait signature: MLX-native `Array` handles
//!
//! Unlike Metal's host-byte-shaped trait, MLX's natural unit is the
//! lazy `Array`. Custom kernels for MLX would typically compose
//! existing `Array` operations (matmul, add, exp, …) into their
//! algorithm — staying inside MLX's lazy graph so MLX's optimizer
//! sees the whole DAG. That's what `mlx::core::compile`-friendly
//! kernels look like.
//!
//! For escape hatches into actual GPU code there's
//! `mlx::fast::metal_kernel` (raw MSL, dispatched by MLX). Wrapping
//! that is a future trait extension.

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use rlx_ir::Shape;

use crate::array::{Array, MlxError};

/// Trait an MLX-side kernel implements for one custom op.
///
/// **v1 contract**: take a slice of input `Array` handles (already
/// mapped from the IR's input `NodeId`s by the lowering pass),
/// produce a fresh `Array` of the requested output shape. The kernel
/// composes MLX `Array` ops to build its result; the resulting
/// `Array` becomes the lazy MLX-graph node for this `Op::Custom`.
pub trait MlxKernel: Send + Sync {
    fn name(&self) -> &str;

    fn execute(
        &self,
        inputs: &[&Array],
        output_shape: &Shape,
        attrs: &[u8],
    ) -> Result<Array, MlxError>;
}

pub struct MlxKernelRegistry {
    kernels: RwLock<HashMap<String, Arc<dyn MlxKernel>>>,
}

impl MlxKernelRegistry {
    pub fn new() -> Self {
        Self {
            kernels: RwLock::new(HashMap::new()),
        }
    }

    pub fn register(&self, k: Arc<dyn MlxKernel>) {
        let name = k.name().to_string();
        let mut g = self.kernels.write().unwrap();
        if g.contains_key(&name) {
            eprintln!(
                "rlx-mlx: MlxKernel '{name}' was already registered — \
                 replacing the previous entry"
            );
        }
        g.insert(name, k);
    }

    pub fn lookup(&self, name: &str) -> Option<Arc<dyn MlxKernel>> {
        self.kernels.read().unwrap().get(name).cloned()
    }
}

impl Default for MlxKernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub fn global_mlx_kernels() -> &'static MlxKernelRegistry {
    static R: OnceLock<MlxKernelRegistry> = OnceLock::new();
    R.get_or_init(MlxKernelRegistry::new)
}

pub fn register_mlx_kernel(k: Arc<dyn MlxKernel>) {
    global_mlx_kernels().register(k);
}

pub fn lookup_mlx_kernel(name: &str) -> Option<Arc<dyn MlxKernel>> {
    global_mlx_kernels().lookup(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::DType;

    struct StubKernel;
    impl MlxKernel for StubKernel {
        fn name(&self) -> &str {
            "stub.mlx"
        }
        fn execute(
            &self,
            inputs: &[&Array],
            _output_shape: &Shape,
            _attrs: &[u8],
        ) -> Result<Array, MlxError> {
            // The simplest possible implementation: clone the first
            // input. Real kernels would compose `Array` ops (matmul,
            // add, …) to build the output.
            inputs[0].clone_handle()
        }
    }

    #[test]
    fn register_and_lookup_round_trips() {
        let reg = MlxKernelRegistry::new();
        reg.register(Arc::new(StubKernel));
        let k = reg
            .lookup("stub.mlx")
            .expect("registered kernel must be findable");
        assert_eq!(k.name(), "stub.mlx");
    }

    #[test]
    fn execute_signature_compiles_and_runs() {
        let k: Arc<dyn MlxKernel> = Arc::new(StubKernel);
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input = Array::from_f32_slice(&data, &[4], DType::F32).expect("input array");
        let out_shape = Shape::new(&[4], DType::F32);
        let result = k
            .execute(&[&input], &out_shape, &[])
            .expect("stub kernel must succeed");
        let result_data = result.to_f32().expect("readback");
        assert_eq!(result_data, data, "stub clones input — values must match");
    }
}
