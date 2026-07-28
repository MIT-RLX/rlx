// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
//!   - ✅ `Custom` is whitelisted in `SUPPORTED_OPS`.
//!   - ✅ Lowering arm in `rlx-mlx/src/lower/env.rs::lower_with_env`
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

#![cfg(rlx_mlx_host)]

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
    if let Some(k) = global_mlx_kernels().lookup(name) {
        return Some(k);
    }
    // No native MLX kernel — fall back to the rlx-cpu reference via host
    // byte staging if one is registered under this name (ScatterElements,
    // GatherND, …).
    if rlx_cpu::op_registry::lookup_cpu_kernel(name).is_some() {
        return Some(Arc::new(OnnxHostDelegate {
            name: name.to_string(),
        }));
    }
    None
}

/// Generic host-delegate for any custom op with a registered rlx-cpu kernel
/// but no native MLX kernel. Stages `Array`s to host bytes, runs the CPU
/// reference, and wraps the result back into an `Array`.
struct OnnxHostDelegate {
    name: String,
}

impl MlxKernel for OnnxHostDelegate {
    fn name(&self) -> &str {
        &self.name
    }

    fn execute(
        &self,
        inputs: &[&Array],
        output_shape: &Shape,
        attrs: &[u8],
    ) -> Result<Array, MlxError> {
        let in_bytes: Vec<Vec<u8>> = inputs
            .iter()
            .map(|a| a.to_bytes())
            .collect::<Result<_, _>>()?;
        // MLX Array doesn't expose dtype in Rust; recover it from
        // bytes / nelems (4 → F32, 8 → I64 for Vocos Scatter/GatherND).
        let mut views: Vec<(Vec<u8>, Shape)> = Vec::with_capacity(inputs.len());
        for (i, (arr, bytes)) in inputs.iter().zip(in_bytes.iter()).enumerate() {
            let dims = arr.shape()?;
            let nelems: usize = dims.iter().product::<usize>().max(1);
            let elem = bytes.len() / nelems;
            let dtype = match elem {
                1 => rlx_ir::DType::Bool,
                2 => rlx_ir::DType::F16,
                4 => rlx_ir::DType::F32,
                8 => rlx_ir::DType::I64,
                _ => {
                    return Err(MlxError(format!(
                        "OnnxHostDelegate({}): input {i} has {} B / {} elems",
                        self.name,
                        bytes.len(),
                        nelems
                    )));
                }
            };
            views.push((bytes.clone(), Shape::new(&dims, dtype)));
        }
        let out_dims: Vec<usize> = output_shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        let out_nelems: usize = out_dims.iter().product::<usize>().max(1);
        let out_dtype = output_shape.dtype();
        let mut out_buf = vec![0u8; out_nelems * out_dtype.size_bytes()];
        let in_refs: Vec<(&[u8], &Shape)> = views.iter().map(|(b, s)| (b.as_slice(), s)).collect();
        rlx_cpu::op_registry::run_custom_op_host(
            &self.name,
            &in_refs,
            (&mut out_buf, output_shape),
            attrs,
        )
        .map_err(MlxError)?;
        if std::env::var("RLX_DBG_CUSTOM").is_ok() {
            let peak = out_buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).abs())
                .fold(0.0f32, f32::max);
            let head: Vec<f32> = out_buf
                .chunks_exact(4)
                .take(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let in0_head: Vec<f32> = views
                .first()
                .map(|(b, _)| {
                    b.chunks_exact(4)
                        .take(8)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                })
                .unwrap_or_default();
            let _ = in0_head;
            let in_peaks: Vec<(f32, usize)> = views
                .iter()
                .map(|(b, s)| {
                    let (peak, ninfs) = match s.dtype() {
                        rlx_ir::DType::F32 => {
                            let mut p = 0.0f32;
                            let mut n = 0usize;
                            for c in b.chunks_exact(4) {
                                let x = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                                if !x.is_finite() {
                                    n += 1;
                                }
                                p = p.max(x.abs());
                            }
                            (p, n)
                        }
                        _ => (0.0, 0),
                    };
                    (peak, ninfs)
                })
                .collect();
            eprintln!(
                "[mlx-host] {} in_shapes={:?} out={:?} in_peaks={in_peaks:?} out_peak={peak:.4} head={head:?}",
                self.name,
                views
                    .iter()
                    .map(|(_, s)| (s.dtype(), s.dims().to_vec()))
                    .collect::<Vec<_>>(),
                (out_dtype, out_dims.clone()),
            );
        }
        Array::from_bytes(&out_buf, &out_dims, out_dtype)
    }
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
