// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Host-delegate collective ops on Metal.
//!
//! The `collective.*` ops are host/transport ops with no device kernel. Metal
//! stages operands via unified memory and delegates to the registered CPU
//! kernel via [`rlx_cpu::op_registry::run_f32_custom_op_host`]. Op names come
//! from [`rlx_gpu_host::COLLECTIVE_OPS`] (shared with CUDA/ROCm/wgpu).

use crate::op_registry::{MetalKernel, register_metal_kernel};
use rlx_gpu_host::COLLECTIVE_OPS;
use rlx_ir::Shape;
use std::sync::Arc;

#[derive(Debug)]
struct CollectiveHostKernel {
    name: &'static str,
}

impl MetalKernel for CollectiveHostKernel {
    fn name(&self) -> &str {
        self.name
    }
    fn execute(
        &self,
        inputs: &[(&[u8], &Shape)],
        output: (&mut [u8], &Shape),
        attrs: &[u8],
    ) -> Result<(), String> {
        rlx_cpu::op_registry::run_f32_custom_op_host(self.name, inputs, output, attrs)
    }
}

/// Register the host-delegate collective kernels on the Metal backend. Called
/// from [`crate::op_registry::ensure_builtins_registered`] so they are always
/// available; the actual transport still requires the consumer to have called
/// `rlx_collectives::register()` and registered a process group.
pub fn register() {
    for &name in COLLECTIVE_OPS {
        register_metal_kernel(Arc::new(CollectiveHostKernel { name }));
    }
}
