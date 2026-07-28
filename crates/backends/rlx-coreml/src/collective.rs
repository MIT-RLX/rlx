// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-delegate collective ops on CoreML/ANE.
//!
//! The `collective.*` ops (all-reduce / all-gather / reduce-scatter and the
//! Megatron `f`/`g` operators) are host/transport ops with no device kernel — a
//! CoreML ML Program on the ANE can't drive the process group. rlx-coreml
//! treats most `Op::Custom` as host ops ([`crate::host_exec::is_host_op`]),
//! running it between CoreML segments; these delegates make the collective ops
//! resolve to the single registered `rlx-cpu` collective kernel via
//! [`rlx_cpu::op_registry::run_f32_custom_op_host`], so ANE and CPU stay
//! bit-for-bit identical. Op-name strings mirror `rlx-collectives`, which
//! rlx-coreml cannot depend on (later publish tier); keep them in sync.

use crate::op_registry::{CoremlKernel, register_coreml_kernel};
use rlx_ir::Shape;
use std::sync::Arc;

/// Collective op names, mirroring `rlx_collectives::{ALL_REDUCE, ALL_GATHER,
/// REDUCE_SCATTER, COPY_TO_PARALLEL, REDUCE_FROM_PARALLEL}`.
const COLLECTIVE_OPS: &[&str] = &[
    "collective.all_reduce",
    "collective.all_gather",
    "collective.reduce_scatter",
    "collective.copy_to_parallel",
    "collective.reduce_from_parallel",
    "collective.broadcast",
    "collective.reduce",
    "collective.all_to_all",
    "collective.moe_dispatch",
    "collective.moe_combine",
    "collective.ppermute",
    "collective.send",
    "collective.recv",
];

#[derive(Debug)]
struct CollectiveHostKernel {
    name: &'static str,
}

impl CoremlKernel for CollectiveHostKernel {
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

/// Register the host-delegate collective kernels on the CoreML backend. Called
/// from [`crate::op_registry::lookup_coreml_kernel`] so they are always
/// available; the transport still needs the consumer to have called
/// `rlx_collectives::register()` and registered a process group.
pub fn register() {
    for &name in COLLECTIVE_OPS {
        register_coreml_kernel(Arc::new(CollectiveHostKernel { name }));
    }
}
