// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # rlx-extend — the stable extension surface for downstream crates
//!
//! One import for everything you need to extend rlx from *outside* the
//! workspace (a model crate in `rlx-models`, a numerics crate, a research op)
//! **without editing core**:
//!
//! ```ignore
//! use rlx_extend::prelude::*;
//! ```
//!
//! This crate is a thin, slow-moving facade: it only re-exports the public
//! extension traits and builders from [`rlx_ir`] and [`rlx_flow`], so the crate
//! *layout* underneath can be refactored without breaking downstream code that
//! depends on this surface. Depend on `rlx-extend` for the extension API rather
//! than reaching into the internal crates directly.
//!
//! ## The seams
//!
//! - **Block seam** — implement [`LayerStage`](rlx_flow::LayerStage) (compose
//!   primitives via [`FlowCtx`](rlx_flow::FlowCtx)'s builders: `ctx.matmul`,
//!   `ctx.linear`, `ctx.rms_norm`, …) and drop it into a flow with
//!   [`ModelFlow::layer_stage`](rlx_flow::ModelFlow::layer_stage). No
//!   `FlowStage` enum variant, no core edit. Publish auxiliary outputs with
//!   [`FlowCtx::publish_side_output`](rlx_flow::FlowCtx::publish_side_output).
//! - **Op seam** — implement [`OpExtension`](rlx_ir::OpExtension) and
//!   [`register_op`](rlx_ir::register_op); build nodes with
//!   [`Graph::custom_op`](rlx_ir::Graph::custom_op) /
//!   [`try_custom_op`](rlx_ir::Graph::try_custom_op). Provide an
//!   [`OpExtension::lower`](rlx_ir::OpExtension::lower) rule to decompose to
//!   primitives (runs on every backend with no kernel), or register a
//!   per-backend kernel for a native path.
//! - **Backend / codegen seam** — lives in `rlx-runtime` (heavier dep, not
//!   re-exported here): `rlx_runtime::register_backend` against a
//!   `rlx_runtime::Device`, or consume an `rlx_ir::Graph` directly for an
//!   ahead-of-time codegen target (the `rlx-fpga` / `rlx-cerebras` pattern).

#![forbid(unsafe_code)]

// Re-export the underlying crates for fully-qualified access when the prelude
// glob is too broad.
pub use rlx_flow;
pub use rlx_ir;

pub mod prelude {
    //! One glob import for extending rlx from a downstream crate.

    // Flow DSL + downstream block seam (ModelFlow, LayerStage, FlowCtx, …).
    pub use rlx_flow::prelude::*;
    pub use rlx_flow::{BlockAsLayer, DynStage};

    // Op extension seam (custom ops: registration, shape/AD/lowering hooks).
    pub use rlx_ir::{
        CustomOpError, DType, Graph, JvpContext, LowerContext, Node, NodeId, Op, OpExtension,
        OpKind, Shape, VjpContext, VmapContext, is_op_registered, lookup_op, register_op,
        register_op_strict,
    };
}

#[cfg(test)]
mod tests {
    // Smoke test: the prelude names actually resolve (the surface is coherent).
    #[allow(unused_imports)]
    use crate::prelude::*;

    #[test]
    fn prelude_surface_resolves() {
        // Reference one name from each seam so a rename upstream breaks here.
        fn _block(_: &dyn LayerStage) {}
        fn _op(_: &dyn OpExtension) {}
        let _ = is_op_registered("definitely_not_registered");
    }
}
