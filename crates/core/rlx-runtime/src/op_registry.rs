// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Op registry — re-exported from `rlx-ir`.
//!
//! The registry was promoted to `rlx-ir` once `Op::Custom` landed: the
//! IR layer needs to dispatch through it during shape inference, so
//! the registry can no longer live above the IR. This module is kept
//! as a thin re-export for backward compatibility with downstream
//! code that imported from `rlx_runtime::op_registry`.

pub use rlx_ir::op_registry::{
    OpExtension, OpRegistry, VjpContext, global_registry, lookup_op, register_op,
};
