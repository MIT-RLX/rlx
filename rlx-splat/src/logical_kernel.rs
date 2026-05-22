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
//! Helpers for compiling splat graphs through the **common IR** logical-kernel path.
//!
//! Use when a backend should not claim [`OpKind::GaussianSplatRender`] natively (e.g. TPU) or
//! for parity tests (`RLX_KERNEL_DISPATCH=common`).

#![cfg(feature = "cpu")]

use rlx_ir::logical_kernel::{KernelDispatchConfig, KernelDispatchPolicy};
use rlx_ir::OpKind;

/// `supported_ops` claim set that forces splat to lower to primitive MIR (no native splat thunk).
pub const PRIMITIVE_SPLAT_SUPPORTED_OPS: &[OpKind] = &[
    OpKind::Input,
    OpKind::Param,
    OpKind::Constant,
    OpKind::Reshape,
    OpKind::Transpose,
    OpKind::Narrow,
    OpKind::Concat,
    OpKind::Expand,
    OpKind::Gather,
    OpKind::Reduce,
    OpKind::Binary,
    OpKind::Activation,
    OpKind::Cast,
    OpKind::Compare,
    OpKind::Where,
    OpKind::Softmax,
    OpKind::MatMul,
    OpKind::LayerNorm,
    OpKind::RmsNorm,
];

/// Default compile policy: native when listed in `supported_ops`, else common IR.
pub const DEFAULT_KERNEL_DISPATCH: KernelDispatchPolicy = KernelDispatchPolicy::PreferNative;

/// Force only splat logical ops to common IR while keeping other native CPU/Metal kernels.
pub const FORCE_COMMON_SPLAT_KINDS: &[OpKind] = &[
    OpKind::GaussianSplatRender,
    OpKind::GaussianSplatRenderBackward,
];

/// Build [`KernelDispatchConfig`] for “native matmul, common splat”.
pub fn splat_common_only_config() -> KernelDispatchConfig {
    KernelDispatchConfig {
        policy: KernelDispatchPolicy::PreferNative,
        force_common_kinds: FORCE_COMMON_SPLAT_KINDS,
        force_native_kinds: &[],
    }
}
