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

//! OpKinds this backend claims for legalization (`Backend::supported_ops`).
//!
//! Source of truth for the Hexagon / QNN coverage matrix. Shared by
//! `qnn_backend::QnnBackend::supported_ops` and `device_ext::qnn_supports`
//! (via `OpKind` for fused claims; per-`Op` detail stays in `device_ext`).

/// Ops the FFI runtime (`runtime::QnnExecutable`) lowers today, plus fused
/// forms that `QnnBackend::compile` decomposes before lowering
/// (`FusedAttentionBlock` → primitive chain via `unfuse_attention_block`).
pub const SUPPORTED_OPS: &[rlx_ir::OpKind] = {
    use rlx_ir::OpKind::*;
    &[
        Input,
        Param,
        Constant,
        MatMul,
        Binary,
        Activation,
        Softmax,
        Reshape,
        Transpose,
        Narrow,
        Concat,
        Gather,
        LayerNorm,
        RmsNorm,
        Reduce,
        Rope,
        Attention,
        // Claimed first-class; `QnnBackend::compile` runs
        // `unfuse_attention_block` to the MatMul/Narrow/Reshape/Transpose/
        // Rope/Attention/Expand chain the FFI path lowers.
        FusedAttentionBlock,
        Expand,
        Conv,
        Quantize,
        Dequantize,
        // Host-dequant → f32 MatMul (GGUF packed weights); see `dequant`.
        DequantMatMul,
        // INT8 accumulate + requantize (no host f32 weight bake); see `qmatmul`.
        QMatMul,
    ]
};
