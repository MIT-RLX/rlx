// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use rlx_ir::op::{Activation, BinaryOp, CmpOp, ReduceOp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatmulCompute {
    F32,
    F16,
    Coop16,
    /// Cooperative-matrix on Apple's `simdgroup_float8x8` — same hardware
    /// GEMM unit as Coop16 but with f32 operands and f32 accumulator.
    /// No precision loss vs F32 baseline; no f16 overflow risk in deep
    /// FFN sums. Used when alignment + features allow but the IR is f32.
    CoopF32,
    /// Vulkan/NVIDIA 16×16 f16 tensor-core matmul with K-slab f32
    /// reduction (avoids Naga mixed f16/f32 coop_mat bugs).
    CoopF16Vk,
    /// Packed-BF16 weight kept 2 B/elem in the `bf16_weight_buffer` side
    /// buffer and unpacked in-shader (`matmul_bf16w`). Reads half the B
    /// bytes; bit-exact to a bf16-rounded f32 matmul (f32 accumulator).
    /// Used only by the plain tiled matmul path for a BF16 Param rhs.
    Bf16Packed,
}

/// Split-write QKV matmul kernel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatmulQkvKind {
    F32,
    CoopF32,
    CoopF16Vk,
}

pub(crate) fn fft_dtype_tag(dtype: rlx_ir::DType) -> u32 {
    match dtype {
        rlx_ir::DType::F32 => 0,
        rlx_ir::DType::F64 => 1,
        rlx_ir::DType::C64 => 2,
        other => panic!("rlx-wgpu Op::Fft: unsupported dtype {other:?}"),
    }
}

pub(crate) fn fft_dtype_from_tag(tag: u32) -> rlx_ir::DType {
    match tag {
        0 => rlx_ir::DType::F32,
        1 => rlx_ir::DType::F64,
        2 => rlx_ir::DType::C64,
        other => panic!("rlx-wgpu Op::Fft: bad dtype tag {other}"),
    }
}

pub(crate) fn binary_op_id(op: BinaryOp) -> u32 {
    op.opcode()
}

pub(crate) fn compare_op_id(op: CmpOp) -> u32 {
    op.opcode()
}

pub(crate) fn reduce_op_id(op: ReduceOp) -> u32 {
    op.opcode()
}

pub(crate) fn activation_op_id(act: Activation) -> u32 {
    act.opcode_relu_first()
}

/// True when a matmul reads its weight `B` from the separate f16 shadow buffer
/// (so `B` is NOT bound through the arena binding). For these precisions the
/// arena window must cover only the activation + output, never the weight.
pub(crate) fn matmul_b_from_f16(precision: MatmulCompute, b_is_param: bool) -> bool {
    b_is_param
        && matches!(
            precision,
            MatmulCompute::F16 | MatmulCompute::Coop16 | MatmulCompute::CoopF16Vk
        )
}

/// True when a matmul reads its weight `B` from the packed-BF16 side buffer
/// (`bf16_weight_buffer`) instead of the arena — so the arena window must
/// cover only the activation + output, never the weight.
pub(crate) fn matmul_b_from_packed_bf16(precision: MatmulCompute, b_is_param: bool) -> bool {
    b_is_param && precision == MatmulCompute::Bf16Packed
}
