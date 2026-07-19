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
    match op {
        BinaryOp::Add => 0,
        BinaryOp::Sub => 1,
        BinaryOp::Mul => 2,
        BinaryOp::Div => 3,
        BinaryOp::Max => 4,
        BinaryOp::Min => 5,
        BinaryOp::Pow => 6,
    }
}

pub(crate) fn compare_op_id(op: CmpOp) -> u32 {
    match op {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Lt => 2,
        CmpOp::Le => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
    }
}

pub(crate) fn reduce_op_id(op: ReduceOp) -> u32 {
    match op {
        ReduceOp::Sum => 0,
        ReduceOp::Mean => 1,
        ReduceOp::Max => 2,
        ReduceOp::Min => 3,
        ReduceOp::Prod => 4,
    }
}

pub(crate) fn activation_op_id(act: Activation) -> u32 {
    match act {
        Activation::Relu => 0,
        Activation::Sigmoid => 1,
        Activation::Tanh => 2,
        Activation::Exp => 3,
        Activation::Log => 4,
        Activation::Sqrt => 5,
        Activation::Rsqrt => 6,
        Activation::Neg => 7,
        Activation::Abs => 8,
        Activation::Gelu => 9,
        Activation::Silu => 10,
        Activation::GeluApprox => 11,
        Activation::Round => 12,
        Activation::Sin => 13,
        Activation::Cos => 14,
        Activation::Tan => 15,
        Activation::Atan => 16,
    }
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
