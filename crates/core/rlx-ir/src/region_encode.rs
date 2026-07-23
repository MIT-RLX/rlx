// RLX - versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Shared [`Op::ElementwiseRegion`] metadata encoding for GPU region kernels.

use crate::op::*;
use crate::shape::Shape;

pub const REGION_META_INPUT_WORDS: usize = 16;
pub const REGION_META_CHAIN_WORDS: usize = 128;
pub const REGION_META_TAIL_WORDS: usize = 6;
pub const REGION_META_WORDS: usize =
    REGION_META_INPUT_WORDS + REGION_META_CHAIN_WORDS + REGION_META_TAIL_WORDS;

/// Max batch slices for the single-launch batch region kernel (CUDA/ROCm/Metal/wgpu).
pub const FK_BATCH_SINGLE_KERNEL_MAX: usize = 64;

/// `RLX_FK_BATCH_SINGLE_KERNEL=1` at compile time.
pub fn fk_batch_single_kernel_enabled() -> bool {
    crate::env::flag("RLX_FK_BATCH_SINGLE_KERNEL")
}

/// Whether `BatchElementwiseRegion` should use one batch-region launch.
pub fn fk_batch_use_single_launch(num_batch: usize, prologue: RegionPrologue) -> bool {
    fk_batch_single_kernel_enabled()
        && prologue == RegionPrologue::None
        && num_batch <= FK_BATCH_SINGLE_KERNEL_MAX
}

pub const REGION_PROLOGUE_NONE: u32 = 0;
pub const REGION_PROLOGUE_RESIZE_NEAREST_2X_NCHW: u32 = 1;

/// NCHW output dimensions for prologue kernels (`n,c,h,w`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionNchwDims {
    pub n: u32,
    pub c: u32,
    pub h: u32,
    pub w: u32,
}

impl RegionNchwDims {
    pub fn from_shape(shape: &Shape) -> Option<Self> {
        if shape.rank() != 4 {
            return None;
        }
        Some(Self {
            n: shape.dim(0).unwrap_static() as u32,
            c: shape.dim(1).unwrap_static() as u32,
            h: shape.dim(2).unwrap_static() as u32,
            w: shape.dim(3).unwrap_static() as u32,
        })
    }

    /// Linear element count for an NCHW tensor.
    pub fn num_elements(self) -> u32 {
        self.n * self.c * self.h * self.w
    }
}

/// 3D launch grid for resize-prologue region kernels (width x height x N*C).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrologueLaunchGrid {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl PrologueLaunchGrid {
    pub fn from_output_shape(shape: &Shape) -> Option<Self> {
        let d = RegionNchwDims::from_shape(shape)?;
        Some(Self {
            width: d.w,
            height: d.h,
            depth: d.n * d.c,
        })
    }
}

/// Encode operand for region chain steps (shared across CUDA / Metal / wgpu).
pub fn encode_chain_operand(op: &ChainOperand) -> u32 {
    match *op {
        ChainOperand::Input(i) => i & 0x7FFF_FFFFu32,
        ChainOperand::Step(i) => 0x8000_0000u32 | (i & 0x7FFF_FFFFu32),
    }
}

pub fn activation_sub(a: Activation) -> u32 {
    match a {
        Activation::Gelu => 0,
        Activation::GeluApprox => 1,
        Activation::Silu => 2,
        Activation::Relu => 3,
        Activation::Sigmoid => 4,
        Activation::Tanh => 5,
        Activation::Exp => 6,
        Activation::Log => 7,
        Activation::Sqrt => 8,
        Activation::Rsqrt => 9,
        Activation::Neg => 10,
        Activation::Abs => 11,
        Activation::Round => 12,
        Activation::Sin => 13,
        Activation::Cos => 14,
        Activation::Tan => 15,
        Activation::Atan => 16,
        Activation::Recip => 17,
    }
}

pub fn binary_sub(b: BinaryOp) -> u32 {
    match b {
        BinaryOp::Add => 0,
        BinaryOp::Sub => 1,
        BinaryOp::Mul => 2,
        BinaryOp::Div => 3,
        BinaryOp::Max => 4,
        BinaryOp::Min => 5,
        BinaryOp::Pow => 6,
    }
}

pub fn compare_sub(c: CmpOp) -> u32 {
    match c {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Lt => 2,
        CmpOp::Le => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
    }
}

/// Pack chain steps into 128 u32 words.
pub fn encode_chain_steps(chain: &[ChainStep]) -> [u32; REGION_META_CHAIN_WORDS] {
    let mut chain_enc = [0u32; REGION_META_CHAIN_WORDS];
    for (k, step) in chain.iter().enumerate() {
        let base = k * 4;
        let (kind, sub, lhs, rhs) = match step {
            ChainStep::Activation(a, src) => {
                (0u32, activation_sub(*a), encode_chain_operand(src), 0u32)
            }
            ChainStep::Cast(_, src) => (1u32, 0, encode_chain_operand(src), 0u32),
            ChainStep::Binary(op, l, r) => (
                2u32,
                binary_sub(*op),
                encode_chain_operand(l),
                encode_chain_operand(r),
            ),
            ChainStep::Compare(op, l, r) => (
                3u32,
                compare_sub(*op),
                encode_chain_operand(l),
                encode_chain_operand(r),
            ),
            ChainStep::Where(c, t, f) => (
                4u32,
                encode_chain_operand(c),
                encode_chain_operand(t),
                encode_chain_operand(f),
            ),
        };
        chain_enc[base] = kind;
        chain_enc[base + 1] = sub;
        chain_enc[base + 2] = lhs;
        chain_enc[base + 3] = rhs;
    }
    chain_enc
}

/// Prologue tag + NCHW shape + external input index for the region output tensor.
pub fn encode_prologue_tail(
    prologue: RegionPrologue,
    out_shape: &Shape,
    prologue_input: u32,
) -> [u32; REGION_META_TAIL_WORDS] {
    let mut tail = [0u32; REGION_META_TAIL_WORDS];
    match prologue {
        RegionPrologue::None => {}
        RegionPrologue::ResizeNearest2x => {
            if let Some(d) = RegionNchwDims::from_shape(out_shape) {
                tail[0] = REGION_PROLOGUE_RESIZE_NEAREST_2X_NCHW;
                tail[1] = d.n;
                tail[2] = d.c;
                tail[3] = d.h;
                tail[4] = d.w;
            }
        }
    }
    tail[5] = prologue_input.min(15);
    tail
}

/// Per-slice output shape for [`Op::BatchElementwiseRegion`] (batch axis 0 ? 1).
pub fn batch_region_slice_shape(batch_out: &Shape) -> Shape {
    if batch_out.rank() >= 1 {
        batch_out.clone().with_dim(0, crate::shape::Dim::Static(1))
    } else {
        batch_out.clone()
    }
}

/// Element count of one batch slice in a contiguous batch output tensor.
pub fn batch_region_slice_elems(batch_out: &Shape, num_batch: usize) -> Option<u32> {
    let total = batch_out.num_elements()?;
    let n = num_batch.max(1);
    Some((total / n) as u32)
}

/// f32-linear offset of batch slice `index` within a packed output buffer.
pub fn batch_region_slice_dst_off_f32(base_dst_off: u32, slice_elems: u32, index: usize) -> u32 {
    base_dst_off.saturating_add(index as u32 * slice_elems)
}

/// Full device metadata buffer for [`Op::ElementwiseRegion`].
pub fn encode_elementwise_region_meta(
    input_offs: &[u32; REGION_META_INPUT_WORDS],
    chain: &[ChainStep],
    prologue: RegionPrologue,
    out_shape: &Shape,
    prologue_input: u32,
) -> [u32; REGION_META_WORDS] {
    let mut meta = [0u32; REGION_META_WORDS];
    meta[..REGION_META_INPUT_WORDS].copy_from_slice(input_offs);
    meta[REGION_META_INPUT_WORDS..REGION_META_INPUT_WORDS + REGION_META_CHAIN_WORDS]
        .copy_from_slice(&encode_chain_steps(chain));
    let tail = encode_prologue_tail(prologue, out_shape, prologue_input);
    let tail_start = REGION_META_INPUT_WORDS + REGION_META_CHAIN_WORDS;
    meta[tail_start..tail_start + REGION_META_TAIL_WORDS].copy_from_slice(&tail);
    meta
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn meta_word_count_matches_layout() {
        assert_eq!(REGION_META_WORDS, 150);
    }

    #[test]
    fn batch_slice_elems_and_dst_off() {
        let shape = Shape::new(&[2, 3, 8, 8], DType::F32);
        assert_eq!(batch_region_slice_elems(&shape, 2), Some(192));
        assert_eq!(batch_region_slice_dst_off_f32(100, 192, 1), 100 + 192);
    }

    #[test]
    fn resize_prologue_tail_packed() {
        let shape = Shape::new(&[1, 3, 16, 16], DType::F32);
        let tail = encode_prologue_tail(RegionPrologue::ResizeNearest2x, &shape, 0);
        assert_eq!(tail[0], REGION_PROLOGUE_RESIZE_NEAREST_2X_NCHW);
        assert_eq!((tail[1], tail[2], tail[3], tail[4]), (1, 3, 16, 16));
        assert_eq!(tail[5], 0);
        let tail1 = encode_prologue_tail(RegionPrologue::ResizeNearest2x, &shape, 1);
        assert_eq!(tail1[5], 1);
    }

    #[test]
    fn fk_batch_single_kernel_cap() {
        assert_eq!(FK_BATCH_SINGLE_KERNEL_MAX, 64);
    }

    #[test]
    fn fk_batch_use_single_launch_gating() {
        assert!(!fk_batch_use_single_launch(2, RegionPrologue::None));
        assert!(!fk_batch_use_single_launch(
            FK_BATCH_SINGLE_KERNEL_MAX + 1,
            RegionPrologue::None,
        ));
        assert!(!fk_batch_use_single_launch(
            2,
            RegionPrologue::ResizeNearest2x
        ));
    }
}
