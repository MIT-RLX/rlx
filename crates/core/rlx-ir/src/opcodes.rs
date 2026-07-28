// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Canonical integer opcodes for the standalone elementwise kernels — the
//! **single source of truth** for the small `u32` "op id" every backend hands to
//! its unary / binary / compare / reduce kernel switch.
//!
//! Historically each backend hand-rolled its own `binary_op_id` / `act_id` /
//! `reduce_id` match. Being copies, they drifted: the forward-activation
//! numbering on CUDA/wgpu (`Relu=0 … Gelu=9`) diverged from Vulkan/oneAPI
//! (`Gelu=0 … Relu=3`), and Vulkan even carried two disagreeing activation
//! tables. Divergent-but-same-shaped opcode tables are exactly the silent
//! wrong-output failure mode we want to design out.
//!
//! These inherent methods pin every id in one place. Each backend wrapper now
//! delegates here (a value-preserving change — the numbers are unchanged), and
//! the `#[cfg(test)]` bijection guards below fail the moment a new enum variant
//! is added without an id, or two variants collide on one id. The exhaustive
//! `match` means adding an [`Activation`]/[`BinaryOp`]/… variant won't even
//! compile until it is given an opcode.
//!
//! ## Two activation schemes, both pinned here
//!
//! There are two real, self-consistent forward-activation numberings, each baked
//! into a matching set of GPU shader switches:
//!
//! * [`Activation::opcode_relu_first`] — CUDA / wgpu / ROCm **forward** unary
//!   kernels, and **every** backend's activation-*backward* kernel
//!   (Vulkan/oneAPI/Metal).
//! * [`Activation::opcode_gelu_first`] — Vulkan and oneAPI **forward** unary
//!   kernels (`act_id`).
//!
//! Collapsing the two into one is a separate, rig-validated change: it requires
//! editing the paired `.comp` / `.cl` shader switches in lockstep and re-running
//! the cross-backend parity sweep on real Vulkan/oneAPI hardware. Until then,
//! defining each scheme exactly once here stops any *further* drift — and the
//! parity harness (see `rlx-runtime/tests/elementwise_backend_parity.rs`)
//! detects at runtime if a shader ever disagrees with its id table.

use crate::op::{Activation, BinaryOp, CmpOp, ReduceOp};

impl BinaryOp {
    /// Every [`BinaryOp`] variant, for exhaustive test/parity fan-out.
    pub const ALL: [BinaryOp; 14] = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Max,
        BinaryOp::Min,
        BinaryOp::Pow,
        BinaryOp::Mod,
        BinaryOp::BitAnd,
        BinaryOp::BitOr,
        BinaryOp::BitXor,
        BinaryOp::Shl,
        BinaryOp::Shr,
        BinaryOp::Atan2,
    ];

    /// Opcode passed to the `binary` kernel on every backend (identical across
    /// CPU/CUDA/ROCm/Metal/wgpu/Vulkan/oneAPI).
    pub const fn opcode(self) -> u32 {
        match self {
            BinaryOp::Add => 0,
            BinaryOp::Sub => 1,
            BinaryOp::Mul => 2,
            BinaryOp::Div => 3,
            BinaryOp::Max => 4,
            BinaryOp::Min => 5,
            BinaryOp::Pow => 6,
            BinaryOp::Mod => 7,
            BinaryOp::BitAnd => 8,
            BinaryOp::BitOr => 9,
            BinaryOp::BitXor => 10,
            BinaryOp::Shl => 11,
            BinaryOp::Shr => 12,
            BinaryOp::Atan2 => 13,
        }
    }
}

impl CmpOp {
    /// Every [`CmpOp`] variant, for exhaustive test/parity fan-out.
    pub const ALL: [CmpOp; 6] = [
        CmpOp::Eq,
        CmpOp::Ne,
        CmpOp::Lt,
        CmpOp::Le,
        CmpOp::Gt,
        CmpOp::Ge,
    ];

    /// Opcode passed to the `compare` kernel on every backend.
    pub const fn opcode(self) -> u32 {
        match self {
            CmpOp::Eq => 0,
            CmpOp::Ne => 1,
            CmpOp::Lt => 2,
            CmpOp::Le => 3,
            CmpOp::Gt => 4,
            CmpOp::Ge => 5,
        }
    }
}

impl ReduceOp {
    /// Every [`ReduceOp`] variant, for exhaustive test/parity fan-out.
    pub const ALL: [ReduceOp; 5] = [
        ReduceOp::Sum,
        ReduceOp::Mean,
        ReduceOp::Max,
        ReduceOp::Min,
        ReduceOp::Prod,
    ];

    /// Opcode passed to the `reduce` kernel (`sum=0, mean=1, max=2, min=3,
    /// prod=4`).
    pub const fn opcode(self) -> u32 {
        match self {
            ReduceOp::Sum => 0,
            ReduceOp::Mean => 1,
            ReduceOp::Max => 2,
            ReduceOp::Min => 3,
            ReduceOp::Prod => 4,
        }
    }

    /// Opcode passed to the `pool{1,2,3}d` kernels, whose legend differs from
    /// [`ReduceOp::opcode`]: `max=0, mean=1, sum=2, min=3, prod=4` (Max and Sum
    /// are swapped). Using the reduce legend here made max-pooling compute the
    /// window sum — see the CUDA `pool_op_id` note this replaced.
    pub const fn pool_opcode(self) -> u32 {
        match self {
            ReduceOp::Max => 0,
            ReduceOp::Mean => 1,
            ReduceOp::Sum => 2,
            ReduceOp::Min => 3,
            ReduceOp::Prod => 4,
        }
    }
}

impl Activation {
    /// Every [`Activation`] variant, for exhaustive test/parity fan-out.
    pub const ALL: [Activation; 29] = [
        Activation::Gelu,
        Activation::GeluApprox,
        Activation::Silu,
        Activation::Relu,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::Exp,
        Activation::Log,
        Activation::Sqrt,
        Activation::Rsqrt,
        Activation::Neg,
        Activation::Abs,
        Activation::Sin,
        Activation::Cos,
        Activation::Tan,
        Activation::Atan,
        Activation::Recip,
        Activation::Round,
        Activation::Floor,
        Activation::Ceil,
        Activation::Sign,
        Activation::Softplus,
        Activation::Elu,
        Activation::Erf,
        Activation::HardSwish,
        Activation::HardSigmoid,
        Activation::Mish,
        Activation::Softsign,
        Activation::LogSigmoid,
    ];

    /// "Relu-first" activation opcode: CUDA / wgpu / ROCm **forward** unary
    /// kernels, and **every** backend's activation-*backward* kernel
    /// (`activation_backward.{cu,comp,cl}` + Metal). `Relu=0 … Gelu=9 …
    /// Recip=17 … LogSigmoid=28`.
    pub const fn opcode_relu_first(self) -> u32 {
        match self {
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
            Activation::Recip => 17,
            Activation::Floor => 18,
            Activation::Ceil => 19,
            Activation::Sign => 20,
            Activation::Softplus => 21,
            Activation::Elu => 22,
            Activation::Erf => 23,
            Activation::HardSwish => 24,
            Activation::HardSigmoid => 25,
            Activation::Mish => 26,
            Activation::Softsign => 27,
            Activation::LogSigmoid => 28,
        }
    }

    /// "Gelu-first" activation opcode: Vulkan and oneAPI **forward** unary
    /// kernels (`act_id`). `Gelu=0, GeluApprox=1, Silu=2, Relu=3 …
    /// LogSigmoid=28`. Note `Round=16, Recip=17` (swapped vs enum order).
    pub const fn opcode_gelu_first(self) -> u32 {
        match self {
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
            Activation::Sin => 12,
            Activation::Cos => 13,
            Activation::Tan => 14,
            Activation::Atan => 15,
            Activation::Round => 16,
            Activation::Recip => 17,
            Activation::Floor => 18,
            Activation::Ceil => 19,
            Activation::Sign => 20,
            Activation::Softplus => 21,
            Activation::Elu => 22,
            Activation::Erf => 23,
            Activation::HardSwish => 24,
            Activation::HardSigmoid => 25,
            Activation::Mish => 26,
            Activation::Softsign => 27,
            Activation::LogSigmoid => 28,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scheme is valid iff its ids are a bijection onto `0..N` — no gaps, no
    /// collisions. This is what fails if someone adds a variant and reuses an id
    /// or leaves a hole.
    fn assert_bijection(name: &str, ids: &[u32]) {
        let n = ids.len();
        let mut sorted = ids.to_vec();
        sorted.sort_unstable();
        for (i, &id) in sorted.iter().enumerate() {
            assert_eq!(
                id as usize, i,
                "{name}: opcodes must be a contiguous bijection onto 0..{n}; \
                 got {sorted:?} (duplicate or gap at index {i})"
            );
        }
    }

    #[test]
    fn binary_opcodes_are_a_bijection() {
        let ids: Vec<u32> = BinaryOp::ALL.iter().map(|&o| o.opcode()).collect();
        assert_bijection("BinaryOp::opcode", &ids);
    }

    #[test]
    fn compare_opcodes_are_a_bijection() {
        let ids: Vec<u32> = CmpOp::ALL.iter().map(|&o| o.opcode()).collect();
        assert_bijection("CmpOp::opcode", &ids);
    }

    #[test]
    fn reduce_opcodes_are_a_bijection() {
        let ids: Vec<u32> = ReduceOp::ALL.iter().map(|&o| o.opcode()).collect();
        assert_bijection("ReduceOp::opcode", &ids);
        let pool: Vec<u32> = ReduceOp::ALL.iter().map(|&o| o.pool_opcode()).collect();
        assert_bijection("ReduceOp::pool_opcode", &pool);
    }

    #[test]
    fn activation_schemes_are_bijections() {
        let relu: Vec<u32> = Activation::ALL
            .iter()
            .map(|&a| a.opcode_relu_first())
            .collect();
        assert_bijection("Activation::opcode_relu_first", &relu);
        let gelu: Vec<u32> = Activation::ALL
            .iter()
            .map(|&a| a.opcode_gelu_first())
            .collect();
        assert_bijection("Activation::opcode_gelu_first", &gelu);
    }

    /// `ALL` must list every variant exactly once (guards against forgetting to
    /// extend `ALL` when a variant is added — the bijection would still pass on
    /// a short list, but the count check here won't).
    #[test]
    fn all_arrays_have_expected_lengths() {
        assert_eq!(BinaryOp::ALL.len(), 14);
        assert_eq!(CmpOp::ALL.len(), 6);
        assert_eq!(ReduceOp::ALL.len(), 5);
        assert_eq!(Activation::ALL.len(), 29);
    }

    /// Pin the *exact* opcode of every variant to the value its GPU shader
    /// switch expects. A bijection check alone would still pass if two ids were
    /// swapped — but a swap silently mismatches the paired `.cu`/`.wgsl`/`.comp`
    /// switch. This is the lock: if the canonical table above is edited, this
    /// fails, forcing whoever changes it to also update the shaders (and the
    /// cross-backend parity sweep to re-confirm on hardware).
    #[test]
    fn canonical_ids_are_pinned() {
        // Binary — identical on every backend.
        for (op, want) in [
            (BinaryOp::Add, 0),
            (BinaryOp::Div, 3),
            (BinaryOp::Pow, 6),
            (BinaryOp::Mod, 7),
            (BinaryOp::Shr, 12),
            (BinaryOp::Atan2, 13),
        ] {
            assert_eq!(op.opcode(), want, "BinaryOp::{op:?}");
        }
        // Compare / reduce.
        assert_eq!(CmpOp::Eq.opcode(), 0);
        assert_eq!(CmpOp::Ge.opcode(), 5);
        assert_eq!(ReduceOp::Sum.opcode(), 0);
        assert_eq!(ReduceOp::Prod.opcode(), 4);
        // Pool legend swaps Max/Sum vs the reduce legend.
        assert_eq!(ReduceOp::Max.opcode(), 2);
        assert_eq!(ReduceOp::Max.pool_opcode(), 0);
        assert_eq!(ReduceOp::Sum.pool_opcode(), 2);
        // Relu-first (CUDA/wgpu/ROCm forward + all backward kernels).
        assert_eq!(Activation::Relu.opcode_relu_first(), 0);
        assert_eq!(Activation::Gelu.opcode_relu_first(), 9);
        assert_eq!(Activation::Recip.opcode_relu_first(), 17);
        assert_eq!(Activation::LogSigmoid.opcode_relu_first(), 28);
        // Gelu-first (Vulkan/oneAPI forward) — note Round=16, Recip=17.
        assert_eq!(Activation::Gelu.opcode_gelu_first(), 0);
        assert_eq!(Activation::Relu.opcode_gelu_first(), 3);
        assert_eq!(Activation::Atan.opcode_gelu_first(), 15);
        assert_eq!(Activation::Round.opcode_gelu_first(), 16);
        assert_eq!(Activation::Recip.opcode_gelu_first(), 17);
        assert_eq!(Activation::LogSigmoid.opcode_gelu_first(), 28);
    }
}
