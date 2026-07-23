//! Vector transcendentals — same API as [`rlx_cpu::vmath`].
//!
//! Host path: Accelerate vForce on Apple, libm elsewhere.
//! Device path: GLSL `unary.comp` — Activation-order ids plus recip.

pub use rlx_cpu::vmath::*;

/// `unary.comp` op id for `Exp` (Activation enum order).
pub const UNARY_OP_EXP: u32 = 6;
/// `unary.comp` op id for `Tanh`.
pub const UNARY_OP_TANH: u32 = 5;
/// `unary.comp` op id for `vvrecf` (`1/x`).
pub const UNARY_OP_REC: u32 = 17;
/// `unary.comp` op id for `Sigmoid`.
pub const UNARY_OP_SIGMOID: u32 = 4;
