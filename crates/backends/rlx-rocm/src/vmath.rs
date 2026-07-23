//! Vector transcendentals — same API as [`rlx_cpu::vmath`].
//!
//! Host path (staging / fallback): Accelerate vForce on Apple, libm elsewhere.
//! Device path: shared `rlx-gpu-kernels` unary op ids (HIP).

pub use rlx_cpu::vmath::*;

/// HIP unary op id for `vvexpf` (`exp`).
pub const UNARY_OP_EXP: u32 = 3;
/// HIP unary op id for `vvtanhf` (`tanh`).
pub const UNARY_OP_TANH: u32 = 2;
/// HIP unary op id for `vvrecf` (`1/x`).
pub const UNARY_OP_REC: u32 = 17;
/// HIP unary op id for logistic sigmoid.
pub const UNARY_OP_SIGMOID: u32 = 1;
