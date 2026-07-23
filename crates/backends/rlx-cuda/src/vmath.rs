//! Vector transcendentals — same API as [`rlx_cpu::vmath`].
//!
//! Host path (staging / fallback): Accelerate vForce on Apple, libm elsewhere.
//! Device path: use unary / inplace kernels with the op ids below.

pub use rlx_cpu::vmath::*;

/// CUDA / ROCm / wgpu `unary` op id for `vvexpf` (`exp`).
pub const UNARY_OP_EXP: u32 = 3;
/// CUDA / ROCm / wgpu `unary` op id for `vvtanhf` (`tanh`).
pub const UNARY_OP_TANH: u32 = 2;
/// CUDA / ROCm / wgpu `unary` op id for `vvrecf` (`1/x`).
pub const UNARY_OP_REC: u32 = 17;
/// CUDA / ROCm / wgpu `unary` op id for logistic sigmoid.
pub const UNARY_OP_SIGMOID: u32 = 1;
