//! Vector transcendentals — same API as [`rlx_cpu::vmath`].
//!
//! Host path: Accelerate vForce on Apple, libm elsewhere.
//! Device path: WGSL `unary` kernel op ids (same numbering as CUDA).

pub use rlx_cpu::vmath::*;

/// WGSL `unary` op id for `vvexpf`.
pub const UNARY_OP_EXP: u32 = 3;
/// WGSL `unary` op id for `vvtanhf`.
pub const UNARY_OP_TANH: u32 = 2;
/// WGSL `unary` op id for `vvrecf` (`1/x`).
pub const UNARY_OP_REC: u32 = 17;
/// WGSL `unary` op id for logistic sigmoid.
pub const UNARY_OP_SIGMOID: u32 = 1;
