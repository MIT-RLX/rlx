//! Vector transcendentals — same API as [`rlx_cpu::vmath`].
//!
//! Host path for staging. Device path: MLX `Exp` / `Tanh` unaries; reciprocal
//! is `1/x` via reciprocal or divide.

pub use rlx_cpu::vmath::*;
