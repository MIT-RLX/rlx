// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! WebGL2 GPGPU backend for RLX.
//!
//! WebGL2 has **no compute shaders** (confirmed in wgpu's own source), so this
//! backend executes IR graphs the classic GPGPU way: every tensor is a
//! single-channel float (`R32F`) texture, and every op is a fragment shader
//! that renders one output element per fragment (render-to-texture).
//!
//! ## How it stays correct without a browser in the loop
//!
//! All the index arithmetic (transpose / reshape / expand → "gather by
//! precomputed index"; reduce → "sum over precomputed index groups") lives in
//! the pure-Rust [`plan`]ner, and a [CPU executor](exec_cpu) runs the very
//! same [`Plan`]. The CPU path is unit-tested against RLX's own CPU autodiff,
//! so the planner + numerics are verified natively. The WebGL fragment shaders
//! ([`exec_gl`], wasm only) mirror those exact formulas, fetching inputs by the
//! planner's precomputed indices — so the GL path inherits the verified math
//! and only the GL plumbing needs in-browser validation.
//!
//! Supported ops (the surface a feed-forward net + its autodiff backward pass
//! needs): `Input`/`Param`/`Constant`, `MatMul`, `Binary{Add,Sub,Mul,Div}`,
//! `Activation(Relu)`, `ReluBackward`, `Reduce{Sum,Mean}`, `Expand`, `Reshape`,
//! `Transpose`.

pub mod exec_cpu;
pub mod plan;

#[cfg(target_arch = "wasm32")]
pub mod exec_gl;

pub use exec_cpu::run_cpu;
pub use plan::{Act, Bin, Cmp, LeafSource, Plan, Red, Step, build_plan, supported_ops};

/// Error type for planning / execution.
#[derive(Debug, Clone)]
pub struct WebglError(pub String);

impl std::fmt::Display for WebglError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rlx-webgl: {}", self.0)
    }
}

impl std::error::Error for WebglError {}

pub type Result<T> = std::result::Result<T, WebglError>;
