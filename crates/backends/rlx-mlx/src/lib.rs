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

//! RLX MLX backend — Apple's array framework via a hand-rolled C++
//! shim. Two execution modes coexist:
//!
//!   - **Lazy** (default) — build the entire MLX graph in `run()`, eval
//!     once. Lets MLX's optimizer see the whole DAG.
//!   - **Eager** — eval after every op. Slower; useful for debugging
//!     where the failure surfaces at the offending op.
//!
//! Mode is selected at compile time via `MlxExecutable::compile_with_mode`
//! or `MlxExecutable::compile_from_fused` (pre-fused LIR graphs), or
//! globally via the `RLX_MLX_MODE=eager|lazy|compiled` env var.
//!
//! Layout mirrors rlx-cpu / rlx-metal:
//! - `ffi`     — re-export of [`rlx_mlx_sys::ffi`] (C++ shim in `rlx-mlx-sys`)
//! - `array`   — RAII `Array` wrapper + `MlxError`
//! - `ops`     — typed wrappers around shim ops
//! - `lower`   — rlx-ir Graph → MLX op chain
//! - `backend` — `MlxExecutable` (set_param / run / handles)

/// Legalization op claim — always available (no MLX host required).
pub mod supported_ops;
pub use supported_ops::SUPPORTED_OPS;

pub mod vmath;

#[cfg(rlx_mlx_host)]
pub(crate) mod ffi {
    pub use rlx_mlx_sys::ffi::*;
}

#[cfg(rlx_mlx_host)]
pub mod array;

#[cfg(rlx_mlx_host)]
pub mod ops;

#[cfg(rlx_mlx_host)]
pub mod attention_bwd;

#[cfg(rlx_mlx_host)]
pub mod lower;

#[cfg(rlx_mlx_host)]
pub(crate) mod sync;

#[cfg(rlx_mlx_host)]
pub mod backend;

#[cfg(rlx_mlx_host)]
pub mod config;

#[cfg(rlx_mlx_host)]
pub mod compiled;

#[cfg(rlx_mlx_host)]
pub mod calibrate;

#[cfg(rlx_mlx_host)]
pub mod op_registry;

#[cfg(rlx_mlx_host)]
pub mod splat;

#[cfg(rlx_mlx_host)]
pub mod batched_lu_kernel;

#[cfg(rlx_mlx_host)]
pub mod dequant_q1_0;

#[cfg(rlx_mlx_host)]
pub mod llada2_gate;
// Host-only: depends on `op_registry` (itself `rlx_mlx_host`-gated). Without
// this gate the module leaks onto non-host targets (e.g. iOS) and fails to
// resolve `crate::op_registry`.
#[cfg(rlx_mlx_host)]
pub mod ms_deform_attn;

#[cfg(rlx_mlx_host)]
pub mod distributed;

#[cfg(rlx_mlx_host)]
pub use array::{Array, MlxError, eval, version};
#[cfg(rlx_mlx_host)]
pub use backend::MlxExecutable;
#[cfg(rlx_mlx_host)]
pub use compiled::CompiledFn;
#[cfg(rlx_mlx_host)]
pub use config::{
    COMPILE_OUTPUT_CAP_ENV, DEFAULT_COMPILE_OUTPUT_CAP, MlxRuntimeConfig, compile_output_cap,
    install_runtime_config, reload_runtime_config, reset_compile_output_cap, runtime_config,
    set_compile_output_cap,
};
#[cfg(rlx_mlx_host)]
pub use distributed::MlxTransport;
#[cfg(rlx_mlx_host)]
pub use lower::MlxMode;

/// True when this target links the native MLX stack (macOS Metal, CPU MLX on
/// Linux / Windows).
#[cfg(rlx_mlx_host)]
pub fn is_available() -> bool {
    true
}

#[cfg(not(rlx_mlx_host))]
pub fn is_available() -> bool {
    false
}
