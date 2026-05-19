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
//! Mode is selected at compile time via `MlxBackend::compile_with_mode`
//! or globally via the `RLX_MLX_MODE=eager|lazy` env var.
//!
//! Layout mirrors rlx-cpu / rlx-metal:
//! - `ffi`     — raw `extern "C"` declarations for the C++ shim
//! - `array`   — RAII `Array` wrapper + `MlxError`
//! - `ops`     — typed wrappers around shim ops
//! - `lower`   — rlx-ir Graph → MLX op chain
//! - `backend` — `MlxExecutable` (set_param / run / handles)

#[cfg(target_os = "macos")]
mod ffi;

#[cfg(target_os = "macos")]
pub mod array;

#[cfg(target_os = "macos")]
pub mod ops;

#[cfg(target_os = "macos")]
pub mod lower;

#[cfg(target_os = "macos")]
pub mod backend;

#[cfg(target_os = "macos")]
pub mod compiled;

#[cfg(target_os = "macos")]
pub mod calibrate;

#[cfg(target_os = "macos")]
pub mod op_registry;

#[cfg(target_os = "macos")]
pub mod batched_lu_kernel;

#[cfg(target_os = "macos")]
pub use array::{Array, MlxError, eval, version};
#[cfg(target_os = "macos")]
pub use backend::MlxExecutable;
#[cfg(target_os = "macos")]
pub use compiled::CompiledFn;
#[cfg(target_os = "macos")]
pub use lower::MlxMode;

/// True if MLX is reachable on this build target. MLX requires Apple
/// Silicon macOS; non-macOS builds compile out the entire backend.
#[cfg(target_os = "macos")]
pub fn is_available() -> bool {
    true
}

#[cfg(not(target_os = "macos"))]
pub fn is_available() -> bool {
    false
}
