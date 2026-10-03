// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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

/// True when MLX can actually **run** here — not merely when it is linked in.
///
/// This used to answer `true` for any target that links the native MLX stack,
/// which is a compile-time fact rather than a device probe, and callers treat
/// it as the latter: `fastest_device()` picks the best *available* backend and
/// then runs on it.
///
/// On an iPad with the MLX kernel library missing from the app bundle, that gap
/// was visible — `is_available(Mlx)` said yes, `fastest_device()` returned MLX,
/// and execution then failed with
///
/// ```text
/// Failed to load the default metallib. library not found
/// ```
///
/// MLX loads that library lazily on the first GPU kernel, so nothing short of
/// running one detects it. The probe is a single-element add plus an `eval` to
/// force materialization — cheap, and it exercises exactly the path that was
/// failing. Cached: the answer cannot change within a process, and callers ask
/// on every device-selection decision.
#[cfg(rlx_mlx_host)]
pub fn is_available() -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(probe_one_kernel)
}

/// Run the smallest possible MLX GPU kernel and report whether it worked.
///
/// Wrapped in `catch_unwind` as well as checked by return code: a missing
/// kernel library surfaces from MLX's C++ side, and a probe that aborts the
/// process would be worse than the wrong answer it exists to prevent.
#[cfg(rlx_mlx_host)]
fn probe_one_kernel() -> bool {
    use rlx_mlx_sys::ffi::{
        MlxDtype, RLX_MLX_OK, mlx_array_t, rlx_mlx_array_free, rlx_mlx_array_from_data,
        rlx_mlx_eval, rlx_mlx_op_add,
    };
    std::panic::catch_unwind(|| unsafe {
        let dims: [std::ffi::c_int; 1] = [1];
        let data: [std::ffi::c_float; 1] = [1.0];
        let mut a: *mut mlx_array_t = std::ptr::null_mut();
        if rlx_mlx_array_from_data(dims.as_ptr(), 1, data.as_ptr(), 1, MlxDtype::F32, &mut a)
            != RLX_MLX_OK
            || a.is_null()
        {
            return false;
        }
        let mut sum: *mut mlx_array_t = std::ptr::null_mut();
        let added = rlx_mlx_op_add(a, a, &mut sum) == RLX_MLX_OK && !sum.is_null();
        // `add` is lazy; only `eval` makes MLX load its kernel library.
        let ok = added && rlx_mlx_eval([sum].as_ptr(), 1) == RLX_MLX_OK;
        if !sum.is_null() {
            rlx_mlx_array_free(sum);
        }
        rlx_mlx_array_free(a);
        ok
    })
    .unwrap_or(false)
}

#[cfg(not(rlx_mlx_host))]
pub fn is_available() -> bool {
    false
}
