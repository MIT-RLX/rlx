// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-process HIP runtime + context singleton.
//!
//! Mirrors `rlx-cuda::device` shape. `rocm_runtime()` returns the
//! resolved `HipRuntime` bundle (function pointers via libloading)
//! or `None` on hosts without `libamdhip64` / `libhiprtc`.
//! `rocm_context()` returns the live `(runtime, ctx, default_stream)`
//! triple — the only thing dispatch actually needs.

use std::ffi::c_uint;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;

use crate::hip::{HipCtx, HipRuntime, HipStream};
use crate::hipblas::{HipblasContext, HipblasRuntime};
use crate::hipblaslt::{HipblasLtContext, HipblasLtRuntime};
use crate::miopen::{MiopenContext, MiopenRuntime};

/// Live HIP context plus the default stream we issue every dispatch
/// on (matches cudarc's `ctx.default_stream()` shape).
pub struct RocmContext {
    pub runtime: Arc<HipRuntime>,
    pub ctx: HipCtx,
    pub default_stream: HipStream,
}

unsafe impl Send for RocmContext {}
unsafe impl Sync for RocmContext {}

impl Drop for RocmContext {
    fn drop(&mut self) {
        unsafe {
            if !self.default_stream.is_null() {
                let _ = (self.runtime.hip_stream_destroy)(self.default_stream);
            }
            if !self.ctx.is_null() {
                let _ = (self.runtime.hip_ctx_destroy)(self.ctx);
            }
        }
    }
}

/// Cached ROCm context.
///
/// A `Mutex<Option<..>>` rather than a `OnceLock<Option<..>>`, and the
/// difference is load-bearing. `OnceLock` caches the FIRST result forever,
/// including a failure — so a single unlucky probe permanently marks ROCm
/// unavailable for the whole process.
///
/// That is not hypothetical. AMD GPUs runtime-suspend when idle
/// (`rocm-smi`: "AMD GPU device(s) is/are in a low-power state"), and a probe
/// against a suspended device fails. On the MI100 rig this made an entire
/// 57-test ROCm suite report "no rocm device — skipping" while `rocminfo`
/// listed four healthy HSA agents seconds later. The hardware was fine; the
/// first probe lost a race with the power state and the answer was frozen.
///
/// Success is cached; failure is not.
static CTX: Mutex<Option<Arc<RocmContext>>> = Mutex::new(None);

/// Set once when `libamdhip64` cannot be loaded at all.
///
/// That failure genuinely cannot change within a process, so it IS worth
/// caching — otherwise every `is_available()` on a machine with no ROCm pays
/// for a failed `dlopen`.
static RUNTIME_MISSING: OnceLock<bool> = OnceLock::new();
static BLAS: OnceLock<Option<Arc<Mutex<HipblasContext>>>> = OnceLock::new();
static BLAS_LT: OnceLock<Option<Arc<HipblasLtContext>>> = OnceLock::new();
static DNN: OnceLock<Option<Arc<MiopenContext>>> = OnceLock::new();

/// Initialise the HIP runtime + create a context on device 0 + create
/// a default stream. Returns `None` cleanly on hosts without HIP
/// (libloading fails to find `libamdhip64`) or when device 0 isn't
/// present.
///
/// **Retries after a failure.** See `CTX`: a device that is merely
/// runtime-suspended will answer on a later call, and the first call is often
/// what wakes it.
pub fn rocm_context() -> Option<Arc<RocmContext>> {
    if *RUNTIME_MISSING.get_or_init(|| false) {
        return None;
    }
    let mut guard = CTX.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(ctx) = guard.as_ref() {
        return Some(ctx.clone());
    }
    let built = build_context();
    if built.is_some() {
        *guard = built.clone();
    }
    built
}

/// One attempt at bringing up HIP. Separated from the caching so the retry
/// policy is visible in one place.
fn build_context() -> Option<Arc<RocmContext>> {
    let Some(runtime) = HipRuntime::load() else {
        // Only this failure is permanent.
        let _ = RUNTIME_MISSING.set(true);
        return None;
    };
    unsafe {
        if (runtime.hip_init)(0).ok().is_err() {
            warn_transient("hipInit failed");
            return None;
        }
        let mut count: i32 = 0;
        if (runtime.hip_get_device_count)(&mut count).ok().is_err() {
            warn_transient("hipGetDeviceCount failed");
            return None;
        }
        if count <= 0 {
            // Reported, not silent: zero devices on a machine that has them is
            // the suspended-device symptom, and a silent skip is how it went
            // unnoticed for a full test suite.
            warn_transient("hipGetDeviceCount returned 0 devices");
            return None;
        }

        let mut device: i32 = 0;
        if (runtime.hip_device_get)(&mut device, 0).ok().is_err() {
            warn_transient("hipDeviceGet(0) failed");
            return None;
        }

        let mut ctx: HipCtx = ptr::null_mut();
        if (runtime.hip_ctx_create)(&mut ctx, 0u32 as c_uint, device)
            .ok()
            .is_err()
        {
            warn_transient("hipCtxCreate failed");
            return None;
        }

        let mut stream: HipStream = ptr::null_mut();
        if (runtime.hip_stream_create)(&mut stream).ok().is_err() {
            warn_transient("hipStreamCreate failed");
            let _ = (runtime.hip_ctx_destroy)(ctx);
            return None;
        }

        Some(Arc::new(RocmContext {
            runtime,
            ctx,
            default_stream: stream,
        }))
    }
}

/// A recoverable bring-up failure. Printed under `RLX_VERBOSE` so a suite that
/// skips every ROCm test can be told apart from one that had no ROCm to begin
/// with.
fn warn_transient(what: &str) {
    if rlx_ir::env::flag("RLX_VERBOSE") {
        eprintln!(
            "rlx-rocm: {what} — treating ROCm as unavailable for now and RETRYING on the \
             next call (an idle AMD GPU can be runtime-suspended)"
        );
    }
}

/// hipBLAS handle bound to the default stream. Wrapped in a Mutex
/// because the handle's `set_stream` mutates state — multi-stream
/// dispatch will rebind per launch (same shape as rlx-cuda::cuda_blas).
pub fn rocm_blas() -> Option<Arc<Mutex<HipblasContext>>> {
    BLAS.get_or_init(|| {
        let ctx = rocm_context()?;
        let runtime = HipblasRuntime::load()?;
        let blas = HipblasContext::new(&runtime, ctx.default_stream)?;
        Some(Arc::new(Mutex::new(blas)))
    })
    .clone()
}

/// hipBLASLt handle for fused matmul + bias + relu/gelu. Falls
/// back to plain hipblas sgemm + matmul_epilogue.cu when libhipblaslt
/// isn't available.
pub fn rocm_blas_lt() -> Option<Arc<HipblasLtContext>> {
    BLAS_LT
        .get_or_init(|| {
            let _ctx = rocm_context()?;
            let runtime = HipblasLtRuntime::load()?;
            let lt = HipblasLtContext::new(&runtime)?;
            Some(Arc::new(lt))
        })
        .clone()
}

/// MIOpen handle for the conv2d fast path. Returns None on hosts
/// without libMIOpen (Mac, ROCm-less Linux) — Conv2d falls through
/// to the custom direct-conv kernel in that case.
pub fn rocm_dnn() -> Option<Arc<MiopenContext>> {
    // Escape hatch: force the im2col+GEMM conv path (MIOpen is unstable on some
    // unofficial archs, e.g. gfx1103 via HSA_OVERRIDE_GFX_VERSION).
    if rlx_ir::env::var("RLX_ROCM_DISABLE_MIOPEN").is_some() {
        return None;
    }
    DNN.get_or_init(|| {
        let ctx = rocm_context()?;
        let runtime = MiopenRuntime::load()?;
        let dnn = MiopenContext::new(&runtime, ctx.default_stream)?;
        Some(Arc::new(dnn))
    })
    .clone()
}

/// Stable label for calibration cache keys.
pub fn device_name() -> Option<String> {
    rocm_context().map(|_| "rocm-0".to_string())
}
