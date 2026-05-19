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

static CTX: OnceLock<Option<Arc<RocmContext>>> = OnceLock::new();
static BLAS: OnceLock<Option<Arc<Mutex<HipblasContext>>>> = OnceLock::new();
static BLAS_LT: OnceLock<Option<Arc<HipblasLtContext>>> = OnceLock::new();
static DNN: OnceLock<Option<Arc<MiopenContext>>> = OnceLock::new();

/// Initialise the HIP runtime + create a context on device 0 + create
/// a default stream. Returns `None` cleanly on hosts without HIP
/// (libloading fails to find `libamdhip64`) or when device 0 isn't
/// present.
pub fn rocm_context() -> Option<Arc<RocmContext>> {
    CTX.get_or_init(|| {
        let runtime = HipRuntime::load()?;
        unsafe {
            (runtime.hip_init)(0).ok().ok()?;
            let mut count: i32 = 0;
            (runtime.hip_get_device_count)(&mut count).ok().ok()?;
            if count <= 0 {
                return None;
            }

            let mut device: i32 = 0;
            (runtime.hip_device_get)(&mut device, 0).ok().ok()?;

            let mut ctx: HipCtx = ptr::null_mut();
            (runtime.hip_ctx_create)(&mut ctx, 0u32 as c_uint, device)
                .ok()
                .ok()?;

            let mut stream: HipStream = ptr::null_mut();
            (runtime.hip_stream_create)(&mut stream).ok().ok()?;

            Some(Arc::new(RocmContext {
                runtime,
                ctx,
                default_stream: stream,
            }))
        }
    })
    .clone()
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
    DNN.get_or_init(|| {
        let ctx = rocm_context()?;
        let runtime = MiopenRuntime::load()?;
        let dnn = MiopenContext::new(&runtime, ctx.default_stream)?;
        Some(Arc::new(dnn))
    })
    .clone()
}
