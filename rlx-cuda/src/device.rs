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

//! Per-process CUDA context singleton.
//!
//! `cudarc::driver::CudaContext` owns the underlying CUcontext + the
//! default stream we use for every dispatch. We hold one in a static
//! `OnceLock`; if libcuda fails to load (e.g., when running on Mac),
//! the `OnceLock` resolves to `None` and `is_available()` reports false.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;

use cudarc::cublas::CudaBlas;
use cudarc::cublaslt::sys as cublaslt_sys;
use cudarc::cudnn::sys as cudnn_sys;
use cudarc::driver::CudaContext;

static CTX: OnceLock<Option<Arc<CudaContext>>> = OnceLock::new();
static BLAS: OnceLock<Option<Arc<Mutex<CudaBlas>>>> = OnceLock::new();
static BLAS_LT_HANDLE: OnceLock<Option<usize>> = OnceLock::new();
static DNN_HANDLE: OnceLock<Option<usize>> = OnceLock::new();

/// Initialise (once) and return the CUDA context Arc, or `None` if the
/// driver couldn't be loaded. cudarc unconditionally panics when the
/// `dynamic-loading` path can't find `libcuda`, so we wrap the call in
/// `catch_unwind` to treat that as "no driver available" instead of a
/// process-level failure. Lets the crate run on Mac and any other host
/// without CUDA — useful for compile-check + IR-lowering unit tests.
pub fn cuda_context() -> Option<Arc<CudaContext>> {
    CTX.get_or_init(|| {
        // Suppress the libcuda-load panic message on stderr — there's
        // no way to dampen a panic's print, but we silence the default
        // panic hook for the duration of this attempt so a missing
        // driver doesn't generate stderr spam during cargo test.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| CudaContext::new(0)));
        std::panic::set_hook(prev);
        match result {
            Ok(Ok(ctx)) => Some(ctx),
            _ => None,
        }
    })
    .clone()
}

/// cuBLAS handle bound to the same default stream as the context. Wrapped
/// in a `Mutex` because cuBLAS calls aren't `Sync` even though our usage
/// is single-threaded; the Mutex makes the static safe to share.
pub fn cuda_blas() -> Option<Arc<Mutex<CudaBlas>>> {
    BLAS.get_or_init(|| {
        let ctx = cuda_context()?;
        let stream = ctx.default_stream();
        CudaBlas::new(stream).ok().map(|b| Arc::new(Mutex::new(b)))
    })
    .clone()
}

/// cuBLASLt handle (raw `cublasLtHandle_t` cast to `usize` for `OnceLock`
/// compatibility — the type is `*mut cublasLtContext`, not `Send`/`Sync`
/// by default but our usage is single-threaded). Lazily created; returns
/// `None` if the driver isn't available or handle creation fails.
pub fn cuda_blas_lt_handle() -> Option<cublaslt_sys::cublasLtHandle_t> {
    BLAS_LT_HANDLE
        .get_or_init(|| {
            let _ctx = cuda_context()?;
            let handle = cudarc::cublaslt::result::create_handle().ok()?;
            Some(handle as usize)
        })
        .map(|h| h as cublaslt_sys::cublasLtHandle_t)
}

/// cuDNN handle bound to the default stream. Same usize-cast trick as
/// cuda_blas_lt_handle for `OnceLock` compatibility. Returns `None` if
/// libcudnn isn't loadable or handle creation fails (graceful fallback
/// to the custom direct-convolution kernels in that case).
///
/// Wrapped in `catch_unwind` for the same reason `cuda_context` is:
/// cudarc's `dynamic-loading` path panics rather than returns `Err`
/// when libcudnn can't be `dlopen`'d, so we have to catch the panic
/// to keep `is_available()` behaviour clean on hosts without cuDNN.
pub fn cuda_dnn_handle() -> Option<cudnn_sys::cudnnHandle_t> {
    DNN_HANDLE
        .get_or_init(|| {
            let ctx = cuda_context()?;
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                let handle = cudarc::cudnn::result::create_handle().ok()?;
                unsafe {
                    let stream = ctx.default_stream();
                    cudarc::cudnn::result::set_stream(
                        handle,
                        stream.cu_stream() as cudnn_sys::cudaStream_t,
                    )
                    .ok()?;
                }
                Some(handle as usize)
            }));
            std::panic::set_hook(prev);
            result.ok().flatten()
        })
        .map(|h| h as cudnn_sys::cudnnHandle_t)
}
