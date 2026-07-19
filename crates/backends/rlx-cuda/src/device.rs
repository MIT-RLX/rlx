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
use cudarc::driver::{CudaContext, CudaSlice};

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

/// Best-effort preload of a real libcudnn so cudarc's soname-based loader binds
/// to it even when cuDNN is present only via a pip/conda wheel
/// (`nvidia-cudnn-cuXX`) or PyTorch, rather than on the system loader path — a
/// very common setup. The pip cuDNN carries `RPATH=$ORIGIN`, so `dlopen`-ing
/// `libcudnn.so.9` by its full path auto-resolves the sub-libs
/// (`libcudnn_cnn`, `_ops`, …) and `cublas` from the sibling wheel dirs; no
/// `LD_LIBRARY_PATH` juggling required. Without this, cudarc dlopens the bare
/// soname, misses the wheel, and every convolution falls to the ~10× slower
/// im2col path. Search order: `RLX_CUDNN_DIR` (explicit override) →
/// `$CONDA_PREFIX`/`$VIRTUAL_ENV` (`lib/` + pip `site-packages/nvidia/cudnn/lib`)
/// → `$LD_LIBRARY_PATH`. No-op when none is found (cudarc then reports cuDNN
/// unavailable and we keep the graceful im2col fallback), or when a system
/// cuDNN is already on the path (this only *adds* a discovery route).
#[cfg(unix)]
fn preload_real_cudnn() {
    use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
    use std::path::PathBuf;

    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(d) = std::env::var("RLX_CUDNN_DIR") {
        dirs.push(PathBuf::from(d));
    }
    for var in ["CONDA_PREFIX", "VIRTUAL_ENV"] {
        if let Ok(p) = std::env::var(var) {
            let lib = PathBuf::from(&p).join("lib");
            // pip installs land in <prefix>/lib/pythonX.Y/site-packages/nvidia/cudnn/lib
            if let Ok(rd) = std::fs::read_dir(&lib) {
                for e in rd.flatten() {
                    dirs.push(e.path().join("site-packages/nvidia/cudnn/lib"));
                }
            }
            dirs.push(lib);
        }
    }
    if let Ok(lp) = std::env::var("LD_LIBRARY_PATH") {
        dirs.extend(lp.split(':').filter(|s| !s.is_empty()).map(PathBuf::from));
    }

    for dir in &dirs {
        for name in ["libcudnn.so.9", "libcudnn.so.8", "libcudnn.so"] {
            let full = dir.join(name);
            if full.is_file() {
                // SAFETY: dlopen of a discovered cuDNN; leaked so it stays resident
                // for cudarc's subsequent soname load to bind against.
                unsafe {
                    if let Ok(lib) = Library::open(Some(&full), RTLD_NOW | RTLD_GLOBAL) {
                        std::mem::forget(lib);
                        return;
                    }
                }
            }
        }
    }
}

#[cfg(not(unix))]
fn preload_real_cudnn() {}

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
            preload_real_cudnn();
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

pub const CUBLASLT_WORKSPACE_BYTES: usize = 4 * 1024 * 1024;
// cuDNN conv scratch, shared once per process. Works with the
// fastest-algo-that-fits selection in `backend.rs` (`pick_conv_*_algo`): that
// alone stops the old im2col cliff (IMPLICIT_GEMM needs ~0 workspace, so a
// fitting algo always exists). This larger 256 MiB budget (was 32 MiB — too
// small for the fastest algo at batch ≥ 512) additionally lets a *faster* algo
// (Winograd/FFT/GEMM) fit at large batch instead of a slower low-workspace one.
// Still negligible against modern GPU VRAM.
pub const CUDNN_WORKSPACE_BYTES: usize = 256 * 1024 * 1024;

static BLAS_LT_WORKSPACE: OnceLock<Option<Arc<Mutex<CudaSlice<u8>>>>> = OnceLock::new();
static DNN_WORKSPACE: OnceLock<Option<Arc<Mutex<CudaSlice<u8>>>>> = OnceLock::new();

/// Shared cuBLASLt scratch (4 MiB). Allocated once per process on first conv/matmul use.
pub fn cuda_blas_lt_workspace() -> Option<Arc<Mutex<CudaSlice<u8>>>> {
    BLAS_LT_WORKSPACE
        .get_or_init(|| {
            cuda_blas_lt_handle()?;
            let ctx = cuda_context()?;
            ctx.default_stream()
                .alloc_zeros::<u8>(CUBLASLT_WORKSPACE_BYTES)
                .ok()
                .map(|buf| Arc::new(Mutex::new(buf)))
        })
        .clone()
}

/// Shared cuDNN scratch (32 MiB). Allocated once per process on first conv use.
pub fn cuda_dnn_workspace() -> Option<Arc<Mutex<CudaSlice<u8>>>> {
    DNN_WORKSPACE
        .get_or_init(|| {
            cuda_dnn_handle()?;
            let ctx = cuda_context()?;
            ctx.default_stream()
                .alloc_zeros::<u8>(CUDNN_WORKSPACE_BYTES)
                .ok()
                .map(|buf| Arc::new(Mutex::new(buf)))
        })
        .clone()
}

/// Stable label for calibration cache keys.
pub fn device_name() -> Option<String> {
    cuda_context().map(|_| "cuda-0".to_string())
}
