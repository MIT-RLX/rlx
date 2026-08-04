// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! hipBLAS shim — sgemm + sgemm_strided_batched + batched LU.
//!
//! AMD's hipBLAS deliberately mirrors cuBLAS's API surface (Hipify
//! literally substitutes `cublas` for `hipblas`), so this module is
//! a near-1:1 port of the cuBLAS bits in `rlx-cuda::backend`. Also
//! loads optional `hipblasSgetrfBatched` / `hipblasSgetrsBatched` for
//! native `Op::BatchedDenseSolve`. Resolved via libloading at runtime
//! so the crate compiles on hosts without ROCm installed.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_int, c_void};
use std::ptr;
use std::sync::Arc;

use libloading::Library;

use crate::hip::HipStream;

// ── Opaque types ─────────────────────────────────────────────────────

pub type HipblasHandle = *mut c_void;

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HipblasError(pub c_int);

impl HipblasError {
    pub fn ok(self) -> Result<(), HipblasError> {
        if self.0 == 0 { Ok(()) } else { Err(self) }
    }
}

impl std::fmt::Display for HipblasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hipblasStatus({})", self.0)
    }
}

impl std::error::Error for HipblasError {}

/// `hipblasOperation_t`. We only ever use `HIPBLAS_OP_N` (= 111 in
/// the AMD enum, matching cuBLAS's CUBLAS_OP_N = 0… actually the
/// values *do* differ between cuBLAS and hipBLAS). hipBLAS uses
/// 111 / 112 / 113 for N / T / C respectively.
#[repr(C)]
#[derive(Clone, Copy)]
pub enum HipblasOperation {
    N = 111,
    T = 112,
    C = 113,
}

/// `hipblasDatatype_t` — the values matter at the C boundary.
/// hipBLAS uses these magic numbers (matching cuBLAS / NVIDIA's
/// cudaDataType_t for binary compatibility on the call boundary).
#[repr(C)]
#[derive(Clone, Copy)]
pub enum HipblasDatatype {
    R32F = 0,   // float
    R16F = 2,   // half
    R16BF = 14, // bfloat16
}

/// `hipblasComputeType_t`.
///   COMPUTE_32F          = 0   plain f32
///   COMPUTE_32F_FAST_16F = 75  f16 inputs + f32 accumulator
///   COMPUTE_32F_FAST_16BF= 76  bf16 inputs + f32 accumulator
///   COMPUTE_32F_FAST_TF32= 74  TF32 (matrix-core) for f32 inputs
#[repr(C)]
#[derive(Clone, Copy)]
pub enum HipblasComputeType {
    F32 = 0,
    F32FastTF32 = 74,
    F32Fast16F = 75,
    F32Fast16BF = 76,
}

/// `hipblasGemmAlgo_t::HIPBLAS_GEMM_DEFAULT`.
const HIPBLAS_GEMM_DEFAULT: c_int = 160;

// ── Function-pointer signatures ──────────────────────────────────────

type FnHipblasCreate = unsafe extern "C" fn(*mut HipblasHandle) -> HipblasError;
type FnHipblasDestroy = unsafe extern "C" fn(HipblasHandle) -> HipblasError;
type FnHipblasSetStream = unsafe extern "C" fn(HipblasHandle, HipStream) -> HipblasError;
/// `hipblasSetMathMode` — accepts a `hipblasMath_t` discriminant.
/// We pass `HIPBLAS_XF32_XDL_MATH` (= 4) to enable TF32-equivalent
/// matrix-core acceleration on f32 sgemm calls. Matches what
/// rlx-cuda does via cublasGemmEx + COMPUTE_32F_FAST_TF32.
type FnHipblasSetMathMode = unsafe extern "C" fn(HipblasHandle, c_int) -> HipblasError;
const HIPBLAS_XF32_XDL_MATH: c_int = 4;
type FnHipblasSgemm = unsafe extern "C" fn(
    HipblasHandle,
    HipblasOperation,
    HipblasOperation,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const f32,
    c_int,
    *const f32,
    c_int,
    *const f32,
    *mut f32,
    c_int,
) -> HipblasError;
type FnHipblasSgemmStridedBatched = unsafe extern "C" fn(
    HipblasHandle,
    HipblasOperation,
    HipblasOperation,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const f32,
    c_int,
    i64,
    *const f32,
    c_int,
    i64,
    *const f32,
    *mut f32,
    c_int,
    i64,
    c_int,
) -> HipblasError;

/// `hipblasGemmEx` — mixed-precision GEMM. Same shape as
/// cublasGemmEx in rlx-cuda's mixed-precision tier.
type FnHipblasGemmEx = unsafe extern "C" fn(
    HipblasHandle,
    HipblasOperation,
    HipblasOperation,
    c_int,
    c_int,
    c_int,
    *const c_void, // alpha
    *const c_void,
    HipblasDatatype,
    c_int, // A, A_type, lda
    *const c_void,
    HipblasDatatype,
    c_int,         // B, B_type, ldb
    *const c_void, // beta
    *mut c_void,
    HipblasDatatype,
    c_int, // C, C_type, ldc
    HipblasComputeType,
    c_int, // algo
) -> HipblasError;

type FnHipblasSgetrfBatched = unsafe extern "C" fn(
    HipblasHandle,
    c_int,
    *const *mut f32,
    c_int,
    *mut c_int,
    *mut c_int,
    c_int,
) -> HipblasError;
type FnHipblasSgetrsBatched = unsafe extern "C" fn(
    HipblasHandle,
    HipblasOperation,
    c_int,
    c_int,
    *const *const f32,
    c_int,
    *const c_int,
    *const *mut f32,
    c_int,
    *mut c_int,
    c_int,
) -> HipblasError;

// ── Loaded runtime ───────────────────────────────────────────────────

pub struct HipblasRuntime {
    _lib: Library,
    pub create: FnHipblasCreate,
    pub destroy: FnHipblasDestroy,
    pub set_stream: FnHipblasSetStream,
    pub set_math_mode: FnHipblasSetMathMode,
    pub sgemm: FnHipblasSgemm,
    pub sgemm_strided: FnHipblasSgemmStridedBatched,
    pub gemm_ex: FnHipblasGemmEx,
    /// Optional — missing on very old hipBLAS; batched dense solve falls back.
    pub sgetrf_batched: Option<FnHipblasSgetrfBatched>,
    pub sgetrs_batched: Option<FnHipblasSgetrsBatched>,
}

/// Public re-export of the `HIPBLAS_GEMM_DEFAULT` algo selector.
pub fn hipblas_gemm_default() -> c_int {
    HIPBLAS_GEMM_DEFAULT
}

unsafe impl Send for HipblasRuntime {}
unsafe impl Sync for HipblasRuntime {}

impl HipblasRuntime {
    pub fn load() -> Option<Arc<Self>> {
        unsafe {
            let lib = Library::new("libhipblas.so")
                .or_else(|_| Library::new("libhipblas.so.2"))
                .or_else(|_| Library::new("libhipblas.so.1"))
                .or_else(|_| Library::new("libhipblas.so.0"))
                .ok()?;
            macro_rules! sym {
                ($name:literal, $ty:ty) => {{
                    let s: libloading::Symbol<$ty> = lib.get($name).ok()?;
                    *s.into_raw()
                }};
            }
            macro_rules! sym_opt {
                ($name:literal, $ty:ty) => {{ lib.get::<$ty>($name).ok().map(|s| *s.into_raw()) }};
            }
            let rt = HipblasRuntime {
                create: sym!(b"hipblasCreate", FnHipblasCreate),
                destroy: sym!(b"hipblasDestroy", FnHipblasDestroy),
                set_stream: sym!(b"hipblasSetStream", FnHipblasSetStream),
                set_math_mode: sym!(b"hipblasSetMathMode", FnHipblasSetMathMode),
                sgemm: sym!(b"hipblasSgemm", FnHipblasSgemm),
                sgemm_strided: sym!(b"hipblasSgemmStridedBatched", FnHipblasSgemmStridedBatched),
                gemm_ex: sym!(b"hipblasGemmEx", FnHipblasGemmEx),
                sgetrf_batched: sym_opt!(b"hipblasSgetrfBatched", FnHipblasSgetrfBatched),
                sgetrs_batched: sym_opt!(b"hipblasSgetrsBatched", FnHipblasSgetrsBatched),
                _lib: lib,
            };
            Some(Arc::new(rt))
        }
    }
}

/// True when batched LU symbols are present on the loaded hipBLAS.
pub fn batched_lu_available(rt: &HipblasRuntime) -> bool {
    rt.sgetrf_batched.is_some() && rt.sgetrs_batched.is_some()
}

/// hipBLAS handle bound to a stream. Mirrors `cudarc::cublas::CudaBlas`
/// shape — Drop releases via `hipblasDestroy`.
pub struct HipblasContext {
    pub runtime: Arc<HipblasRuntime>,
    pub handle: HipblasHandle,
}

unsafe impl Send for HipblasContext {}
unsafe impl Sync for HipblasContext {}

impl HipblasContext {
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // stream is opaque; we only pass it to FFI
    pub fn new(runtime: &Arc<HipblasRuntime>, stream: HipStream) -> Option<Self> {
        unsafe {
            let mut handle: HipblasHandle = ptr::null_mut();
            (runtime.create)(&mut handle).ok().ok()?;
            (runtime.set_stream)(handle, stream).ok().ok()?;
            // Best-effort TF32 enable for sgemm. Matrix-core archs
            // (CDNA + RDNA3+) accelerate f32 GEMM through xDL math
            // when this is set; older archs ignore it. Matches what
            // rlx-cuda does for cuBLAS via cublasSetMathMode.
            //
            // xDL/XF32 truncates both f32 operands to ~TF32 mantissa, which is
            // a real precision loss for the dequant→sgemm path: the dequantized
            // GGUF/int8 weights carry >10 effective mantissa bits after scaling
            // that XF32 discards. Default stays ON (peak perf, matches cuBLAS),
            // but a strict-parity run can force true f32 via RLX_ROCM_NO_TF32 /
            // RLX_ROCM_PARITY — the escape hatch rlx-cuda has (RLX_CUDA_NO_TF32/
            // PARITY) that ROCm was missing.
            if !rlx_ir::env::flag("RLX_ROCM_NO_TF32") && !rlx_ir::env::flag("RLX_ROCM_PARITY") {
                let _ = (runtime.set_math_mode)(handle, HIPBLAS_XF32_XDL_MATH);
            }
            Some(Self {
                runtime: runtime.clone(),
                handle,
            })
        }
    }

    /// Re-bind the handle to a different stream. Used when
    /// MultiStream(n) dispatches a step on a non-default pool stream
    /// and needs the hipBLAS internal kernel launches to follow.
    ///
    /// # Safety
    ///
    /// `stream` must be non-null and belong to the same HIP context
    /// that this handle was created against. Caller must hold the
    /// surrounding `HipblasContext`'s lock so concurrent calls don't
    /// race on the handle's stream binding.
    pub unsafe fn set_stream(&self, stream: HipStream) -> Result<(), HipblasError> {
        unsafe { (self.runtime.set_stream)(self.handle, stream).ok() }
    }
}

impl Drop for HipblasContext {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = (self.runtime.destroy)(self.handle);
            }
        }
    }
}
