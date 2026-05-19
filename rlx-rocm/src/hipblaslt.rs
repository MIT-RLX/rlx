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

//! hipBLASLt shim — fused matmul + bias + activation epilogue.
//!
//! AMD's hipBLASLt deliberately mirrors cuBLASLt's API. The only
//! differences worth tracking:
//!
//!   * Symbol prefix is `hipblasLt` instead of `cublasLt`.
//!   * Layout / desc / pref / handle types are opaque pointers in
//!     both — same shape.
//!   * Matmul desc attribute / matrix layout attribute / pref
//!     attribute enums use the same numeric values as cuBLASLt
//!     for binary compatibility on the call boundary.
//!   * Epilogue enum values match (DEFAULT=1, BIAS=4, RELU=8,
//!     RELU_BIAS=12, GELU=16, GELU_BIAS=20).
//!
//! Bounded scope: f32 inputs + f32 output + RELU / GELU epilogue
//! fusion. Mixed-precision GemmEx is hipblas's job (already wired).

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_int, c_void};
use std::ptr;
use std::sync::Arc;

use libloading::Library;

use crate::hip::HipStream;

pub type HipblasLtHandle = *mut c_void;
pub type HipblasLtMatrixLayout = *mut c_void;
pub type HipblasLtMatmulDesc = *mut c_void;
pub type HipblasLtMatmulPref = *mut c_void;

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HipblasLtError(pub c_int);

impl HipblasLtError {
    pub fn ok(self) -> Result<(), HipblasLtError> {
        if self.0 == 0 { Ok(()) } else { Err(self) }
    }
}

impl std::fmt::Display for HipblasLtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hipblasLtStatus({})", self.0)
    }
}

impl std::error::Error for HipblasLtError {}

/// hipBLASLt epilogue values (matching cuBLASLt's
/// `cublasLtEpilogue_t` numeric layout).
#[repr(C)]
#[derive(Clone, Copy)]
pub enum HipblasLtEpilogue {
    Default = 1,
    Bias = 4,
    Relu = 8,
    ReluBias = 12,
    Gelu = 16,
    GeluBias = 20,
}

/// Common compute / data type / operation values mirror hipBLAS's
/// (so we re-use the existing constants rather than duplicate).
const HIPBLAS_OP_N: c_int = 111;
const HIPBLAS_R_32F: c_int = 0;
const HIPBLAS_COMPUTE_32F_FAST_TF32: c_int = 74;

/// Attribute IDs (matching cuBLASLt for binary-compatibility on the
/// `setAttribute` call boundary).
const ATTR_EPILOGUE: c_int = 9;
const ATTR_BIAS_POINTER: c_int = 10;
const ATTR_PREF_MAX_WORKSPACE_BYTES: c_int = 0;
/// Matrix-layout attributes for strided-batched matmul.
const LAYOUT_ATTR_BATCH_COUNT: c_int = 4;
const LAYOUT_ATTR_STRIDED_BATCH_OFFSET: c_int = 5;

// ── Function-pointer signatures ──────────────────────────────────────

type FnHipblasLtCreate = unsafe extern "C" fn(*mut HipblasLtHandle) -> HipblasLtError;
type FnHipblasLtDestroy = unsafe extern "C" fn(HipblasLtHandle) -> HipblasLtError;

type FnHipblasLtMatrixLayoutCreate =
    unsafe extern "C" fn(*mut HipblasLtMatrixLayout, c_int, u64, u64, i64) -> HipblasLtError;
type FnHipblasLtMatrixLayoutSetAttribute =
    unsafe extern "C" fn(HipblasLtMatrixLayout, c_int, *const c_void, usize) -> HipblasLtError;
type FnHipblasLtMatrixLayoutDestroy = unsafe extern "C" fn(HipblasLtMatrixLayout) -> HipblasLtError;

type FnHipblasLtMatmulDescCreate =
    unsafe extern "C" fn(*mut HipblasLtMatmulDesc, c_int, c_int) -> HipblasLtError;
type FnHipblasLtMatmulDescSetAttribute =
    unsafe extern "C" fn(HipblasLtMatmulDesc, c_int, *const c_void, usize) -> HipblasLtError;
type FnHipblasLtMatmulDescDestroy = unsafe extern "C" fn(HipblasLtMatmulDesc) -> HipblasLtError;

type FnHipblasLtMatmulPrefCreate = unsafe extern "C" fn(*mut HipblasLtMatmulPref) -> HipblasLtError;
type FnHipblasLtMatmulPrefSetAttribute =
    unsafe extern "C" fn(HipblasLtMatmulPref, c_int, *const c_void, usize) -> HipblasLtError;
type FnHipblasLtMatmulPrefDestroy = unsafe extern "C" fn(HipblasLtMatmulPref) -> HipblasLtError;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HipblasLtMatmulHeuristicResult {
    pub algo: [u8; 80], // hipblasLtMatmulAlgo_t opaque blob
    pub workspace_size: usize,
    pub state: c_int,
    pub waves_count: f32,
    pub reserved: [c_int; 4],
}

type FnHipblasLtMatmulAlgoGetHeuristic = unsafe extern "C" fn(
    HipblasLtHandle,
    HipblasLtMatmulDesc,
    HipblasLtMatrixLayout,
    HipblasLtMatrixLayout,
    HipblasLtMatrixLayout,
    HipblasLtMatrixLayout,
    HipblasLtMatmulPref,
    c_int,
    *mut HipblasLtMatmulHeuristicResult,
    *mut c_int,
) -> HipblasLtError;

type FnHipblasLtMatmul = unsafe extern "C" fn(
    HipblasLtHandle,
    HipblasLtMatmulDesc,
    *const c_void,
    *const c_void,
    HipblasLtMatrixLayout,
    *const c_void,
    HipblasLtMatrixLayout,
    *const c_void,
    *const c_void,
    HipblasLtMatrixLayout,
    *mut c_void,
    HipblasLtMatrixLayout,
    *const u8, // algo blob
    *mut c_void,
    usize, // workspace
    HipStream,
) -> HipblasLtError;

pub struct HipblasLtRuntime {
    _lib: Library,
    pub create: FnHipblasLtCreate,
    pub destroy: FnHipblasLtDestroy,
    pub matrix_layout_create: FnHipblasLtMatrixLayoutCreate,
    pub matrix_layout_set_attr: FnHipblasLtMatrixLayoutSetAttribute,
    pub matrix_layout_destroy: FnHipblasLtMatrixLayoutDestroy,
    pub matmul_desc_create: FnHipblasLtMatmulDescCreate,
    pub matmul_desc_set_attr: FnHipblasLtMatmulDescSetAttribute,
    pub matmul_desc_destroy: FnHipblasLtMatmulDescDestroy,
    pub pref_create: FnHipblasLtMatmulPrefCreate,
    pub pref_set_attr: FnHipblasLtMatmulPrefSetAttribute,
    pub pref_destroy: FnHipblasLtMatmulPrefDestroy,
    pub algo_get_heuristic: FnHipblasLtMatmulAlgoGetHeuristic,
    pub matmul: FnHipblasLtMatmul,
}

unsafe impl Send for HipblasLtRuntime {}
unsafe impl Sync for HipblasLtRuntime {}

impl HipblasLtRuntime {
    pub fn load() -> Option<Arc<Self>> {
        unsafe {
            let lib = Library::new("libhipblaslt.so")
                .or_else(|_| Library::new("libhipblaslt.so.0"))
                .ok()?;
            macro_rules! sym {
                ($name:literal, $ty:ty) => {{
                    let s: libloading::Symbol<$ty> = lib.get($name).ok()?;
                    *s.into_raw()
                }};
            }
            let rt = HipblasLtRuntime {
                create: sym!(b"hipblasLtCreate", FnHipblasLtCreate),
                destroy: sym!(b"hipblasLtDestroy", FnHipblasLtDestroy),
                matrix_layout_create: sym!(
                    b"hipblasLtMatrixLayoutCreate",
                    FnHipblasLtMatrixLayoutCreate
                ),
                matrix_layout_set_attr: sym!(
                    b"hipblasLtMatrixLayoutSetAttribute",
                    FnHipblasLtMatrixLayoutSetAttribute
                ),
                matrix_layout_destroy: sym!(
                    b"hipblasLtMatrixLayoutDestroy",
                    FnHipblasLtMatrixLayoutDestroy
                ),
                matmul_desc_create: sym!(b"hipblasLtMatmulDescCreate", FnHipblasLtMatmulDescCreate),
                matmul_desc_set_attr: sym!(
                    b"hipblasLtMatmulDescSetAttribute",
                    FnHipblasLtMatmulDescSetAttribute
                ),
                matmul_desc_destroy: sym!(
                    b"hipblasLtMatmulDescDestroy",
                    FnHipblasLtMatmulDescDestroy
                ),
                pref_create: sym!(
                    b"hipblasLtMatmulPreferenceCreate",
                    FnHipblasLtMatmulPrefCreate
                ),
                pref_set_attr: sym!(
                    b"hipblasLtMatmulPreferenceSetAttribute",
                    FnHipblasLtMatmulPrefSetAttribute
                ),
                pref_destroy: sym!(
                    b"hipblasLtMatmulPreferenceDestroy",
                    FnHipblasLtMatmulPrefDestroy
                ),
                algo_get_heuristic: sym!(
                    b"hipblasLtMatmulAlgoGetHeuristic",
                    FnHipblasLtMatmulAlgoGetHeuristic
                ),
                matmul: sym!(b"hipblasLtMatmul", FnHipblasLtMatmul),
                _lib: lib,
            };
            Some(Arc::new(rt))
        }
    }
}

pub struct HipblasLtContext {
    pub runtime: Arc<HipblasLtRuntime>,
    pub handle: HipblasLtHandle,
}

unsafe impl Send for HipblasLtContext {}
unsafe impl Sync for HipblasLtContext {}

impl HipblasLtContext {
    pub fn new(runtime: &Arc<HipblasLtRuntime>) -> Option<Self> {
        unsafe {
            let mut handle: HipblasLtHandle = ptr::null_mut();
            (runtime.create)(&mut handle).ok().ok()?;
            Some(Self {
                runtime: runtime.clone(),
                handle,
            })
        }
    }
}

impl Drop for HipblasLtContext {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = (self.runtime.destroy)(self.handle);
            }
        }
    }
}

/// Map activation id (rlx-cuda's `act_id` table) to a hipBLASLt
/// epilogue. Returns `None` for activations hipBLASLt doesn't fuse.
pub fn epilogue_for(act_id: u32, has_bias: bool) -> Option<HipblasLtEpilogue> {
    match (act_id, has_bias) {
        (0xFFFFu32, true) => Some(HipblasLtEpilogue::Bias),
        (0xFFFFu32, false) => Some(HipblasLtEpilogue::Default),
        (0, true) => Some(HipblasLtEpilogue::ReluBias),
        (0, false) => Some(HipblasLtEpilogue::Relu),
        (9 | 11, true) => Some(HipblasLtEpilogue::GeluBias),
        (9 | 11, false) => Some(HipblasLtEpilogue::Gelu),
        _ => None,
    }
}

/// True iff hipBLASLt natively fuses this activation (relu / gelu /
/// identity). Other activations need the matmul_epilogue kernel.
pub fn act_supported(act_id: u32) -> bool {
    matches!(act_id, 0xFFFFu32 | 0 | 9 | 11)
}

/// Single fused matmul + bias + relu/gelu launch via hipBLASLt.
/// Returns `Err` on any setup or runtime failure so the caller can
/// fall through to plain hipBLAS sgemm + epilogue kernel.
///
/// # Safety
///
/// `arena_dev_ptr`, `workspace_dev_ptr`, and the offsets into them
/// (`a_off_f32`, `b_off_f32`, `c_off_f32`, `bias_off_f32`) must point
/// to valid HIP device memory of the right shape and lifetime. `lt`
/// must be a live `HipblasLtContext`, and `stream` non-null and
/// bound to the same HIP context as `lt.handle`. Caller is
/// responsible for ensuring the workspace size doesn't lie about
/// the buffer's actual byte length.
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_fused(
    lt: &HipblasLtContext,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    m: u32,
    k: u32,
    n: u32,
    a_off_f32: u32,
    b_off_f32: u32,
    c_off_f32: u32,
    has_bias: bool,
    bias_off_f32: u32,
    epilogue: HipblasLtEpilogue,
    batch: u32,
    a_batch_stride: u32,
    b_batch_stride: u32,
    c_batch_stride: u32,
    stream: HipStream,
) -> Result<(), HipblasLtError> {
    use core::mem;

    let rt = &lt.runtime;

    // A↔B swap so the column-major view computes our row-major matmul.
    let a_ptr = (arena_dev_ptr + (b_off_f32 as u64) * 4) as *const c_void;
    let b_ptr = (arena_dev_ptr + (a_off_f32 as u64) * 4) as *const c_void;
    let c_ptr = (arena_dev_ptr + (c_off_f32 as u64) * 4) as *const c_void;
    let d_ptr = c_ptr as *mut c_void;

    unsafe {
        let mut a_layout: HipblasLtMatrixLayout = ptr::null_mut();
        let mut b_layout: HipblasLtMatrixLayout = ptr::null_mut();
        let mut c_layout: HipblasLtMatrixLayout = ptr::null_mut();
        let mut desc: HipblasLtMatmulDesc = ptr::null_mut();
        let mut pref: HipblasLtMatmulPref = ptr::null_mut();

        (rt.matrix_layout_create)(&mut a_layout, HIPBLAS_R_32F, n as u64, k as u64, n as i64)
            .ok()?;
        (rt.matrix_layout_create)(&mut b_layout, HIPBLAS_R_32F, k as u64, m as u64, k as i64)
            .ok()?;
        (rt.matrix_layout_create)(&mut c_layout, HIPBLAS_R_32F, n as u64, m as u64, n as i64)
            .ok()?;
        (rt.matmul_desc_create)(&mut desc, HIPBLAS_COMPUTE_32F_FAST_TF32, HIPBLAS_R_32F).ok()?;

        // Strided-batch: set BATCH_COUNT + STRIDED_BATCH_OFFSET on
        // every layout. Same A↔B swap as the inner shape — the
        // strides we pass for A_lt are our B-side strides and vice
        // versa.
        if batch > 1 {
            let bc = batch as c_int;
            for &layout in &[a_layout, b_layout, c_layout] {
                (rt.matrix_layout_set_attr)(
                    layout,
                    LAYOUT_ATTR_BATCH_COUNT,
                    &bc as *const _ as *const _,
                    mem::size_of::<c_int>(),
                )
                .ok()?;
            }
            let stride_a_lt = b_batch_stride as i64;
            let stride_b_lt = a_batch_stride as i64;
            let stride_c_lt = c_batch_stride as i64;
            (rt.matrix_layout_set_attr)(
                a_layout,
                LAYOUT_ATTR_STRIDED_BATCH_OFFSET,
                &stride_a_lt as *const _ as *const _,
                mem::size_of::<i64>(),
            )
            .ok()?;
            (rt.matrix_layout_set_attr)(
                b_layout,
                LAYOUT_ATTR_STRIDED_BATCH_OFFSET,
                &stride_b_lt as *const _ as *const _,
                mem::size_of::<i64>(),
            )
            .ok()?;
            (rt.matrix_layout_set_attr)(
                c_layout,
                LAYOUT_ATTR_STRIDED_BATCH_OFFSET,
                &stride_c_lt as *const _ as *const _,
                mem::size_of::<i64>(),
            )
            .ok()?;
        }

        (rt.matmul_desc_set_attr)(
            desc,
            ATTR_EPILOGUE,
            &epilogue as *const _ as *const _,
            mem::size_of::<HipblasLtEpilogue>(),
        )
        .ok()?;
        if has_bias {
            let bias_dev = arena_dev_ptr + (bias_off_f32 as u64) * 4;
            (rt.matmul_desc_set_attr)(
                desc,
                ATTR_BIAS_POINTER,
                &bias_dev as *const _ as *const _,
                mem::size_of::<u64>(),
            )
            .ok()?;
        }

        (rt.pref_create)(&mut pref).ok()?;
        (rt.pref_set_attr)(
            pref,
            ATTR_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size as *const _ as *const _,
            mem::size_of::<usize>(),
        )
        .ok()?;

        let mut heuristic = mem::MaybeUninit::<HipblasLtMatmulHeuristicResult>::uninit();
        let mut returned: c_int = 0;
        (rt.algo_get_heuristic)(
            lt.handle,
            desc,
            a_layout,
            b_layout,
            c_layout,
            c_layout,
            pref,
            1,
            heuristic.as_mut_ptr(),
            &mut returned,
        )
        .ok()?;
        if returned == 0 {
            let _ = (rt.pref_destroy)(pref);
            let _ = (rt.matmul_desc_destroy)(desc);
            let _ = (rt.matrix_layout_destroy)(c_layout);
            let _ = (rt.matrix_layout_destroy)(b_layout);
            let _ = (rt.matrix_layout_destroy)(a_layout);
            return Err(HipblasLtError(-1));
        }
        let heuristic = heuristic.assume_init();

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let workspace_ptr = workspace_dev_ptr as *mut c_void;
        let result = (rt.matmul)(
            lt.handle,
            desc,
            &alpha as *const _ as *const c_void,
            a_ptr,
            a_layout,
            b_ptr,
            b_layout,
            &beta as *const _ as *const c_void,
            c_ptr,
            c_layout,
            d_ptr,
            c_layout,
            heuristic.algo.as_ptr(),
            workspace_ptr,
            workspace_size,
            stream,
        );

        let _ = (rt.pref_destroy)(pref);
        let _ = (rt.matmul_desc_destroy)(desc);
        let _ = (rt.matrix_layout_destroy)(c_layout);
        let _ = (rt.matrix_layout_destroy)(b_layout);
        let _ = (rt.matrix_layout_destroy)(a_layout);

        result.ok()
    }
}
