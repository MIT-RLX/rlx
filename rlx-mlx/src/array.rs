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

//! RAII wrapper over the opaque `mlx_array_t` shim handle.

use std::ffi::CStr;
use std::ptr;

use rlx_ir::DType;

use crate::ffi::{self, MlxDtype, RLX_MLX_OK, mlx_array_t};

/// An MLX array — owns the underlying handle. `Drop` calls
/// `rlx_mlx_array_free`. Can be cloned (cheap — increments MLX's
/// internal refcount via a fresh shared_ptr-backed handle).
pub struct Array {
    pub(crate) ptr: *mut mlx_array_t,
}

impl Array {
    /// Construct an MLX leaf from host f32 data, casting to `dtype`.
    pub fn from_f32_slice(data: &[f32], shape: &[usize], dtype: DType) -> Result<Self, MlxError> {
        let shape_i: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        let mut out: *mut mlx_array_t = ptr::null_mut();
        let rc = unsafe {
            ffi::rlx_mlx_array_from_data(
                shape_i.as_ptr(),
                shape_i.len(),
                data.as_ptr(),
                data.len(),
                map_dtype(dtype),
                &mut out,
            )
        };
        check(rc)?;
        Ok(Self { ptr: out })
    }

    /// Read the array's contents back as f32, evaluating if needed.
    pub fn to_f32(&self) -> Result<Vec<f32>, MlxError> {
        let nelems = self.num_elements()?;
        let mut buf = vec![0f32; nelems];
        let rc = unsafe { ffi::rlx_mlx_array_to_f32(self.ptr, buf.as_mut_ptr(), nelems) };
        check(rc)?;
        Ok(buf)
    }

    /// Build a leaf array directly from raw bytes in the target
    /// dtype — no f32 widen/narrow round-trip. Useful when callers
    /// already hold half-precision (F16/BF16) buffers.
    pub fn from_bytes(data: &[u8], shape: &[usize], dtype: DType) -> Result<Self, MlxError> {
        let shape_i: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        let mut out: *mut mlx_array_t = std::ptr::null_mut();
        let rc = unsafe {
            ffi::rlx_mlx_array_from_bytes(
                shape_i.as_ptr(),
                shape_i.len(),
                data.as_ptr() as *const std::ffi::c_void,
                data.len(),
                map_dtype(dtype),
                &mut out,
            )
        };
        check(rc)?;
        Ok(Self { ptr: out })
    }

    /// Read the array's contents as raw bytes in its native dtype.
    /// No automatic conversion to f32 — pair with `from_bytes` for
    /// round-trip-free F16/BF16 I/O.
    pub fn to_bytes(&self) -> Result<Vec<u8>, MlxError> {
        let nelems = self.num_elements()?;
        // Worst-case dtype width is f64 / i64 (8 B/elem). We don't
        // expose the array's actual dtype to Rust today, so the
        // shim writes its native byte count into `written` and we
        // truncate. Allocating for f32 (4 B/elem) under-fits F64
        // and tripped "dst buffer too small" when downstream f64
        // custom-op kernels read a sparse-LU output through this.
        let mut buf = vec![0u8; nelems * 8];
        let mut written = 0usize;
        let rc = unsafe {
            ffi::rlx_mlx_array_to_bytes(
                self.ptr,
                buf.as_mut_ptr() as *mut std::ffi::c_void,
                buf.len(),
                &mut written,
            )
        };
        check(rc)?;
        buf.truncate(written);
        Ok(buf)
    }

    pub fn shape(&self) -> Result<Vec<usize>, MlxError> {
        let mut tmp = [0i32; 8];
        let mut ndim = 0usize;
        let rc =
            unsafe { ffi::rlx_mlx_array_shape(self.ptr, tmp.as_mut_ptr(), tmp.len(), &mut ndim) };
        check(rc)?;
        Ok(tmp[..ndim].iter().map(|&d| d as usize).collect())
    }

    pub fn num_elements(&self) -> Result<usize, MlxError> {
        Ok(self.shape()?.iter().product())
    }

    pub(crate) fn from_raw(ptr: *mut mlx_array_t) -> Self {
        Self { ptr }
    }

    /// Clone the array handle. Cheap — bumps the underlying
    /// shared_ptr refcount on the C++ side; the new Rust handle
    /// owns its own wrapper so independent Drop is safe.
    pub fn clone_handle(&self) -> Result<Self, MlxError> {
        let mut out: *mut mlx_array_t = std::ptr::null_mut();
        let rc = unsafe { ffi::rlx_mlx_array_clone(self.ptr, &mut out) };
        check(rc)?;
        Ok(Self { ptr: out })
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::rlx_mlx_array_free(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

// SAFETY: MLX arrays are reference-counted internally; moving the
// handle pointer to another thread is safe as long as no two threads
// touch the same handle simultaneously (which the &mut/owned Rust
// semantics enforce).
unsafe impl Send for Array {}

#[derive(Debug, Clone)]
pub struct MlxError(pub String);

impl std::fmt::Display for MlxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mlx error: {}", self.0)
    }
}

impl std::error::Error for MlxError {}

pub(crate) fn check(rc: std::ffi::c_int) -> Result<(), MlxError> {
    if rc == RLX_MLX_OK {
        return Ok(());
    }
    let msg = unsafe {
        let p = ffi::rlx_mlx_last_error();
        if p.is_null() {
            String::from("(no message)")
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    Err(MlxError(msg))
}

pub(crate) fn map_dtype(d: DType) -> MlxDtype {
    match d {
        DType::F32 => MlxDtype::F32,
        DType::F16 => MlxDtype::F16,
        DType::BF16 => MlxDtype::BF16,
        DType::I32 => MlxDtype::I32,
        DType::F64 => MlxDtype::F64,
        DType::I8 => MlxDtype::I8,
        DType::I16 => MlxDtype::I16,
        DType::I64 => MlxDtype::I64,
        DType::U8 => MlxDtype::U8,
        DType::U32 => MlxDtype::U32,
        DType::Bool => MlxDtype::Bool,
        DType::C64 => panic!("rlx-mlx: DType::C64 (complex) not supported"),
    }
}

/// MLX runtime version string.
pub fn version() -> String {
    unsafe {
        let p = ffi::rlx_mlx_version();
        if p.is_null() {
            return String::new();
        }
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

/// Default-device name (e.g. "Apple M2 Pro"). Empty string if MLX
/// can't reach the device-info service for any reason.
pub fn device_name() -> String {
    unsafe {
        let p = ffi::rlx_mlx_device_name();
        if p.is_null() {
            return String::new();
        }
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

/// Force evaluation of a batch of arrays.
pub fn eval(arrays: &[&Array]) -> Result<(), MlxError> {
    if arrays.is_empty() {
        return Ok(());
    }
    let handles: Vec<*mut mlx_array_t> = arrays.iter().map(|a| a.ptr).collect();
    let rc = unsafe { ffi::rlx_mlx_eval(handles.as_ptr(), handles.len()) };
    check(rc)
}

/// Schedule a batch of arrays for evaluation; do not wait for completion.
/// Pair with `synchronize()` (or `eval()`, which also drains) to make
/// the results visible.
pub fn async_eval(arrays: &[&Array]) -> Result<(), MlxError> {
    if arrays.is_empty() {
        return Ok(());
    }
    let handles: Vec<*mut mlx_array_t> = arrays.iter().map(|a| a.ptr).collect();
    let rc = unsafe { ffi::rlx_mlx_async_eval(handles.as_ptr(), handles.len()) };
    check(rc)
}

/// Wait for every in-flight async eval on every MLX stream.
pub fn synchronize() -> Result<(), MlxError> {
    let rc = unsafe { ffi::rlx_mlx_synchronize() };
    check(rc)
}
