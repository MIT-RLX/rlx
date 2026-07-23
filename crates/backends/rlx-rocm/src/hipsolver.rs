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

//! hipSOLVER shim — batched Jacobi eigensolver + dense LU solve.
//!
//! AMD's hipSOLVER mirrors cuSOLVER's dense LAPACK surface. Wired symbols:
//! - `hipsolverSsyevjBatched` (+ SyevjInfo) for native `Op::Eigh` / `EighBatch`
//! - `hipsolverSgetrf` / `hipsolverSgetrs` for native `Op::DenseSolve`
//!
//! Resolved via libloading at runtime so the crate compiles without ROCm.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_int, c_void};
use std::ptr;
use std::sync::{Arc, OnceLock};

use libloading::Library;

use crate::hip::HipStream;

pub type HipsolverHandle = *mut c_void;
pub type HipsolverSyevjInfo = *mut c_void;

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HipsolverError(pub c_int);

impl HipsolverError {
    pub fn ok(self) -> Result<(), HipsolverError> {
        if self.0 == 0 { Ok(()) } else { Err(self) }
    }
}

impl std::fmt::Display for HipsolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hipsolverStatus({})", self.0)
    }
}

impl std::error::Error for HipsolverError {}

/// `hipsolverEigMode_t` — values from hipSOLVER's types header.
#[repr(C)]
#[derive(Clone, Copy)]
pub enum HipsolverEigMode {
    NoVector = 201,
    Vector = 202,
}

/// `hipsolverFillMode_t` (= `hipblasFillMode_t`).
#[repr(C)]
#[derive(Clone, Copy)]
pub enum HipsolverFillMode {
    Upper = 121,
    Lower = 122,
}

/// `hipsolverOperation_t` (= `hipblasOperation_t` values).
#[repr(C)]
#[derive(Clone, Copy)]
pub enum HipsolverOperation {
    N = 111,
    T = 112,
    C = 113,
}

type FnCreate = unsafe extern "C" fn(*mut HipsolverHandle) -> HipsolverError;
type FnDestroy = unsafe extern "C" fn(HipsolverHandle) -> HipsolverError;
type FnSetStream = unsafe extern "C" fn(HipsolverHandle, HipStream) -> HipsolverError;
type FnCreateSyevjInfo = unsafe extern "C" fn(*mut HipsolverSyevjInfo) -> HipsolverError;
type FnDestroySyevjInfo = unsafe extern "C" fn(HipsolverSyevjInfo) -> HipsolverError;
type FnSsyevjBatchedBufferSize = unsafe extern "C" fn(
    HipsolverHandle,
    HipsolverEigMode,
    HipsolverFillMode,
    c_int,
    *mut f32,
    c_int,
    *mut f32,
    *mut c_int,
    HipsolverSyevjInfo,
    c_int,
) -> HipsolverError;
type FnSsyevjBatched = unsafe extern "C" fn(
    HipsolverHandle,
    HipsolverEigMode,
    HipsolverFillMode,
    c_int,
    *mut f32,
    c_int,
    *mut f32,
    *mut f32,
    c_int,
    *mut c_int,
    HipsolverSyevjInfo,
    c_int,
) -> HipsolverError;
type FnSgetrfBufferSize = unsafe extern "C" fn(
    HipsolverHandle,
    c_int,
    c_int,
    *mut f32,
    c_int,
    *mut c_int,
) -> HipsolverError;
type FnSgetrf = unsafe extern "C" fn(
    HipsolverHandle,
    c_int,
    c_int,
    *mut f32,
    c_int,
    *mut f32,
    c_int,
    *mut c_int,
    *mut c_int,
) -> HipsolverError;
type FnSgetrsBufferSize = unsafe extern "C" fn(
    HipsolverHandle,
    HipsolverOperation,
    c_int,
    c_int,
    *mut f32,
    c_int,
    *mut c_int,
    *mut f32,
    c_int,
    *mut c_int,
) -> HipsolverError;
type FnSgetrs = unsafe extern "C" fn(
    HipsolverHandle,
    HipsolverOperation,
    c_int,
    c_int,
    *mut f32,
    c_int,
    *mut c_int,
    *mut f32,
    c_int,
    *mut f32,
    c_int,
    *mut c_int,
) -> HipsolverError;

pub struct HipsolverRuntime {
    _lib: Library,
    pub create: FnCreate,
    pub destroy: FnDestroy,
    pub set_stream: FnSetStream,
    pub create_syevj_info: FnCreateSyevjInfo,
    pub destroy_syevj_info: FnDestroySyevjInfo,
    pub ssyevj_batched_buffer_size: FnSsyevjBatchedBufferSize,
    pub ssyevj_batched: FnSsyevjBatched,
    pub sgetrf_buffer_size: FnSgetrfBufferSize,
    pub sgetrf: FnSgetrf,
    pub sgetrs_buffer_size: FnSgetrsBufferSize,
    pub sgetrs: FnSgetrs,
}

unsafe impl Send for HipsolverRuntime {}
unsafe impl Sync for HipsolverRuntime {}

impl HipsolverRuntime {
    pub fn load() -> Option<Arc<Self>> {
        unsafe {
            let lib = Library::new("libhipsolver.so")
                .or_else(|_| Library::new("libhipsolver.so.0"))
                .or_else(|_| Library::new("libhipsolver.so.1"))
                .ok()?;
            macro_rules! sym {
                ($name:literal, $ty:ty) => {{
                    let s: libloading::Symbol<$ty> = lib.get($name).ok()?;
                    *s.into_raw()
                }};
            }
            let rt = HipsolverRuntime {
                create: sym!(b"hipsolverCreate", FnCreate),
                destroy: sym!(b"hipsolverDestroy", FnDestroy),
                set_stream: sym!(b"hipsolverSetStream", FnSetStream),
                create_syevj_info: sym!(b"hipsolverCreateSyevjInfo", FnCreateSyevjInfo),
                destroy_syevj_info: sym!(b"hipsolverDestroySyevjInfo", FnDestroySyevjInfo),
                ssyevj_batched_buffer_size: sym!(
                    b"hipsolverSsyevjBatched_bufferSize",
                    FnSsyevjBatchedBufferSize
                ),
                ssyevj_batched: sym!(b"hipsolverSsyevjBatched", FnSsyevjBatched),
                sgetrf_buffer_size: sym!(b"hipsolverSgetrf_bufferSize", FnSgetrfBufferSize),
                sgetrf: sym!(b"hipsolverSgetrf", FnSgetrf),
                sgetrs_buffer_size: sym!(b"hipsolverSgetrs_bufferSize", FnSgetrsBufferSize),
                sgetrs: sym!(b"hipsolverSgetrs", FnSgetrs),
                _lib: lib,
            };
            Some(Arc::new(rt))
        }
    }
}

static RUNTIME: OnceLock<Option<Arc<HipsolverRuntime>>> = OnceLock::new();

/// True when `libhipsolver` loads and exposes the batched Jacobi symbols.
pub fn is_available() -> bool {
    RUNTIME.get_or_init(HipsolverRuntime::load).is_some()
}

pub fn runtime() -> Option<Arc<HipsolverRuntime>> {
    RUNTIME.get_or_init(HipsolverRuntime::load).clone()
}

/// hipSOLVER handle + SyevjInfo bound to a stream.
pub struct HipsolverContext {
    pub runtime: Arc<HipsolverRuntime>,
    pub handle: HipsolverHandle,
    pub params: HipsolverSyevjInfo,
}

unsafe impl Send for HipsolverContext {}
unsafe impl Sync for HipsolverContext {}

impl HipsolverContext {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn new(runtime: &Arc<HipsolverRuntime>, stream: HipStream) -> Option<Self> {
        unsafe {
            let mut handle: HipsolverHandle = ptr::null_mut();
            (runtime.create)(&mut handle).ok().ok()?;
            (runtime.set_stream)(handle, stream).ok().ok()?;
            let mut params: HipsolverSyevjInfo = ptr::null_mut();
            (runtime.create_syevj_info)(&mut params).ok().ok()?;
            Some(Self {
                runtime: runtime.clone(),
                handle,
                params,
            })
        }
    }

    /// # Safety
    /// `stream` must belong to the same HIP context as this handle.
    pub unsafe fn set_stream(&self, stream: HipStream) -> Result<(), HipsolverError> {
        unsafe { (self.runtime.set_stream)(self.handle, stream).ok() }
    }
}

impl Drop for HipsolverContext {
    fn drop(&mut self) {
        unsafe {
            if !self.params.is_null() {
                let _ = (self.runtime.destroy_syevj_info)(self.params);
            }
            if !self.handle.is_null() {
                let _ = (self.runtime.destroy)(self.handle);
            }
        }
    }
}
